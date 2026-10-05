# The compiled tier

An accelerator that may always decline. Everything here is optional: turn it
off with `JAITHON_NO_JIT=1` and the interpreter runs every program exactly as
it would have. That is the first thing to know about it, and the reason the
boundary contract below is written the way it is.

`jit.h` is the whole surface the rest of the VM sees; `jit_internal.h` is the
tier's own shared state, and nothing outside this directory includes it.
`jit.c` holds the entry point, the SIGPROF sampler and two stencil compilers.
The rest is one concern per file: `jit_compile.c` drives a whole-function
compile, `jit_body.c` walks the opcodes with the per-family arms beside it in
`jit_body_*.c`, `jit_call*.c` emit calls, `jit_osr.c` is on-stack replacement,
`jit_global.c` resolves globals and module members, `jit_runtime.c` holds the
C thunks compiled code calls out to, `jit_loop.c` is a shape-matched compiler
for one exact loop, and `jit_arena.c` and `jit_arm64.{c,h}` are executable
memory and the instruction encoders.

**This document names functions, not files or lines.** A function name survives
a file being split or moved and a line number does not, and this directory gets
reshaped. Everything below can be found with `grep -rn`.

---

## The two tiers

### The whole-function tier

`jaiJitCompileFunc` / `jaiJitEnterFunc`, in `jit_func.c`. Compiles a whole
function body to arm64 with the interpreter's frame replaced by machine
registers: locals in `x19..x28` and `v8..v15`, the operand stack in the
callee-saved registers above them -- or in `x0..x8` outright for a body that
calls nothing, and in both when the bank splits. Nothing touches the VM stack
while the body runs.

A body reaches it by **entry count**. `callClosure` (vm.c) increments
`fn->entryCount` on every call until it reaches `jaiJitThreshold(fn)` --
`JAI_JIT_THRESHOLD` (64), or `JAI_JIT_TRACE_THRESHOLD` (8) for an `@trace`d
function -- and then calls `jaiJitEnter`, which compiles on that call and
enters the result immediately. The threshold is tied to the inline caches by
two `_Static_assert`s in `jit.h`: an IC's observation window must close before
the tier reads it, or the tier specialises on a half-formed record.

A compile that fails is retried at most five times (`fn->jitAttempts`), then
`fn->jitRefused` is set and the function is never looked at again. Five is
measured, not guessed: `jaiJitEnter`'s comment records that raising the cap to
40 compiles seven more bodies in one workload and costs 20% wall.

A compile that *partially* succeeds is a different thing and has its own,
smaller budget -- see "A partial body is recompiled once its callee compiles"
below.

`jit.c` also holds two toy stencil compilers, `compileReturnNull` and
`compileAccessor`, tried only after `jaiJitCompileFunc` declines. They exist as
the smallest proof that generated code can be entered and returned from.

### The OSR loop tier

`compileOsr` / `jaiJitEnterOsr`, also in `jit_func.c`. Compiles a *loop*, using
the interpreter's own frame slots as the compiled body's locals, and hands back
a bytecode offset for the interpreter to resume from.

A loop reaches it by **sampling**. `jaiJitStartSampling` arms `ITIMER_PROF` at
1 kHz (`JAITHON_JIT_TICK_US` overrides); the `SIGPROF` handler does nothing but
set `jaiInterrupted = 2`. The interpreter's existing `OP_LOOP` safepoint
(`safepoint` in vm.c) notices that and calls `jaiJitSample`. Riding the
safepoint rather than counting back edges is deliberate and priced: a back-edge
counter measured 4.7%-4.7x depending on placement, 11% even switched off, and
punishes tight loops for being tight.

A tick only lands on a back edge, which is the one moment interpreter state
matches what compiled code expects -- `OP_LOOP` sets `ip` to the loop's first
instruction and *then* runs the safepoint. The first such tick arms the body
(`JAI_JIT_HOT_TICKS` is 1); `jaiJitSample` then tries `jaiJitEnterLoop` (the
shape matcher) and falls through to `jaiJitEnterOsr`.

Once a loop has a compiled form, `fn->osrHot` is set and the back edge enters
it **directly**, without waiting for another tick. That matters more than it
sounds: at 4 kHz, `matrix_mul`'s innermost body ran 1.7 million times and the
compiled form was entered fewer than twenty thousand. A head whose entry guard
keeps failing stops being offered (`JAI_OSR_GIVE_UP`), per head, because a
guard failing on one loop says nothing about the others in the same body.

### The shape-matched loop compiler

`jit_loop.c` matches one exact opcode sequence -- two of them, a counted head
and a `for i in 0..n` head over the same body -- and emits a hand-written loop
with a multiply-shift reciprocal for the constant divisor. It is not a general
compiler and is tried first only because it is cheap.

---

## A partial body is recompiled once its callee compiles

`emitUnarmedDeopt` lets a walk stop at an opcode this tier cannot speak and
interpret the rest, rather than giving the whole body up. That partial compile
**succeeds**. Nothing above notices: `jitAttempts` counts declines and never
sees one, `jitRefused` is not set, and the body keeps its truncated form for
the life of the process.

Which is fine when the walk stopped at something permanent, and an accident
when it stopped at `OP_GET_GLOBAL` on a callee that had simply not compiled
*yet*. The caller crossed its own entry count first, the global read found no
`jitFunc`, the walk stopped -- and the callee compiled seconds later with
nothing to re-examine the caller. `_is_ident_start`, whose whole body is a
call to `_is_alpha`, compiled to a single deopt and stayed that way for
13.3 million interpreted instructions of `check --no-cache lib/std`.

So `compileFuncOnce` records **which** callee stopped it, in
`ObjFunction::jitBlockedOn`, recovered from `(fn, e.unarmedAt)` exactly as
`unarmedDetail` recovers the name it prints. `jaiJitEnterFunc` then checks that
one pointer on entry, and when the named callee has a `jitFunc` it did not have
before, calls `jitRecompileBlocked`.

Four things make that not the 20%-slower mistake in disguise.

* **The trigger is specific.** It fires only when *the callee this body's walk
  actually named* has compiled since -- not on "everything that failed". On
  `check --no-cache lib/std` it fires 20 times in a run, across 20 distinct
  bodies, and 13 of them go on to compile further. On `check lib/jaithon`, 17
  times across 16 bodies.
* **The bound is per (caller, callee) pair**, `JAI_JIT_RECOMPILES` of them.
  Per-*body* was measured and is too tight: `_scan_token` needs two, on
  `_is_ident_start` and then on `_is_digit`, and capping at one loses a third
  of the win on `lib/jaithon`. The retry that produced a form blocked on the
  same callee again clears the field rather than asking twice, and the count is
  the backstop against a longer cycle. Once the budget is spent the field is
  NULL, so the entry path's check goes back to costing nothing -- the same
  lesson `jitRefused` records two paragraphs up.
* **The old form is restored when the retry fails**, which it does 5 times in
  20. Without that those bodies fall from partly-compiled to fully interpreted,
  which is worse than what they had. It is cheap because a failed compile
  writes nothing to the `ObjFunction`: every `fn->jit*` store in
  `compileFuncOnce` is in its success tail.
* **Acceptance is not gated on "walked further".** That was built and measured
  and it is worse -- 308.35M interpreted instructions against 300.22M for
  accept-anything. A bytecode offset is not a quality proxy: `_scan_token`'s
  new stop offset is *lower* and its own interpreted work falls from 6,990,397
  to 2,527.

Worth 7.1% of the interpreted instructions of `check --no-cache lib/std` and
7.5% of `check --no-cache lib/jaithon`, A/B'd in one binary on
`JAITHON_JIT_RECOMPILE`. It is not only a compiler-shaped win: `sort_merge`,
whose `sort` stops at `OP_GET_GLOBAL` on `merge`, goes from 14.2M interpreted
instructions to 0.25M on one retry, and 0.33s to 0.27s.

**This is the one thing in the tier that writes `jitFunc` twice.** A baked
direct call is still sound -- the arena is never freed, so the words the old
entry points at do not move, and `emitDirectCall` reads the address and every
field beside it in the same instant and bakes all of them as immediates, so a
site keeps calling the form it was told about rather than pairing old code with
newer metadata. `jaiCallPreparedFn1` already re-validated on `fn->jitFunc !=
p->entry` and its comment already named a recompile as the case.

`jitBlockedOn` is marked in `blackenFunction`. It is a global of the function's
own module, which is marked there already, so while the binding stands this
pins nothing new; a *rebound* global is what it is there for, where the old
function becomes collectable and the field would dangle.

---

## The boundary contract

`jaiJitEnter`, `jaiJitEnterFunc` and `jaiJitEnterOsr` all answer with a verdict,
and **the difference between the verdicts is a correctness one, not a hint.**
Every caller has to say what it does with each.

| verdict | what it promises |
| --- | --- |
| `JAI_JIT_DECLINED` | Nothing was touched. `vm.stackTop` and the frame stack are exactly as they were, and the interpreter must run the call normally. |
| `JAI_JIT_DONE` | The call is **complete**, in exactly `OP_RETURN`'s poststate: return value at `slotBase[0]`, `vm.stackTop == slotBase + 1`, no frame ever pushed. |
| `JAI_JIT_ERROR` | Compiled code called out, the callee raised, and those effects **already happened**. The call must not be re-run. |
| `JAI_JIT_DEOPT` | A guard met a value the body was not compiled for, part-way in. Like `ERROR` this cannot be re-run; unlike it, the interpreter resumes from the exact offset in the deopt record. Push the frame, then call `jaiJitApplyDeopt`. |

The load-bearing distinction is `DECLINED` against everything else. A caller's
decline path pushes a frame and runs the function **from the top**, so anything
the compiled body did before it stopped is done twice. `jaiJitEnter` used to
answer a bool (`== JAI_JIT_DONE`), which made a body that DEOPTED on the very
call that compiled it read as "declined". That stayed invisible while deopting
on the first compiled call was rare; an unarmed opcode now emits an
unconditional deopt, so it is the *normal* outcome for any body that mentions
one, and `Bag.items` -- whose entire body is `self.made += 1` and a list of
tuples -- counted 6002 calls for 6001. One double effect per function, on the
call that compiles it.

A **bail** is different again and is why `jitResultOut` can answer `DECLINED`
after entering: overflow and a low stack are raised by handing the call back,
which is sound only because a body that can bail cannot have written anything
yet. A bail retires the form (`jitFunc = NULL`, `jitRefused = true`) so it is
not re-entered every call to bail again.

The OSR entry has its own three-way answer: `0` declined (nothing touched), `1`
resume at `*resumeAt` with any operand-stack values the loop held already
pushed, `2` an exception is pending. `jaiJitEnterLoop` answers a bool, where
true means it set `frame->ip` itself -- to the loop exit if the loop finished,
to the loop top if it bailed, and the caller cannot tell which.

`jaiJitFinishDeopt` is the fourth door: a compiled body that deoptimised inside
a *self-call* is finished in the interpreter from its own record, at the
innermost frame that sees it, and the value handed back to the compiled caller.

---

## The one divergence that is deliberate

Everything else here exists to make the tier unobservable. Recursion depth is
the exception, and it is worth knowing before a differential run reports it as
a miscompile.

The interpreter counts frames and raises `RuntimeError` past `JAI_FRAMES_MAX`
(1024). A compiled frame is a native frame, not a `vm.frames` entry, so that
count does not apply to it; the tier bails on the real thread stack instead,
derived from the thread's own bounds in `stackLimit`. So `deep(1100)` raises
interpreted and returns compiled, and compiled recursion raises only in the
tens of thousands.

It fails cleanly either way -- the deep case still raises `RuntimeError`, it
just raises later -- and the compiled limit is the more accurate one, since it
measures the stack that actually exists rather than a fixed count chosen for
the interpreter's frame size. Enforcing parity would mean counting frames on
every compiled call, on the hot path, to make a rarely-observed threshold
match. It is not worth that, so it is written down instead.

`tests/fuzz/found/recursion_depth_limit.jai` is the reproducer, kept as a
record of a divergence rather than as a bug.

## SlotKind, and the one invariant

`SlotKind` is what the tier believes about a value. Every local, every operand
stack entry, every parameter and every return has one, and every instruction
the tier emits is chosen from it.

| kind | what the register holds |
| --- | --- |
| `SLOT_INT` | a raw `int64_t` |
| `SLOT_FLOAT` | the bits of a `double` (in an X register, or parked in `v8..v15`) |
| `SLOT_BOOL` | 0 or 1 -- **one byte**, because `BOOL_VAL` writes only the union's `bool` member |
| `SLOT_INST` | `ObjInstance *`, of one pinned class |
| `SLOT_MAYBE_INST` | that same pointer **or a zero** |
| `SLOT_MAYBE_OBJ` | some heap object **or a zero**, with no class at all -- `SLOT_OBJ`'s permissions minus the promise of being non-null |
| `SLOT_LIST` | `ObjList *` |
| `SLOT_OBJ` | some heap object this tier does not model: load, pass, store, root, nothing else |
| `SLOT_ITER` | an `ObjIter` this body built; its index stays in memory |
| `SLOT_NULL` | what `-> void` returns: a defined zero whose tag is `VAL_NULL` |
| `SLOT_OPAQUE` | present in a register, but nothing may be done with it -- the body never reads this slot |
| `SLOT_SELF`, `SLOT_CLASS`, `SLOT_FUNC`, `SLOT_NATIVE` | compile-time constants; `holdsRegister` says these four occupy no register at all |
| `SLOT_DYNAMIC` | **a return kind only** -- the body's return sites disagree, and each one hands up its own Value tag beside the value. Never an operand-stack entry, never a local's kind, never in a deopt record or an OSR form; see below |

Four kinds have a compile-time tag that is a **fact**: `SLOT_INT` is always
`VAL_INT`, `SLOT_FLOAT` always `VAL_FLOAT`, `SLOT_BOOL` always `VAL_BOOL`,
`SLOT_NULL` always `VAL_NULL`. Being right about `SLOT_BOOL`'s tag is not the
same as being right about its payload -- `BOOL_VAL` writes one byte and leaves
the other seven indeterminate, which is a bug this file has shipped more than
once, most recently as a cross-module bool return read eight bytes wide.

Every object kind shares one tag, `VAL_OBJ`, which is why proving "it is an
object" is never enough:
`localIn`, `emitCallOutResult` and the invoke arms all follow a `VAL_OBJ` check
with an `Obj.type` check and, for an instance, a `shapeId` check.

And the two nullable kinds -- `SLOT_MAYBE_INST` and `SLOT_MAYBE_OBJ` -- have
**no compile-time tag at all**. The same register is a pointer or a zero and
only the payload says which. `emitTagFor` is the one place that gets it right:
a `subs`/`csel` off the payload rather than a `movz` of a constant.
`scripts/gate/kind_tag_check.py` exists to keep that true, and it names both
kinds: a second nullable kind the gate has not been taught is exactly the
hazard it was written for, wearing a name it does not know.

`SLOT_MAYBE_OBJ` is `SLOT_MAYBE_INST` with the class dropped, and that is the
whole of the difference: wherever a site consults a shape or a class, it must
refuse the object form; wherever it only loads, passes, stores or roots, the
two are interchangeable. It is what `-> OpKind?` returns -- an enum member on
one edge and null on the other, a pair no other kind covers. Today it is
produced only by `mergeReturnKind`, `pushValue3` refuses it, and
`jitMaybeObjStackOn` (`JAITHON_JIT_MAYBE_OBJ_STACK=1`) lets it onto the operand
stack, which is worth ~352 refusals on `check lib/std` and measured +6 compiled
bodies. It is **off by default** pending a measurement on a quiet machine.

An audit of all 234 sites in `src/vm/jit/` that branch on a stack entry's kind
found 14 that would miscompile with it on. Every one is fixed, and the shape of
the finding is worth keeping: **nine of the fourteen were the same defect**. The
two deopt stubs had diverged -- the function tier's ladder
(`jit_compile.c`) already read both nullable kinds, the OSR tier's read only
`SLOT_MAYBE_INST` -- so every producer of a deopt record (`emitUnarmedDeopt`,
`branchOnDeopt`, `branchOnDeoptInstStart`, `deoptRecordAt`, three arms of
`emitDirectCall`, `emitMaybeInstResult`, `emitCompare`) was reported separately
while being one missing arm. Ranking the reports by count would have pointed at
eight innocent files. The genuinely distinct ones were:

* `adoptLocalKindSeen` -- a stack entry's kind becomes a LOCAL's kind, and a
  local's tag comes from `localTagFor`, a ladder of constants. A nullable kind
  cannot have a constant tag, so this one refuses rather than widens.
* the OSR entry guard's `default: break` -- its premise, "opaque: never read",
  is false for this kind, since the prologue loads every register-homed slot
  eight bytes wide. A form compiled for object-or-null and re-entered holding
  an int would call `7` non-null. The neighbouring `SLOT_OBJ` arm records
  having been fixed for exactly this once before.
* the OSR local write-back, whose csel escape named only the instance kind.

The tiers are complements, and the lesson is the one the bug list at the end of
this file already teaches twice: **a fix applied to one tier's copy of a ladder
is not applied.** Grep the twin.

### A body whose returns disagree

`jaithon.ast.schema._default_field` returns `-1`, `[]`, `Span.none()`,
`false`, `0`, `0.0`, three enum members and `null`, and for a long time it
was the callee that kept `jaithon.ast.node.init` -- the hottest body in the
compiler -- interpreted, refused at every attempt with "OP_RETURN: a body
returning both int and list". No `SlotKind` covers int-or-list, and no widening
rule can invent one.

It does not need one. The body returns exactly one thing per call and **every
return site knows which**, so `mergeReturnKind` joins two kinds it cannot
otherwise represent to `SLOT_DYNAMIC`, and each return site then leaves its own
tag: `emitReturnLeave` runs `emitTagFor` on the payload in `x0` -- the `csel`
off the payload for a nullable kind, a constant for the rest -- and shifts it
into bits 8..15 of `x1`, above the verdict byte that has always been there
(`JIT_RET_TAG_SHIFT`). `jitResultOut` masks the verdict, and for a
`SLOT_DYNAMIC` body rebuilds the `Value` from that tag rather than from the
kind. A float goes through as raw bits, so a NaN payload is not canonicalised
on the way; a bool is read one byte wide, as everywhere.

What keeps this small is that the kind is a **return kind only**:

* It never reaches the operand stack -- `pushValue3` refuses it -- so it need
  not fit the four bits `stackSignatureAt` packs a kind into, and the
  `_Static_assert` beside the enum says exactly that: `SLOT_MAYBE_OBJ` is the
  last kind that may be packed, and `SLOT_DYNAMIC` sits past the limit on the
  strength of never being pushed.
* It never becomes a local's kind (`adoptLocalKindSeen` refuses it), never
  appears in a deopt record, and the OSR tier never sees it because a return
  inside an OSR loop is refused before any merge.
* **Every compiled caller refuses a callee that returns it.** No accept-list
  was widened: a direct call, a self-call, an indirect call, a PIC way, a
  module or static call all read `jitReturnKind` and decline `SLOT_DYNAMIC`
  with its name. The tag bits therefore only ever reach C. A body that both
  returns dynamic and calls itself is refused, since the self-call site reads
  `x1` as a bare verdict.
* The decision is made in the **measuring pass**: the real pass learns
  `Emit::dynamicReturn` from it before its first return site is emitted,
  because `x1`'s meaning has to be one thing for the whole body. A
  disagreement the real pass meets that the measuring pass did not is
  declined rather than patched after the fact.

`OP_RETURN_NULL` used to refuse bare whenever an earlier return was not null;
it is one more site for the merge now, so `-> any` bodies that end in a bare
`return` join too. Off with `JAITHON_JIT_DYNAMIC_RETURN=0`, which restores the
refusal exactly.

Clearing that link exposed the next two in the same body. `return Span.none()`
is `OP_GET_FIELD` + `OP_TAIL_CALL`, not the `OP_INVOKE` that `emitClassCall`
speaks, so the static method read as a value now becomes a `SLOT_FUNC` entry
behind `emitClassCall`'s three guards (`JAITHON_JIT_STATIC_METHOD`). And the
callee it names has often **not returned yet** when the caller's attempts are
spent -- a matter of time, not of kind. It is refused under that name, and a
decline of that shape is not charged to the five-attempt budget: `jaiJitEnter`
resets the entry count instead, up to `JAI_JIT_COLD_RETRIES` times
(`JAITHON_JIT_COLD_RETRY`). The paragraph in `jaiJitEnter` that priced raising
the cap to 40 at 20% wall still stands for every other decline; this one is
sixteen measuring passes that each stop at the same call.

### What the interpreter records about a return

A caller compiling before its callee has none of the callee's own answers, so
it reads what the interpreter watched the callee return: `ObjFunction::
obsReturnKind`, one byte, merged by `jaiFeedbackMerge` in `bytecode/chunk.h`.

That byte used to collapse **every** disagreement to `JAI_FB_MIXED`, and mixed
is unusable, so 433 refusals on `check lib/std` across 121 bodies came down to
"the callee returned two things". Measured, 42 of the 50 first disagreements
were *null meeting an instance* and 4 were *null meeting a string*: only 4 were
two unrelated kinds. A nullable instance is a kind this tier has spoken since
`SLOT_MAYBE_INST` existed -- the byte was throwing away the one fact that
would have compiled the caller.

So there is a third band, `JAI_FB_NULLABLE + ObjType`: "null, and also exactly
one object type". Two properties keep it safe:

* Every decoder **rejects an unknown byte** rather than indexing on it, so a
  reader that predates the band treats it as mixed -- which is what it meant.
  A `_Static_assert` keeps the band clear of both the object band and `MIXED`.
* A null return must no longer clear `obsReturnShape`. The old rule zeroed the
  shape the moment a nullable-instance function returned its null, so the very
  case the band exists to record arrived with no class and was refused anyway.
  The first *non-null* return sets the shape; later ones confirm or zero it,
  and that zero stays sticky.

Believing the band lets a walk past a refusal it used to stop at, and what it
reaches there can decline the **whole** body where a prefix used to compile --
the same bargain the enum-`match` arms struck. `compileFunc` therefore retries
with `gNoNullableFb` set, exactly as it does for `gMatchUsed`.


## Every switch this tier reads

Fifty-odd environment variables reach `src/vm/jit/`, and until this table
existed the only way to find one was to grep. They are here because **a change
with no switch is a change with no number you can trust**: an A/B in one binary,
interleaved, is the only measurement this tier accepts, and that needs the two
forms to coexist in one build.

Unless a row says otherwise, the value `0` turns the switch off and anything
else (including unset) leaves it on. `scripts/gate/switch_doc_check.py` keeps
this table complete in both directions.

### Which tier runs, and when

| switch | default | what it does |
| --- | --- | --- |
| `JAITHON_NO_JIT` | on | Set to anything but `0` to run everything interpreted. The baseline every correctness question is settled against. |
| `JAITHON_JIT_THRESHOLD` | 64 | Calls before the **function tier** looks at a body. `1` compiles on the first call -- the way to make the function tier deterministic. |
| `JAITHON_JIT_TICK_US` | 1000 | `SIGPROF` period for the **OSR tier**, clamped to 50..100000. `50` is the aggressive setting; it changes a *race*, so it can miss arms a slower tick reaches. |
| `JAITHON_JIT_TICK_ARM` | on | Whether the first tick arms a body, rather than waiting for a second. |
| `JAITHON_JIT_MAIN_EDGES` | 4096 | Interpreted back edges, counted from when the program's module body and then its `main()` start, after which the next back edge is a tick -- one that also asks for the enclosing loop at once (`0` off). The first hot loop no longer waits for the timer to land on it, and the loops around it do not wait for a second tick. |
| `JAITHON_JIT_OSR_FORMS` | `JAI_OSR_MAX` | Cap on compiled forms per OSR loop. |
| `JAITHON_JIT_OSR_SLOTS` | `JAI_OSR_SLOTS` | Cap on slots an OSR entry will reconstruct. |
| `JAITHON_JIT_DEPTH_MEMO` | on | Keep each function's verifier depth table (`chunkDepthTable`) from its first compile attempt until the function is freed (`jaiJitForgetFunction`), instead of re-running the verifier on every attempt, retry and OSR variant. 5070 verifier runs became 709 on `check --no-cache lib/jaithon`, 32ms to 10ms of CPU. |
| `JAITHON_JIT_OSR_BACKOFF` | on | A loop head that has failed to compile is retried only after 1, 2, 4, 8, 16 and 32 failures; the ticks in between are charged as failures without the walk, so the head retires on the same tick as before. A head that has waited for a cold callee and has waits left is not backed off, since a wait is never charged. Each attempt is up to four variants of three retries, about 31ms of `check --no-cache lib/jaithon`. Not free: a head that would have compiled on a skipped tick waits for the next power of two, and is lost if its loop is gone by then -- a few heads per run on the compiler workloads, inside their OSR noise. Off retries on every tick. |
| `JAITHON_JIT_LEAN_ENTRY` | on | Short-frame paths for a call from interpreted code: `callClosure` enters a compiled body, or binds and pushes an interpreted one, in a frame that saves three register pairs, and `jaiJitEnterFunc` takes bodies of up to four arguments, and bodies blocked on a callee that is still cold, saving two; everything else (traced, due a recompile, wider, any decline) goes to the old functions, which saved all twelve callee-saved registers on every call. `invokeMethodOnStack`, `jaiCallValue` and a class call reach `callClosure` directly for a closure. 50 instructions fewer per interpreted call into a compiled body, 32 into an interpreted one (1.10-1.12x on a loop of tiny calls, 1.01-1.02x on call-heavy programs with real bodies); `check --no-cache lib/jaithon` 3.6-3.8% fewer instructions, 1.02-1.04x cycles in two independent same-binary runs (floors 0.4% and 1.05%). Off restores the old paths. |
| `JAITHON_JIT_RECOMPILE` | on | Recompile a body once, when the cold callee its walk stopped at finally compiles. Worth 7.3% on `lib/std`; see the recompile section. |

### Stress, for finding bugs the default configuration hides

| switch | default | what it does |
| --- | --- | --- |
| `JAITHON_JIT_DEOPT_STRESS` | off | Take every deopt exit that *can* be taken. Set to anything but `0`. |
| `JAITHON_JIT_SPLIT_STRESS` | off | Split compiled bodies at every opportunity. |
| `JAITHON_MEGA_STRESS` | off | (`vm_cache.c`) Force inline caches megamorphic. |

### Arms, each an A/B of one feature

| switch | default | what it does |
| --- | --- | --- |
| `JAITHON_JIT_MATCH` | on | The four enum-`match` opcodes. Off restores the pre-arm prefix compile. |
| `JAITHON_JIT_KEEP_RETRY_NEEDS` | on | Keep the clashing locals the full attempt reported across the fallback attempts (inlining, `match` arms, nullable band off). Off lets a fallback that stopped earlier overwrite them, so the widening step saw nothing to grow and declined the body -- which is what a `match` whose tag slot was coalesced onto its subject's did every time. |
| `JAITHON_JIT_WRAP` | on | The wrapping operators `+% -% *%`. |
| `JAITHON_JIT_JOIN` | 1 | How exactly a join compares two operand stacks. `0` restores the 2-bit kind hash that **miscompiled** (see the golden `jit_join_kind_collision`); `1` compares the whole kind; `2` also compares `valueDepth`. |
| `JAITHON_JIT_NULLABLE_FB` | on | Believe the nullable return-feedback band. Off reads it as mixed, which is what it meant before the band existed. |
| `JAITHON_JIT_MAYBE_OBJ` | on | Let `mergeReturnKind` widen object-meets-null to `SLOT_MAYBE_OBJ`. |
| `JAITHON_JIT_DYNAMIC_RETURN` | on | Let `mergeReturnKind` join two kinds it cannot otherwise represent to `SLOT_DYNAMIC`, each return site carrying its own tag. Off restores the "a body returning both X and Y" refusal. |
| `JAITHON_JIT_STATIC_METHOD` | on | `Klass.static_fn` read as a value at `OP_GET_FIELD` -- what `return Klass.f()` compiles to -- as a `SLOT_FUNC` entry behind the static-call guards. |
| `JAITHON_JIT_COLD_RETRY` | on | A decline on a callee that has not returned yet does not spend one of the five attempts, up to `JAI_JIT_COLD_RETRIES` times. Off charges it like any other decline. |
| `JAITHON_JIT_MAYBE_OBJ_STACK` | **off** | Let `SLOT_MAYBE_OBJ` reach the operand stack. Off until every site that reads an entry's kind is shown to exclude or handle it; `1` turns it on. |
| `JAITHON_JIT_PAIR_LIST` | on | Let a pair loop over a **list of 2-tuples** be an OSR loop head (iterKind 4), not only a dict-items view. The step is not new -- it is `emitForIterPair`'s non-dict tail, which the function tier already reaches; only `jaiJitEnterOsr`'s head gate refused it. Off restores that gate. |
| `JAITHON_JIT_PIC` | on | The polymorphic inline-cache arm at an invoke. |
| `JAITHON_JIT_LIT_POOL` | on | Load a 64-bit constant a guard compares against -- a closure's function, a callee, an enum type, an object an `is` names -- from a pool after the body (`emitConstCmp`): one `ldr` for three or four `movz`/`movk`. Guards only, because a compare feeds nothing but a predicted branch; pooling every three-word constant cost `object_dispatch` 7%, where the hot ones are an address the allocator call waits on. 8% on `closure_calls` under `JAITHON_JIT_ALIGN=64`, 3% at the default alignment. |
| `JAITHON_JIT_RANGE_FACTS` | on | Drop the overflow check on `i += k`, `n + k` and `n - k` of an int local when a comparison every path to it made already rules the overflow out (`jit_range.c`): `while i < n { ...; i += 1 }`, and both subtractions after `if n < 2 { return n }`. Read off the bytecode's own control-flow graph, never the emitter's model. 2.8% on `closure_calls` (its loop is decode-bound, so one instruction is ~4%); nothing on `fib_recursive`, where a not-taken `b.vs` was free. |
| `JAITHON_JIT_INLINE_METHODS` | on | Inline a straight-line method body at a call whose receiver class is known -- a pinned invoke, or one way of a polymorphic cache -- through the same walker a global function is inlined with (`inlineMethodCall`), field reads off the receiver included. Off restores the narrow `inlineMethodWalk` alone. 2.15x on a getter-per-iteration probe; `poly_dispatch` loses 31% of its instructions and almost none of its cycles (mispredicts and `sdiv` latency). |
| `JAITHON_JIT_OSR_COLD_WAIT` | on | An OSR compile that meets a call to a callee still on its way to the function tier's threshold (called, not refused, no compiled form yet) declines and looks again on a later tick, at most 24 times a head and never charged to its compile attempts (`jitOsrColdWait`). Otherwise a tick landing in a loop's first iterations built a form that called the callee the slow way for the rest of the run: `object_dispatch` 918M cycles against ~440M under `JAITHON_JIT_TICK_US=50`, and the slow outcome in about one default run in ten. |
| `JAITHON_JIT_POLY_LOOP` | on | A `for x in xs` the walk meets away from an OSR head, over a live list whose first elements are instances of more than one class, binds `x` unpinned (the first binding of that local only) so calls on it dispatch through `emitInvokePic1`, instead of pinning element 0's class and deoptimising on every other -- once per call, for a function that walks the list. 1.87x on a 12-shape `area()` loop called 400k times. Makes the polymorphic cache reachable from the whole-function tier, which before this it was not (see "The gate is one NULL"). |
| `JAITHON_JIT_PIC_INLINE_ONLY` | on | Keep a polymorphic way whose callee cannot be called directly -- not compiled yet, or compiled to a specialisation this site cannot pass (`jitPic1Admissible`) -- when its body can be inlined instead, which needs neither; the interpreter's record of its return kind must match the site's. A way whose body the inliner then cannot speak is left empty, both edges of its compare falling to the next way. A loop compiled by an early tick now inlines its ways rather than going round the descriptor for them. |
| `JAITHON_JIT_POLY_PARAM` | on | A function whose compiled form keeps declining calls at entry because an instance parameter holds another class (32 declines) is compiled once more with that parameter unpinned -- `jitArgIn` then asks only for an instance, shape 0 -- so its calls dispatch through `emitInvokePic1`; the old form is restored if the new one will not compile (a field read off the parameter). Never a method's receiver. 3.94x on a probe passing three shape classes through `fn weighted(s: Shape, k: float)`, where two calls in three had been running interpreted. |
| `JAITHON_JIT_PIC_UPGRADE` | on | Re-compile a loop form whose polymorphic site came up short of ways because the timer tick beat its callees' compiles (`jitOsrPicShort`), and let that form's miss path keep teaching the site's cache (`jitInvokeByNameLearn`). On `poly_dispatch` the form was 0-, 1- or 8-way by the race, 400-490M cycles against 120M; under `JAITHON_JIT_TICK_US=50` it was always short, and this is worth 3.48x there. |
| `JAITHON_JIT_CLASS_CALLS` | on | Direct calls to a class constructor. |
| `JAITHON_JIT_MODULE_CALLS` | on | Calls through a module member. |
| `JAITHON_JIT_MODULE_NATIVE` | on | Native calls through a module member. |
| `JAITHON_JIT_MODULE_FIELD` | on | Reading a module member as a field. |
| `JAITHON_JIT_STATIC_FIELD` | on | Reading a class's `statics` table. |
| `JAITHON_JIT_SOFT_FIELD` | on | Refuse a field arm softly rather than declining the body. |
| `JAITHON_JIT_FIELD_DECL_KIND` | on | Trust a field's declared type as its kind. |
| `JAITHON_JIT_ELEM_DECL` | on | Trust a list's declared element type. |
| `JAITHON_JIT_LIST_PROBE` | on | Read an element kind off a live list. |
| `JAITHON_JIT_LIST_RESULT` | on | List-returning native results. |
| `JAITHON_JIT_LIST_SCALAR` | on | Scalar-returning list natives. |
| `JAITHON_JIT_RET_LIST` | on | Let a per-callee record earn `SLOT_LIST` rather than `SLOT_OBJ`. |
| `JAITHON_JIT_RET_OBJTYPE` | on | Record the object type a callee returns. |
| `JAITHON_JIT_RETURN_KNOWN` | on | Require a direct callee's walk to have reached a return. |
| `JAITHON_JIT_ANY_GUARD` | on | Guard an `any`-typed value rather than refusing it. |
| `JAITHON_JIT_STR_GUARD` | on | Guard a declared `str` or `dict` boundary on an object rather than refusing it. |
| `JAITHON_JIT_FDIV_GUARD` | on | Raise on float `/` by zero, as the interpreter does. |
| `JAITHON_JIT_FP_REREAD` | on | A float operator reads a float local's d home instead of the X copy `OP_GET_LOCAL` took, while no local has been written and no join crossed since (`fpOperandReread`). Takes the accumulator of `sum += f(i) * v[j]` off a `fmov d,x`/`fmov x,d` round trip on its loop-carried chain: spectral 1.72x. |
| `JAITHON_JIT_DYN_MEMO` | on | Guard a dynamic local (one slot two kinds share, as `for i in 0..n` then `for b in bodies` do) once per straight line instead of at every read (`Emit::dynGuarded`). Retired at a join, a call out and a write of the slot. nbody: 952M to 805M instructions. |
| `JAITHON_JIT_FIELD_READ_MEMO` | on | Remember a field read whose tag guard passed against the local it was read through (`recordFieldRead`), so a second read skips the guard and a store of the same kind skips its tag write. Retired exactly as a store's memo is, and only kept for a receiver its local still holds (`stackLocalCurrent`). |
| `JAITHON_JIT_WIDE_CARRY` | on | Let one walk carry FP-resident and deferred values into 1024 offsets (and write 256 homes early) rather than 64 (and 32); past the limit every instruction settles, so a large float body ran its tail through X. The post-walk branch check marks fixup targets once instead of rescanning them per offset. |
| `JAITHON_JIT_HOIST_LEAN` | on | A hoist keeps one register for `items`; a list stored into inside its region gets one for its version, loaded and bumped once where the header is, so each store writes it with a plain `str` instead of a load-add-store chained through memory to the previous iteration; counts get what is left. Off: two registers a hoist, as before. heat_2d 1.32x -- the version chain was its critical path. |
| `JAITHON_JIT_FTWO` | on | `2.0 * x` (either order, the literal from `OP_CONST`) compiles as `fadd x, x`: the same double for every x, NaN, the infinities, -0.0 and overflow included, and a cycle shorter on a chain. mandelbrot's `y = 2.0 * xy + y0` is the shape; 1.04x there. |
| `JAITHON_JIT_OSR_PROMOTE` | on | A tick on a hot loop asks for the loop around it (`jaiJitWantEnclosing`), which compiles at its next back edge and asks for the one around it in turn. Without it the enclosing loop waits for a tick to land exactly on its own back edge, which a nest whose inner loop is already compiled hardly ever offers: mandelbrot's per-pixel loop ran 0.8M-11M instructions interpreted a run that way. mandelbrot 1.12x. |
| `JAITHON_JIT_FN_BIND_EARLY` | on | Let a float operator that feeds `OP_BIND` write the local's d home directly in the function tier's planned frame too (`earlyBindTier`), not only in OSR: `dx = bi.x - bj.x` stops being an `fsub` into the bank and an `fmov` out of it. |
| `JAITHON_JIT_BORROW_CREDIT` | on | A read of a local satisfied by borrowing its register counts towards that register in the register plan (`noteLocalBorrowed`). The function tier's measuring pass sees every local in a fixed register, so every read there was a borrow and none was counted: a list subscripted every iteration ranked at zero and was planned into the frame. |
| `JAITHON_JIT_FN_HOIST` | on | Hoist loop-invariant list headers in the function tier too: the measuring pass's ranges are carried into `planHoists` as compileOsr carries its probe's, with x13..x17 as the pool. Storage is not pinned there -- the sample is the first call's arguments -- so accesses keep their dispatch, but the header load, the bounds check a head guard covers and a stored list's version chain go. 1.15x on a dot/axpy probe called 2000 times. |
| `JAITHON_JIT_ITER_RECYCLE` | on | A compiled for-in over a list that it built the iterator for hands the iterator back when the loop is exhausted (`gJitIterSpare`, rooted by `jaiJitMarkFrames`, its source cleared), and the next `jitMakeIter` reuses it instead of allocating. A loop over a short list in a hot function allocated one per execution: graph_bfs 1.27x, nbody 1.15x. A loop left early, by `break`, a raise or a deopt, keeps its iterator. |
| `JAITHON_JIT_HOIST_FRESH_COUNT` | on | A subscript of a hoisted list that has no count register (lean ran out of registers) and that the head guard does not cover keeps the hoisted `items` and loads only `count`, with `ldr w`, instead of reloading the whole header: the count feeds only the bounds branch, so the element address stops waiting on a header load. Measured on the numeric-kernels branch, where pushing regions kept no count: life 1.03x, matrix_mul 1.04x. |
| `JAITHON_JIT_PUSH_REG` | on | A loop whose only append is a push to one local's list, that calls nothing (a keeping grow stub, `JAITHON_JIT_GROW_KEEPS`, is not a call) and never rebinds the local, keeps that list's count and bumped version in two hoist registers (`Emit::pushHoist`): each push compares and stores them but never loads them, so it no longer waits on the previous push through memory. Two appends in one loop keep the memory path -- two locals may name one list. Off with `JAITHON_JIT_GROW_KEEPS=0`, which makes every push a call. On the numeric-kernels branch: a 100k-push build loop 1.91x, sieve 1.37x, list_ops 1.11x. |
| `JAITHON_JIT_ITER_IDX_REG` | on | A for-in over a list in a loop that calls nothing keeps the iterator's index in a hoist register (`Emit::iterHoist`), loaded once at the head and stored back after every step, so the next step no longer reloads what the last one stored. Up to two such loops per body; the step's version and bounds checks are unchanged. A keeping grow stub puts the register back, as it does every register the body names. A summing loop over a 1000-element list 1.10x. |
| `JAITHON_JIT_CLOSURE_HOIST` | on | A closure read out of a local and inlined at an indirect call, inside a loop that calls nothing and never writes that local, has its function guard proved once at the loop head, and the int, float or bool upvalues its body reads loaded there into hoist registers (`Emit::closHoist`, `planClosureHoists`). A closure's `fn` never changes and compiled code has no `OP_SET_UPVALUE`, so with no call nothing can write the cell during the loop; a frame may own an open cell and write it without a call, through a local its chunk captures by reference, so a loop that writes such a local hoists only the guard (`loopWritesCapturedLocal`); a loop tier body calling a closure over a local it does not write gets both (1.33x on a 30M-call probe). A head miss resumes the interpreter at the head. closure_calls 1.90x against the switch off, 2.03x against main. Only from a call site that runs on every pass of the loop (`runsEveryPass`): a head miss sends the whole loop to the interpreter, so a site in a rarely taken arm, called with a different closure each time, ran 10x slower than main before this (tests/vm/hoist_rare_site.sh). |
| `JAITHON_JIT_INLINE_BORROW` | on | An inlined body's plain read of a parameter or bound local borrows the caller entry's register instead of copying it, and an inlined call whose result goes straight into a local (`OP_SET_LOCAL`/`OP_BIND` next, not a branch target) leaves the result a borrow of the inlined bank for that one store. Removes two `mov`s from an inlined call's dependency chain: on the M2 a CHAIN of eliminated moves is not free -- closure_calls' hoisted 12-instruction loop with four chained movs ran SLOWER than the 22-instruction loop it replaced, and 9 instructions with two ran 1.9x faster. The fused-constant and field-read forms still copy, because they write or hand on the entry's own register. |
| `JAITHON_JIT_SHRINK_WRAP` | where it pays | A function-tier body whose first statement is `if <int param> <cmp> <literal 0..4095 or another int param> { return <int param or literal> }` or `if <nullable instance param> ==/!= null { return ... }`, in a function returning a plain int, compiles that test on the incoming argument register AHEAD of its frame (`emitEarlyReturnArm`): the taken arm is `cmp; b.cond; mov; mov x1,#0; ret`, with no frame, no callee-saved stores, no stack-limit check. Sound because the arm holds no guard, no call and nothing that can raise, so no stub that assumes a frame is reachable from it. It pays only when most calls take the arm, since every call that builds the frame now takes a branch at its first instruction, so by default it is emitted only where the bytecode says they do (`earlyArmPays`): the rest of the body is straight-line -- no branch, so no loop, no second base case, no conditional call -- with at least two references to the function's own name, so every call that misses the arm makes two or more calls and the leaves (the calls that take it) outnumber everything else; and at most one operation sits between the last call and the return. Admitted, cycles with the arm vs `0`: fib_recursive 1.23-1.25x, hanoi `h(n-1) + 1 + h(n-1)` 1.17-1.18x, `1 + tri(n-1) + tri(n-2)` 1.20-1.22x, `tri(n-1) + 1 + tri(n-2)` 1.16-1.18x, divide and conquer `dc(lo, mid) + dc(mid + 1, hi)` 1.13x, binary_trees 1.05-1.07x (`build`'s null leaves and `walk`'s null children). Refused, each of which measured slower or flat with the arm forced on (`1`): grid paths and binomial 0.93-0.94x (a second base case the arm does not cover sends half the leaves through the branch AND the frame), collatz 0.95x (one call per level, so the arm runs once per chain), gcd and a linear `spread(lo + 1, hi) + 1` flat, and the binary recursions with two operations after the last call -- `tri(n-1) + tri(n-2) + 1` 0.95-0.99x, `+ n` 0.97x, `h(n-1) + h(n-1) + 1` 1.00x -- which sit at exactly half leaves and where the extra checked add lands on the path every leaf's parent runs (the tail-length condition; it also refuses `dc(lo, mid) + dc(mid + 1, hi) + 1`, which the forced arm won 1.17x). A 0-7 nop pad after the arm moves no result, and `JAITHON_JIT_ALIGN=64`/`128` leave the tri pair at 0.95x and 1.19x, so none of this is layout. Laying the arm out after the body (frame path falls through) made it layout-bound instead: fib 1.005-1.10x and grid paths 0.89-1.15x across the same pads. `0` turns it off, `1` emits it wherever the idiom matches; `tests/vm/shrink_wrap.sh` checks both and the rule. |
| `JAITHON_JIT_SHRINK_SKIP` | on | With a shrink-wrapped early arm, the body's walk starts at the arm's jump target, since every entry has already passed the test; `0` keeps the body's own copy of the test, which costs the fall-through a compare and a TAKEN branch (fib 1.10x instead of 1.25x). Not taken when anything else branches into the arm. |
| `JAITHON_JIT_SHRINK_WIDE` | on | Widens the shrink-wrapped early arm beyond int returns: `return true`/`return false` in a function returning a plain bool, a bare `return` in a function with nothing to return (x0 = 0, as emitReturnNull leaves it), and `return null` in a function returning an instance or null (x0 = 0; that body keeps its own copy of the test, so the walk still merges the null return's kind and shape). A recursive `has(node, v) -> bool` tree search 1.11x; binary_trees 1.065x (`build`'s null leaves). |
| `JAITHON_JIT_GLOBAL_GUARD_HOIST` | on | A loop that runs no Jaithon code and is entered only at its head proves the module-globals `keyVersion` guard once at the head instead of before every global access (`planGuardHoists`). "Runs no Jaithon code" is `regionRebinds`: every clobber site in the loop must be an instance allocation of a simple-init class (`Emit::clobberAlloc`), which reaches only the allocator and the collector, and the collector never touches a module's globals table. Each access still loads the value and checks its tag. alloc_churn (a loop at module scope) 1.23x; the ratchet in noteScratchClobber declines a body whose real pass finds a Jaithon-code call inside such a loop. The loop is chosen for an access that runs on every pass (`runsEveryPass`), so the head checks only what that access would have checked on the first pass. |
| `JAITHON_JIT_GLOBAL_TAG_PROOF` | on | Inside a loop whose globals guard is proved at the head, a module global whose every recorded access in the loop (read or compiled store) is of one scalar kind has its tag checked once at the head, after the key check (`planTagProofs`, `Emit::tagProof`). Reads then skip the tag check; stores skip the old-value-is-an-object test (and the version bump behind it) and the tag store, writing only the payload. Sound for the reason the guard hoist is: nothing in the loop runs Jaithon code, so only the loop's own stores can change the tag, and they all store that kind. alloc_churn 1.13x on top of the guard hoist. Only for a global with a read that runs on every pass and has no store of the loop ahead of it: the head then misses exactly when that read would have on the first pass. A global read only in a rare arm, whose kind changes between calls, kept the per-read check (tests/vm/hoist_rare_site.sh). |
| `JAITHON_JIT_GLOBAL_PROMOTE` | on | Over a tag-proved loop that calls nothing at all (`regionCalls`), each proved global's value also lives in a hoist register: loaded at the head, read as a borrow, and written THROUGH -- every store still goes to the entry and then updates the register. A module-scope loop carries its counters through their entries, a store and a load on the critical path each iteration; this takes the load off it, and because memory is always current no exit, deopt or raise has anything to write back. A loop whose only calls are simple-init allocations takes x13..x17 alone (the inline allocation never names them) and reloads them from their entries after each allocation's slow path (`emitPromotedReload`). A 30M-iteration module-scope accumulate loop 1.38x; alloc_churn 1.135x on top of the tag proof. The ratchet in noteScratchClobber declines a body whose real pass finds any other call in such a loop. |
| `JAITHON_JIT_INLINE_CONSTRUCT` | on | The inliner admits a body that ENDS `return C(...)` for a class whose init only stores its arguments into fields (`jitSimpleInitClass`): the class read and the closing `OP_TAIL_CALL` become the allocation emitCallOut already makes. Its stores go only into the object just allocated, so a guard that deoptimises to the call re-runs it and orphans nothing anyone saw; nothing but the `OP_RETURN` follows the allocation. The allocator's slow path is a real call, so such a body's entries continue the caller's callee-saved bank and count in its save set (`Emit::inlShared`) instead of taking x0..x8; a caller whose own values are in x0..x8 does not inline it. object_dispatch 1.28x (`acc = acc.add(step)`). |
| `JAITHON_JIT_INLINE_LOOPS` | on | A direct call to a small global function or pinned method that branches or loops (at most 160 bytes of bytecode, four per caller) is inlined where the call is (`inlineLoopCall`), where the straight-line inliner refuses it: `while` and `for i in a..b` loops, returns from inside them, `and`/`or`, field reads off the receiver, `float()`/`int()`. The callee's code is walked with every slot operand renumbered into *homes*, slots of the caller's frame above the interpreter's window, so the ordinary local arms, the range facts and the register planner treat them as the caller's own locals; its branches resolve against its own offset map while the caller's fixups are set aside, and every `return` puts its result in one register and branches to the inline's exit. The callee and its arguments stay on the operand stack, so a guard anywhere inside resumes at the caller's call instruction and the interpreter re-runs the whole call: the whitelist therefore admits nothing that stores to the heap or calls (`noteScratchClobber` declines anything that slips through), and the homes are skipped by the deopt stubs and every root fill. An inline whose walk fails part-way (an arm that calls out, such as a dict lookup the whitelist could not tell from a list read) refuses that callee by name and the body is compiled again with every other inline kept (`gLoopInlineRefused`); retrying with all inlining off made such a caller 3.4x slower than the switch off. Function tier only -- an OSR form's locals are the interpreter's own frame, which has no room above its window. queens 1.19x by min, 1.17x by median (`safe` inside `place`, with the storage pin below); every other language row compiles the same code. The self-hosted compiler takes only branch-only inlines of small lexer predicates (`_is_digit`, `_base_of`, `_at`, ...), with no measurable effect on `check --no-cache lib/jaithon`. `tests/fuzz/inline_loops.py` is its differential fuzzer: the main one almost never writes the shape. |
| `JAITHON_JIT_SPILL_BORROW` | on | A function-tier body on the per-slot register plan (`spilled`) borrows a local's register on a read, as the OSR tier does with the same plan, instead of copying it into the operand bank. A loop-bearing inline is what pushes a caller onto that plan. Cuts queens' instructions 18% (899M to 731M) for ~3% of its cycles (1.03x by min): the copies were `mov`s, which rename eliminates. |
| `JAITHON_JIT_INLINE_STG_PIN` | on | A list parameter of a loop-bearing inline that the body never rebinds has its storage checked once at the inline's entry (`inlineLoopCall`, `Emit::homeStgPin`), and every element read of it inside the inlined loop uses that storage with no dispatch of its own (`listAccessFor`). Sound because the inlined body stores nothing and calls nothing; a miss re-runs the call in the interpreter like any other guard in it. queens 1.11x by min on top of the inline: the per-element `ldrb`/`cmp`/`b.ne` was one of eleven branches in `safe`'s loop. |
| `JAITHON_JIT_INLINE_RETIRE` | on | A loop-bearing inline whose guards keep failing is retired: each one the function tier emits is remembered by caller and call offset (`loopSiteRemember`), `jaiJitApplyDeopt` counts the deopts that resume at that call, and at 64 the caller's form is dropped (`jitFunc = NULL`) so its next call compiles it again with that callee behind a real call (`jitLoopInlineSeedRefusals`). A failing guard inside an inline re-runs the whole call and the rest of the caller in the interpreter; behind a real call it costs only the callee's own remainder. A helper handed lists of two storages in turn at one site deoptimised on every other call through the storage pin: 0.44x against no loop inlining, 1.0x with this. Not counted under `JAITHON_JIT_DEOPT_STRESS`, which fails every guard on purpose. A compiled caller that branches straight to the retired form keeps it. |
| `JAITHON_JIT_SINK` | on | Allocation sinking in the OSR tier (`jit_sink.c`). A loop local bound only to `C(...)` of a simple-init class whose fields are ints or floats, and read only by field -- directly, or as the receiver or an argument of a method the tier inlines -- is never allocated: the construction stores its arguments into frame slots, a field read is one load, and the local's own slot is never read or written. `planSinks` proves from the bytecode that the first mention in the loop is the bind and that no branch skips it, so nothing ever reads the local from the frame; every way out (exit, deopt, bail, raised overflow) runs `OSR_SYNC_ITER`, which writes the fields into `gDeopt.sinks`, and `jaiJitEnterOsr` allocates the object into the local and into every stack entry that named it (`SLOT_VREF`) once the frame and stack are rooted. A local the loop reads before it rebinds it (`acc = acc.add(step)`) is entered holding a real object: the prologue checks it is exactly that class with those field kinds (else the form bails), unpacks it into the homes, and a zero "bound" word keeps the frame's object as the local's value -- same identity -- until the first bind; such a loop must not call out (`noteScratchClobber` declines), since a call could write the object the homes copied. An inlined `return C(...)` whose caller binds the result straight into a sunk local is sunk too (`sinkBindAfter`, `sinkConstructInline`). A site the plan could not foresee (a call that is not inlined, a different class or kind) declines the compile, and `compileOsrAny` retries it with sinking off. alloc_churn 1.77x by min (686M to 320M instructions); object_dispatch 1.32x with the fields in frame slots (1131M to 528M), about 2.1x with `JAITHON_JIT_SINK_FP_HOMES`. What it does NOT reach, each measured at about 1.0x against the tree before it: a function-tier body (only OSR loops plan a sink, and nothing is sunk while inlining, so a temporary built inside a small helper the loop calls stays allocated); a module-scope `var` (a global, not a local); a class of more than `JIT_SINK_FIELDS` (4) fields or with any field that is not an int or a float; an init that does more than store its arguments in parameter order (`simpleInitFields`); a method with a `let` before its `return C(...)`; a loop with any opcode outside `planSinks`'s list (`+%`, a field store, a closure, a try). The fuzzer covers it through `progen.py`'s `gen_sink_family`; `tests/fuzz/pic_rate.py --sink` counts the forms that kept a sunk local, and `JAI_JIT_WHY=1` names why a loop sank nothing (`plans no sink`, `does not sink local`). The objects are built only for a form that sank one, which adds `JIT_OSR_SUNK_RET` to everything it returns, so a loop that sinks nothing pays nothing per exit (a clear and a scan of the record on every exit had cost a loop leaving its region every iteration ~36 instructions an exit, +6.4%). |
| `JAITHON_JIT_SINK_FP_HOMES` | on | A sunk float field (`JAITHON_JIT_SINK`) lives in a callee-saved d register beside the float locals (`sinkPlanFpHomes`, counted into `fpLocals` so every prologue saves and every epilogue restores it) instead of a frame slot, and a sunk construction takes its float arguments straight out of the FP bank (`sinkFpFast`), so the walker no longer syncs the bank to X before it. Reads are copies (`fmov`), never borrows, so `Vec2(p.y, p.x)` reads both fields before either home is written. On a loop-carried object the chain per field per iteration is then an `fadd` and a move instead of an `fmov` to X, a store and a forwarded reload: object_dispatch 1.44x on top of the sink (140M to 98M cycles; the memory homes also gained from the skipped sync, 153M to 140M). |
| `JAITHON_JIT_BREAK_ITER` | on | `break` out of a for-in (`OP_POP` of the iterator, then `OP_JUMP`) no longer stops the walk. In the function tier the jump is emitted and the iterator is put back on the model for the code after it, which only the body's own branches reach; in an OSR region compiled for that loop the break is a jump to the iterator's own exit, whose stub has the interpreter drop it. Before, every function and region holding such a loop declined outright: a loop with a `break` 2.58x, `check` of the compiler 1.02x. |
| `JAITHON_JIT_FN_FP_HOMES` | on | A function-tier body whose locals all fit in x19.. still plans its registers when that gives a float local a d home (`fpHomeWanted`). Without it every float local of an ordinary function lives in an X register and each float op on it pays two cross-register-file moves. |
| `JAITHON_JIT_CONCAT_LOCALS` | on | String concatenation into locals. |
| `JAITHON_JIT_STRCMP` | on | String ordering comparisons. |
| `JAITHON_JIT_STRCMP_EQ` | on | String equality. |
| `JAITHON_JIT_STR_ITER` | on | Iterating a string. |
| `JAITHON_JIT_DISPATCH_COLD` | on | An element read that dispatches on its list's storage (int, float or bool elements, storage not pinned) falls through on the predicted unboxed storage and keeps the boxed arm -- storage check, tag check -- out of line after the body; a dispatched append, and a nested `for x in xs` step over an unpinned list, likewise keep their boxed arms out of line. Inline, one arm or the other always took a branch, and with untyped int lists unboxed past eight elements (JAITHON_LIST_SHAPE_GROWN) the taken one was the common one. Off keeps both arms inline. |
| `JAITHON_JIT_BOUNDS_COLD` | on | A list access the loop head did not prove in bounds checks with one unsigned compare that falls through when the index is in range, and keeps the negative-index arm (add the count, check again, deoptimise or come back) out of line after the body. Inline, that arm sat behind a `b.lo` TAKEN by every in-range index -- a taken branch per unproved access. Off restores the inline form. |
| `JAITHON_JIT_FLOOR_COLD` | on | The floor correction of `x % k` / `x // k` for a literal `k > 0` goes out of line (`emitColdFixup`): a compare that falls through for a non-negative remainder, and a branch to a one-instruction stub after the body for a negative one. The inline `tbz` it replaces is TAKEN in the common case, a second taken branch in every loop taking a remainder (loop_sum); a branch-free `and`/`add` fixed that but put the correction on the dependency chain (poly_dispatch -9%). Off restores the `tbz`. |
| `JAITHON_JIT_ITER_STG` | on | Dispatch on list storage at a nested `for-in`. |
| `JAITHON_LIST_SHAPE_GROWN` | on | (`object_collection.c` and `jitListGrow`) An untyped boxed list whose push grows it past its first eight elements, when all eight and the new one are the same int, float or bool, takes that kind's unboxed storage at that growth -- `var xs = []` filled with ints ends up I64, as `list[int]` would be -- instead of staying boxed at sixteen bytes an element. Not on the first push: a record built as `[id, name, score]` would be shaped and boxed again every time (0.39x on a hot record builder). elemKind stays ANY; a later element of another kind de-specialises it once. Both tiers shape on the same test; a compiled push shapes only where it dispatches on storage afterwards, and no OSR form pins a boxed list of eight or fewer, at compile time or at entry (`jitListShapeable`). Off leaves untyped lists boxed. |
| `JAITHON_JIT_SHAPE_SIBLINGS` | on | (`emitGrowStubs`) A dispatching append that could unbox the list it grows (JAITHON_LIST_SHAPE_GROWN) does not when the same body also appends an object to the same local -- ten ints then `r.push("end")` would otherwise reach the last append unboxed, fail its boxed-storage guard and deoptimise every call (0.54x on a hot builder). Declining to shape is never wrong: the append dispatches on whatever storage the list has. Off lets every dispatching append shape. |
| `JAITHON_JIT_INDEXED_LOAD` | on | An element read of static unboxed storage is one register-offset load (`ldr x, [items, i, lsl #3]`, `ldr d`, `ldrb`) off the index -- the loop head's proved one, or the normalised one where nothing proved it -- instead of an `add` and a load (and, for a proved index, a copy into the normalisation scratch first); a store into such storage is likewise one `str`/`strb`. A stencil reads eight of them a cell. Off restores the add-then-load form. |
| `JAITHON_JIT_FUSED_SHAPE` | on | The fused `OP_ADD_INT_CONST` / `OP_SUB_INT_CONST` (`local + k`, `local - k`) carry an index shape (`Emit::idxKnown`) as the unfused GET_LOCAL-and-literal form does, so a subscript the emitter fused -- `mid[c - 1]` in a stencil -- can be proved in bounds at its loop's head and counts toward its slot's span. Off leaves those subscripts checked one by one. |
| `JAITHON_JIT_HOIST_PARTIAL` | on | Hoist a list header out of a loop that holds SOME of its slot's subscripts, not only one that holds all of them: a site outside the loop never asks for the hoist and reloads the header itself. `let d = dist[node]` above an inner loop over `dist` used to cost the inner loop its hoist. Bounds are still proved at the head only for a hoist that holds every subscript, since the span is the slot's. Off restores the all-inside rule. |
| `JAITHON_JIT_SPLIT_HOISTS` | on | A body whose operand stack is SPLIT (some entries callee-saved, the rest in x0..) offers the unused top of x0..x8 to the hoist planner, as a `scratchValues` body already does; a split body never inlines and a hoist only lives in a call-free loop, so they are free there. `valueBankRoom` honours the `scratchRoom` the hoists lower. Off keeps a split body's hoists to x13..x17 -- two headers. |
| `JAITHON_JIT_HOIST_PIN` | on | A hoisted list header also proves the list's storage at the hoist (one byte load, compare, deopt to the loop head), so the accesses inside that loop are emitted at that storage with no per-access dispatch -- for a slot the region writes (a stencil's rows, bound by the outer loop) and so never pinned at entry. Only over a loop with no call and no storage stamp, and only where the compiled frame held a live list to predict from. Off leaves those accesses dispatching. |
| `JAITHON_JIT_GROW_KEEPS` | on | The list-grow stub behind an inlined `push`/append saves and restores every caller-saved register a body can hold a value in (x0..x8, x13..x17, d0..d7, d16..d27), so an append stops being a clobber: a loop that appends keeps its hoisted headers, its pinned storage and an operand stack in x0..x8. What an append still does -- move one list's header -- is recorded per site (`notePushTarget`), and a header is hoisted over an append only when its slot is not the target and a run-time pointer compare at the hoist proves the lists differ. Off makes the grow an ordinary call out again. |
| `JAITHON_JIT_GROW_KEEPS_LOOPS` | on | (`growKeepsHere`) Only an append inside a loop of its body (by the caller's offset, for an inlined one; every append in an OSR form) gets the keeping grow stub. An append in no loop -- the fields of a record built by push -- has nothing hoisted across it and grows each list it builds once, so saving every register cost the whole difference on each call (0.92x on a hot `[i, "k", i * 2]` builder); there the grow is an ordinary call out again. Off keeps the registers at every append. |
| `JAITHON_JIT_GROW_KEEP_USED` | on | (`growKeepMasks`) A keeping grow stub saves only the registers the body could have put a value in: x0..x8 always, x13..x17 when some instruction emitted for the body names them as a destination, and a D register when any field of any SIMD-and-FP instruction names it. Each grow of a list that is fresh per call or per iteration -- a record built by push -- paid all thirty-four saves and restores, 4-10% of such a loop. Off saves all of them in every body. |
| `JAITHON_MAP_RUN` | on | (`jit_entry.c`, driven by `list.map`) Map a flat compiled callee (an int or float parameter; an int, float or bool result) over a whole run of the list in one call (`jaiMapPreparedFn1Run`): the stack window, the staleness test and the two storage switches hoisted out of the per-element loop. `list.filter` takes the same run (`jaiFilterPreparedFn1`) for a flat predicate taking an int or a float. Off sends every element back through `jaiCallPreparedFn1`. 1.25x on `list_ops`. |
| `JAITHON_JIT_ITER_SPARE` | on | (`jit_runtime.c`, `jitIterAlloc`) The inline iterator allocation behind a compiled OP_GET_ITER takes the iterator a finished nested loop handed back (`gJitIterSpare`, see `jitIterRecycle`) before allocating: only the descriptor path (`jitMakeIter`) used to, and the inline path had replaced it, so the spare sat unused and a nested `for nb in g[node]` allocated one iterator per outer iteration. graph_bfs 1.14x by min (334M -> 272M instructions, peak 50MB -> 22MB), nbody 1.04x. Off always allocates. |
| `JAITHON_JIT_LIST_ELEM_REUSE` | on | (`jit_body_index.c`, the boxed SLOT_LIST element read) An element of a list of lists is loaded once: the pointer its object-type check loads (`ldr x12, [x11, #8]`, the payload offset folded into the load) is the result, where the read used to `add` the offset, check, and load the same pointer again. `b[k]` in matrix_mul's inner loop: -4% instructions, 1.077x median / 1.024x by min at load 28. Off loads it twice as before. |
| `JAITHON_JIT_RANGE_COUNTER` | on | (`jit_range.c`, `counterBounds`, read through `jitSlotAddSafe`) The edge into a range loop's body bounds the variable OP_FOR_RANGE_BIND just bound: at most INT64_MAX - 1 when every OP_ITER_RANGE starting that counter is exclusive (an inclusive one can reach INT64_MAX), and at least the smallest start when every one of them starts at a straight-line literal (the counter only counts up). The unfused `j + k` / `j - k` (OP_ADD/OP_SUB with a folded literal, left entry exactly local `j`) now asks `jitSlotAddSafe` too, as the fused forms did. Life's six `c +/- 1` per cell lose their `b.vs`: 1.074x on `life` on top of if-conversion. Off: a range loop's variable gets no bound from its loop. |
| `JAITHON_JIT_VECTOR` | on | (`jit_vector.c`, `emitVectorHead`, from the walk just above the offset map, so it runs on loop ENTRY only and a back edge never re-runs it) A `for j in a..b` whose body is ONE store `L[j + k] = <expr>`, the expression reading only unboxed-float lists at `j + const` or `base + j + const` for a read-only int local `base` (|const| <= 12; each (list, base) pair is a stream with its own pointer and bounds), float literals and float locals with `+ - *`, gets a NEON run in front of its scalar loop: up to four lane pairs (`fadd/fsub/fmul.2d`, `ldur/stur q`) per iteration, in bytecode order, no fma, so every lane rounds as the scalar instruction does. Checked at run time, cheapest first, falling through to the untouched scalar loop on any miss: at least two elements, the stored list is no list read at another offset or base (by slot statically, by pointer at run time -- a recurrence stays scalar), every list is LIST_STORE_F64, every index the whole range reaches is in [0, count), with `cur + base` and `end + base` checked for wrap (so a negative index still wraps and an IndexError still raises at its element, scalar), and, for a based subscript, `cur`, `end` and every base in [-2^62, 2^62): the fold made `x + 12 + row` one offset, but the scalar loop adds 12 to x first and raises OverflowError there near the top of int, and inside that range no step of any subscript can overflow. It runs whole groups of 2U, then lane pairs (so a short row of 2..7 is vectorised too), advances the counter, rebinds the loop variable and bumps the stored list's `version` once, as a hoisted scalar store does; at most one element is left to the scalar loop. The measuring pass never emits a vector block, so a real pass that fails with one in it is compiled again without (`gNoVector`, both tiers). `heat_2d` 1.93x by median cycles, 1.81x by min (94M to 49M median, 538M to 170M instructions, switch in one binary, A-vs-A floor 3.8%), bit-identical; no other language row has the shape. A loop that fails only the alias check pays about 20 instructions an entry for finding out (a 22-trip aliased row: 0.98x). `JAI_JIT_WHY` prints each loop it vectorised and, for a body it walked into, the opcode it stopped at. Fuzzed by `tests/fuzz/vector_differential.py`. Off: every loop is scalar. |
| `JAITHON_JIT_BORROW_GUARDS` | on | (`jit_frame.c`, `noteDeoptBorrows` / `deoptFpSource`; both tiers' deopt stubs) A float stack entry still borrowing a local's d home may reach a guard: the record remembers which home each borrowed entry is in and the stub writes it out of that home, so OP_GET_INDEX no longer releases the borrow at its top. Sound because nothing between a borrow and a guard writes the home -- writing a local releases its borrows first (`fpReleaseHome`). The release was an `fmov d16, d8` on the loop-carried chain of `sum += a[k] * b[k][j]` (FP moves are not eliminated at rename): `fadd d8, d8, d17` now. 1.14x on `matrix_mul`. Off releases at the top of every subscript as before. |
| `JAITHON_JIT_IFCONV` | on | (`jit_body_cmp.c`, `jitTryIfConvert`, from OP_JUMP_IF_CMP_LOCAL_K and from an OP_GET_LOCAL2 feeding OP_JUMP_IF_CMP_FALSE; real pass only) A chain of `if local <op> k { s = v }` or `if a <op> b { s = v }` links (int locals) (an `elif` chain, optionally ending in `else`) that all assign one int local and meet at one join is emitted as one `cmp` + `csel` per link, innermost first, every value computed whether or not its test holds -- for values that cannot fault: a literal, an int local, or an element of an int list whose index the loop head's hoist already proved in bounds. That element's storage test is the one guard, and it resumes at the chain's first test with nothing written. A chain anything else branches into stays branches, and so does a running min or max -- a link `if s <op> x { s = x }` whose `s` is carried round the innermost loop (nothing between its head and the chain writes it): that branch is taken less and less often and predicts, and the csel would put a compare and a select on the loop-carried chain (0.91x over a million random ints before this test; 0.999x after, floor 0.32%). A clamp of a value loaded on the same trip converts. Life's rule mispredicted on most cells: 1.50x on `life`; a clamp and a ReLU over random ints 4.6x, over sorted ints (every branch predicted) 0.93x inside a 7.5% floor. Off compiles every chain as branches. |
| `JAITHON_MAP_RUN_TIGHT` | on | (`jit_entry.c`, `mapRunLoop`) The map run instantiated once per (parameter kind, result kind) so neither kind is decided per element, the frame count read once per run instead of before every call, the result stored at its own width without a storage switch, and the result's version bumped once per run (nothing can iterate a list `map` has not returned). Off keeps the one generic loop. 1.10x on `list_ops` by min, load 20. |
| `JAITHON_MAP_UNBOXED` | on | (`object_collection.c`, `jaiListShapeFor`, from the map run) A map whose callee returns an int, a float or a bool turns its presized boxed result into that kind's unboxed storage before the run fills it, as JAITHON_LIST_SHAPE_GROWN would on growth: a 10M-element map wrote 160MB of boxed Values and faulted every page in. elemKind stays ANY. Off (or JAITHON_LIST_SHAPE_GROWN=0) leaves the result boxed. Peak 246MB -> 166MB on `list_ops`. |
| `JAITHON_JIT_MAP_KERNEL` | on | (`jit_compile.c`, `jaiJitCompileMapKernel`; used by the map run) A one-argument lambda whose body calls nothing (callsOut off for both passes), writes nothing, hoists nothing and returns an int or a float is compiled once more with the map's loop inside it: the prologue and stack check run once, the loop head loads element `i` into x0 and falls into the argument moves, and every return stores to element `i` of the result and branches back. Index, bound and `&gJitMapRun` live in the three callee-saved registers above the body's plan. Every other exit -- deopt, bail, exception, overflow -- leaves with `gJitMapRun.i` naming its element, and the run finishes that element as it finishes any bailed call. Cached per (function, kind) in a table that grows and never evicts, each entry noting the compiled form and module version it was built for; a stale entry compiles again at most 4 times, a declined body is not asked again until its form changes, and kernels share at most an eighth of the code arena and stop once it is three quarters full (a direct-mapped cache had two hot lambdas evict each other on every run until the arena filled: 0.07x on a later hot loop). Taken only for an I64/F64 source, an unboxed result of the callee's kind, and at least 16 elements. `list.filter` takes the same kernel for a predicate returning a bool: index, bound, `&gJitMapRun` and a kept-count `j` in four registers, each true verdict copying element `i` of the source to element `j` of a result first given the source's storage (JAITHON_MAP_UNBOXED) and room for every remaining element, the room it did not use given back when it kept under half (a filter keeping 12 of a million ints held 8 MB per result: peak 35.8MB to 29.0MB, 1.02x). Off calls the body per element. 1.32x on `list_ops` (690M instructions against 1110M); a 5M-element `filter(|x| x > 500)` probe 1.48x (599M instructions against 1690M). |
| `JAITHON_JIT_STR_HEAD` | on | `for c in <string>` as an OSR loop head. |
| `JAITHON_JIT_CONST_ASCII` | on | Mark a one-character ASCII string literal as the ASCII table's own singleton at `OP_CONST`, so `==` and the one-byte ordering arm stop guarding the literal at run time. |
| `JAITHON_JIT_STR_FACTS` | on | Prove once at an OSR loop head, with no register, that a loop-invariant string local is a string, all one-byte (for `s[i]`) and/or interned (for `==`), so the per-iteration guards go even in a loop that calls. Off restores them. |
| `JAITHON_JIT_FIELD_DICT` | on | Predict a declared `dict[K, V]` field read off a receiver with no live sample (an instance the body built itself): tag and `OBJ_DICT` guarded, sampled by an empty exemplar dict so `d[k] = v` and the dict methods can compile. Off restores the decline. |
| `JAITHON_JIT_ITER_SOFT` | on | Function tier: a list or dict loop whose sampled container is empty ("iterating a list/dict with nothing to look at") is left to the interpreter from that instruction instead of declining the whole body. Off restores the decline. |
| `JAITHON_JIT_ITER_EMPTY_SKIP` | on | In front of a soft iterate refusal on a list, branch straight to the loop exit when the list is empty at run time, so only a non-empty list deopts. |
| `JAITHON_JIT_PAIR_INST` | on | A dict-items pair loop binds an instance component as `SLOT_INST` of its sampled class (object type and shape guarded per step) rather than a bare `SLOT_OBJ`, so a loop variable the frame already holds as that class no longer clashes. |
| `JAITHON_JIT_BRANCH_MAP` | on | Compile time only: `offsetIsBranchTarget` decodes each chunk once per compile into a bitmap instead of rescanning the whole chunk on every query. Off restores the scan. |
| `JAITHON_JIT_OSR_SELF_GLOBAL` | on | Inside an OSR loop form, a call to the function's own name goes through the ordinary compiled-global call to its whole-body form (when it has one) instead of the self-call arm, which branches to instruction 0 and so cannot serve a loop form. Off restores the silent `OP_CALL` refusal. |
| `JAITHON_JIT_LEAN_RESET` | on | Compile time only: resetting an Emit between walks leaves the instruction buffer and the fixup list (224KB of its 280KB) unzeroed, since both are read only below their counts. Off zeroes the whole struct. |
| `JAITHON_JIT_EARLY_UNARMED` | on | Decline a whole-body compile whose walk stops (an unarmed opcode) on the straight-line path from the entry, within 24 instructions -- every call would deopt there, which costs more than interpreting the prefix. A stop at a call to a not-yet-compiled function declines as cold, so it is retried. Off installs such bodies. |
| `JAITHON_JIT_ONE_BYTE_LOCAL_K` | on | `OP_JUMP_IF_CMP_LOCAL_K` against a one-byte string literal (`if c == "{"`, `if c < "0"`): the local is guarded a string and compared by its length and its one byte against an immediate, instead of the jaiStringOrder leaf call (which every `==` took when the local's sample was not interned -- a character read with `s[i]` samples the whole subject). |
| `JAITHON_JIT_ITER_ALLOC` | on | `OP_GET_ITER` over a list or a string tries a leaf allocator (`jitIterAlloc`, which declines when a collection is wanted) before the `jitMakeIter` descriptor. Off always takes the descriptor. |
| `JAITHON_JIT_STR_HOIST` | on | Hoist a loop-invariant string local's header (`chars`, `length`) and its string and all-one-byte proofs to the loop head, so `s[i]` inside a call-free OSR loop is a bounds check and a byte load. Off restores the per-character guards. |
| `JAITHON_JIT_COMP_ACC` | on | A list comprehension's append, through the frame. |
| `JAITHON_JIT_BUILTIN_CLASS` | on | Resolve a builtin class (every exception type). |
| `JAITHON_JIT_INVOKE_SOFT` | on | Master switch for emitInvoke's dead-path softening. |
| `JAITHON_JIT_INVOKE_SOFT_RECV` | on | Soften the last-resort receiver-kind refusal. |
| `JAITHON_JIT_INVOKE_SOFT_COLD` | **off** | Soften the cold-callee refusal. |
| `JAITHON_JIT_GLOBAL_ENUM` | on | Pin a module-level enum at compile time. |
| `JAITHON_JIT_OSR_GLOBAL_SOFT` | on | An OSR loop interprets from an uncompiled global rather than declining. |
| `JAITHON_JIT_OBJ_EQ` | on | Object identity comparison. |
| `JAITHON_JIT_NULL_PAIR` | on | The null-compare pair fusion. |
| `JAITHON_JIT_FUSED_DISCARD` | on | Fuse a call whose result is discarded. |
| `JAITHON_JIT_MEMBERSHIP` | on | `in` against a container. |
| `JAITHON_JIT_DICT_LEAF` | on | String- and int-keyed `d.get(k)`, `d[k]`, `d[k] = v` and `k in d` through a leaf call in front of the descriptor call. |
| `JAITHON_JIT_DICT_ADD` | on | `d[k] = d.get(k, n) + c` and `d[k] += c` on a dict, the counting idioms, as one leaf call. |
| `JAITHON_JIT_DICT_ADD_INLINE` | `jitDictAddInline` | `emitDictAddInline`, in front of the `jitDictAddStr` call of both counting arms (`emitDictAddFused`, `emitDictAugAddFused`) for a string or an int key: the update of a key found in the first slot its hash names -- live, holding that very string (or an int of that value, hashed inline as `jaiHashU64Inline` does) and an int value -- done inline, the step added with an overflow check, stored back and the table's version bumped, with no call, no argument moves and none of the leaf's re-checks. Anything else -- a key that is not a string, a typed dict, an empty table, a collision, an absent key, a value that is not an int, an overflow, a step outside an add/sub immediate -- branches to the leaf call, unchanged. Writes only x9..x12. 1.15x in cycles on `dict_ops` on top of the f-string memo, 1.13x on an int-keyed counting loop. |
| `JAITHON_JIT_DICT_SLOT_INLINE` | `jitDictSlotInline` | the same first-slot test as `JAITHON_JIT_DICT_ADD_INLINE` (`emitDictSlotProbe`) in front of the `jitDictGetStr`/`jitDictGetInt` and `jitDictSetStr`/`jitDictSetInt` calls (`emitDictLeafGet`, `emitDictLeafSet`): a present key -- live, in the first slot its hash names, holding that very string or an int of that value (an equal float stored there is the leaf's) -- has its value copied whole into the result, or (an untyped dict only) the new tag and payload stored and the version bumped, with no call. An absent key, a collision, a typed store and everything else is the leaf call, unchanged. Writes only x9..x12. -17% instructions and 1.06-1.10x in cycles on a string-keyed get-then-set loop. |
| `JAITHON_JIT_DICT_PROBE` | on | Compile a dict known only by its predicted type (`var d = {}` in the body) against a probe. |
| `JAITHON_JIT_DICT_KEYS_ITER` | on | `for k in d` in the function tier: the keys stepped inline. |
| `JAITHON_JIT_DUP` | on | `OP_DUP` and `OP_DUP2`, which every augmented subscript assignment (`xs[i] += v`, `d[k] += v`) is made of. |
| `JAITHON_JIT_FMT_LEAF` | on | f-strings through a leaf call in front of the descriptor call. |
| `JAITHON_JIT_FMT_INT_LEAF` | on | An f-string of one int hole between optional string runs through its own leaf. |
| `JAITHON_JIT_FMT_MEMO` | `jitFmtMemoOn` | the f-string memo (value.c): in front of the `jaiValueFormatIntLeaf` call, an inline probe of a direct-mapped table OF THE SITE'S OWN (`JaiFmtSite`, allocated when the site is emitted) keyed by (pre, n, post) -- the runs by identity, indexed by n plus a `crc32cx` of the runs that are there -- whose hit is the interned string the leaf answered last time, with no call. A table shared by every site and indexed by `n ^ pre >> 4` put two sites over the same ints (`f"user:{n}"`, `f"order:{n}"`) on the same slots, every probe of either missed and it ran at 0.86x; one table per site runs that loop 8x. On a miss the call goes through the site's leaf pointer to `jaiValueFormatIntLeafMemo`, which marks an intern hit with bit 0 of the pointer; a `tbnz` sends that answer to an out-of-line stub that strips the bit and files it with `jaiFmtMemoFill` (pre, n and post are still in their callee-saved registers), so the leaf itself keeps the register plan of the plain one. A site starts OFF (`entries` NULL: one load and a `cbz` past the probe), the first intern hit it files turns it on, and 256 probe misses in a row with no intern hit turn it off again inline, so a stream of distinct strings (`string_build`) pays four instructions, not the probe. Only intern hits are kept, so a hit is the object the leaf would have returned; the tables are weak, emptied by every collection (the only time an object is freed or its address reused) and when the intern table reaches its soft cap (past which the leaf builds a fresh string on every call); a table starts at 64 entries and doubles after twice its size in fills, up to the intern soft cap per site and 2 MB over all of them, and a site nothing was filed into between two collections gives its table back. The `str(n)` leaf (`JAITHON_JIT_STR_INT_LEAF`) has a site of its own, as `f"{n}"`. With it off, neither the probe nor the fill exists. 3x in cycles on `dict_ops`. |
| `JAITHON_JIT_STR_INT_LEAF` | on | `str(n)` on an int through the f-string's int leaf. |
| `JAITHON_JIT_LEAF_IN_REG` | on | A string leaf's answer straight into the result's register. |
| `JAITHON_JIT_SLICE_LEAF` | on | `s[a:b]` on a string through a leaf call in front of the descriptor call. |
| `JAITHON_JIT_LIST_SLICE_LEAF` | on | `xs[a:b]` on a list through a leaf call in front of the descriptor call. |
| `JAITHON_JIT_NEGATE` | on | Arithmetic negation. |
| `JAITHON_JIT_TUPLE` | on | Tuple construction and unpacking. |
| `JAITHON_JIT_INLINE_ALLOC` | on | `ClassName(args)` (`emitCallOut`) pops the page-space free mask inline -- load, lowest bit, header from the cursor's `epochHi` -- instead of calling `jitInstanceAlloc`; an empty mask falls through to that call, and past it to the descriptor. No GC test: the refill that handed the word out charged it (gc.h `jaiPageNew`). -11% cycles on `alloc_churn`; but +2.3% on a 3-float `Vec3` built by compiled methods called from an interpreted loop (quiet machine, `JAITHON_JIT_BARE_ALLOC=0` on both sides) -- the same stores, reached sooner, cost more there; see the next row. |
| `JAITHON_JIT_BARE_ALLOC` | on | `ClassName(args)` with a simple `init` (`emitCallOut`): each field is stored as a whole Value with one `stp` -- `emitTagFor` builds the tag with `movz`, so its high half is the padding, zero -- and when the arguments cover every field the allocation is `jitInstanceAllocBare`, which does not zero fields about to be overwritten. Off: two stores a field and a zeroing allocator. -13% cycles on `alloc_churn`, -2.4% on `binary_trees`; re-measured 2026-10-04 as -4% `object_dispatch`, -5.5% a fully compiled 3-float `Vec3` loop. It COSTS +4.9% on the same `Vec3` built by compiled methods called from an interpreted loop (`acc.add(v.scale(k))`), and only with the page space on: with `JAITHON_GC_PAGES=0` the switch measures 0.1%, and with both this and `JAITHON_JIT_INLINE_ALLOC` off the page space alone is 2.2% FASTER on that loop (quiet machine, 2026-10-04; a loaded-machine review had blamed the page space). Neither half dominates: zeroing with whole stores, or two plain `str x` per field without zeroing, each recover 2-4% on that loop and lose 4-5% on the compiled one (where the switch is worth 8.5%), so both stay on. The pair is used only up to slot 29 (`jaiA64PairOffFits`): its offset tops out at +504, and 512 encodes as -512 -- a 31-field class stored its last field below the instance until that was checked. |
| `JAITHON_LAZY_ENUMERATE` | on | `for (i, x) in xs.enumerate()` over a list. Read by the **interpreter** (`jaiLazyEnumerateOn`, vm.c) as well as this tier: `OP_INVOKE` replaces the eager `list.enumerate()` -- N 2-tuples and a list, all taken apart again by the pair head -- with an `ITER_LIST_ENUM` over one boxed snapshot when the next instruction is the `GET_ITER`, and both tiers arm that head (`emitForIterPairEnum`; OSR iterKind **5**, which takes a plain list head's prologue, reserved registers and index write-back -- not iterKind 4's, that being the list-of-2-tuples head, which shares the dict head's instead). A snapshot, not a view: the eager call copied the elements too, so a loop mutating `xs` sees what it always saw. Off restores the eager call and the refusal "a pair loop over something other than a live dict view or a list of 2-tuples", which is 96 of the 106 pair-loop sites in `lib/jaithon`. |

### Limits

| switch | default | what it does |
| --- | --- | --- |
| `JAITHON_JIT_ARENA_MB` | 4 | Capacity of **both** code arenas, in mebibytes, clamped to 1..64. At the old default of 1 the tier declined **70 distinct bodies** on `check lib/jaithon` with "the code arena is full" -- more than any missing opcode arm -- and 4 takes that to zero, 301 compiled bodies to 369, and interpreted instructions down 13%. |
| `JAITHON_JIT_ALIGN` | 32 | Byte alignment of every compiled body in the code arena, a power of two from 32 to 4096. **A measurement tool first**: at 32 a body's offset modulo 64 still depends on the size of everything compiled before it, so a change that shrinks one body moves every later one and the A/B reads a layout shift as an effect -- `object_dispatch` moved 5% between two settings that changed no instruction of its own, and 0.01% at 64. Set it to 64 on both sides of an A/B. As a default it was not a clean win (fib_recursive +13%, queens +10%, heat_2d -4%, nbody -3% on one loaded-machine sweep), so it is not one. |
| `JAITHON_JIT_ARENA_WINDOW` | on | Limit unseal/seal and the instruction-cache invalidation to the range that changed. Off flips the whole mapping on every compile and invalidates everything written so far. **Not a measured speedup** -- five interleaved pairs on `check lib/jaithon` disagreed in sign, so on this workload the mprotect and the invalidate are not hot. It is kept for the shape: the invalidation is O(bytes written) rather than O(all code emitted so far), and the window cannot leave the back catalogue unexecutable. |
| `JAITHON_JIT_ROOT_LIMIT` | `JIT_MAX_ROOTS` | Roots one call descriptor may carry. |
| `JAITHON_JIT_SHAPE_LIMIT` | `JAI_OSR_SHAPES` | Shapes one site may pin. |

### Diagnostics -- output only, no effect on generated code

| switch | what it prints |
| --- | --- |
| `JAI_JIT_WHY` | Every body considered, compiled, or refused, **with the named reason**. The first thing to run. |
| `JAI_JIT_CHAIN` | The refusal chain for a body -- but only when `compileBody` returns false, so a body that compiled a PARTIAL prefix prints nothing. |
| `JAI_JIT_RECON` | Deopt reconstructions, one line each. A body entered and abandoned every call shows up here and nowhere else. |
| `JAI_JIT_TRACE` | A body going hot. |
| `JAI_JIT_DUMP` | Disassembly of the named function (exact name match). |
| `JAI_JIT_PERFMAP` | Appends `start size name` for every installed body (OSR forms as `name@osrN`) to `/tmp/jaithon-perf-<pid>.map`, so a sampling profile (`xctrace record --template 'Time Profiler'`) can attribute compiled code to functions. |
| `JAITHON_JIT_COLLECT_CLASHES` | Kind clashes gathered during a walk. |

`JAI_JIT_ATTRIB=1` with `--stats` gives exact per-function attribution of
interpreted work (`sum(attrib) == vm.instructionCount`);
`scripts/dev/jit_report.py` drives it. Rank refusals by **distinct sites**, not
events -- the two orderings are nearly opposite.

`scripts/dev/ab.py <SWITCH>` runs the A/B, and measures the noise floor first,
because the floor is usually wider than the change. Six identical runs of
`check --no-cache lib/std` spread **3.36%**, monotonically -- drift, not noise,
so averaging launders it instead of cancelling it. `--pin` sets
`JAITHON_JIT_THRESHOLD=1` and `JAITHON_JIT_TICK_US=100000`, which takes the same
six runs to **0.017%** by removing the sampler's race.

Pinning is not free: nothing stays cold in that regime, so it cannot see a
change to what a caller does when its callee has no compiled form. On
`JAITHON_JIT_NULLABLE_FB` it returns exactly 0.00% with a 0.000% floor -- not
"no effect" but "wrong instrument", and the script says so rather than reporting
a win.


### Widening

A local's kind is fixed for a whole compile. When the walk finds two, it does
not give the body up -- `adoptLocalKindSeen` asks for a wider seed and the
compile is retried. `jaiJitCompileFunc` runs up to four attempts around
`compileFuncOnce`; `compileOsr` does the same around `compileOsrOnce`. Dynamic
wins over nullable where a slot asks for both, being the more general form:

* **nullable** (`nullableLocal`): `SLOT_INST` and `SLOT_MAYBE_INST` of the same
  class are not really two kinds. The maybe-instance is the supertype, holds
  the identical pointer-or-zero, and every field offset resolved against the
  class stays right, so the slot simply takes the wider kind. This is what
  `var at = head; at = at.next` needs, which is every list and tree walk there
  is -- worth 3.6x on one, and only as a chain with three other changes.
* **dynamic** (`dynamicLocal`): two genuinely different kinds. The slot loses
  its register, keeps its frame home so there is somewhere to put a run-time
  tag, and **every read of it guards**: `localIn` loads the tag, compares it
  against `localTagFor`, and deopts on a mismatch -- then chases `Obj.type`,
  and for an instance the class `shapeId` as well, because `VAL_OBJ` cannot
  tell a list from a dict a sibling write left there.

A dynamic slot's kind is therefore a **speculation**, not a fact. It is the
thing the guard exists to check, and it is only true after the guard has run.

### The invariant

> **A tag reconstructed from a compile-time kind is a lie whenever the
> run-time value decides.**
>
> A payload decides null-ness. A `dynamic` slot's kind is a speculation, not a
> fact. A register home carries no tag at all. In each case the tag must be
> read from the value, derived from it, or guarded before the value is trusted.

Three miscompiles found in one week are that one sentence in three costumes.

**1. `fn init(self) {}` returned null instead of the object.** `compileReturnNull`
matches any one-instruction `OP_RETURN_NULL` body, and its caller in
`jaiJitEnter` writes `NULL_VAL` over `slotBase[0]`. But an initializer's
`OP_RETURN_NULL` does not mean null: the interpreter's `opReturn` path (vm.c)
replaces the value with `frame->slots[0]`, the receiver, which is what makes
`Point(1, 2)` an expression. The kind of the body -- "returns
null" -- was read off the opcode; what the instruction actually produces
depends on the function it is in. *Fixed: `compileReturnNull` refuses an initializer.*

**2. A SIGSEGV under `JAITHON_JIT_DEOPT_STRESS`.** A local past the arity starts
as a bare zero in its **register** home, and a register carries no tag. The
deopt stub rebuilt every tag from the compile-time `localKind` and wrote an
unconditional `VAL_OBJ` for object kinds, manufacturing `{VAL_OBJ, obj = NULL}`.
`jaiJitEnterOsr`'s slot scan then dereferenced it, because `IS_INSTANCE` is
`IS_OBJ(v) && AS_OBJ(v)->type == OBJ_INSTANCE` and the tag-only half passed.
*Fixed: the stub takes `emitTagFor`'s payload-dependent `csel` for every
`VAL_OBJ` kind, not only for `SLOT_MAYBE_INST`.*

**3. `m ?? 100` yielded 0 for a nullable local.** The OSR tier accepts the
`dynamic` widening -- the kind becomes a speculation when two paths disagree --
but never implemented its READ side. `localIn`'s `if (e->osr)` branch returns
before reaching the tag-check-and-deopt block the function tier uses, so an OSR
dynamic local was read as a bare payload with no tag check at all, and the
operand-stack entry was then stamped with the walk's last-written kind. *Fixed: a shared `localGuardDynamic`, called from both tiers.*

Note that bug 3 was **correct under `JAITHON_JIT_THRESHOLD=1` and wrong under
`JAITHON_JIT_TICK_US=50`**. The two tiers are complements; a suspected
miscompile has to be tried under both.

Two things guard against a fourth. `scripts/gate/kind_tag_check.py` pins every
place in this directory that maps a `SlotKind` onto a `Value` tag, so a new one
has to be justified; its docstring says what the four legitimate justifications
are and what it cannot see -- bug 1, notably, is not in its shape at all.
`tests/fuzz/kind_mutation.py` generates the other half: its `DYN_SHAPES` family
exists because three confirmed miscompiles lived in dynamic locals, and nothing
in its first family reaches one, since every shape there writes each local
exactly once.

---

## The deopt record

`JitDeoptRecord gDeopt` is a **single global**, and that is safe rather than
lucky:

* the VM is single-threaded, so only one body can be deoptimising at a time;
* the compiled frame is gone by the time C looks at it, so the record cannot
  live on that frame;
* it is **consumed at the innermost frame that sees it**, before anything else
  can write another. That is what makes one record enough at any recursion
  depth: `jaiJitFinishDeopt` finishes the callee in the interpreter and hands
  the value back, rather than letting the record survive across a call.

A record must describe, for the exact bytecode offset the interpreter resumes
at:

* `ip` -- that offset, and it must be an offset the interpreter can *start* an
  instruction at. `deoptSite` is what enforces this: inside an inline the
  interpreter has not made the call yet, and a guard fired mid-instruction
  leaves the model deeper than the interpreter's stack, so only entries this
  instruction pushed may be trimmed back.
* `base` and `nlocals`, and each local's Value.
* `skipLocals` -- one bit per slot for "not mine, leave it". `SLOT_OPAQUE`
  means the compiled body never reads the slot, so `jitArgIn` passed a raw 0
  for it -- but the interpreter *does* read it, and `bindCallArgs` has already
  put the caller's real argument there. Writing the record's null over it
  turned `enter_foreign(module)` into `enter_foreign(null)`. The OSR tier never
  had this bug: `OSR_SYNC_ITER` skips a slot whose tag is `VAL_NULL`.
* `nstack` and the operand stack as it stood, including entries that hold no
  register (a class, `self`, a function, a builtin -- baked as constants) and
  the result of a call that has already happened, whose tag comes out of the
  descriptor because it is *whatever the callee really returned*.

Nothing deferred or borrowed may reach a record: `deoptRecordAt` and
`branchOnDeopt` refuse the compile rather than emit one, so a wrong
`deferSurvives` whitelist entry costs a decline, not a wrong answer.

---

## The polymorphic inline cache, and the one shape that reaches it

`emitInvokePic1` is the most intricate arm in this directory and by some
distance the hardest to arrive at. A fuzzing census sampled 100 generated
programs and **not one** of them emitted a `[jit] pic N-way` line under
`JAI_JIT_WHY=1`; hand-written attempts at the obvious shapes missed it too.
This section is what it actually takes, written down so nobody has to
reconstruct it from the code a third time. The generator has since been taught
the shape and the same census now reads 76 of 100, which is the last
subsection.

### The gate is one NULL

`emitInvoke` reaches the arm from exactly one place: the `OP_INVOKE` branch
where the receiver entry is `SLOT_INST` **and its `Emit::stackClass` is NULL**
-- an instance whose class the walk has not pinned. Everything else about the
site is downstream of that one condition.

And this tier has exactly one producer of an unpinned `SLOT_INST`:
`emitForIterBind`'s **loop-head** arm, under `Emit::elemMixed`. Every other
route to a `SLOT_INST` carries a class and guards it.

| where a `SLOT_INST` comes from | what pins the class |
| --- | --- |
| a parameter (`seedLocals`) | the live argument's own class |
| any other local (`adoptLocalKindSeen`) | the class of the first value bound -- and **both widenings keep one**: `nullable` by construction, `dynamic` by taking the last adopted class |
| a list element (`exemplarKind`, `OP_GET_INDEX`) | the sampled element's class, plus a `shapeId` deopt guard |
| a *nested* `for x in xs` (`emitForIterBind`'s ordinary arm) | the same, off the iterator's sample |
| a field (`OP_GET_FIELD*`) | `SLOT_MAYBE_INST` of the field's class, narrowed to `SLOT_INST` by the null compare at the head of `emitInvoke` |
| a global (`globalKind`) | the live global's class |
| a call's result (`observedReturnKind`, and the module and static arms) | refused outright unless `jaiClassForShape` resolves the recorded shape |

`elemMixed` is not computed by the walk. `jaiJitEnterOsr` computes it, for a
loop head whose iterator is `ITER_LIST`: it scans up to 1024 elements from
index 0 and sets the flag if any is an instance of a class other than the
sampled element's. The head arm then discards the shape and class it would
have pinned, clears `localTyped` for the loop variable, and the slot becomes
"an instance, of no class in particular". Nothing else here does that.

**The arm was therefore OSR-only, and loop-head-only.** It no longer is:
`JAITHON_JIT_POLY_LOOP` lets the ordinary (non-head) arm bind a loop
variable unpinned too, when OP_GET_ITER's live list holds several classes
(`Emit::stackMixed`) and this is that local's first binding -- so a function
the whole-function tier compiles reaches the cache as well. A list loop inside
an OSR body still usually pins, because its loop variable was already typed
from the frame when the form was entered.

### The shape

    for op in ops { ... op.method(args) ... }

with all of the following true at once:

* `ops` is a **list holding instances of two or more classes**, mixed within
  its first 1024 elements. One class is not enough; `elemMixed` stays false and
  the head pins. A trait is not required and neither is a declared element
  type -- `var ops = [A(1), B(2)]` is enough. Nulls are allowed but not dense:
  past one in 64 `jaiJitEnterOsr` declines the head outright.
* **that `for` is the loop the OSR tier entered at**, i.e. a `SIGPROF` tick
  landed on its own back edge and made it `osrTop`. Nesting it inside another
  loop is fine and is what `tests/bench/poly_dispatch` does -- the inner head
  is hot enough to collect its own tick and its own form. What does not work is
  the enclosing loop being the entry and this one being walked inside it.
* the method **returns `int`, `float` or `bool`**, and every way of the site's
  cache agrees. `siteInvokeResultKind` merges the per-way `resultKind` bytes;
  a result carrying a class shape is refused by the arm ("an unpinned receiver
  returning something with a shape") and a *disagreement* between two classes
  merges to `JAI_FB_MIXED`, which takes the whole body down one step earlier
  with "an unpinned receiver's result kind".
* **no wrapping operator anywhere a walk has to cross.** This directory has no
  handler for `OP_ADD_WRAP`, `OP_SUB_WRAP` or `OP_MUL_WRAP` at all -- `grep`
  finds them in `vm.c` and nowhere here -- so a walk stops dead at the first
  one and everything after it is interpreted. That matters in two places and
  is easy to miss in both. In the **loop body** a `+%` before the call means
  the walk never reaches the call. In a **callee** it is worse than it looks:
  the way is supposed to be dropped for having no return kind, but a callee
  that wraps is also the shape that trips the gap recorded below, so it does
  not cost the site one way -- it costs the site the arm. Three classes in one
  list, one of whose methods computes `x *% self.k` and the other two of which
  do not, eight runs each:

      all three checked      pic 3-way, 8 of 8
      one of three wrapping  pic 1-way twice, nothing the other six times,
                             and "a direct callee whose walk never reached
                             a return" in every run

  So a method written for this arm folds with the checked `+ - *`, and stays
  safe by bounding both operands with `%` rather than by wrapping. That is
  what `tests/fuzz/progen.py`'s `step` is and why it exists.
* **the callees are already compiled when the head compiles.** Each way needs
  `ObjFunction::jitFunc`, so every implementation must have passed
  `JAI_JIT_THRESHOLD` before the tick that compiles the loop lands. This is the
  one condition that is about *timing* rather than about the program, and it is
  why the configuration below that samples fastest reaches the arm least often.
* the rest is `jitPic1Admissible`, per way: a public method (`InlineCache::payload`
  is 0 -- a non-public one is cached with the byte set so the interpreter can
  re-run `methodPermitted`, which emitted code cannot), a closure rather than a
  bound native, the callee in the caller's own module at the module version the
  callee compiled against, matching arity and parameter kinds, and the callee's
  own slot 0 specialised to exactly the class this way's shape names.
* no float local is live across the call: `emitInvokePic1` refuses on
  `Emit::fpLive` after `fpReleaseAll` ("an unpinned receiver with a value in
  the float bank").

The site's cache state is *not* a constraint worth worrying about. `IC_MONO`,
`IC_POLY` and `IC_MEGA` are all admitted, and `JAI_IC_OBS_BUDGET` closing
early is fine too: one usable way is enough, and every miss simply falls
through to the descriptor the site would have emitted anyway.

### Where it is reached, and under which switches

`tests/bench/poly_dispatch` prints `pic 8-way (of 8 recorded, state 2) at 97`
from `osr main at 87` -- offset 87 is the `OP_FOR_ITER_BIND` of
`for op in ops`, iter kind 2, and 97 is the `op.apply(acc)` inside it.
`tests/lang/test_jit_poly_receiver.jai` reaches it too, as `pic 2-way (of 4
recorded)` in `drive`, but only under `jaithon test` -- the file has no `main`,
so running it directly compiles the module and stops.

The smallest thing that reaches it is about a dozen lines: two classes with a
`pub fn f(self, x: int) -> int`, a list holding both, and a `for` over that
list inside an outer repeat loop. Measured over the differential fuzzer's six
configurations:

| configuration | one 12-line probe | 100 generated programs |
| --- | --- | --- |
| default | yes | 43% |
| `JAITHON_JIT_DEOPT_STRESS=1` | yes | 54% |
| `JAITHON_JIT_SPLIT_STRESS=1` | yes | 53% |
| `JAITHON_JIT_THRESHOLD=1` | yes -- a half-formed cache is not an obstacle | 50% |
| `JAITHON_JIT_TICK_US=50` | **no** | 26% |
| `JAITHON_NO_JIT=1` | n/a | n/a |

The second column is `tests/fuzz/pic_rate.py`, and it is there because the
first column on its own says something false. `JAITHON_JIT_TICK_US=50` really
does miss the arm on the small probe, every run, and the reason is the ordering
condition above: at 50us the tick lands on the list head before the methods in
the list have been called 64 times, so no way has a `jitFunc`, the arm answers
"no way of this site's cache is usable", and the form cached for the rest of
the run has no cache in it. But "the fastest sampler cannot reach this arm" is
the wrong conclusion to draw from one program. Across a corpus it reaches it in
a quarter of them -- fewer than any other configuration, and not none. What the
switch changes is a *race*, and a program with a longer warm-up before its list
loop goes hot wins it.

### Why the obvious candidates do not

All four were tried and all four fail for reasons that are worth knowing.

* **A local swapped between two classes** -- `var v: Op = A(1)` (or `: any`),
  then `v = A(..)` on one branch and `v = B(..)` on the other. The checker
  requires the annotation, and then `adoptLocalKindSeen` sees two `SLOT_INST`
  of different shapes, asks for the `dynamic` widening, and the retried compile
  pins the slot to *one* of them. What follows is not a polymorphic site but a
  loop that barely runs: the compile refuses once with "local 2 was given two
  kinds, instance and instance", and the form that does compile then fails its
  own entry guard on every other iteration -- 996,332 `osr main stopped: a slot
  pinned to a class now holds a different one` in a two-million-iteration probe.
  Class polymorphism through a local is not something this tier models.
* **`ops[j].method()`** instead of `for op in ops`. `OP_GET_INDEX` predicts
  from one live element through `exemplarKind` and guards the `shapeId`, so the
  receiver is pinned to whichever class element 0 happened to be, and every
  other class deoptimises. `[jit] direct method A.apply`, never a pic.
* **A trait-typed `list[Op]` walked in a nested loop** *when the outer loop is
  the OSR entry*. The inner `OP_FOR_ITER_BIND` then takes `emitForIterBind`'s
  ordinary arm, which samples the element and pins -- and since the sample
  disagrees with the slot on the next class, it refuses with
  "OP_FOR_ITER_BIND: loop variable in local 4 has kind instance, not instance".
  The same source *does* reach the arm once the inner loop collects its own
  tick and becomes an `osrTop` of its own, which is the only difference between
  reaching this arm and not.
* **A monomorphic list.** `elemMixed` stays false and the head pins, exactly as
  intended: a one-class list should get the direct call, not a cache.

### What the fuzzer emits, and what it measured

`tests/fuzz/progen.py` now generates this shape on purpose, in `st_inst_list`,
and `tests/fuzz/pic_rate.py` is the census that says whether it arrives -- one
`[jit] pic` line counted per program, under each configuration the differential
fuzzer runs. Same binary, same seeds, generator before and after:

    seeds 1..100     before   after        seeds 201..300   before   after
      default          0/100    43/100       default          0/100    51/100
      tick             0/100    26/100       tick             0/100    31/100
      deopt            0/100    54/100       deopt            0/100    63/100
      split            0/100    53/100       split            0/100    60/100
      thresh           0/100    50/100       thresh           0/100    55/100
      any              0/100    76/100       any              0/100    83/100

The "after" column is sampler-driven and so is not exactly repeatable: the same
100 seeds re-run gave 38/27/52/51/50 and 80 for `any`. The zeroes are not like
that. Every "before" cell was 0 in every run, which is the point -- a number
that moves by a few programs between runs and a number that is structurally
unreachable do not need the same precision to tell apart.

The old statement already emitted "a list of mixed classes walked by one loop"
and still scored zero, which is the part worth remembering: the shape was
right and three details were wrong. Its list was two or three elements drawn
independently from one pool, so it was often monomorphic and never hot enough
for the head to take a tick of its own; and its body folded with `acc = acc +%
e.kind() +% e.geta()`, whose first `+%` stops the walk, on callees whose own
bodies wrap. Fixing those three -- two distinct classes sampled without
replacement, six or more elements, and a `pub fn step(self, x: int) -> int`
whose body is checked arithmetic bounded by `%` -- is the whole difference
between 0 and 76.

Reaching the arm is not the same as finding anything in it. 1,800 programs x 6
configurations after the change -- 300 at the default seeds and 1,500 more from
seed 5000 -- report no disagreements. The arm is covered now; it is not yet
known to be wrong anywhere.

Four later changes moved the same census, seeds 1..60: 77% default, 85% under
`JAITHON_JIT_TICK_US=50` (was 43% and 26% on the first hundred). A way whose
callee cannot be called directly is inlined instead (`JAITHON_JIT_PIC_INLINE_ONLY`),
a form compiled short of ways is compiled again once the stragglers arrive
(`JAITHON_JIT_PIC_UPGRADE` -- which is what the tick configuration was losing
to), and the whole-function tier reaches the arm through a mixed list
(`JAITHON_JIT_POLY_LOOP`) and through a parameter several classes pass
(`JAITHON_JIT_POLY_PARAM`). The ways themselves are mostly inlined now
(`JAITHON_JIT_INLINE_METHODS`): poly_dispatch's eight all are.

### One gap, recorded rather than fixed

`jitPic1Admissible` says it "mirrors every decision `emitDirectCall` makes
before it commits to emitting". It does not mirror `jitReturnKnown`. A way
whose callee compiled but whose walk never reached an `OP_RETURN` passes
`jitPic1Admissible`, and `emitDirectCall` then refuses it -- after the shape
compare is already in the instruction stream, so the arm has nowhere to fall
back to and sets `e->failed`. The whole loop declines rather than the one way
being dropped.

It is a decline, not a wrong answer, but it is not a free one: the loop stops
compiling **at all**, where without the arm it compiled fine. Reproduced with a
three-class list whose second class computed `x *% self.k` -- `apply walked
only to OP_MUL_WRAP`, so that callee has a `jitFunc` and no `jitReturnKnown`.
A/B'd inside one binary, which is what `JAITHON_JIT_PIC` is for, ten runs each:

    JAITHON_JIT_PIC=1   4-8 x "osr probe0 stopped: a direct callee whose
                        walk never reached a return", and the loop never
                        compiles
    JAITHON_JIT_PIC=0   0 x, and the loop compiles at 235 instructions

Removing the one wrapping multiply turns the same program into
`pic 3-way (of 3 recorded, state 2)` and 447 instructions. The fix, if it is
worth making, is one line in `jitPic1Admissible`: drop a way whose
`jitReturnKnown` is false, the same way it already drops one whose module
version has moved.

The smallest form of it is three classes in one list where exactly one method
wraps, which is worth having written down because it is now a shape the fuzzer
can draw: eight runs give `pic 3-way` eight times with all three checked, and
with one of the three wrapping give `pic 1-way` twice, nothing six times, and
the decline in all eight. A site does not lose the wrapping way. It loses the
arm, and the loop around it.

---

## The pair head over a list of 2-tuples, and why it measured nothing

`JAITHON_JIT_PAIR_LIST` (iterKind 4) is the cheapest kind of arm there is: a
**tier-complement gap**. `emitForIterPair`'s non-dict tail was already a
complete inline step for `for (a, b) in xs` over a list of 2-tuples, guarded
and deopt-safe, and the whole-function tier already reached it. Only
`jaiJitEnterOsr`'s head gate refused the same shape, so one source line
compiled when its body went hot by CALL COUNT and declined when it went hot by
SAMPLING. Closing it is 3 accepted `iterKind == 4` arms (the prologue,
`OSR_SYNC_ITER`, `osrReserved`), a `pairIsDict` that asks the iterKind at a
head instead of the operand-stack shape, and the gate itself. No new step, no
new register, no new prologue -- kind 4 shares kind 3's, because both read the
index out of the `ObjIter` every iteration rather than hoisting it, so there is
nothing to write back at an exit.

It works, and on `check --no-cache lib/jaithon` **it is worth nothing**, and
the reason is the one this file already teaches twice.

* **Ranked by refusal count it looked like the largest thing in the tier**:
  "a pair loop over something other than a live dict view", ~1400 events.
  Ranked by WORK it is not, and the gap between those two orderings is the
  whole result. The gate now names the iterator kind it saw, per "name every
  JIT refusal" -- and the naming is what settles which shape the count was
  about. On `check --no-cache lib/jaithon` the answer is: **all of it is this
  shape**. 1451 events over 25 distinct sites, every one a list, and not one
  `.enumerate()` -- the `Iterator[tuple[int, T]]` sites, which are the majority
  of the compiler's pair loops in SOURCE, never become an `osrTop` here, so
  they never reach this gate at all. That is worth knowing precisely because a
  reasonable reading of the source says the opposite; a source-site census and
  a refusal census answer different questions.
* **The site that carries the work is chained.** `jaithon.ast.node.init` is
  6.18% of the run and its `for (name, tag) in fields_of(kind)` at 73 is the
  largest single pair loop in it. With the arm on, that head clears link 1 and
  stops at link 2: `OP_GET_GLOBAL: _default_field is not a compiled global
  function` -- and `jaithon.ast.schema._default_field`, itself 3.17% of the
  run, never compiles ("a body returning both int and list"). The dict arm was
  the oracle for this before the list arm was built: the same body run over a
  DICT, where the head arm already existed, declined at exactly that global.

So the arm lands green and behind a switch, and the honest number is **no
measurable effect** -- 10 interleaved pairs inside a 3.95% noise floor on a
loaded machine, with only two bodies in the whole workload reaching it
(`check.expr._check_args`, 0.67% of interpreted work, and
`check.kinds._build_ordinals`, 0.00%). On an isolated probe of the shape it
is worth **5.3x** (4,437,264 interpreted instructions to 833,173), which is
what the arm is actually worth and what it will be worth here once
`_default_field` compiles.

### The regression the first build shipped, and the density scan

The first build of this arm was held out of the tree by a review that priced
a shape none of the measurements above contained: a pair loop over a
`list[tuple[str, int?]]`. The head samples ONE element and pins BOTH
component tags from it; the step's component guard is a tag compare; so a
null where the sample had an int is a guard failure, and because the loop is
re-entered on the very next element the failure is paid **once per null**,
not once. With nulls at 1 in 3 that ran **3.50x slower than never
compiling** -- the same bail-per-element trap the list-BIND head had already
hit and answered, and the arm had simply not borrowed the answer.

It now does. `jaiJitEnterOsr` scans the same capped prefix (up to 1024
elements) over the tuple COMPONENTS, counting nulls and components of another
tag than the sample's (the guard cannot tell those two apart, so neither does
the scan), and refuses past **1 in 64** -- by density, not presence, for the
reason the list-BIND head's table gives: one null in a thousand is faster
pinned than interpreted, and refusing on presence throws that away. A sample
whose own component is null is refused at the head by name rather than four
compile attempts later in `emitForIterPair`. The refusals are
`a pair loop whose tuple components are too often null` /
`... too often change kind` / `... whose sampled tuple holds a null component`.

The probe: a 4M-iteration `for (name, v) in ps` over 4000 `(str, int?)`
tuples, `if v is null { nulls += 1 } else { acc += v + name.len() }`, whole
process, best of 5 interleaved, switch OFF (HEAD's gate) against ON, load ~5:

                  wall OFF   wall ON   interpreted OFF   interpreted ON
    nulls 1 in 3    0.191s    0.191s       42,768,674      42,768,674
    nulls 1 in 1000 0.215s    0.042s       48,083,345         274,505
    no nulls        0.216s    0.032s       48,079,341         140,344

The 1-in-3 row is the fix: identical to HEAD to the instruction, because the
head refuses and the interpreter runs the loop exactly as it did before (the
pre-scan build read 0.37s on the same row). The other two rows are what the
arm was always worth on the shape -- 5.1x and 6.8x -- and the 1-in-1000 row
carries its ~4 guard failures per pass at no visible cost.

Nothing else moved. On `check --no-cache lib/jaithon` the function tier
compiles 429 bodies (401 distinct) with the switch off and on; the OSR tier
compiles 43 forms off and 48 on, the five being the pair heads this arm
admits; no body is lost. OSR compile attempts (`[jit] osr ... stopped` plus
forms, under `JAI_JIT_WHY`) go 27,156 to 28,335 -- the 1,454 `a pair loop
over a list` refusals become 625 `already past its last element` re-entries
and 5 forms; none of the three density refusals fires on the compiler at all.
`scripts/dev/ab.py JAITHON_JIT_PAIR_LIST --workload "check --no-cache --stats
lib/jaithon"`: floor 3.33% over 6 runs, pairs +0.10% / -3.33% / +3.40%,
**inside the floor**, which is the expected result -- the chain through
`_default_field` above is unchanged, so the workload still has nothing for
the arm to accelerate. That is not a reason to hold it: the arm is correct,
switch-gated, loses nothing, and the shape it was priced against now costs
exactly what HEAD costs.

---

## Environment switches

Every switch is read once through a cached accessor, except `JAI_JIT_WHY` and
`JAI_JIT_DUMP`, which are read inline because they are only ever on when
somebody is watching. `getenv` is O(environ), and uncached on a hot path it
lets the ambient shell environment perturb a benchmark -- `sort_merge` moved
70ms to 100ms on padding alone. Reading once has a second effect that matters
more: a body compiled with an arm and a body compiled without it can never
coexist in one run.

Almost every knob here exists so a change can be A/B'd **inside one binary**.
That is not a nicety: alternating two builds invalidates `__jaicache__`, and
every sample then pays a stdlib recompile larger than the effect being
measured. Three people measuring one change across two builds got 3.9x, 100x
and 6% slower for what one switch settled in a single command.

### Testing and diagnostic switches

| variable | accessor | effect |
| --- | --- | --- |
| `JAITHON_NO_JIT` | `jaiJitEnabled` (jit.c) | set and not `0`: the tier is off entirely. The reference oracle. |
| `JAI_JIT_WHY` | inline, ~38 sites | why each body or loop declined, and what it compiled. The first thing to reach for. |
| `JAI_JIT_CHAIN` | `jitChainOn` | the whole chain of refusals, not just the first. **Link 1 is measured; links below are probed** by forcing link 1 unarmed, which abandons the rest of its block -- treat them as hints. |
| `JAI_JIT_RECON` | `jitReconTrace` | one line per applied deopt record: name, ip, base, nlocals, nstack. |
| `JAI_JIT_TRACE` | inline (jit.c, jit_loop.c) | when a body goes hot, and where a compiled loop landed. |
| `JAI_JIT_DUMP=<fn>` | inline, both tiers | writes that function's words to `jit_<fn>.bin` (`jit_osr_<fn>_<top>.bin` for a loop) and prints the bytecode-offset-to-instruction map. Read it back with `llvm-mc --disassemble --triple=aarch64`. Three of this tier's bugs were found no other way. |
| `JAITHON_JIT_DEOPT_STRESS` | `jitDeoptStress` | set and not `0`: deoptimise wherever a deopt is possible. |
| `JAITHON_JIT_SPLIT_STRESS` | `jitSplitStress` | set and not `0`: put the split operand bank's boundary into every eligible OSR body, not only the ones that need it. The failure it hunts is silent -- a value written one register past the end of the first run. |

### Tuning knobs

All default **on**; all turned off with `=0`, except the four numeric ones.

| variable | accessor | what it gates |
| --- | --- | --- |
| `JAITHON_JIT_TICK_US` | inline (jit.c) | sampler period in microseconds, 50..100000, default 1000. Not a boolean. At the old 250us, 6.17% of a `check --no-cache` run was in `_sigtramp` -- *delivering* the signal, not acting on it, and larger than any refusal left in the tier. |
| `JAITHON_JIT_TICK_ARM` | `jaiJitArmOnFirstTick` | `0` restores the old two-tick wait before a body is armed. |
| `JAITHON_JIT_MAIN_EDGES` | inline (jit.c, `jaiJitMainStarted`) | the warm-up count, not a boolean: `0` off, default 4096, read once. `jaiRunFile` calls `jaiJitMainStarted` before the main module's body and again before `main()`; it sets `jaiInterrupted` to 3, which OP_LOOP's existing test already sends to the safepoint, and the safepoint counts it down -- so the count costs the fast path nothing (a new branch there costs 11% untaken) and the slow path at most 4096 back edges per start. At zero the state becomes a tick (2) on that back edge, marked so that `jaiJitSample` passes it outward at once (`jaiJitWantEnclosing`) even on the body's first tick; a real tick during the count ends it (`onTick` turns 3 into 2). Before it, when the timer landed was the program's luck: mandelbrot's interpreted count read 0.05M-3.4M run to run on identical code, and with the timer slowed to 100ms (`TICK_US=100000`) 47M; with it, 117K every run. Re-arming the timer to fire 100us into `main` was tried first and changed nothing -- macOS delivers ITIMER_PROF on its own accounting granularity. |
| `JAITHON_JIT_OBJ_EQ` | `jitObjEquality` | the call-out arm at the end of each equality chain. |
| `JAITHON_JIT_ELEM_DECL` | `elemDeclOn` | using a list's *declared* element kind when there is no live list to sample -- a fact, not a guess, since the same byte pins `ObjList::stg` while the list is still empty. |
| `JAITHON_JIT_DYNAMIC_RETURN` | `jitDynamicReturn` | the `SLOT_DYNAMIC` join in `mergeReturnKind`. |
| `JAITHON_JIT_STATIC_METHOD` | `jitStaticMethodOn` | the static-method arm of `OP_GET_FIELD` on a `SLOT_CLASS` receiver. |
| `JAITHON_JIT_COLD_RETRY` | `jaiJitColdRetryOn` | not charging a not-yet-returned-callee decline to the attempt budget (`jaiJitEnter`). |
| `JAITHON_JIT_MODULE_CALLS` | `jitModuleCalls` | the module-member call arm at `OP_INVOKE`. |
| `JAITHON_JIT_CLASS_CALLS` | `jitClassCalls` | the static-member call arm at `OP_INVOKE`. |
| `JAITHON_JIT_MODULE_NATIVE` | `jitModuleNativeCalls` | both halves of the `__prim__.f64_sqrt` arm at once -- neither compiles anything useful alone. |
| `JAITHON_JIT_STRCMP` | `jitStrCmpOn` | string comparison instead of a decline. |
| `JAITHON_JIT_STRCMP_EQ` | `jitStrCmpEqOn` | sending `==`/`!=` on not-known-interned strings to the leaf call rather than a pointer arm that deopts every iteration. |
| `JAITHON_JIT_STATIC_FIELD` | `jitStaticFieldEnabled` | the `SLOT_CLASS` arm of `OP_GET_FIELD`. |
| `JAITHON_JIT_MODULE_FIELD` | `moduleFieldOn` | reading `math.PI` and its kin. |
| `JAITHON_JIT_FIELD_DECL_KIND` | `jitDeclaredFieldKindEnabled` | `declaredScalarFieldKind`'s `OP_GET_FIELD` arm. |
| `JAITHON_JIT_SOFT_FIELD` | `jitSoftField` | taking the soft unarmed path for a name that is not a field of the pinned class -- non-OSR only, because inside a loop nothing is cold. |
| `JAITHON_JIT_RET_LIST` | `retListKindOn` | promoting an observed list return from `SLOT_OBJ` to `SLOT_LIST`. |
| `JAITHON_JIT_LIST_PROBE` | `listProbeOn` | accepting a predicted-list receiver. |
| `JAITHON_JIT_RET_OBJTYPE` | `retObjTypeOn` | keeping the callee's observed object type. |
| `JAITHON_JIT_LIST_RESULT` | `jitListResult` | predicting the result of a list method. |
| `JAITHON_JIT_LIST_SCALAR` | `jitListScalarResult` | predicting `sum`/`min`/`max`. |
| `JAITHON_JIT_RETURN_KNOWN` | `jitReturnKnownOn` | refusing a direct callee whose walk never reached a return. `jitReturnKind` is written even then, so without this the arm trusts a value nothing established. |
| `JAITHON_JIT_ANY_GUARD` | `jitAnyGuard` | the `list` / instance / `any` type-guard arms -- `any` is satisfied by every value, so its guard is genuinely nothing. |
| `JAITHON_JIT_STR_GUARD` | `jitStrGuard` | the `str` and `dict` arms of `OP_TYPE_GUARD` on a `SLOT_OBJ`: one `Obj.type` check and a deopt, after which the entry carries the fact (`stackObjType`). Emitted even when `stackObjType` already names the type, since that is a prediction and the guard is a semantic check; only a character from the ASCII table skips it. Before it, "a `str` guard on a object" declined the whole body around any function returning `str(n)` or a value read out of a container. |
| `JAITHON_JIT_FDIV_GUARD` | `jitFdivGuard` | the float divide-by-zero guard in `emitAddSubDiv`. This one is a CORRECTNESS fix, not an optimisation: without it compiled `x / 0.0` yields inf where the interpreter raises `DivisionByZeroError`. The switch exists only to price the guard (one `fcmp`, one not-taken branch per compiled float division) inside one binary. Never ship with it off. |
| `JAITHON_JIT_PIC` | `jitPicEnabled` | the one-way inline cache at an unpinned `OP_INVOKE`. |
| `JAITHON_JIT_NULL_PAIR` | `jitNullPair` | `x == null` on a `SLOT_OBJ`. Equality only, and object kinds only: a `SLOT_INT` is also never null, but zero is a perfectly good int. |
| `JAITHON_JIT_FUSED_DISCARD` | `jitFusedDiscard` | fusing `OP_POP_RETURN_NULL` after a discarded call. |
| `JAITHON_JIT_STR_ITER` | `jitStrIter` | iterating a string. |
| `JAITHON_JIT_STR_HEAD` | `jitStringHead` | `ITER_STRING` at an OSR loop head (`iterKind` 6). Off, `compileOsr` refuses it with "an iterator kind with no loop-head arm", which is what the tier did until 2026-09-07 -- a per-character loop long enough to reach the OSR tier then ran entirely interpreted, 290ms against 30ms for the same work as an indexed `while`. Worth 11% of `fmt --check lib/jaithon`. The head steps ASCII inline and deopts on a byte >= 0x80, so `compileOsr` samples the string first and refuses one that is more than 1-in-64 non-ASCII BYTES. |
| `JAITHON_JIT_INVOKE_SOFT` | `jitInvokeSoft` | master switch for both dead-path softenings in `emitInvoke`. Each also has its own switch, because they do not behave alike. Both interpret ONE invoke instead of declining the whole enclosing body, and both are guarded on having emitted nothing and moved the model nothing (`e->count`/`e->depth` unchanged from the top of `emitInvoke`), settling the FP bank and deferred values first -- the walk's own skip protocol, without which a deferred value reaching the guard becomes a spurious second refusal. |
| `JAITHON_JIT_INVOKE_SOFT_RECV` | `jitInvokeSoftRecv` | the last-resort "a receiver of kind X" arm. Kept ON: it is the one that pays on both workloads -- `fmt --check lib/jaithon` 115.4M interpreted instructions to 109.9M, `check checker.jai` 32.95M to 32.15M. `_scan_punct` is the shape it was found on: 20 of its 21 declines are one OP_INVOKE inside a control-character error branch that a clean run never reaches, and it cost the whole body. |
| `JAITHON_JIT_INVOKE_SOFT_COLD` | `jitInvokeSoftCold` | the "a method that has not returned yet" arm (`obsReturnKind == JAI_FB_NONE`). Kept OFF -- see the note at its definition; it helps `check` and costs `fmt` more. |
| `JAITHON_JIT_OSR_GLOBAL_SOFT` | `jitOsrGlobalSoft` | drops the `!e->osr` gate on `OP_GET_GLOBAL`'s unarmed fallback, so an OSR loop naming a global whose callee has not compiled is compiled UP TO that read and interprets from there -- which the function tier has always done. Off, the whole loop declines. `fmt.comment.scan_comments` is why: after two source-level hoists it had exactly one blocker left at its head, `_dedent_continuations` (itself blocked by `str.split` having no observed result kind), and that one refusal cost the largest interpreted body in `fmt --check lib/jaithon` -- 108.6M interpreted instructions to **83.4M**, with the body dropping off the attribution table entirely. Gate changes here on `tests/fuzz/differential.py` (its TICK_US=50 configuration is the only one that drives the OSR tier) and a gc-stress leg, not on `make test` alone. |
| `JAITHON_JIT_GLOBAL_ENUM` | `jitGlobalEnum` | `OP_GET_GLOBAL` resolving a module-level ENUM by value, the way it already resolves a class, function and native -- and `OP_GET_FIELD`'s enum-variant fold then skipping its Obj.type and shapeId guards because a pinned receiver has nothing left to prove. Off, `TokenKind.Plus` takes the by-address path and costs FOUR deopt records a site; `lexer._punct_kind` has 51 such sites, needed 316 records against a `JIT_MAX_DEOPT` of 160, and did not compile at all. On, that body compiles at deopt=114. Sound for the same reason the other by-value arms are: `jaiValueIsInertGlobal` returns false for `OBJ_ENUM`, so a rebind bumps `ObjModule::version` and retires the form -- teaching this arm a new kind means updating that function too. |
| `JAITHON_JIT_BUILTIN_CLASS` | `jitBuiltinClass` | `globalClass` falling back to `vm.builtins` for a CLASS. Builtin exception types live only there, and no resolver looked for a class in it -- `globalNative`/`globalNamespace` check `vm.builtins` but accept only IS_NATIVE/IS_MODULE -- so `throw KeyError(…)` stopped the walk with "is not a compiled global function", permanently, since no later compile can make a builtin class into one. |
| `JAITHON_JIT_COMP_ACC` | `jitCompAcc` | reaching a list comprehension's accumulator through its FRAME SLOT at `OP_LIST_APPEND`. The accumulator is pushed before the loop head, so it sits below the window the OSR model tracks; off, the append refuses with "an append reaching past the model" and every comprehension in the language runs its loop interpreted. Costs one reserved callee-saved register (`osrReserved`) on any loop region that contains such an append. |
| `JAITHON_JIT_ITER_STG` | `jitIterStorage` | the storage dispatch at a NESTED `for x in <list>` (`emitForIterBind`'s shape-1 arm). Off, that arm emits `emitListBoxedGuard` alone, which is what it did until 2026-09-07: a `push`-built `list[int]` is `LIST_STORE_I64`, so the guard failed on every loop entry and the inner loop ran interpreted. Worth 3.6x on a nested-loop probe and 2x on `graph_bfs`. The switch exists to price it in one binary. |
| `JAITHON_JIT_CONCAT_LOCALS` | `jitConcatLocals` | `a + b` on two `SLOT_OBJ` locals. |
| `JAITHON_JIT_MEMBERSHIP` | `jitMembership` | `in`. |
| `JAITHON_JIT_DICT_LEAF` | `jitDictLeaf` | the string-keyed dict leaves, `jitDictGetStr`, `jitDictSetStr` and `jitDictHasStr`, placed in FRONT of the dict read, `dict.get`, dict store and `in` descriptor calls, which stay behind them as the slow path. A leaf runs no user code and cannot allocate, so it needs no descriptor, no roots and no native dispatch -- ~200 instructions around a ~20-instruction probe. Worth 1.47x in cycles on `dict_ops`, and 2.2x on a get-then-set loop. Anything the leaf cannot settle -- a key that is not a string, a miss under `d[k]`, a stored key of another kind whose hash matches, a value a typed dict refuses -- takes the descriptor call, which is not a deopt. Int keys take their own leaves (`jitDictGetInt` and its kin) with no key guard, answering only from an entry holding that very int; a hash-equal key of another kind -- the equal float -- is the descriptor call's to judge. 3.6x on an int-keyed counting probe. |
| `JAITHON_JIT_DICT_ADD` | `jitDictAddFuse` | `emitDictAddFused`: at the `get` of `d[k] = d.get(k, n) + c` -- `GET d, GET k, GET d, GET k, <n>, INVOKE get 2, INT c, ADD, SET_INDEX`, with both loads of `d` and of `k` from the same locals and `n` an int -- one call to `jitDictAddStr`, which adds `c` to the entry's int value (or to `n` for an absent key) and stores it back: one probe and one call where the unfused form makes two of each. It is emitted in front of the unfused code and branches past the `SET_INDEX` on success, so anything it does not settle -- a key that is not a string, a value that is not an int, an overflow, a typed dict that refuses -- runs that code exactly as before. Worth 1.15x in cycles on `dict_ops`. `emitDictAugAddFused` does the same for `d[k] += c` at its `OP_DUP2` (`DUP2, GET_INDEX, INT c, ADD, SET_INDEX`), with no default: an absent key is the read's KeyError, which the unfused code raises. |
| `JAITHON_JIT_DICT_PROBE` | `jitDictProbeOn` | a dict the model knows only by `stackObjType` -- what a local bound from `{}` in the same body holds, since OP_BUILD_DICT records the type and there is no live dict to sample until the body runs -- compiled as one: an invoke on it resolves its method against `jitDictProbe` (one empty dict, a permanent root), and a store into it and `in` against it take the dict arms. Each guards the real receiver's type at run time, so the probe chooses arms and never removes a guard. Before it, `var counts = {}` followed by any use of `counts` declined the whole function. |
| `JAITHON_JIT_DICT_KEYS_ITER` | `jitDictKeysIterOn` | `OP_GET_ITER` on a dict building the `ITER_DICT_KEYS` iterator (`jitMakeDictKeysIter`), and the shape-6 arm of `OP_FOR_ITER_BIND` stepping it inline -- the dict-items pair head's step, binding the key alone, every guard before the index is written back. Before it, `for k in d` declined the whole function ("iterating a dict, not a list or a range"). The OSR tier still has no head for it. |
| `JAITHON_JIT_DUP` | `jitDupOn` | arming `OP_DUP` and `OP_DUP2` as register copies that carry the entry's facts with them. Unarmed, `xs[i] += v` and `d[k] += v` -- `DUP2, GET_INDEX, <v>, ADD, SET_INDEX` -- stopped the walk, and every loop holding one ran interpreted from that point on. |
| `JAITHON_JIT_FMT_LEAF` | `jitFmtLeafOn` | `jaiValueFormatLeaf` (value.c) in front of the `OP_FORMAT` descriptor call to `jitFormat`, which stays behind it as the slow path. The leaf never collects -- it declines when `jaiGCWanted()` is already true, which is the only state in which its one possible allocation could collect -- so the parts go down as plain arguments with no root fill, no root-range push and pop and no wrapper. Emitted only when every part is an int, a bool or an object that may be a string, and not when the parts the compiler can see -- literal text, and the samples of string locals -- already add up past the 32-byte short limit: such a leaf could only copy up to the limit and give up, every time (a 9-part log line paid ~310 instructions an f-string for it). A float or a long result takes the slow path. `JAITHON_FMT_SHORT=0` (value.c, read through `jaiValueFormatShortOn`) also keeps this leaf and the `str(n)` leaf from being emitted. Worth 1.15x in cycles on `dict_ops`. |
| `JAITHON_JIT_FMT_INT_LEAF` | `jitFormatIntLeaf` | `jaiValueFormatIntLeaf` (value.c) in place of the general f-string leaf for the commonest shape there is: one int hole with at most one string run on either side -- `f"k{i}"`, `f"{name}{i}"`, `f"item-{i},"`. Three registers instead of the parts written to memory and read back, no dispatch on each part's tag, and no loop whose constants the compiler keeps live across the call: 45 fewer instructions an f-string, worth 1.10x in cycles on `dict_ops`. The leaf checks that the runs are strings; anything else is the general path. |
| `JAITHON_JIT_STR_INT_LEAF` | `jitStrIntLeaf` | `str(n)` with an int argument through `jaiValueFormatIntLeaf` -- it is `f"{n}"` -- in front of the call to the native, which stays behind it as the slow path. |
| `JAITHON_JIT_LEAF_IN_REG` | `jitLeafInReg` | the f-string and string-slice leaves moving their answer straight into the register the result will occupy and joining the descriptor path after its load, rather than storing it into the descriptor's result slot for that load to read back -- a store-forwarding round trip on the path from the leaf to whatever consumes the string. |
| `JAITHON_JIT_SLICE_LEAF` | `jitSliceLeaf` | `jaiStringSliceLeaf` (object_string.c) in front of the string arm of `OP_GET_SLICE`'s descriptor call to `jitGetSlice`, which stays behind it as the slow path. It answers only the slices that allocate nothing -- one byte, from the shared ASCII table, and a short slice the intern table already holds -- from a source already known to be ASCII, and with int bounds and no step; everything else is the descriptor call. A scanner cutting the same words out of a text is the shape: worth 1.19x in cycles on `word_freq`. |
| `JAITHON_JIT_LIST_SLICE_LEAF` | `jitListSliceLeaf` | `jaiListSliceLeaf` (object.c, so the page-space pop inlines into it) in front of the list arm of the same descriptor call, through the same emitter (`emitSliceLeaf`). Int bounds, no step, any storage. Unlike the string leaf it allocates -- the list from the page space, unzeroed with every field written, and the array from the small-block bins -- so it declines when `jaiGCWanted()` is true, the one state in which that allocation could collect; otherwise nothing can, and nothing needs rooting. Copies at the SOURCE'S storage width (the stg-vs-elemKind trap in `jaiListSlice`), whole words inline up to 32 bytes. A slice went from ~64 to ~32 cycles: 4M two-element slices 1.79x, `sort_merge` 1.076-1.086x (instructions -14%). |
| `JAITHON_JIT_TUPLE` | `jitTuple` | building and unpacking tuples. |
| `JAITHON_JIT_NEGATE` | `jitNegate` | `OP_NEG`. |
| `JAITHON_JIT_COLLECT_CLASHES` | `jitCollectClashes` | collecting every clashing local in one measuring pass instead of one retry per clash. |
| `JAITHON_JIT_RECOMPILE` | `jitRecompileOn` (jit_compile.c) | recording the callee that truncated a body's walk, and so the whole recompile above. Gated where the record is WRITTEN, not where it is read, so the off side really does nothing. |
| `JAITHON_JIT_ROOT_LIMIT` | `jitRootLimit` | numeric, 1..`JIT_MAX_ROOTS`; `=10` puts the root cap back where it was when it shared the register budget. |
| `JAITHON_JIT_SHAPE_LIMIT` | `jitShapeLimit` | numeric, 1..`JAI_OSR_SHAPES`; the OSR instance-shape cap. |
| `JAITHON_JIT_OSR_FORMS` | `osrFormCap` | numeric, 1..`JAI_OSR_MAX`; compiled loops one body may keep. |
| `JAITHON_JIT_OSR_SLOTS` | `osrSlotCap` | numeric, 1..`JAI_OSR_SLOTS`; slots an OSR form may describe. |

In every numeric case the *array* is always the wider one -- only the refusal
moves -- so these cannot corrupt anything, only decline more.

---

## Debugging a suspected miscompile

The tier answering differently from the interpreter is the only bug class here
that matters. Work it in this order.

1. **Establish it is the tier.** `JAITHON_NO_JIT=1` is the reference. If the
   answer changes, the tier is wrong; if it does not, stop looking here.

2. **Ask which tier.** They are complements and a bug in one is routinely
   invisible in the other:
   * `JAITHON_JIT_THRESHOLD=1` -- the whole-function tier compiles every body
     on its first call instead of its 64th. It overrides the `JAI_JIT_THRESHOLD`
     default in `jit.h`, and it is a TESTING switch: results must be UNCHANGED
     under it, because a lower threshold only hands the tier a half-settled
     inline cache and every prediction is guarded. A difference is a miscompile.
   * `JAITHON_JIT_TICK_US=50` -- the fastest legal sampler rate, so the OSR
     tier reaches loops that a short run would never make hot.

   Bug 3 above was *correct* under the first and *wrong* under the second.

3. **`JAITHON_JIT_DEOPT_STRESS=1`.** Deopt paths are the least-exercised code
   in the tier and the most likely to be wrong, because a deopt is the one
   moment compiled state has to be translated back into interpreter state in
   full. Bug 2 only ever appeared under this. `JAITHON_JIT_SPLIT_STRESS=1` does
   the same job for the split operand bank.

4. **`JAI_JIT_WHY=1`** to see what actually compiled -- a body that "walked only
   to `OP_GET_GLOBAL` at 53" compiled two instructions and interpreted the rest,
   which is not the same thing as compiling. `JAI_JIT_CHAIN=1` for what it would
   stop at next. `JAI_JIT_RECON=1` for the deopts that actually fired.

5. **`JAI_JIT_DUMP=<function>`** and read the code. `llvm-mc --disassemble
   --triple=aarch64` on the emitted bytes, against the printed
   bytecode-offset-to-instruction map. Register-plan mistakes are not visible
   any other way.

6. **`tests/fuzz/`** is the systematic version of all of the above, and is the
   gate for the class of bug that has cost this project the most.
   `kind_mutation.py` warms a loop until it compiles, then puts a *different
   kind* where the tier sampled one, runs each generated program four ways and
   diffs them against the interpreter. Three families, because each reaches
   something the others cannot: containers and fields, `DYN_SHAPES` for a local
   written with two kinds in one body, and `CALL_SHAPES` for a kind violated on
   a *callee's* return. `iter_mutation.py` asks the other half of the question:
   what if the *container* changes under a compiled walk. Both exit non-zero on
   any disagreement; `make kind-fuzz` runs them.

Related gates. In `make test`: `make jit-fusion-check` (no opcode may newly
lack an arm) and `make kind-tag-check` (no new tag reconstructed from a kind),
both pure text. Run deliberately: `make jit-compile-check` (a `test_jit_*` case
must actually reach compiled code), `make jit-declines-check`,
`make jit-split-check`, `make jit-coverage`, `make kind-fuzz`.

One warning that is not about correctness but will waste a day: **never
benchmark two builds against each other here.** Use the switches. And a binary
run from outside the repo cannot find `boot/seed.bin` and does 30x more
interpreted work.
