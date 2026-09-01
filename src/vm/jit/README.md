# The compiled tier

An accelerator that may always decline. Everything here is optional: turn it
off with `JAITHON_NO_JIT=1` and the interpreter runs every program exactly as
it would have. That is the first thing to know about it, and the reason the
boundary contract below is written the way it is.

`jit.h` is the whole surface the rest of the VM sees. `jit.c` holds the entry
point, the SIGPROF sampler and two stencil compilers; `jit_func.c` holds both
real tiers; `jit_loop.c` holds a shape-matched compiler for one exact loop;
`jit_arena.c` and `jit_arm64.{c,h}` are executable memory and the instruction
encoders.

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
| `SLOT_LIST` | `ObjList *` |
| `SLOT_OBJ` | some heap object this tier does not model: load, pass, store, root, nothing else |
| `SLOT_ITER` | an `ObjIter` this body built; its index stays in memory |
| `SLOT_NULL` | what `-> void` returns: a defined zero whose tag is `VAL_NULL` |
| `SLOT_OPAQUE` | present in a register, but nothing may be done with it -- the body never reads this slot |
| `SLOT_SELF`, `SLOT_CLASS`, `SLOT_FUNC`, `SLOT_NATIVE` | compile-time constants; `holdsRegister` says these four occupy no register at all |

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

And `SLOT_MAYBE_INST` has **no compile-time tag at all**. The same register is
a pointer or a zero and only the payload says which. `emitTagFor` is the one
place that gets it right: a `subs`/`csel` off the payload rather than a `movz`
of a constant.

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
depends on the function it is in. *The fix is to refuse initializers in
`compileReturnNull`.*

**2. A SIGSEGV under `JAITHON_JIT_DEOPT_STRESS`.** A local past the arity starts
as a bare zero in its **register** home, and a register carries no tag. The
deopt stub rebuilt every tag from the compile-time `localKind` and wrote an
unconditional `VAL_OBJ` for object kinds, manufacturing `{VAL_OBJ, obj = NULL}`.
`jaiJitEnterOsr`'s slot scan then dereferenced it, because `IS_INSTANCE` is
`IS_OBJ(v) && AS_OBJ(v)->type == OBJ_INSTANCE` and the tag-only half passed.
*The fix is to take `emitTagFor`'s payload-dependent `csel` for every `VAL_OBJ`
kind, not only for `SLOT_MAYBE_INST`.*

**3. `m ?? 100` yielded 0 for a nullable local.** The OSR tier accepts the
`dynamic` widening -- the kind becomes a speculation when two paths disagree --
but never implemented its READ side. `localIn`'s `if (e->osr)` branch returns
before reaching the tag-check-and-deopt block the function tier uses, so an OSR
dynamic local was read as a bare payload with no tag check at all, and the
operand-stack entry was then stamped with the walk's last-written kind. *The fix
is a shared `localGuardDynamic` called from both tiers.*

Note that bug 3 was **correct under `JAI_JIT_THRESHOLD=1` and wrong under
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
| `JAITHON_JIT_OBJ_EQ` | `jitObjEquality` | the call-out arm at the end of each equality chain. |
| `JAITHON_JIT_ELEM_DECL` | `elemDeclOn` | using a list's *declared* element kind when there is no live list to sample -- a fact, not a guess, since the same byte pins `ObjList::stg` while the list is still empty. |
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
| `JAITHON_JIT_PIC` | `jitPicEnabled` | the one-way inline cache at an unpinned `OP_INVOKE`. |
| `JAITHON_JIT_NULL_PAIR` | `jitNullPair` | `x == null` on a `SLOT_OBJ`. Equality only, and object kinds only: a `SLOT_INT` is also never null, but zero is a perfectly good int. |
| `JAITHON_JIT_FUSED_DISCARD` | `jitFusedDiscard` | fusing `OP_POP_RETURN_NULL` after a discarded call. |
| `JAITHON_JIT_STR_ITER` | `jitStrIter` | iterating a string. |
| `JAITHON_JIT_CONCAT_LOCALS` | `jitConcatLocals` | `a + b` on two `SLOT_OBJ` locals. |
| `JAITHON_JIT_MEMBERSHIP` | `jitMembership` | `in`. |
| `JAITHON_JIT_TUPLE` | `jitTuple` | building and unpacking tuples. |
| `JAITHON_JIT_NEGATE` | `jitNegate` | `OP_NEG`. |
| `JAITHON_JIT_COLLECT_CLASHES` | `jitCollectClashes` | collecting every clashing local in one measuring pass instead of one retry per clash. |
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
   * `JAI_JIT_THRESHOLD` -- it is a plain `#define` in `jit.h`, not an env
     var. Set it to 1 and rebuild, and the whole-function tier compiles
     everything on its first call.
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
