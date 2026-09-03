# Jaithon 3.4 — what to build next, and why it is not a rewrite

Four architectures for 3.4 were designed independently and each judged by four
adversarial reviewers — feasibility, payoff, risk, fit — who were told to verify
every claim against the tree rather than to be fair.

    "Keep What Compiled"                        4.75 / 10
    "Ask Once" — a demand-driven front end      4.50
    "the tier gets a shape" — CFG + a total meet 4.25
    "Observe, then compile"                     4.00

**Every proposal took a fatal objection from every lens**, and the objections
were not stylistic. Three of the four had their *headline measurement*
falsified by a reviewer who re-ran it:

* "Keep What Compiled" rested on the claim that the code arena breaks
  measurement validity. Two reviewers reproduced the census under
  `scripts/dev/ab.py --pin` and it did not hold as stated.
* "Ask Once" attributed its 4.9x to the wrong stage: `JAI_JIT_ATTRIB=1` on the
  proposal's own probe put 73% of the run in `_join`, a string built one
  character at a time — not in the signature loading the plan proposed to
  rewrite in 420 lines.
* "the tier gets a shape" declared an ordering constraint its own reviewer
  measured at about a millisecond, and its flagship stage fused fixpoint
  iteration with emission, which the tier cannot survive.

That is the finding. **3.4 is not a rewrite**, and the evidence for that is not
timidity — it is sixteen independent attempts to justify one, all of which
failed against the actual code.

## What survived

The salvage lists converged hard. These items were named by **four of four**
reviewers of their own proposal, and several by reviewers of other proposals:

1. **Raise the code arena.** *(landed)* There are **two** 1 MiB arenas —
   `jaiJitArena()` in `jit_arena.c`, which the function and OSR tiers write
   into, and a second in `jit.c` for the fallback forms — and sizing only one
   measures nothing. That is why this went unfound: the first sweep resized
   `jit.c`'s and concluded capacity was irrelevant. With both raised to 4 MiB,
   `check --no-cache lib/jaithon` goes from **70 distinct bodies declined with
   "the code arena is full" to zero**, from **301 compiled bodies to 369**, and
   interpreted instructions fall **13.3%..12.5%** across three interleaved pairs
   against a 4.75% noise floor. The largest refusal cause in the tier was a
   guessed constant, not a missing opcode arm.

2. **Range-limited unseal/seal.** `jaiCodeArenaUnseal` mprotects the *whole*
   arena back to read-write, so any early return between unseal and seal leaves
   every previously compiled body unexecutable — a failure class documented at
   `jit_loop.c:236-241`. Unsealing only `[page_floor(used), capacity)` removes
   it by construction. Correctness-only, ~150 lines, in a file no seed and no
   cache key depends on.

3. **`jaiChunkCfg`, consumed by nothing.** `verify.c` pass 4 already *is* a
   worklist CFG fixpoint — `boundary[]`, `depth[]`, `work[]`, a per-opcode edge
   enumerator — and it computes the graph, keeps the depths, and throws the
   graph away. Keeping it costs ~200 lines and buys a real CFG for whatever
   wants one later, with a gate over every function in `lib/` and `packages/`.
   Adopted *because* it has no consumer: it is the one piece of every IR
   proposal that cannot miscompile anything.

4. **Fix the attribution dump, then `_join`.** *(landed)* The dump printed
   `fn->name`, so rows were ambiguous -- and `qualifiedName` does NOT fix it,
   because `serialize_read.c` sets it equal to `name` for anything from a cached
   image, which is nearly everything. The DEFINING MODULE does, and one
   `jitFnLabel` helper now qualifies all 41 diagnostic sites, so
   `JAI_JIT_WHY` and the attribution dump agree and `jit_report.py` can join
   them again. Then
   `check/modsig.jai`'s `_join` builds its result one character at a time and is
   73% of a probe reviewers ran. ~110 lines, one reseed, no format change.

5. ~~**Structural intern keys.**~~ **Dropped, measured.**
   `universe.jai:_intern_key` does build a `list[str]` and f-string-format every
   argument to make a dictionary key, and the proposal was right about the code.
   It is **0.20% of the compiler's interpreted work**, ranked 86th, with
   `intern` itself at 0.27%. A structural hash with a collision bucket is real
   work for less than half a percent. This is the fourth proposal claim to fail
   the same way, and the discipline the rest of this file is built on applies to
   this file too.

6. **Two text gates**, in the house style: `_field_kind_code`'s integer literals
   in `emitter.jai` must equal the `FieldKind` enum in `object.h`, and the six
   `.jaic` wire constants must match across the language boundary.

## What was rejected, and why

* **A sea-of-nodes or SSA IR for the tier.** The scope bound the whole case
  rested on — "no SSA and no phis needed, because `valueXReg(index)` is a pure
  function of the operand-stack index" — is false once the lattice admits a
  merge, and the leaf set the proposal shipped showed the lattice had been
  copied from HotSpot rather than derived from jaithon's kinds.
* **A query-system front end.** The measured win was misattributed; the real
  cost was one quadratic string join. Rewriting signature loading in 420 lines
  would have bought what a ten-line fix buys.
* **A per-slot profiling substrate.** A reviewer enumerated the entire
  observation space of a merged per-parameter record and found no case where it
  yields a fact the tier does not already have.
* **A bigger `SlotKind` lattice as a *first* move.** The lattice is genuinely
  the weak point — a join signature that hashed 15 kinds into 2 bits shipped a
  SIGSEGV, see `tests/golden/jit_join_kind_collision.jai` — but the fix that
  pays is a *total meet with proven laws*, and that is item 3's successor, not
  a rewrite.

## The order

Each step lands green, on its own, and is worth having if 3.4 stops there.

    1. arena capacity, both arenas, one tunable            DONE  -13%
    2. the attribution dump names a function unambiguously  DONE
    3. _join stops building a string one character at a time DONE
    4. range-limited unseal/seal, with its two-stencil test DONE  (not a
       speedup -- kept for shape; see JAITHON_JIT_ARENA_WINDOW)
    5. jaiChunkCfg + tests/vm/test_chunk_cfg.c + the gate
    6. the two wire-constant gates                          DONE
    7. DROPPED -- structural intern keys measured at 0.20%, ranked 86th
    7'. the three hottest bodies instead, which the fixed attribution dump
        made legible for the first time: jaithon.ast.node.init (6.2%),
        jaithon.compile.lexer._push (6.1%), jaithon.compile.lexer.init (5.6%)
        -- 18% of the compiler between them, each with a NAMED refusal
    7''. NOT MERGED -- the pair-over-list-of-tuples OSR head arm. Built,
        correct, switch-gated, loses no compiled body, and measured at ZERO on
        two independent A/Bs. Three adversarial reviews: one found no defect,
        one priced an unmentioned cost (compile attempts 445 -> 1035 on
        lib/jaithon), and one found a REGRESSION -- a pair loop whose components
        are often null now compiles and then bails at the entry guard once per
        null element, 3.50x slower than HEAD. The arm samples one element and
        pins both component tags with no null-density check, where the
        list-BIND head refuses past 1 in 64. Held until that check exists AND
        the chain below it is cleared, because until then it buys nothing.
    8. only then: a total meet over SlotKind, with the lattice laws
       (commutativity, associativity, absorption) checked exhaustively by a
       gate rather than argued in a comment

## Where the chain actually leads

The scoping pass for the loop-head arms is the most useful negative result since
the design phase, and it reorders what is left.

**The refusal ranking does not survive contact with the work.** "A pair loop over
something other than a live dict view" is the largest refusal in the compiler by
count -- and only about **10 of the 106** pair-loop sites in `lib/jaithon` are
the shape it names. The other 96 are `.enumerate()`, which returns a user
iterator (`lib/std/core.jai:75`) and hits the same site wearing the same
message. Ranking by that string credited one shape with another's events. Same
lesson as counting events instead of distinct sites, one level further in: the
refusal STRING is not the refusal.

**Both headline refusals are chains, and link 2 is the same in both.**

* `jaithon.ast.node.init` (6.18%) is blocked at its pair loop -- but a probe of
  the identical body over a DICT, where the head arm already exists, still
  declines, at the callee. The real blocker is
  `jaithon.ast.schema._default_field` (3.17%), which never compiles:
  **"OP_RETURN: a body returning both int and list"**.
* `jaithon.compile.lexer.init` (5.61%) needs a string-iteration arm that does
  not exist -- and would still be worth nothing, because its accumulator is a
  comprehension target sitting on the interpreter's stack BELOW the OSR entry,
  which the OSR model cannot see. That is an entry-contract change, not an arm.

`_default_field` returns `-1`, `[]`, `Span.none()`, `false`, `0`, `0.0`. No
single `SlotKind` covers int-or-list, and no widening rule in this tier can
invent one. **The chain ends at the lattice**, which is item 8 -- so item 8 is
not the last thing on the list, it is the thing the list was pointing at.
Measured ceiling on a body that does compile once its chain is clear:
58.5M -> 0.76M instructions, 77x.

## 3.4, measured: what would make it drastically better

Researched 2026-09-01 with the repo's own instruments, after the four
architecture proposals died on falsified measurements. The question was not
"what does rustc do" but "where do the compiler's 568M interpreted instructions
actually go". The answer is not an IR.

### The numbers

`check --no-cache --stats lib/jaithon` (1.12 MB of source, 31.7k lines):

    568.7M interpreted instructions     485 per source byte
     14.1M allocations                   12 per source byte
     24.9M calls
    171 KB/s wall, best of three         (CPython's compiler: tens of MB/s)

Where it goes, by module (`JAI_JIT_ATTRIB=1`, exact, sums to the total):

    33.4%  jaithon.compile.parse
    23.0%  jaithon.compile.lexer          lex + parse = 56%
    20.3%  jaithon.compile.check
    10.8%  jaithon.ast
     4.4%  jaithon.compile.resolve
     3.7%  jaithon.compile.token
     2.9%  jaithon.compile.emit

**The checker is a fifth of it. Lexing and parsing are more than half.** Every
design proposal so far has aimed at the checker or the JIT.

### The shapes, and what they cost

The hot code is written in the slow shape of four data representations. Each
was priced with a pair of probes that compute the same answer
(`tests/bench/shapes/`, `./jaithon run --stats`):

| what | slow shape | fast shape | instructions | allocations |
|---|---|---|---|---|
| lexer scan | `[c for c in source]`, a `list[str]` of 1-char objects | `source.bytes()`, a `list[int]` | 32.7M -> 1.6M, **20x** | 674k -> 37k, **18x** |
| enum -> value table (`_compare_op` and 90 `if kind == TokenKind.X` chains) | `match` chain | `dict[K, V]` | 21.6M -> 9.1M, 2.4x | -- |
| same | `match` chain | `list` indexed by ordinal | 21.6M -> 0.8M, **27x** | -- |
| AST node fields (`Node.fields: dict[str, any]`) | dict per node | `list[any]` slots behind the same accessors | 11.2M -> 0.47M, **24x** | -- |
| same | dict per node | one class per kind, real fields | 11.2M -> 0.14M, **82x** | 401k -> 201k, 2x |
| one token | `Token` + `Span` + substring | `Token(kind, start, end)`, lazy text | ~flat | 219k -> 117k, **1.9x** |

The lexer's 20x is the JIT, not the allocation: with `JAITHON_NO_JIT=1` both
shapes cost ~10M. The `list[int]` loop compiles silently; the `list[str]` loop
refuses at the comprehension -- "an iterator kind with no loop-head arm", the
same refusal that is `lexer.init`'s 5.6% today. **The tier already arms the
fast shapes.** The front end simply never uses them.

The real lexer, measured on 334 KB of synthetic source: 106,827 tokens at
**194 instructions and 3.5 allocations per token**.

### The four changes, in order

All four are in the seeded front end (one reseed each; the emitter is not
touched, so no fixpoint dance). None changes the language, the bytecode, the
`.jaic` format, or the on-disk AST (`ast_encode.jai` reaches `.fields` at two
sites, both through `fields_of`).

1. **Lexer over bytes.** `lexer.jai` and its twin in `repl.jai` (which
   duplicates the `chars`/`offsets` scheme) scan `source.bytes()`; character
   comparisons become integer comparisons; tokens keep byte offsets and
   slice text lazily. API unchanged: `tokenize() -> list[Token]`, 8 call
   sites. Covers 23% + 3.7% of the run and the 5.6% comprehension refusal.
2. **Ordinal tables.** A 5-line builtin exposing the enum tag the VM already
   holds (`vm.c` pushes `AS_ENUM_VAL(v)->tag` for `OP_ENUM_TAG`); then
   `_compare_op`, `_multiplicative_op`, `_additive_op`, `_assign_op` and the
   91 `if kind == TokenKind.X` chains in `parse/` become indexed lists.
   Covers ~6.4% directly and a large slice of `parse`'s 33%.
3. **Slot-array Node.** `Node.fields` becomes a `list[any]` positioned by a
   per-kind field index derived from `_RECORDS`; the seven accessors
   (`get`/`set`/`has`/`child`/`require`/`text`/`children`) keep their names so
   the 363 `.child(` sites and everything else do not change; the two encoder
   lines index by position. `init` stops walking the schema and calling
   `_default_field` per field -- the defaults become one prebuilt list per
   kind, copied. Covers `node.init` 6.2% + `_default_field` 3.2% + the 470
   "dict holds more than one kind" JIT refusals that are all this dict.
   Later, per-kind classes generated from `_RECORDS` get the remaining 3.4x,
   but that touches every builder and is not the first step.
4. **Flat tokens.** `Token(kind, start, end, flags)`, `span` and `text`
   computed on demand. Halves token allocation.

### What to expect, and what is projection

Derived from the attribution, not measured: 1 + 2 + 3 + 4 remove or shrink
about 45% of today's interpreted work outright, and cut allocation by roughly
half (GC is 22% of wall). That is the floor: ~1.8x on `check`. The larger
claim -- that once the lexer and parser run on integers and slot arrays the
existing JIT compiles most of the remaining 33% in `parse` the way it compiled
the probe loops -- is a projection, and the whole point of doing these in
order is that each one is measured before the next is started. A fair target
is 1 MB/s from 171 KB/s. Anyone claiming more than that before step 2 lands is
repeating the mistake the four proposals made.

### Correction: the parser's `TokenKind` chains are NOT the operator-table fix

The note below said "the 91 `if kind == TokenKind.X` chains in `parse/` are the
same fix and are next". They are not, and the difference is the same one that
made the pair-loop refusal misleading: **a shape that looks alike is not alike.**

The four operator tables mapped a kind to a VALUE, so a list indexed by
`ordinal()` replaced the whole chain. The parser's chains DISPATCH -- each arm
builds a different node and returns. There is nothing to put in a table.

Worse, the chains are not what blocks those bodies. Ranked by the interpreted
work of the bodies each refusal stops (`JAI_JIT_WHY` joined to
`JAI_JIT_ATTRIB`, which is the ranking that matters -- see
`rank-refusals-by-work-not-by-count`):

    66.7M  a list loop already past its last element      (mostly OSR timer noise)
    57.1M  a local of no known kind
    54.4M  OP_INVOKE: a method that has not returned yet
    50.9M  OP_GET_INDEX: the live dict is empty or holds more than one kind
    40.8M  a pair loop already past its last element
    36.8M  this loop head is out of compile attempts
    34.0M  OP_FOR_ITER_PAIR: a pair's loop variables have kinds object and ...
    34.0M  OP_CALL: a callee returning dynamic

`_parse_primary` (4.6%) spends all five attempts on **"OP_GET_FIELD_LOCAL: no
live receiver to read local 3's field off"**, and `_parse_unary` (2.0%) on
**"OP_NE: a compare of a int with a object"**. Neither is a chain.

Two entries there are worth naming. The 50.9M dict refusal **is `Node.fields`**
-- step 3 below deletes it rather than teaching the tier to compile it. And
"OP_CALL: a callee returning dynamic" (34.0M) is NEW, created by the dynamic
return kind: callers deliberately refuse `SLOT_DYNAMIC` rather than widen an
accept-list. That was the safe choice for landing it; teaching one caller shape
to consume a tagged return is the obvious follow-up, and it is now measurable.

### Landed so far (2026-09-01)

* `ordinal()` on every enum value -- the 5-line builtin step 2 needed, plus
  the checker rule, the surface gate's thirteenth receiver kind, and a spec
  paragraph. A method the enum declares with that name still wins.
* The four parser operator tables as ordinal-indexed lists: 37.3M -> 11.7M
  instructions with a shared lookup helper, then inlined. Whole run
  568.7M -> 525.8M before inlining. The 91 `if kind == TokenKind.X` chains in
  `parse/` are the same fix and are next.

### Landed: the lazy `enumerate` iterator

`JAITHON_LAZY_ENUMERATE`. The eager `list.enumerate()` -- a C native building N
2-tuples up front -- is replaced, when the next op is `OP_GET_ITER`, by a
snapshot iterator the interpreter steps straight into two slots, plus arms in
both tiers. On a 200k-element probe: **2,017,838 allocations become 1,306**, and
interpreted instructions 384,013 -> 53,893, with identical output.

It was held because it claimed OSR `iterKind == 4`, which the pair-over-tuples
head already used. The renumber to 5 turned out to be the smaller half of the
problem: the enumerate step indexes `ObjList::items` and advances its own index
register, so it is a LIST head and needs kind 2's prologue, `osrReserved == 5u`
and `OSR_SYNC_ITER`'s index write-back. Every one of those ladders keys off "not
3 and not 4" -- numbering it 4 would have silently given it the dict prologue,
with no `JIT_START_REG` and no write-back.

**The review rejected the first build over a real defect.** `jitListHeadSample`
returned early on any non-instance sample, so the 1-in-64 null census never ran
for a `list[int?]`; switching the arm on moved such loops off the protected
kind-4 path onto an unprotected one -- 3,333,335 entry-guard bails and **1.15x
slower than refusing**. That is the same hazard that made the pair arm 3.50x
slower, wearing a different shape. The census now covers non-instance samples,
which also closes an inherited hole in the plain bind head. On a 2M `list[int?]`
at one null in three: no-JIT 0.45s, arm off 0.24s, **arm on 0.20s**, zero bails.

### What this says about the earlier plan

Item 8, the total meet over `SlotKind`, was the end of the refusal chain for
`_default_field`. Step 3 deletes `_default_field`'s mixed return instead of
teaching the tier to compile it. That is the pattern: **change the data so the
tier it already has applies**, before building tier machinery to chase data
that was never in a compilable shape.

## The method, which is the durable part

Every number above is an A/B **in one binary**, with the two forms interleaved
and the noise floor measured first — `scripts/dev/ab.py`. Six identical runs of
`check --no-cache lib/std` spread 3.36% and did so *monotonically*, so averaging
launders drift rather than cancelling it. `--pin` removes the OSR sampler's race
and takes that to 0.017%, at the cost of measuring a regime where nothing stays
cold — which is why it reports exactly 0.00% for a switch it cannot reach, and
says so instead of claiming a win.

Three of four proposals died on measurement, not on design. That is the lesson
worth keeping.
