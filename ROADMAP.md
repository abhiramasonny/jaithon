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

5. **Structural intern keys.** `universe.jai:_intern_key` builds a `list[str]`
   and f-string-formats every argument to make a dictionary key. A structural
   hash with a collision bucket replaces it, behind a switch.

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
    7. structural intern keys, behind a switch
    8. only then: a total meet over SlotKind, with the lattice laws
       (commutativity, associativity, absorption) checked exhaustively by a
       gate rather than argued in a comment

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
