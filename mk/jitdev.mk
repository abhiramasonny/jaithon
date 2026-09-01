# Working ON the JIT: coverage, decline censuses, fuzzers, split checks.
#
# Deliberately NOT in `test`. These read probes under docs/ that are untracked,
# or they measure rather than assert, or they are slow enough that making them a
# gate would get them skipped. They are for the person changing the tier.

# What the tier covers, one ordinary idiom at a time. Each probe in docs/probes/
# is a single hot loop; the script runs it with the tier on and with
# JAITHON_NO_JIT=1 and reads the ratio. It answers the question the decline
# census cannot: a census ranks by FREQUENCY and is not hotness-correct -- it
# put 168 `dict.get` refusals at the top, every one in a setup path that runs
# once per process. Probes are hot by construction.
#
# Not in `test`: docs/ is untracked, so the probes are not guaranteed present,
# and one probe is a KNOWN gap (a nullable scalar, see
# docs/research/PLAN-nullable-scalars.md) that would fail a gate for ever.
.PHONY: jit-coverage
jit-coverage: $(TARGET)
	@./scripts/dev/jit_coverage.sh

# The JIT's decline census, collapsed to distinct reasons and compared against a
# recorded baseline. A NEW reason means the tier stopped compiling something it
# used to; a reason that disappears is fine and needs no commit. Coverage only:
# roadmap.md §7 is explicit that clearing a decline is not itself a speedup.
# Runs the benchmark suite twice (warm, then measure), so it is not in `test`.
.PHONY: jit-declines-check
jit-declines-check:
	@./scripts/dev/jit_declines.sh check

# A `test_jit_*` case can pass with its own fix reverted: `jaithon test` never
# makes the body it means to exercise hot enough to compile, so the test only
# ever proves the interpreter agrees with itself. Measured across six files in
# tests/lang, the fraction of their OWN functions that reached `[jit] compiled`
# or an OSR form ranged 9/18 down to 1/12 (docs/roadmap.md §7). Every author
# verified teeth by reverting and watching the test still pass, so the fixes
# are real -- but nothing enforced it, and an edit that drops a loop below
# JAI_JIT_THRESHOLD hollows the gate with no signal.
#
# scripts/gate/jit_compile_check.py closes that hole with a declaration next to the
# test rather than a separate list that drifts: a `# jit-compiles: name, ...`
# line naming the helpers (never the `test_*` wrappers) the file claims reach
# the compiled tier. It fails if a name is not a real top-level `fn` in that
# file (the marker rotted) and if a name never reaches `[jit] compiled` or
# `[jit] osr` under JAI_JIT_WHY=1 (the thing it claims stopped being true).
#
# Deliberately OUT of `test`. Two reasons, not one:
#
#   - It needs its own per-file `JAI_JIT_WHY=1` process, which is not the
#     single combined `jaithon test tests/lang ...` run_tests.sh already does
#     for the "unit" layer -- turning JAI_JIT_WHY on for that whole run would
#     bury the signal in decline spew from every file, marked or not.
#   - Some marked functions can only be reached through the sampler-driven OSR
#     tier (one has 5 arguments, past JIT_MAX_ARITY; a couple exist
#     specifically to test OSR's own entry mechanics, which the whole-function
#     tier does not share). That tier is not deterministic no matter how long
#     a loop runs -- roadmap.md §7 -- so even with generous warm-up this is a
#     coverage gate with an irreducibly small chance of a false alarm, and
#     tying that to every `make test` risks exactly the flaky-gate problem
#     roadmap.md §7 spends a page warning about. Run it deliberately instead,
#     the same way jit-declines-check and kind-fuzz are.
.PHONY: jit-compile-check
jit-compile-check: $(TARGET)
	@python3 scripts/gate/jit_compile_check.py

# The split operand bank makes the operand stack two runs of registers instead
# of one, and a site that adds an index to a base can then land one past the end
# of the first run -- silently, into a register nothing reads.
# JAITHON_JIT_SPLIT_STRESS=1 puts that boundary into every OSR body that can
# take one rather than only the ones that pay for it, and this runs the
# benchmarks against the interpreter under it. Not in `test`: it runs the
# benchmark suite four times over.
.PHONY: jit-split-check
jit-split-check:
	@./scripts/gate/jit_split_check.sh

# The kind-mutation fuzzer: 144 generated programs, each warming a loop until it
# compiles and then putting a different kind where the tier sampled one, run
# four ways and diffed against the interpreter.
#
# This is the gate for the class of bug that has cost this project the most.
# BOTH silent miscompiles found on 2026-08-12 were "the tier sampled a kind and
# the program changed it": a str in a list bound its POINTER as an integer, and
# a bool local was read eight bytes wide. Neither was caught by anything.
#
# Its teeth are established, not assumed: run against a tree built from d41ec16
# (before the list-element fix) it reports 6 of 144 mismatched, all of them
# list_for-add-int-to-*, printing raw pointers and IEEE bit patterns where the
# interpreter raises TypeError. Against the fixed tree, 0 of 144.
#
# Its companion, iter_mutation.py, asks the other half of the question: not
# "what if the KIND changes" but "what if the CONTAINER does". A compiled loop
# caches an iterator's index, its limit and a pointer to the backing array, and
# ObjList::version is the only thing that says the program moved the ground.
# Teeth established the same way: with that one branchOnDeopt removed from
# OP_FOR_ITER_BIND's list arm, 15 of 21 cases mismatch and the compiled loop
# silently returns 2115/65 -- walking a REALLOCATED array -- where the
# interpreter raises RuntimeError.
#
# Out of `make test` because it is ~4 minutes. Run it when touching the tier.
.PHONY: kind-fuzz
kind-fuzz: $(TARGET)
	@python3 tests/fuzz/kind_mutation.py --warm $(KIND_FUZZ_WARM)
	@python3 tests/fuzz/iter_mutation.py --warm $(KIND_FUZZ_WARM)

# The differential fuzzer: RANDOM whole programs rather than a fixed matrix.
#
# kind-fuzz above enumerates two known bug shapes exhaustively. This asks the
# complementary question -- what shape has nobody thought of -- by generating
# programs from a grammar biased at what the tier compiles, then running each
# one under five configurations of the tier and diffing. The oracle needs no
# expected output: jit.h calls the tier "an accelerator that may always
# decline", so declining is always legal and answering differently never is.
#
# Deliberately OUT of `test`, for the reason roadmap.md §7 gives at length: the
# OSR tier is sampler-driven and not deterministic, so a hit that depends on
# where a SIGPROF landed can appear and vanish between runs. That is exactly
# the flaky-gate problem, and tying it to every `make test` would buy the
# project a gate nobody trusts. Run it deliberately, like jit-declines-check.
#
# Teeth, on the tree it was written against (2c3807cc, no local edits): seeds
# 0..999 turned up 10 programs the tier gets wrong. Seven SEGFAULT, and every
# one of the seven faults at the same instruction -- the `case SLOT_INST` kind
# check in jaiJitEnterOsr, which reaches AS_OBJ(v)->type through a slot holding
# an object TAG over a null pointer. Three print a different number; the
# smallest of those (seed 232) reduces to a nullable local read through `??`
# on the first iteration after its `if` stops firing. None of the 3,258 tests
# in `make test` covers any of it.
#
# 400 programs x 5 configurations at warm=1500 measured 257s wall on twelve
# cores; 600 more measured 507s on ten. A hit prints the seed; reduce it with
#     python3 tests/fuzz/differential.py --shrink SEED
.PHONY: jit-fuzz
jit-fuzz: $(TARGET)
	@python3 tests/fuzz/differential.py --count $(FUZZ_COUNT) \
	    --warm $(FUZZ_WARM)

# The enum-`match` differential: random enums, random match bodies over them,
# every subject the checker lets through, six configurations, diffed. Out of
# `make test` for the same reason kind-fuzz is -- minutes of subprocesses. Its
# own header says how its teeth were established.
.PHONY: match-fuzz
match-fuzz: $(TARGET)
	@python3 tests/fuzz/match_differential.py --count $(MATCH_FUZZ_COUNT)
