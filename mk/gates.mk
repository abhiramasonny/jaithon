# The static gates: pure text, no compiler, cheap enough to be unconditional.
#
# Every one of these exists because two things that had to agree were written
# down twice and nothing checked them. The reason lives above each target; what
# belongs here is only the shared property -- they cost milliseconds, which is
# why `test` can depend on all of them and why a new invariant should arrive as
# one of these rather than as a comment asking people to be careful.

# Three tables describe the opcode list -- JAI_OPCODES in chunk.c (the wire
# format), _OPS in emit.jai (a hand-transcribed copy), and spec/BYTECODE.md --
# and nothing checked they agreed until this. It found four opcodes shipped and
# undocumented. Pure text, so it costs nothing to run on every `make test`.
.PHONY: opcode-check
opcode-check:
	@python3 scripts/gate/opcode_table_check.py

# The tree is layered, but the Makefile globs every .c into one binary, so
# nothing stopped a file including across the layers. This pins the graph in
# src/layers.manifest; a NEW edge has to be written down, which is the review
# it had none of. Pure text, so it costs nothing to run on every `make test`.
.PHONY: layer-check
layer-check:
	@python3 scripts/gate/layer_check.py

# jaicv's recorded cases cover what OpenCV was asked to record, so an export
# nobody recorded is an export nobody ran -- that is how `find_homography` with
# RANSAC shipped raising OverflowError. `test_reachable.jai` closes the gap by
# calling each one, and this keeps that file complete. jaitensor is held to the
# same rule. Pure text, so it costs nothing to run on every `make test`.
.PHONY: exports-check
exports-check:
	@python3 scripts/gate/exports_reachable_check.py

# The other half of that: not "does anyone call this export" but "does the name
# this import asks for exist at all". Modules resolve at run time, so a wrong
# name is an ImportError on the line that first runs, and `check` only reports
# a name the module never declares -- it never asks whether the module EXPORTS
# it, so `from jainum import gpu` type-checks clean and raises at run time.
# Pure text, and it needs no built compiler, so it runs before `check` does.
.PHONY: import-check
import-check:
	@python3 scripts/import_names_check.py

# Splitting a file is when linkage gets widened: every helper the two halves
# shared has to lose `static`, and nothing narrows it again. Nothing catches
# that -- the compiler can only warn about `static`, and a function nothing
# calls links fine. This reads `nm` over the object tree, which is the only
# witness that survives a function whose ADDRESS is taken rather than called,
# so the JIT's runtime thunks are not mistaken for dead. The one gate that is
# not pure text: it needs the objects, hence the prerequisite.
.PHONY: linkage-check
linkage-check: $(OBJS)
	@python3 scripts/gate/dead_code_check.py --build $(BUILD)

# sum(jaiOpCounts) == vm.instructionCount, which is the only evidence that no
# dispatch path skips the census. VM_NEXT_HINT skipped it and loop_sum's
# OP_LOOP -- 14.28% of that run -- reported as zero, so every histogram taken
# before this gate existed was wrong. Out of `test` on purpose: it needs a
# second full build with -DJAI_OPCODE_STATS.
.PHONY: opstats-check
opstats-check:
	@./scripts/gate/opstats_check.sh

# Every instruction offset of every function in lib, tests and examples must
# resolve to the same source span it did before. The line table's encoding has
# no differential oracle behind it (spec/BYTECODE.md §11), so this golden is the
# oracle. Re-capture only when a corpus SOURCE changed:
#   scripts/gate/linetable_golden.sh capture
.PHONY: linetable-check
linetable-check:
	@./scripts/gate/linetable_golden.sh check

# No opcode may NEWLY lack an arm in the function JIT. An unarmed opcode now
# deoptimises at its own offset rather than declining the whole function, so it
# is no longer a coverage cliff -- but the deopt is an unconditional exit from
# compiled code, so one on a hot path still costs the rest of that body, and one
# on a loop's straight-line path costs an entry and a deopt per iteration.
# Measured twice the hard way: 906ms vs 26ms on one shape, and sort_merge
# 270ms -> 510ms on another. Fused opcodes must always be armed; the rest
# ratchet against tests/vm/jit_unarmed.baseline, whose header says what being
# on it costs.
.PHONY: jit-fusion-check
jit-fusion-check:
	@python3 scripts/gate/jit_fusion_check.py

# A tag reconstructed from a compile-time SlotKind is a lie whenever the RUN-TIME
# VALUE decides -- the payload decides null-ness, a dynamic slot's kind is a
# speculation, and a register home carries no tag at all. Two confirmed
# miscompiles were exactly that: a deopt stub that wrote VAL_OBJ over a bare zero
# and handed the interpreter {VAL_OBJ, obj = NULL}, and an OSR dynamic local read
# with no tag check. This pins every SlotKind-to-tag site in src/vm/jit so a new
# one has to be justified, and fails outright on SLOT_MAYBE_INST mapped to a
# constant. Pure text, so it costs nothing to run on every `make test`.
.PHONY: kind-tag-check
kind-tag-check:
	@python3 scripts/gate/kind_tag_check.py

# A switch is how this tier gets measured -- an A/B in one binary, interleaved,
# is the only number it accepts -- so fifty of them accumulated, and with no
# index the only way to find one was to grep. This checks both directions:
# every getenv in src/vm/jit has a row in that README, and every row still has
# a getenv. Pure text, so it runs on every `make test`.
.PHONY: switch-doc-check
switch-doc-check:
	@python3 scripts/gate/switch_doc_check.py

# A seeded module whose import is not itself seeded makes the tree unable to
# compile AT ALL -- and since `make reseed` needs a working compiler to build
# the next seed, it cannot rebuild its way out: boot/ has to be restored by hand
# and every __jaicache__ wiped. `seed-check` catches it, but only by building a
# whole seed and watching the compiler abort. This is text, so it runs first and
# costs nothing.
.PHONY: seed-closure-check
seed-closure-check:
	@python3 scripts/gate/seed_closure_check.py

# Where a branch keeps its displacement is written down twice, in verify.c and
# in opt/chunk.jai, and neither consults the other. A missing entry makes the
# optimiser's rebuild DROP whatever followed the loop -- a chunk that simply
# stops, with no diagnostic. Pure text, so it runs on every `make test`.
.PHONY: branch-table-check
branch-table-check:
	@python3 scripts/gate/branch_table_check.py

# The builtin method surface is written down twice -- k*MethodNames[] in
# builtins.c, which dir() answers from, and twelve dispatch tables in four C
# shapes, which a real call resolves through -- and they drifted to 46 names
# once. An advertised name the runtime will not bind raises AttributeError at
# the user's run time, because `jaithon check` types every method call on a
# builtin as `any` and rejects nothing. tests/stdlib/test_method_tables.jai
# hand-copies six of the twelve lists into Jaithon, so it cannot see a name
# added to builtins.c after it was written; this reads builtins.c itself and
# then asks the built binary about every name -- hence the $(TARGET)
# prerequisite, which the pure-text gates above do not need.
.PHONY: method-surface-check
method-surface-check: $(TARGET)
	@JAITHON=$(CURDIR)/$(TARGET) python3 scripts/gate/method_surface_check.py
