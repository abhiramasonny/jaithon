# The bootstrap seed, and the workspace and package wiring around it.
#
# The seed is the one artefact a broken compiler can poison for good: `reseed`
# needs a working compiler to build the next seed, so a bad seed cannot rebuild
# its way out -- boot/ has to come back from a commit. Every target here is
# shaped by that, which is why reseed re-checks the fixpoint rather than
# trusting the compiler that just wrote it.

#: Regenerate boot/seed.c from the images the front end currently needs.
#:
#: Runs the compiler once to populate __jaicache__, embeds what it finds, then
#: rebuilds and checks the fixpoint still holds. Writing a seed without that
#: check would let a broken compiler seed the next one, which is the one
#: failure a bootstrap cannot recover from on its own.
.PHONY: reseed
reseed: $(TARGET)
	@echo "  SEED    populating __jaicache__"
	@find lib -name '__jaicache__' -type d -exec rm -rf {} + 2>/dev/null || true
# The seed is NOT disabled here. It used to be, so that the front end's own
# closure was compiled from source rather than served by the seed being
# replaced -- but that only worked because the C front end compiled it inside
# the bootstrap window. With one front end the window has only the seed, so
# disabling it leaves nothing that can compile anything. seed_touch.jai boots on
# the previous seed and compiles the next one's sources explicitly, which is the
# same guarantee by a route that does not need a second compiler.
	@JAITHON_PATH=$(CURDIR)/lib ./$(TARGET) run scripts/gate/seed_touch.jai >/dev/null 2>&1 || true
# STALE COMMENT, KEPT AS A WARNING -- do not follow it. It argued for
# `lib/jaithon` rather than `lib`, and the invocation below has said `lib` for
# some time. The comment describes the OLD seed_touch, which populated the cache
# by importing a module and letting the side effects land; that did vary run to
# run. It does not any more: seed_touch walks the tree and compiles each file
# explicitly, so what is collected is exactly what was built.
#
# Narrowing to `lib/jaithon` now would WEDGE THE BOOTSTRAP. The compiler imports
# std modules while it is itself loading -- `std.json` among them -- inside the
# window where the compiler does not yet exist, so std has to be seeded too.
# scripts/gate/seed_touch.jai's SEED_ROOT carries that reasoning in full.
#
# The walk is blanket rather than a dependency closure. Workspace packages live
# outside lib, so they no longer enter the seed. Narrowing this to
# `lib/jaithon` would still break bootstrap because the compiler imports std
# modules while it loads.
#
# Original note, for the record: running the compiler writes cache entries for
# whatever it imports, so collecting the whole tree embeds a set that varies run
# to run. Measured, 44 modules and then 47 across two reseeds of an unchanged
# tree. What is embedded has to be exactly what the step above set out to build.
	@python3 scripts/dev/gen_seed.py lib boot/seed.c --manifest boot/seed.manifest lib
	@$(MAKE) --no-print-directory
# The rebuild above embeds the new seed, which changes JAI_BUILD_ID, which
# invalidates every .jaic just written -- so reseeding used to hand back a tree
# with a cold cache. That is not a correctness problem and it is an expensive
# one: `tests/repl/bindings_gc.repl` runs under --gc-stress, where a collection
# happens per allocation, and compiling std from source under it takes eight
# minutes (measured, twice). Warming here costs a second and removes the cliff.
	@echo "  SEED    warming __jaicache__"
	@JAITHON_PATH=$(CURDIR)/lib ./$(TARGET) --front=jai run scripts/gate/seed_touch.jai >/dev/null 2>&1 || true
	@$(MAKE) --no-print-directory fixpoint-check
	@$(MAKE) --no-print-directory seed-check

#: The seed alone must bootstrap the compiler: wipe every cache, compile with
#: the installed library excluded, and check that the same run fails with the
#: seed disabled. Without the negative half it cannot tell "the seed works"
#: from "the seed was never needed".
.PHONY: seed-check
seed-check: $(TARGET)
	@scripts/gate/seed_check.sh

#: Compile each source twice with the self-hosted front end and compare. With
#: the differential oracle retired this is the gate that says the front end is
#: deterministic: an image that is not reproducible run to run can never
#: satisfy `stage1.jaic == stage2.jaic`.
.PHONY: fixpoint-check
fixpoint-check: $(TARGET)
	@scripts/gate/fixpoint_check.sh $(if $(PATHS),$(PATHS),lib/std)

.PHONY: package-check workspace-sync
#: The members list is derived from the manifests that are present, so that a
#: new package does not need every author to edit the same shared line.
workspace-sync:
	@python3 scripts/dev/sync_workspace.py

package-check: workspace-sync
	@python3 scripts/gate/check_packages.py
