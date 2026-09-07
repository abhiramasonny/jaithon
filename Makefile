# Jaithon build system.
#
#   make            release build -> ./jaithon
#   make debug      -O0 -g, assertions, GC stress available
#   make test       build + run the full test suite
#   make bench              language benches vs python3/c++/java
#   make bench jaicv         image operations vs OpenCV on the CPU (~2 min)
#   make bench jainum        numpy's array surface vs numpy on the CPU
#   make bench jaiframe      pandas' frame surface vs pandas on the CPU
#   make bench jaitensor     GPU/ML benches vs PyTorch MPS (~7 min: twenty
#                            workloads, five runs a side, and a PyTorch process
#                            start for each -- it reports progress on stderr)
#                           (LEVEL=easy|medium|hard, default hard)
#   make fixpoint-check  compile each source twice and compare the images
#   make install    install to $(PREFIX)
#   make clean      remove build artifacts

TARGET      := jaithon
# Every generated path hangs off BUILD_ROOT, and TARGET names the binary, so a
# second build can be given a tree of its own:
#
#   make debug BUILD_ROOT=build-lambda TARGET=jaithon-lambda
#
# Two `make` invocations sharing one build directory are not safe against each
# other: they write the same object files and the same ./jaithon. That has twice
# produced measurements of a binary nobody built, including a 2.9x "regression"
# that did not exist. Override both when running builds concurrently.
BUILD_ROOT  ?= build
BUILD       := $(BUILD_ROOT)
PREFIX      ?= /usr/local

CC          ?= cc
UNAME_S     := $(shell uname -s)

WARNINGS    := -Wall -Wextra -Wshadow -Wstrict-prototypes -Wmissing-prototypes \
               -Wpointer-arith -Wcast-align -Wwrite-strings -Wno-unused-parameter
STD         := -std=c11
BASE_CFLAGS := $(STD) $(WARNINGS) -I. -Isrc -I$(BUILD_ROOT) -MMD -MP -fno-common

# -flto because the interpreter's hot helpers live in other translation units:
# getPropertyInto, bindCallArgs and jaiStringEquals are all called from runLoop
# in vm.c and defined elsewhere. Measured interleaved, LTO is worth 4.0% on
# `check lib/std`, and it costs 6s of build time (2s -> 8s).
#
# It also makes hand-inlining unnecessary: with LTO on, forcing jaiStringEquals
# out of line with __attribute__((noinline)) measures 0.0% difference, because
# LTO inlines it across the TU boundary anyway.
RELEASE_CFLAGS := -O2 -DNDEBUG -fno-strict-aliasing -flto
DEBUG_CFLAGS   := -O0 -g3 -DJAI_DEBUG -fno-omit-frame-pointer

LDFLAGS  :=
LIBS     := -lm -lpthread

# Append-only hooks for one-off flags. Setting CFLAGS or LDFLAGS on the command
# line REPLACES them, and makefile `+=` cannot append to a command-line
# variable, so `make LDFLAGS=-flto` silently drops the readline probe's
# `-L/opt/homebrew/opt/readline/lib` and the link then fails on
# _rl_replace_line. Use these instead:
#
#   make EXTRA_CFLAGS=-flto EXTRA_LDFLAGS=-flto
#
# They are part of CC_ID and LINK_ID below, so changing them invalidates the
# build tree exactly as changing the real flags would.
EXTRA_CFLAGS  ?=
# Collector cadence for `make gc-stress-test`. See that target for the numbers.
GC_STRESS_EVERY ?= 5000
# jaicv's own cadence, ten times coarser. Its tests train seven models, denoise,
# and run two optical flows, so they allocate far harder than the language
# suites: at N=5000 the two files take 97s against 15s at N=50000, and 15s is
# what fits in a gate that finishes in two minutes. `test_against_opencv.jai` is
# left out of the stress leg entirely -- 704 recorded cases replayed under a
# collector is four and a half minutes on its own, and every op it exercises is
# reached by test_reachable.jai as well.
JAICV_GC_STRESS_EVERY ?= 50000
# Loop repetitions before `make kind-fuzz` changes a kind. Must exceed the
# tier's hotness threshold or nothing under test ever compiles.
KIND_FUZZ_WARM ?= 3000
# `make jit-fuzz`: how many random programs, and how many calls each generated
# function gets before the program ends. FUZZ_WARM must clear JAI_JIT_THRESHOLD
# (64, and not settable from the environment) with room to spare, or the tier
# under test never compiles and every program passes on no coverage at all.
FUZZ_COUNT ?= 400
FUZZ_WARM  ?= 1500
# How many generated `match` bodies `make match-fuzz` runs through its six
# configurations. Four to a program, so this is fifty programs at the default.
MATCH_FUZZ_COUNT ?= 200
# How many generated programs `make flow-fuzz` runs. It varies the SHAPE --
# try/catch, defer, labelled break, closures outliving their capture -- where
# the other fuzzers vary a kind, so it is one program per count, not four.
FLOW_FUZZ_COUNT ?= 200
EXTRA_LDFLAGS ?=

# zlib inflates the seed's images. boot/seed.bin holds them deflated, one
# stream per module, and boot/seed_blob.S pulls that file into the binary with
# .incbin; boot/seed.c is only the index and the decoder. zlib is the one
# library assumed present beyond libc -- it ships with macOS and with every
# Linux distribution -- and nothing else links it, so dropping the compression
# again is this line plus the generator.
LIBS     += -lz

ifeq ($(UNAME_S),Darwin)
  LIBS   += -framework Cocoa -framework Metal -framework QuartzCore \
            -framework MetalKit -framework Foundation \
            -framework MetalPerformanceShaders \
            -framework MetalPerformanceShadersGraph \
            -framework AVFoundation -framework CoreMedia -framework CoreVideo \
            -framework CoreML
endif

# --- readline probe ---------------------------------------------------------
#
# JAI_HAVE_READLINE guards more than line editing and completion. The Ctrl-C
# prompt handler in src/cli/repl.c is inside it too, and without that
# takePromptInterrupt() is a constant false, so the fgets path has no way at
# all to abandon a half typed statement: a line the REPL wrongly reads as a
# continuation becomes a prompt that cannot be interrupted. Answering this
# question wrongly costs a working REPL, not a convenience.
#
# It was answered wrongly twice, in ways no file test can catch:
#
#   - the whole probe sat inside the Darwin branch above, so NO Linux build
#     ever defined JAI_HAVE_READLINE and no Linux REPL could be interrupted;
#   - it looked for libreadline.a while the link asks for -lreadline, which
#     resolves to the shared library, so a shared-only install -- the normal
#     one everywhere except Homebrew -- was rejected while being usable.
#
# So compile and link the entry points repl.c actually calls, with the flags
# the real link will use. That is the only check that cannot disagree with the
# link line, and it settles libedit for free: several systems install editline
# under readline's own name and header, and the ones that matter here are the
# ones with no rl_replace_line and no rl_done, which this REPL needs. A library
# that cannot build the probe is the wrong library and the next candidate is
# tried; when none of them work the REPL falls back to fgets, which is what a
# machine with no readline at all has always got.
#
# Cost is one small compile each time this file is read, so two for a `make
# debug`, which re-enters make: about 50ms when the first candidate answers and
# about 130ms in the worst case, where every candidate fails. A READLINE_FLAGS
# given on the command line overrides the assignment below, which means make
# never runs the probe at all -- so that is both the escape hatch for a link
# the list does not cover (READLINE_FLAGS='-lreadline -ltermcap') and the way
# to build without readline deliberately (READLINE_FLAGS=).
#
# make eats an unescaped '#', so the probe's own include lines are built from
# this rather than written literally.
HASH := \#
READLINE_PROBE := $(HASH)include <stdio.h>\n$(HASH)include <stdlib.h>\n \
  $(HASH)include <readline/readline.h>\n$(HASH)include <readline/history.h>\n \
  static char **complete(const char *text, int start, int end) {\n \
  (void)start; (void)end;\n rl_attempted_completion_over = 1;\n \
  return rl_completion_matches(text, 0);\n}\n \
  int main(void) {\n rl_readline_name = "jaithon";\n \
  rl_attempted_completion_function = complete;\n \
  using_history();\n stifle_history(1);\n \
  (void)read_history("");\n (void)write_history("");\n \
  (void)history_truncate_file("", 1);\n \
  rl_free_line_state();\n (void)rl_replace_line("", 0);\n rl_done = 1;\n \
  (void)rl_line_buffer;\n \
  char *line = readline("");\n add_history(line);\n free(line);\n \
  return 0;\n}\n

# Homebrew keeps readline out of the default search path on purpose, because
# macOS ships editline under the same name, so those two prefixes have to be
# named. They are tried first: a Mac that has both keeps linking GNU readline,
# which is what it has always done. Everything after them is a plain link
# against whatever is already on the compiler's own search path, which is the
# Linux case, then the two curses splits that a readline with no recorded
# dependency needs, then editline under its own name for the BSDs.
READLINE_PREFIXES := $(wildcard /opt/homebrew/opt/readline /usr/local/opt/readline)
READLINE_TRY := $(foreach p,$(READLINE_PREFIXES),'-I$(p)/include -L$(p)/lib -lreadline') \
                '-lreadline' '-lreadline -lncurses' '-lreadline -ltinfo' '-ledit'

READLINE_FLAGS := $(shell \
  dir=`mktemp -d 2>/dev/null` || exit 0; \
  printf '%b' '$(READLINE_PROBE)' >$$dir/probe.c; \
  for flags in $(READLINE_TRY); do \
    if $(CC) -o $$dir/probe $$dir/probe.c $$flags >/dev/null 2>&1; then \
      printf '%s' "$$flags"; break; \
    fi; \
  done; \
  rm -rf $$dir)

# Split by flag shape rather than by position, so a candidate can be written
# above as the one link line it is. Each kind then lands in the variable this
# build already puts that kind in, which leaves CFLAGS, LDFLAGS and LIBS on a
# Mac byte for byte what they were when the paths were hardcoded: had the order
# shifted, CC_ID and LINK_ID would differ and the stamp check further down
# would throw away every existing build tree on the first make after this.
ifneq ($(READLINE_FLAGS),)
  BASE_CFLAGS += -DJAI_HAVE_READLINE $(filter -I%,$(READLINE_FLAGS))
  LDFLAGS     += $(filter -L%,$(READLINE_FLAGS))
  LIBS        += $(filter-out -I% -L%,$(READLINE_FLAGS))
endif

ifeq ($(BUILD_TYPE),debug)
  CFLAGS     := $(BASE_CFLAGS) $(DEBUG_CFLAGS)
  BUILD      := $(BUILD_ROOT)/debug
  BUILD_NAME := debug
else
  CFLAGS     := $(BASE_CFLAGS) $(RELEASE_CFLAGS)
  BUILD      := $(BUILD_ROOT)/release
  BUILD_NAME := release
  # Matches the -flto in RELEASE_CFLAGS; the link needs it too, and a debug
  # link with -flto over non-LTO objects would only cost time.
  LDFLAGS    += -flto
endif

# boot/seed.c is generated (scripts/dev/gen_seed.py, `make reseed`) and holds the
# .jaic images the self-hosted front end needs before it can compile anything.
# A wildcard rather than a literal so a tree without a seed still builds -- that
# build simply needs a working front end on disk to start.
SRCS_S  := $(wildcard boot/*.S)
SRCS_C  := $(wildcard boot/*.c) \
           $(wildcard src/common/*.c) \
           $(wildcard src/vm/*.c src/vm/*/*.c) \
           $(wildcard src/runtime/*.c src/runtime/*/*.c src/runtime/*/*/*.c) \
           $(wildcard src/native/*.c src/native/*/*.c) \
           $(wildcard src/cli/*.c src/cli/*/*.c)

ifeq ($(UNAME_S),Darwin)
  SRCS_M := $(wildcard src/native/apple/*.m)
else
  SRCS_M :=
endif

OBJS := $(patsubst %.c,$(BUILD)/%.o,$(SRCS_C)) \
        $(patsubst %.S,$(BUILD)/%.o,$(SRCS_S)) \
        $(patsubst %.m,$(BUILD)/%.o,$(SRCS_M))
# verify_chunk.c is compiled with the same -MMD -MP, so its .d belongs here too
# or a header change would relink it against stale knowledge of its own deps.
DEPS := $(OBJS:.o=.d) $(BUILD)/verify_chunk.d

# --- automatic .jaic cache key ----------------------------------------------
#
# A .jaic is validated against JAI_COMPILER_VERSION, which is a hand-maintained
# constant in common.h. Every time someone changed what codegen emits and did
# not remember to bump it, a newer compiler replayed bytecode written by an
# older one — silently, and with no symptom except that a fix "did not work".
# That cost four separate false diagnoses in this project's history.
#
# So key the cache on the compiler's own sources as well. Any edit to a .c, .h
# or .m under src/ changes the fingerprint and invalidates every cached image.
# The generated header is rewritten only when the value actually changes, so an
# unrelated rebuild does not cascade: the serialize*.c files are the only
# translation units that include it.
# boot/seed.c is excluded on purpose. ALL_SRCS feeds SRC_FINGERPRINT, which
# becomes JAI_BUILD_ID, which every .jaic records and demands back. The seed is
# generated data rather than compiler logic -- it changes what the compiler
# STARTS with, never what it does with a given source -- so including it made
# the seed part of its own identity: reseeding changed seed.c, which changed the
# fingerprint, which changed the build id stamped into the next seed's images.
# That loop cannot converge, and a three-stage fixpoint over it never holds.
#
# Narrowed again, to the files that define the BYTECODE CONTRACT rather than to
# the VM as a whole. The id answers exactly one question -- can this binary load
# this bytecode -- and that is a fact about the opcode set, the object model and
# the .jaic format. It is not a fact about how the interpreter executes an
# opcode.
#
# Hashing all of src/vm made the VM unmodifiable. Editing vm.c changed the id,
# which invalidated every seeded image, and regenerating the seed needs a
# working compiler, which needs the seed: a loop with no way in. Every
# optimisation to the interpreter loop -- which is most VM work -- hit it.
#
# So vm.c, gc.c, table.c and the rest are deliberately NOT here: changing how an
# opcode is executed cannot make existing bytecode unloadable. Changing what
# opcodes exist can, and chunk.h is where that is written down.
#
# WHAT THIS ID DOES AND DOES NOT GATE. It gates __jaicache__ only. The seed is
# loaded with `fromSeed` set, and serialize.c skips the build-id check on that
# path, so a change here invalidates the dev cache and leaves the seed loadable.
# That is what makes adding an opcode possible at all: a new instruction cannot
# make existing bytecode unreadable, because existing bytecode does not use it,
# and the seed has to keep working or there is no compiler left to reseed with.
# Measured: appending to chunk.h moves the id and a wiped cache still boots.
#
# The hole is a change to what an existing opcode *means*. Nothing here catches
# that -- not the id, since the seed ignores it, and not the fingerprint, if the
# table itself is untouched. Bump JAIC_VERSION by hand when you redefine an
# instruction; that is checked, and it is the only thing that is.
#
# boot/seed.c is excluded for a different reason and would be even if it were a
# contract file: it is generated data rather than logic -- it changes what the
# compiler STARTS with, never what it does with a given source -- so including
# it made the seed part of its own identity. Reseeding changed seed.c, which
# changed the fingerprint, which changed the id stamped into the next seed's
# images. That loop cannot converge and a fixpoint over it never holds.
ALL_SRCS := src/vm/bytecode/chunk.h src/vm/bytecode/serialize.h \
            src/vm/bytecode/serialize_internal.h \
            src/vm/bytecode/serialize.c src/vm/bytecode/serialize_read.c \
            src/vm/bytecode/serialize_write.c \
            src/vm/object/object.h src/vm/value.h
SRC_FINGERPRINT := $(shell cat $(ALL_SRCS) 2>/dev/null | shasum -a 256 | cut -c1-8)
# At BUILD_ROOT, not $(BUILD): the fingerprint does not depend on build type,
# and BASE_CFLAGS is expanded before $(BUILD) is narrowed to debug/release.
BUILD_ID_H := $(BUILD_ROOT)/jai_build_id.h

.PHONY: $(BUILD_ID_H).check
$(BUILD_ID_H).check:
	@mkdir -p $(dir $(BUILD_ID_H))
	@printf '/* generated by the Makefile; see the .jaic cache key note */\n#define JAI_BUILD_ID 0x%su\n' '$(SRC_FINGERPRINT)' >$(BUILD_ID_H).tmp
	@cmp -s $(BUILD_ID_H).tmp $(BUILD_ID_H) 2>/dev/null || mv $(BUILD_ID_H).tmp $(BUILD_ID_H)
	@rm -f $(BUILD_ID_H).tmp

# A real file target, not phony: its recipe only runs to satisfy the phony
# .check prerequisite above and never touches the file, so serialize.o only
# rebuilds when the fingerprint actually changed, not on every invocation.
$(BUILD_ID_H): $(BUILD_ID_H).check
	@:

$(BUILD)/src/vm/bytecode/serialize.o: $(BUILD_ID_H)
$(BUILD)/src/vm/bytecode/serialize_read.o: $(BUILD_ID_H)
$(BUILD)/src/vm/bytecode/serialize_write.o: $(BUILD_ID_H)

# --- rebuild stamps ---------------------------------------------------------
#
# Both build types link the same ./jaithon but compile into separate object
# trees, so after `make release` the binary is newer than every object under
# build/debug and `make debug` used to be a silent no-op: the developer went on
# testing the RELEASE binary, with asserts and chunk verification compiled out.
# Two stamps record what the existing output was built from:
#
#   build/.link-id     what ./jaithon was last linked from. Changing build type
#                      (or CC, LDFLAGS, LIBS) invalidates it.
#   $(BUILD)/.cc-id    the compile line for THIS object tree. Changing CFLAGS
#                      invalidates that tree only; switching build type leaves
#                      both trees alone, so alternating debug and release
#                      reuses the objects it already has and pays for the link.
#
# The comparison happens at parse time and *deletes* what it invalidates. Two
# other designs were tried first and both are wrong here:
#
#   - a phony marker among ./jaithon's prerequisites relinks on every make,
#     including a null one, and the link is the slowest step;
#   - a stamp rule that rewrites the stamp so it is "newer" than the binary
#     fails twice over on this machine -- Apple's make is 3.81 with
#     whole-second timestamps, so a stamp written in the same second compares
#     equal, and 3.81 also samples a target's mtime before running its
#     prerequisites, so a rule that deletes the binary is not noticed either.
#
# Deleting the stale output while the makefile is still being read is immune to
# both, and to any clock. The price is that a build which then fails to compile
# leaves no ./jaithon at all instead of the previous type's binary -- which is
# the right trade: no binary is honest, and the wrong binary is the bug.
LINK_STAMP := $(BUILD_ROOT)/.link-id
CC_STAMP   := $(BUILD)/.cc-id
LINK_ID    := $(BUILD_NAME) | $(CC) | $(LDFLAGS) | $(EXTRA_LDFLAGS) | $(LIBS)
CC_ID      := $(CC) | $(CFLAGS) | $(EXTRA_CFLAGS)

# Those checks throw away build products, so they stay out of goals that do not
# build anything themselves. An empty goal list means the default goal, which
# does. `debug` and `release` only re-enter make with BUILD_TYPE set, and this
# outer pass does not know which type is wanted -- letting it judge would delete
# the binary the inner pass is about to decide is current.
BUILD_GOAL := $(filter-out clean distclean help uninstall debug release,\
                           $(or $(MAKECMDGOALS),all))

ifneq ($(BUILD_GOAL),)
  DROP_STALE_LINK := $(shell [ "$$(cat $(LINK_STAMP) 2>/dev/null)" = '$(LINK_ID)' ] \
                             || rm -f $(TARGET))
  DROP_STALE_OBJS := $(shell [ "$$(cat $(CC_STAMP) 2>/dev/null)" = '$(CC_ID)' ] \
                             || rm -rf $(BUILD))
endif

.PHONY: all debug release test verify-test chunk-cfg-test bench jaitensor jaicv jainum jaiframe exports-check import-check install uninstall \
        clean distclean fmt fmt-check fmt-roundtrip check help

# `all` must be the default goal: the recursive release and debug targets below
# invoke make with no goal, so whatever make picks is what a bare `make`, a
# `make debug` and a `make release` all build. Saying it outright rather than
# relying on "first explicit rule in the file", which is not a property anyone
# can see while editing: the generated-header rule above is a real rule with a
# path for a target, so moving it up here silently makes *it* the default and
# every build becomes a no-op that prints nothing and exits 0 — a green build
# of the previous binary. That happened.
.DEFAULT_GOAL := all
all: $(TARGET)

release:
	@$(MAKE) --no-print-directory BUILD_TYPE=release

debug:
	@$(MAKE) --no-print-directory BUILD_TYPE=debug

# The parse-time check above removed this file whenever CFLAGS changed, so a
# plain "make it if it is missing" rule is all that is needed. Order-only
# prerequisite of every object, so it exists before the first compile and is
# written exactly once even under -j.
$(CC_STAMP):
	@mkdir -p $(dir $@)
	@printf '%s' '$(CC_ID)' >$@

# The link stamp is written by the link itself: a failed link must leave the
# old identity recorded, so the next make retries instead of believing it.
$(TARGET): $(OBJS)
	@echo "  LINK    $@ ($(BUILD_NAME))"
	@$(CC) $(LDFLAGS) $(EXTRA_LDFLAGS) -o $@ $(OBJS) $(LIBS)
	@mkdir -p $(dir $(LINK_STAMP))
	@printf '%s' '$(LINK_ID)' >$(LINK_STAMP)

$(BUILD)/%.o: %.c | $(CC_STAMP)
	@mkdir -p $(dir $@)
	@echo "  CC      $<"
	@$(CC) $(CFLAGS) $(EXTRA_CFLAGS) -c $< -o $@

# boot/seed_blob.S is `.S` so cpp runs before the assembler; it .incbin's
# boot/seed.bin, which make cannot see as a dependency on its own.
$(BUILD)/boot/seed_blob.o: boot/seed.bin

$(BUILD)/%.o: %.S | $(CC_STAMP)
	@mkdir -p $(dir $@)
	@echo "  AS      $<"
	@$(CC) $(CFLAGS) $(EXTRA_CFLAGS) -c $< -o $@

$(BUILD)/%.o: %.m | $(CC_STAMP)
	@mkdir -p $(dir $@)
	@echo "  OBJC    $<"
	@$(CC) $(CFLAGS) $(EXTRA_CFLAGS) -fobjc-arc -c $< -o $@

-include $(DEPS)

# run_tests.sh runs the verifier itself, as the first of its four layers, so
# that the run ends in one summary rather than one per layer.
test: package-check opcode-check exports-check layer-check kind-tag-check switch-doc-check seed-closure-check jaic-wire-check field-kind-check linkage-check import-check jit-fusion-check branch-table-check method-surface-check check $(TARGET) $(BUILD)/verify_chunk $(BUILD)/chunk_cfg $(BUILD)/crc32_equiv $(BUILD)/chunk_caches $(BUILD)/linetable_ltv1 $(BUILD)/jit_arena $(BUILD)/jit_arm64 $(BUILD)/field_natives $(BUILD)/invoke_result_kind
	@$(BUILD)/chunk_cfg
	@$(BUILD)/crc32_equiv
	@$(BUILD)/chunk_caches
	@$(BUILD)/linetable_ltv1
	@$(BUILD)/jit_arena
	@$(BUILD)/jit_arm64
	@$(BUILD)/field_natives
	@$(BUILD)/invoke_result_kind
	@./scripts/run_tests.sh
	@./scripts/bench/bench_smoke.sh
	@$(MAKE) --no-print-directory fmt-roundtrip
	@$(MAKE) --no-print-directory gc-stress-test

# The unit suites under a collector that runs every N allocations.
#
# Only the GOLDENS ever got a --gc-stress pass (run_tests.sh runs each one a
# second time under it). The 1000-odd unit tests in tests/lang, tests/stdlib,
# tests/checker and the package trees ran once, plain, so no gate covered them under a collector
# at all -- and a use-after-free or a missed root is exactly what gc stress is
# for. Two silent miscompiles were found on 2026-08-12 by stress modes, neither
# by any other gate.
#
# It was not covered because --gc-stress collects on EVERY allocation, and that
# is quadratic: tests/lang alone runs in 1.09s plain and did not finish in ten
# minutes under it. --gc-stress=N is what makes it affordable. Measured on
# tests/lang: N=500 556s, N=5000 9.6s, N=20000 4.4s. N=5000 is the operating
# point -- under 10x the plain cost, and still collecting at thousands of points
# where the live-bytes threshold collects at tens.
.PHONY: gc-stress-test
gc-stress-test: $(TARGET)
# ONE INVOCATION PER DIRECTORY, not one for all five.
#
# The collector walks the live set, and running the suites together keeps every
# module of every suite alive for the whole run -- so each of the thousands of
# collections walks five suites' worth of heap instead of one. The work is the
# product of the two, and splitting it turns a product back into a sum:
# measured on an idle machine, 463s together against 144s apart, for the same
# tests under the same cadence.
	@for suite in tests/lang tests/stdlib tests/checker \
	              packages/jaiplot/tests packages/jaitensor/tests \
	              packages/jailearn/tests; do \
	    out=$$(JAITHON_PATH=$(CURDIR)/lib ./$(TARGET) test \
	             --gc-stress=$(GC_STRESS_EVERY) "$$suite" 2>&1); \
	    status=$$?; \
	    if [ $$status -ne 0 ]; then printf '%s\n' "$$out"; exit $$status; fi; \
	    printf '%s  %s\n' "$$(printf '%s\n' "$$out" | tail -1)" "$$suite"; \
	  done
	@out=$$(JAITHON_PATH=$(CURDIR)/lib ./$(TARGET) test \
	          --gc-stress=$(JAICV_GC_STRESS_EVERY) \
	          packages/jaicv/tests/test_reachable.jai \
	          packages/jaicv/tests/test_jaicv.jai 2>&1); \
	  status=$$?; \
	  if [ $$status -ne 0 ]; then printf '%s\n' "$$out"; exit $$status; fi; \
	  printf '%s\n' "$$out" | tail -1


# Split out by subject. Make reads every rule before it runs any, so the
# order here is for the reader, not the build.
include mk/gates.mk
include mk/jitdev.mk
include mk/unit.mk
include mk/bench.mk
include mk/seed.mk

#: A prerequisite of `test` since 2026-08-24. It was a target nobody ran, and
#: it was red: twenty-three private-access errors in one jaiframe file, plus —
#: once `check` learned to resolve imports at all — a jainum function calling
#: one that did not exist. Both classes of bug are invisible to `make test`,
#: which only runs what the tests reach. Costs about 1m45 over 422 files.
check: package-check $(TARGET)
	@./$(TARGET) check lib tests/lang tests/stdlib tests/checker tests/bench examples packages

fmt: $(TARGET)
	@./$(TARGET) fmt lib tests examples packages

# The CI gate behind `fmt`: writes nothing, names the first line of every file
# that is not canonical, and exits nonzero if there is one. Kept out of `test`
# because it re-parses the whole tree and takes several times as long as the
# rest of the suite put together; `run_tests.sh --format` runs it inline.
fmt-check: $(TARGET)
	@./$(TARGET) fmt --check lib tests examples packages

# Whether printing a file changes the PROGRAM, which is the question
# `fmt --check` does not ask: a file can be canonical and still have had
# something removed on the way there. Parses every source in the tree, formats
# it, parses it again, and compares the trees. It exists because the printer
# shipped two silent losses -- it never wrote decorators, so every format
# deleted `@trace` and `@gpu_kernel` and turned jaitensor's hot paths back into
# ordinary functions, and it indented the inside of triple-quoted strings, so a
# formatted program's data grew four characters a line.
#
# The comparison is only as good as what `ast_encode` prints: a field it leaves
# out is a field this cannot see. That is why decorators had to be added there
# before this gate could catch them.
# Every benchmark, run once at the smallest level. See the script for why it
# is here: they are programs no gate ran, and one of them was broken.
.PHONY: bench-smoke
bench-smoke: $(TARGET)
	@./scripts/bench/bench_smoke.sh

.PHONY: fmt-roundtrip
fmt-roundtrip: $(TARGET)
	@JAITHON_PATH=$(CURDIR)/lib ./$(TARGET) run scripts/gate/fmt_roundtrip.jai

install: package-check $(TARGET)
	@install -d $(DESTDIR)$(PREFIX)/bin
	@install -m 755 $(TARGET) $(DESTDIR)$(PREFIX)/bin/$(TARGET)
	@install -d $(DESTDIR)$(PREFIX)/share/jaithon
	@cp -R lib $(DESTDIR)$(PREFIX)/share/jaithon/
	@cp -R packages $(DESTDIR)$(PREFIX)/share/jaithon/
	@echo "installed to $(DESTDIR)$(PREFIX)"

uninstall:
	@rm -f $(DESTDIR)$(PREFIX)/bin/$(TARGET)
	@rm -rf $(DESTDIR)$(PREFIX)/share/jaithon

# `build` holds both object trees, both stamps and verify_chunk, so removing it
# is the whole build state. The debug link leaves a .dSYM behind once anything
# has run dsymutil over it, and it is stale the moment the binary is relinked.
clean:
	@rm -rf $(BUILD_ROOT) $(TARGET) $(TARGET).dSYM
	@find . -name '__jaicache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true

# `clean` removes the tree THIS invocation would build, which is all it can know
# about: BUILD_ROOT and TARGET are overridable, so a concurrent build puts its
# objects and its binary somewhere `clean` will never look. Those accumulate at
# the repository root and nothing else knows their names, so `distclean` takes
# the whole family plus the other generated artefacts: the API pages `jaithon
# doc` writes, and the images the raytracer example leaves behind.
#
# Everything removed here is either regenerated by a build or ignored by git;
# the patterns are checked against `git ls-files` and match nothing tracked.
distclean: clean
	@rm -rf build build-* jaithon jaithon-* *.dSYM
	@rm -rf docs/api
	@rm -f *.ppm .DS_Store
	@echo "  CLEAN   removed every build tree, binary and generated artefact"

help:
	@echo "targets: all debug release test bench [jaitensor|jaicv] package-check fixpoint-check reseed check fmt install clean"
