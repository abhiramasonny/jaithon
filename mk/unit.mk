# The C unit tests: one binary each, linking the VM's own objects against a
# single tests/vm/*.c.
#
# They exist for the subsystems that cannot be reached from the language --
# malformed bytecode the verifier must reject, the two CRC32 implementations
# that must agree, arena and encoder behaviour with no program in sight.

# The chunk verifier is C-only: it has to be fed malformed bytecode, which no
# .jai source can express. Everything but the CLI entry point links in.
VERIFY_OBJS := $(filter-out $(BUILD)/src/cli/main.o,$(OBJS))

$(BUILD)/verify_chunk: $(VERIFY_OBJS) tests/vm/verify_chunk.c | $(CC_STAMP)
	@echo "  CC      tests/vm/verify_chunk.c"
	@$(CC) $(CFLAGS) $(LDFLAGS) -o $@ tests/vm/verify_chunk.c \
	    $(VERIFY_OBJS) $(LIBS)

# The verifier on its own, with its own report, for working on it.
verify-test: $(BUILD)/verify_chunk
	@$(BUILD)/verify_chunk

# The chunk CFG, over every function the tree can compile. C rather than .jai
# for the same reason as its neighbour above: JaiChunkCfg reaches no program's
# output at all, and the corpus is the test -- a pass with no consumer can only
# be wrong quietly. Run by `make chunk-cfg-test` (mk/gates.mk).
$(BUILD)/chunk_cfg: $(VERIFY_OBJS) tests/vm/chunk_cfg.c | $(CC_STAMP)
	@echo "  CC      tests/vm/chunk_cfg.c"
	@$(CC) $(CFLAGS) $(LDFLAGS) -o $@ tests/vm/chunk_cfg.c \
	    $(VERIFY_OBJS) $(LIBS)

# The two CRC32 implementations must be one function. Every .jaic carries a CRC
# written by the table path, so a divergence rejects every cached image in the
# tree -- correct output, 100x the load time, and no error anywhere.
$(BUILD)/crc32_equiv: $(VERIFY_OBJS) tests/vm/crc32_equiv.c | $(CC_STAMP)
	@echo "  CC      tests/vm/crc32_equiv.c"
	@$(CC) $(CFLAGS) $(LDFLAGS) -o $@ tests/vm/crc32_equiv.c \
	    $(VERIFY_OBJS) $(LIBS)

crc-test: $(BUILD)/crc32_equiv
	@$(BUILD)/crc32_equiv

# Deserialised cache arrays must be sized exactly: the grow-by-doubling path
# left 1.19 MB of slack across the seed's images.
$(BUILD)/chunk_caches: $(VERIFY_OBJS) tests/vm/chunk_caches.c | $(CC_STAMP)
	@echo "  CC      tests/vm/chunk_caches.c"
	@$(CC) $(CFLAGS) $(LDFLAGS) -o $@ tests/vm/chunk_caches.c \
	    $(VERIFY_OBJS) $(LIBS)

chunk-caches-test: $(BUILD)/chunk_caches
	@$(BUILD)/chunk_caches

# The line table's encoding, at the edges a corpus does not reliably contain:
# backwards span deltas, zero-length spans, u32 extremes, truncated streams.
$(BUILD)/linetable_ltv1: $(VERIFY_OBJS) tests/vm/linetable_ltv1.c | $(CC_STAMP)
	@echo "  CC      tests/vm/linetable_ltv1.c"
	@$(CC) $(CFLAGS) $(LDFLAGS) -o $@ tests/vm/linetable_ltv1.c \
	    $(VERIFY_OBJS) $(LIBS)

linetable-test: $(BUILD)/linetable_ltv1
	@$(BUILD)/linetable_ltv1

# The code arena executes what it was written. C rather than .jai because
# nothing in the language reaches it yet, and because the failure it guards --
# a stale instruction cache on arm64 -- returns a plausible wrong number rather
# than crashing.
$(BUILD)/jit_arena: tests/vm/jit_arena.c src/vm/jit/jit_arena.c | $(CC_STAMP)
	@echo "  CC      $<"
	@$(CC) $(CFLAGS) -o $@ tests/vm/jit_arena.c src/vm/jit/jit_arena.c

# The arm64 encoders, each verified by executing the instruction it builds.
$(BUILD)/jit_arm64: tests/vm/jit_arm64.c src/vm/jit/jit_arm64.c src/vm/jit/jit_arena.c | $(CC_STAMP)
	@echo "  CC      $<"
	@$(CC) $(CFLAGS) -o $@ tests/vm/jit_arm64.c src/vm/jit/jit_arm64.c src/vm/jit/jit_arena.c

# The compiled tier replaces a call to one of these builtins with a single load
# from the receiver (src/vm/jit/jit_field_read.h). Nothing in C connects that
# byte offset to the native it stands in for, or the declared result kind to
# what the native returns -- roadmap.md §6 records "a builtin named len returns
# an int" as an invariant that was true everywhere and checked nowhere. This
# calls every one of them for real and compares. C rather than .jai because a
# .jai test cannot see a struct offset, and would silently test the interpreter
# whenever the loop it warms declines.
$(BUILD)/field_natives: $(VERIFY_OBJS) tests/vm/field_natives.c | $(CC_STAMP)
	@echo "  CC      tests/vm/field_natives.c"
	@$(CC) $(CFLAGS) $(LDFLAGS) -o $@ tests/vm/field_natives.c \
	    $(VERIFY_OBJS) $(LIBS)

.PHONY: field-natives-test
field-natives-test: $(BUILD)/field_natives
	@$(BUILD)/field_natives

# What an OP_INVOKE site records about its own result. C rather than .jai
# because InlineCache::resultKind reaches no program's output, and a test that
# could only see it through the compiled tier would be testing the tier.
$(BUILD)/invoke_result_kind: $(VERIFY_OBJS) tests/vm/invoke_result_kind.c | $(CC_STAMP)
	@echo "  CC      tests/vm/invoke_result_kind.c"
	@$(CC) $(CFLAGS) $(LDFLAGS) -o $@ tests/vm/invoke_result_kind.c \
	    $(VERIFY_OBJS) $(LIBS)

.PHONY: invoke-result-test
invoke-result-test: $(BUILD)/invoke_result_kind
	@$(BUILD)/invoke_result_kind

.PHONY: jit-test
jit-test: $(BUILD)/jit_arena $(BUILD)/jit_arm64
	@$(BUILD)/jit_arena
	@$(BUILD)/jit_arm64
