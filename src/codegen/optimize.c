/* optimize.c — the bytecode optimisation passes and the chunk verifier.
 *
 * Normative reference: spec/BYTECODE.md §3.3 (the fused opcodes) and §8 (the
 * pass list).
 *
 * Every pass here works on a *decoded instruction list*, never on raw bytes.
 * A peephole that rewrites bytes in place cannot fix up the jumps that cross
 * the edit, and that is precisely how these passes normally corrupt a chunk.
 * The shape of every pass is therefore the same:
 *
 *   1. decode the chunk into Instr records, resolving every branch operand to
 *      the *index* of the instruction it lands on (a byte offset that is not
 *      an instruction boundary aborts the pass),
 *   2. mark every instruction that is branched to, is an exception handler, is
 *      an exception-region boundary, or is a default-thunk entry point,
 *   3. transform the list, deleting or rewriting whole instructions,
 *   4. re-encode: lay out new offsets to a fixpoint, recompute every branch
 *      operand from the new layout, and rebuild the line table, the exception
 *      table and the default-thunk offsets against it.
 *
 * Step 4 is what keeps tracebacks and `try` regions pointing at the code they
 * described before the pass ran. If anything in step 4 cannot be represented
 * (a branch that no longer fits in i16), the whole pass is abandoned and the
 * chunk is left exactly as it was — a missed optimisation, never a wrong one.
 */
#include "codegen.h"

#include "../common/common.h"
#include "../vm/chunk.h"
#include "../vm/object.h"

/* ------------------------------------------------------------------ */
/* Instruction list                                                     */
/* ------------------------------------------------------------------ */

/* Widest fixed operand run in the table: OP_MATCH_RANGE at 9 bytes. Only
 * OP_CLOSURE exceeds it, and it is never rewritten, so its operands stay in
 * the original code buffer. */
#define OPT_MAX_FIXED_OPERANDS 12
#define OPT_NO_INDEX (-1)

typedef struct {
    uint8_t        op;
    bool           dead;
    bool           isTarget;      /* branched to, or pinned as a region edge */
    int8_t         branchAt;      /* byte index of the i16 branch operand, -1 */
    int            target;        /* instruction index branched to, or -1 */
    int            operandCount;
    int            origOffset;
    int            newOffset;
    uint32_t       span, spanEnd; /* source span, carried to the new line table */
    const uint8_t *ext;           /* operands still in the original code buffer */
    uint8_t        buf[OPT_MAX_FIXED_OPERANDS];
} Instr;

typedef struct {
    ObjFunction *fn;
    Chunk       *chunk;
    Instr       *data;
    int          count;
    int          capacity;

    int         *offsetIndex;     /* [chunk->count + 1]; OPT_NO_INDEX = not a boundary */
    int          offsetIndexLen;

    /* Side tables recorded as instruction indices so they survive deletion.
     * excCapacity is the allocated length, which stays put even when a rebuild
     * drops entries from the function's table. */
    int         *excStart, *excEnd, *excHandler;
    int          excCount, excCapacity;
    int         *defaults;
    int          defaultCount;
} Code;

static const uint8_t *insOps(const Instr *in) {
    return in->ext != NULL ? in->ext : in->buf;
}

static int insSize(const Instr *in) { return 1 + in->operandCount; }

static void writeU16At(uint8_t *p, uint16_t v) {
    p[0] = (uint8_t)(v & 0xff);
    p[1] = (uint8_t)((v >> 8) & 0xff);
}

static void writeU24At(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)(v & 0xff);
    p[1] = (uint8_t)((v >> 8) & 0xff);
    p[2] = (uint8_t)((v >> 16) & 0xff);
}

/* Byte index of the i16 branch operand inside `op`'s operand run, or -1 when
 * the instruction carries no code address. Keep in sync with chunk.h. */
static int branchOperandAt(uint8_t op) {
    switch (op) {
    case OP_JUMP:
    case OP_JUMP_IF_FALSE:
    case OP_JUMP_IF_TRUE:
    case OP_JUMP_IF_FALSE_KEEP:
    case OP_JUMP_IF_TRUE_KEEP:
    case OP_JUMP_IF_NULL:
    case OP_LOOP:
    case OP_FOR_ITER:
    case OP_FOR_ITER_BIND:
    case OP_PUSH_FINALLY:
    case OP_PUSH_HANDLER:
        return 0;
    case OP_JUMP_IF_CMP_FALSE:
        return 1;
    case OP_JUMP_IF_CMP_LOCAL_K:
        return 6;   /* after the u8 comparison, the u16 slot and the u24 K */
    case OP_MATCH_SEQ:
        return 2;
    case OP_MATCH_CONST:
    case OP_MATCH_TYPE:
    case OP_MATCH_FIELDS:
        return 3;
    case OP_MATCH_RANGE:
        return 7;
    default:
        return -1;
    }
}

/* True when `target` is a control-flow edge taken by the instruction itself.
 * PUSH_HANDLER and PUSH_FINALLY only *register* an address; control does not
 * transfer there at that point, so no stack state flows along the edge. */
static bool opBranchIsEdge(uint8_t op) {
    return branchOperandAt(op) >= 0 && op != OP_PUSH_HANDLER &&
           op != OP_PUSH_FINALLY;
}

static bool opFallsThrough(uint8_t op) {
    switch (op) {
    case OP_JUMP:
    case OP_LOOP:
    case OP_RETURN:
    case OP_RETURN_NULL:
    case OP_THROW:
    case OP_RERAISE:
    case OP_HALT:
        return false;
    default:
        /* OP_TAIL_CALL reuses the frame and normally ends it, but the emitter
         * is free to follow it with a RETURN; keeping the edge is the
         * conservative choice and costs one instruction. */
        return true;
    }
}

/* Operand run of OP_CLOSURE: u24 constant index plus one (u8, u16) descriptor
 * per upvalue of the referenced function. -1 when the constant is missing or
 * is not a function, which means the chunk is malformed. */
static int closureOperandBytes(const Chunk *chunk, int offset) {
    if (offset + 1 + 3 > chunk->count) return -1;
    uint32_t k = jaiReadU24(chunk->code + offset + 1);
    if (k >= (uint32_t)chunk->constants.count) return -1;
    Value v = chunk->constants.data[k];
    if (!IS_FUNCTION(v)) return -1;
    return 3 + 3 * (int)AS_FUNCTION(v)->upvalueCount;
}

static void instrPush(Code *c, const Instr *in) {
    if (c->capacity < c->count + 1) {
        int oldCapacity = c->capacity;
        c->capacity = JAI_GROW_CAP(oldCapacity);
        c->data = JAI_GROW_ARRAY(Instr, c->data, oldCapacity, c->capacity);
    }
    c->data[c->count++] = *in;
}

static void codeFree(Code *c) {
    JAI_FREE_ARRAY(Instr, c->data, c->capacity);
    JAI_FREE_ARRAY(int, c->offsetIndex, c->offsetIndexLen);
    JAI_FREE_ARRAY(int, c->excStart, c->excCapacity);
    JAI_FREE_ARRAY(int, c->excEnd, c->excCapacity);
    JAI_FREE_ARRAY(int, c->excHandler, c->excCapacity);
    JAI_FREE_ARRAY(int, c->defaults, c->defaultCount);
    memset(c, 0, sizeof *c);
}

/* Instruction index for a code offset, or OPT_NO_INDEX when the offset is not
 * an instruction boundary. chunk->count maps to c->count (one past the end). */
static int indexOfOffset(const Code *c, long offset) {
    if (offset < 0 || offset >= c->offsetIndexLen) return OPT_NO_INDEX;
    return c->offsetIndex[offset];
}

/* Decode fn->chunk. Returns false and leaves *c freeable when the chunk is not
 * decodable; every pass then does nothing, and jaiVerifyChunk reports why. */
static bool codeDecode(Code *c, ObjFunction *fn) {
    memset(c, 0, sizeof *c);
    if (fn == NULL) return false;

    c->fn = fn;
    c->chunk = &fn->chunk;
    Chunk *chunk = c->chunk;
    if (chunk->code == NULL || chunk->count <= 0) return false;

    c->offsetIndexLen = chunk->count + 1;
    c->offsetIndex = JAI_ALLOC(int, c->offsetIndexLen);
    for (int i = 0; i < c->offsetIndexLen; i++) c->offsetIndex[i] = OPT_NO_INDEX;

    for (int offset = 0; offset < chunk->count;) {
        uint8_t raw = chunk->code[offset];
        if (raw >= OP_COUNT) return false;

        int operands = jaiOpOperandSize((OpCode)raw);
        if (raw == OP_CLOSURE) operands = closureOperandBytes(chunk, offset);
        if (operands < 0 || offset + 1 + operands > chunk->count) return false;

        Instr in;
        memset(&in, 0, sizeof in);
        in.op = raw;
        in.operandCount = operands;
        in.origOffset = offset;
        in.newOffset = offset;
        in.branchAt = (int8_t)branchOperandAt(raw);
        in.target = OPT_NO_INDEX;
        jaiChunkSpanAt(chunk, offset, &in.span, &in.spanEnd);

        if (operands > 0) {
            if (operands <= OPT_MAX_FIXED_OPERANDS) {
                memcpy(in.buf, chunk->code + offset + 1, (size_t)operands);
            } else {
                /* OP_CLOSURE only. The original buffer stays alive until the
                 * new one has been written, so this pointer is valid for the
                 * whole pass. */
                in.ext = chunk->code + offset + 1;
            }
        }

        c->offsetIndex[offset] = c->count;
        instrPush(c, &in);
        offset += 1 + operands;
    }
    c->offsetIndex[chunk->count] = c->count;

    for (int i = 0; i < c->count; i++) {
        Instr *in = &c->data[i];
        if (in->branchAt < 0) continue;
        int16_t rel = jaiReadI16(insOps(in) + in->branchAt);
        long target = (long)in->origOffset + insSize(in) + rel;
        int idx = indexOfOffset(c, target);
        if (idx == OPT_NO_INDEX || idx >= c->count) return false;
        in->target = idx;
        c->data[idx].isTarget = true;
    }

    /* Exception regions. Marking the start and end instructions as targets is
     * what stops a peephole window from straddling a region edge: a fused
     * instruction that is half inside a protected region cannot be described
     * by (start, end) at all. */
    c->excCount = (int)fn->exceptionCount;
    if (c->excCount > 0) {
        if (fn->exceptions == NULL) {
            c->excCount = 0;
            return false;
        }
        c->excStart = JAI_ALLOC(int, c->excCount);
        c->excEnd = JAI_ALLOC(int, c->excCount);
        c->excHandler = JAI_ALLOC(int, c->excCount);
        c->excCapacity = c->excCount;   /* only now is there anything to free */
        for (int e = 0; e < c->excCount; e++) {
            const ExceptionEntry *entry = &fn->exceptions[e];
            int s = indexOfOffset(c, (long)entry->start);
            int en = indexOfOffset(c, (long)entry->end);
            int h = indexOfOffset(c, (long)entry->handler);
            if (s == OPT_NO_INDEX || en == OPT_NO_INDEX || h == OPT_NO_INDEX ||
                h >= c->count || s > en) {
                return false;
            }
            c->excStart[e] = s;
            c->excEnd[e] = en;
            c->excHandler[e] = h;
            if (s < c->count) c->data[s].isTarget = true;
            if (en < c->count) c->data[en].isTarget = true;
            c->data[h].isTarget = true;
        }
    }

    c->defaultCount = (int)fn->defaultCount;
    if (c->defaultCount > 0 && fn->defaultOffsets != NULL) {
        c->defaults = JAI_ALLOC(int, c->defaultCount);
        for (int d = 0; d < c->defaultCount; d++) {
            int idx = indexOfOffset(c, (long)fn->defaultOffsets[d]);
            if (idx == OPT_NO_INDEX || idx >= c->count) return false;
            c->defaults[d] = idx;
            c->data[idx].isTarget = true;
        }
    } else {
        c->defaultCount = 0;
    }
    return true;
}

/* ------------------------------------------------------------------ */
/* List navigation                                                      */
/* ------------------------------------------------------------------ */

static int nextLive(const Code *c, int i) {
    for (int j = i + 1; j < c->count; j++) {
        if (!c->data[j].dead) return j;
    }
    return OPT_NO_INDEX;
}

static int prevLive(const Code *c, int i) {
    for (int j = i - 1; j >= 0; j--) {
        if (!c->data[j].dead) return j;
    }
    return OPT_NO_INDEX;
}

static int firstLive(const Code *c) {
    for (int j = 0; j < c->count; j++) {
        if (!c->data[j].dead) return j;
    }
    return OPT_NO_INDEX;
}

/* The instruction an address lands on once deleted instructions are gone.
 * Returns c->count for "the end of the code", which is a legal exception
 * region end and nothing else. */
static int liveAtOrAfter(const Code *c, int i) {
    if (i < 0) return c->count;
    for (int j = i; j < c->count; j++) {
        if (!c->data[j].dead) return j;
    }
    return c->count;
}

/* Up to `want` consecutive live instructions starting at `i`. */
static int windowOf(const Code *c, int i, int want, int *out) {
    int found = 0, j = i;
    while (j >= 0 && found < want) {
        out[found++] = j;
        j = nextLive(c, j);
    }
    return found;
}

/* A window may be rewritten when nothing *inside* it is branched to: an edge
 * landing on the first instruction still executes the whole window, so fusing
 * or deleting it as a unit is invisible to that edge. An edge landing further
 * in is not. */
static bool windowClean(const Code *c, const int *idx, int n) {
    for (int k = 1; k < n; k++) {
        if (c->data[idx[k]].isTarget) return false;
    }
    return true;
}

static void setInstr(Instr *in, uint8_t op, const uint8_t *operands, int n) {
    in->op = op;
    in->operandCount = n;
    in->ext = NULL;
    in->branchAt = (int8_t)branchOperandAt(op);
    if (n > 0) memcpy(in->buf, operands, (size_t)n);
    if (in->branchAt < 0) in->target = OPT_NO_INDEX;
}

static void killInstr(Code *c, int i) { c->data[i].dead = true; }

/* ------------------------------------------------------------------ */
/* Re-encoding                                                          */
/* ------------------------------------------------------------------ */

static uint32_t offsetOfIndex(const Code *c, int idx, int endOffset) {
    if (idx >= c->count) return (uint32_t)endOffset;
    return (uint32_t)c->data[idx].newOffset;
}

/* Re-encode the list into the chunk and rebuild every table that names a code
 * offset. Returns false without touching the chunk if the result cannot be
 * encoded, in which case the caller must report zero changes. */
static bool codeRebuild(Code *c) {
    Chunk *chunk = c->chunk;
    ObjFunction *fn = c->fn;

    int liveCount = 0;
    for (int i = 0; i < c->count; i++) {
        if (!c->data[i].dead) liveCount++;
    }
    if (liveCount == 0) return false;

    /* Lay the instructions out to a fixpoint. Every branch operand in this
     * bytecode is a fixed-width i16, so the second round always agrees with
     * the first; the loop is here so that a narrower encoding added later
     * cannot silently produce stale offsets. */
    int total = 0;
    bool settled = false;
    for (int round = 0; round < 8 && !settled; round++) {
        int offset = 0;
        settled = true;
        for (int i = 0; i < c->count; i++) {
            Instr *in = &c->data[i];
            if (in->newOffset != offset) {
                in->newOffset = offset;
                settled = false;
            }
            if (!in->dead) offset += insSize(in);
        }
        total = offset;
    }
    if (!settled) return false;

    /* Resolve branches before allocating anything, so an unrepresentable jump
     * leaves the chunk untouched.
     *
     * Failing here is a *capacity* verdict, not a diagnosis: hoisting splices
     * instructions into a pre-header, and in a function whose jumps already sit
     * near the i16 limit that is enough to make one not fit. Reporting it would
     * make a program that compiles at -O0 fail at -O2, which is the one thing
     * the pass list may never do. The caller reports zero changes and the chunk
     * stays exactly as it was; a chunk that was malformed before the pass is
     * still caught by jaiVerifyChunk afterwards. */
    for (int i = 0; i < c->count; i++) {
        Instr *in = &c->data[i];
        if (in->dead || in->branchAt < 0) continue;

        int targetIdx = liveAtOrAfter(c, in->target);
        if (targetIdx >= c->count) return false;

        long delta = (long)c->data[targetIdx].newOffset -
                     ((long)in->newOffset + insSize(in));
        if (delta < INT16_MIN || delta > INT16_MAX) return false;
    }

    uint8_t *code = JAI_ALLOC(uint8_t, total);
    LineEntry *lines = JAI_ALLOC(LineEntry, liveCount);
    int lineCount = 0;

    for (int i = 0; i < c->count; i++) {
        Instr *in = &c->data[i];
        if (in->dead) continue;

        uint8_t *at = code + in->newOffset;
        at[0] = in->op;
        if (in->operandCount > 0) {
            memcpy(at + 1, insOps(in), (size_t)in->operandCount);
        }
        if (in->branchAt >= 0) {
            int targetIdx = liveAtOrAfter(c, in->target);
            int delta = c->data[targetIdx].newOffset - (in->newOffset + insSize(in));
            writeU16At(at + 1 + in->branchAt, (uint16_t)(int16_t)delta);
        }

        /* One line entry per run of instructions sharing a span, matching what
         * chunk.c's recordSpan would have produced during emission. */
        if (lineCount == 0 || lines[lineCount - 1].span != in->span ||
            lines[lineCount - 1].spanEnd != in->spanEnd) {
            lines[lineCount].offset = (uint32_t)in->newOffset;
            lines[lineCount].span = in->span;
            lines[lineCount].spanEnd = in->spanEnd;
            lineCount++;
        }
    }

    /* Exception table: a region whose body vanished is dropped, otherwise both
     * edges move to the first surviving instruction at or after them. */
    int kept = 0;
    for (int e = 0; e < c->excCount; e++) {
        int handlerIdx = liveAtOrAfter(c, c->excHandler[e]);
        if (handlerIdx >= c->count) continue;
        uint32_t start = offsetOfIndex(c, liveAtOrAfter(c, c->excStart[e]), total);
        uint32_t end = offsetOfIndex(c, liveAtOrAfter(c, c->excEnd[e]), total);
        if (start >= end) continue;

        fn->exceptions[kept].start = start;
        fn->exceptions[kept].end = end;
        fn->exceptions[kept].handler = offsetOfIndex(c, handlerIdx, total);
        fn->exceptions[kept].typeConst = fn->exceptions[e].typeConst;
        kept++;
    }
    if (kept != c->excCount) {
        fn->exceptions = JAI_GROW_ARRAY(ExceptionEntry, fn->exceptions,
                                        c->excCount, kept);
        fn->exceptionCount = (uint16_t)kept;
        c->excCount = kept;   /* the recorded indices are stale from here on */
    }

    for (int d = 0; d < c->defaultCount; d++) {
        fn->defaultOffsets[d] =
            offsetOfIndex(c, liveAtOrAfter(c, c->defaults[d]), total);
    }

    JAI_FREE_ARRAY(uint8_t, chunk->code, chunk->capacity);
    JAI_FREE_ARRAY(LineEntry, chunk->lines, chunk->lineCapacity);
    chunk->code = code;
    chunk->count = total;
    chunk->capacity = total;
    chunk->lines = lines;
    chunk->lineCount = lineCount;
    chunk->lineCapacity = liveCount;
    return true;
}

/* ------------------------------------------------------------------ */
/* Reachability                                                         */
/* ------------------------------------------------------------------ */

/* Mark everything not reachable from an entry point as dead and return how
 * many instructions that removed. Entry points are offset 0 and every
 * default-value thunk, which the call sequence enters directly. An exception
 * handler is *not* an unconditional entry point: it runs only when something
 * inside its protected region throws, so it is reachable exactly when some
 * instruction of that region is. Seeding it regardless resurrects the handler
 * of a `try` the front end has just proved unreachable, and with it — through
 * the backward branch at the end of an enclosing loop body — the loop's
 * OP_FOR_ITER, whose iterator is produced by the setup code that stayed dead. */
static int markUnreachable(Code *c) {
    if (c->count == 0) return 0;

    bool *reached = JAI_ALLOC_ZEROED(bool, c->count);
    int *work = JAI_ALLOC(int, c->count);
    int workCount = 0;

#define OPT_SEED(idx)                                                          \
    do {                                                                       \
        int seed_ = liveAtOrAfter(c, (idx));                                   \
        if (seed_ < c->count && !reached[seed_]) {                             \
            reached[seed_] = true;                                             \
            work[workCount++] = seed_;                                         \
        }                                                                      \
    } while (0)

    OPT_SEED(0);
    for (int d = 0; d < c->defaultCount; d++) OPT_SEED(c->defaults[d]);

    /* Reaching a handler can make another region's body reachable, so the
     * drain and the region test alternate until a round adds nothing. */
    for (;;) {
        while (workCount > 0) {
            int i = work[--workCount];
            const Instr *in = &c->data[i];
            /* Every address an instruction carries counts, including the one
             * PUSH_HANDLER and PUSH_FINALLY register: control does not go
             * there now, but it can later, and only if this instruction runs. */
            if (in->target >= 0) OPT_SEED(in->target);
            if (opFallsThrough(in->op)) {
                int next = nextLive(c, i);
                if (next >= 0) OPT_SEED(next);
            }
        }

        int added = 0;
        for (int e = 0; e < c->excCount; e++) {
            int handler = liveAtOrAfter(c, c->excHandler[e]);
            if (handler >= c->count || reached[handler]) continue;
            bool bodyLives = false;
            for (int i = c->excStart[e]; i < c->excEnd[e] && i < c->count; i++) {
                if (reached[i]) bodyLives = true;
            }
            if (!bodyLives) continue;
            OPT_SEED(handler);
            added++;
        }
        if (added == 0) break;
    }
#undef OPT_SEED

    int removed = 0;
    for (int i = 0; i < c->count; i++) {
        if (!c->data[i].dead && !reached[i]) {
            c->data[i].dead = true;
            removed++;
        }
    }

    JAI_FREE_ARRAY(bool, reached, c->count);
    JAI_FREE_ARRAY(int, work, c->count);
    return removed;
}

int jaiOptDeadCode(ObjFunction *fn) {
    Code c;
    if (!codeDecode(&c, fn)) {
        codeFree(&c);
        return 0;
    }
    int removed = markUnreachable(&c);
    if (removed > 0 && !codeRebuild(&c)) removed = 0;
    codeFree(&c);
    return removed;
}

/* ------------------------------------------------------------------ */
/* Superinstruction selection (spec §3.3, §8.6)                         */
/* ------------------------------------------------------------------ */

/* The integer an instruction pushes, for the fused forms that fold a constant
 * into their operand. Both the immediate and the pool encodings qualify. */
static bool pushedInt(const Code *c, const Instr *in, int64_t *out) {
    if (in->op == OP_INT) {
        *out = jaiReadI16(insOps(in));
        return true;
    }
    if (in->op != OP_CONST) return false;
    uint32_t k = jaiReadU24(insOps(in));
    if (k >= (uint32_t)c->chunk->constants.count) return false;
    Value v = c->chunk->constants.data[k];
    if (!IS_INT(v)) return false;
    *out = AS_INT(v);
    return true;
}

static bool fitsI16(int64_t v) { return v >= INT16_MIN && v <= INT16_MAX; }
static bool fitsI8(int64_t v) { return v >= INT8_MIN && v <= INT8_MAX; }

static uint16_t slotOf(const Instr *in, int at) {
    return jaiReadU16(insOps(in) + at);
}

static void makeSlotImm16(Instr *in, uint8_t op, uint16_t slot, int64_t imm) {
    uint8_t operands[4];
    writeU16At(operands, slot);
    writeU16At(operands + 2, (uint16_t)(int16_t)imm);
    setInstr(in, op, operands, 4);
}

static void makeSlotPair(Instr *in, uint8_t op, uint16_t a, uint16_t b) {
    uint8_t operands[4];
    writeU16At(operands, a);
    writeU16At(operands + 2, b);
    setInstr(in, op, operands, 4);
}

/* The branch operand is left zero and `target` carried on the Instr, which is
 * what codeRebuild resolves branches from; the encoder needs no special case
 * because branchOperandAt already knows where the i16 sits. */
static void makeJumpIfCmpLocalK(Instr *in, uint8_t cmp, uint16_t slot,
                                uint32_t k, int target) {
    uint8_t operands[8];
    operands[0] = cmp;
    writeU16At(operands + 1, slot);
    writeU24At(operands + 3, k);
    writeU16At(operands + 6, 0);
    setInstr(in, OP_JUMP_IF_CMP_LOCAL_K, operands, 8);
    in->target = target;
}

static void makeIncLocal(Instr *in, uint16_t slot, int64_t delta) {
    uint8_t operands[3];
    writeU16At(operands, slot);
    operands[2] = (uint8_t)(int8_t)delta;
    setInstr(in, OP_INC_LOCAL, operands, 3);
}

/* A constant-pool index holding the int `v`, appending it when the pool does
 * not already carry it. Ints are immediate values, not heap objects, so this
 * cannot hand the collector a half-built object the way appending a string
 * would; the only failure mode is a pool that has run out of u24 indices, and
 * the post-check catches the diagnostic jaiChunkAddConstant raises for that
 * rather than letting index 0 be used as if it were the constant. */
static bool poolIntIndex(Code *c, int64_t v, uint32_t *out) {
    Chunk *chunk = c->chunk;
    if (chunk == NULL || chunk->constants.count >= (int)((1u << 24) - 1)) {
        return false;
    }
    uint32_t k = jaiChunkAddConstant(chunk, INT_VAL(v));
    if (k >= (uint32_t)chunk->constants.count) return false;
    Value got = chunk->constants.data[k];
    if (!IS_INT(got) || AS_INT(got) != v) return false;
    *out = k;
    return true;
}

/* The pool index of the constant `in` pushes, for the fused forms that name a
 * constant by index rather than folding it into an immediate. OP_INT carries
 * its value inline and so may have to be interned first. */
static bool constIndexOf(Code *c, const Instr *in, uint32_t *out) {
    if (in->op == OP_CONST) {
        uint32_t k = jaiReadU24(insOps(in));
        if (k >= (uint32_t)c->chunk->constants.count) return false;
        *out = k;
        return true;
    }
    if (in->op == OP_INT) return poolIntIndex(c, jaiReadI16(insOps(in)), out);
    return false;
}

/* True when `op` leaves exactly one bool on the stack for every input, which
 * is what lets OP_JUMP_IF_CMP_FALSE swallow the jump's own bool check. */
static bool isComparison(uint8_t op) {
    switch (op) {
    case OP_EQ: case OP_NE: case OP_LT:
    case OP_LE: case OP_GT: case OP_GE:
        return true;
    default:
        return false;
    }
}

/* Greedy longest match over the fused set of spec §3.3. The first instruction
 * of every window is rewritten in place and the rest are deleted, so an edge
 * that lands on the window head still performs the whole sequence. */
static bool fuseAt(Code *c, int i) {
    int w[4];
    int n = windowOf(c, i, 4, w);
    Instr *head = &c->data[i];

    /* GET_LOCAL a; <int k>; ADD; BIND a  ->  INC_LOCAL a, k.
     *
     * The store is OP_BIND, not SET_LOCAL; POP: the emitter uses BIND for
     * every store-and-discard, so the two rules this replaces matched nothing
     * in the whole tree while `i += 1` — the single hottest pair measured —
     * went unfused.
     *
     * SUB is *not* folded in as INC_LOCAL a, -k, which is what the dead rule
     * used to say. It is only the same instruction for numbers: on anything
     * else both raise, but INC_LOCAL's slow path adds, so `y - 1` on a str
     * reported "unsupported operand types for '+'" from -O1 up and for '-' at
     * -O0. A fused form may be a fast path and nothing else. Folding `-=` back
     * in needs an operator byte on INC_LOCAL. */
    if (n >= 4 && head->op == OP_GET_LOCAL && windowClean(c, w, 4)) {
        const Instr *k = &c->data[w[1]];
        const Instr *arith = &c->data[w[2]];
        const Instr *store = &c->data[w[3]];
        int64_t value;
        if (arith->op == OP_ADD && store->op == OP_BIND &&
            slotOf(store, 0) == slotOf(head, 0) && pushedInt(c, k, &value) &&
            fitsI8(value)) {
            makeIncLocal(head, slotOf(head, 0), value);
            for (int p = 1; p < 4; p++) killInstr(c, w[p]);
            return true;
        }
    }

    /* ADD_INT_CONST a, k; BIND a  ->  INC_LOCAL a, k. Catches the increment
     * left behind when the three-instruction form fused first. */
    if (n >= 2 && head->op == OP_ADD_INT_CONST && windowClean(c, w, 2)) {
        const Instr *store = &c->data[w[1]];
        int64_t delta = jaiReadI16(insOps(head) + 2);
        if (store->op == OP_BIND && slotOf(store, 0) == slotOf(head, 0) &&
            fitsI8(delta)) {
            makeIncLocal(head, slotOf(head, 0), delta);
            killInstr(c, w[1]);
            return true;
        }
    }

    /* A local compared against a constant, branched on:
     *
     *     GET_LOCAL a; <const k>; <cmp>; JUMP_IF_FALSE J
     *     GET_LOCAL a; <const k>; JUMP_IF_CMP_FALSE cmp, J
     *     CMP_LOCAL_CONST_LT a, k; JUMP_IF_FALSE J
     *
     * all collapse to JUMP_IF_CMP_LOCAL_K cmp, a, k, J. That is `while i < N`
     * and `if n < 2` — a loop guard and a recursion base case — in one
     * dispatch instead of three. The three windows are the same shape caught
     * at three stages: the raw one, the one a previous round already half-fused
     * into JUMP_IF_CMP_FALSE, and the one the CMP_LOCAL_CONST_LT rule below
     * claimed first. The last two exist because fusion runs to a fixed point,
     * so a rule that only matched the raw form would lose every window whose
     * head was reached in a later round.
     *
     * K is a pool index rather than an immediate on purpose: the immediate
     * form OP_CMP_LOCAL_CONST_LT already exists and cannot express
     * `i < 5_000_000`, which is exactly the guard worth fusing. */
    if (n >= 4 && head->op == OP_GET_LOCAL && windowClean(c, w, 4) &&
        isComparison(c->data[w[2]].op) &&
        c->data[w[3]].op == OP_JUMP_IF_FALSE) {
        uint32_t k;
        if (constIndexOf(c, &c->data[w[1]], &k)) {
            makeJumpIfCmpLocalK(head, c->data[w[2]].op, slotOf(head, 0), k,
                                c->data[w[3]].target);
            for (int p = 1; p < 4; p++) killInstr(c, w[p]);
            return true;
        }
    }

    if (n >= 3 && head->op == OP_GET_LOCAL && windowClean(c, w, 3) &&
        c->data[w[2]].op == OP_JUMP_IF_CMP_FALSE) {
        uint32_t k;
        if (constIndexOf(c, &c->data[w[1]], &k)) {
            makeJumpIfCmpLocalK(head, insOps(&c->data[w[2]])[0], slotOf(head, 0),
                                k, c->data[w[2]].target);
            killInstr(c, w[1]);
            killInstr(c, w[2]);
            return true;
        }
    }

    if (n >= 2 && head->op == OP_CMP_LOCAL_CONST_LT && windowClean(c, w, 2) &&
        c->data[w[1]].op == OP_JUMP_IF_FALSE) {
        uint32_t k;
        if (poolIntIndex(c, jaiReadI16(insOps(head) + 2), &k)) {
            makeJumpIfCmpLocalK(head, OP_LT, slotOf(head, 0), k,
                                c->data[w[1]].target);
            killInstr(c, w[1]);
            return true;
        }
    }

    /* GET_LOCAL a; GET_FIELD k, c  ->  GET_FIELD_LOCAL a, k, c. `self.x`: the
     * most frequently executed pair in the benchmark suite by a wide margin,
     * and the most frequent one statically too. */
    if (n >= 2 && head->op == OP_GET_LOCAL && windowClean(c, w, 2) &&
        c->data[w[1]].op == OP_GET_FIELD) {
        const Instr *field = &c->data[w[1]];
        uint8_t operands[7];
        writeU16At(operands, slotOf(head, 0));
        memcpy(operands + 2, insOps(field), 5);   /* u24 name, u16 cache */
        setInstr(head, OP_GET_FIELD_LOCAL, operands, 7);
        killInstr(c, w[1]);
        return true;
    }

    /* FOR_ITER J; BIND a  ->  FOR_ITER_BIND J, a. The loop variable goes
     * straight to its slot. The jump operand keeps byte index 0, so the
     * re-encoder resolves the branch with no special case. */
    if (n >= 2 && head->op == OP_FOR_ITER && windowClean(c, w, 2) &&
        c->data[w[1]].op == OP_BIND) {
        uint8_t operands[4];
        int target = head->target;
        writeU16At(operands, 0);   /* patched by codeRebuild from `target` */
        writeU16At(operands + 2, slotOf(&c->data[w[1]], 0));
        setInstr(head, OP_FOR_ITER_BIND, operands, 4);
        head->target = target;
        killInstr(c, w[1]);
        return true;
    }

    /* <cmp>; JUMP_IF_FALSE J  ->  JUMP_IF_CMP_FALSE cmp, J. Only the popping
     * form fuses: the *_KEEP jumps leave the bool behind for `and`/`or`. */
    if (n >= 2 && isComparison(head->op) && windowClean(c, w, 2) &&
        c->data[w[1]].op == OP_JUMP_IF_FALSE) {
        uint8_t operands[3];
        int target = c->data[w[1]].target;
        operands[0] = head->op;
        writeU16At(operands + 1, 0);   /* patched by codeRebuild */
        setInstr(head, OP_JUMP_IF_CMP_FALSE, operands, 3);
        head->target = target;
        killInstr(c, w[1]);
        return true;
    }

    if (n >= 3 && head->op == OP_GET_LOCAL && windowClean(c, w, 3)) {
        const Instr *second = &c->data[w[1]];
        uint8_t third = c->data[w[2]].op;

        /* GET_LOCAL a; GET_LOCAL b; ADD  ->  ADD_LOCALS a, b */
        if (second->op == OP_GET_LOCAL && third == OP_ADD) {
            makeSlotPair(head, OP_ADD_LOCALS, slotOf(head, 0), slotOf(second, 0));
            killInstr(c, w[1]);
            killInstr(c, w[2]);
            return true;
        }

        int64_t value;
        if (pushedInt(c, second, &value) && fitsI16(value)) {
            /* GET_LOCAL a; <int k>; ADD  ->  ADD_INT_CONST a, k */
            if (third == OP_ADD) {
                makeSlotImm16(head, OP_ADD_INT_CONST, slotOf(head, 0), value);
                killInstr(c, w[1]);
                killInstr(c, w[2]);
                return true;
            }
            /* GET_LOCAL a; <int k>; LT  ->  CMP_LOCAL_CONST_LT a, k */
            if (third == OP_LT) {
                makeSlotImm16(head, OP_CMP_LOCAL_CONST_LT, slotOf(head, 0), value);
                killInstr(c, w[1]);
                killInstr(c, w[2]);
                return true;
            }
        }
    }

    /* GET_LOCAL2 a, b; ADD  ->  ADD_LOCALS a, b */
    if (n >= 2 && head->op == OP_GET_LOCAL2 && windowClean(c, w, 2) &&
        c->data[w[1]].op == OP_ADD) {
        head->op = OP_ADD_LOCALS;
        killInstr(c, w[1]);
        return true;
    }

    /* GET_LOCAL a; GET_LOCAL b  ->  GET_LOCAL2 a, b, and GET_LOCAL a; DUP,
     * which pushes the same slot twice and is therefore GET_LOCAL2 a, a. The
     * DUP form is how a compound field assignment (`b.vx -= …`) starts, which
     * makes it 6% of every instruction nbody executes. */
    if (n >= 2 && head->op == OP_GET_LOCAL && windowClean(c, w, 2)) {
        const Instr *second = &c->data[w[1]];
        if (second->op == OP_GET_LOCAL || second->op == OP_DUP) {
            uint16_t a = slotOf(head, 0);
            uint16_t b = second->op == OP_DUP ? a : slotOf(second, 0);
            makeSlotPair(head, OP_GET_LOCAL2, a, b);
            killInstr(c, w[1]);
            return true;
        }
    }
    return false;
}

int jaiOptSuperinstructions(ObjFunction *fn) {
    Code c;
    if (!codeDecode(&c, fn)) {
        codeFree(&c);
        return 0;
    }

    int changes = 0;
    int guard = 8 * c.count + 64;
    int i = firstLive(&c);
    while (i >= 0 && guard-- > 0) {
        if (!fuseAt(&c, i)) {
            i = nextLive(&c, i);
            continue;
        }
        changes++;
        /* The window head always survives a fusion, so staying at or before
         * `i` is what chains GET_LOCAL2 into ADD_LOCALS; backing up one lets
         * the closed hole open a window for the instruction in front of it.
         * Every fusion deletes an instruction, so this still terminates. */
        int back = prevLive(&c, i);
        if (back >= 0) i = back;
    }

    if (changes > 0 && !codeRebuild(&c)) changes = 0;
    codeFree(&c);
    return changes;
}

/* ------------------------------------------------------------------ */
/* Peephole (spec §8.3)                                                 */
/* ------------------------------------------------------------------ */

static bool isConditionalJump(uint8_t op) {
    return op == OP_JUMP_IF_FALSE || op == OP_JUMP_IF_TRUE ||
           op == OP_JUMP_IF_FALSE_KEEP || op == OP_JUMP_IF_TRUE_KEEP;
}

static uint8_t invertedJump(uint8_t op) {
    switch (op) {
    case OP_JUMP_IF_FALSE:      return OP_JUMP_IF_TRUE;
    case OP_JUMP_IF_TRUE:       return OP_JUMP_IF_FALSE;
    case OP_JUMP_IF_FALSE_KEEP: return OP_JUMP_IF_TRUE_KEEP;
    case OP_JUMP_IF_TRUE_KEEP:  return OP_JUMP_IF_FALSE_KEEP;
    default:                    return op;
    }
}

/* The boolean an instruction pushes, if it is a constant push. */
static bool pushedBool(const Code *c, const Instr *in, bool *out) {
    if (in->op == OP_TRUE) { *out = true; return true; }
    if (in->op == OP_FALSE) { *out = false; return true; }
    if (in->op != OP_CONST) return false;
    uint32_t k = jaiReadU24(insOps(in));
    if (k >= (uint32_t)c->chunk->constants.count) return false;
    Value v = c->chunk->constants.data[k];
    if (!IS_BOOL(v)) return false;
    *out = AS_BOOL(v);
    return true;
}

/* Follow a chain of unconditional JUMPs to its end. LOOP is never followed:
 * it is the GC and interrupt safepoint (spec §10), and jumping over one would
 * let a tight loop run uninterrupted. The step bound makes a jump cycle
 * terminate; landing anywhere inside such a cycle is equivalent anyway. */
static int chaseJump(const Code *c, int target) {
    int at = liveAtOrAfter(c, target);
    for (int steps = 0; steps < c->count && at < c->count; steps++) {
        const Instr *in = &c->data[at];
        if (in->op != OP_JUMP || in->target < 0) break;
        int next = liveAtOrAfter(c, in->target);
        if (next == at) break;
        at = next;
    }
    return at;
}

/* Bytes between a branch and its target in the current layout. Deletions only
 * pull the two closer, so a value that fits now still fits after re-encoding;
 * rewrites that would overflow i16 are rejected here rather than aborting the
 * whole pass in codeRebuild. */
static bool retargetFits(const Code *c, int from, int to) {
    long here = (long)c->data[from].newOffset + insSize(&c->data[from]);
    long there = (long)c->data[to].newOffset;
    long delta = there - here;
    return delta >= INT16_MIN && delta <= INT16_MAX;
}

static bool rewriteBranchAt(Code *c, int i) {
    Instr *in = &c->data[i];
    if (in->target < 0) return false;

    int follow = nextLive(c, i);

    /* JUMP to the next instruction is a no-op. Deleting it is safe even when
     * it is itself a target: every edge into it simply lands on the
     * instruction it would have jumped to. */
    if (in->op == OP_JUMP && follow >= 0 && liveAtOrAfter(c, in->target) == follow) {
        killInstr(c, i);
        return true;
    }

    /* JUMP to a JUMP: go straight to the final target. */
    if (opBranchIsEdge(in->op) && in->op != OP_LOOP) {
        int resolved = chaseJump(c, in->target);
        if (resolved < c->count && resolved != liveAtOrAfter(c, in->target) &&
            resolved != i && retargetFits(c, i, resolved)) {
            in->target = resolved;
            c->data[resolved].isTarget = true;
            return true;
        }
    }

    /* JUMP_IF_FALSE over a lone JUMP: invert the test and take the JUMP's
     * target. The head keeps its position, so an edge into it is unaffected. */
    if (isConditionalJump(in->op) && follow >= 0) {
        const Instr *over = &c->data[follow];
        int after = nextLive(c, follow);
        /* `landing < c->count` is the same guard the two rules above carry, and
         * for the same reason: liveAtOrAfter answers c->count when everything
         * from the target on has been deleted, and both retargetFits and the
         * isTarget write below dereference c->data[landing]. Without it those
         * are a read and a write one past the end of the array — reachable when
         * an earlier rewrite in this same scan killed the tail this JUMP points
         * at. Refusing the rewrite is the answer for every input where the read
         * was defined anyway, since codeRebuild rejects a branch resolving off
         * the end. */
        int landing = liveAtOrAfter(c, over->target);
        if (over->op == OP_JUMP && over->target >= 0 && !over->isTarget &&
            after >= 0 && liveAtOrAfter(c, in->target) == after &&
            landing < c->count && retargetFits(c, i, landing)) {
            in->op = invertedJump(in->op);
            in->target = landing;
            c->data[landing].isTarget = true;
            killInstr(c, follow);
            return true;
        }
    }
    return false;
}

/* A constant condition in front of a conditional jump. Both rewrites delete
 * the push, whose effect the jump cancelled, and either delete the jump or
 * demote it to an unconditional one — never the other way round, so the code
 * only ever shrinks. */
static bool rewriteConstCondition(Code *c, int i) {
    int w[2];
    if (windowOf(c, i, 2, w) < 2 || !windowClean(c, w, 2)) return false;

    Instr *push = &c->data[w[0]];
    Instr *jump = &c->data[w[1]];
    bool value;
    if (!pushedBool(c, push, &value)) return false;
    if (jump->target < 0) return false;

    bool taken;
    switch (jump->op) {
    case OP_JUMP_IF_FALSE:      taken = !value; break;
    case OP_JUMP_IF_TRUE:       taken = value;  break;
    case OP_JUMP_IF_FALSE_KEEP:
    case OP_JUMP_IF_TRUE_KEEP: {
        /* The value stays on the stack, so only the never-taken case can be
         * simplified, and only the jump goes. */
        bool keepTaken = jump->op == OP_JUMP_IF_FALSE_KEEP ? !value : value;
        if (keepTaken) return false;
        killInstr(c, w[1]);
        return true;
    }
    default:
        return false;
    }

    if (!taken) {
        killInstr(c, w[0]);
        killInstr(c, w[1]);
        return true;
    }
    /* Reuse the jump's slot: OP_JUMP is the same width as the conditional it
     * replaces, so nothing downstream moves. */
    jump->op = OP_JUMP;
    killInstr(c, w[0]);
    return true;
}

/* Instructions whose result is a bool whatever their operands were. Any of
 * them satisfies OP_NOT's operand check on its own. */
static bool pushesBool(uint8_t op) {
    switch (op) {
    case OP_TRUE:
    case OP_FALSE:
    case OP_EQ: case OP_NE: case OP_LT: case OP_LE: case OP_GT: case OP_GE:
    case OP_IS: case OP_IS_NOT: case OP_IN: case OP_NOT_IN:
    case OP_NOT:
    case OP_IS_INSTANCE:
    case OP_CMP_LOCAL_CONST_LT:
        return true;
    default:
        return false;
    }
}

/* NOT feeding a conditional jump: invert the jump instead. Both instructions
 * change meaning for an incoming edge, so neither may be a branch target.
 *
 * The jump checks its condition is a bool too, so erasing the NOT looks free —
 * but the two report the failure differently ("operand of 'not'" against
 * "condition"), and a program that catches the TypeError can read the message.
 * There is no truthiness in Jaithon (LANGUAGE.md §5.1), so an `any` value here
 * is exactly the case that fails, which is why the rewrite waits for a producer
 * that is a bool by construction: then the check cannot fire at all and which
 * instruction would have run it stops mattering. */
static bool rewriteNotJump(Code *c, int i) {
    int w[2];
    if (windowOf(c, i, 2, w) < 2) return false;
    if (c->data[w[0]].op != OP_NOT || c->data[w[0]].isTarget) return false;
    if (c->data[w[1]].isTarget) return false;

    uint8_t op = c->data[w[1]].op;
    if (op != OP_JUMP_IF_FALSE && op != OP_JUMP_IF_TRUE) return false;

    int producer = prevLive(c, w[0]);
    if (producer < 0 || !pushesBool(c->data[producer].op)) return false;
    /* An edge landing on the producer runs the NOT; one landing between it and
     * the NOT does not exist, since the NOT is not a target. */

    c->data[w[1]].op = invertedJump(op);
    killInstr(c, w[0]);
    return true;
}

/* POP;POP;… -> POPN n, and DUP;POP -> nothing. */
static bool rewriteStackOps(Code *c, int i) {
    Instr *head = &c->data[i];

    if (head->op == OP_DUP) {
        int w[2];
        if (windowOf(c, i, 2, w) == 2 && windowClean(c, w, 2) &&
            c->data[w[1]].op == OP_POP) {
            killInstr(c, w[0]);
            killInstr(c, w[1]);
            return true;
        }
        return false;
    }

    if (head->op != OP_POP && head->op != OP_POPN) return false;

    int total = head->op == OP_POP ? 1 : insOps(head)[0];
    int last = i;
    int merged = 0;
    for (int j = nextLive(c, i); j >= 0; j = nextLive(c, j)) {
        const Instr *in = &c->data[j];
        if (in->isTarget) break;
        int n;
        if (in->op == OP_POP) {
            n = 1;
        } else if (in->op == OP_POPN) {
            n = insOps(in)[0];
        } else {
            break;
        }
        if (total + n > 255) break;
        total += n;
        last = j;
        merged++;
    }
    if (merged == 0) return false;

    uint8_t operand = (uint8_t)total;
    setInstr(head, OP_POPN, &operand, 1);
    for (int j = nextLive(c, i); j >= 0 && j <= last; j = nextLive(c, j)) {
        killInstr(c, j);
    }
    return true;
}

/* Fusion is deliberately NOT here. It used to be the first rule, which meant
 * jaiOptSuperinstructions found every §3.3 window already consumed and
 * reported zero rewrites in all 8,570 functions of the tree — spec §8's two
 * passes were one. Worse, fusing a comparison into its jump before the branch
 * rewrites have run hides `JUMP_IF_FALSE over a lone JUMP` from the rule that
 * would have deleted the JUMP. Pass 3 now finishes the control flow and pass 6
 * fuses what is left. */
static bool rewriteAt(Code *c, int i) {
    if (rewriteBranchAt(c, i)) return true;
    if (rewriteConstCondition(c, i)) return true;
    if (rewriteNotJump(c, i)) return true;
    if (rewriteStackOps(c, i)) return true;
    return false;
}

static int peepholeScan(Code *c) {
    int changes = 0;
    /* Every rewrite but the retarget removes an instruction, and the retarget
     * is idempotent, so this terminates; the guard is insurance against a
     * future rule that is not. */
    int guard = 8 * c->count + 64;
    int i = firstLive(c);
    while (i >= 0 && guard-- > 0) {
        if (!rewriteAt(c, i)) {
            i = nextLive(c, i);
            continue;
        }
        changes++;
        if (!c->data[i].dead) continue;
        /* Back up one: deleting an instruction can make its neighbours
         * adjacent and open a pattern that spans the hole. */
        int resume = prevLive(c, i);
        i = resume >= 0 ? resume : nextLive(c, i);
    }
    return changes;
}

int jaiOptPeephole(ObjFunction *fn) {
    Code c;
    if (!codeDecode(&c, fn)) {
        codeFree(&c);
        return 0;
    }

    /* Rewriting strands blocks, and removing a stranded block puts new
     * neighbours next to each other, so the two feed each other until neither
     * has anything left to do. */
    int changes = 0;
    for (int round = 0; round < 8; round++) {
        int before = changes;
        changes += peepholeScan(&c);
        changes += markUnreachable(&c);
        if (changes == before) break;
    }

    if (changes > 0 && !codeRebuild(&c)) changes = 0;
    codeFree(&c);
    return changes;
}

/* ------------------------------------------------------------------ */
/* Local slot coalescing (spec §8.4)                                    */
/* ------------------------------------------------------------------ */

/* Byte offsets of the u16 local-slot operands of `op`, if any. */
static int slotOperands(uint8_t op, int *out) {
    switch (op) {
    case OP_GET_LOCAL:
    case OP_SET_LOCAL:
    case OP_BIND:
    case OP_CLOSE_UPVALUE:
    case OP_ADD_INT_CONST:
    case OP_INC_LOCAL:
    case OP_CMP_LOCAL_CONST_LT:
    case OP_GET_FIELD_LOCAL:
        out[0] = 0;
        return 1;
    case OP_FOR_ITER_BIND:
        out[0] = 2;   /* the slot follows the i16 jump */
        return 1;
    case OP_JUMP_IF_CMP_LOCAL_K:
        out[0] = 1;   /* the slot follows the u8 comparison */
        return 1;
    case OP_GET_LOCAL2:
    case OP_ADD_LOCALS:
        out[0] = 0;
        out[1] = 2;
        return 2;
    default:
        return 0;
    }
}

/* Slots the caller owns: the callee/self slot plus every parameter. They are
 * filled in by the call sequence, so their numbering is part of the calling
 * convention and can never move. */
static int reservedSlots(const ObjFunction *fn) {
    int params = (int)fn->arity;   /* already counts the defaulted parameters */
    if (fn->flags & FN_VARIADIC) params++;
    if (fn->flags & FN_KWREST) params++;
    if ((int)fn->paramCount > params) params = (int)fn->paramCount;

    int reserved = params + 1;
    if (reserved > (int)fn->maxSlots) reserved = (int)fn->maxSlots;
    return reserved;
}

typedef struct {
    int first, last;   /* instruction indices; -1 when the slot is unused */
    bool pinned;
} SlotRange;

int jaiOptCoalesceSlots(ObjFunction *fn) {
    Code c;
    if (!codeDecode(&c, fn)) {
        codeFree(&c);
        return 0;
    }

    int slotCount = (int)fn->maxSlots;
    int reserved = reservedSlots(fn);
    if (slotCount <= reserved) {
        codeFree(&c);
        return 0;
    }

    SlotRange *range = JAI_ALLOC(SlotRange, slotCount);
    for (int s = 0; s < slotCount; s++) {
        range[s].first = OPT_NO_INDEX;
        range[s].last = OPT_NO_INDEX;
        range[s].pinned = s < reserved;
    }

    /* The first point at which exception flow can bypass a definition. */
    bool bail = false;
    int hazardStart = c.count;
    for (int e = 0; e < c.excCount; e++) {
        if (c.excStart[e] < hazardStart) hazardStart = c.excStart[e];
        if (c.excHandler[e] < hazardStart) hazardStart = c.excHandler[e];
    }

    for (int i = 0; i < c.count && !bail; i++) {
        const Instr *in = &c.data[i];
        if (in->dead) continue;

        if (in->op == OP_PUSH_DEFER) {
            /* A deferred closure runs at frame exit, reading slots in an order
             * that has no relation to this scan's linear one. */
            bail = true;
            break;
        }
        if (in->op == OP_CLOSURE) {
            /* A captured slot is aliased by an open upvalue that is closed by
             * stack position, so both its number and its position must hold. */
            const uint8_t *ops = insOps(in);
            int upvalues = (in->operandCount - 3) / 3;
            for (int u = 0; u < upvalues; u++) {
                const uint8_t *desc = ops + 3 + u * 3;
                /* Bit 0 of the descriptor, not the whole byte: bit 1 is the
                 * by-value flag, so a by-value capture of an *upvalue* reads
                 * as 2 and its operand is an upvalue index, not a slot. */
                if ((desc[0] & 1u) == 0) continue;
                uint16_t slot = jaiReadU16(desc + 1);
                if (slot >= slotCount) { bail = true; break; }
                range[slot].pinned = true;
            }
            continue;
        }

        int at[2];
        int n = slotOperands(in->op, at);
        for (int k = 0; k < n; k++) {
            uint16_t slot = slotOf(in, at[k]);
            if (slot >= slotCount) { bail = true; break; }
            if (range[slot].first < 0) range[slot].first = i;
            range[slot].last = i;
        }

        if (!bail && in->op == OP_CLOSE_UPVALUE) {
            /* Its operand names a position, not just a variable: it closes this
             * slot *and everything above it*. Renumbering it would move the
             * boundary. The captured slots it covers are pinned by their
             * OP_CLOSURE already, so pinning the boundary is enough to keep the
             * "at or above" relation intact. */
            range[slotOf(in, 0)].pinned = true;
        }
    }

    int changed = 0;
    if (bail) goto done;

    /* Anything still live once exception flow can divert control is left
     * alone: a handler may be entered from the middle of the protected region,
     * and this linear scan cannot see which definitions were skipped. */
    for (int s = reserved; s < slotCount; s++) {
        if (range[s].last >= hazardStart) range[s].pinned = true;
    }

    /* A backward edge makes every slot referenced inside the loop live across
     * the whole loop; without this, a slot last read at the top of a loop body
     * would look dead to a slot first written at the bottom of it. */
    for (int round = 0; round < 8; round++) {
        bool grew = false;
        for (int i = 0; i < c.count; i++) {
            const Instr *in = &c.data[i];
            if (in->dead || in->target < 0 || !opBranchIsEdge(in->op)) continue;
            int top = liveAtOrAfter(&c, in->target);
            if (top >= i) continue;
            for (int s = reserved; s < slotCount; s++) {
                if (range[s].first < 0 || range[s].pinned) continue;
                if (range[s].first > i || range[s].last < top) continue;
                if (range[s].first > top) { range[s].first = top; grew = true; }
                if (range[s].last < i) { range[s].last = i; grew = true; }
            }
        }
        if (!grew) break;
    }

    /* Linear scan: walk the candidate slots in order of first reference and
     * give each one the lowest number that is free where it starts.
     *
     * A number can host several candidates in turn, so what has to be tracked
     * per *number* is the furthest point any of its tenants reaches — not the
     * identity of one of them. Tracking a single "current occupant" and asking
     * about that one's range reads the wrong interval as soon as a number has
     * taken a second tenant, and hands out a number that is still live.
     *
     * `liveTo[t]` is that furthest point, made of two parts:
     *   - the candidate still sitting in its own number t, until it moves;
     *   - every candidate already assigned to t.
     * `c.count` is past the last instruction index, so it reads as "never
     * free" for a number that must not be reused at all. */
    int *target = JAI_ALLOC(int, slotCount);
    int *liveTo = JAI_ALLOC(int, slotCount);
    bool *placed = JAI_ALLOC(bool, slotCount);
    const int NEVER_FREE = c.count;

    for (int s = 0; s < slotCount; s++) {
        target[s] = s;
        placed[s] = false;
        /* A pinned number is never reusable, even when nothing in this linear
         * scan references it: a slot captured by a closure is pinned but may
         * carry no GET/SET of its own here, and handing its number to another
         * candidate would alias the open upvalue. */
        if (range[s].pinned) liveTo[s] = NEVER_FREE;
        else if (range[s].first >= 0) liveTo[s] = range[s].last;
        else liveTo[s] = OPT_NO_INDEX;
    }

    int oldTop = reserved;
    for (int s = 0; s < slotCount; s++) {
        if (range[s].first >= 0 && s + 1 > oldTop) oldTop = s + 1;
    }

    for (;;) {
        /* Pick the unplaced candidate with the earliest first reference. */
        int pick = OPT_NO_INDEX;
        for (int s = reserved; s < slotCount; s++) {
            if (range[s].first < 0 || range[s].pinned || placed[s]) continue;
            if (pick < 0 || range[s].first < range[pick].first) pick = s;
        }
        if (pick < 0) break;
        placed[pick] = true;

        /* Only numbers below `pick` are worth trying: candidates are placed in
         * first-reference order, so a number above it is either still held by
         * an unplaced candidate that starts later and overlaps, or unused. */
        int chosen = pick;
        for (int t = reserved; t < pick; t++) {
            if (liveTo[t] >= range[pick].first) continue;
            chosen = t;
            break;
        }

        target[pick] = chosen;
        if (chosen != pick) {
            /* `pick` gives up its own number; whoever comes next may have it. */
            liveTo[pick] = OPT_NO_INDEX;
            if (range[pick].last > liveTo[chosen]) liveTo[chosen] = range[pick].last;
            changed++;
        }
    }

    if (changed > 0) {
        int newTop = reserved;
        for (int s = 0; s < slotCount; s++) {
            if (range[s].first >= 0 && target[s] + 1 > newTop) newTop = target[s] + 1;
        }

        for (int i = 0; i < c.count; i++) {
            Instr *in = &c.data[i];
            if (in->dead) continue;
            int at[2];
            int n = slotOperands(in->op, at);
            for (int k = 0; k < n; k++) {
                uint16_t slot = slotOf(in, at[k]);
                writeU16At(in->buf + at[k], (uint16_t)target[slot]);
            }
        }

        if (!codeRebuild(&c)) {
            changed = 0;
        } else if (newTop < oldTop) {
            /* Shrink the window by exactly the slots that disappeared rather
             * than setting it to newTop: whatever headroom the emitter left
             * above the locals for temporaries has to survive. */
            int shrunk = (int)fn->maxSlots - (oldTop - newTop);
            fn->maxSlots = (uint16_t)(shrunk < newTop ? newTop : shrunk);
        }
    }

    JAI_FREE_ARRAY(int, target, slotCount);
    JAI_FREE_ARRAY(int, liveTo, slotCount);
    JAI_FREE_ARRAY(bool, placed, slotCount);

done:
    JAI_FREE_ARRAY(SlotRange, range, slotCount);
    codeFree(&c);
    return changed;
}

/* ------------------------------------------------------------------ */
/* Loop-invariant GET_GLOBAL hoisting (spec §8.7)                       */
/* ------------------------------------------------------------------ */

#define OPT_MAX_HOIST_PER_LOOP 8

/* Anything that can reach the module table, directly or by transferring
 * control into code that might. Per the pass contract this is the call family,
 * the global writers and the importers: if any of them appears in the loop the
 * global is not provably invariant and the loop is skipped entirely. */
static bool opBlocksHoist(uint8_t op) {
    switch (op) {
    case OP_CALL:
    case OP_CALL_KW:
    case OP_CALL_SPREAD:
    case OP_INVOKE:
    case OP_SUPER_INVOKE:
    case OP_TAIL_CALL:
    case OP_NEW:
    case OP_ENUM_NEW:
    case OP_CLOSURE:
    case OP_IMPORT:
    case OP_IMPORT_FROM:
    case OP_SET_GLOBAL:
    case OP_DEF_GLOBAL:
    case OP_EXPORT:
    case OP_PUSH_DEFER:
    case OP_GET_FIELD:
    case OP_GET_FIELD_LOCAL:   /* a property getter is user code */
    case OP_SET_FIELD:
    case OP_GET_INDEX:
    case OP_SET_INDEX:
    case OP_GET_SLICE:
    case OP_SET_SLICE:
    case OP_GET_ITER:
    case OP_FOR_ITER:
    case OP_FOR_ITER_BIND:     /* a user __next__ is user code */
    case OP_FORMAT:            /* a user __str__, or a rebound `str`, is too */
    case OP_IN:
    case OP_NOT_IN:
    case OP_GET_SUPER:
    case OP_CLASS:
    case OP_METHOD:
    case OP_FIELD_DEF:
    case OP_IMPL_TRAIT:
    case OP_INHERIT:
    case OP_THROW:
    case OP_RERAISE:
    case OP_PUSH_HANDLER:
        return true;
    default:
        return false;
    }
}

typedef struct {
    int      header;                            /* loop header instruction index */
    int      tail;                              /* backward branch index */
    int      count;
    uint32_t nameConst[OPT_MAX_HOIST_PER_LOOP];
    uint16_t slot[OPT_MAX_HOIST_PER_LOOP];
    uint16_t cache[OPT_MAX_HOIST_PER_LOOP];
} HoistPlan;

/* The loop header may only be entered by falling into it from outside or by
 * branching to it from inside the loop; otherwise a pre-header load would be
 * skipped on some path into the loop. */
static bool headerIsPrivate(const Code *c, int header, int tail) {
    for (int i = 0; i < c->count; i++) {
        const Instr *in = &c->data[i];
        if (in->dead || in->target != header) continue;
        if (in->op == OP_PUSH_HANDLER || in->op == OP_PUSH_FINALLY) return false;
        if (i < header || i > tail) return false;
    }
    for (int e = 0; e < c->excCount; e++) {
        if (c->excStart[e] == header || c->excEnd[e] == header ||
            c->excHandler[e] == header) {
            return false;
        }
    }
    for (int d = 0; d < c->defaultCount; d++) {
        if (c->defaults[d] == header) return false;
    }
    if (header > 0) {
        int before = prevLive(c, header);
        if (before < 0 || !opFallsThrough(c->data[before].op)) return false;
    }
    return true;
}

static int highestSlotUsed(const Code *c) {
    int highest = -1;
    for (int i = 0; i < c->count; i++) {
        const Instr *in = &c->data[i];
        if (in->dead) continue;
        if (in->op == OP_CLOSURE) {
            const uint8_t *ops = insOps(in);
            int upvalues = (in->operandCount - 3) / 3;
            for (int u = 0; u < upvalues; u++) {
                const uint8_t *desc = ops + 3 + u * 3;
                if (desc[0] == 0) continue;
                int slot = (int)jaiReadU16(desc + 1);
                if (slot > highest) highest = slot;
            }
            continue;
        }
        int at[2];
        int n = slotOperands(in->op, at);
        for (int k = 0; k < n; k++) {
            int slot = (int)slotOf(in, at[k]);
            if (slot > highest) highest = slot;
        }
    }
    return highest;
}

/* Splice the pre-header loads of every plan into the list and remap every
 * recorded instruction index onto the new numbering. */
static bool applyHoists(Code *c, const HoistPlan *plans, int planCount) {
    int inserted = 0;
    for (int p = 0; p < planCount; p++) inserted += plans[p].count * 3;
    if (inserted == 0) return false;

    int newCount = c->count + inserted;
    Instr *out = JAI_ALLOC(Instr, newCount);
    int *map = JAI_ALLOC(int, c->count + 1);
    int w = 0;

    for (int i = 0; i < c->count; i++) {
        for (int p = 0; p < planCount; p++) {
            if (plans[p].header != i) continue;
            for (int h = 0; h < plans[p].count; h++) {
                Instr load;
                memset(&load, 0, sizeof load);
                load.span = c->data[i].span;
                load.spanEnd = c->data[i].spanEnd;
                load.target = OPT_NO_INDEX;

                uint8_t operands[5];
                writeU24At(operands, plans[p].nameConst[h]);
                writeU16At(operands + 3, plans[p].cache[h]);
                setInstr(&load, OP_GET_GLOBAL, operands, 5);
                out[w++] = load;

                writeU16At(operands, plans[p].slot[h]);
                setInstr(&load, OP_SET_LOCAL, operands, 2);
                out[w++] = load;

                setInstr(&load, OP_POP, NULL, 0);
                out[w++] = load;
            }
        }
        /* The header maps past its own pre-header, so the backward branch
         * still lands on the loop and not on the hoisted load. */
        map[i] = w;
        out[w++] = c->data[i];
    }
    map[c->count] = w;

    for (int i = 0; i < w; i++) {
        if (out[i].target >= 0 && out[i].target <= c->count) {
            out[i].target = map[out[i].target];
        }
    }
    for (int e = 0; e < c->excCount; e++) {
        c->excStart[e] = map[c->excStart[e]];
        c->excEnd[e] = map[c->excEnd[e]];
        c->excHandler[e] = map[c->excHandler[e]];
    }
    for (int d = 0; d < c->defaultCount; d++) c->defaults[d] = map[c->defaults[d]];

    JAI_FREE_ARRAY(int, map, c->count + 1);
    JAI_FREE_ARRAY(Instr, c->data, c->capacity);
    c->data = out;
    c->count = w;
    c->capacity = newCount;

    /* Byte offsets from the old layout are meaningless now; codeRebuild works
     * from indices only. */
    JAI_FREE_ARRAY(int, c->offsetIndex, c->offsetIndexLen);
    c->offsetIndex = NULL;
    c->offsetIndexLen = 0;
    return true;
}

/* One round of hoisting: pick a set of non-overlapping loops, rewrite their
 * in-loop reads and splice in the pre-header loads. Returns the number of
 * globals hoisted, with fn->maxSlots already widened on success. */
static int hoistRound(Code *c, ObjFunction *fn) {
    HoistPlan *plans = NULL;
    int planCount = 0, planCapacity = 0;
    int hoisted = 0;

    int nextSlot = highestSlotUsed(c) + 1;
    if (nextSlot < reservedSlots(fn)) nextSlot = reservedSlots(fn);
    if (nextSlot < (int)fn->maxSlots) nextSlot = (int)fn->maxSlots;

    /* Outermost loops first: hoisting an outer loop removes the reads that an
     * inner one would otherwise hoist to a worse place. */
    for (;;) {
        int bestTail = OPT_NO_INDEX, bestHeader = 0, bestSpan = -1;
        for (int i = 0; i < c->count; i++) {
            const Instr *in = &c->data[i];
            if (in->dead || in->target < 0 || !opBranchIsEdge(in->op)) continue;
            int header = liveAtOrAfter(c, in->target);
            if (header >= i) continue;

            bool overlaps = false;
            for (int p = 0; p < planCount; p++) {
                if (!(i < plans[p].header || header > plans[p].tail)) overlaps = true;
            }
            if (overlaps || i - header <= bestSpan) continue;
            bestSpan = i - header;
            bestHeader = header;
            bestTail = i;
        }
        if (bestTail < 0) break;

        HoistPlan plan;
        memset(&plan, 0, sizeof plan);
        plan.header = bestHeader;
        plan.tail = bestTail;

        bool usable = headerIsPrivate(c, bestHeader, bestTail);
        for (int i = bestHeader; i <= bestTail && usable; i++) {
            const Instr *in = &c->data[i];
            if (in->dead) continue;
            if (opBlocksHoist(in->op)) usable = false;
        }
        if (usable) {
            for (int i = bestHeader; i <= bestTail; i++) {
                const Instr *in = &c->data[i];
                if (in->dead || in->op != OP_GET_GLOBAL) continue;
                uint32_t name = jaiReadU24(insOps(in));
                bool seen = false;
                for (int h = 0; h < plan.count; h++) {
                    if (plan.nameConst[h] == name) seen = true;
                }
                if (seen || plan.count >= OPT_MAX_HOIST_PER_LOOP) continue;
                if (nextSlot + plan.count >= JAI_MAX_LOCALS) break;
                plan.nameConst[plan.count] = name;
                plan.slot[plan.count] = (uint16_t)(nextSlot + plan.count);
                plan.cache[plan.count] = jaiChunkAddCache(c->chunk);
                plan.count++;
            }
        }

        if (plan.count > 0) {
            for (int i = bestHeader; i <= bestTail; i++) {
                Instr *in = &c->data[i];
                if (in->dead || in->op != OP_GET_GLOBAL) continue;
                uint32_t name = jaiReadU24(insOps(in));
                for (int h = 0; h < plan.count; h++) {
                    if (plan.nameConst[h] != name) continue;
                    uint8_t operands[2];
                    writeU16At(operands, plan.slot[h]);
                    setInstr(in, OP_GET_LOCAL, operands, 2);
                    break;
                }
            }
            nextSlot += plan.count;
            hoisted += plan.count;
        }

        /* Recorded even when nothing was hoisted, so the loop is not
         * reconsidered and any nested loop is left for the next round. */
        if (planCapacity < planCount + 1) {
            int oldCapacity = planCapacity;
            planCapacity = JAI_GROW_CAP(oldCapacity);
            plans = JAI_GROW_ARRAY(HoistPlan, plans, oldCapacity, planCapacity);
        }
        plans[planCount++] = plan;
    }

    if (hoisted > 0) {
        if (!applyHoists(c, plans, planCount)) {
            hoisted = 0;
        } else {
            int want = (int)fn->maxSlots + hoisted;
            if (nextSlot > want) want = nextSlot;
            fn->maxSlots = (uint16_t)(want > JAI_MAX_LOCALS ? JAI_MAX_LOCALS : want);
        }
    }
    JAI_FREE_ARRAY(HoistPlan, plans, planCapacity);
    return hoisted;
}

int jaiOptHoistGlobals(ObjFunction *fn) {
    int total = 0;
    for (int round = 0; round < 4; round++) {
        Code c;
        if (!codeDecode(&c, fn)) {
            codeFree(&c);
            break;
        }
        uint16_t slotsBefore = fn->maxSlots;
        int hoisted = hoistRound(&c, fn);
        if (hoisted > 0 && !codeRebuild(&c)) {
            fn->maxSlots = slotsBefore;
            hoisted = 0;
        }
        codeFree(&c);
        if (hoisted == 0) break;
        total += hoisted;
    }
    return total;
}

/* ------------------------------------------------------------------ */
/* Driver                                                               */
/* ------------------------------------------------------------------ */

static void optimizeFunction(ObjFunction *fn, const CodegenOptions *opts, int depth);

/* Nested functions live in the constant pool; optimising them from here means
 * a front end that only calls jaiOptimize on the module body still gets a
 * fully optimised program. The passes are idempotent, so a front end that also
 * optimises each function as it finishes it loses nothing but time. */
static void optimizeNested(ObjFunction *fn, const CodegenOptions *opts, int depth) {
    for (int i = 0; i < fn->chunk.constants.count; i++) {
        Value v = fn->chunk.constants.data[i];
        if (IS_FUNCTION(v) && AS_FUNCTION(v) != fn) {
            optimizeFunction(AS_FUNCTION(v), opts, depth + 1);
        }
    }
}

static void optimizeFunction(ObjFunction *fn, const CodegenOptions *opts, int depth) {
    if (fn == NULL || depth > 64) return;

    int level = opts != NULL ? opts->optLevel : 2;
    optimizeNested(fn, opts, depth);

#ifdef JAI_DEBUG
    char before[256], after[256];
    bool wasValid = jaiVerifyChunk(fn, before, sizeof before);
#endif

    /* Unreachable code is removed at every level: spec §8 disables passes 3-7
     * under --O0, not pass 2. */
    jaiOptDeadCode(fn);

    if (level > 0) {
        /* Fusing exposes new windows and deleting exposes new fusions, so run
         * the byte-level passes to a fixpoint before the ones that depend on
         * the final instruction stream. */
        for (int round = 0; round < 4; round++) {
            int changes = jaiOptPeephole(fn);
            changes += jaiOptSuperinstructions(fn);
            if (changes == 0) break;
        }
        if (!jaiFunctionIsDeferThunk(fn) && jaiOptHoistGlobals(fn) > 0) {
            jaiOptPeephole(fn);
            jaiOptSuperinstructions(fn);
        }
        /* Coalescing runs last so that it sees the final set of locals,
         * including the ones hoisting just introduced. Running it earlier
         * leaves a hoisted slot unpacked until the next call, which makes the
         * pipeline disagree with itself about the frame size.
         *
         * A defer thunk is exempt: its slot numbers are the defining frame's,
         * so renumbering them against its own (empty) parameter list would
         * make it read the wrong local. */
        if (!jaiFunctionIsDeferThunk(fn)) jaiOptCoalesceSlots(fn);
    }

#ifdef JAI_DEBUG
    /* Only blame the optimiser for damage it caused: a chunk that arrived
     * malformed is the emitter's problem and is reported by the test suite. */
    if (wasValid && !jaiVerifyChunk(fn, after, sizeof after)) {
        JAI_PANIC("optimiser produced an invalid chunk: %s", after);
    }
    (void)before;
#endif
}

void jaiOptimize(ObjFunction *fn, const CodegenOptions *opts) {
    optimizeFunction(fn, opts, 0);
}

/* ------------------------------------------------------------------ */
/* Verification                                                         */
/* ------------------------------------------------------------------ */

static bool verifyFail(char *buf, size_t size, const char *fmt, ...) JAI_PRINTF(3, 4);

static bool verifyFail(char *buf, size_t size, const char *fmt, ...) {
    if (buf != NULL && size > 0) {
        va_list args;
        va_start(args, fmt);
        vsnprintf(buf, size, fmt, args);
        va_end(args);
    }
    return false;
}

/* Constant-pool operands of `op`, as byte offsets into its operand run. */
static int constOperands(uint8_t op, int *out) {
    switch (op) {
    case OP_CONST:
    case OP_DEF_GLOBAL:
    case OP_GET_GLOBAL:
    case OP_SET_GLOBAL:
    case OP_CLASS:
    case OP_IMPL_TRAIT:
    case OP_METHOD:
    case OP_FIELD_DEF:
    case OP_GET_FIELD:
    case OP_SET_FIELD:
    case OP_GET_SUPER:
    case OP_NEW:
    case OP_ENUM_NEW:
    case OP_IS_INSTANCE:
    case OP_PUSH_DEFER:
    case OP_MATCH_EXC:
    case OP_MATCH_CONST:
    case OP_MATCH_TYPE:
    case OP_MATCH_FIELDS:
    case OP_IMPORT:
    case OP_IMPORT_FROM:
    case OP_EXPORT:
    case OP_ASSERT_FAIL:
    case OP_TYPE_GUARD:
    case OP_INVOKE:
    case OP_SUPER_INVOKE:
    case OP_CLOSURE:
        out[0] = 0;
        return 1;
    case OP_GET_MODULE:
    case OP_MATCH_RANGE:
        out[0] = 0;
        out[1] = 3;
        return 2;
    case OP_CALL_KW:
        out[0] = 1;
        return 1;
    case OP_JUMP_IF_CMP_LOCAL_K:   /* K follows the u8 comparison and u16 slot */
        out[0] = 3;
        return 1;
    case OP_PUSH_HANDLER:
    case OP_GET_FIELD_LOCAL:   /* the name follows the u16 slot */
        out[0] = 2;
        return 1;
    case OP_FORMAT:            /* the name follows the count and the mask */
        out[0] = 4;
        return 1;
    default:
        return 0;
    }
}

#define cacheOperandAt(op) jaiOpCacheOperand((OpCode)(op))

/* Number of entries in the tuple constant `k`, or -1 when it is not a tuple. */
static int tupleArity(const Chunk *chunk, uint32_t k) {
    if (k >= (uint32_t)chunk->constants.count) return -1;
    Value v = chunk->constants.data[k];
    if (!IS_TUPLE(v)) return -1;
    return (int)AS_TUPLE(v)->count;
}

static int popcount3(unsigned flags) {
    int n = 0;
    for (unsigned bit = 1; bit <= 4; bit <<= 1) {
        if (flags & bit) n++;
    }
    return n;
}

typedef struct {
    int  pops, pushes;
    bool known;
} Effect;

/* Pops and pushes of the instruction at `offset` on its fall-through path.
 * Where only the net effect is tabulated in chunk.c, pops is a lower bound;
 * that keeps the depth check free of false positives. */
static Effect fallEffect(const Chunk *chunk, int offset, uint8_t op,
                         const uint8_t *a) {
    Effect e = {0, 0, true};
    (void)offset;

    switch (op) {
    case OP_ADD: case OP_SUB: case OP_MUL: case OP_DIV: case OP_FLOORDIV:
    case OP_MOD: case OP_POW: case OP_ADD_WRAP: case OP_SUB_WRAP:
    case OP_MUL_WRAP: case OP_BAND: case OP_BOR: case OP_BXOR: case OP_SHL:
    case OP_SHR: case OP_EQ: case OP_NE: case OP_LT: case OP_LE: case OP_GT:
    case OP_GE: case OP_IS: case OP_IS_NOT: case OP_IN: case OP_NOT_IN:
    case OP_CONCAT: case OP_GET_INDEX:
        e.pops = 2;
        e.pushes = 1;
        return e;
    /* OP_ENUM_TAG is not here: like OP_ENUM_FIELD it peeks and pushes, so the
     * enum stays underneath and the table's +1 is the truth. */
    case OP_NEG: case OP_POS: case OP_BNOT: case OP_NOT:
    case OP_GET_ITER: case OP_TYPE_GUARD: case OP_IS_INSTANCE:
    case OP_TO_FLOAT:
        e.pops = 1;
        e.pushes = 1;
        return e;
    case OP_SET_INDEX:
        e.pops = 3;
        return e;
    case OP_POPN:
        e.pops = a[0];
        return e;
    case OP_CALL:
    case OP_CALL_SPREAD:
    case OP_TAIL_CALL:
        e.pops = (int)a[0] + 1;
        e.pushes = 1;
        return e;
    case OP_CALL_KW: {
        int kw = tupleArity(chunk, jaiReadU24(a + 1));
        if (kw < 0) { e.known = false; return e; }
        e.pops = (int)a[0] + kw + 1;
        e.pushes = 1;
        return e;
    }
    case OP_INVOKE:
        e.pops = (int)a[3] + 1;   /* receiver plus arguments */
        e.pushes = 1;
        return e;
    case OP_SUPER_INVOKE:
        e.pops = (int)a[3] + 1;
        e.pushes = 1;
        return e;
    case OP_NEW:
        e.pops = a[3];
        e.pushes = 1;
        return e;
    case OP_ENUM_NEW:
        e.pops = a[4];
        e.pushes = 1;
        return e;
    case OP_BUILD_LIST:
    case OP_BUILD_SET:
    case OP_BUILD_TUPLE:
        e.pops = jaiReadU16(a);
        e.pushes = 1;
        return e;
    case OP_BUILD_DICT:
        e.pops = 2 * (int)jaiReadU16(a);
        e.pushes = 1;
        return e;
    case OP_GET_SLICE:
        e.pops = 1 + popcount3(a[0]);
        e.pushes = 1;
        return e;
    case OP_SET_SLICE:
        e.pops = 2 + popcount3(a[0]);
        return e;
    case OP_UNPACK:
        e.pops = 1;
        e.pushes = a[0];
        return e;
    case OP_FOR_ITER:
        e.pushes = 1;   /* the iterator stays under the produced value */
        return e;
    case OP_FORMAT:
        e.pops = a[0];
        e.pushes = 1;
        return e;
    case OP_MATCH_FIELDS: {
        int fields = tupleArity(chunk, jaiReadU24(a));
        if (fields < 0) { e.known = false; return e; }
        e.pushes = fields;
        return e;
    }
    default:
        break;
    }

    int net = jaiOpStackEffect((OpCode)op);
    if (net == INT32_MIN) {
        e.known = false;
        return e;
    }
    e.pops = net < 0 ? -net : 0;
    e.pushes = e.pops + net;
    return e;
}

bool jaiVerifyChunk(const ObjFunction *fn, char *errBuf, size_t errBufSize) {
    if (errBuf != NULL && errBufSize > 0) errBuf[0] = '\0';
    if (fn == NULL) return verifyFail(errBuf, errBufSize, "function is null");

    const Chunk *chunk = &fn->chunk;
    if (chunk->code == NULL || chunk->count <= 0) {
        return verifyFail(errBuf, errBufSize, "chunk contains no code");
    }

    int n = chunk->count;
    bool ok = true;
    bool *boundary = JAI_ALLOC_ZEROED(bool, n + 1);
    int *depth = JAI_ALLOC(int, n + 1);
    int *work = JAI_ALLOC(int, n + 1);
    int workCount = 0;
    for (int i = 0; i <= n; i++) depth[i] = -1;

#define VFAIL(...)                                                             \
    do {                                                                       \
        ok = verifyFail(errBuf, errBufSize, __VA_ARGS__);                      \
        goto done;                                                             \
    } while (0)

    /* Pass 1: decode linearly, checking operand widths and operand ranges. */
    for (int offset = 0; offset < n;) {
        boundary[offset] = true;
        uint8_t op = chunk->code[offset];
        if (op >= OP_COUNT) {
            VFAIL("offset %d: opcode 0x%02x is not a valid opcode (OP_COUNT = %d)",
                  offset, (unsigned)op, (int)OP_COUNT);
        }

        int operands = jaiOpOperandSize((OpCode)op);
        if (op == OP_CLOSURE) {
            if (offset + 4 > n) {
                VFAIL("offset %d: OP_CLOSURE constant index runs past the end "
                      "of the code", offset);
            }
            uint32_t k = jaiReadU24(chunk->code + offset + 1);
            if (k >= (uint32_t)chunk->constants.count) {
                VFAIL("offset %d: OP_CLOSURE constant index %u is out of range "
                      "(%d constants)", offset, (unsigned)k,
                      chunk->constants.count);
            }
            if (!IS_FUNCTION(chunk->constants.data[k])) {
                VFAIL("offset %d: OP_CLOSURE constant %u is not a function",
                      offset, (unsigned)k);
            }
            operands = 3 + 3 * (int)AS_FUNCTION(chunk->constants.data[k])->upvalueCount;
        }
        if (operands < 0) {
            VFAIL("offset %d: %s has no known operand width", offset,
                  jaiOpName((OpCode)op));
        }
        if (offset + 1 + operands > n) {
            VFAIL("offset %d: %s reads %d operand byte%s past the end of the "
                  "code (%d bytes)", offset, jaiOpName((OpCode)op),
                  offset + 1 + operands - n,
                  offset + 1 + operands - n == 1 ? "" : "s", n);
        }

        const uint8_t *a = chunk->code + offset + 1;

        int constAt[2];
        int constCount = constOperands(op, constAt);
        for (int k = 0; k < constCount; k++) {
            uint32_t index = jaiReadU24(a + constAt[k]);
            if (index >= (uint32_t)chunk->constants.count) {
                VFAIL("offset %d: %s constant index %u is out of range "
                      "(%d constants)", offset, jaiOpName((OpCode)op),
                      (unsigned)index, chunk->constants.count);
            }
        }

        int cacheAt = cacheOperandAt(op);
        if (cacheAt >= 0) {
            uint16_t cache = jaiReadU16(a + cacheAt);
            if (cache >= (uint16_t)chunk->cacheCount) {
                VFAIL("offset %d: %s inline-cache index %u is out of range "
                      "(%d caches)", offset, jaiOpName((OpCode)op),
                      (unsigned)cache, chunk->cacheCount);
            }
        }

        int slotAt[2];
        int slotCount = slotOperands(op, slotAt);
        for (int k = 0; k < slotCount; k++) {
            uint16_t slot = jaiReadU16(a + slotAt[k]);
            if (slot >= fn->maxSlots) {
                VFAIL("offset %d: %s uses local slot %u but the frame has %u",
                      offset, jaiOpName((OpCode)op), (unsigned)slot,
                      (unsigned)fn->maxSlots);
            }
        }

        if (op == OP_GET_UPVALUE || op == OP_SET_UPVALUE) {
            if (a[0] >= fn->upvalueCount) {
                VFAIL("offset %d: %s uses upvalue %u but the closure has %u",
                      offset, jaiOpName((OpCode)op), (unsigned)a[0],
                      (unsigned)fn->upvalueCount);
            }
        }

        if (op == OP_CLOSURE) {
            int upvalues = (operands - 3) / 3;
            for (int u = 0; u < upvalues; u++) {
                const uint8_t *desc = a + 3 + u * 3;
                uint16_t index = jaiReadU16(desc + 1);
                if (desc[0] != 0 && index >= fn->maxSlots) {
                    VFAIL("offset %d: OP_CLOSURE captures local slot %u but the "
                          "frame has %u", offset, (unsigned)index,
                          (unsigned)fn->maxSlots);
                }
                if (desc[0] == 0 && index >= fn->upvalueCount) {
                    VFAIL("offset %d: OP_CLOSURE captures upvalue %u but the "
                          "closure has %u", offset, (unsigned)index,
                          (unsigned)fn->upvalueCount);
                }
            }
        }
        offset += 1 + operands;
    }
    boundary[n] = true;

    /* Pass 2: every code address lands on an instruction boundary. */
    for (int offset = 0; offset < n;) {
        uint8_t op = chunk->code[offset];
        int operands = jaiOpOperandSize((OpCode)op);
        if (op == OP_CLOSURE) {
            uint32_t k = jaiReadU24(chunk->code + offset + 1);
            operands = 3 + 3 * (int)AS_FUNCTION(chunk->constants.data[k])->upvalueCount;
        }
        int at = branchOperandAt(op);
        if (at >= 0) {
            int16_t rel = jaiReadI16(chunk->code + offset + 1 + at);
            long target = (long)offset + 1 + operands + rel;
            if (target < 0 || target >= n) {
                VFAIL("offset %d: %s targets %ld, outside the code (%d bytes)",
                      offset, jaiOpName((OpCode)op), target, n);
            }
            if (!boundary[target]) {
                VFAIL("offset %d: %s targets %ld, which is not an instruction "
                      "boundary", offset, jaiOpName((OpCode)op), target);
            }
        }
        offset += 1 + operands;
    }

    /* Pass 3: exception table and default-thunk entry points. */
    for (int e = 0; e < (int)fn->exceptionCount; e++) {
        const ExceptionEntry *entry = fn->exceptions;
        if (entry == NULL) {
            VFAIL("exception table has %u entries but no storage",
                  (unsigned)fn->exceptionCount);
        }
        entry += e;
        if (entry->start > (uint32_t)n || !boundary[entry->start]) {
            VFAIL("exception entry %d: start %u is not an instruction boundary",
                  e, (unsigned)entry->start);
        }
        if (entry->end > (uint32_t)n || !boundary[entry->end]) {
            VFAIL("exception entry %d: end %u is not an instruction boundary",
                  e, (unsigned)entry->end);
        }
        if (entry->start >= entry->end) {
            VFAIL("exception entry %d: empty protected region [%u, %u)", e,
                  (unsigned)entry->start, (unsigned)entry->end);
        }
        if (entry->handler >= (uint32_t)n || !boundary[entry->handler]) {
            VFAIL("exception entry %d: handler %u is not an instruction "
                  "boundary", e, (unsigned)entry->handler);
        }
        if (entry->typeConst != UINT32_MAX &&
            entry->typeConst >= (uint32_t)chunk->constants.count) {
            VFAIL("exception entry %d: type constant %u is out of range "
                  "(%d constants)", e, (unsigned)entry->typeConst,
                  chunk->constants.count);
        }
    }
    for (int d = 0; d < (int)fn->defaultCount && fn->defaultOffsets != NULL; d++) {
        uint32_t at = fn->defaultOffsets[d];
        if (at >= (uint32_t)n || !boundary[at]) {
            VFAIL("default thunk %d: offset %u is not an instruction boundary",
                  d, (unsigned)at);
        }
    }

    /* Pass 4: stack depth. Every path into an offset must agree on the depth
     * there, and no path may pop more than it has. */
    depth[0] = 0;
    work[workCount++] = 0;

    /* Entry points the VM jumps to directly are seeded between drains, and a
     * seeded block can contain further entry points, so this repeats until a
     * round adds nothing. Precise seeds (a dynamic handler, whose depth is the
     * depth of its own PUSH) go first; the imprecise ones — an exception-table
     * handler and a default thunk, both entered at depth 0 — only when nothing
     * precise is left, so a target that turns out to be reachable keeps its
     * flow-derived depth. */
    for (;;) {
        while (workCount > 0) {
            int offset = work[--workCount];
            int here = depth[offset];
            uint8_t op = chunk->code[offset];
            const uint8_t *a = chunk->code + offset + 1;

            int operands = jaiOpOperandSize((OpCode)op);
            if (op == OP_CLOSURE) {
                uint32_t k = jaiReadU24(a);
                operands = 3 + 3 * (int)AS_FUNCTION(chunk->constants.data[k])->upvalueCount;
            }
            int next = offset + 1 + operands;

            Effect e = fallEffect(chunk, offset, op, a);
            if (!e.known) continue;   /* unmodelled: stop, never guess */
            if (here < e.pops) {
                VFAIL("offset %d: %s pops %d value%s from a stack of depth %d",
                      offset, jaiOpName((OpCode)op), e.pops,
                      e.pops == 1 ? "" : "s", here);
            }
            int after = here - e.pops + e.pushes;

            int fallTo = -1, fallDepth = after;
            int jumpTo = -1, jumpDepth = after;

            if (opFallsThrough(op)) {
                /* OP_TAIL_CALL is modelled as falling through so that a
                 * following RETURN is verified, but it is also a legitimate
                 * last instruction. */
                if (next < n) {
                    fallTo = next;
                } else if (op != OP_TAIL_CALL) {
                    VFAIL("offset %d: %s is the last instruction and execution "
                          "falls off the end of the code", offset,
                          jaiOpName((OpCode)op));
                }
            }

            int at = branchOperandAt(op);
            if (at >= 0 && opBranchIsEdge(op)) {
                jumpTo = (int)((long)offset + 1 + operands +
                               jaiReadI16(chunk->code + offset + 1 + at));
                switch (op) {
                case OP_JUMP:
                case OP_LOOP:
                    jumpDepth = here;
                    break;
                case OP_JUMP_IF_FALSE:
                case OP_JUMP_IF_TRUE:
                    jumpDepth = here - 1;
                    break;
                case OP_JUMP_IF_CMP_FALSE:
                    /* Both operands go, on either path. */
                    jumpDepth = here - 2;
                    break;
                case OP_FOR_ITER:
                case OP_FOR_ITER_BIND:
                    /* Exhaustion pops the iterator and leaves the loop. */
                    jumpDepth = here - 1;
                    break;
                case OP_MATCH_FIELDS:
                    jumpDepth = here;   /* nothing destructured on no-match */
                    break;
                default:
                    jumpDepth = here;   /* the *_KEEP forms and MATCH_* peek */
                    break;
                }
                if (jumpDepth < 0) {
                    VFAIL("offset %d: %s leaves a negative stack depth on its "
                          "taken path", offset, jaiOpName((OpCode)op));
                }
            }

            for (int edge = 0; edge < 2; edge++) {
                int to = edge == 0 ? fallTo : jumpTo;
                int d = edge == 0 ? fallDepth : jumpDepth;
                if (to < 0) continue;
                if (d < 0) {
                    VFAIL("offset %d: stack depth %d is negative after %s",
                          offset, d, jaiOpName((OpCode)op));
                }
                if (depth[to] < 0) {
                    depth[to] = d;
                    work[workCount++] = to;
                } else if (depth[to] != d) {
                    VFAIL("offset %d: stack depth %d here disagrees with depth "
                          "%d reached by another path (from %d, %s)", to, d,
                          depth[to], offset, jaiOpName((OpCode)op));
                }
            }
        }

        /* A dynamic handler records the stack top live when it was pushed and
         * the unwinder restores exactly that, so its target runs at the depth
         * of the push — not at 0, the way an exception-table handler does. */
        int added = 0;
        for (int offset = 0; offset < n;) {
            uint8_t op = chunk->code[offset];
            int operands = jaiOpOperandSize((OpCode)op);
            if (op == OP_CLOSURE) {
                uint32_t k = jaiReadU24(chunk->code + offset + 1);
                operands = 3 + 3 *
                    (int)AS_FUNCTION(chunk->constants.data[k])->upvalueCount;
            }
            if ((op == OP_PUSH_FINALLY || op == OP_PUSH_HANDLER) &&
                depth[offset] >= 0) {
                int at = branchOperandAt(op);
                int to = (int)((long)offset + 1 + operands +
                               jaiReadI16(chunk->code + offset + 1 + at));
                if (depth[to] < 0) {
                    depth[to] = depth[offset];
                    work[workCount++] = to;
                    added++;
                }
            }
            offset += 1 + operands;
        }
        if (added > 0) continue;

        /* Exactly one imprecise seed per round. Seeding them all at once loses
         * the staging: the code a handler falls into can contain the
         * PUSH_FINALLY of a *second* region, and that region's handler is then
         * already pinned at 0 when the precise pass finally gets to look at it.
         * A `finally` inside a loop is the case in the wild — its handler sits
         * under the loop's iterator, at depth 1, and the table entry that
         * shadows it says 0. */
        for (int e = 0; e < (int)fn->exceptionCount && added == 0; e++) {
            uint32_t handler = fn->exceptions[e].handler;
            if (depth[handler] < 0) {
                depth[handler] = 0;
                work[workCount++] = (int)handler;
                added++;
            }
        }
        for (int d = 0; d < (int)fn->defaultCount && added == 0 &&
                        fn->defaultOffsets != NULL; d++) {
            uint32_t at = fn->defaultOffsets[d];
            if (depth[at] < 0) {
                depth[at] = 0;
                work[workCount++] = (int)at;
                added++;
            }
        }
        if (added == 0) break;
    }
#undef VFAIL

done:
    JAI_FREE_ARRAY(bool, boundary, n + 1);
    JAI_FREE_ARRAY(int, depth, n + 1);
    JAI_FREE_ARRAY(int, work, n + 1);
    return ok;
}
