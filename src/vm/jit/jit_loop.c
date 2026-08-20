#include "vm/jit/jit.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "vm/bytecode/chunk.h"
#include "vm/jit/jit_arm64.h"
#include "vm/vm.h"

/* The compiled loop: matches one exact opcode shape (loop_sum's body), not a
 * general compiler. Every write goes to `acc`/`i` locals only, so a guard
 * failure can write back the pre-iteration values and safely re-run in the
 * interpreter; a body that calls, allocates, or writes a field is refused. */

#if defined(__aarch64__) || defined(__arm64__)

typedef struct {
    unsigned accSlot, iSlot;
    int64_t  limit, divisor;
    uint32_t exitOffset;
} LoopShape;

static bool matchLoop(const Chunk *c, uint32_t at, LoopShape *out) {
    const uint8_t *p = c->code;
    if (at + 27 > (uint32_t)c->count) return false;

    if (p[at] != OP_JUMP_IF_CMP_LOCAL_K || p[at + 1] != OP_LT) return false;
    out->iSlot = (unsigned)p[at + 2] | ((unsigned)p[at + 3] << 8);
    uint32_t kIndex = (uint32_t)p[at + 4] | ((uint32_t)p[at + 5] << 8) |
                      ((uint32_t)p[at + 6] << 16);
    int16_t exitJump = (int16_t)((uint16_t)p[at + 7] | ((uint16_t)p[at + 8] << 8));

    uint32_t q = at + 9;
    if (p[q] != OP_GET_LOCAL2) return false;
    out->accSlot = (unsigned)p[q + 1] | ((unsigned)p[q + 2] << 8);
    if (((unsigned)p[q + 3] | ((unsigned)p[q + 4] << 8)) != out->iSlot) return false;

    q += 5;
    if (p[q] != OP_MOD_INT_CONST) return false;
    int16_t divisor = (int16_t)((uint16_t)p[q + 1] | ((uint16_t)p[q + 2] << 8));
    if (divisor <= 0) return false;   /* the emitted remainder assumes positive */
    out->divisor = divisor;

    q += 3;
    if (p[q] != OP_ADD_BIND) return false;
    if (((unsigned)p[q + 1] | ((unsigned)p[q + 2] << 8)) != out->accSlot) return false;

    q += 3;
    if (p[q] != OP_INC_LOCAL) return false;
    if (((unsigned)p[q + 1] | ((unsigned)p[q + 2] << 8)) != out->iSlot) return false;
    if ((int8_t)p[q + 3] != 1) return false;

    q += 4;
    if (p[q] != OP_LOOP) return false;
    /* 9 + 5 + 3 + 3 + 4 + 3 = 27, matching the first instruction's real 8
     * operand bytes (u8 compare, u16 slot, u24 constant, i16 jump); getting
     * this wrong made it decline silently on every tick of a 50M-iteration run. */
    if (q + 3 != at + 27) return false;

    if (kIndex >= (uint32_t)c->constants.count) return false;
    Value k = c->constants.data[kIndex];
    if (!IS_INT(k)) return false;
    out->limit = AS_INT(k);
    out->exitOffset = (uint32_t)((int32_t)at + 9 + exitJump);
    return true;
}

/* The same body behind a `for i in 0..n` head: byte-for-byte what the counted
 * head already compiles, so this is a second entry guard rather than a second
 * code generator. What differs: the limit is a runtime property of the
 * iterator, and the iterator's position has to be written back. */
static bool matchRangeLoop(const Chunk *c, uint32_t at, LoopShape *out) {
    const uint8_t *p = c->code;
    if (at + 19 > (uint32_t)c->count) return false;
    if (p[at] != OP_FOR_ITER_BIND) return false;

    int16_t jump = (int16_t)((uint16_t)p[at + 1] | ((uint16_t)p[at + 2] << 8));
    out->iSlot = (unsigned)p[at + 3] | ((unsigned)p[at + 4] << 8);

    uint32_t q = at + 5;
    if (p[q] != OP_GET_LOCAL2) return false;
    out->accSlot = (unsigned)p[q + 1] | ((unsigned)p[q + 2] << 8);
    if (((unsigned)p[q + 3] | ((unsigned)p[q + 4] << 8)) != out->iSlot) return false;

    q += 5;
    if (p[q] != OP_MOD_INT_CONST) return false;
    int16_t divisor = (int16_t)((uint16_t)p[q + 1] | ((uint16_t)p[q + 2] << 8));
    if (divisor <= 0) return false;
    out->divisor = divisor;

    q += 3;
    if (p[q] != OP_ADD_BIND) return false;
    if (((unsigned)p[q + 1] | ((unsigned)p[q + 2] << 8)) != out->accSlot) return false;

    q += 3;
    if (p[q] != OP_LOOP) return false;
    if (q + 3 != at + 19) return false;

    out->limit = 0;   /* runtime: read from the iterator at entry */
    out->exitOffset = (uint32_t)((int32_t)at + 5 + jump);
    return true;
}

/* movz/movk, skipping halfwords that are already zero. */
static int emitConst(uint32_t *w, int n, unsigned reg, uint64_t value) {
    w[n++] = jaiA64MovzX(reg, (unsigned)(value & 0xffff), 0);
    for (unsigned shift = 1; shift < 4; shift++) {
        unsigned part = (unsigned)((value >> (16 * shift)) & 0xffff);
        if (part != 0) w[n++] = jaiA64MovkX(reg, part, shift);
    }
    return n;
}

/* 0 = the loop ran to completion, 1 = it bailed. `limit` is a parameter
 * rather than baked into the code so one emitted body serves both loop heads:
 * the counted head passes its matched constant, the range head its iterator's
 * limit, known only at entry. */
typedef int (*JaiCompiledLoop)(Value *slots, int64_t limit);

/* x1 holds the accumulator and x2 the counter for the whole loop. That
 * residency, not the removal of dispatch, is where the win is. */

/* The multiply-shift reciprocal for a constant divisor (Hacker's Delight
 * 10-4): sdiv's throughput (not latency) put loop_sum at 4.4 cycles/iter
 * against C++'s 2.2, which gets there via this same reciprocal. Verified
 * against exact arithmetic over 1380 divisor/value pairs (incl. INT64_MIN/MAX)
 * before use, because a reciprocal wrong for one input in 2^64 is a bug
 * nothing would ever reproduce. */
static bool magicDivide(int64_t d, int64_t *magic, unsigned *shift) {
    if (d < 2) return false;          /* 0, 1 and negatives keep sdiv */

    const uint64_t two63 = (uint64_t)1 << 63;
    uint64_t ad = (uint64_t)d;
    uint64_t anc = two63 - 1 - two63 % ad;
    unsigned p = 63;
    uint64_t q1 = two63 / anc, r1 = two63 % anc;
    uint64_t q2 = two63 / ad,  r2 = two63 % ad;
    uint64_t delta;

    do {
        p++;
        q1 *= 2; r1 *= 2;
        if (r1 >= anc) { q1++; r1 -= anc; }
        q2 *= 2; r2 *= 2;
        if (r2 >= ad) { q2++; r2 -= ad; }
        delta = ad - r2;
    } while (q1 < delta || (q1 == delta && r1 == 0));

    *magic = (int64_t)(q2 + 1);
    *shift = p - 64;
    return true;
}

static void *compileLoop(const LoopShape *shape) {
    JaiCodeArena *arena = jaiJitArena();
    /* Unseal rather than decline: the function tier shares this arena and
     * seals it after each compile, so declining would refuse every later loop
     * once a program's first compiled thing was a function. */
    if (arena == NULL || !jaiCodeArenaUnseal(arena)) return NULL;

    unsigned accInt = shape->accSlot * 16 + 8, accTag = shape->accSlot * 16;
    unsigned iInt   = shape->iSlot * 16 + 8,   iTag   = shape->iSlot * 16;

    uint32_t w[96];
    int n = 0;

    /* x1 arrives holding the limit and is about to be reused for the
     * accumulator, so stash it first; doing this after the loads instead
     * produced a loop comparing the accumulator against itself -- a wrong
     * answer, not a crash, which is why this is checked against the
     * interpreter's answer rather than just "did it run". */
    w[n++] = jaiA64MovX(3, 1);

    /* Both locals must already be ints; the body assumes it throughout. */
    w[n++] = jaiA64LdrW(5, 0, accTag);
    w[n++] = jaiA64SubsXImm(31, 5, VAL_INT);
    int toBailNoWrite1 = n++;
    w[n++] = jaiA64LdrW(5, 0, iTag);
    w[n++] = jaiA64SubsXImm(31, 5, VAL_INT);
    int toBailNoWrite2 = n++;

    w[n++] = jaiA64LdrX(2, 0, iInt);
    w[n++] = jaiA64LdrX(1, 0, accInt);
    n = emitConst(w, n, 4, (uint64_t)shape->divisor);

    int64_t  magic = 0;
    unsigned magicShift = 0;
    bool useMagic = magicDivide(shape->divisor, &magic, &magicShift);
    if (useMagic) n = emitConst(w, n, 7, (uint64_t)magic);

    int top = n;
    w[n++] = jaiA64SubsX(31, 2, 3);        /* cmp i, limit */
    int toDone = n++;
    if (useMagic) {
        /* q = i * M >> 64, corrected for a magic that came out negative, then
         * shifted and rounded toward zero by adding back its own sign bit. */
        w[n++] = jaiA64SmulhX(5, 2, 7);
        if (magic < 0) w[n++] = jaiA64AddX(5, 5, 2);
        if (magicShift != 0) w[n++] = jaiA64AsrX(5, 5, magicShift);
        w[n++] = jaiA64LsrX(6, 5, 63);
        w[n++] = jaiA64AddX(5, 5, 6);
    } else {
        w[n++] = jaiA64SdivX(5, 2, 4);
    }
    w[n++] = jaiA64MsubX(5, 5, 4, 2);      /* x5 = i % d */
    w[n++] = jaiA64AddsX(6, 1, 5);
    int toBailWrite = n++;
    w[n++] = jaiA64MovX(1, 6);
    w[n++] = jaiA64AddXImm(2, 2, 1);
    w[n] = jaiA64B(top - n); n++;

    int done = n;
    w[n++] = jaiA64StrX(1, 0, accInt);
    w[n++] = jaiA64StrX(2, 0, iInt);
    w[n++] = jaiA64MovzX(0, 0, 0);
    w[n++] = jaiA64Ret();

    /* x1 and x2 still hold what this iteration started with: the accumulator is
     * updated only after the overflow check and the counter after that. */
    int bailWrite = n;
    w[n++] = jaiA64StrX(1, 0, accInt);
    w[n++] = jaiA64StrX(2, 0, iInt);
    w[n++] = jaiA64MovzX(0, 1, 0);
    w[n++] = jaiA64Ret();

    int bailNoWrite = n;
    w[n++] = jaiA64MovzX(0, 1, 0);
    w[n++] = jaiA64Ret();

    w[toBailNoWrite1] = jaiA64BCond(JAI_A64_NE, bailNoWrite - toBailNoWrite1);
    w[toBailNoWrite2] = jaiA64BCond(JAI_A64_NE, bailNoWrite - toBailNoWrite2);
    w[toDone]         = jaiA64BCond(JAI_A64_GE, done - toDone);
    w[toBailWrite]    = jaiA64BCond(JAI_A64_VS, bailWrite - toBailWrite);

    uint8_t *entry = jaiCodeArenaWrite(arena, w, (size_t)n * sizeof w[0]);
    if (entry == NULL) {
        /* Seal before giving up. The unseal above took the execute bit off
         * every function already in the arena, and the write fails exactly
         * when the arena is full -- so returning here without re-sealing
         * leaves the whole back catalogue unexecutable. See `arenaEmit` in
         * jit_func.c for what that looked like when it happened. */
        jaiCodeArenaSeal(arena);
        return NULL;
    }
    if (getenv("JAI_JIT_TRACE")) {
        fprintf(stderr, "[jit] loop entry %p (mod 64 = %u)\n", (void *)entry,
                (unsigned)((uintptr_t)entry & 63u));
    }
    if (!jaiCodeArenaSeal(arena)) return NULL;
    return entry;
}

bool jaiJitEnterLoop(ObjClosure *closure, uint32_t targetOffset) {
    ObjFunction *fn = closure->fn;
    if (fn->jitLoop == NULL) {
        LoopShape shape;
        uint8_t kind = 0;
        if (matchLoop(&fn->chunk, targetOffset, &shape)) {
            kind = 0;
        } else if (matchRangeLoop(&fn->chunk, targetOffset, &shape)) {
            kind = 1;
        } else {
            return false;
        }
        void *code = compileLoop(&shape);
        if (code == NULL) return false;
        fn->jitLoop = code;
        fn->jitLoopExit = shape.exitOffset;
        fn->jitLoopTop = targetOffset;
        fn->jitLoopLimit = shape.limit;
        fn->jitLoopKind = kind;
    }
    if (fn->jitLoopTop != targetOffset) return false;

    CallFrame *frame = &vm.frames[vm.frameCount - 1];
    int64_t limit = fn->jitLoopLimit;
    ObjIter *iter = NULL;
    unsigned iSlot = 0;

    if (fn->jitLoopKind == 1) {
        /* The iterator is what OP_FOR_ITER_BIND peeks; guarded here rather
         * than at compile time since it's all a runtime fact. Only a range
         * from zero in unit steps makes the yielded value equal the index,
         * which is what lets the counted body run unchanged. */
        Value it = vm.stackTop[-1];
        if (!IS_ITER(it)) return false;
        iter = AS_ITER(it);
        if (iter->kind != ITER_RANGE || !IS_RANGE(iter->source)) return false;
        ObjRange *r = AS_RANGE(iter->source);
        if (r->start != 0 || r->step != 1) return false;

        limit = iter->limit;
        iSlot = (unsigned)((fn->chunk.code[targetOffset + 3]) |
                           (fn->chunk.code[targetOffset + 4] << 8));
        /* The body reads the loop variable from its slot, so seed it. */
        frame->slots[iSlot] = INT_VAL(iter->index);
    }

    int status = ((JaiCompiledLoop)(uintptr_t)fn->jitLoop)(frame->slots, limit);

    if (iter != NULL) {
        /* Completed: the iterator is exhausted and the slot holding `limit`
         * is dead (the `for` binding is loop-scoped). Bailed: that iteration
         * never finished, so it must be yielded again. */
        iter->index = (status == 0) ? limit : AS_INT(frame->slots[iSlot]);
    }

    /* Both outcomes return to the interpreter with only `ip` differing: past
     * the loop when finished, at the loop's own top when bailed (safe to
     * re-run); the caller cannot tell which. */
    frame->ip = fn->chunk.code +
                (status == 0 ? fn->jitLoopExit : fn->jitLoopTop);
    return true;
}

#else   /* not arm64 */

bool jaiJitEnterLoop(ObjClosure *closure, uint32_t targetOffset) {
    (void)closure;
    (void)targetOffset;
    return false;
}

#endif
