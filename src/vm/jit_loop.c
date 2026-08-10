#include "jit.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "chunk.h"
#include "jit_arm64.h"
#include "vm.h"

/* The compiled loop.
 *
 * One shape, recognised by exact opcode sequence:
 *
 *     JUMP_IF_CMP_LOCAL_K LT, i, K, exit
 *     GET_LOCAL2 acc, i
 *     MOD_INT_CONST d
 *     ADD_BIND acc
 *     INC_LOCAL i, 1
 *     LOOP
 *
 * That is `loop_sum`'s body and nothing else. A pattern matcher is not a
 * compiler and this file does not pretend to be one -- the point is to prove
 * the mechanism end to end on a shape that exists, and then widen.
 *
 * Every write in the body goes to `acc` or `i`, both locals. That is what makes
 * the bail sound: on any guard failure the values the iteration STARTED with
 * are written back and the interpreter re-runs that whole iteration, raising
 * the overflow itself. A body that called, allocated or stored to a field could
 * not be re-run and is refused.
 */

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
    /* 9 + 5 + 3 + 3 + 4 + 3 = 27. The first instruction carries a u8 compare,
     * a u16 slot, a u24 constant index and an i16 jump, which is eight operand
     * bytes and not five -- getting that wrong is why this declined silently on
     * every tick while the loop it was meant to match ran fifty million times. */
    if (q + 3 != at + 27) return false;

    if (kIndex >= (uint32_t)c->constants.count) return false;
    Value k = c->constants.data[kIndex];
    if (!IS_INT(k)) return false;
    out->limit = AS_INT(k);
    out->exitOffset = (uint32_t)((int32_t)at + 9 + exitJump);
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

/* 0 = the loop ran to completion, 1 = it bailed. */
typedef int (*JaiCompiledLoop)(Value *slots);

/* x1 holds the accumulator and x2 the counter for the whole loop. That
 * residency, not the removal of dispatch, is where the win is. */
static void *compileLoop(const LoopShape *shape) {
    JaiCodeArena *arena = jaiJitArena();
    if (arena == NULL || arena->sealed) return NULL;

    unsigned accInt = shape->accSlot * 16 + 8, accTag = shape->accSlot * 16;
    unsigned iInt   = shape->iSlot * 16 + 8,   iTag   = shape->iSlot * 16;

    uint32_t w[64];
    int n = 0;

    /* Both locals must already be ints; the body assumes it throughout. */
    w[n++] = jaiA64LdrW(5, 0, accTag);
    w[n++] = jaiA64SubsXImm(31, 5, VAL_INT);
    int toBailNoWrite1 = n++;
    w[n++] = jaiA64LdrW(5, 0, iTag);
    w[n++] = jaiA64SubsXImm(31, 5, VAL_INT);
    int toBailNoWrite2 = n++;

    w[n++] = jaiA64LdrX(1, 0, accInt);
    w[n++] = jaiA64LdrX(2, 0, iInt);
    n = emitConst(w, n, 3, (uint64_t)shape->limit);
    n = emitConst(w, n, 4, (uint64_t)shape->divisor);

    int top = n;
    w[n++] = jaiA64SubsX(31, 2, 3);        /* cmp i, limit */
    int toDone = n++;
    w[n++] = jaiA64SdivX(5, 2, 4);
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
    if (entry == NULL) return NULL;
    if (!jaiCodeArenaSeal(arena)) return NULL;
    return entry;
}

bool jaiJitEnterLoop(ObjClosure *closure, uint32_t targetOffset) {
    ObjFunction *fn = closure->fn;
    if (fn->jitLoop == NULL) {
        LoopShape shape;
        if (!matchLoop(&fn->chunk, targetOffset, &shape)) return false;
        void *code = compileLoop(&shape);
        if (code == NULL) return false;
        fn->jitLoop = code;
        fn->jitLoopExit = shape.exitOffset;
        fn->jitLoopTop = targetOffset;
    }
    if (fn->jitLoopTop != targetOffset) return false;

    CallFrame *frame = &vm.frames[vm.frameCount - 1];
    int status = ((JaiCompiledLoop)(uintptr_t)fn->jitLoop)(frame->slots);

    /* Both outcomes hand control back to the interpreter and only `ip` differs:
     * past the loop when it finished, at the loop's own top when it bailed,
     * where re-running the iteration is safe. The caller cannot tell which. */
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
