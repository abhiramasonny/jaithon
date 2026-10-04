/* jit_operand.c -- reading a literal back out of the instruction stream, and
 * the strength reductions that only apply once one has been read. */

#include "vm/jit/jit_arm64.h"
#include "vm/jit/jit_field_read.h"
#include "vm/bytecode/verify.h"

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "vm/jit/jit_internal.h"

#if (defined(__aarch64__) || defined(__arm64__))

/* ---- literal operands -------------------------------------------------- */

/* Whether anything but fall-through can reach `off`: reading the previous instruction's literal
 * without this check is a miscompile, not a decline -- `x // (if c {2} else {4})` puts OP_INT right before OP_FLOORDIV *and* a jump from the other arm onto it. Scans the WHOLE chunk, not just fixups already emitted (a back edge compiles after its target is walked, so the fixup list would miss loop tops) -- and handler/finally addresses too, since the unwinder can resume there with a stack this walk never saw. */
bool offsetIsBranchTarget(const Chunk *c, uint32_t off) {
    for (int at = 0; at < c->count;) {
        int len = instructionLength(c, at);
        if (len <= 0) return true;      /* undecodable: assume the worst */
        int rel = jaiOpBranchOperandAt(c->code[at]);
        if (rel >= 0) {
            int16_t jump = jaiReadI16(c->code + at + 1 + rel);
            /* Every branch operand is measured from the end of the
             * instruction, which is what `at + len` is. */
            if ((int32_t)(at + len) + jump == (int32_t)off) return true;
        }
        at += len;
    }
    return false;
}

/* OP_INT carries its value inline, OP_CONST names a pool entry -- the only two ways a literal reaches
 * the stack. The adjacency check is belt-and-braces (the walk is linear) but a real bug: OP_FORMAT once advanced `off` by nine instead of ten, and this is what would have caught it. */
bool literalIntOperand(const ObjFunction *fn, int prevOff, int off,
                              int64_t *out) {
    if (prevOff < 0 || prevOff >= off) return false;
    const Chunk *c = &fn->chunk;
    if (prevOff + instructionLength(c, prevOff) != off) return false;
    uint8_t prev = c->code[prevOff];
    if (prev == OP_INT) {
        *out = jaiReadI16(c->code + prevOff + 1);
    } else if (prev == OP_CONST) {
        uint32_t idx = jaiReadU24(c->code + prevOff + 1);
        if (idx >= (uint32_t)c->constants.count) return false;
        Value k = c->constants.data[idx];
        if (!IS_INT(k)) return false;
        *out = AS_INT(k);
    } else {
        return false;
    }
    return !offsetIsBranchTarget(c, (uint32_t)off);
}

/* `k` is 2^shift, for a shift this can name. Positive only: floor division by
 * a negative power of two is not a shift, and `k` is at most 2^62 because 2^63
 * does not fit in a positive int64. */
/* Collapses the general "add divisor back if remainder is non-zero and signs differ" (7 instructions)
 * to 2 when the divisor's sign is known at compile time: msub leaves |r| < |d| with r's sign following the dividend's, so for a positive divisor the whole test is "r < 0" (one bit), and for a negative one "r > 0". `r + d` cannot overflow since |r| < |d| puts the sum strictly between -|d| and |d|. */
void emitFloorFixup(Emit *e, unsigned rrem, unsigned rd,
                           bool signKnown, int64_t divisor, uint32_t fixup) {
    if (signKnown && divisor > 0) {
        emit(e, jaiA64Tbz(rrem, 63u, 2));
        emit(e, fixup);
        return;
    }
    if (signKnown) {
        emit(e, jaiA64SubsXImm(31, rrem, 0));
        emit(e, jaiA64BCond(JAI_A64_LE, 2));
        emit(e, fixup);
        return;
    }
    emit(e, jaiA64SubsXImm(31, rrem, 0));
    emit(e, jaiA64BCond(JAI_A64_EQ, 5));
    emit(e, jaiA64EorX(JIT_SCRATCH_D, rrem, rd));
    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_D, 0));
    emit(e, jaiA64BCond(JAI_A64_GE, 2));
    emit(e, fixup);
}

/* JAITHON_JIT_FLOOR_COLD=0 puts the floor correction of `x % k` and `x // k`
 * for a literal k > 0 back inline, behind a `tbz`. On by default.
 *
 * emitFloorFixup's `tbz` over one instruction is TAKEN for every non-negative
 * dividend -- the common case -- and a core retires one taken branch a cycle,
 * so a loop taking a remainder carried two where its peer carries one:
 * loop_sum's `i % 7` ran 3.3 cycles an iteration against 2.5 without it.
 * Branch-free (`and`/`add` on the smeared sign) fixed that and cost
 * poly_dispatch 9%: there the remainder feeds the next call's argument, and
 * the correction then sits on the dependency chain where the predicted branch
 * had cost nothing. Out of line keeps both -- the compare falls through when
 * the remainder is non-negative, with nothing on the chain, and only a
 * negative one leaves to apply the correction and come back. */
static bool floorColdOn(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("JAITHON_JIT_FLOOR_COLD");
        on = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return on != 0;
}

/* `if rrem < 0 { insn }`, with `insn` emitted after the body (emitGrowStubs)
 * and a branch back. False, emitting nothing, when the switch is off or the
 * table is full; the caller then emits the inline form. `insn` must touch only
 * registers, and nothing it reads may change between here and the stub --
 * which is immediate, since the stub is the branch's only successor. */
bool emitColdFixup(Emit *e, unsigned rrem, uint32_t insn) {
    if (!floorColdOn()) return false;
    if (e->coldCount >= JIT_MAX_COLD) return false;
    if (e->fixupCount >= JIT_MAX_FIXUPS) return false;
    unsigned ci = e->coldCount++;
    emit(e, jaiA64SubsXImm(31, rrem, 0));
    e->fixups[e->fixupCount].instIndex    = (int)e->count;
    e->fixups[e->fixupCount].targetOffset = FIXUP_COLD - ci;
    e->fixups[e->fixupCount].conditional  = true;
    e->fixups[e->fixupCount].depth        = -1;
    e->fixupCount++;
    emit(e, jaiA64BCond(JAI_A64_LT, 0));
    e->cold[ci].stub     = -1;
    e->cold[ci].returnTo = (int)e->count;
    e->cold[ci].insn     = insn;
    return true;
}

bool powerOfTwoShift(int64_t k, unsigned *shift) {
    if (k <= 0) return false;
    uint64_t u = (uint64_t)k;
    if ((u & (u - 1u)) != 0u) return false;
    unsigned s = 0;
    while ((u >> s) != 1u) s++;
    *shift = s;
    return true;
}

#endif /* __aarch64__ || __arm64__ */
