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
/* The answer above is a property of the chunk alone, and a compile asks it of
 * the same chunk once per instruction while a stack proof or a field memo is
 * live, and once per skipped instruction in every unarmed walk -- each time
 * decoding the whole chunk. On `fmt --check` that scan was ~4% of the run.
 * So each chunk's targets are decoded once into a bitmap.
 *
 * Valid for ONE compile: jitBranchTargetsReset clears it at the start of every
 * top-level compile. Within a compile no chunk is created, so a freed chunk's
 * code address cannot be handed to another and answer from a stale map; across
 * compiles that is exactly what could happen, so nothing is kept. Several
 * slots, because an inlined callee's chunk is asked about in between. */
enum { BT_SLOTS = 4 };
static struct {
    const uint8_t *code;
    int            count;
    bool           undecodable;
    uint8_t       *bits;
    size_t         cap;
} gBranchTargets[BT_SLOTS];
static unsigned gBranchTargetsNext;

static bool jitBranchTargetCache(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_BRANCH_MAP");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

void jitBranchTargetsReset(void) {
    for (unsigned i = 0; i < BT_SLOTS; i++) gBranchTargets[i].code = NULL;
}

static bool offsetIsBranchTargetScan(const Chunk *c, uint32_t off);

bool offsetIsBranchTarget(const Chunk *c, uint32_t off) {
    if (!jitBranchTargetCache() || c->code == NULL ||
        off > (uint32_t)c->count) {
        return offsetIsBranchTargetScan(c, off);
    }
    unsigned slot = BT_SLOTS;
    for (unsigned i = 0; i < BT_SLOTS; i++) {
        if (gBranchTargets[i].code == c->code &&
            gBranchTargets[i].count == c->count) {
            slot = i;
            break;
        }
    }
    if (slot == BT_SLOTS) {
        slot = gBranchTargetsNext++ % BT_SLOTS;
        size_t need = ((size_t)c->count + 8u) / 8u;
        if (gBranchTargets[slot].cap < need) {
            uint8_t *grown = (uint8_t *)realloc(gBranchTargets[slot].bits, need);
            if (grown == NULL) {
                gBranchTargets[slot].code = NULL;
                return offsetIsBranchTargetScan(c, off);
            }
            gBranchTargets[slot].bits = grown;
            gBranchTargets[slot].cap = need;
        }
        uint8_t *bits = gBranchTargets[slot].bits;
        memset(bits, 0, need);
        bool bad = false;
        for (int at = 0; at < c->count;) {
            int len = instructionLength(c, at);
            if (len <= 0) { bad = true; break; }
            int rel = jaiOpBranchOperandAt(c->code[at]);
            if (rel >= 0) {
                int16_t jump = jaiReadI16(c->code + at + 1 + rel);
                int32_t to = (int32_t)(at + len) + jump;
                if (to >= 0 && to <= c->count) {
                    bits[(uint32_t)to >> 3] |= (uint8_t)(1u << ((uint32_t)to & 7u));
                }
            }
            at += len;
        }
        gBranchTargets[slot].code = c->code;
        gBranchTargets[slot].count = c->count;
        gBranchTargets[slot].undecodable = bad;
    }
    /* An undecodable instruction anywhere made the scan answer yes for every
     * offset it had not already matched -- and so for every offset. */
    if (gBranchTargets[slot].undecodable) return true;
    return (gBranchTargets[slot].bits[off >> 3] >> (off & 7u)) & 1u;
}

static bool offsetIsBranchTargetScan(const Chunk *c, uint32_t off) {
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
