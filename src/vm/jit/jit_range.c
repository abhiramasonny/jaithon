/* jit_range.c -- what a dominating comparison proves about an int local, so a
 * step that cannot overflow is not checked for it.
 *
 * `while i < n { ...; i += 1 }` checks `i + 1` for overflow on every
 * iteration, and `if n < 2 { return n } return fib(n - 1) + fib(n - 2)` checks
 * both subtractions on every call -- although the comparison the program just
 * made already rules the overflow out: `i < n` puts `i` at most INT64_MAX - 1,
 * and `n >= 2` puts `n - 2` nowhere near INT64_MIN. A not-taken `b.vs` is
 * cheap but not free on this core: compares and branches are what a tight
 * loop is throughput-bound on (see the memory note on instruction cost).
 *
 * The proof is read off the BYTECODE, not the emitter's model, so it does not
 * depend on any arm keeping a fact up to date:
 *
 *   - From the instruction doing the arithmetic, walk back through the
 *     control-flow graph (jaiChunkCfg, the verifier's own) along blocks with
 *     exactly ONE predecessor. Every path to the arithmetic therefore runs
 *     through every block on that chain.
 *   - No instruction on the chain may name the local in anything but a read
 *     (the verifier's slot-operand table decides what names a slot), so its
 *     value at the arithmetic is its value at every comparison on the chain.
 *   - A comparison on the chain whose edge into the chain is known -- the
 *     fall-through of a jump-if-false is the TRUE edge, its target the FALSE
 *     one -- contributes a bound, and only when both operands are ints by the
 *     interpreter's own semantics: a literal int, or a local this compile holds
 *     as an int on every path.
 *
 * Refused outright, rather than reasoned about: a function with an exception
 * handler (a handler is entered from every instruction of its region, edges the
 * graph does not hold), one with default-argument thunks (more entry points),
 * one that makes closures (a capture can write a local behind the walk's back),
 * inlined bodies (their slot numbers are the callee's), and an OSR form whose
 * entry lies on the chain (it enters there without having made the
 * comparison).
 *
 * JAITHON_JIT_RANGE_FACTS=0 keeps every overflow check. */

#include "vm/jit/jit_arm64.h"
#include "vm/jit/jit_field_read.h"
#include "vm/bytecode/verify.h"
#include "vm/vm.h"

#include <stdint.h>
#include <stdlib.h>
#include "vm/jit/jit_internal.h"

#if (defined(__aarch64__) || defined(__arm64__))

static bool rangeFactsOn(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("JAITHON_JIT_RANGE_FACTS");
        on = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return on != 0;
}

/* Whether `op` at `off` names `slot` in anything but a read. The verifier's
 * table lists every operand that names a slot; the opcodes below are the ones
 * known to only READ the slots they name. Everything else that names `slot`
 * counts as a write, which costs a fact, never an answer. */
static bool writesSlot(const Chunk *c, int off, unsigned slot) {
    uint8_t op = c->code[off];
    switch (op) {
    case OP_GET_LOCAL: case OP_GET_LOCAL2: case OP_GET_FIELD_LOCAL:
    case OP_ADD_LOCALS: case OP_ADD_INT_CONST: case OP_SUB_INT_CONST:
    case OP_MUL_INT_CONST: case OP_CMP_LOCAL_CONST_LT:
    case OP_JUMP_IF_CMP_LOCAL_K:
        return false;
    default:
        break;
    }
    int at[3];
    int n = jaiOpSlotOperands(op, at);
    for (int i = 0; i < n; i++) {
        if (jaiReadU16(c->code + off + 1 + at[i]) == slot) return true;
    }
    return false;
}

static bool hasClosures(const Chunk *c) {
    for (int off = 0; off < c->count;) {
        int len = instructionLength(c, off);
        if (len <= 0) return true;
        if (c->code[off] == OP_CLOSURE || c->code[off] == OP_CLOSE_UPVALUE) {
            return true;
        }
        off += len;
    }
    return false;
}

/* A local whose value is an int whenever compiled code reads it. */
static bool intLocal(const Emit *e, unsigned s) {
    return s <= JIT_MAX_SLOTS && e->localKind[s] == SLOT_INT &&
           !e->dynamicLocal[s];
}

/* Tighten [lo, hi] for `slot` from `slot cmp k` being `truth`. */
static void boundFromConst(uint8_t cmp, bool truth, int64_t k,
                           int64_t *lo, int64_t *hi) {
    if (!truth) {
        switch (cmp) {   /* the negation of each comparison */
        case OP_LT: cmp = OP_GE; break;
        case OP_LE: cmp = OP_GT; break;
        case OP_GT: cmp = OP_LE; break;
        case OP_GE: cmp = OP_LT; break;
        case OP_EQ: cmp = OP_NE; break;
        case OP_NE: cmp = OP_EQ; break;
        default: return;
        }
    }
    switch (cmp) {
    case OP_LT: if (k == INT64_MIN) return; if (k - 1 < *hi) *hi = k - 1; break;
    case OP_LE: if (k < *hi) *hi = k; break;
    case OP_GT: if (k == INT64_MAX) return; if (k + 1 > *lo) *lo = k + 1; break;
    case OP_GE: if (k > *lo) *lo = k; break;
    case OP_EQ: if (k < *hi) *hi = k; if (k > *lo) *lo = k; break;
    default: break;   /* != bounds nothing */
    }
}

/* `slot cmp other` (or, with `slotLeft` false, `other cmp slot`) being
 * `truth`, where `other` is some int about which nothing else is known. Only
 * a strict comparison says anything, and only one step's worth. */
static void boundFromLocal(uint8_t cmp, bool truth, bool slotLeft,
                           int64_t *lo, int64_t *hi) {
    if (!truth) {
        switch (cmp) {
        case OP_LT: cmp = OP_GE; break;
        case OP_LE: cmp = OP_GT; break;
        case OP_GT: cmp = OP_LE; break;
        case OP_GE: cmp = OP_LT; break;
        default: return;
        }
    }
    if (!slotLeft) {   /* `other cmp slot` is `slot cmp' other` */
        switch (cmp) {
        case OP_LT: cmp = OP_GT; break;
        case OP_LE: cmp = OP_GE; break;
        case OP_GT: cmp = OP_LT; break;
        case OP_GE: cmp = OP_LE; break;
        default: return;
        }
    }
    if (cmp == OP_LT && INT64_MAX - 1 < *hi) *hi = INT64_MAX - 1;
    if (cmp == OP_GT && INT64_MIN + 1 > *lo) *lo = INT64_MIN + 1;
}

/* What the jump-if-false ending block `p` says about `slot` on the edge that
 * enters the block starting at `into`. */
static void guardBound(const Emit *e, const ObjFunction *fn, const JaiBlock *p,
                       uint32_t into, unsigned slot, int64_t *lo, int64_t *hi) {
    const Chunk *c = &fn->chunk;
    /* The last instruction of the block, and the one before it. */
    int g = -1, prev = -1;
    for (int off = (int)p->start; off < (int)p->end;) {
        int len = instructionLength(c, off);
        if (len <= 0) return;
        prev = g;
        g = off;
        off += len;
    }
    if (g < 0) return;
    uint8_t op = c->code[g];
    if (op != OP_JUMP_IF_CMP_LOCAL_K && op != OP_JUMP_IF_CMP_FALSE) return;
    int len = instructionLength(c, g);
    int rel = jaiOpBranchOperandAt(op);
    if (rel < 0) return;
    uint32_t fall  = (uint32_t)(g + len);
    uint32_t taken = (uint32_t)((int32_t)(g + len) +
                                jaiReadI16(c->code + g + 1 + rel));
    if (fall == taken) return;
    bool truth;
    if (into == fall) truth = true;          /* jump-if-false fell through */
    else if (into == taken) truth = false;
    else return;
    uint8_t cmp = c->code[g + 1];

    if (op == OP_JUMP_IF_CMP_LOCAL_K) {
        unsigned s = jaiReadU16(c->code + g + 2);
        uint32_t k = jaiReadU24(c->code + g + 4);
        if (s != slot || k >= (uint32_t)c->constants.count) return;
        Value kv = c->constants.data[k];
        if (!IS_INT(kv)) return;
        boundFromConst(cmp, truth, AS_INT(kv), lo, hi);
        return;
    }
    /* JUMP_IF_CMP_FALSE compares the two values on top, which only the
     * instruction straight before it in the same block can say: the two
     * locals of a GET_LOCAL2. */
    if (prev < 0 || c->code[prev] != OP_GET_LOCAL2) return;
    unsigned a = jaiReadU16(c->code + prev + 1);
    unsigned b = jaiReadU16(c->code + prev + 3);
    if (a == b || !intLocal(e, a) || !intLocal(e, b)) return;
    if (a == slot) boundFromLocal(cmp, truth, true, lo, hi);
    else if (b == slot) boundFromLocal(cmp, truth, false, lo, hi);
}

static bool jitSlotBoundsAt(const Emit *e, const ObjFunction *fn, uint32_t q,
                            unsigned slot, int64_t *loOut, int64_t *hiOut) {
    if (!rangeFactsOn() || e->inlining || !intLocal(e, slot)) return false;
    if (fn->exceptionCount > 0 || fn->defaultCount > 0) return false;
    const Chunk *c = &fn->chunk;
    if (q >= (uint32_t)c->count || hasClosures(c)) return false;

    JaiChunkCfg *cfg = jaiChunkCfg(fn);
    if (cfg == NULL) return false;
    int64_t lo = INT64_MIN, hi = INT64_MAX;
    uint32_t bi = cfg->blockAt[q];
    uint32_t upto = q;   /* the part of this block that runs before q */
    for (unsigned steps = 0; steps < 64; steps++) {
        const JaiBlock *b = &cfg->blocks[bi];
        if (e->osr && e->osrTop >= b->start && e->osrTop <= upto) break;
        bool wrote = false;
        for (int off = (int)b->start; off < (int)upto;) {
            int len = instructionLength(c, off);
            if (len <= 0 || writesSlot(c, off, slot)) { wrote = true; break; }
            off += len;
        }
        if (wrote || b->predCount != 1) break;
        uint32_t pi = cfg->preds[b->predFirst];
        const JaiBlock *p = &cfg->blocks[pi];
        if (pi == bi) break;
        guardBound(e, fn, p, b->start, slot, &lo, &hi);
        bi = pi;
        upto = p->end;
    }
    jaiChunkCfgFree(cfg);
    *loOut = lo;
    *hiOut = hi;
    return lo != INT64_MIN || hi != INT64_MAX;
}

/* Whether `slot + k` cannot overflow at `q`. */
bool jitSlotAddSafe(const Emit *e, const ObjFunction *fn, uint32_t q,
                    unsigned slot, int64_t k) {
    int64_t lo, hi;
    if (!jitSlotBoundsAt(e, fn, q, slot, &lo, &hi)) return false;
    if (k >= 0) return hi <= INT64_MAX - k;
    return lo >= INT64_MIN - k;
}

#endif /* __aarch64__ || __arm64__ */
