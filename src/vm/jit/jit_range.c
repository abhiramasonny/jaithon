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

bool jitOpWritesSlot(const Chunk *c, int off, unsigned slot) {
    return writesSlot(c, off, slot);
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

/* JAITHON_JIT_RANGE_COUNTER=0: a range loop's variable gets no bound from
 * the loop that binds it. */
static bool rangeCounterOn(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("JAITHON_JIT_RANGE_COUNTER");
        on = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return on != 0;
}

/* The bounds OP_FOR_RANGE_BIND puts on the variable it binds from counter
 * slot `cur`, read off every OP_ITER_RANGE that starts that counter -- the
 * counter and its end are the emitter's own temporaries, written by those and
 * by the binds alone, though one pair may serve several loops in turn.
 *
 *   - The counter only ever counts up from the start, so where every start is
 *     a literal the variable is at least the smallest. The start is the
 *     operand under the stop, which is the instruction before the one that
 *     pushes the stop -- when that one pushes exactly one value and pops none.
 *   - An exclusive range stops before `end` and `end` is at most INT64_MAX,
 *     so the variable is at most INT64_MAX - 1. An inclusive one can reach
 *     INT64_MAX itself (its end wraps), so it bounds nothing above. */
static bool pushesOneValue(uint8_t op) {
    switch (op) {
    case OP_INT: case OP_CONST: case OP_GET_LOCAL: case OP_GET_GLOBAL:
    case OP_ADD_INT_CONST: case OP_SUB_INT_CONST: case OP_MUL_INT_CONST:
        return true;
    default:
        return false;
    }
}

static void counterBounds(const Chunk *c, unsigned cur, int64_t *lo,
                          int64_t *hi) {
    bool allExclusive = true, allLiteral = true, any = false;
    int64_t least = INT64_MAX;
    int prev2 = -1, prev1 = -1;
    for (int off = 0; off < c->count;) {
        int len = instructionLength(c, off);
        if (len <= 0) return;
        if (c->code[off] == OP_ITER_RANGE &&
            jaiReadU16(c->code + off + 2) == cur) {
            any = true;
            if (c->code[off + 1] != 0) allExclusive = false;
            /* Straight-line: a branch landing on the stop's push or on this
             * instruction could bring another start with it. */
            if (prev2 >= 0 && c->code[prev2] == OP_INT &&
                pushesOneValue(c->code[prev1]) &&
                !offsetIsBranchTarget(c, (uint32_t)prev1) &&
                !offsetIsBranchTarget(c, (uint32_t)off)) {
                int64_t k = jaiReadI16(c->code + prev2 + 1);
                if (k < least) least = k;
            } else {
                allLiteral = false;
            }
        }
        prev2 = prev1;
        prev1 = off;
        off += len;
    }
    if (!any) return;
    if (allLiteral && least > *lo) *lo = least;
    if (allExclusive && INT64_MAX - 1 < *hi) *hi = INT64_MAX - 1;
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
    /* The edge into the body of a range loop: `slot` is the variable it just
     * bound. Its exhausted edge binds nothing. */
    if (op == OP_FOR_RANGE_BIND && rangeCounterOn()) {
        if (into != (uint32_t)(g + instructionLength(c, g))) return;
        if (jaiReadU16(c->code + g + 3) != slot) return;
        counterBounds(c, jaiReadU16(c->code + g + 5), lo, hi);
        return;
    }
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

/* The graph and the closure scan, kept for the compile in progress: building
 * the graph runs the whole verifier, and a body asks once per fused step it
 * holds -- per pass. jitRangeReset drops it at the start of every compile, so
 * it can never describe a function other than the one being compiled. */
static const ObjFunction *sCfgFn;
static JaiChunkCfg        *sCfg;
static bool                sCfgClosures;

void jitRangeReset(void) {
    if (sCfg != NULL) jaiChunkCfgFree(sCfg);
    sCfg = NULL;
    sCfgFn = NULL;
}

static JaiChunkCfg *rangeCfg(const ObjFunction *fn, bool *closures) {
    if (sCfgFn != fn) {
        jitRangeReset();
        sCfgFn = fn;
        sCfgClosures = hasClosures(&fn->chunk);
        sCfg = sCfgClosures ? NULL : jaiChunkCfg(fn);
    }
    *closures = sCfgClosures;
    return sCfg;
}

static bool jitSlotBoundsAt(const Emit *e, const ObjFunction *fn, uint32_t q,
                            unsigned slot, int64_t *loOut, int64_t *hiOut) {
    /* A loop-bearing inline is walked over its own renumbered chunk, so its
     * slot numbers are the homes the walk reads (inlineLoopCall resets the
     * cache around it); a straight-line inline still names callee slots. */
    if (!rangeFactsOn() || (e->inlining && !e->inlHomes) ||
        !intLocal(e, slot)) {
        return false;
    }
    if (fn->exceptionCount > 0 || fn->defaultCount > 0) return false;
    const Chunk *c = &fn->chunk;
    if (q >= (uint32_t)c->count) return false;

    bool closures = false;
    JaiChunkCfg *cfg = rangeCfg(fn, &closures);
    if (closures || cfg == NULL) return false;
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
