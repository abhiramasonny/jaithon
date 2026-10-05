/* jit_vector.c -- two doubles per instruction for the simplest counted loop.
 *
 * A stencil row, `out[j] = 0.25 * (up[j] + down[j] + mid[j - 1] + mid[j + 1])`,
 * compiles one element per iteration everywhere else in this tier, while clang
 * runs the C++ peer eight doubles per unrolled iteration (`fadd.2d`, see the
 * memory note disassemble-the-peer-first). This file closes that for the one
 * shape it can prove outright: a `for j in a..b` whose body is a single
 * statement
 *
 *     L0[j + k0] = <expression>
 *
 * where the expression reads only unboxed-float lists at `j + const`, float
 * literals and float locals, with `+ - *`. Nothing else: no call, no division
 * (it raises on zero), no int arithmetic on values, no second statement.
 *
 * It does not replace the scalar loop. It runs IN FRONT of it, on entry only:
 * emitted above the offset map at the loop head, as the hoists are, so a back
 * edge lands on the ordinary head and never re-runs it (see emitHoistsAt).
 * At run time it checks everything it needs -- each list's storage is F64,
 * every index the whole range will touch is in [0, count), no step of a
 * folded subscript can overflow, and the stored list is no list it reads at
 * another offset -- and if any check fails it simply falls through to the
 * scalar loop, which then does all the work exactly as before. When they all
 * pass it runs as many whole groups of 2*U elements as fit, then lane pairs,
 * advances the loop counter past them, and leaves the scalar loop at most one
 * element.
 *
 * Why the answer is bit-identical: each lane of fadd/fsub/fmul.2d rounds
 * exactly as the scalar instruction does, under the same FPCR; the expression
 * is evaluated in the bytecode's own order (no reassociation); there is no
 * fused multiply-add; and with the index range proved and the storage pinned
 * there is nothing in the body that can raise, allocate, or change a storage.
 * The one observable side effect besides the elements is the stored list's
 * `version`, bumped once -- which is all a hoisted scalar store does too.
 *
 * JAITHON_JIT_VECTOR=0 leaves every loop scalar. */

#include "vm/jit/jit_arm64.h"
#include "vm/jit/jit_field_read.h"
#include "vm/vm.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "vm/jit/jit_internal.h"

bool gVectorUsed;
bool gNoVector;

#if (defined(__aarch64__) || defined(__arm64__))

static bool jitVectorOn(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("JAITHON_JIT_VECTOR");
        on = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return on != 0;
}

#define VEC_MAX_LISTS 6u
#define VEC_MAX_INV   4u
#define VEC_MAX_OPS   32u
#define VEC_MAX_DEPTH 8u
/* |offset| on a subscript. ldur's reach is -256..255 bytes, and the unrolled
 * copies add up to 48 to it. */
#define VEC_MAX_OFF   12
/* A list read or written at `j + base + k` for one int local `base` is a
 * stream of its own: its own pointer register, its own bounds. */
#define VEC_MAX_STREAMS 6u
#define VEC_NO_BASE   0xffffu

/* VS_IREF is an int local plus a constant, on its way to becoming the base of
 * an index: `row + x`, `lo + i + 1`. */
enum { VS_LIST, VS_IDX, VS_INT, VS_INV, VS_VEC, VS_IREF };

typedef struct {
    uint8_t  kind;
    uint8_t  list;    /* VS_LIST: index into plan.lists */
    uint8_t  inv;     /* VS_INV: index into plan.inv */
    uint16_t base;    /* VS_IDX, VS_IREF: the int local, or VEC_NO_BASE */
    int64_t  k;       /* VS_IDX, VS_IREF: the offset; VS_INT: the value */
} VecSym;

enum { VO_LOAD, VO_ADD, VO_SUB, VO_MUL, VO_STORE };

typedef struct {
    uint8_t op;
    uint8_t dst;      /* stack position the result lands in (LOAD and arithmetic) */
    uint8_t list;     /* LOAD/STORE */
    uint8_t stream;   /* LOAD/STORE */
    int8_t  off;      /* LOAD/STORE */
    /* Operands of the arithmetic, and the STORE's value: a stack position, or
     * an invariant when the matching *Inv flag is set. */
    uint8_t a, b;
    bool    aInv, bInv;
} VecOp;

typedef struct {
    unsigned listCount;
    uint16_t listSlot[VEC_MAX_LISTS];
    bool     read[VEC_MAX_LISTS];       /* a list other than the stored one, loaded */
    unsigned streamCount;
    uint8_t  streamList[VEC_MAX_STREAMS];
    uint16_t streamBase[VEC_MAX_STREAMS];
    int      lo[VEC_MAX_STREAMS], hi[VEC_MAX_STREAMS];
    unsigned invCount;
    bool     invIsLocal[VEC_MAX_INV];
    uint16_t invSlot[VEC_MAX_INV];
    int64_t  invBits[VEC_MAX_INV];
    unsigned opCount;
    VecOp    ops[VEC_MAX_OPS];
    unsigned maxDepth;
    unsigned storeList;
    uint16_t storeBase;
    int      storeOff;
    uint16_t var, cur, end;
    int      failAt;      /* the body offset the match gave up at, or -1 */
} VecPlan;

static bool vecListIndex(VecPlan *p, uint16_t slot, unsigned *out) {
    for (unsigned i = 0; i < p->listCount; i++) {
        if (p->listSlot[i] == slot) { *out = i; return true; }
    }
    if (p->listCount >= VEC_MAX_LISTS) return false;
    p->listSlot[p->listCount] = slot;
    p->read[p->listCount] = false;
    *out = p->listCount++;
    return true;
}

static bool vecStream(VecPlan *p, unsigned list, uint16_t base, int k,
                      unsigned *out) {
    unsigned i;
    for (i = 0; i < p->streamCount; i++) {
        if (p->streamList[i] == list && p->streamBase[i] == base) break;
    }
    if (i == p->streamCount) {
        if (p->streamCount >= VEC_MAX_STREAMS) return false;
        p->streamList[i] = (uint8_t)list;
        p->streamBase[i] = base;
        p->lo[i] = INT32_MAX;
        p->hi[i] = INT32_MIN;
        p->streamCount++;
    }
    if (k < p->lo[i]) p->lo[i] = k;
    if (k > p->hi[i]) p->hi[i] = k;
    *out = i;
    return true;
}

/* An int local fit to be an index base: read-only in the body (nothing in it
 * writes a local) and none of the loop's own three. */
static bool vecIntBase(const Emit *e, const VecPlan *p, uint16_t slot) {
    if (slot > JIT_MAX_SLOTS || slot >= e->base + e->locals) return false;
    if (e->dynamicLocal[slot] || e->nullableLocal[slot]) return false;
    if (slot == p->var || slot == p->cur || slot == p->end) return false;
    return e->localKind[slot] == SLOT_INT;
}

static bool vecPushLocal(const Emit *e, VecPlan *p, VecSym *st, unsigned *d,
                         uint16_t slot) {
    if (*d >= VEC_MAX_DEPTH) return false;
    if (slot > JIT_MAX_SLOTS || slot >= e->base + e->locals) return false;
    if (e->dynamicLocal[slot] || e->nullableLocal[slot]) return false;
    if (slot == p->cur || slot == p->end) return false;
    VecSym s = { 0 };
    s.base = VEC_NO_BASE;
    if (slot == p->var) {
        s.kind = VS_IDX;
        s.k = 0;
    } else if (vecIntBase(e, p, slot)) {
        s.kind = VS_IREF;
        s.base = slot;
        s.k = 0;
    } else if (e->localKind[slot] == SLOT_LIST) {
        unsigned li;
        if (!vecListIndex(p, slot, &li)) return false;
        s.kind = VS_LIST;
        s.list = (uint8_t)li;
    } else if (e->localKind[slot] == SLOT_FLOAT) {
        unsigned i;
        for (i = 0; i < p->invCount; i++) {
            if (p->invIsLocal[i] && p->invSlot[i] == slot) break;
        }
        if (i == p->invCount) {
            if (p->invCount >= VEC_MAX_INV) return false;
            p->invIsLocal[i] = true;
            p->invSlot[i] = slot;
            p->invCount++;
        }
        s.kind = VS_INV;
        s.inv = (uint8_t)i;
    } else {
        return false;
    }
    st[(*d)++] = s;
    return true;
}

/* The loop's shape, read off the bytecode alone. Every local the body names is
 * only READ (no opcode below writes one), so the kinds the emitter holds at the
 * head are the kinds every iteration sees. */
static bool vecMatch(const Emit *e, ObjFunction *fn, uint32_t off, VecPlan *p) {
    const Chunk *c = &fn->chunk;
    const uint8_t *code = c->code;
    if (off + 9u > (uint32_t)c->count) return false;
    int16_t jump = jaiReadI16(code + off + 1);
    p->var = jaiReadU16(code + off + 3);
    p->cur = jaiReadU16(code + off + 5);
    p->end = jaiReadU16(code + off + 7);
    if (p->var > JIT_MAX_SLOTS || p->cur > JIT_MAX_SLOTS ||
        p->end > JIT_MAX_SLOTS) {
        return false;
    }
    if (e->localKind[p->var] != SLOT_INT || e->localKind[p->cur] != SLOT_INT ||
        e->localKind[p->end] != SLOT_INT) {
        return false;
    }
    if (e->dynamicLocal[p->var] || e->dynamicLocal[p->cur] ||
        e->dynamicLocal[p->end]) {
        return false;
    }
    int32_t exitAt = (int32_t)off + 9 + jump;
    int32_t loopAt = exitAt - 3;
    if (loopAt <= (int32_t)off + 9 || exitAt > c->count) return false;
    if (code[loopAt] != OP_LOOP) return false;
    if (loopAt + 3 + jaiReadI16(code + loopAt + 1) != (int32_t)off) {
        return false;
    }

    VecSym st[VEC_MAX_DEPTH];
    unsigned d = 0;
    bool stored = false;
    p->listCount = p->invCount = p->opCount = p->streamCount = 0;
    p->maxDepth = 0;
    p->failAt = -1;
    for (int32_t at = (int32_t)off + 9; at < loopAt;) {
        int len = instructionLength(c, at);
        if (len <= 0 || stored) return false;
        uint8_t op = code[at];
        p->failAt = at;
        if (p->opCount >= VEC_MAX_OPS) return false;
        switch (op) {
        case OP_GET_LOCAL:
            if (len != 3) return false;
            if (!vecPushLocal(e, p, st, &d, jaiReadU16(code + at + 1))) {
                return false;
            }
            break;
        case OP_GET_LOCAL2:
            if (len != 5) return false;
            if (!vecPushLocal(e, p, st, &d, jaiReadU16(code + at + 1))) {
                return false;
            }
            if (!vecPushLocal(e, p, st, &d, jaiReadU16(code + at + 3))) {
                return false;
            }
            break;
        case OP_ADD_INT_CONST:
        case OP_SUB_INT_CONST: {
            /* `j + k` or `base + k` fused: u16 slot, i16 k. */
            if (len != 5 || d >= VEC_MAX_DEPTH) return false;
            uint16_t ks = jaiReadU16(code + at + 1);
            int64_t k = jaiReadI16(code + at + 3);
            if (op == OP_SUB_INT_CONST) k = -k;
            if (k < -VEC_MAX_OFF || k > VEC_MAX_OFF) return false;
            if (ks == p->var) {
                st[d].kind = VS_IDX;
                st[d].base = VEC_NO_BASE;
            } else if (vecIntBase(e, p, ks)) {
                st[d].kind = VS_IREF;
                st[d].base = ks;
            } else {
                return false;
            }
            st[d].k = k;
            d++;
            break;
        }
        case OP_INT:
            if (len != 3 || d >= VEC_MAX_DEPTH) return false;
            st[d].kind = VS_INT;
            st[d].k = jaiReadI16(code + at + 1);
            d++;
            break;
        case OP_CONST: {
            if (len != 4 || d >= VEC_MAX_DEPTH) return false;
            uint32_t idx = jaiReadU24(code + at + 1);
            if (idx >= (uint32_t)c->constants.count) return false;
            Value k = c->constants.data[idx];
            if (IS_INT(k)) {
                st[d].kind = VS_INT;
                st[d].k = AS_INT(k);
            } else if (IS_FLOAT(k)) {
                if (p->invCount >= VEC_MAX_INV) return false;
                double f = AS_FLOAT(k);
                int64_t bits;
                memcpy(&bits, &f, sizeof bits);
                unsigned i = p->invCount++;
                p->invIsLocal[i] = false;
                p->invBits[i] = bits;
                st[d].kind = VS_INV;
                st[d].inv = (uint8_t)i;
            } else {
                return false;
            }
            d++;
            break;
        }
        case OP_ADD:
        case OP_SUB:
        case OP_MUL: {
            if (d < 2) return false;
            VecSym *a = &st[d - 2], *b = &st[d - 1];
            /* `j + k`, `j - k`, `k + j`, `base + j`, `j + base`, and the
             * same with a base already in: index arithmetic, folded. */
            bool aIx = a->kind == VS_IDX || a->kind == VS_IREF;
            bool bIx = b->kind == VS_IDX || b->kind == VS_IREF;
            if (op != OP_MUL && aIx && b->kind == VS_INT) {
                int64_t k;
                if (op == OP_ADD ? __builtin_add_overflow(a->k, b->k, &k)
                                 : __builtin_sub_overflow(a->k, b->k, &k)) {
                    return false;
                }
                if (k < -VEC_MAX_OFF || k > VEC_MAX_OFF) return false;
                a->k = k;
                d--;
                break;
            }
            if (op == OP_ADD && a->kind == VS_INT && bIx) {
                int64_t k;
                if (__builtin_add_overflow(a->k, b->k, &k)) return false;
                if (k < -VEC_MAX_OFF || k > VEC_MAX_OFF) return false;
                *a = *b;
                a->k = k;
                d--;
                break;
            }
            if (op == OP_ADD &&
                ((a->kind == VS_IDX && a->base == VEC_NO_BASE &&
                  b->kind == VS_IREF) ||
                 (a->kind == VS_IREF && b->kind == VS_IDX &&
                  b->base == VEC_NO_BASE))) {
                int64_t k = a->k + b->k;
                if (k < -VEC_MAX_OFF || k > VEC_MAX_OFF) return false;
                a->base = a->kind == VS_IREF ? a->base : b->base;
                a->kind = VS_IDX;
                a->k = k;
                d--;
                break;
            }
            /* Float arithmetic: both sides vectors or float invariants. An
             * int on either side would be a conversion, and is refused. */
            if ((a->kind != VS_VEC && a->kind != VS_INV) ||
                (b->kind != VS_VEC && b->kind != VS_INV)) {
                return false;
            }
            VecOp *o = &p->ops[p->opCount++];
            o->op = op == OP_ADD ? VO_ADD : op == OP_SUB ? VO_SUB : VO_MUL;
            o->dst = (uint8_t)(d - 2);
            o->aInv = a->kind == VS_INV;
            o->a = o->aInv ? a->inv : (uint8_t)(d - 2);
            o->bInv = b->kind == VS_INV;
            o->b = o->bInv ? b->inv : (uint8_t)(d - 1);
            a->kind = VS_VEC;
            d--;
            break;
        }
        case OP_GET_INDEX: {
            if (d < 2) return false;
            VecSym *l = &st[d - 2], *ix = &st[d - 1];
            if (l->kind != VS_LIST || ix->kind != VS_IDX) return false;
            unsigned li = l->list;
            int k = (int)ix->k;
            unsigned si;
            if (!vecStream(p, li, ix->base, k, &si)) return false;
            VecOp *o = &p->ops[p->opCount++];
            o->op = VO_LOAD;
            o->dst = (uint8_t)(d - 2);
            o->list = (uint8_t)li;
            o->stream = (uint8_t)si;
            o->off = (int8_t)k;
            l->kind = VS_VEC;
            d--;
            break;
        }
        case OP_SET_INDEX: {
            /* The statement form: consumes all three, pushes nothing, and
             * must be the last thing the body does. */
            if (d != 3 || at + len != loopAt) return false;
            VecSym *l = &st[0], *ix = &st[1], *v = &st[2];
            if (l->kind != VS_LIST || ix->kind != VS_IDX) return false;
            if (v->kind != VS_VEC && v->kind != VS_INV) return false;
            unsigned li = l->list;
            int k = (int)ix->k;
            unsigned si;
            if (!vecStream(p, li, ix->base, k, &si)) return false;
            VecOp *o = &p->ops[p->opCount++];
            o->op = VO_STORE;
            o->list = (uint8_t)li;
            o->stream = (uint8_t)si;
            o->off = (int8_t)k;
            o->aInv = v->kind == VS_INV;
            o->a = o->aInv ? v->inv : 2u;
            p->storeList = li;
            p->storeBase = ix->base;
            p->storeOff = k;
            stored = true;
            d = 0;
            break;
        }
        default:
            return false;
        }
        if (d > p->maxDepth) p->maxDepth = d;
        at += len;
    }
    p->failAt = -1;
    if (!stored) return false;
    /* A list read at an offset other than the one stored to: if it is the
     * stored list itself the scalar loop is a recurrence, and two lanes
     * cannot be computed at once. Refused here by slot; checked at run time
     * by pointer for every other slot, which might hold the same list. */
    for (unsigned i = 0; i < p->opCount; i++) {
        const VecOp *o = &p->ops[i];
        if (o->op != VO_LOAD) continue;
        if (o->list == p->storeList &&
            (o->off != p->storeOff ||
             p->streamBase[o->stream] != p->storeBase)) {
            return false;
        }
        if (o->list != p->storeList) p->read[o->list] = true;
    }
    return true;
}

/* Registers the vector block may use: nothing live crosses a loop head at
 * depth zero but the locals' homes (x19 up, v8..v15) and whatever the hoists
 * took from the pool, so every other caller-saved register is free. */
static unsigned vecFreeX(const Emit *e, uint8_t *out) {
    static const uint8_t order[] = { 13, 14, 15, 16, 17, 0, 1, 2, 3, 4, 5,
                                     6, 7, 8, 11, 12 };
    unsigned n = 0;
    for (unsigned i = 0; i < sizeof order; i++) {
        unsigned r = order[i];
        bool taken = false;
        for (unsigned h = 0; h < e->hoistTaken && !taken; h++) {
            taken = e->hoistPool[h] == r;
        }
        for (unsigned s = 0; s < e->base + e->locals && s <= JIT_MAX_SLOTS &&
                             !taken; s++) {
            taken = e->slotXReg[s] == r;
        }
        if (!taken) out[n++] = (uint8_t)r;
    }
    return n;
}

static void vecPatch(Emit *e, unsigned at, unsigned cond, unsigned to) {
    if (at < e->count && to <= JIT_MAX_INSTS) {
        e->code[at] = jaiA64BCond(cond, (int32_t)(to - at));
    }
}

/* The loop body for lane sets 0..nU-1, every instruction of the program
 * issued for all of them before the next: the loads of every set first, so
 * they overlap. Position `pos` of set `u` is v(16 + dense[pos]*U + u); an
 * invariant is v(31 - i). */
static void vecBody(Emit *e, const VecPlan *p, const uint8_t *dense,
                    unsigned U, unsigned nU, const unsigned *xP) {
#define VREG(pos, u) (16u + (unsigned)dense[(pos)] * U + (u))
    for (unsigned i = 0; i < p->opCount; i++) {
        const VecOp *o = &p->ops[i];
        for (unsigned u = 0; u < nU; u++) {
            int32_t at = 8 * o->off + 16 * (int32_t)u;
            unsigned ra = o->aInv ? 31u - o->a : VREG(o->a, u);
            unsigned rb = o->bInv ? 31u - o->b : VREG(o->b, u);
            switch (o->op) {
            case VO_LOAD:
                emit(e, jaiA64LdurQ(VREG(o->dst, u), xP[o->stream], at));
                break;
            case VO_ADD:
                emit(e, jaiA64Fadd2D(VREG(o->dst, u), ra, rb));
                break;
            case VO_SUB:
                emit(e, jaiA64Fsub2D(VREG(o->dst, u), ra, rb));
                break;
            case VO_MUL:
                emit(e, jaiA64Fmul2D(VREG(o->dst, u), ra, rb));
                break;
            case VO_STORE:
                emit(e, jaiA64SturQ(ra, xP[o->stream], at));
                break;
            }
        }
    }
#undef VREG
}

void emitVectorHead(Emit *e, ObjFunction *fn, uint32_t off) {
    if (!jitVectorOn() || gNoVector || e->measuring || e->inlining ||
        e->failed) {
        return;
    }
    if (e->depth != 0 || e->valueDepth != 0) return;
    /* A few hundred words per loop, which the measuring pass never sees. A
     * body already a quarter of the way to JIT_MAX_INSTS has its deopt stubs
     * still to come; outgrowing the buffer would cost a second compile without
     * the vector blocks (gNoVector), so a large body keeps its loops scalar. */
    if (e->count > JIT_MAX_INSTS / 4u) return;
    VecPlan p;
    p.failAt = -1;
    if (!vecMatch(e, fn, off, &p)) {
        /* Only a body the walk got into is worth a line: most loops are not
         * this shape at all, and say so at their first instruction. */
        if (p.failAt > (int)off + 9 && getenv("JAI_JIT_WHY")) {
            fprintf(stderr, "[jit] %s at %u: not vectorised, %s at %d\n",
                    jitFnLabel(fn), off,
                    jaiOpName((OpCode)fn->chunk.code[p.failAt]), p.failAt);
        }
        return;
    }

    /* Lane sets per iteration: as many as the vector registers allow, at most
     * four (eight doubles, what clang unrolls a stencil to). A stack position
     * p of lane set u lives in v(16 + p*U + u); invariants count down from v31. */
    uint8_t dense[VEC_MAX_DEPTH];
    unsigned nPos = 0;
    memset(dense, 0xff, sizeof dense);
    for (unsigned i = 0; i < p.opCount; i++) {
        const VecOp *o = &p.ops[i];
        if (o->op == VO_STORE || o->dst >= VEC_MAX_DEPTH) continue;
        if (dense[o->dst] == 0xff) dense[o->dst] = (uint8_t)nPos++;
    }
    unsigned U = 4;
    while (U > 0 && 16u + nPos * U > 32u - p.invCount) U /= 2;
    if (U == 0) return;
    unsigned lgG = U == 4 ? 3u : U == 2 ? 2u : 1u;   /* log2(2U) */

    uint8_t freeX[16];
    unsigned nFree = vecFreeX(e, freeX);
    bool anyBase = false, anyPlain = false;
    for (unsigned si = 0; si < p.streamCount; si++) {
        if (p.streamBase[si] != VEC_NO_BASE) anyBase = true;
        else anyPlain = true;
    }
    if (nFree < p.streamCount + 3u + (anyBase ? 1u : 0u)) return;
    const unsigned tA = JIT_SCRATCH_A, tB = JIT_SCRATCH_B;
    unsigned fi = 0;
    unsigned xC = freeX[fi++], xE = freeX[fi++], xN = freeX[fi++];
    unsigned xS = anyBase ? freeX[fi++] : 0u;
    unsigned xP[VEC_MAX_STREAMS];
    for (unsigned si = 0; si < p.streamCount; si++) xP[si] = freeX[fi++];

    unsigned skips[6 + 7 * VEC_MAX_STREAMS + VEC_MAX_LISTS];
    unsigned nSkip = 0;
#define VEC_SKIP(cond)                                                       \
    do {                                                                     \
        skips[nSkip] = e->count;                                             \
        skipCond[nSkip++] = (cond);                                          \
        emit(e, jaiA64BCond((cond), 0));                                     \
    } while (0)
    unsigned skipCond[sizeof skips / sizeof skips[0]];

    unsigned r = localIn(e, p.cur, xC);
    if (r != xC) emit(e, jaiA64MovX(xC, r));
    r = localIn(e, p.end, xE);
    if (r != xE) emit(e, jaiA64MovX(xE, r));

    /* The cheapest tests first, so a loop that will not take the vector run
     * finds out in a few instructions. At least one lane pair: end - cur
     * neither wraps nor falls short of 2 (which covers end <= cur). Fewer
     * than 2U elements skip the unrolled loop and take only the pair loop
     * after it. */
    emit(e, jaiA64SubsX(xN, xE, xC));
    VEC_SKIP(JAI_A64_VS);
    emit(e, jaiA64SubsXImm(31, xN, 2u));
    VEC_SKIP(JAI_A64_LT);
    /* The stored list must be no list read at another offset: a compare and
     * a branch a list, before any bounds are worked out. localIn writes only
     * the register it is given, so the stored list stays in tA throughout. */
    {
        unsigned ro = 0;
        bool haveRo = false;
        for (unsigned l = 0; l < p.listCount; l++) {
            if (l == p.storeList || !p.read[l]) continue;
            if (!haveRo) {
                ro = localIn(e, p.listSlot[p.storeList], tA);
                haveRo = true;
            }
            unsigned rl = localIn(e, p.listSlot[l], tB);
            emit(e, jaiA64SubsXReg(31, ro, rl));
            VEC_SKIP(JAI_A64_EQ);
        }
    }
    /* Every plain index is cur + lo or more: cur >= -minLo. */
    if (anyPlain) {
        int minLo = 0;
        for (unsigned si = 0; si < p.streamCount; si++) {
            if (p.streamBase[si] == VEC_NO_BASE && p.lo[si] < minLo) {
                minLo = p.lo[si];
            }
        }
        emit(e, jaiA64SubsXImm(31, xC, (unsigned)(-minLo)));
        VEC_SKIP(JAI_A64_LT);
    }
    /* A based subscript was folded into one offset, but the scalar loop
     * computes it one overflow-checked step at a time: `x + 12 + row` adds 12
     * to x first, and raises there when x is near the top of int even though
     * the whole sum is a small index. Each step it takes is x + k, base + k,
     * or x + base + k, with |k| <= 12 (vecMatch bounds every partial offset as
     * it folds). The last is within 24 of an index the stream checks below
     * prove in [0, count), so it cannot wrap. The first two cannot once cur,
     * end and every base lie in [-2^62, 2^62) -- then no step of any element
     * overflows, and the vector run raises nowhere the scalar loop would.
     * Outside that the scalar loop runs, and raises at its element. The test
     * is that (v >> 62) + 1, unsigned, is 0 or 1. Plain subscripts need none
     * of this: their steps are x + k with x itself bounded by the count. */
    if (anyBase) {
        uint16_t seen[VEC_MAX_STREAMS];
        unsigned nSeen = 0;
        for (unsigned v = 0; v < 2u + p.streamCount; v++) {
            unsigned rv;
            if (v == 0) {
                rv = xC;
            } else if (v == 1) {
                rv = xE;
            } else {
                uint16_t b = p.streamBase[v - 2u];
                bool dup = b == VEC_NO_BASE;
                for (unsigned i = 0; i < nSeen && !dup; i++) dup = seen[i] == b;
                if (dup) continue;
                seen[nSeen++] = b;
                rv = localIn(e, b, tA);
            }
            emit(e, jaiA64AsrX(tB, rv, 62));
            emit(e, jaiA64AddXImm(tB, tB, 1));
            emit(e, jaiA64SubsXImm(31, tB, 2u));
            VEC_SKIP(JAI_A64_HS);
        }
    }

    for (unsigned si = 0; si < p.streamCount; si++) {
        unsigned rl = localIn(e, p.listSlot[p.streamList[si]], tA);
        emit(e, jaiA64LdrByte(tB, rl, (unsigned)offsetof(ObjList, stg)));
        emit(e, jaiA64SubsXImm(31, tB, LIST_STORE_F64));
        VEC_SKIP(JAI_A64_NE);
        /* The last index reached is end - 1 + base + hi < count, i.e.
         * end + base <= count - hi. count is a non-negative int and
         * |hi| <= 12, so the right side cannot wrap. */
        emit(e, jaiA64LdrW(tB, rl, (unsigned)offsetof(ObjList, count)));
        if (p.hi[si] > 0) {
            emit(e, jaiA64SubXImm(tB, tB, (unsigned)p.hi[si]));
        } else if (p.hi[si] < 0) {
            emit(e, jaiA64AddXImm(tB, tB, (unsigned)(-p.hi[si])));
        }
        emit(e, jaiA64LdrX(xP[si], rl, (unsigned)offsetof(ObjList, items)));
        if (p.streamBase[si] == VEC_NO_BASE) {
            emit(e, jaiA64SubsXReg(31, xE, tB));
            VEC_SKIP(JAI_A64_GT);
            emit(e, jaiA64AddXLsl(xP[si], xP[si], xC, 3));
            continue;
        }
        /* A base: end + base and cur + base must not wrap, and the first
         * index reached, cur + base + lo, must not be negative -- a negative
         * index wraps to the far end, which is the scalar loop's business. */
        unsigned rb = localIn(e, p.streamBase[si], tA);
        emit(e, jaiA64AddsX(xS, xE, rb));
        VEC_SKIP(JAI_A64_VS);
        emit(e, jaiA64SubsXReg(31, xS, tB));
        VEC_SKIP(JAI_A64_GT);
        emit(e, jaiA64AddsX(xS, xC, rb));
        VEC_SKIP(JAI_A64_VS);
        emit(e, jaiA64SubsXImm(31, xS,
                               p.lo[si] < 0 ? (unsigned)(-p.lo[si]) : 0u));
        VEC_SKIP(JAI_A64_LT);
        emit(e, jaiA64AddXLsl(xP[si], xP[si], xS, 3));
    }
    for (unsigned i = 0; i < p.invCount; i++) {
        unsigned vr = 31u - i;
        if (p.invIsLocal[i]) {
            unsigned ri = localIn(e, p.invSlot[i], tA);
            emit(e, jaiA64Dup2DX(vr, ri));
        } else {
            emitConst64(e, tA, p.invBits[i]);
            emit(e, jaiA64Dup2DX(vr, tA));
        }
    }

    /* Groups of 2U, and the counter moved past them now: the pointers already
     * carry cur, so xC is not read inside the loop. No whole group (a short
     * row) goes straight to the pair loop. */
    emit(e, jaiA64LsrX(xN, xN, lgG));
    unsigned noGroup = e->count;
    emit(e, jaiA64CbzX(xN, 0));
    emit(e, jaiA64AddXLsl(xC, xC, xN, lgG));

    unsigned top = e->count;
    vecBody(e, &p, dense, U, U, xP);
    for (unsigned si = 0; si < p.streamCount; si++) {
        emit(e, jaiA64AddXImm(xP[si], xP[si], 16u * U));
    }
    emit(e, jaiA64SubsXImm(xN, xN, 1));
    emit(e, jaiA64BCond(JAI_A64_NE, (int32_t)top - (int32_t)e->count));
    if (noGroup < e->count) {
        e->code[noGroup] = jaiA64CbzX(xN, (int32_t)(e->count - noGroup));
    }

    /* What is left is under 2U elements: take its pairs one lane set at a
     * time, so at most one element falls to the scalar loop. A five-point
     * stencil's 498-wide rows leave two, which is ninety scalar instructions
     * a row otherwise. */
    if (U > 1) {
        emit(e, jaiA64SubX(xN, xE, xC));
        emit(e, jaiA64LsrX(xN, xN, 1));
        emit(e, jaiA64SubsXImm(31, xN, 0));
        unsigned none = e->count;
        emit(e, jaiA64BCond(JAI_A64_EQ, 0));
        emit(e, jaiA64AddXLsl(xC, xC, xN, 1));
        unsigned pairTop = e->count;
        vecBody(e, &p, dense, U, 1, xP);
        for (unsigned si = 0; si < p.streamCount; si++) {
            emit(e, jaiA64AddXImm(xP[si], xP[si], 16u));
        }
        emit(e, jaiA64SubsXImm(xN, xN, 1));
        emit(e, jaiA64BCond(JAI_A64_NE,
                            (int32_t)pairTop - (int32_t)e->count));
        vecPatch(e, none, JAI_A64_EQ, e->count);
    }

    /* The counter and the variable as the scalar loop would have left them
     * after its last bind, through a register localOut's tag scratches
     * (x11, x12) cannot clobber -- xC may be one of them, so both values are
     * taken before either write. */
    emit(e, jaiA64MovX(tA, xC));
    emit(e, jaiA64SubXImm(tB, xC, 1));
    localOut(e, p.cur, tA);
    localOut(e, p.var, tB);
    {
        unsigned ro = localIn(e, p.listSlot[p.storeList], tB);
        emit(e, jaiA64LdrW(tA, ro, (unsigned)offsetof(ObjList, version)));
        emit(e, jaiA64AddXImm(tA, tA, 1));
        emit(e, jaiA64StrW(tA, ro, (unsigned)offsetof(ObjList, version)));
    }
    for (unsigned i = 0; i < nSkip; i++) {
        vecPatch(e, skips[i], skipCond[i], e->count);
    }
#undef VEC_SKIP
    gVectorUsed = true;
    if (getenv("JAI_JIT_WHY")) {
        fprintf(stderr, "[jit] %s %s at %u: vectorised %u lists, %u streams "
                "x %u lanes\n", e->osr ? "osr" : "func", jitFnLabel(fn), off,
                p.listCount, p.streamCount, 2u * U);
    }
}

#else
void emitVectorHead(Emit *e, ObjFunction *fn, uint32_t off) {
    (void)e; (void)fn; (void)off;
}
#endif
