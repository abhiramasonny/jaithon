/* jit_body_index.c -- the subscript and slice arms of the opcode walk. */

#include "vm/jit/jit_arm64.h"
#include "vm/jit/jit_field_read.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include "vm/jit/jit_internal.h"

/* JAITHON_JIT_HOIST_FRESH_COUNT: a subscript of a hoisted list whose count is
 * not in a register (the lean pool ran out -- see planHoists)
 * and that the head's guard does not cover keeps the hoisted `items` and reads
 * only `count` fresh. The count feeds the bounds branch and nothing else, so
 * the element's address no longer waits on a header load. */
/* JAITHON_JIT_LIST_ELEM_REUSE=0: an element of a list of lists is loaded
 * twice, once to confirm it is a list and once more as the result. On, the
 * pointer the type check loaded is the result, and its load takes the
 * payload offset as an immediate instead of an `add` ahead of it. */
static bool jitListElemReuse(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("JAITHON_JIT_LIST_ELEM_REUSE");
        on = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return on != 0;
}

static bool jitHoistFreshCount(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_HOIST_FRESH_COUNT");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

#if (defined(__aarch64__) || defined(__arm64__))

/* ------------------------------------------------------------------ */
/* The string-keyed dict leaf                                           */
/* ------------------------------------------------------------------ */

/* A string-keyed probe placed in FRONT of a dict arm's descriptor call, which
 * stays where it was as the slow path:
 *
 *        key not a string ----------------------------\
 *        x0..x4 <- dict, key, ...;  blr leaf            |
 *        leaf said "can't" -------------------------\   |
 *        b done                                      |   |
 *   slow: <the descriptor call, unchanged>  <-------/---/
 *   done: <whatever followed the call>
 *
 * Both paths meet with a read's answer in the descriptor's result slot, where
 * the code after the call already looks, and a store done; so nothing after
 * the call changes and a deopt that reads the result from the descriptor
 * still finds it there.
 *
 * The leaf clobbers x0..x17 and the slow path after it still reads the
 * operands, so every operand must be in a callee-saved register. A site that
 * calls is planned that way already (noteScratchClobber); an operand anywhere
 * else -- an inlined body's own bank -- emits no leaf at all, which leaves the
 * site exactly as it was. */
/* JAITHON_JIT_LEAF_IN_REG=0 hands a string leaf's answer back through the
 * descriptor's result slot, as the descriptor call does, instead of moving it
 * into the result's register and joining after the load. A one-binary A/B. */
bool jitLeafInReg(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_LEAF_IN_REG");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

bool leafRegOk(unsigned r) {
    return r >= JIT_FIRST_SAVED && r < JIT_FIRST_SAVED + JIT_MAX_SAVED;
}

static bool dictLeafValueKind(SlotKind k) {
    return k == SLOT_INT || k == SLOT_FLOAT || k == SLOT_BOOL ||
           k == SLOT_INST || k == SLOT_LIST || k == SLOT_OBJ ||
           k == SLOT_MAYBE_INST || k == SLOT_MAYBE_OBJ;
}

static bool dictLeafKeyKind(SlotKind k) {
    return k == SLOT_OBJ || k == SLOT_INT;
}

/* dictLeafKeyKind for stack entry `idx`, minus the object keys the compiler
 * can already see are not strings: a tuple built in the body, or a local whose
 * value was one when this compiled. The leaves would only guard or call and
 * come back unanswered, every time -- a dict keyed by `(x, y)` paid ~25
 * instructions an access for nothing. A prediction, so a key that turns out
 * to be a string is merely not answered by a leaf. */
bool dictLeafKeyAt(const Emit *e, unsigned idx) {
    SlotKind k = e->stack[idx];
    if (k == SLOT_INT) return true;
    if (k != SLOT_OBJ) return false;
    uint8_t ot = e->stackObjType[idx];
    if (ot != 0 && ot != (uint8_t)(OBJ_STRING + 1)) return false;
    Value v = e->stackSeen[idx];
    if (IS_OBJ(v) && AS_OBJ(v) != NULL && !IS_STRING(v)) return false;
    return true;
}

/* The key must really be a string; anything else takes the slow path, which
 * is not a deopt, so a dict keyed by tuples at this site costs one compare.
 * An int key needs no guard: its kind is the register's. */
static void dictLeafKeyGuard(Emit *e, unsigned rKey, SlotKind keyKind,
                             LeafFix *fx) {
    if (keyKind == SLOT_INT) return;
    emit(e, jaiA64LdrW(JIT_SCRATCH_A, rKey, (unsigned)offsetof(Obj, type)));
    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_STRING));
    fx->slow[0] = (int)e->count;
    fx->cond[0] = JAI_A64_NE;
    emit(e, jaiA64BCond(JAI_A64_NE, 0));
}

static void dictLeafCall(Emit *e, void *helper, LeafFix *fx) {
    emitConst64(e, JIT_SCRATCH_A, (int64_t)(uintptr_t)helper);
    noteScratchClobber(e);
    emit(e, jaiA64Blr(JIT_SCRATCH_A));
    emit(e, jaiA64SubsXImm(31, 0, 0));
    fx->slow[1] = (int)e->count;
    fx->cond[1] = JAI_A64_NE;
    emit(e, jaiA64BCond(JAI_A64_NE, 0));
    fx->done = (int)e->count;
    emit(e, jaiA64B(0));
    fx->on = true;
}

void emitDictLeafGet(Emit *e, unsigned rDict, unsigned rKey, SlotKind keyKind,
                     int defIdx, SlotKind defKind, bool absentSlow,
                     LeafFix *fx) {
    fx->on = false;
    fx->slow[0] = fx->slow[1] = fx->done = -1;
    if (!jitDictLeaf() || e->inlining) return;
    if (!leafRegOk(rDict) || !leafRegOk(rKey)) return;
    if (e->descOffset + (unsigned)offsetof(JitCallDesc, result) > 4095u) return;
    unsigned rDef = 0;
    if (defIdx >= 0) {
        if (!dictLeafValueKind(defKind)) return;
        rDef = valueXReg(e, (unsigned)defIdx);
        if (!leafRegOk(rDef)) return;
    }
    fpSyncAll(e);
    settleAll(e);

    if (!dictLeafKeyKind(keyKind)) return;
    dictLeafKeyGuard(e, rKey, keyKind, fx);
    emit(e, jaiA64MovX(0, rDict));
    emit(e, jaiA64MovX(1, rKey));
    emit(e, jaiA64AddXImm(2, 31, e->descOffset +
                                     (unsigned)offsetof(JitCallDesc, result)));
    if (absentSlow) {
        emit(e, jaiA64MovzX(3, JIT_DICT_ABSENT_SLOW, 0));
        emit(e, jaiA64MovzX(4, 0, 0));
    } else if (defIdx < 0) {
        emit(e, jaiA64MovzX(3, VAL_NULL, 0));
        emit(e, jaiA64MovzX(4, 0, 0));
    } else {
        emitTagFor(e, defKind, rDef, 3, JIT_SCRATCH_A);
        emit(e, jaiA64MovX(4, rDef));
    }
    dictLeafCall(e, keyKind == SLOT_INT ? (void *)&jitDictGetInt
                                        : (void *)&jitDictGetStr, fx);
}

static void emitDictLeafSet(Emit *e, unsigned rDict, unsigned rKey,
                            SlotKind keyKind, unsigned rVal, SlotKind vk,
                            LeafFix *fx) {
    fx->on = false;
    fx->slow[0] = fx->slow[1] = fx->done = -1;
    if (!jitDictLeaf() || e->inlining) return;
    if (!dictLeafValueKind(vk) || !dictLeafKeyKind(keyKind)) return;
    if (!leafRegOk(rDict) || !leafRegOk(rKey) || !leafRegOk(rVal)) {
        return;
    }
    fpSyncAll(e);
    settleAll(e);

    dictLeafKeyGuard(e, rKey, keyKind, fx);
    emit(e, jaiA64MovX(0, rDict));
    emit(e, jaiA64MovX(1, rKey));
    emitTagFor(e, vk, rVal, 2, JIT_SCRATCH_A);
    emit(e, jaiA64MovX(3, rVal));
    dictLeafCall(e, keyKind == SLOT_INT ? (void *)&jitDictSetInt
                                        : (void *)&jitDictSetStr, fx);
}

/* `k in d`: the leaf checks the container and the key itself (see
 * jitDictHasStr), so nothing is guarded here and a miss on either is the
 * descriptor call, not a deopt. */
void emitDictLeafHas(Emit *e, unsigned rDict, unsigned rKey, SlotKind keyKind,
                     bool negate, LeafFix *fx) {
    fx->on = false;
    fx->slow[0] = fx->slow[1] = fx->done = -1;
    if (!jitDictLeaf() || e->inlining || !dictLeafKeyKind(keyKind)) return;
    if (!leafRegOk(rDict) || !leafRegOk(rKey)) return;
    if (e->descOffset + (unsigned)offsetof(JitCallDesc, result) > 4095u) return;
    fpSyncAll(e);
    settleAll(e);

    emit(e, jaiA64MovX(0, rDict));
    emit(e, jaiA64MovX(1, rKey));
    emit(e, jaiA64AddXImm(2, 31, e->descOffset +
                                     (unsigned)offsetof(JitCallDesc, result)));
    emit(e, jaiA64MovzX(3, negate ? 1u : 0u, 0));
    dictLeafCall(e, keyKind == SLOT_INT ? (void *)&jitDictHasInt
                                        : (void *)&jitDictHasStr, fx);
}

/* JAITHON_JIT_DICT_ADD=0 turns the fused counting arm below off, for a
 * one-binary A/B. */
static bool jitDictAddFuse(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_DICT_ADD");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

/* Proves from the BYTECODE that the four entries under the get's default are
 * the same two locals loaded twice, back to back: the instructions that end
 * at `off` are exactly `<loads of a, b, a, b> <default>`, where the loads are
 * OP_GET_LOCAL/OP_GET_LOCAL2 and the default is a literal or a local read --
 * nothing that can write a local -- and nothing but fall-through reaches any
 * of them past the first. stackLocal alone is not this proof: it says where
 * an entry was read from, not that the local still holds it, and an
 * if-expression between the two loads can write `k` or `d` and join back
 * with the stack settled. */
static bool dictAddLoadsAdjacent(const Emit *e, const Chunk *c, int off,
                                 int dictLocal, int keyLocal) {
    enum { RING = 8 };
    int starts[RING];
    int n = 0;
    int at = 0;
    while (at < off) {
        int len = instructionLength(c, at);
        if (len <= 0) return false;
        starts[n % RING] = at;
        n++;
        at += len;
    }
    if (at != off || n < 2) return false;
    /* The default: one instruction that pushes without writing. */
    int defAt = starts[(n - 1) % RING];
    uint8_t defOp = c->code[defAt];
    if (defOp != OP_INT && defOp != OP_CONST && defOp != OP_GET_LOCAL) {
        return false;
    }
    /* Then, walking back, loads until exactly four slots are named. */
    int slots[4];
    int got = 0;
    int i = n - 2;
    int firstAt = -1;
    while (got < 4) {
        if (i < 0 || n - 1 - i >= RING) return false;
        int ld = starts[i % RING];
        uint8_t op = c->code[ld];
        int s[2];
        int k;
        if (op == OP_GET_LOCAL) {
            s[0] = (int)jaiReadU16(c->code + ld + 1);
            k = 1;
        } else if (op == OP_GET_LOCAL2) {
            s[0] = (int)jaiReadU16(c->code + ld + 1);
            s[1] = (int)jaiReadU16(c->code + ld + 3);
            k = 2;
        } else {
            return false;
        }
        if (got + k > 4) return false;
        /* Filled from the back: the last slot named is the fourth entry. */
        for (int j = k - 1; j >= 0; j--) slots[3 - got - (k - 1 - j)] = s[j];
        got += k;
        firstAt = ld;
        i--;
    }
    if (slots[0] != dictLocal || slots[2] != dictLocal ||
        slots[1] != keyLocal || slots[3] != keyLocal) {
        return false;
    }
    /* Straight-line from the first load to the INVOKE. */
    for (int j = i + 2; j < n; j++) {
        int st = starts[j % RING];
        if (st <= firstAt) continue;
        if (offsetIsBranchTarget(c, (uint32_t)st) ||
            popSkipTarget(e, (uint32_t)st)) {
            return false;
        }
    }
    return !offsetIsBranchTarget(c, (uint32_t)off) &&
           !popSkipTarget(e, (uint32_t)off);
}

/* `d[k] = d.get(k, n) + c`, at the OP_INVOKE of the `get`: the bytecode is
 *
 *     GET d, GET k, GET d, GET k, <n>, INVOKE get 2, INT c, ADD, SET_INDEX
 *
 * and when the two loads of `d` and of `k` are of the same locals, back to
 * back (dictAddLoadsAdjacent), the whole statement is jitDictAddStr. Emitted
 * in FRONT of the get's own code: on success it branches past the SET_INDEX,
 * to where the statement ends with the five entries consumed; on anything
 * else it falls into the unfused sequence, which the walk goes on to emit
 * exactly as before -- so a refusal costs one call, never an answer. The
 * caller has guarded the receiver as a dict. True when the fused call was
 * emitted. */
bool emitDictAddFused(Emit *e, const Chunk *chunk, int off, int count,
                      unsigned ridx) {
    const uint8_t *code = chunk->code;
    if (!jitDictAddFuse() || !jitDictLeaf() || e->inlining) return false;
    if (off + 12 > count || code[off + 7] != OP_INT ||
        code[off + 10] != OP_ADD || code[off + 11] != OP_SET_INDEX) {
        return false;
    }
    if (ridx < 2 || e->depth != ridx + 3 || e->valueDepth < 5) return false;
    if (e->stack[ridx - 2] != SLOT_OBJ || e->stack[ridx] != SLOT_OBJ) {
        return false;
    }
    SlotKind keyKind = e->stack[ridx + 1];
    if (!dictLeafKeyAt(e, ridx + 1) || !dictLeafKeyAt(e, ridx - 1) ||
        e->stack[ridx - 1] != keyKind) {
        return false;
    }
    if (e->stack[ridx + 2] != SLOT_INT) return false;
    /* The same dict and the same key, read from the same locals, and (see
     * dictAddLoadsAdjacent) nothing between the two loads can have written
     * either. */
    if (e->stackLocal[ridx - 2] < 0 ||
        e->stackLocal[ridx - 2] != e->stackLocal[ridx] ||
        e->stackLocal[ridx - 1] < 0 ||
        e->stackLocal[ridx - 1] != e->stackLocal[ridx + 1]) {
        return false;
    }
    if (!dictAddLoadsAdjacent(e, chunk, off, e->stackLocal[ridx - 2],
                              e->stackLocal[ridx - 1])) {
        return false;
    }
    unsigned vd = e->valueDepth;
    unsigned rDict = valueXReg(e, vd - 3);
    unsigned rKey = valueXReg(e, vd - 2);
    unsigned rDef = valueXReg(e, vd - 1);
    if (!leafRegOk(rDict) || !leafRegOk(rKey) || !leafRegOk(rDef)) {
        return false;
    }
    fpSyncAll(e);
    settleAll(e);
    emit(e, jaiA64MovX(0, rDict));
    emit(e, jaiA64MovX(1, rKey));
    emit(e, jaiA64MovX(2, rDef));
    emitConst64(e, 3, (int64_t)jaiReadI16(code + off + 8));
    emit(e, jaiA64MovzX(4, 0, 0));
    emitConst64(e, JIT_SCRATCH_A,
                keyKind == SLOT_INT ? (int64_t)(uintptr_t)&jitDictAddInt
                                    : (int64_t)(uintptr_t)&jitDictAddStr);
    noteScratchClobber(e);
    emit(e, jaiA64Blr(JIT_SCRATCH_A));
    emit(e, jaiA64SubsXImm(31, 0, 0));
    /* Done: past the SET_INDEX, where the statement leaves the stack five
     * entries shallower than it is here. */
    branchToDepth(e, (uint32_t)(off + 12), JAI_A64_EQ,
                  stackSignatureAt(e, e->depth - 5));
    e->wroteHeap = true;
    return true;
}

/* `d[k] += c`, at its OP_DUP2: the bytecode is
 *
 *     <d>, <k>, DUP2, GET_INDEX, INT c, ADD, SET_INDEX
 *
 * and when `d` was a dict and `k` an object when this compiled, the whole
 * statement is jitDictAddStr with no default -- an absent key is the read's
 * KeyError, so the leaf hands it back. The leaf checks both kinds itself, as
 * nothing here has guarded `d`. Same layout as emitDictAddFused: in front of
 * the unfused code, branching past the SET_INDEX on success. */
bool emitDictAugAddFused(Emit *e, const uint8_t *code, int off, int count) {
    if (!jitDictAddFuse() || !jitDictLeaf() || e->inlining) return false;
    if (off + 7 > count || code[off + 1] != OP_GET_INDEX ||
        code[off + 2] != OP_INT || code[off + 5] != OP_ADD ||
        code[off + 6] != OP_SET_INDEX) {
        return false;
    }
    if (e->depth < 2 || e->valueDepth < 2) return false;
    SlotKind keyKind = e->stack[e->depth - 1];
    if (e->stack[e->depth - 2] != SLOT_OBJ || !dictLeafKeyAt(e, e->depth - 1) ||
        !JIT_SEEN_OR_PREDICTED_DICT(e, e->depth - 2)) {
        return false;
    }
    unsigned rDict = valueXReg(e, e->valueDepth - 2);
    unsigned rKey = valueXReg(e, e->valueDepth - 1);
    if (!leafRegOk(rDict) || !leafRegOk(rKey)) return false;
    fpSyncAll(e);
    settleAll(e);
    emit(e, jaiA64MovX(0, rDict));
    emit(e, jaiA64MovX(1, rKey));
    emit(e, jaiA64MovzX(2, 0, 0));
    emitConst64(e, 3, (int64_t)jaiReadI16(code + off + 3));
    emit(e, jaiA64MovzX(4, 1, 0));
    emitConst64(e, JIT_SCRATCH_A,
                keyKind == SLOT_INT ? (int64_t)(uintptr_t)&jitDictAddInt
                                    : (int64_t)(uintptr_t)&jitDictAddStr);
    noteScratchClobber(e);
    emit(e, jaiA64Blr(JIT_SCRATCH_A));
    emit(e, jaiA64SubsXImm(31, 0, 0));
    /* Done: past the SET_INDEX, which leaves the container and the key
     * consumed. */
    branchToDepth(e, (uint32_t)(off + 7), JAI_A64_EQ,
                  stackSignatureAt(e, e->depth - 2));
    e->wroteHeap = true;
    return true;
}

/* Where the descriptor call begins: both "can't" branches land here. */
void leafSlowHere(Emit *e, LeafFix *fx) {
    if (!fx->on || e->count > JIT_MAX_INSTS) return;
    for (unsigned i = 0; i < 2; i++) {
        int at = fx->slow[i];
        if (at < 0 || at >= (int)e->count) continue;
        e->code[at] = jaiA64BCond(fx->cond[i], (int32_t)((int)e->count - at));
    }
}

/* Just past the descriptor call: the leaf's answered path rejoins here. */
void leafDoneHere(Emit *e, LeafFix *fx) {
    if (!fx->on || e->count > JIT_MAX_INSTS) return;
    int at = fx->done;
    if (at < 0 || at >= (int)e->count) return;
    e->code[at] = jaiA64B((int32_t)((int)e->count - at));
}

bool emitGetIndex(Emit *e, const uint8_t *code, int *offp, int stop) {
    int off = *offp;
    do {
        /* `s[i]` on a string: every guard is a load+compare, and the result is a table lookup, not an
         * allocation (the 128 one-byte strings are made once and shared). Without this the whole loop around a character scan declines -- why `str_search` ran interpreted end to end, and every lexer scans one byte at a time. */
        if (e->depth >= 2 && e->stack[e->depth - 1] == SLOT_INT &&
            e->stack[e->depth - 2] == SLOT_OBJ &&
            IS_STRING(e->stackSeen[e->depth - 2])) {
            unsigned rIdx = pushReg(e) - 1;
            unsigned rStr = valueXReg(e, e->valueDepth - 2);
            int sSlot = e->stackLocal[e->depth - 2];
            noteSlotIndexed(e, sSlot);
            int sh = hoistForStr(e, sSlot);

            if (sh >= 0) {
                /* The loop head proved both facts below and loaded the
                 * header; see emitHoistsAt. Only the bounds and the byte
                 * itself are per character. */
                emitBoundsNormalise(e, rIdx, e->hoist[sh].countReg,
                                    JIT_SCRATCH_B, false);
                emit(e, jaiA64LdrByteReg(JIT_SCRATCH_A, e->hoist[sh].itemsReg,
                                         JIT_SCRATCH_B));
            } else if (strFactAscii(e, sSlot)) {
            /* The loop head proved the two facts the guards below would
             * (see planStrFacts); the header itself is reloaded, because
             * a loop that calls has no register to keep it in. */
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, rStr,
                               (unsigned)offsetof(ObjString, length)));
            emitBoundsNormalise(e, rIdx, JIT_SCRATCH_A, JIT_SCRATCH_B,
                                false);
            emit(e, jaiA64LdrX(JIT_SCRATCH_C, rStr,
                               (unsigned)offsetof(ObjString, chars)));
            emit(e, jaiA64LdrByteReg(JIT_SCRATCH_A, JIT_SCRATCH_C,
                                     JIT_SCRATCH_B));
            } else {
            /* Really a string, and not something else this object slot
             * happened to hold when the loop was compiled. */
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, rStr,
                               (unsigned)offsetof(Obj, type)));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_STRING));
            branchOnDeopt(e, JAI_A64_NE);

            /* ASCII only: one scalar is one byte, so indexing is indexing.
             * `scalars` is UINT32_MAX until something asks, so the first
             * time through deopts and the interpreter fills it in. */
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, rStr,
                               (unsigned)offsetof(ObjString, length)));
            emit(e, jaiA64LdrW(JIT_SCRATCH_B, rStr,
                               (unsigned)offsetof(ObjString, scalars)));
            emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
            branchOnDeopt(e, JAI_A64_NE);

            /* jaiNormalizeIndex, then one unsigned compare for both ends.
             * `length` came from an `ldr w`, so it is already the whole
             * register. */
            emitBoundsNormalise(e, rIdx, JIT_SCRATCH_A, JIT_SCRATCH_B,
                                false);

            emit(e, jaiA64LdrX(JIT_SCRATCH_C, rStr,
                               (unsigned)offsetof(ObjString, chars)));
            emit(e, jaiA64LdrByteReg(JIT_SCRATCH_A, JIT_SCRATCH_C,
                                     JIT_SCRATCH_B));
            }
            /* 128 is an imm12, so the compare needs no register: a
             * materialised constant on a body this hot is not free the way
             * a register copy is. */
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, 128));
            branchOnDeopt(e, JAI_A64_HS);

            /* The shared one-byte string. jaiVMInit fills all 128 slots, so
             * this is a load and not a load plus a null test -- see
             * jaiAsciiCharsFill. The scaled add folds the shift in. */
            emitConst64(e, JIT_SCRATCH_C,
                        (int64_t)(uintptr_t)jaiAsciiCharTable());

            /* Carry a sample so later instructions know this is a
             * string: the receiver serves, since only its type is read.
             * Without one the interned-equality path below cannot tell
             * what it is holding and declines. */
            Value strSample = e->stackSeen[e->depth - 2];
            unsigned d1, d2;
            if (!popValue(e, &d1, NULL)) return false;
            if (!popValue(e, &d2, NULL)) return false;
            if (!pushValue3(e, SLOT_OBJ, 0, NULL, strSample, -1)) {
                return false;
            }
            /* What the table holds is interned by construction -- see
             * jaiStringChar -- so a consumer that would guard this for
             * being a string, and for being interned, need not. */
            e->stackAscii[e->depth - 1] = true;
            /* Straight into the entry's register, the shift folded into the
             * load: no `add` to form the slot address, no `mov` after. */
            emit(e, jaiA64LdrXRegLsl3(pushReg(e) - 1, JIT_SCRATCH_C,
                                      JIT_SCRATCH_A));
            off += 1;
            break;
        }
        /* `buf[i]` on a `bytes`: a length-guarded byte load, and the
         * result is a plain int, so nothing is allocated. Every binary
         * format in the language is read one byte at a time through this
         * -- the JPEG bit reader is a `bytes` index and nothing else --
         * and without it the whole function around one declined. */
        if (e->depth >= 2 && e->stack[e->depth - 1] == SLOT_INT &&
            e->stack[e->depth - 2] == SLOT_OBJ &&
            IS_BYTES(e->stackSeen[e->depth - 2])) {
            unsigned rIdx = pushReg(e) - 1;
            unsigned rBuf = valueXReg(e, e->valueDepth - 2);

            /* Really a bytes, and not something else this object slot
             * happened to hold when the body was compiled. */
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, rBuf,
                               (unsigned)offsetof(Obj, type)));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_BYTES));
            branchOnDeopt(e, JAI_A64_NE);

            emit(e, jaiA64LdrW(JIT_SCRATCH_A, rBuf,
                               (unsigned)offsetof(ObjBytes, length)));
            emitBoundsNormalise(e, rIdx, JIT_SCRATCH_A, JIT_SCRATCH_B,
                                false);

            /* The payload is inline after the header, so the base needs no
             * load of its own -- unlike a string, which holds a pointer. */
            emit(e, jaiA64AddX(JIT_SCRATCH_C, rBuf, JIT_SCRATCH_B));
            emit(e, jaiA64LdrByte(JIT_SCRATCH_A, JIT_SCRATCH_C,
                                  (unsigned)offsetof(ObjBytes, data)));

            unsigned dByte1, dByte2;
            if (!popValue(e, &dByte1, NULL)) return false;
            if (!popValue(e, &dByte2, NULL)) return false;
            if (!pushValue(e, SLOT_INT, 0, NULL)) return false;
            emit(e, jaiA64MovX(pushReg(e) - 1, JIT_SCRATCH_A));
            off += 1;
            break;
        }
        /* Index normalised as jaiNormalizeIndex does it, one unsigned compare covering both ends. Out of
         * range, or an element not the kind seen at compile time, goes back to the interpreter -- reading an element has no effect, so resuming at this instruction is always sound. */
        if (e->depth < 2) return subWhy(e, "the model is only %u deep", e->depth);
        if (e->stack[e->depth - 2] == SLOT_OBJ &&
            IS_DICT(e->stackSeen[e->depth - 2])) {
            /* `d[k]`, the read half of the OP_SET_INDEX dict arm below.
             * Without it a loop that reads a dict ran interpreted end to
             * end: `t += d["a"]` two million times was 16,280,472
             * interpreted instructions and 1,684 once this landed.
             *
             * Predicted off a live sample and guarded, as the list arm is,
             * except that the sample must be UNIFORM across the dict --
             * see dictUniformValue for why a dict is not a list here. */
            unsigned dsidx = e->depth - 2;
            Value dsample;
            if (!dictUniformValue(AS_DICT(e->stackSeen[dsidx]), &dsample)) {
                return subWhy(e, "the live dict is empty or holds more than "
                                 "one kind of value");
            }
            SlotKind dkind;
            unsigned dtag;
            ObjClass *dcls;
            uint32_t dshape;
            if (!exemplarKind(dsample, &dkind, &dtag, &dcls, &dshape)) {
                return subWhy(e, "a dict value of a kind the tier cannot hold");
            }
            /* SLOT_OBJ pins nothing, so the container is proved to be a
             * dict before anything is consumed: a miss resumes with the
             * dict and the key both still on the interpreter's stack. */
            emit(e, jaiA64LdrW(JIT_SCRATCH_A,
                               valueXReg(e, e->valueDepth - 2),
                               (unsigned)offsetof(Obj, type)));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_DICT));
            branchOnDeopt(e, JAI_A64_NE);

            /* A string key is answered by the leaf; a miss, or anything it
             * cannot settle, falls through to this same descriptor call,
             * which raises the KeyError a miss is owed. */
            LeafFix gfx;
            gfx.on = false;
            if (dictLeafKeyAt(e, e->depth - 1)) {
                emitDictLeafGet(e, valueXReg(e, e->valueDepth - 2),
                                valueXReg(e, e->valueDepth - 1),
                                e->stack[e->depth - 1], -1, SLOT_NULL, true,
                                &gfx);
            }
            leafSlowHere(e, &gfx);
            if (!emitDescriptor(e, NULL_VAL, dsidx, 2,
                                (void *)&jitGetIndexDict)) {
                return false;
            }
            leafDoneHere(e, &gfx);
            for (unsigned i = 0; i < 2; i++) {
                unsigned drop;
                if (!popValue(e, &drop, NULL)) return false;
            }
            /* The sample travels with the entry, as the list arm's does:
             * without it `names["first"].len()` is an invoke on an object
             * the model cannot name, and the body declines one instruction
             * after the read it just learned to make. */
            if (!pushValue3(e, dkind, dshape, dcls, dsample, -1)) {
                return false;
            }

            unsigned drat = e->descOffset +
                            (unsigned)offsetof(JitCallDesc, result);
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, 31, drat));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, dtag));
            /* Resumes AFTER the read. The lookup itself is pure, but it may
             * have raised and been caught, and re-running it would be a
             * second probe of a table the handler could have changed. */
            branchOnDeoptAt(e, JAI_A64_NE, (uint32_t)(off + 1), true);
            unsigned drd = pushReg(e) - 1;
            if (dkind == SLOT_BOOL) {
                emit(e, jaiA64LdrByte(drd, 31, drat + 8));
            } else {
                emit(e, jaiA64LdrX(drd, 31, drat + 8));
            }
            if (dkind == SLOT_INST) {
                /* Two shapes in one dict cannot be told apart by the tag,
                 * and the walk above only sampled a prefix.
                 *
                 * The object type comes first, for the reason the shared
                 * return path gives: VAL_OBJ covers every heap object, and
                 * reading `klass` off a string lands in its length/hash and
                 * dereferences it. A dict holding a Box under one key and a
                 * str under another SEGFAULTED the VM from ordinary code --
                 * `d[k]` in any body hot enough to compile.
                 *
                 * The tag test above cannot stand in for this: it is the
                 * same test the sampled prefix already passed. */
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, drd,
                                   (unsigned)offsetof(Obj, type)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_INSTANCE));
                branchOnDeoptAt(e, JAI_A64_NE, (uint32_t)(off + 1), true);
                emit(e, jaiA64LdrX(JIT_SCRATCH_A, drd,
                                   (unsigned)offsetof(ObjInstance, klass)));
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_A,
                                   (unsigned)offsetof(ObjClass, shapeId)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, dshape));
                branchOnDeoptAt(e, JAI_A64_NE, (uint32_t)(off + 1), false);
            }
            e->wroteHeap = true;
            off += 1;
            break;
        }
        if (e->stack[e->depth - 2] != SLOT_LIST) {
            return subWhy(e, "the container has kind %s, not list",
                          slotKindName(e->stack[e->depth - 2]));
        }
        if (e->stack[e->depth - 1] != SLOT_INT) {
            return subWhy(e, "the subscript is kind %d, not an int",
                          (int)e->stack[e->depth - 1]);
        }
        unsigned rIdx = pushReg(e) - 1;
        unsigned rList = valueXReg(e, e->valueDepth - 2);
        bool gHoisted = false;

        Value seenList = e->stackSeen[e->depth - 2];
        SlotKind kind = SLOT_OPAQUE;
        unsigned tag = VAL_OBJ;
        ObjClass *elemClass = NULL;
        uint32_t  elemShape = 0;
        /* NULL_VAL on the declared route: there is no exemplar to carry,
         * which is the whole reason that route exists. */
        Value elem = NULL_VAL;
        /* No sample, but the list was DECLARED. See Emit::stackElemDecl:
         * for a body-local list filled through an alias there is nothing
         * to sample and never will be, so the declaration is the only
         * fact available -- and it is a fact, not a guess, because the
         * same byte pins ObjList::stg while the list is still empty.
         *
         * Safe even if it were wrong: listAccessFor rejects a kind the
         * pinned storage contradicts at compile time, and a dispatched
         * access still tag-checks the boxed arm at run time, so a bad
         * declaration deoptimises rather than misreading memory. */
        if (!IS_LIST(seenList) && elemDeclOn()) {
            SlotKind dk = SLOT_OPAQUE;
            switch ((unsigned)e->stackElemDecl[e->depth - 2]) {
            case FIELD_KIND_INT   + 1u: dk = SLOT_INT;   tag = VAL_INT;   break;
            case FIELD_KIND_FLOAT + 1u: dk = SLOT_FLOAT; tag = VAL_FLOAT; break;
            case FIELD_KIND_BOOL  + 1u: dk = SLOT_BOOL;  tag = VAL_BOOL;  break;
            default: break;
            }
            if (dk != SLOT_OPAQUE) {
                kind = dk;
                goto haveElemKind;
            }
        }
        if (!IS_LIST(seenList)) {
            return subWhy(e, "no live list to read an element kind off");
        }
        {
        ObjList *sl = AS_LIST(seenList);
        if (sl->count <= 0) {
            return subWhy(e, "the live list is empty, so there is no exemplar");
        }
        elem = jaiListGet(sl, 0);
        if (IS_INT(elem))        { kind = SLOT_INT;   tag = VAL_INT; }
        else if (IS_FLOAT(elem)) { kind = SLOT_FLOAT; tag = VAL_FLOAT; }
        else if (IS_BOOL(elem))  { kind = SLOT_BOOL;  tag = VAL_BOOL; }
        else if (IS_LIST(elem)) {
            /* A list of lists. `matrix_mul` is `b[k][j]` in its innermost
             * loop and could not compile the outer half of it. */
            kind = SLOT_LIST;
            tag = VAL_OBJ;
        }
        else if (rawObjValue(elem)) {
            /* A list of strings, dicts, sets, or tuples, held raw: the same contract as a SLOT_OBJ
             * global or field (sample specialises, the tag guard below confirms, every consumer
             * re-checks Obj.type for itself). `str_search` builds text out of `chunks[seed %% 8]` and
             * declined that whole loop forty times over before the string case alone was admitted;
             * widened from IS_STRING to rawObjValue so every other raw-holdable element kind gets the
             * same treatment rather than only strings. */
            kind = SLOT_OBJ;
            tag = VAL_OBJ;
        }
        else if (IS_INSTANCE(elem) && AS_INSTANCE(elem)->klass != NULL) {
            /* A list of instances, all of one shape -- which the per-read
             * tag check cannot confirm on its own, so the class is checked
             * too. A list holding two shapes deoptimises on the second. */
            kind = SLOT_INST;
            tag = VAL_OBJ;
            elemClass = AS_INSTANCE(elem)->klass;
            elemShape = elemClass->shapeId;
        } else return false;
        }
    haveElemKind:

        /* One `ldp` for both header fields: `items` at +16, `count`/`capacity` the adjacent int32s at +24, so
         * the pair's second half is `count | capacity << 32` and the bounds test reads it with uxtw -- one instruction per element read (life does nine per cell). */
        noteSlotIndexed(e, e->stackLocal[e->depth - 2]);
        {
            int32_t gOff = 0;
            uint8_t gBase = 0;
            bool gShaped = boundsCoveredAtHead(e, e->stackLocal[e->depth - 2],
                                               e->valueDepth - 1, &gOff,
                                               &gBase);
            noteIndexSpan(e, e->stackLocal[e->depth - 2], gShaped, gOff,
                          gBase);
            gHoisted = gShaped;
        }
        ListAccess gAcc = listAccessFor(e, rList,
                                        e->stackLocal[e->depth - 2],
                                        kind, JIT_SCRATCH_D);
        /* The sampled element and a PINNED storage cannot disagree -- an
         * I64 store holds ints and nothing else -- but the kind is what
         * the loads below are emitted for, so it is checked rather than
         * assumed. A dispatched access picks its second arm from the kind,
         * so it cannot disagree by construction. */
        if (!gAcc.dynamic && gAcc.stg != LIST_STORE_BOXED &&
            kind != listStgKind(gAcc.stg)) {
            return subWhy(e, "element kind %d is not storage %u's",
                          (int)kind, gAcc.stg);
        }
        unsigned gItems = JIT_SCRATCH_C, gCount = JIT_SCRATCH_A;
        int gh = hoistFor(e, e->stackLocal[e->depth - 2]);
        /* A lean hoist has no count: a site the head's guard covers never
         * reads one, and one it does not cover loads its own header. */
        if (gh >= 0 && (e->hoist[gh].hasCount || gHoisted)) {
            gItems = e->hoist[gh].itemsReg;
            gCount = e->hoist[gh].hasCount ? e->hoist[gh].countReg
                                           : JIT_SCRATCH_A;
        } else if (gh >= 0 && jitHoistFreshCount()) {
            gItems = e->hoist[gh].itemsReg;
            emit(e, jaiA64LdrW(gCount, rList,
                               (unsigned)offsetof(ObjList, count)));
            gh = -1;
        } else {
            emitListHeader(e, rList, gItems, gCount);
            gh = -1;
        }
        /* Everything about this element is settled before a word of it is
         * read: the head proved the index in bounds and non-negative, the
         * header is in registers, and the storage is static and unboxed,
         * so there is no tag to check and the stride is the access width.
         * One register-offset load does what the add, the copy into the
         * normalisation scratch and the load did (jitIndexedLoadOn). */
        if (!gAcc.dynamic && gAcc.stg != LIST_STORE_BOXED &&
            jitIndexedLoadOn()) {
            /* Not proved at the head: normalised and checked here as every
             * subscript is, and then the same single load off the result. */
            unsigned rI = rIdx;
            if (!(gHoisted && gh >= 0)) {
                emitBoundsNormalise(e, rIdx, gCount, JIT_SCRATCH_B, true);
                rI = JIT_SCRATCH_B;
            }
            unsigned dg1, dg2;
            if (!popValue(e, &dg1, NULL)) return false;
            if (!popValue(e, &dg2, NULL)) return false;
            if (!pushValue3(e, kind, elemShape, elemClass, elem, -1)) {
                return false;
            }
            if (kind == SLOT_BOOL) {
                emit(e, jaiA64LdrByteIdx(pushReg(e) - 1, gItems, rI));
            } else if (kind == SLOT_FLOAT &&
                       fpWorthLoading(e, code, off + 1, stop)) {
                unsigned idx = e->valueDepth - 1;
                emit(e, jaiA64LdrDIdx(fpRegAt(e, idx), gItems, rI));
                fpClaim(e, idx);
            } else {
                emit(e, jaiA64LdrXIdx(pushReg(e) - 1, gItems, rI));
            }
            off += 1;
            break;
        }
        if (gHoisted) {
            /* The head proved it. Only the normalisation copy is left, and
             * a shaped index is non-negative by that same proof, so even
             * that is just a move. */
            emit(e, jaiA64MovX(JIT_SCRATCH_B, rIdx));
        } else {
            emitBoundsNormalise(e, rIdx, gCount, JIT_SCRATCH_B, true);
        }

        /* Both arms below leave JIT_SCRATCH_C on the PAYLOAD rather than
         * on the element, which is what lets one load serve them: a boxed
         * element's payload is eight bytes into it, an unboxed element IS
         * its payload. */
        /* The predicted (unboxed) arm falls through and the boxed one waits
         * after the body (jitDispatchColdOn). Inline, one of the two always
         * took a branch -- the `b.eq` to the unboxed arm, or the boxed arm's
         * `b` over it -- and since an untyped list of ints is unboxed too
         * once it is longer than eight (JAITHON_LIST_SHAPE_GROWN), the taken
         * one was the common one. */
        bool gColdDone = false;
        /* The element pointer of a list of lists, left in JIT_SCRATCH_D by
         * its type check: the result, without loading it again. */
        bool gListInD = false;
        if (gAcc.dynamic && gAcc.stg == LIST_STORE_BOXED &&
            (kind == SLOT_INT || kind == SLOT_FLOAT || kind == SLOT_BOOL) &&
            jitDispatchColdOn() && e->coldCount < JIT_MAX_COLD &&
            e->fixupCount < JIT_MAX_FIXUPS) {
            int dk = deoptRecordNow(e);
            if (dk >= 0) {
                emit(e, jaiA64LdrByte(JIT_SCRATCH_D, rList,
                                      (unsigned)offsetof(ObjList, stg)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_D, gAcc.alt));
                unsigned ci = e->coldCount++;
                e->fixups[e->fixupCount].instIndex    = (int)e->count;
                e->fixups[e->fixupCount].targetOffset = FIXUP_COLD - ci;
                e->fixups[e->fixupCount].conditional  = true;
                e->fixups[e->fixupCount].depth        = -1;
                e->fixupCount++;
                emit(e, jaiA64BCond(JAI_A64_NE, 0));
                emit(e, jaiA64AddXLsl(JIT_SCRATCH_C, gItems, JIT_SCRATCH_B,
                                      listStgShift(gAcc.alt)));
                e->cold[ci].stub     = -1;
                e->cold[ci].returnTo = (int)e->count;
                e->cold[ci].insn     = 0;
                e->cold[ci].kind     = 2;
                e->cold[ci].deoptK   = dk;
                e->cold[ci].rItems   = (uint8_t)gItems;
                e->cold[ci].tag      = tag;
                gColdDone = true;
            }
        }
        if (!gColdDone) {
        int gSkip = listDispatchBegin(e, &gAcc, rList, JIT_SCRATCH_D);

        emit(e, jaiA64AddXLsl(JIT_SCRATCH_C, gItems,
                              JIT_SCRATCH_B, listStgShift(gAcc.stg)));
        /* An unboxed element has no tag to check, and no object behind it
         * to confirm the type of: the storage already said what it is. */
        if (gAcc.stg == LIST_STORE_BOXED) {
        emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_C, 0));
        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, tag));
        branchOnDeopt(e, JAI_A64_NE);
        if (kind == SLOT_LIST && gSkip < 0 && jitListElemReuse()) {
            emit(e, jaiA64LdrX(JIT_SCRATCH_D, JIT_SCRATCH_C, 8));
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_D,
                               (unsigned)offsetof(Obj, type)));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_LIST));
            branchOnDeopt(e, JAI_A64_NE);
            gListInD = true;
        } else {
        emit(e, jaiA64AddXImm(JIT_SCRATCH_C, JIT_SCRATCH_C, 8));
        if (kind == SLOT_INST) {
            /* The tag says "an object", not "an object of this class" --
             * and not even "an instance" yet: VAL_OBJ is every heap
             * object, so a list holding an instance beside a string must
             * have its object type confirmed before `klass` is read,
             * exactly as OP_FOR_ITER_BIND's SLOT_INST arms already do.
             * Without this, a list whose sampled element is an instance
             * but a later element is (say) a string reads that string's
             * header bytes as an ObjInstance's `klass` pointer and
             * segfaults dereferencing it -- not merely a wrong answer. */
            emit(e, jaiA64LdrX(JIT_SCRATCH_D, JIT_SCRATCH_C, 0));
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_D,
                               (unsigned)offsetof(Obj, type)));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_INSTANCE));
            branchOnDeopt(e, JAI_A64_NE);
            emit(e, jaiA64LdrX(JIT_SCRATCH_D, JIT_SCRATCH_D,
                               (unsigned)offsetof(ObjInstance, klass)));
            emit(e, jaiA64LdrW(JIT_SCRATCH_D, JIT_SCRATCH_D,
                               (unsigned)offsetof(ObjClass, shapeId)));
            emitConst64(e, JIT_SCRATCH_A, (int64_t)elemShape);
            emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_D, JIT_SCRATCH_A));
            branchOnDeopt(e, JAI_A64_NE);
        } else if (kind == SLOT_LIST) {
            /* "an object" is not "a list": every SLOT_LIST consumer reads the header with no check of
             * its own, so the object type is confirmed here, once, before the kind is handed out --
             * same contract, same check, as OP_GET_FIELD_LOCAL's SLOT_LIST arm. Without this a
             * heterogeneous list (`[[1, 2], "not a list"]`) passes the generic VAL_OBJ tag check on
             * either element and reads the second one's bytes through ObjList's field offsets. */
            emit(e, jaiA64LdrX(JIT_SCRATCH_D, JIT_SCRATCH_C, 0));
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_D,
                               (unsigned)offsetof(Obj, type)));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_LIST));
            branchOnDeopt(e, JAI_A64_NE);
        }
        }
        }
        if (gSkip >= 0) {
            int gJoin = listDispatchElse(e, gSkip);
            emit(e, jaiA64AddXLsl(JIT_SCRATCH_C, gItems, JIT_SCRATCH_B,
                                  listStgShift(gAcc.alt)));
            listDispatchEnd(e, gJoin);
        }
        }

        unsigned d1, d2;
        if (!popValue(e, &d1, NULL)) return false;
        if (!popValue(e, &d2, NULL)) return false;
        if (!pushValue3(e, kind, elemShape, elemClass, elem, -1)) return false;
        /* Bool payload is one byte (`BOOL_VAL` compiles to `strb`), so the other seven bytes are stale --
         * an 8-byte load would hand a SLOT_BOOL register (required to hold exactly 0 or 1, since every consumer does `cbnz` on the whole word) garbage. */
        if (gListInD) {
            emit(e, jaiA64MovX(pushReg(e) - 1, JIT_SCRATCH_D));
        } else if (kind == SLOT_BOOL) {
            emit(e, jaiA64LdrByte(pushReg(e) - 1, JIT_SCRATCH_C, 0));
        } else if (kind == SLOT_FLOAT &&
                   fpWorthLoading(e, code, off + 1, stop)) {
            /* Straight into the FP bank, for the same reason a float local
             * goes there: `ldr x` followed by `fmov d, x` puts a
             * cross-register-file move between the load and the multiply
             * that wants it, and `ai[k] * b[k][j]` had two of them. */
            unsigned idx = e->valueDepth - 1;
            emit(e, jaiA64LdrD(fpRegAt(e, idx), JIT_SCRATCH_C, 0));
            fpClaim(e, idx);
        } else {
            emit(e, jaiA64LdrX(pushReg(e) - 1, JIT_SCRATCH_C, 0));
        }
        off += 1;
        break;
    } while (0);
    *offp = off;
    return true;
}

bool emitSetIndex(Emit *e, int *offp) {
    int off = *offp;
    do {
        /* Write half of OP_GET_INDEX, normalised the same way; every guard runs before the store, so a deopt
         * here still resumes at an instruction that hasn't happened yet. Sixteen refusals across the benchmarks came from its absence -- `queens` couldn't compile the function that does the work. */
        if (e->depth < 3) return false;
        if (e->stack[e->depth - 3] == SLOT_OBJ) {
            /* `d[k] = v`: a dict is as ordinary a container here as a list -- without this, dict_ops' loop just
             * moved its decline from `get` to this store (a loop that declines anywhere runs interpreted end to end). Object type guarded before anything is consumed, so a miss resumes with container/key/value all still on the interpreter's stack. */
            unsigned sidx = e->depth - 3;
            if (!JIT_SEEN_OR_PREDICTED_DICT(e, sidx)) {
                e->whyNot = "an index store into an object that is not a dict";
                return false;
            }
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, valueXReg(e, e->valueDepth - 3),
                               (unsigned)offsetof(Obj, type)));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_DICT));
            branchOnDeopt(e, JAI_A64_NE);
            LeafFix sfx;
            sfx.on = false;
            if (dictLeafKeyAt(e, e->depth - 2) &&
                holdsRegister(e->stack[e->depth - 1])) {
                emitDictLeafSet(e, valueXReg(e, e->valueDepth - 3),
                                valueXReg(e, e->valueDepth - 2),
                                e->stack[e->depth - 2],
                                valueXReg(e, e->valueDepth - 1),
                                e->stack[e->depth - 1], &sfx);
            }
            leafSlowHere(e, &sfx);
            if (!emitDescriptor(e, NULL_VAL, sidx, 3,
                                (void *)&jitSetIndexDict)) {
                return false;
            }
            leafDoneHere(e, &sfx);
            for (unsigned i = 0; i < 3; i++) {
                unsigned r;
                if (!popValue(e, &r, NULL)) return false;
            }
            e->wroteHeap = true;
            off += 1;
            break;
        }
        if (e->stack[e->depth - 3] != SLOT_LIST) return false;
        if (e->stack[e->depth - 2] != SLOT_INT) return false;
        SlotKind vk = e->stack[e->depth - 1];
        unsigned vtag = vk == SLOT_INT   ? VAL_INT
                      : vk == SLOT_FLOAT ? VAL_FLOAT
                      : vk == SLOT_BOOL  ? VAL_BOOL
                      : (vk == SLOT_INST || vk == SLOT_LIST ||
                         vk == SLOT_OBJ)  ? VAL_OBJ
                                          : 0xffffffffu;
        if (vtag == 0xffffffffu) return false;
        unsigned rVal = pushReg(e) - 1;
        unsigned rIdx = valueXReg(e, e->valueDepth - 2);
        unsigned rList = valueXReg(e, e->valueDepth - 3);

        noteSlotIndexed(e, e->stackLocal[e->depth - 3]);
        noteSlotStored(e, e->stackLocal[e->depth - 3]);
        bool sHoisted;
        {
            int32_t sOff = 0;
            uint8_t sBase = 0;
            sHoisted = boundsCoveredAtHead(e, e->stackLocal[e->depth - 3],
                                           e->valueDepth - 2, &sOff,
                                           &sBase);
            noteIndexSpan(e, e->stackLocal[e->depth - 3], sHoisted, sOff,
                          sBase);
        }
        ListAccess sAcc = listAccessFor(e, rList, e->stackLocal[e->depth - 3],
                                        vk, JIT_SCRATCH_D);
        /* Exactly what jaiListStoreAccepts allows, and for its reason: an
         * int written into a `list[float]` de-specialises the list in the
         * interpreter, which is not something this can do inline. The
         * dispatched form cannot hit it -- its second arm is the storage
         * that holds a `vk` and no other. */
        if (!sAcc.dynamic && sAcc.stg != LIST_STORE_BOXED &&
            vk != listStgKind(sAcc.stg)) {
            return subWhy(e, "storing kind %d into storage %u",
                          (int)vk, sAcc.stg);
        }
        unsigned sItems = JIT_SCRATCH_C, sCount = JIT_SCRATCH_A;
        int sh = hoistFor(e, e->stackLocal[e->depth - 3]);
        if (sh >= 0 && (e->hoist[sh].hasCount || sHoisted)) {
            sItems = e->hoist[sh].itemsReg;
            sCount = e->hoist[sh].hasCount ? e->hoist[sh].countReg
                                           : JIT_SCRATCH_A;
        } else if (sh >= 0 && jitHoistFreshCount()) {
            sItems = e->hoist[sh].itemsReg;
            emit(e, jaiA64LdrW(sCount, rList,
                               (unsigned)offsetof(ObjList, count)));
            sh = -1;
        } else {
            emitListHeader(e, rList, sItems, sCount);
            sh = -1;
        }
        if (!sAcc.dynamic && sAcc.stg != LIST_STORE_BOXED &&
            jitIndexedLoadOn()) {
            /* The store half of the single-load read: static unboxed storage
             * needs no tag and its stride is the access width, so the
             * element is one register-offset store off the index. A double
             * is stored from its X register, as emitElemStoreAt does. */
            unsigned rI = rIdx;
            if (!(sHoisted && sh >= 0)) {
                emitBoundsNormalise(e, rIdx, sCount, JIT_SCRATCH_B, true);
                rI = JIT_SCRATCH_B;
            }
            if (sAcc.stg == LIST_STORE_U8) {
                emit(e, jaiA64StrByteIdx(rVal, sItems, rI));
            } else {
                emit(e, jaiA64StrXIdx(rVal, sItems, rI));
            }
        } else {
        if (sHoisted) {
            emit(e, jaiA64MovX(JIT_SCRATCH_B, rIdx));
        } else {
            emitBoundsNormalise(e, rIdx, sCount, JIT_SCRATCH_B, true);
        }

        int sSkip = listDispatchBegin(e, &sAcc, rList, JIT_SCRATCH_A);
        emitElemStoreAt(e, sAcc.stg, sItems, JIT_SCRATCH_B, vtag, rVal);
        if (sSkip >= 0) {
            int sJoin = listDispatchElse(e, sSkip);
            emitElemStoreAt(e, sAcc.alt, sItems, JIT_SCRATCH_B, vtag, rVal);
            listDispatchEnd(e, sJoin);
        }
        }
        /* jaiListTouch: the count has not changed, so only the version
         * tells an iterator that the list moved under it. A hoist that holds
         * the bumped version writes it with no load -- see Emit::hoist. */
        if (sh >= 0 && e->hoist[sh].hasVer) {
            emit(e, jaiA64StrW(e->hoist[sh].verReg, rList,
                               (unsigned)offsetof(ObjList, version)));
        } else {
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, rList,
                               (unsigned)offsetof(ObjList, version)));
            emit(e, jaiA64AddXImm(JIT_SCRATCH_A, JIT_SCRATCH_A, 1));
            emit(e, jaiA64StrW(JIT_SCRATCH_A, rList,
                               (unsigned)offsetof(ObjList, version)));
        }
        e->wroteHeap = true;

        unsigned d1, d2, d3;
        if (!popValue(e, &d1, NULL)) return false;
        if (!popValue(e, &d2, NULL)) return false;
        if (!popValue(e, &d3, NULL)) return false;
        off += 1;
        break;
    } while (0);
    *offp = off;
    return true;
}

/* JAITHON_JIT_SLICE_LEAF=0 sends every string slice back through its
 * descriptor call, for a one-binary A/B of emitStringSliceLeaf. */
static bool jitSliceLeaf(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_SLICE_LEAF");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

/* `s[a:b]` on a string, through jaiStringSliceLeaf in front of the descriptor
 * call to jitGetSlice -- the layout emitDictLeafGet draws. The container's
 * type guard has already run; the bounds must be ints in registers, since the
 * leaf takes them as plain integers. The leaf allocates nothing, so nothing is
 * rooted, and every slice it does not answer is the descriptor call. */
static void emitStringSliceLeaf(Emit *e, unsigned flags, unsigned nargs,
                                LeafFix *fx) {
    fx->on = false;
    fx->slow[0] = fx->slow[1] = fx->done = -1;
    if (!jitSliceLeaf() || e->inlining) return;
    unsigned first = e->depth - nargs;
    for (unsigned i = 1; i < nargs; i++) {
        if (e->stack[first + i] != SLOT_INT) return;
    }
    unsigned vfirst = e->valueDepth - nargs;
    for (unsigned i = 0; i < nargs; i++) {
        if (!leafRegOk(valueXReg(e, vfirst + i))) return;
    }
    unsigned rat = e->descOffset + (unsigned)offsetof(JitCallDesc, result);
    fpSyncAll(e);
    settleAll(e);

    unsigned at = 1;
    emit(e, jaiA64MovX(0, valueXReg(e, vfirst)));
    if ((flags & 1u) != 0) {
        emit(e, jaiA64MovX(1, valueXReg(e, vfirst + at)));
        at++;
    } else {
        emit(e, jaiA64MovzX(1, 0, 0));
    }
    if ((flags & 2u) != 0) {
        emit(e, jaiA64MovX(2, valueXReg(e, vfirst + at)));
    } else {
        emit(e, jaiA64MovzX(2, 0, 0));
    }
    emit(e, jaiA64MovzX(3, flags & 3u, 0));
    emitConst64(e, JIT_SCRATCH_A, (int64_t)(uintptr_t)&jaiStringSliceLeaf);
    noteScratchClobber(e);
    emit(e, jaiA64Blr(JIT_SCRATCH_A));
    emit(e, jaiA64SubsXImm(31, 0, 0));
    fx->slow[0] = (int)e->count;
    fx->cond[0] = JAI_A64_EQ;
    emit(e, jaiA64BCond(JAI_A64_EQ, 0));
    /* Into the register the slice will occupy -- the container's, once the
     * operands are popped -- and past the descriptor path's load, as the
     * f-string leaf does it (see emitFormatLeaf). */
    if (jitLeafInReg()) {
        emit(e, jaiA64MovX(valueXReg(e, vfirst), 0));
    } else {
        emit(e, jaiA64MovzX(JIT_SCRATCH_A, VAL_OBJ, 0));
        emit(e, jaiA64StrW(JIT_SCRATCH_A, 31, rat));
        emit(e, jaiA64StrX(0, 31, rat + 8));
    }
    fx->done = (int)e->count;
    emit(e, jaiA64B(0));
    fx->on = true;
}

bool emitGetSlice(Emit *e, const uint8_t *code, int *offp) {
    int off = *offp;
    do {
        /* `xs[a:b]` out to the runtime: the clamp arithmetic has three
         * throwing exits and lives in one place, and the work itself is an
         * O(n) copy against which the descriptor's stores are noise. */
        unsigned flags = code[off + 1];
        unsigned nops  = ((flags & 1u) != 0) + ((flags & 2u) != 0) +
                         ((flags & 4u) != 0);
        unsigned nargs = 1u + nops;
        if (e->depth < nargs) return false;
        unsigned cidx = e->depth - nargs;
        Value cseen = e->stackSeen[cidx];
        /* A string slices as readily as a list -- same runtime call, same
         * "the guard pins the type so the result kind follows" argument --
         * and `s[a:b]` is what every hand-written scanner cuts tokens with.
         * Held as SLOT_OBJ, since that is what a string is here. */
        unsigned cType;
        SlotKind sliceKind;
        if (e->stack[cidx] == SLOT_LIST) {
            cType = OBJ_LIST; sliceKind = SLOT_LIST;
        } else if (e->stack[cidx] == SLOT_OBJ && IS_STRING(cseen)) {
            cType = OBJ_STRING; sliceKind = SLOT_OBJ;
        } else if (e->stack[cidx] == SLOT_OBJ && IS_TUPLE(cseen)) {
            /* `jitGetSlice` is a thin wrapper over `jaiSliceGet`, which
             * already handles a tuple container exactly like a list or a
             * string -- only this arm's own type guard was narrower than
             * what the call it makes actually supports. */
            cType = OBJ_TUPLE; sliceKind = SLOT_OBJ;
        } else {
            e->whyNot = "slicing a container this tier does not model";
            return false;
        }

        /* Guard the container, not the result: with its object type pinned
         * the arm jaiSliceGet takes is settled, so the result's kind
         * follows. Before the descriptor and before any pop, so a miss
         * resumes here with everything still on the interpreter's stack. */
        emit(e, jaiA64LdrW(JIT_SCRATCH_A, pushReg(e) - nargs,
                           (unsigned)offsetof(Obj, type)));
        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, cType));
        branchOnDeopt(e, JAI_A64_NE);

        LeafFix lfx;
        lfx.on = false;
        if (cType == OBJ_STRING && (flags & 4u) == 0) {
            emitStringSliceLeaf(e, flags, nargs, &lfx);
        }
        leafSlowHere(e, &lfx);
        emit(e, jaiA64MovzX(JIT_SCRATCH_A, flags, 0));
        emit(e, jaiA64StrX(JIT_SCRATCH_A, 31,
                           e->descOffset +
                               (unsigned)offsetof(JitCallDesc, aux)));
        if (!emitDescriptorStatus(e, NULL_VAL, cidx, nargs,
                                  (void *)&jitGetSlice, false, -1)) {
            return false;
        }
        if (!jitLeafInReg()) leafDoneHere(e, &lfx);
        for (unsigned i = 0; i < nargs; i++) {
            unsigned r;
            if (!popValue(e, &r, NULL)) return false;
        }
        /* The container's own sample types the slice: a slice of a list of
         * ints is a list of ints, a slice of a string is a string, and
         * every element read re-checks its own tag, so this is a hint and
         * not an assumption. */
        if (!pushValue3(e, sliceKind, 0, NULL, cseen, -1)) return false;
        emit(e, jaiA64LdrX(pushReg(e) - 1, 31,
                           e->descOffset +
                               (unsigned)offsetof(JitCallDesc, result) + 8));
        /* After the load: the leaf put its answer in this register itself. */
        if (jitLeafInReg()) leafDoneHere(e, &lfx);
        /* Deliberately not e->wroteHeap: the only effect is a fresh object
         * and an interpreted re-run would make another. Setting it would
         * decline the next self-call, which is the shape `sort` has. */
        off += 2;
        break;
    } while (0);
    *offp = off;
    return true;
}

#endif /* __aarch64__ || __arm64__ */
