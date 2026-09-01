/* jit_body_cmp.c -- the comparison and membership arms of the opcode walk. */
#include "vm/jit/jit.h"

#include "vm/jit/jit_arm64.h"
#include "vm/jit/jit_field_read.h"
#include "vm/vm.h"

#include <stddef.h>
#include <stdint.h>
#include "vm/jit/jit_internal.h"

#if (defined(__aarch64__) || defined(__arm64__))

bool emitCompare(Emit *e, uint8_t op, int *offp) {
    int off = *offp;
    do {
        /* Operands are read without popping them off the model: a NaN sends this back to the interpreter,
         * whose stack still has them. Popping first once left the model two entries short, so the re-run comparison silently read whatever was underneath -- a bug that surfaced one line later instead of not at all. */
        if (e->depth < 2) return subWhy(e, "a compare with nothing under it");
        SlotKind ka = e->stack[e->depth - 2], kb = e->stack[e->depth - 1];
        /* `node == null` puts an instance beside a maybe-instance. Both
         * are a pointer or zero in a register, so the compare is the same
         * one; treat the pair as maybe-instance. */
        if (ka != kb) {
            bool mixable =
                (ka == SLOT_INST && kb == SLOT_MAYBE_INST) ||
                (ka == SLOT_MAYBE_INST && kb == SLOT_INST) ||
                nullLiteralPair(e, op, ka, kb);
            /* Named: 92 declines across four compiler files said only
             * "OP_EQ", and the two operands' kinds are the entire question
             * at this arm. */
            if (!mixable) {
                return subWhy(e, "a compare of a %s with a %s",
                              slotKindName(ka), slotKindName(kb));
            }
            ka = kb = SLOT_MAYBE_INST;
        }
        if (!holdsRegister(ka)) {
            return subWhy(e, "a compare of two %ss, which hold no register",
                          slotKindName(ka));
        }
        /* Read through xHeldIn, not straight out of the bank: an operand
         * that is a plain read of a local is still in the local's own
         * register. See popValue. */
        int64_t kcmp = 0;
        bool foldCmp = ka == SLOT_INT &&
                       pendingImm12(e, e->valueDepth - 1, &kcmp);
        unsigned cond;
        switch (op) {
        case OP_EQ: cond = JAI_A64_EQ; break;
        case OP_NE: cond = JAI_A64_NE; break;
        case OP_LT: cond = ka == SLOT_FLOAT ? JAI_A64_MI : JAI_A64_LT; break;
        case OP_LE: cond = ka == SLOT_FLOAT ? JAI_A64_LS : JAI_A64_LE; break;
        case OP_GT: cond = JAI_A64_GT; break;
        default:    cond = ka == SLOT_FLOAT ? JAI_A64_GE : JAI_A64_GE; break;
        }
        if (ka == SLOT_FLOAT) {
            /* Both operands stay where they are: the compare reads the
             * FP bank, and the NaN guard's stub knows how to write an
             * FP-resident entry out if it is ever taken.
             *
             * Released BEFORE the operands are read, exactly as the fused
             * arm at OP_JUMP_IF_CMP_FALSE does it and for the same reason:
             * this arm is on the borrow whitelist, so an operand can still
             * be held in a local's own d register, and `nanToDeopt`
             * records -- which `deoptRecordAt` asserts nothing may be
             * borrowing at. Only the fused arm had this, so a body whose
             * float compare did NOT fuse into a branch declined with "a
             * float borrow reached OP_GT's guard"; `let a = ys[i] > py`
             * followed by a use of `a` is exactly that shape. Costs one
             * fmov when a borrow is live, against declining the loop. */
            fpReleaseAll(e);
            unsigned db = fpOperand(e, e->valueDepth - 1);
            unsigned da = fpOperand(e, e->valueDepth - 2);
            /* nanToDeopt records, so nothing may still be deferred. A
             * float local can be borrowed out of an X register in OSR
             * mode; settling costs a mov on a path that is about to
             * compare NaNs, and buys the compile. */
            settleAll(e);
            emit(e, jaiA64FcmpD(da, db));
            nanToDeopt(e);
        /* Two bools compare exactly as two ints do. Each is 0 or 1 in
         * its register (see SLOT_BOOL), so `cmp` answers `==` and `!=`
         * directly, and the ordering conditions come out right as well --
         * false sorts below true, which is what the interpreter says.
         *
         * Missing until now, and a whole-function refusal rather than a
         * slow path: `let a = ys[i - 1] > py` / `let b = ys[i] > py` /
         * `if a != b` is how a point-in-polygon crossing test is written,
         * and jaicv's `nearest_gap_squared` carries a comment saying it
         * had to be spelt as two nested `if`s and spelt TWICE because of
         * this.
         *
         * MEASURED. That loop over a 100000-point polygon, forty passes:
         * 215 ms declined against 36 ms compiled, best of five alternating
         * runs with no overlap -- 5.8x, and a decline-to-compile
         * transition rather than a micro-optimisation. The flag form is
         * now as fast as the nested one it was rewritten into (7.23 ms
         * against 7.67 in one binary), so the workaround can go.
         *
         * RULED OUT: the self-hosted compiler does not move. `check
         * --no-cache` over four compiler files is 2015 ms against 2020
         * median of seven, inside the spread, even though the same census
         * counts 66 sites there -- they clear this gate and stop at the
         * next, which is the pattern every arm added today has shown. */
        } else if (ka == SLOT_INT || ka == SLOT_MAYBE_INST ||
                   ka == SLOT_BOOL) {
            unsigned ra = xHeldIn(e, e->valueDepth - 2);
            if (foldCmp) {
                e->kPend &= ~(1u << (e->valueDepth - 1));
                emitCmpImm(e, ra, kcmp);
            } else {
                emit(e, jaiA64SubsXReg(31, ra,
                                       xHeldIn(e, e->valueDepth - 1)));
            }
        } else if ((op == OP_EQ || op == OP_NE) && ka == SLOT_OBJ &&
                   (e->stackUnit[e->depth - 2] ||
                    e->stackUnit[e->depth - 1])) {
            /* Two payload-less enum values are equal exactly when they are the
             * same object: a variant with no payload has no state to tell
             * instances apart, so every mention of it yields the one
             * shared EnumVariant::unit, and enumValsEqual reduces to
             * type-and-tag at zero payload. One side must be the folded
             * constant (stackUnit, a proof), never merely a value observed
             * to be one. The other may be anything on the heap: for two
             * VAL_OBJs of different Obj types jaiValuesEqual returns false
             * without consulting an `__eq__`, which is the same answer the
             * pointer compare gives.
             *
             * Needed for the `Enum.Variant` fold to pay at all -- with the
             * fold alone, OP_EQ and this opcode simply became the new
             * refusal and the same functions still declined. */
            unsigned rb = xHeldIn(e, e->valueDepth - 1);
            unsigned ra = xHeldIn(e, e->valueDepth - 2);
            emit(e, jaiA64SubsXReg(31, ra, rb));
        } else if ((op == OP_EQ || op == OP_NE) && ka == SLOT_OBJ &&
                   IS_STRING(e->stackSeen[e->depth - 2]) &&
                   IS_STRING(e->stackSeen[e->depth - 1]) &&
                   !preferLeafEquality(e, e->depth - 2, e->depth - 1)) {
            /* This path guards, so nothing may still be deferred when it
             * does -- settled here, at the top of the path, which is where
             * the settling is unconditionally executed. */
            settleAll(e);
            unsigned rb = valueXReg(e, e->valueDepth - 1);
            unsigned ra = valueXReg(e, e->valueDepth - 2);
            /* Two interned strings are equal exactly when they are the same object (`text[i] == " "` relies on
             * this: one-character strings are shared singletons). Anything not interned, or not a string, deopts and the interpreter compares properly. */
            for (unsigned side = 0; side < 2; side++) {
                unsigned r = side == 0 ? ra : rb;
                /* Already known to be an interned string: `s[i] == t[j]`
                 * would otherwise guard twelve instructions to reach one
                 * compare. */
                if (e->stackAscii[e->depth - (side == 0 ? 2u : 1u)]) continue;
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, r,
                                   (unsigned)offsetof(Obj, type)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_STRING));
                branchOnDeopt(e, JAI_A64_NE);
                emit(e, jaiA64LdrByte(JIT_SCRATCH_A, r,
                                      (unsigned)offsetof(Obj, subFlag)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, 0));
                branchOnDeopt(e, JAI_A64_EQ);
            }
            emit(e, jaiA64SubsXReg(31, ra, rb));
        } else if (isOrdering(op) && ka == SLOT_OBJ &&
                   oneBytePair(e, e->depth - 2, e->depth - 1)) {
            /* Ordering only. `==` stays on the identity arm above, which
             * is cheaper and holds for strings of any length. See
             * emitOneByteString. This path guards, so settle first. */
            settleAll(e);
            unsigned rb = valueXReg(e, e->valueDepth - 1);
            unsigned ra = valueXReg(e, e->valueDepth - 2);
            emitOneByteString(e, e->depth - 2, ra, JIT_SCRATCH_C);
            emitOneByteString(e, e->depth - 1, rb, JIT_SCRATCH_D);
            emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_C, JIT_SCRATCH_D));
        /* Two strings of any length, all six operators, through the leaf
         * call. LAST of the string arms on purpose: the identity arm above
         * answers `==` with no call at all, and the one-byte arm answers an
         * ordering with two byte loads, so both are strictly better where
         * they apply and this is only what they leave behind.
         *
         * Not inside an inlined body. Its entries are in x0..x8
         * (inlineOwnBank) and a call would run over them; declining here
         * leaves that case exactly as it was. */
        } else if (jitStrCmpOn() && ka == SLOT_OBJ && !e->inlining &&
                   stringOperand(e, e->depth - 2) &&
                   stringOperand(e, e->depth - 1)) {
            emitStringOrder(e);
        } else if ((op == OP_EQ || op == OP_NE) && ka == SLOT_OBJ &&
                   jitObjEquality() && e->callsOut && !e->inlining) {
            /* Two heap objects and nothing above could name them. Rather
             * than decline the whole body for one comparison, call the
             * interpreter's equality and carry on.
             *
             * `self._kind() == kind` on two TokenKinds is the shape, and it
             * is the SHORTEST CHAIN ON A HOT BODY in the self-hosted front
             * end: `_check` is one refusal away from compiling and carries
             * 3.2% of the interpreted work in lexer.jai. The enum arm
             * above cannot help it -- that one needs one side to be a
             * FOLDED constant, and here both are values.
             *
             * Leaves NZCV from a compare of the result against one, which
             * is what the shared tail below expects, so this arm ends the
             * same way every other arm in the chain does. */
            settleAll(e);
            if (!emitDescriptor(e, NULL_VAL, e->depth - 2, 2,
                                (void *)&jitValuesEqual)) {
                return false;
            }
            emit(e, jaiA64LdrByte(JIT_SCRATCH_A, 31,
                                  e->descOffset +
                                      (unsigned)offsetof(JitCallDesc,
                                                         result) + 8));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, 1));
            e->wroteHeap = true;
        } else {                /* Named: this closed the compare chain with a bare refusal, so
             * a census showed 80 declines reading only "OP_NE", with
             * nothing in them to act on. */
            return subWhy(e, "a comparison of %s with %s",
                          slotKindName(e->stack[e->depth - 2]),
                          slotKindName(e->stack[e->depth - 1]));
        }
        unsigned dropA, dropB;
        /* Raw: the compare has read both, they are going away, and
         * materialising them would put an fmov on the hot path for a
         * register nothing will read. */
        if (!popValueRaw(e, &dropB, NULL)) return false;
        if (!popValueRaw(e, &dropA, NULL)) return false;
        if (!pushValue(e, SLOT_BOOL, 0, NULL)) return false;
        emit(e, jaiA64CsetX(pushReg(e) - 1, cond));
        off += 1;
        break;
    } while (0);
    *offp = off;
    return true;
}

bool emitIsTest(Emit *e, uint8_t op, int *offp) {
    int off = *offp;
    do {
        if (e->depth < 2) return false;
        /* Only against a literal null -- but reading that off the kind
         * alone was wrong, and the arm below sat unreachable for every
         * `is null` anyone writes. SLOT_NULL is what a `-> void` call
         * leaves behind; the `null` in source is OP_NULL, which pushes
         * SLOT_MAYBE_INST so that `var x: Box? = null` does not give its
         * local two kinds. So the whole construct took the decline: 71
         * distinct functions on a self-hosted compile, and `is null`
         * outnumbers `is SomeClass` in lib/jaithon 661 to 12. stackNullLit
         * is the half of the question the kind cannot carry, and it is a
         * proof rather than a guess, so it is dropped at any offset a
         * branch can reach (see clearStackProofs).
         *
         * Measured, `for x in xs { if x is null { t += 1 } }` over a 2M
         * list[Node?] with no null in it, twenty passes, whole process and
         * best of three under scripts/gpu_lock.sh: 635 ms declined against
         * 223 ms compiled, 2.85x with ~150 ms of list building inside both
         * figures. The self-hosted compile it was expected to move did NOT
         * move (4932 ms against 4945 ms, a wash): the 71 functions mostly
         * stop at OP_GET_FIELD one instruction later, so this only clears
         * the first of two gates.
         *
         * `x is SomeClass` is still declined. It is a type test through
         * valueMatchesType, nothing here would be right for it, and the
         * unarmed deopt it used to take left the operand model a slot
         * deeper than the bytecode -- a body that compiles only to bail at
         * the same instruction every iteration is slower than one never
         * compiled at all. */
        if (e->stack[e->depth - 1] != SLOT_NULL &&
            !(e->stack[e->depth - 1] == SLOT_MAYBE_INST &&
              e->stackNullLit[e->depth - 1])) {
            e->whyNot = "an `is` against something other than null";
            return false;
        }
        SlotKind sk = e->stack[e->depth - 2];
        bool wantNull = (op == OP_IS);
        unsigned dropNull, dropSubject;

        if (sk == SLOT_MAYBE_INST) {
            unsigned rs = xHeldIn(e, e->valueDepth - 2);
            emit(e, jaiA64SubsXImm(31, rs, 0));
            if (!popValueRaw(e, &dropNull, NULL)) return false;
            if (!popValueRaw(e, &dropSubject, NULL)) return false;
            if (!pushValue(e, SLOT_BOOL, 0, NULL)) return false;
            emit(e, jaiA64CsetX(pushReg(e) - 1,
                                wantNull ? JAI_A64_EQ : JAI_A64_NE));
        } else if (sk == SLOT_INT || sk == SLOT_FLOAT || sk == SLOT_BOOL ||
                   sk == SLOT_INST || sk == SLOT_LIST || sk == SLOT_OBJ) {
            /* None of these kinds can hold a null, so the answer is known
             * here and the compare never runs. */
            if (!popValueRaw(e, &dropNull, NULL)) return false;
            if (!popValueRaw(e, &dropSubject, NULL)) return false;
            if (!pushValue(e, SLOT_BOOL, 0, NULL)) return false;
            emit(e, jaiA64MovzX(pushReg(e) - 1, wantNull ? 0u : 1u, 0));
        } else if (sk == SLOT_NULL) {
            if (!popValueRaw(e, &dropNull, NULL)) return false;
            if (!popValueRaw(e, &dropSubject, NULL)) return false;
            if (!pushValue(e, SLOT_BOOL, 0, NULL)) return false;
            emit(e, jaiA64MovzX(pushReg(e) - 1, wantNull ? 1u : 0u, 0));
        } else {
            e->whyNot = "an `is null` on a kind with no null representation";
            return false;
        }
        off += 1;
        break;
    } while (0);
    *offp = off;
    return true;
}

bool emitJumpIfCmpFalse(Emit *e, const uint8_t *code, int *offp) {
    int off = *offp;
    do {
        uint8_t  cmp  = code[off + 1];
        int16_t  jump = jaiReadI16(code + off + 2);
        if (e->depth < 2) return subWhy(e, "a compare with nothing under it");
        SlotKind ka = e->stack[e->depth - 2], kb = e->stack[e->depth - 1];
        if (ka != kb) {
            bool mixable =
                (ka == SLOT_INST && kb == SLOT_MAYBE_INST) ||
                (ka == SLOT_MAYBE_INST && kb == SLOT_INST) ||
                nullLiteralPair(e, cmp, ka, kb);
            /* Named for the same reason as the unfused twin above: the two
             * operands' kinds are the entire question at this arm. */
            if (!mixable) {
                return subWhy(e, "a compare of a %s with a %s",
                              slotKindName(ka), slotKindName(kb));
            }
            ka = kb = SLOT_MAYBE_INST;
        }
        if (!holdsRegister(ka)) {
            return subWhy(e, "a compare of two %ss, which hold no register",
                          slotKindName(ka));
        }
        /* See the OP_LT..OP_GE arm: operands are read through xHeldIn, a
         * literal right-hand side becomes the compare's own immediate,
         * and the paths that guard settle first. */
        int64_t kcmp2 = 0;
        bool foldCmp2 = ka == SLOT_INT &&
                        pendingImm12(e, e->valueDepth - 1, &kcmp2);
        unsigned cond;
        if (!negatedCondition(cmp, &cond)) {
            return subWhy(e, "a fused compare whose condition has no "
                          "negation");
        }
        if (ka == SLOT_FLOAT) {
            /* fcmp's answers for the ordered comparisons are not the
             * signed-integer ones: less-than is MI and less-or-equal is
             * LS, because an unordered result must come out false. */
            switch (cmp) {
            case OP_LT: cond = JAI_A64_GE; break;   /* not (a < b)  */
            case OP_LE: cond = JAI_A64_GT; break;   /* not (a <= b) */
            case OP_GT: cond = JAI_A64_LS; break;   /* not (a > b)  */
            case OP_GE: cond = JAI_A64_MI; break;   /* not (a >= b) */
            case OP_EQ: cond = JAI_A64_NE; break;
            case OP_NE: cond = JAI_A64_EQ; break;
            default: return false;
            }
            /* Released BEFORE the operands are read, so `fpOperand`
             * answers out of the bank: this arm is on the borrow
             * whitelist, so an operand can still be held in a local's own
             * d register here, and nanToDeopt records -- which
             * deoptRecordAt asserts nothing may be borrowing at. Costs one
             * fmov when a borrow is live, against declining the entire
             * loop, which is what `for v in xs { if v != 0.0 ... }` used
             * to do. Sound at this point because nothing has branched yet.
             */
            fpReleaseAll(e);
            unsigned db = fpOperand(e, e->valueDepth - 1);
            unsigned da = fpOperand(e, e->valueDepth - 2);
            settleAll(e);          /* nanToDeopt records */
            emit(e, jaiA64FcmpD(da, db));
            nanToDeopt(e);
        /* Bools, for the reason given at the unfused arm. The fused
         * form is the one an `if a != b` actually emits, so without it
         * here the unfused arm above would almost never be reached. */
        } else if (ka == SLOT_INT || ka == SLOT_MAYBE_INST ||
                   ka == SLOT_BOOL) {
            unsigned ra = xHeldIn(e, e->valueDepth - 2);
            if (foldCmp2) {
                e->kPend &= ~(1u << (e->valueDepth - 1));
                emitCmpImm(e, ra, kcmp2);
            } else {
                emit(e, jaiA64SubsXReg(31, ra,
                                       xHeldIn(e, e->valueDepth - 1)));
            }
        } else if ((cmp == OP_EQ || cmp == OP_NE) && ka == SLOT_OBJ &&
                   (e->stackUnit[e->depth - 2] ||
                    e->stackUnit[e->depth - 1])) {
            /* Two payload-less enum values are equal exactly when they are the
             * same object: a variant with no payload has no state to tell
             * instances apart, so every mention of it yields the one
             * shared EnumVariant::unit, and enumValsEqual reduces to
             * type-and-tag at zero payload. One side must be the folded
             * constant (stackUnit, a proof), never merely a value observed
             * to be one. The other may be anything on the heap: for two
             * VAL_OBJs of different Obj types jaiValuesEqual returns false
             * without consulting an `__eq__`, which is the same answer the
             * pointer compare gives.
             *
             * Needed for the `Enum.Variant` fold to pay at all -- with the
             * fold alone, OP_EQ and OP_JUMP_IF_CMP_FALSE simply became the new
             * refusal and the same functions still declined. */
            unsigned rb = xHeldIn(e, e->valueDepth - 1);
            unsigned ra = xHeldIn(e, e->valueDepth - 2);
            emit(e, jaiA64SubsXReg(31, ra, rb));
        } else if ((cmp == OP_EQ || cmp == OP_NE) && ka == SLOT_OBJ &&
                   IS_STRING(e->stackSeen[e->depth - 2]) &&
                   IS_STRING(e->stackSeen[e->depth - 1]) &&
                   !preferLeafEquality(e, e->depth - 2, e->depth - 1)) {
            settleAll(e);          /* this path guards */
            unsigned rb = valueXReg(e, e->valueDepth - 1);
            unsigned ra = valueXReg(e, e->valueDepth - 2);
            for (unsigned side = 0; side < 2; side++) {
                unsigned r = side == 0 ? ra : rb;
                /* See the same skip in OP_EQ. */
                if (e->stackAscii[e->depth - (side == 0 ? 2u : 1u)]) continue;
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, r,
                                   (unsigned)offsetof(Obj, type)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_STRING));
                branchOnDeopt(e, JAI_A64_NE);
                emit(e, jaiA64LdrByte(JIT_SCRATCH_A, r,
                                      (unsigned)offsetof(Obj, subFlag)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, 0));
                branchOnDeopt(e, JAI_A64_EQ);
            }
            emit(e, jaiA64SubsXReg(31, ra, rb));
        } else if (isOrdering(cmp) && ka == SLOT_OBJ &&
                   oneBytePair(e, e->depth - 2, e->depth - 1)) {
            /* See the same arm in OP_LT..OP_GE. `if c < "0" { break }`
             * fuses its compare into the branch and would otherwise have
             * declined here after the unfused twin already compiled. */
            settleAll(e);
            unsigned rb = valueXReg(e, e->valueDepth - 1);
            unsigned ra = valueXReg(e, e->valueDepth - 2);
            emitOneByteString(e, e->depth - 2, ra, JIT_SCRATCH_C);
            emitOneByteString(e, e->depth - 1, rb, JIT_SCRATCH_D);
            emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_C, JIT_SCRATCH_D));
        /* See the same arm in OP_LT..OP_GE. This is the one the probe
         * hits: `if a < b { .. }` fuses its compare into the branch, and
         * without it here the unfused twin compiles and the shape anyone
         * actually writes still declines. */
        } else if (jitStrCmpOn() && ka == SLOT_OBJ && !e->inlining &&
                   stringOperand(e, e->depth - 2) &&
                   stringOperand(e, e->depth - 1)) {
            emitStringOrder(e);
        } else if ((cmp == OP_EQ || cmp == OP_NE) && ka == SLOT_OBJ &&
                   jitObjEquality() && e->callsOut && !e->inlining) {
            /* Two heap objects and nothing above could name them. Rather
             * than decline the whole body for one comparison, call the
             * interpreter's equality and carry on.
             *
             * `self._kind() == kind` on two TokenKinds is the shape, and it
             * is the SHORTEST CHAIN ON A HOT BODY in the self-hosted front
             * end: `_check` is one refusal away from compiling and carries
             * 3.2% of the interpreted work in lexer.jai. The enum arm
             * above cannot help it -- that one needs one side to be a
             * FOLDED constant, and here both are values.
             *
             * Leaves NZCV from a compare of the result against one, which
             * is what the shared tail below expects, so this arm ends the
             * same way every other arm in the chain does. */
            settleAll(e);
            if (!emitDescriptor(e, NULL_VAL, e->depth - 2, 2,
                                (void *)&jitValuesEqual)) {
                return false;
            }
            emit(e, jaiA64LdrByte(JIT_SCRATCH_A, 31,
                                  e->descOffset +
                                      (unsigned)offsetof(JitCallDesc,
                                                         result) + 8));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, 1));
            e->wroteHeap = true;
        } else {                /* Named: this closed the compare chain with a bare refusal, so
             * a census showed 80 declines reading only "OP_NE", with
             * nothing in them to act on. */
            return subWhy(e, "a comparison of %s with %s",
                          slotKindName(e->stack[e->depth - 2]),
                          slotKindName(e->stack[e->depth - 1]));
        }
        unsigned dropA2, dropB2;
        if (!popValue(e, &dropB2, NULL)) return false;
        if (!popValue(e, &dropA2, NULL)) return false;
        branchTo(e, (uint32_t)((int32_t)(off + 4) + jump), true, cond);
        off += 4;
        break;
    } while (0);
    *offp = off;
    return true;
}

bool emitJumpIfCmpLocalK(Emit *e, ObjFunction *fn, const uint8_t *code,
                         int *offp) {
    int off = *offp;
    do {
        uint8_t  cmp  = code[off + 1];
        unsigned slot = jaiReadU16(code + off + 2);
        uint32_t kIdx = jaiReadU24(code + off + 4);
        int16_t  jump = jaiReadI16(code + off + 7);
        uint32_t next = (uint32_t)(off + 9);

        unsigned cond;
        if (!negatedCondition(cmp, &cond)) return false;
        if (!localInRange(e, slot)) return false;
        if (kIdx >= (uint32_t)fn->chunk.constants.count) return false;
        Value k = fn->chunk.constants.data[kIdx];

        /* `if c == "{"` where c came out of a string index. Every hand-written scanner in the language is
         * this shape, and the peephole folds it to exactly this instruction, so declining it declined the
         * whole enclosing function -- json_parse's `value` dispatches on six of them. Two interned strings
         * are equal exactly when they are the same object, which the OP_EQ arm already relies on; the
         * difference here is that one side is a constant, so its interning is settled at compile time and
         * only the local needs guarding. Nothing has been written yet, so a guard resumes at this very
         * instruction. */
        Value kLocalSeen = seenLocal(e, slot);
        /* seenLocal has already dropped a VAL_OBJ over a null pointer,
         * which is what an unreached slot can hold. */
        bool kSeenString = IS_STRING(kLocalSeen);
        bool kLocalInterned = kSeenString &&
                              JAI_STR_INTERNED(AS_STRING(kLocalSeen));
        if ((cmp == OP_EQ || cmp == OP_NE) &&
            e->localKind[slot] == SLOT_OBJ && IS_STRING(k) &&
            JAI_STR_INTERNED(AS_STRING(k)) &&
            kSeenString &&
            !(jitStrCmpEqOn() && jitStrCmpOn() && !e->inlining &&
              !kLocalInterned)) {
            if (slot == 0) e->usesSlot0 = true;
            settleAll(e);          /* this path guards */
            unsigned rs = localIn(e, slot, JIT_SCRATCH_C);
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, rs,
                               (unsigned)offsetof(Obj, type)));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_STRING));
            branchOnDeopt(e, JAI_A64_NE);
            /* Not interned means content equality is not pointer equality,
             * and the interpreter is the one that knows how to tell. */
            emit(e, jaiA64LdrByte(JIT_SCRATCH_A, rs,
                                  (unsigned)offsetof(Obj, subFlag)));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, 0));
            branchOnDeopt(e, JAI_A64_EQ);
            emitConst64(e, JIT_SCRATCH_B, (int64_t)(uintptr_t)AS_OBJ(k));
            emit(e, jaiA64SubsXReg(31, rs, JIT_SCRATCH_B));
            branchTo(e, (uint32_t)((int32_t)next + jump), true, cond);
            off += 9;
            break;
        }

        /* `if word < "middle"`, and `==` against a constant the arm above
         * declined because it is not interned. Same leaf call as the arm
         * at OP_JUMP_IF_CMP_FALSE, after the peephole folded the local and
         * the constant into one instruction -- which is what every parser
         * and every dispatch-on-a-name in this language actually emits, so
         * without this the unfused twin compiles and the shape anyone
         * types still declines.
         *
         * The LOCAL is the left operand and the constant the right; the
         * interpreter's own arm spells that out, and reversing it would be
         * a silent wrong answer for four of the six operators rather than
         * a crash. The constant needs no guard: it is a string the emitter
         * is holding, so only the local's Obj.type is in question. */
        if (jitStrCmpOn() && !e->inlining && IS_STRING(k) &&
            e->localKind[slot] == SLOT_OBJ &&
            IS_STRING(seenLocal(e, slot))) {
            if (slot == 0) e->usesSlot0 = true;
            settleAll(e);          /* this path guards */
            fpSyncAll(e);          /* and calls; see emitStringOrder */
            unsigned rs = localIn(e, slot, JIT_SCRATCH_C);
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, rs,
                               (unsigned)offsetof(Obj, type)));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_STRING));
            branchOnDeopt(e, JAI_A64_NE);
            /* rs is a local's home (x19..) or JIT_SCRATCH_C, never x0 --
             * but the test costs nothing and a `mov` reading a register it
             * has already written is a miscompile, not a decline. */
            if (rs != 0) emit(e, jaiA64MovX(0, rs));
            emitConst64(e, 1, (int64_t)(uintptr_t)AS_OBJ(k));
            emitConst64(e, JIT_SCRATCH_A,
                        (int64_t)(uintptr_t)&jaiStringOrder);
            noteScratchClobber(e);
            emit(e, jaiA64Blr(JIT_SCRATCH_A));
            emit(e, jaiA64SubsXImm(31, 0, 0));
            branchTo(e, (uint32_t)((int32_t)next + jump), true, cond);
            off += 9;
            break;
        }

        /* The same instruction over floats: `if coefficient == 0.0`, the
         * early-out every numeric kernel opens its inner loop with. Only
         * the integer form was emitted, so a single float guard declined
         * the whole loop around it -- which is what left the JPEG inverse
         * DCT interpreted.
         *
         * fcmp's answers are not the signed-integer ones (an unordered
         * result has to come out false both ways), so the conditions are
         * the ones OP_JUMP_IF_CMP_FALSE's float arm uses, and an actual
         * NaN operand resumes in the interpreter. */
        if (e->localKind[slot] == SLOT_FLOAT && IS_FLOAT(k) &&
            AS_FLOAT(k) == AS_FLOAT(k)) {
            switch (cmp) {
            case OP_LT: cond = JAI_A64_GE; break;
            case OP_LE: cond = JAI_A64_GT; break;
            case OP_GT: cond = JAI_A64_LS; break;
            case OP_GE: cond = JAI_A64_MI; break;
            case OP_EQ: cond = JAI_A64_NE; break;
            case OP_NE: cond = JAI_A64_EQ; break;
            default: return false;
            }
            if (slot == 0) e->usesSlot0 = true;
            /* This arm reads a LOCAL, so unlike the stack-operand compare
             * it never pops the entries that are borrowing -- and a borrow
             * still live when nanToDeopt records is what deoptRecordAt
             * asserts against. Releasing at the top of the instruction is
             * sound here because nothing has branched yet. */
            settleAll(e);          /* nanToDeopt records */
            fpReleaseAll(e);
            localInFp(e, slot, JIT_FSCRATCH_A);
            if (AS_FLOAT(k) == 0.0) {
                emit(e, jaiA64FcmpDZero(JIT_FSCRATCH_A));
            } else {
                int64_t bits;
                double kd = AS_FLOAT(k);
                memcpy(&bits, &kd, sizeof bits);
                emitConst64(e, JIT_SCRATCH_A, bits);
                emit(e, jaiA64FmovDX(JIT_FSCRATCH_B, JIT_SCRATCH_A));
                emit(e, jaiA64FcmpD(JIT_FSCRATCH_A, JIT_FSCRATCH_B));
            }
            nanToDeopt(e);
            branchTo(e, (uint32_t)((int32_t)next + jump), true, cond);
            off += 9;
            break;
        }

        if (e->localKind[slot] != SLOT_INT) return false;
        if (slot == 0) e->usesSlot0 = true;
        if (!IS_INT(k)) return false;

        /* `while i < n` and `if n < 2` are the same instruction here, and the constant fits the compare's
         * own imm12 far more often than not, so it costs one instruction rather than a movz plus a three-register subs. */
        int64_t kv = AS_INT(k);
        if (kv >= -4095 && kv <= 4095) {
            emitCmpImm(e, localIn(e, slot, JIT_SCRATCH_C), kv);
        } else {
            emitConst64(e, JIT_SCRATCH_A, kv);
            emit(e, jaiA64SubsXReg(31, localIn(e, slot, JIT_SCRATCH_C),
                                   JIT_SCRATCH_A));
        }
        branchTo(e, (uint32_t)((int32_t)next + jump), true, cond);
        off += 9;
        break;
    } while (0);
    *offp = off;
    return true;
}

JitArmResult emitMembership(Emit *e, const uint8_t *code, int *offp) {
    int off = *offp;
    do {
        /* `x in c`. No arm existed, so a membership test ENDED THE WALK:
         * `if k in seen` is the shape of every dedup loop in the corpus and
         * everything after it ran interpreted.
         *
         * The containment itself is not made faster -- it is the same
         * jaiContainsOp the interpreter runs, called out to. What the arm
         * buys is the body around it, which is the whole point of a row
         * over a call that is cheap next to its loop.
         *
         * `not in` is the same call with the sense flipped, in its own
         * entry point rather than an argc flag -- a wider descriptor would
         * name a stack entry past the operands. */
        if (!jitMembership() || !e->callsOut || e->depth < 2) {
            goto unarmedOpcode;
        }
        if (!emitDescriptor(e, NULL_VAL, e->depth - 2, 2,
                            code[off] == OP_IN ? (void *)&jitContains
                                               : (void *)&jitNotContains)) {
            return false;
        }
        for (unsigned i = 0; i < 2; i++) {
            unsigned r;
            if (!popValue(e, &r, NULL)) return false;
        }
        if (!pushValue(e, SLOT_BOOL, 0, NULL)) return false;
        emit(e, jaiA64LdrByte(pushReg(e) - 1, 31,
                              e->descOffset +
                                  (unsigned)offsetof(JitCallDesc, result) +
                                  8));
        /* Containment is not pure: a class can define __contains__, so the
         * call may run Jaithon code that writes. Leaving this unset marked
         * every body holding an `in` jitFuncNoWrite, which lets a direct
         * caller finish the callee by RE-RUNNING it from the start on a
         * bail -- and re-running the writes with it. */
        e->wroteHeap = true;
        off += 1;
        break;
    } while (0);
    *offp = off;
    return JIT_ARM_OK;
unarmedOpcode:
    *offp = off;
    return JIT_ARM_UNARMED;
}

#endif /* __aarch64__ || __arm64__ */
