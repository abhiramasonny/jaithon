/* jit_body_fused.c -- the fused local-arithmetic arms of the opcode walk. */

#include "vm/jit/jit_arm64.h"
#include "vm/jit/jit_field_read.h"

#include <stddef.h>
#include <stdint.h>
#include "vm/jit/jit_internal.h"

#if (defined(__aarch64__) || defined(__arm64__))

bool emitAddLocals(Emit *e, const uint8_t *code, int *offp) {
    int off = *offp;
    do {
        unsigned a = jaiReadU16(code + off + 1);
        unsigned b = jaiReadU16(code + off + 3);
        if (!localInRange(e, a) || !localInRange(e, b)) {
            return subWhy(e, "a fused add of a local the model does not "
                          "cover");
        }
        SlotKind ka2 = e->localKind[a];
        /* Named: this was 148 declines across four compiler files reading
         * only "OP_ADD_LOCALS", with nothing in them to act on. */
        if (ka2 != e->localKind[b]) {
            return subWhy(e, "a fused add of a %s and a %s",
                          slotKindName(ka2), slotKindName(e->localKind[b]));
        }
        if (ka2 == SLOT_OBJ && jitConcatLocals() && e->callsOut &&
            !e->inlining) {
            /* `out = out + piece` -- string building, and the fused form is
             * the one real code emits. The stack `+` already had a concat
             * arm; this opcode did not, so an OSR loop doing the commonest
             * thing a lexer does declined WHOLE: the probe is 4,001,920
             * interpreted instructions and the census showed 28 of these in
             * parser.jai alone reading only "a fused add of two objects".
             *
             * The operands are slot numbers here and emitStringConcat wants
             * stack entries, so they are pushed first. That is the cost the
             * fusion existed to avoid, and it is nothing next to a call
             * that allocates a string. Both operands are guarded inside
             * emitStringConcat, so a sample that turns out wrong deopts at
             * this instruction with both locals untouched. */
            Value sa = seenLocal(e, a);
            Value sb = seenLocal(e, b);
            if (IS_STRING(sa) && IS_STRING(sb)) {
                if (a == 0 || b == 0) e->usesSlot0 = true;
                if (!pushLocalAsValue(e, a)) return false;
                if (!pushLocalAsValue(e, b)) return false;
                if (!emitStringConcat(e, sa)) return false;
                off += 5;
                break;
            }
        }
        if (ka2 != SLOT_INT && ka2 != SLOT_FLOAT) {
            return subWhy(e, "a fused add of two %ss", slotKindName(ka2));
        }
        if (a == 0 || b == 0) e->usesSlot0 = true;
        if (!pushValue(e, ka2, 0, NULL)) return false;
        if (ka2 == SLOT_FLOAT && !e->dynamicLocal[a] &&
            !e->dynamicLocal[b] && !e->fpOff) {
            unsigned idx = e->valueDepth - 1;
            /* Either operand already in a d register of its own is read
             * from there. `x2 + y2` was two fmovs and an add. */
            unsigned da, db;
            if (e->slotFpReg[a] != 0) {
                da = e->slotFpReg[a];
            } else {
                localInFp(e, a, fpRegAt(e, idx));
                da = fpRegAt(e, idx);
            }
            if (e->slotFpReg[b] != 0) {
                db = e->slotFpReg[b];
            } else {
                localInFp(e, b, JIT_FP_BANK + JIT_MAX_SAVED);
                db = JIT_FP_BANK + JIT_MAX_SAVED;
            }
            emit(e, jaiA64FaddD(fpRegAt(e, idx), da, db));
            fpClaim(e, idx);
            off += 5;
            break;
        }
        {
            unsigned ra2 = localIn(e, a, JIT_SCRATCH_C);
            unsigned rb2 = localIn(e, b, JIT_SCRATCH_D);
            if (ka2 == SLOT_FLOAT) {
                emit(e, jaiA64FmovDX(JIT_FSCRATCH_A, ra2));
                emit(e, jaiA64FmovDX(JIT_FSCRATCH_B, rb2));
                emit(e, jaiA64FaddD(JIT_FSCRATCH_A, JIT_FSCRATCH_A,
                                    JIT_FSCRATCH_B));
                emit(e, jaiA64FmovXD(pushReg(e) - 1, JIT_FSCRATCH_A));
            } else {
                emit(e, jaiA64AddsX(pushReg(e) - 1, ra2, rb2));
                branchOnOverflow(e, 0u, JAI_A64_VS);
            }
        }
        off += 5;
        break;
    } while (0);
    *offp = off;
    return true;
}

bool emitAddBind(Emit *e, const uint8_t *code, int *offp) {
    int off = *offp;
    do {
        /* `ADD; BIND a` fused. Floats go through the same fmov pair the
         * plain add uses; ints keep the overflow check. */
        unsigned slot = jaiReadU16(code + off + 1);
        if (!localInRange(e, slot)) return false;
        if (slot == 0) e->usesSlot0 = true;
        if (!e->fpOff && e->depth >= 2 && !e->dynamicLocal[slot] &&
            e->stack[e->depth - 1] == SLOT_FLOAT &&
            e->stack[e->depth - 2] == SLOT_FLOAT) {
            /* The whole operation stays in the FP bank: two operands that
             * are already there, one instruction, and a store straight out
             * of a d register. */
            if (!adoptLocalKind(e, slot, SLOT_FLOAT, 0, NULL)) {
                e->whyNot = kindClash(e, slot);
                return false;
            }
            unsigned ib = e->valueDepth - 1, ia = e->valueDepth - 2;
            unsigned db = fpOperand(e, ib), da = fpOperand(e, ia);
            unsigned rx; SlotKind kx;
            if (!popValueRaw(e, &rx, &kx)) return false;
            if (!popValueRaw(e, &rx, &kx)) return false;
            unsigned dd = fpBindDest(e, slot, fpRegAt(e, ia));
            emit(e, jaiA64FaddD(dd, da, db));
            localOutFp(e, slot, dd);
            off += 3;
            break;
        }
        /* `text = text + piece` on strings. Out to jaiStringConcat behind a
         * pair of type guards -- see emitStringConcat. Before this arm the
         * refusal below gave up word_freq's entire `main` loop, which is
         * this one instruction plus the LCG arithmetic around it. */
        {
            Value csample;
            if (concatOperands(e, &csample)) {
                /* The kind is adopted AFTER the call, not before: the
                 * descriptor's root fill reads every object-kinded local,
                 * and until the store below this slot still holds the old
                 * string -- claiming the new kind first would describe a
                 * slot the call has not written yet. */
                if (!emitStringConcat(e, csample)) return false;
                if (!adoptLocalKindSeen(e, slot, SLOT_OBJ, 0, NULL,
                                        csample)) {
                    e->whyNot = kindClash(e, slot);
                    return false;
                }
                unsigned rc;
                if (!popValue(e, &rc, NULL)) return false;
                localOut(e, slot, rc);
                off += 3;
                break;
            }
        }
        unsigned rb, ra;
        SlotKind kb, ka;
        if (!popValue(e, &rb, &kb)) return false;
        if (!popValue(e, &ra, &ka)) return false;
        if (ka != kb) return false;
        if (!adoptLocalKind(e, slot, ka, 0, NULL)) {
            e->whyNot = kindClash(e, slot);
            return false;
        }
        unsigned rd = localDest(e, slot);
        if (ka == SLOT_FLOAT) {
            emit(e, jaiA64FmovDX(JIT_FSCRATCH_A, ra));
            emit(e, jaiA64FmovDX(JIT_FSCRATCH_B, rb));
            emit(e, jaiA64FaddD(JIT_FSCRATCH_A, JIT_FSCRATCH_A,
                                JIT_FSCRATCH_B));
            emit(e, jaiA64FmovXD(rd, JIT_FSCRATCH_A));
        } else if (ka == SLOT_INT) {
            rd = ovfDest(e, rd);   /* localOut copies it home below */
            emit(e, jaiA64AddsX(rd, ra, rb));
            branchOnOverflow(e, 0u, JAI_A64_VS);
        } else {
            e->whyNot = "add-bind of a kind that is neither int nor float";
            return false;
        }
        localOut(e, slot, rd);
        off += 3;
        break;
    } while (0);
    *offp = off;
    return true;
}

bool emitCmpLocalConstLt(Emit *e, const uint8_t *code, int *offp) {
    int off = *offp;
    do {
        unsigned slot = jaiReadU16(code + off + 1);
        int16_t  imm  = jaiReadI16(code + off + 3);
        if (!localInRange(e, slot)) return false;
        if (e->localKind[slot] != SLOT_INT) return false;
        if (slot == 0) e->usesSlot0 = true;

        /* Compare before pushing: the compare reads the local, and the
         * pushed entry is only the bool the flags produce. */
        if (imm >= -4095 && imm <= 4095) {
            emitCmpImm(e, localIn(e, slot, JIT_SCRATCH_C), imm);
        } else {
            emitConst64(e, JIT_SCRATCH_A, imm);
            emit(e, jaiA64SubsXReg(31, localIn(e, slot, JIT_SCRATCH_C),
                                   JIT_SCRATCH_A));
        }
        if (!pushValue(e, SLOT_BOOL, 0, NULL)) return false;
        emit(e, jaiA64CsetX(pushReg(e) - 1, JAI_A64_LT));
        off += 5;
        break;
    } while (0);
    *offp = off;
    return true;
}

bool emitAddIntConst(Emit *e, const uint8_t *code, int *offp) {
    int off = *offp;
    do {
        unsigned slot = jaiReadU16(code + off + 1);
        int16_t  imm  = jaiReadI16(code + off + 3);
        if (!localInRange(e, slot)) return false;
        if (e->localKind[slot] != SLOT_INT) return false;
        if (slot == 0) e->usesSlot0 = true;
        if (!pushValue(e, SLOT_INT, 0, NULL)) return false;
        {
            unsigned dst = pushReg(e) - 1;
            unsigned cur = localIn(e, slot, JIT_SCRATCH_C);
            /* imm12, so a step outside +/-4095 still goes through a
             * register. See OP_INC_LOCAL for why `subs` is the negative
             * arm rather than a negated constant. */
            if (imm >= 0 && imm <= 4095) {
                emit(e, jaiA64AddsXImm(dst, cur, (unsigned)imm));
            } else if (imm < 0 && imm >= -4095) {
                emit(e, jaiA64SubsXImm(dst, cur, (unsigned)(-(int)imm)));
            } else {
                emitConst64(e, JIT_SCRATCH_A, imm);
                emit(e, jaiA64AddsX(dst, cur, JIT_SCRATCH_A));
            }
        }
        branchOnOverflow(e, 0u, JAI_A64_VS);
        off += 5;
        break;
    } while (0);
    *offp = off;
    return true;
}

bool emitSubIntConst(Emit *e, const uint8_t *code, int *offp) {
    int off = *offp;
    do {
        unsigned slot = jaiReadU16(code + off + 1);
        int16_t  imm  = jaiReadI16(code + off + 3);
        if (!localInRange(e, slot)) return false;
        if (e->localKind[slot] != SLOT_INT) return false;
        if (slot == 0) e->usesSlot0 = true;
        if (!pushValue(e, SLOT_INT, 0, NULL)) return false;
        {
            unsigned dst = pushReg(e) - 1;
            unsigned cur = localIn(e, slot, JIT_SCRATCH_C);
            if (imm >= 0 && imm <= 4095) {
                emit(e, jaiA64SubsXImm(dst, cur, (unsigned)imm));
            } else if (imm < 0 && imm >= -4095) {
                emit(e, jaiA64AddsXImm(dst, cur, (unsigned)(-(int)imm)));
            } else {
                emitConst64(e, JIT_SCRATCH_A, imm);
                emit(e, jaiA64SubsXReg(dst, cur, JIT_SCRATCH_A));
            }
        }
        branchOnOverflow(e, 1u, JAI_A64_VS);
        off += 5;
        break;
    } while (0);
    *offp = off;
    return true;
}

bool emitMulIntConst(Emit *e, const uint8_t *code, int *offp) {
    int off = *offp;
    do {
        unsigned slot = jaiReadU16(code + off + 1);
        int16_t  imm  = jaiReadI16(code + off + 3);
        if (!localInRange(e, slot)) return false;
        if (e->localKind[slot] != SLOT_INT) return false;
        if (slot == 0) e->usesSlot0 = true;
        if (!pushValue(e, SLOT_INT, 0, NULL)) return false;
        unsigned dst = pushReg(e) - 1;
        unsigned cur = localIn(e, slot, JIT_SCRATCH_C);
        unsigned rt = ovfDest(e, dst);
        /* MUL has no immediate form on this encoder, so the constant goes
         * into JIT_SCRATCH_D -- left free by `cur` and `rt` above, so it
         * cannot collide with either even when ovfDest hands back
         * JIT_SCRATCH_B inside a `try`. Same overflow test as plain
         * OP_MUL: the product overflows exactly when smulh's high half is
         * not the low half's sign bit replicated, and it shares that
         * arm's overflow-stub slot (2, the `*` message) since it is the
         * same operator. */
        emitConst64(e, JIT_SCRATCH_D, imm);
        emit(e, jaiA64SmulhX(JIT_SCRATCH_A, cur, JIT_SCRATCH_D));
        emit(e, jaiA64MulX(rt, cur, JIT_SCRATCH_D));
        emit(e, jaiA64SubsXAsr(31, JIT_SCRATCH_A, rt, 63));
        branchOnOverflow(e, 2u, JAI_A64_NE);
        if (rt != dst) emit(e, jaiA64MovX(dst, rt));
        off += 5;
        break;
    } while (0);
    *offp = off;
    return true;
}

bool emitMulBind(Emit *e, const uint8_t *code, int *offp) {
    int off = *offp;
    do {
        unsigned slot = jaiReadU16(code + off + 1);
        if (!localInRange(e, slot)) return false;
        if (slot == 0) e->usesSlot0 = true;
        if (!e->fpOff && e->depth >= 2 && !e->dynamicLocal[slot] &&
            e->stack[e->depth - 1] == SLOT_FLOAT &&
            e->stack[e->depth - 2] == SLOT_FLOAT) {
            if (!adoptLocalKind(e, slot, SLOT_FLOAT, 0, NULL)) {
                e->whyNot = kindClash(e, slot);
                return false;
            }
            unsigned ib = e->valueDepth - 1, ia = e->valueDepth - 2;
            unsigned db = fpOperand(e, ib), da = fpOperand(e, ia);
            unsigned rx; SlotKind kx;
            if (!popValueRaw(e, &rx, &kx)) return false;
            if (!popValueRaw(e, &rx, &kx)) return false;
            unsigned dd = fpBindDest(e, slot, fpRegAt(e, ia));
            emit(e, jaiA64FmulD(dd, da, db));
            localOutFp(e, slot, dd);
            off += 3;
            break;
        }
        unsigned rb, ra;
        SlotKind kb, ka;
        if (!popValue(e, &rb, &kb)) return false;
        if (!popValue(e, &ra, &ka)) return false;
        if (ka != kb) return false;
        if (!adoptLocalKind(e, slot, ka, 0, NULL)) {
            e->whyNot = kindClash(e, slot);
            return false;
        }
        unsigned rd = localDest(e, slot);
        if (ka == SLOT_FLOAT) {
            emit(e, jaiA64FmovDX(JIT_FSCRATCH_A, ra));
            emit(e, jaiA64FmovDX(JIT_FSCRATCH_B, rb));
            emit(e, jaiA64FmulD(JIT_FSCRATCH_A, JIT_FSCRATCH_A,
                                JIT_FSCRATCH_B));
            emit(e, jaiA64FmovXD(rd, JIT_FSCRATCH_A));
        } else if (ka == SLOT_INT) {
            rd = ovfDest(e, rd);   /* localOut copies it home below */
            emit(e, jaiA64SmulhX(JIT_SCRATCH_A, ra, rb));
            emit(e, jaiA64MulX(rd, ra, rb));
            emit(e, jaiA64SubsXAsr(31, JIT_SCRATCH_A, rd, 63));
            branchOnOverflow(e, 2u, JAI_A64_NE);
        } else {
            return subWhy(e, "arithmetic on a %s", slotKindName(ka));
        }
        localOut(e, slot, rd);
        off += 3;
        break;
    } while (0);
    *offp = off;
    return true;
}

bool emitSubBind(Emit *e, const uint8_t *code, int *offp) {
    int off = *offp;
    do {
        unsigned slot = jaiReadU16(code + off + 1);
        if (!localInRange(e, slot)) return false;
        if (slot == 0) e->usesSlot0 = true;
        if (!e->fpOff && e->depth >= 2 && !e->dynamicLocal[slot] &&
            e->stack[e->depth - 1] == SLOT_FLOAT &&
            e->stack[e->depth - 2] == SLOT_FLOAT) {
            if (!adoptLocalKind(e, slot, SLOT_FLOAT, 0, NULL)) {
                e->whyNot = kindClash(e, slot);
                return false;
            }
            unsigned ib = e->valueDepth - 1, ia = e->valueDepth - 2;
            unsigned db = fpOperand(e, ib), da = fpOperand(e, ia);
            unsigned rx; SlotKind kx;
            if (!popValueRaw(e, &rx, &kx)) return false;
            if (!popValueRaw(e, &rx, &kx)) return false;
            unsigned dd = fpBindDest(e, slot, fpRegAt(e, ia));
            emit(e, jaiA64FsubD(dd, da, db));
            localOutFp(e, slot, dd);
            off += 3;
            break;
        }
        unsigned rb, ra;
        SlotKind kb, ka;
        if (!popValue(e, &rb, &kb)) return false;
        if (!popValue(e, &ra, &ka)) return false;
        if (ka != kb) return false;
        if (!adoptLocalKind(e, slot, ka, 0, NULL)) {
            e->whyNot = kindClash(e, slot);
            return false;
        }
        unsigned rd = localDest(e, slot);
        if (ka == SLOT_FLOAT) {
            emit(e, jaiA64FmovDX(JIT_FSCRATCH_A, ra));
            emit(e, jaiA64FmovDX(JIT_FSCRATCH_B, rb));
            emit(e, jaiA64FsubD(JIT_FSCRATCH_A, JIT_FSCRATCH_A,
                                JIT_FSCRATCH_B));
            emit(e, jaiA64FmovXD(rd, JIT_FSCRATCH_A));
        } else if (ka == SLOT_INT) {
            rd = ovfDest(e, rd);   /* localOut copies it home below */
            emit(e, jaiA64SubsXReg(rd, ra, rb));
            branchOnOverflow(e, 1u, JAI_A64_VS);
        } else {
            return subWhy(e, "arithmetic on a %s", slotKindName(ka));
        }
        localOut(e, slot, rd);
        off += 3;
        break;
    } while (0);
    *offp = off;
    return true;
}

bool emitIncLocal(Emit *e, const uint8_t *code, int *offp) {
    int off = *offp;
    do {
        unsigned slot = jaiReadU16(code + off + 1);
        int8_t   imm  = (int8_t)code[off + 3];
        if (!localInRange(e, slot)) return false;
        if (e->localKind[slot] != SLOT_INT) return false;
        if (slot == 0) e->usesSlot0 = true;
        {
            unsigned cur = localIn(e, slot, JIT_SCRATCH_C);
            unsigned dst = ovfDest(e, localDest(e, slot));
            /* Step is an i8, always fits imm12, so the constant never needs a register of its own. `subs` for a
             * negative step rather than a negated `adds`: both set V for the operation actually performed, which is what the overflow guard below reads. */
            if (imm >= 0) {
                emit(e, jaiA64AddsXImm(dst, cur, (unsigned)imm));
            } else {
                emit(e, jaiA64SubsXImm(dst, cur, (unsigned)(-(int)imm)));
            }
            /* The guard is taken before the home is written, not after: it
             * resumes at this instruction inside a `try` (ovfDest), and
             * neither fpSyncAll nor a b.cond disturbs V or `dst`. */
            branchOnOverflow(e, 0u, JAI_A64_VS);
            localOut(e, slot, dst);
        }
        off += 4;
        break;
    } while (0);
    *offp = off;
    return true;
}

bool emitModIntConst(Emit *e, const uint8_t *code, int *offp) {
    int off = *offp;
    do {
        /* `<int k>; MOD` fused: k is known non-zero (fusion requires it); -1 goes back to the interpreter so
         * INT64_MIN %% -1 stays its problem. */
        int16_t imm = jaiReadI16(code + off + 1);
        if (imm == 0 || imm == -1) return false;
        if (e->depth == 0 || e->stack[e->depth - 1] != SLOT_INT) return false;
        unsigned rx = pushReg(e) - 1;

        /* A power of two is the low bits and nothing else -- see OP_MOD. */
        unsigned kshift;
        if (powerOfTwoShift(imm, &kshift)) {
            if (kshift == 0) emitConst64(e, rx, 0);
            else emit(e, jaiA64AndXOnes(rx, rx, kshift));
            off += 3;
            break;
        }

        emitConst64(e, JIT_SCRATCH_A, imm);
        emit(e, jaiA64SdivX(JIT_SCRATCH_B, rx, JIT_SCRATCH_A));
        emit(e, jaiA64MsubX(rx, JIT_SCRATCH_B, JIT_SCRATCH_A, rx));
        emitFloorFixup(e, rx, JIT_SCRATCH_A, true, imm,
                       jaiA64AddX(rx, rx, JIT_SCRATCH_A));
        off += 3;
        break;
    } while (0);
    *offp = off;
    return true;
}

#endif /* __aarch64__ || __arm64__ */
