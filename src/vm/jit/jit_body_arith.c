/* jit_body_arith.c -- the arithmetic and bitwise arms of the opcode walk. */

#include "vm/jit/jit_arm64.h"
#include "vm/jit/jit_field_read.h"

#include <stddef.h>
#include <stdint.h>
#include "vm/jit/jit_internal.h"

#if (defined(__aarch64__) || defined(__arm64__))

bool emitMul(Emit *e, ObjFunction *fn, const uint8_t *code, int *offp,
             int stop) {
    int off = *offp;
    do {
        unsigned rb, ra;
        SlotKind kb, ka;
        if (e->depth >= 2 && e->stack[e->depth - 1] == SLOT_FLOAT &&
            e->stack[e->depth - 2] == SLOT_FLOAT) {
            unsigned ib = e->valueDepth - 1, ia = e->valueDepth - 2;
            unsigned db = fpOperand(e, ib), da = fpOperand(e, ia);
            if (!popValueRaw(e, &rb, &kb)) return false;
            if (!popValueRaw(e, &ra, &ka)) return false;
            if (!pushValue(e, SLOT_FLOAT, 0, NULL)) return false;
            unsigned dm = fpRegAt(e, ia);
            uint32_t bindOffM = 0;
            unsigned homeM = fpBindLookahead(e, code, off + 1, stop, fn,
                                             &bindOffM);
            if (homeM != 0) {
                fpReleaseHome(e, homeM);
                dm = homeM;
            }
            emit(e, jaiA64FmulD(dm, da, db));
            if (homeM != 0) {
                fpBorrowLocal(e, ia, homeM);
                e->homeEarly[e->homeEarlyCount++] = bindOffM;
            } else {
                fpClaim(e, ia);
            }
            off += 1;
            break;
        }
        if (!popValue(e, &rb, &kb)) return false;
        if (!popValue(e, &ra, &ka)) return false;
        if (ka != kb) return false;
        if (ka != SLOT_INT) return false;
        if (!pushValue(e, SLOT_INT, 0, NULL)) return false;
        unsigned rd = pushReg(e) - 1;
        unsigned rt = ovfDest(e, rd);
        /* The product overflows exactly when the high half is not the low
         * half's sign bit replicated, so smulh and one shifted compare
         * decide it. mul must come after smulh reads its inputs, since rd
         * may be one of them -- which is also why `rt` differs inside a
         * `try`: rd IS the first operand's register, and the guard resumes
         * at an instruction whose operands the interpreter still needs. */
        emit(e, jaiA64SmulhX(JIT_SCRATCH_A, ra, rb));
        emit(e, jaiA64MulX(rt, ra, rb));
        emit(e, jaiA64SubsXAsr(31, JIT_SCRATCH_A, rt, 63));
        branchOnOverflow(e, 2u, JAI_A64_NE);
        if (rt != rd) emit(e, jaiA64MovX(rd, rt));
        off += 1;
        break;
    } while (0);
    *offp = off;
    return true;
}

bool emitBitOp(Emit *e, ObjFunction *fn, uint8_t op, int prevOff, int *offp)
{
    int off = *offp;
    do {
        /* None of `& | ^` can fail, so they're one instruction. Shifts have two edges hardware doesn't share:
         * jaithon throws on a negative count and saturates at >=64, while arm64's LSLV/ASRV wrap on the low six bits of the count. Both edges are guarded and deopt; the guards precede the pops, so a deopt resumes at an instruction that hasn't happened yet. */
        unsigned rb, ra;
        SlotKind kb, ka;
        if (e->depth < 2) return false;
        if (e->stack[e->depth - 1] != SLOT_INT) return false;
        if (e->stack[e->depth - 2] != SLOT_INT) return false;

        /* A literal count settles both edges here rather than at run time: the immediate-form shift then IS
         * the interpreter's rule for that count. bitops shifts by 7, 3, 11, 1 and 31 and paid five instructions and two deopt sites for each before this. */
        int64_t kcount;
        bool kcountUsable =
            (op == OP_SHL || op == OP_SHR) &&
            literalIntOperand(fn, prevOff, off, &kcount) &&
            kcount >= 0 && kcount <= 63;

        if ((op == OP_SHL || op == OP_SHR) && !kcountUsable) {
            /* The count reaches a register and a guard, so it has to be
             * in one. OP_INT may have deferred it on the strength of the
             * opcode alone -- a count of 100 is a literal this arm cannot
             * fold -- and this is where that is put right. */
            settleAll(e);
        }
        if (op != OP_SHL && op != OP_SHR) settleAll(e);
        if (kcountUsable) e->kPend &= ~(1u << (e->valueDepth - 1));

        if ((op == OP_SHL || op == OP_SHR) && !kcountUsable) {
            unsigned rCount = pushReg(e) - 1;
            emit(e, jaiA64SubsXImm(31, rCount, 0));
            branchOnDeopt(e, JAI_A64_LT);            /* negative count */
            emitConst64(e, JIT_SCRATCH_A, 64);
            emit(e, jaiA64SubsXReg(31, rCount, JIT_SCRATCH_A));
            branchOnDeopt(e, JAI_A64_GE);            /* 64 or more */
        }

        if (!popValue(e, &rb, &kb)) return false;
        if (!popValue(e, &ra, &ka)) return false;
        if (!pushValue(e, SLOT_INT, 0, NULL)) return false;
        unsigned rd = pushReg(e) - 1;
        if (kcountUsable) {
            emit(e, op == OP_SHL ? jaiA64LslX(rd, ra, (unsigned)kcount)
                                 : jaiA64AsrX(rd, ra, (unsigned)kcount));
            off += 1;
            break;
        }
        emit(e, op == OP_BAND ? jaiA64AndX(rd, ra, rb)
              : op == OP_BOR  ? jaiA64OrrX(rd, ra, rb)
              : op == OP_BXOR ? jaiA64EorX(rd, ra, rb)
              : op == OP_SHL  ? jaiA64LslvX(rd, ra, rb)
                              : jaiA64AsrvX(rd, ra, rb));
        off += 1;
        break;
    } while (0);
    *offp = off;
    return true;
}

bool emitAddSubDiv(Emit *e, ObjFunction *fn, uint8_t op, const uint8_t *code,
                   int *offp, int stop) {
    int off = *offp;
    do {
        unsigned rb, ra;
        SlotKind kb, ka;
        /* `head + "/" + right`: two strings, out to jaiStringConcat, the
         * unfused half of the ADD_BIND arm above. Measured on a self-hosted
         * `check`, where OP_ADD is a few percent of the interpreted work:
         * no change (775ms -> 767ms, noise). The bodies holding it decline
         * for other reasons anyway, so this arm earns its place on the
         * ADD_BIND shape and not on the compiler. */
        if (op == OP_ADD) {
            Value csample;
            if (concatOperands(e, &csample)) {
                if (!emitStringConcat(e, csample)) return false;
                off += 1;
                break;
            }
        }
        if (e->depth >= 2 && e->stack[e->depth - 1] == SLOT_FLOAT &&
            e->stack[e->depth - 2] == SLOT_FLOAT) {
            unsigned ib = e->valueDepth - 1, ia = e->valueDepth - 2;
            unsigned db = fpOperand(e, ib), da = fpOperand(e, ia);
            if (!popValueRaw(e, &rb, &kb)) return false;
            if (!popValueRaw(e, &ra, &ka)) return false;
            if (!pushValue(e, SLOT_FLOAT, 0, NULL)) return false;
            unsigned dd = fpRegAt(e, ia);
            uint32_t bindOff = 0;
            unsigned home = fpBindLookahead(e, code, off + 1, stop, fn,
                                            &bindOff);
            if (home != 0) {
                fpReleaseHome(e, home);
                dd = home;
            }
            emit(e, op == OP_ADD ? jaiA64FaddD(dd, da, db)
                 : op == OP_SUB  ? jaiA64FsubD(dd, da, db)
                                 : jaiA64FdivD(dd, da, db));
            if (home != 0) {
                fpBorrowLocal(e, ia, home);
                e->homeEarly[e->homeEarlyCount++] = bindOff;
            } else {
                fpClaim(e, ia);
            }
            off += 1;
            break;
        }
        /* A literal right operand becomes the immediate `adds`/`subs`
         * already takes, so it never reaches a register. Decided before
         * the pops, which is where the entry index still names it. */
        int64_t kimm = 0;
        bool foldK = op != OP_DIV && e->depth >= 2 &&
                     e->stack[e->depth - 1] == SLOT_INT &&
                     e->stack[e->depth - 2] == SLOT_INT &&
                     pendingImm12(e, e->valueDepth - 1, &kimm);
        if (foldK) e->kPend &= ~(1u << (e->valueDepth - 1));

        /* Carried through the arithmetic: see Emit::idxKnown. Only the
         * folded-literal form, which is exactly what `xs[j - 1]` and
         * `xs[j + 1]` compile to, and only while the offset stays small --
         * the loop head's guard has to add it to a count without
         * overflowing, and a bound keeps that argument short. The overflow
         * branch below deoptimises, so a shape that survives to the
         * subscript describes arithmetic that actually happened. */
        bool    idxCarry     = false;
        uint8_t idxCarryBase = 0;
        int32_t idxCarryOff  = 0;
        if (foldK && e->valueDepth >= 2 &&
            (e->idxKnown & (1u << (e->valueDepth - 2))) != 0 &&
            kimm >= -4096 && kimm <= 4096) {
            unsigned at = e->valueDepth - 2;
            int64_t sum = (int64_t)e->idxOff[at] +
                          (op == OP_SUB ? -kimm : kimm);
            if (sum >= -4096 && sum <= 4096) {
                idxCarry     = true;
                idxCarryBase = e->idxBase[at];
                idxCarryOff  = (int32_t)sum;
            }
        }

        if (!popValue(e, &rb, &kb)) return false;
        if (!popValue(e, &ra, &ka)) return false;
        if (ka != kb) return false;   /* no implicit widening here */

        /* Integer division is not here: it has a zero-divisor error and a
         * truncation rule of its own, and getting either wrong would be a
         * wrong answer rather than a decline. */
        if (ka != SLOT_INT || op == OP_DIV) return false;
        if (!pushValue(e, SLOT_INT, 0, NULL)) return false;
        if (idxCarry) {
            unsigned at = e->valueDepth - 1;
            e->idxKnown |= 1u << at;
            e->idxBase[at] = idxCarryBase;
            e->idxOff[at]  = idxCarryOff;
        }
        unsigned rd = pushReg(e) - 1;
        /* rd is the first operand's own register (two off, one on), so
         * inside a `try` the sum computes elsewhere: the overflow guard
         * resumes at this instruction and the interpreter reads the
         * operands back off its stack. See ovfDest. */
        unsigned rt = ovfDest(e, rd);
        if (foldK) emitAddSubImm(e, rt, ra, kimm, op == OP_SUB);
        else emit(e, op == OP_ADD ? jaiA64AddsX(rt, ra, rb)
                                  : jaiA64SubsXReg(rt, ra, rb));
        branchOnOverflow(e, op == OP_ADD ? 0u : 1u, JAI_A64_VS);
        if (rt != rd) emit(e, jaiA64MovX(rd, rt));
        off += 1;
        break;
    } while (0);
    *offp = off;
    return true;
}

bool emitNegate(Emit *e, int *offp) {
    int off = *offp;
    do {
        /* `-x`. There was no arm at all, for either kind: OP_NEG appeared
         * only in inlinableBody's whitelist, so in an ordinary body it fell
         * to `default` and ENDED THE WALK -- everything after a negation
         * ran interpreted. `-(i & 255)` in a loop was 28,394,773
         * interpreted instructions and the float form 18,034,610.
         *
         * It hid because emitUnarmedDeopt is silent by design (the point of
         * it is to interpret from here rather than decline the body), so
         * nothing was ever printed. What surfaced it was a DIFFERENT
         * message: `is_inf`'s `x == INF or x == -INF` records a forward
         * branch to its OP_RETURN before reaching the negation, and once
         * the walk stops that branch has nowhere to land. The fixup pass
         * now names the opcode that ended the walk, which is how a
         * confusing "branch to offset 25" became "OP_NEG at 23". Every
         * guarded function in std.math goes through `_require_finite` and
         * so through `is_inf`. */
        if (!jitNegate()) {
            return subWhy(e, "the negate arm is switched off");
        }
        if (e->depth < 1) return subWhy(e, "nothing to negate");
        SlotKind nk = e->stack[e->depth - 1];
        if (nk == SLOT_INT) {
            /* `subs` off zero both negates and reports the one input that
             * cannot be: INT64_MIN overflows, and the interpreter raises
             * the OverflowError on re-entry.
             *
             * Into a SCRATCH first, and moved only once the guard has
             * passed -- the same discipline the `abs` arm states, and the
             * reason is this instruction resumes at its own START. Writing
             * the operand's register before the guard hands the deopt
             * record a value that has ALREADY been negated, and the
             * interpreter negates it again: `neg_int(5)` returned 5.
             * Only JAITHON_JIT_DEOPT_STRESS=1 finds it, because in
             * ordinary running the guard fires only on INT64_MIN, which is
             * its own negation and so hides the mistake. */
            unsigned nr = pushReg(e) - 1;
            emit(e, jaiA64SubsX(JIT_SCRATCH_A, JAI_A64_XZR, nr));
            branchOnDeoptInstStart(e, JAI_A64_VS);
            emit(e, jaiA64MovX(nr, JIT_SCRATCH_A));
            off += 1;
            break;
        }
        if (nk == SLOT_FLOAT) {
            /* Straight in the bank. IEEE negation is the sign bit and
             * cannot fail, so there is no guard and nothing to raise --
             * -0.0 and NaN both come out of `fneg` the way the interpreter
             * produces them. */
            unsigned nd = fpOperand(e, e->valueDepth - 1);
            unsigned drop;
            if (!popValueRaw(e, &drop, NULL)) return false;
            if (!pushValue(e, SLOT_FLOAT, 0, NULL)) return false;
            unsigned nidx = e->valueDepth - 1;
            emit(e, jaiA64FnegD(fpRegAt(e, nidx), nd));
            fpClaim(e, nidx);
            off += 1;
            break;
        }
        return subWhy(e, "negating a %s", slotKindName(nk));
    } while (0);
    *offp = off;
    return true;
}

bool emitPow(Emit *e, int *offp) {
    int off = *offp;
    do {
        /* Only `** 0.5` (square root): C's pow(x,0.5) is sqrt(x) for x >= +0, differing only at -0.0 (pow
         * gives +0.0, sqrt gives -0.0) -- so a negative sign bit sends this back to the interpreter. */
        if (e->depth < 2) return false;
        if (e->stack[e->depth - 1] != SLOT_FLOAT) return false;
        if (e->stack[e->depth - 2] != SLOT_FLOAT) return false;
        Value expv = e->stackSeen[e->depth - 1];
        if (!IS_FLOAT(expv) || AS_FLOAT(expv) != 0.5) {
            e->whyNot = "an exponent other than 0.5";
            return false;
        }
        /* Wholly in the FP bank: the sign bit is the only thing needing an integer register, and an `fmov`
         * out of d costs one instruction where routing the operand through X cost four (two for an exponent constant nothing reads, two more around the fsqrt). */
        unsigned ia = e->valueDepth - 2;
        unsigned da = fpOperand(e, ia);
        emit(e, jaiA64FmovXD(JIT_SCRATCH_A, da));
        emit(e, jaiA64LsrX(JIT_SCRATCH_A, JIT_SCRATCH_A, 63));
        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, 0));
        branchOnDeopt(e, JAI_A64_NE);

        unsigned dp1, dp2;
        if (!popValueRaw(e, &dp1, NULL)) return false;
        if (!popValueRaw(e, &dp2, NULL)) return false;
        if (!pushValue(e, SLOT_FLOAT, 0, NULL)) return false;
        {
            unsigned idx = e->valueDepth - 1;
            emit(e, jaiA64FsqrtD(fpRegAt(e, idx), da));
            fpClaim(e, idx);
        }
        off += 1;
        break;
    } while (0);
    *offp = off;
    return true;
}

bool emitMod(Emit *e, ObjFunction *fn, int prevOff, int *offp) {
    int off = *offp;
    do {
        /* Floor remainder, the same rule as the fused form but with the
         * divisor in a register, so zero and -1 both have to be checked. */
        if (e->depth < 2) return false;
        if (e->stack[e->depth - 1] != SLOT_INT) return false;
        if (e->stack[e->depth - 2] != SLOT_INT) return false;
        unsigned ry = pushReg(e) - 1, rx = valueXReg(e, e->valueDepth - 2);

        /* A literal power-of-two modulus decides the whole thing: floor remainder by 2^s is exactly the low
         * s bits, which two's complement already holds -- exact even for negative dividends, unlike the truncating `msub` path below (whose correction exists to fix exactly that). */
        int64_t kmod = 0;
        bool kmodKnown = literalIntOperand(fn, prevOff, off, &kmod) &&
                         kmod != 0 && kmod != -1;
        unsigned mshift;
        if (kmodKnown && powerOfTwoShift(kmod, &mshift)) {
            unsigned q1, q2;
            if (!popValue(e, &q1, NULL)) return false;
            if (!popValue(e, &q2, NULL)) return false;
            if (!pushValue(e, SLOT_INT, 0, NULL)) return false;
            unsigned rd = pushReg(e) - 1;
            if (mshift == 0) emitConst64(e, rd, 0);   /* x %% 1 is 0 */
            else emit(e, jaiA64AndXOnes(rd, rx, mshift));
            off += 1;
            break;
        }

        if (!kmodKnown) {
            emit(e, jaiA64SubsXImm(31, ry, 0));
            branchOnDeopt(e, JAI_A64_EQ);
            emit(e, jaiA64AddXImm(JIT_SCRATCH_A, ry, 1));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, 0));
            branchOnDeopt(e, JAI_A64_EQ);
        }

        /* Remainder lands in `rx`, where the result belongs: two entries come off, one goes on, so the
         * surviving (lower) entry keeps its register -- removes a trailing `mov`. Safe because every guard this arm emits is above this line: nothing below can deopt and find the dividend gone. */
        emit(e, jaiA64SdivX(JIT_SCRATCH_B, rx, ry));
        emit(e, jaiA64MsubX(rx, JIT_SCRATCH_B, ry, rx));
        emitFloorFixup(e, rx, ry, kmodKnown, kmod,
                       jaiA64AddX(rx, rx, ry));

        unsigned dm1, dm2;
        if (!popValue(e, &dm1, NULL)) return false;
        if (!popValue(e, &dm2, NULL)) return false;
        if (!pushValue(e, SLOT_INT, 0, NULL)) return false;
        off += 1;
        break;
    } while (0);
    *offp = off;
    return true;
}

bool emitFloorDiv(Emit *e, ObjFunction *fn, int prevOff, int *offp) {
    int off = *offp;
    do {
        unsigned rb, ra;
        SlotKind kb, ka;
        if (e->depth < 2) return false;
        ka = e->stack[e->depth - 2]; kb = e->stack[e->depth - 1];
        if (ka != SLOT_INT || kb != SLOT_INT) return false;

        /* A literal power-of-two divisor decides the whole thing: floor(x / 2^s) is exactly `asr x, #s` for
         * every int64 x (negative included), since asr already rounds toward minus infinity -- what the correction below exists to reproduce for the general case. */
        int64_t kdiv = 0;
        bool kdivKnown = literalIntOperand(fn, prevOff, off, &kdiv) &&
                         kdiv != 0 && kdiv != -1;
        unsigned dshift;
        if (kdivKnown && powerOfTwoShift(kdiv, &dshift)) {
            /* The divisor is spelt by the shift field, so drop its deferral rather than settle it -- the
             * same move OP_ADD makes for an imm12 -- popValueRaw drops it. The dividend comes back from popValue, not from pushReg, because a borrowed entry lives in the local's register and not in its own. */
            unsigned p1, p2;
            if (!popValueRaw(e, &p1, NULL)) return false;
            if (!popValue(e, &p2, NULL)) return false;
            if (!pushValue(e, SLOT_INT, 0, NULL)) return false;
            emit(e, jaiA64AsrX(pushReg(e) - 1, p2, dshift));
            off += 1;
            break;
        }
        /* Every path below reads both operands out of their own registers,
         * and two of them guard. */
        settleAll(e);
        rb = pushReg(e) - 1; ra = valueXReg(e, e->valueDepth - 2);

        /* Zero and -1 both decline: the interpreter reports division-by-zero, and INT64_MIN / -1 is the one
         * quotient that doesn't fit -- both rare enough that declining -1 outright costs nothing. A literal divisor has already answered both. */
        if (!kdivKnown) {
            emit(e, jaiA64SubsXImm(31, rb, 0));
            branchOnDeopt(e, JAI_A64_EQ);
            emit(e, jaiA64AddXImm(JIT_SCRATCH_A, rb, 1));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, 0));
            branchOnDeopt(e, JAI_A64_EQ);
        }

        unsigned d1, d2;
        if (!popValue(e, &d1, NULL)) return false;
        if (!popValue(e, &d2, NULL)) return false;
        if (!pushValue(e, SLOT_INT, 0, NULL)) return false;
        unsigned rq = pushReg(e) - 1;

        /* Quotient lands in the register the dividend was in (the push reuses it), so the dividend is copied
         * out first -- without that, msub read a value sdiv had already overwritten and 7 // 2 came out 2. */
        emit(e, jaiA64MovX(JIT_SCRATCH_C, ra));

        /* Truncating quotient, then one down when the remainder is nonzero
         * and its sign differs from the divisor's -- which is what makes
         * this floor division rather than C's. */
        emit(e, jaiA64SdivX(rq, JIT_SCRATCH_C, rb));
        emit(e, jaiA64MsubX(JIT_SCRATCH_A, rq, rb, JIT_SCRATCH_C));
        /* A literal divisor makes the correction a single sign test on the remainder (see emitFloorFixup).
         * JIT_SCRATCH_B is free again here -- the quotient is in rq, not in it. */
        emitFloorFixup(e, JIT_SCRATCH_A, rb, kdivKnown, kdiv,
                       jaiA64SubXImm(rq, rq, 1));
        off += 1;
        break;
    } while (0);
    *offp = off;
    return true;
}

JitArmResult emitNot(Emit *e, int *offp) {
    int off = *offp;
    do {
        /* `not x`. The interpreter REQUIREs a bool here, and SLOT_BOOL's
         * contract is "0 or 1 in a register", so the flip is an xor with
         * one and there is nothing to guard. Anything else is not a
         * narrowing this tier declines to do -- it is a program the
         * interpreter would throw on, and it reaches the throw by the
         * unarmed path.
         *
         * No arm existed, so `not` ENDED THE WALK the way OP_NEG did:
         * `if not a` in a loop was 23,252,579 interpreted instructions. */
        if (!jitNegate() || e->depth < 1 ||
            e->stack[e->depth - 1] != SLOT_BOOL) {
            goto unarmedOpcode;
        }
        {
            unsigned nr = pushReg(e) - 1;
            emitConst64(e, JIT_SCRATCH_A, 1);
            emit(e, jaiA64EorX(nr, nr, JIT_SCRATCH_A));
        }
        off += 1;
        break;
    } while (0);
    *offp = off;
    return JIT_ARM_OK;
unarmedOpcode:
    *offp = off;
    return JIT_ARM_UNARMED;
}

JitArmResult emitBitNot(Emit *e, int *offp) {
    int off = *offp;
    do {
        /* `~x` is `x ^ -1`, and the model has already proved the int. */
        if (!jitNegate() || e->depth < 1 ||
            e->stack[e->depth - 1] != SLOT_INT) {
            goto unarmedOpcode;
        }
        {
            unsigned nr = pushReg(e) - 1;
            emitConst64(e, JIT_SCRATCH_A, -1);
            emit(e, jaiA64EorX(nr, nr, JIT_SCRATCH_A));
        }
        off += 1;
        break;
    } while (0);
    *offp = off;
    return JIT_ARM_OK;
unarmedOpcode:
    *offp = off;
    return JIT_ARM_UNARMED;
}

JitArmResult emitUnaryPlus(Emit *e, int *offp) {
    int off = *offp;
    do {
        /* Unary `+` on a number is the identity -- the interpreter checks
         * the type and does nothing else. The model has already proved it,
         * so this emits nothing at all; the point is only that the walk
         * does not stop here. */
        if (!jitNegate() || e->depth < 1 ||
            (e->stack[e->depth - 1] != SLOT_INT &&
             e->stack[e->depth - 1] != SLOT_FLOAT)) {
            goto unarmedOpcode;
        }
        off += 1;
        break;
    } while (0);
    *offp = off;
    return JIT_ARM_OK;
unarmedOpcode:
    *offp = off;
    return JIT_ARM_UNARMED;
}

/* `add`/`sub`/`mul`, and nothing else. These are the checked arms with the
 * overflow guard deleted rather than relaxed: the interpreter defines them as
 * `(int64_t)((uint64_t)x OP (uint64_t)y)`, which is the low 64 bits every one
 * of these instructions already writes. OP_MUL's `smulh` goes with the guard it
 * fed -- it only ever supplied the high half to compare against -- so nothing
 * is left that could observe the difference, and there is no strength-reduced
 * or immediate form of the checked arms to mirror: `*` has none, and the
 * peephole fuses only the checked operators (opt/fuse.jai), so no
 * OP_*_WRAP_INT_CONST exists.
 *
 * Both operands must be SLOT_INT. `+% -% *%` wrap a 64-bit integer and nothing
 * else: the interpreter raises TypeError for a float on either side and looks
 * for __add__/__sub__/__mul__ on anything that is not a number, so every other
 * shape belongs to it. Answered from the model BEFORE the pops, so the refusal
 * hands emitUnarmedDeopt the state as of this instruction's start.
 *
 * No ovfDest, because there is no guard to resume at: the result may go
 * straight to its home even inside a `try`. */
JitArmResult emitWrapArith(Emit *e, uint8_t op, int *offp) {
    int off = *offp;
    do {
        if (!jitWrapArith() || e->depth < 2 ||
            e->stack[e->depth - 1] != SLOT_INT ||
            e->stack[e->depth - 2] != SLOT_INT) {
            goto unarmedOpcode;
        }
        {
            unsigned rbw, raw;
            SlotKind kbw, kaw;
            if (!popValue(e, &rbw, &kbw)) { *offp = off; return JIT_ARM_REFUSED; }
            if (!popValue(e, &raw, &kaw)) { *offp = off; return JIT_ARM_REFUSED; }
            if (!pushValue(e, SLOT_INT, 0, NULL)) { *offp = off; return JIT_ARM_REFUSED; }
            unsigned rdw = pushReg(e) - 1;
            emit(e, op == OP_ADD_WRAP ? jaiA64AddX(rdw, raw, rbw)
                 : op == OP_SUB_WRAP  ? jaiA64SubX(rdw, raw, rbw)
                                      : jaiA64MulX(rdw, raw, rbw));
        }
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
