/* jit_call_stubs.c -- the out-of-line stubs a call leaves behind: the
 * deoptimised-callee continuation and the list-grow slow half. */
#include "vm/jit/jit.h"

#include "vm/jit/jit_arm64.h"
#include "vm/jit/jit_field_read.h"

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "vm/jit/jit_internal.h"

#if (defined(__aarch64__) || defined(__arm64__))

/* Verdict 4: the callee deoptimised part-way and may have written, so the call can't be re-executed
 * or recorded over (gDeopt is a single global) -- instead the callee is FINISHED in the interpreter from its own record, and the value handed back here, consuming the record at the innermost frame that sees it. Makes a recursive body that writes compilable, and (since `callee` may name someone else) a direct call to a writing method too. OSR form: fall-through continues the loop, so nothing syncs here -- every branch out goes to a stub that syncs itself. */
void emitSelfSlowStubs(Emit *e, ObjClosure *closure) {
    for (unsigned si = 0; si < e->selfSlowCount; si++) {
        e->selfSlow[si].stub = (int)e->count;
        unsigned d = e->descOffset;
        unsigned resultAt = d + (unsigned)offsetof(JitCallDesc, result);

        emit(e, jaiA64SubsXImm(31, 1, 2));             /* the callee raised */
        emit(e, jaiA64BCond(JAI_A64_EQ,
                            (int32_t)(e->exceptionExit - (int)e->count)));

        emit(e, jaiA64SubsXImm(31, 1, 4));
        if (e->fixupCount >= JIT_MAX_FIXUPS) { e->failed = true; break; }
        e->fixups[e->fixupCount].instIndex    = (int)e->count;
        e->fixups[e->fixupCount].targetOffset =
            FIXUP_DEOPT - e->selfSlow[si].deoptBail;
        e->fixups[e->fixupCount].conditional  = true;
        e->fixups[e->fixupCount].depth        = -1;
        e->fixupCount++;
        emit(e, jaiA64BCond(JAI_A64_NE, 0));

        /* jaiJitFinishDeopt runs interpreted code, which allocates, so the
         * roots this frame filled before the `bl` go back on the chain. */
        if (e->selfSlow[si].roots > 0) {
            emitConst64(e, JIT_SCRATCH_A, (int64_t)(uintptr_t)&gJitFrames);
            emit(e, jaiA64LdrX(JIT_SCRATCH_B, JIT_SCRATCH_A, 0));
            emit(e, jaiA64AddXImm(JIT_SCRATCH_C, 31, d));
            emit(e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_C,
                               (unsigned)offsetof(JitCallDesc, link)));
            emit(e, jaiA64StrX(JIT_SCRATCH_C, JIT_SCRATCH_A, 0));
        }
        emitConst64(e, 0, (int64_t)(uintptr_t)(e->selfSlow[si].callee != NULL
                                                   ? e->selfSlow[si].callee
                                                   : closure));
        emit(e, jaiA64AddXImm(1, 31, resultAt));
        emitConst64(e, JIT_SCRATCH_D, (int64_t)(uintptr_t)&jaiJitFinishDeopt);
        emit(e, jaiA64Blr(JIT_SCRATCH_D));
        if (e->selfSlow[si].roots > 0) {
            emitConst64(e, JIT_SCRATCH_A, (int64_t)(uintptr_t)&gJitFrames);
            emit(e, jaiA64AddXImm(JIT_SCRATCH_C, 31, d));
            emit(e, jaiA64LdrX(JIT_SCRATCH_B, JIT_SCRATCH_C,
                               (unsigned)offsetof(JitCallDesc, link)));
            emit(e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_A, 0));
        }
        emit(e, jaiA64SubsXImm(31, 0, 0));             /* false: it raised */
        emit(e, jaiA64BCond(JAI_A64_EQ,
                            (int32_t)(e->exceptionExit - (int)e->count)));

        /* The interpreted continuation may return a kind this body was not
         * compiled for. It is in the descriptor with whatever tag it really
         * has, which is exactly what a `lastFromDesc` record writes out. */
        emit(e, jaiA64LdrW(JIT_SCRATCH_A, 31, resultAt));
        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, e->selfSlow[si].tag));
        if (e->fixupCount >= JIT_MAX_FIXUPS) { e->failed = true; break; }
        e->fixups[e->fixupCount].instIndex    = (int)e->count;
        e->fixups[e->fixupCount].targetOffset =
            FIXUP_DEOPT - e->selfSlow[si].deoptKind;
        e->fixups[e->fixupCount].conditional  = true;
        e->fixups[e->fixupCount].depth        = -1;
        e->fixupCount++;
        emit(e, jaiA64BCond(JAI_A64_NE, 0));
        /* One byte for a bool, as every other descriptor-result site loads
         * one. BOOL_VAL writes only the union's one-byte member, so a slot the
         * interpreter last used for a large int or a pointer keeps its upper
         * seven bytes -- verified on this toolchain at -O2 -flto: a slot
         * holding INT_VAL(0x1122334455667788) then assigned BOOL_VAL(false)
         * reads back 0x1122334455667700. Every SLOT_BOOL consumer branches on
         * the whole word, so that `false` reads as `true`.
         *
         * The pinned arm already reached this stub, so the widening predates
         * the polymorphic one -- but the PIC routes a whole new class of site
         * through here, and those sites previously always took the one-byte
         * load. */
        if (e->selfSlow[si].tag == VAL_BOOL) {
            emit(e, jaiA64LdrByte(e->selfSlow[si].resultReg, 31, resultAt + 8));
        } else {
            emit(e, jaiA64LdrX(e->selfSlow[si].resultReg, 31, resultAt + 8));
        }

        /* VAL_OBJ is every heap object -- reading `klass` off the wrong type (e.g. an ObjString) doesn't
         * fault, it answers wrongly, so the type is checked before the shape. */
        if (e->selfSlow[si].retType >= 0) {
            unsigned rr = e->selfSlow[si].resultReg;
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, rr,
                               (unsigned)offsetof(Obj, type)));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A,
                                   (unsigned)e->selfSlow[si].retType));
            if (e->fixupCount >= JIT_MAX_FIXUPS) { e->failed = true; break; }
            e->fixups[e->fixupCount].instIndex    = (int)e->count;
            e->fixups[e->fixupCount].targetOffset =
                FIXUP_DEOPT - e->selfSlow[si].deoptKind;
            e->fixups[e->fixupCount].conditional  = true;
            e->fixups[e->fixupCount].depth        = -1;
            e->fixupCount++;
            emit(e, jaiA64BCond(JAI_A64_NE, 0));
            if (e->selfSlow[si].retShape != 0) {
                emit(e, jaiA64LdrX(JIT_SCRATCH_A, rr,
                                   (unsigned)offsetof(ObjInstance, klass)));
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_A,
                                   (unsigned)offsetof(ObjClass, shapeId)));
                emitConst64(e, JIT_SCRATCH_B,
                            (int64_t)e->selfSlow[si].retShape);
                emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
                if (e->fixupCount >= JIT_MAX_FIXUPS) { e->failed = true; break; }
                e->fixups[e->fixupCount].instIndex    = (int)e->count;
                e->fixups[e->fixupCount].targetOffset =
                    FIXUP_DEOPT - e->selfSlow[si].deoptKind;
                e->fixups[e->fixupCount].conditional  = true;
                e->fixups[e->fixupCount].depth        = -1;
                e->fixupCount++;
                emit(e, jaiA64BCond(JAI_A64_NE, 0));
            }
        }
        emit(e, jaiA64B((int32_t)(e->selfSlow[si].returnTo - (int)e->count)));
    }
}

/* Cold half of `xs.push(v)`: reserve, refill the count the fast path already loaded, branch back in.
 * No descriptor/roots (see jitListGrow). This is a continuation, not an exit -- an OSR form must NOT sync its iterator or locals here, since the loop carries on with them where they are. */
/* Where a keeping stub parks each caller-saved register. x0..x8 are the
 * operand stack's scratch bank and an inlined body's; x13..x17 hold hoisted
 * headers. x9..x12 are the emitter's own scratches: the fast path has nothing
 * live in them at the branch but the count, which the stub reloads anyway.
 * d0..d7 and d16..d27 are every FP register the tier names outside the
 * callee-saved locals (JIT_FP_BANK, JIT_INL_FP_BANK, the local-add temp just
 * past the bank, and the arithmetic scratches). x30 is restored by the
 * epilogue on every path, as for every call this tier makes. */
#define GROW_KEEP_BYTES 288u
static int growKeepOffset(unsigned reg) {
    if (reg <= 8u) return (int)(reg * 8u);
    if (reg >= 13u && reg <= 17u) return (int)((reg - 4u) * 8u);
    return -1;
}

/* Which of the registers above the stub saves (bit n is xn or dn). x0..x8
 * always: they are the operand stack's and an inlined body's, and a C call's
 * result can sit in x0 unnamed by anything after it. The rest only when the
 * body could hold a value in them (see growKeepMasks). */
static void growKeepSave(Emit *e, bool save, uint32_t gpMask, uint32_t fpMask) {
    static const unsigned gp[14] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17};
    for (unsigned i = 0; i < 14; i += 2) {
        unsigned a = gp[i], b = gp[i + 1];
        bool ha = (gpMask >> a) & 1u, hb = (gpMask >> b) & 1u;
        unsigned at = (unsigned)growKeepOffset(a);
        unsigned bt = (unsigned)growKeepOffset(b);
        if (ha && hb) {
            emit(e, save ? jaiA64StpOff(a, b, 31, (int)at)
                         : jaiA64LdpOff(a, b, 31, (int)at));
        } else if (ha) {
            emit(e, save ? jaiA64StrX(a, 31, at) : jaiA64LdrX(a, 31, at));
        } else if (hb) {
            emit(e, save ? jaiA64StrX(b, 31, bt) : jaiA64LdrX(b, 31, bt));
        }
    }
    static const unsigned fp[20] = {0, 1, 2, 3, 4, 5, 6, 7, 16, 17,
                                    18, 19, 20, 21, 22, 23, 24, 25, 26, 27};
    unsigned at = 112u;
    for (unsigned i = 0; i < 20; i += 2, at += 16u) {
        unsigned a = fp[i], b = fp[i + 1];
        bool ha = (fpMask >> a) & 1u, hb = (fpMask >> b) & 1u;
        if (ha && hb) {
            emit(e, save ? jaiA64StpDOff(a, b, 31, (int32_t)at)
                         : jaiA64LdpDOff(a, b, 31, (int32_t)at));
        } else if (ha) {
            emit(e, save ? jaiA64StrD(a, 31, at) : jaiA64LdrD(a, 31, at));
        } else if (hb) {
            emit(e, save ? jaiA64StrD(b, 31, at + 8u)
                         : jaiA64LdrD(b, 31, at + 8u));
        }
    }
}

/* JAITHON_JIT_GROW_KEEP_USED=0 has a keeping grow stub save all fourteen X
 * and twenty D registers in every body. On by default.
 *
 * Saving all of them is most of what the stub costs, and it is paid on every
 * grow -- once a list, for a list that is fresh per call or per iteration (a
 * record built by push), where it was 4-10% of the loop. A register can only
 * hold one of the body's values if an instruction of the body named it, so
 * the code emitted so far (everything a grow stub returns into: the prologue,
 * every inlined callee, the self-call stubs) is scanned once and a register
 * no instruction could have put a value in is left alone:
 *   - a D register is kept if ANY field of any SIMD-and-FP instruction (bits
 *     27 and 26 both set) names it, reads included -- no helper returns a
 *     double, so nothing else can leave one there;
 *   - x13..x17 are kept if a destination could name them: bits 4:0 of any
 *     instruction, and for a load or store (bit 27 set, bit 25 clear) also
 *     the base a writeback updates (9:5) and a pair's second register (14:10).
 *     The tier writes registers nowhere else (Bl/Blr write x30 only).
 * A field that only looks like a register -- an immediate, a condition --
 * costs a save, never a value. */
static bool jitGrowKeepUsedOn(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("JAITHON_JIT_GROW_KEEP_USED");
        on = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return on != 0;
}

static void growKeepMasks(const Emit *e, uint32_t *gpMask, uint32_t *fpMask) {
    if (!jitGrowKeepUsedOn()) {
        *gpMask = 0xffffffffu;
        *fpMask = 0xffffffffu;
        return;
    }
    uint32_t gpm = 0x1ffu, fpm = 0;
    for (unsigned i = 0; i < e->count; i++) {
        uint32_t w = e->code[i];
        uint32_t f0 = 1u << (w & 31u);
        uint32_t f5 = 1u << ((w >> 5) & 31u);
        uint32_t f10 = 1u << ((w >> 10) & 31u);
        uint32_t f16 = 1u << ((w >> 16) & 31u);
        bool ldst = (w & 0x0A000000u) == 0x08000000u;
        if ((w & 0x0C000000u) == 0x0C000000u) {
            fpm |= f0 | f5 | f10 | f16;
            gpm |= f0;
            if (ldst) gpm |= f5;
        } else {
            gpm |= f0;
            if (ldst) gpm |= f5 | f10;
        }
    }
    *gpMask = gpm;
    *fpMask = fpm;
}

/* An argument out of the register it lives in, or -- if that register is one
 * the save area holds -- out of the save area, so no argument can be read
 * after another argument's move has overwritten it. */
static void growKeepArg(Emit *e, unsigned dst, unsigned src,
                        uint32_t gpMask) {
    int at = growKeepOffset(src);
    if (at >= 0 && ((gpMask >> src) & 1u)) {
        emit(e, jaiA64LdrX(dst, 31, (unsigned)at));
    } else {
        emit(e, jaiA64MovX(dst, src));
    }
}

/* JAITHON_JIT_SHAPE_SIBLINGS=0 lets every dispatching append shape. On by
 * default.
 *
 * An append that may unbox the list it grows (Emit::grow[].shape) does not,
 * when the same body also appends an object to the same local: `var r = []`,
 * ten ints, then `r.push("end")`. Shaped, the list reaches that last append
 * unboxed, its boxed-storage guard fails, and the body deoptimises -- every
 * call, 0.54x on a hot builder of such lists. Unshaped, nothing in this body
 * changes; the append after the grow dispatches on whatever storage the list
 * has, so declining to shape is never wrong, only a list left boxed. */
static bool jitShapeSiblingsOn(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("JAITHON_JIT_SHAPE_SIBLINGS");
        on = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return on != 0;
}

void emitGrowStubs(Emit *e) {
    if (jitShapeSiblingsOn()) {
        for (unsigned gi = 0; gi < e->growCount; gi++) {
            if (!e->grow[gi].shape || e->grow[gi].target < 0) continue;
            for (unsigned gj = 0; gj < e->growCount; gj++) {
                if (e->grow[gj].target == e->grow[gi].target &&
                    e->grow[gj].tag == VAL_OBJ) {
                    e->grow[gi].shape = false;
                    break;
                }
            }
        }
    }
    /* The one-instruction cold fixups (emitColdFixup), each straight back to
     * the instruction after its branch; then the grow stubs. */
    for (unsigned ci = 0; ci < e->coldCount; ci++) {
        e->cold[ci].stub = (int)e->count;
        if (e->cold[ci].kind == 1) {
            unsigned ro = e->cold[ci].rOut, rc = e->cold[ci].rCount;
            bool w = e->cold[ci].countW;
            emit(e, w ? jaiA64AddXUxtw(ro, ro, rc) : jaiA64AddX(ro, ro, rc));
            emit(e, w ? jaiA64SubsXUxtw(31, ro, rc)
                      : jaiA64SubsXReg(31, ro, rc));
            if (e->fixupCount >= JIT_MAX_FIXUPS) { e->failed = true; return; }
            bool always = jitDeoptStressOn();
            e->fixups[e->fixupCount].instIndex    = (int)e->count;
            e->fixups[e->fixupCount].targetOffset =
                FIXUP_DEOPT - (unsigned)e->cold[ci].deoptK;
            e->fixups[e->fixupCount].conditional  = !always;
            e->fixups[e->fixupCount].depth        = -1;
            e->fixupCount++;
            emit(e, always ? jaiA64B(0) : jaiA64BCond(JAI_A64_HS, 0));
        } else if (e->cold[ci].kind == 2) {
            /* The boxed arm of a dispatched element read: the storage must be
             * BOXED (anything else is the third storage, which deopts), the
             * element must carry the tag, and the address left behind is the
             * payload's, as the inline arm left it. One record serves both
             * guards: nothing between the dispatch and the tag check moves
             * the model. */
            bool always = jitDeoptStressOn();
            unsigned dk = (unsigned)e->cold[ci].deoptK;
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_D, LIST_STORE_BOXED));
            if (e->fixupCount + 2u > JIT_MAX_FIXUPS) { e->failed = true; return; }
            e->fixups[e->fixupCount].instIndex    = (int)e->count;
            e->fixups[e->fixupCount].targetOffset = FIXUP_DEOPT - dk;
            e->fixups[e->fixupCount].conditional  = !always;
            e->fixups[e->fixupCount].depth        = -1;
            e->fixupCount++;
            emit(e, always ? jaiA64B(0) : jaiA64BCond(JAI_A64_NE, 0));
            emit(e, jaiA64AddXLsl(JIT_SCRATCH_C, e->cold[ci].rItems,
                                  JIT_SCRATCH_B, 4));
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_C, 0));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, e->cold[ci].tag));
            e->fixups[e->fixupCount].instIndex    = (int)e->count;
            e->fixups[e->fixupCount].targetOffset = FIXUP_DEOPT - dk;
            e->fixups[e->fixupCount].conditional  = !always;
            e->fixups[e->fixupCount].depth        = -1;
            e->fixupCount++;
            emit(e, always ? jaiA64B(0) : jaiA64BCond(JAI_A64_NE, 0));
            emit(e, jaiA64AddXImm(JIT_SCRATCH_C, JIT_SCRATCH_C, 8));
        } else if (e->cold[ci].kind == 3) {
            /* The boxed arm of a dispatched append (emitListStore): storage
             * BOXED or deopt, then the tag-and-payload store at the index in
             * JIT_SCRATCH_A off the items in JIT_SCRATCH_C, as the inline arm
             * made it. */
            bool always = jitDeoptStressOn();
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_D, LIST_STORE_BOXED));
            if (e->fixupCount >= JIT_MAX_FIXUPS) { e->failed = true; return; }
            e->fixups[e->fixupCount].instIndex    = (int)e->count;
            e->fixups[e->fixupCount].targetOffset =
                FIXUP_DEOPT - (unsigned)e->cold[ci].deoptK;
            e->fixups[e->fixupCount].conditional  = !always;
            e->fixups[e->fixupCount].depth        = -1;
            e->fixupCount++;
            emit(e, always ? jaiA64B(0) : jaiA64BCond(JAI_A64_NE, 0));
            emitListElemStore(e, LIST_STORE_BOXED, e->cold[ci].tag,
                              e->cold[ci].rOut);
        } else if (e->cold[ci].kind == 4) {
            /* The boxed arm of a nested `for x in xs` step: storage BOXED or
             * deopt, the element's tag or deopt, and JIT_SCRATCH_C left on
             * its payload; JIT_SCRATCH_A, the index, is untouched for the
             * advance after the join. */
            bool always = jitDeoptStressOn();
            unsigned dk = (unsigned)e->cold[ci].deoptK;
            if (e->fixupCount + 2u > JIT_MAX_FIXUPS) { e->failed = true; return; }
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_D, LIST_STORE_BOXED));
            e->fixups[e->fixupCount].instIndex    = (int)e->count;
            e->fixups[e->fixupCount].targetOffset = FIXUP_DEOPT - dk;
            e->fixups[e->fixupCount].conditional  = !always;
            e->fixups[e->fixupCount].depth        = -1;
            e->fixupCount++;
            emit(e, always ? jaiA64B(0) : jaiA64BCond(JAI_A64_NE, 0));
            emit(e, jaiA64LdrX(JIT_SCRATCH_C, JIT_SCRATCH_C,
                               (unsigned)offsetof(ObjList, items)));
            emit(e, jaiA64AddXLsl(JIT_SCRATCH_C, JIT_SCRATCH_C,
                                  JIT_SCRATCH_A, 4));
            emit(e, jaiA64LdrW(JIT_SCRATCH_B, JIT_SCRATCH_C, 0));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_B, e->cold[ci].tag));
            e->fixups[e->fixupCount].instIndex    = (int)e->count;
            e->fixups[e->fixupCount].targetOffset = FIXUP_DEOPT - dk;
            e->fixups[e->fixupCount].conditional  = !always;
            e->fixups[e->fixupCount].depth        = -1;
            e->fixupCount++;
            emit(e, always ? jaiA64B(0) : jaiA64BCond(JAI_A64_NE, 0));
            emit(e, jaiA64AddXImm(JIT_SCRATCH_C, JIT_SCRATCH_C, 8));
        } else {
            emit(e, e->cold[ci].insn);
        }
        emit(e, jaiA64B((int32_t)(e->cold[ci].returnTo - (int)e->count)));
    }
    /* After the cold fixups, so that what growKeepMasks scans is every
     * instruction a grow stub can return into. */
    uint32_t gpMask = 0, fpMask = 0;
    if (e->growCount > 0) growKeepMasks(e, &gpMask, &fpMask);
    for (unsigned gi = 0; gi < e->growCount; gi++) {
        e->grow[gi].stub = (int)e->count;
        if (e->grow[gi].keeps) {
            _Static_assert(GROW_KEEP_BYTES >= 112u + 20u * 8u,
                           "the keep area holds 14 X and 20 D registers");
            _Static_assert(GROW_KEEP_BYTES % 16u == 0u,
                           "sp stays 16-aligned across the call");
            emit(e, jaiA64SubXImm(31, 31, GROW_KEEP_BYTES));
            growKeepSave(e, true, gpMask, fpMask);
            growKeepArg(e, 0, e->grow[gi].listReg, gpMask);
            growKeepArg(e, 2, e->grow[gi].valReg, gpMask);
            emit(e, jaiA64MovzX(1, e->grow[gi].tag, 0));
            emit(e, jaiA64MovzX(3, e->grow[gi].shape ? 1u : 0u, 0));
            emitConst64(e, JIT_SCRATCH_D, (int64_t)(uintptr_t)&jitListGrow);
            emit(e, jaiA64Blr(JIT_SCRATCH_D));
            /* x10 is not in the save area, so the verdict survives the
             * restore. */
            emit(e, jaiA64MovX(JIT_SCRATCH_B, 0));
            growKeepSave(e, false, gpMask, fpMask);
            emit(e, jaiA64AddXImm(31, 31, GROW_KEEP_BYTES));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_B, 0));
            emit(e, jaiA64BCond(JAI_A64_NE,
                                (int32_t)(e->exceptionExit - (int)e->count)));
            emit(e, jaiA64LdrW(e->grow[gi].countReg, e->grow[gi].listReg,
                               (unsigned)offsetof(ObjList, count)));
            emit(e, jaiA64B((int32_t)(e->grow[gi].returnTo - (int)e->count)));
            continue;
        }
        emit(e, jaiA64MovX(0, e->grow[gi].listReg));
        emit(e, jaiA64MovzX(1, e->grow[gi].tag, 0));
        emit(e, jaiA64MovX(2, e->grow[gi].valReg));
        emit(e, jaiA64MovzX(3, e->grow[gi].shape ? 1u : 0u, 0));
        emitConst64(e, JIT_SCRATCH_D, (int64_t)(uintptr_t)&jitListGrow);
        emit(e, jaiA64Blr(JIT_SCRATCH_D));
        emit(e, jaiA64SubsXImm(31, 0, 0));
        emit(e, jaiA64BCond(JAI_A64_NE,
                            (int32_t)(e->exceptionExit - (int)e->count)));
        emit(e, jaiA64LdrW(e->grow[gi].countReg, e->grow[gi].listReg,
                           (unsigned)offsetof(ObjList, count)));
        emit(e, jaiA64B((int32_t)(e->grow[gi].returnTo - (int)e->count)));
    }
}

#endif /* __aarch64__ || __arm64__ */
