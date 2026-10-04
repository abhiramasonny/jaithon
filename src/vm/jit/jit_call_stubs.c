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

static void growKeepSave(Emit *e, bool save) {
    static const unsigned gp[14] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17};
    for (unsigned i = 0; i < 14; i += 2) {
        int at = growKeepOffset(gp[i]);
        emit(e, save ? jaiA64StpOff(gp[i], gp[i + 1], 31, at)
                     : jaiA64LdpOff(gp[i], gp[i + 1], 31, at));
    }
    unsigned at = 112u;
    for (unsigned d = 0; d < 8; d += 2, at += 16u) {
        emit(e, save ? jaiA64StpDOff(d, d + 1, 31, (int32_t)at)
                     : jaiA64LdpDOff(d, d + 1, 31, (int32_t)at));
    }
    for (unsigned d = 16; d < 28; d += 2, at += 16u) {
        emit(e, save ? jaiA64StpDOff(d, d + 1, 31, (int32_t)at)
                     : jaiA64LdpDOff(d, d + 1, 31, (int32_t)at));
    }
}

/* An argument out of the register it lives in, or -- if that register is one
 * the save area holds -- out of the save area, so no argument can be read
 * after another argument's move has overwritten it. */
static void growKeepArg(Emit *e, unsigned dst, unsigned src) {
    int at = growKeepOffset(src);
    if (at >= 0) {
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
    for (unsigned gi = 0; gi < e->growCount; gi++) {
        e->grow[gi].stub = (int)e->count;
        if (e->grow[gi].keeps) {
            _Static_assert(GROW_KEEP_BYTES >= 112u + 20u * 8u,
                           "the keep area holds 14 X and 20 D registers");
            _Static_assert(GROW_KEEP_BYTES % 16u == 0u,
                           "sp stays 16-aligned across the call");
            emit(e, jaiA64SubXImm(31, 31, GROW_KEEP_BYTES));
            growKeepSave(e, true);
            growKeepArg(e, 0, e->grow[gi].listReg);
            growKeepArg(e, 2, e->grow[gi].valReg);
            emit(e, jaiA64MovzX(1, e->grow[gi].tag, 0));
            emit(e, jaiA64MovzX(3, e->grow[gi].shape ? 1u : 0u, 0));
            emitConst64(e, JIT_SCRATCH_D, (int64_t)(uintptr_t)&jitListGrow);
            emit(e, jaiA64Blr(JIT_SCRATCH_D));
            /* x10 is not in the save area, so the verdict survives the
             * restore. */
            emit(e, jaiA64MovX(JIT_SCRATCH_B, 0));
            growKeepSave(e, false);
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
    /* And the one-instruction cold fixups (emitColdFixup), each straight back
     * to the instruction after its branch. */
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
}

#endif /* __aarch64__ || __arm64__ */
