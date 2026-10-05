/* jit_call_pic.c -- the inline cache an OP_INVOKE gets when the model could not
 * pin one receiver class: a guarded direct call per way the site has seen. */

#include "vm/jit/jit_arm64.h"
#include "vm/jit/jit_field_read.h"
#include "vm/vm.h"

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "vm/jit/jit_internal.h"

#if (defined(__aarch64__) || defined(__arm64__))

/* JAITHON_JIT_PIC=0 turns the one-way inline cache below off, so the same
 * binary can be A/B'd around it without a rebuild -- same idiom as
 * jaiListUnboxOn's JAITHON_LIST_UNBOX (object_collection.c) and
 * jaiJitEnabled's JAITHON_NO_JIT. Read once: this sits on every unpinned
 * OP_INVOKE, and an uncached getenv there is its own cost (see
 * jitReconTrace above). */
/* JAITHON_JIT_PIC_INLINE_ONLY=0 takes a polymorphic way only when its callee
 * can be called directly, as before ways could be inlined. */
static bool jitPicInlineOnlyOn(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("JAITHON_JIT_PIC_INLINE_ONLY");
        on = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return on != 0;
}

bool jitPicEnabled(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("JAITHON_JIT_PIC");
        on = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return on != 0;
}

/* Mirrors every decision emitDirectCall makes before it commits to emitting,
 * for the one way this cache is about to speculate on. By the time the shape
 * compare below is in the instruction stream a decline has nowhere to fall
 * back to but e->failed, so this runs FIRST and answers instead of finding
 * out the hard way. */
static bool jitPic1Admissible(Emit *e, ObjFunction *caller, ObjFunction *cfn,
                              unsigned ridx, unsigned argc, uint32_t shape) {
    if (cfn->module != caller->module) return false;
    if (caller->module == NULL ||
        cfn->jitFuncModuleVersion != caller->module->version) {
        return false;
    }
    if (cfn->jitArgBase != 0u) return false;
    unsigned nargs = argc + 1u;
    unsigned calleeArgs = (unsigned)cfn->jitArgCount;
    bool wantsClosure = calleeArgs == nargs + 1u;
    if (!wantsClosure && calleeArgs != nargs) return false;
    if (calleeArgs > JIT_MAX_ARITY) return false;
    if (wantsClosure &&
        (SlotKind)cfn->jitParamKind[nargs] != SLOT_CLOSURE) {
        return false;
    }
    /* The receiver is the callee's slot 0, and the branch about to be
     * emitted is the proof of its class -- so the parameter the callee was
     * specialised for has to be that same class, not merely some instance. */
    if ((SlotKind)cfn->jitParamKind[0] != SLOT_INST) return false;
    if (cfn->jitParamShape[0] != shape) return false;
    for (unsigned i = 1; i < nargs; i++) {
        unsigned idx = ridx + i;
        SlotKind have = e->stack[idx];
        SlotKind want = (SlotKind)cfn->jitParamKind[i];
        if (!holdsRegister(have)) return false;
        if (want == SLOT_OPAQUE) continue;   /* never read; see seedLocals */
        if (want == SLOT_MAYBE_INST) {
            if (have != SLOT_INST && have != SLOT_MAYBE_INST) return false;
        } else if (have != want) {
            return false;
        }
        if ((want == SLOT_INST || want == SLOT_MAYBE_INST) &&
            cfn->jitParamShape[i] != 0 &&
            e->stackShape[idx] != cfn->jitParamShape[i]) {
            return false;
        }
    }
    /* emitDirectCall refuses a callee whose walk never reached a return, and
     * this has to refuse it too: by the time that arm says no, the shape
     * compare for this way is already in the instruction stream, so the
     * refusal sets e->failed and the WHOLE LOOP declines rather than this one
     * way being dropped. Asking here costs the way and keeps the loop. */
    if (!cfn->jitReturnKnown && jitReturnKnownOn()) return false;
    /* A callee that writes is finished in the interpreter from a selfSlow
     * record, and this arm takes one of its own -- the same budget a pinned
     * direct call draws from. */
    if (!cfn->jitFuncNoWrite && e->selfSlowCount >= JIT_MAX_SELF_SLOW) {
        return false;
    }
    if (!cfn->jitFuncNoWrite && finishDropsAnArgument(cfn, nargs)) {
        return false;
    }
    return true;
}

/* The compile-time half of a ONE-WAY inline cache: what the model could not
 * pin about a receiver, read off the site's own InlineCache instead.
 *
 * One way only, and only the FIRST shape this site ever recorded --
 * ic->cached[0] / ic->shapeId[0] -- not the whole walk over every way a
 * polymorphic cache holds. The receiver's class is loaded once and compared
 * against that one shape; a hit branches straight into the recorded callee's
 * compiled entry exactly as a pinned receiver already does (emitDirectCall).
 * A miss -- every OTHER class at a polymorphic site, and the common case at a
 * megamorphic one -- falls through to jitInvokeByName, which is exactly what
 * the site emits today and is unchanged by any of this.
 *
 * The compare is not a guard and a miss is not a deopt: both sides of it run
 * BEFORE the call, so nothing has happened yet that must not happen twice.
 * That is the whole reason this needs no deopt record of its own, unlike the
 * call inside it (emitDirectCall still takes one, for what the CALL can do
 * after it starts).
 *
 * `havePrediction`/`rkind` is the fall-through's own prediction
 * (siteInvokeResultKind), which the one way must also return: only a
 * register-only kind (SLOT_INT/FLOAT/BOOL) is admitted, because those are the
 * only ones that carry no class shape for the two paths to disagree about --
 * see feedbackSlotKind's exclusion of SLOT_INST. `*toEnd` comes back holding
 * the arm's own branch to the merge point, for the caller to patch once it
 * has emitted the fall-through after it.
 *
 * Returns false having emitted nothing whenever there was something to
 * cleanly decline, so the caller can still take the descriptor path alone --
 * UNLESS e->failed, which means the shape compare is already in the stream
 * and there is nowhere left to fall back to (mirrors emitDirectCall's own
 * rule, since this arm ends by calling into it). */
bool emitInvokePic1(Emit *e, ObjFunction *fn, unsigned ridx,
                           unsigned argc, uint32_t callOff, uint32_t after,
                           int siteCache, bool havePrediction, SlotKind rkind,
                           int *toEnd) {
    if (!havePrediction) {
        return subWhy(e, "no predicted result kind to join the arm on");
    }
    if (rkind != SLOT_INT && rkind != SLOT_FLOAT && rkind != SLOT_BOOL) {
        return subWhy(e, "an unpinned receiver returning something with a shape");
    }
    if (siteCache < 0 || fn->chunk.caches == NULL ||
        siteCache >= fn->chunk.cacheCount) {
        return subWhy(e, "no inline cache recorded for this site");
    }
    const InlineCache *ic = &fn->chunk.caches[siteCache];
    /* IC_MEGA is admitted, and it is the case that matters. A site that runs
     * out of ways stops caching ALTOGETHER and re-resolves every call through
     * sMegaCache -- see the comment on that table -- which is exactly the
     * shape a trait with eight implementations makes, and exactly the shape
     * this arm exists for. The ways it recorded before it gave up are still
     * there and still true: a way is (shapeId, method), shapeIds come from a
     * monotonic counter that would need four billion classes to repeat, and
     * every way is re-checked below for a live class and a compiled callee.
     * Four ways against eight classes is a partial cache, not a complete one,
     * and a partial cache is the whole point -- the misses cost one compare
     * each and then do exactly what the site does today. */
    if (ic->state != IC_MONO && ic->state != IC_POLY &&
        ic->state != IC_MEGA) {
        return subWhy(e, "the site's cache is empty");
    }
    if (ic->count == 0) return subWhy(e, "the site's cache has no way filled");
    if (ridx + argc + 1u > JIT_MAX_STACK) {
        return subWhy(e, "past the stack depth the model can describe");
    }
    /* Every way the cache holds is tried, in the order it recorded them, so
     * a site that warmed up on its second-most-common class no longer
     * speculates on the wrong one. The compares chain: way w's compare falls
     * through to way w+1's, and the last falls through to the descriptor the
     * site emits today. A miss therefore costs one compare per way and then
     * does exactly what it did before. */
    /* Collect the ways this compile can actually take. A way is dropped, not
     * fatal: the site keeps its remaining arms and the dropped class simply
     * goes round the descriptor as it does today. */
    unsigned    wayShape[JAI_IC_WAYS];
    Value       wayVal  [JAI_IC_WAYS];
    ObjFunction *wayFn  [JAI_IC_WAYS];
    ObjClass    *wayCls [JAI_IC_WAYS];
    bool        wayInlineOnly[JAI_IC_WAYS];
    unsigned    ways = 0;
    unsigned    curable = 0;   /* dropped ways that time alone can mend */

    for (int w = 0; w < ic->count && w < JAI_IC_WAYS; w++) {
        const char *drop = NULL;
        /* What the cache settles is which method a shape resolves to, not
         * whether THIS caller may call it -- one site can present as two
         * classes at different visibilities, so the interpreter re-decides
         * that on every hit and nothing emitted here can. */
        Value cv = ic->cached[w];
        ObjFunction *cf = IS_CLOSURE(cv) ? AS_CLOSURE(cv)->fn : NULL;
        ObjClass *cc = NULL;
        bool later = false;   /* a drop that time alone can mend */
        if (ic->payload[w] != 0) drop = "not public";
        else if (cf == NULL) drop = "not a closure";
        else if (cf->jitFunc == NULL) {
            drop = "callee not compiled";
            later = true;
        } else if ((SlotKind)cf->jitReturnKind != rkind ||
                 cf->jitReturnShape != 0) {
            drop = "callee returns another kind";
        } else if (!jaiClassForShape(ic->shapeId[w], &cc) || cc == NULL) {
            drop = "class not on record";
            later = true;
        } else if (!jitPic1Admissible(e, fn, cf, ridx, argc, ic->shapeId[w])) {
            drop = "not admissible";
        }
        /* A way this site cannot CALL directly -- the callee has not
         * compiled, or its compiled form was specialised to something else
         * -- can still have its body inlined, which needs neither: only the
         * class, a public method, and a body the inliner speaks. Whether it
         * does is settled at emission, where the way's model exists; a body
         * it cannot speak leaves the way empty and both edges of its compare
         * fall through to the next. The return kind has to match what the
         * merge expects, so the interpreter's record of it is consulted. */
        bool inlineOnly = false;
        if (drop != NULL && cf != NULL && ic->payload[w] == 0 &&
            jitPicInlineOnlyOn() && !e->noInline && cf->upvalueCount == 0 &&
            (cf->jitFunc == NULL || drop[0] == 'n' /* not admissible */)) {
            ObjClass *ic2 = NULL;
            SlotKind ok2;
            uint32_t oshape = 0;
            bool kindOk = cf->jitFunc != NULL
                ? ((SlotKind)cf->jitReturnKind == rkind &&
                   cf->jitReturnShape == 0)
                : (observedReturnKind(cf, &ok2, &oshape, NULL) &&
                   ok2 == rkind && oshape == 0);
            if (kindOk && jaiClassForShape(ic->shapeId[w], &ic2) && ic2 != NULL) {
                cc = ic2;
                inlineOnly = true;
                drop = NULL;
            }
        }
        if (drop != NULL) {
            if (later) curable++;
            if (getenv("JAI_JIT_WHY")) {
                fprintf(stderr, "[jit] pic way %d at %u dropped: %s\n",
                        w, callOff, drop);
            }
            continue;
        }
        wayShape[ways] = ic->shapeId[w];
        wayVal[ways]   = cv;
        wayFn[ways]    = cf;
        wayCls[ways]   = cc;
        wayInlineOnly[ways] = inlineOnly;
        ways++;
    }
    /* Fewer ways than the cache holds is usually a matter of TIME, not of
     * the program: the tick that compiled this loop landed before some of
     * the site's callees had been called often enough to compile. The form
     * would otherwise keep that shortfall for the rest of the run, every
     * missing class going round the descriptor -- poly_dispatch measured
     * 400M cycles for a 1-way form against 125M for the 8-way one, decided
     * by when the timer fired. Noted, so the form can be compiled again
     * once the stragglers have compiled. */
    /* A cache that can still grow is short too: a site whose first callee
     * had barely run was compiled "1-way of 1 recorded", and the seven
     * classes it met afterwards never reached the form. */
    if (e->osr && (curable > 0 ||
                   (ic->state != IC_MEGA && ic->count < JAI_IC_WAYS))) {
        jitOsrPicShort(siteCache, ways, rkind);
    }
    if (ways == 0) return subWhy(e, "no way of this site's cache is usable");

    settleAll(e);
    fpReleaseAll(e);
    if (e->fpLive != 0) {
        return subWhy(e, "an unpinned receiver with a value in the float bank");
    }

    /* Past here the shape compare is in the stream and the site is
     * committed: a failure below stops the compile rather than falling
     * back. */
    unsigned rreg = valueXReg(e, ridx - (e->depth - e->valueDepth));
    /* The receiver's shape is loaded ONCE and every way compares against it. */
    emit(e, jaiA64LdrX(JIT_SCRATCH_A, rreg,
                       (unsigned)offsetof(ObjInstance, klass)));
    emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_A,
                       (unsigned)offsetof(ObjClass, shapeId)));

    /* The model as the fall-through must find it again: emitDirectCall
     * consumes the receiver and the arguments and pushes a result, and the
     * fall-through's own descriptor call has to see the SAME depth and
     * valueDepth it would have without any of this, or the two paths
     * disagree about where the result lands. */
    unsigned  saveDepth      = e->depth;
    unsigned  saveValueDepth = e->valueDepth;
    SlotKind  saveKind [JIT_MAX_STACK];
    uint32_t  saveShape[JIT_MAX_STACK];
    ObjClass *saveClass[JIT_MAX_STACK];
    Value     saveSeen [JIT_MAX_STACK];
    memcpy(saveKind,  e->stack,      sizeof saveKind);
    memcpy(saveShape, e->stackShape, sizeof saveShape);
    memcpy(saveClass, e->stackClass, sizeof saveClass);
    memcpy(saveSeen,  e->stackSeen,  sizeof saveSeen);

    /* One arm a way. Each is: prove the shape, call directly, jump to the
     * merge. A way that does not match falls into the next way's compare, and
     * the last falls into the descriptor path the caller emits -- which is
     * what this site did for every receiver before any of this. */
    int armMiss[JAI_IC_WAYS];
    for (unsigned w = 0; w < ways; w++) {
        /* Shape ids come from a counter, so nearly all fit an imm12 and the
         * compare needs no register -- one word a way, eight ways a chain. */
        if (wayShape[w] <= 4095u) {
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, wayShape[w]));
        } else {
            emitConst64(e, JIT_SCRATCH_B, (int64_t)wayShape[w]);
            emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
        }
        armMiss[w] = (int)e->count;
        emit(e, jaiA64BCond(JAI_A64_NE, 0));

        /* True on this arm alone, which is why it is also what the CALL's own
         * deopt records (inside emitDirectCall, for what happens after the
         * call starts -- not for this compare) should say: the branch just
         * above proved it. */
        e->depth      = saveDepth;
        e->valueDepth = saveValueDepth;
        memcpy(e->stack,      saveKind,  sizeof saveKind);
        memcpy(e->stackShape, saveShape, sizeof saveShape);
        memcpy(e->stackClass, saveClass, sizeof saveClass);
        memcpy(e->stackSeen,  saveSeen,  sizeof saveSeen);
        e->stack[ridx]      = SLOT_INST;
        e->stackShape[ridx] = wayShape[w];
        e->stackClass[ridx] = wayCls[w];

        /* Cheapest first, as at a pinned site: the way's body where the
         * call is. The merge after the ways reads one X register at one
         * depth, so a float result the inline left in the FP bank, and
         * anything it deferred, is settled before the branch out. */
        unsigned inlCount = e->count;
        if (inlineMethodCall(e, fn, AS_CLOSURE(wayVal[w]), argc, callOff)) {
            if (e->depth == 0 || e->stack[e->depth - 1] != rkind) {
                gInlineFailed = true;
                e->failed = true;
                return false;
            }
            fpSyncAll(e);
            settleAll(e);
            if (getenv("JAI_JIT_WHY")) {
                fprintf(stderr, "[jit] pic way %u inlined (%u instructions)\n",
                        w, e->count - inlCount);
            }
        } else if (e->failed) {
            return false;
        } else if (wayInlineOnly[w]) {
            /* Nothing emitted past the compare: the hit edge falls into the
             * next way's compare, as the miss edge does, and the receiver
             * goes round the descriptor. */
            if (e->count != inlCount) { e->failed = true; return false; }
            if (armMiss[w] < (int)e->count && e->count <= JIT_MAX_INSTS) {
                e->code[armMiss[w]] =
                    jaiA64BCond(JAI_A64_NE, (int32_t)((int)e->count - armMiss[w]));
            }
            continue;
        } else if (!emitDirectCall(e, fn, wayFn[w], wayVal[w], -1, ridx, argc,
                            callOff, after, true)) {
            /* jitPic1Admissible said it would take this and it did not: the
             * branch into it is already emitted, so there is nowhere left to
             * fall back to. */
            e->failed = true;
            return false;
        }
        if (e->picExitCount >= JIT_MAX_PIC_EXITS) {
            e->failed = true;
            return false;
        }
        e->picExits[e->picExitCount++] = (int)e->count;
        emit(e, jaiA64B(0));

        /* Same condition as the placeholder (NE: skip the call on a shape
         * that doesn't match), now with the real offset -- flipping it to EQ
         * would call this way's callee on every receiver whose shape did NOT
         * match, reading its fields at the wrong class's layout. That is what
         * test_mixed_list_runs_every_class caught: a wrong answer, not a
         * crash, because the read lands inside the instance's own allocation.
         *
         * Guarded because emit() silently DROPS the word once e->count reaches
         * JIT_MAX_INSTS and only sets e->failed -- so the slot may never have
         * been written, and Emit::code is immediately followed by `count` with
         * no padding between them. */
        if (armMiss[w] < (int)e->count && e->count <= JIT_MAX_INSTS) {
            e->code[armMiss[w]] =
                jaiA64BCond(JAI_A64_NE, (int32_t)((int)e->count - armMiss[w]));
        }
    }

    /* The model the caller's descriptor path must find, restored exactly. */
    e->depth      = saveDepth;
    e->valueDepth = saveValueDepth;
    memcpy(e->stack,      saveKind,  sizeof saveKind);
    memcpy(e->stackShape, saveShape, sizeof saveShape);
    memcpy(e->stackClass, saveClass, sizeof saveClass);
    memcpy(e->stackSeen,  saveSeen,  sizeof saveSeen);

    if (getenv("JAI_JIT_WHY")) {
        fprintf(stderr, "[jit] pic %u-way (of %u recorded, state %d) at %u\n",
                ways, (unsigned)ic->count, (int)ic->state, callOff);
    }

    /* The fall-through runs with the receiver and the arguments untouched,
     * so the caller emits its descriptor path against the model as it was
     * before any of this. */
    e->depth      = saveDepth;
    e->valueDepth = saveValueDepth;
    memcpy(e->stack,      saveKind,  sizeof saveKind);
    memcpy(e->stackShape, saveShape, sizeof saveShape);
    memcpy(e->stackClass, saveClass, sizeof saveClass);
    memcpy(e->stackSeen,  saveSeen,  sizeof saveSeen);
    return true;
}

#endif /* __aarch64__ || __arm64__ */
