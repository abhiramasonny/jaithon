/* jit_call_inline.c -- inlining a callee's body at the call site, for the two
 * shapes the tier admits: a global function and a one-expression method. */

#include "vm/jit/jit_arm64.h"
#include "vm/jit/jit_field_read.h"
#include "vm/vm.h"

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "vm/jit/jit_internal.h"

#if (defined(__aarch64__) || defined(__arm64__))

/* Something inside an inlined body could not be emitted, so the whole compile
 * is worth retrying with inlining off rather than declining: the same call
 * through the descriptor still compiles, and a compiled form with a real call
 * in it beats none at all. A file static for the same reason the Emit buffers
 * are -- compilation is not reentrant, nothing it calls compiles anything. */
bool gInlineFailed;

/* Structural check, answered before anything is emitted (a half-inlined body can't be taken back):
 * no branches (no offset map, no join, no fixup naming a callee offset in the caller's table); exactly one RETURN, last; locals only via the four opcodes the inline frame understands, and only slots this callee actually has; globals only for the two builtins the tier emits inline (else a global VALUE load would bake a JaiEntry from the callee's own table, needing its own guard); nothing that stores (a guard inside re-executes the WHOLE call, so an earlier store would run twice). What's left is straight-line register arithmetic -- the main walker already speaks it, so no second emitter is needed. `evalA` in spectral is fifteen instructions of exactly this shape. */
static bool inlinableBody(ObjClosure *callee, unsigned argc,
                          unsigned *maxSlotOut, bool *readsUpvalueOut) {
    ObjFunction *cfn = callee->fn;
    const Chunk *c = &cfn->chunk;
    if (cfn->arity != argc || cfn->defaultCount != 0) return false;
    if (cfn->flags & (FN_VARIADIC | FN_KWREST | FN_INIT)) return false;
    if (c->count <= 0 || c->count > 128) return false;

    unsigned maxSlot = argc;
    bool sawReturn = false;
    bool readsUpvalue = false;
    for (int off = 0; off < c->count;) {
        uint8_t op = c->code[off];
        int len = instructionLength(c, off);
        if (len <= 0) return false;
        unsigned slot = 0, slot2 = 0;
        switch (op) {
        /* The local frame the caller builds understands exactly these. Any
         * other opcode naming a slot -- a field read off one, a compare
         * against one, an in-place update -- would read the CALLER's local of
         * that number, which is a different variable entirely. */
        case OP_GET_LOCAL:
        case OP_BIND:
            slot = jaiReadU16(c->code + off + 1);
            if (slot > maxSlot) maxSlot = slot;
            break;
        case OP_GET_LOCAL2:
        case OP_ADD_LOCALS:
            slot  = jaiReadU16(c->code + off + 1);
            slot2 = jaiReadU16(c->code + off + 3);
            if (slot > maxSlot) maxSlot = slot;
            if (slot2 > maxSlot) maxSlot = slot2;
            break;
        /* An upvalue is reached through the closure that is actually being
         * called, which is a register the call site has to supply -- so this
         * is only inlinable where that register exists. OP_SET_UPVALUE is not
         * here and falls to `default`: a store would have to be undone if a
         * later guard in the same body deoptimised to the call. */
        case OP_GET_UPVALUE:
            if ((unsigned)c->code[off + 1] >= (unsigned)cfn->upvalueCount) {
                return false;
            }
            readsUpvalue = true;
            break;
        case OP_GET_GLOBAL: {
            uint32_t nameIdx = jaiReadU24(c->code + off + 1);
            Value nv;
            if (globalNative(callee, nameIdx, &nv) == NULL) return false;
            ObjNative *nat = AS_NATIVE(nv);
            const char *nm = nat->name != NULL ? nat->name->chars : "";
            if (strcmp(nm, "float") != 0 && strcmp(nm, "int") != 0) return false;
            break;
        }
        case OP_CALL:
            /* The only callee that can be on the stack here is one of the two
             * builtins above, and the tier emits those as one instruction. */
            if (c->code[off + 1] != 1) return false;
            break;
        case OP_CONST: case OP_INT: case OP_TRUE: case OP_FALSE:
        case OP_ADD: case OP_SUB: case OP_MUL: case OP_DIV:
        case OP_FLOORDIV: case OP_MOD: case OP_POW: case OP_NEG:
        case OP_BAND: case OP_BOR: case OP_BXOR:
        case OP_SHL: case OP_SHR: case OP_BNOT:
        case OP_TYPE_GUARD:
            break;
        case OP_RETURN:
            if (off + len != c->count) return false;
            sawReturn = true;
            break;
        default:
            return false;
        }
        off += len;
    }
    if (!sawReturn) return false;
    if (maxSlot > JIT_MAX_SLOTS) return false;
    *maxSlotOut = maxSlot;
    *readsUpvalueOut = readsUpvalue;
    return true;
}

/* Inlines the callee's body: slot 1+i IS entry cidx+1+i already on the stack, so nothing is copied in;
 * a bound slot pins one more entry underneath what's pushed after it, sound only because the body is straight-line. Callee's module must be the caller's, and its baked builtins are retired by the CALLER's own module-version check (the callee's is never run). `calleeReg`: needed only if the body reads an upvalue, since `callee` is a SAMPLE closure at an indirect site -- one ObjFunction, many closures (`|x| x + step`), so its captured cells aren't necessarily the next call's. Constants/globals are safe from the sample since they belong to the function/module, not the closure. */
bool inlineGlobalCall(Emit *e, ObjFunction *caller, ObjClosure *callee,
                             unsigned argc, uint32_t callOff, int calleeReg) {
    if (e->noInline) return false;
    /* An inlined body's entries want x0..x8 (inlineOwnBank) and a split bank
     * is already using them, so the plan withholds the split from a body the
     * measuring pass saw inline. Refusing here as well is what makes that a
     * fact rather than an agreement between two passes: the worst this can do
     * is decline an inline the probe never took. */
    if (e->splitAt != 0) return false;
    ObjFunction *cfn = callee->fn;
    if (cfn->module != caller->module) return false;
    if (e->inlining) return false;             /* one level, no recursion */
    /* The inlined body's offsets are the callee's, so `inProtected` describes
     * the CALLER's regions throughout -- a `try` of the callee's own would go
     * unseen. inlinableBody's whitelist already refuses every opcode a handler
     * needs; this says so rather than relying on it. */
    if (cfn->exceptionCount > 0) return false;
    unsigned cidx = e->depth - argc - 1;
    unsigned maxSlot = 0;
    bool readsUpvalue = false;
    if (!inlinableBody(callee, argc, &maxSlot, &readsUpvalue)) return false;
    if (readsUpvalue && calleeReg < 0) return false;
    if (getenv("JAI_JIT_WHY")) {
        fprintf(stderr, "[jit] inlining %s\n",
                cfn->name ? cfn->name->chars : "<anon>");
    }

    /* Every argument has to be in a register, since that is where the body
     * will read its parameters from. */
    for (unsigned i = 0; i < argc; i++) {
        if (!holdsRegister(e->stack[cidx + 1u + i])) return false;
    }

    int savedSlot[JIT_MAX_SLOTS + 1];
    memcpy(savedSlot, e->inlSlot, sizeof savedSlot);
    for (unsigned i = 0; i <= JIT_MAX_SLOTS; i++) e->inlSlot[i] = -1;
    for (unsigned i = 0; i < argc; i++) e->inlSlot[1u + i] = (int)(cidx + 1u + i);

    /* No noteScratchClobber here. An inlined body cannot call -- inlinableBody
     * admits nothing that does -- so it destroys x0..x8 only by USING them,
     * which is not a clobber but an allocation: when the caller already owns
     * that bank the two share one numbering (see inlineOwnBank), and when it
     * does not, the inlined entries have x0..x8 to themselves as before.
     * Anything inside that really does call still reaches noteScratchClobber
     * on its own, and under scratchValues that declines the compile. */
    e->inlining     = true;
    e->inlDepth     = cidx + 1u + argc;
    e->inlPinned    = 0;
    e->inlValueBase = e->valueDepth;
    e->inlIp        = callOff;
    e->inlClosureReg = calleeReg;

    /* The callee's own offset map, so its offsets cannot land in the
     * caller's. Nothing reads it back -- there are no branches -- but
     * compileBody writes one entry per instruction either way. */
    int cmap[129];
    int64_t cdepths[129];
    for (int i = 0; i <= cfn->chunk.count; i++) { cmap[i] = -1; cdepths[i] = -1; }
    int *savedMap = e->offsetToInst;
    int64_t *savedDepths = e->offsetToDepth;
    unsigned savedCarry = e->fpCarryCount;
    uint32_t savedCurOffset = e->curOffset;
    unsigned savedInstDepth = e->instDepth;
    unsigned savedInstValue = e->instValueDepth;
    e->offsetToInst = cmap;
    e->offsetToDepth = cdepths;

    bool ok = compileBody(e, callee);

    e->offsetToInst = savedMap;
    e->offsetToDepth = savedDepths;
    e->fpCarryCount = savedCarry;
    e->curOffset = savedCurOffset;
    e->instDepth = savedInstDepth;
    e->instValueDepth = savedInstValue;

    if (!ok || e->failed) {
        /* Instructions have been written; there is no taking them back. The
         * whole compile is retried with inlining off, which is the same answer
         * the register budget already gets. */
        e->inlining = false;
        memcpy(e->inlSlot, savedSlot, sizeof savedSlot);
        gInlineFailed = true;
        e->failed = true;
        return false;
    }

    /* OP_RETURN left the result on top and everything the body pinned beneath
     * it. Both are read while `inlining` is still set, because that is what
     * says which bank they are in; only the result's new home belongs to the
     * caller. */
    unsigned rres;
    SlotKind kres;
    uint32_t rshape;
    ObjClass *rcls;
    if (e->depth <= cidx) { e->failed = true; return false; }
    rshape = e->stackShape[e->depth - 1];
    rcls   = e->stackClass[e->depth - 1];
    /* Read while `inlining` is still set, so this names the inlined bank's d
     * register; the caller's own is taken after it is cleared. */
    bool rfp = (e->fpLive & (1u << (e->valueDepth - 1))) != 0;
    unsigned rfpReg = rfp ? fpHeldIn(e, e->valueDepth - 1) : 0;
    if (rfp) {
        if (!popValueRaw(e, &rres, &kres)) { e->failed = true; return false; }
    } else if (!popValue(e, &rres, &kres)) { e->failed = true; return false; }
    /* Raw, because nothing reads these again: the body is over and its pinned
     * locals go with it, so materialising one costs an instruction whose
     * destination is dead. */
    while (e->depth > cidx) {
        if (holdsRegister(e->stack[e->depth - 1])) {
            unsigned r;
            if (!popValueRaw(e, &r, NULL)) { e->failed = true; return false; }
        } else {
            e->depth--;
        }
    }
    e->inlining = false;
    memcpy(e->inlSlot, savedSlot, sizeof savedSlot);

    if (!pushValue(e, kres, rshape, rcls)) { e->failed = true; return false; }
    if (rfp) {
        unsigned dd = fpRegAt(e, e->valueDepth - 1);
        if (dd != rfpReg) emit(e, jaiA64FmovDD(dd, rfpReg));
        fpClaim(e, e->valueDepth - 1);
    } else {
        unsigned dst = pushReg(e) - 1;
        if (dst != rres) emit(e, jaiA64MovX(dst, rres));
    }
    e->inlined = true;
    return true;
}

/* Emit a method's body directly, when that body is one expression.
 *
 * Deliberately narrow: no jumps, no stores, no calls, only field reads of its
 * own parameters and int or float arithmetic. Those restrictions are what make
 * a second walker over the callee's bytecode safe to write -- with no branches
 * there is no offset map to keep, and with no stores there is nothing to undo
 * if a guard inside it deoptimises to the call site.
 *
 * Reached through inlineMethod, which is what puts the model back when this
 * declines -- see there. */
static bool inlineMethodWalk(Emit *e, ObjClosure *closure, uint32_t nameIdx,
                             unsigned argc, int callOff) {
    if (e->noInline) return false;
    if (e->splitAt != 0) return false;   /* see inlineGlobalCall */
    unsigned ridx = e->depth - argc - 1;
    ObjClass *rcls = e->stackClass[ridx];
    if (rcls == NULL) return false;
    ObjFunction *cfn = closure->fn;
    if (nameIdx >= (uint32_t)cfn->chunk.constants.count) return false;
    Value mname = cfn->chunk.constants.data[nameIdx];
    if (!IS_STRING(mname)) return false;
    Value method;
    if (!jaiClassFindMethod(rcls, AS_STRING(mname), &method)) return false;
    if (!IS_CLOSURE(method)) return false;
    ObjFunction *mfn = AS_CLOSURE(method)->fn;
    if (mfn->exceptionCount > 0) return false;   /* see inlineGlobalCall */
    if (mfn->arity != argc || mfn->defaultCount != 0) return false;
    if (mfn->flags & (FN_VARIADIC | FN_KWREST | FN_INIT)) return false;
    if (mfn->upvalueCount != 0) return false;
    if (mfn->chunk.count > 96) return false;

    unsigned inReg[JIT_MAX_ARGS_OUT + 1];
    Value    inSeen[JIT_MAX_ARGS_OUT + 1];
    ObjClass *inCls[JIT_MAX_ARGS_OUT + 1];
    for (unsigned i = 0; i <= argc; i++) {
        unsigned idx = ridx + i;
        if (!holdsRegister(e->stack[idx])) return false;
        inReg[i]  = valueBankReg(e, idx - (e->depth - e->valueDepth));
        inSeen[i] = e->stackSeen[idx];
        inCls[i]  = e->stackClass[idx];
    }

    /* A dry walk first: nothing is emitted until the whole body is known to
     * be expressible, because a half-inlined body cannot be taken back. */
    const uint8_t *c = mfn->chunk.code;
    int n = mfn->chunk.count;
    for (int pass = 0; pass < 2; pass++) {
        int depth0 = (int)e->depth;
        for (int o = 0; o < n;) {
            uint8_t op = c[o];
            if (op == OP_GET_FIELD_LOCAL) {
                unsigned slot = jaiReadU16(c + o + 1);
                uint32_t nidx = jaiReadU24(c + o + 3);
                if (slot > argc) return false;
                if (e->stack[ridx + slot] != SLOT_INST) return false;
                if (nidx >= (uint32_t)mfn->chunk.constants.count) return false;
                Value fname = mfn->chunk.constants.data[nidx];
                if (!IS_STRING(fname)) return false;
                const FieldInfo *fi =
                    jaiClassFieldInfo(inCls[slot], AS_STRING(fname));
                if (fi == NULL || fi->isStatic) return false;
                if (!IS_INSTANCE(inSeen[slot])) return false;
                ObjInstance *si = AS_INSTANCE(inSeen[slot]);
                if (fi->slot >= si->fieldCount) return false;
                Value fv = si->fields[fi->slot];
                SlotKind fk; unsigned ftag;
                if (IS_INT(fv))        { fk = SLOT_INT;   ftag = VAL_INT; }
                else if (IS_FLOAT(fv)) { fk = SLOT_FLOAT; ftag = VAL_FLOAT; }
                else return false;
                unsigned fbase = (unsigned)offsetof(ObjInstance, fields) +
                                 (unsigned)fi->slot * (unsigned)sizeof(Value);
                if (pass == 1) {
                    emit(e, jaiA64LdrW(JIT_SCRATCH_A, inReg[slot], fbase));
                    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, ftag));
                    branchOnDeoptAt(e, JAI_A64_NE, (uint32_t)callOff, false);
                }
                if (!pushValue(e, fk, 0, NULL)) return false;
                if (pass == 1) {
                    emit(e, jaiA64LdrX(pushReg(e) - 1, inReg[slot], fbase + 8));
                }
                o += 8;
                continue;
            }
            if (op == OP_ADD || op == OP_SUB || op == OP_MUL) {
                unsigned rb, ra; SlotKind kb, ka;
                if (!popValue(e, &rb, &kb)) return false;
                if (!popValue(e, &ra, &ka)) return false;
                if (ka != kb) return false;
                if (ka != SLOT_INT && ka != SLOT_FLOAT) return false;
                if (!pushValue(e, ka, 0, NULL)) return false;
                unsigned rd = pushReg(e) - 1;
                if (pass == 1) {
                    if (ka == SLOT_FLOAT) {
                        emit(e, jaiA64FmovDX(JIT_FSCRATCH_A, ra));
                        emit(e, jaiA64FmovDX(JIT_FSCRATCH_B, rb));
                        emit(e, op == OP_ADD
                                 ? jaiA64FaddD(JIT_FSCRATCH_A, JIT_FSCRATCH_A, JIT_FSCRATCH_B)
                             : op == OP_SUB
                                 ? jaiA64FsubD(JIT_FSCRATCH_A, JIT_FSCRATCH_A, JIT_FSCRATCH_B)
                                 : jaiA64FmulD(JIT_FSCRATCH_A, JIT_FSCRATCH_A, JIT_FSCRATCH_B));
                        emit(e, jaiA64FmovXD(rd, JIT_FSCRATCH_A));
                    } else if (op == OP_MUL) {
                        emit(e, jaiA64SmulhX(JIT_SCRATCH_A, ra, rb));
                        emit(e, jaiA64MulX(rd, ra, rb));
                        emit(e, jaiA64SubsXAsr(31, JIT_SCRATCH_A, rd, 63));
                        branchOnOverflow(e, 2u, JAI_A64_NE);
                    } else {
                        emit(e, op == OP_ADD ? jaiA64AddsX(rd, ra, rb)
                                             : jaiA64SubsXReg(rd, ra, rb));
                        branchOnOverflow(e, op == OP_ADD ? 0u : 1u, JAI_A64_VS);
                    }
                }
                o += 1;
                continue;
            }
            if (op == OP_RETURN) {
                if ((int)e->depth != depth0 + 1) return false;
                o += 1;
                if (o != n) return false;
                break;
            }
            return false;
        }
        if (pass == 0) {
            while ((int)e->depth > depth0) {
                unsigned r; if (!popValue(e, &r, NULL)) return false;
            }
        }
    }

    unsigned rres;
    SlotKind kres;
    if (!popValue(e, &rres, &kres)) return false;
    for (unsigned i = 0; i <= argc; i++) {
        unsigned r; if (!popValue(e, &r, NULL)) return false;
    }
    if (!pushValue(e, kres, 0, NULL)) return false;
    unsigned dst = pushReg(e) - 1;
    if (dst != rres) emit(e, jaiA64MovX(dst, rres));
    e->inlined = true;
    return true;
}

/* The model must be exactly where it was if the inline did not happen.
 *
 * inlineMethodWalk's dry pass pushes and pops as it reads the callee, and every
 * one of its two dozen refusals returns from the middle of that -- so on its
 * own it leaves the model as deep as the walk got. Its caller does NOT decline
 * when it declines: the OP_INVOKE arm falls through to the descriptor path,
 * which then names every later entry's register from an index that is too high
 * and, far worse, writes deopt records describing an operand stack the
 * interpreter does not have. `_crossings` in lib/std/gui/path.jai is the shape
 * that found this: `edge.crossing(y)` gets three instructions into `crossing`
 * before an OP_BIND stops the walk, so every deopt after it handed the
 * interpreter the receiver and the argument a second time and the next
 * instruction read a float where a list belonged.
 *
 * Unwinding here rather than at each `return false` is deliberate: there are
 * far too many of them to keep right by hand, and the dry pass's own tail
 * already pops back to its starting depth in exactly this way.
 *
 * A failure that has already emitted cannot be unwound at all -- the caller
 * would stack a second call sequence on top of half of this one -- so that
 * declines the compile instead. */
bool inlineMethod(Emit *e, ObjClosure *closure, uint32_t nameIdx,
                         unsigned argc, int callOff) {
    unsigned depth0 = e->depth;
    unsigned count0 = e->count;
    if (inlineMethodWalk(e, closure, nameIdx, argc, callOff)) return true;
    if (e->failed) return false;
    if (e->count != count0) { e->failed = true; return false; }
    while (e->depth > depth0) {
        unsigned r;
        if (!popValue(e, &r, NULL)) { e->failed = true; return false; }
    }
    /* Below where it started is not something an unwind can repair: the
     * entries are the caller's and their registers are gone. */
    if (e->depth != depth0) { e->failed = true; return false; }
    return false;
}

#endif /* __aarch64__ || __arm64__ */
