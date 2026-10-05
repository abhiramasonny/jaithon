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

/* JAITHON_JIT_INLINE_METHODS=0 restores the narrow method inliner alone
 * (inlineMethodWalk), and keeps field reads out of every inlined body. */
static bool jitInlineMethodsOn(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("JAITHON_JIT_INLINE_METHODS");
        on = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return on != 0;
}

/* Structural check, answered before anything is emitted (a half-inlined body can't be taken back):
 * no branches (no offset map, no join, no fixup naming a callee offset in the caller's table); exactly one RETURN, last; locals only via the four opcodes the inline frame understands, and only slots this callee actually has; globals only for the two builtins the tier emits inline (else a global VALUE load would bake a JaiEntry from the callee's own table, needing its own guard); nothing that stores (a guard inside re-executes the WHOLE call, so an earlier store would run twice). What's left is straight-line register arithmetic -- the main walker already speaks it, so no second emitter is needed. `evalA` in spectral is fifteen instructions of exactly this shape. */
/* JAITHON_JIT_INLINE_CONSTRUCT: see inlinableBody's OP_TAIL_CALL. */
static bool jitInlineConstructOn(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("JAITHON_JIT_INLINE_CONSTRUCT");
        on = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return on != 0;
}

static bool inlinableBody(ObjClosure *callee, unsigned argc,
                          unsigned *maxSlotOut, bool *readsUpvalueOut,
                          bool *constructsOut) {
    ObjFunction *cfn = callee->fn;
    const Chunk *c = &cfn->chunk;
    if (cfn->arity != argc || cfn->defaultCount != 0) return false;
    if (cfn->flags & (FN_VARIADIC | FN_KWREST | FN_INIT)) return false;
    if (c->count <= 0 || c->count > 128) return false;

    unsigned maxSlot = argc;
    bool sawReturn = false;
    bool readsUpvalue = false;
    ObjClass *ctor = NULL;      /* a class this body constructs, last */
    bool constructs = false;
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
        /* A field read is a load, and the walker's own OP_GET_FIELD arm is
         * written against an operand-stack receiver, which is exactly what
         * an inlined body has: compileBody reads OP_GET_FIELD_LOCAL as a copy
         * of the slot's entry plus that arm (inlineFieldLocal), and refuses
         * any receiver that is not an instance of a pinned class. Every
         * guard in it deoptimises to the call, which is sound for the same
         * reason the arithmetic is: nothing here has stored anything. */
        case OP_GET_FIELD_LOCAL:
            if (!jitInlineMethodsOn()) return false;
            slot = jaiReadU16(c->code + off + 1);
            if (slot > maxSlot) maxSlot = slot;
            break;
        case OP_GET_FIELD:
            if (!jitInlineMethodsOn()) return false;
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
            ObjClass *gc = NULL;
            if (globalNative(callee, nameIdx, &nv) == NULL &&
                (gc = globalClass(callee, nameIdx)) != NULL) {
                /* A class, resolved at compile time and pinned by the
                 * caller's module-version check (the callee's module is the
                 * caller's). Admitted only as the callee of the closing
                 * construction -- see OP_TAIL_CALL. */
                if (!jitInlineMethodsOn() || !jitInlineConstructOn()) {
                    return false;
                }
                if (ctor != NULL) return false;
                ctor = gc;
                break;
            }
            if (gc == NULL && globalNative(callee, nameIdx, &nv) == NULL) {
                return false;
            }
            ObjNative *nat = AS_NATIVE(nv);
            const char *nm = nat->name != NULL ? nat->name->chars : "";
            if (strcmp(nm, "float") != 0 && strcmp(nm, "int") != 0) return false;
            break;
        }
        case OP_CALL:
            /* The only callee that can be on the stack here is one of the two
             * builtins above, and the tier emits those as one instruction --
             * unless a class was read, which only a tail call may consume. */
            if (c->code[off + 1] != 1 || ctor != NULL) return false;
            break;
        /* `return C(a, b)` closing the body: an allocation, and stores into
         * the object just allocated, of a class whose init does nothing else
         * (jitSimpleInitClass). Those stores are the one exception to the no
         * stores rule, and they are safe for the reason the rule exists: a
         * guard that deoptimises to the call re-runs the whole call, and the
         * only thing the first run left behind is an object nothing refers
         * to. Nothing follows it but the OP_RETURN, so no guard comes after
         * the allocation anyway. The allocator's slow path is a real call,
         * which is why such a body shares the caller's bank (Emit::inlShared). */
        case OP_TAIL_CALL:
            if (ctor == NULL || constructs) return false;
            if (!jitSimpleInitClass(ctor, c->code[off + 1])) return false;
            if (off + len >= c->count || c->code[off + len] != OP_RETURN ||
                off + len + 1 != c->count) {
                return false;
            }
            constructs = true;
            break;
        /* `x % k` with a small literal k, fused: it pops and pushes the top
         * entry and names no local, so it reads the inlined body's own
         * operand stack exactly as OP_MOD does. */
        case OP_MOD_INT_CONST:
            if (!jitInlineMethodsOn()) return false;
            break;
        /* `x + k`, `x - k`, `x * k` fused with the read of `x`: the slot is
         * a parameter or a bound local of this body, read through inlSlot
         * like any other (inlineIntConstOp), and refused up front unless it
         * is an int (inlineFieldsReadable). `return n - 1` is the commonest
         * body there is, and this is what it compiles to. */
        case OP_ADD_INT_CONST:
        case OP_SUB_INT_CONST:
        case OP_MUL_INT_CONST:
            if (!jitInlineMethodsOn()) return false;
            slot = jaiReadU16(c->code + off + 1);
            if (slot > maxSlot) maxSlot = slot;
            break;
        /* A predicate -- `return self.pos >= self.len` -- is a compare and a
         * `cset`: straight-line, and emitCompare reads its operands through
         * xHeldIn/fpOperand, so the inlined bank is where it looks. Its
         * string and object-equality arms, the two that call out, already
         * refuse inside an inline. */
        case OP_EQ: case OP_NE: case OP_LT: case OP_LE: case OP_GT: case OP_GE:
            if (!jitInlineMethodsOn()) return false;
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
    if (ctor != NULL && !constructs) return false;
    *maxSlotOut = maxSlot;
    *readsUpvalueOut = readsUpvalue;
    *constructsOut = constructs;
    return true;
}

/* Inlines the callee's body: slot 1+i IS entry cidx+1+i already on the stack, so nothing is copied in;
 * a bound slot pins one more entry underneath what's pushed after it, sound only because the body is straight-line. Callee's module must be the caller's, and its baked builtins are retired by the CALLER's own module-version check (the callee's is never run). `calleeReg`: needed only if the body reads an upvalue, since `callee` is a SAMPLE closure at an indirect site -- one ObjFunction, many closures (`|x| x + step`), so its captured cells aren't necessarily the next call's. Constants/globals are safe from the sample since they belong to the function/module, not the closure. */
static bool inlineCallAt(Emit *e, ObjFunction *caller, ObjClosure *callee,
                         unsigned argc, uint32_t callOff, int calleeReg,
                         bool method);

/* Whether the entry an inlined body's slot names is a receiver whose field
 * `name` the OP_GET_FIELD arm can read: an instance of a pinned class, a
 * plain instance field, and a kind for it -- off the entry's own sample when
 * that is of the same class, else from the declaration. */
static bool inlineFieldOf(const Emit *e, int idx, const ObjFunction *cfn,
                          uint32_t nameIdx) {
    if (idx < 0 || (unsigned)idx >= e->depth) return false;
    if (e->stack[idx] != SLOT_INST) return false;
    ObjClass *klass = e->stackClass[idx];
    if (klass == NULL) return false;
    if (nameIdx >= (uint32_t)cfn->chunk.constants.count) return false;
    Value nv = cfn->chunk.constants.data[nameIdx];
    if (!IS_STRING(nv)) return false;
    const FieldInfo *fi = jaiClassFieldInfo(klass, AS_STRING(nv));
    if (fi == NULL || fi->isStatic) return false;
    Value seen = e->stackSeen[idx];
    if (IS_INSTANCE(seen) && AS_INSTANCE(seen)->klass == klass) {
        if (fi->slot >= AS_INSTANCE(seen)->fieldCount) return false;
        Value fv = AS_INSTANCE(seen)->fields[fi->slot];
        return IS_INT(fv) || IS_FLOAT(fv) || IS_BOOL(fv);
    }
    SlotKind dk;
    unsigned dtag;
    return jitDeclaredFieldKindEnabled() &&
           declaredScalarFieldKind(fi->typeId, &dk, &dtag);
}

/* Every field read of the body, against the slot mapping just made. An
 * OP_GET_FIELD is only admitted straight after the local read that put its
 * receiver on top, which is the shape the compiler emits for `p.x`; anything
 * else (a chain, a field of a call's result) is refused here rather than met
 * half-way. */
static bool inlineFieldsReadable(const Emit *e, ObjClosure *callee) {
    const ObjFunction *cfn = callee->fn;
    const Chunk *c = &cfn->chunk;
    int lastSlot = -1;   /* the slot the previous instruction left on top */
    for (int off = 0; off < c->count;) {
        uint8_t op = c->code[off];
        int len = instructionLength(c, off);
        if (len <= 0) return false;
        int top = -1;
        switch (op) {
        case OP_GET_LOCAL:
            top = (int)jaiReadU16(c->code + off + 1);
            break;
        case OP_GET_LOCAL2:
            top = (int)jaiReadU16(c->code + off + 3);
            break;
        case OP_GET_FIELD_LOCAL: {
            unsigned slot = jaiReadU16(c->code + off + 1);
            if (slot > JIT_MAX_SLOTS) return false;
            if (!inlineFieldOf(e, e->inlSlot[slot], cfn,
                               jaiReadU24(c->code + off + 3))) {
                return false;
            }
            break;
        }
        case OP_GET_FIELD:
            if (lastSlot < 0 || lastSlot > (int)JIT_MAX_SLOTS) return false;
            if (!inlineFieldOf(e, e->inlSlot[lastSlot], cfn,
                               jaiReadU24(c->code + off + 1))) {
                return false;
            }
            break;
        case OP_ADD_INT_CONST:
        case OP_SUB_INT_CONST:
        case OP_MUL_INT_CONST: {
            /* A bound local's entry does not exist yet, and the arm checks
             * it again when it does; a parameter's is checked here. */
            unsigned slot = jaiReadU16(c->code + off + 1);
            if (slot > JIT_MAX_SLOTS) return false;
            int idx = e->inlSlot[slot];
            if (idx >= 0 && ((unsigned)idx >= e->depth ||
                             e->stack[idx] != SLOT_INT)) {
                return false;
            }
            break;
        }
        default:
            break;
        }
        lastSlot = top;
        off += len;
    }
    return true;
}

bool inlineGlobalCall(Emit *e, ObjFunction *caller, ObjClosure *callee,
                             unsigned argc, uint32_t callOff, int calleeReg) {
    return inlineCallAt(e, caller, callee, argc, callOff, calleeReg, false);
}

/* A method's body where the call is, through the same walker a global
 * function's is inlined with: slot 0 is the receiver entry, already on the
 * operand stack, rather than a callee entry that holds no register. The
 * receiver's class must be pinned (the caller resolved `method` against it),
 * and its field reads are emitted against that class with a tag guard each.
 *
 * Wider than inlineMethodWalk, which it backs up: that one speaks field
 * reads off parameters and + - * only, so `(x + self.k) % M` -- a constant,
 * a modulo, a field read off a copy of `self` -- went through a full direct
 * call, prologue, stack check, root fill and verdict, for four instructions
 * of work. */
bool inlineMethodCall(Emit *e, ObjFunction *caller, ObjClosure *method,
                      unsigned argc, uint32_t callOff) {
    if (!jitInlineMethodsOn()) return false;
    if (e->depth < argc + 1u) return false;
    unsigned ridx = e->depth - argc - 1u;
    if (e->stack[ridx] != SLOT_INST || e->stackClass[ridx] == NULL) return false;
    if (method->fn->upvalueCount != 0) return false;
    return inlineCallAt(e, caller, method, argc, callOff, -1, true);
}

static bool inlineCallAt(Emit *e, ObjFunction *caller, ObjClosure *callee,
                         unsigned argc, uint32_t callOff, int calleeReg,
                         bool method) {
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
    bool constructs = false;
    if (!inlinableBody(callee, argc, &maxSlot, &readsUpvalue, &constructs)) {
        /* Not straight-line: a direct call to a small body with loops in it
         * may still stand where the call is, with its locals in homes. */
        if (!method && calleeReg < 0) {
            return inlineLoopCall(e, caller, callee, argc, callOff);
        }
        return false;
    }
    /* A body that calls out cannot live in x0..x8, and a caller whose own
     * values are there (scratchValues) has no other bank to give it. */
    if (constructs && e->scratchValues) return false;
    if (readsUpvalue && calleeReg < 0) return false;
    if (getenv("JAI_JIT_WHY")) {
        fprintf(stderr, "[jit] inlining %s\n",
                cfn->name ? cfn->name->chars : "<anon>");
    }

    /* Every argument has to be in a register, since that is where the body
     * will read its parameters from -- and for a method, the receiver too. */
    for (unsigned i = 0; i < argc; i++) {
        if (!holdsRegister(e->stack[cidx + 1u + i])) return false;
    }
    if (method && !holdsRegister(e->stack[cidx])) return false;

    int savedSlot[JIT_MAX_SLOTS + 1];
    memcpy(savedSlot, e->inlSlot, sizeof savedSlot);
    for (unsigned i = 0; i <= JIT_MAX_SLOTS; i++) e->inlSlot[i] = -1;
    for (unsigned i = 0; i < argc; i++) e->inlSlot[1u + i] = (int)(cidx + 1u + i);
    /* A plain function's slot 0 is its own closure, which no opcode the
     * whitelist admits can read; a method's is the receiver. */
    if (method) e->inlSlot[0] = (int)cidx;
    /* A field read that fails half-way through an inlined body cannot be
     * taken back, and the price of that is the whole compile retried with
     * inlining OFF -- every other call this body inlined goes with it. So
     * each one is proved readable before a word is emitted. */
    if (!inlineFieldsReadable(e, callee)) {
        memcpy(e->inlSlot, savedSlot, sizeof savedSlot);
        return false;
    }

    /* No noteScratchClobber here. An inlined body cannot call -- inlinableBody
     * admits nothing that does -- so it destroys x0..x8 only by USING them,
     * which is not a clobber but an allocation: when the caller already owns
     * that bank the two share one numbering (see inlineOwnBank), and when it
     * does not, the inlined entries have x0..x8 to themselves as before.
     * Anything inside that really does call still reaches noteScratchClobber
     * on its own, and under scratchValues that declines the compile. */
    e->inlining     = true;
    e->inlShared    = constructs;
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
        e->inlShared = false;
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
    e->inlShared = false;
    memcpy(e->inlSlot, savedSlot, sizeof savedSlot);

    if (!pushValue(e, kres, rshape, rcls)) { e->failed = true; return false; }
    if (rfp) {
        unsigned dd = fpRegAt(e, e->valueDepth - 1);
        if (dd != rfpReg) emit(e, jaiA64FmovDD(dd, rfpReg));
        fpClaim(e, e->valueDepth - 1);
    } else {
        unsigned dst = pushReg(e) - 1;
        if (dst != rres && e->inlBorrowResult && jitInlineBorrow()) {
            /* Consumed by the store that follows before anything can write
             * the inlined bank again; see Emit::inlBorrowResult. */
            xBorrowLocal(e, e->valueDepth - 1, rres);
        } else if (dst != rres) {
            emit(e, jaiA64MovX(dst, rres));
        }
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


/* ------------------------------------------------------------------ */
/* Loop-bearing callees                                                */
/* ------------------------------------------------------------------ */

/* JAITHON_JIT_INLINE_LOOPS=0 keeps every callee with a branch or a loop in
 * it behind a real call, as before. */
static bool jitInlineLoopsOn(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("JAITHON_JIT_INLINE_LOOPS");
        on = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return on != 0;
}

/* Bounds on code growth: the arena is shared, and a full one changes which
 * tier compiles everything after it. A callee this small with a loop in it
 * is the shape the call overhead dominates (queens' `safe` is 78 bytes);
 * anything bigger pays its frame back over its own loop. */
#define JIT_INLINE_LOOP_MAX_CODE  160
#define JIT_INLINE_LOOP_MAX_SITES 4
/* Highest home slot: the deopt stub's skip mask is one bit per local below
 * 64, and every per-slot table is JIT_MAX_SLOTS + 1 wide. */
#define JIT_INLINE_LOOP_MAX_SLOT  62u

/* The opcodes a loop-bearing body may hold. Nothing that stores to the heap
 * and nothing that calls: a guard anywhere in the body resumes at the
 * caller's OP_CALL and runs the whole call again in the interpreter, which
 * is sound only if the first, partial run left nothing behind -- and the
 * body's locals live in homes no root fill names, so nothing in it may
 * collect either (noteScratchClobber refuses any call that slips through).
 * Writes to its OWN locals are fine: they are renumbered into homes the
 * interpreter never sees. */
static bool inlinableLoopBody(ObjClosure *callee, unsigned argc,
                              unsigned *maxSlotOut) {
    const ObjFunction *cfn = callee->fn;
    const Chunk *c = &cfn->chunk;
    if (cfn->arity != argc || cfn->defaultCount != 0) return false;
    if (cfn->flags & (FN_VARIADIC | FN_KWREST | FN_INIT)) return false;
    if (cfn->upvalueCount != 0 || cfn->exceptionCount > 0) return false;
    if (c->count <= 0 || c->count > JIT_INLINE_LOOP_MAX_CODE) return false;
    if (c->code[c->count - 1] != OP_RETURN) return false;
    unsigned maxSlot = argc;
    bool branches = false;
    unsigned natives = 0;     /* `float`/`int` read and not yet called */
    for (int off = 0; off < c->count;) {
        uint8_t op = c->code[off];
        int len = instructionLength(c, off);
        if (len <= 0 || off + len > c->count) return false;
        unsigned s1 = 0, s2 = 0;
        switch (op) {
        /* `float(i)` and `int(x)`, the two builtins the tier emits as one
         * instruction rather than a call, exactly as the straight-line
         * inliner admits them; the caller's module-version check retires
         * the binding (the callee's module is the caller's). */
        case OP_GET_GLOBAL: {
            Value nv;
            ObjNative *nat = globalNative(callee, jaiReadU24(c->code + off + 1),
                                          &nv);
            if (nat == NULL || nat->name == NULL) return false;
            if (strcmp(nat->name->chars, "float") != 0 &&
                strcmp(nat->name->chars, "int") != 0) {
                return false;
            }
            natives++;
            break;
        }
        case OP_CALL:
            if (c->code[off + 1] != 1 || natives == 0) return false;
            natives--;
            break;
        case OP_GET_LOCAL: case OP_SET_LOCAL: case OP_BIND:
        case OP_INC_LOCAL: case OP_ADD_INT_CONST: case OP_SUB_INT_CONST:
        case OP_MUL_INT_CONST: case OP_CMP_LOCAL_CONST_LT:
        case OP_ADD_BIND: case OP_SUB_BIND: case OP_MUL_BIND:
            s1 = jaiReadU16(c->code + off + 1);
            if (s1 == 0) return false;      /* the closure itself */
            break;
        case OP_GET_LOCAL2: case OP_ADD_LOCALS:
            s1 = jaiReadU16(c->code + off + 1);
            s2 = jaiReadU16(c->code + off + 3);
            if (s1 == 0 || s2 == 0) return false;
            break;
        case OP_JUMP_IF_CMP_LOCAL_K:
            s1 = jaiReadU16(c->code + off + 2);
            if (s1 == 0) return false;
            branches = true;
            break;
        /* `for i in a..b`: the counter and the end are hidden LOCALS, not
         * an iterator on the stack, so they take homes like any other. */
        case OP_ITER_RANGE:
            s1 = jaiReadU16(c->code + off + 2);
            s2 = jaiReadU16(c->code + off + 4);
            if (s1 == 0 || s2 == 0) return false;
            break;
        case OP_FOR_RANGE_BIND: {
            unsigned s3 = jaiReadU16(c->code + off + 7);
            s1 = jaiReadU16(c->code + off + 3);
            s2 = jaiReadU16(c->code + off + 5);
            if (s1 == 0 || s2 == 0 || s3 == 0) return false;
            if (s3 > maxSlot) maxSlot = s3;
            branches = true;
            break;
        }
        case OP_JUMP: case OP_JUMP_IF_FALSE: case OP_JUMP_IF_TRUE:
        case OP_JUMP_IF_CMP_FALSE: case OP_LOOP:
            branches = true;
            break;
        case OP_GET_INDEX:
        case OP_CONST: case OP_INT: case OP_TRUE: case OP_FALSE:
        case OP_POP:
        case OP_ADD: case OP_SUB: case OP_MUL: case OP_DIV:
        case OP_ADD_WRAP: case OP_SUB_WRAP: case OP_MUL_WRAP:
        case OP_TYPE_GUARD: case OP_TO_FLOAT:
        case OP_FLOORDIV: case OP_MOD: case OP_MOD_INT_CONST: case OP_NEG:
        case OP_BAND: case OP_BOR: case OP_BXOR:
        case OP_SHL: case OP_SHR: case OP_BNOT:
        case OP_EQ: case OP_NE: case OP_LT: case OP_LE: case OP_GT: case OP_GE:
        case OP_NOT:
        case OP_RETURN:
            break;
        default:
            if (getenv("JAI_JIT_WHY")) {
                fprintf(stderr, "[jit] loop inline of %s refused at %s\n",
                        cfn->name ? cfn->name->chars : "<anon>",
                        jaiOpName((OpCode)op));
            }
            return false;
        }
        if (s1 > maxSlot) maxSlot = s1;
        if (s2 > maxSlot) maxSlot = s2;
        off += len;
    }
    if (!branches) return false;
    *maxSlotOut = maxSlot;
    return true;
}

static void putU16(uint8_t *p, unsigned v) {
    p[0] = (uint8_t)(v & 0xffu);
    p[1] = (uint8_t)(v >> 8);
}

/* The callee's code with every slot operand moved up by `hb`, so the
 * ordinary local arms address its homes. Offsets are unchanged. */
static uint8_t *renumberedCode(const Chunk *c, unsigned hb) {
    uint8_t *buf = (uint8_t *)malloc((size_t)c->count);
    if (buf == NULL) return NULL;
    memcpy(buf, c->code, (size_t)c->count);
    for (int off = 0; off < c->count;) {
        uint8_t op = c->code[off];
        int len = instructionLength(c, off);
        int ats[3] = { -1, -1, -1 };
        switch (op) {
        case OP_GET_LOCAL: case OP_SET_LOCAL: case OP_BIND:
        case OP_INC_LOCAL: case OP_ADD_INT_CONST: case OP_SUB_INT_CONST:
        case OP_MUL_INT_CONST: case OP_CMP_LOCAL_CONST_LT:
        case OP_ADD_BIND: case OP_SUB_BIND: case OP_MUL_BIND:
            ats[0] = off + 1; break;
        case OP_GET_LOCAL2: case OP_ADD_LOCALS:
            ats[0] = off + 1; ats[1] = off + 3; break;
        case OP_JUMP_IF_CMP_LOCAL_K:
            ats[0] = off + 2; break;
        case OP_ITER_RANGE:
            ats[0] = off + 2; ats[1] = off + 4; break;
        case OP_FOR_RANGE_BIND:
            ats[0] = off + 3; ats[1] = off + 5; ats[2] = off + 7; break;
        default: break;
        }
        for (int k = 0; k < 3; k++) {
            if (ats[k] < 0) continue;
            unsigned v = jaiReadU16(c->code + ats[k]) + hb;
            putU16(buf + ats[k], v);
            if (jaiReadU16(buf + ats[k]) != v) { free(buf); return NULL; }
        }
        off += len;
    }
    return buf;
}

/* Patches one branch word the way the final fixup pass does. */
static void patchBranch(Emit *e, const Fixup *f, int target) {
    int rel = target - f->instIndex;
    uint32_t word = e->code[f->instIndex];
    if ((word & 0xfc000000u) == 0x94000000u) {
        e->code[f->instIndex] = jaiA64Bl(rel);
    } else if (f->conditional && jaiA64IsCbz(word)) {
        e->code[f->instIndex] = jaiA64CbzRetarget(word, rel);
    } else if (f->conditional) {
        e->code[f->instIndex] = jaiA64BCond(word & 0xfu, rel);
    } else {
        e->code[f->instIndex] = jaiA64B(rel);
    }
}

/* An OP_RETURN of a loop-bearing inline: the result goes to the one entry
 * every return agrees on -- just above the callee and its arguments, which
 * stay on the stack for the guards' sake -- and all but the last branch to
 * the exit. Popped again afterwards, so the walk past it starts from the
 * callee's own empty stack, which is what the branch landing there holds. */
bool inlineLoopReturn(Emit *e, bool last) {
    if (e->depth < e->inlDepth + 1u ||
        !holdsRegister(e->stack[e->depth - 1]) ||
        (last && e->depth != e->inlDepth + 1u)) {
        e->whyNot = "an inlined loop returning with more than its result";
        return false;
    }
    SlotKind k = e->stack[e->depth - 1];
    uint32_t shape = e->stackShape[e->depth - 1];
    ObjClass *cls = e->stackClass[e->depth - 1];
    if (!e->inlRetSet) {
        e->inlRetSet = true;
        e->inlRetKind = k;
        e->inlRetShape = shape;
        e->inlRetClass = cls;
    } else if (k != e->inlRetKind || shape != e->inlRetShape ||
               cls != e->inlRetClass) {
        e->whyNot = "an inlined loop's returns disagree about the result";
        return false;
    }
    /* Every entry in its own X register, as a branch wants them. */
    fpSyncAll(e);
    settleAll(e);
    if (last) return true;
    /* Whatever else the body still holds (nothing, for every opcode the
     * whitelist admits today) is dead past a return, so the result can go
     * straight into the register the exit reads it from. */
    unsigned rres = valueXReg(e, e->valueDepth - 1);
    unsigned rexit = valueXReg(e, e->inlRetVi);
    if (rres != rexit) emit(e, jaiA64MovX(rexit, rres));
    if (e->fixupCount >= JIT_MAX_FIXUPS) { e->failed = true; return false; }
    e->fixups[e->fixupCount].instIndex    = (int)e->count;
    e->fixups[e->fixupCount].targetOffset = e->inlExitOff;
    e->fixups[e->fixupCount].conditional  = false;
    e->fixups[e->fixupCount].depth        = -1;
    e->fixupCount++;
    emit(e, jaiA64B(0));
    /* The walk carries on from whatever branch lands on the next offset,
     * which holds the body's stack without this result. */
    unsigned r;
    return popValueRaw(e, &r, NULL);
}

/* Inlines a small body that branches and loops. Its locals are renumbered
 * into slots of the caller's frame above the interpreter's window (homes),
 * so the register planner gives them registers or frame slots as it does
 * the caller's own, and the walk uses the ordinary local arms. Its branches
 * resolve against its own offset map before the caller's fixups come back,
 * so the two numberings never meet in one table. */
bool inlineLoopCall(Emit *e, ObjFunction *caller, ObjClosure *callee,
                    unsigned argc, uint32_t callOff) {
    if (!jitInlineLoopsOn()) return false;
    if (e->osr || e->inlHomeLo == 0 || e->mapKernel) return false;
    if (e->inlLoopCount >= JIT_INLINE_LOOP_MAX_SITES) return false;
    ObjFunction *cfn = callee->fn;
    if (cfn == caller) return false;           /* recursion */
    unsigned maxSlot = 0;
    if (!inlinableLoopBody(callee, argc, &maxSlot)) return false;
    unsigned hb = e->inlHomeNext != 0 ? e->inlHomeNext : e->inlHomeLo;
    if (hb + maxSlot > JIT_INLINE_LOOP_MAX_SLOT) return false;
    unsigned need = hb + maxSlot + 1u;         /* one past the highest home */
    if (e->measuring) {
        if (need > e->base + e->locals) e->locals = need - e->base;
    } else if (need > e->base + e->locals) {
        return false;      /* the measuring pass did not inline this one */
    }
    if (e->depth < argc + 1u) return false;
    unsigned cidx = e->depth - argc - 1u;
    for (unsigned i = 0; i < argc; i++) {
        if (!holdsRegister(e->stack[cidx + 1u + i])) return false;
    }

    int count = cfn->chunk.count;
    uint8_t *code = renumberedCode(&cfn->chunk, hb);
    if (code == NULL) return false;
    int *cmap = (int *)malloc(sizeof(int) * (size_t)(count + 1));
    int64_t *cdepths = (int64_t *)malloc(sizeof(int64_t) * (size_t)(count + 1));
    uint8_t *cloop = (uint8_t *)calloc((size_t)count + 1u, 1);
    Fixup *saved = (Fixup *)malloc(sizeof(Fixup) * (e->fixupCount + 1u));
    if (cmap == NULL || cdepths == NULL || cloop == NULL || saved == NULL) {
        free(code); free(cmap); free(cdepths); free(cloop); free(saved);
        return false;
    }
    if (getenv("JAI_JIT_WHY")) {
        fprintf(stderr, "[jit] inlining loop %s\n",
                cfn->name ? cfn->name->chars : "<anon>");
    }
    for (int i = 0; i <= count; i++) { cmap[i] = -1; cdepths[i] = -1; }

    /* Loop nesting for the slot ranking, on top of the call site's own. */
    unsigned d0 = 0;
    if (e->loopDepth != NULL && callOff < e->loopDepthCount) {
        d0 = e->loopDepth[callOff];
    }
    for (int i = 0; i <= count; i++) cloop[i] = (uint8_t)d0;
    for (int off = 0; off < count;) {
        int len = instructionLength(&cfn->chunk, off);
        if (cfn->chunk.code[off] == OP_LOOP) {
            int top = off + 3 + jaiReadI16(cfn->chunk.code + off + 1);
            for (int x = top; x <= off && x >= 0; x++) {
                if (cloop[x] < 250) cloop[x]++;
            }
        }
        off += len;
    }

    /* Every entry in its own X register, so each argument can be copied
     * into its parameter's home and still stand where the guards need it. */
    fpSyncAll(e);
    settleAll(e);
    for (unsigned s = hb; s <= hb + maxSlot; s++) {
        e->localKind[s]     = SLOT_INT;
        e->localShape[s]    = 0;
        e->localClass[s]    = NULL;
        e->localTyped[s]    = false;
        e->localSeen[s]     = NULL_VAL;
        e->localElemDecl[s] = 0;
        e->localObjType[s]  = 0;
    }
    e->inlHomeNext  = need;
    e->inlLoopCount++;

    e->inlining      = true;
    e->inlHomes      = true;
    e->inlShared     = false;
    e->inlDepth      = cidx + 1u + argc;
    e->inlPinned     = 0;
    e->inlValueBase  = e->valueDepth;
    e->inlIp         = callOff;
    e->inlClosureReg = -1;
    e->inlExitOff    = (uint32_t)count;
    e->inlRetSet     = false;
    e->inlRetVi      = e->valueDepth;   /* the result's value index */

    bool ok = true;
    unsigned vi = 0;
    for (unsigned idx = 0; idx < cidx + 1u + argc && ok; idx++) {
        if (!holdsRegister(e->stack[idx])) continue;
        unsigned v = vi++;
        if (idx <= cidx) continue;
        unsigned slot = hb + (idx - cidx);
        if (!localInRange(e, slot) ||
            !adoptLocalKindSeen(e, slot, e->stack[idx], e->stackShape[idx],
                                e->stackClass[idx], e->stackSeen[idx])) {
            ok = false;
            break;
        }
        e->localElemDecl[slot] = e->stackElemDecl[idx];
        e->localObjType[slot]  = e->stackObjType[idx];
        localOut(e, slot, xHeldIn(e, v));
    }

    /* The walk sees only its own fixups. */
    unsigned nCaller = e->fixupCount;
    memcpy(saved, e->fixups, sizeof(Fixup) * nCaller);
    e->fixupCount = 0;
    unsigned fp0 = e->fpCarryCount, he0 = e->homeEarlyCount;
    unsigned dc0 = e->deferCarryCount;
    int *savedMap = e->offsetToInst;
    int64_t *savedDepths = e->offsetToDepth;
    const uint8_t *savedLoop = e->loopDepth;
    unsigned savedLoopCount = e->loopDepthCount;
    uint32_t savedCurOffset = e->curOffset;
    unsigned savedInstDepth = e->instDepth;
    unsigned savedInstValue = e->instValueDepth;
    e->offsetToInst = cmap;
    e->offsetToDepth = cdepths;
    e->loopDepth = cloop;
    e->loopDepthCount = (unsigned)count + 1u;

    ObjFunction ffn = *cfn;
    ffn.chunk.code = code;
    /* The verifier behind the range facts' graph checks every slot operand
     * against the window; the homes are the window now. */
    ffn.maxSlots = (uint16_t)need;
    ObjClosure fcl = *callee;
    fcl.fn = &ffn;
    /* The range-fact cache keys on the function pointer, and `ffn` is a
     * stack copy whose address the next inline may reuse for another body. */
    jitRangeReset();
    if (ok) ok = compileBody(e, &fcl) && !e->failed;
    jitRangeReset();
    unsigned exitInst = e->count;
    if (ok) {
        cmap[count] = (int)exitInst;
        cdepths[count] = stackSignature(e);
    }

    /* Resolve this body's own branches; keep the sentinels for the end. */
    unsigned kept = 0;
    if (ok) {
        uint8_t *landed = (uint8_t *)calloc((size_t)count + 1u, 1);
        if (landed == NULL) ok = false;
        for (unsigned f = 0; ok && f < e->fixupCount; f++) {
            uint32_t t = e->fixups[f].targetOffset;
            if (t <= (uint32_t)count) landed[t] = 1;
        }
        for (unsigned i = fp0; ok && i < e->fpCarryCount; i++) {
            if (e->fpCarry[i] <= (uint32_t)count && landed[e->fpCarry[i]]) ok = false;
        }
        for (unsigned i = he0; ok && i < e->homeEarlyCount; i++) {
            if (e->homeEarly[i] <= (uint32_t)count && landed[e->homeEarly[i]]) ok = false;
        }
        for (unsigned i = dc0; ok && i < e->deferCarryCount; i++) {
            if (e->deferCarry[i] <= (uint32_t)count && landed[e->deferCarry[i]]) ok = false;
        }
        free(landed);
        if (!ok) e->whyNot = "a branch inside an inlined loop lands mid-expression";
        for (unsigned f = 0; ok && f < e->fixupCount; f++) {
            Fixup fx = e->fixups[f];
            if (fx.targetOffset > (uint32_t)count) {
                e->fixups[kept++] = fx;
                continue;
            }
            int target = cmap[fx.targetOffset];
            if (target < 0 ||
                (fx.depth >= 0 && cdepths[fx.targetOffset] != fx.depth)) {
                e->whyNot = "an inlined loop's branch has no consistent landing";
                ok = false;
                break;
            }
            patchBranch(e, &fx, target);
        }
        if (ok && nCaller + kept > JIT_MAX_FIXUPS) ok = false;
    }
    if (ok) {
        memmove(&e->fixups[nCaller], &e->fixups[0], sizeof(Fixup) * kept);
        memcpy(&e->fixups[0], saved, sizeof(Fixup) * nCaller);
        e->fixupCount = nCaller + kept;
    } else {
        /* The compile is abandoned; the caller's table only has to be whole
         * enough for nothing to read past it. */
        memcpy(&e->fixups[0], saved, sizeof(Fixup) * nCaller);
        e->fixupCount = nCaller;
    }
    e->fpCarryCount = fp0;
    e->homeEarlyCount = he0;
    e->deferCarryCount = dc0;
    e->offsetToInst = savedMap;
    e->offsetToDepth = savedDepths;
    e->loopDepth = savedLoop;
    e->loopDepthCount = savedLoopCount;
    e->curOffset = savedCurOffset;
    e->instDepth = savedInstDepth;
    e->instValueDepth = savedInstValue;
    /* The cache keys on the code pointer, which the next malloc may reuse
     * for a different body. */
    jitBranchTargetsReset();
    free(code); free(cmap); free(cdepths); free(cloop); free(saved);

    if (ok && (e->depth != e->inlDepth + 1u || !e->inlRetSet ||
               e->count != exitInst)) {
        e->whyNot = "an inlined loop ended without its result";
        ok = false;
    }
    if (!ok) {
        e->inlining = false;
        e->inlHomes = false;
        gInlineFailed = true;
        e->failed = true;
        return false;
    }

    unsigned rres;
    SlotKind kres;
    uint32_t rshape = e->stackShape[e->depth - 1];
    ObjClass *rcls = e->stackClass[e->depth - 1];
    if (!popValue(e, &rres, &kres)) { e->failed = true; return false; }
    while (e->depth > cidx) {
        if (holdsRegister(e->stack[e->depth - 1])) {
            unsigned r;
            if (!popValueRaw(e, &r, NULL)) { e->failed = true; return false; }
        } else {
            e->depth--;
        }
    }
    e->inlining = false;
    e->inlHomes = false;
    if (!pushValue(e, kres, rshape, rcls)) { e->failed = true; return false; }
    unsigned dst = pushReg(e) - 1;
    if (dst != rres) emit(e, jaiA64MovX(dst, rres));
    e->inlined = true;
    return true;
}

#endif /* __aarch64__ || __arm64__ */
