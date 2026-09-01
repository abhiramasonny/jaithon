/* jit_body_return.c -- the three ways a compiled body leaves, and the rule that
 * merges what they each said it returns. */

#include "vm/jit/jit_arm64.h"
#include "vm/jit/jit_field_read.h"

#include <stddef.h>
#include <stdint.h>
#include "vm/jit/jit_internal.h"

#if (defined(__aarch64__) || defined(__arm64__))

/* `build` in binary_trees returns `null` on one path and `Node(..)` on another -- an instance merged
 * with a maybe-instance becomes a maybe-instance (shape survives only if both sides agree). Written once before and reverted when it made binary_trees 11x slower -- not this merge's fault: at the time a self-call rooted nothing and emitRootFill's operand-stack-to-register mapping was wrong, and `build` was the first body to hold a fresh allocation across an allocating self-call. Both bugs are now fixed. */
bool mergeReturnKind(Emit *e, SlotKind k, uint32_t shape) {
    if (!e->sawReturn) {
        e->sawReturn = true; e->returnKind = k; e->returnShape = shape;
        return true;
    }
    if (e->returnKind == k) {
        if (e->returnShape != shape) e->returnShape = 0;
        return true;
    }
    bool nullable = (e->returnKind == SLOT_INST && k == SLOT_MAYBE_INST) ||
                    (e->returnKind == SLOT_MAYBE_INST && k == SLOT_INST);
    /* Named, because bare this was the whole of what a census said about
     * OP_RETURN: which two kinds a body cannot agree on is the entire question,
     * and an instance meeting a nullable instance is already merged above. */
    if (!nullable) {
        return subWhy(e, "a body returning both %s and %s",
                      slotKindName(e->returnKind), slotKindName(k));
    }
    if (e->returnShape != shape) e->returnShape = 0;
    e->returnKind = SLOT_MAYBE_INST;
    return true;
}

bool emitReturn(Emit *e, int *offp) {
    int off = *offp;
    do {
        /* Same OSR resume-offset hazard as OP_RETURN_NULL -- see there. */
        if (e->osr) {
            e->whyNot = "a return inside an OSR loop";
            return false;
        }
        uint32_t rsh = e->depth > 0 ? e->stackShape[e->depth - 1] : 0;
        unsigned r;
        SlotKind k;
        if (e->depth == 0) return subWhy(e, "a return with an empty stack");
        if (!popValue(e, &r, &k)) {
            return subWhy(e, "returning a %s, which holds no register",
                          slotKindName(e->stack[e->depth - 1]));
        }
        /* One return kind per function: the entry point rebuilds a Value
         * from it, and it cannot rebuild two. */
        if (!mergeReturnKind(e, k, rsh)) return false;
        emit(e, jaiA64MovX(0, r));
        emitEpilogue(e, 0);
        off += 1;
        break;
    } while (0);
    *offp = off;
    return true;
}

bool emitReturnNull(Emit *e, ObjFunction *fn, int *offp) {
    int off = *offp;
    do {
        /* An OSR form's x0 is the resume bytecode offset (see jaiJitEnterOsr's `*resumeAt = at`), but this
         * return sequence is the function tier's and leaves the VALUE in x0 -- a `return` inside a compiled loop once handed the interpreter an int/pointer as an instruction offset instead, miscompiling `for i in 0..n { if .. { return x } }` (14/20 wrong in released builds, 10/10 under --gc-stress). */
        if (e->osr) {
            e->whyNot = "a return inside an OSR loop";
            return false;
        }
        if ((fn->flags & FN_INIT) == 0) {
            /* A function with nothing to return. No register carries the
             * answer; the entry point builds a null from the kind alone. */
            if (e->sawReturn && e->returnKind != SLOT_NULL) return false;
            e->sawReturn  = true;
            e->returnKind = SLOT_NULL;
            emit(e, jaiA64MovzX(0, 0, 0));
            emitEpilogue(e, 0);
            off += 1;
            break;
        }
        /* An initializer yields the object it initialised, which is what
         * makes `Point(1, 2)` an expression. */
        if (!localInRange(e, 0) || e->localKind[0] != SLOT_INST) return false;
        e->usesSlot0 = true;
        if (e->sawReturn && e->returnKind != SLOT_INST) return false;
        e->sawReturn  = true;
        e->returnKind = SLOT_INST;
        e->returnShape = e->localShape[0];
        {
            unsigned src = localIn(e, 0, 0);
            if (src != 0) emit(e, jaiA64MovX(0, src));
        }
        emitEpilogue(e, 0);
        off += 1;
        break;
    } while (0);
    *offp = off;
    return true;
}

bool emitPopReturnNull(Emit *e, ObjFunction *fn, int *offp) {
    int off = *offp;
    do {
        /* POP's half: forget a deferred/borrowed entry rather than settle
         * it, same as plain OP_POP -- nothing reads a value being thrown
         * away. */
        unsigned r;
        if (e->depth > 0 && holdsRegister(e->stack[e->depth - 1]) &&
            e->valueDepth > 0) {
            unsigned idx = e->valueDepth - 1;
            e->kPend   &= ~(1u << idx);
            e->xBorrow &= ~(1u << idx);
        }
        if (!popValue(e, &r, NULL)) return false;

        /* RETURN_NULL's half, unchanged. */
        if (e->osr) {
            e->whyNot = "a return inside an OSR loop";
            return false;
        }
        if ((fn->flags & FN_INIT) == 0) {
            if (e->sawReturn && e->returnKind != SLOT_NULL) return false;
            e->sawReturn  = true;
            e->returnKind = SLOT_NULL;
            emit(e, jaiA64MovzX(0, 0, 0));
            emitEpilogue(e, 0);
            off += 1;
            break;
        }
        if (!localInRange(e, 0) || e->localKind[0] != SLOT_INST) return false;
        e->usesSlot0 = true;
        if (e->sawReturn && e->returnKind != SLOT_INST) return false;
        e->sawReturn  = true;
        e->returnKind = SLOT_INST;
        e->returnShape = e->localShape[0];
        {
            unsigned src = localIn(e, 0, 0);
            if (src != 0) emit(e, jaiA64MovX(0, src));
        }
        emitEpilogue(e, 0);
        off += 1;
        break;
    } while (0);
    *offp = off;
    return true;
}

#endif /* __aarch64__ || __arm64__ */
