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
/* The kinds a SLOT_DYNAMIC body may return from any one site: exactly those
 * emitTagFor can tag off the register and jitResultOut can rebuild a Value
 * from. SLOT_OPAQUE is the one register-holding kind kept out -- its payload
 * is nothing in particular. SLOT_DYNAMIC itself is admitted so a third
 * disagreeing site joins an already-dynamic body. */
static bool dynamicReturnKind(SlotKind k) {
    switch (k) {
    case SLOT_INT: case SLOT_FLOAT: case SLOT_BOOL: case SLOT_NULL:
    case SLOT_INST: case SLOT_MAYBE_INST: case SLOT_LIST: case SLOT_OBJ:
    case SLOT_MAYBE_OBJ: case SLOT_DYNAMIC:
        return true;
    default:
        return false;
    }
}

/* Leave the body with the value in x0. A SLOT_DYNAMIC body also says WHICH
 * kind of value: emitTagFor reads the tag off the payload for the nullable
 * kinds and names it for the rest, and it goes up in x1 above a zero verdict
 * byte. Every other body leaves x1 as the bare verdict it always was. */
void emitReturnLeave(Emit *e, SlotKind k) {
    if (!e->dynamicReturn) { emitEpilogue(e, 0); return; }
    emitTagFor(e, k, 0, JIT_SCRATCH_A, JIT_SCRATCH_B);
    emit(e, jaiA64LslX(1, JIT_SCRATCH_A, JIT_RET_TAG_SHIFT));
    emitEpilogueKeepX1(e);
}

bool mergeReturnKind(Emit *e, SlotKind k, uint32_t shape) {
    if (!e->sawReturn) {
        e->sawReturn = true; e->returnKind = k; e->returnShape = shape;
        return true;
    }
    if (e->returnKind == k) {
        if (e->returnShape != shape) e->returnShape = 0;
        return true;
    }
    if (e->returnKind == SLOT_DYNAMIC) {
        if (!dynamicReturnKind(k)) {
            return subWhy(e, "a body returning dynamic and also %s",
                          slotKindName(k));
        }
        return true;
    }
    bool nullable = (e->returnKind == SLOT_INST && k == SLOT_MAYBE_INST) ||
                    (e->returnKind == SLOT_MAYBE_INST && k == SLOT_INST);
    /* `-> OpKind?` returns an enum member on one edge and null on the other:
     * SLOT_OBJ meeting SLOT_MAYBE_INST, which nothing above merges. Neither
     * side's promise survives -- the object is not an instance of the
     * nullable side's shape, and the nullable side is not non-null -- so the
     * join is the weaker SLOT_MAYBE_OBJ, whose tag emitTagFor already reads
     * off the payload the same way. Shape is dropped with it. */
    bool widerNullable =
        jitMaybeObjOn() &&
        ((e->returnKind == SLOT_OBJ &&
          (k == SLOT_MAYBE_INST || k == SLOT_NULL || k == SLOT_MAYBE_OBJ)) ||
         (k == SLOT_OBJ &&
          (e->returnKind == SLOT_MAYBE_INST || e->returnKind == SLOT_NULL ||
           e->returnKind == SLOT_MAYBE_OBJ)) ||
         (e->returnKind == SLOT_MAYBE_OBJ &&
          (k == SLOT_INST || k == SLOT_MAYBE_INST || k == SLOT_NULL)) ||
         (k == SLOT_MAYBE_OBJ &&
          (e->returnKind == SLOT_INST || e->returnKind == SLOT_MAYBE_INST ||
           e->returnKind == SLOT_NULL)));
    if (widerNullable) {
        e->returnShape = 0;
        e->returnKind = SLOT_MAYBE_OBJ;
        return true;
    }
    /* Two kinds nothing above can join -- `-1` on one edge and `[]` on
     * another. The body still returns exactly one of them per call, and each
     * site knows which: under JAITHON_JIT_DYNAMIC_RETURN the join is
     * SLOT_DYNAMIC and every return site hands its own tag up in x1
     * (emitReturnLeave), for jitResultOut to rebuild the Value from. Only
     * kinds emitTagFor can tag and jitResultOut can rebuild are admitted; a
     * self-call reads x1 as a bare verdict and x0 as one fixed kind, so a body
     * that recurses is refused here rather than miscompiled. */
    if (!nullable && jitDynamicReturn() && !e->osr &&
        dynamicReturnKind(e->returnKind) && dynamicReturnKind(k)) {
        if (e->hasSelfCall || e->selfSlowCount > 0) {
            return subWhy(e, "a body returning both %s and %s that also "
                             "calls itself",
                          slotKindName(e->returnKind), slotKindName(k));
        }
        if (!e->measuring && !e->dynamicReturn) {
            /* The real pass met a disagreement the measuring pass did not,
             * so the sites already emitted carry no tag. Cannot be re-tagged
             * after the fact; decline rather than guess. */
            return subWhy(e, "a body returning both %s and %s, seen only "
                             "in the second pass",
                          slotKindName(e->returnKind), slotKindName(k));
        }
        e->returnShape = 0;
        e->returnKind = SLOT_DYNAMIC;
        return true;
    }
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

/* OP_RETURN_NULL's half of the merge. It used to refuse bare, with no reason
 * named, whenever any earlier return was not null; now it is one more site
 * for mergeReturnKind to join -- which is what lets `-> any` bodies that end
 * in a bare `return` reach SLOT_DYNAMIC. With the switch off the old check
 * stands exactly as it was. */
static bool mergeReturnNull(Emit *e) {
    if (e->sawReturn && e->returnKind != SLOT_NULL) {
        if (!jitDynamicReturn()) return false;
        return mergeReturnKind(e, SLOT_NULL, 0);
    }
    e->sawReturn  = true;
    e->returnKind = SLOT_NULL;
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
        emitReturnLeave(e, k);
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
            if (!mergeReturnNull(e)) return false;
            emit(e, jaiA64MovzX(0, 0, 0));
            emitReturnLeave(e, SLOT_NULL);
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
        emitReturnLeave(e, SLOT_INST);
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
            if (!mergeReturnNull(e)) return false;
            emit(e, jaiA64MovzX(0, 0, 0));
            emitReturnLeave(e, SLOT_NULL);
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
        emitReturnLeave(e, SLOT_INST);
        off += 1;
        break;
    } while (0);
    *offp = off;
    return true;
}

#endif /* __aarch64__ || __arm64__ */
