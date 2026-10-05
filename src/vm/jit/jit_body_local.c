/* jit_body_local.c -- the arms that read and write a named slot: locals, their
 * binding, and the upvalues a closure captured them into. */

#include "vm/jit/jit_arm64.h"
#include "vm/jit/jit_field_read.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include "vm/jit/jit_internal.h"

#if (defined(__aarch64__) || defined(__arm64__))

bool emitGetLocal(Emit *e, const uint8_t *code, int *offp, int stop) {
    int off = *offp;
    do {
        unsigned slot = jaiReadU16(code + off + 1);
        /* Both refusals used to be silent, so the census said only
         * "OP_GET_LOCAL" for two unrelated causes -- one a window the OSR
         * form does not cover, the other a slot whose kind is not known
         * yet. They want different fixes; they should not share a line. */
        /* A sunk local (jit_sink.c) is a reference to fields, not a value,
         * and has no place in the window the probe measured. */
        if (slot <= JIT_MAX_SLOTS && e->sinkOf[slot] != 0) {
            if (!sinkPushRef(e, slot)) return false;
            off += 3;
            break;
        }
        if (!localInRange(e, slot)) {
            e->whyNot = "a local outside the compiled window";
            return false;
        }
        if (e->localKind[slot] == SLOT_OPAQUE) {
            e->whyNot = "a local of no known kind";
            return false;
        }
        if (slot == 0) e->usesSlot0 = true;
        if (!pushValue3(e, e->localKind[slot], e->localShape[slot],
                        e->localClass[slot],
                        seenLocal(e, slot),
                        (int)slot)) {
            return false;
        }
        /* The seed of the index shape: this entry IS this local, offset
         * zero. See Emit::idxKnown. */
        if (e->localKind[slot] == SLOT_INT && slot <= UINT8_MAX) {
            unsigned at = e->valueDepth - 1;
            e->idxKnown |= 1u << at;
            e->idxBase[at] = (uint8_t)slot;
            e->idxOff[at]  = 0;
        }
        if (e->localKind[slot] == SLOT_FLOAT && !e->dynamicLocal[slot] &&
            fpWorthLoading(e, code, off + 3, stop)) {
            unsigned idx = e->valueDepth - 1;
            if (e->slotFpReg[slot] != 0) {
                fpBorrowLocal(e, idx, e->slotFpReg[slot]);
            } else {
                localInFp(e, slot, fpRegAt(e, idx));
                fpClaim(e, idx);
            }
        } else {
            unsigned home = localHomeX(e, slot);
            if (home != 0) {
                /* The copy this used to always emit is the whole cost of reading a local (six of them in `fib`) --
                 * borrowing defers it; if nothing consumes the value before an instruction that can't read a borrow, the settle there emits exactly the same mov, so this never costs more. */
                xBorrowLocal(e, e->valueDepth - 1, home);
                noteLocalBorrowed(e, slot);
            } else {
                unsigned dst = pushReg(e) - 1;
                unsigned src = localIn(e, slot, dst);
                if (src != dst) emit(e, jaiA64MovX(dst, src));
            }
            if (e->localKind[slot] == SLOT_FLOAT && !e->dynamicLocal[slot]) {
                fpSrcNote(e, e->valueDepth - 1, slot);
            }
        }
        off += 3;
        break;
    } while (0);
    *offp = off;
    return true;
}

bool emitGetLocal2(Emit *e, const uint8_t *code, int *offp, int stop) {
    int off = *offp;
    do {
        unsigned a = jaiReadU16(code + off + 1);
        unsigned b = jaiReadU16(code + off + 3);
        /* One or both sunk (jit_sink.c): each read on its own, the plain one
         * through OP_GET_LOCAL's arm with the code after this one behind it
         * for its look-ahead. */
        if (e->sinkCount != 0 &&
            ((a <= JIT_MAX_SLOTS && e->sinkOf[a] != 0) ||
             (b <= JIT_MAX_SLOTS && e->sinkOf[b] != 0))) {
            for (unsigned k = 0; k < 2; k++) {
                unsigned slot = k == 0 ? a : b;
                if (slot <= JIT_MAX_SLOTS && e->sinkOf[slot] != 0) {
                    if (!sinkPushRef(e, slot)) return false;
                    continue;
                }
                uint8_t synth[64];
                int rest = stop - (off + 5);
                if (rest < 0) rest = 0;
                if (rest > (int)sizeof synth - 3) rest = (int)sizeof synth - 3;
                synth[0] = OP_GET_LOCAL;
                synth[1] = (uint8_t)(slot & 0xffu);
                synth[2] = (uint8_t)(slot >> 8);
                memcpy(synth + 3, code + off + 5, (size_t)rest);
                int soff = 0;
                if (!emitGetLocal(e, synth, &soff, 3 + rest)) return false;
            }
            off += 5;
            break;
        }
        /* The second local may guard (if dynamic), and a guard can't be reached with a borrow live -- the
         * deopt stub writes float entries out of fpRegAt, where a borrowed one isn't. So the FIRST local takes a copy instead of a borrow whenever the second is going to guard: `dt * b.vx` in nbody's second loop is exactly this shape (`dt` has a home, `b` is dynamic). */
        bool guardFollows = b <= JIT_MAX_SLOTS && e->dynamicLocal[b];
        for (unsigned k = 0; k < 2; k++) {
            unsigned slot = k == 0 ? a : b;
            /* Named, for the reason given at OP_GET_LOCAL. */
            if (!localInRange(e, slot)) {
                e->whyNot = "a local outside the compiled window";
                return false;
            }
            if (e->localKind[slot] == SLOT_OPAQUE) {
                e->whyNot = "a local of no known kind";
                return false;
            }
            if (slot == 0) e->usesSlot0 = true;
            if (!pushValue3(e, e->localKind[slot], e->localShape[slot],
                            e->localClass[slot],
                            seenLocal(e, slot),
                            (int)slot)) {
                return false;
            }
            /* Same seed as OP_GET_LOCAL, and this is the arm that matters:
             * the emitter fuses `xs[j]` into GET_LOCAL2, so every subscript
             * in a stencil arrives here and nowhere else. */
            if (e->localKind[slot] == SLOT_INT && slot <= UINT8_MAX) {
                unsigned at = e->valueDepth - 1;
                e->idxKnown |= 1u << at;
                e->idxBase[at] = (uint8_t)slot;
                e->idxOff[at]  = 0;
            }
            if (e->localKind[slot] == SLOT_FLOAT &&
                !e->dynamicLocal[slot] &&
                fpWorthLoading(e, code, off + 5, stop)) {
                unsigned idx = e->valueDepth - 1;
                if (e->slotFpReg[slot] != 0 && !(k == 0 && guardFollows)) {
                    fpBorrowLocal(e, idx, e->slotFpReg[slot]);
                } else {
                    localInFp(e, slot, fpRegAt(e, idx));
                    fpClaim(e, idx);
                }
            } else {
                unsigned home = localHomeX(e, slot);
                if (home != 0) {            /* see OP_GET_LOCAL */
                    xBorrowLocal(e, e->valueDepth - 1, home);
                    noteLocalBorrowed(e, slot);
                } else {
                    unsigned dst = pushReg(e) - 1;
                    unsigned src = localIn(e, slot, dst);
                    if (src != dst) emit(e, jaiA64MovX(dst, src));
                }
                if (e->localKind[slot] == SLOT_FLOAT &&
                    !e->dynamicLocal[slot]) {
                    fpSrcNote(e, e->valueDepth - 1, slot);
                }
            }
        }
        off += 5;
        break;
    } while (0);
    *offp = off;
    return true;
}

bool emitSetLocal(Emit *e, const uint8_t *code, int *offp) {
    int off = *offp;
    do {
        /* Assigns without popping: the value stays as the statement's
         * result, which is what the interpreter does. */
        unsigned slot = jaiReadU16(code + off + 1);
        if (!localInRange(e, slot)) return false;
        if (slot == 0) e->usesSlot0 = true;
        if (e->depth == 0 || !holdsRegister(e->stack[e->depth - 1])) return false;
        /* A local keeps one kind for the whole function. Two kinds would
         * mean the reads of it cannot be compiled to one instruction, and
         * the join check works on the operand stack, not on locals. */
        if (!adoptLocalKind(e, slot, e->stack[e->depth - 1],
                            e->stackShape[e->depth - 1],
                            e->stackClass[e->depth - 1])) {
            e->whyNot = kindClash(e, slot);
            return false;
        }
        if (!e->fpOff && !e->dynamicLocal[slot] &&
            e->stack[e->depth - 1] == SLOT_FLOAT &&
            (e->fpLive & (1u << (e->valueDepth - 1)))) {
            localOutFp(e, slot, fpHeldIn(e, e->valueDepth - 1));
        } else {
            localOut(e, slot, xHeldIn(e, e->valueDepth - 1));
        }
        off += 3;
        break;
    } while (0);
    *offp = off;
    return true;
}

bool emitBind(Emit *e, const uint8_t *code, int *offp) {
    int off = *offp;
    do {
        unsigned slot = jaiReadU16(code + off + 1);
        if (!localInRange(e, slot)) return false;
        if (slot == 0) e->usesSlot0 = true;
        if (e->depth == 0 || !holdsRegister(e->stack[e->depth - 1])) return false;
        if (!adoptLocalKindSeen(e, slot, e->stack[e->depth - 1],
                                e->stackShape[e->depth - 1],
                                e->stackClass[e->depth - 1],
                                e->stackSeen[e->depth - 1])) {
            e->whyNot = kindClash(e, slot);
            return false;
        }
        if (e->stackElemDecl[e->depth - 1] != 0) {
            e->localElemDecl[slot] = e->stackElemDecl[e->depth - 1];
        }
        e->localObjType[slot] = e->stackObjType[e->depth - 1];
        if (!e->fpOff && !e->dynamicLocal[slot] &&
            e->stack[e->depth - 1] == SLOT_FLOAT &&
            (e->fpLive & (1u << (e->valueDepth - 1)))) {
            unsigned idx = e->valueDepth - 1;
            unsigned held = fpHeldIn(e, idx);
            unsigned r2; SlotKind k2;
            if (!popValueRaw(e, &r2, &k2)) return false;
            localOutFp(e, slot, held);
            off += 3;
            break;
        }
        unsigned r;
        if (!popValue(e, &r, NULL)) return false;
        localOut(e, slot, r);
        off += 3;
        break;
    } while (0);
    *offp = off;
    return true;
}

/* The hoist register holding upvalue `index` of the closure being inlined,
 * if its loop head loaded it as the kind this read would push, else -1. The
 * read is then a borrow with no guard, which is why compileBody lets a
 * deferred entry live across it. */
static int closUpReg(const Emit *e, ObjClosure *closure, unsigned index,
                     SlotKind *kindOut, Value *seenOut) {
    if (!e->inlining || e->inlClosHoist == 0 ||
        e->inlClosHoist > e->closHoistCount) {
        return -1;
    }
    unsigned h = e->inlClosHoist - 1u;
    Value seen = NULL_VAL;
    if (index < (unsigned)closure->upvalueCount &&
        closure->upvalues[index] != NULL) {
        seen = *closure->upvalues[index]->location;
    }
    SlotKind sk = IS_INT(seen) ? SLOT_INT
                : IS_FLOAT(seen) ? SLOT_FLOAT
                : IS_BOOL(seen) ? SLOT_BOOL : SLOT_OPAQUE;
    for (unsigned u = 0; u < e->closHoist[h].upCount; u++) {
        if (e->closHoist[h].upIdx[u] != index) continue;
        if ((SlotKind)e->closHoist[h].upKind[u] != sk) return -1;
        if (kindOut != NULL) *kindOut = sk;
        if (seenOut != NULL) *seenOut = seen;
        return e->closHoist[h].upReg[u];
    }
    return -1;
}

bool closUpHoisted(const Emit *e, ObjClosure *closure, unsigned index) {
    return jitInlineBorrow() && closUpReg(e, closure, index, NULL, NULL) >= 0;
}

bool emitGetUpvalue(Emit *e, ObjFunction *fn, ObjClosure *closure,
                    const uint8_t *code, int *offp) {
    int off = *offp;
    do {
        unsigned index = code[off + 1];
        if (index >= (unsigned)fn->upvalueCount) return false;
        /* Whose closure. An inlined body's is in the register the call
         * site guarded, not in the caller's own closure register: the
         * caller may have no upvalues at all and still be inlining a body
         * that has them. */
        unsigned creg;
        {
            SlotKind sk;
            Value seen;
            int ur = closUpReg(e, closure, index, &sk, &seen);
            if (ur >= 0) {
                /* Loaded at the loop head (planClosureHoists), and nothing
                 * writes a hoisted register inside its loop. */
                if (!pushValue3(e, sk, 0, NULL, seen, -1)) return false;
                if (jitInlineBorrow()) {
                    xBorrowLocal(e, e->valueDepth - 1, (unsigned)ur);
                } else {
                    emit(e, jaiA64MovX(pushReg(e) - 1, (unsigned)ur));
                }
                *offp = off + 2;
                return true;
            }
        }
        if (e->inlining) {
            if (e->inlClosureReg < 0) return false;
            creg = (unsigned)e->inlClosureReg;
        } else {
            if (!e->usesUpvalues) return false; /* decided before this pass */
            creg = closureReg(e);
        }

        /* closure->upvalues[index]->location, then the Value there. The
         * upvalue may still be open, pointing into the VM stack, so the
         * location is followed rather than assumed closed. */
        emit(e, jaiA64LdrX(JIT_SCRATCH_A, creg,
                           (unsigned)offsetof(ObjClosure, upvalues)));
        emit(e, jaiA64LdrX(JIT_SCRATCH_A, JIT_SCRATCH_A, index * 8u));
        emit(e, jaiA64LdrX(JIT_SCRATCH_A, JIT_SCRATCH_A,
                           (unsigned)offsetof(ObjUpvalue, location)));
        emit(e, jaiA64LdrW(JIT_SCRATCH_B, JIT_SCRATCH_A, 0));

        /* An upvalue's type is whatever the capture put there, so it is
         * read once here and checked on every entry into the loop. */
        Value seen = NULL_VAL;
        ObjClosure *cl = closure;
        if (index < (unsigned)cl->upvalueCount && cl->upvalues[index] != NULL) {
            seen = *cl->upvalues[index]->location;
        }
        SlotKind kind;
        unsigned tag;
        uint32_t seenShape = 0;
        ObjClass *seenClass = NULL;
        if (IS_INT(seen))        { kind = SLOT_INT;   tag = VAL_INT; }
        else if (IS_FLOAT(seen)) { kind = SLOT_FLOAT; tag = VAL_FLOAT; }
        else if (IS_BOOL(seen))  { kind = SLOT_BOOL;  tag = VAL_BOOL; }
        else if (IS_LIST(seen))  { kind = SLOT_LIST;  tag = VAL_OBJ; }
        else if (rawObjValue(seen)) { kind = SLOT_OBJ; tag = VAL_OBJ; }
        else if (IS_INSTANCE(seen) && AS_INSTANCE(seen)->klass != NULL) {
            /* Same reasoning as every other raw-object read site in this
             * tier: nothing about an upvalue makes its capture special,
             * it is read the same way a global or a field is. A closure
             * over a str/list/dict/instance never compiled before this. */
            kind = SLOT_INST; tag = VAL_OBJ;
            seenClass = AS_INSTANCE(seen)->klass;
            seenShape = seenClass->shapeId;
        } else return false;

        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_B, tag));
        branchOnDeopt(e, JAI_A64_NE);
        if (kind == SLOT_LIST) {
            /* "an object" is not "a list": same contract as every other
             * SLOT_LIST arm in this tier. JIT_SCRATCH_A holds the
             * upvalue's location and must survive to the load below. */
            emit(e, jaiA64LdrX(JIT_SCRATCH_D, JIT_SCRATCH_A, 8));
            emit(e, jaiA64LdrW(JIT_SCRATCH_B, JIT_SCRATCH_D,
                               (unsigned)offsetof(Obj, type)));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_B, OBJ_LIST));
            branchOnDeopt(e, JAI_A64_NE);
        } else if (kind == SLOT_INST) {
            emit(e, jaiA64LdrX(JIT_SCRATCH_D, JIT_SCRATCH_A, 8));
            emit(e, jaiA64LdrW(JIT_SCRATCH_B, JIT_SCRATCH_D,
                               (unsigned)offsetof(Obj, type)));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_B, OBJ_INSTANCE));
            branchOnDeopt(e, JAI_A64_NE);
            emit(e, jaiA64LdrX(JIT_SCRATCH_D, JIT_SCRATCH_D,
                               (unsigned)offsetof(ObjInstance, klass)));
            emit(e, jaiA64LdrW(JIT_SCRATCH_D, JIT_SCRATCH_D,
                               (unsigned)offsetof(ObjClass, shapeId)));
            emitConst64(e, JIT_SCRATCH_B, (int64_t)seenShape);
            emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_D, JIT_SCRATCH_B));
            branchOnDeopt(e, JAI_A64_NE);
        }
        if (!pushValue3(e, kind, seenShape, seenClass, seen, -1)) return false;
        if (kind == SLOT_BOOL) {
            emit(e, jaiA64LdrByte(pushReg(e) - 1, JIT_SCRATCH_A, 8));
        } else {
            emit(e, jaiA64LdrX(pushReg(e) - 1, JIT_SCRATCH_A, 8));
        }
        off += 2;
        break;
    } while (0);
    *offp = off;
    return true;
}

#endif /* __aarch64__ || __arm64__ */
