/* jit_body_match.c -- the enum-variant `match` arms of the opcode walk. */

#include "vm/jit/jit_arm64.h"
#include "vm/jit/jit_field_read.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include "vm/jit/jit_internal.h"

#if (defined(__aarch64__) || defined(__arm64__))

/* Whether an arm of the enum-`match` idiom emitted anything this attempt, and
 * whether the attempt has been told not to. The pair exists because arming a
 * body's `match` can turn a PARTIAL compile into an outright decline, which is
 * a regression: walking past OP_MATCH_TYPE_POP reaches arm bodies the tier
 * could not have seen before, and `A => OpKind.Add, _ => null` reaches its
 * OP_RETURN carrying an object on one edge and a maybe-instance on the other --
 * a join this tier declines the whole body for. Twenty bodies in lib/std are
 * that shape.
 *
 * So the compile is retried with the arms off, exactly as gInlineFailed retries
 * with inlining off and for the same reason: whatever the body compiled before
 * this arm existed, it must still compile. Same file-static argument as
 * gInlineFailed -- compilation is not reentrant. */
bool gMatchUsed;
bool gNoMatchArm;

/* May this walk arm the enum-`match` opcodes at all?
 *
 * OSR is excluded because matchMissResume's whole argument rests on
 * chunkDepth, and modelAgreesWithChunk does not hold an OSR model to it -- the
 * window is entered mid-body with the loop's own operand stack. Inlining is
 * excluded because the offsets are the callee's while the fixups are the
 * caller's. Both are outright refusals, not gaps to close later without a
 * measurement: the idiom's own shape is a whole small function. */
static bool matchArmOn(const Emit *e) {
    return !e->osr && !e->inlining && !gNoMatchArm && jitMatchArm();
}

/* `match e { Op.Add => .., Op.Sub => .. }` lowers each alternative to five
 * instructions -- OP_MATCH_TYPE_POP, OP_GET_LOCAL, OP_ENUM_TAG, OP_SWAP_POP,
 * OP_MATCH_CONST_POP (emit.jai's `_test_enum`) -- and none of the four new ones
 * had an arm, so a body built out of them compiled a prefix and deoptimised on
 * every call. They are armed one at a time below; what they share is this file.
 *
 * The enum an OP_MATCH_TYPE_POP names, resolved now and pinned by the module
 * version check at entry. Exactly globalClass's contract and sound for the same
 * reason: jaiValueIsInertGlobal answers false for an ObjEnum, so binding or
 * rebinding one bumps ObjModule::version and retires this compiled form.
 *
 * The constant is a STRING and not the enum itself -- codegen writes the
 * owner's NAME, or "Owner.Variant" when the tag was not visible where the
 * pattern compiled. jaiValueMatchesType resolves it against the running
 * frame's module and then vm.builtins; only the module arm is taken here,
 * because a compiled body pushes no frame and so `topFrame()` inside one is the
 * CALLER's. Resolving against this function's own module is what the
 * interpreter does when this function is the frame, which is the only reading
 * that can be right. A name bound to anything but an enum, and a dotted one
 * (which does not resolve through a module at all), are refused rather than
 * guessed at. */
static ObjEnum *matchEnumType(ObjClosure *closure, uint32_t nameIdx) {
    ObjFunction *fn = closure->fn;
    if (fn->module == NULL) return NULL;
    if (nameIdx >= (uint32_t)fn->chunk.constants.count) return NULL;
    Value k = fn->chunk.constants.data[nameIdx];
    if (!IS_STRING(k)) return NULL;
    ObjString *name = AS_STRING(k);
    if (name->length <= 0) return NULL;
    /* Not strchr: an ObjString carries its length and need not be NUL-terminated. */
    if (memchr(name->chars, '.', (size_t)name->length) != NULL) return NULL;
    Value bound;
    if (!jaiModuleGet(fn->module, name, &bound)) return NULL;
    return IS_ENUM(bound) ? AS_ENUM(bound) : NULL;
}

/* Where a failed `match` test resumes.
 *
 * Every miss edge in the lowering lands on a lone OP_POP that discards the
 * subject -- and the compiled form has not pushed one, because both arms pop
 * their own entry on both edges. So the branch goes PAST that OP_POP.
 *
 * That is not merely tidier, it is the only depth the two arms can both reach
 * the join at. OP_MATCH_TYPE_POP misses holding the SUBJECT and
 * OP_MATCH_CONST_POP misses holding the TAG; the operand-stack signature
 * carries kinds as well as depth, so two edges into one offset carrying an
 * object and an int are a join this tier declines outright. One instruction
 * later the stack is empty on every edge and there is nothing left to disagree
 * about.
 *
 * It is also what lets the alternation shape (`A | B => e`) compile at all.
 * There each failed alternative ends in an unconditional OP_JUMP to the arm
 * body sitting immediately before that OP_POP, so the linear walk arrives at
 * the OP_POP holding an empty stack where the bytecode says one entry --
 * and reconcileAfterUncond only TRUNCATES a model, so it cannot restate the
 * deeper one the miss edges carry. The arm skips the OP_POP instead.
 *
 * Refuses unless the shape is exactly that, which keeps the retarget honest:
 * a forward branch, onto an OP_POP, at a depth the bytecode itself says is one
 * entry going to none. */
static bool matchMissResume(Emit *e, const ObjFunction *fn, int off, int len,
                            int16_t jump, uint32_t *out) {
    /* OSR walks a window, not a body, and modelAgreesWithChunk does not hold
     * the model to chunkDepth there -- so the depth reasoning below has no
     * oracle. An inlined body's offsets are the callee's and mean nothing in
     * the caller's fixup table. */
    if (e->osr || e->inlining) return false;
    if (e->chunkDepth == NULL) return false;
    if (jump <= 0) return false;             /* every miss edge is forward */
    if (e->depth != 1 || e->valueDepth != 1) return false;
    int64_t t = (int64_t)off + len + jump;
    if (t <= 0 || t + 1 >= (int64_t)fn->chunk.count) return false;
    if (t + 1 >= (int64_t)e->chunkDepthCount) return false;
    if (fn->chunk.code[t] != OP_POP) return false;
    if (e->chunkDepth[t] != 1 || e->chunkDepth[t + 1] != 0) return false;
    *out = (uint32_t)(t + 1);
    return true;
}

bool popSkipTarget(const Emit *e, uint32_t at) {
    for (unsigned i = 0; i < e->popSkipCount; i++) {
        if (e->popSkip[i] == at) return true;
    }
    return false;
}

static bool notePopSkip(Emit *e, uint32_t at) {
    if (popSkipTarget(e, at)) return true;
    if (e->popSkipCount >= JIT_MAX_POPSKIP) return false;
    e->popSkip[e->popSkipCount++] = at;
    return true;
}

/* Every load below is a scaled-immediate form, so its offset has to divide by
 * the width it reads: a field that moved to an odd address would otherwise be
 * read from a rounded-down one, silently. Checked here rather than trusted,
 * because the three widths are the whole point -- see the tag read in
 * OP_ENUM_TAG. */
_Static_assert(offsetof(Obj, type) % 4 == 0,
               "Obj::type is read with a scaled 32-bit load");
_Static_assert(offsetof(ObjEnumVal, type) % 8 == 0,
               "ObjEnumVal::type is read with a scaled 64-bit load");
_Static_assert(offsetof(ObjEnumVal, tag) % 2 == 0,
               "ObjEnumVal::tag is read with a scaled 16-bit load");
_Static_assert(sizeof ((ObjEnumVal *)0)->tag == 2,
               "OP_ENUM_TAG's arm emits a halfword load for the tag");

/* Prove `rSubj` holds an ObjEnumVal of `type`, branching to `miss` when it does
 * not -- which is what jaiValueMatchesType's IS_ENUM arm does, spelled out.
 * The pointer compare is against an enum this body has pinned, so nothing but
 * the subject itself can change the answer.
 *
 * VAL_OBJ needs no check: a SLOT_OBJ entry is a heap object by construction,
 * and the Obj::type read below is the same one every other narrowing arm in
 * this file makes before it dereferences. */
static void emitEnumTypeGuard(Emit *e, unsigned rSubj, ObjEnum *type,
                              uint32_t miss) {
    emit(e, jaiA64LdrW(JIT_SCRATCH_A, rSubj, (unsigned)offsetof(Obj, type)));
    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_ENUM_VAL));
    branchTo(e, miss, true, JAI_A64_NE);
    emit(e, jaiA64LdrX(JIT_SCRATCH_A, rSubj,
                       (unsigned)offsetof(ObjEnumVal, type)));
    emitConst64(e, JIT_SCRATCH_B, (int64_t)(uintptr_t)type);
    emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
    branchTo(e, miss, true, JAI_A64_NE);
}

/* `[a b] -> [b]`, with b's register moved down into a's. dropCalleeEntry does
 * the model half of this for an entry that holds no register; here both do, so
 * the value moves as well. Every per-entry claim travels with the entry it is
 * about, for the reason dropCalleeEntry gives. */
static void swapPopEntry(Emit *e) {
    unsigned t = e->depth - 1, u = e->depth - 2;
    e->stack[u]         = e->stack[t];
    e->stackShape[u]    = e->stackShape[t];
    e->stackClass[u]    = e->stackClass[t];
    e->stackSeen[u]     = e->stackSeen[t];
    e->stackLocal[u]    = e->stackLocal[t];
    e->stackAscii[u]    = e->stackAscii[t];
    e->stackUnit[u]     = e->stackUnit[t];
    e->stackNullLit[u]  = e->stackNullLit[t];
    e->stackObjType[u]  = e->stackObjType[t];
    e->stackElem[u]     = e->stackElem[t];
    e->stackElemDecl[u] = e->stackElemDecl[t];
    e->depth--;
    e->valueDepth--;
    e->fpLive   &= ~(1u << e->valueDepth);
    e->fpBorrow &= ~(1u << e->valueDepth);
    e->kPend    &= ~(1u << e->valueDepth);
    e->kKnown   &= ~(1u << e->valueDepth);
    e->xBorrow  &= ~(1u << e->valueDepth);
    e->idxKnown &= ~(1u << e->valueDepth);
}

/* The four opcodes an enum-variant `match` is made of. Arming any ONE
 * of them buys exactly nothing -- the next link of the chain is two
 * instructions later -- so they land together. See matchEnumType and
 * matchMissResume above for the two facts they share. */
JitArmResult emitMatchTypePop(Emit *e, ObjClosure *closure,
                              const uint8_t *code, int *offp) {
    int off = *offp;
    ObjFunction *fn = closure->fn;
    do {
        if (!matchArmOn(e)) goto unarmedOpcode;
        uint32_t nameIdx = jaiReadU24(code + off + 1);
        int16_t  jump    = jaiReadI16(code + off + 4);
        uint32_t miss;
        /* Everything is asked before the model is touched: the unarmed
         * path below resumes the interpreter at this instruction, which is
         * only right while the operand stack is still as the bytecode
         * describes it. */
        if (!matchMissResume(e, fn, off, 6, jump, &miss)) goto unarmedOpcode;
        if (e->stack[e->depth - 1] != SLOT_OBJ) goto unarmedOpcode;
        ObjEnum *et = matchEnumType(closure, nameIdx);
        if (et == NULL) goto unarmedOpcode;
        if (!notePopSkip(e, miss)) goto unarmedOpcode;
        unsigned rSubj;
        if (!popValue(e, &rSubj, NULL)) goto unarmedOpcode;
        gMatchUsed = true;
        emitEnumTypeGuard(e, rSubj, et, miss);
        off += 6;
        break;
    } while (0);
    *offp = off;
    return JIT_ARM_OK;
unarmedOpcode:
    *offp = off;
    return JIT_ARM_UNARMED;
}

JitArmResult emitEnumTag(Emit *e, int *offp) {
    int off = *offp;
    do {
        if (!matchArmOn(e)) goto unarmedOpcode;
        /* PEEKS: the subject stays where it is and the tag goes on top of
         * it, which is what OP_SWAP_POP below is there to clean up. */
        if (e->depth == 0) goto unarmedOpcode;
        if (e->stack[e->depth - 1] != SLOT_OBJ) goto unarmedOpcode;
        /* Room asked for BEFORE anything is emitted: a push that fails
         * once the guard is out declines the whole body, where a refusal
         * would have cost only this instruction. Repeats pushValue3's own
         * two tests, minus the inlined-bank one this arm cannot reach. */
        if (e->depth >= JIT_MAX_STACK) goto unarmedOpcode;
        if (!e->measuring && e->valueDepth + 1 > valueBankRoom(e)) {
            goto unarmedOpcode;
        }
        gMatchUsed = true;
        unsigned rSubj = xHeldIn(e, e->valueDepth - 1);
        /* Not free, and not removable either: the guard the MATCH_TYPE_POP
         * before it emitted proved a fact about a value that has since been
         * popped and re-read from its local, and nothing in the model
         * carries that across. A non-enum here RAISES in the interpreter,
         * which the tier cannot do, so this deoptimises rather than
         * branching. */
        emit(e, jaiA64LdrW(JIT_SCRATCH_A, rSubj,
                           (unsigned)offsetof(Obj, type)));
        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_ENUM_VAL));
        branchOnDeopt(e, JAI_A64_NE);
        if (e->failed) return JIT_ARM_REFUSED;
        if (!pushValue(e, SLOT_INT, 0, NULL)) return JIT_ARM_REFUSED;
        /* Sixteen bits, not thirty-two: `count` shares the tag's word, and
         * a word-wide load would hand the compare below the neighbour in
         * its high bits. */
        emit(e, jaiA64LdrHalf(pushReg(e) - 1, rSubj,
                              (unsigned)offsetof(ObjEnumVal, tag)));
        off += 1;
        break;
    } while (0);
    *offp = off;
    return JIT_ARM_OK;
unarmedOpcode:
    *offp = off;
    return JIT_ARM_UNARMED;
}

JitArmResult emitSwapPop(Emit *e, int *offp) {
    int off = *offp;
    do {
        if (!matchArmOn(e)) goto unarmedOpcode;
        if (e->depth < 2) goto unarmedOpcode;
        if (!holdsRegister(e->stack[e->depth - 1]) ||
            !holdsRegister(e->stack[e->depth - 2])) goto unarmedOpcode;
        /* An index proof names a value slot, and this moves a value
         * between two of them; refusing is cheaper than moving the proof,
         * and the shape this arm exists for never carries one. */
        if ((e->idxKnown & (3u << (e->valueDepth - 2))) != 0) {
            goto unarmedOpcode;
        }
        gMatchUsed = true;
        /* Both entries in their own X register, which is what the move and
         * the model shuffle below both assume. */
        fpSyncAll(e);
        settleAll(e);
        unsigned rTop   = valueXReg(e, e->valueDepth - 1);
        unsigned rUnder = valueXReg(e, e->valueDepth - 2);
        if (rTop != rUnder) emit(e, jaiA64MovX(rUnder, rTop));
        swapPopEntry(e);
        off += 1;
        break;
    } while (0);
    *offp = off;
    return JIT_ARM_OK;
unarmedOpcode:
    *offp = off;
    return JIT_ARM_UNARMED;
}

JitArmResult emitMatchConstPop(Emit *e, ObjFunction *fn, const uint8_t *code,
                               int *offp, bool *afterUncondp) {
    int off = *offp;
    do {
        if (!matchArmOn(e)) goto unarmedOpcode;
        uint32_t kIdx = jaiReadU24(code + off + 1);
        int16_t  jump = jaiReadI16(code + off + 4);
        uint32_t miss;
        if (kIdx >= (uint32_t)fn->chunk.constants.count) goto unarmedOpcode;
        Value k = fn->chunk.constants.data[kIdx];
        /* The enum arm of the lowering compares a tag, so the constant is
         * an int and the entry is the int OP_ENUM_TAG pushed.
         * jaiValuesEqual answers those two with `==` and nothing else, so
         * a compare is the whole of it. Every other pairing goes to the
         * interpreter. */
        if (!IS_INT(k)) goto unarmedOpcode;
        if (!matchMissResume(e, fn, off, 6, jump, &miss)) goto unarmedOpcode;
        if (e->stack[e->depth - 1] != SLOT_INT) goto unarmedOpcode;
        /* `A | B => e` ends each failed alternative with an unconditional
         * jump to the arm body, sitting immediately before the OP_POP the
         * miss edge skips. The walk cannot reach that OP_POP with the entry
         * the bytecode says is there -- see matchMissResume -- so this arm
         * emits the jump itself and steps over both. */
        int succ = off + 6;
        int64_t armBody = -1;
        if (miss == (uint32_t)(succ + 4) && code[succ] == OP_JUMP) {
            armBody = (int64_t)succ + 3 + jaiReadI16(code + succ + 1);
            if (armBody < 0 || armBody >= fn->chunk.count) {
                goto unarmedOpcode;
            }
        }
        if (!notePopSkip(e, miss)) goto unarmedOpcode;
        unsigned rTag;
        if (!popValue(e, &rTag, NULL)) goto unarmedOpcode;
        gMatchUsed = true;
        int64_t kv = AS_INT(k);
        if (kv >= 0 && kv <= 4095) {
            emitCmpImm(e, rTag, kv);
        } else {
            emitConst64(e, JIT_SCRATCH_A, kv);
            emit(e, jaiA64SubsXReg(31, rTag, JIT_SCRATCH_A));
        }
        branchTo(e, miss, true, JAI_A64_NE);
        if (armBody >= 0) {
            branchTo(e, (uint32_t)armBody, false, 0);
            off = (int)miss;
            *afterUncondp = true;   /* the fall-through edge is gone */
        } else {
            off = succ;
        }
        break;
    } while (0);
    *offp = off;
    return JIT_ARM_OK;
unarmedOpcode:
    *offp = off;
    return JIT_ARM_UNARMED;
}

#endif /* __aarch64__ || __arm64__ */
