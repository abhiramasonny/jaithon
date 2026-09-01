/* jit_call_direct.c -- branching straight to a compiled callee's entry: the
 * argument-kind match that makes it sound, and reading its result back. */

#include "vm/jit/jit_arm64.h"
#include "vm/jit/jit_field_read.h"
#include "vm/vm.h"

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "vm/jit/jit_internal.h"

#if (defined(__aarch64__) || defined(__arm64__))

/* Direct-branch arguments arrive as a raw payload, so the caller's kind must match what the callee
 * was specialised for -- and for an instance, the same class shape, since every field offset was resolved against it. jaiJitEnterFunc is the only check standing between a float's bits and a body that treats them as a pointer. `firstIdx`: the entry above the callee for a plain call, or the receiver (callee's slot 0) for a method. */
bool directCallArgsMatch(Emit *e, const ObjFunction *cfn,
                                unsigned firstIdx, unsigned argc) {
    for (unsigned i = 0; i < argc; i++) {
        unsigned idx = firstIdx + i;
        SlotKind have = e->stack[idx];
        SlotKind want = (SlotKind)cfn->jitParamKind[i];
        if (!holdsRegister(have)) {
            e->whyNot = "a direct call argument that is not in a register";
            return false;
        }
        if (want == SLOT_OPAQUE) continue;   /* never read; see seedLocals */
        if (want == SLOT_MAYBE_INST) {
            if (have != SLOT_INST && have != SLOT_MAYBE_INST) {
                e->whyNot = "a direct call argument is not the parameter's kind";
                return false;
            }
        } else if (have != want) {
            e->whyNot = "a direct call argument is not the parameter's kind";
            return false;
        }
        if ((want == SLOT_INST || want == SLOT_MAYBE_INST) &&
            e->stackShape[idx] != cfn->jitParamShape[i]) {
            e->whyNot = "a direct call passing a different class";
            return false;
        }
    }
    return true;
}

/* Branches straight to a compiled callee's entry, skipping the descriptor/jaiCallValue/interpreter-
 * frame path. Convention: raw payloads in x0.., closure in the last arg register if the callee reads an upvalue, x0/x1 = value/verdict on return -- the same one jaiJitEnterFunc checks and a self-call already uses, so skipping that entry means answering its checks here instead: module version (why the callee must live in the caller's module), every parameter's kind+shape (by the caller's model), and the verdict (below). Nonzero verdict: a callee that writes NOTHING can have the whole call abandoned and re-executed from the pre-call stack (two compares, no stub); a callee that WRITES cannot be re-run -- verdict 4 means it deoptimised part-way and is FINISHED in the interpreter from its own record, sharing the `selfSlow` machinery a recursive self-call already uses. A raised exception goes to the throw exit instead, since its effects already happened. `calleeReg`: the ObjClosure register, or -1 if baked in. `cidx`: operand-stack index of the callee entry (the RECEIVER for a method, i.e. its slot 0). `after`: offset of the fall-through instruction. */
/* A `-> T?` result read back out of a call descriptor.
 *
 * Two tags are acceptable where every other kind has exactly one, so the single
 * compare the other arms use cannot serve. VAL_NULL is zero, which is what lets
 * "object or null" be two csels and a compare rather than a branch: afterwards
 * D holds the payload or a defined zero, and B is zero exactly when the tag was
 * one of the two.
 *
 * The class checks apply only to a non-null. The field arm at
 * OP_GET_FIELD_LOCAL keeps its equivalents branch-free by redirecting the loads
 * at its live receiver; a call result has no such pointer to borrow, so they
 * sit behind a forward branch instead. A deopt inside that span is sound for
 * the reason emitBoundsNormalise's is: the record is written from the
 * descriptor, which holds the true Value whichever way the branch went.
 *
 * Worth having because after this the refusal it removes was the tier's single
 * largest on the self-hosted compiler: one `check --no-cache` of
 * compile/parser.jai stopped 64 bodies at "callee's return kind not usable"
 * with the kind being exactly SLOT_MAYBE_INST, and every one of them had a
 * resolvable class. `-> Node?` is what a parser's methods return. */
void emitMaybeInstResult(Emit *e, unsigned dst, unsigned rat,
                                uint32_t rshape, uint32_t deoptIp) {
    emit(e, jaiA64LdrW(JIT_SCRATCH_A, 31, rat));
    emit(e, jaiA64LdrX(JIT_SCRATCH_D, 31, rat + 8));
    emit(e, jaiA64MovzX(JIT_SCRATCH_B, 0, 0));
    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, VAL_OBJ));
    emit(e, jaiA64CselX(JIT_SCRATCH_D, JIT_SCRATCH_D, JIT_SCRATCH_B,
                        JAI_A64_EQ));
    emit(e, jaiA64CselX(JIT_SCRATCH_B, JIT_SCRATCH_B, JIT_SCRATCH_A,
                        JAI_A64_EQ));
    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_B, 0));
    branchOnDeoptAt(e, JAI_A64_NE, deoptIp, true);

    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_D, 0));
    unsigned skip = e->count;
    emit(e, jaiA64BCond(JAI_A64_EQ, 0));
    emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_D,
                       (unsigned)offsetof(Obj, type)));
    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_INSTANCE));
    branchOnDeoptAt(e, JAI_A64_NE, deoptIp, true);
    emit(e, jaiA64LdrX(JIT_SCRATCH_A, JIT_SCRATCH_D,
                       (unsigned)offsetof(ObjInstance, klass)));
    emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_A,
                       (unsigned)offsetof(ObjClass, shapeId)));
    emitConst64(e, JIT_SCRATCH_B, (int64_t)rshape);
    emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
    branchOnDeoptAt(e, JAI_A64_NE, deoptIp, true);
    /* Nothing to jump over means the arena filled mid-sequence; leaving the
     * branch unpatched would run the class loads on a null pointer. */
    if (skip < e->count && e->count <= JIT_MAX_INSTS) {
        e->code[skip] = jaiA64BCond(JAI_A64_EQ, (int32_t)(e->count - skip));
    } else {
        e->failed = true;
    }
    emit(e, jaiA64MovX(dst, JIT_SCRATCH_D));
}

bool jitAnyGuard(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_ANY_GUARD");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

bool jitReturnKnownOn(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_RETURN_KNOWN");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

bool emitDirectCall(Emit *e, ObjFunction *caller, ObjFunction *cfn,
                           Value calleeVal, int calleeReg, unsigned cidx,
                           unsigned argc, uint32_t callOff, uint32_t after,
                           bool method) {
    /* Either verdict path below can come back with an exception pending. */
    if (!raiseExitAllowed(e, "a call that can raise inside a try")) return false;
    if (cfn->module != caller->module) {
        e->whyNot = "a direct callee from another module";
        return false;
    }
    /* Slot 0 is the closure for a plain function and the receiver for a
     * method, so which of the two this is decides where the arguments start
     * and whether the receiver is one of them. */
    if (cfn->jitArgBase != (method ? 0u : 1u)) {
        e->whyNot = method
            ? "a direct method that does not take its receiver in slot 0"
            : "a direct callee that is not a plain function";
        return false;
    }
    /* The callee's baked classes/closures/natives are pinned by ITS jitFuncModuleVersion, checked at the
     * entry this call skips -- valid only if the caller's own check agrees NOW: a global rebound after the callee compiled would leave jitFuncModuleVersion stale but still reachable via a direct call, and a caller compiling afterwards would silently pin the newer version. */
    if (caller->module == NULL ||
        cfn->jitFuncModuleVersion != caller->module->version) {
        e->whyNot = "a direct callee compiled against an older module";
        return false;
    }
    /* A callee that writes is finished in the interpreter on verdict 4, and
     * that needs its ObjClosure -- which only a callee baked in at compile
     * time provides. */
    bool writes = !cfn->jitFuncNoWrite;
    if (writes && (calleeReg >= 0 || !IS_CLOSURE(calleeVal))) {
        e->whyNot = "a direct callee that writes and is not known here";
        return false;
    }
    /* `nargs` is how many registers the branch fills. A method's receiver is
     * one of them; a plain call's callee entry holds no register at all. */
    unsigned nargs = method ? argc + 1u : argc;
    unsigned firstIdx = method ? cidx : cidx + 1u;
    unsigned calleeArgs = (unsigned)cfn->jitArgCount;
    bool wantsClosure = calleeArgs == nargs + 1u;
    if (!wantsClosure && calleeArgs != nargs) {
        e->whyNot = "a direct callee with a different arity";
        return false;
    }
    if (calleeArgs > JIT_MAX_ARITY) {
        e->whyNot = "a direct callee with too many arguments";
        return false;
    }

    if (!directCallArgsMatch(e, cfn, firstIdx, nargs)) return false;
    if (wantsClosure &&
        (SlotKind)cfn->jitParamKind[nargs] != SLOT_CLOSURE) {
        e->whyNot = "a direct callee whose trailing argument is not its closure";
        return false;
    }
    if (wantsClosure && calleeReg < 0 && !IS_CLOSURE(calleeVal)) {
        e->whyNot = "a direct callee that wants a closure it has not got";
        return false;
    }
    /* SLOT_NULL: a `-> void` function. Epilogue leaves x0 zero, so the pushed entry has a fixed tag and
     * zero payload -- same treatment a self-call to a void function gets. Refusing it declined every caller of a procedure, which in nbody is the whole of `main`. */
    /* A callee whose walk never reached an OP_RETURN has no return kind to be
     * the contract, only the SLOT_INT of a zeroed Emit. `_is_ident_start` in
     * the lexer walks only to OP_GET_GLOBAL at offset 0 -- a cold `throw` on
     * its first instruction -- and still claimed to return an int, which made
     * `_is_ident_start(c) or _is_digit(c)` decline on "a branch on a int, not
     * a bool" and left `_ident_run_end`'s OSR loop retrying it eighty times.
     *
     * Declining here is not a coverage loss: the caller falls back to the
     * guarded emitGlobalCall path, which asks the interpreter's own
     * observation first and gets the right answer. */
    if (!cfn->jitReturnKnown && jitReturnKnownOn()) {
        e->whyNot = "a direct callee whose walk never reached a return";
        return false;
    }
    SlotKind rk = (SlotKind)cfn->jitReturnKind;
    ObjClass *rcls = NULL;
    /* SLOT_MAYBE_INST rides with SLOT_INST here and needs no guard of its own:
     * a direct branch takes the callee's raw payload, and for a nullable
     * instance that payload IS the pointer or a zero -- the same
     * representation this caller will hold. The callee's declared return kind
     * is the contract, exactly as it is for every other kind at this arm. */
    if (rk != SLOT_INT && rk != SLOT_FLOAT && rk != SLOT_BOOL &&
        rk != SLOT_INST && rk != SLOT_MAYBE_INST && rk != SLOT_LIST &&
        rk != SLOT_OBJ && rk != SLOT_NULL) {
        return subWhy(e, "a direct callee returning %s", slotKindName(rk));
    }
    if ((rk == SLOT_INST || rk == SLOT_MAYBE_INST) &&
        (cfn->jitReturnShape == 0 ||
         !jaiClassForShape(cfn->jitReturnShape, &rcls) || rcls == NULL)) {
        e->whyNot = "callee's return class not on record";
        return false;
    }
    /* Asked here rather than where the slot is taken: below this point the
     * root fill has been emitted and the descriptor linked onto the collector's
     * chain, so there is no falling back to the descriptor path any more. */
    if (writes && e->selfSlowCount >= JIT_MAX_SELF_SLOW) {
        e->whyNot = "more slow call sites than the tier tracks";
        return false;
    }

    /* Past here everything is settled and this call is happening: a failure
     * below is the emitter running out of room, not a decision, so it stops
     * the compile rather than falling back to the descriptor path onto a
     * half-written instruction stream. */

    /* Roots before the branch: a `bl` pushes none, and the callee may
     * allocate -- OP_GET_SLICE builds a fresh list without ever counting as a
     * heap write. */
    unsigned callRoots = 0;
    if (!emitRootFill(e, e->descOffset, &callRoots)) { e->failed = true; return false; }
    if (callRoots > 0) {
        unsigned dd = e->descOffset;
        emit(e, jaiA64MovzX(JIT_SCRATCH_A, callRoots, 0));
        emit(e, jaiA64StrX(JIT_SCRATCH_A, 31,
                           dd + (unsigned)offsetof(JitCallDesc, nroots)));
        emitConst64(e, JIT_SCRATCH_A, (int64_t)(uintptr_t)&gJitFrames);
        emit(e, jaiA64LdrX(JIT_SCRATCH_B, JIT_SCRATCH_A, 0));
        emit(e, jaiA64AddXImm(JIT_SCRATCH_C, 31, dd));
        emit(e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_C,
                           (unsigned)offsetof(JitCallDesc, link)));
        emit(e, jaiA64StrX(JIT_SCRATCH_C, JIT_SCRATCH_A, 0));
    }

    unsigned firstArg = firstIdx - (e->depth - e->valueDepth);
    for (unsigned i = 0; i < nargs; i++) {
        emit(e, jaiA64MovX(i, valueXReg(e, firstArg + i)));
    }
    if (wantsClosure) {
        if (calleeReg >= 0) emit(e, jaiA64MovX(nargs, (unsigned)calleeReg));
        else emitConst64(e, nargs, (int64_t)(uintptr_t)AS_OBJ(calleeVal));
    }
    emitConst64(e, JIT_SCRATCH_D, (int64_t)(uintptr_t)cfn->jitFunc);
    noteScratchClobber(e);
    emit(e, jaiA64Blr(JIT_SCRATCH_D));

    if (callRoots > 0) {
        unsigned dd = e->descOffset;
        emitConst64(e, JIT_SCRATCH_A, (int64_t)(uintptr_t)&gJitFrames);
        emit(e, jaiA64AddXImm(JIT_SCRATCH_C, 31, dd));
        emit(e, jaiA64LdrX(JIT_SCRATCH_B, JIT_SCRATCH_C,
                           (unsigned)offsetof(JitCallDesc, link)));
        emit(e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_A, 0));
    }

    unsigned si = 0;
    if (writes) {
        /* One compare and one not-taken branch on the fast path; every other
         * answer is the shared cold block. The record has to be taken here,
         * where the model still holds the receiver and the arguments, even
         * though the block is emitted with the stubs. */
        if (e->selfSlowCount >= JIT_MAX_SELF_SLOW) { e->failed = true; return false; }
        si = e->selfSlowCount++;
        e->selfSlow[si].roots    = callRoots;
        e->selfSlow[si].stub     = -1;
        e->selfSlow[si].callee   = AS_CLOSURE(calleeVal);
        e->selfSlow[si].retShape = rk == SLOT_INST ? cfn->jitReturnShape : 0;
        e->selfSlow[si].retType  = rk == SLOT_INST ? (int)OBJ_INSTANCE
                                 : rk == SLOT_LIST ? (int)OBJ_LIST
                                                   : -1;
        if (!deoptRecordAt(e, callOff, false, &e->selfSlow[si].deoptBail)) {
            e->failed = true;
            return false;
        }
        emit(e, jaiA64SubsXImm(31, 1, 0));
        if (e->fixupCount >= JIT_MAX_FIXUPS) { e->failed = true; return false; }
        e->fixups[e->fixupCount].instIndex    = (int)e->count;
        e->fixups[e->fixupCount].targetOffset = FIXUP_SELFSLOW - si;
        e->fixups[e->fixupCount].conditional  = true;
        e->fixups[e->fixupCount].depth        = -1;
        e->fixupCount++;
        emit(e, jaiA64BCond(JAI_A64_NE, 0));
    } else {
        /* Verdict 2 is a pending exception: the interpreter owns it and this
         * call must not run again. */
        emit(e, jaiA64SubsXImm(31, 1, 2));
        if (e->fixupCount >= JIT_MAX_FIXUPS) { e->failed = true; return false; }
        e->fixups[e->fixupCount].instIndex    = (int)e->count;
        e->fixups[e->fixupCount].targetOffset = FIXUP_THREW;
        e->fixups[e->fixupCount].conditional  = true;
        e->fixups[e->fixupCount].depth        = -1;
        e->fixupCount++;
        emit(e, jaiA64BCond(JAI_A64_EQ, 0));
        /* Anything else -- a bail, or a guard that failed inside the callee --
         * hands the whole call back. The record is taken with the callee and
         * its arguments still on the model's stack, which is what the
         * interpreter expects to find at this offset. */
        emit(e, jaiA64SubsXImm(31, 1, 0));
        branchOnDeoptAt(e, JAI_A64_NE, callOff, false);
    }

    for (unsigned i = 0; i < nargs; i++) {
        unsigned r;
        if (!popValue(e, &r, NULL)) { e->failed = true; return false; }
    }
    if (!method) {
        if (e->depth == 0) { e->failed = true; return false; }
        e->depth--;
    }
    if (!pushValue(e, rk, cfn->jitReturnShape, rcls)) { e->failed = true; return false; }
    emit(e, jaiA64MovX(pushReg(e) - 1, 0));
    if (writes) {
        e->selfSlow[si].resultReg = pushReg(e) - 1;
        e->selfSlow[si].returnTo  = (int)e->count;
        e->selfSlow[si].tag = rk == SLOT_INT   ? VAL_INT
                            : rk == SLOT_FLOAT ? VAL_FLOAT
                            : rk == SLOT_BOOL  ? VAL_BOOL
                                               : VAL_OBJ;
        /* The interpreted continuation is typed by nothing this compiled for,
         * so what it hands back is checked and a surprise resumes AFTER the
         * call -- which has happened and must not happen twice. */
        if (!deoptRecordAt(e, after, true, &e->selfSlow[si].deoptKind)) {
            e->failed = true;
            return false;
        }
        /* A call that writes is an effect, so no bail may follow it -- the
         * same rule the descriptor path lives under. */
        e->wroteHeap = true;
    }
    /* Deliberately NOT wroteHeap for a non-writing callee, unlike the descriptor path: it stores and
     * calls nothing, so a re-run repeats no visible effect. It may still allocate (OP_GET_SLICE does) -- a fresh object isn't an observable effect. */
    return true;
}

#endif /* __aarch64__ || __arm64__ */
