/* jit_entry.c -- the interpreter's doors into compiled code: argument marshalling,
 * result unpacking, and the flat one-argument call paths. */
#include "vm/jit/jit.h"

/* For jaiJitFieldReadFor: which builtins are one load from their receiver. */
#include "vm/jit/jit_field_read.h"
/* For jaiBuiltinMethod: resolving `xs.len()` to a native needs the runtime's name table. */
/* For jaiOpBranchOperandAt: says which opcodes carry a branch target. */
#include "vm/vm.h"

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "vm/jit/jit_internal.h"

#if (defined(__aarch64__) || defined(__arm64__))

/* ------------------------------------------------------------------ */
/* Entry from the interpreter                                          */
/* ------------------------------------------------------------------ */

typedef JitResult (*Fn0)(void);
typedef JitResult (*Fn1)(int64_t);
typedef JitResult (*Fn2)(int64_t, int64_t);
typedef JitResult (*Fn3)(int64_t, int64_t, int64_t);
typedef JitResult (*Fn4)(int64_t, int64_t, int64_t, int64_t);
typedef JitResult (*Fn5)(int64_t, int64_t, int64_t, int64_t, int64_t);
typedef JitResult (*Fn6)(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
typedef JitResult (*Fn7)(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
typedef JitResult (*Fn8)(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);

/* One incoming argument, converted from the Value the interpreter holds to the
 * raw payload the compiled prologue expects, with every check the compiled
 * body was allowed to assume. */
static inline bool jitArgIn(ObjClosure *closure, const Value *slotBase,
                            unsigned i, int64_t *out) {
    const ObjFunction *fn = closure->fn;
    Value v = slotBase[fn->jitArgBase + i];
    {
        switch ((SlotKind)fn->jitParamKind[i]) {
        case SLOT_INT:
            if (!IS_INT(v)) return false;
            *out = AS_INT(v);
            break;
        case SLOT_FLOAT: {
            if (!IS_FLOAT(v)) return false;
            double d = AS_FLOAT(v);
            memcpy(out, &d, sizeof *out);
            break;
        }
        case SLOT_MAYBE_INST: {
            /* The pointer, or zero for null -- the register form. The class is
             * still checked when there is one, because a self-call branches
             * straight to the prologue and this is the only place it is
             * established. */
            if (IS_NULL(v)) { *out = 0; break; }
            if (!IS_INSTANCE(v)) return false;
            ObjInstance *mi = AS_INSTANCE(v);
            if (mi->klass == NULL ||
                mi->klass->shapeId != fn->jitParamShape[i]) {
                return false;
            }
            *out = (int64_t)(uintptr_t)mi;
            break;
        }
        case SLOT_INST: {
            /* Class as well as type: every field offset in the body was resolved against this one shape. Holding
             * the raw pointer is safe only because the body can't allocate (no collection can run meanwhile), and the argument slots keep the instance reachable. */
            if (!IS_INSTANCE(v)) return false;
            ObjInstance *inst = AS_INSTANCE(v);
            if (inst->klass == NULL ||
                inst->klass->shapeId != fn->jitParamShape[i]) {
                return false;
            }
            *out = (int64_t)(uintptr_t)inst;
            break;
        }
        case SLOT_LIST:
            if (!IS_LIST(v)) return false;
            *out = (int64_t)(uintptr_t)AS_LIST(v);
            break;
        case SLOT_OBJ:
            if (!IS_OBJ(v)) return false;
            *out = (int64_t)(uintptr_t)AS_OBJ(v);
            break;
        case SLOT_BOOL:
            if (!IS_BOOL(v)) return false;
            *out = AS_BOOL(v) ? 1 : 0;
            break;
        case SLOT_OPAQUE:
            *out = 0;   /* never read; see seedLocals */
            break;
        case SLOT_CLOSURE:
            /* Not a slot at all: the closure the interpreter is calling. Safe
             * to hold raw for the same reason every other pointer here is --
             * the body cannot allocate, and the caller holds this closure. */
            *out = (int64_t)(uintptr_t)closure;
            break;
        default:
            return false;
        }
    }
    return true;
}

static inline JaiJitOutcome jitResultOut(ObjFunction *fn, JitResult r,
                                         Value *slotBase) {
    /* The verdict is the low byte. A SLOT_DYNAMIC body's return site puts its
     * Value tag in the byte above (JIT_RET_TAG_SHIFT); every other exit --
     * bail, raise, deopt -- writes a bare 1, 2 or 4 with nothing above it. */
    int64_t verdict = r.bailed & 0xff;
    if (verdict == 2) return JAI_JIT_ERROR;
    if (verdict == 4) return JAI_JIT_DEOPT;
    if (verdict) {
        /* Overflow or a low stack: nothing was written (the body cannot write), so handing the call back to
         * the interpreter is enough -- it raises the error with a traceback. Refused permanently so a bailing body isn't re-entered every call only to bail again. */
        fn->jitRefused = true;
        fn->jitFunc = NULL;
        /* Nothing will look at this body again, so the blocked-on callee is
         * held for a retry that cannot happen. */
        fn->jitBlockedOn = NULL;
        return JAI_JIT_DECLINED;
    }

    if ((SlotKind)fn->jitReturnKind == SLOT_DYNAMIC) {
        /* The tag came from the return site -- emitTagFor, off the payload for
         * a nullable kind -- so this is the one place the tier rebuilds a
         * Value from a run-time tag rather than a compile-time kind. */
        switch ((r.bailed >> JIT_RET_TAG_SHIFT) & 0xff) {
        case VAL_INT:
            slotBase[0] = INT_VAL(r.value);
            break;
        case VAL_FLOAT: {
            /* Raw bits: never through a double conversion, which could
             * canonicalise a NaN payload. */
            double d;
            memcpy(&d, &r.value, sizeof d);
            slotBase[0] = FLOAT_VAL(d);
            break;
        }
        case VAL_BOOL:
            /* One byte: BOOL_VAL writes only the union's bool member and a
             * bool register is trusted only that wide. */
            slotBase[0] = BOOL_VAL((r.value & 0xff) != 0);
            break;
        case VAL_NULL:
            slotBase[0] = NULL_VAL;
            break;
        case VAL_OBJ:
            /* emitTagFor already answers VAL_NULL for a zero payload of a
             * nullable kind, and the other object kinds are never zero; the
             * test stays so that {VAL_OBJ, obj = NULL} -- the shipped bug
             * shape -- cannot be built here under any future producer. */
            slotBase[0] = r.value == 0 ? NULL_VAL
                                       : OBJ_VAL((Obj *)(uintptr_t)r.value);
            break;
        default:
            return JAI_JIT_DECLINED;
        }
        vm.stackTop = slotBase + 1;
        return JAI_JIT_DONE;
    }
    if ((SlotKind)fn->jitReturnKind == SLOT_MAYBE_INST ||
        (SlotKind)fn->jitReturnKind == SLOT_MAYBE_OBJ) {
        slotBase[0] = r.value == 0 ? NULL_VAL
                                   : OBJ_VAL((Obj *)(uintptr_t)r.value);
        vm.stackTop = slotBase + 1;
        return JAI_JIT_DONE;
    }
    switch ((SlotKind)fn->jitReturnKind) {
    case SLOT_INT:
        slotBase[0] = INT_VAL(r.value);
        break;
    case SLOT_FLOAT: {
        double d;
        memcpy(&d, &r.value, sizeof d);
        slotBase[0] = FLOAT_VAL(d);
        break;
    }
    case SLOT_INST:
    case SLOT_LIST:
    case SLOT_OBJ:
        slotBase[0] = OBJ_VAL((Obj *)(uintptr_t)r.value);
        break;
    case SLOT_BOOL:
        slotBase[0] = BOOL_VAL(r.value != 0);
        break;
    case SLOT_NULL:
        slotBase[0] = NULL_VAL;
        break;
    default:
        return JAI_JIT_DECLINED;
    }
    vm.stackTop = slotBase + 1;
    return JAI_JIT_DONE;
}

/* The callee this body's walk stopped at has a compiled form now. Compile the
 * body again, from nothing: the retried compile shares no state with the
 * truncated one -- jaiJitCompileFunc reseeds the model from this call's live
 * arguments exactly as the first compile did -- so the form it produces is
 * correct on its own terms or it is not produced.
 *
 * Out of line and called only from behind a NULL test, because this sits on the
 * path of every interpreted call into every compiled body.
 *
 * The old form is put back whenever the retry does not succeed, and that is not
 * a nicety: five of twenty retries on `check lib/std` fail outright, and
 * without the restore those bodies fall from partly-compiled to entirely
 * interpreted -- strictly worse than the truncated form they had. It is cheap
 * because a failed compile writes NOTHING to the function: every fn->jit* store
 * in compileFuncOnce is in its success tail, past the last `return false`, so
 * the only field to undo is the jitFunc pointer this cleared itself. The rest
 * is saved and restored anyway, so that stays true of whatever the tail writes
 * next.
 *
 * Guarantees fn->jitFunc is not NULL on return. */
static JAI_NOINLINE void jitRecompileBlocked(ObjClosure *closure,
                                             ObjFunction *fn, Value *slotBase) {
    ObjFunction *blocker = fn->jitBlockedOn;
    /* Still cold. Ask again next call -- that is the whole point, the callee
     * has not reached its own threshold yet. */
    if (blocker->jitFunc == NULL) return;

    uint8_t *old       = fn->jitFunc;
    uint8_t  oldArgC   = fn->jitArgCount;
    uint8_t  oldArgB   = fn->jitArgBase;
    uint8_t  oldRet    = fn->jitReturnKind;
    bool     oldRetK   = fn->jitReturnKnown;
    uint32_t oldRetS   = fn->jitReturnShape;
    bool     oldNoWr   = fn->jitFuncNoWrite;
    uint32_t oldVer    = fn->jitFuncModuleVersion;
    uint8_t  oldKind[8];
    uint32_t oldShape[8];
    memcpy(oldKind,  fn->jitParamKind,  sizeof oldKind);
    memcpy(oldShape, fn->jitParamShape, sizeof oldShape);

    /* jitFunc NULL for the duration, which is also what stops anything
     * reentering this body's compiled form while it has none; jitBlockedOn is
     * deliberately LEFT SET, because it is the only thing marking `blocker`
     * for a collector that may run inside the compile. The success tail
     * overwrites it with whatever stops the new walk, after the last
     * allocation, so the comparison below still reads the old value here. */
    fn->jitRecompiles++;
    fn->jitFunc = NULL;
    if (jaiJitCompileFunc(closure, slotBase)) {
        fn->jitFuncModuleVersion = fn->module->version;
        if (getenv("JAI_JIT_WHY")) {
            fprintf(stderr, "[jit] recompiled %s: `%s` has compiled since\n",
                    jitFnLabel(fn),
                    blocker->name ? blocker->name->chars : "<anon>");
        }
        /* One retry per (caller, callee) PAIR. The fresh walk has recorded
         * whatever stops it NOW, which is usually a different callee and is
         * where the second retry's win comes from; the same callee again is
         * not a new pair and asking twice would be a loop.
         *
         * Deliberately NOT gated on the new form having walked further. That
         * was built and measured and it is worse -- 308.35M against 300.22M --
         * because a bytecode offset is not a quality proxy: `_scan_token`'s
         * new stop offset is LOWER and its interpreted work falls from
         * 6,990,397 to 2,527. */
        if (fn->jitBlockedOn == blocker ||
            fn->jitRecompiles >= JAI_JIT_RECOMPILES) {
            fn->jitBlockedOn = NULL;
        }
        return;
    }

    /* A compile that failed on this callee fails again on it, so the pair is
     * spent whether or not it produced anything. */
    fn->jitBlockedOn         = NULL;
    fn->jitFunc              = old;
    fn->jitArgCount          = oldArgC;
    fn->jitArgBase           = oldArgB;
    fn->jitReturnKind        = oldRet;
    fn->jitReturnKnown       = oldRetK;
    fn->jitReturnShape       = oldRetS;
    fn->jitFuncNoWrite       = oldNoWr;
    fn->jitFuncModuleVersion = oldVer;
    memcpy(fn->jitParamKind,  oldKind,  sizeof oldKind);
    memcpy(fn->jitParamShape, oldShape, sizeof oldShape);
    if (getenv("JAI_JIT_WHY")) {
        fprintf(stderr, "[jit] recompile of %s failed -- old form restored\n",
                jitFnLabel(fn));
    }
}

JaiJitOutcome jaiJitEnterFunc(ObjClosure *closure, Value *slotBase) {
    ObjFunction *fn = closure->fn;
    if (fn->jitFunc == NULL) return JAI_JIT_DECLINED;

    /* One load, and NULL for every body that compiled whole -- and for every
     * body at all once JAITHON_JIT_RECOMPILE=0 stops it being recorded. Both
     * the version test and the budget live inside the cold branch so this
     * stays the only thing the common path pays. */
    if (JAI_UNLIKELY(fn->jitBlockedOn != NULL)) {
        if (fn->module == NULL ||
            fn->module->version != fn->jitFuncModuleVersion) {
            /* This form is retired; rebuilding it is a different mechanism's
             * job and this one has no business holding the callee alive for a
             * retry that will never happen. */
            fn->jitBlockedOn = NULL;
            return JAI_JIT_DECLINED;
        }
        jitRecompileBlocked(closure, fn, slotBase);
    }

    /* Compiled code reads the global naming this function exactly once, at compile time, then calls it
     * directly. Rebinding the name must invalidate that; the module's version counter moves on every global mutation, so one comparison covers it -- conservative (any global write in the module retires the form), which is the safe direction. */
    if (fn->module == NULL || fn->module->version != fn->jitFuncModuleVersion) {
        return JAI_JIT_DECLINED;
    }

    unsigned arity = fn->jitArgCount;
    int64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
    /* Unrolled rather than a loop over `int64_t a[JIT_MAX_ARITY]` -- measured, not tidiness: an array of
     * int64 makes clang add a stack-protector prologue/epilogue and sends every argument out to the frame and back on its way to the register the call reads it from. This function sits on the path of every interpreted call into a compiled body, so both costs are paid per call. */
    if (arity > JIT_MAX_ARITY) return JAI_JIT_DECLINED;
    if (arity > 0 && !jitArgIn(closure, slotBase, 0, &a0)) return JAI_JIT_DECLINED;
    if (arity > 1 && !jitArgIn(closure, slotBase, 1, &a1)) return JAI_JIT_DECLINED;
    if (arity > 2 && !jitArgIn(closure, slotBase, 2, &a2)) return JAI_JIT_DECLINED;
    if (arity > 3 && !jitArgIn(closure, slotBase, 3, &a3)) return JAI_JIT_DECLINED;
    if (arity > 4 && !jitArgIn(closure, slotBase, 4, &a4)) return JAI_JIT_DECLINED;
    if (arity > 5 && !jitArgIn(closure, slotBase, 5, &a5)) return JAI_JIT_DECLINED;
    if (arity > 6 && !jitArgIn(closure, slotBase, 6, &a6)) return JAI_JIT_DECLINED;
    if (arity > 7 && !jitArgIn(closure, slotBase, 7, &a7)) return JAI_JIT_DECLINED;

    JitResult r;
    switch (arity) {
    case 0: r = ((Fn0)(uintptr_t)fn->jitFunc)(); break;
    case 1: r = ((Fn1)(uintptr_t)fn->jitFunc)(a0); break;
    case 2: r = ((Fn2)(uintptr_t)fn->jitFunc)(a0, a1); break;
    case 3: r = ((Fn3)(uintptr_t)fn->jitFunc)(a0, a1, a2); break;
    case 4: r = ((Fn4)(uintptr_t)fn->jitFunc)(a0, a1, a2, a3); break;
    case 5: r = ((Fn5)(uintptr_t)fn->jitFunc)(a0, a1, a2, a3, a4); break;
    case 6: r = ((Fn6)(uintptr_t)fn->jitFunc)(a0, a1, a2, a3, a4, a5); break;
    case 7: r = ((Fn7)(uintptr_t)fn->jitFunc)(a0, a1, a2, a3, a4, a5, a6); break;
    default: r = ((Fn8)(uintptr_t)fn->jitFunc)(a0, a1, a2, a3, a4, a5, a6, a7); break;
    }
    return jitResultOut(fn, r, slotBase);
}

/* See vm.h. Everything jaiCallValue1 and jaiJitEnterFunc do for one argument,
 * in one frame: the checks are unchanged and in the same order, but the
 * callee-saved prologue, the argument's round trip through the caller's frame
 * and the second `bl` are gone. Higher-order builtins drive this once per
 * element, so all of that was per-element cost.
 *
 * Only jitArgCount 1 is taken, plus a trailing SLOT_CLOSURE for a callee that
 * reads an upvalue -- that is not a slot, it is the closure itself, so a
 * capturing lambda still gets the flat path. Anything else declines and
 * jaiCallValue1 handles it exactly as before. */
static JAI_NOINLINE bool callFn1Rerun(Value *base, Value *out) {
    Value callee = base[0], arg = base[1];
    vm.stackTop = base;
    return jaiCallValue1(callee, arg, out);
}

bool jaiCallFn1(Value callee, Value arg, Value *out) {
    if (JAI_UNLIKELY(!IS_CLOSURE(callee))) return jaiCallValue1(callee, arg, out);
    ObjClosure *closure = AS_CLOSURE(callee);
    ObjFunction *fn = closure->fn;
    if (JAI_UNLIKELY(fn->jitFunc == NULL || fn->arity != 1))
        return jaiCallValue1(callee, arg, out);

    unsigned nargs = fn->jitArgCount;
    if (JAI_UNLIKELY(nargs == 0 || nargs > 2))
        return jaiCallValue1(callee, arg, out);
    if (nargs == 2 && (SlotKind)fn->jitParamKind[1] != SLOT_CLOSURE)
        return jaiCallValue1(callee, arg, out);

    /* Same guard jaiJitEnterFunc makes: compiled code read this module's
     * globals once, at compile time. */
    if (JAI_UNLIKELY(fn->module == NULL ||
                     fn->module->version != fn->jitFuncModuleVersion)) {
        return jaiCallValue1(callee, arg, out);
    }

    if (JAI_UNLIKELY(vm.stack == NULL ||
                     vm.stackTop + 3 > vm.stack + JAI_STACK_MAX)) {
        return jaiCallValue1(callee, arg, out);
    }

    /* The two cells are what keeps the closure and the argument reachable
     * while the compiled body runs, exactly as in jaiCallValue1: a compiled
     * body may allocate and a collection scans the VM stack. */
    Value *base = vm.stackTop;
    base[0] = callee;
    base[1] = arg;
    vm.stackTop = base + 2;

    int64_t a0 = 0;
    if (JAI_UNLIKELY(!jitArgIn(closure, base, 0, &a0))) {
        vm.stackTop = base;
        return jaiCallValue1(callee, arg, out);
    }

    int frameBase = vm.frameCount;
    JitResult r = nargs == 1
                      ? ((Fn1)(uintptr_t)fn->jitFunc)(a0)
                      : ((Fn2)(uintptr_t)fn->jitFunc)(a0,
                                                      (int64_t)(uintptr_t)closure);
    JaiJitOutcome outcome = jitResultOut(fn, r, base);
    if (JAI_LIKELY(outcome == JAI_JIT_DONE)) {
        *out = base[0];
        vm.stackTop = base;
        return true;
    }
    if (outcome == JAI_JIT_ERROR) {
        vm.stackTop = base;
        return false;
    }
    if (outcome == JAI_JIT_DEOPT) {
        return jaiFinishJitDeopt1(closure, base, frameBase, out);
    }
    /* Refused mid-flight (a bail retires the form), so re-run it interpreted.
     * Reading the callee and the argument back out of the two cells rather
     * than off the parameters is what keeps them dead across the `blr`: with
     * four live registers fewer, the callee-saved set this function has to
     * spill on every element drops from six pairs to two. */
    return callFn1Rerun(base, out);
}

/* See vm.h. The same tests jaiCallFn1 makes, in the same order, up to the
 * point where the answer stops depending on the callee alone. */
void jaiPrepareFn1(Value callee, JaiPreparedFn1 *prepared) {
    prepared->callee = callee;
    prepared->flat   = false;
    if (!IS_CLOSURE(callee)) return;
    ObjClosure *closure = AS_CLOSURE(callee);
    ObjFunction *fn = closure->fn;
    if (fn->jitFunc == NULL || fn->arity != 1) return;

    unsigned nargs = fn->jitArgCount;
    if (nargs == 0 || nargs > 2) return;
    if (nargs == 2 && (SlotKind)fn->jitParamKind[1] != SLOT_CLOSURE) return;
    if (fn->module == NULL) return;
    if (fn->module->version != fn->jitFuncModuleVersion) return;
    if (vm.stack == NULL) return;
    /* jitResultOut has to be able to answer for whatever comes back; a kind it
     * would decline is not worth preparing, since every element would fall
     * through to jaiCallFn1 anyway. */
    switch ((SlotKind)fn->jitReturnKind) {
    case SLOT_INT: case SLOT_FLOAT: case SLOT_INST: case SLOT_LIST:
    case SLOT_OBJ: case SLOT_BOOL: case SLOT_NULL: case SLOT_MAYBE_INST:
    case SLOT_MAYBE_OBJ:
        break;
    default:
        return;
    }

    prepared->closure       = closure;
    prepared->fn            = fn;
    prepared->entry         = fn->jitFunc;
    /* Fixed for the life of the VM: the stack is one allocation made at start
     * up and freed at teardown, so the room test is a compare against a
     * constant rather than two loads and an add. */
    prepared->limit         = vm.stack + JAI_STACK_MAX - 3;
    prepared->moduleVersion = fn->module->version;
    prepared->nargs         = (uint8_t)nargs;
    prepared->returnKind    = fn->jitReturnKind;
    prepared->intArg        = (SlotKind)fn->jitParamKind[0] == SLOT_INT &&
                              fn->jitArgBase == 1;
    prepared->flat          = true;
}

/* This element the long way, and then another attempt to prepare. Two
 * different states arrive here and both want the same treatment: the callee
 * has not compiled YET (the ordinary state for the first sixty-four elements,
 * since that is when the tier first looks at it), or the form prepared against
 * has been retired under the loop. Out of line so the flat path's prologue
 * stays as small as the body it is calling. */
static JAI_NOINLINE bool preparedFn1Slow(JaiPreparedFn1 *p, Value arg,
                                         Value *out) {
    Value callee = p->callee;
    bool ok = jaiCallFn1(callee, arg, out);
    jaiPrepareFn1(callee, p);
    return ok;
}

bool jaiCallPreparedFn1(JaiPreparedFn1 *p, Value arg, Value *out) {
    ObjFunction *fn = p->fn;
    /* The whole of the staleness check. `entry` is never NULL in a prepared
     * struct, so a body that bailed -- jitResultOut sets jitFunc to NULL -- is
     * caught by the same compare as a body recompiled under it. */
    if (JAI_UNLIKELY(!p->flat || fn->jitFunc != p->entry ||
                     fn->module->version != p->moduleVersion)) {
        return preparedFn1Slow(p, arg, out);
    }
    Value *base = vm.stackTop;
    if (JAI_UNLIKELY(base > p->limit)) {
        return jaiCallFn1(p->callee, arg, out);
    }

    /* The two cells that keep the closure and the argument reachable while the
     * compiled body runs; see jaiCallFn1, whose window this is. */
    base[0] = p->callee;
    base[1] = arg;
    vm.stackTop = base + 2;

    int64_t a0;
    if (JAI_LIKELY(p->intArg)) {
        if (JAI_UNLIKELY(!IS_INT(arg))) {
            vm.stackTop = base;
            return jaiCallFn1(p->callee, arg, out);
        }
        a0 = AS_INT(arg);
    } else if (JAI_UNLIKELY(!jitArgIn(p->closure, base, 0, &a0))) {
        vm.stackTop = base;
        return jaiCallFn1(p->callee, arg, out);
    }

    int frameBase = vm.frameCount;
    JitResult r =
        p->nargs == 1
            ? ((Fn1)(uintptr_t)p->entry)(a0)
            : ((Fn2)(uintptr_t)p->entry)(a0, (int64_t)(uintptr_t)p->closure);

    /* An int result needs no store into the window and no read back out of it:
     * nothing between here and the caller's use of it can collect, and an int
     * is not a root in any case. Every other kind goes the long way, which is
     * where the tag and the bail verdicts are decided. */
    if (JAI_LIKELY(r.bailed == 0 && p->returnKind == (uint8_t)SLOT_INT)) {
        *out = INT_VAL(r.value);
        vm.stackTop = base;
        return true;
    }

    JaiJitOutcome outcome = jitResultOut(fn, r, base);
    if (JAI_LIKELY(outcome == JAI_JIT_DONE)) {
        *out = base[0];
        vm.stackTop = base;
        return true;
    }
    if (outcome == JAI_JIT_ERROR) {
        vm.stackTop = base;
        return false;
    }
    if (outcome == JAI_JIT_DEOPT) {
        return jaiFinishJitDeopt1(p->closure, base, frameBase, out);
    }
    return callFn1Rerun(base, out);
}

#else

JaiJitOutcome jaiJitEnterFunc(ObjClosure *closure, Value *slotBase) {
    (void)closure; (void)slotBase; return JAI_JIT_DECLINED;
}
bool jaiCallFn1(Value callee, Value arg, Value *out) {
    return jaiCallValue1(callee, arg, out);
}
void jaiPrepareFn1(Value callee, JaiPreparedFn1 *prepared) {
    prepared->callee = callee;
    prepared->flat   = false;
}
bool jaiCallPreparedFn1(JaiPreparedFn1 *prepared, Value arg, Value *out) {
    return jaiCallValue1(prepared->callee, arg, out);
}

#endif
