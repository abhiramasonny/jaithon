/* jit_entry.c -- the interpreter's doors into compiled code: argument marshalling,
 * result unpacking, and the flat one-argument call paths. */
#include "vm/jit/jit.h"

#include "vm/jit/jit_arm64.h"
/* For jaiJitFieldReadFor: which builtins are one load from their receiver. */
#include "vm/jit/jit_field_read.h"
#include "vm/gc.h"
/* For jaiBuiltinMethod: resolving `xs.len()` to a native needs the runtime's name table. */
#include "runtime/runtime.h"
/* For jaiOpBranchOperandAt: says which opcodes carry a branch target. */
#include "vm/bytecode/verify.h"
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
    if (r.bailed == 2) return JAI_JIT_ERROR;
    if (r.bailed == 4) return JAI_JIT_DEOPT;
    if (r.bailed) {
        /* Overflow or a low stack: nothing was written (the body cannot write), so handing the call back to
         * the interpreter is enough -- it raises the error with a traceback. Refused permanently so a bailing body isn't re-entered every call only to bail again. */
        fn->jitRefused = true;
        fn->jitFunc = NULL;
        return JAI_JIT_DECLINED;
    }

    if ((SlotKind)fn->jitReturnKind == SLOT_MAYBE_INST) {
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

JaiJitOutcome jaiJitEnterFunc(ObjClosure *closure, Value *slotBase) {
    ObjFunction *fn = closure->fn;
    if (fn->jitFunc == NULL) return JAI_JIT_DECLINED;

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
