/* vm_call.c — call and frame machinery: building a frame window, binding
 * arguments (positional, defaulted, variadic and by keyword), and every entry
 * point native and compiled code uses to call back into the interpreter.
 */
#include "vm/vm_internal.h"
#include "vm/jit/jit.h"
#include "vm/trace/trace.h"
#include "runtime/runtime.h"

bool valueIsCallable(Value v) {
    if (!IS_OBJ(v)) return false;
    switch (OBJ_TYPE(v)) {
    case OBJ_CLOSURE:
    case OBJ_FUNCTION:
    case OBJ_NATIVE:
    case OBJ_BOUND:
    case OBJ_CLASS:
    case OBJ_ENUM_CTOR:
        return true;
    case OBJ_INSTANCE: {
        ObjClass *k = AS_INSTANCE(v)->klass;
        return k != NULL && !IS_NULL(k->dunderCall);
    }
    default:
        return false;
    }
}

/* Name to use for a callable in an error message. */
static const char *callableName(Value v) {
    if (IS_CLOSURE(v)) {
        ObjFunction *fn = AS_CLOSURE(v)->fn;
        if (fn != NULL && fn->name != NULL) return fn->name->chars;
        return "<anonymous>";
    }
    if (IS_FUNCTION(v) && AS_FUNCTION(v)->name != NULL) {
        return AS_FUNCTION(v)->name->chars;
    }
    if (IS_NATIVE(v) && AS_NATIVE(v)->name != NULL) return AS_NATIVE(v)->name->chars;
    if (IS_CLASS(v) && AS_CLASS(v)->name != NULL) return AS_CLASS(v)->name->chars;
    if (IS_BOUND(v)) return callableName(AS_BOUND(v)->method);
    return "<callable>";
}

bool ensureStack(int extra) {
    if (vm.stack == NULL) return false;
    if (vm.stackTop + extra <= vm.stack + JAI_STACK_MAX) return true;
    return jaiThrow(vm.cRuntimeError, "value stack overflow (%d slots)",
                    JAI_STACK_MAX);
}

/* ------------------------------------------------------------------ */
/* Upvalues                                                             */
/* ------------------------------------------------------------------ */

ObjUpvalue *captureUpvalue(Value *local) {
    ObjUpvalue *prev = NULL;
    ObjUpvalue *upvalue = vm.openUpvalues;
    while (upvalue != NULL && upvalue->location > local) {
        prev = upvalue;
        upvalue = upvalue->next;
    }
    if (upvalue != NULL && upvalue->location == local) return upvalue;

    ObjUpvalue *created = jaiUpvalueNew(local);
    created->next = upvalue;
    if (prev == NULL) {
        vm.openUpvalues = created;
    } else {
        prev->next = created;
    }
    return created;
}

void closeUpvalues(Value *last) {
    while (vm.openUpvalues != NULL && vm.openUpvalues->location >= last) {
        ObjUpvalue *upvalue = vm.openUpvalues;
        upvalue->closed = *upvalue->location;
        upvalue->location = &upvalue->closed;
        vm.openUpvalues = upvalue->next;
    }
}

/* ------------------------------------------------------------------ */
/* Calls                                                                */
/*                                                                      */
/* Frame layout (spec §1): slots[0] is the callee or the receiver and    */
/* the declared parameters are slots[1..arity]. A variadic parameter     */
/* follows at slots[arity+1] and a keyword-rest dict after that. The     */
/* window is `maxSlots` wide and stackTop starts above it, so locals are */
/* always below stackTop and therefore always visible to the collector.  */
/* ------------------------------------------------------------------ */

int frameWindowSize(const ObjFunction *fn) {
    int declared = 1 + (int)fn->arity;
    if (fn->flags & FN_VARIADIC) declared++;
    if (fn->flags & FN_KWREST) declared++;
    return (int)fn->maxSlots > declared ? (int)fn->maxSlots : declared;
}

/* Where a callee's keyword-rest dict lives in its frame window. */
int kwRestSlotOf(const ObjFunction *fn) {
    return 1 + (int)fn->arity + ((fn->flags & FN_VARIADIC) ? 1 : 0);
}

bool pushFrame(ObjClosure *closure, Value *slotBase) {
    if (vm.frameCount >= vm.frameCapacity) {
        return jaiThrow(vm.cRecursionError,
                        "maximum recursion depth exceeded (%d frames)",
                        vm.frameCapacity);
    }
    CallFrame *frame = &vm.frames[vm.frameCount++];
    frame->closure = closure;
    frame->ip = closure->fn->chunk.code;
    frame->slots = slotBase;
    frame->base = slotBase;
    frame->handlerBase = vm.handlers.count;
    frame->deferBase = vm.defers.count;

    ObjModule *module = closure->fn->module;
    if (module == NULL) {
        CallFrame *caller = vm.frameCount > 1 ? &vm.frames[vm.frameCount - 2] : NULL;
        module = caller != NULL ? caller->module : vm.mainModule;
    }
    frame->module = module != NULL ? module : vm.builtins;
    vm.callCount++;
    return true;
}

/* Run one default-value thunk: a code region in the callee's own chunk that
 * leaves a value on the stack and ends with OP_RETURN. Spec §6 requires this
 * to happen on every call, which is what keeps a mutable default from being
 * shared between calls. */
static bool evalDefaultThunk(ObjClosure *closure, uint32_t codeOffset,
                             Value *out) {
    ObjFunction *fn = closure->fn;
    if (codeOffset >= (uint32_t)fn->chunk.count) {
        return jaiThrow(vm.cRuntimeError,
                        "corrupt default-value thunk in function '%s'",
                        fn->name != NULL ? fn->name->chars : "?");
    }
    int window = frameWindowSize(fn);
    if (!ensureRoom(vm.stackTop, window + JAI_FRAME_SLACK)) return false;

    Value *base = vm.stackTop;
    *vm.stackTop++ = OBJ_VAL(closure);
    for (int i = 1; i < window; i++) *vm.stackTop++ = NULL_VAL;

    int frameBase = vm.frameCount;
    if (!pushFrame(closure, base)) {
        vm.stackTop = base;
        return false;
    }
    vm.frames[vm.frameCount - 1].ip = fn->chunk.code + codeOffset;

    int savedThunkFrame = sThunkFrame;
    sThunkFrame = vm.frameCount - 1;
    JaiRunResult result = run(frameBase);
    sThunkFrame = savedThunkFrame;
    if (result != JAI_RUN_OK) {
        vm.stackTop = base;
        return false;
    }
    *out = *(--vm.stackTop);
    vm.stackTop = base;
    return true;
}

/* Turn `argc` positional arguments sitting above the callee slot into a fully
 * populated frame window. Returns false with an exception pending. */
/* Everything the fast path in bindCallArgs below could not handle: a wrong
 * argument count, a variadic tail, a default to evaluate, a keyword-rest dict.
 * Out of line on purpose — it is large, mostly diagnostics, and inlining it
 * was what stopped clang inlining the fast path with it. */
bool bindCallArgsSlow(ObjClosure *closure, int argc, Value *slotBase) {
    ObjFunction *fn = closure->fn;
    int arity = (int)fn->arity;

    bool variadic = (fn->flags & FN_VARIADIC) != 0;
    int required = arity - (int)fn->defaultCount;
    if (required < 0) required = 0;

    if (argc < required || (!variadic && argc > arity)) {
        const char *name = fn->name != NULL ? fn->name->chars : "<anonymous>";
        if (required == arity && !variadic) {
            return jaiThrow(vm.cTypeError,
                            "'%s' takes %d argument%s but %d %s given", name,
                            arity, arity == 1 ? "" : "s", argc,
                            argc == 1 ? "was" : "were");
        }
        if (argc < required) {
            return jaiThrow(vm.cTypeError,
                            "'%s' takes at least %d argument%s but %d %s given",
                            name, required, required == 1 ? "" : "s", argc,
                            argc == 1 ? "was" : "were");
        }
        return jaiThrow(vm.cTypeError,
                        "'%s' takes at most %d argument%s but %d were given",
                        name, arity, arity == 1 ? "" : "s", argc);
    }

    int window = frameWindowSize(fn);
    if (!ensureRoom(slotBase, window + JAI_FRAME_SLACK)) return false;

    /* Pack the variadic tail first, while the extra arguments are still live
     * on the stack and therefore visible to a collection. */
    ObjList *rest = NULL;
    if (variadic) {
        int extra = argc > arity ? argc - arity : 0;
        rest = jaiListNew(extra);
        for (int i = 0; i < extra; i++) jaiListPut(rest, i, slotBase[1 + arity + i]);
        rest->count = extra;
        if (argc > arity) argc = arity;
    }

    /* Every slot the frame owns must hold a real Value before anything can
     * allocate: the collector scans the whole window once stackTop is above it.
     * Nothing between jaiListNew above and this fill allocates. */
    for (int i = 1 + argc; i < window; i++) slotBase[i] = NULL_VAL;
    if (rest != NULL) slotBase[arity + 1] = OBJ_VAL(rest);
    vm.stackTop = slotBase + window;

    for (int i = argc; i < arity; i++) {
        int thunk = i - required;
        if (fn->defaultOffsets == NULL || thunk < 0 ||
            thunk >= (int)fn->defaultCount) {
            continue;                       /* already NULL_VAL */
        }
        Value def;
        if (!evalDefaultThunk(closure, fn->defaultOffsets[thunk], &def)) {
            return false;
        }
        slotBase[i + 1] = def;
    }

    if (fn->flags & FN_KWREST) {
        slotBase[kwRestSlotOf(fn)] = OBJ_VAL(jaiDictNew());
    }
    vm.stackTop = slotBase + window;
    return true;
}

/* CALL_FRAME when a frame was pushed and the interpreter must run it,
 * CALL_DONE when the compiled tier finished the call outright.
 *
 * The distinction is not cosmetic. jaiCallValue runs the interpreter whenever
 * it is told a frame was pushed, so reporting CALL_FRAME for a call the tier
 * had already completed made it run the CALLER's frame a second time -- which
 * is how `xs.map(|x| x * 2)` came back holding an int instead of a list. The
 * interpreter's own OP_CALL never saw it, because LOAD_STATE reloads whatever
 * frame is on top and that happened to be the right one. */
CallOutcome callClosure(ObjClosure *closure, int argc) {
    Value *slotBase = vm.stackTop - argc - 1;
    ObjFunction *fn = closure->fn;
    bool traced = (fn->flags & FN_TRACE) != 0;
    if (traced) jaiTraceEnter(fn);

    /* Ahead of bindCallArgs on purpose. Compiled code never reads the VM
     * stack, so clearing a frame window it will not look at, and growing the
     * stack for slots it will not use, is pure cost -- and on a small hot
     * function called from an interpreted loop that entry cost was most of
     * what the tier had to give. */
    if (fn->jitFunc != NULL && argc == (int)fn->arity) {
        JaiJitOutcome outcome = jaiJitEnterFunc(closure, slotBase);
        if (outcome == JAI_JIT_DONE) {
            if (traced) jaiTraceLeave(fn);
            return CALL_DONE;
        }
        /* An exception from a call the compiled body made: the effects up to
         * it already happened, so this is not a decline. */
        if (outcome == JAI_JIT_ERROR) {
            if (traced) jaiTraceLeave(fn);
            return CALL_ERROR;
        }
        if (outcome == JAI_JIT_DEOPT) {
            /* A guard failed part-way in. The interpreter takes over from that
             * exact instruction, holding what the compiled body held, so the
             * work already done is neither lost nor repeated. */
            if (!bindCallArgs(closure, argc, slotBase)) {
                if (traced) jaiTraceLeave(fn);
                return CALL_ERROR;
            }
            if (!pushFrame(closure, slotBase)) {
                if (traced) jaiTraceLeave(fn);
                return CALL_ERROR;
            }
            if (!jaiJitApplyDeopt(closure, slotBase)) {
                if (traced) jaiTraceLeave(fn);
                return CALL_ERROR;
            }
            return CALL_FRAME;
        }
    }

    if (!bindCallArgs(closure, argc, slotBase)) {
        if (traced) jaiTraceLeave(fn);
        return CALL_ERROR;
    }
    if (fn->entryCount < jaiJitThreshold(fn)) {
        fn->entryCount++;
    } else if (!fn->jitRefused && jaiJitEnabled()) {
        /* Every verdict, not just DONE. The body that just compiled may have
         * run and then deoptimised, in which case falling through to the
         * pushFrame below would run the function a SECOND time from the top --
         * see jaiJitEnter. bindCallArgs is already done, which is the one
         * thing the branch above this has to do for itself. */
        JaiJitOutcome outcome = jaiJitEnter(closure, slotBase);
        if (outcome == JAI_JIT_DONE) {
            if (traced) jaiTraceLeave(fn);
            return CALL_DONE;
        }
        if (outcome == JAI_JIT_ERROR) {
            if (traced) jaiTraceLeave(fn);
            return CALL_ERROR;
        }
        if (outcome == JAI_JIT_DEOPT) {
            if (!pushFrame(closure, slotBase)) {
                if (traced) jaiTraceLeave(fn);
                return CALL_ERROR;
            }
            if (!jaiJitApplyDeopt(closure, slotBase)) {
                if (traced) jaiTraceLeave(fn);
                return CALL_ERROR;
            }
            return CALL_FRAME;
        }
    }

    if (!pushFrame(closure, slotBase)) {
        if (traced) jaiTraceLeave(fn);
        return CALL_ERROR;
    }
    return CALL_FRAME;
}

/* `args` and `count` differ from the raw stack window for bound natives, whose
 * receiver occupies the callee slot and is passed as args[0]. */
bool callNativeAt(ObjNative *native, Value *args, int count,
                         Value *resultSlot) {
    int minArity = native->minArity;
    int maxArity = native->maxArity;
    const char *name = native->name != NULL ? native->name->chars : "<native>";

    if (count < minArity || (maxArity >= 0 && count > maxArity)) {
        if (maxArity == minArity) {
            return jaiThrow(vm.cTypeError,
                            "'%s' takes %d argument%s but %d %s given", name,
                            minArity, minArity == 1 ? "" : "s", count,
                            count == 1 ? "was" : "were");
        }
        return jaiThrow(vm.cTypeError,
                        "'%s' takes %d to %d arguments but %d were given", name,
                        minArity, maxArity, count);
    }

    Value result = NULL_VAL;
    if (!native->fn(count, args, &result)) {
        if (!vm.hasException) {
            (void)jaiThrow(vm.cRuntimeError, "native function '%s' failed", name);
        }
        return false;
    }
    *resultSlot = result;
    return true;
}

/* Call `callable` where the slot at vm.stackTop - argc - 1 is already the
 * frame's slot 0 (the callee for a plain function, the receiver for a method).
 * On CALL_DONE the result has replaced that slot and everything above it. */
CallOutcome invokeCallable(Value callable, int argc) {
    if (!IS_OBJ(callable)) {
        (void)jaiThrow(vm.cTypeError, "'%s' value is not callable",
                       jaiTypeNameStatic(callable));
        return CALL_ERROR;
    }

    Value *slot = vm.stackTop - argc - 1;

    switch (OBJ_TYPE(callable)) {
    case OBJ_CLOSURE:
        return callClosure(AS_CLOSURE(callable), argc);

    case OBJ_FUNCTION: {
        /* A bare function reaching a call site has no upvalues by
         * construction; wrap it so the frame protocol stays uniform. */
        ObjClosure *closure = jaiClosureNew(AS_FUNCTION(callable));
        return callClosure(closure, argc);
    }

    case OBJ_NATIVE: {
        Value result;
        if (!callNativeAt(AS_NATIVE(callable), vm.stackTop - argc, argc,
                          &result)) {
            return CALL_ERROR;
        }
        vm.stackTop = slot;
        *vm.stackTop++ = result;
        return CALL_DONE;
    }

    case OBJ_BOUND: {
        ObjBound *bound = AS_BOUND(callable);
        *slot = bound->receiver;
        if (IS_NATIVE(bound->method)) {
            /* Built-in methods take the receiver as args[0]. */
            Value result;
            if (!callNativeAt(AS_NATIVE(bound->method), slot, argc + 1,
                              &result)) {
                return CALL_ERROR;
            }
            vm.stackTop = slot;
            *vm.stackTop++ = result;
            return CALL_DONE;
        }
        return invokeCallable(bound->method, argc);
    }

    case OBJ_CLASS: {
        ObjClass *klass = AS_CLASS(callable);
        if (klass->isAbstract) {
            (void)jaiThrow(vm.cTypeError, "cannot instantiate abstract class '%s'",
                           klass->name != NULL ? klass->name->chars : "?");
            return CALL_ERROR;
        }
        ObjInstance *instance = jaiInstanceNew(klass);
        slot = vm.stackTop - argc - 1;      /* the allocation may have collected */
        *slot = OBJ_VAL(instance);

        if (IS_NULL(klass->initializer)) {
            if (argc != 0) {
                (void)jaiThrow(vm.cTypeError,
                               "class '%s' has no init but %d argument%s given",
                               klass->name != NULL ? klass->name->chars : "?",
                               argc, argc == 1 ? " was" : "s were");
                return CALL_ERROR;
            }
            vm.stackTop = slot + 1;
            return CALL_DONE;
        }
        /* OP_RETURN substitutes slot 0 for an initializer's result, so the
         * constructed instance is what the call expression yields. */
        if (IS_NATIVE(klass->initializer)) {
            /* A native init is a method: it takes the receiver as args[0] and
             * counts it in argc, exactly as the OBJ_BOUND path does. Going
             * through invokeCallable would hand it the arguments alone and
             * strand `self`. Its return value is discarded for the same reason
             * OP_RETURN discards a closure init's. */
            Value ignored;
            if (!callNativeAt(AS_NATIVE(klass->initializer), slot, argc + 1,
                              &ignored)) {
                return CALL_ERROR;
            }
            vm.stackTop = slot + 1;
            return CALL_DONE;
        }
        return invokeCallable(klass->initializer, argc);
    }

    case OBJ_ENUM_CTOR: {
        ObjEnumCtor *ctor = AS_ENUM_CTOR(callable);
        ObjEnum *type = ctor->type;
        EnumVariant *variant = &type->variants[ctor->tag];
        if (argc != (int)variant->arity) {
            (void)jaiThrow(vm.cTypeError,
                           "%s.%s() takes %d argument%s but %d %s given",
                           type->name != NULL ? type->name->chars : "?",
                           variant->name != NULL ? variant->name->chars : "?",
                           (int)variant->arity, variant->arity == 1 ? "" : "s",
                           argc, argc == 1 ? "was" : "were");
            return CALL_ERROR;
        }
        ObjEnumVal *built = jaiEnumValNew(type, ctor->tag, vm.stackTop - argc, argc);
        slot = vm.stackTop - argc - 1;      /* the allocation may have collected */
        vm.stackTop = slot;
        *vm.stackTop++ = OBJ_VAL(built);
        return CALL_DONE;
    }

    case OBJ_INSTANCE: {
        ObjClass *klass = AS_INSTANCE(callable)->klass;
        if (klass == NULL || IS_NULL(klass->dunderCall)) break;
        return invokeCallable(klass->dunderCall, argc);
    }

    default:
        break;
    }

    (void)jaiThrow(vm.cTypeError, "'%s' value is not callable",
                   jaiTypeNameStatic(callable));
    return CALL_ERROR;
}

/* Same, for a site where slot 0 holds the *receiver* of a method call rather
 * than the callee. Every callable shape reads slot 0 that way already, with one
 * exception: a raw ObjNative. invokeCallable has to assume a plain call for it,
 * where slot 0 is the function itself and the arguments start above — so it
 * hands the native args[0] = the first argument and drops `self` on the floor.
 * A native found in a class's method table is a built-in method and wants the
 * receiver as args[0], counted in argc, exactly as the OBJ_BOUND path does. */
CallOutcome invokeMethodOnStack(Value callable, int argc) {
    if (!IS_NATIVE(callable)) return invokeCallable(callable, argc);

    Value *slot = vm.stackTop - argc - 1;
    Value result;
    if (!callNativeAt(AS_NATIVE(callable), slot, argc + 1, &result)) {
        return CALL_ERROR;
    }
    vm.stackTop = slot;
    *vm.stackTop++ = result;
    return CALL_DONE;
}

/* Entry point for a `CALL`-shaped site: the callee itself occupies slot 0
 * unless it is a bound method or a class, which invokeCallable rewrites. */
CallOutcome callValueOnStack(int argc) {
    Value callee = vm.stackTop[-argc - 1];
    return invokeCallable(callee, argc);
}

/* Call a built-in method with the receiver already in args[0], which is where
 * one wants it. Exposed for the compiled tier: going through jaiCallValue
 * would mean building an ObjBound per call, and `xs.len()` in a loop cannot
 * afford an allocation to ask a question. */
/* Call a method with the receiver in slot 0, which is where a method's body
 * expects `self`. jaiCallValue puts the callee there instead, so it cannot be
 * used for this. `count` includes the receiver. */
/* The method `name` resolves to on `klass`, following the inheritance chain.
 * Exposed for the compiled tier, which resolves a call site once. */
bool jaiClassFindMethod(ObjClass *klass, ObjString *name, Value *out) {
    return findMethod(klass, name, out);
}

bool jaiCallMethodWithReceiver(Value method, Value *argsWithReceiver, int count,
                               Value *out) {
    if (count < 1) return false;
    if (!ensureStack(count + 1)) return false;

    Value *base = vm.stackTop;
    for (int i = 0; i < count; i++) *vm.stackTop++ = argsWithReceiver[i];

    int frameBase = vm.frameCount;
    CallOutcome outcome = invokeMethodOnStack(method, count - 1);
    if (outcome == CALL_ERROR) { vm.stackTop = base; return false; }
    if (outcome == CALL_FRAME && run(frameBase) != JAI_RUN_OK) {
        vm.stackTop = base;
        return false;
    }
    *out = *(--vm.stackTop);
    vm.stackTop = base;
    return true;
}

bool jaiInvokeNativeWithReceiver(Value native, Value *argsWithReceiver,
                                 int count, Value *out) {
    if (!IS_NATIVE(native)) return false;
    return callNativeAt(AS_NATIVE(native), argsWithReceiver, count, out);
}

bool jaiCallValue(Value callee, int argc, Value *args, Value *out) {
    if (argc < 0) argc = 0;
    if (!ensureStack(argc + 2)) return false;

    Value *base = vm.stackTop;
    *vm.stackTop++ = callee;
    for (int i = 0; i < argc; i++) vm.stackTop[i] = args[i];
    vm.stackTop += argc;

    int frameBase = vm.frameCount;
    CallOutcome outcome = invokeCallable(callee, argc);
    if (outcome == CALL_ERROR) {
        vm.stackTop = base;
        return false;
    }
    if (outcome == CALL_FRAME && run(frameBase) != JAI_RUN_OK) {
        vm.stackTop = base;
        return false;
    }
    *out = *(--vm.stackTop);
    vm.stackTop = base;
    return true;
}

/* See jit.h. A compiled self-call is a bare `bl`: the callee's arguments never
 * reached the VM stack, so its frame is built here from nothing but the deopt
 * record, which carries every local -- parameters included -- and the operand
 * stack as they stood at the guard that failed. Slot 0 is the closure, which
 * a self-call knows because it is its own. */
bool jaiJitFinishDeopt(ObjClosure *closure, Value *out) {
    ObjFunction *fn = closure->fn;
    int window = frameWindowSize(fn);
    if (!ensureRoom(vm.stackTop, window + JAI_FRAME_SLACK)) return false;

    Value *base = vm.stackTop;
    *vm.stackTop++ = OBJ_VAL(closure);
    for (int i = 1; i < window; i++) *vm.stackTop++ = NULL_VAL;

    int frameBase = vm.frameCount;
    if (!pushFrame(closure, base) || !jaiJitApplyDeopt(closure, base) ||
        run(frameBase) != JAI_RUN_OK) {
        vm.stackTop = base;
        return false;
    }
    *out = *(--vm.stackTop);
    vm.stackTop = base;
    return true;
}

/* See vm.h. The two stack cells stay: jaiJitEnterFunc reads its arguments from
 * `base[jitArgBase + i]` and writes the result to `base[0]`, and while the
 * compiled body runs those two cells are the only thing keeping the closure
 * and the argument reachable -- a compiled body may allocate, and a collection
 * scans the VM stack.
 *
 * A deopt is handled here rather than fallen through to jaiCallValue, and that
 * is not a nicety: the compiled body ran partway and may have written, so
 * re-entering the call from the top would repeat those writes. It is the same
 * sequence callClosure runs for JAI_JIT_DEOPT, with the interpreter driven to
 * completion the way jaiCallValue would have. */
bool jaiCallValue1(Value callee, Value arg, Value *out) {
    if (JAI_LIKELY(IS_CLOSURE(callee))) {
        ObjClosure *closure = AS_CLOSURE(callee);
        ObjFunction *fn = closure->fn;
        if (fn->jitFunc != NULL && fn->arity == 1) {
            if (!ensureStack(3)) return false;
            Value *base = vm.stackTop;
            base[0] = callee;
            base[1] = arg;
            vm.stackTop = base + 2;

            int frameBase = vm.frameCount;
            JaiJitOutcome outcome = jaiJitEnterFunc(closure, base);
            if (outcome == JAI_JIT_DONE) {
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
            vm.stackTop = base;   /* declined; the general path below */
        }
    }
    return jaiCallValue(callee, 1, &arg, out);
}

/* See vm.h. Split out of the branch above so that a caller which entered the
 * compiled body for itself -- jaiCallFn1, which exists so the boundary is one
 * C frame rather than three -- can finish the same way without duplicating
 * the sequence or reaching for vm.c's statics. */
bool jaiFinishJitDeopt1(ObjClosure *closure, Value *base, int frameBase,
                        Value *out) {
    vm.stackTop = base + 2;
    if (!bindCallArgs(closure, 1, base) ||
        !pushFrame(closure, base) ||
        !jaiJitApplyDeopt(closure, base) ||
        run(frameBase) != JAI_RUN_OK) {
        vm.stackTop = base;
        return false;
    }
    *out = *(--vm.stackTop);
    vm.stackTop = base;
    return true;
}

/* Resolve `receiver.name` to something callable with the receiver in slot 0.
 * Returns false *without* an exception when there is simply no such method:
 * callers such as value.c use that to fall back to a default behaviour.
 *
 * `*isMethod` says whether slot 0 ends up holding the receiver, which is what
 * decides between invokeMethodOnStack and invokeCallable. Only a class's own
 * method table can hand back a bare ObjNative, so only that path sets it; every
 * other branch either yields a closure, where slot 0 is the frame's slot 0
 * either way, or a bound method that carries its own receiver. */
bool resolveInvokeTarget(Value receiver, ObjString *name, Value *method,
                                Value *slotZero, bool *isMethod) {
    *slotZero = receiver;
    *isMethod = false;

    if (IS_INSTANCE(receiver)) {
        ObjInstance *inst = AS_INSTANCE(receiver);
        if (findMethod(inst->klass, name, method)) {
            /* Raises on a denied private, and the callers below distinguish
             * "no such method" from a pending exception — reporting a typo
             * for a method that plainly exists would misdirect the reader. */
            *isMethod = true;
            return methodPermitted(inst->klass, name, true);
        }
        /* A field holding a callable is invocable too. It is a plain call, not
         * a method call: the field goes in slot 0 and gets no receiver. */
        const FieldInfo *info = jaiClassFieldInfo(inst->klass, name);
        if (info != NULL && info->slot < inst->fieldCount) {
            Value field = inst->fields[info->slot];
            if (valueIsCallable(field)) {
                *method = field;
                *slotZero = field;
                return true;
            }
        }
        return false;
    }

    Value member;
    if (IS_CLASS(receiver)) {
        ObjClass *klass = AS_CLASS(receiver);
        if (jaiTableGetInterned(&klass->statics, name, &member) ||
            jaiTableGetInterned(&klass->methods, name, &member)) {
            if (!methodPermitted(klass, name, true)) return false;
            *method = member;
            return true;
        }
        return false;
    }
    if (IS_MODULE(receiver)) {
        /* jaiModuleMethod applies the same export rule, adds the introspection
         * helpers, and tells "present but private" from "no such name": the
         * first raises E0802 rather than reporting a missing method, which
         * would send the reader hunting for a typo. */
        if (jaiBuiltinMethod(receiver, name, &member)) {
            *method = member;
            return true;
        }
        return false;
    }
    if (IS_ENUM(receiver) || IS_ENUM_VAL(receiver)) {
        if (getPropertyInto(receiver, name, &member, false, NULL)) {
            *method = member;
            return true;
        }
        return false;
    }

    /* Primitive receivers: jaiBuiltinMethod already binds the receiver. */
    if (jaiBuiltinMethod(receiver, name, &member)) {
        *method = member;
        return true;
    }
    return false;
}

bool jaiInvokeMethod(Value receiver, ObjString *name, int argc, Value *args,
                     Value *out) {
    if (name == NULL) return false;
    Value method, slotZero;
    bool isMethod;
    if (!resolveInvokeTarget(receiver, name, &method, &slotZero, &isMethod))
        return false;
    if (argc < 0) argc = 0;
    if (!ensureStack(argc + 2)) return false;

    Value *base = vm.stackTop;
    *vm.stackTop++ = slotZero;
    for (int i = 0; i < argc; i++) vm.stackTop[i] = args[i];
    vm.stackTop += argc;

    int frameBase = vm.frameCount;
    CallOutcome outcome = isMethod ? invokeMethodOnStack(method, argc)
                                   : invokeCallable(method, argc);
    if (outcome == CALL_ERROR) {
        vm.stackTop = base;
        return false;
    }
    if (outcome == CALL_FRAME && run(frameBase) != JAI_RUN_OK) {
        vm.stackTop = base;
        return false;
    }
    *out = *(--vm.stackTop);
    vm.stackTop = base;
    return true;
}

/* Invoke by name on a receiver whose class the caller could not pin.
 *
 * The compiled tier resolves a method while it emits when the receiver's class
 * is fixed, which is nearly every call site. A loop over a list holding several
 * implementations of one trait is where it is not, and before this such a loop
 * declined outright and ran interpreted. The shared megamorphic table answers
 * here for the reason it answers in OP_INVOKE: one table sized for the whole
 * program makes the ninth class cost what the fifth does.
 *
 * Only a hit in `klass->methods` is cached, checked against that table's
 * version, so nothing has to remember to invalidate this -- the same contract
 * the interpreter's megamorphic arm keeps, and statics and trait defaults
 * simply take the general path below.
 *
 * Raises rather than returning a bare false for a missing method: the caller is
 * compiled code with no name or receiver left to build a message from. */
bool jaiInvokeMethodByName(ObjString *name, Value *argsWithReceiver, int count,
                           Value *out) {
    if (name == NULL || count < 1) return false;
    const Value receiver = argsWithReceiver[0];

    if (IS_INSTANCE(receiver)) {
        ObjClass *klass = AS_INSTANCE(receiver)->klass;
        if (klass != NULL) {
            MegaEntry *me = &sMegaCache[megaSlot(klass, name)];
            if (me->klass == klass && me->name == name &&
                me->tableVersion == klass->methods.version) {
                vm.icHits++;
                if (me->recheck && !methodPermitted(klass, name, true)) {
                    return false;
                }
                return jaiCallMethodWithReceiver(me->method, argsWithReceiver,
                                                 count, out);
            }
            Value found;
            if (jaiTableGetInterned(&klass->methods, name, &found)) {
                MethodInfo mi;
                vm.icMisses++;
                me->klass        = klass;
                me->name         = name;
                me->method       = found;
                me->tableVersion = klass->methods.version;
                me->recheck      = jaiClassRestrictedMethod(klass, name, &mi);
                if (me->recheck && !methodPermitted(klass, name, true)) {
                    return false;
                }
                return jaiCallMethodWithReceiver(found, argsWithReceiver, count,
                                                 out);
            }
        }
    }

    if (jaiInvokeMethod(receiver, name, count - 1, argsWithReceiver + 1, out)) {
        return true;
    }
    if (vm.hasException) return false;
    return jaiThrow(vm.cAttributeError, "'%s' object has no method '%s'",
                    jaiTypeNameStatic(receiver), name->chars);
}

/* ------------------------------------------------------------------ */
/* Keyword and spread calls                                             */
/* ------------------------------------------------------------------ */

static bool targetFunctionOf(Value callee, ObjFunction **out) {
    if (IS_CLOSURE(callee)) { *out = AS_CLOSURE(callee)->fn; return true; }
    if (IS_FUNCTION(callee)) { *out = AS_FUNCTION(callee); return true; }
    if (IS_BOUND(callee)) return targetFunctionOf(AS_BOUND(callee)->method, out);
    if (IS_CLASS(callee)) {
        ObjClass *klass = AS_CLASS(callee);
        if (!IS_NULL(klass->initializer)) {
            return targetFunctionOf(klass->initializer, out);
        }
    }
    return false;
}

/* The bound native behind `f(key: v)`, or NULL. Only a native that declares
 * parameter names can be called by keyword; the rest have no names to bind to
 * and say so through the TypeError below. */
static ObjNative *targetNativeOf(Value callee) {
    Value inner = callee;
    while (IS_BOUND(inner)) inner = AS_BOUND(inner)->method;
    if (!IS_NATIVE(inner)) return NULL;
    ObjNative *native = AS_NATIVE(inner);
    /* A variadic native has no fixed parameter list to name positions in. */
    if (native->paramNames == NULL || native->maxArity < 0) return NULL;
    return native;
}

/* Spec §6 lets any callee be called by name, but a Jaithon function and a
 * native reach their arguments differently: the first gets a frame window
 * whose holes are filled by default-value thunks, the second gets a flat
 * argument array whose optional slots read as absent when they are null. This
 * is what a keyword call needs to know about either one. */
typedef struct {
    ObjFunction       *fn;         /* NULL for a native */
    ObjNative         *native;     /* NULL for a Jaithon function */
    const char        *name;
    const char *const *nativeNames; /* args[0..] names; args[0] may be `self` */
    int                selfOffset;  /* 1 when args[0] is a bound receiver */
    int                arity;       /* positional parameters, defaults included */
    int                required;
    bool               variadic;
    bool               kwRest;
} CalleeShape;

/* Index of `name` among the callee's declared parameters, or -1. The array is
 * parallel to the argument positions the call has to end up producing. */
static int parameterIndex(const CalleeShape *shape, ObjString *name) {
    if (shape->fn != NULL) {
        const ObjFunction *fn = shape->fn;
        if (fn->paramNames == NULL) return -1;
        for (uint16_t i = 0; i < fn->paramCount; i++) {
            if (fn->paramNames[i] == name ||
                jaiStringEquals(fn->paramNames[i], name)) {
                return (int)i;
            }
        }
        return -1;
    }
    for (int i = 0; i < shape->arity; i++) {
        const char *declared = shape->nativeNames[i + shape->selfOffset];
        if (declared != NULL && strcmp(declared, name->chars) == 0) return i;
    }
    return -1;
}

static const char *parameterName(const CalleeShape *shape, int index) {
    if (shape->fn != NULL) {
        const ObjFunction *fn = shape->fn;
        if (fn->paramNames == NULL || index >= (int)fn->paramCount) return "?";
        return fn->paramNames[index] != NULL ? fn->paramNames[index]->chars : "?";
    }
    const char *declared = shape->nativeNames[index + shape->selfOffset];
    return declared != NULL ? declared : "?";
}

/* Turn `f(a, b, key: v)` into the flat positional window the frame protocol
 * wants. Missing middle parameters get their defaults here rather than in
 * bindCallArgs, which can only fill a contiguous trailing run.
 *
 * Returns the new argument count, or -1 with an exception pending. */
int prepareKeywordCall(int posCount, ObjTuple *names,
                              ObjDict **outKwRest) {
    int kwCount = (int)names->count;
    int total = posCount + kwCount;
    Value *argBase = vm.stackTop - total;
    Value callee = argBase[-1];
    *outKwRest = NULL;

    CalleeShape shape;
    memset(&shape, 0, sizeof shape);
    ObjFunction *fn = NULL;
    if (targetFunctionOf(callee, &fn)) {
        shape.fn = fn;
        shape.name = fn->name != NULL ? fn->name->chars : "<anonymous>";
        shape.arity = (int)fn->arity;
        shape.required = (int)fn->arity - (int)fn->defaultCount;
        shape.variadic = (fn->flags & FN_VARIADIC) != 0;
        shape.kwRest = (fn->flags & FN_KWREST) != 0;
    } else if ((shape.native = targetNativeOf(callee)) != NULL) {
        ObjNative *native = shape.native;
        shape.name = native->name != NULL ? native->name->chars : "<native>";
        shape.nativeNames = native->paramNames;
        shape.selfOffset = IS_BOUND(callee) ? 1 : 0;
        shape.arity = (int)native->maxArity - shape.selfOffset;
        shape.required = (int)native->minArity - shape.selfOffset;
    } else {
        (void)jaiThrow(vm.cTypeError,
                       "'%s' does not accept keyword arguments",
                       callableName(callee));
        return -1;
    }
    if (shape.arity < 0) shape.arity = 0;
    if (shape.required < 0) shape.required = 0;

    int arity = shape.arity;
    if (posCount > arity && !shape.variadic) {
        (void)jaiThrow(vm.cTypeError,
                       "'%s' takes %d argument%s but %d positional were given",
                       shape.name, arity, arity == 1 ? "" : "s", posCount);
        return -1;
    }

    Value filled[JAI_MAX_ARGS];
    bool  present[JAI_MAX_ARGS];
    int   bound = arity < JAI_MAX_ARGS ? arity : JAI_MAX_ARGS;
    for (int i = 0; i < bound; i++) {
        filled[i] = NULL_VAL;
        present[i] = false;
    }

    int positionalKept = posCount < arity ? posCount : arity;
    for (int i = 0; i < positionalKept; i++) {
        filled[i] = argBase[i];
        present[i] = true;
    }

    ObjDict *kwRest = NULL;
    if (shape.kwRest) {
        kwRest = jaiDictNew();
        jaiGCPushRoot(OBJ_VAL(kwRest));
        argBase = vm.stackTop - total;   /* the allocation may have collected */
    }

    for (int i = 0; i < kwCount; i++) {
        Value nameValue = names->items[i];
        Value argument = argBase[posCount + i];
        if (!IS_STRING(nameValue)) {
            if (kwRest != NULL) jaiGCPopRoot();
            (void)jaiThrow(vm.cTypeError, "keyword argument names must be str");
            return -1;
        }
        ObjString *name = AS_STRING(nameValue);
        int index = parameterIndex(&shape, name);
        if (index < 0 || index >= bound) {
            if (kwRest != NULL) {
                (void)jaiDictSet(kwRest, nameValue, argument);
                continue;
            }
            (void)jaiThrow(vm.cTypeError,
                           "'%s' got an unexpected keyword argument '%s'",
                           shape.name, name->chars);
            return -1;
        }
        if (present[index]) {
            if (kwRest != NULL) jaiGCPopRoot();
            (void)jaiThrow(vm.cTypeError,
                           "'%s' got multiple values for argument '%s'",
                           shape.name, name->chars);
            return -1;
        }
        filled[index] = argument;
        present[index] = true;
    }

    int required = shape.required;
    /* Evaluated defaults are held on the stack as well as in `filled`, since
     * the next thunk can allocate and `filled` is invisible to the collector. */
    int rooted = 0;
    if (!ensureRoom(vm.stackTop, arity + JAI_FRAME_SLACK)) {
        if (kwRest != NULL) jaiGCPopRoot();
        return -1;
    }
    for (int i = 0; i < bound; i++) {
        if (present[i]) continue;
        if (i < required ||
            (shape.fn != NULL && shape.fn->defaultOffsets == NULL)) {
            vm.stackTop -= rooted;
            if (kwRest != NULL) jaiGCPopRoot();
            (void)jaiThrow(vm.cTypeError,
                           "'%s' is missing required argument '%s'",
                           shape.name, parameterName(&shape, i));
            return -1;
        }
        /* A native reads an omitted optional argument as null, so the hole it
         * left is already filled; only a Jaithon function has a thunk to run,
         * and spec §6 says it runs at the call. */
        if (shape.fn == NULL) {
            present[i] = true;
            continue;
        }
        ObjClosure *closure = IS_CLOSURE(callee) ? AS_CLOSURE(callee) : NULL;
        if (closure == NULL) {
            Value inner = callee;
            while (IS_BOUND(inner)) inner = AS_BOUND(inner)->method;
            if (IS_CLASS(inner)) inner = AS_CLASS(inner)->initializer;
            if (IS_CLOSURE(inner)) closure = AS_CLOSURE(inner);
        }
        if (closure == NULL) {
            vm.stackTop -= rooted;
            if (kwRest != NULL) jaiGCPopRoot();
            (void)jaiThrow(vm.cTypeError,
                           "cannot evaluate the default for '%s' on this callee",
                           parameterName(&shape, i));
            return -1;
        }
        Value def;
        if (!evalDefaultThunk(closure, shape.fn->defaultOffsets[i - required],
                              &def)) {
            vm.stackTop -= rooted;
            if (kwRest != NULL) jaiGCPopRoot();
            return -1;
        }
        filled[i] = def;
        present[i] = true;
        *vm.stackTop++ = def;
        rooted++;
    }

    /* Rewrite the window: arity values in declared order, then any variadic
     * overflow that came in positionally. */
    vm.stackTop -= rooted;
    argBase = vm.stackTop - total;
    int extra = posCount > arity ? posCount - arity : 0;
    Value overflow[JAI_MAX_ARGS];
    for (int i = 0; i < extra && i < JAI_MAX_ARGS; i++) {
        overflow[i] = argBase[arity + i];
    }
    for (int i = 0; i < bound; i++) argBase[i] = filled[i];
    for (int i = 0; i < extra && i < JAI_MAX_ARGS; i++) {
        argBase[bound + i] = overflow[i];
    }
    vm.stackTop = argBase + bound + extra;

    if (kwRest != NULL) {
        jaiGCPopRoot();
        *outKwRest = kwRest;
    }
    return bound + extra;
}
