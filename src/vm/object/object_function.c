/* object_function.c — the callable object kinds: ObjFunction (a compiled
 * function's constant/debug data — the JIT's own per-function state lives in
 * the struct too, but is written by jit.c/jit_func.c, not here), ObjUpvalue,
 * ObjClosure, ObjNative and ObjBound.
 *
 * All five are what a call instruction can end up invoking, directly or
 * through a bound receiver, and all five constructors are the same shape:
 * root the pieces being assembled, allocate, wire the fields, return. Nothing
 * here executes a call or knows about the interpreter's frame stack — that is
 * vm.c's job — so despite the name this file is exactly as inert as
 * object_collection.c's; it just happens to build the objects the *call*
 * opcodes act on rather than the ones the *data* opcodes do.
 */

#include "vm/object/object.h"
#include "vm/object/object_internal.h"   /* pushObjRoot */

#include "vm/gc.h"
#include "vm/table.h"
#include "vm/vm.h"

/* ------------------------------------------------------------------ */
/* Functions, closures, natives                                         */
/* ------------------------------------------------------------------ */

ObjFunction *jaiFunctionNew(void) {
    ObjFunction *fn = JAI_ALLOCATE_OBJ(ObjFunction, OBJ_FUNCTION);
    /* All scalar/pointer bookkeeping starts at zero/NULL by allocator contract. */
    jaiChunkInit(&fn->chunk, -1);
    return fn;
}

bool jaiFunctionIsDeferThunk(const ObjFunction *fn) {
    return fn != NULL && fn->name != NULL && fn->name->length == 5 &&
           memcmp(fn->name->chars, "defer", 5) == 0;
}

ObjUpvalue *jaiUpvalueNew(Value *slot) {
    ObjUpvalue *up = JAI_ALLOCATE_OBJ(ObjUpvalue, OBJ_UPVALUE);
    up->location = slot;
    up->closed = NULL_VAL;
    return up;
}

ObjUpvalue *jaiUpvalueClosed(Value v) {
    const bool root = IS_OBJ(v);
    if (root) jaiGCPushRoot(v);

    ObjUpvalue *up = JAI_ALLOCATE_OBJ(ObjUpvalue, OBJ_UPVALUE);

    if (root) jaiGCPopRoot();

    up->closed = v;
    up->location = &up->closed;
    return up;
}

ObjClosure *jaiClosureNew(ObjFunction *fn) {
    const int count = fn == NULL ? 0 : (int)fn->upvalueCount;

    pushObjRoot(fn);
    ObjUpvalue **upvalues = NULL;
    if (count > 0)
        upvalues = JAI_ALLOC_ZEROED(ObjUpvalue *, count);

    ObjClosure *closure = JAI_ALLOCATE_OBJ(ObjClosure, OBJ_CLOSURE);
    jaiGCPopRoot();

    closure->fn = fn;
    closure->upvalues = upvalues;
    closure->upvalueCount = count;
    return closure;
}

ObjNative *jaiNativeNew(JaiNativeFn fn, const char *name,
                        int minArity, int maxArity,
                        const char *const *paramNames) {
    ObjString *interned = jaiStringInternC(name);
    pushObjRoot(interned);
    ObjNative *native = JAI_ALLOCATE_OBJ(ObjNative, OBJ_NATIVE);
    jaiGCPopRoot();

    native->fn = fn;
    native->name = interned;
    native->minArity = (int8_t)(minArity < -128 ? -128 :
                                 (minArity > 127 ? 127 : minArity));
    native->maxArity = (int8_t)(maxArity < -128 ? -128 :
                                 (maxArity > 127 ? 127 : maxArity));
    native->paramNames = paramNames;
    return native;
}

ObjBound *jaiBoundNew(Value receiver, Value method) {
    const bool rootReceiver = IS_OBJ(receiver);
    const bool rootMethod = IS_OBJ(method);
    int roots = 0;

    if (rootReceiver) {
        jaiGCPushRoot(receiver);
        ++roots;
    }
    if (rootMethod) {
        jaiGCPushRoot(method);
        ++roots;
    }

    ObjBound *bound = JAI_ALLOCATE_OBJ(ObjBound, OBJ_BOUND);

    if (roots != 0) jaiGCPopRoots(roots);

    bound->receiver = receiver;
    bound->method = method;
    return bound;
}
