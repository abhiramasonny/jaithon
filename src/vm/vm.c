/* vm.c — the bytecode interpreter (spec/BYTECODE.md).
 *
 * Three invariants hold everywhere below:
 *   1. ip/stackTop/slots/constants/frame are C locals throughout runLoop();
 *      SAVE_STATE()/LOAD_STATE() sync them with the VM around anything that
 *      can allocate, call, or throw -- the GC scans up to vm.stackTop, so a
 *      stale one is a collected live object, not just a slow path.
 *   2. Every re-entrant helper works off vm.stackTop, not the loop's local,
 *      and restores it after (plus whatever it pushed) -- this is what lets
 *      jaiCallValue and friends be called from native code at any depth.
 *   3. No setjmp/longjmp: a raise sets vm.pendingException and jumps to
 *      `vmThrow`, which walks handlers and frames explicitly.
 */
#include <inttypes.h>
#include <math.h>
#include <signal.h>

#include "vm/vm.h"
#include "vm/jit/jit.h"

#include "vm/gc.h"
#include "vm/object/object.h"
#include "vm/table.h"
#include "common/diag.h"
#include "runtime/runtime.h"

VM vm;

/* Nested runLoop() invocations, one per re-entry from native code. Each costs
 * a C stack frame, so it is bounded well below the interpreter frame limit. */
#define JAI_MAX_NESTED_RUN 128
static int sRunDepth;

/* An exception suspended by a `finally` that was reached while unwinding.
 * OP_END_FINALLY resumes the unwind when this is nonzero. */
static int sFinallyPending;

/* Frame index of the default-value thunk currently running, or -1. A thunk
 * shares its function record with the method it belongs to, so OP_RETURN has
 * to know not to apply the initializer's "return self" rule to it. Thunks
 * nest strictly, so one saved index is enough. */
static int sThunkFrame = -1;

/* Handler sentinel for PUSH_FINALLY: matches every exception like a catch-all,
 * but does not consume it. UINT32_MAX is the catch-all constant index from
 * spec §3.8, so the finally marker sits one below it. */
#define JAI_HANDLER_CATCH_ALL UINT32_MAX
#define JAI_HANDLER_FINALLY   (UINT32_MAX - 1u)

/* Set from SIGINT so a runaway program can be stopped at the LOOP safepoint. */
/* 0 = nothing, 1 = Ctrl-C, 2 = a sampling tick from the JIT's timer.
 *
 * One flag with three states rather than two flags, so a tick rides the back
 * edge's existing test instead of adding one. Measured: a new branch in OP_LOOP
 * costs 11% even when it is never taken. */
volatile sig_atomic_t jaiInterrupted;
static bool sSignalInstalled;

/* ------------------------------------------------------------------ */
/* Forward declarations                                                 */
/* ------------------------------------------------------------------ */

typedef enum {
    CALL_ERROR,   /* exception pending */
    CALL_DONE,    /* result already pushed; nothing to interpret */
    CALL_FRAME,   /* a new frame was pushed; resume the loop in it */
} CallOutcome;

static CallOutcome invokeCallable(Value callable, int argc);
static JaiRunResult run(int baseFrameCount);
static bool getPropertyInto(Value receiver, ObjString *name, Value *out,
                            bool raise, InlineCache *ic);
static void closeUpvalues(Value *last);
static bool runFrameDefers(CallFrame *frame);
static bool callDeferred(CallFrame *definer, Value deferred);

/* Almost no frame registers a defer, and asking runFrameDefers to work that
 * out — it is the first thing it does — cost a call on every single return,
 * 4.5% of tests/bench/fib_recursive, which contains no `defer` at all. */
#define FRAME_HAS_DEFERS(f) (vm.defers.count > (f)->deferBase)

/* ------------------------------------------------------------------ */
/* Small utilities                                                      */
/* ------------------------------------------------------------------ */

static CallFrame *topFrame(void) {
    return vm.frameCount > 0 ? &vm.frames[vm.frameCount - 1] : NULL;
}

static Chunk *frameChunk(CallFrame *frame) { return &frame->closure->fn->chunk; }

/* One cache line, or NULL when the operand is out of range (a chunk that came
 * from a corrupt cache file must degrade to the slow path, never crash). */
static InlineCache *cacheAt(Chunk *chunk, uint16_t index) {
    if (chunk->caches == NULL || (int)index >= chunk->cacheCount) return NULL;
    return &chunk->caches[index];
}

/* Inline-cache key for a *builtin* receiver at an INVOKE site.
 *
 * A built-in method is a pure function of (receiver type, name), and the name
 * is baked into the instruction, so the type alone identifies the target. That
 * makes `xs.push(v)` cacheable exactly like an instance method call — which
 * matters because the uncached path allocates a throwaway ObjBound per call.
 *
 * The tag shares the `shapeId` field with ObjClass.shapeId, so it carries the
 * high bit: shape ids are handed out from a counter starting at 1 and could
 * never reach 2^31. A site that sees both instances and lists therefore keeps
 * both kinds of way in one cache without either matching the other's key.
 *
 * Excluded on purpose: modules (members are dynamic and export-checked), files,
 * and everything whose lookup is not a table walk. 0 means "do not cache". */
#define IC_BUILTIN_TAG 0x80000000u

static uint32_t builtinShapeTag(Value v) {
    switch (jaiValueType(v)) {
    case VAL_INT:   return IC_BUILTIN_TAG | 1u;
    case VAL_FLOAT: return IC_BUILTIN_TAG | 2u;
    case VAL_OBJ:   break;
    default:        return 0;
    }
    switch (OBJ_TYPE(v)) {
    case OBJ_STRING: return IC_BUILTIN_TAG | 3u;
    case OBJ_LIST:   return IC_BUILTIN_TAG | 4u;
    case OBJ_DICT:   return IC_BUILTIN_TAG | 5u;
    case OBJ_SET:    return IC_BUILTIN_TAG | 6u;
    case OBJ_TUPLE:  return IC_BUILTIN_TAG | 7u;
    case OBJ_RANGE:  return IC_BUILTIN_TAG | 8u;
    case OBJ_BYTES:  return IC_BUILTIN_TAG | 9u;
    case OBJ_ITER:   return IC_BUILTIN_TAG | 10u;
    default:         return 0;
    }
}

uint8_t jaiInvokeResultFeedback(const Chunk *chunk, uint16_t cacheIdx,
                                Value receiver) {
    uint32_t tag = builtinShapeTag(receiver);
    if (tag == 0 || chunk->caches == NULL ||
        (int)cacheIdx >= chunk->cacheCount) {
        return JAI_FB_NONE;
    }
    const InlineCache *ic = &chunk->caches[cacheIdx];
    for (int w = 0; w < ic->count; w++) {
        if (ic->shapeId[w] == tag) return ic->resultKind[w];
    }
    return JAI_FB_NONE;
}

static bool valueIsCallable(Value v) {
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

/* Short rendering for diagnostics. Deliberately does not call __str__: an
 * error message must never be able to raise a second exception. */
static void describeValue(Value v, char *buf, size_t size) {
    switch (jaiValueType(v)) {
    case VAL_NULL:  snprintf(buf, size, "null"); return;
    case VAL_BOOL:  snprintf(buf, size, "%s", AS_BOOL(v) ? "true" : "false"); return;
    case VAL_INT:   snprintf(buf, size, "%" PRId64, AS_INT(v)); return;
    case VAL_FLOAT: snprintf(buf, size, "%g", AS_FLOAT(v)); return;
    case VAL_OBJ:   break;
    }
    if (IS_STRING(v)) {
        ObjString *s = AS_STRING(v);
        int shown = (int)(s->length > 32 ? 32 : s->length);
        snprintf(buf, size, "'%.*s'%s", shown, s->chars,
                 s->length > 32 ? "..." : "");
        return;
    }
    snprintf(buf, size, "<%s>", jaiTypeNameStatic(v));
}

static bool ensureStack(int extra) {
    if (vm.stack == NULL) return false;
    if (vm.stackTop + extra <= vm.stack + JAI_STACK_MAX) return true;
    return jaiThrow(vm.cRuntimeError, "value stack overflow (%d slots)",
                    JAI_STACK_MAX);
}

/* ------------------------------------------------------------------ */
/* GC roots                                                             */
/* ------------------------------------------------------------------ */

void jaiPushRoot(Value v)   { jaiGCPushRoot(v); }
void jaiPopRoot(void)       { jaiGCPopRoot(); }
void jaiPopRoots(int n)     { jaiGCPopRoots(n); }

/* ------------------------------------------------------------------ */
/* Tracebacks                                                           */
/* ------------------------------------------------------------------ */

/* A traceback captured at the moment an exception escaped, before the frames
 * that produced it were popped. The names are copied into `blob` because the
 * ObjStrings they came from may be collected before the report is printed. */
typedef struct {
    JaiFrameInfo *frames;
    int           count;
    char         *blob;
    size_t        blobSize;
} SavedTraceback;

static SavedTraceback sSavedTb;

static void freeSavedTraceback(void) {
    JAI_FREE_ARRAY(JaiFrameInfo, sSavedTb.frames, sSavedTb.count);
    JAI_FREE_ARRAY(char, sSavedTb.blob, sSavedTb.blobSize);
    sSavedTb.frames = NULL;
    sSavedTb.blob = NULL;
    sSavedTb.count = 0;
    sSavedTb.blobSize = 0;
}

/* Source span of the instruction a frame is currently executing. frame->ip
 * points just past the instruction, so step back one byte before looking the
 * span up. */
static JaiSpan frameSpan(const CallFrame *frame) {
    const Chunk *chunk = &frame->closure->fn->chunk;
    JaiSpan span = JAI_SPAN_NONE;
    if (chunk->code == NULL) return span;

    ptrdiff_t offset = frame->ip - chunk->code;
    if (offset > 0) offset--;
    if (offset < 0) offset = 0;
    if (offset > chunk->count) offset = chunk->count;

    uint32_t start = 0, end = 0;
    jaiChunkSpanAt(chunk, (int)offset, &start, &end);
    span.start = start;
    span.end = end;
    span.file = chunk->sourceFileId;
    return span;
}

static const char *frameFunctionName(const CallFrame *frame) {
    ObjFunction *fn = frame->closure->fn;
    if (fn == NULL) return "<script>";
    if (fn->qualifiedName != NULL) return fn->qualifiedName->chars;
    if (fn->name != NULL) return fn->name->chars;
    return "<script>";
}

static const char *frameModulePath(const CallFrame *frame) {
    if (frame->module != NULL && frame->module->path != NULL) {
        return frame->module->path->chars;
    }
    if (frame->module != NULL && frame->module->name != NULL) {
        return frame->module->name->chars;
    }
    return "<unknown>";
}

JaiFrameInfo *jaiBuildTraceback(int *outCount) {
    int n = vm.frameCount;
    if (outCount != NULL) *outCount = n;
    if (n <= 0) return NULL;

    JaiFrameInfo *frames = JAI_ALLOC(JaiFrameInfo, n);
    for (int i = 0; i < n; i++) {
        const CallFrame *frame = &vm.frames[i];
        frames[i].functionName = frameFunctionName(frame);
        frames[i].modulePath = frameModulePath(frame);
        frames[i].span = frameSpan(frame);
    }
    return frames;
}

/* Snapshot the live frames with their names copied, so the report survives the
 * unwind that is about to discard them. */
static void captureTraceback(void) {
    freeSavedTraceback();
    int n = vm.frameCount;
    if (n <= 0) return;

    size_t bytes = 0;
    for (int i = 0; i < n; i++) {
        bytes += strlen(frameFunctionName(&vm.frames[i])) + 1;
        bytes += strlen(frameModulePath(&vm.frames[i])) + 1;
    }

    char *blob = JAI_ALLOC(char, bytes);
    JaiFrameInfo *frames = JAI_ALLOC(JaiFrameInfo, n);
    size_t at = 0;
    for (int i = 0; i < n; i++) {
        const CallFrame *frame = &vm.frames[i];
        const char *fnName = frameFunctionName(frame);
        const char *modPath = frameModulePath(frame);

        size_t fnLen = strlen(fnName) + 1;
        memcpy(blob + at, fnName, fnLen);
        frames[i].functionName = blob + at;
        at += fnLen;

        size_t modLen = strlen(modPath) + 1;
        memcpy(blob + at, modPath, modLen);
        frames[i].modulePath = blob + at;
        at += modLen;

        frames[i].span = frameSpan(frame);
    }

    sSavedTb.frames = frames;
    sSavedTb.count = n;
    sSavedTb.blob = blob;
    sSavedTb.blobSize = bytes;
}

/* ------------------------------------------------------------------ */
/* Exceptions                                                           */
/* ------------------------------------------------------------------ */

bool jaiThrowValue(Value exception) {
    vm.pendingException = exception;
    vm.hasException = true;
    return false;
}

bool jaiThrow(ObjClass *klass, const char *fmt, ...) {
    char message[512];
    va_list ap;
    va_start(ap, fmt);
    int written = vsnprintf(message, sizeof message, fmt, ap);
    va_end(ap);
    if (written < 0) message[0] = '\0';

    /* An exception raised while one is already unwinding replaces it: the new
     * failure is the one closest to the fault. */
    ObjClass *target = klass;
    if (target == NULL) target = vm.cRuntimeError;
    if (target == NULL) target = vm.cError;

    if (target == NULL) {
        /* Before jaiRegisterErrorClasses has run there is nothing to
         * instantiate; carry the text itself so nothing is lost. */
        ObjString *text = jaiStringNew(message, strlen(message));
        return jaiThrowValue(text != NULL ? OBJ_VAL(text) : NULL_VAL);
    }
    return jaiThrowValue(jaiMakeException(target, message));
}

void jaiClearException(void) {
    vm.hasException = false;
    vm.pendingException = NULL_VAL;
    sFinallyPending = 0;
    freeSavedTraceback();
}

/* The `message` field of an exception instance, read without dispatching to
 * user code. */
static const char *exceptionMessage(Value exception) {
    if (!IS_INSTANCE(exception)) return NULL;
    ObjInstance *inst = AS_INSTANCE(exception);
    if (inst->klass == NULL || vm.strMessage == NULL) return NULL;
    int slot = jaiClassFieldSlot(inst->klass, vm.strMessage);
    if (slot < 0 || slot >= (int)inst->fieldCount) return NULL;
    Value v = inst->fields[slot];
    return IS_STRING(v) ? AS_CSTRING(v) : NULL;
}

static const char *exceptionTypeName(Value exception) {
    if (IS_INSTANCE(exception)) {
        ObjClass *k = AS_INSTANCE(exception)->klass;
        if (k != NULL && k->name != NULL) return k->name->chars;
        return "Error";
    }
    if (IS_CLASS(exception) && AS_CLASS(exception)->name != NULL) {
        return AS_CLASS(exception)->name->chars;
    }
    return jaiTypeNameStatic(exception);
}

void jaiReportUncaught(Value exception) {
    char fallback[256];
    const char *message = exceptionMessage(exception);
    if (message == NULL) {
        describeValue(exception, fallback, sizeof fallback);
        message = fallback;
    }

    bool ownFrames = false;
    JaiFrameInfo *frames = sSavedTb.frames;
    int count = sSavedTb.count;
    if (frames == NULL) {
        frames = jaiBuildTraceback(&count);
        ownFrames = true;
    }

    jaiPrintTraceback(stderr, frames, count, exceptionTypeName(exception),
                      message, gDiags.colorOutput);

    if (ownFrames) JAI_FREE_ARRAY(JaiFrameInfo, frames, count);
    freeSavedTraceback();
}

/* ------------------------------------------------------------------ */
/* Upvalues                                                             */
/* ------------------------------------------------------------------ */

static ObjUpvalue *captureUpvalue(Value *local) {
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

static void closeUpvalues(Value *last) {
    while (vm.openUpvalues != NULL && vm.openUpvalues->location >= last) {
        ObjUpvalue *upvalue = vm.openUpvalues;
        upvalue->closed = *upvalue->location;
        upvalue->location = &upvalue->closed;
        vm.openUpvalues = upvalue->next;
    }
}

/* ------------------------------------------------------------------ */
/* Visibility                                                           */
/*                                                                      */
/* The checker rejects what it can prove (E0701); this is the dynamic    */
/* half, for `any`-typed receivers.                                      */
/* ------------------------------------------------------------------ */

static bool accessPermitted(const ObjClass *owner, Visibility vis) {
    if (vis == VIS_PUBLIC) return true;
    CallFrame *frame = topFrame();
    if (frame == NULL || owner == NULL) return false;

    /* fn->owner (the declaring class, spec §7.1) is checked first and is
     * exact; slot 0 -- the instance, or for a `static fn` the class itself --
     * is only a fallback for a lambda with no owner. A tail-called static
     * factory compiles to GET_FIELD+TAIL_CALL, leaving the function itself in
     * slot 0, so relying on slot 0 alone locked every static factory out of
     * its own fields. */
    ObjClass *selfClass = NULL;
    if (frame->closure != NULL && frame->closure->fn != NULL)
        selfClass = frame->closure->fn->owner;
    if (selfClass == NULL) {
        Value self = frame->slots[0];
        if (IS_INSTANCE(self))   selfClass = AS_INSTANCE(self)->klass;
        else if (IS_CLASS(self)) selfClass = AS_CLASS(self);
    }
    if (selfClass == NULL) return false;

    if (vis == VIS_PROTECTED) {
        return jaiClassIsSubclassOf(selfClass, owner) ||
               jaiClassIsSubclassOf(owner, selfClass);
    }
    /* Private: the executing method must belong to the declaring class or to
     * a subclass that inherited the field slot. */
    return jaiClassIsSubclassOf(selfClass, owner);
}

/* The class a field was declared on, for the visibility test above. Walks up
 * from `klass` to the topmost class that still declares the slot. */
/* Callers must gate this on the field being non-public. It walks the whole
 * superclass chain running a linear field scan at every level, and
 * accessPermitted discards it for VIS_PUBLIC -- but C evaluates arguments
 * before the call, so writing it as an argument paid for it on every public
 * field read. That was ~11% of a compile in getPropertyInto. */
static const ObjClass *fieldOwner(ObjClass *klass, ObjString *name) {
    const ObjClass *owner = klass;
    for (ObjClass *k = klass; k != NULL; k = k->superclass) {
        if (jaiClassFieldInfo(k, name) != NULL) owner = k;
    }
    return owner;
}

/* The dynamic half of spec §7.1 for methods. Returns true when `name` may be
 * reached from the running frame; when it may not, raises AttributeError (or
 * stays quiet if `raise` is false) and returns false. Public and unknown names
 * — everything, in a class that declares no non-public method — cost one load
 * inside jaiClassRestrictedMethod. */
static bool methodPermitted(ObjClass *klass, ObjString *name, bool raise) {
    MethodInfo mi;
    if (!jaiClassRestrictedMethod(klass, name, &mi)) return true;
    if (accessPermitted(mi.owner, mi.visibility)) return true;
    if (!raise) return false;
    return jaiThrow(vm.cAttributeError, "method '%s' of class '%s' is %s",
                    name->chars,
                    mi.owner != NULL && mi.owner->name != NULL
                        ? mi.owner->name->chars : "?",
                    mi.visibility == VIS_PROTECTED ? "protected" : "private");
}

/* ------------------------------------------------------------------ */
/* Method and property lookup                                           */
/* ------------------------------------------------------------------ */

/* Trait default implementations are copied into the class at OP_IMPL_TRAIT;
 * this walk is the safety net for classes built by other means. */
static bool findTraitDefault(const ObjClass *klass, ObjString *name, Value *out) {
    for (const ObjClass *k = klass; k != NULL; k = k->superclass) {
        for (uint16_t i = 0; i < k->traitCount; i++) {
            ObjTrait *t = k->traits[i];
            if (t == NULL) continue;
            if (jaiTableGetInterned(&t->defaults, name, out)) return true;
            for (uint16_t j = 0; j < t->superCount; j++) {
                if (t->supers[j] != NULL &&
                    jaiTableGetInterned(&t->supers[j]->defaults, name, out)) {
                    return true;
                }
            }
        }
    }
    return false;
}

/* The raw (unbound) method for `name` on `klass`, including inherited and
 * trait-default methods. */
static bool findMethod(ObjClass *klass, ObjString *name, Value *out) {
    if (klass == NULL || name == NULL) return false;
    if (jaiTableGetInterned(&klass->methods, name, out)) return true;
    if (jaiTableGetInterned(&klass->statics, name, out)) return true;
    return findTraitDefault(klass, name, out);
}

/* Index of the variant named `name`, or -1. */
static int jaiEnumVariantIndex(const ObjEnum *e, const ObjString *name) {
    if (e == NULL || name == NULL) return -1;
    for (uint16_t i = 0; i < e->variantCount; i++) {
        ObjString *have = e->variants[i].name;
        if (have == name || jaiStringEquals(have, name)) return (int)i;
    }
    return -1;
}

/* Zero-arity enum variant, or the enum's own method, for `Color.Red`.
 *
 * The variant's one value is cached on the variant. Building a fresh one per
 * mention made `x is Token.LParen` false for an `x` that *was* `Token.LParen`
 * — `is` is identity (spec §4.2) — and allocated on a path as hot as a parser's
 * token test. A payload-less variant has no state to distinguish instances by,
 * so one value is all there can be. */
static bool enumMember(ObjEnum *e, ObjString *name, Value *out) {
    if (jaiTableGetInterned(&e->methods, name, out)) return true;
    for (uint16_t i = 0; i < e->variantCount; i++) {
        EnumVariant *v = &e->variants[i];
        if (v->name != name && !jaiStringEquals(v->name, name)) continue;
        if (v->arity != 0) {
            /* Not a value yet, but a function of its payload — and the checker
             * types it as exactly that, so `let make = Shape.Circle` has to
             * work rather than report a variant that plainly exists as
             * missing. The direct-call sites could special-case this, and one
             * of them did; a tail call could not, because the callee is a
             * value by then. */
            if (v->ctor == NULL) v->ctor = jaiEnumCtorNew(e, i);
            *out = OBJ_VAL(v->ctor);
            return true;
        }
        if (v->unit == NULL) v->unit = jaiEnumValNew(e, i, NULL, 0);
        *out = OBJ_VAL(v->unit);
        return true;
    }
    return false;
}

/* A module member, honouring the export list. A module with no explicit
 * exports (no `export` block and no `pub`) exposes everything: restricting it
 * would make an un-annotated module unusable rather than encapsulated. */
static bool moduleMember(ObjModule *m, ObjString *name, Value *out,
                         bool *outHidden) {
    *outHidden = false;
    if (!jaiModuleGet(m, name, out)) return false;
    if (m->exports.count > 0 && !jaiModuleIsExported(m, name)) {
        *outHidden = true;
        return false;
    }
    return true;
}

/* Shared implementation of `receiver.name` for reads. `ic`, when non-NULL, is
 * filled on a plain instance-field hit so the next execution skips all of
 * this. `raise` selects between "AttributeError" and "quietly false". */
static bool getPropertyInto(Value receiver, ObjString *name, Value *out,
                            bool raise, InlineCache *ic) {
    if (name == NULL) return false;
#ifdef JAI_PROP_STATS
    { extern uint64_t jaiPropRecv[]; jaiPropRecv[IS_OBJ(receiver) ? 8 + (int)AS_OBJ(receiver)->type : (int)receiver.type]++; }
#endif

    if (IS_INSTANCE(receiver)) {
        ObjInstance *inst = AS_INSTANCE(receiver);
        ObjClass *klass = inst->klass;

        /* Getters win over the raw field: a property exists precisely to
         * intercept the access (spec §7.1). */
        Value getter;
        if (klass != NULL && jaiTableGetInterned(&klass->getters, name, &getter)) {
            if (!methodPermitted(klass, name, raise)) return false;
            return jaiCallValue(OBJ_VAL(jaiBoundNew(receiver, getter)), 0, NULL,
                                out);
        }

        const FieldInfo *info = jaiClassFieldInfo(klass, name);
        if (info != NULL) {
            if (info->visibility != VIS_PUBLIC &&
                !accessPermitted(fieldOwner(klass, name), info->visibility)) {
                if (!raise) return false;
                return jaiThrow(vm.cAttributeError,
                                "field '%s' of class '%s' is private",
                                name->chars,
                                klass->name != NULL ? klass->name->chars : "?");
            }
            if (info->slot >= inst->fieldCount) {
                if (!raise) return false;
                return jaiThrow(vm.cAttributeError,
                                "field '%s' is not present on this instance",
                                name->chars);
            }
            if (ic != NULL && klass != NULL) {
                /* Only monomorphic-to-polymorphic growth; a megamorphic site
                 * stops caching rather than thrashing four ways forever. */
                if (ic->state == IC_MEGA) {
                    /* leave it alone */
                } else if (ic->count < JAI_IC_WAYS) {
                    ic->shapeId[ic->count] = klass->shapeId;
                    ic->payload[ic->count] = info->slot;
                    ic->cached[ic->count] = NULL_VAL;
                    ic->count++;
                    ic->state = (ic->count == 1) ? IC_MONO : IC_POLY;
                } else {
                    ic->state = IC_MEGA;
                }
            }
            *out = inst->fields[info->slot];
            return true;
        }

        Value method;
        if (findMethod(klass, name, &method)) {
            if (!methodPermitted(klass, name, raise)) return false;
            *out = OBJ_VAL(jaiBoundNew(receiver, method));
            return true;
        }
        if (!raise) return false;
        return jaiThrow(vm.cAttributeError, "'%s' object has no attribute '%s'",
                        klass != NULL && klass->name != NULL ? klass->name->chars
                                                             : "instance",
                        name->chars);
    }

    if (IS_CLASS(receiver)) {
        ObjClass *klass = AS_CLASS(receiver);
        if (jaiTableGetInterned(&klass->statics, name, out) ||
            jaiTableGetInterned(&klass->methods, name, out)) {
            return methodPermitted(klass, name, raise);
        }
        if (!raise) return false;
        return jaiThrow(vm.cAttributeError, "class '%s' has no member '%s'",
                        klass->name != NULL ? klass->name->chars : "?",
                        name->chars);
    }

    if (IS_MODULE(receiver)) {
        ObjModule *m = AS_MODULE(receiver);
        bool hidden = false;
        if (moduleMember(m, name, out, &hidden)) return true;
        if (!raise) return false;
        if (hidden) {
            return jaiThrow(vm.cImportError,
                            "'%s' is not exported by module '%s'", name->chars,
                            m->name != NULL ? m->name->chars : "?");
        }
        /* Reading `mod.members` without calling it has to find the same
         * introspection helpers `mod.members()` does. */
        if (jaiBuiltinMethod(receiver, name, out)) return true;
        return jaiThrow(vm.cAttributeError, "module '%s' has no member '%s'",
                        m->name != NULL ? m->name->chars : "?", name->chars);
    }

    if (IS_ENUM(receiver)) {
        if (enumMember(AS_ENUM(receiver), name, out)) {
            /* Safe to memoise: a variant's unit value and ctor are created
             * once and kept on the variant, the methods table is filled only
             * while the enum is being defined, and any addition bumps shapeId.
             * The GC marks ic->cached (gc.c:201), so this is a strong
             * reference and cannot dangle. */
            if (ic != NULL && ic->state != IC_MEGA) {
                if (ic->count < JAI_IC_WAYS) {
                    ic->shapeId[ic->count] = AS_ENUM(receiver)->shapeId;
                    ic->payload[ic->count] = 0;
                    ic->cached[ic->count] = *out;
                    ic->count++;
                    ic->state = (ic->count == 1) ? IC_MONO : IC_POLY;
                } else {
                    ic->state = IC_MEGA;
                }
            }
            return true;
        }
        if (!raise) return false;
        return jaiThrow(vm.cAttributeError, "enum '%s' has no variant '%s'",
                        AS_ENUM(receiver)->name != NULL
                            ? AS_ENUM(receiver)->name->chars : "?",
                        name->chars);
    }

    if (IS_ENUM_VAL(receiver)) {
        ObjEnumVal *ev = AS_ENUM_VAL(receiver);
        if (ev->type != NULL && ev->tag < ev->type->variantCount) {
            EnumVariant *variant = &ev->type->variants[ev->tag];
            for (uint8_t i = 0; i < variant->arity && i < ev->count; i++) {
                if (variant->fieldNames == NULL) break;
                if (variant->fieldNames[i] == name ||
                    jaiStringEquals(variant->fieldNames[i], name)) {
                    *out = ev->payload[i];
                    return true;
                }
            }
        }
        if (ev->type != NULL && jaiTableGetInterned(&ev->type->methods, name, out)) {
            *out = OBJ_VAL(jaiBoundNew(receiver, *out));
            return true;
        }
        if (!raise) return false;
        return jaiThrow(vm.cAttributeError, "'%s' has no member '%s'",
                        jaiTypeNameStatic(receiver), name->chars);
    }

    /* str, list, dict, ... : the built-in method tables hand back a bound
     * native, so the call path downstream is identical to a user method. */
    if (jaiBuiltinMethod(receiver, name, out)) return true;
    if (!raise) return false;
    return jaiThrow(vm.cAttributeError, "'%s' object has no attribute '%s'",
                    jaiTypeNameStatic(receiver), name->chars);
}

bool jaiGetProperty(Value receiver, ObjString *name, Value *out) {
    return getPropertyInto(receiver, name, out, true, NULL);
}

static bool throwFieldKind(const FieldInfo *info, Value v) {
    return jaiThrow(vm.cTypeError,
                    "cannot assign %s to field '%s' declared %s",
                    jaiTypeNameStatic(v), info->name->chars,
                    jaiFieldKindName(info->typeId));
}

bool jaiSetProperty(Value receiver, ObjString *name, Value value) {
    if (name == NULL) return false;

    if (IS_INSTANCE(receiver)) {
        ObjInstance *inst = AS_INSTANCE(receiver);
        ObjClass *klass = inst->klass;

        Value setter;
        if (klass != NULL && jaiTableGetInterned(&klass->setters, name, &setter)) {
            /* A property is a method; a private one is no more assignable from
             * outside than a private field is. */
            if (!methodPermitted(klass, name, true)) return false;
            Value ignored;
            Value arg = value;
            return jaiCallValue(OBJ_VAL(jaiBoundNew(receiver, setter)), 1, &arg,
                                &ignored);
        }

        const FieldInfo *info = jaiClassFieldInfo(klass, name);
        if (info == NULL) {
            /* Spec §7.1: fields are declared, never conjured by assignment. */
            return jaiThrow(vm.cAttributeError,
                            "'%s' object has no field '%s'; fields must be "
                            "declared in the class body",
                            klass != NULL && klass->name != NULL
                                ? klass->name->chars : "instance",
                            name->chars);
        }
        if (info->visibility != VIS_PUBLIC &&
                !accessPermitted(fieldOwner(klass, name), info->visibility)) {
            return jaiThrow(vm.cAttributeError,
                            "field '%s' of class '%s' is private", name->chars,
                            klass->name != NULL ? klass->name->chars : "?");
        }
        if (info->slot >= inst->fieldCount) {
            return jaiThrow(vm.cAttributeError,
                            "field '%s' is not present on this instance",
                            name->chars);
        }
        /* The checker could not have caught this: reaching a field through an
         * `any` receiver means it did not know the class, so it did not know
         * what the field was declared as. FieldInfo is the only place that
         * knows, so the runtime is the only place this can be rejected. */
        if (!jaiKindAccepts(info->typeId, value)) {
            return throwFieldKind(info, value);
        }
        inst->fields[info->slot] = value;
        return true;
    }

    if (IS_CLASS(receiver)) {
        ObjClass *klass = AS_CLASS(receiver);
        Value existing;
        if (!jaiTableGetInterned(&klass->statics, name, &existing)) {
            return jaiThrow(vm.cAttributeError,
                            "class '%s' has no static member '%s'",
                            klass->name != NULL ? klass->name->chars : "?",
                            name->chars);
        }
        jaiGCPushRoot(receiver);
        jaiGCPushRoot(value);
        (void)jaiTableSetInterned(&klass->statics, name, value);
        jaiGCPopRoots(2);
        return true;
    }

    if (IS_MODULE(receiver)) {
        ObjModule *m = AS_MODULE(receiver);
        Value existing;
        if (!jaiModuleGet(m, name, &existing)) {
            return jaiThrow(vm.cAttributeError, "module '%s' has no member '%s'",
                            m->name != NULL ? m->name->chars : "?", name->chars);
        }
        jaiModuleSet(m, name, value);
        return true;
    }

    return jaiThrow(vm.cAttributeError,
                    "cannot set attribute '%s' on a '%s' value", name->chars,
                    jaiTypeNameStatic(receiver));
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

static int frameWindowSize(const ObjFunction *fn) {
    int declared = 1 + (int)fn->arity;
    if (fn->flags & FN_VARIADIC) declared++;
    if (fn->flags & FN_KWREST) declared++;
    return (int)fn->maxSlots > declared ? (int)fn->maxSlots : declared;
}

/* Where a callee's keyword-rest dict lives in its frame window. */
static int kwRestSlotOf(const ObjFunction *fn) {
    return 1 + (int)fn->arity + ((fn->flags & FN_VARIADIC) ? 1 : 0);
}

/* Headroom above a frame's window for expression temporaries. maxSlots counts
 * the register window; nested expressions push above it. */
#define JAI_FRAME_SLACK 256

static bool ensureRoom(const Value *from, int slots) {
    if (vm.stack != NULL && from + slots <= vm.stack + JAI_STACK_MAX) return true;
    return jaiThrow(vm.cRuntimeError, "value stack overflow (%d slots)",
                    JAI_STACK_MAX);
}

static bool pushFrame(ObjClosure *closure, Value *slotBase) {
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
static bool bindCallArgsSlow(ObjClosure *closure, int argc, Value *slotBase) {
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
        for (int i = 0; i < extra; i++) rest->items[i] = slotBase[1 + arity + i];
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

/* Fixed arity, fully applied — every call in a hot loop. Nothing the slow path
 * does applies: the arity checks cannot fire, there is no variadic tail to
 * pack, no default to evaluate and no keyword-rest dict to make, so all that is
 * left is clearing the frame's window.
 *
 * Inline because this runs 49.8M times in one `check lib/std` and the work it
 * does is a branch and ~3.6 stores. As one function with the slow path it was
 * too big for clang to inline and showed up in the profile as its own symbol,
 * paying call overhead per call to do almost nothing. */
static inline bool bindCallArgs(ObjClosure *closure, int argc, Value *slotBase) {
    ObjFunction *fn = closure->fn;
    int arity = (int)fn->arity;

    if (JAI_LIKELY(argc == arity && fn->defaultCount == 0 &&
                   (fn->flags & (FN_VARIADIC | FN_KWREST)) == 0)) {
        int window = (int)fn->maxSlots > 1 + arity ? (int)fn->maxSlots : 1 + arity;
        if (!ensureRoom(slotBase, window + JAI_FRAME_SLACK)) return false;
        /* The collector scans the whole window as soon as stackTop is above
         * it, so no slot may be left holding whatever the last frame did. */
        for (int i = 1 + argc; i < window; i++) slotBase[i] = NULL_VAL;
        vm.stackTop = slotBase + window;
        return true;
    }
    return bindCallArgsSlow(closure, argc, slotBase);
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
static CallOutcome callClosure(ObjClosure *closure, int argc) {
    Value *slotBase = vm.stackTop - argc - 1;
    ObjFunction *fn = closure->fn;

    /* Ahead of bindCallArgs on purpose. Compiled code never reads the VM
     * stack, so clearing a frame window it will not look at, and growing the
     * stack for slots it will not use, is pure cost -- and on a small hot
     * function called from an interpreted loop that entry cost was most of
     * what the tier had to give. */
    if (fn->jitFunc != NULL && argc == (int)fn->arity) {
        JaiJitOutcome outcome = jaiJitEnterFunc(closure, slotBase);
        if (outcome == JAI_JIT_DONE) return CALL_DONE;
        /* An exception from a call the compiled body made: the effects up to
         * it already happened, so this is not a decline. */
        if (outcome == JAI_JIT_ERROR) return CALL_ERROR;
        if (outcome == JAI_JIT_DEOPT) {
            /* A guard failed part-way in. The interpreter takes over from that
             * exact instruction, holding what the compiled body held, so the
             * work already done is neither lost nor repeated. */
            if (!bindCallArgs(closure, argc, slotBase)) return CALL_ERROR;
            if (!pushFrame(closure, slotBase)) return CALL_ERROR;
            if (!jaiJitApplyDeopt(closure, slotBase)) return CALL_ERROR;
            return CALL_FRAME;
        }
    }

    if (!bindCallArgs(closure, argc, slotBase)) return CALL_ERROR;
    if (fn->entryCount < JAI_JIT_THRESHOLD) {
        fn->entryCount++;
    } else if (!fn->jitRefused && jaiJitEnabled() &&
               jaiJitEnter(closure, slotBase)) {
        return CALL_DONE;
    }

    if (!pushFrame(closure, slotBase)) return CALL_ERROR;
    return CALL_FRAME;
}

/* `args` and `count` differ from the raw stack window for bound natives, whose
 * receiver occupies the callee slot and is passed as args[0]. */
static bool callNativeAt(ObjNative *native, Value *args, int count,
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
static CallOutcome invokeCallable(Value callable, int argc) {
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
static CallOutcome invokeMethodOnStack(Value callable, int argc) {
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
static CallOutcome callValueOnStack(int argc) {
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
/* The class carrying `shape`, found by walking the classes the tier has
 * already seen. Only used at compile time. */
static ObjClass *sShapeCache[64];
void jaiClassRememberShape(ObjClass *c) {
    if (c == NULL) return;
    sShapeCache[c->shapeId & 63u] = c;
}
bool jaiClassForShape(uint32_t shape, ObjClass **out) {
    ObjClass *c = sShapeCache[shape & 63u];
    if (c == NULL || c->shapeId != shape) return false;
    *out = c;
    return true;
}

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
static bool resolveInvokeTarget(Value receiver, ObjString *name, Value *method,
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

/* ------------------------------------------------------------------ */
/* Operators                                                            */
/* ------------------------------------------------------------------ */

static const char *opSymbol(OpCode op) {
    switch (op) {
    case OP_ADD: case OP_ADD_WRAP:  return op == OP_ADD ? "+" : "+%";
    case OP_SUB: case OP_SUB_WRAP:  return op == OP_SUB ? "-" : "-%";
    case OP_MUL: case OP_MUL_WRAP:  return op == OP_MUL ? "*" : "*%";
    case OP_DIV:                    return "/";
    case OP_FLOORDIV:               return "//";
    case OP_MOD:                    return "%";
    case OP_POW:                    return "**";
    case OP_BAND:                   return "&";
    case OP_BOR:                    return "|";
    case OP_BXOR:                   return "^";
    case OP_SHL:                    return "<<";
    case OP_SHR:                    return ">>";
    case OP_LT:                     return "<";
    case OP_LE:                     return "<=";
    case OP_GT:                     return ">";
    case OP_GE:                     return ">=";
    case OP_CONCAT:                 return "+";
    default:                        return "?";
    }
}

static bool badOperands(OpCode op, Value a, Value b) {
    return jaiThrow(vm.cTypeError,
                    "unsupported operand types for '%s': '%s' and '%s'",
                    opSymbol(op), jaiTypeNameStatic(a), jaiTypeNameStatic(b));
}

static bool intOverflow(OpCode op) {
    /* Only `+ - *` have a wrapping spelling (spec §2.5). Suggesting `**%` or
     * `//%` sends the reader looking for an operator the grammar does not have. */
    switch (op) {
    case OP_ADD: case OP_SUB: case OP_MUL:
        return jaiThrow(vm.cOverflowError,
                        "integer overflow in '%s'; use '%s%%' to wrap",
                        opSymbol(op), opSymbol(op));
    default:
        return jaiThrow(vm.cOverflowError, "integer overflow in '%s'", opSymbol(op));
    }
}

/* Python's floor semantics: the result takes the sign of the divisor, so
 * -7 // 2 is -4 and -7 % 3 is 2. C truncates toward zero, which is why both
 * results are corrected here rather than used directly. */
static bool intFloorDivMod(int64_t a, int64_t b, int64_t *outDiv,
                           int64_t *outMod) {
    if (b == 0) {
        return jaiThrow(vm.cDivisionByZeroError, "integer division by zero");
    }
    if (a == INT64_MIN && b == -1) {
        *outDiv = INT64_MIN;   /* only reached by MOD, which wants 0 */
        *outMod = 0;
        return true;
    }
    int64_t q = a / b;
    int64_t r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) {
        q -= 1;
        r += b;
    }
    *outDiv = q;
    *outMod = r;
    return true;
}

static bool floatFloorDivMod(double a, double b, double *outDiv, double *outMod) {
    if (b == 0.0) {
        return jaiThrow(vm.cDivisionByZeroError, "float division by zero");
    }
    double r = fmod(a, b);
    if (r != 0.0 && ((r < 0.0) != (b < 0.0))) r += b;
    *outDiv = floor(a / b);
    *outMod = r;
    return true;
}

static bool intPow(int64_t base, int64_t exp, int64_t *out) {
    if (exp < 0) {
        return jaiThrow(vm.cValueError,
                        "negative exponent on int '**'; convert to float first");
    }
    int64_t result = 1;
    int64_t acc = base;
    while (exp > 0) {
        if (exp & 1) {
            if (__builtin_mul_overflow(result, acc, &result)) {
                return intOverflow(OP_POW);
            }
        }
        exp >>= 1;
        if (exp == 0) break;
        if (__builtin_mul_overflow(acc, acc, &acc)) return intOverflow(OP_POW);
    }
    *out = result;
    return true;
}

/* Repeat a sequence: "ab" * 3 and [0] * 4. */
static bool repeatString(ObjString *s, int64_t times, Value *out) {
    if (times <= 0) {
        ObjString *empty = jaiStringIntern("", 0);
        if (empty == NULL) return false;
        *out = OBJ_VAL(empty);
        return true;
    }
    if ((uint64_t)times * (uint64_t)s->length > (uint64_t)UINT32_MAX) {
        return jaiThrow(vm.cOverflowError,
                        "repeated string exceeds the maximum length");
    }
    size_t total = (size_t)times * (size_t)s->length;
    char *buf = JAI_ALLOC(char, total + 1);
    for (int64_t i = 0; i < times; i++) {
        memcpy(buf + (size_t)i * s->length, s->chars, s->length);
    }
    buf[total] = '\0';
    ObjString *result = jaiStringTake(buf, total);
    if (result == NULL) return false;
    *out = OBJ_VAL(result);
    return true;
}

static bool repeatList(ObjList *list, int64_t times, Value *out) {
    if (times < 0) times = 0;
    /* Divide rather than multiply: the product itself can overflow. */
    if (list->count > 0 && times > (int64_t)INT32_MAX / list->count) {
        return jaiThrow(vm.cRuntimeError, "list cannot grow beyond %d items",
                        INT32_MAX);
    }
    jaiGCPushRoot(OBJ_VAL(list));
    ObjList *result = jaiListNew((int)(times * list->count));
    for (int64_t i = 0; i < times; i++) {
        for (int j = 0; j < list->count; j++) {
            result->items[result->count++] = list->items[j];
        }
    }
    jaiGCPopRoot();
    *out = OBJ_VAL(result);
    return true;
}

/* The dunder method for a binary operator on an instance receiver, or
 * NULL_VAL when the class does not overload it. */
static Value operatorDunder(ObjClass *klass, OpCode op) {
    if (klass == NULL) return NULL_VAL;
    switch (op) {
    case OP_ADD: case OP_ADD_WRAP: case OP_CONCAT: return klass->dunderAdd;
    case OP_SUB: case OP_SUB_WRAP:                 return klass->dunderSub;
    case OP_MUL: case OP_MUL_WRAP:                 return klass->dunderMul;
    case OP_DIV:                                   return klass->dunderDiv;
    case OP_MOD:                                   return klass->dunderMod;
    case OP_POW:                                   return klass->dunderPow;
    default:                                       return NULL_VAL;
    }
}

/* Instance fallback. Only `__add__`-style forward dispatch exists — the spec's
 * dunder list has no reflected forms — so when only the right operand is an
 * instance its own dunder is tried with the operands as written. */
static bool binaryDunder(OpCode op, Value a, Value b, Value *out, bool *handled) {
    *handled = false;
    Value method = NULL_VAL;
    Value receiver = a, arg = b;

    if (IS_INSTANCE(a)) {
        method = operatorDunder(AS_INSTANCE(a)->klass, op);
    }
    if (IS_NULL(method) && IS_INSTANCE(b)) {
        method = operatorDunder(AS_INSTANCE(b)->klass, op);
        receiver = b;
        arg = a;
    }
    if (IS_NULL(method)) return true;

    *handled = true;
    Value bound = OBJ_VAL(jaiBoundNew(receiver, method));
    return jaiCallValue(bound, 1, &arg, out);
}

/* The float half of `arithmetic`, split out because two operand shapes reach
 * it: two floats, and an int widened against a float. `a` and `b` are carried
 * only so that an operator with no float form names its real operands. */
static bool floatArithmetic(OpCode op, double x, double y, Value a, Value b,
                            Value *out) {
    switch (op) {
    case OP_ADD: *out = FLOAT_VAL(x + y); return true;
    case OP_SUB: *out = FLOAT_VAL(x - y); return true;
    case OP_MUL: *out = FLOAT_VAL(x * y); return true;
    case OP_DIV:
        if (y == 0.0) {
            return jaiThrow(vm.cDivisionByZeroError, "float division by zero");
        }
        *out = FLOAT_VAL(x / y);
        return true;
    case OP_FLOORDIV: {
        double q, m;
        if (!floatFloorDivMod(x, y, &q, &m)) return false;
        *out = FLOAT_VAL(q);
        return true;
    }
    case OP_MOD: {
        double q, m;
        if (!floatFloorDivMod(x, y, &q, &m)) return false;
        *out = FLOAT_VAL(m);
        return true;
    }
    case OP_POW: *out = FLOAT_VAL(pow(x, y)); return true;
    default:     return badOperands(op, a, b);
    }
}

static bool arithmetic(OpCode op, Value a, Value b, Value *out) {
    if (IS_INT(a) && IS_INT(b)) {
        int64_t x = AS_INT(a), y = AS_INT(b), r = 0;
        switch (op) {
        case OP_ADD:
            if (__builtin_add_overflow(x, y, &r)) return intOverflow(op);
            break;
        case OP_SUB:
            if (__builtin_sub_overflow(x, y, &r)) return intOverflow(op);
            break;
        case OP_MUL:
            if (__builtin_mul_overflow(x, y, &r)) return intOverflow(op);
            break;
        case OP_ADD_WRAP: r = (int64_t)((uint64_t)x + (uint64_t)y); break;
        case OP_SUB_WRAP: r = (int64_t)((uint64_t)x - (uint64_t)y); break;
        case OP_MUL_WRAP: r = (int64_t)((uint64_t)x * (uint64_t)y); break;
        case OP_DIV:
            /* int / int is float (spec §3.3); use // for an integer result. */
            if (y == 0) {
                return jaiThrow(vm.cDivisionByZeroError, "division by zero");
            }
            *out = FLOAT_VAL((double)x / (double)y);
            return true;
        case OP_FLOORDIV: {
            int64_t q, m;
            if (!intFloorDivMod(x, y, &q, &m)) return false;
            if (x == INT64_MIN && y == -1) return intOverflow(op);
            r = q;
            break;
        }
        case OP_MOD: {
            int64_t q, m;
            if (!intFloorDivMod(x, y, &q, &m)) return false;
            r = m;
            break;
        }
        case OP_POW:
            if (!intPow(x, y, &r)) return false;
            break;
        default:
            return badOperands(op, a, b);
        }
        *out = INT_VAL(r);
        return true;
    }

    if (IS_FLOAT(a) && IS_FLOAT(b)) {
        return floatArithmetic(op, AS_FLOAT(a), AS_FLOAT(b), a, b, out);
    }

    /* One int and one float: the int widens and the result is a float (spec
     * §2.5). Where both types are known the checker has already inserted the
     * conversion, so what reaches here came through `any`.
     *
     * The wrapping forms are the exception. They are 64-bit integer
     * operations, `float` has nothing to wrap, and quietly widening `+%` into
     * ordinary float addition would answer a question about overflow with a
     * number that cannot overflow. */
    if (IS_NUMBER(a) && IS_NUMBER(b)) {
        if (op == OP_ADD_WRAP || op == OP_SUB_WRAP || op == OP_MUL_WRAP) {
            return jaiThrow(vm.cTypeError,
                            "'%s' wraps a 64-bit integer and does not apply to "
                            "'float'", opSymbol(op));
        }
        double x = IS_INT(a) ? (double)AS_INT(a) : AS_FLOAT(a);
        double y = IS_INT(b) ? (double)AS_INT(b) : AS_FLOAT(b);
        return floatArithmetic(op, x, y, a, b, out);
    }

    if (op == OP_ADD || op == OP_CONCAT) {
        if (IS_STRING(a) && IS_STRING(b)) {
            ObjString *s = jaiStringConcat(AS_STRING(a), AS_STRING(b));
            if (s == NULL) return false;
            *out = OBJ_VAL(s);
            return true;
        }
        if (IS_LIST(a) && IS_LIST(b)) {
            ObjList *l = jaiListConcat(AS_LIST(a), AS_LIST(b));
            if (l == NULL) return false;
            *out = OBJ_VAL(l);
            return true;
        }
        if (IS_TUPLE(a) && IS_TUPLE(b)) {
            ObjTuple *ta = AS_TUPLE(a), *tb = AS_TUPLE(b);
            jaiGCPushRoot(a);
            jaiGCPushRoot(b);
            ObjTuple *result = jaiTupleNew(NULL, (int)(ta->count + tb->count));
            for (uint32_t i = 0; i < ta->count; i++) result->items[i] = ta->items[i];
            for (uint32_t i = 0; i < tb->count; i++) {
                result->items[ta->count + i] = tb->items[i];
            }
            jaiGCPopRoots(2);
            *out = OBJ_VAL(result);
            return true;
        }
    }

    if (op == OP_MUL) {
        if (IS_STRING(a) && IS_INT(b)) return repeatString(AS_STRING(a), AS_INT(b), out);
        if (IS_INT(a) && IS_STRING(b)) return repeatString(AS_STRING(b), AS_INT(a), out);
        if (IS_LIST(a) && IS_INT(b))   return repeatList(AS_LIST(a), AS_INT(b), out);
        if (IS_INT(a) && IS_LIST(b))   return repeatList(AS_LIST(b), AS_INT(a), out);
    }

    bool handled = false;
    if (!binaryDunder(op, a, b, out, &handled)) return false;
    if (handled) return true;

    return badOperands(op, a, b);
}

static bool bitwise(OpCode op, Value a, Value b, Value *out) {
    if (!IS_INT(a) || !IS_INT(b)) {
        bool handled = false;
        if (!binaryDunder(op, a, b, out, &handled)) return false;
        if (handled) return true;
        return badOperands(op, a, b);
    }
    int64_t x = AS_INT(a), y = AS_INT(b);

    switch (op) {
    case OP_BAND: *out = INT_VAL(x & y); return true;
    case OP_BOR:  *out = INT_VAL(x | y); return true;
    case OP_BXOR: *out = INT_VAL(x ^ y); return true;
    case OP_SHL:
        if (y < 0) return jaiThrow(vm.cValueError, "negative shift count");
        /* Bits shifted off the top are discarded, not an overflow: spec §2.5
         * scopes OverflowError to `+ - *`, and `__prim__.shl` has always
         * wrapped. Shifting through the unsigned domain also avoids the C UB
         * of a signed left shift that crosses the sign bit. */
        *out = INT_VAL(y >= 64 ? 0 : (int64_t)((uint64_t)x << (uint64_t)y));
        return true;
    case OP_SHR:
        if (y < 0) return jaiThrow(vm.cValueError, "negative shift count");
        /* Arithmetic shift, saturating: >> 64 or more leaves the sign. */
        *out = INT_VAL(y >= 64 ? (x < 0 ? -1 : 0) : (x >> y));
        return true;
    default:
        return badOperands(op, a, b);
    }
}

static bool compareOp(OpCode op, Value a, Value b, Value *out) {
    int cmp = 0;
    if (!jaiValueCompare(a, b, &cmp)) {
        if (vm.hasException) return false;
        return jaiThrow(vm.cTypeError,
                        "'%s' is not supported between '%s' and '%s'",
                        opSymbol(op), jaiTypeNameStatic(a), jaiTypeNameStatic(b));
    }
    switch (op) {
    case OP_LT: *out = BOOL_VAL(cmp < 0);  return true;
    case OP_LE: *out = BOOL_VAL(cmp <= 0); return true;
    case OP_GT: *out = BOOL_VAL(cmp > 0);  return true;
    case OP_GE: *out = BOOL_VAL(cmp >= 0); return true;
    default:    return badOperands(op, a, b);
    }
}

/* `a == b` when both are strings: the one object case jaiValuesEqual cannot be
 * reached without a call.
 *
 * Every `==` whose operands are not both ints goes out of line through
 * jaiValuesEqual, wrapped in SAVE_STATE/LOAD_STATE because in general it can
 * dispatch to a user `__eq__` and therefore re-enter the interpreter. Two
 * strings can do none of that: jaiValuesEqual's OBJ_STRING arm is exactly
 * jaiStringEquals, which is already inline, allocates nothing and cannot
 * throw. Comparing a scanned character against a literal is the single
 * hottest comparison shape there is -- `text[at] == " "` was 17% of
 * tests/bench/word_freq's scan by sample -- and it was paying a call and eight
 * memory operations to reach a pointer compare. */
JAI_INLINE bool valuesEqualFast(Value a, Value b, bool *equal) {
    if (JAI_UNLIKELY(!IS_OBJ(a) || !IS_OBJ(b))) return false;
    Obj *ao = AS_OBJ(a), *bo = AS_OBJ(b);
    if (JAI_UNLIKELY(ao->type != OBJ_STRING || bo->type != OBJ_STRING))
        return false;
    *equal = jaiStringEquals((ObjString *)ao, (ObjString *)bo);
    return true;
}

/* `element in container`. */
static bool containsOp(Value container, Value element, bool *out) {
    if (IS_STRING(container)) {
        if (!IS_STRING(element)) {
            return jaiThrow(vm.cTypeError,
                            "'in' on a str requires a str, not '%s'",
                            jaiTypeNameStatic(element));
        }
        ObjString *hay = AS_STRING(container), *needle = AS_STRING(element);
        if (needle->length == 0) { *out = true; return true; }
        if (needle->length > hay->length) { *out = false; return true; }
        for (uint32_t i = 0; i + needle->length <= hay->length; i++) {
            if (memcmp(hay->chars + i, needle->chars, needle->length) == 0) {
                *out = true;
                return true;
            }
        }
        *out = false;
        return true;
    }
    if (IS_LIST(container)) {
        ObjList *list = AS_LIST(container);
        for (int i = 0; i < list->count; i++) {
            if (jaiValuesEqual(list->items[i], element)) { *out = true; return true; }
            if (vm.hasException) return false;
        }
        *out = false;
        return true;
    }
    if (IS_TUPLE(container)) {
        ObjTuple *t = AS_TUPLE(container);
        for (uint32_t i = 0; i < t->count; i++) {
            if (jaiValuesEqual(t->items[i], element)) { *out = true; return true; }
            if (vm.hasException) return false;
        }
        *out = false;
        return true;
    }
    if (IS_DICT(container)) {
        Value ignored;
        *out = jaiDictGet(AS_DICT(container), element, &ignored);
        return !vm.hasException;
    }
    if (IS_SET(container)) {
        *out = jaiSetHas(AS_SET(container), element);
        return !vm.hasException;
    }
    if (IS_RANGE(container)) {
        if (!IS_INT(element)) { *out = false; return true; }
        ObjRange *r = AS_RANGE(container);
        int64_t v = AS_INT(element), n = jaiRangeLength(r);
        if (n <= 0) { *out = false; return true; }
        int64_t delta = v - r->start;
        if (r->step == 0 || delta % r->step != 0) { *out = false; return true; }
        int64_t index = delta / r->step;
        *out = index >= 0 && index < n;
        return true;
    }
    if (IS_INSTANCE(container)) {
        ObjClass *klass = AS_INSTANCE(container)->klass;
        if (klass != NULL && !IS_NULL(klass->dunderContains)) {
            Value result, arg = element;
            Value bound = OBJ_VAL(jaiBoundNew(container, klass->dunderContains));
            if (!jaiCallValue(bound, 1, &arg, &result)) return false;
            if (!IS_BOOL(result)) {
                return jaiThrow(vm.cTypeError,
                                "__contains__ must return bool, not %s",
                                jaiTypeNameStatic(result));
            }
            *out = AS_BOOL(result);
            return true;
        }
    }
    return jaiThrow(vm.cTypeError, "'in' is not supported for '%s'",
                    jaiTypeNameStatic(container));
}

static bool unaryNegate(Value v, Value *out) {
    if (IS_INT(v)) {
        if (AS_INT(v) == INT64_MIN) {
            return jaiThrow(vm.cOverflowError, "integer overflow in unary '-'");
        }
        *out = INT_VAL(-AS_INT(v));
        return true;
    }
    if (IS_FLOAT(v)) {
        *out = FLOAT_VAL(-AS_FLOAT(v));
        return true;
    }
    if (IS_INSTANCE(v)) {
        ObjClass *klass = AS_INSTANCE(v)->klass;
        if (klass != NULL && !IS_NULL(klass->dunderNeg)) {
            return jaiCallValue(OBJ_VAL(jaiBoundNew(v, klass->dunderNeg)), 0,
                                NULL, out);
        }
    }
    return jaiThrow(vm.cTypeError, "unary '-' is not supported for '%s'",
                    jaiTypeNameStatic(v));
}

/* ------------------------------------------------------------------ */
/* Indexing and slicing                                                 */
/* ------------------------------------------------------------------ */

/* The part of `c[i]` that can neither allocate, call, nor throw, so the
 * interpreter can answer it without saving and restoring its state.
 *
 * indexGet below is a real call behind SAVE_STATE/LOAD_STATE -- two stores and
 * six loads, one of them a three-deep chase to the constant pool -- and for a
 * string it reaches the general slice machinery (scalar count, sliceCount,
 * step and ASCII analysis) to produce one character. That is what a scanner
 * does per byte: tests/bench/word_freq runs this 956,166 times and gets no
 * help from the JIT, and every lexer written in this language has the same
 * shape.
 *
 * Anything this declines -- a non-int index, out of range, a non-ASCII string,
 * a character not yet in the shared one-byte table, any other container --
 * falls through to indexGet unchanged, so the error messages and the slow
 * paths stay in exactly one place. */
JAI_INLINE bool indexGetFast(Value container, Value index, Value *out) {
    if (JAI_UNLIKELY(!IS_OBJ(container) || !IS_INT(index))) return false;

    Obj *o = AS_OBJ(container);
    const int64_t raw = AS_INT(index);

    if (o->type == OBJ_LIST) {
        ObjList *list = (ObjList *)o;
        int at;
        if (JAI_UNLIKELY(!jaiNormalizeIndex(raw, list->count, &at))) return false;
        *out = list->items[at];
        return true;
    }

    if (o->type == OBJ_STRING) {
        ObjString *s = (ObjString *)o;
        /* Indexing is by scalar. `scalars` is UINT32_MAX until something asks,
         * so the first index of any string goes the slow way and fills it in;
         * after that this is the ASCII test, one byte per scalar. */
        if (JAI_UNLIKELY(s->scalars != s->length)) return false;
        int at;
        if (JAI_UNLIKELY(!jaiNormalizeIndex(raw, (int)s->length, &at))) return false;
        const unsigned char c = (unsigned char)s->chars[at];
        if (JAI_UNLIKELY(c >= 128)) return false;
        /* Every slot is filled by jaiVMInit, so there is no null to test. */
        *out = OBJ_VAL(jaiAsciiCharTable()[c]);
        return true;
    }

    if (o->type == OBJ_TUPLE) {
        ObjTuple *t = (ObjTuple *)o;
        int at;
        if (JAI_UNLIKELY(!jaiNormalizeIndex(raw, (int)t->count, &at))) return false;
        *out = t->items[at];
        return true;
    }

    return false;
}

static bool indexGet(Value container, Value index, Value *out) {
    if (IS_LIST(container)) {
        if (!IS_INT(index)) {
            return jaiThrow(vm.cTypeError, "list indices must be int, not '%s'",
                            jaiTypeNameStatic(index));
        }
        ObjList *list = AS_LIST(container);
        int at;
        if (!jaiNormalizeIndex(AS_INT(index), list->count, &at)) {
            return jaiThrow(vm.cIndexError,
                            "list index %" PRId64 " out of range for length %d",
                            AS_INT(index), list->count);
        }
        *out = list->items[at];
        return true;
    }
    if (IS_TUPLE(container)) {
        if (!IS_INT(index)) {
            return jaiThrow(vm.cTypeError, "tuple indices must be int, not '%s'",
                            jaiTypeNameStatic(index));
        }
        ObjTuple *t = AS_TUPLE(container);
        int at;
        if (!jaiNormalizeIndex(AS_INT(index), (int)t->count, &at)) {
            return jaiThrow(vm.cIndexError,
                            "tuple index %" PRId64 " out of range for length %u",
                            AS_INT(index), (unsigned)t->count);
        }
        *out = t->items[at];
        return true;
    }
    if (IS_STRING(container)) {
        if (!IS_INT(index)) {
            return jaiThrow(vm.cTypeError, "str indices must be int, not '%s'",
                            jaiTypeNameStatic(index));
        }
        ObjString *s = AS_STRING(container);
        int at;
        if (!jaiNormalizeIndex(AS_INT(index), (int)jaiStringScalarCount(s), &at)) {
            return jaiThrow(vm.cIndexError,
                            "str index %" PRId64 " out of range for length %u",
                            AS_INT(index), (unsigned)jaiStringScalarCount(s));
        }
        ObjString *scalar = jaiStringSlice(s, at, at + 1, 1);
        if (scalar == NULL) return false;
        *out = OBJ_VAL(scalar);
        return true;
    }
    if (IS_BYTES(container)) {
        if (!IS_INT(index)) {
            return jaiThrow(vm.cTypeError, "bytes indices must be int, not '%s'",
                            jaiTypeNameStatic(index));
        }
        ObjBytes *b = AS_BYTES(container);
        int at;
        if (!jaiNormalizeIndex(AS_INT(index), (int)b->length, &at)) {
            return jaiThrow(vm.cIndexError,
                            "bytes index %" PRId64 " out of range for length %u",
                            AS_INT(index), (unsigned)b->length);
        }
        *out = INT_VAL(b->data[at]);
        return true;
    }
    if (IS_DICT(container)) {
        if (jaiDictGet(AS_DICT(container), index, out)) return true;
        if (vm.hasException) return false;
        char buf[96];
        describeValue(index, buf, sizeof buf);
        return jaiThrow(vm.cKeyError, "key %s not found", buf);
    }
    if (IS_RANGE(container)) {
        if (!IS_INT(index)) {
            return jaiThrow(vm.cTypeError, "range indices must be int, not '%s'",
                            jaiTypeNameStatic(index));
        }
        ObjRange *r = AS_RANGE(container);
        int64_t n = jaiRangeLength(r);
        int at;
        if (n > INT32_MAX || !jaiNormalizeIndex(AS_INT(index), (int)n, &at)) {
            return jaiThrow(vm.cIndexError, "range index %" PRId64 " out of range",
                            AS_INT(index));
        }
        *out = INT_VAL(r->start + (int64_t)at * r->step);
        return true;
    }
    if (IS_INSTANCE(container)) {
        ObjClass *klass = AS_INSTANCE(container)->klass;
        if (klass != NULL && !IS_NULL(klass->dunderGetItem)) {
            Value arg = index;
            return jaiCallValue(OBJ_VAL(jaiBoundNew(container, klass->dunderGetItem)),
                                1, &arg, out);
        }
    }
    return jaiThrow(vm.cTypeError, "'%s' value is not indexable",
                    jaiTypeNameStatic(container));
}

static bool indexSet(Value container, Value index, Value value) {
    if (IS_LIST(container)) {
        if (!IS_INT(index)) {
            return jaiThrow(vm.cTypeError, "list indices must be int, not '%s'",
                            jaiTypeNameStatic(index));
        }
        ObjList *list = AS_LIST(container);
        int at;
        if (!jaiNormalizeIndex(AS_INT(index), list->count, &at)) {
            return jaiThrow(vm.cIndexError,
                            "list index %" PRId64 " out of range for length %d",
                            AS_INT(index), list->count);
        }
        if (!jaiCheckKind(list->elemKind, value, "an element")) return false;
        list->items[at] = value;
        jaiListTouch(list);      /* the count is unchanged; only the version tells */
        return true;
    }
    if (IS_DICT(container)) {
        (void)jaiDictSet(AS_DICT(container), index, value);
        return !vm.hasException;
    }
    if (IS_SET(container)) {
        return jaiThrow(vm.cTypeError, "set does not support index assignment");
    }
    if (IS_INSTANCE(container)) {
        ObjClass *klass = AS_INSTANCE(container)->klass;
        if (klass != NULL && !IS_NULL(klass->dunderSetItem)) {
            Value args[2] = {index, value};
            Value ignored;
            return jaiCallValue(OBJ_VAL(jaiBoundNew(container, klass->dunderSetItem)),
                                2, args, &ignored);
        }
    }
    return jaiThrow(vm.cTypeError, "'%s' value does not support item assignment",
                    jaiTypeNameStatic(container));
}

/* Slice bounds from the optional start/stop/step pushed under the flags. */
static bool sliceBounds(Value startV, Value stopV, Value stepV, bool hasStart,
                        bool hasStop, bool hasStep, int64_t length,
                        int64_t *start, int64_t *stop, int64_t *step) {
    *step = 1;
    if (hasStep) {
        if (!IS_INT(stepV)) {
            return jaiThrow(vm.cTypeError, "slice step must be int, not '%s'",
                            jaiTypeNameStatic(stepV));
        }
        *step = AS_INT(stepV);
        if (*step == 0) {
            return jaiThrow(vm.cValueError, "slice step cannot be zero");
        }
    }
    *start = (*step > 0) ? 0 : length - 1;
    *stop = (*step > 0) ? length : -length - 1;

    if (hasStart) {
        if (!IS_INT(startV)) {
            return jaiThrow(vm.cTypeError, "slice bounds must be int, not '%s'",
                            jaiTypeNameStatic(startV));
        }
        *start = AS_INT(startV);
    }
    if (hasStop) {
        if (!IS_INT(stopV)) {
            return jaiThrow(vm.cTypeError, "slice bounds must be int, not '%s'",
                            jaiTypeNameStatic(stopV));
        }
        *stop = AS_INT(stopV);
    }
    return true;
}

/* The whole of OP_GET_SLICE, so the compiled tier runs this rather than a copy
 * of it. The caller must have saved VM state; returns false with an exception
 * pending. */
bool jaiSliceGet(Value container, Value startValue, Value stopValue,
                 Value stepValue, bool hasStart, bool hasStop, bool hasStep,
                 Value *out) {
    int64_t length;
    if (IS_LIST(container))        length = AS_LIST(container)->count;
    else if (IS_STRING(container)) length = jaiStringScalarCount(AS_STRING(container));
    else if (IS_TUPLE(container))  length = AS_TUPLE(container)->count;
    else {
        return jaiThrow(vm.cTypeError, "'%s' value does not support slicing",
                        jaiTypeNameStatic(container));
    }

    int64_t start, stop, step;
    if (!sliceBounds(startValue, stopValue, stepValue, hasStart, hasStop,
                     hasStep, length, &start, &stop, &step)) {
        return false;
    }
    if (IS_LIST(container)) {
        ObjList *sliced = jaiListSlice(AS_LIST(container), start, stop, step);
        if (sliced == NULL) return false;
        *out = OBJ_VAL(sliced);
        return true;
    }
    if (IS_STRING(container)) {
        ObjString *sliced = jaiStringSlice(AS_STRING(container), start, stop, step);
        if (sliced == NULL) return false;
        *out = OBJ_VAL(sliced);
        return true;
    }
    {
        ObjTuple *tuple = AS_TUPLE(container);
        /* Reuse the list slicer for index arithmetic, then re-wrap. */
        ObjList *temp = jaiListNew((int)tuple->count);
        jaiGCPushRoot(OBJ_VAL(temp));
        for (uint32_t i = 0; i < tuple->count; i++) {
            temp->items[temp->count++] = tuple->items[i];
        }
        ObjList *sliced = jaiListSlice(temp, start, stop, step);
        jaiGCPopRoot();
        if (sliced == NULL) return false;
        jaiGCPushRoot(OBJ_VAL(sliced));
        ObjTuple *outT = jaiTupleNew(sliced->items, sliced->count);
        jaiGCPopRoot();
        *out = OBJ_VAL(outT);
        return true;
    }
}


/* ------------------------------------------------------------------ */
/* defer                                                                */
/* ------------------------------------------------------------------ */

/* Run one deferred block in the frame that registered it.
 *
 * A `defer` body is compiled as a thunk over the *defining* frame's slot
 * numbering and upvalue indices (see emitDefer), because spec §5.4 has it read
 * and write that function's locals. So it is entered with `slots` pointing at
 * the definer's window and `base` at the free stack above it: local reads land
 * in the definer, and the return rewinds only what the thunk itself pushed.
 *
 * A callable that came from somewhere else — a closure value handed to the
 * runtime — has its own window and goes through the ordinary call path. */
static bool callDeferred(CallFrame *definer, Value deferred) {
    if (!IS_CLOSURE(deferred) ||
        !jaiFunctionIsDeferThunk(AS_CLOSURE(deferred)->fn)) {
        Value ignored;
        return jaiCallValue(deferred, 0, NULL, &ignored);
    }

    ObjClosure *thunk = AS_CLOSURE(deferred);
    Value *base = vm.stackTop;
    if (!ensureRoom(base, (int)thunk->fn->maxSlots + JAI_FRAME_SLACK)) return false;

    int frameBase = vm.frameCount;
    if (!pushFrame(thunk, definer->slots)) return false;
    vm.frames[vm.frameCount - 1].base = base;

    JaiRunResult result = run(frameBase);
    vm.stackTop = base;
    return result == JAI_RUN_OK;
}

/* Run and clear this frame's deferred closures, in reverse registration order
 * (spec §5.4). Every exit path goes through here: normal return, a break out
 * of the function, and unwinding. */
static bool runFrameDefers(CallFrame *frame) {
    if (vm.defers.count <= frame->deferBase) return true;

    bool ok = true;
    bool hadException = vm.hasException;
    Value carried = vm.pendingException;
    /* vm.pendingException is a root, but a defer that catches something of its
     * own overwrites it — so the exception being carried needs its own. */
    jaiGCPushRoot(carried);

    while (vm.defers.count > frame->deferBase) {
        Value deferred = vm.defers.data[--vm.defers.count];
        if (!valueIsCallable(deferred)) continue;
        /* Each defer runs with a clean exception state so that a failure in
         * one does not look like the exception that is already unwinding. */
        vm.hasException = false;
        if (!callDeferred(frame, deferred)) {
            /* The newest failure wins; the one being carried is replaced. */
            hadException = true;
            jaiGCPopRoot();
            carried = vm.pendingException;
            jaiGCPushRoot(carried);
            ok = false;
        }
    }

    jaiGCPopRoot();
    vm.hasException = hadException;
    vm.pendingException = carried;
    return ok || !hadException;
}

/* ------------------------------------------------------------------ */
/* Unwinding                                                            */
/* ------------------------------------------------------------------ */

/* Resolve a handler's type constant against the exception being thrown.
 * UINT32_MAX is the catch-all; the finally sentinel matches everything too. */
static bool handlerMatches(ObjFunction *fn, uint32_t typeConst, Value exception) {
    if (typeConst == JAI_HANDLER_CATCH_ALL || typeConst == JAI_HANDLER_FINALLY) {
        return true;
    }
    if (typeConst >= (uint32_t)fn->chunk.constants.count) return false;
    Value expected = fn->chunk.constants.data[typeConst];

    ObjClass *actual = NULL;
    if (IS_INSTANCE(exception)) actual = AS_INSTANCE(exception)->klass;
    else if (IS_CLASS(exception)) actual = AS_CLASS(exception);

    if (IS_TUPLE(expected)) {
        ObjTuple *t = AS_TUPLE(expected);
        for (uint32_t i = 0; i < t->count; i++) {
            if (IS_CLASS(t->items[i]) &&
                jaiClassIsSubclassOf(actual, AS_CLASS(t->items[i]))) {
                return true;
            }
        }
        return false;
    }
    if (IS_CLASS(expected)) return jaiClassIsSubclassOf(actual, AS_CLASS(expected));
    if (IS_STRING(expected)) {
        /* The class was not resolvable at compile time; look it up now. */
        Value klass;
        ObjString *name = AS_STRING(expected);
        /* An untyped `catch e` compiles to a handler on `any`, which is the
         * u24-representable spelling of the catch-all sentinel. */
        if (name->length == 3 && memcmp(name->chars, "any", 3) == 0) return true;
        CallFrame *frame = topFrame();
        if ((frame != NULL && frame->module != NULL &&
             jaiModuleGet(frame->module, name, &klass)) ||
            (vm.builtins != NULL && jaiModuleGet(vm.builtins, name, &klass))) {
            return IS_CLASS(klass) && jaiClassIsSubclassOf(actual, AS_CLASS(klass));
        }
        return false;
    }
    return false;
}

/* Discard the innermost frame after its defers have run. */
static void popFrameForUnwind(void) {
    CallFrame *frame = &vm.frames[vm.frameCount - 1];
    (void)runFrameDefers(frame);
    closeUpvalues(frame->base);
    if (vm.handlers.count > frame->handlerBase) vm.handlers.count = frame->handlerBase;
    if (vm.defers.count > frame->deferBase) vm.defers.count = frame->deferBase;
    vm.stackTop = frame->base;
    vm.frameCount--;
}

/* Enter `handler` in the frame at `frameIndex`: restore the stack, publish the
 * exception, and leave the VM ready to resume at the handler's code offset. */
static void enterHandler(int frameIndex, uint32_t handlerOffset, Value *restoreTop,
                         bool isFinally) {
    while (vm.frameCount > frameIndex + 1) popFrameForUnwind();

    CallFrame *frame = &vm.frames[frameIndex];
    vm.stackTop = restoreTop;
    frame->ip = frameChunk(frame)->code + handlerOffset;

    /* The exception stays in vm.pendingException so GET_EXC, MATCH_EXC and
     * RERAISE can reach it — it is deliberately *not* pushed, because spec
     * §3.8 keeps handlers off the value stack and the handler code codegen
     * emits opens with its own GET_EXC. hasException goes false because the
     * unwind has found its destination. */
    vm.hasException = false;
    if (isFinally) sFinallyPending++;
}

/* Walk handlers and frames until one matches, popping frames (and running
 * their defers) as it goes. Returns false when the exception escapes the
 * window owned by this run(), with the frames already unwound to `base`. */
static bool unwindToHandler(int base, uint32_t throwOffset) {
    Value exception = vm.pendingException;
    bool haveOffset = true;

    /* Snapshot the call chain before any of it is discarded. Handlers are the
     * common case, so this is thrown away more often than it is used — but an
     * exception is already an exceptional cost, and a traceback assembled
     * after the fact would be missing exactly the frames that matter. */
    captureTraceback();

    while (vm.frameCount > base) {
        int frameIndex = vm.frameCount - 1;
        CallFrame *frame = &vm.frames[frameIndex];
        ObjFunction *fn = frame->closure->fn;

        uint32_t offset = throwOffset;
        if (!haveOffset) {
            ptrdiff_t at = frame->ip - fn->chunk.code;
            if (at > 0) at--;
            offset = (uint32_t)(at < 0 ? 0 : at);
        }

        /* Dynamic handlers first: they were pushed by the code actually
         * executing and are strictly more precise than the static table. */
        for (int i = vm.handlers.count - 1; i >= frame->handlerBase; i--) {
            ExcHandler *h = &vm.handlers.data[i];
            if (h->frameIndex != frameIndex) continue;
            if (!handlerMatches(fn, h->typeConst, exception)) continue;
            uint32_t handlerOffset = h->handlerOffset;
            Value *restore = h->stackTop;
            bool isFinally = (h->typeConst == JAI_HANDLER_FINALLY);
            vm.handlers.count = i;   /* the handler consumes itself */
            enterHandler(frameIndex, handlerOffset, restore, isFinally);
            freeSavedTraceback();
            return true;
        }

        /* Then the per-function exception table (spec §3.8): the narrowest
         * region covering the faulting instruction wins. */
        const ExceptionEntry *best = NULL;
        for (uint16_t i = 0; i < fn->exceptionCount; i++) {
            const ExceptionEntry *e = &fn->exceptions[i];
            if (offset < e->start || offset >= e->end) continue;
            if (!handlerMatches(fn, e->typeConst, exception)) continue;
            if (best == NULL || (e->end - e->start) < (best->end - best->start)) {
                best = e;
            }
        }
        if (best != NULL) {
            /* Temporaries are gone; locals survive. The handler code is
             * generated against exactly this depth. */
            Value *restore = frame->slots + frameWindowSize(fn);
            vm.handlers.count = frame->handlerBase;
            enterHandler(frameIndex, best->handler, restore, false);
            freeSavedTraceback();
            return true;
        }

        popFrameForUnwind();
        haveOffset = false;
        exception = vm.pendingException;   /* a defer may have replaced it */
    }

    vm.hasException = true;
    vm.pendingException = exception;
    sFinallyPending = 0;
    return false;
}

/* ------------------------------------------------------------------ */
/* Safepoint                                                            */
/* ------------------------------------------------------------------ */

static void interruptHandler(int signum) {
    (void)signum;
    jaiInterrupted = 1;
}

static void installInterruptHandler(void) {
    /* Only claim SIGINT when nothing else has: a host embedding the VM (or the
     * CLI's own REPL handler) must keep priority. */
    void (*previous)(int) = signal(SIGINT, interruptHandler);
    if (previous == SIG_DFL) {
        sSignalInstalled = true;
        return;
    }
    (void)signal(SIGINT, previous);
}

static void removeInterruptHandler(void) {
    if (!sSignalInstalled) return;
    (void)signal(SIGINT, SIG_DFL);
    sSignalInstalled = false;
}

/* The LOOP safepoint (spec/BYTECODE.md §10): the one place a long-running
 * program can be interrupted and the one place a collection is guaranteed to
 * be able to run. */
static bool safepoint(void) {
    if (JAI_UNLIKELY(jaiInterrupted == 2)) {
        jaiInterrupted = 0;
        if (vm.frameCount > 0) {
            CallFrame *top = &vm.frames[vm.frameCount - 1];
            if (!jaiJitSample(top->closure,
                              (uint32_t)(top->ip -
                                         top->closure->fn->chunk.code))) {
                return false;   /* the compiled loop raised */
            }
        }
    } else if (vm.frameCount > 0) {
        /* A loop that already has a compiled form enters it here, on the back
         * edge, rather than waiting for the next timer tick.
         *
         * Waiting was the whole reason the compiled form did not matter. A
         * tick arrives at 4kHz and only counts when it lands on a back edge,
         * so a loop that runs a hundred iterations and exits is almost never
         * entered: matrix_mul's innermost body runs 1.7 million times and the
         * compiled form was entered fewer than twenty thousand. Reaching it
         * from the back edge instead costs one load and a compare on a path
         * that has no compiled form, measured at about 1% of the suite. */
        CallFrame *top = &vm.frames[vm.frameCount - 1];
        ObjFunction *f = top->closure->fn;
        if (f->osrHot) {
            uint32_t at = (uint32_t)(top->ip - f->chunk.code);
            bool compiled = false;
            for (unsigned i = 0; i < f->osrCount; i++) {
                if (f->osrForms[i].top == at) { compiled = true; break; }
            }
            if (compiled) {
                uint32_t resumeAt = 0;
                int outcome = jaiJitEnterOsr(top->closure, at, &resumeAt);
                if (outcome == 2) return false;
                if (outcome == 1) {
                    top->ip = f->chunk.code + resumeAt;
                    f->osrDeclines = 0;
                } else if (++f->osrDeclines >= 8) {
                    f->osrHot = false;
                }
            }
        }
    }
    /* Only 1 is Ctrl-C. A tick that arrived while compiled code was running
     * leaves 2 here, and treating any non-zero value as an interrupt turned
     * every long compiled loop into a spurious RuntimeError. */
    if (JAI_UNLIKELY(jaiInterrupted == 1)) {
        jaiInterrupted = 0;
        return jaiThrow(vm.cRuntimeError, "interrupted");
    }
    jaiGCMaybeCollect();
    return true;
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
static int prepareKeywordCall(int posCount, ObjTuple *names,
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

/* ------------------------------------------------------------------ */
/* Type tests                                                           */
/* ------------------------------------------------------------------ */

/* Does `value` satisfy the type named by a constant? The constant is a class,
 * a trait, an enum, or — when the type was not resolvable at compile time —
 * the spelling of a primitive type. */
static bool valueMatchesType(Value value, Value typeConstant) {
    if (IS_CLASS(typeConstant)) {
        ObjClass *expected = AS_CLASS(typeConstant);
        if (IS_INSTANCE(value)) {
            return jaiClassIsSubclassOf(AS_INSTANCE(value)->klass, expected);
        }
        return IS_CLASS(value) && jaiClassIsSubclassOf(AS_CLASS(value), expected);
    }
    if (IS_TRAIT(typeConstant)) {
        return IS_INSTANCE(value) &&
               jaiClassImplements(AS_INSTANCE(value)->klass, AS_TRAIT(typeConstant));
    }
    if (IS_ENUM(typeConstant)) {
        return IS_ENUM_VAL(value) && AS_ENUM_VAL(value)->type == AS_ENUM(typeConstant);
    }
    if (IS_STRING(typeConstant)) {
        ObjString *name = AS_STRING(typeConstant);
        if (name->length == 3 && memcmp(name->chars, "any", 3) == 0) return true;
        /* `Enum.Variant`: codegen writes this when the enum came from another
         * module, so its tag numbering was not visible where the pattern was
         * compiled. The variant list is on the runtime enum, so resolve it
         * here. */
        const char *dot = memchr(name->chars, '.', name->length);
        if (dot != NULL) {
            if (!IS_ENUM_VAL(value)) return false;
            ObjEnumVal *ev = AS_ENUM_VAL(value);
            size_t typeLen = (size_t)(dot - name->chars);
            if (ev->type == NULL || ev->type->name == NULL ||
                ev->type->name->length != typeLen ||
                memcmp(ev->type->name->chars, name->chars, typeLen) != 0)
                return false;
            const char *want = dot + 1;
            size_t wantLen = name->length - typeLen - 1;
            if (ev->tag >= ev->type->variantCount) return false;
            ObjString *have = ev->type->variants[ev->tag].name;
            return have != NULL && have->length == wantLen &&
                   memcmp(have->chars, want, wantLen) == 0;
        }
        /* A declared type is a name here because codegen cannot see the
         * runtime object; resolving it now is what makes a guard accept a
         * subclass or a trait implementer rather than only the exact name. */
        Value declared;
        CallFrame *frame = topFrame();
        if ((frame != NULL && frame->module != NULL &&
             jaiModuleGet(frame->module, name, &declared)) ||
            (vm.builtins != NULL && jaiModuleGet(vm.builtins, name, &declared))) {
            if (IS_CLASS(declared) || IS_TRAIT(declared) || IS_ENUM(declared)) {
                return valueMatchesType(value, declared);
            }
        }
        return strcmp(jaiTypeNameStatic(value), name->chars) == 0;
    }
    if (IS_TUPLE(typeConstant)) {          /* a union type */
        ObjTuple *members = AS_TUPLE(typeConstant);
        for (uint32_t i = 0; i < members->count; i++) {
            if (valueMatchesType(value, members->items[i])) return true;
        }
        return false;
    }
    if (IS_NULL(typeConstant)) return IS_NULL(value);
    return false;
}

/* The builtin type names (spec §9) are bound as conversion *functions*, so
 * `x is int` has an ObjNative on its right and there is no runtime type object
 * to compare against. Recognised by identity against the binding in
 * `vm.builtins`, not by name alone, so a user value that happens to be called
 * `int` is still just a value. */
static bool nativeNamesType(ObjNative *native) {
    static const char *const kTypeNames[] = {
        "int", "float", "str", "bool", "bytes",
        "list", "dict", "set", "tuple", "range",
    };
    if (native->name == NULL || vm.builtins == NULL) return false;
    for (size_t i = 0; i < sizeof kTypeNames / sizeof kTypeNames[0]; i++) {
        if (strcmp(native->name->chars, kTypeNames[i]) != 0) continue;
        Value bound;
        return jaiModuleGet(vm.builtins, native->name, &bound) &&
               IS_NATIVE(bound) && AS_NATIVE(bound) == native;
    }
    return false;
}

/* `is` compares identity (spec §4.2) *except* when its right operand denotes a
 * type, which is the reading the checker narrows on — see `typeOperand` in
 * sema/check.c, whose comment says the type reading wins over the value one.
 * Without this the two disagree: every `if x is C` guard narrowed statically
 * and then took the false branch at run time. Deciding it here rather than in
 * codegen keeps a class reached through an import, an alias or a variable on
 * the same path as one named directly, and keeps both front ends honest. */
static bool valueIsTest(Value subject, Value target) {
    if (IS_CLASS(target) || IS_TRAIT(target) || IS_ENUM(target)) {
        return valueMatchesType(subject, target);
    }
    if (IS_NATIVE(target) && nativeNamesType(AS_NATIVE(target))) {
        return valueMatchesType(subject, OBJ_VAL(AS_NATIVE(target)->name));
    }
    return jaiValuesIdentical(subject, target);
}

static const char *typeConstantName(Value typeConstant) {
    if (IS_CLASS(typeConstant) && AS_CLASS(typeConstant)->name != NULL) {
        return AS_CLASS(typeConstant)->name->chars;
    }
    if (IS_TRAIT(typeConstant) && AS_TRAIT(typeConstant)->name != NULL) {
        return AS_TRAIT(typeConstant)->name->chars;
    }
    if (IS_ENUM(typeConstant) && AS_ENUM(typeConstant)->name != NULL) {
        return AS_ENUM(typeConstant)->name->chars;
    }
    if (IS_STRING(typeConstant)) return AS_CSTRING(typeConstant);
    return jaiTypeNameStatic(typeConstant);
}

static bool classDeclareField(ObjClass *klass, ObjString *name, uint8_t info) {
    if (jaiClassFieldInfo(klass, name) != NULL) return true;   /* redeclared */
    if (klass->fieldCount == UINT16_MAX) {
        return jaiThrow(vm.cRuntimeError, "class '%s' has too many fields",
                        klass->name != NULL ? klass->name->chars : "?");
    }
    uint16_t oldCount = klass->fieldCount;
    klass->fields = JAI_GROW_ARRAY(FieldInfo, klass->fields, oldCount, oldCount + 1);
    FieldInfo *field = &klass->fields[oldCount];
    field->name = name;
    field->slot = oldCount;
    field->visibility = (Visibility)(info & 0x3);
    field->isStatic = (info & 0x4) != 0;
    field->isLet = (info & 0x8) != 0;
    /* Bits 4-7 are the declared kind (spec §3.7). Zero -- which is every image
     * written before this encoding existed -- is FIELD_KIND_ANY. */
    field->typeId = (uint32_t)((info >> 4) & 0xF);
    klass->fieldCount = (uint16_t)(oldCount + 1);

    /* A static field lives on the class, not in an instance window, so it needs
     * an entry in `statics` before anything can read or assign it: both
     * jaiSetProperty and the class-member read refuse a name that is not
     * already there. Codegen stores its initialiser right after this. */
    if (field->isStatic) {
        jaiGCPushRoot(OBJ_VAL(klass));
        (void)jaiTableSetInterned(&klass->statics, name, NULL_VAL);
        jaiGCPopRoot();
    }
    return true;
}

/* The class-spec constant that codegen emits for OP_CLASS is a 6-tuple
 *     (INT kind, STR name, STR|NULL superName, BOOL isAbstract,
 *      TUPLE required, TUPLE variants)
 * where kind selects class/trait/enum (spec/BYTECODE.md §6). `required` and
 * `variants` are flattened pair lists; superName is informational only,
 * because the link is made by the OP_INHERIT that codegen emits right after.
 *
 * This tuple is the whole spec. There is no separate on-disk class record: it
 * serialises as an ordinary tuple constant, and the fields, methods and traits
 * it does not carry arrive as FIELD_DEF, METHOD and IMPL_TRAIT in the class
 * body, so a class is built by running code. */
#define SPEC_KIND_CLASS 0
#define SPEC_KIND_TRAIT 1
#define SPEC_KIND_ENUM  2

/* Builds the runtime object a class spec describes. Returns NULL and leaves an
 * exception pending on a malformed spec. */
static Obj *classSpecInstantiate(Value spec) {
    if (!IS_TUPLE(spec) || AS_TUPLE(spec)->count != 6) {
        jaiThrow(vm.cRuntimeError, "CLASS operand is not a class spec");
        return NULL;
    }
    const Value *items = AS_TUPLE(spec)->items;
    if (!IS_INT(items[0]) || !IS_STRING(items[1])) {
        jaiThrow(vm.cRuntimeError, "CLASS operand is not a class spec");
        return NULL;
    }
    int64_t kind = AS_INT(items[0]);
    ObjString *name = AS_STRING(items[1]);

    switch (kind) {
    case SPEC_KIND_CLASS: {
        ObjClass *klass = jaiClassNew(name, NULL);
        klass->isAbstract = IS_BOOL(items[3]) && AS_BOOL(items[3]);
        return (Obj *)klass;
    }

    case SPEC_KIND_TRAIT: {
        if (!IS_TUPLE(items[4])) {
            jaiThrow(vm.cRuntimeError, "trait spec has no requirement list");
            return NULL;
        }
        ObjTrait *trait = jaiTraitNew(name);
        jaiGCPushRoot(OBJ_VAL(trait));
        ObjTuple *required = AS_TUPLE(items[4]);
        for (uint32_t i = 0; i + 1 < required->count; i += 2) {
            if (!IS_STRING(required->items[i]) ||
                !IS_INT(required->items[i + 1])) {
                continue;
            }
            jaiTableSet(&trait->required, required->items[i],
                        required->items[i + 1]);
        }
        jaiGCPopRoot();
        return (Obj *)trait;
    }

    case SPEC_KIND_ENUM: {
        if (!IS_TUPLE(items[5])) {
            jaiThrow(vm.cRuntimeError, "enum spec has no variant list");
            return NULL;
        }
        ObjTuple *variants = AS_TUPLE(items[5]);
        uint32_t count = variants->count / 2;
        if (count > UINT16_MAX) {
            jaiThrow(vm.cRuntimeError, "enum '%s' has too many variants",
                     name->chars);
            return NULL;
        }
        ObjEnum *e = jaiEnumNew(name);
        jaiGCPushRoot(OBJ_VAL(e));
        /* Variant names and field names are already interned constants held
         * live by the chunk, so only the arrays need allocating here. */
        e->variants = count > 0 ? JAI_ALLOC(EnumVariant, count) : NULL;
        e->variantCount = (uint16_t)count;
        for (uint32_t i = 0; i < count; i++) {
            EnumVariant *v = &e->variants[i];
            Value vname = variants->items[i * 2];
            Value fields = variants->items[i * 2 + 1];
            v->name = IS_STRING(vname) ? AS_STRING(vname) : name;
            uint32_t arity = IS_TUPLE(fields) ? AS_TUPLE(fields)->count : 0;
            if (arity > 255) arity = 255;
            v->arity = (uint8_t)arity;
            v->unit = NULL;   /* both filled on the variant's first mention */
            v->ctor = NULL;
            v->fieldNames = arity > 0 ? JAI_ALLOC(ObjString *, arity) : NULL;
            for (uint32_t p = 0; p < arity; p++) {
                Value fname = AS_TUPLE(fields)->items[p];
                v->fieldNames[p] = IS_STRING(fname) ? AS_STRING(fname) : NULL;
            }
        }
        jaiGCPopRoot();
        return (Obj *)e;
    }

    default:
        jaiThrow(vm.cRuntimeError, "unknown class-spec kind %lld",
                 (long long)kind);
        return NULL;
    }
}

#ifdef JAI_OPCODE_STATS
uint64_t jaiOpCounts[OP_COUNT];
#endif

#ifdef JAI_PROP_STATS
uint64_t jaiPropRecv[32];
#endif

/* ------------------------------------------------------------------ */
/* The interpreter loop                                                 */
/* ------------------------------------------------------------------ */

#if JAI_COMPUTED_GOTO
#  define VM_CASE(name)  L_##name
#  ifdef JAI_OPCODE_STATS
#    define VM_NEXT()      do { DISPATCH_TRACE(); instStart = ip;                \
                                jaiOpCounts[*ip]++;                              \
                                goto *jaiDispatchTable[*ip++]; } while (0)
#  else
#    define VM_NEXT()      do { DISPATCH_TRACE(); instStart = ip;                \
                              goto *jaiDispatchTable[*ip++]; } while (0)
#  endif
/* Clang emits exactly ONE indirect branch per function for computed goto --
 * CodeGenFunction caches a single indirect-goto block and every `goto *p`
 * branches to it -- so all 132 opcodes share one entry in the branch-target
 * predictor no matter how this is written. VM_NEXT_HINT names the successor
 * the census says is overwhelmingly likely and reaches it by a direct branch,
 * which the predictor keys on its own address. A wrong guess costs one
 * compare and falls through to the shared branch, so the only hints worth
 * taking are the lopsided ones: a 50/50 hint replaces a predictable indirect
 * branch with an unpredictable direct one. */
#  ifdef JAI_OPCODE_STATS
/* Counted on both arms, and before the compare, so a hinted dispatch is not
 * invisible to the census. It used to be: the five hint sites cover OP_LOOP,
 * OP_INC_LOCAL and OP_BIND, and loop_sum's OP_LOOP -- 14.28% of that run --
 * reported as zero. sum(jaiOpCounts) == vm.instructionCount is the invariant
 * that says no dispatch path is missing, and scripts/opstats_check.sh is what
 * asserts it. */
#    define VM_NEXT_HINT(nextOp)                                               \
        do { DISPATCH_TRACE(); instStart = ip;                                 \
             jaiOpCounts[*ip]++;                                               \
             if (JAI_LIKELY(*ip == (nextOp))) { ip++; goto L_##nextOp; }       \
             goto *jaiDispatchTable[*ip++]; } while (0)
#  else
#    define VM_NEXT_HINT(nextOp)                                               \
        do { DISPATCH_TRACE(); instStart = ip;                                 \
             if (JAI_LIKELY(*ip == (nextOp))) { ip++; goto L_##nextOp; }       \
             goto *jaiDispatchTable[*ip++]; } while (0)
#  endif
/* The trailing `;` lets the case labels that follow live in a plain block;
 * labels have function scope, so the nesting costs nothing. */
#  define VM_DISPATCH()  VM_NEXT();
#else
#  define VM_CASE(name)  case name
#  define VM_NEXT()      goto vmDispatch
#  define VM_NEXT_HINT(nextOp)  ((void)(nextOp), VM_NEXT())
#  define VM_DISPATCH()  vmDispatch:                                           \
                         DISPATCH_TRACE();                                     \
                         instStart = ip;                                       \
                         switch ((OpCode)*ip++)
#endif

/* Hot state lives in locals; these two macros are the only bridge back to the
 * VM struct. Anything that can allocate, call, or throw must be bracketed by
 * them — the collector reads vm.stackTop, and a stale one frees live values. */
#define SAVE_STATE()                                                           \
    do {                                                                       \
        frame->ip = ip;                                                        \
        vm.stackTop = stackTop;                                                \
    } while (0)

/* After a call that completed without pushing a frame, the frame, its ip, its
 * slots and its constants are all exactly as they were -- the frame array and
 * the value stack are both fixed-capacity, so neither can have moved. Only the
 * stack top changed. LOAD_STATE's six loads, one of them a three-deep chase to
 * the constant pool, buy nothing there, and every native call was paying
 * them. instStart needs no restoring because VM_NEXT sets it. */
#define LOAD_STACK_ONLY()  (stackTop = vm.stackTop)

#define LOAD_STATE()                                                           \
    do {                                                                       \
        frame = &vm.frames[vm.frameCount - 1];                                 \
        ip = frame->ip;                                                        \
        instStart = ip;                                                        \
        stackTop = vm.stackTop;                                                \
        slots = frame->slots;                                                  \
        constants = frame->closure->fn->chunk.constants.data;                  \
    } while (0)

/* One step of the three iterator kinds that can neither allocate, call, nor
 * throw: a range computes an int, and a list or a tuple reads a slot it has
 * already bounds-checked. Those three are what `for i in 0..n` and
 * `for x in xs` are in every interpreted loop in the language.
 *
 * Worth taking apart from jaiIterNext because of what the general path costs
 * around it rather than in it: SAVE_STATE, a cross-translation-unit call, and
 * LOAD_STATE's six reloads, one of them a three-deep chase through
 * frame->closure->fn->chunk to the constant pool. None of that buys anything
 * when the step cannot move the frame array or the value stack. A list whose
 * version moved, and every other kind, still goes the long way -- that is what
 * ITER_STEP_SLOW is for, and the fast path has written nothing by then. */
typedef enum { ITER_STEP_DONE, ITER_STEP_VALUE, ITER_STEP_SLOW } IterStep;

JAI_INLINE IterStep iterStepFast(ObjIter *it, Value *out) {
    const int64_t index = it->index;

    switch (it->kind) {
        case ITER_RANGE: {
            if (index >= it->limit) return ITER_STEP_DONE;

            ObjRange *const range = AS_RANGE(it->source);
            uint64_t value;

            if (range->step == 1) {
                value = (uint64_t)range->start + (uint64_t)index;
            } else if (range->step == -1) {
                value = (uint64_t)range->start - (uint64_t)index;
            } else {
                value = (uint64_t)range->start +
                        (uint64_t)index * (uint64_t)range->step;
            }

            *out = INT_VAL((int64_t)value);
            it->index = index + 1;
            return ITER_STEP_VALUE;
        }

        case ITER_LIST: {
            ObjList *const list = AS_LIST(it->source);
            /* A mutated list is an error jaiIterNext raises; leave it there. */
            if (JAI_UNLIKELY(list->version != it->version))
                return ITER_STEP_SLOW;
            if (index >= it->limit) return ITER_STEP_DONE;

            *out = list->items[index];
            it->index = index + 1;
            return ITER_STEP_VALUE;
        }

        case ITER_TUPLE:
            if (index >= it->limit) return ITER_STEP_DONE;

            *out = AS_TUPLE(it->source)->items[index];
            it->index = index + 1;
            return ITER_STEP_VALUE;

        default:
            return ITER_STEP_SLOW;
    }
}

/* The same step for a loop that binds a PAIR, `for (a, b) in …`. Two things
 * are different and both are the point of OP_FOR_ITER_PAIR.
 *
 * A dict-items iterator is served here rather than declared slow: it is the
 * one built-in kind whose item does not exist until jaiIterNext builds it, and
 * building it is pure cost when the very next instruction takes it apart
 * again. Walking the table writes key and value straight out. Nothing here
 * allocates, calls or throws, so no SAVE_STATE is owed -- jaiTableNext only
 * scans the order array.
 *
 * Every other kind produces its item the ordinary way and the item is split in
 * place, so a list of pairs or a user iterator costs exactly what it did.
 * PAIR_STEP_BAD is separate from PAIR_STEP_SLOW because the step has already
 * advanced by then: retrying it on the slow path would skip an entry before
 * raising. The caller raises from `*a` instead. */
typedef enum {
    PAIR_STEP_DONE, PAIR_STEP_VALUE, PAIR_STEP_BAD, PAIR_STEP_SLOW
} PairStep;

JAI_INLINE bool pairSplit(Value item, Value *a, Value *b) {
    if (IS_TUPLE(item)) {
        ObjTuple *const tuple = AS_TUPLE(item);
        if (tuple->count != 2) return false;
        *a = tuple->items[0];
        *b = tuple->items[1];
        return true;
    }
    if (IS_LIST(item)) {
        ObjList *const list = AS_LIST(item);
        if (list->count != 2) return false;
        *a = list->items[0];
        *b = list->items[1];
        return true;
    }
    return false;
}

/* Word for word what `UNPACK 2 255` raises for `item`, because that is the
 * sequence this replaces and an optimisation may not change a message. */
static bool pairSplitFail(Value item) {
    if (!IS_LIST(item) && !IS_TUPLE(item)) {
        return jaiThrow(vm.cTypeError, "cannot destructure a '%s' value",
                        jaiTypeNameStatic(item));
    }
    const int available = IS_LIST(item) ? AS_LIST(item)->count
                                        : (int)AS_TUPLE(item)->count;
    return jaiThrow(vm.cValueError, "cannot unpack %d value%s into 2 targets",
                    available, available == 1 ? "" : "s");
}

JAI_INLINE PairStep iterStepPairFast(ObjIter *it, Value *a, Value *b) {
    if (it->kind == ITER_DICT_ITEMS) {
        /* A dict that changed under the loop must raise, and jaiIterNext is
         * where that message lives; hand it over untouched and unadvanced. */
        JaiTable *const table = &AS_DICT(it->source)->table;
        if (JAI_UNLIKELY(table->version != it->version)) return PAIR_STEP_SLOW;

        int slot = (int)it->index;
        Value key, value;
        if (!jaiTableNext(table, &slot, &key, &value)) {
            it->index = slot;
            return PAIR_STEP_DONE;
        }
        it->index = slot;
        *a = key;
        *b = value;
        return PAIR_STEP_VALUE;
    }

    Value item;
    switch (iterStepFast(it, &item)) {
    case ITER_STEP_VALUE:
        return pairSplit(item, a, b) ? PAIR_STEP_VALUE
                                     : (*a = item, PAIR_STEP_BAD);
    case ITER_STEP_DONE:
        return PAIR_STEP_DONE;
    default:
        return PAIR_STEP_SLOW;
    }
}

#define READ_BYTE()  (*ip++)
#define READ_I8()    ((int8_t)*ip++)
#define READ_U16()   (ip += 2, jaiReadU16(ip - 2))
#define READ_U24()   (ip += 3, jaiReadU24(ip - 3))
#define READ_I16()   (ip += 2, jaiReadI16(ip - 2))
#define READ_CONST() (constants[READ_U24()])

#define PUSH(v)  (*stackTop++ = (v))
#define POP()    (*(--stackTop))
#define PEEK(d)  (stackTop[-1 - (d)])
#define DROP(n)  (stackTop -= (n))

/* Raise and unwind. The state must be current: the unwinder inspects the
 * frame's ip to find the protected region the fault occurred in. */
#define THROW(...)                                                             \
    do {                                                                       \
        SAVE_STATE();                                                          \
        (void)jaiThrow(__VA_ARGS__);                                           \
        goto vmThrow;                                                          \
    } while (0)

#define DISPATCH_TRACE()                                                       \
    do {                                                                       \
        if (JAI_UNLIKELY(countInsts)) {                                        \
            vm.instructionCount++;                                             \
            if (vm.debugTrace) {                                               \
                vm.stackTop = stackTop;                                        \
                traceInstruction(frame, ip);                                   \
            }                                                                  \
        }                                                                      \
    } while (0)

static void traceInstruction(CallFrame *frame, const uint8_t *ip) {
    jaiVMPrintStack(stdout);
    Chunk *chunk = frameChunk(frame);
    (void)jaiDisassembleInstruction(stdout, chunk, (int)(ip - chunk->code));
}

/* The program rebound the global `str`, so OP_FORMAT owes its holes a call to
 * whatever that name means now — and owes it only to the holes: the literal
 * runs between them are source text the old lowering never converted, which is
 * what `litmask` records. Converts the top `count` stack slots in place; they
 * are re-addressed through vm.stackTop on every iteration because a call can
 * grow the value stack and move it. */
static bool formatViaUserStr(ObjModule *module, ObjString *name, int count,
                             uint32_t litmask) {
    Value strFn;
    if (module == NULL || !jaiTableGetInterned(&module->globals, name, &strFn)) {
        return true;                     /* vanished again; the builtin applies */
    }
    jaiGCPushRoot(strFn);                /* the call may rebind the name */
    for (int i = 0; i < count; i++) {
        if (litmask & (1u << i)) continue;
        Value arg = vm.stackTop[i - count];
        Value converted;
        if (!jaiCallValue(strFn, 1, &arg, &converted)) {
            jaiGCPopRoot();
            return false;
        }
        vm.stackTop[i - count] = converted;
    }
    jaiGCPopRoot();
    return true;
}

/* Both binary helpers share this shape: pop two, push one, with the slow path
 * free to re-enter the VM through a dunder method. */
#define BINARY(fn, opcode)                                                     \
    do {                                                                       \
        SAVE_STATE();                                                          \
        Value _result;                                                         \
        if (!fn((opcode), stackTop[-2], stackTop[-1], &_result)) goto vmThrow;  \
        LOAD_STATE();                                                          \
        stackTop -= 2;                                                         \
        PUSH(_result);                                                         \
        VM_NEXT();                                                             \
    } while (0)

/* Ordered comparison of two same-typed numbers, decided in the dispatch arm.
 * The generic path calls jaiValueCompare, which was 7.2% of loop_sum's whole
 * run just to answer `int < int`. Falls through to BINARY on anything else.
 * NaN is deliberately excluded: an unordered pair is a TypeError here (spec
 * §3.3), not a false, and compareDoubles is the one that reports it. */
#define CMP_FAST(cop)                                                          \
    do {                                                                       \
        Value _b = stackTop[-1], _a = stackTop[-2];                            \
        if (JAI_LIKELY(IS_INT(_a) && IS_INT(_b))) {                            \
            bool _r = AS_INT(_a) cop AS_INT(_b);                               \
            DROP(1);                                                           \
            stackTop[-1] = BOOL_VAL(_r);                                       \
            VM_NEXT();                                                         \
        }                                                                      \
        if (IS_FLOAT(_a) && IS_FLOAT(_b)) {                                    \
            double _x = AS_FLOAT(_a), _y = AS_FLOAT(_b);                       \
            if (JAI_LIKELY(!isnan(_x) && !isnan(_y))) {                        \
                bool _r = _x cop _y;                                           \
                DROP(1);                                                       \
                stackTop[-1] = BOOL_VAL(_r);                                   \
                VM_NEXT();                                                     \
            }                                                                  \
        }                                                                      \
    } while (0)

/* A jump condition must be a bool. There is no truthiness in Jaithon (spec
 * §5.1) and `any` values reach here unchecked, so the VM is the last line of
 * defence for the rule. */
#define REQUIRE_BOOL(v, what)                                                  \
    do {                                                                       \
        if (JAI_UNLIKELY(!IS_BOOL(v))) {                                       \
            THROW(vm.cTypeError,                                               \
                  "%s must be bool, not '%s'; there is no truthiness in "      \
                  "Jaithon", (what), jaiTypeNameStatic(v));                    \
        }                                                                      \
    } while (0)

static JaiRunResult runLoop(int baseFrameCount) {
#if JAI_COMPUTED_GOTO
    static const void *const jaiDispatchTable[] = {
        [OP_NOP]                = &&L_OP_NOP,
        [OP_CONST]              = &&L_OP_CONST,
        [OP_NULL]               = &&L_OP_NULL,
        [OP_TRUE]               = &&L_OP_TRUE,
        [OP_FALSE]              = &&L_OP_FALSE,
        [OP_INT]                = &&L_OP_INT,
        [OP_POP]                = &&L_OP_POP,
        [OP_POPN]               = &&L_OP_POPN,
        [OP_DUP]                = &&L_OP_DUP,
        [OP_DUP2]               = &&L_OP_DUP2,
        [OP_SWAP]               = &&L_OP_SWAP,
        [OP_ROT3]               = &&L_OP_ROT3,
        [OP_GET_LOCAL]          = &&L_OP_GET_LOCAL,
        [OP_SET_LOCAL]          = &&L_OP_SET_LOCAL,
        [OP_GET_UPVALUE]        = &&L_OP_GET_UPVALUE,
        [OP_SET_UPVALUE]        = &&L_OP_SET_UPVALUE,
        [OP_CLOSE_UPVALUE]      = &&L_OP_CLOSE_UPVALUE,
        [OP_GET_GLOBAL]         = &&L_OP_GET_GLOBAL,
        [OP_SET_GLOBAL]         = &&L_OP_SET_GLOBAL,
        [OP_DEF_GLOBAL]         = &&L_OP_DEF_GLOBAL,
        [OP_GET_MODULE]         = &&L_OP_GET_MODULE,
        [OP_ADD]                = &&L_OP_ADD,
        [OP_SUB]                = &&L_OP_SUB,
        [OP_MUL]                = &&L_OP_MUL,
        [OP_DIV]                = &&L_OP_DIV,
        [OP_FLOORDIV]           = &&L_OP_FLOORDIV,
        [OP_MOD]                = &&L_OP_MOD,
        [OP_POW]                = &&L_OP_POW,
        [OP_ADD_WRAP]           = &&L_OP_ADD_WRAP,
        [OP_SUB_WRAP]           = &&L_OP_SUB_WRAP,
        [OP_MUL_WRAP]           = &&L_OP_MUL_WRAP,
        [OP_NEG]                = &&L_OP_NEG,
        [OP_POS]                = &&L_OP_POS,
        [OP_BAND]               = &&L_OP_BAND,
        [OP_BOR]                = &&L_OP_BOR,
        [OP_BXOR]               = &&L_OP_BXOR,
        [OP_SHL]                = &&L_OP_SHL,
        [OP_SHR]                = &&L_OP_SHR,
        [OP_BNOT]               = &&L_OP_BNOT,
        [OP_EQ]                 = &&L_OP_EQ,
        [OP_NE]                 = &&L_OP_NE,
        [OP_LT]                 = &&L_OP_LT,
        [OP_LE]                 = &&L_OP_LE,
        [OP_GT]                 = &&L_OP_GT,
        [OP_GE]                 = &&L_OP_GE,
        [OP_IS]                 = &&L_OP_IS,
        [OP_IS_NOT]             = &&L_OP_IS_NOT,
        [OP_IN]                 = &&L_OP_IN,
        [OP_NOT_IN]             = &&L_OP_NOT_IN,
        [OP_NOT]                = &&L_OP_NOT,
        [OP_CONCAT]             = &&L_OP_CONCAT,
        [OP_ADD_INT_CONST]      = &&L_OP_ADD_INT_CONST,
        [OP_MOD_INT_CONST]      = &&L_OP_MOD_INT_CONST,
        [OP_ADD_BIND]           = &&L_OP_ADD_BIND,
        [OP_SUB_BIND]           = &&L_OP_SUB_BIND,
        [OP_MUL_BIND]           = &&L_OP_MUL_BIND,
        [OP_ELEM_KIND]          = &&L_OP_ELEM_KIND,
        [OP_GET_ITER_ITEMS]     = &&L_OP_GET_ITER_ITEMS,
        [OP_FOR_ITER_PAIR]      = &&L_OP_FOR_ITER_PAIR,
        [OP_ITER_RANGE]         = &&L_OP_ITER_RANGE,
        [OP_FOR_RANGE_BIND]     = &&L_OP_FOR_RANGE_BIND,
        [OP_INC_LOCAL]          = &&L_OP_INC_LOCAL,
        [OP_CMP_LOCAL_CONST_LT] = &&L_OP_CMP_LOCAL_CONST_LT,
        [OP_GET_LOCAL2]         = &&L_OP_GET_LOCAL2,
        [OP_ADD_LOCALS]         = &&L_OP_ADD_LOCALS,
        [OP_JUMP]               = &&L_OP_JUMP,
        [OP_JUMP_IF_FALSE]      = &&L_OP_JUMP_IF_FALSE,
        [OP_JUMP_IF_TRUE]       = &&L_OP_JUMP_IF_TRUE,
        [OP_JUMP_IF_FALSE_KEEP] = &&L_OP_JUMP_IF_FALSE_KEEP,
        [OP_JUMP_IF_TRUE_KEEP]  = &&L_OP_JUMP_IF_TRUE_KEEP,
        [OP_JUMP_IF_NULL]       = &&L_OP_JUMP_IF_NULL,
        [OP_LOOP]               = &&L_OP_LOOP,
        [OP_GET_ITER]           = &&L_OP_GET_ITER,
        [OP_FOR_ITER]           = &&L_OP_FOR_ITER,
        [OP_CALL]               = &&L_OP_CALL,
        [OP_CALL_KW]            = &&L_OP_CALL_KW,
        [OP_CALL_SPREAD]        = &&L_OP_CALL_SPREAD,
        [OP_INVOKE]             = &&L_OP_INVOKE,
        [OP_SUPER_INVOKE]       = &&L_OP_SUPER_INVOKE,
        [OP_TAIL_CALL]          = &&L_OP_TAIL_CALL,
        [OP_RETURN]             = &&L_OP_RETURN,
        [OP_RETURN_NULL]        = &&L_OP_RETURN_NULL,
        [OP_CLOSURE]            = &&L_OP_CLOSURE,
        [OP_BUILD_LIST]         = &&L_OP_BUILD_LIST,
        [OP_BUILD_DICT]         = &&L_OP_BUILD_DICT,
        [OP_BUILD_SET]          = &&L_OP_BUILD_SET,
        [OP_BUILD_TUPLE]        = &&L_OP_BUILD_TUPLE,
        [OP_BUILD_RANGE]        = &&L_OP_BUILD_RANGE,
        [OP_LIST_APPEND]        = &&L_OP_LIST_APPEND,
        [OP_DICT_INSERT]        = &&L_OP_DICT_INSERT,
        [OP_SET_ADD]            = &&L_OP_SET_ADD,
        [OP_GET_INDEX]          = &&L_OP_GET_INDEX,
        [OP_SET_INDEX]          = &&L_OP_SET_INDEX,
        [OP_GET_SLICE]          = &&L_OP_GET_SLICE,
        [OP_SET_SLICE]          = &&L_OP_SET_SLICE,
        [OP_UNPACK]             = &&L_OP_UNPACK,
        [OP_CLASS]              = &&L_OP_CLASS,
        [OP_INHERIT]            = &&L_OP_INHERIT,
        [OP_IMPL_TRAIT]         = &&L_OP_IMPL_TRAIT,
        [OP_METHOD]             = &&L_OP_METHOD,
        [OP_FIELD_DEF]          = &&L_OP_FIELD_DEF,
        [OP_GET_FIELD]          = &&L_OP_GET_FIELD,
        [OP_SET_FIELD]          = &&L_OP_SET_FIELD,
        [OP_GET_SUPER]          = &&L_OP_GET_SUPER,
        [OP_NEW]                = &&L_OP_NEW,
        [OP_ENUM_NEW]           = &&L_OP_ENUM_NEW,
        [OP_ENUM_TAG]           = &&L_OP_ENUM_TAG,
        [OP_ENUM_FIELD]         = &&L_OP_ENUM_FIELD,
        [OP_IS_INSTANCE]        = &&L_OP_IS_INSTANCE,
        [OP_THROW]              = &&L_OP_THROW,
        [OP_RERAISE]            = &&L_OP_RERAISE,
        [OP_PUSH_HANDLER]       = &&L_OP_PUSH_HANDLER,
        [OP_POP_HANDLER]        = &&L_OP_POP_HANDLER,
        [OP_PUSH_FINALLY]       = &&L_OP_PUSH_FINALLY,
        [OP_END_FINALLY]        = &&L_OP_END_FINALLY,
        [OP_PUSH_DEFER]         = &&L_OP_PUSH_DEFER,
        [OP_RUN_DEFERS]         = &&L_OP_RUN_DEFERS,
        [OP_MATCH_EXC]          = &&L_OP_MATCH_EXC,
        [OP_GET_EXC]            = &&L_OP_GET_EXC,
        [OP_MATCH_CONST]        = &&L_OP_MATCH_CONST,
        [OP_MATCH_RANGE]        = &&L_OP_MATCH_RANGE,
        [OP_MATCH_TYPE]         = &&L_OP_MATCH_TYPE,
        [OP_MATCH_SEQ]          = &&L_OP_MATCH_SEQ,
        [OP_MATCH_FIELDS]       = &&L_OP_MATCH_FIELDS,
        [OP_BIND]               = &&L_OP_BIND,
        [OP_IMPORT]             = &&L_OP_IMPORT,
        [OP_IMPORT_FROM]        = &&L_OP_IMPORT_FROM,
        [OP_EXPORT]             = &&L_OP_EXPORT,
        [OP_ASSERT_FAIL]        = &&L_OP_ASSERT_FAIL,
        [OP_TYPE_GUARD]         = &&L_OP_TYPE_GUARD,
        [OP_HALT]               = &&L_OP_HALT,
        [OP_GET_FIELD_LOCAL]    = &&L_OP_GET_FIELD_LOCAL,
        [OP_FOR_ITER_BIND]      = &&L_OP_FOR_ITER_BIND,
        [OP_JUMP_IF_CMP_FALSE]  = &&L_OP_JUMP_IF_CMP_FALSE,
        [OP_FORMAT]             = &&L_OP_FORMAT,
        [OP_JUMP_IF_CMP_LOCAL_K] = &&L_OP_JUMP_IF_CMP_LOCAL_K,
        [OP_TO_FLOAT]           = &&L_OP_TO_FLOAT,
    };
    /* A designated initialiser leaves a hole as NULL rather than failing to
     * compile, so the count is what catches an opcode added without a case. */
    _Static_assert(sizeof(jaiDispatchTable) / sizeof(jaiDispatchTable[0]) ==
                       OP_COUNT,
                   "dispatch table is out of sync with enum OpCode");
#endif

    CallFrame *frame;
    uint8_t *ip;
    uint8_t *instStart;
    Value *stackTop;
    Value *slots;
    Value *constants;
    Value retval = NULL_VAL;
    /* vm.countInstructions is written once, by the CLI, before anything runs.
     * Read from the struct it is an adrp, a load, a compare and a branch on
     * every single dispatch -- four of the fourteen instructions OP_NOP costs.
     * In a local it is a load off the frame and a cbnz. */
    const bool countInsts = vm.countInstructions;

    LOAD_STATE();

    VM_DISPATCH() {

    /* --- constants and stack (spec §3.1) --- */

    VM_CASE(OP_NOP):
        VM_NEXT();

    VM_CASE(OP_CONST):
        PUSH(READ_CONST());
        VM_NEXT();

    VM_CASE(OP_NULL):
        PUSH(NULL_VAL);
        VM_NEXT();

    VM_CASE(OP_TRUE):
        PUSH(BOOL_VAL(true));
        VM_NEXT();

    VM_CASE(OP_FALSE):
        PUSH(BOOL_VAL(false));
        VM_NEXT();

    VM_CASE(OP_INT):
        PUSH(INT_VAL(READ_I16()));
        VM_NEXT();

    VM_CASE(OP_POP):
        DROP(1);
        VM_NEXT_HINT(OP_LOOP);

    VM_CASE(OP_POPN):
        DROP(READ_BYTE());
        VM_NEXT();

    VM_CASE(OP_DUP): {
        Value top = PEEK(0);
        PUSH(top);
        VM_NEXT();
    }

    VM_CASE(OP_DUP2): {
        Value a = PEEK(1), b = PEEK(0);
        PUSH(a);
        PUSH(b);
        VM_NEXT();
    }

    VM_CASE(OP_SWAP): {
        Value top = stackTop[-1];
        stackTop[-1] = stackTop[-2];
        stackTop[-2] = top;
        VM_NEXT();
    }

    VM_CASE(OP_ROT3): {
        /* a b c -> c a b */
        Value c = stackTop[-1], b = stackTop[-2], a = stackTop[-3];
        stackTop[-3] = c;
        stackTop[-2] = a;
        stackTop[-1] = b;
        VM_NEXT();
    }

    /* --- variables (spec §3.2) --- */

    VM_CASE(OP_GET_LOCAL):
        PUSH(slots[READ_U16()]);
        VM_NEXT();

    VM_CASE(OP_SET_LOCAL): {
        uint16_t slot = READ_U16();
        slots[slot] = PEEK(0);
        VM_NEXT();
    }

    VM_CASE(OP_GET_UPVALUE): {
        uint8_t index = READ_BYTE();
        ObjUpvalue *upvalue = frame->closure->upvalues[index];
        PUSH(upvalue != NULL ? *upvalue->location : NULL_VAL);
        VM_NEXT();
    }

    VM_CASE(OP_SET_UPVALUE): {
        uint8_t index = READ_BYTE();
        ObjUpvalue *upvalue = frame->closure->upvalues[index];
        if (upvalue != NULL) *upvalue->location = PEEK(0);
        VM_NEXT();
    }

    /* Locals live in a slot window, not on the operand stack, so the scope
     * being closed is named by its lowest slot: everything at or above it
     * belongs to that scope. Emitted at each iteration boundary of a loop
     * whose body has a by-reference capture, so the next iteration's
     * OP_CLOSURE allocates a fresh cell (spec §5.2). No SAVE_STATE: closing
     * neither allocates nor throws, and it never reads vm.stackTop. */
    VM_CASE(OP_CLOSE_UPVALUE): {
        uint16_t slot = READ_U16();
        closeUpvalues(frame->slots + slot);
        VM_NEXT();
    }

    VM_CASE(OP_GET_GLOBAL): {
        uint32_t nameIdx = READ_U24();
        uint16_t cacheIdx = READ_U16();
        ObjModule *module = frame->module;
        InlineCache *ic = cacheAt(frameChunk(frame), cacheIdx);

        /* Validated by IDENTITY, not by a version.
         *
         * Global names are interned -- the globals table itself probes by
         * pointer (findEntryInterned) -- so the entry that holds this very
         * ObjString IS this name's binding, whatever has happened to the table
         * since the index was cached. That covers every way the index can go
         * stale at once, with no counter to keep in step:
         *   - a rehash moved things: the slot holds a different key, or none;
         *   - the name was deleted: the key is JAI_TOMBSTONE, VAL_OBJ carrying
         *     a NULL Obj*, which the old `!IS_NULL(entry->key)` test let
         *     through and then returned NULL_VAL instead of raising NameError;
         *   - the chunk is running against a module other than the one it was
         *     cached against (pushFrame falls back to the caller's module for
         *     a function that has none of its own).
         * And, the point of it: assigning to a global that already exists
         * changes JaiEntry::value and nothing else, so it can no longer miss. */
        Value nameVal = constants[nameIdx];
        if (ic != NULL && ic->state == IC_MONO && module != NULL) {
            int index = (int)ic->payload[0];
            if (index >= 0 && index < module->globals.capacity) {
                JaiEntry *entry = &module->globals.entries[index];
                if (IS_OBJ(entry->key) && AS_OBJ(entry->key) == AS_OBJ(nameVal)) {
                    vm.icHits++;
                    PUSH(entry->value);
                    VM_NEXT();
                }
            }
        }

        ObjString *name = AS_STRING(nameVal);
        if (module != NULL) {
            int index = jaiTableFindIndex(&module->globals, OBJ_VAL(name));
            if (index >= 0) {
                vm.icMisses++;
                if (ic != NULL) {
                    ic->state = IC_MONO;
                    ic->count = 1;
                    ic->shapeId[0] = 0;   /* unused: the key check validates */
                    ic->payload[0] = (uint32_t)index;
                    ic->cached[0] = NULL_VAL;
                }
                PUSH(module->globals.entries[index].value);
                VM_NEXT();
            }
        }
        Value value;
        if (vm.builtins != NULL && jaiModuleGet(vm.builtins, name, &value)) {
            PUSH(value);
            VM_NEXT();
        }
        THROW(vm.cNameError, "undefined name '%s'", name->chars);
    }

    VM_CASE(OP_SET_GLOBAL): {
        uint32_t nameIdx = READ_U24();
        (void)READ_U16();                 /* the cache is read-side only */
        ObjString *name = AS_STRING(constants[nameIdx]);
        ObjModule *module = frame->module;
        Value existing;
        if (module == NULL || !jaiModuleGet(module, name, &existing)) {
            THROW(vm.cNameError, "undefined name '%s'", name->chars);
        }
        SAVE_STATE();
        jaiModuleSet(module, name, PEEK(0));
        VM_NEXT();
    }

    VM_CASE(OP_DEF_GLOBAL): {
        ObjString *name = AS_STRING(READ_CONST());
        SAVE_STATE();
        if (frame->module == NULL) {
            THROW(vm.cRuntimeError, "no module in scope to define '%s' in",
                  name->chars);
        }
        jaiModuleSet(frame->module, name, PEEK(0));
        DROP(1);
        VM_NEXT();
    }

    VM_CASE(OP_GET_MODULE): {
        Value moduleRef = READ_CONST();
        ObjString *member = AS_STRING(READ_CONST());
        SAVE_STATE();

        ObjModule *module = NULL;
        if (IS_MODULE(moduleRef)) {
            module = AS_MODULE(moduleRef);
        } else if (IS_STRING(moduleRef)) {
            Value found;
            if (frame->module != NULL &&
                jaiModuleGet(frame->module, AS_STRING(moduleRef), &found) &&
                IS_MODULE(found)) {
                module = AS_MODULE(found);
            } else if (jaiTableGetInterned(&vm.modules, AS_STRING(moduleRef),
                                           &found) && IS_MODULE(found)) {
                module = AS_MODULE(found);
            }
        }
        if (module == NULL) {
            THROW(vm.cNameError, "unknown module in member access '%s'",
                  member->chars);
        }
        Value value;
        if (!getPropertyInto(OBJ_VAL(module), member, &value, true, NULL)) {
            goto vmThrow;
        }
        LOAD_STATE();
        PUSH(value);
        VM_NEXT();
    }

    /* --- arithmetic and logic (spec §3.3) --- */

    VM_CASE(OP_ADD): {
        if (IS_INT(stackTop[-1]) && IS_INT(stackTop[-2])) {
            int64_t r;
            if (JAI_UNLIKELY(__builtin_add_overflow(AS_INT(stackTop[-2]),
                                                    AS_INT(stackTop[-1]), &r))) {
                THROW(vm.cOverflowError,
                      "integer overflow in '+'; use '+%%' to wrap");
            }
            DROP(1);
            stackTop[-1] = INT_VAL(r);
            VM_NEXT();
        }
        if (IS_FLOAT(stackTop[-1]) && IS_FLOAT(stackTop[-2])) {
            double r = AS_FLOAT(stackTop[-2]) + AS_FLOAT(stackTop[-1]);
            DROP(1);
            stackTop[-1] = FLOAT_VAL(r);
            VM_NEXT();
        }
        BINARY(arithmetic, OP_ADD);
    }

    VM_CASE(OP_SUB): {
        if (IS_INT(stackTop[-1]) && IS_INT(stackTop[-2])) {
            int64_t r;
            if (JAI_UNLIKELY(__builtin_sub_overflow(AS_INT(stackTop[-2]),
                                                    AS_INT(stackTop[-1]), &r))) {
                THROW(vm.cOverflowError,
                      "integer overflow in '-'; use '-%%' to wrap");
            }
            DROP(1);
            stackTop[-1] = INT_VAL(r);
            VM_NEXT();
        }
        if (IS_FLOAT(stackTop[-1]) && IS_FLOAT(stackTop[-2])) {
            double r = AS_FLOAT(stackTop[-2]) - AS_FLOAT(stackTop[-1]);
            DROP(1);
            stackTop[-1] = FLOAT_VAL(r);
            VM_NEXT();
        }
        BINARY(arithmetic, OP_SUB);
    }

    VM_CASE(OP_MUL): {
        if (IS_INT(stackTop[-1]) && IS_INT(stackTop[-2])) {
            int64_t r;
            if (JAI_UNLIKELY(__builtin_mul_overflow(AS_INT(stackTop[-2]),
                                                    AS_INT(stackTop[-1]), &r))) {
                THROW(vm.cOverflowError,
                      "integer overflow in '*'; use '*%%' to wrap");
            }
            DROP(1);
            stackTop[-1] = INT_VAL(r);
            VM_NEXT();
        }
        if (IS_FLOAT(stackTop[-1]) && IS_FLOAT(stackTop[-2])) {
            double r = AS_FLOAT(stackTop[-2]) * AS_FLOAT(stackTop[-1]);
            DROP(1);
            stackTop[-1] = FLOAT_VAL(r);
            VM_NEXT();
        }
        BINARY(arithmetic, OP_MUL);
    }

    VM_CASE(OP_DIV):       BINARY(arithmetic, OP_DIV);
    VM_CASE(OP_FLOORDIV):  BINARY(arithmetic, OP_FLOORDIV);

    VM_CASE(OP_MOD): {
        /* Floor remainder, inlined for int %% int: it is 8.3% of loop_sum and
         * reaching it through `arithmetic` cost that benchmark 13% of its run.
         * y == 0 and INT64_MIN %% -1 (C UB) go the slow way, which reports the
         * division-by-zero and computes the 0. */
        if (JAI_LIKELY(IS_INT(stackTop[-1]) && IS_INT(stackTop[-2]))) {
            int64_t y = AS_INT(stackTop[-1]), x = AS_INT(stackTop[-2]);
            if (JAI_LIKELY(y != 0 && !(x == INT64_MIN && y == -1))) {
                int64_t r = x % y;
                if (r != 0 && ((r < 0) != (y < 0))) r += y;
                DROP(1);
                stackTop[-1] = INT_VAL(r);
                VM_NEXT();
            }
        }
        BINARY(arithmetic, OP_MOD);
    }

    VM_CASE(OP_POW):       BINARY(arithmetic, OP_POW);
    VM_CASE(OP_ADD_WRAP):  BINARY(arithmetic, OP_ADD_WRAP);
    VM_CASE(OP_SUB_WRAP):  BINARY(arithmetic, OP_SUB_WRAP);
    VM_CASE(OP_MUL_WRAP):  BINARY(arithmetic, OP_MUL_WRAP);
    VM_CASE(OP_CONCAT):    BINARY(arithmetic, OP_CONCAT);
    VM_CASE(OP_BAND):      BINARY(bitwise, OP_BAND);
    VM_CASE(OP_BOR):       BINARY(bitwise, OP_BOR);
    VM_CASE(OP_BXOR):      BINARY(bitwise, OP_BXOR);
    VM_CASE(OP_SHL):       BINARY(bitwise, OP_SHL);
    VM_CASE(OP_SHR):       BINARY(bitwise, OP_SHR);
    VM_CASE(OP_LT): CMP_FAST(<);  BINARY(compareOp, OP_LT);
    VM_CASE(OP_LE): CMP_FAST(<=); BINARY(compareOp, OP_LE);
    VM_CASE(OP_GT): CMP_FAST(>);  BINARY(compareOp, OP_GT);
    VM_CASE(OP_GE): CMP_FAST(>=); BINARY(compareOp, OP_GE);

    VM_CASE(OP_NEG): {
        if (IS_INT(stackTop[-1]) && AS_INT(stackTop[-1]) != INT64_MIN) {
            stackTop[-1] = INT_VAL(-AS_INT(stackTop[-1]));
            VM_NEXT();
        }
        if (IS_FLOAT(stackTop[-1])) {
            stackTop[-1] = FLOAT_VAL(-AS_FLOAT(stackTop[-1]));
            VM_NEXT();
        }
        SAVE_STATE();
        Value result;
        if (!unaryNegate(stackTop[-1], &result)) goto vmThrow;
        LOAD_STATE();
        stackTop[-1] = result;
        VM_NEXT();
    }

    VM_CASE(OP_POS):
        if (!IS_NUMBER(PEEK(0))) {
            THROW(vm.cTypeError, "unary '+' is not supported for '%s'",
                  jaiTypeNameStatic(PEEK(0)));
        }
        VM_NEXT();

    VM_CASE(OP_BNOT):
        if (!IS_INT(PEEK(0))) {
            THROW(vm.cTypeError, "'~' requires an int, not '%s'",
                  jaiTypeNameStatic(PEEK(0)));
        }
        stackTop[-1] = INT_VAL(~AS_INT(stackTop[-1]));
        VM_NEXT();

    VM_CASE(OP_EQ):
    VM_CASE(OP_NE): {
        bool wantEqual = (instStart[0] == OP_EQ);
        bool fast;
        if (JAI_LIKELY(valuesEqualFast(stackTop[-2], stackTop[-1], &fast))) {
            DROP(2);
            PUSH(BOOL_VAL(fast == wantEqual));
            VM_NEXT();
        }
        SAVE_STATE();
        bool equal = jaiValuesEqual(stackTop[-2], stackTop[-1]);
        if (vm.hasException) goto vmThrow;
        LOAD_STATE();
        DROP(2);
        PUSH(BOOL_VAL(equal == wantEqual));
        VM_NEXT();
    }

    VM_CASE(OP_IS): {
        bool same = valueIsTest(stackTop[-2], stackTop[-1]);
        DROP(2);
        PUSH(BOOL_VAL(same));
        VM_NEXT();
    }

    VM_CASE(OP_IS_NOT): {
        bool same = valueIsTest(stackTop[-2], stackTop[-1]);
        DROP(2);
        PUSH(BOOL_VAL(!same));
        VM_NEXT();
    }

    VM_CASE(OP_IN):
    VM_CASE(OP_NOT_IN): {
        bool wantIn = (instStart[0] == OP_IN);
        SAVE_STATE();
        bool contains = false;
        if (!containsOp(stackTop[-1], stackTop[-2], &contains)) goto vmThrow;
        LOAD_STATE();
        DROP(2);
        PUSH(BOOL_VAL(contains == wantIn));
        VM_NEXT();
    }

    VM_CASE(OP_NOT):
        REQUIRE_BOOL(PEEK(0), "operand of 'not'");
        stackTop[-1] = BOOL_VAL(!AS_BOOL(stackTop[-1]));
        VM_NEXT();

    /* The checker emits this only where it proved the value is an int, so the
     * float case is not a coercion but the harmless identity that keeps a
     * hand-written or optimised chunk from throwing. Above 2^53 the double
     * cannot hold the integer exactly and the low bits are lost, which is what
     * every language with this rule does and what spec §2.5 records. */
    VM_CASE(OP_TO_FLOAT):
        if (IS_INT(PEEK(0))) {
            stackTop[-1] = FLOAT_VAL((double)AS_INT(stackTop[-1]));
        } else if (!IS_FLOAT(PEEK(0))) {
            THROW(vm.cTypeError, "expected 'int' but got '%s'",
                  jaiTypeNameStatic(PEEK(0)));
        }
        VM_NEXT();

    /* --- peephole-fused forms ---
     *
     * Each is exactly the sequence it replaced (spec §3.3), so the int case is
     * a fast path and never a restriction: the peephole reads bytecode and has
     * no types, so it fuses reads of an `any` local just as readily as of an
     * `int` one. Anything but two ints goes to the same helper the unfused
     * opcode would have called — a float compares, a str concatenates, a class
     * gets its dunder, and the diagnostic on a real mismatch is the one the
     * user would have seen at -O0. */

    VM_CASE(OP_ADD_BIND): {
        /* `ADD; BIND a` fused: the int path stores straight into the slot,
         * skipping the push-then-pop the pair performed. */
        uint16_t slot = READ_U16();
        if (JAI_LIKELY(IS_INT(stackTop[-1]) && IS_INT(stackTop[-2]))) {
            int64_t r;
            if (JAI_LIKELY(!__builtin_add_overflow(AS_INT(stackTop[-2]),
                                                   AS_INT(stackTop[-1]), &r))) {
                DROP(2);
                slots[slot] = INT_VAL(r);
                VM_NEXT();
            }
        }
        SAVE_STATE();
        Value sum;
        if (!arithmetic(OP_ADD, stackTop[-2], stackTop[-1], &sum)) goto vmThrow;
        LOAD_STATE();
        DROP(2);
        slots[slot] = sum;
        VM_NEXT();
    }

    VM_CASE(OP_SUB_BIND): {
        uint16_t slot = READ_U16();
        if (JAI_LIKELY(IS_INT(stackTop[-1]) && IS_INT(stackTop[-2]))) {
            int64_t r;
            if (JAI_LIKELY(!__builtin_sub_overflow(AS_INT(stackTop[-2]),
                                                   AS_INT(stackTop[-1]), &r))) {
                DROP(2);
                slots[slot] = INT_VAL(r);
                VM_NEXT();
            }
        }
        SAVE_STATE();
        Value out;
        if (!arithmetic(OP_SUB, stackTop[-2], stackTop[-1], &out)) goto vmThrow;
        LOAD_STATE();
        DROP(2);
        slots[slot] = out;
        VM_NEXT();
    }

    /* Stamp `list[T]` / `dict[K, V]` onto the container the literal just built,
     * so the mutation guards below have something to check. Peeks rather than
     * pops: the container is still the expression's value. */
    VM_CASE(OP_ELEM_KIND): {
        uint8_t packed = READ_BYTE();
        Value target = PEEK(0);
        if (IS_LIST(target)) {
            AS_LIST(target)->elemKind = (uint8_t)(packed & 0xFu);
        } else if (IS_DICT(target)) {
            AS_DICT(target)->keyKind = (uint8_t)((packed >> 4) & 0xFu);
            AS_DICT(target)->valKind = (uint8_t)(packed & 0xFu);
        }
        /* Anything else: the annotation named a shape this does not model, and
         * an unstamped container is simply unguarded. */
        VM_NEXT();
    }

    VM_CASE(OP_MUL_BIND): {
        uint16_t slot = READ_U16();
        if (JAI_LIKELY(IS_INT(stackTop[-1]) && IS_INT(stackTop[-2]))) {
            int64_t r;
            if (JAI_LIKELY(!__builtin_mul_overflow(AS_INT(stackTop[-2]),
                                                   AS_INT(stackTop[-1]), &r))) {
                DROP(2);
                slots[slot] = INT_VAL(r);
                VM_NEXT();
            }
        }
        SAVE_STATE();
        Value out;
        if (!arithmetic(OP_MUL, stackTop[-2], stackTop[-1], &out)) goto vmThrow;
        LOAD_STATE();
        DROP(2);
        slots[slot] = out;
        VM_NEXT();
    }

    VM_CASE(OP_MOD_INT_CONST): {
        /* `<int k>; MOD` fused. The same floor-remainder rule as OP_MOD with an
         * immediate divisor; k is known non-zero at fusion time, so only
         * INT64_MIN % -1 has to reach the slow path. */
        int16_t imm = READ_I16();
        if (JAI_LIKELY(IS_INT(stackTop[-1]))) {
            int64_t y = (int64_t)imm, x = AS_INT(stackTop[-1]);
            if (JAI_LIKELY(!(x == INT64_MIN && y == -1))) {
                int64_t r = x % y;
                if (r != 0 && ((r < 0) != (y < 0))) r += y;
                stackTop[-1] = INT_VAL(r);
                VM_NEXT();
            }
        }
        SAVE_STATE();
        Value result;
        if (!arithmetic(OP_MOD, stackTop[-1], INT_VAL(imm), &result)) goto vmThrow;
        LOAD_STATE();
        stackTop[-1] = result;
        VM_NEXT();
    }

    VM_CASE(OP_ADD_INT_CONST): {
        uint16_t slot = READ_U16();
        int16_t imm = READ_I16();
        Value local = slots[slot];
        if (JAI_LIKELY(IS_INT(local))) {
            int64_t r;
            if (JAI_UNLIKELY(__builtin_add_overflow(AS_INT(local), (int64_t)imm, &r))) {
                THROW(vm.cOverflowError, "integer overflow in '+'; use '+%%' to wrap");
            }
            PUSH(INT_VAL(r));
            VM_NEXT();
        }
        SAVE_STATE();
        Value result;
        if (!arithmetic(OP_ADD, local, INT_VAL(imm), &result)) goto vmThrow;
        LOAD_STATE();
        PUSH(result);
        VM_NEXT();
    }

    VM_CASE(OP_INC_LOCAL): {
        uint16_t slot = READ_U16();
        int8_t imm = READ_I8();
        Value local = slots[slot];
        if (JAI_LIKELY(IS_INT(local))) {
            int64_t r;
            if (JAI_UNLIKELY(__builtin_add_overflow(AS_INT(local), (int64_t)imm, &r))) {
                THROW(vm.cOverflowError, "integer overflow in '+'; use '+%%' to wrap");
            }
            slots[slot] = INT_VAL(r);
            VM_NEXT_HINT(OP_LOOP);
        }
        SAVE_STATE();
        Value result;
        if (!arithmetic(OP_ADD, local, INT_VAL(imm), &result)) goto vmThrow;
        LOAD_STATE();
        /* `slots` is reloaded above: the generic path can call into Jaithon and
         * grow the value stack, which moves every frame's window. */
        slots[slot] = result;
        VM_NEXT();
    }

    VM_CASE(OP_CMP_LOCAL_CONST_LT): {
        uint16_t slot = READ_U16();
        int16_t imm = READ_I16();
        Value local = slots[slot];
        if (JAI_LIKELY(IS_INT(local))) {
            PUSH(BOOL_VAL(AS_INT(local) < (int64_t)imm));
            VM_NEXT();
        }
        SAVE_STATE();
        Value result;
        if (!compareOp(OP_LT, local, INT_VAL(imm), &result)) goto vmThrow;
        LOAD_STATE();
        PUSH(result);
        VM_NEXT();
    }

    VM_CASE(OP_GET_LOCAL2): {
        uint16_t a = READ_U16(), b = READ_U16();
        PUSH(slots[a]);
        PUSH(slots[b]);
        VM_NEXT();
    }

    VM_CASE(OP_ADD_LOCALS): {
        uint16_t a = READ_U16(), b = READ_U16();
        Value x = slots[a], y = slots[b];
        if (IS_INT(x) && IS_INT(y)) {
            int64_t r;
            if (JAI_UNLIKELY(__builtin_add_overflow(AS_INT(x), AS_INT(y), &r))) {
                THROW(vm.cOverflowError,
                      "integer overflow in '+'; use '+%%' to wrap");
            }
            PUSH(INT_VAL(r));
            VM_NEXT();
        }
        SAVE_STATE();
        Value result;
        if (!arithmetic(OP_ADD, x, y, &result)) goto vmThrow;
        LOAD_STATE();
        PUSH(result);
        VM_NEXT();
    }

    /* --- control flow (spec §3.4) --- */

    VM_CASE(OP_JUMP): {
        int16_t offset = READ_I16();
        ip += offset;
        VM_NEXT();
    }

    VM_CASE(OP_JUMP_IF_FALSE): {
        int16_t offset = READ_I16();
        Value condition = POP();
        REQUIRE_BOOL(condition, "condition");
        if (!AS_BOOL(condition)) ip += offset;
        VM_NEXT_HINT(OP_INC_LOCAL);
    }

    VM_CASE(OP_JUMP_IF_TRUE): {
        int16_t offset = READ_I16();
        Value condition = POP();
        REQUIRE_BOOL(condition, "condition");
        if (AS_BOOL(condition)) ip += offset;
        VM_NEXT();
    }

    VM_CASE(OP_JUMP_IF_FALSE_KEEP): {
        int16_t offset = READ_I16();
        REQUIRE_BOOL(PEEK(0), "operand of 'and'");
        if (!AS_BOOL(PEEK(0))) ip += offset;
        VM_NEXT();
    }

    VM_CASE(OP_JUMP_IF_TRUE_KEEP): {
        int16_t offset = READ_I16();
        REQUIRE_BOOL(PEEK(0), "operand of 'or'");
        if (AS_BOOL(PEEK(0))) ip += offset;
        VM_NEXT();
    }

    VM_CASE(OP_JUMP_IF_NULL): {
        int16_t offset = READ_I16();
        if (IS_NULL(PEEK(0))) ip += offset;
        VM_NEXT();
    }

    /* <cmp>; JUMP_IF_FALSE fused (spec §3.3). The bool never reaches the stack,
     * and with it goes JUMP_IF_FALSE's REQUIRE_BOOL: every producer this pass
     * accepts is bool by construction, which is the same fact `pushesBool` in
     * optimize.c already relies on to fold OP_NOT into a jump. Only int/int is
     * inlined; float and everything else take the identical slow path the
     * unfused pair would have, so NaN and __lt__ behave exactly as before. */
    VM_CASE(OP_JUMP_IF_CMP_FALSE): {
        uint8_t cmp = READ_BYTE();
        int16_t offset = READ_I16();
        Value a = stackTop[-2], b = stackTop[-1];

        if (JAI_LIKELY(IS_INT(a) && IS_INT(b))) {
            int64_t x = AS_INT(a), y = AS_INT(b);
            bool taken;
            switch (cmp) {
            case OP_EQ: taken = x == y; break;
            case OP_NE: taken = x != y; break;
            case OP_LT: taken = x <  y; break;
            case OP_LE: taken = x <= y; break;
            case OP_GT: taken = x >  y; break;
            case OP_GE: taken = x >= y; break;
            default:
                THROW(vm.cRuntimeError, "JUMP_IF_CMP_FALSE has opcode %u, "
                      "which is not a comparison", (unsigned)cmp);
            }
            stackTop -= 2;
            if (!taken) ip += offset;
            VM_NEXT();
        }

        if (cmp == OP_EQ || cmp == OP_NE) {
            bool fast;
            if (JAI_LIKELY(valuesEqualFast(a, b, &fast))) {
                stackTop -= 2;
                if (fast != (cmp == OP_EQ)) ip += offset;
                VM_NEXT();
            }
        }

        SAVE_STATE();
        bool condition;
        if (cmp == OP_EQ || cmp == OP_NE) {
            bool equal = jaiValuesEqual(a, b);
            if (vm.hasException) goto vmThrow;
            condition = equal == (cmp == OP_EQ);
        } else {
            Value result;
            if (!compareOp((OpCode)cmp, a, b, &result)) goto vmThrow;
            condition = AS_BOOL(result);
        }
        LOAD_STATE();
        stackTop -= 2;
        if (!condition) ip += offset;
        VM_NEXT();
    }

    /* GET_LOCAL S; CONST K; JUMP_IF_CMP_FALSE cmp fused (spec §3.3): a loop
     * guard or a recursion base case in one instruction. Neither operand is
     * ever pushed, so unlike JUMP_IF_CMP_FALSE there is no stack to unwind on
     * either path — but both are still rooted while the slow path runs, the
     * local by the frame it lives in and the constant by the chunk. */
    VM_CASE(OP_JUMP_IF_CMP_LOCAL_K): {
        uint8_t cmp = READ_BYTE();
        Value a = slots[READ_U16()];
        Value b = constants[READ_U24()];
        int16_t offset = READ_I16();

        if (JAI_LIKELY(IS_INT(a) && IS_INT(b))) {
            int64_t x = AS_INT(a), y = AS_INT(b);
            bool taken;
            switch (cmp) {
            case OP_EQ: taken = x == y; break;
            case OP_NE: taken = x != y; break;
            case OP_LT: taken = x <  y; break;
            case OP_LE: taken = x <= y; break;
            case OP_GT: taken = x >  y; break;
            case OP_GE: taken = x >= y; break;
            default:
                THROW(vm.cRuntimeError, "JUMP_IF_CMP_LOCAL_K has opcode %u, "
                      "which is not a comparison", (unsigned)cmp);
            }
            if (!taken) ip += offset;
            VM_NEXT();
        }

        if (cmp == OP_EQ || cmp == OP_NE) {
            bool fast;
            if (JAI_LIKELY(valuesEqualFast(a, b, &fast))) {
                if (fast != (cmp == OP_EQ)) ip += offset;
                VM_NEXT();
            }
        }

        SAVE_STATE();
        bool condition;
        if (cmp == OP_EQ || cmp == OP_NE) {
            bool equal = jaiValuesEqual(a, b);
            if (vm.hasException) goto vmThrow;
            condition = equal == (cmp == OP_EQ);
        } else {
            Value result;
            if (!compareOp((OpCode)cmp, a, b, &result)) goto vmThrow;
            condition = AS_BOOL(result);
        }
        LOAD_STATE();
        if (!condition) ip += offset;
        VM_NEXT();
    }

    /* FOR_ITER J; BIND S fused (spec §3.3): `for x in xs` in one instruction.
     * The produced item goes straight to its slot instead of being pushed and
     * popped back off. */
    VM_CASE(OP_FOR_ITER_BIND): {
        int16_t offset = READ_I16();
        uint16_t slot = READ_U16();
        Value iterator = PEEK(0);
        if (!IS_ITER(iterator)) {
            THROW(vm.cTypeError, "for-loop expected an iterator, not '%s'",
                  jaiTypeNameStatic(iterator));
        }
        Value item;
        IterStep step = iterStepFast(AS_ITER(iterator), &item);
        if (JAI_LIKELY(step == ITER_STEP_VALUE)) {
            slots[slot] = item;
            VM_NEXT();
        }
        if (step == ITER_STEP_DONE) {
            DROP(1);
            ip += offset;
            VM_NEXT();
        }
        SAVE_STATE();
        bool advanced = jaiIterNext(AS_ITER(iterator), &item);
        if (!advanced && vm.hasException) goto vmThrow;
        LOAD_STATE();
        if (!advanced) {
            DROP(1);            /* exhausted: the iterator goes with the loop */
            ip += offset;
            VM_NEXT();
        }
        /* `slots` is reloaded above: jaiIterNext can re-enter the VM through a
         * user __next__ and grow the value stack, moving every frame window. */
        slots[slot] = item;
        VM_NEXT();
    }

    /* FOR_ITER J; UNPACK 2 255; BIND A; BIND B fused (spec §3.3):
     * `for (a, b) in …` in one instruction and, on a dict, with no pair object
     * built at all. See OP_FOR_ITER_PAIR in chunk.h for what that is worth. */
    VM_CASE(OP_FOR_ITER_PAIR): {
        int16_t  offset = READ_I16();
        uint16_t slotA  = READ_U16();
        uint16_t slotB  = READ_U16();
        Value iterator = PEEK(0);
        if (!IS_ITER(iterator)) {
            THROW(vm.cTypeError, "for-loop expected an iterator, not '%s'",
                  jaiTypeNameStatic(iterator));
        }
        Value a, b;
        PairStep step = iterStepPairFast(AS_ITER(iterator), &a, &b);
        if (JAI_LIKELY(step == PAIR_STEP_VALUE)) {
            slots[slotA] = a;
            slots[slotB] = b;
            VM_NEXT();
        }
        if (step == PAIR_STEP_DONE) {
            DROP(1);
            ip += offset;
            VM_NEXT();
        }
        if (step == PAIR_STEP_BAD) {
            SAVE_STATE();
            (void)pairSplitFail(a);
            goto vmThrow;
        }
        SAVE_STATE();
        Value item;
        bool advanced = jaiIterNext(AS_ITER(iterator), &item);
        if (!advanced && vm.hasException) goto vmThrow;
        LOAD_STATE();
        if (!advanced) {
            DROP(1);            /* exhausted: the iterator goes with the loop */
            ip += offset;
            VM_NEXT();
        }
        if (!pairSplit(item, &a, &b)) {
            SAVE_STATE();
            (void)pairSplitFail(item);
            goto vmThrow;
        }
        /* `slots` is reloaded above: jaiIterNext can re-enter the VM through a
         * user __next__ and grow the value stack, moving every frame window. */
        slots[slotA] = a;
        slots[slotB] = b;
        VM_NEXT();
    }

    /* `for x in a..b`, opened (spec §3.4). Where BUILD_RANGE; GET_ITER built an
     * ObjRange and an ObjIter per loop ENTRY, this writes the loop's whole
     * state into two int frame slots and allocates nothing at all.
     *
     * `end` is one past the last value, computed WRAPPING, so the counter test
     * downstream is `!=` rather than `<` and `a..=INT64_MAX` still terminates:
     * the counter wraps to INT64_MIN and meets it there. See OP_ITER_RANGE in
     * chunk.h for the single range where that disagrees with ObjIter's
     * saturating limit, and why it does not matter. */
    VM_CASE(OP_ITER_RANGE): {
        bool     inclusive = READ_BYTE() != 0;
        uint16_t curSlot   = READ_U16();
        uint16_t endSlot   = READ_U16();
        Value stopValue = PEEK(0), startValue = PEEK(1);
        /* The same check, and the same wording, OP_BUILD_RANGE makes: this is a
         * fast path for one loop shape, never a second set of rules. */
        if (!IS_INT(startValue) || !IS_INT(stopValue)) {
            THROW(vm.cTypeError, "range bounds must be int, not '%s' and '%s'",
                  jaiTypeNameStatic(startValue), jaiTypeNameStatic(stopValue));
        }
        int64_t start = AS_INT(startValue), stop = AS_INT(stopValue);
        int64_t end = stop < start
                          ? start          /* empty: end where it begins */
                          : (inclusive
                                 ? (int64_t)((uint64_t)stop + 1u)
                                 : stop);
        DROP(2);
        slots[curSlot] = INT_VAL(start);
        slots[endSlot] = INT_VAL(end);
        VM_NEXT();
    }

    /* One step of that loop (spec §3.4). Both slots were written by the
     * OP_ITER_RANGE above and by this instruction and by nothing else — the
     * emitter hands out fresh temporaries for them — so the payloads are read
     * without re-testing a tag the loop itself is the only writer of. */
    VM_CASE(OP_FOR_RANGE_BIND): {
        int16_t  offset  = READ_I16();
        uint16_t slot    = READ_U16();
        uint16_t curSlot = READ_U16();
        uint16_t endSlot = READ_U16();
        int64_t cur = AS_INT(slots[curSlot]);
        if (cur == AS_INT(slots[endSlot])) {
            ip += offset;
            VM_NEXT();
        }
        slots[slot]    = INT_VAL(cur);
        slots[curSlot] = INT_VAL((int64_t)((uint64_t)cur + 1u));
        VM_NEXT();
    }

    /* An f-string, whole (spec §3.6). The parts are on the stack in order and
     * the result replaces them.
     *
     * The lowering this replaces was, per hole, a global lookup of `str`, a
     * native call and an OP_CONCAT: two heap strings, two hashes and two
     * intern probes to interpolate one integer. jaiValueFormat measures the
     * parts and allocates once.
     *
     * The cache answers one question — is `str` still the builtin? — and is
     * keyed on the globals table's KEY version, which moves when a name is
     * added or removed and not when one is merely assigned to. That is exactly
     * the fact being memoised, so a module-scope loop that writes globals no
     * longer throws this cache away on every iteration. */
    VM_CASE(OP_FORMAT): {
        uint8_t  count    = READ_BYTE();
        uint32_t litmask  = READ_U24();
        uint32_t nameIdx  = READ_U24();
        uint16_t cacheIdx = READ_U16();

        ObjModule *module = frame->module;
        InlineCache *ic = cacheAt(frameChunk(frame), cacheIdx);
        bool builtin;
        if (JAI_LIKELY(ic != NULL && ic->state == IC_MONO && module != NULL &&
                       ic->shapeId[0] == module->globals.keyVersion &&
                       ic->payload[1] ==
                           (uint32_t)((uintptr_t)module >> 4))) {
            builtin = ic->payload[0] != 0;
        } else {
            ObjString *name = AS_STRING(constants[nameIdx]);
            Value shadow;
            builtin = module == NULL ||
                      !jaiTableGetInterned(&module->globals, name, &shadow);
            if (ic != NULL && module != NULL) {
                ic->state = IC_MONO;
                ic->count = 1;
                ic->shapeId[0] = module->globals.keyVersion;
                /* A chunk whose function has no module of its own runs against
                 * the caller's (pushFrame), so one cache slot can be reached
                 * with two ObjModule*. keyVersion is small and dense, so pin
                 * the module too -- no deref, so no rooting question. */
                ic->payload[1] = (uint32_t)((uintptr_t)module >> 4);
                ic->payload[0] = builtin ? 1u : 0u;
                ic->cached[0] = NULL_VAL;
            }
        }

        SAVE_STATE();
        if (JAI_UNLIKELY(!builtin) &&
            !formatViaUserStr(module, AS_STRING(constants[nameIdx]), count,
                              litmask)) {
            goto vmThrow;
        }
        ObjString *formatted = jaiValueFormat(vm.stackTop - count, count);
        if (formatted == NULL) goto vmThrow;
        LOAD_STATE();
        stackTop -= count;
        PUSH(OBJ_VAL(formatted));
        VM_NEXT_HINT(OP_BIND);
    }

    VM_CASE(OP_LOOP): {
        int16_t offset = READ_I16();
        ip += offset;
        /* Both halves of the safepoint are almost always "nothing to do", so
         * only the taken case pays for writing the frame state back: the
         * common back edge is one predictable branch, not a
         * SAVE_STATE/call/LOAD_STATE round trip. */
        {
            const ObjFunction *lf = frame->closure->fn;
            if (JAI_UNLIKELY(jaiInterrupted || jaiGCWanted() ||
                             lf->osrHot)) {
                SAVE_STATE();
                if (!safepoint()) goto vmThrow;
                LOAD_STATE();
            }
        }
        VM_NEXT();
    }

    VM_CASE(OP_GET_ITER): {
        SAVE_STATE();
        Value iterator;
        if (!jaiGetIter(stackTop[-1], &iterator)) goto vmThrow;
        LOAD_STATE();
        stackTop[-1] = iterator;
        VM_NEXT();
    }

    /* `for … in X.items()`. On a dict, iterate the table directly instead of
     * building the whole list of pairs first: jaiDictItems allocates an
     * N-element list of N fresh 2-tuples per call and keeps all N alive at
     * once, so every collection during the loop marks the lot. The lazy form
     * keeps one tuple alive at a time. Measured on tests/bench/dict_iter:
     * peak RSS 37.1 MB -> 30.3 MB, instructions -22%.
     *
     * Anything else is LEFT ALONE and the ordinary `INVOKE items; GET_ITER`
     * that follows handles it, so a user class defining `items()` still has its
     * own method called. Invoking that method from here instead was the first
     * attempt and --gc-stress rejected it: a Jaithon call needs the frame
     * machinery an opcode body does not have. */
    VM_CASE(OP_GET_ITER_ITEMS): {
        int16_t offset = READ_I16();
        Value target = PEEK(0);
        if (IS_DICT(target)) {
            SAVE_STATE();
            ObjIter *it = jaiIterNew(ITER_DICT_ITEMS, target);
            if (it == NULL) goto vmThrow;
            LOAD_STATE();
            stackTop[-1] = OBJ_VAL(it);
            ip += offset;
        }
        VM_NEXT();
    }

    VM_CASE(OP_FOR_ITER): {
        int16_t offset = READ_I16();
        Value iterator = PEEK(0);
        if (!IS_ITER(iterator)) {
            THROW(vm.cTypeError, "for-loop expected an iterator, not '%s'",
                  jaiTypeNameStatic(iterator));
        }
        Value item;
        IterStep step = iterStepFast(AS_ITER(iterator), &item);
        if (JAI_LIKELY(step == ITER_STEP_VALUE)) {
            PUSH(item);
            VM_NEXT();
        }
        if (step == ITER_STEP_DONE) {
            DROP(1);
            ip += offset;
            VM_NEXT();
        }
        SAVE_STATE();
        bool advanced = jaiIterNext(AS_ITER(iterator), &item);
        if (!advanced && vm.hasException) goto vmThrow;
        LOAD_STATE();
        if (!advanced) {
            DROP(1);            /* exhausted: the iterator goes with the loop */
            ip += offset;
            VM_NEXT();
        }
        PUSH(item);
        VM_NEXT();
    }

    /* --- calls (spec §3.5) --- */

    VM_CASE(OP_CALL): {
        int argc = READ_BYTE();
        SAVE_STATE();
        /* A closure is what almost every call site holds, and reaching
         * callClosure through callValueOnStack costs an out-of-line call and
         * invokeCallable's IS_OBJ test and type switch on the way. A call is
         * 15ns of overhead measured against the same loop written inline, and
         * this is the part of it that buys nothing on the common path.
         *
         * Anything else -- natives, bound methods, classes, functions without a
         * closure -- takes the general path unchanged. */
        Value callee = vm.stackTop[-argc - 1];
        CallOutcome outcome = JAI_LIKELY(IS_CLOSURE(callee))
                                  ? callClosure(AS_CLOSURE(callee), argc)
                                  : callValueOnStack(argc);
        if (outcome == CALL_ERROR) goto vmThrow;
        if (outcome == CALL_DONE) { LOAD_STACK_ONLY(); VM_NEXT(); }
        LOAD_STATE();
        VM_NEXT();
    }

    VM_CASE(OP_CALL_KW): {
        int posCount = READ_BYTE();
        Value nameTuple = READ_CONST();
        SAVE_STATE();
        if (!IS_TUPLE(nameTuple)) {
            THROW(vm.cRuntimeError,
                  "CALL_KW expected a tuple of keyword names");
        }
        ObjDict *kwRest = NULL;
        int argc = prepareKeywordCall(posCount, AS_TUPLE(nameTuple), &kwRest);
        if (argc < 0) goto vmThrow;

        Value callee = vm.stackTop[-argc - 1];
        if (kwRest != NULL) jaiGCPushRoot(OBJ_VAL(kwRest));
        CallOutcome outcome = invokeCallable(callee, argc);
        if (outcome == CALL_ERROR) {
            if (kwRest != NULL) jaiGCPopRoot();
            goto vmThrow;
        }
        if (kwRest != NULL) {
            if (outcome == CALL_FRAME) {
                CallFrame *calleeFrame = &vm.frames[vm.frameCount - 1];
                int slot = kwRestSlotOf(calleeFrame->closure->fn);
                calleeFrame->slots[slot] = OBJ_VAL(kwRest);
            }
            jaiGCPopRoot();
        }
        LOAD_STATE();
        VM_NEXT();
    }

    VM_CASE(OP_CALL_SPREAD): {
        int argc = READ_BYTE();
        SAVE_STATE();
        if (argc < 1) THROW(vm.cRuntimeError, "CALL_SPREAD with no arguments");
        Value spread = stackTop[-1];
        if (!IS_LIST(spread)) {
            THROW(vm.cTypeError, "spread argument must be a list, not '%s'",
                  jaiTypeNameStatic(spread));
        }
        ObjList *list = AS_LIST(spread);
        DROP(1);
        SAVE_STATE();
        if (!ensureStack(list->count + 1)) goto vmThrow;
        for (int i = 0; i < list->count; i++) PUSH(list->items[i]);
        int total = argc - 1 + list->count;
        SAVE_STATE();
        if (callValueOnStack(total) == CALL_ERROR) goto vmThrow;
        LOAD_STATE();
        VM_NEXT();
    }

    VM_CASE(OP_INVOKE): {
        uint32_t nameIdx = READ_U24();
        int argc = READ_BYTE();
        uint16_t cacheIdx = READ_U16();
        Value receiver = stackTop[-argc - 1];
        uint32_t builtinTag = 0;
        /* Set only when this pass FILLS a builtin way below, which is the one
         * moment the site's result kind can be recorded without costing the
         * fast path anything. See InlineCache::resultKind. */
        InlineCache *fbCache = NULL;
        int fbWay = 0;

        /* Fast path: the receiver's class is one the cache has already seen,
         * so the method is known without touching a hash table. */
        if (IS_INSTANCE(receiver)) {
            InlineCache *ic = cacheAt(frameChunk(frame), cacheIdx);
            ObjClass *klass = AS_INSTANCE(receiver)->klass;
            if (ic != NULL && klass != NULL && ic->state != IC_EMPTY &&
                ic->state != IC_MEGA) {
                for (int w = 0; w < ic->count; w++) {
                    if (ic->shapeId[w] != klass->shapeId) continue;
                    vm.icHits++;
                    SAVE_STATE();
                    /* The payload marks a way whose method is not public. What
                     * the cache settles is which method a shape resolves to;
                     * whether this frame may call it depends on the *caller*,
                     * which one site can present as two classes, so that half
                     * is re-decided on every hit. */
                    if (ic->payload[w] != 0 &&
                        !methodPermitted(klass, AS_STRING(constants[nameIdx]),
                                         true)) {
                        goto vmThrow;
                    }
                    /* The cache is only filled when slot 0 is the receiver
                     * (see the fill site below), so this is a method call. */
                    if (invokeMethodOnStack(ic->cached[w], argc) == CALL_ERROR) {
                        goto vmThrow;
                    }
                    LOAD_STATE();
                    VM_NEXT();
                }
            }
        } else if ((builtinTag = builtinShapeTag(receiver)) != 0) {
            /* Same idea for `xs.push(v)`. The cached value is the ObjNative
             * itself, not a bound wrapper: the receiver is already sitting in
             * the callee slot, which is exactly where a built-in method wants
             * its args[0], so the call needs no intermediate object at all. */
            InlineCache *ic = cacheAt(frameChunk(frame), cacheIdx);
            if (ic != NULL && ic->state != IC_EMPTY && ic->state != IC_MEGA) {
                for (int w = 0; w < ic->count; w++) {
                    if (ic->shapeId[w] != builtinTag) continue;
                    vm.icHits++;
                    SAVE_STATE();
                    Value *slot = vm.stackTop - argc - 1;
                    Value result;
                    if (!callNativeAt(AS_NATIVE(ic->cached[w]), slot, argc + 1,
                                      &result)) {
                        goto vmThrow;
                    }
                    vm.stackTop = slot;
                    *vm.stackTop++ = result;
                    /* A built-in method pushes no frame: `xs.push(v)` in a hot
                     * loop reaches here, and LOAD_STATE's constant-pool chase
                     * was the largest thing left in it. */
                    LOAD_STACK_ONLY();
                    VM_NEXT();
                }
            }
        }

        SAVE_STATE();
        ObjString *name = AS_STRING(constants[nameIdx]);

        /* `Shape.Circle(2.0)`: a variant with a payload is not a value, it is
         * a constructor, and this is the only place the arguments and the
         * variant name are both in hand. Zero-arity variants are values and
         * `enumMember` already produced one. */
        if (IS_ENUM(receiver)) {
            ObjEnum *enumType = AS_ENUM(receiver);
            int tag = jaiEnumVariantIndex(enumType, name);
            if (tag >= 0 && enumType->variants[tag].arity > 0) {
                if (argc != enumType->variants[tag].arity) {
                    THROW(vm.cTypeError,
                          "%s.%s() takes %d argument%s but %d were given",
                          enumType->name != NULL ? enumType->name->chars : "?",
                          name->chars, (int)enumType->variants[tag].arity,
                          enumType->variants[tag].arity == 1 ? "" : "s", argc);
                }
                ObjEnumVal *built = jaiEnumValNew(enumType, (uint16_t)tag,
                                                  vm.stackTop - argc, argc);
                LOAD_STATE();
                DROP(argc + 1);            /* the arguments and the enum */
                PUSH(OBJ_VAL(built));
                VM_NEXT();
            }
        }

        Value method, slotZero;
        bool isMethod;
        if (!resolveInvokeTarget(receiver, name, &method, &slotZero, &isMethod)) {
            /* The lookup may already have raised something more precise than
             * "no such method" — E0802 for a module member that exists but is
             * private. Overwriting it would send the reader hunting a typo. */
            if (vm.hasException) goto vmThrow;
            THROW(vm.cAttributeError, "'%s' object has no method '%s'",
                  jaiTypeNameStatic(receiver), name->chars);
        }
        vm.stackTop[-argc - 1] = slotZero;

        if (IS_INSTANCE(receiver)) {
            vm.icMisses++;
            InlineCache *ic = cacheAt(frameChunk(frame), cacheIdx);
            ObjClass *klass = AS_INSTANCE(receiver)->klass;
            /* Only a real method is cacheable: a callable field is per
             * instance, not per shape. A non-public one is cached with the
             * payload set, which tells the fast path above to re-run the
             * visibility test it cannot cache. */
            MethodInfo restricted;
            uint32_t recheck =
                jaiClassRestrictedMethod(klass, name, &restricted) ? 1u : 0u;
            if (ic != NULL && klass != NULL && AS_OBJ(slotZero) == AS_OBJ(receiver) &&
                ic->state != IC_MEGA) {
                if (ic->count < JAI_IC_WAYS) {
                    ic->shapeId[ic->count] = klass->shapeId;
                    ic->payload[ic->count] = recheck;
                    ic->cached[ic->count] = method;
                    ic->count++;
                    ic->state = (ic->count == 1) ? IC_MONO : IC_POLY;
                } else {
                    ic->state = IC_MEGA;
                }
            }
        } else if (builtinTag != 0 && IS_BOUND(method) &&
                   IS_NATIVE(AS_BOUND(method)->method)) {
            /* jaiBuiltinMethod hands back a bound native whose receiver is the
             * one we passed; cache the native and drop the wrapper. Anything
             * shaped differently (a module member, a __format__ on a value the
             * tag does not cover) simply is not cached. */
            vm.icMisses++;
            InlineCache *ic = cacheAt(frameChunk(frame), cacheIdx);
            if (ic != NULL && ic->state != IC_MEGA) {
                if (ic->count < JAI_IC_WAYS) {
                    ic->shapeId[ic->count] = builtinTag;
                    ic->payload[ic->count] = 0;
                    ic->cached[ic->count] = AS_BOUND(method)->method;
                    ic->resultKind[ic->count] = JAI_FB_NONE;
                    ic->count++;
                    ic->state = (ic->count == 1) ? IC_MONO : IC_POLY;
                    fbCache = ic;
                    fbWay   = ic->count - 1;
                } else {
                    ic->state = IC_MEGA;
                }
            }
        }
        if ((isMethod ? invokeMethodOnStack(method, argc)
                      : invokeCallable(method, argc)) == CALL_ERROR) {
            goto vmThrow;
        }
        /* A bound native pushes no frame, so the result is on the stack now.
         * Only reached on the way that just filled the cache -- once per site
         * and receiver type, never on the path a loop repeats. */
        if (fbCache != NULL) {
            fbCache->resultKind[fbWay] =
                jaiFeedbackMerge(fbCache->resultKind[fbWay],
                                 jaiFeedbackKind(vm.stackTop[-1]));
        }
        LOAD_STATE();
        VM_NEXT();
    }

    VM_CASE(OP_SUPER_INVOKE): {
        uint32_t nameIdx = READ_U24();
        int argc = READ_BYTE();
        SAVE_STATE();
        Value receiver = stackTop[-argc - 1];
        ObjString *name = AS_STRING(constants[nameIdx]);

        ObjClass *start = NULL;
        if (IS_INSTANCE(receiver)) start = AS_INSTANCE(receiver)->klass;
        else if (IS_CLASS(receiver)) start = AS_CLASS(receiver);
        if (start == NULL || start->superclass == NULL) {
            THROW(vm.cRuntimeError, "'super.%s' has no superclass to dispatch to",
                  name->chars);
        }
        Value method;
        if (!findMethod(start->superclass, name, &method)) {
            THROW(vm.cAttributeError, "superclass '%s' has no method '%s'",
                  start->superclass->name != NULL ? start->superclass->name->chars
                                                  : "?",
                  name->chars);
        }
        /* Slot 0 is the receiver here whatever the parent's method turns out
         * to be, so `super.f()` reaches a native parent method with `self`. */
        if (invokeMethodOnStack(method, argc) == CALL_ERROR) goto vmThrow;
        LOAD_STATE();
        VM_NEXT();
    }

    VM_CASE(OP_TAIL_CALL): {
        int argc = READ_BYTE();
        SAVE_STATE();
        Value callee = stackTop[-argc - 1];
        ObjFunction *fn = frame->closure->fn;

        /* Reuse the window only for the straightforward shape; anything with
         * defaults, a variadic tail, or a non-closure callee goes through the
         * ordinary path, which is correct if less frugal. */
        if (IS_CLOSURE(callee) && AS_CLOSURE(callee)->fn->defaultCount == 0 &&
            !(AS_CLOSURE(callee)->fn->flags & (FN_VARIADIC | FN_KWREST)) &&
            AS_CLOSURE(callee)->fn->arity == argc &&
            !(fn->flags & FN_INIT)) {
            ObjClosure *target = AS_CLOSURE(callee);

            /* A tail call reuses this frame instead of going through
             * callClosure, so without this the compiled tier never saw the
             * callee at all -- not even to count it as hot. `sort` tail-calls
             * `merge`, which is the whole of that benchmark's inner loop, and
             * it was invisible. The compiled form finishes the call outright,
             * and finishing a tail call is returning from this frame. */
            ObjFunction *tfn = target->fn;
            Value *tailBase = vm.stackTop - argc - 1;
            if (tfn->jitFunc != NULL) {
                JaiJitOutcome outcome = jaiJitEnterFunc(target, tailBase);
                if (outcome == JAI_JIT_ERROR) goto vmThrow;
                if (outcome == JAI_JIT_DEOPT) {
                    /* Push a real frame rather than reusing this one: the
                     * callee is resuming part-way through its own body. The
                     * OP_RETURN the compiler puts after a tail call then
                     * returns its result, which is what a tail call means. */
                    if (!bindCallArgs(target, argc, tailBase)) goto vmThrow;
                    if (!pushFrame(target, tailBase)) goto vmThrow;
                    if (!jaiJitApplyDeopt(target, tailBase)) goto vmThrow;
                    LOAD_STATE();
                    VM_NEXT();
                }
                if (outcome == JAI_JIT_DONE) {
                    retval = tailBase[0];
                    stackTop = vm.stackTop;   /* opReturn saves this back */
                    goto opReturn;
                }
            } else if (tfn->entryCount < JAI_JIT_THRESHOLD) {
                tfn->entryCount++;
            } else if (!tfn->jitRefused && jaiJitEnabled() &&
                       jaiJitEnter(target, tailBase)) {
                retval = tailBase[0];
                stackTop = vm.stackTop;
                goto opReturn;
            }

            if (FRAME_HAS_DEFERS(frame) && !runFrameDefers(frame)) goto vmThrow;
            if (vm.hasException) goto vmThrow;
            closeUpvalues(frame->slots);

            Value *window = frame->slots;
            memmove(window, vm.stackTop - argc - 1,
                    sizeof(Value) * (size_t)(argc + 1));
            vm.stackTop = window + argc + 1;

            int newWindow = frameWindowSize(target->fn);
            for (int i = argc + 1; i < newWindow; i++) window[i] = NULL_VAL;
            vm.stackTop = window + newWindow;

            vm.handlers.count = frame->handlerBase;
            vm.defers.count = frame->deferBase;
            frame->closure = target;
            frame->ip = target->fn->chunk.code;
            if (target->fn->module != NULL) frame->module = target->fn->module;
            vm.callCount++;
            LOAD_STATE();
            VM_NEXT();
        }
        if (callValueOnStack(argc) == CALL_ERROR) goto vmThrow;
        LOAD_STATE();
        VM_NEXT();
    }

    VM_CASE(OP_RETURN):
        retval = POP();
        goto opReturn;

    VM_CASE(OP_RETURN_NULL):
        retval = NULL_VAL;
        goto opReturn;

    opReturn: {
        SAVE_STATE();
        ObjFunction *fn = frame->closure->fn;
        /* An initializer yields the object it initialised, whatever its body
         * returned; that is what makes `Point(1, 2)` an expression. A default
         * thunk borrows the same function record, so it is excluded. */
        if (((fn->flags & FN_INIT) || (fn->name != NULL && fn->name == vm.strInit)) &&
            vm.frameCount - 1 != sThunkFrame) {
            retval = frame->slots[0];
        }
        if (FRAME_HAS_DEFERS(frame)) (void)runFrameDefers(frame);
        if (vm.hasException) goto vmThrow;

        closeUpvalues(frame->base);
        vm.handlers.count = frame->handlerBase;
        vm.defers.count = frame->deferBase;
        vm.frameCount--;
        vm.stackTop = frame->base;
        *vm.stackTop++ = retval;

        if (vm.frameCount <= baseFrameCount) return JAI_RUN_OK;
        LOAD_STATE();
        VM_NEXT();
    }

    VM_CASE(OP_CLOSURE): {
        Value fnValue = READ_CONST();
        SAVE_STATE();
        if (!IS_FUNCTION(fnValue)) {
            THROW(vm.cRuntimeError, "CLOSURE operand is not a function");
        }
        ObjFunction *fn = AS_FUNCTION(fnValue);
        ObjClosure *closure = jaiClosureNew(fn);
        LOAD_STATE();
        /* Read the upvalue descriptors before pushing: the closure is only
         * reachable from this local until then. */
        jaiGCPushRoot(OBJ_VAL(closure));
        for (int i = 0; i < closure->upvalueCount; i++) {
            uint8_t how = READ_BYTE();
            uint16_t index = READ_U16();
            bool isLocal = (how & 1u) != 0;
            if ((how & 2u) != 0) {
                /* By value (spec §6's `let`): the closure gets the value the
                 * binding holds now, so a loop that builds one closure per
                 * iteration keeps one value per iteration. */
                Value snapshot = isLocal
                                     ? frame->slots[index]
                                     : *frame->closure->upvalues[index]->location;
                SAVE_STATE();
                closure->upvalues[i] = jaiUpvalueClosed(snapshot);
                LOAD_STATE();
            } else if (isLocal) {
                SAVE_STATE();
                closure->upvalues[i] = captureUpvalue(frame->slots + index);
                LOAD_STATE();
            } else {
                closure->upvalues[i] = frame->closure->upvalues[index];
            }
        }
        jaiGCPopRoot();
        PUSH(OBJ_VAL(closure));
        VM_NEXT();
    }

    /* --- data structures (spec §3.6) --- */

    VM_CASE(OP_BUILD_LIST): {
        int count = READ_U16();
        SAVE_STATE();
        ObjList *list = jaiListNew(count);
        LOAD_STATE();
        for (int i = 0; i < count; i++) list->items[i] = stackTop[-count + i];
        list->count = count;
        DROP(count);
        PUSH(OBJ_VAL(list));
        VM_NEXT();
    }

    VM_CASE(OP_BUILD_DICT): {
        int count = READ_U16();
        SAVE_STATE();
        ObjDict *dict = jaiDictNew();
        jaiGCPushRoot(OBJ_VAL(dict));
        for (int i = 0; i < count; i++) {
            Value key = vm.stackTop[-2 * count + 2 * i];
            Value value = vm.stackTop[-2 * count + 2 * i + 1];
            (void)jaiDictSet(dict, key, value);
            if (vm.hasException) { jaiGCPopRoot(); goto vmThrow; }
        }
        jaiGCPopRoot();
        LOAD_STATE();
        DROP(2 * count);
        PUSH(OBJ_VAL(dict));
        VM_NEXT();
    }

    VM_CASE(OP_BUILD_SET): {
        int count = READ_U16();
        SAVE_STATE();
        ObjSet *set = jaiSetNew();
        jaiGCPushRoot(OBJ_VAL(set));
        for (int i = 0; i < count; i++) {
            (void)jaiSetAdd(set, vm.stackTop[-count + i]);
            if (vm.hasException) { jaiGCPopRoot(); goto vmThrow; }
        }
        jaiGCPopRoot();
        LOAD_STATE();
        DROP(count);
        PUSH(OBJ_VAL(set));
        VM_NEXT();
    }

    VM_CASE(OP_BUILD_TUPLE): {
        int count = READ_U16();
        SAVE_STATE();
        ObjTuple *tuple = jaiTupleNew(vm.stackTop - count, count);
        LOAD_STATE();
        DROP(count);
        PUSH(OBJ_VAL(tuple));
        VM_NEXT();
    }

    VM_CASE(OP_BUILD_RANGE): {
        bool inclusive = READ_BYTE() != 0;
        Value stopValue = PEEK(0), startValue = PEEK(1);
        if (!IS_INT(startValue) || !IS_INT(stopValue)) {
            THROW(vm.cTypeError, "range bounds must be int, not '%s' and '%s'",
                  jaiTypeNameStatic(startValue), jaiTypeNameStatic(stopValue));
        }
        SAVE_STATE();
        ObjRange *range = jaiRangeNew(AS_INT(startValue), AS_INT(stopValue), 1,
                                      inclusive);
        LOAD_STATE();
        DROP(2);
        PUSH(OBJ_VAL(range));
        VM_NEXT();
    }

    VM_CASE(OP_LIST_APPEND): {
        int depth = READ_U16();
        /* The operand is the container's peek index outright (spec §3.6). */
        Value target = PEEK(depth);
        if (!IS_LIST(target)) {
            THROW(vm.cTypeError, "comprehension target is not a list");
        }
        SAVE_STATE();
        jaiListPush(AS_LIST(target), PEEK(0));
        if (vm.hasException) goto vmThrow;
        LOAD_STATE();
        DROP(1);
        VM_NEXT();
    }

    VM_CASE(OP_DICT_INSERT): {
        int depth = READ_U16();
        /* The operand is the container's peek index outright (spec §3.6). */
        Value target = PEEK(depth);
        if (!IS_DICT(target)) {
            THROW(vm.cTypeError, "comprehension target is not a dict");
        }
        SAVE_STATE();
        (void)jaiDictSet(AS_DICT(target), PEEK(1), PEEK(0));
        if (vm.hasException) goto vmThrow;
        LOAD_STATE();
        DROP(2);
        VM_NEXT();
    }

    VM_CASE(OP_SET_ADD): {
        int depth = READ_U16();
        /* The operand is the container's peek index outright (spec §3.6). */
        Value target = PEEK(depth);
        if (!IS_SET(target)) {
            THROW(vm.cTypeError, "comprehension target is not a set");
        }
        SAVE_STATE();
        (void)jaiSetAdd(AS_SET(target), PEEK(0));
        if (vm.hasException) goto vmThrow;
        LOAD_STATE();
        DROP(1);
        VM_NEXT();
    }

    VM_CASE(OP_GET_INDEX): {
        Value result;
        if (JAI_LIKELY(indexGetFast(stackTop[-2], stackTop[-1], &result))) {
            DROP(2);
            PUSH(result);
            VM_NEXT();
        }
        SAVE_STATE();
        if (!indexGet(stackTop[-2], stackTop[-1], &result)) goto vmThrow;
        LOAD_STATE();
        DROP(2);
        PUSH(result);
        VM_NEXT();
    }

    VM_CASE(OP_SET_INDEX): {
        SAVE_STATE();
        if (!indexSet(stackTop[-3], stackTop[-2], stackTop[-1])) goto vmThrow;
        LOAD_STATE();
        DROP(3);
        VM_NEXT_HINT(OP_LOOP);
    }

    VM_CASE(OP_GET_SLICE): {
        uint8_t flags = READ_BYTE();
        bool hasStart = (flags & 1) != 0;
        bool hasStop = (flags & 2) != 0;
        bool hasStep = (flags & 4) != 0;
        int operands = (hasStart ? 1 : 0) + (hasStop ? 1 : 0) + (hasStep ? 1 : 0);

        SAVE_STATE();
        Value *args = stackTop - operands;
        int at = 0;
        Value startValue = hasStart ? args[at++] : NULL_VAL;
        Value stopValue = hasStop ? args[at++] : NULL_VAL;
        Value stepValue = hasStep ? args[at++] : NULL_VAL;
        Value result;
        if (!jaiSliceGet(args[-1], startValue, stopValue, stepValue,
                         hasStart, hasStop, hasStep, &result)) {
            goto vmThrow;
        }
        LOAD_STATE();
        DROP(operands + 1);
        PUSH(result);
        VM_NEXT();
    }

    VM_CASE(OP_SET_SLICE): {
        uint8_t flags = READ_BYTE();
        bool hasStart = (flags & 1) != 0;
        bool hasStop = (flags & 2) != 0;
        bool hasStep = (flags & 4) != 0;
        int operands = (hasStart ? 1 : 0) + (hasStop ? 1 : 0) + (hasStep ? 1 : 0);

        SAVE_STATE();
        Value assigned = stackTop[-1];
        Value *args = stackTop - 1 - operands;
        Value container = args[-1];
        if (!IS_LIST(container)) {
            THROW(vm.cTypeError, "'%s' value does not support slice assignment",
                  jaiTypeNameStatic(container));
        }
        int at = 0;
        Value startValue = hasStart ? args[at++] : NULL_VAL;
        Value stopValue = hasStop ? args[at++] : NULL_VAL;
        Value stepValue = hasStep ? args[at++] : NULL_VAL;

        ObjList *list = AS_LIST(container);
        int64_t start, stop, step;
        if (!sliceBounds(startValue, stopValue, stepValue, hasStart, hasStop,
                         hasStep, list->count, &start, &stop, &step)) {
            goto vmThrow;
        }

        const Value *source = NULL;
        int sourceCount = 0;
        if (IS_LIST(assigned)) {
            source = AS_LIST(assigned)->items;
            sourceCount = AS_LIST(assigned)->count;
        } else if (IS_TUPLE(assigned)) {
            source = AS_TUPLE(assigned)->items;
            sourceCount = (int)AS_TUPLE(assigned)->count;
        } else {
            THROW(vm.cTypeError,
                  "slice assignment requires a list or tuple, not '%s'",
                  jaiTypeNameStatic(assigned));
        }

        /* Only same-length replacement: resizing through a slice would make
         * the surrounding indices silently wrong. */
        int64_t selected = 0;
        for (int64_t i = start; (step > 0) ? (i < stop) : (i > stop); i += step) {
            if (i < 0 || i >= list->count) break;
            selected++;
        }
        if (selected != sourceCount) {
            THROW(vm.cValueError,
                  "slice assignment needs %" PRId64 " values but %d were given",
                  selected, sourceCount);
        }
        int64_t written = 0;
        for (int64_t i = start; (step > 0) ? (i < stop) : (i > stop); i += step) {
            if (i < 0 || i >= list->count) break;
            list->items[i] = source[written++];
        }
        if (written > 0) jaiListTouch(list);
        LOAD_STATE();
        DROP(operands + 2);
        VM_NEXT();
    }

    VM_CASE(OP_UNPACK): {
        int count = READ_BYTE();
        int restIndex = READ_BYTE();
        SAVE_STATE();
        Value source = stackTop[-1];

        const Value *items = NULL;
        int available = 0;
        if (IS_LIST(source)) {
            items = AS_LIST(source)->items;
            available = AS_LIST(source)->count;
        } else if (IS_TUPLE(source)) {
            items = AS_TUPLE(source)->items;
            available = (int)AS_TUPLE(source)->count;
        } else {
            THROW(vm.cTypeError, "cannot destructure a '%s' value",
                  jaiTypeNameStatic(source));
        }

        bool hasRest = (restIndex != 255);
        int fixed = hasRest ? count - 1 : count;
        if ((hasRest && available < fixed) || (!hasRest && available != count)) {
            THROW(vm.cValueError,
                  "cannot unpack %d value%s into %d target%s", available,
                  available == 1 ? "" : "s", count, count == 1 ? "" : "s");
        }

        Value unpacked[JAI_MAX_ARGS];
        if (count > JAI_MAX_ARGS) {
            THROW(vm.cRuntimeError, "too many destructuring targets");
        }
        int restCount = available - fixed;
        int read = 0;
        for (int i = 0; i < count; i++) {
            if (hasRest && i == restIndex) {
                ObjList *rest = jaiListNew(restCount);
                for (int j = 0; j < restCount; j++) rest->items[j] = items[read + j];
                rest->count = restCount;
                read += restCount;
                unpacked[i] = OBJ_VAL(rest);
                /* jaiListNew can collect; re-read the (still rooted) source. */
                items = IS_LIST(source) ? AS_LIST(source)->items
                                        : AS_TUPLE(source)->items;
            } else {
                unpacked[i] = items[read++];
            }
        }
        LOAD_STATE();
        DROP(1);
        /* Right to left, so the leftmost target is on top and a run of
         * SET_LOCAL/POP assigns in source order. */
        for (int i = count - 1; i >= 0; i--) PUSH(unpacked[i]);
        VM_NEXT();
    }

    /* --- objects, classes, traits (spec §3.7) --- */

    VM_CASE(OP_CLASS): {
        Value spec = READ_CONST();
        SAVE_STATE();
        Obj *created = classSpecInstantiate(spec);
        if (created == NULL) goto vmThrow;
        LOAD_STATE();
        PUSH(OBJ_VAL(created));
        VM_NEXT();
    }

    VM_CASE(OP_INHERIT): {
        Value subValue = PEEK(0), superValue = PEEK(1);
        if (!IS_CLASS(superValue)) {
            THROW(vm.cTypeError, "a superclass must be a class, not '%s'",
                  jaiTypeNameStatic(superValue));
        }
        if (!IS_CLASS(subValue)) {
            THROW(vm.cTypeError, "INHERIT expected a class on top of the stack");
        }
        if (jaiClassIsSubclassOf(AS_CLASS(superValue), AS_CLASS(subValue))) {
            THROW(vm.cRuntimeError, "cyclic inheritance involving class '%s'",
                  AS_CLASS(subValue)->name != NULL
                      ? AS_CLASS(subValue)->name->chars : "?");
        }
        SAVE_STATE();
        jaiClassInherit(AS_CLASS(subValue), AS_CLASS(superValue));
        if (vm.hasException) goto vmThrow;
        LOAD_STATE();
        VM_NEXT();
    }

    VM_CASE(OP_IMPL_TRAIT): {
        Value traitRef = READ_CONST();
        SAVE_STATE();
        Value implementor = PEEK(0);
        if (!IS_CLASS(implementor) && !IS_TRAIT(implementor)) {
            THROW(vm.cTypeError,
                  "IMPL_TRAIT expected a class or trait on the stack");
        }
        ObjClass *klass = IS_CLASS(implementor) ? AS_CLASS(implementor) : NULL;

        ObjTrait *trait = NULL;
        if (IS_TRAIT(traitRef)) {
            trait = AS_TRAIT(traitRef);
        } else if (IS_STRING(traitRef)) {
            Value found;
            if ((frame->module != NULL &&
                 jaiModuleGet(frame->module, AS_STRING(traitRef), &found) &&
                 IS_TRAIT(found)) ||
                (vm.builtins != NULL &&
                 jaiModuleGet(vm.builtins, AS_STRING(traitRef), &found) &&
                 IS_TRAIT(found))) {
                trait = AS_TRAIT(found);
            }
        }
        if (trait == NULL) {
            THROW(vm.cTypeError, "unknown trait in `impl` for '%s'",
                  jaiTypeNameStatic(implementor));
        }

        if (klass == NULL) {
            /* `trait Sub: Super` — record the supertrait and inherit both its
             * requirements and its defaults, so a class implementing Sub is
             * checked against, and inherits, the whole chain. */
            ObjTrait *sub = AS_TRAIT(implementor);
            uint16_t oldSupers = sub->superCount;
            sub->supers = JAI_GROW_ARRAY(ObjTrait *, sub->supers, oldSupers,
                                         oldSupers + 1);
            sub->supers[oldSupers] = trait;
            sub->superCount = (uint16_t)(oldSupers + 1);

            int si = 0;
            Value skey, svalue, sexisting;
            while (jaiTableNext(&trait->required, &si, &skey, &svalue)) {
                if (!jaiTableGet(&sub->required, skey, &sexisting))
                    jaiTableSet(&sub->required, skey, svalue);
            }
            si = 0;
            while (jaiTableNext(&trait->defaults, &si, &skey, &svalue)) {
                if (!jaiTableGet(&sub->defaults, skey, &sexisting))
                    jaiTableSet(&sub->defaults, skey, svalue);
            }
            LOAD_STATE();
            VM_NEXT();
        }

        uint16_t oldCount = klass->traitCount;
        klass->traits = JAI_GROW_ARRAY(ObjTrait *, klass->traits, oldCount,
                                       oldCount + 1);
        klass->traits[oldCount] = trait;
        klass->traitCount = (uint16_t)(oldCount + 1);

        /* Copy the trait's default implementations in without overwriting
         * anything the class declares, so the two possible emission orders
         * (methods before or after the impl) both come out right. */
        int i = 0;
        Value key, value;
        while (jaiTableNext(&trait->defaults, &i, &key, &value)) {
            if (!IS_STRING(key)) continue;
            Value existing;
            if (jaiTableGetInterned(&klass->methods, AS_STRING(key), &existing)) {
                continue;
            }
            jaiClassAddMethod(klass, AS_STRING(key), value, VIS_PUBLIC, 0);
        }
        jaiClassRefreshDunders(klass);
        LOAD_STATE();
        VM_NEXT();
    }

    VM_CASE(OP_METHOD): {
        ObjString *name = AS_STRING(READ_CONST());
        uint8_t flags = READ_BYTE();
        uint8_t visByte = READ_BYTE();
        SAVE_STATE();
        Value method = PEEK(0);
        Value owner = PEEK(1);
        /* The flags byte is full of FunctionFlags, so visibility rides in a
         * byte of its own; a value from a corrupt chunk is read as private,
         * which can only ever deny an access. */
        Visibility vis = visByte <= (uint8_t)VIS_PUBLIC ? (Visibility)visByte
                                                        : VIS_PRIVATE;
        /* Record where the function was declared, for the visibility test:
         * "private to the declaring class" is a property of the code, and slot
         * 0 only reports it for a call that came through OP_INVOKE. */
        if (IS_CLASS(owner)) {
            ObjFunction *fn = IS_CLOSURE(method)    ? AS_CLOSURE(method)->fn
                              : IS_FUNCTION(method) ? AS_FUNCTION(method)
                                                    : NULL;
            if (fn != NULL) fn->owner = AS_CLASS(owner);
        }
        if (IS_CLASS(owner)) {
            jaiClassAddMethod(AS_CLASS(owner), name, method, vis, flags);
        } else if (IS_TRAIT(owner)) {
            /* A trait's only bodied methods are its defaults; the requirements
             * travelled in the spec constant. */
            jaiTableSet(&AS_TRAIT(owner)->defaults, OBJ_VAL(name), method);
        } else if (IS_ENUM(owner)) {
            jaiTableSet(&AS_ENUM(owner)->methods, OBJ_VAL(name), method);
            /* A member cache keyed on the old id can no longer be reached, so
             * anything memoised before this method existed is stale by
             * construction. Only runs while the enum is being defined. */
            AS_ENUM(owner)->shapeId = jaiFreshShapeId();
        } else {
            THROW(vm.cTypeError,
                  "METHOD expected a class, trait, or enum under the closure");
        }
        LOAD_STATE();
        DROP(1);
        VM_NEXT();
    }

    VM_CASE(OP_FIELD_DEF): {
        ObjString *name = AS_STRING(READ_CONST());
        uint8_t info = READ_BYTE();
        SAVE_STATE();
        Value classValue = PEEK(0);
        if (!IS_CLASS(classValue)) {
            THROW(vm.cTypeError, "FIELD_DEF expected a class on the stack");
        }
        if (!classDeclareField(AS_CLASS(classValue), name, info)) goto vmThrow;
        LOAD_STATE();
        VM_NEXT();
    }

    VM_CASE(OP_GET_FIELD): {
        uint32_t nameIdx = READ_U24();
        uint16_t cacheIdx = READ_U16();
        Value receiver = PEEK(0);

        if (IS_INSTANCE(receiver)) {
            ObjInstance *instance = AS_INSTANCE(receiver);
            InlineCache *ic = cacheAt(frameChunk(frame), cacheIdx);
            if (ic != NULL && instance->klass != NULL && ic->state != IC_EMPTY) {
                for (int w = 0; w < ic->count; w++) {
                    if (ic->shapeId[w] != instance->klass->shapeId) continue;
                    if (ic->payload[w] >= instance->fieldCount) break;
                    vm.icHits++;
                    stackTop[-1] = instance->fields[ic->payload[w]];
                    VM_NEXT();
                }
            }
            vm.icMisses++;
        }

        if (IS_ENUM(receiver)) {
            InlineCache *ic = cacheAt(frameChunk(frame), cacheIdx);
            if (ic != NULL && ic->state != IC_EMPTY) {
                ObjEnum *en = AS_ENUM(receiver);
                for (int w = 0; w < ic->count; w++) {
                    /* Shape ids are unique across classes and enums alike, so
                     * an instance way can never answer here by accident. */
                    if (ic->shapeId[w] != en->shapeId) continue;
                    vm.icHits++;
                    stackTop[-1] = ic->cached[w];
                    VM_NEXT();
                }
            }
            vm.icMisses++;
        }

        SAVE_STATE();
        ObjString *name = AS_STRING(constants[nameIdx]);
        Value result;
        if (!getPropertyInto(receiver, name, &result, true,
                             cacheAt(frameChunk(frame), cacheIdx))) {
            goto vmThrow;
        }
        LOAD_STATE();
        stackTop[-1] = result;
        VM_NEXT();
    }

    /* GET_LOCAL S; GET_FIELD K,C fused (spec §3.3). The receiver never reaches
     * the stack, which is the point: `self.x` is the single most frequent pair
     * the VM executes. It is still a GC root — a frame's slots are part of the
     * value stack — so the slow path below is as safe as OP_GET_FIELD's. */
    VM_CASE(OP_GET_FIELD_LOCAL): {
        uint16_t slot = READ_U16();
        uint32_t nameIdx = READ_U24();
        uint16_t cacheIdx = READ_U16();
        Value receiver = slots[slot];

        if (IS_INSTANCE(receiver)) {
            ObjInstance *instance = AS_INSTANCE(receiver);
            InlineCache *ic = cacheAt(frameChunk(frame), cacheIdx);
            if (ic != NULL && instance->klass != NULL && ic->state != IC_EMPTY) {
                for (int w = 0; w < ic->count; w++) {
                    if (ic->shapeId[w] != instance->klass->shapeId) continue;
                    if (ic->payload[w] >= instance->fieldCount) break;
                    vm.icHits++;
                    PUSH(instance->fields[ic->payload[w]]);
                    VM_NEXT();
                }
            }
            vm.icMisses++;
        }

        if (IS_ENUM(receiver)) {
            InlineCache *ic = cacheAt(frameChunk(frame), cacheIdx);
            if (ic != NULL && ic->state != IC_EMPTY) {
                ObjEnum *en = AS_ENUM(receiver);
                for (int w = 0; w < ic->count; w++) {
                    /* Shape ids are unique across classes and enums alike, so
                     * an instance way can never answer here by accident. */
                    if (ic->shapeId[w] != en->shapeId) continue;
                    vm.icHits++;
                    PUSH(ic->cached[w]);
                    VM_NEXT();
                }
            }
            vm.icMisses++;
        }

        SAVE_STATE();
        ObjString *name = AS_STRING(constants[nameIdx]);
        Value result;
        if (!getPropertyInto(receiver, name, &result, true,
                             cacheAt(frameChunk(frame), cacheIdx))) {
            goto vmThrow;
        }
        LOAD_STATE();
        PUSH(result);
        VM_NEXT();
    }

    VM_CASE(OP_SET_FIELD): {
        uint32_t nameIdx = READ_U24();
        uint16_t cacheIdx = READ_U16();
        Value receiver = PEEK(1);
        Value value = PEEK(0);

        if (IS_INSTANCE(receiver)) {
            ObjInstance *instance = AS_INSTANCE(receiver);
            InlineCache *ic = cacheAt(frameChunk(frame), cacheIdx);
            if (ic != NULL && instance->klass != NULL && ic->state != IC_EMPTY) {
                for (int w = 0; w < ic->count; w++) {
                    if (ic->shapeId[w] != instance->klass->shapeId) continue;
                    if (ic->payload[w] >= instance->fieldCount) break;
                    /* The kind guard has to be here too, not only in
                     * jaiSetProperty: once a site's cache is warm every later
                     * write takes this path and never reaches it. A field's
                     * slot is its index in klass->fields by construction
                     * (classDeclareField assigns slot = fieldCount as it
                     * appends), so this is one indexed load, not a lookup. */
                    if (ic->payload[w] < instance->klass->fieldCount) {
                        const FieldInfo *fi =
                            &instance->klass->fields[ic->payload[w]];
                        if (!jaiKindAccepts(fi->typeId, value)) {
                            SAVE_STATE();
                            (void)throwFieldKind(fi, value);
                            goto vmThrow;
                        }
                    }
                    vm.icHits++;
                    instance->fields[ic->payload[w]] = value;
                    DROP(2);
                    VM_NEXT();
                }
            }
            vm.icMisses++;
        }

        SAVE_STATE();
        ObjString *name = AS_STRING(constants[nameIdx]);
        if (!jaiSetProperty(receiver, name, value)) goto vmThrow;

        /* Cache only a plain field write: a setter has to keep running. */
        if (IS_INSTANCE(receiver)) {
            ObjInstance *instance = AS_INSTANCE(receiver);
            ObjClass *klass = instance->klass;
            Value setter;
            InlineCache *ic = cacheAt(frameChunk(frame), cacheIdx);
            const FieldInfo *field = jaiClassFieldInfo(klass, name);
            if (ic != NULL && field != NULL && klass != NULL &&
                !jaiTableGetInterned(&klass->setters, name, &setter) &&
                ic->state != IC_MEGA) {
                if (ic->count < JAI_IC_WAYS) {
                    ic->shapeId[ic->count] = klass->shapeId;
                    ic->payload[ic->count] = field->slot;
                    ic->cached[ic->count] = NULL_VAL;
                    ic->count++;
                    ic->state = (ic->count == 1) ? IC_MONO : IC_POLY;
                } else {
                    ic->state = IC_MEGA;
                }
            }
        }
        LOAD_STATE();
        DROP(2);
        VM_NEXT();
    }

    VM_CASE(OP_GET_SUPER): {
        ObjString *name = AS_STRING(READ_CONST());
        SAVE_STATE();
        Value receiver = PEEK(0);
        ObjClass *start = NULL;
        if (IS_INSTANCE(receiver)) start = AS_INSTANCE(receiver)->klass;
        else if (IS_CLASS(receiver)) start = AS_CLASS(receiver);
        if (start == NULL || start->superclass == NULL) {
            THROW(vm.cRuntimeError, "'super.%s' has no superclass", name->chars);
        }
        Value method;
        if (!findMethod(start->superclass, name, &method)) {
            THROW(vm.cAttributeError, "superclass '%s' has no member '%s'",
                  start->superclass->name != NULL ? start->superclass->name->chars
                                                  : "?",
                  name->chars);
        }
        ObjBound *bound = jaiBoundNew(receiver, method);
        LOAD_STATE();
        stackTop[-1] = OBJ_VAL(bound);
        VM_NEXT();
    }

    VM_CASE(OP_NEW): {
        Value classRef = READ_CONST();
        int argc = READ_BYTE();
        SAVE_STATE();

        Value classValue = classRef;
        if (IS_STRING(classRef)) {
            Value found;
            if ((frame->module != NULL &&
                 jaiModuleGet(frame->module, AS_STRING(classRef), &found)) ||
                (vm.builtins != NULL &&
                 jaiModuleGet(vm.builtins, AS_STRING(classRef), &found))) {
                classValue = found;
            } else {
                THROW(vm.cNameError, "undefined class '%s'",
                      AS_CSTRING(classRef));
            }
        }
        if (!IS_CLASS(classValue)) {
            THROW(vm.cTypeError, "NEW operand is not a class");
        }
        /* The arguments are already on the stack but the callee slot is not,
         * so open one below them for the instance. */
        if (!ensureStack(1)) goto vmThrow;
        Value *args = vm.stackTop - argc;
        memmove(args + 1, args, sizeof(Value) * (size_t)argc);
        args[0] = classValue;
        vm.stackTop++;
        if (invokeCallable(classValue, argc) == CALL_ERROR) goto vmThrow;
        LOAD_STATE();
        VM_NEXT();
    }

    VM_CASE(OP_ENUM_NEW): {
        Value enumRef = READ_CONST();
        int tag = READ_BYTE();
        int argc = READ_BYTE();
        SAVE_STATE();

        Value enumValue = enumRef;
        if (IS_STRING(enumRef)) {
            Value found;
            if ((frame->module != NULL &&
                 jaiModuleGet(frame->module, AS_STRING(enumRef), &found)) ||
                (vm.builtins != NULL &&
                 jaiModuleGet(vm.builtins, AS_STRING(enumRef), &found))) {
                enumValue = found;
            }
        }
        if (!IS_ENUM(enumValue)) {
            THROW(vm.cTypeError, "ENUM_NEW operand is not an enum");
        }
        ObjEnum *enumType = AS_ENUM(enumValue);
        if (tag >= (int)enumType->variantCount) {
            THROW(vm.cRuntimeError, "enum '%s' has no variant %d",
                  enumType->name != NULL ? enumType->name->chars : "?", tag);
        }
        /* Share the one value of a payload-less variant, exactly as
         * `enumMember` does: two spellings of `Color.Red` must be the same
         * object or `is` stops meaning identity (spec §4.2). */
        EnumVariant *variant = &enumType->variants[tag];
        ObjEnumVal *value;
        if (variant->arity == 0 && argc == 0) {
            if (variant->unit == NULL) {
                variant->unit = jaiEnumValNew(enumType, (uint16_t)tag, NULL, 0);
            }
            value = variant->unit;
        } else {
            value = jaiEnumValNew(enumType, (uint16_t)tag, vm.stackTop - argc, argc);
        }
        LOAD_STATE();
        DROP(argc);
        PUSH(OBJ_VAL(value));
        VM_NEXT();
    }

    VM_CASE(OP_ENUM_TAG): {
        Value value = PEEK(0);
        if (!IS_ENUM_VAL(value)) {
            THROW(vm.cTypeError, "ENUM_TAG expected an enum value, not '%s'",
                  jaiTypeNameStatic(value));
        }
        PUSH(INT_VAL(AS_ENUM_VAL(value)->tag));
        VM_NEXT();
    }

    VM_CASE(OP_ENUM_FIELD): {
        uint8_t index = READ_BYTE();
        Value value = PEEK(0);
        if (!IS_ENUM_VAL(value) || index >= AS_ENUM_VAL(value)->count) {
            THROW(vm.cIndexError, "enum payload field %u is out of range",
                  (unsigned)index);
        }
        PUSH(AS_ENUM_VAL(value)->payload[index]);
        VM_NEXT();
    }

    VM_CASE(OP_IS_INSTANCE): {
        Value typeConstant = READ_CONST();
        stackTop[-1] = BOOL_VAL(valueMatchesType(stackTop[-1], typeConstant));
        VM_NEXT();
    }

    /* --- exceptions and defer (spec §3.8) --- */

    VM_CASE(OP_THROW): {
        Value exception = POP();
        SAVE_STATE();
        if (IS_CLASS(exception)) {
            /* `throw SomeError` without a call: construct it here so handlers
             * always see an instance. */
            Value constructed;
            if (!jaiCallValue(exception, 0, NULL, &constructed)) goto vmThrow;
            exception = constructed;
        }
        (void)jaiThrowValue(exception);
        goto vmThrow;
    }

    VM_CASE(OP_RERAISE):
        SAVE_STATE();
        if (IS_NULL(vm.pendingException)) {
            THROW(vm.cRuntimeError, "no exception to re-raise");
        }
        (void)jaiThrowValue(vm.pendingException);
        goto vmThrow;

    VM_CASE(OP_PUSH_HANDLER): {
        int16_t offset = READ_I16();
        uint32_t typeConst = READ_U24();
        ExcHandler handler;
        handler.handlerOffset = (uint32_t)((ip + offset) - frameChunk(frame)->code);
        handler.typeConst = typeConst;
        handler.frameIndex = vm.frameCount - 1;
        handler.stackTop = stackTop;
        SAVE_STATE();
        JAI_VEC_PUSH(ExcHandler, &vm.handlers, handler);
        VM_NEXT();
    }

    VM_CASE(OP_POP_HANDLER):
        if (vm.handlers.count > frame->handlerBase) vm.handlers.count--;
        VM_NEXT();

    VM_CASE(OP_PUSH_FINALLY): {
        int16_t offset = READ_I16();
        ExcHandler handler;
        handler.handlerOffset = (uint32_t)((ip + offset) - frameChunk(frame)->code);
        handler.typeConst = JAI_HANDLER_FINALLY;
        handler.frameIndex = vm.frameCount - 1;
        handler.stackTop = stackTop;
        SAVE_STATE();
        JAI_VEC_PUSH(ExcHandler, &vm.handlers, handler);
        VM_NEXT();
    }

    VM_CASE(OP_END_FINALLY):
        /* Reached either by falling out of the try (nothing to do) or by the
         * unwinder, which recorded that the exception is still in flight. */
        if (sFinallyPending > 0) {
            sFinallyPending--;
            SAVE_STATE();
            (void)jaiThrowValue(vm.pendingException);
            goto vmThrow;
        }
        VM_NEXT();

    VM_CASE(OP_PUSH_DEFER): {
        Value deferred = READ_CONST();
        SAVE_STATE();
        if (IS_FUNCTION(deferred)) {
            ObjClosure *thunk = jaiClosureNew(AS_FUNCTION(deferred));
            /* emitDefer gave the thunk the enclosing function's upvalue count
             * and indices, so the whole array is shared rather than captured. */
            for (int i = 0; i < thunk->upvalueCount &&
                            i < frame->closure->upvalueCount; i++) {
                thunk->upvalues[i] = frame->closure->upvalues[i];
            }
            deferred = OBJ_VAL(thunk);
        } else if (!valueIsCallable(deferred)) {
            /* The constant is a placeholder; the closure was built by a
             * preceding OP_CLOSURE and is on the stack. */
            deferred = vm.stackTop > vm.stack ? vm.stackTop[-1] : NULL_VAL;
        }
        if (!valueIsCallable(deferred)) {
            THROW(vm.cTypeError, "defer requires a callable");
        }
        JAI_VEC_PUSH(Value, &vm.defers, deferred);
        LOAD_STATE();
        VM_NEXT();
    }

    VM_CASE(OP_RUN_DEFERS):
        SAVE_STATE();
        (void)runFrameDefers(frame);
        if (vm.hasException) goto vmThrow;
        LOAD_STATE();
        VM_NEXT();

    VM_CASE(OP_MATCH_EXC): {
        Value typeConstant = READ_CONST();
        PUSH(BOOL_VAL(valueMatchesType(vm.pendingException, typeConstant)));
        VM_NEXT();
    }

    VM_CASE(OP_GET_EXC):
        PUSH(vm.pendingException);
        VM_NEXT();

    /* --- pattern matching (spec §3.9) --- */

    VM_CASE(OP_MATCH_CONST): {
        Value expected = READ_CONST();
        int16_t offset = READ_I16();
        SAVE_STATE();
        bool equal = jaiValuesEqual(PEEK(0), expected);
        if (vm.hasException) goto vmThrow;
        LOAD_STATE();
        if (!equal) ip += offset;
        VM_NEXT();
    }

    VM_CASE(OP_MATCH_RANGE): {
        Value low = READ_CONST();
        Value high = READ_CONST();
        bool inclusive = READ_BYTE() != 0;
        int16_t offset = READ_I16();
        Value subject = PEEK(0);

        int cmpLow = 0, cmpHigh = 0;
        SAVE_STATE();
        bool ordered = jaiValueCompare(subject, low, &cmpLow) &&
                       jaiValueCompare(subject, high, &cmpHigh);
        if (vm.hasException) goto vmThrow;
        LOAD_STATE();
        bool inside = ordered && cmpLow >= 0 &&
                      (inclusive ? cmpHigh <= 0 : cmpHigh < 0);
        if (!inside) ip += offset;
        VM_NEXT();
    }

    VM_CASE(OP_MATCH_TYPE): {
        Value typeConstant = READ_CONST();
        int16_t offset = READ_I16();
        if (!valueMatchesType(PEEK(0), typeConstant)) ip += offset;
        VM_NEXT();
    }

    VM_CASE(OP_MATCH_SEQ): {
        int count = READ_BYTE();
        bool hasRest = READ_BYTE() != 0;
        int16_t offset = READ_I16();
        Value subject = PEEK(0);

        int length = -1;
        if (IS_LIST(subject))       length = AS_LIST(subject)->count;
        else if (IS_TUPLE(subject)) length = (int)AS_TUPLE(subject)->count;

        bool matched = length >= 0 &&
                       (hasRest ? (length >= count) : (length == count));
        if (!matched) ip += offset;
        VM_NEXT();
    }

    VM_CASE(OP_MATCH_FIELDS): {
        Value fieldNames = READ_CONST();
        int16_t offset = READ_I16();
        Value subject = PEEK(0);
        SAVE_STATE();

        const Value *names = NULL;
        int count = 0;
        if (IS_TUPLE(fieldNames)) {
            names = AS_TUPLE(fieldNames)->items;
            count = (int)AS_TUPLE(fieldNames)->count;
        } else if (IS_STRING(fieldNames)) {
            names = &fieldNames;
            count = 1;
        }
        if (!IS_INSTANCE(subject) || names == NULL) {
            ip += offset;
            VM_NEXT();
        }

        ObjInstance *instance = AS_INSTANCE(subject);
        Value extracted[JAI_MAX_ARGS];
        if (count > JAI_MAX_ARGS) {
            THROW(vm.cRuntimeError, "too many fields in a class pattern");
        }
        bool ok = true;
        for (int i = 0; i < count && ok; i++) {
            if (!IS_STRING(names[i])) { ok = false; break; }
            const FieldInfo *field = jaiClassFieldInfo(instance->klass,
                                                       AS_STRING(names[i]));
            if (field == NULL || field->slot >= instance->fieldCount) {
                ok = false;
                break;
            }
            extracted[i] = instance->fields[field->slot];
        }
        if (!ok) {
            ip += offset;
            VM_NEXT();
        }
        if (!ensureStack(count)) goto vmThrow;
        LOAD_STATE();
        for (int i = 0; i < count; i++) PUSH(extracted[i]);
        VM_NEXT();
    }

    VM_CASE(OP_BIND): {
        uint16_t slot = READ_U16();
        slots[slot] = POP();
        VM_NEXT();
    }

    /* --- modules and misc (spec §3.10) --- */

    VM_CASE(OP_IMPORT): {
        ObjString *path = AS_STRING(READ_CONST());
        SAVE_STATE();
        char dirBuffer[JAI_MAX_PATH];
        const char *fromDir = NULL;
        if (frame->module != NULL && frame->module->path != NULL &&
            frame->module->path->length > 0) {
            jaiPathDirname(dirBuffer, sizeof dirBuffer,
                           frame->module->path->chars);
            fromDir = dirBuffer;
        }
        ObjModule *imported = jaiImportModule(path->chars, fromDir);
        if (imported == NULL) {
            if (!vm.hasException) {
                (void)jaiThrow(vm.cImportError, "cannot import module '%s'",
                               path->chars);
            }
            goto vmThrow;
        }
        LOAD_STATE();
        PUSH(OBJ_VAL(imported));
        VM_NEXT();
    }

    VM_CASE(OP_IMPORT_FROM): {
        ObjString *name = AS_STRING(READ_CONST());
        SAVE_STATE();
        Value moduleValue = PEEK(0);
        if (!IS_MODULE(moduleValue)) {
            THROW(vm.cImportError, "'from' import expected a module, not '%s'",
                  jaiTypeNameStatic(moduleValue));
        }
        ObjModule *module = AS_MODULE(moduleValue);
        Value member;
        bool hidden = false;
        if (!moduleMember(module, name, &member, &hidden)) {
            /* E0802: the name exists but is not part of the module's surface,
             * or it does not exist at all. */
            THROW(vm.cImportError, "'%s' is not exported by module '%s'",
                  name->chars, module->name != NULL ? module->name->chars : "?");
        }
        LOAD_STATE();
        PUSH(member);
        VM_NEXT();
    }

    VM_CASE(OP_EXPORT): {
        ObjString *name = AS_STRING(READ_CONST());
        SAVE_STATE();
        if (frame->module == NULL) {
            THROW(vm.cRuntimeError, "no module in scope to export '%s' from",
                  name->chars);
        }
        jaiGCPushRoot(OBJ_VAL(frame->module));
        (void)jaiTableSetInterned(&frame->module->exports, name, BOOL_VAL(true));
        jaiGCPopRoot();
        LOAD_STATE();
        VM_NEXT();
    }

    VM_CASE(OP_ASSERT_FAIL): {
        Value message = READ_CONST();
        SAVE_STATE();
        const char *text = "assertion failed";
        if (IS_STRING(message)) {
            text = AS_CSTRING(message);
        } else if (stackTop > slots && IS_STRING(PEEK(0))) {
            text = AS_CSTRING(PEEK(0));
        }
        THROW(vm.cAssertionError, "%s", text);
    }

    VM_CASE(OP_TYPE_GUARD): {
        Value typeConstant = READ_CONST();
        Value subject = PEEK(0);
        if (!valueMatchesType(subject, typeConstant)) {
            /* An int arriving at a float boundary widens (spec §2.2), the same
             * as it would have where the checker could see it. Only the bare
             * `float` guard converts: `x is float` and `match` ask what a value
             * *is*, and an int is not a float, so valueMatchesType above is
             * left alone. */
            if (IS_INT(subject) && IS_STRING(typeConstant) &&
                AS_STRING(typeConstant)->length == 5 &&
                memcmp(AS_STRING(typeConstant)->chars, "float", 5) == 0) {
                stackTop[-1] = FLOAT_VAL((double)AS_INT(subject));
                VM_NEXT();
            }
            THROW(vm.cTypeError, "expected '%s' but got '%s'",
                  typeConstantName(typeConstant), jaiTypeNameStatic(subject));
        }
        VM_NEXT();
    }

    VM_CASE(OP_HALT):
        SAVE_STATE();
        while (vm.frameCount > baseFrameCount) {
            CallFrame *halting = &vm.frames[vm.frameCount - 1];
            closeUpvalues(halting->base);
            vm.stackTop = halting->base;
            vm.frameCount--;
        }
        vm.handlers.count = 0;
        vm.defers.count = 0;
        *vm.stackTop++ = NULL_VAL;
        return JAI_RUN_OK;

#if !JAI_COMPUTED_GOTO
    default:
        SAVE_STATE();
        (void)jaiThrow(vm.cRuntimeError, "unknown opcode 0x%02x",
                       (unsigned)instStart[0]);
        goto vmThrow;
#endif

    }   /* VM_DISPATCH */

vmThrow: {
        if (!vm.hasException) {
            /* A helper failed without raising; make the failure visible rather
             * than resuming with a corrupt stack. */
            (void)jaiThrow(vm.cRuntimeError,
                           "internal error: failed operation raised nothing");
        }
        CallFrame *faulting = &vm.frames[vm.frameCount - 1];
        Chunk *chunk = frameChunk(faulting);
        ptrdiff_t at = instStart - chunk->code;
        if (at < 0 || at > chunk->count) at = 0;

        if (!unwindToHandler(baseFrameCount, (uint32_t)at)) {
            return JAI_RUN_RUNTIME_ERROR;
        }
        LOAD_STATE();
        VM_NEXT();
    }
}

static JaiRunResult run(int baseFrameCount) {
    if (sRunDepth >= JAI_MAX_NESTED_RUN) {
        (void)jaiThrow(vm.cRecursionError,
                       "maximum native re-entry depth exceeded (%d)",
                       JAI_MAX_NESTED_RUN);
        while (vm.frameCount > baseFrameCount) popFrameForUnwind();
        return JAI_RUN_RUNTIME_ERROR;
    }
    sRunDepth++;
    JaiRunResult result = runLoop(baseFrameCount);
    sRunDepth--;
    return result;
}

/* ------------------------------------------------------------------ */
/* Lifecycle                                                            */
/* ------------------------------------------------------------------ */

void jaiVMResetStack(void) {
    vm.stackTop = vm.stack;
    vm.frameCount = 0;
    vm.openUpvalues = NULL;
    vm.handlers.count = 0;
    vm.defers.count = 0;
    sFinallyPending = 0;
    sRunDepth = 0;
}

/* The dunder and well-known names the VM itself dispatches on. Interning them
 * once turns every later lookup into a pointer comparison. */
static void internWellKnownNames(void) {
    vm.strInit     = jaiStringInternC("init");
    vm.strItems    = jaiStringInternC("items");
    vm.strStr      = jaiStringInternC("__str__");
    vm.strRepr     = jaiStringInternC("__repr__");
    vm.strEq       = jaiStringInternC("__eq__");
    vm.strLt       = jaiStringInternC("__lt__");
    vm.strHash     = jaiStringInternC("__hash__");
    vm.strLen      = jaiStringInternC("__len__");
    vm.strGetItem  = jaiStringInternC("__getitem__");
    vm.strSetItem  = jaiStringInternC("__setitem__");
    vm.strContains = jaiStringInternC("__contains__");
    vm.strIter     = jaiStringInternC("__iter__");
    vm.strNext     = jaiStringInternC("__next__");
    vm.strCall     = jaiStringInternC("__call__");
    vm.strAdd      = jaiStringInternC("__add__");
    vm.strSub      = jaiStringInternC("__sub__");
    vm.strMul      = jaiStringInternC("__mul__");
    vm.strDiv      = jaiStringInternC("__div__");
    vm.strMod      = jaiStringInternC("__mod__");
    vm.strPow      = jaiStringInternC("__pow__");
    vm.strNeg      = jaiStringInternC("__neg__");
    vm.strMain     = jaiStringInternC("main");
    vm.strSelf     = jaiStringInternC("self");
    vm.strMessage  = jaiStringInternC("message");
}

void jaiVMInit(void) {
    /* Flags may have been set from the command line before the VM came up, so
     * only the fields this function owns are reset. */
    vm.stack = JAI_ALLOC(Value, JAI_STACK_MAX);
    vm.frames = JAI_ALLOC(CallFrame, JAI_FRAMES_MAX);
    vm.frameCapacity = JAI_FRAMES_MAX;
    vm.frameCount = 0;
    vm.stackTop = vm.stack;
    vm.openUpvalues = NULL;

    JAI_VEC_INIT(&vm.handlers);
    JAI_VEC_INIT(&vm.defers);
    JAI_VEC_INIT(&vm.modulePath);

    vm.pendingException = NULL_VAL;
    vm.hasException = false;
    vm.mainModule = NULL;
    vm.builtins = NULL;

    vm.instructionCount = 0;
    vm.callCount = 0;
    vm.allocCount = 0;
    vm.icHits = 0;
    vm.icMisses = 0;

    sRunDepth = 0;
    sFinallyPending = 0;
    jaiInterrupted = 0;

    /* The collector has to exist before the first allocation below. */
    GCState *gc = JAI_ALLOC(GCState, 1);
    jaiGCInit(gc);
    gc->stress = vm.gcStress;
    gc->verbose = vm.debugGC;
    /* stress is one of jaiGCLimit's four inputs and this is the only place
     * outside gc.c that writes one. Without this, --gc-stress kept the
     * threshold it was initialised with and stressed 11% less. */
    jaiGCSyncLimit();

    jaiInternTableInit();
    jaiTableInit(&vm.modules);

    /* Before anything else interns: from here on every reader of the one-byte
     * ASCII table may assume a slot is populated, which is a branch each of
     * them used to carry. See jaiAsciiCharsFill. */
    jaiAsciiCharsFill();

    internWellKnownNames();

    /* The intern table is a weak root, so `builtinsName` would be swept by the
     * collection the *next* allocation triggers. Root it across that one. */
    ObjString *builtinsName = jaiStringInternC("__builtins__");
    jaiGCPushRoot(OBJ_VAL(builtinsName));
    ObjString *builtinsPath = jaiStringInternC("");
    vm.builtins = jaiModuleNew(builtinsName, builtinsPath);
    jaiGCPopRoot();
    vm.builtins->state = MOD_LOADED;

    jaiRegisterErrorClasses();
    jaiRegisterAllBuiltins();

    installInterruptHandler();
}

void jaiVMFree(void) {
    removeInterruptHandler();
    freeSavedTraceback();

    jaiVMResetStack();
    vm.pendingException = NULL_VAL;
    vm.hasException = false;
    vm.mainModule = NULL;
    vm.builtins = NULL;

    jaiTableFree(&vm.modules);
    JAI_VEC_FREE(ExcHandler, &vm.handlers);
    JAI_VEC_FREE(Value, &vm.defers);
    JAI_VEC_FREE(ObjString *, &vm.modulePath);

    vm.strInit = vm.strStr = vm.strRepr = vm.strEq = vm.strLt = NULL;
    vm.strHash = vm.strLen = vm.strGetItem = vm.strSetItem = NULL;
    vm.strContains = vm.strIter = vm.strNext = vm.strCall = NULL;
    vm.strAdd = vm.strSub = vm.strMul = vm.strDiv = vm.strMod = NULL;
    vm.strPow = vm.strNeg = vm.strMain = vm.strSelf = vm.strMessage = NULL;

    vm.cError = vm.cTypeError = vm.cValueError = vm.cNameError = NULL;
    vm.cIndexError = vm.cKeyError = vm.cAttributeError = NULL;
    vm.cArithmeticError = vm.cDivisionByZeroError = vm.cOverflowError = NULL;
    vm.cIOError = vm.cOSError = vm.cRuntimeError = vm.cRecursionError = NULL;
    vm.cStopIteration = vm.cAssertionError = vm.cImportError = NULL;
    vm.cFileNotFoundError = vm.cPermissionError = vm.cParseError = NULL;
    vm.cLookupError = NULL;

    /* Objects first (the sweep needs the intern table intact), then the
     * table that weakly referenced them. */
    GCState *gc = vm.gc;
    if (gc != NULL) {
        jaiGCFree(gc);
        JAI_FREE(GCState, gc);
    }
    vm.gc = NULL;
    /* After the sweep the 128 slots point at freed objects. */
    jaiAsciiCharsReset();
    jaiInternTableFree();

    JAI_FREE_ARRAY(Value, vm.stack, JAI_STACK_MAX);
    JAI_FREE_ARRAY(CallFrame, vm.frames, vm.frameCapacity);
    vm.stack = NULL;
    vm.stackTop = NULL;
    vm.frames = NULL;
    vm.frameCapacity = 0;
}

JaiRunResult jaiVMRunModule(ObjModule *module, ObjFunction *body) {
    if (module == NULL || body == NULL) return JAI_RUN_COMPILE_ERROR;

    jaiVMResetStack();
    jaiClearException();

    body->module = module;
    ObjClosure *closure = jaiClosureNew(body);
    module->body = closure;
    if (vm.mainModule == NULL) vm.mainModule = module;

    Value *base = vm.stackTop;
    *vm.stackTop++ = OBJ_VAL(closure);
    int window = frameWindowSize(body);
    if (base + window + 1 > vm.stack + JAI_STACK_MAX) {
        (void)jaiThrow(vm.cRuntimeError, "module body needs too many slots");
        jaiReportUncaught(vm.pendingException);
        jaiClearException();
        return JAI_RUN_RUNTIME_ERROR;
    }
    for (int i = 1; i < window; i++) *vm.stackTop++ = NULL_VAL;

    if (!pushFrame(closure, base)) {
        jaiReportUncaught(vm.pendingException);
        jaiClearException();
        return JAI_RUN_RUNTIME_ERROR;
    }
    vm.frames[vm.frameCount - 1].module = module;

    JaiRunResult result = run(0);
    if (result != JAI_RUN_OK) {
        module->state = MOD_FAILED;
        jaiReportUncaught(vm.pendingException);
        jaiClearException();
        jaiVMResetStack();
        return result;
    }

    module->state = MOD_LOADED;
    jaiVMResetStack();
    return JAI_RUN_OK;
}

/* ------------------------------------------------------------------ */
/* Debug output                                                         */
/* ------------------------------------------------------------------ */

void jaiVMPrintStack(FILE *out) {
    if (out == NULL || vm.stack == NULL) return;
    fputs("          ", out);
    for (Value *slot = vm.stack; slot < vm.stackTop; slot++) {
        fputs("[ ", out);
        jaiPrintValue(out, *slot, true);
        fputs(" ]", out);
    }
    fputc('\n', out);
}

void jaiVMPrintStats(FILE *out) {
    if (out == NULL) return;
    uint64_t lookups = vm.icHits + vm.icMisses;
    double hitRate = lookups > 0 ? (100.0 * (double)vm.icHits / (double)lookups)
                                 : 0.0;
#ifdef JAI_OPCODE_STATS
    {
        uint64_t tot = 0;
        for (int i = 0; i < OP_COUNT; i++) tot += jaiOpCounts[i];
        /* Every non-zero opcode, not a top slice: a filtered histogram cannot
         * be summed, and summing it against vm.instructionCount is the only
         * check that no dispatch path skips the census. */
        for (int i = 0; i < OP_COUNT; i++)
            if (jaiOpCounts[i] > 0)
                fprintf(out, "op %3d %-24s %10" PRIu64 "  %5.2f%%\n", i,
                        jaiOpName((OpCode)i), jaiOpCounts[i],
                        100.0 * (double)jaiOpCounts[i] / (double)tot);
    }
#endif
#ifdef JAI_PROP_STATS
    for (int i = 0; i < 32; i++)
        if (jaiPropRecv[i] > 0)
            fprintf(out, "getProperty recv type %d: %" PRIu64 "\n", i, jaiPropRecv[i]);
#endif
#ifdef JAI_ALLOC_CENSUS
    jaiAllocPrintCensus(out);
#endif
    fprintf(out, "vm: %" PRIu64 " instructions, %" PRIu64 " calls, %" PRIu64
                 " allocations\n",
            vm.instructionCount, vm.callCount, vm.allocCount);
    fprintf(out, "inline caches: %" PRIu64 " hits, %" PRIu64 " misses (%.1f%%)\n",
            vm.icHits, vm.icMisses, hitRate);
    jaiGCPrintStats(out);
}
