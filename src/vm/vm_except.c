/* vm_except.c — raising, reporting, unwinding, and defer: everything between
 * the instruction that faults and the handler that catches it.
 */
#include <inttypes.h>

#include "vm/vm_internal.h"
#include "vm/trace/trace.h"
#include "common/diag.h"
#include "runtime/runtime.h"

/* Short rendering for diagnostics. Deliberately does not call __str__: an
 * error message must never be able to raise a second exception. */
void describeValue(Value v, char *buf, size_t size) {
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

void freeSavedTraceback(void) {
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
bool runFrameDefers(CallFrame *frame) {
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
void popFrameForUnwind(void) {
    CallFrame *frame = &vm.frames[vm.frameCount - 1];
    /* This call produced no value, and the chunk sResultSite points into is
     * kept alive only by the frames being discarded here. */
    sResultSite.ic = NULL;
    (void)runFrameDefers(frame);
    closeUpvalues(frame->base);
    if (vm.handlers.count > frame->handlerBase) vm.handlers.count = frame->handlerBase;
    if (vm.defers.count > frame->deferBase) vm.defers.count = frame->deferBase;
    if (frame->closure->fn->flags & FN_TRACE) jaiTraceLeave(frame->closure->fn);
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
bool unwindToHandler(int base, uint32_t throwOffset) {
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
