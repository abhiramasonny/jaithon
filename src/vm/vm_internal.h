/* vm_internal.h — the pieces of the interpreter that vm.c shares with the
 * vm_*.c files split out of it: the call/frame, exception, class, operator,
 * cache and lifecycle halves. Nothing outside src/vm includes this.
 *
 * The inline bodies below are here rather than in one .c file because runLoop
 * must keep them inlined; each one's own comment records what it costs when it
 * is not.
 */
#ifndef JAI_VM_INTERNAL_H
#define JAI_VM_INTERNAL_H

#include <signal.h>

#include "vm/vm.h"
#include "vm/gc.h"
#include "vm/object/object.h"
#include "vm/table.h"

/* Handler sentinel for PUSH_FINALLY: matches every exception like a catch-all,
 * but does not consume it. UINT32_MAX is the catch-all constant index from
 * spec §3.8, so the finally marker sits one below it. */
#define JAI_HANDLER_CATCH_ALL UINT32_MAX
#define JAI_HANDLER_FINALLY   (UINT32_MAX - 1u)

/* ------------------------------------------------------------------ */
/* Forward declarations                                                 */
/* ------------------------------------------------------------------ */

typedef enum {
    CALL_ERROR,   /* exception pending */
    CALL_DONE,    /* result already pushed; nothing to interpret */
    CALL_FRAME,   /* a new frame was pushed; resume the loop in it */
} CallOutcome;

/* Almost no frame registers a defer, and asking runFrameDefers to work that
 * out — it is the first thing it does — cost a call on every single return,
 * 4.5% of tests/bench/fib_recursive, which contains no `defer` at all. */
#define FRAME_HAS_DEFERS(f) (vm.defers.count > (f)->deferBase)

/* ------------------------------------------------------------------ */
/* Small utilities                                                      */
/* ------------------------------------------------------------------ */

static inline CallFrame *topFrame(void) {
    return vm.frameCount > 0 ? &vm.frames[vm.frameCount - 1] : NULL;
}

static inline Chunk *frameChunk(CallFrame *frame) { return &frame->closure->fn->chunk; }

/* One cache line, or NULL when the operand is out of range (a chunk that came
 * from a corrupt cache file must degrade to the slow path, never crash). */
static inline InlineCache *cacheAt(Chunk *chunk, uint16_t index) {
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

/* One shared, direct-mapped cache for call sites that have gone megamorphic.
 *
 * A site whose inline cache runs out of ways is marked IC_MEGA and then stops
 * caching ALTOGETHER: every call re-resolves the method. That is the shape a
 * trait with eight implementations produces, and it is the one shape a
 * per-site cache cannot help with, because the problem is not that the site
 * forgot -- it is that four ways cannot hold eight answers. Widening the ways
 * is not the fix either: it costs every cache in every chunk memory for a case
 * almost none of them see, and a ninth class puts the site right back where it
 * was. One table shared by every megamorphic site in the program makes the
 * cost of the ninth class the same as the cost of the fifth.
 *
 * Only a hit in `klass->methods` is cached, and `methods.version` -- which
 * JaiTable bumps on EVERY set, key or value -- is stored alongside and checked
 * on use. findMethod consults `methods` first, so a name found there can only
 * start meaning something else if that table changes, and then the version
 * says so. Nothing has to remember to invalidate this, which is the property
 * worth having: statics and trait defaults are simply not cached here and take
 * the slow path they take today.
 *
 * Whether the CALLER may call what it found is not a property of the class, so
 * `recheck` records only that a visibility test is owed and the test itself is
 * re-run per call -- the same split InlineCache::payload makes.
 *
 * Held weakly: jaiMethodCacheRemoveWhite drops any entry whose class, name or
 * method the marker did not reach, in the same phase jaiTableRemoveWhite runs
 * for the intern table. So the cache neither keeps a dead class alive nor is
 * ever left pointing at one. */
#define JAI_MEGA_WAYS 64u        /* power of two */

typedef struct {
    ObjClass  *klass;
    ObjString *name;
    Value      method;
    uint32_t   tableVersion;
    bool       recheck;
} MegaEntry;

/* ------------------------------------------------------------------ */
/* Recording an INVOKE's result kind per site and per way.
 *
 * A builtin method pushes no frame, so the invoke can read its own result off
 * the stack. An INSTANCE method cannot: invokeMethodOnStack hands back
 * CALL_FRAME for a closure method -- measured, 350 of 352 instance invokes in a
 * plain run -- and the result does not exist until that frame returns. So the
 * SITE travels with the call, and the record is made at OP_RETURN.
 *
 * ONE DEEP, on purpose. An armed invoke inside the callee overwrites this one,
 * so the outer call loses that observation. It can never record a WRONG one:
 * the return that lands is matched on both the callee's frame depth and the
 * caller's resume address, and a resume address is a pointer into one chunk's
 * code at one offset -- two different sites cannot share one. Losing
 * observations only makes the record settle more slowly, and a recursive method
 * loses nothing at all, since the recursion runs through the same site.
 *
 * Cleared in popFrameForUnwind, which is the only way a frame leaves without an
 * OP_RETURN. That is not belt-and-braces: `ic` points into a chunk kept alive by
 * the caller's frame, and after an unwind that frame is gone. */
struct JaiResultSite {
    InlineCache   *ic;
    const uint8_t *resumeIp;   /* the caller's ip: the call site's identity */
    int            depth;      /* vm.frameCount as the callee sees it */
    uint8_t        way;
};

/* Headroom above a frame's window for expression temporaries. maxSlots counts
 * the register window; nested expressions push above it. */
#define JAI_FRAME_SLACK 256

static inline bool ensureRoom(const Value *from, int slots) {
    if (vm.stack != NULL && from + slots <= vm.stack + JAI_STACK_MAX) return true;
    return jaiThrow(vm.cRuntimeError, "value stack overflow (%d slots)",
                    JAI_STACK_MAX);
}

/* ------------------------------------------------------------------ */
/* Shared state and the declarations that cross a file boundary         */
/* ------------------------------------------------------------------ */

/* vm.c */
extern int sRunDepth;
extern int sFinallyPending;
extern int sThunkFrame;
extern volatile sig_atomic_t jaiInterrupted;
JaiRunResult run(int baseFrameCount);
#ifdef JAI_OPCODE_STATS
extern uint64_t jaiOpCounts[OP_COUNT];
extern uint64_t jaiOpPairs[OP_COUNT][OP_COUNT];
extern uint8_t  jaiPrevOp;
#endif
#ifdef JAI_PROP_STATS
extern uint64_t jaiPropRecv[32];
#endif

/* vm_cache.c */
extern MegaEntry sMegaCache[JAI_MEGA_WAYS];
extern unsigned sMegaMask;
extern struct JaiResultSite sResultSite;
uint32_t builtinShapeTag(Value v);

/* vm_except.c */
void describeValue(Value v, char *buf, size_t size);
void freeSavedTraceback(void);
bool runFrameDefers(CallFrame *frame);
void popFrameForUnwind(void);
bool unwindToHandler(int base, uint32_t throwOffset);

/* vm_class.c */
bool methodPermitted(ObjClass *klass, ObjString *name, bool raise);
bool findMethod(ObjClass *klass, ObjString *name, Value *out);
int jaiEnumVariantIndex(const ObjEnum *e, const ObjString *name);
bool enumMember(ObjEnum *e, ObjString *name, Value *out);
bool moduleMember(ObjModule *m, ObjString *name, Value *out, bool *outHidden);
bool getPropertyInto(Value receiver, ObjString *name, Value *out,
                     bool raise, InlineCache *ic);
bool throwFieldKind(const FieldInfo *info, Value v);
bool valueMatchesType(Value value, Value typeConstant);
bool valueIsTest(Value subject, Value target);
const char *typeConstantName(Value typeConstant);
bool classDeclareField(ObjClass *klass, ObjString *name, uint8_t info);
Obj *classSpecInstantiate(Value spec);

/* vm_call.c */
bool valueIsCallable(Value v);
bool ensureStack(int extra);
ObjUpvalue *captureUpvalue(Value *local);
void closeUpvalues(Value *last);
int frameWindowSize(const ObjFunction *fn);
int kwRestSlotOf(const ObjFunction *fn);
bool pushFrame(ObjClosure *closure, Value *slotBase);
bool bindCallArgsSlow(ObjClosure *closure, int argc, Value *slotBase);
CallOutcome callClosure(ObjClosure *closure, int argc);
bool callNativeAt(ObjNative *native, Value *args, int count, Value *resultSlot);
CallOutcome invokeCallable(Value callable, int argc);
CallOutcome invokeMethodOnStack(Value callable, int argc);
CallOutcome callValueOnStack(int argc);
bool resolveInvokeTarget(Value receiver, ObjString *name, Value *method,
                         Value *slotZero, bool *isMethod);
int prepareKeywordCall(int posCount, ObjTuple *names, ObjDict **outKwRest);

/* vm_operator.c */
bool arithmetic(OpCode op, Value a, Value b, Value *out);
bool bitwise(OpCode op, Value a, Value b, Value *out);
bool compareOp(OpCode op, Value a, Value b, Value *out);
bool unaryNegate(Value v, Value *out);
bool indexSet(Value container, Value index, Value value);
bool sliceBounds(Value startV, Value stopV, Value stepV, bool hasStart,
                 bool hasStop, bool hasStep, int64_t length,
                 int64_t *start, int64_t *stop, int64_t *step);

/* vm_lifecycle.c */
void attributeInstruction(CallFrame *frame);
void traceInstruction(CallFrame *frame, const uint8_t *ip);

static inline unsigned megaSlot(const ObjClass *k, const ObjString *n) {
    uint64_t h = ((uint64_t)(uintptr_t)k >> 4) ^ (n->hash * 0x9E3779B97F4A7C15ull);
    h ^= h >> 29;
    return (unsigned)(h & sMegaMask);
}

JAI_INLINE void recordInvokeResult(InlineCache *ic, unsigned way, Value result) {
    ic->resultKind[way] = jaiFeedbackMerge(ic->resultKind[way],
                                           jaiFeedbackKind(result));
}

JAI_INLINE void armInvokeResult(InlineCache *ic, unsigned way,
                                const uint8_t *resumeIp) {
    ic->obsBudget--;
    sResultSite.ic       = ic;
    sResultSite.resumeIp = resumeIp;
    sResultSite.depth    = vm.frameCount + 1;
    sResultSite.way      = (uint8_t)way;
}

static inline bool bindCallArgs(ObjClosure *closure, int argc, Value *slotBase) {
    ObjFunction *fn = closure->fn;
    int arity = (int)fn->arity;

    if (JAI_LIKELY(argc == arity && fn->defaultCount == 0 &&
                   (fn->flags & (FN_VARIADIC | FN_KWREST)) == 0)) {
        int window = (int)fn->maxSlots > 1 + arity ? (int)fn->maxSlots : 1 + arity;
        if (!ensureRoom(slotBase, window + JAI_FRAME_SLACK)) return false;
        for (int i = 1 + argc; i < window; i++) slotBase[i] = NULL_VAL;
        vm.stackTop = slotBase + window;
        return true;
    }
    return bindCallArgsSlow(closure, argc, slotBase);
}

JAI_INLINE bool valuesEqualFast(Value a, Value b, bool *equal) {
    if (JAI_UNLIKELY(!IS_OBJ(a) || !IS_OBJ(b))) return false;
    Obj *ao = AS_OBJ(a), *bo = AS_OBJ(b);
    if (JAI_UNLIKELY(ao->type != OBJ_STRING || bo->type != OBJ_STRING))
        return false;
    *equal = jaiStringEquals((ObjString *)ao, (ObjString *)bo);
    return true;
}

JAI_INLINE bool indexGetFast(Value container, Value index, Value *out) {
    if (JAI_UNLIKELY(!IS_OBJ(container) || !IS_INT(index))) return false;

    Obj *o = AS_OBJ(container);
    const int64_t raw = AS_INT(index);

    if (o->type == OBJ_LIST) {
        ObjList *list = (ObjList *)o;
        int at;
        if (JAI_UNLIKELY(!jaiNormalizeIndex(raw, list->count, &at))) return false;
        *out = jaiListGet(list, at);
        return true;
    }

    if (o->type == OBJ_STRING) {
        ObjString *s = (ObjString *)o;
        if (JAI_UNLIKELY(s->scalars != s->length)) return false;
        int at;
        if (JAI_UNLIKELY(!jaiNormalizeIndex(raw, (int)s->length, &at))) return false;
        const unsigned char c = (unsigned char)s->chars[at];
        if (JAI_UNLIKELY(c >= 128)) return false;
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
            if (JAI_UNLIKELY(list->version != it->version))
                return ITER_STEP_SLOW;
            if (index >= it->limit) return ITER_STEP_DONE;

            *out = jaiListGet(list, index);
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
        *a = jaiListGet(list, 0);
        *b = jaiListGet(list, 1);
        return true;
    }
    return false;
}

JAI_INLINE PairStep iterStepPairFast(ObjIter *it, Value *a, Value *b) {
    /* `for (i, x) in xs.enumerate()` over the snapshot OP_INVOKE took: the
     * index is the first component and the element the second, and neither
     * is boxed into a tuple on the way. No version test -- the snapshot is
     * this iterator's own and nothing else can reach it. */
    if (it->kind == ITER_LIST_ENUM) {
        const int64_t index = it->index;
        if (index >= it->limit) return PAIR_STEP_DONE;
        *a = INT_VAL(index);
        *b = jaiListGet(AS_LIST(it->source), (int)index);
        it->index = index + 1;
        return PAIR_STEP_VALUE;
    }
    if (it->kind == ITER_DICT_ITEMS) {
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

#endif /* JAI_VM_INTERNAL_H */
