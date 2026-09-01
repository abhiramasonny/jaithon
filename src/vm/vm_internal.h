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

/* Arm the record for a call about to be made from `resumeIp` in the current
 * frame. Spends one of the site's observations whether or not the return is
 * ever seen, so a site whose callee always throws still stops paying. */
JAI_INLINE void armInvokeResult(InlineCache *ic, unsigned way,
                                const uint8_t *resumeIp) {
    ic->obsBudget--;
    sResultSite.ic       = ic;
    sResultSite.resumeIp = resumeIp;
    sResultSite.depth    = vm.frameCount + 1;
    sResultSite.way      = (uint8_t)way;
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
        *out = jaiListGet(list, at);
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
        *a = jaiListGet(list, 0);
        *b = jaiListGet(list, 1);
        return true;
    }
    return false;
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

#endif /* JAI_VM_INTERNAL_H */
