/* jit_runtime.c -- the C helpers compiled code calls back into, and the state it
 * shares with the collector and the deopt path. */
#include "vm/jit/jit.h"

/* For jaiJitFieldReadFor: which builtins are one load from their receiver. */
#include "vm/jit/jit_field_read.h"
#include "vm/gc.h"
#include "vm/table_inline.h"
/* For jaiBuiltinMethod: resolving `xs.len()` to a native needs the runtime's name table. */
/* For jaiOpBranchOperandAt: says which opcodes carry a branch target. */
#include "vm/vm.h"

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "vm/jit/jit_internal.h"

#if (defined(__aarch64__) || defined(__arm64__))

/* A descriptor call pushes its roots via a C helper; a self-call (bare `bl`) does not, so callee-saved
 * registers are invisible to a collection inside the callee -- the tier refuses allocate-then-self-call for exactly this reason. */
JitCallDesc *gJitFrames;

/* An empty dict the tier hands out as the SAMPLE for a value it knows is a
 * dict but cannot see -- a declared `dict[K, V]` field read off a receiver
 * the body built itself. Arms ask the sample which arm to emit and guard
 * OBJ_DICT at run time regardless; nothing compiled ever holds this pointer.
 * Kept here, and marked below, because a compile may allocate and so collect
 * while an entry still names it. */
static ObjDict *gDictExemplar;

ObjDict *jitDictExemplar(void) {
    if (gDictExemplar == NULL) gDictExemplar = jaiDictNew();
    return gDictExemplar;
}

/* A list iterator a compiled loop finished with, kept for the next
 * OP_GET_ITER (see jitMakeIter and the exhausted arm of emitForIterBind).
 * Rooted here, so a collection never frees it under the pointer. */
ObjIter *gJitIterSpare;

/* After a VM teardown's sweep the exemplar and the spare iterator are freed
 * memory. */
void jaiJitExemplarsReset(void) {
    gDictExemplar = NULL;
    gJitIterSpare = NULL;
}

/* JAITHON_JIT_FIELD_DICT=0 stops a declared `dict` field being predicted when
 * there is no live receiver to read it off, which is the decline it was. */
bool jitFieldDict(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_FIELD_DICT");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

/* JAITHON_JIT_ITER_SOFT=0 makes a list or dict loop whose container had no
 * element to sample decline the whole function again, rather than leave the
 * loop to the interpreter and compile the rest. The sampled container is
 * whatever the call that crossed the threshold happened to pass: a tree walk
 * that recurses through leaves samples an empty child list nearly every time,
 * and declined five attempts out of five for it. */
bool jitIterSoft(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_ITER_SOFT");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

/* JAITHON_JIT_ITER_EMPTY_SKIP=0 drops the empty-list exit in front of a soft
 * iterate refusal (see OP_GET_ITER), so every entry to such a loop deopts,
 * empty or not. */
bool jitIterEmptySkip(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_ITER_EMPTY_SKIP");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

/* JAITHON_JIT_PAIR_INST=0 binds a dict-items loop's instance components as
 * bare SLOT_OBJ again, so a loop variable already held as an instance clashes
 * with them. */
bool jitPairInstance(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_PAIR_INST");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

/* JAITHON_JIT_OSR_SELF_GLOBAL=0 sends a compiled loop's call to its own
 * function back down the self-call arm, which cannot serve a loop form and
 * refuses it. */
bool jitOsrSelfGlobal(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_OSR_SELF_GLOBAL");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}
void jaiJitMarkFrames(void) {
    for (JitCallDesc *f = gJitFrames; f != NULL; f = f->link) {
        for (int64_t i = 0; i < f->nroots; i++) jaiGCMarkValue(f->roots[i]);
    }
    if (gDictExemplar != NULL) jaiGCMarkObject((Obj *)gDictExemplar);
    if (gJitIterSpare != NULL) jaiGCMarkValue(OBJ_VAL((Obj *)gJitIterSpare));
}

/* JAITHON_JIT_ITER_RECYCLE: a compiled for-in over a list hands its iterator
 * back on exhaustion and the next one reuses it. */
bool jitIterRecycle(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_ITER_RECYCLE");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

/* Roots go in as a RANGE (jaiGCPushRootRange/Pop), not copied one at a time -- copying individually
 * was O(roots) per call-out; a bulk jaiGCPushRoots variant was tried and reverted, since it still copied every value. Returns 0 on success, 1 with an exception pending. */
/* Not a deopt: overflow raises directly, since the interpreter would also throw here. A bail is only
 * sound before any write; raising stays sound after a field store has already happened. */
void jitThrowOverflow(int64_t which) {
    static const char *ops[3]  = { "+",  "-",  "*"  };
    static const char *wrap[3] = { "+%", "-%", "*%" };
    int i = (which >= 0 && which < 3) ? (int)which : 0;
    (void)jaiThrow(vm.cOverflowError,
                   "integer overflow in '%s'; use '%s' to wrap", ops[i], wrap[i]);
}

JitDeoptRecord gDeopt;

/* Cached: getenv is O(environ) and this sits on the hot deopt path, so leaving it uncached let ambient
 * shell-exported variable count perturb benchmarks (sort_merge moved 70ms->100ms on padding alone). Same idiom as jaiJitEnabled. */
static bool jitReconTrace(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAI_JIT_RECON");
        cached = (v != NULL && v[0] != '\0') ? 1 : 0;
    }
    return cached != 0;
}

bool jaiJitApplyDeopt(ObjClosure *closure, Value *slotBase) {
    ObjFunction *fn = closure->fn;
    if (gDeopt.ip < 0 || gDeopt.ip >= fn->chunk.count) return false;

    for (int64_t i = 0; i < gDeopt.nlocals; i++) {
        if (i < 64 && (((uint64_t)gDeopt.skipLocals >> i) & 1u) != 0) continue;
        slotBase[gDeopt.base + i] = gDeopt.locals[i];
    }
    /* The operand stack sits above the frame's window, which bindCallArgs has
     * already set vm.stackTop to. */
    for (int64_t i = 0; i < gDeopt.nstack; i++) {
        *vm.stackTop++ = gDeopt.stack[i];
    }
    CallFrame *frame = &vm.frames[vm.frameCount - 1];
    frame->ip = fn->chunk.code + gDeopt.ip;
    if (jitReconTrace()) {
        fprintf(stderr, "[deopt] %s ip=%lld base=%lld nlocals=%lld nstack=%lld\n",
                jitFnLabel(fn), (long long)gDeopt.ip,
                (long long)gDeopt.base, (long long)gDeopt.nlocals,
                (long long)gDeopt.nstack);
    }
    return true;
}

/* Receiver is args[0], exactly where callNativeAt wants it, so no bound wrapper is made. Roots as jitCallOut does, since push and its kin allocate. */
int jitInvokeMethod(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    bool ok = jaiCallMethodWithReceiver(d->callee, d->args, (int)d->argc,
                                        &d->result);
    jaiGCPopRootRange();
    return ok ? 0 : 1;
}

/* Receiver whose class the model could not pin. The method NAME travels in the
 * descriptor's callee slot -- there is no method Value to put there -- and the
 * resolve happens per call against the shared megamorphic table. */
int jitInvokeByName(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    bool ok = jaiInvokeMethodByName(AS_STRING(d->callee), d->args,
                                    (int)d->argc, &d->result);
    jaiGCPopRootRange();
    return ok ? 0 : 1;
}

/* The same, at a site whose loop form may be re-compiled for more ways
 * (jitOsrPicShort): `aux` carries the site's InlineCache, and a call that
 * succeeds teaches it the receiver's class. */
int jitInvokeByNameLearn(JitCallDesc *d) {
    Value receiver = d->args[0];
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    bool ok = jaiInvokeMethodByName(AS_STRING(d->callee), d->args,
                                    (int)d->argc, &d->result);
    jaiGCPopRootRange();
    if (!ok) return 1;
    jaiInvokeCacheLearn((InlineCache *)(uintptr_t)d->aux, receiver,
                        AS_STRING(d->callee));
    return 0;
}

int jitInvokeNative(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    bool ok = jaiInvokeNativeWithReceiver(d->callee, d->args, (int)d->argc,
                                          &d->result);
    jaiGCPopRootRange();
    return ok ? 0 : 1;
}

/* Allocates (jaiListNew), so roots go down first as for any call out of compiled code. */
int jitBuildList(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    ObjList *list = jaiListNew((int)d->argc);
    for (int64_t i = 0; i < d->argc; i++) jaiListPut(list, (int)i, d->args[i]);
    list->count = (int)d->argc;
    d->result = OBJ_VAL(list);
    jaiGCPopRootRange();
    return 0;
}

/* `x in c`. The needle is args[0] and the container args[1], matching the
 * operand order the interpreter peeks. The answer is stored as a Value so the
 * one-byte BOOL_VAL member is the thing the caller's LdrByte reads -- see the
 * SLOT_BOOL result convention in emitDescriptorStatus.
 *
 * Roots go down because a container's __contains__ can run Jaithon code and
 * therefore collect. Returns 1 having thrown. */
int jitContains(JitCallDesc *d) {
    bool contains = false;
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    bool ok = jaiContainsOp(d->args[1], d->args[0], &contains);
    jaiGCPopRootRange();
    if (!ok) return 1;
    d->result = BOOL_VAL(contains);
    return 0;
}

int jitNotContains(JitCallDesc *d) {
    if (jitContains(d) != 0) return 1;
    d->result = BOOL_VAL(!AS_BOOL(d->result));
    return 0;
}

/* `{a: 1, b: 2}`. Keys and values alternate in the operand list, which is the
 * order the interpreter reads them in, so the args array is used as-is.
 *
 * jaiDictSet hashes the key and can raise on an unhashable one, so the root
 * range goes down and a raise comes back as 1 for the descriptor's threw
 * branch. */
int jitBuildDict(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    ObjDict *dict = jaiDictNew();
    jaiGCPushRoot(OBJ_VAL(dict));
    for (int64_t i = 0; i + 1 < d->argc; i += 2) {
        (void)jaiDictSet(dict, d->args[i], d->args[i + 1]);
        if (vm.hasException) {
            jaiGCPopRoot();
            jaiGCPopRootRange();
            return 1;
        }
    }
    jaiGCPopRoot();
    jaiGCPopRootRange();
    d->result = OBJ_VAL(dict);
    return 0;
}

/* `{a, b}`. Same shape, one operand per element. */
int jitBuildSet(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    ObjSet *set = jaiSetNew();
    jaiGCPushRoot(OBJ_VAL(set));
    for (int64_t i = 0; i < d->argc; i++) {
        (void)jaiSetAdd(set, d->args[i]);
        if (vm.hasException) {
            jaiGCPopRoot();
            jaiGCPopRootRange();
            return 1;
        }
    }
    jaiGCPopRoot();
    jaiGCPopRootRange();
    d->result = OBJ_VAL(set);
    return 0;
}

int jitBuildTuple(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    ObjTuple *tuple = jaiTupleNew(d->args, (int)d->argc);
    jaiGCPopRootRange();
    d->result = OBJ_VAL(tuple);
    return 0;
}

/* `a == b` on two heap objects the tier can say nothing else about.
 *
 * The interpreter's own equality, so an `__eq__` behaves and a comparison that
 * raises raises the same message. It can therefore allocate and throw: roots go
 * down, and 1 comes back for the descriptor's threw branch.
 *
 * This is the LAST arm in each equality chain on purpose. Every arm above it
 * answers without a call -- interned string pointers, a folded enum unit, two
 * registers -- and each is strictly better where it applies. */
int jitValuesEqual(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    bool equal = jaiValuesEqual(d->args[0], d->args[1]);
    jaiGCPopRootRange();
    if (vm.hasException) return 1;
    d->result = BOOL_VAL(equal);
    return 0;
}

bool jitObjEquality(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_OBJ_EQ");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

/* `a + b` on two strings. Both operands are guarded for OBJ_STRING at the call
 * site, so this is jaiStringConcat and nothing else -- the general arithmetic()
 * fallback would have to be answered with a tag test on the result, and there
 * is no shape of `+` other than this one that reaches here.
 *
 * Allocates (and appends into an existing ObjStrBuf), so roots go down first as
 * for any call out of compiled code. NULL back means the concatenation
 * overflowed UINT32_MAX and already threw. */
int jitStringConcat(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    ObjString *s = jaiStringConcat(AS_STRING(d->args[0]), AS_STRING(d->args[1]));
    jaiGCPopRootRange();
    if (s == NULL) return 1;
    d->result = OBJ_VAL(s);
    return 0;
}

/* args: start, stop, inclusive-flag. Allocates twice (range + iterator), so roots go down first as usual. */
int jitMakeRangeIter(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    ObjRange *r = jaiRangeNew(AS_INT(d->args[0]), AS_INT(d->args[1]), 1,
                              AS_INT(d->args[2]) != 0);
    jaiGCPushRoot(OBJ_VAL(r));
    ObjIter *it = jaiIterNew(ITER_RANGE, OBJ_VAL(r));
    jaiGCPopRoot();
    d->result = OBJ_VAL(it);
    jaiGCPopRootRange();
    return 0;
}

/* The iterator for a `for` whose source the caller has already proved. A string
 * yields one-CHARACTER strings and a list its elements; both are the same
 * ObjIter the interpreter builds, so nothing here is a second implementation.
 *
 * The fallthrough RAISES rather than returning 1 quietly. Returning 1 without
 * setting an exception is what the caller reads as "the callee threw", and the
 * VM then reports `internal error: failed operation raised nothing` -- 104 test
 * failures with no useful message, which is how this arm's first draft
 * announced that it had reached here with a string. */
int jitMakeIter(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    Value src = d->args[0];
    IterKind k;
    if (IS_LIST(src)) {
        k = ITER_LIST;
    } else if (IS_STRING(src)) {
        k = ITER_STRING;
    } else {
        jaiGCPopRootRange();
        (void)jaiThrow(vm.cTypeError, "'%s' is not iterable",
                       jaiTypeNameStatic(src));
        return 1;
    }
    ObjIter *it = NULL;
    if (k == ITER_LIST && gJitIterSpare != NULL) {
        /* jaiIterNew's ITER_LIST arm, on an object nothing else holds. */
        ObjList *list = AS_LIST(src);
        it = gJitIterSpare;
        gJitIterSpare = NULL;
        it->kind    = ITER_LIST;
        it->source  = src;
        it->index   = 0;
        it->limit   = list->count;
        it->version = list->version;
    } else {
        it = jaiIterNew(k, src);
    }
    d->result = OBJ_VAL(it);
    jaiGCPopRootRange();
    return 0;
}

/* OP_GET_ITER_ITEMS' dict case: the lazy view `for (k, v) in d.items()` walks,
 * built once per loop entry rather than the N 2-tuples the eager `items()`
 * materialises. Allocates, so roots go down first as for any call out of
 * compiled code. The emitted guard has already proved the receiver is a dict;
 * the test here is the same belt-and-braces jitMakeIter carries. */
int jitMakeItemsIter(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    Value src = d->args[0];
    ObjIter *it = IS_DICT(src) ? jaiIterNew(ITER_DICT_ITEMS, src) : NULL;
    if (it != NULL) d->result = OBJ_VAL(it);
    jaiGCPopRootRange();
    return it != NULL ? 0 : 1;
}

/* OP_GET_ITER on a dict, for compiled code: the ITER_DICT_KEYS iterator
 * `for k in d` walks, which is what jaiGetIter builds for a dict. The emitted
 * guard has already proved the receiver is a dict; the test here is the same
 * belt-and-braces jitMakeItemsIter carries. */
int jitMakeDictKeysIter(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    Value src = d->args[0];
    ObjIter *it = IS_DICT(src) ? jaiIterNew(ITER_DICT_KEYS, src) : NULL;
    if (it != NULL) d->result = OBJ_VAL(it);
    jaiGCPopRootRange();
    return it != NULL ? 0 : 1;
}

/* OP_INVOKE's lazy `xs.enumerate()` head, for compiled code: the same
 * ITER_LIST_ENUM snapshot the interpreter builds at that site (vm.c), so the
 * pair loop that follows walks the same thing under either tier. The emitted
 * arm only reaches here with a SLOT_LIST receiver; the test is the same
 * belt-and-braces jitMakeItemsIter carries, and it RAISES on the way out for
 * the reason jitMakeIter's comment gives. */
int jitMakeEnumIter(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    Value src = d->args[0];
    if (!IS_LIST(src)) {
        jaiGCPopRootRange();
        (void)jaiThrow(vm.cTypeError, "'%s' object has no method 'enumerate'",
                       jaiTypeNameStatic(src));
        return 1;
    }
    ObjIter *it = jaiIterNewListEnum(AS_LIST(src));
    d->result = OBJ_VAL(it);
    jaiGCPopRootRange();
    return 0;
}

/* f-string: the interpreter's parts, read off the operand stack, land here contiguously in args[].
 * Builtin path only -- compiler checks at compile time that the module hasn't rebound `str`; a rebind retires this form. */
int jitFormat(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    ObjString *formatted = jaiValueFormat(d->args, (int)d->argc);
    jaiGCPopRootRange();
    if (formatted == NULL) return 1;
    d->result = OBJ_VAL(formatted);
    return 0;
}

/* A list whose storage is not settled yet: a push that grows it past
 * JAI_LIST_SHAPE_AT may still unbox it (jaiListShapeOnGrow, from jaiListPush
 * or from jitListGrow below), so a loop form must not pin the BOXED it has
 * now. Capacity only grows without a call, so a list past that size at a
 * form's entry stays past it for the whole region, and one under it is never
 * pinned -- not at compile time (compileOsrOnce) and not at entry
 * (osrFormStorageFits). Conservative about content on purpose: a compiled
 * store can turn a list of strings into a list of ints before the push. */
bool jitListShapeable(Value v) {
    if (!IS_LIST(v)) return false;
    ObjList *l = AS_LIST(v);
    return l->capacity <= JAI_LIST_SHAPE_AT &&
           l->stg == LIST_STORE_BOXED && l->elemKind == FIELD_KIND_ANY &&
           jaiListShapeGrownOn();
}

/* No descriptor/roots: growing a list cannot collect -- jaiListReserve->jaiRealloc never triggers the
 * marker (gc.c: collections only happen at jaiGCMaybeCollect safepoints). Returns 1 if it raised (list past INT32_MAX). */
/* The compiled half of jaiListPush's shaping at growth: the same test on the
 * same list, so an untyped list comes out the same width whichever tier
 * pushed it past its first eight. Only for a site that dispatches on storage
 * after the grow -- a site emitted at one pinned storage would go on storing
 * at that width. jaiListShapeOnGrow's allocation is the one jaiListReserve
 * would have made, and collects no more than it. */
int jitListGrow(ObjList *list, uint64_t tag, int64_t payload,
                int64_t mayShape) {
    Value pending;
    pending.type = (ValueType)tag;
    pending.as.integer = payload;
    if (mayShape != 0 && jaiListShapeOnGrow(list, pending)) return 0;
    jaiGCPushRoot(OBJ_VAL(list));
    jaiGCPushRoot(pending);
    if (list->capacity > INT32_MAX / 2) {
        jaiGCPopRoots(2);
        (void)jaiThrow(vm.cRuntimeError,
                       "list cannot grow beyond %d items", INT32_MAX);
        return 1;
    }
    jaiListReserve(list, JAI_GROW_CAP(list->capacity));
    jaiGCPopRoots(2);
    return 0;
}

/* Safe because jaiGCWanted()==false is gc.c's own proof no collection can happen here (collections begin
 * only in jaiGCMaybeCollect), and the object is fully built before being linked in. NULL means "declined": caller falls back to the descriptor path, which roots and may collect. */
static JAI_NOINLINE ObjInstance *jitInstanceAllocSlow(ObjClass *cls);

/* The page-space fast path, written so it needs no frame: everything that
 * could call out -- a refill, the bins, the slab -- is in the slow half, which
 * this tail-calls. The full-word stores matter too: every byte of the block is
 * written, `next` and the padding included, so no store has to wait for the
 * block's old contents to arrive before it can merge.
 *
 * `zero` is false only for jitInstanceAllocBare, whose caller stores every
 * field before anything else runs. */
JAI_INLINE ObjInstance *instanceAllocFast(ObjClass *cls, bool zero) {
    /* No jaiGCWanted() test and no jaiHeapBytes charge: the refill that handed
     * out this word made both (gc.h, jaiPageNew), and the slow half below
     * makes the test before it allocates anything else. */
    const unsigned count = cls->fieldCount;
    const unsigned c = 2u + count;   /* grains: a 32-byte header, 16 a field */
    if (JAI_LIKELY(c <= JAI_SMALL_CLASSES)) {
        JaiPageCursor *pc = &jaiPageCursor[c];
        uint64_t m = pc->freeMask;
        if (JAI_LIKELY(m != 0)) {
            pc->freeMask = m & (m - 1);
            uint64_t *w = (uint64_t *)(void *)(pc->wordBase +
                                               ((size_t)__builtin_ctzll(m) << 4));
            _Static_assert(offsetof(Obj, type) == 0 && offsetof(Obj, isMarked) == 4 &&
                           offsetof(Obj, subFlag) == 5 && offsetof(Obj, subFlag2) == 6 &&
                           offsetof(Obj, next) == 8,
                           "the header is written as two words");
            _Static_assert(offsetof(ObjInstance, klass) == 16 &&
                           offsetof(ObjInstance, fieldCount) == 24 &&
                           offsetof(ObjInstance, fields) == 32,
                           "an instance is written as words");
            _Static_assert(VAL_NULL == 0, "NULL_VAL is all zero bits");
            w[0] = (uint64_t)OBJ_INSTANCE | pc->epochHi;
            w[1] = 0;
            w[2] = (uint64_t)(uintptr_t)cls;
            w[3] = count;
            if (zero) {
                /* The empty asm keeps this a loop of paired stores: as a plain
                 * loop it became a call to bzero, and the call needed a frame. */
                uint64_t *f = w + 4;
                for (unsigned i = 0; i < count; i++) {
                    __asm__("" : "+r"(f));
                    f[0] = 0;
                    f[1] = 0;
                    f += 2;
                }
            }
            return (ObjInstance *)(void *)w;   /* counted by the refill */
        }
    }
    return jitInstanceAllocSlow(cls);
}

ObjInstance *jitInstanceAlloc(ObjClass *cls) {
    return instanceAllocFast(cls, true);
}

/* For a caller that writes every field, whole Values, before anything else can
 * run: zeroing them first was two paired stores a field on alloc_churn's hot
 * path for nothing (-3% cycles, -60M instructions without it). The slow half
 * still zeroes, which costs nothing it did not cost before. */
ObjInstance *jitInstanceAllocBare(ObjClass *cls) {
    return instanceAllocFast(cls, false);
}

static JAI_NOINLINE ObjInstance *jitInstanceAllocSlow(ObjClass *cls) {
    if (JAI_UNLIKELY(jaiGCWanted())) return NULL;

    GCState *g = jaiGCActive;
    if (JAI_UNLIKELY(g == NULL || cls == NULL)) return NULL;

    const uint16_t count = cls->fieldCount;
    const size_t size = sizeof(ObjInstance) + sizeof(Value) * (size_t)count;
    if (JAI_UNLIKELY(!jaiSmallServes(size))) return NULL;

    /* The page space first (gc.h): no free-list link to load out of the block,
     * and nothing to link into GCState.objects. */
    const unsigned c = (unsigned)((size + (JAI_SMALL_GRAIN - 1u)) >> 4);
    ObjInstance *inst = NULL;
    if (JAI_LIKELY(jaiPageKind[OBJ_INSTANCE]))
        inst = (ObjInstance *)jaiPageNew(c);
    const bool paged = inst != NULL;
    if (!paged) inst = (ObjInstance *)jaiSmallNew(size);
    Obj *obj = (Obj *)inst;

    obj->type = OBJ_INSTANCE;
    obj->isMarked = jaiGCEpoch;
    obj->subFlag = false;
    obj->subFlag2 = false;

    inst->klass = cls;
    inst->fieldCount = count;
    /* Zeroes every field, not just the ones about to be overwritten: the marker reads all `count` of them,
 * and an unwritten field is whatever the last occupant of this bin left behind. */
    for (uint16_t i = 0; i < count; ++i) inst->fields[i] = NULL_VAL;

    /* Linked last: a mid-collection isMarked state can't arise here, since a collection in progress makes jaiGCLimit zero and the first line above would already have declined. */
    if (!paged) {
        obj->next = g->objects;
        g->objects = obj;
        vm.allocCount++;   /* a paged block was counted by its refill */
    }

    return inst;
}

/* JAITHON_JIT_ITER_ALLOC=0 builds every loop iterator through the jitMakeIter
 * descriptor again. */
bool jitIterAllocOn(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_ITER_ALLOC");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

/* The iterator `for x in xs` / `for c in text` opens with, made the way
 * jitInstanceAlloc makes an instance: a leaf that cannot collect, declining
 * (NULL) whenever a collection is wanted, so the caller falls back to the
 * jitMakeIter descriptor and its roots. Every field is the one jaiIterNew
 * writes for ITER_LIST and ITER_STRING -- kind, source, the limit sampled
 * now, the list's version -- and the index starts at zero. A loop entered
 * per call over a short word or a three-element list paid the descriptor's
 * stores and root push for 40 bytes; that was most of what such a call cost
 * over the same loop written with an index. */
ObjIter *jitIterAlloc(Obj *source) {
#ifdef JAI_ALLOC_CENSUS
    (void)source;
    return NULL;   /* keep the census counting every iterator */
#else
    if (JAI_UNLIKELY(jaiGCWanted())) return NULL;
    GCState *g = jaiGCActive;
    if (JAI_UNLIKELY(g == NULL || source == NULL)) return NULL;
    if (JAI_UNLIKELY(!jaiSmallServes(sizeof(ObjIter)))) return NULL;

    IterKind kind;
    int64_t limit;
    uint32_t version = 0;
    if (source->type == OBJ_LIST) {
        kind = ITER_LIST;
        limit = ((ObjList *)source)->count;
        version = ((ObjList *)source)->version;
    } else if (source->type == OBJ_STRING) {
        kind = ITER_STRING;
        limit = (int64_t)((ObjString *)source)->length;
    } else {
        return NULL;
    }

    /* Through the allocator rather than a block linked onto GCState.objects
     * by hand: with the page space on, small objects live in pages and are
     * never on that list, and a block on both would be freed twice. It cannot
     * collect here -- jaiGCWanted() was false above, and a page refill only
     * charges bytes. */
    ObjIter *it = (ObjIter *)jaiAllocateObject(sizeof(ObjIter), OBJ_ITER);
    it->kind = kind;
    it->source = OBJ_VAL(source);
    it->index = 0;
    it->limit = limit;
    it->version = version;
    return it;
#endif
}

int jitNewInstance(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    ObjInstance *inst = jaiInstanceNew((ObjClass *)(uintptr_t)AS_OBJ(d->callee));
    d->result = OBJ_VAL(inst);
    jaiGCPopRootRange();
    return 0;
}

int jitGetSlice(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    uint8_t flags = (uint8_t)d->aux;
    bool hasStart = (flags & 1) != 0, hasStop = (flags & 2) != 0,
         hasStep = (flags & 4) != 0;
    int at = 1;
    Value startV = hasStart ? d->args[at++] : NULL_VAL;
    Value stopV  = hasStop  ? d->args[at++] : NULL_VAL;
    Value stepV  = hasStep  ? d->args[at++] : NULL_VAL;
    bool ok = jaiSliceGet(d->args[0], startV, stopV, stepV,
                          hasStart, hasStop, hasStep, &d->result);
    jaiGCPopRootRange();
    return ok ? 0 : 1;
}

/* Dict read. Calls the interpreter's own OP_GET_INDEX so a missing key raises
 * the KeyError it would have raised, with the message it would have used --
 * naming the key -- rather than a second spelling of the same error that has to
 * be kept in step with it. Pure apart from that throw, but it can allocate the
 * exception, so the roots go down first. */
int jitGetIndexDict(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    bool ok = jaiIndexGet(d->args[0], d->args[1], &d->result);
    jaiGCPopRootRange();
    return ok ? 0 : 1;
}

/* Dict store: unlike a list store there's no offset to normalise, it's a table probe either way --
 * this only saves the dispatch and indexSet's type ladder, not the probe itself. */
int jitSetIndexDict(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    (void)jaiDictSet(AS_DICT(d->args[0]), d->args[1], d->args[2]);
    jaiGCPopRootRange();
    return vm.hasException ? 1 : 0;
}

/* JAITHON_JIT_DICT_PROBE=0 refuses a dict the model knows only by its
 * predicted type (stackObjType), as the tier did before, for a one-binary
 * A/B: an invoke on one, a store into one, `in` against one. */
bool jitDictProbeOn(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_DICT_PROBE");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

/* The exemplar a predicted dict is compiled against: empty, made once, and a
 * permanent root, so a model entry may hold it for as long as any compile
 * runs. Nothing writes to it -- compiled code guards the real receiver's type
 * and works on that. */
ObjDict *jitDictProbe(void) {
    static ObjDict *probe;
    if (probe == NULL) {
        probe = jaiDictNew();
        jaiGCAddPermanentRoot(OBJ_VAL((Obj *)probe));
    }
    return probe;
}

/* JAITHON_JIT_DICT_LEAF=0 sends every dict read and store back through its
 * descriptor call, for a one-binary A/B of the leaves below. */
bool jitDictLeaf(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_DICT_LEAF");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

/* `d.get(k)`, `d.get(k, default)` and `d[k]` with a string key, called as a
 * LEAF: no descriptor, no root range, no native dispatch. jaiTableFindStr and
 * its quick form run no user code, allocate nothing and cannot raise, so nothing here can
 * collect and the operands need no rooting -- the same argument
 * jitInstanceAlloc and jaiStringOrder rest on.
 *
 * What it replaces is not the probe but everything around it: the descriptor's
 * stores and root fill, a root-range push and pop, and
 * jitInvokeNative -> callNativeAt -> dictGet -> jaiTableGet, each re-checking
 * what the guards in front of this call already proved. That was ~200
 * instructions around a ~20-instruction probe.
 *
 * Returns 0 with `*result` written, and 1 -- having written nothing -- when
 * only the descriptor path can answer: a stored key whose hash matches and
 * which is not a string, or an absent key when `defTag` is
 * JIT_DICT_ABSENT_SLOW (`d[k]`, whose miss raises the interpreter's own
 * KeyError). An absent key otherwise answers the default the caller passed as
 * a tag and payload. */
JAI_INLINE int64_t dictGetAnswer(JaiEntry *e, Value *result, uint64_t defTag,
                                  int64_t defPayload) {
    if (e != NULL) {
        *result = e->value;
        return 0;
    }
    if (defTag == JIT_DICT_ABSENT_SLOW) return 1;
    Value v;
    v.type = (ValueType)defTag;
    v.as.integer = defPayload;
    *result = v;
    return 0;
}

/* What jaiTableFindStrQuick could not settle -- a lazy hash, a byte compare --
 * through the full probe. Its own function, reached by a tail call, so that
 * the fast path above it has no call in it and so no frame. */
static JAI_NOINLINE int64_t dictGetStrSlow(ObjDict *d, ObjString *key,
                                           Value *result, uint64_t defTag,
                                           int64_t defPayload) {
    JaiEntry *e = jaiTableFindStr(&d->table, key);
    if (JAI_UNLIKELY(e == JAI_TABLE_SLOW)) return 1;
    return dictGetAnswer(e, result, defTag, defPayload);
}

int64_t jitDictGetStr(ObjDict *d, ObjString *key, Value *result,
                      uint64_t defTag, int64_t defPayload) {
    JaiEntry *e = jaiTableFindStrQuick(&d->table, key);
    if (JAI_UNLIKELY(e == JAI_TABLE_SLOW)) {
        return dictGetStrSlow(d, key, result, defTag, defPayload);
    }
    return dictGetAnswer(e, result, defTag, defPayload);
}

/* `d[k] = v` with a string key, as a leaf for the reason jitDictGetStr gives.
 * A typed dict checks its kinds with jaiKindAccepts, the pure half of what
 * jaiDictSet asks jaiCheckKind; a value the dict refuses goes back to the
 * descriptor path, which raises. An insert can grow the table, but only
 * through JAI_ALLOC, which never collects (collections begin only in
 * jaiGCMaybeCollect). Returns 0 stored, 1 untouched.
 *
 * Split the same way as jitDictGetStr: the update of a key already present,
 * which is what a counting loop does on every iteration but the first, is the
 * call-free fast path, and everything else -- a typed dict, an insert, a probe
 * the quick form could not settle -- is one tail call away. */
static JAI_NOINLINE int64_t dictSetStrSlow(ObjDict *d, ObjString *key,
                                           uint64_t tag, int64_t payload) {
    Value v;
    v.type = (ValueType)tag;
    v.as.integer = payload;
    if (JAI_UNLIKELY(d->keyKind != FIELD_KIND_ANY ||
                     d->valKind != FIELD_KIND_ANY) &&
        (!jaiKindAccepts(d->keyKind, OBJ_VAL(key)) ||
         !jaiKindAccepts(d->valKind, v))) {
        return 1;
    }
    JaiEntry *e = jaiTableFindStr(&d->table, key);
    if (JAI_UNLIKELY(e == JAI_TABLE_SLOW)) return 1;
    if (e != NULL) {
        /* insertAt's update half: the value and the version, nothing else. */
        e->value = v;
        ++d->table.version;
        return 0;
    }
    /* Through jaiStringHash, never `key->hash`: an empty table answers NULL
     * before the probe has forced the lazy hash, and a long run-time string
     * still holds zero there -- filing it under 0 made the next lookup of an
     * equal key miss and insert a second copy. */
    (void)jaiTableSetHashed(&d->table, OBJ_VAL(key), jaiStringHash(key), v);
    return 0;
}

int64_t jitDictSetStr(ObjDict *d, ObjString *key, uint64_t tag,
                      int64_t payload) {
    if (JAI_LIKELY(d->keyKind == FIELD_KIND_ANY &&
                   d->valKind == FIELD_KIND_ANY)) {
        JaiEntry *e = jaiTableFindStrQuick(&d->table, key);
        if (JAI_LIKELY(e != NULL && e != JAI_TABLE_SLOW)) {
            e->value.type = (ValueType)tag;
            e->value.as.integer = payload;
            ++d->table.version;
            return 0;
        }
    }
    return dictSetStrSlow(d, key, tag, payload);
}

/* The insert half of jitDictAddStr, and its typed-dict check, out of line so
 * the update of a key already present -- every iteration of a counting loop
 * but the first for each key -- is a leaf with no frame. */
static JAI_NOINLINE int64_t dictAddStrSlow(ObjDict *d, ObjString *key,
                                           int64_t defPayload,
                                           int64_t addend,
                                           int64_t absentSlow) {
    if (d->keyKind != FIELD_KIND_ANY || d->valKind != FIELD_KIND_ANY) {
        if (!jaiKindAccepts(d->keyKind, OBJ_VAL(key)) ||
            !jaiKindAccepts(d->valKind, INT_VAL(0))) {
            return 1;
        }
    }
    JaiEntry *e = jaiTableFindStr(&d->table, key);
    if (JAI_UNLIKELY(e == JAI_TABLE_SLOW)) return 1;
    int64_t base = defPayload;
    if (e != NULL) {
        if (e->value.type != VAL_INT) return 1;
        base = e->value.as.integer;
    } else if (absentSlow != 0) {
        return 1;
    }
    int64_t sum;
    if (__builtin_add_overflow(base, addend, &sum)) return 1;
    if (e != NULL) {
        e->value = INT_VAL(sum);
        ++d->table.version;
        return 0;
    }
    (void)jaiTableSetHashed(&d->table, OBJ_VAL(key), jaiStringHash(key),
                            INT_VAL(sum));
    return 0;
}

/* `d[k] = d.get(k, n) + c` -- the counting idiom, `counts[w] =
 * counts.get(w, 0) + 1` -- in one leaf, for an int default and an int
 * constant step: the entry's int value (or the default, for an absent key)
 * plus the step, stored back. One probe where the unfused form makes two, and
 * one call where it makes two, with the read, the add and the store between
 * them done here.
 *
 * `d[k] += c` is the same leaf with `absentSlow` set: there is no default, and
 * an absent key is the KeyError the unfused read raises.
 *
 * Returns 0 done, and 1 having written nothing whenever the unfused sequence
 * would do something else: a container that is not a dict, a key that is not
 * a string, a value that is not an int, an add that overflows (which raises),
 * a typed dict that refuses, a probe the quick form cannot settle. The caller
 * then runs that sequence.
 * Leaf-safe for the reasons jitDictSetStr gives: nothing here runs user code,
 * raises, or allocates an object. */
int64_t jitDictAddStr(Obj *dictObj, Obj *keyObj, int64_t defPayload,
                      int64_t addend, int64_t absentSlow) {
    if (JAI_UNLIKELY(dictObj->type != OBJ_DICT ||
                     keyObj->type != OBJ_STRING)) {
        return 1;
    }
    ObjDict *d = (ObjDict *)dictObj;
    ObjString *key = (ObjString *)keyObj;
    if (JAI_LIKELY(d->keyKind == FIELD_KIND_ANY &&
                   d->valKind == FIELD_KIND_ANY)) {
        JaiEntry *e = jaiTableFindStrQuick(&d->table, key);
        if (JAI_LIKELY(e != NULL && e != JAI_TABLE_SLOW &&
                       e->value.type == VAL_INT)) {
            int64_t sum;
            if (__builtin_add_overflow(e->value.as.integer, addend, &sum)) {
                return 1;
            }
            e->value.as.integer = sum;
            ++d->table.version;
            return 0;
        }
    }
    return dictAddStrSlow(d, key, defPayload, addend, absentSlow);
}

/* The int-keyed dict leaves: jitDictGetStr, jitDictSetStr, jitDictHasStr and
 * jitDictAddStr for a key the compiled code holds as an int, with the same
 * contracts. Every probe is jaiTableFindIntQuick, and whatever it cannot
 * settle -- a hash-equal key of another kind, which only the general path's
 * equality may judge -- is the descriptor call, never a guess. */
int64_t jitDictGetInt(ObjDict *d, int64_t key, Value *result, uint64_t defTag,
                      int64_t defPayload) {
    JaiEntry *e = jaiTableFindIntQuick(&d->table, key,
                                       jaiHashU64Inline((uint64_t)key));
    if (JAI_UNLIKELY(e == JAI_TABLE_SLOW)) return 1;
    return dictGetAnswer(e, result, defTag, defPayload);
}

static JAI_NOINLINE int64_t dictSetIntSlow(ObjDict *d, int64_t key,
                                           uint64_t tag, int64_t payload) {
    Value v;
    v.type = (ValueType)tag;
    v.as.integer = payload;
    if (!jaiKindAccepts(d->keyKind, INT_VAL(key)) ||
        !jaiKindAccepts(d->valKind, v)) {
        return 1;
    }
    const uint64_t hash = jaiHashU64Inline((uint64_t)key);
    JaiEntry *e = jaiTableFindIntQuick(&d->table, key, hash);
    if (JAI_UNLIKELY(e == JAI_TABLE_SLOW)) return 1;
    if (e != NULL) {
        e->value = v;
        ++d->table.version;
        return 0;
    }
    (void)jaiTableSetHashed(&d->table, INT_VAL(key), hash, v);
    return 0;
}

int64_t jitDictSetInt(ObjDict *d, int64_t key, uint64_t tag, int64_t payload) {
    if (JAI_UNLIKELY(d->keyKind != FIELD_KIND_ANY ||
                     d->valKind != FIELD_KIND_ANY)) {
        return dictSetIntSlow(d, key, tag, payload);
    }
    const uint64_t hash = jaiHashU64Inline((uint64_t)key);
    JaiEntry *e = jaiTableFindIntQuick(&d->table, key, hash);
    if (JAI_UNLIKELY(e == JAI_TABLE_SLOW)) return 1;
    Value v;
    v.type = (ValueType)tag;
    v.as.integer = payload;
    if (e != NULL) {
        e->value = v;
        ++d->table.version;
        return 0;
    }
    /* After a NULL from the quick probe the insert walks the same slots and
     * meets no hash-equal key, so it compares nothing. */
    (void)jaiTableSetHashed(&d->table, INT_VAL(key), hash, v);
    return 0;
}

int64_t jitDictHasInt(Obj *container, int64_t key, Value *result,
                      int64_t negate) {
    if (container->type != OBJ_DICT) return 1;
    JaiEntry *e = jaiTableFindIntQuick(&((ObjDict *)container)->table, key,
                                       jaiHashU64Inline((uint64_t)key));
    if (JAI_UNLIKELY(e == JAI_TABLE_SLOW)) return 1;
    *result = BOOL_VAL((e != NULL) != (negate != 0));
    return 0;
}

static JAI_NOINLINE int64_t dictAddIntSlow(ObjDict *d, int64_t key,
                                           int64_t defPayload, int64_t addend,
                                           int64_t absentSlow) {
    if (!jaiKindAccepts(d->keyKind, INT_VAL(key)) ||
        !jaiKindAccepts(d->valKind, INT_VAL(0))) {
        return 1;
    }
    const uint64_t hash = jaiHashU64Inline((uint64_t)key);
    JaiEntry *e = jaiTableFindIntQuick(&d->table, key, hash);
    if (JAI_UNLIKELY(e == JAI_TABLE_SLOW)) return 1;
    int64_t base = defPayload;
    if (e != NULL) {
        if (e->value.type != VAL_INT) return 1;
        base = e->value.as.integer;
    } else if (absentSlow != 0) {
        return 1;
    }
    int64_t sum;
    if (__builtin_add_overflow(base, addend, &sum)) return 1;
    if (e != NULL) {
        e->value = INT_VAL(sum);
        ++d->table.version;
        return 0;
    }
    (void)jaiTableSetHashed(&d->table, INT_VAL(key), hash, INT_VAL(sum));
    return 0;
}

int64_t jitDictAddInt(Obj *dictObj, int64_t key, int64_t defPayload,
                      int64_t addend, int64_t absentSlow) {
    if (JAI_UNLIKELY(dictObj->type != OBJ_DICT)) return 1;
    ObjDict *d = (ObjDict *)dictObj;
    if (JAI_LIKELY(d->keyKind == FIELD_KIND_ANY &&
                   d->valKind == FIELD_KIND_ANY)) {
        JaiEntry *e = jaiTableFindIntQuick(&d->table, key,
                                           jaiHashU64Inline((uint64_t)key));
        if (JAI_LIKELY(e != NULL && e != JAI_TABLE_SLOW &&
                       e->value.type == VAL_INT)) {
            int64_t sum;
            if (__builtin_add_overflow(e->value.as.integer, addend, &sum)) {
                return 1;
            }
            e->value.as.integer = sum;
            ++d->table.version;
            return 0;
        }
    }
    return dictAddIntSlow(d, key, defPayload, addend, absentSlow);
}

/* `k in d` and `k not in d` with a string `k`, as a leaf for the reason
 * jitDictGetStr gives. The container is only predicted to be a dict, so the
 * leaf checks both kinds itself, and anything else -- a list, a set, a class
 * with `__contains__`, a probe the quick form cannot settle -- goes back to
 * jitContains, which is jaiContainsOp. Returns 0 with the answer written as
 * BOOL_VAL, as jitContains writes it, or 1 having written nothing. */
int64_t jitDictHasStr(Obj *container, Obj *needle, Value *result,
                      int64_t negate) {
    if (container->type != OBJ_DICT || needle->type != OBJ_STRING) return 1;
    JaiEntry *e = jaiTableFindStrQuick(&((ObjDict *)container)->table,
                                       (ObjString *)needle);
    if (JAI_UNLIKELY(e == JAI_TABLE_SLOW)) return 1;
    *result = BOOL_VAL((e != NULL) != (negate != 0));
    return 0;
}

int jitCallOut(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    bool ok = jaiCallValue(d->callee, (int)d->argc, d->args, &d->result);
    jaiGCPopRootRange();
    return ok ? 0 : 1;
}

#else

/* Nothing compiles here, so there is never a record to resume from and never a
 * compiled frame to mark. */
bool jaiJitApplyDeopt(ObjClosure *closure, Value *slotBase) {
    (void)closure; (void)slotBase; return false;
}
void jaiJitMarkFrames(void) {
}

void jaiJitExemplarsReset(void) {
}

#endif
