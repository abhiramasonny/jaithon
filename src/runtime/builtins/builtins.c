/* builtins.c — the builtin namespace, the argument helpers, and the method
 * dispatcher for primitive receivers. */

#include <ctype.h>
#include <stdlib.h>
#include <string.h>

#include "runtime/builtins/builtins.h"
#include "runtime/methods.h"
#include "runtime/runtime.h"

#include "vm/gc.h"

/* ------------------------------------------------------------------ */
/* Callability                                                          */
/* ------------------------------------------------------------------ */

bool jaiBuiltinIsCallable(Value v) {
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
            ObjInstance *const inst = AS_INSTANCE(v);
            ObjClass *const klass = inst->klass;
            return klass != NULL && !IS_NULL(klass->dunderCall);
        }

        default:
            return false;
    }
}

/* ------------------------------------------------------------------ */
/* Registration                                                         */
/* ------------------------------------------------------------------ */

static ObjModule *namespaceFor(const char **name);

static ObjModule *makeNamespace(ObjModule *parent, const char *name, size_t len) {
    ObjString *key = jaiStringIntern(name, len);
    jaiGCPushRoot(OBJ_VAL(key));

    Value existing;
    if (jaiTableGetInterned(&parent->globals, key, &existing)) {
        if (IS_MODULE(existing)) {
            jaiGCPopRoot();
            return AS_MODULE(existing);
        }
        JAI_PANIC("builtin namespace `%s` collides with an existing global",
                  key->chars);
    }

    ObjModule *ns = jaiModuleNew(key, key);
    ns->state = MOD_LOADED;
    jaiGCPushRoot(OBJ_VAL(ns));
    jaiModuleSet(parent, key, OBJ_VAL(ns));
    jaiGCPopRoots(2);

    return ns;
}

static ObjModule *namespaceFor(const char **name) {
    ObjModule *module = vm.builtins;
    if (module == NULL) return NULL;

    const char *s = *name;
    const char *dot;
    while ((dot = strchr(s, '.')) != NULL) {
        if (dot == s) break;                 /* leading dot: not a namespace */
        module = makeNamespace(module, s, (size_t)(dot - s));
        s = dot + 1;
    }
    *name = s;
    return module;
}

static void defineIn(ObjModule *module, const char *name, Value value) {
    jaiGCPushRoot(value);
    ObjString *key = jaiStringInternC(name);
    jaiGCPushRoot(OBJ_VAL(key));
    jaiModuleSet(module, key, value);
    (void)jaiTableSetInterned(&module->exports, key, BOOL_VAL(true));
    jaiGCPopRoots(2);
}

void jaiDefineGlobal(const char *name, Value value) {
    if (name == NULL) return;
    if (vm.builtins == NULL) JAI_PANIC("jaiDefineGlobal(%s) before jaiVMInit", name);

    const char *member = name;
    ObjModule *module = namespaceFor(&member);
    if (module == NULL || *member == '\0') return;

    defineIn(module, member, value);
}

void jaiDefineNative(const char *name, JaiNativeFn fn, int minArity, int maxArity) {
    if (name == NULL || fn == NULL) return;
    if (vm.builtins == NULL) JAI_PANIC("jaiDefineNative(%s) before jaiVMInit", name);

    const char *member = name;
    ObjModule *module = namespaceFor(&member);
    if (module == NULL || *member == '\0') return;

    ObjNative *native = jaiNativeNew(fn, member, minArity, maxArity, NULL);
    defineIn(module, member, OBJ_VAL(native));
}

/* ------------------------------------------------------------------ */
/* Argument helpers                                                     */
/* ------------------------------------------------------------------ */

bool jaiBuiltinArgTypeError(int index, const char *fnName, const char *expected,
                         Value got) {
    return jaiThrow(vm.cTypeError, "%s() argument %d: expected %s, got %s",
                    fnName != NULL ? fnName : "<native>", index, expected,
                    jaiTypeNameStatic(got));
}

bool jaiArgInt(Value v, int index, const char *fnName, int64_t *out) {
    if (!IS_INT(v)) return jaiBuiltinArgTypeError(index, fnName, "int", v);
    *out = AS_INT(v);
    return true;
}

bool jaiArgFloat(Value v, int index, const char *fnName, double *out) {
    if (!IS_FLOAT(v)) return jaiBuiltinArgTypeError(index, fnName, "float", v);
    *out = AS_FLOAT(v);
    return true;
}

bool jaiArgNumber(Value v, int index, const char *fnName, double *out) {
    if (IS_FLOAT(v)) {
        *out = AS_FLOAT(v);
        return true;
    }
    if (IS_INT(v)) {
        *out = (double)AS_INT(v);
        return true;
    }
    return jaiBuiltinArgTypeError(index, fnName, "int or float", v);
}

bool jaiArgBool(Value v, int index, const char *fnName, bool *out) {
    if (!IS_BOOL(v)) return jaiBuiltinArgTypeError(index, fnName, "bool", v);
    *out = AS_BOOL(v);
    return true;
}

bool jaiArgString(Value v, int index, const char *fnName, ObjString **out) {
    if (!IS_STRING(v)) return jaiBuiltinArgTypeError(index, fnName, "str", v);
    *out = AS_STRING(v);
    return true;
}

bool jaiArgList(Value v, int index, const char *fnName, ObjList **out) {
    if (!IS_LIST(v)) return jaiBuiltinArgTypeError(index, fnName, "list", v);
    *out = AS_LIST(v);
    return true;
}

bool jaiArgDict(Value v, int index, const char *fnName, ObjDict **out) {
    if (!IS_DICT(v)) return jaiBuiltinArgTypeError(index, fnName, "dict", v);
    *out = AS_DICT(v);
    return true;
}

bool jaiArgCallable(Value v, int index, const char *fnName) {
    if (!jaiBuiltinIsCallable(v)) return jaiBuiltinArgTypeError(index, fnName, "callable", v);
    return true;
}

/* ------------------------------------------------------------------ */
/* Registration entry point                                             */
/* ------------------------------------------------------------------ */

void jaiRegisterAllBuiltins(void) {
    if (vm.cError == NULL) jaiRegisterErrorClasses();

    jaiMethodTablesInit();
    jaiRegisterCoreBuiltins();

    jaiRegisterMathPrimitives();
    jaiRegisterCanvasPrimitives();
    jaiRegisterCompressPrimitives();
    jaiRegisterStringPrimitives();
    jaiRegisterListPrimitives();
    jaiRegisterDictPrimitives();
    jaiRegisterIOPrimitives();
    jaiRegisterOSPrimitives();
    jaiRegisterProcessPrimitives();
    jaiRegisterTimePrimitives();
    jaiRegisterRandomPrimitives();
    jaiRegisterThreadPrimitives();
    jaiRegisterGCPrimitives();
    jaiRegisterReflectPrimitives();
    jaiRegisterGuiPrimitives();
    jaiRegisterCameraPrimitives();
    jaiRegisterGpuPrimitives();
    jaiRegisterGraphPrimitives();
    jaiRegisterCoreMLPrimitives();
}

/* ------------------------------------------------------------------ */
/* Bound natives for built-in methods                                   */
/* ------------------------------------------------------------------ */

typedef struct {
    JaiNativeFn fn;
    ObjNative  *native;
    uint64_t    hash;
} NativeCacheEntry;

static NativeCacheEntry *gNativeCache;
static int               gNativeCacheCap;      /* power of two, or 0 */
static int               gNativeCacheCount;
static ObjList          *gNativeAnchor;
static ObjModule        *gNativeAnchorModule;

static const char kNativeAnchorName[] = "__natives__";

static inline uint64_t hashNativeFn(JaiNativeFn fn) {
    uint64_t bits = 0;
    const size_t n = sizeof fn < sizeof bits ? sizeof fn : sizeof bits;
    memcpy(&bits, &fn, n);
    return jaiHashU64(bits);
}

static inline void resetNativeCache(void) {
    JAI_FREE_ARRAY(NativeCacheEntry, gNativeCache, gNativeCacheCap);
    gNativeCache = NULL;
    gNativeCacheCap = 0;
    gNativeCacheCount = 0;
    gNativeAnchor = NULL;
    gNativeAnchorModule = NULL;
}

static inline bool ensureNativeAnchor(void) {
    if (gNativeAnchor != NULL && gNativeAnchorModule == vm.builtins)
        return true;

    if (vm.builtins == NULL)
        return false;

    /* A different builtins module means a different heap. */
    resetNativeCache();

    ObjList *anchor = jaiListNew(0);
    jaiGCPushRoot(OBJ_VAL(anchor));

    ObjString *key =
        jaiStringIntern(kNativeAnchorName, sizeof kNativeAnchorName - 1);
    jaiGCPushRoot(OBJ_VAL(key));

    jaiModuleSet(vm.builtins, key, OBJ_VAL(anchor));
    jaiGCPopRoots(2);

    gNativeAnchor = anchor;
    gNativeAnchorModule = vm.builtins;
    return true;
}

static inline ObjNative *nativeCacheFind(JaiNativeFn fn, const char *name,
                                         uint64_t hash) {
    const int capacity = gNativeCacheCap;
    if (capacity == 0) return NULL;

    const uint32_t mask = (uint32_t)capacity - 1u;
    uint32_t i = (uint32_t)hash & mask;

    for (;;) {
        NativeCacheEntry *const entry = gNativeCache + i;
        ObjNative *const native = entry->native;

        if (native == NULL)
            return NULL;

        if (entry->hash == hash && entry->fn == fn &&
            strcmp(native->name->chars, name) == 0)
            return native;

        i = (i + 1u) & mask;
    }
}

static inline void nativeCacheInsertRaw(NativeCacheEntry *table, int capacity,
                                        JaiNativeFn fn, ObjNative *native,
                                        uint64_t hash) {
    const uint32_t mask = (uint32_t)capacity - 1u;
    uint32_t i = (uint32_t)hash & mask;

    while (table[i].native != NULL)
        i = (i + 1u) & mask;

    table[i].fn = fn;
    table[i].native = native;
    table[i].hash = hash;
}

static void nativeCacheInsert(JaiNativeFn fn, ObjNative *native,
                              uint64_t hash) {
    int capacity = gNativeCacheCap;

    if (gNativeCacheCount + 1 > capacity - capacity / 4) {
        const int newCap = capacity < 64 ? 64 : capacity * 2;
        NativeCacheEntry *grown =
            JAI_ALLOC_ZEROED(NativeCacheEntry, newCap);

        for (int i = 0; i < capacity; ++i) {
            const NativeCacheEntry *const old = gNativeCache + i;
            if (old->native == NULL) continue;

            nativeCacheInsertRaw(grown, newCap, old->fn,
                                 old->native, old->hash);
        }

        JAI_FREE_ARRAY(NativeCacheEntry, gNativeCache, capacity);
        gNativeCache = grown;
        gNativeCacheCap = newCap;
        capacity = newCap;
    }

    nativeCacheInsertRaw(gNativeCache, capacity, fn, native, hash);
    ++gNativeCacheCount;
}

static Value bindNativeSlow(Value receiver, const char *name, JaiNativeFn fn,
                            int minArity, int maxArity,
                            const char *const *paramNames, uint64_t hash) {
    const bool rootReceiver = IS_OBJ(receiver);

    if (rootReceiver)
        jaiGCPushRoot(receiver);

    const bool anchored = ensureNativeAnchor();

    ObjNative *native = jaiNativeNew(fn, name, minArity, maxArity, paramNames);

    jaiGCPushRoot(OBJ_VAL(native));

    if (anchored) {
        jaiListPush(gNativeAnchor, OBJ_VAL(native));
        nativeCacheInsert(fn, native, hash);
    }

    ObjBound *bound = jaiBoundNew(receiver, OBJ_VAL(native));

    jaiGCPopRoot(); /* native */

    if (rootReceiver)
        jaiGCPopRoot();

    return OBJ_VAL(bound);
}

Value jaiBindNative(Value receiver, const char *name, JaiNativeFn fn,
                    int minArity, int maxArity,
                    const char *const *paramNames) {
    const uint64_t hash = hashNativeFn(fn);

    if (gNativeAnchor != NULL && gNativeAnchorModule == vm.builtins) {
        ObjNative *native = nativeCacheFind(fn, name, hash);

        if (native != NULL) {
            if (!IS_OBJ(receiver))
                return OBJ_VAL(jaiBoundNew(receiver, OBJ_VAL(native)));

            jaiGCPushRoot(receiver);
            ObjBound *bound = jaiBoundNew(receiver, OBJ_VAL(native));
            jaiGCPopRoot();

            return OBJ_VAL(bound);
        }
    }

    return bindNativeSlow(receiver, name, fn,
                          minArity, maxArity, paramNames, hash);
}
#ifdef JAI_DEBUG
static void assertMethodTablesInSync(void);
#endif

void jaiMethodTablesInit(void) {
    resetNativeCache();
    (void)ensureNativeAnchor();   /* a no-op until the builtins module exists */
#ifdef JAI_DEBUG
    assertMethodTablesInSync();
#endif
}

/* ------------------------------------------------------------------ */
/* Method dispatch for primitive receivers                              */
/* ------------------------------------------------------------------ */

typedef bool (*MethodLookup)(Value, ObjString *, Value *);

static inline MethodLookup lookupFor(Value receiver) {
    switch (jaiValueType(receiver)) {
        case VAL_INT:   return jaiIntMethod;
        case VAL_FLOAT: return jaiFloatMethod;
        case VAL_NULL:
        case VAL_BOOL:  return NULL;
        case VAL_OBJ:   break;
    }

    switch (OBJ_TYPE(receiver)) {
        case OBJ_STRING: return jaiStrMethod;
        case OBJ_LIST:   return jaiListMethod;
        case OBJ_DICT:   return jaiDictMethod;
        case OBJ_SET:    return jaiSetMethod;
        case OBJ_TUPLE:  return jaiTupleMethod;
        case OBJ_RANGE:  return jaiRangeMethod;
        case OBJ_BYTES:  return jaiBytesMethod;
        case OBJ_FILE:   return jaiFileMethod;
        case OBJ_MODULE: return jaiModuleMethod;
        case OBJ_ITER:   return jaiIterMethod;
        default:         return NULL;
    }
}

bool jaiBuiltinMethod(Value receiver, ObjString *name, Value *out) {
    if (name == NULL) return false;

    switch (jaiValueType(receiver)) {
        case VAL_INT:
            if (jaiIntMethod(receiver, name, out)) return true;
            break;

        case VAL_FLOAT:
            if (jaiFloatMethod(receiver, name, out)) return true;
            break;

        case VAL_NULL:
        case VAL_BOOL:
            break;

        case VAL_OBJ:
            switch (OBJ_TYPE(receiver)) {
                case OBJ_STRING:
                    if (jaiStrMethod(receiver, name, out)) return true;
                    break;
                case OBJ_LIST:
                    if (jaiListMethod(receiver, name, out)) return true;
                    break;
                case OBJ_DICT:
                    if (jaiDictMethod(receiver, name, out)) return true;
                    break;
                case OBJ_SET:
                    if (jaiSetMethod(receiver, name, out)) return true;
                    break;
                case OBJ_TUPLE:
                    if (jaiTupleMethod(receiver, name, out)) return true;
                    break;
                case OBJ_RANGE:
                    if (jaiRangeMethod(receiver, name, out)) return true;
                    break;
                case OBJ_BYTES:
                    if (jaiBytesMethod(receiver, name, out)) return true;
                    break;
                case OBJ_FILE:
                    if (jaiFileMethod(receiver, name, out)) return true;
                    break;
                case OBJ_MODULE:
                    if (jaiModuleMethod(receiver, name, out)) return true;
                    break;
                case OBJ_ITER:
                    if (jaiIterMethod(receiver, name, out)) return true;
                    break;
                default:
                    break;
            }
            break;
    }

    return jaiValueFormatMethod(receiver, name, out);
}

/* ------------------------------------------------------------------ */
/* Introspection                                                        */
/* ------------------------------------------------------------------ */

static const char *const kStrMethodNames[] = {
    "capitalize", "center", "chars", "contains", "count", "ends_with", "find",
    "format", "index", "is_alnum", "is_alpha", "is_digit", "is_empty",
    "is_lower", "is_space", "is_upper", "iter", "join", "len", "lower",
    "lstrip", "pad_left", "pad_right", "repeat", "replace", "rfind", "rstrip",
    "slice", "split", "splitlines", "starts_with", "strip", "title",
    "to_float", "to_int", "to_str", "upper",
};

/* `clone`, `concat`, `find`, `fold` and `position` were advertised and have
 * never existed. Each is a second name for something that does: `copy`,
 * `extend` (and `a + b`), `index`/`filter`, `reduce`, `index`. `find` was the
 * one worth not guessing -- predicate or value, index or element, throw or
 * null -- and the two spellings that DO exist already cover both readings. */
static const char *const kListMethodNames[] = {
    "all", "any", "at", "chunks", "clear", "contains",
    "copy", "count", "enumerate", "extend", "filter", "first",
    "flatten", "index", "insert", "is_empty", "iter", "join", "last",
    "len", "map", "max", "min", "pop", "push", "reduce", "remove",
    "reverse", "reversed", "set", "shuffle", "slice", "sort", "sorted", "sum",
    "unique", "zip",
};

static const char *const kDictMethodNames[] = {
    "clear", "contains", "copy", "filter", "get", "get_or_insert", "has",
    "items", "iter", "keys", "len", "map_values", "merge", "pop", "remove",
    "set", "update", "values",
};

static const char *const kSetMethodNames[] = {
    "add", "clear", "contains", "copy", "difference", "discard", "has",
    "intersection", "is_subset", "is_superset", "iter", "len", "remove",
    "symmetric_difference", "to_list", "union",
};

static const char *const kTupleMethodNames[] = {
    "at", "contains", "count", "get", "index", "is_empty", "iter", "len",
    "to_list",
};

static const char *const kRangeMethodNames[] = {
    "contains", "is_empty", "iter", "len", "reversed", "start", "step", "stop",
    "to_list",
};

static const char *const kBytesMethodNames[] = {
    "at", "concat", "contains", "decode", "get", "hex", "is_empty", "iter",
    "len", "slice", "to_list",
};

static const char *const kIntMethodNames[] = {
    "abs", "bit_length", "clamp", "hash", "is_even", "is_odd", "max", "min",
    "pow", "sign", "to_float", "to_int", "to_str",
};

static const char *const kFloatMethodNames[] = {
    "abs", "ceil", "clamp", "floor", "hash", "is_finite", "is_inf", "is_nan",
    "round", "sign", "sqrt", "to_float", "to_int", "to_str", "trunc",
};

static const char *const kFileMethodNames[] = {
    "close", "flush", "is_closed", "iter", "lines", "read", "read_line",
    "read_lines", "seek", "tell", "write", "write_line",
};

static const char *const kModuleMethodNames[] = {
    "get", "has", "members", "name", "path",
};

/* The adapter surface deliberately lives in `lib/std/iter`, whose `iter(xs)`
 * wrapper is what `.chunks(2).collect()` is a method of. The runtime `ObjIter`
 * has only ever carried `collect`, `drop`, `iter`, `next` and `take`, and
 * advertising eight adapters it has never had is the bug this table is being
 * repaired for. `to_list` additionally duplicates `collect`, which is the name
 * LANGUAGE.md uses. Anything wanted later belongs in `std.iter`, not here.
 *
 * `iter` goes the other way: it is implemented (builtins_seq.c) and works, and
 * was missing from this list, so `dir()` hid it. */
static const char *const kIterMethodNames[] = {
    "collect", "drop", "iter", "next", "take",
};

#define NAME_TABLE(table)                                                      \
    do {                                                                       \
        *count = (int)(sizeof(table) / sizeof((table)[0]));                    \
        return (table);                                                        \
    } while (0)

static const char *const *candidatesFor(Value receiver, int *count) {
    *count = 0;
    switch (jaiValueType(receiver)) {
    case VAL_INT:   NAME_TABLE(kIntMethodNames);
    case VAL_FLOAT: NAME_TABLE(kFloatMethodNames);
    case VAL_NULL:
    case VAL_BOOL:  return NULL;
    case VAL_OBJ:   break;
    }
    switch (OBJ_TYPE(receiver)) {
    case OBJ_STRING: NAME_TABLE(kStrMethodNames);
    case OBJ_LIST:   NAME_TABLE(kListMethodNames);
    case OBJ_DICT:   NAME_TABLE(kDictMethodNames);
    case OBJ_SET:    NAME_TABLE(kSetMethodNames);
    case OBJ_TUPLE:  NAME_TABLE(kTupleMethodNames);
    case OBJ_RANGE:  NAME_TABLE(kRangeMethodNames);
    case OBJ_BYTES:  NAME_TABLE(kBytesMethodNames);
    case OBJ_FILE:   NAME_TABLE(kFileMethodNames);
    case OBJ_MODULE: NAME_TABLE(kModuleMethodNames);
    case OBJ_ITER:   NAME_TABLE(kIterMethodNames);
    default:         return NULL;
    }
}

#undef NAME_TABLE

ObjList *jaiBuiltinMethodNames(Value receiver) {
    ObjList *names = jaiListNew(0);
    MethodLookup lookup = lookupFor(receiver);
    if (lookup == NULL) return names;

    int count = 0;
    const char *const *candidates = candidatesFor(receiver, &count);
    if (candidates == NULL) return names;

    jaiGCPushRoot(OBJ_VAL(names));
    jaiGCPushRoot(receiver);
    for (int i = 0; i < count; i++) {
        ObjString *name = jaiStringInternC(candidates[i]);
        jaiGCPushRoot(OBJ_VAL(name));
        Value bound;
        bool present = lookup(receiver, name, &bound);
        if (present) jaiListPush(names, OBJ_VAL(name));
        jaiGCPopRoot();
        if (vm.hasException) break;
    }
    jaiGCPopRoots(2);
    return names;
}

#ifdef JAI_DEBUG
/* ------------------------------------------------------------------ */
/* Debug-build self-check: advertised names vs. runtime dispatch        */
/* ------------------------------------------------------------------ */

/* The k*MethodNames[] tables above and the k*Methods[] dispatch tables
 * scattered across src/runtime/builtins/ (four incompatible shapes: the
 * JaiSeqMethod/JAI_METHOD tables the collections sources share, the bare
 * JaiStrMethodEntry literals str and bytes use, builtins_math.c's file-local
 * MethodEntry/METHOD_ENTRY for int and float, and the FILE_METHOD/
 * MODULE_METHOD strcmp chains for file and module) are two hand-maintained
 * lists of the same 12 receiver kinds' methods, kept in sync by nobody. They
 * drifted: 46 names advertised here (list.is_empty, iter.map, str.slice,
 * ...) type-checked and then raised AttributeError at a user's run time,
 * because ./jaithon check never consults either table (see
 * docs/agents/methods-single-source.md) and dir() -- the only real reader of
 * these lists -- silently drops whatever candidatesFor() offers that
 * lookupFor() refuses, so the gap was invisible short of trying every name.
 *
 * Unifying the four shapes into one generated table was rejected as too
 * expensive for what this bug needs (see the doc above): four parsers for
 * four incompatible C shapes, a new build step, and a decision on whether to
 * commit generated output, to fix a problem this cheaper check also fixes.
 * Instead: reuse the exact lookupFor/candidatesFor pair jaiBuiltinMethodNames
 * uses to build dir()'s answer, but assert presence instead of silently
 * filtering it. One representative, harmless instance per receiver kind
 * (an empty list, a closed-handle file, ...) is enough, because every
 * lookup* function decides presence by name and (for the few that check it)
 * receiver *kind*, never by a receiver's contents.
 *
 * This runs once, at every debug-build startup -- before any user code, so
 * drift aborts the process for the developer who introduced it, not for a
 * user running a script months later. It costs nothing in a release build
 * (this whole function is compiled out under NDEBUG, which JAI_DEBUG and
 * RELEASE_CFLAGS never both define) and nothing meaningful in a debug one:
 * a dozen small allocations and 218 table lookups (every advertised name,
 * summed across all twelve receiver kinds), once, at startup. */
static void assertMethodTablesInSync(void) {
    ObjList *emptyList = jaiListNew(0);
    jaiGCPushRoot(OBJ_VAL(emptyList));
    ObjString *emptyStr = jaiStringInternC("");
    jaiGCPushRoot(OBJ_VAL(emptyStr));
    ObjBytes *emptyBytes = jaiBytesNew(NULL, 0);
    jaiGCPushRoot(OBJ_VAL(emptyBytes));
    ObjTuple *emptyTuple = jaiTupleNew(NULL, 0);
    jaiGCPushRoot(OBJ_VAL(emptyTuple));
    ObjDict *emptyDict = jaiDictNew();
    jaiGCPushRoot(OBJ_VAL(emptyDict));
    ObjSet *emptySet = jaiSetNew();
    jaiGCPushRoot(OBJ_VAL(emptySet));
    ObjRange *emptyRange = jaiRangeNew(0, 0, 1, false);
    jaiGCPushRoot(OBJ_VAL(emptyRange));
    /* A NULL handle is a real, closed-file state (see object.c's OBJ_FILE
     * free case) -- no filesystem I/O happens here. */
    ObjFile *dummyFile = jaiFileNew(NULL, NULL, "r");
    jaiGCPushRoot(OBJ_VAL(dummyFile));
    ObjIter *emptyIter = jaiIterNew(ITER_LIST, OBJ_VAL(emptyList));
    jaiGCPushRoot(OBJ_VAL(emptyIter));
    enum { kRootCount = 9 };

    /* vm.builtins is live by the time jaiMethodTablesInit runs: jaiVMInit
     * assigns it before calling jaiRegisterAllBuiltins (see vm.c). */
    struct { const char *kind; Value receiver; } probes[] = {
        {"int",    INT_VAL(0)},
        {"float",  FLOAT_VAL(0.0)},
        {"str",    OBJ_VAL(emptyStr)},
        {"list",   OBJ_VAL(emptyList)},
        {"dict",   OBJ_VAL(emptyDict)},
        {"set",    OBJ_VAL(emptySet)},
        {"tuple",  OBJ_VAL(emptyTuple)},
        {"range",  OBJ_VAL(emptyRange)},
        {"bytes",  OBJ_VAL(emptyBytes)},
        {"file",   OBJ_VAL(dummyFile)},
        {"module", OBJ_VAL(vm.builtins)},
        {"iter",   OBJ_VAL(emptyIter)},
    };

    char report[4096];
    size_t used = 0;
    report[0] = '\0';
    int missing = 0;

    for (size_t p = 0; p < sizeof(probes) / sizeof(probes[0]); p++) {
        Value receiver = probes[p].receiver;
        MethodLookup lookup = lookupFor(receiver);
        int count = 0;
        const char *const *candidates = candidatesFor(receiver, &count);
        if (lookup == NULL || candidates == NULL) continue;

        for (int i = 0; i < count; i++) {
            ObjString *name = jaiStringInternC(candidates[i]);
            Value bound;
            if (lookup(receiver, name, &bound)) continue;

            missing++;
            int n = snprintf(report + used, sizeof(report) - used, "%s%s.%s",
                             used > 0 ? ", " : "", probes[p].kind,
                             candidates[i]);
            if (n > 0 && (size_t)n < sizeof(report) - used) {
                used += (size_t)n;
            } else {
                used = sizeof(report) - 1;   /* stop appending; report is full */
            }
        }
    }

    jaiGCPopRoots(kRootCount);

    if (missing > 0) {
        /* Names the two ways out, because whoever trips this has just added
         * one line to a table and will want to be told which they meant. */
        JAI_PANIC("%d builtin method name%s advertised in k*MethodNames[] "
                 "(builtins.c) %s no matching entry in the runtime dispatch "
                 "table that dir() and method calls actually use, so "
                 "`jaithon check` accepts a call to %s and `jaithon run` then "
                 "raises AttributeError. Missing (kind.name): %s. Either "
                 "implement %s in the matching k*Methods[] table, or drop the "
                 "name from k*MethodNames[]. See "
                 "docs/agents/methods-single-source.md.",
                 missing, missing == 1 ? "" : "s",
                 missing == 1 ? "has" : "have",
                 missing == 1 ? "it" : "them", report,
                 missing == 1 ? "it" : "them");
    }
}
#endif /* JAI_DEBUG */
