/* builtins.c — the builtin namespace, the argument helpers, and the method
 * dispatcher for primitive receivers.
 *
 * Two namespaces are built here (spec §9 and Appendix C):
 *
 *   - the *builtins module* (`vm.builtins`), which is the implicit global scope
 *     every module sees: print, len, range, the conversions, and the exception
 *     classes registered by errors.c;
 *   - `__prim__`, a module object living in the builtins, holding the raw C
 *     primitives that `lib/std` wraps. Primitives are registered by passing a
 *     dotted name to jaiDefineNative ("__prim__.add"), which is what keeps them
 *     out of the global namespace.
 *
 * Every name defined here is also registered with the resolver so that the
 * front end binds it at compile time instead of reporting E0200.
 *
 * The natives themselves are next door: builtins_core.c holds the ones a
 * program can name, builtins_prim.c the `__prim__` operators.
 */

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

/* Resolve the leading dotted components of `name` to a namespace module inside
 * the builtins, creating the modules on the way, and advance `name` past them.
 * "__prim__.f64_sqrt" therefore defines `f64_sqrt` in the `__prim__` module. */
static ObjModule *namespaceFor(const char **name);

/* A builtin that *is* a class, a trait or an enum used to be registered with
 * the C resolver, so that `class P: Printable` could name it in a type
 * position. That registry was part of the C checker and went with it; the
 * self-hosted checker learns the same names from the prelude it compiles. */

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

/* Define `value` under `name` in `module`, exporting it so that
 * `from __prim__ import x` and `module.x` both see it. */
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

/* One wording for every argument mismatch, so that a native reads as a
 * straight line of guards and every message looks the same:
 *     print() argument 1: expected int, got str
 * `index` is 1-based — it is what the user counted when writing the call. */
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
    /* Error classes come first: every other native throws through them. The VM
     * may already have built them during jaiVMInit. */
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
    jaiRegisterGpuPrimitives();
}

/* ------------------------------------------------------------------ */
/* Bound natives for built-in methods                                   */
/* ------------------------------------------------------------------ */

/* `xs.push` must not allocate a fresh ObjNative on every lookup, so the natives
 * are memoised per (name, fn) pair. The cache holds raw pointers, so every
 * cached native is also kept in an anchor list stored in the builtins module —
 * that list is what the GC traces. Without a builtins module to anchor to,
 * caching is skipped rather than risking a dangling pointer. */

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
    /* Hash the low representation bytes without a non-portable function-pointer
     * cast. On normal 32/64-bit ABIs this compiles to a move + jaiHashU64(). */
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

    /*
     * Reaching here means either:
     *
     *  1. the anchor/cache was already valid and jaiBindNative() proved the
     *     native was absent, or
     *  2. ensureNativeAnchor() just created/reset the cache, so it is empty.
     *
     * Either way there is no reason to probe it again.
     */
    ObjNative *native = jaiNativeNew(fn, name, minArity, maxArity, paramNames);

    /*
     * native is only a raw C pointer at this point. Keep it rooted through
     * cache insertion and jaiBoundNew(). This is particularly important when
     * caching is unavailable: jaiBoundNew() itself may allocate/collect.
     */
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

    /*
     * Normal case after VM startup: no helper call, no anchor check function,
     * no GC-root traffic until the allocation itself.
     */
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
void jaiMethodTablesInit(void) {
    resetNativeCache();
    (void)ensureNativeAnchor();   /* a no-op until the builtins module exists */
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

/* dir() cannot enumerate a lookup function, so it probes it with the names each
 * type is known to answer to. A name the implementation does not (yet) provide
 * simply does not appear; nothing here is load-bearing for dispatch. */

static const char *const kStrMethodNames[] = {
    "at", "capitalize", "center", "chars", "code_at", "contains", "count",
    "encode", "ends_with", "find", "format", "index", "is_alnum", "is_alpha",
    "is_digit", "is_empty", "is_lower", "is_space", "is_upper", "iter", "join",
    "len", "lines", "lower", "lstrip", "pad_left", "pad_right", "repeat",
    "replace", "reversed", "rfind", "rstrip", "slice", "split", "splitlines",
    "starts_with", "strip", "title", "to_float", "to_int", "to_str", "upper",
};

static const char *const kListMethodNames[] = {
    "all", "any", "append", "at", "chunks", "clear", "clone", "concat",
    "contains", "copy", "count", "drop", "enumerate", "extend", "filter",
    "find", "first", "flatten", "fold", "index", "insert", "is_empty", "iter",
    "join", "last", "len", "map", "max", "min", "pop", "position", "push",
    "reduce", "remove", "reverse", "reversed", "set", "shuffle", "slice",
    "sort", "sorted", "sum", "take", "to_list", "unique", "zip",
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
    "at", "contains", "count", "first", "get", "index", "is_empty", "iter",
    "last", "len", "second", "slice", "to_list",
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

static const char *const kIterMethodNames[] = {
    "all", "any", "chain", "collect", "count", "drop", "enumerate", "filter",
    "find", "fold", "last", "map", "next", "nth", "peekable", "take",
    "to_list", "zip",
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
