/* object.h — heap object layouts and constructors. */
#ifndef JAI_OBJECT_H
#define JAI_OBJECT_H

#include "value.h"
#include "chunk.h"
#include "table.h"

/* ------------------------------------------------------------------ */
/* Strings — immutable, interned, length-prefixed, hash-cached          */
/* ------------------------------------------------------------------ */

struct ObjString {
    Obj      obj;
    uint32_t length;      /* bytes, excluding NUL */
    uint32_t scalars;     /* UTF-8 scalar count, computed lazily (UINT32_MAX = unknown) */
    /* Memo for scalar indexing: cursorByte is the byte offset of scalar
     * cursorScalar. Indexing is by scalar, so on a string holding any byte
     * above 127 the offset has to be found by decoding — and a lexer walking
     * forwards one scalar at a time would rescan from the start every step,
     * which is quadratic. Zero/zero is always a valid starting memo. A pure
     * cache: it never changes what the string is. */
    uint32_t cursorScalar;
    uint32_t cursorByte;
    uint64_t hash;
    /* `interned` lives in Obj.subFlag: a bool of its own here would be padded
     * out to eight bytes ahead of the flexible array. Use the accessor. */
    char     chars[];     /* flexible array; always NUL-terminated */
};

/* Bytes to allocate for a string of `length` characters. sizeof(ObjString) is
 * rounded up to the struct's alignment and so overshoots where `chars`
 * actually begins; using it wasted seven bytes on every string in the heap. */
#define JAI_STRING_ALLOC(length) (offsetof(ObjString, chars) + (size_t)(length) + 1)

#define JAI_STR_INTERNED(s)      ((s)->obj.subFlag)

/* Interning: every string literal and identifier the compiler or the
 * deserialiser produces goes through the intern table, so that for interned
 * strings pointer equality is string equality — which is what lets member and
 * global lookup use jaiTableGetInterned. Strings built at run time are *not*
 * interned; equality on them is by content either way. */
ObjString *jaiStringIntern(const char *chars, size_t length);
ObjString *jaiStringInternC(const char *cstr);
/* The interned twin of `s`, interning `s` itself if none exists. Required
 * before using a run-time string as a key into a pointer-keyed table — i.e.
 * only on the reflective paths (get_field, module.get, exec's namespace),
 * since every name the compiler emits arrives interned already. */
ObjString *jaiStringCanonical(ObjString *s);
/* Not interned. The constructor for any string that is data rather than a name. */
ObjString *jaiStringNew(const char *chars, size_t length);
/* Takes ownership of a heap buffer allocated with jaiRealloc. */
ObjString *jaiStringTake(char *chars, size_t length);
ObjString *jaiStringConcat(ObjString *a, ObjString *b);
/* Concatenates `count` byte runs into one string, sized exactly, under the
 * same run-time interning policy jaiStringNew applies. The runs must stay
 * valid across a collection — allocating the result can trigger one. */
ObjString *jaiStringFromParts(const char *const *runs, const uint32_t *lens,
                              int count, size_t total);
/* An uninitialised string of exactly `length` bytes. The caller fills chars[]
 * — without allocating, since the string is not yet rooted — and then hands it
 * to jaiStringSeal, which applies the run-time interning policy and returns
 * the string to use (an existing equal one, if interning found it). */
ObjString *jaiStringReserve(size_t length);
ObjString *jaiStringSeal(ObjString *s);
ObjString *jaiStringSlice(ObjString *s, int64_t start, int64_t stop, int64_t step);
uint32_t   jaiStringScalarCount(ObjString *s);
/* The cached content hash, computed on first use. A string that never becomes
 * a dict key and never takes part in interning is then never hashed at all:
 * hashing in the constructor meant walking every byte of, for instance, the
 * 2.2 MB result of a join that the program only prints. Zero doubles as "not
 * computed yet"; a string that really does hash to zero just recomputes. */
static inline uint64_t jaiStringHash(ObjString *s) {
    if (s->hash == 0) s->hash = jaiHashBytes(s->chars, s->length);
    return s->hash;
}
/* The tail of jaiStringEquals: two strings that are neither identical nor both
 * interned, so the answer needs their bytes. */
bool       jaiStringEqualsSlow(const ObjString *a, const ObjString *b);

/* Inline because the callers are linear scans over parameter and field names
 * (vm.c bindCallArgs, jaiClassFieldInfo) that run tens of millions of times in
 * a compile, and for interned names every one of those iterations answers from
 * the two cheap tests below. Out of line, `jaiStringEquals` was the second
 * hottest function in the whole program by sample count -- almost entirely the
 * cost of calling it, not of anything it did. */
static inline bool jaiStringEquals(const ObjString *a, const ObjString *b) {
    if (a == b) return true;
    if (a == NULL || b == NULL) return false;
    /* Two distinct interned strings are never equal, by construction. */
    if (JAI_STR_INTERNED(a) && JAI_STR_INTERNED(b)) return false;
    return jaiStringEqualsSlow(a, b);
}

/* ------------------------------------------------------------------ */
/* Bytes                                                                */
/* ------------------------------------------------------------------ */

struct ObjBytes {
    Obj      obj;
    uint32_t length;
    uint8_t  data[];
};

ObjBytes *jaiBytesNew(const uint8_t *data, size_t length);

/* ------------------------------------------------------------------ */
/* List — growable Value array                                          */
/* ------------------------------------------------------------------ */

struct ObjList {
    Obj      obj;
    Value   *items;
    int      count;
    int      capacity;
    uint32_t version;   /* bumped on every mutation; iterators snapshot it */
};

ObjList *jaiListNew(int initialCapacity);
void     jaiListPush(ObjList *list, Value v);
Value    jaiListPop(ObjList *list);
void     jaiListInsert(ObjList *list, int index, Value v);
Value    jaiListRemove(ObjList *list, int index);
void     jaiListReserve(ObjList *list, int capacity);
/* Records a mutation the list functions above did not perform: an in-place
 * store through `items`, a sort, or a direct write to `count`. Any such site
 * on a list the program can already reach must call this, or a live iterator
 * will not see the change. Filling a list that has not escaped yet is exempt:
 * nothing can be iterating it. */
void     jaiListTouch(ObjList *list);
ObjList *jaiListSlice(ObjList *list, int64_t start, int64_t stop, int64_t step);
ObjList *jaiListConcat(ObjList *a, ObjList *b);
/* Normalises a possibly-negative index. Returns false if out of range. */
bool     jaiNormalizeIndex(int64_t raw, int length, int *out);

/* ------------------------------------------------------------------ */
/* Tuple — fixed-size, hashable                                         */
/* ------------------------------------------------------------------ */

struct ObjTuple {
    Obj      obj;
    uint32_t count;
    uint64_t hash;      /* 0 = not yet computed */
    Value    items[];
};

ObjTuple *jaiTupleNew(const Value *items, int count);

/* ------------------------------------------------------------------ */
/* Dict and Set — open addressing with tombstones (see table.h)         */
/* ------------------------------------------------------------------ */

struct ObjDict {
    Obj      obj;
    JaiTable table;      /* Value keys */
};

struct ObjSet {
    Obj      obj;
    JaiTable table;      /* keys only; values are NULL_VAL */
};

ObjDict *jaiDictNew(void);
bool     jaiDictGet(ObjDict *d, Value key, Value *out);
bool     jaiDictSet(ObjDict *d, Value key, Value value);   /* true if new key */
bool     jaiDictDelete(ObjDict *d, Value key);
ObjList *jaiDictKeys(ObjDict *d);
ObjList *jaiDictValues(ObjDict *d);
ObjList *jaiDictItems(ObjDict *d);                         /* list of 2-tuples */

ObjSet  *jaiSetNew(void);
bool     jaiSetAdd(ObjSet *s, Value v);
bool     jaiSetHas(ObjSet *s, Value v);
bool     jaiSetDelete(ObjSet *s, Value v);

/* ------------------------------------------------------------------ */
/* Range                                                                */
/* ------------------------------------------------------------------ */

struct ObjRange {
    Obj     obj;
    int64_t start, stop, step;
    bool    inclusive;
};

ObjRange *jaiRangeNew(int64_t start, int64_t stop, int64_t step, bool inclusive);
int64_t   jaiRangeLength(ObjRange *r);

/* ------------------------------------------------------------------ */
/* Functions and closures                                               */
/* ------------------------------------------------------------------ */

typedef enum {
    FN_VARIADIC  = 1 << 0,
    FN_KWREST    = 1 << 1,
    FN_METHOD    = 1 << 2,
    FN_STATIC    = 1 << 3,
    FN_GENERATOR = 1 << 4,
    FN_ASYNC     = 1 << 5,
    FN_GETTER    = 1 << 6,
    FN_SETTER    = 1 << 7,
    FN_INIT      = 1 << 8,
} FunctionFlags;

/* One entry of a function's exception table. */
typedef struct {
    uint32_t start;       /* protected region, code offset, inclusive */
    uint32_t end;         /* exclusive */
    uint32_t handler;     /* code offset of the handler */
    uint32_t typeConst;   /* constant index of the caught class, or UINT32_MAX for catch-all */
} ExceptionEntry;

struct ObjFunction {
    Obj         obj;
    ObjString  *name;
    ObjString  *qualifiedName;   /* "module.Class.method" for tracebacks */
    uint8_t     arity;           /* declared positional params, defaults included */
    uint8_t     defaultCount;    /* trailing params with defaults */
    uint32_t    flags;           /* FunctionFlags */
    uint16_t    maxSlots;
    uint16_t    upvalueCount;
    Chunk       chunk;
    ObjString **paramNames;      /* arity + variadic + kwrest entries */
    uint16_t    paramCount;
    ExceptionEntry *exceptions;
    uint16_t    exceptionCount;
    /* Entries so far, saturating at JAI_JIT_THRESHOLD. The compiled tier reads
     * it to decide when a function is worth compiling; it costs one increment
     * on the call path and stops counting once hot. */
    uint16_t    entryCount;
    /* Entry point of this function's compiled form, or NULL. Owned by the JIT's
     * arena, which outlives every function, so this is not freed here. */
    void       *jitCode;
    /* Which calling convention `jitCode` uses; see jit.c. */
    uint8_t     jitKind;
    /* Compiled whole-function form: a native routine taking int64 arguments
     * and returning one, calling itself directly. See jit_func.c. */
    uint8_t    *jitFunc;
    /* The module's global-mutation counter as it stood when jitFunc was
     * built. Compiled code resolved this function's own name once; if any
     * global has moved since, the compiled form is retired. */
    uint32_t    jitModuleVersion;
    /* What the compiled form was specialised to: the kind of each parameter,
     * the class shape where that kind is an instance, and the kind it returns.
     * The entry guard re-checks every one on every call. */
    uint8_t     jitParamKind[4];
    uint32_t    jitParamShape[4];
    uint8_t     jitReturnKind;
    uint32_t    jitReturnShape;   /* class shape when the kind is an instance */
    uint8_t     jitArgBase;    /* first slot passed in: 0 for a method */
    uint8_t     jitArgCount;
    /* On-stack replacement: a compiled loop entered from the interpreter, with
     * the interpreter's own slots as its locals. This is what reaches a loop
     * in a function that runs once -- `main`, mostly. */
    uint8_t    *osrCode;
    uint32_t    osrTop;        /* bytecode offset of the loop head */
    uint8_t     osrKinds[16];  /* what each slot must hold on entry */
    uint8_t     osrSlots;
    bool        osrRefused;
    /* Set once the tier has looked at this function and refused it. Without it
     * every call past the threshold pays a call into jaiJitEnter to be told no
     * again -- 2.6% of `check lib/std`, on a workload the tier does not help at
     * all. A decline has to be free after the first one, which is the same
     * lesson the loop back edge taught. */
    bool        jitRefused;
    /* Sampling ticks that landed in this function, saturating once hot. */
    uint16_t    tickCount;
    /* A compiled form of one loop in this function, plus the two bytecode
     * offsets the tier hands back to the interpreter: where the loop exits and
     * where it starts. Owned by the JIT arena, so not freed here. */
    void       *jitLoop;
    uint32_t    jitLoopExit;
    uint32_t    jitLoopTop;
    /* The limit the compiled loop runs to. Held here rather than baked into the
     * code so one emitted body can serve a counted head and a range head. */
    int64_t     jitLoopLimit;
    /* 0 = counted head (JUMP_IF_CMP_LOCAL_K), 1 = range head
     * (FOR_ITER_BIND over a 0-start unit-step range). */
    uint8_t     jitLoopKind;
    /* Code offsets of the default-value thunks, indexed from arity-defaultCount. */
    uint32_t   *defaultOffsets;
    ObjModule  *module;          /* defining module, for globals resolution */
    /* The class, trait or enum this function was declared in, or NULL for a
     * free function. Set by OP_METHOD, which runs whether the module came from
     * source or from a cached image, so it needs no place in the image format.
     * Visibility asks for it: slot 0 answers "which class is running" only for
     * a call that went through OP_INVOKE, and a `static fn` reached in tail
     * position arrives with the function there instead. */
    ObjClass   *owner;
};

ObjFunction *jaiFunctionNew(void);

/* A deferred block is compiled as a thunk over the *defining* frame's slot
 * numbering and upvalue indices (spec §5.4: it reads and writes that
 * function's locals). Nothing may renumber its slots, and the VM enters it
 * with the definer's window. `defer` is a keyword, so the name is unambiguous:
 * no user function can be called this. */
bool jaiFunctionIsDeferThunk(const ObjFunction *fn);

struct ObjUpvalue {
    Obj         obj;
    Value      *location;   /* points into the stack while open */
    Value       closed;     /* holds the value once closed */
    ObjUpvalue *next;       /* sorted open-upvalue list */
};

ObjUpvalue *jaiUpvalueNew(Value *slot);
/* An upvalue that is closed from the start, holding a copy of `v`. This is how
 * a `let` is captured (spec §6): the closure keeps the value the binding had
 * when it was built, not the slot it lived in. */
ObjUpvalue *jaiUpvalueClosed(Value v);

struct ObjClosure {
    Obj          obj;
    ObjFunction *fn;
    ObjUpvalue **upvalues;
    int          upvalueCount;
};

ObjClosure *jaiClosureNew(ObjFunction *fn);

/* Native functions. `argc` is checked by the VM against arity before the call.
 * Return false and set the pending exception (jaiThrow*) to signal an error. */
typedef bool (*JaiNativeFn)(int argc, Value *args, Value *out);

struct ObjNative {
    Obj          obj;
    JaiNativeFn  fn;
    ObjString   *name;
    int8_t       minArity;
    int8_t       maxArity;   /* -1 = variadic */
    /* Parameter names, parallel to args[0..maxArity-1] and so beginning with
     * the receiver for a method. NULL when the native declares none, which is
     * what makes `f(key: v)` on it a TypeError instead of a silent misbind.
     * Points at static storage owned by the method table. */
    const char *const *paramNames;
};

ObjNative *jaiNativeNew(JaiNativeFn fn, const char *name, int minArity, int maxArity,
                        const char *const *paramNames);

struct ObjBound {
    Obj    obj;
    Value  receiver;
    Value  method;      /* ObjClosure or ObjNative */
};

ObjBound *jaiBoundNew(Value receiver, Value method);

/* ------------------------------------------------------------------ */
/* Classes, traits, instances                                           */
/* ------------------------------------------------------------------ */

typedef enum { VIS_PRIVATE = 0, VIS_PROTECTED = 1, VIS_PUBLIC = 2 } Visibility;

typedef struct {
    ObjString *name;
    uint16_t   slot;        /* index into ObjInstance.fields */
    Visibility visibility;
    bool       isStatic;
    bool       isLet;       /* immutable after init */
    uint32_t   typeId;      /* index into the type registry, 0 = any */
} FieldInfo;

/* What jaiClassRestrictedMethod reports about a non-public method. The method
 * value itself is not here: every caller already has it, or wants only the
 * verdict, and finding it again would mean probing four tables. */
typedef struct {
    ObjString      *name;
    Visibility      visibility;
    uint32_t        flags;       /* FunctionFlags */
    const ObjClass *owner;       /* declaring class, for the private test */
} MethodInfo;

struct ObjClass {
    Obj         obj;
    ObjString  *name;
    ObjString  *qualifiedName;
    ObjClass   *superclass;
    uint32_t    shapeId;        /* unique, monotonically assigned; inline-cache key */
    FieldInfo  *fields;         /* instance fields, parents first */
    uint16_t    fieldCount;
    JaiTable    methods;        /* ObjString* -> Value (method), includes inherited */
    JaiTable    statics;        /* ObjString* -> Value */
    JaiTable    getters;        /* ObjString* -> Value */
    JaiTable    setters;        /* ObjString* -> Value */
    /* The non-public methods, name -> INT_VAL(vis | flags<<8 | ownerShape<<24),
     * inherited entries copied down like the method tables themselves. Only
     * non-public methods appear, so `restricted.count == 0` — every class in
     * the common case — settles the runtime visibility test for a whole class
     * with one load, instead of a hash probe on every dispatch. A parallel
     * MethodInfo array would have cost a linear scan per lookup on the hottest
     * path in the VM. The declaring class travels as its shapeId so that
     * finding it is a walk up the superclass pointers with no hashing. */
    JaiTable    restricted;
    ObjTrait  **traits;
    uint16_t    traitCount;
    Value       initializer;    /* the `init` closure, or NULL_VAL */
    bool        isAbstract;
    /* Cached dunder lookups; NULL_VAL if absent. Filled at class creation. */
    Value       dunderStr, dunderRepr, dunderEq, dunderLt, dunderHash;
    Value       dunderAdd, dunderSub, dunderMul, dunderDiv, dunderMod, dunderPow;
    Value       dunderNeg, dunderLen, dunderGetItem, dunderSetItem;
    Value       dunderContains, dunderIter, dunderNext, dunderCall;
};

ObjClass *jaiClassNew(ObjString *name, ObjClass *superclass);
/* Copies down parent fields/methods and assigns field slots. Call once, after
 * the superclass link is set and before methods are added. */
void      jaiClassInherit(ObjClass *sub, ObjClass *super);
void      jaiClassAddMethod(ObjClass *c, ObjString *name, Value method,
                            Visibility vis, uint32_t flags);
/* Fills *out and returns true when `name` names a method of `c` that is not
 * public; false — the answer for every public or absent name — otherwise.
 * The VM calls this before handing a method out to a caller. */
bool      jaiClassRestrictedMethod(ObjClass *c, ObjString *name, MethodInfo *out);
int       jaiClassFieldSlot(ObjClass *c, ObjString *name);   /* -1 if absent */
const FieldInfo *jaiClassFieldInfo(ObjClass *c, ObjString *name);
bool      jaiClassIsSubclassOf(const ObjClass *sub, const ObjClass *super);
bool      jaiClassImplements(const ObjClass *c, const ObjTrait *t);
/* Recomputes the dunder cache; call after any method mutation. */
void      jaiClassRefreshDunders(ObjClass *c);

struct ObjTrait {
    Obj        obj;
    ObjString *name;
    JaiTable   required;     /* name -> INT_VAL(arity) */
    JaiTable   defaults;     /* name -> default method */
    ObjTrait **supers;
    uint16_t   superCount;
};

ObjTrait *jaiTraitNew(ObjString *name);

struct ObjInstance {
    Obj       obj;
    ObjClass *klass;
    uint16_t  fieldCount;
    Value     fields[];      /* inline, indexed by FieldInfo.slot */
};

ObjInstance *jaiInstanceNew(ObjClass *klass);

/* ------------------------------------------------------------------ */
/* Enums                                                                */
/* ------------------------------------------------------------------ */

typedef struct {
    ObjString  *name;
    uint8_t     arity;
    ObjString **fieldNames;
    /* The one value of a payload-less variant, made on first mention and
     * shared from then on. A variant with no payload has no state to tell two
     * instances apart, so `Color.Red is Color.Red` has to hold — `is` is
     * identity (spec §4.2) — and there is no reason to allocate twice.
     * NULL for a variant that takes a payload: those are built per call. */
    ObjEnumVal *unit;
    /* The callable form, for a variant that does take a payload. Cached for
     * the same reason: `Shape.Circle is Shape.Circle` should hold. */
    ObjEnumCtor *ctor;
} EnumVariant;

struct ObjEnum {
    Obj          obj;
    ObjString   *name;
    EnumVariant *variants;
    uint16_t     variantCount;
    JaiTable     methods;
    /* Inline-cache key, from the same counter ObjClass.shapeId uses so that an
     * enum way and a class way in one cache can never collide. Monotonic, so a
     * freed enum whose address gets reused cannot be mistaken for the original
     * -- which is what makes caching members by identity safe at all. */
    uint32_t     shapeId;
};

struct ObjEnumVal {
    Obj      obj;
    ObjEnum *type;
    uint16_t tag;
    uint8_t  count;
    Value    payload[];
};

/* The callable form of a variant that takes a payload.
 *
 * `Shape.Circle` on its own is not a value the way `Color.Red` is — it still
 * needs its arguments. It is a *function* of them, which is how the checker
 * types it, so it has to be one at run time too: `let make = Shape.Circle`
 * and `radii.map(Shape.Circle)` are ordinary code. Cached on the variant, so
 * two mentions give the same object. */
struct ObjEnumCtor {
    Obj      obj;
    ObjEnum *type;
    uint16_t tag;
};

ObjEnum     *jaiEnumNew(ObjString *name);
/* A fresh shape id, for invalidating caches that memoised an enum's members. */
uint32_t     jaiFreshShapeId(void);
ObjEnumVal  *jaiEnumValNew(ObjEnum *e, uint16_t tag, const Value *payload, int count);
ObjEnumCtor *jaiEnumCtorNew(ObjEnum *e, uint16_t tag);

/* ------------------------------------------------------------------ */
/* Modules                                                              */
/* ------------------------------------------------------------------ */

typedef enum {
    MOD_UNLOADED, MOD_LOADING, MOD_LOADED, MOD_FAILED
} ModuleState;

struct ObjModule {
    Obj          obj;
    ObjString   *name;         /* dotted: "std.math" */
    ObjString   *path;         /* absolute filesystem path */
    JaiTable     globals;      /* ObjString* -> Value */
    JaiTable     exports;      /* ObjString* -> BOOL_VAL(true) */
    uint32_t     version;      /* bumped on every global mutation; inline-cache key */
    ModuleState  state;
    ObjClosure  *body;
    int          sourceFileId; /* JaiSourceFile.id for diagnostics */
};

ObjModule *jaiModuleNew(ObjString *name, ObjString *path);
bool       jaiModuleGet(ObjModule *m, ObjString *name, Value *out);
void       jaiModuleSet(ObjModule *m, ObjString *name, Value v);
bool       jaiModuleIsExported(ObjModule *m, ObjString *name);

/* ------------------------------------------------------------------ */
/* Iterators                                                            */
/* ------------------------------------------------------------------ */

/* ITER_USER drives spec §7.1's `__next__` dunder, which ends on StopIteration;
 * ITER_TRAIT drives std.core's `trait Iterator`, whose `next` ends by returning
 * null. Both are user objects, and a class may implement either. */
typedef enum {
    ITER_LIST, ITER_TUPLE, ITER_STRING, ITER_DICT_KEYS, ITER_DICT_ITEMS,
    ITER_SET, ITER_RANGE, ITER_USER, ITER_TRAIT, ITER_GENERATOR
} IterKind;

struct ObjIter {
    Obj      obj;
    IterKind kind;
    Value    source;
    int64_t  index;
    int64_t  limit;      /* also the snapshot element count, for mutable sources */
    uint32_t version;    /* snapshot of container version; detects mutation */
};

ObjIter *jaiIterNew(IterKind kind, Value source);
/* Advance. Returns false when exhausted (no exception). Sets *out otherwise.
 * Raises RuntimeError if the underlying container was mutated. */
bool     jaiIterNext(ObjIter *it, Value *out);
/* Produce an iterator for any iterable, calling __iter__ if needed. */
bool     jaiGetIter(Value v, Value *out);

/* ------------------------------------------------------------------ */
/* Files                                                                */
/* ------------------------------------------------------------------ */

struct ObjFile {
    Obj        obj;
    FILE      *handle;
    ObjString *path;
    bool       readable, writable, binary, closed;
};

ObjFile *jaiFileNew(FILE *handle, ObjString *path, const char *mode);

/* ------------------------------------------------------------------ */
/* Allocation and lifetime                                              */
/* ------------------------------------------------------------------ */

/* Allocate a GC-tracked object of `size` bytes with the given type tag. */
Obj  *jaiAllocateObject(size_t size, ObjType type);
/* Header only: the caller must initialise every remaining field, including any
 * the type would otherwise have got for free from the zeroing above. */
Obj  *jaiAllocateObjectRaw(size_t size, ObjType type);
#define JAI_ALLOCATE_OBJ(type, objType)                                        \
    ((type *)jaiAllocateObject(sizeof(type), objType))

void  jaiFreeObject(Obj *obj);
const char *jaiObjTypeName(ObjType t);

/* ------------------------------------------------------------------ */
/* Hashing                                                              */
/* ------------------------------------------------------------------ */

/* jaiValueHash with the one case that dominates real dictionaries — a string
 * key — settled inline, against an out-of-line call that maintains the
 * recursion guard and switches twice. Everything else defers, so there is
 * exactly one line here to keep in step with valueHashInner.
 *
 * It must go through jaiStringHash and not read `->hash`. The field is lazy:
 * jaiStringSeal leaves it at zero for any run-time string past
 * JAI_INTERN_MAX, and zero means "not computed yet". Reading it raw filed
 * every such key under hash 0, and the first later call that did force the
 * hash rewrote the field on the key the table was already holding — so the
 * entry became unreachable through the very object keys() hands back, and
 * `for k in d.keys() { k in d }` could be false. That is the line this comment
 * used to claim did not need to exist. */
JAI_INLINE uint64_t jaiValueHashFast(Value v, bool *ok) {
    if (IS_STRING(v)) {
        *ok = true;
        return jaiStringHash(AS_STRING(v));
    }
    return jaiValueHash(v, ok);
}

#endif /* JAI_OBJECT_H */
