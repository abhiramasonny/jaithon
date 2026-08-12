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
     * cursorScalar. Any byte above 127 makes finding an offset require
     * decoding, so a lexer walking forward one scalar at a time would
     * otherwise rescan from the start every step. Zero/zero always starts
     * valid; this is a pure cache, never changing what the string is. */
    uint32_t cursorScalar;
    uint32_t cursorByte;
    uint64_t hash;
    /* `interned` lives in Obj.subFlag: a bool of its own here would be padded
     * out to eight bytes ahead of the flexible array. Use the accessor. */
    /* A pointer, not a flexible array, so a string can address bytes it does
     * not own -- a slice of a shared append buffer -- without touching the
     * hundreds of places that read `s->chars`. Ordinary strings are still
     * NUL-terminated; one whose terminator got overwritten by a later append
     * into the same buffer is flagged, and jaiStringCStr is the only safe way
     * to get a C string. */
    char    *chars;
    /* The buffer these bytes live in, or NULL when the string owns them.
     * Several strings share one buffer while a concatenation chain grows: an
     * append only ever writes past every existing string's end, so no string's
     * bytes ever change under it. */
    struct ObjStrBuf *owner;
};

/* Bytes to allocate for a string of `length` characters: header, bytes, NUL --
 * the header ends on an eight-byte boundary, so nothing is wasted between. */
#define JAI_STRING_ALLOC(length) (sizeof(ObjString) + (size_t)(length) + 1)

#define JAI_STR_INTERNED(s)      ((s)->obj.subFlag)
/* True when a later append overwrote the NUL that used to sit at this string's
 * end. The bytes are still correct for `length`; only C-string use is unsafe. */
#define JAI_STR_UNTERMINATED(s)  ((s)->obj.subFlag2)

/* A growable byte buffer shared by a chain of concatenation results. */
typedef struct ObjStrBuf {
    Obj      obj;
    uint32_t capacity;   /* bytes in `data`, excluding the NUL slot */
    uint32_t used;       /* bytes written; data[used] is the live NUL */
    char     data[];
} ObjStrBuf;

/* A NUL-terminated view of `s`, copying only when a later append overwrote the
 * terminator. Use this anywhere a bare `char *` is handed to printf or str*. */
const char *jaiStringCStr(ObjString *s);

/* Every string literal and identifier the compiler or deserialiser produces
 * goes through the intern table, so for interned strings pointer equality is
 * string equality -- what lets member/global lookup use jaiTableGetInterned.
 * Run-time strings are *not* interned; equality on them is always by content. */
ObjString *jaiStringIntern(const char *chars, size_t length);
ObjString *jaiStringInternC(const char *cstr);
/* The interned twin of `s`, interning `s` itself if none exists. Needed before
 * using a run-time string as a key into a pointer-keyed table -- the
 * reflective paths only (get_field, module.get, exec's namespace), since
 * every compiler-emitted name arrives interned already. */
ObjString *jaiStringCanonical(ObjString *s);
/* Not interned. The constructor for any string that is data rather than a name. */
ObjString *jaiStringNew(const char *chars, size_t length);
/* Takes ownership of a heap buffer allocated with jaiRealloc. */
ObjString *jaiStringTake(char *chars, size_t length);
ObjString *jaiStringConcat(ObjString *a, ObjString *b);
/* The shared one-byte ASCII string. NULL for c >= 128. */
ObjString *jaiStringChar(unsigned char c);
/* The 128 one-byte strings, addressable as an array so compiled code can index
 * it directly. An entry is NULL until first asked for. */
ObjString **jaiAsciiCharTable(void);
void       jaiMarkAsciiChars(void);
/* Concatenates `count` byte runs into one string, sized exactly, under the
 * same run-time interning policy jaiStringNew applies. The runs must stay
 * valid across a collection — allocating the result can trigger one. */
ObjString *jaiStringFromParts(const char *const *runs, const uint32_t *lens,
                              int count, size_t total);
/* An uninitialised string of exactly `length` bytes: the caller fills chars[]
 * (without allocating, since it's not yet rooted) then hands it to
 * jaiStringSeal, which applies the interning policy and returns the string to
 * use (an existing equal one, if interning found it). */
ObjString *jaiStringReserve(size_t length);
ObjString *jaiStringSeal(ObjString *s);
ObjString *jaiStringSlice(ObjString *s, int64_t start, int64_t stop, int64_t step);
uint32_t   jaiStringScalarCount(ObjString *s);
/* The cached content hash, computed on first use, so a string that never
 * becomes a dict key or takes part in interning is never hashed at all (vs.
 * hashing eagerly, which meant walking e.g. a 2.2 MB join the program only
 * prints). Zero doubles as "not computed"; a real zero hash just recomputes. */
static inline uint64_t jaiStringHash(ObjString *s) {
    if (s->hash == 0) s->hash = jaiHashBytes(s->chars, s->length);
    return s->hash;
}
/* The tail of jaiStringEquals: two strings that are neither identical nor both
 * interned, so the answer needs their bytes. */
bool       jaiStringEqualsSlow(const ObjString *a, const ObjString *b);

/* Inline because the callers are linear scans over parameter and field names
 * (vm.c bindCallArgs, jaiClassFieldInfo) run tens of millions of times per
 * compile, answered by the two cheap tests below for interned names. Out of
 * line, this was the second hottest function in the program by sample count
 * -- almost entirely the cost of the call, not of the work. */
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
/* Records a mutation the list functions above did not perform (an in-place
 * store through `items`, a sort, a direct write to `count`); any such site on
 * a list the program can already reach must call this or a live iterator
 * won't see the change. Exempt while the list hasn't escaped yet -- nothing
 * can be iterating it. */
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

#define JAI_OSR_MAX 4

/* A compiled loop: where it starts, and what each slot must hold to enter. */
typedef struct {
    uint8_t  *code;
    uint32_t  top;
    uint8_t   slots;
    /* 0 no iterator, 1 a unit-step range, 2 a list. A form compiled for one
     * must never be entered with the other: the prologue reads a different
     * object out of ObjIter.source for each. */
    uint8_t   iterKind;
    uint8_t   kinds[40];   /* what each slot must hold on entry */
} JaiOsrForm;

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
    /* Entries so far, saturating at JAI_JIT_THRESHOLD: read by the compiled
     * tier to decide when a function is worth compiling. */
    uint16_t    entryCount;
    /* Entry point of this function's compiled form, or NULL. Owned by the JIT's
     * arena, which outlives every function, so this is not freed here. */
    void       *jitCode;
    /* Which calling convention `jitCode` uses; see jit.c. */
    uint8_t     jitKind;
    /* Compiled whole-function form: a native routine taking int64 arguments
     * and returning one, calling itself directly. See jit_func.c. */
    uint8_t    *jitFunc;
    /* The module's global-mutation counter as it stood when each tier's form
     * was built; a form is retired once a binding it resolved at compile time
     * (class/closure/native) may have moved. NOT bumped by a write that merely
     * updates a global to another inert value -- see jaiModuleSet and
     * jaiValueIsInertGlobal.
     *
     * ONE PER TIER, and that is load-bearing: a shared field once let
     * compileOsr re-arm a whole-function form that a rebinding had already
     * retired, resurrecting stale baked callees. See tests/jit/rearm. */
    uint32_t    jitFuncModuleVersion;
    uint32_t    jitOsrModuleVersion;
    /* A builtin resolved at compile time is pinned by the builtins module's
     * own version, the same way a module global is pinned by its module's. */
    uint32_t    jitBuiltinsVersion;
    /* What the compiled form was specialised to: each parameter's kind, the
     * class shape when that kind is an instance, and the return kind. The
     * entry guard re-checks every one on every call. */
    uint8_t     jitParamKind[4];
    uint32_t    jitParamShape[4];
    uint8_t     jitReturnKind;
    uint32_t    jitReturnShape;   /* class shape when the kind is an instance */
    uint8_t     jitArgBase;    /* first slot passed in: 0 for a method */
    uint8_t     jitArgCount;
    /* The compiled whole-function form never stores to the heap, so re-running
     * it from the top is indistinguishable from not having run it -- which a
     * caller that branched straight to its entry depends on: it can't hand the
     * interpreter a half-finished callee frame, so a nonzero verdict must be
     * answered by re-executing the whole call. */
    bool        jitFuncNoWrite;
    /* On-stack replacement: a compiled loop entered from the interpreter, with
     * the interpreter's own slots as its locals -- what reaches a loop in a
     * function that runs once (`main`, mostly). One form per loop head, not
     * per function: a second offset meeting `osrTop != top` is refused, so
     * e.g. `sieve`'s `main` only ever compiles its first of four loops. */
    JaiOsrForm  osrForms[JAI_OSR_MAX];
    uint8_t     osrCount;
    /* Attempts so far. A body can only use a callee's return kind once that
     * callee has itself compiled, which depends on tick timing, so one look
     * isn't enough and refusing forever on the first miss made compilation
     * timing-dependent.
     *
     * COUNTED PER LOOP HEAD: a single counter is a starvation bug, not just
     * imprecision -- whichever loop ran first could spend the whole budget
     * and get every OTHER loop in the body refused for the program's life
     * with no decline ever reported (real case: word_freq's concatenation
     * loop, which the tier can't compile, burning all 20 attempts before the
     * 64%-of-benchmark scan loop got to run once). osrMissTop/osrMissAttempts
     * split that budget by head; osrRefused means every head the table can
     * hold is spent, and osrAttempts is the whole-function backstop for more
     * uncompilable heads than the table has room for. */
    uint8_t     osrAttempts;
    uint32_t    osrMissTop[JAI_OSR_MAX];
    uint8_t     osrMissAttempts[JAI_OSR_MAX];
    uint8_t     osrMissCount;
    uint8_t     jitAttempts;
    bool        osrRefused;
    /* The back edge enters the compiled loop directly. An entry that keeps
     * declining (slots aren't the kinds it was compiled for) would otherwise
     * pay a call per iteration for nothing -- cost `sieve` 20%. After a few in
     * a row the back edge stops trying; the timer tick is the only way back in. */
    bool        osrHot;
    uint8_t     osrDeclines;
    /* Set once the tier has looked at this function and refused it. Without
     * it, every call past the threshold pays a call into jaiJitEnter to be
     * told no again -- 2.6% of `check lib/std`, for no benefit. A decline has
     * to be free after the first one, the same lesson the loop back edge taught. */
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
     * free function. Set by OP_METHOD, which runs regardless of source vs.
     * cached image, so needs no place in the image format. Visibility needs
     * it because slot 0 answers "which class is running" only for a call
     * through OP_INVOKE; a `static fn` reached in tail position arrives with
     * the function here instead. */
    ObjClass   *owner;
};

ObjFunction *jaiFunctionNew(void);

/* A deferred block is compiled as a thunk over the *defining* frame's slot
 * numbering and upvalue indices (spec §5.4: it reads/writes that function's
 * locals) -- nothing may renumber its slots, and the VM enters it with the
 * definer's window. `defer` is a keyword, so no user function can be named this. */
bool jaiFunctionIsDeferThunk(const ObjFunction *fn);

struct ObjUpvalue {
    Obj         obj;
    Value      *location;   /* points into the stack while open */
    Value       closed;     /* holds the value once closed */
    ObjUpvalue *next;       /* sorted open-upvalue list */
};

ObjUpvalue *jaiUpvalueNew(Value *slot);
/* An upvalue closed from the start, holding a copy of `v` -- how a `let` is
 * captured (spec §6): the closure keeps the value the binding had when built,
 * not the slot it lived in. */
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
    /* Parameter names, parallel to args[0..maxArity-1], beginning with the
     * receiver for a method. NULL when the native declares none, making
     * `f(key: v)` on it a TypeError instead of a silent misbind. Points at
     * static storage owned by the method table. */
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
    /* The non-public methods, name -> INT_VAL(vis | flags<<8 | ownerShape<<24).
     * Only non-public methods appear, so `restricted.count == 0` (the common
     * case) settles the visibility test for a whole class with one load
     * instead of a hash probe on every dispatch -- a parallel MethodInfo array
     * would cost a linear scan on the hottest path in the VM. The declaring
     * class travels as its shapeId, so finding it needs no hashing. */
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
     * shared from then on: with no state to tell two instances apart,
     * `Color.Red is Color.Red` must hold (`is` is identity, spec §4.2).
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
    /* Inline-cache key, from the same counter ObjClass.shapeId uses, so an
     * enum way and a class way in one cache can never collide. Monotonic, so
     * a freed enum whose address gets reused can't be mistaken for the
     * original -- what makes caching members by identity safe at all. */
    uint32_t     shapeId;
};

struct ObjEnumVal {
    Obj      obj;
    ObjEnum *type;
    uint16_t tag;
    uint8_t  count;
    Value    payload[];
};

/* The callable form of a variant that takes a payload: `Shape.Circle` alone
 * isn't a value the way `Color.Red` is, it's a *function* of its arguments
 * (how the checker types it), so `let make = Shape.Circle` and
 * `radii.map(Shape.Circle)` need to be ordinary code. Cached on the variant,
 * so two mentions give the same object. */
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
    /* "Can compiled code still trust the bindings it baked in?" jit_func.c
     * resolves a class/closure/native at compile time and calls it directly,
     * so a rebinding must retire the compiled form; this moves on such a
     * write and NOT on one that merely updates a global to another inert
     * value (see jaiModuleSet, jaiValueIsInertGlobal).
     *
     * A NEW READER MUST PICK THE RIGHT COUNTER: anything memoising a global's
     * VALUE or a resolved callee keys on this; anything memoising a table
     * slot's address or a name's absence keys on globals.keyVersion instead. */
    uint32_t     version;
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

/* ITER_USER drives spec §7.1's `__next__` dunder, ending on StopIteration;
 * ITER_TRAIT drives std.core's `trait Iterator`, whose `next` ends by
 * returning null. Both are user objects; a class may implement either. */
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

/* jaiValueHash with the one case that dominates real dictionaries -- a string
 * key -- settled inline, against an out-of-line call that maintains the
 * recursion guard and switches twice.
 *
 * Must go through jaiStringHash and not read `->hash` directly: the field is
 * lazy (zero means "not computed"), and reading it raw filed every such key
 * under hash 0 -- then the first later call that forced the hash rewrote the
 * field on the key the table was already holding, making that entry
 * unreachable through the very object keys() hands back. */
JAI_INLINE uint64_t jaiValueHashFast(Value v, bool *ok) {
    if (IS_STRING(v)) {
        *ok = true;
        return jaiStringHash(AS_STRING(v));
    }
    return jaiValueHash(v, ok);
}

#endif /* JAI_OBJECT_H */
