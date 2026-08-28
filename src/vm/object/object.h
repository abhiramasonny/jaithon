/* object.h — heap object layouts and constructors. */
#ifndef JAI_OBJECT_H
#define JAI_OBJECT_H

#include "vm/value.h"
#include "vm/bytecode/chunk.h"
#include "vm/table.h"

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
    /* A pointer, not a flexible array. For an ordinary string it addresses the
     * bytes immediately after this header, which is the same layout the
     * flexible array had plus one word. Making it a pointer is what lets a
     * string address bytes it does not own -- a slice of a shared append
     * buffer -- without changing any of the several hundred places that read
     * `s->chars`. Ordinary strings are still NUL-terminated; a string whose
     * terminator was overwritten by a later append into the same buffer is
     * flagged, and jaiStringCStr is the only safe way to get a C string. */
    char    *chars;
    /* The buffer these bytes live in, or NULL when the string owns them.
     * Several strings share one buffer while a concatenation chain grows: an
     * append only ever writes past every existing string's end, so no string's
     * bytes ever change under it. */
    struct ObjStrBuf *owner;
};

/* Bytes to allocate for a string of `length` characters: the header, then the
 * bytes, then a NUL. The header now ends on an eight-byte boundary, so the
 * bytes start immediately after it with nothing wasted between. */
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
/* The rooting-safe half of jaiStringCStr: the same string when it is already
 * terminated, otherwise a fresh terminated copy the caller can root. */
ObjString *jaiStringTerminated(ObjString *s);

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
/* The shared one-byte ASCII string. NULL for c >= 128. */
ObjString *jaiStringChar(unsigned char c);
/* Fills all 128 slots, so that from the end of jaiVMInit onwards
 * `jaiAsciiCharTable()[c]` for c < 128 is never NULL. That is a promise three
 * readers rely on to drop a branch: the interpreter's `s[i]`, the string
 * iterator, and the JIT's OP_GET_INDEX arm, which would otherwise have to
 * deopt the first time a program met a character. 128 objects, ~7KB, once.
 * jaiAsciiCharsReset undoes it: the slots hold raw pointers into the heap the
 * collector is about to free, so a VM teardown must not leave them dangling. */
void       jaiAsciiCharsFill(void);
void       jaiAsciiCharsReset(void);
/* Backing storage for the 128 one-byte ASCII strings (object_string.c owns
 * writing to it: jaiStringChar fills a slot on first use, jaiMarkAsciiChars
 * keeps filled slots alive across a collection). External linkage rather than
 * file-static because jaiAsciiCharTable below has to reach it from any
 * translation unit at zero cost -- the string iterator's per-character fast
 * path (object_iter.c) reads one slot per scalar of a `for c in s` loop, and
 * a real function call there was measurable on tests/bench/str_search. */
extern ObjString *jaiAsciiChars[128];
/* The 128 one-byte strings, addressable as an array so compiled code (and the
 * string iterator) can index it directly. An entry is NULL until first asked
 * for. `static inline` so every caller, including cross-TU ones, compiles
 * this down to the bare array access it used to be when both sides lived in
 * the same file. */
static inline ObjString **jaiAsciiCharTable(void) { return jaiAsciiChars; }
void       jaiMarkAsciiChars(void);
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
/* List — growable element array, boxed or not                          */
/* ------------------------------------------------------------------ */

/* How the backing array is laid out. A Value is sixteen bytes of {tag,
 * payload}, so a boxed `list[float]` spends half its cache lines on a tag that
 * `list[float]` already proved, loads and compares that tag once per element,
 * re-stamps it on every store, and addresses the array with `lsl #4`. The
 * declared element kind is a promise the mutation guards already enforce on
 * every entry point (jaiListPush, indexSet, jaiListInsert), so for the three
 * scalar kinds the tag carries no information at all and the payload can be
 * stored bare: eight bytes for an int or a float, one for a bool.
 *
 * BOXED is zero so that a list nobody typed -- and any list that has been
 * handed out as a `Value *` -- is the representation it always was. */
typedef enum {
    LIST_STORE_BOXED = 0,   /* Value[]   */
    LIST_STORE_I64   = 1,   /* int64_t[] */
    LIST_STORE_F64   = 2,   /* double[]  */
    LIST_STORE_U8    = 3,   /* uint8_t[], one byte per bool -- see jaiListStoreWidth */
} ListStore;

/* Not a storage: what JaiOsrForm records for a list slot whose storage the
 * compile did not pin, so the entry guard has nothing to prove about it. */
#define LIST_STG_ANY 0x0Fu

/* Bytes per element. A byte per bool rather than a bit: a bit needs a shift, a
 * mask and a read-modify-write per store, and the sieve was measured at 7.2 ms
 * over a byte array against 36 ms over std::vector<bool>. */
JAI_INLINE size_t jaiListStoreWidth(uint8_t stg) {
    switch ((ListStore)stg) {
    case LIST_STORE_I64: return sizeof(int64_t);
    case LIST_STORE_F64: return sizeof(double);
    case LIST_STORE_U8:  return 1;
    case LIST_STORE_BOXED: break;
    }
    return sizeof(Value);
}

struct ObjList {
    Obj      obj;
    /* Value[], int64_t[], double[] or uint8_t[] as `stg` says. Untyped, so
     * that every one of the two hundred sites that used to walk it as a
     * `Value *` had to be looked at rather than silently reading a double as a
     * tag -- jaiListBox is the escape hatch for the ones that still want the
     * boxed array, and it de-specialises the list to give them one. */
    void    *items;
    int      count;
    int      capacity;
    uint32_t version;   /* bumped on every mutation; iterators snapshot it */
    /* What `list[T]` promised, as a FieldKind, or FIELD_KIND_ANY when nothing
     * was promised. A declared element type is otherwise a compile-time fact
     * only, so a list reaching a function through `any` could be pushed a `str`
     * with no diagnostic at all -- the checker cannot see it, because with an
     * `any` receiver it does not know what the list was declared to hold.
     *
     * Lands in the struct's existing tail padding, so a list costs no more
     * than it did. */
    uint8_t  elemKind;
    /* A ListStore. Set once, from elemKind, while the list is still empty, and
     * only ever changed in the one direction -- back to BOXED, by jaiListBox.
     * The tier reads it and deoptimises on anything but the storage its guards
     * were emitted for, so a de-specialisation cannot be missed. */
    uint8_t  stg;
};

/* Boxes element `i`. No bounds check: every caller has already normalised. */
JAI_INLINE Value jaiListGet(const ObjList *l, int i) {
    switch ((ListStore)l->stg) {
    case LIST_STORE_I64: return INT_VAL(((const int64_t *)l->items)[i]);
    case LIST_STORE_F64: return FLOAT_VAL(((const double *)l->items)[i]);
    case LIST_STORE_U8:  return BOOL_VAL(((const uint8_t *)l->items)[i] != 0);
    case LIST_STORE_BOXED: break;
    }
    return ((const Value *)l->items)[i];
}

/* Whether `v` can be stored as it is, without the store changing what the
 * program reads back.
 *
 * jaiCheckKind is not this question. FIELD_KIND_FLOAT accepts an int, so
 * `let fs: list[float] = []` followed by `fs.push(3)` passes it -- but an F64
 * store would widen that 3 to a 3.0 and `print(fs)` would say `[3.0]` where
 * every other tier says `[3]`. tests/golden/container_elem_kind pins exactly
 * that line, and a storage choice is not allowed to move it: the list
 * de-specialises instead, which costs the unboxing for a float list that is
 * handed an int and costs nothing at all for one that is not. */
JAI_INLINE bool jaiListStoreAccepts(const ObjList *l, Value v) {
    switch ((ListStore)l->stg) {
    case LIST_STORE_I64: return IS_INT(v);
    case LIST_STORE_F64: return IS_FLOAT(v);
    case LIST_STORE_U8:  return IS_BOOL(v);
    case LIST_STORE_BOXED: break;
    }
    return true;
}

/* Unboxes `v` into element `i`. The caller must already have run the value past
 * jaiListStoreAccepts, which is what proves it fits the store. */
JAI_INLINE void jaiListSetRaw(ObjList *l, int i, Value v) {
    switch ((ListStore)l->stg) {
    case LIST_STORE_I64: ((int64_t *)l->items)[i] = AS_INT(v);      return;
    case LIST_STORE_F64: ((double  *)l->items)[i] = jaiAsDouble(v); return;
    case LIST_STORE_U8:  ((uint8_t *)l->items)[i] = AS_BOOL(v);     return;
    case LIST_STORE_BOXED: break;
    }
    ((Value *)l->items)[i] = v;
}

/* The boxed array, de-specialising the list if it is not already boxed.
 *
 * Every consumer that wants to memcpy the elements, hand them to qsort, or
 * walk them as Values comes through here. The de-specialisation is permanent
 * and deliberate: the pointer outlives the call, and a list that went back to
 * boxed cannot surprise anything holding one. Returns NULL only when the list
 * is empty and has never been reserved. */
Value   *jaiListBox(ObjList *list);
/* Chooses the storage a `list[T]` annotation asks for, on a list that is still
 * empty. Anything else -- an unannotated list, a list of strings, a list that
 * already holds something -- stays boxed. */
void     jaiListSpecialise(ObjList *list, uint8_t elemKind);
/* Whether jaiListSpecialise will do anything. The tier asks before emitting
 * the storage store an empty list literal gets. */
bool     jaiListUnboxOn(void);

/* An in-place store from runtime code that has NOT run the value past
 * jaiCheckKind -- a reverse, a sort, a fill. It cannot throw, so a value the
 * store cannot hold de-specialises the list rather than refusing it: a
 * `list[int]` reached through `any` and written a string is a diagnostic the
 * interpreter's own path raises, and this one must not lose the value on the
 * way. The caller still owes jaiListTouch. */
JAI_INLINE void jaiListPut(ObjList *l, int i, Value v) {
    switch ((ListStore)l->stg) {
    case LIST_STORE_I64: if (!IS_INT(v))    break; goto raw;
    case LIST_STORE_F64: if (!IS_FLOAT(v))  break; goto raw;
    case LIST_STORE_U8:  if (!IS_BOOL(v))   break; goto raw;
    case LIST_STORE_BOXED: ((Value *)l->items)[i] = v; return;
    }
    jaiListBox(l);
    ((Value *)l->items)[i] = v;
    return;
raw:
    jaiListSetRaw(l, i, v);
}

ObjList *jaiListNew(int initialCapacity);
void     jaiListPush(ObjList *list, Value v);
Value    jaiListPop(ObjList *list);
void     jaiListInsert(ObjList *list, int index, Value v);
Value    jaiListRemove(ObjList *list, int index);
void     jaiListReserve(ObjList *list, int capacity);
/* Room for exactly `count` elements, ready to be written through `items`.
 *
 * The alternative is pushing them one at a time, which re-checks capacity per
 * element; over the millions a downloaded image or tensor runs to, that check
 * is most of the cost. False means the reserve did not fit, and the caller
 * should give up rather than write past the end. Roots the list across the
 * reserve, since growing the backing array can collect. */
bool     jaiListReserveExact(ObjList *list, int count);
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
    /* What `dict[K, V]` promised, as FieldKinds. See ObjList::elemKind. */
    uint8_t  keyKind;
    uint8_t  valKind;
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
    FN_TRACE       = 1 << 9,
    FN_GPU_KERNEL  = 1 << 10,
} FunctionFlags;

/* One entry of a function's exception table. */
typedef struct {
    uint32_t start;       /* protected region, code offset, inclusive */
    uint32_t end;         /* exclusive */
    uint32_t handler;     /* code offset of the handler */
    uint32_t typeConst;   /* constant index of the caught class, or UINT32_MAX for catch-all */
} ExceptionEntry;

#define JAI_OSR_MAX 4
/* Consecutive entry-guard failures on one loop head before it is left to
 * the interpreter. */
#define JAI_OSR_GIVE_UP 8

/* Instance-typed slots one compiled loop may pin a class shape on. A (slot,
 * shape) pair list rather than a shape beside every kind: 40 shapes per form
 * times JAI_OSR_MAX forms would be 640 bytes on every ObjFunction. A form that
 * would need more is refused rather than compiled unguarded.
 *
 * Was 8, on the grounds that "a loop with more than eight distinct instance
 * locals does not exist in this tree". It does: jaicv's `connected_components`
 * has two such heads and each is refused eighty times in one `imgproc` run.
 * Sixteen costs 5 bytes a shape times 4 forms = 160 more bytes on every
 * ObjFunction. */
#define JAI_OSR_SHAPES 16

/* A compiled loop: where it starts, and what each slot must hold to enter. */
typedef struct {
    uint8_t  *code;
    uint32_t  top;
    uint8_t   slots;
    /* 0 no iterator, 1 a unit-step range, 2 a list. A form compiled for one
     * must never be entered with the other: the prologue reads a different
     * object out of ObjIter.source for each. */
    uint8_t   iterKind;
    /* What each slot must hold on entry: a SlotKind in the low four bits, and
     * for a list slot the ListStore it must be backed by in the two above. A
     * list's storage is as much a compile-time commitment as an instance's
     * class -- the element loads are emitted at one width -- and checking it
     * here rather than at every subscript is what keeps the boxed case as
     * cheap as it was. */
    uint8_t   kinds[40];
    /* For iterKind 2: the ListStore of the list the head iterator walks,
     * checked the same way and for the same reason. */
    uint8_t   iterStg;

    /* Which CLASS an instance slot held, not merely that it held an instance.
     *
     * A field read compiles to a load at a baked slot index, and two classes
     * declaring the same field name put it at different indices whenever their
     * declaration order differs. Checking only IS_INSTANCE let a loop compiled
     * for one class be entered holding another and read the wrong field --
     * silently, with a plausible number. The probe:
     *
     *   class A { pub var x: int  pub var y: int }   # x is slot 0
     *   class B { pub var y: int  pub var x: int }   # x is slot 1
     *   fn sum_x(o: any, n: int) -> int { ... o.x ... }
     *   sum_x(A(), 2_000_000)   # compiles the loop against A
     *   sum_x(B(), 2_000_000)   # entered it holding a B: read 333, not 444
     *
     * A shape of 0 means the compile pinned no class to that slot, so any
     * instance may enter it. */
    uint8_t   shapeSlot[JAI_OSR_SHAPES];
    uint32_t  shapeId[JAI_OSR_SHAPES];
    uint8_t   shapeCount;
    /* Consecutive entry-guard failures on THIS head. It was once one counter
     * for the whole function, and eight failures on any single head switched
     * back-edge entry off for every compiled loop in it. Which head lost that
     * race varied per process, so an ORB run came out at either 312 ms or
     * 570 ms depending on nothing the program did. */
    uint8_t   declines;
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
    /* Entries so far, saturating at JAI_JIT_THRESHOLD. The compiled tier reads
     * it to decide when a function is worth compiling; it costs one increment
     * on the call path and stops counting once hot. */
    uint16_t    entryCount;
    /* What this function has been observed to RETURN, merged over its first
     * JAI_JIT_THRESHOLD interpreted returns and then frozen: JAI_FB_NONE never
     * seen, 1+ValueType for a non-object, JAI_FB_OBJ+ObjType for an object,
     * JAI_FB_MIXED for more than one. `obsReturnShape` is the class shape when
     * every observation was an instance of one class, else 0.
     *
     * A PREDICTION, never a fact -- the caller that uses it guards the tag (and
     * the shape) of what actually comes back, exactly as it does for a callee
     * that HAS compiled. Its whole point is the callee that has not: two
     * methods that call each other can never take turns being the first to
     * compile, so `jitReturnKind` is unreachable for both and a compiled caller
     * would otherwise have to decline every such site. Recorded per callee
     * rather than per call site because the observation has to be made where
     * the value is produced, and a per-site record armed at the call is lost
     * whenever the callee calls the same site again before returning -- which
     * is precisely the recursive shape this exists for. */
    uint8_t     obsReturnKind;
    uint32_t    obsReturnShape;
    /* Entry point of this function's compiled form, or NULL. Owned by the JIT's
     * arena, which outlives every function, so this is not freed here. */
    void       *jitCode;
    /* Which calling convention `jitCode` uses; see jit.c. */
    uint8_t     jitKind;
    /* Compiled whole-function form: a native routine taking int64 arguments
     * and returning one, calling itself directly. See jit_func.c. */
    uint8_t    *jitFunc;
    /* The module's global-mutation counter as it stood when each tier's form
     * was built. Compiled code resolves a class, a closure or a native once;
     * if any such binding may have moved since, that form is retired.
     *
     * ONE PER TIER, and that is load-bearing. A single field let compileOsr
     * re-arm a whole-function form that a rebinding had already retired: the
     * function tier's guard compares against whatever the OSR tier last
     * stored, so compiling a loop resurrected stale baked callees. See
     * tests/jit/rearm. */
    uint32_t    jitFuncModuleVersion;
    uint32_t    jitOsrModuleVersion;
    /* A builtin resolved at compile time is pinned by the builtins module's
     * own version, the same way a module global is pinned by its module's. */
    uint32_t    jitBuiltinsVersion;
    /* What the compiled form was specialised to: the kind of each parameter,
     * the class shape where that kind is an instance, and the kind it returns.
     * The entry guard re-checks every one on every call. */
    uint8_t     jitParamKind[8];
    uint32_t    jitParamShape[8];
    uint8_t     jitReturnKind;
    uint32_t    jitReturnShape;   /* class shape when the kind is an instance */
    uint8_t     jitArgBase;    /* first slot passed in: 0 for a method */
    uint8_t     jitArgCount;
    /* The compiled whole-function form never stores to the heap, so running it
     * again from the top is indistinguishable from not having run it. A caller
     * that branched straight to its entry needs this: it cannot hand the
     * interpreter a half-finished callee frame, so a nonzero verdict has to be
     * answered by re-executing the whole call, and that is only sound when the
     * abandoned attempt left nothing behind. */
    bool        jitFuncNoWrite;
    /* On-stack replacement: a compiled loop entered from the interpreter, with
     * the interpreter's own slots as its locals. This is what reaches a loop
     * in a function that runs once -- `main`, mostly. */
    /* One form per loop head, not one per function. `main` in `sieve` has
     * four loops and only the first ever compiled, because a second offset met
     * `osrTop != top` and was refused: the loop that does the sieving was
     * interpreted for the life of the program. */
    JaiOsrForm  osrForms[JAI_OSR_MAX];
    uint8_t     osrCount;
    /* Attempts so far. One look is not enough: a body can only use a callee's
     * return kind once that callee has itself compiled, and which of them have
     * depends on when the sampler happened to fire. Refusing forever on the
     * first miss made compilation depend on tick timing. */
    uint8_t     osrAttempts;
    /* Counted per loop head, not once per function: one counter starved every
     * other loop once the first one spent the budget, which was invisible
     * because a never-offered head reports no decline. Same attempts per head
     * as before; osrAttempts is the backstop for more heads than fit here. */
    uint32_t    osrMissTop[JAI_OSR_MAX];
    uint8_t     osrMissAttempts[JAI_OSR_MAX];
    uint8_t     osrMissCount;
    uint8_t     jitAttempts;
    bool        osrRefused;
    /* The back edge enters the compiled loop directly. An entry that keeps
     * declining -- the slots are not the kinds it was compiled for -- would
     * otherwise pay a call per iteration for nothing, which cost `sieve` 20%.
     * After a few in a row the back edge stops trying and the timer tick is
     * the only way back in. */
    bool        osrHot;
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

/* A field's declared kind, carried in bits 4-7 of OP_FIELD_DEF's info byte.
 *
 * A HINT, not a guarantee. A value can still reach a field through an `any`
 * receiver, which is exactly why the guard in OP_SET_FIELD exists -- and a
 * generic parameter, a nullable, or any shape this encoding cannot name is
 * FIELD_KIND_ANY, which accepts everything. ANY is zero so that every image
 * written before this encoding existed, whose bits 4-7 are zero, decodes to
 * "no claim" rather than to a wrong claim. */
typedef enum {
    FIELD_KIND_ANY      = 0,
    FIELD_KIND_INT      = 1,
    FIELD_KIND_FLOAT    = 2,
    FIELD_KIND_BOOL     = 3,
    FIELD_KIND_STR      = 4,
    FIELD_KIND_LIST     = 5,
    FIELD_KIND_DICT     = 6,
    FIELD_KIND_INSTANCE = 7,
} FieldKind;

/* "any", "int", … -- for diagnostics and for the disassembler. */
const char *jaiFieldKindName(uint32_t kind);
/* Whether a value satisfies a declared kind. Shared by the field guard and the
 * container-element guard, so the two can never drift apart. */
bool jaiKindAccepts(uint32_t kind, Value v);
/* As jaiKindAccepts, but throws TypeError on a mismatch. `what` names the thing
 * for the message, e.g. "an element". */
bool jaiCheckKind(uint32_t kind, Value v, const char *what);

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
    /* "Can compiled code still trust the bindings it baked in?"
     *
     * jit_func.c resolves a class, a closure or a native at compile time and
     * calls it directly, so a rebinding has to retire the compiled form. This
     * moves on any write that could change such a binding, and NOT on one that
     * merely updates a global to another inert value -- see jaiModuleSet and
     * jaiValueIsInertGlobal.
     *
     * A NEW READER MUST PICK THE RIGHT COUNTER. Anything memoising a global's
     * VALUE, or a resolved callee, keys on this. Anything memoising a table
     * slot's address or the absence of a name keys on globals.keyVersion. */
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

/* ITER_USER drives spec §7.1's `__next__` dunder, which ends on StopIteration;
 * ITER_TRAIT drives std.core's `trait Iterator`, whose `next` ends by returning
 * null. Both are user objects, and a class may implement either. */
typedef enum {
    ITER_LIST, ITER_TUPLE, ITER_STRING, ITER_BYTES, ITER_DICT_KEYS,
    ITER_DICT_ITEMS, ITER_SET, ITER_RANGE, ITER_USER, ITER_TRAIT,
    ITER_GENERATOR
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

/* The byte size of an object whose entire footprint is the one block it was
 * allocated in, or 0 for a kind that also owns arrays, tables or a FILE and so
 * has to go through jaiFreeObject's switch.
 *
 * It exists so the sweep can free the common kinds without a call.
 * jaiFreeObject is 550 instructions and saves six register pairs on entry, and
 * the sweep reaches it once per object the program ever allocated -- 8.0M times
 * in tests/bench/alloc_churn, where 94% of what the sweep visits is a corpse.
 * Twelve stack accesses of prologue and epilogue is most of what a dead
 * ObjInstance costs.
 *
 * jaiFreeObject asks this first and frees exactly what it returns, so the two
 * cannot drift: a size stated in two places that disagree puts a block in the
 * wrong bin, and the next request served from that bin is short. The switch is
 * exhaustive over ObjType for the same reason the one in jaiFreeObject is --
 * so -Wswitch names a new kind here rather than letting it fall to 0 and quietly
 * take the slow path forever. */
JAI_INLINE size_t jaiObjSoleBlock(const Obj *obj) {
    switch (obj->type) {
    case OBJ_STRING: {
        /* A slice carries no bytes of its own; the buffer it views is swept
         * separately once nothing views it. */
        const ObjString *s = (const ObjString *)obj;
        return s->owner != NULL ? sizeof(ObjString) : JAI_STRING_ALLOC(s->length);
    }
    case OBJ_STRBUF:
        return sizeof(ObjStrBuf) + ((const ObjStrBuf *)obj)->capacity + 1;
    case OBJ_BYTES:
        return sizeof(ObjBytes) + ((const ObjBytes *)obj)->length;
    case OBJ_TUPLE:
        return sizeof(ObjTuple) +
               sizeof(Value) * (size_t)((const ObjTuple *)obj)->count;
    case OBJ_INSTANCE:
        return sizeof(ObjInstance) +
               sizeof(Value) * (size_t)((const ObjInstance *)obj)->fieldCount;
    case OBJ_ENUM_VAL:
        return sizeof(ObjEnumVal) +
               sizeof(Value) * (size_t)((const ObjEnumVal *)obj)->count;
    case OBJ_RANGE:     return sizeof(ObjRange);
    case OBJ_UPVALUE:   return sizeof(ObjUpvalue);
    case OBJ_NATIVE:    return sizeof(ObjNative);
    case OBJ_BOUND:     return sizeof(ObjBound);
    case OBJ_ITER:      return sizeof(ObjIter);
    case OBJ_ENUM_CTOR: return sizeof(ObjEnumCtor);

    /* Own something besides their own block. */
    case OBJ_LIST:      /* items array */
    case OBJ_DICT:      /* table */
    case OBJ_SET:       /* table */
    case OBJ_FUNCTION:  /* chunk, param names, exception table, defaults */
    case OBJ_CLOSURE:   /* upvalue array */
    case OBJ_CLASS:     /* fields, traits, five tables */
    case OBJ_TRAIT:     /* two tables, supers */
    case OBJ_MODULE:    /* two tables */
    case OBJ_ENUM:      /* variants and their field names, methods table */
    case OBJ_FILE:      /* an open FILE * to close */
    case OBJ_TYPE_COUNT:
        return 0;
    }
    return 0;
}

#ifdef JAI_ALLOC_CENSUS
/* Allocations by object kind. See the census note in object.c. */
void jaiAllocPrintCensus(FILE *out);
#endif

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
