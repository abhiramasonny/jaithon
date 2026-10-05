/* vm.h — the bytecode interpreter. */
#ifndef JAI_VM_H
#define JAI_VM_H

#include "common/diag.h"
#include "vm/object/object.h"
#include "vm/bytecode/chunk.h"
#include "vm/table.h"

/* ------------------------------------------------------------------ */
/* Call frames. Inline caches live in the Chunk (chunk.h) so they survive */
/* serialisation boundaries and stay reachable from the code generator.   */
/* ------------------------------------------------------------------ */

typedef struct {
    ObjClosure *closure;
    uint8_t    *ip;
    Value      *slots;         /* the local window this frame reads and writes */
    /* Where the value stack rewinds to on exit; equal to `slots` for an
     * ordinary call. A deferred block runs in the defining frame's window
     * (spec §7.3), so it borrows `slots` and owns only the stack above it. */
    Value      *base;
    int         handlerBase;   /* index into vm->handlers */
    int         deferBase;     /* index into vm->defers */
    ObjModule  *module;
} CallFrame;

typedef struct {
    uint32_t handlerOffset;    /* code offset of the catch/finally target */
    uint32_t typeConst;        /* class constant index, UINT32_MAX = catch-all */
    int      frameIndex;
    Value   *stackTop;         /* restore point */
} ExcHandler;

/* ------------------------------------------------------------------ */
/* The machine                                                          */
/* ------------------------------------------------------------------ */

typedef struct VM {
    Value      *stack;         /* heap-allocated, JAI_STACK_MAX entries */
    Value      *stackTop;
    CallFrame  *frames;
    int         frameCount;
    int         frameCapacity;

    JAI_VEC(ExcHandler) handlers;
    JAI_VEC(Value)      defers;

    ObjUpvalue *openUpvalues;  /* sorted by stack address, descending */

    /* Modules */
    JaiTable    modules;       /* path string -> ObjModule* */
    ObjModule  *mainModule;
    ObjModule  *builtins;      /* the implicit global scope */
    JAI_VEC(ObjString *) modulePath;   /* JAITHON_PATH entries */

    /* Current exception, if unwinding. */
    Value        pendingException;
    bool         hasException;

    /* Interned dunder names, resolved once at startup. */
    ObjString   *strInit, *strStr, *strRepr, *strEq, *strLt, *strHash, *strLen,
                *strGetItem, *strSetItem, *strContains, *strIter, *strNext,
                /* `items`, for OP_GET_ITER_ITEMS's non-dict fallback. */
                *strItems,
                /* `enumerate`, for OP_INVOKE's lazy list.enumerate() head. */
                *strEnumerate,
                *strCall, *strAdd, *strSub, *strMul, *strDiv, *strMod,
                *strPow, *strNeg, *strMain, *strSelf, *strMessage;

    /* Well-known classes, created at startup. */
    ObjClass    *cError, *cTypeError, *cValueError, *cNameError, *cIndexError,
                *cKeyError, *cAttributeError, *cArithmeticError,
                *cDivisionByZeroError, *cOverflowError, *cIOError, *cOSError,
                *cRuntimeError, *cRecursionError, *cStopIteration,
                *cAssertionError, *cImportError, *cFileNotFoundError,
                *cPermissionError, *cParseError, *cLookupError;

    /* Flags */
    bool         debugTrace;
    /* Per-instruction bookkeeping: an unconditional `instructionCount++` in
     * the dispatch macro cost 1-3% of every benchmark, so it's only kept when
     * something will read it (--stats, --trace-exec). */
    bool         countInstructions;
    /* Attribute each interpreted instruction to the function running it, so a
     * census can be ranked by where the work IS rather than by how often a
     * refusal was printed. Only ever read under countInstructions, and set from
     * JAI_JIT_ATTRIB=1, so it costs one predictable branch on a path that is
     * already off in every build that matters.
     *
     * It exists because ranking by refusal frequency is not ranking by cost:
     * 168 refusals in one census were all once-per-process kernel setup and
     * were worth nothing, while eighty retries of one loop were worth 15x. */
    bool         attributeInstructions;
    bool         debugGC;
    bool         gcStress;
    unsigned     gcStressEvery;   /* see GCState::stressEvery */
    bool         releaseMode;
    int          optLevel;

    /* Statistics */
    uint64_t     instructionCount;
    /* Functions that have run at least one interpreted instruction, in first-
     * touch order. Registered from the dispatch path, so the list is the
     * working set and not the whole program. */
    struct ObjFunction **attributed;
    unsigned     attributedCount, attributedCap;
    uint64_t     callCount;
    uint64_t     allocCount;
    uint64_t     icHits, icMisses;
    /* The one-way PIC in jit_func.c's OP_INVOKE arm: how many unpinned-
     * receiver call sites it compiled a shape-guarded direct branch for
     * (jitPicAdmits) versus declined and left to jitInvokeByName alone
     * (jitPicRefusals). JAITHON_JIT_PIC=0 stops it engaging at all, so
     * these are the only way to tell it fired without instrumenting a
     * benchmark by hand. */
    uint64_t     jitPicAdmits, jitPicRefusals;

    /* GC state; see gc.h */
    struct GCState *gc;
} VM;

extern VM vm;

typedef enum {
    JAI_RUN_OK,
    JAI_RUN_COMPILE_ERROR,
    JAI_RUN_RUNTIME_ERROR,
} JaiRunResult;

void jaiVMInit(void);
void jaiVMFree(void);
void jaiVMResetStack(void);

JAI_INLINE void jaiPush(Value v) { *vm.stackTop++ = v; }
JAI_INLINE Value jaiPop(void)    { return *(--vm.stackTop); }
JAI_INLINE Value jaiPeek(int d)  { return vm.stackTop[-1 - d]; }

/* Run a compiled module body to completion. */
JaiRunResult jaiVMRunModule(ObjModule *module, ObjFunction *body);

/* Call any callable from C (natives, the REPL, dunder dispatch). Returns false
 * with the pending exception set on error. */
void jaiClassRememberShape(ObjClass *c);
bool jaiClassForShape(uint32_t shape, ObjClass **out);
/* Drops every remembered class the marker did not reach. Same phase as
 * jaiMethodCacheRemoveWhite: after tracing, before the sweep. */
void jaiShapeCacheRemoveWhite(void);
bool jaiClassFindMethod(ObjClass *klass, ObjString *name, Value *out);
bool jaiCallMethodWithReceiver(Value method, Value *argsWithReceiver,
                               int count, Value *out);
bool jaiInvokeNativeWithReceiver(Value native, Value *argsWithReceiver,
                                 int count, Value *out);
/* JAITHON_FMT_SHORT: whether the short f-string path is on, which the
 * compiled tier asks before it emits a leaf that is that path. */
bool jaiValueFormatShortOn(void);
/* An f-string's result for compiled code, or NULL when only jaiValueFormat
 * can make it. Never collects; see its definition in value.c. */
ObjString *jaiValueFormatLeaf(const Value *parts, int64_t count);
/* The same for an f-string of one int hole between optional string runs. */
ObjString *jaiValueFormatIntLeaf(Obj *pre, int64_t n, Obj *post);
/* The same for a caller that probed its memo site and missed: an intern
 * hit comes back with bit 0 of the pointer set, to be stripped and filed with
 * jaiFmtMemoFill. See value.c. */
ObjString *jaiValueFormatIntLeafMemo(Obj *pre, int64_t n, Obj *post);
/* What jaiValueFormatIntLeaf answered from the intern table, by (pre, n,
 * post), weak and direct-mapped, one table per compiled call site; compiled
 * code probes its site's table inline in front of the call. value.c says why
 * an entry is sound and when a site turns its probe off. An entry is the four
 * words in this order, and the JIT reads a site's first four words at these
 * offsets: `entries` and `mask` as one pair, then `budget` and `leaf`. */
typedef struct {
    Obj       *pre;
    Obj       *post;
    int64_t    n;
    ObjString *s;
} JaiFmtMemoEntry;
typedef struct JaiFmtSite {
    JaiFmtMemoEntry   *entries;   /* what the probe reads; NULL while off */
    uint64_t           mask;
    int64_t            budget;    /* probe misses left with no intern hit */
    void              *leaf;      /* jaiValueFormatIntLeafMemo */
    JaiFmtMemoEntry   *table;     /* kept while off; NULL until a first fill */
    uint64_t           fills;     /* since the last growth or collection */
    struct JaiFmtSite *nextOwner; /* the sites holding a table */
    bool               dirty;     /* filed into since the last collection */
} JaiFmtSite;
#define JAI_FMT_SITE_ENTRIES 0
#define JAI_FMT_SITE_BUDGET  16
#define JAI_FMT_SITE_LEAF    24
#define JAI_FMT_MEMO_EMPTY ((Obj *)(uintptr_t)1)
/* The slot for (pre, n, post) in a table of mask + 1 entries: n plus a CRC32C
 * of the runs that are there (an absent one adds nothing), so one site's
 * dense ints fill consecutive slots and two runs at nearby addresses still
 * land far apart. The JIT emits the same sum with crc32cx. */
static inline uint64_t jaiFmtMemoIndex(const Obj *pre, int64_t n,
                                       const Obj *post, uint64_t mask) {
    uint32_t c = 0;
#if defined(__ARM_FEATURE_CRC32)
    if (pre != NULL) c = __builtin_arm_crc32cd(c, (uint64_t)(uintptr_t)pre);
    if (post != NULL) c = __builtin_arm_crc32cd(c, (uint64_t)(uintptr_t)post);
#else
    if (pre != NULL) c = (uint32_t)(((uintptr_t)pre * 0x9e3779b97f4a7c15ull) >> 32);
    if (post != NULL) c ^= (uint32_t)(((uintptr_t)post * 0xbf58476d1ce4e5b9ull) >> 32);
#endif
    return ((uint64_t)n + c) & mask;
}
/* A new site, off and with no table, for the JIT to embed; NULL past the
 * cap on sites, when the site goes without a memo. Never freed. */
JaiFmtSite *jaiFmtSiteNew(void);
/* Every entry of every site dropped, and the tables of sites nothing was
 * filed into since the last call given back; the collector calls it, since
 * an entry holds its strings weakly, and so does the intern table at its
 * soft cap. */
void jaiFmtMemoClear(void);
/* Files `s`, an interned string, as the answer for (pre, n, post) at `site`,
 * turning the site's probe on and growing its table when it has been
 * refilled twice over; returns `s`. Never collects. */
ObjString *jaiFmtMemoFill(JaiFmtSite *site, Obj *pre, int64_t n, Obj *post,
                          ObjString *s);
/* `s[a:b]` for compiled code, or NULL when only jaiSliceGet can make it.
 * Never allocates; see its definition in object_string.c. */
ObjString *jaiStringSliceLeaf(ObjString *s, int64_t start, int64_t stop,
                              int64_t flags);
/* The result kind an OP_INVOKE site has observed for a receiver of
 * `receiver`'s type, or JAI_FB_NONE. A PREDICTION, not a guarantee -- see
 * InlineCache::resultKind; every caller must guard what it emits. */
uint8_t jaiInvokeResultFeedback(const Chunk *chunk, uint16_t cacheIdx,
                                Value receiver);

/* The megamorphic method cache is held weakly: this drops every entry the
 * marker did not reach, and must run in the same phase jaiTableRemoveWhite
 * runs for the intern table -- after tracing, before the sweep. */
void jaiMethodCacheRemoveWhite(void);
/* Reads JAITHON_MEGA_STRESS, which collapses that cache to one entry so every
 * key collides. Called by jaiVMInit. */
void jaiMethodCacheInit(void);

/* OP_GET_INDEX's semantics, callable from the compiled tier -- so the tier's
 * dict arm raises the interpreter's own KeyError, message and all, rather than
 * a second spelling of it. */
bool jaiIndexGet(Value container, Value index, Value *out);

/* `x in c` and `x not in c`, exported for the same reason: the tier's OP_IN arm
 * runs the interpreter's own containment, so a container it does not know still
 * behaves and still throws the interpreter's message. Returns false having
 * thrown. */
bool jaiContainsOp(Value container, Value element, bool *out);

/* OP_GET_SLICE's semantics, callable from the compiled tier. */
bool jaiSliceGet(Value container, Value startValue, Value stopValue,
                 Value stepValue, bool hasStart, bool hasStop, bool hasStep,
                 Value *out);

bool jaiCallValue(Value callee, int argc, Value *args, Value *out);

/* jaiCallValue for the one-argument case, what every higher-order list/
 * iterator builtin does per element: the general form's push/dispatch/unwind
 * through invokeCallable and callClosure measured as 42% of the loop on
 * `xs.map(|x| x * 2)` over ten million elements when the callee was a
 * compiled closure. This keeps just the two stack cells that root the callee
 * and argument, and falls through to jaiCallValue for anything else. */
bool jaiCallValue1(Value callee, Value arg, Value *out);

/* What a higher-order builtin should call per element. Same contract as
 * jaiCallValue1 -- it falls back to it for anything it does not recognise --
 * but the compiled-callee case is one C frame instead of three.
 *
 * The three were the builtin's element loop, jaiCallValue1 and
 * jaiJitEnterFunc, each with its own callee-saved prologue: 175 arm64
 * instructions retired per `xs.map(|x| x * 2)` element against 19 in the
 * compiled body itself, and 36 of the memory operations were register saves.
 * Defined in jit_func.c, where the argument and result conversions live, so
 * that flattening them costs no duplicated knowledge of SlotKind. */
bool jaiCallFn1(Value callee, Value arg, Value *out);

/* Everything jaiCallFn1 asks about the CALLEE, answered once instead of once
 * per element. A higher-order builtin calls one closure over a whole list, so
 * every one of those questions -- is it a closure, has it compiled, is its
 * arity one, does it want its own closure as a trailing argument, was it
 * compiled against this module's globals as they are now -- has the same
 * answer on element nine million as on element one, and asking them again is
 * eight loads and eight branches around a body that is nineteen instructions.
 *
 * The hoisted answers are allowed to go stale, never to be wrong. Both things
 * that can retire a compiled form under a running loop are visible in a field
 * this keeps a copy of: jitResultOut nulls fn->jitFunc when a body bails, and
 * a global rebound anywhere in the module moves module->version. One compare
 * against each is what remains per element, and either mismatch drops the
 * element back onto jaiCallFn1 and prepares again from scratch. */
typedef struct {
    ObjClosure  *closure;
    ObjFunction *fn;
    void        *entry;        /* fn->jitFunc when this was prepared */
    Value        callee;
    Value       *limit;        /* highest base a two-cell window may start at */
    uint32_t     moduleVersion;
    uint8_t      nargs;        /* 1, or 2 for a callee that reads an upvalue */
    uint8_t      returnKind;   /* SlotKind, as ObjFunction stores it */
    /* The argument is an int and it is slot 1, so the conversion is a tag test
     * and an untag rather than jitArgIn's switch and its read back out of the
     * window. Every lambda in the benchmark suite's map/filter/sort is this. */
    bool         intArg;
    /* Whether there was anything to hoist. False is the ordinary state at the
     * TOP of a loop, not an error: the tier looks at a function only after
     * sixty-four calls, so a lambda written at the call site has not compiled
     * when the loop starts and only becomes preparable sixty-four elements in.
     * jaiCallPreparedFn1 keeps trying until it does. */
    bool         flat;
} JaiPreparedFn1;

/* Ready `callee` for a loop that will call it once per element. Cannot fail:
 * a callee with nothing to hoist leaves `flat` false and every element goes
 * the long way, exactly as it did before. */
void jaiPrepareFn1(Value callee, JaiPreparedFn1 *prepared);

/* One element. Identical in effect to jaiCallFn1(prepared->callee, arg, out),
 * including every fallback it makes. */
bool jaiCallPreparedFn1(JaiPreparedFn1 *prepared, Value arg, Value *out);

/* The compiled tier keeps per-function tables that outlive one compile
 * attempt (jit_compile.c's depth memo). freeObject calls this for every
 * function it frees; it is one test when the tier never kept anything. */
void jaiJitForgetFunction(const ObjFunction *fn);

/* The length at which an untyped boxed list that has held only one of int,
 * float or bool takes that kind's unboxed storage, on the push that grows it
 * past this many (object_collection.c, jaiListShapeOnGrow). Returns true when
 * it did, with room made for that push; false leaves the list untouched for
 * the ordinary growth. JAITHON_LIST_SHAPE_GROWN=0 turns it off. */
#define JAI_LIST_SHAPE_AT 8
bool jaiListShapeOnGrow(ObjList *list, Value v);
bool jaiListShapeGrownOn(void);
/* The same shaping, asked for up front by a builder that knows the kind of
 * everything it is about to write (a map run whose callee returns an int, a
 * float or a bool): an untyped boxed list whose elements so far are all of
 * that kind takes the unboxed storage `stg` now, keeping its capacity. The
 * point is the reservation: a presized boxed array is twice the pages an
 * unboxed one is, and every one of them is faulted in by the fill. True when
 * the list is (now) in `stg`. JAITHON_MAP_UNBOXED=0 turns it off. */
bool jaiListShapeFor(ObjList *list, uint8_t stg);

/* A run of jaiCallPreparedFn1 over `src` from element `from`, appending each
 * result to `dst` -- for a flat callee that takes an int or a float and
 * returns an int, a float or a bool, with every per-callee check and the stack
 * window hoisted out of the loop. Stops
 * at the first element it cannot take that way and returns its index (`from`
 * itself when it took none); the caller takes that element the ordinary way
 * and may call again. `*ok` false means the callee raised, and the caller must
 * stop. JAITHON_MAP_RUN=0 turns it off (jaiMapRunOn). */
int  jaiMapPreparedFn1Run(JaiPreparedFn1 *prepared, ObjList *src, int from,
                          ObjList *dst, bool *ok);
bool jaiMapRunOn(void);
/* The same run for list.filter: a flat callee taking an int or a float and
 * returning a bool; each element it keeps is appended to `dst`. */
int  jaiFilterPreparedFn1(JaiPreparedFn1 *prepared, ObjList *src, int from,
                          ObjList *dst, bool *ok);

/* Finish, in the interpreter, a one-argument compiled call that deoptimised
 * part-way: the body already ran and may have written, so it must not be
 * re-entered from the top. `base` is the two-cell window [closure, argument]
 * the entry was made with and `frameBase` the frame count before it. */
bool jaiFinishJitDeopt1(ObjClosure *closure, Value *base, int frameBase,
                        Value *out);

/* Call a method by name on a receiver; false if the method does not exist. */
bool jaiInvokeMethod(Value receiver, ObjString *name, int argc, Value *args,
                     Value *out);

/* The same call for a caller holding the receiver at args[0] and no idea what
 * class it is, which is what the compiled tier has at a site whose receiver
 * varies. Answers from the shared megamorphic table, and raises rather than
 * returning false when there is no such method. */
bool jaiInvokeMethodByName(ObjString *name, Value *argsWithReceiver, int count,
                           Value *out);
/* Add the receiver's class as one more way of an OP_INVOKE site's cache, as
 * the interpreter's miss would. See jaiInvokeCacheLearn in vm_call.c. */
void jaiInvokeCacheLearn(InlineCache *ic, Value receiver, ObjString *name);

/* Field access honouring visibility and properties. */
bool jaiGetProperty(Value receiver, ObjString *name, Value *out);
bool jaiSetProperty(Value receiver, ObjString *name, Value value);

/* ------------------------------------------------------------------ */
/* Exceptions                                                           */
/* ------------------------------------------------------------------ */

/* Raise an exception of `klass` with a formatted message. Always returns
 * false so callers can `return jaiThrow(...)`. */
bool jaiThrow(ObjClass *klass, const char *fmt, ...) JAI_PRINTF(2, 3);
bool jaiThrowValue(Value exception);
void jaiClearException(void);
/* Build the frame list for a traceback. Caller frees with jaiRealloc. */
JaiFrameInfo *jaiBuildTraceback(int *outCount);
void jaiReportUncaught(Value exception);

/* ------------------------------------------------------------------ */
/* Roots for the GC                                                     */
/* ------------------------------------------------------------------ */

void jaiPushRoot(Value v);
void jaiPopRoot(void);
void jaiPopRoots(int n);

/* ------------------------------------------------------------------ */
/* Debug                                                                */
/* ------------------------------------------------------------------ */

void jaiVMPrintStack(FILE *out);
void jaiVMPrintStats(FILE *out);

/* JAITHON_LAZY_ENUMERATE, read once. On (the default) `for (i, x) in
 * xs.enumerate()` over a list walks an ITER_LIST_ENUM snapshot built at the
 * OP_INVOKE instead of the N-tuple list `list.enumerate()` materialises, and
 * both compiled tiers arm that head. `=0` restores the eager call everywhere,
 * which is the A/B. Read by the interpreter and by src/vm/jit. */
bool jaiLazyEnumerateOn(void);

#endif /* JAI_VM_H */
