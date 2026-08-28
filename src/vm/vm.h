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
    bool         debugGC;
    bool         gcStress;
    unsigned     gcStressEvery;   /* see GCState::stressEvery */
    bool         releaseMode;
    int          optLevel;

    /* Statistics */
    uint64_t     instructionCount;
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
bool jaiClassFindMethod(ObjClass *klass, ObjString *name, Value *out);
bool jaiCallMethodWithReceiver(Value method, Value *argsWithReceiver,
                               int count, Value *out);
bool jaiInvokeNativeWithReceiver(Value native, Value *argsWithReceiver,
                                 int count, Value *out);
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

#endif /* JAI_VM_H */
