/* vm.h — the bytecode interpreter. */
#ifndef JAI_VM_H
#define JAI_VM_H

#include "../common/diag.h"
#include "object.h"
#include "chunk.h"
#include "table.h"

/* ------------------------------------------------------------------ */
/* Call frames                                                          */
/*                                                                      */
/* Inline caches live in the Chunk (see chunk.h) so that they survive   */
/* serialisation boundaries and are reachable from the code generator.  */
/* ------------------------------------------------------------------ */

typedef struct {
    ObjClosure *closure;
    uint8_t    *ip;
    Value      *slots;         /* the local window this frame reads and writes */
    /* Where the value stack rewinds to when the frame exits. Equal to `slots`
     * for an ordinary call. A deferred block is the exception: it runs in the
     * defining frame's window (spec §7.3 — it sees that function's locals), so
     * it borrows `slots` and owns only the stack above them. */
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
    /* Per-instruction bookkeeping. An unconditional `instructionCount++` in the
     * dispatch macro cost 1-3% of every benchmark, so the counter is only kept
     * when something will read it (--stats, --trace-exec). */
    bool         countInstructions;
    bool         debugGC;
    bool         gcStress;
    bool         releaseMode;
    int          optLevel;

    /* Statistics */
    uint64_t     instructionCount;
    uint64_t     callCount;
    uint64_t     allocCount;
    uint64_t     icHits, icMisses;

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
bool jaiInvokeNativeWithReceiver(Value native, Value *argsWithReceiver,
                                 int count, Value *out);
bool jaiCallValue(Value callee, int argc, Value *args, Value *out);
/* Call a method by name on a receiver; false if the method does not exist. */
bool jaiInvokeMethod(Value receiver, ObjString *name, int argc, Value *args,
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
