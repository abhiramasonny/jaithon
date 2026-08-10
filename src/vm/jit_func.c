/* jit_func.c — compiling a whole function, calls and all.
 *
 * The loop tier in jit_loop.c compiles a counted loop out of a function that
 * the interpreter still owns. This one compiles the function itself, into an
 * ordinary arm64 routine with a native calling convention, so that a call it
 * makes to itself is a `bl` and not a CallFrame. That is the entire point: at
 * 15ns per interpreted call, `fib(30)` spends nearly all of its time in
 * pushFrame and OP_RETURN, and no amount of faster dispatch reaches C.
 *
 * The language it accepts is small and the restrictions are what make it
 * sound rather than what make it easy:
 *
 *   - integers only, and every operand is an int by construction: the entry
 *     guard checks the arguments, and the only values the body can make are
 *     results of int arithmetic or of a call to this same function
 *   - no allocation, and no globals but this function's own name
 *   - a store to an instance field is allowed, but only where no bail can
 *     follow it. A bail throws the compiled work away and lets the interpreter
 *     run the call from the top, and that is sound exactly while partial
 *     execution is invisible; a store that had already happened would be
 *     applied twice
 *   - self-recursion only. A general call would need a compiled callee and an
 *     agreed convention between them; that comes later
 *
 * Values live in registers for the whole call. Locals get x19 upward and the
 * operand stack continues from there, all callee-saved, so a value that is
 * live across a recursive call survives it without a spill. That caps the
 * function at nine live values, which is generous for anything this tier can
 * compile and is checked rather than assumed.
 *
 * Two things can go wrong at run time, and both take the same exit: signed
 * overflow, and the native stack running low. Compiled code sets a flag and
 * returns; the entry point sees the flag, discards the result, and hands the
 * call back to the interpreter, which then produces the real OverflowError or
 * RecursionError with a traceback. The function is refused from then on --
 * once a body has bailed, compiling it again only buys another bail.
 */
#include "jit.h"

#include "jit_arm64.h"
#include "gc.h"
/* For jaiBuiltinMethod: resolving `xs.len()` to its native happens at compile
 * time, so the tier has to ask the runtime what a name means. */
#include "runtime/runtime.h"
#include "vm.h"

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>

#if (defined(__aarch64__) || defined(__arm64__))

#include <pthread.h>

/* ------------------------------------------------------------------ */
/* Run-time state shared with compiled code                             */
/* ------------------------------------------------------------------ */

/* Compiled code returns two words: the value in x0 and whether it gave up in
 * x1. AAPCS64 returns a 16-byte struct of integers in exactly that pair, so
 * the flag costs no memory traffic at all -- it used to be a global, which was
 * a store and a load on every single call. A recursive call propagates it:
 * a callee that bailed sends its caller straight to its own bail block, so the
 * whole recursion unwinds at once instead of computing on with a junk value. */
typedef struct { int64_t value; int64_t bailed; } JitResult;

/* The lowest stack address compiled code may use, computed from the thread's
 * real bounds rather than from wherever the compile happened to run. Deriving
 * it from the compiling frame's sp looked simpler and is wrong: a function
 * compiled near the top of the stack and later entered from deep inside the
 * interpreter would bail on entry every time, and silently refuse itself. */
static uintptr_t stackLimit(void) {
    pthread_t self = pthread_self();
    void  *top  = pthread_get_stackaddr_np(self);   /* high address */
    size_t size = pthread_get_stacksize_np(self);
    if (top == NULL || size == 0) return 0;
    /* The margin has to cover the deepest single compiled frame plus whatever
     * the interpreter needs to unwind and report the error afterwards. */
    return (uintptr_t)top - size + (256u * 1024u);
}


/* ------------------------------------------------------------------ */
/* Register plan                                                        */
/* ------------------------------------------------------------------ */

#define JIT_FIRST_SAVED 19u   /* x19..x28 are callee-saved and ours */
#define JIT_MAX_SAVED   10u
/* How many slots the compile-time model can describe. Not the register budget:
 * a function may declare a wide frame and touch very little of it, and the
 * measuring pass has to walk the whole body to find that out. Only the second
 * pass is held to JIT_MAX_SAVED. */
#define JIT_MAX_SLOTS   64u
#define JIT_MAX_ARITY    4u   /* arguments arrive in x0..x3 */
/* Deopt stubs are the reason this is not small: each one writes out every
 * local and every live stack entry, so a body with a dozen guards spends more
 * on its cold paths than on its hot one. `merge` needed 512 and did not say
 * so, which is what the diagnostics were for. */
#define JIT_MAX_INSTS 20000u
#define JIT_MAX_FIXUPS 6000u
#define JIT_SCRATCH_A    9u
#define JIT_SCRATCH_B   10u
#define JIT_SCRATCH_C   11u
#define JIT_SCRATCH_D   12u

/* ------------------------------------------------------------------ */
/* Calling out of compiled code                                         */
/* ------------------------------------------------------------------ */

#define JIT_MAX_ARGS_OUT 4

/* Built on the compiled frame and handed to the helper below. Values first, so
 * every field is 8-aligned and the emitted stores can use the scaled forms. */
typedef struct {
    Value   callee;
    Value   args[JIT_MAX_ARGS_OUT];
    Value   roots[JIT_MAX_SAVED];
    Value   result;
    int64_t argc;
    int64_t nroots;
} JitCallDesc;

/* The one thing compiled code cannot do for itself.
 *
 * `roots` is why this is a C helper rather than an emitted sequence: the callee
 * can allocate, and an allocation can collect, and every instance this body is
 * holding lives in a callee-saved register the collector has never heard of.
 * Pushing them as temporary roots is three lines here and a page of emission
 * otherwise. The pointers in the registers stay valid across it because the
 * collector does not move objects -- only reachability was ever in question.
 *
 * Returns 0 on success and 1 with an exception pending. */
/* Raise the overflow the interpreter would have raised.
 *
 * An overflow is not a reason to deoptimise: the interpreter's answer to it is
 * to throw, and compiled code can throw the same thing. That matters more than
 * it sounds -- a bail hands the call back to be run from the top, which is
 * sound only while nothing has been written, so every arithmetic operation
 * after a field store used to decline the whole function. Raising instead
 * makes overflow an exception exit, which is sound after any amount of
 * writing: the interpreter would have written exactly the same things before
 * throwing. */
static void jitThrowOverflow(int64_t which) {
    static const char *ops[3]  = { "+",  "-",  "*"  };
    static const char *wrap[3] = { "+%", "-%", "*%" };
    int i = (which >= 0 && which < 3) ? (int)which : 0;
    (void)jaiThrow(vm.cOverflowError,
                   "integer overflow in '%s'; use '%s' to wrap", ops[i], wrap[i]);
}

/* Where a deoptimising body leaves what it was holding.
 *
 * A global rather than a frame field: the compiled frame is gone by the time
 * the C side looks, and one record is enough because only one compiled body
 * can be deoptimising at a time -- the VM is single-threaded and the record is
 * consumed before anything else runs. */
typedef struct {
    int64_t ip;        /* bytecode offset to resume at */
    int64_t base;      /* first local slot the record covers */
    int64_t nlocals;
    int64_t nstack;
    Value   locals[JIT_MAX_SLOTS + 1];
    Value   stack[JIT_MAX_SAVED + 1];
} JitDeoptRecord;

static JitDeoptRecord gDeopt;

bool jaiJitApplyDeopt(ObjClosure *closure, Value *slotBase) {
    ObjFunction *fn = closure->fn;
    if (gDeopt.ip < 0 || gDeopt.ip >= fn->chunk.count) return false;

    for (int64_t i = 0; i < gDeopt.nlocals; i++) {
        slotBase[gDeopt.base + i] = gDeopt.locals[i];
    }
    /* The operand stack sits above the frame's window, which bindCallArgs has
     * already set vm.stackTop to. */
    for (int64_t i = 0; i < gDeopt.nstack; i++) {
        *vm.stackTop++ = gDeopt.stack[i];
    }
    CallFrame *frame = &vm.frames[vm.frameCount - 1];
    frame->ip = fn->chunk.code + gDeopt.ip;
    if (getenv("JAI_JIT_RECON")) {
        fprintf(stderr, "[deopt] %s ip=%lld base=%lld nlocals=%lld nstack=%lld\n",
                fn->name ? fn->name->chars : "<anon>", (long long)gDeopt.ip,
                (long long)gDeopt.base, (long long)gDeopt.nlocals,
                (long long)gDeopt.nstack);
    }
    return true;
}

/* Invoke a built-in method: the receiver is args[0], which is exactly where
 * callNativeAt wants it, so no bound wrapper is made. Roots as jitCallOut
 * does, because push and its kin allocate. */
/* A method on an instance: the receiver is args[0]. */
static int jitInvokeMethod(JitCallDesc *d) {
    for (int64_t i = 0; i < d->nroots; i++) jaiGCPushRoot(d->roots[i]);
    bool ok = jaiCallMethodWithReceiver(d->callee, d->args, (int)d->argc,
                                        &d->result);
    jaiGCPopRoots((int)d->nroots);
    return ok ? 0 : 1;
}

static int jitInvokeNative(JitCallDesc *d) {
    for (int64_t i = 0; i < d->nroots; i++) jaiGCPushRoot(d->roots[i]);
    bool ok = jaiInvokeNativeWithReceiver(d->callee, d->args, (int)d->argc,
                                          &d->result);
    jaiGCPopRoots((int)d->nroots);
    return ok ? 0 : 1;
}

/* Build a list from the descriptor's arguments. Allocating, so the roots go
 * down first exactly as for a call. */
static int jitBuildList(JitCallDesc *d) {
    for (int64_t i = 0; i < d->nroots; i++) jaiGCPushRoot(d->roots[i]);
    ObjList *list = jaiListNew((int)d->argc);
    for (int64_t i = 0; i < d->argc; i++) list->items[i] = d->args[i];
    list->count = (int)d->argc;
    d->result = OBJ_VAL(list);
    jaiGCPopRoots((int)d->nroots);
    return 0;
}

/* Build the range and its iterator in one step. args[0] is the start, args[1]
 * the stop, args[2] whether it is inclusive. Allocating twice, so the roots go
 * down first as for any other call out of compiled code. */
static int jitMakeRangeIter(JitCallDesc *d) {
    for (int64_t i = 0; i < d->nroots; i++) jaiGCPushRoot(d->roots[i]);
    ObjRange *r = jaiRangeNew(AS_INT(d->args[0]), AS_INT(d->args[1]), 1,
                              AS_INT(d->args[2]) != 0);
    jaiGCPushRoot(OBJ_VAL(r));
    ObjIter *it = jaiIterNew(ITER_RANGE, OBJ_VAL(r));
    jaiGCPopRoot();
    d->result = OBJ_VAL(it);
    jaiGCPopRoots((int)d->nroots);
    return 0;
}

/* Make an iterator over whatever args[0] is. */
static int jitMakeIter(JitCallDesc *d) {
    for (int64_t i = 0; i < d->nroots; i++) jaiGCPushRoot(d->roots[i]);
    Value src = d->args[0];
    IterKind k = IS_LIST(src) ? ITER_LIST : ITER_LIST;
    ObjIter *it = jaiIterNew(k, src);
    d->result = OBJ_VAL(it);
    jaiGCPopRoots((int)d->nroots);
    return IS_LIST(src) ? 0 : 1;
}

/* One step. 0 advanced with the item in `result`, 1 exhausted, 2 raised. */
/* An f-string. The interpreter reads its parts straight off the operand stack;
 * the descriptor's args array is contiguous in the same way, so the pieces go
 * there and this is the whole of it. Only the builtin path -- the compiler
 * checks at compile time that the module has not bound its own `str`, and the
 * module version retires this form if one appears. */
static int jitFormat(JitCallDesc *d) {
    for (int64_t i = 0; i < d->nroots; i++) jaiGCPushRoot(d->roots[i]);
    ObjString *formatted = jaiValueFormat(d->args, (int)d->argc);
    jaiGCPopRoots((int)d->nroots);
    if (formatted == NULL) return 1;
    d->result = OBJ_VAL(formatted);
    return 0;
}

static int jitIterStep(JitCallDesc *d) {
    for (int64_t i = 0; i < d->nroots; i++) jaiGCPushRoot(d->roots[i]);
    Value item;
    bool ok = jaiIterNext(AS_ITER(d->args[0]), &item);
    jaiGCPopRoots((int)d->nroots);
    if (!ok) return vm.hasException ? 2 : 1;
    d->result = item;
    return 0;
}

/* Allocate an instance of args[0]'s class, nothing more. The fields are stored
 * by the compiled code that called this. */
static int jitNewInstance(JitCallDesc *d) {
    for (int64_t i = 0; i < d->nroots; i++) jaiGCPushRoot(d->roots[i]);
    ObjInstance *inst = jaiInstanceNew((ObjClass *)(uintptr_t)AS_OBJ(d->callee));
    d->result = OBJ_VAL(inst);
    jaiGCPopRoots((int)d->nroots);
    return 0;
}

static int jitCallOut(JitCallDesc *d) {
    for (int64_t i = 0; i < d->nroots; i++) jaiGCPushRoot(d->roots[i]);
    bool ok = jaiCallValue(d->callee, (int)d->argc, d->args, &d->result);
    jaiGCPopRoots((int)d->nroots);
    return ok ? 0 : 1;
}

/* One entry of the compile-time operand stack. A `self` entry is the callee of
 * a recursive call: it names this function and occupies no register, which is
 * why register numbers are derived from the count of value entries below an
 * entry rather than from its depth. */
typedef enum {
    SLOT_INT,     /* int64 payload, in an X register */
    SLOT_FLOAT,   /* double, held as raw bits in an X register */
    SLOT_INST,    /* ObjInstance *, raw, of a class fixed at compile time */
    SLOT_SELF,    /* this function, as a callee; occupies no register */
    SLOT_OPAQUE,  /* present in a register, but nothing may be done with it */
    SLOT_CLOSURE, /* the running closure, for reaching its upvalues */
    SLOT_CLASS,   /* a class resolved at compile time; only ever a callee */
    SLOT_FUNC,    /* a global function resolved at compile time, likewise */
    SLOT_NATIVE,  /* a builtin resolved at compile time, likewise */
    SLOT_ITER,    /* an ObjIter this body built, held raw. Its index stays in
                   * memory rather than a register, which costs a load and a
                   * store an iteration and makes a deopt need nothing: the
                   * iterator on the stack is always current. */
    SLOT_BOOL,    /* 0 or 1 in a register; a Value's boolean member is its low
                   * byte, so the same word serves both */
    SLOT_NULL,    /* a function with no value to return; never in a register */
    SLOT_OBJ,     /* some heap object, raw, of a type this tier does not model.
                   * It can be read, passed, stored and rooted -- nothing else.
                   * A closure held in a variable is the common case. */
    SLOT_LIST     /* ObjList *, raw. Safe for the same reason an instance is:
                   * nothing moves, and a call spills it as a root first */
} SlotKind;

/* Floats live in X registers and visit d0/d1 only for the arithmetic itself.
 * Three extra fmovs per operation, against a second register bank with its own
 * allocator, its own save set and its own spill rules -- for a tier this young
 * the simpler thing that is obviously correct is worth more than the three
 * instructions. Nothing calls between the fmovs, so the scratch pair is safe. */
#define JIT_FSCRATCH_A 0u
#define JIT_FSCRATCH_B 1u

static bool holdsRegister(SlotKind k) {
    return k != SLOT_SELF && k != SLOT_CLASS && k != SLOT_FUNC &&
           k != SLOT_NATIVE;
}

/* Two targets are not bytecode offsets at all. */
#define FIXUP_BAIL   UINT32_MAX
#define FIXUP_ENTRY  (UINT32_MAX - 1u)
#define FIXUP_THREW  (UINT32_MAX - 2u)
#define FIXUP_OVF    (UINT32_MAX - 3u)   /* + 0,1,2 for the three operators */
#define FIXUP_DEOPT  (UINT32_MAX - 7u)   /* minus the deopt-site index */
#define FIXUP_EXIT   (UINT32_MAX - 100u) /* minus the loop-exit index */

#define JIT_MAX_DEOPT 160

typedef struct {
    int      instIndex;    /* which emitted instruction to patch */
    uint32_t targetOffset; /* bytecode offset, or FIXUP_BAIL / FIXUP_ENTRY */
    bool     conditional;  /* b.cond rather than b */
    int      depth;        /* value-stack depth where the branch leaves from */
} Fixup;

typedef struct {
    uint32_t  code[JIT_MAX_INSTS];
    unsigned  count;

    SlotKind  stack[JIT_MAX_SAVED];
    uint32_t  stackShape[JIT_MAX_SAVED];  /* class shapeId for SLOT_INST */
    ObjClass *stackClass[JIT_MAX_SAVED];
    Value     stackSeen[JIT_MAX_SAVED];   /* the live value, for field feedback */
    int       stackLocal[JIT_MAX_SAVED];  /* which local it was copied from, or -1 */

    /* Fields this body has already stored, and with what kind. A read of one
     * of them needs no tag check: nothing can have changed it, because the
     * body cannot call. That matters more than it sounds -- `self.n = self.n +
     * 1; return self.n` is the commonest shape there is, and without this the
     * trailing read is a bail after a write, which the tier refuses. */
    struct { int local; uint16_t field; SlotKind kind; } known[16];
    unsigned  knownCount;
    SlotKind  localKind[JIT_MAX_SLOTS + 1];
    uint32_t  localShape[JIT_MAX_SLOTS + 1];
    ObjClass *localClass[JIT_MAX_SLOTS + 1];
    bool      localTyped[JIT_MAX_SLOTS + 1];   /* parameter, or already bound */
    /* A value of the kind this local holds, kept only so a field read on it
     * has something to read the field's type off. A local bound from a list
     * element has no argument to look at, which is why `bi` in nbody's
     * advance could not have its fields read. */
    Value     localSeen[JIT_MAX_SLOTS + 1];
    Value    *observed;      /* the live arguments, for field-type feedback */
    bool      assumedIntReturn;
    unsigned  depth;       /* entries on the operand stack */
    unsigned  valueDepth;  /* of those, how many hold a register */
    unsigned  maxValue;    /* high-water mark, for the save set */

    Fixup     fixups[JIT_MAX_FIXUPS];
    unsigned  fixupCount;

    int      *offsetToInst;  /* bytecode offset -> instruction index, or -1 */
    int      *offsetToDepth; /* bytecode offset -> stack signature, or -1 */

    unsigned  arity;
    /* Slot 0 is the callee for a plain call and the RECEIVER for a method, so
     * a method's `self.x` reads it. When the body touches it, it becomes an
     * ordinary local and an extra incoming argument; when it does not -- every
     * plain function -- it costs nothing. */
    unsigned  base;        /* first slot held in a register: 0 or 1 */
    bool      usesSlot0;
    /* The highest slot the body actually names. maxSlots is the frame window
     * the interpreter reserves, which is routinely larger -- and every slot
     * costs one of the ten callee-saved registers here. */
    unsigned  maxSlotUsed;
    /* The first pass exists to find maxValue and maxSlotUsed, so it must not
     * stop at a budget computed from a slot count it is still discovering. */
    bool      measuring;
    /* Locals live in the compiled frame instead of registers. A body with more
     * live values than there are callee-saved registers would otherwise be
     * declined outright, and nbody's `advance` wants nineteen. The operand
     * stack stays in registers either way -- that is where the arithmetic is. */
    bool      spilled;
    unsigned  localsFrameOffset;
    /* On-stack replacement: the locals ARE the interpreter's frame slots,
     * reached through a pointer handed to the entry. Nothing is copied either
     * way, which is also what makes a deopt cheap here -- the slots are
     * already what the interpreter expects, so only the operand stack needs
     * rebuilding. */
    bool      osr;
    uint32_t  osrTop;
    uint32_t  osrEnd;
    /* A `for i in a..b` loop, compiled as a counted one. The iterator object
     * stays on the interpreter's stack untouched; only its index rides in a
     * register, and every way out writes it back so the interpreter can carry
     * on from wherever this stopped. */
    bool      hasIter;
    /* Locals in registers rather than written through to the slots on every
     * access. The entry already checks every slot's kind before calling, so
     * the prologue only has to load payloads; every exit writes them back. */
    bool      osrRegLocals;
    /* Inlining a method widens the live range of everything it reads, so a
     * loop that fitted the registers as a call may not fit as an expression.
     * The compile is retried with this set when that is what went wrong. */
    bool      noInline;
    bool      inlined;
    /* A local whose kind is not the same on every path into some point. It
     * lives in the frame with its tag and every read of it guards, which is
     * what lets two paths disagree. The compiler reuses one slot for the
     * induction variables of loops that do not overlap, so this is not exotic:
     * it is nbody's advance. */
    bool      dynamicLocal[JIT_MAX_SLOTS + 1];
    bool      needDynamic[JIT_MAX_SLOTS + 1];   /* found during this attempt */
    bool      pendingRange;
    bool      rangeInclusive;
    unsigned  iterSlot;
    uint32_t  iterExit;
    int       exitStub[8];       /* one per distinct offset jumped out to */
    uint32_t  exitOffset[8];
    unsigned  exitCount;
    /* A body that reads an upvalue needs the closure itself, which is not any
     * slot: for a method slot 0 is the receiver, not the callee. It arrives as
     * one extra argument and takes the register just past the locals. */
    bool      usesUpvalues;
    /* Set once the body has written to the heap. After that a bail is no
     * longer free: re-running the call interpreted would apply the write a
     * second time. See branchOnCondition. */
    bool      wroteHeap;
    bool      bailAfterWrite;
    bool      callsOut;      /* the body reaches out to another function */
    const char *whyNot;
    uint8_t     lastOp;
    unsigned  descOffset;    /* JitCallDesc within the compiled frame */
    int       exceptionExit; /* instruction index of the "callee threw" exit */
    bool      overflowUsed[3];
    int       overflowStub[3];

    /* One per guard: the bytecode offset to resume at, and a snapshot of the
     * compile-time model there. The stub that writes them is emitted after the
     * body, so the hot path keeps a single not-taken branch. */
    struct {
        uint32_t ip;
        unsigned depth;
        unsigned valueDepth;
        SlotKind kinds[JIT_MAX_SAVED + 1];
        ObjClass *classes[JIT_MAX_SAVED + 1];
        /* The topmost entry is the result of a call that already happened, so
         * it lives in the descriptor rather than a register. */
        bool     lastFromDesc;
        int      stub;
    } deopt[JIT_MAX_DEOPT];
    unsigned  deoptCount;
    uint32_t  curOffset;
    bool      hasSelfCall;
    unsigned  locals;      /* slots base..base+locals-1 live in registers */
    unsigned  frameBytes;
    unsigned  savedCount;

    int       limitLiteral;  /* instruction index of the stack-limit word */
    int       bailBlock;     /* instruction index of the bail sequence */

    SlotKind  returnKind;
    uint32_t  returnShape;
    bool      sawReturn;

    bool      failed;
} Emit;

static void emit(Emit *e, uint32_t word) {
    if (e->count >= JIT_MAX_INSTS) { e->failed = true; return; }
    e->code[e->count++] = word;
}

/* Registers the operand stack starts after: none of them, once locals live in
 * the frame. */
/* x19 carries the slots pointer in OSR mode, so the operand stack starts one
 * register later. */
#define JIT_SLOTS_REG (JIT_FIRST_SAVED)
#define JIT_ITER_REG  (JIT_FIRST_SAVED + 1u)   /* the ObjIter itself */
#define JIT_IDX_REG   (JIT_FIRST_SAVED + 2u)
#define JIT_LIM_REG   (JIT_FIRST_SAVED + 3u)
#define JIT_START_REG (JIT_FIRST_SAVED + 4u)   /* the range's first value */

/* Registers OSR keeps for itself: the slots pointer, and for a range loop the
 * iterator, its index and its limit. */
static unsigned osrReserved(const Emit *e) { return e->hasIter ? 5u : 1u; }

static unsigned regBase(const Emit *e) {
    if (e->osr) {
        return osrReserved(e) + (e->osrRegLocals ? e->locals : 0u);
    }
    return e->spilled ? 0u : e->locals;
}

static unsigned localReg(const Emit *e, unsigned slot) {
    if (e->osr) return JIT_FIRST_SAVED + osrReserved(e) + slot;
    return JIT_FIRST_SAVED + (slot - e->base);
}

/* Sixteen bytes each, not eight: the tag travels with the value so a local
 * whose kind varies can be read behind a guard. */
static unsigned localFrameOff(const Emit *e, unsigned slot) {
    return e->localsFrameOffset + (slot - e->base) * 16u;
}

static void branchOnDeopt(Emit *e, unsigned cond);

static unsigned localTagFor(const Emit *e, unsigned slot) {
    SlotKind k = e->localKind[slot];
    return k == SLOT_INT    ? VAL_INT
         : k == SLOT_FLOAT  ? VAL_FLOAT
         : k == SLOT_BOOL   ? VAL_BOOL
         : k == SLOT_OPAQUE ? VAL_NULL
                            : VAL_OBJ;
}

/* A register holding `slot`'s value. In register mode that is the local's own
 * register and `scratch` goes unused; in memory mode the value is loaded into
 * `scratch`. Only the payload moves: a local's kind is fixed for the whole
 * function, so the tag never needs storing. */
static unsigned localIn(Emit *e, unsigned slot, unsigned scratch) {
    if (e->osr && e->osrRegLocals) return localReg(e, slot);
    if (e->osr) {
        emit(e, jaiA64LdrX(scratch, JIT_SLOTS_REG, slot * 16u + 8u));
        return scratch;
    }
    if (!e->spilled) return localReg(e, slot);
    if (e->dynamicLocal[slot]) {
        /* Two paths reached here disagreeing about this slot, so what it holds
         * is a runtime fact: check it against what this read was compiled for
         * and hand the instruction back otherwise. */
        /* Into `scratch`, not a fixed register: the caller has already
         * given this one up -- it is where the payload is about to land --
         * whereas JIT_SCRATCH_A may be holding an operand. `i + 1` on a
         * dynamic slot loaded the tag over the constant and added VAL_INT
         * instead, so `for j in i + 1..n` ran from i+2 and every nested loop
         * was one iteration short. */
        emit(e, jaiA64LdrW(scratch, 31, localFrameOff(e, slot)));
        emit(e, jaiA64SubsXImm(31, scratch, localTagFor(e, slot)));
        branchOnDeopt(e, JAI_A64_NE);
    }
    emit(e, jaiA64LdrX(scratch, 31, localFrameOff(e, slot) + 8));
    return scratch;
}

/* Where an operation that writes a local should compute its result.
 *
 * Only register-resident locals can be written in place. In OSR mode the
 * locals are the interpreter's slots and `localReg` names a register that is
 * holding something else entirely -- the loop counter, as it turned out. A
 * float add landed in x21 and the induction variable became a bit pattern, so
 * the loop finished early or never finished, depending on the value. */
static unsigned localDest(const Emit *e, unsigned slot) {
    if (e->osr && e->osrRegLocals) return localReg(e, slot);
    if (e->osr || e->spilled) return JIT_SCRATCH_C;
    return localReg(e, slot);
}

static void localOut(Emit *e, unsigned slot, unsigned src) {
    if (e->osr && e->osrRegLocals) {
        if (src != localReg(e, slot)) emit(e, jaiA64MovX(localReg(e, slot), src));
        return;
    }
    if (e->osr) {
        /* Tag as well as payload: writing straight through is what lets a
         * deopt here cost nothing. The kind is fixed for the compile. */
        SlotKind k = e->localKind[slot];
        unsigned tag = k == SLOT_INT   ? VAL_INT
                     : k == SLOT_FLOAT ? VAL_FLOAT
                     : k == SLOT_BOOL  ? VAL_BOOL
                                       : VAL_OBJ;
        emit(e, jaiA64MovzX(JIT_SCRATCH_D, tag, 0));
        emit(e, jaiA64StrW(JIT_SCRATCH_D, JIT_SLOTS_REG, slot * 16u));
        emit(e, jaiA64StrX(src, JIT_SLOTS_REG, slot * 16u + 8u));
        return;
    }
    if (!e->spilled) {
        if (src != localReg(e, slot)) emit(e, jaiA64MovX(localReg(e, slot), src));
        return;
    }
    emit(e, jaiA64MovzX(JIT_SCRATCH_D, localTagFor(e, slot), 0));
    emit(e, jaiA64StrW(JIT_SCRATCH_D, 31, localFrameOff(e, slot)));
    emit(e, jaiA64StrX(src, 31, localFrameOff(e, slot) + 8));
}

/* Slots the body may name at all. */
static bool localInRange(Emit *e, unsigned slot) {
    if (e->osr) {
        if (slot >= e->locals) return false;
        if (slot > e->maxSlotUsed) e->maxSlotUsed = slot;
        return true;
    }
    if (slot < e->base || slot >= e->base + e->locals) return false;
    if (slot > e->maxSlotUsed) e->maxSlotUsed = slot;
    return true;
}

/* Slots whose value this call can be looked at, for type feedback. */
static bool localObserved(Emit *e, unsigned slot) {
    if (e->osr) return slot < e->locals;
    if (slot < e->base || slot > e->arity) return false;
    if (slot > e->maxSlotUsed) e->maxSlotUsed = slot;
    return true;
}

/* The register a new value entry would occupy. */
static unsigned closureReg(const Emit *e) {
    return JIT_FIRST_SAVED + regBase(e);
}

static unsigned pushReg(const Emit *e) {
    return JIT_FIRST_SAVED + regBase(e) + (e->usesUpvalues ? 1u : 0u) +
           e->valueDepth;
}

static bool pushValue3(Emit *e, SlotKind kind, uint32_t shape, ObjClass *klass,
                       Value seen, int fromLocal) {
    if (e->depth >= JIT_MAX_SAVED) {
        e->whyNot = "the operand stack is deeper than the model allows";
        return false;
    }
    if (!e->measuring &&
        regBase(e) + (e->usesUpvalues ? 1u : 0u) + e->valueDepth + 1 >
            JIT_MAX_SAVED) {
        e->whyNot = "more live values than there are callee-saved registers";
        return false;
    }
    e->stackShape[e->depth] = shape;
    e->stackClass[e->depth] = klass;
    e->stackSeen[e->depth]  = seen;
    e->stackLocal[e->depth] = fromLocal;
    e->stack[e->depth++] = kind;
    e->valueDepth++;
    if (e->valueDepth > e->maxValue) e->maxValue = e->valueDepth;
    return true;
}

static bool pushValue(Emit *e, SlotKind kind, uint32_t shape, ObjClass *klass) {
    return pushValue3(e, kind, shape, klass, NULL_VAL, -1);
}

/* The kind a field is known to hold, or SLOT_SELF for "not known". */
static SlotKind knownFieldKind(const Emit *e, int local, uint16_t field) {
    if (local < 0) return SLOT_SELF;
    for (unsigned i = 0; i < e->knownCount; i++) {
        if (e->known[i].local == local && e->known[i].field == field) {
            return e->known[i].kind;
        }
    }
    return SLOT_SELF;
}

/* Record a store, and drop what any other receiver claimed about the same
 * field: two locals can name the same object, so a store through one has to
 * retire the other's knowledge. */
static void recordFieldStore(Emit *e, int local, uint16_t field, SlotKind kind) {
    unsigned out = 0;
    for (unsigned i = 0; i < e->knownCount; i++) {
        if (e->known[i].field == field) continue;
        e->known[out++] = e->known[i];
    }
    e->knownCount = out;
    if (local < 0) return;
    if (e->knownCount >= 16) return;
    e->known[e->knownCount].local = local;
    e->known[e->knownCount].field = field;
    e->known[e->knownCount].kind  = kind;
    e->knownCount++;
}

static bool pushSelf(Emit *e) {
    if (e->depth >= JIT_MAX_SAVED) return false;
    e->stack[e->depth++] = SLOT_SELF;
    return true;
}

/* Pop a value entry, reporting the register it was in and what it held. */
static bool popValue(Emit *e, unsigned *reg, SlotKind *kind) {
    if (e->depth == 0 || !holdsRegister(e->stack[e->depth - 1])) return false;
    e->depth--;
    e->valueDepth--;
    if (kind != NULL) *kind = e->stack[e->depth];
    *reg = JIT_FIRST_SAVED + regBase(e) + (e->usesUpvalues ? 1u : 0u) +
           e->valueDepth;
    return true;
}

/* Depth and the kind of every entry, in one word. Registers are assigned from
 * the depth and instructions are chosen from the kinds, so a join reached with
 * either one different is a join this tier cannot compile. */
static uint32_t stackSignatureAt(const Emit *e, unsigned depth) {
    uint32_t sig = depth & 0xfu;
    for (unsigned i = 0; i < depth && i < 9; i++) {
        sig |= ((uint32_t)e->stack[i] & 3u) << (4 + 2 * i);
    }
    return sig;
}

static uint32_t stackSignature(const Emit *e) {
    return stackSignatureAt(e, e->depth);
}

/* ------------------------------------------------------------------ */
/* Constants                                                            */
/* ------------------------------------------------------------------ */

/* Materialise a full 64-bit constant. One instruction for the small cases
 * that matter (`n < 2`, `n - 1`), up to four for anything else. */
static void emitConst64(Emit *e, unsigned rd, int64_t value) {
    if (value >= 0 && value <= 0xffff) {
        emit(e, jaiA64MovzX(rd, (unsigned)value, 0));
        return;
    }
    if (value < 0 && value >= -0x10000) {
        emit(e, jaiA64MovnX(rd, (unsigned)(~(uint64_t)value & 0xffffu)));
        return;
    }
    uint64_t bits = (uint64_t)value;
    emit(e, jaiA64MovzX(rd, (unsigned)(bits & 0xffffu), 0));
    for (unsigned shift = 1; shift < 4; shift++) {
        unsigned part = (unsigned)((bits >> (16 * shift)) & 0xffffu);
        if (part != 0) emit(e, jaiA64MovkX(rd, part, shift));
    }
}

/* ------------------------------------------------------------------ */
/* Prologue and epilogue                                                */
/* ------------------------------------------------------------------ */

static void emitSaveRestore(Emit *e, bool save) {
    for (unsigned i = 0; i < e->savedCount; i += 2) {
        unsigned r1 = JIT_FIRST_SAVED + i;
        int32_t at = (int32_t)(16 + 8 * i);
        if (i + 1 < e->savedCount) {
            emit(e, save ? jaiA64StpOff(r1, r1 + 1, 31, at)
                         : jaiA64LdpOff(r1, r1 + 1, 31, at));
        } else {
            emit(e, save ? jaiA64StrX(r1, 31, (unsigned)at)
                         : jaiA64LdrX(r1, 31, (unsigned)at));
        }
    }
}

/* STP's pre-index immediate is a signed seven-bit field scaled by eight, so it
 * reaches -512 and no further. Past that the stack pointer has to move on its
 * own: the encoder truncates silently, so a 528-byte frame moved SP by 16 and
 * every write meant for the frame landed in the caller's, which showed up as a
 * failed stack check inside `jaiJitEnter` with nothing wrong at the crash site.
 * nbody's `advance` is the first body big enough to want one -- twelve spilled
 * locals and a call descriptor come to 528. */
static void emitFrameEnter(Emit *e) {
    if (e->frameBytes <= 512u) {
        emit(e, jaiA64StpPre(29, 30, 31, -(int32_t)e->frameBytes));
        return;
    }
    emit(e, jaiA64SubXImm(31, 31, e->frameBytes));
    emit(e, jaiA64StpOff(29, 30, 31, 0));
}

static void emitFrameLeave(Emit *e) {
    if (e->frameBytes <= 512u) {
        emit(e, jaiA64LdpPost(29, 30, 31, (int32_t)e->frameBytes));
        return;
    }
    emit(e, jaiA64LdpOff(29, 30, 31, 0));
    emit(e, jaiA64AddXImm(31, 31, e->frameBytes));
}

static void emitEpilogue(Emit *e, unsigned bailed) {
    emit(e, jaiA64MovzX(1, bailed, 0));
    emitSaveRestore(e, false);
    emitFrameLeave(e);
    emit(e, jaiA64Ret());
}

/* ------------------------------------------------------------------ */
/* The body                                                            */
/* ------------------------------------------------------------------ */

/* In OSR mode a jump out of the compiled range leaves the loop: it becomes a
 * stub that reports the offset the interpreter should carry on from. */
static uint32_t exitTargetFor(Emit *e, uint32_t target) {
    for (unsigned i = 0; i < e->exitCount; i++) {
        if (e->exitOffset[i] == target) return FIXUP_EXIT - i;
    }
    if (e->exitCount >= 8) { e->whyNot = "too many ways out of the loop"; e->failed = true; return FIXUP_EXIT; }
    e->exitOffset[e->exitCount] = target;
    return FIXUP_EXIT - e->exitCount++;
}

static void branchToDepth(Emit *e, uint32_t targetOffset, unsigned cond,
                          int depthOverride);

static void branchTo(Emit *e, uint32_t targetOffset, bool conditional,
                     unsigned cond) {
    if (e->fixupCount >= JIT_MAX_FIXUPS) { e->failed = true; return; }
    if (e->osr && targetOffset < UINT32_MAX - 64u &&
        (targetOffset < e->osrTop || targetOffset >= e->osrEnd)) {
        targetOffset = exitTargetFor(e, targetOffset);
    }
    e->fixups[e->fixupCount].instIndex    = (int)e->count;
    e->fixups[e->fixupCount].targetOffset = targetOffset;
    e->fixups[e->fixupCount].conditional  = conditional;
    e->fixups[e->fixupCount].depth        = (int)stackSignature(e);
    e->fixupCount++;
    emit(e, conditional ? jaiA64BCond(cond, 0) : jaiA64B(0));
}

/* A guard failed: hand this exact point to the interpreter.
 *
 * Not a bail. A bail re-runs the call from the top, which stops being sound
 * the moment the body has written anything -- and the guards that matter are
 * on field reads inside loops that write. This records where the interpreter
 * should pick up and what it should be holding; the stub that writes it out is
 * emitted after the body, so the hot path keeps one not-taken branch. */
static bool jitDeoptStress(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_DEOPT_STRESS");
        cached = (v != NULL && v[0] != '\0' && strcmp(v, "0") != 0) ? 1 : 0;
    }
    return cached != 0;
}

static void branchOnDeoptAt(Emit *e, unsigned cond, uint32_t ip,
                            bool lastFromDesc) {
    if (e->fixupCount >= JIT_MAX_FIXUPS) { e->failed = true; return; }
    if (e->deoptCount >= JIT_MAX_DEOPT) {
        e->whyNot = "too many guards to record";
        e->failed = true;
        return;
    }
    unsigned k = e->deoptCount++;
    e->deopt[k].ip           = ip;
    e->deopt[k].depth        = e->depth;
    e->deopt[k].valueDepth   = e->valueDepth;
    e->deopt[k].lastFromDesc = lastFromDesc;
    for (unsigned i = 0; i < e->depth; i++) {
        e->deopt[k].kinds[i]   = e->stack[i];
        e->deopt[k].classes[i] = e->stackClass[i];
    }
    bool always = jitDeoptStress();
    e->fixups[e->fixupCount].instIndex    = (int)e->count;
    e->fixups[e->fixupCount].targetOffset = FIXUP_DEOPT - k;
    e->fixups[e->fixupCount].conditional  = !always;
    e->fixups[e->fixupCount].depth        = -1;
    e->fixupCount++;
    emit(e, always ? jaiA64B(0) : jaiA64BCond(cond, 0));
}

static void branchOnDeopt(Emit *e, unsigned cond) {
    if (e->fixupCount >= JIT_MAX_FIXUPS) { e->failed = true; return; }
    if (e->deoptCount >= JIT_MAX_DEOPT) {
        e->whyNot = "too many guards to record";
        e->failed = true;
        return;
    }
    unsigned k = e->deoptCount++;
    e->deopt[k].ip         = e->curOffset;
    e->deopt[k].depth      = e->depth;
    e->deopt[k].valueDepth = e->valueDepth;
    for (unsigned i = 0; i < e->depth; i++) {
        e->deopt[k].kinds[i]   = e->stack[i];
        e->deopt[k].classes[i] = e->stackClass[i];
    }
    e->fixups[e->fixupCount].instIndex    = (int)e->count;
    e->fixups[e->fixupCount].targetOffset = FIXUP_DEOPT - k;
    /* JAITHON_JIT_DEOPT_STRESS makes every guard fail. Compiled code then
     * deoptimises at the first one it meets, so the whole test suite becomes a
     * test of the resume path -- which is otherwise reached only when a
     * program changes a field's type, and almost none do. */
    bool always = jitDeoptStress();
    e->fixups[e->fixupCount].conditional  = !always;
    e->fixups[e->fixupCount].depth        = -1;
    e->fixupCount++;
    emit(e, always ? jaiA64B(0) : jaiA64BCond(cond, 0));
}

/* Branch to the bail block on `cond`. The block's index is not known yet, so
 * it is patched with the rest. */
/* Comparing a NaN is a TypeError here, not false: the interpreter tests isnan
 * before it compares, and falls through to the slow path that raises. fcmp
 * reports unordered in V, so an unordered result goes back to the interpreter
 * to raise exactly what it would have raised. */
static void nanToDeopt(Emit *e) { branchOnDeopt(e, JAI_A64_VS); }

static void branchOnCondition(Emit *e, unsigned cond) {
    if (e->fixupCount >= JIT_MAX_FIXUPS) { e->failed = true; return; }
    if (e->wroteHeap) e->bailAfterWrite = true;
    e->fixups[e->fixupCount].instIndex    = (int)e->count;
    e->fixups[e->fixupCount].targetOffset = FIXUP_BAIL;
    e->fixups[e->fixupCount].conditional  = true;
    e->fixups[e->fixupCount].depth        = -1;   /* bails do not join */
    e->fixupCount++;
    emit(e, jaiA64BCond(cond, 0));
}

/* Overflow goes to a per-operator stub that throws, not to the bail block.
 *
 * `cond` is not always VS. adds and subs set the overflow flag, but the
 * multiply test is a comparison of the product's high half against the low
 * half's replicated sign, so its answer is NE. Routing it through VS meant
 * multiply overflow was simply never detected -- 4 * 2^62 came back as 0
 * instead of raising, which is a wrong answer, not a crash. */
static void branchOnOverflow(Emit *e, unsigned which, unsigned cond) {
    if (e->fixupCount >= JIT_MAX_FIXUPS) { e->failed = true; return; }
    e->overflowUsed[which] = true;
    e->fixups[e->fixupCount].instIndex    = (int)e->count;
    e->fixups[e->fixupCount].targetOffset = FIXUP_OVF - which;
    e->fixups[e->fixupCount].conditional  = true;
    e->fixups[e->fixupCount].depth        = -1;
    e->fixupCount++;
    emit(e, jaiA64BCond(cond, 0));
}

/* The condition to branch on when the comparison is FALSE: the opcode jumps
 * over the taken side, `if (!taken) ip += offset`. */
static bool negatedCondition(uint8_t cmp, unsigned *out) {
    switch (cmp) {
    case OP_EQ: *out = JAI_A64_NE; return true;
    case OP_NE: *out = JAI_A64_EQ; return true;
    case OP_LT: *out = JAI_A64_GE; return true;
    case OP_LE: *out = JAI_A64_GT; return true;
    case OP_GT: *out = JAI_A64_LE; return true;
    case OP_GE: *out = JAI_A64_LT; return true;
    default: return false;
    }
}

/* Does this OP_GET_GLOBAL name the function being compiled? */
/* The class a global names, or NULL when it names something else. */
static ObjClass *globalClass(ObjClosure *closure, uint32_t nameIdx) {
    ObjFunction *fn = closure->fn;
    if (fn->module == NULL) return NULL;
    if (nameIdx >= (uint32_t)fn->chunk.constants.count) return NULL;
    Value name = fn->chunk.constants.data[nameIdx];
    if (!IS_STRING(name)) return NULL;
    Value bound;
    if (!jaiModuleGet(fn->module, AS_STRING(name), &bound)) return NULL;
    return IS_CLASS(bound) ? AS_CLASS(bound) : NULL;
}

/* The global function a name refers to, or NULL. */
static ObjFunction *globalFunction(ObjClosure *closure, uint32_t nameIdx,
                                   Value *out) {
    ObjFunction *fn = closure->fn;
    if (fn->module == NULL) return NULL;
    if (nameIdx >= (uint32_t)fn->chunk.constants.count) return NULL;
    Value name = fn->chunk.constants.data[nameIdx];
    if (!IS_STRING(name)) return NULL;
    Value bound;
    if (!jaiModuleGet(fn->module, AS_STRING(name), &bound)) return NULL;
    if (!IS_CLOSURE(bound)) return NULL;
    *out = bound;
    return AS_CLOSURE(bound)->fn;
}

/* A builtin, resolved the way the interpreter resolves one: the module first,
 * and `vm.builtins` only when the module has no such name. A module-level
 * binding of the same name is therefore never mistaken for the builtin, and if
 * one appears later the module's version retires this compiled form. */
static ObjNative *globalNative(ObjClosure *closure, uint32_t nameIdx,
                               Value *out) {
    ObjFunction *fn = closure->fn;
    if (fn->module == NULL || vm.builtins == NULL) return NULL;
    if (nameIdx >= (uint32_t)fn->chunk.constants.count) return NULL;
    Value name = fn->chunk.constants.data[nameIdx];
    if (!IS_STRING(name)) return NULL;
    Value shadow;
    if (jaiModuleGet(fn->module, AS_STRING(name), &shadow)) return NULL;
    Value bound;
    if (!jaiModuleGet(vm.builtins, AS_STRING(name), &bound)) return NULL;
    if (!IS_NATIVE(bound)) return NULL;
    *out = bound;
    return AS_NATIVE(bound);
}

static bool globalIsSelf(ObjClosure *closure, uint32_t nameIdx) {
    ObjFunction *fn = closure->fn;
    if (fn->module == NULL) return false;
    if (nameIdx >= (uint32_t)fn->chunk.constants.count) return false;
    Value name = fn->chunk.constants.data[nameIdx];
    if (!IS_STRING(name)) return false;

    Value bound;
    if (!jaiModuleGet(fn->module, AS_STRING(name), &bound)) return false;
    return IS_CLOSURE(bound) && AS_CLOSURE(bound)->fn == fn;
}

/* A callee that is not this function: only a class, whose result is an
 * instance of a shape known here. Anything else would need a guard on a return
 * value nothing can predict. */
/* An initializer that does nothing but store its arguments into fields, in
 * order: `GET_LOCAL2 0 k; SET_FIELD f` repeated, then RETURN_NULL. Anything
 * else -- a default, a computed field, a call, a branch -- is not this, and
 * goes the long way.
 *
 * Recognising it is what lets `Point(a, b)` become an allocation and two
 * stores instead of a descriptor, jaiCallValue, invokeCallable's type switch
 * and a compiled init. That machinery is most of the 35ns an allocation costs
 * here, against 5ns for the same thing in C++. */
static bool simpleInitFields(ObjClass *cls, unsigned argc, uint16_t *slots) {
    Value initv;
    if (cls == NULL) return false;
    if (!jaiClassFindMethod(cls, vm.strInit, &initv)) return false;
    if (!IS_CLOSURE(initv)) return false;
    ObjFunction *ifn = AS_CLOSURE(initv)->fn;
    if (ifn->arity != argc || ifn->defaultCount != 0) return false;
    if (ifn->flags & (FN_VARIADIC | FN_KWREST)) return false;
    if (ifn->upvalueCount != 0) return false;

    const uint8_t *c = ifn->chunk.code;
    int n = ifn->chunk.count;
    int off = 0;
    for (unsigned i = 0; i < argc; i++) {
        if (off + 5 > n || c[off] != OP_GET_LOCAL2) return false;
        if (jaiReadU16(c + off + 1) != 0) return false;          /* self */
        if (jaiReadU16(c + off + 3) != i + 1) return false;      /* arg i */
        off += 5;
        if (off + 6 > n || c[off] != OP_SET_FIELD) return false;
        uint32_t nameIdx = jaiReadU24(c + off + 1);
        if (nameIdx >= (uint32_t)ifn->chunk.constants.count) return false;
        Value nm = ifn->chunk.constants.data[nameIdx];
        if (!IS_STRING(nm)) return false;
        const FieldInfo *fi = jaiClassFieldInfo(cls, AS_STRING(nm));
        if (fi == NULL || fi->isStatic) return false;
        slots[i] = fi->slot;
        off += 6;
    }
    return off < n && c[off] == OP_RETURN_NULL;
}

static bool isClassCallee(const Emit *e, unsigned argc) {
    return e->depth >= argc + 1u &&
           e->stack[e->depth - argc - 1] == SLOT_CLASS;
}

static bool adoptLocalKind(Emit *e, unsigned slot, SlotKind kind,
                           uint32_t shape, ObjClass *klass);
static bool adoptLocalKindSeen(Emit *e, unsigned slot, SlotKind kind,
                               uint32_t shape, ObjClass *klass, Value seen);

/* `ownStatus` means the caller decodes the helper's return itself, so the
 * built-in "nonzero means raised" test is not emitted. Only the iterator step
 * wants that: it answers 0 yielded, 1 exhausted, 2 raised, and the default test
 * sends `exhausted` to the throw stub -- which reports an error the interpreter
 * then cannot find, and the run dies on "internal error: failed operation
 * raised nothing". The hand-written EQ/GT tests at the call site were dead code
 * until now, because control never reached them with a nonzero status. */
static bool emitDescriptorStatus(Emit *e, Value calleeVal, unsigned first,
                                 unsigned nargs, void *helper, bool ownStatus,
                                 int calleeReg) {
    if (nargs > JIT_MAX_ARGS_OUT) { e->whyNot = "call argc"; return false; }
    if (!e->callsOut) { e->whyNot = "callsOut off"; return false; }

    unsigned d = e->descOffset;

    /* The callee, as a whole Value. From a register when the callee is only
     * known at run time -- a closure held in a local. Baking the one that
     * happened to be live at compile time would freeze its upvalues, and
     * `closure_calls` builds a fresh closure over a different `step` on every
     * outer iteration. */
    if (calleeReg >= 0) {
        emit(e, jaiA64MovzX(JIT_SCRATCH_A, VAL_OBJ, 0));
        emit(e, jaiA64StrW(JIT_SCRATCH_A, 31,
                           d + (unsigned)offsetof(JitCallDesc, callee)));
        emit(e, jaiA64StrX((unsigned)calleeReg, 31,
                           d + (unsigned)offsetof(JitCallDesc, callee) + 8));
    } else {
        emit(e, jaiA64MovzX(JIT_SCRATCH_A, (unsigned)calleeVal.type, 0));
        emit(e, jaiA64StrW(JIT_SCRATCH_A, 31,
                           d + (unsigned)offsetof(JitCallDesc, callee)));
        emitConst64(e, JIT_SCRATCH_A, (int64_t)(uintptr_t)calleeVal.as.obj);
        emit(e, jaiA64StrX(JIT_SCRATCH_A, 31,
                           d + (unsigned)offsetof(JitCallDesc, callee) + 8));
    }

    /* The arguments, which for an invoke begin with the receiver. */
    for (unsigned i = 0; i < nargs; i++) {
        unsigned idx = first + i;
        SlotKind k = e->stack[idx];
        if (k != SLOT_INT && k != SLOT_FLOAT && k != SLOT_BOOL &&
            k != SLOT_INST && k != SLOT_LIST && k != SLOT_OBJ &&
            k != SLOT_ITER) {
            e->whyNot = "an argument kind this call cannot pass";
            return false;
        }
        unsigned tag = k == SLOT_INT   ? VAL_INT
                     : k == SLOT_FLOAT ? VAL_FLOAT
                     : k == SLOT_BOOL  ? VAL_BOOL
                                       : VAL_OBJ;
        unsigned at = d + (unsigned)offsetof(JitCallDesc, args) +
                      i * (unsigned)sizeof(Value);
        unsigned reg = JIT_FIRST_SAVED + regBase(e) +
                       (e->usesUpvalues ? 1u : 0u) +
                       (idx - (e->depth - e->valueDepth));
        emit(e, jaiA64MovzX(JIT_SCRATCH_B, tag, 0));
        emit(e, jaiA64StrW(JIT_SCRATCH_B, 31, at));
        emit(e, jaiA64StrX(reg, 31, at + 8));
    }

    /* Everything this body is holding that the collector must see. */
    unsigned nroots = 0;
    for (unsigned slot = e->base; slot < e->base + e->locals; slot++) {
        if (e->localKind[slot] != SLOT_INST &&
            e->localKind[slot] != SLOT_LIST &&
            e->localKind[slot] != SLOT_OBJ &&
            e->localKind[slot] != SLOT_ITER) {
            continue;
        }
        if (nroots >= JIT_MAX_SAVED) { e->whyNot = "too many roots"; return false; }
        unsigned at = d + (unsigned)offsetof(JitCallDesc, roots) +
                      nroots * (unsigned)sizeof(Value);
        emit(e, jaiA64MovzX(JIT_SCRATCH_B, VAL_OBJ, 0));
        emit(e, jaiA64StrW(JIT_SCRATCH_B, 31, at));
        emit(e, jaiA64StrX(localIn(e, slot, JIT_SCRATCH_C), 31, at + 8));
        nroots++;
    }

    /* And what the operand stack is holding. Locals alone were enough only
     * while nothing object-shaped stayed on the stack across a call -- but an
     * iterator does exactly that: OP_GET_ITER pushes an ObjIter that lives in
     * a register for the whole loop, and the call this descriptor belongs to
     * may be the one that collects it. Visible only under --gc-stress, and
     * only once a body with a loop like that could compile at all. */
    for (unsigned idx = e->depth - e->valueDepth; idx < e->depth; idx++) {
        SlotKind k = e->stack[idx];
        if (k != SLOT_INST && k != SLOT_LIST && k != SLOT_OBJ &&
            k != SLOT_ITER) {
            continue;
        }
        if (nroots >= JIT_MAX_SAVED) { e->whyNot = "too many roots"; return false; }
        unsigned at = d + (unsigned)offsetof(JitCallDesc, roots) +
                      nroots * (unsigned)sizeof(Value);
        unsigned reg = JIT_FIRST_SAVED + regBase(e) +
                       (e->usesUpvalues ? 1u : 0u) +
                       (idx - (e->depth - e->valueDepth));
        emit(e, jaiA64MovzX(JIT_SCRATCH_B, VAL_OBJ, 0));
        emit(e, jaiA64StrW(JIT_SCRATCH_B, 31, at));
        emit(e, jaiA64StrX(reg, 31, at + 8));
        nroots++;
    }

    emit(e, jaiA64MovzX(JIT_SCRATCH_A, nargs, 0));
    emit(e, jaiA64StrX(JIT_SCRATCH_A, 31,
                       d + (unsigned)offsetof(JitCallDesc, argc)));
    emit(e, jaiA64MovzX(JIT_SCRATCH_A, nroots, 0));
    emit(e, jaiA64StrX(JIT_SCRATCH_A, 31,
                       d + (unsigned)offsetof(JitCallDesc, nroots)));

    emit(e, jaiA64AddXImm(0, 31, d));
    emitConst64(e, JIT_SCRATCH_A, (int64_t)(uintptr_t)helper);
    emit(e, jaiA64Blr(JIT_SCRATCH_A));

    if (ownStatus) return true;

    /* Nonzero means the callee raised; the interpreter owns it from here. */
    emit(e, jaiA64SubsXImm(31, 0, 0));
    if (e->fixupCount >= JIT_MAX_FIXUPS) { e->failed = true; return false; }
    e->fixups[e->fixupCount].instIndex    = (int)e->count;
    e->fixups[e->fixupCount].targetOffset = FIXUP_THREW;
    e->fixups[e->fixupCount].conditional  = true;
    e->fixups[e->fixupCount].depth        = -1;
    e->fixupCount++;
    emit(e, jaiA64BCond(JAI_A64_NE, 0));
    return true;
}

static bool emitDescriptor(Emit *e, Value calleeVal, unsigned first,
                           unsigned nargs, void *helper) {
    return emitDescriptorStatus(e, calleeVal, first, nargs, helper, false, -1);
}

/* A call to a global function that has itself compiled. Its return kind types
 * the result; the tag that actually comes back is checked, and a surprise
 * deopts to the instruction after the call, since the call has happened. */
static bool emitGlobalCall(Emit *e, unsigned argc, uint32_t after) {
    unsigned cidx = e->depth - argc - 1;
    Value cv = e->stackSeen[cidx];
    if (!IS_CLOSURE(cv)) { e->whyNot = "callee vanished"; return false; }
    ObjFunction *cfn = AS_CLOSURE(cv)->fn;
    if (cfn->jitFunc == NULL) { e->whyNot = "callee no longer compiled"; return false; }
    SlotKind rk = (SlotKind)cfn->jitReturnKind;
    uint32_t rshape = cfn->jitReturnShape;
    ObjClass *rcls = NULL;
    if (rk == SLOT_INST) {
        if (rshape == 0 || !jaiClassForShape(rshape, &rcls) || rcls == NULL) {
            e->whyNot = "callee's return class not on record";
            return false;
        }
    }
    if (rk != SLOT_INT && rk != SLOT_FLOAT && rk != SLOT_BOOL &&
        rk != SLOT_INST && rk != SLOT_LIST && rk != SLOT_OBJ) {
        e->whyNot = "callee's return kind not usable";
        return false;
    }

    if (!emitDescriptor(e, cv, e->depth - argc, argc, (void *)&jitCallOut)) {
        return false;
    }
    for (unsigned i = 0; i < argc; i++) {
        unsigned r;
        if (!popValue(e, &r, NULL)) return false;
    }
    if (e->depth == 0 || e->stack[e->depth - 1] != SLOT_FUNC) return false;
    e->depth--;
    if (!pushValue(e, rk, rshape, rcls)) return false;

    unsigned rat = e->descOffset + (unsigned)offsetof(JitCallDesc, result);
    unsigned wantTag = rk == SLOT_INT   ? VAL_INT
                     : rk == SLOT_FLOAT ? VAL_FLOAT
                     : rk == SLOT_BOOL  ? VAL_BOOL
                                        : VAL_OBJ;
    emit(e, jaiA64LdrW(JIT_SCRATCH_A, 31, rat));
    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, wantTag));
    branchOnDeoptAt(e, JAI_A64_NE, after, true);
    emit(e, jaiA64LdrX(pushReg(e) - 1, 31, rat + 8));
    e->wroteHeap = true;
    return true;
}

/* Emit a method's body directly, when that body is one expression.
 *
 * Deliberately narrow: no jumps, no stores, no calls, only field reads of its
 * own parameters and int or float arithmetic. Those restrictions are what make
 * a second walker over the callee's bytecode safe to write -- with no branches
 * there is no offset map to keep, and with no stores there is nothing to undo
 * if a guard inside it deoptimises to the call site. */
static bool inlineMethod(Emit *e, ObjClosure *closure, uint32_t nameIdx,
                         unsigned argc, int callOff) {
    if (e->noInline) return false;
    unsigned ridx = e->depth - argc - 1;
    ObjClass *rcls = e->stackClass[ridx];
    if (rcls == NULL) return false;
    ObjFunction *cfn = closure->fn;
    if (nameIdx >= (uint32_t)cfn->chunk.constants.count) return false;
    Value mname = cfn->chunk.constants.data[nameIdx];
    if (!IS_STRING(mname)) return false;
    Value method;
    if (!jaiClassFindMethod(rcls, AS_STRING(mname), &method)) return false;
    if (!IS_CLOSURE(method)) return false;
    ObjFunction *mfn = AS_CLOSURE(method)->fn;
    if (mfn->arity != argc || mfn->defaultCount != 0) return false;
    if (mfn->flags & (FN_VARIADIC | FN_KWREST | FN_INIT)) return false;
    if (mfn->upvalueCount != 0) return false;
    if (mfn->chunk.count > 96) return false;

    /* The receiver and arguments, as the caller holds them. */
    unsigned base = JIT_FIRST_SAVED + regBase(e) + (e->usesUpvalues ? 1u : 0u);
    unsigned inReg[JIT_MAX_ARGS_OUT + 1];
    Value    inSeen[JIT_MAX_ARGS_OUT + 1];
    ObjClass *inCls[JIT_MAX_ARGS_OUT + 1];
    for (unsigned i = 0; i <= argc; i++) {
        unsigned idx = ridx + i;
        if (!holdsRegister(e->stack[idx])) return false;
        inReg[i]  = base + (idx - (e->depth - e->valueDepth));
        inSeen[i] = e->stackSeen[idx];
        inCls[i]  = e->stackClass[idx];
    }

    /* A dry walk first: nothing is emitted until the whole body is known to
     * be expressible, because a half-inlined body cannot be taken back. */
    const uint8_t *c = mfn->chunk.code;
    int n = mfn->chunk.count;
    for (int pass = 0; pass < 2; pass++) {
        int depth0 = (int)e->depth;
        for (int o = 0; o < n;) {
            uint8_t op = c[o];
            if (op == OP_GET_FIELD_LOCAL) {
                unsigned slot = jaiReadU16(c + o + 1);
                uint32_t nidx = jaiReadU24(c + o + 3);
                if (slot > argc) return false;
                if (e->stack[ridx + slot] != SLOT_INST) return false;
                if (nidx >= (uint32_t)mfn->chunk.constants.count) return false;
                Value fname = mfn->chunk.constants.data[nidx];
                if (!IS_STRING(fname)) return false;
                const FieldInfo *fi =
                    jaiClassFieldInfo(inCls[slot], AS_STRING(fname));
                if (fi == NULL || fi->isStatic) return false;
                if (!IS_INSTANCE(inSeen[slot])) return false;
                ObjInstance *si = AS_INSTANCE(inSeen[slot]);
                if (fi->slot >= si->fieldCount) return false;
                Value fv = si->fields[fi->slot];
                SlotKind fk; unsigned ftag;
                if (IS_INT(fv))        { fk = SLOT_INT;   ftag = VAL_INT; }
                else if (IS_FLOAT(fv)) { fk = SLOT_FLOAT; ftag = VAL_FLOAT; }
                else return false;
                unsigned fbase = (unsigned)offsetof(ObjInstance, fields) +
                                 (unsigned)fi->slot * (unsigned)sizeof(Value);
                if (pass == 1) {
                    emit(e, jaiA64LdrW(JIT_SCRATCH_A, inReg[slot], fbase));
                    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, ftag));
                    branchOnDeoptAt(e, JAI_A64_NE, (uint32_t)callOff, false);
                }
                if (!pushValue(e, fk, 0, NULL)) return false;
                if (pass == 1) {
                    emit(e, jaiA64LdrX(pushReg(e) - 1, inReg[slot], fbase + 8));
                }
                o += 8;
                continue;
            }
            if (op == OP_ADD || op == OP_SUB || op == OP_MUL) {
                unsigned rb, ra; SlotKind kb, ka;
                if (!popValue(e, &rb, &kb)) return false;
                if (!popValue(e, &ra, &ka)) return false;
                if (ka != kb) return false;
                if (ka != SLOT_INT && ka != SLOT_FLOAT) return false;
                if (!pushValue(e, ka, 0, NULL)) return false;
                unsigned rd = pushReg(e) - 1;
                if (pass == 1) {
                    if (ka == SLOT_FLOAT) {
                        emit(e, jaiA64FmovDX(JIT_FSCRATCH_A, ra));
                        emit(e, jaiA64FmovDX(JIT_FSCRATCH_B, rb));
                        emit(e, op == OP_ADD
                                 ? jaiA64FaddD(JIT_FSCRATCH_A, JIT_FSCRATCH_A, JIT_FSCRATCH_B)
                             : op == OP_SUB
                                 ? jaiA64FsubD(JIT_FSCRATCH_A, JIT_FSCRATCH_A, JIT_FSCRATCH_B)
                                 : jaiA64FmulD(JIT_FSCRATCH_A, JIT_FSCRATCH_A, JIT_FSCRATCH_B));
                        emit(e, jaiA64FmovXD(rd, JIT_FSCRATCH_A));
                    } else if (op == OP_MUL) {
                        emit(e, jaiA64SmulhX(JIT_SCRATCH_A, ra, rb));
                        emit(e, jaiA64MulX(rd, ra, rb));
                        emit(e, jaiA64SubsXAsr(31, JIT_SCRATCH_A, rd, 63));
                        branchOnOverflow(e, 2u, JAI_A64_NE);
                    } else {
                        emit(e, op == OP_ADD ? jaiA64AddsX(rd, ra, rb)
                                             : jaiA64SubsXReg(rd, ra, rb));
                        branchOnOverflow(e, op == OP_ADD ? 0u : 1u, JAI_A64_VS);
                    }
                }
                o += 1;
                continue;
            }
            if (op == OP_RETURN) {
                if ((int)e->depth != depth0 + 1) return false;
                o += 1;
                if (o != n) return false;
                break;
            }
            return false;
        }
        if (pass == 0) {
            /* Undo the model changes the dry walk made. */
            while ((int)e->depth > depth0) {
                unsigned r; if (!popValue(e, &r, NULL)) return false;
            }
        }
    }

    /* The result is on top; the receiver and arguments below it go away. */
    unsigned rres;
    SlotKind kres;
    if (!popValue(e, &rres, &kres)) return false;
    for (unsigned i = 0; i <= argc; i++) {
        unsigned r; if (!popValue(e, &r, NULL)) return false;
    }
    if (!pushValue(e, kres, 0, NULL)) return false;
    unsigned dst = pushReg(e) - 1;
    if (dst != rres) emit(e, jaiA64MovX(dst, rres));
    e->inlined = true;
    return true;
}

static bool emitCallOut(Emit *e, unsigned argc) {
    ObjClass *cls = e->stackClass[e->depth - argc - 1];
    if (cls == NULL) { e->whyNot = "callee class"; return false; }

    uint16_t fslots[JIT_MAX_ARGS_OUT];
    if (argc <= JIT_MAX_ARGS_OUT && simpleInitFields(cls, argc, fslots)) {
        /* Allocate, then store the arguments into their fields here. */
        unsigned first = e->depth - argc;
        SlotKind kinds[JIT_MAX_ARGS_OUT];
        unsigned regs[JIT_MAX_ARGS_OUT];
        for (unsigned i = 0; i < argc; i++) {
            kinds[i] = e->stack[first + i];
            if (kinds[i] != SLOT_INT && kinds[i] != SLOT_FLOAT &&
                kinds[i] != SLOT_BOOL && kinds[i] != SLOT_INST &&
                kinds[i] != SLOT_LIST && kinds[i] != SLOT_OBJ) {
                e->whyNot = "an argument kind a field cannot take";
                return false;
            }
            regs[i] = JIT_FIRST_SAVED + regBase(e) +
                      (e->usesUpvalues ? 1u : 0u) +
                      (first + i - (e->depth - e->valueDepth));
        }
        if (!emitDescriptor(e, OBJ_VAL((Obj *)cls), first, 0,
                            (void *)&jitNewInstance)) {
            return false;
        }
        for (unsigned i = 0; i < argc; i++) {
            unsigned r;
            if (!popValue(e, &r, NULL)) return false;
        }
        if (e->depth == 0 || e->stack[e->depth - 1] != SLOT_CLASS) return false;
        e->depth--;
        if (!pushValue(e, SLOT_INST, cls->shapeId, cls)) return false;
        /* The result reuses the register the first argument was in, so the
         * instance is held in a scratch until every field has been stored.
         * Loading it into its final register first overwrote the argument it
         * was about to store, and alloc_churn came back in 5ms with the wrong
         * answer -- the same aliasing mistake this tier keeps making. */
        unsigned rinst = pushReg(e) - 1;
        emit(e, jaiA64LdrX(JIT_SCRATCH_C, 31,
                           e->descOffset +
                               (unsigned)offsetof(JitCallDesc, result) + 8));
        for (unsigned i = 0; i < argc; i++) {
            unsigned tag = kinds[i] == SLOT_INT   ? VAL_INT
                         : kinds[i] == SLOT_FLOAT ? VAL_FLOAT
                         : kinds[i] == SLOT_BOOL  ? VAL_BOOL
                                                  : VAL_OBJ;
            unsigned at = (unsigned)offsetof(ObjInstance, fields) +
                          (unsigned)fslots[i] * (unsigned)sizeof(Value);
            emit(e, jaiA64MovzX(JIT_SCRATCH_A, tag, 0));
            emit(e, jaiA64StrW(JIT_SCRATCH_A, JIT_SCRATCH_C, at));
            emit(e, jaiA64StrX(regs[i], JIT_SCRATCH_C, at + 8));
        }
        emit(e, jaiA64MovX(rinst, JIT_SCRATCH_C));
        e->wroteHeap = true;
        return true;
    }

    if (!emitDescriptor(e, OBJ_VAL((Obj *)cls), e->depth - argc, argc,
                        (void *)&jitCallOut)) {
        return false;
    }

    for (unsigned i = 0; i < argc; i++) {
        unsigned r;
        if (!popValue(e, &r, NULL)) { e->whyNot = "call argument"; return false; }
    }
    if (e->depth == 0 || e->stack[e->depth - 1] != SLOT_CLASS) {
        e->whyNot = "callee was not where it should be";
        return false;
    }
    e->depth--;

    if (!pushValue(e, SLOT_INST, cls->shapeId, cls)) return false;
    emit(e, jaiA64LdrX(pushReg(e) - 1, 31,
                       e->descOffset + (unsigned)offsetof(JitCallDesc, result) + 8));
    /* A call is an effect: no bail may follow it, for the same reason no bail
     * may follow a store. */
    e->wroteHeap = true;
    return true;
}

/* An unconditional OP_LOOP or OP_JUMP does not fall through, so the instruction
 * after it is reachable only by a branch. The linear walk would otherwise carry
 * the preceding instruction's operand stack across that gap. Take the stack
 * from a branch that targets this offset instead, accepting only a pure
 * truncation: register entries are the top `valueDepth` of the stack, so
 * popping from the top keeps `depth - valueDepth`, and with it every register
 * index below the join. */
static void reconcileAfterUncond(Emit *e, uint32_t off) {
    for (unsigned i = 0; i < e->fixupCount; i++) {
        if (e->fixups[i].targetOffset != off) continue;
        int want = e->fixups[i].depth;
        if (want < 0) continue;
        unsigned d = (unsigned)want & 0xfu;
        if (d > e->depth) continue;
        if ((int)stackSignatureAt(e, d) != want) continue;
        unsigned popped = e->depth - d;
        if (popped > e->valueDepth) continue;
        e->depth = d;
        e->valueDepth -= popped;
        return;
    }
}

static bool compileBody(Emit *e, ObjClosure *closure) {
    ObjFunction *fn = closure->fn;
    const uint8_t *code = fn->chunk.code;
    int count = fn->chunk.count;

    int start = e->osr ? (int)e->osrTop : 0;
    int stop  = e->osr ? (int)e->osrEnd : count;
    bool afterUncond = false;
    for (int off = start; off < stop && !e->failed;) {
        if (afterUncond) reconcileAfterUncond(e, (uint32_t)off);
        afterUncond = (code[off] == OP_JUMP || code[off] == OP_LOOP);
        e->offsetToInst[off]  = (int)e->count;
        e->offsetToDepth[off] = (int)stackSignature(e);
        uint8_t op = code[off];
        e->lastOp = op;
        e->curOffset = (uint32_t)off;

        switch (op) {
        case OP_GET_LOCAL: {
            unsigned slot = jaiReadU16(code + off + 1);
            if (!localInRange(e, slot)) return false;
            if (e->localKind[slot] == SLOT_OPAQUE) return false;
            if (slot == 0) e->usesSlot0 = true;
            if (!pushValue3(e, e->localKind[slot], e->localShape[slot],
                            e->localClass[slot],
                            localObserved(e, slot) ? e->observed[slot]
                                                   : e->localSeen[slot],
                            (int)slot)) {
                return false;
            }
            {
                unsigned dst = pushReg(e) - 1;
                unsigned src = localIn(e, slot, dst);
                if (src != dst) emit(e, jaiA64MovX(dst, src));
            }
            off += 3;
            break;
        }

        case OP_INT: {
            int16_t k = jaiReadI16(code + off + 1);
            if (!pushValue(e, SLOT_INT, 0, NULL)) return false;
            emitConst64(e, pushReg(e) - 1, k);
            off += 3;
            break;
        }

        case OP_TRUE:
        case OP_FALSE: {
            /* A bool is a payload of 1 or 0 in a register, the same as any
             * other value here. Their absence declined `queens` and `sieve`
             * outright -- `return false` and a list of flags are not exotic. */
            if (!pushValue(e, SLOT_BOOL, 0, NULL)) return false;
            emitConst64(e, pushReg(e) - 1, op == OP_TRUE ? 1 : 0);
            off += 1;
            break;
        }

        case OP_SET_LOCAL: {
            /* Assigns without popping: the value stays as the statement's
             * result, which is what the interpreter does. */
            unsigned slot = jaiReadU16(code + off + 1);
            if (!localInRange(e, slot)) return false;
            if (slot == 0) e->usesSlot0 = true;
            if (e->depth == 0 || !holdsRegister(e->stack[e->depth - 1])) return false;
            /* A local keeps one kind for the whole function. Two kinds would
             * mean the reads of it cannot be compiled to one instruction, and
             * the join check works on the operand stack, not on locals. */
            if (!adoptLocalKind(e, slot, e->stack[e->depth - 1],
                                e->stackShape[e->depth - 1],
                                e->stackClass[e->depth - 1])) {
                e->whyNot = "a local was given two different kinds";
                return false;
            }
            localOut(e, slot, pushReg(e) - 1);
            off += 3;
            break;
        }

        case OP_ADD_LOCALS: {
            unsigned a = jaiReadU16(code + off + 1);
            unsigned b = jaiReadU16(code + off + 3);
            if (!localInRange(e, a) || !localInRange(e, b)) return false;
            SlotKind ka2 = e->localKind[a];
            if (ka2 != e->localKind[b]) return false;
            if (ka2 != SLOT_INT && ka2 != SLOT_FLOAT) return false;
            if (a == 0 || b == 0) e->usesSlot0 = true;
            if (!pushValue(e, ka2, 0, NULL)) return false;
            {
                unsigned ra2 = localIn(e, a, JIT_SCRATCH_C);
                unsigned rb2 = localIn(e, b, JIT_SCRATCH_D);
                if (ka2 == SLOT_FLOAT) {
                    emit(e, jaiA64FmovDX(JIT_FSCRATCH_A, ra2));
                    emit(e, jaiA64FmovDX(JIT_FSCRATCH_B, rb2));
                    emit(e, jaiA64FaddD(JIT_FSCRATCH_A, JIT_FSCRATCH_A,
                                        JIT_FSCRATCH_B));
                    emit(e, jaiA64FmovXD(pushReg(e) - 1, JIT_FSCRATCH_A));
                } else {
                    emit(e, jaiA64AddsX(pushReg(e) - 1, ra2, rb2));
                    branchOnOverflow(e, 0u, JAI_A64_VS);
                }
            }
            off += 5;
            break;
        }

        case OP_ADD_BIND: {
            /* `ADD; BIND a` fused. Floats go through the same fmov pair the
             * plain add uses; ints keep the overflow check. */
            unsigned slot = jaiReadU16(code + off + 1);
            if (!localInRange(e, slot)) return false;
            if (slot == 0) e->usesSlot0 = true;
            unsigned rb, ra;
            SlotKind kb, ka;
            if (!popValue(e, &rb, &kb)) return false;
            if (!popValue(e, &ra, &ka)) return false;
            if (ka != kb) return false;
            if (!adoptLocalKind(e, slot, ka, 0, NULL)) {
                e->whyNot = "a local was given two different kinds";
                return false;
            }
            unsigned rd = localDest(e, slot);
            if (ka == SLOT_FLOAT) {
                emit(e, jaiA64FmovDX(JIT_FSCRATCH_A, ra));
                emit(e, jaiA64FmovDX(JIT_FSCRATCH_B, rb));
                emit(e, jaiA64FaddD(JIT_FSCRATCH_A, JIT_FSCRATCH_A,
                                    JIT_FSCRATCH_B));
                emit(e, jaiA64FmovXD(rd, JIT_FSCRATCH_A));
            } else if (ka == SLOT_INT) {
                emit(e, jaiA64AddsX(rd, ra, rb));
                branchOnOverflow(e, 0u, JAI_A64_VS);
            } else {
                e->whyNot = "add-bind of a kind that is neither int nor float";
                return false;
            }
            localOut(e, slot, rd);
            off += 3;
            break;
        }

        case OP_ADD_INT_CONST: {
            unsigned slot = jaiReadU16(code + off + 1);
            int16_t  imm  = jaiReadI16(code + off + 3);
            if (!localInRange(e, slot)) return false;
            if (e->localKind[slot] != SLOT_INT) return false;
            if (slot == 0) e->usesSlot0 = true;
            if (!pushValue(e, SLOT_INT, 0, NULL)) return false;
            emitConst64(e, JIT_SCRATCH_A, imm);
            emit(e, jaiA64AddsX(pushReg(e) - 1,
                                localIn(e, slot, JIT_SCRATCH_C),
                                JIT_SCRATCH_A));
            branchOnOverflow(e, 0u, JAI_A64_VS);
            off += 5;
            break;
        }

        case OP_MUL_BIND: {
            unsigned slot = jaiReadU16(code + off + 1);
            if (!localInRange(e, slot)) return false;
            if (slot == 0) e->usesSlot0 = true;
            unsigned rb, ra;
            SlotKind kb, ka;
            if (!popValue(e, &rb, &kb)) return false;
            if (!popValue(e, &ra, &ka)) return false;
            if (ka != kb) return false;
            if (!adoptLocalKind(e, slot, ka, 0, NULL)) {
                e->whyNot = "a local was given two different kinds";
                return false;
            }
            unsigned rd = localDest(e, slot);
            if (ka == SLOT_FLOAT) {
                emit(e, jaiA64FmovDX(JIT_FSCRATCH_A, ra));
                emit(e, jaiA64FmovDX(JIT_FSCRATCH_B, rb));
                emit(e, jaiA64FmulD(JIT_FSCRATCH_A, JIT_FSCRATCH_A,
                                    JIT_FSCRATCH_B));
                emit(e, jaiA64FmovXD(rd, JIT_FSCRATCH_A));
            } else if (ka == SLOT_INT) {
                emit(e, jaiA64SmulhX(JIT_SCRATCH_A, ra, rb));
                emit(e, jaiA64MulX(rd, ra, rb));
                emit(e, jaiA64SubsXAsr(31, JIT_SCRATCH_A, rd, 63));
                branchOnOverflow(e, 2u, JAI_A64_NE);
            } else {
                return false;
            }
            localOut(e, slot, rd);
            off += 3;
            break;
        }

        case OP_SUB_BIND: {
            unsigned slot = jaiReadU16(code + off + 1);
            if (!localInRange(e, slot)) return false;
            if (slot == 0) e->usesSlot0 = true;
            unsigned rb, ra;
            SlotKind kb, ka;
            if (!popValue(e, &rb, &kb)) return false;
            if (!popValue(e, &ra, &ka)) return false;
            if (ka != kb) return false;
            if (!adoptLocalKind(e, slot, ka, 0, NULL)) {
                e->whyNot = "a local was given two different kinds";
                return false;
            }
            unsigned rd = localDest(e, slot);
            if (ka == SLOT_FLOAT) {
                emit(e, jaiA64FmovDX(JIT_FSCRATCH_A, ra));
                emit(e, jaiA64FmovDX(JIT_FSCRATCH_B, rb));
                emit(e, jaiA64FsubD(JIT_FSCRATCH_A, JIT_FSCRATCH_A,
                                    JIT_FSCRATCH_B));
                emit(e, jaiA64FmovXD(rd, JIT_FSCRATCH_A));
            } else if (ka == SLOT_INT) {
                emit(e, jaiA64SubsXReg(rd, ra, rb));
                branchOnOverflow(e, 1u, JAI_A64_VS);
            } else {
                return false;
            }
            localOut(e, slot, rd);
            off += 3;
            break;
        }

        case OP_BIND: {
            unsigned slot = jaiReadU16(code + off + 1);
            if (!localInRange(e, slot)) return false;
            if (slot == 0) e->usesSlot0 = true;
            if (e->depth == 0 || !holdsRegister(e->stack[e->depth - 1])) return false;
            if (!adoptLocalKindSeen(e, slot, e->stack[e->depth - 1],
                                    e->stackShape[e->depth - 1],
                                    e->stackClass[e->depth - 1],
                                    e->stackSeen[e->depth - 1])) {
                e->whyNot = "a local was given two different kinds";
                return false;
            }
            unsigned r;
            if (!popValue(e, &r, NULL)) return false;
            localOut(e, slot, r);
            off += 3;
            break;
        }

        case OP_INC_LOCAL: {
            unsigned slot = jaiReadU16(code + off + 1);
            int8_t   imm  = (int8_t)code[off + 3];
            if (!localInRange(e, slot)) return false;
            if (e->localKind[slot] != SLOT_INT) return false;
            if (slot == 0) e->usesSlot0 = true;
            emitConst64(e, JIT_SCRATCH_A, imm);
            {
                unsigned cur = localIn(e, slot, JIT_SCRATCH_C);
                unsigned dst = localDest(e, slot);
                emit(e, jaiA64AddsX(dst, cur, JIT_SCRATCH_A));
                localOut(e, slot, dst);
            }
            branchOnOverflow(e, 0u, JAI_A64_VS);
            off += 4;
            break;
        }

        case OP_EQ: case OP_NE:
        case OP_LT: case OP_LE: case OP_GT: case OP_GE: {
            /* Read the operands without taking them off the model: a NaN sends
             * this instruction back to the interpreter, whose stack still has
             * them. Popping first left it two entries short, and the
             * comparison it re-ran then read whatever was underneath -- the
             * error came out one line late instead of not at all, which is a
             * good deal harder to notice. */
            if (e->depth < 2) return false;
            SlotKind ka = e->stack[e->depth - 2], kb = e->stack[e->depth - 1];
            if (ka != kb) return false;
            if (!holdsRegister(ka)) return false;
            unsigned rb = pushReg(e) - 1, ra = pushReg(e) - 2;
            unsigned cond;
            switch (op) {
            case OP_EQ: cond = JAI_A64_EQ; break;
            case OP_NE: cond = JAI_A64_NE; break;
            case OP_LT: cond = ka == SLOT_FLOAT ? JAI_A64_MI : JAI_A64_LT; break;
            case OP_LE: cond = ka == SLOT_FLOAT ? JAI_A64_LS : JAI_A64_LE; break;
            case OP_GT: cond = JAI_A64_GT; break;
            default:    cond = ka == SLOT_FLOAT ? JAI_A64_GE : JAI_A64_GE; break;
            }
            if (ka == SLOT_FLOAT) {
                emit(e, jaiA64FmovDX(JIT_FSCRATCH_A, ra));
                emit(e, jaiA64FmovDX(JIT_FSCRATCH_B, rb));
                emit(e, jaiA64FcmpD(JIT_FSCRATCH_A, JIT_FSCRATCH_B));
                nanToDeopt(e);
            } else if (ka == SLOT_INT) {
                emit(e, jaiA64SubsXReg(31, ra, rb));
            } else {
                return false;
            }
            unsigned dropA, dropB;
            if (!popValue(e, &dropB, NULL)) return false;
            if (!popValue(e, &dropA, NULL)) return false;
            if (!pushValue(e, SLOT_BOOL, 0, NULL)) return false;
            emit(e, jaiA64CsetX(pushReg(e) - 1, cond));
            off += 1;
            break;
        }

        case OP_JUMP_IF_FALSE:
        case OP_JUMP_IF_TRUE:
        case OP_JUMP_IF_FALSE_KEEP:
        case OP_JUMP_IF_TRUE_KEEP: {
            int16_t jump = jaiReadI16(code + off + 1);
            bool keep = (op == OP_JUMP_IF_FALSE_KEEP ||
                         op == OP_JUMP_IF_TRUE_KEEP);
            bool wantTrue = (op == OP_JUMP_IF_TRUE ||
                             op == OP_JUMP_IF_TRUE_KEEP);
            if (e->depth == 0 || e->stack[e->depth - 1] != SLOT_BOOL) return false;
            unsigned r = pushReg(e) - 1;
            if (!keep) {
                unsigned popped;
                if (!popValue(e, &popped, NULL)) return false;
            }
            emit(e, jaiA64SubsXImm(31, r, 0));
            branchTo(e, (uint32_t)((int32_t)(off + 3) + jump), true,
                     wantTrue ? JAI_A64_NE : JAI_A64_EQ);
            off += 3;
            break;
        }

        case OP_JUMP_IF_CMP_FALSE: {
            uint8_t  cmp  = code[off + 1];
            int16_t  jump = jaiReadI16(code + off + 2);
            if (e->depth < 2) return false;
            SlotKind ka = e->stack[e->depth - 2], kb = e->stack[e->depth - 1];
            if (ka != kb) return false;
            if (!holdsRegister(ka)) return false;
            unsigned rb = pushReg(e) - 1, ra = pushReg(e) - 2;
            unsigned cond;
            if (!negatedCondition(cmp, &cond)) return false;
            if (ka == SLOT_FLOAT) {
                /* fcmp's answers for the ordered comparisons are not the
                 * signed-integer ones: less-than is MI and less-or-equal is
                 * LS, because an unordered result must come out false. */
                switch (cmp) {
                case OP_LT: cond = JAI_A64_GE; break;   /* not (a < b)  */
                case OP_LE: cond = JAI_A64_GT; break;   /* not (a <= b) */
                case OP_GT: cond = JAI_A64_LS; break;   /* not (a > b)  */
                case OP_GE: cond = JAI_A64_MI; break;   /* not (a >= b) */
                case OP_EQ: cond = JAI_A64_NE; break;
                case OP_NE: cond = JAI_A64_EQ; break;
                default: return false;
                }
                emit(e, jaiA64FmovDX(JIT_FSCRATCH_A, ra));
                emit(e, jaiA64FmovDX(JIT_FSCRATCH_B, rb));
                emit(e, jaiA64FcmpD(JIT_FSCRATCH_A, JIT_FSCRATCH_B));
                nanToDeopt(e);
            } else if (ka == SLOT_INT) {
                emit(e, jaiA64SubsXReg(31, ra, rb));
            } else {
                return false;
            }
            unsigned dropA2, dropB2;
            if (!popValue(e, &dropB2, NULL)) return false;
            if (!popValue(e, &dropA2, NULL)) return false;
            branchTo(e, (uint32_t)((int32_t)(off + 4) + jump), true, cond);
            off += 4;
            break;
        }

        case OP_POP: {
            unsigned r;
            if (!popValue(e, &r, NULL)) return false;
            off += 1;
            break;
        }

        case OP_CONST: {
            uint32_t idx = jaiReadU24(code + off + 1);
            if (idx >= (uint32_t)fn->chunk.constants.count) return false;
            Value k = fn->chunk.constants.data[idx];
            if (IS_INT(k)) {
                if (!pushValue3(e, SLOT_INT, 0, NULL, k, -1)) return false;
                emitConst64(e, pushReg(e) - 1, AS_INT(k));
            } else if (IS_FLOAT(k)) {
                /* The bits, not the number: a float lives in an X register
                 * exactly as it does in a Value's payload. */
                double d = AS_FLOAT(k);
                int64_t bits;
                memcpy(&bits, &d, sizeof bits);
                if (!pushValue3(e, SLOT_FLOAT, 0, NULL, k, -1)) return false;
                emitConst64(e, pushReg(e) - 1, bits);
            } else if (IS_BOOL(k)) {
                if (!pushValue3(e, SLOT_BOOL, 0, NULL, k, -1)) return false;
                emitConst64(e, pushReg(e) - 1, AS_BOOL(k) ? 1 : 0);
            } else if (IS_STRING(k)) {
                /* A pointer to the constant pool's own string. Safe to hold
                 * raw for the same reason the resolved globals above are: the
                 * pool belongs to the chunk, the chunk to the function, and
                 * the caller is holding the closure for the whole call. A
                 * string constant was the commonest reason this tier declined
                 * a body -- thirty refusals across the benchmark suite. */
                if (!pushValue3(e, SLOT_OBJ, 0, NULL, k, -1)) return false;
                emitConst64(e, pushReg(e) - 1, (int64_t)(uintptr_t)AS_OBJ(k));
            } else {
                return false;
            }
            off += 4;
            break;
        }

        case OP_TYPE_GUARD: {
            /* A declared boundary -- a parameter or a return type. The kind is
             * already known here, so the guard is either nothing at all or the
             * int-to-float widening the interpreter does at the same place
             * (spec 2.2). Anything the kinds cannot settle is declined rather
             * than guessed: `evalA` in spectral is a one-line function whose
             * whole body was refused for want of this. */
            uint32_t idx = jaiReadU24(code + off + 1);
            if (idx >= (uint32_t)fn->chunk.constants.count) return false;
            Value t = fn->chunk.constants.data[idx];
            if (!IS_STRING(t)) return false;
            if (e->depth == 0) return false;
            const char *tn = AS_STRING(t)->chars;
            SlotKind k = e->stack[e->depth - 1];
            if (strcmp(tn, "float") == 0) {
                if (k == SLOT_INT) {
                    unsigned r = pushReg(e) - 1;
                    emit(e, jaiA64ScvtfDX(JIT_FSCRATCH_A, r));
                    emit(e, jaiA64FmovXD(r, JIT_FSCRATCH_A));
                    e->stack[e->depth - 1]      = SLOT_FLOAT;
                    e->stackShape[e->depth - 1] = 0;
                    e->stackClass[e->depth - 1] = NULL;
                    e->stackSeen[e->depth - 1]  = NULL_VAL;
                    e->stackLocal[e->depth - 1] = -1;
                } else if (k != SLOT_FLOAT) {
                    e->whyNot = "a type guard the kinds cannot settle";
                    return false;
                }
            } else if ((strcmp(tn, "int") == 0 && k == SLOT_INT) ||
                       (strcmp(tn, "bool") == 0 && k == SLOT_BOOL)) {
                /* Already what the boundary asks for. */
            } else {
                e->whyNot = "a type guard the kinds cannot settle";
                return false;
            }
            off += 4;
            break;
        }

        case OP_FORMAT: {
            /* Ninety refusals across the benchmarks, the largest entry in the
             * census by a wide margin: every f-string is one of these, and
             * dict_ops, word_freq and string_build all build their keys with
             * one. */
            unsigned parts = code[off + 1];
            if (parts == 0 || parts > JIT_MAX_ARGS_OUT) {
                e->whyNot = "an f-string with more parts than the descriptor holds";
                return false;
            }
            if (e->depth < parts) return false;
            /* `str` bound in the module means every part goes through it
             * instead, which is a call this does not make. */
            {
                ObjModule *fmod = closure->fn->module;
                Value bound;
                ObjString *sname = jaiStringIntern("str", 3);
                if (fmod == NULL || sname == NULL ||
                    jaiTableGetInterned(&fmod->globals, sname, &bound)) {
                    e->whyNot = "the module binds its own str";
                    return false;
                }
            }
            if (!emitDescriptor(e, NULL_VAL, e->depth - parts, parts,
                                (void *)&jitFormat)) {
                return false;
            }
            for (unsigned i = 0; i < parts; i++) {
                unsigned drop;
                if (!popValue(e, &drop, NULL)) return false;
            }
            if (!pushValue(e, SLOT_OBJ, 0, NULL)) return false;
            emit(e, jaiA64LdrX(pushReg(e) - 1, 31,
                               e->descOffset +
                                   (unsigned)offsetof(JitCallDesc, result) + 8));
            e->wroteHeap = true;
            /* count u8, litmask u24, name u24, cache u16 -- nine after the
             * opcode. */
            off += 10;
            break;
        }

        case OP_JUMP: {
            int16_t jump = jaiReadI16(code + off + 1);
            branchTo(e, (uint32_t)((int32_t)(off + 3) + jump), false, 0);
            off += 3;
            break;
        }

        case OP_LOOP: {
            /* The interpreter runs a safepoint on the back edge; compiled code
             * cannot, so a compiled loop is not interruptible and does not get
             * sampled. Both are acceptable only because this tier bails on any
             * unbounded construct: the loop is over ints, cannot allocate, and
             * the stack guard still catches runaway recursion. Ctrl-C during a
             * long compiled loop waits for the loop, which is a real cost and
             * the reason the trip count is not unbounded in practice. */
            int16_t jump = jaiReadI16(code + off + 1);
            branchTo(e, (uint32_t)((int32_t)(off + 3) + jump), false, 0);
            off += 3;
            break;
        }

        case OP_MUL: {
            unsigned rb, ra;
            SlotKind kb, ka;
            if (!popValue(e, &rb, &kb)) return false;
            if (!popValue(e, &ra, &ka)) return false;
            if (ka != kb) return false;

            if (ka == SLOT_FLOAT) {
                if (!pushValue(e, SLOT_FLOAT, 0, NULL)) return false;
                unsigned rf = pushReg(e) - 1;
                emit(e, jaiA64FmovDX(JIT_FSCRATCH_A, ra));
                emit(e, jaiA64FmovDX(JIT_FSCRATCH_B, rb));
                emit(e, jaiA64FmulD(JIT_FSCRATCH_A, JIT_FSCRATCH_A,
                                    JIT_FSCRATCH_B));
                emit(e, jaiA64FmovXD(rf, JIT_FSCRATCH_A));
                off += 1;
                break;
            }
            if (ka != SLOT_INT) return false;
            if (!pushValue(e, SLOT_INT, 0, NULL)) return false;
            unsigned rd = pushReg(e) - 1;
            /* The product overflows exactly when the high half is not the low
             * half's sign bit replicated, so smulh and one shifted compare
             * decide it. mul must come after smulh reads its inputs, since rd
             * may be one of them. */
            emit(e, jaiA64SmulhX(JIT_SCRATCH_A, ra, rb));
            emit(e, jaiA64MulX(rd, ra, rb));
            emit(e, jaiA64SubsXAsr(31, JIT_SCRATCH_A, rd, 63));
            branchOnOverflow(e, 2u, JAI_A64_NE);
            off += 1;
            break;
        }

        case OP_ADD:
        case OP_SUB:
        case OP_DIV: {
            unsigned rb, ra;
            SlotKind kb, ka;
            if (!popValue(e, &rb, &kb)) return false;
            if (!popValue(e, &ra, &ka)) return false;
            if (ka != kb) return false;   /* no implicit widening here */

            if (ka == SLOT_FLOAT) {
                if (!pushValue(e, SLOT_FLOAT, 0, NULL)) return false;
                unsigned rd = pushReg(e) - 1;
                emit(e, jaiA64FmovDX(JIT_FSCRATCH_A, ra));
                emit(e, jaiA64FmovDX(JIT_FSCRATCH_B, rb));
                emit(e, op == OP_ADD
                            ? jaiA64FaddD(JIT_FSCRATCH_A, JIT_FSCRATCH_A,
                                          JIT_FSCRATCH_B)
                        : op == OP_SUB
                            ? jaiA64FsubD(JIT_FSCRATCH_A, JIT_FSCRATCH_A,
                                          JIT_FSCRATCH_B)
                            : jaiA64FdivD(JIT_FSCRATCH_A, JIT_FSCRATCH_A,
                                          JIT_FSCRATCH_B));
                emit(e, jaiA64FmovXD(rd, JIT_FSCRATCH_A));
                off += 1;
                break;
            }

            /* Integer division is not here: it has a zero-divisor error and a
             * truncation rule of its own, and getting either wrong would be a
             * wrong answer rather than a decline. */
            if (ka != SLOT_INT || op == OP_DIV) return false;
            if (!pushValue(e, SLOT_INT, 0, NULL)) return false;
            unsigned rd = pushReg(e) - 1;
            emit(e, op == OP_ADD ? jaiA64AddsX(rd, ra, rb)
                                 : jaiA64SubsXReg(rd, ra, rb));
            branchOnOverflow(e, op == OP_ADD ? 0u : 1u, JAI_A64_VS);
            off += 1;
            break;
        }

        case OP_JUMP_IF_CMP_LOCAL_K: {
            uint8_t  cmp  = code[off + 1];
            unsigned slot = jaiReadU16(code + off + 2);
            uint32_t kIdx = jaiReadU24(code + off + 4);
            int16_t  jump = jaiReadI16(code + off + 7);
            uint32_t next = (uint32_t)(off + 9);

            unsigned cond;
            if (!negatedCondition(cmp, &cond)) return false;
            if (!localInRange(e, slot)) return false;
            if (e->localKind[slot] != SLOT_INT) return false;
            if (slot == 0) e->usesSlot0 = true;
            if (kIdx >= (uint32_t)fn->chunk.constants.count) return false;
            Value k = fn->chunk.constants.data[kIdx];
            if (!IS_INT(k)) return false;

            emitConst64(e, JIT_SCRATCH_A, AS_INT(k));
            emit(e, jaiA64SubsXReg(31, localIn(e, slot, JIT_SCRATCH_C),
                                   JIT_SCRATCH_A));
            branchTo(e, (uint32_t)((int32_t)next + jump), true, cond);
            off += 9;
            break;
        }

        case OP_GET_FIELD_LOCAL: {
            unsigned slot    = jaiReadU16(code + off + 1);
            uint32_t nameIdx = jaiReadU24(code + off + 3);

            if (e->localKind[slot] != SLOT_INST) return false;
            if (slot == 0) e->usesSlot0 = true;
            if (nameIdx >= (uint32_t)fn->chunk.constants.count) return false;
            Value nameVal = fn->chunk.constants.data[nameIdx];
            if (!IS_STRING(nameVal)) return false;

            const FieldInfo *info =
                jaiClassFieldInfo(e->localClass[slot], AS_STRING(nameVal));
            if (info == NULL || info->isStatic) return false;

            /* Which type this field holds is read off the live receiver, so
             * the tier specialises to what the program actually stores rather
             * than to a declaration. That is only possible for a parameter,
             * which is why the slot is capped at the arity above: a local
             * assigned further in has no value to look at yet. */
            /* A parameter has its argument to look at; anything else has
             * whatever was bound to it. */
            Value seen = localObserved(e, slot) ? e->observed[slot]
                                                : e->localSeen[slot];
            if (!IS_INSTANCE(seen)) return false;
            ObjInstance *inst = AS_INSTANCE(seen);
            if (info->slot >= inst->fieldCount) return false;
            Value fieldVal = inst->fields[info->slot];

            SlotKind kind;
            unsigned tag;
            if (IS_INT(fieldVal))        { kind = SLOT_INT;   tag = VAL_INT; }
            else if (IS_FLOAT(fieldVal)) { kind = SLOT_FLOAT; tag = VAL_FLOAT; }
            else return false;

            unsigned base = (unsigned)offsetof(ObjInstance, fields) +
                            (unsigned)info->slot * (unsigned)sizeof(Value);
            /* The tag is checked every time, unless this body wrote the field
             * itself. A field is not typed by the runtime, so a later int
             * where a float was seen must bail rather than be read as one. */
            unsigned recv = localIn(e, slot, JIT_SCRATCH_C);
            SlotKind already = knownFieldKind(e, (int)slot, info->slot);
            if (already != SLOT_SELF) {
                kind = already;
            } else {
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, recv, base));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, tag));
                branchOnDeopt(e, JAI_A64_NE);
            }

            if (!pushValue(e, kind, 0, NULL)) return false;
            emit(e, jaiA64LdrX(pushReg(e) - 1, recv, base + 8));
            off += 8;
            break;
        }

        case OP_GET_LOCAL2: {
            unsigned a = jaiReadU16(code + off + 1);
            unsigned b = jaiReadU16(code + off + 3);
            for (unsigned k = 0; k < 2; k++) {
                unsigned slot = k == 0 ? a : b;
                if (!localInRange(e, slot)) return false;
                if (e->localKind[slot] == SLOT_OPAQUE) return false;
                if (slot == 0) e->usesSlot0 = true;
                if (!pushValue3(e, e->localKind[slot], e->localShape[slot],
                                e->localClass[slot],
                                localObserved(e, slot) ? e->observed[slot]
                                                       : e->localSeen[slot],
                                (int)slot)) {
                    return false;
                }
                {
                    unsigned dst = pushReg(e) - 1;
                    unsigned src = localIn(e, slot, dst);
                    if (src != dst) emit(e, jaiA64MovX(dst, src));
                }
            }
            off += 5;
            break;
        }

        case OP_SET_FIELD: {
            uint32_t nameIdx = jaiReadU24(code + off + 1);
            /* The stack is receiver then value, and both are dropped. The
             * receiver's class is read off its stack entry, not guessed from
             * the locals: two instance locals of different classes would make
             * any guess a silently wrong field offset. */
            if (e->depth < 2) return false;
            ObjClass *klass = e->stackClass[e->depth - 2];
            int recvLocal = e->stackLocal[e->depth - 2];

            unsigned rv, rr;
            SlotKind kv, kr;
            if (!popValue(e, &rv, &kv)) return false;
            if (!popValue(e, &rr, &kr)) return false;
            if (kr != SLOT_INST) return false;
            /* Only a number goes in. Storing an object would put a pointer the
             * collector has not seen into a field, and while this tier cannot
             * allocate -- so nothing can move or be swept while it runs -- the
             * value would still have to be one the caller already had rooted.
             * Not worth the argument for what it buys. */
            if (kv != SLOT_INT && kv != SLOT_FLOAT) return false;
            if (nameIdx >= (uint32_t)fn->chunk.constants.count) return false;
            Value nameVal = fn->chunk.constants.data[nameIdx];
            if (!IS_STRING(nameVal)) return false;

            if (klass == NULL) return false;
            const FieldInfo *info = jaiClassFieldInfo(klass, AS_STRING(nameVal));
            if (info == NULL || info->isStatic) return false;

            unsigned base = (unsigned)offsetof(ObjInstance, fields) +
                            (unsigned)info->slot * (unsigned)sizeof(Value);
            emit(e, jaiA64MovzX(JIT_SCRATCH_A,
                                kv == SLOT_INT ? VAL_INT : VAL_FLOAT, 0));
            emit(e, jaiA64StrW(JIT_SCRATCH_A, rr, base));
            emit(e, jaiA64StrX(rv, rr, base + 8));
            recordFieldStore(e, recvLocal, info->slot, kv);
            e->wroteHeap = true;
            off += 6;
            break;
        }

        case OP_RETURN_NULL: {
            if ((fn->flags & FN_INIT) == 0) {
                /* A function with nothing to return. No register carries the
                 * answer; the entry point builds a null from the kind alone. */
                if (e->sawReturn && e->returnKind != SLOT_NULL) return false;
                e->sawReturn  = true;
                e->returnKind = SLOT_NULL;
                emit(e, jaiA64MovzX(0, 0, 0));
                emitEpilogue(e, 0);
                off += 1;
                break;
            }
            /* An initializer yields the object it initialised, which is what
             * makes `Point(1, 2)` an expression. */
            if (!localInRange(e, 0) || e->localKind[0] != SLOT_INST) return false;
            e->usesSlot0 = true;
            if (e->sawReturn && e->returnKind != SLOT_INST) return false;
            e->sawReturn  = true;
            e->returnKind = SLOT_INST;
            e->returnShape = e->localShape[0];
            {
                unsigned src = localIn(e, 0, 0);
                if (src != 0) emit(e, jaiA64MovX(0, src));
            }
            emitEpilogue(e, 0);
            off += 1;
            break;
        }

        case OP_GET_UPVALUE: {
            unsigned index = code[off + 1];
            if (index >= (unsigned)fn->upvalueCount) return false;
            if (!e->usesUpvalues) return false;   /* decided before this pass */

            /* closure->upvalues[index]->location, then the Value there. The
             * upvalue may still be open, pointing into the VM stack, so the
             * location is followed rather than assumed closed. */
            emit(e, jaiA64LdrX(JIT_SCRATCH_A, closureReg(e),
                               (unsigned)offsetof(ObjClosure, upvalues)));
            emit(e, jaiA64LdrX(JIT_SCRATCH_A, JIT_SCRATCH_A, index * 8u));
            emit(e, jaiA64LdrX(JIT_SCRATCH_A, JIT_SCRATCH_A,
                               (unsigned)offsetof(ObjUpvalue, location)));
            emit(e, jaiA64LdrW(JIT_SCRATCH_B, JIT_SCRATCH_A, 0));

            /* An upvalue's type is whatever the capture put there, so it is
             * read once here and checked on every entry into the loop. */
            Value seen = NULL_VAL;
            ObjClosure *cl = closure;
            if (index < (unsigned)cl->upvalueCount && cl->upvalues[index] != NULL) {
                seen = *cl->upvalues[index]->location;
            }
            SlotKind kind;
            unsigned tag;
            if (IS_INT(seen))        { kind = SLOT_INT;   tag = VAL_INT; }
            else if (IS_FLOAT(seen)) { kind = SLOT_FLOAT; tag = VAL_FLOAT; }
            else return false;

            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_B, tag));
            branchOnDeopt(e, JAI_A64_NE);
            if (!pushValue(e, kind, 0, NULL)) return false;
            emit(e, jaiA64LdrX(pushReg(e) - 1, JIT_SCRATCH_A, 8));
            off += 2;
            break;
        }

        case OP_GET_FIELD: {
            /* Same as OP_GET_FIELD_LOCAL, but the receiver arrives on the
             * operand stack -- which is what a compound assignment emits,
             * since it pushes the receiver twice. */
            uint32_t nameIdx = jaiReadU24(code + off + 1);
            if (e->depth == 0) return false;
            ObjClass *klass = e->stackClass[e->depth - 1];
            Value seen = e->stackSeen[e->depth - 1];
            int fromLocal = e->stackLocal[e->depth - 1];

            if (e->stack[e->depth - 1] != SLOT_INST || klass == NULL) return false;
            if (nameIdx >= (uint32_t)fn->chunk.constants.count) return false;
            Value nameVal = fn->chunk.constants.data[nameIdx];
            if (!IS_STRING(nameVal)) return false;

            const FieldInfo *info = jaiClassFieldInfo(klass, AS_STRING(nameVal));
            if (info == NULL || info->isStatic) return false;
            if (!IS_INSTANCE(seen)) return false;
            ObjInstance *inst = AS_INSTANCE(seen);
            if (info->slot >= inst->fieldCount) return false;
            Value fieldVal = inst->fields[info->slot];

            SlotKind kind;
            unsigned tag;
            if (IS_INT(fieldVal))        { kind = SLOT_INT;   tag = VAL_INT; }
            else if (IS_FLOAT(fieldVal)) { kind = SLOT_FLOAT; tag = VAL_FLOAT; }
            else return false;

            unsigned fbase = (unsigned)offsetof(ObjInstance, fields) +
                             (unsigned)info->slot * (unsigned)sizeof(Value);
            unsigned rr = JIT_FIRST_SAVED + regBase(e) +
                          (e->usesUpvalues ? 1u : 0u) + e->valueDepth - 1;
            SlotKind already = knownFieldKind(e, fromLocal, info->slot);
            if (already != SLOT_SELF) {
                kind = already;
            } else {
                /* Guard BEFORE the receiver comes off the model: a deopt here
                 * resumes at this instruction, and the interpreter's stack
                 * still has the receiver on it. */
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, rr, fbase));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, tag));
                branchOnDeopt(e, JAI_A64_NE);
            }
            unsigned popped;
            SlotKind kr;
            if (!popValue(e, &popped, &kr)) return false;
            if (!pushValue(e, kind, 0, NULL)) return false;
            emit(e, jaiA64LdrX(pushReg(e) - 1, rr, fbase + 8));
            off += 6;
            break;
        }

        case OP_BUILD_LIST: {
            unsigned n = jaiReadU16(code + off + 1);
            if (n > JIT_MAX_ARGS_OUT) return false;
            if (!e->callsOut) return false;
            if (e->depth < n) return false;
            if (!emitDescriptor(e, NULL_VAL, e->depth - n, n,
                                (void *)&jitBuildList)) {
                return false;
            }
            for (unsigned i = 0; i < n; i++) {
                unsigned r;
                if (!popValue(e, &r, NULL)) return false;
            }
            if (!pushValue(e, SLOT_LIST, 0, NULL)) return false;
            emit(e, jaiA64LdrX(pushReg(e) - 1, 31,
                               e->descOffset +
                                   (unsigned)offsetof(JitCallDesc, result) + 8));
            e->wroteHeap = true;
            off += 3;
            break;
        }

        case OP_POW: {
            /* Only `** 0.5`, which is a square root. C says pow(x, 0.5) is
             * sqrt(x) for every x >= +0, and differs only for -0.0, where pow
             * gives +0.0 and sqrt gives -0.0. So a negative sign bit -- which
             * is what tells -0.0 from +0.0 -- goes back to the interpreter. */
            if (e->depth < 2) return false;
            if (e->stack[e->depth - 1] != SLOT_FLOAT) return false;
            if (e->stack[e->depth - 2] != SLOT_FLOAT) return false;
            Value expv = e->stackSeen[e->depth - 1];
            if (!IS_FLOAT(expv) || AS_FLOAT(expv) != 0.5) {
                e->whyNot = "an exponent other than 0.5";
                return false;
            }
            unsigned rbase = pushReg(e) - 2;
            emit(e, jaiA64LsrX(JIT_SCRATCH_A, rbase, 63));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, 0));
            branchOnDeopt(e, JAI_A64_NE);

            unsigned dp1, dp2;
            if (!popValue(e, &dp1, NULL)) return false;
            if (!popValue(e, &dp2, NULL)) return false;
            if (!pushValue(e, SLOT_FLOAT, 0, NULL)) return false;
            emit(e, jaiA64FmovDX(JIT_FSCRATCH_A, rbase));
            emit(e, jaiA64FsqrtD(JIT_FSCRATCH_A, JIT_FSCRATCH_A));
            emit(e, jaiA64FmovXD(pushReg(e) - 1, JIT_FSCRATCH_A));
            off += 1;
            break;
        }

        case OP_MOD: {
            /* Floor remainder, the same rule as the fused form but with the
             * divisor in a register, so zero and -1 both have to be checked. */
            if (e->depth < 2) return false;
            if (e->stack[e->depth - 1] != SLOT_INT) return false;
            if (e->stack[e->depth - 2] != SLOT_INT) return false;
            unsigned ry = pushReg(e) - 1, rx = pushReg(e) - 2;

            emit(e, jaiA64SubsXImm(31, ry, 0));
            branchOnDeopt(e, JAI_A64_EQ);
            emit(e, jaiA64AddXImm(JIT_SCRATCH_A, ry, 1));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, 0));
            branchOnDeopt(e, JAI_A64_EQ);

            emit(e, jaiA64SdivX(JIT_SCRATCH_B, rx, ry));
            emit(e, jaiA64MsubX(JIT_SCRATCH_C, JIT_SCRATCH_B, ry, rx));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_C, 0));
            emit(e, jaiA64BCond(JAI_A64_EQ, 5));
            emit(e, jaiA64EorX(JIT_SCRATCH_D, JIT_SCRATCH_C, ry));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_D, 0));
            emit(e, jaiA64BCond(JAI_A64_GE, 2));
            emit(e, jaiA64AddX(JIT_SCRATCH_C, JIT_SCRATCH_C, ry));

            unsigned dm1, dm2;
            if (!popValue(e, &dm1, NULL)) return false;
            if (!popValue(e, &dm2, NULL)) return false;
            if (!pushValue(e, SLOT_INT, 0, NULL)) return false;
            emit(e, jaiA64MovX(pushReg(e) - 1, JIT_SCRATCH_C));
            off += 1;
            break;
        }

        case OP_MOD_INT_CONST: {
            /* `<int k>; MOD` fused: floor remainder by a constant. k cannot be
             * zero -- fusion only happens when it is known non-zero -- and -1
             * goes back to the interpreter so INT64_MIN %% -1 stays its
             * problem. */
            int16_t imm = jaiReadI16(code + off + 1);
            if (imm == 0 || imm == -1) return false;
            if (e->depth == 0 || e->stack[e->depth - 1] != SLOT_INT) return false;
            unsigned rx = pushReg(e) - 1;

            emitConst64(e, JIT_SCRATCH_A, imm);
            emit(e, jaiA64SdivX(JIT_SCRATCH_B, rx, JIT_SCRATCH_A));
            emit(e, jaiA64MsubX(JIT_SCRATCH_C, JIT_SCRATCH_B, JIT_SCRATCH_A, rx));
            /* r += k when r is nonzero and its sign differs from k's. */
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_C, 0));
            emit(e, jaiA64BCond(JAI_A64_EQ, 5));
            emit(e, jaiA64EorX(JIT_SCRATCH_D, JIT_SCRATCH_C, JIT_SCRATCH_A));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_D, 0));
            emit(e, jaiA64BCond(JAI_A64_GE, 2));
            emit(e, jaiA64AddX(JIT_SCRATCH_C, JIT_SCRATCH_C, JIT_SCRATCH_A));
            emit(e, jaiA64MovX(rx, JIT_SCRATCH_C));
            off += 3;
            break;
        }

        case OP_FLOORDIV: {
            unsigned rb, ra;
            SlotKind kb, ka;
            if (e->depth < 2) return false;
            ka = e->stack[e->depth - 2]; kb = e->stack[e->depth - 1];
            if (ka != SLOT_INT || kb != SLOT_INT) return false;
            rb = pushReg(e) - 1; ra = pushReg(e) - 2;

            /* Zero, and -1 with it: the interpreter reports the
             * division-by-zero, and INT64_MIN / -1 is the one quotient that
             * does not fit. Both are rare enough that declining -1 outright
             * costs nothing and removes the special case. */
            emit(e, jaiA64SubsXImm(31, rb, 0));
            branchOnDeopt(e, JAI_A64_EQ);
            emit(e, jaiA64AddXImm(JIT_SCRATCH_A, rb, 1));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, 0));
            branchOnDeopt(e, JAI_A64_EQ);

            unsigned d1, d2;
            if (!popValue(e, &d1, NULL)) return false;
            if (!popValue(e, &d2, NULL)) return false;
            if (!pushValue(e, SLOT_INT, 0, NULL)) return false;
            unsigned rq = pushReg(e) - 1;

            /* The quotient lands in the register the dividend was in -- the
             * push reuses it -- so the dividend is copied out first. Without
             * that, msub read a value sdiv had already overwritten and 7 // 2
             * came out 2. */
            emit(e, jaiA64MovX(JIT_SCRATCH_C, ra));

            /* Truncating quotient, then one down when the remainder is nonzero
             * and its sign differs from the divisor's -- which is what makes
             * this floor division rather than C's. */
            emit(e, jaiA64SdivX(rq, JIT_SCRATCH_C, rb));
            emit(e, jaiA64MsubX(JIT_SCRATCH_A, rq, rb, JIT_SCRATCH_C));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, 0));
            emit(e, jaiA64BCond(JAI_A64_EQ, 5));
            emit(e, jaiA64EorX(JIT_SCRATCH_B, JIT_SCRATCH_A, rb));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_B, 0));
            emit(e, jaiA64BCond(JAI_A64_GE, 2));
            emit(e, jaiA64SubXImm(rq, rq, 1));
            off += 1;
            break;
        }

        case OP_INVOKE: {
            uint32_t nameIdx = jaiReadU24(code + off + 1);
            unsigned argc    = code[off + 4];
            if (argc > JIT_MAX_ARGS_OUT - 1) return false;
            if (!e->callsOut) return false;
            if (e->depth < argc + 1) return false;

            unsigned ridx = e->depth - argc - 1;
            SlotKind rk = e->stack[ridx];

            if (rk == SLOT_LIST && argc == 1 &&
                nameIdx < (uint32_t)fn->chunk.constants.count &&
                IS_STRING(fn->chunk.constants.data[nameIdx]) &&
                strcmp(AS_STRING(fn->chunk.constants.data[nameIdx])->chars,
                       "push") == 0) {
                /* Appending to a list is a bounds check and two stores, and
                 * going through a descriptor and a native for it costs far
                 * more than the work. `list_ops` pushes a million elements and
                 * spent all of it on the call. Growing is left to the
                 * interpreter: a full list deopts to this instruction, which
                 * has written nothing yet. */
                SlotKind vk = e->stack[e->depth - 1];
                unsigned vtag = vk == SLOT_INT   ? VAL_INT
                              : vk == SLOT_FLOAT ? VAL_FLOAT
                              : vk == SLOT_BOOL  ? VAL_BOOL
                              : (vk == SLOT_INST || vk == SLOT_LIST ||
                                 vk == SLOT_OBJ)  ? VAL_OBJ
                                                  : 0xffffffffu;
                if (vtag == 0xffffffffu) {
                    e->whyNot = "pushing a kind the tier cannot store";
                    return false;
                }
                unsigned rVal  = pushReg(e) - 1;
                unsigned rList = pushReg(e) - 2;

                emit(e, jaiA64LdrW(JIT_SCRATCH_A, rList,
                                   (unsigned)offsetof(ObjList, count)));
                emit(e, jaiA64LdrW(JIT_SCRATCH_B, rList,
                                   (unsigned)offsetof(ObjList, capacity)));
                emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
                branchOnDeopt(e, JAI_A64_GE);

                emit(e, jaiA64LdrX(JIT_SCRATCH_C, rList,
                                   (unsigned)offsetof(ObjList, items)));
                emit(e, jaiA64LslX(JIT_SCRATCH_D, JIT_SCRATCH_A, 4));
                emit(e, jaiA64AddX(JIT_SCRATCH_C, JIT_SCRATCH_C,
                                   JIT_SCRATCH_D));
                emit(e, jaiA64MovzX(JIT_SCRATCH_D, vtag, 0));
                emit(e, jaiA64StrW(JIT_SCRATCH_D, JIT_SCRATCH_C, 0));
                emit(e, jaiA64StrX(rVal, JIT_SCRATCH_C, 8));
                emit(e, jaiA64AddXImm(JIT_SCRATCH_A, JIT_SCRATCH_A, 1));
                emit(e, jaiA64StrW(JIT_SCRATCH_A, rList,
                                   (unsigned)offsetof(ObjList, count)));
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, rList,
                                   (unsigned)offsetof(ObjList, version)));
                emit(e, jaiA64AddXImm(JIT_SCRATCH_A, JIT_SCRATCH_A, 1));
                emit(e, jaiA64StrW(JIT_SCRATCH_A, rList,
                                   (unsigned)offsetof(ObjList, version)));
                e->wroteHeap = true;

                /* push returns the list, which is the receiver entry already
                 * sitting under the argument. */
                unsigned drop;
                if (!popValue(e, &drop, NULL)) return false;
                off += 7;
                break;
            }

            if (rk == SLOT_INST) {
                /* A method whose whole body is one arithmetic expression over
                 * its receiver and arguments is worth putting inline: the call
                 * around it costs more than the expression does. Vec2.dot is
                 * four field reads, two multiplies and an add, reached through
                 * a descriptor, a helper and a compiled entry. */
                if (inlineMethod(e, closure, nameIdx, argc, off)) {
                    off += 7;
                    break;
                }

                /* A method on an instance. The class is fixed here, so the
                 * method is resolved now; what it returns is not knowable, so
                 * the tag that comes back is checked and a surprise deopts to
                 * the instruction AFTER the call -- the call has happened and
                 * must not happen twice. */
                ObjClass *rcls = e->stackClass[ridx];
                if (rcls == NULL) return false;
                if (nameIdx >= (uint32_t)fn->chunk.constants.count) return false;
                Value mname = fn->chunk.constants.data[nameIdx];
                if (!IS_STRING(mname)) return false;
                Value method;
                if (!jaiClassFindMethod(rcls, AS_STRING(mname), &method)) {
                    return false;
                }
                if (!IS_CLOSURE(method)) return false;
                ObjFunction *mfn = AS_CLOSURE(method)->fn;
                if (mfn->jitFunc == NULL) return false;   /* kind unknown yet */
                SlotKind rkind = (SlotKind)mfn->jitReturnKind;
                uint32_t rshape = mfn->jitReturnShape;
                ObjClass *rrcls = NULL;
                if (rkind == SLOT_INST) {
                    if (rshape == 0) return false;
                    if (!jaiClassForShape(rshape, &rrcls) || rrcls == NULL) {
                        return false;
                    }
                }
                if (rkind != SLOT_INT && rkind != SLOT_FLOAT &&
                    rkind != SLOT_BOOL && rkind != SLOT_INST &&
                    rkind != SLOT_LIST) {
                    return false;
                }

                if (!emitDescriptor(e, method, ridx, argc + 1,
                                    (void *)&jitInvokeMethod)) {
                    return false;
                }
                for (unsigned i = 0; i <= argc; i++) {
                    unsigned r;
                    if (!popValue(e, &r, NULL)) return false;
                }
                if (!pushValue(e, rkind, rshape, rrcls)) return false;

                unsigned rat = e->descOffset +
                               (unsigned)offsetof(JitCallDesc, result);
                unsigned wantTag = rkind == SLOT_INT   ? VAL_INT
                                 : rkind == SLOT_FLOAT ? VAL_FLOAT
                                 : rkind == SLOT_BOOL  ? VAL_BOOL
                                                       : VAL_OBJ;
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, 31, rat));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, wantTag));
                branchOnDeoptAt(e, JAI_A64_NE, (uint32_t)(off + 7), true);
                emit(e, jaiA64LdrX(pushReg(e) - 1, 31, rat + 8));
                if (rkind == SLOT_INST) {
                    /* "an object" is not "an object of this class", and a
                     * method entered with another specialisation runs
                     * interpreted and may return either. */
                    emit(e, jaiA64LdrX(JIT_SCRATCH_A, pushReg(e) - 1,
                                       (unsigned)offsetof(ObjInstance, klass)));
                    emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_A,
                                       (unsigned)offsetof(ObjClass, shapeId)));
                    emitConst64(e, JIT_SCRATCH_B, (int64_t)rshape);
                    emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
                    branchOnDeoptAt(e, JAI_A64_NE, (uint32_t)(off + 7), true);
                }

                e->wroteHeap = true;
                off += 7;
                break;
            }

            if (rk != SLOT_LIST) return false;
            if (nameIdx >= (uint32_t)fn->chunk.constants.count) return false;
            Value nameVal = fn->chunk.constants.data[nameIdx];
            if (!IS_STRING(nameVal)) return false;

            /* Which method a name means depends on the receiver's type, not on
             * what it holds -- so when the receiver is a list this body built
             * and there is no sample to look at, an empty one answers just as
             * well. Rooted across the lookup because resolving allocates the
             * bound wrapper; only the native is kept, and that outlives it. */
            Value probe = e->stackSeen[ridx];
            bool madeProbe = false;
            if (!IS_LIST(probe)) {
                ObjList *tmp = jaiListNew(0);
                probe = OBJ_VAL(tmp);
                jaiGCPushRoot(probe);
                madeProbe = true;
            }
            Value bound;
            bool found = jaiBuiltinMethod(probe, AS_STRING(nameVal), &bound);
            if (madeProbe) jaiGCPopRoot();
            if (!found) return false;
            Value nativeVal = IS_BOUND(bound) ? AS_BOUND(bound)->method : bound;
            if (!IS_NATIVE(nativeVal)) return false;

            /* What comes back. `len` is an int; anything whose result is
             * dropped on the next instruction needs no kind at all. Nothing
             * else, because a call's result cannot be guarded -- the call
             * already happened, so a wrong guess has nowhere to go. */
            const char *mname = AS_STRING(nameVal)->chars;
            bool discarded = (off + 7 < count && code[off + 7] == OP_POP);
            bool isLen = strcmp(mname, "len") == 0 && argc == 0;
            if (!isLen && !discarded) return false;

            if (!emitDescriptor(e, nativeVal, ridx, argc + 1,
                                (void *)&jitInvokeNative)) {
                return false;
            }

            for (unsigned i = 0; i <= argc; i++) {
                unsigned r;
                if (!popValue(e, &r, NULL)) return false;
            }
            if (isLen) {
                if (!pushValue(e, SLOT_INT, 0, NULL)) return false;
                emit(e, jaiA64LdrX(pushReg(e) - 1, 31,
                                   e->descOffset +
                                       (unsigned)offsetof(JitCallDesc, result) + 8));
            }
            e->wroteHeap = true;
            off += 7;
            if (!isLen) off += 1;      /* the OP_POP this consumed */
            break;
        }

        case OP_BUILD_RANGE: {
            /* Deferred: the range is only worth building alongside its
             * iterator, which the next instruction asks for. */
            if (e->depth < 2) return false;
            if (e->stack[e->depth - 1] != SLOT_INT) return false;
            if (e->stack[e->depth - 2] != SLOT_INT) return false;
            /* OP_BUILD_RANGE carries one operand byte, so the next opcode is
             * two along. */
            if (off + 2 >= count || code[off + 2] != OP_GET_ITER) {
                e->whyNot = "a range that is not immediately iterated";
                return false;
            }
            e->rangeInclusive = code[off + 1] != 0;
            e->pendingRange = true;
            off += 2;
            break;
        }

        case OP_GET_ITER: {
            if (!e->pendingRange) {
                /* Not a range: iterate whatever it is, through the runtime. */
                if (e->depth == 0) return false;
                if (e->stack[e->depth - 1] != SLOT_LIST) {
                    e->whyNot = "iterating something other than a list or range";
                    return false;
                }
                if (!e->callsOut) return false;
                /* Carry one element forward: the loop variable's kind comes
                 * from it, and the iterator itself says nothing about what it
                 * will yield. */
                Value srcv = e->stackSeen[e->depth - 1];
                Value sample = NULL_VAL;
                if (IS_LIST(srcv) && AS_LIST(srcv)->count > 0) {
                    sample = AS_LIST(srcv)->items[0];
                }
                if (IS_NULL(sample)) {
                    e->whyNot = "iterating a list with nothing to look at";
                    return false;
                }
                if (!emitDescriptor(e, NULL_VAL, e->depth - 1, 1,
                                    (void *)&jitMakeIter)) {
                    return false;
                }
                unsigned rdrop;
                if (!popValue(e, &rdrop, NULL)) return false;
                /* Shape 1 marks an iterator the runtime has to step; a
                 * range is 0 and gets the inline path. It rides on the stack
                 * entry rather than on the Emit because a function can build
                 * both -- nbody's `advance` runs two range loops and then a
                 * list loop, and one flag for the whole compile made the path
                 * a FOR_ITER_BIND took depend on what came before it. */
                if (!pushValue3(e, SLOT_ITER, 1, NULL, sample, -1)) return false;
                emit(e, jaiA64LdrX(pushReg(e) - 1, 31,
                                   e->descOffset +
                                       (unsigned)offsetof(JitCallDesc, result) + 8));
                e->wroteHeap = true;
                off += 1;
                break;
            }
            e->pendingRange = 0;
            if (!e->callsOut) return false;
            /* start, stop and the inclusive flag go in as arguments. */
            if (!pushValue(e, SLOT_INT, 0, NULL)) return false;
            emitConst64(e, pushReg(e) - 1, e->rangeInclusive ? 1 : 0);
            if (!emitDescriptor(e, NULL_VAL, e->depth - 3, 3,
                                (void *)&jitMakeRangeIter)) {
                return false;
            }
            for (unsigned i = 0; i < 3; i++) {
                unsigned r;
                if (!popValue(e, &r, NULL)) return false;
            }
            if (!pushValue(e, SLOT_ITER, 0, NULL)) return false;
            emit(e, jaiA64LdrX(pushReg(e) - 1, 31,
                               e->descOffset +
                                   (unsigned)offsetof(JitCallDesc, result) + 8));
            e->wroteHeap = true;
            off += 1;
            break;
        }

        case OP_FOR_ITER_BIND: {
            /* A loop this body built the iterator for: the index lives in the
             * iterator, so every iteration loads and stores it, and a deopt
             * needs nothing -- what is on the stack is already current. */
            if (!e->osr || !e->hasIter) {
                if (e->depth == 0 || e->stack[e->depth - 1] != SLOT_ITER) {
                    return false;
                }
                int16_t  fjump = jaiReadI16(code + off + 1);
                unsigned fslot = jaiReadU16(code + off + 3);
                if (!localInRange(e, fslot)) return false;
                unsigned rIt = pushReg(e) - 1;

                if (e->stackShape[e->depth - 1] != 0) {
                    /* A list iterator: ask the runtime for each element and
                     * take the kind from the one it is holding now. */
                    Value sample = e->stackSeen[e->depth - 1];
                    SlotKind ek; uint32_t esh = 0; ObjClass *ecl = NULL;
                    if (IS_INT(sample))        ek = SLOT_INT;
                    else if (IS_FLOAT(sample)) ek = SLOT_FLOAT;
                    else if (IS_INSTANCE(sample) && AS_INSTANCE(sample)->klass) {
                        ek = SLOT_INST;
                        ecl = AS_INSTANCE(sample)->klass;
                        esh = ecl->shapeId;
                    } else { e->whyNot = "element kind unknown"; return false; }

                    if (!emitDescriptorStatus(e, NULL_VAL, e->depth - 1, 1,
                                              (void *)&jitIterStep, true,
                                              -1)) {
                        return false;
                    }
                    /* 1 is exhausted, anything higher is a raise. */
                    emit(e, jaiA64SubsXImm(31, 0, 1));
                    branchToDepth(e, (uint32_t)((int32_t)(off + 5) + fjump),
                                  JAI_A64_EQ,
                                  (int)stackSignatureAt(e, e->depth - 1));
                    emit(e, jaiA64SubsXImm(31, 0, 1));
                    if (e->fixupCount >= JIT_MAX_FIXUPS) { e->failed = true; return false; }
                    e->fixups[e->fixupCount].instIndex    = (int)e->count;
                    e->fixups[e->fixupCount].targetOffset = FIXUP_THREW;
                    e->fixups[e->fixupCount].conditional  = true;
                    e->fixups[e->fixupCount].depth        = -1;
                    e->fixupCount++;
                    emit(e, jaiA64BCond(JAI_A64_GT, 0));

                    if (!adoptLocalKindSeen(e, fslot, ek, esh, ecl, sample)) {
                        e->whyNot = "loop variable took two kinds";
                        return false;
                    }
                    emit(e, jaiA64LdrX(JIT_SCRATCH_A, 31,
                                       e->descOffset +
                                           (unsigned)offsetof(JitCallDesc, result) + 8));
                    localOut(e, fslot, JIT_SCRATCH_A);
                    e->wroteHeap = true;
                    off += 5;
                    break;
                }

                emit(e, jaiA64LdrX(JIT_SCRATCH_A, rIt,
                                   (unsigned)offsetof(ObjIter, index)));
                emit(e, jaiA64LdrX(JIT_SCRATCH_B, rIt,
                                   (unsigned)offsetof(ObjIter, limit)));
                if (!adoptLocalKind(e, fslot, SLOT_INT, 0, NULL)) return false;
                emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
                /* The exhausted arm drops the iterator, so the target is
                 * reached one entry shallower than this branch leaves from. */
                branchToDepth(e, (uint32_t)((int32_t)(off + 5) + fjump),
                              JAI_A64_GE,
                              (int)stackSignatureAt(e, e->depth - 1));
                /* A range yields `start + index * step`, not the index --
                 * see jaiIterNext's ITER_RANGE case. The index is always
                 * zero-based, so using it as the value is only right for
                 * `0..n` in unit steps. `for j in i + 1..n` counted from zero
                 * instead of from i+1, which is a plausible wrong answer
                 * rather than a crash: nested loops summed the wrong pairs.
                 * The limit register is dead after the compare, so it carries
                 * the index across to the increment. */
                emit(e, jaiA64MovX(JIT_SCRATCH_B, JIT_SCRATCH_A));
                emit(e, jaiA64LdrX(JIT_SCRATCH_C, rIt,
                                   (unsigned)offsetof(ObjIter, source) + 8));
                emit(e, jaiA64LdrX(JIT_SCRATCH_D, JIT_SCRATCH_C,
                                   (unsigned)offsetof(ObjRange, step)));
                emit(e, jaiA64MulX(JIT_SCRATCH_A, JIT_SCRATCH_A,
                                   JIT_SCRATCH_D));
                emit(e, jaiA64LdrX(JIT_SCRATCH_D, JIT_SCRATCH_C,
                                   (unsigned)offsetof(ObjRange, start)));
                emit(e, jaiA64AddX(JIT_SCRATCH_A, JIT_SCRATCH_D,
                                   JIT_SCRATCH_A));
                localOut(e, fslot, JIT_SCRATCH_A);
                emit(e, jaiA64AddXImm(JIT_SCRATCH_B, JIT_SCRATCH_B, 1));
                emit(e, jaiA64StrX(JIT_SCRATCH_B, rIt,
                                   (unsigned)offsetof(ObjIter, index)));
                off += 5;
                break;
            }
            /* Only as the head of the loop being compiled, and only for a
             * range from zero in unit steps -- that is what makes the yielded
             * value the index itself. Everything else about the iterator is a
             * runtime fact, checked at the entry. */
            if (!e->osr || !e->hasIter) return false;
            if ((uint32_t)off != e->osrTop) return false;
            int16_t  jump = jaiReadI16(code + off + 1);
            unsigned slot = jaiReadU16(code + off + 3);
            if (!localInRange(e, slot)) return false;
            if (!adoptLocalKind(e, slot, SLOT_INT, 0, NULL)) return false;
            e->iterSlot = slot;
            e->iterExit = (uint32_t)((int32_t)(off + 5) + jump);

            emit(e, jaiA64SubsXReg(31, JIT_IDX_REG, JIT_LIM_REG));
            branchTo(e, e->iterExit, true, JAI_A64_GE);
            /* The value is start + index, not the index: `for j in i + 1..n`
             * is the inner loop of nbody's advance. */
            emit(e, jaiA64AddX(JIT_SCRATCH_C, JIT_START_REG, JIT_IDX_REG));
            localOut(e, slot, JIT_SCRATCH_C);
            emit(e, jaiA64AddXImm(JIT_IDX_REG, JIT_IDX_REG, 1));
            off += 5;
            break;
        }

        case OP_GET_INDEX: {
            /* list[i], with the index normalised the way jaiNormalizeIndex
             * does it and one unsigned compare covering both ends. Anything
             * out of range, or an element that is not the kind seen at compile
             * time, goes back to the interpreter -- which raises the IndexError
             * or re-reads the element, whichever it is. Reading an element has
             * no effect, so resuming at this instruction is sound. */
            if (e->depth < 2) return false;
            if (e->stack[e->depth - 2] != SLOT_LIST) return false;
            if (e->stack[e->depth - 1] != SLOT_INT) return false;
            unsigned rIdx = pushReg(e) - 1, rList = pushReg(e) - 2;

            Value seenList = e->stackSeen[e->depth - 2];
            if (!IS_LIST(seenList)) return false;
            ObjList *sl = AS_LIST(seenList);
            if (sl->count <= 0) return false;
            Value elem = sl->items[0];
            SlotKind kind;
            unsigned tag;
            ObjClass *elemClass = NULL;
            uint32_t  elemShape = 0;
            if (IS_INT(elem))        { kind = SLOT_INT;   tag = VAL_INT; }
            else if (IS_FLOAT(elem)) { kind = SLOT_FLOAT; tag = VAL_FLOAT; }
            else if (IS_LIST(elem)) {
                /* A list of lists. `matrix_mul` is `b[k][j]` in its innermost
                 * loop and could not compile the outer half of it. */
                kind = SLOT_LIST;
                tag = VAL_OBJ;
            }
            else if (IS_INSTANCE(elem) && AS_INSTANCE(elem)->klass != NULL) {
                /* A list of instances, all of one shape -- which the per-read
                 * tag check cannot confirm on its own, so the class is checked
                 * too. A list holding two shapes deoptimises on the second. */
                kind = SLOT_INST;
                tag = VAL_OBJ;
                elemClass = AS_INSTANCE(elem)->klass;
                elemShape = elemClass->shapeId;
            } else return false;

            emit(e, jaiA64LdrW(JIT_SCRATCH_A, rList,
                               (unsigned)offsetof(ObjList, count)));
            emit(e, jaiA64MovX(JIT_SCRATCH_B, rIdx));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_B, 0));
            emit(e, jaiA64BCond(JAI_A64_GE, 2));
            emit(e, jaiA64AddX(JIT_SCRATCH_B, JIT_SCRATCH_B, JIT_SCRATCH_A));
            emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_B, JIT_SCRATCH_A));
            branchOnDeopt(e, JAI_A64_HS);

            emit(e, jaiA64LdrX(JIT_SCRATCH_C, rList,
                               (unsigned)offsetof(ObjList, items)));
            emit(e, jaiA64LslX(JIT_SCRATCH_D, JIT_SCRATCH_B, 4));
            emit(e, jaiA64AddX(JIT_SCRATCH_C, JIT_SCRATCH_C, JIT_SCRATCH_D));
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_C, 0));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, tag));
            branchOnDeopt(e, JAI_A64_NE);
            if (kind == SLOT_INST) {
                /* The tag says "an object", not "an object of this class". */
                emit(e, jaiA64LdrX(JIT_SCRATCH_D, JIT_SCRATCH_C, 8));
                emit(e, jaiA64LdrX(JIT_SCRATCH_D, JIT_SCRATCH_D,
                                   (unsigned)offsetof(ObjInstance, klass)));
                emit(e, jaiA64LdrW(JIT_SCRATCH_D, JIT_SCRATCH_D,
                                   (unsigned)offsetof(ObjClass, shapeId)));
                emitConst64(e, JIT_SCRATCH_A, (int64_t)elemShape);
                emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_D, JIT_SCRATCH_A));
                branchOnDeopt(e, JAI_A64_NE);
            }

            unsigned d1, d2;
            if (!popValue(e, &d1, NULL)) return false;
            if (!popValue(e, &d2, NULL)) return false;
            if (!pushValue3(e, kind, elemShape, elemClass, elem, -1)) return false;
            emit(e, jaiA64LdrX(pushReg(e) - 1, JIT_SCRATCH_C, 8));
            off += 1;
            break;
        }

        case OP_SET_INDEX: {
            /* list[i] = v, the write half of OP_GET_INDEX and normalised the
             * same way. Every guard runs before the store, so a deopt here
             * still resumes at an instruction that has not happened yet.
             * Sixteen refusals across the benchmarks came from its absence --
             * `queens` could not compile the function that does the work. */
            if (e->depth < 3) return false;
            if (e->stack[e->depth - 3] != SLOT_LIST) return false;
            if (e->stack[e->depth - 2] != SLOT_INT) return false;
            SlotKind vk = e->stack[e->depth - 1];
            unsigned vtag = vk == SLOT_INT   ? VAL_INT
                          : vk == SLOT_FLOAT ? VAL_FLOAT
                          : vk == SLOT_BOOL  ? VAL_BOOL
                          : (vk == SLOT_INST || vk == SLOT_LIST ||
                             vk == SLOT_OBJ)  ? VAL_OBJ
                                              : 0xffffffffu;
            if (vtag == 0xffffffffu) return false;
            unsigned rVal = pushReg(e) - 1;
            unsigned rIdx = pushReg(e) - 2;
            unsigned rList = pushReg(e) - 3;

            emit(e, jaiA64LdrW(JIT_SCRATCH_A, rList,
                               (unsigned)offsetof(ObjList, count)));
            emit(e, jaiA64MovX(JIT_SCRATCH_B, rIdx));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_B, 0));
            emit(e, jaiA64BCond(JAI_A64_GE, 2));
            emit(e, jaiA64AddX(JIT_SCRATCH_B, JIT_SCRATCH_B, JIT_SCRATCH_A));
            emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_B, JIT_SCRATCH_A));
            branchOnDeopt(e, JAI_A64_HS);

            emit(e, jaiA64LdrX(JIT_SCRATCH_C, rList,
                               (unsigned)offsetof(ObjList, items)));
            emit(e, jaiA64LslX(JIT_SCRATCH_D, JIT_SCRATCH_B, 4));
            emit(e, jaiA64AddX(JIT_SCRATCH_C, JIT_SCRATCH_C, JIT_SCRATCH_D));
            emit(e, jaiA64MovzX(JIT_SCRATCH_A, vtag, 0));
            emit(e, jaiA64StrW(JIT_SCRATCH_A, JIT_SCRATCH_C, 0));
            emit(e, jaiA64StrX(rVal, JIT_SCRATCH_C, 8));
            /* jaiListTouch: the count has not changed, so only the version
             * tells an iterator that the list moved under it. */
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, rList,
                               (unsigned)offsetof(ObjList, version)));
            emit(e, jaiA64AddXImm(JIT_SCRATCH_A, JIT_SCRATCH_A, 1));
            emit(e, jaiA64StrW(JIT_SCRATCH_A, rList,
                               (unsigned)offsetof(ObjList, version)));
            e->wroteHeap = true;

            unsigned d1, d2, d3;
            if (!popValue(e, &d1, NULL)) return false;
            if (!popValue(e, &d2, NULL)) return false;
            if (!popValue(e, &d3, NULL)) return false;
            off += 1;
            break;
        }

        case OP_GET_GLOBAL: {
            uint32_t nameIdx = jaiReadU24(code + off + 1);
            if (globalIsSelf(closure, nameIdx)) {
                if (!pushSelf(e)) return false;
                off += 6;
                break;
            }
            /* A class, resolved now and pinned by the module version check at
             * entry: rebinding the name retires the compiled form. It occupies
             * no register -- it is baked into the call sequence. */
            ObjClass *cls = globalClass(closure, nameIdx);
            if (cls != NULL) {
                if (e->depth >= JIT_MAX_SAVED) return false;
                e->stackShape[e->depth] = cls->shapeId;
                e->stackClass[e->depth] = cls;
                e->stackSeen[e->depth]  = NULL_VAL;
                e->stackLocal[e->depth] = -1;
                e->stack[e->depth++]    = SLOT_CLASS;
                off += 6;
                break;
            }
            /* A plain function, resolved now and pinned by the same
             * module-version check. Only one that has itself compiled, since
             * its return kind is the only way to type the result. */
            Value gv;
            ObjFunction *gfn = globalFunction(closure, nameIdx, &gv);
            if (gfn == NULL || gfn->jitFunc == NULL) {
                Value nv;
                ObjNative *nat = globalNative(closure, nameIdx, &nv);
                if (nat == NULL) {
                    e->whyNot = "callee is not a compiled global function";
                    return false;
                }
                if (e->depth >= JIT_MAX_SAVED) return false;
                e->stackShape[e->depth] = 0;
                e->stackClass[e->depth] = (ObjClass *)(void *)AS_OBJ(nv);
                e->stackSeen[e->depth]  = nv;
                e->stackLocal[e->depth] = -1;
                e->stack[e->depth++]    = SLOT_NATIVE;
                off += 6;
                break;
            }
            if (e->depth >= JIT_MAX_SAVED) return false;
            e->stackShape[e->depth] = 0;
            /* The stub that writes a deopt record materialises a callee from
             * here, so it has to be the object and not NULL. */
            e->stackClass[e->depth] = (ObjClass *)(void *)AS_OBJ(gv);
            e->stackSeen[e->depth]  = gv;
            e->stackLocal[e->depth] = -1;
            e->stack[e->depth++]    = SLOT_FUNC;
            off += 6;
            break;
        }

        case OP_CALL: {
            unsigned argc = code[off + 1];

            if (isClassCallee(e, argc)) {
                if (!emitCallOut(e, argc)) return false;
                off += 2;
                break;
            }
            if (e->depth >= argc + 1u &&
                e->stack[e->depth - argc - 1] == SLOT_FUNC) {
                if (!emitGlobalCall(e, argc, (uint32_t)(off + 2))) return false;
                off += 2;
                break;
            }
            if (e->depth >= argc + 1u &&
                e->stack[e->depth - argc - 1] == SLOT_NATIVE) {
                /* `float(i)` and `int(x)` are one instruction each, so they are
                 * emitted rather than called. They are what `spectral` and
                 * `matrix_mul` have in their inner loops, and the whole body
                 * was declined for want of them. Every other builtin still
                 * declines: a call needs a result kind, and only these have one
                 * that is known without running anything. */
                if (argc != 1) { e->whyNot = "builtin arity"; return false; }
                Value cv = e->stackSeen[e->depth - 2];
                ObjNative *nat = IS_NATIVE(cv) ? AS_NATIVE(cv) : NULL;
                const char *nm = nat != NULL && nat->name != NULL
                                     ? nat->name->chars : "";
                SlotKind ak = e->stack[e->depth - 1];
                unsigned ar = pushReg(e) - 1;
                bool toFloat = strcmp(nm, "float") == 0;
                bool toInt   = strcmp(nm, "int") == 0;
                if (toFloat && ak == SLOT_INT) {
                    emit(e, jaiA64ScvtfDX(JIT_FSCRATCH_A, ar));
                    emit(e, jaiA64FmovXD(ar, JIT_FSCRATCH_A));
                } else if (toInt && ak == SLOT_FLOAT) {
                    emit(e, jaiA64FmovDX(JIT_FSCRATCH_A, ar));
                    emit(e, jaiA64FcvtzsXD(ar, JIT_FSCRATCH_A));
                } else if (!((toFloat && ak == SLOT_FLOAT) ||
                             (toInt && ak == SLOT_INT))) {
                    e->whyNot = "a builtin with no known result kind";
                    return false;
                }
                /* The result stays in the argument's register. Dropping the
                 * callee entry, which holds none, leaves it on top. */
                e->stack[e->depth - 2]      = toFloat ? SLOT_FLOAT : SLOT_INT;
                e->stackShape[e->depth - 2] = 0;
                e->stackClass[e->depth - 2] = NULL;
                e->stackSeen[e->depth - 2]  = NULL_VAL;
                e->stackLocal[e->depth - 2] = -1;
                e->depth--;
                off += 2;
                break;
            }

            if (e->depth >= argc + 1u &&
                e->stack[e->depth - argc - 1] == SLOT_OBJ) {
                /* A closure held in a local -- `apply_n(f, ..)` doing
                 * `acc = f(acc)`, which is the shape most library code takes.
                 *
                 * The guard is on the closure's FUNCTION, not on the closure.
                 * `closure_calls` builds a fresh closure over a different
                 * `step` every outer iteration, so guarding the closure itself
                 * would deoptimise twenty times; all twenty share one
                 * ObjFunction and differ only in an upvalue, so the function
                 * pointer is monomorphic. The callee Value still comes from the
                 * register, which is what keeps the upvalues right. */
                unsigned cidx = e->depth - argc - 1;
                Value cv = e->stackSeen[cidx];
                if (!IS_CLOSURE(cv)) {
                    e->whyNot = "an indirect callee that is not a closure";
                    return false;
                }
                ObjFunction *cfn = AS_CLOSURE(cv)->fn;
                if (cfn->jitFunc == NULL || cfn->arity != argc) {
                    e->whyNot = "the closure this calls has not compiled";
                    return false;
                }
                SlotKind rkind = (SlotKind)cfn->jitReturnKind;
                if (rkind != SLOT_INT && rkind != SLOT_FLOAT &&
                    rkind != SLOT_BOOL) {
                    e->whyNot = "an indirect call whose result kind is not scalar";
                    return false;
                }
                unsigned rCallee = JIT_FIRST_SAVED + regBase(e) +
                                   (e->usesUpvalues ? 1u : 0u) +
                                   (cidx - (e->depth - e->valueDepth));

                emit(e, jaiA64LdrX(JIT_SCRATCH_A, rCallee,
                                   (unsigned)offsetof(ObjClosure, fn)));
                emitConst64(e, JIT_SCRATCH_B, (int64_t)(uintptr_t)cfn);
                emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
                branchOnDeopt(e, JAI_A64_NE);

                if (!emitDescriptorStatus(e, NULL_VAL, cidx + 1, argc,
                                          (void *)&jitCallOut, false,
                                          (int)rCallee)) {
                    return false;
                }
                for (unsigned i = 0; i <= argc; i++) {
                    unsigned r;
                    if (!popValue(e, &r, NULL)) return false;
                }
                if (!pushValue(e, rkind, 0, NULL)) return false;

                unsigned rat = e->descOffset +
                               (unsigned)offsetof(JitCallDesc, result);
                unsigned wantTag = rkind == SLOT_INT   ? VAL_INT
                                 : rkind == SLOT_FLOAT ? VAL_FLOAT
                                                       : VAL_BOOL;
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, 31, rat));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, wantTag));
                branchOnDeoptAt(e, JAI_A64_NE, (uint32_t)(off + 2), true);
                emit(e, jaiA64LdrX(pushReg(e) - 1, 31, rat + 8));
                e->wroteHeap = true;
                off += 2;
                break;
            }

            if (argc != e->arity) return false;
            /* Recorded, not rejected here: the measuring pass always runs
             * with slot 0 available, so testing the base in this pass aborted
             * every recursive function and quietly cost fib_recursive its
             * compiled form -- 8.8ms back to 83ms. The decision belongs after
             * the base is chosen. */
            e->hasSelfCall = true;
            if (e->depth < argc + 1) return false;
            if (e->stack[e->depth - argc - 1] != SLOT_SELF) return false;

            /* The arguments sit in the top `argc` value registers, in order.
             * They move to x0.. which nothing else is using. */
            unsigned first = JIT_FIRST_SAVED + regBase(e) +
                             (e->usesUpvalues ? 1u : 0u) + e->valueDepth - argc;
            for (unsigned i = 0; i < argc; i++) {
                emit(e, jaiA64MovX(i, first + i));
            }

            /* To instruction 0, the prologue -- NOT to the first instruction
             * of the body. A recursive call that skipped the prologue would
             * not save x19 upward, so the callee would overwrite the caller's
             * locals and the recursion would never terminate. */
            if (!e->sawReturn) e->assumedIntReturn = true;

            branchTo(e, FIXUP_ENTRY, false, 0);
            e->code[e->count - 1] = jaiA64Bl(0);
            /* x1 carries the callee's verdict; a bail there is a bail here. */
            emit(e, jaiA64SubsXImm(31, 1, 0));
            branchOnCondition(e, JAI_A64_NE);

            e->depth      -= argc + 1;
            e->valueDepth -= argc;
            if (!pushValue(e, e->returnKind, e->returnShape, NULL)) return false;
            emit(e, jaiA64MovX(pushReg(e) - 1, 0));
            off += 2;
            break;
        }

        case OP_TAIL_CALL: {
            /* `return C(...)` compiles to this. The call is made exactly as
             * OP_CALL makes it and its result is returned. */
            unsigned argc = code[off + 1];
            if (isClassCallee(e, argc)) {
                if (!emitCallOut(e, argc)) return false;
            } else if (e->depth >= argc + 1u &&
                       e->stack[e->depth - argc - 1] == SLOT_FUNC) {
                if (!emitGlobalCall(e, argc, (uint32_t)(off + 2))) return false;
            } else {
                e->whyNot = "tail callee is neither a class nor a compiled function";
                return false;
            }
            /* Which class came back, not just that an object did: a caller
             * binding this result to a local cannot compile without it. */
            uint32_t tshape = e->depth > 0 ? e->stackShape[e->depth - 1] : 0;
            unsigned r;
            SlotKind k;
            if (!popValue(e, &r, &k)) return false;
            if (e->sawReturn && e->returnKind != k) return false;
            e->sawReturn  = true;
            e->returnKind = k;
            e->returnShape = tshape;
            emit(e, jaiA64MovX(0, r));
            emitEpilogue(e, 0);
            off += 2;
            /* The compiler emits OP_RETURN after a tail call and the
             * interpreter never reaches it, because the tail call returned.
             * Walking into it would try to pop from an empty stack. */
            if (off < count && code[off] == OP_RETURN) off += 1;
            break;
        }

        case OP_RETURN: {
            uint32_t rsh = e->depth > 0 ? e->stackShape[e->depth - 1] : 0;
            unsigned r;
            SlotKind k;
            if (!popValue(e, &r, &k)) return false;
            /* One return kind per function: the entry point rebuilds a Value
             * from it, and it cannot rebuild two. */
            if (e->sawReturn && e->returnKind != k) return false;
            e->sawReturn = true;
            e->returnKind = k;
            e->returnShape = rsh;
            emit(e, jaiA64MovX(0, r));
            emitEpilogue(e, 0);
            off += 1;
            break;
        }

        default:
            if (getenv("JAI_JIT_WHY")) {
                fprintf(stderr, "[jit] %s declined at %s\n",
                        fn->name ? fn->name->chars : "<anon>",
                        jaiOpName((OpCode)op));
            }
            return false;   /* an opcode this tier does not speak */
        }
    }
    return !e->failed;
}

/* ------------------------------------------------------------------ */
/* Assembly                                                             */
/* ------------------------------------------------------------------ */


/* The kinds of the parameters, read off the arguments this call was made with.
 * Everything downstream is specialised to them, and the entry guard re-checks
 * them on every later call. */
static bool seedLocals(Emit *e, Value *slotBase) {
    e->observed = slotBase;
    for (unsigned i = 0; i < e->base + e->locals; i++) {
        e->localKind[i]   = SLOT_INT;
        e->localShape[i]  = 0;
        e->localClass[i]  = NULL;
        e->localTyped[i]  = false;
    }
    for (unsigned i = e->base; i <= e->arity; i++) {
        Value v = slotBase[i];
        if (IS_INT(v)) {
            e->localKind[i] = SLOT_INT;
        } else if (IS_FLOAT(v)) {
            e->localKind[i] = SLOT_FLOAT;
        } else if (IS_INSTANCE(v) && AS_INSTANCE(v)->klass != NULL) {
            e->localKind[i]  = SLOT_INST;
            e->localClass[i] = AS_INSTANCE(v)->klass;
            e->localShape[i] = AS_INSTANCE(v)->klass->shapeId;
        } else if (IS_LIST(v)) {
            e->localKind[i] = SLOT_LIST;
        } else if (IS_OBJ(v)) {
            e->localKind[i] = SLOT_OBJ;
        } else if (i == 0) {
            /* A plain function's slot 0 is the closure being called. Nothing
             * can be done with it, but the body has no reason to read it. */
            e->localKind[i] = SLOT_OPAQUE;
        } else {
            e->whyNot = "a parameter is not an int, a float or an instance";
            return false;
        }
        e->localTyped[i] = true;
    }
    return true;
}

/* A local that is not a parameter takes the kind of the first thing bound to
 * it; after that it must keep it, because every read of it was compiled to one
 * instruction chosen from that kind. */
static bool adoptLocalKindSeen(Emit *e, unsigned slot, SlotKind kind,
                               uint32_t shape, ObjClass *klass, Value seen) {
    if (!IS_NULL(seen)) e->localSeen[slot] = seen;
    if (!e->localTyped[slot]) {
        e->localKind[slot]  = kind;
        e->localShape[slot] = shape;
        e->localClass[slot] = klass;
        e->localTyped[slot] = true;
        return true;
    }
    if (e->localKind[slot] == kind && e->localShape[slot] == shape) return true;

    /* Two kinds for one slot. Rather than give the function up, note that this
     * slot has to carry its tag and let the caller compile again with that
     * decided from the start -- every read of it then guards, so the two
     * kinds stop being a contradiction. */
    if (!e->dynamicLocal[slot]) {
        e->needDynamic[slot] = true;
        return false;
    }
    e->localKind[slot]  = kind;
    e->localShape[slot] = shape;
    e->localClass[slot] = klass;
    return true;
}

static bool adoptLocalKind(Emit *e, unsigned slot, SlotKind kind,
                           uint32_t shape, ObjClass *klass) {
    return adoptLocalKindSeen(e, slot, kind, shape, klass, NULL_VAL);
}

static void jitFree(int *map, int *depths, int count) {
    JAI_FREE_ARRAY(int, map, count);
    JAI_FREE_ARRAY(int, depths, count);
}

static bool eligible(ObjFunction *fn) {
    const char *why = NULL;
    /* Arity 0 is allowed: a method whose only parameter is the receiver has
     * arity 0, and getters are the commonest shape there is. A plain function
     * with no arguments reaches here too and declines on its opcodes. */
    if (fn->arity > JIT_MAX_ARITY) why = "arity";
    else if (fn->maxSlots < 1 || (unsigned)fn->maxSlots > JIT_MAX_SLOTS) why = "maxSlots";
    else if (fn->defaultCount != 0) why = "defaults";
    else if (fn->flags & (FN_VARIADIC | FN_KWREST)) why = "flags";

    else if (fn->module == NULL) why = "no module";
    else if (fn->chunk.count <= 0) why = "empty";
    if (why != NULL) {
        if (getenv("JAI_JIT_WHY")) {
            fprintf(stderr, "[jit] %s ineligible: %s (arity=%d maxSlots=%d)\n",
                    fn->name ? fn->name->chars : "<anon>", why, (int)fn->arity,
                    (int)fn->maxSlots);
        }
        return false;
    }
    return true;
}

static bool compileFuncOnce(ObjClosure *closure, Value *slotBase,
                            const bool *dynamic, bool *needDynamic);

bool jaiJitCompileFunc(ObjClosure *closure, Value *slotBase) {
    ObjFunction *fn = closure->fn;
    if (!eligible(fn)) return false;

    /* Up to a few attempts: each one may discover another slot that two paths
     * disagree about, and the next begins knowing it. */
    bool dynamic[JIT_MAX_SLOTS + 1];
    bool need[JIT_MAX_SLOTS + 1];
    memset(dynamic, 0, sizeof dynamic);
    for (int attempt = 0; attempt < 4; attempt++) {
        memset(need, 0, sizeof need);
        if (compileFuncOnce(closure, slotBase, dynamic, need)) return true;
        bool grew = false;
        for (unsigned i = 0; i <= JIT_MAX_SLOTS; i++) {
            if (need[i] && !dynamic[i]) { dynamic[i] = true; grew = true; }
        }
        if (!grew) return false;
    }
    return false;
}

static bool compileFuncOnce(ObjClosure *closure, Value *slotBase,
                            const bool *dynamic, bool *needDynamic) {
    ObjFunction *fn = closure->fn;

    if (getenv("JAI_JIT_WHY")) {
        fprintf(stderr, "[jit] considering %s\n",
                fn->name ? fn->name->chars : "<anon>");
    }

    JaiCodeArena *arena = jaiJitArena();
    if (arena == NULL) return false;

    int *map = JAI_ALLOC(int, fn->chunk.count + 1);
    int *depths = JAI_ALLOC(int, fn->chunk.count + 1);
    for (int i = 0; i <= fn->chunk.count; i++) { map[i] = -1; depths[i] = -1; }


    /* Static, not automatic: at this size two of them would be a large stack
     * frame, and compilation is not reentrant -- nothing it calls compiles
     * anything. */
    static Emit e;
    memset(&e, 0, sizeof e);
    memcpy(e.dynamicLocal, dynamic, sizeof e.dynamicLocal);
    e.arity        = fn->arity;
    e.offsetToInst  = map;
    e.offsetToDepth = depths;
    e.limitLiteral = -1;
    e.bailBlock    = -1;

    /* The prologue cannot be emitted first: its save set depends on how deep
     * the operand stack gets, which only the body knows. So the body goes into
     * the buffer at a fixed offset and the prologue is written in front of it
     * afterwards, with every instruction index shifted by the same amount. */
    static Emit body;
    memset(&body, 0, sizeof body);
    memcpy(body.dynamicLocal, dynamic, sizeof body.dynamicLocal);
    body.arity        = fn->arity;
    /* The measuring pass runs with slot 0 available, purely to find out
     * whether the body reads it; the real pass then drops it if not. */
    body.base         = 0;
    body.locals       = (unsigned)fn->maxSlots;
    body.usesUpvalues = fn->upvalueCount > 0;
    body.callsOut     = true;      /* the measuring pass may emit one */
    body.measuring    = true;
    body.descOffset   = 16u;
    body.offsetToInst = map;
    body.offsetToDepth = depths;
    if (!seedLocals(&body, slotBase)) {
        if (getenv("JAI_JIT_WHY")) {
            fprintf(stderr, "[jit] %s stopped: %s\n",
                    fn->name ? fn->name->chars : "<anon>",
                    body.whyNot ? body.whyNot : "its arguments");
        }
        jitFree(map, depths, fn->chunk.count + 1);
        return false;
    }

    /* A first pass with a provisional frame, only to learn maxValue. The
     * emitted words are thrown away: frameBytes appears in the epilogue, so
     * they would be wrong. */
    body.savedCount = JIT_MAX_SAVED;
    body.frameBytes = 16 + 8 * JIT_MAX_SAVED + 8;   /* 16-aligned below */
    body.frameBytes = (body.frameBytes + 15u) & ~15u;
    if (!compileBody(&body, closure)) {
        memcpy(needDynamic, body.needDynamic, sizeof body.needDynamic);
        if (getenv("JAI_JIT_WHY")) {
            fprintf(stderr, "[jit] %s stopped (measuring): %s\n",
                    fn->name ? fn->name->chars : "<anon>",
                    body.whyNot ? body.whyNot : jaiOpName((OpCode)body.lastOp));
        }
        jitFree(map, depths, fn->chunk.count + 1);
        return false;
    }

    /* A self-call cannot reproduce slot 0: no register holds the callee. A
     * body that both recurses and reads slot 0 is not compiled. */
    if (body.usesSlot0 && body.hasSelfCall) {
        jitFree(map, depths, fn->chunk.count + 1);
        return false;
    }
    e.base         = body.usesSlot0 ? 0u : 1u;
    /* Only as far as the body reaches, never the whole window. */
    unsigned highest = body.maxSlotUsed;
    if (highest < fn->arity) highest = fn->arity;
    e.locals       = highest + 1u - e.base;
    e.usesUpvalues = fn->upvalueCount > 0;

    unsigned argCount = fn->arity + 1u - e.base;
    unsigned closureArg = argCount;
    if (e.usesUpvalues) argCount++;
    if (argCount > JIT_MAX_ARITY) {
        e.whyNot = "more incoming arguments than argument registers";
    }

    unsigned extra = e.usesUpvalues ? 1u : 0u;
    unsigned saved = e.locals + extra + body.maxValue;
    for (unsigned i = 0; i <= JIT_MAX_SLOTS; i++) {
        /* A tag only has somewhere to live in the frame. */
        if (e.dynamicLocal[i]) { saved = JIT_MAX_SAVED + 1; break; }
    }
    if (saved > JIT_MAX_SAVED) {
        /* Too many to keep in registers, so the locals go to the frame and the
         * operand stack keeps the registers. Only the operand stack has to fit
         * now, which is a far lower bar: it is expression depth, not the
         * number of variables a function happens to declare. */
        e.spilled = true;
        saved = extra + body.maxValue;
        if (saved > JIT_MAX_SAVED) {
            e.whyNot = "the operand stack alone exceeds the registers";
        }
    }
    if (e.whyNot != NULL) {
        if (getenv("JAI_JIT_WHY")) {
            fprintf(stderr, "[jit] %s stopped: %s (%u locals, %u stack)\n",
                    fn->name ? fn->name->chars : "<anon>", e.whyNot, e.locals,
                    body.maxValue);
        }
        jitFree(map, depths, fn->chunk.count + 1);
        return false;
    }

    e.savedCount = saved;
    e.callsOut   = body.callsOut;
    unsigned frame = 16u + 8u * saved;
    if (e.spilled) {
        e.localsFrameOffset = frame;
        frame += 16u * e.locals;
        frame = (frame + 15u) & ~15u;
    }
    if (e.callsOut) {
        e.descOffset = frame;
        frame += (unsigned)sizeof(JitCallDesc);
    }
    e.frameBytes = (frame + 15u) & ~15u;
    if (e.frameBytes > 4095u) {
        if (getenv("JAI_JIT_WHY")) {
            fprintf(stderr, "[jit] %s stopped: frame of %u bytes\n",
                    fn->name ? fn->name->chars : "<anon>", e.frameBytes);
        }
        jitFree(map, depths, fn->chunk.count + 1);
        return false;
    }

    /* Prologue. */
    emitFrameEnter(&e);
    emitSaveRestore(&e, true);
    /* The real arguments land in the local registers in order; the closure,
     * when there is one, is the last incoming register but lives just past the
     * locals, where closureReg expects it. Placing it by argument index
     * instead put it three registers low and the first upvalue read
     * dereferenced whatever was there. */
    unsigned realArgs = e.usesUpvalues ? argCount - 1u : argCount;
    for (unsigned i = 0; i < realArgs; i++) {
        if (e.spilled) {
            /* A spilled local is a whole Value: tag first, payload eight bytes
             * on. Writing the payload at the tag's offset instead leaves the
             * payload word untouched, so the first read of an argument gets
             * whatever the frame happened to hold -- which is a small integer
             * often enough that it reads as a pointer and the crash lands
             * somewhere else entirely. */
            /* From the measuring pass: `e`'s own kinds are seeded after this
             * point, so reading them here would take whatever the struct was
             * zeroed to. */
            emit(&e, jaiA64MovzX(JIT_SCRATCH_D,
                                 localTagFor(&body, e.base + i), 0));
            emit(&e, jaiA64StrW(JIT_SCRATCH_D, 31,
                                localFrameOff(&e, e.base + i)));
            emit(&e, jaiA64StrX(i, 31, localFrameOff(&e, e.base + i) + 8));
        } else {
            emit(&e, jaiA64MovX(JIT_FIRST_SAVED + i, i));
        }
    }
    if (e.usesUpvalues) {
        emit(&e, jaiA64MovX(closureReg(&e), realArgs));
    }
    /* A local the interpreter would have left as NULL_VAL starts at zero here.
     * The checker guarantees definite assignment before any read, so this is
     * belt and braces -- but a register holding the last call's value would be
     * a bug that only shows up under recursion. */
    for (unsigned i = realArgs; i < e.locals; i++) {
        if (e.spilled) {
            emit(&e, jaiA64MovzX(JIT_SCRATCH_C, 0, 0));
            emit(&e, jaiA64MovzX(JIT_SCRATCH_D, VAL_NULL, 0));
            emit(&e, jaiA64StrW(JIT_SCRATCH_D, 31,
                                localFrameOff(&e, e.base + i)));
            emit(&e, jaiA64StrX(JIT_SCRATCH_C, 31,
                                localFrameOff(&e, e.base + i) + 8));
        } else {
            emit(&e, jaiA64MovzX(JIT_FIRST_SAVED + i, 0, 0));
        }
    }
    /* Stack guard: bail rather than run off the end of the thread's stack,
     * so that runaway recursion still becomes a RecursionError. */
    int guardLoad = (int)e.count;
    emit(&e, jaiA64LdrLit(JIT_SCRATCH_A, 0));         /* patched below */
    emit(&e, jaiA64AddXImm(JIT_SCRATCH_B, 31, 0));    /* mov x10, sp */
    emit(&e, jaiA64SubsXReg(31, JIT_SCRATCH_B, JIT_SCRATCH_A));
    unsigned guardBranch = e.count;
    emit(&e, jaiA64BCond(JAI_A64_LO, 0));             /* patched below */

    unsigned prologue = e.count;

    for (int i = 0; i <= fn->chunk.count; i++) { map[i] = -1; depths[i] = -1; }
    e.offsetToInst  = map;
    e.offsetToDepth = depths;
    if (!seedLocals(&e, slotBase)) {
        jitFree(map, depths, fn->chunk.count + 1);
        return false;
    }
    if (!compileBody(&e, closure) && getenv("JAI_JIT_WHY")) {
        fprintf(stderr, "[jit] %s stopped: %s\n", fn->name ? fn->name->chars : "<anon>",
                e.whyNot ? e.whyNot : "an unsupported operand form");
    }
    if (e.failed || e.whyNot != NULL)
        { jitFree(map, depths, fn->chunk.count + 1); return false; }
    if (e.failed) {
        if (getenv("JAI_JIT_WHY")) {
            fprintf(stderr, "[jit] %s stopped: %s\n",
                    fn->name ? fn->name->chars : "<anon>",
                    e.whyNot ? e.whyNot : "the emitter ran out of room");
        }
        jitFree(map, depths, fn->chunk.count + 1);
        return false;
    }
    (void)prologue;

    /* The bail block: say so, return anything, and let the caller throw the
     * whole computation away. */
    e.bailBlock = (int)e.count;
    emit(&e, jaiA64MovzX(0, 0, 0));
    emitEpilogue(&e, 1);

    /* The callee raised. The interpreter owns the exception and must not run
     * this call again, so this is a third answer, not a bail. */
    e.exceptionExit = (int)e.count;
    emit(&e, jaiA64MovzX(0, 0, 0));
    emitEpilogue(&e, 2);

    /* One stub per guard, out of line. Each writes the record the interpreter
     * resumes from: the locals, the operand stack as it stood at that
     * instruction, and the offset of the instruction itself. */
    for (unsigned k = 0; k < e.deoptCount; k++) {
        e.deopt[k].stub = (int)e.count;
        emitConst64(&e, JIT_SCRATCH_A, (int64_t)(uintptr_t)&gDeopt);

        for (unsigned i = 0; i < (e.osr ? 0u : e.locals); i++) {
            unsigned slot = e.base + i;
            SlotKind kind = e.localKind[slot];
            unsigned tag = kind == SLOT_INT   ? VAL_INT
                         : kind == SLOT_FLOAT ? VAL_FLOAT
                         : kind == SLOT_BOOL  ? VAL_BOOL
                         : (kind == SLOT_INST || kind == SLOT_LIST) ? VAL_OBJ
                                              : VAL_NULL;
            unsigned at = (unsigned)offsetof(JitDeoptRecord, locals) +
                          i * (unsigned)sizeof(Value);
            emit(&e, jaiA64MovzX(JIT_SCRATCH_B, tag, 0));
            emit(&e, jaiA64StrW(JIT_SCRATCH_B, JIT_SCRATCH_A, at));
            if (tag == VAL_NULL) {
                emit(&e, jaiA64MovzX(JIT_SCRATCH_B, 0, 0));
                emit(&e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_A, at + 8));
            } else if (e.spilled) {
                /* Tag included: for a dynamic local the compile-time kind is
                 * not what it holds. */
                emit(&e, jaiA64LdrW(JIT_SCRATCH_C, 31, localFrameOff(&e, slot)));
                emit(&e, jaiA64StrW(JIT_SCRATCH_C, JIT_SCRATCH_A, at));
                emit(&e, jaiA64LdrX(JIT_SCRATCH_C, 31,
                                    localFrameOff(&e, slot) + 8));
                emit(&e, jaiA64StrX(JIT_SCRATCH_C, JIT_SCRATCH_A, at + 8));
            } else {
                emit(&e, jaiA64StrX(JIT_FIRST_SAVED + i, JIT_SCRATCH_A, at + 8));
            }
        }

        unsigned valueSeen = 0;
        for (unsigned i = 0; i < e.deopt[k].depth; i++) {
            SlotKind kind = e.deopt[k].kinds[i];
            unsigned at = (unsigned)offsetof(JitDeoptRecord, stack) +
                          i * (unsigned)sizeof(Value);
            if (kind == SLOT_CLASS || kind == SLOT_SELF ||
                kind == SLOT_FUNC || kind == SLOT_NATIVE) {
                /* Neither holds a register; both are compile-time constants. */
                uintptr_t p = kind != SLOT_SELF
                                  ? (uintptr_t)e.deopt[k].classes[i]
                                  : (uintptr_t)closure;
                emit(&e, jaiA64MovzX(JIT_SCRATCH_B, VAL_OBJ, 0));
                emit(&e, jaiA64StrW(JIT_SCRATCH_B, JIT_SCRATCH_A, at));
                emitConst64(&e, JIT_SCRATCH_B, (int64_t)p);
                emit(&e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_A, at + 8));
                continue;
            }
            bool fromDesc = e.deopt[k].lastFromDesc &&
                            i + 1 == e.deopt[k].depth;
            if (fromDesc) {
                /* The result of a call that has already happened: it is in the
                 * descriptor, not in a register, and its tag is whatever the
                 * callee actually returned. */
                unsigned rat = e.descOffset +
                               (unsigned)offsetof(JitCallDesc, result);
                emit(&e, jaiA64LdrW(JIT_SCRATCH_B, 31, rat));
                emit(&e, jaiA64StrW(JIT_SCRATCH_B, JIT_SCRATCH_A, at));
                emit(&e, jaiA64LdrX(JIT_SCRATCH_B, 31, rat + 8));
                emit(&e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_A, at + 8));
                valueSeen++;
                continue;
            }
            unsigned tag = kind == SLOT_INT   ? VAL_INT
                         : kind == SLOT_FLOAT ? VAL_FLOAT
                         : kind == SLOT_BOOL  ? VAL_BOOL
                                              : VAL_OBJ;
            unsigned reg = JIT_FIRST_SAVED + regBase(&e) +
                           (e.usesUpvalues ? 1u : 0u) + valueSeen;
            valueSeen++;
            emit(&e, jaiA64MovzX(JIT_SCRATCH_B, tag, 0));
            emit(&e, jaiA64StrW(JIT_SCRATCH_B, JIT_SCRATCH_A, at));
            emit(&e, jaiA64StrX(reg, JIT_SCRATCH_A, at + 8));
        }

        emitConst64(&e, JIT_SCRATCH_B, (int64_t)e.deopt[k].ip);
        emit(&e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_A,
                            (unsigned)offsetof(JitDeoptRecord, ip)));
        emit(&e, jaiA64MovzX(JIT_SCRATCH_B, e.base, 0));
        emit(&e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_A,
                            (unsigned)offsetof(JitDeoptRecord, base)));
        emit(&e, jaiA64MovzX(JIT_SCRATCH_B, e.locals, 0));
        emit(&e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_A,
                            (unsigned)offsetof(JitDeoptRecord, nlocals)));
        emit(&e, jaiA64MovzX(JIT_SCRATCH_B, e.deopt[k].depth, 0));
        emit(&e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_A,
                            (unsigned)offsetof(JitDeoptRecord, nstack)));

        emit(&e, jaiA64MovzX(0, 0, 0));
        emitEpilogue(&e, 4);
    }

    /* One throwing stub per operator, out of line: the hot path keeps the same
     * single not-taken b.vs it always had. */
    for (unsigned i = 0; i < 3; i++) {
        if (!e.overflowUsed[i]) { e.overflowStub[i] = -1; continue; }
        e.overflowStub[i] = (int)e.count;
        emit(&e, jaiA64MovzX(0, i, 0));
        emitConst64(&e, JIT_SCRATCH_A, (int64_t)(uintptr_t)&jitThrowOverflow);
        emit(&e, jaiA64Blr(JIT_SCRATCH_A));
        emit(&e, jaiA64MovzX(0, 0, 0));
        emitEpilogue(&e, 2);
    }

    /* Literal pool, 8-byte aligned so the 64-bit loads are aligned. */
    if ((e.count & 1u) != 0) emit(&e, jaiA64Nop());
    e.limitLiteral = (int)e.count;
    uintptr_t limit = stackLimit();
    if (limit == 0) {
        if (getenv("JAI_JIT_WHY")) {
            fprintf(stderr, "[jit] %s stopped: no stack bound available\n",
                    fn->name ? fn->name->chars : "<anon>");
        }
        jitFree(map, depths, fn->chunk.count + 1);
        return false;
    }
    emit(&e, (uint32_t)(uint64_t)limit);
    emit(&e, (uint32_t)((uint64_t)limit >> 32));

    if (e.failed) {
        if (getenv("JAI_JIT_WHY")) {
            fprintf(stderr, "[jit] %s stopped: %s\n",
                    fn->name ? fn->name->chars : "<anon>",
                    e.whyNot ? e.whyNot : "the emitter ran out of room");
        }
        jitFree(map, depths, fn->chunk.count + 1);
        return false;
    }

    /* Patch the two literal loads and the guard branch. */
    e.code[guardLoad] = jaiA64LdrLit(JIT_SCRATCH_A, e.limitLiteral - guardLoad);
    e.code[guardBranch] =
        jaiA64BCond(JAI_A64_LO, e.bailBlock - (int)guardBranch);

    /* Patch every branch and the recursive calls. */
    for (unsigned i = 0; i < e.fixupCount; i++) {
        const Fixup *f = &e.fixups[i];
        int target;
        if (f->targetOffset == FIXUP_BAIL) {
            target = e.bailBlock;
        } else if (f->targetOffset == FIXUP_THREW) {
            target = e.exceptionExit;
        } else if (f->targetOffset <= FIXUP_EXIT &&
                   f->targetOffset > FIXUP_EXIT - 8u) {
            target = e.exitStub[FIXUP_EXIT - f->targetOffset];
            if (target < 0) {
                if (getenv("JAI_JIT_WHY")) {
                    fprintf(stderr, "[jit] %s stopped: a stub it branches to "
                                    "was never emitted\n",
                            fn->name ? fn->name->chars : "<anon>");
                }
                jitFree(map, depths, fn->chunk.count + 1);
                return false;
            }
        } else if (f->targetOffset <= FIXUP_DEOPT &&
                   f->targetOffset > FIXUP_DEOPT - JIT_MAX_DEOPT) {
            target = e.deopt[FIXUP_DEOPT - f->targetOffset].stub;
            if (target < 0) {
                if (getenv("JAI_JIT_WHY")) {
                    fprintf(stderr, "[jit] %s stopped: a stub it branches to "
                                    "was never emitted\n",
                            fn->name ? fn->name->chars : "<anon>");
                }
                jitFree(map, depths, fn->chunk.count + 1);
                return false;
            }
        } else if (f->targetOffset <= FIXUP_OVF &&
                   f->targetOffset >= FIXUP_OVF - 2u) {
            target = e.overflowStub[FIXUP_OVF - f->targetOffset];
            if (target < 0) {
                if (getenv("JAI_JIT_WHY")) {
                    fprintf(stderr, "[jit] %s stopped: a stub it branches to "
                                    "was never emitted\n",
                            fn->name ? fn->name->chars : "<anon>");
                }
                jitFree(map, depths, fn->chunk.count + 1);
                return false;
            }
        } else if (f->targetOffset == FIXUP_ENTRY) {
            target = 0;
        } else {
            if (f->targetOffset > (uint32_t)fn->chunk.count) {
                jitFree(map, depths, fn->chunk.count + 1);
                return false;
            }
            target = map[f->targetOffset];
            if (target < 0) {
                jitFree(map, depths, fn->chunk.count + 1);
                return false;
            }
            /* Registers are assigned from the operand-stack depth at each
             * point, so a join reached at two different depths would read a
             * value out of a register that holds something else. The walk
             * through the bytecode is linear and cannot see that, so it is
             * checked here and the function is declined rather than
             * mis-compiled. */
            if (f->depth >= 0 && depths[f->targetOffset] != f->depth) {
                if (getenv("JAI_JIT_WHY")) {
                    fprintf(stderr, "[jit] %s stopped: offset %u is reached "
                                    "with two different operand stacks\n",
                            fn->name ? fn->name->chars : "<anon>",
                            f->targetOffset);
                }
                jitFree(map, depths, fn->chunk.count + 1);
                return false;
            }
        }
        int rel = target - f->instIndex;
        uint32_t word = e.code[f->instIndex];
        if ((word & 0xfc000000u) == 0x94000000u) {
            e.code[f->instIndex] = jaiA64Bl(rel);
        } else if (f->conditional) {
            e.code[f->instIndex] = jaiA64BCond(word & 0xfu, rel);
        } else {
            e.code[f->instIndex] = jaiA64B(rel);
        }
    }
    /* JAI_JIT_DUMP=<function> writes that function's words to
     * jit_<function>.bin and prints the bytecode-offset-to-instruction map, so
     * generated code can be read back:
     *
     *     llvm-mc --disassemble --triple=aarch64
     *
     * on the file's bytes. Reading the code is how the register plan gets
     * checked at all -- three of the bugs in this tier were found no other
     * way, and none of them were visible in the emitter's own bookkeeping. */
    {
        const char *dump = getenv("JAI_JIT_DUMP");
        if (dump != NULL && fn->name != NULL &&
            strcmp(dump, fn->name->chars) == 0) {
            char path[256];
            snprintf(path, sizeof path, "jit_%s.bin", fn->name->chars);
            FILE *fp = fopen(path, "wb");
            if (fp != NULL) {
                fwrite(e.code, sizeof e.code[0], e.count, fp);
                fclose(fp);
                fprintf(stderr, "[jit] %u words to %s\n", e.count, path);
                for (unsigned i = 0; i <= (unsigned)fn->chunk.count; i++) {
                    if (map[i] >= 0) {
                        fprintf(stderr, "[jit] bc %u is inst %d\n", i, map[i]);
                    }
                }
            }
        }
    }
    jitFree(map, depths, fn->chunk.count + 1);

    /* A `bl` at instruction i must reach instruction 0 of this function, so
     * the recursive-call fixups above are relative to the function's own
     * start, which is where the arena is about to place it. */
    if (!jaiCodeArenaUnseal(arena)) return false;
    /* 8-align the entry so the literal pool's alignment is what it looks. */
    while ((arena->used & 7u) != 0) {
        uint32_t pad = jaiA64Nop();
        if (jaiCodeArenaWrite(arena, &pad, sizeof pad) == NULL) return false;
    }
    uint8_t *entry = jaiCodeArenaWrite(arena, e.code, e.count * sizeof e.code[0]);
    if (entry == NULL) return false;
    if (!jaiCodeArenaSeal(arena)) return false;

    /* The tier's whole bail protocol rests on partial execution being
     * invisible. Field writes ended that: a body that stores to an instance
     * and then bails would have the store applied again by the interpreted
     * re-run. Nothing in the suite hits this -- an initializer's writes come
     * after its last guard -- but "hard to construct" is not the standard a
     * compiled tier gets to work to. */
    if (e.whyNot != NULL && getenv("JAI_JIT_WHY")) {
        fprintf(stderr, "[jit] %s stopped: %s\n",
                fn->name ? fn->name->chars : "<anon>", e.whyNot);
    }
    if (e.bailAfterWrite) {
        if (getenv("JAI_JIT_WHY")) {
            fprintf(stderr, "[jit] %s declined: a bail follows a heap write\n",
                    fn->name ? fn->name->chars : "<anon>");
        }
        jitFree(map, depths, fn->chunk.count + 1);
        return false;
    }

    if (e.assumedIntReturn && e.returnKind != SLOT_INT) {
        if (getenv("JAI_JIT_WHY")) {
            fprintf(stderr, "[jit] %s stopped: a self-call had to guess the "
                            "return kind and guessed wrong\n",
                    fn->name ? fn->name->chars : "<anon>");
        }
        /* A self-call before the first return had to guess, and guessed
         * wrong. */
        return false;
    }
    for (unsigned i = 0; i < argCount; i++) {
        if (e.usesUpvalues && i == closureArg) {
            fn->jitParamKind[i]  = (uint8_t)SLOT_CLOSURE;
            fn->jitParamShape[i] = 0;
            continue;
        }
        fn->jitParamKind[i]  = (uint8_t)e.localKind[i + e.base];
        fn->jitParamShape[i] = e.localShape[i + e.base];
    }
    fn->jitReturnKind = (uint8_t)e.returnKind;
    fn->jitReturnShape = e.returnShape;
    if (e.returnKind == SLOT_INST && e.returnShape != 0) {
        ObjClass *rc = NULL;
        for (unsigned i = 0; i < e.base + e.locals; i++) {
            if (e.localClass[i] != NULL &&
                e.localClass[i]->shapeId == e.returnShape) {
                rc = e.localClass[i];
                break;
            }
        }
        jaiClassRememberShape(rc);
    }
    fn->jitArgBase    = (uint8_t)e.base;
    fn->jitArgCount   = (uint8_t)argCount;

    if (getenv("JAI_JIT_WHY")) {
        fprintf(stderr,
                "[jit] compiled %s  arity=%u locals=%u insts=%u saved=%u "
                "spill=%d fix=%u deopt=%u maxval=%u base=%u\n",
                fn->name ? fn->name->chars : "<anon>", e.arity, e.locals,
                e.count, e.savedCount, (int)e.spilled, e.fixupCount,
                e.deoptCount, body.maxValue, e.base);
    }
    fn->jitFunc = entry;
    return true;
}

/* ------------------------------------------------------------------ */
/* On-stack replacement                                                 */
/* ------------------------------------------------------------------ */

/* Returns the bytecode offset the interpreter should continue from. */
typedef int64_t (*OsrFn)(Value *slots);
typedef int64_t (*OsrFnIter)(Value *slots, ObjIter *iter);

/* The OP_LOOP that jumps back to `top`, and so the end of the loop. Gives up
 * on OP_CLOSURE, whose length depends on its operands. */
/* Is `top` where an instruction actually starts?
 *
 * findLoopEnd walks from `top` itself, so if that offset is in the middle of
 * an instruction the walk decodes operands as opcodes and can find a plausible
 * OP_LOOP that is not one. nbody's advance was being compiled from offset 128,
 * which is inside a GET_LOCAL2's operands. Walking from the start costs a scan
 * once per compile and removes the question. */
static bool isInstructionStart(const Chunk *c, uint32_t top) {
    for (int off = 0; off < c->count;) {
        if ((uint32_t)off == top) return true;
        uint8_t op = c->code[off];
        if (op == OP_CLOSURE) return false;   /* variable length */
        int len = 1 + jaiOpOperandSize((OpCode)op);
        if (len <= 0) return false;
        off += len;
    }
    return false;
}

static uint32_t findLoopEnd(const Chunk *c, uint32_t top) {
    for (int off = (int)top; off < c->count;) {
        uint8_t op = c->code[off];
        if (op == OP_CLOSURE) return 0;
        int len = 1 + jaiOpOperandSize((OpCode)op);
        if (len <= 0) return 0;
        if (op == OP_LOOP) {
            int16_t jump = jaiReadI16(c->code + off + 1);
            if ((uint32_t)((int32_t)(off + 3) + jump) == top) {
                return (uint32_t)(off + 3);
            }
        }
        off += len;
    }
    return 0;
}

static bool compileOsr(ObjClosure *closure, uint32_t top, Value *slots,
                       bool hasIter, bool noInline) {
    ObjFunction *fn = closure->fn;
    if (!isInstructionStart(&fn->chunk, top)) return false;
    uint32_t end = findLoopEnd(&fn->chunk, top);
    if (end == 0 || end <= top) return false;
    /* The entry re-checks every slot, so this is the size of that record --
     * nbody's advance declares nineteen. */
    if (fn->maxSlots < 1 || (unsigned)fn->maxSlots > 40) return false;

    JaiCodeArena *arena = jaiJitArena();
    if (arena == NULL) return false;

    int *map = JAI_ALLOC(int, fn->chunk.count + 1);
    int *depths = JAI_ALLOC(int, fn->chunk.count + 1);
    for (int i = 0; i <= fn->chunk.count; i++) { map[i] = -1; depths[i] = -1; }

    static Emit e;
    memset(&e, 0, sizeof e);
    e.osr = true;
    e.hasIter = hasIter;
    e.osrTop = top;
    e.osrEnd = end;
    e.base = 0;
    e.locals = (unsigned)fn->maxSlots;
    e.limitLiteral = -1;
    e.bailBlock = -1;
    e.exceptionExit = -1;
    e.callsOut = true;
    e.noInline = noInline;
    e.observed = slots;
    e.offsetToInst = map;
    e.offsetToDepth = depths;
    e.savedCount = JIT_MAX_SAVED;
    /* Registers if they fit, memory otherwise. The reserved four (or one) plus
     * the locals plus the deepest expression must all sit inside the ten
     * callee-saved registers; a first pass measures the last of those. */
    {
        static Emit probe;
        memset(&probe, 0, sizeof probe);
        probe.osr = true; probe.measuring = true; probe.hasIter = hasIter;
        probe.osrTop = top; probe.osrEnd = end; probe.base = 0;
        probe.noInline = noInline;
        probe.locals = e.locals; probe.callsOut = true; probe.observed = slots;
        probe.offsetToInst = map; probe.offsetToDepth = depths;
        probe.limitLiteral = -1; probe.bailBlock = -1; probe.exceptionExit = -1;
        for (unsigned i = 0; i < e.locals; i++) {
            probe.localKind[i]  = e.localKind[i];
            probe.localShape[i] = e.localShape[i];
            probe.localClass[i] = e.localClass[i];
            probe.localTyped[i] = e.localTyped[i];
            probe.localSeen[i]  = e.localSeen[i];
        }
        if (compileBody(&probe, closure) && !probe.failed &&
            osrReserved(&e) + e.locals + probe.maxValue <= JIT_MAX_SAVED) {
            e.osrRegLocals = true;
        }
        for (int i = 0; i <= fn->chunk.count; i++) { map[i] = -1; depths[i] = -1; }
    }

    /* Each slot takes the kind it holds right now. The entry re-checks them on
     * every later entry, so this is a specialisation, not an assumption. */
    for (unsigned i = 0; i < e.locals; i++) {
        Value v = slots[i];
        e.localTyped[i] = true;
        if (IS_INT(v))        e.localKind[i] = SLOT_INT;
        else if (IS_FLOAT(v)) e.localKind[i] = SLOT_FLOAT;
        else if (IS_BOOL(v))  e.localKind[i] = SLOT_BOOL;
        else if (IS_LIST(v))  e.localKind[i] = SLOT_LIST;
        else if (IS_INSTANCE(v) && AS_INSTANCE(v)->klass != NULL) {
            e.localKind[i]  = SLOT_INST;
            e.localClass[i] = AS_INSTANCE(v)->klass;
            e.localShape[i] = AS_INSTANCE(v)->klass->shapeId;
        } else if (IS_OBJ(v)) {
            e.localKind[i] = SLOT_OBJ;
        } else {
            /* Nothing recognisable in it yet -- a slot the loop assigns before
             * it reads. It takes its kind from the first thing bound to it. */
            e.localKind[i] = SLOT_OPAQUE;
            e.localTyped[i] = false;
        }
    }

    unsigned frame = 16u + 8u * JIT_MAX_SAVED + (unsigned)sizeof(JitCallDesc);
    e.descOffset = 16u + 8u * JIT_MAX_SAVED;
    e.frameBytes = (frame + 15u) & ~15u;

    emitFrameEnter(&e);
    emitSaveRestore(&e, true);
    emit(&e, jaiA64MovX(JIT_SLOTS_REG, 0));
    if (e.osrRegLocals) {
        /* Payloads only: the entry has already checked every slot's kind, so
         * there is nothing left to guard here. */
        for (unsigned i = 0; i < e.locals; i++) {
            emit(&e, jaiA64LdrX(localReg(&e, i), JIT_SLOTS_REG, i * 16u + 8u));
        }
    }
    if (hasIter) {
        emit(&e, jaiA64MovX(JIT_ITER_REG, 1));
        emit(&e, jaiA64LdrX(JIT_IDX_REG, JIT_ITER_REG,
                            (unsigned)offsetof(ObjIter, index)));
        emit(&e, jaiA64LdrX(JIT_LIM_REG, JIT_ITER_REG,
                            (unsigned)offsetof(ObjIter, limit)));
        /* A range yields start + index, so a loop that does not begin at zero
         * needs its start too. ObjIter.source is a Value, so the object
         * pointer sits eight bytes into it. */
        emit(&e, jaiA64LdrX(JIT_START_REG, JIT_ITER_REG,
                            (unsigned)offsetof(ObjIter, source) + 8));
        emit(&e, jaiA64LdrX(JIT_START_REG, JIT_START_REG,
                            (unsigned)offsetof(ObjRange, start)));
    }

    if (!compileBody(&e, closure) || e.failed) {
        if (getenv("JAI_JIT_WHY")) {
            fprintf(stderr, "[jit] osr at %u stopped: %s\n", top,
                    e.whyNot ? e.whyNot : jaiOpName((OpCode)e.lastOp));
        }
        jitFree(map, depths, fn->chunk.count + 1);
        return false;
    }

    /* Falling off the end of the compiled range is the loop exiting there. */
    if (e.exitCount >= 8) { jitFree(map, depths, fn->chunk.count + 1); return false; }

/* Every way out writes back what the loop was holding: the iterator's index,
 * and the locals if they were living in registers. Miss one and the
 * interpreter carries on from stale values. */
#define OSR_SYNC_ITER()                                                        \
    do {                                                                       \
        if (hasIter) {                                                         \
            emit(&e, jaiA64StrX(JIT_IDX_REG, JIT_ITER_REG,                     \
                                (unsigned)offsetof(ObjIter, index)));          \
        }                                                                      \
        if (e.osrRegLocals) {                                                  \
            for (unsigned li = 0; li < e.locals; li++) {                       \
                SlotKind lk = e.localKind[li];                                 \
                unsigned lt = lk == SLOT_INT   ? VAL_INT                       \
                            : lk == SLOT_FLOAT ? VAL_FLOAT                     \
                            : lk == SLOT_BOOL  ? VAL_BOOL                      \
                            : lk == SLOT_OPAQUE ? VAL_NULL                     \
                                               : VAL_OBJ;                      \
                if (lt == VAL_NULL) continue;                                  \
                emit(&e, jaiA64MovzX(JIT_SCRATCH_D, lt, 0));                   \
                emit(&e, jaiA64StrW(JIT_SCRATCH_D, JIT_SLOTS_REG, li * 16u));  \
                emit(&e, jaiA64StrX(localReg(&e, li), JIT_SLOTS_REG,           \
                                    li * 16u + 8u));                           \
            }                                                                  \
        }                                                                      \
    } while (0)

    e.bailBlock = (int)e.count;
    OSR_SYNC_ITER();
    emitConst64(&e, 0, (int64_t)-1);          /* -1: could not continue */
    emitEpilogue(&e, 0);
    e.exceptionExit = (int)e.count;
    OSR_SYNC_ITER();
    emitConst64(&e, 0, (int64_t)-2);          /* -2: an exception is pending */
    emitEpilogue(&e, 0);

    for (unsigned i = 0; i < 3; i++) {
        if (!e.overflowUsed[i]) { e.overflowStub[i] = -1; continue; }
        e.overflowStub[i] = (int)e.count;
        OSR_SYNC_ITER();
        emit(&e, jaiA64MovzX(0, i, 0));
        emitConst64(&e, JIT_SCRATCH_A, (int64_t)(uintptr_t)&jitThrowOverflow);
        emit(&e, jaiA64Blr(JIT_SCRATCH_A));
        emitConst64(&e, 0, (int64_t)-2);
        emitEpilogue(&e, 0);
    }

    for (unsigned i = 0; i < e.exitCount; i++) {
        e.exitStub[i] = (int)e.count;
        OSR_SYNC_ITER();
        /* Leaving by the iterator's own exit means it is exhausted, and the
         * interpreter drops it there; any other way out leaves it in place. */
        emitConst64(&e, JIT_SCRATCH_A,
                    (int64_t)(uintptr_t)&gDeopt.base);
        emitConst64(&e, JIT_SCRATCH_B,
                    (e.hasIter && e.exitOffset[i] == e.iterExit) ? 1 : 0);
        emit(&e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_A, 0));
        emitConst64(&e, 0, (int64_t)e.exitOffset[i]);
        emitEpilogue(&e, 0);
    }

    for (unsigned k = 0; k < e.deoptCount; k++) {
        e.deopt[k].stub = (int)e.count;
        OSR_SYNC_ITER();
        emitConst64(&e, JIT_SCRATCH_A, (int64_t)(uintptr_t)&gDeopt);
        unsigned valueSeen = 0;
        for (unsigned i = 0; i < e.deopt[k].depth; i++) {
            SlotKind kind = e.deopt[k].kinds[i];
            unsigned at = (unsigned)offsetof(JitDeoptRecord, stack) +
                          i * (unsigned)sizeof(Value);
            if (kind == SLOT_CLASS || kind == SLOT_SELF ||
                kind == SLOT_FUNC || kind == SLOT_NATIVE) {
                uintptr_t pv = kind != SLOT_SELF
                                   ? (uintptr_t)e.deopt[k].classes[i]
                                   : (uintptr_t)closure;
                emit(&e, jaiA64MovzX(JIT_SCRATCH_B, VAL_OBJ, 0));
                emit(&e, jaiA64StrW(JIT_SCRATCH_B, JIT_SCRATCH_A, at));
                emitConst64(&e, JIT_SCRATCH_B, (int64_t)pv);
                emit(&e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_A, at + 8));
                continue;
            }
            if (e.deopt[k].lastFromDesc && i + 1 == e.deopt[k].depth) {
                /* The result of a call that already happened lives in the
                 * descriptor, with whatever tag the callee actually returned.
                 * The function tier's stub knew this; this one did not, and
                 * alloc_churn came back 982406343 instead of 550770565 under
                 * deopt stress -- which is the entire reason that switch
                 * exists. */
                unsigned rat = e.descOffset +
                               (unsigned)offsetof(JitCallDesc, result);
                emit(&e, jaiA64LdrW(JIT_SCRATCH_B, 31, rat));
                emit(&e, jaiA64StrW(JIT_SCRATCH_B, JIT_SCRATCH_A, at));
                emit(&e, jaiA64LdrX(JIT_SCRATCH_B, 31, rat + 8));
                emit(&e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_A, at + 8));
                valueSeen++;
                continue;
            }
            unsigned tag = kind == SLOT_INT   ? VAL_INT
                         : kind == SLOT_FLOAT ? VAL_FLOAT
                         : kind == SLOT_BOOL  ? VAL_BOOL
                                              : VAL_OBJ;
            unsigned reg = JIT_FIRST_SAVED + regBase(&e) + valueSeen;
            valueSeen++;
            emit(&e, jaiA64MovzX(JIT_SCRATCH_B, tag, 0));
            emit(&e, jaiA64StrW(JIT_SCRATCH_B, JIT_SCRATCH_A, at));
            emit(&e, jaiA64StrX(reg, JIT_SCRATCH_A, at + 8));
        }
        emit(&e, jaiA64MovzX(JIT_SCRATCH_B, e.deopt[k].depth, 0));
        emit(&e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_A,
                            (unsigned)offsetof(JitDeoptRecord, nstack)));
        emit(&e, jaiA64MovzX(JIT_SCRATCH_B, 0, 0));
        emit(&e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_A,
                            (unsigned)offsetof(JitDeoptRecord, base)));
        emitConst64(&e, 0, (int64_t)e.deopt[k].ip);
        emitEpilogue(&e, 0);
    }
#undef OSR_SYNC_ITER

    if (e.failed) { jitFree(map, depths, fn->chunk.count + 1); return false; }

    for (unsigned i = 0; i < e.fixupCount; i++) {
        const Fixup *f = &e.fixups[i];
        int target;
        if (f->targetOffset == FIXUP_BAIL) target = e.bailBlock;
        else if (f->targetOffset == FIXUP_THREW) target = e.exceptionExit;
        else if (f->targetOffset <= FIXUP_EXIT && f->targetOffset > FIXUP_EXIT - 8u)
            target = e.exitStub[FIXUP_EXIT - f->targetOffset];
        else if (f->targetOffset <= FIXUP_DEOPT &&
                 f->targetOffset > FIXUP_DEOPT - JIT_MAX_DEOPT)
            target = e.deopt[FIXUP_DEOPT - f->targetOffset].stub;
        else if (f->targetOffset <= FIXUP_OVF && f->targetOffset >= FIXUP_OVF - 2u)
            target = e.overflowStub[FIXUP_OVF - f->targetOffset];
        else {
            if (f->targetOffset > (uint32_t)fn->chunk.count) { jitFree(map, depths, fn->chunk.count + 1); return false; }
            target = map[f->targetOffset];
            if (target < 0 && getenv("JAI_JIT_WHY")) {
                fprintf(stderr, "[jit] %s stopped: a branch to %u, which is "
                                "not an instruction this compiled\n",
                        fn->name ? fn->name->chars : "<anon>", f->targetOffset);
            }
            if (target < 0 ||
                (f->depth >= 0 && depths[f->targetOffset] != f->depth)) {
                jitFree(map, depths, fn->chunk.count + 1);
                return false;
            }
        }
        if (target < 0) { jitFree(map, depths, fn->chunk.count + 1); return false; }
        int rel = target - f->instIndex;
        uint32_t word = e.code[f->instIndex];
        if ((word & 0xfc000000u) == 0x94000000u) e.code[f->instIndex] = jaiA64Bl(rel);
        else if (f->conditional) e.code[f->instIndex] = jaiA64BCond(word & 0xfu, rel);
        else e.code[f->instIndex] = jaiA64B(rel);
    }
    jitFree(map, depths, fn->chunk.count + 1);

    if (!jaiCodeArenaUnseal(arena)) return false;
    while ((arena->used & 7u) != 0) {
        uint32_t pad = jaiA64Nop();
        if (jaiCodeArenaWrite(arena, &pad, sizeof pad) == NULL) return false;
    }
    uint8_t *entry = jaiCodeArenaWrite(arena, e.code, e.count * sizeof e.code[0]);
    if (entry == NULL) return false;
    if (!jaiCodeArenaSeal(arena)) return false;

    fn->osrCode = entry;
    fn->osrTop = top;
    fn->osrHot = true;
    fn->osrDeclines = 0;
    fn->jitModuleVersion = fn->module != NULL ? fn->module->version : 0;
    fn->osrSlots = (uint8_t)e.locals;
    for (unsigned i = 0; i < e.locals; i++) fn->osrKinds[i] = (uint8_t)e.localKind[i];
    if (getenv("JAI_JIT_WHY")) {
        fprintf(stderr, "[jit] osr %s at %u: %u instructions\n",
                fn->name ? fn->name->chars : "<anon>", top, e.count);
    }
    return true;
}

int jaiJitEnterOsr(ObjClosure *closure, uint32_t top, uint32_t *resumeAt) {
    ObjFunction *fn = closure->fn;
    CallFrame *frame = &vm.frames[vm.frameCount - 1];

    /* A for-loop head keeps its iterator on the stack. Only a range from zero
     * in unit steps is taken, because that is what makes the yielded value the
     * index and lets the body run against a plain counter. */
    bool hasIter = top < (uint32_t)fn->chunk.count &&
                   fn->chunk.code[top] == OP_FOR_ITER_BIND;
    ObjIter *iter = NULL;
    if (hasIter) {
        if (vm.stackTop <= frame->slots) return 0;
        Value it = vm.stackTop[-1];
        if (!IS_ITER(it)) return 0;
        iter = AS_ITER(it);
        if (iter->kind != ITER_RANGE || !IS_RANGE(iter->source)) return 0;
        ObjRange *r = AS_RANGE(iter->source);
        /* Unit steps only -- that is what makes the yielded value start plus
         * the index. The start itself is loaded at entry and need not be 0. */
        if (r->step != 1) return 0;
    }

    if (fn->osrCode == NULL) {
        if (fn->osrRefused) return 0;
        if (!compileOsr(closure, top, frame->slots, hasIter, false) &&
            !compileOsr(closure, top, frame->slots, hasIter, true)) {
            /* Inlining widens live ranges; a loop that will not fit with it
             * may fit without, and a compiled call beats no compile at all. */
            if (++fn->osrAttempts >= 5) fn->osrRefused = true;
            return 0;
        }
    }
    if (fn->osrTop != top) return 0;
    if (fn->module == NULL || fn->module->version != fn->jitModuleVersion) return 0;

    /* Every slot must still hold what it held when this was compiled. */
    for (unsigned i = 0; i < fn->osrSlots; i++) {
        Value v = frame->slots[i];
        switch ((SlotKind)fn->osrKinds[i]) {
        case SLOT_INT:   if (!IS_INT(v))   return 0; break;
        case SLOT_FLOAT: if (!IS_FLOAT(v)) return 0; break;
        case SLOT_BOOL:  if (!IS_BOOL(v))  return 0; break;
        case SLOT_LIST:  if (!IS_LIST(v))  return 0; break;
        case SLOT_INST:  if (!IS_INSTANCE(v)) return 0; break;
        default: break;   /* opaque: never read */
        }
    }

    gDeopt.nstack = 0;
    gDeopt.base = 0;
    int64_t at = ((OsrFnIter)(uintptr_t)fn->osrCode)(frame->slots, iter);
    if (at == -1) return 0;
    if (at == -2) return 2;              /* an exception is pending */
    if (gDeopt.base != 0) vm.stackTop--;   /* the exhausted iterator */
    for (int64_t i = 0; i < gDeopt.nstack; i++) *vm.stackTop++ = gDeopt.stack[i];
    *resumeAt = (uint32_t)at;
    return 1;
}

/* ------------------------------------------------------------------ */
/* Entry from the interpreter                                          */
/* ------------------------------------------------------------------ */

typedef JitResult (*Fn0)(void);
typedef JitResult (*Fn1)(int64_t);
typedef JitResult (*Fn2)(int64_t, int64_t);
typedef JitResult (*Fn3)(int64_t, int64_t, int64_t);
typedef JitResult (*Fn4)(int64_t, int64_t, int64_t, int64_t);

JaiJitOutcome jaiJitEnterFunc(ObjClosure *closure, Value *slotBase) {
    ObjFunction *fn = closure->fn;
    if (fn->jitFunc == NULL) return JAI_JIT_DECLINED;

    /* Compiled code reads the global that names this function exactly once,
     * at compile time, and then calls it directly. Rebinding the name has to
     * invalidate that, and a module's version counter moves on every global
     * mutation, so one comparison covers it. It is conservative -- any global
     * write in the module retires the compiled form -- and conservative is the
     * safe direction. */
    if (fn->module == NULL || fn->module->version != fn->jitModuleVersion) {
        return JAI_JIT_DECLINED;
    }

    unsigned arity = fn->jitArgCount;
    int64_t a[JIT_MAX_ARITY];
    for (unsigned i = 0; i < arity; i++) {
        Value v = slotBase[fn->jitArgBase + i];
        switch ((SlotKind)fn->jitParamKind[i]) {
        case SLOT_INT:
            if (!IS_INT(v)) return JAI_JIT_DECLINED;
            a[i] = AS_INT(v);
            break;
        case SLOT_FLOAT: {
            if (!IS_FLOAT(v)) return JAI_JIT_DECLINED;
            double d = AS_FLOAT(v);
            memcpy(&a[i], &d, sizeof a[i]);
            break;
        }
        case SLOT_INST: {
            /* The class as well as the type: every field offset in the body
             * was resolved against this one shape. Compiled code holds the
             * raw pointer, which is safe only because the body cannot
             * allocate -- no collection can run while it does -- and the
             * argument slots keep the instance reachable meanwhile. */
            if (!IS_INSTANCE(v)) return JAI_JIT_DECLINED;
            ObjInstance *inst = AS_INSTANCE(v);
            if (inst->klass == NULL ||
                inst->klass->shapeId != fn->jitParamShape[i]) {
                return JAI_JIT_DECLINED;
            }
            a[i] = (int64_t)(uintptr_t)inst;
            break;
        }
        case SLOT_LIST:
            if (!IS_LIST(v)) return JAI_JIT_DECLINED;
            a[i] = (int64_t)(uintptr_t)AS_LIST(v);
            break;
        case SLOT_OBJ:
            if (!IS_OBJ(v)) return JAI_JIT_DECLINED;
            a[i] = (int64_t)(uintptr_t)AS_OBJ(v);
            break;
        case SLOT_BOOL:
            if (!IS_BOOL(v)) return JAI_JIT_DECLINED;
            a[i] = AS_BOOL(v) ? 1 : 0;
            break;
        case SLOT_OPAQUE:
            a[i] = 0;   /* never read; see seedLocals */
            break;
        case SLOT_CLOSURE:
            /* Not a slot at all: the closure the interpreter is calling. Safe
             * to hold raw for the same reason every other pointer here is --
             * the body cannot allocate, and the caller holds this closure. */
            a[i] = (int64_t)(uintptr_t)closure;
            break;
        default:
            return JAI_JIT_DECLINED;
        }
    }

    JitResult r;
    switch (arity) {
    case 0: r = ((Fn0)(uintptr_t)fn->jitFunc)(); break;
    case 1: r = ((Fn1)(uintptr_t)fn->jitFunc)(a[0]); break;
    case 2: r = ((Fn2)(uintptr_t)fn->jitFunc)(a[0], a[1]); break;
    case 3: r = ((Fn3)(uintptr_t)fn->jitFunc)(a[0], a[1], a[2]); break;
    case 4: r = ((Fn4)(uintptr_t)fn->jitFunc)(a[0], a[1], a[2], a[3]); break;
    default: return JAI_JIT_DECLINED;
    }

    if (r.bailed == 2) return JAI_JIT_ERROR;
    if (r.bailed == 4) return JAI_JIT_DEOPT;
    if (r.bailed) {
        /* Overflow or a stack that ran low. Nothing was written -- the body
         * cannot write -- so handing the call back to the interpreter is
         * enough, and it will raise the error with a traceback. Refusing the
         * function permanently keeps a bailing body from being re-entered on
         * every call only to bail again. */
        fn->jitRefused = true;
        fn->jitFunc = NULL;
        return JAI_JIT_DECLINED;
    }

    switch ((SlotKind)fn->jitReturnKind) {
    case SLOT_INT:
        slotBase[0] = INT_VAL(r.value);
        break;
    case SLOT_FLOAT: {
        double d;
        memcpy(&d, &r.value, sizeof d);
        slotBase[0] = FLOAT_VAL(d);
        break;
    }
    case SLOT_INST:
    case SLOT_LIST:
    case SLOT_OBJ:
        slotBase[0] = OBJ_VAL((Obj *)(uintptr_t)r.value);
        break;
    case SLOT_BOOL:
        slotBase[0] = BOOL_VAL(r.value != 0);
        break;
    case SLOT_NULL:
        slotBase[0] = NULL_VAL;
        break;
    default:
        return JAI_JIT_DECLINED;
    }
    vm.stackTop = slotBase + 1;
    return JAI_JIT_DONE;
}

#else

bool jaiJitCompileFunc(ObjClosure *closure, Value *slotBase) {
    (void)closure; (void)slotBase; return false;
}
JaiJitOutcome jaiJitEnterFunc(ObjClosure *closure, Value *slotBase) {
    (void)closure; (void)slotBase; return JAI_JIT_DECLINED;
}

#endif

/* A conditional branch whose target is reached with a different operand stack
 * than the branch leaves from -- the exhausted arm of a for-loop, where the
 * interpreter drops the iterator. */
static void branchToDepth(Emit *e, uint32_t targetOffset, unsigned cond,
                          int depthOverride) {
    if (e->fixupCount >= JIT_MAX_FIXUPS) { e->failed = true; return; }
    if (e->osr && targetOffset < UINT32_MAX - 64u &&
        (targetOffset < e->osrTop || targetOffset >= e->osrEnd)) {
        targetOffset = exitTargetFor(e, targetOffset);
    }
    e->fixups[e->fixupCount].instIndex    = (int)e->count;
    e->fixups[e->fixupCount].targetOffset = targetOffset;
    e->fixups[e->fixupCount].conditional  = true;
    e->fixups[e->fixupCount].depth        = depthOverride;
    e->fixupCount++;
    emit(e, jaiA64BCond(cond, 0));
}
