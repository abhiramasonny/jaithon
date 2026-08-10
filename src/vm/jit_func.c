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
#define JIT_MAX_INSTS 4096u
#define JIT_MAX_FIXUPS 2048u
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
    SLOT_BOOL,    /* 0 or 1 in a register; a Value's boolean member is its low
                   * byte, so the same word serves both */
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
    return k != SLOT_SELF && k != SLOT_CLASS;
}

/* Two targets are not bytecode offsets at all. */
#define FIXUP_BAIL   UINT32_MAX
#define FIXUP_ENTRY  (UINT32_MAX - 1u)
#define FIXUP_THREW  (UINT32_MAX - 2u)
#define FIXUP_OVF    (UINT32_MAX - 3u)   /* + 0,1,2 for the three operators */
#define FIXUP_DEOPT  (UINT32_MAX - 7u)   /* minus the deopt-site index */
#define FIXUP_EXIT   (UINT32_MAX - 100u) /* minus the loop-exit index */

#define JIT_MAX_DEOPT 64

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

static unsigned regBase(const Emit *e) {
    if (e->osr) return e->hasIter ? 4u : 1u;
    return e->spilled ? 0u : e->locals;
}

static unsigned localReg(const Emit *e, unsigned slot) {
    return JIT_FIRST_SAVED + (slot - e->base);
}

static unsigned localFrameOff(const Emit *e, unsigned slot) {
    return e->localsFrameOffset + (slot - e->base) * 8u;
}

/* A register holding `slot`'s value. In register mode that is the local's own
 * register and `scratch` goes unused; in memory mode the value is loaded into
 * `scratch`. Only the payload moves: a local's kind is fixed for the whole
 * function, so the tag never needs storing. */
static unsigned localIn(Emit *e, unsigned slot, unsigned scratch) {
    if (e->osr) {
        emit(e, jaiA64LdrX(scratch, JIT_SLOTS_REG, slot * 16u + 8u));
        return scratch;
    }
    if (!e->spilled) return localReg(e, slot);
    emit(e, jaiA64LdrX(scratch, 31, localFrameOff(e, slot)));
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
    if (e->osr || e->spilled) return JIT_SCRATCH_C;
    return localReg(e, slot);
}

static void localOut(Emit *e, unsigned slot, unsigned src) {
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
    emit(e, jaiA64StrX(src, 31, localFrameOff(e, slot)));
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
static uint32_t stackSignature(const Emit *e) {
    uint32_t sig = e->depth & 0xfu;
    for (unsigned i = 0; i < e->depth && i < 9; i++) {
        sig |= ((uint32_t)e->stack[i] & 3u) << (4 + 2 * i);
    }
    return sig;
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

static void emitEpilogue(Emit *e, unsigned bailed) {
    emit(e, jaiA64MovzX(1, bailed, 0));
    emitSaveRestore(e, false);
    emit(e, jaiA64LdpPost(29, 30, 31, (int32_t)e->frameBytes));
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
static bool isClassCallee(const Emit *e, unsigned argc) {
    return e->depth >= argc + 1u &&
           e->stack[e->depth - argc - 1] == SLOT_CLASS;
}

static bool adoptLocalKind(Emit *e, unsigned slot, SlotKind kind,
                           uint32_t shape, ObjClass *klass);

static bool emitDescriptor(Emit *e, Value calleeVal, unsigned first,
                           unsigned nargs, void *helper) {
    if (nargs > JIT_MAX_ARGS_OUT) { e->whyNot = "call argc"; return false; }
    if (!e->callsOut) { e->whyNot = "callsOut off"; return false; }

    unsigned d = e->descOffset;

    /* The callee, as a whole Value. */
    emit(e, jaiA64MovzX(JIT_SCRATCH_A, (unsigned)calleeVal.type, 0));
    emit(e, jaiA64StrW(JIT_SCRATCH_A, 31,
                       d + (unsigned)offsetof(JitCallDesc, callee)));
    emitConst64(e, JIT_SCRATCH_A, (int64_t)(uintptr_t)calleeVal.as.obj);
    emit(e, jaiA64StrX(JIT_SCRATCH_A, 31,
                       d + (unsigned)offsetof(JitCallDesc, callee) + 8));

    /* The arguments, which for an invoke begin with the receiver. */
    for (unsigned i = 0; i < nargs; i++) {
        unsigned idx = first + i;
        SlotKind k = e->stack[idx];
        if (k != SLOT_INT && k != SLOT_FLOAT && k != SLOT_BOOL &&
            k != SLOT_INST && k != SLOT_LIST) {
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
            e->localKind[slot] != SLOT_LIST) {
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

    emit(e, jaiA64MovzX(JIT_SCRATCH_A, nargs, 0));
    emit(e, jaiA64StrX(JIT_SCRATCH_A, 31,
                       d + (unsigned)offsetof(JitCallDesc, argc)));
    emit(e, jaiA64MovzX(JIT_SCRATCH_A, nroots, 0));
    emit(e, jaiA64StrX(JIT_SCRATCH_A, 31,
                       d + (unsigned)offsetof(JitCallDesc, nroots)));

    emit(e, jaiA64AddXImm(0, 31, d));
    emitConst64(e, JIT_SCRATCH_A, (int64_t)(uintptr_t)helper);
    emit(e, jaiA64Blr(JIT_SCRATCH_A));

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

static bool emitCallOut(Emit *e, unsigned argc) {
    ObjClass *cls = e->stackClass[e->depth - argc - 1];
    if (cls == NULL) { e->whyNot = "callee class"; return false; }

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

static bool compileBody(Emit *e, ObjClosure *closure) {
    ObjFunction *fn = closure->fn;
    const uint8_t *code = fn->chunk.code;
    int count = fn->chunk.count;

    int start = e->osr ? (int)e->osrTop : 0;
    int stop  = e->osr ? (int)e->osrEnd : count;
    for (int off = start; off < stop && !e->failed;) {
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
                                                   : NULL_VAL,
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
            if (!adoptLocalKind(e, slot, e->stack[e->depth - 1],
                                e->stackShape[e->depth - 1],
                                e->stackClass[e->depth - 1])) {
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
                if (!pushValue(e, SLOT_INT, 0, NULL)) return false;
                emitConst64(e, pushReg(e) - 1, AS_INT(k));
            } else if (IS_FLOAT(k)) {
                /* The bits, not the number: a float lives in an X register
                 * exactly as it does in a Value's payload. */
                double d = AS_FLOAT(k);
                int64_t bits;
                memcpy(&bits, &d, sizeof bits);
                if (!pushValue(e, SLOT_FLOAT, 0, NULL)) return false;
                emitConst64(e, pushReg(e) - 1, bits);
            } else {
                return false;
            }
            off += 4;
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

            if (!localObserved(e, slot)) return false;   /* see below */
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
            Value seen = e->observed[slot];
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
                                                       : NULL_VAL,
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
            /* Only an initializer, which yields the object it initialised --
             * that is what makes `Point(1, 2)` an expression. A plain function
             * returning null has no int64 to hand back. */
            if ((fn->flags & FN_INIT) == 0) return false;
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

            if (rk == SLOT_INST) {
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

        case OP_FOR_ITER_BIND: {
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
            localOut(e, slot, JIT_IDX_REG);
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
            if (!pushValue(e, kind, elemShape, elemClass)) return false;
            emit(e, jaiA64LdrX(pushReg(e) - 1, JIT_SCRATCH_C, 8));
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
            if (cls == NULL) return false;
            if (e->depth >= JIT_MAX_SAVED) return false;
            e->stackShape[e->depth] = cls->shapeId;
            e->stackClass[e->depth] = cls;
            e->stackSeen[e->depth]  = NULL_VAL;
            e->stackLocal[e->depth] = -1;
            e->stack[e->depth++]    = SLOT_CLASS;
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
            if (!isClassCallee(e, argc)) { e->whyNot = "tail callee not a class"; return false; }
            if (!emitCallOut(e, argc)) return false;
            unsigned r;
            SlotKind k;
            if (!popValue(e, &r, &k)) return false;
            if (e->sawReturn && e->returnKind != k) return false;
            e->sawReturn  = true;
            e->returnKind = k;
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
            unsigned r;
            SlotKind k;
            if (!popValue(e, &r, &k)) return false;
            /* One return kind per function: the entry point rebuilds a Value
             * from it, and it cannot rebuild two. */
            if (e->sawReturn && e->returnKind != k) return false;
            e->sawReturn = true;
            e->returnKind = k;
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
static bool adoptLocalKind(Emit *e, unsigned slot, SlotKind kind,
                           uint32_t shape, ObjClass *klass) {
    if (!e->localTyped[slot]) {
        e->localKind[slot]  = kind;
        e->localShape[slot] = shape;
        e->localClass[slot] = klass;
        e->localTyped[slot] = true;
        return true;
    }
    return e->localKind[slot] == kind && e->localShape[slot] == shape;
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

bool jaiJitCompileFunc(ObjClosure *closure, Value *slotBase) {
    ObjFunction *fn = closure->fn;
    if (!eligible(fn)) return false;

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
        frame += 8u * e.locals;
        frame = (frame + 7u) & ~7u;
    }
    if (e.callsOut) {
        e.descOffset = frame;
        frame += (unsigned)sizeof(JitCallDesc);
    }
    e.frameBytes = (frame + 15u) & ~15u;

    /* Prologue. */
    emit(&e, jaiA64StpPre(29, 30, 31, -(int32_t)e.frameBytes));
    emitSaveRestore(&e, true);
    /* The real arguments land in the local registers in order; the closure,
     * when there is one, is the last incoming register but lives just past the
     * locals, where closureReg expects it. Placing it by argument index
     * instead put it three registers low and the first upvalue read
     * dereferenced whatever was there. */
    unsigned realArgs = e.usesUpvalues ? argCount - 1u : argCount;
    for (unsigned i = 0; i < realArgs; i++) {
        if (e.spilled) {
            emit(&e, jaiA64StrX(i, 31, localFrameOff(&e, e.base + i)));
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
            emit(&e, jaiA64StrX(JIT_SCRATCH_C, 31,
                                localFrameOff(&e, e.base + i)));
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
                emit(&e, jaiA64LdrX(JIT_SCRATCH_C, 31,
                                    localFrameOff(&e, slot)));
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
            if (kind == SLOT_CLASS || kind == SLOT_SELF) {
                /* Neither holds a register; both are compile-time constants. */
                uintptr_t p = kind == SLOT_CLASS
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
    if (limit == 0) { jitFree(map, depths, fn->chunk.count + 1); return false; }
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
            if (target < 0) { jitFree(map, depths, fn->chunk.count + 1); return false; }
        } else if (f->targetOffset <= FIXUP_DEOPT &&
                   f->targetOffset > FIXUP_DEOPT - JIT_MAX_DEOPT) {
            target = e.deopt[FIXUP_DEOPT - f->targetOffset].stub;
            if (target < 0) { jitFree(map, depths, fn->chunk.count + 1); return false; }
        } else if (f->targetOffset <= FIXUP_OVF &&
                   f->targetOffset >= FIXUP_OVF - 2u) {
            target = e.overflowStub[FIXUP_OVF - f->targetOffset];
            if (target < 0) { jitFree(map, depths, fn->chunk.count + 1); return false; }
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
        fprintf(stderr, "[jit] compiled %s  arity=%u locals=%u insts=%u\n",
                fn->name ? fn->name->chars : "<anon>", e.arity, e.locals,
                e.count);
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
                       bool hasIter) {
    ObjFunction *fn = closure->fn;
    uint32_t end = findLoopEnd(&fn->chunk, top);
    if (end == 0 || end <= top) return false;
    if (fn->maxSlots < 1 || (unsigned)fn->maxSlots > 16) return false;

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
    e.observed = slots;
    e.offsetToInst = map;
    e.offsetToDepth = depths;
    e.savedCount = JIT_MAX_SAVED;

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

    emit(&e, jaiA64StpPre(29, 30, 31, -(int32_t)e.frameBytes));
    emitSaveRestore(&e, true);
    emit(&e, jaiA64MovX(JIT_SLOTS_REG, 0));
    if (hasIter) {
        emit(&e, jaiA64MovX(JIT_ITER_REG, 1));
        emit(&e, jaiA64LdrX(JIT_IDX_REG, JIT_ITER_REG,
                            (unsigned)offsetof(ObjIter, index)));
        emit(&e, jaiA64LdrX(JIT_LIM_REG, JIT_ITER_REG,
                            (unsigned)offsetof(ObjIter, limit)));
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

#define OSR_SYNC_ITER()                                                        \
    do {                                                                       \
        if (hasIter) {                                                         \
            emit(&e, jaiA64StrX(JIT_IDX_REG, JIT_ITER_REG,                     \
                                (unsigned)offsetof(ObjIter, index)));          \
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
            if (kind == SLOT_CLASS || kind == SLOT_SELF) {
                uintptr_t pv = kind == SLOT_CLASS
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
        if (r->start != 0 || r->step != 1) return 0;
    }

    if (fn->osrCode == NULL) {
        if (fn->osrRefused) return 0;
        fn->osrRefused = true;
        if (!compileOsr(closure, top, frame->slots, hasIter)) return 0;
        fn->osrRefused = false;
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
        slotBase[0] = OBJ_VAL((Obj *)(uintptr_t)r.value);
        break;
    case SLOT_BOOL:
        slotBase[0] = BOOL_VAL(r.value != 0);
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
