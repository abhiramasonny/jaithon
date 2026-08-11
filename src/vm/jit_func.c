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
/* For jaiOpBranchOperandAt: the one list of which opcodes carry a code address,
 * which is what says whether an offset can be reached by anything but the
 * fall-through. */
#include "verify.h"
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
/* Entries the operand-stack MODEL can hold, which is no longer the same as the
 * number of registers: an inlined body's entries live in a bank of their own,
 * so the model has to describe more values than x19..x28 could hold. */
#define JIT_MAX_STACK   20u
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
typedef struct JitCallDesc {
    /* `link` and `nroots` come first so that `roots` sits at a fixed offset
     * from the head of a chain the collector can walk. A descriptor whose
     * `link` is non-NULL is on that chain. */
    struct JitCallDesc *link;
    int64_t nroots;
    Value   roots[JIT_MAX_SAVED];
    Value   callee;
    Value   args[JIT_MAX_ARGS_OUT];
    Value   result;
    int64_t argc;
    /* Whatever else a helper needs that is not a Value. Only OP_GET_SLICE uses
     * it, for the flags byte saying which of start, stop and step are present:
     * presence cannot be read off the values, because `xs[null:3]` is a
     * TypeError and must not read as `xs[:3]`. */
    int64_t aux;
} JitCallDesc;

/* Compiled frames whose roots the collector must see.
 *
 * A call through the descriptor hands its roots to a C helper, which pushes
 * them. A *self*-call does not: it is a bare `bl` to the prologue, so whatever
 * the caller holds in callee-saved registers is invisible to a collection that
 * runs inside the callee. Today the tier refuses that shape -- a body that
 * allocates and then self-calls declines with "a bail follows a heap write" --
 * but any operation that allocates without setting that flag would reach it,
 * `OP_GET_SLICE` in a recursive `sort` being the first. The emitted code links
 * its descriptor on here before the `bl` and unlinks after. */
static JitCallDesc *gJitFrames;

void jaiJitMarkFrames(void) {
    for (JitCallDesc *f = gJitFrames; f != NULL; f = f->link) {
        for (int64_t i = 0; i < f->nroots; i++) jaiGCMarkValue(f->roots[i]);
    }
}

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
    Value   stack[JIT_MAX_STACK + 1];
} JitDeoptRecord;

static JitDeoptRecord gDeopt;

/* Cached, because this sits on the deopt path and a deopt is not rare -- a
 * guard that misses once per loop iteration reaches this every time. getenv
 * walks the whole environment, so leaving it uncached made every measurement a
 * function of how many variables happened to be exported: sort_merge moved
 * 70ms to 100ms on shell padding alone. Same idiom as jaiJitEnabled. */
static bool jitReconTrace(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAI_JIT_RECON");
        cached = (v != NULL && v[0] != '\0') ? 1 : 0;
    }
    return cached != 0;
}

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
    if (jitReconTrace()) {
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

/* Make room for one more element in `list`, and nothing else.
 *
 * No descriptor and no roots, which is what makes the call site three
 * instructions instead of thirty. That is sound because growing a list cannot
 * collect: `jaiListReserve` reaches `jaiRealloc`, and gc.c is explicit that
 * the marker never runs from inside it -- collections happen only at the
 * safepoints that call jaiGCMaybeCollect. Nothing this body is holding in a
 * callee-saved register can be freed across this call, so nothing has to be
 * made visible to the collector first.
 *
 * The element about to be stored travels anyway, as a tag and a payload, so
 * this does not depend on that invariant for the one value whose only
 * reference really is a register.
 *
 * Returns 1 when it raised -- a list past INT32_MAX is the only way. */
static int jitListGrow(ObjList *list, uint64_t tag, int64_t payload) {
    Value pending;
    pending.type = (ValueType)tag;
    pending.as.integer = payload;
    jaiGCPushRoot(OBJ_VAL(list));
    jaiGCPushRoot(pending);
    if (list->capacity > INT32_MAX / 2) {
        jaiGCPopRoots(2);
        (void)jaiThrow(vm.cRuntimeError,
                       "list cannot grow beyond %d items", INT32_MAX);
        return 1;
    }
    jaiListReserve(list, JAI_GROW_CAP(list->capacity));
    jaiGCPopRoots(2);
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

static int jitGetSlice(JitCallDesc *d) {
    for (int64_t i = 0; i < d->nroots; i++) jaiGCPushRoot(d->roots[i]);
    uint8_t flags = (uint8_t)d->aux;
    bool hasStart = (flags & 1) != 0, hasStop = (flags & 2) != 0,
         hasStep = (flags & 4) != 0;
    int at = 1;
    Value startV = hasStart ? d->args[at++] : NULL_VAL;
    Value stopV  = hasStop  ? d->args[at++] : NULL_VAL;
    Value stepV  = hasStep  ? d->args[at++] : NULL_VAL;
    bool ok = jaiSliceGet(d->args[0], startV, stopV, stepV,
                          hasStart, hasStop, hasStep, &d->result);
    jaiGCPopRoots((int)d->nroots);
    return ok ? 0 : 1;
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
    /* An instance of a fixed class, or null, held as the pointer or as zero --
     * which is how C would spell it, and makes `x == null` a compare against
     * zero. `T?` is how this language writes optional, so refusing it stopped
     * six hundred stdlib bodies. The one thing it costs is that the tag stops
     * being a property of the kind: materialising one picks VAL_NULL or
     * VAL_OBJ off the register at run time. */
    SLOT_MAYBE_INST,
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

/* ...and that measured 3.7x. The recurrence `x2 = x*x; y2 = y*y; xy = x*y;
 * y = 2*xy + y0; x = x2 - y2 + x0` runs at 3.3ns an iteration with its doubles
 * in d registers and 12.1ns with them in X registers and an fmov pair per
 * operation -- worse, in fact, than leaving them in memory and loading them
 * straight into d registers, which is 8.5ns. The three instructions are not
 * three instructions: each one is a cross-file move on the dependency chain.
 *
 * So a SLOT_FLOAT operand-stack entry may now *live* in an FP register between
 * the instruction that computes it and the one that consumes it. Entry i's
 * canonical home is still x(19 + regBase + i); `fpLive` bit i says that copy is
 * stale and v(16 + i) holds the value instead. The bank is v16..v25, which is
 * caller-saved on purpose: nothing here is allowed to survive a call, so there
 * is nothing for the prologue to preserve and no save set to get wrong. The
 * index is shared with the X bank, so two live entries can never name the same
 * FP register. */
#define JIT_FP_BANK 16u

/* ...and the same argument applies to a float LOCAL, which is why there is a
 * second callee-saved bank. v8..v15 are preserved across a call by the
 * platform ABI, nothing else this tier emits names them (the operand stack's
 * bank starts at v16 and the scratches are v0/v1), and only fmov, ldr d and
 * str d ever touch a local's home -- never arithmetic. So a slot parked here
 * is a bit-exact 64-bit home: a slot whose kind the allocator guessed wrong
 * costs an fmov, it cannot be read back as the wrong bits. */
#define JIT_FP_FIRST_SAVED 8u
#define JIT_FP_MAX_SAVED   8u

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
/* Self-calls and direct calls to a callee that writes share this table, so it
 * is sized for a body with several of each rather than for recursion alone. */
#define JIT_MAX_SELF_SLOW 32u
#define FIXUP_SELFSLOW (UINT32_MAX - 200u) /* minus the self-call index */
#define JIT_MAX_GROW   16u
#define FIXUP_GROW   (UINT32_MAX - 400u) /* minus the list-growth index */

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

    SlotKind  stack[JIT_MAX_STACK];
    uint32_t  stackShape[JIT_MAX_STACK];  /* class shapeId for SLOT_INST */
    ObjClass *stackClass[JIT_MAX_STACK];
    Value     stackSeen[JIT_MAX_STACK];   /* the live value, for field feedback */
    int       stackLocal[JIT_MAX_STACK];  /* which local it was copied from, or -1 */

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
    uint8_t   iterKind;   /* 1 a unit-step range, 2 a list */
    Value     elemSample; /* a live element, for a list head */
    /* Where each slot lives, per slot. The entry already checks every slot's
     * kind before calling, so the prologue only has to load payloads; every
     * way out writes back the ones that took a register.
     *
     * This used to be one flag for the whole loop, and the test it turned on
     * was `reserved + every slot in the enclosing function's frame window +
     * the deepest expression <= ten`. No loop in a real function passes that,
     * so every one of them ran with all of its locals in memory as full
     * Values -- mandelbrot's recurrence measured 8.7ns an iteration against
     * the 8.5 the hand-written memory kernel gets and the 3.3 the register one
     * does. Three things together fix it and none of them alone does:
     *
     *   - only slots the body actually NAMES take a register. A function
     *     declares a wide frame and a loop inside it touches a third of it.
     *   - a float slot takes one of v8..v15 (see JIT_FP_FIRST_SAVED) rather
     *     than an X register, which both keeps it out of the X budget and
     *     puts it where the arithmetic wants it.
     *   - what is left over stays in the frame exactly as before, so a body
     *     one slot over budget loses one slot rather than all of them. The
     *     busiest slots win, counted in the measuring pass and weighted by
     *     how deeply nested the loop naming them is. */
    uint8_t   slotXReg[JIT_MAX_SLOTS + 1];   /* x19..x28, or 0 for none */
    uint8_t   slotFpReg[JIT_MAX_SLOTS + 1];  /* d8..d15, or 0 for none */
    unsigned  xLocals;                       /* how many took an X register */
    unsigned  fpLocals;                      /* ...and how many a d register */
    uint32_t  slotUse[JIT_MAX_SLOTS + 1];    /* sites, weighted by loop depth */
    const uint8_t *loopDepth;                /* per bytecode offset */
    unsigned  loopDepthCount;
    unsigned  fpSaveOffset;                  /* where d8.. are preserved */
    /* Where a range loop parks the ObjIter, since it holds no register for it.
     * Written once in the prologue and read once in each exit stub. */
    unsigned  iterFrameOffset;
    /* Inlining a method widens the live range of everything it reads, so a
     * loop that fitted the registers as a call may not fit as an expression.
     * The compile is retried with this set when that is what went wrong. */
    bool      noInline;
    bool      inlined;
    /* A callee whose body is being emitted where the call to it was.
     *
     * Its locals are operand-stack entries of the CALLER's frame: slots 1..n
     * are the argument entries that are already sitting there, and anything it
     * binds pins one more. Nothing is copied and no frame appears, which is
     * the whole point -- but it also means the interpreter has no idea any of
     * this is happening, so every guard inside the inlined body deoptimises to
     * `inlIp`, the caller's own OP_CALL, with the model as it stood at
     * `inlDepth`: the callee and its arguments, untouched, exactly what the
     * interpreter expects to find at that offset. */
    bool      inlining;
    unsigned  inlDepth;
    unsigned  inlPinned;      /* how many of its locals it has bound so far */
    unsigned  inlValueBase;   /* value entries at or above this are the callee's */
    uint32_t  inlIp;
    int       inlSlot[JIT_MAX_SLOTS + 1];
    /* A local whose kind is not the same on every path into some point. It
     * lives in the frame with its tag and every read of it guards, which is
     * what lets two paths disagree. The compiler reuses one slot for the
     * induction variables of loops that do not overlap, so this is not exotic:
     * it is nbody's advance. */
    bool      dynamicLocal[JIT_MAX_SLOTS + 1];
    bool      needDynamic[JIT_MAX_SLOTS + 1];   /* found during this attempt */
    /* A slot that holds an instance on one path and null on another. Unlike a
     * dynamic local it stays in a register -- the pointer, or zero -- and its
     * tag is built when it is materialised. Per slot, because widening every
     * instance parameter of any body that merely mentions null cost
     * object_dispatch a factor of two. */
    bool      nullableLocal[JIT_MAX_SLOTS + 1];
    bool      needNullable[JIT_MAX_SLOTS + 1];
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
    /* Module globals baked into this body. Every global this body touches
     * lives in one table -- the defining module's -- so one guard covers all
     * of them: `keyVersion` changes only when a live entry's ADDRESS or KEY
     * could have changed (a new key, a rehash, a delete, a clear), which is
     * exactly the condition that makes a baked JaiEntry* unusable.
     * ObjModule::version, which the function tier's entry check uses, is
     * neither necessary here (a value write moves no entry) nor sufficient
     * (under jaiModuleSet's narrowed rule an inert write does not move it). */
    JaiTable *globalsTable;
    uint32_t  globalsKeyVersion;
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
        SlotKind kinds[JIT_MAX_STACK + 1];
        ObjClass *classes[JIT_MAX_STACK + 1];
        /* The topmost entry is the result of a call that already happened, so
         * it lives in the descriptor rather than a register. */
        bool     lastFromDesc;
        uint32_t fpLive;
        int      stub;
    } deopt[JIT_MAX_DEOPT];
    unsigned  deoptCount;
    /* What a self-call does when the callee's verdict is not zero. Emitted
     * with the stubs rather than inline: the fast path is three instructions
     * and a not-taken branch, and putting thirty instructions of cold code
     * between two recursive call sites cost fib_recursive 25% -- measured,
     * with the same code laid out both ways. */
    struct {
        int      returnTo;   /* instruction index to resume the body at */
        int      stub;
        unsigned roots;      /* the descriptor is on the frame chain when >0 */
        unsigned deoptBail;  /* record for verdict 1: re-execute the call */
        unsigned deoptKind;  /* record for a result of an unexpected kind */
        unsigned tag;        /* the tag the compiled body was built for */
        unsigned resultReg;
        /* Which closure the interpreter finishes on verdict 4. NULL means
         * this one -- a self-call knows its own. A direct call to another
         * compiled function that writes names it here instead. */
        ObjClosure *callee;
        /* What the continuation's object must be, when the fast path's kind
         * says more than "a heap object". A shape alone is not enough to
         * check: VAL_OBJ is EVERY heap object, and reading `klass` off an
         * ObjString answers wrongly rather than faulting -- the same mistake
         * the list-element head made. So the type is checked first. */
        int      retType;    /* an ObjType, or -1 for no check */
        uint32_t retShape;   /* class shapeId, or 0 for no check */
    } selfSlow[JIT_MAX_SELF_SLOW];
    unsigned  selfSlowCount;
    /* A `xs.push(v)` whose list is full. Out of line for the same reason the
     * self-call's cold half is: the store itself is nine instructions, and a
     * realloc call sitting between them would be most of the loop.
     *
     * Before this existed the full case was a deopt, and that is worth
     * spelling out because it is what made the function tier nearly worthless
     * for any body that builds a list: a deopt hands the WHOLE REST of the
     * function to the interpreter, and `out` starts empty, so the very first
     * push abandoned the compiled body. merge in sort_merge deopted once per
     * call -- 62434 times for 62500 items -- and ran interpreted from its
     * first push onward every time. */
    struct {
        int      returnTo;   /* instruction index to resume the body at */
        int      stub;
        unsigned listReg;
        unsigned valReg;
        unsigned tag;
        unsigned countReg;   /* the scratch the reload must refill */
    } grow[JIT_MAX_GROW];
    unsigned  growCount;
    uint32_t  curOffset;
    /* The model as it stood at the start of `curOffset`, before any of that
     * instruction's own pushes. A guard fires part-way through an instruction
     * and the interpreter resumes at its start, so this is what it is holding
     * there -- see deoptSite. */
    unsigned  instDepth;
    unsigned  instValueDepth;
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

    /* Value entries whose payload is in v(16 + index) rather than in their X
     * register. See JIT_FP_BANK. */
    uint32_t  fpLive;
    bool      fpOff;         /* the retry that puts every float back in X */
    /* Offsets this walk carried an FP-resident value *into*. A branch that
     * lands on one of them would arrive with the value only in X, so the
     * compile is declined and retried with fpOff. Straight-line float
     * expressions are never branch targets, so this is a safety net rather
     * than a path; JAI_JIT_WHY says so when it fires. */
    uint32_t  fpCarry[64];
    unsigned  fpCarryCount;
    /* Value entries that are a plain read of a float local and are being held
     * in that LOCAL's own d register rather than copied into the bank. `x * x`
     * was two fmovs and a multiply; borrowing makes it the multiply. The
     * borrow is read-only, so the whole obligation is that nothing writes the
     * register underneath it -- and the only two places that ever write a
     * local's d home are localOut and localOutFp, both of which release first.
     * Cleared on push, on pop, on claim, and before any deopt record, so an
     * entry that survives one of those is back in the bank where the rest of
     * this file expects it. */
    uint32_t  fpBorrow;
    uint8_t   fpBorrowReg[32];
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
#define JIT_IDX_REG   (JIT_FIRST_SAVED + 1u)
#define JIT_LIM_REG   (JIT_FIRST_SAVED + 2u)
#define JIT_START_REG (JIT_FIRST_SAVED + 3u)   /* the range's first value */
#define JIT_ITER_REG  (JIT_FIRST_SAVED + 4u)   /* the ObjIter itself */

/* Registers OSR keeps for itself: the slots pointer, and for a range loop the
 * iterator, its index and its limit.
 *
 * A range loop does not keep the ObjIter in a register. Everything the head
 * needs is read out of it once in the prologue -- index, limit, and the
 * range's start -- and the only later use is writing the index back on the way
 * out, which happens once per exit stub and never in the loop. So it lives in
 * the frame instead, and the operand stack gets a sixth register. That is
 * exactly the amount that was missing: every "more live values" decline in the
 * benchmark suite wanted six and had five. JIT_ITER_REG is therefore the last
 * of the reserved block, so dropping it leaves the rest contiguous. */
static unsigned osrReserved(const Emit *e) {
    if (!e->hasIter) return 1u;
    return e->iterKind == 1 ? 4u : 5u;
}

static unsigned regBase(const Emit *e) {
    if (e->osr) return osrReserved(e) + e->xLocals;
    return e->spilled ? 0u : e->locals;
}

static unsigned localReg(const Emit *e, unsigned slot) {
    if (e->osr) return e->slotXReg[slot];
    return JIT_FIRST_SAVED + (slot - e->base);
}

/* Sixteen bytes each, not eight: the tag travels with the value so a local
 * whose kind varies can be read behind a guard. */
static unsigned localFrameOff(const Emit *e, unsigned slot) {
    return e->localsFrameOffset + (slot - e->base) * 16u;
}

static void branchOnDeopt(Emit *e, unsigned cond);
static void fpReleaseHome(Emit *e, unsigned reg);

/* Put the tag for `kind` in `tagReg`. Every kind but SLOT_MAYBE_INST has one
 * fixed at compile time; that one reads it off the payload. */
static void emitTagFor(Emit *e, SlotKind kind, unsigned payloadReg,
                       unsigned tagReg, unsigned spare) {
    if (kind != SLOT_MAYBE_INST) {
        unsigned tag = kind == SLOT_INT    ? VAL_INT
                     : kind == SLOT_FLOAT  ? VAL_FLOAT
                     : kind == SLOT_BOOL   ? VAL_BOOL
                     : kind == SLOT_OPAQUE ? VAL_NULL
                                           : VAL_OBJ;
        emit(e, jaiA64MovzX(tagReg, tag, 0));
        return;
    }
    emit(e, jaiA64SubsXImm(31, payloadReg, 0));
    emit(e, jaiA64MovzX(tagReg, VAL_OBJ, 0));
    emit(e, jaiA64MovzX(spare, VAL_NULL, 0));
    emit(e, jaiA64CselX(tagReg, spare, tagReg, JAI_A64_EQ));
}

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
    if (e->osr) {
        if (e->slotXReg[slot] != 0) return e->slotXReg[slot];
        if (e->slotFpReg[slot] != 0) {
            emit(e, jaiA64FmovXD(scratch, e->slotFpReg[slot]));
            return scratch;
        }
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
    if (e->osr) {
        return e->slotXReg[slot] != 0 ? e->slotXReg[slot] : JIT_SCRATCH_C;
    }
    if (e->spilled) return JIT_SCRATCH_C;
    return localReg(e, slot);
}

static void localOut(Emit *e, unsigned slot, unsigned src) {
    if (e->osr) {
        if (e->slotFpReg[slot] != 0) fpReleaseHome(e, e->slotFpReg[slot]);
        if (e->slotXReg[slot] != 0) {
            if (src != e->slotXReg[slot]) {
                emit(e, jaiA64MovX(e->slotXReg[slot], src));
            }
            return;
        }
        if (e->slotFpReg[slot] != 0) {
            emit(e, jaiA64FmovDX(e->slotFpReg[slot], src));
            return;
        }
        /* Tag as well as payload: writing straight through is what lets a
         * deopt here cost nothing. The kind is fixed for the compile. */
        SlotKind k = e->localKind[slot];
        emitTagFor(e, k, src, JIT_SCRATCH_D, JIT_SCRATCH_C);
        emit(e, jaiA64StrW(JIT_SCRATCH_D, JIT_SLOTS_REG, slot * 16u));
        emit(e, jaiA64StrX(src, JIT_SLOTS_REG, slot * 16u + 8u));
        return;
    }
    if (!e->spilled) {
        if (src != localReg(e, slot)) emit(e, jaiA64MovX(localReg(e, slot), src));
        return;
    }
    emitTagFor(e, e->localKind[slot], src, JIT_SCRATCH_D, JIT_SCRATCH_C);
    emit(e, jaiA64StrW(JIT_SCRATCH_D, 31, localFrameOff(e, slot)));
    emit(e, jaiA64StrX(src, 31, localFrameOff(e, slot) + 8));
}

/* The float half of localIn/localOut. A float local reached through the FP
 * bank never visits an X register at all: in memory mode that turns
 * `ldr x; fmov d, x` into one `ldr d` and `fmov x, d; str x` into one `str d`,
 * which is the difference between 14.5ns and 8.5ns an iteration on the
 * mandelbrot recurrence. Only for a local whose kind is fixed -- a dynamic one
 * has a tag to check and goes the ordinary way. */
static void localInFp(Emit *e, unsigned slot, unsigned dst) {
    if (e->osr) {
        if (e->slotFpReg[slot] != 0) {
            /* Both banks, so the numbers never collide: a local's home is
             * v8..v15 and the operand stack's is v16 up. */
            if (dst != e->slotFpReg[slot]) {
                emit(e, jaiA64FmovDD(dst, e->slotFpReg[slot]));
            }
            return;
        }
        if (e->slotXReg[slot] != 0) {
            emit(e, jaiA64FmovDX(dst, e->slotXReg[slot]));
            return;
        }
        emit(e, jaiA64LdrD(dst, JIT_SLOTS_REG, slot * 16u + 8u));
        return;
    }
    if (!e->spilled) {
        emit(e, jaiA64FmovDX(dst, localReg(e, slot)));
        return;
    }
    emit(e, jaiA64LdrD(dst, 31, localFrameOff(e, slot) + 8));
}

static void localOutFp(Emit *e, unsigned slot, unsigned src) {
    if (e->osr) {
        if (e->slotFpReg[slot] != 0) {
            fpReleaseHome(e, e->slotFpReg[slot]);
            if (src != e->slotFpReg[slot]) {
                emit(e, jaiA64FmovDD(e->slotFpReg[slot], src));
            }
            return;
        }
        if (e->slotXReg[slot] != 0) {
            emit(e, jaiA64FmovXD(e->slotXReg[slot], src));
            return;
        }
        emit(e, jaiA64MovzX(JIT_SCRATCH_D, VAL_FLOAT, 0));
        emit(e, jaiA64StrW(JIT_SCRATCH_D, JIT_SLOTS_REG, slot * 16u));
        emit(e, jaiA64StrD(src, JIT_SLOTS_REG, slot * 16u + 8u));
        return;
    }
    if (!e->spilled) {
        emit(e, jaiA64FmovXD(localReg(e, slot), src));
        return;
    }
    emit(e, jaiA64MovzX(JIT_SCRATCH_D, VAL_FLOAT, 0));
    emit(e, jaiA64StrW(JIT_SCRATCH_D, 31, localFrameOff(e, slot)));
    emit(e, jaiA64StrD(src, 31, localFrameOff(e, slot) + 8));
}

/* Slots the body may name at all. */
static bool localInRange(Emit *e, unsigned slot) {
    if (e->osr) {
        if (slot >= e->locals) return false;
        if (slot > e->maxSlotUsed) e->maxSlotUsed = slot;
        /* Which slots deserve the registers. A site inside a nested loop is
         * worth four of one outside it -- the alternative, counting sites
         * flat, gives `width` and `height` the same claim as the recurrence
         * variable they are read once per row to set up. */
        if (e->measuring && !e->inlining && e->loopDepth != NULL &&
            e->curOffset < e->loopDepthCount && slot <= JIT_MAX_SLOTS) {
            unsigned d = e->loopDepth[e->curOffset];
            if (d > 6u) d = 6u;
            e->slotUse[slot] += 1u << (2u * d);
        }
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

/* The X register value entry `idx` belongs in, counting from the bottom of the
 * operand stack rather than from the top the way pushReg does.
 *
 * An inlined body gets a bank of its own. It cannot call anything -- that is
 * the first thing inlinableBody checks -- so every register a call would
 * destroy is free for the whole of it, and its temporaries cost the caller no
 * callee-saved register at all. That is not a refinement: `evalA` inlined into
 * spectral's inner loop wants eight live values where the OSR form had six
 * left, so without this it does not fit and is not inlined. x9..x12 stay out
 * of it, being the emitter's own scratches. */
#define JIT_INL_BANK   0u    /* x0..x8, all caller-saved */
#define JIT_INL_COUNT  9u

static unsigned valueXReg(const Emit *e, unsigned idx) {
    if (e->inlining && idx >= e->inlValueBase) {
        return JIT_INL_BANK + (idx - e->inlValueBase);
    }
    return JIT_FIRST_SAVED + regBase(e) + (e->usesUpvalues ? 1u : 0u) + idx;
}

/* One past the top entry's register. Every `pushReg(e) - 1` in this file means
 * "the entry just pushed", and expressing it this way rather than from the
 * bottom is what keeps that idiom true across the two banks. */
static unsigned pushReg(const Emit *e) {
    if (e->valueDepth == 0) return valueXReg(e, 0);
    return valueXReg(e, e->valueDepth - 1) + 1;
}

/* v16.. for the ordinary bank, v2..v7 for an inlined body -- caller-saved, and
 * far enough from v16+JIT_MAX_SAVED, which the local-add path uses as a temp. */
#define JIT_INL_FP_BANK 2u

static unsigned fpRegAt(const Emit *e, unsigned idx) {
    if (e->inlining && idx >= e->inlValueBase) {
        return JIT_INL_FP_BANK + (idx - e->inlValueBase);
    }
    return JIT_FP_BANK + idx;
}

/* The d register entry `idx` is actually in: its own, or the local's it
 * borrowed. Every READ of a live FP entry goes through this; the writes keep
 * naming fpRegAt, which is what a borrow is released back into. */
static unsigned fpHeldIn(const Emit *e, unsigned idx) {
    if (e->fpBorrow & (1u << idx)) return e->fpBorrowReg[idx];
    return fpRegAt(e, idx);
}

/* Put entry `idx` back where every other part of this file expects it. */
static void fpSyncOne(Emit *e, unsigned idx) {
    if (!(e->fpLive & (1u << idx))) return;
    unsigned d = fpHeldIn(e, idx);
    e->fpLive   &= ~(1u << idx);
    e->fpBorrow &= ~(1u << idx);
    emit(e, jaiA64FmovXD(valueXReg(e, idx), d));
}

/* A local's d register is about to be written, so every entry borrowing it
 * takes a copy of its own first. */
static void fpReleaseHome(Emit *e, unsigned reg) {
    if (e->fpBorrow == 0 || reg == 0) return;
    for (unsigned i = 0; i < 32u; i++) {
        if ((e->fpBorrow & (1u << i)) == 0) continue;
        if (e->fpBorrowReg[i] != reg) continue;
        e->fpBorrow &= ~(1u << i);
        emit(e, jaiA64FmovDD(fpRegAt(e, i), reg));
    }
}

/* ...and the same for all of them, before anything records where the
 * interpreter should resume: the stub writes entries out of fpRegAt. */
static void fpReleaseAll(Emit *e) {
    if (e->fpBorrow == 0) return;
    for (unsigned i = 0; i < 32u; i++) {
        if ((e->fpBorrow & (1u << i)) == 0) continue;
        e->fpBorrow &= ~(1u << i);
        emit(e, jaiA64FmovDD(fpRegAt(e, i), e->fpBorrowReg[i]));
    }
}

static void fpSyncAll(Emit *e) {
    if (e->fpLive == 0) return;
    for (unsigned i = 0; i < 32u; i++) fpSyncOne(e, i);
}

/* A d register holding entry `idx`, loaded from its X register if that is
 * where it still is. Leaves fpLive alone: after this the two copies agree, and
 * claiming the X one is stale when it is not would cost a sync for nothing. */
static unsigned fpOperand(Emit *e, unsigned idx) {
    if (e->fpBorrow & (1u << idx)) return e->fpBorrowReg[idx];
    unsigned d = fpRegAt(e, idx);
    if (!(e->fpLive & (1u << idx))) {
        emit(e, jaiA64FmovDX(d, valueXReg(e, idx)));
    }
    return d;
}

/* Entry `idx` has just been computed into v(16 + idx); its X register is now
 * stale until something asks for it. */
static void fpClaim(Emit *e, unsigned idx) {
    e->fpLive |= 1u << idx;
    e->fpBorrow &= ~(1u << idx);
}

/* Entry `idx` is a read of a float local that lives in `reg`, and is held
 * there rather than copied. */
static void fpBorrowLocal(Emit *e, unsigned idx, unsigned reg) {
    e->fpLive |= 1u << idx;
    e->fpBorrow |= 1u << idx;
    e->fpBorrowReg[idx] = (uint8_t)reg;
}

static bool pushValue3(Emit *e, SlotKind kind, uint32_t shape, ObjClass *klass,
                       Value seen, int fromLocal) {
    if (e->depth >= JIT_MAX_STACK) {
        e->whyNot = "the operand stack is deeper than the model allows";
        return false;
    }
    if (!e->measuring && e->inlining &&
        e->valueDepth + 1u - e->inlValueBase > JIT_INL_COUNT) {
        e->whyNot = "an inlined body wants more registers than a call leaves free";
        return false;
    }
    if (!e->measuring && !e->inlining &&
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
    e->fpLive   &= ~(1u << e->valueDepth);
    e->fpBorrow &= ~(1u << e->valueDepth);
    e->valueDepth++;
    /* An inlined body's entries are not in the caller's bank, so they do not
     * widen its save set -- which is the whole reason they fit. */
    if (e->valueDepth > e->maxValue &&
        !(e->inlining && e->valueDepth > e->inlValueBase)) {
        e->maxValue = e->valueDepth;
    }
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
    if (e->depth >= JIT_MAX_STACK) return false;
    e->stack[e->depth++] = SLOT_SELF;
    return true;
}

/* Pop a value entry, reporting the register it was in and what it held. */
/* Pop an entry that has already been read out of its FP register, or that was
 * never in one. Only the float paths may call this; everything else goes
 * through popValue, which materialises first. */
static bool popValueRaw(Emit *e, unsigned *reg, SlotKind *kind) {
    if (e->depth == 0 || !holdsRegister(e->stack[e->depth - 1])) return false;
    e->depth--;
    e->valueDepth--;
    e->fpLive   &= ~(1u << e->valueDepth);
    e->fpBorrow &= ~(1u << e->valueDepth);
    if (kind != NULL) *kind = e->stack[e->depth];
    *reg = valueXReg(e, e->valueDepth);
    return true;
}

static bool popValue(Emit *e, unsigned *reg, SlotKind *kind) {
    if (e->depth == 0 || !holdsRegister(e->stack[e->depth - 1])) return false;
    fpSyncOne(e, e->valueDepth - 1);
    return popValueRaw(e, reg, kind);
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
 * locals and a call descriptor come to 528.
 *
 * The field is signed, so the pre-index going in reaches -512 but the post-index
 * coming out reaches only +504: `imm / 8` is 64 for 512, and 64 read back as a
 * signed seven-bit field is -64. A 512-byte frame therefore passed the test on
 * the way in and truncated on the way out, emitting `ldp x29, x30, [sp], #-512`.
 * That restores the right x30 and returns correctly, having moved the stack
 * pointer a kilobyte the wrong way, so the crash lands somewhere else entirely
 * in a caller whose frame is gone. Both ends ask the same question here so they
 * cannot disagree; whether any body lands on exactly 512 is a matter of luck. */
static bool framePairFits(const Emit *e) { return e->frameBytes <= 504u; }

static void emitFrameEnter(Emit *e) {
    if (framePairFits(e)) {
        emit(e, jaiA64StpPre(29, 30, 31, -(int32_t)e->frameBytes));
        return;
    }
    emit(e, jaiA64SubXImm(31, 31, e->frameBytes));
    emit(e, jaiA64StpOff(29, 30, 31, 0));
}

static void emitFrameLeave(Emit *e) {
    if (framePairFits(e)) {
        emit(e, jaiA64LdpPost(29, 30, 31, (int32_t)e->frameBytes));
        return;
    }
    emit(e, jaiA64LdpOff(29, 30, 31, 0));
    emit(e, jaiA64AddXImm(31, 31, e->frameBytes));
}

/* The float half of the save set. v8..v15 are callee-saved only in their low
 * 64 bits, which is exactly what a double is, so `str d`/`ldr d` is the whole
 * protocol. Two instructions each rather than a paired form because there is
 * no STP for FP in this encoder and this runs once per entry and once per way
 * out, never in a loop. */
static void emitFpSaveRestore(Emit *e, bool save) {
    for (unsigned i = 0; i < e->fpLocals; i++) {
        unsigned r = JIT_FP_FIRST_SAVED + i;
        unsigned at = e->fpSaveOffset + 8u * i;
        emit(e, save ? jaiA64StrD(r, 31, at) : jaiA64LdrD(r, 31, at));
    }
}

static void emitEpilogue(Emit *e, unsigned bailed) {
    emit(e, jaiA64MovzX(1, bailed, 0));
    emitFpSaveRestore(e, false);
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
    /* A join has to agree about where every value is, so nothing crosses a
     * branch in an FP register. fmov does not touch NZCV, so this is still
     * safe after the compare that set the condition. */
    fpSyncAll(e);
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

/* Where a guard here has to send the interpreter, and what it must be holding.
 *
 * Inside an inlined body neither is the model's current state: the call has
 * not happened as far as the interpreter is concerned, so it resumes at the
 * OP_CALL holding the callee and its arguments, and every entry the inlined
 * body has pushed above them -- its own locals and temporaries -- is not part
 * of the picture.
 *
 * And outside one it is still not the model's current state, because a guard
 * sits PART-WAY THROUGH an instruction. `OP_GET_LOCAL2` pushes its operand and
 * then guards that operand's tag, so by the time the guard is written the model
 * is two entries deeper than the interpreter's stack is at that offset. The
 * interpreter resumes at the instruction's start and pushes those entries
 * itself; handing them to it as well leaves them stranded underneath, and a
 * `for` whose iterator is two entries down reads the loop variable as its
 * iterator. `instDepth` is the model at the instruction's start, which is
 * exactly what the interpreter holds there.
 *
 * Only entries this instruction pushed can be dropped and they are the topmost
 * ones, so every entry that remains keeps the register it was assigned. The
 * other direction -- an instruction that has already popped an entry the
 * interpreter still holds -- cannot be repaired by trimming and is refused. */
static bool deoptSite(Emit *e, uint32_t ip, uint32_t *ipOut,
                      unsigned *depthOut, unsigned *valueDepthOut) {
    if (e->inlining) {
        *ipOut = e->inlIp;
        *depthOut = e->inlDepth;
        unsigned seen = 0;
        for (unsigned i = 0; i < e->inlDepth; i++) {
            if (holdsRegister(e->stack[i])) seen++;
        }
        *valueDepthOut = seen;
        return true;
    }
    *ipOut = ip;
    *depthOut = e->depth;
    *valueDepthOut = e->valueDepth;
    /* Only for a guard that resumes at the instruction being compiled. A guard
     * that names a later offset -- the one after a call, where the result is
     * already on the stack -- is describing a point this walk has not reached
     * and `instDepth` says nothing about it. */
    if (ip != e->curOffset) return true;
    if (e->depth < e->instDepth) {
        e->whyNot = "a guard resumes an instruction whose operands it has "
                    "already consumed";
        return false;
    }
    if (e->depth == e->instDepth) return true;
    unsigned seen = 0;
    for (unsigned i = 0; i < e->instDepth; i++) {
        if (holdsRegister(e->stack[i])) seen++;
    }
    *depthOut = e->instDepth;
    *valueDepthOut = seen;
    return true;
}

/* Take the record without emitting the branch to it. The model is what it is
 * at this moment, so a site whose *code* is emitted later -- a self-call's
 * cold block, which lives with the stubs -- still has to record here. */
static bool deoptRecordAt(Emit *e, uint32_t ip, bool lastFromDesc,
                          unsigned *out) {
    /* A deopt stub writes every entry out of fpRegAt, so nothing may still be
     * borrowing a local's register here. Releasing it at this point was tried
     * and is wrong: a guard is often emitted inside a span some earlier branch
     * skips over -- emitBoundsNormalise's is -- so the fmov would land on a
     * path that is not taken and matrix_mul read `sum` from a register nothing
     * had written. The release happens at the top of the instruction instead,
     * where it is unconditionally on the path; this is the assertion that it
     * did, and it declines rather than miscompiles if some opcode is reached
     * that fpBorrowSurvives should not have allowed through. */
    if (e->fpBorrow != 0) {
        e->whyNot = "a float borrow reached a guard";
        e->failed = true;
        return false;
    }
    if (e->deoptCount >= JIT_MAX_DEOPT) {
        e->whyNot = "too many guards to record";
        e->failed = true;
        return false;
    }
    unsigned k = e->deoptCount++;
    if (!deoptSite(e, ip, &e->deopt[k].ip, &e->deopt[k].depth,
                   &e->deopt[k].valueDepth)) {
        e->failed = true;
        return false;
    }
    if (lastFromDesc && !e->inlining && e->deopt[k].depth != e->depth) {
        /* The from-descriptor entry is the top of the record, so a record that
         * was trimmed is no longer describing it. No site does both today. */
        e->whyNot = "a call's result guard resumes before the call";
        e->failed = true;
        return false;
    }
    e->deopt[k].lastFromDesc = lastFromDesc;
    e->deopt[k].fpLive       = e->fpLive;
    for (unsigned i = 0; i < e->deopt[k].depth; i++) {
        e->deopt[k].kinds[i]   = e->stack[i];
        e->deopt[k].classes[i] = e->stackClass[i];
    }
    *out = k;
    return true;
}

static void branchOnDeoptAt(Emit *e, unsigned cond, uint32_t ip,
                            bool lastFromDesc) {
    if (e->fixupCount >= JIT_MAX_FIXUPS) { e->failed = true; return; }
    unsigned k;
    if (!deoptRecordAt(e, ip, lastFromDesc, &k)) return;
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
    if (e->fpBorrow != 0) {          /* see deoptRecordAt */
        e->whyNot = "a float borrow reached a guard";
        e->failed = true;
        return;
    }
    if (e->deoptCount >= JIT_MAX_DEOPT) {
        e->whyNot = "too many guards to record";
        e->failed = true;
        return;
    }
    unsigned k = e->deoptCount++;
    if (!deoptSite(e, e->curOffset, &e->deopt[k].ip, &e->deopt[k].depth,
                   &e->deopt[k].valueDepth)) {
        e->failed = true;
        return;
    }
    e->deopt[k].lastFromDesc = false;
    e->deopt[k].fpLive     = e->fpLive;
    for (unsigned i = 0; i < e->deopt[k].depth; i++) {
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

/* `jaiNormalizeIndex` and the bounds test in one, leaving the settled index in
 * `rOut`. `rCount` must hold a zero-extended 32-bit count, which is what an
 * `ldrw` of ObjList.count or ObjString.length gives, so one *unsigned* compare
 * answers both ends: a negative index is a huge unsigned and fails the same
 * test an index past the end does.
 *
 * That is four instructions on the path every real loop takes, against the
 * seven the signed form needed -- and indexing was 42 of the 62 instructions in
 * matrix_mul's innermost body, three of these back to back. `xs[-1]` is still
 * exact; it just walks the three instructions the branch skips. */
static void emitBoundsNormalise(Emit *e, unsigned rIdx, unsigned rCount,
                                unsigned rOut) {
    emit(e, jaiA64MovX(rOut, rIdx));
    emit(e, jaiA64SubsXReg(31, rOut, rCount));
    /* The skip used to be a hand-counted four instructions. branchOnDeopt is
     * allowed to emit -- it releases FP borrows so the stub can find every
     * entry -- and one extra instruction inside the span turned the skip into
     * a jump onto the bail branch, which read `sum` from a register nothing
     * had written. Measuring the span is the same instruction and cannot rot. */
    unsigned skip = e->count;
    emit(e, jaiA64BCond(JAI_A64_LO, 0));
    emit(e, jaiA64AddX(rOut, rOut, rCount));
    emit(e, jaiA64SubsXReg(31, rOut, rCount));
    branchOnDeopt(e, JAI_A64_HS);
    if (skip < e->count && e->count <= JIT_MAX_INSTS) {
        e->code[skip] = jaiA64BCond(JAI_A64_LO, (int32_t)(e->count - skip));
    }
}

static void branchOnCondition(Emit *e, unsigned cond) {
    if (e->fixupCount >= JIT_MAX_FIXUPS) { e->failed = true; return; }
    fpSyncAll(e);
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
    fpSyncAll(e);
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

/* ------------------------------------------------------------------ */
/* Module globals                                                       */
/* ------------------------------------------------------------------ */

/* The storage for a module global, or NULL when the name has none here.
 *
 * A JaiEntry's address is stable for as long as the table does not rehash,
 * delete or clear -- jaiTableSetInterned reaches ensureRoom only when it is
 * about to insert a NEW key, so overwriting an existing global never moves
 * anything. That is what makes a compiled load one `ldr` from a baked pointer
 * rather than a probe. `JaiTable.keyVersion` counts every event that breaks
 * it, and emitGlobalsGuard checks it. */
static JaiEntry *globalSlot(Emit *e, ObjClosure *closure, uint32_t nameIdx,
                            Value *out) {
    ObjFunction *fn = closure->fn;
    if (fn->module == NULL) return NULL;
    if (nameIdx >= (uint32_t)fn->chunk.constants.count) return NULL;
    Value name = fn->chunk.constants.data[nameIdx];
    if (!IS_STRING(name)) return NULL;
    JaiTable *t = &fn->module->globals;
    JaiEntry *slot = jaiTableFindEntryInterned(t, AS_STRING(name));
    if (slot == NULL) return NULL;
    /* The whole body reads one table, so one guard covers every slot. */
    if (e->globalsTable == NULL) {
        e->globalsTable = t;
        e->globalsKeyVersion = t->keyVersion;
    } else if (e->globalsTable != t) {
        return NULL;
    }
    if (out != NULL) *out = slot->value;
    return slot;
}

/* The table has not rehashed since this was compiled, so every baked slot
 * address is still the live storage for the name it came from.
 *
 * Emitted before EVERY access rather than hoisted. Hoisting is sound only with
 * a claim about control flow -- a call-out between a guard and a later access
 * on a back edge would be unguarded -- and the cheap version of that claim is
 * the kind of reasoning this file has been bitten by. Measured cost is in the
 * plan; it is four instructions on a predictable branch. */
static void emitGlobalsGuard(Emit *e) {
    uint32_t at = e->globalsKeyVersion;
    emitConst64(e, JIT_SCRATCH_D,
                (int64_t)(uintptr_t)&e->globalsTable->keyVersion);
    emit(e, jaiA64LdrW(JIT_SCRATCH_C, JIT_SCRATCH_D, 0));
    if (at <= 0xfffu) {
        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_C, at));
    } else {
        emitConst64(e, JIT_SCRATCH_B, (int64_t)at);
        emit(e, jaiA64SubsX(31, JIT_SCRATCH_C, JIT_SCRATCH_B));
    }
    branchOnDeopt(e, JAI_A64_NE);
}

/* `m->version` retires the caches that resolved a NAME to a heap object or to
 * presence: the interpreter's global inline cache (which stores an entry
 * index, so a value write cannot invalidate it), OP_FORMAT's "is `str` still
 * the builtin", and this tier's own baked classes, closures and natives --
 * every one of which required the bound value to BE a heap object.
 *
 * So a store that replaces a non-object with a non-object changes nothing any
 * of them cached, and may skip the bump. That matters a great deal: the bump
 * is what makes `total = total + 1` at module scope retire every compiled
 * function in the module, measured at 6.0M declined entries and 19% of the
 * running time on a Point-allocating loop. */
static void emitVersionBump(Emit *e, ObjModule *m) {
    emitConst64(e, JIT_SCRATCH_A, (int64_t)(uintptr_t)&m->version);
    emit(e, jaiA64LdrW(JIT_SCRATCH_B, JIT_SCRATCH_A, 0));
    emit(e, jaiA64AddXImm(JIT_SCRATCH_B, JIT_SCRATCH_B, 1));
    emit(e, jaiA64StrW(JIT_SCRATCH_B, JIT_SCRATCH_A, 0));
}

/* The kind a global's live value has, or false when the tier has no kind for
 * it. Read off the value the way a parameter is seeded. */
static bool globalKind(Value v, SlotKind *k, uint32_t *shape, ObjClass **kls) {
    *shape = 0; *kls = NULL;
    if (IS_INT(v))      { *k = SLOT_INT;   return true; }
    if (IS_FLOAT(v))    { *k = SLOT_FLOAT; return true; }
    if (IS_BOOL(v))     { *k = SLOT_BOOL;  return true; }
    if (IS_LIST(v))     { *k = SLOT_LIST;  return true; }
    if (IS_INSTANCE(v)) {
        ObjInstance *inst = AS_INSTANCE(v);
        if (inst->klass == NULL) return false;
        *k = SLOT_INST; *kls = inst->klass; *shape = inst->klass->shapeId;
        return true;
    }
    /* Anything else on the heap -- a dict, a string, a closure -- can be
     * loaded, passed and stored, and nothing else. A class, a function and a
     * native never reach here: OP_GET_GLOBAL resolves those to their own stack
     * kinds above, which occupy no register. */
    if (IS_OBJ(v) && AS_OBJ(v) != NULL && !IS_CLASS(v) && !IS_CLOSURE(v) &&
        !IS_NATIVE(v)) {
        *k = SLOT_OBJ; return true;
    }
    return false;
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
/* Write every object this body is holding into the descriptor's root array.
 *
 * Shared by the descriptor path, where a C helper pushes them, and by the
 * self-call path, where the emitted code links the descriptor onto the
 * collector's frame chain instead -- a `bl` to the prologue pushes nothing. */
static bool emitRootFill(Emit *e, unsigned d, unsigned *nrootsOut) {
    unsigned nroots = 0;
    for (unsigned slot = e->base; slot < e->base + e->locals; slot++) {
        if (e->localKind[slot] != SLOT_INST &&
            e->localKind[slot] != SLOT_LIST &&
            e->localKind[slot] != SLOT_OBJ &&
            e->localKind[slot] != SLOT_ITER &&
            e->localKind[slot] != SLOT_MAYBE_INST) {
            continue;
        }
        if (nroots >= JIT_MAX_SAVED) { e->whyNot = "too many roots"; return false; }
        unsigned at = d + (unsigned)offsetof(JitCallDesc, roots) +
                      nroots * (unsigned)sizeof(Value);
        unsigned rslot = localIn(e, slot, JIT_SCRATCH_C);
        emitTagFor(e, e->localKind[slot], rslot, JIT_SCRATCH_B, JIT_SCRATCH_A);
        emit(e, jaiA64StrW(JIT_SCRATCH_B, 31, at));
        emit(e, jaiA64StrX(rslot, 31, at + 8));
        nroots++;
    }

    /* And what the operand stack is holding. Locals alone were enough only
     * while nothing object-shaped stayed on the stack across a call -- but an
     * iterator does exactly that: OP_GET_ITER pushes an ObjIter that lives in
     * a register for the whole loop, and the call this descriptor belongs to
     * may be the one that collects it. Visible only under --gc-stress, and
     * only once a body with a loop like that could compile at all. */
    /* Count the register-holding entries from the bottom rather than assuming
     * they are the top `valueDepth` of them. A class, a resolved function, a
     * builtin and `self` occupy no register and can sit anywhere -- a callee
     * pushed before its arguments puts one squarely in the middle, which is
     * exactly what `join(f(a), f(b))` does. Subtracting valueDepth names the
     * wrong register for everything above such an entry, and worse, starts the
     * walk past entries that still need rooting. The deopt stub has always
     * counted this way. */
    unsigned seen = 0;
    for (unsigned idx = 0; idx < e->depth; idx++) {
        SlotKind k = e->stack[idx];
        if (!holdsRegister(k)) continue;
        unsigned reg = JIT_FIRST_SAVED + regBase(e) +
                       (e->usesUpvalues ? 1u : 0u) + seen;
        seen++;
        if (k != SLOT_INST && k != SLOT_LIST && k != SLOT_OBJ &&
            k != SLOT_ITER && k != SLOT_MAYBE_INST) {
            continue;
        }
        if (nroots >= JIT_MAX_SAVED) { e->whyNot = "too many roots"; return false; }
        unsigned at = d + (unsigned)offsetof(JitCallDesc, roots) +
                      nroots * (unsigned)sizeof(Value);
        emitTagFor(e, k, reg, JIT_SCRATCH_B, JIT_SCRATCH_A);
        emit(e, jaiA64StrW(JIT_SCRATCH_B, 31, at));
        emit(e, jaiA64StrX(reg, 31, at + 8));
        nroots++;
    }

    *nrootsOut = nroots;
    return true;
}

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
            k != SLOT_ITER && k != SLOT_MAYBE_INST) {
            e->whyNot = "an argument kind this call cannot pass";
            return false;
        }
        unsigned at = d + (unsigned)offsetof(JitCallDesc, args) +
                      i * (unsigned)sizeof(Value);
        unsigned reg = JIT_FIRST_SAVED + regBase(e) +
                       (e->usesUpvalues ? 1u : 0u) +
                       (idx - (e->depth - e->valueDepth));
        /* A maybe-instance's tag is not a property of its kind, and this Value
         * reaches jaiCallValue: writing VAL_OBJ over a zero payload would hand
         * the interpreter a null pointer dressed as an object. */
        emitTagFor(e, k, reg, JIT_SCRATCH_B, JIT_SCRATCH_A);
        emit(e, jaiA64StrW(JIT_SCRATCH_B, 31, at));
        emit(e, jaiA64StrX(reg, 31, at + 8));
    }

    unsigned nroots = 0;
    if (!emitRootFill(e, d, &nroots)) return false;
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

/* A call whose callee did not return cleanly. Out of line so that the body
 * keeps one compare and one not-taken branch per call.
 *
 * Verdict 4 is the one that needed building: the callee deoptimised part-way
 * and may have written, so the call can neither be re-executed nor recorded
 * over -- gDeopt is a single global. The callee is FINISHED in the interpreter
 * from its own record instead, and its value handed back here, which consumes
 * the record at the innermost frame that sees it. That is why one record still
 * suffices however deep the recursion goes, and it is what makes a recursive
 * body that writes compilable -- and, since `callee` may name someone else,
 * what makes a DIRECT call to a method that writes compilable too.
 *
 * `closure` is the body being compiled, which is what a self-call finishes.
 *
 * Emitted by both tiers. In an OSR form the fall-through is a continuation of
 * the loop, so nothing is written back here: every branch out of this block
 * goes to a stub that does its own syncing. */
static void emitSelfSlowStubs(Emit *e, ObjClosure *closure) {
    for (unsigned si = 0; si < e->selfSlowCount; si++) {
        e->selfSlow[si].stub = (int)e->count;
        unsigned d = e->descOffset;
        unsigned resultAt = d + (unsigned)offsetof(JitCallDesc, result);

        emit(e, jaiA64SubsXImm(31, 1, 2));             /* the callee raised */
        emit(e, jaiA64BCond(JAI_A64_EQ,
                            (int32_t)(e->exceptionExit - (int)e->count)));

        emit(e, jaiA64SubsXImm(31, 1, 4));
        if (e->fixupCount >= JIT_MAX_FIXUPS) { e->failed = true; break; }
        e->fixups[e->fixupCount].instIndex    = (int)e->count;
        e->fixups[e->fixupCount].targetOffset =
            FIXUP_DEOPT - e->selfSlow[si].deoptBail;
        e->fixups[e->fixupCount].conditional  = true;
        e->fixups[e->fixupCount].depth        = -1;
        e->fixupCount++;
        emit(e, jaiA64BCond(JAI_A64_NE, 0));

        /* jaiJitFinishDeopt runs interpreted code, which allocates, so the
         * roots this frame filled before the `bl` go back on the chain. */
        if (e->selfSlow[si].roots > 0) {
            emitConst64(e, JIT_SCRATCH_A, (int64_t)(uintptr_t)&gJitFrames);
            emit(e, jaiA64LdrX(JIT_SCRATCH_B, JIT_SCRATCH_A, 0));
            emit(e, jaiA64AddXImm(JIT_SCRATCH_C, 31, d));
            emit(e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_C,
                               (unsigned)offsetof(JitCallDesc, link)));
            emit(e, jaiA64StrX(JIT_SCRATCH_C, JIT_SCRATCH_A, 0));
        }
        emitConst64(e, 0, (int64_t)(uintptr_t)(e->selfSlow[si].callee != NULL
                                                   ? e->selfSlow[si].callee
                                                   : closure));
        emit(e, jaiA64AddXImm(1, 31, resultAt));
        emitConst64(e, JIT_SCRATCH_D, (int64_t)(uintptr_t)&jaiJitFinishDeopt);
        emit(e, jaiA64Blr(JIT_SCRATCH_D));
        if (e->selfSlow[si].roots > 0) {
            emitConst64(e, JIT_SCRATCH_A, (int64_t)(uintptr_t)&gJitFrames);
            emit(e, jaiA64AddXImm(JIT_SCRATCH_C, 31, d));
            emit(e, jaiA64LdrX(JIT_SCRATCH_B, JIT_SCRATCH_C,
                               (unsigned)offsetof(JitCallDesc, link)));
            emit(e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_A, 0));
        }
        emit(e, jaiA64SubsXImm(31, 0, 0));             /* false: it raised */
        emit(e, jaiA64BCond(JAI_A64_EQ,
                            (int32_t)(e->exceptionExit - (int)e->count)));

        /* The interpreted continuation may return a kind this body was not
         * compiled for. It is in the descriptor with whatever tag it really
         * has, which is exactly what a `lastFromDesc` record writes out. */
        emit(e, jaiA64LdrW(JIT_SCRATCH_A, 31, resultAt));
        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, e->selfSlow[si].tag));
        if (e->fixupCount >= JIT_MAX_FIXUPS) { e->failed = true; break; }
        e->fixups[e->fixupCount].instIndex    = (int)e->count;
        e->fixups[e->fixupCount].targetOffset =
            FIXUP_DEOPT - e->selfSlow[si].deoptKind;
        e->fixups[e->fixupCount].conditional  = true;
        e->fixups[e->fixupCount].depth        = -1;
        e->fixupCount++;
        emit(e, jaiA64BCond(JAI_A64_NE, 0));
        emit(e, jaiA64LdrX(e->selfSlow[si].resultReg, 31, resultAt + 8));

        /* VAL_OBJ is every heap object, so a compiled body that was promised
         * an instance or a list has to see the object's type before it treats
         * the pointer as one: reading `klass` off an ObjString does not fault,
         * it answers wrongly. Both checks go back to the same record. */
        if (e->selfSlow[si].retType >= 0) {
            unsigned rr = e->selfSlow[si].resultReg;
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, rr,
                               (unsigned)offsetof(Obj, type)));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A,
                                   (unsigned)e->selfSlow[si].retType));
            if (e->fixupCount >= JIT_MAX_FIXUPS) { e->failed = true; break; }
            e->fixups[e->fixupCount].instIndex    = (int)e->count;
            e->fixups[e->fixupCount].targetOffset =
                FIXUP_DEOPT - e->selfSlow[si].deoptKind;
            e->fixups[e->fixupCount].conditional  = true;
            e->fixups[e->fixupCount].depth        = -1;
            e->fixupCount++;
            emit(e, jaiA64BCond(JAI_A64_NE, 0));
            if (e->selfSlow[si].retShape != 0) {
                emit(e, jaiA64LdrX(JIT_SCRATCH_A, rr,
                                   (unsigned)offsetof(ObjInstance, klass)));
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_A,
                                   (unsigned)offsetof(ObjClass, shapeId)));
                emitConst64(e, JIT_SCRATCH_B,
                            (int64_t)e->selfSlow[si].retShape);
                emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
                if (e->fixupCount >= JIT_MAX_FIXUPS) { e->failed = true; break; }
                e->fixups[e->fixupCount].instIndex    = (int)e->count;
                e->fixups[e->fixupCount].targetOffset =
                    FIXUP_DEOPT - e->selfSlow[si].deoptKind;
                e->fixups[e->fixupCount].conditional  = true;
                e->fixups[e->fixupCount].depth        = -1;
                e->fixupCount++;
                emit(e, jaiA64BCond(JAI_A64_NE, 0));
            }
        }
        emit(e, jaiA64B((int32_t)(e->selfSlow[si].returnTo - (int)e->count)));
    }
}

/* The cold half of `xs.push(v)`: reserve, refill the count the fast path had
 * already loaded, and branch back into it.
 *
 * Emitted with the other stubs so the store keeps one not-taken branch. No
 * descriptor and no roots -- see jitListGrow for why a realloc cannot collect.
 * `e->exceptionExit` must already be emitted, which it is at both call sites.
 *
 * This is a continuation, not a way out, so an OSR form must NOT sync its
 * iterator or its locals here: the loop carries on with them where they are. */
static void emitGrowStubs(Emit *e) {
    for (unsigned gi = 0; gi < e->growCount; gi++) {
        e->grow[gi].stub = (int)e->count;
        emit(e, jaiA64MovX(0, e->grow[gi].listReg));
        emit(e, jaiA64MovzX(1, e->grow[gi].tag, 0));
        emit(e, jaiA64MovX(2, e->grow[gi].valReg));
        emitConst64(e, JIT_SCRATCH_D, (int64_t)(uintptr_t)&jitListGrow);
        emit(e, jaiA64Blr(JIT_SCRATCH_D));
        emit(e, jaiA64SubsXImm(31, 0, 0));
        emit(e, jaiA64BCond(JAI_A64_NE,
                            (int32_t)(e->exceptionExit - (int)e->count)));
        emit(e, jaiA64LdrW(e->grow[gi].countReg, e->grow[gi].listReg,
                           (unsigned)offsetof(ObjList, count)));
        emit(e, jaiA64B((int32_t)(e->grow[gi].returnTo - (int)e->count)));
    }
}

/* Every argument of a direct branch arrives as a raw payload, so the kind the
 * caller holds has to be the kind the callee was specialised to -- and where
 * that kind is an instance, the same class shape, because every field offset
 * in the callee's body was resolved against it. jaiJitEnterFunc checks exactly
 * this, one Value at a time, and it is the only thing standing between a
 * float's bits and a body that will treat them as a pointer.
 *
 * `firstIdx` is the operand-stack index of the first thing passed in a
 * register. For a plain call that is the entry above the callee; for a method
 * it is the receiver itself, which is the callee's slot 0. */
static bool directCallArgsMatch(Emit *e, const ObjFunction *cfn,
                                unsigned firstIdx, unsigned argc) {
    for (unsigned i = 0; i < argc; i++) {
        unsigned idx = firstIdx + i;
        SlotKind have = e->stack[idx];
        SlotKind want = (SlotKind)cfn->jitParamKind[i];
        if (!holdsRegister(have)) {
            e->whyNot = "a direct call argument that is not in a register";
            return false;
        }
        if (want == SLOT_OPAQUE) continue;   /* never read; see seedLocals */
        if (want == SLOT_MAYBE_INST) {
            if (have != SLOT_INST && have != SLOT_MAYBE_INST) {
                e->whyNot = "a direct call argument is not the parameter's kind";
                return false;
            }
        } else if (have != want) {
            e->whyNot = "a direct call argument is not the parameter's kind";
            return false;
        }
        if ((want == SLOT_INST || want == SLOT_MAYBE_INST) &&
            e->stackShape[idx] != cfn->jitParamShape[i]) {
            e->whyNot = "a direct call passing a different class";
            return false;
        }
    }
    return true;
}

/* Branch straight to a compiled callee's entry instead of building a
 * descriptor and going out through jaiCallValue and an interpreter frame.
 *
 * The convention is the one jaiJitEnterFunc uses and a self-call already
 * emits: raw payloads in x0.., the closure itself in the last argument
 * register when the callee's body reads an upvalue, and on return x0 the value
 * with x1 the verdict. Skipping jaiJitEnterFunc means skipping everything it
 * checks, so each of those checks has to have an answer here:
 *
 *   - the module version, by the caller's own entry guard, which is why the
 *     callee has to live in the caller's module;
 *   - every parameter's kind and class shape, by the caller's model -- this is
 *     what makes passing a raw payload rather than a Value sound;
 *   - the verdict, below.
 *
 * A nonzero verdict is the part that is not obvious, and there are two answers
 * depending on what the callee is allowed to have done.
 *
 *   - A callee that writes nothing can simply have the whole CALL abandoned:
 *     the interpreter re-executes it from the operand stack as it stood
 *     before, and the abandoned attempt left nothing behind. Two compares on
 *     the fast path, no stub.
 *   - A callee that DOES write cannot be re-run, and this is not a rare case
 *     -- `Vec2.add` returning a fresh Vec2 writes, and so does any method that
 *     stores a field. Verdict 4 there means the callee deoptimised part-way,
 *     so it is FINISHED in the interpreter from its own record, exactly as a
 *     recursive self-call already does. That is what the `selfSlow` block is,
 *     which is why this shares it rather than growing a second copy.
 *
 * A raised exception is different again: its effects have happened and the
 * interpreter owns it, so that one goes to the throw exit rather than back to
 * the call.
 *
 * `calleeReg` holds the ObjClosure when the callee is only known at run time,
 * or -1 when it is baked in. `cidx` is the operand-stack index of the callee
 * entry -- for a method, of the RECEIVER, which is the callee's slot 0 and
 * therefore an argument as well. `after` is the offset of the instruction the
 * call falls through to. */
static bool emitDirectCall(Emit *e, ObjFunction *caller, ObjFunction *cfn,
                           Value calleeVal, int calleeReg, unsigned cidx,
                           unsigned argc, uint32_t callOff, uint32_t after,
                           bool method) {
    if (cfn->module != caller->module) {
        e->whyNot = "a direct callee from another module";
        return false;
    }
    /* Slot 0 is the closure for a plain function and the receiver for a
     * method, so which of the two this is decides where the arguments start
     * and whether the receiver is one of them. */
    if (cfn->jitArgBase != (method ? 0u : 1u)) {
        e->whyNot = method
            ? "a direct method that does not take its receiver in slot 0"
            : "a direct callee that is not a plain function";
        return false;
    }
    /* The callee's own baked classes, closures and natives are pinned by ITS
     * jitFuncModuleVersion, checked at an entry this is about to skip. Handing
     * that job to the caller's check works only if the two agree NOW: a global
     * rebound after the callee compiled leaves a form jaiJitEnterFunc would
     * refuse forever, still reachable through jitFunc, and the caller
     * compiling afterwards would pin the newer version and never notice. */
    if (caller->module == NULL ||
        cfn->jitFuncModuleVersion != caller->module->version) {
        e->whyNot = "a direct callee compiled against an older module";
        return false;
    }
    /* A callee that writes is finished in the interpreter on verdict 4, and
     * that needs its ObjClosure -- which only a callee baked in at compile
     * time provides. */
    bool writes = !cfn->jitFuncNoWrite;
    if (writes && (calleeReg >= 0 || !IS_CLOSURE(calleeVal))) {
        e->whyNot = "a direct callee that writes and is not known here";
        return false;
    }
    /* `nargs` is how many registers the branch fills. A method's receiver is
     * one of them; a plain call's callee entry holds no register at all. */
    unsigned nargs = method ? argc + 1u : argc;
    unsigned firstIdx = method ? cidx : cidx + 1u;
    unsigned calleeArgs = (unsigned)cfn->jitArgCount;
    bool wantsClosure = calleeArgs == nargs + 1u;
    if (!wantsClosure && calleeArgs != nargs) {
        e->whyNot = "a direct callee with a different arity";
        return false;
    }
    if (calleeArgs > JIT_MAX_ARITY) {
        e->whyNot = "a direct callee with too many arguments";
        return false;
    }

    if (!directCallArgsMatch(e, cfn, firstIdx, nargs)) return false;
    if (wantsClosure &&
        (SlotKind)cfn->jitParamKind[nargs] != SLOT_CLOSURE) {
        e->whyNot = "a direct callee whose trailing argument is not its closure";
        return false;
    }
    if (wantsClosure && calleeReg < 0 && !IS_CLOSURE(calleeVal)) {
        e->whyNot = "a direct callee that wants a closure it has not got";
        return false;
    }
    SlotKind rk = (SlotKind)cfn->jitReturnKind;
    ObjClass *rcls = NULL;
    if (rk != SLOT_INT && rk != SLOT_FLOAT && rk != SLOT_BOOL &&
        rk != SLOT_INST && rk != SLOT_LIST && rk != SLOT_OBJ) {
        e->whyNot = "callee's return kind not usable";
        return false;
    }
    if (rk == SLOT_INST &&
        (cfn->jitReturnShape == 0 ||
         !jaiClassForShape(cfn->jitReturnShape, &rcls) || rcls == NULL)) {
        e->whyNot = "callee's return class not on record";
        return false;
    }
    /* Asked here rather than where the slot is taken: below this point the
     * root fill has been emitted and the descriptor linked onto the collector's
     * chain, so there is no falling back to the descriptor path any more. */
    if (writes && e->selfSlowCount >= JIT_MAX_SELF_SLOW) {
        e->whyNot = "more slow call sites than the tier tracks";
        return false;
    }

    /* Past here everything is settled and this call is happening: a failure
     * below is the emitter running out of room, not a decision, so it stops
     * the compile rather than falling back to the descriptor path onto a
     * half-written instruction stream. */

    /* Roots before the branch: a `bl` pushes none, and the callee may
     * allocate -- OP_GET_SLICE builds a fresh list without ever counting as a
     * heap write. */
    unsigned callRoots = 0;
    if (!emitRootFill(e, e->descOffset, &callRoots)) { e->failed = true; return false; }
    if (callRoots > 0) {
        unsigned dd = e->descOffset;
        emit(e, jaiA64MovzX(JIT_SCRATCH_A, callRoots, 0));
        emit(e, jaiA64StrX(JIT_SCRATCH_A, 31,
                           dd + (unsigned)offsetof(JitCallDesc, nroots)));
        emitConst64(e, JIT_SCRATCH_A, (int64_t)(uintptr_t)&gJitFrames);
        emit(e, jaiA64LdrX(JIT_SCRATCH_B, JIT_SCRATCH_A, 0));
        emit(e, jaiA64AddXImm(JIT_SCRATCH_C, 31, dd));
        emit(e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_C,
                           (unsigned)offsetof(JitCallDesc, link)));
        emit(e, jaiA64StrX(JIT_SCRATCH_C, JIT_SCRATCH_A, 0));
    }

    unsigned firstArg = valueXReg(e, firstIdx - (e->depth - e->valueDepth));
    for (unsigned i = 0; i < nargs; i++) {
        emit(e, jaiA64MovX(i, firstArg + i));
    }
    if (wantsClosure) {
        if (calleeReg >= 0) emit(e, jaiA64MovX(nargs, (unsigned)calleeReg));
        else emitConst64(e, nargs, (int64_t)(uintptr_t)AS_OBJ(calleeVal));
    }
    emitConst64(e, JIT_SCRATCH_D, (int64_t)(uintptr_t)cfn->jitFunc);
    emit(e, jaiA64Blr(JIT_SCRATCH_D));

    if (callRoots > 0) {
        unsigned dd = e->descOffset;
        emitConst64(e, JIT_SCRATCH_A, (int64_t)(uintptr_t)&gJitFrames);
        emit(e, jaiA64AddXImm(JIT_SCRATCH_C, 31, dd));
        emit(e, jaiA64LdrX(JIT_SCRATCH_B, JIT_SCRATCH_C,
                           (unsigned)offsetof(JitCallDesc, link)));
        emit(e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_A, 0));
    }

    unsigned si = 0;
    if (writes) {
        /* One compare and one not-taken branch on the fast path; every other
         * answer is the shared cold block. The record has to be taken here,
         * where the model still holds the receiver and the arguments, even
         * though the block is emitted with the stubs. */
        if (e->selfSlowCount >= JIT_MAX_SELF_SLOW) { e->failed = true; return false; }
        si = e->selfSlowCount++;
        e->selfSlow[si].roots    = callRoots;
        e->selfSlow[si].stub     = -1;
        e->selfSlow[si].callee   = AS_CLOSURE(calleeVal);
        e->selfSlow[si].retShape = rk == SLOT_INST ? cfn->jitReturnShape : 0;
        e->selfSlow[si].retType  = rk == SLOT_INST ? (int)OBJ_INSTANCE
                                 : rk == SLOT_LIST ? (int)OBJ_LIST
                                                   : -1;
        if (!deoptRecordAt(e, callOff, false, &e->selfSlow[si].deoptBail)) {
            e->failed = true;
            return false;
        }
        emit(e, jaiA64SubsXImm(31, 1, 0));
        if (e->fixupCount >= JIT_MAX_FIXUPS) { e->failed = true; return false; }
        e->fixups[e->fixupCount].instIndex    = (int)e->count;
        e->fixups[e->fixupCount].targetOffset = FIXUP_SELFSLOW - si;
        e->fixups[e->fixupCount].conditional  = true;
        e->fixups[e->fixupCount].depth        = -1;
        e->fixupCount++;
        emit(e, jaiA64BCond(JAI_A64_NE, 0));
    } else {
        /* Verdict 2 is a pending exception: the interpreter owns it and this
         * call must not run again. */
        emit(e, jaiA64SubsXImm(31, 1, 2));
        if (e->fixupCount >= JIT_MAX_FIXUPS) { e->failed = true; return false; }
        e->fixups[e->fixupCount].instIndex    = (int)e->count;
        e->fixups[e->fixupCount].targetOffset = FIXUP_THREW;
        e->fixups[e->fixupCount].conditional  = true;
        e->fixups[e->fixupCount].depth        = -1;
        e->fixupCount++;
        emit(e, jaiA64BCond(JAI_A64_EQ, 0));
        /* Anything else -- a bail, or a guard that failed inside the callee --
         * hands the whole call back. The record is taken with the callee and
         * its arguments still on the model's stack, which is what the
         * interpreter expects to find at this offset. */
        emit(e, jaiA64SubsXImm(31, 1, 0));
        branchOnDeoptAt(e, JAI_A64_NE, callOff, false);
    }

    for (unsigned i = 0; i < nargs; i++) {
        unsigned r;
        if (!popValue(e, &r, NULL)) { e->failed = true; return false; }
    }
    if (!method) {
        if (e->depth == 0) { e->failed = true; return false; }
        e->depth--;                              /* the callee entry */
    }
    if (!pushValue(e, rk, cfn->jitReturnShape, rcls)) { e->failed = true; return false; }
    emit(e, jaiA64MovX(pushReg(e) - 1, 0));
    if (writes) {
        e->selfSlow[si].resultReg = pushReg(e) - 1;
        e->selfSlow[si].returnTo  = (int)e->count;
        e->selfSlow[si].tag = rk == SLOT_INT   ? VAL_INT
                            : rk == SLOT_FLOAT ? VAL_FLOAT
                            : rk == SLOT_BOOL  ? VAL_BOOL
                                               : VAL_OBJ;
        /* The interpreted continuation is typed by nothing this compiled for,
         * so what it hands back is checked and a surprise resumes AFTER the
         * call -- which has happened and must not happen twice. */
        if (!deoptRecordAt(e, after, true, &e->selfSlow[si].deoptKind)) {
            e->failed = true;
            return false;
        }
        /* A call that writes is an effect, so no bail may follow it -- the
         * same rule the descriptor path lives under. */
        e->wroteHeap = true;
    }
    /* For a callee that writes nothing this is deliberately NOT wroteHeap,
     * unlike the descriptor path: it stores nothing and calls nothing, so an
     * interpreted re-run of the whole caller would repeat no effect. It may
     * still allocate -- OP_GET_SLICE does, and deliberately does not set the
     * flag either -- and a fresh object is not an effect anything can see. */
    return true;
}

static bool compileBody(Emit *e, ObjClosure *closure);
static int instructionLength(const Chunk *c, int off);

/* ---- literal operands -------------------------------------------------- */

/* Can anything but the fall-through arrive at `off`?
 *
 * The walk through a body is linear, so the instruction it visited last is the
 * one lexically before this -- but only when nothing jumps here. `x // (if c {
 * 2 } else { 4 })` puts an OP_INT immediately before the OP_FLOORDIV *and* a
 * jump from the other arm onto it, so "the previous instruction pushed 2" is
 * true on one path and false on the other. Reading the constant off the
 * previous instruction without this test is a miscompile, not a decline.
 *
 * Every branch in the chunk is scanned, not just the ones already emitted: a
 * back edge is compiled after its target is walked, so consulting the fixup
 * list would miss exactly the loop tops. Handler and finally addresses count
 * too -- the unwinder resumes at one with a stack this walk never saw. */
static bool offsetIsBranchTarget(const Chunk *c, uint32_t off) {
    for (int at = 0; at < c->count;) {
        int len = instructionLength(c, at);
        if (len <= 0) return true;      /* undecodable: assume the worst */
        int rel = jaiOpBranchOperandAt(c->code[at]);
        if (rel >= 0) {
            int16_t jump = jaiReadI16(c->code + at + 1 + rel);
            /* Every branch operand is measured from the end of the
             * instruction, which is what `at + len` is. */
            if ((int32_t)(at + len) + jump == (int32_t)off) return true;
        }
        at += len;
    }
    return false;
}

/* The int literal the instruction at `prevOff` pushes, when the instruction at
 * `off` is guaranteed to see it on top of the stack.
 *
 * `OP_INT` carries the value in its operand and `OP_CONST` names a pool entry;
 * those are the only two ways a literal reaches the stack. The adjacency check
 * is belt and braces -- the walk is linear, so it always holds -- but an
 * opcode arm that advanced `off` by the wrong amount would break it, and that
 * has happened here before (OP_FORMAT advanced by nine instead of ten). */
static bool literalIntOperand(const ObjFunction *fn, int prevOff, int off,
                              int64_t *out) {
    if (prevOff < 0 || prevOff >= off) return false;
    const Chunk *c = &fn->chunk;
    if (prevOff + instructionLength(c, prevOff) != off) return false;
    uint8_t prev = c->code[prevOff];
    if (prev == OP_INT) {
        *out = jaiReadI16(c->code + prevOff + 1);
    } else if (prev == OP_CONST) {
        uint32_t idx = jaiReadU24(c->code + prevOff + 1);
        if (idx >= (uint32_t)c->constants.count) return false;
        Value k = c->constants.data[idx];
        if (!IS_INT(k)) return false;
        *out = AS_INT(k);
    } else {
        return false;
    }
    return !offsetIsBranchTarget(c, (uint32_t)off);
}

/* `k` is 2^shift, for a shift this can name. Positive only: floor division by
 * a negative power of two is not a shift, and `k` is at most 2^62 because 2^63
 * does not fit in a positive int64. */
static bool powerOfTwoShift(int64_t k, unsigned *shift) {
    if (k <= 0) return false;
    uint64_t u = (uint64_t)k;
    if ((u & (u - 1u)) != 0u) return false;
    unsigned s = 0;
    while ((u >> s) != 1u) s++;
    *shift = s;
    return true;
}

/* Something inside an inlined body could not be emitted, so the whole compile
 * is worth retrying with inlining off rather than declining: the same call
 * through the descriptor still compiles, and a compiled form with a real call
 * in it beats none at all. A file static for the same reason the Emit buffers
 * are -- compilation is not reentrant, nothing it calls compiles anything. */
static bool gInlineFailed;

/* Can this callee's body stand in for the call to it?
 *
 * Structural, and answered before anything is emitted, because a half-inlined
 * body cannot be taken back. The list is short on purpose and each item buys
 * something specific:
 *
 *   no branches      -- so there is no offset map to keep, no join to
 *                       reconcile, and no fixup that could name one of the
 *                       callee's offsets in the caller's table;
 *   one RETURN, last -- so there is one place the result appears;
 *   locals only in the four opcodes the frame below understands, and only
 *                       slots this callee actually has;
 *   globals only where they name one of the two builtins the tier emits
 *                       inline, so a global VALUE load -- which would bake a
 *                       JaiEntry from the callee's table and needs its own
 *                       guard -- never arises;
 *   nothing that stores -- because a guard inside the inlined body sends the
 *                       interpreter back to re-execute the whole call, and a
 *                       store made before it would then happen twice.
 *
 * What survives is straight-line arithmetic over registers, which is what the
 * main walker already speaks -- so it, and not a second one, is what emits
 * this. `evalA` in spectral is fifteen instructions of exactly this shape. */
static bool inlinableBody(ObjClosure *callee, unsigned argc,
                          unsigned *maxSlotOut) {
    ObjFunction *cfn = callee->fn;
    const Chunk *c = &cfn->chunk;
    if (cfn->arity != argc || cfn->defaultCount != 0) return false;
    if (cfn->flags & (FN_VARIADIC | FN_KWREST | FN_INIT)) return false;
    if (cfn->upvalueCount != 0) return false;
    if (c->count <= 0 || c->count > 128) return false;

    unsigned maxSlot = argc;
    bool sawReturn = false;
    for (int off = 0; off < c->count;) {
        uint8_t op = c->code[off];
        int len = instructionLength(c, off);
        if (len <= 0) return false;
        unsigned slot = 0, slot2 = 0;
        switch (op) {
        /* The local frame the caller builds understands exactly these. Any
         * other opcode naming a slot -- a field read off one, a compare
         * against one, an in-place update -- would read the CALLER's local of
         * that number, which is a different variable entirely. */
        case OP_GET_LOCAL:
        case OP_BIND:
            slot = jaiReadU16(c->code + off + 1);
            if (slot > maxSlot) maxSlot = slot;
            break;
        case OP_GET_LOCAL2:
        case OP_ADD_LOCALS:
            slot  = jaiReadU16(c->code + off + 1);
            slot2 = jaiReadU16(c->code + off + 3);
            if (slot > maxSlot) maxSlot = slot;
            if (slot2 > maxSlot) maxSlot = slot2;
            break;
        case OP_GET_GLOBAL: {
            uint32_t nameIdx = jaiReadU24(c->code + off + 1);
            Value nv;
            if (globalNative(callee, nameIdx, &nv) == NULL) return false;
            ObjNative *nat = AS_NATIVE(nv);
            const char *nm = nat->name != NULL ? nat->name->chars : "";
            if (strcmp(nm, "float") != 0 && strcmp(nm, "int") != 0) return false;
            break;
        }
        case OP_CALL:
            /* The only callee that can be on the stack here is one of the two
             * builtins above, and the tier emits those as one instruction. */
            if (c->code[off + 1] != 1) return false;
            break;
        case OP_CONST: case OP_INT: case OP_TRUE: case OP_FALSE:
        case OP_ADD: case OP_SUB: case OP_MUL: case OP_DIV:
        case OP_FLOORDIV: case OP_MOD: case OP_POW: case OP_NEG:
        case OP_BAND: case OP_BOR: case OP_BXOR:
        case OP_SHL: case OP_SHR: case OP_BNOT:
        case OP_TYPE_GUARD:
            break;
        case OP_RETURN:
            if (off + len != c->count) return false;
            sawReturn = true;
            break;
        default:
            return false;
        }
        off += len;
    }
    if (!sawReturn) return false;
    if (maxSlot > JIT_MAX_SLOTS) return false;
    *maxSlotOut = maxSlot;
    return true;
}

/* Emit the callee's body in place of the call.
 *
 * Only the argument entries already on the operand stack are needed: slot 1+i
 * IS entry cidx+1+i, so nothing is copied in. A slot the body binds pins one
 * further entry, which stays put underneath everything pushed after it --
 * sound only because the body is straight-line, so the stack above it is
 * balanced by the time the result is taken.
 *
 * The callee's module must be the caller's. Its body bakes builtins the same
 * way any compiled body does, and what retires those is the module-version
 * check at the CALLER's entry, since the callee's own is never run. */
static bool inlineGlobalCall(Emit *e, ObjFunction *caller, ObjClosure *callee,
                             unsigned argc, uint32_t callOff) {
    if (e->noInline) return false;
    ObjFunction *cfn = callee->fn;
    if (cfn->module != caller->module) return false;
    if (e->inlining) return false;             /* one level, no recursion */
    unsigned cidx = e->depth - argc - 1;
    unsigned maxSlot = 0;
    if (!inlinableBody(callee, argc, &maxSlot)) return false;
    if (getenv("JAI_JIT_WHY")) {
        fprintf(stderr, "[jit] inlining %s\n",
                cfn->name ? cfn->name->chars : "<anon>");
    }

    /* Every argument has to be in a register, since that is where the body
     * will read its parameters from. */
    for (unsigned i = 0; i < argc; i++) {
        if (!holdsRegister(e->stack[cidx + 1u + i])) return false;
    }

    int savedSlot[JIT_MAX_SLOTS + 1];
    memcpy(savedSlot, e->inlSlot, sizeof savedSlot);
    for (unsigned i = 0; i <= JIT_MAX_SLOTS; i++) e->inlSlot[i] = -1;
    for (unsigned i = 0; i < argc; i++) e->inlSlot[1u + i] = (int)(cidx + 1u + i);

    e->inlining     = true;
    e->inlDepth     = cidx + 1u + argc;
    e->inlPinned    = 0;
    e->inlValueBase = e->valueDepth;
    e->inlIp        = callOff;

    /* The callee's own offset map, so its offsets cannot land in the
     * caller's. Nothing reads it back -- there are no branches -- but
     * compileBody writes one entry per instruction either way. */
    int cmap[129], cdepths[129];
    for (int i = 0; i <= cfn->chunk.count; i++) { cmap[i] = -1; cdepths[i] = -1; }
    int *savedMap = e->offsetToInst, *savedDepths = e->offsetToDepth;
    unsigned savedCarry = e->fpCarryCount;
    uint32_t savedCurOffset = e->curOffset;
    unsigned savedInstDepth = e->instDepth;
    unsigned savedInstValue = e->instValueDepth;
    e->offsetToInst = cmap;
    e->offsetToDepth = cdepths;

    bool ok = compileBody(e, callee);

    e->offsetToInst = savedMap;
    e->offsetToDepth = savedDepths;
    e->fpCarryCount = savedCarry;
    e->curOffset = savedCurOffset;
    e->instDepth = savedInstDepth;
    e->instValueDepth = savedInstValue;

    if (!ok || e->failed) {
        /* Instructions have been written; there is no taking them back. The
         * whole compile is retried with inlining off, which is the same answer
         * the register budget already gets. */
        e->inlining = false;
        memcpy(e->inlSlot, savedSlot, sizeof savedSlot);
        gInlineFailed = true;
        e->failed = true;
        return false;
    }

    /* OP_RETURN left the result on top and everything the body pinned beneath
     * it. Both are read while `inlining` is still set, because that is what
     * says which bank they are in; only the result's new home belongs to the
     * caller. */
    unsigned rres;
    SlotKind kres;
    uint32_t rshape;
    ObjClass *rcls;
    if (e->depth <= cidx) { e->failed = true; return false; }
    rshape = e->stackShape[e->depth - 1];
    rcls   = e->stackClass[e->depth - 1];
    if (!popValue(e, &rres, &kres)) { e->failed = true; return false; }
    while (e->depth > cidx) {
        if (holdsRegister(e->stack[e->depth - 1])) {
            unsigned r;
            if (!popValue(e, &r, NULL)) { e->failed = true; return false; }
        } else {
            e->depth--;
        }
    }
    e->inlining = false;
    memcpy(e->inlSlot, savedSlot, sizeof savedSlot);

    if (!pushValue(e, kres, rshape, rcls)) { e->failed = true; return false; }
    unsigned dst = pushReg(e) - 1;
    if (dst != rres) emit(e, jaiA64MovX(dst, rres));
    e->inlined = true;
    return true;
}

/* A call to a global function that has itself compiled. Its return kind types
 * the result; the tag that actually comes back is checked, and a surprise
 * deopts to the instruction after the call, since the call has happened. */
static bool emitGlobalCall(Emit *e, ObjFunction *caller, unsigned argc,
                           uint32_t callOff, uint32_t after) {
    unsigned cidx = e->depth - argc - 1;
    Value cv = e->stackSeen[cidx];
    if (!IS_CLOSURE(cv)) { e->whyNot = "callee vanished"; return false; }
    ObjFunction *cfn = AS_CLOSURE(cv)->fn;
    if (cfn->jitFunc == NULL) { e->whyNot = "callee no longer compiled"; return false; }

    /* Straight to the callee's entry when everything jaiJitEnterFunc would
     * have checked can be checked here instead. Falling back rather than
     * declining matters: the descriptor path speaks a much wider language --
     * any argument kind, any module, a callee that writes -- and a call
     * through jaiCallValue still beats no compiled loop at all. */
    {
        const char *saved = e->whyNot;
        if (emitDirectCall(e, caller, cfn, cv, -1, cidx, argc, callOff,
                           after, false)) {
            return true;
        }
        if (e->failed) return false;   /* it had started emitting */
        e->whyNot = saved;
    }

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
                kinds[i] != SLOT_LIST && kinds[i] != SLOT_OBJ &&
                kinds[i] != SLOT_MAYBE_INST) {
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
            unsigned at = (unsigned)offsetof(ObjInstance, fields) +
                          (unsigned)fslots[i] * (unsigned)sizeof(Value);
            /* SCRATCH_C holds the instance, so the dynamic tag needs two
             * other scratches. */
            emitTagFor(e, kinds[i], regs[i], JIT_SCRATCH_A, JIT_SCRATCH_B);
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

/* Fold one more `return` into what this function is known to return.
 *
 * `build` in binary_trees ends `return null` on one path and `return Node(..)`
 * on the other, which is what an optional return type means, and demanding a
 * single kind refused it outright. An instance joined with a maybe-instance is
 * a maybe-instance; the shape survives only when both sides agree on it.
 *
 * This was written once before and reverted, because compiling `build` made the
 * program eleven times slower. That was not the merge: `build` is the first body
 * in the language to hold a freshly allocated object on the operand stack across
 * an allocating self-call, and at the time a self-call rooted nothing and the
 * operand-stack-to-register mapping in emitRootFill was wrong. Both are fixed,
 * and the same merge now makes binary_trees faster than the interpreter. */
static bool mergeReturnKind(Emit *e, SlotKind k, uint32_t shape) {
    if (!e->sawReturn) {
        e->sawReturn = true; e->returnKind = k; e->returnShape = shape;
        return true;
    }
    if (e->returnKind == k) {
        if (e->returnShape != shape) e->returnShape = 0;
        return true;
    }
    bool nullable = (e->returnKind == SLOT_INST && k == SLOT_MAYBE_INST) ||
                    (e->returnKind == SLOT_MAYBE_INST && k == SLOT_INST);
    if (!nullable) return false;
    if (e->returnShape != shape) e->returnShape = 0;
    e->returnKind = SLOT_MAYBE_INST;
    return true;
}

/* Opcodes that know how to take a float operand out of the FP bank. Every
 * other opcode sees the model exactly as it did before this existed, because
 * the dispatch loop materialises everything before handing it one. Adding an
 * opcode here without teaching it fpOperand is a miscompile, not a decline --
 * that is the whole risk of this design, and the reason the list is short. */
static bool fpFastOp(uint8_t op) {
    switch (op) {
    case OP_ADD: case OP_SUB: case OP_MUL: case OP_DIV:
    case OP_ADD_BIND: case OP_SUB_BIND: case OP_MUL_BIND:
    case OP_GET_LOCAL: case OP_GET_LOCAL2:
    case OP_ADD_LOCALS: case OP_BIND: case OP_SET_LOCAL:
    /* These three only ever write the entry they push, and the type guard at
     * a `float` boundary on something already float emits nothing at all --
     * it sat between the ADD and the BIND in mandelbrot's `x = x2 - y2 + x0`
     * and made the whole expression materialise for nothing. */
    case OP_CONST: case OP_INT: case OP_TYPE_GUARD:
    case OP_LT: case OP_LE: case OP_GT: case OP_GE:
    case OP_EQ: case OP_NE:
    case OP_JUMP_IF_CMP_FALSE:
    /* Reading an element writes only the entry it pushes and the four
     * scratch X registers; it touches no v register and calls nothing. Until
     * it was on this list an FP-resident accumulator was materialised the
     * moment a subscript appeared, which is every loop that sums over an
     * array -- `sum += ai[k] * b[k][j]` being the whole of matrix_mul. */
    case OP_GET_INDEX:
    /* Reading a field is the same argument as reading an element, and the
     * benchmark that wanted it is nbody: `bi.vx -= dx * bj.mass * mag` reads
     * three fields between the local it multiplies and the multiply itself,
     * so without these two on the list every operand of every field
     * expression materialised into an X register and came back through an
     * fmov. Neither opcode writes a v register, and neither reads an X
     * register whose FP copy could be the live one -- a receiver is an
     * instance and a local is always current in X. */
    case OP_GET_FIELD: case OP_GET_FIELD_LOCAL:
    /* Both take their operand out of the bank -- see fpConsumer -- so both
     * belong here too, or the dispatch loop syncs it back out first and the
     * arm never sees a live entry. `**0.5` is a square root and sat between
     * every `d2` and the divide that uses it. */
    case OP_SET_FIELD: case OP_POW:
        return true;
    default:
        return false;
    }
}

/* The instructions that actually read a float operand out of the FP bank.
 *
 * This used to be the whole of `fpWorthLoading`, tested against the very next
 * opcode on the reasoning that a bank entry only survives if its consumer is
 * adjacent. That was true when OP_GET_INDEX materialised everything; it is not
 * true now, and `fpWorthLoading` below walks to the consumer instead. */
static bool fpConsumer(uint8_t op) {
    switch (op) {
    case OP_ADD: case OP_SUB: case OP_MUL: case OP_DIV:
    case OP_ADD_BIND: case OP_SUB_BIND: case OP_MUL_BIND:
    case OP_BIND: case OP_SET_LOCAL:
    case OP_LT: case OP_LE: case OP_GT: case OP_GE:
    case OP_EQ: case OP_NE:
    case OP_JUMP_IF_CMP_FALSE:
    case OP_SET_FIELD: case OP_POW:
        return true;
    default:
        return false;
    }
}

/* How many entries below `idx` hold a register, which is what names the one
 * `idx` itself is in. Counted rather than subtracted from the top: a class, a
 * resolved function, a builtin and `self` occupy no register and can sit
 * anywhere -- a callee pushed below its own arguments puts one squarely in the
 * middle, which is exactly the shape an inlined call site has. */
static unsigned valueIndexOf(const Emit *e, unsigned idx) {
    unsigned seen = 0;
    for (unsigned i = 0; i < idx; i++) {
        if (holdsRegister(e->stack[i])) seen++;
    }
    return seen;
}

static bool pushCopyOfEntry(Emit *e, unsigned idx) {
    if (idx >= e->depth || !holdsRegister(e->stack[idx])) return false;
    unsigned vi = valueIndexOf(e, idx);
    fpSyncOne(e, vi);
    unsigned src = valueXReg(e, vi);
    if (!pushValue3(e, e->stack[idx], e->stackShape[idx], e->stackClass[idx],
                    e->stackSeen[idx], -1)) {
        return false;
    }
    unsigned dst = pushReg(e) - 1;
    if (dst != src) emit(e, jaiA64MovX(dst, src));
    return true;
}

/* The local opcodes of an inlined body, answered against its own frame.
 *
 * That frame is the caller's operand stack: a parameter is the argument entry
 * already sitting there, and a bind pins whatever is on top rather than
 * copying it anywhere. Reading these through the main switch instead would
 * read the CALLER's local of the same number, which is a different variable. */
static bool inlineLocalOp(Emit *e, const uint8_t *code, int off) {
    uint8_t op = code[off];
    unsigned a = jaiReadU16(code + off + 1);
    if (a > JIT_MAX_SLOTS) return false;

    if (op == OP_BIND) {
        if (e->inlSlot[a] >= 0) {
            /* Straight-line code binds each `let` once. A second bind would
             * have to move the value into the pinned entry's register, and
             * that register may sit below something live. */
            e->whyNot = "an inlined body binding a local twice";
            return false;
        }
        /* Pinning is only sound where the value being bound is the ONLY thing
         * above the region already pinned -- which is what `let x = expr` as a
         * statement looks like, the expression stack being empty at the start
         * of one. Two binds from one expression (`let a, b = ..`) would leave
         * this entry the top for both of them, and the two slots would name
         * one register: the aliasing mistake this tier keeps making, caught
         * here by measuring the depth rather than by trusting the shape. */
        if (e->depth != e->inlDepth + e->inlPinned + 1u) {
            e->whyNot = "an inlined body binding with an expression under it";
            return false;
        }
        if (!holdsRegister(e->stack[e->depth - 1])) return false;
        fpSyncOne(e, e->valueDepth - 1);
        e->inlSlot[a] = (int)(e->depth - 1);
        e->inlPinned++;
        return true;
    }

    if (e->inlSlot[a] < 0) {
        e->whyNot = "an inlined body reading a local it never bound";
        return false;
    }
    if (op == OP_GET_LOCAL) return pushCopyOfEntry(e, (unsigned)e->inlSlot[a]);

    unsigned b = jaiReadU16(code + off + 3);
    if (b > JIT_MAX_SLOTS || e->inlSlot[b] < 0) {
        e->whyNot = "an inlined body reading a local it never bound";
        return false;
    }
    if (op == OP_GET_LOCAL2) {
        return pushCopyOfEntry(e, (unsigned)e->inlSlot[a]) &&
               pushCopyOfEntry(e, (unsigned)e->inlSlot[b]);
    }

    /* OP_ADD_LOCALS */
    unsigned ia = (unsigned)e->inlSlot[a], ib = (unsigned)e->inlSlot[b];
    SlotKind k = e->stack[ia];
    if (k != e->stack[ib] || (k != SLOT_INT && k != SLOT_FLOAT)) return false;
    unsigned via = valueIndexOf(e, ia), vib = valueIndexOf(e, ib);
    fpSyncOne(e, via);
    fpSyncOne(e, vib);
    unsigned ra = valueXReg(e, via), rb = valueXReg(e, vib);
    if (!pushValue(e, k, 0, NULL)) return false;
    unsigned rd = pushReg(e) - 1;
    if (k == SLOT_FLOAT) {
        emit(e, jaiA64FmovDX(JIT_FSCRATCH_A, ra));
        emit(e, jaiA64FmovDX(JIT_FSCRATCH_B, rb));
        emit(e, jaiA64FaddD(JIT_FSCRATCH_A, JIT_FSCRATCH_A, JIT_FSCRATCH_B));
        emit(e, jaiA64FmovXD(rd, JIT_FSCRATCH_A));
    } else {
        emit(e, jaiA64AddsX(rd, ra, rb));
        branchOnOverflow(e, 0u, JAI_A64_VS);
    }
    return true;
}

/* Is a float operand worth putting in the FP bank rather than an X register?
 *
 * A *consumer* has to be reachable, not merely another opcode on the list.
 * The original rule looked only at the very next instruction, which said no
 * for `sum` in matrix_mul: it is followed by OP_GET_LOCAL2 and then the whole
 * of `ai[k] * b[k][j]` before any float operator. That cost the loop its
 * accumulator -- `str d` out, `ldr x` plus `fmov d, x` back in, twice a
 * cross-register-file move on the loop-carried chain, which measured as
 * `sum += 1.0` running at 15 cycles an iteration against 2 for `sum += 1`.
 *
 * So walk forward instead, through the instructions that leave an FP-resident
 * entry where it is (`fpFastOp`, which is the same set the main loop uses to
 * decide whether to sync), and answer on the first thing that is neither. The
 * walk is bounded: none of those opcodes is variable length, so
 * `jaiOpOperandSize` is enough and no Chunk is needed. */
/* Opcodes a float borrow may live across.
 *
 * Deliberately a whitelist and deliberately short: every one of these reads
 * float entries through fpOperand or fpHeldIn, and none of them records a
 * deopt while one is live -- an int overflow inside OP_ADD goes through
 * fpSyncAll, which materialises a borrow the same way it materialises a bank
 * entry. Anything else releases at the top of the instruction, where the
 * release is unconditionally on the path. deoptRecordAt declines if one gets
 * through anyway, so the cost of this list being wrong is a decline. */
static bool fpBorrowSurvives(uint8_t op) {
    switch (op) {
    case OP_GET_LOCAL: case OP_GET_LOCAL2:
    case OP_CONST: case OP_INT:
    case OP_ADD: case OP_SUB: case OP_MUL: case OP_DIV:
    case OP_ADD_BIND: case OP_SUB_BIND: case OP_MUL_BIND:
    case OP_BIND: case OP_SET_LOCAL:
    case OP_LT: case OP_LE: case OP_GT: case OP_GE:
    case OP_EQ: case OP_NE:
    case OP_JUMP_IF_CMP_FALSE:
        return true;
    default:
        return false;
    }
}

static bool fpWorthLoading(const Emit *e, const uint8_t *code, int next,
                           int stop) {
    if (e->fpOff) return false;
    for (unsigned step = 0; step < 24u && next < stop; step++) {
        uint8_t op = code[next];
        if (fpConsumer(op)) return true;
        if (!fpFastOp(op)) return false;
        int size = jaiOpOperandSize((OpCode)op);
        if (size < 0) return false;
        next += 1 + size;
    }
    return false;
}

static bool compileBody(Emit *e, ObjClosure *closure) {
    ObjFunction *fn = closure->fn;
    const uint8_t *code = fn->chunk.code;
    int count = fn->chunk.count;

    /* An inlined body is walked whole; the OSR window belongs to the caller. */
    int start = (!e->inlining && e->osr) ? (int)e->osrTop : 0;
    int stop  = (!e->inlining && e->osr) ? (int)e->osrEnd : count;
    bool afterUncond = false;
    /* The offset the walk visited before this one, for the arms that want to
     * know whether the value on top of the stack came from a literal. Updated
     * at the top rather than the bottom because arms leave the loop body by
     * `break`, by `continue` and by `return` alike. */
    int prevOff = -1, thisOff = -1;
    for (int off = start; off < stop && !e->failed;) {
        prevOff = thisOff;
        thisOff = off;
        if (afterUncond) reconcileAfterUncond(e, (uint32_t)off);
        afterUncond = (code[off] == OP_JUMP || code[off] == OP_LOOP);
        e->offsetToInst[off]  = (int)e->count;
        e->offsetToDepth[off] = (int)stackSignature(e);
        uint8_t op = code[off];
        e->lastOp = op;
        /* A borrow ends here unless the instruction is one of the few it is
         * allowed to live across. The top of an instruction is the one place
         * the release is guaranteed to be on the executed path. */
        if (e->fpBorrow != 0 && !fpBorrowSurvives(op)) fpReleaseAll(e);
        e->curOffset = (uint32_t)off;
        e->instDepth = e->depth;
        e->instValueDepth = e->valueDepth;

        if (e->fpLive != 0) {
            bool joinsHere = false;
            for (unsigned f = 0; f < e->fixupCount && !joinsHere; f++) {
                joinsHere = (e->fixups[f].targetOffset == (uint32_t)off);
            }
            /* Inside an inlined body the offsets are the callee's, so a
             * caller fixup that happens to name the same number is not a join
             * here at all -- and there are no branches in an inlined body for
             * one to be. */
            if (e->inlining) joinsHere = false;
            if (!fpFastOp(op) || joinsHere) {
                fpSyncAll(e);
            } else if (e->fpCarryCount < 64) {
                e->fpCarry[e->fpCarryCount++] = (uint32_t)off;
            } else {
                fpSyncAll(e);
            }
        }

        /* The four local opcodes an inlined body is allowed, answered against
         * its own frame -- entries on the caller's operand stack -- before the
         * main switch can read them as the caller's slot numbers. */
        if (e->inlining && (op == OP_GET_LOCAL || op == OP_GET_LOCAL2 ||
                            op == OP_ADD_LOCALS || op == OP_BIND)) {
            if (!inlineLocalOp(e, code, off)) return false;
            off += instructionLength(&fn->chunk, off);
            continue;
        }
        if (e->inlining && op == OP_RETURN) {
            /* The result is on top and stays there; the caller's driver takes
             * it from the model. */
            if (e->depth == 0 || !holdsRegister(e->stack[e->depth - 1])) {
                e->whyNot = "an inlined body returning a value with no register";
                return false;
            }
            fpSyncOne(e, e->valueDepth - 1);
            break;
        }

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
            if (e->localKind[slot] == SLOT_FLOAT && !e->dynamicLocal[slot] &&
                fpWorthLoading(e, code, off + 3, stop)) {
                unsigned idx = e->valueDepth - 1;
                if (e->osr && e->slotFpReg[slot] != 0) {
                    fpBorrowLocal(e, idx, e->slotFpReg[slot]);
                } else {
                    localInFp(e, slot, fpRegAt(e, idx));
                    fpClaim(e, idx);
                }
            } else {
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
            if (!e->fpOff && !e->dynamicLocal[slot] &&
                e->stack[e->depth - 1] == SLOT_FLOAT &&
                (e->fpLive & (1u << (e->valueDepth - 1)))) {
                localOutFp(e, slot, fpHeldIn(e, e->valueDepth - 1));
            } else {
                localOut(e, slot, pushReg(e) - 1);
            }
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
            if (ka2 == SLOT_FLOAT && !e->dynamicLocal[a] &&
                !e->dynamicLocal[b] && !e->fpOff) {
                unsigned idx = e->valueDepth - 1;
                /* Either operand already in a d register of its own is read
                 * from there. `x2 + y2` was two fmovs and an add. */
                unsigned da, db;
                if (e->osr && e->slotFpReg[a] != 0) {
                    da = e->slotFpReg[a];
                } else {
                    localInFp(e, a, fpRegAt(e, idx));
                    da = fpRegAt(e, idx);
                }
                if (e->osr && e->slotFpReg[b] != 0) {
                    db = e->slotFpReg[b];
                } else {
                    localInFp(e, b, JIT_FP_BANK + JIT_MAX_SAVED);
                    db = JIT_FP_BANK + JIT_MAX_SAVED;
                }
                emit(e, jaiA64FaddD(fpRegAt(e, idx), da, db));
                fpClaim(e, idx);
                off += 5;
                break;
            }
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
            if (!e->fpOff && e->depth >= 2 && !e->dynamicLocal[slot] &&
                e->stack[e->depth - 1] == SLOT_FLOAT &&
                e->stack[e->depth - 2] == SLOT_FLOAT) {
                /* The whole operation stays in the FP bank: two operands that
                 * are already there, one instruction, and a store straight out
                 * of a d register. */
                if (!adoptLocalKind(e, slot, SLOT_FLOAT, 0, NULL)) {
                    e->whyNot = "a local was given two different kinds";
                    return false;
                }
                unsigned ib = e->valueDepth - 1, ia = e->valueDepth - 2;
                unsigned db = fpOperand(e, ib), da = fpOperand(e, ia);
                unsigned rx; SlotKind kx;
                if (!popValueRaw(e, &rx, &kx)) return false;
                if (!popValueRaw(e, &rx, &kx)) return false;
                emit(e, jaiA64FaddD(fpRegAt(e, ia), da, db));
                localOutFp(e, slot, fpRegAt(e, ia));
                off += 3;
                break;
            }
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
            if (!e->fpOff && e->depth >= 2 && !e->dynamicLocal[slot] &&
                e->stack[e->depth - 1] == SLOT_FLOAT &&
                e->stack[e->depth - 2] == SLOT_FLOAT) {
                /* The whole operation stays in the FP bank: two operands that
                 * are already there, one instruction, and a store straight out
                 * of a d register. */
                if (!adoptLocalKind(e, slot, SLOT_FLOAT, 0, NULL)) {
                    e->whyNot = "a local was given two different kinds";
                    return false;
                }
                unsigned ib = e->valueDepth - 1, ia = e->valueDepth - 2;
                unsigned db = fpOperand(e, ib), da = fpOperand(e, ia);
                unsigned rx; SlotKind kx;
                if (!popValueRaw(e, &rx, &kx)) return false;
                if (!popValueRaw(e, &rx, &kx)) return false;
                emit(e, jaiA64FmulD(fpRegAt(e, ia), da, db));
                localOutFp(e, slot, fpRegAt(e, ia));
                off += 3;
                break;
            }
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
            if (!e->fpOff && e->depth >= 2 && !e->dynamicLocal[slot] &&
                e->stack[e->depth - 1] == SLOT_FLOAT &&
                e->stack[e->depth - 2] == SLOT_FLOAT) {
                /* The whole operation stays in the FP bank: two operands that
                 * are already there, one instruction, and a store straight out
                 * of a d register. */
                if (!adoptLocalKind(e, slot, SLOT_FLOAT, 0, NULL)) {
                    e->whyNot = "a local was given two different kinds";
                    return false;
                }
                unsigned ib = e->valueDepth - 1, ia = e->valueDepth - 2;
                unsigned db = fpOperand(e, ib), da = fpOperand(e, ia);
                unsigned rx; SlotKind kx;
                if (!popValueRaw(e, &rx, &kx)) return false;
                if (!popValueRaw(e, &rx, &kx)) return false;
                emit(e, jaiA64FsubD(fpRegAt(e, ia), da, db));
                localOutFp(e, slot, fpRegAt(e, ia));
                off += 3;
                break;
            }
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
            if (!e->fpOff && !e->dynamicLocal[slot] &&
                e->stack[e->depth - 1] == SLOT_FLOAT &&
                (e->fpLive & (1u << (e->valueDepth - 1)))) {
                unsigned idx = e->valueDepth - 1;
                unsigned held = fpHeldIn(e, idx);
                unsigned r2; SlotKind k2;
                if (!popValueRaw(e, &r2, &k2)) return false;
                localOutFp(e, slot, held);
                off += 3;
                break;
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
            /* `node == null` puts an instance beside a maybe-instance. Both
             * are a pointer or zero in a register, so the compare is the same
             * one; treat the pair as maybe-instance. */
            if (ka != kb) {
                bool mixable =
                    (ka == SLOT_INST && kb == SLOT_MAYBE_INST) ||
                    (ka == SLOT_MAYBE_INST && kb == SLOT_INST);
                if (!mixable) return false;
                ka = kb = SLOT_MAYBE_INST;
            }
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
                /* Both operands stay where they are: the compare reads the
                 * FP bank, and the NaN guard's stub knows how to write an
                 * FP-resident entry out if it is ever taken. */
                unsigned db = fpOperand(e, e->valueDepth - 1);
                unsigned da = fpOperand(e, e->valueDepth - 2);
                emit(e, jaiA64FcmpD(da, db));
                nanToDeopt(e);
            } else if (ka == SLOT_INT || ka == SLOT_MAYBE_INST) {
                emit(e, jaiA64SubsXReg(31, ra, rb));
            } else if ((op == OP_EQ || op == OP_NE) && ka == SLOT_OBJ &&
                       IS_STRING(e->stackSeen[e->depth - 2]) &&
                       IS_STRING(e->stackSeen[e->depth - 1])) {
                /* Two interned strings are equal exactly when they are the
                 * same object -- that is what interning buys, and it is what
                 * `text[i] == " "` is, since one-character strings are shared
                 * singletons. Anything not interned, or not a string, deopts
                 * and the interpreter compares properly. */
                for (unsigned side = 0; side < 2; side++) {
                    unsigned r = side == 0 ? ra : rb;
                    emit(e, jaiA64LdrW(JIT_SCRATCH_A, r,
                                       (unsigned)offsetof(Obj, type)));
                    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_STRING));
                    branchOnDeopt(e, JAI_A64_NE);
                    emit(e, jaiA64LdrByte(JIT_SCRATCH_A, r,
                                          (unsigned)offsetof(Obj, subFlag)));
                    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, 0));
                    branchOnDeopt(e, JAI_A64_EQ);
                }
                emit(e, jaiA64SubsXReg(31, ra, rb));
            } else {
                return false;
            }
            unsigned dropA, dropB;
            /* Raw: the compare has read both, they are going away, and
             * materialising them would put an fmov on the hot path for a
             * register nothing will read. */
            if (!popValueRaw(e, &dropB, NULL)) return false;
            if (!popValueRaw(e, &dropA, NULL)) return false;
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
            /* `node == null` puts an instance beside a maybe-instance. Both
             * are a pointer or zero in a register, so the compare is the same
             * one; treat the pair as maybe-instance. */
            if (ka != kb) {
                bool mixable =
                    (ka == SLOT_INST && kb == SLOT_MAYBE_INST) ||
                    (ka == SLOT_MAYBE_INST && kb == SLOT_INST);
                if (!mixable) return false;
                ka = kb = SLOT_MAYBE_INST;
            }
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
                /* Both operands stay where they are: the compare reads the
                 * FP bank, and the NaN guard's stub knows how to write an
                 * FP-resident entry out if it is ever taken. */
                unsigned db = fpOperand(e, e->valueDepth - 1);
                unsigned da = fpOperand(e, e->valueDepth - 2);
                emit(e, jaiA64FcmpD(da, db));
                nanToDeopt(e);
            } else if (ka == SLOT_INT || ka == SLOT_MAYBE_INST) {
                emit(e, jaiA64SubsXReg(31, ra, rb));
            } else if ((cmp == OP_EQ || cmp == OP_NE) && ka == SLOT_OBJ &&
                       IS_STRING(e->stackSeen[e->depth - 2]) &&
                       IS_STRING(e->stackSeen[e->depth - 1])) {
                /* Two interned strings are equal exactly when they are the
                 * same object -- that is what interning buys, and it is what
                 * `text[i] == " "` is, since one-character strings are shared
                 * singletons. Anything not interned, or not a string, deopts
                 * and the interpreter compares properly. */
                for (unsigned side = 0; side < 2; side++) {
                    unsigned r = side == 0 ? ra : rb;
                    emit(e, jaiA64LdrW(JIT_SCRATCH_A, r,
                                       (unsigned)offsetof(Obj, type)));
                    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_STRING));
                    branchOnDeopt(e, JAI_A64_NE);
                    emit(e, jaiA64LdrByte(JIT_SCRATCH_A, r,
                                          (unsigned)offsetof(Obj, subFlag)));
                    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, 0));
                    branchOnDeopt(e, JAI_A64_EQ);
                }
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

        case OP_NULL: {
            /* Zero, which is what a null instance is in a register. */
            if (!pushValue(e, SLOT_MAYBE_INST, 0, NULL)) return false;
            emit(e, jaiA64MovzX(pushReg(e) - 1, 0, 0));
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
                 * exactly as it does in a Value's payload -- unless the value
                 * is one FMOV's eight-bit immediate can name, in which case it
                 * goes straight to the FP bank in one instruction instead of
                 * two into an X register and an fmov out of it. */
                double d = AS_FLOAT(k);
                int64_t bits;
                memcpy(&bits, &d, sizeof bits);
                if (!pushValue3(e, SLOT_FLOAT, 0, NULL, k, -1)) return false;
                unsigned imm8;
                if (!e->fpOff && jaiA64FpImm8(d, &imm8)) {
                    unsigned idx = e->valueDepth - 1;
                    emit(e, jaiA64FmovDImm(fpRegAt(e, idx), imm8));
                    fpClaim(e, idx);
                } else {
                    emitConst64(e, pushReg(e) - 1, bits);
                }
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
            if (e->depth >= 2 && e->stack[e->depth - 1] == SLOT_FLOAT &&
                e->stack[e->depth - 2] == SLOT_FLOAT) {
                unsigned ib = e->valueDepth - 1, ia = e->valueDepth - 2;
                unsigned db = fpOperand(e, ib), da = fpOperand(e, ia);
                if (!popValueRaw(e, &rb, &kb)) return false;
                if (!popValueRaw(e, &ra, &ka)) return false;
                if (!pushValue(e, SLOT_FLOAT, 0, NULL)) return false;
                emit(e, jaiA64FmulD(fpRegAt(e, ia), da, db));
                fpClaim(e, ia);
                off += 1;
                break;
            }
            if (!popValue(e, &rb, &kb)) return false;
            if (!popValue(e, &ra, &ka)) return false;
            if (ka != kb) return false;
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

        case OP_BAND:
        case OP_BOR:
        case OP_BXOR:
        case OP_SHL:
        case OP_SHR: {
            /* The rest of the integer instruction set. None of `& | ^` can
             * fail, so they are three registers and one instruction.
             *
             * The shifts have two edges the hardware does not share: jaithon
             * throws on a negative count and saturates at 64 or more, while
             * arm64's LSLV and ASRV use the low six bits of the count and so
             * wrap. Both edges are guarded and deopt, which is right twice
             * over -- they are vanishingly rare, and the interpreter already
             * has the exact rule. The guards precede the pops, so a deopt
             * resumes at an instruction that has not happened. */
            unsigned rb, ra;
            SlotKind kb, ka;
            if (e->depth < 2) return false;
            if (e->stack[e->depth - 1] != SLOT_INT) return false;
            if (e->stack[e->depth - 2] != SLOT_INT) return false;

            /* A literal count settles both edges here rather than at run time,
             * and the immediate-form shift is then exactly the interpreter's
             * rule for that count: `<<` is the unsigned shift LSL performs and
             * `>>` the arithmetic one ASR performs, for every count in 0..63.
             * bitops shifts by 7, 3, 11, 1 and 31 and paid five instructions
             * and two deopt sites for each. */
            int64_t kcount;
            bool kcountUsable =
                (op == OP_SHL || op == OP_SHR) &&
                literalIntOperand(fn, prevOff, off, &kcount) &&
                kcount >= 0 && kcount <= 63;

            if ((op == OP_SHL || op == OP_SHR) && !kcountUsable) {
                unsigned rCount = pushReg(e) - 1;
                emit(e, jaiA64SubsXImm(31, rCount, 0));
                branchOnDeopt(e, JAI_A64_LT);            /* negative count */
                emitConst64(e, JIT_SCRATCH_A, 64);
                emit(e, jaiA64SubsXReg(31, rCount, JIT_SCRATCH_A));
                branchOnDeopt(e, JAI_A64_GE);            /* 64 or more */
            }

            if (!popValue(e, &rb, &kb)) return false;
            if (!popValue(e, &ra, &ka)) return false;
            if (!pushValue(e, SLOT_INT, 0, NULL)) return false;
            unsigned rd = pushReg(e) - 1;
            if (kcountUsable) {
                emit(e, op == OP_SHL ? jaiA64LslX(rd, ra, (unsigned)kcount)
                                     : jaiA64AsrX(rd, ra, (unsigned)kcount));
                off += 1;
                break;
            }
            emit(e, op == OP_BAND ? jaiA64AndX(rd, ra, rb)
                  : op == OP_BOR  ? jaiA64OrrX(rd, ra, rb)
                  : op == OP_BXOR ? jaiA64EorX(rd, ra, rb)
                  : op == OP_SHL  ? jaiA64LslvX(rd, ra, rb)
                                  : jaiA64AsrvX(rd, ra, rb));
            off += 1;
            break;
        }

        case OP_ADD:
        case OP_SUB:
        case OP_DIV: {
            unsigned rb, ra;
            SlotKind kb, ka;
            if (e->depth >= 2 && e->stack[e->depth - 1] == SLOT_FLOAT &&
                e->stack[e->depth - 2] == SLOT_FLOAT) {
                unsigned ib = e->valueDepth - 1, ia = e->valueDepth - 2;
                unsigned db = fpOperand(e, ib), da = fpOperand(e, ia);
                if (!popValueRaw(e, &rb, &kb)) return false;
                if (!popValueRaw(e, &ra, &ka)) return false;
                if (!pushValue(e, SLOT_FLOAT, 0, NULL)) return false;
                unsigned dd = fpRegAt(e, ia);
                emit(e, op == OP_ADD ? jaiA64FaddD(dd, da, db)
                     : op == OP_SUB  ? jaiA64FsubD(dd, da, db)
                                     : jaiA64FdivD(dd, da, db));
                fpClaim(e, ia);
                off += 1;
                break;
            }
            if (!popValue(e, &rb, &kb)) return false;
            if (!popValue(e, &ra, &ka)) return false;
            if (ka != kb) return false;   /* no implicit widening here */

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

            /* A maybe-instance reads like an instance once it is known not
             * to be null. The program has usually just tested it -- `if node
             * == null { return 0 }` -- but the tier does not track that, so
             * the guard stands and costs one compare against zero. A null
             * really arriving here deopts, and the interpreter raises the
             * error it would have raised anyway. */
            if (e->localKind[slot] != SLOT_INST &&
                e->localKind[slot] != SLOT_MAYBE_INST) {
                return false;
            }
            if (e->localClass[slot] == NULL) return false;
            if (e->localKind[slot] == SLOT_MAYBE_INST) {
                unsigned rcv = localIn(e, slot, JIT_SCRATCH_A);
                emit(e, jaiA64SubsXImm(31, rcv, 0));
                branchOnDeopt(e, JAI_A64_EQ);
            }
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
            ObjClass *fcls = NULL;
            if (IS_INT(fieldVal))        { kind = SLOT_INT;   tag = VAL_INT; }
            else if (IS_FLOAT(fieldVal)) { kind = SLOT_FLOAT; tag = VAL_FLOAT; }
            else if (IS_INSTANCE(fieldVal) || IS_NULL(fieldVal)) {
                /* A tree's `left` is a Node on one call and null on the next,
                 * and its leaves are half of it. A leaf has no class to read,
                 * so the receiver's own class is the guess -- a nullable
                 * instance field is a linked structure far more often than
                 * not -- and the class guard below makes the guess safe: being
                 * wrong deopts, it does not miscompile. */
                kind = SLOT_MAYBE_INST;
                tag  = VAL_OBJ;
                fcls = IS_INSTANCE(fieldVal) ? AS_INSTANCE(fieldVal)->klass
                                             : e->localClass[slot];
                if (fcls == NULL) return false;
            }
            else return false;

            unsigned base = (unsigned)offsetof(ObjInstance, fields) +
                            (unsigned)info->slot * (unsigned)sizeof(Value);
            /* The tag is checked every time, unless this body wrote the field
             * itself. A field is not typed by the runtime, so a later int
             * where a float was seen must bail rather than be read as one. */
            unsigned recv = localIn(e, slot, JIT_SCRATCH_C);
            SlotKind already = knownFieldKind(e, (int)slot, info->slot);
            if (kind == SLOT_MAYBE_INST) {
                /* Three guards, and the receiver is the trick that makes them
                 * branch-free. It is a live instance already -- the entry
                 * guard or the null check above saw to that -- so when the
                 * field is null the loads below are aimed at the receiver
                 * instead, which is a real object of the right type. Nothing
                 * ever dereferences zero. */
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, recv, base));       /* tag */
                emit(e, jaiA64LdrX(JIT_SCRATCH_D, recv, base + 8));   /* value */
                emit(e, jaiA64MovzX(JIT_SCRATCH_B, 0, 0));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, VAL_OBJ));
                emit(e, jaiA64CselX(JIT_SCRATCH_D, JIT_SCRATCH_D,
                                    JIT_SCRATCH_B, JAI_A64_EQ));
                emit(e, jaiA64CselX(JIT_SCRATCH_B, JIT_SCRATCH_B,
                                    JIT_SCRATCH_A, JAI_A64_EQ));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_B, 0));
                branchOnDeopt(e, JAI_A64_NE);      /* neither object nor null */

                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_D, 0));
                emit(e, jaiA64CselX(JIT_SCRATCH_B, recv, JIT_SCRATCH_D,
                                    JAI_A64_EQ));
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_B,
                                   (unsigned)offsetof(Obj, type)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_INSTANCE));
                branchOnDeopt(e, JAI_A64_NE);

                emit(e, jaiA64LdrX(JIT_SCRATCH_A, JIT_SCRATCH_B,
                                   (unsigned)offsetof(ObjInstance, klass)));
                emitConst64(e, JIT_SCRATCH_B, (int64_t)(uintptr_t)fcls);
                if (fcls != e->localClass[slot]) {
                    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_D, 0));
                    emit(e, jaiA64CselX(JIT_SCRATCH_A, JIT_SCRATCH_B,
                                        JIT_SCRATCH_A, JAI_A64_EQ));
                }
                emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
                branchOnDeopt(e, JAI_A64_NE);
            } else if (already != SLOT_SELF) {
                kind = already;
            } else {
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, recv, base));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, tag));
                branchOnDeopt(e, JAI_A64_NE);
            }

            if (kind == SLOT_MAYBE_INST) {
                if (!pushValue3(e, kind, fcls->shapeId, fcls,
                                IS_INSTANCE(fieldVal) ? fieldVal : NULL_VAL,
                                -1)) {
                    return false;
                }
                emit(e, jaiA64MovX(pushReg(e) - 1, JIT_SCRATCH_D));
            } else {
                if (!pushValue(e, kind, 0, NULL)) return false;
                /* A float field goes straight to the FP bank when something
                 * downstream will read it there: `ldr d` instead of `ldr x`
                 * followed by the `fmov` its consumer would emit. */
                if (kind == SLOT_FLOAT &&
                    fpWorthLoading(e, code, off + 8, stop)) {
                    unsigned idx = e->valueDepth - 1;
                    emit(e, jaiA64LdrD(fpRegAt(e, idx), recv, base + 8));
                    fpClaim(e, idx);
                } else {
                    emit(e, jaiA64LdrX(pushReg(e) - 1, recv, base + 8));
                }
            }
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
                if (e->localKind[slot] == SLOT_FLOAT &&
                    !e->dynamicLocal[slot] &&
                    fpWorthLoading(e, code, off + 5, stop)) {
                    unsigned idx = e->valueDepth - 1;
                    if (e->osr && e->slotFpReg[slot] != 0) {
                        fpBorrowLocal(e, idx, e->slotFpReg[slot]);
                    } else {
                        localInFp(e, slot, fpRegAt(e, idx));
                        fpClaim(e, idx);
                    }
                } else {
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
            /* A float already in the FP bank is stored from there: `str d`
             * rather than the `fmov x, d` popValue would emit followed by
             * `str x`. Captured before the pop, because popping is what clears
             * the bit and renames the index. */
            bool vIsFp = e->depth >= 1 && e->valueDepth >= 1 &&
                         e->stack[e->depth - 1] == SLOT_FLOAT &&
                         (e->fpLive & (1u << (e->valueDepth - 1))) != 0;
            unsigned dv = vIsFp ? fpHeldIn(e, e->valueDepth - 1) : 0u;
            if (vIsFp) {
                if (!popValueRaw(e, &rv, &kv)) return false;
            } else if (!popValue(e, &rv, &kv)) {
                return false;
            }
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
            if (vIsFp) emit(e, jaiA64StrD(dv, rr, base + 8));
            else       emit(e, jaiA64StrX(rv, rr, base + 8));
            recordFieldStore(e, recvLocal, info->slot, kv);
            e->wroteHeap = true;
            off += 6;
            break;
        }

        case OP_RETURN_NULL: {
            /* An OSR form's x0 is the bytecode offset to resume at, not a
             * value -- see jaiJitEnterOsr, which does `*resumeAt = at`. The
             * return sequence below is the function tier's and leaves the
             * returned value in x0, so a `return` inside a compiled loop hands
             * the interpreter an integer or a pointer as an instruction
             * offset. `for i in 0..n { if .. { return x } }` miscompiled that
             * way in released builds: 14 runs in 20 wrong, and 10 in 10 under
             * --gc-stress. */
            if (e->osr) {
                e->whyNot = "a return inside an OSR loop";
                return false;
            }
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
            if (kind == SLOT_FLOAT && fpWorthLoading(e, code, off + 6, stop)) {
                unsigned idx = e->valueDepth - 1;
                emit(e, jaiA64LdrD(fpRegAt(e, idx), rr, fbase + 8));
                fpClaim(e, idx);
            } else {
                emit(e, jaiA64LdrX(pushReg(e) - 1, rr, fbase + 8));
            }
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
            /* Wholly in the FP bank. The sign bit is the one thing that needs
             * an integer register, and taking it with an `fmov` out of the d
             * register costs one instruction where routing the operand
             * through X cost four -- two for the exponent constant nothing
             * reads, two more around the fsqrt. */
            unsigned ia = e->valueDepth - 2;
            unsigned da = fpOperand(e, ia);
            emit(e, jaiA64FmovXD(JIT_SCRATCH_A, da));
            emit(e, jaiA64LsrX(JIT_SCRATCH_A, JIT_SCRATCH_A, 63));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, 0));
            branchOnDeopt(e, JAI_A64_NE);

            unsigned dp1, dp2;
            if (!popValueRaw(e, &dp1, NULL)) return false;
            if (!popValueRaw(e, &dp2, NULL)) return false;
            if (!pushValue(e, SLOT_FLOAT, 0, NULL)) return false;
            {
                unsigned idx = e->valueDepth - 1;
                emit(e, jaiA64FsqrtD(fpRegAt(e, idx), da));
                fpClaim(e, idx);
            }
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

            /* A literal modulus decides both guards, and a literal power of
             * two decides the whole thing. Floor remainder by a positive m is
             * the unique r in [0, m) with x = m*q + r; for m = 2^s that is the
             * low s bits, which two's complement already holds -- so it is
             * exact for negative dividends, where the truncating `msub` below
             * is not and the correction after it exists to fix that. */
            int64_t kmod;
            bool kmodKnown = literalIntOperand(fn, prevOff, off, &kmod) &&
                             kmod != 0 && kmod != -1;
            unsigned mshift;
            if (kmodKnown && powerOfTwoShift(kmod, &mshift)) {
                unsigned q1, q2;
                if (!popValue(e, &q1, NULL)) return false;
                if (!popValue(e, &q2, NULL)) return false;
                if (!pushValue(e, SLOT_INT, 0, NULL)) return false;
                unsigned rd = pushReg(e) - 1;
                if (mshift == 0) emitConst64(e, rd, 0);   /* x %% 1 is 0 */
                else emit(e, jaiA64AndXOnes(rd, rx, mshift));
                off += 1;
                break;
            }

            if (!kmodKnown) {
                emit(e, jaiA64SubsXImm(31, ry, 0));
                branchOnDeopt(e, JAI_A64_EQ);
                emit(e, jaiA64AddXImm(JIT_SCRATCH_A, ry, 1));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, 0));
                branchOnDeopt(e, JAI_A64_EQ);
            }

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

            /* A power of two is the low bits and nothing else -- see OP_MOD. */
            unsigned kshift;
            if (powerOfTwoShift(imm, &kshift)) {
                if (kshift == 0) emitConst64(e, rx, 0);
                else emit(e, jaiA64AndXOnes(rx, rx, kshift));
                off += 3;
                break;
            }

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

            /* A literal divisor decides both guards at compile time, and a
             * literal power of two decides the whole thing: floor(x / 2^s) is
             * `asr x, #s` for every int64 x, negative ones included, because
             * asr rounds toward minus infinity and that is precisely what the
             * correction below exists to reproduce. Fifteen instructions and
             * two deopt sites become one instruction and none. */
            int64_t kdiv;
            bool kdivKnown = literalIntOperand(fn, prevOff, off, &kdiv) &&
                             kdiv != 0 && kdiv != -1;
            unsigned dshift;
            if (kdivKnown && powerOfTwoShift(kdiv, &dshift)) {
                unsigned p1, p2;
                if (!popValue(e, &p1, NULL)) return false;
                if (!popValue(e, &p2, NULL)) return false;
                if (!pushValue(e, SLOT_INT, 0, NULL)) return false;
                emit(e, jaiA64AsrX(pushReg(e) - 1, ra, dshift));
                off += 1;
                break;
            }

            /* Zero, and -1 with it: the interpreter reports the
             * division-by-zero, and INT64_MIN / -1 is the one quotient that
             * does not fit. Both are rare enough that declining -1 outright
             * costs nothing and removes the special case. A literal divisor
             * has already answered both. */
            if (!kdivKnown) {
                emit(e, jaiA64SubsXImm(31, rb, 0));
                branchOnDeopt(e, JAI_A64_EQ);
                emit(e, jaiA64AddXImm(JIT_SCRATCH_A, rb, 1));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, 0));
                branchOnDeopt(e, JAI_A64_EQ);
            }

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
                 * spent all of it on the call. A full list goes out to a
                 * three-argument realloc helper and comes straight back --
                 * see the `grow` stubs, and the note there for why this used
                 * to be a deopt and what that cost. */
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
                if (e->growCount >= JIT_MAX_GROW) {
                    e->whyNot = "more list pushes than the tier tracks";
                    return false;
                }
                {
                    unsigned gi = e->growCount++;
                    e->grow[gi].listReg  = rList;
                    e->grow[gi].valReg   = rVal;
                    e->grow[gi].tag      = vtag;
                    e->grow[gi].countReg = JIT_SCRATCH_A;
                    e->grow[gi].stub     = -1;
                    if (e->fixupCount >= JIT_MAX_FIXUPS) {
                        e->failed = true;
                        return false;
                    }
                    e->fixups[e->fixupCount].instIndex    = (int)e->count;
                    e->fixups[e->fixupCount].targetOffset = FIXUP_GROW - gi;
                    e->fixups[e->fixupCount].conditional  = true;
                    e->fixups[e->fixupCount].depth        = -1;
                    e->fixupCount++;
                    emit(e, jaiA64BCond(JAI_A64_GE, 0));
                    e->grow[gi].returnTo = (int)e->count;
                }

                emit(e, jaiA64LdrX(JIT_SCRATCH_C, rList,
                                   (unsigned)offsetof(ObjList, items)));
                emit(e, jaiA64AddXLsl(JIT_SCRATCH_C, JIT_SCRATCH_C,
                                      JIT_SCRATCH_A, 4));
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

                /* Straight to the method's compiled entry when the receiver's
                 * class is fixed here -- which it is, or this arm would not
                 * have been reached: SLOT_INST carries a shape the caller has
                 * already guarded, so the "inline cache" for the dispatch is
                 * the model itself and costs nothing at run time.
                 *
                 * What that skips is the whole of jitInvokeMethod ->
                 * jaiCallMethodWithReceiver -> invokeCallable -> callClosure
                 * -> jaiJitEnterFunc -> jitArgIn, which is C glue between two
                 * compiled bodies. On object_dispatch, by `sample`, those five
                 * were 43% of the benchmark.
                 *
                 * Falling back rather than declining matters for the same
                 * reason it does for a global call: the descriptor path speaks
                 * a wider language. */
                {
                    const char *saved = e->whyNot;
                    if (emitDirectCall(e, fn, mfn, method, -1, ridx, argc,
                                       (uint32_t)off, (uint32_t)(off + 7),
                                       true)) {
                        if (getenv("JAI_JIT_WHY")) {
                            fprintf(stderr, "[jit] direct method %s.%s at %d\n",
                                    rcls->name ? rcls->name->chars : "?",
                                    AS_STRING(mname)->chars, off);
                        }
                        off += 7;
                        break;
                    }
                    if (e->failed) return false;   /* it had started emitting */
                    e->whyNot = saved;
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

            /* `xs.len()` on a list is one field. Going out through the
             * descriptor for it meant a GC root push and pop, a bound-method
             * resolve, an arity check and a native call, all to read a 32-bit
             * count -- and `while i < xs.len()` is the ordinary loop in this
             * language, so that was paid once per iteration. The receiver's
             * kind is already known here, which is the whole guard needed. */
            if (isLen && e->stack[ridx] == SLOT_LIST) {
                unsigned rList = pushReg(e) - 1;
                unsigned r;
                if (!popValue(e, &r, NULL)) return false;
                if (!pushValue(e, SLOT_INT, 0, NULL)) return false;
                emit(e, jaiA64LdrW(pushReg(e) - 1, rList,
                                   (unsigned)offsetof(ObjList, count)));
                off += 7;
                break;
            }

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
            /* Only the loop at the OSR entry point owns the reserved
             * iterator registers. A FOR_ITER_BIND nested inside it built its
             * own iterator and is an ordinary one -- refusing it stopped the
             * outer loops of spectral, mandelbrot, matrix_mul and life from
             * compiling at all, while their inner loops compiled fine. */
            if (!e->osr || !e->hasIter || (uint32_t)off != e->osrTop) {
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

            if (e->iterKind == 2) {
                /* A list at the loop head. The reserved registers mean what
                 * they do for a range except that JIT_START_REG holds the
                 * ObjList rather than a first value. Without this a top-level
                 * `for x in xs` was never even attempted -- the gate refused
                 * anything that was not a range before compileOsr was called,
                 * so not even a decline was recorded. */
                Value sample = e->elemSample;
                SlotKind ek;
                unsigned etag;
                uint32_t esh = 0;
                ObjClass *ecl = NULL;
                if (IS_INT(sample))        { ek = SLOT_INT;   etag = VAL_INT; }
                else if (IS_FLOAT(sample)) { ek = SLOT_FLOAT; etag = VAL_FLOAT; }
                else if (IS_BOOL(sample))  { ek = SLOT_BOOL;  etag = VAL_BOOL; }
                else if (IS_INSTANCE(sample) &&
                         AS_INSTANCE(sample)->klass != NULL) {
                    /* The object type is checked before the class is read,
                     * because VAL_OBJ is every heap object and a list holding
                     * a string beside the sampled instance would otherwise
                     * read `klass` one word past an ObjString's header. */
                    ek = SLOT_INST; etag = VAL_OBJ;
                    ecl = AS_INSTANCE(sample)->klass;
                    esh = ecl->shapeId;
                } else {
                    e->whyNot = "list element kind unknown";
                    return false;
                }
                if (!adoptLocalKindSeen(e, slot, ek, esh, ecl, sample)) {
                    e->whyNot = "loop variable took two kinds";
                    return false;
                }
                e->iterSlot = slot;
                e->iterExit = (uint32_t)((int32_t)(off + 5) + jump);

                /* Mutation first: a list that grew or shrank under the loop
                 * must raise, and the version is the only thing that says so.
                 * Nothing has happened yet, so this resumes at this very
                 * instruction and the interpreter raises it properly. */
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_START_REG,
                                   (unsigned)offsetof(ObjList, version)));
                emit(e, jaiA64LdrW(JIT_SCRATCH_B, JIT_ITER_REG,
                                   (unsigned)offsetof(ObjIter, version)));
                emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
                branchOnDeopt(e, JAI_A64_NE);

                emit(e, jaiA64SubsXReg(31, JIT_IDX_REG, JIT_LIM_REG));
                branchTo(e, e->iterExit, true, JAI_A64_GE);

                /* Reload items each time rather than hoisting: a reallocation
                 * bumps the version so the guard above covers it, and one ldr
                 * removes the question entirely. */
                emit(e, jaiA64LdrX(JIT_SCRATCH_C, JIT_START_REG,
                                   (unsigned)offsetof(ObjList, items)));
                emit(e, jaiA64AddXLsl(JIT_SCRATCH_C, JIT_SCRATCH_C,
                                      JIT_IDX_REG, 4));

                emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_C, 0));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, etag));
                branchOnDeopt(e, JAI_A64_NE);

                if (ek == SLOT_INST) {
                    emit(e, jaiA64LdrX(JIT_SCRATCH_D, JIT_SCRATCH_C, 8));
                    /* VAL_OBJ is every heap object, not just an instance. A
                     * list holding a string beside the instance the sample saw
                     * passes the tag test, and reading `klass` off an ObjString
                     * is a load one word past its header -- which is a live
                     * pointer in several subtypes, so it does not fault, it
                     * just answers wrongly. Check the type first. */
                    emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_D,
                                       (unsigned)offsetof(Obj, type)));
                    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_INSTANCE));
                    branchOnDeopt(e, JAI_A64_NE);
                    emit(e, jaiA64LdrX(JIT_SCRATCH_D, JIT_SCRATCH_D,
                                       (unsigned)offsetof(ObjInstance, klass)));
                    emit(e, jaiA64LdrW(JIT_SCRATCH_D, JIT_SCRATCH_D,
                                       (unsigned)offsetof(ObjClass, shapeId)));
                    emitConst64(e, JIT_SCRATCH_A, (int64_t)esh);
                    emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_D, JIT_SCRATCH_A));
                    branchOnDeopt(e, JAI_A64_NE);
                }

                /* One byte for a bool: see the note in OP_GET_INDEX. `strb` is
                 * what BOOL_VAL compiles to, so the rest of the payload word is
                 * stale, and a SLOT_BOOL register must hold 0 or 1. */
                if (ek == SLOT_BOOL) {
                    emit(e, jaiA64LdrByte(JIT_SCRATCH_A, JIT_SCRATCH_C, 8));
                } else {
                    emit(e, jaiA64LdrX(JIT_SCRATCH_A, JIT_SCRATCH_C, 8));
                }
                localOut(e, slot, JIT_SCRATCH_A);
                emit(e, jaiA64AddXImm(JIT_IDX_REG, JIT_IDX_REG, 1));
                off += 5;
                break;
            }

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
            /* `s[i]` on a string. Every guard is a load and a compare, and the
             * result is a table lookup rather than an allocation, because the
             * 128 one-byte strings are made once and shared. Without this the
             * whole loop around a character scan declines, which is why
             * `str_search` ran interpreted end to end -- and a scanner reading
             * one byte at a time is the shape of every lexer. */
            if (e->depth >= 2 && e->stack[e->depth - 1] == SLOT_INT &&
                e->stack[e->depth - 2] == SLOT_OBJ &&
                IS_STRING(e->stackSeen[e->depth - 2])) {
                unsigned rIdx = pushReg(e) - 1, rStr = pushReg(e) - 2;

                /* Really a string, and not something else this object slot
                 * happened to hold when the loop was compiled. */
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, rStr,
                                   (unsigned)offsetof(Obj, type)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_STRING));
                branchOnDeopt(e, JAI_A64_NE);

                /* ASCII only: one scalar is one byte, so indexing is indexing.
                 * `scalars` is UINT32_MAX until something asks, so the first
                 * time through deopts and the interpreter fills it in. */
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, rStr,
                                   (unsigned)offsetof(ObjString, length)));
                emit(e, jaiA64LdrW(JIT_SCRATCH_B, rStr,
                                   (unsigned)offsetof(ObjString, scalars)));
                emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
                branchOnDeopt(e, JAI_A64_NE);

                /* jaiNormalizeIndex, then one unsigned compare for both ends. */
                emitBoundsNormalise(e, rIdx, JIT_SCRATCH_A, JIT_SCRATCH_B);

                emit(e, jaiA64LdrX(JIT_SCRATCH_C, rStr,
                                   (unsigned)offsetof(ObjString, chars)));
                emit(e, jaiA64AddX(JIT_SCRATCH_C, JIT_SCRATCH_C, JIT_SCRATCH_B));
                emit(e, jaiA64LdrByte(JIT_SCRATCH_A, JIT_SCRATCH_C, 0));
                emitConst64(e, JIT_SCRATCH_B, 128);
                emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
                branchOnDeopt(e, JAI_A64_HS);

                /* The shared one-byte string. NULL until first asked for, so a
                 * character this program has not seen deopts once. */
                emitConst64(e, JIT_SCRATCH_C,
                            (int64_t)(uintptr_t)jaiAsciiCharTable());
                emit(e, jaiA64LslX(JIT_SCRATCH_B, JIT_SCRATCH_A, 3));
                emit(e, jaiA64AddX(JIT_SCRATCH_C, JIT_SCRATCH_C, JIT_SCRATCH_B));
                emit(e, jaiA64LdrX(JIT_SCRATCH_C, JIT_SCRATCH_C, 0));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_C, 0));
                branchOnDeopt(e, JAI_A64_EQ);

                /* Carry a sample so later instructions know this is a
                 * string: the receiver serves, since only its type is read.
                 * Without one the interned-equality path below cannot tell
                 * what it is holding and declines. */
                Value strSample = e->stackSeen[e->depth - 2];
                unsigned d1, d2;
                if (!popValue(e, &d1, NULL)) return false;
                if (!popValue(e, &d2, NULL)) return false;
                if (!pushValue3(e, SLOT_OBJ, 0, NULL, strSample, -1)) {
                    return false;
                }
                emit(e, jaiA64MovX(pushReg(e) - 1, JIT_SCRATCH_C));
                off += 1;
                break;
            }
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
            else if (IS_BOOL(elem))  { kind = SLOT_BOOL;  tag = VAL_BOOL; }
            else if (IS_LIST(elem)) {
                /* A list of lists. `matrix_mul` is `b[k][j]` in its innermost
                 * loop and could not compile the outer half of it. */
                kind = SLOT_LIST;
                tag = VAL_OBJ;
            }
            else if (IS_STRING(elem)) {
                /* A list of strings, held opaquely. Everything that then wants
                 * to know it is a string -- the interned compare, `s[i]` --
                 * checks OBJ_STRING for itself, which is the same contract a
                 * SLOT_OBJ local has always had: the sample says what to
                 * specialise for, the guard says whether it was right.
                 * `str_search` builds its text out of `chunks[seed % 8]` and
                 * declined that whole loop forty times over. */
                kind = SLOT_OBJ;
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
            emitBoundsNormalise(e, rIdx, JIT_SCRATCH_A, JIT_SCRATCH_B);

            emit(e, jaiA64LdrX(JIT_SCRATCH_C, rList,
                               (unsigned)offsetof(ObjList, items)));
            emit(e, jaiA64AddXLsl(JIT_SCRATCH_C, JIT_SCRATCH_C,
                                  JIT_SCRATCH_B, 4));
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
            /* A bool's payload is one byte: `BOOL_VAL` compiles to `strb`, so
             * the seven bytes above it are whatever the slot held before. An
             * 8-byte load would put that in a register a SLOT_BOOL is required
             * to hold 0 or 1 in, and `if xs[i]` is `cbnz` on the whole word. */
            if (kind == SLOT_BOOL) {
                emit(e, jaiA64LdrByte(pushReg(e) - 1, JIT_SCRATCH_C, 8));
            } else if (kind == SLOT_FLOAT &&
                       fpWorthLoading(e, code, off + 1, stop)) {
                /* Straight into the FP bank, for the same reason a float local
                 * goes there: `ldr x` followed by `fmov d, x` puts a
                 * cross-register-file move between the load and the multiply
                 * that wants it, and `ai[k] * b[k][j]` had two of them. */
                unsigned idx = e->valueDepth - 1;
                emit(e, jaiA64LdrD(fpRegAt(e, idx), JIT_SCRATCH_C, 8));
                fpClaim(e, idx);
            } else {
                emit(e, jaiA64LdrX(pushReg(e) - 1, JIT_SCRATCH_C, 8));
            }
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
            emitBoundsNormalise(e, rIdx, JIT_SCRATCH_A, JIT_SCRATCH_B);

            emit(e, jaiA64LdrX(JIT_SCRATCH_C, rList,
                               (unsigned)offsetof(ObjList, items)));
            emit(e, jaiA64AddXLsl(JIT_SCRATCH_C, JIT_SCRATCH_C,
                                  JIT_SCRATCH_B, 4));
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

        case OP_GET_SLICE: {
            /* `xs[a:b]` out to the runtime: the clamp arithmetic has three
             * throwing exits and lives in one place, and the work itself is an
             * O(n) copy against which the descriptor's stores are noise. */
            unsigned flags = code[off + 1];
            unsigned nops  = ((flags & 1u) != 0) + ((flags & 2u) != 0) +
                             ((flags & 4u) != 0);
            unsigned nargs = 1u + nops;
            if (e->depth < nargs) return false;
            unsigned cidx = e->depth - nargs;
            if (e->stack[cidx] != SLOT_LIST) {
                e->whyNot = "slicing a container this tier does not model";
                return false;
            }
            Value cseen = e->stackSeen[cidx];

            /* Guard the container, not the result: with its object type pinned
             * the arm jaiSliceGet takes is settled, so the result's kind
             * follows. Before the descriptor and before any pop, so a miss
             * resumes here with everything still on the interpreter's stack. */
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, pushReg(e) - nargs,
                               (unsigned)offsetof(Obj, type)));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_LIST));
            branchOnDeopt(e, JAI_A64_NE);

            emit(e, jaiA64MovzX(JIT_SCRATCH_A, flags, 0));
            emit(e, jaiA64StrX(JIT_SCRATCH_A, 31,
                               e->descOffset +
                                   (unsigned)offsetof(JitCallDesc, aux)));
            if (!emitDescriptorStatus(e, NULL_VAL, cidx, nargs,
                                      (void *)&jitGetSlice, false, -1)) {
                return false;
            }
            for (unsigned i = 0; i < nargs; i++) {
                unsigned r;
                if (!popValue(e, &r, NULL)) return false;
            }
            /* The container's own sample types the slice: a slice of a list of
             * ints is a list of ints, and every element read re-checks its own
             * tag, so this is a hint and not an assumption. */
            if (!pushValue3(e, SLOT_LIST, 0, NULL, cseen, -1)) return false;
            emit(e, jaiA64LdrX(pushReg(e) - 1, 31,
                               e->descOffset +
                                   (unsigned)offsetof(JitCallDesc, result) + 8));
            /* Deliberately not e->wroteHeap: the only effect is a fresh object
             * and an interpreted re-run would make another. Setting it would
             * decline the next self-call, which is the shape `sort` has. */
            off += 2;
            break;
        }

        case OP_GET_GLOBAL: {
            /* TWO WAYS OUT OF HERE, and they carry different obligations.
             *
             * BY VALUE: globalIsSelf, globalClass, globalFunction and
             * globalNative resolve the binding at compile time and bake it in.
             * Nothing re-checks it at run time, so ObjModule::version has to
             * retire the whole form when such a binding could have changed --
             * which is what jaiModuleSet's jaiValueIsInertGlobal test decides.
             * Teach any of those four a further kind of value, or teach this
             * arm to constant-fold an int out of `let ITERS = ..`, and
             * jaiValueIsInertGlobal must stop calling that kind inert.
             *
             * BY ADDRESS: the value case below bakes the JaiEntry*, not the
             * value, and re-loads it behind a tag guard -- plus an Obj.type
             * guard and a class-shape guard where the kind needs one -- on
             * EVERY access. It depends on the table's layout, guarded by
             * JaiTable::keyVersion, and on nothing ObjModule::version
             * protects. */
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
                if (e->depth >= JIT_MAX_STACK) return false;
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
                    /* A global holding a plain value -- an int, a float, a
                     * list, an instance. Its storage is a JaiEntry whose
                     * address is fixed once the entry exists, so the load is
                     * one `ldr` behind two guards: the table has not moved,
                     * and the value still has the kind this was compiled for.
                     *
                     * Refusing this is what declined every loop that so much
                     * as reads a module-level variable, which since the
                     * benchmarks moved to module scope is most of them. */
                    Value gvv = NULL_VAL;
                    JaiEntry *gslot = globalSlot(e, closure, nameIdx, &gvv);
                    SlotKind gk = SLOT_OPAQUE;
                    uint32_t gshape = 0;
                    ObjClass *gcls = NULL;
                    if (gslot != NULL && globalKind(gvv, &gk, &gshape, &gcls)) {
                        unsigned dst = pushReg(e);
                        unsigned tag = gk == SLOT_INT   ? VAL_INT
                                     : gk == SLOT_FLOAT ? VAL_FLOAT
                                     : gk == SLOT_BOOL  ? VAL_BOOL
                                                        : VAL_OBJ;
                        /* Every guard runs against the model as it is BEFORE
                         * the value is pushed, so a deopt here resumes at this
                         * instruction with the operand stack the interpreter
                         * expects. */
                        emitGlobalsGuard(e);
                        emitConst64(e, JIT_SCRATCH_D,
                                    (int64_t)(uintptr_t)gslot);
                        emit(e, jaiA64LdrW(JIT_SCRATCH_C, JIT_SCRATCH_D,
                                           (unsigned)offsetof(JaiEntry, value)));
                        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_C, tag));
                        branchOnDeopt(e, JAI_A64_NE);
                        emit(e, jaiA64LdrX(JIT_SCRATCH_C, JIT_SCRATCH_D,
                                           (unsigned)offsetof(JaiEntry, value) + 8u));
                        if (gk == SLOT_INST || gk == SLOT_LIST) {
                            /* VAL_OBJ is every heap object, so the tag alone
                             * does not say this is an instance -- reading a
                             * class pointer off an ObjString answers wrongly
                             * rather than faulting. */
                            emit(e, jaiA64LdrByte(JIT_SCRATCH_B, JIT_SCRATCH_C,
                                                  (unsigned)offsetof(Obj, type)));
                            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_B,
                                                   gk == SLOT_INST ? OBJ_INSTANCE
                                                                   : OBJ_LIST));
                            branchOnDeopt(e, JAI_A64_NE);
                        }
                        if (gk == SLOT_INST) {
                            emit(e, jaiA64LdrX(JIT_SCRATCH_B, JIT_SCRATCH_C,
                                               (unsigned)offsetof(ObjInstance, klass)));
                            emit(e, jaiA64LdrW(JIT_SCRATCH_B, JIT_SCRATCH_B,
                                               (unsigned)offsetof(ObjClass, shapeId)));
                            emitConst64(e, JIT_SCRATCH_A, (int64_t)gshape);
                            emit(e, jaiA64SubsX(31, JIT_SCRATCH_B, JIT_SCRATCH_A));
                            branchOnDeopt(e, JAI_A64_NE);
                        }
                        if (!pushValue3(e, gk, gshape, gcls, gvv, -1)) return false;
                        emit(e, jaiA64MovX(dst, JIT_SCRATCH_C));
                        off += 6;
                        break;
                    }
                    /* Distinguish the two, because the census reads these and
                     * "a global of a kind the tier has no slot for" is a very
                     * different backlog item from an uncompiled callee. */
                    e->whyNot =
                        (gslot != NULL && !IS_CLOSURE(gvv) &&
                         !IS_CLASS(gvv) && !IS_NATIVE(gvv))
                            ? "a global of a kind the tier has no slot for"
                            : "callee is not a compiled global function";
                    return false;
                }
                if (e->depth >= JIT_MAX_STACK) return false;
                e->stackShape[e->depth] = 0;
                e->stackClass[e->depth] = (ObjClass *)(void *)AS_OBJ(nv);
                e->stackSeen[e->depth]  = nv;
                e->stackLocal[e->depth] = -1;
                e->stack[e->depth++]    = SLOT_NATIVE;
                off += 6;
                break;
            }
            if (e->depth >= JIT_MAX_STACK) return false;
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

        case OP_SET_GLOBAL: {
            /* Assigns without popping, exactly as the interpreter does: the
             * value is the statement's result and an OP_POP follows. */
            uint32_t nameIdx = jaiReadU24(code + off + 1);
            Value gvv;
            JaiEntry *gslot = globalSlot(e, closure, nameIdx, &gvv);
            if (gslot == NULL) {
                /* The interpreter raises NameError for a name with no binding;
                 * there is nothing to store into and nothing to compile. */
                e->whyNot = "a global with no storage to store into";
                return false;
            }
            if (e->depth == 0 || !holdsRegister(e->stack[e->depth - 1])) {
                e->whyNot = "nothing on the stack to store into a global";
                return false;
            }
            SlotKind sk = e->stack[e->depth - 1];
            if (sk != SLOT_INT && sk != SLOT_FLOAT && sk != SLOT_BOOL &&
                sk != SLOT_LIST && sk != SLOT_INST && sk != SLOT_MAYBE_INST) {
                /* SLOT_OBJ is deliberately NOT on this list. It means "some
                 * heap object of a type this tier does not model", which
                 * includes a closure -- the OP_CALL arm below calls one --
                 * and storing a closure into a global rebinds a callee that a
                 * compiled form, possibly this one a few instructions above,
                 * has already baked BY VALUE. ObjModule::version retires that
                 * form at its next ENTRY, not in the middle of a body, so it
                 * would be a wrong answer rather than a decline. Every kind
                 * left here is inert by jaiValueIsInertGlobal, which is what
                 * makes the bump rule below sound. */
                e->whyNot = "a global store of a kind that has no Value form";
                return false;
            }
            emitGlobalsGuard(e);
            {
                unsigned src = pushReg(e) - 1;
                emitConst64(e, JIT_SCRATCH_D, (int64_t)(uintptr_t)gslot);
                /* ObjModule::version only has to move when a write takes a
                 * class, a closure or a native AWAY or puts one IN; see
                 * jaiValueIsInertGlobal, whose C form is the authority. The
                 * value going IN is provably inert here (SLOT_OBJ is refused
                 * above), so only the value coming OUT can matter, and this
                 * recognises a STRICT SUBSET of the inert types: not an object
                 * at all, an ObjInstance, or an ObjList. Everything else
                 * bumps, which is only ever conservative. Widening
                 * jaiValueIsInertGlobal needs no change here; taking
                 * OBJ_INSTANCE or OBJ_LIST out of it does. */
                unsigned skip[3];
                unsigned nskip = 0;
                emit(e, jaiA64LdrW(JIT_SCRATCH_B, JIT_SCRATCH_D,
                                   (unsigned)offsetof(JaiEntry, value)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_B, VAL_OBJ));
                skip[nskip++] = e->count;
                emit(e, jaiA64BCond(JAI_A64_NE, 0));
                emit(e, jaiA64LdrX(JIT_SCRATCH_B, JIT_SCRATCH_D,
                                   (unsigned)offsetof(JaiEntry, value) + 8u));
                emit(e, jaiA64LdrByte(JIT_SCRATCH_C, JIT_SCRATCH_B,
                                      (unsigned)offsetof(Obj, type)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_C, OBJ_INSTANCE));
                skip[nskip++] = e->count;
                emit(e, jaiA64BCond(JAI_A64_EQ, 0));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_C, OBJ_LIST));
                skip[nskip++] = e->count;
                emit(e, jaiA64BCond(JAI_A64_EQ, 0));
                emitVersionBump(e, closure->fn->module);
                for (unsigned si = 0; si < nskip; si++) {
                    unsigned at = skip[si];
                    if (at < JIT_MAX_INSTS && e->count > at) {
                        e->code[at] = jaiA64BCond(
                            si == 0 ? JAI_A64_NE : JAI_A64_EQ,
                            (int32_t)(e->count - at));
                    }
                }
                /* No write barrier: the collector is a plain mark-sweep and
                 * the module's globals are traced as a root at every
                 * collection, so a raw store is all a store is. Nothing
                 * between the two writes can allocate, so no collection can
                 * see a tag and a payload that disagree. */
                emitTagFor(e, sk, src, JIT_SCRATCH_B, JIT_SCRATCH_A);
                emit(e, jaiA64StrW(JIT_SCRATCH_B, JIT_SCRATCH_D,
                                   (unsigned)offsetof(JaiEntry, value)));
                emit(e, jaiA64StrX(src, JIT_SCRATCH_D,
                                   (unsigned)offsetof(JaiEntry, value) + 8u));
            }
            /* Re-running this call interpreted would apply the store twice, so
             * from here a bail is no longer free. */
            e->wroteHeap = true;
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
                /* Cheapest first: a body small enough to stand where the call
                 * is costs neither the frame nor the argument shuffle. */
                Value cvv = e->stackSeen[e->depth - argc - 1];
                if (IS_CLOSURE(cvv) &&
                    inlineGlobalCall(e, fn, AS_CLOSURE(cvv), argc,
                                     (uint32_t)off)) {
                    off += 2;
                    break;
                }
                if (e->failed) return false;
                if (!emitGlobalCall(e, fn, argc, (uint32_t)off,
                                    (uint32_t)(off + 2))) return false;
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
                /* The same two things the direct global call checks, and for
                 * the same reasons: a raw payload is only sound if the callee
                 * was specialised to that kind and shape, and the caller's
                 * module-version guard only stands in for the callee's entry
                 * check if the two were compiled against the same version. */
                if (!directCallArgsMatch(e, cfn, cidx + 1u, argc)) return false;
                if (fn->module == NULL ||
                    cfn->jitFuncModuleVersion != fn->module->version) {
                    e->whyNot = "an indirect callee compiled against an older module";
                    return false;
                }
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

                /* Straight to the callee's compiled entry rather than out
                 * through jaiCallValue and an interpreter frame. The
                 * convention is the one a self-call already uses and the one
                 * jaiJitEnterFunc invokes: raw payloads in x0.., the closure
                 * itself in the last argument register when the body reads an
                 * upvalue, and on return x0 the value with x1 the verdict.
                 *
                 * The callee must live in this module, because it is the
                 * caller's module-version check at entry that stands in for
                 * the one jaiJitEnterFunc would have made. The arena is never
                 * freed and jitFunc is written once, so the address baked in
                 * here cannot go stale; only the ObjFunction identity has to
                 * be guarded, and it is, above. */
                unsigned calleeArgs = (unsigned)cfn->jitArgCount;
                bool wantsClosure = calleeArgs == argc + 1u;
                if (cfn->module != fn->module || cfn->jitArgBase != 1u ||
                    (!wantsClosure && calleeArgs != argc)) {
                    e->whyNot = "an indirect callee this tier cannot enter directly";
                    return false;
                }
                if (calleeArgs > JIT_MAX_ARITY) {
                    e->whyNot = "an indirect callee with too many arguments";
                    return false;
                }

                /* Roots before the branch: a `blr` pushes none, and the callee
                 * may allocate. */
                unsigned callRoots = 0;
                if (!emitRootFill(e, e->descOffset, &callRoots)) return false;
                if (callRoots > 0) {
                    unsigned dd = e->descOffset;
                    emit(e, jaiA64MovzX(JIT_SCRATCH_A, callRoots, 0));
                    emit(e, jaiA64StrX(JIT_SCRATCH_A, 31,
                                       dd + (unsigned)offsetof(JitCallDesc, nroots)));
                    emitConst64(e, JIT_SCRATCH_A, (int64_t)(uintptr_t)&gJitFrames);
                    emit(e, jaiA64LdrX(JIT_SCRATCH_B, JIT_SCRATCH_A, 0));
                    emit(e, jaiA64AddXImm(JIT_SCRATCH_C, 31, dd));
                    emit(e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_C,
                                       (unsigned)offsetof(JitCallDesc, link)));
                    emit(e, jaiA64StrX(JIT_SCRATCH_C, JIT_SCRATCH_A, 0));
                }

                unsigned firstArg = JIT_FIRST_SAVED + regBase(e) +
                                    (e->usesUpvalues ? 1u : 0u) +
                                    (cidx + 1u - (e->depth - e->valueDepth));
                for (unsigned i = 0; i < argc; i++) {
                    emit(e, jaiA64MovX(i, firstArg + i));
                }
                if (wantsClosure) emit(e, jaiA64MovX(argc, rCallee));
                emitConst64(e, JIT_SCRATCH_D,
                            (int64_t)(uintptr_t)cfn->jitFunc);
                emit(e, jaiA64Blr(JIT_SCRATCH_D));

                if (callRoots > 0) {
                    unsigned dd = e->descOffset;
                    emitConst64(e, JIT_SCRATCH_A, (int64_t)(uintptr_t)&gJitFrames);
                    emit(e, jaiA64AddXImm(JIT_SCRATCH_C, 31, dd));
                    emit(e, jaiA64LdrX(JIT_SCRATCH_B, JIT_SCRATCH_C,
                                       (unsigned)offsetof(JitCallDesc, link)));
                    emit(e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_A, 0));
                }
                /* x1 carries the callee's verdict; a bail there is a bail
                 * here, exactly as for a self-call. */
                emit(e, jaiA64SubsXImm(31, 1, 0));
                branchOnCondition(e, JAI_A64_NE);

                for (unsigned i = 0; i <= argc; i++) {
                    unsigned r;
                    if (!popValue(e, &r, NULL)) return false;
                }
                if (!pushValue(e, rkind, 0, NULL)) return false;
                emit(e, jaiA64MovX(pushReg(e) - 1, 0));
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

            /* A self-call branches to the prologue, past the entry guard, so
             * nothing else checks what it hands over. Passing a maybe-instance
             * into a slot typed as a plain instance would let a later field
             * read dereference zero. Record the slot and let the retry seed it
             * as a maybe-instance from the start -- per slot, so a body whose
             * other parameters are never null pays nothing for this one. */
            {
                bool retry = false;
                for (unsigned i = 0; i < argc; i++) {
                    unsigned pslot = 1u + i;   /* parameters are slots 1..arity */
                    unsigned aidx  = e->depth - argc + i;
                    SlotKind ak = e->stack[aidx];
                    SlotKind pk = e->localKind[pslot];
                    if (ak == SLOT_MAYBE_INST && pk == SLOT_INST) {
                        e->needNullable[pslot] = true;
                        retry = true;
                        continue;
                    }
                    if (ak != pk) {
                        e->whyNot = "a self-call argument is not the parameter's kind";
                        return false;
                    }
                    if ((pk == SLOT_INST || pk == SLOT_MAYBE_INST) &&
                        e->stackClass[aidx] != e->localClass[pslot]) {
                        e->whyNot = "a self-call passing a different class";
                        return false;
                    }
                }
                if (retry) {
                    e->whyNot = "a parameter is sometimes null";
                    return false;
                }
            }

            /* A `bl` pushes no roots, so anything this body holds in a
             * callee-saved register is invisible to a collection inside the
             * callee. Link the descriptor onto the collector's frame chain
             * around the call instead. Costs six instructions and only when
             * there is something to root. */
            unsigned selfRoots = 0;
            if (!emitRootFill(e, e->descOffset, &selfRoots)) return false;
            if (selfRoots > 0) {
                unsigned d = e->descOffset;
                emit(e, jaiA64MovzX(JIT_SCRATCH_A, selfRoots, 0));
                emit(e, jaiA64StrX(JIT_SCRATCH_A, 31,
                                   d + (unsigned)offsetof(JitCallDesc, nroots)));
                emitConst64(e, JIT_SCRATCH_A, (int64_t)(uintptr_t)&gJitFrames);
                emit(e, jaiA64LdrX(JIT_SCRATCH_B, JIT_SCRATCH_A, 0));
                emit(e, jaiA64AddXImm(JIT_SCRATCH_C, 31, d));
                emit(e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_C,
                                   (unsigned)offsetof(JitCallDesc, link)));
                emit(e, jaiA64StrX(JIT_SCRATCH_C, JIT_SCRATCH_A, 0));
            }

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
            /* Unlink before anything can leave: the bail branch below and every
             * later exit go through the epilogue, and a frame still on the
             * chain then points at a stack slot that no longer exists. x0 and
             * x1 carry the callee's answer, so only the scratches are free. */
            if (selfRoots > 0) {
                unsigned d = e->descOffset;
                emitConst64(e, JIT_SCRATCH_A, (int64_t)(uintptr_t)&gJitFrames);
                emit(e, jaiA64AddXImm(JIT_SCRATCH_C, 31, d));
                emit(e, jaiA64LdrX(JIT_SCRATCH_B, JIT_SCRATCH_C,
                                   (unsigned)offsetof(JitCallDesc, link)));
                emit(e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_A, 0));
            }
            /* x1 carries the callee's verdict, and each of the three answers
             * needs a different one here. This used to be one `bail`, which is
             * why `place` in queens has never compiled: a bail re-runs the
             * whole caller from the top, and `cols[row] = col` sits above the
             * recursion, so `bailAfterWrite` declined the body outright.
             *
             *   0  the value is in x0. The whole cost of this is one compare
             *      and one not-taken branch.
             *   2  the callee raised. The interpreter owns it; leave.
             *   1  the callee bailed, which it may only do having written
             *      nothing, so the CALL can be re-executed. Not a bail here
             *      though -- a deopt at this instruction, which resumes the
             *      interpreter with the caller's own earlier writes done and
             *      not repeated. That distinction is the whole unlock.
             *   4  the callee deoptimised part-way and may have written. The
             *      call cannot be re-executed, and the caller cannot record a
             *      second deopt over the callee's. So the callee is FINISHED
             *      in the interpreter, from its own record, and its value
             *      handed back here -- gDeopt is consumed at the innermost
             *      frame that sees it, which is why one record still suffices
             *      however deep the recursion goes.
             *
             * The interpreted continuation may return a kind this body did not
             * compile for -- a `return` the compiler typed from one
             * observation. That lands in the descriptor's result slot with
             * whatever tag it really has, and the existing `lastFromDesc`
             * deopt writes it out from there. */
            /* The fast path is what a recursive body pays per call: one
             * compare and one branch that is not taken. Every other answer is
             * a jump to a block emitted with the stubs -- inline, the two call
             * sites in `fib` had thirty instructions of cold code between them
             * and the benchmark lost 25%.
             *
             * The records have to be taken HERE even though the code is
             * emitted later, because they are of the model as it stands at
             * this instruction, and the model has moved on by then. */
            if (e->selfSlowCount >= JIT_MAX_SELF_SLOW) {
                e->whyNot = "more self-calls than the tier tracks";
                return false;
            }
            unsigned si = e->selfSlowCount++;
            unsigned resultReg = valueXReg(e, e->valueDepth - argc);
            e->selfSlow[si].roots     = selfRoots;
            e->selfSlow[si].resultReg = resultReg;
            e->selfSlow[si].stub      = -1;
            /* This body is its own callee, and it checks the tag only: -1 is
             * "no type check", and it has to be said rather than left to the
             * zeroed struct, where 0 is OBJ_STRING. */
            e->selfSlow[si].callee    = NULL;
            e->selfSlow[si].retType   = -1;
            e->selfSlow[si].retShape  = 0;
            /* Verdict 1: the callee bailed, and it may only do that having
             * written nothing, so the CALL is re-executed. Not a bail here --
             * a bail re-runs this whole body from the top, which is what
             * `cols[row] = col` above the recursion in queens' `place` made
             * unsound and is why that function has never compiled. A deopt at
             * this instruction resumes the interpreter with the caller's own
             * earlier writes done and not repeated. */
            if (!deoptRecordAt(e, (uint32_t)off, false,
                               &e->selfSlow[si].deoptBail)) {
                return false;
            }

            emit(e, jaiA64SubsXImm(31, 1, 0));
            if (e->fixupCount >= JIT_MAX_FIXUPS) { e->failed = true; return false; }
            e->fixups[e->fixupCount].instIndex    = (int)e->count;
            e->fixups[e->fixupCount].targetOffset = FIXUP_SELFSLOW - si;
            e->fixups[e->fixupCount].conditional  = true;
            e->fixups[e->fixupCount].depth        = -1;
            e->fixupCount++;
            emit(e, jaiA64BCond(JAI_A64_NE, 0));
            emit(e, jaiA64MovX(resultReg, 0));
            e->selfSlow[si].returnTo = (int)e->count;

            /* The call has happened, so the model moves on before the second
             * record: an unexpected result kind resumes AFTER the call,
             * holding what the descriptor carries. */
            e->depth      -= argc + 1;
            e->valueDepth -= argc;
            if (!pushValue(e, e->returnKind, e->returnShape, NULL)) return false;
            e->selfSlow[si].tag =
                  e->returnKind == SLOT_INT   ? VAL_INT
                : e->returnKind == SLOT_FLOAT ? VAL_FLOAT
                : e->returnKind == SLOT_BOOL  ? VAL_BOOL
                : e->returnKind == SLOT_NULL  ? VAL_NULL
                                              : VAL_OBJ;
            if (!deoptRecordAt(e, (uint32_t)off + 2u, true,
                               &e->selfSlow[si].deoptKind)) {
                return false;
            }
            off += 2;
            break;
        }

        case OP_TAIL_CALL: {
            /* An OSR form's x0 is the bytecode offset to resume at, not a
             * value -- see jaiJitEnterOsr, which does `*resumeAt = at`. The
             * return sequence below is the function tier's and leaves the
             * returned value in x0, so a `return` inside a compiled loop hands
             * the interpreter an integer or a pointer as an instruction
             * offset. `for i in 0..n { if .. { return x } }` miscompiled that
             * way in released builds: 14 runs in 20 wrong, and 10 in 10 under
             * --gc-stress. */
            if (e->osr) {
                e->whyNot = "a return inside an OSR loop";
                return false;
            }
            /* `return C(...)` compiles to this. The call is made exactly as
             * OP_CALL makes it and its result is returned. */
            unsigned argc = code[off + 1];
            if (isClassCallee(e, argc)) {
                if (!emitCallOut(e, argc)) return false;
            } else if (e->depth >= argc + 1u &&
                       e->stack[e->depth - argc - 1] == SLOT_FUNC) {
                if (!emitGlobalCall(e, fn, argc, (uint32_t)off,
                                    (uint32_t)(off + 2))) return false;
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
            if (!mergeReturnKind(e, k, tshape)) return false;
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
            /* An OSR form's x0 is the bytecode offset to resume at, not a
             * value -- see jaiJitEnterOsr, which does `*resumeAt = at`. The
             * return sequence below is the function tier's and leaves the
             * returned value in x0, so a `return` inside a compiled loop hands
             * the interpreter an integer or a pointer as an instruction
             * offset. `for i in 0..n { if .. { return x } }` miscompiled that
             * way in released builds: 14 runs in 20 wrong, and 10 in 10 under
             * --gc-stress. */
            if (e->osr) {
                e->whyNot = "a return inside an OSR loop";
                return false;
            }
            uint32_t rsh = e->depth > 0 ? e->stackShape[e->depth - 1] : 0;
            unsigned r;
            SlotKind k;
            if (!popValue(e, &r, &k)) return false;
            /* One return kind per function: the entry point rebuilds a Value
             * from it, and it cannot rebuild two. */
            if (!mergeReturnKind(e, k, rsh)) return false;
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
    fpSyncAll(e);
    /* An inlined body's offsets are the callee's; matching them against the
     * caller's fixups compares two different numbering schemes. There is
     * nothing to check either way -- it has no branches. */
    if (e->inlining) return !e->failed;
    /* Nothing may branch to an offset this walk carried a float into: the
     * branch would arrive with that value only in its X register, and the
     * instruction there would read the FP bank. Declining is the safe answer,
     * and the caller retries with the FP bank turned off. */
    for (unsigned i = 0; i < e->fpCarryCount; i++) {
        for (unsigned f = 0; f < e->fixupCount; f++) {
            if (e->fixups[f].targetOffset != e->fpCarry[i]) continue;
            e->whyNot = "a branch lands inside a float expression";
            return false;
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
            /* A slot an earlier attempt found is sometimes null holds the
             * pointer or zero. The class still comes from the argument this
             * entry carried: a maybe-instance is a correct supertype of an
             * instance, so widening cannot make a field offset wrong. */
            e->localKind[i]  = e->nullableLocal[i] ? SLOT_MAYBE_INST
                                                   : SLOT_INST;
            e->localClass[i] = AS_INSTANCE(v)->klass;
            e->localShape[i] = AS_INSTANCE(v)->klass->shapeId;
        } else if (IS_LIST(v)) {
            e->localKind[i] = SLOT_LIST;
        } else if (IS_OBJ(v)) {
            e->localKind[i] = SLOT_OBJ;
        } else if (IS_BOOL(v)) {
            e->localKind[i] = SLOT_BOOL;
        } else if (IS_NULL(v)) {
            /* A null argument -- a defaulted parameter, mostly. Nothing can be
             * done with it, but a body that never reads it compiles, and the
             * entry guard needs no check because an opaque slot is never read.
             * Refusing outright stopped every stdlib function with a defaulted
             * parameter, which is most of them. */
            e->localKind[i] = SLOT_OPAQUE;
        } else if (i == 0) {
            /* A plain function's slot 0 is the closure being called. Nothing
             * can be done with it, but the body has no reason to read it. */
            e->localKind[i] = SLOT_OPAQUE;
        } else {
            e->whyNot = "a parameter of a kind the tier has no register for";
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
                            const bool *dynamic, bool *needDynamic,
                            const bool *nullable, bool *needNullable,
                            bool noInline);

bool jaiJitCompileFunc(ObjClosure *closure, Value *slotBase) {
    ObjFunction *fn = closure->fn;
    if (!eligible(fn)) return false;

    /* Up to a few attempts: each one may discover another slot that two paths
     * disagree about, and the next begins knowing it. */
    bool dynamic[JIT_MAX_SLOTS + 1];
    bool need[JIT_MAX_SLOTS + 1];
    bool nullable[JIT_MAX_SLOTS + 1];
    bool needNull[JIT_MAX_SLOTS + 1];
    memset(dynamic, 0, sizeof dynamic);
    memset(nullable, 0, sizeof nullable);
    for (int attempt = 0; attempt < 4; attempt++) {
        memset(need, 0, sizeof need);
        memset(needNull, 0, sizeof needNull);
        if (compileFuncOnce(closure, slotBase, dynamic, need, nullable,
                            needNull, false)) {
            return true;
        }
        /* An inlined body that could not be emitted is not a decline: the
         * same call through the descriptor still compiles, and a compiled
         * form with a real call in it beats none at all. */
        if (gInlineFailed &&
            compileFuncOnce(closure, slotBase, dynamic, need, nullable,
                            needNull, true)) {
            return true;
        }
        bool grew = false;
        for (unsigned i = 0; i <= JIT_MAX_SLOTS; i++) {
            if (need[i] && !dynamic[i]) { dynamic[i] = true; grew = true; }
            /* Dynamic wins: it is the more general representation, and a slot
             * that wants both is one the nullable form cannot describe. */
            if (needNull[i] && !dynamic[i] && !nullable[i]) {
                nullable[i] = true; grew = true;
            }
        }
        if (!grew) return false;
    }
    return false;
}

static bool compileFuncOnce(ObjClosure *closure, Value *slotBase,
                            const bool *dynamic, bool *needDynamic,
                            const bool *nullable, bool *needNullable,
                            bool noInline) {
    ObjFunction *fn = closure->fn;
    gInlineFailed = false;

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
    memcpy(e.nullableLocal, nullable, sizeof e.nullableLocal);
    e.arity        = fn->arity;
    e.noInline     = noInline;
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
    memcpy(body.nullableLocal, nullable, sizeof body.nullableLocal);
    body.arity        = fn->arity;
    body.noInline     = noInline;
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
        memcpy(needNullable, body.needNullable, sizeof body.needNullable);
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

    emitSelfSlowStubs(&e, closure);

    emitGrowStubs(&e);

    /* One stub per guard, out of line. Each writes the record the interpreter
     * resumes from: the locals, the operand stack as it stood at that
     * instruction, and the offset of the instruction itself. */
    for (unsigned k = 0; k < e.deoptCount; k++) {
        e.deopt[k].stub = (int)e.count;
        emitConst64(&e, JIT_SCRATCH_A, (int64_t)(uintptr_t)&gDeopt);

        for (unsigned i = 0; i < (e.osr ? 0u : e.locals); i++) {
            unsigned slot = e.base + i;
            SlotKind kind = e.localKind[slot];
            /* Only an opaque slot is null; everything else that is not a
             * scalar is an object. Listing the object kinds instead meant a
             * SLOT_OBJ local -- a string, a dict, a closure -- was written out
             * as null on every deopt, which stayed invisible until a body
             * holding one could compile and then deopt. */
            unsigned tag = kind == SLOT_INT    ? VAL_INT
                         : kind == SLOT_FLOAT  ? VAL_FLOAT
                         : kind == SLOT_BOOL   ? VAL_BOOL
                         : kind == SLOT_OPAQUE ? VAL_NULL
                                               : VAL_OBJ;
            unsigned at = (unsigned)offsetof(JitDeoptRecord, locals) +
                          i * (unsigned)sizeof(Value);
            if (kind == SLOT_MAYBE_INST && !e.spilled) {
                unsigned pr = JIT_FIRST_SAVED + i;
                emitTagFor(&e, kind, pr, JIT_SCRATCH_B, JIT_SCRATCH_C);
                emit(&e, jaiA64StrW(JIT_SCRATCH_B, JIT_SCRATCH_A, at));
                emit(&e, jaiA64StrX(pr, JIT_SCRATCH_A, at + 8));
                continue;
            }
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
            unsigned reg0 = JIT_FIRST_SAVED + regBase(&e) +
                            (e.usesUpvalues ? 1u : 0u) + valueSeen;
            /* The stub is reached by a branch from the guard, so the FP bank
             * still holds whatever it held there. Moving it here rather than
             * before the guard is what keeps the hot path free of it. */
            if (e.deopt[k].fpLive & (1u << valueSeen)) {
                emit(&e, jaiA64FmovXD(reg0, fpRegAt(&e, valueSeen)));
            }
            if (kind == SLOT_MAYBE_INST) {
                emitTagFor(&e, kind, reg0, JIT_SCRATCH_B, JIT_SCRATCH_C);
                emit(&e, jaiA64StrW(JIT_SCRATCH_B, JIT_SCRATCH_A, at));
                emit(&e, jaiA64StrX(reg0, JIT_SCRATCH_A, at + 8));
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
        } else if (f->targetOffset <= FIXUP_SELFSLOW &&
                   f->targetOffset > FIXUP_SELFSLOW - JIT_MAX_SELF_SLOW) {
            target = e.selfSlow[FIXUP_SELFSLOW - f->targetOffset].stub;
            if (target < 0) {
                if (getenv("JAI_JIT_WHY")) {
                    fprintf(stderr, "[jit] %s stopped: a self-call block was "
                                    "never emitted\n",
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
        } else if (f->targetOffset <= FIXUP_GROW &&
                   f->targetOffset > FIXUP_GROW - JIT_MAX_GROW) {
            target = e.grow[FIXUP_GROW - f->targetOffset].stub;
            if (target < 0) {
                if (getenv("JAI_JIT_WHY")) {
                    fprintf(stderr, "[jit] %s stopped: a list-growth block was "
                                    "never emitted\n",
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
    fn->jitFuncNoWrite = !e.wroteHeap;

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
/* Bytes of the instruction at `off`, or 0 when it cannot be decoded.
 *
 * OP_CLOSURE is the one variable-length instruction: a u24 constant index and
 * then three bytes per upvalue, and the count is on the function that index
 * names. Treating it as undecodable refused OSR for the WHOLE CHUNK -- both
 * walks below start from offset 0 or scan forward -- so a module that declares
 * a class or a function before its hot loop, which is every one of them, could
 * never enter a compiled loop at module scope at all. Nothing reported it:
 * compileOsr was not reached, so there was no decline to see. */
static int instructionLength(const Chunk *c, int off) {
    uint8_t op = c->code[off];
    if (op != OP_CLOSURE) {
        int size = jaiOpOperandSize((OpCode)op);
        return size < 0 ? 0 : 1 + size;
    }
    if (off + 4 > c->count) return 0;
    uint32_t index = jaiReadU24(c->code + off + 1);
    if (index >= (uint32_t)c->constants.count) return 0;
    Value fnv = c->constants.data[index];
    if (!IS_FUNCTION(fnv)) return 0;
    return 4 + 3 * (int)AS_FUNCTION(fnv)->upvalueCount;
}

static bool isInstructionStart(const Chunk *c, uint32_t top) {
    for (int off = 0; off < c->count;) {
        if ((uint32_t)off == top) return true;
        int len = instructionLength(c, off);
        if (len <= 0) return false;
        off += len;
    }
    return false;
}

static uint32_t findLoopEnd(const Chunk *c, uint32_t top) {
    for (int off = (int)top; off < c->count;) {
        uint8_t op = c->code[off];
        int len = instructionLength(c, off);
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

/* How many loops enclose each byte of the chunk. Every OP_LOOP is a back edge
 * and the range it jumps over is its body, so one pass over the chunk counting
 * those ranges answers "how hot is this site relative to that one" well enough
 * to rank slots by. It does not need to be exact -- it only has to put the
 * innermost loop's variables ahead of the setup around them.
 *
 * A file static rather than an allocation, because both Emit structures read
 * the same table and compilation is not reentrant, which is already why they
 * are static themselves. A chunk longer than this goes unweighted: that costs
 * ranking quality and nothing else. */
#define JIT_MAX_DEPTH_MAP 8192
static uint8_t gLoopDepth[JIT_MAX_DEPTH_MAP];

static unsigned loopDepthTable(const Chunk *c) {
    int n = c->count + 1 < JIT_MAX_DEPTH_MAP ? c->count + 1 : JIT_MAX_DEPTH_MAP;
    for (int i = 0; i < n; i++) gLoopDepth[i] = 0;
    for (int off = 0; off < c->count;) {
        int len = instructionLength(c, off);
        if (len <= 0) break;
        if (c->code[off] == OP_LOOP) {
            int16_t jump = jaiReadI16(c->code + off + 1);
            int target = off + 3 + jump;
            if (target >= 0 && target <= off && off < n) {
                for (int j = target; j <= off; j++) {
                    if (gLoopDepth[j] < 200) gLoopDepth[j]++;
                }
            }
        }
        off += len;
    }
    return (unsigned)n;
}

static bool compileOsr(ObjClosure *closure, uint32_t top, Value *slots,
                       uint8_t iterKind, Value elemSample, bool noInline) {
    bool hasIter = iterKind != 0;
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
    e.loopDepth = gLoopDepth;
    e.loopDepthCount = loopDepthTable(&fn->chunk);
    e.hasIter = hasIter;
    e.iterKind = iterKind;
    e.elemSample = elemSample;
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


    /* The probe runs AFTER the seeding above, not before it. It used to copy
     * the kinds while they were still zeroed, so it measured a body in which
     * every slot was an untyped int -- a different program from the one being
     * compiled, and the maxValue the register decision rests on was measured on
     * the wrong one.
     *
     * It also narrows `locals` to what the loop actually names. OSR started
     * from `fn->maxSlots`, the whole slot window of the enclosing function --
     * eighteen or twenty slots for these bodies -- so
     * `reserved + locals + maxValue <= 10` could never hold and register-
     * resident locals were unreachable for any real function. The function tier
     * has always narrowed to the highest slot its body touches; this does the
     * same. */
    /* Registers for the slots that earn them, memory for the rest. The
     * reserved four (or one) plus the X locals plus the deepest expression
     * must all sit inside the ten callee-saved registers; a first pass
     * measures the last of those, and which slots are named at all, and how
     * often each of them is named inside the innermost loop. */
    {
        static Emit probe;
        memset(&probe, 0, sizeof probe);
        probe.osr = true; probe.measuring = true; probe.hasIter = hasIter;
        probe.iterKind = iterKind; probe.elemSample = elemSample;
        probe.osrTop = top; probe.osrEnd = end; probe.base = 0;
        probe.noInline = noInline;
        probe.locals = e.locals; probe.callsOut = true; probe.observed = slots;
        probe.offsetToInst = map; probe.offsetToDepth = depths;
        probe.limitLiteral = -1; probe.bailBlock = -1; probe.exceptionExit = -1;
        probe.loopDepth = gLoopDepth;
        probe.loopDepthCount = e.loopDepthCount;
        for (unsigned i = 0; i < e.locals; i++) {
            probe.localKind[i]  = e.localKind[i];
            probe.localShape[i] = e.localShape[i];
            probe.localClass[i] = e.localClass[i];
            probe.localTyped[i] = e.localTyped[i];
            probe.localSeen[i]  = e.localSeen[i];
        }
        if (compileBody(&probe, closure) && !probe.failed) {
            unsigned used = probe.maxSlotUsed + 1u;
            if (used < e.locals) e.locals = used;

            /* What is left over after the loop's own reserved registers and
             * the deepest expression the body builds. maxValue is model state,
             * not a register number, so measuring it in memory mode and
             * spending it here is sound. */
            unsigned overhead = osrReserved(&e) + probe.maxValue;
            unsigned availX = overhead < JIT_MAX_SAVED
                                  ? JIT_MAX_SAVED - overhead : 0u;
            unsigned availFp = JIT_FP_MAX_SAVED;
            /* Busiest first, and only slots the body names. A slot whose kind
             * varies at run time keeps its tag check and stays in memory. */
            uint8_t order[JIT_MAX_SLOTS + 1];
            unsigned n = 0;
            for (unsigned i = 0; i < e.locals; i++) {
                if (probe.slotUse[i] == 0) continue;
                if (probe.dynamicLocal[i]) continue;
                order[n++] = (uint8_t)i;
            }
            for (unsigned i = 0; i + 1 < n; i++) {
                unsigned best = i;
                for (unsigned j = i + 1; j < n; j++) {
                    if (probe.slotUse[order[j]] > probe.slotUse[order[best]]) {
                        best = j;
                    }
                }
                uint8_t t = order[i]; order[i] = order[best]; order[best] = t;
            }
            for (unsigned i = 0; i < n; i++) {
                unsigned slot = order[i];
                if (probe.localKind[slot] == SLOT_FLOAT && availFp > 0) {
                    e.slotFpReg[slot] =
                        (uint8_t)(JIT_FP_FIRST_SAVED + e.fpLocals++);
                    availFp--;
                } else if (availX > 0) {
                    e.slotXReg[slot] = (uint8_t)(JIT_FIRST_SAVED +
                                                 osrReserved(&e) + e.xLocals++);
                    availX--;
                }
            }
        }
        for (int i = 0; i <= fn->chunk.count; i++) { map[i] = -1; depths[i] = -1; }
    }

    unsigned frame = 16u + 8u * JIT_MAX_SAVED + (unsigned)sizeof(JitCallDesc);
    e.descOffset = 16u + 8u * JIT_MAX_SAVED;
    /* The ObjIter, for the heads that do not keep it in a register. Sixteen
     * bytes so the frame stays 16-aligned without a second rounding. */
    e.iterFrameOffset = (frame + 7u) & ~7u;   /* str/ldr scale the offset by 8 */
    frame = e.iterFrameOffset + 16u;
    e.fpSaveOffset = frame;
    frame += 8u * JIT_FP_MAX_SAVED;
    e.frameBytes = (frame + 15u) & ~15u;

    emitFrameEnter(&e);
    emitSaveRestore(&e, true);
    emitFpSaveRestore(&e, true);
    emit(&e, jaiA64MovX(JIT_SLOTS_REG, 0));
    /* Payloads only: the entry has already checked every slot's kind, so there
     * is nothing left to guard here. */
    for (unsigned i = 0; i < e.locals; i++) {
        if (e.slotXReg[i] != 0) {
            emit(&e, jaiA64LdrX(e.slotXReg[i], JIT_SLOTS_REG, i * 16u + 8u));
        } else if (e.slotFpReg[i] != 0) {
            emit(&e, jaiA64LdrD(e.slotFpReg[i], JIT_SLOTS_REG, i * 16u + 8u));
        }
    }
    if (hasIter) {
        /* x1 is the iterator on entry. A range head reads everything it wants
         * out of it here and parks the pointer in the frame; a list head keeps
         * it, because its version guard reads the iterator every iteration. */
        unsigned rIter = iterKind == 1 ? 1u : JIT_ITER_REG;
        if (iterKind == 1) {
            emit(&e, jaiA64StrX(1, 31, e.iterFrameOffset));
        } else {
            emit(&e, jaiA64MovX(JIT_ITER_REG, 1));
        }
        emit(&e, jaiA64LdrX(JIT_IDX_REG, rIter,
                            (unsigned)offsetof(ObjIter, index)));
        emit(&e, jaiA64LdrX(JIT_LIM_REG, rIter,
                            (unsigned)offsetof(ObjIter, limit)));
        /* A range yields start + index, so a loop that does not begin at zero
         * needs its start too. ObjIter.source is a Value, so the object
         * pointer sits eight bytes into it. */
        emit(&e, jaiA64LdrX(JIT_START_REG, rIter,
                            (unsigned)offsetof(ObjIter, source) + 8));
        if (iterKind == 1) {
            emit(&e, jaiA64LdrX(JIT_START_REG, JIT_START_REG,
                                (unsigned)offsetof(ObjRange, start)));
        }
        /* A list head leaves JIT_START_REG holding the ObjList itself. */
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
            /* A range head left the iterator in the frame rather than in a    \
             * register, so it comes back here. Every stub this expands into   \
             * is a way out of the loop, so the load is off the hot path and   \
             * JIT_SCRATCH_A is dead at the top of all of them. */             \
            unsigned rIt = JIT_ITER_REG;                                       \
            if (iterKind == 1) {                                               \
                rIt = JIT_SCRATCH_A;                                           \
                emit(&e, jaiA64LdrX(rIt, 31, e.iterFrameOffset));              \
            }                                                                  \
            emit(&e, jaiA64StrX(JIT_IDX_REG, rIt,                              \
                                (unsigned)offsetof(ObjIter, index)));          \
        }                                                                      \
        for (unsigned li = 0; li < e.locals; li++) {                           \
            /* Slots still in the frame were written through as they went. */  \
            unsigned lx = e.slotXReg[li], lf = e.slotFpReg[li];                \
            if (lx == 0 && lf == 0) continue;                                  \
            SlotKind lk = e.localKind[li];                                     \
            unsigned lt = lk == SLOT_INT   ? VAL_INT                           \
                        : lk == SLOT_FLOAT ? VAL_FLOAT                         \
                        : lk == SLOT_BOOL  ? VAL_BOOL                          \
                        : lk == SLOT_OPAQUE ? VAL_NULL                         \
                                           : VAL_OBJ;                          \
            /* A d-resident slot the walk decided was not a float after all:   \
             * the home is bit-exact either way, so it only has to come back   \
             * through an X register to be tagged. */                          \
            unsigned rp = lx;                                                  \
            if (lf != 0 && (lk == SLOT_MAYBE_INST || lt != VAL_FLOAT)) {       \
                rp = JIT_SCRATCH_B;                                            \
                emit(&e, jaiA64FmovXD(rp, lf));                                \
                lf = 0;                                                        \
            }                                                                  \
            if (lk == SLOT_MAYBE_INST) {                                       \
                emitTagFor(&e, lk, rp, JIT_SCRATCH_D, JIT_SCRATCH_C);          \
                emit(&e, jaiA64StrW(JIT_SCRATCH_D, JIT_SLOTS_REG, li * 16u));  \
                emit(&e, jaiA64StrX(rp, JIT_SLOTS_REG, li * 16u + 8u));        \
                continue;                                                      \
            }                                                                  \
            if (lt == VAL_NULL) continue;                                      \
            emit(&e, jaiA64MovzX(JIT_SCRATCH_D, lt, 0));                       \
            emit(&e, jaiA64StrW(JIT_SCRATCH_D, JIT_SLOTS_REG, li * 16u));      \
            if (lf != 0) {                                                     \
                emit(&e, jaiA64StrD(lf, JIT_SLOTS_REG, li * 16u + 8u));        \
            } else {                                                           \
                emit(&e, jaiA64StrX(rp, JIT_SLOTS_REG, li * 16u + 8u));        \
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

    emitSelfSlowStubs(&e, closure);

    emitGrowStubs(&e);

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
            unsigned reg0 = JIT_FIRST_SAVED + regBase(&e) +
                            (e.usesUpvalues ? 1u : 0u) + valueSeen;
            if (e.deopt[k].fpLive & (1u << valueSeen)) {
                emit(&e, jaiA64FmovXD(reg0, fpRegAt(&e, valueSeen)));
            }
            if (kind == SLOT_MAYBE_INST) {
                emitTagFor(&e, kind, reg0, JIT_SCRATCH_B, JIT_SCRATCH_C);
                emit(&e, jaiA64StrW(JIT_SCRATCH_B, JIT_SCRATCH_A, at));
                emit(&e, jaiA64StrX(reg0, JIT_SCRATCH_A, at + 8));
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
        else if (f->targetOffset <= FIXUP_SELFSLOW &&
                 f->targetOffset > FIXUP_SELFSLOW - JIT_MAX_SELF_SLOW)
            target = e.selfSlow[FIXUP_SELFSLOW - f->targetOffset].stub;
        else if (f->targetOffset <= FIXUP_DEOPT &&
                 f->targetOffset > FIXUP_DEOPT - JIT_MAX_DEOPT)
            target = e.deopt[FIXUP_DEOPT - f->targetOffset].stub;
        else if (f->targetOffset <= FIXUP_GROW &&
                 f->targetOffset > FIXUP_GROW - JIT_MAX_GROW)
            target = e.grow[FIXUP_GROW - f->targetOffset].stub;
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

    if (fn->osrCount >= JAI_OSR_MAX) return false;
    JaiOsrForm *form = &fn->osrForms[fn->osrCount];
    form->code  = entry;
    form->top   = top;
    form->slots = (uint8_t)e.locals;
    form->iterKind = iterKind;
    for (unsigned i = 0; i < e.locals; i++) form->kinds[i] = (uint8_t)e.localKind[i];
    fn->osrCount++;
    fn->osrHot = true;
    fn->osrDeclines = 0;
    fn->jitOsrModuleVersion = fn->module != NULL ? fn->module->version : 0;
    if (getenv("JAI_JIT_WHY")) {
        fprintf(stderr, "[jit] osr %s at %u: %u instructions iter=%u\n",
                fn->name ? fn->name->chars : "<anon>", top, e.count,
                (unsigned)iterKind);
    }
    /* The same dump the whole-function tier has. A compiled loop is where most
     * of the time goes, so it is the code most worth reading, and until now it
     * was the only tier that could not be read at all. */
    {
        const char *dump = getenv("JAI_JIT_DUMP");
        if (dump != NULL && fn->name != NULL &&
            strcmp(dump, fn->name->chars) == 0) {
            char path[256];
            snprintf(path, sizeof path, "jit_osr_%s_%u.bin",
                     fn->name->chars, top);
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
    uint8_t iterKind = 0;
    Value elemSample = NULL_VAL;
    if (hasIter) {
        if (vm.stackTop <= frame->slots) return 0;
        Value it = vm.stackTop[-1];
        if (!IS_ITER(it)) return 0;
        iter = AS_ITER(it);
        if (iter->kind == ITER_RANGE && IS_RANGE(iter->source)) {
            /* Unit steps only -- that is what makes the yielded value start
             * plus the index. The start need not be 0; it is loaded at entry. */
            if (AS_RANGE(iter->source)->step != 1) return 0;
            iterKind = 1;
        } else if (iter->kind == ITER_LIST && IS_LIST(iter->source)) {
            ObjList *src = AS_LIST(iter->source);
            /* The element the loop is about to bind, taken from the list
             * itself. Reading the loop variable's slot instead gives whatever
             * the previous iteration left there -- nothing at all on the first
             * entry -- and a kind guard built from that is aimed at the wrong
             * type. That is what made the first attempt at this crash. */
            int at = (int)iter->index;
            if (at < 0 || at >= src->count) return 0;
            elemSample = src->items[at];
            iterKind = 2;
        } else {
            return 0;
        }
    }

    JaiOsrForm *form = NULL;
    for (unsigned i = 0; i < fn->osrCount; i++) {
        /* The kind as well as the offset: a form compiled for a range head
          * entered with a list iterator would read ObjRange::start out of an
          * ObjList, and `for x in cond ? xs : 0..n` is enough to arrange it. */
        if (fn->osrForms[i].top == top &&
            fn->osrForms[i].iterKind == iterKind) {
            form = &fn->osrForms[i];
            break;
        }
    }
    if (form == NULL) {
        if (fn->osrRefused || fn->osrCount >= JAI_OSR_MAX) return 0;
        if (!compileOsr(closure, top, frame->slots, iterKind, elemSample,
                        false) &&
            !compileOsr(closure, top, frame->slots, iterKind, elemSample,
                        true)) {
            /* Inlining widens live ranges; a loop that will not fit with it
             * may fit without, and a compiled call beats no compile at all. */
            if (++fn->osrAttempts >= 5 * JAI_OSR_MAX) fn->osrRefused = true;
            return 0;
        }
        form = &fn->osrForms[fn->osrCount - 1];
    }
    if (fn->module == NULL || fn->module->version != fn->jitOsrModuleVersion) return 0;

    /* Every slot must still hold what it held when this was compiled. */
    for (unsigned i = 0; i < form->slots; i++) {
        Value v = frame->slots[i];
        if ((SlotKind)form->kinds[i] == SLOT_MAYBE_INST) {
            if (!IS_NULL(v) && !IS_INSTANCE(v)) return 0;
            continue;
        }
        switch ((SlotKind)form->kinds[i]) {
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
    int64_t at = ((OsrFnIter)(uintptr_t)form->code)(frame->slots, iter);
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

/* One incoming argument, converted from the Value the interpreter holds to the
 * raw payload the compiled prologue expects, with every check the compiled
 * body was allowed to assume. */
static inline bool jitArgIn(ObjClosure *closure, const Value *slotBase,
                            unsigned i, int64_t *out) {
    const ObjFunction *fn = closure->fn;
    Value v = slotBase[fn->jitArgBase + i];
    {
        switch ((SlotKind)fn->jitParamKind[i]) {
        case SLOT_INT:
            if (!IS_INT(v)) return false;
            *out = AS_INT(v);
            break;
        case SLOT_FLOAT: {
            if (!IS_FLOAT(v)) return false;
            double d = AS_FLOAT(v);
            memcpy(out, &d, sizeof *out);
            break;
        }
        case SLOT_MAYBE_INST: {
            /* The pointer, or zero for null -- the register form. The class is
             * still checked when there is one, because a self-call branches
             * straight to the prologue and this is the only place it is
             * established. */
            if (IS_NULL(v)) { *out = 0; break; }
            if (!IS_INSTANCE(v)) return false;
            ObjInstance *mi = AS_INSTANCE(v);
            if (mi->klass == NULL ||
                mi->klass->shapeId != fn->jitParamShape[i]) {
                return false;
            }
            *out = (int64_t)(uintptr_t)mi;
            break;
        }
        case SLOT_INST: {
            /* The class as well as the type: every field offset in the body
             * was resolved against this one shape. Compiled code holds the
             * raw pointer, which is safe only because the body cannot
             * allocate -- no collection can run while it does -- and the
             * argument slots keep the instance reachable meanwhile. */
            if (!IS_INSTANCE(v)) return false;
            ObjInstance *inst = AS_INSTANCE(v);
            if (inst->klass == NULL ||
                inst->klass->shapeId != fn->jitParamShape[i]) {
                return false;
            }
            *out = (int64_t)(uintptr_t)inst;
            break;
        }
        case SLOT_LIST:
            if (!IS_LIST(v)) return false;
            *out = (int64_t)(uintptr_t)AS_LIST(v);
            break;
        case SLOT_OBJ:
            if (!IS_OBJ(v)) return false;
            *out = (int64_t)(uintptr_t)AS_OBJ(v);
            break;
        case SLOT_BOOL:
            if (!IS_BOOL(v)) return false;
            *out = AS_BOOL(v) ? 1 : 0;
            break;
        case SLOT_OPAQUE:
            *out = 0;   /* never read; see seedLocals */
            break;
        case SLOT_CLOSURE:
            /* Not a slot at all: the closure the interpreter is calling. Safe
             * to hold raw for the same reason every other pointer here is --
             * the body cannot allocate, and the caller holds this closure. */
            *out = (int64_t)(uintptr_t)closure;
            break;
        default:
            return false;
        }
    }
    return true;
}

/* The verdict and the returned payload, turned back into what the
 * interpreter holds. */
static inline JaiJitOutcome jitResultOut(ObjFunction *fn, JitResult r,
                                         Value *slotBase) {
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

    if ((SlotKind)fn->jitReturnKind == SLOT_MAYBE_INST) {
        slotBase[0] = r.value == 0 ? NULL_VAL
                                   : OBJ_VAL((Obj *)(uintptr_t)r.value);
        vm.stackTop = slotBase + 1;
        return JAI_JIT_DONE;
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

JaiJitOutcome jaiJitEnterFunc(ObjClosure *closure, Value *slotBase) {
    ObjFunction *fn = closure->fn;
    if (fn->jitFunc == NULL) return JAI_JIT_DECLINED;

    /* Compiled code reads the global that names this function exactly once,
     * at compile time, and then calls it directly. Rebinding the name has to
     * invalidate that, and a module's version counter moves on every global
     * mutation, so one comparison covers it. It is conservative -- any global
     * write in the module retires the compiled form -- and conservative is the
     * safe direction. */
    if (fn->module == NULL || fn->module->version != fn->jitFuncModuleVersion) {
        return JAI_JIT_DECLINED;
    }

    unsigned arity = fn->jitArgCount;
    int64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    /* Unrolled rather than a loop over `int64_t a[JIT_MAX_ARITY]`, and that is
     * measured rather than tidy. An array of four int64 is what makes clang
     * give this function a stack-protector prologue and epilogue, and it made
     * every argument travel out to the frame and back on its way to the
     * register the call is about to read it from. This function is on the path
     * of every interpreted call to a compiled body -- 43% of `xs.map(|x| x*2)`
     * and 13% of queens, by `sample` -- so both of those are paid per call. */
    if (arity > JIT_MAX_ARITY) return JAI_JIT_DECLINED;
    if (arity > 0 && !jitArgIn(closure, slotBase, 0, &a0)) return JAI_JIT_DECLINED;
    if (arity > 1 && !jitArgIn(closure, slotBase, 1, &a1)) return JAI_JIT_DECLINED;
    if (arity > 2 && !jitArgIn(closure, slotBase, 2, &a2)) return JAI_JIT_DECLINED;
    if (arity > 3 && !jitArgIn(closure, slotBase, 3, &a3)) return JAI_JIT_DECLINED;

    JitResult r;
    switch (arity) {
    case 0: r = ((Fn0)(uintptr_t)fn->jitFunc)(); break;
    case 1: r = ((Fn1)(uintptr_t)fn->jitFunc)(a0); break;
    case 2: r = ((Fn2)(uintptr_t)fn->jitFunc)(a0, a1); break;
    case 3: r = ((Fn3)(uintptr_t)fn->jitFunc)(a0, a1, a2); break;
    default: r = ((Fn4)(uintptr_t)fn->jitFunc)(a0, a1, a2, a3); break;
    }
    return jitResultOut(fn, r, slotBase);
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
