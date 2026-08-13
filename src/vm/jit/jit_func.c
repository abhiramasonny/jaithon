/* jit_func.c -- whole-function JIT tier: compiles self-recursive, integer-only bodies to native arm64, bailing to the interpreter on overflow, deep recursion, or an unsupported shape. */
#include "vm/jit/jit.h"

#include "vm/jit/jit_arm64.h"
#include "vm/gc.h"
/* For jaiBuiltinMethod: resolving `xs.len()` to a native needs the runtime's name table. */
#include "runtime/runtime.h"
/* For jaiOpBranchOperandAt: says which opcodes carry a branch target. */
#include "vm/bytecode/verify.h"
#include "vm/vm.h"

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>

#if (defined(__aarch64__) || defined(__arm64__))

#include <pthread.h>

/* ------------------------------------------------------------------ */
/* Run-time state shared with compiled code                             */
/* ------------------------------------------------------------------ */

/* Two words in x0/x1 per AAPCS64 struct return (no store/load like the old global flag). A bailed
 * callee sends its caller straight to its own bail block, so recursion unwinds in one shot. */
typedef struct { int64_t value; int64_t bailed; } JitResult;

/* Derived from the thread's real bounds, not the compiling frame's sp -- that would bail on
 * every entry for a function compiled near the top of stack and called later from deep in the interpreter. */
static uintptr_t stackLimit(void) {
    pthread_t self = pthread_self();
    void  *top  = pthread_get_stackaddr_np(self);
    size_t size = pthread_get_stacksize_np(self);
    if (top == NULL || size == 0) return 0;
    /* Margin covers the deepest compiled frame plus what the interpreter needs to unwind and report the error. */
    return (uintptr_t)top - size + (256u * 1024u);
}


/* ------------------------------------------------------------------ */
/* Register plan                                                        */
/* ------------------------------------------------------------------ */

#define JIT_FIRST_SAVED 19u   /* x19..x28 are callee-saved and ours */
#define JIT_MAX_SAVED   10u
/* Model entries, not register count -- an inlined body's operand-stack entries live in their own bank, wider than x19..x28. */
#define JIT_MAX_STACK   20u
/* Slots the compile-time model can describe, not the register budget: a declared frame can be wider
 * than what it touches; only the second (measuring) pass is held to JIT_MAX_SAVED. */
#define JIT_MAX_SLOTS   64u
#define JIT_MAX_ARITY    4u   /* arguments arrive in x0..x3 */
/* Deopt stubs dominate this size: each writes out every local and live stack entry. `merge` silently needed 512 -- hence the diagnostics. */
#define JIT_MAX_INSTS 20000u
#define JIT_MAX_FIXUPS 6000u
#define JIT_SCRATCH_A    9u
#define JIT_SCRATCH_B   10u
#define JIT_SCRATCH_C   11u
#define JIT_SCRATCH_D   12u
/* x13..x17 are caller-saved and the tier names none of them, so a body that
 * calls nothing owns them outright. x18 is Darwin's platform register and is
 * NOT in the range. They are the only registers a loop-invariant hoist can
 * spend without taking one from the locals -- see planHoists. */
#define JIT_FREE_FIRST  13u
#define JIT_FREE_COUNT   5u
/* Two registers each; four list headers is every stencil seen so far. */
#define JIT_MAX_HOIST    4u
/* Repeated from the register plan below, which cannot be declared this early:
 * x0..x8, the bank a call-free body's operand stack uses. */
#define JIT_SCRATCH_BANK_COUNT 9u

/* ------------------------------------------------------------------ */
/* Calling out of compiled code                                         */
/* ------------------------------------------------------------------ */

#define JIT_MAX_ARGS_OUT 4

/* Values first in JitCallDesc so every field is 8-aligned and the emitted stores can use scaled forms. */
typedef struct JitCallDesc {
    /* link/nroots come first so `roots` sits at a fixed offset from the chain head; link != NULL means this descriptor is on the collector's walk chain. */
    struct JitCallDesc *link;
    int64_t nroots;
    Value   roots[JIT_MAX_SAVED];
    Value   callee;
    Value   args[JIT_MAX_ARGS_OUT];
    Value   result;
    int64_t argc;
    /* aux: only OP_GET_SLICE uses it, for which of start/stop/step are present -- `xs[null:3]` vs `xs[:3]` can't be told apart from the values alone. */
    int64_t aux;
} JitCallDesc;

/* A descriptor call pushes its roots via a C helper; a self-call (bare `bl`) does not, so callee-saved
 * registers are invisible to a collection inside the callee -- the tier refuses allocate-then-self-call for exactly this reason. */
static JitCallDesc *gJitFrames;

void jaiJitMarkFrames(void) {
    for (JitCallDesc *f = gJitFrames; f != NULL; f = f->link) {
        for (int64_t i = 0; i < f->nroots; i++) jaiGCMarkValue(f->roots[i]);
    }
}

/* Roots go in as a RANGE (jaiGCPushRootRange/Pop), not copied one at a time -- copying individually
 * was O(roots) per call-out; a bulk jaiGCPushRoots variant was tried and reverted, since it still copied every value. Returns 0 on success, 1 with an exception pending. */
/* Not a deopt: overflow raises directly, since the interpreter would also throw here. A bail is only
 * sound before any write; raising stays sound after a field store has already happened. */
static void jitThrowOverflow(int64_t which) {
    static const char *ops[3]  = { "+",  "-",  "*"  };
    static const char *wrap[3] = { "+%", "-%", "*%" };
    int i = (which >= 0 && which < 3) ? (int)which : 0;
    (void)jaiThrow(vm.cOverflowError,
                   "integer overflow in '%s'; use '%s' to wrap", ops[i], wrap[i]);
}

/* Global, not a frame field: the compiled frame is gone by the time C looks, and the VM is single-threaded so only one body can be deoptimising at a time. */
typedef struct {
    int64_t ip;
    int64_t base;
    int64_t nlocals;
    int64_t nstack;
    Value   locals[JIT_MAX_SLOTS + 1];
    Value   stack[JIT_MAX_STACK + 1];
} JitDeoptRecord;

static JitDeoptRecord gDeopt;

/* Cached: getenv is O(environ) and this sits on the hot deopt path, so leaving it uncached let ambient
 * shell-exported variable count perturb benchmarks (sort_merge moved 70ms->100ms on padding alone). Same idiom as jaiJitEnabled. */
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

/* Receiver is args[0], exactly where callNativeAt wants it, so no bound wrapper is made. Roots as jitCallOut does, since push and its kin allocate. */
static int jitInvokeMethod(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    bool ok = jaiCallMethodWithReceiver(d->callee, d->args, (int)d->argc,
                                        &d->result);
    jaiGCPopRootRange();
    return ok ? 0 : 1;
}

static int jitInvokeNative(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    bool ok = jaiInvokeNativeWithReceiver(d->callee, d->args, (int)d->argc,
                                          &d->result);
    jaiGCPopRootRange();
    return ok ? 0 : 1;
}

/* Allocates (jaiListNew), so roots go down first as for any call out of compiled code. */
static int jitBuildList(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    ObjList *list = jaiListNew((int)d->argc);
    for (int64_t i = 0; i < d->argc; i++) list->items[i] = d->args[i];
    list->count = (int)d->argc;
    d->result = OBJ_VAL(list);
    jaiGCPopRootRange();
    return 0;
}

/* args: start, stop, inclusive-flag. Allocates twice (range + iterator), so roots go down first as usual. */
static int jitMakeRangeIter(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    ObjRange *r = jaiRangeNew(AS_INT(d->args[0]), AS_INT(d->args[1]), 1,
                              AS_INT(d->args[2]) != 0);
    jaiGCPushRoot(OBJ_VAL(r));
    ObjIter *it = jaiIterNew(ITER_RANGE, OBJ_VAL(r));
    jaiGCPopRoot();
    d->result = OBJ_VAL(it);
    jaiGCPopRootRange();
    return 0;
}

static int jitMakeIter(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    Value src = d->args[0];
    IterKind k = IS_LIST(src) ? ITER_LIST : ITER_LIST;
    ObjIter *it = jaiIterNew(k, src);
    d->result = OBJ_VAL(it);
    jaiGCPopRootRange();
    return IS_LIST(src) ? 0 : 1;
}

/* f-string: the interpreter's parts, read off the operand stack, land here contiguously in args[].
 * Builtin path only -- compiler checks at compile time that the module hasn't rebound `str`; a rebind retires this form. */
static int jitFormat(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    ObjString *formatted = jaiValueFormat(d->args, (int)d->argc);
    jaiGCPopRootRange();
    if (formatted == NULL) return 1;
    d->result = OBJ_VAL(formatted);
    return 0;
}

/* No descriptor/roots: growing a list cannot collect -- jaiListReserve->jaiRealloc never triggers the
 * marker (gc.c: collections only happen at jaiGCMaybeCollect safepoints). Returns 1 if it raised (list past INT32_MAX). */
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

/* Safe because jaiGCWanted()==false is gc.c's own proof no collection can happen here (collections begin
 * only in jaiGCMaybeCollect), and the object is fully built before being linked in. NULL means "declined": caller falls back to the descriptor path, which roots and may collect. */
static ObjInstance *jitInstanceAlloc(ObjClass *cls) {
    if (JAI_UNLIKELY(jaiGCWanted())) return NULL;

    GCState *g = jaiGCActive;
    if (JAI_UNLIKELY(g == NULL || cls == NULL)) return NULL;

    const uint16_t count = cls->fieldCount;
    const size_t size = sizeof(ObjInstance) + sizeof(Value) * (size_t)count;
    if (JAI_UNLIKELY(!jaiSmallServes(size))) return NULL;

    ObjInstance *inst = (ObjInstance *)jaiSmallNew(size);
    Obj *obj = (Obj *)inst;

    obj->type = OBJ_INSTANCE;
    obj->isMarked = false;
    obj->subFlag = false;
    obj->subFlag2 = false;

    inst->klass = cls;
    inst->fieldCount = count;
    /* Zeroes every field, not just the ones about to be overwritten: the marker reads all `count` of them,
 * and an unwritten field is whatever the last occupant of this bin left behind. */
    for (uint16_t i = 0; i < count; ++i) inst->fields[i] = NULL_VAL;

    /* Linked last: a mid-collection isMarked state can't arise here, since a collection in progress makes jaiGCLimit zero and the first line above would already have declined. */
    obj->next = g->objects;
    g->objects = obj;

    vm.allocCount++;
    return inst;
}

static int jitNewInstance(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    ObjInstance *inst = jaiInstanceNew((ObjClass *)(uintptr_t)AS_OBJ(d->callee));
    d->result = OBJ_VAL(inst);
    jaiGCPopRootRange();
    return 0;
}

static int jitGetSlice(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    uint8_t flags = (uint8_t)d->aux;
    bool hasStart = (flags & 1) != 0, hasStop = (flags & 2) != 0,
         hasStep = (flags & 4) != 0;
    int at = 1;
    Value startV = hasStart ? d->args[at++] : NULL_VAL;
    Value stopV  = hasStop  ? d->args[at++] : NULL_VAL;
    Value stepV  = hasStep  ? d->args[at++] : NULL_VAL;
    bool ok = jaiSliceGet(d->args[0], startV, stopV, stepV,
                          hasStart, hasStop, hasStep, &d->result);
    jaiGCPopRootRange();
    return ok ? 0 : 1;
}

/* Dict store: unlike a list store there's no offset to normalise, it's a table probe either way --
 * this only saves the dispatch and indexSet's type ladder, not the probe itself. */
static int jitSetIndexDict(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    (void)jaiDictSet(AS_DICT(d->args[0]), d->args[1], d->args[2]);
    jaiGCPopRootRange();
    return vm.hasException ? 1 : 0;
}

static int jitCallOut(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    bool ok = jaiCallValue(d->callee, (int)d->argc, d->args, &d->result);
    jaiGCPopRootRange();
    return ok ? 0 : 1;
}

/* A `self` entry (the callee of a recursive call) occupies no register; register numbers are
 * derived from the count of value entries below an entry, not from its depth. */
typedef enum {
    SLOT_INT,
    SLOT_FLOAT,
    SLOT_INST,
    /* Fixed class or null, held as the pointer or zero (`x == null` is then a compare against zero);
 * refusing this stopped six hundred stdlib bodies. Cost: materialising picks VAL_NULL/VAL_OBJ off the register at run time -- the tag isn't a static property of the kind. */
    SLOT_MAYBE_INST,
    SLOT_SELF,
    SLOT_OPAQUE,  /* present in a register, but nothing may be done with it */
    SLOT_CLOSURE,
    SLOT_CLASS,
    SLOT_FUNC,
    SLOT_NATIVE,
    SLOT_ITER,    /* ObjIter this body built, held raw; its index stays in memory (not a register) -- costs a load/store
                   * per iteration but means a deopt needs no write-back, since the stack's iterator is always current. */
    SLOT_BOOL,    /* 0 or 1 in a register -- a Value's boolean member is its low byte, so the same word serves both. */
    SLOT_NULL,    /* What `-> void` returns: a defined zero in a register (droppable, or written out by a deopt) whose
                   * tag is VAL_NULL rather than the VAL_OBJ every other kind chain in this file falls through to. */
    SLOT_OBJ,     /* Heap object of a type this tier doesn't model, held raw: may only be read, passed, stored and rooted. */
    SLOT_LIST     /* ObjList *, raw -- safe for the same reason an instance is: nothing moves, and a call spills it as a root first. */
} SlotKind;

/* Floats live in X registers, visiting d0/d1 only for the arithmetic itself -- simpler-but-correct beats
 * a second register bank (own allocator/save-set/spill rules) for a tier this young. Nothing calls between the fmovs, so the scratch pair is safe. */
#define JIT_FSCRATCH_A 0u
#define JIT_FSCRATCH_B 1u

/* A SLOT_FLOAT entry may live in an FP register between the instruction that computes it and the one
 * that consumes it: entry i's canonical home is x(19+regBase+i), but `fpLive` bit i means that copy is stale and v(16+i) holds the value instead. Bank v16..v25 is caller-saved on purpose (nothing here survives a call); the index is shared with the X bank so two live entries can never collide. */
#define JIT_FP_BANK 16u

/* Same argument for a float LOCAL: v8..v15 is ABI-preserved across a call, nothing else here names it,
 * and only fmov/ldr d/str d ever touch a local's home -- never arithmetic -- so a slot parked here is a bit-exact 64-bit home (a kind the allocator guessed wrong costs an fmov, never wrong bits). */
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
    int      instIndex;
    uint32_t targetOffset; /* bytecode offset, or FIXUP_BAIL / FIXUP_ENTRY */
    bool     conditional;
    int      depth;
} Fixup;

typedef struct {
    uint32_t  code[JIT_MAX_INSTS];
    unsigned  count;

    SlotKind  stack[JIT_MAX_STACK];
    uint32_t  stackShape[JIT_MAX_STACK];
    ObjClass *stackClass[JIT_MAX_STACK];
    Value     stackSeen[JIT_MAX_STACK];
    int       stackLocal[JIT_MAX_STACK];
    /* This entry is not merely a string by sample -- it came out of the shared
     * one-byte ASCII table, so it IS an interned ObjString, and the guards a
     * string compare would otherwise emit for it are dead code. A proof, not a
     * prediction, so it may delete a guard rather than only choose one.
     *
     * A prediction survives a branch merge harmlessly (a wrong guess still
     * guards); a proof does not, because the other edge into a join carries a
     * value this walk never saw. The linear walk models only the fall-through
     * edge, so every flag is dropped at any offset something else can reach --
     * see the clearAsciiProofs call in the walk. */
    bool      stackAscii[JIT_MAX_STACK];

    /* Fields already stored this call, with their kind: a read of one needs no tag check since nothing
     * can have changed it (the body can't call) -- e.g. `self.n = self.n + 1; return self.n` would otherwise bail-after-write, which the tier refuses. */
    struct { int local; uint16_t field; SlotKind kind; } known[16];
    unsigned  knownCount;
    SlotKind  localKind[JIT_MAX_SLOTS + 1];
    uint32_t  localShape[JIT_MAX_SLOTS + 1];
    ObjClass *localClass[JIT_MAX_SLOTS + 1];
    bool      localTyped[JIT_MAX_SLOTS + 1];
    /* Kept only so a field read on this local has something to read the field's type off; a local bound
     * from a list element has no argument to look at (why nbody's `advance`'s `bi` couldn't have its fields read). */
    Value     localSeen[JIT_MAX_SLOTS + 1];
    Value    *observed;
    bool      assumedIntReturn;
    unsigned  depth;
    unsigned  valueDepth;
    unsigned  maxValue;
    unsigned  maxValueAll;

    Fixup     fixups[JIT_MAX_FIXUPS];
    unsigned  fixupCount;

    int      *offsetToInst;
    int      *offsetToDepth;

    unsigned  arity;
    /* Slot 0 is the callee for a plain call, the RECEIVER for a method (so `self.x` reads it). Touching
     * it turns it into an ordinary local plus an extra incoming argument; untouched (every plain function), it costs nothing. */
    unsigned  base;
    bool      usesSlot0;
    /* Highest slot the body actually names -- not maxSlots, the (routinely larger) frame window the
     * interpreter reserves. Every slot here costs one of the ten callee-saved registers. */
    unsigned  maxSlotUsed;
    /* The first pass exists to find maxValue and maxSlotUsed, so it must not
     * stop at a budget computed from a slot count it is still discovering. */
    bool      measuring;
    /* Locals spill to the frame instead of registers when the body has more live values than callee-saved
     * registers (nbody's `advance` wants nineteen); the operand stack always stays in registers. Flags that the PER-SLOT plan (slotXReg/slotFpReg) is in force: busiest slots keep a register, the rest live in the frame, as the OSR tier does. */
    bool      spilled;
    unsigned  localsFrameOffset;
    /* OSR: locals ARE the interpreter's frame slots via a pointer handed to the entry -- nothing is copied
     * either way, which is also what makes a deopt cheap here: only the operand stack needs rebuilding. */
    bool      osr;
    uint32_t  osrTop;
    uint32_t  osrEnd;
    /* `for i in a..b` compiled as a counted loop: the iterator object stays on the interpreter's stack
     * untouched, only its index rides in a register, and every exit writes it back. */
    bool      hasIter;
    uint8_t   iterKind;   /* 1 a unit-step range, 2 a list */
    Value     elemSample;
    /* Per-slot register assignment. Historically one flag gated the whole loop on a budget check that
     * almost nothing passed, so most loops ran fully in memory. Three things fixed it: only NAMED slots take a register; a float slot takes v8..v15 instead of an X register; overflow slots stay in the frame (busiest slots win, weighted by loop nesting) rather than all-or-nothing. */
    uint8_t   slotXReg[JIT_MAX_SLOTS + 1];   /* x19..x28, or 0 for none */
    uint8_t   slotFpReg[JIT_MAX_SLOTS + 1];  /* d8..d15, or 0 for none */
    unsigned  xLocals;
    unsigned  fpLocals;
    uint32_t  slotUse[JIT_MAX_SLOTS + 1];
    /* Same idea for the function tier, but counted in INSTRUCTIONS SAVED, not sites (see noteSlotCost):
     * a float read through the FP bank costs one `ldr d` plus one `fmov d,x` from an X register, so a flat per-use count would wrongly credit an X register for such a slot. Two banks, two ledgers. */
    uint32_t  slotSaveX[JIT_MAX_SLOTS + 1];
    uint32_t  slotSaveFp[JIT_MAX_SLOTS + 1];
    const uint8_t *loopDepth;
    unsigned  loopDepthCount;
    unsigned  fpSaveOffset;
    /* Where a range loop parks the ObjIter, since it holds no register for it -- written once in the prologue, read once per exit stub. */
    unsigned  iterFrameOffset;
    /* Inlining widens the live range of everything the callee reads, so a loop that fit the registers as
     * a call may not fit as an expression; the compile retries with this set when that's what went wrong. */
    bool      noInline;
    bool      inlined;
    /* Inlined callee: its locals are operand-stack entries of the CALLER's frame (slots 1..n are the
     * already-present argument entries); nothing is copied, no frame appears -- but the interpreter has no idea, so every guard inside deoptimises to `inlIp` (the caller's OP_CALL) with the model as of `inlDepth`. */
    bool      inlining;
    unsigned  inlDepth;
    unsigned  inlPinned;
    unsigned  inlValueBase;
    uint32_t  inlIp;
    /* Register holding the ObjClosure being inlined (-1 if none) -- the CALLEE's closure, not the caller's;
     * only meaningful while `inlining`, the only thing that writes it, so a zeroed Emit never reads stale state. */
    int       inlClosureReg;
    int       inlSlot[JIT_MAX_SLOTS + 1];
    /* A local whose kind differs across paths into some point: lives in the frame with its tag, and every
     * read guards. Not exotic -- the compiler reuses one slot for non-overlapping loop induction variables (nbody's advance). */
    bool      dynamicLocal[JIT_MAX_SLOTS + 1];
    bool      needDynamic[JIT_MAX_SLOTS + 1];
    /* A slot holding an instance on one path, null on another: unlike a dynamic local it stays in a register
     * (pointer or zero), tag built at materialisation. Per-slot, since widening every nullable-mentioning parameter cost object_dispatch 2x. */
    bool      nullableLocal[JIT_MAX_SLOTS + 1];
    bool      needNullable[JIT_MAX_SLOTS + 1];
    bool      pendingRange;
    bool      rangeInclusive;
    /* Whether the pending range's low end was an integer literal, and which
     * one. A range this body builds always steps by 1 (jitMakeRangeIter takes
     * no step), so with the start known too the value a nested FOR_ITER_BIND
     * yields is `index + K` -- no ObjRange to reach through at all. */
    bool      rangeStartKnown;
    int64_t   rangeStartVal;
    /* Where the deferred OP_BUILD_RANGE was, so a guard inside the header it
     * folded into can resume at an offset the model still describes. */
    uint32_t  rangeBuildIp;
    unsigned  iterSlot;
    uint32_t  iterExit;
    int       exitStub[8];
    uint32_t  exitOffset[8];
    unsigned  exitCount;
    /* A body that reads an upvalue needs the closure itself -- not any slot (a method's slot 0 is the
     * receiver, not the callee) -- so it arrives as one extra argument, in the register just past the locals. */
    bool      usesUpvalues;
    /* Set once the body has written to the heap. After that a bail is no
     * longer free: re-running the call interpreted would apply the write a
     * second time. See branchOnCondition. */
    bool      wroteHeap;
    /* Baked globals share the defining module's table, so one `keyVersion` guard covers all of them --
     * it changes only when a live entry's address or key could move (new key, rehash, delete, clear). ObjModule::version (used by the function-tier entry check) is neither necessary nor sufficient here. */
    JaiTable *globalsTable;
    uint32_t  globalsKeyVersion;
    bool      bailAfterWrite;
    bool      callsOut;
    /* Set the moment anything is emitted that can destroy x0..x8 while the
     * body is still running: a call out, or one of the two stubs that call and
     * then branch BACK into the body (list grow, self-call slow path). It is
     * what `scratchValues` below is decided from, and the measuring pass is
     * what observes it. */
    bool      clobbersScratch;
    /* The operand stack lives in x0..x8 rather than above the locals in the
     * callee-saved bank. Sound exactly when nothing can clobber a caller-saved
     * register between a push and its use, i.e. `!clobbersScratch` and the
     * whole stack -- an inlined body's entries included, which is what
     * maxValueAll counts -- fits the nine. Worth having because
     * the two banks were competing for the same ten registers: a stencil whose
     * expression is seven deep left NOTHING for its four row pointers and its
     * index, and reloaded all five from the frame every iteration. */
    bool      scratchValues;
    /* probe.clobbersScratch, carried into the real pass. `no call anywhere in
     * this body` is a stronger statement than `the values may live in x0..x8`
     * -- scratchValues also wants the stack to fit nine -- and the hoist below
     * needs the stronger one. */
    bool      bodyCalls;
    /* Where the measuring pass saw each slot written, and where it saw each one
     * used as the base of a subscript. Bounds rather than a set: a loop is a
     * contiguous range of offsets, so "written inside this loop" is a range
     * overlap, and widening a range can only lose a hoist, never make a wrong
     * one. lo > hi means "never". */
    uint32_t  slotWriteLo[JIT_MAX_SLOTS + 1];
    uint32_t  slotWriteHi[JIT_MAX_SLOTS + 1];
    uint32_t  slotIndexLo[JIT_MAX_SLOTS + 1];
    uint32_t  slotIndexHi[JIT_MAX_SLOTS + 1];
    unsigned  slotIndexUse[JIT_MAX_SLOTS + 1];
    /* Loop-invariant list headers. See planHoists. */
    struct {
        uint32_t top, end;
        uint8_t  slot, itemsReg, countReg;
    } hoist[JIT_MAX_HOIST];
    unsigned  hoistCount;
    uint8_t   hoistPool[JIT_FREE_COUNT + JIT_SCRATCH_BANK_COUNT];
    unsigned  hoistPoolCount;
    unsigned  hoistTaken;
    /* How many entries the scratch bank may still hold once the hoists have
     * taken theirs off the top. The probe said the body never goes deeper, but
     * "said" is not "cannot": lowering the room here is what turns a walk that
     * disagreed with the probe into a decline instead of a value read out of a
     * register a hoisted header owns. */
    unsigned  scratchRoom;
    const char *whyNot;
    uint8_t     lastOp;
    unsigned  descOffset;
    int       exceptionExit;
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
    /* Self-call cold path, emitted with the stubs, not inline: the fast path is three instructions plus a
     * not-taken branch; interleaving thirty instructions of cold code between recursive call sites cost fib_recursive 25% (measured, same code, two layouts). */
    struct {
        int      returnTo;
        int      stub;
        unsigned roots;      /* the descriptor is on the frame chain when >0 */
        unsigned deoptBail;  /* record for verdict 1: re-execute the call */
        unsigned deoptKind;  /* record for a result of an unexpected kind */
        unsigned tag;
        unsigned resultReg;
        /* Closure to finish on verdict 4: NULL means this one (a self-call knows its own); a direct call to
         * another compiled function that writes names it here instead. */
        ObjClosure *callee;
        /* Continuation's required object type/shape, when the fast path's kind says more than "a heap object":
         * VAL_OBJ covers every heap object, and reading `klass` off an ObjString answers wrongly rather than faulting (the list-element-head bug) -- so the type is checked first. */
        int      retType;
        uint32_t retShape;
    } selfSlow[JIT_MAX_SELF_SLOW];
    unsigned  selfSlowCount;
    /* Out-of-line list-grow path (like the self-call cold half, to keep the store's 9 instructions and a
     * realloc call from sitting in the loop). Before this existed, list-full was a full deopt -- handing the WHOLE rest of the function to the interpreter -- which made compiled list-building bodies nearly worthless (merge in sort_merge deopted on every single push, 62434 times for 62500 items). */
    struct {
        int      returnTo;
        int      stub;
        unsigned listReg;
        unsigned valReg;
        unsigned tag;
        unsigned countReg;
    } grow[JIT_MAX_GROW];
    unsigned  growCount;
    uint32_t  curOffset;
    /* Model as it stood at the start of `curOffset`, before that instruction's own pushes -- a guard fires
     * mid-instruction and the interpreter resumes at its start, so this is what's live there (see deoptSite). */
    unsigned  instDepth;
    unsigned  instValueDepth;
    bool      hasSelfCall;
    unsigned  locals;
    unsigned  frameBytes;
    unsigned  savedCount;

    int       limitLiteral;
    int       bailBlock;

    SlotKind  returnKind;
    uint32_t  returnShape;
    bool      sawReturn;

    bool      failed;

    /* Value entries whose payload is in v(16 + index) rather than in their X
     * register. See JIT_FP_BANK. */
    uint32_t  fpLive;
    bool      fpOff;
    /* Offsets this walk carried an FP-resident value INTO: a branch landing on one would arrive with the
     * value only in X, so the compile declines and retries with fpOff. Safety net, not a real path -- straight-line float expressions are never branch targets; JAI_JIT_WHY reports it when it fires. */
    uint32_t  fpCarry[64];
    unsigned  fpCarryCount;
    /* Offsets of an OP_BIND whose local's d register was written EARLY by the float operator just above it,
     * so the bind itself emits nothing -- nothing may branch here, since an arriving path skipped the operator. Checked against fixups at the end of the walk, since a back-edge isn't in the list yet mid-walk (same reason as fpCarry). */
    uint32_t  homeEarly[32];
    unsigned  homeEarlyCount;
    /* Value entries that are a plain read of a float local, held in that LOCAL's own d register instead of
     * copied into the bank (`x * x` becomes one multiply instead of two fmovs + a multiply). Read-only borrow: only localOut/localOutFp ever write a local's d home, and both release the borrow first. Cleared on push/pop/claim/before any deopt record. */
    uint32_t  fpBorrow;
    uint8_t   fpBorrowReg[32];

    /* Two deferred-materialisation flavors, one mechanism: kPend is an int literal (OP_INT/OP_CONST) left
     * unmaterialised so a consumer can fold it as an imm12 (`n - 1` becomes just `subs`); xBorrow is a plain read of a register-resident local held in THAT register (X-twin of fpBorrow, `fib(n-1)` loses its `mov`). Both settle at the TOP of the next instruction (the one point unconditionally on the executed path -- a guard is not) unless whitelisted by deferSurvives. Nothing deferred may reach a deopt record: deoptRecordAt/branchOnDeopt assert it, so a wrong whitelist entry costs a decline, not a wrong answer. */
    uint32_t  kPend;
    int64_t   kPendVal[32];
    /* Which entries are known to BE a given integer literal, whether or not
     * they were materialised. kPend answers "may I fold this into the next
     * instruction" and is gone the moment the literal reaches a register;
     * this answers "what is this value", which stays true afterwards. Set
     * where a literal is pushed, cleared with every other per-entry mask on
     * push and pop, so it cannot outlive the entry it describes. */
    uint32_t  kKnown;
    int64_t   kKnownVal[32];
    uint32_t  xBorrow;
    uint8_t   xBorrowReg[32];
    /* Offsets this walk carried a deferred entry into (same reason as fpCarry): a BACKWARD branch there
     * would arrive with the value in-register while the instruction reads the borrow. Forward branches are caught during the walk; this catches the rest. */
    uint32_t  deferCarry[64];
    unsigned  deferCarryCount;
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
/* The ObjList a list head is walking. A range head has no use for a third register (see osrReserved),
 * so this is numbered after the two a range keeps, and a range's locals begin where it would have been. */
#define JIT_START_REG (JIT_FIRST_SAVED + 3u)
#define JIT_ITER_REG  (JIT_FIRST_SAVED + 4u)

/* OSR reserves only the slots pointer plus (for a range loop) the iterator's index and limit --
 * not the ObjIter or the start, which are folded in via a prologue bias. Bias is sound only while start+limit fits int64; jaiJitEnterOsr refuses entry otherwise since that's a property of the iterator, not the code. */
static unsigned osrReserved(const Emit *e) {
    if (!e->hasIter) return 1u;
    return e->iterKind == 1 ? 3u : 5u;
}

static unsigned regBase(const Emit *e) {
    if (e->osr) return osrReserved(e) + e->xLocals;
    return e->spilled ? e->xLocals : e->locals;
}

static unsigned localReg(const Emit *e, unsigned slot) {
    if (e->osr || e->spilled) return e->slotXReg[slot];
    return JIT_FIRST_SAVED + (slot - e->base);
}

/* Sixteen bytes per slot, not eight: the tag travels with the value so a local whose kind varies can
 * be read behind a guard. Every slot keeps a frame home even with a register too -- dense layout costs one multiply instead of a table, and the wasted words are nothing against the 4095-byte frame limit. */
static unsigned localFrameOff(const Emit *e, unsigned slot) {
    return e->localsFrameOffset + (slot - e->base) * 16u;
}

/* A frame home only needs a stored tag when something will READ it: a fixed-kind slot's tag is
 * rebuildable by the deopt stub from the kind, cutting a spilled write from three instructions to one. Only a dynamic slot (kind disagrees across paths) has a runtime tag, and only its reads check it. */
static bool localTagInFrame(const Emit *e, unsigned slot) {
    return e->dynamicLocal[slot];
}

/* Per-access instruction savings from giving `slot` a register instead of a frame home, accumulated
 * by the accessors during the measuring pass and weighted by loop nesting. Zero-saving slots are excluded outright, not just ranked last -- otherwise a tie-break could hand a register to a slot the loop never touches, paying a prologue write for nothing. */
static void noteSlotCost(Emit *e, unsigned slot, unsigned saveX,
                         unsigned saveFp) {
    if (!e->measuring || e->osr || e->inlining) return;
    if (slot > JIT_MAX_SLOTS) return;
    unsigned w = 1u;
    if (e->loopDepth != NULL && e->curOffset < e->loopDepthCount) {
        unsigned d = e->loopDepth[e->curOffset];
        if (d > 6u) d = 6u;
        w = 1u << (2u * d);
    }
    e->slotSaveX[slot]  += w * saveX;
    e->slotSaveFp[slot] += w * saveFp;
}

static void branchOnDeopt(Emit *e, unsigned cond);
static void fpReleaseHome(Emit *e, unsigned reg);

/* Every kind but SLOT_MAYBE_INST has a tag fixed at compile time; that one reads it off the payload (null vs non-null). */
static void emitTagFor(Emit *e, SlotKind kind, unsigned payloadReg,
                       unsigned tagReg, unsigned spare) {
    if (kind != SLOT_MAYBE_INST) {
        unsigned tag = kind == SLOT_INT    ? VAL_INT
                     : kind == SLOT_FLOAT  ? VAL_FLOAT
                     : kind == SLOT_BOOL   ? VAL_BOOL
                     : kind == SLOT_OPAQUE ? VAL_NULL
                     : kind == SLOT_NULL   ? VAL_NULL
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
         : k == SLOT_NULL   ? VAL_NULL
                            : VAL_OBJ;
}

/* Register mode: the local's own register, `scratch` unused. Memory mode: loaded into `scratch`.
 * Only the payload moves -- a local's kind is fixed for the whole function, so the tag is never restored. */
static unsigned localIn(Emit *e, unsigned slot, unsigned scratch) {
    if (e->osr) {
        if (e->slotXReg[slot] != 0) return e->slotXReg[slot];
        if (e->slotFpReg[slot] != 0) {
            emit(e, jaiA64FmovXD(scratch, e->slotFpReg[slot]));
            return scratch;
        }
        /* One byte for a bool. The slot is the interpreter's, and BOOL_VAL is a
         * `strb`, so the seven bytes above it are whatever the slot held
         * before. See the OSR prologue for the same load and what it cost. */
        if (e->localKind[slot] == SLOT_BOOL) {
            emit(e, jaiA64LdrByte(scratch, JIT_SLOTS_REG, slot * 16u + 8u));
        } else {
            emit(e, jaiA64LdrX(scratch, JIT_SLOTS_REG, slot * 16u + 8u));
        }
        return scratch;
    }
    noteSlotCost(e, slot, 1u, 0u);
    if (!e->spilled) return localReg(e, slot);
    if (e->slotXReg[slot] != 0) return e->slotXReg[slot];
    if (e->slotFpReg[slot] != 0) {
        emit(e, jaiA64FmovXD(scratch, e->slotFpReg[slot]));
        return scratch;
    }
    if (e->dynamicLocal[slot]) {
        /* Two paths reached here disagreeing about this slot, so what it holds
         * is a runtime fact: check it against what this read was compiled for
         * and hand the instruction back otherwise. */
        /* Into `scratch`, not JIT_SCRATCH_A, since the caller may already be holding an operand there: this
         * exact mistake once loaded the tag over the constant on a dynamic slot, so `for j in i + 1..n` silently ran from i+2 and every nested loop was one iteration short. */
        emit(e, jaiA64LdrW(scratch, 31, localFrameOff(e, slot)));
        emit(e, jaiA64SubsXImm(31, scratch, localTagFor(e, slot)));
        branchOnDeopt(e, JAI_A64_NE);
    }
    emit(e, jaiA64LdrX(scratch, 31, localFrameOff(e, slot) + 8));
    return scratch;
}

/* Only register-resident locals can be written in place. In OSR mode `localReg` would name a register
 * already holding something else (the loop counter) -- once let a float add land in x21, turning the induction variable into a bit pattern, so the loop finished early or never finished. */
static unsigned localDest(const Emit *e, unsigned slot) {
    if (e->osr) {
        return e->slotXReg[slot] != 0 ? e->slotXReg[slot] : JIT_SCRATCH_C;
    }
    if (e->spilled) {
        return e->slotXReg[slot] != 0 ? e->slotXReg[slot] : JIT_SCRATCH_C;
    }
    return localReg(e, slot);
}

/* A local's X register is about to be written, so nothing may still be borrowing it. Can't just copy
 * the borrow out here (same reason as deoptRecordAt: the write can sit inside a span an earlier branch skips) -- declines instead. Can't fire from the whitelist as it stands: no whitelisted opcode writes a local. */
static void xHomeWritten(Emit *e, unsigned reg) {
    if (e->xBorrow == 0 || reg == 0) return;
    for (unsigned i = 0; i < 32u; i++) {
        if ((e->xBorrow & (1u << i)) == 0) continue;
        if (e->xBorrowReg[i] != reg) continue;
        e->whyNot = "a local was written under a borrow of its register";
        e->failed = true;
        return;
    }
}

/* Every write to a local goes through localOut or localOutFp, so recording it
 * in those two places is what makes "this slot does not change inside that
 * loop" a fact about the emitter rather than a re-reading of the bytecode. */
static void noteSlotWrite(Emit *e, unsigned slot) {
    if (slot > JIT_MAX_SLOTS) return;
    if (!e->measuring) {
        /* The real pass writing a slot the measuring pass said this loop never
         * touches means the two walks disagreed, and a hoisted header would
         * then be stale. It cannot happen -- both walk the same bytecode with
         * the same inlining -- so this is a ratchet, not a path: it costs a
         * compile, never a wrong answer. */
        for (unsigned i = 0; i < e->hoistCount; i++) {
            if (e->hoist[i].slot != (uint8_t)slot) continue;
            if (e->curOffset < e->hoist[i].top) continue;
            if (e->curOffset >= e->hoist[i].end) continue;
            e->whyNot = "a hoisted list header's local was written after all";
            e->failed = true;
        }
        return;
    }
    if (e->inlining) return;
    if (e->curOffset < e->slotWriteLo[slot]) e->slotWriteLo[slot] = e->curOffset;
    if (e->curOffset > e->slotWriteHi[slot]) e->slotWriteHi[slot] = e->curOffset;
}

/* Twin of noteSlotWrite for the other half of the question: where a local is
 * used as the base of a subscript, and how hot those sites are. Weighted by
 * loop nesting for the same reason slotUse is -- a header read once per row
 * must not outrank one read every iteration. */
static void noteSlotIndexed(Emit *e, int slot) {
    if (!e->measuring || e->inlining || slot < 0 || slot > (int)JIT_MAX_SLOTS) {
        return;
    }
    unsigned w = 1u;
    if (e->loopDepth != NULL && e->curOffset < e->loopDepthCount) {
        unsigned d = e->loopDepth[e->curOffset];
        if (d > 6u) d = 6u;
        w = 1u << (2u * d);
    }
    e->slotIndexUse[slot] += w;
    if (e->curOffset < e->slotIndexLo[slot]) e->slotIndexLo[slot] = e->curOffset;
    if (e->curOffset > e->slotIndexHi[slot]) e->slotIndexHi[slot] = e->curOffset;
}

static void localOut(Emit *e, unsigned slot, unsigned src) {
    noteSlotWrite(e, slot);
    xHomeWritten(e, e->osr ? e->slotXReg[slot]
                           : (e->spilled ? 0u : localReg(e, slot)));
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
        /* Tag as well as payload, written straight through -- what makes a deopt here free; the kind is fixed for the whole compile. */
        SlotKind k = e->localKind[slot];
        emitTagFor(e, k, src, JIT_SCRATCH_D, JIT_SCRATCH_C);
        emit(e, jaiA64StrW(JIT_SCRATCH_D, JIT_SLOTS_REG, slot * 16u));
        emit(e, jaiA64StrX(src, JIT_SLOTS_REG, slot * 16u + 8u));
        return;
    }
    noteSlotCost(e, slot, 1u, 0u);
    if (!e->spilled) {
        if (src != localReg(e, slot)) emit(e, jaiA64MovX(localReg(e, slot), src));
        return;
    }
    if (e->slotFpReg[slot] != 0) fpReleaseHome(e, e->slotFpReg[slot]);
    if (e->slotXReg[slot] != 0) {
        if (src != e->slotXReg[slot]) emit(e, jaiA64MovX(e->slotXReg[slot], src));
        return;
    }
    if (e->slotFpReg[slot] != 0) {
        emit(e, jaiA64FmovDX(e->slotFpReg[slot], src));
        return;
    }
    if (localTagInFrame(e, slot)) {
        emitTagFor(e, e->localKind[slot], src, JIT_SCRATCH_D, JIT_SCRATCH_C);
        emit(e, jaiA64StrW(JIT_SCRATCH_D, 31, localFrameOff(e, slot)));
    }
    emit(e, jaiA64StrX(src, 31, localFrameOff(e, slot) + 8));
}

/* Float half of localIn/localOut: an FP-bank local never visits an X register at all (memory mode
 * collapses `ldr x; fmov d,x` to one `ldr d`, and the store side likewise) -- only for a fixed-kind local; a dynamic one has a tag to check and goes the ordinary (X) way. */
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
    noteSlotCost(e, slot, 0u, 1u);
    if (!e->spilled) {
        emit(e, jaiA64FmovDX(dst, localReg(e, slot)));
        return;
    }
    if (e->slotFpReg[slot] != 0) {
        if (dst != e->slotFpReg[slot]) {
            emit(e, jaiA64FmovDD(dst, e->slotFpReg[slot]));
        }
        return;
    }
    if (e->slotXReg[slot] != 0) {
        emit(e, jaiA64FmovDX(dst, e->slotXReg[slot]));
        return;
    }
    emit(e, jaiA64LdrD(dst, 31, localFrameOff(e, slot) + 8));
}

static void localOutFp(Emit *e, unsigned slot, unsigned src) {
    noteSlotWrite(e, slot);
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
    /* No noteSlotCost: a float write is one instruction whichever home it has
     * (`fmov d,d`, `fmov x,d` or `str d`), so it makes no case for a register.
     * The reads are what pay. */
    if (!e->spilled) {
        emit(e, jaiA64FmovXD(localReg(e, slot), src));
        return;
    }
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
    if (localTagInFrame(e, slot)) {
        emit(e, jaiA64MovzX(JIT_SCRATCH_D, VAL_FLOAT, 0));
        emit(e, jaiA64StrW(JIT_SCRATCH_D, 31, localFrameOff(e, slot)));
    }
    emit(e, jaiA64StrD(src, 31, localFrameOff(e, slot) + 8));
}

static bool localInRange(Emit *e, unsigned slot) {
    if (e->osr) {
        if (slot >= e->locals) return false;
        if (slot > e->maxSlotUsed) e->maxSlotUsed = slot;
        /* Nested-loop sites are worth more: flat counting would give `width`/`height` (read once per row)
         * the same claim as the recurrence variable read every iteration. */
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

static bool localObserved(Emit *e, unsigned slot) {
    if (e->osr) return slot < e->locals;
    if (slot < e->base || slot > e->arity) return false;
    if (slot > e->maxSlotUsed) e->maxSlotUsed = slot;
    return true;
}

static unsigned closureReg(const Emit *e) {
    return JIT_FIRST_SAVED + regBase(e);
}

/* Counts from the bottom of the operand stack, not the top (unlike pushReg). An inlined body gets its
 * own bank (x0..x8, minus the emitter's x9..x12 scratches): it can't call anything, so every caller-saved register is free and costs the caller nothing -- `evalA` inlined into spectral's inner loop needs eight live values where the OSR form had only six left, so without this it wouldn't fit. */
#define JIT_INL_BANK   0u    /* x0..x8, all caller-saved */
#define JIT_INL_COUNT  JIT_SCRATCH_BANK_COUNT

/* Where entry 0 of the ordinary operand stack sits. Two answers, one rule: it
 * is x0 when the body proved nothing can clobber a caller-saved register while
 * it runs (see Emit::scratchValues), and otherwise the register just past the
 * locals, where it has always been. Every site that used to spell this sum out
 * by hand goes through here, so the two banks cannot disagree. */
static unsigned valueBankBase(const Emit *e) {
    if (e->scratchValues) return JIT_INL_BANK;
    return JIT_FIRST_SAVED + regBase(e) + (e->usesUpvalues ? 1u : 0u);
}

/* How many entries the operand stack may hold. The scratch bank is a fixed
 * nine; the callee-saved one is whatever the locals and the reserved registers
 * left behind. */
static unsigned valueBankRoom(const Emit *e) {
    if (e->scratchValues) return e->scratchRoom;
    unsigned taken = regBase(e) + (e->usesUpvalues ? 1u : 0u);
    return taken < JIT_MAX_SAVED ? JIT_MAX_SAVED - taken : 0u;
}

/* Something about to be emitted can destroy x0..x8 with the body still live:
 * a call out, or one of the two stubs that call and then branch back in. The
 * measuring pass records it so the real pass can choose a bank; the real pass
 * declines if it happens anyway, so a missed site costs a decline and never a
 * value read out of a register a helper overwrote. */
static void noteScratchClobber(Emit *e) {
    e->clobbersScratch = true;
    if (e->scratchValues) {
        e->whyNot = "a call reached a body whose values are in scratch";
        e->failed = true;
    }
}

/* An inlined body needs its own bank only when the caller's is somewhere else.
 * Once the caller's operand stack is already x0..x8 the two are the SAME bank,
 * and restarting at x0 would overwrite the entries the call site is standing
 * on -- so the inlined entries simply continue the caller's numbering, which
 * `scratchValues` has already proved fits (probe.maxValueAll <= the bank). */
static bool inlineOwnBank(const Emit *e) {
    return e->inlining && !e->scratchValues;
}

static unsigned valueXReg(const Emit *e, unsigned idx) {
    if (inlineOwnBank(e) && idx >= e->inlValueBase) {
        return JIT_INL_BANK + (idx - e->inlValueBase);
    }
    return valueBankBase(e) + idx;
}

/* One past the top entry's register. Expressing it this way (not from the bottom) is what keeps
 * `pushReg(e) - 1` == "the entry just pushed" true across both register banks. */
static unsigned pushReg(const Emit *e) {
    if (e->valueDepth == 0) return valueXReg(e, 0);
    return valueXReg(e, e->valueDepth - 1) + 1;
}

/* v16.. for the ordinary bank, v2..v7 for an inlined body -- caller-saved, and
 * far enough from v16+JIT_MAX_SAVED, which the local-add path uses as a temp. */
#define JIT_INL_FP_BANK 2u

static unsigned fpRegAt(const Emit *e, unsigned idx) {
    if (inlineOwnBank(e) && idx >= e->inlValueBase) {
        return JIT_INL_FP_BANK + (idx - e->inlValueBase);
    }
    return JIT_FP_BANK + idx;
}

/* The d register entry `idx` is actually in: its own, or a borrowed local's. Every READ of a live FP
 * entry goes through this; writes keep naming fpRegAt, which is what a borrow releases back into. */
static unsigned fpHeldIn(const Emit *e, unsigned idx) {
    if (e->fpBorrow & (1u << idx)) return e->fpBorrowReg[idx];
    return fpRegAt(e, idx);
}

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

/* d register holding entry `idx`, loaded from X if that's where it still lives. Leaves fpLive
 * untouched -- the two copies now agree, and marking the X copy stale when it isn't would cost a needless sync. */
static unsigned xHeldIn(Emit *e, unsigned idx);

static unsigned fpOperand(Emit *e, unsigned idx) {
    if (e->fpBorrow & (1u << idx)) return e->fpBorrowReg[idx];
    unsigned d = fpRegAt(e, idx);
    if (!(e->fpLive & (1u << idx))) {
        /* xHeldIn, not valueXReg: a float entry that's a plain read of an X-resident local lives in THAT
         * register; reading its own would read whatever the bank last held. */
        emit(e, jaiA64FmovDX(d, xHeldIn(e, idx)));
    }
    return d;
}

/* Entry `idx` has just been computed into v(16 + idx); its X register is now
 * stale until something asks for it. */
static void fpClaim(Emit *e, unsigned idx) {
    e->fpLive |= 1u << idx;
    e->fpBorrow &= ~(1u << idx);
}

static void fpBorrowLocal(Emit *e, unsigned idx, unsigned reg) {
    e->fpLive |= 1u << idx;
    e->fpBorrow |= 1u << idx;
    e->fpBorrowReg[idx] = (uint8_t)reg;
}

/* Float op writing straight to a local's home instead of the bank + fmov (saves the trailing `fmov`
 * on every `*_BIND`). Safe only because the borrow release happens HERE, before the operator, not after in localOutFp -- a borrower wants the pre-operator value, and since arm64 reads sources before writing its destination, `sum += x` naming the home among its own sources is fine once the borrow is already released. No FP register on the local: bank returned unchanged, so callers see one shape either way. */
static unsigned fpBindDest(Emit *e, unsigned slot, unsigned bank) {
    if (!e->osr || e->slotFpReg[slot] == 0) return bank;
    fpReleaseHome(e, e->slotFpReg[slot]);
    return e->slotFpReg[slot];
}

/* Fuses an OP_ADD/OP_SUB immediately followed by OP_BIND to a float local, by aiming the operator's
 * result straight at the local's home register instead of the bank + fmov. Only safe when nothing between them can deopt/branch into the gap (tracked via homeEarly, checked against fixups post-walk) and fpBindDest already gave the local sole ownership of the register. Returns 0 (ordinary path) otherwise. */
static unsigned fpBindLookahead(Emit *e, const uint8_t *code, int next,
                                int stop, const ObjFunction *fn,
                                uint32_t *bindOffOut) {
    if (!e->osr || e->fpOff || e->inlining) return 0;
    if (e->homeEarlyCount >= 32) return 0;
    if (next < stop && code[next] == OP_TYPE_GUARD) {
        /* Only the settled form: a guard that widens an int emits `scvtf` (making the entry an int, so this
         * arm is never reached); a guard naming any other type declines below. */
        if (next + 4 > stop) return 0;
        uint32_t idx = jaiReadU24(code + next + 1);
        if (idx >= (uint32_t)fn->chunk.constants.count) return 0;
        Value t = fn->chunk.constants.data[idx];
        if (!IS_STRING(t) || strcmp(AS_STRING(t)->chars, "float") != 0) return 0;
        next += 4;
    }
    if (next + 3 > stop || code[next] != OP_BIND) return 0;
    unsigned slot = jaiReadU16(code + next + 1);
    /* Deliberately NOT localInRange: that one counts a use for the register
     * allocator, and a slot merely peeked at has not been used. The bound it
     * would apply is applied here instead. */
    if (slot > JIT_MAX_SLOTS || slot >= e->locals) return 0;
    if (e->dynamicLocal[slot] || e->slotFpReg[slot] == 0) return 0;
    *bindOffOut = (uint32_t)next;
    return e->slotFpReg[slot];
}

/* ------------------------------------------------------------------ */
/* Deferred X entries: pending constants and borrowed locals            */
/* ------------------------------------------------------------------ */

static void emitConst64(Emit *e, unsigned rd, int64_t value);

static bool anyDeferred(const Emit *e) {
    return (e->kPend | e->xBorrow) != 0;
}

/* Puts entry `idx` in its own register. Both forms are one instruction and neither touches NZCV
 * (movz/movk, or `orr xd,xzr,xs`), so this stays safe between a compare and the branch reading its flags. */
static void settleEntry(Emit *e, unsigned idx) {
    if (e->kPend & (1u << idx)) {
        e->kPend &= ~(1u << idx);
        emitConst64(e, valueXReg(e, idx), e->kPendVal[idx]);
        return;
    }
    if (e->xBorrow & (1u << idx)) {
        unsigned src = e->xBorrowReg[idx];
        e->xBorrow &= ~(1u << idx);
        emit(e, jaiA64MovX(valueXReg(e, idx), src));
    }
}

static void settleAll(Emit *e) {
    if (!anyDeferred(e)) return;
    for (unsigned i = 0; i < 32u; i++) settleEntry(e, i);
}

/* Register entry `idx` may be READ from. A borrow answers with the local's own register for free; a
 * pending constant must first be materialised -- exactly the instruction deferral was avoiding -- so an arm that can fold checks kPend before calling this. */
static unsigned xHeldIn(Emit *e, unsigned idx) {
    if (e->xBorrow & (1u << idx)) return e->xBorrowReg[idx];
    if (e->kPend & (1u << idx)) settleEntry(e, idx);
    return valueXReg(e, idx);
}

/* X register a local permanently lives in, or 0 when it lives nowhere a stack entry could borrow
 * (a spilled frame slot, or an OSR slot the register plan left in memory). */
static unsigned localHomeX(const Emit *e, unsigned slot) {
    if (e->osr) return e->slotXReg[slot];
    if (e->spilled) return 0u;
    return localReg(e, slot);
}

static void xBorrowLocal(Emit *e, unsigned idx, unsigned reg) {
    e->xBorrow |= 1u << idx;
    e->xBorrowReg[idx] = (uint8_t)reg;
}

static void kPendLocal(Emit *e, unsigned idx, int64_t k) {
    e->kPend |= 1u << idx;
    e->kPendVal[idx] = k;
}

/* The pending literal on top, when it fits the imm12 both `adds` and `subs`
 * take. Reports it without consuming it: the caller folds only once it knows
 * the rest of the shape allows it. */
static bool pendingImm12(const Emit *e, unsigned idx, int64_t *out) {
    if ((e->kPend & (1u << idx)) == 0) return false;
    int64_t k = e->kPendVal[idx];
    if (k < -4095 || k > 4095) return false;
    *out = k;
    return true;
}

/* `rn - k`, flags only, in one instruction. A negative `k` becomes `cmn`,
 * which subtracts it just as exactly -- the same trick OP_INC_LOCAL uses for a
 * negative step, and for the same reason: both forms set V for the operation
 * actually performed. */
static void emitCmpImm(Emit *e, unsigned rn, int64_t k) {
    if (k >= 0) emit(e, jaiA64SubsXImm(31, rn, (unsigned)k));
    else        emit(e, jaiA64AddsXImm(31, rn, (unsigned)(-k)));
}

/* `rd = rn + k` for OP_ADD, `rd = rn - k` for OP_SUB, with V set for the sum
 * or difference that was actually asked for. */
static void emitAddSubImm(Emit *e, unsigned rd, unsigned rn, int64_t k,
                          bool subtract) {
    bool down = subtract ? (k >= 0) : (k < 0);
    unsigned m = (unsigned)(k >= 0 ? k : -k);
    if (down) emit(e, jaiA64SubsXImm(rd, rn, m));
    else      emit(e, jaiA64AddsXImm(rd, rn, m));
}

static bool pushValue3(Emit *e, SlotKind kind, uint32_t shape, ObjClass *klass,
                       Value seen, int fromLocal) {
    if (e->depth >= JIT_MAX_STACK) {
        e->whyNot = "the operand stack is deeper than the model allows";
        return false;
    }
    if (!e->measuring && inlineOwnBank(e) &&
        e->valueDepth + 1u - e->inlValueBase > JIT_SCRATCH_BANK_COUNT) {
        e->whyNot = "an inlined body wants more registers than a call leaves free";
        return false;
    }
    if (!e->measuring && !inlineOwnBank(e) &&
        e->valueDepth + 1 > valueBankRoom(e)) {
        e->whyNot = "more live values than there are callee-saved registers";
        return false;
    }
    e->stackShape[e->depth] = shape;
    e->stackClass[e->depth] = klass;
    e->stackSeen[e->depth]  = seen;
    e->stackLocal[e->depth] = fromLocal;
    e->stackAscii[e->depth] = false;
    e->stack[e->depth++] = kind;
    e->fpLive   &= ~(1u << e->valueDepth);
    e->fpBorrow &= ~(1u << e->valueDepth);
    e->kPend    &= ~(1u << e->valueDepth);
    e->kKnown   &= ~(1u << e->valueDepth);
    e->xBorrow  &= ~(1u << e->valueDepth);
    e->valueDepth++;
    /* An inlined body's entries are not in the caller's bank, so they do not
     * widen its save set -- which is the whole reason they fit. */
    if (e->valueDepth > e->maxValue &&
        !(e->inlining && e->valueDepth > e->inlValueBase)) {
        e->maxValue = e->valueDepth;
    }
    /* The same number counted the other way: how wide the stack gets when the
     * inlined entries are NOT given a bank of their own. That is the question
     * "may this body's values live in x0..x8" asks, and maxValue cannot answer
     * it -- it deliberately stops counting at the inline boundary. */
    if (e->valueDepth > e->maxValueAll) e->maxValueAll = e->valueDepth;
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

/* Records a store and drops what any OTHER receiver claimed about the same field: two locals can
 * alias the same object, so a store through one must retire the other's knowledge of it. */
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
    e->stackAscii[e->depth] = false;
    e->stack[e->depth++] = SLOT_SELF;
    return true;
}

/* Retire every "came out of the ASCII table" proof. See Emit::stackAscii. */
static void clearAsciiProofs(Emit *e) {
    for (unsigned i = 0; i < JIT_MAX_STACK; i++) e->stackAscii[i] = false;
}

static bool anyAsciiProof(const Emit *e) {
    for (unsigned i = 0; i < e->depth; i++) {
        if (e->stackAscii[i]) return true;
    }
    return false;
}

/* Pop an entry that has already been read out of its FP register, or that was
 * never in one. Only the float paths may call this; everything else goes
 * through popValue, which materialises first. */
static bool popValueRaw(Emit *e, unsigned *reg, SlotKind *kind) {
    if (e->depth == 0 || !holdsRegister(e->stack[e->depth - 1])) return false;
    e->depth--;
    e->valueDepth--;
    e->fpLive   &= ~(1u << e->valueDepth);
    e->fpBorrow &= ~(1u << e->valueDepth);
    e->kPend    &= ~(1u << e->valueDepth);
    e->kKnown   &= ~(1u << e->valueDepth);
    e->xBorrow  &= ~(1u << e->valueDepth);
    if (kind != NULL) *kind = e->stack[e->depth];
    *reg = valueXReg(e, e->valueDepth);
    return true;
}

/* Pops, reporting the register the value is ACTUALLY in -- a borrowed entry reports the local's own
 * register, removing the copy, since every arm reads one register and writes `pushReg(e)-1`. No arm writes back into a popped register; the whitelist is what keeps that true. */
static bool popValue(Emit *e, unsigned *reg, SlotKind *kind) {
    if (e->depth == 0 || !holdsRegister(e->stack[e->depth - 1])) return false;
    fpSyncOne(e, e->valueDepth - 1);
    unsigned held = xHeldIn(e, e->valueDepth - 1);
    if (!popValueRaw(e, reg, kind)) return false;
    *reg = held;
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

/* STP's pre-index immediate is a signed 7-bit field scaled by 8: reaches -512 going in but only +504
 * coming out (imm/8=64 read back as a signed 7-bit field is -64), so an exactly-512-byte frame passed entry and silently truncated on exit -- `ldp x29,x30,[sp],#-512` moves SP a kilobyte the WRONG way, corrupting a caller's frame instead of crashing at the fault site. framePairFits()'s <=504 bound is what both ends agree on. */
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

/* v8..v15 are callee-saved only in their low 64 bits -- exactly a double -- so `str d`/`ldr d` is the
 * whole protocol; no FP STP in this encoder, but it only runs once per entry/exit, never in a loop. */
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
    /* A join must agree where every value is: nothing may cross a branch in an FP register or deferred.
     * Settling here stays safe after a compare, since neither fmov, mov, nor movz touches NZCV. */
    fpSyncAll(e);
    settleAll(e);
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

/* A guard failed: not a bail (unsound once the body has written anything, and the guards that matter
 * guard field reads inside loops that write) -- records where the interpreter resumes and what it holds; the stub is emitted after the body so the hot path keeps one not-taken branch. */
static bool jitDeoptStress(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_DEOPT_STRESS");
        cached = (v != NULL && v[0] != '\0' && strcmp(v, "0") != 0) ? 1 : 0;
    }
    return cached != 0;
}

/* Neither an inline's state nor mid-instruction state is the model's actual current state. Inside an
 * inline the interpreter hasn't made the call yet, so it resumes at OP_CALL holding just callee+args, not the inlined body's locals/temporaries. Outside one, a guard mid-instruction (OP_GET_LOCAL2 pushes then guards) leaves the model deeper than the interpreter's stack there -- handing over those extra entries strands them, and a loop can read its own iterator as its loop variable. `instDepth` is the model at the instruction's START, matching the interpreter; only entries THIS instruction pushed (the topmost) may be trimmed back to it -- an instruction that already popped something the interpreter still holds cannot be repaired and is refused. */
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
    /* OP_BUILD_RANGE emits nothing (deferred, folded into the following OP_GET_ITER), so a deopt record
     * taken at this offset would hand the interpreter two ints where it expects the range object -- it resumed, ran OP_GET_ITER, and reported 'int' object is not iterable. Fix: resume one instruction EARLIER, at the OP_BUILD_RANGE the model still describes (nothing between them has run). Reachable once a guard can fire inside the header itself, e.g. via root-filling a dynamic-local iterator descriptor; needs no spilling and no floats to reproduce. */
    if (e->pendingRange) {
        *ipOut = e->rangeBuildIp;
        *depthOut = e->instDepth;
        unsigned nseen = 0;
        for (unsigned i = 0; i < e->instDepth; i++) {
            if (holdsRegister(e->stack[i])) nseen++;
        }
        *valueDepthOut = nseen;
        return true;
    }
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
    /* Assertion, not the fix: a deopt stub writes every entry out of fpRegAt, so nothing may still be
     * borrowing a local's register here. Releasing HERE (rather than at the top of the instruction) was tried and is wrong -- a guard can sit inside a span an earlier branch skips (emitBoundsNormalise's does), so the fmov landed on a not-taken path and matrix_mul read `sum` from a register nothing had written. Declines rather than miscompiles if fpBorrowSurvives let something through it shouldn't have. */
    if (e->fpBorrow != 0) {
        e->whyNot = "a float borrow reached a guard";
        e->failed = true;
        return false;
    }
    /* Same reasoning as above, for a pending constant or X-register borrow: an assertion that the
     * whitelist held, declining rather than miscompiling if it didn't. */
    if (anyDeferred(e)) {
        e->whyNot = "a deferred value reached a guard";
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
    if (anyDeferred(e)) {            /* see deoptRecordAt */
        e->whyNot = "a deferred value reached a guard";
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
    /* JAITHON_JIT_DEOPT_STRESS makes every guard fail, so the whole test suite exercises the resume path
     * -- otherwise reached only when a program changes a field's type, which almost none do. */
    bool always = jitDeoptStress();
    e->fixups[e->fixupCount].conditional  = !always;
    e->fixups[e->fixupCount].depth        = -1;
    e->fixupCount++;
    emit(e, always ? jaiA64B(0) : jaiA64BCond(cond, 0));
}

/* Branch to the bail block on `cond`. The block's index is not known yet, so
 * it is patched with the rest. */
/* NaN comparison is a TypeError here, not false, matching the interpreter's isnan check: fcmp sets V
 * on an unordered result, and that's routed to a deopt so the interpreter raises exactly what it would have. */
static void nanToDeopt(Emit *e) { branchOnDeopt(e, JAI_A64_VS); }

static int instructionLength(const Chunk *c, int off);

/* One past the LAST back edge to `top`, or 0 if nothing branches back there.
 * Not findLoopEnd, which stops at the first: `continue` is a second back edge
 * to the same head, and stopping at it would call the rest of the body
 * "outside the loop" -- which is exactly the half a hoist must not believe. */
static uint32_t loopBodyEnd(const Chunk *c, uint32_t top) {
    uint32_t last = 0;
    for (int off = (int)top; off < c->count;) {
        int len = instructionLength(c, off);
        if (len <= 0) return 0;
        if (c->code[off] == OP_LOOP) {
            int16_t jump = jaiReadI16(c->code + off + 1);
            if ((uint32_t)((int32_t)(off + 3) + jump) == top) {
                last = (uint32_t)(off + 3);
            }
        }
        off += len;
    }
    return last;
}

/* items+count via one ldp: rCount comes back as `count | capacity << 32` (the two int32s share the
 * pair's second doubleword), so every reader must use the uxtw forms. Layout is asserted below, not assumed -- a field reordered in object.h would load a pointer as a count and index off the end of the array. */
static void emitListHeader(Emit *e, unsigned rList, unsigned rItems,
                           unsigned rCount) {
    _Static_assert(offsetof(ObjList, count) == offsetof(ObjList, items) + 8,
                   "ObjList.count must follow items for the ldp");
    _Static_assert(offsetof(ObjList, capacity) == offsetof(ObjList, count) + 4,
                   "ObjList.capacity must share the count's doubleword");
    emit(e, jaiA64LdpOff(rItems, rCount, rList,
                         (int32_t)offsetof(ObjList, items)));
}

/* The hoisted header for a subscript whose base is a plain read of local
 * `slot`, or -1. `curOffset` is checked against the loop the entry was made
 * for, so an entry the walk has already left cannot be picked up again by a
 * later loop that happens to name the same slot. */
static int hoistFor(const Emit *e, int slot) {
    if (slot < 0 || e->inlining) return -1;
    for (unsigned i = 0; i < e->hoistCount; i++) {
        if (e->hoist[i].slot != (uint8_t)slot) continue;
        if (e->curOffset < e->hoist[i].top) continue;
        if (e->curOffset >= e->hoist[i].end) continue;
        return (int)i;
    }
    return -1;
}

/* Nothing outside [top, end) may branch to `top`. The hoist is emitted just
 * ABOVE the loop head, so a path that arrives at the head from anywhere else
 * would run the body with registers nobody loaded. Back edges are fine by
 * construction -- they are inside the loop, where the slot cannot change. */
static bool onlyBackEdgesEnter(const Chunk *c, uint32_t top, uint32_t end) {
    for (int at = 0; at < c->count;) {
        int len = instructionLength(c, at);
        if (len <= 0) return false;
        int rel = jaiOpBranchOperandAt(c->code[at]);
        if (rel >= 0) {
            int16_t jump = jaiReadI16(c->code + at + 1 + rel);
            if ((int32_t)(at + len) + jump == (int32_t)top &&
                ((uint32_t)at < top || (uint32_t)at >= end)) {
                return false;
            }
        }
        at += len;
    }
    return true;
}

/* Loop-invariant list headers, hoisted above the loop head.
 *
 * Every `xs[i]` reloads `items` and `count` off the ObjList, and in a stencil
 * (five neighbours, four of them from rows the loop is not walking) that is
 * five loads an iteration of something no iteration changes. The reload is
 * also what makes the read SOUND without a version guard, so hoisting it needs
 * the guard back -- unless the body can be shown to contain nothing that could
 * move a list at all, which is exactly what `bodyCalls` answers. Every way a
 * list is resized (push, insert, remove, clear, slice, anything through a
 * descriptor) is a call out, and so is every collection; an in-place `xs[i] =
 * v` moves neither `items` nor `count`. So in a call-free body a header is
 * invariant for as long as the LOCAL is, and that is a fact the measuring pass
 * already recorded (noteSlotWrite).
 *
 * Registers come from x13..x17, which the tier otherwise never names, plus
 * whatever the operand stack left unused at the top of the scratch bank. Both
 * are caller-saved, which is the same reason they are free and the reason this
 * is confined to a body that calls nothing. */
/* Chooses the loop each candidate is hoisted out of and pays for the registers
 * busiest first, ONCE, before a line of the body is emitted. Doing it at the
 * loop heads as the walk reaches them spends the pool in program order, which
 * means the outermost loop -- the one whose header load happens least often --
 * takes the registers the innermost one wanted. Weight is the measuring pass's
 * own loop-depth-weighted count of subscript sites, the same currency the
 * local allocator ranks slots in.
 *
 * The loop chosen is the OUTERMOST one the slot is invariant across, so the
 * load runs as rarely as the proof allows. */
static void planHoists(Emit *e, ObjFunction *fn) {
    if (e->measuring || !e->osr || e->bodyCalls) return;
    const Chunk *c = &fn->chunk;

    struct { uint32_t top, end, use; uint8_t slot; } cand[JIT_MAX_SLOTS + 1];
    unsigned ncand = 0;

    for (unsigned s = 0; s < e->locals && s <= JIT_MAX_SLOTS; s++) {
        if (e->localKind[s] != SLOT_LIST) continue;
        if (e->slotXReg[s] == 0) continue;   /* no register to load from */
        if (e->slotIndexUse[s] == 0) continue;
        uint32_t bestTop = 0, bestEnd = 0;
        for (int at = (int)e->osrTop; at < (int)e->osrEnd;) {
            int len = instructionLength(c, at);
            if (len <= 0) break;
            uint32_t lt = (uint32_t)at;
            uint32_t le = loopBodyEnd(c, lt);
            at += len;
            if (le == 0 || le <= lt || le > e->osrEnd) continue;
            /* Every subscript of this slot inside the loop... */
            if (e->slotIndexLo[s] < lt || e->slotIndexHi[s] >= le) continue;
            /* ...and no write to it anywhere in the loop. */
            if (e->slotWriteHi[s] >= lt && e->slotWriteLo[s] < le) continue;
            if (!onlyBackEdgesEnter(c, lt, le)) continue;
            if (bestEnd == 0 || le - lt > bestEnd - bestTop) {
                bestTop = lt; bestEnd = le;
            }
        }
        if (bestEnd == 0) continue;
        cand[ncand].top  = bestTop;
        cand[ncand].end  = bestEnd;
        cand[ncand].use  = e->slotIndexUse[s];
        cand[ncand].slot = (uint8_t)s;
        ncand++;
    }

    while (e->hoistCount < JIT_MAX_HOIST &&
           e->hoistPoolCount - e->hoistTaken >= 2u) {
        unsigned pick = ncand, bestUse = 0;
        for (unsigned i = 0; i < ncand; i++) {
            if (cand[i].use > bestUse) { bestUse = cand[i].use; pick = i; }
        }
        if (pick == ncand) break;
        cand[pick].use = 0;                  /* taken */
        unsigned rI = e->hoistPool[e->hoistTaken++];
        unsigned rC = e->hoistPool[e->hoistTaken++];
        if (rI < e->scratchRoom) e->scratchRoom = rI;
        if (rC < e->scratchRoom) e->scratchRoom = rC;
        e->hoist[e->hoistCount].top      = cand[pick].top;
        e->hoist[e->hoistCount].end      = cand[pick].end;
        e->hoist[e->hoistCount].slot     = cand[pick].slot;
        e->hoist[e->hoistCount].itemsReg = (uint8_t)rI;
        e->hoist[e->hoistCount].countReg = (uint8_t)rC;
        e->hoistCount++;
    }
}

/* The loads themselves, emitted just above the head of the loop they were
 * planned out of -- which is where the walk is when it reaches that offset. */
static void emitHoistsAt(Emit *e, uint32_t off) {
    if (e->inlining) return;
    for (unsigned i = 0; i < e->hoistCount; i++) {
        if (e->hoist[i].top != off) continue;
        emitListHeader(e, e->slotXReg[e->hoist[i].slot],
                       e->hoist[i].itemsReg, e->hoist[i].countReg);
    }
}

/* Normalises the index and bounds-checks it in one unsigned compare: a negative index is a huge
 * unsigned value and fails the same test as one past the end. `countW` means rCount's low half only, as an ObjList header `ldp` leaves it (`count | capacity << 32`) -- every read goes through uxtw; an ObjString length (no capacity above it) passes false and uses the plain register form instead. */
static void emitBoundsNormalise(Emit *e, unsigned rIdx, unsigned rCount,
                                unsigned rOut, bool countW) {
    emit(e, jaiA64MovX(rOut, rIdx));
    emit(e, countW ? jaiA64SubsXUxtw(31, rOut, rCount)
                   : jaiA64SubsXReg(31, rOut, rCount));
    /* Skip length used to be hand-counted (four instructions) -- branchOnDeopt may itself emit an FP-
     * borrow release, and one extra instruction inside the span turned the skip into a jump onto the bail branch (same matrix_mul `sum`-from-nothing bug as deoptRecordAt). Measured with `e->count`, so it can't rot. */
    unsigned skip = e->count;
    emit(e, jaiA64BCond(JAI_A64_LO, 0));
    emit(e, countW ? jaiA64AddXUxtw(rOut, rOut, rCount)
                   : jaiA64AddX(rOut, rOut, rCount));
    emit(e, countW ? jaiA64SubsXUxtw(31, rOut, rCount)
                   : jaiA64SubsXReg(31, rOut, rCount));
    branchOnDeopt(e, JAI_A64_HS);
    if (skip < e->count && e->count <= JIT_MAX_INSTS) {
        e->code[skip] = jaiA64BCond(JAI_A64_LO, (int32_t)(e->count - skip));
    }
}

/* Neither of the two blocks below reads an operand-stack entry -- a bail says
 * "could not continue" and the caller throws the whole computation away, and an
 * overflow raises -- so the entries do NOT have to be in their X homes to
 * branch to one. The fpSyncAll both used to do was pure hot-path cost: in a
 * float expression carrying an int guard through it (a stencil's `mid[j-1]`),
 * each one wrote every live float out and the next use read it back, four
 * cross-bank fmovs sitting in the middle of the accumulate chain. A join
 * (branchTo) and a deopt record are different and still settle -- the first
 * because the other edge must agree, the second because the record is read. */
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

/* `cond` isn't always VS: adds/subs set the overflow flag, but the multiply test compares the
 * product's high half against the low half's replicated sign, so its answer is NE -- routing multiply through VS meant its overflow was never detected (4 * 2^62 silently came back as 0). */
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

/* Resolved the way the interpreter resolves a builtin: the module first, `vm.builtins` only when the
 * module has no such name -- a module-level binding is never mistaken for it, and if one appears later the module's version retires this compiled form. */
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

/* A JaiEntry's address is stable as long as the table doesn't rehash/delete/clear -- overwriting an
 * existing global never moves it (ensureRoom only runs for a NEW key) -- which is what lets a compiled load be one `ldr` from a baked pointer. `keyVersion` counts every event that breaks this; emitGlobalsGuard checks it. */
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

/* Emitted before EVERY access, not hoisted: hoisting is sound only given a control-flow claim (no
 * call-out between a guard and a later access on a back edge) -- exactly the kind of reasoning this file has been bitten by before. Costs four instructions on a predictable branch. */
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

/* `m->version` retires every cache that resolved a NAME to a heap object or presence (interpreter's
 * global inline cache, OP_FORMAT's builtin-str check, this tier's baked classes/closures/natives) -- all of which required the bound value to BE a heap object. A store replacing a non-object with a non-object changes none of them and may skip the bump; getting this wrong made `total = total + 1` at module scope retire every compiled function in the module. */
static void emitVersionBump(Emit *e, ObjModule *m) {
    emitConst64(e, JIT_SCRATCH_A, (int64_t)(uintptr_t)&m->version);
    emit(e, jaiA64LdrW(JIT_SCRATCH_B, JIT_SCRATCH_A, 0));
    emit(e, jaiA64AddXImm(JIT_SCRATCH_B, JIT_SCRATCH_B, 1));
    emit(e, jaiA64StrW(JIT_SCRATCH_B, JIT_SCRATCH_A, 0));
}

/* Predicted stack kind + guard tag for a recorded OP_INVOKE result; false when the tier has no use
 * for the byte. SLOT_INST is deliberately excluded: it carries a class shape this byte can't encode, and admitting it would silently lose the shape every field offset was resolved against. Class/closure/native excluded too -- they have register-free stack kinds of their own. */
static bool feedbackSlotKind(uint8_t fb, SlotKind *k, unsigned *tag) {
    switch (fb) {
    case 1u + VAL_INT:   *k = SLOT_INT;   *tag = VAL_INT;   return true;
    case 1u + VAL_FLOAT: *k = SLOT_FLOAT; *tag = VAL_FLOAT; return true;
    case 1u + VAL_BOOL:  *k = SLOT_BOOL;  *tag = VAL_BOOL;  return true;
    default: break;
    }
    if (fb < JAI_FB_OBJ || fb >= JAI_FB_OBJ + (unsigned)OBJ_TYPE_COUNT) {
        return false;
    }
    switch ((ObjType)(fb - JAI_FB_OBJ)) {
    case OBJ_INSTANCE: case OBJ_CLASS: case OBJ_TRAIT:
    case OBJ_CLOSURE:  case OBJ_FUNCTION: case OBJ_NATIVE:
    case OBJ_BOUND:    case OBJ_ENUM: case OBJ_ENUM_CTOR:
        return false;
    default: break;
    }
    /* Anything else on the heap can be loaded, passed and stored and nothing
     * else, which is exactly SLOT_OBJ. The tag guard below is the whole of
     * what makes that sound: whatever object comes back, it is an object. */
    *k = SLOT_OBJ; *tag = VAL_OBJ;
    return true;
}

/* The same test feedbackSlotKind applies to a call's result, as a predicate on a Value: is this a heap
 * object with no stack kind of its own, so SLOT_OBJ's "read, pass, store, root and nothing else" describes it exactly? Excludes the kinds a later arm resolves to a register-free entry (class/closure/native/...), which SLOT_OBJ would silently outrank. */
static bool rawObjValue(Value v) {
    if (!IS_OBJ(v) || AS_OBJ(v) == NULL) return false;
    switch (OBJ_TYPE(v)) {
    case OBJ_INSTANCE: case OBJ_CLASS: case OBJ_TRAIT:
    case OBJ_CLOSURE:  case OBJ_FUNCTION: case OBJ_NATIVE:
    case OBJ_BOUND:    case OBJ_ENUM: case OBJ_ENUM_CTOR:
        return false;
    default: break;
    }
    return true;
}

/* Predicted stack kind for a callee that has not compiled, from what it has been observed to return
 * (ObjFunction::obsReturnKind). Same contract as feedbackSlotKind's: a prediction the caller must guard.
 * SLOT_INST is admissible here where it is not there, because a per-callee record can carry the class
 * shape a one-byte-per-way cache cannot -- and the shape is guarded after the call like the tag. */
static bool observedReturnKind(const ObjFunction *cfn, SlotKind *k,
                               uint32_t *shape) {
    uint8_t fb = cfn->obsReturnKind;
    *shape = 0;
    if (fb == 1u + (unsigned)VAL_NULL) { *k = SLOT_NULL; return true; }
    if (fb == JAI_FB_OBJ + (unsigned)OBJ_INSTANCE) {
        if (cfn->obsReturnShape == 0) return false;
        *k = SLOT_INST;
        *shape = cfn->obsReturnShape;
        return true;
    }
    unsigned tag;
    return feedbackSlotKind(fb, k, &tag);
}

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
    /* Anything else on the heap (dict, string, closure) can be loaded, passed and stored, nothing else.
 * A class, function or native never reach here: OP_GET_GLOBAL resolves those to their own register-free stack kinds first. */
    if (IS_OBJ(v) && AS_OBJ(v) != NULL && !IS_CLASS(v) && !IS_CLOSURE(v) &&
        !IS_NATIVE(v)) {
        *k = SLOT_OBJ; return true;
    }
    return false;
}

/* A callee that is not this function: only a class, whose result is an
 * instance of a shape known here. Anything else would need a guard on a return
 * value nothing can predict. */
/* Recognises an initializer that does nothing but store its arguments into fields in order
 * (`GET_LOCAL2 0 k; SET_FIELD f` repeated, then RETURN_NULL) -- anything else (a default, a computed field, a call, a branch) goes the long way. Lets `Point(a, b)` become an allocation and two stores instead of a descriptor + jaiCallValue + invokeCallable's type switch + a compiled init. */
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
        if (jaiReadU16(c + off + 1) != 0) return false;
        if (jaiReadU16(c + off + 3) != i + 1) return false;
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

/* ownStatus: caller decodes the helper's return itself, skipping the built-in "nonzero means raised"
 * test. Written for the iterator step (0 yielded, 1 exhausted, 2 raised), whose call-out the list arm of
 * OP_FOR_ITER_BIND no longer makes -- the default test sent `exhausted` to the throw stub, which found no pending exception and died on "internal error: failed operation raised nothing". Kept because any helper with a three-way answer needs it, and because the lesson is not rediscoverable from the code. */
/* Root-fills the descriptor: shared by the descriptor path (a C helper pushes them) and the self-call
 * path (the emitted code links the descriptor onto the collector's frame chain instead, since a bare `bl` pushes nothing). */
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

    /* Locals alone weren't enough once OP_GET_ITER started leaving an ObjIter live in a register across
 * a call that might collect -- only visible under --gc-stress, and only once such a loop could compile at all. */
    /* Counts register-holding entries from the bottom, not by assuming they're the top `valueDepth` --
 * a no-register entry (class/function/builtin/self) can sit in the middle of the stack, e.g. `join(f(a), f(b))` pushes a callee before its arguments. Subtracting valueDepth would name the wrong register above it and skip entries that still need rooting; the deopt stub has always counted this way. */
    unsigned seen = 0;
    for (unsigned idx = 0; idx < e->depth; idx++) {
        SlotKind k = e->stack[idx];
        if (!holdsRegister(k)) continue;
        unsigned reg = valueBankBase(e) + seen;
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

    /* Callee as a whole Value, from a register when only known at run time (a closure held in a local):
     * baking the compile-time-live closure would freeze its upvalues -- `closure_calls` builds a fresh closure over a different `step` every outer iteration. */
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
        unsigned reg = valueBankBase(e) +
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
    noteScratchClobber(e);
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

/* Verdict 4: the callee deoptimised part-way and may have written, so the call can't be re-executed
 * or recorded over (gDeopt is a single global) -- instead the callee is FINISHED in the interpreter from its own record, and the value handed back here, consuming the record at the innermost frame that sees it. Makes a recursive body that writes compilable, and (since `callee` may name someone else) a direct call to a writing method too. OSR form: fall-through continues the loop, so nothing syncs here -- every branch out goes to a stub that syncs itself. */
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

        /* VAL_OBJ is every heap object -- reading `klass` off the wrong type (e.g. an ObjString) doesn't
         * fault, it answers wrongly, so the type is checked before the shape. */
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

/* Cold half of `xs.push(v)`: reserve, refill the count the fast path already loaded, branch back in.
 * No descriptor/roots (see jitListGrow). This is a continuation, not an exit -- an OSR form must NOT sync its iterator or locals here, since the loop carries on with them where they are. */
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

/* Direct-branch arguments arrive as a raw payload, so the caller's kind must match what the callee
 * was specialised for -- and for an instance, the same class shape, since every field offset was resolved against it. jaiJitEnterFunc is the only check standing between a float's bits and a body that treats them as a pointer. `firstIdx`: the entry above the callee for a plain call, or the receiver (callee's slot 0) for a method. */
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

/* Branches straight to a compiled callee's entry, skipping the descriptor/jaiCallValue/interpreter-
 * frame path. Convention: raw payloads in x0.., closure in the last arg register if the callee reads an upvalue, x0/x1 = value/verdict on return -- the same one jaiJitEnterFunc checks and a self-call already uses, so skipping that entry means answering its checks here instead: module version (why the callee must live in the caller's module), every parameter's kind+shape (by the caller's model), and the verdict (below). Nonzero verdict: a callee that writes NOTHING can have the whole call abandoned and re-executed from the pre-call stack (two compares, no stub); a callee that WRITES cannot be re-run -- verdict 4 means it deoptimised part-way and is FINISHED in the interpreter from its own record, sharing the `selfSlow` machinery a recursive self-call already uses. A raised exception goes to the throw exit instead, since its effects already happened. `calleeReg`: the ObjClosure register, or -1 if baked in. `cidx`: operand-stack index of the callee entry (the RECEIVER for a method, i.e. its slot 0). `after`: offset of the fall-through instruction. */
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
    /* The callee's baked classes/closures/natives are pinned by ITS jitFuncModuleVersion, checked at the
     * entry this call skips -- valid only if the caller's own check agrees NOW: a global rebound after the callee compiled would leave jitFuncModuleVersion stale but still reachable via a direct call, and a caller compiling afterwards would silently pin the newer version. */
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
    /* SLOT_NULL: a `-> void` function. Epilogue leaves x0 zero, so the pushed entry has a fixed tag and
     * zero payload -- same treatment a self-call to a void function gets. Refusing it declined every caller of a procedure, which in nbody is the whole of `main`. */
    SlotKind rk = (SlotKind)cfn->jitReturnKind;
    ObjClass *rcls = NULL;
    if (rk != SLOT_INT && rk != SLOT_FLOAT && rk != SLOT_BOOL &&
        rk != SLOT_INST && rk != SLOT_LIST && rk != SLOT_OBJ &&
        rk != SLOT_NULL) {
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
    noteScratchClobber(e);
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
        e->depth--;
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
    /* Deliberately NOT wroteHeap for a non-writing callee, unlike the descriptor path: it stores and
     * calls nothing, so a re-run repeats no visible effect. It may still allocate (OP_GET_SLICE does) -- a fresh object isn't an observable effect. */
    return true;
}

static bool compileBody(Emit *e, ObjClosure *closure);
static int instructionLength(const Chunk *c, int off);

/* ---- literal operands -------------------------------------------------- */

/* Whether anything but fall-through can reach `off`: reading the previous instruction's literal
 * without this check is a miscompile, not a decline -- `x // (if c {2} else {4})` puts OP_INT right before OP_FLOORDIV *and* a jump from the other arm onto it. Scans the WHOLE chunk, not just fixups already emitted (a back edge compiles after its target is walked, so the fixup list would miss loop tops) -- and handler/finally addresses too, since the unwinder can resume there with a stack this walk never saw. */
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

/* OP_INT carries its value inline, OP_CONST names a pool entry -- the only two ways a literal reaches
 * the stack. The adjacency check is belt-and-braces (the walk is linear) but a real bug: OP_FORMAT once advanced `off` by nine instead of ten, and this is what would have caught it. */
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
/* Collapses the general "add divisor back if remainder is non-zero and signs differ" (7 instructions)
 * to 2 when the divisor's sign is known at compile time: msub leaves |r| < |d| with r's sign following the dividend's, so for a positive divisor the whole test is "r < 0" (one bit), and for a negative one "r > 0". `r + d` cannot overflow since |r| < |d| puts the sum strictly between -|d| and |d|. */
static void emitFloorFixup(Emit *e, unsigned rrem, unsigned rd,
                           bool signKnown, int64_t divisor, uint32_t fixup) {
    if (signKnown && divisor > 0) {
        emit(e, jaiA64Tbz(rrem, 63u, 2));
        emit(e, fixup);
        return;
    }
    if (signKnown) {
        emit(e, jaiA64SubsXImm(31, rrem, 0));
        emit(e, jaiA64BCond(JAI_A64_LE, 2));
        emit(e, fixup);
        return;
    }
    emit(e, jaiA64SubsXImm(31, rrem, 0));
    emit(e, jaiA64BCond(JAI_A64_EQ, 5));
    emit(e, jaiA64EorX(JIT_SCRATCH_D, rrem, rd));
    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_D, 0));
    emit(e, jaiA64BCond(JAI_A64_GE, 2));
    emit(e, fixup);
}

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

/* Structural check, answered before anything is emitted (a half-inlined body can't be taken back):
 * no branches (no offset map, no join, no fixup naming a callee offset in the caller's table); exactly one RETURN, last; locals only via the four opcodes the inline frame understands, and only slots this callee actually has; globals only for the two builtins the tier emits inline (else a global VALUE load would bake a JaiEntry from the callee's own table, needing its own guard); nothing that stores (a guard inside re-executes the WHOLE call, so an earlier store would run twice). What's left is straight-line register arithmetic -- the main walker already speaks it, so no second emitter is needed. `evalA` in spectral is fifteen instructions of exactly this shape. */
static bool inlinableBody(ObjClosure *callee, unsigned argc,
                          unsigned *maxSlotOut, bool *readsUpvalueOut) {
    ObjFunction *cfn = callee->fn;
    const Chunk *c = &cfn->chunk;
    if (cfn->arity != argc || cfn->defaultCount != 0) return false;
    if (cfn->flags & (FN_VARIADIC | FN_KWREST | FN_INIT)) return false;
    if (c->count <= 0 || c->count > 128) return false;

    unsigned maxSlot = argc;
    bool sawReturn = false;
    bool readsUpvalue = false;
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
        /* An upvalue is reached through the closure that is actually being
         * called, which is a register the call site has to supply -- so this
         * is only inlinable where that register exists. OP_SET_UPVALUE is not
         * here and falls to `default`: a store would have to be undone if a
         * later guard in the same body deoptimised to the call. */
        case OP_GET_UPVALUE:
            if ((unsigned)c->code[off + 1] >= (unsigned)cfn->upvalueCount) {
                return false;
            }
            readsUpvalue = true;
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
    *readsUpvalueOut = readsUpvalue;
    return true;
}

/* Inlines the callee's body: slot 1+i IS entry cidx+1+i already on the stack, so nothing is copied in;
 * a bound slot pins one more entry underneath what's pushed after it, sound only because the body is straight-line. Callee's module must be the caller's, and its baked builtins are retired by the CALLER's own module-version check (the callee's is never run). `calleeReg`: needed only if the body reads an upvalue, since `callee` is a SAMPLE closure at an indirect site -- one ObjFunction, many closures (`|x| x + step`), so its captured cells aren't necessarily the next call's. Constants/globals are safe from the sample since they belong to the function/module, not the closure. */
static bool inlineGlobalCall(Emit *e, ObjFunction *caller, ObjClosure *callee,
                             unsigned argc, uint32_t callOff, int calleeReg) {
    if (e->noInline) return false;
    ObjFunction *cfn = callee->fn;
    if (cfn->module != caller->module) return false;
    if (e->inlining) return false;             /* one level, no recursion */
    unsigned cidx = e->depth - argc - 1;
    unsigned maxSlot = 0;
    bool readsUpvalue = false;
    if (!inlinableBody(callee, argc, &maxSlot, &readsUpvalue)) return false;
    if (readsUpvalue && calleeReg < 0) return false;
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

    /* No noteScratchClobber here. An inlined body cannot call -- inlinableBody
     * admits nothing that does -- so it destroys x0..x8 only by USING them,
     * which is not a clobber but an allocation: when the caller already owns
     * that bank the two share one numbering (see inlineOwnBank), and when it
     * does not, the inlined entries have x0..x8 to themselves as before.
     * Anything inside that really does call still reaches noteScratchClobber
     * on its own, and under scratchValues that declines the compile. */
    e->inlining     = true;
    e->inlDepth     = cidx + 1u + argc;
    e->inlPinned    = 0;
    e->inlValueBase = e->valueDepth;
    e->inlIp        = callOff;
    e->inlClosureReg = calleeReg;

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

    /* Straight to the callee's entry when everything jaiJitEnterFunc would
     * have checked can be checked here instead. Falling back rather than
     * declining matters: the descriptor path speaks a much wider language --
     * any argument kind, any module, a callee that writes -- and a call
     * through jaiCallValue still beats no compiled loop at all. */
    if (cfn->jitFunc != NULL) {
        const char *saved = e->whyNot;
        if (emitDirectCall(e, caller, cfn, cv, -1, cidx, argc, callOff,
                           after, false)) {
            return true;
        }
        if (e->failed) return false;   /* it had started emitting */
        e->whyNot = saved;
    }

    /* A callee with no compiled form of its own still knows what it has been
     * returning; see ObjFunction::obsReturnKind and the twin case at OP_INVOKE.
     * A recursive function is the ordinary way to reach this -- the loop being
     * compiled is inside the very function the call names, so there is nothing
     * for `jitReturnKind` to have been written by yet. */
    SlotKind rk = SLOT_NULL;
    uint32_t rshape = 0;
    ObjClass *rcls = NULL;
    bool haveKind;
    if (cfn->jitFunc != NULL) {
        rk = (SlotKind)cfn->jitReturnKind;
        rshape = cfn->jitReturnShape;
        haveKind = true;
    } else {
        haveKind = observedReturnKind(cfn, &rk, &rshape);
    }
    if (haveKind && rk == SLOT_INST &&
        (rshape == 0 || !jaiClassForShape(rshape, &rcls) || rcls == NULL)) {
        e->whyNot = "callee's return class not on record";
        return false;
    }
    if (!haveKind ||
        (rk != SLOT_INT && rk != SLOT_FLOAT && rk != SLOT_BOOL &&
         rk != SLOT_INST && rk != SLOT_LIST && rk != SLOT_OBJ &&
         rk != SLOT_NULL)) {
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
                     : rk == SLOT_NULL  ? VAL_NULL
                                        : VAL_OBJ;
    emit(e, jaiA64LdrW(JIT_SCRATCH_A, 31, rat));
    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, wantTag));
    branchOnDeoptAt(e, JAI_A64_NE, after, true);
    /* A null carries no payload worth loading, but the register still stands
     * for the entry and a deopt materialises it, so it gets a defined zero
     * rather than whatever the descriptor happened to leave behind. */
    if (rk == SLOT_NULL) emit(e, jaiA64MovzX(pushReg(e) - 1, 0, 0));
    else emit(e, jaiA64LdrX(pushReg(e) - 1, 31, rat + 8));
    if (rk == SLOT_INST) {
        /* The tag says "an object", which is not "an instance of this class",
         * and every field offset resolved against the entry below assumes it
         * is. The object type is checked before `klass` is read for the same
         * reason it is at the invoke arm: VAL_OBJ covers every heap object and
         * a returned string's header is shorter than an instance's. */
        emit(e, jaiA64LdrW(JIT_SCRATCH_A, pushReg(e) - 1,
                           (unsigned)offsetof(Obj, type)));
        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_INSTANCE));
        branchOnDeoptAt(e, JAI_A64_NE, after, true);
        emit(e, jaiA64LdrX(JIT_SCRATCH_A, pushReg(e) - 1,
                           (unsigned)offsetof(ObjInstance, klass)));
        emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_A,
                           (unsigned)offsetof(ObjClass, shapeId)));
        emitConst64(e, JIT_SCRATCH_B, (int64_t)rshape);
        emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
        branchOnDeoptAt(e, JAI_A64_NE, after, true);
    }
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

    unsigned base = valueBankBase(e);
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
            while ((int)e->depth > depth0) {
                unsigned r; if (!popValue(e, &r, NULL)) return false;
            }
        }
    }

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
            regs[i] = valueBankBase(e) +
                      (first + i - (e->depth - e->valueDepth));
        }
        /* Fast path (jitInstanceAlloc) is a leaf that cannot collect -- it declines whenever jaiGCWanted(),
     * exactly when jaiInstanceNew would have collected -- so needs no descriptor/roots, versus the descriptor's dozen stores plus a root push/pop just to allocate 64 bytes. NULL means it did nothing, so falling into the descriptor path is always correct; both paths land at the load below with the instance in SCRATCH_C. Skipped for a class the small-object bins can't serve. */
        const size_t instBytes =
            sizeof(ObjInstance) + sizeof(Value) * (size_t)cls->fieldCount;
        unsigned skipSlow = 0;
        bool haveFast = jaiSmallServes(instBytes);
        if (haveFast) {
            emitConst64(e, 0, (int64_t)(uintptr_t)cls);
            emitConst64(e, JIT_SCRATCH_A,
                        (int64_t)(uintptr_t)&jitInstanceAlloc);
            noteScratchClobber(e);
            emit(e, jaiA64Blr(JIT_SCRATCH_A));
            emit(e, jaiA64MovX(JIT_SCRATCH_C, 0));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_C, 0));
            skipSlow = e->count;
            emit(e, jaiA64BCond(JAI_A64_NE, 0));   /* patched below */
        }
        if (!emitDescriptor(e, OBJ_VAL((Obj *)cls), first, 0,
                            (void *)&jitNewInstance)) {
            return false;
        }
        emit(e, jaiA64LdrX(JIT_SCRATCH_C, 31,
                           e->descOffset +
                               (unsigned)offsetof(JitCallDesc, result) + 8));
        /* The span is measured rather than counted: emitDescriptor's length
         * moves with the number of roots this body holds and with how many
         * halfwords the class pointer needs. */
        if (haveFast && skipSlow < e->count && e->count <= JIT_MAX_INSTS) {
            e->code[skipSlow] =
                jaiA64BCond(JAI_A64_NE, (int32_t)(e->count - skipSlow));
        }
        for (unsigned i = 0; i < argc; i++) {
            unsigned r;
            if (!popValue(e, &r, NULL)) return false;
        }
        if (e->depth == 0 || e->stack[e->depth - 1] != SLOT_CLASS) return false;
        e->depth--;
        if (!pushValue(e, SLOT_INST, cls->shapeId, cls)) return false;
        /* Instance held in a scratch (SCRATCH_C) until every field is stored, since the result reuses the
     * first argument's register: loading it into its final register first overwrote the argument about to be stored into it -- alloc_churn came back in 5ms with a wrong answer. */
        unsigned rinst = pushReg(e) - 1;
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

/* After an unconditional OP_LOOP/OP_JUMP there's no fall-through, so the linear walk can't carry the
 * preceding instruction's stack across the gap -- reconciles from a branch that targets this offset instead, accepting only a pure truncation (register entries are the top `valueDepth`, so popping from the top preserves every index below the join). */
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

/* `build` in binary_trees returns `null` on one path and `Node(..)` on another -- an instance merged
 * with a maybe-instance becomes a maybe-instance (shape survives only if both sides agree). Written once before and reverted when it made binary_trees 11x slower -- not this merge's fault: at the time a self-call rooted nothing and emitRootFill's operand-stack-to-register mapping was wrong, and `build` was the first body to hold a fresh allocation across an allocating self-call. Both bugs are now fixed. */
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

/* Opcodes that pull a float operand straight out of the FP bank; every other opcode sees the model
 * materialised as before. Adding an opcode here without also teaching it fpOperand is a MISCOMPILE, not a decline -- the whole risk of this design, and why the list stays short. */
static bool fpFastOp(uint8_t op) {
    switch (op) {
    case OP_ADD: case OP_SUB: case OP_MUL: case OP_DIV:
    case OP_ADD_BIND: case OP_SUB_BIND: case OP_MUL_BIND:
    case OP_GET_LOCAL: case OP_GET_LOCAL2:
    case OP_ADD_LOCALS: case OP_BIND: case OP_SET_LOCAL:
    /* These three only ever write the entry they push; a `float`-boundary type guard on something
     * already float emits nothing -- it sat between the ADD and the BIND in mandelbrot's `x = x2 - y2 + x0` and made the whole expression materialise for nothing. */
    case OP_CONST: case OP_INT: case OP_TYPE_GUARD:
    case OP_LT: case OP_LE: case OP_GT: case OP_GE:
    case OP_EQ: case OP_NE:
    case OP_JUMP_IF_CMP_FALSE:
    /* Reading an element writes only the entry it pushes plus the four scratch X registers -- no v
     * register, no call. Until this was listed, an FP-resident accumulator materialised the moment a subscript appeared, i.e. every array-summing loop (`sum += ai[k] * b[k][j]`, all of matrix_mul). */
    case OP_GET_INDEX:
    /* Same argument as reading an element; nbody wanted it: `bi.vx -= dx * bj.mass * mag` reads three
     * fields between the local it multiplies and the multiply, so without this every field-expression operand materialised into X and came back through an fmov. Neither opcode writes a v register or reads an X register whose FP copy could be the live one. */
    case OP_GET_FIELD: case OP_GET_FIELD_LOCAL:
    /* Both take their operand out of the bank (see fpConsumer), or the dispatch loop syncs it back out
     * first and the arm never sees a live entry. `**0.5` (square root) sat between every `d2` and the divide that used it. */
    case OP_SET_FIELD: case OP_POW:
        return true;
    default:
        return false;
    }
}

/* Used to be the whole of `fpWorthLoading`, tested against just the very next opcode -- true only
 * while OP_GET_INDEX still materialised everything; fpWorthLoading below now walks to the consumer instead. */
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

/* Counted rather than subtracted from the top: a no-register entry (class/function/builtin/self) can
 * sit in the middle of the stack -- exactly the shape an inlined call site has. */
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
    unsigned src = xHeldIn(e, vi);
    if (!pushValue3(e, e->stack[idx], e->stackShape[idx], e->stackClass[idx],
                    e->stackSeen[idx], -1)) {
        return false;
    }
    unsigned dst = pushReg(e) - 1;
    if (dst != src) emit(e, jaiA64MovX(dst, src));
    return true;
}

/* Answered against the inlined body's own frame -- the CALLER's operand stack: a parameter is the
 * argument entry already sitting there, a bind pins whatever's on top. Reading these through the main switch would read the CALLER's local of the same number, a different variable entirely. */
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
        /* Sound only when the bound value is the ONLY thing above the already-pinned region (`let x = expr`
     * as a statement) -- `let a, b = ..` would leave this entry the top for both binds, aliasing two slots to one register. Caught by measuring depth, not by trusting the shape. */
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

/* A *consumer* must be reachable, not merely the next instruction: the original rule looked only at
 * the very next opcode, which said no for `sum` in matrix_mul (followed by OP_GET_LOCAL2 then all of `ai[k] * b[k][j]` before any float operator) and cost the loop its accumulator to a cross-register-file bounce every iteration. Walks forward instead through fpFastOp's opcodes and answers on the first one that isn't; bounded since none of them is variable length. */
/* Deliberately a short whitelist: every one of these reads float entries through fpOperand/fpHeldIn
 * and none records a deopt while one is live (an int overflow inside OP_ADD goes through fpSyncAll first). Anything else releases at the top of the instruction; deoptRecordAt declines if one gets through anyway, so a wrong entry here costs a decline, not a miscompile. */
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
    /* A settled type guard emits/records nothing, but sits on this list because it stands between an
     * unfused float op and its OP_BIND (`y = 2.0*xy+y0` is ADD,TYPE_GUARD,BIND) -- releasing the borrow at the guard would reintroduce the exact `fmov` the lookahead removed. */
    case OP_TYPE_GUARD:
        return true;
    default:
        return false;
    }
}

/* Same shape as fpBorrowSurvives, just as deliberately short: each reads its operands via popValue/
 * xHeldIn, computes into `pushReg(e)-1`, and takes no deopt record with a deferred entry live. OP_INT/OP_CONST earn their place standing between `GET_LOCAL n` and the `SUB` consuming it. */
static bool deferSurvives(uint8_t op) {
    switch (op) {
    case OP_GET_LOCAL: case OP_GET_LOCAL2:
    case OP_INT: case OP_CONST:
    case OP_ADD: case OP_SUB: case OP_MUL: case OP_DIV:
    case OP_SHL: case OP_SHR:
    case OP_LT: case OP_LE: case OP_GT: case OP_GE:
    case OP_EQ: case OP_NE:
    case OP_JUMP_IF_CMP_FALSE:
    case OP_BIND: case OP_SET_LOCAL:
    case OP_POP:
    case OP_RETURN:
        return true;
    default:
        return false;
    }
}

/* The opcodes that can spell an integer literal as an immediate, which is the
 * whole reason OP_INT is ever allowed to push nothing. Kept in step with the
 * arms below by hand, and harmless if it says yes too often: the consumer
 * settles what it cannot fold. */
static bool foldsIntLiteral(uint8_t op) {
    switch (op) {
    case OP_ADD: case OP_SUB:
    case OP_LT: case OP_LE: case OP_GT: case OP_GE:
    case OP_EQ: case OP_NE:
    case OP_JUMP_IF_CMP_FALSE:
    /* Not because a shift takes an imm12 (it doesn't) -- the shift count becomes part of the instruction
     * encoding when it's a literal 0..63, so the register that used to hold it was written and never read. bitops paid a `movz` for every one of its shifts before this. */
    case OP_SHL: case OP_SHR:
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
        uint8_t op = code[off];
        /* An ASCII-table proof is only good along the fall-through edge this
         * walk is following. offsetIsBranchTarget scans the whole chunk, so it
         * catches a back edge whose branch has not been emitted yet -- and it
         * is only asked while a proof is actually live, which is the two or
         * three instructions between `s[i]` and whatever consumes it. */
        if (anyAsciiProof(e) &&
            offsetIsBranchTarget(&fn->chunk, (uint32_t)off)) {
            clearAsciiProofs(e);
        }
        /* Settles any deferred entry BEFORE the offset map records the instruction start, so a branch landing
 * here (arriving with everything in its own register) skips the settle and only the fall-through pays -- both paths then agree, which a join requires. Forward branches are known here; backward ones checked at the end. */
        if (anyDeferred(e)) {
            bool joinsHere = false;
            if (!e->inlining) {
                for (unsigned f = 0; f < e->fixupCount && !joinsHere; f++) {
                    joinsHere = (e->fixups[f].targetOffset == (uint32_t)off);
                }
            }
            if (joinsHere || !deferSurvives(op) || e->deferCarryCount >= 64) {
                settleAll(e);
            } else {
                e->deferCarry[e->deferCarryCount++] = (uint32_t)off;
            }
        }
        /* Above the offset map on purpose: a back edge to `off` must land on
         * the loop head, not on the loads that were hoisted out of it. */
        emitHoistsAt(e, (uint32_t)off);
        e->offsetToInst[off]  = (int)e->count;
        e->offsetToDepth[off] = (int)stackSignature(e);
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

        /* The four local opcodes an inlined body is allowed, answered against its own frame, before the main
         * switch reads them as the caller's slot numbers. */
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
            settleAll(e);   /* the caller reads the result out of valueXReg */
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
                if (e->slotFpReg[slot] != 0) {
                    fpBorrowLocal(e, idx, e->slotFpReg[slot]);
                } else {
                    localInFp(e, slot, fpRegAt(e, idx));
                    fpClaim(e, idx);
                }
            } else {
                unsigned home = localHomeX(e, slot);
                if (home != 0) {
                    /* The copy this used to always emit is the whole cost of reading a local (six of them in `fib`) --
                     * borrowing defers it; if nothing consumes the value before an instruction that can't read a borrow, the settle there emits exactly the same mov, so this never costs more. */
                    xBorrowLocal(e, e->valueDepth - 1, home);
                } else {
                    unsigned dst = pushReg(e) - 1;
                    unsigned src = localIn(e, slot, dst);
                    if (src != dst) emit(e, jaiA64MovX(dst, src));
                }
            }
            off += 3;
            break;
        }

        case OP_INT: {
            int16_t k = jaiReadI16(code + off + 1);
            if (!pushValue(e, SLOT_INT, 0, NULL)) return false;
            e->kKnown |= 1u << (e->valueDepth - 1);
            e->kKnownVal[e->valueDepth - 1] = k;
            /* Push nothing when the next instruction can say the literal as an
             * immediate: `n - 1` is one `subs`, not a `movz` and a `subs`. The
             * consumer settles it if the rest of its shape turns out not to
             * allow the fold, so being wrong here costs the instruction it
             * would have spent anyway. */
            if (off + 3 < stop && foldsIntLiteral(code[off + 3]) &&
                k >= -4095 && k <= 4095) {
                kPendLocal(e, e->valueDepth - 1, k);
            } else {
                emitConst64(e, pushReg(e) - 1, k);
            }
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
                localOut(e, slot, xHeldIn(e, e->valueDepth - 1));
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
                if (e->slotFpReg[a] != 0) {
                    da = e->slotFpReg[a];
                } else {
                    localInFp(e, a, fpRegAt(e, idx));
                    da = fpRegAt(e, idx);
                }
                if (e->slotFpReg[b] != 0) {
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
                unsigned dd = fpBindDest(e, slot, fpRegAt(e, ia));
                emit(e, jaiA64FaddD(dd, da, db));
                localOutFp(e, slot, dd);
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

        /* `slots[S] < imm`, pushing the bool.
         *
         * This arm was missing until 2026-08-12, so any function containing the
         * opcode declined WHOLE -- which is the failure the peephole's own
         * history records: OP_CMP_LOCAL_CONST_LT shipped without a JIT arm once
         * before and cost 14 distinct declines. Nothing in the benchmark suite
         * happened to hit it this time, which is exactly why it went unnoticed;
         * `make jit-fusion-check` now refuses a build where a fused opcode has
         * no arm here. */
        /* Stamp the declared element kind onto the container on top.
         *
         * Needs an arm rather than a decline: it sits right after a container
         * literal, so `var out: list[int] = []` inside a hot function put it in
         * the middle of one. Without this, sort_merge's `merge` -- whose whole
         * body is pushes onto exactly such a list -- declined and ran
         * interpreted: 270ms to 510ms. A new opcode in ordinary code is a JIT
         * admission question before it is anything else. */
        case OP_ELEM_KIND: {
            uint8_t packed = code[off + 1];
            if (e->depth == 0) return false;
            /* The STATIC kind, not the sampled value: the measuring pass runs
             * with no sample, so keying on stackSeen declined every time and
             * cost the whole function. OP_BUILD_LIST pushes SLOT_LIST, which is
             * exactly what the emitter puts this opcode after. */
            if (e->stack[e->depth - 1] != SLOT_LIST) {
                e->whyNot = "elem-kind target is not a known list";
                return false;
            }
            unsigned r = valueXReg(e, e->valueDepth - 1);
            emitConst64(e, JIT_SCRATCH_A, (int64_t)(packed & 0xFu));
            emit(e, jaiA64StrByte(JIT_SCRATCH_A, r,
                                  (unsigned)offsetof(ObjList, elemKind)));
            e->wroteHeap = true;
            off += 2;
            break;
        }

        case OP_CMP_LOCAL_CONST_LT: {
            unsigned slot = jaiReadU16(code + off + 1);
            int16_t  imm  = jaiReadI16(code + off + 3);
            if (!localInRange(e, slot)) return false;
            if (e->localKind[slot] != SLOT_INT) return false;
            if (slot == 0) e->usesSlot0 = true;

            /* Compare before pushing: the compare reads the local, and the
             * pushed entry is only the bool the flags produce. */
            if (imm >= -4095 && imm <= 4095) {
                emitCmpImm(e, localIn(e, slot, JIT_SCRATCH_C), imm);
            } else {
                emitConst64(e, JIT_SCRATCH_A, imm);
                emit(e, jaiA64SubsXReg(31, localIn(e, slot, JIT_SCRATCH_C),
                                       JIT_SCRATCH_A));
            }
            if (!pushValue(e, SLOT_BOOL, 0, NULL)) return false;
            emit(e, jaiA64CsetX(pushReg(e) - 1, JAI_A64_LT));
            off += 5;
            break;
        }

        case OP_ADD_INT_CONST: {
            unsigned slot = jaiReadU16(code + off + 1);
            int16_t  imm  = jaiReadI16(code + off + 3);
            if (!localInRange(e, slot)) return false;
            if (e->localKind[slot] != SLOT_INT) return false;
            if (slot == 0) e->usesSlot0 = true;
            if (!pushValue(e, SLOT_INT, 0, NULL)) return false;
            {
                unsigned dst = pushReg(e) - 1;
                unsigned cur = localIn(e, slot, JIT_SCRATCH_C);
                /* imm12, so a step outside +/-4095 still goes through a
                 * register. See OP_INC_LOCAL for why `subs` is the negative
                 * arm rather than a negated constant. */
                if (imm >= 0 && imm <= 4095) {
                    emit(e, jaiA64AddsXImm(dst, cur, (unsigned)imm));
                } else if (imm < 0 && imm >= -4095) {
                    emit(e, jaiA64SubsXImm(dst, cur, (unsigned)(-(int)imm)));
                } else {
                    emitConst64(e, JIT_SCRATCH_A, imm);
                    emit(e, jaiA64AddsX(dst, cur, JIT_SCRATCH_A));
                }
            }
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
                if (!adoptLocalKind(e, slot, SLOT_FLOAT, 0, NULL)) {
                    e->whyNot = "a local was given two different kinds";
                    return false;
                }
                unsigned ib = e->valueDepth - 1, ia = e->valueDepth - 2;
                unsigned db = fpOperand(e, ib), da = fpOperand(e, ia);
                unsigned rx; SlotKind kx;
                if (!popValueRaw(e, &rx, &kx)) return false;
                if (!popValueRaw(e, &rx, &kx)) return false;
                unsigned dd = fpBindDest(e, slot, fpRegAt(e, ia));
                emit(e, jaiA64FmulD(dd, da, db));
                localOutFp(e, slot, dd);
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
                if (!adoptLocalKind(e, slot, SLOT_FLOAT, 0, NULL)) {
                    e->whyNot = "a local was given two different kinds";
                    return false;
                }
                unsigned ib = e->valueDepth - 1, ia = e->valueDepth - 2;
                unsigned db = fpOperand(e, ib), da = fpOperand(e, ia);
                unsigned rx; SlotKind kx;
                if (!popValueRaw(e, &rx, &kx)) return false;
                if (!popValueRaw(e, &rx, &kx)) return false;
                unsigned dd = fpBindDest(e, slot, fpRegAt(e, ia));
                emit(e, jaiA64FsubD(dd, da, db));
                localOutFp(e, slot, dd);
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
            {
                unsigned cur = localIn(e, slot, JIT_SCRATCH_C);
                unsigned dst = localDest(e, slot);
                /* Step is an i8, always fits imm12, so the constant never needs a register of its own. `subs` for a
                 * negative step rather than a negated `adds`: both set V for the operation actually performed, which is what the overflow guard below reads. */
                if (imm >= 0) {
                    emit(e, jaiA64AddsXImm(dst, cur, (unsigned)imm));
                } else {
                    emit(e, jaiA64SubsXImm(dst, cur, (unsigned)(-(int)imm)));
                }
                localOut(e, slot, dst);
            }
            branchOnOverflow(e, 0u, JAI_A64_VS);
            off += 4;
            break;
        }

        case OP_EQ: case OP_NE:
        case OP_LT: case OP_LE: case OP_GT: case OP_GE: {
            /* Operands are read without popping them off the model: a NaN sends this back to the interpreter,
             * whose stack still has them. Popping first once left the model two entries short, so the re-run comparison silently read whatever was underneath -- a bug that surfaced one line later instead of not at all. */
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
            /* Read through xHeldIn, not straight out of the bank: an operand
             * that is a plain read of a local is still in the local's own
             * register. See popValue. */
            int64_t kcmp = 0;
            bool foldCmp = ka == SLOT_INT &&
                           pendingImm12(e, e->valueDepth - 1, &kcmp);
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
                /* nanToDeopt records, so nothing may still be deferred. A
                 * float local can be borrowed out of an X register in OSR
                 * mode; settling costs a mov on a path that is about to
                 * compare NaNs, and buys the compile. */
                settleAll(e);
                emit(e, jaiA64FcmpD(da, db));
                nanToDeopt(e);
            } else if (ka == SLOT_INT || ka == SLOT_MAYBE_INST) {
                unsigned ra = xHeldIn(e, e->valueDepth - 2);
                if (foldCmp) {
                    e->kPend &= ~(1u << (e->valueDepth - 1));
                    emitCmpImm(e, ra, kcmp);
                } else {
                    emit(e, jaiA64SubsXReg(31, ra,
                                           xHeldIn(e, e->valueDepth - 1)));
                }
            } else if ((op == OP_EQ || op == OP_NE) && ka == SLOT_OBJ &&
                       IS_STRING(e->stackSeen[e->depth - 2]) &&
                       IS_STRING(e->stackSeen[e->depth - 1])) {
                /* This path guards, so nothing may still be deferred when it
                 * does -- settled here, at the top of the path, which is where
                 * the settling is unconditionally executed. */
                settleAll(e);
                unsigned rb = valueXReg(e, e->valueDepth - 1);
                unsigned ra = valueXReg(e, e->valueDepth - 2);
                /* Two interned strings are equal exactly when they are the same object (`text[i] == " "` relies on
                 * this: one-character strings are shared singletons). Anything not interned, or not a string, deopts and the interpreter compares properly. */
                for (unsigned side = 0; side < 2; side++) {
                    unsigned r = side == 0 ? ra : rb;
                    /* Already known to be an interned string: `s[i] == t[j]`
                     * would otherwise guard twelve instructions to reach one
                     * compare. */
                    if (e->stackAscii[e->depth - (side == 0 ? 2u : 1u)]) continue;
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
            if (ka != kb) {
                bool mixable =
                    (ka == SLOT_INST && kb == SLOT_MAYBE_INST) ||
                    (ka == SLOT_MAYBE_INST && kb == SLOT_INST);
                if (!mixable) return false;
                ka = kb = SLOT_MAYBE_INST;
            }
            if (!holdsRegister(ka)) return false;
            /* See the OP_LT..OP_GE arm: operands are read through xHeldIn, a
             * literal right-hand side becomes the compare's own immediate,
             * and the paths that guard settle first. */
            int64_t kcmp2 = 0;
            bool foldCmp2 = ka == SLOT_INT &&
                            pendingImm12(e, e->valueDepth - 1, &kcmp2);
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
                unsigned db = fpOperand(e, e->valueDepth - 1);
                unsigned da = fpOperand(e, e->valueDepth - 2);
                settleAll(e);          /* nanToDeopt records */
                emit(e, jaiA64FcmpD(da, db));
                nanToDeopt(e);
            } else if (ka == SLOT_INT || ka == SLOT_MAYBE_INST) {
                unsigned ra = xHeldIn(e, e->valueDepth - 2);
                if (foldCmp2) {
                    e->kPend &= ~(1u << (e->valueDepth - 1));
                    emitCmpImm(e, ra, kcmp2);
                } else {
                    emit(e, jaiA64SubsXReg(31, ra,
                                           xHeldIn(e, e->valueDepth - 1)));
                }
            } else if ((cmp == OP_EQ || cmp == OP_NE) && ka == SLOT_OBJ &&
                       IS_STRING(e->stackSeen[e->depth - 2]) &&
                       IS_STRING(e->stackSeen[e->depth - 1])) {
                settleAll(e);          /* this path guards */
                unsigned rb = valueXReg(e, e->valueDepth - 1);
                unsigned ra = valueXReg(e, e->valueDepth - 2);
                for (unsigned side = 0; side < 2; side++) {
                    unsigned r = side == 0 ? ra : rb;
                    /* See the same skip in OP_EQ. */
                    if (e->stackAscii[e->depth - (side == 0 ? 2u : 1u)]) continue;
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
            /* Nothing reads a value that is being thrown away, so a deferred
             * entry is forgotten rather than settled. Without this, `a = b`
             * paid at the POP the copy the borrow had just saved. */
            if (e->depth > 0 && holdsRegister(e->stack[e->depth - 1]) &&
                e->valueDepth > 0) {
                unsigned idx = e->valueDepth - 1;
                e->kPend   &= ~(1u << idx);
                e->xBorrow &= ~(1u << idx);
            }
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
                int64_t kv = AS_INT(k);
                if (off + 4 < stop && foldsIntLiteral(code[off + 4]) &&
                    kv >= -4095 && kv <= 4095) {   /* see OP_INT */
                    kPendLocal(e, e->valueDepth - 1, kv);
                } else {
                    emitConst64(e, pushReg(e) - 1, kv);
                }
            } else if (IS_FLOAT(k)) {
                /* The bits, not the number: a float lives in an X register exactly as in a Value's payload -- unless
                 * the value is one FMOV's 8-bit immediate can name, in which case it goes straight to the FP bank in one instruction instead of two plus an fmov. */
                double d = AS_FLOAT(k);
                int64_t bits;
                memcpy(&bits, &d, sizeof bits);
                if (!pushValue3(e, SLOT_FLOAT, 0, NULL, k, -1)) return false;
                unsigned imm8;
                if (!e->fpOff && jaiA64FpImm8(d, &imm8)) {
                    /* Not `idx`: that is this instruction's constant index, and
                     * shadowing it here was the tree's only build warning. This
                     * one is a position on the operand stack. */
                    unsigned at = e->valueDepth - 1;
                    emit(e, jaiA64FmovDImm(fpRegAt(e, at), imm8));
                    fpClaim(e, at);
                } else {
                    emitConst64(e, pushReg(e) - 1, bits);
                }
            } else if (IS_BOOL(k)) {
                if (!pushValue3(e, SLOT_BOOL, 0, NULL, k, -1)) return false;
                emitConst64(e, pushReg(e) - 1, AS_BOOL(k) ? 1 : 0);
            } else if (IS_STRING(k)) {
                /* Pointer to the constant pool's own string, safe to hold raw for the same reason resolved globals
                 * are: the pool belongs to the chunk, the chunk to the function, and the caller holds the closure for the whole call. Was the commonest reason this tier declined a body -- thirty refusals across the benchmark suite. */
                if (!pushValue3(e, SLOT_OBJ, 0, NULL, k, -1)) return false;
                emitConst64(e, pushReg(e) - 1, (int64_t)(uintptr_t)AS_OBJ(k));
            } else {
                return false;
            }
            off += 4;
            break;
        }

        case OP_TYPE_GUARD: {
            /* A declared boundary (parameter or return type). The kind is already known here, so the guard is
             * either nothing at all or the int-to-float widening the interpreter does at the same place (spec 2.2). Anything the kinds can't settle is declined, not guessed: `evalA` in spectral was refused outright for want of this. */
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
            } else {
                e->whyNot = "a type guard the kinds cannot settle";
                return false;
            }
            off += 4;
            break;
        }

        case OP_FORMAT: {
            /* Largest single refusal reason across the benchmark census (ninety) -- every f-string is one, and
             * dict_ops, word_freq and string_build all build their keys with one. */
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
            /* Compiled code has no safepoint on the back edge (uninterruptible, unsampled) -- acceptable only
             * because this tier bails on anything unbounded: the loop is over ints, can't allocate, and the stack guard still catches runaway recursion. Ctrl-C during a long compiled loop waits for the loop to end. */
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
                unsigned dm = fpRegAt(e, ia);
                uint32_t bindOffM = 0;
                unsigned homeM = fpBindLookahead(e, code, off + 1, stop, fn,
                                                 &bindOffM);
                if (homeM != 0) {
                    fpReleaseHome(e, homeM);
                    dm = homeM;
                }
                emit(e, jaiA64FmulD(dm, da, db));
                if (homeM != 0) {
                    fpBorrowLocal(e, ia, homeM);
                    e->homeEarly[e->homeEarlyCount++] = bindOffM;
                } else {
                    fpClaim(e, ia);
                }
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
            /* None of `& | ^` can fail, so they're one instruction. Shifts have two edges hardware doesn't share:
             * jaithon throws on a negative count and saturates at >=64, while arm64's LSLV/ASRV wrap on the low six bits of the count. Both edges are guarded and deopt; the guards precede the pops, so a deopt resumes at an instruction that hasn't happened yet. */
            unsigned rb, ra;
            SlotKind kb, ka;
            if (e->depth < 2) return false;
            if (e->stack[e->depth - 1] != SLOT_INT) return false;
            if (e->stack[e->depth - 2] != SLOT_INT) return false;

            /* A literal count settles both edges here rather than at run time: the immediate-form shift then IS
             * the interpreter's rule for that count. bitops shifts by 7, 3, 11, 1 and 31 and paid five instructions and two deopt sites for each before this. */
            int64_t kcount;
            bool kcountUsable =
                (op == OP_SHL || op == OP_SHR) &&
                literalIntOperand(fn, prevOff, off, &kcount) &&
                kcount >= 0 && kcount <= 63;

            if ((op == OP_SHL || op == OP_SHR) && !kcountUsable) {
                /* The count reaches a register and a guard, so it has to be
                 * in one. OP_INT may have deferred it on the strength of the
                 * opcode alone -- a count of 100 is a literal this arm cannot
                 * fold -- and this is where that is put right. */
                settleAll(e);
            }
            if (op != OP_SHL && op != OP_SHR) settleAll(e);
            if (kcountUsable) e->kPend &= ~(1u << (e->valueDepth - 1));

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
                uint32_t bindOff = 0;
                unsigned home = fpBindLookahead(e, code, off + 1, stop, fn,
                                                &bindOff);
                if (home != 0) {
                    fpReleaseHome(e, home);
                    dd = home;
                }
                emit(e, op == OP_ADD ? jaiA64FaddD(dd, da, db)
                     : op == OP_SUB  ? jaiA64FsubD(dd, da, db)
                                     : jaiA64FdivD(dd, da, db));
                if (home != 0) {
                    fpBorrowLocal(e, ia, home);
                    e->homeEarly[e->homeEarlyCount++] = bindOff;
                } else {
                    fpClaim(e, ia);
                }
                off += 1;
                break;
            }
            /* A literal right operand becomes the immediate `adds`/`subs`
             * already takes, so it never reaches a register. Decided before
             * the pops, which is where the entry index still names it. */
            int64_t kimm = 0;
            bool foldK = op != OP_DIV && e->depth >= 2 &&
                         e->stack[e->depth - 1] == SLOT_INT &&
                         e->stack[e->depth - 2] == SLOT_INT &&
                         pendingImm12(e, e->valueDepth - 1, &kimm);
            if (foldK) e->kPend &= ~(1u << (e->valueDepth - 1));

            if (!popValue(e, &rb, &kb)) return false;
            if (!popValue(e, &ra, &ka)) return false;
            if (ka != kb) return false;   /* no implicit widening here */

            /* Integer division is not here: it has a zero-divisor error and a
             * truncation rule of its own, and getting either wrong would be a
             * wrong answer rather than a decline. */
            if (ka != SLOT_INT || op == OP_DIV) return false;
            if (!pushValue(e, SLOT_INT, 0, NULL)) return false;
            unsigned rd = pushReg(e) - 1;
            if (foldK) emitAddSubImm(e, rd, ra, kimm, op == OP_SUB);
            else emit(e, op == OP_ADD ? jaiA64AddsX(rd, ra, rb)
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
            if (kIdx >= (uint32_t)fn->chunk.constants.count) return false;
            Value k = fn->chunk.constants.data[kIdx];

            /* `if c == "{"` where c came out of a string index. Every hand-written scanner in the language is
             * this shape, and the peephole folds it to exactly this instruction, so declining it declined the
             * whole enclosing function -- json_parse's `value` dispatches on six of them. Two interned strings
             * are equal exactly when they are the same object, which the OP_EQ arm already relies on; the
             * difference here is that one side is a constant, so its interning is settled at compile time and
             * only the local needs guarding. Nothing has been written yet, so a guard resumes at this very
             * instruction. */
            if ((cmp == OP_EQ || cmp == OP_NE) &&
                e->localKind[slot] == SLOT_OBJ && IS_STRING(k) &&
                JAI_STR_INTERNED(AS_STRING(k)) &&
                IS_STRING(localObserved(e, slot) ? e->observed[slot]
                                                 : e->localSeen[slot])) {
                if (slot == 0) e->usesSlot0 = true;
                settleAll(e);          /* this path guards */
                unsigned rs = localIn(e, slot, JIT_SCRATCH_C);
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, rs,
                                   (unsigned)offsetof(Obj, type)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_STRING));
                branchOnDeopt(e, JAI_A64_NE);
                /* Not interned means content equality is not pointer equality,
                 * and the interpreter is the one that knows how to tell. */
                emit(e, jaiA64LdrByte(JIT_SCRATCH_A, rs,
                                      (unsigned)offsetof(Obj, subFlag)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, 0));
                branchOnDeopt(e, JAI_A64_EQ);
                emitConst64(e, JIT_SCRATCH_B, (int64_t)(uintptr_t)AS_OBJ(k));
                emit(e, jaiA64SubsXReg(31, rs, JIT_SCRATCH_B));
                branchTo(e, (uint32_t)((int32_t)next + jump), true, cond);
                off += 9;
                break;
            }

            if (e->localKind[slot] != SLOT_INT) return false;
            if (slot == 0) e->usesSlot0 = true;
            if (!IS_INT(k)) return false;

            /* `while i < n` and `if n < 2` are the same instruction here, and the constant fits the compare's
             * own imm12 far more often than not, so it costs one instruction rather than a movz plus a three-register subs. */
            int64_t kv = AS_INT(k);
            if (kv >= -4095 && kv <= 4095) {
                emitCmpImm(e, localIn(e, slot, JIT_SCRATCH_C), kv);
            } else {
                emitConst64(e, JIT_SCRATCH_A, kv);
                emit(e, jaiA64SubsXReg(31, localIn(e, slot, JIT_SCRATCH_C),
                                       JIT_SCRATCH_A));
            }
            branchTo(e, (uint32_t)((int32_t)next + jump), true, cond);
            off += 9;
            break;
        }

        case OP_GET_FIELD_LOCAL: {
            unsigned slot    = jaiReadU16(code + off + 1);
            uint32_t nameIdx = jaiReadU24(code + off + 3);

            /* A maybe-instance reads like an instance once known not-null. The program has usually just tested
             * it (`if node == null { return 0 }`) but the tier doesn't track that, so the guard stands and costs one compare against zero; a null arriving for real just deopts. */
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

            /* Field type is read off the LIVE receiver, so the tier specialises to what the program actually
             * stores rather than a declaration -- only possible for a parameter (hence the arity cap above): a local assigned further in has no value yet to look at. */
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
                /* A tree's `left` is a Node on one call and null on the next; a leaf has no class to read, so the
                 * receiver's own class is the guess (a nullable instance field is usually a linked structure) -- safe because the class guard below just deopts if wrong, never miscompiles. */
                kind = SLOT_MAYBE_INST;
                tag  = VAL_OBJ;
                fcls = IS_INSTANCE(fieldVal) ? AS_INSTANCE(fieldVal)->klass
                                             : e->localClass[slot];
                if (fcls == NULL) return false;
            }
            /* A list field earns the stronger kind: SLOT_OBJ can be passed and stored but not iterated,
             * indexed or pushed to, and `for x in self.items` / `self.items.push(v)` is the commonest thing
             * a class holding a list does. Paid for with the OBJ_LIST check below, since VAL_OBJ alone
             * would let a str reach a header read. */
            else if (IS_LIST(fieldVal)) { kind = SLOT_LIST; tag = VAL_OBJ; }
            /* A str/dict/set field, held raw -- the same contract as a SLOT_OBJ global or list element:
             * the tag guard below says "an object", the sample says which type it was, and every consumer
             * (index, invoke, compare) re-checks Obj.type for itself before it does anything type-specific.
             * Refusing this declined the whole enclosing FUNCTION, which is most object-oriented code. */
            else if (rawObjValue(fieldVal)) { kind = SLOT_OBJ; tag = VAL_OBJ; }
            else return false;

            unsigned base = (unsigned)offsetof(ObjInstance, fields) +
                            (unsigned)info->slot * (unsigned)sizeof(Value);
            /* The tag is checked every time, unless this body wrote the field
             * itself. A field is not typed by the runtime, so a later int
             * where a float was seen must bail rather than be read as one. */
            unsigned recv = localIn(e, slot, JIT_SCRATCH_C);
            SlotKind already = knownFieldKind(e, (int)slot, info->slot);
            if (kind == SLOT_MAYBE_INST) {
                /* Three guards, branch-free via a trick: when the field is null, the loads below are redirected at
                 * the receiver instead (already a live instance, per the entry guard/null check above) -- a real object of the right type, so nothing ever dereferences zero. */
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
                /* "an object" is not "a list": every SLOT_LIST consumer reads the header with no check of
                 * its own, so the object type is confirmed here, once, before the kind is handed out.
                 * `already` is never SLOT_LIST (see recordFieldStore's caller), so this arm sees them all. */
                if (kind == SLOT_LIST) {
                    emit(e, jaiA64LdrX(JIT_SCRATCH_D, recv, base + 8));
                    emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_D,
                                       (unsigned)offsetof(Obj, type)));
                    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_LIST));
                    branchOnDeopt(e, JAI_A64_NE);
                }
            }

            if (kind == SLOT_MAYBE_INST || kind == SLOT_LIST) {
                if (!pushValue3(e, kind,
                                kind == SLOT_MAYBE_INST ? fcls->shapeId : 0,
                                kind == SLOT_MAYBE_INST ? fcls : NULL,
                                kind == SLOT_LIST ? fieldVal
                                : IS_INSTANCE(fieldVal) ? fieldVal : NULL_VAL,
                                -1)) {
                    return false;
                }
                emit(e, jaiA64MovX(pushReg(e) - 1, JIT_SCRATCH_D));
            } else {
                /* SLOT_OBJ needs the sample carried: it is the only record of which object type this was,
                 * and a consumer with nothing to look at declines. NULL_VAL when `already` overrode the
                 * observation, since then the stored kind and the sampled value need not agree. */
                if (!pushValue3(e, kind, 0, NULL,
                                kind == SLOT_OBJ && rawObjValue(fieldVal)
                                    ? fieldVal : NULL_VAL,
                                -1)) {
                    return false;
                }
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
            /* The second local may guard (if dynamic), and a guard can't be reached with a borrow live -- the
             * deopt stub writes float entries out of fpRegAt, where a borrowed one isn't. So the FIRST local takes a copy instead of a borrow whenever the second is going to guard: `dt * b.vx` in nbody's second loop is exactly this shape (`dt` has a home, `b` is dynamic). */
            bool guardFollows = b <= JIT_MAX_SLOTS && e->dynamicLocal[b];
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
                    if (e->slotFpReg[slot] != 0 && !(k == 0 && guardFollows)) {
                        fpBorrowLocal(e, idx, e->slotFpReg[slot]);
                    } else {
                        localInFp(e, slot, fpRegAt(e, idx));
                        fpClaim(e, idx);
                    }
                } else {
                    unsigned home = localHomeX(e, slot);
                    if (home != 0) {            /* see OP_GET_LOCAL */
                        xBorrowLocal(e, e->valueDepth - 1, home);
                    } else {
                        unsigned dst = pushReg(e) - 1;
                        unsigned src = localIn(e, slot, dst);
                        if (src != dst) emit(e, jaiA64MovX(dst, src));
                    }
                }
            }
            off += 5;
            break;
        }

        case OP_SET_FIELD: {
            uint32_t nameIdx = jaiReadU24(code + off + 1);
            /* Receiver then value, both dropped. The receiver's class comes from its STACK entry, not guessed
             * from the locals: two instance locals of different classes would make any guess a silently wrong field offset. */
            if (e->depth < 2) return false;
            ObjClass *klass = e->stackClass[e->depth - 2];
            int recvLocal = e->stackLocal[e->depth - 2];

            unsigned rv, rr;
            SlotKind kv, kr;
            /* A float already in the FP bank is stored from there (`str d`, not popValue's `fmov x,d` + `str x`)
             * -- captured before the pop, since popping is what clears the bit and renames the index. */
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
            /* An object goes in as readily as a number. The collector is a plain mark-sweep with no write
             * barrier and nothing moves, so the only question is reachability: the receiver is rooted (it
             * is a live SLOT_INST here), and after the store the value hangs off it, while before the store
             * it was rooted in its own right by emitRootFill. What is refused is a kind with no payload
             * register to store (class/function/native/self) and SLOT_ITER, whose index lives in memory. */
            if (kv != SLOT_INT && kv != SLOT_FLOAT && kv != SLOT_BOOL &&
                kv != SLOT_OBJ && kv != SLOT_LIST && kv != SLOT_INST &&
                kv != SLOT_MAYBE_INST) {
                e->whyNot = "storing a field kind this tier cannot write";
                return false;
            }
            if (nameIdx >= (uint32_t)fn->chunk.constants.count) return false;
            Value nameVal = fn->chunk.constants.data[nameIdx];
            if (!IS_STRING(nameVal)) return false;

            if (klass == NULL) return false;
            const FieldInfo *info = jaiClassFieldInfo(klass, AS_STRING(nameVal));
            if (info == NULL || info->isStatic) return false;

            unsigned base = (unsigned)offsetof(ObjInstance, fields) +
                            (unsigned)info->slot * (unsigned)sizeof(Value);
            /* Not a constant tag any more: a maybe-instance's is null-or-object,
             * read off the payload, which is exactly what emitTagFor does. */
            emitTagFor(e, kv, rv, JIT_SCRATCH_A, JIT_SCRATCH_B);
            emit(e, jaiA64StrW(JIT_SCRATCH_A, rr, base));
            if (vIsFp) emit(e, jaiA64StrD(dv, rr, base + 8));
            else       emit(e, jaiA64StrX(rv, rr, base + 8));
            /* Only a kind a later read can replay EXACTLY is remembered; the rest merely retire what was
             * known of this field (local -1), because the read side would otherwise take SLOT_INST with no
             * class behind it -- a shape every field offset it then resolved would be resolved against. */
            bool replayable = kv == SLOT_INT || kv == SLOT_FLOAT ||
                              kv == SLOT_BOOL || kv == SLOT_OBJ;
            recordFieldStore(e, replayable ? recvLocal : -1, info->slot, kv);
            e->wroteHeap = true;
            off += 6;
            break;
        }

        case OP_RETURN_NULL: {
            /* An OSR form's x0 is the resume bytecode offset (see jaiJitEnterOsr's `*resumeAt = at`), but this
             * return sequence is the function tier's and leaves the VALUE in x0 -- a `return` inside a compiled loop once handed the interpreter an int/pointer as an instruction offset instead, miscompiling `for i in 0..n { if .. { return x } }` (14/20 wrong in released builds, 10/10 under --gc-stress). */
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
            /* Whose closure. An inlined body's is in the register the call
             * site guarded, not in the caller's own closure register: the
             * caller may have no upvalues at all and still be inlining a body
             * that has them. */
            unsigned creg;
            if (e->inlining) {
                if (e->inlClosureReg < 0) return false;
                creg = (unsigned)e->inlClosureReg;
            } else {
                if (!e->usesUpvalues) return false; /* decided before this pass */
                creg = closureReg(e);
            }

            /* closure->upvalues[index]->location, then the Value there. The
             * upvalue may still be open, pointing into the VM stack, so the
             * location is followed rather than assumed closed. */
            emit(e, jaiA64LdrX(JIT_SCRATCH_A, creg,
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
            /* As in OP_GET_FIELD_LOCAL: an object-typed field is held raw
             * rather than declining the enclosing function, and a list earns
             * the stronger kind at the price of an OBJ_LIST check. */
            else if (IS_LIST(fieldVal))     { kind = SLOT_LIST; tag = VAL_OBJ; }
            else if (rawObjValue(fieldVal)) { kind = SLOT_OBJ;  tag = VAL_OBJ; }
            else return false;

            unsigned fbase = (unsigned)offsetof(ObjInstance, fields) +
                             (unsigned)info->slot * (unsigned)sizeof(Value);
            unsigned rr = valueBankBase(e) + e->valueDepth - 1;
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
                /* And that it is a list, not merely an object -- same reason as
                 * in OP_GET_FIELD_LOCAL, and likewise while the receiver is
                 * still on the model, since this guard resumes here too. */
                if (kind == SLOT_LIST) {
                    emit(e, jaiA64LdrX(JIT_SCRATCH_D, rr, fbase + 8));
                    emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_D,
                                       (unsigned)offsetof(Obj, type)));
                    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_LIST));
                    branchOnDeopt(e, JAI_A64_NE);
                }
            }
            unsigned popped;
            SlotKind kr;
            if (!popValue(e, &popped, &kr)) return false;
            if (!pushValue3(e, kind, 0, NULL,
                            (kind == SLOT_OBJ && rawObjValue(fieldVal)) ||
                                    (kind == SLOT_LIST && IS_LIST(fieldVal))
                                ? fieldVal : NULL_VAL,
                            -1)) {
                return false;
            }
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
            /* Only `** 0.5` (square root): C's pow(x,0.5) is sqrt(x) for x >= +0, differing only at -0.0 (pow
             * gives +0.0, sqrt gives -0.0) -- so a negative sign bit sends this back to the interpreter. */
            if (e->depth < 2) return false;
            if (e->stack[e->depth - 1] != SLOT_FLOAT) return false;
            if (e->stack[e->depth - 2] != SLOT_FLOAT) return false;
            Value expv = e->stackSeen[e->depth - 1];
            if (!IS_FLOAT(expv) || AS_FLOAT(expv) != 0.5) {
                e->whyNot = "an exponent other than 0.5";
                return false;
            }
            /* Wholly in the FP bank: the sign bit is the only thing needing an integer register, and an `fmov`
             * out of d costs one instruction where routing the operand through X cost four (two for an exponent constant nothing reads, two more around the fsqrt). */
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

            /* A literal power-of-two modulus decides the whole thing: floor remainder by 2^s is exactly the low
             * s bits, which two's complement already holds -- exact even for negative dividends, unlike the truncating `msub` path below (whose correction exists to fix exactly that). */
            int64_t kmod = 0;
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

            /* Remainder lands in `rx`, where the result belongs: two entries come off, one goes on, so the
             * surviving (lower) entry keeps its register -- removes a trailing `mov`. Safe because every guard this arm emits is above this line: nothing below can deopt and find the dividend gone. */
            emit(e, jaiA64SdivX(JIT_SCRATCH_B, rx, ry));
            emit(e, jaiA64MsubX(rx, JIT_SCRATCH_B, ry, rx));
            emitFloorFixup(e, rx, ry, kmodKnown, kmod,
                           jaiA64AddX(rx, rx, ry));

            unsigned dm1, dm2;
            if (!popValue(e, &dm1, NULL)) return false;
            if (!popValue(e, &dm2, NULL)) return false;
            if (!pushValue(e, SLOT_INT, 0, NULL)) return false;
            off += 1;
            break;
        }

        case OP_MOD_INT_CONST: {
            /* `<int k>; MOD` fused: k is known non-zero (fusion requires it); -1 goes back to the interpreter so
             * INT64_MIN %% -1 stays its problem. */
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
            emit(e, jaiA64MsubX(rx, JIT_SCRATCH_B, JIT_SCRATCH_A, rx));
            emitFloorFixup(e, rx, JIT_SCRATCH_A, true, imm,
                           jaiA64AddX(rx, rx, JIT_SCRATCH_A));
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

            /* A literal power-of-two divisor decides the whole thing: floor(x / 2^s) is exactly `asr x, #s` for
             * every int64 x (negative included), since asr already rounds toward minus infinity -- what the correction below exists to reproduce for the general case. */
            int64_t kdiv = 0;
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

            /* Zero and -1 both decline: the interpreter reports division-by-zero, and INT64_MIN / -1 is the one
             * quotient that doesn't fit -- both rare enough that declining -1 outright costs nothing. A literal divisor has already answered both. */
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

            /* Quotient lands in the register the dividend was in (the push reuses it), so the dividend is copied
             * out first -- without that, msub read a value sdiv had already overwritten and 7 // 2 came out 2. */
            emit(e, jaiA64MovX(JIT_SCRATCH_C, ra));

            /* Truncating quotient, then one down when the remainder is nonzero
             * and its sign differs from the divisor's -- which is what makes
             * this floor division rather than C's. */
            emit(e, jaiA64SdivX(rq, JIT_SCRATCH_C, rb));
            emit(e, jaiA64MsubX(JIT_SCRATCH_A, rq, rb, JIT_SCRATCH_C));
            /* A literal divisor makes the correction a single sign test on the remainder (see emitFloorFixup).
             * JIT_SCRATCH_B is free again here -- the quotient is in rq, not in it. */
            emitFloorFixup(e, JIT_SCRATCH_A, rb, kdivKnown, kdiv,
                           jaiA64SubXImm(rq, rq, 1));
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
                /* Appending to a list is a bounds check and two stores -- a descriptor+native round trip costs far
                 * more than the work itself (list_ops spent all its time on the call). A full list goes out to the `grow` stubs' realloc helper and comes straight back; see there for why this used to be a deopt and what that cost. */
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
                    noteScratchClobber(e);
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
                /* A method whose whole body is one arithmetic expression over receiver+arguments is worth inlining:
                 * the call costs more than the expression. Vec2.dot is four field reads, two multiplies and an add, reached through a descriptor, a helper and a compiled entry. */
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
                /* The callee's own compiled form states its return kind exactly; without one the
                 * interpreter's record of what it has been returning stands in. That case is not exotic --
                 * two methods that call each other can never take turns being the first to compile, so
                 * neither ever has a jitFunc to ask, and json_parse's whole parser is that shape. Either
                 * way it is only a prediction: the tag guard below is what makes it sound. */
                SlotKind rkind = SLOT_NULL;
                uint32_t rshape = 0;
                ObjClass *rrcls = NULL;
                bool haveKind;
                if (mfn->jitFunc != NULL) {
                    rkind = (SlotKind)mfn->jitReturnKind;
                    rshape = mfn->jitReturnShape;
                    haveKind = true;
                } else {
                    haveKind = observedReturnKind(mfn, &rkind, &rshape);
                }
                if (haveKind && rkind == SLOT_INST &&
                    (rshape == 0 || !jaiClassForShape(rshape, &rrcls) ||
                     rrcls == NULL)) {
                    haveKind = false;
                    rrcls = NULL;
                }
                if (haveKind && rkind != SLOT_INT && rkind != SLOT_FLOAT &&
                    rkind != SLOT_BOOL && rkind != SLOT_INST &&
                    rkind != SLOT_LIST && rkind != SLOT_OBJ &&
                    rkind != SLOT_NULL) {
                    haveKind = false;
                }
                /* A result the very next instruction pops needs no kind at all -- which is every `-> void`
                 * method called as a statement, and the reason a parser's `self.skip()` used to decline the
                 * function around it. Same relaxation the builtin arm below already makes. */
                bool mdiscarded = (off + 7 < count && code[off + 7] == OP_POP);
                if (!haveKind && !mdiscarded) {
                    e->whyNot = "callee's return kind not usable";
                    return false;
                }

                /* Straight to the method's compiled entry since the receiver's class is fixed here (SLOT_INST
                 * carries a shape the caller already guarded) -- the "inline cache" is the model itself, free at run time. Skips the whole jitInvokeMethod -> jaiCallMethodWithReceiver -> invokeCallable -> callClosure -> jaiJitEnterFunc -> jitArgIn chain of C glue between two compiled bodies, which was a large fraction of object_dispatch. Falls back rather than declines, for the same reason a global call does: the descriptor path speaks a wider language. */
                if (mfn->jitFunc != NULL) {
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
                if (!haveKind) {
                    /* Nothing observes the result, so nothing has to be
                     * predicted or guarded about it. */
                    e->wroteHeap = true;
                    off += 8;          /* the OP_POP this consumed */
                    break;
                }
                if (!pushValue(e, rkind, rshape, rrcls)) return false;

                unsigned rat = e->descOffset +
                               (unsigned)offsetof(JitCallDesc, result);
                unsigned wantTag = rkind == SLOT_INT   ? VAL_INT
                                 : rkind == SLOT_FLOAT ? VAL_FLOAT
                                 : rkind == SLOT_BOOL  ? VAL_BOOL
                                 : rkind == SLOT_NULL  ? VAL_NULL
                                                       : VAL_OBJ;
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, 31, rat));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, wantTag));
                branchOnDeoptAt(e, JAI_A64_NE, (uint32_t)(off + 7), true);
                /* Bool result is one byte, not eight (see the SLOT_OBJ arm below for why) -- latent here since this
                 * was written: needs a bool-returning method emitDirectCall declines, which nothing in the suite exercises. */
                if (rkind == SLOT_BOOL) {
                    emit(e, jaiA64LdrByte(pushReg(e) - 1, 31, rat + 8));
                } else if (rkind == SLOT_NULL) {
                    /* A null Value's payload word is not written by NULL_VAL, so
                     * there is nothing to load; SLOT_NULL is a defined zero. */
                    emit(e, jaiA64MovzX(pushReg(e) - 1, 0, 0));
                } else {
                    emit(e, jaiA64LdrX(pushReg(e) - 1, 31, rat + 8));
                }
                if (rkind == SLOT_INST) {
                    /* "an object" is not "an object of this class", and a
                     * method entered with another specialisation runs
                     * interpreted and may return either. The object TYPE goes
                     * first: VAL_OBJ is every heap object, so a method that
                     * returned a string here would have `klass` read one word
                     * past its header and the shape loaded through whatever was
                     * there -- a segfault, reachable from ordinary code. */
                    emit(e, jaiA64LdrW(JIT_SCRATCH_A, pushReg(e) - 1,
                                       (unsigned)offsetof(Obj, type)));
                    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_INSTANCE));
                    branchOnDeoptAt(e, JAI_A64_NE, (uint32_t)(off + 7), true);
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

            if (rk == SLOT_OBJ) {
                /* Built-in method on a receiver typed only as "some object" (dict/string/set/tuple). Three things:
                 * WHICH METHOD -- from the observed receiver, like the SLOT_LIST arm below (a builtin is a function of receiver-type + name). THAT IT'S STILL THAT TYPE -- SLOT_OBJ pins nothing (`for x in [d, "s"]` mixes types), so the object type is guarded before anything is consumed; a miss resumes with receiver+args untouched. WHAT COMES BACK -- predicted via InlineCache::resultKind (no per-call-site feedback existed before), and the tag guard after the call is what makes the prediction sound, deopting to the instruction AFTER the call since it already happened. */
                Value oseen = e->stackSeen[ridx];
                if (!IS_OBJ(oseen)) {
                    e->whyNot = "an invoke on an object with nothing to look at";
                    return false;
                }
                if (nameIdx >= (uint32_t)fn->chunk.constants.count) return false;
                Value oname = fn->chunk.constants.data[nameIdx];
                if (!IS_STRING(oname)) return false;
                Value obound;
                if (!jaiBuiltinMethod(oseen, AS_STRING(oname), &obound)) {
                    e->whyNot = "an invoke that is not a builtin of the "
                                "observed receiver's type";
                    return false;
                }
                Value onative = IS_BOUND(obound) ? AS_BOUND(obound)->method
                                                 : obound;
                if (!IS_NATIVE(onative)) return false;

                bool odiscarded = (off + 7 < count && code[off + 7] == OP_POP);
                SlotKind orkind = SLOT_INT;
                unsigned owantTag = VAL_INT;
                if (!odiscarded) {
                    uint8_t fb = jaiInvokeResultFeedback(
                        &fn->chunk, jaiReadU16(code + off + 5), oseen);
                    if (!feedbackSlotKind(fb, &orkind, &owantTag)) {
                        e->whyNot = "a builtin whose result kind has not been "
                                    "observed";
                        return false;
                    }
                }

                emit(e, jaiA64LdrW(JIT_SCRATCH_A, pushReg(e) - argc - 1,
                                   (unsigned)offsetof(Obj, type)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A,
                                       (unsigned)OBJ_TYPE(oseen)));
                branchOnDeopt(e, JAI_A64_NE);

                if (!emitDescriptor(e, onative, ridx, argc + 1,
                                    (void *)&jitInvokeNative)) {
                    return false;
                }
                for (unsigned i = 0; i <= argc; i++) {
                    unsigned r;
                    if (!popValue(e, &r, NULL)) return false;
                }
                e->wroteHeap = true;
                if (odiscarded) {
                    off += 8;          /* the OP_POP this consumed */
                    break;
                }
                if (!pushValue(e, orkind, 0, NULL)) return false;
                unsigned orat = e->descOffset +
                                (unsigned)offsetof(JitCallDesc, result);
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, 31, orat));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, owantTag));
                branchOnDeoptAt(e, JAI_A64_NE, (uint32_t)(off + 7), true);
                /* One byte for a bool: BOOL_VAL writes only the union's `boolean` member, so an 8-byte load would
                 * pull in whatever garbage was above it in that Value's slot, and every SLOT_BOOL consumer does `cbnz` on the whole word. Cost a day: the (self-hosted) front end's own `flags.get(name, false)` came back true from garbage bits, misreporting a non-variadic parameter as `list[fn(T) -> bool]`. */
                if (orkind == SLOT_BOOL) {
                    emit(e, jaiA64LdrByte(pushReg(e) - 1, 31, orat + 8));
                } else {
                    emit(e, jaiA64LdrX(pushReg(e) - 1, 31, orat + 8));
                }
                off += 7;
                break;
            }

            if (rk != SLOT_LIST) return false;
            if (nameIdx >= (uint32_t)fn->chunk.constants.count) return false;
            Value nameVal = fn->chunk.constants.data[nameIdx];
            if (!IS_STRING(nameVal)) return false;

            /* Which method a name means depends on the receiver's type, not what it holds -- so for a list this
             * body built with no sample to look at, an empty probe list answers just as well. Rooted across the lookup since resolving allocates the bound wrapper; only the native is kept, and that outlives it. */
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

            /* What comes back: `len` is an int; anything whose result is dropped on the next instruction needs no
             * kind at all. Nothing else -- a call's result can't be guarded, since the call already happened and a wrong guess has nowhere to go. */
            const char *mname = AS_STRING(nameVal)->chars;
            bool discarded = (off + 7 < count && code[off + 7] == OP_POP);
            bool isLen = strcmp(mname, "len") == 0 && argc == 0;
            if (!isLen && !discarded) return false;

            /* `xs.len()` on a list is one field read. Through the descriptor it meant a GC root push/pop, a
             * bound-method resolve, an arity check and a native call just to read a 32-bit count -- paid every iteration of the ordinary `while i < xs.len()` loop. The receiver's kind is already known here, which is the whole guard needed. */
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
            e->rangeBuildIp = (uint32_t)off;
            /* Both ends hold registers, so the low end is the entry one below
             * the top in the value bank as well as on the stack. */
            {
                unsigned lo = e->valueDepth - 2;
                e->rangeStartKnown = (e->kKnown & (1u << lo)) != 0;
                e->rangeStartVal   = e->rangeStartKnown ? e->kKnownVal[lo] : 0;
            }
            off += 2;
            break;
        }

        case OP_GET_ITER: {
            if (!e->pendingRange) {
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
                /* Shape 1 marks an iterator the runtime has to step; a range is 0 and gets the inline path. Rides on
                 * the stack entry, not the Emit, since a function can build both -- nbody's `advance` runs two range loops then a list loop, and one whole-compile flag made the path taken depend on what came before it. */
                if (!pushValue3(e, SLOT_ITER, 1, NULL, sample, -1)) return false;
                emit(e, jaiA64LdrX(pushReg(e) - 1, 31,
                                   e->descOffset +
                                       (unsigned)offsetof(JitCallDesc, result) + 8));
                e->wroteHeap = true;
                off += 1;
                break;
            }
            if (!e->callsOut) return false;
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
            /* Shape 2 says this body built the range itself, so its step is 1
             * by construction; shape 3 adds a start the emitter knows, carried
             * as the entry's sample. Shape 0 stays the general form, for an
             * ObjIter that arrived from anywhere else. */
            if (!pushValue3(e, SLOT_ITER, e->rangeStartKnown ? 3u : 2u, NULL,
                            e->rangeStartKnown ? INT_VAL(e->rangeStartVal)
                                               : NULL_VAL,
                            -1)) {
                return false;
            }
            emit(e, jaiA64LdrX(pushReg(e) - 1, 31,
                               e->descOffset +
                                   (unsigned)offsetof(JitCallDesc, result) + 8));
            e->wroteHeap = true;
            e->pendingRange = false;
            off += 1;
            break;
        }

        case OP_ITER_RANGE: {
            /* `for x in a..b` opened, with neither object built. The two ints
             * this writes are the whole loop, so there is no descriptor, no
             * root fill and no call out -- and, being ordinary int locals, they
             * compete for registers on the same terms as everything else
             * instead of reserving four the way an ObjIter head does.
             *
             * `end` is one past the last value, WRAPPING, which is what makes
             * `a..=INT64_MAX` terminate: the counter meets INT64_MIN there.
             * Same arithmetic as the interpreter's, deliberately -- see
             * OP_ITER_RANGE in chunk.h. */
            bool     inclusive = code[off + 1] != 0;
            unsigned curSlot   = jaiReadU16(code + off + 2);
            unsigned endSlot   = jaiReadU16(code + off + 4);
            if (e->depth < 2) return false;
            if (e->stack[e->depth - 1] != SLOT_INT) return false;
            if (e->stack[e->depth - 2] != SLOT_INT) return false;
            if (!localInRange(e, curSlot) || !localInRange(e, endSlot)) {
                return false;
            }
            if (!adoptLocalKind(e, curSlot, SLOT_INT, 0, NULL)) return false;
            if (!adoptLocalKind(e, endSlot, SLOT_INT, 0, NULL)) return false;
            if (curSlot == 0 || endSlot == 0) e->usesSlot0 = true;

            unsigned rHi, rLo;
            if (!popValue(e, &rHi, NULL)) return false;
            if (!popValue(e, &rLo, NULL)) return false;
            /* Both ends stay read-only: a popped register may be a local's own,
             * borrowed, and writing it would rewrite the local. */
            if (inclusive) {
                emit(e, jaiA64AddXImm(JIT_SCRATCH_A, rHi, 1));
            } else {
                emit(e, jaiA64MovX(JIT_SCRATCH_A, rHi));
            }
            emit(e, jaiA64SubsXReg(31, rHi, rLo));
            /* An empty range ends where it begins, so the first test already
             * fails and the body never runs. */
            emit(e, jaiA64CselX(JIT_SCRATCH_B, rLo, JIT_SCRATCH_A, JAI_A64_LT));
            localOut(e, endSlot, JIT_SCRATCH_B);
            localOut(e, curSlot, rLo);
            off += 6;
            break;
        }

        case OP_FOR_RANGE_BIND: {
            /* One step of that loop: a compare, a branch, a bind and an add,
             * with nothing to load from the heap and nothing to guard. The two
             * slots are written by OP_ITER_RANGE and by this instruction and by
             * nothing else -- the emitter hands out fresh temporaries for them
             * -- so their kind is a fact of the shape rather than a sample, and
             * this arm has no deopt of its own. A bail lands on this
             * instruction with the counter unadvanced. */
            int16_t  jump    = jaiReadI16(code + off + 1);
            unsigned slot    = jaiReadU16(code + off + 3);
            unsigned curSlot = jaiReadU16(code + off + 5);
            unsigned endSlot = jaiReadU16(code + off + 7);
            if (!localInRange(e, slot)) return false;
            if (!localInRange(e, curSlot) || !localInRange(e, endSlot)) {
                return false;
            }
            if (e->localKind[curSlot] != SLOT_INT ||
                e->localKind[endSlot] != SLOT_INT) {
                return false;
            }
            if (!adoptLocalKind(e, slot, SLOT_INT, 0, NULL)) {
                e->whyNot = "loop variable took two kinds";
                return false;
            }
            if (slot == 0) e->usesSlot0 = true;

            unsigned rCur = localIn(e, curSlot, JIT_SCRATCH_A);
            unsigned rEnd = localIn(e, endSlot, JIT_SCRATCH_B);
            emit(e, jaiA64SubsXReg(31, rCur, rEnd));
            /* Nothing of this loop's is on the operand stack, so the exit is
             * reached at exactly the depth this branch leaves from. */
            branchTo(e, (uint32_t)((int32_t)(off + 9) + jump), true,
                     JAI_A64_EQ);
            /* Bind before stepping: in register mode the counter's home IS
             * rCur, so the add would destroy the value about to be bound. */
            localOut(e, slot, rCur);
            {
                unsigned dst = localDest(e, curSlot);
                emit(e, jaiA64AddXImm(dst, rCur, 1));
                localOut(e, curSlot, dst);
            }
            off += 9;
            break;
        }

        case OP_FOR_ITER_BIND: {
            /* A loop this body built the iterator for: the index lives in the
             * iterator, so every iteration loads and stores it, and a deopt
             * needs nothing -- what is on the stack is already current. */
            /* Only the loop at the OSR entry point owns the reserved iterator registers -- a nested FOR_ITER_BIND
             * built its own iterator and is an ordinary one. Refusing it stopped the outer loops of spectral, mandelbrot, matrix_mul and life from compiling at all, while their inner loops compiled fine. */
            if (!e->osr || !e->hasIter || (uint32_t)off != e->osrTop) {
                if (e->depth == 0 || e->stack[e->depth - 1] != SLOT_ITER) {
                    return false;
                }
                int16_t  fjump = jaiReadI16(code + off + 1);
                unsigned fslot = jaiReadU16(code + off + 3);
                if (!localInRange(e, fslot)) return false;
                unsigned rIt = pushReg(e) - 1;

                /* 0 is a range, 1 an iterator the runtime has to step, 2 and 3
                 * ranges this body built (see OP_GET_ITER). Anything else is
                 * not something the inline range form may assume, so it keeps
                 * the stepped path this test has always sent it down. */
                uint32_t iterShape = e->stackShape[e->depth - 1];
                if (iterShape != 0 && iterShape != 2 && iterShape != 3) {
                    /* A list iterator, stepped inline (jaiIterNext's ITER_LIST
                     * case, instruction for instruction) rather than through
                     * jitIterStep. The kind still comes from the element the
                     * list is holding now, but the tag of what the step
                     * actually produces is GUARDED, and every guard runs
                     * BEFORE the index advances.
                     *
                     * Calling out cannot be made sound: jitIterStep advances
                     * the iterator before it returns, so a guard on its result
                     * has nowhere to resume -- this instruction would re-run
                     * and SKIP an element, and under
                     * JAITHON_JIT_DEOPT_STRESS (where branchOnDeopt is
                     * unconditional) it would skip one on every iteration of
                     * every list loop. Reading the payload out of the
                     * descriptor with no tag check at all was worse: a list
                     * sampled as int and later pushed a str bound the string's
                     * POINTER as an integer (probe: 41080394656 where the
                     * interpreter raises TypeError), and a float's IEEE bits
                     * likewise. Inline, nothing has happened when a guard
                     * fires, so the resume point is this instruction and the
                     * interpreter does the raise. */
                    Value sample = e->stackSeen[e->depth - 1];
                    SlotKind ek; unsigned etag; uint32_t esh = 0;
                    ObjClass *ecl = NULL;
                    if (IS_INT(sample))        { ek = SLOT_INT;   etag = VAL_INT; }
                    else if (IS_FLOAT(sample)) { ek = SLOT_FLOAT; etag = VAL_FLOAT; }
                    else if (IS_INSTANCE(sample) && AS_INSTANCE(sample)->klass) {
                        ek = SLOT_INST; etag = VAL_OBJ;
                        ecl = AS_INSTANCE(sample)->klass;
                        esh = ecl->shapeId;
                    } else { e->whyNot = "element kind unknown"; return false; }

                    if (!adoptLocalKindSeen(e, fslot, ek, esh, ecl, sample)) {
                        e->whyNot = "loop variable took two kinds";
                        return false;
                    }

                    /* Only OP_GET_ITER's list arm makes a shape-nonzero
                     * SLOT_ITER, and SLOT_ITER is never adopted into a local,
                     * so this is an ITER_LIST over an ObjList. Checked anyway,
                     * one load: the alternative is reading an ObjString or an
                     * ObjDict through ObjList's offsets if another iterator
                     * shape is ever added above, and a wrong answer is the one
                     * failure mode this tier is not allowed. */
                    emit(e, jaiA64LdrW(JIT_SCRATCH_A, rIt,
                                       (unsigned)offsetof(ObjIter, kind)));
                    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, ITER_LIST));
                    branchOnDeopt(e, JAI_A64_NE);

                    emit(e, jaiA64LdrX(JIT_SCRATCH_C, rIt,
                                       (unsigned)offsetof(ObjIter, source) + 8));

                    /* Mutation first, as jaiIterNext tests it: a list that grew
                     * or shrank under the loop must raise, and the version is
                     * the only thing that says so. */
                    emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_C,
                                       (unsigned)offsetof(ObjList, version)));
                    emit(e, jaiA64LdrW(JIT_SCRATCH_B, rIt,
                                       (unsigned)offsetof(ObjIter, version)));
                    emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
                    branchOnDeopt(e, JAI_A64_NE);

                    /* `limit` is the snapshot count, not the live one -- the
                     * version guard above owns any disagreement between them. */
                    emit(e, jaiA64LdrX(JIT_SCRATCH_A, rIt,
                                       (unsigned)offsetof(ObjIter, index)));
                    emit(e, jaiA64LdrX(JIT_SCRATCH_B, rIt,
                                       (unsigned)offsetof(ObjIter, limit)));
                    emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
                    /* The exhausted arm drops the iterator, so the target is
                     * reached one entry shallower than this branch leaves
                     * from. */
                    branchToDepth(e, (uint32_t)((int32_t)(off + 5) + fjump),
                                  JAI_A64_GE,
                                  (int)stackSignatureAt(e, e->depth - 1));

                    /* Reload items rather than hoisting: a reallocation bumps
                     * the version, which the guard above covers. */
                    emit(e, jaiA64LdrX(JIT_SCRATCH_C, JIT_SCRATCH_C,
                                       (unsigned)offsetof(ObjList, items)));
                    emit(e, jaiA64AddXLsl(JIT_SCRATCH_C, JIT_SCRATCH_C,
                                          JIT_SCRATCH_A, 4));

                    emit(e, jaiA64LdrW(JIT_SCRATCH_B, JIT_SCRATCH_C, 0));
                    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_B, etag));
                    branchOnDeopt(e, JAI_A64_NE);

                    if (ek == SLOT_INST) {
                        /* VAL_OBJ is every heap object, so the object type is
                         * checked before `klass` is read -- otherwise a list
                         * that gained a string reads `klass` one word past an
                         * ObjString's header. JIT_SCRATCH_A still holds the
                         * index and must survive to the store below. */
                        emit(e, jaiA64LdrX(JIT_SCRATCH_B, JIT_SCRATCH_C, 8));
                        emit(e, jaiA64LdrW(JIT_SCRATCH_D, JIT_SCRATCH_B,
                                           (unsigned)offsetof(Obj, type)));
                        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_D, OBJ_INSTANCE));
                        branchOnDeopt(e, JAI_A64_NE);
                        emit(e, jaiA64LdrX(JIT_SCRATCH_B, JIT_SCRATCH_B,
                                           (unsigned)offsetof(ObjInstance, klass)));
                        emit(e, jaiA64LdrW(JIT_SCRATCH_B, JIT_SCRATCH_B,
                                           (unsigned)offsetof(ObjClass, shapeId)));
                        emitConst64(e, JIT_SCRATCH_D, (int64_t)esh);
                        emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_B, JIT_SCRATCH_D));
                        branchOnDeopt(e, JAI_A64_NE);
                    }

                    /* Past the last guard: advance, then bind. The advance goes
                     * first because localOut may use JIT_SCRATCH_C/D for the
                     * tag and the index has to be stored out of a register the
                     * write cannot touch. */
                    emit(e, jaiA64AddXImm(JIT_SCRATCH_B, JIT_SCRATCH_A, 1));
                    emit(e, jaiA64StrX(JIT_SCRATCH_B, rIt,
                                       (unsigned)offsetof(ObjIter, index)));
                    emit(e, jaiA64LdrX(JIT_SCRATCH_A, JIT_SCRATCH_C, 8));
                    localOut(e, fslot, JIT_SCRATCH_A);
                    /* The index store is a heap write, as the call-out this
                     * replaced was: a bail after it would re-run the loop from
                     * the top with the iterator already advanced. */
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
                /* A range yields `start + index * step`, not the index itself (see jaiIterNext's ITER_RANGE case) --
                 * the index is always zero-based, so using it directly is only right for `0..n` in unit steps. `for j in i + 1..n` once counted from zero instead of i+1, a plausible wrong answer (nested loops summed the wrong pairs), not a crash. The dead-after-compare limit register carries the index across to the increment. */
                emit(e, jaiA64MovX(JIT_SCRATCH_B, JIT_SCRATCH_A));
                /* Both halves of that map are loop-invariant, and for a range
                 * this body built (shape 2) the step is 1 by construction --
                 * jitMakeRangeIter has no step argument. With the start a
                 * literal too (shape 3) nothing about the ObjRange has to be
                 * read at all, which is five loads and a multiply off the back
                 * of every nested `for k in 0..n`: matrix_mul spends fourteen
                 * of its innermost forty-nine instructions on this counter. */
                /* Shape 3's constant travels in the entry's sample, and an
                 * entry can reach here having been through a local, where the
                 * sample need not have come with it. No sample, no shortcut:
                 * the general form below is right for any range. */
                if (iterShape == 3 && !IS_INT(e->stackSeen[e->depth - 1])) {
                    iterShape = 2;
                }
                if (iterShape == 3) {
                    int64_t k = AS_INT(e->stackSeen[e->depth - 1]);
                    if (k > 0 && k <= 4095) {
                        emit(e, jaiA64AddXImm(JIT_SCRATCH_A, JIT_SCRATCH_A,
                                              (unsigned)k));
                    } else if (k < 0 && k >= -4095) {
                        emit(e, jaiA64SubXImm(JIT_SCRATCH_A, JIT_SCRATCH_A,
                                              (unsigned)(-k)));
                    } else if (k != 0) {
                        emitConst64(e, JIT_SCRATCH_D, k);
                        emit(e, jaiA64AddX(JIT_SCRATCH_A, JIT_SCRATCH_D,
                                           JIT_SCRATCH_A));
                    }
                } else {
                    emit(e, jaiA64LdrX(JIT_SCRATCH_C, rIt,
                                       (unsigned)offsetof(ObjIter, source) + 8));
                    if (iterShape != 2) {
                        emit(e, jaiA64LdrX(JIT_SCRATCH_D, JIT_SCRATCH_C,
                                           (unsigned)offsetof(ObjRange, step)));
                        emit(e, jaiA64MulX(JIT_SCRATCH_A, JIT_SCRATCH_A,
                                           JIT_SCRATCH_D));
                    }
                    emit(e, jaiA64LdrX(JIT_SCRATCH_D, JIT_SCRATCH_C,
                                       (unsigned)offsetof(ObjRange, start)));
                    emit(e, jaiA64AddX(JIT_SCRATCH_A, JIT_SCRATCH_D,
                                       JIT_SCRATCH_A));
                }
                localOut(e, fslot, JIT_SCRATCH_A);
                emit(e, jaiA64AddXImm(JIT_SCRATCH_B, JIT_SCRATCH_B, 1));
                emit(e, jaiA64StrX(JIT_SCRATCH_B, rIt,
                                   (unsigned)offsetof(ObjIter, index)));
                off += 5;
                break;
            }
            /* Only as the head of the loop being compiled, and only for a range from zero in unit steps -- that's
             * what makes the yielded value the index itself. Everything else about the iterator is a runtime fact, checked at entry. */
            if (!e->osr || !e->hasIter) return false;
            if ((uint32_t)off != e->osrTop) return false;
            int16_t  jump = jaiReadI16(code + off + 1);
            unsigned slot = jaiReadU16(code + off + 3);
            if (!localInRange(e, slot)) return false;

            if (e->iterKind == 2) {
                /* A list at the loop head: reserved registers mean what they do for a range, except JIT_START_REG
                 * holds the ObjList instead of a first value. Without this, a top-level `for x in xs` was never even attempted -- the gate refused anything not a range before compileOsr ran, so not even a decline was recorded. */
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
                    /* Same hazard as above: VAL_OBJ covers every heap object, so the type is checked before `klass` is read. */
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
            /* Value is start + index, not the index (`for j in i + 1..n` is nbody advance's inner loop): both
             * registers were biased by the start in the prologue, so the value IS the index register and the add that used to be here is gone. */
            localOut(e, slot, JIT_IDX_REG);
            emit(e, jaiA64AddXImm(JIT_IDX_REG, JIT_IDX_REG, 1));
            off += 5;
            break;
        }

        case OP_FOR_ITER_PAIR: {
            /* `for (a, b) in xs` over a list of 2-tuples, stepped inline.
             *
             * Deliberately not a call-out to a helper that steps the iterator.
             * OP_FOR_ITER_BIND's list arm used to be one, and it is inline for
             * the same reason this is: such a helper advances the iterator
             * before returning, so a component-kind guard after it would deopt
             * to an instruction that has already happened -- and under
             * JAITHON_JIT_DEOPT_STRESS every guard is turned into an
             * unconditional branch, which would then skip an element on every
             * pass. Every guard here is placed BEFORE anything is written, so
             * resuming at this very instruction is exact whether the guard
             * failed for a real reason or because the stress flag forced it.
             *
             * The reachable iterator is always a list one -- shape != 0 comes
             * only from OP_GET_ITER's jitMakeIter, over a SLOT_LIST -- but the
             * kind is guarded anyway rather than assumed, because that fact
             * lives two arms away. */
            if (e->depth == 0 || e->stack[e->depth - 1] != SLOT_ITER) {
                return false;
            }
            if (e->stackShape[e->depth - 1] == 0) {
                /* A range head yields ints, which never destructure. */
                e->whyNot = "destructuring what a range yields";
                return false;
            }
            int16_t  pjump = jaiReadI16(code + off + 1);
            unsigned pslotA = jaiReadU16(code + off + 3);
            unsigned pslotB = jaiReadU16(code + off + 5);
            if (!localInRange(e, pslotA) || !localInRange(e, pslotB)) {
                return false;
            }
            if (pslotA == pslotB) {
                /* `for (x, x) in …`: legal, and the second write wins. Not
                 * worth a special case; the interpreter keeps it. */
                e->whyNot = "a pair loop binding one slot twice";
                return false;
            }

            /* Component kinds come from the element the list was holding when
             * the iterator was built (OP_GET_ITER carries it forward), and the
             * guards below are what make that a specialisation rather than an
             * assumption. */
            Value psample = e->stackSeen[e->depth - 1];
            if (!IS_TUPLE(psample) || AS_TUPLE(psample)->count != 2) {
                e->whyNot = "pair element is not a 2-tuple";
                return false;
            }
            SlotKind pk[2];
            unsigned ptag[2];
            Value pseen[2] = { AS_TUPLE(psample)->items[0],
                               AS_TUPLE(psample)->items[1] };
            for (unsigned i = 0; i < 2; i++) {
                Value v = pseen[i];
                if (IS_INT(v))        { pk[i] = SLOT_INT;   ptag[i] = VAL_INT; }
                else if (IS_FLOAT(v)) { pk[i] = SLOT_FLOAT; ptag[i] = VAL_FLOAT; }
                else if (IS_BOOL(v))  { pk[i] = SLOT_BOOL;  ptag[i] = VAL_BOOL; }
                else if (IS_OBJ(v) && AS_OBJ(v) != NULL) {
                    /* Held raw, like any other SLOT_OBJ: the tag guard is the
                     * whole of what this promises, and an arm that wants to
                     * know WHICH object type checks that itself. */
                    pk[i] = SLOT_OBJ; ptag[i] = VAL_OBJ;
                } else {
                    e->whyNot = "pair component kind unknown";
                    return false;
                }
            }
            if (!adoptLocalKindSeen(e, pslotA, pk[0], 0, NULL, pseen[0]) ||
                !adoptLocalKindSeen(e, pslotB, pk[1], 0, NULL, pseen[1])) {
                e->whyNot = "loop variable took two kinds";
                return false;
            }

            unsigned rIter = pushReg(e) - 1;

            /* Really a list iterator, and its source really a list. */
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, rIter,
                               (unsigned)offsetof(ObjIter, kind)));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, ITER_LIST));
            branchOnDeopt(e, JAI_A64_NE);
            emit(e, jaiA64LdrX(JIT_SCRATCH_B, rIter,
                               (unsigned)offsetof(ObjIter, source) + 8));
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_B,
                               (unsigned)offsetof(Obj, type)));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_LIST));
            branchOnDeopt(e, JAI_A64_NE);

            /* A list that grew or shrank under the loop must raise, and the
             * version is the only thing that says so. Nothing has happened
             * yet, so the interpreter raises it from this instruction. */
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_B,
                               (unsigned)offsetof(ObjList, version)));
            emit(e, jaiA64LdrW(JIT_SCRATCH_C, rIter,
                               (unsigned)offsetof(ObjIter, version)));
            emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_C));
            branchOnDeopt(e, JAI_A64_NE);

            emit(e, jaiA64LdrX(JIT_SCRATCH_C, rIter,
                               (unsigned)offsetof(ObjIter, index)));
            emit(e, jaiA64LdrX(JIT_SCRATCH_D, rIter,
                               (unsigned)offsetof(ObjIter, limit)));
            emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_C, JIT_SCRATCH_D));
            /* The exhausted arm drops the iterator, so the target is reached
             * one entry shallower than this branch leaves from. */
            branchToDepth(e, (uint32_t)((int32_t)(off + 7) + pjump),
                          JAI_A64_GE,
                          (int)stackSignatureAt(e, e->depth - 1));

            /* items is reloaded rather than hoisted: a reallocation bumps the
             * version, which the guard above covers, and one ldr removes the
             * question. */
            emit(e, jaiA64LdrX(JIT_SCRATCH_B, JIT_SCRATCH_B,
                               (unsigned)offsetof(ObjList, items)));
            emit(e, jaiA64AddXLsl(JIT_SCRATCH_B, JIT_SCRATCH_B,
                                  JIT_SCRATCH_C, 4));

            /* The element is a 2-tuple whose components have the sampled
             * kinds. Object type is checked before `count` is read: VAL_OBJ
             * covers every heap object, so a string beside the sampled tuple
             * would otherwise have `count` read out of an ObjString. */
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_B, 0));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, VAL_OBJ));
            branchOnDeopt(e, JAI_A64_NE);
            emit(e, jaiA64LdrX(JIT_SCRATCH_B, JIT_SCRATCH_B, 8));
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_B,
                               (unsigned)offsetof(Obj, type)));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_TUPLE));
            branchOnDeopt(e, JAI_A64_NE);
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_B,
                               (unsigned)offsetof(ObjTuple, count)));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, 2));
            branchOnDeopt(e, JAI_A64_NE);
            for (unsigned i = 0; i < 2; i++) {
                unsigned at = (unsigned)offsetof(ObjTuple, items) +
                              i * (unsigned)sizeof(Value);
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_B, at));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, ptag[i]));
                branchOnDeopt(e, JAI_A64_NE);
            }

            /* Past the last guard: from here nothing may fail, and every write
             * below is what the interpreter would have left behind. The index
             * goes first because localOut spends JIT_SCRATCH_C and
             * JIT_SCRATCH_D on the tag of a slot that lives in the frame; only
             * the element pointer in JIT_SCRATCH_B survives it. */
            emit(e, jaiA64AddXImm(JIT_SCRATCH_C, JIT_SCRATCH_C, 1));
            emit(e, jaiA64StrX(JIT_SCRATCH_C, rIter,
                               (unsigned)offsetof(ObjIter, index)));
            /* A bool is one byte (see OP_GET_INDEX) -- the rest of its payload
             * word is stale, and a SLOT_BOOL register must hold 0 or 1. */
            for (unsigned i = 0; i < 2; i++) {
                unsigned at = (unsigned)offsetof(ObjTuple, items) +
                              i * (unsigned)sizeof(Value) + 8u;
                if (pk[i] == SLOT_BOOL) {
                    emit(e, jaiA64LdrByte(JIT_SCRATCH_A, JIT_SCRATCH_B, at));
                } else {
                    emit(e, jaiA64LdrX(JIT_SCRATCH_A, JIT_SCRATCH_B, at));
                }
                localOut(e, i == 0 ? pslotA : pslotB, JIT_SCRATCH_A);
            }
            e->wroteHeap = true;
            off += 7;
            break;
        }

        case OP_GET_INDEX: {
            /* `s[i]` on a string: every guard is a load+compare, and the result is a table lookup, not an
             * allocation (the 128 one-byte strings are made once and shared). Without this the whole loop around a character scan declines -- why `str_search` ran interpreted end to end, and every lexer scans one byte at a time. */
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

                /* jaiNormalizeIndex, then one unsigned compare for both ends.
                 * `length` came from an `ldr w`, so it is already the whole
                 * register. */
                emitBoundsNormalise(e, rIdx, JIT_SCRATCH_A, JIT_SCRATCH_B,
                                    false);

                emit(e, jaiA64LdrX(JIT_SCRATCH_C, rStr,
                                   (unsigned)offsetof(ObjString, chars)));
                emit(e, jaiA64AddX(JIT_SCRATCH_C, JIT_SCRATCH_C, JIT_SCRATCH_B));
                emit(e, jaiA64LdrByte(JIT_SCRATCH_A, JIT_SCRATCH_C, 0));
                /* 128 is an imm12, so the compare needs no register: a
                 * materialised constant on a body this hot is not free the way
                 * a register copy is. */
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, 128));
                branchOnDeopt(e, JAI_A64_HS);

                /* The shared one-byte string. jaiVMInit fills all 128 slots, so
                 * this is a load and not a load plus a null test -- see
                 * jaiAsciiCharsFill. The scaled add folds the shift in. */
                emitConst64(e, JIT_SCRATCH_C,
                            (int64_t)(uintptr_t)jaiAsciiCharTable());
                emit(e, jaiA64AddXLsl(JIT_SCRATCH_C, JIT_SCRATCH_C,
                                      JIT_SCRATCH_A, 3));
                emit(e, jaiA64LdrX(JIT_SCRATCH_C, JIT_SCRATCH_C, 0));

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
                /* What the table holds is interned by construction -- see
                 * jaiStringChar -- so a consumer that would guard this for
                 * being a string, and for being interned, need not. */
                e->stackAscii[e->depth - 1] = true;
                emit(e, jaiA64MovX(pushReg(e) - 1, JIT_SCRATCH_C));
                off += 1;
                break;
            }
            /* Index normalised as jaiNormalizeIndex does it, one unsigned compare covering both ends. Out of
             * range, or an element not the kind seen at compile time, goes back to the interpreter -- reading an element has no effect, so resuming at this instruction is always sound. */
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
                /* A list of strings, held opaquely: the interned compare and `s[i]` each check OBJ_STRING for
                 * themselves, same contract as any SLOT_OBJ local (sample specialises, guard confirms). `str_search` builds text out of `chunks[seed %% 8]` and declined that whole loop forty times over before this. */
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

            /* One `ldp` for both header fields: `items` at +16, `count`/`capacity` the adjacent int32s at +24, so
             * the pair's second half is `count | capacity << 32` and the bounds test reads it with uxtw -- one instruction per element read (life does nine per cell). */
            noteSlotIndexed(e, e->stackLocal[e->depth - 2]);
            unsigned gItems = JIT_SCRATCH_C, gCount = JIT_SCRATCH_A;
            int gh = hoistFor(e, e->stackLocal[e->depth - 2]);
            if (gh >= 0) {
                gItems = e->hoist[gh].itemsReg;
                gCount = e->hoist[gh].countReg;
            } else {
                emitListHeader(e, rList, gItems, gCount);
            }
            emitBoundsNormalise(e, rIdx, gCount, JIT_SCRATCH_B, true);

            emit(e, jaiA64AddXLsl(JIT_SCRATCH_C, gItems,
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
            /* Bool payload is one byte (`BOOL_VAL` compiles to `strb`), so the other seven bytes are stale --
             * an 8-byte load would hand a SLOT_BOOL register (required to hold exactly 0 or 1, since every consumer does `cbnz` on the whole word) garbage. */
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
            /* Write half of OP_GET_INDEX, normalised the same way; every guard runs before the store, so a deopt
             * here still resumes at an instruction that hasn't happened yet. Sixteen refusals across the benchmarks came from its absence -- `queens` couldn't compile the function that does the work. */
            if (e->depth < 3) return false;
            if (e->stack[e->depth - 3] == SLOT_OBJ) {
                /* `d[k] = v`: a dict is as ordinary a container here as a list -- without this, dict_ops' loop just
                 * moved its decline from `get` to this store (a loop that declines anywhere runs interpreted end to end). Object type guarded before anything is consumed, so a miss resumes with container/key/value all still on the interpreter's stack. */
                unsigned sidx = e->depth - 3;
                if (!IS_DICT(e->stackSeen[sidx])) {
                    e->whyNot = "an index store into an object that is not a dict";
                    return false;
                }
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, pushReg(e) - 3,
                                   (unsigned)offsetof(Obj, type)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_DICT));
                branchOnDeopt(e, JAI_A64_NE);
                if (!emitDescriptor(e, NULL_VAL, sidx, 3,
                                    (void *)&jitSetIndexDict)) {
                    return false;
                }
                for (unsigned i = 0; i < 3; i++) {
                    unsigned r;
                    if (!popValue(e, &r, NULL)) return false;
                }
                e->wroteHeap = true;
                off += 1;
                break;
            }
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

            noteSlotIndexed(e, e->stackLocal[e->depth - 3]);
            unsigned sItems = JIT_SCRATCH_C, sCount = JIT_SCRATCH_A;
            int sh = hoistFor(e, e->stackLocal[e->depth - 3]);
            if (sh >= 0) {
                sItems = e->hoist[sh].itemsReg;
                sCount = e->hoist[sh].countReg;
            } else {
                emitListHeader(e, rList, sItems, sCount);
            }
            emitBoundsNormalise(e, rIdx, sCount, JIT_SCRATCH_B, true);

            emit(e, jaiA64AddXLsl(JIT_SCRATCH_C, sItems,
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
            Value cseen = e->stackSeen[cidx];
            /* A string slices as readily as a list -- same runtime call, same
             * "the guard pins the type so the result kind follows" argument --
             * and `s[a:b]` is what every hand-written scanner cuts tokens with.
             * Held as SLOT_OBJ, since that is what a string is here. */
            unsigned cType;
            SlotKind sliceKind;
            if (e->stack[cidx] == SLOT_LIST) {
                cType = OBJ_LIST; sliceKind = SLOT_LIST;
            } else if (e->stack[cidx] == SLOT_OBJ && IS_STRING(cseen)) {
                cType = OBJ_STRING; sliceKind = SLOT_OBJ;
            } else {
                e->whyNot = "slicing a container this tier does not model";
                return false;
            }

            /* Guard the container, not the result: with its object type pinned
             * the arm jaiSliceGet takes is settled, so the result's kind
             * follows. Before the descriptor and before any pop, so a miss
             * resumes here with everything still on the interpreter's stack. */
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, pushReg(e) - nargs,
                               (unsigned)offsetof(Obj, type)));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, cType));
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
             * ints is a list of ints, a slice of a string is a string, and
             * every element read re-checks its own tag, so this is a hint and
             * not an assumption. */
            if (!pushValue3(e, sliceKind, 0, NULL, cseen, -1)) return false;
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
            /* Two resolution paths with different obligations. BY VALUE (globalIsSelf/globalClass/globalFunction/
             * globalNative): resolved and baked at compile time, nothing rechecked at run time, so ObjModule::version must retire the whole form whenever such a binding could change (jaiModuleSet's jaiValueIsInertGlobal decides that) -- teaching any of the four a new value kind, or constant-folding a module int here, means updating jaiValueIsInertGlobal too. BY ADDRESS (the value case below): bakes the JaiEntry*, not the value, and re-loads it behind a tag guard (+ Obj.type/class-shape guards where needed) on EVERY access -- depends on JaiTable::keyVersion, not on ObjModule::version. */
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
                    /* A global holding a plain value: storage is a JaiEntry whose address is fixed once it exists, so the
                     * load is one `ldr` behind two guards (table hasn't moved, value still has the compiled-for kind). Refusing this declined every loop reading a module-level variable -- most benchmarks, once they moved to module scope. */
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
                            /* VAL_OBJ is every heap object; the tag alone doesn't confirm the specific type, so it's checked before the class pointer is read. */
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
                /* SLOT_OBJ deliberately excluded: it covers a closure too, and storing a closure into a global
                 * rebinds a callee some compiled form may have already baked BY VALUE -- ObjModule::version only retires that form at its next ENTRY, not mid-body, so this would be a silently wrong answer, not a decline. Every remaining kind is inert per jaiValueIsInertGlobal, which is what makes the bump rule below sound. */
                e->whyNot = "a global store of a kind that has no Value form";
                return false;
            }
            emitGlobalsGuard(e);
            {
                unsigned src = pushReg(e) - 1;
                emitConst64(e, JIT_SCRATCH_D, (int64_t)(uintptr_t)gslot);
                /* Version only needs to move when a class/closure/native leaves or arrives (jaiValueIsInertGlobal is
                 * the authority). The incoming value is already provably inert (SLOT_OBJ refused above), so only what's overwritten matters -- this checks a STRICT SUBSET of the inert types (non-object, ObjInstance, ObjList); anything else conservatively bumps. Widening jaiValueIsInertGlobal needs no change here; narrowing it (removing OBJ_INSTANCE/OBJ_LIST) does. */
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
                /* No write barrier: mark-sweep traces module globals as a root every collection, so a raw store is
                 * enough. Nothing between the tag and payload writes can allocate, so no collection can ever see them disagree. */
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
                                     (uint32_t)off, -1)) {
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
                /* `float(i)`/`int(x)` are one instruction each, so they're emitted rather than called -- what
                 * `spectral`/`matrix_mul` have in their inner loops, and the whole body was declined for want of them. Every other builtin still declines: a call needs a result kind known without running anything, and only these two have one. */
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
                /* Closure held in a local (`apply_n(f, ..)` doing `acc = f(acc)`, the shape most library code takes).
                 * Guard is on the FUNCTION, not the closure: `closure_calls` builds a fresh closure over a different `step` every outer iteration, so guarding the closure itself would deoptimise every time -- all share one ObjFunction (monomorphic) and differ only by upvalue. The callee Value still comes from the register, which is what keeps the upvalues right. */
                unsigned cidx = e->depth - argc - 1;
                Value cv = e->stackSeen[cidx];
                if (!IS_CLOSURE(cv)) {
                    e->whyNot = "an indirect callee that is not a closure";
                    return false;
                }
                ObjFunction *cfn = AS_CLOSURE(cv)->fn;
                unsigned rCallee0 = valueBankBase(e) +
                                    (cidx - (e->depth - e->valueDepth));

                /* The guard comes first now, because what follows it is a
                 * choice between two ways of making the call and both need it:
                 * once this register is known to name `cfn`, the body behind
                 * it is known too, and that is the whole licence to inline. */
                emit(e, jaiA64LdrX(JIT_SCRATCH_A, rCallee0,
                                   (unsigned)offsetof(ObjClosure, fn)));
                emitConst64(e, JIT_SCRATCH_B, (int64_t)(uintptr_t)cfn);
                emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
                branchOnDeopt(e, JAI_A64_NE);

                /* Cheapest first, as the direct call does it: a body small enough to stand where the call is costs
                 * neither frame, argument shuffle, nor root fill. An indirect site needs NONE of the checks below to inline -- no argument kind has to match a specialisation, and the callee need not have compiled at all, since arguments stay in the caller's own registers with the caller's own kinds. */
                if (inlineGlobalCall(e, fn, AS_CLOSURE(cv), argc,
                                     (uint32_t)off, (int)rCallee0)) {
                    off += 2;
                    break;
                }
                if (e->failed) return false;

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
                unsigned rCallee = rCallee0;   /* guarded above */

                /* Straight to the callee's compiled entry, not through jaiCallValue/an interpreter frame -- same
                 * convention a self-call and jaiJitEnterFunc use. Callee must live in this module since the caller's module-version guard stands in for the callee's own entry check. The baked jitFunc address can't go stale: the arena is never freed and jitFunc is written once; only the ObjFunction identity needs guarding (done above). */
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

                unsigned firstArg = valueBankBase(e) +
                                    (cidx + 1u - (e->depth - e->valueDepth));
                for (unsigned i = 0; i < argc; i++) {
                    emit(e, jaiA64MovX(i, firstArg + i));
                }
                if (wantsClosure) emit(e, jaiA64MovX(argc, rCallee));
                emitConst64(e, JIT_SCRATCH_D,
                            (int64_t)(uintptr_t)cfn->jitFunc);
                noteScratchClobber(e);
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
            /* Recorded, not rejected, here: the measuring pass always runs with slot 0 available, so testing the
             * base in THIS pass silently aborted every recursive function's compile (fib_recursive: 8.8ms back to 83ms). The decision belongs after the base is actually chosen. */
            e->hasSelfCall = true;
            if (e->depth < argc + 1) return false;
            if (e->stack[e->depth - argc - 1] != SLOT_SELF) return false;

            /* A self-call branches past the entry guard, so nothing else checks what it hands over -- passing a
             * maybe-instance into a slot typed as a plain instance would let a later field read dereference zero. Recorded per-slot so the retry seeds only that slot as nullable, costing nothing for parameters that are never null. */
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
            unsigned first = valueBankBase(e) + e->valueDepth - argc;
            for (unsigned i = 0; i < argc; i++) {
                emit(e, jaiA64MovX(i, first + i));
            }

            /* To instruction 0, the prologue -- NOT to the first instruction
             * of the body. A recursive call that skipped the prologue would
             * not save x19 upward, so the callee would overwrite the caller's
             * locals and the recursion would never terminate. */
            if (!e->sawReturn) e->assumedIntReturn = true;

            noteScratchClobber(e);
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
            /* x1 is the callee's verdict, each needing a different response here (this used to be one `bail`,
             * which is why queens' `place` never compiled -- a bail re-runs the whole caller, unsound above `cols[row] = col`):
             *   0  value is in x0. Costs one compare, one not-taken branch.
             *   2  callee raised; the interpreter owns it, leave.
             *   1  callee bailed (only possible having written nothing) -- NOT a bail here, but a deopt at
             *      this instruction, re-executing just the call with the caller's own earlier writes intact.
             *   4  callee deoptimised part-way and may have written -- can't be re-executed or recorded over
             *      (gDeopt is one global), so it's FINISHED in the interpreter from its own record, value handed back here. One record suffices at any recursion depth since it's consumed at the innermost frame that sees it.
             * A kind the body wasn't compiled for lands in the descriptor's result slot with its real tag,
             * written out by the existing `lastFromDesc` deopt. */
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
            /* Same OSR resume-offset hazard as OP_RETURN_NULL -- see there. */
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
            /* Same OSR resume-offset hazard as OP_RETURN_NULL -- see there. */
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
    settleAll(e);
    /* An inlined body's offsets are the callee's; matching them against the
     * caller's fixups compares two different numbering schemes. There is
     * nothing to check either way -- it has no branches. */
    if (e->inlining) return !e->failed;
    /* Nothing may branch to an offset this walk carried a float into (see fpCarry) -- declines, and the caller retries with the FP bank off. */
    for (unsigned i = 0; i < e->fpCarryCount; i++) {
        for (unsigned f = 0; f < e->fixupCount; f++) {
            if (e->fixups[f].targetOffset != e->fpCarry[i]) continue;
            e->whyNot = "a branch lands inside a float expression";
            return false;
        }
    }
    /* Mirror image for homeEarly (see fpBindLookahead): a back edge to such a bind is only visible here, after the whole walk. */
    for (unsigned i = 0; i < e->homeEarlyCount; i++) {
        for (unsigned f = 0; f < e->fixupCount; f++) {
            if (e->fixups[f].targetOffset != e->homeEarly[i]) continue;
            e->whyNot = "a branch lands on a bind whose local was written early";
            return false;
        }
    }
    /* Same for a deferred X entry: forward branches were settled during the walk, this catches a
     * backward one (a loop head sitting between an OP_INT and the operator consuming it). Nothing in the suite reaches it -- the point is that it costs a decline, not a register nothing wrote. */
    for (unsigned i = 0; i < e->deferCarryCount; i++) {
        for (unsigned f = 0; f < e->fixupCount; f++) {
            if (e->fixups[f].targetOffset != e->deferCarry[i]) continue;
            e->whyNot = "a branch lands where a value was deferred";
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
            /* A slot an earlier attempt found sometimes-null holds the pointer or zero; the class still comes
         * from this call's argument -- a maybe-instance is a correct supertype, so widening can't make a field offset wrong. */
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
            /* A null argument (a defaulted parameter, mostly): nothing can be done with it, but a body that
             * never reads it compiles, and the entry guard needs no check since an opaque slot is never read. Refusing outright stopped every stdlib function with a defaulted parameter -- most of them. */
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

/* Filled by loopDepthTable, which lives with the OSR entry below. */
static const uint8_t *loopDepthFor(const Chunk *c, unsigned *count);

/* Greedy: most instructions-saved first, across BOTH register banks at once (an int slot read 4x in
 * a loop outranks a float slot read 2x, different pools, but comparing them in one order keeps that meaningful when only one pool runs out -- see noteSlotCost). Three exclusions: a zero-saving slot is skipped outright, not just ranked last (else a tie-break could pay a prologue write for nothing); a dynamic slot keeps its frame home, since only the frame has anywhere to put a run-time tag; a wrong guess costs an `fmov`, never a wrong answer, since only fmov/ldr/str ever touch a home. */
static void planSlotRegisters(Emit *e, const Emit *m, unsigned availX) {
    unsigned availFp = JIT_FP_MAX_SAVED;
    unsigned top = e->base + e->locals;
    if (top > JIT_MAX_SLOTS + 1u) top = JIT_MAX_SLOTS + 1u;
    while (availX > 0 || availFp > 0) {
        unsigned bestSlot = 0, bestGain = 0;
        bool bestFp = false;
        for (unsigned slot = e->base; slot < top; slot++) {
            if (e->slotXReg[slot] != 0 || e->slotFpReg[slot] != 0) continue;
            if (e->dynamicLocal[slot]) continue;
            if (availFp > 0 && m->slotSaveFp[slot] > bestGain) {
                bestGain = m->slotSaveFp[slot];
                bestSlot = slot;
                bestFp   = true;
            }
            if (availX > 0 && m->slotSaveX[slot] > bestGain) {
                bestGain = m->slotSaveX[slot];
                bestSlot = slot;
                bestFp   = false;
            }
        }
        if (bestGain == 0) break;
        if (bestFp) {
            e->slotFpReg[bestSlot] =
                (uint8_t)(JIT_FP_FIRST_SAVED + e->fpLocals++);
            availFp--;
        } else {
            e->slotXReg[bestSlot] = (uint8_t)(JIT_FIRST_SAVED + e->xLocals++);
            availX--;
        }
    }
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

    /* The prologue can't be emitted first: its save set depends on how deep the operand stack gets,
     * which only the body knows. So the body goes into the buffer at a fixed offset and the prologue is written in front of it afterwards, with every instruction index shifted by the same amount. */
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
    /* So the per-slot charges the accessors record are weighted by how deeply
     * nested the site is. Without it every site in the function counts the
     * same and `n`, read once to set the loop up, outranks nothing. */
    body.loopDepth = loopDepthFor(&fn->chunk, &body.loopDepthCount);
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
        /* A tag only has somewhere to live in the frame. One dynamic slot used
         * to send every local to the frame with it; now it sends only itself,
         * because the plan below is per slot. */
        if (e.dynamicLocal[i]) { saved = JIT_MAX_SAVED + 1; break; }
    }
    if (saved > JIT_MAX_SAVED) {
        /* Too many to keep in registers, so the operand stack takes the
         * registers first -- that is expression depth, not the number of
         * variables a function happens to declare -- and whatever is left over
         * goes to the slots that earn it. What does not earn one lives in the
         * frame. */
        e.spilled = true;
        saved = extra + body.maxValue;
        if (saved > JIT_MAX_SAVED) {
            e.whyNot = "the operand stack alone exceeds the registers";
        } else {
            planSlotRegisters(&e, &body, JIT_MAX_SAVED - saved);
            saved += e.xLocals;
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
    if (e.fpLocals > 0) {
        /* v8..v15 are callee-saved in their low 64 bits, which is exactly a
         * double, so str d / ldr d is the whole protocol -- but this tier is
         * itself a callee, so the ones it takes have to be put back. */
        e.fpSaveOffset = (frame + 7u) & ~7u;
        frame = e.fpSaveOffset + 8u * e.fpLocals;
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
    emitFpSaveRestore(&e, true);
    /* Real arguments land in local registers in order; the closure (if any) lives just past the locals,
     * where closureReg expects it. Placing it by argument index instead once put it three registers low, and the first upvalue read dereferenced whatever was there. */
    unsigned realArgs = e.usesUpvalues ? argCount - 1u : argCount;
    for (unsigned i = 0; i < realArgs; i++) {
        unsigned slot = e.base + i;
        if (!e.spilled) {
            emit(&e, jaiA64MovX(JIT_FIRST_SAVED + i, i));
        } else if (e.slotXReg[slot] != 0) {
            /* Arguments arrive in x0..x3 and homes start at x19, so no
             * destination here can be a later argument's source. */
            emit(&e, jaiA64MovX(e.slotXReg[slot], i));
        } else if (e.slotFpReg[slot] != 0) {
            emit(&e, jaiA64FmovDX(e.slotFpReg[slot], i));
        } else {
            /* Spilled local is a whole Value: tag first, payload 8 bytes on. Writing the payload at the tag's
             * offset instead leaves the payload word untouched -- the first read then gets whatever garbage the frame held, often a small int that reads as a pointer, crashing somewhere else entirely. */
            /* From the measuring pass: `e`'s own kinds are seeded after this
             * point, so reading them here would take whatever the struct was
             * zeroed to. */
            if (localTagInFrame(&e, slot)) {
                emit(&e, jaiA64MovzX(JIT_SCRATCH_D,
                                     localTagFor(&body, slot), 0));
                emit(&e, jaiA64StrW(JIT_SCRATCH_D, 31, localFrameOff(&e, slot)));
            }
            emit(&e, jaiA64StrX(i, 31, localFrameOff(&e, slot) + 8));
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
        unsigned slot = e.base + i;
        if (!e.spilled) {
            emit(&e, jaiA64MovzX(JIT_FIRST_SAVED + i, 0, 0));
        } else if (e.slotXReg[slot] != 0) {
            emit(&e, jaiA64MovzX(e.slotXReg[slot], 0, 0));
        } else if (e.slotFpReg[slot] != 0) {
            emit(&e, jaiA64MovzX(JIT_SCRATCH_C, 0, 0));
            emit(&e, jaiA64FmovDX(e.slotFpReg[slot], JIT_SCRATCH_C));
        } else {
            emit(&e, jaiA64MovzX(JIT_SCRATCH_C, 0, 0));
            if (localTagInFrame(&e, slot)) {
                emit(&e, jaiA64MovzX(JIT_SCRATCH_D, VAL_NULL, 0));
                emit(&e, jaiA64StrW(JIT_SCRATCH_D, 31, localFrameOff(&e, slot)));
            }
            emit(&e, jaiA64StrX(JIT_SCRATCH_C, 31, localFrameOff(&e, slot) + 8));
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
            /* Only an opaque slot is null; everything else non-scalar is an object. Listing object kinds
             * explicitly instead once wrote a SLOT_OBJ local (string/dict/closure) out as null on every deopt -- invisible until a body holding one could compile and then actually deopt. */
            unsigned tag = kind == SLOT_INT    ? VAL_INT
                         : kind == SLOT_FLOAT  ? VAL_FLOAT
                         : kind == SLOT_BOOL   ? VAL_BOOL
                         : kind == SLOT_OPAQUE ? VAL_NULL
                         : kind == SLOT_NULL   ? VAL_NULL
                                               : VAL_OBJ;
            unsigned at = (unsigned)offsetof(JitDeoptRecord, locals) +
                          i * (unsigned)sizeof(Value);

            /* Nothing to read: the payload is a defined zero either way. */
            if (tag == VAL_NULL) {
                emit(&e, jaiA64MovzX(JIT_SCRATCH_B, tag, 0));
                emit(&e, jaiA64StrW(JIT_SCRATCH_B, JIT_SCRATCH_A, at));
                emit(&e, jaiA64MovzX(JIT_SCRATCH_B, 0, 0));
                emit(&e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_A, at + 8));
                continue;
            }

            /* Must describe every local correctly whichever of the three homes it came from -- the one place a
             * register-plan mistake becomes a wrong answer, not a crash. A dynamic slot is the only one whose tag isn't a compile-time fact and the only one the plan never gives a register, so it's the only case that copies a tag through. */
            if (e.spilled && e.slotXReg[slot] == 0 && e.slotFpReg[slot] == 0 &&
                localTagInFrame(&e, slot)) {
                emit(&e, jaiA64LdrW(JIT_SCRATCH_C, 31, localFrameOff(&e, slot)));
                emit(&e, jaiA64StrW(JIT_SCRATCH_C, JIT_SCRATCH_A, at));
                emit(&e, jaiA64LdrX(JIT_SCRATCH_C, 31,
                                    localFrameOff(&e, slot) + 8));
                emit(&e, jaiA64StrX(JIT_SCRATCH_C, JIT_SCRATCH_A, at + 8));
                continue;
            }

            unsigned pr;
            if (!e.spilled) {
                pr = JIT_FIRST_SAVED + i;
            } else if (e.slotXReg[slot] != 0) {
                pr = e.slotXReg[slot];
            } else if (e.slotFpReg[slot] != 0) {
                emit(&e, jaiA64FmovXD(JIT_SCRATCH_C, e.slotFpReg[slot]));
                pr = JIT_SCRATCH_C;
            } else {
                emit(&e, jaiA64LdrX(JIT_SCRATCH_C, 31,
                                    localFrameOff(&e, slot) + 8));
                pr = JIT_SCRATCH_C;
            }
            if (kind == SLOT_MAYBE_INST) {
                /* Not JIT_SCRATCH_C when C is holding the payload, which it is
                 * whenever the slot came from the frame or an FP home. */
                unsigned spare = pr == JIT_SCRATCH_C ? JIT_SCRATCH_D
                                                     : JIT_SCRATCH_C;
                emitTagFor(&e, kind, pr, JIT_SCRATCH_B, spare);
            } else {
                emit(&e, jaiA64MovzX(JIT_SCRATCH_B, tag, 0));
            }
            emit(&e, jaiA64StrW(JIT_SCRATCH_B, JIT_SCRATCH_A, at));
            emit(&e, jaiA64StrX(pr, JIT_SCRATCH_A, at + 8));
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
            unsigned reg0 = valueBankBase(&e) + valueSeen;
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
            /* SLOT_NULL is a void call's result entry: register payload zero, tag NOT VAL_OBJ. Reaching this
             * chain through the default arm once wrote a null pointer out tagged as an object. */
            unsigned tag = kind == SLOT_INT   ? VAL_INT
                         : kind == SLOT_FLOAT ? VAL_FLOAT
                         : kind == SLOT_BOOL  ? VAL_BOOL
                         : kind == SLOT_NULL  ? VAL_NULL
                                              : VAL_OBJ;
            unsigned reg = valueBankBase(&e) + valueSeen;
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

    e.code[guardLoad] = jaiA64LdrLit(JIT_SCRATCH_A, e.limitLiteral - guardLoad);
    e.code[guardBranch] =
        jaiA64BCond(JAI_A64_LO, e.bailBlock - (int)guardBranch);

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
            /* Registers are assigned from the operand-stack depth at each point, so a join reached at two
             * different depths would read a value out of a register holding something else. The linear bytecode walk can't see that, so it's checked here and declined rather than mis-compiled. */
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
    /* JAI_JIT_DUMP=<function> writes that function's words to jit_<function>.bin and prints the
     * bytecode-offset-to-instruction map, so the code can be read back with `llvm-mc --disassemble --triple=aarch64` on the file's bytes. Reading the code is how the register plan gets checked at all -- three of this tier's bugs were found no other way. */
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
    /* 32-align the entry, which is two things at once.
     *
     * The literal pool's alignment is what it looks, as the 8-align this
     * replaced already gave. And every body now starts at a fixed offset
     * modulo the fetch block, so a size change ANYWHERE upstream stops
     * relabelling where every later body lands. That relabelling was the
     * suite's largest source of false A/B results: sweeping 0-7 padding nops
     * moved bitops +/-13% and loop_sum +/-7% with byte-identical loop bodies,
     * and matrix_mul +/-18.9% and list_ops +/-15.4% were both observed between
     * binaries with identical dynamic instruction counts. The cost is at most
     * 28 wasted bytes per compiled body in a 1 MB arena. */
    while ((arena->used & 31u) != 0) {
        uint32_t pad = jaiA64Nop();
        if (jaiCodeArenaWrite(arena, &pad, sizeof pad) == NULL) return false;
    }
    uint8_t *entry = jaiCodeArenaWrite(arena, e.code, e.count * sizeof e.code[0]);
    if (entry == NULL) return false;
    if (!jaiCodeArenaSeal(arena)) return false;

    /* The tier's whole bail protocol rests on partial execution being invisible. Field writes end that:
     * a body that stores to an instance and then bails would have the store applied again by the interpreted re-run. Nothing in the suite hits this, but "hard to construct" isn't the standard a compiled tier gets to work to. */
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
                "spill=%d xloc=%u fploc=%u fix=%u deopt=%u maxval=%u base=%u\n",
                fn->name ? fn->name->chars : "<anon>", e.arity, e.locals,
                e.count, e.savedCount, (int)e.spilled, e.xLocals, e.fpLocals,
                e.fixupCount, e.deoptCount, body.maxValue, e.base);
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
/* findLoopEnd walks from `top` itself, so if that offset is mid-instruction the walk decodes operand
 * bytes as opcodes and can find a plausible-but-wrong OP_LOOP -- nbody's advance was once compiled from offset 128, inside a GET_LOCAL2's operands. Walking from the start (isInstructionStart, below) costs one scan per compile and removes the question. */
/* OP_CLOSURE is the one variable-length instruction (u24 constant index + 3 bytes/upvalue, count
 * from the function the index names). Treating it as undecodable once refused OSR for the WHOLE CHUNK (both walks below scan from offset 0), so a module declaring a class or function before its hot loop -- i.e. every one of them -- could never enter a compiled loop at module scope, silently: compileOsr was never even reached, so nothing was there to report a decline. */
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

/* A slot captured BY REFERENCE by a closure is aliased by an ObjUpvalue pointing straight into the
 * VM's slot array; caching it in a register during OSR makes them different storage for the loop's duration -- reads through the closure see a frozen value, writes through it are lost. (`var base=3; let f=|x| x+base; while .. { acc=f(acc); base+=1 }` once returned a different, always-short sum every run, since OSR triggers on a timer tick.) Scans the WHOLE enclosing function, not just the loop region, since the capture can be anywhere. `how` bit 1 = by-value capture (safe, copies into a closed cell); bit 0 alone = by-reference. Marks individual slots, not chunk-wide, since the register plan is per-slot. Returns false (keep everything in memory) if the chunk doesn't decode. */
static bool chunkByRefCaptures(const Chunk *c, bool *byRef, unsigned nslots) {
    for (unsigned i = 0; i < nslots; i++) byRef[i] = false;
    for (int off = 0; off < c->count;) {
        int len = instructionLength(c, off);
        if (len <= 0) return false;
        if (c->code[off] == OP_CLOSURE) {
            for (int u = off + 4; u + 3 <= off + len; u += 3) {
                uint8_t how = c->code[u];
                if ((how & 1u) == 0 || (how & 2u) != 0) continue;
                unsigned slot = jaiReadU16(c->code + u + 1);
                if (slot < nslots) byRef[slot] = true;
            }
        }
        off += len;
    }
    return true;
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

/* How many loops enclose each byte, so slots can be ranked by heat: every OP_LOOP is a back edge, the
 * range it jumps over is its body, and one pass counting those ranges is enough to put the innermost loop's variables ahead of the setup around them (doesn't need to be exact). File static since both Emit structures read the same table and compilation isn't reentrant; a chunk longer than JIT_MAX_DEPTH_MAP just goes unweighted. */
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

/* The table, filled and handed back, so the function tier can ask for it
 * without gLoopDepth being visible where it is declared. */
static const uint8_t *loopDepthFor(const Chunk *c, unsigned *count) {
    *count = loopDepthTable(c);
    return gLoopDepth;
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
    e.scratchRoom = JIT_SCRATCH_BANK_COUNT;
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


    /* Runs AFTER the seeding above, not before: it used to measure a body where every slot was still an
     * untyped int (zeroed), a different program from the one being compiled, corrupting the maxValue the register decision rests on. Also narrows `locals` to what the loop actually names -- OSR started from the whole enclosing function's `maxSlots` (18-20 for real bodies), so `reserved + locals + maxValue <= 10` could never hold and register-resident locals were unreachable for any real function. */
    /* Registers for the slots that earn them, memory for the rest. The
     * reserved four (or one) plus the X locals plus the deepest expression
     * must all sit inside the ten callee-saved registers; a first pass
     * measures the last of those, and which slots are named at all, and how
     * often each of them is named inside the innermost loop. */
    unsigned probeMaxValue = 0;
    bool probeRan = false;
    {
        static Emit probe;
        memset(&probe, 0, sizeof probe);
        probe.osr = true; probe.measuring = true; probe.hasIter = hasIter;
        probe.iterKind = iterKind; probe.elemSample = elemSample;
        probe.osrTop = top; probe.osrEnd = end; probe.base = 0;
        probe.noInline = noInline;
        probe.locals = e.locals; probe.callsOut = true; probe.observed = slots;
        probe.scratchRoom = JIT_SCRATCH_BANK_COUNT;
        probe.offsetToInst = map; probe.offsetToDepth = depths;
        probe.limitLiteral = -1; probe.bailBlock = -1; probe.exceptionExit = -1;
        probe.loopDepth = gLoopDepth;
        probe.loopDepthCount = e.loopDepthCount;
        /* "Never seen" is an empty range, not offset zero. */
        for (unsigned i = 0; i <= JIT_MAX_SLOTS; i++) {
            probe.slotWriteLo[i] = UINT32_MAX;
            probe.slotIndexLo[i] = UINT32_MAX;
        }
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
            probeMaxValue = probe.maxValue;
            probeRan = true;

            /* A body that never calls out and never inlines can put its
             * operand stack in x0..x8 instead of above the locals, and then
             * the whole callee-saved bank is the locals'. This is the
             * difference between a five-deep expression leaving room for two
             * locals and a seven-deep one leaving room for none -- and the
             * loops that go seven deep are exactly the ones with several rows
             * or accumulators to keep. */
            e.scratchValues = !probe.clobbersScratch &&
                              probe.maxValueAll <= JIT_INL_COUNT;
            /* What is left over after the loop's own reserved registers and
             * the deepest expression the body builds. maxValue is model state,
             * not a register number, so measuring it in memory mode and
             * spending it here is sound. */
            unsigned overhead = osrReserved(&e) +
                                (e.scratchValues ? 0u : probe.maxValue);
            unsigned availX = overhead < JIT_MAX_SAVED
                                  ? JIT_MAX_SAVED - overhead : 0u;
            unsigned availFp = JIT_FP_MAX_SAVED;
            /* Busiest first, and only slots the body names. A slot whose kind
             * varies at run time keeps its tag check and stays in memory. */
            uint8_t order[JIT_MAX_SLOTS + 1];
            bool byRef[JIT_MAX_SLOTS + 1];
            bool decoded = chunkByRefCaptures(&fn->chunk, byRef, e.locals);
            unsigned n = 0;
            for (unsigned i = 0; decoded && i < e.locals; i++) {
                if (probe.slotUse[i] == 0) continue;
                if (probe.dynamicLocal[i]) continue;
                /* Captured by reference (see chunkByRefCaptures): stays in memory, since a register copy would be different storage from what the closure reads/writes. */
                if (byRef[i]) continue;
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

            /* What planHoists needs: where each slot was written and where it
             * was subscripted, and the registers nothing else claims. The pool
             * exists only for a body that calls nothing -- which is also the
             * condition that makes a hoisted header sound (see planHoists). */
            e.bodyCalls = probe.clobbersScratch;
            for (unsigned i = 0; i <= JIT_MAX_SLOTS; i++) {
                e.slotWriteLo[i]  = probe.slotWriteLo[i];
                e.slotWriteHi[i]  = probe.slotWriteHi[i];
                e.slotIndexLo[i]  = probe.slotIndexLo[i];
                e.slotIndexHi[i]  = probe.slotIndexHi[i];
                e.slotIndexUse[i] = probe.slotIndexUse[i];
            }
            if (!e.bodyCalls) {
                for (unsigned r = 0; r < JIT_FREE_COUNT; r++) {
                    e.hoistPool[e.hoistPoolCount++] =
                        (uint8_t)(JIT_FREE_FIRST + r);
                }
                /* Whatever the operand stack left at the top of its own bank.
                 * Only when it IS that bank -- otherwise an inlined body has
                 * x0..x8 to itself and none of it is spare. */
                if (e.scratchValues) {
                    for (unsigned r = probe.maxValueAll;
                         r < JIT_SCRATCH_BANK_COUNT; r++) {
                        e.hoistPool[e.hoistPoolCount++] =
                            (uint8_t)(JIT_INL_BANK + r);
                    }
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
     * is nothing left to guard here.
     *
     * Except the width. Unlike the function tier, which marshals every argument
     * through jitArgIn and hands a bool over as a clean 0 or 1, these slots
     * are the interpreter's own and BOOL_VAL is a one-byte `strb` -- the seven
     * bytes above a bool are whatever that slot last held. An eight-byte load
     * puts that garbage in a SLOT_BOOL register, and every consumer of one
     * tests the whole word, so `if flag {` took the wrong arm whenever the
     * slot had previously held anything with a high byte set. Silent: the
     * program ran, and computed the other branch. */
    for (unsigned i = 0; i < e.locals; i++) {
        if (e.slotXReg[i] != 0 && e.localKind[i] == SLOT_BOOL) {
            emit(&e, jaiA64LdrByte(e.slotXReg[i], JIT_SLOTS_REG, i * 16u + 8u));
        } else if (e.slotXReg[i] != 0) {
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
        if (iterKind == 1) {
            /* A range yields start + index, so both bounds are shifted by the
             * start once, here, and the loop runs on the yielded value
             * directly. ObjIter.source is a Value, so the object pointer sits
             * eight bytes into it. jaiJitEnterOsr has already established that
             * start + limit does not overflow. */
            emit(&e, jaiA64LdrX(JIT_SCRATCH_A, rIter,
                                (unsigned)offsetof(ObjIter, source) + 8));
            emit(&e, jaiA64LdrX(JIT_SCRATCH_A, JIT_SCRATCH_A,
                                (unsigned)offsetof(ObjRange, start)));
            emit(&e, jaiA64AddX(JIT_IDX_REG, JIT_IDX_REG, JIT_SCRATCH_A));
            emit(&e, jaiA64AddX(JIT_LIM_REG, JIT_LIM_REG, JIT_SCRATCH_A));
        } else {
            /* A list head keeps the ObjList itself in JIT_START_REG. */
            emit(&e, jaiA64LdrX(JIT_START_REG, rIter,
                                (unsigned)offsetof(ObjIter, source) + 8));
        }
    }

    planHoists(&e, fn);

    if (!compileBody(&e, closure) || e.failed) {
        if (getenv("JAI_JIT_WHY")) {
            /* The register arithmetic, not just the verdict: a "more live
             * values" decline is unreadable without knowing how many the head
             * reserved, how deep the body went and which bank it was using --
             * the shortfall is those three against JIT_MAX_SAVED. Its own
             * line, and without the words the decline census greps for, so
             * that adding it cannot invent census entries.
             *
             * Only when the measuring pass got through: a loop that declined
             * for some other reason never reached the register plan, and
             * printing its zeroed numbers reads as "nought entries deep",
             * which is a different and false claim. */
            if (probeRan) fprintf(stderr,
                    "[jit] osr at %u registers: %u reserved, %u stack, "
                    "%u x-locals, bank %s, of %u\n",
                    top, osrReserved(&e), probeMaxValue, e.xLocals,
                    e.scratchValues ? "x0" : "callee-saved", JIT_MAX_SAVED);
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
            unsigned rIx = JIT_IDX_REG;                                        \
            if (iterKind == 1) {                                               \
                rIt = JIT_SCRATCH_A;                                           \
                emit(&e, jaiA64LdrX(rIt, 31, e.iterFrameOffset));              \
                /* The index register carries the yielded value, not the       \
                 * index; ObjIter::index is zero-based, so the start comes      \
                 * back off. Two loads and a subtract, once per way out. */    \
                emit(&e, jaiA64LdrX(JIT_SCRATCH_B, rIt,                        \
                                    (unsigned)offsetof(ObjIter, source) + 8)); \
                emit(&e, jaiA64LdrX(JIT_SCRATCH_B, JIT_SCRATCH_B,              \
                                    (unsigned)offsetof(ObjRange, start)));     \
                emit(&e, jaiA64SubsXReg(JIT_SCRATCH_B, JIT_IDX_REG,            \
                                        JIT_SCRATCH_B));                       \
                rIx = JIT_SCRATCH_B;                                           \
            }                                                                  \
            emit(&e, jaiA64StrX(rIx, rIt,                                      \
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
                /* Result of an already-happened call lives in the descriptor with whatever tag the callee actually
                 * returned. The function tier's stub knew this; this OSR one didn't, and alloc_churn came back 982406343 instead of 550770565 under deopt stress -- the entire reason this lastFromDesc branch exists. */
                unsigned rat = e.descOffset +
                               (unsigned)offsetof(JitCallDesc, result);
                emit(&e, jaiA64LdrW(JIT_SCRATCH_B, 31, rat));
                emit(&e, jaiA64StrW(JIT_SCRATCH_B, JIT_SCRATCH_A, at));
                emit(&e, jaiA64LdrX(JIT_SCRATCH_B, 31, rat + 8));
                emit(&e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_A, at + 8));
                valueSeen++;
                continue;
            }
            unsigned reg0 = valueBankBase(&e) + valueSeen;
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
            /* Same SLOT_NULL hazard as the function tier's stub -- see there. */
            unsigned tag = kind == SLOT_INT   ? VAL_INT
                         : kind == SLOT_FLOAT ? VAL_FLOAT
                         : kind == SLOT_BOOL  ? VAL_BOOL
                         : kind == SLOT_NULL  ? VAL_NULL
                                              : VAL_OBJ;
            unsigned reg = valueBankBase(&e) + valueSeen;
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
    /* Same 32-alignment as the function tier above, for the same two reasons. */
    while ((arena->used & 31u) != 0) {
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

    /* Pin the class of every instance slot the compile specialised on. Without
     * this the entry guard checks only IS_INSTANCE, and a loop compiled for one
     * class entered holding another reads a field at the wrong slot index --
     * see JaiOsrForm's note for the probe. Refuse rather than compile
     * unguarded if the pairs do not fit. */
    form->shapeCount = 0;
    for (unsigned i = 0; i < e.locals; i++) {
        SlotKind k = e.localKind[i];
        if (k != SLOT_INST && k != SLOT_MAYBE_INST) continue;
        if (e.localShape[i] == 0) continue;   /* no class pinned to this slot */
        if (form->shapeCount >= JAI_OSR_SHAPES) {
            if (getenv("JAI_JIT_WHY")) {
                fprintf(stderr, "[jit] osr %s at %u stopped: more than %d "
                        "instance slots to pin\n",
                        fn->name ? fn->name->chars : "<anon>", top,
                        JAI_OSR_SHAPES);
            }
            return false;
        }
        form->shapeSlot[form->shapeCount] = (uint8_t)i;
        form->shapeId[form->shapeCount]   = e.localShape[i];
        form->shapeCount++;
    }

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
            /* Compiled head runs on `start + index` in one register, so the limit is biased by the start too (see
             * osrReserved). `limit` saturates at INT64_MAX for a range spanning the whole type, and a biased limit that wrapped would compare the wrong way round -- such a loop is left to the interpreter. */
            {
                int64_t biased;
                if (__builtin_add_overflow(AS_RANGE(iter->source)->start,
                                           iter->limit, &biased)) {
                    return 0;
                }
            }
            iterKind = 1;
        } else if (iter->kind == ITER_LIST && IS_LIST(iter->source)) {
            ObjList *src = AS_LIST(iter->source);
            /* Element the loop is about to bind, taken from the list itself: reading the loop variable's slot
             * instead gives whatever the previous iteration left there (nothing on the first entry), aiming the kind guard at the wrong type -- what crashed the first attempt at this. */
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
        /* Kind as well as offset: a form compiled for a range head, entered with a list iterator, would read
          * ObjRange::start out of an ObjList -- `for x in cond ? xs : 0..n` is enough to arrange it. */
        if (fn->osrForms[i].top == top &&
            fn->osrForms[i].iterKind == iterKind) {
            form = &fn->osrForms[i];
            break;
        }
    }
    if (form == NULL) {
        if (fn->osrRefused || fn->osrCount >= JAI_OSR_MAX) return 0;
        /* This head's own share of the budget. See ObjFunction::osrMissTop for
         * why the count cannot be per function: the first loop to run spends
         * it, and every later loop in the same body is then refused a look. */
        unsigned miss = fn->osrMissCount;
        for (unsigned i = 0; i < fn->osrMissCount; i++) {
            if (fn->osrMissTop[i] == top) { miss = i; break; }
        }
        if (miss < JAI_OSR_MAX && fn->osrMissAttempts[miss] >= 5 * JAI_OSR_MAX) {
            return 0;   /* this head is spent; other heads are not */
        }
        if (!compileOsr(closure, top, frame->slots, iterKind, elemSample,
                        false) &&
            !compileOsr(closure, top, frame->slots, iterKind, elemSample,
                        true)) {
            /* Inlining widens live ranges; a loop that will not fit with it
             * may fit without, and a compiled call beats no compile at all. */
            if (miss == fn->osrMissCount && miss < JAI_OSR_MAX) {
                fn->osrMissTop[miss]      = top;
                fn->osrMissAttempts[miss] = 0;
                fn->osrMissCount++;
            }
            if (miss < JAI_OSR_MAX) fn->osrMissAttempts[miss]++;
            /* The whole-function backstop, for a body with more uncompilable
             * heads than the table holds: without it those heads share the
             * untracked path and would be retried for the life of the run. */
            if (++fn->osrAttempts >= 5 * JAI_OSR_MAX * JAI_OSR_MAX) {
                fn->osrRefused = true;
            } else if (fn->osrMissCount >= JAI_OSR_MAX) {
                bool allSpent = true;
                for (unsigned i = 0; i < JAI_OSR_MAX; i++) {
                    if (fn->osrMissAttempts[i] < 5 * JAI_OSR_MAX) {
                        allSpent = false;
                        break;
                    }
                }
                if (allSpent) fn->osrRefused = true;
            }
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
        /* jitArgIn (function tier) has always checked this; the loop tier let it through via `default`, so a
         * slot compiled as "some object" could be entered holding an int and have its payload read as a pointer. Unexercised until something emitted a load off such a slot (the invoke arm's receiver guard, below). */
        case SLOT_OBJ:   if (!IS_OBJ(v))   return 0; break;
        default: break;   /* opaque: never read */
        }
    }

    /* And it must be the same CLASS, not merely an instance. The kind check
     * above passes any instance; the compiled code reads fields at slot indices
     * baked in from one particular class. */
    for (unsigned i = 0; i < form->shapeCount; i++) {
        Value v = frame->slots[form->shapeSlot[i]];
        if (IS_NULL(v)) continue;    /* SLOT_MAYBE_INST, and it is the null */
        if (!IS_INSTANCE(v)) return 0;
        ObjClass *klass = AS_INSTANCE(v)->klass;
        if (klass == NULL || klass->shapeId != form->shapeId[i]) return 0;
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
            /* Class as well as type: every field offset in the body was resolved against this one shape. Holding
             * the raw pointer is safe only because the body can't allocate (no collection can run meanwhile), and the argument slots keep the instance reachable. */
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

static inline JaiJitOutcome jitResultOut(ObjFunction *fn, JitResult r,
                                         Value *slotBase) {
    if (r.bailed == 2) return JAI_JIT_ERROR;
    if (r.bailed == 4) return JAI_JIT_DEOPT;
    if (r.bailed) {
        /* Overflow or a low stack: nothing was written (the body cannot write), so handing the call back to
         * the interpreter is enough -- it raises the error with a traceback. Refused permanently so a bailing body isn't re-entered every call only to bail again. */
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

    /* Compiled code reads the global naming this function exactly once, at compile time, then calls it
     * directly. Rebinding the name must invalidate that; the module's version counter moves on every global mutation, so one comparison covers it -- conservative (any global write in the module retires the form), which is the safe direction. */
    if (fn->module == NULL || fn->module->version != fn->jitFuncModuleVersion) {
        return JAI_JIT_DECLINED;
    }

    unsigned arity = fn->jitArgCount;
    int64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    /* Unrolled rather than a loop over `int64_t a[JIT_MAX_ARITY]` -- measured, not tidiness: an array of
     * four int64 makes clang add a stack-protector prologue/epilogue and sends every argument out to the frame and back on its way to the register the call reads it from. This function sits on the path of every interpreted call into a compiled body, so both costs are paid per call. */
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

/* See vm.h. Everything jaiCallValue1 and jaiJitEnterFunc do for one argument,
 * in one frame: the checks are unchanged and in the same order, but the
 * callee-saved prologue, the argument's round trip through the caller's frame
 * and the second `bl` are gone. Higher-order builtins drive this once per
 * element, so all of that was per-element cost.
 *
 * Only jitArgCount 1 is taken, plus a trailing SLOT_CLOSURE for a callee that
 * reads an upvalue -- that is not a slot, it is the closure itself, so a
 * capturing lambda still gets the flat path. Anything else declines and
 * jaiCallValue1 handles it exactly as before. */
static JAI_NOINLINE bool callFn1Rerun(Value *base, Value *out) {
    Value callee = base[0], arg = base[1];
    vm.stackTop = base;
    return jaiCallValue1(callee, arg, out);
}

bool jaiCallFn1(Value callee, Value arg, Value *out) {
    if (JAI_UNLIKELY(!IS_CLOSURE(callee))) return jaiCallValue1(callee, arg, out);
    ObjClosure *closure = AS_CLOSURE(callee);
    ObjFunction *fn = closure->fn;
    if (JAI_UNLIKELY(fn->jitFunc == NULL || fn->arity != 1))
        return jaiCallValue1(callee, arg, out);

    unsigned nargs = fn->jitArgCount;
    if (JAI_UNLIKELY(nargs == 0 || nargs > 2))
        return jaiCallValue1(callee, arg, out);
    if (nargs == 2 && (SlotKind)fn->jitParamKind[1] != SLOT_CLOSURE)
        return jaiCallValue1(callee, arg, out);

    /* Same guard jaiJitEnterFunc makes: compiled code read this module's
     * globals once, at compile time. */
    if (JAI_UNLIKELY(fn->module == NULL ||
                     fn->module->version != fn->jitFuncModuleVersion)) {
        return jaiCallValue1(callee, arg, out);
    }

    if (JAI_UNLIKELY(vm.stack == NULL ||
                     vm.stackTop + 3 > vm.stack + JAI_STACK_MAX)) {
        return jaiCallValue1(callee, arg, out);
    }

    /* The two cells are what keeps the closure and the argument reachable
     * while the compiled body runs, exactly as in jaiCallValue1: a compiled
     * body may allocate and a collection scans the VM stack. */
    Value *base = vm.stackTop;
    base[0] = callee;
    base[1] = arg;
    vm.stackTop = base + 2;

    int64_t a0 = 0;
    if (JAI_UNLIKELY(!jitArgIn(closure, base, 0, &a0))) {
        vm.stackTop = base;
        return jaiCallValue1(callee, arg, out);
    }

    int frameBase = vm.frameCount;
    JitResult r = nargs == 1
                      ? ((Fn1)(uintptr_t)fn->jitFunc)(a0)
                      : ((Fn2)(uintptr_t)fn->jitFunc)(a0,
                                                      (int64_t)(uintptr_t)closure);
    JaiJitOutcome outcome = jitResultOut(fn, r, base);
    if (JAI_LIKELY(outcome == JAI_JIT_DONE)) {
        *out = base[0];
        vm.stackTop = base;
        return true;
    }
    if (outcome == JAI_JIT_ERROR) {
        vm.stackTop = base;
        return false;
    }
    if (outcome == JAI_JIT_DEOPT) {
        return jaiFinishJitDeopt1(closure, base, frameBase, out);
    }
    /* Refused mid-flight (a bail retires the form), so re-run it interpreted.
     * Reading the callee and the argument back out of the two cells rather
     * than off the parameters is what keeps them dead across the `blr`: with
     * four live registers fewer, the callee-saved set this function has to
     * spill on every element drops from six pairs to two. */
    return callFn1Rerun(base, out);
}

#else

bool jaiJitCompileFunc(ObjClosure *closure, Value *slotBase) {
    (void)closure; (void)slotBase; return false;
}
JaiJitOutcome jaiJitEnterFunc(ObjClosure *closure, Value *slotBase) {
    (void)closure; (void)slotBase; return JAI_JIT_DECLINED;
}
bool jaiCallFn1(Value callee, Value arg, Value *out) {
    return jaiCallValue1(callee, arg, out);
}

#endif

/* A conditional branch whose target is reached with a different operand stack
 * than the branch leaves from -- the exhausted arm of a for-loop, where the
 * interpreter drops the iterator. */
static void branchToDepth(Emit *e, uint32_t targetOffset, unsigned cond,
                          int depthOverride) {
    if (e->fixupCount >= JIT_MAX_FIXUPS) { e->failed = true; return; }
    settleAll(e);   /* see branchTo: a join agrees about where every value is */
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
