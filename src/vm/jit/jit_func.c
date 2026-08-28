/* jit_func.c -- whole-function JIT tier: compiles self-recursive, integer-only bodies to native arm64, bailing to the interpreter on overflow, deep recursion, or an unsupported shape. */
#include "vm/jit/jit.h"

#include "vm/jit/jit_arm64.h"
/* For jaiJitFieldReadFor: which builtins are one load from their receiver. */
#include "vm/jit/jit_field_read.h"
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
/* How many live values one call out can root, which is NOT the register budget
 * above even though it was the same number for a long time. The register count
 * is a hardware fact: x19..x28 is ten and cannot be more. The root count is the
 * length of an ARRAY on the frame, and what it bounds is mostly LOCALS --
 * emitRootFill walks every object-kind local, and a local that earned no
 * register lives in memory, so the count is not tied to the ten at all.
 *
 * Sharing the constant made "too many roots" one of the hottest refusals in
 * jaicv: eighty attempts at each of five offsets in one `imgproc` run, on a
 * package whose functions routinely hold a dozen Mats. The price of the split
 * is 14 more Values (224 bytes) on the frame of a body that calls out, which
 * carries it past the 504 bytes that keep the cheap stp-pre prologue -- one or
 * two extra instructions once per body, against a whole body compiling. */
#define JIT_MAX_ROOTS   24u
/* Model entries, not register count -- an inlined body's operand-stack entries live in their own bank, wider than x19..x28. */
#define JIT_MAX_STACK   20u
/* Slots the compile-time model can describe, not the register budget: a declared frame can be wider
 * than what it touches; only the second (measuring) pass is held to JIT_MAX_SAVED. */
#define JIT_MAX_SLOTS   64u
#define JIT_MAX_ARITY    8u   /* arguments arrive in x0..x7 */
/* Deopt stubs dominate this size: each writes out every local and live stack entry. `merge` silently needed 512 -- hence the diagnostics. */
#define JIT_MAX_INSTS 20000u
#define JIT_MAX_FIXUPS 6000u
/* How many links of a refusal chain JAI_JIT_CHAIN will walk out. Each costs one
 * extra compile of the body, and a chain longer than this is not a backlog item
 * anybody is going to clear in one go. */
#define JIT_MAX_CHAIN 8u
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
#define JIT_MAX_PIC_EXITS JAI_IC_WAYS
#define JIT_MAX_HOIST    4u
/* How many distinct clobber SITES the measuring pass will remember, so that
 * "does x0..x8 survive across this bytecode range" can be asked of a range
 * rather than of the whole body. Past this the body answers yes everywhere,
 * which is the same answer it gave before the range existed. */
#define JIT_MAX_CLOBBER 24u
/* Repeated from the register plan below, which cannot be declared this early:
 * x0..x8, the bank a call-free body's operand stack uses. */
#define JIT_SCRATCH_BANK_COUNT 9u

/* ------------------------------------------------------------------ */
/* Calling out of compiled code                                         */
/* ------------------------------------------------------------------ */

/* How many Values one call out of compiled code can carry. It bounds four
 * things at once: a descriptor call's arguments, an invoke's (receiver + this
 * minus one), an f-string's parts, and the fields a simple constructor stores.
 *
 * Was 4. The self-hosted compiler checking parser.jai stopped at
 * "OP_INVOKE: 5 arguments, past the cap of 3" eighty times at ONE osr loop --
 * a hot one -- and also wanted six and seven elsewhere; 8 covers every arity
 * that corpus asks for. The price is 4 more Values (64 bytes) on the frame of
 * every compiled body that calls out, which is stack, never touched beyond
 * what is used, and leaves the frame under the 504 bytes that keep the cheap
 * stp-pre prologue. */
#define JIT_MAX_ARGS_OUT 8

/* Values first in JitCallDesc so every field is 8-aligned and the emitted stores can use scaled forms. */
typedef struct JitCallDesc {
    /* link/nroots come first so `roots` sits at a fixed offset from the chain head; link != NULL means this descriptor is on the collector's walk chain. */
    struct JitCallDesc *link;
    int64_t nroots;
    Value   roots[JIT_MAX_ROOTS];
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
    /* Bit i: local i is NOT described by this record and must be left as the
     * frame already has it. SLOT_OPAQUE means "the compiled body never reads
     * this slot", which jitArgIn relies on to pass a raw 0 for an argument of
     * a kind the tier has no register for -- but a DEOPT hands the frame to
     * the interpreter, and the interpreter does read it. Writing the record's
     * null over it turned `enter_foreign(module)` into `enter_foreign(null)`
     * and the whole compiler then failed to resolve an imported type.
     * bindCallArgs runs before jaiJitApplyDeopt at every call site, so the
     * real argument is already there; the fix is to not touch it.
     * The OSR tier has always got this right -- OSR_SYNC_ITER skips a slot
     * whose tag is VAL_NULL -- which is why only the function tier was wrong. */
    int64_t skipLocals;
    Value   locals[JIT_MAX_SLOTS + 1];
    Value   stack[JIT_MAX_STACK + 1];
} JitDeoptRecord;

_Static_assert(JIT_MAX_SLOTS <= 64,
               "skipLocals is one bit per local slot");

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
        if (i < 64 && (((uint64_t)gDeopt.skipLocals >> i) & 1u) != 0) continue;
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

/* Receiver whose class the model could not pin. The method NAME travels in the
 * descriptor's callee slot -- there is no method Value to put there -- and the
 * resolve happens per call against the shared megamorphic table. */
static int jitInvokeByName(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    bool ok = jaiInvokeMethodByName(AS_STRING(d->callee), d->args,
                                    (int)d->argc, &d->result);
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
    for (int64_t i = 0; i < d->argc; i++) jaiListPut(list, (int)i, d->args[i]);
    list->count = (int)d->argc;
    d->result = OBJ_VAL(list);
    jaiGCPopRootRange();
    return 0;
}

/* `x in c`. The needle is args[0] and the container args[1], matching the
 * operand order the interpreter peeks. The answer is stored as a Value so the
 * one-byte BOOL_VAL member is the thing the caller's LdrByte reads -- see the
 * SLOT_BOOL result convention in emitDescriptorStatus.
 *
 * Roots go down because a container's __contains__ can run Jaithon code and
 * therefore collect. Returns 1 having thrown. */
static int jitContains(JitCallDesc *d) {
    bool contains = false;
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    bool ok = jaiContainsOp(d->args[1], d->args[0], &contains);
    jaiGCPopRootRange();
    if (!ok) return 1;
    d->result = BOOL_VAL(contains);
    return 0;
}

static int jitNotContains(JitCallDesc *d) {
    if (jitContains(d) != 0) return 1;
    d->result = BOOL_VAL(!AS_BOOL(d->result));
    return 0;
}

/* `{a: 1, b: 2}`. Keys and values alternate in the operand list, which is the
 * order the interpreter reads them in, so the args array is used as-is.
 *
 * jaiDictSet hashes the key and can raise on an unhashable one, so the root
 * range goes down and a raise comes back as 1 for the descriptor's threw
 * branch. */
static int jitBuildDict(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    ObjDict *dict = jaiDictNew();
    jaiGCPushRoot(OBJ_VAL(dict));
    for (int64_t i = 0; i + 1 < d->argc; i += 2) {
        (void)jaiDictSet(dict, d->args[i], d->args[i + 1]);
        if (vm.hasException) {
            jaiGCPopRoot();
            jaiGCPopRootRange();
            return 1;
        }
    }
    jaiGCPopRoot();
    jaiGCPopRootRange();
    d->result = OBJ_VAL(dict);
    return 0;
}

/* `{a, b}`. Same shape, one operand per element. */
static int jitBuildSet(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    ObjSet *set = jaiSetNew();
    jaiGCPushRoot(OBJ_VAL(set));
    for (int64_t i = 0; i < d->argc; i++) {
        (void)jaiSetAdd(set, d->args[i]);
        if (vm.hasException) {
            jaiGCPopRoot();
            jaiGCPopRootRange();
            return 1;
        }
    }
    jaiGCPopRoot();
    jaiGCPopRootRange();
    d->result = OBJ_VAL(set);
    return 0;
}

static int jitBuildTuple(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    ObjTuple *tuple = jaiTupleNew(d->args, (int)d->argc);
    jaiGCPopRootRange();
    d->result = OBJ_VAL(tuple);
    return 0;
}

/* `a == b` on two heap objects the tier can say nothing else about.
 *
 * The interpreter's own equality, so an `__eq__` behaves and a comparison that
 * raises raises the same message. It can therefore allocate and throw: roots go
 * down, and 1 comes back for the descriptor's threw branch.
 *
 * This is the LAST arm in each equality chain on purpose. Every arm above it
 * answers without a call -- interned string pointers, a folded enum unit, two
 * registers -- and each is strictly better where it applies. */
static int jitValuesEqual(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    bool equal = jaiValuesEqual(d->args[0], d->args[1]);
    jaiGCPopRootRange();
    if (vm.hasException) return 1;
    d->result = BOOL_VAL(equal);
    return 0;
}

static bool jitObjEquality(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_OBJ_EQ");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

/* `a + b` on two strings. Both operands are guarded for OBJ_STRING at the call
 * site, so this is jaiStringConcat and nothing else -- the general arithmetic()
 * fallback would have to be answered with a tag test on the result, and there
 * is no shape of `+` other than this one that reaches here.
 *
 * Allocates (and appends into an existing ObjStrBuf), so roots go down first as
 * for any call out of compiled code. NULL back means the concatenation
 * overflowed UINT32_MAX and already threw. */
static int jitStringConcat(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    ObjString *s = jaiStringConcat(AS_STRING(d->args[0]), AS_STRING(d->args[1]));
    jaiGCPopRootRange();
    if (s == NULL) return 1;
    d->result = OBJ_VAL(s);
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

/* OP_GET_ITER_ITEMS' dict case: the lazy view `for (k, v) in d.items()` walks,
 * built once per loop entry rather than the N 2-tuples the eager `items()`
 * materialises. Allocates, so roots go down first as for any call out of
 * compiled code. The emitted guard has already proved the receiver is a dict;
 * the test here is the same belt-and-braces jitMakeIter carries. */
static int jitMakeItemsIter(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    Value src = d->args[0];
    ObjIter *it = IS_DICT(src) ? jaiIterNew(ITER_DICT_ITEMS, src) : NULL;
    if (it != NULL) d->result = OBJ_VAL(it);
    jaiGCPopRootRange();
    return it != NULL ? 0 : 1;
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

/* Dict read. Calls the interpreter's own OP_GET_INDEX so a missing key raises
 * the KeyError it would have raised, with the message it would have used --
 * naming the key -- rather than a second spelling of the same error that has to
 * be kept in step with it. Pure apart from that throw, but it can allocate the
 * exception, so the roots go down first. */
static int jitGetIndexDict(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    bool ok = jaiIndexGet(d->args[0], d->args[1], &d->result);
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

/* A SlotKind's name, for JAI_JIT_WHY. Several refusals used to print the raw
 * enumerator ("the container is kind 13, not a list"), which names the one
 * thing a reader of the message cannot look up. */
static const char *slotKindName(SlotKind k) {
    switch (k) {
    case SLOT_INT:        return "int";
    case SLOT_FLOAT:      return "float";
    case SLOT_INST:       return "instance";
    case SLOT_MAYBE_INST: return "instance-or-null";
    case SLOT_SELF:       return "self";
    case SLOT_OPAQUE:     return "opaque";
    case SLOT_CLOSURE:    return "closure";
    case SLOT_CLASS:      return "class";
    case SLOT_FUNC:       return "function";
    case SLOT_NATIVE:     return "builtin";
    case SLOT_ITER:       return "iterator";
    case SLOT_BOOL:       return "bool";
    case SLOT_NULL:       return "null";
    case SLOT_OBJ:        return "object";
    case SLOT_LIST:       return "list";
    }
    return "an unnamed kind";
}

static bool holdsRegister(SlotKind k) {
    return k != SLOT_SELF && k != SLOT_CLASS && k != SLOT_FUNC &&
           k != SLOT_NATIVE;
}

/* Some fixup targets are not bytecode offsets at all: they are sentinels at the
 * top of the u32 range, each one a base minus an index into a table.
 *
 * The bases are DERIVED from the table sizes rather than written down, because
 * hand-picked ones overlapped. FIXUP_DEOPT was UINT32_MAX-7 minus an index into
 * a 160-entry table, so it ran down to UINT32_MAX-166 -- straight through
 * FIXUP_EXIT at UINT32_MAX-100. The resolver tests EXIT first, so **deopt sites
 * 93 through 100 would have resolved to an exit stub**: a compiled body jumping
 * out of its loop where it meant to hand back to the interpreter.
 *
 * It was never reachable, which is why it sat here: the largest deopt count
 * anyone had seen was 16. It is 47 today across the benchmark suite, because a
 * day of adding guards to the tier moved it halfway. Deriving the bases makes
 * the overlap impossible rather than merely unlikely, and the assertions below
 * fail the build if a future table size reintroduces it. */
#define JIT_MAX_DEOPT     160u
#define JIT_MAX_EXIT      8u
/* Self-calls and direct calls to a callee that writes share this table, so it
 * is sized for a body with several of each rather than for recursion alone. */
#define JIT_MAX_SELF_SLOW 32u
#define JIT_MAX_GROW      16u

#define FIXUP_BAIL   UINT32_MAX
#define FIXUP_ENTRY  (UINT32_MAX - 1u)
#define FIXUP_THREW  (UINT32_MAX - 2u)
#define FIXUP_OVF    (UINT32_MAX - 3u)   /* minus 0,1,2 for the three operators */
#define FIXUP_DEOPT    (UINT32_MAX - 7u)                     /* minus a deopt index */
#define FIXUP_EXIT     (FIXUP_DEOPT - JIT_MAX_DEOPT)         /* minus an exit index */
#define FIXUP_SELFSLOW (FIXUP_EXIT - JIT_MAX_EXIT)           /* minus a self-call index */
#define FIXUP_GROW     (FIXUP_SELFSLOW - JIT_MAX_SELF_SLOW)  /* minus a growth index */

/* Every sentinel range must stay above any offset a real chunk can have. A
 * chunk that large is not representable long before this matters, so half the
 * u32 range is an enormous margin -- the point is that the build fails if the
 * tables ever grow enough to reach down into bytecode-offset territory. */
_Static_assert(FIXUP_GROW - JIT_MAX_GROW > UINT32_MAX / 2u,
               "jit fixup sentinels have grown down into bytecode offsets");
_Static_assert(FIXUP_EXIT < FIXUP_DEOPT - (JIT_MAX_DEOPT - 1u),
               "jit deopt and exit fixup ranges overlap");
_Static_assert(FIXUP_SELFSLOW < FIXUP_EXIT - (JIT_MAX_EXIT - 1u),
               "jit exit and self-call fixup ranges overlap");
_Static_assert(FIXUP_GROW < FIXUP_SELFSLOW - (JIT_MAX_SELF_SLOW - 1u),
               "jit self-call and growth fixup ranges overlap");

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
    /* The object type this SLOT_OBJ entry is expected to have, as ObjType + 1,
     * or 0 for "no expectation". For results the tier produces itself and so
     * has no Value to sample -- an f-string's, a builtin method's predicted
     * from its site's feedback, a `str()` call's -- where the type is known
     * even though the object does not exist until run time.
     *
     * A PREDICTION, not a proof: every consumer already re-checks Obj.type and
     * deoptimises on a miss, so this only chooses which guard to emit. That is
     * why it survives a branch merge and is not listed in clearStackProofs --
     * see the contrast drawn at stackAscii below. */
    /* The kind that could not be adopted into a local already typed otherwise;
     * read only by kindClash, to say WHICH two kinds disagreed. */
    SlotKind  clashKind;
    /* See emitUnarmedDeopt: the opcode and offset the walk stopped at, so the
     * fixup pass can name the cause and not just the symptom. */
    uint8_t   unarmedOp;
    uint32_t  unarmedAt;
    uint8_t   stackObjType[JIT_MAX_STACK];
    /* An exemplar of what THIS list entry's elements are, for a list the body
     * built itself and so has no live sample of. `OP_BUILD_LIST` knows the kind
     * of everything it just popped; nothing downstream does, because the list
     * does not exist until run time.
     *
     * A prediction of the same standing as stackSeen: whatever reads it emits
     * the guard it would have emitted anyway. For an int, float or bool the
     * exemplar is a synthesised immediate and so has no lifetime at all; for an
     * object it is the element's own live sample, which was already being held
     * here and is reachable for the same reasons it was. */
    Value     stackElem[JIT_MAX_STACK];
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
     * see the clearStackProofs call in the walk. */
    bool      stackAscii[JIT_MAX_STACK];

    /* This entry is the shared payload-less value of the enum variant named
     * by stackSeen -- baked as a constant pointer by the `Enum.Variant` fold
     * in OP_GET_FIELD, not merely observed to be one. A proof of the same
     * kind as stackAscii and retired at the same offsets, and for the same
     * reason: it deletes work (the pointer compare in OP_EQ stands in for
     * jaiValuesEqual) rather than choosing a guard. */
    bool      stackUnit[JIT_MAX_STACK];

    /* This entry is the `null` literal itself -- OP_NULL, not merely something
     * whose kind admits a null. It has to be tracked separately because
     * OP_NULL pushes SLOT_MAYBE_INST (a null literal and an instance have to
     * agree on a kind, or `var x: Box? = null` gives its local two of them),
     * so the kind alone cannot tell `x is null` from `x is y`. A proof of the
     * same kind as the two above and retired at the same offsets. */
    bool      stackNullLit[JIT_MAX_STACK];

    /* Fields already stored this call, with their kind: a read of one needs no tag check since nothing
     * can have changed it -- e.g. `self.n = self.n + 1; return self.n` would otherwise bail-after-write, which the tier refuses.
     *
     * "Nothing can have changed it" was once true because the body could not
     * call at all. It can now, so the claim is only as good as its
     * invalidations, and there are four (see forgetFieldKinds and
     * forgetFieldKindsOfLocal): another store to the same field slot
     * (recordFieldStore), a call, an offset a branch can land on, and a write
     * to the local the entry names. */
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
    /* What the BYTECODE says the operand stack is at each offset, from
     * jaiChunkStackDepths -- an oracle this file did not write, checked against
     * every deopt record. NULL only when the chunk would not verify, which is
     * already impossible by the time anything is compiled. See modelAgreesWithChunk. */
    const int *chunkDepth;
    int        chunkDepthCount;

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
    uint8_t   iterKind;   /* 1 a unit-step range, 2 a list, 3 a dict-items view */
    Value     elemSample;
    /* The sampled element's class is one of several the list holds, so the
     * loop variable is an instance of no particular class. Set by the driver,
     * which is the only place the whole list is in hand. */
    bool      elemMixed;
    /* The ListStore each list local is backed by, pinned at compile time and
     * re-checked by the entry guard -- the element loads are emitted at one
     * width, so this is as much a commitment as an instance slot's class. The
     * function tier samples nothing and leaves every entry BOXED, which is the
     * storage a list built by compiled code always has. */
    /* Storage of the list an iterKind 2 head walks, from the same sample. */
    uint8_t   elemStg;
    /* Which of those pins the emission is allowed to BELIEVE. Decided once,
     * between the measuring pass and the real one, because the measuring pass
     * is what says where a slot is written and where the body calls out --
     * planHoists settles its own question in the same place for the same
     * reason. False everywhere in the probe, which therefore emits the boxed
     * form; the two passes already differ over hoisting. */
    bool      localStgPin[JIT_MAX_SLOTS + 1];
    bool      elemStgPin;
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
    int       exitStub[JIT_MAX_EXIT];
    uint32_t  exitOffset[8];
    unsigned  exitCount;
    /* A body that reads an upvalue needs the closure itself -- not any slot (a method's slot 0 is the
     * receiver, not the callee) -- so it arrives as one extra argument, in the register just past the locals. */
    bool      usesUpvalues;
    /* Set once the body has written to the heap. A deopt after that is still
     * fine -- it resumes AT an instruction, so the writes before it are not
     * re-run -- which is what makes the retired bail path unnecessary. */
    bool      wroteHeap;
    /* The instruction being compiled is inside a protected region of the
     * function's static exception table (spec §3.8) -- i.e. inside a `try`.
     *
     * A raise from compiled code unwinds from a frame the interpreter never
     * pushed (function tier) or from the loop head rather than the faulting
     * instruction (OSR tier), so vmThrow consults the wrong offset and the
     * function's OWN handler is skipped. Nothing used to check this because no
     * function containing a `try` could compile at all -- every catch block
     * holds OP_GET_EXC, which had no arm and declined the whole function. The
     * unarmed-opcode deopt below removes that accident, so the rule is now
     * explicit: inside a protected region an overflow resumes at its
     * instruction (branchOnDeoptInstStart) and lets the interpreter raise it,
     * and anything else that can raise -- a call, a list growth -- declines. */
    bool      inProtected;
    /* Every operand-stack entry was in its own register at the top of the
     * instruction being compiled -- nothing deferred, nothing borrowed. See
     * branchOnDeoptInstStart, whose record describes entries the arm may
     * already have popped. */
    bool      instClean;
    /* Baked globals share the defining module's table, so one `keyVersion` guard covers all of them --
     * it changes only when a live entry's address or key could move (new key, rehash, delete, clear). ObjModule::version (used by the function-tier entry check) is neither necessary nor sufficient here. */
    JaiTable *globalsTable;
    uint32_t  globalsKeyVersion;
    /* Same plan, one class's `statics` table (OP_GET_FIELD's SLOT_CLASS arm):
     * a body that reads static fields off two different classes declines the
     * second rather than pretend one table's keyVersion stands for both. */
    JaiTable *staticsTable;
    uint32_t  staticsKeyVersion;
    bool      callsOut;
    /* Set the moment anything is emitted that can destroy x0..x8 while the
     * body is still running: a call out, or one of the two stubs that call and
     * then branch BACK into the body (list grow, self-call slow path). It is
     * what `scratchValues` below is decided from, and the measuring pass is
     * what observes it. */
    bool      clobbersScratch;
    /* WHERE each of those sites was, in bytecode offsets, so the same question
     * can be asked of one loop instead of the whole body: a nest whose inner
     * loop calls nothing still owns x0..x8 and x13..x17 *inside* that loop,
     * however many times the outer one calls. Recorded by the measuring pass
     * and copied into the real one. Overflowing the array sets `clobberSpill`,
     * which answers "yes, everywhere" -- the whole-body answer, so running out
     * of room costs a hoist and never a wrong one. */
    uint32_t  clobberOff[JIT_MAX_CLOBBER];
    unsigned  clobberCount;
    bool      clobberSpill;
    /* The deepest the operand stack ever was at one of those sites. Every entry
     * live when a helper runs is below it, so every entry AT or ABOVE it is
     * provably never live across a call -- which is the whole condition for
     * putting one in a caller-saved register. Unlike a region, this is an
     * index, so where an entry lives stays a function of its index alone and
     * every join, deopt record and stub keeps agreeing about it for free. */
    unsigned  clobberDepth;
    /* The operand stack lives in x0..x8 rather than above the locals in the
     * callee-saved bank. Sound exactly when nothing can clobber a caller-saved
     * register between a push and its use, i.e. `!clobbersScratch` and the
     * whole stack -- an inlined body's entries included, which is what
     * maxValueAll counts -- fits the nine. Worth having because
     * the two banks were competing for the same ten registers: a stencil whose
     * expression is seven deep left NOTHING for its four row pointers and its
     * index, and reloaded all five from the frame every iteration. */
    bool      scratchValues;
    /* The same idea one granularity down, for a body that DOES call. Entries
     * below `splitAt` stay in the callee-saved bank; entries at or above it
     * live in x0..x8. `splitAt` is the deepest the operand stack ever is at a
     * clobber site (Emit::clobberDepth), so an entry at or above it is one no
     * helper can ever be running underneath -- which is the whole soundness
     * condition, stated about an entry's live range rather than about the
     * body. Zero means no split.
     *
     * A split INDEX rather than a region is what keeps this cheap: where an
     * entry lives stays a function of its index alone, so joins, deopt records
     * and stubs all keep agreeing about it without being told anything. The
     * price is that the operand stack is no longer one run of registers, and
     * every site that used to add an index to a base has to say `valueBankReg`
     * instead. */
    unsigned  splitAt;
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
        /* The counted-range head of the loop this hoist covers, when it has
         * one. Per hoist rather than per form because the loop that matters is
         * often INNER: a stencil's outer `for i` assigns the row locals, so
         * nothing hoists there, while the inner `for j` hoists all four and is
         * where every subscript actually happens. */
        bool     rangeOk;
        uint16_t rVar, rCur, rEnd;
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
    /* Somewhere to build a reason that names a number. `whyNot` is otherwise
     * always a literal, and "a local was given two different kinds" without
     * saying which one left the reader to guess -- a poor trade for ninety
     * bytes that live exactly as long as the emitter does. */
    char        whyBuf[96];
    /* WHICH of an arm's unnamed refusals fired, and on what value.
     *
     * `whyNot` is a decision: an arm that names one has told the reader
     * something durable. Several arms instead return false from a dozen places
     * without naming any of them, and `JAI_JIT_WHY` then prints the bare
     * opcode -- so one census row like "OP_GET_FIELD_LOCAL 129" is a merge of
     * eleven unrelated causes wearing one label, and the fixes they want are
     * nothing alike. This is that missing half: printed only when `whyNot` is
     * NULL, never read by control flow, and cleared at the top of every
     * instruction so a note cannot outlive the arm that wrote it.
     *
     * Inert by construction and checked as such: with it set at every site
     * below, the decline TOTAL is unchanged run for run. */
    char        whySub[80];
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
    /* Diagnostic only (JAI_JIT_CHAIN=1). Offsets this walk must not try to arm,
     * so that a body which refuses at one instruction can be walked PAST it to
     * find what it would refuse at next.
     *
     * A refusal is a chain, and the single most expensive question about this
     * tier is "what would this body stop at next?" -- answered until now by
     * building the fix and re-running, which is a day per link and how three
     * separate changes came to measure exactly zero. Forcing the unarmed path
     * at a known offset and recompiling answers it in a second, and it reuses
     * a well-tested mechanism rather than continuing a walk whose model has
     * gone inconsistent (which segfaults). */
    uint32_t  chainSkip[JIT_MAX_CHAIN];
    unsigned  chainSkipCount;
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
    /* Which entries are known to be a LOCAL plus a constant, and which local
     * and what constant. Not a value like kKnown -- a shape. It exists so a
     * subscript can say "this index is the loop counter, minus one" and have
     * its bounds check hoisted to the loop head, where one compare covers
     * every iteration instead of two instructions per element.
     *
     * Only int-kinded entries carry it, and only while the arithmetic stays
     * exact: an add or subtract that could overflow drops the shape rather
     * than describing a value the loop head's guard would not cover. Cleared
     * with every other per-entry mask on push and pop. */
    /* Where each admitted PIC arm jumps to reach the merge. One per way, and
     * the caller patches them all once the descriptor path it emits after the
     * chain is behind it. */
    int       picExits[JAI_IC_WAYS];
    unsigned  picExitCount;

    uint32_t  idxKnown;
    uint8_t   idxBase[32];
    int32_t   idxOff[32];
    /* The span of offsets each list slot is subscripted at, over the sites
     * whose index turned out to be the loop counter plus a constant. The
     * measuring pass fills it; the real pass turns it into ONE compare at the
     * loop head and lets those sites skip their own. A slot with no shaped
     * site keeps `spanLo > spanHi`, which is how "nothing to hoist" reads. */
    int32_t   spanLo[JIT_MAX_SLOTS + 1];
    int32_t   spanHi[JIT_MAX_SLOTS + 1];
    bool      spanOk[JIT_MAX_SLOTS + 1];
    /* Which loop variable the span is measured against. Two loops subscripting
     * the same list off different variables cannot share one span, so the
     * second one seen turns the slot off rather than widening it. */
    uint8_t   spanBase[JIT_MAX_SLOTS + 1];
    bool      spanSeen[JIT_MAX_SLOTS + 1];
    /* The OSR loop is a counted range whose head is OP_FOR_RANGE_BIND, and
     * these are its three slots: the variable the body reads, the counter, and
     * the end. The last two are fresh temporaries the emitter hands out per
     * loop and nothing else writes -- see the arm for OP_FOR_RANGE_BIND --
     * which is what makes the end a loop invariant a single compare can use.
     * Decoded once at setup because the guard is emitted ABOVE the head, before
     * the walk reaches the instruction that would otherwise name them. */
    bool      rangeHead;
    uint16_t  rangeVar, rangeCur, rangeEnd;
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
/* The ObjIter a dict-items head is walking, and the ONLY register that head
 * reserves: it keeps no index and no limit, because the step reads both out of
 * the iterator and writes the index straight back, so nothing has to be
 * unwound at an exit. Numbered second so a dict head's locals start where a
 * range head's index would have been -- see osrReserved. */
#define JIT_PAIR_ITER_REG (JIT_FIRST_SAVED + 1u)

/* OSR reserves only the slots pointer plus (for a range loop) the iterator's index and limit --
 * not the ObjIter or the start, which are folded in via a prologue bias. Bias is sound only while start+limit fits int64; jaiJitEnterOsr refuses entry otherwise since that's a property of the iterator, not the code. */
static unsigned osrReserved(const Emit *e) {
    if (!e->hasIter) return 1u;
    if (e->iterKind == 1) return 3u;
    if (e->iterKind == 3) return 2u;   /* dict items: the ObjIter, nothing else */
    return 5u;
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
 * by the accessors during the measuring pass and weighted by loop nesting. Zero-saving slots are excluded outright, not just ranked last -- otherwise a tie-break could hand a register to a slot the loop never touches, paying a prologue write for nothing.
 *
 * Both tiers feed this, but with their OWN instruction counts, which is why the
 * numbers at the call sites differ rather than the mechanism. A function-tier
 * frame home is a bare payload word for every slot the plan will consider (only
 * a dynamic slot stores a tag, and a dynamic slot never gets a register), so a
 * write costs one `str` there and makes a weak case for a register -- weak
 * enough that a float write makes none at all, see localOutFp. An OSR home is
 * the INTERPRETER's own Value slot, so the same write stores tag and payload
 * both and costs three; in OSR a write is worth as much as two reads. Ranking
 * OSR on the function tier's numbers would therefore under-rank exactly the
 * write-heavy loop variables OSR exists to speed up. */
static void noteSlotCost(Emit *e, unsigned slot, unsigned saveX,
                         unsigned saveFp) {
    if (!e->measuring || e->inlining) return;
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
static void emitConst64(Emit *e, unsigned rd, int64_t value);

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
        /* An X home answers with no instruction at all, an FP home costs the
         * `fmov` below and memory costs the `ldr` -- so an X register saves
         * one here and an FP register saves nothing, exactly as in the
         * function tier's arm. */
        noteSlotCost(e, slot, 1u, 0u);
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

        /* VAL_OBJ is shared by SLOT_LIST, SLOT_OBJ and SLOT_INST alike (see
         * localTagFor), so the tag check above cannot tell a list from a dict
         * a sibling write left in this slot -- confirmed the same way
         * OP_GET_INDEX's own SLOT_LIST arm does, once, before a consumer
         * trusts it with no check of its own. Chained through `scratch`
         * alone (no second register): the payload is reloaded fresh into it,
         * then `Obj.type` is loaded from that address back into the same
         * register -- valid on this encoder elsewhere (e.g. the
         * OP_GET_INDEX/OP_SET_INDEX list arms chain JIT_SCRATCH_C the same
         * way), and it never needs the pointer again afterward, since the
         * unconditional reload below re-reads it from the frame regardless. */
        if (e->localKind[slot] == SLOT_LIST) {
            emit(e, jaiA64LdrX(scratch, 31, localFrameOff(e, slot) + 8));
            emit(e, jaiA64LdrW(scratch, scratch, (unsigned)offsetof(Obj, type)));
            emit(e, jaiA64SubsXImm(31, scratch, OBJ_LIST));
            branchOnDeopt(e, JAI_A64_NE);
        } else if (e->localKind[slot] == SLOT_INST ||
                   e->localKind[slot] == SLOT_MAYBE_INST) {
            /* Same hazard, for a slot that took two different classes: a tag
             * of VAL_OBJ says "an instance", not "an instance of THIS class",
             * so the object type and its class shape are both confirmed here
             * before a consumer reads a field at an offset only this class
             * has. JIT_SCRATCH_D is free at every call site that can carry an
             * instance-kinded dynamic local through this helper (audited:
             * OP_GET_LOCAL, OP_GET_LOCAL2, OP_GET_FIELD_LOCAL's two reads of
             * its receiver, the init-returns-self arms of
             * OP_RETURN_NULL/OP_POP_RETURN_NULL, emitRootFill's root loop --
             * every other site names a scalar kind and cannot reach here).
             * `scratch` keeps the instance pointer as the base throughout --
             * loading FROM it doesn't clobber it -- until it is chained into
             * the class pointer and then the shapeId, since nothing after
             * this needs the original pointer back (the unconditional reload
             * below restores it for the return regardless).
             *
             * SLOT_MAYBE_INST shares the arm rather than going unchecked: it
             * is a kind a dynamic slot really does take, both from the
             * nullable-parameter seed and from any OP_BIND of a nullable
             * instance field, and its tag is the same VAL_OBJ, so without
             * this a `Bird` left in the slot by a sibling write was read at
             * `Dog`'s field offsets. What it does NOT share is the pointer
             * being known non-null. A null cannot be chased and cannot be
             * jumped over either -- every branch this file emits leaves the
             * block for a stub -- so it deopts, and the interpreter finishes
             * the instruction. That costs one deopt on a value the tier could
             * in principle have carried, which is the price of not letting a
             * guard load off address zero. (The prologue's own tag write is
             * payload-dependent for the same reason, so a null argument is
             * usually already stopped by the tag check above.) */
            emit(e, jaiA64LdrX(scratch, 31, localFrameOff(e, slot) + 8));
            if (e->localKind[slot] == SLOT_MAYBE_INST) {
                emit(e, jaiA64SubsXImm(31, scratch, 0));
                branchOnDeopt(e, JAI_A64_EQ);
            }
            emit(e, jaiA64LdrW(JIT_SCRATCH_D, scratch, (unsigned)offsetof(Obj, type)));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_D, OBJ_INSTANCE));
            branchOnDeopt(e, JAI_A64_NE);
            emit(e, jaiA64LdrX(scratch, scratch, (unsigned)offsetof(ObjInstance, klass)));
            emit(e, jaiA64LdrW(scratch, scratch, (unsigned)offsetof(ObjClass, shapeId)));
            emitConst64(e, JIT_SCRATCH_D, (int64_t)e->localShape[slot]);
            emit(e, jaiA64SubsXReg(31, scratch, JIT_SCRATCH_D));
            branchOnDeopt(e, JAI_A64_NE);
        }
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

/* Retires every field-kind memo (Emit::known, written by recordFieldStore and
 * read by knownFieldKind). An entry says "a store THIS body emitted put kind K
 * in that field, and nothing since has put another kind there" -- which holds
 * only while this walk is the only thing that can have run, and only along the
 * fall-through edge that made it. Two things end that, and neither is a store
 * the walk can see:
 *
 *   * a call. The callee reaches the same object through the argument it was
 *     handed, through a global or upvalue, or as the receiver it was invoked
 *     on, and `o.v = "s"` in there leaves the memo claiming SLOT_INT -- so the
 *     read after the call skips its tag guard and hands a consumer an
 *     ObjString pointer as an integer, or an integer as a pointer to
 *     dereference. Hooked in noteScratchClobber, which every call out passes
 *     through.
 *
 *   * an offset something other than fall-through can reach. The store may sit
 *     on the arm a branch skipped (`if c { o.v = 7 }` then a read), and a back
 *     edge re-enters above stores further down the body (a read at a loop top
 *     whose second iteration meets the kind the bottom of the body stored).
 *     Hooked in the walk, on the same offsetIsBranchTarget test the ASCII
 *     proofs above it use, and for the same reason.
 *
 * Clearing all of them rather than one is deliberate: a call can write any
 * field of any object it can reach, so there is nothing narrower to say. */
static void forgetFieldKinds(Emit *e) { e->knownCount = 0; }

/* The narrower one: an entry names a LOCAL, so writing that local retires it.
 * The slot may now hold a different object entirely -- `b.v = 7; b = c; b.v`
 * read c's field at b's recorded kind before this existed. */
static void forgetFieldKindsOfLocal(Emit *e, unsigned slot) {
    unsigned out = 0;
    for (unsigned i = 0; i < e->knownCount; i++) {
        if (e->known[i].local == (int)slot) continue;
        e->known[out++] = e->known[i];
    }
    e->knownCount = out;
}

/* Every write to a local goes through localOut or localOutFp, so recording it
 * in those two places is what makes "this slot does not change inside that
 * loop" a fact about the emitter rather than a re-reading of the bytecode. */
static void noteSlotWrite(Emit *e, unsigned slot) {
    /* Above the range check on purpose: the memo is keyed on the same slot
     * numbers, so a slot too high to record is still one to retire. */
    if (e->knownCount != 0) forgetFieldKindsOfLocal(e, slot);
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
/* Records that `slot` was subscripted at the loop counter plus `off`, or --
 * when `shaped` is false -- that one of its subscripts is not a shape the loop
 * head can guard. One unshaped site does not spoil the others: they still skip
 * their checks and it still emits its own. What it DOES spoil is nothing, so
 * `spanOk` exists only to record that a slot was seen at all. */
static void noteIndexSpan(Emit *e, int slot, bool shaped, int32_t off,
                          uint8_t base) {
    if (!e->measuring || e->inlining) return;
    if (slot < 0 || slot > (int)JIT_MAX_SLOTS) return;
    if (!shaped) return;
    if (!e->spanSeen[slot]) {
        e->spanSeen[slot] = true;
        e->spanBase[slot] = base;
    } else if (e->spanBase[slot] != base) {
        e->spanOk[slot] = false;
        return;
    }
    if (off < e->spanLo[slot]) e->spanLo[slot] = off;
    if (off > e->spanHi[slot]) e->spanHi[slot] = off;
}

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
        /* The memory arm below is three instructions -- the tag built, the tag
         * stored, the payload stored -- against one `mov` into an X home or one
         * `fmov` into an FP home. Two saved either way, so a write votes for a
         * register without voting for a bank; the reads pick the bank. */
        noteSlotCost(e, slot, 2u, 2u);
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
        /* Mirrors the function tier's charge for the same read: the FP home is
         * the one that pays, and an X home is worth no more than the load. */
        noteSlotCost(e, slot, 0u, 1u);
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
        /* Unlike the function tier's float write just below -- which is one
         * instruction into any of the three homes and so charges nothing --
         * the OSR memory arm here writes the tag as well, so this write really
         * does buy two, the same as localOut's. */
        noteSlotCost(e, slot, 2u, 2u);
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

/* The ListStore the frame's slot is backed by, or BOXED for anything else.
 *
 * Read one slot at a time, and only once the walk has decided the slot holds a
 * LIST. Sweeping them all up front is what the first version did, and it
 * segfaulted the COMPILER: an OSR frame's slots run to fn->maxSlots, and the
 * ones the program has not reached yet hold whatever the last frame at that
 * depth left behind -- including a VAL_OBJ tag over a null pointer. IS_LIST
 * dereferences without checking, so the sweep read `Obj::type` off address
 * zero. Once in eight runs of tests/bench/jaiframe/frameops, and never in
 * 3041 tests, because it needs a slot that is both stale and untouched. */
static uint8_t localStgOf(const Emit *e, unsigned slot) {
    if (slot > JIT_MAX_SLOTS || e->observed == NULL) return LIST_STORE_BOXED;
    if (e->localKind[slot] != SLOT_LIST) return LIST_STORE_BOXED;
    Value v = e->observed[slot];
    if (!IS_LIST(v)) return LIST_STORE_BOXED;
    return AS_LIST(v)->stg;
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
 * own bank (x0..x8, minus the emitter's x9..x12 scratches) whenever the caller's stack is NOT already
 * there: it can't call anything, so every caller-saved register is free and costs the caller nothing -- `evalA` inlined into spectral's inner loop needs eight live values where the OSR form had only six left, so without this it wouldn't fit. When the caller IS on that bank the two share one numbering instead; see inlineOwnBank. */
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
    unsigned saved = taken < JIT_MAX_SAVED ? JIT_MAX_SAVED - taken : 0u;
    /* Split: `splitAt` entries in what is left of the callee-saved bank and
     * nine more above them. If the callee-saved half does not itself fit --
     * which the plan sized it not to, but `usesUpvalues` is decided later --
     * the answer is that smaller number, so the push declines rather than
     * running off the end of the bank. */
    if (e->splitAt != 0) {
        return saved < e->splitAt ? saved
                                  : e->splitAt + JIT_SCRATCH_BANK_COUNT;
    }
    return saved;
}

/* Something about to be emitted can destroy x0..x8 with the body still live:
 * a call out, or one of the two stubs that call and then branch back in. The
 * measuring pass records it so the real pass can choose a bank; the real pass
 * declines if it happens anyway, so a missed site costs a decline and never a
 * value read out of a register a helper overwrote. */
static void noteScratchClobber(Emit *e) {
    e->clobbersScratch = true;
    /* The one place every call out passes through, which makes it the one
     * place a field-kind memo can be retired at all of them; see
     * forgetFieldKinds. The sites that reach here without running user code
     * (a list grow, an instance allocation) only over-retire, which costs the
     * tag guard the read would have emitted anyway. */
    forgetFieldKinds(e);
    if (e->measuring) {
        /* An inlined body's offsets are the CALLEE's, so they say nothing
         * about where in the caller this sits; `inlIp` is the caller's own
         * OP_CALL, which is the offset every range in this file is measured
         * in. An inlined body is not supposed to reach here at all (see
         * inlineGlobalCall) -- but "not supposed to" is not "cannot", and a
         * callee offset landing inside a caller loop by coincidence would
         * over-report, not under-report, which is the safe direction. */
        uint32_t at = e->inlining ? e->inlIp : e->curOffset;
        if (e->clobberCount < JIT_MAX_CLOBBER) {
            e->clobberOff[e->clobberCount++] = at;
        } else {
            e->clobberSpill = true;
        }
        if (e->valueDepth > e->clobberDepth) e->clobberDepth = e->valueDepth;
    }
    if (e->scratchValues) {
        e->whyNot = "a call reached a body whose values are in scratch";
        e->failed = true;
    }
    /* Same ratchet one granularity down. A hoisted header sits in a
     * caller-saved register for the length of a loop the measuring pass called
     * call-free; a call turning up inside that loop in the real pass means the
     * two walks disagreed, and the header would be read out of a register the
     * helper had overwritten. It cannot happen -- both walk the same bytecode
     * with the same inlining -- so this costs a compile, never an answer. */
    if (!e->measuring && e->hoistCount > 0) {
        uint32_t at = e->inlining ? e->inlIp : e->curOffset;
        for (unsigned i = 0; i < e->hoistCount; i++) {
            if (at < e->hoist[i].top || at >= e->hoist[i].end) continue;
            e->whyNot = "a call reached a loop a header was hoisted out of";
            e->failed = true;
            return;
        }
    }
    /* And the ratchet the split bank rests on. `splitAt` was chosen as the
     * deepest the measuring pass ever saw the stack at one of these, so a real
     * pass standing deeper means the two walks disagreed and an entry in
     * x0..x8 is about to be run over by the helper. Costs a compile. */
    if (!e->measuring && e->splitAt != 0 && e->valueDepth > e->splitAt) {
        e->whyNot = "a call stood deeper than the split bank allows";
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

/* Where entry `idx` of the ordinary (non-inlined) operand stack lives. One run
 * of registers unless the bank is split, in which case the entries at and above
 * `splitAt` continue in x0.. instead. Every site that used to write
 * `valueBankBase(e) + idx` says this now, so the split cannot be half-applied:
 * with splitAt == 0 the two are the same expression. */
static unsigned valueBankReg(const Emit *e, unsigned idx) {
    if (e->splitAt != 0 && idx >= e->splitAt) {
        return JIT_INL_BANK + (idx - e->splitAt);
    }
    return valueBankBase(e) + idx;
}

static unsigned valueXReg(const Emit *e, unsigned idx) {
    if (inlineOwnBank(e) && idx >= e->inlValueBase) {
        return JIT_INL_BANK + (idx - e->inlValueBase);
    }
    return valueBankReg(e, idx);
}

/* One past the top entry's register. Expressing it this way (not from the bottom) is what keeps
 * `pushReg(e) - 1` == "the entry just pushed" true across both register banks.
 *
 * Subtracting TWO or more from it is not safe under a split bank -- the entry
 * below the top may be in the other half -- so those sites name the entry they
 * mean, as `valueXReg(e, e->valueDepth - n)`. */
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
    e->stackNullLit[e->depth] = false;
    e->stackUnit[e->depth]  = false;
    e->stackObjType[e->depth] = 0;
    e->stackElem[e->depth] = NULL_VAL;
    e->stack[e->depth++] = kind;
    e->fpLive   &= ~(1u << e->valueDepth);
    e->fpBorrow &= ~(1u << e->valueDepth);
    e->kPend    &= ~(1u << e->valueDepth);
    e->kKnown   &= ~(1u << e->valueDepth);
    e->xBorrow  &= ~(1u << e->valueDepth);
    e->idxKnown &= ~(1u << e->valueDepth);
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

/* The kind a field is known to hold, or SLOT_SELF for "not known". Trusting an
 * answer here is skipping a tag guard, so what makes it safe is not this
 * lookup but the four things that retire an entry: recordFieldStore below,
 * plus forgetFieldKinds (a call, a branch target) and forgetFieldKindsOfLocal
 * (a write to the local named). */
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
    e->stackNullLit[e->depth] = false;
    e->stackUnit[e->depth]  = false;
    e->stackObjType[e->depth] = 0;
    e->stackElem[e->depth] = NULL_VAL;
    e->stack[e->depth++] = SLOT_SELF;
    return true;
}

/* Retire every proof about an operand-stack entry -- "came out of the ASCII
 * table", "is a variant's shared unit value" and "is the null literal" alike.
 * See Emit::stackAscii. */
static void clearStackProofs(Emit *e) {
    for (unsigned i = 0; i < JIT_MAX_STACK; i++) {
        e->stackAscii[i]   = false;
        e->stackUnit[i]    = false;
        e->stackNullLit[i] = false;
    }
}

static bool anyStackProof(const Emit *e) {
    for (unsigned i = 0; i < e->depth; i++) {
        if (e->stackAscii[i] || e->stackUnit[i] || e->stackNullLit[i]) {
            return true;
        }
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
    e->idxKnown &= ~(1u << e->valueDepth);
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

/* Removes the callee entry a builtin leaves under its result. That entry holds
 * no register, so copying the result's entry down over it and shortening the
 * stack leaves the result on top, still in the register it was computed into. */
static void dropCalleeEntry(Emit *e) {
    e->stack[e->depth - 2]      = e->stack[e->depth - 1];
    e->stackShape[e->depth - 2] = 0;
    e->stackClass[e->depth - 2] = NULL;
    e->stackSeen[e->depth - 2]  = NULL_VAL;
    e->stackLocal[e->depth - 2] = -1;
    /* The proofs travel down with the entry they are about. Leaving them
     * behind would let a claim made about the callee slot be read as one
     * about the result that replaced it. */
    e->stackAscii[e->depth - 2] = e->stackAscii[e->depth - 1];
    e->stackUnit[e->depth - 2]  = e->stackUnit[e->depth - 1];
    e->stackObjType[e->depth - 2] = e->stackObjType[e->depth - 1];
    e->stackElem[e->depth - 2] = e->stackElem[e->depth - 1];
    e->depth--;
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
    /* MOVZ where the first non-zero chunk is, not always at chunk 0: it zeroes the other three either way,
     * so a `movz rd,#0` under a single `movk` was one instruction spent writing nothing. Every float constant has this shape -- IEEE-754 puts sign, exponent and the leading mantissa bits in the TOP chunk -- so `1.0` was two instructions and is now one, everywhere a float literal reaches a register. */
    uint64_t bits = (uint64_t)value;
    unsigned first = 0;
    while (first < 3 && ((bits >> (16 * first)) & 0xffffu) == 0) first++;
    emit(e, jaiA64MovzX(rd, (unsigned)((bits >> (16 * first)) & 0xffffu), first));
    for (unsigned shift = first + 1; shift < 4; shift++) {
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
    if (e->exitCount >= JIT_MAX_EXIT) { e->whyNot = "too many ways out of the loop"; e->failed = true; return FIXUP_EXIT; }
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

/* JAITHON_JIT_MODULE_CALLS=0 turns off the module-member call arm at OP_INVOKE,
 * so the two shapes can be compared inside ONE binary -- alternating two builds
 * cannot be trusted here, since each switch invalidates __jaicache__ and every
 * sample then pays a stdlib recompile.
 *
 * Default ON. Read once: a body compiled with the arm and a body compiled
 * without it must not coexist in one run. */
static bool jitModuleCalls(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_MODULE_CALLS");
        cached = (v != NULL && strcmp(v, "0") == 0) ? 0 : 1;
    }
    return cached != 0;
}

/* JAITHON_JIT_CLASS_CALLS=0 turns off the static-member call arm at OP_INVOKE,
 * for the same reason jitModuleCalls exists: the two shapes have to be
 * comparable inside ONE binary. Alternating two builds is not an A/B here --
 * every switch invalidates __jaicache__ and each sample then pays a stdlib
 * recompile, which is larger than the effect being measured.
 *
 * Default ON. Read once, so a body compiled with the arm and a body compiled
 * without it cannot coexist in one run. */
static bool jitClassCalls(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_CLASS_CALLS");
        cached = (v != NULL && strcmp(v, "0") == 0) ? 0 : 1;
    }
    return cached != 0;
}

/* JAITHON_JIT_SPLIT_STRESS=1 puts the split bank's boundary into every OSR body
 * that can take one, instead of only the ones that pay for it -- the same idea
 * as JAITHON_JIT_DEOPT_STRESS, for the same reason.
 *
 * A split makes the operand stack two runs of registers instead of one, and the
 * failure mode is a site that adds an index to a base and lands one past the
 * end of the first run. That is silent: the value is written to a register
 * nothing reads. It shipped once already -- OP_GET_GLOBAL wrote through
 * `pushReg`, one past the CURRENT top rather than the register the next push
 * lands in, and bitops printed 68720029766 for 999625 on the runs where its
 * loop compiled. It was found by the benchmark differential because bitops
 * happened to be split-eligible AND to read a global at exactly the boundary.
 * Under this flag it would have been found by any of them. */
static bool jitSplitStress(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_SPLIT_STRESS");
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

/* Does the operand model still say what the BYTECODE says at this offset?
 *
 * The model is maintained by ~140 hand-written arms plus two inliners, and
 * until this check nothing asked it to agree with anything: a wrong depth still
 * emits a self-consistent body, because every register is derived from an index
 * and the indices all shift together. What it corrupts is the deopt record --
 * the one part of the model another component reads -- which then hands the
 * interpreter operand entries it has not got. That is not a crash, it is a
 * wrong answer, and the two it produced were `'float' object has no method
 * 'push'` out of lib/std/gui/path.jai (inlineMethod's dry walk left two entries
 * behind) and the same class from a walk resuming past an unarmed deopt with a
 * stale model (see emitUnarmedDeopt). Neither was visible in the generated
 * code; both are one comparison away here.
 *
 * jaiChunkStackDepths is the oracle, and this file did not write it: it is the
 * verifier's own pass, the one the interpreter's stack discipline is defined
 * by. -1 is "no answer" -- an offset no path reaches, one an unmodelled opcode
 * stopped the walk at, or one whose depth came from an imprecise handler seed
 * -- and is never a failure.
 *
 * Asked only of the function tier's own walk. An OSR body's model starts at the
 * loop head rather than at offset 0, so its depth is relative and disagrees by
 * a constant; an inlined body's offsets are the callee's and mean nothing in
 * the caller's table.
 *
 * Declines, so an arm that drifts in future costs coverage and not an answer. */
static bool modelAgreesWithChunk(const Emit *e, uint32_t off) {
    if (e->osr || e->inlining || e->chunkDepth == NULL) return true;
    if (off >= (uint32_t)e->chunkDepthCount) return true;
    int want = e->chunkDepth[off];
    return want < 0 || want == (int)e->depth;
}

/* Names the opcode whose arm let the borrow through. Which arm it was is the
 * whole question when this fires, and without the name the message only says
 * where the loop started. */
static const char *borrowWhyFor(const Emit *e) {
    static char why[80];
    snprintf(why, sizeof why, "a float borrow reached %s's guard",
             jaiOpName((OpCode)e->lastOp));
    return why;
}

/* Take the record without emitting the branch to it. The model is what it is
 * at this moment, so a site whose *code* is emitted later -- a self-call's
 * cold block, which lives with the stubs -- still has to record here. */
static bool deoptRecordAt(Emit *e, uint32_t ip, bool lastFromDesc,
                          unsigned *out) {
    /* Assertion, not the fix: a deopt stub writes every entry out of fpRegAt, so nothing may still be
     * borrowing a local's register here. Releasing HERE (rather than at the top of the instruction) was tried and is wrong -- a guard can sit inside a span an earlier branch skips (emitBoundsNormalise's does), so the fmov landed on a not-taken path and matrix_mul read `sum` from a register nothing had written. Declines rather than miscompiles if fpBorrowSurvives let something through it shouldn't have. */
    if (e->fpBorrow != 0) {
        /* Names the opcode: which arm let the borrow through is the whole
         * question, and without it the message only says where the loop
         * started. */
        static char borrowWhy[80];
        snprintf(borrowWhy, sizeof borrowWhy, "a float borrow reached %s's guard",
                 jaiOpName((OpCode)e->lastOp));
        e->whyNot = borrowWhy;
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
        e->whyNot = borrowWhyFor(e);
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

/* A guard that resumes at the START of the instruction being compiled, whatever
 * that instruction's arm has already popped.
 *
 * deoptRecordAt refuses this case ("a guard resumes an instruction whose
 * operands it has already consumed") because the entries below `depth` are the
 * only ones it will describe. They are still readable here: popValue moves
 * `depth` and `valueDepth` and nothing else, so entry i's kind is still in
 * stack[i] and its register is still valueXReg(i) -- the mapping is positional.
 * What is NOT guaranteed is that the arm has left those registers alone, which
 * is why this is not a general facility: its one caller is the overflow guard
 * inside a `try`, and the arms that reach it compute into a scratch (ovfDest)
 * so nothing an entry lives in, and no local, has been written when it fires.
 *
 * fpLive is read as it stands rather than as it was: popValue calls fpSyncOne
 * first, so an entry whose bit this instruction cleared has its X register
 * current, which is exactly what the stub then writes out. */
static void branchOnDeoptInstStart(Emit *e, unsigned cond) {
    if (e->fixupCount >= JIT_MAX_FIXUPS) { e->failed = true; return; }
    if (e->fpBorrow != 0) {          /* see deoptRecordAt */
        e->whyNot = borrowWhyFor(e);
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
    /* The record below describes entries this instruction may already have
     * popped, and popValue clears the bit that said an entry was only ever a
     * borrow of a local's register. So the question has to be asked of the
     * instruction's START, where the walk recorded it. */
    if (!e->instClean && !e->inlining) {
        e->whyNot = "a raise resumes an instruction whose operands were borrowed";
        e->failed = true;
        return;
    }
    unsigned k = e->deoptCount++;
    if (e->inlining) {
        /* Inside an inline the interpreter has not made the call yet, so the
         * only resume point is the caller's OP_CALL -- which deoptSite already
         * answers, and which is already "the start of an instruction". */
        if (!deoptSite(e, e->curOffset, &e->deopt[k].ip, &e->deopt[k].depth,
                       &e->deopt[k].valueDepth)) {
            e->failed = true;
            return;
        }
    } else {
        e->deopt[k].ip         = e->curOffset;
        e->deopt[k].depth      = e->instDepth;
        e->deopt[k].valueDepth = e->instValueDepth;
    }
    e->deopt[k].lastFromDesc = false;
    e->deopt[k].fpLive       = e->fpLive;
    for (unsigned i = 0; i < e->deopt[k].depth; i++) {
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

/* Branch to the bail block on `cond`. The block's index is not known yet, so
 * it is patched with the rest. */
/* NaN comparison is a TypeError here, not false, matching the interpreter's isnan check: fcmp sets V
 * on an unordered result, and that's routed to a deopt so the interpreter raises exactly what it would have. */
static void nanToDeopt(Emit *e) { branchOnDeopt(e, JAI_A64_VS); }

/* `c < "0"` and its three siblings, both operands a string one byte long.
 *
 * Ordering on strings had no arm at all, so one `character >= "0"` declined
 * the whole body around it: json_parse's `integer` is 50% of that benchmark's
 * interpreted instructions and the self-hosted lexer's `_is_digit`/`_is_alpha`
 * 3.5% of a compile's. Only the one-byte shape is compiled -- compareStrings
 * on anything longer is a memcmp, which is a call, and a character-class test
 * is the shape that actually occurs. The sample says one byte and the emitted
 * code GUARDS it, so a longer string arriving later deoptimises rather than
 * being answered wrongly.
 *
 * The bytes go in through LDRB, which zero-extends, so both are 0..255 and the
 * signed conditions the integer arm already computed give memcmp's unsigned
 * answer unchanged -- no separate condition table. */
static bool isOrdering(uint8_t op) {
    return op == OP_LT || op == OP_LE || op == OP_GT || op == OP_GE;
}

static bool stringOperand(const Emit *e, unsigned at) {
    return e->stackAscii[at] || IS_STRING(e->stackSeen[at]);
}

/* Statically one byte: the ASCII table's own singleton, or a sample that is a
 * one-character string -- which for the literal side of a character-class test
 * is the constant-pool entry itself and so cannot be anything else. */
static bool knownOneByte(const Emit *e, unsigned at) {
    Value v = e->stackSeen[at];
    return e->stackAscii[at] || (IS_STRING(v) && AS_STRING(v)->length == 1);
}

/* One side has to be known one byte before this is worth compiling. The other
 * is only guarded, and a sample can lie: OP_GET_INDEX hands its result the
 * RECEIVER as a sample (only the type is read from it), so `s[i]`'s sample is
 * the whole subject string. Without the literal side to anchor on, a general
 * `a < b` over long strings would compile and then deopt every iteration,
 * which is slower than never compiling the body at all. */
static bool oneBytePair(const Emit *e, unsigned a, unsigned b) {
    return stringOperand(e, a) && stringOperand(e, b) &&
           (knownOneByte(e, a) || knownOneByte(e, b));
}

static void emitOneByteString(Emit *e, unsigned at, unsigned reg, unsigned dst) {
    /* What the ASCII table produced is a one-byte interned string by
     * construction, so it needs neither guard -- see the same skip in OP_EQ. */
    if (!e->stackAscii[at]) {
        emit(e, jaiA64LdrW(JIT_SCRATCH_A, reg, (unsigned)offsetof(Obj, type)));
        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_STRING));
        branchOnDeopt(e, JAI_A64_NE);
        emit(e, jaiA64LdrW(JIT_SCRATCH_A, reg,
                           (unsigned)offsetof(ObjString, length)));
        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, 1));
        branchOnDeopt(e, JAI_A64_NE);
    }
    /* `chars` is a pointer, not an inline array: a string may address bytes it
     * does not own. Two loads, and the second is the byte itself. */
    emit(e, jaiA64LdrX(dst, reg, (unsigned)offsetof(ObjString, chars)));
    emit(e, jaiA64LdrByte(dst, dst, 0));
}

/* A/B switch, default on, and not optional: the machine runs several agents at
 * once, so before and after have to be the same binary minutes apart rather
 * than two binaries. JAITHON_JIT_STRCMP=0 puts the arm below back to the
 * decline it was, with nothing else changed. */
static bool jitStrCmpOn(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("JAITHON_JIT_STRCMP");
        on = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return on != 0;
}

/* PROTOTYPE, second switch: send `==`/`!=` to the leaf call when the operands
 * are not KNOWN interned, instead of letting the pointer arm compile a guard
 * that deoptimises on every iteration. Default on; JAITHON_JIT_STRCMP_EQ=0
 * restores the arm exactly as proposed. */
static bool jitStrCmpEqOn(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("JAITHON_JIT_STRCMP_EQ");
        on = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return on != 0;
}

/* A sample that is interned says the site is an interned-string site, which is
 * a prediction the pointer arm's own subFlag guard already checks. What it is
 * used for here is only which arm to emit. */
static bool knownInternedString(const Emit *e, unsigned at) {
    if (e->stackAscii[at]) return true;
    Value v = e->stackSeen[at];
    return IS_STRING(v) && JAI_STR_INTERNED(AS_STRING(v));
}

/* True when the pointer arm should stand aside for the leaf call at this
 * equality site: the leaf call has to be available (same switch, same
 * inlining rule) and at least one operand must not be known interned. */
static bool preferLeafEquality(const Emit *e, unsigned da, unsigned db) {
    if (!jitStrCmpEqOn() || !jitStrCmpOn() || e->inlining) return false;
    return !(knownInternedString(e, da) && knownInternedString(e, db));
}

/* `a <=> b` for two strings whose lengths nothing knows, as a guarded LEAF
 * call. Leaves NZCV set from `cmp x0, #0`, which is the shape every other arm
 * in these switches leaves behind, so the `cset` or the branch after it is
 * unchanged and all six operators come out of the one sequence: the order is
 * -1, 0 or 1, and `a OP b` is `order OP 0` for every one of them.
 *
 * WHY THIS IS NOT THE PREDICTION THE ONE-BYTE ARM REFUSED. That comment says a
 * sample can lie -- OP_GET_INDEX hands its result the RECEIVER as a sample --
 * so a general compare compiled on a guess about LENGTH would deopt every
 * iteration, which is worse than never compiling the body. Nothing here
 * predicts a length. `Obj.type == OBJ_STRING` is EXACT: every string passes it,
 * so there is no deopt loop, and the sample is used only to decide that the
 * arm is worth emitting at all.
 *
 * WHY A LEAF AND NOT A DESCRIPTOR. jaiStringOrder allocates nothing, roots
 * nothing and cannot re-enter the interpreter, so it needs none of the dozen
 * stores, the root fill or the collector-chain link a descriptor call pays --
 * far more than an eleven-byte memcmp costs, and it would have measured zero
 * or worse. Modelled on the jitInstanceAlloc call in emitCallOut, which is a
 * leaf for the same reason.
 *
 * WHAT IT CLOBBERS, and how each is accounted for:
 *   - x0..x17 and x30, per AAPCS64. The operand stack is in x0..x8 whenever
 *     the body is otherwise call-free (Emit::scratchValues) or split
 *     (Emit::splitAt), so this goes through noteScratchClobber like every
 *     other call out: the measuring pass records the site, which turns
 *     scratchValues off for the whole body and puts the split boundary at or
 *     above this depth, and the real pass fails the compile if it reaches here
 *     with values in scratch anyway. It also retires the field-kind memos,
 *     which over-retires (this callee writes no field) and costs a tag guard.
 *   - v16.. -- the FP half of the operand bank is caller-saved on purpose.
 *     fpSyncAll below writes every live float entry back to its X home first.
 *     Float LOCALS are in v8..v15, which the ABI preserves.
 *   - x13..x17, which planHoists spends on loop-invariant list headers. The
 *     same noteScratchClobber recorded the offset, so regionCalls reports this
 *     loop as calling and no header is hoisted out of it; the ratchet in
 *     noteScratchClobber fails the compile if the two passes ever disagree.
 *   - x30, saved by emitFrameEnter at entry on every path.
 * The body is NOT call-free for register planning afterwards, and that is the
 * real price of this arm: a body whose only call is this one loses x0..x8 for
 * its operand stack and can decline for want of registers where it used to
 * compile. Measured anyway -- see docs/agents/string-compare.md. */
static void emitStringOrder(Emit *e) {
    unsigned da = e->depth - 2, db = e->depth - 1;
    /* Settled before anything is read: the guards record deopts, which cannot
     * describe a deferred entry, and the call would destroy a borrowed one. */
    settleAll(e);
    fpSyncAll(e);
    unsigned ra = valueXReg(e, e->valueDepth - 2);
    unsigned rb = valueXReg(e, e->valueDepth - 1);

    /* Both operands, before either is consumed, so a miss resumes at this
     * instruction with the pair still on the interpreter's stack -- and the
     * interpreter then raises whatever the operator raises for the pair it
     * actually has. */
    for (unsigned side = 0; side < 2; side++) {
        /* What the ASCII table produced is a string by construction; see the
         * same skip in OP_EQ. */
        if (e->stackAscii[side == 0 ? da : db]) continue;
        unsigned r = side == 0 ? ra : rb;
        emit(e, jaiA64LdrW(JIT_SCRATCH_A, r, (unsigned)offsetof(Obj, type)));
        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_STRING));
        branchOnDeopt(e, JAI_A64_NE);
    }

    /* Into x0/x1 without assuming where the operands live. The split bank puts
     * entries in x0..x8, and although this call's own depth is what the
     * boundary is chosen from, a `mov` that reads a register it has already
     * written is a miscompile rather than a decline -- so the aliasing is
     * handled instead of argued away. */
    if (rb == 0) {
        emit(e, jaiA64MovX(JIT_SCRATCH_C, rb));
        rb = JIT_SCRATCH_C;
    }
    if (ra != 0) emit(e, jaiA64MovX(0, ra));
    if (rb != 1) emit(e, jaiA64MovX(1, rb));
    emitConst64(e, JIT_SCRATCH_A, (int64_t)(uintptr_t)&jaiStringOrder);
    noteScratchClobber(e);
    emit(e, jaiA64Blr(JIT_SCRATCH_A));
    /* The whole of x0: jaiStringOrder returns int64_t precisely so this does
     * not have to trust the top half of a 32-bit return. */
    emit(e, jaiA64SubsXImm(31, 0, 0));
}

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

/* Proves the list in `rList` is still the boxed Value[] every element access
 * below is emitted against. An unboxed `list[int]` packs its elements eight
 * bytes apart with no tag, so a body compiled for the boxed layout would read
 * two of them as one tagged pair -- and the storage is a property of the LIST,
 * not of the slot, so nothing the walk knows can rule it out.
 *
 * Two instructions, on the boxed path as well. The loop tier pins the storage
 * per slot instead and jaiJitEnterOsr proves it once at entry (JaiOsrForm::
 * kinds), so the sites that can name their slot skip this entirely; this is
 * for the ones that cannot -- a nested iterator's source, a list reached
 * through a field, anything the function tier subscripts. */
static void emitListBoxedGuard(Emit *e, unsigned rList, unsigned scratch) {
    emit(e, jaiA64LdrByte(scratch, rList, (unsigned)offsetof(ObjList, stg)));
    emit(e, jaiA64SubsXImm(31, scratch, LIST_STORE_BOXED));
    branchOnDeopt(e, JAI_A64_NE);
}

static bool subWhy(Emit *e, const char *fmt, ...);

/* The ListStore a site may emit its element accesses against, and whatever
 * guard makes that true.
 *
 * A slot the loop tier pinned needs no runtime check at all -- jaiJitEnterOsr
 * proved it at entry, the same way it proves an instance slot's class, so the
 * stride below is right before the body starts. Anything the walk cannot name
 * a slot for -- a list reached through a field, a temporary, anything the
 * function tier subscripts -- is proved here instead, and BOXED is the only
 * answer those two instructions accept. */
/* How an element access should be emitted.
 *
 * `stg` is the storage the first arm is emitted at. When `dynamic` is set a
 * second arm at `alt` follows it, with a runtime test on ObjList::stg picking
 * between the two -- see listDispatchBegin. */
typedef struct {
    uint8_t stg;
    uint8_t alt;
    bool    dynamic;
} ListAccess;

/* The one unboxed storage a list holding a `vk` could have. BOXED means there
 * is no other possibility: a list of instances, strings or lists is boxed and
 * nothing else, so those sites need no second arm. */
static uint8_t listAltFor(SlotKind vk) {
    switch (vk) {
    case SLOT_INT:   return (uint8_t)LIST_STORE_I64;
    case SLOT_FLOAT: return (uint8_t)LIST_STORE_F64;
    case SLOT_BOOL:  return (uint8_t)LIST_STORE_U8;
    default:         return (uint8_t)LIST_STORE_BOXED;
    }
}

/* Decides between the three ways a site can know its storage, and emits
 * whatever guard the chosen one owes.
 *
 * PINNED. The loop tier sampled the storage from the live frame and
 * jaiJitEnterOsr proves it at entry, the way it proves an instance slot's
 * class. One arm, no test, no tag check: this is the whole point of the
 * unboxing, and it is where the 1.3-1.5x lives.
 *
 * PROVED. Nothing pinned it, and no unboxed storage could hold a `vk` anyway
 * (a list of instances). Two instructions say BOXED or deoptimise, and BOXED
 * is a fact that stays true -- `stg` only ever moves towards boxed.
 *
 * DISPATCHED. Nothing pinned it and two storages are possible. Emitting the
 * boxed arm behind a deopt guard is what the first version did, and it is
 * ruinous rather than merely slower: the guard fails on EVERY access, and each
 * failure leaves the compiled loop and re-enters it. `var f: list[bool] = []`
 * at the top of a loop body is enough to arrange it -- the slot is written, so
 * nothing pins it -- and building a 300k sieve thirty times went from 80ms to
 * 300ms. Two arms and one compare cost four instructions and never leave. */
static ListAccess listAccessFor(Emit *e, unsigned rList, int slot,
                                SlotKind vk, unsigned scratch) {
    ListAccess a;
    if (e->osr && slot >= 0 && slot <= (int)JIT_MAX_SLOTS &&
        e->localStgPin[slot]) {
        a.stg = localStgOf(e, slot);
        a.alt = a.stg;
        a.dynamic = false;
        return a;
    }
    a.alt = listAltFor(vk);
    a.stg = (uint8_t)LIST_STORE_BOXED;
    a.dynamic = a.alt != LIST_STORE_BOXED;
    if (!a.dynamic) emitListBoxedGuard(e, rList, scratch);
    return a;
}

/* Opens the runtime test. Returns the index of the placeholder branch, or -1
 * when the access is static and no second arm follows.
 *
 * Tests for `alt` EXACTLY, not merely for "not boxed". `alt` is the storage a
 * list holding a `vk` would have -- it is a prediction, not a reading, and the
 * list is free to have a third one. `xs[i] = 3` on a `list[bool]` predicts I64
 * from the value's kind while the array is one byte per element, and an
 * eight-byte store at `i << 3` then runs off the end of it. So the third case
 * deoptimises: the interpreter's jaiListPut de-specialises or raises, which is
 * an answer this cannot emit inline. */
static int listDispatchBegin(Emit *e, const ListAccess *a, unsigned rList,
                             unsigned scratch) {
    if (!a->dynamic) return -1;
    emit(e, jaiA64LdrByte(scratch, rList, (unsigned)offsetof(ObjList, stg)));
    emit(e, jaiA64SubsXImm(31, scratch, a->alt));
    int skip = (int)e->count;
    emit(e, jaiA64BCond(JAI_A64_EQ, 0));
    emit(e, jaiA64SubsXImm(31, scratch, LIST_STORE_BOXED));
    branchOnDeopt(e, JAI_A64_NE);
    return skip;
}

/* Closes the first arm and opens the second. Returns the join placeholder. */
static int listDispatchElse(Emit *e, int skip) {
    int join = (int)e->count;
    emit(e, jaiA64B(0));
    e->code[skip] = jaiA64BCond(JAI_A64_EQ, (int32_t)((int)e->count - skip));
    return join;
}

static void listDispatchEnd(Emit *e, int join) {
    e->code[join] = jaiA64B((int32_t)((int)e->count - join));
}


/* log2 of the element stride: a boxed Value is sixteen bytes, an int or a
 * double eight, a bool one. */
static unsigned listStgShift(uint8_t stg) {
    switch ((ListStore)stg) {
    case LIST_STORE_I64:
    case LIST_STORE_F64: return 3;
    case LIST_STORE_U8:  return 0;
    case LIST_STORE_BOXED: break;
    }
    return 4;
}

/* The SlotKind every element of an unboxed store has. The storage IS the type,
 * which is most of what the unboxing buys: the boxed path spends a load, a
 * compare and a branch per element proving what this knows statically. */
static SlotKind listStgKind(uint8_t stg) {
    switch ((ListStore)stg) {
    case LIST_STORE_I64: return SLOT_INT;
    case LIST_STORE_F64: return SLOT_FLOAT;
    case LIST_STORE_U8:  return SLOT_BOOL;
    case LIST_STORE_BOXED: break;
    }
    return SLOT_OPAQUE;
}

/* The same store with the base and index named, for the subscript arm, whose
 * items pointer may be a hoisted register rather than JIT_SCRATCH_C. */
static void emitElemStoreAt(Emit *e, uint8_t stg, unsigned rItems,
                            unsigned rIdx, unsigned vtag, unsigned rVal) {
    emit(e, jaiA64AddXLsl(JIT_SCRATCH_C, rItems, rIdx, listStgShift(stg)));
    if (stg == LIST_STORE_BOXED) {
        emit(e, jaiA64MovzX(JIT_SCRATCH_A, vtag, 0));
        emit(e, jaiA64StrW(JIT_SCRATCH_A, JIT_SCRATCH_C, 0));
        emit(e, jaiA64StrX(rVal, JIT_SCRATCH_C, 8));
    } else if (stg == LIST_STORE_U8) {
        emit(e, jaiA64StrByte(rVal, JIT_SCRATCH_C, 0));
    } else {
        emit(e, jaiA64StrX(rVal, JIT_SCRATCH_C, 0));
    }
}

/* One element store, at whatever width `stg` says, from JIT_SCRATCH_C (items)
 * and JIT_SCRATCH_A (index). Leaves JIT_SCRATCH_C pointing at the element. */
static void emitListElemStore(Emit *e, uint8_t stg, unsigned vtag,
                              unsigned rVal) {
    emit(e, jaiA64AddXLsl(JIT_SCRATCH_C, JIT_SCRATCH_C, JIT_SCRATCH_A,
                          listStgShift(stg)));
    if (stg == LIST_STORE_BOXED) {
        emit(e, jaiA64MovzX(JIT_SCRATCH_D, vtag, 0));
        emit(e, jaiA64StrW(JIT_SCRATCH_D, JIT_SCRATCH_C, 0));
        emit(e, jaiA64StrX(rVal, JIT_SCRATCH_C, 8));
    } else if (stg == LIST_STORE_U8) {
        emit(e, jaiA64StrByte(rVal, JIT_SCRATCH_C, 0));
    } else {
        /* A double's bits and an int's are both the whole register: the tier
         * keeps a SLOT_FLOAT payload in an X register exactly as the boxed
         * store did. */
        emit(e, jaiA64StrX(rVal, JIT_SCRATCH_C, 0));
    }
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

/* Can anything in [lo, hi) destroy a caller-saved register? The measuring pass
 * recorded every site that can (noteScratchClobber), so this is a lookup and
 * not a re-derivation -- which matters, because the set of things that clobber
 * is not the set of things that look like calls: the list-grow stub is an
 * OP_LIST_APPEND, and the self-call slow path is an OP_CALL that never leaves
 * the body. Whatever reaches noteScratchClobber is in here by construction. */
static bool regionCalls(const Emit *e, uint32_t lo, uint32_t hi) {
    if (e->clobberSpill) return true;
    for (unsigned i = 0; i < e->clobberCount; i++) {
        if (e->clobberOff[i] >= lo && e->clobberOff[i] < hi) return true;
    }
    return false;
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

/* Nothing outside [top, end) may branch INTO it. The hoisted loads sit just
 * above the head, so any path that reaches the body without passing through
 * them would run it against registers nobody loaded. Written as "into the
 * range", not "to the head", because the head is only the entrance a
 * structured loop is supposed to have -- this is what makes that a checked
 * fact rather than an assumption about the emitter. */
static bool onlyBackEdgesEnter(const Chunk *c, uint32_t top, uint32_t end) {
    for (int at = 0; at < c->count;) {
        int len = instructionLength(c, at);
        if (len <= 0) return false;
        if ((uint32_t)at >= top && (uint32_t)at < end) { at += len; continue; }
        int rel = jaiOpBranchOperandAt(c->code[at]);
        if (rel >= 0) {
            int32_t to = (int32_t)(at + len) +
                         jaiReadI16(c->code + at + 1 + rel);
            if (to >= (int32_t)top && to < (int32_t)end) return false;
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
 * the guard back -- unless the stretch of code the load is hoisted over can be
 * shown to contain nothing that could move a list at all. Every way a list is
 * resized (push, insert, remove, clear, slice, anything through a descriptor)
 * is a call out, and so is every collection; an in-place `xs[i] = v` moves
 * neither `items` nor `count`. So across a call-free stretch a header is
 * invariant for as long as the LOCAL is, and that is a fact the measuring pass
 * already recorded (noteSlotWrite).
 *
 * That stretch is the CANDIDATE LOOP, not the body. `bodyCalls` -- the whole
 * body -- is what this used to ask, and it is both sound and far too strong:
 * matrix_mul's `k` loop calls nothing, but the `i` loop it sits in pushes a row
 * per iteration, so the body answered "calls" and the innermost loop in the
 * benchmark got nothing. A hoist out of a call-free loop is sound for exactly
 * the same two reasons it was before -- nothing between the load and its uses
 * can move the list, and nothing between them can overwrite a caller-saved
 * register -- because both are statements about the code the value is live
 * across, and that is the loop.
 *
 * Registers come from x13..x17, which the tier otherwise never names, plus
 * whatever the operand stack left unused at the top of the scratch bank. Both
 * are caller-saved, which is what makes them free and also what confines a
 * hoist to a call-free region: a call outside the loop may destroy them, and is
 * welcome to -- by then the hoisted header is dead, and re-entering the loop
 * re-runs the load that sits above its head. The scratch-bank half is offered
 * only under `scratchValues`, which is still a whole-body claim, since those
 * registers are the operand stack's everywhere else in the body. */
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
    if (e->measuring || !e->osr) return;
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
            /* ...and nothing in the loop that could resize the list or take
             * back the registers the header is being put in. */
            if (regionCalls(e, lt, le)) continue;
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
        e->hoist[e->hoistCount].rangeOk  = false;
        uint32_t ht = cand[pick].top;
        if (ht + 9u <= (uint32_t)c->count && c->code[ht] == OP_FOR_RANGE_BIND) {
            e->hoist[e->hoistCount].rangeOk = true;
            e->hoist[e->hoistCount].rVar = jaiReadU16(c->code + ht + 3);
            e->hoist[e->hoistCount].rCur = jaiReadU16(c->code + ht + 5);
            e->hoist[e->hoistCount].rEnd = jaiReadU16(c->code + ht + 7);
        }
        e->hoistCount++;
    }
}

/* Whether the loop head's guard already covers this subscript, so the compare
 * and branch here can go.
 *
 * planHoists proves two of the three things that need to be true: it hoists a
 * slot's header only over a region with no write to the slot and no call that
 * could resize the list, which is exactly what makes `count` a loop invariant
 * worth checking once. The third is that the index IS the loop counter plus a
 * constant (Emit::idxKnown), and that the counter is written nowhere but the
 * loop's own head -- `for j in ...` that assigns to `j` inside the body would
 * otherwise index with a value the head's guard never saw.
 *
 * Restricted to a hoist over the OSR loop ITSELF, because the guard reads
 * JIT_IDX_REG and JIT_LIM_REG: an inner loop nested inside the compiled one
 * has its own counter and those registers describe the outer. */
static bool boundsCoveredAtHead(const Emit *e, int slot, unsigned vidx,
                                int32_t *offOut, uint8_t *baseOut) {
    if (!e->osr) return false;
    if (slot < 0 || slot > (int)JIT_MAX_SLOTS) return false;
    if ((e->idxKnown & (1u << vidx)) == 0) return false;
    unsigned base = e->idxBase[vidx];
    if (base > JIT_MAX_SLOTS) return false;
    /* The index has to be a loop VARIABLE, written by its head's bind and by
     * nothing else: a body that assigns to `j` indexes with a value no head
     * ever bounded. One write site, and it is a range head. */
    if (e->slotWriteLo[base] != e->slotWriteHi[base]) return false;
    uint32_t bindAt = e->slotWriteLo[base];
    if (bindAt >= (uint32_t)e->chunkDepthCount) return false;
    *offOut  = e->idxOff[vidx];
    *baseOut = (uint8_t)base;
    /* The measuring pass answers "is this index a shape", not "is it covered".
     * Two reasons it cannot answer the second: the span it is being asked
     * about is the one it is still building, and hoists do not exist yet --
     * planHoists runs BETWEEN the passes. Both made an earlier version record
     * nothing at all and measure exactly 1.000x. */
    if (e->measuring) return true;
    if (!e->spanOk[slot] || e->spanLo[slot] > e->spanHi[slot]) return false;
    int h = hoistFor(e, slot);
    if (h < 0 || !e->hoist[h].rangeOk) return false;
    /* The guard is emitted at THIS hoist's loop head, so it is that loop's
     * variable the index has to be measured against. */
    if (e->hoist[h].rVar != base) return false;
    if (*offOut < e->spanLo[slot] || *offOut > e->spanHi[slot]) return false;
    return true;
}

/* The loads themselves, emitted just above the head of the loop they were
 * planned out of -- which is where the walk is when it reaches that offset. */
static void emitHoistsAt(Emit *e, uint32_t off) {
    /* `off` is also the resume point for the bounds guard below: the hoists are
     * emitted ABOVE the offset map, so e->curOffset still names the PREVIOUS
     * instruction and a guard taken against it resumes somewhere whose operand
     * model has already been consumed. */
    if (e->inlining) return;
    for (unsigned i = 0; i < e->hoistCount; i++) {
        if (e->hoist[i].top != off) continue;
        /* Outside the loop, so the function tier pays its two instructions
         * once per entry rather than per element. */
        if (!e->osr) {
            emitListBoxedGuard(e, e->slotXReg[e->hoist[i].slot], JIT_SCRATCH_A);
        }
        emitListHeader(e, e->slotXReg[e->hoist[i].slot],
                       e->hoist[i].itemsReg, e->hoist[i].countReg);

        /* And, once, the bounds every subscript of this slot inside the loop
         * would otherwise check for itself. The counter runs [IDX, LIM), so
         * the indices reached are [IDX + spanLo, LIM - 1 + spanHi]; proving
         * both ends here lets each site drop a compare and a branch, which is
         * 10 of the ~39 instructions a five-point stencil executes per cell.
         *
         * Skipped when the loop will not run at all: LIM <= IDX makes the
         * lower end of that interval meaningless, and deoptimising an empty
         * loop would hand the whole rest of the function to the interpreter
         * for no reason. */
        unsigned sl = e->hoist[i].slot;
        if (!e->osr || !e->hoist[i].rangeOk) continue;
        if (!e->spanOk[sl] || e->spanLo[sl] > e->spanHi[sl]) continue;
        if (!e->spanSeen[sl] || e->spanBase[sl] != e->hoist[i].rVar) continue;
        /* localIn, not localHomeX: the counter and the end are temporaries the
         * emitter hands out per loop, and they often miss out on a register
         * entirely. A load apiece is nothing here -- this runs once per entry
         * to the compiled loop, not once per iteration -- and requiring homes
         * made the guard decline on every stencil that has one. */
        unsigned rCur = localIn(e, e->hoist[i].rCur, JIT_SCRATCH_B);
        unsigned rEnd = localIn(e, e->hoist[i].rEnd, JIT_SCRATCH_C);

        /* The counter has not necessarily started at the range's own start --
         * OSR is entered on a back edge, so the loop may be half run -- but
         * that only narrows the interval, and narrowing is sound. Skipped
         * entirely when nothing is left to run: deoptimising an empty loop
         * would hand the rest of the function to the interpreter for nothing. */
        emit(e, jaiA64SubsXReg(31, rCur, rEnd));
        unsigned empty = e->count;
        emit(e, jaiA64BCond(JAI_A64_GE, 0));

        /* emitAddSubImm carries the sign itself, so the span goes in as it
         * stands: cur + spanLo is the first index the loop still reaches, and
         * end - 1 + spanHi the last. */
        emitAddSubImm(e, JIT_SCRATCH_A, rCur, (int64_t)e->spanLo[sl], false);
        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, 0));
        branchOnDeoptAt(e, JAI_A64_LT, off, false);

        emitAddSubImm(e, JIT_SCRATCH_A, rEnd, (int64_t)e->spanHi[sl] - 1,
                      false);
        emit(e, jaiA64SubsXUxtw(31, JIT_SCRATCH_A, e->hoist[i].countReg));
        branchOnDeoptAt(e, JAI_A64_HS, off, false);

        if (empty < e->count && e->count <= JIT_MAX_INSTS) {
            e->code[empty] = jaiA64BCond(JAI_A64_GE,
                                         (int32_t)(e->count - empty));
        }
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

/* An overflow stub reads no operand-stack entry -- it raises -- so the entries
 * do NOT have to be in their X homes to branch to one. The fpSyncAll it used
 * to do was pure hot-path cost: in a float expression carrying an int guard
 * through it (a stencil's `mid[j-1]`), each one wrote every live float out and
 * the next use read it back, four cross-bank fmovs sitting in the middle of the
 * accumulate chain. A join (branchTo) and a deopt record are different and
 * still settle -- the first because the other edge must agree, the second
 * because the record is read.
 *
 * There used to be a `branchOnCondition` here that took a body to its BAIL
 * block on a condition, and one arm used it: an indirect call whose callee
 * came back with a non-zero verdict. That arm takes a deopt at the call offset
 * now (see it for why a bail was unsound in the OSR tier), which leaves the
 * entry stack-limit guard as the only thing that can reach the bail block --
 * and that fires before a single body instruction runs. So a bail can no
 * longer follow a write of any kind, and the `bailAfterWrite` decline that
 * guarded against it went with the function. */

/* `cond` isn't always VS: adds/subs set the overflow flag, but the multiply test compares the
 * product's high half against the low half's replicated sign, so its answer is NE -- routing multiply through VS meant its overflow was never detected (4 * 2^62 silently came back as 0). */
static void branchOnOverflow(Emit *e, unsigned which, unsigned cond) {
    if (e->fixupCount >= JIT_MAX_FIXUPS) { e->failed = true; return; }
    /* Inside a `try` the overflow stub's raise would unwind past the region
     * that should have caught it (see Emit::inProtected). Resume at this
     * instruction instead: the interpreter re-executes it, overflows too, and
     * raises with a frame and an ip that name the right handler. The deopt
     * record IS read, so that path settles the FP bank; the overflow stub is
     * not, which is why the unconditional fpSyncAll above it is gone. */
    if (e->inProtected) {
        fpSyncAll(e);
        branchOnDeoptInstStart(e, cond);
        return;
    }
    e->overflowUsed[which] = true;
    e->fixups[e->fixupCount].instIndex    = (int)e->count;
    e->fixups[e->fixupCount].targetOffset = FIXUP_OVF - which;
    e->fixups[e->fixupCount].conditional  = true;
    e->fixups[e->fixupCount].depth        = -1;
    e->fixupCount++;
    emit(e, jaiA64BCond(cond, 0));
}

/* Where an integer arm that can overflow computes its result.
 *
 * Inside a `try` the overflow guard resumes at the instruction (see
 * branchOnOverflow), so the result must not have reached its home yet -- the
 * interpreter would otherwise apply the operation a second time on top of the
 * wrapped value, and `total += x` inside a caught `try` would come back wrong.
 * A scratch keeps every canonical register untouched until the guard is past;
 * the copy out is a `mov`, which §5 of the roadmap prices at zero on this core,
 * and it only appears inside a protected region at all. */
static unsigned ovfDest(const Emit *e, unsigned home) {
    return e->inProtected ? JIT_SCRATCH_B : home;
}

/* Whether a raise that leaves compiled code with the exception pending can be
 * emitted here. It cannot inside a `try`: the effects already happened, so the
 * site cannot resume at its instruction the way an overflow can, and the
 * unwinder would consult an offset outside the protected region. Declines --
 * which is exactly what the whole function did before OP_GET_EXC had a deopt,
 * so no shape that used to compile stops. */
static bool raiseExitAllowed(Emit *e, const char *what) {
    if (!e->inProtected) return true;
    e->whyNot = what;
    e->failed = true;
    return false;
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

/* The first entry a dict walk would yield, for the component kinds the compiled
 * pair head specialises on -- the dict equivalent of taking items[0] off a list.
 * False for a dict with nothing live in it, which declines rather than guessing.
 * Reads the order array exactly as jaiTableNext does, so "first" here and
 * "first" at run time are the same entry. */
static bool firstLiveEntry(const JaiTable *t, Value *key, Value *value) {
    if (t->entries == NULL) return false;
    for (int i = 0; i < t->orderCount; i++) {
        const int32_t slot = t->order[i];
        if (slot < 0) continue;
        *key   = t->entries[slot].key;
        *value = t->entries[slot].value;
        return true;
    }
    return false;
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

/* JAITHON_JIT_STATIC_FIELD=0 turns the SLOT_CLASS arm of OP_GET_FIELD off, so
 * the same binary can be A/B'd around it without a rebuild -- same idiom as
 * jaiListUnboxOn's JAITHON_LIST_UNBOX (object_collection.c) and
 * jitPicEnabled's JAITHON_JIT_PIC. Read once: OP_GET_FIELD is hot enough that
 * an uncached getenv on every static access would be its own cost. */
static bool jitStaticFieldEnabled(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("JAITHON_JIT_STATIC_FIELD");
        on = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return on != 0;
}

/* `klass->statics` is a JaiTable exactly like a module's globals table (same
 * struct, same keyVersion), so a static field is read the same way a module
 * global is read BY ADDRESS above: bake the JaiEntry*, not the value. A
 * static is reassignable at runtime -- jaiSetProperty's IS_CLASS arm (vm.c)
 * takes any value with no isLet check, and the checker's own
 * _check_field_assign skips its immutability error whenever `static_access`
 * is true, so `let` buys no promise here that the interpreter or the checker
 * actually keeps. Nothing below may bake the VALUE, only its address, behind
 * the same two guards a module global stands on. One table per body, same
 * plan as globalsTable -- see that field's comment on the struct. */
static JaiEntry *staticFieldSlot(Emit *e, ObjClass *klass, ObjString *name) {
    JaiTable *t = &klass->statics;
    JaiEntry *slot = jaiTableFindEntryInterned(t, name);
    if (slot == NULL) return NULL;
    if (e->staticsTable == NULL) {
        e->staticsTable = t;
        e->staticsKeyVersion = t->keyVersion;
    } else if (e->staticsTable != t) {
        return NULL;
    }
    return slot;
}

/* emitGlobalsGuard's counterpart for e->staticsTable: not hoisted, for the
 * same reason. */
static void emitStaticsGuard(Emit *e) {
    uint32_t at = e->staticsKeyVersion;
    emitConst64(e, JIT_SCRATCH_D,
                (int64_t)(uintptr_t)&e->staticsTable->keyVersion);
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
/* What an InlineCache::resultKind byte says, for JAI_JIT_WHY. A site the tier
 * refuses for want of a result kind is refused for one of three quite different
 * reasons, and the fix differs for each. */
static const char *jaiFeedbackName(uint8_t fb) {
    if (fb == JAI_FB_NONE) return "never observed";
    if (fb == JAI_FB_MIXED) return "mixed";
    if (fb >= JAI_FB_OBJ && fb < JAI_FB_OBJ + (unsigned)OBJ_TYPE_COUNT) {
        return jaiObjTypeName((ObjType)(fb - JAI_FB_OBJ));
    }
    switch ((ValueType)(fb - 1u)) {
    case VAL_NULL:  return "null";
    case VAL_BOOL:  return "bool";
    case VAL_INT:   return "int";
    case VAL_FLOAT: return "float";
    default: break;
    }
    return "something the tier does not name";
}

/* `objType` reports the ObjType the feedback named, as ObjType + 1, or 0 when
 * the result is not an object. See Emit::stackObjType for what it is for. */
static bool feedbackSlotKind(uint8_t fb, SlotKind *k, unsigned *tag,
                             uint8_t *objType) {
    *objType = 0;
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
    *objType = (uint8_t)(fb - JAI_FB_OBJ + 1u);
    return true;
}

/* The store half of `list.push` and of a comprehension's append: they differ
 * only in where the list sits on the stack and in what is left behind, so the
 * bounds check, the grow fixup and the two stores live here. Returns false
 * with `whyNot` set when the value's kind has no tag to store.
 *
 * Appending is a bounds check and two stores -- a descriptor+native round trip
 * costs far more than the work itself (list_ops spent all its time on the
 * call). A full list goes out to the `grow` stubs' realloc helper and comes
 * straight back; see there for why this used to be a deopt and what it cost. */
static bool emitListStore(Emit *e, SlotKind vk, unsigned rList, unsigned rVal,
                          int slot) {
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

    /* jitListGrow only reserves, and jaiListReserve is width-aware, so the
     * growth half of this needs nothing; it is the store below that has to
     * know how wide an element is. */
    /* jitListGrow only reserves, and jaiListReserve is width-aware, so the
     * growth half of this needs nothing; it is the store below that has to
     * know how wide an element is. */
    ListAccess pAcc = listAccessFor(e, rList, slot, vk, JIT_SCRATCH_A);
    if (!pAcc.dynamic && pAcc.stg != LIST_STORE_BOXED &&
        vk != listStgKind(pAcc.stg)) {
        return subWhy(e, "pushing kind %d onto storage %u", (int)vk, pAcc.stg);
    }

    emit(e, jaiA64LdrW(JIT_SCRATCH_A, rList,
                       (unsigned)offsetof(ObjList, count)));
    emit(e, jaiA64LdrW(JIT_SCRATCH_B, rList,
                       (unsigned)offsetof(ObjList, capacity)));
    emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
    if (e->growCount >= JIT_MAX_GROW) {
        e->whyNot = "more list pushes than the tier tracks";
        return false;
    }
    /* jitListGrow can raise, and the stub routes that to the exception exit
     * (see emitGrowStubs). */
    if (!raiseExitAllowed(e, "a list growth inside a try")) return false;

    noteScratchClobber(e);
    unsigned gi = e->growCount++;
    e->grow[gi].listReg  = rList;
    e->grow[gi].valReg   = rVal;
    e->grow[gi].tag      = vtag;
    e->grow[gi].countReg = JIT_SCRATCH_A;
    e->grow[gi].stub     = -1;
    if (e->fixupCount >= JIT_MAX_FIXUPS) { e->failed = true; return false; }
    e->fixups[e->fixupCount].instIndex    = (int)e->count;
    e->fixups[e->fixupCount].targetOffset = FIXUP_GROW - gi;
    e->fixups[e->fixupCount].conditional  = true;
    e->fixups[e->fixupCount].depth        = -1;
    e->fixupCount++;
    emit(e, jaiA64BCond(JAI_A64_GE, 0));
    e->grow[gi].returnTo = (int)e->count;

    emit(e, jaiA64LdrX(JIT_SCRATCH_C, rList,
                       (unsigned)offsetof(ObjList, items)));
    /* JIT_SCRATCH_C is the items pointer and JIT_SCRATCH_A the index; every
     * arm below starts from those two, so the test costs a load, a compare and
     * two branches and touches nothing else. */
    int pSkip = listDispatchBegin(e, &pAcc, rList, JIT_SCRATCH_D);
    emitListElemStore(e, pAcc.stg, vtag, rVal);
    if (pSkip >= 0) {
        int pJoin = listDispatchElse(e, pSkip);
        emitListElemStore(e, pAcc.alt, vtag, rVal);
        listDispatchEnd(e, pJoin);
    }
    emit(e, jaiA64AddXImm(JIT_SCRATCH_A, JIT_SCRATCH_A, 1));
    emit(e, jaiA64StrW(JIT_SCRATCH_A, rList,
                       (unsigned)offsetof(ObjList, count)));
    emit(e, jaiA64LdrW(JIT_SCRATCH_A, rList,
                       (unsigned)offsetof(ObjList, version)));
    emit(e, jaiA64AddXImm(JIT_SCRATCH_A, JIT_SCRATCH_A, 1));
    emit(e, jaiA64StrW(JIT_SCRATCH_A, rList,
                       (unsigned)offsetof(ObjList, version)));
    e->wroteHeap = true;
    return true;
}

/* What an OP_INVOKE site has been observed to return, merged over every way its
 * inline cache holds.
 *
 * For a site whose receiver class the model cannot pin there is no callee to
 * ask -- but the interpreter watched the same site run, and eight
 * implementations of one trait method that all return an int agree on that
 * much. Ways that recorded nothing are skipped rather than merged: NONE
 * against a real kind is MIXED, which would throw away the evidence the other
 * ways did gather.
 *
 * A PREDICTION. The tag guard the caller emits after the call is the whole of
 * what makes it sound; a way that later returns something else deoptimises. */
static bool siteInvokeResultKind(const Chunk *chunk, uint16_t cacheIdx,
                                 SlotKind *k, unsigned *tag) {
    if (chunk->caches == NULL || (int)cacheIdx >= chunk->cacheCount) {
        return false;
    }
    const InlineCache *ic = &chunk->caches[cacheIdx];
    uint8_t merged = JAI_FB_NONE;
    for (int w = 0; w < ic->count && w < JAI_IC_WAYS; w++) {
        if (ic->resultKind[w] == JAI_FB_NONE) continue;
        merged = jaiFeedbackMerge(merged, ic->resultKind[w]);
    }
    if (merged == JAI_FB_NONE || merged == JAI_FB_MIXED) return false;
    uint8_t objType;
    return feedbackSlotKind(merged, k, tag, &objType);
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
    uint8_t objType;
    return feedbackSlotKind(fb, k, &tag, &objType);
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

/* A scalar field's DECLARED kind (FieldInfo::typeId, OP_FIELD_DEF's bits 4-7,
 * spec Sec3.7), for a receiver with a pinned class but no sample Value to read
 * `inst->fields[slot]` off -- OP_GET_FIELD's twin of OP_ELEM_KIND's own use of
 * the same bits (see that case) rather than a sampled container.
 *
 * Not a new promise: OP_SET_FIELD's guard (jaiKindAccepts, vm.c) already
 * refuses any store that disagrees with this field's declared kind, so a
 * caller here is only reading a fact the runtime enforces on every write, and
 * the tag is checked again at the load below regardless -- a stale or wrong
 * record still deopts rather than answers.
 *
 * INT/FLOAT/BOOL only. FIELD_KIND_LIST names the box but not the element, so
 * admitting it here would still leave a consumer that iterates the field with
 * nothing to look at (the "iterating a list with nothing to look at" refusal,
 * unresolved either way); FIELD_KIND_INSTANCE names no specific class to guard
 * against; FIELD_KIND_ANY and FIELD_KIND_STR/DICT promise nothing scalar. All
 * four are left to the sampled path, unchanged. */
static bool declaredScalarFieldKind(uint32_t typeId, SlotKind *k, unsigned *tag) {
    switch (typeId) {
    case FIELD_KIND_INT:   *k = SLOT_INT;   *tag = VAL_INT;   return true;
    case FIELD_KIND_FLOAT: *k = SLOT_FLOAT; *tag = VAL_FLOAT; return true;
    case FIELD_KIND_BOOL:  *k = SLOT_BOOL;  *tag = VAL_BOOL;  return true;
    /* A declared `list[T]` field. VAL_OBJ is every heap object, so the caller
     * must ALSO prove OBJ_LIST before anything reads ObjList's header off it --
     * the same hazard that segfaulted the VM through the dict-index arm. It is
     * separated from the three scalars above because it is the only kind here
     * that needs a second guard.
     *
     * Worth predicting because a list field is where a chain bottoms out:
     * `code.data[i]` is a field read the tier could not classify, and behind
     * that one refusal sat eleven functions and 14.27% of one file's
     * interpreted work. */
    case FIELD_KIND_LIST:  *k = SLOT_LIST;  *tag = VAL_OBJ;   return true;
    default: return false;
    }
}

/* JAITHON_JIT_FIELD_DECL_KIND=0 turns declaredScalarFieldKind's OP_GET_FIELD
 * arm off, reproducing the pre-fix decline for an A/B inside one binary --
 * same cached-getenv idiom as jitDeoptStress above, but default ON since this
 * is a fix, not a stress knob. */
static bool compileBody(Emit *e, ObjClosure *closure);
static const char *declineReason(Emit *e);

/* JAI_JIT_CHAIN=1: print the whole chain of refusals a body would hit, not just
 * the first one.
 *
 * "What would this body stop at NEXT?" is the question that decides whether an
 * arm is worth building, and until now it was answered by BUILDING the arm and
 * re-running -- a day per link, and how three separate changes came to measure
 * exactly zero after clearing one link of a longer chain.
 *
 * The mechanism is deliberately dumb: recompile the body with the offending
 * offset forced onto the unarmed path, and see what it says next. That reuses a
 * path the tier already exercises constantly, rather than continuing a walk
 * whose model has gone inconsistent -- which was tried, and segfaults.
 *
 * Diagnostic only. Each link costs one extra compile of one body, and nothing
 * here runs unless the env var is set. */
static bool jitChainOn(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAI_JIT_CHAIN");
        cached = (v != NULL && v[0] != '\0' && v[0] != '0') ? 1 : 0;
    }
    return cached != 0;
}

static void reportChain(const Emit *proto, Emit *first, ObjClosure *closure,
                        ObjFunction *fn) {
    static Emit probe;
    uint32_t skips[JIT_MAX_CHAIN];
    unsigned n = 0;
    const char *name = fn->name != NULL ? fn->name->chars : "<anon>";

    fprintf(stderr, "[jit] chain %s:\n", name);
    fprintf(stderr, "[jit]   1. %s  (at %u)\n", declineReason(first),
            first->curOffset);
    skips[n++] = first->curOffset;

    for (unsigned link = 2; link <= JIT_MAX_CHAIN; link++) {
        memcpy(&probe, proto, sizeof probe);
        memcpy(probe.chainSkip, skips, n * sizeof skips[0]);
        probe.chainSkipCount = n;
        if (compileBody(&probe, closure)) {
            fprintf(stderr, "[jit]   %u. compiles, once the %u above %s "
                            "cleared\n", link, n, n == 1 ? "is" : "are");
            return;
        }
        /* Refusing again at the SAME offset means the unarmed path cannot step
         * over that instruction: deoptSite has nowhere to resume, which is a
         * real property of the instruction and not an artefact of this probe.
         * Say so rather than numbering it as the next link, because it is not
         * one -- it is where the walk stops being able to look. */
        if (probe.curOffset == skips[n - 1]) {
            fprintf(stderr,
                    "[jit]   ... cannot look past link %u: stepping over it "
                    "gives \"%s\"\n", n, declineReason(&probe));
            return;
        }
        fprintf(stderr, "[jit]   %u. %s  (at %u)\n", link,
                declineReason(&probe), probe.curOffset);
        if (n >= JIT_MAX_CHAIN) {
            fprintf(stderr, "[jit]   ... and the chain runs longer than %u\n",
                    JIT_MAX_CHAIN);
            return;
        }
        skips[n++] = probe.curOffset;
    }
}

static bool jitDeclaredFieldKindEnabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_FIELD_DECL_KIND");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
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

/* Which local the tier could not settle on one kind for.
 *
 * The reason on its own says a body has such a local but not which, and a
 * body with twenty of them then has to be read line by line to find it. The
 * slot number is what the disassembly labels its locals with, so the two can
 * be put side by side.
 */
/* Record which unnamed refusal an arm took, and return false so the call sites
 * read as `return subWhy(e, "...")`. See Emit::whySub. */
static bool subWhy(Emit *e, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(e->whySub, sizeof e->whySub, fmt, ap);
    va_end(ap);
    return false;
}

/* What `JAI_JIT_WHY` prints: the named reason when there is one, otherwise the
 * opcode with whatever the arm noted about it. */
static const char *declineReason(Emit *e) {
    if (e->whyNot != NULL) return e->whyNot;
    const char *name = jaiOpName((OpCode)e->lastOp);
    if (e->whySub[0] == '\0') return name;
    snprintf(e->whyBuf, sizeof e->whyBuf, "%s: %s", name, e->whySub);
    return e->whyBuf;
}

static const char *kindClash(Emit *e, unsigned slot) {
    /* WHICH two kinds, not just which slot. The slot number says where to look
     * and the pair says what to do about it: an int meeting a float is a
     * widening the tier could learn, an instance meeting a list is a genuinely
     * polymorphic local and nothing will help it. Without the pair, ~290 of
     * these across four compiler files were one undifferentiated heap. */
    snprintf(e->whyBuf, sizeof e->whyBuf,
             "local %u was given two kinds, %s and %s", slot,
             slotKindName(e->localKind[slot]), slotKindName(e->clashKind));
    return e->whyBuf;
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
/* JAITHON_JIT_SHAPE_LIMIT=8 puts the OSR instance-shape cap back where it was,
 * for a one-binary A/B. The array in JaiOsrForm is always the wider one, so
 * only the refusal moves. */
static unsigned jitShapeLimit(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_SHAPE_LIMIT");
        cached = (v != NULL) ? atoi(v) : (int)JAI_OSR_SHAPES;
        if (cached < 1 || cached > (int)JAI_OSR_SHAPES) cached = (int)JAI_OSR_SHAPES;
    }
    return (unsigned)cached;
}

/* JAITHON_JIT_ROOT_LIMIT=10 puts the root cap back where it was when it shared
 * the register budget, for a one-binary A/B. The ARRAY is always the wider one,
 * so only the refusal moves. */
static unsigned jitRootLimit(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_ROOT_LIMIT");
        cached = (v != NULL) ? atoi(v) : (int)JIT_MAX_ROOTS;
        if (cached < 1 || cached > (int)JIT_MAX_ROOTS) cached = (int)JIT_MAX_ROOTS;
    }
    return (unsigned)cached;
}

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
        if (nroots >= jitRootLimit()) {
            e->whyNot = "too many roots"; return false;
        }
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
        unsigned reg = valueBankReg(e, seen);
        seen++;
        if (k != SLOT_INST && k != SLOT_LIST && k != SLOT_OBJ &&
            k != SLOT_ITER && k != SLOT_MAYBE_INST) {
            continue;
        }
        if (nroots >= jitRootLimit()) {
            e->whyNot = "too many roots"; return false;
        }
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

    /* The value index of the first argument. Counted rather than derived from
     * `depth - valueDepth`, which is the number of register-free entries below
     * `depth` ANYWHERE: a class argument is one of those, and every argument
     * above it would then be read out of the wrong register. */
    unsigned vidx = e->valueDepth;
    for (unsigned idx = first; idx < e->depth; idx++) {
        if (holdsRegister(e->stack[idx])) vidx--;
    }

    /* The arguments, which for an invoke begin with the receiver. */
    for (unsigned i = 0; i < nargs; i++) {
        unsigned idx = first + i;
        SlotKind k = e->stack[idx];
        unsigned at = d + (unsigned)offsetof(JitCallDesc, args) +
                      i * (unsigned)sizeof(Value);
        /* A class occupies no register -- it is a constant of the module, and
         * the class the interpreter would have found is the one the model
         * recorded. Baking it is the same trust the callee slot above already
         * takes, and it is what lets `isinstance(x, T)` be called at all. */
        if (k == SLOT_CLASS) {
            ObjClass *argCls = e->stackClass[idx];
            if (argCls == NULL) {
                e->whyNot = "a class argument the model did not pin";
                return false;
            }
            emit(e, jaiA64MovzX(JIT_SCRATCH_A, VAL_OBJ, 0));
            emit(e, jaiA64StrW(JIT_SCRATCH_A, 31, at));
            emitConst64(e, JIT_SCRATCH_A, (int64_t)(uintptr_t)argCls);
            emit(e, jaiA64StrX(JIT_SCRATCH_A, 31, at + 8));
            continue;
        }
        if (k != SLOT_INT && k != SLOT_FLOAT && k != SLOT_BOOL &&
            k != SLOT_INST && k != SLOT_LIST && k != SLOT_OBJ &&
            k != SLOT_ITER && k != SLOT_MAYBE_INST) {
            e->whyNot = "an argument kind this call cannot pass";
            return false;
        }
        unsigned reg = valueBankReg(e, vidx);
        vidx++;
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
    if (!raiseExitAllowed(e, "a call that can raise inside a try")) return false;
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

/* Is the pair on top of the stack `str + str`?
 *
 * A string is SLOT_OBJ here, which pins nothing, so the sample is what makes
 * this worth emitting and the guards below are what make it sound. Only ONE
 * side needs a string sample: OP_FORMAT pushes its result without one (there
 * is no Value to carry at compile time), and `text = text + f"..."` -- the
 * shape word_freq's whole hot loop is -- has exactly that on the right. */
static bool concatOperands(const Emit *e, Value *sample) {
    /* Not inside an inlined body. OP_ADD is on inlinableBody's whitelist
     * because every arm it had emitted straight-line code; this one calls, and
     * an inlined body that calls breaks two things at once. Its entries live in
     * x0..x8 when the caller's bank is callee-saved (inlineOwnBank), which is
     * the register file the call destroys, and emitDescriptor reads its
     * arguments through valueBankReg -- which does not know about that bank, so
     * the descriptor is filled from callee-saved registers holding something
     * else entirely. `fn cat(a: any) -> any { return a + "x" }` in a loop
     * segfaults without this line. Declining costs nothing: the inline fails,
     * and both tiers retry the whole body with inlining off, where this arm
     * fires normally. */
    if (e->inlining) return false;
    if (e->depth < 2) return false;
    if (e->stack[e->depth - 1] != SLOT_OBJ) return false;
    if (e->stack[e->depth - 2] != SLOT_OBJ) return false;
    Value sa = e->stackSeen[e->depth - 2], sb = e->stackSeen[e->depth - 1];
    if (!IS_STRING(sa) && !IS_STRING(sb)) return false;
    *sample = IS_STRING(sa) ? sa : sb;
    return true;
}

/* `a + b` out to jaiStringConcat. Concatenation allocates, so there is nothing
 * to inline; the point is that the rest of the loop body stops being given up.
 * word_freq declined its whole `main` loop on the ADD_BIND this replaces -- one
 * refusal costing an LCG step, an f-string and the append around it.
 *
 * Measured (best of five, alternating builds, under the GPU lock): a 3M-step
 * `text = text + f"w{n} "` loop 204.4ms -> 128.4ms, 1.59x; word_freq at 3M
 * words 392.8ms -> 313.9ms end to end, 1.25x; word_freq as shipped 35.2ms ->
 * 28.2ms, 1.85x once the ~20ms startup floor is taken off. Ruled out: a
 * self-hosted `check` of compile/parser.jai does NOT move (775ms -> 767ms,
 * noise) even though OP_ADD is a few percent of its interpreted work -- the
 * bodies holding it decline for other reasons anyway, so freeing this one
 * changes nothing there.
 *
 * Both operands are guarded, not just the sample-less one: SLOT_OBJ says
 * "heap object" and no more, and a guard here resumes at this instruction with
 * both operands still on the interpreter's stack, so a miss costs nothing --
 * a loop alternating `str + str` with `list + list` agrees with the
 * interpreter under both --gc-stress and JAITHON_JIT_DEOPT_STRESS.
 *
 * Deliberately NOT the general `arithmetic()` fallback: that answers int, float
 * and string alike, so the result would need a tag test after the call to know
 * whether the register holds a pointer. Guarding the two operands instead
 * settles the result kind before the call is made, which is the same argument
 * OP_GET_SLICE makes for guarding its container. */
static bool emitStringConcat(Emit *e, Value sample) {
    settleAll(e);                    /* this path guards */
    unsigned ra = valueXReg(e, e->valueDepth - 2);
    unsigned rb = valueXReg(e, e->valueDepth - 1);

    emit(e, jaiA64LdrW(JIT_SCRATCH_A, ra, (unsigned)offsetof(Obj, type)));
    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_STRING));
    branchOnDeopt(e, JAI_A64_NE);
    emit(e, jaiA64LdrW(JIT_SCRATCH_A, rb, (unsigned)offsetof(Obj, type)));
    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_STRING));
    branchOnDeopt(e, JAI_A64_NE);

    if (!emitDescriptor(e, NULL_VAL, e->depth - 2, 2,
                        (void *)&jitStringConcat)) {
        return false;
    }
    unsigned drop;
    if (!popValue(e, &drop, NULL)) return false;
    if (!popValue(e, &drop, NULL)) return false;
    /* Carry a sample so the next instruction still knows this is a string --
     * the same reason the string-index arm carries its receiver's. */
    if (!pushValue3(e, SLOT_OBJ, 0, NULL, sample, -1)) return false;
    emit(e, jaiA64LdrX(pushReg(e) - 1, 31,
                       e->descOffset +
                           (unsigned)offsetof(JitCallDesc, result) + 8));
    e->wroteHeap = true;
    return true;
}

/* Put a local on the model stack, the way OP_GET_LOCAL does, for an arm that
 * needs stack operands but was handed slot numbers. Only the general X path --
 * a float local wants OP_GET_LOCAL's FP handling and no caller here has one. */
/* `x == null` where the model calls x an object -- a string, a list, a dict.
 *
 * The arm already mixes SLOT_INST with SLOT_MAYBE_INST because both are a
 * pointer or a zero in a register. A SLOT_OBJ is the same shape, and stronger:
 * emitTagFor gives it VAL_OBJ unconditionally, so its register never holds a
 * zero and the answer is always "not null" -- which is exactly what comparing
 * it against the null literal's zero register produces.
 *
 * EQUALITY ONLY. `x < null` is a TypeError in the interpreter, and compiling it
 * as a register compare would answer where it should raise.
 *
 * SLOT_OBJ ONLY, and that is the load-bearing half. A SLOT_INT is also "never
 * null", but its register holds a NUMBER -- and zero is a perfectly good int,
 * so `0 == null` would compare equal and answer true. Only a kind whose
 * register holds a pointer may be compared against the null literal's zero.
 *
 * 108 declines across four compiler files, in `_parse_postfix`,
 * `_parse_decorators`, `lookup_type_name` and their kin -- the shape is
 * `if tok == null` on something the model did not name more precisely. */
static bool jitNullPair(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_NULL_PAIR");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

static bool nullLiteralPair(const Emit *e, uint8_t op, SlotKind ka, SlotKind kb) {
    if (!jitNullPair()) return false;
    if (op != OP_EQ && op != OP_NE) return false;
    if (e->depth < 2) return false;
    if (ka == SLOT_OBJ && kb == SLOT_MAYBE_INST &&
        e->stackNullLit[e->depth - 1]) {
        return true;
    }
    return kb == SLOT_OBJ && ka == SLOT_MAYBE_INST &&
           e->stackNullLit[e->depth - 2];
}

static bool pushLocalAsValue(Emit *e, unsigned slot) {
    if (!pushValue3(e, e->localKind[slot], e->localShape[slot],
                    e->localClass[slot],
                    localObserved(e, slot) ? e->observed[slot]
                                           : e->localSeen[slot],
                    (int)slot)) {
        return false;
    }
    unsigned home = localHomeX(e, slot);
    if (home != 0) {
        xBorrowLocal(e, e->valueDepth - 1, home);
    } else {
        unsigned dst = pushReg(e) - 1;
        unsigned src = localIn(e, slot, dst);
        if (src != dst) emit(e, jaiA64MovX(dst, src));
    }
    return true;
}

static bool jitConcatLocals(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_CONCAT_LOCALS");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

/* A builtin whose whole body is a load from its receiver (jit_field_read.h).
 * The caller has already guarded that the receiver is `fr->type`, so the load
 * is the entire call: no callee Value, no argument Value, no root fill, no
 * status test, no result tag test. `k.len()` was 39 instructions and a `blr`
 * into jitInvokeNative for a 32-bit field.
 *
 * A lazily-computed field (ObjString::scalars) keeps the descriptor call as a
 * slow path *inline*, reached only when the memo is empty, so the native fills
 * it and every later call takes the load. Deopting there instead would be
 * wrong: a loop over freshly built strings would leave the compiled body on
 * every iteration. Both paths land on the same register, and the span over the
 * slow path is measured rather than counted -- emitDescriptor's length moves
 * with the number of roots the body holds. */
static bool emitFieldRead(Emit *e, const JaiJitFieldRead *fr, Value nativeVal,
                          unsigned ridx, unsigned argc, uint32_t afterIp) {
    if (fr->tag != VAL_INT) {
        e->whyNot = "a field-reading builtin whose result is not an int";
        return false;
    }
    unsigned rRecv   = pushReg(e) - argc - 1;
    unsigned skipSlow = 0;

    if (fr->width == 4) {
        emit(e, jaiA64LdrW(JIT_SCRATCH_A, rRecv, fr->offset));
    } else {
        emit(e, jaiA64LdrX(JIT_SCRATCH_A, rRecv, fr->offset));
    }

    if (fr->lazy) {
        emitConst64(e, JIT_SCRATCH_B, (int64_t)(uint64_t)fr->sentinel);
        emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
        skipSlow = e->count;
        emit(e, jaiA64BCond(JAI_A64_NE, 0));      /* patched below */
        if (!emitDescriptor(e, nativeVal, ridx, argc + 1,
                            (void *)&jitInvokeNative)) {
            return false;
        }
    }

    for (unsigned i = 0; i <= argc; i++) {
        unsigned r;
        if (!popValue(e, &r, NULL)) return false;
    }
    if (!pushValue(e, SLOT_INT, 0, NULL)) return false;

    if (fr->lazy) {
        unsigned rat = e->descOffset +
                       (unsigned)offsetof(JitCallDesc, result);
        emit(e, jaiA64LdrW(JIT_SCRATCH_B, 31, rat));
        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_B, VAL_INT));
        branchOnDeoptAt(e, JAI_A64_NE, afterIp, true);
        emit(e, jaiA64LdrX(JIT_SCRATCH_A, 31, rat + 8));
        if (skipSlow < e->count && e->count <= JIT_MAX_INSTS) {
            e->code[skipSlow] =
                jaiA64BCond(JAI_A64_NE, (int32_t)(e->count - skipSlow));
        }
        /* The slow path calls out, and a native may write. */
        e->wroteHeap = true;
    }
    emit(e, jaiA64MovX(pushReg(e) - 1, JIT_SCRATCH_A));
    return true;
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
        /* One byte for a bool, as every other descriptor-result site loads
         * one. BOOL_VAL writes only the union's one-byte member, so a slot the
         * interpreter last used for a large int or a pointer keeps its upper
         * seven bytes -- verified on this toolchain at -O2 -flto: a slot
         * holding INT_VAL(0x1122334455667788) then assigned BOOL_VAL(false)
         * reads back 0x1122334455667700. Every SLOT_BOOL consumer branches on
         * the whole word, so that `false` reads as `true`.
         *
         * The pinned arm already reached this stub, so the widening predates
         * the polymorphic one -- but the PIC routes a whole new class of site
         * through here, and those sites previously always took the one-byte
         * load. */
        if (e->selfSlow[si].tag == VAL_BOOL) {
            emit(e, jaiA64LdrByte(e->selfSlow[si].resultReg, 31, resultAt + 8));
        } else {
            emit(e, jaiA64LdrX(e->selfSlow[si].resultReg, 31, resultAt + 8));
        }

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
/* A `-> T?` result read back out of a call descriptor.
 *
 * Two tags are acceptable where every other kind has exactly one, so the single
 * compare the other arms use cannot serve. VAL_NULL is zero, which is what lets
 * "object or null" be two csels and a compare rather than a branch: afterwards
 * D holds the payload or a defined zero, and B is zero exactly when the tag was
 * one of the two.
 *
 * The class checks apply only to a non-null. The field arm at
 * OP_GET_FIELD_LOCAL keeps its equivalents branch-free by redirecting the loads
 * at its live receiver; a call result has no such pointer to borrow, so they
 * sit behind a forward branch instead. A deopt inside that span is sound for
 * the reason emitBoundsNormalise's is: the record is written from the
 * descriptor, which holds the true Value whichever way the branch went.
 *
 * Worth having because after this the refusal it removes was the tier's single
 * largest on the self-hosted compiler: one `check --no-cache` of
 * compile/parser.jai stopped 64 bodies at "callee's return kind not usable"
 * with the kind being exactly SLOT_MAYBE_INST, and every one of them had a
 * resolvable class. `-> Node?` is what a parser's methods return. */
static void emitMaybeInstResult(Emit *e, unsigned dst, unsigned rat,
                                uint32_t rshape, uint32_t deoptIp) {
    emit(e, jaiA64LdrW(JIT_SCRATCH_A, 31, rat));
    emit(e, jaiA64LdrX(JIT_SCRATCH_D, 31, rat + 8));
    emit(e, jaiA64MovzX(JIT_SCRATCH_B, 0, 0));
    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, VAL_OBJ));
    emit(e, jaiA64CselX(JIT_SCRATCH_D, JIT_SCRATCH_D, JIT_SCRATCH_B,
                        JAI_A64_EQ));
    emit(e, jaiA64CselX(JIT_SCRATCH_B, JIT_SCRATCH_B, JIT_SCRATCH_A,
                        JAI_A64_EQ));
    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_B, 0));
    branchOnDeoptAt(e, JAI_A64_NE, deoptIp, true);

    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_D, 0));
    unsigned skip = e->count;
    emit(e, jaiA64BCond(JAI_A64_EQ, 0));
    emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_D,
                       (unsigned)offsetof(Obj, type)));
    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_INSTANCE));
    branchOnDeoptAt(e, JAI_A64_NE, deoptIp, true);
    emit(e, jaiA64LdrX(JIT_SCRATCH_A, JIT_SCRATCH_D,
                       (unsigned)offsetof(ObjInstance, klass)));
    emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_A,
                       (unsigned)offsetof(ObjClass, shapeId)));
    emitConst64(e, JIT_SCRATCH_B, (int64_t)rshape);
    emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
    branchOnDeoptAt(e, JAI_A64_NE, deoptIp, true);
    /* Nothing to jump over means the arena filled mid-sequence; leaving the
     * branch unpatched would run the class loads on a null pointer. */
    if (skip < e->count && e->count <= JIT_MAX_INSTS) {
        e->code[skip] = jaiA64BCond(JAI_A64_EQ, (int32_t)(e->count - skip));
    } else {
        e->failed = true;
    }
    emit(e, jaiA64MovX(dst, JIT_SCRATCH_D));
}

static bool jitAnyGuard(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_ANY_GUARD");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

static bool jitReturnKnownOn(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_RETURN_KNOWN");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

static bool emitDirectCall(Emit *e, ObjFunction *caller, ObjFunction *cfn,
                           Value calleeVal, int calleeReg, unsigned cidx,
                           unsigned argc, uint32_t callOff, uint32_t after,
                           bool method) {
    /* Either verdict path below can come back with an exception pending. */
    if (!raiseExitAllowed(e, "a call that can raise inside a try")) return false;
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
    /* A callee whose walk never reached an OP_RETURN has no return kind to be
     * the contract, only the SLOT_INT of a zeroed Emit. `_is_ident_start` in
     * the lexer walks only to OP_GET_GLOBAL at offset 0 -- a cold `throw` on
     * its first instruction -- and still claimed to return an int, which made
     * `_is_ident_start(c) or _is_digit(c)` decline on "a branch on a int, not
     * a bool" and left `_ident_run_end`'s OSR loop retrying it eighty times.
     *
     * Declining here is not a coverage loss: the caller falls back to the
     * guarded emitGlobalCall path, which asks the interpreter's own
     * observation first and gets the right answer. */
    if (!cfn->jitReturnKnown && jitReturnKnownOn()) {
        e->whyNot = "a direct callee whose walk never reached a return";
        return false;
    }
    SlotKind rk = (SlotKind)cfn->jitReturnKind;
    ObjClass *rcls = NULL;
    /* SLOT_MAYBE_INST rides with SLOT_INST here and needs no guard of its own:
     * a direct branch takes the callee's raw payload, and for a nullable
     * instance that payload IS the pointer or a zero -- the same
     * representation this caller will hold. The callee's declared return kind
     * is the contract, exactly as it is for every other kind at this arm. */
    if (rk != SLOT_INT && rk != SLOT_FLOAT && rk != SLOT_BOOL &&
        rk != SLOT_INST && rk != SLOT_MAYBE_INST && rk != SLOT_LIST &&
        rk != SLOT_OBJ && rk != SLOT_NULL) {
        e->whyNot = "callee's return kind not usable";
        return false;
    }
    if ((rk == SLOT_INST || rk == SLOT_MAYBE_INST) &&
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

    unsigned firstArg = firstIdx - (e->depth - e->valueDepth);
    for (unsigned i = 0; i < nargs; i++) {
        emit(e, jaiA64MovX(i, valueXReg(e, firstArg + i)));
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

/* JAITHON_JIT_PIC=0 turns the one-way inline cache below off, so the same
 * binary can be A/B'd around it without a rebuild -- same idiom as
 * jaiListUnboxOn's JAITHON_LIST_UNBOX (object_collection.c) and
 * jaiJitEnabled's JAITHON_NO_JIT. Read once: this sits on every unpinned
 * OP_INVOKE, and an uncached getenv there is its own cost (see
 * jitReconTrace above). */
static bool jitPicEnabled(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("JAITHON_JIT_PIC");
        on = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return on != 0;
}

/* Mirrors every decision emitDirectCall makes before it commits to emitting,
 * for the one way this cache is about to speculate on. By the time the shape
 * compare below is in the instruction stream a decline has nowhere to fall
 * back to but e->failed, so this runs FIRST and answers instead of finding
 * out the hard way. */
static bool jitPic1Admissible(Emit *e, ObjFunction *caller, ObjFunction *cfn,
                              unsigned ridx, unsigned argc, uint32_t shape) {
    if (cfn->module != caller->module) return false;
    if (caller->module == NULL ||
        cfn->jitFuncModuleVersion != caller->module->version) {
        return false;
    }
    if (cfn->jitArgBase != 0u) return false;
    unsigned nargs = argc + 1u;
    unsigned calleeArgs = (unsigned)cfn->jitArgCount;
    bool wantsClosure = calleeArgs == nargs + 1u;
    if (!wantsClosure && calleeArgs != nargs) return false;
    if (calleeArgs > JIT_MAX_ARITY) return false;
    if (wantsClosure &&
        (SlotKind)cfn->jitParamKind[nargs] != SLOT_CLOSURE) {
        return false;
    }
    /* The receiver is the callee's slot 0, and the branch about to be
     * emitted is the proof of its class -- so the parameter the callee was
     * specialised for has to be that same class, not merely some instance. */
    if ((SlotKind)cfn->jitParamKind[0] != SLOT_INST) return false;
    if (cfn->jitParamShape[0] != shape) return false;
    for (unsigned i = 1; i < nargs; i++) {
        unsigned idx = ridx + i;
        SlotKind have = e->stack[idx];
        SlotKind want = (SlotKind)cfn->jitParamKind[i];
        if (!holdsRegister(have)) return false;
        if (want == SLOT_OPAQUE) continue;   /* never read; see seedLocals */
        if (want == SLOT_MAYBE_INST) {
            if (have != SLOT_INST && have != SLOT_MAYBE_INST) return false;
        } else if (have != want) {
            return false;
        }
        if ((want == SLOT_INST || want == SLOT_MAYBE_INST) &&
            e->stackShape[idx] != cfn->jitParamShape[i]) {
            return false;
        }
    }
    /* A callee that writes is finished in the interpreter from a selfSlow
     * record, and this arm takes one of its own -- the same budget a pinned
     * direct call draws from. */
    if (!cfn->jitFuncNoWrite && e->selfSlowCount >= JIT_MAX_SELF_SLOW) {
        return false;
    }
    return true;
}

/* The compile-time half of a ONE-WAY inline cache: what the model could not
 * pin about a receiver, read off the site's own InlineCache instead.
 *
 * One way only, and only the FIRST shape this site ever recorded --
 * ic->cached[0] / ic->shapeId[0] -- not the whole walk over every way a
 * polymorphic cache holds. The receiver's class is loaded once and compared
 * against that one shape; a hit branches straight into the recorded callee's
 * compiled entry exactly as a pinned receiver already does (emitDirectCall).
 * A miss -- every OTHER class at a polymorphic site, and the common case at a
 * megamorphic one -- falls through to jitInvokeByName, which is exactly what
 * the site emits today and is unchanged by any of this.
 *
 * The compare is not a guard and a miss is not a deopt: both sides of it run
 * BEFORE the call, so nothing has happened yet that must not happen twice.
 * That is the whole reason this needs no deopt record of its own, unlike the
 * call inside it (emitDirectCall still takes one, for what the CALL can do
 * after it starts).
 *
 * `havePrediction`/`rkind` is the fall-through's own prediction
 * (siteInvokeResultKind), which the one way must also return: only a
 * register-only kind (SLOT_INT/FLOAT/BOOL) is admitted, because those are the
 * only ones that carry no class shape for the two paths to disagree about --
 * see feedbackSlotKind's exclusion of SLOT_INST. `*toEnd` comes back holding
 * the arm's own branch to the merge point, for the caller to patch once it
 * has emitted the fall-through after it.
 *
 * Returns false having emitted nothing whenever there was something to
 * cleanly decline, so the caller can still take the descriptor path alone --
 * UNLESS e->failed, which means the shape compare is already in the stream
 * and there is nowhere left to fall back to (mirrors emitDirectCall's own
 * rule, since this arm ends by calling into it). */
static bool emitInvokePic1(Emit *e, ObjFunction *fn, unsigned ridx,
                           unsigned argc, uint32_t callOff, uint32_t after,
                           int siteCache, bool havePrediction, SlotKind rkind,
                           int *toEnd) {
    if (!havePrediction) {
        return subWhy(e, "no predicted result kind to join the arm on");
    }
    if (rkind != SLOT_INT && rkind != SLOT_FLOAT && rkind != SLOT_BOOL) {
        return subWhy(e, "an unpinned receiver returning something with a shape");
    }
    if (siteCache < 0 || fn->chunk.caches == NULL ||
        siteCache >= fn->chunk.cacheCount) {
        return subWhy(e, "no inline cache recorded for this site");
    }
    const InlineCache *ic = &fn->chunk.caches[siteCache];
    /* IC_MEGA is admitted, and it is the case that matters. A site that runs
     * out of ways stops caching ALTOGETHER and re-resolves every call through
     * sMegaCache -- see the comment on that table -- which is exactly the
     * shape a trait with eight implementations makes, and exactly the shape
     * this arm exists for. The ways it recorded before it gave up are still
     * there and still true: a way is (shapeId, method), shapeIds come from a
     * monotonic counter that would need four billion classes to repeat, and
     * every way is re-checked below for a live class and a compiled callee.
     * Four ways against eight classes is a partial cache, not a complete one,
     * and a partial cache is the whole point -- the misses cost one compare
     * each and then do exactly what the site does today. */
    if (ic->state != IC_MONO && ic->state != IC_POLY &&
        ic->state != IC_MEGA) {
        return subWhy(e, "the site's cache is empty");
    }
    if (ic->count == 0) return subWhy(e, "the site's cache has no way filled");
    if (ridx + argc + 1u > JIT_MAX_STACK) {
        return subWhy(e, "past the stack depth the model can describe");
    }
    /* Every way the cache holds is tried, in the order it recorded them, so
     * a site that warmed up on its second-most-common class no longer
     * speculates on the wrong one. The compares chain: way w's compare falls
     * through to way w+1's, and the last falls through to the descriptor the
     * site emits today. A miss therefore costs one compare per way and then
     * does exactly what it did before. */
    /* Collect the ways this compile can actually take. A way is dropped, not
     * fatal: the site keeps its remaining arms and the dropped class simply
     * goes round the descriptor as it does today. */
    unsigned    wayShape[JAI_IC_WAYS];
    Value       wayVal  [JAI_IC_WAYS];
    ObjFunction *wayFn  [JAI_IC_WAYS];
    ObjClass    *wayCls [JAI_IC_WAYS];
    unsigned    ways = 0;

    for (int w = 0; w < ic->count && w < JAI_IC_WAYS; w++) {
        /* What the cache settles is which method a shape resolves to, not
         * whether THIS caller may call it -- one site can present as two
         * classes at different visibilities, so the interpreter re-decides
         * that on every hit and nothing emitted here can. */
        if (ic->payload[w] != 0) continue;
        Value cv = ic->cached[w];
        if (!IS_CLOSURE(cv)) continue;
        ObjFunction *cf = AS_CLOSURE(cv)->fn;
        if (cf->jitFunc == NULL) continue;
        if ((SlotKind)cf->jitReturnKind != rkind || cf->jitReturnShape != 0) {
            continue;
        }
        ObjClass *cc = NULL;
        if (!jaiClassForShape(ic->shapeId[w], &cc) || cc == NULL) continue;
        if (!jitPic1Admissible(e, fn, cf, ridx, argc, ic->shapeId[w])) continue;
        wayShape[ways] = ic->shapeId[w];
        wayVal[ways]   = cv;
        wayFn[ways]    = cf;
        wayCls[ways]   = cc;
        ways++;
    }
    if (ways == 0) return subWhy(e, "no way of this site's cache is usable");

    settleAll(e);
    fpReleaseAll(e);
    if (e->fpLive != 0) {
        return subWhy(e, "an unpinned receiver with a value in the float bank");
    }

    /* Past here the shape compare is in the stream and the site is
     * committed: a failure below stops the compile rather than falling
     * back. */
    unsigned rreg = valueXReg(e, ridx - (e->depth - e->valueDepth));
    /* The receiver's shape is loaded ONCE and every way compares against it. */
    emit(e, jaiA64LdrX(JIT_SCRATCH_A, rreg,
                       (unsigned)offsetof(ObjInstance, klass)));
    emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_A,
                       (unsigned)offsetof(ObjClass, shapeId)));

    /* The model as the fall-through must find it again: emitDirectCall
     * consumes the receiver and the arguments and pushes a result, and the
     * fall-through's own descriptor call has to see the SAME depth and
     * valueDepth it would have without any of this, or the two paths
     * disagree about where the result lands. */
    unsigned  saveDepth      = e->depth;
    unsigned  saveValueDepth = e->valueDepth;
    SlotKind  saveKind [JIT_MAX_STACK];
    uint32_t  saveShape[JIT_MAX_STACK];
    ObjClass *saveClass[JIT_MAX_STACK];
    Value     saveSeen [JIT_MAX_STACK];
    memcpy(saveKind,  e->stack,      sizeof saveKind);
    memcpy(saveShape, e->stackShape, sizeof saveShape);
    memcpy(saveClass, e->stackClass, sizeof saveClass);
    memcpy(saveSeen,  e->stackSeen,  sizeof saveSeen);

    /* One arm a way. Each is: prove the shape, call directly, jump to the
     * merge. A way that does not match falls into the next way's compare, and
     * the last falls into the descriptor path the caller emits -- which is
     * what this site did for every receiver before any of this. */
    int armMiss[JAI_IC_WAYS];
    for (unsigned w = 0; w < ways; w++) {
        emitConst64(e, JIT_SCRATCH_B, (int64_t)wayShape[w]);
        emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
        armMiss[w] = (int)e->count;
        emit(e, jaiA64BCond(JAI_A64_NE, 0));

        /* True on this arm alone, which is why it is also what the CALL's own
         * deopt records (inside emitDirectCall, for what happens after the
         * call starts -- not for this compare) should say: the branch just
         * above proved it. */
        e->depth      = saveDepth;
        e->valueDepth = saveValueDepth;
        memcpy(e->stack,      saveKind,  sizeof saveKind);
        memcpy(e->stackShape, saveShape, sizeof saveShape);
        memcpy(e->stackClass, saveClass, sizeof saveClass);
        memcpy(e->stackSeen,  saveSeen,  sizeof saveSeen);
        e->stack[ridx]      = SLOT_INST;
        e->stackShape[ridx] = wayShape[w];
        e->stackClass[ridx] = wayCls[w];

        if (!emitDirectCall(e, fn, wayFn[w], wayVal[w], -1, ridx, argc,
                            callOff, after, true)) {
            /* jitPic1Admissible said it would take this and it did not: the
             * branch into it is already emitted, so there is nowhere left to
             * fall back to. */
            e->failed = true;
            return false;
        }
        if (e->picExitCount >= JIT_MAX_PIC_EXITS) {
            e->failed = true;
            return false;
        }
        e->picExits[e->picExitCount++] = (int)e->count;
        emit(e, jaiA64B(0));

        /* Same condition as the placeholder (NE: skip the call on a shape
         * that doesn't match), now with the real offset -- flipping it to EQ
         * would call this way's callee on every receiver whose shape did NOT
         * match, reading its fields at the wrong class's layout. That is what
         * test_mixed_list_runs_every_class caught: a wrong answer, not a
         * crash, because the read lands inside the instance's own allocation.
         *
         * Guarded because emit() silently DROPS the word once e->count reaches
         * JIT_MAX_INSTS and only sets e->failed -- so the slot may never have
         * been written, and Emit::code is immediately followed by `count` with
         * no padding between them. */
        if (armMiss[w] < (int)e->count && e->count <= JIT_MAX_INSTS) {
            e->code[armMiss[w]] =
                jaiA64BCond(JAI_A64_NE, (int32_t)((int)e->count - armMiss[w]));
        }
    }

    /* The model the caller's descriptor path must find, restored exactly. */
    e->depth      = saveDepth;
    e->valueDepth = saveValueDepth;
    memcpy(e->stack,      saveKind,  sizeof saveKind);
    memcpy(e->stackShape, saveShape, sizeof saveShape);
    memcpy(e->stackClass, saveClass, sizeof saveClass);
    memcpy(e->stackSeen,  saveSeen,  sizeof saveSeen);

    if (getenv("JAI_JIT_WHY")) {
        fprintf(stderr, "[jit] pic %u-way (of %u recorded, state %d) at %u\n",
                ways, (unsigned)ic->count, (int)ic->state, callOff);
    }

    /* The fall-through runs with the receiver and the arguments untouched,
     * so the caller emits its descriptor path against the model as it was
     * before any of this. */
    e->depth      = saveDepth;
    e->valueDepth = saveValueDepth;
    memcpy(e->stack,      saveKind,  sizeof saveKind);
    memcpy(e->stackShape, saveShape, sizeof saveShape);
    memcpy(e->stackClass, saveClass, sizeof saveClass);
    memcpy(e->stackSeen,  saveSeen,  sizeof saveSeen);
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
    /* An inlined body's entries want x0..x8 (inlineOwnBank) and a split bank
     * is already using them, so the plan withholds the split from a body the
     * measuring pass saw inline. Refusing here as well is what makes that a
     * fact rather than an agreement between two passes: the worst this can do
     * is decline an inline the probe never took. */
    if (e->splitAt != 0) return false;
    ObjFunction *cfn = callee->fn;
    if (cfn->module != caller->module) return false;
    if (e->inlining) return false;             /* one level, no recursion */
    /* The inlined body's offsets are the callee's, so `inProtected` describes
     * the CALLER's regions throughout -- a `try` of the callee's own would go
     * unseen. inlinableBody's whitelist already refuses every opcode a handler
     * needs; this says so rather than relying on it. */
    if (cfn->exceptionCount > 0) return false;
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
    /* Read while `inlining` is still set, so this names the inlined bank's d
     * register; the caller's own is taken after it is cleared. */
    bool rfp = (e->fpLive & (1u << (e->valueDepth - 1))) != 0;
    unsigned rfpReg = rfp ? fpHeldIn(e, e->valueDepth - 1) : 0;
    if (rfp) {
        if (!popValueRaw(e, &rres, &kres)) { e->failed = true; return false; }
    } else if (!popValue(e, &rres, &kres)) { e->failed = true; return false; }
    /* Raw, because nothing reads these again: the body is over and its pinned
     * locals go with it, so materialising one costs an instruction whose
     * destination is dead. */
    while (e->depth > cidx) {
        if (holdsRegister(e->stack[e->depth - 1])) {
            unsigned r;
            if (!popValueRaw(e, &r, NULL)) { e->failed = true; return false; }
        } else {
            e->depth--;
        }
    }
    e->inlining = false;
    memcpy(e->inlSlot, savedSlot, sizeof savedSlot);

    if (!pushValue(e, kres, rshape, rcls)) { e->failed = true; return false; }
    if (rfp) {
        unsigned dd = fpRegAt(e, e->valueDepth - 1);
        if (dd != rfpReg) emit(e, jaiA64FmovDD(dd, rfpReg));
        fpClaim(e, e->valueDepth - 1);
    } else {
        unsigned dst = pushReg(e) - 1;
        if (dst != rres) emit(e, jaiA64MovX(dst, rres));
    }
    e->inlined = true;
    return true;
}

/* What a descriptor call does with its result, once the arguments are consumed:
 * the predicted kind types the entry pushed for it, the tag that actually comes
 * back is checked, and a surprise deopts to the instruction AFTER the call --
 * which has happened and must not happen twice.
 *
 * Shared by the global-call and module-call arms below. They differ only in
 * what they consume before it and in how the callee was resolved; from the
 * descriptor's `result` onwards there is nothing to tell them apart. */
static bool emitCallOutResult(Emit *e, SlotKind rk, uint32_t rshape,
                              ObjClass *rcls, uint32_t after) {
    if (!pushValue(e, rk, rshape, rcls)) return false;

    unsigned rat = e->descOffset + (unsigned)offsetof(JitCallDesc, result);
    if (rk == SLOT_MAYBE_INST) {
        emitMaybeInstResult(e, pushReg(e) - 1, rat, rshape, after);
        return true;
    }
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
    /* A byte, not a word: BOOL_VAL writes the union's `bool` member and leaves
     * the other seven bytes of the payload indeterminate, so a 64-bit load
     * brings back whatever the slot held before. The register stands for a
     * bool from here on and everything downstream tests it against zero, so
     * those bytes read as true -- `values.map(|v| is_nan(v))` came back all
     * true over a list with no NaN in it. Every other descriptor return site
     * already splits the two; this one did not. */
    else if (rk == SLOT_BOOL) emit(e, jaiA64LdrByte(pushReg(e) - 1, 31, rat + 8));
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
    } else if (rk == SLOT_LIST) {
        /* Same hazard as SLOT_INST above: a callee entered with another
         * specialisation runs interpreted and may return any type, so
         * VAL_OBJ alone does not prove the payload is a list before
         * downstream code reads ObjList's fields off it unguarded. */
        emit(e, jaiA64LdrW(JIT_SCRATCH_A, pushReg(e) - 1,
                           (unsigned)offsetof(Obj, type)));
        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_LIST));
        branchOnDeoptAt(e, JAI_A64_NE, after, true);
    }
    e->wroteHeap = true;
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
    /* Observed first, compiled kind second -- the order the sibling site at
     * emitGlobalCall spells out, and which this one did not have.
     *
     * `jitFunc != NULL` does not make `jitReturnKind` a fact; it is stored
     * unconditionally at the end of a compile, so a body that compiled a
     * PREFIX and took the unarmed path before any OP_RETURN advertises the
     * SLOT_INT a zeroed Emit starts on. That is not hypothetical here:
     * `_is_ident_start` in the lexer walks only to OP_GET_GLOBAL at offset 0
     * -- a cold `throw` on its first instruction -- so it compiles nothing and
     * still claims to return an int. `_is_ident_cont`, which is
     * `_is_ident_start(c) or _is_digit(c)`, then declined on "a branch on a
     * int, not a bool", and `_ident_run_end`'s OSR loop retried it eighty
     * times waiting for a function that could never compile.
     *
     * A refusal is a chain, and this was three links of one. */
    SlotKind rk = SLOT_NULL;
    uint32_t rshape = 0;
    ObjClass *rcls = NULL;
    bool haveKind = observedReturnKind(cfn, &rk, &rshape);
    if (!haveKind && cfn->jitFunc != NULL) {
        rk = (SlotKind)cfn->jitReturnKind;
        rshape = cfn->jitReturnShape;
        haveKind = true;
    }
    if (haveKind && (rk == SLOT_INST || rk == SLOT_MAYBE_INST) &&
        (rshape == 0 || !jaiClassForShape(rshape, &rcls) || rcls == NULL)) {
        e->whyNot = "callee's return class not on record";
        return false;
    }
    if (!haveKind ||
        (rk != SLOT_INT && rk != SLOT_FLOAT && rk != SLOT_BOOL &&
         rk != SLOT_INST && rk != SLOT_MAYBE_INST && rk != SLOT_LIST &&
         rk != SLOT_OBJ && rk != SLOT_NULL)) {
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
    return emitCallOutResult(e, rk, rshape, rcls, after);
}

/* `math.sqrt(x)` is not a method call. It is a global call whose callee is
 * resolved through ANOTHER module, so what was missing was never the call
 * machinery -- it is the pinning.
 *
 * Two guards, in this order, both before anything is consumed so a miss resumes
 * at the invoke with the receiver and the arguments untouched:
 *
 *   THIS module. OP_GET_GLOBAL loads `math` by address behind a VAL_OBJ tag
 *   guard, and the arm's own object-type guard would only prove "some module",
 *   so the receiver is compared against the ObjModule this compiled against.
 *   Everything below -- the version word's address, the resolved closure --
 *   belongs to that one module.
 *
 *   STILL this binding. ObjModule::version is the counter for a memoised
 *   global VALUE or a resolved callee, and it is the one that moves when
 *   `math.sqrt = f` overwrites the member (jaiModuleSet bumps it because a
 *   closure is not inert). `globals.keyVersion` is the WRONG counter here and
 *   would be silent: overwriting an existing key never moves its entry, which
 *   is the whole reason keyVersion exists. Guarding it per call rather than
 *   at entry also covers a rebinding from inside the loop, which the entry
 *   check by itself does not -- see the note at OP_SET_GLOBAL.
 *
 * The callee must have compiled. The standing warning at OP_GET_GLOBAL says
 * admitting a callee whose jitFunc is NULL MISCOMPILES for a reason nobody has
 * written down; this arm does not cross it, and pays nothing for that --
 * `math.sqrt` compiles long before any loop calling it does.
 *
 * The receiver is dropped rather than passed: a module is not an argument. */
static bool emitModuleCall(Emit *e, ObjModule *m, Value calleeVal,
                           unsigned ridx, unsigned argc, uint32_t after) {
    ObjFunction *cfn = AS_CLOSURE(calleeVal)->fn;
    if (cfn->jitFunc == NULL) {
        return subWhy(e, "a module member that has not compiled");
    }
    /* WHICH RECORD SAYS WHAT COMES BACK, and it is not the obvious one.
     * `jitFunc != NULL` does NOT make `jitReturnKind` a fact: it is stored
     * unconditionally at the end of a compile, so a body whose walk never
     * reached an OP_RETURN -- one that compiled a prefix and bails -- leaves it
     * at the SLOT_INT a zeroed Emit starts on. math.sqrt and math.sin are both
     * exactly that (`sawReturn=0`, `obsReturnKind=float`), so trusting the
     * compiled kind here predicted int, the tag guard below failed on the first
     * call, and the loop left compiled code every iteration: p27 did not move at
     * all. The interpreter's own per-callee record is a measured fact and is
     * asked first; the compiled kind stands in only when there is none.
     *
     * Either way it is a prediction, and the tag guard after the call is what
     * makes it sound -- the same contract emitGlobalCall states. */
    SlotKind rk = SLOT_NULL;
    uint32_t rshape = 0;
    ObjClass *rcls = NULL;
    if (!observedReturnKind(cfn, &rk, &rshape)) {
        rk = (SlotKind)cfn->jitReturnKind;
        rshape = cfn->jitReturnShape;
    }
    if (rk != SLOT_INT && rk != SLOT_FLOAT && rk != SLOT_BOOL &&
        rk != SLOT_INST && rk != SLOT_MAYBE_INST && rk != SLOT_LIST &&
        rk != SLOT_OBJ && rk != SLOT_NULL) {
        return subWhy(e, "a module member's return kind (%d)", (int)rk);
    }
    if ((rk == SLOT_INST || rk == SLOT_MAYBE_INST) &&
        (rshape == 0 || !jaiClassForShape(rshape, &rcls) || rcls == NULL)) {
        return subWhy(e, "a module member's return class is not on record");
    }
    /* Every entry this consumes, receiver included, has to be one the
     * descriptor can pass -- asked HERE rather than left to emitDescriptor,
     * which refuses only after the guards below have been emitted and would
     * turn a graceful decline into a whole-body one. It also settles the
     * register arithmetic: the receiver is named by counting back from the top
     * of the value bank, which is the same thing as counting back from the top
     * of the model only while every entry between them holds a register, and
     * every kind admitted here does. */
    for (unsigned i = 0; i <= argc; i++) {
        SlotKind k = e->stack[ridx + i];
        if (k != SLOT_INT && k != SLOT_FLOAT && k != SLOT_BOOL &&
            k != SLOT_INST && k != SLOT_LIST && k != SLOT_OBJ &&
            k != SLOT_ITER && k != SLOT_MAYBE_INST) {
            return subWhy(e, "a module call over an entry of kind %d", (int)k);
        }
    }
    /* Past here the guards are emitted, so a later refusal stops the compile
     * rather than falling back onto a half-written instruction stream. */

    settleAll(e);
    unsigned rRecv = valueXReg(e, e->valueDepth - argc - 1);
    emitConst64(e, JIT_SCRATCH_C, (int64_t)(uintptr_t)m);
    emit(e, jaiA64SubsXReg(31, rRecv, JIT_SCRATCH_C));
    branchOnDeopt(e, JAI_A64_NE);

    emitConst64(e, JIT_SCRATCH_D, (int64_t)(uintptr_t)&m->version);
    emit(e, jaiA64LdrW(JIT_SCRATCH_C, JIT_SCRATCH_D, 0));
    emitConst64(e, JIT_SCRATCH_B, (int64_t)(uint32_t)m->version);
    emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_C, JIT_SCRATCH_B));
    branchOnDeopt(e, JAI_A64_NE);

    if (!emitDescriptor(e, calleeVal, ridx + 1, argc, (void *)&jitCallOut)) {
        return false;
    }
    for (unsigned i = 0; i <= argc; i++) {
        unsigned r;
        if (!popValue(e, &r, NULL)) return false;
    }
    return emitCallOutResult(e, rk, rshape, rcls, after);
}

/* `Klass.static_method(args)` -- an OP_INVOKE whose receiver is a CLASS. The
 * same job emitModuleCall does for `math.sqrt(x)`, and for the same reason: a
 * member of another namespace, resolved at compile time, called with the
 * receiver DROPPED. The interpreter drops it too. resolveInvokeTarget (vm.c)
 * leaves the class in slot 0 and reports `isMethod == false`, so the call goes
 * through invokeCallable and the closure's first parameter is the first
 * ARGUMENT, not the class -- which is exactly what jitCallOut does with a
 * descriptor holding argc arguments and no receiver.
 *
 * Three things differ from the module case, and only the third is machinery.
 *
 *   THE RECEIVER NEEDS NO GUARD. A module receiver is loaded from a global by
 *   address behind a bare VAL_OBJ tag check, so emitModuleCall has to compare
 *   it against the one ObjModule it compiled against. A class receiver is not
 *   in a register at all: SLOT_CLASS holds nothing (holdsRegister says so) and
 *   the ObjClass came from OP_GET_GLOBAL's `globalClass` arm, which resolves it
 *   BY VALUE and is retired wholesale by the module version check at entry if
 *   the name is rebound. There is nothing here that could be a different class
 *   at run time than it was at compile time.
 *
 *   THE MEMBER IS RESOLVED OUT OF `klass->statics`, DIRECTLY. Not through
 *   jaiBuiltinMethod, and not through anything that can hand back a BoundMethod
 *   to unwrap: unwrapping one and then dropping the receiver as this arm does
 *   loses both halves and calls an unbound closure with the first real argument
 *   sitting where `self` belongs. Only an IS_CLOSURE value straight out of the
 *   table is admitted. `klass->methods` is deliberately NOT consulted -- see
 *   the call site, which declines an instance method named through the class.
 *
 *   WHAT RETIRES THE BAKED CALLEE is the BINDING itself, re-read from the
 *   statics entry on every call and compared against the closure this site was
 *   compiled against. `Klass.name = v` reaches jaiSetProperty's IS_CLASS arm,
 *   which requires the key to be present already and then overwrites in place,
 *   so the entry never moves and reading it back asks the direct question: is
 *   this still what I compiled for.
 *
 *   It was a COUNTER first -- `klass->statics.version`, which tableSetHashed
 *   bumps on every value write -- and that was unsound. The field is uint32_t
 *   (table.h) and nothing filters the bump the way jaiModuleSet filters
 *   ObjModule::version through jaiValueIsInertGlobal, so an ordinary
 *   `Klass.counter = n` loop drives it at 51M writes/sec, measured. Land the
 *   count exactly 2^32 on from the bake and the guard reads the value it baked
 *   while the binding has changed: `Box.make` rebound after 2^32 writes
 *   answered 400008, against 2000000 from the interpreter, from the arm
 *   switched off, and from a control one write short. That is about 84 seconds
 *   of a loop any program might contain, not an unreachable corner.
 *
 *   Reading the binding is also strictly less trigger-happy than the counter,
 *   which retired the callee whenever any OTHER static of the same class was
 *   written. It costs 1.8% of the win (0.551s -> 0.561s on a 32M-call probe
 *   against 1.130s with the arm off).
 *
 * `keyVersion` is still needed, for a different job: it makes the entry ADDRESS
 * trustworthy. It moves on rehash, delete and clear -- never on an overwrite
 * (table.c: insertAt bumps it only for a NEW key) -- so unlike `version` a
 * running program cannot drive it. It is the guard the static-FIELD arm stands
 * on, baked per site here rather than through e->staticsTable so that a body
 * naming two classes still compiles both. */
static bool emitClassCall(Emit *e, ObjClass *klass, JaiEntry *slot,
                          Value calleeVal, unsigned ridx, unsigned argc,
                          uint32_t after) {
    ObjFunction *cfn = AS_CLOSURE(calleeVal)->fn;
    if (cfn->jitFunc == NULL) {
        return subWhy(e, "a static that has not compiled");
    }
    /* Which record says what comes back: emitModuleCall's finding, and it is
     * not the obvious one. `jitFunc != NULL` does NOT make `jitReturnKind` a
     * fact -- it is stored unconditionally at the end of a compile, so a body
     * whose walk never reached an OP_RETURN leaves it at the SLOT_INT a zeroed
     * Emit starts on. The interpreter's own per-callee record is a measured
     * fact and is asked first; the compiled kind stands in only when there is
     * none. Either way it is a prediction, and emitCallOutResult's tag guard
     * after the call is what makes it sound. */
    SlotKind rk = SLOT_NULL;
    uint32_t rshape = 0;
    ObjClass *rcls = NULL;
    if (!observedReturnKind(cfn, &rk, &rshape)) {
        rk = (SlotKind)cfn->jitReturnKind;
        rshape = cfn->jitReturnShape;
    }
    if (rk != SLOT_INT && rk != SLOT_FLOAT && rk != SLOT_BOOL &&
        rk != SLOT_INST && rk != SLOT_MAYBE_INST && rk != SLOT_LIST &&
        rk != SLOT_OBJ && rk != SLOT_NULL) {
        return subWhy(e, "a static's return kind (%s)", slotKindName(rk));
    }
    if ((rk == SLOT_INST || rk == SLOT_MAYBE_INST) &&
        (rshape == 0 || !jaiClassForShape(rshape, &rcls) || rcls == NULL)) {
        return subWhy(e, "a static's return class is not on record");
    }
    /* The ARGUMENTS only. The receiver is skipped where emitModuleCall checks
     * it, because a class entry holds no register and emitDescriptorStatus
     * already knows how to bake one -- but it is never passed here, so even
     * that does not arise. Asked HERE rather than left to emitDescriptor, which
     * refuses only after the guard below has been emitted and would turn a
     * graceful decline into a whole-body one. */
    for (unsigned i = 1; i <= argc; i++) {
        SlotKind k = e->stack[ridx + i];
        if (k != SLOT_INT && k != SLOT_FLOAT && k != SLOT_BOOL &&
            k != SLOT_INST && k != SLOT_LIST && k != SLOT_OBJ &&
            k != SLOT_ITER && k != SLOT_MAYBE_INST) {
            return subWhy(e, "a static call over an argument of kind %s",
                          slotKindName(k));
        }
    }
    /* Past here the guard is emitted, so a later refusal stops the compile
     * rather than falling back onto a half-written instruction stream. */

    settleAll(e);
    emitConst64(e, JIT_SCRATCH_D,
                (int64_t)(uintptr_t)&klass->statics.keyVersion);
    emit(e, jaiA64LdrW(JIT_SCRATCH_C, JIT_SCRATCH_D, 0));
    emitConst64(e, JIT_SCRATCH_B, (int64_t)(uint32_t)klass->statics.keyVersion);
    emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_C, JIT_SCRATCH_B));
    branchOnDeopt(e, JAI_A64_NE);

    /* The tag before the pointer, so a static rebound to an int whose payload
     * happened to equal the closure's address is not called as if it were the
     * closure.
     *
     * Comparing against the LIVE binding is also what makes address recycling
     * harmless rather than dangerous. If the old closure were collected and a
     * new object took its address, the object bound NOW is the one at that
     * address, and emitDescriptor's callee slot goes through jaiCallValue,
     * which dispatches on the value dynamically -- including raising, if what
     * is bound there is not callable. */
    emitConst64(e, JIT_SCRATCH_D, (int64_t)(uintptr_t)slot);
    emit(e, jaiA64LdrW(JIT_SCRATCH_C, JIT_SCRATCH_D,
                       (unsigned)offsetof(JaiEntry, value)));
    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_C, VAL_OBJ));
    branchOnDeopt(e, JAI_A64_NE);
    emit(e, jaiA64LdrX(JIT_SCRATCH_C, JIT_SCRATCH_D,
                       (unsigned)offsetof(JaiEntry, value) + 8u));
    emitConst64(e, JIT_SCRATCH_B, (int64_t)(uintptr_t)AS_OBJ(calleeVal));
    emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_C, JIT_SCRATCH_B));
    branchOnDeopt(e, JAI_A64_NE);

    if (!emitDescriptor(e, calleeVal, ridx + 1, argc, (void *)&jitCallOut)) {
        return false;
    }
    for (unsigned i = 0; i < argc; i++) {
        unsigned r;
        if (!popValue(e, &r, NULL)) return false;
    }
    /* The receiver held no register, so dropping it is the whole of popping it
     * -- popValue would refuse it via holdsRegister. Same as the static-field
     * arm at OP_GET_FIELD. */
    if (e->depth == 0 || e->stack[e->depth - 1] != SLOT_CLASS) return false;
    e->depth--;
    return emitCallOutResult(e, rk, rshape, rcls, after);
}

/* Emit a method's body directly, when that body is one expression.
 *
 * Deliberately narrow: no jumps, no stores, no calls, only field reads of its
 * own parameters and int or float arithmetic. Those restrictions are what make
 * a second walker over the callee's bytecode safe to write -- with no branches
 * there is no offset map to keep, and with no stores there is nothing to undo
 * if a guard inside it deoptimises to the call site.
 *
 * Reached through inlineMethod, which is what puts the model back when this
 * declines -- see there. */
static bool inlineMethodWalk(Emit *e, ObjClosure *closure, uint32_t nameIdx,
                             unsigned argc, int callOff) {
    if (e->noInline) return false;
    if (e->splitAt != 0) return false;   /* see inlineGlobalCall */
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
    if (mfn->exceptionCount > 0) return false;   /* see inlineGlobalCall */
    if (mfn->arity != argc || mfn->defaultCount != 0) return false;
    if (mfn->flags & (FN_VARIADIC | FN_KWREST | FN_INIT)) return false;
    if (mfn->upvalueCount != 0) return false;
    if (mfn->chunk.count > 96) return false;

    unsigned inReg[JIT_MAX_ARGS_OUT + 1];
    Value    inSeen[JIT_MAX_ARGS_OUT + 1];
    ObjClass *inCls[JIT_MAX_ARGS_OUT + 1];
    for (unsigned i = 0; i <= argc; i++) {
        unsigned idx = ridx + i;
        if (!holdsRegister(e->stack[idx])) return false;
        inReg[i]  = valueBankReg(e, idx - (e->depth - e->valueDepth));
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

/* The model must be exactly where it was if the inline did not happen.
 *
 * inlineMethodWalk's dry pass pushes and pops as it reads the callee, and every
 * one of its two dozen refusals returns from the middle of that -- so on its
 * own it leaves the model as deep as the walk got. Its caller does NOT decline
 * when it declines: the OP_INVOKE arm falls through to the descriptor path,
 * which then names every later entry's register from an index that is too high
 * and, far worse, writes deopt records describing an operand stack the
 * interpreter does not have. `_crossings` in lib/std/gui/path.jai is the shape
 * that found this: `edge.crossing(y)` gets three instructions into `crossing`
 * before an OP_BIND stops the walk, so every deopt after it handed the
 * interpreter the receiver and the argument a second time and the next
 * instruction read a float where a list belonged.
 *
 * Unwinding here rather than at each `return false` is deliberate: there are
 * far too many of them to keep right by hand, and the dry pass's own tail
 * already pops back to its starting depth in exactly this way.
 *
 * A failure that has already emitted cannot be unwound at all -- the caller
 * would stack a second call sequence on top of half of this one -- so that
 * declines the compile instead. */
static bool inlineMethod(Emit *e, ObjClosure *closure, uint32_t nameIdx,
                         unsigned argc, int callOff) {
    unsigned depth0 = e->depth;
    unsigned count0 = e->count;
    if (inlineMethodWalk(e, closure, nameIdx, argc, callOff)) return true;
    if (e->failed) return false;
    if (e->count != count0) { e->failed = true; return false; }
    while (e->depth > depth0) {
        unsigned r;
        if (!popValue(e, &r, NULL)) { e->failed = true; return false; }
    }
    /* Below where it started is not something an unwind can repair: the
     * entries are the caller's and their registers are gone. */
    if (e->depth != depth0) { e->failed = true; return false; }
    return false;
}

/* The kind a container's elements may be held as, read off ONE live element.
 *
 * The sample specialises and the tag guard at the read site confirms: a
 * container that later holds something else deoptimises rather than being
 * answered wrongly. An instance carries its class too, since a tag check alone
 * cannot tell two shapes apart. */
static bool exemplarKind(Value elem, SlotKind *kind, unsigned *tag,
                         ObjClass **cls, uint32_t *shape) {
    *cls = NULL;
    *shape = 0;
    if (IS_INT(elem))        { *kind = SLOT_INT;   *tag = VAL_INT;   return true; }
    if (IS_FLOAT(elem))      { *kind = SLOT_FLOAT; *tag = VAL_FLOAT; return true; }
    if (IS_BOOL(elem))       { *kind = SLOT_BOOL;  *tag = VAL_BOOL;  return true; }
    if (IS_LIST(elem))       { *kind = SLOT_LIST;  *tag = VAL_OBJ;   return true; }
    if (rawObjValue(elem))   { *kind = SLOT_OBJ;   *tag = VAL_OBJ;   return true; }
    if (IS_INSTANCE(elem) && AS_INSTANCE(elem)->klass != NULL) {
        *kind  = SLOT_INST;
        *tag   = VAL_OBJ;
        *cls   = AS_INSTANCE(elem)->klass;
        *shape = (*cls)->shapeId;
        return true;
    }
    return false;
}

/* An exemplar for the elements a freshly-built list is about to hold.
 *
 * `[1, 2, 3]` has no live list to sample -- the list does not exist until the
 * compiled code runs -- but the model knows the kind of every entry that went
 * into it. For a scalar that is enough: a synthesised zero of the right kind
 * answers every question the iterate arm asks of a sample, and being an
 * immediate it has no lifetime to worry about. For an object the element's own
 * sample is reused, which was already being held on the operand stack.
 *
 * All `n` must agree, and an instance must agree on its class too, for the same
 * reason the dict walk insists on it: a mispredicted element kind deoptimises on
 * every read, which is worse than not compiling at all.
 *
 * Returns false when the list is empty (a comprehension's accumulator) or the
 * elements disagree -- in both cases there is simply nothing to predict. */
static bool buildListExemplar(const Emit *e, unsigned first, unsigned n,
                              Value *out) {
    if (n == 0) return false;
    Value chosen = NULL_VAL;
    for (unsigned i = 0; i < n; i++) {
        unsigned idx = first + i;
        Value here;
        switch (e->stack[idx]) {
        case SLOT_INT:   here = INT_VAL(0);        break;
        case SLOT_FLOAT: here = FLOAT_VAL(0.0);    break;
        case SLOT_BOOL:  here = BOOL_VAL(false);   break;
        case SLOT_OBJ:
        case SLOT_LIST:
        case SLOT_INST:
            here = e->stackSeen[idx];
            if (!IS_OBJ(here) || AS_OBJ(here) == NULL) return false;
            break;
        default:
            return false;
        }
        if (i == 0) { chosen = here; continue; }
        if (jaiValueType(here) != jaiValueType(chosen)) return false;
        if (IS_OBJ(here) && OBJ_TYPE(here) != OBJ_TYPE(chosen)) return false;
        if (IS_INSTANCE(here) &&
            AS_INSTANCE(here)->klass != AS_INSTANCE(chosen)->klass) {
            return false;
        }
    }
    *out = chosen;
    return true;
}

/* One value out of a live dict, and only if every value in it agrees.
 *
 * A LIST is sampled at index 0 alone, because a list's elements are usually
 * built by one loop and a wrong guess costs a deopt per read. A dict is not:
 * `{"name": "x", "count": 3}` is an ordinary dict and its values disagree, so
 * predicting off the first entry would deoptimise every iteration -- measured
 * elsewhere at 5.7x worse than declining outright (chunk.h states the same for
 * an invoke's result). Refusing a mixed dict is the point of the walk.
 *
 * Capped, so compiling a body that indexes a large dict does not walk it. Past
 * the cap the guard still holds; only the prediction is made on a prefix. */
#define JIT_DICT_SAMPLE_MAX 256u

static bool dictUniformValue(ObjDict *dict, Value *out) {
    const JaiTable *t = &dict->table;
    if (t->entries == NULL || t->count <= 0) return false;
    bool have = false;
    Value first = NULL_VAL;
    unsigned seen = 0;
    for (int i = 0; i < t->capacity && seen < JIT_DICT_SAMPLE_MAX; i++) {
        /* A live entry is one with a nonnegative order; empty and tombstoned
         * slots both carry a negative one (see table.c's entryIsLive). */
        if (t->entries[i].order < 0) continue;
        Value v = t->entries[i].value;
        seen++;
        if (!have) { first = v; have = true; continue; }
        if (jaiValueType(v) != jaiValueType(first)) return false;
        if (IS_OBJ(v) && OBJ_TYPE(v) != OBJ_TYPE(first)) return false;
        if (IS_INSTANCE(v) && AS_INSTANCE(v)->klass != AS_INSTANCE(first)->klass) {
            return false;
        }
    }
    if (!have) return false;
    *out = first;
    return true;
}

/* Builtins whose result kind is a property of the FUNCTION and not of its
 * arguments: `str(x)` is a string whatever x is, `len(x)` is an int, `bool(x)`
 * is a bool. That is the only thing the surrounding body needs to know, so the
 * call can be an ordinary call out and everything around it stays compiled.
 *
 * Worth having because the alternative was not a slower call but no compiled
 * body at all -- one `str()` in a loop declined the whole enclosing function.
 * A probe doing `str(i % 10_000)` per iteration ran 90,000,323 interpreted
 * instructions against 1,231 for the f-string spelling of the same thing.
 *
 * The kinds are read off the natives in builtins_core.c, and the returned tag
 * is guarded regardless: a wrong row here costs a deopt, never an answer. */
typedef struct {
    const char *name;
    unsigned    argc;
    SlotKind    kind;
    uint8_t     tag;
} NativeResult;

static const NativeResult kNativeResults[] = {
    { "str",        1, SLOT_OBJ,  VAL_OBJ  },
    { "repr",       1, SLOT_OBJ,  VAL_OBJ  },
    { "chr",        1, SLOT_OBJ,  VAL_OBJ  },
    { "type_of",    1, SLOT_OBJ,  VAL_OBJ  },
    { "len",        1, SLOT_INT,  VAL_INT  },
    { "hash",       1, SLOT_INT,  VAL_INT  },
    { "id",         1, SLOT_INT,  VAL_INT  },
    { "ord",        1, SLOT_INT,  VAL_INT  },
    { "int",        1, SLOT_INT,  VAL_INT  },
    { "int",        2, SLOT_INT,  VAL_INT  },
    { "bool",       1, SLOT_BOOL, VAL_BOOL },
    { "callable",   1, SLOT_BOOL, VAL_BOOL },
    { "isinstance", 2, SLOT_BOOL, VAL_BOOL },
    /* No `range` row. It would emit, but the loop that consumes the result
     * declines one instruction later ("iterating something other than a list
     * or range") because SLOT_OBJ does not say `range` -- so the body is
     * refused either way and the row only buys an allocation. */
};

/* `sum`, `min` and `max` over a LIST answer with the element's own kind: an int
 * list sums to an int and a float list to a float (checked against `type_of`,
 * not assumed). So unlike every row in kNativeResults their result is a
 * property of the ARGUMENT, and the table cannot state it.
 *
 * Worth the separate arm because they are cheap next to the loop around them,
 * which is the test a builtin row has to pass -- `sorted` is not, and its rows
 * were built and discarded for measuring zero
 * (docs/research/FALSIFIED-list-returning-builtins.md). A five-element `sum` is
 * five adds; `docs/probes/p20_sum_min_max.jai` ran 17,000,327 interpreted
 * instructions, i.e. the whole loop, on account of this one refusal.
 *
 * The exemplar comes from a live list if the model has one and otherwise from
 * `stackElem`, which is what OP_BUILD_LIST recorded -- the same two sources the
 * iterate arm reads, in the same order. */
/* JAITHON_JIT_LIST_SCALAR=0 turns the arm below off, so it can be A/B'd inside
 * ONE binary. Not a nicety: two-binary comparisons are where this tree's
 * measurements go wrong -- each switch invalidates __jaicache__, and three
 * people measuring one change tonight two-binary got 3.9x, 100x and 6%
 * SLOWER for what a switch settled in one command. */
/* JAITHON_JIT_NEGATE=0 turns off the OP_NEG arm, for a one-binary A/B. */
static bool jitSoftField(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_SOFT_FIELD");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

static bool jitMembership(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_MEMBERSHIP");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

static bool jitTuple(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_TUPLE");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

static bool jitNegate(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_NEGATE");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

/* JAITHON_JIT_LIST_RESULT=0 turns off the predicted result for a list method
 * that is neither a field read nor discarded, for a one-binary A/B. */
static bool jitListResult(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_LIST_RESULT");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

static bool jitListScalarResult(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_LIST_SCALAR");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

static bool listScalarResult(const Emit *e, const char *nm, unsigned argc,
                             SlotKind *kind, uint8_t *tag) {
    if (!jitListScalarResult()) return false;
    if (argc != 1 && !(argc == 2 && strcmp(nm, "sum") == 0)) return false;
    if (strcmp(nm, "sum") != 0 && strcmp(nm, "min") != 0 &&
        strcmp(nm, "max") != 0) {
        return false;
    }
    unsigned idx = e->depth - argc;
    if (e->stack[idx] != SLOT_LIST) return false;

    Value elem = NULL_VAL;
    Value seen = e->stackSeen[idx];
    if (IS_LIST(seen) && AS_LIST(seen)->count > 0) {
        elem = jaiListGet(AS_LIST(seen), 0);
    }
    if (IS_NULL(elem)) elem = e->stackElem[idx];

    if (IS_INT(elem))   { *kind = SLOT_INT;   *tag = VAL_INT;   return true; }
    if (IS_FLOAT(elem)) { *kind = SLOT_FLOAT; *tag = VAL_FLOAT; return true; }
    return false;
}

/* 1 emitted, 0 no row for this builtin, -1 the emit failed. */
static int emitNativeResultCall(Emit *e, Value cv, const char *nm,
                                unsigned argc, uint32_t afterIp) {
    /* inlinableBody admits OP_CALL only for the two builtins the tier emits as
     * a single instruction, on the grounds that an inlined body cannot call.
     * Refusing here keeps that true even if a constant string reaches an
     * `int()` inside one. */
    if (e->inlining) return 0;

    NativeResult derived;
    const NativeResult *nr = NULL;
    for (size_t i = 0; i < sizeof kNativeResults / sizeof kNativeResults[0]; i++) {
        if (kNativeResults[i].argc == argc &&
            strcmp(kNativeResults[i].name, nm) == 0) {
            nr = &kNativeResults[i];
            break;
        }
    }
    if (nr == NULL) {
        SlotKind dk;
        uint8_t dtag;
        if (!listScalarResult(e, nm, argc, &dk, &dtag)) return 0;
        derived.name = nm;
        derived.argc = argc;
        derived.kind = dk;
        derived.tag  = dtag;
        nr = &derived;
    }

    if (!emitDescriptor(e, cv, e->depth - argc, argc, (void *)&jitCallOut)) {
        return -1;
    }
    for (unsigned i = 0; i < argc; i++) {
        /* A class argument occupies no register, so there is nothing to pop --
         * only the entry to drop. */
        if (e->depth > 0 && !holdsRegister(e->stack[e->depth - 1])) {
            e->depth--;
            continue;
        }
        unsigned r;
        if (!popValue(e, &r, NULL)) { e->whyNot = "call argument"; return -1; }
    }
    if (e->depth == 0 || e->stack[e->depth - 1] != SLOT_NATIVE) {
        e->whyNot = "callee was not where it should be";
        return -1;
    }
    e->depth--;
    if (!pushValue(e, nr->kind, 0, NULL)) return -1;

    unsigned at = e->descOffset + (unsigned)offsetof(JitCallDesc, result);
    emit(e, jaiA64LdrW(JIT_SCRATCH_A, 31, at));
    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, nr->tag));
    /* Resumes AFTER the call, taking the result from the descriptor: the
     * native has already run and may have written, so re-running it is not on
     * offer. `lastFromDesc` is what hands the interpreter the Value the native
     * actually produced, whatever tag it turned out to have. */
    branchOnDeoptAt(e, JAI_A64_NE, afterIp, true);
    /* A bool is ONE byte of the Value union; reading eight would carry the
     * neighbouring bytes of the result slot into the register. */
    if (nr->kind == SLOT_BOOL) {
        emit(e, jaiA64LdrByte(pushReg(e) - 1, 31, at + 8));
    } else {
        emit(e, jaiA64LdrX(pushReg(e) - 1, 31, at + 8));
    }
    /* A call is an effect: no bail may follow it. */
    e->wroteHeap = true;
    return 1;
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
            regs[i] = valueBankReg(e, first + i - (e->depth - e->valueDepth));
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
/* The depth an emitted branch to `off` would reconcile the model to, or -1 for
 * none. Split out because emitUnarmedDeopt has to ask the question WITHOUT
 * answering it: a resume point it cannot reconcile is one it must not stop at.
 * A "no" here is usually the kinds disagreeing, not the depth -- the signature
 * carries both -- and that is exactly a join this tier cannot compile. */
static int reconcileDepth(const Emit *e, uint32_t off) {
    for (unsigned i = 0; i < e->fixupCount; i++) {
        if (e->fixups[i].targetOffset != off) continue;
        int want = e->fixups[i].depth;
        if (want < 0) continue;
        unsigned d = (unsigned)want & 0xfu;
        if (d > e->depth) continue;
        if ((int)stackSignatureAt(e, d) != want) continue;
        if (e->depth - d > e->valueDepth) continue;
        return (int)d;
    }
    return -1;
}

static void reconcileAfterUncond(Emit *e, uint32_t off) {
    int d = reconcileDepth(e, off);
    if (d < 0) return;
    e->valueDepth -= e->depth - (unsigned)d;
    e->depth = (unsigned)d;
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
    /* Its arm settles both operands itself on every path that reads a register
     * for them, and takes no deopt record before doing so. */
    case OP_FLOORDIV:
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
    /* Same again: a power-of-two divisor is spelt by the `asr`'s own shift field. Says yes for divisors
     * that are not powers of two too, which is harmless -- that path settles what it cannot fold, exactly as this list's contract allows. spectral's `s * (s + 1) // 2` paid a `movz x3,#2` into a register no instruction read, once per inner iteration. */
    case OP_FLOORDIV:
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

/* Inside a `try`: an entry of the function's own static exception table covers
 * this offset. Linear over the table because a function has one or two entries,
 * never a table worth indexing. */
static bool offsetIsProtected(const ObjFunction *fn, uint32_t off) {
    for (uint16_t i = 0; i < fn->exceptionCount; i++) {
        const ExceptionEntry *x = &fn->exceptions[i];
        if (off >= x->start && off < x->end) return true;
    }
    return false;
}

/* An opcode with no arm in the switch below.
 *
 * It used to `return false`, which declines the WHOLE enclosing function -- so
 * a `try` whose catch block holds OP_GET_EXC, or a `{}` literal on a path that
 * never runs, evicted every function containing it from the compiled tier for
 * good. 66 of 137 opcodes were in that position (tests/vm/jit_unarmed.baseline)
 * and `tests/bench/error_paths`' eight-million-iteration counted loop was one
 * of them: 124ms interpreted against 12ms compiled, decided by a catch block
 * the opcode histogram says runs zero times.
 *
 * Instead, deoptimise unconditionally here. The interpreter resumes at this
 * exact instruction holding exactly what the model says, which is what every
 * failed guard already does -- the cold path is interpreted and the hot path
 * around it still compiles. It declines only when a deopt site cannot be built
 * at this offset, which is deoptSite's existing "a guard resumes an instruction
 * whose operands it has already consumed" test and exactly the right question.
 *
 * Nothing on the fall-through edge past an unconditional deopt can execute, so
 * the walk does not model it: it skips ahead. Advancing the model by the
 * opcode's static stack effect instead was the other option and buys nothing --
 * every instruction it would emit is unreachable -- while needing a kind for
 * whatever the opcode produced, which the tier by definition does not have for
 * an opcode it cannot compile. A skipped offset keeps offsetToInst == -1, so a
 * branch that does target one declines when the fixups are resolved rather than
 * jumping into nothing.
 *
 * WHERE it may resume is the part that has to be earned. Skipping to the next
 * offset anything in the chunk can branch to was not enough: the branch that
 * reaches it may itself have been inside the skipped run, in which case nothing
 * establishes the model there and the walk carries on with the one it had at
 * the deopt. That is silent -- the registers all shift by the same amount, so
 * the body stays self-consistent -- right up until a deopt record names an
 * operand stack the interpreter has not got. `_is_any` in
 * lib/jaithon/compile/check/decl.jai is the shape: its `t is null` is unarmed,
 * and the only branch to the code after it is the `if`'s own jump, three bytes
 * into the skipped run.
 *
 * So it resumes only where the model can be stated:
 *   - an offset an already-emitted branch targets, which reconcileAfterUncond
 *     trims to that branch's own depth on the next iteration; or
 *   - an offset the BYTECODE says has an empty operand stack, where there are
 *     no entries and therefore no kinds to invent. This is what keeps a `while`
 *     whose head has not been compiled yet -- a back edge is a branch target
 *     the walk has not reached -- from being lost.
 * Anything else keeps skipping, and running out means the rest of the function
 * is not compiled: the last thing emitted is the unconditional branch to the
 * deopt stub, so control never falls off the end. */
static bool skipResumeOk(const Emit *e, uint32_t at) {
    if (reconcileDepth(e, at) >= 0) return true;
    return e->chunkDepth != NULL && at < (uint32_t)e->chunkDepthCount &&
           e->chunkDepth[at] == 0;
}

static bool emitUnarmedDeopt(Emit *e, const Chunk *c, int *off, int stop) {
    /* Remembered for the fixup pass. When this stops the walk before a forward
     * branch's target, that branch cannot be resolved and the error reported is
     * "a branch to an offset this walk never emitted" -- which names the
     * SYMPTOM. The opcode that actually stopped the walk is the cause, and
     * without it the reader has a bytecode offset and no idea why. */
    e->unarmedOp = e->lastOp;
    e->unarmedAt = e->curOffset;
    if (e->inlining) {
        /* Half an inlined body cannot be taken back, and the caller reads the
         * result out of the model -- past a deopt there is none. */
        e->whyNot = "an inlined body reaching an opcode this tier cannot speak";
        return false;
    }
    unsigned k;
    if (!deoptRecordAt(e, e->curOffset, false, &k)) return false;
    if (e->fixupCount >= JIT_MAX_FIXUPS) { e->failed = true; return false; }
    e->fixups[e->fixupCount].instIndex    = (int)e->count;
    e->fixups[e->fixupCount].targetOffset = FIXUP_DEOPT - k;
    e->fixups[e->fixupCount].conditional  = false;
    e->fixups[e->fixupCount].depth        = -1;
    e->fixupCount++;
    emit(e, jaiA64B(0));

    int at = *off;
    for (;;) {
        int len = instructionLength(c, at);
        if (len <= 0) return false;      /* undecodable: the walk is lost */
        at += len;
        if (at >= stop) break;
        if (offsetIsBranchTarget(c, (uint32_t)at) &&
            skipResumeOk(e, (uint32_t)at)) {
            /* Nothing an emitted branch reaches, so reconcileAfterUncond has
             * nothing to trim to and the empty stack the bytecode promises is
             * the whole model. Cleared raw rather than through popValue: this
             * sits immediately after an unconditional branch, so an fmov
             * settling an entry nobody will read is dead code. */
            if (reconcileDepth(e, (uint32_t)at) < 0) {
                e->depth      = 0;
                e->valueDepth = 0;
                e->fpLive     = 0;
                e->fpBorrow   = 0;
                e->kPend      = 0;
                e->xBorrow    = 0;
            }
            break;
        }
    }
    if (at > stop) at = stop;
    *off = at;
    return true;
}

/* Why an arm jumps to `unarmedOpcode` instead of returning a reason.
 *
 * Every opcode added to the switch converts what used to be the `default`
 * case's partial walk into an outright refusal, and that is a regression: a
 * body that compiled a prefix and interpreted the rest now compiles none of
 * it. Adding `not` alone cost 80 whole-body declines in the resolver, all of
 * them the measuring pass seeing SLOT_INT where a bool would later be.
 *
 * It also makes each arm's kill switch honest. With a hard refusal the "off"
 * side of an A/B declines the body, which is worse than the no-arm behaviour
 * it stands in for, and every ratio measured against it is inflated.
 *
 * The jump has to land in `default` rather than in a helper: the unarmed path
 * emits an unconditional branch, so the fall-through edge is gone and the walk
 * has to be told (`afterUncond`). A helper returning "I handled it" cannot say
 * that. */


/* Where the body proper starts. A defaulted parameter compiles to a thunk of
 * its own at the TOP of the chunk with the body hopped to over an
 * unconditional jump -- `fn matches(x: int, want: int = -1)` is exactly
 * `0000 OP_JUMP +4 -> 0007 / 0003 OP_INT -1 / 0006 OP_RETURN / 0007 body`.
 *
 * Walking those three dead instructions cost far more than emitting them: a
 * thunk ends in OP_RETURN like any other return, so mergeReturnKind recorded
 * the DEFAULT's kind and every real return of a different kind then clashed
 * with it and declined the whole function. `matches` above is INT-then-BOOL
 * and stopped at `OP_RETURN` for it; in a self-hosted compile the same clash
 * is what stopped window_clean, Lexer._push, Universe.intern, parser._type and
 * modsig.load.
 *
 * MEASURED, best of nine alternating runs under scripts/gpu_lock.sh, the two
 * sample sets not overlapping at all in either workload: `check --no-cache`
 * over four compiler files 2196ms -> 2106ms (4.3%), and over four others
 * 1469ms -> 1404ms (4.6%). A 3M-call probe on `matches` itself is 137ms ->
 * 29ms wall, ~13x on the loop once the ~20ms process floor is taken off both
 * sides -- the callee's body is too small for the OSR tier to rescue, which is
 * the shape that pays most. Where the loop is INSIDE the defaulted function,
 * OSR already had it and the same change is only 1.36x. The durable figure is
 * the decline count: 66 stops at OP_RETURN in one `check --no-cache` of
 * check/expr.jai, and zero after.
 *
 * SAFE because the skipped region is unreachable from the whole-function
 * entry, not merely unwalked:
 *   - The interpreter runs a thunk by ENTERING at its own offset
 *     (evalDefaultThunk, off fn->defaultOffsets), never by falling into it,
 *     and bindCallArgsSlow does all of that BEFORE the frame runs -- so by the
 *     time a compiled instruction executes the defaults are already in slots.
 *   - Every path into the whole-function form checks argc == arity first
 *     (callClosure, jaiCallValue1, run()'s tail call, and a compiled self-call,
 *     which declines outright unless argc == arity).
 *   - Nothing branches into the region: every offset in it is checked against
 *     offsetIsBranchTarget, whose operand table includes OP_PUSH_HANDLER and
 *     OP_PUSH_FINALLY, so a default expression holding a try edge fails the
 *     test and keeps the old behaviour. (It would be safe anyway -- the tier
 *     has no arm for OP_PUSH_HANDLER, so only the interpreter ever runs that
 *     thunk -- but the check does not have to know that.)
 *
 * Ruled out as a narrower fix: having mergeReturnKind ignore returns inside the
 * region. It would recover the return kind but still emit the thunks and still
 * charge them to the instruction budget, and it leaves the walk modelling code
 * the entry cannot reach. */
static int bodyEntryOffset(const ObjFunction *fn) {
    const Chunk *c = &fn->chunk;
    if (fn->defaultCount == 0 || fn->defaultOffsets == NULL) return 0;
    if (c->count < 3 || c->code[0] != OP_JUMP) return 0;
    int target = 3 + (int)jaiReadI16(c->code + 1);
    if (target <= 3 || target >= c->count) return 0;
    for (unsigned d = 0; d < fn->defaultCount; d++) {
        uint32_t at = fn->defaultOffsets[d];
        if (at < 3u || at >= (uint32_t)target) return 0;
    }
    for (int at = 3; at < target;) {
        if (offsetIsBranchTarget(c, (uint32_t)at)) return 0;
        int len = instructionLength(c, at);
        if (len <= 0) return 0;
        at += len;
    }
    return target;
}

static bool compileBody(Emit *e, ObjClosure *closure) {
    ObjFunction *fn = closure->fn;
    const uint8_t *code = fn->chunk.code;
    int count = fn->chunk.count;

    /* An inlined body is walked whole; the OSR window belongs to the caller. */
    int start = (!e->inlining && e->osr) ? (int)e->osrTop : bodyEntryOffset(fn);
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
        bool fellIn = !afterUncond;
        if (afterUncond) reconcileAfterUncond(e, (uint32_t)off);
        /* The one place the model is asked to agree with anything. Which of the
         * two answers is right depends on how the walk got here: along a
         * fall-through edge a disagreement means an arm moved the model by the
         * wrong amount, and there is nothing to do but decline; arriving
         * without one means this offset is reached only by a branch, and if
         * reconcileAfterUncond could not restate the model from that branch
         * then this is code the compiled body has no way in to. Stopping is
         * right there and declining would be a coverage loss for nothing: the
         * last thing emitted is unconditional, so control never falls off the
         * end, and a branch that does target a skipped offset declines when the
         * fixups are resolved. */
        if (!modelAgreesWithChunk(e, (uint32_t)off)) {
            if (fellIn) {
                e->whyNot = "the operand model disagrees with the bytecode";
                if (getenv("JAI_JIT_WHY")) {
                    fprintf(stderr,
                            "[jit] at %d: model depth %u, the bytecode says %d\n",
                            off, e->depth, e->chunkDepth[off]);
                }
                return false;
            }
            break;
        }
        uint8_t op = code[off];
        afterUncond = !jaiOpFallsThrough(op);
        /* Walked out of the way for the chain diagnostic; see Emit::chainSkip.
         * Never set in an ordinary compile. */
        for (unsigned ci = 0; ci < e->chainSkipCount; ci++) {
            if (e->chainSkip[ci] == (uint32_t)off) {
                e->curOffset = (uint32_t)off;
                e->lastOp = op;
                /* Settled first, as a branch join is. The ordinary unarmed path
                 * is only ever reached from an arm that declined BEFORE
                 * touching the model, whereas this one steps over an
                 * instruction whose arm may have left a value deferred or in
                 * the FP bank -- and the deopt record cannot describe those.
                 * Without this the diagnostic reported "a deferred value
                 * reached a guard" as the second link of every chain, which is
                 * an artefact of the skip and not a fact about the program. */
                fpSyncAll(e);
                settleAll(e);
                goto unarmedOpcode;
            }
        }
        /* Whose regions these are matters: inside an inline the offsets are the
         * callee's while every guard resumes at the CALLER's call site, so the
         * caller's answer is the one that stands. inlineGlobalCall/inlineMethod
         * refuse a callee with a table of its own. */
        if (!e->inlining) {
            e->inProtected = offsetIsProtected(fn, (uint32_t)off);
        }
        /* A stack proof is only good along the fall-through edge this walk is
         * following. offsetIsBranchTarget scans the whole chunk, so it catches
         * a back edge whose branch has not been emitted yet -- and it is only
         * asked while a proof is actually live, which is the two or three
         * instructions between `s[i]` and whatever consumes it, or the single
         * instruction between OP_NULL and OP_IS. */
        if (anyStackProof(e) &&
            offsetIsBranchTarget(&fn->chunk, (uint32_t)off)) {
            clearStackProofs(e);
        }
        /* A field-kind memo is good along the same one edge, and goes for the
         * same reason -- see forgetFieldKinds. `fn` is whichever body is being
         * walked, so an inlined one is measured against its own chunk. */
        if (e->knownCount != 0 &&
            offsetIsBranchTarget(&fn->chunk, (uint32_t)off)) {
            forgetFieldKinds(e);
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
            /* Inside a `try` nothing may cross an instruction boundary
             * unmaterialised: branchOnDeoptInstStart describes the model as of
             * the instruction's START, which means describing entries this
             * instruction has already popped -- and a popped entry that was
             * only ever a borrow of a local's register never wrote its own.
             * Settling here is the same conservative branch a non-whitelisted
             * opcode already takes, and it costs `mov`s that §5 prices at zero.
             */
            if (joinsHere || e->inProtected || !deferSurvives(op) ||
                e->deferCarryCount >= 64) {
                settleAll(e);
            } else {
                e->deferCarry[e->deferCarryCount++] = (uint32_t)off;
            }
        }
        /* Above the offset map for the same reason the deferred settle is:
         * branchTo empties the FP bank before it records a fixup, so an edge
         * arriving here holds every entry in its own X register -- and must
         * not run the fall-through's `fmov x, d` over a d register it never
         * wrote. `if flag { a[i] } else { a[0] }` is the shape: one arm ends
         * at the jump with its value in X, the other flows into the OP_ADD
         * and leaves it in the bank. Settled here, both edges agree. */
        if (e->fpLive != 0 && !e->inlining) {
            for (unsigned f = 0; f < e->fixupCount; f++) {
                if (e->fixups[f].targetOffset != (uint32_t)off) continue;
                fpSyncAll(e);
                break;
            }
        }
        /* Above the offset map on purpose: a back edge to `off` must land on
         * the loop head, not on the loads that were hoisted out of it. */
        emitHoistsAt(e, (uint32_t)off);
        e->offsetToInst[off]  = (int)e->count;
        e->offsetToDepth[off] = (int)stackSignature(e);
        e->lastOp = op;
        e->whySub[0] = '\0';
        /* A borrow ends here unless the instruction is one of the few it is
         * allowed to live across. The top of an instruction is the one place
         * the release is guaranteed to be on the executed path. */
        if (e->fpBorrow != 0 && (e->inProtected || !fpBorrowSurvives(op))) {
            fpReleaseAll(e);
        }
        e->curOffset = (uint32_t)off;
        e->instDepth = e->depth;
        e->instValueDepth = e->valueDepth;
        /* Every entry is in its own register right now, so a record taken of
         * this instruction's START stays readable however far the arm below
         * gets. An assertion, not the mechanism: the settles above are what
         * make it true inside a protected region, and branchOnDeoptInstStart
         * declines rather than describing a register nothing wrote. */
        e->instClean = (e->kPend == 0 && e->xBorrow == 0 && e->fpBorrow == 0);

        if (e->fpLive != 0) {
            /* A join already settled above, so anything still live here is on
             * a single edge. An inlined OP_RETURN is not a sync point either:
             * the only entry that outlives it is the result, and
             * inlineGlobalCall carries that one across in the bank. Everything
             * under it is discarded unread. */
            if (e->inlining && op == OP_RETURN) {
                /* handled below */
            } else if (!fpFastOp(op) || e->inProtected) {
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
            /* A float result stays in its d register: inlineGlobalCall moves it into the caller's bank with
             * one FP-to-FP move, where syncing here spent an `fmov x,d` and then made the caller's first float operator pay an `fmov d,x` to undo it. settleAll still runs -- it is X-side only (kPend/xBorrow), and those bits are never set on an entry the bank holds. */
            if (!(e->fpLive & (1u << (e->valueDepth - 1)))) {
                fpSyncOne(e, e->valueDepth - 1);
            }
            settleAll(e);   /* the caller reads the result out of valueXReg */
            break;
        }

        switch (op) {
        case OP_GET_LOCAL: {
            unsigned slot = jaiReadU16(code + off + 1);
            /* Both refusals used to be silent, so the census said only
             * "OP_GET_LOCAL" for two unrelated causes -- one a window the OSR
             * form does not cover, the other a slot whose kind is not known
             * yet. They want different fixes; they should not share a line. */
            if (!localInRange(e, slot)) {
                e->whyNot = "a local outside the compiled window";
                return false;
            }
            if (e->localKind[slot] == SLOT_OPAQUE) {
                e->whyNot = "a local of no known kind";
                return false;
            }
            if (slot == 0) e->usesSlot0 = true;
            if (!pushValue3(e, e->localKind[slot], e->localShape[slot],
                            e->localClass[slot],
                            localObserved(e, slot) ? e->observed[slot]
                                                   : e->localSeen[slot],
                            (int)slot)) {
                return false;
            }
            /* The seed of the index shape: this entry IS this local, offset
             * zero. See Emit::idxKnown. */
            if (e->localKind[slot] == SLOT_INT && slot <= UINT8_MAX) {
                unsigned at = e->valueDepth - 1;
                e->idxKnown |= 1u << at;
                e->idxBase[at] = (uint8_t)slot;
                e->idxOff[at]  = 0;
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
                e->whyNot = kindClash(e, slot);
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
            if (!localInRange(e, a) || !localInRange(e, b)) {
                return subWhy(e, "a fused add of a local the model does not "
                              "cover");
            }
            SlotKind ka2 = e->localKind[a];
            /* Named: this was 148 declines across four compiler files reading
             * only "OP_ADD_LOCALS", with nothing in them to act on. */
            if (ka2 != e->localKind[b]) {
                return subWhy(e, "a fused add of a %s and a %s",
                              slotKindName(ka2), slotKindName(e->localKind[b]));
            }
            if (ka2 == SLOT_OBJ && jitConcatLocals() && e->callsOut &&
                !e->inlining) {
                /* `out = out + piece` -- string building, and the fused form is
                 * the one real code emits. The stack `+` already had a concat
                 * arm; this opcode did not, so an OSR loop doing the commonest
                 * thing a lexer does declined WHOLE: the probe is 4,001,920
                 * interpreted instructions and the census showed 28 of these in
                 * parser.jai alone reading only "a fused add of two objects".
                 *
                 * The operands are slot numbers here and emitStringConcat wants
                 * stack entries, so they are pushed first. That is the cost the
                 * fusion existed to avoid, and it is nothing next to a call
                 * that allocates a string. Both operands are guarded inside
                 * emitStringConcat, so a sample that turns out wrong deopts at
                 * this instruction with both locals untouched. */
                Value sa = localObserved(e, a) ? e->observed[a]
                                               : e->localSeen[a];
                Value sb = localObserved(e, b) ? e->observed[b]
                                               : e->localSeen[b];
                if (IS_STRING(sa) && IS_STRING(sb)) {
                    if (a == 0 || b == 0) e->usesSlot0 = true;
                    if (!pushLocalAsValue(e, a)) return false;
                    if (!pushLocalAsValue(e, b)) return false;
                    if (!emitStringConcat(e, sa)) return false;
                    off += 5;
                    break;
                }
            }
            if (ka2 != SLOT_INT && ka2 != SLOT_FLOAT) {
                return subWhy(e, "a fused add of two %ss", slotKindName(ka2));
            }
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
                    e->whyNot = kindClash(e, slot);
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
            /* `text = text + piece` on strings. Out to jaiStringConcat behind a
             * pair of type guards -- see emitStringConcat. Before this arm the
             * refusal below gave up word_freq's entire `main` loop, which is
             * this one instruction plus the LCG arithmetic around it. */
            {
                Value csample;
                if (concatOperands(e, &csample)) {
                    /* The kind is adopted AFTER the call, not before: the
                     * descriptor's root fill reads every object-kinded local,
                     * and until the store below this slot still holds the old
                     * string -- claiming the new kind first would describe a
                     * slot the call has not written yet. */
                    if (!emitStringConcat(e, csample)) return false;
                    if (!adoptLocalKindSeen(e, slot, SLOT_OBJ, 0, NULL,
                                            csample)) {
                        e->whyNot = kindClash(e, slot);
                        return false;
                    }
                    unsigned rc;
                    if (!popValue(e, &rc, NULL)) return false;
                    localOut(e, slot, rc);
                    off += 3;
                    break;
                }
            }
            unsigned rb, ra;
            SlotKind kb, ka;
            if (!popValue(e, &rb, &kb)) return false;
            if (!popValue(e, &ra, &ka)) return false;
            if (ka != kb) return false;
            if (!adoptLocalKind(e, slot, ka, 0, NULL)) {
                e->whyNot = kindClash(e, slot);
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
                rd = ovfDest(e, rd);   /* localOut copies it home below */
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
                /* The interpreter stamps a dict's two nibbles as well, and
                 * does NOTHING for any other container -- "an unstamped
                 * container is simply unguarded". Both of those are arms.
                 *
                 * They became reachable the day the dict and set literals got
                 * arms of their own: before that the walk stopped AT the
                 * literal, so this opcode was never reached with a non-list on
                 * top. `var d: dict[str, int] = {}` in a hot body then
                 * declined the WHOLE function, which is strictly worse than
                 * the partial walk it replaced. */
                uint8_t built = e->stackObjType[e->depth - 1];
                if (built == (uint8_t)(OBJ_SET + 1) ||
                    built == (uint8_t)(OBJ_TUPLE + 1)) {
                    off += 2;
                    break;
                }
                if (built != (uint8_t)(OBJ_DICT + 1)) {
                    goto unarmedOpcode;
                }
                unsigned dr = valueXReg(e, e->valueDepth - 1);
                /* The prediction came from the build instruction just below,
                 * but a guard costs two instructions and does not depend on
                 * the emitter keeping them adjacent. */
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, dr,
                                   (unsigned)offsetof(Obj, type)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_DICT));
                branchOnDeoptInstStart(e, JAI_A64_NE);
                emitConst64(e, JIT_SCRATCH_A, (int64_t)((packed >> 4) & 0xFu));
                emit(e, jaiA64StrByte(JIT_SCRATCH_A, dr,
                                      (unsigned)offsetof(ObjDict, keyKind)));
                emitConst64(e, JIT_SCRATCH_A, (int64_t)(packed & 0xFu));
                emit(e, jaiA64StrByte(JIT_SCRATCH_A, dr,
                                      (unsigned)offsetof(ObjDict, valKind)));
                e->wroteHeap = true;
                off += 2;
                break;
            }
            unsigned r = valueXReg(e, e->valueDepth - 1);
            emitConst64(e, JIT_SCRATCH_A, (int64_t)(packed & 0xFu));
            emit(e, jaiA64StrByte(JIT_SCRATCH_A, r,
                                  (unsigned)offsetof(ObjList, elemKind)));
            /* And the storage, on the same terms jaiListSpecialise takes: an
             * empty list with nothing reserved, which is what a `[]` literal
             * is. Six instructions rather than a call, and no allocation --
             * that is the whole reason the interpreter's half refuses a
             * non-empty list too. Without this the two tiers build the same
             * literal at different widths and a pinned loop form is denied
             * entry for half the lists it meets; see jaiListSpecialise. */
            uint8_t kStg = listAltFor(
                (packed & 0xFu) == FIELD_KIND_INT   ? SLOT_INT
              : (packed & 0xFu) == FIELD_KIND_FLOAT ? SLOT_FLOAT
              : (packed & 0xFu) == FIELD_KIND_BOOL  ? SLOT_BOOL
                                                    : SLOT_OPAQUE);
            if (kStg != LIST_STORE_BOXED && jaiListUnboxOn()) {
                /* JIT_SCRATCH_A only: this arm has always used one scratch,
                 * and e->scratchRoom is what says how many the body actually
                 * reserved -- reaching for a second clobbered a live value
                 * register and miscompiled the self-hosted emitter. */
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, r,
                                   (unsigned)offsetof(ObjList, count)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, 0));
                int kA = (int)e->count;
                emit(e, jaiA64BCond(JAI_A64_NE, 0));
                emit(e, jaiA64LdrX(JIT_SCRATCH_A, r,
                                   (unsigned)offsetof(ObjList, items)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, 0));
                int kB = (int)e->count;
                emit(e, jaiA64BCond(JAI_A64_NE, 0));
                emitConst64(e, JIT_SCRATCH_A, (int64_t)kStg);
                emit(e, jaiA64StrByte(JIT_SCRATCH_A, r,
                                      (unsigned)offsetof(ObjList, stg)));
                e->code[kA] = jaiA64BCond(JAI_A64_NE,
                                          (int32_t)((int)e->count - kA));
                e->code[kB] = jaiA64BCond(JAI_A64_NE,
                                          (int32_t)((int)e->count - kB));
            }
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

        case OP_SUB_INT_CONST: {
            unsigned slot = jaiReadU16(code + off + 1);
            int16_t  imm  = jaiReadI16(code + off + 3);
            if (!localInRange(e, slot)) return false;
            if (e->localKind[slot] != SLOT_INT) return false;
            if (slot == 0) e->usesSlot0 = true;
            if (!pushValue(e, SLOT_INT, 0, NULL)) return false;
            {
                unsigned dst = pushReg(e) - 1;
                unsigned cur = localIn(e, slot, JIT_SCRATCH_C);
                if (imm >= 0 && imm <= 4095) {
                    emit(e, jaiA64SubsXImm(dst, cur, (unsigned)imm));
                } else if (imm < 0 && imm >= -4095) {
                    emit(e, jaiA64AddsXImm(dst, cur, (unsigned)(-(int)imm)));
                } else {
                    emitConst64(e, JIT_SCRATCH_A, imm);
                    emit(e, jaiA64SubsXReg(dst, cur, JIT_SCRATCH_A));
                }
            }
            branchOnOverflow(e, 1u, JAI_A64_VS);
            off += 5;
            break;
        }

        case OP_MUL_INT_CONST: {
            unsigned slot = jaiReadU16(code + off + 1);
            int16_t  imm  = jaiReadI16(code + off + 3);
            if (!localInRange(e, slot)) return false;
            if (e->localKind[slot] != SLOT_INT) return false;
            if (slot == 0) e->usesSlot0 = true;
            if (!pushValue(e, SLOT_INT, 0, NULL)) return false;
            unsigned dst = pushReg(e) - 1;
            unsigned cur = localIn(e, slot, JIT_SCRATCH_C);
            unsigned rt = ovfDest(e, dst);
            /* MUL has no immediate form on this encoder, so the constant goes
             * into JIT_SCRATCH_D -- left free by `cur` and `rt` above, so it
             * cannot collide with either even when ovfDest hands back
             * JIT_SCRATCH_B inside a `try`. Same overflow test as plain
             * OP_MUL: the product overflows exactly when smulh's high half is
             * not the low half's sign bit replicated, and it shares that
             * arm's overflow-stub slot (2, the `*` message) since it is the
             * same operator. */
            emitConst64(e, JIT_SCRATCH_D, imm);
            emit(e, jaiA64SmulhX(JIT_SCRATCH_A, cur, JIT_SCRATCH_D));
            emit(e, jaiA64MulX(rt, cur, JIT_SCRATCH_D));
            emit(e, jaiA64SubsXAsr(31, JIT_SCRATCH_A, rt, 63));
            branchOnOverflow(e, 2u, JAI_A64_NE);
            if (rt != dst) emit(e, jaiA64MovX(dst, rt));
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
                    e->whyNot = kindClash(e, slot);
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
                e->whyNot = kindClash(e, slot);
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
                rd = ovfDest(e, rd);   /* localOut copies it home below */
                emit(e, jaiA64SmulhX(JIT_SCRATCH_A, ra, rb));
                emit(e, jaiA64MulX(rd, ra, rb));
                emit(e, jaiA64SubsXAsr(31, JIT_SCRATCH_A, rd, 63));
                branchOnOverflow(e, 2u, JAI_A64_NE);
            } else {
                return subWhy(e, "arithmetic on a %s", slotKindName(ka));
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
                    e->whyNot = kindClash(e, slot);
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
                e->whyNot = kindClash(e, slot);
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
                rd = ovfDest(e, rd);   /* localOut copies it home below */
                emit(e, jaiA64SubsXReg(rd, ra, rb));
                branchOnOverflow(e, 1u, JAI_A64_VS);
            } else {
                return subWhy(e, "arithmetic on a %s", slotKindName(ka));
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
                e->whyNot = kindClash(e, slot);
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
                unsigned dst = ovfDest(e, localDest(e, slot));
                /* Step is an i8, always fits imm12, so the constant never needs a register of its own. `subs` for a
                 * negative step rather than a negated `adds`: both set V for the operation actually performed, which is what the overflow guard below reads. */
                if (imm >= 0) {
                    emit(e, jaiA64AddsXImm(dst, cur, (unsigned)imm));
                } else {
                    emit(e, jaiA64SubsXImm(dst, cur, (unsigned)(-(int)imm)));
                }
                /* The guard is taken before the home is written, not after: it
                 * resumes at this instruction inside a `try` (ovfDest), and
                 * neither fpSyncAll nor a b.cond disturbs V or `dst`. */
                branchOnOverflow(e, 0u, JAI_A64_VS);
                localOut(e, slot, dst);
            }
            off += 4;
            break;
        }

        case OP_EQ: case OP_NE:
        case OP_LT: case OP_LE: case OP_GT: case OP_GE: {
            /* Operands are read without popping them off the model: a NaN sends this back to the interpreter,
             * whose stack still has them. Popping first once left the model two entries short, so the re-run comparison silently read whatever was underneath -- a bug that surfaced one line later instead of not at all. */
            if (e->depth < 2) return subWhy(e, "a compare with nothing under it");
            SlotKind ka = e->stack[e->depth - 2], kb = e->stack[e->depth - 1];
            /* `node == null` puts an instance beside a maybe-instance. Both
             * are a pointer or zero in a register, so the compare is the same
             * one; treat the pair as maybe-instance. */
            if (ka != kb) {
                bool mixable =
                    (ka == SLOT_INST && kb == SLOT_MAYBE_INST) ||
                    (ka == SLOT_MAYBE_INST && kb == SLOT_INST) ||
                    nullLiteralPair(e, op, ka, kb);
                /* Named: 92 declines across four compiler files said only
                 * "OP_EQ", and the two operands' kinds are the entire question
                 * at this arm. */
                if (!mixable) {
                    return subWhy(e, "a compare of a %s with a %s",
                                  slotKindName(ka), slotKindName(kb));
                }
                ka = kb = SLOT_MAYBE_INST;
            }
            if (!holdsRegister(ka)) {
                return subWhy(e, "a compare of two %ss, which hold no register",
                              slotKindName(ka));
            }
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
                 * FP-resident entry out if it is ever taken.
                 *
                 * Released BEFORE the operands are read, exactly as the fused
                 * arm at OP_JUMP_IF_CMP_FALSE does it and for the same reason:
                 * this arm is on the borrow whitelist, so an operand can still
                 * be held in a local's own d register, and `nanToDeopt`
                 * records -- which `deoptRecordAt` asserts nothing may be
                 * borrowing at. Only the fused arm had this, so a body whose
                 * float compare did NOT fuse into a branch declined with "a
                 * float borrow reached OP_GT's guard"; `let a = ys[i] > py`
                 * followed by a use of `a` is exactly that shape. Costs one
                 * fmov when a borrow is live, against declining the loop. */
                fpReleaseAll(e);
                unsigned db = fpOperand(e, e->valueDepth - 1);
                unsigned da = fpOperand(e, e->valueDepth - 2);
                /* nanToDeopt records, so nothing may still be deferred. A
                 * float local can be borrowed out of an X register in OSR
                 * mode; settling costs a mov on a path that is about to
                 * compare NaNs, and buys the compile. */
                settleAll(e);
                emit(e, jaiA64FcmpD(da, db));
                nanToDeopt(e);
            /* Two bools compare exactly as two ints do. Each is 0 or 1 in
             * its register (see SLOT_BOOL), so `cmp` answers `==` and `!=`
             * directly, and the ordering conditions come out right as well --
             * false sorts below true, which is what the interpreter says.
             *
             * Missing until now, and a whole-function refusal rather than a
             * slow path: `let a = ys[i - 1] > py` / `let b = ys[i] > py` /
             * `if a != b` is how a point-in-polygon crossing test is written,
             * and jaicv's `nearest_gap_squared` carries a comment saying it
             * had to be spelt as two nested `if`s and spelt TWICE because of
             * this.
             *
             * MEASURED. That loop over a 100000-point polygon, forty passes:
             * 215 ms declined against 36 ms compiled, best of five alternating
             * runs with no overlap -- 5.8x, and a decline-to-compile
             * transition rather than a micro-optimisation. The flag form is
             * now as fast as the nested one it was rewritten into (7.23 ms
             * against 7.67 in one binary), so the workaround can go.
             *
             * RULED OUT: the self-hosted compiler does not move. `check
             * --no-cache` over four compiler files is 2015 ms against 2020
             * median of seven, inside the spread, even though the same census
             * counts 66 sites there -- they clear this gate and stop at the
             * next, which is the pattern every arm added today has shown. */
            } else if (ka == SLOT_INT || ka == SLOT_MAYBE_INST ||
                       ka == SLOT_BOOL) {
                unsigned ra = xHeldIn(e, e->valueDepth - 2);
                if (foldCmp) {
                    e->kPend &= ~(1u << (e->valueDepth - 1));
                    emitCmpImm(e, ra, kcmp);
                } else {
                    emit(e, jaiA64SubsXReg(31, ra,
                                           xHeldIn(e, e->valueDepth - 1)));
                }
            } else if ((op == OP_EQ || op == OP_NE) && ka == SLOT_OBJ &&
                       (e->stackUnit[e->depth - 2] ||
                        e->stackUnit[e->depth - 1])) {
                /* Two payload-less enum values are equal exactly when they are the
                 * same object: a variant with no payload has no state to tell
                 * instances apart, so every mention of it yields the one
                 * shared EnumVariant::unit, and enumValsEqual reduces to
                 * type-and-tag at zero payload. One side must be the folded
                 * constant (stackUnit, a proof), never merely a value observed
                 * to be one. The other may be anything on the heap: for two
                 * VAL_OBJs of different Obj types jaiValuesEqual returns false
                 * without consulting an `__eq__`, which is the same answer the
                 * pointer compare gives.
                 *
                 * Needed for the `Enum.Variant` fold to pay at all -- with the
                 * fold alone, OP_EQ and this opcode simply became the new
                 * refusal and the same functions still declined. */
                unsigned rb = xHeldIn(e, e->valueDepth - 1);
                unsigned ra = xHeldIn(e, e->valueDepth - 2);
                emit(e, jaiA64SubsXReg(31, ra, rb));
            } else if ((op == OP_EQ || op == OP_NE) && ka == SLOT_OBJ &&
                       IS_STRING(e->stackSeen[e->depth - 2]) &&
                       IS_STRING(e->stackSeen[e->depth - 1]) &&
                       !preferLeafEquality(e, e->depth - 2, e->depth - 1)) {
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
            } else if (isOrdering(op) && ka == SLOT_OBJ &&
                       oneBytePair(e, e->depth - 2, e->depth - 1)) {
                /* Ordering only. `==` stays on the identity arm above, which
                 * is cheaper and holds for strings of any length. See
                 * emitOneByteString. This path guards, so settle first. */
                settleAll(e);
                unsigned rb = valueXReg(e, e->valueDepth - 1);
                unsigned ra = valueXReg(e, e->valueDepth - 2);
                emitOneByteString(e, e->depth - 2, ra, JIT_SCRATCH_C);
                emitOneByteString(e, e->depth - 1, rb, JIT_SCRATCH_D);
                emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_C, JIT_SCRATCH_D));
            /* Two strings of any length, all six operators, through the leaf
             * call. LAST of the string arms on purpose: the identity arm above
             * answers `==` with no call at all, and the one-byte arm answers an
             * ordering with two byte loads, so both are strictly better where
             * they apply and this is only what they leave behind.
             *
             * Not inside an inlined body. Its entries are in x0..x8
             * (inlineOwnBank) and a call would run over them; declining here
             * leaves that case exactly as it was. */
            } else if (jitStrCmpOn() && ka == SLOT_OBJ && !e->inlining &&
                       stringOperand(e, e->depth - 2) &&
                       stringOperand(e, e->depth - 1)) {
                emitStringOrder(e);
            } else if ((op == OP_EQ || op == OP_NE) && ka == SLOT_OBJ &&
                       jitObjEquality() && e->callsOut && !e->inlining) {
                /* Two heap objects and nothing above could name them. Rather
                 * than decline the whole body for one comparison, call the
                 * interpreter's equality and carry on.
                 *
                 * `self._kind() == kind` on two TokenKinds is the shape, and it
                 * is the SHORTEST CHAIN ON A HOT BODY in the self-hosted front
                 * end: `_check` is one refusal away from compiling and carries
                 * 3.2% of the interpreted work in lexer.jai. The enum arm
                 * above cannot help it -- that one needs one side to be a
                 * FOLDED constant, and here both are values.
                 *
                 * Leaves NZCV from a compare of the result against one, which
                 * is what the shared tail below expects, so this arm ends the
                 * same way every other arm in the chain does. */
                settleAll(e);
                if (!emitDescriptor(e, NULL_VAL, e->depth - 2, 2,
                                    (void *)&jitValuesEqual)) {
                    return false;
                }
                emit(e, jaiA64LdrByte(JIT_SCRATCH_A, 31,
                                      e->descOffset +
                                          (unsigned)offsetof(JitCallDesc,
                                                             result) + 8));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, 1));
                e->wroteHeap = true;
            } else {                /* Named: this closed the compare chain with a bare refusal, so
                 * a census showed 80 declines reading only "OP_NE", with
                 * nothing in them to act on. */
                return subWhy(e, "a comparison of %s with %s",
                              slotKindName(e->stack[e->depth - 2]),
                              slotKindName(e->stack[e->depth - 1]));
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

        /* `x is null` and `x is not null`, which is how every optional in
         * this language is tested. `valueIsTest` against a null target is
         * plain identity, so the whole question is whether the subject's tag
         * is VAL_NULL -- and the tier already represents a null instance as a
         * zero in the register, so for the kind that actually occurs it is a
         * compare and a `cset`.
         *
         * Worth an arm on its own: without one the operator fell to the
         * unarmed-opcode deopt a few instructions into the body, and a body
         * that compiles and then immediately bails is SLOWER than one that was
         * never compiled -- measured at 212 ms against the interpreter's
         * 162 ms on a guard doing nothing else. It was the only construct
         * found that made the tier a net loss. */
        case OP_IS:
        case OP_IS_NOT: {
            if (e->depth < 2) return false;
            /* Only against a literal null -- but reading that off the kind
             * alone was wrong, and the arm below sat unreachable for every
             * `is null` anyone writes. SLOT_NULL is what a `-> void` call
             * leaves behind; the `null` in source is OP_NULL, which pushes
             * SLOT_MAYBE_INST so that `var x: Box? = null` does not give its
             * local two kinds. So the whole construct took the decline: 71
             * distinct functions on a self-hosted compile, and `is null`
             * outnumbers `is SomeClass` in lib/jaithon 661 to 12. stackNullLit
             * is the half of the question the kind cannot carry, and it is a
             * proof rather than a guess, so it is dropped at any offset a
             * branch can reach (see clearStackProofs).
             *
             * Measured, `for x in xs { if x is null { t += 1 } }` over a 2M
             * list[Node?] with no null in it, twenty passes, whole process and
             * best of three under scripts/gpu_lock.sh: 635 ms declined against
             * 223 ms compiled, 2.85x with ~150 ms of list building inside both
             * figures. The self-hosted compile it was expected to move did NOT
             * move (4932 ms against 4945 ms, a wash): the 71 functions mostly
             * stop at OP_GET_FIELD one instruction later, so this only clears
             * the first of two gates.
             *
             * `x is SomeClass` is still declined. It is a type test through
             * valueMatchesType, nothing here would be right for it, and the
             * unarmed deopt it used to take left the operand model a slot
             * deeper than the bytecode -- a body that compiles only to bail at
             * the same instruction every iteration is slower than one never
             * compiled at all. */
            if (e->stack[e->depth - 1] != SLOT_NULL &&
                !(e->stack[e->depth - 1] == SLOT_MAYBE_INST &&
                  e->stackNullLit[e->depth - 1])) {
                e->whyNot = "an `is` against something other than null";
                return false;
            }
            SlotKind sk = e->stack[e->depth - 2];
            bool wantNull = (op == OP_IS);
            unsigned dropNull, dropSubject;

            if (sk == SLOT_MAYBE_INST) {
                unsigned rs = xHeldIn(e, e->valueDepth - 2);
                emit(e, jaiA64SubsXImm(31, rs, 0));
                if (!popValueRaw(e, &dropNull, NULL)) return false;
                if (!popValueRaw(e, &dropSubject, NULL)) return false;
                if (!pushValue(e, SLOT_BOOL, 0, NULL)) return false;
                emit(e, jaiA64CsetX(pushReg(e) - 1,
                                    wantNull ? JAI_A64_EQ : JAI_A64_NE));
            } else if (sk == SLOT_INT || sk == SLOT_FLOAT || sk == SLOT_BOOL ||
                       sk == SLOT_INST || sk == SLOT_LIST || sk == SLOT_OBJ) {
                /* None of these kinds can hold a null, so the answer is known
                 * here and the compare never runs. */
                if (!popValueRaw(e, &dropNull, NULL)) return false;
                if (!popValueRaw(e, &dropSubject, NULL)) return false;
                if (!pushValue(e, SLOT_BOOL, 0, NULL)) return false;
                emit(e, jaiA64MovzX(pushReg(e) - 1, wantNull ? 0u : 1u, 0));
            } else if (sk == SLOT_NULL) {
                if (!popValueRaw(e, &dropNull, NULL)) return false;
                if (!popValueRaw(e, &dropSubject, NULL)) return false;
                if (!pushValue(e, SLOT_BOOL, 0, NULL)) return false;
                emit(e, jaiA64MovzX(pushReg(e) - 1, wantNull ? 1u : 0u, 0));
            } else {
                e->whyNot = "an `is null` on a kind with no null representation";
                return false;
            }
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
            if (e->depth == 0 || e->stack[e->depth - 1] != SLOT_BOOL) {
                /* Named because the bare refusal was unreadable in a census:
                 * `_is_ident_cont` -- `c.is_alnum() or c == "_"`, called per
                 * character by the lexer -- reported only "OP_JUMP_IF_TRUE_KEEP"
                 * while three OSR loops retried it 80 times each waiting for it
                 * to compile. */
                return subWhy(e, "a branch on a %s, not a bool",
                              e->depth > 0
                                  ? slotKindName(e->stack[e->depth - 1])
                                  : "empty stack");
            }
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
            if (e->depth < 2) return subWhy(e, "a compare with nothing under it");
            SlotKind ka = e->stack[e->depth - 2], kb = e->stack[e->depth - 1];
            if (ka != kb) {
                bool mixable =
                    (ka == SLOT_INST && kb == SLOT_MAYBE_INST) ||
                    (ka == SLOT_MAYBE_INST && kb == SLOT_INST) ||
                    nullLiteralPair(e, cmp, ka, kb);
                /* Named for the same reason as the unfused twin above: the two
                 * operands' kinds are the entire question at this arm. */
                if (!mixable) {
                    return subWhy(e, "a compare of a %s with a %s",
                                  slotKindName(ka), slotKindName(kb));
                }
                ka = kb = SLOT_MAYBE_INST;
            }
            if (!holdsRegister(ka)) {
                return subWhy(e, "a compare of two %ss, which hold no register",
                              slotKindName(ka));
            }
            /* See the OP_LT..OP_GE arm: operands are read through xHeldIn, a
             * literal right-hand side becomes the compare's own immediate,
             * and the paths that guard settle first. */
            int64_t kcmp2 = 0;
            bool foldCmp2 = ka == SLOT_INT &&
                            pendingImm12(e, e->valueDepth - 1, &kcmp2);
            unsigned cond;
            if (!negatedCondition(cmp, &cond)) {
                return subWhy(e, "a fused compare whose condition has no "
                              "negation");
            }
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
                /* Released BEFORE the operands are read, so `fpOperand`
                 * answers out of the bank: this arm is on the borrow
                 * whitelist, so an operand can still be held in a local's own
                 * d register here, and nanToDeopt records -- which
                 * deoptRecordAt asserts nothing may be borrowing at. Costs one
                 * fmov when a borrow is live, against declining the entire
                 * loop, which is what `for v in xs { if v != 0.0 ... }` used
                 * to do. Sound at this point because nothing has branched yet.
                 */
                fpReleaseAll(e);
                unsigned db = fpOperand(e, e->valueDepth - 1);
                unsigned da = fpOperand(e, e->valueDepth - 2);
                settleAll(e);          /* nanToDeopt records */
                emit(e, jaiA64FcmpD(da, db));
                nanToDeopt(e);
            /* Bools, for the reason given at the unfused arm. The fused
             * form is the one an `if a != b` actually emits, so without it
             * here the unfused arm above would almost never be reached. */
            } else if (ka == SLOT_INT || ka == SLOT_MAYBE_INST ||
                       ka == SLOT_BOOL) {
                unsigned ra = xHeldIn(e, e->valueDepth - 2);
                if (foldCmp2) {
                    e->kPend &= ~(1u << (e->valueDepth - 1));
                    emitCmpImm(e, ra, kcmp2);
                } else {
                    emit(e, jaiA64SubsXReg(31, ra,
                                           xHeldIn(e, e->valueDepth - 1)));
                }
            } else if ((cmp == OP_EQ || cmp == OP_NE) && ka == SLOT_OBJ &&
                       (e->stackUnit[e->depth - 2] ||
                        e->stackUnit[e->depth - 1])) {
                /* Two payload-less enum values are equal exactly when they are the
                 * same object: a variant with no payload has no state to tell
                 * instances apart, so every mention of it yields the one
                 * shared EnumVariant::unit, and enumValsEqual reduces to
                 * type-and-tag at zero payload. One side must be the folded
                 * constant (stackUnit, a proof), never merely a value observed
                 * to be one. The other may be anything on the heap: for two
                 * VAL_OBJs of different Obj types jaiValuesEqual returns false
                 * without consulting an `__eq__`, which is the same answer the
                 * pointer compare gives.
                 *
                 * Needed for the `Enum.Variant` fold to pay at all -- with the
                 * fold alone, OP_EQ and OP_JUMP_IF_CMP_FALSE simply became the new
                 * refusal and the same functions still declined. */
                unsigned rb = xHeldIn(e, e->valueDepth - 1);
                unsigned ra = xHeldIn(e, e->valueDepth - 2);
                emit(e, jaiA64SubsXReg(31, ra, rb));
            } else if ((cmp == OP_EQ || cmp == OP_NE) && ka == SLOT_OBJ &&
                       IS_STRING(e->stackSeen[e->depth - 2]) &&
                       IS_STRING(e->stackSeen[e->depth - 1]) &&
                       !preferLeafEquality(e, e->depth - 2, e->depth - 1)) {
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
            } else if (isOrdering(cmp) && ka == SLOT_OBJ &&
                       oneBytePair(e, e->depth - 2, e->depth - 1)) {
                /* See the same arm in OP_LT..OP_GE. `if c < "0" { break }`
                 * fuses its compare into the branch and would otherwise have
                 * declined here after the unfused twin already compiled. */
                settleAll(e);
                unsigned rb = valueXReg(e, e->valueDepth - 1);
                unsigned ra = valueXReg(e, e->valueDepth - 2);
                emitOneByteString(e, e->depth - 2, ra, JIT_SCRATCH_C);
                emitOneByteString(e, e->depth - 1, rb, JIT_SCRATCH_D);
                emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_C, JIT_SCRATCH_D));
            /* See the same arm in OP_LT..OP_GE. This is the one the probe
             * hits: `if a < b { .. }` fuses its compare into the branch, and
             * without it here the unfused twin compiles and the shape anyone
             * actually writes still declines. */
            } else if (jitStrCmpOn() && ka == SLOT_OBJ && !e->inlining &&
                       stringOperand(e, e->depth - 2) &&
                       stringOperand(e, e->depth - 1)) {
                emitStringOrder(e);
            } else if ((cmp == OP_EQ || cmp == OP_NE) && ka == SLOT_OBJ &&
                       jitObjEquality() && e->callsOut && !e->inlining) {
                /* Two heap objects and nothing above could name them. Rather
                 * than decline the whole body for one comparison, call the
                 * interpreter's equality and carry on.
                 *
                 * `self._kind() == kind` on two TokenKinds is the shape, and it
                 * is the SHORTEST CHAIN ON A HOT BODY in the self-hosted front
                 * end: `_check` is one refusal away from compiling and carries
                 * 3.2% of the interpreted work in lexer.jai. The enum arm
                 * above cannot help it -- that one needs one side to be a
                 * FOLDED constant, and here both are values.
                 *
                 * Leaves NZCV from a compare of the result against one, which
                 * is what the shared tail below expects, so this arm ends the
                 * same way every other arm in the chain does. */
                settleAll(e);
                if (!emitDescriptor(e, NULL_VAL, e->depth - 2, 2,
                                    (void *)&jitValuesEqual)) {
                    return false;
                }
                emit(e, jaiA64LdrByte(JIT_SCRATCH_A, 31,
                                      e->descOffset +
                                          (unsigned)offsetof(JitCallDesc,
                                                             result) + 8));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, 1));
                e->wroteHeap = true;
            } else {                /* Named: this closed the compare chain with a bare refusal, so
                 * a census showed 80 declines reading only "OP_NE", with
                 * nothing in them to act on. */
                return subWhy(e, "a comparison of %s with %s",
                              slotKindName(e->stack[e->depth - 2]),
                              slotKindName(e->stack[e->depth - 1]));
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
            /* The kind says "may be null"; this says "IS null", which is what
             * the `is` arm needs and cannot recover from the kind. */
            e->stackNullLit[e->depth - 1] = true;
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
                /* The lookahead, not just the encodability: a constant with no float consumer ahead of it is
                 * synced back out to X at the next ordinary opcode, so the bank form costs `fmov d,#imm` plus that `fmov x,d` where one `movz` would have done. Every value FMOV's imm8 can name has its whole payload in the top 16 bits, so emitConst64 is exactly one instruction for all of them. spectral's `1.0 / float(..)` is the shape: OP_GET_GLOBAL stands between the constant and its divide. */
                if (!e->fpOff && jaiA64FpImm8(d, &imm8) &&
                    fpWorthLoading(e, code, off + 4, stop)) {
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
                return subWhy(e, "a constant of a kind the tier cannot hold");
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
                    e->stackAscii[e->depth - 1] = false;
                    e->stackNullLit[e->depth - 1] = false;
                    e->stackUnit[e->depth - 1]  = false;
                    e->stackObjType[e->depth - 1] = 0;
                    e->stackElem[e->depth - 1] = NULL_VAL;
                } else if (k != SLOT_FLOAT) {
                    e->whyNot = "a type guard the kinds cannot settle";
                    return false;
                }
            } else if ((strcmp(tn, "int") == 0 && k == SLOT_INT) ||
                       (strcmp(tn, "bool") == 0 && k == SLOT_BOOL)) {
            } else if (jitAnyGuard() && strcmp(tn, "list") == 0 &&
                       k == SLOT_LIST) {
                /* SLOT_LIST is a guarded fact, not a hope: every arm that puts
                 * one on the stack has already proved OBJ_LIST, because VAL_OBJ
                 * covers every heap object and the shared return path says so
                 * in as many words. `jaiTypeNameStatic` calls an ObjList "list"
                 * (value.c), which is the name this guard compares against, so
                 * there is nothing left to check.
                 *
                 * Same exposure as the `float`/`int`/`bool` cases above and no
                 * more: all four reason from the type NAME, and a module global
                 * shadowing one of those names with a class would change what
                 * the interpreter does. That is the arm's existing contract. */
            } else if (jitAnyGuard() && k == SLOT_INST &&
                       e->stackClass[e->depth - 1] != NULL &&
                       e->stackClass[e->depth - 1]->name != NULL &&
                       strcmp(e->stackClass[e->depth - 1]->name->chars, tn) == 0) {
                /* The pinned class IS the one the boundary names. Like
                 * SLOT_LIST this is a guarded fact -- every arm that puts a
                 * SLOT_INST on the stack with a class pinned has emitted the
                 * shapeId check that proves it -- so the guard has nothing left
                 * to do.
                 *
                 * An exact name match is sufficient, not necessary:
                 * jaiValueMatchesType also accepts a subclass and a trait
                 * implementer, so anything this does not recognise still
                 * declines rather than being waved through. The names that
                 * turn up are `Tensor`, `Mat` and `NDArray` -- the ML packages
                 * annotate their boundaries, so one of these sat in the middle
                 * of a hot body and declined all of it. */
            } else if (jitAnyGuard() && strcmp(tn, "any") == 0) {
                /* `any` is satisfied by every value, so this guard is a no-op
                 * for every kind -- not a narrowing the tier is guessing at.
                 * jaiValueMatchesType returns true for the name "any" before
                 * looking at the subject at all (vm.c), which is what makes
                 * emitting nothing here the same thing the interpreter does.
                 *
                 * It is common enough to matter because `dict[str, any]` is how
                 * this compiler carries AST records: a parameter or return
                 * declared `any` put one of these in the middle of a body and
                 * declined all of it. */
            } else {
                return subWhy(e, "a `%s` guard on a %s", tn, slotKindName(k));
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
            /* jitFormat always builds a string, and there is no Value to carry
             * as a sample, so the expectation is recorded instead: without it
             * `f"{a}-{b}".len()` declined the loop around it at the very next
             * instruction ("an invoke on an object with nothing to look at"). */
            e->stackObjType[e->depth - 1] = (uint8_t)(OBJ_STRING + 1);
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
            unsigned rt = ovfDest(e, rd);
            /* The product overflows exactly when the high half is not the low
             * half's sign bit replicated, so smulh and one shifted compare
             * decide it. mul must come after smulh reads its inputs, since rd
             * may be one of them -- which is also why `rt` differs inside a
             * `try`: rd IS the first operand's register, and the guard resumes
             * at an instruction whose operands the interpreter still needs. */
            emit(e, jaiA64SmulhX(JIT_SCRATCH_A, ra, rb));
            emit(e, jaiA64MulX(rt, ra, rb));
            emit(e, jaiA64SubsXAsr(31, JIT_SCRATCH_A, rt, 63));
            branchOnOverflow(e, 2u, JAI_A64_NE);
            if (rt != rd) emit(e, jaiA64MovX(rd, rt));
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
            /* `head + "/" + right`: two strings, out to jaiStringConcat, the
             * unfused half of the ADD_BIND arm above. Measured on a self-hosted
             * `check`, where OP_ADD is a few percent of the interpreted work:
             * no change (775ms -> 767ms, noise). The bodies holding it decline
             * for other reasons anyway, so this arm earns its place on the
             * ADD_BIND shape and not on the compiler. */
            if (op == OP_ADD) {
                Value csample;
                if (concatOperands(e, &csample)) {
                    if (!emitStringConcat(e, csample)) return false;
                    off += 1;
                    break;
                }
            }
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

            /* Carried through the arithmetic: see Emit::idxKnown. Only the
             * folded-literal form, which is exactly what `xs[j - 1]` and
             * `xs[j + 1]` compile to, and only while the offset stays small --
             * the loop head's guard has to add it to a count without
             * overflowing, and a bound keeps that argument short. The overflow
             * branch below deoptimises, so a shape that survives to the
             * subscript describes arithmetic that actually happened. */
            bool    idxCarry     = false;
            uint8_t idxCarryBase = 0;
            int32_t idxCarryOff  = 0;
            if (foldK && e->valueDepth >= 2 &&
                (e->idxKnown & (1u << (e->valueDepth - 2))) != 0 &&
                kimm >= -4096 && kimm <= 4096) {
                unsigned at = e->valueDepth - 2;
                int64_t sum = (int64_t)e->idxOff[at] +
                              (op == OP_SUB ? -kimm : kimm);
                if (sum >= -4096 && sum <= 4096) {
                    idxCarry     = true;
                    idxCarryBase = e->idxBase[at];
                    idxCarryOff  = (int32_t)sum;
                }
            }

            if (!popValue(e, &rb, &kb)) return false;
            if (!popValue(e, &ra, &ka)) return false;
            if (ka != kb) return false;   /* no implicit widening here */

            /* Integer division is not here: it has a zero-divisor error and a
             * truncation rule of its own, and getting either wrong would be a
             * wrong answer rather than a decline. */
            if (ka != SLOT_INT || op == OP_DIV) return false;
            if (!pushValue(e, SLOT_INT, 0, NULL)) return false;
            if (idxCarry) {
                unsigned at = e->valueDepth - 1;
                e->idxKnown |= 1u << at;
                e->idxBase[at] = idxCarryBase;
                e->idxOff[at]  = idxCarryOff;
            }
            unsigned rd = pushReg(e) - 1;
            /* rd is the first operand's own register (two off, one on), so
             * inside a `try` the sum computes elsewhere: the overflow guard
             * resumes at this instruction and the interpreter reads the
             * operands back off its stack. See ovfDest. */
            unsigned rt = ovfDest(e, rd);
            if (foldK) emitAddSubImm(e, rt, ra, kimm, op == OP_SUB);
            else emit(e, op == OP_ADD ? jaiA64AddsX(rt, ra, rb)
                                      : jaiA64SubsXReg(rt, ra, rb));
            branchOnOverflow(e, op == OP_ADD ? 0u : 1u, JAI_A64_VS);
            if (rt != rd) emit(e, jaiA64MovX(rd, rt));
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
            Value kLocalSeen = localObserved(e, slot) ? e->observed[slot]
                                                     : e->localSeen[slot];
            bool kLocalInterned = IS_STRING(kLocalSeen) &&
                                  JAI_STR_INTERNED(AS_STRING(kLocalSeen));
            if ((cmp == OP_EQ || cmp == OP_NE) &&
                e->localKind[slot] == SLOT_OBJ && IS_STRING(k) &&
                JAI_STR_INTERNED(AS_STRING(k)) &&
                IS_STRING(kLocalSeen) &&
                !(jitStrCmpEqOn() && jitStrCmpOn() && !e->inlining &&
                  !kLocalInterned)) {
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

            /* `if word < "middle"`, and `==` against a constant the arm above
             * declined because it is not interned. Same leaf call as the arm
             * at OP_JUMP_IF_CMP_FALSE, after the peephole folded the local and
             * the constant into one instruction -- which is what every parser
             * and every dispatch-on-a-name in this language actually emits, so
             * without this the unfused twin compiles and the shape anyone
             * types still declines.
             *
             * The LOCAL is the left operand and the constant the right; the
             * interpreter's own arm spells that out, and reversing it would be
             * a silent wrong answer for four of the six operators rather than
             * a crash. The constant needs no guard: it is a string the emitter
             * is holding, so only the local's Obj.type is in question. */
            if (jitStrCmpOn() && !e->inlining && IS_STRING(k) &&
                e->localKind[slot] == SLOT_OBJ &&
                IS_STRING(localObserved(e, slot) ? e->observed[slot]
                                                 : e->localSeen[slot])) {
                if (slot == 0) e->usesSlot0 = true;
                settleAll(e);          /* this path guards */
                fpSyncAll(e);          /* and calls; see emitStringOrder */
                unsigned rs = localIn(e, slot, JIT_SCRATCH_C);
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, rs,
                                   (unsigned)offsetof(Obj, type)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_STRING));
                branchOnDeopt(e, JAI_A64_NE);
                /* rs is a local's home (x19..) or JIT_SCRATCH_C, never x0 --
                 * but the test costs nothing and a `mov` reading a register it
                 * has already written is a miscompile, not a decline. */
                if (rs != 0) emit(e, jaiA64MovX(0, rs));
                emitConst64(e, 1, (int64_t)(uintptr_t)AS_OBJ(k));
                emitConst64(e, JIT_SCRATCH_A,
                            (int64_t)(uintptr_t)&jaiStringOrder);
                noteScratchClobber(e);
                emit(e, jaiA64Blr(JIT_SCRATCH_A));
                emit(e, jaiA64SubsXImm(31, 0, 0));
                branchTo(e, (uint32_t)((int32_t)next + jump), true, cond);
                off += 9;
                break;
            }

            /* The same instruction over floats: `if coefficient == 0.0`, the
             * early-out every numeric kernel opens its inner loop with. Only
             * the integer form was emitted, so a single float guard declined
             * the whole loop around it -- which is what left the JPEG inverse
             * DCT interpreted.
             *
             * fcmp's answers are not the signed-integer ones (an unordered
             * result has to come out false both ways), so the conditions are
             * the ones OP_JUMP_IF_CMP_FALSE's float arm uses, and an actual
             * NaN operand resumes in the interpreter. */
            if (e->localKind[slot] == SLOT_FLOAT && IS_FLOAT(k) &&
                AS_FLOAT(k) == AS_FLOAT(k)) {
                switch (cmp) {
                case OP_LT: cond = JAI_A64_GE; break;
                case OP_LE: cond = JAI_A64_GT; break;
                case OP_GT: cond = JAI_A64_LS; break;
                case OP_GE: cond = JAI_A64_MI; break;
                case OP_EQ: cond = JAI_A64_NE; break;
                case OP_NE: cond = JAI_A64_EQ; break;
                default: return false;
                }
                if (slot == 0) e->usesSlot0 = true;
                /* This arm reads a LOCAL, so unlike the stack-operand compare
                 * it never pops the entries that are borrowing -- and a borrow
                 * still live when nanToDeopt records is what deoptRecordAt
                 * asserts against. Releasing at the top of the instruction is
                 * sound here because nothing has branched yet. */
                settleAll(e);          /* nanToDeopt records */
                fpReleaseAll(e);
                localInFp(e, slot, JIT_FSCRATCH_A);
                if (AS_FLOAT(k) == 0.0) {
                    emit(e, jaiA64FcmpDZero(JIT_FSCRATCH_A));
                } else {
                    int64_t bits;
                    double kd = AS_FLOAT(k);
                    memcpy(&bits, &kd, sizeof bits);
                    emitConst64(e, JIT_SCRATCH_A, bits);
                    emit(e, jaiA64FmovDX(JIT_FSCRATCH_B, JIT_SCRATCH_A));
                    emit(e, jaiA64FcmpD(JIT_FSCRATCH_A, JIT_FSCRATCH_B));
                }
                nanToDeopt(e);
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
                return subWhy(e, "receiver local %u has kind %s, not instance",
                              slot, slotKindName(e->localKind[slot]));
            }
            if (e->localClass[slot] == NULL) {
                return subWhy(e, "receiver local %u has no pinned class", slot);
            }
            /* The name is resolved BEFORE the receiver guard is emitted, so
             * that a name this arm cannot read reaches `unarmedOpcode` with
             * nothing half-emitted -- the unarmed path takes over at the start
             * of an instruction. */
            if (nameIdx >= (uint32_t)fn->chunk.constants.count) {
                return subWhy(e, "the field name is not in the pool");
            }
            Value nameVal = fn->chunk.constants.data[nameIdx];
            if (!IS_STRING(nameVal)) return subWhy(e, "the field name is not a string");

            const FieldInfo *info =
                jaiClassFieldInfo(e->localClass[slot], AS_STRING(nameVal));
            /* The commonest of the four by a wide margin, and the one that
             * reads as a puzzle without being named: `self.method` is a METHOD,
             * and jaiClassFieldInfo only knows fields, so every
             * `return self.m(x)` written as a value lands here.
             *
             * It is 88 of resolve.jai's 104 OP_GET_FIELD_LOCAL declines, and
             * every one is the same shape: a keyword call to a private method,
             * `self._error(code, start, end, msg, help: "...")`. A keyword
             * argument rules out OP_INVOKE, so codegen has to materialise the
             * callee, and reading a method name as a value lands here.
             *
             * Those calls are on ERROR paths -- the lexer reaching a malformed
             * token -- which is what makes the unarmed path the right answer
             * rather than a decline, and the same reason OP_GET_GLOBAL takes it
             * for `throw ValueError(...)`. Softening a HOT refusal is a loss:
             * the same change on OP_GET_INDEX's `d[k]` measured 18% slower,
             * because a body that deopts every iteration costs more than one
             * that is simply interpreted. Cold is the whole condition.
             *
             * `!e->osr` is the same rule again, and measured too: inside a loop
             * nothing is cold. Dropping that guard compiles six MORE bodies in
             * resolve.jai (277 -> 283) and cancels the win -- 0.272s becomes
             * 0.274-0.294s against 0.274-0.284s for the hard refusal, where the
             * non-OSR-only form was a clean 3%. A loop that deopts inside
             * itself pays the entry and the exit every iteration. */
            if (info == NULL) {
                if (!e->osr && jitSoftField()) goto unarmedOpcode;
                return subWhy(e, "`%s` is not a field of %s",
                              AS_STRING(nameVal)->chars,
                              e->localClass[slot]->name
                                  ? e->localClass[slot]->name->chars : "?");
            }
            if (e->localKind[slot] == SLOT_MAYBE_INST) {
                unsigned rcv = localIn(e, slot, JIT_SCRATCH_A);
                emit(e, jaiA64SubsXImm(31, rcv, 0));
                branchOnDeopt(e, JAI_A64_EQ);
            }
            if (slot == 0) e->usesSlot0 = true;
            if (info->isStatic) {
                return subWhy(e, "`%s` is a static", AS_STRING(nameVal)->chars);
            }

            /* Field type is read off the LIVE receiver, so the tier specialises to what the program actually
             * stores rather than a declaration -- only possible for a parameter (hence the arity cap above): a local assigned further in has no value yet to look at. */
            Value seen = localObserved(e, slot) ? e->observed[slot]
                                                : e->localSeen[slot];
            if (!IS_INSTANCE(seen)) {
                return subWhy(e, "no live receiver to read local %u's field off", slot);
            }
            ObjInstance *inst = AS_INSTANCE(seen);
            if (info->slot >= inst->fieldCount) {
                return subWhy(e, "field slot %u is past the live receiver's %d",
                              (unsigned)info->slot, inst->fieldCount);
            }
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
            /* A bool field. Most classes carry one, and without this arm the
             * whole enclosing function declined -- `_check_live` in jaicv's
             * cascade is a method whose only field read is a bool, and it cost
             * 107 ms against 11.8 ms for the byte-identical method over an int
             * field. The load below is a `ldrb`, because BOOL_VAL writes only
             * the union's one-byte member. */
            else if (IS_BOOL(fieldVal)) { kind = SLOT_BOOL; tag = VAL_BOOL; }
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
                } else if (kind == SLOT_BOOL) {
                    /* One byte: BOOL_VAL writes only the union's `boolean`
                     * member, so the seven above it are whatever the field
                     * held before, and every SLOT_BOOL consumer tests the
                     * whole register. */
                    emit(e, jaiA64LdrByte(pushReg(e) - 1, recv, base + 8));
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
                /* Named, for the reason given at OP_GET_LOCAL. */
                if (!localInRange(e, slot)) {
                    e->whyNot = "a local outside the compiled window";
                    return false;
                }
                if (e->localKind[slot] == SLOT_OPAQUE) {
                    e->whyNot = "a local of no known kind";
                    return false;
                }
                if (slot == 0) e->usesSlot0 = true;
                if (!pushValue3(e, e->localKind[slot], e->localShape[slot],
                                e->localClass[slot],
                                localObserved(e, slot) ? e->observed[slot]
                                                       : e->localSeen[slot],
                                (int)slot)) {
                    return false;
                }
                /* Same seed as OP_GET_LOCAL, and this is the arm that matters:
                 * the emitter fuses `xs[j]` into GET_LOCAL2, so every subscript
                 * in a stencil arrives here and nowhere else. */
                if (e->localKind[slot] == SLOT_INT && slot <= UINT8_MAX) {
                    unsigned at = e->valueDepth - 1;
                    e->idxKnown |= 1u << at;
                    e->idxBase[at] = (uint8_t)slot;
                    e->idxOff[at]  = 0;
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

        case OP_POP_RETURN_NULL: {
            /* POP's half: forget a deferred/borrowed entry rather than settle
             * it, same as plain OP_POP -- nothing reads a value being thrown
             * away. */
            unsigned r;
            if (e->depth > 0 && holdsRegister(e->stack[e->depth - 1]) &&
                e->valueDepth > 0) {
                unsigned idx = e->valueDepth - 1;
                e->kPend   &= ~(1u << idx);
                e->xBorrow &= ~(1u << idx);
            }
            if (!popValue(e, &r, NULL)) return false;

            /* RETURN_NULL's half, unchanged. */
            if (e->osr) {
                e->whyNot = "a return inside an OSR loop";
                return false;
            }
            if ((fn->flags & FN_INIT) == 0) {
                if (e->sawReturn && e->returnKind != SLOT_NULL) return false;
                e->sawReturn  = true;
                e->returnKind = SLOT_NULL;
                emit(e, jaiA64MovzX(0, 0, 0));
                emitEpilogue(e, 0);
                off += 1;
                break;
            }
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
            uint32_t seenShape = 0;
            ObjClass *seenClass = NULL;
            if (IS_INT(seen))        { kind = SLOT_INT;   tag = VAL_INT; }
            else if (IS_FLOAT(seen)) { kind = SLOT_FLOAT; tag = VAL_FLOAT; }
            else if (IS_BOOL(seen))  { kind = SLOT_BOOL;  tag = VAL_BOOL; }
            else if (IS_LIST(seen))  { kind = SLOT_LIST;  tag = VAL_OBJ; }
            else if (rawObjValue(seen)) { kind = SLOT_OBJ; tag = VAL_OBJ; }
            else if (IS_INSTANCE(seen) && AS_INSTANCE(seen)->klass != NULL) {
                /* Same reasoning as every other raw-object read site in this
                 * tier: nothing about an upvalue makes its capture special,
                 * it is read the same way a global or a field is. A closure
                 * over a str/list/dict/instance never compiled before this. */
                kind = SLOT_INST; tag = VAL_OBJ;
                seenClass = AS_INSTANCE(seen)->klass;
                seenShape = seenClass->shapeId;
            } else return false;

            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_B, tag));
            branchOnDeopt(e, JAI_A64_NE);
            if (kind == SLOT_LIST) {
                /* "an object" is not "a list": same contract as every other
                 * SLOT_LIST arm in this tier. JIT_SCRATCH_A holds the
                 * upvalue's location and must survive to the load below. */
                emit(e, jaiA64LdrX(JIT_SCRATCH_D, JIT_SCRATCH_A, 8));
                emit(e, jaiA64LdrW(JIT_SCRATCH_B, JIT_SCRATCH_D,
                                   (unsigned)offsetof(Obj, type)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_B, OBJ_LIST));
                branchOnDeopt(e, JAI_A64_NE);
            } else if (kind == SLOT_INST) {
                emit(e, jaiA64LdrX(JIT_SCRATCH_D, JIT_SCRATCH_A, 8));
                emit(e, jaiA64LdrW(JIT_SCRATCH_B, JIT_SCRATCH_D,
                                   (unsigned)offsetof(Obj, type)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_B, OBJ_INSTANCE));
                branchOnDeopt(e, JAI_A64_NE);
                emit(e, jaiA64LdrX(JIT_SCRATCH_D, JIT_SCRATCH_D,
                                   (unsigned)offsetof(ObjInstance, klass)));
                emit(e, jaiA64LdrW(JIT_SCRATCH_D, JIT_SCRATCH_D,
                                   (unsigned)offsetof(ObjClass, shapeId)));
                emitConst64(e, JIT_SCRATCH_B, (int64_t)seenShape);
                emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_D, JIT_SCRATCH_B));
                branchOnDeopt(e, JAI_A64_NE);
            }
            if (!pushValue3(e, kind, seenShape, seenClass, seen, -1)) return false;
            if (kind == SLOT_BOOL) {
                emit(e, jaiA64LdrByte(pushReg(e) - 1, JIT_SCRATCH_A, 8));
            } else {
                emit(e, jaiA64LdrX(pushReg(e) - 1, JIT_SCRATCH_A, 8));
            }
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

            /* `Enum.Variant` for a payload-less variant, folded to the one
             * value that variant will ever have (EnumVariant::unit, made on
             * first mention and shared from then on so that `is` keeps
             * meaning identity). Without it the receiver is a global enum,
             * which globalKind classifies SLOT_OBJ, and the SLOT_INST
             * requirement below declined the whole enclosing function.
             *
             * The guard is the enum's shapeId, not its address: shapeId comes
             * from a monotonic counter precisely so that a freed enum whose
             * address is reused cannot be mistaken for the original, which is
             * what makes baking `unit` as a constant safe. `rcv` holds this
             * iteration's load of the global, so reading through it touches a
             * live object.
             *
             * The methods table is consulted first because enumMember does:
             * a method may shadow a variant name, and folding past that would
             * change what the program means.
             *
             * MEASURED. A loop dispatching on `k == Kind.X` over a token list
             * went 654 ms to 62 ms (10.5x, best of five alternating), and a
             * bare `Color.Red`/`Color.Green` compare loop 208 ms to 31 ms:
             * before this the whole loop was interpreted, and this was the
             * tier's single largest refusal on the self-hosted front end --
             * 189 distinct declining sites against 90 for the next reason,
             * the declined list being the whole of the lexer and the parser.
             *
             * RULED OUT: the front end itself does not get faster. It drops
             * 594 distinct declines to 540 and 469M interpreted instructions
             * to 459M (-2.0%), and 56 parser functions newly compile, but
             * `check` over ten large sources measured 3253 ms against 3253 ms
             * and `fmt --check lib` 3514 against 3471 -- both inside the
             * run-to-run spread. The functions that hold the time
             * (_parse_expression, _scan_punct, _punct_kind, keyword_lookup)
             * clear this refusal only to stop at the next one. Kept for the
             * loops it does unblock and as the prerequisite for that work,
             * not as a win on the compiler.
             *
             * REACH. Only an operand-stack enum receiver, which in practice
             * means a global: `Kind.X` written against a module-level enum.
             * The same mention through a LOCAL holding the enum arrives at
             * OP_GET_FIELD_LOCAL, which has no fold and still declines. That
             * is the next site, and the arm here ports to it unchanged. */
            if (e->stack[e->depth - 1] == SLOT_OBJ &&
                IS_OBJ_TYPE(seen, OBJ_ENUM) &&
                nameIdx < (uint32_t)fn->chunk.constants.count &&
                IS_STRING(fn->chunk.constants.data[nameIdx])) {
                ObjEnum *en = (ObjEnum *)AS_OBJ(seen);
                ObjString *vname = AS_STRING(fn->chunk.constants.data[nameIdx]);
                Value shadow;
                int tag = -1;
                if (!jaiTableGetInterned(&en->methods, vname, &shadow)) {
                    for (uint16_t vi = 0; vi < en->variantCount; vi++) {
                        if (en->variants[vi].name == vname ||
                            jaiStringEquals(en->variants[vi].name, vname)) {
                            tag = (int)vi;
                            break;
                        }
                    }
                }
                if (tag >= 0 && en->variants[tag].arity == 0 &&
                    en->variants[tag].unit != NULL) {
                    ObjEnumVal *unit = en->variants[tag].unit;
                    /* This path guards, so nothing may still be deferred. */
                    settleAll(e);
                    unsigned rcv = valueXReg(e, e->valueDepth - 1);
                    emit(e, jaiA64LdrW(JIT_SCRATCH_A, rcv,
                                       (unsigned)offsetof(Obj, type)));
                    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_ENUM));
                    branchOnDeopt(e, JAI_A64_NE);
                    emit(e, jaiA64LdrW(JIT_SCRATCH_A, rcv,
                                       (unsigned)offsetof(ObjEnum, shapeId)));
                    emitConst64(e, JIT_SCRATCH_B, (int64_t)en->shapeId);
                    emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
                    branchOnDeopt(e, JAI_A64_NE);
                    unsigned droppedEnum;
                    if (!popValue(e, &droppedEnum, NULL)) return false;
                    if (!pushValue3(e, SLOT_OBJ, 0, NULL, OBJ_VAL(unit), -1)) {
                        return false;
                    }
                    e->stackUnit[e->depth - 1] = true;
                    emitConst64(e, pushReg(e) - 1, (int64_t)(uintptr_t)unit);
                    off += 6;
                    break;
                }
            }

            /* `Klass.STATIC_NAME`: a static field read off a class object
             * known at compile time. The receiver came from OP_GET_GLOBAL's
             * `globalClass` arm (the only producer of SLOT_CLASS -- see its
             * comment: "resolved now and pinned by the module version check
             * at entry" and "occupies no register, baked into the call
             * sequence"), so `klass` here is not a guess.
             *
             * What is NOT known at compile time is the static's VALUE: it
             * lives in `klass->statics`, a plain JaiTable exactly like a
             * module's globals table, reassignable at any point after class
             * definition (see staticFieldSlot's comment -- `isLet` is not
             * enforced by the interpreter or the checker for a static, only
             * for an INSTANCE field written through `self`). So this reads it
             * the same way OP_GET_GLOBAL reads a plain module global BY
             * ADDRESS: the JaiEntry* is baked, and two guards (the table
             * hasn't rehashed, the value still has the kind observed at
             * compile time) stand between the load and trusting it. A wrong
             * guess, or a rebind between compile and this call, deopts; nothing
             * here can return a stale or mistyped answer.
             *
             * REACH. `TokenFlags.AFTER_NEWLINE` in Lexer._push
             * (lib/jaithon/compile/lexer.jai) -- 10/10 attempts hit this
             * exact gap with no secondary reason (docs/agents/fix-1.md). */
            if (jitStaticFieldEnabled() && e->stack[e->depth - 1] == SLOT_CLASS) {
                if (klass == NULL) {
                    return subWhy(e, "a static receiver with no class pinned");
                }
                if (nameIdx >= (uint32_t)fn->chunk.constants.count) {
                    return subWhy(e, "the field name is not in the pool");
                }
                Value sNameVal = fn->chunk.constants.data[nameIdx];
                if (!IS_STRING(sNameVal)) {
                    return subWhy(e, "the field name is not a string");
                }
                ObjString *sname = AS_STRING(sNameVal);

                const FieldInfo *sinfo = jaiClassFieldInfo(klass, sname);
                if (sinfo == NULL) {
                    return subWhy(e, "`%s` is not a field of %s", sname->chars,
                                  klass->name ? klass->name->chars : "?");
                }
                if (!sinfo->isStatic) {
                    return subWhy(e, "`%s` is an instance field, not a static",
                                  sname->chars);
                }

                JaiEntry *sslot = staticFieldSlot(e, klass, sname);
                if (sslot == NULL) {
                    return subWhy(e, "`%s` has no static storage, or is a "
                                  "second class's statics in one body",
                                  sname->chars);
                }
                Value sseen = sslot->value;

                SlotKind sk = SLOT_OPAQUE;
                uint32_t sshape = 0;
                ObjClass *sfcls = NULL;
                if (!globalKind(sseen, &sk, &sshape, &sfcls)) {
                    return subWhy(e, "static `%s` is a kind this tier cannot "
                                  "hold", sname->chars);
                }

                /* Named ahead of the guards, same reason as OP_GET_GLOBAL's
                 * BY-ADDRESS arm: they run against the model as it is now. */
                unsigned dst = valueXReg(e, e->valueDepth);
                unsigned stag = sk == SLOT_INT   ? VAL_INT
                              : sk == SLOT_FLOAT ? VAL_FLOAT
                              : sk == SLOT_BOOL  ? VAL_BOOL
                                                 : VAL_OBJ;

                emitStaticsGuard(e);
                emitConst64(e, JIT_SCRATCH_D, (int64_t)(uintptr_t)sslot);
                emit(e, jaiA64LdrW(JIT_SCRATCH_C, JIT_SCRATCH_D,
                                   (unsigned)offsetof(JaiEntry, value)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_C, stag));
                branchOnDeopt(e, JAI_A64_NE);
                emit(e, jaiA64LdrX(JIT_SCRATCH_C, JIT_SCRATCH_D,
                                   (unsigned)offsetof(JaiEntry, value) + 8u));
                if (sk == SLOT_INST || sk == SLOT_LIST) {
                    emit(e, jaiA64LdrByte(JIT_SCRATCH_B, JIT_SCRATCH_C,
                                          (unsigned)offsetof(Obj, type)));
                    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_B,
                                           sk == SLOT_INST ? OBJ_INSTANCE
                                                            : OBJ_LIST));
                    branchOnDeopt(e, JAI_A64_NE);
                }
                if (sk == SLOT_INST) {
                    emit(e, jaiA64LdrX(JIT_SCRATCH_B, JIT_SCRATCH_C,
                                       (unsigned)offsetof(ObjInstance, klass)));
                    emit(e, jaiA64LdrW(JIT_SCRATCH_B, JIT_SCRATCH_B,
                                       (unsigned)offsetof(ObjClass, shapeId)));
                    emitConst64(e, JIT_SCRATCH_A, (int64_t)sshape);
                    emit(e, jaiA64SubsX(31, JIT_SCRATCH_B, JIT_SCRATCH_A));
                    branchOnDeopt(e, JAI_A64_NE);
                }

                /* The receiver held no register (SLOT_CLASS -- see its
                 * comment on the enum): dropping it is the whole of popping
                 * it. popValue would refuse it via holdsRegister. */
                e->depth--;
                if (!pushValue3(e, sk, sshape, sfcls, sseen, -1)) return false;
                emit(e, jaiA64MovX(dst, JIT_SCRATCH_C));
                off += 6;
                break;
            }

            /* And, as at the local-receiver arm, a maybe-instance reads like an
             * instance once known not-null -- which is what `a.b.c` needs, since
             * `.b` came back SLOT_MAYBE_INST and `.c` wants a receiver.
             *
             * This arm and the SLOT_MAYBE_INST field arm below were the two
             * halves of the same hole: the whole chain `a.b.c.d.v` declined its
             * enclosing loop, and it now compiles. Measured on a loop reading
             * one four-deep chain per iteration, alternating old/new binaries
             * and warmed by the clock, best of five each: 0.4356 s -> 0.1107 s,
             * 3.9x. What did NOT move is a self-hosted compile (`check` over
             * parser.jai + emit.jai): 1.4253 s -> 1.4353 s, inside the 2% a
             * base-against-base control shows -- only eleven of the compiler's
             * bodies decline for these two reasons. The enum receiver that stops
             * 110 more of them is the arm directly above, and it did not move
             * the compile either. */
            if (e->stack[e->depth - 1] != SLOT_INST &&
                e->stack[e->depth - 1] != SLOT_MAYBE_INST) {
                return false;
            }
            if (klass == NULL) return false;
            if (e->stack[e->depth - 1] == SLOT_MAYBE_INST) {
                emit(e, jaiA64SubsXImm(31, valueBankReg(e, e->valueDepth - 1), 0));
                branchOnDeopt(e, JAI_A64_EQ);
            }
            if (nameIdx >= (uint32_t)fn->chunk.constants.count) return false;
            Value nameVal = fn->chunk.constants.data[nameIdx];
            if (!IS_STRING(nameVal)) return false;

            const FieldInfo *info = jaiClassFieldInfo(klass, AS_STRING(nameVal));
            if (info == NULL || info->isStatic) return false;
            if (!IS_INSTANCE(seen)) {
                /* No sample to classify the field from. The commonest cause
                 * is a receiver OP_INVOKE itself predicted: `klass` came from
                 * observedReturnKind's shape (or a compiled callee's
                 * jitReturnShape) by way of jaiClassForShape, which pins the
                 * class exactly, but pushValue leaves `seen` at NULL_VAL
                 * because there is no actual instance behind a prediction,
                 * only a shape id -- `self._peek().kind`, `self._chunk().depth`
                 * and every other zero-arg-accessor-then-field-read chain in
                 * the self-hosted front end is this shape, and it declined
                 * outright before this arm existed. See declaredScalarFieldKind
                 * for why only a scalar can be predicted from here. */
                SlotKind dkind;
                unsigned dtag;
                if (!jitDeclaredFieldKindEnabled() ||
                    !declaredScalarFieldKind(info->typeId, &dkind, &dtag)) {
                    return subWhy(e,
                        "no live receiver to read `%s` off, and its declared "
                        "kind is not one this predicts",
                        AS_STRING(nameVal)->chars);
                }
                unsigned dbase = (unsigned)offsetof(ObjInstance, fields) +
                                 (unsigned)info->slot * (unsigned)sizeof(Value);
                unsigned drr = valueBankReg(e, e->valueDepth - 1);
                /* Guard BEFORE the receiver comes off the model, as the
                 * sampled path below does: a deopt here resumes at this
                 * instruction, and the interpreter's stack still has the
                 * receiver on it. */
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, drr, dbase));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, dtag));
                branchOnDeopt(e, JAI_A64_NE);
                if (dkind == SLOT_LIST) {
                    /* VAL_OBJ said "a heap object" and no more. Prove OBJ_LIST
                     * before the entry claims to be one, or the next arm reads
                     * ObjList's count out of a string's header. */
                    emit(e, jaiA64LdrX(JIT_SCRATCH_B, drr, dbase + 8));
                    emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_B,
                                       (unsigned)offsetof(Obj, type)));
                    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_LIST));
                    branchOnDeopt(e, JAI_A64_NE);
                }
                unsigned dpopped;
                SlotKind dkr;
                if (!popValue(e, &dpopped, &dkr)) return false;
                if (!pushValue3(e, dkind, 0, NULL, NULL_VAL, -1)) return false;
                if (dkind == SLOT_FLOAT &&
                    fpWorthLoading(e, code, off + 6, stop)) {
                    unsigned idx = e->valueDepth - 1;
                    emit(e, jaiA64LdrD(fpRegAt(e, idx), drr, dbase + 8));
                    fpClaim(e, idx);
                } else if (dkind == SLOT_BOOL) {
                    /* One byte, for the reason given at the local-receiver
                     * arm: BOOL_VAL writes only the union's bool member. */
                    emit(e, jaiA64LdrByte(pushReg(e) - 1, drr, dbase + 8));
                } else {
                    emit(e, jaiA64LdrX(pushReg(e) - 1, drr, dbase + 8));
                }
                off += 6;
                break;
            }
            ObjInstance *inst = AS_INSTANCE(seen);
            if (info->slot >= inst->fieldCount) return false;
            Value fieldVal = inst->fields[info->slot];

            SlotKind kind;
            unsigned tag;
            ObjClass *fcls = NULL;
            if (IS_INT(fieldVal))        { kind = SLOT_INT;   tag = VAL_INT; }
            else if (IS_FLOAT(fieldVal)) { kind = SLOT_FLOAT; tag = VAL_FLOAT; }
            /* A nullable instance field, same guess and same guards as the
             * local-receiver arm: a leaf's null has no class to read, so the
             * receiver's own class stands in and the class guard below deopts
             * if that was wrong. Without this arm `rawObjValue` (which excludes
             * OBJ_INSTANCE) let an instance-valued field fall off the end of
             * the chain and decline the whole function. */
            else if (IS_INSTANCE(fieldVal) || IS_NULL(fieldVal)) {
                kind = SLOT_MAYBE_INST;
                tag  = VAL_OBJ;
                fcls = IS_INSTANCE(fieldVal) ? AS_INSTANCE(fieldVal)->klass
                                             : klass;
                if (fcls == NULL) return false;
            }
            /* As in OP_GET_FIELD_LOCAL: an object-typed field is held raw
             * rather than declining the enclosing function, and a list earns
             * the stronger kind at the price of an OBJ_LIST check. */
            else if (IS_LIST(fieldVal))     { kind = SLOT_LIST; tag = VAL_OBJ; }
            /* A bool field. Most classes carry one, and without this arm the
             * whole enclosing function declined -- `_check_live` in jaicv's
             * cascade is a method whose only field read is a bool, and it cost
             * 107 ms against 11.8 ms for the byte-identical method over an int
             * field. The load below is a `ldrb`, because BOOL_VAL writes only
             * the union's one-byte member. */
            else if (IS_BOOL(fieldVal)) { kind = SLOT_BOOL; tag = VAL_BOOL; }
            else if (rawObjValue(fieldVal)) { kind = SLOT_OBJ;  tag = VAL_OBJ; }
            else return false;

            unsigned fbase = (unsigned)offsetof(ObjInstance, fields) +
                             (unsigned)info->slot * (unsigned)sizeof(Value);
            unsigned rr = valueBankReg(e, e->valueDepth - 1);
            SlotKind already = knownFieldKind(e, fromLocal, info->slot);
            if (kind == SLOT_MAYBE_INST) {
                /* The same three guards the local arm emits, branch-free by the
                 * same trick: a null payload redirects the loads at the receiver,
                 * which the entry check above has already proved is a live
                 * instance, so nothing ever dereferences zero. All of it stands
                 * before the receiver comes off the model, since every deopt
                 * here resumes at this instruction with it still on the stack. */
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, rr, fbase));       /* tag */
                emit(e, jaiA64LdrX(JIT_SCRATCH_D, rr, fbase + 8));   /* value */
                emit(e, jaiA64MovzX(JIT_SCRATCH_B, 0, 0));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, VAL_OBJ));
                emit(e, jaiA64CselX(JIT_SCRATCH_D, JIT_SCRATCH_D,
                                    JIT_SCRATCH_B, JAI_A64_EQ));
                emit(e, jaiA64CselX(JIT_SCRATCH_B, JIT_SCRATCH_B,
                                    JIT_SCRATCH_A, JAI_A64_EQ));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_B, 0));
                branchOnDeopt(e, JAI_A64_NE);      /* neither object nor null */

                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_D, 0));
                emit(e, jaiA64CselX(JIT_SCRATCH_B, rr, JIT_SCRATCH_D,
                                    JAI_A64_EQ));
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_B,
                                   (unsigned)offsetof(Obj, type)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_INSTANCE));
                branchOnDeopt(e, JAI_A64_NE);

                emit(e, jaiA64LdrX(JIT_SCRATCH_A, JIT_SCRATCH_B,
                                   (unsigned)offsetof(ObjInstance, klass)));
                emitConst64(e, JIT_SCRATCH_B, (int64_t)(uintptr_t)fcls);
                if (fcls != klass) {
                    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_D, 0));
                    emit(e, jaiA64CselX(JIT_SCRATCH_A, JIT_SCRATCH_B,
                                        JIT_SCRATCH_A, JAI_A64_EQ));
                }
                emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
                branchOnDeopt(e, JAI_A64_NE);
            } else if (already != SLOT_SELF) {
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
            if (kind == SLOT_MAYBE_INST) {
                if (!pushValue3(e, kind, fcls->shapeId, fcls,
                                IS_INSTANCE(fieldVal) ? fieldVal : NULL_VAL,
                                -1)) {
                    return false;
                }
                /* Already in hand: the guards above loaded the payload (or a
                 * zero) into D, so there is nothing left to read. */
                emit(e, jaiA64MovX(pushReg(e) - 1, JIT_SCRATCH_D));
            } else {
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
                } else if (kind == SLOT_BOOL) {
                    /* One byte, for the reason given at the local-receiver arm. */
                    emit(e, jaiA64LdrByte(pushReg(e) - 1, rr, fbase + 8));
                } else {
                    emit(e, jaiA64LdrX(pushReg(e) - 1, rr, fbase + 8));
                }
            }
            off += 6;
            break;
        }

        case OP_BUILD_LIST: {
            unsigned n = jaiReadU16(code + off + 1);
            if (n > JIT_MAX_ARGS_OUT) return false;
            if (!e->callsOut) return false;
            if (e->depth < n) return false;
            Value elemSeen = NULL_VAL;
            if (!buildListExemplar(e, e->depth - n, n, &elemSeen)) {
                elemSeen = NULL_VAL;
            }
            if (!emitDescriptor(e, NULL_VAL, e->depth - n, n,
                                (void *)&jitBuildList)) {
                return false;
            }
            for (unsigned i = 0; i < n; i++) {
                unsigned r;
                if (!popValue(e, &r, NULL)) return false;
            }
            if (!pushValue(e, SLOT_LIST, 0, NULL)) return false;
            /* Computed BEFORE the pops above, since it reads the entries they
             * remove. See buildListExemplar for why the list itself cannot
             * answer this. */
            e->stackElem[e->depth - 1] = elemSeen;
            emit(e, jaiA64LdrX(pushReg(e) - 1, 31,
                               e->descOffset +
                                   (unsigned)offsetof(JitCallDesc, result) + 8));
            e->wroteHeap = true;
            off += 3;
            break;
        }

        case OP_IN:
        case OP_NOT_IN: {
            /* `x in c`. No arm existed, so a membership test ENDED THE WALK:
             * `if k in seen` is the shape of every dedup loop in the corpus and
             * everything after it ran interpreted.
             *
             * The containment itself is not made faster -- it is the same
             * jaiContainsOp the interpreter runs, called out to. What the arm
             * buys is the body around it, which is the whole point of a row
             * over a call that is cheap next to its loop.
             *
             * `not in` is the same call with the sense flipped, in its own
             * entry point rather than an argc flag -- a wider descriptor would
             * name a stack entry past the operands. */
            if (!jitMembership() || !e->callsOut || e->depth < 2) {
                goto unarmedOpcode;
            }
            if (!emitDescriptor(e, NULL_VAL, e->depth - 2, 2,
                                code[off] == OP_IN ? (void *)&jitContains
                                                   : (void *)&jitNotContains)) {
                return false;
            }
            for (unsigned i = 0; i < 2; i++) {
                unsigned r;
                if (!popValue(e, &r, NULL)) return false;
            }
            if (!pushValue(e, SLOT_BOOL, 0, NULL)) return false;
            emit(e, jaiA64LdrByte(pushReg(e) - 1, 31,
                                  e->descOffset +
                                      (unsigned)offsetof(JitCallDesc, result) +
                                      8));
            /* Containment is not pure: a class can define __contains__, so the
             * call may run Jaithon code that writes. Leaving this unset marked
             * every body holding an `in` jitFuncNoWrite, which lets a direct
             * caller finish the callee by RE-RUNNING it from the start on a
             * bail -- and re-running the writes with it. */
            e->wroteHeap = true;
            off += 1;
            break;
        }

        case OP_BUILD_DICT:
        case OP_BUILD_SET: {
            /* The remaining container literals, on the OP_BUILD_LIST template.
             *
             * Both were top of the partial-walk census over the self-hosted
             * parser: `_node(kind, span, fields: dict = {})` builds a dict for
             * every AST node, and the walk stopped there sixteen times in one
             * file. */
            bool isDict = code[off] == OP_BUILD_DICT;
            unsigned n = jaiReadU16(code + off + 1);
            unsigned operands = isDict ? n * 2u : n;
            if (!jitTuple() || operands > JIT_MAX_ARGS_OUT || !e->callsOut ||
                e->depth < operands) {
                goto unarmedOpcode;
            }
            if (!emitDescriptor(e, NULL_VAL, e->depth - operands, operands,
                                isDict ? (void *)&jitBuildDict
                                       : (void *)&jitBuildSet)) {
                return false;
            }
            for (unsigned i = 0; i < operands; i++) {
                unsigned r;
                if (!popValue(e, &r, NULL)) return false;
            }
            if (!pushValue(e, SLOT_OBJ, 0, NULL)) return false;
            /* SLOT_OBJ does not say which container this is, and OP_ELEM_KIND
             * comes straight after a literal and has to know. */
            e->stackObjType[e->depth - 1] =
                (uint8_t)((isDict ? OBJ_DICT : OBJ_SET) + 1);
            emit(e, jaiA64LdrX(pushReg(e) - 1, 31,
                               e->descOffset +
                                   (unsigned)offsetof(JitCallDesc, result) + 8));
            e->wroteHeap = true;
            off += 3;
            break;
        }

        case OP_BUILD_TUPLE: {
            /* Same shape as OP_BUILD_LIST above, and simpler: jaiTupleNew
             * copies the operands itself and cannot throw. No exemplar is
             * kept -- a tuple has no element arm to feed, so the entry is a
             * plain SLOT_OBJ.
             *
             * Worth an arm only because there was none: a tuple build ENDED
             * THE WALK, and `let p = (x, y)` in a loop body is common enough
             * that the whole body after it ran interpreted. */
            unsigned n = jaiReadU16(code + off + 1);
            if (!jitTuple() || n > JIT_MAX_ARGS_OUT || !e->callsOut ||
                e->depth < n) {
                goto unarmedOpcode;
            }
            if (!emitDescriptor(e, NULL_VAL, e->depth - n, n,
                                (void *)&jitBuildTuple)) {
                return false;
            }
            for (unsigned i = 0; i < n; i++) {
                unsigned r;
                if (!popValue(e, &r, NULL)) return false;
            }
            if (!pushValue(e, SLOT_OBJ, 0, NULL)) return false;
            e->stackObjType[e->depth - 1] = (uint8_t)(OBJ_TUPLE + 1);
            emit(e, jaiA64LdrX(pushReg(e) - 1, 31,
                               e->descOffset +
                                   (unsigned)offsetof(JitCallDesc, result) + 8));
            e->wroteHeap = true;
            off += 3;
            break;
        }

        case OP_NOT: {
            /* `not x`. The interpreter REQUIREs a bool here, and SLOT_BOOL's
             * contract is "0 or 1 in a register", so the flip is an xor with
             * one and there is nothing to guard. Anything else is not a
             * narrowing this tier declines to do -- it is a program the
             * interpreter would throw on, and it reaches the throw by the
             * unarmed path.
             *
             * No arm existed, so `not` ENDED THE WALK the way OP_NEG did:
             * `if not a` in a loop was 23,252,579 interpreted instructions. */
            if (!jitNegate() || e->depth < 1 ||
                e->stack[e->depth - 1] != SLOT_BOOL) {
                goto unarmedOpcode;
            }
            {
                unsigned nr = pushReg(e) - 1;
                emitConst64(e, JIT_SCRATCH_A, 1);
                emit(e, jaiA64EorX(nr, nr, JIT_SCRATCH_A));
            }
            off += 1;
            break;
        }

        case OP_BNOT: {
            /* `~x` is `x ^ -1`, and the model has already proved the int. */
            if (!jitNegate() || e->depth < 1 ||
                e->stack[e->depth - 1] != SLOT_INT) {
                goto unarmedOpcode;
            }
            {
                unsigned nr = pushReg(e) - 1;
                emitConst64(e, JIT_SCRATCH_A, -1);
                emit(e, jaiA64EorX(nr, nr, JIT_SCRATCH_A));
            }
            off += 1;
            break;
        }

        case OP_POS: {
            /* Unary `+` on a number is the identity -- the interpreter checks
             * the type and does nothing else. The model has already proved it,
             * so this emits nothing at all; the point is only that the walk
             * does not stop here. */
            if (!jitNegate() || e->depth < 1 ||
                (e->stack[e->depth - 1] != SLOT_INT &&
                 e->stack[e->depth - 1] != SLOT_FLOAT)) {
                goto unarmedOpcode;
            }
            off += 1;
            break;
        }

        case OP_NEG: {
            /* `-x`. There was no arm at all, for either kind: OP_NEG appeared
             * only in inlinableBody's whitelist, so in an ordinary body it fell
             * to `default` and ENDED THE WALK -- everything after a negation
             * ran interpreted. `-(i & 255)` in a loop was 28,394,773
             * interpreted instructions and the float form 18,034,610.
             *
             * It hid because emitUnarmedDeopt is silent by design (the point of
             * it is to interpret from here rather than decline the body), so
             * nothing was ever printed. What surfaced it was a DIFFERENT
             * message: `is_inf`'s `x == INF or x == -INF` records a forward
             * branch to its OP_RETURN before reaching the negation, and once
             * the walk stops that branch has nowhere to land. The fixup pass
             * now names the opcode that ended the walk, which is how a
             * confusing "branch to offset 25" became "OP_NEG at 23". Every
             * guarded function in std.math goes through `_require_finite` and
             * so through `is_inf`. */
            if (!jitNegate()) {
                return subWhy(e, "the negate arm is switched off");
            }
            if (e->depth < 1) return subWhy(e, "nothing to negate");
            SlotKind nk = e->stack[e->depth - 1];
            if (nk == SLOT_INT) {
                /* `subs` off zero both negates and reports the one input that
                 * cannot be: INT64_MIN overflows, and the interpreter raises
                 * the OverflowError on re-entry.
                 *
                 * Into a SCRATCH first, and moved only once the guard has
                 * passed -- the same discipline the `abs` arm states, and the
                 * reason is this instruction resumes at its own START. Writing
                 * the operand's register before the guard hands the deopt
                 * record a value that has ALREADY been negated, and the
                 * interpreter negates it again: `neg_int(5)` returned 5.
                 * Only JAITHON_JIT_DEOPT_STRESS=1 finds it, because in
                 * ordinary running the guard fires only on INT64_MIN, which is
                 * its own negation and so hides the mistake. */
                unsigned nr = pushReg(e) - 1;
                emit(e, jaiA64SubsX(JIT_SCRATCH_A, JAI_A64_XZR, nr));
                branchOnDeoptInstStart(e, JAI_A64_VS);
                emit(e, jaiA64MovX(nr, JIT_SCRATCH_A));
                off += 1;
                break;
            }
            if (nk == SLOT_FLOAT) {
                /* Straight in the bank. IEEE negation is the sign bit and
                 * cannot fail, so there is no guard and nothing to raise --
                 * -0.0 and NaN both come out of `fneg` the way the interpreter
                 * produces them. */
                unsigned nd = fpOperand(e, e->valueDepth - 1);
                unsigned drop;
                if (!popValueRaw(e, &drop, NULL)) return false;
                if (!pushValue(e, SLOT_FLOAT, 0, NULL)) return false;
                unsigned nidx = e->valueDepth - 1;
                emit(e, jaiA64FnegD(fpRegAt(e, nidx), nd));
                fpClaim(e, nidx);
                off += 1;
                break;
            }
            return subWhy(e, "negating a %s", slotKindName(nk));
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
            unsigned ry = pushReg(e) - 1, rx = valueXReg(e, e->valueDepth - 2);

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

            /* A literal power-of-two divisor decides the whole thing: floor(x / 2^s) is exactly `asr x, #s` for
             * every int64 x (negative included), since asr already rounds toward minus infinity -- what the correction below exists to reproduce for the general case. */
            int64_t kdiv = 0;
            bool kdivKnown = literalIntOperand(fn, prevOff, off, &kdiv) &&
                             kdiv != 0 && kdiv != -1;
            unsigned dshift;
            if (kdivKnown && powerOfTwoShift(kdiv, &dshift)) {
                /* The divisor is spelt by the shift field, so drop its deferral rather than settle it -- the
                 * same move OP_ADD makes for an imm12 -- popValueRaw drops it. The dividend comes back from popValue, not from pushReg, because a borrowed entry lives in the local's register and not in its own. */
                unsigned p1, p2;
                if (!popValueRaw(e, &p1, NULL)) return false;
                if (!popValue(e, &p2, NULL)) return false;
                if (!pushValue(e, SLOT_INT, 0, NULL)) return false;
                emit(e, jaiA64AsrX(pushReg(e) - 1, p2, dshift));
                off += 1;
                break;
            }
            /* Every path below reads both operands out of their own registers,
             * and two of them guard. */
            settleAll(e);
            rb = pushReg(e) - 1; ra = valueXReg(e, e->valueDepth - 2);

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

        /* A list comprehension's append. It is the same two stores `push`
         * makes, but it was the one list write with no arm at all, so every
         * comprehension in the language -- `[0 for _i in 0..n]`, the way this
         * codebase preallocates -- left its loop running interpreted. */
        case OP_LIST_APPEND: {
            unsigned back = jaiReadU16(code + off + 1);
            if (e->depth < back + 1u) {
                e->whyNot = "an append reaching past the model"; return false;
            }
            unsigned lidx = e->depth - 1u - back;
            if (e->stack[lidx] != SLOT_LIST) {
                e->whyNot = "an append to an entry the tier does not know is a list";
                return false;
            }
            if (!emitListStore(e, e->stack[e->depth - 1],
                               valueXReg(e, valueIndexOf(e, lidx)),
                               pushReg(e) - 1, e->stackLocal[lidx])) {
                return false;
            }
            /* The value is consumed; the target stays where it was. */
            unsigned dropAppend;
            if (!popValue(e, &dropAppend, NULL)) return false;
            off += 3;
            break;
        }

        case OP_INVOKE: {
            uint32_t nameIdx = jaiReadU24(code + off + 1);
            unsigned argc    = code[off + 4];
            /* The argc cap is a fixed limit rather than a property of this
             * call, and it was indistinguishable in the census from every
             * other reason an invoke declines -- which is how it went a long
             * time without anyone knowing how many sites it costs. */
            if (argc > JIT_MAX_ARGS_OUT - 1) {
                return subWhy(e, "%u arguments, past the cap of %d",
                              argc, JIT_MAX_ARGS_OUT - 1);
            }
            if (!e->callsOut) return subWhy(e, "this region may not call out");
            if (e->depth < argc + 1) {
                return subWhy(e, "the model is %u deep for %u arguments",
                              e->depth, argc);
            }

            unsigned ridx = e->depth - argc - 1;
            SlotKind rk = e->stack[ridx];

            /* A maybe-instance receiver is an instance once it is known not to
             * be null, and the whole arm below already resolves against the
             * class the entry carries -- which a maybe-instance carries too.
             * So one compare buys it: prove it here, then let it read as
             * SLOT_INST for the rest of the instruction.
             *
             * Order matters. The guard's record is taken while the entry is
             * still MAYBE_INST, because that is what the interpreter has on
             * its stack if the guard fires; the narrowing happens only after,
             * on the path where the proof holds. Both kinds hold a bare
             * pointer in a register, so nothing about the allocation moves.
             *
             * This is what `var at = head` / `while at is not null` /
             * `at = at.follow()` needs -- the loop shape every list and tree
             * walk has. Without it the OSR retry that widens `at` to
             * MAYBE_INST only moved the refusal from the assignment to the
             * call. */
            if (rk == SLOT_MAYBE_INST && e->stackClass[ridx] != NULL) {
                settleAll(e);
                emit(e, jaiA64SubsXImm(31, valueXReg(e, ridx - (e->depth - e->valueDepth)), 0));
                branchOnDeopt(e, JAI_A64_EQ);
                e->stack[ridx] = SLOT_INST;
                rk = SLOT_INST;
            }

            if (rk == SLOT_LIST && argc == 1 &&
                nameIdx < (uint32_t)fn->chunk.constants.count &&
                IS_STRING(fn->chunk.constants.data[nameIdx]) &&
                strcmp(AS_STRING(fn->chunk.constants.data[nameIdx])->chars,
                       "push") == 0) {
                /* Appending to a list is a bounds check and two stores -- a descriptor+native round trip costs far
                 * more than the work itself (list_ops spent all its time on the call). A full list goes out to the `grow` stubs' realloc helper and comes straight back; see there for why this used to be a deopt and what that cost. */
                if (!emitListStore(e, e->stack[e->depth - 1],
                                   valueXReg(e, e->valueDepth - 2),
                                   pushReg(e) - 1, e->stackLocal[ridx])) {
                    return false;
                }
                /* push returns the list, which is the receiver entry already
                 * sitting under the argument. */
                unsigned drop;
                if (!popValue(e, &drop, NULL)) return false;
                off += 7;
                break;
            }

            /* `Klass.static_method(x)`. NOT a field read plus a call, which is
             * what an earlier attempt at this assumed and why it measured
             * nothing: the compiler emits OP_GET_GLOBAL "Box" and then an
             * OP_INVOKE whose receiver is the class, so an arm in OP_GET_FIELD
             * never sees it. See emitClassCall. */
            if (rk == SLOT_CLASS && jitClassCalls()) {
                ObjClass *scls = e->stackClass[ridx];
                if (scls == NULL) {
                    return subWhy(e, "a class receiver the model did not pin");
                }
                if (nameIdx >= (uint32_t)fn->chunk.constants.count) {
                    return subWhy(e, "the member name is not in the pool");
                }
                Value sname = fn->chunk.constants.data[nameIdx];
                if (!IS_STRING(sname)) {
                    return subWhy(e, "the member name is not a string");
                }
                Value smember;
                if (!jaiTableGetInterned(&scls->statics, AS_STRING(sname),
                                         &smember)) {
                    /* The interpreter's own order: statics first, then
                     * `methods`. An instance method IS reachable this way and
                     * must not be admitted -- resolveInvokeTarget hands the
                     * bare closure to invokeCallable with the CLASS sitting in
                     * slot 0, so `self` is the class and the body raises the
                     * moment it touches a field ("class 'Box' has no member
                     * 'v'", measured). Compiling that faithfully would take a
                     * receiver this arm exists to drop, and the reward would be
                     * a faster way to reach the same exception. */
                    Value smeth;
                    if (jaiTableGetInterned(&scls->methods, AS_STRING(sname),
                                            &smeth)) {
                        return subWhy(e, "`%s.%s` is an instance method reached "
                                         "through the class",
                                      scls->name != NULL ? scls->name->chars
                                                         : "?",
                                      AS_STRING(sname)->chars);
                    }
                    return subWhy(e, "`%s` is not a static of %s",
                                  AS_STRING(sname)->chars,
                                  scls->name != NULL ? scls->name->chars : "?");
                }
                /* Visibility is the interpreter's methodPermitted, and it is
                 * NOT a compile-time property in general: accessPermitted reads
                 * the running frame's owner class. A non-public static DOES
                 * reach here -- the checker rejects `Box.hidden(i)` from
                 * outside with E0701, but a `static fn` with no `pub` called
                 * from a method of its own class type-checks and runs. This arm
                 * resolves once and never calls methodPermitted again, so
                 * admitting one would be skipping a check the interpreter makes
                 * on every call. Restricted means non-public, which is the
                 * whole condition.
                 *
                 * Reached, not hypothetical: `Hidden.walk` in
                 * tests/lang/test_jit_class_call.jai stops here, and the
                 * targeted suites hit it nine times. */
                MethodInfo smi;
                if (jaiClassRestrictedMethod(scls, AS_STRING(sname), &smi)) {
                    return subWhy(e, "`%s.%s` is not public",
                                  scls->name != NULL ? scls->name->chars : "?",
                                  AS_STRING(sname)->chars);
                }
                /* A closure straight out of the table, never a bound method and
                 * never a native: this arm drops the receiver, and both of the
                 * others want one. See emitClassCall's second paragraph. */
                if (!IS_CLOSURE(smember)) {
                    return subWhy(e, "`%s.%s` is a %s, not a function",
                                  scls->name != NULL ? scls->name->chars : "?",
                                  AS_STRING(sname)->chars,
                                  jaiTypeNameStatic(smember));
                }
                /* The ENTRY, not just the value: the guard reads the
                 * binding back out of this slot on every call. */
                JaiEntry *sslot =
                    jaiTableFindEntryInterned(&scls->statics, AS_STRING(sname));
                if (sslot == NULL) {
                    return subWhy(e, "a static with no table entry");
                }
                if (!emitClassCall(e, scls, sslot, smember, ridx, argc,
                                   (uint32_t)(off + 7))) {
                    return false;
                }
                if (getenv("JAI_JIT_WHY")) {
                    fprintf(stderr, "[jit] static call %s.%s at %d\n",
                            scls->name != NULL ? scls->name->chars : "?",
                            AS_STRING(sname)->chars, off);
                }
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
                if (nameIdx >= (uint32_t)fn->chunk.constants.count) return false;
                Value mname = fn->chunk.constants.data[nameIdx];
                if (!IS_STRING(mname)) return false;

                /* An instance whose class the model could not pin -- what
                 * `for op in ops` produces when the list holds more than one
                 * implementation of a trait. There is no method to resolve
                 * here, so the name goes out instead and jitInvokeByName
                 * resolves per call. Declining instead was worth 2.4x on the
                 * two-class form of tests/bench/poly_dispatch, entirely
                 * because the loop around it then ran interpreted. */
                if (rcls == NULL) {
                    SlotKind rkind;
                    unsigned rtag;
                    const bool discarded =
                        (off + 7 < count && code[off + 7] == OP_POP);
                    const uint16_t invokeCache = jaiReadU16(code + off + 5);
                    const bool havePrediction =
                        siteInvokeResultKind(&fn->chunk, invokeCache,
                                             &rkind, &rtag);
                    if (!havePrediction && !discarded) {
                        e->whyNot = "an unpinned receiver's result kind";
                        return false;
                    }

                    /* The site's own inline cache first: the ONE class it
                     * saw first gets a compare and a direct branch into its
                     * compiled entry, and only a receiver that isn't that
                     * class pays for the descriptor below -- which, absent
                     * this, is every receiver. See emitInvokePic1. */
                    int picToEnd = -1;
                    if (jitPicEnabled()) {
                        if (emitInvokePic1(e, fn, ridx, argc, (uint32_t)off,
                                          (uint32_t)(off + 7),
                                          (int)invokeCache, havePrediction,
                                          havePrediction ? rkind : SLOT_NULL,
                                          &picToEnd)) {
                            vm.jitPicAdmits++;
                        } else {
                            if (e->failed) return false;
                            vm.jitPicRefusals++;
                        }
                    }

                    if (!emitDescriptor(e, mname, ridx, argc + 1,
                                        (void *)&jitInvokeByName)) {
                        return false;
                    }
                    for (unsigned i = 0; i <= argc; i++) {
                        unsigned r;
                        if (!popValue(e, &r, NULL)) return false;
                    }
                    e->wroteHeap = true;
                    if (!havePrediction) {
                        off += 8;          /* the OP_POP this consumed */
                        break;
                    }
                    if (!pushValue(e, rkind, 0, NULL)) return false;
                    unsigned rat = e->descOffset +
                                   (unsigned)offsetof(JitCallDesc, result);
                    emit(e, jaiA64LdrW(JIT_SCRATCH_A, 31, rat));
                    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, rtag));
                    branchOnDeoptAt(e, JAI_A64_NE, (uint32_t)(off + 7), true);
                    if (rkind == SLOT_BOOL) {
                        emit(e, jaiA64LdrByte(pushReg(e) - 1, 31, rat + 8));
                    } else {
                        emit(e, jaiA64LdrX(pushReg(e) - 1, 31, rat + 8));
                    }
                    /* The merge: the one way branched here rather than round
                     * the descriptor, arriving with the same entry in the
                     * same register this path just loaded -- the whole of
                     * what the two paths had to agree about for the arm to
                     * be emitted at all. */
                    for (unsigned pw = 0; pw < e->picExitCount; pw++) {
                        const int at = e->picExits[pw];
                        if (at >= 0 && at < (int)e->count &&
                            e->count <= JIT_MAX_INSTS) {
                            e->code[at] =
                                jaiA64B((int32_t)((int)e->count - at));
                        }
                    }
                    e->picExitCount = 0;
                    off += 7;
                    break;
                }
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
                if (haveKind &&
                    (rkind == SLOT_INST || rkind == SLOT_MAYBE_INST) &&
                    (rshape == 0 || !jaiClassForShape(rshape, &rrcls) ||
                     rrcls == NULL)) {
                    haveKind = false;
                    rrcls = NULL;
                }
                if (haveKind && rkind != SLOT_INT && rkind != SLOT_FLOAT &&
                    rkind != SLOT_BOOL && rkind != SLOT_INST &&
                    rkind != SLOT_MAYBE_INST && rkind != SLOT_LIST &&
                    rkind != SLOT_OBJ && rkind != SLOT_NULL) {
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
                if (rkind == SLOT_MAYBE_INST) {
                    emitMaybeInstResult(e, pushReg(e) - 1, rat, rshape,
                                        (uint32_t)(off + 7));
                    off += 7;
                    break;
                }
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
                } else if (rkind == SLOT_LIST) {
                    /* Same hazard as SLOT_INST above: a method entered with
                     * another specialisation runs interpreted and may
                     * return any type, so VAL_OBJ alone does not prove the
                     * payload is a list before ObjList's fields get read
                     * off it unguarded downstream. */
                    emit(e, jaiA64LdrW(JIT_SCRATCH_A, pushReg(e) - 1,
                                       (unsigned)offsetof(Obj, type)));
                    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_LIST));
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
                bool  oProbe = false;
                /* Read before the refusal below can fire, so the refusal can
                 * name the method. Without the name it said only "an invoke on
                 * an object with nothing to look at", which is the top
                 * STATE refusal by attributed cost -- 9.2% of the interpreted
                 * work in one file sat behind it with nothing to act on. */
                Value oNameEarly =
                    nameIdx < (uint32_t)fn->chunk.constants.count
                        ? fn->chunk.constants.data[nameIdx] : NULL_VAL;
                const char *oNameChars =
                    IS_STRING(oNameEarly) ? AS_STRING(oNameEarly)->chars : "?";
                if (!IS_OBJ(oseen)) {
                    /* No sample, but the model may still know the TYPE -- an
                     * f-string's result, or an earlier invoke's predicted from
                     * its site's feedback. A probe of that type answers the
                     * method lookup just as well, exactly as the SLOT_LIST arm
                     * below builds an empty list for the same reason. Only
                     * strings are probed: they are the types this tier
                     * currently learns without a Value, and a probe has to be
                     * something cheap and permanent to make.
                     *
                     * The object type is guarded at run time below either way,
                     * so the probe chooses a guard and never deletes one. */
                    if (e->stackObjType[ridx] == (uint8_t)(OBJ_STRING + 1)) {
                        ObjString *empty = jaiStringIntern("", 0);
                        if (empty == NULL) return false;
                        oseen = OBJ_VAL((Obj *)empty);
                        oProbe = true;
                    } else {
                        /* Naming the method is what priced this: the census
                         * showed `.len()` and nothing else, and the attribution
                         * instrument put **14.27% of lexer.jai's interpreted
                         * work** in eleven functions behind it.
                         *
                         * An arm for it was built and REVERTED. jaiInvokeByName
                         * resolves against any receiver, and `len` returns an
                         * int whatever it is called on, so the call compiles --
                         * but the bodies still do not. `_fuse_at` moves to the
                         * very next instruction, `OP_GET_INDEX: the container
                         * has kind object, not list`, whose own cause is a
                         * FIELD whose kind only the declaration knows. Measured
                         * flat on the clock.
                         *
                         * THE CHAIN, measured link by link, because that is the
                         * only useful thing to leave behind here:
                         *
                         *   1. this refusal -- armed, `_fuse_at` moves on;
                         *   2. `OP_GET_INDEX: the container has kind object,
                         *      not list` -- a declared `list[T]` FIELD, armed
                         *      below in declaredScalarFieldKind, worth +5
                         *      compiled bodies and still flat;
                         *   3. the same message again, and this time it is not
                         *      a field at all: `w` is the result of a CALL
                         *      whose return kind is predicted SLOT_OBJ rather
                         *      than SLOT_LIST.
                         *
                         * Both of the first two were built and measured
                         * TOGETHER -- 0.31s and 0.61s either way on lexer.jai
                         * and parser.jai, four samples, twice. Clearing two
                         * links of a three-link chain buys nothing, which is
                         * the rule this tier keeps teaching. Start at link 3,
                         * the call return kind, or not at all. */
                        return subWhy(e, "`.%s()` on an object with no sample "
                                      "and no known type", oNameChars);
                    }
                }
                if (nameIdx >= (uint32_t)fn->chunk.constants.count) return false;
                Value oname = fn->chunk.constants.data[nameIdx];
                if (!IS_STRING(oname)) return false;
                Value obound;
                /* Rooted across the lookup: resolving allocates the bound
                 * wrapper, and a probe interned a moment ago is otherwise
                 * unreachable. Only the native is kept, and that outlives it. */
                if (oProbe) jaiGCPushRoot(oseen);
                bool oFound = jaiBuiltinMethod(oseen, AS_STRING(oname), &obound);
                if (oProbe) jaiGCPopRoot();
                if (!oFound) {
                    /* Names the pair, since which builtin is missing is the
                     * whole question and the bare reason only says that one
                     * was. */
                    return subWhy(e, "`%s.%s` is not a builtin of the observed "
                                     "receiver's type",
                                  jaiTypeNameStatic(oseen),
                                  AS_STRING(oname)->chars);
                }
                Value onative = IS_BOUND(obound) ? AS_BOUND(obound)->method
                                                 : obound;
                if (!IS_NATIVE(onative)) {
                    /* A module member written in Jaithon lands here: `math.sqrt`
                     * is a CLOSURE wrapping __prim__.f64_sqrt, not a builtin,
                     * and so is every other one. See emitModuleCall -- it is a
                     * global call through another module, not a method call.
                     *
                     * NOT a bound one. `onative` above has already thrown the
                     * receiver away, which is right for the builtin arm below
                     * (callNativeAt reads args[0], and the receiver is already
                     * in the callee slot) and wrong here, because emitModuleCall
                     * DROPS the receiver as well: nothing is left to be `self`.
                     * A module member whose value is a bound method --
                     * `pub let handler = obj.method` -- then called the unbound
                     * closure with the first argument where `self` belongs, and
                     * `self.base` raised "'fn' object has no attribute 'base'"
                     * on code the interpreter answers correctly. Declining is
                     * enough: the shape is rare and the descriptor would have to
                     * carry `obound`, not `onative`, to do better. */
                    if (IS_MODULE(oseen) && !IS_BOUND(obound) &&
                        IS_CLOSURE(onative) && jitModuleCalls()) {
                        if (!emitModuleCall(e, AS_MODULE(oseen), onative, ridx,
                                            argc, (uint32_t)(off + 7))) {
                            return false;
                        }
                        if (getenv("JAI_JIT_WHY")) {
                            fprintf(stderr, "[jit] module call %s.%s at %d\n",
                                    AS_MODULE(oseen)->name != NULL
                                        ? AS_MODULE(oseen)->name->chars : "?",
                                    AS_STRING(oname)->chars, off);
                        }
                        off += 7;
                        break;
                    }
                    /* Named, because bare this was invisible: the census only
                     * ever said "OP_INVOKE", and p27_float_math cost a night to
                     * trace to `math.sqrt` being an ordinary function. */
                    return subWhy(e, "%s.%s is a %s, not a builtin",
                                  jaiTypeNameStatic(oseen),
                                  AS_STRING(oname)->chars,
                                  jaiTypeNameStatic(onative));
                }

                /* A builtin that only reads a field of its receiver needs
                 * neither the feedback nor the call: the type guard below is
                 * already the whole precondition, and the table states what
                 * comes back. See jit_field_read.h. */
                const JaiJitFieldRead *ofr = jaiJitFieldReadFor(
                    OBJ_TYPE(oseen), AS_STRING(oname)->chars,
                    AS_STRING(oname)->length, argc);

                bool odiscarded = (off + 7 < count && code[off + 7] == OP_POP);
                SlotKind orkind = SLOT_INT;
                unsigned owantTag = VAL_INT;
                uint8_t  orObjType = 0;
                if (ofr == NULL && !odiscarded) {
                    uint8_t fb = jaiInvokeResultFeedback(
                        &fn->chunk, jaiReadU16(code + off + 5), oseen);
                    if (!feedbackSlotKind(fb, &orkind, &owantTag, &orObjType)) {
                        /* Which of the three it is decides what to do about
                         * it: nothing recorded means the window never opened,
                         * mixed means the site really does return two things,
                         * and a named type means the tier has no slot for it. */
                        return subWhy(e, "`%s.%s` has no usable result kind (%s)",
                                      jaiTypeNameStatic(oseen),
                                      AS_STRING(oname)->chars,
                                      jaiFeedbackName(fb));
                    }
                }

                emit(e, jaiA64LdrW(JIT_SCRATCH_A, pushReg(e) - argc - 1,
                                   (unsigned)offsetof(Obj, type)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A,
                                       (unsigned)OBJ_TYPE(oseen)));
                branchOnDeopt(e, JAI_A64_NE);

                if (ofr != NULL) {
                    if (!emitFieldRead(e, ofr, onative, ridx, argc,
                                       (uint32_t)(off + 7))) {
                        return false;
                    }
                    off += 7;   /* a discarded result is the next OP_POP's */
                    break;
                }

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
                /* The feedback named the object type, and nothing else will:
                 * the result does not exist until run time, so there is no
                 * sample. `s.lower().len()` chains two invokes and the second
                 * declined the loop for want of exactly this. */
                if (orkind == SLOT_OBJ) {
                    e->stackObjType[e->depth - 1] = orObjType;
                }
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

            if (rk != SLOT_LIST) {
                /* The last arm. Naming the kind is what turns a bare
                 * "OP_INVOKE" in the census into something actionable: every
                 * receiver kind the arms above do not take lands here, and a
                 * class is the biggest of them -- `Klass.static_method(x)` is
                 * an invoke on a SLOT_CLASS receiver, not the field read plus
                 * call it looks like, and docs/probes/p33_static_call.jai runs
                 * 72,434,801 interpreted instructions on account of it. */
                return subWhy(e, "a receiver of kind %s", slotKindName(rk));
            }
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

            /* What comes back: a field-reading builtin says so itself (`len`
             * is the list's count); anything whose result is dropped on the
             * next instruction needs no kind at all.
             *
             * Nothing else -- and the stated reason for that, "a wrong guess
             * has nowhere to go", is no longer true. The SLOT_OBJ arm below
             * predicts from InlineCache::resultKind and resumes AFTER the call
             * with the result read out of the descriptor
             * (branchOnDeoptAt's `lastFromDesc`), which is exactly the
             * somewhere. This arm never got the same treatment, so every list
             * method that is neither a field read nor discarded declines the
             * whole body: `xs.enumerate()` costs
             * docs/probes/p25_enumerate.jai its 12,800,326 interpreted
             * instructions here.
             *
             * Worth doing only for the methods that are CHEAP next to the loop
             * around them -- `contains`, `index`, `count` on a short list.
             * `enumerate`, `map`, `filter` and `sorted` all allocate a list per
             * call, and rows for that shape were built and measured at zero
             * (docs/research/FALSIFIED-list-returning-builtins.md). */
            bool discarded = (off + 7 < count && code[off + 7] == OP_POP);
            const JaiJitFieldRead *lfr = jaiJitFieldReadFor(
                OBJ_LIST, AS_STRING(nameVal)->chars, AS_STRING(nameVal)->length,
                argc);
            SlotKind lrkind = SLOT_INT;
            unsigned lrtag = VAL_INT;
            uint8_t lrObjType = 0;
            if (lfr == NULL && !discarded) {
                /* Predicted from the site's own feedback and guarded after the
                 * call, exactly as the SLOT_OBJ arm below does it. */
                if (!jitListResult()) {
                    return subWhy(e, "`list.%s` is neither a field read nor "
                                     "discarded", AS_STRING(nameVal)->chars);
                }
                uint8_t lfb = jaiInvokeResultFeedback(
                    &fn->chunk, jaiReadU16(code + off + 5), probe);
                if (!feedbackSlotKind(lfb, &lrkind, &lrtag, &lrObjType)) {
                    return subWhy(e, "`list.%s` has no usable result kind (%s)",
                                  AS_STRING(nameVal)->chars,
                                  jaiFeedbackName(lfb));
                }
            }

            /* `xs.len()` on a list is one field read. Through the descriptor it meant a GC root push/pop, a
             * bound-method resolve, an arity check and a native call just to read a 32-bit count -- paid every iteration of the ordinary `while i < xs.len()` loop. SLOT_LIST is the type guard, already made. */
            if (lfr != NULL) {
                if (!emitFieldRead(e, lfr, nativeVal, ridx, argc,
                                   (uint32_t)(off + 7))) {
                    return false;
                }
                off += 7;   /* a discarded result is the next OP_POP's */
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
            e->wroteHeap = true;
            if (discarded) {
                off += 8;      /* the OP_POP this consumed */
                break;
            }
            if (!pushValue(e, lrkind, 0, NULL)) return false;
            if (lrkind == SLOT_OBJ) e->stackObjType[e->depth - 1] = lrObjType;
            unsigned lrat = e->descOffset +
                            (unsigned)offsetof(JitCallDesc, result);
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, 31, lrat));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, lrtag));
            branchOnDeoptAt(e, JAI_A64_NE, (uint32_t)(off + 7), true);
            /* One byte for a bool: BOOL_VAL writes only the union's `boolean`
             * member, and every SLOT_BOOL consumer tests the whole word. */
            if (lrkind == SLOT_BOOL) {
                emit(e, jaiA64LdrByte(pushReg(e) - 1, 31, lrat + 8));
            } else {
                emit(e, jaiA64LdrX(pushReg(e) - 1, 31, lrat + 8));
            }
            off += 7;
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
                    sample = jaiListGet(AS_LIST(srcv), 0);
                }
                /* A list this body built has no live sample to read an element
                 * off -- it does not exist yet -- but OP_BUILD_LIST recorded
                 * what went into it. `[expr for x in [a, b, c]]` is the shape
                 * that wanted this: the source list is a literal built one
                 * instruction earlier, and without it every comprehension over
                 * one ran interpreted. */
                if (IS_NULL(sample)) sample = e->stackElem[e->depth - 1];
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
                return subWhy(e, "loop variable in local %u has kind %s, "
                                 "not int", slot,
                              slotKindName(e->localKind[slot]));
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

        case OP_GET_ITER_ITEMS: {
            /* `for (a, b) in X.items()`. The emitter cannot know X's type, so
             * it plants this ahead of an ordinary `INVOKE items; GET_ITER` and
             * lets the opcode jump over the pair when X turns out to be a dict,
             * building a lazy ITER_DICT_ITEMS instead. Unarmed, it declined the
             * whole enclosing function -- which is why dict_iter's two loops
             * ran interpreted end to end.
             *
             * The dict case is specialised and the branch is resolved HERE, by
             * walking on at the target rather than by emitting a jump: the
             * skipped `INVOKE items` never executes in this form, so its inline
             * cache is empty and compiling it as dead code would decline. That
             * is only sound because the region really is the emitter's own, so
             * nothing branches into it -- checked below, and backstopped by the
             * fixup resolver, which declines a branch to an offset the walk
             * never reached rather than mis-resolving it.
             *
             * Specialising rather than falling through to the eager `items()`
             * is required, not merely faster: the lazy view raises when the
             * dict changes under the loop and the materialised list does not,
             * so a compiled body that took the other path would answer
             * differently from the interpreter. */
            if (e->depth == 0) return false;
            unsigned sidx = e->depth - 1;
            if (e->stack[sidx] != SLOT_OBJ || !IS_DICT(e->stackSeen[sidx])) {
                e->whyNot = "items() on something that is not a dict";
                return false;
            }
            if (!e->callsOut) return false;
            /* Read before the entry is popped below, not through the model
             * afterwards: the push that replaces it overwrites this cell. */
            Value    itemsDict = e->stackSeen[sidx];
            int16_t  ijump = jaiReadI16(code + off + 1);
            int32_t  after = (int32_t)(off + 3) + ijump;
            /* The emitter's shape exactly: OP_INVOKE (7 bytes) then
             * OP_GET_ITER (1), and the head that follows must be the pair form,
             * since that is the only one shape 4 has an arm for. */
            if (after != off + 11 || after >= stop ||
                code[off + 3] != OP_INVOKE || code[off + 10] != OP_GET_ITER ||
                code[after] != OP_FOR_ITER_PAIR) {
                e->whyNot = "an items() head this tier does not recognise";
                return false;
            }

            emit(e, jaiA64LdrW(JIT_SCRATCH_A, pushReg(e) - 1,
                               (unsigned)offsetof(Obj, type)));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_DICT));
            branchOnDeopt(e, JAI_A64_NE);

            if (!emitDescriptor(e, NULL_VAL, sidx, 1,
                                (void *)&jitMakeItemsIter)) {
                return false;
            }
            unsigned rdrop;
            if (!popValue(e, &rdrop, NULL)) return false;
            /* Shape 4 is an ITER_DICT_ITEMS, and it carries the DICT as its
             * sample rather than an element: the pair head reads the first live
             * entry off it for the component kinds, exactly as the list form
             * reads items[0]. */
            if (!pushValue3(e, SLOT_ITER, 4, NULL, itemsDict, -1)) {
                return false;
            }
            emit(e, jaiA64LdrX(pushReg(e) - 1, 31,
                               e->descOffset +
                                   (unsigned)offsetof(JitCallDesc, result) + 8));
            e->wroteHeap = true;
            off = after;
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
                /* Shape 4 is a dict-items view, and the arm below would read
                 * ObjList's offsets out of an ObjDict were the kind guard not
                 * there to stop it. OP_GET_ITER_ITEMS only makes one when a
                 * pair head follows, so this is unreachable -- kept because the
                 * fact lives in another arm and a decline is the cheap side. */
                if (iterShape == 4) {
                    e->whyNot = "a non-destructuring loop over dict items";
                    return false;
                }
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
                    else if (IS_BOOL(sample))  { ek = SLOT_BOOL;  etag = VAL_BOOL; }
                    else if (IS_LIST(sample))  { ek = SLOT_LIST;  etag = VAL_OBJ; }
                    else if (rawObjValue(sample)) { ek = SLOT_OBJ; etag = VAL_OBJ; }
                    else if (IS_INSTANCE(sample) && AS_INSTANCE(sample)->klass) {
                        ek = SLOT_INST; etag = VAL_OBJ;
                        ecl = AS_INSTANCE(sample)->klass;
                        esh = ecl->shapeId;
                    } else { e->whyNot = "element kind unknown"; return false; }

                    if (!adoptLocalKindSeen(e, fslot, ek, esh, ecl, sample)) {
                        return subWhy(e, "loop variable in local %u has kind "
                                         "%s, not %s", fslot,
                                      slotKindName(e->localKind[fslot]),
                                      slotKindName(ek));
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
                    /* Nothing names this list -- it is whatever the iterator
                     * was built over -- so the storage is proved rather than
                     * pinned. */
                    emitListBoxedGuard(e, JIT_SCRATCH_C, JIT_SCRATCH_A);

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
                    } else if (ek == SLOT_LIST) {
                        /* Same contract as OP_GET_INDEX's own SLOT_LIST arm:
                         * VAL_OBJ is every heap object, not specifically a
                         * list, so the object type is confirmed here, once,
                         * before a SLOT_LIST consumer trusts it with no check
                         * of its own. JIT_SCRATCH_A still holds the index and
                         * must survive to the store below. */
                        emit(e, jaiA64LdrX(JIT_SCRATCH_B, JIT_SCRATCH_C, 8));
                        emit(e, jaiA64LdrW(JIT_SCRATCH_D, JIT_SCRATCH_B,
                                           (unsigned)offsetof(Obj, type)));
                        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_D, OBJ_LIST));
                        branchOnDeopt(e, JAI_A64_NE);
                    }

                    /* Past the last guard: advance, then bind. The advance goes
                     * first because localOut may use JIT_SCRATCH_C/D for the
                     * tag and the index has to be stored out of a register the
                     * write cannot touch. One byte for a bool: see the note in
                     * OP_GET_INDEX -- `strb` is what BOOL_VAL compiles to, so
                     * the rest of the payload word is stale. */
                    emit(e, jaiA64AddXImm(JIT_SCRATCH_B, JIT_SCRATCH_A, 1));
                    emit(e, jaiA64StrX(JIT_SCRATCH_B, rIt,
                                       (unsigned)offsetof(ObjIter, index)));
                    if (ek == SLOT_BOOL) {
                        emit(e, jaiA64LdrByte(JIT_SCRATCH_A, JIT_SCRATCH_C, 8));
                    } else {
                        emit(e, jaiA64LdrX(JIT_SCRATCH_A, JIT_SCRATCH_C, 8));
                    }
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
                else if (IS_LIST(sample))  { ek = SLOT_LIST;  etag = VAL_OBJ; }
                else if (rawObjValue(sample)) { ek = SLOT_OBJ; etag = VAL_OBJ; }
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
                /* The list holds more than one class, so the loop variable is
                 * an instance of no particular one. Two classes are not two
                 * KINDS -- the representation is the same untagged pointer --
                 * so the slot widens rather than the compile failing, and the
                 * call sites inside dispatch by name. Only sound at THIS
                 * instruction: an OSR body is walked from its loop head, so
                 * nothing has been emitted against the class being dropped. */
                if (ek == SLOT_INST && e->elemMixed) {
                    esh = 0;
                    ecl = NULL;
                    e->localTyped[slot] = false;
                }
                if (!adoptLocalKindSeen(e, slot, ek, esh, ecl, sample)) {
                    return subWhy(e, "loop variable in local %u has kind %s, "
                                     "not %s", slot,
                                  slotKindName(e->localKind[slot]),
                                  slotKindName(ek));
                }
                e->iterSlot = slot;
                e->iterExit = (uint32_t)((int32_t)(off + 5) + jump);

                /* Mutation first: a list that grew or shrank under the loop
                 * must raise, and the version is the only thing that says so.
                 * Nothing has happened yet, so this resumes at this very
                 * instruction and the interpreter raises it properly. */
                /* Storage is pinned per form, not checked here: jaiJitEnterOsr
                 * matches JaiOsrForm::iterStg against the list this head is
                 * about to walk, so by the time the body runs the stride below
                 * is already the right one. */
                /* Unpinned is unproved, and the form records LIST_STG_ANY
                 * for the head -- but a deopt guard here is not the answer.
                 * `e->elemStgPin` is false for any body that calls out, and a
                 * `push` is a call, so `for x in xs { out.push(f(x)) }` over a
                 * `list[int]` would fail that guard on its FIRST element and
                 * on every entry after: 11x slower than boxed, and the head's
                 * give-up counter never fires because a bail is not a decline.
                 * So the head dispatches like every other site. */
                ListAccess iAcc;
                iAcc.stg = e->elemStgPin ? e->elemStg
                                         : (uint8_t)LIST_STORE_BOXED;
                iAcc.alt = e->elemStgPin ? iAcc.stg : listAltFor(ek);
                iAcc.dynamic = !e->elemStgPin && iAcc.alt != LIST_STORE_BOXED;
                uint8_t iStg = iAcc.stg;
                if (iStg != LIST_STORE_BOXED && ek != listStgKind(iStg)) {
                    return subWhy(e, "element kind %d is not storage %u's",
                                  (int)ek, iStg);
                }

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
                int iSkip = listDispatchBegin(e, &iAcc, JIT_START_REG,
                                              JIT_SCRATCH_A);
                emit(e, jaiA64LdrX(JIT_SCRATCH_C, JIT_START_REG,
                                   (unsigned)offsetof(ObjList, items)));
                emit(e, jaiA64AddXLsl(JIT_SCRATCH_C, JIT_SCRATCH_C,
                                      JIT_IDX_REG, listStgShift(iStg)));

                /* Nothing to check on an unboxed element: no tag, and no
                 * object behind it whose type could surprise the arms below. */
                if (iStg == LIST_STORE_BOXED) {
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
                    /* The class is checked only when one was pinned. A widened
                     * slot has no class to check against, and that it is an
                     * instance at all -- which the guard above settles -- is
                     * everything a by-name call needs of it. */
                    if (esh != 0) {
                        emit(e, jaiA64LdrX(JIT_SCRATCH_D, JIT_SCRATCH_D,
                                           (unsigned)offsetof(ObjInstance, klass)));
                        emit(e, jaiA64LdrW(JIT_SCRATCH_D, JIT_SCRATCH_D,
                                           (unsigned)offsetof(ObjClass, shapeId)));
                        emitConst64(e, JIT_SCRATCH_A, (int64_t)esh);
                        emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_D, JIT_SCRATCH_A));
                        branchOnDeopt(e, JAI_A64_NE);
                    }
                } else if (ek == SLOT_LIST) {
                    /* Same contract as OP_GET_INDEX's own SLOT_LIST arm: VAL_OBJ
                     * is every heap object, not specifically a list, so the
                     * object type is confirmed here, once, before a SLOT_LIST
                     * consumer trusts it with no check of its own. */
                    emit(e, jaiA64LdrX(JIT_SCRATCH_D, JIT_SCRATCH_C, 8));
                    emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_D,
                                       (unsigned)offsetof(Obj, type)));
                    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_LIST));
                    branchOnDeopt(e, JAI_A64_NE);
                }
                }

                /* Both arms leave JIT_SCRATCH_C on the payload, as
                 * OP_GET_INDEX's do, so the load below serves either. */
                if (iStg == LIST_STORE_BOXED) {
                    emit(e, jaiA64AddXImm(JIT_SCRATCH_C, JIT_SCRATCH_C, 8));
                }
                if (iSkip >= 0) {
                    int iJoin = listDispatchElse(e, iSkip);
                    emit(e, jaiA64LdrX(JIT_SCRATCH_C, JIT_START_REG,
                                       (unsigned)offsetof(ObjList, items)));
                    emit(e, jaiA64AddXLsl(JIT_SCRATCH_C, JIT_SCRATCH_C,
                                          JIT_IDX_REG,
                                          listStgShift(iAcc.alt)));
                    listDispatchEnd(e, iJoin);
                }

                /* One byte for a bool: see the note in OP_GET_INDEX. `strb` is
                 * what BOOL_VAL compiles to, so the rest of the payload word is
                 * stale, and a SLOT_BOOL register must hold 0 or 1. */
                unsigned eAt = 0;
                if (ek == SLOT_BOOL) {
                    emit(e, jaiA64LdrByte(JIT_SCRATCH_A, JIT_SCRATCH_C, eAt));
                } else {
                    emit(e, jaiA64LdrX(JIT_SCRATCH_A, JIT_SCRATCH_C, eAt));
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
            /* As the head of an OSR loop the iterator is not on the modelled
             * operand stack at all -- it arrives in a reserved register and
             * stays on the interpreter's stack, which is what lets an exit
             * leave without unwinding anything. Only iterKind 3 gets here: a
             * range or list head is an OP_FOR_ITER_BIND. */
            bool pairHead = e->osr && e->hasIter && e->iterKind == 3 &&
                            (uint32_t)off == e->osrTop;
            if (!pairHead &&
                (e->depth == 0 || e->stack[e->depth - 1] != SLOT_ITER)) {
                return false;
            }
            if (!pairHead && e->stackShape[e->depth - 1] == 0) {
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

            /* Component kinds come from the pair the source was holding when
             * the iterator was built (OP_GET_ITER / OP_GET_ITER_ITEMS carries
             * it forward), and the guards below are what make that a
             * specialisation rather than an assumption. */
            bool pairIsDict = pairHead || e->stackShape[e->depth - 1] == 4;
            Value psample = pairHead ? e->elemSample : e->stackSeen[e->depth - 1];
            SlotKind pk[2];
            unsigned ptag[2];
            Value pseen[2];
            if (pairIsDict) {
                /* Shape 4 carries the dict itself, so the sample is its first
                 * live entry -- the one the loop is about to yield. */
                if (!IS_DICT(psample) ||
                    !firstLiveEntry(&AS_DICT(psample)->table,
                                    &pseen[0], &pseen[1])) {
                    e->whyNot = "iterating a dict with nothing to look at";
                    return false;
                }
            } else {
                if (!IS_TUPLE(psample) || AS_TUPLE(psample)->count != 2) {
                    e->whyNot = "pair element is not a 2-tuple";
                    return false;
                }
                pseen[0] = AS_TUPLE(psample)->items[0];
                pseen[1] = AS_TUPLE(psample)->items[1];
            }
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
                return subWhy(e, "a pair's loop variables (locals %u and %u) "
                                 "have kinds %s and %s", pslotA, pslotB,
                              slotKindName(e->localKind[pslotA]),
                              slotKindName(e->localKind[pslotB]));
            }

            unsigned rIter = pairHead ? JIT_PAIR_ITER_REG : pushReg(e) - 1;
            uint32_t pairExit = (uint32_t)((int32_t)(off + 7) + pjump);
            /* A head's exit leaves the model at the depth it is already at --
             * the iterator it drops was never in the model. Registering it as
             * iterExit is what makes the exit stub tell the interpreter to pop
             * the exhausted iterator off its own stack. */
            int pairExitDepth = pairHead
                                    ? (int)stackSignature(e)
                                    : (int)stackSignatureAt(e, e->depth - 1);
            if (pairHead) e->iterExit = pairExit;

            if (pairIsDict) {
                /* iterStepPairFast's ITER_DICT_ITEMS case plus the jaiTableNext
                 * it calls, inline. Same discipline as the list arm below:
                 * every guard, and the whole scan, runs before the index is
                 * written back, so a deopt -- forced or real -- resumes at this
                 * instruction with the iterator exactly as the interpreter left
                 * it and re-does the scan. */
                _Static_assert(sizeof(JaiEntry) == 48,
                               "the dict-items step scales the order index by "
                               "hand: slot * 16 * 3");
                const unsigned tOff = (unsigned)offsetof(ObjDict, table);

                emit(e, jaiA64LdrW(JIT_SCRATCH_A, rIter,
                                   (unsigned)offsetof(ObjIter, kind)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, ITER_DICT_ITEMS));
                branchOnDeopt(e, JAI_A64_NE);
                emit(e, jaiA64LdrX(JIT_SCRATCH_B, rIter,
                                   (unsigned)offsetof(ObjIter, source) + 8));
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_B,
                                   (unsigned)offsetof(Obj, type)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_DICT));
                branchOnDeopt(e, JAI_A64_NE);

                /* A dict that changed under the loop must raise, and only the
                 * version says so. jaiIterNext owns that message, so the guard
                 * hands the whole instruction back unadvanced. */
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_B,
                                   tOff + (unsigned)offsetof(JaiTable, version)));
                emit(e, jaiA64LdrW(JIT_SCRATCH_C, rIter,
                                   (unsigned)offsetof(ObjIter, version)));
                emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_C));
                branchOnDeopt(e, JAI_A64_NE);

                emit(e, jaiA64LdrX(JIT_SCRATCH_C, rIter,
                                   (unsigned)offsetof(ObjIter, index)));
                emit(e, jaiA64LdrW(JIT_SCRATCH_D, JIT_SCRATCH_B,
                                   tOff +
                                       (unsigned)offsetof(JaiTable, orderCount)));

                /* The scan. `order` holds an entry index per insertion
                 * position, negative where a delete left a hole, so a dict with
                 * deletions in it costs one extra pass per hole and nothing
                 * else. orderCount is hoisted because only a mutation can move
                 * it and the version guard above has already excluded one.
                 *
                 * branchToDepth inside a loop is sound only because it settles
                 * nothing here: the branchOnDeopt three lines up fails the
                 * compile outright if a deferred value is live, so the settle
                 * it performs is a no-op and cannot be re-executed. */
                unsigned scanTop = e->count;
                emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_C, JIT_SCRATCH_D));
                /* The exhausted arm drops the iterator, so the target is
                 * reached one entry shallower than this branch leaves from. */
                branchToDepth(e, pairExit, JAI_A64_GE, pairExitDepth);
                emit(e, jaiA64LdrX(JIT_SCRATCH_A, JIT_SCRATCH_B,
                                   tOff + (unsigned)offsetof(JaiTable, order)));
                emit(e, jaiA64AddXLsl(JIT_SCRATCH_A, JIT_SCRATCH_A,
                                      JIT_SCRATCH_C, 2));
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_A, 0));
                emit(e, jaiA64AddXImm(JIT_SCRATCH_C, JIT_SCRATCH_C, 1));
                /* A hole: the slot is int32 and negative, which after the
                 * zero-extending load is bit 31 set. Measured against e->count
                 * so an instruction added above cannot rot the distance. */
                emit(e, jaiA64Tbnz(JIT_SCRATCH_A, 31,
                                   (int32_t)scanTop - (int32_t)e->count));

                /* entries + slot * sizeof(JaiEntry): slot << 4, then + itself
                 * twice over, which is the 48 the assert above pins. */
                emit(e, jaiA64LdrX(JIT_SCRATCH_D, JIT_SCRATCH_B,
                                   tOff + (unsigned)offsetof(JaiTable, entries)));
                emit(e, jaiA64LslX(JIT_SCRATCH_A, JIT_SCRATCH_A, 4));
                emit(e, jaiA64AddXLsl(JIT_SCRATCH_A, JIT_SCRATCH_A,
                                      JIT_SCRATCH_A, 1));
                emit(e, jaiA64AddX(JIT_SCRATCH_B, JIT_SCRATCH_D, JIT_SCRATCH_A));

                /* Key and value carry the kinds sampled off the first live
                 * entry; a dict that later holds another kind fails here with
                 * nothing written. */
                for (unsigned i = 0; i < 2; i++) {
                    unsigned at = i == 0 ? (unsigned)offsetof(JaiEntry, key)
                                         : (unsigned)offsetof(JaiEntry, value);
                    emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_B, at));
                    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, ptag[i]));
                    branchOnDeopt(e, JAI_A64_NE);
                }

                /* Past the last guard. The index goes first because localOut
                 * spends JIT_SCRATCH_C and JIT_SCRATCH_D on a frame-resident
                 * slot's tag; only JIT_SCRATCH_B survives it. */
                emit(e, jaiA64StrX(JIT_SCRATCH_C, rIter,
                                   (unsigned)offsetof(ObjIter, index)));
                for (unsigned i = 0; i < 2; i++) {
                    unsigned at = (i == 0 ? (unsigned)offsetof(JaiEntry, key)
                                          : (unsigned)offsetof(JaiEntry, value))
                                  + 8u;
                    /* A bool is one byte (see OP_GET_INDEX) -- the rest of its
                     * payload word is stale, and a SLOT_BOOL register must hold
                     * 0 or 1. */
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
            branchToDepth(e, pairExit, JAI_A64_GE, pairExitDepth);

            /* items is reloaded rather than hoisted: a reallocation bumps the
             * version, which the guard above covers, and one ldr removes the
             * question. */
            emitListBoxedGuard(e, JIT_SCRATCH_B, JIT_SCRATCH_A);
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
                unsigned rIdx = pushReg(e) - 1;
                unsigned rStr = valueXReg(e, e->valueDepth - 2);

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
            /* `buf[i]` on a `bytes`: a length-guarded byte load, and the
             * result is a plain int, so nothing is allocated. Every binary
             * format in the language is read one byte at a time through this
             * -- the JPEG bit reader is a `bytes` index and nothing else --
             * and without it the whole function around one declined. */
            if (e->depth >= 2 && e->stack[e->depth - 1] == SLOT_INT &&
                e->stack[e->depth - 2] == SLOT_OBJ &&
                IS_BYTES(e->stackSeen[e->depth - 2])) {
                unsigned rIdx = pushReg(e) - 1;
                unsigned rBuf = valueXReg(e, e->valueDepth - 2);

                /* Really a bytes, and not something else this object slot
                 * happened to hold when the body was compiled. */
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, rBuf,
                                   (unsigned)offsetof(Obj, type)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_BYTES));
                branchOnDeopt(e, JAI_A64_NE);

                emit(e, jaiA64LdrW(JIT_SCRATCH_A, rBuf,
                                   (unsigned)offsetof(ObjBytes, length)));
                emitBoundsNormalise(e, rIdx, JIT_SCRATCH_A, JIT_SCRATCH_B,
                                    false);

                /* The payload is inline after the header, so the base needs no
                 * load of its own -- unlike a string, which holds a pointer. */
                emit(e, jaiA64AddX(JIT_SCRATCH_C, rBuf, JIT_SCRATCH_B));
                emit(e, jaiA64LdrByte(JIT_SCRATCH_A, JIT_SCRATCH_C,
                                      (unsigned)offsetof(ObjBytes, data)));

                unsigned dByte1, dByte2;
                if (!popValue(e, &dByte1, NULL)) return false;
                if (!popValue(e, &dByte2, NULL)) return false;
                if (!pushValue(e, SLOT_INT, 0, NULL)) return false;
                emit(e, jaiA64MovX(pushReg(e) - 1, JIT_SCRATCH_A));
                off += 1;
                break;
            }
            /* Index normalised as jaiNormalizeIndex does it, one unsigned compare covering both ends. Out of
             * range, or an element not the kind seen at compile time, goes back to the interpreter -- reading an element has no effect, so resuming at this instruction is always sound. */
            if (e->depth < 2) return subWhy(e, "the model is only %u deep", e->depth);
            if (e->stack[e->depth - 2] == SLOT_OBJ &&
                IS_DICT(e->stackSeen[e->depth - 2])) {
                /* `d[k]`, the read half of the OP_SET_INDEX dict arm below.
                 * Without it a loop that reads a dict ran interpreted end to
                 * end: `t += d["a"]` two million times was 16,280,472
                 * interpreted instructions and 1,684 once this landed.
                 *
                 * Predicted off a live sample and guarded, as the list arm is,
                 * except that the sample must be UNIFORM across the dict --
                 * see dictUniformValue for why a dict is not a list here. */
                unsigned dsidx = e->depth - 2;
                Value dsample;
                if (!dictUniformValue(AS_DICT(e->stackSeen[dsidx]), &dsample)) {
                    return subWhy(e, "the live dict is empty or holds more than "
                                     "one kind of value");
                }
                SlotKind dkind;
                unsigned dtag;
                ObjClass *dcls;
                uint32_t dshape;
                if (!exemplarKind(dsample, &dkind, &dtag, &dcls, &dshape)) {
                    return subWhy(e, "a dict value of a kind the tier cannot hold");
                }
                /* SLOT_OBJ pins nothing, so the container is proved to be a
                 * dict before anything is consumed: a miss resumes with the
                 * dict and the key both still on the interpreter's stack. */
                emit(e, jaiA64LdrW(JIT_SCRATCH_A,
                                   valueXReg(e, e->valueDepth - 2),
                                   (unsigned)offsetof(Obj, type)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_DICT));
                branchOnDeopt(e, JAI_A64_NE);

                if (!emitDescriptor(e, NULL_VAL, dsidx, 2,
                                    (void *)&jitGetIndexDict)) {
                    return false;
                }
                for (unsigned i = 0; i < 2; i++) {
                    unsigned drop;
                    if (!popValue(e, &drop, NULL)) return false;
                }
                /* The sample travels with the entry, as the list arm's does:
                 * without it `names["first"].len()` is an invoke on an object
                 * the model cannot name, and the body declines one instruction
                 * after the read it just learned to make. */
                if (!pushValue3(e, dkind, dshape, dcls, dsample, -1)) {
                    return false;
                }

                unsigned drat = e->descOffset +
                                (unsigned)offsetof(JitCallDesc, result);
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, 31, drat));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, dtag));
                /* Resumes AFTER the read. The lookup itself is pure, but it may
                 * have raised and been caught, and re-running it would be a
                 * second probe of a table the handler could have changed. */
                branchOnDeoptAt(e, JAI_A64_NE, (uint32_t)(off + 1), true);
                unsigned drd = pushReg(e) - 1;
                if (dkind == SLOT_BOOL) {
                    emit(e, jaiA64LdrByte(drd, 31, drat + 8));
                } else {
                    emit(e, jaiA64LdrX(drd, 31, drat + 8));
                }
                if (dkind == SLOT_INST) {
                    /* Two shapes in one dict cannot be told apart by the tag,
                     * and the walk above only sampled a prefix.
                     *
                     * The object type comes first, for the reason the shared
                     * return path gives: VAL_OBJ covers every heap object, and
                     * reading `klass` off a string lands in its length/hash and
                     * dereferences it. A dict holding a Box under one key and a
                     * str under another SEGFAULTED the VM from ordinary code --
                     * `d[k]` in any body hot enough to compile.
                     *
                     * The tag test above cannot stand in for this: it is the
                     * same test the sampled prefix already passed. */
                    emit(e, jaiA64LdrW(JIT_SCRATCH_A, drd,
                                       (unsigned)offsetof(Obj, type)));
                    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_INSTANCE));
                    branchOnDeoptAt(e, JAI_A64_NE, (uint32_t)(off + 1), true);
                    emit(e, jaiA64LdrX(JIT_SCRATCH_A, drd,
                                       (unsigned)offsetof(ObjInstance, klass)));
                    emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_A,
                                       (unsigned)offsetof(ObjClass, shapeId)));
                    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, dshape));
                    branchOnDeoptAt(e, JAI_A64_NE, (uint32_t)(off + 1), false);
                }
                e->wroteHeap = true;
                off += 1;
                break;
            }
            if (e->stack[e->depth - 2] != SLOT_LIST) {
                return subWhy(e, "the container has kind %s, not list",
                              slotKindName(e->stack[e->depth - 2]));
            }
            if (e->stack[e->depth - 1] != SLOT_INT) {
                return subWhy(e, "the subscript is kind %d, not an int",
                              (int)e->stack[e->depth - 1]);
            }
            unsigned rIdx = pushReg(e) - 1;
            unsigned rList = valueXReg(e, e->valueDepth - 2);
            bool gHoisted = false;

            Value seenList = e->stackSeen[e->depth - 2];
            if (!IS_LIST(seenList)) {
                return subWhy(e, "no live list to read an element kind off");
            }
            ObjList *sl = AS_LIST(seenList);
            if (sl->count <= 0) {
                return subWhy(e, "the live list is empty, so there is no exemplar");
            }
            Value elem = jaiListGet(sl, 0);
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
            else if (rawObjValue(elem)) {
                /* A list of strings, dicts, sets, or tuples, held raw: the same contract as a SLOT_OBJ
                 * global or field (sample specialises, the tag guard below confirms, every consumer
                 * re-checks Obj.type for itself). `str_search` builds text out of `chunks[seed %% 8]` and
                 * declined that whole loop forty times over before the string case alone was admitted;
                 * widened from IS_STRING to rawObjValue so every other raw-holdable element kind gets the
                 * same treatment rather than only strings. */
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
            {
                int32_t gOff = 0;
                uint8_t gBase = 0;
                bool gShaped = boundsCoveredAtHead(e, e->stackLocal[e->depth - 2],
                                                   e->valueDepth - 1, &gOff,
                                                   &gBase);
                noteIndexSpan(e, e->stackLocal[e->depth - 2], gShaped, gOff,
                              gBase);
                gHoisted = gShaped;
            }
            ListAccess gAcc = listAccessFor(e, rList,
                                            e->stackLocal[e->depth - 2],
                                            kind, JIT_SCRATCH_D);
            /* The sampled element and a PINNED storage cannot disagree -- an
             * I64 store holds ints and nothing else -- but the kind is what
             * the loads below are emitted for, so it is checked rather than
             * assumed. A dispatched access picks its second arm from the kind,
             * so it cannot disagree by construction. */
            if (!gAcc.dynamic && gAcc.stg != LIST_STORE_BOXED &&
                kind != listStgKind(gAcc.stg)) {
                return subWhy(e, "element kind %d is not storage %u's",
                              (int)kind, gAcc.stg);
            }
            unsigned gItems = JIT_SCRATCH_C, gCount = JIT_SCRATCH_A;
            int gh = hoistFor(e, e->stackLocal[e->depth - 2]);
            if (gh >= 0) {
                gItems = e->hoist[gh].itemsReg;
                gCount = e->hoist[gh].countReg;
            } else {
                emitListHeader(e, rList, gItems, gCount);
            }
            if (gHoisted) {
                /* The head proved it. Only the normalisation copy is left, and
                 * a shaped index is non-negative by that same proof, so even
                 * that is just a move. */
                emit(e, jaiA64MovX(JIT_SCRATCH_B, rIdx));
            } else {
                emitBoundsNormalise(e, rIdx, gCount, JIT_SCRATCH_B, true);
            }

            /* Both arms below leave JIT_SCRATCH_C on the PAYLOAD rather than
             * on the element, which is what lets one load serve them: a boxed
             * element's payload is eight bytes into it, an unboxed element IS
             * its payload. */
            int gSkip = listDispatchBegin(e, &gAcc, rList, JIT_SCRATCH_D);

            emit(e, jaiA64AddXLsl(JIT_SCRATCH_C, gItems,
                                  JIT_SCRATCH_B, listStgShift(gAcc.stg)));
            /* An unboxed element has no tag to check, and no object behind it
             * to confirm the type of: the storage already said what it is. */
            if (gAcc.stg == LIST_STORE_BOXED) {
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_C, 0));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, tag));
            branchOnDeopt(e, JAI_A64_NE);
            emit(e, jaiA64AddXImm(JIT_SCRATCH_C, JIT_SCRATCH_C, 8));
            if (kind == SLOT_INST) {
                /* The tag says "an object", not "an object of this class" --
                 * and not even "an instance" yet: VAL_OBJ is every heap
                 * object, so a list holding an instance beside a string must
                 * have its object type confirmed before `klass` is read,
                 * exactly as OP_FOR_ITER_BIND's SLOT_INST arms already do.
                 * Without this, a list whose sampled element is an instance
                 * but a later element is (say) a string reads that string's
                 * header bytes as an ObjInstance's `klass` pointer and
                 * segfaults dereferencing it -- not merely a wrong answer. */
                emit(e, jaiA64LdrX(JIT_SCRATCH_D, JIT_SCRATCH_C, 0));
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_D,
                                   (unsigned)offsetof(Obj, type)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_INSTANCE));
                branchOnDeopt(e, JAI_A64_NE);
                emit(e, jaiA64LdrX(JIT_SCRATCH_D, JIT_SCRATCH_D,
                                   (unsigned)offsetof(ObjInstance, klass)));
                emit(e, jaiA64LdrW(JIT_SCRATCH_D, JIT_SCRATCH_D,
                                   (unsigned)offsetof(ObjClass, shapeId)));
                emitConst64(e, JIT_SCRATCH_A, (int64_t)elemShape);
                emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_D, JIT_SCRATCH_A));
                branchOnDeopt(e, JAI_A64_NE);
            } else if (kind == SLOT_LIST) {
                /* "an object" is not "a list": every SLOT_LIST consumer reads the header with no check of
                 * its own, so the object type is confirmed here, once, before the kind is handed out --
                 * same contract, same check, as OP_GET_FIELD_LOCAL's SLOT_LIST arm. Without this a
                 * heterogeneous list (`[[1, 2], "not a list"]`) passes the generic VAL_OBJ tag check on
                 * either element and reads the second one's bytes through ObjList's field offsets. */
                emit(e, jaiA64LdrX(JIT_SCRATCH_D, JIT_SCRATCH_C, 0));
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_D,
                                   (unsigned)offsetof(Obj, type)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_LIST));
                branchOnDeopt(e, JAI_A64_NE);
            }
            }
            if (gSkip >= 0) {
                int gJoin = listDispatchElse(e, gSkip);
                emit(e, jaiA64AddXLsl(JIT_SCRATCH_C, gItems, JIT_SCRATCH_B,
                                      listStgShift(gAcc.alt)));
                listDispatchEnd(e, gJoin);
            }

            unsigned d1, d2;
            if (!popValue(e, &d1, NULL)) return false;
            if (!popValue(e, &d2, NULL)) return false;
            if (!pushValue3(e, kind, elemShape, elemClass, elem, -1)) return false;
            /* Bool payload is one byte (`BOOL_VAL` compiles to `strb`), so the other seven bytes are stale --
             * an 8-byte load would hand a SLOT_BOOL register (required to hold exactly 0 or 1, since every consumer does `cbnz` on the whole word) garbage. */
            if (kind == SLOT_BOOL) {
                emit(e, jaiA64LdrByte(pushReg(e) - 1, JIT_SCRATCH_C, 0));
            } else if (kind == SLOT_FLOAT &&
                       fpWorthLoading(e, code, off + 1, stop)) {
                /* Straight into the FP bank, for the same reason a float local
                 * goes there: `ldr x` followed by `fmov d, x` puts a
                 * cross-register-file move between the load and the multiply
                 * that wants it, and `ai[k] * b[k][j]` had two of them. */
                unsigned idx = e->valueDepth - 1;
                emit(e, jaiA64LdrD(fpRegAt(e, idx), JIT_SCRATCH_C, 0));
                fpClaim(e, idx);
            } else {
                emit(e, jaiA64LdrX(pushReg(e) - 1, JIT_SCRATCH_C, 0));
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
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, valueXReg(e, e->valueDepth - 3),
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
            unsigned rIdx = valueXReg(e, e->valueDepth - 2);
            unsigned rList = valueXReg(e, e->valueDepth - 3);

            noteSlotIndexed(e, e->stackLocal[e->depth - 3]);
            bool sHoisted;
            {
                int32_t sOff = 0;
                uint8_t sBase = 0;
                sHoisted = boundsCoveredAtHead(e, e->stackLocal[e->depth - 3],
                                               e->valueDepth - 2, &sOff,
                                               &sBase);
                noteIndexSpan(e, e->stackLocal[e->depth - 3], sHoisted, sOff,
                              sBase);
            }
            ListAccess sAcc = listAccessFor(e, rList, e->stackLocal[e->depth - 3],
                                            vk, JIT_SCRATCH_D);
            /* Exactly what jaiListStoreAccepts allows, and for its reason: an
             * int written into a `list[float]` de-specialises the list in the
             * interpreter, which is not something this can do inline. The
             * dispatched form cannot hit it -- its second arm is the storage
             * that holds a `vk` and no other. */
            if (!sAcc.dynamic && sAcc.stg != LIST_STORE_BOXED &&
                vk != listStgKind(sAcc.stg)) {
                return subWhy(e, "storing kind %d into storage %u",
                              (int)vk, sAcc.stg);
            }
            unsigned sItems = JIT_SCRATCH_C, sCount = JIT_SCRATCH_A;
            int sh = hoistFor(e, e->stackLocal[e->depth - 3]);
            if (sh >= 0) {
                sItems = e->hoist[sh].itemsReg;
                sCount = e->hoist[sh].countReg;
            } else {
                emitListHeader(e, rList, sItems, sCount);
            }
            if (sHoisted) {
                emit(e, jaiA64MovX(JIT_SCRATCH_B, rIdx));
            } else {
                emitBoundsNormalise(e, rIdx, sCount, JIT_SCRATCH_B, true);
            }

            int sSkip = listDispatchBegin(e, &sAcc, rList, JIT_SCRATCH_A);
            emitElemStoreAt(e, sAcc.stg, sItems, JIT_SCRATCH_B, vtag, rVal);
            if (sSkip >= 0) {
                int sJoin = listDispatchElse(e, sSkip);
                emitElemStoreAt(e, sAcc.alt, sItems, JIT_SCRATCH_B, vtag, rVal);
                listDispatchEnd(e, sJoin);
            }
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
            } else if (e->stack[cidx] == SLOT_OBJ && IS_TUPLE(cseen)) {
                /* `jitGetSlice` is a thin wrapper over `jaiSliceGet`, which
                 * already handles a tuple container exactly like a list or a
                 * string -- only this arm's own type guard was narrower than
                 * what the call it makes actually supports. */
                cType = OBJ_TUPLE; sliceKind = SLOT_OBJ;
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
                e->stackAscii[e->depth] = false;
                e->stackNullLit[e->depth] = false;
                e->stackUnit[e->depth]  = false;
                e->stack[e->depth++]    = SLOT_CLASS;
                off += 6;
                break;
            }
            /* A plain function, resolved now and pinned by the same
             * module-version check. Only one that has itself compiled -- and
             * the reason is NOT soundness, which is what this comment used to
             * say and what two separate investigations went looking for.
             *
             * The miscompile it used to warn about is real and is FIXED. It was
             * never a property of the callee: a descriptor `result` holding a
             * bool was loaded eight bytes wide where BOOL_VAL writes one, so
             * `false` came back with dirty high bytes and read as true. That is
             * why the corruption was one-directional and looked deterministic
             * in a lexer -- delta-debugging 63 admitted callees reduced it to
             * `_is_alpha` alone. Both sites are closed: the ordinary return in
             * 5036099, the verdict-4 slow stub in emitSelfSlowStubs above.
             * docs/agents/uncompiled-callee.md has the reduction.
             *
             * With both fixed, dropping the check is SOUND -- 1274 green plain,
             * under JAITHON_NO_JIT, deopt stress and split stress -- and it
             * does let the mutually recursive groups in: json_parse's `value`,
             * `object`, `array` and `integer` all reach the tier.
             *
             * It is kept because removing it does not pay, which nobody had
             * measured. Net on `check --no-cache parser.jai`, stable across
             * three runs each: 284 function-tier bodies with the check, 275
             * without -- five gained, fourteen lost. An independent measurement
             * at a different commit found the same direction (265 to 258, seven
             * gained and thirteen lost). json_parse's own wall clock does not
             * move (0.05s either way) and neither does the compiler's.
             *
             * TWO things make it come out that way, and the second is the one
             * that is easy to miss. Declining here falls into emitUnarmedDeopt
             * just below, which interprets from this instruction ONWARD rather
             * than giving the whole body up; admitting the callee skips that
             * escape hatch and the body walks on to a refusal that costs all of
             * it -- `value` itself then stops at "callee's return kind not
             * usable". AND: admitting callees earlier changes how much
             * interpreted work happens before OTHER, unrelated functions cross
             * their own call-count hotness threshold in a fixed workload, so
             * some fall short of a threshold they used to clear and others
             * clear one they used to miss. The name sets differ in both
             * directions for that reason, and neither count is a general
             * verdict about the tier.
             *
             * So: removable, and not worth removing. If the unarmed-deopt path
             * ever stops being the better half of that trade, this is one line.
             */
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
                        /* The register the push BELOW will land in, named
                         * ahead of it because the guards have to run against
                         * the model as it is now. `pushReg` is one past the
                         * CURRENT top, which is the same register only while
                         * the bank is one contiguous run -- at a split
                         * boundary it is the last callee-saved one and the
                         * push goes to x0. That mismatch loaded a global into
                         * a register nothing then read, and bitops printed
                         * 68720029766 for 999625. */
                        unsigned dst = valueXReg(e, e->valueDepth);
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
                    /* Interpreted from here rather than the whole function
                     * declined. The case that matters is a global read on a
                     * branch that never runs: `throw ValueError(...)` in an
                     * argument check is the shape, there are hundreds of them
                     * in lib/std alone, and declining for one cost 110 ms
                     * against 14 ms for the same function guarded by a plain
                     * `return`. The unarmed path already exists for exactly
                     * this -- see emitUnarmedDeopt. */
                    if (!e->osr && emitUnarmedDeopt(e, &fn->chunk, &off, stop)) {
                        break;
                    }
                    if (e->failed) return false;
                    /* Named, because this is the top state-level refusal by
                     * attempt count -- 406 in one file, 82 of them retries of
                     * a single loop -- and "callee is not a compiled global
                     * function" gave the reader nothing to act on. Which
                     * callee it is decides whether this is a tier-ordering
                     * problem that resolves itself or a function that never
                     * compiles at all. */
                    Value gNameVal = nameIdx < (uint32_t)fn->chunk.constants.count
                                         ? fn->chunk.constants.data[nameIdx]
                                         : NULL_VAL;
                    const char *gName = IS_STRING(gNameVal)
                                            ? AS_STRING(gNameVal)->chars
                                            : "?";
                    if (gslot != NULL && !IS_CLOSURE(gvv) && !IS_CLASS(gvv) &&
                        !IS_NATIVE(gvv)) {
                        return subWhy(e, "`%s` is a global of a kind the tier "
                                      "has no slot for", gName);
                    }
                    return subWhy(e, "`%s` is not a compiled global function",
                                  gName);
                }
                if (e->depth >= JIT_MAX_STACK) return false;
                e->stackShape[e->depth] = 0;
                e->stackClass[e->depth] = (ObjClass *)(void *)AS_OBJ(nv);
                e->stackSeen[e->depth]  = nv;
                e->stackLocal[e->depth] = -1;
                e->stackAscii[e->depth] = false;
                e->stackNullLit[e->depth] = false;
                e->stackUnit[e->depth]  = false;
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
            e->stackAscii[e->depth] = false;
            e->stackNullLit[e->depth] = false;
            e->stackUnit[e->depth]  = false;
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
                if (argc != 1 && argc != 2) {
                    e->whyNot = "builtin arity"; return false;
                }
                Value cv = e->stackSeen[e->depth - argc - 1];
                ObjNative *nat = IS_NATIVE(cv) ? AS_NATIVE(cv) : NULL;
                const char *nm = nat != NULL && nat->name != NULL
                                     ? nat->name->chars : "";
                /* `min`/`max` of two ints are a compare and a csel, and image
                 * code reaches for them per pixel -- the JPEG block reader
                 * calls `min` twice a sample, so declining them left the whole
                 * body interpreted. Only the two-argument integer form is
                 * emitted: the float form throws on a NaN operand (see
                 * compareDoubles), which a bare fcsel would silently swallow,
                 * and the one-argument form takes an iterable. */
                bool isMin = strcmp(nm, "min") == 0;
                bool isMax = strcmp(nm, "max") == 0;
                if (argc == 2) {
                    if (!isMin && !isMax) {
                        int done = emitNativeResultCall(e, cv, nm, argc,
                                                        (uint32_t)(off + 2));
                        if (done < 0) return false;
                        if (done > 0) { off += 2; break; }
                        return subWhy(e, "`%s` has no known result kind", nm);
                    }
                    if (e->stack[e->depth - 1] != SLOT_INT ||
                        e->stack[e->depth - 2] != SLOT_INT) {
                        e->whyNot = "a builtin with no known result kind";
                        return false;
                    }
                    unsigned rb, ra2;
                    if (!popValue(e, &rb, NULL)) return false;
                    if (!popValue(e, &ra2, NULL)) return false;
                    if (!pushValue(e, SLOT_INT, 0, NULL)) return false;
                    unsigned rd = pushReg(e) - 1;
                    /* `min(a, b)` keeps `a` unless `b` is STRICTLY smaller
                     * (extremum() only replaces on a nonzero order), so the
                     * compare is of b against a and the condition is strict. */
                    emit(e, jaiA64SubsX(JAI_A64_XZR, rb, ra2));
                    emit(e, jaiA64CselX(rd, rb, ra2,
                                        isMax ? JAI_A64_GT : JAI_A64_LT));
                    dropCalleeEntry(e);
                    off += 2;
                    break;
                }
                SlotKind ak = e->stack[e->depth - 1];
                unsigned ar = pushReg(e) - 1;
                if (strcmp(nm, "abs") == 0 && ak == SLOT_INT) {
                    /* `subs` off zero both negates and reports the one input
                     * abs() rejects: only INT64_MIN overflows the negation, and
                     * the interpreter raises the OverflowError on re-entry. */
                    emit(e, jaiA64SubsX(JIT_SCRATCH_A, JAI_A64_XZR, ar));
                    branchOnDeoptInstStart(e, JAI_A64_VS);
                    emit(e, jaiA64CselX(ar, ar, JIT_SCRATCH_A, JAI_A64_MI));
                    dropCalleeEntry(e);
                    off += 2;
                    break;
                }
                /* `ord(c)` on a one-byte ASCII string is that byte, which is
                 * the inverse of the string-index arm above and restricted the
                 * same way: a multi-byte scalar needs decoding, and an empty
                 * or longer string is an error nOrd raises. All three deopt to
                 * the interpreter, which finishes the call properly.
                 *
                 * Worth an arm because it is the ONLY builtin json_parse
                 * declines on -- 141 of its 73 declined sites name it, and
                 * every one takes the whole enclosing loop down with it. A
                 * scanner reads its input a character at a time and asks what
                 * that character is; that is the shape. */
                if (strcmp(nm, "ord") == 0 && ak == SLOT_OBJ &&
                    IS_STRING(e->stackSeen[e->depth - 1])) {
                    emit(e, jaiA64LdrW(JIT_SCRATCH_A, ar,
                                       (unsigned)offsetof(Obj, type)));
                    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_STRING));
                    branchOnDeoptInstStart(e, JAI_A64_NE);
                    emit(e, jaiA64LdrW(JIT_SCRATCH_A, ar,
                                       (unsigned)offsetof(ObjString, length)));
                    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, 1));
                    branchOnDeoptInstStart(e, JAI_A64_NE);
                    emit(e, jaiA64LdrX(JIT_SCRATCH_A, ar,
                                       (unsigned)offsetof(ObjString, chars)));
                    emit(e, jaiA64LdrByte(JIT_SCRATCH_A, JIT_SCRATCH_A, 0));
                    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, 128));
                    branchOnDeoptInstStart(e, JAI_A64_HS);
                    /* The argument's register is not written until every guard
                     * has passed, so a bail still reconstructs the call with
                     * the string the program passed. */
                    emit(e, jaiA64MovX(ar, JIT_SCRATCH_A));
                    e->stack[e->depth - 1] = SLOT_INT;
                    /* The entry keeps its register and changes kind, so the
                     * claims made about the STRING that was in it go with it. */
                    e->stackObjType[e->depth - 1] = 0;
                    e->stackElem[e->depth - 1] = NULL_VAL;
                    dropCalleeEntry(e);
                    off += 2;
                    break;
                }
                bool toFloat = strcmp(nm, "float") == 0;
                bool toInt   = strcmp(nm, "int") == 0;
                if (toFloat && ak == SLOT_INT) {
                    /* Straight into the bank, not out through X: what consumes a `float(i)` is a float
                     * operator, and one reads the bank through fpOperand. Computing into d0 and storing the bits to X cost spectral's inner loop two cross-register-file fmovs per iteration -- the store, and the load the very next instruction made of it. */
                    unsigned fidx = e->valueDepth - 1;
                    if (!e->fpOff) {
                        emit(e, jaiA64ScvtfDX(fpRegAt(e, fidx), ar));
                        fpClaim(e, fidx);
                    } else {
                        emit(e, jaiA64ScvtfDX(JIT_FSCRATCH_A, ar));
                        emit(e, jaiA64FmovXD(ar, JIT_FSCRATCH_A));
                    }
                } else if (toInt && ak == SLOT_FLOAT) {
                    emit(e, jaiA64FcvtzsXD(ar, fpOperand(e, e->valueDepth - 1)));
                } else if (!((toFloat && ak == SLOT_FLOAT) ||
                             (toInt && ak == SLOT_INT))) {
                    int done = emitNativeResultCall(e, cv, nm, argc,
                                                    (uint32_t)(off + 2));
                    if (done < 0) return false;
                    if (done > 0) { off += 2; break; }
                    /* Names it: the table in kNativeResults is what would have
                     * to grow, and the reason alone never said which row. */
                    return subWhy(e, "`%s` has no known result kind for an "
                                     "argument of kind %s", nm,
                                  slotKindName(ak));
                }
                /* The result stays in the argument's register, so the claims
                 * made about what WAS in it go with it. */
                e->stack[e->depth - 1] = toFloat ? SLOT_FLOAT : SLOT_INT;
                e->stackObjType[e->depth - 1] = 0;
                e->stackElem[e->depth - 1] = NULL_VAL;
                dropCalleeEntry(e);
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
                unsigned rCallee0 =
                    valueBankReg(e, cidx - (e->depth - e->valueDepth));

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

                unsigned firstArg = cidx + 1u - (e->depth - e->valueDepth);
                for (unsigned i = 0; i < argc; i++) {
                    emit(e, jaiA64MovX(i, valueBankReg(e, firstArg + i)));
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
                /* x1 carries the callee's verdict. It used to BAIL here,
                 * which is sound only where partial execution is invisible --
                 * true of the function tier, whose locals are registers and
                 * whose frame is entered fresh, and false of the OSR tier,
                 * whose locals ARE the interpreter's frame slots and whose
                 * every way out syncs them back. An OSR bail mid-iteration
                 * therefore kept the loop variable already advanced and the
                 * locals already written, and the interpreter resumed at the
                 * LOOP HEAD -- which advanced the iterator again and dropped
                 * the rest of that iteration's work.
                 *
                 * That is the whole of the "closure call loses one count per
                 * collection" bug: `for v in xs { count += rule(v) }` came
                 * back short by one per bail, and under --gc-stress the callee
                 * deopts often enough to bail regularly. Measured on
                 * docs/repro/osr-closure-loses-counts.jai at --gc-stress=200:
                 * 170 of 77027 lost before, 0 after.
                 *
                 * A deopt at the CALL OFFSET instead, which is exactly what
                 * emitDirectCall does at the same point and for the same
                 * reason: the record is taken with the callee and its
                 * arguments still on the model's stack, so the interpreter
                 * re-runs the call from the top. Re-running is sound because
                 * the callee is a compiled body that bailed, and the function
                 * tier declines any body whose bail can follow a heap write.
                 *
                 * It costs nothing measurable -- two instructions on the path
                 * and a deopt record. Alternating binaries under
                 * scripts/gpu_lock.sh, best of five each: closure_calls
                 * 123/123, poly_dispatch 138/138, json_parse 110/110,
                 * object_dispatch 151/150, sort_merge 353/358. */
                emit(e, jaiA64SubsXImm(31, 1, 2));
                if (e->fixupCount >= JIT_MAX_FIXUPS) { e->failed = true; return false; }
                e->fixups[e->fixupCount].instIndex    = (int)e->count;
                e->fixups[e->fixupCount].targetOffset = FIXUP_THREW;
                e->fixups[e->fixupCount].conditional  = true;
                e->fixups[e->fixupCount].depth        = -1;
                e->fixupCount++;
                emit(e, jaiA64BCond(JAI_A64_EQ, 0));
                emit(e, jaiA64SubsXImm(31, 1, 0));
                branchOnDeoptAt(e, JAI_A64_NE, (uint32_t)off, false);

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
            unsigned first = e->valueDepth - argc;
            for (unsigned i = 0; i < argc; i++) {
                emit(e, jaiA64MovX(i, valueBankReg(e, first + i)));
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
            /* Its cold block routes a raised exception to the exception exit
             * (see emitSelfSlowStubs). */
            if (!raiseExitAllowed(e, "a call that can raise inside a try")) {
                return false;
            }
            unsigned si = e->selfSlowCount++;
            unsigned resultReg = valueXReg(e, e->valueDepth - argc);
            e->selfSlow[si].roots     = selfRoots;
            e->selfSlow[si].resultReg = resultReg;
            e->selfSlow[si].stub      = -1;
            /* This body is its own callee, but its own returns are not the
             * only way the deopt continuation can answer: `emitUnarmedDeopt`
             * can skip a run of instructions whose own `return`s never reach
             * `mergeReturnKind`'s walk, so `e->returnKind` is not proven to
             * bound every value this path can actually hand back. VAL_OBJ is
             * every heap object, so trusting a SLOT_LIST/SLOT_INST tag alone
             * risks the same wrong-object-shape read `emitDirectCall`'s
             * sibling stub (just above) already guards against; -1 is
             * "no type check", used for every other kind, and said explicitly
             * rather than left to the zeroed struct, where 0 is OBJ_STRING. */
            e->selfSlow[si].callee    = NULL;
            e->selfSlow[si].retType   = e->returnKind == SLOT_INST ? (int)OBJ_INSTANCE
                                       : e->returnKind == SLOT_LIST ? (int)OBJ_LIST
                                                                     : -1;
            e->selfSlow[si].retShape  = e->returnKind == SLOT_INST ? e->returnShape : 0;
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
        unarmedOpcode:
            /* An opcode this tier does not speak -- or an arm that cannot emit
             * for the shape in front of it: interpreted from here, rather than
             * the whole function interpreted. See emitUnarmedDeopt. */
            if (!emitUnarmedDeopt(e, &fn->chunk, &off, stop)) {
                if (getenv("JAI_JIT_WHY")) {
                    fprintf(stderr, "[jit] %s declined at %s\n",
                            fn->name ? fn->name->chars : "<anon>",
                            jaiOpName((OpCode)op));
                }
                return false;
            }
            afterUncond = true;   /* the fall-through edge is gone */
            continue;
        }
    }
    /* An inlined body's offsets are the callee's; matching them against the
     * caller's fixups compares two different numbering schemes. There is
     * nothing to check either way -- it has no branches. */
    if (e->inlining) {
        /* All but the result, which OP_RETURN deliberately left in the bank for
         * inlineGlobalCall to carry across. Everything under it is either the
         * caller's (settled before the call) or about to be discarded. */
        unsigned keep = e->valueDepth > 0 ? e->valueDepth - 1 : 32u;
        for (unsigned i = 0; i < 32u; i++) {
            if (i != keep) fpSyncOne(e, i);
        }
        settleAll(e);
        return !e->failed;
    }
    fpSyncAll(e);
    settleAll(e);
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
        /* A slot an earlier attempt asked to be widened takes the wider kind
         * the FIRST time something is bound to it too, not only when it was
         * seeded from a live value. A local declared inside the loop holds
         * nothing at the moment the loop tier looks -- it seeds SLOT_OPAQUE --
         * so without this the retry seeded nothing and found the same clash
         * again, which is exactly what `var at = head` inside the outer loop
         * does. */
        e->localKind[slot]  = (kind == SLOT_INST && e->nullableLocal[slot])
                                  ? SLOT_MAYBE_INST : kind;
        e->localShape[slot] = shape;
        e->localClass[slot] = klass;
        e->localTyped[slot] = true;
        return true;
    }
    if (e->localKind[slot] == kind && e->localShape[slot] == shape) return true;

    /* Two kinds for one slot. Rather than give the function up, note that this
     * slot has to carry its tag and let the caller compile again with that
     * decided from the start -- every read of it then guards, so the two
     * kinds stop being a contradiction.
     *
     * An instance and a nullable instance of the SAME class are not really two
     * kinds, though: the maybe-instance is the supertype, holds the identical
     * pointer-or-zero, and every field offset resolved against the class stays
     * right. So that clash asks for the nullable seed first and only falls to
     * the dynamic one if a second attempt still disagrees -- which matters
     * because dynamic keeps the slot in memory with a run-time tag, and this
     * is the shape every list walk has: `var at = head` then
     * `at = at.next`. */
    /* An instance into a slot already typed maybe-instance is not a clash at
     * all: it is the subtype going into the supertype, the register holds the
     * same bare pointer, and the tag a materialisation computes off that
     * pointer is the right one. The slot keeps the wider kind. Only the other
     * direction has to ask for anything, because a null reaching a slot typed
     * as a plain instance is what lets a later field read dereference zero. */
    if (e->localKind[slot] == SLOT_MAYBE_INST && kind == SLOT_INST &&
        e->localShape[slot] == shape) {
        return true;
    }
    if (e->localKind[slot] == SLOT_INST && kind == SLOT_MAYBE_INST &&
        e->localShape[slot] == shape &&
        !e->nullableLocal[slot] && !e->dynamicLocal[slot]) {
        e->needNullable[slot] = true;
        e->clashKind = kind;
        return false;
    }
    if (!e->dynamicLocal[slot]) {
        e->needDynamic[slot] = true;
        e->clashKind = kind;
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

static void jitFree(int *map, int *depths, int *chunkDepth, int count) {
    JAI_FREE_ARRAY(int, map, count);
    JAI_FREE_ARRAY(int, depths, count);
    JAI_FREE_ARRAY(int, chunkDepth, count);
}

/* The bytecode's own answer for every offset, or NULL if it cannot be had.
 * NULL is not a failure: modelAgreesWithChunk simply has nothing to check
 * against, which is the state the tier was in before this existed. */
static int *chunkDepthTable(const ObjFunction *fn) {
    int *d = JAI_ALLOC(int, fn->chunk.count + 1);
    if (d == NULL) return NULL;
    if (!jaiChunkStackDepths(fn, d)) {
        JAI_FREE_ARRAY(int, d, fn->chunk.count + 1);
        return NULL;
    }
    return d;
}

static bool eligible(ObjFunction *fn) {
    const char *why = NULL;
    /* Arity 0 is allowed: a method whose only parameter is the receiver has
     * arity 0, and getters are the commonest shape there is. A plain function
     * with no arguments reaches here too and declines on its opcodes. */
    if (fn->arity > JIT_MAX_ARITY) why = "arity";
    else if (fn->maxSlots < 1 || (unsigned)fn->maxSlots > JIT_MAX_SLOTS) why = "maxSlots";
    /* Defaults used to make a function categorically ineligible, which ruled
     * out most library code -- an API designed with optional arguments was
     * uncompilable by construction. It is safe because every entry already
     * requires the arguments to have been supplied in full: callClosure and
     * the one-argument path both test `argc == fn->arity`, and OP_TAIL_CALL
     * reuses its window only for a callee with no defaults at all. A call that
     * omits one therefore never reaches compiled code; it fills the defaults
     * and runs interpreted, exactly as before. */
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
/* Both tiers plan through this, on the same ledgers (see noteSlotCost). What is
 * still per-tier is named here rather than duplicated:
 *
 *   `skip`     slots the caller has already ruled out for reasons the ledgers
 *              cannot see -- OSR passes the ones captured by reference by a
 *              closure, which must stay in memory whatever they save, and the
 *              ones its probe found to be dynamic. NULL for the function tier,
 *              whose dynamic slots are in e->dynamicLocal already.
 *   xBase      OSR's callee-saved window opens ABOVE the registers the loop
 *              head reserved for its iterator; the function tier's opens at
 *              x19. Matches regBase, which the operand stack is numbered from.
 *   the FP gate  the function tier lets the two ledgers pick the bank, because
 *              jitArgIn marshals every incoming argument and an FP home there
 *              only ever sees a clean value. OSR's homes are the interpreter's
 *              own slots and its prologue fills an FP home with a full-width
 *              `ldr d`; for a SLOT_BOOL that loads the seven bytes above a
 *              one-byte `strb`, which is the garbage-high-byte bug the OSR
 *              prologue's X arm narrows against. Only a float may take an FP
 *              home there.
 *   `strandedOut`  counted for the OSR census: slots that earned a register and
 *              found none left, which is what a wider bank would buy. */
static void planSlotRegisters(Emit *e, const Emit *m, unsigned availX,
                              const bool *skip, unsigned *strandedOut) {
    unsigned availFp = JIT_FP_MAX_SAVED;
    unsigned xBase = e->osr ? osrReserved(e) : 0u;
    unsigned top = e->base + e->locals;
    if (top > JIT_MAX_SLOTS + 1u) top = JIT_MAX_SLOTS + 1u;
    while (availX > 0 || availFp > 0) {
        unsigned bestSlot = 0, bestGain = 0;
        bool bestFp = false;
        for (unsigned slot = e->base; slot < top; slot++) {
            if (e->slotXReg[slot] != 0 || e->slotFpReg[slot] != 0) continue;
            if (e->dynamicLocal[slot]) continue;
            if (skip != NULL && skip[slot]) continue;
            bool fpOk = !e->osr || m->localKind[slot] == SLOT_FLOAT;
            if (availFp > 0 && fpOk && m->slotSaveFp[slot] > bestGain) {
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
            e->slotXReg[bestSlot] =
                (uint8_t)(JIT_FIRST_SAVED + xBase + e->xLocals++);
            availX--;
        }
    }
    if (strandedOut == NULL) return;
    for (unsigned slot = e->base; slot < top; slot++) {
        if (e->slotXReg[slot] != 0 || e->slotFpReg[slot] != 0) continue;
        if (e->dynamicLocal[slot]) continue;
        if (skip != NULL && skip[slot]) continue;
        if (m->slotSaveX[slot] == 0 && m->slotSaveFp[slot] == 0) continue;
        (*strandedOut)++;
    }
}

/* Writes a body into the arena and seals it again, or seals and reports
 * failure.
 *
 * The sealing is the point. `jaiCodeArenaUnseal` turns the WHOLE arena back to
 * read-write, which takes the execute bit off every function already compiled
 * into it -- so an early return between the unseal and the seal leaves the
 * tier's entire back catalogue unexecutable, and the next call into any of it
 * dies with KERN_PROTECTION_FAILURE inside `callClosure`. That is not
 * hypothetical: the write fails exactly when the 1 MB arena is full, which a
 * long enough run reaches, and it produced an intermittent SIGBUS in the test
 * suite that moved around as unrelated changes altered how much got compiled.
 * Every exit from here re-seals. */
static uint8_t *arenaEmit(JaiCodeArena *arena, const uint32_t *code,
                          unsigned count) {
    while ((arena->used & 31u) != 0) {
        uint32_t pad = jaiA64Nop();
        if (jaiCodeArenaWrite(arena, &pad, sizeof pad) == NULL) {
            jaiCodeArenaSeal(arena);
            return NULL;
        }
    }
    uint8_t *entry = jaiCodeArenaWrite(arena, code, count * sizeof code[0]);
    if (entry == NULL) {
        jaiCodeArenaSeal(arena);
        return NULL;
    }
    if (!jaiCodeArenaSeal(arena)) return NULL;
    return entry;
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
    int *chunkDepth = chunkDepthTable(fn);
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
    e.chunkDepth = chunkDepth;
    e.chunkDepthCount = fn->chunk.count + 1;
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
    body.chunkDepth = chunkDepth;
    body.chunkDepthCount = fn->chunk.count + 1;
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
        jitFree(map, depths, chunkDepth, fn->chunk.count + 1);
        return false;
    }

    /* A first pass with a provisional frame, only to learn maxValue. The
     * emitted words are thrown away: frameBytes appears in the epilogue, so
     * they would be wrong. */
    body.savedCount = JIT_MAX_SAVED;
    body.frameBytes = 16 + 8 * JIT_MAX_SAVED + 8;   /* 16-aligned below */
    body.frameBytes = (body.frameBytes + 15u) & ~15u;
    /* Snapshot before the walk, so the chain diagnostic can re-run it. */
    static Emit chainProto;
    if (jitChainOn()) memcpy(&chainProto, &body, sizeof body);
    if (!compileBody(&body, closure)) {
        memcpy(needDynamic, body.needDynamic, sizeof body.needDynamic);
        memcpy(needNullable, body.needNullable, sizeof body.needNullable);
        if (getenv("JAI_JIT_WHY")) {
            fprintf(stderr, "[jit] %s stopped (measuring): %s\n",
                    fn->name ? fn->name->chars : "<anon>",
                    declineReason(&body));
        }
        if (jitChainOn()) reportChain(&chainProto, &body, closure, fn);
        jitFree(map, depths, chunkDepth, fn->chunk.count + 1);
        return false;
    }

    /* A self-call cannot reproduce slot 0: no register holds the callee. A
     * body that both recurses and reads slot 0 is not compiled. */
    if (body.usesSlot0 && body.hasSelfCall) {
        if (getenv("JAI_JIT_WHY")) {
            fprintf(stderr, "[jit] %s stopped: a body that both recurses and reads slot 0\n",
                    fn->name ? fn->name->chars : "<anon>");
        }
        jitFree(map, depths, chunkDepth, fn->chunk.count + 1);
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
            planSlotRegisters(&e, &body, JIT_MAX_SAVED - saved, NULL, NULL);
            saved += e.xLocals;
        }
    }
    if (e.whyNot != NULL) {
        if (getenv("JAI_JIT_WHY")) {
            fprintf(stderr, "[jit] %s stopped: %s (%u locals, %u stack)\n",
                    fn->name ? fn->name->chars : "<anon>", e.whyNot, e.locals,
                    body.maxValue);
        }
        jitFree(map, depths, chunkDepth, fn->chunk.count + 1);
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
        jitFree(map, depths, chunkDepth, fn->chunk.count + 1);
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
            /* Arguments arrive in x0..x7 and homes start at x19, so no
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
                unsigned tag = localTagFor(&body, slot);
                /* Payload-dependent for every object-ish kind, exactly as
                 * localOut writes one: a null argument arrives as a zero
                 * payload (jitArgIn's SLOT_MAYBE_INST arm), and a constant
                 * VAL_OBJ over it would leave the frame claiming an object at
                 * address zero -- which localIn's guard would then trust as
                 * far as `Obj.type`, and which a deopt copies out verbatim as
                 * OBJ_VAL(NULL) for the interpreter to trip over. The other
                 * VAL_OBJ kinds cannot be null (jitArgIn refuses), so for
                 * them the csel below only ever picks the same VAL_OBJ. */
                if (tag == VAL_OBJ) {
                    emitTagFor(&e, SLOT_MAYBE_INST, i, JIT_SCRATCH_D,
                               JIT_SCRATCH_C);
                } else {
                    emit(&e, jaiA64MovzX(JIT_SCRATCH_D, tag, 0));
                }
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
    e.chunkDepth = chunkDepth;
    e.chunkDepthCount = fn->chunk.count + 1;
    if (!seedLocals(&e, slotBase)) {
        if (getenv("JAI_JIT_WHY")) {
            fprintf(stderr, "[jit] %s stopped: its locals could not be seeded on the real pass\n",
                    fn->name ? fn->name->chars : "<anon>");
        }
        jitFree(map, depths, chunkDepth, fn->chunk.count + 1);
        return false;
    }
    /* declineReason, not e.whyNot: an arm that noted only a whySub used to
     * print "an unsupported operand form", which is how math.sqrt's own body
     * managed to stop on a named refusal and report nothing. */
    if (!compileBody(&e, closure) && getenv("JAI_JIT_WHY")) {
        fprintf(stderr, "[jit] %s stopped: %s\n", fn->name ? fn->name->chars : "<anon>",
                declineReason(&e));
    }
    if (e.failed || e.whyNot != NULL)
        { jitFree(map, depths, chunkDepth, fn->chunk.count + 1); return false; }
    if (e.failed) {
        if (getenv("JAI_JIT_WHY")) {
            fprintf(stderr, "[jit] %s stopped: %s\n",
                    fn->name ? fn->name->chars : "<anon>",
                    e.whyNot ? e.whyNot : "the emitter ran out of room");
        }
        jitFree(map, depths, chunkDepth, fn->chunk.count + 1);
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

        uint64_t skipLocals = 0;
        for (unsigned i = 0; i < (e.osr ? 0u : e.locals); i++) {
            unsigned slot = e.base + i;
            SlotKind kind = e.localKind[slot];
            /* An opaque slot is one the compiled body never reads, so the tier
             * never learned what is in it -- and jitArgIn passed a raw 0 for
             * it. The interpreter this record hands over to DOES read it, and
             * bindCallArgs has already put the caller's real argument in the
             * frame, so the record must say "not mine" rather than null.
             * See JitDeoptRecord::skipLocals. */
            if (kind == SLOT_OPAQUE) {
                skipLocals |= (uint64_t)1 << i;
                continue;
            }
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
        /* Written unconditionally, including the zero: gDeopt is one global and
         * a mask left over from another function's stub would silently drop a
         * local this one does describe. */
        emitConst64(&e, JIT_SCRATCH_B, (int64_t)skipLocals);
        emit(&e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_A,
                            (unsigned)offsetof(JitDeoptRecord, skipLocals)));

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
            unsigned reg0 = valueBankReg(&e, valueSeen);
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
            unsigned reg = valueBankReg(&e, valueSeen);
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
        jitFree(map, depths, chunkDepth, fn->chunk.count + 1);
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
        jitFree(map, depths, chunkDepth, fn->chunk.count + 1);
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
                   f->targetOffset > FIXUP_EXIT - JIT_MAX_EXIT) {
            target = e.exitStub[FIXUP_EXIT - f->targetOffset];
            if (target < 0) {
                if (getenv("JAI_JIT_WHY")) {
                    fprintf(stderr, "[jit] %s stopped: a stub it branches to "
                                    "was never emitted\n",
                            fn->name ? fn->name->chars : "<anon>");
                }
                jitFree(map, depths, chunkDepth, fn->chunk.count + 1);
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
                jitFree(map, depths, chunkDepth, fn->chunk.count + 1);
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
                jitFree(map, depths, chunkDepth, fn->chunk.count + 1);
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
                jitFree(map, depths, chunkDepth, fn->chunk.count + 1);
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
                jitFree(map, depths, chunkDepth, fn->chunk.count + 1);
                return false;
            }
        } else if (f->targetOffset == FIXUP_ENTRY) {
            target = 0;
        } else {
            if (f->targetOffset > (uint32_t)fn->chunk.count) {
                if (getenv("JAI_JIT_WHY")) {
                    fprintf(stderr, "[jit] %s stopped: a branch target past the end of the chunk\n",
                            fn->name ? fn->name->chars : "<anon>");
                }
                jitFree(map, depths, chunkDepth, fn->chunk.count + 1);
                return false;
            }
            target = map[f->targetOffset];
            if (target < 0) {
                if (getenv("JAI_JIT_WHY")) {
                    if (e.unarmedOp != 0) {
                        fprintf(stderr, "[jit] %s stopped: %s at %u ended the "
                                "walk, so the branch to offset %u (%s) has "
                                "nowhere to land\n",
                                fn->name ? fn->name->chars : "<anon>",
                                jaiOpName((OpCode)e.unarmedOp), e.unarmedAt,
                                f->targetOffset,
                                f->targetOffset < (uint32_t)fn->chunk.count
                                    ? jaiOpName((OpCode)fn->chunk.code[f->targetOffset])
                                    : "past the end");
                    } else {
                        fprintf(stderr, "[jit] %s stopped: a branch to offset "
                                "%u, which this walk never emitted (%s)\n",
                                fn->name ? fn->name->chars : "<anon>",
                                f->targetOffset,
                                f->targetOffset < (uint32_t)fn->chunk.count
                                    ? jaiOpName((OpCode)fn->chunk.code[f->targetOffset])
                                    : "past the end");
                    }
                }
                jitFree(map, depths, chunkDepth, fn->chunk.count + 1);
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
                jitFree(map, depths, chunkDepth, fn->chunk.count + 1);
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
    jitFree(map, depths, chunkDepth, fn->chunk.count + 1);

    /* A `bl` at instruction i must reach instruction 0 of this function, so
     * the recursive-call fixups above are relative to the function's own
     * start, which is where the arena is about to place it. */
    if (!jaiCodeArenaUnseal(arena)) return false;
    /* The entry is 32-aligned, which is two things at once.
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
    uint8_t *entry = arenaEmit(arena, e.code, e.count);
    if (entry == NULL) return false;

    if (e.whyNot != NULL && getenv("JAI_JIT_WHY")) {
        fprintf(stderr, "[jit] %s stopped: %s\n",
                fn->name ? fn->name->chars : "<anon>", e.whyNot);
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
    fn->jitReturnKnown = e.sawReturn;
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
        /* A partial compile used to be entirely silent: emitUnarmedDeopt
         * interprets from an unsupported opcode ONWARD rather than declining
         * the body, so "compiled" was printed for a body of which two
         * instructions were compiled and forty were not. That is how OP_NEG
         * stayed hidden long enough to cost 9x, and it was only ever visible
         * when a forward branch happened to dangle past the stopping point.
         *
         * Naming it here turns "which opcode should I arm next" from a manual
         * audit into a grep. */
        if (e.unarmedOp != 0) {
            fprintf(stderr,
                    "[jit] %s walked only to %s at %u -- the rest of the body "
                    "is interpreted\n",
                    fn->name ? fn->name->chars : "<anon>",
                    jaiOpName((OpCode)e.unarmedOp), e.unarmedAt);
        }
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

/* Where the loop whose head is `top` ends: after its LAST back edge, not its
 * first.
 *
 * A loop has one back edge for the ordinary path off the bottom and one more
 * for every `continue`, and a `continue` comes FIRST -- it is written near the
 * top of the body, which is the point of it. Stopping at the first meant the
 * compiled region was the prefix before the `continue` and everything after it
 * ran interpreted, so the more often the `continue` was SKIPPED the slower the
 * loop got: measured over three hundred thousand elements, 0.25ms when every
 * iteration took the `continue` against 7.6ms when none did, for the same loop
 * written with an `if` in 0.29ms.
 *
 * Only this loop's own back edges name `top`: an inner loop's name the inner
 * head, and a later loop's name its own. So the last one is this loop's bottom
 * however many `continue`s are between. */
static uint32_t findLoopEnd(const Chunk *c, uint32_t top, bool wholeBody) {
    uint32_t end = 0;
    for (int off = (int)top; off < c->count;) {
        uint8_t op = c->code[off];
        int len = instructionLength(c, off);
        if (len <= 0) return end;
        if (op == OP_LOOP) {
            int16_t jump = jaiReadI16(c->code + off + 1);
            if ((uint32_t)((int32_t)(off + 3) + jump) == top) {
                end = (uint32_t)(off + 3);
                if (!wholeBody) return end;
            }
        }
        off += len;
    }
    return end;
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

static bool compileOsrOnce(ObjClosure *closure, uint32_t top, Value *slots,
                           uint8_t iterKind, Value elemSample, bool elemMixed,
                           uint8_t elemStg,
                           bool wholeBody, bool noInline,
                           const bool *nullable, bool *needNullable,
                           const bool *dynamic, bool *needDynamic);

/* The loop tier had no retry at all, where the function tier has had one since
 * `nullableLocal` existed: a slot the walk cannot settle on one kind for simply
 * declined the whole loop. That is `var at = head` followed by `at = at.next`
 * -- an instance seeded from the slot, then a maybe-instance assigned into it
 * -- which is every list and tree walk there is, and the shape a parser's inner
 * loops are made of. Two attempts: the second seeds the slots the first asked
 * for as nullable.
 *
 * MEASURED, and only as a chain. On its own this changes nothing: widening the
 * local moves the refusal from the assignment to the `at.follow()` call, and
 * accepting a nullable RETURN kind (see emitMaybeInstResult) moves it back to
 * the assignment. All three together plus the maybe-instance receiver at
 * OP_INVOKE take a 200-node list walked 20000 times from 137 ms to 38 ms,
 * best of seven alternating runs under scripts/gpu_lock.sh with no overlap
 * between the two sets -- 3.6x, and it is a decline-to-compile transition,
 * not a micro-optimisation: `walk` goes from no compiled form at either loop
 * head to both.
 *
 * RULED OUT: the self-hosted compiler does not move. `check --no-cache` over
 * four compiler files is 2143 ms against 2146 ms median of nine, inside the
 * spread, even though the return-kind gate alone stops 64 sites in one file.
 * Those bodies clear it and stop at the next wall (OP_INVOKE on a receiver
 * whose class the model cannot pin, and OP_GET_FIELD_LOCAL). Kept for the
 * loops it does unblock, on the same reasoning the enum fold was. */
static bool compileOsr(ObjClosure *closure, uint32_t top, Value *slots,
                       uint8_t iterKind, Value elemSample, bool elemMixed,
                       uint8_t elemStg,
                       bool wholeBody, bool noInline) {
    bool nullable[JIT_MAX_SLOTS + 1];
    bool needNullable[JIT_MAX_SLOTS + 1];
    /* Same ledger the function tier has kept since it grew one: a slot given
     * two kinds does not give the loop up, it asks to carry its tag and the
     * walk runs again with that decided from the start. OSR had the nullable
     * half of this and not the dynamic half, and a comprehension is exactly
     * what needed it -- the front end reuses one slot for the loop variable and
     * for the result, so `[x * 2 for x in xs]` seeds the slot as a list from
     * the previous iteration and then binds an int into it. Three attempts,
     * because a body can want both widenings and neither implies the other. */
    bool dynamic[JIT_MAX_SLOTS + 1];
    bool needDynamic[JIT_MAX_SLOTS + 1];
    memset(nullable, 0, sizeof nullable);
    memset(dynamic, 0, sizeof dynamic);
    for (int attempt = 0; attempt < 3; attempt++) {
        memset(needNullable, 0, sizeof needNullable);
        memset(needDynamic, 0, sizeof needDynamic);
        if (compileOsrOnce(closure, top, slots, iterKind, elemSample, elemMixed,
                           elemStg, wholeBody, noInline, nullable,
                           needNullable, dynamic, needDynamic)) {
            return true;
        }
        bool grew = false;
        for (unsigned i = 0; i <= JIT_MAX_SLOTS; i++) {
            if (needNullable[i] && !nullable[i]) { nullable[i] = true; grew = true; }
            if (needDynamic[i] && !dynamic[i]) { dynamic[i] = true; grew = true; }
        }
        if (!grew) return false;
    }
    return false;
}

static bool compileOsrOnce(ObjClosure *closure, uint32_t top, Value *slots,
                           uint8_t iterKind, Value elemSample, bool elemMixed,
                           uint8_t elemStg,
                           bool wholeBody, bool noInline,
                           const bool *nullable, bool *needNullable,
                           const bool *dynamic, bool *needDynamic) {
    bool hasIter = iterKind != 0;
    ObjFunction *fn = closure->fn;
    if (!isInstructionStart(&fn->chunk, top)) return false;
    uint32_t end = findLoopEnd(&fn->chunk, top, wholeBody);
    if (end == 0 || end <= top) return false;
    /* The entry re-checks every slot, so this is the size of that record --
     * nbody's advance declares nineteen. */
    if (fn->maxSlots < 1 || (unsigned)fn->maxSlots > 40) return false;

    JaiCodeArena *arena = jaiJitArena();
    if (arena == NULL) return false;

    int *map = JAI_ALLOC(int, fn->chunk.count + 1);
    int *depths = JAI_ALLOC(int, fn->chunk.count + 1);
    int *chunkDepth = chunkDepthTable(fn);
    for (int i = 0; i <= fn->chunk.count; i++) { map[i] = -1; depths[i] = -1; }

    static Emit e;
    memset(&e, 0, sizeof e);
    e.osr = true;
    e.loopDepth = gLoopDepth;
    e.loopDepthCount = loopDepthTable(&fn->chunk);
    e.hasIter = hasIter;
    e.iterKind = iterKind;
    e.elemSample = elemSample;
    e.elemMixed  = elemMixed;
    e.elemStg    = elemStg;
    /* See Emit::rangeHead. */
    if (top + 9u <= (uint32_t)fn->chunk.count &&
        fn->chunk.code[top] == OP_FOR_RANGE_BIND) {
        e.rangeHead = true;
        e.rangeVar  = jaiReadU16(fn->chunk.code + top + 3);
        e.rangeCur  = jaiReadU16(fn->chunk.code + top + 5);
        e.rangeEnd  = jaiReadU16(fn->chunk.code + top + 7);
    }
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
    e.chunkDepth = chunkDepth;
    e.chunkDepthCount = fn->chunk.count + 1;
    e.savedCount = JIT_MAX_SAVED;
    memcpy(e.nullableLocal, nullable, sizeof e.nullableLocal);
    memcpy(e.dynamicLocal, dynamic, sizeof e.dynamicLocal);
    /* Each slot takes the kind it holds right now. The entry re-checks them on
     * every later entry, so this is a specialisation, not an assumption. */
    for (unsigned i = 0; i < e.locals; i++) {
        Value v = slots[i];
        e.localTyped[i] = true;
        /* A frame's slots run to fn->maxSlots, and the ones the program has
         * not reached yet hold whatever the last frame at that depth left --
         * including a VAL_OBJ tag over a NULL pointer. IS_LIST and IS_INSTANCE
         * both dereference before they test, so without this the COMPILER
         * reads Obj::type off address zero: EXC_BAD_ACCESS inside
         * compileOsrOnce, once in eight runs of tests/bench/jaiframe/frameops
         * and never in the test suite, because it needs a slot that is both
         * stale and untouched at the moment a loop goes hot. */
        if (IS_OBJ(v) && AS_OBJ(v) == NULL) {
            e.localKind[i] = SLOT_OPAQUE;
            e.localTyped[i] = false;
            continue;
        }
        if (IS_INT(v))        e.localKind[i] = SLOT_INT;
        else if (IS_FLOAT(v)) e.localKind[i] = SLOT_FLOAT;
        else if (IS_BOOL(v))  e.localKind[i] = SLOT_BOOL;
        else if (IS_LIST(v))  e.localKind[i] = SLOT_LIST;
        else if (IS_INSTANCE(v) && AS_INSTANCE(v)->klass != NULL) {
            /* A slot an earlier attempt found sometimes-null takes the wider
             * kind from the start; the class still comes from what is in the
             * slot NOW, which a maybe-instance is a correct supertype of. */
            e.localKind[i]  = nullable[i] ? SLOT_MAYBE_INST : SLOT_INST;
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
    unsigned probeMaxValueAll = 0;
    unsigned probeStranded = 0;
    unsigned probeClobberDepth = 0;
    bool probeRan = false;
    {
        static Emit probe;
        memset(&probe, 0, sizeof probe);
        probe.osr = true; probe.measuring = true; probe.hasIter = hasIter;
        probe.iterKind = iterKind; probe.elemSample = elemSample;
        probe.elemMixed = elemMixed;
        probe.elemStg = elemStg;
        probe.rangeHead = e.rangeHead;
        probe.rangeVar  = e.rangeVar;
        probe.rangeCur  = e.rangeCur;
        probe.rangeEnd  = e.rangeEnd;
        probe.osrTop = top; probe.osrEnd = end; probe.base = 0;
        probe.noInline = noInline;
        probe.locals = e.locals; probe.callsOut = true; probe.observed = slots;
        probe.scratchRoom = JIT_SCRATCH_BANK_COUNT;
        probe.offsetToInst = map; probe.offsetToDepth = depths;
        probe.chunkDepth = chunkDepth; probe.chunkDepthCount = fn->chunk.count + 1;
        probe.limitLiteral = -1; probe.bailBlock = -1; probe.exceptionExit = -1;
        probe.loopDepth = gLoopDepth;
        probe.loopDepthCount = e.loopDepthCount;
        /* "Never seen" is an empty range, not offset zero. */
        for (unsigned i = 0; i <= JIT_MAX_SLOTS; i++) {
            probe.slotWriteLo[i] = UINT32_MAX;
            probe.slotIndexLo[i] = UINT32_MAX;
            probe.spanLo[i]   = INT32_MAX;
            probe.spanHi[i]   = INT32_MIN;
            probe.spanOk[i]   = true;
            probe.spanSeen[i] = false;
        }
        memcpy(probe.nullableLocal, nullable, sizeof probe.nullableLocal);
        memcpy(probe.dynamicLocal, dynamic, sizeof probe.dynamicLocal);
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
            probeMaxValueAll = probe.maxValueAll;
            probeClobberDepth = probe.clobberDepth;
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
            /* Under split stress the split is preferred to the whole-scratch
             * bank, because it is the two-run layout that has the boundary in
             * it and the whole-scratch one does not. */
            if (jitSplitStress()) e.scratchValues = false;
            /* A body that DOES call still need not keep its WHOLE stack in the
             * callee-saved bank -- only the part of it that can be live while
             * a helper runs, which is everything below the deepest the stack
             * ever is at a call. life's `step` goes five deep summing nine
             * neighbours and two deep at its `row.push`, so three of its five
             * operand registers were being held against a call that can never
             * see them, while five of its locals sat in memory.
             *
             * Not offered to a body that inlines: an inlined body takes x0..x8
             * for its own entries (inlineOwnBank), and both cannot have them.
             * Not offered to the function tier either, which does not run this
             * code -- x0..x3 are its arguments and the roadmap prices the same
             * change there at +-1%. */
            unsigned wantSplit = probe.clobberDepth;
            /* Stress: a body with no calls at all has clobberDepth 0, and a
             * split at 0 is no split. Moving the boundary up to 1 is still
             * sound there -- nothing can clobber x0..x8 in a body that never
             * calls -- and it is what puts the two-run layout under every
             * existing gate rather than under the few bodies that want it. */
            if (jitSplitStress() && wantSplit == 0) wantSplit = 1;
            if (!e.scratchValues && !probe.inlined &&
                (probe.clobbersScratch || jitSplitStress()) &&
                probe.maxValue > wantSplit &&
                probe.maxValue - wantSplit <= JIT_SCRATCH_BANK_COUNT) {
                e.splitAt = wantSplit;
            }
            /* What is left over after the loop's own reserved registers and
             * the deepest expression the body builds. maxValue is model state,
             * not a register number, so measuring it in memory mode and
             * spending it here is sound. Under a split only the half below
             * `splitAt` is charged to the callee-saved bank. */
            unsigned overhead = osrReserved(&e) +
                                (e.scratchValues ? 0u
                                 : e.splitAt != 0 ? e.splitAt
                                                  : probe.maxValue);
            unsigned availX = overhead < JIT_MAX_SAVED
                                  ? JIT_MAX_SAVED - overhead : 0u;
            /* Ranked by what a register SAVES, on the same ledgers and through
             * the same planner as the function tier -- not by how often the
             * body names the slot, which is what this used to do. Both counts
             * are weighted by loop nesting the same way, so the difference is
             * not "flat versus weighted": it is that one pooled counter had to
             * answer two questions it cannot tell apart. It could not say which
             * BANK a slot wants (a float read costs an `ldr d` from memory and
             * an `fmov` from an X home, so the two banks are not worth the same
             * to it), and it could not say that a WRITE is worth twice a read
             * here, because an OSR slot's memory home is the interpreter's own
             * Value and takes the tag as well as the payload. A loop variable
             * assigned every iteration is exactly the case both mistakes fell
             * on, which is the case OSR exists for.
             *
             * The slots the ledgers cannot rule out on their own are handed
             * over as `skip`: one captured by reference by a closure, which is
             * aliased by an ObjUpvalue pointing into the VM's slot array and so
             * must stay in memory whatever it saves (see chunkByRefCaptures);
             * one the probe found dynamic, which keeps its run-time tag and has
             * nowhere but the frame to put it; and, when the chunk would not
             * decode at all, every slot -- the by-reference scan is what makes
             * a register safe here, so failing it means registers for nobody.
             *
             * `slotUse` stays in that mask rather than being retired for the
             * ledgers, so the set of slots this can place is the old one
             * narrowed, never widened. The two disagree in one direction only:
             * slotUse goes uncounted past JIT_MAX_DEPTH_MAP (a chunk over 8KB),
             * where the ledgers still count at weight 1, so retiring it would
             * hand registers to slots in a region no gate has ever planned for.
             * Whatever the ledgers rank last is dropped by the zero-gain
             * exclusion anyway, which is the accurate half of the old test:
             * slotUse counts localInRange, a bounds CHECK made at more sites
             * than actually read or write the slot. */
            bool byRef[JIT_MAX_SLOTS + 1];
            bool decoded = chunkByRefCaptures(&fn->chunk, byRef, e.locals);
            bool skip[JIT_MAX_SLOTS + 1];
            for (unsigned i = 0; i <= JIT_MAX_SLOTS; i++) {
                skip[i] = !decoded || i >= e.locals || byRef[i] ||
                          probe.dynamicLocal[i] || probe.slotUse[i] == 0;
            }
            /* `probeStranded` is the count of slots that earned a register and
             * found none left -- what a wider bank would buy, and 0 far more
             * often than the decline census suggests. */
            planSlotRegisters(&e, &probe, availX, skip, &probeStranded);

            /* What planHoists needs: where each slot was written, where it was
             * subscripted, where the body can destroy a caller-saved register,
             * and the registers nothing else claims. */
            e.bodyCalls = probe.clobbersScratch;
            for (unsigned i = 0; i <= JIT_MAX_SLOTS; i++) {
                e.slotWriteLo[i]  = probe.slotWriteLo[i];
                e.slotWriteHi[i]  = probe.slotWriteHi[i];
                e.slotIndexLo[i]  = probe.slotIndexLo[i];
                e.slotIndexHi[i]  = probe.slotIndexHi[i];
                e.slotIndexUse[i] = probe.slotIndexUse[i];
                e.spanLo[i]       = probe.spanLo[i];
                e.spanHi[i]       = probe.spanHi[i];
                e.spanOk[i]       = probe.spanOk[i];
                e.spanBase[i]     = probe.spanBase[i];
                e.spanSeen[i]     = probe.spanSeen[i];
            }
            e.clobberCount = probe.clobberCount;
            e.clobberSpill = probe.clobberSpill;
            for (unsigned i = 0; i < probe.clobberCount; i++) {
                e.clobberOff[i] = probe.clobberOff[i];
            }

            /* Which sampled storages the real pass may emit against.
             *
             * A slot ASSIGNED inside the region cannot be pinned: the tier's
             * OP_ELEM_KIND arm writes elemKind without specialising, so a
             * `var f: list[bool] = []` at the top of a loop body makes a BOXED
             * list every round while the pin, taken from the frame before
             * entry, still says U8. A sieve written that way read every
             * element one byte wide out of a sixteen-byte array and printed a
             * different prime count on each run.
             *
             * A region that CALLS OUT cannot pin either: jaiListBox is
             * reachable from ordinary builtins -- a sort, a concat of two
             * storages, a put of a kind the store cannot hold -- and it
             * de-specialises the list with nothing the compiled body could
             * watch. The same test planHoists makes of a hoisted header, for
             * the same reason: the entry guard proves a fact, and this is
             * whether the body can invalidate it. */
            bool bodyCallsOut = regionCalls(&e, top, end);
            for (unsigned i = 0; i <= JIT_MAX_SLOTS; i++) {
                e.localStgPin[i] =
                    !bodyCallsOut &&
                    !(e.slotWriteHi[i] >= top && e.slotWriteLo[i] < end);
            }
            e.elemStgPin = !bodyCallsOut;
            /* x13..x17 are nobody's in any body -- a call destroys them, which
             * is why only a call-free REGION may hold anything there, and
             * planHoists is what tests that. Offering them unconditionally is
             * the whole per-region change: they used to be withheld from every
             * body containing a call, including the ones whose inner loop is
             * the entire benchmark. */
            for (unsigned r = 0; r < JIT_FREE_COUNT; r++) {
                e.hoistPool[e.hoistPoolCount++] = (uint8_t)(JIT_FREE_FIRST + r);
            }
            /* Whatever the operand stack left at the top of its own bank.
             * Only when it IS that bank -- otherwise those registers are
             * carrying operands, or an inlined body has x0..x8 to itself and
             * none of it is spare. `scratchValues` is a whole-body claim and
             * has to stay one: unlike x13..x17, these registers have another
             * owner outside the region. */
            if (e.scratchValues) {
                for (unsigned r = probe.maxValueAll;
                     r < JIT_SCRATCH_BANK_COUNT; r++) {
                    e.hoistPool[e.hoistPoolCount++] =
                        (uint8_t)(JIT_INL_BANK + r);
                }
            }
        } else {
            /* The probe is where a kind clash is found -- the real walk below
             * runs on the same seed and would only find it again -- so its
             * request for a wider one has to travel back to the retry loop
             * from here. */
            memcpy(needNullable, probe.needNullable, sizeof probe.needNullable);
            memcpy(needDynamic, probe.needDynamic, sizeof probe.needDynamic);
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
    if (hasIter && iterKind == 3) {
        /* A dict-items head keeps only the pointer: its index, limit and
         * version all live in the ObjIter and the step reads them there, so
         * there is nothing to hoist here and nothing to write back at an exit. */
        emit(&e, jaiA64MovX(JIT_PAIR_ITER_REG, 1));
    } else if (hasIter) {
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
    if (getenv("JAI_JIT_WHY")) {
        /* The register arithmetic, not just the verdict, and for a body that
         * COMPILES as much as one that does not: "declined for want of a
         * register" is only half the census -- the other half is a body that
         * fitted with nothing left over, which is what a bank decision is
         * chosen against. Printed here rather than in the failure arm because
         * this is the point where every number in it is final.
         *
         * Only when the measuring pass got through: a loop that declined
         * before the register plan has zeroed numbers, and printing them reads
         * as "nought entries deep", which is a different and false claim.
         *
         * Its own line, and without the words the decline census greps for, so
         * that adding it cannot invent census entries. */
        if (probeRan) fprintf(stderr,
                "[jit] osr %s at %u registers: %u reserved, %u stack "
                "(%u incl. inlined), %u deep at a call, %u x-locals, "
                "%u fp-locals, %u stranded, bank %s, of %u; %s\n",
                fn->name ? fn->name->chars : "<anon>", top,
                osrReserved(&e), probeMaxValue, probeMaxValueAll,
                probeClobberDepth, e.xLocals, e.fpLocals, probeStranded,
                e.scratchValues ? "x0"
                                : e.splitAt != 0 ? "split" : "callee-saved",
                JIT_MAX_SAVED, e.bodyCalls ? "calls" : "call-free");
        for (unsigned i = 0; i < e.hoistCount; i++) {
            fprintf(stderr,
                    "[jit] osr at %u hoists slot %u's header out of %u..%u "
                    "into x%u/x%u\n",
                    top, e.hoist[i].slot, e.hoist[i].top, e.hoist[i].end,
                    e.hoist[i].itemsReg, e.hoist[i].countReg);
        }
    }

    if (!compileBody(&e, closure) || e.failed) {
        for (unsigned i = 0; i <= JIT_MAX_SLOTS; i++) {
            if (e.needNullable[i]) needNullable[i] = true;
            if (e.needDynamic[i])  needDynamic[i]  = true;
        }
        if (getenv("JAI_JIT_WHY")) {
            fprintf(stderr, "[jit] osr %s at %u stopped: %s\n",
                    fn->name ? fn->name->chars : "<anon>", top,
                    declineReason(&e));
        }
        jitFree(map, depths, chunkDepth, fn->chunk.count + 1);
        return false;
    }

    /* Falling off the end of the compiled range is the loop exiting there. */
    if (e.exitCount >= JIT_MAX_EXIT) { jitFree(map, depths, chunkDepth, fn->chunk.count + 1); return false; }

/* Every way out writes back what the loop was holding: the iterator's index,
 * and the locals if they were living in registers. Miss one and the
 * interpreter carries on from stale values. */
#define OSR_SYNC_ITER()                                                        \
    do {                                                                       \
        /* iterKind 3 keeps nothing in a register but the pointer, and the step \
         * has already stored the index it advanced -- so every way out of a    \
         * dict-items loop finds the ObjIter already current. */                \
        if (hasIter && iterKind != 3) {                                        \
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
        /* An ordinary way out carries no operand stack of its own, and has to
         * say so.
         *
         * `gDeopt` is one global record, on the reasoning that only one body
         * can be deoptimising at a time. That holds for a body and its own
         * stubs; it does not hold across a call. A compiled callee that
         * deopts part way through this region writes the record, is put back
         * on its feet by its own return path, and leaves `nstack` behind --
         * and the C side reads `nstack` after every way out of the region,
         * not only after a deopt. So a region that had called such a callee
         * and then finished its loop normally pushed the callee's leftover
         * operand stack onto this frame's.
         *
         * What that cost: `for n in xs` around an inner loop calling a
         * compiled method left two extra values above the enclosing loop's
         * iterator, so the next `OP_FOR_ITER_BIND` peeked one of them and
         * the loop died with "expected an iterator, not 'int'". Only under
         * `--gc-stress`, which is what made the callee deopt reliably.
         *
         * Clearing it here is the narrow fix: every way out of a region now
         * states its own record rather than inheriting one. */
        emitConst64(&e, JIT_SCRATCH_A,
                    (int64_t)(uintptr_t)&gDeopt.nstack);
        emitConst64(&e, JIT_SCRATCH_B, 0);
        emit(&e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_A, 0));
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
            unsigned reg0 = valueBankReg(&e, valueSeen);
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
            unsigned reg = valueBankReg(&e, valueSeen);
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

    if (e.failed) { jitFree(map, depths, chunkDepth, fn->chunk.count + 1); return false; }

    for (unsigned i = 0; i < e.fixupCount; i++) {
        const Fixup *f = &e.fixups[i];
        int target;
        if (f->targetOffset == FIXUP_BAIL) target = e.bailBlock;
        else if (f->targetOffset == FIXUP_THREW) target = e.exceptionExit;
        else if (f->targetOffset <= FIXUP_EXIT &&
                 f->targetOffset > FIXUP_EXIT - JIT_MAX_EXIT)
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
            if (f->targetOffset > (uint32_t)fn->chunk.count) { jitFree(map, depths, chunkDepth, fn->chunk.count + 1); return false; }
            target = map[f->targetOffset];
            if (target < 0 && getenv("JAI_JIT_WHY")) {
                fprintf(stderr, "[jit] %s stopped: a branch to %u, which is "
                                "not an instruction this compiled\n",
                        fn->name ? fn->name->chars : "<anon>", f->targetOffset);
            }
            if (target < 0 ||
                (f->depth >= 0 && depths[f->targetOffset] != f->depth)) {
                jitFree(map, depths, chunkDepth, fn->chunk.count + 1);
                return false;
            }
        }
        if (target < 0) { jitFree(map, depths, chunkDepth, fn->chunk.count + 1); return false; }
        int rel = target - f->instIndex;
        uint32_t word = e.code[f->instIndex];
        if ((word & 0xfc000000u) == 0x94000000u) e.code[f->instIndex] = jaiA64Bl(rel);
        else if (f->conditional) e.code[f->instIndex] = jaiA64BCond(word & 0xfu, rel);
        else e.code[f->instIndex] = jaiA64B(rel);
    }
    jitFree(map, depths, chunkDepth, fn->chunk.count + 1);

    if (!jaiCodeArenaUnseal(arena)) return false;
    /* Same 32-alignment as the function tier above, for the same two reasons. */
    uint8_t *entry = arenaEmit(arena, e.code, e.count);
    if (entry == NULL) return false;

    if (fn->osrCount >= JAI_OSR_MAX) return false;
    JaiOsrForm *form = &fn->osrForms[fn->osrCount];
    form->code  = entry;
    form->top   = top;
    form->slots = (uint8_t)e.locals;
    form->iterKind = iterKind;
    /* Kind in the low nibble, ListStore in the high one. See JaiOsrForm::kinds:
     * a list slot's storage was pinned when its element loads were emitted, so
     * the entry guard has to prove it the same way it proves a class. */
    for (unsigned i = 0; i < e.locals; i++) {
        /* Only a slot the emission actually believed needs proving. An
         * unpinned one was compiled at the boxed stride behind a guard of its
         * own, and demanding a storage of it here would refuse entry to a loop
         * the body would have run correctly. */
        uint8_t stg = e.localKind[i] == SLOT_LIST && e.localStgPin[i]
                          ? localStgOf(&e, i)
                          : (uint8_t)LIST_STG_ANY;
        form->kinds[i] = (uint8_t)((unsigned)e.localKind[i] | ((unsigned)stg << 4));
    }
    form->iterStg = e.elemStgPin ? e.elemStg : (uint8_t)LIST_STG_ANY;

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
        if (form->shapeCount >= jitShapeLimit()) {
            if (getenv("JAI_JIT_WHY")) {
                fprintf(stderr, "[jit] osr %s at %u stopped: more than %u "
                        "instance slots to pin\n",
                        fn->name ? fn->name->chars : "<anon>", top,
                        jitShapeLimit());
            }
            return false;
        }
        form->shapeSlot[form->shapeCount] = (uint8_t)i;
        form->shapeId[form->shapeCount]   = e.localShape[i];
        form->shapeCount++;
    }

    fn->osrCount++;
    fn->osrHot = true;
    fn->jitOsrModuleVersion = fn->module != NULL ? fn->module->version : 0;
    if (getenv("JAI_JIT_WHY")) {
        fprintf(stderr, "[jit] osr %s at %u: %u instructions iter=%u\n",
                fn->name ? fn->name->chars : "<anon>", top, e.count,
                (unsigned)iterKind);
        /* Same reason as the function tier's line: a loop that walked two
         * instructions and interpreted the other forty reported success. */
        if (e.unarmedOp != 0) {
            fprintf(stderr,
                    "[jit] osr %s at %u walked only to %s at %u -- the rest of "
                    "the loop is interpreted\n",
                    fn->name ? fn->name->chars : "<anon>", top,
                    jaiOpName((OpCode)e.unarmedOp), e.unarmedAt);
        }
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

/* Whether this form's PINNED STORAGES describe the lists in hand.
 *
 * Storage takes part in choosing a form, not merely in admitting one. A form
 * pinned to F64 is not WRONG for a boxed list -- it simply does not apply --
 * and rejecting it after it has already been selected leaves the head with
 * nothing, for ever: the search below breaks on the first (top, iterKind)
 * match, so a second form for the other storage is never compiled. One
 * jaiListBox -- a sort, a slice assignment, an int put into a `list[float]` --
 * then cost that loop its compiled form for the rest of the program. Skipping
 * the form here instead lets the head compile another one. */
static bool osrFormStorageFits(const JaiOsrForm *form, const Value *slots,
                               uint8_t iterKind, uint8_t elemStg) {
    if (iterKind == 2 && form->iterStg != LIST_STG_ANY &&
        form->iterStg != elemStg) {
        return false;
    }
    for (unsigned i = 0; i < form->slots; i++) {
        uint8_t want = (uint8_t)(form->kinds[i] >> 4);
        if (want == LIST_STG_ANY) continue;
        if ((SlotKind)(form->kinds[i] & 0x0Fu) != SLOT_LIST) continue;
        if (!IS_LIST(slots[i])) return false;
        if (AS_LIST(slots[i])->stg != want) return false;
    }
    return true;
}

int jaiJitEnterOsr(ObjClosure *closure, uint32_t top, uint32_t *resumeAt) {
    ObjFunction *fn = closure->fn;
    CallFrame *frame = &vm.frames[vm.frameCount - 1];
    /* A for-loop head keeps its iterator on the stack. Only a range from zero
     * in unit steps is taken, because that is what makes the yielded value the
     * index and lets the body run against a plain counter. */
    bool pairTop = top < (uint32_t)fn->chunk.count &&
                   fn->chunk.code[top] == OP_FOR_ITER_PAIR;
    bool hasIter = pairTop ||
                   (top < (uint32_t)fn->chunk.count &&
                    fn->chunk.code[top] == OP_FOR_ITER_BIND);
    ObjIter *iter = NULL;
    uint8_t iterKind = 0;
    Value elemSample = NULL_VAL;
    bool  elemMixed = false;
    uint8_t elemStg = (uint8_t)LIST_STORE_BOXED;
    if (hasIter) {
        if (vm.stackTop <= frame->slots) return 0;
        Value it = vm.stackTop[-1];
        if (!IS_ITER(it)) return 0;
        iter = AS_ITER(it);
        if (pairTop) {
            /* `for (k, v) in d.items()` at the top of the loop being entered.
             * Only the lazy dict view: a pair loop over a LIST of tuples has no
             * head arm, and letting it through here would enter a form compiled
             * for a dict with an ObjList in the register. The sample is the
             * dict itself -- the head arm reads the first live entry out of it
             * for the component kinds, as the list head reads items[index]. */
            if (iter->kind != ITER_DICT_ITEMS || !IS_DICT(iter->source)) {
                return 0;
            }
            elemSample = iter->source;
            iterKind = 3;
        } else if (iter->kind == ITER_RANGE && IS_RANGE(iter->source)) {
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
            elemSample = jaiListGet(src, at);
            elemStg = src->stg;
            iterKind = 2;
            /* Whether the list holds one class or several. One sample cannot
             * say, and getting it wrong is not merely a slower loop: a form
             * compiled pinned to the sampled class fails its own entry guard
             * on every later class, so the loop runs interpreted for the rest
             * of the program with no second chance to notice. Capped because
             * this runs per compile attempt and a list can be enormous; past
             * the cap the pinned form is compiled as before and deoptimises if
             * it was wrong, which is where this started. */
            if (IS_INSTANCE(elemSample)) {
                const ObjClass *first = AS_INSTANCE(elemSample)->klass;
                int scan = src->count < 1024 ? src->count : 1024;
                int nulls = 0;
                for (int i = 0; i < scan; i++) {
                    Value v = jaiListGet(src, i);
                    if (IS_NULL(v)) { nulls++; continue; }
                    if (!IS_INSTANCE(v)) continue;
                    if (AS_INSTANCE(v)->klass != first) { elemMixed = true; }
                }
                /* A `list[T?]` holding nulls beside instances. The element
                 * guard this form emits is a tag check against VAL_OBJ, so a
                 * null bails out of the compiled loop -- and the loop is
                 * re-entered on the next element, so the bail is paid per null
                 * and not once.
                 *
                 * Refusing on PRESENCE is the obvious answer and it is the
                 * wrong one; the question is DENSITY. Same probe throughout --
                 * a 2M list[Node?], twenty passes of
                 * `for x in xs { if x is null { t += 1 } }`, whole process,
                 * best of three alternating runs under scripts/gpu_lock.sh:
                 *
                 *                    no nulls   1 in 3    1 in 1000
                 *   declined            635       661        661
                 *   no refusal          223     11604        296
                 *   refuse on presence  219       661        661
                 *   refuse past 1/64    223       662        280
                 *
                 * One null in three costs 17.6x with no refusal at all, which
                 * is why a refusal has to exist. One null in a thousand is
                 * 2.36x FASTER pinned than interpreted, which is what refusing
                 * on presence throws away. The threshold keeps both ends.
                 *
                 * The proper fix is to widen the element to SLOT_MAYBE_INST so
                 * the guard accepts a null and no bail happens at all, which
                 * would retire this whole test. Until then, this.
                 *
                 * The class scan no longer stops at the first mismatch: the
                 * null count needs the whole prefix, and the cap is what
                 * bounds the cost. */
                if (nulls * 64 > scan) return 0;
            }
        } else {
            return 0;
        }
    }

    JaiOsrForm *form = NULL;
    for (unsigned i = 0; i < fn->osrCount; i++) {
        /* Kind as well as offset: a form compiled for a range head, entered with a list iterator, would read
          * ObjRange::start out of an ObjList -- `for x in cond ? xs : 0..n` is enough to arrange it. */
        if (fn->osrForms[i].top == top &&
            fn->osrForms[i].iterKind == iterKind &&
            osrFormStorageFits(&fn->osrForms[i], frame->slots, iterKind,
                               elemStg)) {
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
        /* The whole body first, then just the part before the first
         * `continue`.
         *
         * A body that reaches past a `continue` is the one worth having --
         * without it the tail runs interpreted and a loop whose `continue`
         * rarely fires is twenty times slower than the same loop written with
         * an `if`. But the tail can also hold something the tier will not
         * compile, an uncompiled callee most often, and then the whole loop
         * declines and NOTHING is compiled. tests/bench's contour follower is
         * exactly that shape. So the prefix stays as the fallback: some of the
         * loop compiled beats none of it. */
        if (!compileOsr(closure, top, frame->slots, iterKind, elemSample,
                        elemMixed, elemStg, true, false) &&
            !compileOsr(closure, top, frame->slots, iterKind, elemSample,
                        elemMixed, elemStg, true, true) &&
            !compileOsr(closure, top, frame->slots, iterKind, elemSample,
                        elemMixed, elemStg, false, false) &&
            !compileOsr(closure, top, frame->slots, iterKind, elemSample,
                        elemMixed, elemStg, false, true)) {
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
        SlotKind want = (SlotKind)(form->kinds[i] & 0x0Fu);
        if (want == SLOT_MAYBE_INST) {
            if (!IS_NULL(v) && !IS_INSTANCE(v)) return 0;
            continue;
        }
        switch (want) {
        case SLOT_INT:   if (!IS_INT(v))   return 0; break;
        case SLOT_FLOAT: if (!IS_FLOAT(v)) return 0; break;
        case SLOT_BOOL:  if (!IS_BOOL(v))  return 0; break;
        /* Storage is settled by osrFormStorageFits, which ran as part of
         * choosing this form: a body compiled against a boxed list reads a
         * Value every sixteen bytes, and the same body entered holding a
         * `list[int]` would read two elements as one tagged pair. */
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
    /* An OSR form's locals ARE the frame slots and its stub writes none, so
     * nothing here fills the mask -- clear it rather than leave the last
     * function-tier stub's behind for whoever reads the record next. */
    gDeopt.skipLocals = 0;
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
typedef JitResult (*Fn5)(int64_t, int64_t, int64_t, int64_t, int64_t);
typedef JitResult (*Fn6)(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
typedef JitResult (*Fn7)(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
typedef JitResult (*Fn8)(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);

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
    int64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
    /* Unrolled rather than a loop over `int64_t a[JIT_MAX_ARITY]` -- measured, not tidiness: an array of
     * int64 makes clang add a stack-protector prologue/epilogue and sends every argument out to the frame and back on its way to the register the call reads it from. This function sits on the path of every interpreted call into a compiled body, so both costs are paid per call. */
    if (arity > JIT_MAX_ARITY) return JAI_JIT_DECLINED;
    if (arity > 0 && !jitArgIn(closure, slotBase, 0, &a0)) return JAI_JIT_DECLINED;
    if (arity > 1 && !jitArgIn(closure, slotBase, 1, &a1)) return JAI_JIT_DECLINED;
    if (arity > 2 && !jitArgIn(closure, slotBase, 2, &a2)) return JAI_JIT_DECLINED;
    if (arity > 3 && !jitArgIn(closure, slotBase, 3, &a3)) return JAI_JIT_DECLINED;
    if (arity > 4 && !jitArgIn(closure, slotBase, 4, &a4)) return JAI_JIT_DECLINED;
    if (arity > 5 && !jitArgIn(closure, slotBase, 5, &a5)) return JAI_JIT_DECLINED;
    if (arity > 6 && !jitArgIn(closure, slotBase, 6, &a6)) return JAI_JIT_DECLINED;
    if (arity > 7 && !jitArgIn(closure, slotBase, 7, &a7)) return JAI_JIT_DECLINED;

    JitResult r;
    switch (arity) {
    case 0: r = ((Fn0)(uintptr_t)fn->jitFunc)(); break;
    case 1: r = ((Fn1)(uintptr_t)fn->jitFunc)(a0); break;
    case 2: r = ((Fn2)(uintptr_t)fn->jitFunc)(a0, a1); break;
    case 3: r = ((Fn3)(uintptr_t)fn->jitFunc)(a0, a1, a2); break;
    case 4: r = ((Fn4)(uintptr_t)fn->jitFunc)(a0, a1, a2, a3); break;
    case 5: r = ((Fn5)(uintptr_t)fn->jitFunc)(a0, a1, a2, a3, a4); break;
    case 6: r = ((Fn6)(uintptr_t)fn->jitFunc)(a0, a1, a2, a3, a4, a5); break;
    case 7: r = ((Fn7)(uintptr_t)fn->jitFunc)(a0, a1, a2, a3, a4, a5, a6); break;
    default: r = ((Fn8)(uintptr_t)fn->jitFunc)(a0, a1, a2, a3, a4, a5, a6, a7); break;
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

/* See vm.h. The same tests jaiCallFn1 makes, in the same order, up to the
 * point where the answer stops depending on the callee alone. */
void jaiPrepareFn1(Value callee, JaiPreparedFn1 *prepared) {
    prepared->callee = callee;
    prepared->flat   = false;
    if (!IS_CLOSURE(callee)) return;
    ObjClosure *closure = AS_CLOSURE(callee);
    ObjFunction *fn = closure->fn;
    if (fn->jitFunc == NULL || fn->arity != 1) return;

    unsigned nargs = fn->jitArgCount;
    if (nargs == 0 || nargs > 2) return;
    if (nargs == 2 && (SlotKind)fn->jitParamKind[1] != SLOT_CLOSURE) return;
    if (fn->module == NULL) return;
    if (fn->module->version != fn->jitFuncModuleVersion) return;
    if (vm.stack == NULL) return;
    /* jitResultOut has to be able to answer for whatever comes back; a kind it
     * would decline is not worth preparing, since every element would fall
     * through to jaiCallFn1 anyway. */
    switch ((SlotKind)fn->jitReturnKind) {
    case SLOT_INT: case SLOT_FLOAT: case SLOT_INST: case SLOT_LIST:
    case SLOT_OBJ: case SLOT_BOOL: case SLOT_NULL: case SLOT_MAYBE_INST:
        break;
    default:
        return;
    }

    prepared->closure       = closure;
    prepared->fn            = fn;
    prepared->entry         = fn->jitFunc;
    /* Fixed for the life of the VM: the stack is one allocation made at start
     * up and freed at teardown, so the room test is a compare against a
     * constant rather than two loads and an add. */
    prepared->limit         = vm.stack + JAI_STACK_MAX - 3;
    prepared->moduleVersion = fn->module->version;
    prepared->nargs         = (uint8_t)nargs;
    prepared->returnKind    = fn->jitReturnKind;
    prepared->intArg        = (SlotKind)fn->jitParamKind[0] == SLOT_INT &&
                              fn->jitArgBase == 1;
    prepared->flat          = true;
}

/* This element the long way, and then another attempt to prepare. Two
 * different states arrive here and both want the same treatment: the callee
 * has not compiled YET (the ordinary state for the first sixty-four elements,
 * since that is when the tier first looks at it), or the form prepared against
 * has been retired under the loop. Out of line so the flat path's prologue
 * stays as small as the body it is calling. */
static JAI_NOINLINE bool preparedFn1Slow(JaiPreparedFn1 *p, Value arg,
                                         Value *out) {
    Value callee = p->callee;
    bool ok = jaiCallFn1(callee, arg, out);
    jaiPrepareFn1(callee, p);
    return ok;
}

bool jaiCallPreparedFn1(JaiPreparedFn1 *p, Value arg, Value *out) {
    ObjFunction *fn = p->fn;
    /* The whole of the staleness check. `entry` is never NULL in a prepared
     * struct, so a body that bailed -- jitResultOut sets jitFunc to NULL -- is
     * caught by the same compare as a body recompiled under it. */
    if (JAI_UNLIKELY(!p->flat || fn->jitFunc != p->entry ||
                     fn->module->version != p->moduleVersion)) {
        return preparedFn1Slow(p, arg, out);
    }
    Value *base = vm.stackTop;
    if (JAI_UNLIKELY(base > p->limit)) {
        return jaiCallFn1(p->callee, arg, out);
    }

    /* The two cells that keep the closure and the argument reachable while the
     * compiled body runs; see jaiCallFn1, whose window this is. */
    base[0] = p->callee;
    base[1] = arg;
    vm.stackTop = base + 2;

    int64_t a0;
    if (JAI_LIKELY(p->intArg)) {
        if (JAI_UNLIKELY(!IS_INT(arg))) {
            vm.stackTop = base;
            return jaiCallFn1(p->callee, arg, out);
        }
        a0 = AS_INT(arg);
    } else if (JAI_UNLIKELY(!jitArgIn(p->closure, base, 0, &a0))) {
        vm.stackTop = base;
        return jaiCallFn1(p->callee, arg, out);
    }

    int frameBase = vm.frameCount;
    JitResult r =
        p->nargs == 1
            ? ((Fn1)(uintptr_t)p->entry)(a0)
            : ((Fn2)(uintptr_t)p->entry)(a0, (int64_t)(uintptr_t)p->closure);

    /* An int result needs no store into the window and no read back out of it:
     * nothing between here and the caller's use of it can collect, and an int
     * is not a root in any case. Every other kind goes the long way, which is
     * where the tag and the bail verdicts are decided. */
    if (JAI_LIKELY(r.bailed == 0 && p->returnKind == (uint8_t)SLOT_INT)) {
        *out = INT_VAL(r.value);
        vm.stackTop = base;
        return true;
    }

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
        return jaiFinishJitDeopt1(p->closure, base, frameBase, out);
    }
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
void jaiPrepareFn1(Value callee, JaiPreparedFn1 *prepared) {
    prepared->callee = callee;
    prepared->flat   = false;
}
bool jaiCallPreparedFn1(JaiPreparedFn1 *prepared, Value arg, Value *out) {
    return jaiCallValue1(prepared->callee, arg, out);
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
