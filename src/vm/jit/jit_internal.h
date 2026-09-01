/* jit_internal.h -- types, constants and cross-file declarations shared by the
 * whole-function JIT tier's translation units (jit_func.c and its siblings). */
#ifndef JAI_VM_JIT_INTERNAL_H
#define JAI_VM_JIT_INTERNAL_H

#include "vm/jit/jit.h"
#include "vm/vm.h"

#include <stdbool.h>
#include <stdint.h>

#if (defined(__aarch64__) || defined(__arm64__))

/* Two words in x0/x1 per AAPCS64 struct return (no store/load like the old global flag). A bailed
 * callee sends its caller straight to its own bail block, so recursion unwinds in one shot. */
typedef struct { int64_t value; int64_t bailed; } JitResult;

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
 * what is used.
 *
 * Then 8 was not enough either, and the way that surfaced is worth keeping.
 * Teaching OP_GET_GLOBAL to resolve the `__prim__` namespace made jaicv's
 * `span` twice as slow, because `__prim__.fill_span(...)` takes TEN arguments
 * and `fill_convex` eleven. While `__prim__` had no arm the walk took the SOFT
 * unarmed path and skipped the receiver, all ten pushes and the invoke as one
 * block, so span's prologue compiled; once it resolved, the walk reached this
 * cap, which refuses HARD and declined the whole function.
 *
 * Two agents tried to dodge the wall with a lookahead that only resolves
 * `__prim__` when the paired invoke would fit. That was refuted: it finds the
 * FIRST invoke, not the paired one, so an argument containing its own method
 * call walks straight back into the wall. Removing the wall is the fix.
 *
 * 12 covers `fill_convex`'s eleven. The price is 4 more Values (64 bytes) on
 * a calling body's frame, and it carries that frame past the 504 bytes that
 * keep the cheap stp-pre prologue -- which JIT_MAX_ROOTS above already does,
 * for the same reason and at the same price: framePairFits() falls back to one
 * or two extra instructions once per body, against a whole body compiling. */
#define JIT_MAX_ARGS_OUT 12

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


/* Defined in jit_runtime.c. */
extern JitCallDesc *gJitFrames;
extern JitDeoptRecord gDeopt;

void jitThrowOverflow(int64_t which);
int jitInvokeMethod(JitCallDesc *d);
int jitInvokeByName(JitCallDesc *d);
int jitInvokeNative(JitCallDesc *d);
int jitBuildList(JitCallDesc *d);
int jitContains(JitCallDesc *d);
int jitNotContains(JitCallDesc *d);
int jitBuildDict(JitCallDesc *d);
int jitBuildSet(JitCallDesc *d);
int jitBuildTuple(JitCallDesc *d);
int jitValuesEqual(JitCallDesc *d);
bool jitObjEquality(void);
int jitStringConcat(JitCallDesc *d);
int jitMakeRangeIter(JitCallDesc *d);
int jitMakeIter(JitCallDesc *d);
int jitMakeItemsIter(JitCallDesc *d);
int jitFormat(JitCallDesc *d);
int jitListGrow(ObjList *list, uint64_t tag, int64_t payload);
ObjInstance *jitInstanceAlloc(ObjClass *cls);
int jitNewInstance(JitCallDesc *d);
int jitGetSlice(JitCallDesc *d);
int jitGetIndexDict(JitCallDesc *d);
int jitSetIndexDict(JitCallDesc *d);
int jitCallOut(JitCallDesc *d);

#endif /* arm64 */

#endif /* JAI_VM_JIT_INTERNAL_H */
