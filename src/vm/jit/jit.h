#ifndef JAI_VM_JIT_H
#define JAI_VM_JIT_H

#include "vm/bytecode/chunk.h"
#include "vm/object/object.h"

/* The compiled tier: an accelerator that may always decline -- jaiJitEnter
 * answers JAI_JIT_DECLINED and the interpreter runs the function exactly as it
 * would. Any other answer means the call has begun and must not be re-run. */

/* How many entries before a function is considered hot: arbitrary, just high
 * enough the counter costs nothing on cold code and low enough a benchmark
 * reaches it. */
#define JAI_JIT_THRESHOLD 64
#define JAI_JIT_TRACE_THRESHOLD 8

static inline uint32_t jaiJitThreshold(const ObjFunction *fn) {
    if (fn != NULL && (fn->flags & FN_TRACE) != 0) return JAI_JIT_TRACE_THRESHOLD;
    return JAI_JIT_THRESHOLD;
}

/* An inline cache's observation window is meant to close before the tier first
 * asks what a site returns; a window that outlived the threshold would hand the
 * tier a half-formed record, which is the "first observation" failure this
 * whole mechanism exists to avoid. chunk.h states the intent and cannot see
 * this constant, so the two are tied here. */
_Static_assert(JAI_IC_OBS_BUDGET <= JAI_JIT_THRESHOLD,
               "an inline cache must settle before the tier reads it");
_Static_assert(JAI_IC_OBS_BUDGET_TRACE <= JAI_JIT_TRACE_THRESHOLD,
               "a traced function's caches must settle before it compiles");

/* DECLINED: nothing touched, interpreter should run the call. ERROR: compiled
 * code called out, the callee raised, and those effects already happened, so
 * the call must not be re-run. */
typedef enum {
    JAI_JIT_DECLINED,
    JAI_JIT_DONE,
    JAI_JIT_ERROR,
    /* Met a value it wasn't compiled for; unlike DECLINED cannot re-run, so
     * the interpreter resumes from the exact offset instead. Call
     * jaiJitApplyDeopt once a frame exists. */
    JAI_JIT_DEOPT
} JaiJitOutcome;

/* Called on entry to a Jaithon function once it has crossed the threshold.
 *
 * Boundary contract: JAI_JIT_DECLINED leaves vm.stackTop and the frame stack
 * untouched, and the interpreter proceeds as if this was never called.
 * JAI_JIT_DONE means the call is COMPLETE, in exactly OP_RETURN's poststate --
 * return value at `slotBase[0]`, `vm.stackTop == slotBase + 1`, frame never
 * pushed. The other two say the call has already started and must NOT be run
 * again from the top; every caller has to say what it does with them. */
JaiJitOutcome jaiJitEnter(ObjClosure *closure, Value *slotBase);

/* The whole-function tier (jit_func.c). Compile returns false for anything it
 * does not speak; enter obeys the same boundary contract as jaiJitEnter. */
bool jaiJitCompileFunc(ObjClosure *closure, Value *slotBase);

/* Populate the freshly pushed frame from the deopt record. */
bool jaiJitApplyDeopt(ObjClosure *closure, Value *slotBase);

/* Finish, in the interpreter, a compiled body that deoptimised part-way,
 * building its frame entirely from the deopt record. `*out` gets the return
 * value; false means an exception is pending. The record is a single global,
 * safe because it's consumed here, at the innermost frame, before anything
 * else can write another. */
bool jaiJitFinishDeopt(ObjClosure *closure, Value *out);

/* Mark the roots of every compiled frame that has linked itself. */
void jaiJitMarkFrames(void);

/* Compile and enter the loop at `top` with the interpreter's own slots. On
 * success `*resumeAt` is the bytecode offset the interpreter should continue
 * from, and any operand-stack values the loop was holding have been pushed.
 * Returns 0 declined, 1 resume at *resumeAt, 2 an exception is pending. */
int jaiJitEnterOsr(ObjClosure *closure, uint32_t top, uint32_t *resumeAt);
JaiJitOutcome jaiJitEnterFunc(ObjClosure *closure, Value *slotBase);

/* Start the sampling timer, if the tier is on. Safe to call more than once. */
void jaiJitStartSampling(void);

/* A sampling tick landed while `closure` was executing at `offset`. Rides the
 * interpreter's existing back-edge safepoint rather than counting back edges,
 * which cost 4.7%-4.7x depending on counter placement (11% even switched off,
 * from the branch alone) and punishes tight loops for being tight.
 * False when the compiled loop left an exception pending. */
bool jaiJitSample(ObjClosure *closure, uint32_t offset);

/* Enter a compiled loop at `targetOffset`, compiling it first if this is the
 * first hot tick to land there. See jit_loop.c for the contract on `ip`. */
bool jaiJitEnterLoop(ObjClosure *closure, uint32_t targetOffset);

/* Whether the tier is enabled at all. JAITHON_NO_JIT=1 turns it off, so a
 * measurement can be taken against the interpreter without rebuilding. */
bool jaiJitEnabled(void);

/* ------------------------------------------------------------------ */
/* Executable memory                                                    */
/* ------------------------------------------------------------------ */

/* A page of code, written then sealed. arm64 forbids writable+executable at
 * once, so the arena is mapped RW, filled, then flipped to RX; an unsigned
 * binary can do this with plain mmap/mprotect, no MAP_JIT or entitlement
 * needed. Instruction cache must be invalidated after writing, or arm64's
 * incoherent I/D caches fetch stale code -- looks like random, undebuggable
 * corruption. */
typedef struct {
    uint8_t *code;      /* base of the mapping */
    size_t   capacity;
    size_t   used;
    bool     sealed;    /* true once flipped to RX; writing after this is a bug */
} JaiCodeArena;

/* Reserve `capacity` bytes of writable memory. False when the map fails. */
bool jaiCodeArenaInit(JaiCodeArena *arena, size_t capacity);
/* Append `length` bytes, returning where they landed, or NULL when full or
 * sealed. */
uint8_t *jaiCodeArenaWrite(JaiCodeArena *arena, const void *bytes, size_t length);
/* Flip to read-execute and invalidate the instruction cache. */
bool jaiCodeArenaSeal(JaiCodeArena *arena);
/* Make a sealed arena writable again so another function can be added. */
bool jaiCodeArenaUnseal(JaiCodeArena *arena);
void jaiCodeArenaFree(JaiCodeArena *arena);

/* The one arena compiled code lives in. Never freed: a compiled function is
 * reachable for the life of the process. */
JaiCodeArena *jaiJitArena(void);

#endif /* JAI_VM_JIT_H */
