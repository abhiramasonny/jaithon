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

/* Non-zero replaces both thresholds, from JAITHON_JIT_THRESHOLD.
 *
 * It exists for TESTING, not tuning. Most programs never call anything 64
 * times, so most of the corpus never reaches the compiled tier at all: 40 of
 * the 61 programs in tests/golden compile nothing, which means a differential
 * against JAITHON_NO_JIT=1 over them compares the interpreter with itself.
 * That is how three silent JIT bugs shipped -- one of them made
 * `std.fmt.green()` return uncoloured text from call 65 onward, invisible to
 * every test because no test called it 65 times.
 *
 * Setting it to 1 makes every body compile on first call, so the whole suite
 * becomes a JIT test. Results must be UNCHANGED: a lower threshold hands the
 * tier a half-formed inline cache (JAI_IC_OBS_BUDGET no longer settles first),
 * which costs prediction quality, and every prediction is guarded. A
 * difference under this switch is a miscompile, which is the point. */
extern uint32_t jaiJitThresholdOverride;

/* How many times a body may be recompiled because the callee its walk stopped
 * at has since compiled. See ObjFunction::jitBlockedOn.
 *
 * This is NOT the budget the comment in jaiJitEnter warns about, and the
 * difference is worth stating because the two look identical from a distance.
 * That one is jitAttempts, and it counts DECLINES: raising it from 5 to 40 spent
 * thirty-five more compiles each on bodies that fail for reasons which do not
 * resolve, and cost 20% wall. This one counts retries of bodies that SUCCEEDED
 * -- partially -- and only ever fires when the specific named callee that
 * truncated the walk has a compiled form it did not have before. Measured on
 * `check --no-cache lib/std`: it fires 20 times in a run, across 20 distinct
 * bodies, and 13 of those compile further.
 *
 * Two is what the workloads need. Every body on `check lib/std` retries exactly
 * once; `_scan_token` on `check lib/jaithon` needs two, on two DIFFERENT
 * callees, and capping at one there loses a third of the win. Nothing observed
 * wants a third, and the cap is also the backstop that stops a cycle:
 * jaiJitEnterFunc refuses an immediate repeat of the same pair, and at two
 * retries a longer cycle cannot close. */
#define JAI_JIT_RECOMPILES 2
/* Declines that name a callee which has NOT RETURNED YET are a matter of time,
 * not of kind: the caller crossed its threshold first. Such a decline does not
 * spend one of the five attempts; it resets the entry count and is looked at
 * again sixty-four calls later, this many times at most. Each retry is one
 * measuring pass that stops at the same call, so the cap prices the worst
 * case -- a callee that never runs -- at sixteen short walks. */
#define JAI_JIT_COLD_RETRIES 16
/* Set by the compile when its LAST decline was of that shape; read by
 * jaiJitEnter, which owns the attempt budget. */
extern bool gJitColdDecline;
bool jaiJitColdRetryOn(void);

static inline uint32_t jaiJitThreshold(const ObjFunction *fn) {
    if (jaiJitThresholdOverride != 0) return jaiJitThresholdOverride;
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
    /* The arena is append-only, so everything below `windowFrom` is finished
     * code that never changes again. Unsealing flips only [windowFrom,
     * capacity) and sealing flips exactly that range back, instead of the
     * whole mapping -- which matters because the mapping is now four
     * mebibytes and the flip happens on every single compile.
     *
     * `dirtyFrom` is the same idea for the instruction cache: arm64's I and D
     * caches are not coherent, so written code must be invalidated, but only
     * the bytes actually written. Invalidating [code, used) on every seal made
     * that O(total code emitted so far) per compile -- quadratic across a run,
     * and paid 400+ times on `check lib/std` alone. */
    size_t   windowFrom;
    size_t   dirtyFrom;
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

/* "module.function", for every diagnostic the tier prints. The bare name is
 * ambiguous -- `check lib/std` has three hot functions called `init` -- and
 * qualifiedName does not help, because a cached image stores none. */
const char *jitFnLabel(const ObjFunction *fn);

/* The capacity both arenas are built at, JAITHON_JIT_ARENA_MB or one mebibyte. */
size_t jaiCodeArenaDefaultCapacity(void);

#endif /* JAI_VM_JIT_H */
