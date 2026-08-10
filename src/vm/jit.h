#ifndef JAI_VM_JIT_H
#define JAI_VM_JIT_H

#include "object.h"

/* The compiled tier.
 *
 * The interpreter stays authoritative. The JIT is an accelerator that may
 * always decline: `jaiJitEnter` answers false and the interpreter runs the
 * function exactly as it would have. Every stage of this is built that way, so
 * a tier that is wrong is slow rather than fatal, and so the suite can stay
 * green while the compiler behind it is still a stub.
 *
 * Nothing here compiles anything yet. This is the plumbing -- the entry count,
 * the threshold, the decline path -- proved on its own before a byte of machine
 * code exists. */

/* How many entries before a function is considered hot. Arbitrary for now; the
 * only requirement is that it is high enough that the counter itself does not
 * cost anything on cold code and low enough that a benchmark reaches it. */
#define JAI_JIT_THRESHOLD 64

/* Called on entry to a Jaithon function once it has crossed the threshold.
 *
 * THE BOUNDARY CONTRACT, which is the whole safety argument:
 *
 *   On false -- nothing has been touched. `vm.stackTop` and the frame stack are
 *   exactly as they were, and the interpreter proceeds as if this was never
 *   called. Declining must be free of side effects, because it is the path
 *   every unsupported function takes.
 *
 *   On true -- the call is COMPLETE. The callee's frame was never pushed, the
 *   arguments and callee slot are gone, and the single return value sits at
 *   `slotBase[0]` with `vm.stackTop == slotBase + 1`. That is precisely the
 *   state `OP_RETURN` leaves behind, so the caller cannot tell which tier ran.
 *
 * Why the frame is never pushed: a compiled region has no `CallFrame`, so a
 * traceback taken inside it would show the caller's frame and nothing else.
 * That is acceptable only because compiled regions cannot yet throw, call, or
 * allocate. The moment one of those becomes possible, this contract needs a
 * frame -- or a way to reconstruct one -- and that is the next hard problem
 * rather than a detail.
 *
 * The interpreter caches `ip` and `stackTop` in locals across an instruction
 * (`SAVE_STATE`/`LOAD_STATE` in vm.c). `callClosure` is called from a point
 * where that state is already saved, so this runs against memory that is
 * current -- but anything here that re-enters the VM has to save and restore it
 * the same way. */
bool jaiJitEnter(ObjClosure *closure, Value *slotBase);

/* The whole-function tier (jit_func.c). Compile returns false for anything it
 * does not speak; enter obeys the same boundary contract as jaiJitEnter. */
bool jaiJitCompileFunc(ObjClosure *closure, Value *slotBase);

/* Three answers, not two. DECLINED means nothing was touched and the
 * interpreter should run the call. ERROR means the compiled code called out,
 * the callee raised, and the effects up to that point already happened -- so
 * running the call again would repeat them. */
typedef enum {
    JAI_JIT_DECLINED,
    JAI_JIT_DONE,
    JAI_JIT_ERROR,
    /* The compiled body met a value that was not the kind it was compiled for.
     * Unlike DECLINED this cannot re-run the call: the body may already have
     * written something. The interpreter takes over from the exact bytecode
     * offset instead, with the locals and operand stack the compiled code was
     * holding. Call jaiJitApplyDeopt once a frame exists. */
    JAI_JIT_DEOPT
} JaiJitOutcome;

/* Populate the freshly pushed frame from the deopt record. */
bool jaiJitApplyDeopt(ObjClosure *closure, Value *slotBase);
JaiJitOutcome jaiJitEnterFunc(ObjClosure *closure, Value *slotBase);

/* Start the sampling timer, if the tier is on. Safe to call more than once. */
void jaiJitStartSampling(void);

/* A sampling tick landed while `closure` was executing at `offset`.
 *
 * Called from the interpreter's existing safepoint, which already runs on the
 * back edge, so noticing a hot loop costs nothing on the common path. Counting
 * back edges instead cost between 4.7% and 4.7x depending on where the counter
 * lived -- and 11% even with the counter switched off, purely from the branch
 * existing in OP_LOOP. There is no budget for a new test on that edge.
 *
 * Sampling has a second property the counter lacked: its cost is proportional
 * to wall time rather than to iterations, so a tight loop is not punished for
 * being tight. */
void jaiJitSample(ObjClosure *closure, uint32_t offset);

/* Enter a compiled loop at `targetOffset`, compiling it first if this is the
 * first hot tick to land there. See jit_loop.c for the contract on `ip`. */
bool jaiJitEnterLoop(ObjClosure *closure, uint32_t targetOffset);

/* Whether the tier is enabled at all. JAITHON_NO_JIT=1 turns it off, so a
 * measurement can be taken against the interpreter without rebuilding. */
bool jaiJitEnabled(void);

/* ------------------------------------------------------------------ */
/* Executable memory                                                    */
/* ------------------------------------------------------------------ */

/* A page of code, written then sealed.
 *
 * arm64 will not let a page be writable and executable at once, so the arena is
 * mapped RW, filled, and flipped to RX before anything jumps into it. Measured
 * on this machine: an unsigned binary can do that with plain mmap and mprotect;
 * neither MAP_JIT nor the allow-jit entitlement is needed, which is worth
 * knowing because the alternative would have meant codesigning every build.
 *
 * The instruction cache must be invalidated after writing. On arm64 the data
 * and instruction caches are not coherent, so code that was just stored is not
 * necessarily what gets fetched -- this is the failure that looks like random
 * corruption and is not reproducible under a debugger. */
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
 * reachable for the life of the process, and reclaiming code while a frame
 * might return into it is a problem this tier does not have yet. */
JaiCodeArena *jaiJitArena(void);

#endif /* JAI_VM_JIT_H */
