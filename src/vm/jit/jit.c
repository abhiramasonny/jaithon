#include "vm/jit/jit.h"

#include "vm/vm.h"

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/mman.h>

#if defined(__APPLE__)
#  include <libkern/OSCacheControl.h>
#endif

bool jaiJitEnabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *off = getenv("JAITHON_NO_JIT");
        cached = (off != NULL && off[0] != '\0' && strcmp(off, "0") != 0) ? 0 : 1;
    }
    return cached != 0;
}

/* The one arena every compiled function lives in. Never freed. */
static JaiCodeArena sArena;
static bool sArenaReady;

static bool arenaReady(void) {
    if (!sArenaReady) {
        if (!jaiCodeArenaInit(&sArena, 1u << 20)) return false;
        sArenaReady = true;
    }
    return true;
}

/* The compiled tier's entire repertoire, for now: a body that is exactly
 * `OP_RETURN_NULL`. Useless as an optimisation, deliberately -- it's the
 * smallest proof that generated code can be entered, run, and returned from
 * leaving the stack exactly as OP_RETURN would. */
static bool compileReturnNull(ObjFunction *fn) {
    if (fn->chunk.count != 1 || fn->chunk.code[0] != OP_RETURN_NULL) return false;
    if (!arenaReady() || sArena.sealed) return false;

#if defined(__aarch64__) || defined(__arm64__)
    /* mov w0, #0 ; ret -- the return value is carried by the C caller below,
     * so the generated code only has to come back. */
    const uint32_t code[] = { 0x52800000u, 0xd65f03c0u };
#elif defined(__x86_64__)
    /* xor eax, eax ; ret */
    const uint8_t code[] = { 0x31, 0xc0, 0xc3 };
#else
    return false;
#endif

    uint8_t *entry = jaiCodeArenaWrite(&sArena, code, sizeof code);
    if (entry == NULL) return false;
    /* Sealing the whole arena per function is wasteful and temporary: a real
     * tier writes many functions and seals in batches. */
    if (!jaiCodeArenaSeal(&sArena)) return false;
    fn->jitCode = entry;
    fn->jitKind = 0;   /* returns null; ignores its arguments */
    return true;
}

/* `OP_GET_LOCAL k; OP_RETURN` -- an accessor, and the first compiled body that
 * has to move data. Cannot fail (reading a slot throws/overflows nothing), so
 * this settles the calling convention (slot base in x0, result written to
 * slotBase[0]) with no deopt path needed. A Value is 16 bytes, moved in one
 * ldp/stp pair; LDP's imm7 caps slot k at 31, beyond which this declines. */
static bool compileAccessor(ObjFunction *fn) {
    const Chunk *c = &fn->chunk;
    if (c->count != 4) return false;
    if (c->code[0] != OP_GET_LOCAL || c->code[3] != OP_RETURN) return false;

    unsigned slot = (unsigned)c->code[1] | ((unsigned)c->code[2] << 8);
    if (slot == 0 || slot > 31) return false;   /* see the imm7 note above */
    if (!arenaReady() || sArena.sealed) return false;

#if defined(__aarch64__) || defined(__arm64__)
    uint32_t imm7 = (uint32_t)(slot * 2);       /* (slot * 16) / 8 */
    const uint32_t code[] = {
        /* ldp x9, x10, [x0, #slot*16] */
        0xa9400000u | (imm7 << 15) | (10u << 10) | (0u << 5) | 9u,
        /* stp x9, x10, [x0]          */
        0xa9000000u | (10u << 10) | (0u << 5) | 9u,
        /* ret                        */
        0xd65f03c0u,
    };
#else
    return false;   /* no stencil for this architecture yet */
#endif

    uint8_t *entry = jaiCodeArenaWrite(&sArena, code, sizeof code);
    if (entry == NULL) return false;
    if (!jaiCodeArenaSeal(&sArena)) return false;
    fn->jitCode = entry;
    fn->jitKind = 1;   /* accessor: takes the slot base */
    return true;
}

typedef int (*JaiCompiledFn)(void);
typedef void (*JaiCompiledAccessor)(Value *slotBase);

bool jaiJitEnter(ObjClosure *closure, Value *slotBase) {
    ObjFunction *fn = closure->fn;

    /* Whole-function tier first: the only one that makes a hot function
     * meaningfully faster, and it declines quickly otherwise. */
    if (fn->jitFunc != NULL) return jaiJitEnterFunc(closure, slotBase) == JAI_JIT_DONE;

    if (fn->jitCode == NULL) {
        if (jaiJitCompileFunc(closure, slotBase)) {
            fn->jitFuncModuleVersion = fn->module->version;
            return jaiJitEnterFunc(closure, slotBase) == JAI_JIT_DONE;
        }
        if (!compileReturnNull(fn) && !compileAccessor(fn)) {
            /* Same reasoning as the loop tier: a body that calls something not
             * yet compiled may compile perfectly well a moment later. */
            if (++fn->jitAttempts >= 5) fn->jitRefused = true;
            else fn->entryCount = 0;
            return false;
        }
    }

    if (fn->jitKind == 1) {
        ((JaiCompiledAccessor)(uintptr_t)fn->jitCode)(slotBase);
    } else {
        (void)((JaiCompiledFn)(uintptr_t)fn->jitCode)();
        slotBase[0] = NULL_VAL;
    }

    /* The contract in jit.h: the call is complete, the frame was never pushed,
     * and the result sits at slotBase[0]. */
    vm.stackTop = slotBase + 1;
    return true;
}

/* ------------------------------------------------------------------ */
/* Sampling                                                             */
/* ------------------------------------------------------------------ */

/* Set by the timer, read by the interpreter's safepoint. `2` rather than `1`
 * so the existing test fires without a new branch: 1 stays Ctrl-C, 2 is a tick. */
extern volatile sig_atomic_t jaiInterrupted;

static void onTick(int signum) {
    (void)signum;
    /* Nothing but the flag. A signal handler that touched VM state or
     * allocated would be a reentrancy bug that reproduces once a month. */
    if (jaiInterrupted == 0) jaiInterrupted = 2;
}

void jaiJitStartSampling(void) {
    static bool started;
    if (started || !jaiJitEnabled()) return;
    started = true;

    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = onTick;
    sa.sa_flags = SA_RESTART;
    if (sigaction(SIGPROF, &sa, NULL) != 0) return;

    /* ITIMER_PROF counts CPU time, so a process blocked on IO isn't sampled --
     * right, since it has no hot loop to find. */
    struct itimerval it;
    it.it_interval.tv_sec = 0;
    /* 4kHz: a tick only counts on a back-edge ip, so raising the rate directly
     * shortens the wait before compiling; 10kHz measured worse (signal cost
     * outweighs the earlier compile), so this is a peak, not a slope. */
    it.it_interval.tv_usec = 250;
    it.it_value = it.it_interval;
    (void)setitimer(ITIMER_PROF, &it, NULL);
}

/* Ticks seen in one function before its loop is compiled. Was 20, then 3;
 * lowering kept winning benchmarks because a tick only counts on a back-edge
 * ip, so the real wait is longer than the rate suggests -- and compiling
 * costs microseconds, so there's little to gain by waiting longer. */
#define JAI_JIT_HOT_TICKS 1

bool jaiJitSample(ObjClosure *closure, uint32_t offset) {
    ObjFunction *fn = closure->fn;
    if (fn->tickCount >= JAI_JIT_HOT_TICKS) {
        /* Already hot. A tick on a loop top is the OSR entry point, the only
         * moment interpreter state matches what compiled code expects: OP_LOOP
         * sets ip to the loop's first instruction, *then* runs the safepoint. */
        if (jaiJitEnterLoop(closure, offset)) return true;
        /* The shape matcher covers one loop; this covers the rest, with the
         * interpreter's own slots as the compiled body's locals. */
        uint32_t resumeAt = 0;
        int outcome = jaiJitEnterOsr(closure, offset, &resumeAt);
        if (outcome == 2) return false;
        if (outcome == 1) {
            CallFrame *frame = &vm.frames[vm.frameCount - 1];
            frame->ip = fn->chunk.code + resumeAt;
        }
        return true;
    }
    if (fn->tickCount < JAI_JIT_HOT_TICKS) {
        fn->tickCount++;
        if (fn->tickCount == JAI_JIT_HOT_TICKS && getenv("JAI_JIT_TRACE")) {
            fprintf(stderr, "[jit] hot: %s at offset %u\n",
                    fn->name ? fn->name->chars : "<anon>", offset);
        }
    }
    return true;
}
