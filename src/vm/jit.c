#include "jit.h"

#include "vm.h"

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

/* The one arena every compiled function lives in. Never freed: a compiled
 * function is reachable for the life of the process, and reclaiming code while
 * a frame might return into it is a whole problem this tier does not have yet.
 */
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
 * `OP_RETURN_NULL`.
 *
 * Useless as an optimisation -- such a function is not hot and returning null
 * is not slow. That is deliberate. It is the smallest thing that proves the
 * hard part: that generated code can be entered from the interpreter, run, and
 * returned from, leaving the stack exactly as OP_RETURN would. Everything after
 * this widens the set of opcodes; nothing after this changes the mechanism. */
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
    /* Sealing the whole arena after one function is wasteful and temporary: a
     * real tier writes many functions and seals in batches. It is correct,
     * which is what this step is for. */
    if (!jaiCodeArenaSeal(&sArena)) return false;
    fn->jitCode = entry;
    fn->jitKind = 0;   /* returns null; ignores its arguments */
    return true;
}

/* `OP_GET_LOCAL k; OP_RETURN` -- an accessor, and the first compiled body that
 * has to move data.
 *
 * It cannot fail: reading a slot throws nothing and overflows nothing, so this
 * needs no deopt path and settles the calling convention while the hard part is
 * still out of scope. The convention is the narrowest one that works: the slot
 * base arrives in x0, the result is written to slotBase[0], and nothing is
 * returned -- the caller already knows where to look.
 *
 * A Value is 16 bytes, so slot k sits at x0 + k*16 and moves in one ldp/stp
 * pair. LDP's immediate is scaled by 8 and signed 7-bit, which caps k at 31;
 * beyond that this declines rather than encoding a second form, because a
 * 32-slot accessor is not the case worth the extra encoding. */
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

    if (fn->jitCode == NULL) {
        if (!compileReturnNull(fn) && !compileAccessor(fn)) return false;
    }

    if (fn->jitKind == 1) {
        ((JaiCompiledAccessor)fn->jitCode)(slotBase);
    } else {
        (void)((JaiCompiledFn)fn->jitCode)();
        slotBase[0] = NULL_VAL;
    }

    /* The contract in jit.h: the call is complete, the frame was never pushed,
     * and the result sits at slotBase[0]. */
    vm.stackTop = slotBase + 1;
    return true;
}

/* ------------------------------------------------------------------ */
/* Executable memory                                                    */
/* ------------------------------------------------------------------ */

bool jaiCodeArenaInit(JaiCodeArena *arena, size_t capacity) {
    memset(arena, 0, sizeof *arena);
    void *p = mmap(NULL, capacity, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANON, -1, 0);
    if (p == MAP_FAILED) return false;
    arena->code = (uint8_t *)p;
    arena->capacity = capacity;
    return true;
}

uint8_t *jaiCodeArenaWrite(JaiCodeArena *arena, const void *bytes,
                           size_t length) {
    if (arena->sealed || arena->used + length > arena->capacity) return NULL;
    uint8_t *at = arena->code + arena->used;
    memcpy(at, bytes, length);
    arena->used += length;
    return at;
}

bool jaiCodeArenaSeal(JaiCodeArena *arena) {
    if (arena->sealed) return true;
    if (mprotect(arena->code, arena->capacity, PROT_READ | PROT_EXEC) != 0) {
        return false;
    }
    /* Not optional on arm64: the data and instruction caches are not coherent,
     * so without this the CPU can fetch whatever was in the line before. */
#if defined(__APPLE__)
    sys_icache_invalidate(arena->code, arena->used);
#elif defined(__GNUC__)
    __builtin___clear_cache((char *)arena->code, (char *)arena->code + arena->used);
#endif
    arena->sealed = true;
    return true;
}

void jaiCodeArenaFree(JaiCodeArena *arena) {
    if (arena->code != NULL) munmap(arena->code, arena->capacity);
    memset(arena, 0, sizeof *arena);
}

/* ------------------------------------------------------------------ */
/* Sampling                                                             */
/* ------------------------------------------------------------------ */

/* Set by the timer, read by the interpreter's safepoint. `2` rather than `1`
 * so the existing `sInterrupted` test fires without a new branch: 1 stays
 * Ctrl-C and throws, 2 is a tick. */
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

    /* ITIMER_PROF counts CPU time, so a process blocked on IO is not sampled --
     * which is right, since it has no hot loop to find. */
    struct itimerval it;
    it.it_interval.tv_sec = 0;
    it.it_interval.tv_usec = 1000;   /* 1kHz */
    it.it_value = it.it_interval;
    (void)setitimer(ITIMER_PROF, &it, NULL);
}

/* Ticks seen in one function. A loop that is hot collects them; a function that
 * merely runs once does not. */
#define JAI_JIT_HOT_TICKS 20

void jaiJitSample(ObjClosure *closure, uint32_t offset) {
    ObjFunction *fn = closure->fn;
    if (fn->tickCount < JAI_JIT_HOT_TICKS) {
        fn->tickCount++;
        if (fn->tickCount == JAI_JIT_HOT_TICKS && getenv("JAI_JIT_TRACE")) {
            fprintf(stderr, "[jit] hot: %s at offset %u\n",
                    fn->name ? fn->name->chars : "<anon>", offset);
        }
    }
}
