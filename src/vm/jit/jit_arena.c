#include "vm/jit/jit.h"

#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#if defined(__APPLE__)
#  include <libkern/OSCacheControl.h>
#endif

/* The code arena, split out of the tier so a test can link it without the VM.
 * Nothing here touches interpreter state, which is what makes that possible. */

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

/* Back to writable, so a second function can be compiled into the same arena.
 * Without this the first seal froze the tier for the life of the process:
 * every later compile found `sealed` and declined. */
bool jaiCodeArenaUnseal(JaiCodeArena *arena) {
    if (!arena->sealed) return true;
    if (mprotect(arena->code, arena->capacity, PROT_READ | PROT_WRITE) != 0) {
        return false;
    }
    arena->sealed = false;
    return true;
}

void jaiCodeArenaFree(JaiCodeArena *arena) {
    if (arena->code != NULL) munmap(arena->code, arena->capacity);
    memset(arena, 0, sizeof *arena);
}


/* Four mebibytes, and the previous one was a guess that cost more than anything
 * else the tier refuses for.
 *
 * On `check --no-cache lib/jaithon` -- the compiler's own source -- a 1 MB
 * arena declined SEVENTY distinct bodies with "the code arena is full", making
 * it the single largest refusal cause in the tier, ahead of every missing
 * opcode arm. At 4 MB that count is ZERO and 301 compiled bodies become 369.
 * Measured with scripts/dev/ab.py: -13.3%..-12.5% interpreted instructions
 * across three interleaved pairs against a 4.75% noise floor.
 *
 * It went unmeasured because there are TWO arenas of one mebibyte -- this one,
 * which the function and OSR tiers write into, and a second in jit.c for the
 * fallback forms -- and sizing only one measures nothing. A first sweep resized
 * jit.c's and concluded capacity was irrelevant. They share this now so an A/B
 * moves both.
 *
 * The mapping is lazy, so the cost of the headroom is address space rather than
 * resident pages. Clamped: under a mebibyte is not worth testing, over 64 is a
 * mapping large enough to be its own problem. */
size_t jaiCodeArenaDefaultCapacity(void) {
    static size_t cached;
    if (cached == 0) {
        unsigned mb = 4u;
        const char *v = getenv("JAITHON_JIT_ARENA_MB");
        if (v != NULL) {
            long n = strtol(v, NULL, 10);
            if (n >= 1 && n <= 64) mb = (unsigned)n;
        }
        cached = (size_t)mb << 20;
    }
    return cached;
}

JaiCodeArena *jaiJitArena(void) {
    static JaiCodeArena arena;
    static bool tried;
    if (!tried) {
        tried = true;
        if (!jaiCodeArenaInit(&arena, jaiCodeArenaDefaultCapacity())) return NULL;
    }
    return arena.code != NULL ? &arena : NULL;
}
