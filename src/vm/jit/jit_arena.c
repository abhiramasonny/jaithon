#include "vm/jit/jit.h"

#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

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

/* Whether unseal/seal and the instruction-cache invalidation are limited to
 * the range that changed. Off restores the whole-mapping flip and the
 * invalidate-everything-written-so-far behaviour.
 *
 * MEASURED, and it is not a speedup: five interleaved pairs on
 * `check --no-cache lib/jaithon` disagreed in sign. Kept anyway, for two
 * reasons that are about shape rather than this workload -- the invalidation
 * becomes O(bytes written) instead of O(all code emitted so far), which is the
 * difference between linear and quadratic across a long run, and a window that
 * seals exactly what it unsealed cannot leave the back catalogue
 * unexecutable. The switch exists so the first claim stays checkable. */
static bool arenaWindowOn(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_ARENA_WINDOW");
        cached = (v != NULL && strcmp(v, "0") == 0) ? 0 : 1;
    }
    return cached != 0;
}

static size_t pageFloor(size_t n) {
    long page = sysconf(_SC_PAGESIZE);
    size_t p = (page > 0) ? (size_t)page : 4096u;
    return n & ~(p - 1u);
}

bool jaiCodeArenaSeal(JaiCodeArena *arena) {
    if (arena->sealed) return true;
    size_t from = arenaWindowOn() ? arena->windowFrom : 0;
    if (from > arena->capacity) from = 0;
    if (mprotect(arena->code + from, arena->capacity - from,
                 PROT_READ | PROT_EXEC) != 0) {
        return false;
    }
    /* Not optional on arm64: the data and instruction caches are not coherent,
     * so without this the CPU can fetch whatever was in the line before. Only
     * the range actually written since the last seal needs it -- everything
     * below was invalidated when IT was sealed, and the arena never rewrites
     * it. */
    size_t dirty = (arenaWindowOn() && arena->dirtyFrom <= arena->used)
                       ? arena->dirtyFrom : 0;
    size_t length = arena->used - dirty;
    if (length > 0) {
#if defined(__APPLE__)
        sys_icache_invalidate(arena->code + dirty, length);
#elif defined(__GNUC__)
        __builtin___clear_cache((char *)arena->code + dirty,
                                (char *)arena->code + arena->used);
#endif
    }
    arena->sealed = true;
    return true;
}

/* Back to writable, so a second function can be compiled into the same arena.
 * Without this the first seal froze the tier for the life of the process:
 * every later compile found `sealed` and declined. */
bool jaiCodeArenaUnseal(JaiCodeArena *arena) {
    if (!arena->sealed) return true;
    /* From the page `used` lands in, not from the base. The tail of the last
     * body shares that page and loses its execute bit for the duration, which
     * is sound because nothing compiled runs between an unseal and its seal --
     * the window is inside one compile, and a compile calls no compiled code. */
    size_t from = arenaWindowOn() ? pageFloor(arena->used) : 0;
    if (mprotect(arena->code + from, arena->capacity - from,
                 PROT_READ | PROT_WRITE) != 0) {
        return false;
    }
    arena->windowFrom = from;
    arena->dirtyFrom  = arena->used;
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
