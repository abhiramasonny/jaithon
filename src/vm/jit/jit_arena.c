#include "vm/jit/jit.h"

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


JaiCodeArena *jaiJitArena(void) {
    static JaiCodeArena arena;
    static bool tried;
    if (!tried) {
        tried = true;
        if (!jaiCodeArenaInit(&arena, 1u << 20)) return NULL;
    }
    return arena.code != NULL ? &arena : NULL;
}
