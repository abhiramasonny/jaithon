#include "jit.h"

#include <stdlib.h>
#include <string.h>
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

/* Always declines. The interpreter then runs the function, which is what makes
 * this safe to land: the counter and the threshold are exercised by the whole
 * test suite while the answer is still produced entirely by the interpreter. */
bool jaiJitEnter(ObjClosure *closure, Value *slotBase) {
    (void)closure;
    (void)slotBase;
    return false;
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
