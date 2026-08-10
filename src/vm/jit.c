#include "jit.h"

#include "vm.h"

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
    return true;
}

typedef int (*JaiCompiledFn)(void);

bool jaiJitEnter(ObjClosure *closure, Value *slotBase) {
    ObjFunction *fn = closure->fn;

    if (fn->jitCode == NULL && !compileReturnNull(fn)) return false;

    /* Entering and leaving generated code. Nothing is passed yet because the
     * only compiled body ignores everything. */
    (void)((JaiCompiledFn)fn->jitCode)();

    /* The contract in jit.h: the call is complete, the frame was never pushed,
     * and the result sits at slotBase[0]. */
    slotBase[0] = NULL_VAL;
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
