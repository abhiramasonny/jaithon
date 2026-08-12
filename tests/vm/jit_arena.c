/* The code arena must execute what it was written.
 *
 * A C test rather than a .jai one because there is nothing in the language that
 * can reach this yet, and because the failure being guarded against -- a stale
 * instruction cache on arm64 -- is invisible from above: the wrong code runs and
 * returns a plausible number.
 */
#include <stdio.h>
#include <stdint.h>
#include "vm/jit/jit.h"

typedef int (*Fn0)(void);

int main(void) {
    JaiCodeArena arena;
    if (!jaiCodeArenaInit(&arena, 4096)) {
        fprintf(stderr, "jit_arena: mmap failed\n");
        return 1;
    }

#if defined(__aarch64__) || defined(__arm64__)
    /* mov w0, #42 ; ret */
    const uint32_t code[] = { 0x52800540u, 0xd65f03c0u };
#elif defined(__x86_64__)
    /* mov eax, 42 ; ret */
    const uint8_t code[] = { 0xb8, 0x2a, 0x00, 0x00, 0x00, 0xc3 };
#else
    fprintf(stderr, "jit_arena: no stencil for this architecture\n");
    jaiCodeArenaFree(&arena);
    return 77;   /* skip */
#endif

    uint8_t *entry = jaiCodeArenaWrite(&arena, code, sizeof code);
    if (entry == NULL) {
        fprintf(stderr, "jit_arena: write refused\n");
        return 1;
    }
    if (jaiCodeArenaWrite(&arena, code, arena.capacity) != NULL) {
        fprintf(stderr, "jit_arena: overlong write was accepted\n");
        return 1;
    }
    if (!jaiCodeArenaSeal(&arena)) {
        fprintf(stderr, "jit_arena: seal failed\n");
        return 1;
    }
    if (jaiCodeArenaWrite(&arena, code, sizeof code) != NULL) {
        fprintf(stderr, "jit_arena: write after seal was accepted\n");
        return 1;
    }

    int got = ((Fn0)entry)();
    if (got != 42) {
        fprintf(stderr, "jit_arena: generated code returned %d, want 42\n", got);
        return 1;
    }

    jaiCodeArenaFree(&arena);
    printf("jit_arena: ok\n");
    return 0;
}
