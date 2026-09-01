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

    /* A THEN B, across a page boundary.
     *
     * Sealing and unsealing are range-limited: unseal flips only
     * [page_floor(used), capacity) and seal flips exactly that back, so that a
     * four-mebibyte arena is not re-protected end to end on every compile. The
     * property that has to hold is that A -- written and sealed BEFORE the
     * window -- is still executable after B is written and sealed inside it.
     *
     * The stencils are deliberately put on different pages, because that is
     * the case a whole-mapping mprotect cannot get wrong and a windowed one
     * can: A is below `windowFrom`, B is above it, and only a correct seal
     * leaves both executable with a coherent instruction cache. */
    if (!jaiCodeArenaInit(&arena, 3u * 4096u)) {
        fprintf(stderr, "jit_arena: second mmap failed\n");
        return 1;
    }
#if defined(__aarch64__) || defined(__arm64__)
    /* mov w0, #7 ; ret */
    const uint32_t codeB[] = { 0x528000e0u, 0xd65f03c0u };
#else
    /* mov eax, 7 ; ret */
    const uint8_t codeB[] = { 0xb8, 0x07, 0x00, 0x00, 0x00, 0xc3 };
#endif
    uint8_t *a = jaiCodeArenaWrite(&arena, code, sizeof code);
    if (a == NULL || !jaiCodeArenaSeal(&arena)) {
        fprintf(stderr, "jit_arena: A write/seal failed\n");
        return 1;
    }
    if (((Fn0)a)() != 42) {
        fprintf(stderr, "jit_arena: A wrong before the window\n");
        return 1;
    }
    if (!jaiCodeArenaUnseal(&arena)) {
        fprintf(stderr, "jit_arena: unseal failed\n");
        return 1;
    }
    /* Push past the page A sits in, so B lands above windowFrom. */
    {
        uint8_t pad[64];
        for (size_t i = 0; i < sizeof pad; i++) pad[i] = 0;
        while (arena.used < 4096u + 32u) {
            if (jaiCodeArenaWrite(&arena, pad, sizeof pad) == NULL) {
                fprintf(stderr, "jit_arena: padding refused\n");
                return 1;
            }
        }
    }
    uint8_t *b = jaiCodeArenaWrite(&arena, codeB, sizeof codeB);
    if (b == NULL || !jaiCodeArenaSeal(&arena)) {
        fprintf(stderr, "jit_arena: B write/seal failed\n");
        return 1;
    }
    if ((size_t)(b - arena.code) < 4096u) {
        fprintf(stderr, "jit_arena: B did not land on a later page\n");
        return 1;
    }
    if (((Fn0)b)() != 7) {
        fprintf(stderr, "jit_arena: B wrong after the window\n");
        return 1;
    }
    if (((Fn0)a)() != 42) {
        fprintf(stderr, "jit_arena: A stopped working across the window\n");
        return 1;
    }
    jaiCodeArenaFree(&arena);

    printf("jit_arena: ok\n");
    return 0;
}
