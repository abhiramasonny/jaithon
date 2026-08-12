/* jaiChunkReserveCaches must allocate exactly what was asked for.
 *
 * The point of the function is that it does NOT round up. The deserialiser
 * knows the exact slot count before it allocates, and jaiChunkAddCache's
 * JAI_GROW_CAP rounding cost 1.19 MB of slack across the seed's images -- out
 * of 3.2 MB, which is 52% of the seed's post-load heap. A test that checked
 * only cacheCount would pass with the rounding still in place, so this checks
 * capacity, and it checks that a reserved slot is initialised exactly as
 * jaiChunkAddCache would have initialised it. */
#include <stdio.h>

#include "vm/bytecode/chunk.h"

static int failures = 0;

static void expect(int cond, const char *what) {
    if (!cond) { printf("FAIL: %s\n", what); failures++; }
}

int main(void) {
    Chunk chunk;

    /* 9 is deliberately just past a growth boundary: JAI_GROW_CAP rounds it to
     * 16 and leaves seven slots of slack. */
    jaiChunkInit(&chunk, -1);
    expect(jaiChunkReserveCaches(&chunk, 9), "reserve 9 succeeds");
    expect(chunk.cacheCount == 9, "cacheCount is 9");
    expect(chunk.cacheCapacity == 9, "cacheCapacity is exactly 9, not rounded");
    jaiChunkFree(&chunk);

    /* A reserved slot must be indistinguishable from an added one. */
    {
        Chunk grown, reserved;
        jaiChunkInit(&grown, -1);
        jaiChunkInit(&reserved, -1);
        for (int i = 0; i < 5; i++) (void)jaiChunkAddCache(&grown);
        expect(jaiChunkReserveCaches(&reserved, 5), "reserve 5 succeeds");
        expect(grown.cacheCount == reserved.cacheCount,
               "reserved and grown agree on cacheCount");
        int same = 1;
        for (int i = 0; i < 5; i++) {
            if (grown.caches[i].state != reserved.caches[i].state) same = 0;
            for (int w = 0; w < JAI_IC_WAYS; w++) {
                if (!IS_NULL(reserved.caches[i].cached[w])) same = 0;
            }
        }
        expect(same, "a reserved slot is initialised like an added one");
        jaiChunkFree(&grown);
        jaiChunkFree(&reserved);
    }

    /* Zero must allocate nothing at all, not a minimum block. */
    jaiChunkInit(&chunk, -1);
    expect(jaiChunkReserveCaches(&chunk, 0), "reserve 0 succeeds");
    expect(chunk.cacheCount == 0, "cacheCount is 0");
    expect(chunk.caches == NULL, "no allocation for zero caches");
    jaiChunkFree(&chunk);

    /* Refusals: negative, past the u16 operand's range, and double-reserve. */
    jaiChunkInit(&chunk, -1);
    expect(!jaiChunkReserveCaches(&chunk, -1), "negative count refused");
    expect(!jaiChunkReserveCaches(&chunk, UINT16_MAX + 2),
           "over-range count refused");
    expect(jaiChunkReserveCaches(&chunk, 4), "reserve 4 succeeds");
    expect(!jaiChunkReserveCaches(&chunk, 4), "second reserve refused");
    jaiChunkFree(&chunk);

    printf("chunk_caches: %d failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
