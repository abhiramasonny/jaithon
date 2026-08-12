/* The hardware CRC32 path must be indistinguishable from the table path.
 *
 * Both are reflected 0xEDB88320. The hardware form consumes eight bytes at a
 * time, so its tail handling and its behaviour on unaligned starts are where a
 * divergence would hide -- this walks every length from 0 to 512 at sixteen
 * different alignments, which covers every tail residue and every alignment
 * case, then spot-checks long ranges.
 *
 * Every .jaic in the tree carries a CRC computed by the table path, so a
 * divergence here is not a slow path: it rejects every cached image in the
 * tree and silently turns a 33 ms warm load into a 3.5 s cold compile. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common/common.h"

#define BUF_BYTES (1u << 16)

int main(void) {
    uint8_t *buf = malloc(BUF_BYTES);
    if (buf == NULL) return 2;
    /* Deterministic, and every byte value appears. */
    for (size_t i = 0; i < BUF_BYTES; i++) buf[i] = (uint8_t)(i * 31u + (i >> 8));

    long checked = 0, bad = 0;

    for (size_t align = 0; align < 16; align++) {
        for (size_t len = 0; len <= 512; len++) {
            uint32_t a = jaiCrc32(buf + align, len);
            uint32_t b = jaiCrc32Table(buf + align, len);
            checked++;
            if (a != b) {
                if (bad < 10) {
                    printf("MISMATCH align=%zu len=%zu hw=%08x table=%08x\n",
                           align, len, a, b);
                }
                bad++;
            }
        }
    }

    for (size_t len = 1024; len <= BUF_BYTES; len *= 2) {
        uint32_t a = jaiCrc32(buf, len);
        uint32_t b = jaiCrc32Table(buf, len);
        checked++;
        if (a != b) {
            printf("MISMATCH long len=%zu hw=%08x table=%08x\n", len, a, b);
            bad++;
        }
    }

    /* A NULL buffer answers 0 on both paths rather than faulting: a truncated
     * cache file reaches here. */
    if (jaiCrc32(NULL, 16) != 0u || jaiCrc32Table(NULL, 16) != 0u) {
        printf("MISMATCH null buffer\n");
        bad++;
    }
    checked++;

    free(buf);
    printf("crc32_equiv: %ld checked, %ld mismatched\n", checked, bad);
    return bad == 0 ? 0 : 1;
}
