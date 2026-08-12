/* LTV1 round trip, at the cases a corpus does not reliably contain.
 *
 * scripts/linetable_golden.sh already proves the encoding answers identically
 * over 189,000 real instruction offsets, but a corpus only exercises the shapes
 * its sources happen to produce. This pins the edges directly: a backwards span
 * delta (an operand emitted after the expression that owns it), a zero-length
 * span, a span at UINT32_MAX, a single-entry table, and a lookup before the
 * first entry.
 *
 * It also pins the SIZE, because the whole point of the encoding is the size:
 * a change that keeps the answers but doubles the bytes should fail here. */
#include <stdio.h>
#include <string.h>

#include "vm/bytecode/chunk.h"

static int failures = 0;

static void expect(int cond, const char *what) {
    if (!cond) { printf("FAIL: %s\n", what); failures++; }
}

/* Encode `n` entries into `buf` exactly as a chunk would. */
static size_t encodeAll(uint8_t *buf, size_t cap, const LineEntry *es, int n) {
    size_t used = 0;
    for (int i = 0; i < n; i++) {
        used += jaiLtv1EncodeEntry(buf + used, cap - used,
                                   i > 0 ? &es[i - 1] : NULL, &es[i]);
    }
    return used;
}

static void roundTrip(const char *label, const LineEntry *es, int n) {
    uint8_t buf[4096];
    size_t used = encodeAll(buf, sizeof buf, es, n);
    expect(used <= sizeof buf, "encoded table fits the scratch buffer");

    /* Every entry must be recoverable at its own offset, and at every offset up
     * to the next entry's -- that range is what the entry covers. */
    for (int i = 0; i < n; i++) {
        uint32_t limit = (i + 1 < n) ? es[i + 1].offset : es[i].offset + 3;
        for (uint32_t off = es[i].offset; off < limit; off++) {
            uint32_t s = 0xDEADu, e = 0xBEEFu;
            bool ok = jaiLtv1Lookup(buf, used, off, &s, &e);
            if (!ok || s != es[i].span || e != es[i].spanEnd) {
                printf("FAIL: %s entry %d at offset %u: ok=%d got %u..%u "
                       "want %u..%u\n", label, i, off, (int)ok, s, e,
                       es[i].span, es[i].spanEnd);
                failures++;
                return;
            }
        }
    }
    expect(jaiLtv1Count(buf, used) == n, "entry count round-trips");
}

int main(void) {
    /* Ordinary forward progress. */
    const LineEntry plain[] = {
        {0, 10, 20}, {4, 22, 30}, {9, 31, 44}, {17, 45, 46},
    };
    roundTrip("plain", plain, 4);

    /* Spans that move BACKWARDS between entries -- the case zigzag exists for.
     * An operand's span follows the expression that owns it, so this is common,
     * not hypothetical. */
    const LineEntry backwards[] = {
        {0, 500, 520}, {3, 480, 495}, {6, 400, 460}, {12, 700, 705},
    };
    roundTrip("backwards", backwards, 4);

    /* Zero-length spans, and a span whose end equals its start. */
    const LineEntry empty[] = { {0, 7, 7}, {2, 7, 7}, {5, 9, 9} };
    roundTrip("zero-length", empty, 3);

    /* The top of the range: a source file cannot really be 4 GB, but the fields
     * are u32 and the codec must not lose the high bits. */
    const LineEntry huge[] = {
        {0, 0xFFFFFFF0u, 0xFFFFFFFFu}, {8, 1, 2},
    };
    roundTrip("u32 extremes", huge, 2);

    /* One entry covers everything from its offset on. */
    const LineEntry single[] = { {0, 3, 9} };
    roundTrip("single entry", single, 1);

    /* A lookup BEFORE the first entry has no answer, and must say so rather
     * than inventing the first entry's span. */
    {
        const LineEntry late[] = { {5, 100, 110} };
        uint8_t buf[64];
        size_t used = encodeAll(buf, sizeof buf, late, 1);
        uint32_t s = 0xAAAAu, e = 0xBBBBu;
        expect(!jaiLtv1Lookup(buf, used, 0, &s, &e),
               "offset before the first entry is not found");
        expect(s == 0xAAAAu && e == 0xBBBBu,
               "a miss leaves the caller's values untouched");
        expect(jaiLtv1Lookup(buf, used, 5, &s, &e), "offset at the entry hits");
        expect(s == 100 && e == 110, "and returns that entry's span");
    }

    /* An empty stream answers nothing rather than reading off the front. */
    {
        uint32_t s = 1, e = 2;
        expect(!jaiLtv1Lookup(NULL, 0, 0, &s, &e), "null stream is a miss");
        expect(jaiLtv1Count(NULL, 0) == 0, "null stream counts zero");
    }

    /* A truncated stream must stop, not run away. Every prefix of a valid
     * stream is fed in; none may report more entries than it holds. */
    {
        const LineEntry es[] = { {0, 10, 20}, {4, 22, 30}, {9, 31, 44} };
        uint8_t buf[128];
        size_t used = encodeAll(buf, sizeof buf, es, 3);
        for (size_t cut = 0; cut < used; cut++) {
            uint32_t s = 0, e = 0;
            (void)jaiLtv1Lookup(buf, cut, 100, &s, &e);   /* must return */
            expect(jaiLtv1Count(buf, cut) <= 3, "truncated stream over-counts");
        }
    }

    /* Size: the reason the encoding exists. The old form was 12 bytes flat. */
    {
        uint8_t buf[4096];
        size_t used = encodeAll(buf, sizeof buf, plain, 4);
        double perEntry = (double)used / 4.0;
        printf("linetable_ltv1: %.2f bytes/entry on the plain table "
               "(was 12.00)\n", perEntry);
        expect(perEntry < 6.0, "encoding is at least half the size of 12 bytes");
    }

    printf("linetable_ltv1: %d failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
