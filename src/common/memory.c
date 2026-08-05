/* memory.c — the allocator chokepoint, arena, byte buffer, hashing, UTF-8,
 * and the file/path/clock helpers declared in common.h.
 *
 * This is the one translation unit allowed to call malloc/realloc/free: every
 * other module reaches the C allocator through jaiRealloc so the GC can see
 * the traffic.
 */

/* Feature macros must precede every include. jaiReadFile needs fstat/fileno
 * and jaiPathAbsolute needs realpath, none of which -std=c11 exposes on its
 * own. */
#if !defined(_POSIX_C_SOURCE)
#  define _POSIX_C_SOURCE 200809L
#endif
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#  define _DARWIN_C_SOURCE
#endif

#include "common.h"

#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#ifndef PATH_MAX
#  define PATH_MAX 4096
#endif

/* ------------------------------------------------------------------ */
/* Panic                                                               */
/* ------------------------------------------------------------------ */

JAI_NORETURN void jaiPanic(const char *file, int line, const char *fmt, ...) {
    va_list args;

    fflush(stdout);   /* order the panic after whatever the program printed */
    fputs("jaithon: internal error: ", stderr);
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n  at %s:%d\n", file != NULL ? file : "<unknown>", line);
    fflush(stderr);
    abort();
}

/* ------------------------------------------------------------------ */
/* Allocation                                                          */
/* ------------------------------------------------------------------ */

size_t jaiHeapBytes = 0;

/* Shrinks and frees count too. Tracking only growth left the collector's live
 * total monotonically increasing: nextGC = live * factor then ratcheted upward
 * after every cycle until collections effectively stopped. */
static inline void accountDelta(size_t oldSize, size_t newSize) {
    if (newSize >= oldSize) {
        jaiHeapBytes += newSize - oldSize;
    } else {
        size_t delta = oldSize - newSize;
        /* A caller that misreports oldSize would otherwise wrap the counter to
         * near SIZE_MAX and make the GC believe it can never free enough. */
        jaiHeapBytes = delta > jaiHeapBytes ? 0 : jaiHeapBytes - delta;
    }
}

/* ------------------------------------------------------------------ */
/* Small-object recycling                                              */
/* ------------------------------------------------------------------ */

/* An interpreted program's heap traffic is overwhelmingly small, uniform and
 * short-lived: a bound method is 48 bytes and dies on the next instruction, a
 * short string is 40 + n, a tuple 32 + 16n. Measured on tests/bench, libc's
 * malloc and free together were 9-17% of a run and nearly everything passing
 * through them was under 128 bytes. Recycling those sizes here costs a shift
 * and two loads on each side, and was worth 10-12% on every benchmark that
 * allocates (and, as it should be, 0% on the two that do not).
 *
 * This works only because oldSize is exact. Every caller reaching jaiRealloc
 * reports the size it was given, which the JAI_*_ARRAY macros derive from the
 * same type and count as the allocation. The two places that did not — the
 * lexer freeing a cooked literal by strlen, io_listdir freeing the name array
 * one pointer short — were found by building with each block carrying its true
 * size and checking every report against it, and are fixed. Re-run that check
 * before trusting a new caller: a lie puts the block in the wrong bin, and the
 * next request served from that bin is short.
 *
 * Bins are never returned to libc. What they hold is bounded by the peak
 * garbage of each size class between collections, which is what libc's own
 * free lists would have held anyway; measured peak RSS moved by under 4%. */

#define JAI_SMALL_GRAIN   16u
#define JAI_SMALL_MAX     512u
#define JAI_SMALL_CLASSES (JAI_SMALL_MAX / JAI_SMALL_GRAIN)

/* Bin 0 is unused (a zero-byte request never reaches here); bin c serves
 * requests of 16(c-1)+1 .. 16c bytes and holds blocks of exactly 16c. */
static void *gBin[JAI_SMALL_CLASSES + 1];

/* 0 for anything the bins do not serve. */
static inline unsigned smallClass(size_t n) {
    return n - 1 < JAI_SMALL_MAX
               ? (unsigned)((n + (JAI_SMALL_GRAIN - 1)) / JAI_SMALL_GRAIN)
               : 0u;
}

static void *smallAlloc(unsigned cls) {
    void *p = gBin[cls];
    if (p != NULL) {
        /* The free list threads through the first word of each block, which is
         * safe because the smallest class is already pointer-sized. */
        gBin[cls] = *(void **)p;
        return p;
    }
    p = malloc((size_t)cls * JAI_SMALL_GRAIN);
    if (JAI_UNLIKELY(p == NULL))
        JAI_PANIC("out of memory: cannot allocate %u bytes (%zu live)",
                  cls * JAI_SMALL_GRAIN, jaiHeapBytes);
    return p;
}

static void smallFree(void *p, unsigned cls) {
    *(void **)p = gBin[cls];
    gBin[cls] = p;
}

void *jaiRealloc(void *ptr, size_t oldSize, size_t newSize) {
    unsigned oldCls = ptr != NULL ? smallClass(oldSize) : 0u;

    if (newSize == 0) {
        accountDelta(oldSize, 0);
        if (oldCls != 0) smallFree(ptr, oldCls);
        else free(ptr);
        return NULL;
    }

    /* Account first so a caller that reads jaiAllocatedBytes() sees the pending
     * request. What is accounted is the size asked for, not the class it rounds
     * up to, so the GC's thresholds mean what they meant before the bins
     * existed. */
    accountDelta(oldSize, newSize);

    unsigned newCls = smallClass(newSize);

    /* Same class: the block is already big enough and shrinking it would gain
     * nothing. This covers a growing buffer's early life and every
     * shrink-to-exact-size. */
    if (newCls != 0 && newCls == oldCls) return ptr;

    if (newCls != 0) {
        void *result = smallAlloc(newCls);
        if (ptr != NULL) {
            memcpy(result, ptr, oldSize < newSize ? oldSize : newSize);
            if (oldCls != 0) smallFree(ptr, oldCls);
            else free(ptr);
        }
        return result;
    }

    /* Leaving the bins: a binned block is not a libc block, so it cannot be
     * handed to realloc. */
    if (oldCls != 0) {
        void *result = malloc(newSize);
        if (JAI_UNLIKELY(result == NULL))
            JAI_PANIC("out of memory: cannot allocate %zu bytes (%zu live)",
                      newSize, jaiHeapBytes);
        memcpy(result, ptr, oldSize < newSize ? oldSize : newSize);
        smallFree(ptr, oldCls);
        return result;
    }

    void *result = realloc(ptr, newSize);
    if (JAI_UNLIKELY(result == NULL)) {
        JAI_PANIC("out of memory: cannot allocate %zu bytes (%zu live)", newSize,
                  jaiHeapBytes);
    }
    return result;
}

void *jaiCalloc(size_t elemSize, size_t count) {
    if (elemSize == 0 || count == 0) return NULL;
    if (count > SIZE_MAX / elemSize) {
        JAI_PANIC("allocation size overflow: %zu * %zu", elemSize, count);
    }

    size_t total = elemSize * count;
    void *p = jaiRealloc(NULL, 0, total);
    memset(p, 0, total);
    return p;
}

char *jaiStrdup(const char *s) {
    if (s == NULL) return NULL;
    return jaiMemdup(s, strlen(s));
}

char *jaiMemdup(const char *s, size_t n) {
    if (s == NULL) return NULL;

    /* Exactly `n` bytes: its one caller hands the same `n` to jaiSourceAdd and
     * jaiLexerInit, so a copy that stopped at an embedded NUL would leave them
     * reading past the end of the allocation. */
    char *copy = JAI_ALLOC(char, n + 1);
    if (n > 0) memcpy(copy, s, n);
    copy[n] = '\0';
    return copy;
}

/* ------------------------------------------------------------------ */
/* Arena                                                               */
/* ------------------------------------------------------------------ */

#define JAI_ARENA_ALIGN         16u
#define JAI_ARENA_DEFAULT_BLOCK (64u * 1024u)

struct JaiArenaBlock {
    JaiArenaBlock *next;
    uint8_t       *base;      /* first aligned payload byte */
    size_t         capacity;  /* payload bytes at `base` */
    size_t         used;
    size_t         allocSize; /* what jaiRealloc handed out, for the free */
};

static size_t alignUpSize(size_t n) {
    size_t mask = (size_t)JAI_ARENA_ALIGN - 1;
    if (n > SIZE_MAX - mask) JAI_PANIC("arena allocation size overflow: %zu", n);
    return (n + mask) & ~mask;
}

/* Blocks carry JAI_ARENA_ALIGN of slack so the payload can start on an aligned
 * address regardless of what malloc returned. */
static JaiArenaBlock *arenaNewBlock(JaiArena *arena, size_t payload) {
    size_t total = sizeof(JaiArenaBlock) + JAI_ARENA_ALIGN;
    if (payload > SIZE_MAX - total) JAI_PANIC("arena block size overflow: %zu", payload);
    total += payload;

    JaiArenaBlock *b = (JaiArenaBlock *)jaiRealloc(NULL, 0, total);
    uintptr_t raw     = (uintptr_t)b + sizeof(JaiArenaBlock);
    uintptr_t aligned = (raw + (JAI_ARENA_ALIGN - 1)) & ~(uintptr_t)(JAI_ARENA_ALIGN - 1);

    b->next      = NULL;
    b->base      = (uint8_t *)aligned;
    b->capacity  = total - (size_t)(aligned - (uintptr_t)b);
    b->used      = 0;
    b->allocSize = total;

    arena->totalBytes += total;
    return b;
}

void jaiArenaInit(JaiArena *arena, size_t blockSize) {
    if (arena == NULL) return;
    arena->head       = NULL;
    arena->blockSize  = blockSize > 0 ? blockSize : JAI_ARENA_DEFAULT_BLOCK;
    arena->totalBytes = 0;
}

void *jaiArenaAlloc(JaiArena *arena, size_t size) {
    if (arena == NULL) return NULL;
    if (arena->blockSize == 0) arena->blockSize = JAI_ARENA_DEFAULT_BLOCK;

    /* A zero-size request still gets a distinct address; returning NULL would
     * make JAI_ARENA_NEW_ARRAY(a, T, 0) indistinguishable from failure. */
    if (size == 0) size = 1;
    size = alignUpSize(size);

    JaiArenaBlock *b = arena->head;
    if (b == NULL || b->capacity - b->used < size) {
        /* After jaiArenaReset the retained blocks are empty again, so look
         * before allocating. The scan only runs when the head block fills. */
        JaiArenaBlock *scan = b != NULL ? b->next : NULL;
        while (scan != NULL && scan->capacity - scan->used < size) scan = scan->next;

        if (scan != NULL) {
            b = scan;
        } else if (size > arena->blockSize) {
            /* Oversized allocations get a dedicated block, parked behind the
             * head so the head stays the active bump block. */
            b = arenaNewBlock(arena, size);
            if (arena->head == NULL) {
                arena->head = b;
            } else {
                b->next = arena->head->next;
                arena->head->next = b;
            }
        } else {
            b = arenaNewBlock(arena, arena->blockSize);
            b->next = arena->head;
            arena->head = b;
        }
    }

    void *p = b->base + b->used;
    b->used += size;
    return p;
}

void *jaiArenaAllocZeroed(JaiArena *arena, size_t size) {
    void *p = jaiArenaAlloc(arena, size);
    if (p != NULL && size > 0) memset(p, 0, size);
    return p;
}

char *jaiArenaMemdup(JaiArena *arena, const char *s, size_t n) {
    if (s == NULL) return NULL;

    /* This used to stop at the first NUL, the way strndup does, while every
     * caller went on recording `n` as the length: `"a\0b"` was stored as the
     * two bytes `a\0` and read back as three, so the third came from whatever
     * followed in the arena. Length-prefixed means length-prefixed. */
    char *copy = (char *)jaiArenaAlloc(arena, n + 1);
    if (copy == NULL) return NULL;
    if (n > 0) memcpy(copy, s, n);
    copy[n] = '\0';
    return copy;
}

void jaiArenaReset(JaiArena *arena) {
    if (arena == NULL) return;
    for (JaiArenaBlock *b = arena->head; b != NULL; b = b->next) b->used = 0;
}

void jaiArenaFree(JaiArena *arena) {
    if (arena == NULL) return;

    JaiArenaBlock *b = arena->head;
    while (b != NULL) {
        JaiArenaBlock *next = b->next;
        jaiRealloc(b, b->allocSize, 0);
        b = next;
    }
    arena->head       = NULL;
    arena->totalBytes = 0;
}

/* ------------------------------------------------------------------ */
/* Byte buffer                                                         */
/* ------------------------------------------------------------------ */

#define JAI_BUF_MIN_CAP 16u

void jaiBufInit(JaiBuf *b) {
    if (b == NULL) return;
    b->data     = NULL;
    b->count    = 0;
    b->capacity = 0;
}

void jaiBufFree(JaiBuf *b) {
    if (b == NULL) return;
    JAI_FREE_ARRAY(uint8_t, b->data, b->capacity);
    jaiBufInit(b);
}

void jaiBufReserve(JaiBuf *b, size_t extra) {
    if (b == NULL || extra == 0) return;
    if (b->capacity - b->count >= extra) return;
    if (extra > SIZE_MAX - b->count) JAI_PANIC("buffer size overflow");

    size_t needed = b->count + extra;
    size_t cap = b->capacity < JAI_BUF_MIN_CAP ? JAI_BUF_MIN_CAP : b->capacity;
    while (cap < needed) {
        if (cap > SIZE_MAX / 2) { cap = needed; break; }
        cap *= 2;
    }

    b->data = JAI_GROW_ARRAY(uint8_t, b->data, b->capacity, cap);
    b->capacity = cap;
}

void jaiBufPush(JaiBuf *b, uint8_t byte) {
    if (b == NULL) return;
    jaiBufReserve(b, 1);
    b->data[b->count++] = byte;
}

void jaiBufAppend(JaiBuf *b, const void *bytes, size_t n) {
    if (b == NULL || n == 0) return;
    if (bytes == NULL) return;
    jaiBufReserve(b, n);
    memcpy(b->data + b->count, bytes, n);
    b->count += n;
}

void jaiBufAppendStr(JaiBuf *b, const char *s) {
    if (s == NULL) return;
    jaiBufAppend(b, s, strlen(s));
}

void jaiBufPrintf(JaiBuf *b, const char *fmt, ...) {
    if (b == NULL || fmt == NULL) return;

    va_list args, sizing;
    va_start(args, fmt);
    va_copy(sizing, args);

    int need = vsnprintf(NULL, 0, fmt, sizing);
    va_end(sizing);
    if (need < 0) {
        va_end(args);
        JAI_PANIC("vsnprintf failed for format \"%s\"", fmt);
    }

    /* vsnprintf always writes a NUL, so reserve one byte past the text; it is
     * overwritten by the next append and never counted. */
    jaiBufReserve(b, (size_t)need + 1);
    vsnprintf((char *)b->data + b->count, (size_t)need + 1, fmt, args);
    va_end(args);
    b->count += (size_t)need;
}

void jaiBufWriteU16(JaiBuf *b, uint16_t v) {
    if (b == NULL) return;
    jaiBufReserve(b, 2);
    b->data[b->count++] = (uint8_t)(v & 0xFFu);
    b->data[b->count++] = (uint8_t)((v >> 8) & 0xFFu);
}

void jaiBufWriteU24(JaiBuf *b, uint32_t v) {
    if (b == NULL) return;
    if (v > 0xFFFFFFu) JAI_PANIC("u24 operand out of range: %u", v);
    jaiBufReserve(b, 3);
    b->data[b->count++] = (uint8_t)(v & 0xFFu);
    b->data[b->count++] = (uint8_t)((v >> 8) & 0xFFu);
    b->data[b->count++] = (uint8_t)((v >> 16) & 0xFFu);
}

void jaiBufWriteU32(JaiBuf *b, uint32_t v) {
    if (b == NULL) return;
    jaiBufReserve(b, 4);
    for (int i = 0; i < 4; i++) b->data[b->count++] = (uint8_t)((v >> (8 * i)) & 0xFFu);
}

void jaiBufWriteU64(JaiBuf *b, uint64_t v) {
    if (b == NULL) return;
    jaiBufReserve(b, 8);
    for (int i = 0; i < 8; i++) b->data[b->count++] = (uint8_t)((v >> (8 * i)) & 0xFFu);
}

void jaiBufWriteI16(JaiBuf *b, int16_t v) { jaiBufWriteU16(b, (uint16_t)v); }

void jaiBufWriteF64(JaiBuf *b, double v) {
    uint64_t bits;
    memcpy(&bits, &v, sizeof bits);   /* IEEE-754 bit pattern, little-endian */
    jaiBufWriteU64(b, bits);
}

char *jaiBufTakeCString(JaiBuf *b, size_t *outLen) {
    if (b == NULL) {
        if (outLen != NULL) *outLen = 0;
        return NULL;
    }

    jaiBufReserve(b, 1);
    b->data[b->count] = '\0';
    size_t len = b->count;

    /* Shrink to the exact size so the caller can release it with
     * JAI_FREE_ARRAY(char, s, len + 1) and keep the byte accounting honest. */
    char *s = (char *)jaiRealloc(b->data, b->capacity, len + 1);
    jaiBufInit(b);

    if (outLen != NULL) *outLen = len;
    return s;
}

/* ------------------------------------------------------------------ */
/* Hashing                                                             */
/* ------------------------------------------------------------------ */

uint64_t jaiHashBytes(const void *data, size_t len) {
    uint64_t hash = 14695981039346656037ULL;   /* FNV-1a 64 offset basis */
    const uint8_t *p = (const uint8_t *)data;

    if (p == NULL) return hash;
    for (size_t i = 0; i < len; i++) {
        hash ^= (uint64_t)p[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

uint64_t jaiHashU64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return x;
}

static uint32_t gCrcTable[256];
static bool     gCrcTableReady = false;

static void crcBuildTable(void) {
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int k = 0; k < 8; k++) {
            c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
        }
        gCrcTable[i] = c;
    }
    gCrcTableReady = true;
}

uint32_t jaiCrc32(const void *data, size_t len) {
    /* The table is pure: a concurrent first use recomputes identical values,
     * so no lock is needed. */
    if (!gCrcTableReady) crcBuildTable();

    const uint8_t *p = (const uint8_t *)data;
    uint32_t crc = 0xFFFFFFFFu;

    if (p == NULL) return 0u;
    for (size_t i = 0; i < len; i++) {
        crc = gCrcTable[(crc ^ p[i]) & 0xFFu] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFFu;
}

/* ------------------------------------------------------------------ */
/* UTF-8                                                               */
/* ------------------------------------------------------------------ */

#define UTF8_CONT(b) (((b) & 0xC0u) == 0x80u)

int32_t jaiUtf8Decode(const char *s, const char *end, int *outLen) {
    int len = 1;   /* invalid input always advances one byte */
    int32_t cp = -1;

    if (s == NULL || end == NULL || s >= end) goto done;

    {
        const uint8_t *p = (const uint8_t *)s;
        size_t avail = (size_t)(end - s);
        uint8_t b0 = p[0];

        /* Unicode 15 table 3-7: the ranges of the second byte encode the
         * overlong and surrogate rejections directly. */
        if (b0 < 0x80u) {
            cp = (int32_t)b0;
        } else if (b0 < 0xC2u) {
            /* continuation byte, or C0/C1 which can only be overlong */
        } else if (b0 < 0xE0u) {
            if (avail >= 2 && UTF8_CONT(p[1])) {
                cp = (int32_t)(((uint32_t)(b0 & 0x1Fu) << 6) | (uint32_t)(p[1] & 0x3Fu));
                len = 2;
            }
        } else if (b0 < 0xF0u) {
            uint8_t lo = (b0 == 0xE0u) ? 0xA0u : 0x80u;   /* E0: no overlong */
            uint8_t hi = (b0 == 0xEDu) ? 0x9Fu : 0xBFu;   /* ED: no surrogate */
            if (avail >= 3 && p[1] >= lo && p[1] <= hi && UTF8_CONT(p[2])) {
                cp = (int32_t)(((uint32_t)(b0 & 0x0Fu) << 12) |
                               ((uint32_t)(p[1] & 0x3Fu) << 6) |
                               (uint32_t)(p[2] & 0x3Fu));
                len = 3;
            }
        } else if (b0 < 0xF5u) {
            uint8_t lo = (b0 == 0xF0u) ? 0x90u : 0x80u;   /* F0: no overlong */
            uint8_t hi = (b0 == 0xF4u) ? 0x8Fu : 0xBFu;   /* F4: cap at 10FFFF */
            if (avail >= 4 && p[1] >= lo && p[1] <= hi && UTF8_CONT(p[2]) &&
                UTF8_CONT(p[3])) {
                cp = (int32_t)(((uint32_t)(b0 & 0x07u) << 18) |
                               ((uint32_t)(p[1] & 0x3Fu) << 12) |
                               ((uint32_t)(p[2] & 0x3Fu) << 6) |
                               (uint32_t)(p[3] & 0x3Fu));
                len = 4;
            }
        }
        if (cp < 0) len = 1;
    }

done:
    if (outLen != NULL) *outLen = len;
    return cp;
}

int jaiUtf8Encode(int32_t cp, char *out) {
    if (out == NULL) return 0;
    if (cp < 0 || cp > 0x10FFFF) return 0;
    if (cp >= 0xD800 && cp <= 0xDFFF) return 0;

    uint32_t c = (uint32_t)cp;
    uint8_t *o = (uint8_t *)out;

    if (c < 0x80u) {
        o[0] = (uint8_t)c;
        return 1;
    }
    if (c < 0x800u) {
        o[0] = (uint8_t)(0xC0u | (c >> 6));
        o[1] = (uint8_t)(0x80u | (c & 0x3Fu));
        return 2;
    }
    if (c < 0x10000u) {
        o[0] = (uint8_t)(0xE0u | (c >> 12));
        o[1] = (uint8_t)(0x80u | ((c >> 6) & 0x3Fu));
        o[2] = (uint8_t)(0x80u | (c & 0x3Fu));
        return 3;
    }
    o[0] = (uint8_t)(0xF0u | (c >> 18));
    o[1] = (uint8_t)(0x80u | ((c >> 12) & 0x3Fu));
    o[2] = (uint8_t)(0x80u | ((c >> 6) & 0x3Fu));
    o[3] = (uint8_t)(0x80u | (c & 0x3Fu));
    return 4;
}

size_t jaiUtf8Length(const char *s, size_t len) {
    if (s == NULL) return 0;

    const char *p   = s;
    const char *end = s + len;
    size_t count = 0;

    while (p < end) {
        /* An ASCII run's scalar count is its byte count, so skip it eight
         * bytes at a time rather than decoding each one. `len()` on the 2.3 MB
         * string tests/bench/string_build joins was 8% of that benchmark, all
         * of it in the per-byte decoder. */
        while (end - p >= (ptrdiff_t)sizeof(uint64_t)) {
            uint64_t word;
            memcpy(&word, p, sizeof word);
            if (word & UINT64_C(0x8080808080808080)) break;
            p += sizeof word;
            count += sizeof word;
        }
        while (p < end && (unsigned char)*p < 0x80) { p++; count++; }
        if (p >= end) break;

        int step = 1;
        jaiUtf8Decode(p, end, &step);   /* invalid bytes count as one scalar */
        p += step;
        count++;
    }
    return count;
}

size_t jaiUtf8Offset(const char *s, size_t len, size_t i) {
    if (s == NULL) return 0;

    const char *p   = s;
    const char *end = s + len;
    size_t seen = 0;

    while (p < end && seen < i) {
        int step = 1;
        jaiUtf8Decode(p, end, &step);
        p += step;
        seen++;
    }
    return seen == i ? (size_t)(p - s) : len;
}

bool jaiUtf8Validate(const char *s, size_t len) {
    if (len == 0) return true;
    if (s == NULL) return false;

    const char *p   = s;
    const char *end = s + len;

    while (p < end) {
        int step = 1;
        if (jaiUtf8Decode(p, end, &step) < 0) return false;
        p += step;
    }
    return true;
}

/* ------------------------------------------------------------------ */
/* Files                                                               */
/* ------------------------------------------------------------------ */

char *jaiReadFile(const char *path, size_t *outLen) {
    if (outLen != NULL) *outLen = 0;
    if (path == NULL) return NULL;

    FILE *f = fopen(path, "rb");
    if (f == NULL) return NULL;

    struct stat st;
    if (fstat(fileno(f), &st) != 0 || !S_ISREG(st.st_mode)) {
        fclose(f);
        return NULL;   /* directories and devices are not source files */
    }
    if ((uintmax_t)st.st_size > (uintmax_t)(SIZE_MAX - 64)) {
        fclose(f);
        return NULL;
    }

    /* st_size is only a hint — the file may change under us — so the read loop
     * is authoritative and grows when it has to. Two bytes of slack (NUL plus
     * one probe byte) let the common case hit EOF without ever growing. */
    size_t cap = (size_t)st.st_size + 2;
    if (cap < 64) cap = 64;
    char *buf = JAI_ALLOC(char, cap);
    size_t total = 0;

    for (;;) {
        if (total + 1 >= cap) {
            size_t newCap = cap > SIZE_MAX / 2 ? SIZE_MAX : cap * 2;
            if (newCap == cap) {
                JAI_FREE_ARRAY(char, buf, cap);
                fclose(f);
                return NULL;
            }
            buf = JAI_GROW_ARRAY(char, buf, cap, newCap);
            cap = newCap;
        }
        size_t got = fread(buf + total, 1, cap - 1 - total, f);
        total += got;
        if (got == 0) break;
    }

    if (ferror(f)) {
        JAI_FREE_ARRAY(char, buf, cap);
        fclose(f);
        return NULL;
    }
    fclose(f);

    buf[total] = '\0';   /* binary-safe: callers use outLen, lexers use the NUL */
    if (cap != total + 1) {
        buf = JAI_GROW_ARRAY(char, buf, cap, total + 1);
    }
    if (outLen != NULL) *outLen = total;
    return buf;
}

bool jaiWriteFile(const char *path, const void *data, size_t len) {
    if (path == NULL) return false;
    if (data == NULL && len > 0) return false;

    FILE *f = fopen(path, "wb");
    if (f == NULL) return false;

    bool ok = (len == 0) || (fwrite(data, 1, len, f) == len);
    if (fclose(f) != 0) ok = false;   /* the final flush can still fail */
    return ok;
}

bool jaiPathExists(const char *path) {
    struct stat st;
    if (path == NULL) return false;
    return stat(path, &st) == 0;
}

bool jaiPathIsDir(const char *path) {
    struct stat st;
    if (path == NULL) return false;
    return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

static bool makeOneDir(const char *path) {
    if (mkdir(path, 0777) == 0) return true;
    if (errno == EEXIST) return jaiPathIsDir(path);
    return false;
}

bool jaiMakeDirs(const char *path) {
    if (path == NULL || path[0] == '\0') return false;

    size_t len = strlen(path);
    if (len + 1 > JAI_MAX_PATH) return false;

    char buf[JAI_MAX_PATH];
    memcpy(buf, path, len + 1);

    /* Start at 1: index 0 is either a leading '/' (nothing to create) or the
     * first character of a relative component. */
    for (size_t i = 1; i <= len; i++) {
        if (buf[i] != '/' && buf[i] != '\0') continue;
        char saved = buf[i];
        buf[i] = '\0';
        if (!makeOneDir(buf)) {
            buf[i] = saved;
            return false;
        }
        buf[i] = saved;
    }
    return jaiPathIsDir(path);
}

/* ------------------------------------------------------------------ */
/* Paths                                                               */
/* ------------------------------------------------------------------ */

/* These all return void or bool with no way to report truncation, and a
 * truncated path silently names a *different* file. On overflow they leave
 * `out` empty instead, which fails every subsequent operation loudly.
 *
 * memmove, not memcpy: `out` may alias `path`, since jaiPathDirname(p, n, p)
 * is the natural way to walk up a directory chain. */
static bool pathStore(char *out, size_t outSize, const char *src, size_t len) {
    if (out == NULL || outSize == 0) return false;
    if (len + 1 > outSize) {
        out[0] = '\0';
        return false;
    }
    if (len > 0) memmove(out, src, len);
    out[len] = '\0';
    return true;
}

/* Appends `s`, collapsing runs of '/' into one. */
static bool pathPushNorm(char *dst, size_t dstSize, size_t *pos, const char *s) {
    for (const char *c = s; *c != '\0'; c++) {
        if (*c == '/' && *pos > 0 && dst[*pos - 1] == '/') continue;
        if (*pos + 1 >= dstSize) return false;
        dst[(*pos)++] = *c;
    }
    return true;
}

void jaiPathJoin(char *out, size_t outSize, const char *a, const char *b) {
    if (out == NULL || outSize == 0) return;

    /* Assembled in a temporary so `out` may alias `a` or `b`. That caps a
     * joined path at JAI_MAX_PATH, which is the limit anyway. */
    char tmp[JAI_MAX_PATH];
    size_t pos = 0;

    /* An absolute second half discards the first, as every path join does. */
    if (b != NULL && b[0] == '/') a = NULL;

    bool haveA = a != NULL && a[0] != '\0';
    bool haveB = b != NULL && b[0] != '\0';

    if (haveA && !pathPushNorm(tmp, sizeof tmp, &pos, a)) { out[0] = '\0'; return; }
    if (haveA && haveB && tmp[pos - 1] != '/') {
        if (pos + 1 >= sizeof tmp) { out[0] = '\0'; return; }
        tmp[pos++] = '/';
    }
    if (haveB && !pathPushNorm(tmp, sizeof tmp, &pos, b)) { out[0] = '\0'; return; }

    while (pos > 1 && tmp[pos - 1] == '/') pos--;   /* no trailing separator */
    pathStore(out, outSize, tmp, pos);
}

void jaiPathDirname(char *out, size_t outSize, const char *path) {
    if (out == NULL || outSize == 0) return;
    if (path == NULL) { pathStore(out, outSize, ".", 1); return; }

    size_t len = strlen(path);
    size_t original = len;

    while (len > 0 && path[len - 1] == '/') len--;
    if (len == 0) {
        pathStore(out, outSize, original > 0 ? "/" : ".", 1);
        return;
    }
    while (len > 0 && path[len - 1] != '/') len--;   /* drop last component */
    if (len == 0) { pathStore(out, outSize, ".", 1); return; }
    while (len > 1 && path[len - 1] == '/') len--;   /* and its separators */

    pathStore(out, outSize, path, len);
}

void jaiPathBasename(char *out, size_t outSize, const char *path) {
    if (out == NULL || outSize == 0) return;
    if (path == NULL || path[0] == '\0') { pathStore(out, outSize, ".", 1); return; }

    size_t len = strlen(path);
    while (len > 0 && path[len - 1] == '/') len--;
    if (len == 0) { pathStore(out, outSize, "/", 1); return; }

    size_t start = len;
    while (start > 0 && path[start - 1] != '/') start--;

    pathStore(out, outSize, path + start, len - start);
}

bool jaiPathAbsolute(char *out, size_t outSize, const char *path) {
    if (out == NULL || outSize == 0) return false;
    out[0] = '\0';
    if (path == NULL) return false;

    char resolved[PATH_MAX];
    if (realpath(path, resolved) == NULL) return false;
    return pathStore(out, outSize, resolved, strlen(resolved));
}

/* ------------------------------------------------------------------ */
/* Clock                                                               */
/* ------------------------------------------------------------------ */

double jaiClockMonotonic(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0.0;
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
