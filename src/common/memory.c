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
    /* Grain is a power of two, so avoid an integer divide on this allocator
     * hot path. n==0 deliberately maps to class 0. */
    if (n == 0 || n > JAI_SMALL_MAX) return 0u;
    return (unsigned)((n + (JAI_SMALL_GRAIN - 1u)) >> 4);
}

static inline void smallFree(void *p, unsigned cls) {
    *(void **)p = gBin[cls];
    gBin[cls] = p;
}

/* When a bin is empty the block has to come from somewhere, and one malloc per
 * object is what that used to mean.
 *
 * The bins only recycle what has already died, so a program whose live set is
 * *growing* -- building a list of two million strings, a tree of half a million
 * nodes -- finds every bin empty and pays libc for every object. Sampled on
 * tests/bench/string_build, malloc's internals were 29% of the whole run.
 *
 * Carving those blocks out of a slab instead makes the common case a compare,
 * an add and two stores, and calls malloc once per slab rather than once per
 * object. Nothing else changes: a slab block is indistinguishable from a
 * malloc'd one to every caller, and when it dies it goes onto the same free
 * list it would have before.
 *
 * Three properties make it safe, and all three already held:
 *
 *   - malloc returns 16-aligned memory and every size class is a multiple of
 *     the 16-byte grain, so the cursor stays 16-aligned for the whole slab;
 *   - a slab block is never handed to free() or realloc(), because those are
 *     reached only when smallClass(oldSize) is 0 and a slab block's size is by
 *     construction one the bins serve -- the same exact-oldSize invariant the
 *     bins have always depended on;
 *   - blocks are never returned to libc, which the bins above already
 *     documented as their own behaviour.
 *
 * The slab is not freed at exit for the same reason the bins are not. */
#define JAI_SLAB_BYTES (64u * 1024u)

static char  *gSlabNext;
static size_t gSlabLeft;

static void *slabCarve(size_t need) {
    if (JAI_UNLIKELY(gSlabLeft < need)) {
        /* The tail is too small for this request but not too small to be a
         * block: hand it to the bin it exactly fits rather than leaking it.
         * gSlabLeft < need <= JAI_SMALL_MAX, so the class is in range, and it
         * is at least one grain, which is wider than the free-list pointer. */
        unsigned tail = (unsigned)(gSlabLeft / JAI_SMALL_GRAIN);
        if (tail != 0) smallFree(gSlabNext, tail);

        gSlabNext = (char *)malloc(JAI_SLAB_BYTES);
        if (JAI_UNLIKELY(gSlabNext == NULL)) {
            JAI_PANIC("out of memory: cannot allocate %u bytes (%zu live)",
                      JAI_SLAB_BYTES, jaiHeapBytes);
        }
        gSlabLeft = JAI_SLAB_BYTES;
    }

    void *p = gSlabNext;
    gSlabNext += need;
    gSlabLeft -= need;
    return p;
}

static inline void *smallAlloc(unsigned cls) {
    void *p = gBin[cls];

    if (p != NULL) {
        gBin[cls] = *(void **)p;
        return p;
    }

    return slabCarve((size_t)cls * JAI_SMALL_GRAIN);
}

void *jaiRealloc(void *ptr, size_t oldSize, size_t newSize) {
    const unsigned oldCls = ptr != NULL ? smallClass(oldSize) : 0u;

    if (JAI_UNLIKELY(newSize == 0)) {
        accountDelta(oldSize, 0);

        if (oldCls != 0)
            smallFree(ptr, oldCls);
        else
            free(ptr);

        return NULL;
    }

    /* Very common no-op resize (especially exact-capacity helpers). */
    if (JAI_UNLIKELY(ptr != NULL && oldSize == newSize))
        return ptr;

    accountDelta(oldSize, newSize);

    const unsigned newCls = smallClass(newSize);

    /* Fresh small allocation: by far the hottest allocation path. */
    if (ptr == NULL) {
        if (newCls != 0)
            return smallAlloc(newCls);

        void *result = malloc(newSize);
        if (JAI_UNLIKELY(result == NULL)) {
            JAI_PANIC("out of memory: cannot allocate %zu bytes (%zu live)",
                      newSize, jaiHeapBytes);
        }
        return result;
    }

    /* Same small class: physical capacity already satisfies the request. */
    if (newCls != 0 && newCls == oldCls)
        return ptr;

    if (newCls != 0) {
        void *result = smallAlloc(newCls);
        memcpy(result, ptr, oldSize < newSize ? oldSize : newSize);

        if (oldCls != 0)
            smallFree(ptr, oldCls);
        else
            free(ptr);

        return result;
    }

    /* Leaving the bins: a recycled block is not safe to hand to realloc(). */
    if (oldCls != 0) {
        void *result = malloc(newSize);
        if (JAI_UNLIKELY(result == NULL)) {
            JAI_PANIC("out of memory: cannot allocate %zu bytes (%zu live)",
                      newSize, jaiHeapBytes);
        }

        memcpy(result, ptr, oldSize < newSize ? oldSize : newSize);
        smallFree(ptr, oldCls);
        return result;
    }

    void *result = realloc(ptr, newSize);
    if (JAI_UNLIKELY(result == NULL)) {
        JAI_PANIC("out of memory: cannot allocate %zu bytes (%zu live)",
                  newSize, jaiHeapBytes);
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

static inline size_t alignUpSize(size_t n) {
    const size_t mask = (size_t)JAI_ARENA_ALIGN - 1u;
    if (JAI_UNLIKELY(n > SIZE_MAX - mask))
        JAI_PANIC("arena allocation size overflow: %zu", n);
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
    if (arena->blockSize == 0)
        arena->blockSize = JAI_ARENA_DEFAULT_BLOCK;

    if (size == 0) size = 1;
    size = alignUpSize(size);

    JaiArenaBlock *b = arena->head;

    if (JAI_UNLIKELY(b == NULL || b->capacity - b->used < size)) {
        /*
         * After reset there may already be a suitable retained block deeper in
         * the chain. Move the one we find to the head, so subsequent bump
         * allocations do not rescan the same prefix again.
         */
        JaiArenaBlock *prev = b;
        JaiArenaBlock *scan = b != NULL ? b->next : NULL;

        while (scan != NULL && scan->capacity - scan->used < size) {
            prev = scan;
            scan = scan->next;
        }

        if (scan != NULL) {
            prev->next = scan->next;
            scan->next = arena->head;
            arena->head = scan;
            b = scan;
        } else if (size > arena->blockSize) {
            /* Dedicated oversized block. Keep the normal bump block at head
             * when one exists, because small allocations dominate. */
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

    const size_t count = b->count;
    const size_t capacity = b->capacity;

    if (capacity - count >= extra)
        return;

    if (JAI_UNLIKELY(extra > SIZE_MAX - count))
        JAI_PANIC("buffer size overflow");

    const size_t needed = count + extra;
    size_t cap = capacity < JAI_BUF_MIN_CAP ? JAI_BUF_MIN_CAP : capacity;

    /* Usually one doubling is enough; keep that case branch-light. */
    if (cap < needed) {
        if (cap <= SIZE_MAX / 2)
            cap *= 2;

        if (cap < needed) {
            while (cap < needed) {
                if (cap > SIZE_MAX / 2) {
                    cap = needed;
                    break;
                }
                cap *= 2;
            }
        }
    }

    b->data = JAI_GROW_ARRAY(uint8_t, b->data, capacity, cap);
    b->capacity = cap;
}

void jaiBufPush(JaiBuf *b, uint8_t byte) {
    if (b == NULL) return;

    if (JAI_UNLIKELY(b->count == b->capacity))
        jaiBufReserve(b, 1);

    b->data[b->count++] = byte;
}

void jaiBufAppend(JaiBuf *b, const void *bytes, size_t n) {
    if (b == NULL || bytes == NULL || n == 0) return;

    const size_t count = b->count;

    if (JAI_UNLIKELY(b->capacity - count < n))
        jaiBufReserve(b, n);

    memcpy(b->data + count, bytes, n);
    b->count = count + n;
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

    const size_t count = b->count;
    if (JAI_UNLIKELY(b->capacity - count < 2))
        jaiBufReserve(b, 2);

    uint8_t *p = b->data + count;
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
    b->count = count + 2;
}

void jaiBufWriteU24(JaiBuf *b, uint32_t v) {
    if (b == NULL) return;
    if (JAI_UNLIKELY(v > 0xFFFFFFu))
        JAI_PANIC("u24 operand out of range: %u", v);

    const size_t count = b->count;
    if (JAI_UNLIKELY(b->capacity - count < 3))
        jaiBufReserve(b, 3);

    uint8_t *p = b->data + count;
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    b->count = count + 3;
}

void jaiBufWriteU32(JaiBuf *b, uint32_t v) {
    if (b == NULL) return;

    const size_t count = b->count;
    if (JAI_UNLIKELY(b->capacity - count < 4))
        jaiBufReserve(b, 4);

    uint8_t *p = b->data + count;
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
    b->count = count + 4;
}

void jaiBufWriteU64(JaiBuf *b, uint64_t v) {
    if (b == NULL) return;

    const size_t count = b->count;
    if (JAI_UNLIKELY(b->capacity - count < 8))
        jaiBufReserve(b, 8);

    uint8_t *p = b->data + count;
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
    p[4] = (uint8_t)(v >> 32);
    p[5] = (uint8_t)(v >> 40);
    p[6] = (uint8_t)(v >> 48);
    p[7] = (uint8_t)(v >> 56);
    b->count = count + 8;
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
    uint64_t hash = UINT64_C(14695981039346656037);
    const uint8_t *p = (const uint8_t *)data;

    if (p == NULL) return hash;

    /*
     * FNV-1a is serial, so SIMD cannot break its dependency chain. Unrolling
     * four bytes still removes most loop-control overhead on names/strings
     * without changing the hash ABI.
     */
    while (len >= 4) {
        hash ^= (uint64_t)p[0];
        hash *= UINT64_C(1099511628211);
        hash ^= (uint64_t)p[1];
        hash *= UINT64_C(1099511628211);
        hash ^= (uint64_t)p[2];
        hash *= UINT64_C(1099511628211);
        hash ^= (uint64_t)p[3];
        hash *= UINT64_C(1099511628211);
        p += 4;
        len -= 4;
    }

    while (len-- != 0) {
        hash ^= (uint64_t)*p++;
        hash *= UINT64_C(1099511628211);
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
    if (s == NULL || end == NULL || s >= end) {
        if (outLen != NULL) *outLen = 1;
        return -1;
    }

    const uint8_t *p = (const uint8_t *)s;
    const uint8_t b0 = p[0];

    /* ASCII dominates text processing. Avoid computing end-s and entering the
     * multibyte decision tree for it. */
    if (b0 < 0x80u) {
        if (outLen != NULL) *outLen = 1;
        return (int32_t)b0;
    }

    const size_t avail = (size_t)(end - s);
    int len = 1;
    int32_t cp = -1;

    if (b0 < 0xC2u) {
        /* continuation byte, or C0/C1 overlong */
    } else if (b0 < 0xE0u) {
        if (avail >= 2 && UTF8_CONT(p[1])) {
            cp = (int32_t)(((uint32_t)(b0 & 0x1Fu) << 6) |
                           (uint32_t)(p[1] & 0x3Fu));
            len = 2;
        }
    } else if (b0 < 0xF0u) {
        const uint8_t lo = b0 == 0xE0u ? 0xA0u : 0x80u;
        const uint8_t hi = b0 == 0xEDu ? 0x9Fu : 0xBFu;

        if (avail >= 3 && p[1] >= lo && p[1] <= hi &&
            UTF8_CONT(p[2])) {
            cp = (int32_t)(((uint32_t)(b0 & 0x0Fu) << 12) |
                           ((uint32_t)(p[1] & 0x3Fu) << 6) |
                           (uint32_t)(p[2] & 0x3Fu));
            len = 3;
        }
    } else if (b0 < 0xF5u) {
        const uint8_t lo = b0 == 0xF0u ? 0x90u : 0x80u;
        const uint8_t hi = b0 == 0xF4u ? 0x8Fu : 0xBFu;

        if (avail >= 4 && p[1] >= lo && p[1] <= hi &&
            UTF8_CONT(p[2]) && UTF8_CONT(p[3])) {
            cp = (int32_t)(((uint32_t)(b0 & 0x07u) << 18) |
                           ((uint32_t)(p[1] & 0x3Fu) << 12) |
                           ((uint32_t)(p[2] & 0x3Fu) << 6) |
                           (uint32_t)(p[3] & 0x3Fu));
            len = 4;
        }
    }

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

    const char *p = s;
    const char *const end = s + len;
    size_t count = 0;

    while (p < end) {
        /*
         * 16-byte ASCII fast path using unaligned-safe memcpy loads. Two words
         * at once reduce loop branches on long source files and joined strings.
         */
        while ((size_t)(end - p) >= 16u) {
            uint64_t a, b;
            memcpy(&a, p, sizeof a);
            memcpy(&b, p + 8, sizeof b);

            if ((a | b) & UINT64_C(0x8080808080808080))
                break;

            p += 16;
            count += 16;
        }

        while ((size_t)(end - p) >= 8u) {
            uint64_t word;
            memcpy(&word, p, sizeof word);

            if (word & UINT64_C(0x8080808080808080))
                break;

            p += 8;
            count += 8;
        }

        while (p < end && (unsigned char)*p < 0x80u) {
            ++p;
            ++count;
        }

        if (p >= end)
            break;

        int step = 1;
        (void)jaiUtf8Decode(p, end, &step);
        p += step;
        ++count;
    }

    return count;
}

size_t jaiUtf8Offset(const char *s, size_t len, size_t i) {
    if (s == NULL) return 0;

    const char *p = s;
    const char *const end = s + len;
    size_t seen = 0;

    while (p < end && seen < i) {
        size_t remaining = i - seen;

        while (remaining >= 8u && (size_t)(end - p) >= 8u) {
            uint64_t word;
            memcpy(&word, p, sizeof word);

            if (word & UINT64_C(0x8080808080808080))
                break;

            p += 8;
            seen += 8;
            remaining -= 8;
        }

        while (p < end && seen < i && (unsigned char)*p < 0x80u) {
            ++p;
            ++seen;
        }

        if (p >= end || seen >= i)
            break;

        int step = 1;
        (void)jaiUtf8Decode(p, end, &step);
        p += step;
        ++seen;
    }

    return seen == i ? (size_t)(p - s) : len;
}

bool jaiUtf8Validate(const char *s, size_t len) {
    if (len == 0) return true;
    if (s == NULL) return false;

    const char *p = s;
    const char *const end = s + len;

    while (p < end) {
        while ((size_t)(end - p) >= 8u) {
            uint64_t word;
            memcpy(&word, p, sizeof word);

            if (word & UINT64_C(0x8080808080808080))
                break;

            p += 8;
        }

        while (p < end && (unsigned char)*p < 0x80u)
            ++p;

        if (p >= end)
            break;

        int step = 1;
        if (jaiUtf8Decode(p, end, &step) < 0)
            return false;

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
