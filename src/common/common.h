/* common.h — foundational types, macros, and allocator for Jaithon.
 *
 * Everything in Jaithon includes this header first. It must stay free of
 * dependencies on the language proper (no Value, no Obj).
 */
#ifndef JAI_COMMON_H
#define JAI_COMMON_H

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Compiler feature detection                                          */
/* ------------------------------------------------------------------ */

#if defined(__GNUC__) || defined(__clang__)
#  define JAI_LIKELY(x)     __builtin_expect(!!(x), 1)
#  define JAI_UNLIKELY(x)   __builtin_expect(!!(x), 0)
#  define JAI_NORETURN      __attribute__((noreturn))
#  define JAI_INLINE        static inline __attribute__((always_inline))
#  define JAI_NOINLINE      __attribute__((noinline))
#  define JAI_PRINTF(a, b)  __attribute__((format(printf, a, b)))
#  define JAI_UNUSED        __attribute__((unused))
#  define JAI_COMPUTED_GOTO 1
#else
#  define JAI_LIKELY(x)     (x)
#  define JAI_UNLIKELY(x)   (x)
#  define JAI_NORETURN
#  define JAI_INLINE        static inline
#  define JAI_NOINLINE
#  define JAI_PRINTF(a, b)
#  define JAI_UNUSED
#  define JAI_COMPUTED_GOTO 0
#endif

#define JAI_VERSION_MAJOR 3
#define JAI_VERSION_MINOR 1
#define JAI_VERSION_PATCH 1
#define JAI_VERSION_STRING "3.1.1"

/* Bumped whenever bytecode emission changes; invalidates .jaic caches.
 *
 * Raising this is safe and is meant to be routine: the seed is checked against
 * the range below rather than for equality, so an older seed still bootstraps
 * the newer binary and `make reseed` then brings it up to date. See the note
 * above `deserialize` in serialize.c for why equality deadlocked. */
#define JAI_COMPILER_VERSION 19u

/* The oldest emission this binary can still EXECUTE, and so the oldest seed it
 * will bootstrap from. Opcodes are append-only (see OP_COUNT in chunk.h), so
 * anything this VM has ever emitted it can still run, and this only has to move
 * for a change append-only cannot express -- renumbering an opcode, or widening
 * an operand.
 *
 * Moving it is a two-release migration, not a one-line edit: the release that
 * introduces the break has to read BOTH formats so the seed can be regenerated
 * across it, and the release after that drops the old reader. Setting it equal
 * to JAI_COMPILER_VERSION without doing that is how the bootstrap wedges.
 * JAITHON_SEED_ANY=1 is the way out if it happens anyway. */
#define JAI_SEED_MIN_VERSION 18u

/* A floor above the ceiling accepts no seed at all, which is a tree that cannot
 * build itself by construction rather than by accident. Caught here, where the
 * two numbers are next to each other, instead of as a missing front end an hour
 * later. */
_Static_assert(JAI_SEED_MIN_VERSION <= JAI_COMPILER_VERSION,
               "JAI_SEED_MIN_VERSION is above JAI_COMPILER_VERSION: no seed "
               "can satisfy that range, so nothing could ever bootstrap");

/* ------------------------------------------------------------------ */
/* Limits                                                              */
/* ------------------------------------------------------------------ */

#define JAI_FRAMES_MAX      1024
#define JAI_STACK_MAX       (JAI_FRAMES_MAX * 64)
#define JAI_MAX_LOCALS      65535
#define JAI_MAX_UPVALUES    255
#define JAI_MAX_ARGS        255
#define JAI_MAX_PATH        4096

/* ------------------------------------------------------------------ */
/* Assertions and panics                                               */
/* ------------------------------------------------------------------ */

JAI_NORETURN void jaiPanic(const char *file, int line, const char *fmt, ...)
    JAI_PRINTF(3, 4);

#define JAI_PANIC(...) jaiPanic(__FILE__, __LINE__, __VA_ARGS__)

#ifdef JAI_DEBUG
#  define JAI_ASSERT(cond, msg)                                                \
      do {                                                                     \
          if (!(cond)) JAI_PANIC("assertion failed: %s (%s)", #cond, msg);     \
      } while (0)
#  define JAI_UNREACHABLE() JAI_PANIC("unreachable code reached")
#else
#  define JAI_ASSERT(cond, msg) ((void)0)
#  if defined(__GNUC__) || defined(__clang__)
#    define JAI_UNREACHABLE() __builtin_unreachable()
#  else
#    define JAI_UNREACHABLE() ((void)0)
#  endif
#endif

/* ------------------------------------------------------------------ */
/* Allocation                                                          */
/* ------------------------------------------------------------------ */

/* The single allocation chokepoint. All heap traffic goes through here so the
 * GC can account for it and so out-of-memory is handled in one place.
 *
 *   jaiRealloc(NULL, 0,       n) -> allocate n bytes
 *   jaiRealloc(p,    old,     n) -> resize
 *   jaiRealloc(p,    old,     0) -> free, returns NULL
 *
 * Never returns NULL for a nonzero size; it panics instead. */
void *jaiRealloc(void *ptr, size_t oldSize, size_t newSize);

#define JAI_ALLOC(type, count)                                                 \
    ((type *)jaiRealloc(NULL, 0, sizeof(type) * (size_t)(count)))
#define JAI_ALLOC_ZEROED(type, count) ((type *)jaiCalloc(sizeof(type), (size_t)(count)))
#define JAI_GROW_CAP(cap) ((cap) < 8 ? 8 : (cap) * 2)
#define JAI_GROW_ARRAY(type, ptr, oldCount, newCount)                          \
    ((type *)jaiRealloc(ptr, sizeof(type) * (size_t)(oldCount),                \
                        sizeof(type) * (size_t)(newCount)))
#define JAI_FREE_ARRAY(type, ptr, oldCount)                                    \
    ((void)jaiRealloc(ptr, sizeof(type) * (size_t)(oldCount), 0))
#define JAI_FREE(type, ptr) ((void)jaiRealloc(ptr, sizeof(type), 0))

void *jaiCalloc(size_t elemSize, size_t count);
char *jaiStrdup(const char *s);
/* Copies exactly `n` bytes and appends a terminator; see jaiArenaMemdup on
 * why this is not `strndup`. Callers pass `n` on to whatever reads the copy. */
char *jaiMemdup(const char *s, size_t n);

/* Bytes currently handed out by jaiRealloc. This is the number the collector
 * compares against its threshold, and it is read on the interpreter's loop
 * back edge, so it is a plain global rather than something reached through a
 * call or mirrored into the GC by a hook: the mirror was a second counter
 * maintained on every single allocation to hold exactly the same value, and
 * removing it was 3.5% of tests/bench/dict_ops. Written only by jaiRealloc. */
extern size_t jaiHeapBytes;
static inline size_t jaiAllocatedBytes(void) { return jaiHeapBytes; }

/* ------------------------------------------------------------------ */
/* The small-object fast path, exposed                                 */
/* ------------------------------------------------------------------ */

/* memory.c owns the bins and the slab and is still the only place that calls
 * malloc/realloc/free. What is published here is the half of `jaiRealloc(NULL,
 * 0, n)` that touches neither: pop a bin, or bump the slab cursor. The slab
 * *refill*, which is the only part that reaches libc, stays out of line in
 * memory.c behind jaiSlabRefill.
 *
 * This exists because the JIT's allocation site is a leaf that must not call
 * anything, and because a call per object was measurably a tenth of a program
 * that only allocates. Every invariant the bins already depended on still
 * holds and is unchanged: a block's size class is derived from the same exact
 * size on both sides, a slab block never reaches free()/realloc(), and nothing
 * here is ever returned to libc.
 *
 * A caller that uses this rather than jaiRealloc takes on exactly one duty:
 * `size` must be one the bins serve (0 < size <= JAI_SMALL_MAX), which
 * jaiSmallServes answers. */
#define JAI_SMALL_GRAIN   16u
#define JAI_SMALL_MAX     512u
#define JAI_SMALL_CLASSES (JAI_SMALL_MAX / JAI_SMALL_GRAIN)

extern void  *jaiSmallBin[JAI_SMALL_CLASSES + 1];
extern char  *jaiSlabNext;
extern size_t jaiSlabLeft;

/* Hands back a fresh JAI_SLAB_BYTES slab with at least `need` bytes in it,
 * having binned whatever tail the old one had left. Out of line: it is the
 * one path here that calls malloc. */
void jaiSlabRefill(size_t need);

/* True when `size` is a request jaiSmallNew can serve. */
JAI_INLINE bool jaiSmallServes(size_t size) {
    return size != 0 && size <= JAI_SMALL_MAX;
}

/* jaiRealloc(NULL, 0, size) for a size jaiSmallServes accepts, with the
 * accounting jaiRealloc would have done. Never NULL. */
JAI_INLINE void *jaiSmallNew(size_t size) {
    unsigned cls = (unsigned)((size + (JAI_SMALL_GRAIN - 1u)) >> 4);
    void *p = jaiSmallBin[cls];

    jaiHeapBytes += size;

    if (JAI_LIKELY(p != NULL)) {
        jaiSmallBin[cls] = *(void **)p;
        return p;
    }

    size_t need = (size_t)cls * JAI_SMALL_GRAIN;
    if (JAI_UNLIKELY(jaiSlabLeft < need)) jaiSlabRefill(need);

    p = jaiSlabNext;
    jaiSlabNext += need;
    jaiSlabLeft -= need;
    return p;
}

/* The other half of the same pair: jaiRealloc(ptr, size, 0) for a size the
 * bins serve, with the accounting jaiRealloc would have done.
 *
 * This buys nothing wherever LTO can already see jaiRealloc -- it inlines this
 * exact sequence into jaiFreeObject, and putting it in the source there
 * measured as pure code growth. It exists for the one caller LTO cannot help:
 * the GC sweep, whose cost is the *call* to jaiFreeObject and not what
 * jaiFreeObject then does.
 *
 * The duty is jaiSmallNew's, unchanged: `size` must be the exact size the
 * block was allocated with, and one jaiSmallServes accepts. */
JAI_INLINE void jaiSmallDelete(void *p, size_t size) {
    unsigned cls = (unsigned)((size + (JAI_SMALL_GRAIN - 1u)) >> 4);

    /* The clamp jaiRealloc's accountDelta applies, for the same reason: a
     * caller that misreports size must not wrap the counter to near SIZE_MAX
     * and convince the collector it can never free enough. */
    jaiHeapBytes = size > jaiHeapBytes ? 0 : jaiHeapBytes - size;

    *(void **)p = jaiSmallBin[cls];
    jaiSmallBin[cls] = p;
}


/* ------------------------------------------------------------------ */
/* Arena — bump allocator for AST nodes and other phase-scoped data    */
/* ------------------------------------------------------------------ */

typedef struct JaiArenaBlock JaiArenaBlock;

typedef struct {
    JaiArenaBlock *head;
    size_t blockSize;
    size_t totalBytes;
} JaiArena;

void  jaiArenaInit(JaiArena *arena, size_t blockSize);
void *jaiArenaAlloc(JaiArena *arena, size_t size);
void *jaiArenaAllocZeroed(JaiArena *arena, size_t size);
/* Copies exactly `n` bytes and appends a terminator. Deliberately not
 * `strndup`: a Jaithon string is length-prefixed and may hold an embedded NUL
 * (`"a\0b"`), so stopping at one both shortens the text and leaves the
 * caller's recorded length pointing past the allocation. */
char *jaiArenaMemdup(JaiArena *arena, const char *s, size_t n);
void  jaiArenaReset(JaiArena *arena);   /* keeps blocks, resets offsets */
void  jaiArenaFree(JaiArena *arena);

#define JAI_ARENA_NEW(arena, type)                                             \
    ((type *)jaiArenaAllocZeroed((arena), sizeof(type)))
#define JAI_ARENA_NEW_ARRAY(arena, type, n)                                    \
    ((type *)jaiArenaAllocZeroed((arena), sizeof(type) * (size_t)(n)))

/* ------------------------------------------------------------------ */
/* Byte buffer — growable byte/char sequence                           */
/* ------------------------------------------------------------------ */

typedef struct {
    uint8_t *data;
    size_t   count;
    size_t   capacity;
} JaiBuf;

void   jaiBufInit(JaiBuf *b);
void   jaiBufFree(JaiBuf *b);
void   jaiBufReserve(JaiBuf *b, size_t extra);
void   jaiBufPush(JaiBuf *b, uint8_t byte);
void   jaiBufAppend(JaiBuf *b, const void *bytes, size_t n);
void   jaiBufAppendStr(JaiBuf *b, const char *s);
void   jaiBufPrintf(JaiBuf *b, const char *fmt, ...) JAI_PRINTF(2, 3);
void   jaiBufWriteU16(JaiBuf *b, uint16_t v);   /* little-endian */
void   jaiBufWriteU24(JaiBuf *b, uint32_t v);
void   jaiBufWriteU32(JaiBuf *b, uint32_t v);
void   jaiBufWriteU64(JaiBuf *b, uint64_t v);
void   jaiBufWriteI16(JaiBuf *b, int16_t v);
void   jaiBufWriteF64(JaiBuf *b, double v);
/* Detach the buffer contents as a NUL-terminated heap string; resets `b`. */
char  *jaiBufTakeCString(JaiBuf *b, size_t *outLen);

/* ------------------------------------------------------------------ */
/* Generic growable vector of a POD type                               */
/* ------------------------------------------------------------------ */

#define JAI_VEC(type)                                                          \
    struct {                                                                   \
        type *data;                                                            \
        int   count;                                                           \
        int   capacity;                                                        \
    }

#define JAI_VEC_INIT(v)                                                        \
    do {                                                                       \
        (v)->data = NULL;                                                      \
        (v)->count = 0;                                                        \
        (v)->capacity = 0;                                                     \
    } while (0)

#define JAI_VEC_FREE(type, v)                                                  \
    do {                                                                       \
        JAI_FREE_ARRAY(type, (v)->data, (v)->capacity);                        \
        JAI_VEC_INIT(v);                                                       \
    } while (0)

#define JAI_VEC_PUSH(type, v, value)                                           \
    do {                                                                       \
        if ((v)->capacity < (v)->count + 1) {                                  \
            int _oldCap = (v)->capacity;                                       \
            (v)->capacity = JAI_GROW_CAP(_oldCap);                             \
            (v)->data = JAI_GROW_ARRAY(type, (v)->data, _oldCap, (v)->capacity);\
        }                                                                      \
        (v)->data[(v)->count++] = (value);                                     \
    } while (0)

#define JAI_VEC_POP(v) ((v)->data[--(v)->count])
#define JAI_VEC_LAST(v) ((v)->data[(v)->count - 1])

/* ------------------------------------------------------------------ */
/* Hashing                                                             */
/* ------------------------------------------------------------------ */

uint64_t jaiHashBytes(const void *data, size_t len);   /* FNV-1a 64 */
uint64_t jaiHashU64(uint64_t x);                       /* splitmix64 finaliser */
uint32_t jaiCrc32(const void *data, size_t len);
/* The portable table implementation. jaiCrc32 uses the ARMv8 CRC32
 * instructions where the target has them and falls back to this everywhere
 * else; the two must be one function, which tests/vm/crc32_equiv.c asserts. */
uint32_t jaiCrc32Table(const void *data, size_t len);

/* ------------------------------------------------------------------ */
/* UTF-8                                                               */
/* ------------------------------------------------------------------ */

/* Decode one scalar at `s`; writes its byte length to *outLen.
 * Returns -1 and *outLen = 1 on invalid input. */
int32_t jaiUtf8Decode(const char *s, const char *end, int *outLen);
/* Encode `cp` into `out` (>= 4 bytes). Returns bytes written, 0 if invalid. */
int     jaiUtf8Encode(int32_t cp, char *out);
/* Number of scalars in [s, s+len). */
size_t  jaiUtf8Length(const char *s, size_t len);
/* Byte offset of scalar index `i`, or len if out of range. */
size_t  jaiUtf8Offset(const char *s, size_t len, size_t i);
bool    jaiUtf8Validate(const char *s, size_t len);

/* ------------------------------------------------------------------ */
/* Files and paths                                                     */
/* ------------------------------------------------------------------ */

/* Reads a whole file. Returns NULL on failure. Caller frees with jaiRealloc. */
char *jaiReadFile(const char *path, size_t *outLen);
bool  jaiWriteFile(const char *path, const void *data, size_t len);
bool  jaiPathExists(const char *path);
bool  jaiPathIsDir(const char *path);
bool  jaiMakeDirs(const char *path);
/* Joins with '/', normalising duplicate separators. Writes into `out`. */
void  jaiPathJoin(char *out, size_t outSize, const char *a, const char *b);
void  jaiPathDirname(char *out, size_t outSize, const char *path);
void  jaiPathBasename(char *out, size_t outSize, const char *path);
/* Absolute, symlink-resolved path. Returns false if the path does not exist. */
bool  jaiPathAbsolute(char *out, size_t outSize, const char *path);

/* Monotonic seconds since an arbitrary epoch; for timing. */
double jaiClockMonotonic(void);

#endif /* JAI_COMMON_H */
