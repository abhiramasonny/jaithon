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
#define JAI_VERSION_MINOR 2
#define JAI_VERSION_PATCH 0
#define JAI_VERSION_STRING "3.2.0"

#define JAI_COMPILER_VERSION 25u
#define JAI_SEED_MIN_VERSION 18u

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
char *jaiMemdup(const char *s, size_t n);

extern size_t jaiHeapBytes;
static inline size_t jaiAllocatedBytes(void) { return jaiHeapBytes; }

/* ------------------------------------------------------------------ */
/* The small-object fast path, exposed                                 */
/* ------------------------------------------------------------------ */

#define JAI_SMALL_GRAIN   16u
#define JAI_SMALL_MAX     512u
#define JAI_SMALL_CLASSES (JAI_SMALL_MAX / JAI_SMALL_GRAIN)

extern void  *jaiSmallBin[JAI_SMALL_CLASSES + 1];
extern char  *jaiSlabNext;
extern size_t jaiSlabLeft;

void jaiSlabRefill(size_t need);

JAI_INLINE bool jaiSmallServes(size_t size) {
    return size != 0 && size <= JAI_SMALL_MAX;
}

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

JAI_INLINE void jaiSmallDeleteUnaccounted(void *p, size_t size) {
    unsigned cls = (unsigned)((size + (JAI_SMALL_GRAIN - 1u)) >> 4);
    *(void **)p = jaiSmallBin[cls];
    jaiSmallBin[cls] = p;
}

JAI_INLINE void jaiSmallDelete(void *p, size_t size) {
    jaiHeapBytes = size > jaiHeapBytes ? 0 : jaiHeapBytes - size;
    jaiSmallDeleteUnaccounted(p, size);
}

JAI_INLINE void jaiHeapAccountFreed(size_t total) {
    jaiHeapBytes = total > jaiHeapBytes ? 0 : jaiHeapBytes - total;
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
char *jaiArenaMemdup(JaiArena *arena, const char *s, size_t n);
void  jaiArenaReset(JaiArena *arena);
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
uint32_t jaiCrc32Table(const void *data, size_t len);

/* ------------------------------------------------------------------ */
/* UTF-8                                                               */
/* ------------------------------------------------------------------ */

int32_t jaiUtf8Decode(const char *s, const char *end, int *outLen);
int     jaiUtf8Encode(int32_t cp, char *out);
size_t  jaiUtf8Length(const char *s, size_t len);
size_t  jaiUtf8Offset(const char *s, size_t len, size_t i);
bool    jaiUtf8Validate(const char *s, size_t len);

/* ------------------------------------------------------------------ */
/* Files and paths                                                     */
/* ------------------------------------------------------------------ */

char *jaiReadFile(const char *path, size_t *outLen);
bool  jaiWriteFile(const char *path, const void *data, size_t len);
bool  jaiPathExists(const char *path);
bool  jaiPathIsDir(const char *path);
bool  jaiMakeDirs(const char *path);
void  jaiPathJoin(char *out, size_t outSize, const char *a, const char *b);
void  jaiPathDirname(char *out, size_t outSize, const char *path);
void  jaiPathBasename(char *out, size_t outSize, const char *path);
bool  jaiPathAbsolute(char *out, size_t outSize, const char *path);

double jaiClockMonotonic(void);

#endif /* JAI_COMMON_H */
