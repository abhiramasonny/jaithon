/* serialize.h — the .jaic container (spec/BYTECODE.md §4-§7).
 *
 * Caching rule: a cache file is used only when every validity field matches.
 * A mismatch, a truncated file, or a bad CRC always means "recompile", never
 * "crash" and never "run stale code".
 */
#ifndef JAI_SERIALIZE_H
#define JAI_SERIALIZE_H

#include "object.h"

#define JAIC_MAGIC       "JAIC"
/* Bump on any change to what codegen emits, not only to the file format: a
 * cache entry is validated against its source's mtime, so an unchanged source
 * compiled by a newer compiler is otherwise served stale.
 * 11: top-level `fn` declarations are hoisted ahead of top-level statements. */
#define JAIC_VERSION     11

typedef enum {
    JAIC_FLAG_DEBUG   = 1 << 0,
    JAIC_FLAG_RELEASE = 1 << 1,
    /* Which front end produced this image. A __jaicache__ entry used to record
     * no front end at all, so the two compilers could not share a cache
     * directory and --front=jai had to bypass the cache entirely. With the bit
     * stored, an entry written by one is an ordinary cache miss for the other,
     * and both can use the cache. It becomes vestigial when only one front end
     * is left, but harmless: the bit is simply always set. */
    JAIC_FLAG_SELFHOSTED = 1 << 2,

    /* The -O level the chunks were optimised at, two bits.
     *
     * Part of the key because the levels produce different code: cold,
     * `loop_sum` runs 1.37s at -O0 and 0.79s at -O2. Without it a warm cache
     * written by one level was handed to another, so `-O0` silently ran
     * optimised bytecode -- the flags claimed the image matched when the thing
     * that determined the image was not among them. */
    JAIC_FLAG_OPT_SHIFT = 3,
    JAIC_FLAG_OPT_MASK  = 3u << 3,
} JaicFlags;

/* Serialise a compiled module. Returns a heap buffer (jaiRealloc) or NULL. */
uint8_t *jaiSerializeModule(ObjModule *module, ObjFunction *body,
                            uint64_t sourceHash, uint32_t flags,
                            size_t *outSize);

/* Deserialise into `module`. Returns the module body function, or NULL if the
 * data is invalid, stale, or corrupt. Never partially mutates `module` on
 * failure. */
/* The seed's images, which ship inside this binary and were produced by the
 * previous one. Same reader, but the build id is not required to match: it is a
 * cache key, and the seed is not a cache. Format compatibility is still checked
 * through JAIC_VERSION and JAI_COMPILER_VERSION. */
ObjFunction *jaiDeserializeSeed(const uint8_t *data, size_t size,
                                ObjModule *module, uint64_t expectedHash);

ObjFunction *jaiDeserializeModule(const uint8_t *data, size_t size,
                                  ObjModule *module, uint64_t expectedHash);

/* Cache file management. */
void  jaiCachePathFor(const char *sourcePath, char *out, size_t outSize);
bool  jaiCacheStore(const char *sourcePath, ObjModule *module,
                    ObjFunction *body, uint64_t sourceHash, uint32_t flags);
ObjFunction *jaiCacheLoad(const char *sourcePath, ObjModule *module,
                          uint64_t sourceHash);
void  jaiCacheClear(const char *rootDir);

uint64_t jaiSourceHash(const char *source, size_t length);

/* The `buildId` field of §7: a fingerprint of the C sources this binary was
 * built from, which the reader insists on. The self-hosted front end has to
 * stamp it into the images it writes, and it cannot compute a hash of C it
 * never sees, so the running build hands it over (`__prim__.jaic_build_id`).
 * Only serialize.c sees the generated header that defines it. */
uint32_t jaiBuildId(void);


#endif /* JAI_SERIALIZE_H */
