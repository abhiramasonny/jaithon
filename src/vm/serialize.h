/* serialize.h — the .jaic container (spec/BYTECODE.md §4-§7). Caching rule: a
 * cache file is used only when every validity field matches; a mismatch, a
 * truncated file, or a bad CRC always means "recompile", never "crash" and
 * never "run stale code". */
#ifndef JAI_SERIALIZE_H
#define JAI_SERIALIZE_H

#include "object.h"

#define JAIC_MAGIC       "JAIC"
/* Bump on any change to what codegen emits, not only the file format: a cache
 * entry is validated against its source's mtime, so an unchanged source
 * compiled by a newer compiler is otherwise served stale.
 * 11: top-level `fn` declarations are hoisted ahead of top-level statements. */
#define JAIC_VERSION     11

typedef enum {
    JAIC_FLAG_DEBUG   = 1 << 0,
    JAIC_FLAG_RELEASE = 1 << 1,
    /* Which front end produced this image. Without this bit the two compilers
     * couldn't share a cache directory and --front=jai had to bypass the
     * cache entirely; now an entry from one is just an ordinary miss for the
     * other. */
    JAIC_FLAG_SELFHOSTED = 1 << 2,

    /* The -O level the chunks were optimised at, two bits. Part of the key
     * because the levels produce different code (cold `loop_sum`: 1.37s at
     * -O0, 0.79s at -O2) -- without it, `-O0` could silently run a cache
     * entry written at -O2. */
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
/* For the seed images shipped inside this binary: same reader, but the build
 * id need not match (it's a cache key, and the seed isn't a cache). Format
 * compatibility is still checked via JAIC_VERSION and JAI_COMPILER_VERSION. */
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
 * built from. The self-hosted front end stamps it into images it writes via
 * `__prim__.jaic_build_id`, since it can't hash C it never sees. */
uint32_t jaiBuildId(void);


#endif /* JAI_SERIALIZE_H */
