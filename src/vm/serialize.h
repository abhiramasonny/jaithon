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
} JaicFlags;

/* Serialise a compiled module. Returns a heap buffer (jaiRealloc) or NULL. */
uint8_t *jaiSerializeModule(ObjModule *module, ObjFunction *body,
                            uint64_t sourceHash, uint32_t flags,
                            size_t *outSize);

/* Deserialise into `module`. Returns the module body function, or NULL if the
 * data is invalid, stale, or corrupt. Never partially mutates `module` on
 * failure. */
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

/* Byte-for-byte comparison of two serialised functions, ignoring debug-only
 * fields. Used by --bootstrap-verify. Writes a human-readable explanation of
 * the first difference into `diff`. */
bool jaiCompareFunctions(const ObjFunction *a, const ObjFunction *b,
                         char *diff, size_t diffSize);

#endif /* JAI_SERIALIZE_H */
