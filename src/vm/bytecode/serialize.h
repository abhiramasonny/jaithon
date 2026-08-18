/* serialize.h — the .jaic container (spec/BYTECODE.md §4-§7). Caching rule: a
 * cache file is used only when every validity field matches; a mismatch, a
 * truncated file, or a bad CRC always means "recompile", never "crash" and
 * never "run stale code". */
#ifndef JAI_SERIALIZE_H
#define JAI_SERIALIZE_H

#include "vm/object/object.h"

#define JAIC_MAGIC       "JAIC"
/* Bump on any change to what codegen emits, not only the file format: a cache
 * entry is validated against its source's mtime, so an unchanged source
 * compiled by a newer compiler is otherwise served stale.
 * 11: top-level `fn` declarations are hoisted ahead of top-level statements.
 * 12: the line table is LTV1 (delta+LEB128) instead of 12-byte records.
 * 13: string constants are K_STRREF indices into one module string table.
 * 14: function flags are u16 (FN_TRACE and future high bits). */
#define JAIC_VERSION     14

#define JAIC_VERSION_FNFLAGS16 14

/* Oldest container the reader accepts.
 *
 * LOWER THIS BEFORE BUMPING JAIC_VERSION, not after. A strict reader refuses
 * boot/seed.bin -- whose images were written by the PREVIOUS generation -- and
 * the tree then has no front end and no way to build one; the failure is
 * `internal error: no front end ... module.c:370` and JAITHON_SEED_ANY=1 does
 * not rescue it. The sequence that works: widen the range, bump the version,
 * `make reseed` TWICE (generation 1 still runs the old compiler out of the old
 * seed and writes the old version; generation 2 writes the new one), and only
 * then raise this back up.
 *
 * At 13 because that sequence completed for v12 and v13 and the seed now
 * carries v13. The pre-LTV1 line table and the inline-K_STR pool are gone from
 * the reader with it: code no image can reach is code no test can check. */
#define JAIC_VERSION_MIN 13

/* First container version whose line table is LTV1. Below it, the table is
 * JAIC_LINE_ENTRY-sized absolute records and the reader re-encodes on load. */
#define JAIC_VERSION_LTV1 12

/* First container version carrying a module string table, whose strings every
 * pool then names by K_STRREF index. Below it, each pool holds its own K_STR
 * bytes. */
#define JAIC_VERSION_STRTAB 13

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
/* The image, and optionally its debug sidecar.
 *
 * When `flags` lacks JAIC_FLAG_DEBUG and `outSidecar` is non-NULL, the line
 * tables are stripped from the image and returned separately as .jaid content
 * -- 37% of what a .jaic weighs once LTV1 has shrunk it, and of no use to a
 * release build until something throws. Both buffers are the caller's to free.
 * `outSidecar` is left NULL when nothing was stripped. */
uint8_t *jaiSerializeModule(ObjModule *module, ObjFunction *body,
                            uint64_t sourceHash, uint32_t flags,
                            size_t *outSize, uint8_t **outSidecar,
                            size_t *outSidecarSize);

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
/* The whole cached image for `sourcePath`, or NULL. The caller owns the buffer
 * and releases it with jaiCacheReadFree.
 *
 * Exposed so a caller that must also check the flags header can do both from
 * one read: opening the file a second time for eight bytes cost 18.49 us per
 * module, more than reading the entire 116 KB image (16.92 us). */
uint8_t *jaiCacheRead(const char *sourcePath, size_t *outLength);
void     jaiCacheReadFree(uint8_t *data, size_t length);

/* Deserialise an already-read cache image, attaching the .jaid beside it when
 * the image was written stripped. Callers that read the file themselves must
 * use this rather than jaiDeserializeModule, or a release image silently loads
 * with no source spans at all. */
ObjFunction *jaiDeserializeCached(const uint8_t *data, size_t size,
                                  ObjModule *module, uint64_t sourceHash,
                                  const char *sourcePath);

ObjFunction *jaiCacheLoad(const char *sourcePath, ObjModule *module,
                          uint64_t sourceHash);
void  jaiCacheClear(const char *rootDir);

uint64_t jaiSourceHash(const char *source, size_t length);

/* The `buildId` field of §7: a fingerprint of the C sources this binary was
 * built from. The self-hosted front end stamps it into images it writes via
 * `__prim__.jaic_build_id`, since it can't hash C it never sees. */
uint32_t jaiBuildId(void);


#endif /* JAI_SERIALIZE_H */
