/* serialize.c — the .jaic bytecode container (spec/BYTECODE.md §4-§7).
 *
 * Every integer is little-endian. Writing goes through JaiBuf; reading goes
 * through a bounds-checked Cursor whose every accessor validates the remaining
 * length first, because a .jaic file is untrusted input that may be truncated,
 * corrupt, or hostile. A malformed file always yields NULL: never an
 * out-of-bounds read, never an allocation sized from a corrupt length field,
 * and never a half-updated ObjModule.
 *
 * Where a value cannot be expressed on disk the writer refuses it rather than
 * writing an approximation, so a cache miss (recompile) is the worst outcome.
 * The one such case left is FN_INIT: it is bit 8 of ObjFunction.flags but §5
 * stores flags in one byte, so the bit is recomputed at load time from
 * "FN_METHOD and named init", and the writer verifies that rule holds for the
 * function it is writing rather than dropping the bit silently.
 *
 * A class is not a constant and has no pool tag. §6's spec constant is an
 * ordinary tuple, so it round-trips through K_TUPLE with no help from here;
 * the fields, methods and traits it does not carry arrive as FIELD_DEF,
 * METHOD and IMPL_TRAIT instructions in the class body, which are just code.
 *
 * Inline caches are not stored. They must start empty anyway, so the loader
 * sizes them by scanning the code for cache-bearing opcodes, which doubles as
 * a decodability check on the instruction stream.
 *
 * Chunk.constIndex is not stored either: it is derived from the constant pool,
 * and jaiFunctionNew leaves it NULL so jaiChunkAddConstant rebuilds it from the
 * loaded pool if a later pass appends to it.
 */

#include "vm/bytecode/serialize_internal.h"

/* ------------------------------------------------------------------ */
/* Cache files                                                          */
/* ------------------------------------------------------------------ */

uint64_t jaiSourceHash(const char *source, size_t length) {
    return jaiHashBytes(source, source != NULL ? length : 0);
}

uint32_t jaiBuildId(void) { return JAI_BUILD_ID; }

void jaiCachePathFor(const char *sourcePath, char *out, size_t outSize) {
    if (out == NULL || outSize == 0) return;
    out[0] = '\0';
    if (sourcePath == NULL || sourcePath[0] == '\0') return;

    char dir[JAI_MAX_PATH];
    char base[JAI_MAX_PATH];
    jaiPathDirname(dir, sizeof dir, sourcePath);
    jaiPathBasename(base, sizeof base, sourcePath);
    if (dir[0] == '\0' || base[0] == '\0') return;
    if (strcmp(base, ".") == 0 || strcmp(base, "..") == 0 ||
        strcmp(base, "/") == 0) {
        return;
    }

    /* "src/util.jai" -> "src/__jaicache__/util.jaic". */
    size_t len = strlen(base);
    size_t extLen = strlen(JAIC_SOURCE_EXT);
    if (len > extLen && memcmp(base + len - extLen, JAIC_SOURCE_EXT, extLen) == 0) {
        len -= extLen;
        base[len] = '\0';
    }
    if (len == 0) return;

    char cacheDir[JAI_MAX_PATH];
    jaiPathJoin(cacheDir, sizeof cacheDir, dir, JAIC_CACHE_DIR);
    if (cacheDir[0] == '\0') return;

    char file[JAI_MAX_PATH];
    int written = snprintf(file, sizeof file, "%s%s", base, JAIC_SUFFIX);
    if (written < 0 || (size_t)written >= sizeof file) return;

    /* jaiPathJoin empties `out` rather than truncating, which is what the
     * callers below test for. */
    jaiPathJoin(out, outSize, cacheDir, file);
}

/* "…/util.jaic" -> "…/util.jaid". False when it does not fit. */
static bool sidecarPathFor(const char *cachePath, char *out, size_t outSize) {
    size_t len = strlen(cachePath);
    size_t extLen = strlen(JAIC_SUFFIX);
    if (len <= extLen || len + 1 > outSize) return false;
    if (memcmp(cachePath + len - extLen, JAIC_SUFFIX, extLen) != 0) return false;
    memcpy(out, cachePath, len);
    memcpy(out + len - extLen, JAID_SUFFIX, extLen);
    out[len] = '\0';
    return true;
}

bool jaiCacheStore(const char *sourcePath, ObjModule *module,
                   ObjFunction *body, uint64_t sourceHash, uint32_t flags) {
    char path[JAI_MAX_PATH];
    jaiCachePathFor(sourcePath, path, sizeof path);
    if (path[0] == '\0') return false;

    char dir[JAI_MAX_PATH];
    jaiPathDirname(dir, sizeof dir, path);
    if (dir[0] == '\0' || !jaiMakeDirs(dir)) return false;

    size_t size = 0, sidecarSize = 0;
    uint8_t *sidecar = NULL;
    uint8_t *data = jaiSerializeModule(module, body, sourceHash, flags, &size,
                                       &sidecar, &sidecarSize);
    if (data == NULL) return false;

    /* Best effort, and deliberately before the image is renamed into place: a
     * .jaid without its .jaic is inert, whereas a .jaic whose .jaid never
     * arrived would silently lose spans. A failure here costs source spans in
     * release tracebacks and nothing else, so it does not fail the store. */
    if (sidecar != NULL) {
        char sidecarPath[JAI_MAX_PATH];
        if (sidecarPathFor(path, sidecarPath, sizeof sidecarPath)) {
            (void)jaiWriteFile(sidecarPath, sidecar, sidecarSize);
        }
        (void)jaiRealloc(sidecar, sidecarSize, 0);
    } else {
        /* A debug rebuild over a stripped one must not leave the old sidecar
         * behind: its records would no longer line up. */
        char sidecarPath[JAI_MAX_PATH];
        if (sidecarPathFor(path, sidecarPath, sizeof sidecarPath)) {
            (void)unlink(sidecarPath);
        }
    }

    /* Write to a pid-tagged temporary and rename: rename is atomic within a
     * directory, so a concurrent reader sees either the old file or the whole
     * new one, never a partial write. */
    char tmp[JAI_MAX_PATH];
    int written = snprintf(tmp, sizeof tmp, "%s.tmp%ld", path, (long)getpid());
    if (written < 0 || (size_t)written >= sizeof tmp) {
        (void)jaiRealloc(data, size, 0);
        return false;
    }

    bool ok = jaiWriteFile(tmp, data, size);
    (void)jaiRealloc(data, size, 0);
    if (!ok) {
        (void)unlink(tmp);
        return false;
    }
    if (rename(tmp, path) != 0) {
        (void)unlink(tmp);
        return false;
    }
    return true;
}

uint8_t *jaiCacheRead(const char *sourcePath, size_t *outLength) {
    char path[JAI_MAX_PATH];
    jaiCachePathFor(sourcePath, path, sizeof path);
    if (path[0] == '\0') return NULL;

    size_t length = 0;
    char *data = jaiReadFile(path, &length);
    if (data == NULL) return NULL;

    *outLength = length;
    return (uint8_t *)data;
}

void jaiCacheReadFree(uint8_t *data, size_t length) {
    /* jaiReadFile allocates length + 1 for its terminator; freeing the wrong
     * size corrupts the allocator's accounting, which is what decides when the
     * collector runs. */
    if (data != NULL) (void)jaiRealloc(data, length + 1, 0);
}

/* The .jaid beside `sourcePath`'s image, validated and positioned past its
 * header, or NULL. Never an error: an absent, stale or corrupt sidecar costs
 * source spans in tracebacks, nothing more. */
static uint8_t *jaiSidecarRead(const char *sourcePath, uint64_t sourceHash,
                        size_t *outLength, size_t *outOffset) {
    char path[JAI_MAX_PATH];
    char sidecarPath[JAI_MAX_PATH];
    jaiCachePathFor(sourcePath, path, sizeof path);
    if (path[0] == '\0') return NULL;
    if (!sidecarPathFor(path, sidecarPath, sizeof sidecarPath)) return NULL;

    size_t length = 0;
    char *data = jaiReadFile(sidecarPath, &length);
    if (data == NULL) return NULL;

    const uint8_t *p = (const uint8_t *)data;
    bool ok = length >= JAID_HEADER + 4 &&
              memcmp(p, JAID_MAGIC, 4) == 0 &&
              ((uint16_t)p[4] | (uint16_t)(p[5] << 8)) == JAID_VERSION &&
              jaiCrc32(p, length - 4) == readU32At(p + length - 4);
    if (ok) {
        uint64_t recorded = 0;
        for (int i = 0; i < 8; i++) recorded |= (uint64_t)p[8 + i] << (8 * i);
        /* The tie to its image. Records are matched by position, so a sidecar
         * from a different compile would silently attach the wrong spans. */
        ok = recorded == sourceHash;
    }
    if (!ok) {
        (void)jaiRealloc(data, length + 1, 0);
        return NULL;
    }

    *outLength = length - 4;      /* the cursor stops short of the checksum */
    *outOffset = JAID_HEADER;
    return (uint8_t *)data;
}

ObjFunction *jaiDeserializeCached(const uint8_t *data, size_t size,
                                  ObjModule *module, uint64_t sourceHash,
                                  const char *sourcePath) {
    size_t sidecarLen = 0, sidecarOffset = 0;
    uint8_t *sidecar = jaiSidecarRead(sourcePath, sourceHash, &sidecarLen,
                                      &sidecarOffset);

    ObjFunction *fn = deserializeWithSidecar(data, size, module, sourceHash,
                                             false, sidecar, sidecarLen,
                                             sidecarOffset);
    if (sidecar != NULL) {
        /* jaiReadFile allocated sidecarLen + 4 (checksum) + 1 (terminator). */
        (void)jaiRealloc(sidecar, sidecarLen + 5, 0);
    }
    return fn;
}

ObjFunction *jaiCacheLoad(const char *sourcePath, ObjModule *module,
                          uint64_t sourceHash) {
    size_t length = 0;
    uint8_t *data = jaiCacheRead(sourcePath, &length);
    if (data == NULL) return NULL;

    ObjFunction *fn = jaiDeserializeCached(data, length, module, sourceHash,
                                           sourcePath);
    jaiCacheReadFree(data, length);
    return fn;
}

static bool hasSuffix(const char *s, const char *suffix) {
    size_t sl = strlen(s), fl = strlen(suffix);
    return sl >= fl && memcmp(s + sl - fl, suffix, fl) == 0;
}

/* Deletes the artefacts this module creates — cache files and any temporary
 * left behind by an interrupted store — then the directory if it is now
 * empty. Foreign files are left alone. */
static void clearCacheDir(const char *dir) {
    DIR *d = opendir(dir);
    if (d == NULL) return;

    for (struct dirent *e = readdir(d); e != NULL; e = readdir(d)) {
        if (strcmp(e->d_name, ".") == 0 || strcmp(e->d_name, "..") == 0) continue;
        if (!hasSuffix(e->d_name, JAIC_SUFFIX) &&
            strstr(e->d_name, JAIC_SUFFIX ".tmp") == NULL) {
            continue;
        }

        char path[JAI_MAX_PATH];
        jaiPathJoin(path, sizeof path, dir, e->d_name);
        if (path[0] == '\0') continue;

        struct stat st;
        if (lstat(path, &st) != 0 || !S_ISREG(st.st_mode)) continue;
        (void)unlink(path);
    }
    (void)closedir(d);
    (void)rmdir(dir);   /* succeeds only if nothing else was in there */
}

static void clearCacheTree(const char *dir, int depth) {
    if (depth > 32) return;   /* a symlink loop cannot be walked forever */

    DIR *d = opendir(dir);
    if (d == NULL) return;

    for (struct dirent *e = readdir(d); e != NULL; e = readdir(d)) {
        if (strcmp(e->d_name, ".") == 0 || strcmp(e->d_name, "..") == 0) continue;

        char path[JAI_MAX_PATH];
        jaiPathJoin(path, sizeof path, dir, e->d_name);
        if (path[0] == '\0') continue;

        /* lstat, not stat: a symlinked directory is not descended into, so a
         * clear can never delete outside the tree it was given. */
        struct stat st;
        if (lstat(path, &st) != 0 || !S_ISDIR(st.st_mode)) continue;

        if (strcmp(e->d_name, JAIC_CACHE_DIR) == 0) {
            clearCacheDir(path);
        } else {
            clearCacheTree(path, depth + 1);
        }
    }
    (void)closedir(d);
}

void jaiCacheClear(const char *rootDir) {
    if (rootDir == NULL || rootDir[0] == '\0') return;
    if (!jaiPathIsDir(rootDir)) return;

    char base[JAI_MAX_PATH];
    jaiPathBasename(base, sizeof base, rootDir);
    if (strcmp(base, JAIC_CACHE_DIR) == 0) {
        clearCacheDir(rootDir);
        return;
    }
    clearCacheTree(rootDir, 0);
}
