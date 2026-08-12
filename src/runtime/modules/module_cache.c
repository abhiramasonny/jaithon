/* module_cache.c — .jaic cache-flag computation and header decoding.
 *
 * Split out of module.c (see that file's header comment). Everything here is
 * a pure function of its arguments or of two env vars read fresh each call:
 * nothing in this file touches sOptions, sLoadingFrontEnd, or any other piece
 * of the bootstrap window's state, so none of it takes part in the deadlock
 * hazard those comments describe. Three related jobs:
 *
 *   - cacheFlagsFor turns a codegen configuration into the u16 a .jaic header
 *     records.
 *   - cacheFlagsMatch is the cheap 8-byte probe module.c uses to decide,
 *     without a full load, whether an on-disk cache entry is even worth
 *     trying — this is also the seed-lookup gate: maybeWarmFor in module.c
 *     asks it before deciding whether the front end needs warming yet.
 *   - jaicRejectionReason is the diagnostic-formatting counterpart: once a
 *     full jaiDeserializeModule/jaiDeserializeSeed has already said no, it
 *     re-reads the same fixed header to say which field did not match.
 *
 * traceLoads and seedDisabled live here too. They are the env-flag gates
 * (JAITHON_TRACE_LOAD, JAITHON_NO_SEED) module.c consults around every cache
 * and seed lookup; grouping them with the flag/header code they gate keeps
 * every place that reads raw .jaic bytes or decides whether to in one file.
 */

#include <stdlib.h>

#include "runtime/runtime.h"
#include "runtime/modules/module_internal.h"
#include "vm/bytecode/serialize.h"

/* JAITHON_TRACE_LOAD=1 reports where each module body came from.
 *
 * Where a module comes from is not observable from Jaithon, and it has to be:
 * the seed serves the compiler even when its source has moved, so an edit to
 * the compiler can be silently ignored, and every measurement taken after it
 * describes a compiler that was never built. That happened, and it invalidated
 * a day of conclusions about a fusion rule that may well have been correct. */
bool traceLoads(void) {
    const char *flag = getenv("JAITHON_TRACE_LOAD");
    return flag != NULL && flag[0] != '\0' && strcmp(flag, "0") != 0;
}

/* `make reseed` sets this. The seed must not serve the compilation that
 * produces its own replacement: with the seed answering, the front end never
 * compiles its own closure from source, so nothing lands in __jaicache__ and
 * the next seed is built from whatever little remains. Measured, the seed went
 * 39 modules -> 10 in one generation that way, and would have reached zero. */
bool seedDisabled(void) {
    const char *flag = getenv("JAITHON_NO_SEED");
    return flag != NULL && flag[0] != '\0' && strcmp(flag, "0") != 0;
}

uint32_t cacheFlagsFor(const CodegenOptions *opts, bool selfHosted) {
    uint32_t flags = 0;
    if (opts->debugInfo) flags |= JAIC_FLAG_DEBUG;
    if (opts->stripAsserts) flags |= JAIC_FLAG_RELEASE;
    if (selfHosted) flags |= JAIC_FLAG_SELFHOSTED;

    int level = opts->optLevel;
    if (level < 0) level = 0;
    if (level > 3) level = 3;
    flags |= ((uint32_t)level << JAIC_FLAG_OPT_SHIFT) & JAIC_FLAG_OPT_MASK;
    return flags;
}

/* A .jaic starts with magic, u16 version and u16 flags (spec/BYTECODE.md §7),
 * and the loader treats the flags as informational. Without this probe a cache
 * written by a debug build would be handed to a --release run with its asserts
 * still in it; with it, that is an ordinary cache miss. */
bool cacheFlagsMatch(const char *sourcePath, uint32_t flags) {
    char path[JAI_MAX_PATH];
    jaiCachePathFor(sourcePath, path, sizeof path);
    if (path[0] == '\0') return false;

    FILE *f = fopen(path, "rb");
    if (f == NULL) return false;
    uint8_t head[8];
    bool complete = fread(head, 1, sizeof head, f) == sizeof head;
    fclose(f);
    if (!complete) return false;

    return jaiCacheFlagsMatchBuffer(head, sizeof head, flags);
}

/* The same decision against a buffer the caller has already read.
 *
 * A caller that goes on to load the module must use this rather than the path
 * form above: opening the file a second time for eight bytes cost 18.49 us per
 * module, which is more than reading the whole 116 KB image (16.92 us). The
 * path form survives for maybeWarmFor, which asks the question and then does
 * NOT load -- there is no second read to fold into. */
bool jaiCacheFlagsMatchBuffer(const uint8_t *head, size_t length,
                              uint32_t flags) {
    if (head == NULL || length < 8) return false;

    if (memcmp(head, JAIC_MAGIC, 4) != 0) return false;
    uint16_t version = (uint16_t)((uint16_t)head[4] | (uint16_t)(head[5] << 8));
    uint16_t stored  = (uint16_t)((uint16_t)head[6] | (uint16_t)(head[7] << 8));

    /* JAIC_FLAG_SELFHOSTED records WHO produced the image, and that has no
     * bearing on whether this binary can load it. Comparing it made every
     * cached module miss inside the bootstrap window, where `selfHosting()` is
     * false and the cached image says self-hosted -- so the seed was
     * deserialised on every single run instead of the cache being used, and
     * `jaithon run` on an empty program cost 20ms that vanished with
     * JAITHON_NO_SEED=1.
     *
     * Debug and release still have to match: those change what the bytecode
     * contains, not who wrote it. */
    const uint16_t kLoadability = (uint16_t)~(uint16_t)JAIC_FLAG_SELFHOSTED;
    return version == JAIC_VERSION &&
           (stored & kLoadability) == ((uint16_t)flags & kLoadability);
}

static uint16_t readLE16(const uint8_t *p) {
    return (uint16_t)((uint16_t)p[0] | (uint16_t)((uint16_t)p[1] << 8));
}

static uint32_t readLE32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static uint64_t readLE64(const uint8_t *p) {
    return (uint64_t)readLE32(p) | ((uint64_t)readLE32(p + 4) << 32);
}

/* jaiDeserializeModule is written as a cache probe: it answers yes or no and
 * says nothing, because for a cache "no" only ever means "recompile". Here "no"
 * is the entire report, so the fixed header is re-read to name the field that
 * did not match. Field order follows spec/BYTECODE.md §7 and the reader in
 * serialize.c; anything past the header is the reader's business. */
const char *jaicRejectionReason(const uint8_t *data, size_t size,
                                uint64_t expectedHash, char *buf,
                                size_t bufSize) {
    if (size < 32) {
        snprintf(buf, bufSize,
                 "the image is %zu bytes, too short to hold a header", size);
        return buf;
    }
    if (memcmp(data, JAIC_MAGIC, 4) != 0) {
        return "it does not begin with the four magic bytes `JAIC`";
    }
    uint32_t crc = jaiCrc32(data, size - 4);
    uint32_t stored = readLE32(data + size - 4);
    if (crc != stored) {
        snprintf(buf, bufSize,
                 "the trailing CRC32 is 0x%08x but the bytes hash to 0x%08x",
                 stored, crc);
        return buf;
    }
    uint16_t version = readLE16(data + 4);
    if (version != (uint16_t)JAIC_VERSION) {
        snprintf(buf, bufSize,
                 "it declares container version %u and this build reads "
                 "version %d (`VERSION` in lib/jaithon/compile/jaic.jai)",
                 (unsigned)version, JAIC_VERSION);
        return buf;
    }
    uint32_t compiler = readLE32(data + 8);
    if (compiler != JAI_COMPILER_VERSION) {
        snprintf(buf, bufSize,
                 "it declares compiler version %u and this build is %u "
                 "(`COMPILER_VERSION` in lib/jaithon/compile/jaic.jai)",
                 (unsigned)compiler, (unsigned)JAI_COMPILER_VERSION);
        return buf;
    }
    uint32_t buildId = readLE32(data + 12);
    if (buildId != jaiBuildId()) {
        snprintf(buf, bufSize,
                 "it declares build id 0x%08x and this binary is 0x%08x "
                 "(`build_id()` in lib/jaithon/compile/jaic.jai, which reads "
                 "`__prim__.jaic_build_id()`)",
                 (unsigned)buildId, (unsigned)jaiBuildId());
        return buf;
    }
    uint64_t hash = readLE64(data + 16);
    if (hash != expectedHash) {
        return "the source hash it records is not the hash of this file";
    }
    return "the header is well formed, so it is the body the reader rejected";
}
