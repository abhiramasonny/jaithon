/* module_cache.c — .jaic cache-flag computation and header decoding. */

#include <stdlib.h>

#include "runtime/runtime.h"
#include "runtime/modules/module_internal.h"
#include "vm/bytecode/serialize.h"

bool traceLoads(void) {
    const char *flag = getenv("JAITHON_TRACE_LOAD");
    return flag != NULL && flag[0] != '\0' && strcmp(flag, "0") != 0;
}

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

bool jaiCacheFlagsMatchBuffer(const uint8_t *head, size_t length,
                              uint32_t flags) {
    if (head == NULL || length < 8) return false;

    if (memcmp(head, JAIC_MAGIC, 4) != 0) return false;
    uint16_t version = (uint16_t)((uint16_t)head[4] | (uint16_t)(head[5] << 8));
    uint16_t stored  = (uint16_t)((uint16_t)head[6] | (uint16_t)(head[7] << 8));

    const uint16_t kLoadability = (uint16_t)~(uint16_t)JAIC_FLAG_SELFHOSTED;
    return version >= JAIC_VERSION_MIN && version <= JAIC_VERSION &&
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
