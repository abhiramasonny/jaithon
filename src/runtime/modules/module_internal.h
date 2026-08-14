/* module_internal.h — glue shared by the module-loading translation units. */
#ifndef JAI_MODULE_INTERNAL_H
#define JAI_MODULE_INTERNAL_H

#include "runtime/runtime.h"

#define JAI_MODULE_EXT ".jai"

/* -- module.c, used by module_path.c --------------------------------- */

bool importFailure(JaiDiagCode code, ObjClass *klass, const char *fmt, ...)
    JAI_PRINTF(3, 4);

/* -- module_path.c, used by module.c ----------------------------------- */

void ensurePathReady(void);

bool isRegularFile(const char *path);

bool storeResolved(char *out, size_t outSize, const char *candidate);

const char *displayName(const char *dotted);

/* -- module_cache.c, used by module.c ---------------------------------- */

bool traceLoads(void);

bool seedDisabled(void);

uint32_t cacheFlagsFor(const CodegenOptions *opts, bool selfHosted);

bool cacheFlagsMatch(const char *sourcePath, uint32_t flags);
bool jaiCacheFlagsMatchBuffer(const uint8_t *head, size_t length,
                              uint32_t flags);

const char *jaicRejectionReason(const uint8_t *data, size_t size,
                                uint64_t expectedHash, char *buf,
                                size_t bufSize);

#endif /* JAI_MODULE_INTERNAL_H */
