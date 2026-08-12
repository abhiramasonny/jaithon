/* module_internal.h — glue shared by the module-loading translation units.
 *
 * Private to src/runtime/modules/: nothing outside this directory should
 * include it. It exists only because module.c's core (the bootstrap window,
 * the importer, the top-level driver — see module.c's own header comment) was
 * kept together as one cohesive state machine, while the pieces that do not
 * touch that state — path resolution (module_path.c), .jaic cache-flag and
 * seed-gate computation (module_cache.c) — moved out to their own files. A
 * handful of small helpers cross that boundary in both directions, and this
 * header is where they are declared so each stays `static` everywhere else.
 */
#ifndef JAI_MODULE_INTERNAL_H
#define JAI_MODULE_INTERNAL_H

#include "runtime/runtime.h"

/* The one file extension a module's own name can end in; shared by
 * module.c's fileStem (naming a loaded module from its path) and
 * module_path.c's tryDirectory (turning a dotted name into a path). */
#define JAI_MODULE_EXT ".jai"

/* -- module.c, used by module_path.c --------------------------------- */

/* Report a resolution/import failure through whichever channel is live: a
 * catchable ImportError while the machine is running, an ordinary diagnostic
 * before it starts. Defined in module.c because every import failure — not
 * just the ones path resolution finds — goes through it. */
bool importFailure(JaiDiagCode code, ObjClass *klass, const char *fmt, ...)
    JAI_PRINTF(3, 4);

/* -- module_path.c, used by module.c ----------------------------------- */

/* Make sure the search path has been initialised at least once with default
 * settings. Every entry point that can run before jaiModulePathInit is called
 * explicitly (embedding, tests) goes through this first. */
void ensurePathReady(void);

/* True if `path` names a file (not a directory, not absent). */
bool isRegularFile(const char *path);

/* Absolute, symlink-resolved form of `candidate`, so two spellings of the
 * same file share one entry in vm.modules. Falls back to a verbatim copy if
 * resolution fails and `candidate` fits. */
bool storeResolved(char *out, size_t outSize, const char *candidate);

/* The name a module is known by: `dotted` with any leading relative-import
 * dots stripped. `.util` is the module `util`. */
const char *displayName(const char *dotted);

/* -- module_cache.c, used by module.c ---------------------------------- */

/* JAITHON_TRACE_LOAD=1: report where each module body came from. */
bool traceLoads(void);

/* JAITHON_NO_SEED=1: never consult the seed, even inside the bootstrap
 * window. */
bool seedDisabled(void);

/* The .jaic header flags this compilation would produce, from the codegen
 * options and whether the self-hosted front end is the producer. */
uint32_t cacheFlagsFor(const CodegenOptions *opts, bool selfHosted);

/* Whether the .jaic cache entry for `sourcePath` (if any) was written with
 * loadability-relevant flags matching `flags`. A cheap 8-byte header probe,
 * not a full load. */
bool cacheFlagsMatch(const char *sourcePath, uint32_t flags);
bool jaiCacheFlagsMatchBuffer(const uint8_t *head, size_t length,
                              uint32_t flags);

/* Why jaiDeserializeModule/jaiDeserializeSeed said no, in one sentence, by
 * re-reading the fixed header this build's writer would have produced.
 * `buf`/`bufSize` back a reason that needs to be formatted; a literal is
 * returned directly when no formatting is needed. */
const char *jaicRejectionReason(const uint8_t *data, size_t size,
                                uint64_t expectedHash, char *buf,
                                size_t bufSize);

#endif /* JAI_MODULE_INTERNAL_H */
