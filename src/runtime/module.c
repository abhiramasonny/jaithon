/* module.c — the module search path, the importer, and the top-level driver.
 *
 * A module is registered in vm.modules as MOD_LOADING before its body runs,
 * which is what makes an import cycle detectable (E0801) instead of infinite.
 */

#include <stdlib.h>

#include "runtime.h"
#include "boot/seed.h"
#include "frontend.h"
#include "../vm/jit.h"
#include "methods.h"

#include "../common/diag.h"
#include "../native/native.h"
#include "../vm/serialize.h"

CodegenOptions jaiCodegenDefaults(void) {
    CodegenOptions opts;
    opts.optLevel = 2;
    opts.debugInfo = true;
    opts.stripAsserts = false;
    opts.emitTailCalls = true;
    return opts;
}

#define JAI_MODULE_EXT   ".jai"
#define JAI_PACKAGE_FILE "mod.jai"

/* Enough to hold a legitimate package chain; a longer one is a runaway import
 * that would otherwise recurse the C stack (each level nests a VM run). */
#define JAI_MAX_IMPORT_DEPTH 64

/* Directories listed in an E0800 note before the list is elided. */
#define JAI_MAX_SEARCH_REPORTED 8

/* ------------------------------------------------------------------ */
/* Run options                                                          */
/* ------------------------------------------------------------------ */

/* jaiImportModule takes no options, so the ones the driver was given are kept
 * here and inherited by every module the entry module pulls in. */
static JaiRunOptions sOptions;
static bool          sOptionsSet;

/* True while the self-hosted front end is itself being imported: compiling it
 * with itself would recurse, so the C front end handles this window instead. */
static bool          sLoadingFrontEnd;
/* Set once the front end has been pulled in; see loadModuleBody. */
static bool          sFrontEndWarmed;

JaiRunOptions jaiRunDefaults(void) {
    JaiRunOptions o;
    memset(&o, 0, sizeof o);
    o.entryPath  = NULL;
    o.codegen    = jaiCodegenDefaults();
    o.useCache   = true;
    o.writeCache = true;
    /* The self-hosted front end is the default; `--front=c` stays so a
     * regression is bisectable rather than only observable. */
    o.selfHosted = true;
    o.checkOnly  = false;
    o.verbose    = false;
    return o;
}

static const JaiRunOptions *options(void) {
    if (!sOptionsSet) {
        sOptions = jaiRunDefaults();
        sOptionsSet = true;
    }
    return &sOptions;
}

static void setOptions(const JaiRunOptions *opts) {
    sOptions = opts != NULL ? *opts : jaiRunDefaults();
    sOptionsSet = true;
}

/* ------------------------------------------------------------------ */
/* Failure reporting                                                    */
/* ------------------------------------------------------------------ */

/* Messages stay short on purpose: jaiThrow renders into 512 bytes; the long
 * payload (the searched directory list) is elided by the caller, not here. */
static bool importFailure(JaiDiagCode code, ObjClass *klass, const char *fmt, ...)
    JAI_PRINTF(3, 4);

static bool importFailure(JaiDiagCode code, ObjClass *klass, const char *fmt, ...) {
    char message[512];
    va_list ap;
    va_start(ap, fmt);
    int written = vsnprintf(message, sizeof message, fmt, ap);
    va_end(ap);
    if (written < 0) message[0] = '\0';

    if (vm.frameCount > 0) {
        return jaiThrow(klass != NULL ? klass : vm.cImportError, "%s: %s",
                        jaiDiagCodeString(code), message);
    }
    (void)jaiDiagError(code, JAI_SPAN_NONE, "%s", message);
    return false;
}

/* ------------------------------------------------------------------ */
/* Search path                                                          */
/* ------------------------------------------------------------------ */

typedef JAI_VEC(char *) DirList;

/* sUserDirs (JAITHON_PATH + jaiModulePathAdd) and sLibDirs (the installed
 * library) hold C strings, not ObjStrings, so the path works before the VM exists. */
static DirList sUserDirs;
static DirList sLibDirs;
static bool    sPathReady;

static bool dirListHas(const DirList *list, const char *dir) {
    for (int i = 0; i < list->count; i++) {
        if (strcmp(list->data[i], dir) == 0) return true;
    }
    return false;
}

static void dirListAdd(DirList *list, const char *dir) {
    if (dir == NULL || dir[0] == '\0') return;

    /* Absolute form where possible: it makes the duplicate check meaningful
     * and keeps `..` out of the directory list an error message prints. */
    char absolute[JAI_MAX_PATH];
    const char *entry = jaiPathAbsolute(absolute, sizeof absolute, dir)
                            ? absolute
                            : dir;
    if (dirListHas(list, entry)) return;

    char *copy = jaiStrdup(entry);
    if (copy == NULL) return;
    JAI_VEC_PUSH(char *, list, copy);
}

static void dirListClear(DirList *list) {
    for (int i = 0; i < list->count; i++) {
        char *s = list->data[i];
        if (s != NULL) (void)jaiRealloc(s, strlen(s) + 1, 0);
    }
    list->count = 0;
}

/* vm.modulePath is a mirror: nothing resolves through it, so it is rebuilt
 * wholesale whenever the path changes. */
static void syncModulePathMirror(void) {
    if (vm.gc == NULL) return;   /* before jaiVMInit there is nothing to intern */

    vm.modulePath.count = 0;
    for (int i = 0; i < sUserDirs.count; i++) {
        JAI_VEC_PUSH(ObjString *, &vm.modulePath,
                     jaiStringInternC(sUserDirs.data[i]));
    }
    for (int i = 0; i < sLibDirs.count; i++) {
        JAI_VEC_PUSH(ObjString *, &vm.modulePath,
                     jaiStringInternC(sLibDirs.data[i]));
    }
}

static void addLibDir(const char *dir) {
    if (dir == NULL || dir[0] == '\0' || !jaiPathIsDir(dir)) return;
    dirListAdd(&sLibDirs, dir);
}

static void addLibDirRelative(const char *base, const char *suffix) {
    char candidate[JAI_MAX_PATH];
    jaiPathJoin(candidate, sizeof candidate, base, suffix);
    addLibDir(candidate);
}

void jaiModulePathInit(const char *execDir) {
    /* Idempotent: the derived library directories are recomputed and anything
     * added by hand in the meantime is kept. */
    dirListClear(&sLibDirs);
    sPathReady = true;

    const char *env = getenv("JAITHON_PATH");
    if (env != NULL) {
        const char *p = env;
        for (;;) {
            const char *sep = strchr(p, ':');
            size_t n = sep != NULL ? (size_t)(sep - p) : strlen(p);
            if (n > 0 && n < JAI_MAX_PATH) {
                char entry[JAI_MAX_PATH];
                memcpy(entry, p, n);
                entry[n] = '\0';
                dirListAdd(&sUserDirs, entry);
            }
            if (sep == NULL) break;
            p = sep + 1;
        }
    }

    /* `make install` puts lib in <prefix>/share/jaithon/lib; a build tree has
     * ./lib next to the binary. Both are derived from the binary's own path. */
    char derived[JAI_MAX_PATH];
    if (execDir == NULL || execDir[0] == '\0') {
        const char *exe = jaiExecutablePath();
        if (exe != NULL && exe[0] != '\0') {
            jaiPathDirname(derived, sizeof derived, exe);
            execDir = derived;
        }
    }
    if (execDir != NULL && execDir[0] != '\0') {
        addLibDirRelative(execDir, "lib");
        addLibDirRelative(execDir, "../lib");
        addLibDirRelative(execDir, "../share/jaithon/lib");
        addLibDirRelative(execDir, "../share/jaithon");
    }

    /* JAITHON_NO_DEFAULT_PATH exists so a test can't quietly pass by falling
     * back to an already-installed copy of the library. */
    if (getenv("JAITHON_NO_DEFAULT_PATH") == NULL) {
        addLibDir("/usr/local/share/jaithon/lib");
        addLibDir("/usr/local/share/jaithon");
        addLibDir("/opt/homebrew/share/jaithon/lib");
        addLibDir("/opt/homebrew/share/jaithon");
    }

    syncModulePathMirror();
}

void jaiModulePathAdd(const char *dir) {
    if (dir == NULL || dir[0] == '\0') return;
    dirListAdd(&sUserDirs, dir);
    syncModulePathMirror();
}

static void ensurePathReady(void) {
    if (!sPathReady) jaiModulePathInit(NULL);
}

/* ------------------------------------------------------------------ */
/* Dotted name -> file                                                  */
/* ------------------------------------------------------------------ */

/* Path components come from source text and must not escape a search
 * directory; non-ASCII is let through (spec §2.1), separators and `..` are not. */
static bool isNameByte(char c) {
    unsigned char u = (unsigned char)c;
    if (u >= 0x80) return true;
    return (u >= 'a' && u <= 'z') || (u >= 'A' && u <= 'Z') ||
           (u >= '0' && u <= '9') || u == '_';
}

static bool splitModuleName(const char *dotted, int *outDots, char *relative,
                            size_t relSize) {
    *outDots = 0;
    if (relSize > 0) relative[0] = '\0';

    if (dotted == NULL || dotted[0] == '\0') {
        return importFailure(E0804_INVALID_MODULE_PATH, vm.cImportError,
                             "empty module path");
    }

    const char *p = dotted;
    while (*p == '.') { (*outDots)++; p++; }
    if (*p == '\0') {
        return importFailure(E0804_INVALID_MODULE_PATH, vm.cImportError,
                             "module path '%s' names no module", dotted);
    }

    size_t pos = 0;
    while (*p != '\0') {
        size_t start = pos;
        while (*p != '\0' && *p != '.') {
            if (!isNameByte(*p)) {
                return importFailure(E0804_INVALID_MODULE_PATH, vm.cImportError,
                                     "module path '%s' is not a dotted name",
                                     dotted);
            }
            if (pos + 1 >= relSize) {
                return importFailure(E0804_INVALID_MODULE_PATH, vm.cImportError,
                                     "module path '%s' is too long", dotted);
            }
            relative[pos++] = *p++;
        }
        if (pos == start) {
            /* An empty component: "a..b", or a trailing dot. */
            return importFailure(E0804_INVALID_MODULE_PATH, vm.cImportError,
                                 "module path '%s' has an empty component",
                                 dotted);
        }
        if (*p == '.') {
            p++;
            if (pos + 1 >= relSize) {
                return importFailure(E0804_INVALID_MODULE_PATH, vm.cImportError,
                                     "module path '%s' is too long", dotted);
            }
            relative[pos++] = '/';
        }
    }
    relative[pos] = '\0';
    return true;
}

/* The name a module is known by. Leading dots are an instruction to the
 * resolver, not part of the identity: `.util` is the module `util`. */
static const char *displayName(const char *dotted) {
    const char *p = dotted;
    while (*p == '.') p++;
    return *p != '\0' ? p : dotted;
}

static bool isRegularFile(const char *path) {
    return path[0] != '\0' && jaiPathExists(path) && !jaiPathIsDir(path);
}

/* Absolute, symlink-resolved form of `candidate`, so that two spellings of the
 * same file share one entry in vm.modules. */
static bool storeResolved(char *out, size_t outSize, const char *candidate) {
    if (jaiPathAbsolute(out, outSize, candidate)) return true;
    size_t len = strlen(candidate);
    if (len + 1 > outSize) {
        out[0] = '\0';
        return false;
    }
    memcpy(out, candidate, len + 1);
    return true;
}

static bool tryDirectory(const char *dir, const char *relative, char *out,
                         size_t outSize) {
    char leaf[JAI_MAX_PATH];
    char candidate[JAI_MAX_PATH];

    int n = snprintf(leaf, sizeof leaf, "%s%s", relative, JAI_MODULE_EXT);
    if (n > 0 && (size_t)n < sizeof leaf) {
        jaiPathJoin(candidate, sizeof candidate, dir, leaf);
        if (isRegularFile(candidate)) return storeResolved(out, outSize, candidate);
    }

    n = snprintf(leaf, sizeof leaf, "%s/%s", relative, JAI_PACKAGE_FILE);
    if (n > 0 && (size_t)n < sizeof leaf) {
        jaiPathJoin(candidate, sizeof candidate, dir, leaf);
        if (isRegularFile(candidate)) return storeResolved(out, outSize, candidate);
    }
    return false;
}

static void noteSearched(JaiBuf *searched, int *count, const char *dir) {
    (*count)++;
    if (*count > JAI_MAX_SEARCH_REPORTED) return;
    if (searched->count > 0) jaiBufAppendStr(searched, ", ");
    jaiBufAppendStr(searched, dir);
}

/* Base directory of a relative import: one dot is the importer's directory,
 * each further dot climbs one level. */
static bool relativeBase(const char *fromDir, int dots, char *out,
                         size_t outSize) {
    const char *start = (fromDir != NULL && fromDir[0] != '\0') ? fromDir : ".";
    if (!storeResolved(out, outSize, start)) return false;

    for (int i = 1; i < dots; i++) {
        char parent[JAI_MAX_PATH];
        jaiPathDirname(parent, sizeof parent, out);
        if (parent[0] == '\0' || strcmp(parent, out) == 0) return false;
        if (!storeResolved(out, outSize, parent)) return false;
    }
    return true;
}

bool jaiResolveModulePath(const char *dottedName, const char *fromDir,
                          char *out, size_t outSize) {
    if (out == NULL || outSize == 0) return false;
    out[0] = '\0';
    ensurePathReady();

    int dots = 0;
    char relative[JAI_MAX_PATH];
    if (!splitModuleName(dottedName, &dots, relative, sizeof relative)) return false;

    JaiBuf searched;
    jaiBufInit(&searched);
    int searchedCount = 0;
    bool found = false;

    if (dots > 0) {
        char base[JAI_MAX_PATH];
        if (!relativeBase(fromDir, dots, base, sizeof base)) {
            jaiBufFree(&searched);
            return importFailure(E0804_INVALID_MODULE_PATH, vm.cImportError,
                                 "relative import '%s' climbs past the root",
                                 dottedName);
        }
        found = tryDirectory(base, relative, out, outSize);
        if (!found) noteSearched(&searched, &searchedCount, base);
    } else {
        if (fromDir != NULL && fromDir[0] != '\0') {
            found = tryDirectory(fromDir, relative, out, outSize);
            if (!found) noteSearched(&searched, &searchedCount, fromDir);
        }
        for (int i = 0; !found && i < sUserDirs.count; i++) {
            found = tryDirectory(sUserDirs.data[i], relative, out, outSize);
            if (!found) noteSearched(&searched, &searchedCount, sUserDirs.data[i]);
        }
        for (int i = 0; !found && i < sLibDirs.count; i++) {
            found = tryDirectory(sLibDirs.data[i], relative, out, outSize);
            if (!found) noteSearched(&searched, &searchedCount, sLibDirs.data[i]);
        }
    }

    if (found && out[0] != '\0') {
        jaiBufFree(&searched);
        return true;
    }
    if (found) {
        /* The file exists but its path does not fit in the caller's buffer. */
        jaiBufFree(&searched);
        return importFailure(E0804_INVALID_MODULE_PATH, vm.cImportError,
                             "path of module '%s' is too long", dottedName);
    }

    if (searchedCount > JAI_MAX_SEARCH_REPORTED) {
        jaiBufPrintf(&searched, " and %d more",
                     searchedCount - JAI_MAX_SEARCH_REPORTED);
    }
    jaiBufPush(&searched, '\0');
    const char *dirs = (searchedCount > 0 && searched.data != NULL)
                           ? (const char *)searched.data
                           : "no directories";

    if (vm.frameCount > 0) {
        (void)jaiThrow(vm.cImportError, "%s: cannot find module '%s'; searched %s",
                       jaiDiagCodeString(E0800_MODULE_NOT_FOUND), dottedName, dirs);
    } else {
        JaiDiag *d = jaiDiagError(E0800_MODULE_NOT_FOUND, JAI_SPAN_NONE,
                                  "cannot find module `%s`", dottedName);
        jaiDiagAddNote(d, "searched %s", dirs);
        if (sLibDirs.count == 0) {
            jaiDiagAddHelp(d, "no installed library was found; set JAITHON_PATH "
                              "to the directory holding `std`");
        }
    }
    jaiBufFree(&searched);
    return false;
}

/* Speculative variant: jaiResolveModulePath always reports a miss, but here a
 * "no" is a valid answer, so diagnostics/exception state is saved and restored. */
bool jaiResolveModulePathQuiet(const char *dottedName, const char *fromDir,
                               char *out, size_t outSize) {
    JaiDiagBag live = gDiags;
    jaiDiagInit(&gDiags);
    bool hadException = vm.hasException;

    bool found = jaiResolveModulePath(dottedName, fromDir, out, outSize);

    jaiDiagFree(&gDiags);
    gDiags = live;
    if (!hadException && vm.hasException) jaiClearException();
    return found;
}

/* ------------------------------------------------------------------ */
/* Front end                                                            */
/* ------------------------------------------------------------------ */

/* Registers a source buffer with the diagnostic engine, which takes ownership.
 * The copy must be exact -- an embedded NUL must not shorten it (jaiMemdup). */
static int registerSource(const char *path, const char *source, size_t length) {
    return jaiSourceAdd(path, jaiMemdup(source, length), length);
}

/* Compile a source string into a module body.
 *
 * A *fragment* is text compiled into a module that already exists; it defers
 * unresolved names instead of reporting E0200, unlike a fresh CLI module. */
ObjFunction *jaiCompileSource(const char *source, size_t length,
                              const char *path, ObjModule *module,
                              const CodegenOptions *opts) {
    if (source == NULL) return NULL;

    CodegenOptions defaults = jaiCodegenDefaults();
    if (opts == NULL) opts = &defaults;

    const char *label = path != NULL ? path : "<source>";
    int fileId = registerSource(label, source, length);
    if (module != NULL) module->sourceFileId = fileId;

    bool fragment = module != NULL &&
                    (module->state == MOD_LOADED || vm.frameCount > 0);

    JaiReplCompileOptions o;
    o.path = label;
    o.fileId = fileId;
    o.optLevel = opts->optLevel;
    o.echo = NULL;
    o.wholeFile = true;
    o.record = false;
    o.sourceDir = NULL;
    o.strict = false;
    o.lateGlobals = fragment;

    ObjFunction *body = jaiFrontEndReplCompile(source, length, &o, module, NULL);
    (void)jaiDiagFlush(&gDiags, stderr);
    return body;
}

/* ------------------------------------------------------------------ */
/* Cache handshake                                                      */
/* ------------------------------------------------------------------ */

/* JAIC_FLAG_SELFHOSTED records who produced the image, not whether --front=jai
 * was passed -- imports still compile via C even in that mode. */
/* `make reseed` sets this. The seed must not serve its own replacement's
 * compile, or nothing new lands in __jaicache__ and the seed stops advancing. */
/* JAITHON_TRACE_LOAD=1 reports where each module body came from: the seed can
 * serve a compiler whose source moved, silently invalidating measurements. */
static bool traceLoads(void) {
    const char *flag = getenv("JAITHON_TRACE_LOAD");
    return flag != NULL && flag[0] != '\0' && strcmp(flag, "0") != 0;
}


static bool seedDisabled(void) {
    const char *flag = getenv("JAITHON_NO_SEED");
    return flag != NULL && flag[0] != '\0' && strcmp(flag, "0") != 0;
}

/* Whether THIS compilation goes through the self-hosted front end. */
static bool selfHosting(void) {
    return sOptions.selfHosted && !sLoadingFrontEnd;
}

static uint32_t cacheFlagsFor(const CodegenOptions *opts, bool selfHosted) {
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

/* Probes the .jaic header (magic, u16 version, u16 flags; spec/BYTECODE.md §7)
 * so a debug-built cache can't be silently loaded into a --release run. */
static bool cacheFlagsMatch(const char *sourcePath, uint32_t flags) {
    char path[JAI_MAX_PATH];
    jaiCachePathFor(sourcePath, path, sizeof path);
    if (path[0] == '\0') return false;

    FILE *f = fopen(path, "rb");
    if (f == NULL) return false;
    uint8_t head[8];
    bool complete = fread(head, 1, sizeof head, f) == sizeof head;
    fclose(f);
    if (!complete) return false;

    if (memcmp(head, JAIC_MAGIC, 4) != 0) return false;
    uint16_t version = (uint16_t)((uint16_t)head[4] | (uint16_t)(head[5] << 8));
    uint16_t stored  = (uint16_t)((uint16_t)head[6] | (uint16_t)(head[7] << 8));

    /* JAIC_FLAG_SELFHOSTED (who produced the image) is excluded from this
     * comparison -- including it caused spurious cache misses during bootstrap. */
    const uint16_t kLoadability = (uint16_t)~(uint16_t)JAIC_FLAG_SELFHOSTED;
    return version == JAIC_VERSION &&
           (stored & kLoadability) == ((uint16_t)flags & kLoadability);
}

/* Reads `path`, then deserialises its cache or compiles it. The source is
 * registered first: a cache hit still needs the text for tracebacks. */
#define JAI_SELF_HOSTED_MODULE "jaithon.compile"

/* Pulls in the whole front end once. Must run before the triggering module is
 * marked MOD_LOADING (see maybeWarmFor), or warming mid-import falsely cycles. */
static void warmFrontEnd(void) {
    if (!sOptions.selfHosted || sLoadingFrontEnd || sFrontEndWarmed) return;
    sFrontEndWarmed = true;
    sLoadingFrontEnd = true;
    (void)jaiImportModule(JAI_SELF_HOSTED_MODULE, NULL);
    sLoadingFrontEnd = false;
    jaiClearException();
}

/* Warms on either a cache miss (safe to warm early; a missed warm cycles, see
 * above) or a lib/jaithon module (else the seed can mix two compiler generations). */
static bool cacheFlagsMatch(const char *sourcePath, uint32_t flags);

static bool maybeWarmFor(const char *path) {
    if (!sOptions.selfHosted || sLoadingFrontEnd || sFrontEndWarmed) return false;

    /* Matches the seed's library-relative path ("jaithon/ast.jai") rather than
     * the absolute path, which would match the whole repo (also named jaithon). */
    const JaiSeedEntry *seeded = seedDisabled() ? NULL : jaiSeedFind(path);
    bool ownedByFrontEnd = seeded != NULL &&
                           strncmp(seeded->module, "jaithon/", 8) == 0;
    if (!ownedByFrontEnd) {
        const JaiRunOptions *opts = options();
        uint32_t flags = cacheFlagsFor(&opts->codegen, true);
        if (opts->useCache && cacheFlagsMatch(path, flags)) return false;
    }
    warmFrontEnd();
    return true;
}

static ObjFunction *loadModuleBody(ObjModule *module, const char *path) {
    size_t length = 0;
    char *text = jaiReadFile(path, &length);
    if (text == NULL) {
        (void)importFailure(E0800_MODULE_NOT_FOUND, vm.cIOError,
                            "cannot read module file '%s'", path);
        return NULL;
    }

    uint64_t hash = jaiSourceHash(text, length);
    int fileId = jaiSourceAdd(path, text, length);   /* takes ownership of text */
    module->sourceFileId = fileId;

    const JaiSourceFile *file = jaiSourceGet(fileId);
    if (file == NULL) {
        (void)importFailure(E0902_INTERNAL_ERROR, vm.cImportError,
                            "cannot register source for '%s'", path);
        return NULL;
    }

    const JaiRunOptions *opts = options();
    uint32_t flags = cacheFlagsFor(&opts->codegen, selfHosting());

    if (opts->useCache && cacheFlagsMatch(path, flags)) {
        ObjFunction *cached = jaiCacheLoad(path, module, hash);
        if (cached != NULL) {
            if (traceLoads()) fprintf(stderr, "load cache   %s\n", path);
            return cached;
        }
        /* Stale, corrupt, or from another compiler: recompile silently. */
    }

    /* Inside the bootstrap window the compiler's own closure has no compiler to
     * call, so it comes from the seed. A miss falls through to the front end:
     * a tree whose sources have moved past the seed recompiles rather than
     * running stale bytecode, because jaiDeserializeModule checks the source
     * hash. */
    if (sLoadingFrontEnd && !seedDisabled()) {
        const JaiSeedEntry *seeded = jaiSeedFind(path);
        if (seeded != NULL) {
            ObjFunction *fromSeed = jaiDeserializeSeed(seeded->image,
                                                       seeded->length,
                                                       module, hash);
            if (fromSeed != NULL) {
                if (traceLoads()) fprintf(stderr, "load seed    %s\n", path);
                return fromSeed;
            }
        }
    }

    ObjFunction *body = NULL;
    if (selfHosting()) {
        body = jaiSelfHostedCompileInto(file->source, length, path, module,
                                        hash, opts->codegen.optLevel);
        if (body == NULL) return NULL;
        /* Self-hosted bodies are named from the file stem (`b`, not `sub.b`);
         * a cached body with that name can't resolve its own imports later. */
        if (module->name != NULL) {
            body->name = module->name;
            body->qualifiedName = module->name;
        }
    } else {
        /* Unreachable: the branch above is the only front end that remains;
         * nothing makes selfHosting() false outside the bootstrap window. */
        JAI_PANIC("no front end for `%s`", path);
    }

    if (opts->writeCache) {
        jaiPushRoot(OBJ_VAL(body));
        (void)jaiCacheStore(path, module, body, hash, flags);   /* best effort */
        jaiPopRoot();
    }
    return body;
}

/* ------------------------------------------------------------------ */
/* Import                                                               */
/* ------------------------------------------------------------------ */

typedef struct {
    char *name;
    char *path;
} ImportFrame;

static JAI_VEC(ImportFrame) sImportStack;

static void importStackPush(const char *name, const char *path) {
    ImportFrame frame;
    frame.name = jaiStrdup(name);
    frame.path = jaiStrdup(path);
    JAI_VEC_PUSH(ImportFrame, &sImportStack, frame);
}

static void importStackPop(void) {
    if (sImportStack.count <= 0) return;
    ImportFrame frame = JAI_VEC_POP(&sImportStack);
    if (frame.name != NULL) (void)jaiRealloc(frame.name, strlen(frame.name) + 1, 0);
    if (frame.path != NULL) (void)jaiRealloc(frame.path, strlen(frame.path) + 1, 0);
}

/* "std.a -> std.b -> std.a": the chain from the first appearance of the module
 * that is being imported again, so the cycle itself is what the user reads. */
static bool reportCycle(const char *name, const char *path) {
    JaiBuf chain;
    jaiBufInit(&chain);

    int start = 0;
    for (int i = 0; i < sImportStack.count; i++) {
        if (sImportStack.data[i].path != NULL &&
            strcmp(sImportStack.data[i].path, path) == 0) {
            start = i;
            break;
        }
    }
    for (int i = start; i < sImportStack.count; i++) {
        const char *step = sImportStack.data[i].name;
        jaiBufAppendStr(&chain, step != NULL ? step : "?");
        jaiBufAppendStr(&chain, " -> ");
    }
    jaiBufAppendStr(&chain, name);
    jaiBufPush(&chain, '\0');

    const char *text = chain.data != NULL ? (const char *)chain.data : name;
    (void)importFailure(E0801_CIRCULAR_IMPORT, vm.cImportError,
                        "circular import: %s", text);
    jaiBufFree(&chain);
    return false;
}

/* Create the module object and publish it before its body runs, so that a
 * cycle finds a MOD_LOADING module instead of recursing forever. */
static ObjModule *createModule(const char *name, ObjString *pathKey) {
    jaiPushRoot(OBJ_VAL(pathKey));
    ObjString *nameStr = jaiStringInternC(name);
    jaiPushRoot(OBJ_VAL(nameStr));

    ObjModule *module = jaiModuleNew(nameStr, pathKey);
    jaiPushRoot(OBJ_VAL(module));
    module->state = MOD_LOADING;
    (void)jaiTableSetInterned(&vm.modules, pathKey, OBJ_VAL(module));
    jaiPopRoots(3);
    return module;
}

static void forgetModule(ObjString *pathKey) {
    (void)jaiTableDelete(&vm.modules, OBJ_VAL(pathKey));
}

/* jaiVMRunModule resets the interpreter stack -- fine for the entry module,
 * but it would discard a live import's frame, so a running VM calls instead. */
static bool runModuleBody(ObjModule *module, ObjFunction *body) {
    if (vm.frameCount == 0) {
        return jaiVMRunModule(module, body) == JAI_RUN_OK;
    }

    body->module = module;
    ObjClosure *closure = jaiClosureNew(body);
    module->body = closure;

    Value ignored = NULL_VAL;
    if (!jaiCallValue(OBJ_VAL(closure), 0, NULL, &ignored)) {
        module->state = MOD_FAILED;
        return false;
    }
    module->state = MOD_LOADED;
    return true;
}

ObjModule *jaiImportModule(const char *dottedName, const char *fromDir) {
    jaiJitStartSampling();
    if (vm.builtins == NULL) JAI_PANIC("jaiImportModule before jaiVMInit");
    ensurePathReady();

    char path[JAI_MAX_PATH];
    if (!jaiResolveModulePath(dottedName, fromDir, path, sizeof path)) return NULL;

    const char *name = displayName(dottedName);
    ObjString *pathKey = jaiStringIntern(path, strlen(path));

    ObjModule *module = NULL;
    Value existing;
    if (jaiTableGetInterned(&vm.modules, pathKey, &existing) &&
        IS_MODULE(existing)) {
        ObjModule *known = AS_MODULE(existing);
        switch (known->state) {
        case MOD_LOADED:
            return known;
        case MOD_LOADING:
            (void)reportCycle(name, path);
            return NULL;
        case MOD_FAILED:
            (void)importFailure(E0800_MODULE_NOT_FOUND, vm.cImportError,
                                "module '%s' failed to load earlier", name);
            return NULL;
        case MOD_UNLOADED:
            /* Registered but never started. Reuse the object: something may
             * already be holding a reference to it. */
            module = known;
            module->state = MOD_LOADING;
            break;
        }
    }

    if (sImportStack.count >= JAI_MAX_IMPORT_DEPTH) {
        (void)importFailure(E0801_CIRCULAR_IMPORT, vm.cImportError,
                            "import of '%s' nests more than %d deep", name,
                            JAI_MAX_IMPORT_DEPTH);
        return NULL;
    }

    /* Runs before createModule (else the compiler's own import of this module
     * looks like a cycle); pathKey is rooted because warming can GC it away. */
    jaiPushRoot(OBJ_VAL(pathKey));
    bool warmed = maybeWarmFor(path);
    jaiPopRoot();

    /* Warming can itself load this module, so re-check here -- otherwise a
     * duplicate ObjModule is built (symptom: KeyError on NodeKind). */
    if (warmed &&
        jaiTableGetInterned(&vm.modules, pathKey, &existing) &&
        IS_MODULE(existing) && AS_MODULE(existing)->state == MOD_LOADED) {
        return AS_MODULE(existing);
    }

    if (module == NULL) module = createModule(name, pathKey);
    importStackPush(name, path);

    ObjFunction *body = loadModuleBody(module, path);
    bool ok = false;
    if (body != NULL) {
        jaiPushRoot(OBJ_VAL(body));
        ok = runModuleBody(module, body);
        jaiPopRoot();
    }

    importStackPop();

    if (!ok) {
        module->state = MOD_FAILED;
        /* Drop the registration so a later attempt reports the real error
         * again instead of "failed earlier". */
        forgetModule(pathKey);
        if (!vm.hasException && vm.frameCount > 0) {
            (void)jaiThrow(vm.cImportError, "%s: module '%s' failed to load",
                           jaiDiagCodeString(E0800_MODULE_NOT_FOUND), name);
        }
        return NULL;
    }

    module->state = MOD_LOADED;
    return module;
}

ObjModule *jaiImportFrontEndModule(const char *dottedName) {
    bool wasLoading = sLoadingFrontEnd;
    sLoadingFrontEnd = true;
    ObjModule *module = jaiImportModule(dottedName, NULL);
    sLoadingFrontEnd = wasLoading;
    return module;
}

/* ------------------------------------------------------------------ */
/* Prelude                                                              */
/* ------------------------------------------------------------------ */

static bool sPreludeLoaded;
static bool sPreludeTried;

static bool preludeDisabled(void) {
    const char *flag = getenv("JAITHON_NO_PRELUDE");
    return flag != NULL && flag[0] != '\0' && strcmp(flag, "0") != 0;
}

bool jaiLoadPrelude(void) {
    if (sPreludeLoaded) return true;
    if (sPreludeTried) return false;   /* one warning per process, not per import */
    sPreludeTried = true;

    if (vm.builtins == NULL) JAI_PANIC("jaiLoadPrelude before jaiVMInit");
    ensurePathReady();

    ObjModule *prelude = jaiImportModule("std.prelude", NULL);
    if (prelude == NULL) {
        /* A tree without lib/std still works (prelude only re-exports optional
         * names); the error is printed then dropped, not left pending. */
        jaiClearException();
        (void)jaiDiagFlush(&gDiags, stderr);
        fprintf(stderr, "jaithon: warning: could not load std.prelude; the "
                        "names of spec §9's std.core are unavailable\n");
        return false;
    }

    /* The prelude's export table (spec §9) is exactly the implicit global
     * scope; jaiDefineGlobal also registers each name so E0200 doesn't fire. */
    int slot = 0;
    Value key, unused;
    while (jaiTableNext(&prelude->exports, &slot, &key, &unused)) {
        if (!IS_STRING(key)) continue;
        ObjString *exported = AS_STRING(key);
        /* Length-bounded: a string can be a view into a shared buffer. */
        if (memchr(exported->chars, '.', exported->length) != NULL) continue;

        Value value;
        if (!jaiModuleGet(prelude, exported, &value)) continue;   /* already builtin */
        jaiDefineGlobal(exported->chars, value);
    }

    sPreludeLoaded = true;
    return true;
}

/* ------------------------------------------------------------------ */
/* Entry point                                                          */
/* ------------------------------------------------------------------ */

/* Builds argv (script path first, per spec §8.4) into the global `__argv__`,
 * which `__prim__.os_argv` reads -- libc can't hand argv back portably. */
static ObjList *installArgv(const char *scriptPath, int argc, char **argv) {
    int extra = argc > 0 ? argc : 0;
    ObjList *list = jaiListNew(extra + 1);
    jaiPushRoot(OBJ_VAL(list));

    ObjString *script = jaiStringNew(scriptPath, strlen(scriptPath));
    jaiPushRoot(OBJ_VAL(script));
    jaiListPush(list, OBJ_VAL(script));
    jaiPopRoot();

    for (int i = 0; i < extra; i++) {
        const char *arg = (argv != NULL) ? argv[i] : NULL;
        if (arg == NULL) continue;
        ObjString *s = jaiStringNew(arg, strlen(arg));
        jaiPushRoot(OBJ_VAL(s));
        jaiListPush(list, OBJ_VAL(s));
        jaiPopRoot();
    }

    jaiDefineGlobal("__argv__", OBJ_VAL(list));
    jaiPopRoot();
    return list;
}

static bool isCallableValue(Value v) {
    if (!IS_OBJ(v)) return false;
    switch (OBJ_TYPE(v)) {
    case OBJ_CLOSURE:
    case OBJ_FUNCTION:
    case OBJ_NATIVE:
    case OBJ_BOUND:
        return true;
    default:
        return false;
    }
}

/* How many arguments `main` wants: spec §8.4 allows main() and main(args). */
static int calleeArity(Value v) {
    if (IS_CLOSURE(v)) return (int)AS_CLOSURE(v)->fn->arity;
    if (IS_FUNCTION(v)) return (int)AS_FUNCTION(v)->arity;
    if (IS_NATIVE(v)) return (int)AS_NATIVE(v)->minArity;
    if (IS_BOUND(v)) return calleeArity(AS_BOUND(v)->method);
    return 0;
}

static int exitCodeOf(Value result) {
    if (!IS_INT(result)) return 0;
    /* A process status is eight bits; wrap the way exit() does rather than
     * turning 256 into a silent success. */
    return (int)(AS_INT(result) & 0xFF);
}

static int callMain(ObjModule *module, ObjList *args) {
    ObjString *name = vm.strMain != NULL ? vm.strMain : jaiStringInternC("main");
    Value entry;
    if (!jaiModuleGet(module, name, &entry)) return 0;
    if (!isCallableValue(entry)) return 0;   /* a variable named main is not one */

    Value argument = OBJ_VAL(args);
    Value result = NULL_VAL;
    int argc = calleeArity(entry) >= 1 ? 1 : 0;

    if (!jaiCallValue(entry, argc, &argument, &result)) {
        if (vm.hasException) {
            jaiReportUncaught(vm.pendingException);
            jaiClearException();
        }
        return 1;
    }
    return exitCodeOf(result);
}

static void reportTiming(double load, double run, double total) {
    fprintf(stderr, "jaithon: load %.3f ms | run %.3f ms | total %.3f ms\n",
            load * 1e3, run * 1e3, total * 1e3);
}

/* ------------------------------------------------------------------ */
/* The self-hosted front end (--front=jai)                              */
/* ------------------------------------------------------------------ */

/* --front=jai hands over only the entry module (imports still load via the C
 * front end); nothing here silently falls back to C on failure. */

#define JAI_SELF_HOSTED_ENTRY  "compile_source"

static bool instanceField(Value v, const char *name, Value *out) {
    if (!IS_INSTANCE(v)) return false;
    ObjInstance *inst = AS_INSTANCE(v);
    int slot = jaiClassFieldSlot(inst->klass, jaiStringInternC(name));
    if (slot < 0 || slot >= (int)inst->fieldCount) return false;
    *out = inst->fields[slot];
    return true;
}

/* `Compiled.image` is a list of byte-sized ints (spec/BYTECODE.md §7 as seen
 * from Jaithon, which has no writable bytes type). */
static ObjBytes *bytesFromByteList(ObjList *list, const char *path) {
    size_t count = list->count > 0 ? (size_t)list->count : 0;
    uint8_t *raw = JAI_ALLOC(uint8_t, count > 0 ? count : 1);

    for (size_t i = 0; i < count; i++) {
        Value item = list->items[i];
        if (!IS_INT(item)) {
            JAI_FREE_ARRAY(uint8_t, raw, count > 0 ? count : 1);
            (void)jaiDiagError(E0902_INTERNAL_ERROR, JAI_SPAN_NONE,
                               "%s: the self-hosted front end put %s at byte %zu "
                               "of its .jaic image", path,
                               jaiTypeNameStatic(item), i);
            return NULL;
        }
        raw[i] = (uint8_t)(AS_INT(item) & 0xFF);
    }

    ObjBytes *bytes = jaiBytesNew(raw, count);
    JAI_FREE_ARRAY(uint8_t, raw, count > 0 ? count : 1);
    return bytes;
}

static ObjBytes *selfHostedImage(const char *source, size_t length,
                                 const char *path, int optLevel, int fileId) {
    /* The compiler's own closure is compiled by C (later, served by the seed):
     * compiling it with itself is the recursion this guard exists to stop. */
    bool wasLoading = sLoadingFrontEnd;
    sLoadingFrontEnd = true;
    ObjModule *compiler = jaiImportModule(JAI_SELF_HOSTED_MODULE, NULL);
    sLoadingFrontEnd = wasLoading;
    if (compiler == NULL) {
        jaiClearException();
        JaiDiag *d = jaiDiagError(E0800_MODULE_NOT_FOUND, JAI_SPAN_NONE,
                                  "--front=jai needs the self-hosted front end "
                                  "in `%s`, which could not be imported",
                                  JAI_SELF_HOSTED_MODULE);
        jaiDiagAddHelp(d, "install the standard library, or point JAITHON_PATH "
                          "at the directory that contains `std`");
        return NULL;
    }
    jaiPushRoot(OBJ_VAL(compiler));

    Value entry;
    if (!jaiModuleGet(compiler, jaiStringInternC(JAI_SELF_HOSTED_ENTRY), &entry)) {
        (void)jaiDiagError(E0802_NOT_EXPORTED, JAI_SPAN_NONE,
                           "`%s` does not export `%s(source, path)`",
                           JAI_SELF_HOSTED_MODULE, JAI_SELF_HOSTED_ENTRY);
        jaiPopRoot();
        return NULL;
    }

    /* Rooted: interning the second value can collect the first. optLevel and
     * fileId aren't placeholders -- a stale optLevel once silently ignored
     * -O0, and fileId=0 broke every span's line number in tracebacks. */
    Value args[5];
    args[0] = OBJ_VAL(jaiStringNew(source, length));
    jaiPushRoot(args[0]);
    args[1] = OBJ_VAL(jaiStringInternC(path));
    jaiPushRoot(args[1]);
    args[2] = BOOL_VAL(false);         /* release  */
    args[3] = INT_VAL(fileId);         /* fileId   */
    args[4] = INT_VAL(optLevel);

    Value produced = NULL_VAL;
    bool called = jaiCallValue(entry, 5, args, &produced);
    jaiPopRoots(2);

    if (!called) {
        jaiReportUncaught(vm.pendingException);
        jaiClearException();
        (void)jaiDiagError(E0902_INTERNAL_ERROR, JAI_SPAN_NONE,
                           "%s: the self-hosted front end raised while "
                           "compiling this file (traceback above)", path);
        jaiPopRoot();
        return NULL;
    }

    jaiPushRoot(produced);
    ObjBytes *image = NULL;

    /* Accepts both a plain `bytes` (the shape compile_source is expected to
     * settle on) and today's `Compiled` record, so the module can evolve
     * without a silent fallback to C. */
    Value field;
    if (IS_BYTES(produced)) {
        image = AS_BYTES(produced);
    } else if (instanceField(produced, "image", &field) && IS_LIST(field)) {
        if (AS_LIST(field)->count == 0) {
            /* The front end's own diagnostics (flushed like C's) say why; E0902
             * fires only if it produced neither an image nor a reason. */
            if (!jaiFrontEndTransferDiagnostics(produced)) {
                (void)jaiDiagError(E0902_INTERNAL_ERROR, JAI_SPAN_NONE,
                                   "%s: the self-hosted front end emitted "
                                   "neither an image nor a diagnostic", path);
            }
        } else {
            image = bytesFromByteList(AS_LIST(field), path);
        }
    } else {
        (void)jaiDiagError(E0902_INTERNAL_ERROR, JAI_SPAN_NONE,
                           "%s: `%s.%s` returned %s; expected `bytes` or a "
                           "record with an `image` field", path,
                           JAI_SELF_HOSTED_MODULE, JAI_SELF_HOSTED_ENTRY,
                           jaiTypeNameStatic(produced));
    }

    jaiPopRoots(2);   /* produced, compiler */
    return image;
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

/* jaiDeserializeModule only answers yes/no (a cache miss just means recompile),
 * so this re-reads the fixed header to name which field actually mismatched. */
static const char *jaicRejectionReason(const uint8_t *data, size_t size,
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

ObjFunction *jaiSelfHostedCompileInto(const char *source, size_t length,
                                      const char *path, ObjModule *module,
                                      uint64_t hash, int optLevel) {
    ObjBytes *image = selfHostedImage(source, length, path, optLevel,
                                      module != NULL ? module->sourceFileId : 0);
    if (image == NULL) return NULL;

    /* The image stays rooted across the diagnostic: rendering one allocates,
     * and the reason is read back out of these very bytes. */
    jaiPushRoot(OBJ_VAL(image));
    ObjFunction *body = jaiDeserializeModule(image->data, image->length, module,
                                             hash);

    if (body == NULL) {
        char why[256];
        const char *reason = jaicRejectionReason(image->data, image->length,
                                                 hash, why, sizeof why);
        JaiDiag *d = jaiDiagError(E0902_INTERNAL_ERROR, JAI_SPAN_NONE,
                                  "%s: the self-hosted front end produced a "
                                  ".jaic image this build cannot load", path);
        jaiDiagAddNote(d, "%s", reason);
        jaiDiagAddHelp(d, "`make bootstrap` compares the two front ends field "
                          "by field and reports the first divergence");
    }
    jaiPopRoot();
    return body;
}

/* --front=jai counterpart of loadModuleBody: same contract, but compiles via
 * lib/jaithon/compile. JAIC_FLAG_SELFHOSTED lets either front end's cache
 * entries coexist without cross-compiler mix-ups. */
static ObjFunction *selfHostedModuleBody(ObjModule *module, const char *path) {
    size_t length = 0;
    char *text = jaiReadFile(path, &length);
    if (text == NULL) {
        (void)jaiDiagError(E0800_MODULE_NOT_FOUND, JAI_SPAN_NONE,
                           "cannot read module file '%s'", path);
        return NULL;
    }

    uint64_t hash = jaiSourceHash(text, length);
    int fileId = jaiSourceAdd(path, text, length);   /* takes ownership of text */
    module->sourceFileId = fileId;

    const JaiSourceFile *file = jaiSourceGet(fileId);
    if (file == NULL) {
        (void)jaiDiagError(E0902_INTERNAL_ERROR, JAI_SPAN_NONE,
                           "cannot register source for '%s'", path);
        return NULL;
    }

    const JaiRunOptions *opts = options();
    uint32_t flags = cacheFlagsFor(&opts->codegen, true);

    if (opts->useCache && cacheFlagsMatch(path, flags)) {
        ObjFunction *cached = jaiCacheLoad(path, module, hash);
        if (cached != NULL) {
            if (traceLoads()) fprintf(stderr, "load cache   %s\n", path);
            return cached;
        }
        /* Stale, corrupt, or from the other front end: compile it. */
    }

    ObjFunction *body = jaiSelfHostedCompileInto(file->source, length, path,
                                                 module, hash,
                                                 opts->codegen.optLevel);
    if (body == NULL) return NULL;

    if (opts->writeCache) {
        jaiPushRoot(OBJ_VAL(body));
        (void)jaiCacheStore(path, module, body, hash, flags);   /* best effort */
        jaiPopRoot();
    }
    return body;
}

int jaiRunFile(const char *path, const JaiRunOptions *opts, int argc,
               char **argv) {
    if (vm.builtins == NULL) JAI_PANIC("jaiRunFile before jaiVMInit");

    setOptions(opts);
    ensurePathReady();

    const char *entry = (path != NULL && path[0] != '\0') ? path
                                                          : sOptions.entryPath;
    if (entry == NULL || entry[0] == '\0') {
        (void)jaiDiagError(E0800_MODULE_NOT_FOUND, JAI_SPAN_NONE,
                           "no input file");
        (void)jaiDiagFlush(&gDiags, stderr);
        return 1;
    }
    if (sOptions.checkOnly) return jaiCheckFile(entry, opts);

    char absolute[JAI_MAX_PATH];
    if (!jaiPathAbsolute(absolute, sizeof absolute, entry) ||
        !isRegularFile(absolute)) {
        (void)jaiDiagError(E0800_MODULE_NOT_FOUND, JAI_SPAN_NONE,
                           "cannot open `%s`", entry);
        (void)jaiDiagFlush(&gDiags, stderr);
        return 1;
    }

    double started = jaiClockMonotonic();

    ObjList *args = installArgv(absolute, argc, argv);
    if (!preludeDisabled()) (void)jaiLoadPrelude();

    maybeWarmFor(absolute);

    ObjString *pathKey = jaiStringIntern(absolute, strlen(absolute));
    ObjModule *module = createModule("__main__", pathKey);
    vm.mainModule = module;

    importStackPush("__main__", absolute);
    ObjFunction *body = sOptions.selfHosted
                            ? selfHostedModuleBody(module, absolute)
                            : loadModuleBody(module, absolute);
    if (body == NULL) (void)jaiDiagFlush(&gDiags, stderr);
    double compiled = jaiClockMonotonic();

    bool ok = false;
    if (body != NULL) {
        jaiPushRoot(OBJ_VAL(body));
        ok = runModuleBody(module, body);
        jaiPopRoot();
    }
    importStackPop();

    if (!ok) {
        module->state = MOD_FAILED;
        forgetModule(pathKey);
        if (vm.hasException) {
            jaiReportUncaught(vm.pendingException);
            jaiClearException();
        }
        if (sOptions.verbose) {
            double now = jaiClockMonotonic();
            reportTiming(compiled - started, now - compiled, now - started);
        }
        return 1;
    }

    module->state = MOD_LOADED;
    int code = callMain(module, args);
    double finished = jaiClockMonotonic();

    if (sOptions.verbose) {
        reportTiming(compiled - started, finished - compiled, finished - started);
    }
    return code;
}

/* ------------------------------------------------------------------ */
/* Static import graph                                                  */
static void fileStem(char *out, size_t outSize, const char *path) {
    jaiPathBasename(out, outSize, path);
    size_t len = strlen(out);
    size_t ext = strlen(JAI_MODULE_EXT);
    if (len > ext && memcmp(out + len - ext, JAI_MODULE_EXT, ext) == 0) {
        out[len - ext] = '\0';
    }
}

/* fileStem, with `__main__` as fallback for an empty stem. A second copy
 * lives in `module_name_for` (lib/jaithon/compile/mod.jai) and must agree
 * with this one, or a cached module can't resolve its own imports. */
void jaiModuleNameFor(const char *path, char *out, size_t outSize) {
    fileStem(out, outSize, path);
    if (out[0] == '\0') snprintf(out, outSize, "__main__");
}

/* ------------------------------------------------------------------ */
/* Import cycles                                                        */
/* ------------------------------------------------------------------ */

/* A cycle is a whole-program property, not a per-file one, so `check` asks
 * this (via `import_cycles` in lib/jaithon/compile/mod.jai) before the front
 * end -- otherwise every missing export in the cycle triggers its own E0200. */
static void checkImportCycles(const char *path, int fileId) {
    (void)fileId;
    Value arg = OBJ_VAL(jaiStringInternC(path));
    jaiPushRoot(arg);
    Value produced = NULL_VAL;
    bool asked = jaiFrontEndInvoke(JAI_SELF_HOSTED_MODULE, "import_cycles", 1,
                                   &arg, &produced);
    jaiPopRoot();
    if (!asked) return;

    jaiPushRoot(produced);
    (void)jaiFrontEndTransferDiagnostics(produced);
    jaiPopRoot();
}

/* Nothing to release: the graph the C built is gone and the front end owns
 * whatever it allocates. Kept because the CLI calls it. */
void jaiImportGraphFree(void) {
}

int jaiCheckFile(const char *path, const JaiRunOptions *opts) {
    if (vm.builtins == NULL) JAI_PANIC("jaiCheckFile before jaiVMInit");

    setOptions(opts);
    ensurePathReady();

    const char *entry = (path != NULL && path[0] != '\0') ? path
                                                          : sOptions.entryPath;
    if (entry == NULL || entry[0] == '\0') {
        (void)jaiDiagError(E0800_MODULE_NOT_FOUND, JAI_SPAN_NONE, "no input file");
        (void)jaiDiagFlush(&gDiags, stderr);
        return 1;
    }

    char absolute[JAI_MAX_PATH];
    if (!jaiPathAbsolute(absolute, sizeof absolute, entry)) {
        (void)storeResolved(absolute, sizeof absolute, entry);
    }

    size_t length = 0;
    char *text = jaiReadFile(absolute, &length);
    if (text == NULL) {
        (void)jaiDiagError(E0800_MODULE_NOT_FOUND, JAI_SPAN_NONE,
                           "cannot read `%s`", entry);
        (void)jaiDiagFlush(&gDiags, stderr);
        return 1;
    }
    int fileId = jaiSourceAdd(absolute, text, length);   /* takes ownership */

    /* Runs before the front end so a cycle is reported once, not as an E0200 storm. */
    checkImportCycles(absolute, fileId);

    /* Checking needs a module for the resolver to hang globals off, but it must
     * not join vm.modules: nothing here runs, so the module would be a shell
     * that a later import mistook for a loaded one. */
    char stem[JAI_MAX_PATH];
    fileStem(stem, sizeof stem, absolute);

    ObjString *nameStr = jaiStringInternC(stem[0] != '\0' ? stem : "<check>");
    jaiPushRoot(OBJ_VAL(nameStr));
    ObjString *pathStr = jaiStringIntern(absolute, strlen(absolute));
    jaiPushRoot(OBJ_VAL(pathStr));
    ObjModule *module = jaiModuleNew(nameStr, pathStr);
    /* Stamped into every span the front end reports; without it, diagnostics
     * render with no source line or caret (file 0). */
    module->sourceFileId = fileId;
    jaiPushRoot(OBJ_VAL(module));

    bool ok;
    /* `check` promises "compile and type-check" (spec 8.5), and codegen is
     * where the bytecode verifier runs, so it emits too and drops the body.
     * One front end, one branch. */
    const JaiSourceFile *entryFile = jaiSourceGet(fileId);
    ObjFunction *checked = entryFile == NULL
                             ? NULL
                             : jaiSelfHostedCompileInto(
                                   entryFile->source, entryFile->length,
                                   absolute, module,
                                   jaiSourceHash(entryFile->source,
                                                 entryFile->length),
                                   sOptions.codegen.optLevel);
    ok = checked != NULL && !jaiDiagHasErrors(&gDiags);
    (void)jaiDiagFlush(&gDiags, stderr);
    jaiPopRoots(3);

    if (sOptions.verbose && ok) {
        fprintf(stderr, "jaithon: %s is clean\n", entry);
    }
    return ok ? 0 : 1;
}

/* ------------------------------------------------------------------ */
/* Module members                                                      */
/* ------------------------------------------------------------------ */

/* Reflective path only (dir(), or a C caller on the method tables) --
 * `mod.thing` resolves directly in the VM. An un-annotated module exposes
 * everything (spec §8.2). */

static bool moduleExposes(ObjModule *m, ObjString *name, Value *out) {
    /* Reflective: `name` came from the caller, and a module table is keyed by
     * pointer. */
    name = jaiStringCanonical(name);
    if (!jaiModuleGet(m, name, out)) return false;
    return m->exports.count == 0 || jaiModuleIsExported(m, name);
}

static bool requireModuleArg(Value v, const char *fnName, ObjModule **out) {
    if (!IS_MODULE(v))
        return jaiThrow(vm.cTypeError, "%s() expected a module, not %s", fnName,
                        jaiTypeNameStatic(v));
    *out = AS_MODULE(v);
    return true;
}

static bool nModuleName(int argc, Value *args, Value *out) {
    (void)argc;
    ObjModule *m;
    if (!requireModuleArg(args[0], "name", &m)) return false;
    *out = m->name != NULL ? OBJ_VAL(m->name) : OBJ_VAL(jaiStringIntern("", 0));
    return true;
}

static bool nModulePath(int argc, Value *args, Value *out) {
    (void)argc;
    ObjModule *m;
    if (!requireModuleArg(args[0], "path", &m)) return false;
    *out = m->path != NULL ? OBJ_VAL(m->path) : OBJ_VAL(jaiStringIntern("", 0));
    return true;
}

static bool nModuleHas(int argc, Value *args, Value *out) {
    (void)argc;
    ObjModule *m;
    ObjString *name;
    if (!requireModuleArg(args[0], "has", &m)) return false;
    if (!jaiArgString(args[1], 2, "has", &name)) return false;
    Value ignored;
    *out = BOOL_VAL(moduleExposes(m, name, &ignored));
    return true;
}

/* get(name, default = null): the reflective read, so a missing name is a value
 * rather than an exception — a caller who wants the raise uses `mod.name`. */
static bool nModuleGet(int argc, Value *args, Value *out) {
    ObjModule *m;
    ObjString *name;
    if (!requireModuleArg(args[0], "get", &m)) return false;
    if (!jaiArgString(args[1], 2, "get", &name)) return false;

    if (moduleExposes(m, name, out)) return true;
    *out = argc >= 3 ? args[2] : NULL_VAL;
    return true;
}

static int compareNames(const void *a, const void *b) {
    ObjString *left = *(ObjString *const *)a;
    ObjString *right = *(ObjString *const *)b;
    uint32_t shortest = left->length < right->length ? left->length : right->length;
    int order = memcmp(left->chars, right->chars, shortest);
    if (order != 0) return order;
    return left->length < right->length ? -1
         : left->length > right->length ? 1 : 0;
}

/* Sorted, because a hash table's order is an implementation detail and a
 * program that prints `members()` should not change output between runs. */
static bool nModuleMembers(int argc, Value *args, Value *out) {
    (void)argc;
    ObjModule *m;
    if (!requireModuleArg(args[0], "members", &m)) return false;

    int slot = 0;
    Value key, value;
    int found = 0;
    while (jaiTableNext(&m->globals, &slot, &key, &value)) {
        if (IS_STRING(key) && moduleExposes(m, AS_STRING(key), &value)) found++;
    }

    ObjString **names = found > 0 ? JAI_ALLOC(ObjString *, found) : NULL;
    int written = 0;
    slot = 0;
    while (jaiTableNext(&m->globals, &slot, &key, &value) && written < found) {
        if (IS_STRING(key) && moduleExposes(m, AS_STRING(key), &value))
            names[written++] = AS_STRING(key);
    }
    if (written > 1) qsort(names, (size_t)written, sizeof *names, compareNames);

    ObjList *list = jaiListNew(written);
    jaiPushRoot(OBJ_VAL(list));
    for (int i = 0; i < written; i++) jaiListPush(list, OBJ_VAL(names[i]));
    jaiPopRoot();
    if (names != NULL) JAI_FREE_ARRAY(ObjString *, names, found);

    *out = OBJ_VAL(list);
    return true;
}

bool jaiModuleMethod(Value receiver, ObjString *name, Value *out) {
    if (!IS_MODULE(receiver) || name == NULL) return false;
    ObjModule *m = AS_MODULE(receiver);

    /* A member the module exports outranks the introspection helpers: a module
     * defining `get` means its own `get` everywhere it is named. */
    if (moduleExposes(m, name, out)) return true;

    const char *text = name->chars;

/* Arities count the receiver, which the VM passes as args[0]. */
#define MODULE_METHOD(label, fn, minArity, maxArity)                           \
    if (strcmp(text, (label)) == 0) {                                          \
        *out = jaiBindNative(receiver, (label), (fn), (minArity), (maxArity),  \
                             NULL);                                            \
        return true;                                                           \
    }

    MODULE_METHOD("get",     nModuleGet,     2, 3)
    MODULE_METHOD("has",     nModuleHas,     2, 2)
    MODULE_METHOD("members", nModuleMembers, 1, 1)
    MODULE_METHOD("name",    nModuleName,    1, 1)
    MODULE_METHOD("path",    nModulePath,    1, 1)

#undef MODULE_METHOD

    /* Present but private: this is E0802's wording (same as the VM and
     * `from ... import`), not "no such member", so it doesn't read as a typo. */
    Value hidden;
    if (jaiModuleGet(m, name, &hidden))
        return jaiThrow(vm.cImportError, "'%s' is not exported by module '%s'",
                        name->chars, m->name != NULL ? m->name->chars : "?");
    return false;
}
