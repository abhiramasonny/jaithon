/* module.c — the module search path, the importer, and the top-level driver.
 *
 * Everything that turns a file into a running program lives here: the search
 * rules of spec §8, the guarantee that a module body runs exactly once, the
 * .jaic cache handshake, and the entry-point protocol of §8.4.
 *
 * Two facts shape the code.
 *
 *   - A module is registered in vm.modules *before* its body runs, in the
 *     MOD_LOADING state. That is what makes a cycle detectable (E0801) instead
 *     of infinite, and it is why an importer caught in a cycle can observe a
 *     half-initialised module rather than a fresh empty one.
 *   - An import happens either before the machine is running (the prelude, the
 *     entry module) or from inside a live frame (OP_IMPORT). The first wants a
 *     diagnostic, the second a catchable ImportError. Every failure path here
 *     goes through one reporting helper that picks between them, so the E-code
 *     is the same either way.
 */

#include <stdlib.h>   /* getenv */

#include "runtime.h"
#include "boot/seed.h"
#include "methods.h"

#include "../codegen/codegen.h"
#include "../common/diag.h"
#include "../lang/lexer.h"
#include "../lang/parser.h"
#include "../native/native.h"
#include "../sema/check.h"
#include "../sema/resolve.h"
#include "../sema/semadump.h"
#include "../vm/serialize.h"

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

/* True while the self-hosted front end is itself being imported.
 *
 * The front end is a set of Jaithon modules, so compiling them with the
 * self-hosted front end would need the self-hosted front end: importing
 * `jaithon.compile` under --front=jai recurses without this. Inside the window
 * the C front end compiles whatever the compiler's own closure needs -- which
 * is precisely the job the seed takes over once C is gone. */
static bool          sLoadingFrontEnd;

JaiRunOptions jaiRunDefaults(void) {
    JaiRunOptions o;
    memset(&o, 0, sizeof o);
    o.entryPath  = NULL;
    o.codegen    = jaiCodegenDefaults();
    o.useCache   = true;
    o.writeCache = true;
    /* Still C by default. The cache-order bug that blocked this is fixed, and
     * the golden `modules` tests now pass warm under --front=jai, but two
     * failures remain when the default flips: the REPL golden
     * `bindings_persist` loses one line, and `jaithon test`. Both need a
     * working repro before the flip lands. */
    o.selfHosted = false;
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

/* Messages stay short on purpose: jaiThrow renders into 512 bytes, and the one
 * long payload an import failure has (the searched directory list) is elided
 * explicitly by the caller rather than truncated here. */
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

/* Two lists, searched after the importing file's own directory: `sUserDirs`
 * (JAITHON_PATH plus whatever jaiModulePathAdd was given) and `sLibDirs` (the
 * installed library). They hold C strings rather than ObjStrings so the path
 * survives being configured before the VM is up; vm.modulePath mirrors both
 * for the collector and for tools that want to show the path. */
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

    /* `make install` puts the binary in <prefix>/bin and the library in
     * <prefix>/share/jaithon/lib; a build tree has ./jaithon next to ./lib.
     * Both layouts are derived from wherever this executable actually is, so a
     * relocated tree keeps working. */
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

    addLibDir("/usr/local/share/jaithon/lib");
    addLibDir("/usr/local/share/jaithon");
    addLibDir("/opt/homebrew/share/jaithon/lib");
    addLibDir("/opt/homebrew/share/jaithon");

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

/* Path components come from source text, so they must not be able to name
 * anything outside a search directory. Spec §2.1 allows non-ASCII identifiers,
 * so the high half is let through and everything else — separators, `..`,
 * control bytes — is not. */
static bool isNameByte(char c) {
    unsigned char u = (unsigned char)c;
    if (u >= 0x80) return true;
    return (u >= 'a' && u <= 'z') || (u >= 'A' && u <= 'Z') ||
           (u >= '0' && u <= '9') || u == '_';
}

/* Split `dotted` into a leading-dot count and a '/'-separated relative path.
 * Reports E0804 and returns false on anything that is not
 * `'.'* ident ('.' ident)*`. */
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

/* <dir>/std/math.jai, then the package form <dir>/std/math/mod.jai. */
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

/* The same search, asked speculatively. `jaiResolveModulePath` reports a miss —
 * as a diagnostic before the VM starts and as a raised ImportError once it is
 * running — because every caller so far was an import that has to fail. The
 * type checker is not: it asks whether a module it can see the name of has a
 * readable source, and a "no" is an answer, not an error. Both channels are
 * therefore restored to what they were. */
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

/* Register a source buffer with the diagnostic engine, which takes ownership
 * of it. The copy is exact — an embedded NUL must not shorten it, or the
 * registry believes it owns more bytes than it does. That is what jaiMemdup is
 * for, and this predates it. */
static int registerSource(const char *path, const char *source, size_t length) {
    return jaiSourceAdd(path, jaiMemdup(source, length), length);
}

/* Lex, parse, resolve, check and — when `emit` — generate code for an already
 * registered source. Returns false when any diagnostic was an error; the bag is
 * flushed either way, so warnings from a successful compile are still shown. */
/* `lateGlobals` compiles the text as a *fragment* of a module that is already
 * running: a name that module's body defined at run time lives in no symbol
 * table here, so it resolves to a late-bound global instead of E0200. Only
 * `__prim__.eval/exec/compile` want that; loading a module's own source keeps
 * the strict rule. */
static bool frontEnd(ObjModule *module, int fileId, const CodegenOptions *opts,
                     bool emit, ObjFunction **outBody, bool lateGlobals) {
    if (outBody != NULL) *outBody = NULL;

    JaiSourceFile *file = jaiSourceGet(fileId);
    if (file == NULL || file->source == NULL) {
        (void)jaiDiagError(E0902_INTERNAL_ERROR, JAI_SPAN_NONE,
                           "source %d was never registered", fileId);
        (void)jaiDiagFlush(&gDiags, stderr);
        return false;
    }
    if (module != NULL) module->sourceFileId = fileId;

    AstContext ast;
    jaiAstContextInit(&ast);

    Lexer lex;
    AstNode *program = jaiParseSource(&ast, &lex, file->source, file->length,
                                      fileId);
    /* Every string the parser kept was copied into the AST arena, so the token
     * stream and its cooked literals are dead the moment parsing ends. */
    jaiLexerFree(&lex);

    Resolver resolver;
    Checker  checker;
    bool haveResolver = false;
    bool haveChecker = false;
    ObjFunction *body = NULL;

    if (program != NULL && !jaiDiagHasErrors(&gDiags)) {
        jaiResolverInit(&resolver, &ast);
        haveResolver = true;
        resolver.module = module;
        resolver.replMode = lateGlobals;

        if (jaiResolveProgram(&resolver, program) && !jaiDiagHasErrors(&gDiags)) {
            jaiCheckerInit(&checker, &resolver, &ast);
            haveChecker = true;
            checker.foldConstants = opts->optLevel > 0;

            if (jaiCheckProgram(&checker, program) && !jaiDiagHasErrors(&gDiags) &&
                emit) {
                body = jaiCompileProgram(program, module, opts);
                if (body != NULL && jaiDiagHasErrors(&gDiags)) body = NULL;
                if (body == NULL && !jaiDiagHasErrors(&gDiags)) {
                    (void)jaiDiagError(E0902_INTERNAL_ERROR, program->span,
                                       "code generation produced no module body");
                }
            }
        }
    }

    /* The teardown below frees arenas, which can hand memory back to the
     * collector; the body is the only thing here that must survive it. */
    if (body != NULL) jaiPushRoot(OBJ_VAL(body));
    if (haveChecker) jaiCheckerFree(&checker);
    if (haveResolver) jaiResolverFree(&resolver);
    jaiAstContextFree(&ast);
    if (body != NULL) jaiPopRoot();

    /* A phase that fails without saying why would leave the caller reporting
     * "module failed to load" and nothing else. */
    if (program == NULL && !jaiDiagHasErrors(&gDiags)) {
        (void)jaiDiagError(E0902_INTERNAL_ERROR, JAI_SPAN_NONE,
                           "the parser produced no tree for `%s` and reported "
                           "no error", file->path != NULL ? file->path : "?");
    }

    if (jaiDiagFlush(&gDiags, stderr)) return false;
    if (program == NULL) return false;

    if (outBody != NULL) *outBody = body;
    return !emit || body != NULL;
}

ObjFunction *jaiCompileSource(const char *source, size_t length,
                              const char *path, ObjModule *module,
                              const CodegenOptions *opts) {
    if (source == NULL) return NULL;

    CodegenOptions defaults = jaiCodegenDefaults();
    if (opts == NULL) opts = &defaults;

    int fileId = registerSource(path != NULL ? path : "<source>", source, length);

    /* A fragment is text compiled into a module that already exists: either one
     * that finished loading, or the very module whose body is on the frame
     * stack right now (`__prim__.exec` from `__main__`). A fresh module handed
     * over by the CLI is neither, and keeps strict name resolution. */
    bool fragment = module != NULL &&
                    (module->state == MOD_LOADED ||
                     (vm.frameCount > 0 &&
                      vm.frames[vm.frameCount - 1].module == module));

    ObjFunction *body = NULL;
    if (!frontEnd(module, fileId, opts, true, &body, fragment)) return NULL;
    return body;
}

/* ------------------------------------------------------------------ */
/* Cache handshake                                                      */
/* ------------------------------------------------------------------ */

/* `selfHosted` is who actually compiled this image, NOT whether --front=jai was
 * passed. Only the entry file goes through the self-hosted front end today;
 * every import is compiled by C (loadModuleBody). Deriving the flag from the
 * option instead of the producer stamped those C-compiled imports as
 * self-hosted, which is exactly the cross-contamination the flag exists to
 * prevent. */
/* `make reseed` sets this. The seed must not serve the compilation that
 * produces its own replacement: with the seed answering, the front end never
 * compiles its own closure from source, so nothing lands in __jaicache__ and
 * the next seed is built from whatever little remains. Measured, the seed went
 * 39 modules -> 10 in one generation that way, and would have reached zero. */
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
    return flags;
}

/* A .jaic starts with magic, u16 version and u16 flags (spec/BYTECODE.md §7),
 * and the loader treats the flags as informational. Without this probe a cache
 * written by a debug build would be handed to a --release run with its asserts
 * still in it; with it, that is an ordinary cache miss. */
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
    return version == JAIC_VERSION && stored == (uint16_t)flags;
}

/* Read `path`, then either deserialise its cache or compile it. The source is
 * registered before the cache is consulted because a cache-loaded chunk carries
 * byte offsets into it: without the registration a traceback through a cached
 * module would have no text to quote. */
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
        if (cached != NULL) return cached;
        /* Stale, corrupt, or from another compiler: recompile silently. */
    }

    /* Inside the bootstrap window the compiler's own closure has no compiler to
     * call, so it comes from the seed. A miss falls through to the front end:
     * a tree whose sources have moved past the seed recompiles rather than
     * running stale bytecode, because jaiDeserializeModule checks the source
     * hash. */
    if (sLoadingFrontEnd && module->name != NULL && !seedDisabled()) {
        const JaiSeedEntry *seeded = jaiSeedFind(module->name->chars);
        if (seeded != NULL) {
            ObjFunction *fromSeed = jaiDeserializeModule(seeded->image,
                                                         seeded->length,
                                                         module, hash);
            if (fromSeed != NULL) return fromSeed;
        }
    }

    ObjFunction *body = NULL;
    if (selfHosting()) {
        body = jaiSelfHostedCompileInto(file->source, length, path, module,
                                        hash, opts->codegen.optLevel);
        if (body == NULL) return NULL;
        /* The self-hosted front end is handed a path, not a module name, so it
         * names the module body from the file's stem: `b` where the C front end
         * uses the registered name `sub.b`. Only the importer knows the
         * qualified name, so it is applied here.
         *
         * This is not cosmetic. The name is serialised into the cache entry,
         * and a cached module whose name has lost its package cannot resolve
         * its own imports when loaded back -- a warm `--front=jai` run failed
         * where a cold one passed. `--bootstrap-verify` cannot see it either:
         * compare_functions checks arity, flags, frame size, code and
         * constants, but not the function's name. */
        if (module->name != NULL) {
            body->name = module->name;
            body->qualifiedName = module->name;
        }
    } else if (!frontEnd(module, fileId, &opts->codegen, true, &body, false) ||
               body == NULL) {
        return NULL;
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

/* Run a module body.
 *
 * jaiVMRunModule resets the interpreter stack, which is right for the entry
 * module and fatal for an import: OP_IMPORT runs inside a live frame whose
 * slots would be discarded. While the machine is running the body is therefore
 * invoked as an ordinary call, which is reentrant. Either way the closure
 * carries the module, so the body's frame gets the right global scope. */
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
        /* A tree without lib/std is still usable: the prelude only re-exports
         * names a program may never mention. Whatever went wrong is printed
         * here — including the list of directories searched — and then dropped,
         * because leaving an error in the bag or an exception pending would
         * abort the next compile for a reason it had nothing to do with. */
        jaiClearException();
        (void)jaiDiagFlush(&gDiags, stderr);
        fprintf(stderr, "jaithon: warning: could not load std.prelude; the "
                        "names of spec §9's std.core are unavailable\n");
        return false;
    }

    /* The prelude is a list of re-exports (spec §9), so its export table is the
     * exact set of names that belong in the implicit global scope. Going
     * through jaiDefineGlobal also registers each with the resolver, which is
     * what stops the front end reporting E0200 for them. */
    int slot = 0;
    Value key, unused;
    while (jaiTableNext(&prelude->exports, &slot, &key, &unused)) {
        if (!IS_STRING(key)) continue;
        ObjString *exported = AS_STRING(key);
        if (strchr(exported->chars, '.') != NULL) continue;   /* not a plain name */

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

/* The program's argument vector, first entry the script itself — the list spec
 * §8.4 hands to main(). libc cannot hand argv back portably, so the builtin
 * global `__argv__` is where `__prim__.os_argv` reads it from. */
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

/* lib/jaithon/compile is itself a Jaithon program, so reaching it means running the
 * VM: the C front end compiles `jaithon.compile`, and `jaithon.compile` then compiles
 * the user's file. Stage 0 is always C; there is nothing else to bootstrap
 * from. Only the entry module is handed over — whatever it imports is loaded by
 * the ordinary importer, which is the C front end. That is a real limit of
 * `--front=jai` and it is documented here rather than hidden, but it is not a
 * lie: the file the user named really was compiled by the self-hosted compiler.
 *
 * Every failure below names what failed. Nothing here falls back to the C front
 * end: a compiler that quietly substitutes a different compiler is worse than
 * one that refuses, because the output looks like a passing test. */

#define JAI_SELF_HOSTED_MODULE "jaithon.compile"
#define JAI_SELF_HOSTED_ENTRY  "compile_source"

static bool instanceField(Value v, const char *name, Value *out) {
    if (!IS_INSTANCE(v)) return false;
    ObjInstance *inst = AS_INSTANCE(v);
    int slot = jaiClassFieldSlot(inst->klass, jaiStringInternC(name));
    if (slot < 0 || slot >= (int)inst->fieldCount) return false;
    *out = inst->fields[slot];
    return true;
}

/* `Compiled.report()` renders the front end's own diagnostics. Printing them is
 * the whole point of the exercise: without them a rejected file says only that
 * the self-hosted compiler said no. */
static void printSelfHostedDiagnostics(Value compiled) {
    if (!IS_INSTANCE(compiled)) return;
    ObjInstance *inst = AS_INSTANCE(compiled);

    Value method;
    if (!jaiTableGetInterned(&inst->klass->methods, jaiStringInternC("report"),
                             &method)) {
        return;
    }

    jaiPushRoot(compiled);
    Value bound = OBJ_VAL(jaiBoundNew(compiled, method));
    jaiPushRoot(bound);
    Value text = NULL_VAL;
    bool ok = jaiCallValue(bound, 0, NULL, &text);
    jaiPopRoots(2);

    if (!ok) { jaiClearException(); return; }
    if (IS_STRING(text) && AS_STRING(text)->length > 0) {
        fprintf(stderr, "%s\n", AS_STRING(text)->chars);
    }
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

    /* Two live objects before the call, so both are rooted: the second
     * jaiStringInternC can collect the first.
     *
     * The trailing three are `compile_source`'s defaulted parameters
     * (release, fileId, optLevel). Only the last carries information, and it
     * has to: `--bootstrap-verify -O0` compiles the C side at the level it was
     * given, so a bridge that always took the default left the self-hosted
     * side at -O2 and turned the comparison into C-at-O0 against Jaithon-at-O2.
     * That is not a bug in either front end but it reads as one in every line
     * of the report, and it costs the only tool that can separate an emitter
     * divergence from an optimiser divergence.
     *
     * `fileId` carries information too. It was a hardcoded zero on the
     * reasoning that nothing read it; something does. Every span the
     * self-hosted front end emits records it, the line table is made of spans,
     * and a disassembler resolves a span back to a line through the source
     * registered under that id. With zero, a `--front=jai` build disassembled
     * with no line numbers and its tracebacks could not name a line —
     * invisible to `--bootstrap-verify`, which excludes the line table from
     * the comparison by design (spec §11). Every caller therefore has to have
     * registered its source and set `module->sourceFileId` first. */
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

    /* A plain `bytes` is the shape the driver would rather have and the shape
     * `compile_source` is expected to settle on; the `Compiled` record is what
     * lib/jaithon/compile returns today. Both are understood so that the module can
     * change without the driver going silently back to the C front end. */
    Value field;
    if (IS_BYTES(produced)) {
        image = AS_BYTES(produced);
    } else if (instanceField(produced, "image", &field) && IS_LIST(field)) {
        if (AS_LIST(field)->count == 0) {
            printSelfHostedDiagnostics(produced);
            (void)jaiDiagError(E0902_INTERNAL_ERROR, JAI_SPAN_NONE,
                               "%s: the self-hosted front end emitted no .jaic "
                               "image", path);
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

/* jaiDeserializeModule is written as a cache probe: it answers yes or no and
 * says nothing, because for a cache "no" only ever means "recompile". Here "no"
 * is the entire report, so the fixed header is re-read to name the field that
 * did not match. Field order follows spec/BYTECODE.md §7 and the reader in
 * serialize.c; anything past the header is the reader's business. */
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

/* The --front=jai counterpart of loadModuleBody: same source registration, same
 * contract (a body, or NULL with the reason in gDiags), but the bytecode
 * arrives as a .jaic image from lib/jaithon/compile.
 *
 * The cache is consulted and written, exactly as loadModuleBody does. It used
 * not to be, because a __jaicache__ entry recorded no front end and one
 * compiler reading the other's would be the mix-up this flag exists to expose.
 * JAIC_FLAG_SELFHOSTED now records the producer, so an entry written by the
 * other front end is an ordinary cache miss rather than a hazard.
 *
 * This is what makes the warm path measurable at all: a warm run deserialises a
 * .jaic and never reaches a front end, so it costs the same whichever compiler
 * filled the cache -- but only if the self-hosted path is allowed to fill it. */
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
        if (cached != NULL) return cached;
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

    /* Load the front end once, before any module body runs.
     *
     * Without this the first thing that needs a compile might be an `import`
     * executed from inside a module body that was itself served from cache --
     * and loading the compiler there means loading it inside a running body.
     * That is why a warm run failed where a cold one passed: compiling the
     * entry file cold pulls the front end in first, and a cache hit skips that.
     * Doing it up front makes the two orders identical. */
    if (sOptions.selfHosted) {
        bool wasLoading = sLoadingFrontEnd;
        sLoadingFrontEnd = true;
        (void)jaiImportModule(JAI_SELF_HOSTED_MODULE, NULL);
        sLoadingFrontEnd = wasLoading;
        jaiClearException();
    }

    ObjList *args = installArgv(absolute, argc, argv);
    if (!preludeDisabled()) (void)jaiLoadPrelude();

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
/* ------------------------------------------------------------------ */

/* `check` compiles one file and runs nothing, so the MOD_LOADING marker that
 * makes a cycle visible to the importer is never set and E0801 would only ever
 * appear once the program was run. Spec §8 wants the cycle reported by the
 * front end too, so the graph is walked here instead.
 *
 * Only the import statements of each reachable file are wanted, but they come
 * from a real parse: hand-scanning the token stream would be a second, quietly
 * divergent grammar for the one construct that decides whether the program can
 * load at all. The edges are cached per file for the life of the process while
 * the depth-first colouring is not, so `check a.jai b.jai` parses each module
 * once and still reports the cycle against both files.
 *
 * Imports nested inside a block are skipped: they run when the enclosing
 * function is called, not while the module body loads, which is exactly the
 * workaround the help line offers. */

typedef struct {
    char   *name;     /* dotted path as written, for the rendered chain */
    char   *path;     /* resolved absolute path, or NULL when unresolvable */
    JaiSpan span;
} ImportEdge;

enum { IG_WHITE = 0, IG_GREY, IG_BLACK };

typedef struct {
    char       *path;       /* absolute; the identity of the node */
    char       *name;       /* file stem, for the rendered chain */
    ImportEdge *edges;
    int         edgeCount;
    int         edgeCap;    /* what the array was allocated with */
    int         walk;       /* the walk `colour` was set by */
    uint8_t     colour;
} ImportGraphNode;

typedef JAI_VEC(ImportEdge) ImportEdgeVec;

static JAI_VEC(ImportGraphNode) sImportGraph;
static JAI_VEC(int)             sWalkStack;   /* node indices, outermost first */
static int                      sWalkId;

/* The name a file is known by inside an import chain: its basename without the
 * extension, which is the last component of the dotted path that names it. */
static void fileStem(char *out, size_t outSize, const char *path) {
    jaiPathBasename(out, outSize, path);
    size_t len = strlen(out);
    size_t ext = strlen(JAI_MODULE_EXT);
    if (len > ext && memcmp(out + len - ext, JAI_MODULE_EXT, ext) == 0) {
        out[len - ext] = '\0';
    }
}

/* The module name a *path* carries, which is `fileStem` plus the fallback an
 * import chain does not want: a path with nothing left after the extension is
 * stripped names the main module, not the empty module.
 *
 * Shared rather than static because three callers need the same answer and had
 * been spelling it out separately — the CLI, the self-hosted bridge, and
 * __prim__.compile_image. A fourth copy lives in `module_name_for`
 * (lib/jaithon/compile/mod.jai) and must agree with this one, because
 * --bootstrap-verify compiles the same file with both front ends and the name
 * is a constant in the record's pool. */
void jaiModuleNameFor(const char *path, char *out, size_t outSize) {
    fileStem(out, outSize, path);
    if (out[0] == '\0') snprintf(out, outSize, "__main__");
}

static void freeOwned(char *s) {
    if (s != NULL) (void)jaiRealloc(s, strlen(s) + 1, 0);
}

static int graphFind(const char *path) {
    for (int i = 0; i < sImportGraph.count; i++) {
        if (strcmp(sImportGraph.data[i].path, path) == 0) return i;
    }
    return -1;
}

static void collectImports(ImportEdgeVec *out, AstNode **stmts, int count,
                           const char *fromDir) {
    for (int i = 0; i < count; i++) {
        AstNode *n = stmts[i];
        if (n == NULL) continue;
        const char *dotted = NULL;
        if (n->kind == AST_IMPORT)           dotted = n->as.import.path;
        else if (n->kind == AST_FROM_IMPORT) dotted = n->as.fromImport.path;
        if (dotted == NULL) continue;

        ImportEdge edge;
        edge.name = jaiStrdup(displayName(dotted));
        edge.span = n->span;
        char resolved[JAI_MAX_PATH];
        edge.path = jaiResolveModulePath(dotted, fromDir, resolved, sizeof resolved)
                        ? jaiStrdup(resolved)
                        : NULL;
        JAI_VEC_PUSH(ImportEdge, out, edge);
    }
}

/* Parse `path` for its top-level imports and add the node. `fileId` is the
 * already-registered source when the caller has one, -1 to read the file.
 * Returns the node index, or -1 when the file cannot be read.
 *
 * Whatever the parse and the path search have to say belongs to the scanned
 * file's own compilation, not to the one being checked — a syntax error two
 * modules away must not be printed here, and a module that cannot be found is
 * the importer's E0800 at import time. The bag is swapped for a scratch one so
 * those findings can be dropped wholesale. */
static int graphAdd(const char *path, int fileId) {
    if (fileId < 0) {
        size_t length = 0;
        char *text = jaiReadFile(path, &length);
        if (text == NULL) return -1;
        fileId = jaiSourceAdd(path, text, length);   /* takes ownership */
    }
    JaiSourceFile *file = jaiSourceGet(fileId);
    if (file == NULL || file->source == NULL) return -1;

    char stem[JAI_MAX_PATH];
    fileStem(stem, sizeof stem, path);
    char dir[JAI_MAX_PATH];
    jaiPathDirname(dir, sizeof dir, path);

    ImportEdgeVec edges;
    JAI_VEC_INIT(&edges);

    JaiDiagBag live = gDiags;
    jaiDiagInit(&gDiags);

    AstContext ast;
    jaiAstContextInit(&ast);
    Lexer lex;
    AstNode *program = jaiParseSource(&ast, &lex, file->source, file->length,
                                      fileId);
    jaiLexerFree(&lex);
    /* A file that does not parse has no reliable import list; it will report
     * its own E01xx when something actually compiles it. */
    if (program != NULL && program->kind == AST_PROGRAM) {
        collectImports(&edges, program->as.block.stmts, program->as.block.count,
                       dir);
    }
    jaiAstContextFree(&ast);

    jaiDiagFree(&gDiags);
    gDiags = live;

    ImportGraphNode node;
    memset(&node, 0, sizeof node);
    node.path = jaiStrdup(path);
    node.name = jaiStrdup(stem[0] != '\0' ? stem : path);
    node.edges = edges.data;
    node.edgeCount = edges.count;
    node.edgeCap = edges.capacity;
    JAI_VEC_PUSH(ImportGraphNode, &sImportGraph, node);
    return sImportGraph.count - 1;
}

/* "a -> b -> a", starting at the first appearance of the module that is being
 * imported again, so the cycle itself is what the user reads. */
static void reportCheckCycle(int target, const ImportEdge *closing) {
    int start = 0;
    for (int i = 0; i < sWalkStack.count; i++) {
        if (sWalkStack.data[i] == target) { start = i; break; }
    }

    JaiBuf chain;
    jaiBufInit(&chain);
    for (int i = start; i < sWalkStack.count; i++) {
        jaiBufAppendStr(&chain, sImportGraph.data[sWalkStack.data[i]].name);
        jaiBufAppendStr(&chain, " -> ");
    }
    jaiBufAppendStr(&chain, sImportGraph.data[target].name);
    jaiBufPush(&chain, '\0');

    JaiDiag *d = jaiDiagError(E0801_CIRCULAR_IMPORT, closing->span,
                              "circular import: %s", (const char *)chain.data);
    jaiDiagAddLabel(d, closing->span, "this import closes the cycle");

    /* The edge that entered the cycle, so both ends of it are on screen. */
    if (start + 1 < sWalkStack.count) {
        const ImportGraphNode *first = &sImportGraph.data[sWalkStack.data[start]];
        for (int i = 0; i < first->edgeCount; i++) {
            if (first->edges[i].path != NULL &&
                strcmp(first->edges[i].path,
                       sImportGraph.data[sWalkStack.data[start + 1]].path) == 0) {
                jaiDiagAddLabel(d, first->edges[i].span, "the cycle starts here");
                break;
            }
        }
    }
    jaiDiagAddHelp(d, "move the shared declarations into a module every step of "
                      "the cycle can import, or move the import inside the "
                      "function that needs it");
    jaiBufFree(&chain);
}

/* Returns false once a cycle has been reported: one E0801 says everything the
 * rest of the walk would repeat. */
static bool graphVisit(int index) {
    ImportGraphNode *node = &sImportGraph.data[index];
    if (node->walk == sWalkId && node->colour == IG_BLACK) return true;
    node->walk = sWalkId;
    node->colour = IG_GREY;
    JAI_VEC_PUSH(int, &sWalkStack, index);

    /* The edge array is its own allocation, so it survives the reallocation of
     * sImportGraph that graphAdd below can trigger; the node struct does not. */
    ImportEdge *edges = node->edges;
    int edgeCount = node->edgeCount;

    bool ok = true;
    for (int i = 0; ok && i < edgeCount; i++) {
        if (edges[i].path == NULL) continue;   /* E0800 is the importer's to report */
        int next = graphFind(edges[i].path);
        if (next < 0) next = graphAdd(edges[i].path, -1);
        if (next < 0) continue;

        if (sImportGraph.data[next].walk == sWalkId &&
            sImportGraph.data[next].colour == IG_GREY) {
            reportCheckCycle(next, &edges[i]);
            ok = false;
            break;
        }
        ok = graphVisit(next);
    }

    sWalkStack.count--;
    sImportGraph.data[index].colour = IG_BLACK;
    return ok;
}

/* Reports E0801 into gDiags when the imports reachable from `path` close a
 * cycle. `fileId` is the entry file's already-registered source. */
static void checkImportCycles(const char *path, int fileId) {
    int root = graphFind(path);
    if (root < 0) root = graphAdd(path, fileId);
    if (root < 0) return;

    sWalkId++;
    (void)graphVisit(root);
}

void jaiImportGraphFree(void) {
    for (int i = 0; i < sImportGraph.count; i++) {
        ImportGraphNode *n = &sImportGraph.data[i];
        for (int j = 0; j < n->edgeCount; j++) {
            freeOwned(n->edges[j].name);
            freeOwned(n->edges[j].path);
        }
        JAI_FREE_ARRAY(ImportEdge, n->edges, n->edgeCap);
        freeOwned(n->path);
        freeOwned(n->name);
    }
    JAI_VEC_FREE(ImportGraphNode, &sImportGraph);
    JAI_VEC_FREE(int, &sWalkStack);
    sWalkId = 0;
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

    /* Before the front end, so that a cycle is the first thing reported: every
     * name a module in the cycle should have exported is missing, and the
     * E0200 storm that follows is a consequence, not a second problem. */
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
    jaiPushRoot(OBJ_VAL(module));

    /* The dump belongs to the file the user named. Imports run through the
     * same checker, so beginning here rather than inside the front end is what
     * stops each of them overwriting the entry file's decisions. */
    jaiSemaDumpBegin(entry);

    bool ok;
    if (sOptions.selfHosted) {
        /* `--front=jai check` runs the self-hosted front end over the entry
         * file and reports what it says, which is the only way to compare the
         * two checkers on a file rather than on a chunk of bytecode. It is a
         * weaker `check` than the C one until the self-hosted side has a
         * checker of its own: what it verifies today is that the file lexes,
         * parses, resolves and emits. That is stated rather than hidden,
         * because a `check` that passes without checking is worse than one
         * that refuses. */
        const JaiSourceFile *entryFile = jaiSourceGet(fileId);
        ObjFunction *body = entryFile == NULL
                              ? NULL
                              : jaiSelfHostedCompileInto(entryFile->source,
                                                         entryFile->length,
                                                         absolute, module,
                                                         jaiSourceHash(entryFile->source,
                                                                       entryFile->length),
                                                         sOptions.codegen.optLevel);
        ok = body != NULL && !jaiDiagHasErrors(&gDiags);
        (void)jaiDiagFlush(&gDiags, stderr);
    } else {
        /* Emit as well: `check` promises "compile and type-check" (spec §8.5),
         * and codegen is where the bytecode verifier runs. Checking without it
         * would pass files that cannot actually be loaded. The body is
         * discarded. */
        ok = frontEnd(module, fileId, &sOptions.codegen, true, NULL, false);
    }
    jaiPopRoots(3);

    if (!jaiSemaDumpEnd()) {
        (void)jaiDiagError(JAI_OK, JAI_SPAN_NONE,
                           "cannot write the sema dump for `%s`", entry);
        (void)jaiDiagFlush(&gDiags, stderr);
        return 1;
    }

    if (sOptions.verbose && ok) {
        fprintf(stderr, "jaithon: %s is clean\n", entry);
    }
    return ok ? 0 : 1;
}

/* ------------------------------------------------------------------ */
/* Module members                                                       */
/*                                                                      */
/* `mod.thing` is resolved by the VM directly, so what reaches here is   */
/* the reflective path: dir(), and any C caller that asks the built-in   */
/* method tables about a module receiver. Both need the same answer the  */
/* VM would give, which is why the export rule is repeated rather than   */
/* approximated: a module with no explicit exports exposes everything,   */
/* because restricting an un-annotated module would make it unusable     */
/* rather than encapsulated (spec §8.2).                                 */
/* ------------------------------------------------------------------ */

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

    /* Present but private. Saying "no such member" would send the reader
     * looking for a typo, so this is E0802's wording, the same one the VM and
     * `from ... import` raise. */
    Value hidden;
    if (jaiModuleGet(m, name, &hidden))
        return jaiThrow(vm.cImportError, "'%s' is not exported by module '%s'",
                        name->chars, m->name != NULL ? m->name->chars : "?");
    return false;
}
