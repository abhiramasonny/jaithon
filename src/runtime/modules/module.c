/* module.c — module loading and the self-hosting bootstrap window.
 *
 * Everything that turns a file into a running program lives here: the
 * guarantee that a module body runs exactly once, the .jaic cache handshake,
 * the entry-point protocol of §8.4, and the bootstrap window
 * (sLoadingFrontEnd) that lets the self-hosted front end compile itself.
 *
 * Three pieces this file used to hold moved out to siblings, because none of
 * them share the state described below:
 *
 *   - module_path.c: the search path (JAITHON_PATH, jaiModulePathAdd) and
 *     turning a dotted module name into a file on disk
 *     (jaiResolveModulePath, spec §8). Pure directory lookups and dotted-name
 *     syntax; it never reads sOptions, sLoadingFrontEnd, or the import stack.
 *   - module_cache.c: computing and probing .jaic header flags
 *     (cacheFlagsFor/cacheFlagsMatch), the JAITHON_TRACE_LOAD/JAITHON_NO_SEED
 *     env gates, and decoding a rejected .jaic header into a reason
 *     (jaicRejectionReason). Pure functions of their arguments, or of an env
 *     var read fresh each call.
 *   - module_methods.c: the native methods on module objects themselves
 *     (mod.get/.has/.members/.name/.path) — behaviour of an already-loaded
 *     module, not part of loading one, and sharing none of this file's
 *     private state.
 *
 * A handful of small helpers (importFailure, ensurePathReady, isRegularFile,
 * storeResolved, displayName, and the module_cache.c functions above) cross
 * those file boundaries in one direction or the other; module_internal.h is
 * where each is declared, private to this directory.
 *
 * What stayed is one cohesive state machine and is not split further. Two
 * facts shape it.
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
 *
 * The bootstrap window adds a third fact, and it is why warmFrontEnd,
 * maybeWarmFor, loadModuleBody, jaiImportModule and the self-hosted-compile
 * functions below stay together rather than being teased apart further:
 * sLoadingFrontEnd and sFrontEndWarmed are read and set across all of them,
 * and getting the order wrong is a two-sided deadlock hazard — see
 * warmFrontEnd's and maybeWarmFor's own comments for two ways that has
 * actually gone wrong (a cycle reported against std.math; two generations of
 * the compiler loaded into one process). Splitting this dispatch apart would
 * not make it easier to follow; it would just move the shared state into a
 * header where the ordering constraints are no longer visible in one place.
 */

#include <stdlib.h>

#include "runtime/runtime.h"
#include "boot/seed.h"
#include "runtime/modules/frontend.h"
#include "vm/jit/jit.h"
#include "vm/gc.h"
#include "runtime/modules/module_internal.h"

#include "common/diag.h"
#include "vm/bytecode/serialize.h"

CodegenOptions jaiCodegenDefaults(void) {
    CodegenOptions opts;
    opts.optLevel = 2;
    opts.debugInfo = true;
    opts.stripAsserts = false;
    opts.emitTailCalls = true;
    return opts;
}

/* Enough to hold a legitimate package chain; a longer one is a runaway import
 * that would otherwise recurse the C stack (each level nests a VM run). */
#define JAI_MAX_IMPORT_DEPTH 64

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
void jaiSnapshotAudit(const char *when);

/* Front-end load timing, JAI_FRONTEND_TIME only. */
static double gFeDeser;
static int    gFeModules;
static bool   gFeTiming;

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
    /* The self-hosted front end is the default. `--front=c` stays while the C
     * front end exists, so a regression is bisectable rather than only
     * observable. */
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

/* Messages stay short on purpose: jaiThrow renders into 512 bytes, and the one
 * long payload an import failure has (the searched directory list) is elided
 * explicitly by the caller rather than truncated here. */
bool importFailure(JaiDiagCode code, ObjClass *klass, const char *fmt, ...) {
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
/* Front end                                                            */
/* ------------------------------------------------------------------ */

/* Register a source buffer with the diagnostic engine, which takes ownership
 * of it. The copy is exact — an embedded NUL must not shorten it, or the
 * registry believes it owns more bytes than it does. That is what jaiMemdup is
 * for, and this predates it. */
static int registerSource(const char *path, const char *source, size_t length) {
    return jaiSourceAdd(path, jaiMemdup(source, length), length);
}

/* Compile a source string into a module body.
 *
 * `__prim__.eval`, `exec` and `compile` reach this, and so does the CLI. It ran
 * the C front end until that front end was deleted; it now asks the self-hosted
 * one, which is the compiler every other path already used.
 *
 * A *fragment* is text compiled into a module that already exists -- one that
 * finished loading, or the very module whose body is on the frame stack right
 * now. A name that module's body defined at run time lives in no symbol table
 * this compilation can see, so a fragment defers it rather than reporting
 * E0200. A fresh module handed over by the CLI is neither, and keeps the strict
 * rule. */
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

/* `selfHosted` is who actually compiled this image, NOT whether --front=jai was
 * passed. Only the entry file goes through the self-hosted front end today;
 * every import is compiled by C (loadModuleBody). Deriving the flag from the
 * option instead of the producer stamped those C-compiled imports as
 * self-hosted, which is exactly the cross-contamination the flag exists to
 * prevent. */
/* Whether THIS compilation goes through the self-hosted front end. */
static bool selfHosting(void) {
    return sOptions.selfHosted && !sLoadingFrontEnd;
}

/* Read `path`, then either deserialise its cache or compile it. The source is
 * registered before the cache is consulted because a cache-loaded chunk carries
 * byte offsets into it: without the registration a traceback through a cached
 * module would have no text to quote. */
#define JAI_SELF_HOSTED_MODULE "jaithon.compile"

/* Pull the whole front end in, once, before any compile begins.
 *
 * This used to run at the top of every module load, so `jaithon run` on a
 * fully cached program still deserialised 35 compiler modules out of the seed
 * in order to compile nothing: 12ms of the 18ms an empty program cost.
 *
 * Warming it at the first compile instead is wrong in a way that only shows up
 * when the entry file is cached and something it imports is not. The compiler
 * imports std.math, std.str and std.json, so warming while one of those is
 * itself mid-load finds it MOD_LOADING and reports a cycle -- the front end
 * then fails, the failure is swallowed, and the import that triggered it dies
 * with "module 'std.math' failed to load". A cold run passes, because there
 * the entry misses first and the warm happens with nothing on the import
 * stack.
 *
 * So the warm happens at the first module that is going to need compiling,
 * BEFORE that module is published as MOD_LOADING -- see maybeWarmFor. Then
 * the compiler's own imports find std.math untouched and load it normally. */
/* Loading the front end builds ~10MB of objects that live for the rest of
 * the process, and it used to trigger one collection partway through: a full
 * mark of everything built so far that freed ~53KB, ~5% of an edit-then-run.
 * So collection is paused while the front end loads, and the bytes built under
 * the pause are then credited to the collector as permanent (jaiGCCredit):
 * they never count toward the next collection, and the budget is at least what
 * a collection at the end would have given them. Measured together with the
 * intern sizing below, as an env A/B in one binary (cycles.py -n 7): -3.9%
 * cycles on a one-line `check` (floor 2.1%), -1.2% on edit-then-run (floor
 * 0.4%). --gc-stress keeps collecting, since finding what a collection breaks
 * is its whole job.
 *
 * The credit is the bytes allocated DURING the pause, never the heap as it
 * stands. Every eval, REPL line, test case and `check` file comes through here
 * again with the front end already loaded; when the credit was "4x the whole
 * heap", garbage included, each of those calls pushed the next collection
 * further out and the collector stopped (check lib: 0 collections, 3.3GB). A
 * call that finds the front end loaded now allocates ~nothing under the pause
 * and credits ~nothing, and a load that fails credits nothing at all, since
 * what it built is garbage.
 *
 * The intern table is sized for the front end's ~5,000 strings up front too:
 * it otherwise doubles its way from 8 slots to 16K, rehashing every string
 * interned so far at each step.
 *
 * JAITHON_FRONTEND_LOAD_TUNE=0 does neither. */
#define JAI_FRONTEND_INTERNED 6000

static bool frontEndTuneOn(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *s = getenv("JAITHON_FRONTEND_LOAD_TUNE");
        cached = (s != NULL && strcmp(s, "0") == 0) ? 0 : 1;
    }
    return cached != 0;
}

typedef struct {
    bool paused;
    size_t heapBefore;
} FrontEndPause;

static FrontEndPause frontEndLoadBegin(void) {
    FrontEndPause p = {false, 0};
    if (!frontEndTuneOn()) return p;
    jaiTableReserve(jaiInternTable(), JAI_FRONTEND_INTERNED);
    if (vm.gcStress || vm.gc == NULL || !vm.gc->enabled) return p;
    jaiGCEnable(false);
    p.paused = true;
    p.heapBefore = jaiHeapBytes;
    return p;
}

static void frontEndLoadEnd(FrontEndPause p, bool loaded) {
    if (!p.paused) return;
    jaiGCEnable(true);
    size_t now = jaiHeapBytes;
    if (loaded && now > p.heapBefore) jaiGCCredit(now - p.heapBefore);
}

static void warmFrontEnd(void) {
    if (!sOptions.selfHosted || sLoadingFrontEnd || sFrontEndWarmed) return;
    sFrontEndWarmed = true;
    sLoadingFrontEnd = true;
    FrontEndPause pause = frontEndLoadBegin();
    ObjModule *loaded = jaiImportModule(JAI_SELF_HOSTED_MODULE, NULL);
    frontEndLoadEnd(pause, loaded != NULL);
    sLoadingFrontEnd = false;
    jaiClearException();
    /* The candidate snapshot point: the front end is built and no user code has
     * run. Reports only, and only under JAITHON_SNAPSHOT_AUDIT. See
     * src/vm/snapshot.c and docs/research/PLAN-startup-snapshot.md. */
    jaiSnapshotAudit("warmFrontEnd");
}

/* A library module the seed carries -- std.math, std.str, std.json and the
 * rest of the compiler's own imports -- is served from the seed when its cache
 * misses, as long as the seed's image is exactly what a cache entry would
 * have to be: compiled from this source (the hash its header records) with
 * these flags. It is then a cache hit from a different store, and the
 * compiler does not have to be built to produce it.
 *
 * Those modules never had cache files: the warm below loaded them from the
 * seed inside the bootstrap window, so the ordinary door that compiles and
 * caches a module was never reached for them. Every run of a program that
 * imported std.math therefore built the whole front end -- 98 modules, 109.6M
 * instructions for `import std.math; print(math.PI)` against 18.6M for a hello
 * world -- to hand it std.math from the seed at the end. So did every run of
 * every program on jaicv, jaitensor, jainum or jaiframe.
 *
 * Front-end modules (lib/jaithon) are excluded: they must arrive as one set
 * with the compiler, see maybeWarmFor. When the seed does not match, the warm
 * happens exactly as before. `--no-cache` is left alone, and so is the entry
 * file, which is __main__ and not the module the seed compiled.
 * JAITHON_SEED_SERVES_LIBRARY=0 restores the warm.
 *
 * Serving one must not start a warm part-way through it -- the cycle
 * warmFrontEnd's comment describes, since the compiler imports these very
 * modules. Every import of a seeded module is seeded (`make
 * seed-closure-check`) and no lib/std module imports lib/jaithon, but an
 * import goes where the search path sends it, not to the seed: with std.math
 * edited and not yet reseeded, or std.str overridden on JAITHON_PATH, std.json
 * still matched, was served, and its own import of the other one needed the
 * compiler while std.json was half loaded. So the library is served as a
 * unit or not at all: see seedLibraryIntact. */
static bool seedServesLibrary(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_SEED_SERVES_LIBRARY");
        cached = (v != NULL && strcmp(v, "0") == 0) ? 0 : 1;
    }
    return cached == 1;
}

/* Whether a seed image whose header is `head` (at least 32 bytes) is what
 * compiling a source hashing to `hash` with this run's flags would produce. */
static bool seedHeaderMatches(const uint8_t *head, size_t length, uint64_t hash) {
    uint64_t recorded = 0;
    if (!jaicRecordedHash(head, length, &recorded) || recorded != hash)
        return false;
    uint32_t flags = cacheFlagsFor(&options()->codegen, true);
    return jaiCacheFlagsMatchBuffer(head, length, flags);
}

static bool isFrontEndKey(const char *key) {
    return strncmp(key, "jaithon/", 8) == 0;
}

/* The seed entry jaiSeedFind would answer `path` with, by index and without
 * inflating it: the first key that is a trailing component run of the path.
 * -1 when there is none. */
static long seedIndexFor(const char *path) {
    size_t pathLen = strlen(path);
    for (size_t i = 0; i < jaiSeedCount(); i++) {
        const char *key = jaiSeedModuleAt(i);
        size_t keyLen = key != NULL ? strlen(key) : 0;
        if (keyLen == 0 || keyLen > pathLen) continue;
        const char *tail = path + (pathLen - keyLen);
        if (strcmp(tail, key) != 0) continue;
        if (tail != path && tail[-1] != '/') continue;
        return (long)i;
    }
    return -1;
}

/* The dotted name a seed key is imported by: "std/math.jai" is std.math and
 * "std/x/mod.jai" is std.x. */
static bool seedKeyDottedName(const char *key, char *out, size_t size) {
    size_t n = strlen(key);
    size_t ext = strlen(JAI_MODULE_EXT);
    size_t pkg = strlen("/" JAI_PACKAGE_FILE);
    if (n > pkg && strcmp(key + n - pkg, "/" JAI_PACKAGE_FILE) == 0) n -= pkg;
    else if (n > ext && strcmp(key + n - ext, JAI_MODULE_EXT) == 0) n -= ext;
    else return false;
    if (n + 1 > size) return false;
    for (size_t i = 0; i < n; i++) out[i] = key[i] == '/' ? '.' : key[i];
    out[n] = '\0';
    return true;
}

/* Paths and directories already judged, for the life of the process. */
typedef struct {
    char *key;
    bool  ok;
} SeedVerdict;

static JAI_VEC(SeedVerdict) sSeedFileVerdicts;
static JAI_VEC(SeedVerdict) sSeedDirVerdicts;

static const SeedVerdict *verdictFind(const SeedVerdict *v, int count,
                                      const char *key) {
    for (int i = 0; i < count; i++)
        if (strcmp(v[i].key, key) == 0) return &v[i];
    return NULL;
}

static void verdictNote(int which, const char *key, bool ok) {
    SeedVerdict v = { jaiStrdup(key), ok };
    if (which == 0) JAI_VEC_PUSH(SeedVerdict, &sSeedFileVerdicts, v);
    else JAI_VEC_PUSH(SeedVerdict, &sSeedDirVerdicts, v);
}

/* Whether the file at `path` is, byte for byte, the source of seed entry
 * `index`, and the entry the seed would serve it from. Reads the image's
 * header only: a program that imports std.math alone should not inflate
 * std.json to learn that it is current. */
static bool seedFileIntact(const char *path, size_t index) {
    if (seedIndexFor(path) != (long)index) return false;
    const SeedVerdict *known = verdictFind(sSeedFileVerdicts.data,
                                           sSeedFileVerdicts.count, path);
    if (known != NULL) return known->ok;
    bool ok = false;
    uint8_t head[32];
    if (jaiSeedPeekAt(index, head, sizeof head) == sizeof head) {
        size_t length = 0;
        char *text = jaiReadFile(path, &length);
        if (text != NULL) {
            ok = seedHeaderMatches(head, sizeof head, jaiSourceHash(text, length));
            JAI_FREE_ARRAY(char, text, length + 1);
        }
    }
    verdictNote(0, path, ok);
    return ok;
}

/* Whether every seeded library module -- each seed entry outside
 * lib/jaithon -- is, imported by its dotted name from `dir`, a file the seed
 * carries unchanged, and the same holds from each directory those files are
 * in. Then a module served from the seed imports only modules that are served
 * from the seed too (or from a cache made of the same source), and nothing it
 * pulls in can need the compiler. One stale or shadowed member turns the
 * shortcut off for all of them, and the warm happens before the first one is
 * published, as it always did.
 *
 * Checked once per directory, and only once some program reaches a seeded
 * module that its cache does not have: three reads and three resolutions,
 * which the imports that follow find already memoised. */
static bool seedLibraryIntact(const char *dir) {
    const SeedVerdict *known = verdictFind(sSeedDirVerdicts.data,
                                           sSeedDirVerdicts.count, dir);
    if (known != NULL) return known->ok;

    JAI_VEC(char *) pending;
    JAI_VEC_INIT(&pending);
    JAI_VEC(char *) seen;
    JAI_VEC_INIT(&seen);
    JAI_VEC_PUSH(char *, &pending, jaiStrdup(dir));
    JAI_VEC_PUSH(char *, &seen, jaiStrdup(dir));

    bool ok = jaiSeedCount() > 0;
    while (ok && pending.count > 0) {
        char *from = JAI_VEC_POP(&pending);
        for (size_t i = 0; ok && i < jaiSeedCount(); i++) {
            const char *key = jaiSeedModuleAt(i);
            if (key == NULL || isFrontEndKey(key)) continue;
            char name[JAI_MAX_PATH];
            char path[JAI_MAX_PATH];
            if (!seedKeyDottedName(key, name, sizeof name) ||
                !jaiResolveModulePathQuiet(name, from, path, sizeof path) ||
                !seedFileIntact(path, i)) {
                ok = false;
                break;
            }
            char next[JAI_MAX_PATH];
            jaiPathDirname(next, sizeof next, path);
            bool visited = false;
            for (int j = 0; j < seen.count && !visited; j++)
                visited = strcmp(seen.data[j], next) == 0;
            if (!visited) {
                JAI_VEC_PUSH(char *, &pending, jaiStrdup(next));
                JAI_VEC_PUSH(char *, &seen, jaiStrdup(next));
            }
        }
        JAI_FREE_ARRAY(char, from, strlen(from) + 1);
    }

    /* Every directory reached answers for a subset of what `dir` answers
     * for, so a success is theirs too; a failure is only `dir`'s. */
    if (ok) {
        for (int j = 0; j < seen.count; j++)
            if (verdictFind(sSeedDirVerdicts.data, sSeedDirVerdicts.count,
                            seen.data[j]) == NULL)
                verdictNote(1, seen.data[j], true);
    } else {
        verdictNote(1, dir, false);
    }
    while (pending.count > 0) {
        char *left = JAI_VEC_POP(&pending);
        JAI_FREE_ARRAY(char, left, strlen(left) + 1);
    }
    for (int j = 0; j < seen.count; j++)
        JAI_FREE_ARRAY(char, seen.data[j], strlen(seen.data[j]) + 1);
    JAI_VEC_FREE(char *, &pending);
    JAI_VEC_FREE(char *, &seen);
    return ok;
}

/* The seed's image for `path` when it may stand in for a cache entry of a
 * source hashing to `hash`, else NULL. */
static const JaiSeedEntry *seedStandsInFor(const char *path, uint64_t hash) {
    const JaiRunOptions *opts = options();
    if (!opts->useCache || !seedServesLibrary() || seedDisabled()) return NULL;
    const JaiSeedEntry *seeded = jaiSeedFind(path);
    if (seeded == NULL || isFrontEndKey(seeded->module)) return NULL;
    if (!seedHeaderMatches(seeded->image, seeded->length, hash)) return NULL;
    /* Judged already: the closure below need not read this file again. */
    if (verdictFind(sSeedFileVerdicts.data, sSeedFileVerdicts.count, path) == NULL)
        verdictNote(0, path, true);
    char dir[JAI_MAX_PATH];
    jaiPathDirname(dir, sizeof dir, path);
    if (!seedLibraryIntact(dir)) return NULL;
    return seeded;
}

/* seedStandsInFor, for a caller that has not read the source yet. */
static bool seedStandsInForFile(const char *path) {
    if (!seedServesLibrary()) return false;
    size_t length = 0;
    char *text = jaiReadFile(path, &length);
    if (text == NULL) return false;
    uint64_t hash = jaiSourceHash(text, length);
    JAI_FREE_ARRAY(char, text, length + 1);
    return seedStandsInFor(path, hash) != NULL;
}

/* maybeWarmFor has to know whether a module's cache is loadable before the
 * module exists, and loadModuleBody then reads the same file in full. On this
 * machine an open() is ~150K instructions -- more than reading a 14KB image --
 * so the probe reads the whole image once, keeps it here, and the next load
 * of the same path takes it instead of opening the file again. It is one
 * slot, consumed or dropped by the very next load; a warm in between cannot
 * leave it behind for a later one, because every load clears it. 250 cached
 * modules into a package import, that is 250 opens fewer.
 * JAITHON_CACHE_PREFETCH=0 goes back to the 8-byte probe. */
static struct {
    char    *path;
    uint8_t *data;
    size_t   length;
} sPrefetch;

static bool cachePrefetchOn(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_CACHE_PREFETCH");
        cached = (v != NULL && strcmp(v, "0") == 0) ? 0 : 1;
    }
    return cached == 1;
}

static void prefetchDrop(void) {
    if (sPrefetch.data != NULL) jaiCacheReadFree(sPrefetch.data, sPrefetch.length);
    if (sPrefetch.path != NULL)
        JAI_FREE_ARRAY(char, sPrefetch.path, strlen(sPrefetch.path) + 1);
    sPrefetch.path = NULL;
    sPrefetch.data = NULL;
    sPrefetch.length = 0;
}

/* Whether `path`'s cache carries `flags`, reading it whole for the load that
 * follows. */
static bool prefetchedFlagsMatch(const char *path, uint32_t flags) {
    if (!cachePrefetchOn()) return cacheFlagsMatch(path, flags);
    prefetchDrop();
    size_t length = 0;
    uint8_t *data = jaiCacheRead(path, &length);
    if (data == NULL) return false;
    sPrefetch.path = jaiStrdup(path);
    sPrefetch.data = data;
    sPrefetch.length = length;
    return jaiCacheFlagsMatchBuffer(data, length, flags);
}

/* `path`'s cache image: the prefetched one when it is for this path, else
 * read now. Either way the slot is empty afterwards. */
static uint8_t *cacheReadForLoad(const char *path, size_t *length) {
    if (sPrefetch.data != NULL && strcmp(sPrefetch.path, path) == 0) {
        uint8_t *data = sPrefetch.data;
        *length = sPrefetch.length;
        sPrefetch.data = NULL;
        prefetchDrop();
        return data;
    }
    prefetchDrop();
    return jaiCacheRead(path, length);
}

/* Warm the front end if loading `path` is about to need it.
 *
 * Two reasons to warm, and both are necessary.
 *
 * A cache miss means this module has to be compiled, so the compiler has to be
 * here. The probe reads the cache entry's 8-byte header, not the module: a hit
 * only means the flags are loadable, and the source hash can still reject it
 * further in. That asymmetry is the safe direction -- a stale cache warms the
 * compiler slightly early, where a missed warm is the cycle above.
 *
 * A module under lib/jaithon is one of the front end's own, and those must
 * never be loaded through the ordinary door first. `jaithon fmt` imports
 * jaithon.ast directly, so without this it got ast.jai from the cache and then
 * the warm got compile/mod.jai from the seed -- two generations of the
 * compiler in one process, which failed importing std.json. Warming here makes
 * the whole front end arrive as one set, from one source, and the caller's
 * re-check then finds the module already loaded.
 *
 * The test is the directory, not seed membership: the seed also carries
 * std.math, std.str and std.json, which the compiler imports but user code
 * owns just as much. Warming for those put the 12ms straight back, because
 * std.core is among them and every program loads it.
 *
 * `entry` is the file `run` was given. It becomes __main__, which the seed
 * never compiled, so the seed cannot stand in for it: it is loaded by
 * selfHostedModuleBody, which compiles. */
static bool maybeWarmFor(const char *path, bool entry) {
    if (!sOptions.selfHosted || sLoadingFrontEnd || sFrontEndWarmed) return false;

    /* The seed keys on the library-relative path ("jaithon/ast.jai"), which is
     * the only reliable way to ask this question. Matching "/jaithon/" against
     * the absolute path instead matched every file in the tree, because the
     * repository directory is itself called jaithon -- so everything warmed
     * and the 12ms came straight back. */
    const JaiSeedEntry *seeded = seedDisabled() ? NULL : jaiSeedFind(path);
    bool ownedByFrontEnd = seeded != NULL &&
                           strncmp(seeded->module, "jaithon/", 8) == 0;
    if (!ownedByFrontEnd) {
        const JaiRunOptions *opts = options();
        uint32_t flags = cacheFlagsFor(&opts->codegen, true);
        if (opts->useCache && prefetchedFlagsMatch(path, flags)) return false;
        if (!entry && seeded != NULL && seedStandsInForFile(path)) return false;
    }
    warmFrontEnd();
    return true;
}

/* Whether a seeded module's source is read only when something needs it.
 *
 * In the bootstrap window a module comes from the seed, and the seed does not
 * look at the source beside it at all (serialize_read.c: "The seed is allowed
 * to be out of date with the source beside it"). The text was nonetheless read
 * and FNV-hashed up front for every one of the 98 modules -- 1.1MB, ~3-4ms of
 * every `check` and edit-then-run -- because the hash is what a __jaicache__
 * entry is validated against, and the cache is probed first. In practice the
 * front end's modules have no cache entries, so the hash went unused.
 *
 * Now the source is registered lazily (jaiSourceAddLazy) and hashed only if a
 * cache entry turns up; a diagnostic or traceback that points into the file
 * reads it then. Readability is still checked up front, so a file that cannot
 * be read fails the import exactly as before. JAITHON_DEFER_SEED_SOURCE=0
 * restores the eager read. */
static bool deferSeedSourceOn(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *s = getenv("JAITHON_DEFER_SEED_SOURCE");
        cached = (s != NULL && strcmp(s, "0") == 0) ? 0 : 1;
    }
    return cached != 0;
}

static ObjFunction *loadModuleBody(ObjModule *module, const char *path) {
    const JaiSeedEntry *seeded =
        (sLoadingFrontEnd && !seedDisabled()) ? jaiSeedFind(path) : NULL;

    uint64_t hash = 0;
    bool haveHash = false;
    int fileId;
    if (seeded != NULL && deferSeedSourceOn()) {
        if (!jaiPathReadable(path)) {
            (void)importFailure(E0800_MODULE_NOT_FOUND, vm.cIOError,
                                "cannot read module file '%s'", path);
            return NULL;
        }
        fileId = jaiSourceAddLazy(path, 0);
    } else {
        size_t length = 0;
        char *text = jaiReadFile(path, &length);
        if (text == NULL) {
            (void)importFailure(E0800_MODULE_NOT_FOUND, vm.cIOError,
                                "cannot read module file '%s'", path);
            return NULL;
        }
        hash = jaiSourceHash(text, length);
        haveHash = true;
        fileId = jaiSourceAdd(path, text, length);   /* takes ownership of text */
    }
    module->sourceFileId = fileId;

    if (fileId < 0) {
        (void)importFailure(E0902_INTERNAL_ERROR, vm.cImportError,
                            "cannot register source for '%s'", path);
        return NULL;
    }

    const JaiRunOptions *opts = options();
    uint32_t flags = cacheFlagsFor(&opts->codegen, selfHosting());

    /* One read, not two: the flags header and the image come from the same
     * buffer. The separate 8-byte probe cost 18.49 us per module, more than
     * reading the whole 116 KB image. */
    if (opts->useCache) {
        size_t cacheLen = 0;
        uint8_t *cacheData = cacheReadForLoad(path, &cacheLen);
        if (cacheData != NULL) {
            ObjFunction *cached = NULL;
            if (jaiCacheFlagsMatchBuffer(cacheData, cacheLen, flags)) {
                if (!haveHash) {
                    const JaiSourceFile *src = jaiSourceGet(fileId);
                    hash = jaiSourceHash(src->source, src->length);
                    haveHash = true;
                }
                cached = jaiDeserializeCached(cacheData, cacheLen, module, hash,
                                              path);
            }
            jaiCacheReadFree(cacheData, cacheLen);
            if (cached != NULL) {
                if (traceLoads()) fprintf(stderr, "load cache   %s\n", path);
                return cached;
            }
            /* Stale, corrupt, or from another compiler: recompile silently. */
        }
    }

    /* A seeded library module, outside the window: see seedServesLibrary. */
    if (!sLoadingFrontEnd && sOptions.selfHosted) {
        const JaiSeedEntry *seeded = seedStandsInFor(path, hash);
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

    /* Inside the bootstrap window the compiler's own closure has no compiler to
     * call, so it comes from the seed. A miss falls through to the front end:
     * a tree whose sources have moved past the seed recompiles rather than
     * running stale bytecode, because jaiDeserializeModule checks the source
     * hash. */
    if (sLoadingFrontEnd && !seedDisabled()) {
        if (seeded != NULL) {
            double dt0 = gFeTiming ? jaiClockMonotonic() : 0.0;
            ObjFunction *fromSeed = jaiDeserializeSeed(seeded->image,
                                                       seeded->length,
                                                       module, hash);
            if (gFeTiming) { gFeDeser += jaiClockMonotonic() - dt0; gFeModules++; }
            if (fromSeed != NULL) {
                if (traceLoads()) fprintf(stderr, "load seed    %s\n", path);
                return fromSeed;
            }
        }
    }

    ObjFunction *body = NULL;
    if (selfHosting()) {
        const JaiSourceFile *file = jaiSourceGet(fileId);
        if (!haveHash) {
            hash = jaiSourceHash(file->source, file->length);
            haveHash = true;
        }
        body = jaiSelfHostedCompileInto(file->source, file->length, path,
                                        module, hash, opts->codegen.optLevel);
        if (body == NULL) return NULL;
        /* The self-hosted front end is handed a path, not a module name, so it
         * names the module body from the file's stem: `b` where the C front end
         * uses the registered name `sub.b`. Only the importer knows the
         * qualified name, so it is applied here.
         *
         * This is not cosmetic. The name is serialised into the cache entry,
         * and a cached module whose name has lost its package cannot resolve
         * its own imports when loaded back -- a warm run failed where a cold
         * one passed. The differential oracle could not see it either: it
         * compared arity, flags, frame size, code and constants, but never the
         * function's name. */
        if (module->name != NULL) {
            body->name = module->name;
            body->qualifiedName = module->name;
        }
    } else {
        /* One front end remains and the branch above is the one that runs it.
         * Reaching here would mean `selfHosting()` said no outside the
         * bootstrap window, which nothing does. */
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
    /* NOT timed here: a module top-level imports other modules, so runModuleBody
     * nests and any accumulated total double-counts (it read 45-77ms against a
     * 17.7ms wall). Deserialisation below does not recurse, so that number is
     * clean and the remainder is reported by subtraction. */
    bool ranOk = jaiCallValue(OBJ_VAL(closure), 0, NULL, &ignored);
    if (!ranOk) {
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

    /* Before createModule: publishing this module as MOD_LOADING first would
     * make the compiler's own import of it look like a cycle.
     *
     * Rooted, because warming loads the whole compiler and every allocation in
     * it can collect. `pathKey` is interned but nothing else refers to it yet,
     * so without this it is freed underneath createModule -- a segfault that
     * appears only when the entry file is cached and an import is not. */
    jaiPushRoot(OBJ_VAL(pathKey));
    bool warmed = maybeWarmFor(path, false);
    jaiPopRoot();

    /* The warm can load this very module: `jaithon fmt` imports the compiler
     * as ordinary user code, so the import that triggered the warm is often
     * one the warm itself satisfies. The lookup above ran before it, so
     * without re-checking here a second ObjModule is built for a path that
     * already has one -- two copies of ast.jai, two NodeKind enums, and a dict
     * built under one that cannot be read with the other. That surfaced as
     * `KeyError: key <NodeKind> not found` from the formatter. */
    if (warmed &&
        jaiTableGetInterned(&vm.modules, pathKey, &existing) &&
        IS_MODULE(existing) && AS_MODULE(existing)->state == MOD_LOADED) {
        prefetchDrop();
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
        resolveMemoForgetAll();
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
    double t0 = 0.0;
    if (!wasLoading && getenv("JAI_FRONTEND_TIME")) {
        gFeTiming = true; gFeDeser = 0.0; gFeModules = 0;
        t0 = jaiClockMonotonic();
    }
    sLoadingFrontEnd = true;
    FrontEndPause pause = {false, 0};
    if (!wasLoading) pause = frontEndLoadBegin();
    ObjModule *module = jaiImportModule(dottedName, NULL);
    frontEndLoadEnd(pause, module != NULL);
    sLoadingFrontEnd = wasLoading;
    if (t0 != 0.0)
    {
        gFeTiming = false;
        fprintf(stderr, "[fe] import %s: %.3f ms total = %.3f deserialise "
                        "(%d modules) + %.3f running their top levels\n",
                dottedName, (jaiClockMonotonic() - t0) * 1000.0,
                gFeDeser * 1000.0, gFeModules,
                ((jaiClockMonotonic() - t0) - gFeDeser) * 1000.0);
    }
    /* The other candidate snapshot point, and the one that matters most.
     * `check`, `fmt`, `ast`, `doc` and `test` never reach warmFrontEnd: this
     * function sets sLoadingFrontEnd itself, which is precisely the guard
     * maybeWarmFor bails on, so they build the front end HERE instead. Audited
     * once per process and only under JAITHON_SNAPSHOT_AUDIT. */
    if (!wasLoading && module != NULL) {
        static bool audited = false;
        if (!audited) {
            audited = true;
            jaiSnapshotAudit("jaiImportFrontEndModule");
        }
    }
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
bool jaiSkipBodyOptimise = false;

static ObjBytes *bytesFromByteList(ObjList *list, const char *path) {
    size_t count = list->count > 0 ? (size_t)list->count : 0;
    uint8_t *raw = JAI_ALLOC(uint8_t, count > 0 ? count : 1);

    for (size_t i = 0; i < count; i++) {
        Value item = jaiListGet(list, i);
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
    FrontEndPause pause = {false, 0};
    if (!wasLoading) pause = frontEndLoadBegin();
    ObjModule *compiler = jaiImportModule(JAI_SELF_HOSTED_MODULE, NULL);
    frontEndLoadEnd(pause, compiler != NULL);
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
     * has to: a bridge that always took the default compiled at -O2 whatever
     * the caller asked for, so `-O0` was accepted and ignored.
     *
     * `fileId` carries information too. It was a hardcoded zero on the
     * reasoning that nothing read it; something does. Every span the
     * self-hosted front end emits records it, the line table is made of spans,
     * and a disassembler resolves a span back to a line through the source
     * registered under that id. With zero, a `--front=jai` build disassembled
     * with no line numbers and its tracebacks could not name a line. Every
     * caller therefore has to have registered its source and set
     * `module->sourceFileId` first. */
    Value args[6];
    args[0] = OBJ_VAL(jaiStringNew(source, length));
    jaiPushRoot(args[0]);
    args[1] = OBJ_VAL(jaiStringInternC(path));
    jaiPushRoot(args[1]);
    args[2] = BOOL_VAL(false);         /* release  */
    args[3] = INT_VAL(fileId);         /* fileId   */
    args[4] = INT_VAL(optLevel);
    /* `optimise_body`. See jaiSkipBodyOptimise. */
    args[5] = BOOL_VAL(!jaiSkipBodyOptimise);

    Value produced = NULL_VAL;
    bool called = jaiCallValue(entry, 6, args, &produced);
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
            /* The front end's own diagnostics say why, in the bag the driver
             * flushes, rendered the same way the C's are. E0902 is only for
             * the case where it produced neither an image nor a reason, which
             * is a bug in the front end rather than in the file. */
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

    /* One read, not two -- see the note at the other cache site. */
    if (opts->useCache) {
        size_t cacheLen = 0;
        uint8_t *cacheData = cacheReadForLoad(path, &cacheLen);
        if (cacheData != NULL) {
            ObjFunction *cached = NULL;
            if (jaiCacheFlagsMatchBuffer(cacheData, cacheLen, flags)) {
                cached = jaiDeserializeCached(cacheData, cacheLen, module, hash,
                                              path);
            }
            jaiCacheReadFree(cacheData, cacheLen);
            if (cached != NULL) {
                if (traceLoads()) fprintf(stderr, "load cache   %s\n", path);
                return cached;
            }
            /* Stale, corrupt, or from the other front end: compile it. */
        }
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

    maybeWarmFor(absolute, true);

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

/* The module name a *path* carries: `fileStem` plus the fallback an import
 * chain does not want -- a path with nothing left after the extension is
 * stripped names the main module, not the empty module.
 *
 * Shared rather than static because the CLI and the self-hosted bridge need the
 * same answer. A second copy lives in `module_name_for`
 * (lib/jaithon/compile/mod.jai) and must agree with this one: the name is a
 * constant in the record's pool, and a cached module whose name has lost its
 * package cannot resolve its own imports. */
void jaiModuleNameFor(const char *path, char *out, size_t outSize) {
    fileStem(out, outSize, path);
    if (out[0] == '\0') snprintf(out, outSize, "__main__");
}

/* ------------------------------------------------------------------ */
/* Import cycles                                                        */
/* ------------------------------------------------------------------ */

/* A module in a cycle compiles perfectly well on its own; it is the graph the
 * files form together that is wrong, so this is a whole-program question and
 * separate from compiling any one file. `check` asks it before the front end,
 * because every name a module in the cycle should have exported is missing and
 * the E0200 storm that follows is a consequence rather than a second problem.
 *
 * The C walked the graph itself, over trees the C parser built. The front end
 * answers the same question -- `import_cycles` in lib/jaithon/compile/mod.jai --
 * so the walk went with the parser that fed it. */
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
    /* The self-hosted front end is handed this id and stamps it into every span
     * it reports, so a diagnostic can be pointed back at the text. Without it
     * the spans name file 0 and every diagnostic renders with no source line
     * and no caret. */
    module->sourceFileId = fileId;
    jaiPushRoot(OBJ_VAL(module));

    bool ok;
    /* `check` promises "compile and type-check" (spec 8.5), so it emits too
     * and drops the body. One front end, one branch.
     *
     * The reason is NOT that codegen runs the bytecode verifier -- it does
     * not. `jaiVerifyChunk` has exactly one caller in the tree,
     * cli_cmd_build.c, so `check` never verifies anything. This comment said
     * otherwise for long enough that it was quoted back as authority.
     *
     * The real reason is that EMIT REPORTS DIAGNOSTICS THE CHECKER DOES NOT.
     * There are nine `_unsupported()` sites in lib/jaithon/compile/emit/emitter.jai
     * for shapes the checker deliberately accepts and the instruction set
     * cannot express -- `yield` and `await` (no generators), a generator
     * expression (needs a coroutine), `from m import *` (the loader needs an
     * explicit name list), and several pattern cases. Each is a clean E0902
     * with the checker reporting nothing at all. Drop the emit and `check`
     * starts passing files that cannot compile.
     *
     * Worth knowing before trying: NONE of those nine fires anywhere in
     * lib/std, lib/jaithon, packages or tests, so a corpus diff over the tree
     * comes back with zero differences and makes removing emit look safe.
     * See tests/errors/import_star.jai, added to give that path one live
     * example. */
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
