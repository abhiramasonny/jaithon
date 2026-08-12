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
static void warmFrontEnd(void) {
    if (!sOptions.selfHosted || sLoadingFrontEnd || sFrontEndWarmed) return;
    sFrontEndWarmed = true;
    sLoadingFrontEnd = true;
    (void)jaiImportModule(JAI_SELF_HOSTED_MODULE, NULL);
    sLoadingFrontEnd = false;
    jaiClearException();
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
 * std.core is among them and every program loads it. */
static bool maybeWarmFor(const char *path) {
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

    /* Before createModule: publishing this module as MOD_LOADING first would
     * make the compiler's own import of it look like a cycle.
     *
     * Rooted, because warming loads the whole compiler and every allocation in
     * it can collect. `pathKey` is interned but nothing else refers to it yet,
     * so without this it is freed underneath createModule -- a segfault that
     * appears only when the entry file is cached and an import is not. */
    jaiPushRoot(OBJ_VAL(pathKey));
    bool warmed = maybeWarmFor(path);
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
