/* cli_compile.c — compiling one file into a module body.
 *
 * Both `build` and `disasm` need to turn a source path into a compiled
 * ObjFunction before they can do their own, different things with it
 * (serialize it to a .jaic, or walk it printing bytecode). This is that
 * shared step: naming the module, choosing the C or self-hosted front end,
 * and handing back a rooted function or a rendered diagnostic.
 */
#include "cli_internal.h"

#include "../runtime/frontend.h"
#include "../vm/serialize.h"

/* --front=jai, for the compile paths that are handed codegen options rather
 * than the whole invocation. */
bool gSelfHosted;

/* jaiCompileSource takes a const buffer, so it registers its own copy with the
 * diagnostic engine. Should a front end ever hand this exact buffer to
 * jaiSourceAdd instead, freeing it here would leave the registry dangling — so
 * check before letting go of it. */
static void releaseSource(char *source, size_t length, const ObjModule *module) {
    if (module != NULL) {
        const JaiSourceFile *file = jaiSourceGet(module->sourceFileId);
        if (file != NULL && file->source == source) return;
    }
    JAI_FREE_ARRAY(char, source, length + 1);
}

/* Compile `path` into a module body. On success the module and the body are
 * both left on the GC root stack (module pushed first) and the caller pops two
 * roots once it is done with them. On failure nothing is rooted and the
 * diagnostics have already been rendered. */
ObjFunction *compileOwnedSource(const char *path,
                                char *source, size_t length,
                                const CodegenOptions *codegen,
                                ObjModule **outModule,
                                uint64_t *outSourceHash) {
    char name[256];
    char absolute[JAI_MAX_PATH];

    jaiModuleNameFor(path, name, sizeof name);
    if (!jaiPathAbsolute(absolute, sizeof absolute, path)) {
        snprintf(absolute, sizeof absolute, "%s", path);
    }

    /*
     * build needs the source hash and the self-hosted front end needs the same
     * hash. Compute it from the exact buffer being compiled, never by reopening
     * the file after compilation.
     */
    uint64_t sourceHash = 0;
    if (gSelfHosted || outSourceHash != NULL)
        sourceHash = jaiSourceHash(source, length);

    if (outSourceHash != NULL)
        *outSourceHash = sourceHash;

    ObjString *moduleName = jaiStringInternC(name);
    jaiPushRoot(OBJ_VAL(moduleName));

    ObjString *modulePath = jaiStringInternC(absolute);
    jaiPushRoot(OBJ_VAL(modulePath));

    ObjModule *module = jaiModuleNew(moduleName, modulePath);

    jaiPopRoots(2);
    jaiPushRoot(OBJ_VAL(module));

    ObjFunction *body;

    if (gSelfHosted) {
        /*
         * The self-hosted bridge does not register the source itself, so give
         * ownership to the diagnostic registry before compiling it.
         */
        const int fileId = jaiSourceAdd(absolute, source, length);
        module->sourceFileId = fileId;

        const JaiSourceFile *const registered = jaiSourceGet(fileId);
        body = jaiSelfHostedCompileInto(
            registered != NULL ? registered->source : source,
            registered != NULL ? registered->length : length,
            absolute, module, sourceHash, codegen->optLevel);
    } else {
        body = jaiCompileSource(source, length, absolute, module, codegen);
        releaseSource(source, length, module);
    }

    const bool hadErrors = cliFlush();

    if (body == NULL || hadErrors) {
        jaiPopRoot();

        if (body == NULL && !hadErrors) {
            cliError("%s: compilation failed", path);
            (void)cliFlush();
        }

        return NULL;
    }

    jaiPushRoot(OBJ_VAL(body));

    if (outModule != NULL)
        *outModule = module;

    return body;
}

ObjFunction *compileFile(const char *path, const CodegenOptions *codegen,
                         ObjModule **outModule,
                         uint64_t *outSourceHash) {
    size_t length = 0;
    char *source = jaiReadFile(path, &length);

    if (source == NULL) {
        cliError("cannot read %s", path);
        (void)cliFlush();
        return NULL;
    }

    return compileOwnedSource(path, source, length, codegen,
                              outModule, outSourceHash);
}
