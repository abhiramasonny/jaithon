/* cli_compile.c — shared step to compile a source path into an ObjFunction. */
#include "cli/cli_internal.h"

#include "runtime/modules/frontend.h"
#include "vm/bytecode/serialize.h"

bool gSelfHosted;

static void releaseSource(char *source, size_t length, const ObjModule *module) {
    if (module != NULL) {
        const JaiSourceFile *file = jaiSourceGet(module->sourceFileId);
        if (file != NULL && file->source == source) return;
    }
    JAI_FREE_ARRAY(char, source, length + 1);
}

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
