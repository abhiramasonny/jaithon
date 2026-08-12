/* cli_cmd_build.c — `build`: compile to a .jaic image or the __jaicache__. */
#include "cli_internal.h"

#include "../vm/serialize.h"
#include "../vm/verify.h"

static int buildOne(const char *path, const JaiCliOptions *opts) {
    ObjModule *module = NULL;
    uint64_t hash = 0;

    ObjFunction *body =
        compileFile(path, &opts->run.codegen, &module, &hash);

    if (body == NULL) return 1;

    int status = 0;
    char problem[256];
    if (!jaiVerifyChunk(body, problem, sizeof problem)) {
        (void)jaiDiagError(E0902_INTERNAL_ERROR, JAI_SPAN_NONE,
                           "%s: the generated chunk is malformed: %s", path,
                           problem);
        (void)cliFlush();
        jaiPopRoots(2);
        return 1;
    }

    const uint32_t flags =
        (opts->run.codegen.debugInfo ? JAIC_FLAG_DEBUG : 0u) |
        (opts->run.codegen.stripAsserts ? JAIC_FLAG_RELEASE : 0u);

    if (opts->output != NULL) {
        size_t size = 0;
        uint8_t *image = jaiSerializeModule(module, body, hash, flags, &size);
        if (image == NULL) {
            cliError("%s: could not serialise the compiled module", path);
            status = 1;
        } else {
            if (jaiWriteFile(opts->output, image, size)) {
                printf("wrote %s (%zu bytes)\n", opts->output, size);
            } else {
                cliError("cannot write %s", opts->output);
                status = 1;
            }
            JAI_FREE_ARRAY(uint8_t, image, size);
        }
    } else if (jaiCacheStore(path, module, body, hash, flags)) {
        char cachePath[JAI_MAX_PATH];
        jaiCachePathFor(path, cachePath, sizeof cachePath);
        printf("wrote %s\n", cachePath);
    } else {
        cliError("%s: could not write the bytecode cache", path);
        status = 1;
    }

    (void)cliFlush();
    jaiPopRoots(2);
    return status;
}

int cmdBuild(const JaiCliOptions *opts) {
    if (opts->output != NULL && opts->inputCount > 1) {
        cliError("--out takes a single output file, but %d inputs were given",
                 opts->inputCount);
        (void)cliFlush();
        return 1;
    }

    PathList files;
    if (!collectAllInputs(opts, &files, NULL)) {
        (void)cliFlush();
        pathListFree(&files);
        return 1;
    }

    int status = 0;
    for (int i = 0; i < files.count; i++) {
        if (buildOne(files.data[i], opts) != 0) status = 1;
    }
    pathListFree(&files);
    return status;
}
