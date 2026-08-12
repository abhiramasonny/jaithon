/* cli_cmd_inspect.c — disasm, ast and tokens: dumping a compiled or parsed
 * form of the source rather than running it. All three share openOutput
 * (write to --out, or stdout) and the same "collect paths, loop, report a
 * combined exit status" shape, which is what keeps them in one file even
 * though disasm and ast/tokens otherwise go through different front-end
 * entry points.
 */
#include "cli/cli_internal.h"

#include "runtime/modules/frontend.h"
#include "vm/bytecode/chunk.h"
#include "vm/bytecode/serialize.h"

/* Open --out, or stdout when it was not given. */
static FILE *openOutput(const JaiCliOptions *opts, FILE **outOpened) {
    *outOpened = NULL;
    if (opts->output == NULL) return stdout;

    FILE *f = fopen(opts->output, "w");
    if (f == NULL) {
        cliError("cannot write %s", opts->output);
        (void)cliFlush();
        return NULL;
    }
    *outOpened = f;
    return f;
}

/* Nested functions live in the enclosing chunk's constant pool, so walking the
 * constants reaches every function the file compiled to. */
static void disassembleTree(FILE *out, const ObjFunction *fn, int depth) {
    if (fn == NULL || depth > 32) return;

    const char *name = fn->qualifiedName != NULL ? fn->qualifiedName->chars
                     : fn->name != NULL          ? fn->name->chars
                                                 : "<anonymous>";
    jaiDisassembleChunk(out, &fn->chunk, name);
    if (fn->exceptionCount > 0) {
        fprintf(out, "; exception table (%u entries)\n",
                (unsigned)fn->exceptionCount);
        for (uint16_t i = 0; i < fn->exceptionCount; i++) {
            const ExceptionEntry *e = &fn->exceptions[i];
            fprintf(out, ";   [%04u,%04u) -> %04u  type=", e->start, e->end,
                    e->handler);
            if (e->typeConst == UINT32_MAX) fputs("any\n", out);
            else fprintf(out, "K%u\n", e->typeConst);
        }
    }
    fputc('\n', out);

    for (int i = 0; i < fn->chunk.constants.count; i++) {
        Value v = fn->chunk.constants.data[i];
        if (IS_FUNCTION(v)) disassembleTree(out, AS_FUNCTION(v), depth + 1);
    }
}


/* ------------------------------------------------------------------ */
/* Disassembling a .jaic image                                          */
/* ------------------------------------------------------------------ */

/* A .jaic is bytecode, not source, so `disasm` has to recognise it rather than
 * hand it to the lexer — which used to report `unexpected control character
 * U+0000` on the header. The layout is spec/BYTECODE.md 7; only the fixed
 * prefix is decoded here, and jaiDeserializeModule validates the rest. */
static inline bool imageIsJaic(const uint8_t *data, size_t size) {
    return size >= 4 && memcmp(data, JAIC_MAGIC, 4) == 0;
}

static inline uint16_t imageU16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }

static inline uint32_t imageU32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16)
         | ((uint32_t)p[3] << 24);
}

static inline uint64_t imageU64(const uint8_t *p) {
    return (uint64_t)imageU32(p) | ((uint64_t)imageU32(p + 4) << 32);
}

/* Print the header, then the code. Returns false only when the image cannot be
 * read at all; a version or build mismatch is reported and the body is still
 * attempted, because refusing to show a stale image is unhelpful in a viewer. */
static bool disassembleImage(FILE *out, const char *path, const uint8_t *data,
                             size_t size) {
    /* magic(4) version(2) flags(2) compiler(4) buildId(4) srcHash(8)
     * srcPathLen(2) ... */
    if (size < 26) {
        cliError("%s: truncated .jaic image (%zu bytes)", path, size);
        return false;
    }
    const uint16_t version  = imageU16(data + 4);
    const uint16_t flags    = imageU16(data + 6);
    const uint32_t compiler = imageU32(data + 8);
    const uint32_t buildId  = imageU32(data + 12);
    const uint64_t srcHash  = imageU64(data + 16);
    const uint16_t pathLen  = imageU16(data + 24);
    const uint32_t thisBuildId = jaiBuildId();

    fprintf(out, "; %s\n", path);
    fprintf(out, "; container version %u", version);
    if (version != JAIC_VERSION) fprintf(out, "  (this build expects %d)", JAIC_VERSION);
    fputc('\n', out);
    fprintf(out, "; compiler version  %u\n", compiler);
    fprintf(out, "; build id          0x%08x%s\n", buildId,
            buildId == thisBuildId ? "  (this build)" : "  (a different build)");
    fprintf(out, "; source hash       0x%016llx\n", (unsigned long long)srcHash);
    fprintf(out, "; flags             %s%s\n",
            (flags & JAIC_FLAG_DEBUG) ? "debug " : "",
            (flags & JAIC_FLAG_RELEASE) ? "release" : "");
    if (pathLen > 0 && (size_t)26 + pathLen <= size)
        fprintf(out, "; source            %.*s\n", (int)pathLen, (const char *)data + 26);
    fputc('\n', out);

    ObjString *modulePath = jaiStringInternC(path);
    jaiPushRoot(OBJ_VAL(modulePath));

    ObjModule *module = jaiModuleNew(modulePath, modulePath);

    jaiPopRoot();
    if (module == NULL) { cliError("%s: out of memory", path); return false; }
    jaiPushRoot(OBJ_VAL(module));

    /* Pass the image's own source hash so the check is a no-op: the point here
     * is to look at what the file contains, not to decide whether it is a
     * usable cache entry for some source file. */
    ObjFunction *body = jaiDeserializeModule(data, size, module, srcHash);
    if (body == NULL) {
        jaiPopRoot();
        cliError("%s: this build cannot load the image", path);
        if (compiler != JAI_COMPILER_VERSION || buildId != thisBuildId)
            cliError("it was written by a different compiler build; recompile "
                     "the source to inspect it");
        return false;
    }
    jaiPushRoot(OBJ_VAL(body));
    disassembleTree(out, body, 0);
    jaiPopRoots(2);
    return true;
}

int cmdDisasm(const JaiCliOptions *opts) {
    FILE *opened = NULL;
    FILE *out = openOutput(opts, &opened);
    if (out == NULL) return 1;

    PathList files;
    if (!collectAllInputs(opts, &files, NULL)) {
        (void)cliFlush();
        pathListFree(&files);
        if (opened != NULL) fclose(opened);
        return 1;
    }

    int status = 0;
    for (int i = 0; i < files.count; i++) {
        size_t imageSize = 0;
        char *image = jaiReadFile(files.data[i], &imageSize);
        if (image != NULL && imageIsJaic((const uint8_t *)image, imageSize)) {
            if (!disassembleImage(out, files.data[i], (const uint8_t *)image,
                                  imageSize))
                status = 1;
            JAI_FREE_ARRAY(char, image, imageSize + 1);
            continue;
        }
        if (image == NULL) {
            cliError("cannot read %s", files.data[i]);
            (void)cliFlush();
            status = 1;
            continue;
        }

        ObjModule *module = NULL;

        /* compileOwnedSource takes ownership of image. */
        ObjFunction *body =
            compileOwnedSource(files.data[i], image, imageSize,
                               &opts->run.codegen, &module, NULL);

        if (body == NULL) {
            status = 1;
            continue;
        }
        fprintf(out, "; %s\n", files.data[i]);
        disassembleTree(out, body, 0);
        jaiPopRoots(2);
    }

    pathListFree(&files);
    if (opened != NULL && fclose(opened) != 0) {
        cliError("cannot write %s", opts->output);
        (void)cliFlush();
        status = 1;
    }
    return status;
}

/* Parse one file for `ast` and `tokens`; the source buffer is handed to the
 * diagnostic registry, which owns it until jaiSourceFreeAll. */
int cmdParseOnly(const JaiCliOptions *opts, bool tokensOnly) {
    FILE *opened = NULL;
    FILE *out = openOutput(opts, &opened);
    if (out == NULL) return 1;

    PathList files;
    if (!collectAllInputs(opts, &files, NULL)) {
        (void)cliFlush();
        pathListFree(&files);
        if (opened != NULL) fclose(opened);
        return 1;
    }

    int status = 0;
    for (int i = 0; i < files.count; i++) {
        const char *path = files.data[i];
        size_t length = 0;
        char *source = jaiReadFile(path, &length);
        if (source == NULL) {
            cliError("cannot read %s", path);
            (void)cliFlush();
            status = 1;
            continue;
        }
        int fileId = jaiSourceAdd(path, source, length);

        if (tokensOnly) {
            ObjString *dump = jaiFrontEndTokenText(source, length, fileId);
            if (dump == NULL) {
                status = 1;
            } else {
                if (files.count > 1) fprintf(out, "; %s\n", path);
                fwrite(dump->chars, 1, dump->length, out);
            }
            if (cliFlush()) status = 1;
            continue;
        }

        ObjString *dump = jaiFrontEndAstText(source, length, path, fileId,
                                             opts->jsonOutput);
        if (dump == NULL) {
            /* The front end threw: it said why, and there is no tree to print. */
            jaiClearException();
            status = 1;
        } else {
            if (!opts->jsonOutput && files.count > 1) {
                fprintf(out, "; %s\n", path);
            }
            fwrite(dump->chars, 1, dump->length, out);
            /* Both forms end in a newline, as `jaiAstPrint` and `jaiAstToJson`
             * did: a dump that runs into the next one is not readable. */
            fputc('\n', out);
        }
        if (cliFlush()) status = 1;
    }

    pathListFree(&files);
    if (opened != NULL && fclose(opened) != 0) {
        cliError("cannot write %s", opts->output);
        (void)cliFlush();
        status = 1;
    }
    return status;
}
