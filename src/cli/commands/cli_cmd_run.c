/* cli_cmd_run.c — run, check and --eval: the commands that execute or verify
 * source directly, without producing an artifact (build) or dumping a form
 * of it (disasm/ast/tokens).
 */
#include "cli/cli_internal.h"

/* ------------------------------------------------------------------ */
/* run / check                                                          */
/* ------------------------------------------------------------------ */

int cmdRun(const JaiCliOptions *opts) {
    return jaiRunFile(opts->inputs[0], &opts->run, opts->scriptArgc,
                      opts->scriptArgv);
}

int cmdCheck(const JaiCliOptions *opts) {
    PathList files;
    if (!collectAllInputs(opts, &files, NULL)) {
        (void)cliFlush();
        pathListFree(&files);
        return 1;
    }

    int failed = 0;
    for (int i = 0; i < files.count; i++) {
        if (jaiCheckFile(files.data[i], &opts->run) != 0) failed++;
    }

    if (failed == 0 && files.count > 1) {
        fprintf(stderr, "checked %d files, no errors\n", files.count);
    }
    jaiImportGraphFree();
    pathListFree(&files);
    return failed == 0 ? 0 : 1;
}

/* ------------------------------------------------------------------ */
/* --eval                                                               */
/* ------------------------------------------------------------------ */

int cmdEval(const JaiCliOptions *opts) {
    bool incomplete = false;
    jaiReplConfigure(opts);

    char stackLine[512];
    const char *cursor = opts->eval;

    while (*cursor != '\0') {
        const char *const end = strchr(cursor, '\n');
        const size_t length =
            end != NULL ? (size_t)(end - cursor) : strlen(cursor);

        char *line = stackLine;

        if (length + 1 > sizeof stackLine)
            line = JAI_ALLOC(char, length + 1);

        memcpy(line, cursor, length);
        line[length] = '\0';

        (void)jaiReplFeed(line, &incomplete);

        if (line != stackLine)
            JAI_FREE_ARRAY(char, line, length + 1);

        if (end == NULL)
            break;

        cursor = end + 1;
    }

    fflush(stdout);

    if (incomplete) {
        cliError("--eval expects a complete expression, but `%s` is the start "
                 "of one", opts->eval);
        (void)cliFlush();
        return 1;
    }

    return jaiReplFailed() ? 1 : 0;
}
