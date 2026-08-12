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

/* One input, evaluated by the REPL, and then exit.
 *
 * jaiReplFeed is the door a typed line goes through, so `--eval EXPR` answers
 * exactly as `>>> EXPR` does: a statement runs and shows nothing, a bare
 * expression echoes its value, and a null value shows nothing either, because
 * null is the absence of a result rather than a result worth printing.
 * `--eval 'print(1)'` prints `1` and not the null that `print` returned.
 * Sharing the door is the point: two evaluators that agreed today would not
 * agree for long.
 *
 * The session around the door is carried by the other two entry points, for
 * the same reason: `jaithon --eval EXPR` and `printf 'EXPR\n' | jaithon repl`
 * have to answer alike or the flag is a second evaluator after all.
 * jaiReplConfigure hands the command line in, so -O0 and --release mean here
 * what they mean there instead of being accepted and ignored; jaiReplFailed
 * reads the verdict back out, because a diagnostic or an escaping exception is
 * reported by the REPL to stderr and then cleared from the bag and the module
 * state, and that flag is all that is left of it.
 *
 * An input that never ran because it is not a whole one is this command's own
 * failure rather than the session's — the prompt would have asked for the rest
 * of it — so it is reported here and joins the same status.
 *
 * The argument is fed a line at a time, because that door takes a line and a
 * shell hands over whatever was between the quotes. Handing it the whole thing
 * at once reads it as one input, so `--eval $'fn f() { ... }\nf()'` reported
 * `unexpected \`f\` after the statement` for a script the same three lines
 * typed at the prompt run happily. */
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
