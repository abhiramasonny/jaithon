/* cli.h — command dispatch, argument parsing, and the REPL. */
#ifndef JAI_CLI_H
#define JAI_CLI_H

#include "runtime/runtime.h"

typedef enum {
    CMD_RUN,        /* default: jaithon file.jai  */
    CMD_REPL,
    CMD_CHECK,
    CMD_BUILD,
    CMD_FMT,
    CMD_TEST,
    CMD_DOC,
    CMD_BENCH,
    CMD_DISASM,
    CMD_AST,
    CMD_TOKENS,
    CMD_VERSION,
    CMD_HELP,
    CMD_EVAL,       /* --eval EXPR: one input, then exit */
} JaiCommand;

typedef struct {
    JaiCommand   command;
    const char **inputs;
    int          inputCount;
    const char  *output;
    const char  *eval;          /* the expression --eval was given */

    JaiRunOptions run;

    bool  showTiming;
    bool  showStats;
    bool  traceExec;
    bool  gcStress;
    unsigned gcStressEvery;
    bool  noPrelude;
    bool  fmtCheck;
    bool  jsonOutput;
    int   threads;
    bool  noGpu;

    /* Everything that follows a tool command (fmt, test, doc, bench): the
     * options the CLI does not recognise *and* the paths, kept in the order
     * the user typed them and never rewritten. Those tools own their own
     * flags — `--verbose`, `--doc`, `--filter X` — and the CLI has no business
     * knowing which of them take a value; splitting flags from paths would
     * reorder `fmt PATH --filter X` into `fmt --filter PATH X`. Anything
     * unknown before a tool command is named is still an error.
     *
     * Both `--name value` and `--name=value` reach the tool intact, because
     * every tool accepts both (lib/jaithon/tool/args.jai). */
    const char **toolArgs;
    int          toolArgCount;

    /* Arguments forwarded to the script after `--`. */
    char **scriptArgv;
    int    scriptArgc;
} JaiCliOptions;

/* Parse argv. Returns false and prints usage on a bad invocation. */
bool jaiCliParse(int argc, char **argv, JaiCliOptions *out);
int  jaiCliDispatch(const JaiCliOptions *opts);
void jaiCliPrintUsage(FILE *out);
void jaiCliPrintVersion(FILE *out);

/* ------------------------------------------------------------------ */
/* REPL                                                                 */
/* ------------------------------------------------------------------ */

int  jaiReplRun(const JaiCliOptions *opts);
/* Feed one line; returns true when the statement is complete and was run. */
bool jaiReplFeed(const char *line, bool *outIncomplete);

/* The two halves of a session that `--eval` needs without the loop between
 * them: the command line going in, and the verdict coming back out. */
void jaiReplConfigure(const JaiCliOptions *opts);
bool jaiReplFailed(void);

#endif /* JAI_CLI_H */
