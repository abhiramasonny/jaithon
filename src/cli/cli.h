/* cli.h — command dispatch, argument parsing, and the REPL. */
#ifndef JAI_CLI_H
#define JAI_CLI_H

#include "runtime/runtime.h"

typedef enum {
    CMD_RUN,
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
    CMD_EVAL,
} JaiCommand;

typedef struct {
    JaiCommand   command;
    const char **inputs;
    int          inputCount;
    const char  *output;
    const char  *eval;

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

    const char **toolArgs;
    int          toolArgCount;

    char **scriptArgv;
    int    scriptArgc;
} JaiCliOptions;

bool jaiCliParse(int argc, char **argv, JaiCliOptions *out);
int  jaiCliDispatch(const JaiCliOptions *opts);
void jaiCliPrintUsage(FILE *out);
void jaiCliPrintVersion(FILE *out);

/* ------------------------------------------------------------------ */
/* REPL                                                                 */
/* ------------------------------------------------------------------ */

int  jaiReplRun(const JaiCliOptions *opts);
bool jaiReplFeed(const char *line, bool *outIncomplete);

void jaiReplConfigure(const JaiCliOptions *opts);
bool jaiReplFailed(void);

#endif /* JAI_CLI_H */
