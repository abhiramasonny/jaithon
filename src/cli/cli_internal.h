/* Plumbing shared among the .c files in src/cli. */
#ifndef JAI_CLI_INTERNAL_H
#define JAI_CLI_INTERNAL_H

#include "cli/cli.h"

/* ------------------------------------------------------------------ */
/* Diagnostics for command-line problems (main.c)                       */
/* ------------------------------------------------------------------ */

void cliError(const char *fmt, ...) JAI_PRINTF(1, 2);

static inline bool cliFlush(void) { return jaiDiagFlush(&gDiags, stderr); }

/* ------------------------------------------------------------------ */
/* Parse-time state that JaiCliOptions has no field for (cli_parse.c)   */
/* ------------------------------------------------------------------ */

typedef enum { COLOR_AUTO, COLOR_ALWAYS, COLOR_NEVER } ColorMode;

extern const char **gIncludeDirs;
extern int          gIncludeCount;
extern ColorMode    gColorMode;
extern bool         gStrict;

void cliFreeOptions(JaiCliOptions *opts);

/* ------------------------------------------------------------------ */
/* Path collection (cli_paths.c)                                        */
/* ------------------------------------------------------------------ */

typedef JAI_VEC(char *) PathList;

void pathListFree(PathList *list);

bool collectAllInputs(const JaiCliOptions *opts, PathList *out,
                      const char *fallback);

/* ------------------------------------------------------------------ */
/* Compiling one file with the C front end (cli_compile.c)              */
/* ------------------------------------------------------------------ */

extern bool gSelfHosted;

ObjFunction *compileFile(const char *path, const CodegenOptions *codegen,
                         ObjModule **outModule, uint64_t *outSourceHash);

ObjFunction *compileOwnedSource(const char *path, char *source, size_t length,
                                const CodegenOptions *codegen,
                                ObjModule **outModule, uint64_t *outSourceHash);

/* ------------------------------------------------------------------ */
/* Subcommand handlers (cli_cmd_*.c)                                    */
/* ------------------------------------------------------------------ */

int cmdRun(const JaiCliOptions *opts);
int cmdCheck(const JaiCliOptions *opts);
int cmdEval(const JaiCliOptions *opts);

int cmdBuild(const JaiCliOptions *opts);

int cmdDisasm(const JaiCliOptions *opts);
int cmdParseOnly(const JaiCliOptions *opts, bool tokensOnly);

int runJaithonTool(const JaiCliOptions *opts, const char *tool);

#endif /* JAI_CLI_INTERNAL_H */
