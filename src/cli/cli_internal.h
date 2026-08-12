/* cli_internal.h — plumbing shared only among the .c files in src/cli.
 *
 * main.c used to be one 1590-line file with a dozen clearly commented
 * sections (parse argv, collect source paths, compile one file, run each
 * subcommand, tear the process down). Each section now has its own file --
 * see the note at the top of main.c for the map of what moved where. None of
 * that split changes the public surface: cli.h is still the only thing the
 * rest of the tree links against. What follows is the connective tissue
 * between the pieces that used to be able to see each other simply by living
 * in the same translation unit, and now has to say so.
 */
#ifndef JAI_CLI_INTERNAL_H
#define JAI_CLI_INTERNAL_H

#include "cli.h"

/* ------------------------------------------------------------------ */
/* Diagnostics for command-line problems (main.c)                       */
/* ------------------------------------------------------------------ */

/* A bad invocation is not a language error, but it is still a user-facing one,
 * so it goes through the same renderer (code JAI_OK prints a bare "error:"). */
void cliError(const char *fmt, ...) JAI_PRINTF(1, 2);

static inline bool cliFlush(void) { return jaiDiagFlush(&gDiags, stderr); }

/* ------------------------------------------------------------------ */
/* Parse-time state that JaiCliOptions has no field for (cli_parse.c)   */
/* ------------------------------------------------------------------ */

typedef enum { COLOR_AUTO, COLOR_ALWAYS, COLOR_NEVER } ColorMode;

/* All consumed by main() immediately after jaiCliParse returns. The include
 * directories cannot be applied during parsing because jaiModulePathAdd writes
 * into vm.modulePath, which does not exist until jaiVMInit. */
extern const char **gIncludeDirs;
extern int          gIncludeCount;
extern ColorMode    gColorMode;
extern bool         gStrict;

/* Frees the storage jaiCliParse allocated for opts->inputs, opts->toolArgs,
 * opts->scriptArgv and gIncludeDirs -- one slab backs all four. */
void cliFreeOptions(JaiCliOptions *opts);

/* ------------------------------------------------------------------ */
/* Path collection (cli_paths.c)                                        */
/* ------------------------------------------------------------------ */

typedef JAI_VEC(char *) PathList;

void pathListFree(PathList *list);

/* Gathers every source path `opts` names -- expanding any directory into the
 * .jai files beneath it, sorted -- or the sources under `fallback` when no
 * paths were given and `fallback` is not NULL. False and a diagnostic on any
 * path that does not exist or cannot be read. */
bool collectAllInputs(const JaiCliOptions *opts, PathList *out,
                      const char *fallback);

/* ------------------------------------------------------------------ */
/* Compiling one file with the C front end (cli_compile.c)              */
/* ------------------------------------------------------------------ */

/* --front=jai, for the compile paths that are handed codegen options rather
 * than the whole invocation. */
extern bool gSelfHosted;

/* Reads `path` and compiles it. On success the module and the body are both
 * left on the GC root stack (module pushed first) and the caller pops two
 * roots once it is done with them. On failure nothing is rooted and the
 * diagnostics have already been rendered. */
ObjFunction *compileFile(const char *path, const CodegenOptions *codegen,
                         ObjModule **outModule, uint64_t *outSourceHash);

/* Same contract as compileFile, but for a buffer the caller already has --
 * takes ownership of `source` (length bytes, NUL-terminated). Used where the
 * buffer was read for another reason first: build hashes it before compiling,
 * disasm reads it to check for the .jaic magic before falling back to
 * treating it as source. */
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

/* Dispatches to the Jaithon-implemented tools: fmt, test, doc, bench. */
int runJaithonTool(const JaiCliOptions *opts, const char *tool);

#endif /* JAI_CLI_INTERNAL_H */
