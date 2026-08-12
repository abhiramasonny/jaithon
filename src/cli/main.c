/* main.c — the jaithon command line.
 *
 * Parse argv into JaiCliOptions, bring the VM up with the flags the user asked
 * for, dispatch exactly one command, and tear everything down again cleanly
 * enough that a leak checker has nothing to say.
 *
 * This file used to hold every part of that: argument parsing, source-path
 * collection, single-file compilation, and each subcommand's handler, in one
 * 1590-line stack of clearly commented sections. Those sections were their
 * own concerns even when they lived together, so each now has its own file
 * (cli_internal.h is the private header that lets them still call each
 * other):
 *
 *   cli_usage.c         --help / --version text (pure formatting)
 *   cli_parse.c         argv -> JaiCliOptions
 *   cli_paths.c         expanding paths/directories into source file lists
 *   cli_compile.c       compiling one file into a module body
 *   cli_cmd_run.c       run, check, --eval
 *   cli_cmd_build.c     build
 *   cli_cmd_inspect.c   disasm, ast, tokens
 *   cli_cmd_tool.c      fmt, test, doc, bench (dispatch into jaithon.tool.*)
 *   repl.c              the interactive REPL (already split out before this)
 *
 * What stays here is the glue none of those pieces owns on their own: the
 * diagnostic helper every one of them calls into (cliError/cliFlush), the
 * dispatch switch that ties a parsed command to its handler, and process
 * setup/teardown (locale, color detection, the module search path, the VM
 * lifecycle) that runs exactly once regardless of which command was chosen.
 */

#if !defined(_POSIX_C_SOURCE)
#  define _POSIX_C_SOURCE 200809L
#endif
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#  define _DARWIN_C_SOURCE
#endif

#include <langinfo.h>
#include <locale.h>
#include <stdlib.h>
#include <unistd.h>

#include "cli_internal.h"

#include "../native/native.h"

/* ------------------------------------------------------------------ */
/* Diagnostics for command-line problems                                */
/* ------------------------------------------------------------------ */

/* A bad invocation is not a language error, but it is still a user-facing one,
 * so it goes through the same renderer (code JAI_OK prints a bare "error:"). */
void cliError(const char *fmt, ...) {
    char message[512];
    va_list args;
    va_start(args, fmt);
    vsnprintf(message, sizeof message, fmt, args);
    va_end(args);
    (void)jaiDiagError(JAI_OK, JAI_SPAN_NONE, "%s", message);
}

/* ------------------------------------------------------------------ */
/* Dispatch                                                             */
/* ------------------------------------------------------------------ */

/* The prelude is what makes unqualified names resolve (spec §9), so every
 * command that compiles or runs Jaithon needs it. `tokens` only lexes, and
 * `help`/`version` touch no source at all. */
static inline bool commandNeedsPrelude(JaiCommand command) {
    switch (command) {
    case CMD_RUN:
        /* jaiRunFile owns run-session prelude loading. */
    case CMD_TOKENS:
    case CMD_VERSION:
    case CMD_HELP:
        return false;
    default:
        return true;
    }
}

/* JAITHON_NO_PRELUDE is the switch the runtime reads, so --no-prelude sets it
 * and both sides agree however the process was started. */
static inline bool preludeDisabled(const JaiCliOptions *opts) {
    if (opts->noPrelude) return true;
    const char *flag = getenv("JAITHON_NO_PRELUDE");
    return flag != NULL && flag[0] != '\0' && strcmp(flag, "0") != 0;
}



int jaiCliDispatch(const JaiCliOptions *opts) {
    if (opts == NULL) return 1;

    gSelfHosted = opts->run.selfHosted;

    /* A tree without lib/std is still usable, so a missing prelude is a
     * warning — one that jaiLoadPrelude has already printed, including the
     * directories it searched. */
    if (!preludeDisabled(opts) && commandNeedsPrelude(opts->command)) {
        (void)jaiLoadPrelude();
    }

    switch (opts->command) {
    case CMD_RUN:     return cmdRun(opts);
    case CMD_REPL:    return jaiReplRun(opts);
    case CMD_EVAL:    return cmdEval(opts);
    case CMD_CHECK:   return cmdCheck(opts);
    case CMD_BUILD:   return cmdBuild(opts);
    case CMD_FMT:     return runJaithonTool(opts, "fmt");
    case CMD_TEST:    return runJaithonTool(opts, "test");
    case CMD_DOC:     return runJaithonTool(opts, "doc");
    case CMD_BENCH:   return runJaithonTool(opts, "bench");
    case CMD_DISASM:  return cmdDisasm(opts);
    case CMD_AST:     return cmdParseOnly(opts, false);
    case CMD_TOKENS:  return cmdParseOnly(opts, true);
    case CMD_VERSION: jaiCliPrintVersion(stdout); return 0;
    case CMD_HELP:    jaiCliPrintUsage(stdout);   return 0;
    }

    cliError("unimplemented command");
    (void)cliFlush();
    return 1;
}

/* ------------------------------------------------------------------ */
/* Process setup                                                        */
/* ------------------------------------------------------------------ */

/* The lexer, the string builtins and the diagnostic renderer all assume UTF-8,
 * so honour the user's locale when it already is UTF-8 and force one when it
 * is not. */
static void initLocale(void) {
    if (setlocale(LC_ALL, "") != NULL) {
        const char *codeset = nl_langinfo(CODESET);
        if (codeset != NULL &&
            (strcmp(codeset, "UTF-8") == 0 || strcmp(codeset, "utf8") == 0 ||
             strcmp(codeset, "UTF8") == 0 || strcmp(codeset, "utf-8") == 0)) {
            return;
        }
    }
    static const char *const candidates[] = {"C.UTF-8", "en_US.UTF-8", "UTF-8"};
    for (size_t i = 0; i < sizeof candidates / sizeof candidates[0]; i++) {
        if (setlocale(LC_ALL, candidates[i]) != NULL) return;
    }
    (void)setlocale(LC_CTYPE, "UTF-8");   /* last resort; failure is survivable */
}

/* NO_COLOR wins over CLICOLOR_FORCE only when it is set to a non-empty value,
 * which is what the two conventions agree on. --color overrides both. */
static bool detectColor(void) {
    switch (gColorMode) {
    case COLOR_ALWAYS: return true;
    case COLOR_NEVER:  return false;
    case COLOR_AUTO:   break;
    }

    const char *noColor = getenv("NO_COLOR");
    if (noColor != NULL && noColor[0] != '\0') return false;

    const char *force = getenv("CLICOLOR_FORCE");
    if (force != NULL && force[0] != '\0' && strcmp(force, "0") != 0) return true;

    const char *term = getenv("TERM");
    if (term != NULL && strcmp(term, "dumb") == 0) return false;

    return isatty(STDERR_FILENO) != 0;
}

/* The module search path starts at the directory holding the executable so
 * that an uninstalled build finds its own lib/. */
static void initModulePath(void) {
    const char *executable = jaiExecutablePath();
    char directory[JAI_MAX_PATH];
    directory[0] = '\0';
    if (executable != NULL) jaiPathDirname(directory, sizeof directory, executable);

    jaiModulePathInit(directory[0] != '\0' ? directory : NULL);
    for (int i = 0; i < gIncludeCount; i++) jaiModulePathAdd(gIncludeDirs[i]);
}

/* Settings with no home in JaiRunOptions travel to the runtime in the
 * environment, which is also where a spawned child process picks them up. */
static void exportEnvironmentFlags(const JaiCliOptions *opts) {
    char buffer[32];
    if (opts->threads > 0) {
        snprintf(buffer, sizeof buffer, "%d", opts->threads);
        (void)setenv("JAITHON_THREADS", buffer, 1);
    }
    if (opts->noGpu) (void)setenv("JAITHON_NO_GPU", "1", 1);
    if (gStrict)     (void)setenv("JAITHON_STRICT", "1", 1);
    /* jaiRunFile loads the prelude itself, and JAITHON_NO_PRELUDE is the switch
     * it reads to decide not to. */
    if (opts->noPrelude) (void)setenv("JAITHON_NO_PRELUDE", "1", 1);
}

int main(int argc, char **argv) {
    initLocale();
    jaiDiagInit(&gDiags);

    JaiCliOptions opts;
    if (!jaiCliParse(argc, argv, &opts)) {
        cliFreeOptions(&opts);
        jaiDiagFree(&gDiags);
        jaiSourceFreeAll();
        return 2;
    }

    /*
     * Help/version are pure process-level queries. Avoid VM initialization,
     * module-path discovery, environment export, prelude logic, and VM teardown
     * entirely. jaiCliPrintVersion's GPU query is independent of the VM.
     */
    if (opts.command == CMD_HELP || opts.command == CMD_VERSION) {
        if (opts.command == CMD_HELP)
            jaiCliPrintUsage(stdout);
        else
            jaiCliPrintVersion(stdout);

        cliFreeOptions(&opts);
        jaiDiagFree(&gDiags);
        jaiSourceFreeAll();
        fflush(stdout);
        return 0;
    }

    gDiags.colorOutput = detectColor();

    /* jaiVMInit keeps the flag fields it does not own, so these must be set
     * before the machine comes up. */
    vm.debugTrace  = opts.traceExec;
    vm.countInstructions = opts.traceExec || opts.showStats;
    vm.gcStress    = opts.gcStress;
    vm.releaseMode = opts.run.codegen.stripAsserts;
    vm.optLevel    = opts.run.codegen.optLevel;

    exportEnvironmentFlags(&opts);
    jaiVMInit();
    initModulePath();

    double started = 0.0;
    if (opts.showTiming)
        started = jaiClockMonotonic();

    int status = jaiCliDispatch(&opts);

    double elapsed = 0.0;
    if (opts.showTiming)
        elapsed = jaiClockMonotonic() - started;

    /* Ordered after whatever the program printed, even when stdout is a pipe
     * and therefore block-buffered. */
    if (opts.showTiming || opts.showStats) fflush(stdout);
    if (opts.showTiming) fprintf(stderr, "time: %.3f s\n", elapsed);
    if (opts.showStats)  jaiVMPrintStats(stderr);

    cliFreeOptions(&opts);
    jaiVMFree();
    /* After the type universe: an imported declaration's type is interned under
     * a name that lives in the signature's own arena. */
    jaiDiagFree(&gDiags);
    jaiSourceFreeAll();

    fflush(stdout);
    return status;
}
