// main.c, the jaithon command line.

#if !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE
#endif

#include <langinfo.h>
#include <locale.h>
#include <stdlib.h>
#include <unistd.h>

#include "cli/cli_internal.h"

#include "native/native.h"

void cliError(const char *fmt, ...)
{
    char message[512];
    va_list args;
    va_start(args, fmt);
    vsnprintf(message, sizeof message, fmt, args);
    va_end(args);
    (void)jaiDiagError(JAI_OK, JAI_SPAN_NONE, "%s", message);
}

static inline bool commandNeedsPrelude(JaiCommand command)
{
    switch (command)
    {
    case CMD_RUN:
    case CMD_TOKENS:
    case CMD_VERSION:
    case CMD_HELP:
        return false;
    default:
        return true;
    }
}

static inline bool preludeDisabled(const JaiCliOptions *opts)
{
    if (opts->noPrelude)
        return true;
    const char *flag = getenv("JAITHON_NO_PRELUDE");
    return flag != NULL && flag[0] != '\0' && strcmp(flag, "0") != 0;
}

int jaiCliDispatch(const JaiCliOptions *opts)
{
    if (opts == NULL)
        return 1;

    gSelfHosted = opts->run.selfHosted;

    if (!preludeDisabled(opts) && commandNeedsPrelude(opts->command))
    {
        (void)jaiLoadPrelude();
    }

    switch (opts->command)
    {
    case CMD_RUN:
        return cmdRun(opts);
    case CMD_REPL:
        return jaiReplRun(opts);
    case CMD_EVAL:
        return cmdEval(opts);
    case CMD_CHECK:
        return cmdCheck(opts);
    case CMD_BUILD:
        return cmdBuild(opts);
    case CMD_FMT:
        return runJaithonTool(opts, "fmt");
    case CMD_TEST:
        return runJaithonTool(opts, "test");
    case CMD_DOC:
        return runJaithonTool(opts, "doc");
    case CMD_BENCH:
        return runJaithonTool(opts, "bench");
    case CMD_DISASM:
        return cmdDisasm(opts);
    case CMD_AST:
        return cmdParseOnly(opts, false);
    case CMD_TOKENS:
        return cmdParseOnly(opts, true);
    case CMD_VERSION:
        jaiCliPrintVersion(stdout);
        return 0;
    case CMD_HELP:
        jaiCliPrintUsage(stdout);
        return 0;
    }

    cliError("unimplemented command");
    (void)cliFlush();
    return 1;
}

static void initLocale(void)
{
    if (setlocale(LC_ALL, "") != NULL)
    {
        const char *codeset = nl_langinfo(CODESET);
        if (codeset != NULL &&
            (strcmp(codeset, "UTF-8") == 0 || strcmp(codeset, "utf8") == 0 ||
             strcmp(codeset, "UTF8") == 0 || strcmp(codeset, "utf-8") == 0))
        {
            return;
        }
    }
    static const char *const candidates[] = {"C.UTF-8", "en_US.UTF-8", "UTF-8"};
    for (size_t i = 0; i < sizeof candidates / sizeof candidates[0]; i++)
    {
        if (setlocale(LC_ALL, candidates[i]) != NULL)
            return;
    }
    (void)setlocale(LC_CTYPE, "UTF-8");
}

static bool detectColor(void)
{
    switch (gColorMode)
    {
    case COLOR_ALWAYS:
        return true;
    case COLOR_NEVER:
        return false;
    case COLOR_AUTO:
        break;
    }

    const char *noColor = getenv("NO_COLOR");
    if (noColor != NULL && noColor[0] != '\0')
        return false;

    const char *force = getenv("CLICOLOR_FORCE");
    if (force != NULL && force[0] != '\0' && strcmp(force, "0") != 0)
        return true;

    const char *term = getenv("TERM");
    if (term != NULL && strcmp(term, "dumb") == 0)
        return false;

    return isatty(STDERR_FILENO) != 0;
}

static void initModulePath(void)
{
    const char *executable = jaiExecutablePath();
    char directory[JAI_MAX_PATH];
    directory[0] = '\0';
    if (executable != NULL)
        jaiPathDirname(directory, sizeof directory, executable);

    jaiModulePathInit(directory[0] != '\0' ? directory : NULL);
    for (int i = 0; i < gIncludeCount; i++)
        jaiModulePathAdd(gIncludeDirs[i]);
}

static void exportEnvironmentFlags(const JaiCliOptions *opts)
{
    char buffer[32];
    if (opts->threads > 0)
    {
        snprintf(buffer, sizeof buffer, "%d", opts->threads);
        (void)setenv("JAITHON_THREADS", buffer, 1);
    }
    if (opts->noGpu)
        (void)setenv("JAITHON_NO_GPU", "1", 1);
    if (gStrict)
        (void)setenv("JAITHON_STRICT", "1", 1);
    if (opts->noPrelude)
        (void)setenv("JAITHON_NO_PRELUDE", "1", 1);
}

int main(int argc, char **argv)
{
    initLocale();
    jaiDiagInit(&gDiags);

    JaiCliOptions opts;
    if (!jaiCliParse(argc, argv, &opts))
    {
        cliFreeOptions(&opts);
        jaiDiagFree(&gDiags);
        jaiSourceFreeAll();
        return 2;
    }

    if (opts.command == CMD_HELP || opts.command == CMD_VERSION)
    {
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

    vm.debugTrace = opts.traceExec;
    vm.countInstructions = opts.traceExec || opts.showStats;
    vm.gcStress = opts.gcStress;
    vm.releaseMode = opts.run.codegen.stripAsserts;
    vm.optLevel = opts.run.codegen.optLevel;

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

    if (opts.showTiming || opts.showStats)
        fflush(stdout);
    if (opts.showTiming)
        fprintf(stderr, "time: %.3f s\n", elapsed);
    if (opts.showStats)
        jaiVMPrintStats(stderr);

    cliFreeOptions(&opts);
    jaiVMFree();
    jaiDiagFree(&gDiags);
    jaiSourceFreeAll();

    fflush(stdout);
    return status;
}
