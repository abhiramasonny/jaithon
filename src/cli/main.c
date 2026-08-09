/* main.c — the jaithon command line.
 *
 * Parse argv into JaiCliOptions, bring the VM up with the flags the user asked
 * for, dispatch exactly one command, and tear everything down again cleanly
 * enough that a leak checker has nothing to say.
 *
 * The four commands that are written in Jaithon — fmt, test, doc and bench —
 * are dispatched by importing their `jaithon.tool.*` module and calling its `main`.
 * There is deliberately no C implementation of them here: the formatter is
 * canonical (spec §14) and having two of it is exactly the Jaithon 2 mistake.
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

#include "cli.h"

#include "../codegen/codegen.h"
#include "../lang/ast.h"
#include "../lang/lexer.h"
#include "../lang/parser.h"
#include "../native/native.h"
#include "../sema/modsig.h"
#include "../sema/semadump.h"
#include "../vm/chunk.h"
#include "../vm/serialize.h"

/* ------------------------------------------------------------------ */
/* Parse-time state that JaiCliOptions has no field for                 */
/* ------------------------------------------------------------------ */

typedef enum { COLOR_AUTO, COLOR_ALWAYS, COLOR_NEVER } ColorMode;

/* All consumed by main() immediately after jaiCliParse returns. The include
 * directories cannot be applied during parsing because jaiModulePathAdd writes
 * into vm.modulePath, which does not exist until jaiVMInit. */
static const char **gIncludeDirs;
static int          gIncludeCount;
static int          gArgSlots;        /* capacity of inputs[] and scriptArgv[] */
static ColorMode    gColorMode = COLOR_AUTO;
static bool         gStrict;
/* --front=jai, for the compile paths that are handed codegen options rather
 * than the whole invocation. */
static bool         gSelfHosted;

/* ------------------------------------------------------------------ */
/* Diagnostics for command-line problems                                */
/* ------------------------------------------------------------------ */

/* A bad invocation is not a language error, but it is still a user-facing one,
 * so it goes through the same renderer (code JAI_OK prints a bare "error:"). */
static void cliError(const char *fmt, ...) JAI_PRINTF(1, 2);

static void cliError(const char *fmt, ...) {
    char message[512];
    va_list args;
    va_start(args, fmt);
    vsnprintf(message, sizeof message, fmt, args);
    va_end(args);
    (void)jaiDiagError(JAI_OK, JAI_SPAN_NONE, "%s", message);
}

static bool cliFlush(void) { return jaiDiagFlush(&gDiags, stderr); }

/* ------------------------------------------------------------------ */
/* Usage and version                                                    */
/* ------------------------------------------------------------------ */

void jaiCliPrintUsage(FILE *out) {
    if (out == NULL) return;
    fputs(
        "jaithon " JAI_VERSION_STRING " — the Jaithon programming language\n"
        "\n"
        "usage:\n"
        "  jaithon                        start the REPL\n"
        "  jaithon FILE [args...]         run FILE (shorthand for `run`)\n"
        "  jaithon COMMAND [options] [paths...]\n"
        "\n"
        "commands:\n"
        "  run FILE [-- args...]      compile FILE, run it, then call its main()\n"
        "  repl                       interactive read-eval-print loop\n"
        "  check PATH...              compile and type-check without running\n"
        "  build FILE...              compile to a .jaic image\n"
        "  fmt [--check|--diff] PATH... format source          (jaithon.tool.fmt)\n"
        "  test [PATH...]             discover and run tests   (jaithon.tool.test)\n"
        "  doc [--out DIR] [PATH...]  generate documentation   (jaithon.tool.doc)\n"
        "  bench [PATH...]            run benchmarks           (jaithon.tool.bench)\n"
        "  disasm FILE...             print the compiled bytecode\n"
        "  ast FILE...                print the syntax tree\n"
        "  tokens FILE...             print the token stream\n"
        "  version                    print version information\n"
        "  help                       print this message\n"
        "\n"
        "options:\n"
        "  -h, --help                 print this message and exit\n"
        "  -v, --version              print version information and exit\n"
        "      --eval EXPR            evaluate EXPR, print its value, and exit\n"
        "  -O0 -O1 -O2 -O3            optimisation level (default -O2)\n"
        "      --release              strip asserts and debug information\n"
        "      --debug-trace          trace every instruction as it executes\n"
        "      --gc-stress            collect on every allocation\n"
        "      --no-cache             ignore and do not write __jaicache__\n"
        "      --no-prelude           do not load std.prelude\n"
        "      --strict               unannotated parameters are an error\n"
        "      --time                 print elapsed wall time\n"
        "      --stats                print VM, inline-cache and GC statistics\n"
        "      --threads=N            worker threads for std.thread\n"
        "      --no-gpu               disable GPU acceleration\n"
        "      --front=c|jai          front end: C (default) or self-hosted\n"
        "      --emit=ast|bc|tokens   compile and dump the given form\n"
        "      --json                 machine-readable output where supported\n"
        "      --color=auto|always|never\n"
        "      --out PATH             write the output to PATH\n"
        "      --dump-sema PATH       write the checker's per-node decisions\n"
        "\n"
        "Every option that takes a value accepts both `--name value` and\n"
        "`--name=value`, here and in fmt, test, doc and bench alike.\n"
        "  -I PATH                    add PATH to the module search path\n"
        "      --bootstrap-verify     diff the C and self-hosted front ends\n"
        "  --                         end options; the rest goes to the script\n",
        out);
}

void jaiCliPrintVersion(FILE *out) {
    if (out == NULL) return;
    fprintf(out, "jaithon %s\n", JAI_VERSION_STRING);
    fprintf(out, "bytecode %u, .jaic container %d\n",
            (unsigned)JAI_COMPILER_VERSION, JAIC_VERSION);
    if (jaiGpuAvailable()) {
        const char *device = jaiGpuDeviceName();
        fprintf(out, "gpu: %s\n", device != NULL ? device : "available");
    } else {
        fputs("gpu: unavailable\n", out);
    }
#ifdef JAI_HAVE_READLINE
    fputs("repl: readline\n", out);
#else
    fputs("repl: plain\n", out);
#endif
}

/* ------------------------------------------------------------------ */
/* Argument parsing                                                     */
/* ------------------------------------------------------------------ */

static const struct {
    const char *name;
    JaiCommand  command;
} kCommands[] = {
    {"run", CMD_RUN},         {"repl", CMD_REPL},   {"check", CMD_CHECK},
    {"build", CMD_BUILD},     {"fmt", CMD_FMT},     {"test", CMD_TEST},
    {"doc", CMD_DOC},         {"bench", CMD_BENCH}, {"disasm", CMD_DISASM},
    {"ast", CMD_AST},         {"tokens", CMD_TOKENS},
    {"version", CMD_VERSION}, {"help", CMD_HELP},
};

static bool commandFromName(const char *name, JaiCommand *out) {
    for (size_t i = 0; i < sizeof kCommands / sizeof kCommands[0]; i++) {
        if (strcmp(kCommands[i].name, name) == 0) {
            *out = kCommands[i].command;
            return true;
        }
    }
    return false;
}

/* The two commands that are spelled as options are not in the table, because
 * `jaithon eval` and `jaithon bootstrap-verify` are not things to type. */
static const char *commandName(JaiCommand command) {
    for (size_t i = 0; i < sizeof kCommands / sizeof kCommands[0]; i++) {
        if (kCommands[i].command == command) return kCommands[i].name;
    }
    return command == CMD_EVAL ? "--eval" : "--bootstrap-verify";
}

/* Match `--name` or `--name=VALUE`. *outValue is NULL for the bare form. */
static bool optionIs(const char *arg, const char *name, const char **outValue) {
    size_t n = strlen(name);
    if (strncmp(arg, name, n) != 0) return false;
    if (arg[n] == '\0') { *outValue = NULL; return true; }
    if (arg[n] == '=')  { *outValue = arg + n + 1; return true; }
    return false;
}

/* The value of a two-word option; advances past it. NULL when it is missing. */
static const char *takeValue(int argc, char **argv, int *i, const char *flag) {
    if (*i + 1 >= argc) {
        cliError("option `%s` needs a value", flag);
        return NULL;
    }
    return argv[++(*i)];
}

typedef struct {
    JaiCliOptions *out;
    bool haveCommand;
    int  forced;      /* CMD_HELP or CMD_VERSION once seen, else -1 */
} ParseState;

static bool parseThreads(const char *text, JaiCliOptions *out) {
    char *end = NULL;
    long value = strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 1 || value > 4096) {
        cliError("--threads expects a count between 1 and 4096, got `%s`", text);
        return false;
    }
    out->threads = (int)value;
    return true;
}

static bool parseEmit(const char *text, ParseState *st) {
    if (strcmp(text, "ast") == 0) {
        st->out->command = CMD_AST;
        st->out->jsonOutput = true;   /* spec §11: --emit=ast is JSON */
    } else if (strcmp(text, "bc") == 0) {
        st->out->command = CMD_DISASM;
    } else if (strcmp(text, "tokens") == 0) {
        st->out->command = CMD_TOKENS;
    } else {
        cliError("--emit expects `ast`, `bc` or `tokens`, got `%s`", text);
        return false;
    }
    st->haveCommand = true;
    return true;
}

/* Commands whose implementation lives in lib/jaithon/tool and owns its own flags. */
static bool commandIsTool(JaiCommand command) {
    switch (command) {
    case CMD_FMT:
    case CMD_TEST:
    case CMD_DOC:
    case CMD_BENCH:
        return true;
    default:
        return false;
    }
}

/* Parse one option. `*i` indexes the option and is advanced past its value. */
static bool parseOption(int argc, char **argv, int *i, ParseState *st) {
    JaiCliOptions *out = st->out;
    const char *arg = argv[*i];
    const char *value = NULL;

    if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
        st->forced = CMD_HELP;
        return true;
    }
    /* `-v` after a tool command is the tool's own short option — every one of
     * them reads it as `--verbose` — so it is forwarded rather than claimed
     * here. `jaithon -v`, `jaithon --version` and `jaithon version` all still
     * print the version. */
    if (strcmp(arg, "--version") == 0 ||
        (strcmp(arg, "-v") == 0 && !(st->haveCommand && commandIsTool(out->command)))) {
        st->forced = CMD_VERSION;
        return true;
    }
    if (strcmp(arg, "--bootstrap-verify") == 0) {
        out->command = CMD_BOOTSTRAP_VERIFY;
        st->haveCommand = true;
        return true;
    }

    if (arg[0] == '-' && arg[1] == 'O' && arg[2] != '\0' && arg[3] == '\0' &&
        arg[2] >= '0' && arg[2] <= '3') {
        out->run.codegen.optLevel = arg[2] - '0';
        return true;
    }

    if (strcmp(arg, "--release") == 0) {
        out->run.codegen.stripAsserts = true;
        out->run.codegen.debugInfo = false;
        return true;
    }
    if (strcmp(arg, "--debug-trace") == 0) { out->traceExec = true;  return true; }
    if (strcmp(arg, "--gc-stress") == 0)   { out->gcStress = true;   return true; }
    if (strcmp(arg, "--no-prelude") == 0)  { out->noPrelude = true;  return true; }
    if (strcmp(arg, "--time") == 0)        { out->showTiming = true; return true; }
    if (strcmp(arg, "--stats") == 0)       { out->showStats = true;  return true; }
    if (strcmp(arg, "--json") == 0)        { out->jsonOutput = true; return true; }
    if (strcmp(arg, "--check") == 0)       { out->fmtCheck = true;   return true; }
    if (strcmp(arg, "--no-gpu") == 0)      { out->noGpu = true;      return true; }
    if (strcmp(arg, "--strict") == 0)      { gStrict = true;         return true; }
    if (strcmp(arg, "--no-cache") == 0) {
        out->run.useCache = false;
        out->run.writeCache = false;
        return true;
    }

    if (optionIs(arg, "--threads", &value)) {
        if (value == NULL) value = takeValue(argc, argv, i, "--threads");
        return value != NULL && parseThreads(value, out);
    }
    if (optionIs(arg, "--front", &value)) {
        if (value == NULL) value = takeValue(argc, argv, i, "--front");
        if (value == NULL) return false;
        if (strcmp(value, "c") == 0) {
            out->run.selfHosted = false;
        } else if (strcmp(value, "jai") == 0) {
            out->run.selfHosted = true;
        } else {
            cliError("--front expects `c` or `jai`, got `%s`", value);
            return false;
        }
        return true;
    }
    if (optionIs(arg, "--emit", &value)) {
        if (value == NULL) value = takeValue(argc, argv, i, "--emit");
        return value != NULL && parseEmit(value, st);
    }
    /* --eval names what to run rather than where to find it, so it is the
     * command as much as `run` is, and claiming it here is what makes
     * `jaithon --eval EXPR file.jai` say that the file has no place on that
     * command line instead of quietly running the file. */
    if (optionIs(arg, "--eval", &value)) {
        if (value == NULL) value = takeValue(argc, argv, i, "--eval");
        if (value == NULL) return false;
        if (*value == '\0') {
            cliError("--eval expects an expression");
            return false;
        }
        out->eval = value;
        out->command = CMD_EVAL;
        st->haveCommand = true;
        return true;
    }
    if (optionIs(arg, "--color", &value)) {
        if (value == NULL) value = takeValue(argc, argv, i, "--color");
        if (value == NULL) return false;
        if (strcmp(value, "auto") == 0)        gColorMode = COLOR_AUTO;
        else if (strcmp(value, "always") == 0) gColorMode = COLOR_ALWAYS;
        else if (strcmp(value, "never") == 0)  gColorMode = COLOR_NEVER;
        else {
            cliError("--color expects `auto`, `always` or `never`, got `%s`", value);
            return false;
        }
        return true;
    }
    if (optionIs(arg, "--out", &value)) {
        if (value == NULL) value = takeValue(argc, argv, i, "--out");
        if (value == NULL) return false;
        out->output = value;
        return true;
    }
    if (optionIs(arg, "--dump-sema", &value)) {
        if (value == NULL) value = takeValue(argc, argv, i, "--dump-sema");
        if (value == NULL) return false;
        out->dumpSema = value;
        return true;
    }

    if (arg[0] == '-' && arg[1] == 'I') {
        const char *dir = arg[2] != '\0' ? arg + 2 : takeValue(argc, argv, i, "-I");
        if (dir == NULL) return false;
        if (gIncludeCount < gArgSlots) gIncludeDirs[gIncludeCount++] = dir;
        return true;
    }

    /* fmt, test, doc and bench are written in Jaithon and parse their own
     * flags. Hand anything unrecognised straight through, unchanged and in
     * place, rather than teaching the C driver a second copy of each tool's
     * option table. The token after a value-taking flag is a positional as far
     * as this parser can tell, so it is appended to the same list in the same
     * order and the tool — which does know its option table — pairs them up. */
    if (st->haveCommand && commandIsTool(out->command)) {
        if (out->toolArgCount < gArgSlots) out->toolArgs[out->toolArgCount++] = arg;
        return true;
    }

    cliError("unknown option `%s`", arg);
    return false;
}

/* Commands that consume their positional arguments as source paths. */
static bool commandTakesInputs(JaiCommand command) {
    switch (command) {
    case CMD_REPL:
    case CMD_EVAL:
    case CMD_VERSION:
    case CMD_HELP:
        return false;
    default:
        return true;
    }
}

static bool commandTakesOutput(JaiCommand command) {
    switch (command) {
    case CMD_BUILD:
    case CMD_DOC:
    case CMD_AST:
    case CMD_DISASM:
    case CMD_TOKENS:
        return true;
    default:
        return false;
    }
}

/* Commands for which "no paths given" is an error rather than "use the
 * current directory". */
static bool commandNeedsInput(JaiCommand command) {
    switch (command) {
    case CMD_RUN:
    case CMD_CHECK:
    case CMD_BUILD:
    case CMD_DISASM:
    case CMD_AST:
    case CMD_TOKENS:
        return true;
    default:
        return false;
    }
}

bool jaiCliParse(int argc, char **argv, JaiCliOptions *out) {
    if (out == NULL) return false;

    memset(out, 0, sizeof *out);
    out->run = jaiRunDefaults();
    out->command = CMD_REPL;

    gArgSlots = argc > 0 ? argc : 1;
    out->inputs = JAI_ALLOC(const char *, gArgSlots);
    out->toolArgs = JAI_ALLOC(const char *, gArgSlots);
    out->scriptArgv = JAI_ALLOC(char *, gArgSlots);
    gIncludeDirs = JAI_ALLOC(const char *, gArgSlots);
    gIncludeCount = 0;

    ParseState st = {out, false, -1};
    bool endOfOptions = false;

    for (int i = 1; i < argc; i++) {
        char *arg = argv[i];

        if (endOfOptions) {
            out->scriptArgv[out->scriptArgc++] = arg;
            continue;
        }
        if (strcmp(arg, "--") == 0) {
            endOfOptions = true;
            continue;
        }
        if (arg[0] == '-' && arg[1] != '\0') {
            if (!parseOption(argc, argv, &i, &st)) {
                (void)cliFlush();
                jaiCliPrintUsage(stderr);
                return false;
            }
            continue;
        }

        if (!st.haveCommand) {
            JaiCommand named;
            st.haveCommand = true;
            if (commandFromName(arg, &named)) {
                out->command = named;
                continue;
            }
            /* `jaithon file.jai` is shorthand for `jaithon run file.jai`. */
            out->command = CMD_RUN;
            out->inputs[out->inputCount++] = arg;
            continue;
        }

        /* Everything after the script name belongs to the script. */
        if (out->command == CMD_RUN && out->inputCount >= 1) {
            out->scriptArgv[out->scriptArgc++] = arg;
            continue;
        }
        out->inputs[out->inputCount++] = arg;
        /* A tool's paths also go in the forwarding list, in place, so that the
         * tool sees the command line the user actually wrote. */
        if (commandIsTool(out->command) && out->toolArgCount < gArgSlots) {
            out->toolArgs[out->toolArgCount++] = arg;
        }
    }

    if (st.forced >= 0) out->command = (JaiCommand)st.forced;

    if (out->inputCount > 0 && !commandTakesInputs(out->command)) {
        cliError("`%s` takes no arguments, but `%s` was given",
                 commandName(out->command), out->inputs[0]);
        (void)cliFlush();
        jaiCliPrintUsage(stderr);
        return false;
    }
    if (out->inputCount == 0 && commandNeedsInput(out->command)) {
        cliError("`%s` needs at least one source file", commandName(out->command));
        (void)cliFlush();
        jaiCliPrintUsage(stderr);
        return false;
    }
    if (out->output != NULL && !commandTakesOutput(out->command)) {
        cliError("--out has no meaning for `%s`", commandName(out->command));
        (void)cliFlush();
        return false;
    }

    out->run.entryPath = out->inputCount > 0 ? out->inputs[0] : NULL;
    out->run.checkOnly = out->command == CMD_CHECK;
    return true;
}

static void cliFreeOptions(JaiCliOptions *opts) {
    if (opts == NULL) return;
    JAI_FREE_ARRAY(const char *, opts->inputs, gArgSlots);
    JAI_FREE_ARRAY(const char *, opts->toolArgs, gArgSlots);
    JAI_FREE_ARRAY(char *, opts->scriptArgv, gArgSlots);
    JAI_FREE_ARRAY(const char *, gIncludeDirs, gArgSlots);
    opts->inputs = NULL;
    opts->toolArgs = NULL;
    opts->scriptArgv = NULL;
    gIncludeDirs = NULL;
    gIncludeCount = 0;
}

/* ------------------------------------------------------------------ */
/* Path collection                                                      */
/* ------------------------------------------------------------------ */

typedef JAI_VEC(char *) PathList;

static void pathListFree(PathList *list) {
    for (int i = 0; i < list->count; i++) {
        char *p = list->data[i];
        if (p != NULL) JAI_FREE_ARRAY(char, p, strlen(p) + 1);
    }
    JAI_VEC_FREE(char *, list);
}

static void freeNameList(char **names, int count) {
    if (names == NULL) return;
    for (int i = 0; i < count; i++) {
        if (names[i] != NULL) JAI_FREE_ARRAY(char, names[i], strlen(names[i]) + 1);
    }
    JAI_FREE_ARRAY(char *, names, count + 1);
}

static int compareNames(const void *a, const void *b) {
    return strcmp(*(char *const *)a, *(char *const *)b);
}

static bool hasJaiExtension(const char *name) {
    size_t len = strlen(name);
    return len > 4 && strcmp(name + len - 4, ".jai") == 0;
}

/* Append `path` if it is a file, or every .jai file beneath it if it is a
 * directory. Directory order is sorted so that output is reproducible. */
static bool collectSources(const char *path, PathList *out) {
    if (!jaiPathExists(path)) {
        cliError("no such file or directory: %s", path);
        return false;
    }
    if (!jaiPathIsDir(path)) {
        JAI_VEC_PUSH(char *, out, jaiStrdup(path));
        return true;
    }

    int count = 0;
    char **names = jaiListDir(path, &count);
    if (names == NULL) {
        cliError("cannot read directory: %s", path);
        return false;
    }
    if (count > 1) qsort(names, (size_t)count, sizeof(char *), compareNames);

    bool ok = true;
    for (int i = 0; i < count; i++) {
        const char *name = names[i];
        if (name == NULL || name[0] == '.') continue;
        if (strcmp(name, "__jaicache__") == 0) continue;

        char child[JAI_MAX_PATH];
        jaiPathJoin(child, sizeof child, path, name);
        if (jaiPathIsDir(child)) {
            if (!collectSources(child, out)) { ok = false; break; }
        } else if (hasJaiExtension(name)) {
            JAI_VEC_PUSH(char *, out, jaiStrdup(child));
        }
    }

    freeNameList(names, count);
    return ok;
}

static bool collectAllInputs(const JaiCliOptions *opts, PathList *out,
                             const char *fallback) {
    JAI_VEC_INIT(out);
    if (opts->inputCount == 0 && fallback != NULL) {
        return collectSources(fallback, out);
    }
    for (int i = 0; i < opts->inputCount; i++) {
        if (!collectSources(opts->inputs[i], out)) return false;
    }
    return true;
}

/* ------------------------------------------------------------------ */
/* Compiling one file with the C front end                              */
/* ------------------------------------------------------------------ */

/* jaiCompileSource takes a const buffer, so it registers its own copy with the
 * diagnostic engine. Should a front end ever hand this exact buffer to
 * jaiSourceAdd instead, freeing it here would leave the registry dangling — so
 * check before letting go of it. */
static void releaseSource(char *source, size_t length, const ObjModule *module) {
    if (module != NULL) {
        const JaiSourceFile *file = jaiSourceGet(module->sourceFileId);
        if (file != NULL && file->source == source) return;
    }
    JAI_FREE_ARRAY(char, source, length + 1);
}

/* Compile `path` into a module body. On success the module and the body are
 * both left on the GC root stack (module pushed first) and the caller pops two
 * roots once it is done with them. On failure nothing is rooted and the
 * diagnostics have already been rendered. */
static ObjFunction *compileFile(const char *path, const CodegenOptions *codegen,
                                ObjModule **outModule) {
    size_t length = 0;
    char *source = jaiReadFile(path, &length);
    if (source == NULL) {
        cliError("cannot read %s", path);
        (void)cliFlush();
        return NULL;
    }

    char name[256];
    char absolute[JAI_MAX_PATH];
    jaiModuleNameFor(path, name, sizeof name);
    if (!jaiPathAbsolute(absolute, sizeof absolute, path)) {
        snprintf(absolute, sizeof absolute, "%s", path);
    }

    ObjModule *module = jaiModuleNew(jaiStringInternC(name),
                                     jaiStringInternC(absolute));
    jaiPushRoot(OBJ_VAL(module));

    /* Which front end. `--front=jai` has to reach every command that compiles,
     * or `build` and `disasm` would quietly keep using the C one and the flag
     * would be a lie on two of the commands most likely to be pointed at it. */
    ObjFunction *body;
    if (gSelfHosted) {
        /* Register before compiling. `jaiCompileSource` does this on the C
         * path and the bridge does not, so without it the module keeps file id
         * zero and every span the self-hosted front end emits points at a file
         * nothing registered — a build with no line numbers. */
        uint64_t hash = jaiSourceHash(source, length);
        int fileId = jaiSourceAdd(absolute, source, length); /* takes source */
        module->sourceFileId = fileId;
        const JaiSourceFile *registered = jaiSourceGet(fileId);
        body = jaiSelfHostedCompileInto(
            registered != NULL ? registered->source : source,
            registered != NULL ? registered->length : length,
            absolute, module, hash, codegen->optLevel);
    } else {
        body = jaiCompileSource(source, length, absolute, module, codegen);
        releaseSource(source, length, module);
    }

    bool hadErrors = cliFlush();
    if (body == NULL || hadErrors) {
        jaiPopRoot();
        if (body == NULL && !hadErrors) {
            cliError("%s: compilation failed", path);
            (void)cliFlush();
        }
        return NULL;
    }

    jaiPushRoot(OBJ_VAL(body));
    if (outModule != NULL) *outModule = module;
    return body;
}

/* ------------------------------------------------------------------ */
/* run / check                                                          */
/* ------------------------------------------------------------------ */

static int cmdRun(const JaiCliOptions *opts) {
    return jaiRunFile(opts->inputs[0], &opts->run, opts->scriptArgc,
                      opts->scriptArgv);
}

static int cmdCheck(const JaiCliOptions *opts) {
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
static int cmdEval(const JaiCliOptions *opts) {
    bool incomplete = false;
    jaiReplConfigure(opts);

    const char *cursor = opts->eval;
    while (*cursor != '\0') {
        const char *end = strchr(cursor, '\n');
        size_t length = end != NULL ? (size_t)(end - cursor) : strlen(cursor);
        char *line = JAI_ALLOC(char, length + 1);
        memcpy(line, cursor, length);
        line[length] = '\0';
        (void)jaiReplFeed(line, &incomplete);
        JAI_FREE_ARRAY(char, line, length + 1);
        if (end == NULL) break;
        cursor = end + 1;
    }
    /* The REPL flushes once per input for the same reason: diagnostics leave
     * as they are made, and stdout to a pipe would otherwise arrive after. */
    fflush(stdout);

    /* Only the last line's verdict: an earlier one that was incomplete is a
     * continuation, which is the whole reason the loop above exists. */
    if (incomplete) {
        cliError("--eval expects a complete expression, but `%s` is the start "
                 "of one", opts->eval);
        (void)cliFlush();
        return 1;
    }
    return jaiReplFailed() ? 1 : 0;
}

/* ------------------------------------------------------------------ */
/* build                                                                */
/* ------------------------------------------------------------------ */

static int buildOne(const char *path, const JaiCliOptions *opts) {
    ObjModule *module = NULL;
    ObjFunction *body = compileFile(path, &opts->run.codegen, &module);
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

    size_t sourceLength = 0;
    char *source = jaiReadFile(path, &sourceLength);
    uint64_t hash = source != NULL ? jaiSourceHash(source, sourceLength) : 0;
    if (source != NULL) JAI_FREE_ARRAY(char, source, sourceLength + 1);

    uint32_t flags = 0;
    if (opts->run.codegen.debugInfo) flags |= JAIC_FLAG_DEBUG;
    if (opts->run.codegen.stripAsserts) flags |= JAIC_FLAG_RELEASE;

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

static int cmdBuild(const JaiCliOptions *opts) {
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

/* ------------------------------------------------------------------ */
/* disasm / ast / tokens                                                */
/* ------------------------------------------------------------------ */

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
static bool imageIsJaic(const uint8_t *data, size_t size) {
    return size >= 4 && memcmp(data, JAIC_MAGIC, 4) == 0;
}

static uint16_t imageU16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }

static uint32_t imageU32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16)
         | ((uint32_t)p[3] << 24);
}

static uint64_t imageU64(const uint8_t *p) {
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
    uint16_t version  = imageU16(data + 4);
    uint16_t flags    = imageU16(data + 6);
    uint32_t compiler = imageU32(data + 8);
    uint32_t buildId  = imageU32(data + 12);
    uint64_t srcHash  = imageU64(data + 16);
    uint16_t pathLen  = imageU16(data + 24);

    fprintf(out, "; %s\n", path);
    fprintf(out, "; container version %u", version);
    if (version != JAIC_VERSION) fprintf(out, "  (this build expects %d)", JAIC_VERSION);
    fputc('\n', out);
    fprintf(out, "; compiler version  %u\n", compiler);
    fprintf(out, "; build id          0x%08x%s\n", buildId,
            buildId == jaiBuildId() ? "  (this build)" : "  (a different build)");
    fprintf(out, "; source hash       0x%016llx\n", (unsigned long long)srcHash);
    fprintf(out, "; flags             %s%s\n",
            (flags & JAIC_FLAG_DEBUG) ? "debug " : "",
            (flags & JAIC_FLAG_RELEASE) ? "release" : "");
    if (pathLen > 0 && (size_t)26 + pathLen <= size)
        fprintf(out, "; source            %.*s\n", (int)pathLen, (const char *)data + 26);
    fputc('\n', out);

    ObjModule *module = jaiModuleNew(jaiStringInternC(path), jaiStringInternC(path));
    if (module == NULL) { cliError("%s: out of memory", path); return false; }
    jaiPushRoot(OBJ_VAL(module));

    /* Pass the image's own source hash so the check is a no-op: the point here
     * is to look at what the file contains, not to decide whether it is a
     * usable cache entry for some source file. */
    ObjFunction *body = jaiDeserializeModule(data, size, module, srcHash);
    if (body == NULL) {
        jaiPopRoot();
        cliError("%s: this build cannot load the image", path);
        if (compiler != JAI_COMPILER_VERSION || buildId != jaiBuildId())
            cliError("it was written by a different compiler build; recompile "
                     "the source to inspect it");
        return false;
    }
    jaiPushRoot(OBJ_VAL(body));
    disassembleTree(out, body, 0);
    jaiPopRoots(2);
    return true;
}

static int cmdDisasm(const JaiCliOptions *opts) {
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
        if (image != NULL) JAI_FREE_ARRAY(char, image, imageSize + 1);

        ObjModule *module = NULL;
        ObjFunction *body = compileFile(files.data[i], &opts->run.codegen,
                                        &module);
        if (body == NULL) { status = 1; continue; }
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
static int cmdParseOnly(const JaiCliOptions *opts, bool tokensOnly) {
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
            Lexer lex;
            jaiLexerInit(&lex, source, length, fileId);
            if (!jaiLexerRun(&lex)) status = 1;
            if (files.count > 1) fprintf(out, "; %s\n", path);
            jaiLexerDump(out, &lex);
            jaiLexerFree(&lex);
            if (cliFlush()) status = 1;
            continue;
        }

        AstContext ast;
        Lexer lex;
        jaiAstContextInit(&ast);
        AstNode *program = jaiParseSource(&ast, &lex, source, length, fileId);
        if (cliFlush() || program == NULL) {
            status = 1;
        } else if (opts->jsonOutput) {
            jaiAstToJson(out, program);
            fputc('\n', out);
        } else {
            if (files.count > 1) fprintf(out, "; %s\n", path);
            jaiAstPrint(out, program, 0);
        }
        jaiLexerFree(&lex);
        jaiAstContextFree(&ast);
    }

    pathListFree(&files);
    if (opened != NULL && fclose(opened) != 0) {
        cliError("cannot write %s", opts->output);
        (void)cliFlush();
        status = 1;
    }
    return status;
}

/* ------------------------------------------------------------------ */
/* fmt / test / doc / bench — implemented in Jaithon                    */
/* ------------------------------------------------------------------ */

static void toolPush(ObjList *args, const char *text) {
    Value v = OBJ_VAL(jaiStringInternC(text));
    jaiPushRoot(v);
    jaiListPush(args, v);
    jaiPopRoot();
}

/* Build the list[str] handed to a tool's main().
 *
 * The rule is that the tool sees the command line the user wrote. The few
 * flags the CLI understands for itself and still has to pass on are re-emitted
 * in exactly the spelling the tool's usage line advertises — never folded into
 * a `--name=value` word the tool would then have to guess at — and everything
 * else is forwarded verbatim, in order. No path is invented either: `jaithon
 * doc` with no paths must reach `jaithon.tool.doc` with no paths, so that the
 * tool's own documented default (`lib` for doc, `tests` for test, `tests/bench`
 * for bench) is what applies, rather than a `.` from here that would send bench
 * walking the whole tree. */
static ObjList *toolArguments(const JaiCliOptions *opts) {
    ObjList *args = jaiListNew(opts->toolArgCount + 4);
    jaiPushRoot(OBJ_VAL(args));

    /* `--check` and `--json` are spelled the same on both sides; a tool that
     * does not know one says so rather than having it silently dropped. */
    if (opts->fmtCheck)   toolPush(args, "--check");
    if (opts->jsonOutput) toolPush(args, "--json");
    if (opts->output != NULL) {
        toolPush(args, "--out");
        toolPush(args, opts->output);
    }
    for (int i = 0; i < opts->toolArgCount; i++) toolPush(args, opts->toolArgs[i]);
    return args;   /* still rooted; the caller pops */
}

/* True when the tool's `main` should be called with no arguments at all. */
static bool toolMainTakesNoArgs(Value entry) {
    const ObjFunction *fn = NULL;
    if (IS_CLOSURE(entry)) fn = AS_CLOSURE(entry)->fn;
    else if (IS_FUNCTION(entry)) fn = AS_FUNCTION(entry);
    else if (IS_NATIVE(entry)) return AS_NATIVE(entry)->maxArity == 0;
    if (fn == NULL) return false;
    return fn->arity == 0 && fn->paramCount == 0 &&
           (fn->flags & (FN_VARIADIC | FN_KWREST)) == 0;
}

static int runJaithonTool(const JaiCliOptions *opts, const char *tool) {
    char moduleName[64];
    snprintf(moduleName, sizeof moduleName, "jaithon.tool.%s", tool);

    ObjModule *module = jaiImportModule(moduleName, NULL);
    if (module == NULL) {
        jaiClearException();
        JaiDiag *d = jaiDiagError(E0800_MODULE_NOT_FOUND, JAI_SPAN_NONE,
                                  "`%s` is written in Jaithon and needs the "
                                  "standard library module `%s`",
                                  tool, moduleName);
        jaiDiagAddHelp(d, "install the standard library, or point JAITHON_PATH "
                          "at the directory that contains `std`");
        (void)cliFlush();
        return 1;
    }
    jaiPushRoot(OBJ_VAL(module));

    Value entry;
    if (!jaiModuleGet(module, jaiStringInternC("main"), &entry)) {
        (void)jaiDiagError(E0802_NOT_EXPORTED, JAI_SPAN_NONE,
                           "module `%s` does not define `main`", moduleName);
        (void)cliFlush();
        jaiPopRoot();
        return 1;
    }

    bool ok;
    Value result = NULL_VAL;
    if (toolMainTakesNoArgs(entry)) {
        ok = jaiCallValue(entry, 0, NULL, &result);
    } else {
        ObjList *args = toolArguments(opts);
        Value argument = OBJ_VAL(args);
        ok = jaiCallValue(entry, 1, &argument, &result);
        jaiPopRoot();
    }
    jaiPopRoot();

    if (!ok) {
        jaiReportUncaught(vm.pendingException);
        jaiClearException();
        return 1;
    }
    if (IS_INT(result)) {
        int64_t code = AS_INT(result);
        return (int)(code & 0xff);
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* --bootstrap-verify                                                   */
/* ------------------------------------------------------------------ */

/* The self-hosted front end is reached through jaiSelfHostedCompileInto, the
 * same bridge `--front=jai` runs on. Deliberately the same: a bootstrap run
 * then exercises the path a user takes, and neither can rot without the other
 * noticing. What comes back is a module body comparable opcode for opcode
 * against what src/codegen produced, or NULL with the reason already stated. */
static ObjFunction *compileSelfHosted(const char *path, const char *source,
                                      size_t length, ObjModule *into,
                                      uint64_t hash, int optLevel) {
    ObjFunction *body = jaiSelfHostedCompileInto(source, length, path, into,
                                                 hash, optLevel);
    if (body == NULL) (void)cliFlush();
    return body;
}

/* Spec §11 compares *serialised* function records, and only one of the two
 * sides had been through the container: the self-hosted body arrives as a
 * .jaic, the C body never left memory. That round trip is not the identity.
 * writeFunction appends a record's own name and its parameter names to the
 * constant pool (§5), so a record read back has entries the same record in
 * memory does not, and comparing across the boundary reported a pool-size
 * difference on every named function with a pool — a fact about serialize.c,
 * not about either front end. Both sides go through the same door.
 *
 * Returns NULL only when the C front end's own output will not round-trip,
 * which is a serialize.c bug worth failing on rather than routing around. */
static ObjFunction *roundTrip(ObjModule *module, ObjFunction *body,
                              uint64_t hash) {
    size_t size = 0;
    uint8_t *image = jaiSerializeModule(module, body, hash, JAIC_FLAG_DEBUG,
                                        &size);
    if (image == NULL) return NULL;

    ObjModule *fresh = jaiModuleNew(module->name, module->path);
    jaiPushRoot(OBJ_VAL(fresh));
    fresh->sourceFileId = module->sourceFileId;
    ObjFunction *reread = jaiDeserializeModule(image, size, fresh, hash);
    jaiPopRoot();

    (void)jaiRealloc(image, size, 0);
    return reread;
}

/* What one file's comparison produced. `VERIFY_UNCOMPARABLE` is not a pass:
 * tests/errors holds files the C front end is *supposed* to reject, and a file
 * the reference will not compile yields no bytecode to compare against. Those
 * were being counted as divergences, which pinned `make bootstrap` at 31
 * failures it could never fix and hid the real number underneath them. They
 * are reported and counted on their own line instead. */
typedef enum {
    VERIFY_SAME,
    VERIFY_DIFFERENT,
    VERIFY_UNCOMPARABLE,
} VerifyOutcome;

static VerifyOutcome verifyOne(const char *path, const CodegenOptions *codegen) {
    size_t length = 0;
    char *source = jaiReadFile(path, &length);
    if (source == NULL) {
        cliError("cannot read %s", path);
        (void)cliFlush();
        return VERIFY_DIFFERENT;
    }
    uint64_t hash = jaiSourceHash(source, length);

    ObjModule *cModule = NULL;
    ObjFunction *cBody = compileFile(path, codegen, &cModule);
    if (cBody == NULL) {
        printf("skip %s\n       the C front end rejects this file, so there is "
               "no bytecode to compare against\n", path);
        JAI_FREE_ARRAY(char, source, length + 1);
        return VERIFY_UNCOMPARABLE;
    }

    char name[256];
    char absolute[JAI_MAX_PATH];
    jaiModuleNameFor(path, name, sizeof name);
    if (!jaiPathAbsolute(absolute, sizeof absolute, path)) {
        snprintf(absolute, sizeof absolute, "%s", path);
    }
    ObjModule *jModule = jaiModuleNew(jaiStringInternC(name),
                                      jaiStringInternC(absolute));
    /* The C side above registered this file; reuse its id rather than leaving
     * the self-hosted module at zero. A span carrying an id `jaiSourceGet` does
     * not know reads as *unknown*, and an unknown span silently costs whatever
     * depends on it — the assert message falls back to its bare form, which is
     * a constant-pool difference this very comparison then reports. */
    jModule->sourceFileId = cModule != NULL ? cModule->sourceFileId : 0;
    jaiPushRoot(OBJ_VAL(jModule));

    VerifyOutcome status = VERIFY_SAME;
    ObjFunction *cReread = roundTrip(cModule, cBody, hash);
    if (cReread == NULL) {
        cliError("%s: the C front end's own .jaic image does not round-trip; "
                 "there is nothing to compare against", path);
        (void)cliFlush();
        jaiPopRoot();      /* jModule */
        jaiPopRoots(2);    /* cModule, cBody */
        JAI_FREE_ARRAY(char, source, length + 1);
        return VERIFY_DIFFERENT;
    }
    jaiPushRoot(OBJ_VAL(cReread));

    /* Both sides at the same level, or the report blames the front ends for
     * the flag: see the note in selfHostedImage. */
    ObjFunction *jBody = compileSelfHosted(absolute, source, length, jModule,
                                           hash, codegen->optLevel);
    if (jBody == NULL) {
        status = VERIFY_DIFFERENT;
    } else {
        jaiPushRoot(OBJ_VAL(jBody));
        char difference[512];
        if (jaiCompareFunctions(cReread, jBody, difference, sizeof difference)) {
            printf("ok   %s\n", path);
        } else {
            printf("DIFF %s\n       %s\n", path, difference);
            status = VERIFY_DIFFERENT;
        }
        jaiPopRoot();
    }

    jaiPopRoot();          /* cReread */
    jaiPopRoot();          /* jModule */
    jaiPopRoots(2);        /* cModule, cBody */
    JAI_FREE_ARRAY(char, source, length + 1);
    return status;
}

/* A differential check that compared nothing must not report success: `make
 * bootstrap` is the only thing standing between "the two front ends agree" and
 * "nobody has ever run the second one", and a green line it did not earn is
 * worse than no check at all. Every early exit below is a failure, including
 * the one where the path list came out empty. */
static int cmdBootstrapVerify(const JaiCliOptions *opts) {
    ObjModule *compiler = jaiImportModule("jaithon.compile", NULL);
    if (compiler == NULL) {
        jaiClearException();
        JaiDiag *d = jaiDiagError(E0800_MODULE_NOT_FOUND, JAI_SPAN_NONE,
                                  "--bootstrap-verify needs the self-hosted "
                                  "front end in `jaithon.compile`");
        jaiDiagAddHelp(d, "install the standard library, or point JAITHON_PATH "
                          "at the directory that contains `std`");
        (void)cliFlush();
        return 1;
    }
    jaiPushRoot(OBJ_VAL(compiler));

    Value entry;
    if (!jaiModuleGet(compiler, jaiStringInternC("compile_source"), &entry)) {
        (void)jaiDiagError(E0802_NOT_EXPORTED, JAI_SPAN_NONE,
                           "`jaithon.compile` does not export "
                           "`compile_source(source, path)`");
        (void)cliFlush();
        jaiPopRoot();
        return 1;
    }
    /* Looked up only to fail here, before a single file is compiled, rather
     * than once per file inside jaiSelfHostedCompile. */
    (void)entry;

    PathList files;
    if (!collectAllInputs(opts, &files, ".")) {
        (void)cliFlush();
        pathListFree(&files);
        jaiPopRoot();
        return 1;
    }
    if (files.count == 0) {
        JaiDiag *d = jaiDiagError(E0800_MODULE_NOT_FOUND, JAI_SPAN_NONE,
                                  "--bootstrap-verify found no .jai files to "
                                  "compare the two front ends on");
        jaiDiagAddHelp(d, "name a file or a directory that contains one");
        (void)cliFlush();
        pathListFree(&files);
        jaiPopRoot();
        return 1;
    }

    int differing = 0, skipped = 0;
    for (int i = 0; i < files.count; i++) {
        switch (verifyOne(files.data[i], &opts->run.codegen)) {
        case VERIFY_DIFFERENT:   differing++; break;
        case VERIFY_UNCOMPARABLE: skipped++;  break;
        case VERIFY_SAME:                     break;
        }
    }
    int compared = files.count - skipped;
    printf("%d file%s checked, %d compared, %d differing, %d not comparable\n",
           files.count, files.count == 1 ? "" : "s", compared, differing,
           skipped);

    pathListFree(&files);
    jaiPopRoot();
    /* Comparing nothing is a failure however few files diverged: see the
     * comment above cmdBootstrapVerify. */
    if (compared == 0) {
        JaiDiag *d = jaiDiagError(E0902_INTERNAL_ERROR, JAI_SPAN_NONE,
                                  "--bootstrap-verify compared no files: the C "
                                  "front end rejected every input");
        jaiDiagAddHelp(d, "name a file the reference front end accepts");
        (void)cliFlush();
        return 1;
    }
    return differing == 0 ? 0 : 1;
}

/* ------------------------------------------------------------------ */
/* Dispatch                                                             */
/* ------------------------------------------------------------------ */

/* The prelude is what makes unqualified names resolve (spec §9), so every
 * command that compiles or runs Jaithon needs it. `tokens` only lexes, and
 * `help`/`version` touch no source at all. */
static bool commandNeedsPrelude(JaiCommand command) {
    switch (command) {
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
static bool preludeDisabled(const JaiCliOptions *opts) {
    if (opts->noPrelude) return true;
    const char *flag = getenv("JAITHON_NO_PRELUDE");
    return flag != NULL && flag[0] != '\0' && strcmp(flag, "0") != 0;
}

/* --front=jai chooses which compiler produces the bytecode, so a command that
 * cannot honour it has to say so. Only `run` goes through jaiRunFile, which is
 * where the bridge to lib/jaithon/compile lives; --bootstrap-verify uses both front
 * ends by definition and is not affected by the flag either way. Quietly using
 * the C front end for the rest would mean the output of `jaithon --front=jai
 * build x.jai` was not what its command line says it is. */
static bool commandHonoursFrontEnd(JaiCommand command) {
    switch (command) {
    case CMD_RUN:
    case CMD_BOOTSTRAP_VERIFY:
    /* `check` runs the self-hosted front end over the entry file so that the
     * two checkers can be compared on a file rather than on bytecode, which is
     * what `make sema-diff` needs to have two sides at all. */
    case CMD_CHECK:
    /* Neither compiles anything, so the flag is vacuous rather than ignored. */
    case CMD_VERSION:
    case CMD_HELP:
        return true;
    /* Both compile, and both now go through the front end the flag names. */
    case CMD_BUILD:
    case CMD_DISASM:
        return true;
    /* These dispatch to Jaithon programs (jaithon.tool.*) rather than compiling
     * anything themselves, and their imports now route through the front end
     * the flag names. */
    case CMD_FMT:
    case CMD_TEST:
    case CMD_DOC:
    case CMD_BENCH:
        return true;
    /* The REPL and `eval` compile a snippet through the same module machinery,
     * so they follow the flag like `run` does. */
    case CMD_REPL:
    case CMD_EVAL:
        return true;
    /* These two print the C front end's own tree and token stream. There is
     * nothing for the flag to select until the self-hosted side can print the
     * same, so they stay on C and say so rather than pretending. */
    case CMD_AST:
    case CMD_TOKENS:
        return false;
    }
    return false;
}

int jaiCliDispatch(const JaiCliOptions *opts) {
    if (opts == NULL) return 1;

    gSelfHosted = opts->run.selfHosted;

    /* Only `check` writes a dump. Accepting the flag elsewhere and quietly
     * writing nothing would let a comparison "pass" because one side never
     * produced a file to compare. */
    if (opts->dumpSema != NULL) {
        if (opts->command != CMD_CHECK) {
            JaiDiag *d = jaiDiagError(JAI_OK, JAI_SPAN_NONE,
                                      "--dump-sema is only implemented for "
                                      "`check`, not `%s`",
                                      commandName(opts->command));
            jaiDiagAddHelp(d, "run `jaithon check --dump-sema PATH FILE`");
            (void)cliFlush();
            return 1;
        }
        jaiSemaDumpArm(opts->dumpSema);
    }

    if (opts->run.selfHosted && !commandHonoursFrontEnd(opts->command)) {
        JaiDiag *d = jaiDiagError(JAI_OK, JAI_SPAN_NONE,
                                  "--front=jai is not implemented for `%s`; "
                                  "only `run` uses the self-hosted front end",
                                  commandName(opts->command));
        jaiDiagAddHelp(d, "drop --front=jai to use the C front end, or run "
                          "`make bootstrap` to compare the two");
        (void)cliFlush();
        return 1;
    }

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
    case CMD_BOOTSTRAP_VERIFY: return cmdBootstrapVerify(opts);
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

    double started = jaiClockMonotonic();
    int status = jaiCliDispatch(&opts);
    double elapsed = jaiClockMonotonic() - started;

    /* Ordered after whatever the program printed, even when stdout is a pipe
     * and therefore block-buffered. */
    if (opts.showTiming || opts.showStats) fflush(stdout);
    if (opts.showTiming) fprintf(stderr, "time: %.3f s\n", elapsed);
    if (opts.showStats)  jaiVMPrintStats(stderr);

    cliFreeOptions(&opts);
    jaiVMFree();
    jaiTypesFree();
    /* After the type universe: an imported declaration's type is interned under
     * a name that lives in the signature's own arena. */
    jaiModuleSigFreeAll();
    jaiDiagFree(&gDiags);
    jaiSourceFreeAll();

    fflush(stdout);
    return status;
}
