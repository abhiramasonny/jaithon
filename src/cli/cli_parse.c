/* cli_parse.c — turning argv into a JaiCliOptions.
 *
 * This is one cohesive little recursive-descent parser: commandFromName,
 * parseOption and jaiCliParse all thread the same ParseState through option
 * after option, and commandTakesInputs/commandTakesOutput/commandNeedsInput
 * exist only to answer questions jaiCliParse asks once it reaches the end of
 * argv. Splitting those apart would not make any one of them easier to read,
 * so they stay together as the one section of the old main.c that really was
 * "argument parsing" and nothing else.
 *
 * The four static globals below are the exception: they are parse results
 * with nowhere else to live (JaiCliOptions has no field for them), consumed
 * by other files in this directory after jaiCliParse returns, so they are declared
 * extern in cli_internal.h.
 */
#include "cli_internal.h"

#include <stdlib.h>   /* strtol */

const char **gIncludeDirs;
int          gIncludeCount;
ColorMode    gColorMode = COLOR_AUTO;
bool         gStrict;

/* Parse-time-only storage: never read outside this file. */
static void **gArgStorage;      /* one slab backing all argv pointer arrays */
static int    gArgSlots;        /* capacity of inputs[] and scriptArgv[] */

/* ------------------------------------------------------------------ */
/* Command names                                                        */
/* ------------------------------------------------------------------ */

static bool commandFromName(const char *name, JaiCommand *out) {
    if (name == NULL || out == NULL) return false;

    /*
     * argc parsing happens once, but this is cheaper than walking thirteen
     * strcmp() calls and keeps the common shorthand/file case short.
     */
    const size_t n = strlen(name);

    switch (n) {
    case 3:
        if (name[0] == 'r' && memcmp(name, "run", 3) == 0) {
            *out = CMD_RUN;
            return true;
        }
        if (name[0] == 'f' && memcmp(name, "fmt", 3) == 0) {
            *out = CMD_FMT;
            return true;
        }
        if (name[0] == 'd' && memcmp(name, "doc", 3) == 0) {
            *out = CMD_DOC;
            return true;
        }
        if (name[0] == 'a' && memcmp(name, "ast", 3) == 0) {
            *out = CMD_AST;
            return true;
        }
        break;

    case 4:
        if (name[0] == 'r' && memcmp(name, "repl", 4) == 0) {
            *out = CMD_REPL;
            return true;
        }
        if (name[0] == 't' && memcmp(name, "test", 4) == 0) {
            *out = CMD_TEST;
            return true;
        }
        if (name[0] == 'h' && memcmp(name, "help", 4) == 0) {
            *out = CMD_HELP;
            return true;
        }
        break;

    case 5:
        if (name[0] == 'c' && memcmp(name, "check", 5) == 0) {
            *out = CMD_CHECK;
            return true;
        }
        if (name[0] == 'b' && name[1] == 'u' &&
            memcmp(name, "build", 5) == 0) {
            *out = CMD_BUILD;
            return true;
        }
        if (name[0] == 'b' && name[1] == 'e' &&
            memcmp(name, "bench", 5) == 0) {
            *out = CMD_BENCH;
            return true;
        }
        break;

    case 6:
        if (name[0] == 'd' && memcmp(name, "disasm", 6) == 0) {
            *out = CMD_DISASM;
            return true;
        }
        if (name[0] == 't' && memcmp(name, "tokens", 6) == 0) {
            *out = CMD_TOKENS;
            return true;
        }
        break;

    case 7:
        if (name[0] == 'v' && memcmp(name, "version", 7) == 0) {
            *out = CMD_VERSION;
            return true;
        }
        break;

    default:
        break;
    }

    return false;
}

/* `--eval` is spelled as an option rather than a command, because
 * `jaithon eval` is not a thing to type. */
static const char *commandName(JaiCommand command) {
    switch (command) {
    case CMD_RUN:     return "run";
    case CMD_REPL:    return "repl";
    case CMD_CHECK:   return "check";
    case CMD_BUILD:   return "build";
    case CMD_FMT:     return "fmt";
    case CMD_TEST:    return "test";
    case CMD_DOC:     return "doc";
    case CMD_BENCH:   return "bench";
    case CMD_DISASM:  return "disasm";
    case CMD_AST:     return "ast";
    case CMD_TOKENS:  return "tokens";
    case CMD_VERSION: return "version";
    case CMD_HELP:    return "help";
    case CMD_EVAL:    return "--eval";
    }
    return "?";
}

/* ------------------------------------------------------------------ */
/* Options                                                              */
/* ------------------------------------------------------------------ */

/* Match `--name` or `--name=VALUE`. *outValue is NULL for the bare form. */
static inline bool optionIs(const char *arg, const char *name, size_t n,
                            const char **outValue) {
    if (strncmp(arg, name, n) != 0) return false;

    const char tail = arg[n];
    if (tail == '\0') {
        *outValue = NULL;
        return true;
    }
    if (tail == '=') {
        *outValue = arg + n + 1;
        return true;
    }
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
static inline bool commandIsTool(JaiCommand command) {
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

    if (optionIs(arg, "--threads", 9, &value)) {
        if (value == NULL) value = takeValue(argc, argv, i, "--threads");
        return value != NULL && parseThreads(value, out);
    }
    if (optionIs(arg, "--front", 7, &value)) {
        if (value == NULL) value = takeValue(argc, argv, i, "--front");
        if (value == NULL) return false;
        /* `jai` is the only front end. `c` is still recognised so that a
         * command line carrying it gets an answer rather than "unknown
         * option": it named a compiler that no longer exists, and silently
         * running the other one would be the one outcome worse than failing. */
        if (strcmp(value, "jai") == 0) {
            out->run.selfHosted = true;
            return true;
        }
        if (strcmp(value, "c") == 0) {
            JaiDiag *d = jaiDiagError(JAI_OK, JAI_SPAN_NONE,
                                      "--front=c names the C front end, which "
                                      "no longer exists");
            jaiDiagAddNote(d, "Jaithon compiles itself; `--front=jai` is the "
                              "only front end and is the default");
            (void)cliFlush();
            return false;
        }
        cliError("--front expects `jai`, got `%s`", value);
        return false;
    }
    if (optionIs(arg, "--emit", 6, &value)) {
        if (value == NULL) value = takeValue(argc, argv, i, "--emit");
        return value != NULL && parseEmit(value, st);
    }
    /* --eval names what to run rather than where to find it, so it is the
     * command as much as `run` is, and claiming it here is what makes
     * `jaithon --eval EXPR file.jai` say that the file has no place on that
     * command line instead of quietly running the file. */
    if (optionIs(arg, "--eval", 6, &value)) {
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
    if (optionIs(arg, "--color", 7, &value)) {
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
    if (optionIs(arg, "--out", 5, &value)) {
        if (value == NULL) value = takeValue(argc, argv, i, "--out");
        if (value == NULL) return false;
        out->output = value;
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

/* ------------------------------------------------------------------ */
/* What a command accepts                                               */
/* ------------------------------------------------------------------ */

/* Commands that consume their positional arguments as source paths. */
static inline bool commandTakesInputs(JaiCommand command) {
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

static inline bool commandTakesOutput(JaiCommand command) {
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
static inline bool commandNeedsInput(JaiCommand command) {
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

/* ------------------------------------------------------------------ */
/* jaiCliParse                                                          */
/* ------------------------------------------------------------------ */

bool jaiCliParse(int argc, char **argv, JaiCliOptions *out) {
    if (out == NULL) return false;

    memset(out, 0, sizeof *out);
    out->run = jaiRunDefaults();
    out->command = CMD_REPL;

    gArgSlots = argc > 0 ? argc : 1;

    const size_t slots = (size_t)gArgSlots;
    gArgStorage = JAI_ALLOC(void *, slots * 4u);

    out->inputs = (const char **)(gArgStorage);
    out->toolArgs = (const char **)(gArgStorage + slots);
    out->scriptArgv = (char **)(gArgStorage + slots * 2u);
    gIncludeDirs = (const char **)(gArgStorage + slots * 3u);
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

void cliFreeOptions(JaiCliOptions *opts) {
    if (opts == NULL) return;

    if (gArgStorage != NULL) {
        JAI_FREE_ARRAY(void *, gArgStorage, (size_t)gArgSlots * 4u);
    }

    opts->inputs = NULL;
    opts->toolArgs = NULL;
    opts->scriptArgv = NULL;
    gIncludeDirs = NULL;
    gArgStorage = NULL;
    gIncludeCount = 0;
    gArgSlots = 0;
}
