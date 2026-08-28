/* repl.c — the interactive read-eval-print loop.*/

#if !defined(_POSIX_C_SOURCE)
#  define _POSIX_C_SOURCE 200809L
#endif
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#  define _DARWIN_C_SOURCE
#endif

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

/* The readline headers use FILE without including <stdio.h> themselves. */
#ifdef JAI_HAVE_READLINE
#  include <readline/history.h>
#  include <readline/readline.h>
#endif

#include "cli/cli.h"

#include "runtime/modules/frontend.h"

#include "vm/bytecode/chunk.h"

#define REPL_PROMPT      ">>> "
#define REPL_CONTINUE    "... "
#define REPL_HISTORY_MAX 1000

typedef enum {
    REPL_EXEC,
    REPL_QUIET,
    REPL_TYPE,
    REPL_AST,
    REPL_DISASM,
} ReplAction;

static struct {
    ObjModule     *module;      /* kept alive by vm.modules, a GC root */
    JaiBuf         pending;
    CodegenOptions codegen;
    int            inputCount;
    bool           interactive;
    bool           strict;
    bool           quit;
    bool           failed;
} gRepl;

/* ------------------------------------------------------------------ */
/* Small helpers                                                        */
/* ------------------------------------------------------------------ */

static inline void replNoteFailure(void) { gRepl.failed = true; }

static void replError(const char *fmt, ...) JAI_PRINTF(1, 2);

static void replError(const char *fmt, ...) {
    char message[512];
    va_list args;
    va_start(args, fmt);
    vsnprintf(message, sizeof message, fmt, args);
    va_end(args);
    (void)jaiDiagError(JAI_OK, JAI_SPAN_NONE, "%s", message);
    (void)jaiDiagFlush(&gDiags, stderr);
    replNoteFailure();
}

static bool replFlushDiags(void) {
    bool hadErrors = jaiDiagHasErrors(&gDiags);
    if (hadErrors) replNoteFailure();   /* an input with an error never runs */
    bool first = true;
    for (int i = 0; i < gDiags.diags.count; i++) {
        const JaiDiag *d = &gDiags.diags.data[i];
        if (d->code == W0101_UNUSED_BINDING || d->code == W0100_UNUSED_IMPORT) continue;
        if (!first) fputc('\n', stderr);
        jaiDiagRender(d, stderr, gDiags.colorOutput);
        first = false;
    }
    if (!first) fflush(stderr);
    jaiDiagReset(&gDiags);
    return hadErrors;
}

static inline const char *skipSpace(const char *s) {
    while (*s == ' ' || *s == '\t' || *s == '\r') ++s;
    return s;
}

/* Names the implementation owns are noise in :vars and in completion. */
static inline bool isHiddenName(const char *name) {
    return name[0] == '_' && name[1] == '_';
}

static void freeLine(char *line) {
    if (line != NULL) JAI_FREE_ARRAY(char, line, strlen(line) + 1);
}

/* ------------------------------------------------------------------ */
/* The two natives the REPL injects                                     */
/* ------------------------------------------------------------------ */

static bool replEcho(int argc, Value *args, Value *out) {
    (void)argc;
    *out = NULL_VAL;
    if (IS_NULL(args[0])) return true;   /* null is the absence of a result */

    ObjString *text = jaiValueToRepr(args[0]);
    if (text == NULL) return false;      /* __repr__ threw; let it unwind */
    fwrite(text->chars, 1, text->length, stdout);
    fputc('\n', stdout);
    return true;
}

static bool replTypeOf(int argc, Value *args, Value *out) {
    (void)argc;
    *out = NULL_VAL;
    printf("%s\n", jaiTypeNameStatic(args[0]));
    return true;
}

static void defineHelper(ObjModule *module, const char *name, JaiNativeFn fn) {
    ObjString *key = jaiStringInternC(name);
    jaiPushRoot(OBJ_VAL(key));

    ObjNative *native = jaiNativeNew(fn, name, 1, 1, NULL);
    jaiPushRoot(OBJ_VAL(native));

    jaiModuleSet(module, key, OBJ_VAL(native));
    jaiPopRoots(2);
}

/* ------------------------------------------------------------------ */
/* The persistent module                                                */
/* ------------------------------------------------------------------ */

static bool replInit(void) {
    if (gRepl.module != NULL) return true;

    if (vm.builtins == NULL) {
        replError("the virtual machine is not running");
        return false;
    }

    gRepl.codegen = jaiCodegenDefaults();
    gRepl.codegen.debugInfo = true;
    gRepl.strict = getenv("JAITHON_STRICT") != NULL;

    ObjString *path = jaiStringInternC("<repl>");
    jaiPushRoot(OBJ_VAL(path));

    ObjString *name = jaiStringInternC("__repl__");
    jaiPushRoot(OBJ_VAL(name));

    ObjModule *module = jaiModuleNew(name, path);
    jaiPushRoot(OBJ_VAL(module));
    module->state = MOD_LOADED;

    jaiTableSet(&vm.modules, OBJ_VAL(path), OBJ_VAL(module));
    defineHelper(module, "__repl_echo__", replEcho);
    defineHelper(module, "__repl_type__", replTypeOf);

    jaiPopRoots(3);

    gRepl.module = module;
    if (vm.mainModule == NULL) vm.mainModule = module;
    return true;
}

static void replReset(void) {
    if (gRepl.module != NULL) {
        (void)jaiTableDelete(&vm.modules, OBJ_VAL(gRepl.module->path));
        if (vm.mainModule == gRepl.module) vm.mainModule = NULL;
        gRepl.module = NULL;
    }
    jaiFrontEndReplForget();
    gRepl.pending.count = 0;

    CodegenOptions codegen = gRepl.codegen;
    bool strict = gRepl.strict;
    (void)replInit();
    gRepl.codegen = codegen;
    gRepl.strict = strict;
}

/* ------------------------------------------------------------------ */
/* Deciding whether more input is needed                                */
/* ------------------------------------------------------------------ */

static void replScan(const char *source, size_t length, JaiReplScan *out) {
    (void)jaiFrontEndReplScan(source, length, out);
}

static void replReportMismatch(const JaiReplScan *scan) {
    if (scan->opener != '\0') {
        replError("`%c` at line %d, column %d does not close the `%c` opened at line %d",
                  scan->closer, scan->closerLine, scan->closerCol,
                  scan->opener, scan->openerLine);
    } else {
        replError("`%c` at line %d, column %d closes nothing",
                  scan->closer, scan->closerLine, scan->closerCol);
    }
}

static bool scanPending(JaiReplScan *out) {
    if (gRepl.pending.count == 0) return false;
    replScan((const char *)gRepl.pending.data, gRepl.pending.count, out);
    return true;
}

static void replReportUnclosed(void) {
    JaiReplScan scan;
    if (!scanPending(&scan)) return;
    switch (scan.open) {
    case JAI_REPL_OPEN_BRACKET:
        replError("unexpected end of input: `%c` opened at line %d is never closed",
                  scan.opener, scan.openerLine);
        break;
    case JAI_REPL_OPEN_STRING:
        replError("unexpected end of input: the triple-quoted string opened at "
                  "line %d is never closed", scan.openerLine);
        break;
    case JAI_REPL_OPEN_COMMENT:
        replError("unexpected end of input: the block comment opened at line %d "
                  "is never closed", scan.openerLine);
        break;
    default:
        replError("unexpected end of input: the last line is not a statement yet");
        break;
    }
}

/* ------------------------------------------------------------------ */
/* Compiling and running one input                                      */

static void disassembleTree(FILE *out, const ObjFunction *fn, int depth) {
    if (fn == NULL || depth > 32) return;
    const char *name = fn->name != NULL ? fn->name->chars : "<anonymous>";
    jaiDisassembleChunk(out, &fn->chunk, name);
    fputc('\n', out);
    for (int i = 0; i < fn->chunk.constants.count; i++) {
        Value v = fn->chunk.constants.data[i];
        if (IS_FUNCTION(v)) disassembleTree(out, AS_FUNCTION(v), depth + 1);
    }
}

static void replPrintTree(const char *source, size_t length, int fileId) {
    ObjString *dump = jaiFrontEndAstText(source, length, "<repl>", fileId, false);

    if (dump != NULL)
        jaiPushRoot(OBJ_VAL(dump));

    (void)replFlushDiags();

    if (dump == NULL) {
        jaiClearException();
        return;
    }

    fwrite(dump->chars, 1, dump->length, stdout);
    fputc('\n', stdout);
    jaiPopRoot();
}

static void replExecuteOwned(char *owned, size_t length, ReplAction action,
                              bool wholeFile, const char *label) {
    if (!replInit()) {
        JAI_FREE_ARRAY(char, owned, length + 1);
        return;
    }

    const int fileId = jaiSourceAdd(label, owned, length);
    gRepl.module->sourceFileId = fileId;

    if (action == REPL_AST) {
        replPrintTree(owned, length, fileId);
        return;
    }

    JaiReplCompileOptions opts;
    opts.path = label;
    opts.fileId = fileId;
    opts.optLevel = gRepl.codegen.optLevel;
    opts.wholeFile = wholeFile;
    opts.record = action != REPL_DISASM;

    char cwd[JAI_MAX_PATH];
    opts.sourceDir = getcwd(cwd, sizeof cwd) != NULL ? cwd : "";
    opts.strict = gRepl.strict;
    opts.lateGlobals = true;
    opts.echo = action == REPL_EXEC ? "__repl_echo__"
              : action == REPL_TYPE ? "__repl_type__"
                                    : NULL;

    bool wasExpression = false;
    ObjFunction *body =
        jaiFrontEndReplCompile(owned, length, &opts,
                               gRepl.module, &wasExpression);

    if (body != NULL)
        jaiPushRoot(OBJ_VAL(body));

    if (action == REPL_TYPE && !wholeFile && !wasExpression) {
        jaiDiagReset(&gDiags);
        replError(":type expects an expression");
        if (body != NULL) jaiPopRoot();
        return;
    }

    if (replFlushDiags()) {
        if (body != NULL) jaiPopRoot();
        return;
    }

    if (body == NULL)
        return;

    if (action == REPL_DISASM) {
        disassembleTree(stdout, body, 0);
    } else {
        if (jaiVMRunModule(gRepl.module, body) != JAI_RUN_OK)
            replNoteFailure();
        gRepl.module->state = MOD_LOADED;
    }

    jaiPopRoot();
}

static void replExecute(const char *source, size_t length, ReplAction action,
                        bool wholeFile, const char *label) {
    char *owned = jaiMemdup(source, length);
    if (owned == NULL) {
        replError("out of memory while copying REPL input");
        return;
    }
    replExecuteOwned(owned, length, action, wholeFile, label);
}

static const char *nextLabel(void) {
    static char label[32];
    snprintf(label, sizeof label, "<repl:%d>", ++gRepl.inputCount);
    return label;
}

/* ------------------------------------------------------------------ */
/* Meta-commands                                                        */
/* ------------------------------------------------------------------ */

static const char *const kMetaCommands[] = {
    ":help", ":quit", ":clear", ":vars", ":type", ":ast",
    ":disasm", ":time", ":load", ":reset", ":cancel",
};
#define REPL_META_COUNT (sizeof kMetaCommands / sizeof kMetaCommands[0])

static void metaHelp(void) {
    fputs(
        "meta-commands:\n"
        "  :help            this message\n"
        "  :quit            leave the REPL (Ctrl-D does the same)\n"
        "  :clear           clear the screen\n"
        "  :vars            list the bindings of this session\n"
        "  :type EXPR       evaluate EXPR and print the type of the result\n"
        "  :ast EXPR        print the syntax tree of EXPR without running it\n"
        "  :disasm EXPR     print the bytecode EXPR compiles to, without running it\n"
        "  :time EXPR       evaluate EXPR and report how long it took\n"
        "  :load PATH       run a file in this session\n"
        "  :reset           forget every binding and start over\n"
        "  :cancel          throw away the input being typed at the `...` prompt\n"
        "\n"
        "EXPR may be any statement. A line that does not start with `:` is\n"
        "Jaithon: bindings persist, and a bare expression prints its value.\n",
        stdout);
}

static void metaClear(void) {
    if (gRepl.interactive && isatty(STDOUT_FILENO)) fputs("\033[2J\033[H", stdout);
}

static void metaVars(void) {
    if (gRepl.module == NULL) return;

    int index = 0;
    int shown = 0;
    Value key, value;

    while (jaiTableNext(&gRepl.module->globals, &index, &key, &value)) {
        if (!IS_STRING(key)) continue;

        ObjString *name = AS_STRING(key);
        if (isHiddenName(name->chars)) continue;

        ObjString *text = jaiValueToRepr(value);

        if (text == NULL) {
            jaiClearException();
            printf("  %-16s <unprintable>\n", name->chars);
        } else if (text->length <= 57) {
            /* text is a repr, which can be a view into a growing
             * concatenation buffer whose NUL a later append moved past
             * (object_string.c); take its length rather than trust %s to
             * find the end. */
            printf("  %-16s %.*s\n", name->chars, (int)text->length, text->chars);
        } else {
            size_t cut = jaiUtf8Offset(text->chars, text->length, 57);
            if (cut < text->length)
                printf("  %-16s %.*s...\n", name->chars, (int)cut, text->chars);
            else
                printf("  %-16s %.*s\n", name->chars, (int)text->length, text->chars);
        }

        ++shown;
    }

    if (shown == 0)
        fputs("  (no bindings yet)\n", stdout);
}

static void metaLoad(const char *path) {
    if (*path == '\0') {
        replError(":load expects a path");
        return;
    }

    size_t length = 0;
    char *source = jaiReadFile(path, &length);

    if (source == NULL) {
        replError("cannot read %s", path);
        return;
    }

    /* Transfer the exact file buffer directly to the source registry. */
    replExecuteOwned(source, length, REPL_QUIET, true, path);
}

/* Run one expression through the pipeline for :type / :ast / :disasm / :time. */
static void metaExpression(const char *command, const char *text,
                           ReplAction action) {
    if (*text == '\0') {
        replError("%s expects an expression", command);
        return;
    }

    const size_t length = strlen(text);
    JaiReplScan scan;
    replScan(text, length, &scan);

    if (scan.mismatched) {
        replReportMismatch(&scan);
        return;
    }

    if (scan.incomplete) {
        replError("%s: the expression is incomplete", command);
        return;
    }

    if (action == REPL_EXEC) {
        const double started = jaiClockMonotonic();
        replExecute(text, length, action, false, nextLabel());
        printf("time: %.3f ms\n",
               (jaiClockMonotonic() - started) * 1000.0);
        return;
    }

    replExecute(text, length, action, false, nextLabel());
}

/* Returns false when the command was `:quit`. */
static bool replMeta(const char *line) {
    const char *command = skipSpace(line + 1);
    const char *end = command;

    while (*end != '\0' && *end != ' ' && *end != '\t')
        ++end;

    const size_t n = (size_t)(end - command);
    const char *rest = skipSpace(end);

#define META_IS(lit) \
    (n == sizeof(lit) - 1u && memcmp(command, lit, sizeof(lit) - 1u) == 0)

    if (META_IS("help"))   { metaHelp();  return true; }
    if (META_IS("vars"))   { metaVars();  return true; }
    if (META_IS("quit"))   { return false; }
    if (META_IS("clear"))  { metaClear(); return true; }

    if (META_IS("reset")) {
        replReset();
        fputs("session reset\n", stdout);
        return true;
    }

    if (META_IS("cancel")) {
        if (gRepl.pending.count == 0)
            fputs("nothing to cancel\n", stdout);
        else {
            gRepl.pending.count = 0;
            fputs("input cancelled\n", stdout);
        }
        return true;
    }

    if (META_IS("load"))   { metaLoad(rest); return true; }
    if (META_IS("type"))   { metaExpression(":type", rest, REPL_TYPE); return true; }
    if (META_IS("ast"))    { metaExpression(":ast", rest, REPL_AST); return true; }
    if (META_IS("disasm")) { metaExpression(":disasm", rest, REPL_DISASM); return true; }
    if (META_IS("time"))   { metaExpression(":time", rest, REPL_EXEC); return true; }

#undef META_IS

    replError("unknown meta-command `:%.*s`; :help lists them all", (int)n, command);
    return true;
}

/* ------------------------------------------------------------------ */
/* Feeding lines                                                        */
/* ------------------------------------------------------------------ */

static bool firstWordIs(const char *s, const char *word) {
    size_t n = strlen(word);
    if (strncmp(s, word, n) != 0) return false;
    return s[n] == '\0' || s[n] == ' ' || s[n] == '\t' || s[n] == '\r';
}

static bool isCancelCommand(const char *trimmed) {
    if (*trimmed != ':') return false;
    const char *command = skipSpace(trimmed + 1);
    return firstWordIs(command, "cancel") || firstWordIs(command, "quit") ||
           firstWordIs(command, "reset");
}

static const char *metaCommandNamed(const char *trimmed) {
    if (*trimmed != ':') return NULL;
    const char *command = skipSpace(trimmed + 1);
    for (size_t i = 0; i < REPL_META_COUNT; i++) {
        if (firstWordIs(command, kMetaCommands[i] + 1)) return kMetaCommands[i];
    }
    return NULL;
}

static bool blankLineAbandons(void) {
    JaiReplScan scan;
    if (!scanPending(&scan)) return false;
    return scan.incomplete && scan.open == JAI_REPL_OPEN_OPERATOR;
}

bool jaiReplFeed(const char *line, bool *outIncomplete) {
    if (outIncomplete != NULL) *outIncomplete = false;
    if (line == NULL || !replInit()) return false;

    const bool continuing = gRepl.pending.count > 0;
    const char *trimmed = skipSpace(line);

    if (!continuing) {
        if (*trimmed == '\0' || (*trimmed == '#' && trimmed[1] != '*'))
            return false;

        if (*trimmed == ':') {
            if (!replMeta(trimmed)) gRepl.quit = true;
            return true;
        }
    } else if (isCancelCommand(trimmed)) {
        if (!replMeta(trimmed)) gRepl.quit = true;
        return true;
    } else if (*trimmed == '\0' && blankLineAbandons()) {
        gRepl.pending.count = 0;
        return false;
    } else {
        const char *meta = metaCommandNamed(trimmed);
        if (meta != NULL) {
            replError("`%s` is not available while an input is being typed; "
                      "finish it, or `:cancel` to abandon it", meta);
            if (outIncomplete != NULL) *outIncomplete = true;
            return true;
        }
    }

    const size_t lineLen = strlen(line);

    jaiBufReserve(&gRepl.pending, lineLen + 2);
    memcpy(gRepl.pending.data + gRepl.pending.count, line, lineLen);
    gRepl.pending.count += lineLen;
    gRepl.pending.data[gRepl.pending.count++] = '\n';
    gRepl.pending.data[gRepl.pending.count] = '\0';

    const char *source = (const char *)gRepl.pending.data;
    const size_t length = gRepl.pending.count;

    JaiReplScan scan;
    replScan(source, length, &scan);

    if (scan.mismatched) {
        replReportMismatch(&scan);
        gRepl.pending.count = 0;
        return true;
    }

    if (scan.incomplete) {
        if (outIncomplete != NULL) *outIncomplete = true;
        return false;
    }

    size_t ownedLength = 0;
    char *owned = jaiBufTakeCString(&gRepl.pending, &ownedLength);

    replExecuteOwned(owned, ownedLength, REPL_EXEC, false, nextLabel());
    return true;
}

/* ------------------------------------------------------------------ */
/* Line input                                                          */
/* ------------------------------------------------------------------ */

#ifdef JAI_HAVE_READLINE

/* readline uses malloc/free, not jaiRealloc */
static char *readlineOwnedCopy(const char *s) {
    size_t size = strlen(s) + 1;
    char *copy = (char *)malloc(size);
    if (copy != NULL) memcpy(copy, s, size);
    return copy;
}

static bool isBlank(const char *s) { return *skipSpace(s) == '\0'; }

static int    gCompletionPhase;
static int    gCompletionIndex;
static size_t gCompletionMeta;
static size_t gCompletionTextLength;

static char *completionGenerator(const char *text, int state) {
    if (state == 0) {
        gCompletionPhase = 0;
        gCompletionIndex = 0;
        gCompletionMeta = 0;
        gCompletionTextLength = strlen(text);
    }

    const size_t length = gCompletionTextLength;

    if (text[0] == ':') {
        while (gCompletionMeta < REPL_META_COUNT) {
            const char *candidate = kMetaCommands[gCompletionMeta++];
            if (strncmp(candidate, text, length) == 0)
                return readlineOwnedCopy(candidate);
        }
        return NULL;
    }

    while (gCompletionPhase < 2) {
        JaiTable *table = NULL;

        if (gCompletionPhase == 0 && gRepl.module != NULL)
            table = &gRepl.module->globals;
        else if (gCompletionPhase == 1 && vm.builtins != NULL)
            table = &vm.builtins->globals;

        Value key, value;
        while (table != NULL &&
               jaiTableNext(table, &gCompletionIndex, &key, &value)) {
            if (!IS_STRING(key)) continue;

            ObjString *name = AS_STRING(key);
            if (isHiddenName(name->chars)) continue;
            if (name->length < length) continue;
            if (memcmp(name->chars, text, length) != 0) continue;

            return readlineOwnedCopy(name->chars);
        }

        ++gCompletionPhase;
        gCompletionIndex = 0;
    }

    return NULL;
}

static char **replCompletion(const char *text, int start, int end) {
    (void)start;
    (void)end;
    if (rl_line_buffer != NULL && strncmp(rl_line_buffer, ":load", 5) == 0) {
        return NULL;
    }
    rl_attempted_completion_over = 1;
    return rl_completion_matches(text, completionGenerator);
}

static void historyPath(char *out, size_t outSize) {
    const char *home = getenv("HOME");
    if (home == NULL || home[0] == '\0') {
        out[0] = '\0';
        return;
    }
    jaiPathJoin(out, outSize, home, ".jaithon_history");
}

static void historyLoad(void) {
    char path[JAI_MAX_PATH];
    historyPath(path, sizeof path);
    using_history();
    stifle_history(REPL_HISTORY_MAX);
    if (path[0] != '\0') (void)read_history(path);
}

static void historySave(void) {
    char path[JAI_MAX_PATH];
    historyPath(path, sizeof path);
    if (path[0] == '\0') return;
    if (write_history(path) == 0) (void)history_truncate_file(path, REPL_HISTORY_MAX);
}

static volatile sig_atomic_t sPromptInterrupted;

static void promptInterrupt(int signum) {
    (void)signum;
    sPromptInterrupted = 1;
    rl_free_line_state();
    rl_replace_line("", 0);
    rl_done = 1;
}

#endif /* JAI_HAVE_READLINE */

static bool takePromptInterrupt(void) {
#ifdef JAI_HAVE_READLINE
    if (sPromptInterrupted) {
        sPromptInterrupted = 0;
        return true;
    }
#endif
    return false;
}

static char *readInputLine(const char *prompt) {
#ifdef JAI_HAVE_READLINE
    if (gRepl.interactive) {
        sPromptInterrupted = 0;
        void (*previous)(int) = signal(SIGINT, promptInterrupt);
        char *raw = readline(prompt);
        (void)signal(SIGINT, previous);

        if (sPromptInterrupted) {
            free(raw);
            fputc('\n', stdout);
            return jaiStrdup("");
        }

        if (raw == NULL) return NULL;

        char *copy = jaiStrdup(raw);
        free(raw);

        if (!isBlank(copy))
            add_history(copy);

        return copy;
    }
#endif

    if (gRepl.interactive) {
        fputs(prompt, stdout);
        fflush(stdout);
    }

    char chunk[512];
    if (fgets(chunk, sizeof chunk, stdin) == NULL)
        return NULL;

    size_t n = strlen(chunk);

    if (n > 0 && chunk[n - 1] == '\n') {
        --n;
        if (n > 0 && chunk[n - 1] == '\r') --n;

        char *line = JAI_ALLOC(char, n + 1);
        memcpy(line, chunk, n);
        line[n] = '\0';
        return line;
    }

    JaiBuf buf;
    jaiBufInit(&buf);
    jaiBufAppend(&buf, chunk, n);

    while (fgets(chunk, sizeof chunk, stdin) != NULL) {
        n = strlen(chunk);
        if (n > 0 && chunk[n - 1] == '\n') {
            jaiBufAppend(&buf, chunk, n - 1);
            break;
        }
        jaiBufAppend(&buf, chunk, n);
    }

    if (buf.count > 0 && buf.data[buf.count - 1] == '\r')
        --buf.count;

    size_t length = 0;
    return jaiBufTakeCString(&buf, &length);
}

/* ------------------------------------------------------------------ */
/* The loop                                                             */
/* ------------------------------------------------------------------ */

void jaiReplConfigure(const JaiCliOptions *opts) {
    if (opts == NULL || !replInit()) return;
    gRepl.codegen = opts->run.codegen;
    gRepl.codegen.debugInfo = true;   /* tracebacks are the point of a REPL */
}

bool jaiReplFailed(void) { return gRepl.failed; }

int jaiReplRun(const JaiCliOptions *opts) {
    if (!replInit()) return 1;

    jaiReplConfigure(opts);
    gRepl.interactive = isatty(STDIN_FILENO) != 0;
    gRepl.quit = false;
    gRepl.failed = false;

    if (gRepl.interactive) {
        printf("Jaithon %s — :help for commands, :quit to exit\n",
               JAI_VERSION_STRING);
    }

#ifdef JAI_HAVE_READLINE
    if (gRepl.interactive) {
        rl_readline_name = "jaithon";
        rl_attempted_completion_function = replCompletion;
        historyLoad();
    }
#endif

    while (!gRepl.quit) {
        const char *prompt = gRepl.pending.count > 0 ? REPL_CONTINUE : REPL_PROMPT;
        char *line = readInputLine(prompt);
        if (line == NULL) {
            if (gRepl.pending.count > 0) {
                replReportUnclosed();
                gRepl.pending.count = 0;
                if (gRepl.interactive) {
                    fputc('\n', stdout);
                    continue;
                }
                break;
            }
            if (gRepl.interactive) fputc('\n', stdout);
            break;
        }
        if (takePromptInterrupt()) {
            gRepl.pending.count = 0;
            freeLine(line);
            continue;
        }
        bool incomplete = false;
        (void)jaiReplFeed(line, &incomplete);
        freeLine(line);
        fflush(stdout);
    }

#ifdef JAI_HAVE_READLINE
    if (gRepl.interactive) historySave();
#endif

    jaiBufFree(&gRepl.pending);
    gRepl.module = NULL;

    return !gRepl.interactive && gRepl.failed ? 1 : 0;
}
