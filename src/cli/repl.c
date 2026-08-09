/* repl.c — the interactive read-eval-print loop.
 *
 * Every input is compiled into an anonymous module body that shares one
 * persistent module, so bindings survive from line to line: the resolver runs
 * in `replMode`, where a name it has never seen becomes a deferred global that
 * the VM looks up in that module at run time.
 *
 * Incomplete input is a first-class state, not an error: a bracket, a
 * triple-quoted string or a block comment that is still open, or a line that
 * ends on an operator, keeps the loop reading with a continuation prompt.
 * Every other outcome is decided now, so the prompt never waits for a line
 * that cannot arrive, and end of input never drops what was typed in silence.
 */

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

#include "cli.h"

#include "../runtime/frontend.h"

#include "../codegen/codegen.h"
#include "../lang/ast.h"
#include "../lang/lexer.h"
#include "../lang/parser.h"
#include "../vm/chunk.h"

#define REPL_PROMPT      ">>> "
#define REPL_CONTINUE    "... "
#define REPL_HISTORY_MAX 1000

typedef enum {
    REPL_EXEC,     /* run it; echo the value of a bare expression  */
    REPL_QUIET,    /* run it; never echo (:load, :time's own timer) */
    REPL_TYPE,     /* run it; report the type of a bare expression  */
    REPL_AST,      /* print the syntax tree, do not run             */
    REPL_DISASM,   /* print the bytecode, do not run                */
} ReplAction;

static struct {
    ObjModule     *module;      /* kept alive by vm.modules, a GC root */
    JaiBuf         pending;     /* lines accumulated for an unfinished input */
    CodegenOptions codegen;
    /* The class, trait and enum names this session has declared, so that a
     * later line can use them as types. See "Types declared at the prompt". */
    JAI_VEC(Symbol *) types;
    int            inputCount;  /* names the source files: <repl:1>, <repl:2> */
    bool           interactive;
    bool           strict;
    bool           quit;
    bool           failed;      /* an input did not run, or threw. See jaiReplRun. */
} gRepl;

/* ------------------------------------------------------------------ */
/* Small helpers                                                        */
/* ------------------------------------------------------------------ */

/* Something the user asked for did not happen: a diagnostic stopped an input
 * before it ran, an exception escaped one that did, or end of input arrived
 * with a statement still half typed. The session carries on regardless, so
 * this is remembered rather than acted on; jaiReplRun turns it into the exit
 * status. A warning is not one of these, and does not reach here unless the
 * session was asked to treat warnings as errors. */
static void replNoteFailure(void) { gRepl.failed = true; }

static void replError(const char *fmt, ...) JAI_PRINTF(1, 2);

/* REPL usage problems go through the diagnostic renderer too, so they look
 * like every other error the user sees. */
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

/* Render and clear the bag, dropping the two warnings that only make sense for
 * a whole file: at a prompt every binding is unused until the next line uses
 * it, and every import is there to be used interactively. */
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

static const char *skipSpace(const char *s) {
    while (*s == ' ' || *s == '\t' || *s == '\r') s++;
    return s;
}

/* Names the implementation owns are noise in :vars and in completion. */
static bool isHiddenName(const char *name) {
    return name[0] == '_' && name[1] == '_';
}

/* The lexer reads up to the NUL as well as the length, so the accumulated
 * input must always have one just past its last byte. */
static void bufTerminate(JaiBuf *b) {
    jaiBufPush(b, '\0');
    b->count--;
}

static void freeLine(char *line) {
    if (line != NULL) JAI_FREE_ARRAY(char, line, strlen(line) + 1);
}

/* ------------------------------------------------------------------ */
/* The two natives the REPL injects                                     */
/* ------------------------------------------------------------------ */

/* `expr` at the prompt is compiled as `__repl_echo__(expr)`. */
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

/* `:type expr` is compiled as `__repl_type__(expr)`. */
static bool replTypeOf(int argc, Value *args, Value *out) {
    (void)argc;
    *out = NULL_VAL;
    printf("%s\n", jaiTypeNameStatic(args[0]));
    return true;
}

static void defineHelper(ObjModule *module, const char *name, JaiNativeFn fn) {
    ObjNative *native = jaiNativeNew(fn, name, 1, 1, NULL);
    jaiPushRoot(OBJ_VAL(native));
    jaiModuleSet(module, jaiStringInternC(name), OBJ_VAL(native));
    jaiPopRoot();
}

/* ------------------------------------------------------------------ */
/* Types declared at the prompt                                         */
/* ------------------------------------------------------------------ */

/* A class survives to the next line as a value like any other binding, but an
 * annotation is not a value: `fn make() -> Point` asks the checker for a type
 * named `Point`, and the declaration that could answer went away with the tree
 * of the line that introduced it.
 *
 * That list used to live here, in the C resolver's built-in registry. It now
 * lives in the `ReplSession` the front end keeps (lib/jaithon/compile/mod.jai),
 * because the registry is part of `src/sema` and the session is not. Forgetting
 * it is `:reset`, and `jaiFrontEndReplForget` is the whole of what that takes.
 */


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
    gRepl.codegen.debugInfo = true;   /* tracebacks are the point of a REPL */
    gRepl.strict = getenv("JAITHON_STRICT") != NULL;

    ObjString *path = jaiStringInternC("<repl>");
    jaiPushRoot(OBJ_VAL(path));
    ObjModule *module = jaiModuleNew(jaiStringInternC("__repl__"), path);
    jaiPushRoot(OBJ_VAL(module));
    module->state = MOD_LOADED;

    /* vm.modules is a GC root, so registering the module is also what keeps it
     * alive between lines. */
    jaiTableSet(&vm.modules, OBJ_VAL(path), OBJ_VAL(module));
    defineHelper(module, "__repl_echo__", replEcho);
    defineHelper(module, "__repl_type__", replTypeOf);
    jaiPopRoots(2);

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

    /* :reset forgets the bindings, not the session: replInit installs the
     * defaults, but -O0 and --strict came from the command line and outlive
     * every module this session builds. */
    CodegenOptions codegen = gRepl.codegen;
    bool strict = gRepl.strict;
    (void)replInit();
    gRepl.codegen = codegen;
    gRepl.strict = strict;
}

/* ------------------------------------------------------------------ */
/* Deciding whether more input is needed                                */
/* ------------------------------------------------------------------ */

/* Whether the buffer is a whole input is a question about syntax, so the front
 * end answers it. Nothing is registered and no diagnostic escapes: a line
 * still being typed must not reach the source table, and the compile that
 * follows a complete input produces the real diagnostics with real spans.
 *
 * A front end that cannot be reached leaves the verdict zeroed, which reads as
 * "complete" -- the compile below then says why, which beats a prompt hanging
 * on a line nothing can finish. */
static void replScan(const char *source, size_t length, JaiReplScan *out) {
    (void)jaiFrontEndReplScan(source, length, out);
}

/* A closer that matches nothing is reported here rather than left to the
 * compiler: the parser's own "needs more input" test sees the same unbalanced
 * count and would ask for a line that cannot fix it. */
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

/* Ask the same question about the input being typed. Only the rare paths need
 * it, abandoning a line and end of input, so lexing the buffer again there is
 * cheaper than carrying the last verdict around. */
static bool scanPending(JaiReplScan *out) {
    if (gRepl.pending.count == 0) return false;
    replScan((const char *)gRepl.pending.data, gRepl.pending.count, out);
    return true;
}

/* End of input with a half-typed statement in hand. Dropping it without a word
 * is how a piped session loses its last function whole, so name what was left
 * open. */
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
/* ------------------------------------------------------------------ */

static AstNode *programOf(AstContext *ast, AstNode *stmt) {
    AstNode *program = jaiAstNew(ast, AST_PROGRAM, stmt->span);
    AstNode **stmts = jaiAstNodeArray(ast, 1);
    stmts[0] = stmt;
    program->as.block.stmts = stmts;
    program->as.block.count = 1;
    return program;
}

/* Nested functions live in the enclosing chunk's constant pool. */
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

/* Print the C front end's syntax tree for one input. The last thing the prompt
 * asks of `src/lang`: `jaiAstPrint`'s s-expression form has golden tests and no
 * self-hosted equivalent yet. */
static void replPrintCTree(const char *source, size_t length, int fileId) {
    AstContext ast;
    Lexer lex;
    Parser parser;
    jaiAstContextInit(&ast);
    jaiLexerInit(&lex, source, length, fileId);
    bool lexOk = jaiLexerRun(&lex);
    jaiParserInit(&parser, &lex, &ast);

    AstNode *program = NULL;
    if (lexOk) {
        bool incomplete = false;
        AstNode *stmt = jaiParseREPLLine(&parser, &incomplete);
        if (stmt != NULL) program = programOf(&ast, stmt);
    }

    (void)replFlushDiags();
    if (program != NULL) jaiAstPrint(stdout, program, 0);
    jaiLexerFree(&lex);
    jaiAstContextFree(&ast);
}

/* Compile one complete input against the persistent module and act on it.
 * Diagnostics and uncaught exceptions are reported here; the session always
 * continues afterwards. */
static void replExecute(const char *source, size_t length, ReplAction action,
                        bool wholeFile, const char *label) {
    if (!replInit()) return;

    /* jaiSourceAdd takes ownership: the copy outlives this function because
     * compiled chunks refer to it by file id for tracebacks. */
    char *owned = jaiMemdup(source, length);
    int fileId = jaiSourceAdd(label, owned, length);
    /* The front end stamps this id into every span it reports and into the line
     * table, so a diagnostic can name a column and a traceback a line. */
    gRepl.module->sourceFileId = fileId;

    if (action == REPL_AST) {
        replPrintCTree(owned, length, fileId);
        return;
    }

    JaiReplCompileOptions opts;
    opts.path      = label;
    opts.fileId    = fileId;
    opts.optLevel  = gRepl.codegen.optLevel;
    opts.wholeFile = wholeFile;
    /* `:disasm` compiles an input and never runs it, so what it declares is
     * not the session's. */
    opts.record    = action != REPL_DISASM;
    /* `from m import T` at the prompt looks where the prompt was started, the
     * way the C resolver did. The label names no directory of its own. */
    char cwd[JAI_MAX_PATH];
    opts.sourceDir = getcwd(cwd, sizeof cwd) != NULL ? cwd : "";
    opts.strict    = gRepl.strict;
    opts.echo      = action == REPL_EXEC ? "__repl_echo__"
                   : action == REPL_TYPE ? "__repl_type__"
                                         : NULL;

    bool wasExpression = false;
    ObjFunction *body = jaiFrontEndReplCompile(owned, length, &opts,
                                               gRepl.module, &wasExpression);

    /* `:type` is the one action that needs a value, and a statement has none.
     * Asked after the compile because only the front end knows which of the two
     * readings the line got. */
    if (action == REPL_TYPE && !wholeFile && !wasExpression) {
        jaiDiagReset(&gDiags);
        replError(":type expects an expression");
        return;
    }

    if (replFlushDiags()) return;   /* an input with an error never runs */
    /* No body and no diagnostic is an input with nothing in it -- a comment, a
     * blank line -- which is not a failure. Everything that *is* one reports
     * before getting here, and replFlushDiags has already noted it. */
    if (body == NULL) return;

    jaiPushRoot(OBJ_VAL(body));
    if (action == REPL_DISASM) {
        disassembleTree(stdout, body, 0);
    } else {
        /* jaiVMRunModule prints the traceback of anything that escapes. */
        if (jaiVMRunModule(gRepl.module, body) != JAI_RUN_OK) replNoteFailure();
        gRepl.module->state = MOD_LOADED;   /* a bad line does not end the session */
    }
    jaiPopRoot();
}

static const char *nextLabel(void) {
    static char label[32];
    snprintf(label, sizeof label, "<repl:%d>", ++gRepl.inputCount);
    return label;
}

/* ------------------------------------------------------------------ */
/* Meta-commands                                                        */
/* ------------------------------------------------------------------ */

/* One of three lists that have to say the same thing: this one, the dispatch in
 * replMeta, and the text metaHelp prints. tests/repl/meta_help.repl pins that
 * text, so a command reaches the completer and the user together or not at all. */
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

/* The escape sequence means something to a terminal and nothing to a pipe or a
 * file, where it is seven bytes of noise in the middle of the transcript. The
 * screen being cleared is the one stdout writes to, so that stream is asked as
 * well: a session driven from a keyboard can still have its output redirected,
 * and a session driven from a pipe has nobody watching a screen to clear. */
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
        if (text == NULL) {           /* a throwing __repr__ must not kill :vars */
            jaiClearException();
            printf("  %-16s <unprintable>\n", name->chars);
        } else {
            /* Elide at a scalar boundary: the terminal must never be handed
             * half a UTF-8 sequence. */
            size_t cut = jaiUtf8Offset(text->chars, text->length, 57);
            if (cut < text->length) {
                printf("  %-16s %.*s...\n", name->chars, (int)cut, text->chars);
            } else {
                printf("  %-16s %s\n", name->chars, text->chars);
            }
        }
        shown++;
    }
    if (shown == 0) fputs("  (no bindings yet)\n", stdout);
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
    replExecute(source, length, REPL_QUIET, true, path);
    JAI_FREE_ARRAY(char, source, length + 1);
}

/* Run one expression through the pipeline for :type / :ast / :disasm / :time. */
static void metaExpression(const char *command, const char *text,
                           ReplAction action) {
    if (*text == '\0') {
        replError("%s expects an expression", command);
        return;
    }
    size_t length = strlen(text);
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

    double started = jaiClockMonotonic();
    replExecute(text, length, action, false, nextLabel());
    if (action == REPL_EXEC) {
        printf("time: %.3f ms\n", (jaiClockMonotonic() - started) * 1000.0);
    }
}

/* Returns false when the command was `:quit`. */
static bool replMeta(const char *line) {
    const char *cursor = skipSpace(line + 1);
    char command[32];
    size_t n = 0;
    while (cursor[n] != '\0' && cursor[n] != ' ' && cursor[n] != '\t' &&
           n + 1 < sizeof command) {
        command[n] = cursor[n];
        n++;
    }
    command[n] = '\0';
    const char *rest = skipSpace(cursor + n);

    if (strcmp(command, "help") == 0)   { metaHelp();  return true; }
    if (strcmp(command, "vars") == 0)   { metaVars();  return true; }
    if (strcmp(command, "quit") == 0)   { return false; }
    if (strcmp(command, "clear") == 0)  { metaClear(); return true; }
    if (strcmp(command, "reset") == 0)  {
        replReset();
        fputs("session reset\n", stdout);
        return true;
    }
    if (strcmp(command, "cancel") == 0) {
        if (gRepl.pending.count == 0) {
            fputs("nothing to cancel\n", stdout);
        } else {
            gRepl.pending.count = 0;
            fputs("input cancelled\n", stdout);
        }
        return true;
    }
    if (strcmp(command, "load") == 0)   { metaLoad(rest); return true; }
    if (strcmp(command, "type") == 0)   { metaExpression(":type", rest, REPL_TYPE); return true; }
    if (strcmp(command, "ast") == 0)    { metaExpression(":ast", rest, REPL_AST); return true; }
    if (strcmp(command, "disasm") == 0) { metaExpression(":disasm", rest, REPL_DISASM); return true; }
    if (strcmp(command, "time") == 0)   { metaExpression(":time", rest, REPL_EXEC); return true; }

    replError("unknown meta-command `:%s`; :help lists them all", command);
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

/* The meta-commands that stay meta at the continuation prompt. Each one ends
 * the input being typed, so appending it as source instead would leave the
 * session with no way out at all where readline is not linked in and Ctrl-C
 * therefore does nothing. */
static bool isCancelCommand(const char *trimmed) {
    if (*trimmed != ':') return false;
    const char *command = skipSpace(trimmed + 1);
    return firstWordIs(command, "cancel") || firstWordIs(command, "quit") ||
           firstWordIs(command, "reset");
}

/* The meta-command a line names, or NULL. The other eight do not end the input
 * being typed, so at the `... ` prompt they cannot run — but they must not
 * become source either: spliced into the buffer, `:vars` used to fail to parse
 * and take the half-typed input down with it, reporting `expected an
 * expression, found :` about a line the user never meant as code. */
static const char *metaCommandNamed(const char *trimmed) {
    if (*trimmed != ':') return NULL;
    const char *command = skipSpace(trimmed + 1);
    for (size_t i = 0; i < REPL_META_COUNT; i++) {
        if (firstWordIs(command, kMetaCommands[i] + 1)) return kMetaCommands[i];
    }
    return NULL;
}

/* A blank line at the `... ` prompt means "forget it" only when nothing but a
 * trailing operator is holding the input open. Inside a bracket, a string or a
 * comment a blank line is part of what is being typed, and a pasted block that
 * contains one has to arrive intact. */
static bool blankLineAbandons(void) {
    JaiReplScan scan;
    if (!scanPending(&scan)) return false;
    return scan.incomplete && scan.open == JAI_REPL_OPEN_OPERATOR;
}

bool jaiReplFeed(const char *line, bool *outIncomplete) {
    if (outIncomplete != NULL) *outIncomplete = false;
    if (line == NULL || !replInit()) return false;

    bool continuing = gRepl.pending.count > 0;
    const char *trimmed = skipSpace(line);
    if (!continuing) {
        /* A line comment is nothing to run, but `#*` opens a block that the
         * next lines close, so that one has to be accumulated like any other
         * unfinished construct. */
        if (*trimmed == '\0' || (*trimmed == '#' && trimmed[1] != '*')) return false;
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
            /* The pending input is left exactly as it was: the answer to a
             * mistyped command is to go on finishing the input, not to lose it. */
            replError("`%s` is not available while an input is being typed; "
                      "finish it, or `:cancel` to abandon it", meta);
            if (outIncomplete != NULL) *outIncomplete = true;
            return true;
        }
    }

    jaiBufAppendStr(&gRepl.pending, line);
    jaiBufPush(&gRepl.pending, '\n');
    bufTerminate(&gRepl.pending);

    const char *source = (const char *)gRepl.pending.data;
    size_t length = gRepl.pending.count;
    JaiReplScan scan;
    replScan(source, length, &scan);

    if (scan.mismatched) {
        /* No later line can rescue this one, and keeping it would swallow
         * every line that follows it into an input that never ends. */
        replReportMismatch(&scan);
        gRepl.pending.count = 0;
        return true;
    }
    if (scan.incomplete) {
        if (outIncomplete != NULL) *outIncomplete = true;
        return false;
    }

    replExecute(source, length, REPL_EXEC, false, nextLabel());
    gRepl.pending.count = 0;
    return true;
}

/* ------------------------------------------------------------------ */
/* Line input                                                          */
/* ------------------------------------------------------------------ */

#ifdef JAI_HAVE_READLINE

/* readline owns what it hands back and releases it with libc free(), and it
 * frees the completion matches the same way. Those buffers are the only ones
 * in Jaithon that do not come from jaiRealloc. */
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

static char *completionGenerator(const char *text, int state) {
    if (state == 0) {
        gCompletionPhase = 0;
        gCompletionIndex = 0;
        gCompletionMeta = 0;
    }
    size_t length = strlen(text);

    if (text[0] == ':') {
        while (gCompletionMeta < REPL_META_COUNT) {
            const char *candidate = kMetaCommands[gCompletionMeta++];
            if (strncmp(candidate, text, length) == 0) {
                return readlineOwnedCopy(candidate);
            }
        }
        return NULL;
    }

    while (gCompletionPhase < 2) {
        JaiTable *table = NULL;
        if (gCompletionPhase == 0 && gRepl.module != NULL) {
            table = &gRepl.module->globals;
        } else if (gCompletionPhase == 1 && vm.builtins != NULL) {
            table = &vm.builtins->globals;
        }

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
        gCompletionPhase++;
        gCompletionIndex = 0;
    }
    return NULL;
}

static char **replCompletion(const char *text, int start, int end) {
    (void)start;
    (void)end;
    /* `:load` wants paths, so leave that one to readline's filename default. */
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

/* Ctrl-C at the prompt abandons the line rather than leaving half a statement
 * behind for the next one. This handler is installed only while readline is
 * waiting, so the VM keeps its own SIGINT — the one that stops a running
 * program at the next safepoint — for everything else. */
static void promptInterrupt(int signum) {
    (void)signum;
    sPromptInterrupted = 1;
    rl_free_line_state();
    rl_replace_line("", 0);
    rl_done = 1;
}

#endif /* JAI_HAVE_READLINE */

/* True once, when the last prompt was cut short by Ctrl-C. */
static bool takePromptInterrupt(void) {
#ifdef JAI_HAVE_READLINE
    if (sPromptInterrupted) {
        sPromptInterrupted = 0;
        return true;
    }
#endif
    return false;
}

/* Read one line without its terminator, or NULL at end of input. */
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
            return jaiStrdup("");        /* the caller drops it and reprompts */
        }
        if (raw == NULL) return NULL;
        char *copy = jaiStrdup(raw);
        free(raw);                       /* libc's, not ours */
        if (!isBlank(copy)) add_history(copy);
        return copy;
    }
#endif
    if (gRepl.interactive) {
        fputs(prompt, stdout);
        fflush(stdout);
    }

    JaiBuf buf;
    jaiBufInit(&buf);
    char chunk[512];
    bool any = false;
    while (fgets(chunk, sizeof chunk, stdin) != NULL) {
        any = true;
        size_t n = strlen(chunk);
        if (n > 0 && chunk[n - 1] == '\n') {
            jaiBufAppend(&buf, chunk, n - 1);
            break;
        }
        jaiBufAppend(&buf, chunk, n);
    }
    if (!any && buf.count == 0) {
        jaiBufFree(&buf);
        return NULL;
    }
    if (buf.count > 0 && buf.data[buf.count - 1] == '\r') buf.count--;

    size_t length = 0;
    return jaiBufTakeCString(&buf, &length);
}

/* ------------------------------------------------------------------ */
/* The loop                                                             */
/* ------------------------------------------------------------------ */

/* What a session takes from the command line. Split out of jaiReplRun because
 * `--eval` is one input rather than a session: it has no loop to enter, but it
 * is the same evaluator and must answer -O0 and --release the same way, or the
 * flags are accepted and silently ignored. */
void jaiReplConfigure(const JaiCliOptions *opts) {
    if (opts == NULL || !replInit()) return;
    gRepl.codegen = opts->run.codegen;
    gRepl.codegen.debugInfo = true;   /* tracebacks are the point of a REPL */
}

/* Whether any input has failed since the process started, by the rule on
 * replNoteFailure. jaiReplRun turns it into the status of a piped session;
 * `--eval` reads it directly, because there the one input *is* the run. */
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
            /* End of input with something half typed: say what was left open
             * before dropping it. At a terminal that is Ctrl-D, and the
             * session continues on the next prompt; in a pipe there is no
             * next prompt, so the loop ends here. */
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
    /* The Symbols themselves belong to the resolver's registry, which outlives
     * the session on purpose; only the list of which ones were ours is here. */
    JAI_VEC_FREE(Symbol *, &gRepl.types);
    gRepl.module = NULL;   /* vm.modules still owns it; freed with the VM */

    /* A session driven by a pipe or a file is a program that was run, and a
     * program that could not run part of what it was given did not succeed:
     * without this, `jaithon repl < script` is unusable in a shell that tests
     * its status, and useless as a check in CI.
     *
     * At a terminal the status reports the session rather than the transcript,
     * and the session is what the person at the keyboard chose to end. Every
     * error was printed as it happened and answered by typing the next line,
     * so a typo an hour ago is not a verdict on `:quit`, and a shell prompt
     * that shows the last status would carry it there for no reason. */
    return !gRepl.interactive && gRepl.failed ? 1 : 0;
}
