/* parser.c — the parser's shared machinery and its public entry points.
 *
 * The parser reads the lexer's finished token array and writes AST nodes into
 * the AST arena. It executes nothing, opens nothing, and keeps no state beyond
 * the Parser struct plus the file-static flags that error recovery and the
 * REPL's trial parses use.
 *
 * Error recovery: exactly one diagnostic is reported per synchronisation
 * window. `panicMode` suppresses everything until jaiPSynchronize() finds the next
 * statement boundary, so N independent syntax errors produce N diagnostics.
 *
 * The grammar itself is four files beside this one — parser_expr.c,
 * parser_stmt.c, parser_decl.c and parser_type.c — over the interface in
 * parser_internal.h, whose comment says what may cross between them.
 */

#include "parser_internal.h"

/* Deepest nesting of the recursive productions before E0107 is reported. Each
 * level costs roughly 2 KB of C stack across the expression ladder. */
#define MAX_RECURSION 200
#define MSG_BUF       512
/* ------------------------------------------------------------------ */
/* Growable scratch vectors (heap) copied into the arena when finished  */
/* ------------------------------------------------------------------ */
void *jaiPVecTake(Parser *p, void *data, int count, int capacity, size_t elemSize) {
    void *out = NULL;
    if (count > 0 && data != NULL) {
        out = jaiArenaAlloc(&p->ast->arena, elemSize * (size_t)count);
        memcpy(out, data, elemSize * (size_t)count);
    }
    if (data != NULL) (void)jaiRealloc(data, elemSize * (size_t)capacity, 0);
    return out;
}
/* Yield tracking: a function is a generator iff its own body contains `yield`,
 * so the counter is saved and restored around every nested function body. */
int jaiPYieldCount = 0;

/* ------------------------------------------------------------------ */
/* Token access                                                         */
/* ------------------------------------------------------------------ */

static Token sEofToken = { TOK_EOF, 0, 0, 0, { 0 } };

Token *jaiPTokAt(Parser *p, int idx) {
    if (idx < 0 || idx >= p->lex->tokens.count || p->lex->tokens.data == NULL) {
        return &sEofToken;
    }
    return &p->lex->tokens.data[idx];
}

Token *jaiPCur(Parser *p) { return jaiPTokAt(p, p->current); }

TokenKind jaiPCurKind(Parser *p) { return (TokenKind)jaiPCur(p)->kind; }

TokenKind jaiPKindAt(Parser *p, int offset) {
    return (TokenKind)jaiPTokAt(p, p->current + offset)->kind;
}

/* Lexical errors were already reported by the lexer; the parser skips the
 * placeholder tokens without reporting them a second time. */
static void skipErrorTokens(Parser *p) {
    while (jaiPTokAt(p, p->current)->kind == TOK_ERROR) {
        p->hadError = true;
        p->current++;
    }
}

bool jaiPCheck(Parser *p, TokenKind kind) { return jaiPCurKind(p) == kind; }

Token *jaiPAdvance(Parser *p) {
    Token *t = jaiPCur(p);
    if (t->kind != TOK_EOF) {
        p->current++;
        skipErrorTokens(p);
    }
    return t;
}

bool jaiPMatch(Parser *p, TokenKind kind) {
    if (!jaiPCheck(p, kind)) return false;
    jaiPAdvance(p);
    return true;
}

void jaiPSkipNewlines(Parser *p) {
    while (jaiPCheck(p, TOK_NEWLINE)) jaiPAdvance(p);
}

JaiSpan jaiPSpanOf(Parser *p, const Token *t) { return jaiTokenSpan(p->lex, t); }
JaiSpan jaiPCurSpan(Parser *p) { return jaiPSpanOf(p, jaiPCur(p)); }

/* Span covering tokens [startIdx, current), ignoring a trailing newline. */
JaiSpan jaiPSpanSince(Parser *p, int startIdx) {
    int endIdx = p->current - 1;
    while (endIdx > startIdx && jaiPTokAt(p, endIdx)->kind == TOK_NEWLINE) endIdx--;
    if (endIdx < startIdx) endIdx = startIdx;
    return jaiSpanJoin(jaiPSpanOf(p, jaiPTokAt(p, startIdx)), jaiPSpanOf(p, jaiPTokAt(p, endIdx)));
}

const char *jaiPRawText(Parser *p, const Token *t, size_t *outLen) {
    return jaiTokenText(p->lex, t, outLen);
}

bool jaiPTokenTextIs(Parser *p, const Token *t, const char *text) {
    size_t len = 0;
    const char *s = jaiPRawText(p, t, &len);
    if (s == NULL) return false;
    size_t want = strlen(text);
    return len == want && memcmp(s, text, want) == 0;
}

/* Is the token `offset` ahead a plain identifier spelled `text`? */
bool jaiPCheckIdentText(Parser *p, int offset, const char *text) {
    Token *t = jaiPTokAt(p, p->current + offset);
    return t->kind == TOK_IDENT && jaiPTokenTextIs(p, t, text);
}

const char *jaiPInternText(Parser *p, const char *s, size_t len) {
    if (s == NULL) return "";
    return jaiArenaMemdup(&p->ast->arena, s, len);
}

const char *jaiPInternToken(Parser *p, const Token *t) {
    size_t len = 0;
    const char *s = jaiPRawText(p, t, &len);
    return jaiPInternText(p, s, len);
}

/* Label tokens carry their leading quote; the AST stores the bare name. */
const char *jaiPInternLabel(Parser *p, const Token *t) {
    size_t len = 0;
    const char *s = jaiPRawText(p, t, &len);
    if (s != NULL && len > 0 && s[0] == '\'') { s++; len--; }
    return jaiPInternText(p, s, len);
}

/* ------------------------------------------------------------------ */
/* Diagnostics                                                          */
/* ------------------------------------------------------------------ */

/* Trial parsing: the REPL reads a line twice, once to learn how the reading
 * ends and once for real. While a trial runs no diagnostic is reported, since
 * whichever reading wins reports its own, but the first one is remembered:
 * the token the parser was on when it fired is what tells "there is more to
 * type" apart from "what was typed is wrong". Set and cleared only by
 * replTrial() below, the sole user. */
static bool sTrialActive   = false;
static bool sTrialFailed   = false;
static int  sTrialErrorTok = 0;

/* Remember the first diagnostic of a trial and swallow it. */
static JaiDiag *trialNoteError(Parser *p) {
    if (!sTrialFailed) {
        sTrialFailed = true;
        sTrialErrorTok = p->current;
    }
    return NULL;
}

void jaiPDescribeToken(Parser *p, const Token *t, char *buf, size_t bufSize) {
    switch ((TokenKind)t->kind) {
    case TOK_EOF:     snprintf(buf, bufSize, "end of file"); return;
    case TOK_NEWLINE: snprintf(buf, bufSize, "a newline"); return;
    default: break;
    }
    size_t len = 0;
    const char *s = jaiPRawText(p, t, &len);
    if (s == NULL || len == 0) {
        snprintf(buf, bufSize, "`%s`", jaiTokenKindText((TokenKind)t->kind));
        return;
    }
    if (len > 32) len = 32;
    snprintf(buf, bufSize, "`%.*s`", (int)len, s);
}

/* The single error entry point. Returns NULL when the diagnostic was
 * suppressed, so every caller must null-check before attaching help. */
JAI_PRINTF(4, 5)
JaiDiag *jaiPErrorAt(Parser *p, JaiSpan span, JaiDiagCode code, const char *fmt, ...) {
    if (p->panicMode) return NULL;
    p->panicMode = true;
    p->hadError = true;
    if (sTrialActive) return trialNoteError(p);

    char buf[MSG_BUF];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof buf, fmt, args);
    va_end(args);
    return jaiDiagError(code, span, "%s", buf);
}

JAI_PRINTF(3, 4)
JaiDiag *jaiPErrorAtCur(Parser *p, JaiDiagCode code, const char *fmt, ...) {
    if (p->panicMode) return NULL;
    char buf[MSG_BUF];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof buf, fmt, args);
    va_end(args);
    return jaiPErrorAt(p, jaiPCurSpan(p), code, "%s", buf);
}

/* A construct we understood well enough to keep parsing: report it, but do not
 * enter panic mode, so the rest of the line still produces good diagnostics.
 * Used for the Jaithon 2 spellings. */
JAI_PRINTF(4, 5)
JaiDiag *jaiPErrorRecoveredCode(Parser *p, JaiSpan span, JaiDiagCode code,
                                   const char *fmt, ...) {
    if (p->panicMode) return NULL;
    p->hadError = true;
    if (sTrialActive) return trialNoteError(p);

    char buf[MSG_BUF];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof buf, fmt, args);
    va_end(args);
    return jaiDiagError(code, span, "%s", buf);
}

/* A warning, suppressed during a trial parse. A trial's errors are swallowed
 * because the reading that wins reports its own; a warning has to be swallowed
 * for the same reason and one more. The REPL reads every line as a trial and
 * then again for real, so a warning that reached the bag from the trial is
 * rendered twice — and three times for a line starting with `{`, which is
 * tried both ways. `p` is unused today and named anyway, so that a warning can
 * later ask about panic mode the way an error does. */
JAI_PRINTF(4, 5)
JaiDiag *jaiPWarnAt(Parser *p, JaiSpan span, JaiDiagCode code, const char *fmt, ...) {
    (void)p;
    if (sTrialActive) return NULL;

    char buf[MSG_BUF];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof buf, fmt, args);
    va_end(args);
    return jaiDiagWarn(code, span, "%s", buf);
}

JAI_PRINTF(3, 4)
JaiDiag *jaiPErrorRecovered(Parser *p, JaiSpan span, const char *fmt, ...) {
    if (p->panicMode) return NULL;

    char buf[MSG_BUF];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof buf, fmt, args);
    va_end(args);
    return jaiPErrorRecoveredCode(p, span, E0100_UNEXPECTED_TOKEN, "%s", buf);
}

bool jaiPExpect(Parser *p, TokenKind kind, const char *what) {
    if (jaiPCheck(p, kind)) {
        jaiPAdvance(p);
        return true;
    }
    char found[80];
    jaiPDescribeToken(p, jaiPCur(p), found, sizeof found);
    jaiPErrorAt(p, jaiPCurSpan(p), E0101_EXPECTED_TOKEN, "expected %s, found %s", what, found);
    return false;
}

/* Reports E0107 against the delimiter that was never closed rather than the
 * token the parser gave up on: that token is usually end of file, whose line
 * is empty, so a caret there underlines nothing and the arrow sends the reader
 * to the bottom of the file instead of to the place the fix belongs. Where the
 * parser stopped is still shown, as the secondary. */
void jaiPUnclosed(Parser *p, int openIdx, const char *closeText, const char *what) {
    char found[80];
    jaiPDescribeToken(p, jaiPCur(p), found, sizeof found);
    JaiSpan open = jaiPSpanOf(p, jaiPTokAt(p, openIdx));
    JaiDiag *d = jaiPErrorAt(p, open, E0107_UNCLOSED_DELIMITER,
                         "expected `%s` to close this %s, found %s", closeText, what, found);
    if (d != NULL) {
        jaiDiagAddLabel(d, open, "unclosed %s opened here", what);
        /* End of file has no line of its own to underline, and the message
         * already names it; anything else is worth pointing at. */
        if (!jaiPCheck(p, TOK_EOF)) {
            jaiDiagAddLabel(d, jaiPCurSpan(p), "%s reached before the `%s`", found,
                            closeText);
        }
        jaiDiagAddHelp(d, "add the matching `%s`", closeText);
    }
}

const char *jaiPExpectIdentName(Parser *p, const char *what) {
    if (jaiPCheck(p, TOK_IDENT)) {
        return jaiPInternToken(p, jaiPAdvance(p));
    }
    if (jaiPCheck(p, TOK_UNDERSCORE)) {
        jaiPAdvance(p);
        return "_";
    }
    char found[80];
    jaiPDescribeToken(p, jaiPCur(p), found, sizeof found);
    jaiPErrorAt(p, jaiPCurSpan(p), E0106_EXPECTED_IDENTIFIER, "expected %s, found %s", what, found);
    return NULL;
}

/* ------------------------------------------------------------------ */
/* Node construction                                                    */
/* ------------------------------------------------------------------ */

AstNode *jaiPNewNode(Parser *p, AstKind kind, JaiSpan span) {
    return jaiAstNew(p->ast, kind, span);
}
bool jaiPEnterRecursion(Parser *p) {
    if (++p->exprDepth > MAX_RECURSION) {
        jaiPErrorAtCur(p, E0107_UNCLOSED_DELIMITER,
                   "this construct nests more than %d levels deep", MAX_RECURSION);
        return false;
    }
    return true;
}

/* ------------------------------------------------------------------ */
/* Statement termination and synchronisation                            */
/* ------------------------------------------------------------------ */

static bool startsDeclaration(TokenKind k) {
    switch (k) {
    case TOK_KW_IMPORT: case TOK_KW_FROM:   case TOK_KW_EXPORT: case TOK_KW_MODULE:
    case TOK_KW_CLASS:  case TOK_KW_TRAIT:  case TOK_KW_ENUM:   case TOK_KW_TYPE:
    case TOK_KW_FN:     case TOK_KW_LET:    case TOK_KW_VAR:    case TOK_KW_CONST:
    case TOK_KW_PUB:    case TOK_KW_PROT:   case TOK_KW_STATIC: case TOK_KW_IF:
    case TOK_KW_WHILE:  case TOK_KW_FOR:    case TOK_KW_LOOP:   case TOK_KW_MATCH:
    case TOK_KW_RETURN: case TOK_KW_BREAK:  case TOK_KW_CONTINUE:
    case TOK_KW_THROW:  case TOK_KW_TRY:    case TOK_KW_DEFER:  case TOK_KW_ASSERT:
        return true;
    default:
        return false;
    }
}

/* Skip to the next statement boundary at the current brace depth. */
void jaiPSynchronize(Parser *p) {
    /* At EOF there is nothing left to recover to: staying in panic mode stops
     * every enclosing block from reporting its own unclosed-delimiter error. */
    if (jaiPCheck(p, TOK_EOF)) return;
    p->panicMode = false;
    if (jaiPCheck(p, TOK_RBRACE)) return;
    /* Already at a statement boundary — do not skip the following line. */
    if (jaiPCheck(p, TOK_NEWLINE)) { jaiPSkipNewlines(p); return; }

    int depth = 0;
    jaiPAdvance(p);   /* always make progress past the offending token */

    while (!jaiPCheck(p, TOK_EOF)) {
        TokenKind k = jaiPCurKind(p);
        if (depth == 0) {
            if (k == TOK_NEWLINE) { jaiPSkipNewlines(p); return; }
            if (k == TOK_RBRACE)  return;
            if (startsDeclaration(k)) return;
        }
        switch (k) {
        case TOK_LBRACE: case TOK_LPAREN: case TOK_LBRACKET:
            depth++;
            break;
        case TOK_RBRACE:
            if (depth == 0) return;   /* belongs to an enclosing block */
            depth--;
            break;
        case TOK_RPAREN: case TOK_RBRACKET:
            /* A stray closer at statement level cannot close anything the
             * caller is waiting for; swallow it instead of erroring on it. */
            if (depth > 0) depth--;
            break;
        default:
            break;
        }
        jaiPAdvance(p);
    }
}

void jaiPEndStatement(Parser *p) {
    if (jaiPCheck(p, TOK_SEMICOLON)) {
        JaiDiag *d = jaiPErrorRecovered(p, jaiPCurSpan(p), "Jaithon has no statement separators");
        if (d != NULL) jaiDiagAddHelp(d, "delete the `;` — a newline ends a statement");
        jaiPAdvance(p);
    }
    if (p->panicMode) return;
    if (jaiPCheck(p, TOK_NEWLINE)) { jaiPSkipNewlines(p); return; }
    if (jaiPCheck(p, TOK_RBRACE) || jaiPCheck(p, TOK_EOF)) return;

    char found[80];
    jaiPDescribeToken(p, jaiPCur(p), found, sizeof found);
    jaiPErrorAtCur(p, E0101_EXPECTED_TOKEN,
                   "expected a newline after this statement, found %s", found);
}

/* ------------------------------------------------------------------ */
/* Public entry points                                                  */
/* ------------------------------------------------------------------ */

void jaiParserInit(Parser *p, Lexer *lex, AstContext *ast) {
    p->lex = lex;
    p->ast = ast;
    p->current = 0;
    p->hadError = false;
    p->panicMode = false;
    p->loopDepth = 0;
    p->fnDepth = 0;
    p->exprDepth = 0;
    if (lex != NULL) skipErrorTokens(p);
}

AstNode *jaiParseProgram(Parser *p) {
    if (p == NULL || p->lex == NULL || p->ast == NULL) return NULL;

    jaiPYieldCount = 0;
    skipErrorTokens(p);
    jaiPSkipNewlines(p);

    NodeVec stmts;
    JAI_VEC_INIT(&stmts);
    while (!jaiPCheck(p, TOK_EOF)) {
        int before = p->current;
        AstNode *decl = jaiParseDeclaration(p);
        if (decl != NULL) JAI_VEC_PUSH(AstNode *, &stmts, decl);
        if (p->panicMode) jaiPSynchronize(p);
        if (p->current == before) jaiPAdvance(p);   /* guarantee progress */
        jaiPSkipNewlines(p);
    }

    JaiSpan span;
    span.file = p->lex->fileId;
    span.start = 0;
    span.end = (uint32_t)p->lex->length;

    AstNode *program = jaiPNewNode(p, AST_PROGRAM, span);
    program->as.block.count = stmts.count;
    program->as.block.stmts = JAI_VEC_TAKE(p, &stmts, AstNode *);
    program->as.block.scope = NULL;
    program->as.block.captureBase = -1;

    return p->hadError ? NULL : program;
}

/* ------------------------------------------------------------------ */
/* The REPL line: which reading, and is there more of it                */
/* ------------------------------------------------------------------ */

/* Nothing may follow a REPL line but newlines. */
static void replExpectEnd(Parser *p) {
    if (jaiPCheck(p, TOK_EOF) || p->hadError) return;
    char found[80];
    jaiPDescribeToken(p, jaiPCur(p), found, sizeof found);
    jaiPErrorAtCur(p, E0108_TRAILING_TOKENS, "unexpected %s after the statement", found);
}

/* A whole REPL line in one of its two readings. Both set `hadError` when the
 * line was not written that way. */
typedef AstNode *(*ReplReader)(Parser *p);

/* The reading a file gives the line, and the one nearly every line gets. */
static AstNode *replReadStatement(Parser *p) {
    AstNode *node = jaiParseDeclaration(p);
    if (p->panicMode) jaiPSynchronize(p);
    jaiPSkipNewlines(p);
    replExpectEnd(p);
    return p->hadError ? NULL : node;
}

/* The reading only the prompt has. A statement that opens with `{` is a block,
 * so in a file `{"a": 1}` is a block whose first statement is a string, and it
 * has to stay one; at the prompt a dict or a set literal would otherwise be
 * the one kind of value that cannot be typed. Reached only from
 * jaiParseREPLLine, never from jaiParseProgram. */
static AstNode *replReadExpression(Parser *p) {
    int startIdx = p->current;
    AstNode *expr = jaiParseExpression(p);
    if (p->panicMode) jaiPSynchronize(p);
    jaiPSkipNewlines(p);
    replExpectEnd(p);
    if (p->hadError || expr == NULL) return NULL;

    AstNode *stmt = jaiPNewNode(p, AST_EXPR_STMT, jaiPSpanSince(p, startIdx));
    stmt->as.exprStmt.expr = expr;
    return stmt;
}

/* Everything a trial parse moves and must put back. The arena is not on the
 * list: it has no free, and the nodes a trial allocated are unreachable once
 * `current` is where it was. */
typedef struct {
    int  current;
    int  loopDepth;
    int  fnDepth;
    int  exprDepth;
    int  yieldCount;
    bool hadError;
    bool panicMode;
} ParserMark;

/* How a reading ended. `errorTok` doubles as how far the reading got.
 *
 * `ranOut` is the whole question the prompt is asking, and the token the first
 * diagnostic fired on answers it: end of file means the input simply stopped,
 * so the next line is the fix, while anything else means the parser had a
 * token in hand and rejected it, which no later line can undo. The lexer is
 * what makes this exact: it emits no NEWLINE after a token a statement cannot
 * end on (§1.3), so `1 +` reaches the parser as `1 + <eof>` while `class`
 * reaches it as `class <newline> <eof>`. */
typedef struct {
    bool ok;         /* the whole line parsed, with no diagnostic */
    bool ranOut;     /* the first diagnostic fired at end of file */
    int  errorTok;   /* the token it fired on */
} TrialResult;

/* Read the line one way, learn how that ended, and undo all of it. */
static TrialResult replTrial(Parser *p, ReplReader read) {
    ParserMark mark = { p->current, p->loopDepth, p->fnDepth, p->exprDepth,
                        jaiPYieldCount, p->hadError, p->panicMode };
    sTrialActive = true;
    sTrialFailed = false;
    sTrialErrorTok = p->current;

    (void)read(p);

    TrialResult r;
    r.ok = !sTrialFailed && !p->hadError;
    r.errorTok = sTrialErrorTok;
    r.ranOut = sTrialFailed && jaiPTokAt(p, sTrialErrorTok)->kind == TOK_EOF;
    sTrialActive = false;

    p->current = mark.current;
    p->loopDepth = mark.loopDepth;
    p->fnDepth = mark.fnDepth;
    p->exprDepth = mark.exprDepth;
    jaiPYieldCount = mark.yieldCount;
    p->hadError = mark.hadError;
    p->panicMode = mark.panicMode;
    return r;
}

typedef enum { REPL_AS_STATEMENT, REPL_AS_EXPRESSION, REPL_NEED_MORE } ReplReading;

/* Decided by parsing, because a count of open brackets cannot tell an input
 * that stopped early from one the parser rejected: it says "incomplete" for
 * both, and a line with a syntax error in it then holds the prompt open and
 * swallows the lines typed after it. */
static ReplReading replChooseReading(Parser *p) {
    TrialResult stmt = replTrial(p, replReadStatement);
    if (stmt.ok) return REPL_AS_STATEMENT;
    if (stmt.ranOut) return REPL_NEED_MORE;
    if (!jaiPCheck(p, TOK_LBRACE)) return REPL_AS_STATEMENT;

    TrialResult expr = replTrial(p, replReadExpression);
    if (expr.ok) return REPL_AS_EXPRESSION;
    if (expr.ranOut) return REPL_NEED_MORE;

    /* Neither reading works, so the line is wrong either way. Report the one
     * that got further, which is the one it was more nearly written as; a tie
     * goes to the block, so the prompt says what a file would say. */
    return expr.errorTok > stmt.errorTok ? REPL_AS_EXPRESSION : REPL_AS_STATEMENT;
}

AstNode *jaiParseREPLLine(Parser *p, bool *outIncomplete) {
    if (outIncomplete != NULL) *outIncomplete = false;
    if (p == NULL || p->lex == NULL || p->ast == NULL) return NULL;

    jaiPYieldCount = 0;
    skipErrorTokens(p);
    jaiPSkipNewlines(p);
    if (jaiPCheck(p, TOK_EOF)) return NULL;

    switch (replChooseReading(p)) {
    case REPL_NEED_MORE:
        if (outIncomplete != NULL) *outIncomplete = true;
        return NULL;
    case REPL_AS_EXPRESSION:
        return replReadExpression(p);
    default:
        return replReadStatement(p);
    }
}

AstNode *jaiParseExpressionOnly(Parser *p) {
    if (p == NULL || p->lex == NULL || p->ast == NULL) return NULL;

    jaiPYieldCount = 0;
    skipErrorTokens(p);
    jaiPSkipNewlines(p);
    AstNode *expr = jaiParseExpression(p);
    jaiPSkipNewlines(p);

    if (!jaiPCheck(p, TOK_EOF) && !p->panicMode) {
        char found[80];
        jaiPDescribeToken(p, jaiPCur(p), found, sizeof found);
        jaiPErrorAtCur(p, E0108_TRAILING_TOKENS, "unexpected %s after the expression", found);
    }
    return p->hadError ? NULL : expr;
}

AstNode *jaiParseSource(AstContext *ast, Lexer *lexOut, const char *source,
                        size_t length, int fileId) {
    if (ast == NULL || lexOut == NULL || source == NULL) return NULL;

    jaiLexerInit(lexOut, source, length, fileId);
    bool lexOk = jaiLexerRun(lexOut);

    Parser parser;
    jaiParserInit(&parser, lexOut, ast);
    AstNode *program = jaiParseProgram(&parser);
    return lexOk ? program : NULL;
}
