/* parser_stmt.c — statements and control flow (spec §5).
 *
 * `if` and `match` are here in both of their readings: a statement when they
 * stand alone and an expression when parser_expr.c asks for one, which is why
 * each takes an `asExpr` flag rather than existing twice.
 */

#include "parser_internal.h"

/* ------------------------------------------------------------------ */
/* Statements                                                           */
/* ------------------------------------------------------------------ */

static AstNode *parseBlock(Parser *p) {
    int startIdx = p->current;
    int openIdx = p->current;
    jaiPExpect(p, TOK_LBRACE, "`{` to open a block");

    NodeVec stmts;
    JAI_VEC_INIT(&stmts);
    jaiPSkipNewlines(p);
    while (!jaiPCheck(p, TOK_RBRACE) && !jaiPCheck(p, TOK_EOF)) {
        int before = p->current;
        AstNode *stmt = jaiParseDeclaration(p);
        if (stmt != NULL) JAI_VEC_PUSH(AstNode *, &stmts, stmt);
        if (p->panicMode) jaiPSynchronize(p);
        if (p->current == before) jaiPAdvance(p);   /* guarantee progress */
        jaiPSkipNewlines(p);
    }
    if (!jaiPCheck(p, TOK_RBRACE)) {
        jaiPUnclosed(p, openIdx, "}", "block");
    } else {
        jaiPAdvance(p);
    }

    AstNode *n = jaiPNewNode(p, AST_BLOCK, jaiPSpanSince(p, startIdx));
    n->as.block.count = stmts.count;
    n->as.block.stmts = JAI_VEC_TAKE(p, &stmts, AstNode *);
    n->as.block.scope = NULL;
    n->as.block.captureBase = -1;   /* the resolver fills this in */
    return n;
}

static AstNode *emptyBlock(Parser *p) {
    AstNode *n = jaiPNewNode(p, AST_BLOCK, jaiPCurSpan(p));
    n->as.block.stmts = NULL;
    n->as.block.count = 0;
    n->as.block.scope = NULL;
    n->as.block.captureBase = -1;
    return n;
}

AstNode *jaiParseExpectBlock(Parser *p, const char *what) {
    if (jaiPCheck(p, TOK_LBRACE)) return parseBlock(p);

    /* Jaithon 2 wrote `if c then do ... end`. */
    if (jaiPCheckIdentText(p, 0, "then") || jaiPCheckIdentText(p, 0, "do")) {
        size_t len = 0;
        const char *word = jaiPRawText(p, jaiPCur(p), &len);
        JaiDiag *d = jaiPErrorAt(p, jaiPCurSpan(p), E0104_EXPECTED_BLOCK,
                             "`%.*s` is not a Jaithon 3 keyword", (int)len, word != NULL ? word : "");
        if (d != NULL) jaiDiagAddHelp(d, "blocks are braces: `%s { ... }`", what);
        return emptyBlock(p);
    }

    char found[80];
    jaiPDescribeToken(p, jaiPCur(p), found, sizeof found);
    jaiPErrorAt(p, jaiPCurSpan(p), E0104_EXPECTED_BLOCK,
                "expected `{` to open the %s body, found %s",
            what, found);
    return emptyBlock(p);
}

AstNode *jaiParseIf(Parser *p, bool asExpr) {
    int startIdx = p->current;
    jaiPAdvance(p);   /* `if` or `elif` */
    AstNode *cond = jaiParseExpression(p);
    AstNode *thenBranch = jaiParseExpectBlock(p, "if");
    AstNode *elseBranch = NULL;

    if (jaiPCheck(p, TOK_KW_ELIF)) {
        elseBranch = jaiParseIf(p, asExpr);
    } else if (jaiPCheck(p, TOK_KW_ELSE)) {
        jaiPAdvance(p);
        if (jaiPCheck(p, TOK_KW_IF)) {
            elseBranch = jaiParseIf(p, asExpr);
        } else {
            elseBranch = jaiParseExpectBlock(p, "else");
        }
    }

    AstNode *n = jaiPNewNode(p, asExpr ? AST_IF_EXPR : AST_IF, jaiPSpanSince(p, startIdx));
    n->as.conditional.cond = cond;
    n->as.conditional.thenBranch = thenBranch;
    n->as.conditional.elseBranch = elseBranch;
    return n;
}

static AstNode *parseWhile(Parser *p, const char *label, int startIdx) {
    jaiPAdvance(p);   /* while */
    AstNode *cond = jaiParseExpression(p);
    p->loopDepth++;
    AstNode *body = jaiParseExpectBlock(p, "while");
    p->loopDepth--;
    AstNode *n = jaiPNewNode(p, AST_WHILE, jaiPSpanSince(p, startIdx));
    n->as.loop.cond = cond;
    n->as.loop.body = body;
    n->as.loop.label = label;
    n->as.loop.captureBase = -1;   /* the resolver fills this in */
    return n;
}

static AstNode *parseLoop(Parser *p, const char *label, int startIdx) {
    jaiPAdvance(p);   /* loop */
    p->loopDepth++;
    AstNode *body = jaiParseExpectBlock(p, "loop");
    p->loopDepth--;
    AstNode *n = jaiPNewNode(p, AST_LOOP, jaiPSpanSince(p, startIdx));
    n->as.loop.cond = NULL;
    n->as.loop.body = body;
    n->as.loop.label = label;
    n->as.loop.captureBase = -1;   /* the resolver fills this in */
    return n;
}

static AstNode *parseFor(Parser *p, const char *label, int startIdx) {
    jaiPAdvance(p);   /* for */
    AstNode *pattern = jaiParsePattern(p);
    jaiPExpect(p, TOK_KW_IN, "`in` after the loop pattern");
    AstNode *iterable = jaiParseExpression(p);
    p->loopDepth++;
    AstNode *body = jaiParseExpectBlock(p, "for");
    p->loopDepth--;
    AstNode *n = jaiPNewNode(p, AST_FOR, jaiPSpanSince(p, startIdx));
    n->as.forLoop.pattern = pattern;
    n->as.forLoop.iterable = iterable;
    n->as.forLoop.body = body;
    n->as.forLoop.label = label;
    n->as.forLoop.captureBase = -1;   /* the resolver fills this in */
    return n;
}

AstNode *jaiParseMatch(Parser *p, bool asExpr) {
    int startIdx = p->current;
    jaiPAdvance(p);   /* match */
    AstNode *subject = jaiParseExpression(p);

    ArmVec arms;
    JAI_VEC_INIT(&arms);
    int openIdx = p->current;
    if (!jaiPExpect(p, TOK_LBRACE, "`{` to open the match body")) {
        AstNode *n = jaiPNewNode(p, asExpr ? AST_MATCH_EXPR : AST_MATCH,
                                 jaiPSpanSince(p, startIdx));
        n->as.match.subject = subject;
        n->as.match.arms = NULL;
        n->as.match.armCount = 0;
        return n;
    }

    jaiPSkipNewlines(p);
    while (!jaiPCheck(p, TOK_RBRACE) && !jaiPCheck(p, TOK_EOF)) {
        int before = p->current;
        AstMatchArm arm;
        arm.pattern = jaiParsePattern(p);
        arm.guard = NULL;
        arm.body = NULL;
        arm.span = JAI_SPAN_NONE;
        if (jaiPMatch(p, TOK_KW_IF)) arm.guard = jaiParseExpression(p);
        jaiPExpect(p, TOK_FAT_ARROW, "`=>` after a match pattern");
        /* In arm position a `{` opens a block, never a dict literal. */
        arm.body = jaiPCheck(p, TOK_LBRACE) ? parseBlock(p) : jaiParseExpression(p);
        arm.span = jaiPSpanSince(p, before);
        JAI_VEC_PUSH(AstMatchArm, &arms, arm);

        if (p->panicMode) {
            jaiPSynchronize(p);
        } else {
            jaiPMatch(p, TOK_COMMA);
        }
        if (p->current == before) jaiPAdvance(p);
        jaiPSkipNewlines(p);
    }
    if (!jaiPCheck(p, TOK_RBRACE)) {
        jaiPUnclosed(p, openIdx, "}", "match body");
    } else {
        jaiPAdvance(p);
    }

    AstNode *n = jaiPNewNode(p, asExpr ? AST_MATCH_EXPR : AST_MATCH, jaiPSpanSince(p, startIdx));
    n->as.match.subject = subject;
    n->as.match.armCount = arms.count;
    n->as.match.arms = JAI_VEC_TAKE(p, &arms, AstMatchArm);
    return n;
}

static AstNode *parseTry(Parser *p) {
    int startIdx = p->current;
    jaiPAdvance(p);   /* try */
    AstNode *body = jaiParseExpectBlock(p, "try");

    CatchVec catches;
    JAI_VEC_INIT(&catches);
    while (jaiPCheck(p, TOK_KW_CATCH)) {
        int catchStart = p->current;
        jaiPAdvance(p);
        AstCatch clause;
        clause.name = NULL;
        clause.types = NULL;
        clause.typeCount = 0;
        clause.body = NULL;
        clause.symbol = NULL;
        clause.span = JAI_SPAN_NONE;

        if (jaiPCheck(p, TOK_IDENT) || jaiPCheck(p, TOK_UNDERSCORE)) {
            clause.name = jaiPExpectIdentName(p, "an exception binding name");
        }
        if (jaiPMatch(p, TOK_COLON)) {
            TypeVec types;
            JAI_VEC_INIT(&types);
            for (;;) {
                jaiParsePushTypeFlattened(&types, jaiParseType(p));
                if (!jaiPMatch(p, TOK_COMMA)) break;
                if (jaiPCheck(p, TOK_EOF) || p->panicMode) break;
            }
            clause.typeCount = types.count;
            clause.types = JAI_VEC_TAKE(p, &types, AstType *);
        }
        clause.body = jaiParseExpectBlock(p, "catch");
        clause.span = jaiPSpanSince(p, catchStart);
        JAI_VEC_PUSH(AstCatch, &catches, clause);
        if (p->panicMode) break;
    }

    AstNode *finallyBlock = NULL;
    if (jaiPCheck(p, TOK_KW_FINALLY)) {
        jaiPAdvance(p);
        finallyBlock = jaiParseExpectBlock(p, "finally");
    }

    if (catches.count == 0 && finallyBlock == NULL) {
        jaiPErrorAtCur(p, E0101_EXPECTED_TOKEN,
                       "a `try` block needs at least one `catch` or a `finally`");
    }

    AstNode *n = jaiPNewNode(p, AST_TRY, jaiPSpanSince(p, startIdx));
    n->as.tryStmt.body = body;
    n->as.tryStmt.catchCount = catches.count;
    n->as.tryStmt.catches = JAI_VEC_TAKE(p, &catches, AstCatch);
    n->as.tryStmt.finallyBlock = finallyBlock;
    return n;
}

/* True when the statement starts with a destructuring target: a bracketed
 * pattern whose matching close bracket is followed by `=`. */
static bool looksLikeDestructuring(Parser *p) {
    TokenKind open = jaiPCurKind(p);
    if (open != TOK_LPAREN && open != TOK_LBRACKET) return false;
    TokenKind close = (open == TOK_LPAREN) ? TOK_RPAREN : TOK_RBRACKET;

    int depth = 0;
    for (int i = p->current; ; i++) {
        Token *t = jaiPTokAt(p, i);
        if (t->kind == TOK_EOF) return false;
        if (t->kind == TOK_LPAREN || t->kind == TOK_LBRACKET || t->kind == TOK_LBRACE) {
            depth++;
        } else if (t->kind == TOK_RPAREN || t->kind == TOK_RBRACKET || t->kind == TOK_RBRACE) {
            depth--;
            if (depth == 0) {
                if (t->kind != close) return false;
                return jaiPTokAt(p, i + 1)->kind == TOK_EQ;
            }
            if (depth < 0) return false;
        }
    }
}

static bool assignOpFor(TokenKind kind, OpKind *out) {
    switch (kind) {
    case TOK_PLUS_EQ:        *out = OPK_ADD; return true;
    case TOK_MINUS_EQ:       *out = OPK_SUB; return true;
    case TOK_STAR_EQ:        *out = OPK_MUL; return true;
    case TOK_SLASH_EQ:       *out = OPK_DIV; return true;
    case TOK_SLASH_SLASH_EQ: *out = OPK_FLOORDIV; return true;
    case TOK_PERCENT_EQ:     *out = OPK_MOD; return true;
    case TOK_STAR_STAR_EQ:   *out = OPK_POW; return true;
    case TOK_AMP_EQ:         *out = OPK_BAND; return true;
    case TOK_PIPE_EQ:        *out = OPK_BOR; return true;
    case TOK_CARET_EQ:       *out = OPK_BXOR; return true;
    case TOK_SHL_EQ:         *out = OPK_SHL; return true;
    case TOK_SHR_EQ:         *out = OPK_SHR; return true;
    default: return false;
    }
}

static bool isAssignTarget(const AstNode *n) {
    switch (n->kind) {
    case AST_IDENT: case AST_MEMBER: case AST_INDEX: case AST_SLICE:
    case AST_PAT_BIND: case AST_PAT_TUPLE: case AST_PAT_LIST: case AST_PAT_WILDCARD:
        return true;
    default:
        return false;
    }
}

static AstNode *parseExprOrAssign(Parser *p) {
    int startIdx = p->current;
    AstNode *target;

    if (looksLikeDestructuring(p)) {
        target = jaiParsePattern(p);
    } else {
        target = jaiParseExpression(p);
    }

    OpKind op = OPK_ADD;
    bool compound = assignOpFor(jaiPCurKind(p), &op);
    if (!compound && !jaiPCheck(p, TOK_EQ)) {
        AstNode *n = jaiPNewNode(p, AST_EXPR_STMT, jaiPSpanSince(p, startIdx));
        n->as.exprStmt.expr = target;
        jaiPEndStatement(p);
        n->span = jaiPSpanSince(p, startIdx);
        return n;
    }

    JaiSpan opSpan = jaiPCurSpan(p);
    jaiPAdvance(p);   /* the assignment operator */

    /* `(a, b) = point` and `[x, ...rest] = xs` destructure. */
    if (!compound && (target->kind == AST_TUPLE_LIT || target->kind == AST_LIST_LIT)) {
        AstNode *asPattern = jaiParseExprToPattern(p, target);
        if (asPattern != NULL) target = asPattern;
    }

    if (!isAssignTarget(target)) {
        JaiDiag *d = jaiPErrorAt(p, target->span, E0109_INVALID_ASSIGN_TARGET,
                             "this expression cannot be assigned to");
        if (d != NULL) {
            jaiDiagAddLabel(d, opSpan, "assignment happens here");
            if (target->kind == AST_CALL) {
                jaiDiagAddHelp(d, "a call result is a value; assign to a name, field, or index");
            }
        }
    }

    AstNode *value = jaiParseExpression(p);
    AstNode *n = jaiPNewNode(p, AST_ASSIGN, jaiPSpanSince(p, startIdx));
    n->as.assign.target = target;
    n->as.assign.value = value;
    n->as.assign.op = op;
    n->as.assign.isCompound = compound;
    jaiPEndStatement(p);
    n->span = jaiPSpanSince(p, startIdx);
    return n;
}

/* `print x`, `input name`, `system "ls"` were statements in Jaithon 2. */
static AstNode *parseLegacyCallStatement(Parser *p, const char *name) {
    int startIdx = p->current;
    Token *nameTok = jaiPCur(p);
    jaiPAdvance(p);
    JaiDiag *d = jaiPErrorRecovered(p, jaiPSpanOf(p, nameTok),
                                    "`%s` is a function in Jaithon 3", name);

    AstNode *callee = jaiPNewNode(p, AST_IDENT, jaiPSpanOf(p, nameTok));
    callee->as.ident.name = jaiPInternToken(p, nameTok);

    ArgVec args;
    JAI_VEC_INIT(&args);
    int argStart = p->current;
    for (;;) {
        AstArg arg;
        arg.name = NULL;
        arg.isSpread = false;
        int one = p->current;
        arg.value = jaiParseExpression(p);
        arg.span = jaiPSpanSince(p, one);
        JAI_VEC_PUSH(AstArg, &args, arg);
        if (!jaiPMatch(p, TOK_COMMA)) break;
        if (!jaiParseStartsExpression(p) || p->panicMode) break;
    }
    JaiSpan argSpan = jaiPSpanSince(p, argStart);

    if (d != NULL) {
        const char *src = p->lex->source;
        if (src != NULL && argSpan.file >= 0 && argSpan.end > argSpan.start &&
            (size_t)argSpan.end <= p->lex->length) {
            jaiDiagAddHelp(d, "write `%s(%.*s)`", name,
                           (int)(argSpan.end - argSpan.start), src + argSpan.start);
        } else {
            jaiDiagAddHelp(d, "write `%s(...)`", name);
        }
    }

    AstNode *call = jaiPNewNode(p, AST_CALL, jaiPSpanSince(p, startIdx));
    call->as.call.callee = callee;
    call->as.call.argCount = args.count;
    call->as.call.args = JAI_VEC_TAKE(p, &args, AstArg);

    AstNode *stmt = jaiPNewNode(p, AST_EXPR_STMT, call->span);
    stmt->as.exprStmt.expr = call;
    jaiPEndStatement(p);
    return stmt;
}

/* Diagnostics for the Jaithon 2 spellings that are now plain identifiers. */
static AstNode *checkLegacyStatement(Parser *p, bool *handled) {
    *handled = false;
    if (!jaiPCheck(p, TOK_IDENT)) return NULL;

    static const char *const callWords[] = { "print", "input", "system" };
    for (size_t i = 0; i < sizeof callWords / sizeof callWords[0]; i++) {
        if (!jaiPTokenTextIs(p, jaiPCur(p), callWords[i])) continue;
        TokenKind next = jaiPKindAt(p, 1);
        bool bare = next == TOK_IDENT || next == TOK_INT || next == TOK_FLOAT ||
                    next == TOK_STRING || next == TOK_FSTRING_START ||
                    next == TOK_KW_TRUE || next == TOK_KW_FALSE ||
                    next == TOK_KW_NULL || next == TOK_KW_SELF;
        if (!bare) return NULL;
        *handled = true;
        return parseLegacyCallStatement(p, callWords[i]);
    }

    /* `end` is a plain identifier in Jaithon 3 (spec §1.5 does not reserve it),
     * so `end += 1` is a statement, not a stray block terminator. Only the
     * Jaithon 2 spelling — `end` alone on its line — is diagnosed. */
    if (jaiPTokenTextIs(p, jaiPCur(p), "end")) {
        TokenKind next = jaiPKindAt(p, 1);
        if (next != TOK_NEWLINE && next != TOK_EOF && next != TOK_RBRACE) {
            return NULL;
        }
        JaiDiag *d = jaiPErrorAt(p, jaiPCurSpan(p), E0100_UNEXPECTED_TOKEN,
                             "`end` is not a Jaithon 3 keyword");
        if (d != NULL) jaiDiagAddHelp(d, "blocks close with `}`");
        jaiPAdvance(p);
        *handled = true;
        return NULL;
    }

    if ((jaiPTokenTextIs(p, jaiPCur(p), "func") || jaiPTokenTextIs(p, jaiPCur(p), "def")) &&
        jaiPKindAt(p, 1) == TOK_IDENT) {
        size_t len = 0;
        const char *word = jaiPRawText(p, jaiPCur(p), &len);
        JaiDiag *d = jaiPErrorRecovered(p, jaiPCurSpan(p), "`%.*s` is not a Jaithon 3 keyword",
                                    (int)len, word != NULL ? word : "");
        if (d != NULL) jaiDiagAddHelp(d, "functions are declared with `fn`");
        *handled = true;
        int startIdx = p->current;
        return jaiParseFnRest(p, AST_VIS_PRIVATE, false, startIdx, true, false);
    }

    return NULL;
}

AstNode *jaiParseStatement(Parser *p) {
    int startIdx = p->current;

    switch (jaiPCurKind(p)) {
    case TOK_LBRACE:
        return parseBlock(p);
    case TOK_KW_IF:
        return jaiParseIf(p, false);
    case TOK_KW_WHILE:
        return parseWhile(p, NULL, startIdx);
    case TOK_KW_LOOP:
        return parseLoop(p, NULL, startIdx);
    case TOK_KW_FOR:
        return parseFor(p, NULL, startIdx);
    case TOK_KW_MATCH:
        return jaiParseMatch(p, false);
    case TOK_KW_TRY:
        return parseTry(p);

    case TOK_LABEL: {
        const char *label = jaiPInternLabel(p, jaiPCur(p));
        jaiPAdvance(p);
        jaiPExpect(p, TOK_COLON, "`:` after a loop label");
        switch (jaiPCurKind(p)) {
        case TOK_KW_WHILE: return parseWhile(p, label, startIdx);
        case TOK_KW_LOOP:  return parseLoop(p, label, startIdx);
        case TOK_KW_FOR:   return parseFor(p, label, startIdx);
        default: break;
        }
        JaiDiag *d = jaiPErrorAt(p, jaiPSpanSince(p, startIdx), E0115_LABEL_NOT_ON_LOOP,
                             "`'%s` labels something that is not a loop", label);
        if (d != NULL) jaiDiagAddHelp(d, "only `while`, `loop`, and `for` may be labelled");
        return NULL;
    }

    case TOK_KW_RETURN: {
        jaiPAdvance(p);
        AstNode *n = jaiPNewNode(p, AST_RETURN, jaiPSpanSince(p, startIdx));
        n->as.ret.value = jaiParseStartsExpression(p) ? jaiParseExpression(p) : NULL;
        jaiPEndStatement(p);
        n->span = jaiPSpanSince(p, startIdx);
        return n;
    }
    case TOK_KW_THROW: {
        jaiPAdvance(p);
        AstNode *n = jaiPNewNode(p, AST_THROW, jaiPSpanSince(p, startIdx));
        n->as.ret.value = jaiParseExpression(p);
        jaiPEndStatement(p);
        n->span = jaiPSpanSince(p, startIdx);
        return n;
    }
    case TOK_KW_BREAK: case TOK_KW_CONTINUE: {
        bool isBreak = jaiPCheck(p, TOK_KW_BREAK);
        jaiPAdvance(p);
        AstNode *n = jaiPNewNode(p, isBreak ? AST_BREAK : AST_CONTINUE, jaiPSpanSince(p, startIdx));
        n->as.jump.label = NULL;
        if (jaiPCheck(p, TOK_LABEL)) {
            n->as.jump.label = jaiPInternLabel(p, jaiPCur(p));
            jaiPAdvance(p);
        }
        jaiPEndStatement(p);
        n->span = jaiPSpanSince(p, startIdx);
        return n;
    }
    case TOK_KW_DEFER: {
        jaiPAdvance(p);
        AstNode *n = jaiPNewNode(p, AST_DEFER, jaiPSpanSince(p, startIdx));
        n->as.defer.body = jaiParseExpectBlock(p, "defer");
        n->span = jaiPSpanSince(p, startIdx);
        return n;
    }
    case TOK_KW_ASSERT: {
        jaiPAdvance(p);
        AstNode *n = jaiPNewNode(p, AST_ASSERT, jaiPSpanSince(p, startIdx));
        n->as.assertStmt.cond = jaiParseExpression(p);
        n->as.assertStmt.message = jaiPMatch(p, TOK_COMMA) ? jaiParseExpression(p) : NULL;
        jaiPEndStatement(p);
        n->span = jaiPSpanSince(p, startIdx);
        return n;
    }

    case TOK_KW_ELSE: case TOK_KW_ELIF: case TOK_KW_CATCH: case TOK_KW_FINALLY: {
        const char *word = jaiTokenKindText(jaiPCurKind(p));
        JaiDiag *d = jaiPErrorAt(p, jaiPCurSpan(p), E0100_UNEXPECTED_TOKEN,
                             "`%s` has no matching statement before it", word);
        if (d != NULL) jaiDiagAddHelp(d, "put `%s` on the same line as the closing `}`", word);
        return NULL;
    }

    case TOK_KW_IMPL: {
        JaiDiag *d = jaiPErrorAt(p, jaiPCurSpan(p), E0100_UNEXPECTED_TOKEN,
                             "`impl` blocks are not part of Jaithon 3");
        if (d != NULL) {
            jaiDiagAddHelp(d, "declare methods inside the class body and list traits with `:`");
        }
        return NULL;
    }

    case TOK_IDENT: {
        bool handled = false;
        AstNode *legacy = checkLegacyStatement(p, &handled);
        if (handled) return legacy;
        break;
    }

    default:
        break;
    }

    return parseExprOrAssign(p);
}
