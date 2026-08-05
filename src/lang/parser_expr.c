/* parser_expr.c — the expression grammar: precedence climbing over the ladder
 * in spec §4.1, and the postfix, literal and comprehension forms under it.
 *
 * One function per precedence level, each calling the next one up, so the
 * table in parser.h and the call chain here are the same list read twice. An
 * `if` or a `match` in expression position hands back to parser_stmt.c, which
 * is what makes them expressions and statements at once.
 */

#include "parser_internal.h"

/* The one production that recurses upward through the ladder rather
 * than down it: `**` is right-associative and unary binds tighter. */
static AstNode *parseUnary(Parser *p);

/* ------------------------------------------------------------------ */
/* Expression nodes                                                     */
/* ------------------------------------------------------------------ */

/* Placeholder returned instead of NULL so that no expression caller has to
 * null-check; the tree is discarded anyway once hadError is set. */
static AstNode *errorExpr(Parser *p, JaiSpan span) {
    return jaiPNewNode(p, AST_NULL_LIT, span);
}

static AstNode *makeBinaryNode(Parser *p, AstKind kind, OpKind op, AstNode *l, AstNode *r) {
    AstNode *n = jaiPNewNode(p, kind, jaiSpanJoin(l->span, r->span));
    n->as.binary.op = op;
    n->as.binary.left = l;
    n->as.binary.right = r;
    return n;
}

/* ------------------------------------------------------------------ */
/* Expressions                                                          */
/* ------------------------------------------------------------------ */

bool jaiParseStartsExpression(Parser *p) {
    switch (jaiPCurKind(p)) {
    case TOK_INT: case TOK_FLOAT: case TOK_STRING:
    case TOK_FSTRING_START: case TOK_FSTRING_END:
    case TOK_IDENT: case TOK_UNDERSCORE:
    case TOK_LPAREN: case TOK_LBRACKET: case TOK_LBRACE:
    case TOK_MINUS: case TOK_PLUS: case TOK_TILDE: case TOK_PIPE:
    case TOK_KW_NOT: case TOK_KW_TRUE: case TOK_KW_FALSE: case TOK_KW_NULL:
    case TOK_KW_SELF: case TOK_KW_SUPER: case TOK_KW_FN: case TOK_KW_IF:
    case TOK_KW_MATCH: case TOK_KW_YIELD:
    /* `throw` in expression position has type `never` (spec §4.3). The
     * statement form is matched earlier in parseStatement, so this only
     * affects places that genuinely expect an expression. */
    case TOK_KW_THROW:
        return true;
    default:
        return false;
    }
}

static bool isNumericLiteral(const AstNode *n) {
    return n != NULL && (n->kind == AST_INT_LIT || n->kind == AST_FLOAT_LIT);
}

/* Was the literal written in base ten? A `0b`/`0x`/`0o` literal is a bit
 * pattern, and nobody ever wrote a Jaithon 2 power that way, so `0b1100 ^
 * 0b1010` must not be mistaken for one. */
static bool isDecimalLiteral(const AstNode *n) {
    if (!isNumericLiteral(n)) return false;
    const JaiSourceFile *file = jaiSourceGet(n->span.file);
    if (file == NULL || file->source == NULL) return true;
    if (n->span.start + 1 >= (uint32_t)file->length) return true;
    if (file->source[n->span.start] != '0') return true;
    char tag = file->source[n->span.start + 1];
    return tag != 'b' && tag != 'B' && tag != 'x' && tag != 'X' &&
           tag != 'o' && tag != 'O';
}

/* --- argument lists ------------------------------------------------ */

/* Parses arguments up to and including the closing `)`. */
static void parseArgs(Parser *p, int openIdx, AstArg **outArgs, int *outCount) {
    ArgVec args;
    JAI_VEC_INIT(&args);
    jaiPSkipNewlines(p);
    if (!jaiPCheck(p, TOK_RPAREN)) {
        for (;;) {
            jaiPSkipNewlines(p);
            AstArg arg;
            arg.name = NULL;
            arg.value = NULL;
            arg.isSpread = false;
            arg.span = jaiPCurSpan(p);
            int argStart = p->current;

            if (jaiPCheck(p, TOK_ELLIPSIS)) {
                jaiPAdvance(p);
                arg.isSpread = true;
                arg.value = jaiParseExpression(p);
            } else {
                /* `name: value` is a keyword argument; nothing else puts a
                 * colon directly after an identifier here. */
                if (jaiPCheck(p, TOK_IDENT) && jaiPKindAt(p, 1) == TOK_COLON) {
                    arg.name = jaiPInternToken(p, jaiPAdvance(p));
                    jaiPAdvance(p);   /* ':' */
                }
                arg.value = jaiParseExpression(p);
            }
            arg.span = jaiPSpanSince(p, argStart);
            JAI_VEC_PUSH(AstArg, &args, arg);

            if (args.count > JAI_MAX_ARGS) {
                jaiPErrorAt(p, arg.span, E0100_UNEXPECTED_TOKEN,
                        "more than %d arguments in one call", JAI_MAX_ARGS);
                break;
            }
            jaiPSkipNewlines(p);
            if (!jaiPMatch(p, TOK_COMMA)) break;
            jaiPSkipNewlines(p);
            if (jaiPCheck(p, TOK_RPAREN) || jaiPCheck(p, TOK_EOF)) break;
        }
    }
    if (!jaiPCheck(p, TOK_RPAREN)) {
        jaiPUnclosed(p, openIdx, ")", "argument list");
    } else {
        jaiPAdvance(p);
    }
    *outCount = args.count;
    *outArgs = JAI_VEC_TAKE(p, &args, AstArg);
}

/* --- f-strings ----------------------------------------------------- */

static AstNode *stringPartNode(Parser *p, const Token *t) {
    size_t len = 0;
    const char *text = jaiTokenStringValue(p->lex, t, &len);
    if (text == NULL) text = jaiPRawText(p, t, &len);
    AstNode *n = jaiPNewNode(p, AST_STR_LIT, jaiPSpanOf(p, t));
    n->as.strLit.chars = jaiPInternText(p, text, len);
    n->as.strLit.length = len;
    return n;
}

static bool stringPartIsEmpty(Parser *p, const Token *t) {
    size_t len = 0;
    const char *text = jaiTokenStringValue(p->lex, t, &len);
    if (text == NULL) return false;
    return len == 0;
}

/* A format spec (`{x:.2f}`) becomes an explicit `__format__` call so that code
 * generation needs no special case for it. */
static AstNode *applyFormatSpec(Parser *p, AstNode *value) {
    if (!jaiPCheck(p, TOK_COLON)) return value;
    jaiPAdvance(p);
    uint32_t specStart = jaiPCur(p)->start;
    uint32_t specEnd = specStart;
    while (!jaiPCheck(p, TOK_FSTRING_MID) && !jaiPCheck(p, TOK_FSTRING_END) &&
           !jaiPCheck(p, TOK_EOF)) {
        Token *t = jaiPAdvance(p);
        specEnd = t->start + t->length;
    }
    if (specEnd <= specStart) return value;

    const char *src = p->lex->source;
    size_t specLen = (size_t)(specEnd - specStart);
    JaiSpan span = value->span;

    AstNode *spec = jaiPNewNode(p, AST_STR_LIT, span);
    spec->as.strLit.chars = (src != NULL) ? jaiPInternText(p, src + specStart, specLen) : "";
    spec->as.strLit.length = (src != NULL) ? specLen : 0;

    AstNode *member = jaiPNewNode(p, AST_MEMBER, span);
    member->as.member.object = value;
    member->as.member.name = "__format__";
    member->as.member.cacheSlot = -1;

    AstArg *args = JAI_ARENA_NEW_ARRAY(&p->ast->arena, AstArg, 1);
    args[0].name = NULL;
    args[0].value = spec;
    args[0].isSpread = false;
    args[0].span = span;

    AstNode *call = jaiPNewNode(p, AST_CALL, span);
    call->as.call.callee = member;
    call->as.call.args = args;
    call->as.call.argCount = 1;
    return call;
}

static AstNode *parseFString(Parser *p) {
    int startIdx = p->current;
    NodeVec parts;
    JAI_VEC_INIT(&parts);

    /* An f-string with no holes may arrive as a lone FSTRING_END chunk. */
    Token *startTok = jaiPAdvance(p);
    if (!stringPartIsEmpty(p, startTok)) {
        JAI_VEC_PUSH(AstNode *, &parts, stringPartNode(p, startTok));
    }
    bool complete = startTok->kind == TOK_FSTRING_END;

    while (!complete) {
        if (jaiPCheck(p, TOK_EOF)) {
            jaiPErrorAtCur(p, E0107_UNCLOSED_DELIMITER, "unterminated interpolated string");
            break;
        }
        /* An empty `{}` is a hole with no expression. */
        if (!jaiPCheck(p, TOK_FSTRING_MID) && !jaiPCheck(p, TOK_FSTRING_END) &&
            !jaiPCheck(p, TOK_COLON)) {
            AstNode *value = jaiParseExpression(p);
            value = applyFormatSpec(p, value);
            JAI_VEC_PUSH(AstNode *, &parts, value);
        } else if (jaiPCheck(p, TOK_COLON)) {
            AstNode *empty = errorExpr(p, jaiPCurSpan(p));
            jaiPErrorAtCur(p, E0102_EXPECTED_EXPRESSION,
                           "expected an expression before the format spec");
            JAI_VEC_PUSH(AstNode *, &parts, empty);
        }

        if (jaiPCheck(p, TOK_FSTRING_MID)) {
            Token *mid = jaiPAdvance(p);
            if (!stringPartIsEmpty(p, mid)) JAI_VEC_PUSH(AstNode *, &parts, stringPartNode(p, mid));
            continue;
        }
        if (jaiPCheck(p, TOK_FSTRING_END)) {
            Token *end = jaiPAdvance(p);
            if (!stringPartIsEmpty(p, end)) JAI_VEC_PUSH(AstNode *, &parts, stringPartNode(p, end));
            break;
        }
        if (!p->panicMode) {
            char found[80];
            jaiPDescribeToken(p, jaiPCur(p), found, sizeof found);
            jaiPErrorAtCur(p, E0101_EXPECTED_TOKEN,
                       "expected `}` to close the interpolation, found %s", found);
        }
        break;
    }

    AstNode *n = jaiPNewNode(p, AST_FSTRING, jaiPSpanSince(p, startIdx));
    if (parts.count == 0) {
        AstNode *empty = jaiPNewNode(p, AST_STR_LIT, n->span);
        empty->as.strLit.chars = "";
        empty->as.strLit.length = 0;
        JAI_VEC_PUSH(AstNode *, &parts, empty);
    }
    n->as.fstring.partCount = parts.count;
    n->as.fstring.parts = JAI_VEC_TAKE(p, &parts, AstNode *);
    return n;
}

/* --- comprehensions ------------------------------------------------ */

static void parseCompClauses(Parser *p, AstCompClause **outClauses, int *outCount) {
    ClauseVec clauses;
    JAI_VEC_INIT(&clauses);
    while (jaiPCheck(p, TOK_KW_FOR)) {
        int clauseStart = p->current;
        jaiPAdvance(p);
        AstCompClause clause;
        clause.pattern = jaiParsePattern(p);
        clause.iterable = NULL;
        clause.conditions = NULL;
        clause.conditionCount = 0;
        clause.span = JAI_SPAN_NONE;
        jaiPExpect(p, TOK_KW_IN, "`in` after the comprehension pattern");
        clause.iterable = jaiParseExpression(p);

        NodeVec conds;
        JAI_VEC_INIT(&conds);
        while (jaiPCheck(p, TOK_KW_IF)) {
            jaiPAdvance(p);
            JAI_VEC_PUSH(AstNode *, &conds, jaiParseExpression(p));
            if (jaiPCheck(p, TOK_EOF)) break;
        }
        clause.conditionCount = conds.count;
        clause.conditions = JAI_VEC_TAKE(p, &conds, AstNode *);
        clause.span = jaiPSpanSince(p, clauseStart);
        JAI_VEC_PUSH(AstCompClause, &clauses, clause);
        if (jaiPCheck(p, TOK_EOF) || p->panicMode) break;
    }
    *outCount = clauses.count;
    *outClauses = JAI_VEC_TAKE(p, &clauses, AstCompClause);
}

static AstNode *finishComprehension(Parser *p, int startIdx, CompKind kind,
                                    AstNode *keyExpr, AstNode *element,
                                    TokenKind closer, const char *closeText,
                                    const char *what, int openIdx) {
    AstNode *n = jaiPNewNode(p, AST_COMPREHENSION, jaiPCurSpan(p));
    n->as.comp.kind = kind;
    n->as.comp.keyExpr = keyExpr;
    n->as.comp.element = element;
    parseCompClauses(p, &n->as.comp.clauses, &n->as.comp.clauseCount);
    jaiPSkipNewlines(p);
    if (!jaiPCheck(p, closer)) {
        jaiPUnclosed(p, openIdx, closeText, what);
    } else {
        jaiPAdvance(p);
    }
    n->span = jaiPSpanSince(p, startIdx);
    return n;
}

/* --- collection literals ------------------------------------------- */

static AstNode *parseListLiteral(Parser *p) {
    int startIdx = p->current;
    int openIdx = p->current;
    jaiPAdvance(p);   /* '[' */
    jaiPSkipNewlines(p);

    if (jaiPCheck(p, TOK_RBRACKET)) {
        jaiPAdvance(p);
        AstNode *n = jaiPNewNode(p, AST_LIST_LIT, jaiPSpanSince(p, startIdx));
        n->as.sequence.items = NULL;
        n->as.sequence.count = 0;
        return n;
    }

    AstNode *first = jaiParseExpression(p);
    if (jaiPCheck(p, TOK_KW_FOR)) {
        return finishComprehension(p, startIdx, COMP_LIST, NULL, first,
                                   TOK_RBRACKET, "]", "list comprehension", openIdx);
    }

    NodeVec items;
    JAI_VEC_INIT(&items);
    JAI_VEC_PUSH(AstNode *, &items, first);
    jaiPSkipNewlines(p);
    while (jaiPMatch(p, TOK_COMMA)) {
        jaiPSkipNewlines(p);
        if (jaiPCheck(p, TOK_RBRACKET) || jaiPCheck(p, TOK_EOF)) break;
        JAI_VEC_PUSH(AstNode *, &items, jaiParseExpression(p));
        jaiPSkipNewlines(p);
        if (p->panicMode) break;
    }
    if (!jaiPCheck(p, TOK_RBRACKET)) {
        jaiPUnclosed(p, openIdx, "]", "list literal");
    } else {
        jaiPAdvance(p);
    }
    AstNode *n = jaiPNewNode(p, AST_LIST_LIT, jaiPSpanSince(p, startIdx));
    n->as.sequence.count = items.count;
    n->as.sequence.items = JAI_VEC_TAKE(p, &items, AstNode *);
    return n;
}

/* `{}` is the empty dict; `{a, b}` is a set; `{k: v}` is a dict. */
static AstNode *parseBraceLiteral(Parser *p) {
    int startIdx = p->current;
    int openIdx = p->current;
    jaiPAdvance(p);   /* '{' */
    jaiPSkipNewlines(p);

    if (jaiPCheck(p, TOK_RBRACE)) {
        jaiPAdvance(p);
        AstNode *n = jaiPNewNode(p, AST_DICT_LIT, jaiPSpanSince(p, startIdx));
        n->as.dict.keys = NULL;
        n->as.dict.values = NULL;
        n->as.dict.count = 0;
        return n;
    }

    AstNode *first = jaiParseExpression(p);

    if (jaiPMatch(p, TOK_COLON)) {
        jaiPSkipNewlines(p);
        AstNode *firstValue = jaiParseExpression(p);
        if (jaiPCheck(p, TOK_KW_FOR)) {
            return finishComprehension(p, startIdx, COMP_DICT, first, firstValue,
                                       TOK_RBRACE, "}", "dict comprehension", openIdx);
        }
        NodeVec keys, values;
        JAI_VEC_INIT(&keys);
        JAI_VEC_INIT(&values);
        JAI_VEC_PUSH(AstNode *, &keys, first);
        JAI_VEC_PUSH(AstNode *, &values, firstValue);
        jaiPSkipNewlines(p);
        while (jaiPMatch(p, TOK_COMMA)) {
            jaiPSkipNewlines(p);
            if (jaiPCheck(p, TOK_RBRACE) || jaiPCheck(p, TOK_EOF)) break;
            JAI_VEC_PUSH(AstNode *, &keys, jaiParseExpression(p));
            jaiPExpect(p, TOK_COLON, "`:` between a dict key and its value");
            jaiPSkipNewlines(p);
            JAI_VEC_PUSH(AstNode *, &values, jaiParseExpression(p));
            jaiPSkipNewlines(p);
            if (p->panicMode) break;
        }
        if (!jaiPCheck(p, TOK_RBRACE)) {
            jaiPUnclosed(p, openIdx, "}", "dict literal");
        } else {
            jaiPAdvance(p);
        }
        AstNode *n = jaiPNewNode(p, AST_DICT_LIT, jaiPSpanSince(p, startIdx));
        n->as.dict.count = keys.count;
        n->as.dict.keys = JAI_VEC_TAKE(p, &keys, AstNode *);
        n->as.dict.values = JAI_VEC_TAKE(p, &values, AstNode *);
        return n;
    }

    if (jaiPCheck(p, TOK_KW_FOR)) {
        return finishComprehension(p, startIdx, COMP_SET, NULL, first,
                                   TOK_RBRACE, "}", "set comprehension", openIdx);
    }

    NodeVec items;
    JAI_VEC_INIT(&items);
    JAI_VEC_PUSH(AstNode *, &items, first);
    jaiPSkipNewlines(p);
    while (jaiPMatch(p, TOK_COMMA)) {
        jaiPSkipNewlines(p);
        if (jaiPCheck(p, TOK_RBRACE) || jaiPCheck(p, TOK_EOF)) break;
        JAI_VEC_PUSH(AstNode *, &items, jaiParseExpression(p));
        jaiPSkipNewlines(p);
        if (p->panicMode) break;
    }
    if (!jaiPCheck(p, TOK_RBRACE)) {
        jaiPUnclosed(p, openIdx, "}", "set literal");
    } else {
        jaiPAdvance(p);
    }
    AstNode *n = jaiPNewNode(p, AST_SET_LIT, jaiPSpanSince(p, startIdx));
    n->as.sequence.count = items.count;
    n->as.sequence.items = JAI_VEC_TAKE(p, &items, AstNode *);
    return n;
}

static AstNode *parseParenExpr(Parser *p) {
    int startIdx = p->current;
    int openIdx = p->current;
    jaiPAdvance(p);   /* '(' */
    jaiPSkipNewlines(p);

    if (jaiPCheck(p, TOK_RPAREN)) {
        jaiPAdvance(p);
        AstNode *n = jaiPNewNode(p, AST_TUPLE_LIT, jaiPSpanSince(p, startIdx));
        n->as.sequence.items = NULL;
        n->as.sequence.count = 0;
        return n;
    }

    AstNode *first = jaiParseExpression(p);
    jaiPSkipNewlines(p);

    if (jaiPCheck(p, TOK_KW_FOR)) {
        return finishComprehension(p, startIdx, COMP_GENERATOR, NULL, first,
                                   TOK_RPAREN, ")", "generator expression", openIdx);
    }

    if (jaiPCheck(p, TOK_COMMA)) {
        NodeVec items;
        JAI_VEC_INIT(&items);
        JAI_VEC_PUSH(AstNode *, &items, first);
        while (jaiPMatch(p, TOK_COMMA)) {
            jaiPSkipNewlines(p);
            if (jaiPCheck(p, TOK_RPAREN) || jaiPCheck(p, TOK_EOF)) break;
            JAI_VEC_PUSH(AstNode *, &items, jaiParseExpression(p));
            jaiPSkipNewlines(p);
            if (p->panicMode) break;
        }
        if (!jaiPCheck(p, TOK_RPAREN)) {
            jaiPUnclosed(p, openIdx, ")", "tuple literal");
        } else {
            jaiPAdvance(p);
        }
        AstNode *n = jaiPNewNode(p, AST_TUPLE_LIT, jaiPSpanSince(p, startIdx));
        n->as.sequence.count = items.count;
        n->as.sequence.items = JAI_VEC_TAKE(p, &items, AstNode *);
        return n;
    }

    if (!jaiPCheck(p, TOK_RPAREN)) {
        jaiPUnclosed(p, openIdx, ")", "parenthesised expression");
    } else {
        jaiPAdvance(p);
    }
    return first;
}

/* --- lambdas and anonymous functions -------------------------------- */

static AstNode *parseLambda(Parser *p) {
    int startIdx = p->current;
    jaiPAdvance(p);   /* '|' */

    ParamVec params;
    JAI_VEC_INIT(&params);
    if (!jaiPCheck(p, TOK_PIPE)) {
        for (;;) {
            AstParam param;
            memset(&param, 0, sizeof param);
            param.span = jaiPCurSpan(p);
            Token *nameTok = jaiPCur(p);
            const char *name = jaiPExpectIdentName(p, "a lambda parameter name");
            if (name == NULL) break;
            param.name = name;
            param.span = jaiPSpanOf(p, nameTok);
            if (jaiPMatch(p, TOK_COLON)) param.type = jaiParseType(p);
            for (int i = 0; i < params.count; i++) {
                if (strcmp(params.data[i].name, name) == 0) {
                    JaiDiag *d = jaiPErrorRecoveredCode(p, param.span, E0111_DUPLICATE_PARAMETER,
                                                    "duplicate parameter `%s`", name);
                    if (d != NULL) jaiDiagAddLabel(d, params.data[i].span, "first declared here");
                    break;
                }
            }
            JAI_VEC_PUSH(AstParam, &params, param);
            if (!jaiPMatch(p, TOK_COMMA)) break;
            if (jaiPCheck(p, TOK_EOF)) break;
        }
    }
    jaiPExpect(p, TOK_PIPE, "`|` to close the lambda parameter list");

    int savedYield = jaiPYieldCount;
    jaiPYieldCount = 0;
    p->fnDepth++;
    AstNode *body = jaiParseExpression(p);
    p->fnDepth--;

    AstNode *n = jaiPNewNode(p, AST_LAMBDA, jaiPSpanSince(p, startIdx));
    n->as.fn.name = NULL;
    n->as.fn.paramCount = params.count;
    n->as.fn.params = JAI_VEC_TAKE(p, &params, AstParam);
    n->as.fn.body = body;
    n->as.fn.isExprBody = true;
    n->as.fn.visibility = AST_VIS_PRIVATE;
    n->as.fn.isGenerator = jaiPYieldCount > 0;
    jaiPYieldCount = savedYield;
    return n;
}

/* --- primary -------------------------------------------------------- */

AstNode *jaiParsePrimary(Parser *p) {
    int startIdx = p->current;
    Token *t = jaiPCur(p);

    switch ((TokenKind)t->kind) {
    case TOK_INT: {
        jaiPAdvance(p);
        AstNode *n = jaiPNewNode(p, AST_INT_LIT, jaiPSpanOf(p, t));
        n->as.intLit = t->v.intValue;
        return n;
    }
    case TOK_FLOAT: {
        jaiPAdvance(p);
        AstNode *n = jaiPNewNode(p, AST_FLOAT_LIT, jaiPSpanOf(p, t));
        n->as.floatLit = t->v.floatValue;
        return n;
    }
    case TOK_STRING: {
        jaiPAdvance(p);
        return stringPartNode(p, t);
    }
    case TOK_FSTRING_START: case TOK_FSTRING_END:
        return parseFString(p);
    /* `throw e` is an expression of type `never` (spec §4.3), which is what
     * lets a match arm or a `??` right-hand side raise. The statement form in
     * parseStatement handles the common case and produces AST_THROW; this one
     * only fires where an expression is expected. */
    case TOK_KW_THROW: {
        jaiPAdvance(p);
        AstNode *n = jaiPNewNode(p, AST_THROW_EXPR, jaiPSpanSince(p, startIdx));
        n->as.wrap.operand = jaiParseExpression(p);
        n->span = jaiPSpanSince(p, startIdx);
        return n;
    }
    case TOK_KW_TRUE: case TOK_KW_FALSE: {
        jaiPAdvance(p);
        AstNode *n = jaiPNewNode(p, AST_BOOL_LIT, jaiPSpanOf(p, t));
        n->as.boolLit = (t->kind == TOK_KW_TRUE);
        return n;
    }
    case TOK_KW_NULL:
        jaiPAdvance(p);
        return jaiPNewNode(p, AST_NULL_LIT, jaiPSpanOf(p, t));
    case TOK_KW_SELF:
        jaiPAdvance(p);
        return jaiPNewNode(p, AST_SELF, jaiPSpanOf(p, t));
    case TOK_KW_SUPER:
        jaiPAdvance(p);
        return jaiPNewNode(p, AST_SUPER, jaiPSpanOf(p, t));
    case TOK_UNDERSCORE: {
        jaiPAdvance(p);
        AstNode *n = jaiPNewNode(p, AST_IDENT, jaiPSpanOf(p, t));
        n->as.ident.name = "_";
        return n;
    }
    case TOK_LPAREN:
        return parseParenExpr(p);
    case TOK_LBRACKET:
        return parseListLiteral(p);
    case TOK_LBRACE:
        return parseBraceLiteral(p);
    case TOK_PIPE:
        return parseLambda(p);
    case TOK_KW_IF:
        return jaiParseIf(p, true);
    case TOK_KW_MATCH:
        return jaiParseMatch(p, true);
    case TOK_KW_FN:
        return jaiParseFnRest(p, AST_VIS_PRIVATE, false, startIdx, false, false);
    case TOK_KW_YIELD: {
        jaiPAdvance(p);
        jaiPYieldCount++;
        AstNode *n = jaiPNewNode(p, AST_YIELD, jaiPSpanOf(p, t));
        n->as.wrap.operand = jaiParseStartsExpression(p) ? jaiParseExpression(p) : NULL;
        n->span = jaiPSpanSince(p, startIdx);
        return n;
    }
    case TOK_IDENT: {
        /* `new Foo()` — the keyword was removed in Jaithon 3. */
        if (jaiPTokenTextIs(p, t, "new") && jaiPKindAt(p, 1) == TOK_IDENT) {
            size_t clsLen = 0;
            const char *cls = jaiPRawText(p, jaiPTokAt(p, p->current + 1), &clsLen);
            JaiDiag *d = jaiPErrorRecovered(
                p, jaiPSpanOf(p, t), "the `new` keyword was removed in Jaithon 3");
            if (d != NULL && cls != NULL) {
                jaiDiagAddHelp(d, "a class is callable: write `%.*s(...)`", (int)clsLen, cls);
            } else if (d != NULL) {
                jaiDiagAddHelp(d, "a class is callable: write `ClassName(...)`");
            }
            jaiPAdvance(p);
            return jaiParsePrimary(p);
        }
        /* `await expr` is contextual: std.async defines it, so it is only an
         * operator when an operand plainly follows. */
        if (jaiPTokenTextIs(p, t, "await")) {
            TokenKind next = jaiPKindAt(p, 1);
            if (next == TOK_IDENT || next == TOK_KW_SELF || next == TOK_INT ||
                next == TOK_FLOAT || next == TOK_STRING || next == TOK_FSTRING_START) {
                jaiPAdvance(p);
                AstNode *n = jaiPNewNode(p, AST_AWAIT, jaiPSpanOf(p, t));
                n->as.wrap.operand = parseUnary(p);
                n->span = jaiPSpanSince(p, startIdx);
                return n;
            }
        }
        jaiPAdvance(p);
        AstNode *n = jaiPNewNode(p, AST_IDENT, jaiPSpanOf(p, t));
        n->as.ident.name = jaiPInternToken(p, t);
        return n;
    }
    default:
        break;
    }

    char found[80];
    jaiPDescribeToken(p, t, found, sizeof found);
    JaiDiag *d = jaiPErrorAt(p, jaiPSpanOf(p, t), E0102_EXPECTED_EXPRESSION,
                         "expected an expression, found %s", found);
    if (d != NULL) {
        switch ((TokenKind)t->kind) {
        case TOK_KW_ELSE: case TOK_KW_ELIF:
            jaiDiagAddHelp(d, "put `%s` on the same line as the closing `}`",
                           jaiTokenKindText((TokenKind)t->kind));
            break;
        case TOK_EQ:
            jaiDiagAddHelp(d, "`==` compares, `=` assigns");
            break;
        default:
            break;
        }
    }
    return errorExpr(p, jaiPSpanOf(p, t));
}

/* --- postfix -------------------------------------------------------- */

static AstNode *parseSubscript(Parser *p, AstNode *object, int openIdx) {
    jaiPAdvance(p);   /* '[' */
    AstNode *start = NULL;
    if (!jaiPCheck(p, TOK_COLON)) {
        if (jaiPCheck(p, TOK_RBRACKET)) {
            jaiPErrorAtCur(p, E0102_EXPECTED_EXPRESSION, "expected an index expression");
            jaiPAdvance(p);
            return errorExpr(p, jaiSpanJoin(object->span, jaiPSpanSince(p, openIdx)));
        }
        start = jaiParseExpression(p);
    }

    if (jaiPCheck(p, TOK_COLON)) {
        jaiPAdvance(p);
        AstNode *stop = NULL;
        AstNode *step = NULL;
        if (!jaiPCheck(p, TOK_COLON) && !jaiPCheck(p, TOK_RBRACKET)) stop = jaiParseExpression(p);
        if (jaiPMatch(p, TOK_COLON)) {
            if (!jaiPCheck(p, TOK_RBRACKET)) step = jaiParseExpression(p);
        }
        if (!jaiPCheck(p, TOK_RBRACKET)) {
            jaiPUnclosed(p, openIdx, "]", "slice");
        } else {
            jaiPAdvance(p);
        }
        AstNode *n = jaiPNewNode(p, AST_SLICE,
                                 jaiSpanJoin(object->span,
                                             jaiPSpanSince(p, openIdx)));
        n->as.slice.object = object;
        n->as.slice.start = start;
        n->as.slice.stop = stop;
        n->as.slice.step = step;
        return n;
    }

    /* `x[a, b]` is never a valid index — an index takes exactly one expression
     * — so a comma here means generic type arguments: `Pair[B, A](x, y)`.
     * The list is collected into a tuple and left as an AST_INDEX; the
     * resolver decides which it is, because only the symbol table knows
     * whether `Pair` names a class. Keeping that decision out of the parser is
     * what makes `Box[int]` unambiguous without lookahead hacks. */
    if (jaiPCheck(p, TOK_COMMA)) {
        NodeVec args;
        JAI_VEC_INIT(&args);
        JAI_VEC_PUSH(AstNode *, &args, start);
        while (jaiPMatch(p, TOK_COMMA)) {
            jaiPSkipNewlines(p);
            if (jaiPCheck(p, TOK_RBRACKET)) break;      /* trailing comma */
            JAI_VEC_PUSH(AstNode *, &args, jaiParseExpression(p));
            if (p->panicMode) break;
        }
        int count = args.count;
        AstNode *tuple = jaiPNewNode(p, AST_TUPLE_LIT, jaiPSpanSince(p, openIdx));
        tuple->as.sequence.items = JAI_VEC_TAKE(p, &args, AstNode *);
        tuple->as.sequence.count = count;

        if (!jaiPCheck(p, TOK_RBRACKET)) {
            jaiPUnclosed(p, openIdx, "]", "generic argument list");
        } else {
            jaiPAdvance(p);
        }
        AstNode *g = jaiPNewNode(p, AST_INDEX,
                                 jaiSpanJoin(object->span,
                                             jaiPSpanSince(p, openIdx)));
        g->as.index.object = object;
        g->as.index.index = tuple;
        tuple->span = g->span;
        return g;
    }

    if (!jaiPCheck(p, TOK_RBRACKET)) {
        jaiPUnclosed(p, openIdx, "]", "index expression");
    } else {
        jaiPAdvance(p);
    }
    AstNode *n = jaiPNewNode(p, AST_INDEX, jaiSpanJoin(object->span, jaiPSpanSince(p, openIdx)));
    n->as.index.object = object;
    n->as.index.index = (start != NULL) ? start : errorExpr(p, n->span);
    return n;
}

static AstNode *parsePostfix(Parser *p) {
    AstNode *expr = jaiParsePrimary(p);

    for (;;) {
        if (jaiPCheck(p, TOK_DOT) || jaiPCheck(p, TOK_QUESTION_DOT)) {
            bool optional = jaiPCheck(p, TOK_QUESTION_DOT);
            jaiPAdvance(p);
            Token *nameTok = jaiPCur(p);
            const char *name = jaiPExpectIdentName(p, "a member name after `.`");
            if (name == NULL) break;
            AstNode *n = jaiPNewNode(p, optional ? AST_OPT_MEMBER : AST_MEMBER,
                                 jaiSpanJoin(expr->span, jaiPSpanOf(p, nameTok)));
            n->as.member.object = expr;
            n->as.member.name = name;
            n->as.member.cacheSlot = -1;
            expr = n;
        } else if (jaiPCheck(p, TOK_LPAREN)) {
            int openIdx = p->current;
            jaiPAdvance(p);
            AstNode *n = jaiPNewNode(p, AST_CALL, expr->span);
            n->as.call.callee = expr;
            parseArgs(p, openIdx, &n->as.call.args, &n->as.call.argCount);
            n->span = jaiSpanJoin(expr->span, jaiPSpanSince(p, openIdx));
            expr = n;
        } else if (jaiPCheck(p, TOK_LBRACKET)) {
            expr = parseSubscript(p, expr, p->current);
        } else {
            break;
        }
        if (p->panicMode) break;
    }
    return expr;
}

/* --- the precedence ladder ------------------------------------------ */

static AstNode *parsePower(Parser *p) {
    AstNode *left = parsePostfix(p);
    if (jaiPCheck(p, TOK_STAR_STAR)) {
        jaiPAdvance(p);
        /* right-associative, and binds looser than a unary on its right so
         * that `2 ** -1` parses */
        AstNode *right = parseUnary(p);
        return makeBinaryNode(p, AST_BINARY, OPK_POW, left, right);
    }
    return left;
}

/* `-9223372036854775808` is the one way to write INT64_MIN, so the `-` in front
 * of a TOKF_NEG_MAGNITUDE literal is part of the literal, not an operator: the
 * lexer already stored the value negated, and asking the VM for `-INT64_MIN`
 * would overflow. Matching the token pair — rather than folding any AST_INT_LIT
 * that happens to equal INT64_MIN — spends the sign exactly once, so
 * `-(-9223372036854775808)` and `- -0x8000000000000000` still negate INT64_MIN
 * and raise. Call with the cursor on the `-`; returns NULL when it is an
 * ordinary negation. */
AstNode *jaiParseNegMagnitude(Parser *p) {
    const Token *lit = jaiPTokAt(p, p->current + 1);
    if (lit->kind != TOK_INT || (lit->flags & TOKF_NEG_MAGNITUDE) == 0) return NULL;
    int startIdx = p->current;
    int64_t value = lit->v.intValue;
    jaiPAdvance(p);                                  /* the `-`      */
    jaiPAdvance(p);                                  /* the literal  */
    AstNode *n = jaiPNewNode(p, AST_INT_LIT, jaiPSpanSince(p, startIdx));
    n->as.intLit = value;
    return n;
}

static AstNode *parseUnary(Parser *p) {
    if (!jaiPEnterRecursion(p)) {
        p->exprDepth--;
        return errorExpr(p, jaiPCurSpan(p));
    }
    AstNode *result;
    TokenKind k = jaiPCurKind(p);
    if (k == TOK_MINUS && (result = jaiParseNegMagnitude(p)) != NULL) {
        p->exprDepth--;
        return result;
    }
    if (k == TOK_MINUS || k == TOK_PLUS || k == TOK_TILDE) {
        int startIdx = p->current;
        jaiPAdvance(p);
        AstNode *operand = parseUnary(p);
        OpKind op = (k == TOK_MINUS) ? OPK_NEG : (k == TOK_PLUS) ? OPK_POS : OPK_BNOT;
        result = jaiPNewNode(p, AST_UNARY, jaiPSpanSince(p, startIdx));
        result->as.unary.op = op;
        result->as.unary.operand = operand;
    } else {
        result = parsePower(p);
    }
    p->exprDepth--;
    return result;
}

static AstNode *parseMultiplicative(Parser *p) {
    AstNode *left = parseUnary(p);
    for (;;) {
        OpKind op;
        switch (jaiPCurKind(p)) {
        case TOK_STAR:         op = OPK_MUL; break;
        case TOK_SLASH:        op = OPK_DIV; break;
        case TOK_SLASH_SLASH:  op = OPK_FLOORDIV; break;
        case TOK_PERCENT:      op = OPK_MOD; break;
        case TOK_AT:           op = OPK_MATMUL; break;
        case TOK_STAR_PERCENT: op = OPK_MUL_WRAP; break;
        default: return left;
        }
        jaiPAdvance(p);
        left = makeBinaryNode(p, AST_BINARY, op, left, parseUnary(p));
        if (p->panicMode) return left;
    }
}

static AstNode *parseAdditive(Parser *p) {
    AstNode *left = parseMultiplicative(p);
    for (;;) {
        OpKind op;
        switch (jaiPCurKind(p)) {
        case TOK_PLUS:          op = OPK_ADD; break;
        case TOK_MINUS:         op = OPK_SUB; break;
        case TOK_PLUS_PERCENT:  op = OPK_ADD_WRAP; break;
        case TOK_MINUS_PERCENT: op = OPK_SUB_WRAP; break;
        default: return left;
        }
        jaiPAdvance(p);
        left = makeBinaryNode(p, AST_BINARY, op, left, parseMultiplicative(p));
        if (p->panicMode) return left;
    }
}

static AstNode *parseShift(Parser *p) {
    AstNode *left = parseAdditive(p);
    for (;;) {
        OpKind op;
        switch (jaiPCurKind(p)) {
        case TOK_SHL: op = OPK_SHL; break;
        case TOK_SHR: op = OPK_SHR; break;
        default: return left;
        }
        jaiPAdvance(p);
        left = makeBinaryNode(p, AST_BINARY, op, left, parseAdditive(p));
        if (p->panicMode) return left;
    }
}

static AstNode *parseBitAnd(Parser *p) {
    AstNode *left = parseShift(p);
    while (jaiPCheck(p, TOK_AMP)) {
        jaiPAdvance(p);
        left = makeBinaryNode(p, AST_BINARY, OPK_BAND, left, parseShift(p));
        if (p->panicMode) break;
    }
    return left;
}

static AstNode *parseBitXor(Parser *p) {
    AstNode *left = parseBitAnd(p);
    while (jaiPCheck(p, TOK_CARET)) {
        JaiSpan opSpan = jaiPCurSpan(p);
        jaiPAdvance(p);
        AstNode *right = parseBitAnd(p);
        /* Two decimal literals around `^` is almost always a Jaithon 2 power. */
        if (isDecimalLiteral(left) && isDecimalLiteral(right)) {
            JaiDiag *d = jaiPWarnAt(p, opSpan, W0105_DEPRECATED,
                                    "`^` is bitwise xor in Jaithon 3, not exponentiation");
            if (d != NULL) jaiDiagAddNote(d, "the power operator is `**`");
        }
        left = makeBinaryNode(p, AST_BINARY, OPK_BXOR, left, right);
        if (p->panicMode) break;
    }
    return left;
}

static AstNode *parseBitOr(Parser *p) {
    AstNode *left = parseBitXor(p);
    while (jaiPCheck(p, TOK_PIPE)) {
        jaiPAdvance(p);
        left = makeBinaryNode(p, AST_BINARY, OPK_BOR, left, parseBitXor(p));
        if (p->panicMode) break;
    }
    return left;
}

static AstNode *parseRange(Parser *p) {
    AstNode *left = parseBitOr(p);
    if (jaiPCheck(p, TOK_DOTDOT) || jaiPCheck(p, TOK_DOTDOT_EQ)) {
        bool inclusive = jaiPCheck(p, TOK_DOTDOT_EQ);
        jaiPAdvance(p);
        AstNode *stop = jaiParseStartsExpression(p) ? parseBitOr(p) : NULL;
        AstNode *n = jaiPNewNode(p, AST_RANGE,
                             jaiSpanJoin(left->span, stop != NULL ? stop->span : left->span));
        n->as.range.start = left;
        n->as.range.stop = stop;
        n->as.range.inclusive = inclusive;
        return n;
    }
    return left;
}

/* Level-6 operator, if one is here. `tokens` is how many tokens it spans;
 * `isLegacy` marks the Jaithon 2 word forms. */
static bool peekCompareOp(Parser *p, OpKind *outOp, int *outTokens, bool *isLegacy) {
    *isLegacy = false;
    *outTokens = 1;
    switch (jaiPCurKind(p)) {
    case TOK_LT:      *outOp = OPK_LT; return true;
    case TOK_LE:      *outOp = OPK_LE; return true;
    case TOK_GT:      *outOp = OPK_GT; return true;
    case TOK_GE:      *outOp = OPK_GE; return true;
    case TOK_EQ_EQ:   *outOp = OPK_EQ; return true;
    case TOK_BANG_EQ: *outOp = OPK_NE; return true;
    case TOK_KW_IN:   *outOp = OPK_IN; return true;
    case TOK_KW_IS:
        if (jaiPKindAt(p, 1) == TOK_KW_NOT) { *outOp = OPK_IS_NOT; *outTokens = 2; }
        else                            { *outOp = OPK_IS; }
        return true;
    case TOK_KW_NOT:
        if (jaiPKindAt(p, 1) == TOK_KW_IN) { *outOp = OPK_NOT_IN; *outTokens = 2; return true; }
        return false;
    case TOK_IDENT:
        if (jaiPCheckIdentText(p, 0, "eq"))  { *outOp = OPK_EQ; *isLegacy = true; return true; }
        if (jaiPCheckIdentText(p, 0, "ne") || jaiPCheckIdentText(p, 0, "neq")) {
            *outOp = OPK_NE; *isLegacy = true; return true;
        }
        return false;
    default:
        return false;
    }
}

/* `a < b <= c` is one chain node with `b` evaluated once (spec §4.2). */
static AstNode *parseComparison(Parser *p) {
    AstNode *first = parseRange(p);

    OpKind op;
    int opTokens;
    bool legacy;
    if (!peekCompareOp(p, &op, &opTokens, &legacy)) return first;

    NodeVec operands;
    OpVec ops;
    JAI_VEC_INIT(&operands);
    JAI_VEC_INIT(&ops);
    JAI_VEC_PUSH(AstNode *, &operands, first);

    while (peekCompareOp(p, &op, &opTokens, &legacy)) {
        if (legacy) {
            JaiDiag *d = jaiPErrorRecovered(p, jaiPCurSpan(p),
                                            "`%s` is not an operator in Jaithon 3",
                                        op == OPK_EQ ? "eq" : "ne");
            if (d != NULL) jaiDiagAddHelp(d, "write `%s`", op == OPK_EQ ? "==" : "!=");
        }
        for (int i = 0; i < opTokens; i++) jaiPAdvance(p);
        JAI_VEC_PUSH(OpKind, &ops, op);
        JAI_VEC_PUSH(AstNode *, &operands, parseRange(p));
        if (p->panicMode || jaiPCheck(p, TOK_EOF)) break;
    }

    AstNode *n;
    if (ops.count == 1) {
        n = makeBinaryNode(p, AST_BINARY, ops.data[0], operands.data[0], operands.data[1]);
        (void)JAI_VEC_TAKE(p, &operands, AstNode *);
        (void)JAI_VEC_TAKE(p, &ops, OpKind);
        return n;
    }
    AstNode *last = operands.data[operands.count - 1];
    n = jaiPNewNode(p, AST_COMPARE_CHAIN, jaiSpanJoin(first->span, last->span));
    n->as.chain.opCount = ops.count;
    n->as.chain.operands = JAI_VEC_TAKE(p, &operands, AstNode *);
    n->as.chain.ops = JAI_VEC_TAKE(p, &ops, OpKind);
    return n;
}

static AstNode *parseNot(Parser *p) {
    if (jaiPCheck(p, TOK_KW_NOT)) {
        if (!jaiPEnterRecursion(p)) {
            p->exprDepth--;
            return errorExpr(p, jaiPCurSpan(p));
        }
        int startIdx = p->current;
        jaiPAdvance(p);
        AstNode *operand = parseNot(p);
        AstNode *n = jaiPNewNode(p, AST_UNARY, jaiPSpanSince(p, startIdx));
        n->as.unary.op = OPK_NOT;
        n->as.unary.operand = operand;
        p->exprDepth--;
        return n;
    }
    return parseComparison(p);
}

static AstNode *parseAnd(Parser *p) {
    AstNode *left = parseNot(p);
    while (jaiPCheck(p, TOK_KW_AND)) {
        jaiPAdvance(p);
        left = makeBinaryNode(p, AST_LOGICAL, OPK_AND, left, parseNot(p));
        if (p->panicMode) break;
    }
    return left;
}

static AstNode *parseOr(Parser *p) {
    AstNode *left = parseAnd(p);
    while (jaiPCheck(p, TOK_KW_OR)) {
        jaiPAdvance(p);
        left = makeBinaryNode(p, AST_LOGICAL, OPK_OR, left, parseAnd(p));
        if (p->panicMode) break;
    }
    return left;
}

static AstNode *parseCoalesce(Parser *p) {
    AstNode *left = parseOr(p);
    while (jaiPCheck(p, TOK_QUESTION_QUESTION)) {
        jaiPAdvance(p);
        AstNode *right = parseOr(p);
        AstNode *n = jaiPNewNode(p, AST_COALESCE, jaiSpanJoin(left->span, right->span));
        n->as.coalesce.left = left;
        n->as.coalesce.right = right;
        left = n;
        if (p->panicMode) break;
    }
    return left;
}

AstNode *jaiParseExpression(Parser *p) {
    if (!jaiPEnterRecursion(p)) {
        p->exprDepth--;
        return errorExpr(p, jaiPCurSpan(p));
    }
    AstNode *cond = parseCoalesce(p);
    if (jaiPCheck(p, TOK_QUESTION)) {
        jaiPAdvance(p);
        AstNode *thenExpr = jaiParseExpression(p);
        jaiPExpect(p, TOK_COLON, "`:` in a conditional expression");
        AstNode *elseExpr = jaiParseExpression(p);
        AstNode *n = jaiPNewNode(p, AST_TERNARY, jaiSpanJoin(cond->span, elseExpr->span));
        n->as.ternary.cond = cond;
        n->as.ternary.thenExpr = thenExpr;
        n->as.ternary.elseExpr = elseExpr;
        p->exprDepth--;
        return n;
    }
    p->exprDepth--;
    return cond;
}
