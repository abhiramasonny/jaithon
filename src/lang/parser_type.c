/* parser_type.c — the two syntactic categories that are neither expressions
 * nor statements: type syntax (spec §2) and patterns (spec §5.3).
 *
 * They share a file because they share a shape. Both are read where an
 * expression could otherwise stand, both are built from names, literals and
 * bracketed groups rather than from operators, and a pattern's literal arms
 * are parsed by calling back into the expression file.
 */

#include "parser_internal.h"

/* ------------------------------------------------------------------ */
/* Types                                                                */
/* ------------------------------------------------------------------ */

static AstType *newType(Parser *p, AstTypeKind kind, JaiSpan span) {
    return jaiAstTypeNew(p->ast, kind, span);
}

static AstType *parsePrimaryType(Parser *p) {
    int startIdx = p->current;

    if (jaiPCheck(p, TOK_KW_FN)) {
        jaiPAdvance(p);
        AstType *t = NULL;
        TypeVec params;
        JAI_VEC_INIT(&params);
        jaiPExpect(p, TOK_LPAREN, "`(` after `fn` in a function type");
        jaiPSkipNewlines(p);
        if (!jaiPCheck(p, TOK_RPAREN)) {
            do {
                jaiPSkipNewlines(p);
                JAI_VEC_PUSH(AstType *, &params, jaiParseType(p));
                jaiPSkipNewlines(p);
            } while (jaiPMatch(p, TOK_COMMA) && !jaiPCheck(p, TOK_RPAREN) &&
                     !jaiPCheck(p, TOK_EOF));
        }
        jaiPExpect(p, TOK_RPAREN, "`)` to close the parameter list");
        AstType *ret = NULL;
        if (jaiPMatch(p, TOK_ARROW)) {
            ret = jaiParseType(p);
        } else {
            jaiPErrorAtCur(p, E0101_EXPECTED_TOKEN,
                           "expected `->` and a return type in a function type");
            ret = newType(p, TYPE_INFER, jaiPCurSpan(p));
        }
        t = newType(p, TYPE_FN, jaiPSpanSince(p, startIdx));
        t->argCount = params.count;
        t->args = JAI_VEC_TAKE(p, &params, AstType *);
        t->inner = ret;
        return t;
    }

    if (jaiPCheck(p, TOK_LPAREN)) {
        jaiPAdvance(p);
        TypeVec members;
        JAI_VEC_INIT(&members);
        bool trailingComma = false;
        jaiPSkipNewlines(p);
        if (!jaiPCheck(p, TOK_RPAREN)) {
            for (;;) {
                jaiPSkipNewlines(p);
                JAI_VEC_PUSH(AstType *, &members, jaiParseType(p));
                jaiPSkipNewlines(p);
                if (!jaiPMatch(p, TOK_COMMA)) break;
                jaiPSkipNewlines(p);
                if (jaiPCheck(p, TOK_RPAREN)) { trailingComma = true; break; }
                if (jaiPCheck(p, TOK_EOF)) break;
            }
        }
        jaiPExpect(p, TOK_RPAREN, "`)` to close the tuple type");
        if (members.count == 1 && !trailingComma) {
            AstType *only = members.data[0];
            (void)JAI_VEC_TAKE(p, &members, AstType *);
            return only;
        }
        AstType *t = newType(p, TYPE_TUPLE, jaiPSpanSince(p, startIdx));
        t->argCount = members.count;
        t->args = JAI_VEC_TAKE(p, &members, AstType *);
        return t;
    }

    if (jaiPCheck(p, TOK_UNDERSCORE)) {
        jaiPAdvance(p);
        return newType(p, TYPE_INFER, jaiPSpanSince(p, startIdx));
    }

    if (jaiPCheck(p, TOK_KW_NULL)) {
        jaiPAdvance(p);
        AstType *t = newType(p, TYPE_NAME, jaiPSpanSince(p, startIdx));
        t->name = "null";
        return t;
    }

    if (!jaiPCheck(p, TOK_IDENT)) {
        char found[80];
        jaiPDescribeToken(p, jaiPCur(p), found, sizeof found);
        jaiPErrorAt(p, jaiPCurSpan(p), E0105_EXPECTED_TYPE, "expected a type, found %s", found);
        return newType(p, TYPE_INFER, jaiPCurSpan(p));
    }

    /* Dotted name: module-qualified types such as `std.math.Vec`. */
    JaiBuf name;
    jaiBufInit(&name);
    size_t len = 0;
    const char *text = jaiPRawText(p, jaiPAdvance(p), &len);
    if (text != NULL) jaiBufAppend(&name, text, len);
    while (jaiPCheck(p, TOK_DOT) && jaiPKindAt(p, 1) == TOK_IDENT) {
        jaiPAdvance(p);
        jaiBufPush(&name, '.');
        text = jaiPRawText(p, jaiPAdvance(p), &len);
        if (text != NULL) jaiBufAppend(&name, text, len);
    }
    const char *interned = jaiPInternText(p, (const char *)name.data, name.count);
    jaiBufFree(&name);

    if (jaiPCheck(p, TOK_LBRACKET)) {
        int openIdx = p->current;
        jaiPAdvance(p);
        TypeVec args;
        JAI_VEC_INIT(&args);
        jaiPSkipNewlines(p);
        if (!jaiPCheck(p, TOK_RBRACKET)) {
            for (;;) {
                jaiPSkipNewlines(p);
                JAI_VEC_PUSH(AstType *, &args, jaiParseType(p));
                jaiPSkipNewlines(p);
                if (!jaiPMatch(p, TOK_COMMA)) break;
                jaiPSkipNewlines(p);
                if (jaiPCheck(p, TOK_RBRACKET) || jaiPCheck(p, TOK_EOF)) break;
            }
        }
        if (!jaiPCheck(p, TOK_RBRACKET)) {
            jaiPUnclosed(p, openIdx, "]", "type argument list");
        } else {
            jaiPAdvance(p);
        }
        AstType *t = newType(p, TYPE_GENERIC, jaiPSpanSince(p, startIdx));
        t->name = interned;
        t->argCount = args.count;
        t->args = JAI_VEC_TAKE(p, &args, AstType *);
        return t;
    }

    AstType *t = newType(p, TYPE_NAME, jaiPSpanSince(p, startIdx));
    t->name = interned;
    return t;
}

static AstType *parsePostfixType(Parser *p) {
    int startIdx = p->current;
    AstType *t = parsePrimaryType(p);
    while (jaiPCheck(p, TOK_QUESTION)) {
        jaiPAdvance(p);
        AstType *opt = newType(p, TYPE_OPTIONAL, jaiPSpanSince(p, startIdx));
        opt->inner = t;
        t = opt;
    }
    return t;
}

AstType *jaiParseType(Parser *p) {
    if (!jaiPEnterRecursion(p)) {
        p->exprDepth--;
        return newType(p, TYPE_INFER, jaiPCurSpan(p));
    }
    int startIdx = p->current;
    AstType *first = parsePostfixType(p);
    if (!jaiPCheck(p, TOK_PIPE)) {
        p->exprDepth--;
        return first;
    }
    TypeVec parts;
    JAI_VEC_INIT(&parts);
    JAI_VEC_PUSH(AstType *, &parts, first);
    while (jaiPMatch(p, TOK_PIPE)) {
        JAI_VEC_PUSH(AstType *, &parts, parsePostfixType(p));
        if (jaiPCheck(p, TOK_EOF)) break;
    }
    AstType *u = newType(p, TYPE_UNION, jaiPSpanSince(p, startIdx));
    u->argCount = parts.count;
    u->args = JAI_VEC_TAKE(p, &parts, AstType *);
    p->exprDepth--;
    return u;
}

/* `catch e: A | B` and `class C: A, B` both flatten into a type list. */
void jaiParsePushTypeFlattened(TypeVec *out, AstType *t) {
    if (t != NULL && t->kind == TYPE_UNION) {
        for (int i = 0; i < t->argCount; i++) JAI_VEC_PUSH(AstType *, out, t->args[i]);
    } else if (t != NULL) {
        JAI_VEC_PUSH(AstType *, out, t);
    }
}

/* ------------------------------------------------------------------ */
/* Patterns                                                             */
/* ------------------------------------------------------------------ */

static AstNode *parsePatternPrimary(Parser *p);

/* Elements of a `(...)` / `[...]` pattern, including one `...rest`. */
static void parsePatternList(Parser *p, TokenKind closer, const char *closeText,
                             const char *what, int openIdx,
                             AstNode ***outElems, int *outCount, int *outRestIndex,
                             bool *outSawComma) {
    NodeVec elems;
    JAI_VEC_INIT(&elems);
    int restIndex = -1;
    bool sawComma = false;

    jaiPSkipNewlines(p);
    if (!jaiPCheck(p, closer)) {
        for (;;) {
            jaiPSkipNewlines(p);
            if (jaiPCheck(p, TOK_ELLIPSIS)) {
                JaiSpan restSpan = jaiPCurSpan(p);
                jaiPAdvance(p);
                AstNode *rest;
                if (jaiPCheck(p, TOK_IDENT)) {
                    Token *nameTok = jaiPCur(p);
                    rest = jaiPNewNode(
                        p, AST_PAT_BIND,
                        jaiSpanJoin(restSpan, jaiPSpanOf(p, nameTok)));
                    rest->as.patBind.name = jaiPInternToken(p, jaiPAdvance(p));
                } else {
                    rest = jaiPNewNode(p, AST_PAT_WILDCARD, restSpan);
                }
                if (restIndex >= 0) {
                    jaiPErrorAt(p, rest->span, E0110_INVALID_PATTERN,
                            "a pattern may bind only one `...` rest element");
                } else {
                    restIndex = elems.count;
                }
                JAI_VEC_PUSH(AstNode *, &elems, rest);
            } else {
                JAI_VEC_PUSH(AstNode *, &elems, jaiParsePattern(p));
            }
            jaiPSkipNewlines(p);
            if (!jaiPMatch(p, TOK_COMMA)) break;
            sawComma = true;
            jaiPSkipNewlines(p);
            if (jaiPCheck(p, closer) || jaiPCheck(p, TOK_EOF)) break;
            if (p->panicMode) break;
        }
    }
    if (!jaiPCheck(p, closer)) {
        jaiPUnclosed(p, openIdx, closeText, what);
    } else {
        jaiPAdvance(p);
    }
    *outCount = elems.count;
    *outElems = JAI_VEC_TAKE(p, &elems, AstNode *);
    *outRestIndex = restIndex;
    if (outSawComma != NULL) *outSawComma = sawComma;
}

static bool startsPatternLiteral(Parser *p) {
    switch (jaiPCurKind(p)) {
    case TOK_INT: case TOK_FLOAT: case TOK_STRING: case TOK_FSTRING_START:
    case TOK_KW_TRUE: case TOK_KW_FALSE: case TOK_KW_NULL: case TOK_MINUS:
        return true;
    default:
        return false;
    }
}

static AstNode *parsePatternLiteralValue(Parser *p) {
    int startIdx = p->current;
    if (jaiPCheck(p, TOK_MINUS)) {
        AstNode *min = jaiParseNegMagnitude(p);     /* `-9223372036854775808` */
        if (min != NULL) return min;
        jaiPAdvance(p);
        AstNode *inner = jaiParsePrimary(p);
        AstNode *n = jaiPNewNode(p, AST_UNARY, jaiPSpanSince(p, startIdx));
        n->as.unary.op = OPK_NEG;
        n->as.unary.operand = inner;
        return n;
    }
    return jaiParsePrimary(p);
}

static AstNode *parsePatternPrimary(Parser *p) {
    int startIdx = p->current;

    if (jaiPCheck(p, TOK_UNDERSCORE)) {
        jaiPAdvance(p);
        return jaiPNewNode(p, AST_PAT_WILDCARD, jaiPSpanSince(p, startIdx));
    }

    if (jaiPCheck(p, TOK_LPAREN)) {
        int openIdx = p->current;
        jaiPAdvance(p);
        AstNode **elems;
        int count, restIndex;
        bool sawComma;
        parsePatternList(p, TOK_RPAREN, ")", "tuple pattern", openIdx,
                         &elems, &count, &restIndex, &sawComma);
        if (count == 1 && restIndex < 0 && !sawComma) return elems[0];
        AstNode *n = jaiPNewNode(p, AST_PAT_TUPLE, jaiPSpanSince(p, startIdx));
        n->as.patSeq.elems = elems;
        n->as.patSeq.count = count;
        n->as.patSeq.restIndex = restIndex;
        return n;
    }

    if (jaiPCheck(p, TOK_LBRACKET)) {
        int openIdx = p->current;
        jaiPAdvance(p);
        AstNode **elems;
        int count, restIndex;
        parsePatternList(p, TOK_RBRACKET, "]", "list pattern", openIdx,
                         &elems, &count, &restIndex, NULL);
        AstNode *n = jaiPNewNode(p, AST_PAT_LIST, jaiPSpanSince(p, startIdx));
        n->as.patSeq.elems = elems;
        n->as.patSeq.count = count;
        n->as.patSeq.restIndex = restIndex;
        return n;
    }

    if (startsPatternLiteral(p)) {
        AstNode *lo = parsePatternLiteralValue(p);
        if (jaiPCheck(p, TOK_DOTDOT) || jaiPCheck(p, TOK_DOTDOT_EQ)) {
            bool inclusive = jaiPCheck(p, TOK_DOTDOT_EQ);
            jaiPAdvance(p);
            AstNode *hi = startsPatternLiteral(p) ? parsePatternLiteralValue(p) : NULL;
            AstNode *n = jaiPNewNode(p, AST_PAT_RANGE, jaiPSpanSince(p, startIdx));
            n->as.patRange.lo = lo;
            n->as.patRange.hi = hi;
            n->as.patRange.inclusive = inclusive;
            return n;
        }
        AstNode *n = jaiPNewNode(p, AST_PAT_LITERAL, jaiPSpanSince(p, startIdx));
        n->as.patLiteral.value = lo;
        return n;
    }

    if (jaiPCheck(p, TOK_IDENT)) {
        const char *name = jaiPInternToken(p, jaiPAdvance(p));

        /* `Shape.Circle(r)` — enum variant, with optional payload. */
        if (jaiPCheck(p, TOK_DOT)) {
            jaiPAdvance(p);
            const char *variant = jaiPExpectIdentName(p, "an enum variant name");
            if (variant == NULL) {
                return jaiPNewNode(p, AST_PAT_WILDCARD,
                                   jaiPSpanSince(p, startIdx));
            }
            AstNode *n = jaiPNewNode(p, AST_PAT_ENUM, jaiPSpanSince(p, startIdx));
            n->as.patClass.typeName = name;
            n->as.patClass.variantName = variant;
            if (jaiPCheck(p, TOK_LPAREN)) {
                int openIdx = p->current;
                jaiPAdvance(p);
                AstNode **elems;
                int count, restIndex;
                parsePatternList(p, TOK_RPAREN, ")", "variant pattern", openIdx,
                                 &elems, &count, &restIndex, NULL);
                n->as.patClass.subPatterns = elems;
                n->as.patClass.count = count;
                n->span = jaiPSpanSince(p, startIdx);
            }
            return n;
        }

        /* `Point{x, y}` / `Point{x: 1, y}` — class pattern. */
        if (jaiPCheck(p, TOK_LBRACE)) {
            int openIdx = p->current;
            jaiPAdvance(p);
            NodeVec subs;
            NameVec fields;
            JAI_VEC_INIT(&subs);
            JAI_VEC_INIT(&fields);
            jaiPSkipNewlines(p);
            if (!jaiPCheck(p, TOK_RBRACE)) {
                for (;;) {
                    jaiPSkipNewlines(p);
                    Token *fieldTok = jaiPCur(p);
                    const char *field = jaiPExpectIdentName(p, "a field name in a class pattern");
                    if (field == NULL) break;
                    AstNode *sub;
                    if (jaiPMatch(p, TOK_COLON)) {
                        sub = jaiParsePattern(p);
                    } else {
                        sub = jaiPNewNode(p, AST_PAT_BIND, jaiPSpanOf(p, fieldTok));
                        sub->as.patBind.name = field;
                    }
                    JAI_VEC_PUSH(const char *, &fields, field);
                    JAI_VEC_PUSH(AstNode *, &subs, sub);
                    jaiPSkipNewlines(p);
                    if (!jaiPMatch(p, TOK_COMMA)) break;
                    jaiPSkipNewlines(p);
                    if (jaiPCheck(p, TOK_RBRACE) || jaiPCheck(p, TOK_EOF)) break;
                }
            }
            if (!jaiPCheck(p, TOK_RBRACE)) {
                jaiPUnclosed(p, openIdx, "}", "class pattern");
            } else {
                jaiPAdvance(p);
            }
            AstNode *n = jaiPNewNode(p, AST_PAT_CLASS, jaiPSpanSince(p, startIdx));
            n->as.patClass.typeName = name;
            n->as.patClass.count = subs.count;
            n->as.patClass.subPatterns = JAI_VEC_TAKE(p, &subs, AstNode *);
            n->as.patClass.fieldNames = JAI_VEC_TAKE(p, &fields, const char *);
            return n;
        }

        /* `Some(x)` — positional class pattern. */
        if (jaiPCheck(p, TOK_LPAREN)) {
            int openIdx = p->current;
            jaiPAdvance(p);
            AstNode **elems;
            int count, restIndex;
            parsePatternList(p, TOK_RPAREN, ")", "class pattern", openIdx,
                             &elems, &count, &restIndex, NULL);
            AstNode *n = jaiPNewNode(p, AST_PAT_CLASS, jaiPSpanSince(p, startIdx));
            n->as.patClass.typeName = name;
            n->as.patClass.subPatterns = elems;
            n->as.patClass.count = count;
            return n;
        }

        AstNode *n = jaiPNewNode(p, AST_PAT_BIND, jaiPSpanSince(p, startIdx));
        n->as.patBind.name = name;
        return n;
    }

    char found[80];
    jaiPDescribeToken(p, jaiPCur(p), found, sizeof found);
    jaiPErrorAt(p, jaiPCurSpan(p), E0110_INVALID_PATTERN, "expected a pattern, found %s", found);
    return jaiPNewNode(p, AST_PAT_WILDCARD, jaiPCurSpan(p));
}

AstNode *jaiParsePattern(Parser *p) {
    if (!jaiPEnterRecursion(p)) {
        p->exprDepth--;
        return jaiPNewNode(p, AST_PAT_WILDCARD, jaiPCurSpan(p));
    }
    int startIdx = p->current;
    AstNode *first = parsePatternPrimary(p);
    if (!jaiPCheck(p, TOK_PIPE)) {
        p->exprDepth--;
        return first;
    }
    NodeVec alts;
    JAI_VEC_INIT(&alts);
    JAI_VEC_PUSH(AstNode *, &alts, first);
    while (jaiPMatch(p, TOK_PIPE)) {
        JAI_VEC_PUSH(AstNode *, &alts, parsePatternPrimary(p));
        if (p->panicMode || jaiPCheck(p, TOK_EOF)) break;
    }
    AstNode *n = jaiPNewNode(p, AST_PAT_OR, jaiPSpanSince(p, startIdx));
    n->as.patSeq.count = alts.count;
    n->as.patSeq.elems = JAI_VEC_TAKE(p, &alts, AstNode *);
    n->as.patSeq.restIndex = -1;
    p->exprDepth--;
    return n;
}

/* Reinterpret an already-parsed expression as an assignment pattern, for
 * `(a, b) = point`. Returns NULL when the expression is not a valid pattern. */
AstNode *jaiParseExprToPattern(Parser *p, AstNode *e) {
    switch (e->kind) {
    case AST_IDENT: {
        AstNode *n = jaiPNewNode(p, AST_PAT_BIND, e->span);
        n->as.patBind.name = e->as.ident.name;
        return n;
    }
    case AST_TUPLE_LIT:
    case AST_LIST_LIT: {
        NodeVec elems;
        JAI_VEC_INIT(&elems);
        for (int i = 0; i < e->as.sequence.count; i++) {
            AstNode *sub = jaiParseExprToPattern(p, e->as.sequence.items[i]);
            if (sub == NULL) {
                (void)JAI_VEC_TAKE(p, &elems, AstNode *);
                return NULL;
            }
            JAI_VEC_PUSH(AstNode *, &elems, sub);
        }
        AstNode *n = jaiPNewNode(
            p, e->kind == AST_TUPLE_LIT ? AST_PAT_TUPLE : AST_PAT_LIST, e->span);
        n->as.patSeq.count = elems.count;
        n->as.patSeq.elems = JAI_VEC_TAKE(p, &elems, AstNode *);
        n->as.patSeq.restIndex = -1;
        return n;
    }
    default:
        return NULL;
    }
}
