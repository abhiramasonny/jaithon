/* parser_decl.c — declarations: imports, functions, classes, traits, enums,
 * type aliases and bindings (spec §6, §7).
 *
 * The longest productions in the grammar, and the ones that carry the most
 * recovery: a malformed parameter list or class body still has to produce a
 * tree the checker can report against, so most of the length here is a
 * diagnostic with a fix-it rather than grammar.
 */

#include "parser_internal.h"

/* ------------------------------------------------------------------ */
/* Declarations                                                         */
/* ------------------------------------------------------------------ */

/* §1.4: `__name__` is reserved for the implementation. */
static void checkReservedName(Parser *p, const char *name, JaiSpan span) {
    if (name == NULL) return;
    size_t len = strlen(name);
    if (len < 5 || name[0] != '_' || name[1] != '_') return;
    if (name[len - 1] != '_' || name[len - 2] != '_') return;
    JaiDiag *d = jaiPErrorRecoveredCode(p, span, E0114_RESERVED_IDENTIFIER,
                                    "`%s` is reserved for the implementation", name);
    if (d != NULL) jaiDiagAddHelp(d, "names of the form `__x__` may not be declared here");
}

/* §7.2: `Self` is in scope in every class, trait and enum body as a name for
 * the implementing type, so nothing may declare a type of that name — a
 * `class Self` would be reachable only from outside its own body. */
static void checkReservedTypeName(Parser *p, const char *name, JaiSpan span) {
    if (name == NULL || strcmp(name, "Self") != 0) return;
    JaiDiag *d = jaiPErrorRecoveredCode(p, span, E0114_RESERVED_IDENTIFIER,
                                    "`Self` is reserved: it names the implementing type");
    if (d != NULL)
        jaiDiagAddHelp(d, "`Self` is already in scope inside every class, trait "
                          "and enum body");
}

static void parseGenerics(Parser *p, AstGeneric **outGenerics, int *outCount) {
    *outGenerics = NULL;
    *outCount = 0;
    if (!jaiPCheck(p, TOK_LBRACKET)) return;

    int openIdx = p->current;
    jaiPAdvance(p);
    GenericVec generics;
    JAI_VEC_INIT(&generics);
    if (!jaiPCheck(p, TOK_RBRACKET)) {
        for (;;) {
            AstGeneric g;
            g.name = NULL;
            g.bound = NULL;
            g.span = jaiPCurSpan(p);
            Token *nameTok = jaiPCur(p);
            g.name = jaiPExpectIdentName(p, "a type parameter name");
            if (g.name == NULL) break;
            g.span = jaiPSpanOf(p, nameTok);
            checkReservedTypeName(p, g.name, g.span);
            if (jaiPMatch(p, TOK_COLON)) g.bound = jaiParseType(p);
            JAI_VEC_PUSH(AstGeneric, &generics, g);
            if (!jaiPMatch(p, TOK_COMMA)) break;
            if (jaiPCheck(p, TOK_RBRACKET) || jaiPCheck(p, TOK_EOF)) break;
        }
    }
    if (!jaiPCheck(p, TOK_RBRACKET)) {
        jaiPUnclosed(p, openIdx, "]", "type parameter list");
    } else {
        jaiPAdvance(p);
    }
    *outCount = generics.count;
    *outGenerics = JAI_VEC_TAKE(p, &generics, AstGeneric);
}

/* Parses a parameter list up to and including the closing `)`. */
static void parseParams(Parser *p, int openIdx, AstParam **outParams, int *outCount) {
    ParamVec params;
    JAI_VEC_INIT(&params);
    bool sawRest = false;
    bool sawDefault = false;
    JaiSpan restSpan = JAI_SPAN_NONE;
    JaiSpan defaultSpan = JAI_SPAN_NONE;

    jaiPSkipNewlines(p);
    if (!jaiPCheck(p, TOK_RPAREN)) {
        for (;;) {
            jaiPSkipNewlines(p);
            int paramStart = p->current;
            AstParam param;
            memset(&param, 0, sizeof param);

            if (jaiPCheck(p, TOK_ELLIPSIS)) {
                jaiPAdvance(p);
                param.isVariadic = true;
            } else if (jaiPCheck(p, TOK_STAR_STAR)) {
                jaiPAdvance(p);
                param.isKwRest = true;
            }

            if (jaiPCheck(p, TOK_KW_SELF)) {
                param.name = "self";
                param.span = jaiPCurSpan(p);
                jaiPAdvance(p);
            } else {
                Token *nameTok = jaiPCur(p);
                param.name = jaiPExpectIdentName(p, "a parameter name");
                if (param.name == NULL) break;
                param.span = jaiPSpanOf(p, nameTok);
            }

            if (jaiPMatch(p, TOK_COLON)) param.type = jaiParseType(p);
            if (jaiPMatch(p, TOK_EQ)) param.defaultValue = jaiParseExpression(p);
            param.span = jaiPSpanSince(p, paramStart);

            for (int i = 0; i < params.count; i++) {
                if (params.data[i].name != NULL && strcmp(params.data[i].name, param.name) == 0) {
                    JaiDiag *d = jaiPErrorRecoveredCode(p, param.span, E0111_DUPLICATE_PARAMETER,
                                                    "duplicate parameter `%s`", param.name);
                    if (d != NULL) jaiDiagAddLabel(d, params.data[i].span, "first declared here");
                    break;
                }
            }
            if (sawRest) {
                JaiDiag *d = jaiPErrorRecoveredCode(p, param.span, E0112_PARAM_AFTER_VARIADIC,
                                                "parameter `%s` cannot follow a rest parameter",
                                                param.name);
                if (d != NULL) jaiDiagAddLabel(d, restSpan, "rest parameter declared here");
            } else if (sawDefault && param.defaultValue == NULL &&
                       !param.isVariadic && !param.isKwRest) {
                JaiDiag *d = jaiPErrorRecoveredCode(p, param.span, E0113_DEFAULT_BEFORE_REQUIRED,
                                                "required parameter `%s` follows a defaulted parameter",
                                                param.name);
                if (d != NULL) jaiDiagAddLabel(d, defaultSpan, "this parameter has a default value");
            }
            if (param.isVariadic || param.isKwRest) {
                sawRest = true;
                restSpan = param.span;
            }
            if (param.defaultValue != NULL) {
                sawDefault = true;
                defaultSpan = param.span;
            }

            JAI_VEC_PUSH(AstParam, &params, param);
            jaiPSkipNewlines(p);
            if (!jaiPMatch(p, TOK_COMMA)) break;
            jaiPSkipNewlines(p);
            if (jaiPCheck(p, TOK_RPAREN) || jaiPCheck(p, TOK_EOF) || p->panicMode) break;
        }
    }
    if (!jaiPCheck(p, TOK_RPAREN)) {
        jaiPUnclosed(p, openIdx, ")", "parameter list");
    } else {
        jaiPAdvance(p);
    }
    *outCount = params.count;
    *outParams = JAI_VEC_TAKE(p, &params, AstParam);
}

/* Shared by `fn` declarations, methods, trait members, and anonymous `fn`.
 * The current token is the `fn` keyword (or the legacy spelling). */
AstNode *jaiParseFnRest(Parser *p, AstVisibility vis, bool isStatic, int startIdx,
                            bool named, bool bodyOptional) {
    jaiPAdvance(p);   /* fn */

    const char *name = NULL;
    if (named) {
        name = jaiPExpectIdentName(p, "a function name");
        if (name == NULL) return NULL;
    }

    AstGeneric *generics = NULL;
    int genericCount = 0;
    parseGenerics(p, &generics, &genericCount);

    AstParam *params = NULL;
    int paramCount = 0;
    int openIdx = p->current;
    if (jaiPExpect(p, TOK_LPAREN, "`(` to open the parameter list")) {
        parseParams(p, openIdx, &params, &paramCount);
    }

    AstType *returnType = jaiPMatch(p, TOK_ARROW) ? jaiParseType(p) : NULL;

    AstNode *body = NULL;
    bool exprBody = false;
    int savedYield = jaiPYieldCount;
    jaiPYieldCount = 0;
    p->fnDepth++;
    int savedLoopDepth = p->loopDepth;
    p->loopDepth = 0;

    if (bodyOptional && !jaiPCheck(p, TOK_LBRACE)) {
        body = NULL;
    } else if (jaiPCheck(p, TOK_FAT_ARROW)) {
        /* Not in the grammar, but a common typo from lambda syntax. */
        JaiDiag *d = jaiPErrorAt(p, jaiPCurSpan(p), E0104_EXPECTED_BLOCK,
                             "a function body is a block, not `=>`");
        if (d != NULL) jaiDiagAddHelp(d, "write `{ return <expr> }`, or use a lambda `|x| <expr>`");
        jaiPAdvance(p);
        body = jaiParseExpression(p);
        exprBody = true;
    } else {
        body = jaiParseExpectBlock(p, "function");
    }

    p->loopDepth = savedLoopDepth;
    p->fnDepth--;
    bool isGenerator = jaiPYieldCount > 0;
    jaiPYieldCount = savedYield;

    AstNode *n = jaiPNewNode(p, named ? AST_FN_DECL : AST_ANON_FN, jaiPSpanSince(p, startIdx));
    n->as.fn.name = name;
    n->as.fn.generics = generics;
    n->as.fn.genericCount = genericCount;
    n->as.fn.params = params;
    n->as.fn.paramCount = paramCount;
    n->as.fn.returnType = returnType;
    n->as.fn.body = body;
    n->as.fn.visibility = vis;
    n->as.fn.isStatic = isStatic;
    n->as.fn.isExprBody = exprBody;
    n->as.fn.isGenerator = isGenerator;
    n->as.fn.isAsync = false;
    return n;
}

/* `get area() -> float { ... }` / `set width(v: float) { ... }` */
static AstNode *parseAccessor(Parser *p, AstVisibility vis, bool isStatic, bool isSetter) {
    int startIdx = p->current;
    jaiPAdvance(p);   /* `get` / `set` */
    const char *name = jaiPExpectIdentName(p, "a property name");
    if (name == NULL) return NULL;

    AstParam *params = NULL;
    int paramCount = 0;
    int openIdx = p->current;
    if (jaiPExpect(p, TOK_LPAREN, "`(` after the property name")) {
        parseParams(p, openIdx, &params, &paramCount);
    }
    AstType *returnType = jaiPMatch(p, TOK_ARROW) ? jaiParseType(p) : NULL;

    int savedYield = jaiPYieldCount;
    jaiPYieldCount = 0;
    p->fnDepth++;
    AstNode *body = jaiParseExpectBlock(p, isSetter ? "set" : "get");
    p->fnDepth--;
    jaiPYieldCount = savedYield;

    AstNode *n = jaiPNewNode(p, AST_FN_DECL, jaiPSpanSince(p, startIdx));
    n->as.fn.name = name;
    n->as.fn.params = params;
    n->as.fn.paramCount = paramCount;
    n->as.fn.returnType = returnType;
    n->as.fn.body = body;
    n->as.fn.visibility = vis;
    n->as.fn.isStatic = isStatic;
    return n;
}

static AstNode *parseVarDecl(Parser *p, AstVisibility vis, int startIdx) {
    VarDeclKind kind;
    switch (jaiPCurKind(p)) {
    case TOK_KW_LET:   kind = VD_LET; break;
    case TOK_KW_VAR:   kind = VD_VAR; break;
    default:           kind = VD_CONST; break;
    }
    jaiPAdvance(p);

    AstNode *pattern = jaiParsePattern(p);
    if (pattern != NULL && pattern->kind == AST_PAT_BIND) {
        checkReservedName(p, pattern->as.patBind.name, pattern->span);
    }
    AstType *declared = jaiPMatch(p, TOK_COLON) ? jaiParseType(p) : NULL;
    AstNode *init = NULL;
    if (jaiPMatch(p, TOK_EQ)) {
        init = jaiParseExpression(p);
    } else if (kind != VD_VAR && !p->panicMode) {
        JaiDiag *d = jaiPErrorAt(p, jaiPSpanSince(p, startIdx), E0304_MISSING_INITIALIZER,
                             "`%s` bindings must be initialised",
                             kind == VD_LET ? "let" : "const");
        if (d != NULL) jaiDiagAddHelp(d, "add `= <expression>`, or use `var` to declare it uninitialised");
    }
    jaiPEndStatement(p);

    AstNode *n = jaiPNewNode(p, AST_VAR_DECL, jaiPSpanSince(p, startIdx));
    n->as.varDecl.kind = kind;
    n->as.varDecl.pattern = pattern;
    n->as.varDecl.declaredType = declared;
    n->as.varDecl.init = init;
    n->as.varDecl.visibility = vis;
    return n;
}

static AstNode *parseClassDecl(Parser *p, AstVisibility vis, int startIdx) {
    jaiPAdvance(p);   /* class */
    Token *declNameTok = jaiPCur(p);
    const char *name = jaiPExpectIdentName(p, "a class name");
    if (name == NULL) return NULL;
    checkReservedTypeName(p, name, jaiPSpanOf(p, declNameTok));

    AstGeneric *generics = NULL;
    int genericCount = 0;
    parseGenerics(p, &generics, &genericCount);

    AstType *superclass = jaiPMatch(p, TOK_KW_EXTENDS) ? jaiParseType(p) : NULL;

    TypeVec traits;
    JAI_VEC_INIT(&traits);
    if (jaiPMatch(p, TOK_COLON)) {
        for (;;) {
            jaiParsePushTypeFlattened(&traits, jaiParseType(p));
            if (!jaiPMatch(p, TOK_COMMA)) break;
            if (jaiPCheck(p, TOK_EOF) || p->panicMode) break;
        }
    }

    FieldVec fields;
    NodeVec methods, getters, setters;
    JAI_VEC_INIT(&fields);
    JAI_VEC_INIT(&methods);
    JAI_VEC_INIT(&getters);
    JAI_VEC_INIT(&setters);

    int openIdx = p->current;
    if (jaiPExpect(p, TOK_LBRACE, "`{` to open the class body")) {
        jaiPSkipNewlines(p);
        while (!jaiPCheck(p, TOK_RBRACE) && !jaiPCheck(p, TOK_EOF)) {
            int before = p->current;
            AstVisibility memberVis = AST_VIS_PRIVATE;
            if (jaiPMatch(p, TOK_KW_PUB))       memberVis = AST_VIS_PUBLIC;
            else if (jaiPMatch(p, TOK_KW_PROT)) memberVis = AST_VIS_PROTECTED;
            bool memberStatic = jaiPMatch(p, TOK_KW_STATIC);

            if (jaiPCheck(p, TOK_KW_VAR) || jaiPCheck(p, TOK_KW_LET)) {
                bool isLet = jaiPCheck(p, TOK_KW_LET);
                int fieldStart = p->current;
                jaiPAdvance(p);
                AstField field;
                memset(&field, 0, sizeof field);
                field.visibility = memberVis;
                field.isStatic = memberStatic;
                field.isLet = isLet;
                Token *nameTok = jaiPCur(p);
                field.name = jaiPExpectIdentName(p, "a field name");
                field.span = jaiPSpanOf(p, nameTok);
                if (field.name != NULL) {
                    if (jaiPMatch(p, TOK_COLON)) {
                        field.type = jaiParseType(p);
                    } else {
                        JaiDiag *d = jaiPErrorAt(p, field.span, E0105_EXPECTED_TYPE,
                                             "field `%s` needs a type annotation", field.name);
                        if (d != NULL) jaiDiagAddHelp(d, "write `%s %s: <type>`",
                                                      isLet ? "let" : "var", field.name);
                    }
                    if (jaiPMatch(p, TOK_EQ)) field.defaultValue = jaiParseExpression(p);
                    field.span = jaiPSpanSince(p, fieldStart);
                    JAI_VEC_PUSH(AstField, &fields, field);
                }
                jaiPEndStatement(p);
            } else if (jaiPCheck(p, TOK_KW_CONST)) {
                JaiDiag *d = jaiPErrorAt(p, jaiPCurSpan(p), E0100_UNEXPECTED_TOKEN,
                                     "class fields are declared with `var` or `let`");
                if (d != NULL) jaiDiagAddHelp(d, "write `static let NAME: <type> = <value>`");
            } else if (jaiPCheck(p, TOK_KW_FN)) {
                AstNode *m = jaiParseFnRest(p, memberVis, memberStatic, p->current, true, false);
                if (m != NULL) JAI_VEC_PUSH(AstNode *, &methods, m);
            } else if (jaiPCheckIdentText(p, 0, "get") && jaiPKindAt(p, 1) == TOK_IDENT) {
                AstNode *g = parseAccessor(p, memberVis, memberStatic, false);
                if (g != NULL) JAI_VEC_PUSH(AstNode *, &getters, g);
            } else if (jaiPCheckIdentText(p, 0, "set") && jaiPKindAt(p, 1) == TOK_IDENT) {
                AstNode *s = parseAccessor(p, memberVis, memberStatic, true);
                if (s != NULL) JAI_VEC_PUSH(AstNode *, &setters, s);
            } else {
                char found[80];
                jaiPDescribeToken(p, jaiPCur(p), found, sizeof found);
                JaiDiag *d = jaiPErrorAt(p, jaiPCurSpan(p), E0103_EXPECTED_STATEMENT,
                                     "expected a class member, found %s", found);
                if (d != NULL) {
                    jaiDiagAddHelp(d, "members are `var`/`let` fields, `fn` methods, "
                                      "or `get`/`set` properties");
                }
            }

            if (p->panicMode) jaiPSynchronize(p);
            if (p->current == before) jaiPAdvance(p);
            jaiPSkipNewlines(p);
        }
        if (!jaiPCheck(p, TOK_RBRACE)) {
            jaiPUnclosed(p, openIdx, "}", "class body");
        } else {
            jaiPAdvance(p);
        }
    }

    AstNode *n = jaiPNewNode(p, AST_CLASS_DECL, jaiPSpanSince(p, startIdx));
    n->as.classDecl.name = name;
    n->as.classDecl.generics = generics;
    n->as.classDecl.genericCount = genericCount;
    n->as.classDecl.superclass = superclass;
    n->as.classDecl.traitCount = traits.count;
    n->as.classDecl.traits = JAI_VEC_TAKE(p, &traits, AstType *);
    n->as.classDecl.fieldCount = fields.count;
    n->as.classDecl.fields = JAI_VEC_TAKE(p, &fields, AstField);
    n->as.classDecl.methodCount = methods.count;
    n->as.classDecl.methods = JAI_VEC_TAKE(p, &methods, AstNode *);
    n->as.classDecl.getterCount = getters.count;
    n->as.classDecl.getters = JAI_VEC_TAKE(p, &getters, AstNode *);
    n->as.classDecl.setterCount = setters.count;
    n->as.classDecl.setters = JAI_VEC_TAKE(p, &setters, AstNode *);
    n->as.classDecl.visibility = vis;
    n->as.classDecl.isAbstract = false;
    return n;
}

static AstNode *parseTraitDecl(Parser *p, AstVisibility vis, int startIdx) {
    jaiPAdvance(p);   /* trait */
    Token *declNameTok = jaiPCur(p);
    const char *name = jaiPExpectIdentName(p, "a trait name");
    if (name == NULL) return NULL;
    checkReservedTypeName(p, name, jaiPSpanOf(p, declNameTok));

    AstGeneric *generics = NULL;
    int genericCount = 0;
    parseGenerics(p, &generics, &genericCount);

    TypeVec supers;
    JAI_VEC_INIT(&supers);
    if (jaiPMatch(p, TOK_COLON)) {
        for (;;) {
            jaiParsePushTypeFlattened(&supers, jaiParseType(p));
            if (!jaiPMatch(p, TOK_COMMA)) break;
            if (jaiPCheck(p, TOK_EOF) || p->panicMode) break;
        }
    }

    NodeVec methods;
    JAI_VEC_INIT(&methods);
    int openIdx = p->current;
    if (jaiPExpect(p, TOK_LBRACE, "`{` to open the trait body")) {
        jaiPSkipNewlines(p);
        while (!jaiPCheck(p, TOK_RBRACE) && !jaiPCheck(p, TOK_EOF)) {
            int before = p->current;
            AstVisibility memberVis = AST_VIS_PRIVATE;
            if (jaiPMatch(p, TOK_KW_PUB))       memberVis = AST_VIS_PUBLIC;
            else if (jaiPMatch(p, TOK_KW_PROT)) memberVis = AST_VIS_PROTECTED;
            bool memberStatic = jaiPMatch(p, TOK_KW_STATIC);

            if (jaiPCheck(p, TOK_KW_FN)) {
                AstNode *m = jaiParseFnRest(p, memberVis, memberStatic, p->current, true, true);
                if (m != NULL) {
                    JAI_VEC_PUSH(AstNode *, &methods, m);
                    if (m->as.fn.body == NULL) jaiPEndStatement(p);
                }
            } else {
                char found[80];
                jaiPDescribeToken(p, jaiPCur(p), found, sizeof found);
                jaiPErrorAt(p, jaiPCurSpan(p), E0103_EXPECTED_STATEMENT,
                        "expected `fn` in a trait body, found %s", found);
            }
            if (p->panicMode) jaiPSynchronize(p);
            if (p->current == before) jaiPAdvance(p);
            jaiPSkipNewlines(p);
        }
        if (!jaiPCheck(p, TOK_RBRACE)) {
            jaiPUnclosed(p, openIdx, "}", "trait body");
        } else {
            jaiPAdvance(p);
        }
    }

    AstNode *n = jaiPNewNode(p, AST_TRAIT_DECL, jaiPSpanSince(p, startIdx));
    n->as.traitDecl.name = name;
    n->as.traitDecl.generics = generics;
    n->as.traitDecl.genericCount = genericCount;
    n->as.traitDecl.superCount = supers.count;
    n->as.traitDecl.supers = JAI_VEC_TAKE(p, &supers, AstType *);
    n->as.traitDecl.methodCount = methods.count;
    n->as.traitDecl.methods = JAI_VEC_TAKE(p, &methods, AstNode *);
    n->as.traitDecl.visibility = vis;
    return n;
}

static AstNode *parseEnumDecl(Parser *p, AstVisibility vis, int startIdx) {
    jaiPAdvance(p);   /* enum */
    Token *declNameTok = jaiPCur(p);
    const char *name = jaiPExpectIdentName(p, "an enum name");
    if (name == NULL) return NULL;
    checkReservedTypeName(p, name, jaiPSpanOf(p, declNameTok));

    AstGeneric *generics = NULL;
    int genericCount = 0;
    parseGenerics(p, &generics, &genericCount);

    VariantVec variants;
    NodeVec methods;
    JAI_VEC_INIT(&variants);
    JAI_VEC_INIT(&methods);

    int openIdx = p->current;
    if (jaiPExpect(p, TOK_LBRACE, "`{` to open the enum body")) {
        jaiPSkipNewlines(p);
        while (!jaiPCheck(p, TOK_RBRACE) && !jaiPCheck(p, TOK_EOF)) {
            int before = p->current;
            if (jaiPCheck(p, TOK_KW_FN) || jaiPCheck(p, TOK_KW_PUB) ||
                jaiPCheck(p, TOK_KW_STATIC)) {
                AstVisibility memberVis = AST_VIS_PRIVATE;
                if (jaiPMatch(p, TOK_KW_PUB)) memberVis = AST_VIS_PUBLIC;
                bool memberStatic = jaiPMatch(p, TOK_KW_STATIC);
                if (jaiPCheck(p, TOK_KW_FN)) {
                    AstNode *m = jaiParseFnRest(p, memberVis, memberStatic,
                                                p->current, true, false);
                    if (m != NULL) JAI_VEC_PUSH(AstNode *, &methods, m);
                } else {
                    jaiPErrorAtCur(p, E0103_EXPECTED_STATEMENT,
                                   "expected `fn` after `pub`/`static`");
                }
            } else {
                int variantStart = p->current;
                AstVariant variant;
                memset(&variant, 0, sizeof variant);
                Token *nameTok = jaiPCur(p);
                variant.name = jaiPExpectIdentName(p, "a variant name");
                variant.span = jaiPSpanOf(p, nameTok);
                if (variant.name != NULL) {
                    if (jaiPCheck(p, TOK_LPAREN)) {
                        int payloadIdx = p->current;
                        jaiPAdvance(p);
                        parseParams(p, payloadIdx, &variant.params, &variant.paramCount);
                    }
                    variant.span = jaiPSpanSince(p, variantStart);
                    JAI_VEC_PUSH(AstVariant, &variants, variant);
                }
                jaiPMatch(p, TOK_COMMA);
            }
            if (p->panicMode) jaiPSynchronize(p);
            if (p->current == before) jaiPAdvance(p);
            jaiPSkipNewlines(p);
        }
        if (!jaiPCheck(p, TOK_RBRACE)) {
            jaiPUnclosed(p, openIdx, "}", "enum body");
        } else {
            jaiPAdvance(p);
        }
    }

    AstNode *n = jaiPNewNode(p, AST_ENUM_DECL, jaiPSpanSince(p, startIdx));
    n->as.enumDecl.name = name;
    n->as.enumDecl.generics = generics;
    n->as.enumDecl.genericCount = genericCount;
    n->as.enumDecl.variantCount = variants.count;
    n->as.enumDecl.variants = JAI_VEC_TAKE(p, &variants, AstVariant);
    n->as.enumDecl.methodCount = methods.count;
    n->as.enumDecl.methods = JAI_VEC_TAKE(p, &methods, AstNode *);
    n->as.enumDecl.visibility = vis;
    return n;
}

static AstNode *parseTypeDecl(Parser *p, AstVisibility vis, int startIdx) {
    jaiPAdvance(p);   /* type */
    Token *declNameTok = jaiPCur(p);
    const char *name = jaiPExpectIdentName(p, "a type name");
    if (name == NULL) return NULL;
    checkReservedTypeName(p, name, jaiPSpanOf(p, declNameTok));

    AstGeneric *generics = NULL;
    int genericCount = 0;
    parseGenerics(p, &generics, &genericCount);

    jaiPExpect(p, TOK_EQ, "`=` in a type alias");
    AstType *aliased = jaiParseType(p);
    jaiPEndStatement(p);

    AstNode *n = jaiPNewNode(p, AST_TYPE_DECL, jaiPSpanSince(p, startIdx));
    n->as.typeDecl.name = name;
    n->as.typeDecl.generics = generics;
    n->as.typeDecl.genericCount = genericCount;
    n->as.typeDecl.aliased = aliased;
    n->as.typeDecl.visibility = vis;
    return n;
}

/* `std.math`, `.sibling`, `..pkg.mod` */
static const char *parseModulePath(Parser *p) {
    JaiBuf buf;
    jaiBufInit(&buf);
    size_t len = 0;
    const char *text = NULL;

    while (jaiPCheck(p, TOK_DOT) || jaiPCheck(p, TOK_DOTDOT) || jaiPCheck(p, TOK_ELLIPSIS)) {
        text = jaiPRawText(p, jaiPAdvance(p), &len);
        if (text != NULL) jaiBufAppend(&buf, text, len);
    }

    if (!jaiPCheck(p, TOK_IDENT)) {
        jaiBufFree(&buf);
        char found[80];
        jaiPDescribeToken(p, jaiPCur(p), found, sizeof found);
        jaiPErrorAt(p, jaiPCurSpan(p), E0804_INVALID_MODULE_PATH,
                    "expected a module path, found %s", found);
        return NULL;
    }
    text = jaiPRawText(p, jaiPAdvance(p), &len);
    if (text != NULL) jaiBufAppend(&buf, text, len);
    while (jaiPCheck(p, TOK_DOT) && jaiPKindAt(p, 1) == TOK_IDENT) {
        jaiPAdvance(p);
        jaiBufPush(&buf, '.');
        text = jaiPRawText(p, jaiPAdvance(p), &len);
        if (text != NULL) jaiBufAppend(&buf, text, len);
    }

    const char *path = jaiPInternText(p, (const char *)buf.data, buf.count);
    jaiBufFree(&buf);
    return path;
}

static AstNode *parseImport(Parser *p, int startIdx) {
    jaiPAdvance(p);   /* import */
    const char *path = parseModulePath(p);
    if (path == NULL) return NULL;
    const char *alias = NULL;
    if (jaiPMatch(p, TOK_KW_AS)) alias = jaiPExpectIdentName(p, "an import alias");
    jaiPEndStatement(p);

    AstNode *n = jaiPNewNode(p, AST_IMPORT, jaiPSpanSince(p, startIdx));
    n->as.import.path = path;
    n->as.import.alias = alias;
    return n;
}

static AstNode *parseFromImport(Parser *p, int startIdx) {
    jaiPAdvance(p);   /* from */
    const char *path = parseModulePath(p);
    if (path == NULL) return NULL;
    jaiPExpect(p, TOK_KW_IMPORT, "`import` after the module path");

    bool wildcard = false;
    ItemVec items;
    JAI_VEC_INIT(&items);

    if (jaiPCheck(p, TOK_STAR)) {
        jaiPAdvance(p);
        wildcard = true;
    } else {
        bool parens = jaiPMatch(p, TOK_LPAREN);
        for (;;) {
            jaiPSkipNewlines(p);
            AstImportItem item;
            memset(&item, 0, sizeof item);
            Token *nameTok = jaiPCur(p);
            item.name = jaiPExpectIdentName(p, "an imported name");
            if (item.name == NULL) break;
            item.span = jaiPSpanOf(p, nameTok);
            if (jaiPMatch(p, TOK_KW_AS)) item.alias = jaiPExpectIdentName(p, "an import alias");
            JAI_VEC_PUSH(AstImportItem, &items, item);
            if (!jaiPMatch(p, TOK_COMMA)) break;
            if (jaiPCheck(p, TOK_EOF) || p->panicMode) break;
        }
        if (parens) {
            jaiPSkipNewlines(p);
            jaiPExpect(p, TOK_RPAREN, "`)` to close the import list");
        }
    }
    jaiPEndStatement(p);

    AstNode *n = jaiPNewNode(p, AST_FROM_IMPORT, jaiPSpanSince(p, startIdx));
    n->as.fromImport.path = path;
    n->as.fromImport.isWildcard = wildcard;
    n->as.fromImport.itemCount = items.count;
    n->as.fromImport.items = JAI_VEC_TAKE(p, &items, AstImportItem);
    return n;
}

static AstNode *parseExportDecl(Parser *p, int startIdx) {
    jaiPAdvance(p);   /* export */
    NameVec names;
    JAI_VEC_INIT(&names);

    int openIdx = p->current;
    if (jaiPExpect(p, TOK_LBRACE, "`{` to open the export list")) {
        jaiPSkipNewlines(p);
        if (!jaiPCheck(p, TOK_RBRACE)) {
            for (;;) {
                jaiPSkipNewlines(p);
                const char *name = jaiPExpectIdentName(p, "an exported name");
                if (name == NULL) break;
                JAI_VEC_PUSH(const char *, &names, name);
                jaiPSkipNewlines(p);
                if (!jaiPMatch(p, TOK_COMMA)) break;
                jaiPSkipNewlines(p);
                if (jaiPCheck(p, TOK_RBRACE) || jaiPCheck(p, TOK_EOF)) break;
            }
        }
        jaiPSkipNewlines(p);
        if (!jaiPCheck(p, TOK_RBRACE)) {
            jaiPUnclosed(p, openIdx, "}", "export list");
        } else {
            jaiPAdvance(p);
        }
    }
    jaiPEndStatement(p);

    AstNode *n = jaiPNewNode(p, AST_EXPORT, jaiPSpanSince(p, startIdx));
    n->as.exportDecl.count = names.count;
    n->as.exportDecl.names = JAI_VEC_TAKE(p, &names, const char *);
    return n;
}

static AstNode *parseModuleDecl(Parser *p, int startIdx) {
    jaiPAdvance(p);   /* module */
    const char *name = parseModulePath(p);
    if (name == NULL) return NULL;
    jaiPEndStatement(p);

    AstNode *n = jaiPNewNode(p, AST_MODULE_DECL, jaiPSpanSince(p, startIdx));
    n->as.moduleDecl.name = name;
    return n;
}

AstNode *jaiParseDeclaration(Parser *p) {
    int startIdx = p->current;
    AstVisibility vis = AST_VIS_PRIVATE;
    bool sawVisibility = false;
    if (jaiPCheck(p, TOK_KW_PUB)) {
        jaiPAdvance(p);
        vis = AST_VIS_PUBLIC;
        sawVisibility = true;
    } else if (jaiPCheck(p, TOK_KW_PROT)) {
        jaiPAdvance(p);
        vis = AST_VIS_PROTECTED;
        sawVisibility = true;
    }
    bool isStatic = false;
    if (jaiPCheck(p, TOK_KW_STATIC)) {
        jaiPAdvance(p);
        isStatic = true;
    }

    switch (jaiPCurKind(p)) {
    case TOK_KW_CLASS:  return parseClassDecl(p, vis, startIdx);
    case TOK_KW_TRAIT:  return parseTraitDecl(p, vis, startIdx);
    case TOK_KW_ENUM:   return parseEnumDecl(p, vis, startIdx);
    case TOK_KW_TYPE:   return parseTypeDecl(p, vis, startIdx);
    case TOK_KW_LET: case TOK_KW_VAR: case TOK_KW_CONST:
        return parseVarDecl(p, vis, startIdx);
    case TOK_KW_FN:
        /* `fn name(...)` declares; `fn(...)` is an anonymous function value. */
        if (jaiPKindAt(p, 1) == TOK_IDENT) {
            return jaiParseFnRest(p, vis, isStatic, startIdx, true, false);
        }
        break;
    case TOK_KW_IMPORT: case TOK_KW_FROM: case TOK_KW_EXPORT: case TOK_KW_MODULE:
        if (sawVisibility || isStatic) {
            JaiDiag *d = jaiPErrorAt(p, jaiPSpanSince(p, startIdx), E0100_UNEXPECTED_TOKEN,
                                 "imports and exports take no visibility modifier");
            if (d != NULL) jaiDiagAddHelp(d, "delete the modifier");
            return NULL;
        }
        switch (jaiPCurKind(p)) {
        case TOK_KW_IMPORT: return parseImport(p, startIdx);
        case TOK_KW_FROM:   return parseFromImport(p, startIdx);
        case TOK_KW_EXPORT: return parseExportDecl(p, startIdx);
        default:            return parseModuleDecl(p, startIdx);
        }
    default:
        break;
    }

    if (sawVisibility || isStatic) {
        char found[80];
        jaiPDescribeToken(p, jaiPCur(p), found, sizeof found);
        JaiDiag *d = jaiPErrorAt(p, jaiPSpanSince(p, startIdx), E0103_EXPECTED_STATEMENT,
                             "expected a declaration after `%s`, found %s",
                             isStatic ? "static" : (vis == AST_VIS_PUBLIC ? "pub" : "prot"), found);
        if (d != NULL) {
            jaiDiagAddHelp(d, "`pub` may precede `fn`, `class`, `trait`, `enum`, `type`, "
                              "`let`, `var`, or `const`");
        }
        return NULL;
    }

    return jaiParseStatement(p);
}
