/* check_stmt.c — statements, control flow, and the declarations that appear
 * inside a body: patterns and match, `let`/`var`/`const`, assignment, loops,
 * try/catch, and function bodies.
 *
 * Two things run through all of it. There is no truthiness (spec §5.1), so
 * every condition is checked for `bool` rather than coerced; and narrowing is
 * scoped — a fact learned by `if x is C` is pushed on entry to the arm it
 * holds in and popped on the way out, which is why so many functions here are
 * a mark/restore pair around a recursive call.
 */
#include "check_internal.h"

/* ------------------------------------------------------------------ */
/* Patterns                                                             */
/* ------------------------------------------------------------------ */

#define MAX_BINDS 64

typedef struct {
    const char *names[MAX_BINDS];
    JaiSpan     spans[MAX_BINDS];
    int         count;
} BindSet;

static void patternWalk(Checker *c, AstNode *pat, JaiType *subject, BindSet *binds);

static void recordBinding(Checker *c, BindSet *binds, const char *name, JaiSpan span) {
    if (binds == NULL || name == NULL || name[0] == '_') return;
    for (int i = 0; i < binds->count; i++)
        if (jaiChkSameName(binds->names[i], name)) {
            JaiDiag *d = ERR(c, E0503_DUPLICATE_BINDING_IN_PATTERN, span,
                             "`%s` is bound twice in the same pattern", name);
            jaiDiagAddLabel(d, binds->spans[i], "first bound here");
            return;
        }
    if (binds->count >= MAX_BINDS) return;
    binds->names[binds->count] = name;
    binds->spans[binds->count] = span;
    binds->count++;
}

/* Element type of a sequence subject; tuples answer per position. */
static JaiType *sequenceElement(JaiType *subject, int index) {
    if (subject == NULL) return gTypes.tAny;
    switch (subject->kind) {
    case TY_LIST:
    case TY_SET:
        return subject->argCount == 1 ? subject->args[0] : gTypes.tAny;
    case TY_TUPLE:
        return (index >= 0 && index < subject->argCount) ? subject->args[index]
                                                         : gTypes.tAny;
    default:
        return gTypes.tAny;
    }
}

static void patternSequence(Checker *c, AstNode *pat, JaiType *subject, BindSet *binds) {
    int count = pat->as.patSeq.count;
    int rest = pat->as.patSeq.restIndex;

    if (subject != NULL && subject->kind == TY_TUPLE && rest < 0 &&
        count != subject->argCount) {
        char got[TYPE_BUF];
        jaiChkRenderType(subject, got, sizeof got);
        ERR(c, E0502_PATTERN_ARITY, pat->span,
            "this pattern has %d element%s but `%s` has %d", count,
            count == 1 ? "" : "s", got, subject->argCount);
    }

    for (int i = 0; i < count; i++) {
        AstNode *sub = pat->as.patSeq.elems[i];
        if (sub == NULL) continue;
        JaiType *elem;
        if (i == rest) {
            /* `...rest` collects the remainder, so it is a list of elements. */
            elem = jaiTypeList(sequenceElement(subject, -1));
        } else {
            int index = (rest >= 0 && i > rest) ? -1 : i;
            elem = sequenceElement(subject, index);
        }
        patternWalk(c, sub, elem, binds);
    }
}

static void patternEnum(Checker *c, AstNode *pat, JaiType *subject, BindSet *binds) {
    const char *typeName = pat->as.patClass.typeName;
    const char *variantName = pat->as.patClass.variantName;
    int count = pat->as.patClass.count;

    TypeDecl *decl = jaiTypeDeclFind(typeName);
    if (decl == NULL) {
        TypeDecl *fromSubject = jaiChkTypeDeclOf(subject);
        if (fromSubject != NULL && jaiChkSameName(fromSubject->name, typeName)) decl = fromSubject;
    }
    if (decl == NULL) {
        if (!jaiChkIsAny(subject))
            ERR(c, E0402_UNKNOWN_TYPE, pat->span, "unknown type `%s` in pattern",
                typeName == NULL ? "?" : typeName);
        for (int i = 0; i < count; i++)
            patternWalk(c, pat->as.patClass.subPatterns[i], gTypes.tAny, binds);
        return;
    }
    jaiChkLayoutDecl(c, jaiChkDeclEntry(decl));

    int vi = jaiChkFindFieldIndex(decl, variantName);
    if (vi < 0 || vi >= decl->variantCount) {
        JaiDiag *d = ERR(c, E0410_UNKNOWN_MEMBER, pat->span,
                         "`%s` has no variant `%s`", decl->name,
                         variantName == NULL ? "?" : variantName);
        jaiChkSuggestMember(d, decl, variantName == NULL ? "" : variantName);
        for (int i = 0; i < count; i++)
            patternWalk(c, pat->as.patClass.subPatterns[i], gTypes.tAny, binds);
        return;
    }

    JaiType *payload = decl->fields[vi].type;
    int arity = (payload != NULL && payload->kind == TY_FN) ? payload->argCount : 0;
    if (count != arity) {
        JaiDiag *d = ERR(c, E0502_PATTERN_ARITY, pat->span,
                         "`%s.%s` takes %d field%s but the pattern binds %d",
                         decl->name, variantName, arity, arity == 1 ? "" : "s", count);
        jaiDiagAddLabel(d, decl->span, "`%s` is declared here", decl->name);
    }
    for (int i = 0; i < count; i++) {
        JaiType *field = (payload != NULL && payload->kind == TY_FN && i < payload->argCount)
                             ? payload->args[i]
                             : gTypes.tAny;
        patternWalk(c, pat->as.patClass.subPatterns[i], field, binds);
    }
}

static void patternClass(Checker *c, AstNode *pat, JaiType *subject, BindSet *binds) {
    if (pat->as.patClass.variantName != NULL) { patternEnum(c, pat, subject, binds); return; }

    const char *typeName = pat->as.patClass.typeName;
    int count = pat->as.patClass.count;
    TypeDecl *decl = jaiTypeDeclFind(typeName);
    if (decl == NULL) {
        if (!jaiChkIsAny(subject))
            ERR(c, E0402_UNKNOWN_TYPE, pat->span, "unknown type `%s` in pattern",
                typeName == NULL ? "?" : typeName);
        for (int i = 0; i < count; i++)
            patternWalk(c, pat->as.patClass.subPatterns[i], gTypes.tAny, binds);
        return;
    }
    jaiChkLayoutDecl(c, jaiChkDeclEntry(decl));

    /* An enum written without a variant is a whole-enum pattern. */
    if (decl->isEnum && count == 0) return;

    if (count > decl->fieldCount) {
        ERR(c, E0502_PATTERN_ARITY, pat->span,
            "`%s` has %d field%s but the pattern binds %d", decl->name,
            decl->fieldCount, decl->fieldCount == 1 ? "" : "s", count);
    }

    for (int i = 0; i < count; i++) {
        const char *fieldName = pat->as.patClass.fieldNames != NULL
                                    ? pat->as.patClass.fieldNames[i]
                                    : NULL;
        JaiType *fieldType = gTypes.tAny;
        int fi = fieldName != NULL ? jaiChkFindFieldIndex(decl, fieldName)
                                   : (i < decl->fieldCount ? i : -1);
        if (fieldName != NULL && fi < 0) {
            JaiDiag *d = ERR(c, E0410_UNKNOWN_MEMBER, pat->span,
                             "`%s` has no field `%s`", decl->name, fieldName);
            jaiChkSuggestMember(d, decl, fieldName);
        } else if (fi >= 0) {
            fieldType = jaiChkOrAny(decl->fields[fi].type);
            const TypeDecl *owner = jaiChkOwnerOfField(decl, decl->fields[fi].name);
            if (!jaiChkVisibleFrom(c, owner, decl->fields[fi].visibility))
                ERR(c, E0701_PRIVATE_ACCESS, pat->span,
                    "`%s` is %s to `%s`", decl->fields[fi].name,
                    jaiChkVisibilityWord(decl->fields[fi].visibility), owner->name);
        }

        AstNode *sub = pat->as.patClass.subPatterns != NULL
                           ? pat->as.patClass.subPatterns[i]
                           : NULL;
        if (sub != NULL) patternWalk(c, sub, fieldType, binds);
        else if (fieldName != NULL) recordBinding(c, binds, fieldName, pat->span);
    }
}

static void patternWalk(Checker *c, AstNode *pat, JaiType *subject, BindSet *binds) {
    if (pat == NULL) return;
    subject = jaiChkOrAny(subject);

    switch (pat->kind) {
    case AST_PAT_WILDCARD:
        break;

    case AST_PAT_BIND: {
        JaiType *bound = pat->as.patBind.type != NULL
                             ? jaiChkResolveAstType(c, pat->as.patBind.type)
                             : subject;
        recordBinding(c, binds, pat->as.patBind.name, pat->span);
        if (pat->as.patBind.symbol != NULL) pat->as.patBind.symbol->type = jaiChkOrAny(bound);
        pat->type = jaiChkOrAny(bound);
        return;
    }

    case AST_PAT_LITERAL: {
        if (pat->as.patLiteral.value != NULL) {
            jaiChkApplyContext(c, pat->as.patLiteral.value, subject);
            JaiType *lit = jaiChkValue(c, pat->as.patLiteral.value);
            bool guard = false;
            if (!jaiChkIsAny(subject) && !jaiChkIsAny(lit) &&
                !jaiTypeAssignable(lit, subject, &guard) &&
                !jaiTypeAssignable(subject, lit, &guard)) {
                char want[TYPE_BUF], got[TYPE_BUF];
                jaiChkRenderType(subject, want, sizeof want);
                jaiChkRenderType(lit, got, sizeof got);
                ERR(c, E0400_TYPE_MISMATCH, pat->span,
                    "this pattern can never match: expected `%s`, found `%s`", want, got);
            }
        }
        break;
    }

    case AST_PAT_RANGE:
        if (pat->as.patRange.lo != NULL) {
            jaiChkApplyContext(c, pat->as.patRange.lo, subject);
            jaiChkValue(c, pat->as.patRange.lo);
        }
        if (pat->as.patRange.hi != NULL) {
            jaiChkApplyContext(c, pat->as.patRange.hi, subject);
            jaiChkValue(c, pat->as.patRange.hi);
        }
        break;

    case AST_PAT_TUPLE:
    case AST_PAT_LIST:
        patternSequence(c, pat, subject, binds);
        break;

    case AST_PAT_CLASS:
    case AST_PAT_ENUM:
        patternClass(c, pat, subject, binds);
        break;

    case AST_PAT_OR:
        /* Each alternative binds the same names, so they get their own scope
         * rather than colliding with one another. */
        for (int i = 0; i < pat->as.patSeq.count; i++) {
            BindSet alt;
            memset(&alt, 0, sizeof alt);
            patternWalk(c, pat->as.patSeq.elems[i], subject, &alt);
        }
        break;

    default:
        /* A destructuring assignment target reuses expression nodes, which
         * keep the type the expression checker gave them. */
        if (jaiAstIsExpression(pat->kind)) return (void)jaiChkExpr(c, pat);
        break;
    }

    pat->type = subject;
}

void jaiChkPattern(Checker *c, AstNode *pat, JaiType *subject) {
    BindSet binds;
    memset(&binds, 0, sizeof binds);
    patternWalk(c, pat, subject, &binds);
}

/* ------------------------------------------------------------------ */
/* Match                                                                */
/* ------------------------------------------------------------------ */

#define MAX_TRACKED_LITERALS 64
#define MAX_TRACKED_VARIANTS 256

#define MAX_TRACKED_LENGTHS 64

typedef struct {
    bool     sawCatchAll;
    bool     coveredTrue, coveredFalse;
    uint8_t  variants[MAX_TRACKED_VARIANTS];
    ConstValue literals[MAX_TRACKED_LITERALS];
    int      literalCount;
    bool     literalOverflow;
    /* Sequence arms are tracked by the lengths they accept: a bit per exact
     * length, plus the shortest length from which a `[..., ...rest]` arm takes
     * everything. `match xs { [] => .., [a, ...r] => .. }` is total over a
     * list, and a match that cannot fall through does not yield `T?`. */
    uint64_t coveredLengths;
    int      coveredFrom;          /* MAX_TRACKED_LENGTHS = no open arm */
} Coverage;

/* Only an arm that binds unconditionally covers its shape; `[1, x]` still
 * falls through for a list of two whose head is not 1. */
static bool patternIsIrrefutable(const AstNode *pat) {
    if (pat == NULL) return false;
    return pat->kind == AST_PAT_WILDCARD || pat->kind == AST_PAT_BIND;
}

static bool sequenceArmIsIrrefutable(const AstNode *pat) {
    for (int i = 0; i < pat->as.patSeq.count; i++) {
        if (i == pat->as.patSeq.restIndex) continue;
        if (!patternIsIrrefutable(pat->as.patSeq.elems[i])) return false;
    }
    return true;
}

static bool sameConst(const ConstValue *a, const ConstValue *b) {
    if (a->kind != b->kind) return false;
    switch (a->kind) {
    case CONST_INT:   return a->as.i == b->as.i;
    case CONST_FLOAT: return a->as.f == b->as.f;
    case CONST_BOOL:  return a->as.b == b->as.b;
    case CONST_STR:
        return a->as.s.length == b->as.s.length &&
               memcmp(a->as.s.chars, b->as.s.chars, a->as.s.length) == 0;
    case CONST_NULL:  return true;
    default:          return false;
    }
}

/* Is this pattern already covered by the arms seen so far? */
static bool patternCovered(Checker *c, AstNode *pat, TypeDecl *enumDecl, Coverage *cov) {
    if (pat == NULL) return false;
    switch (pat->kind) {
    case AST_PAT_WILDCARD:
    case AST_PAT_BIND:
        return cov->sawCatchAll;

    case AST_PAT_LITERAL: {
        ConstValue v = jaiConstEval(c, pat->as.patLiteral.value);
        if (v.kind == CONST_NONE) return false;
        if (v.kind == CONST_BOOL)
            return v.as.b ? cov->coveredTrue : cov->coveredFalse;
        for (int i = 0; i < cov->literalCount; i++)
            if (sameConst(&cov->literals[i], &v)) return true;
        return false;
    }

    case AST_PAT_ENUM:
    case AST_PAT_CLASS: {
        if (enumDecl == NULL || pat->as.patClass.variantName == NULL) return false;
        int vi = jaiChkFindFieldIndex(enumDecl, pat->as.patClass.variantName);
        if (vi < 0 || vi >= MAX_TRACKED_VARIANTS) return false;
        /* Only a payload-free variant is fully covered by an earlier arm; a
         * payload arm may have narrowed on its sub-patterns. */
        return cov->variants[vi] != 0 && pat->as.patClass.count == 0;
    }

    case AST_PAT_OR:
        for (int i = 0; i < pat->as.patSeq.count; i++)
            if (!patternCovered(c, pat->as.patSeq.elems[i], enumDecl, cov)) return false;
        return pat->as.patSeq.count > 0;

    default:
        return false;
    }
}

static void coverPattern(Checker *c, AstNode *pat, TypeDecl *enumDecl, Coverage *cov) {
    if (pat == NULL) return;
    switch (pat->kind) {
    case AST_PAT_WILDCARD:
    case AST_PAT_BIND:
        cov->sawCatchAll = true;
        break;

    case AST_PAT_LITERAL: {
        ConstValue v = jaiConstEval(c, pat->as.patLiteral.value);
        if (v.kind == CONST_NONE) break;
        if (v.kind == CONST_BOOL) {
            if (v.as.b) cov->coveredTrue = true; else cov->coveredFalse = true;
            break;
        }
        if (cov->literalCount >= MAX_TRACKED_LITERALS) { cov->literalOverflow = true; break; }
        cov->literals[cov->literalCount++] = v;
        break;
    }

    case AST_PAT_ENUM:
    case AST_PAT_CLASS: {
        if (enumDecl == NULL || pat->as.patClass.variantName == NULL) break;
        int vi = jaiChkFindFieldIndex(enumDecl, pat->as.patClass.variantName);
        if (vi >= 0 && vi < MAX_TRACKED_VARIANTS) cov->variants[vi] = 1;
        break;
    }

    case AST_PAT_LIST:
    case AST_PAT_TUPLE: {
        if (!sequenceArmIsIrrefutable(pat)) break;
        int fixed = pat->as.patSeq.count;
        if (pat->as.patSeq.restIndex >= 0) {
            fixed--;                          /* `...rest` matches zero or more */
            if (fixed < cov->coveredFrom) cov->coveredFrom = fixed;
        } else if (fixed < MAX_TRACKED_LENGTHS) {
            cov->coveredLengths |= (uint64_t)1 << fixed;
        }
        break;
    }

    case AST_PAT_OR:
        for (int i = 0; i < pat->as.patSeq.count; i++)
            coverPattern(c, pat->as.patSeq.elems[i], enumDecl, cov);
        break;

    default:
        break;
    }
}

static void reportNonExhaustive(Checker *c, AstNode *node, JaiType *subject,
                                TypeDecl *enumDecl, const Coverage *cov) {
    if (cov->sawCatchAll) return;

    if (enumDecl != NULL) {
        char missing[256];
        size_t used = 0;
        int count = 0;
        for (int i = 0; i < enumDecl->variantCount && i < MAX_TRACKED_VARIANTS; i++) {
            if (cov->variants[i] != 0) continue;
            count++;
            int written = snprintf(missing + used, sizeof missing - used, "%s`%s.%s`",
                                   used == 0 ? "" : ", ", enumDecl->name,
                                   enumDecl->fields[i].name);
            if (written < 0 || (size_t)written >= sizeof missing - used) {
                used = sizeof missing - 1;
                break;
            }
            used += (size_t)written;
        }
        if (count == 0) return;
        missing[used] = '\0';
        JaiDiag *d = ERR(c, E0501_NON_EXHAUSTIVE_MATCH, node->span,
                         "non-exhaustive match: %s not covered", missing);
        jaiDiagAddLabel(d, enumDecl->span, "`%s` is declared here", enumDecl->name);
        jaiDiagAddHelp(d, "add the missing arm%s, or a `_ => ...` arm",
                       count == 1 ? "" : "s");
        return;
    }

    if (subject != NULL && subject->kind == TY_BOOL &&
        (!cov->coveredTrue || !cov->coveredFalse)) {
        JaiDiag *d = ERR(c, E0501_NON_EXHAUSTIVE_MATCH, node->span,
                         "non-exhaustive match: `%s` is not covered",
                         cov->coveredTrue ? "false" : "true");
        jaiDiagAddHelp(d, "add the missing arm, or a `_ => ...` arm");
    }
}

/* Do the sequence arms leave no length uncovered? An open `[..., ...rest]` arm
 * takes everything from `coveredFrom` up, so the shorter lengths are the only
 * ones that still need an exact arm. Only a list or tuple subject can be
 * exhausted this way: for `any` the value need not be a sequence at all. */
static bool sequenceCoverageComplete(JaiType *subject, const Coverage *cov) {
    if (subject == NULL) return false;
    if (subject->kind != TY_LIST && subject->kind != TY_TUPLE) return false;
    if (cov->coveredFrom >= MAX_TRACKED_LENGTHS) return false;
    for (int len = 0; len < cov->coveredFrom; len++)
        if ((cov->coveredLengths & ((uint64_t)1 << len)) == 0) return false;
    return true;
}

JaiType *jaiChkMatchExpr(Checker *c, AstNode *node, bool isStatement) {
    JaiType *subject = jaiChkValue(c, node->as.match.subject);
    TypeDecl *enumDecl = jaiChkTypeDeclOf(subject);
    if (enumDecl != NULL && !enumDecl->isEnum) enumDecl = NULL;
    if (enumDecl != NULL) jaiChkLayoutDecl(c, jaiChkDeclEntry(enumDecl));

    Coverage cov;
    memset(&cov, 0, sizeof cov);
    cov.coveredFrom = MAX_TRACKED_LENGTHS;
    JaiType *result = NULL;

    for (int i = 0; i < node->as.match.armCount; i++) {
        AstMatchArm *arm = &node->as.match.arms[i];

        if (patternCovered(c, arm->pattern, enumDecl, &cov)) {
            JaiDiag *d = ERR(c, E0500_UNREACHABLE_ARM, arm->span,
                             "unreachable match arm: an earlier arm already matches");
            jaiDiagAddHelp(d, "remove it, or move it above the arm that subsumes it");
        }

        int mark = jaiChkNarrowMark();
        jaiChkPattern(c, arm->pattern, subject);
        if (arm->guard != NULL) jaiChkCondition(c, arm->guard, "a match guard");

        JaiType *body = arm->body != NULL ? jaiChkExpr(c, arm->body) : gTypes.tVoid;
        jaiChkNarrowRestore(mark);

        if (!isStatement && !jaiChkIsVoid(body) && !jaiChkIsNever(body))
            result = result == NULL ? body : jaiTypeJoin(result, body);

        /* A guarded arm does not cover its pattern: the guard may be false. */
        if (arm->guard == NULL) coverPattern(c, arm->pattern, enumDecl, &cov);
    }

    reportNonExhaustive(c, node, subject, enumDecl, &cov);
    if (isStatement) return gTypes.tVoid;
    if (result == NULL) return gTypes.tAny;
    /* A match expression that may fall through yields null on the missing arm.
     * A statically unknown subject — `any`, or an enum this module only
     * imported — is not evidence of a missing arm: nothing was reported above,
     * so the arms are taken at their word here too rather than infecting every
     * result with `| null`. */
    bool total = cov.sawCatchAll || enumDecl != NULL || jaiChkIsAny(subject) ||
                 sequenceCoverageComplete(subject, &cov);
    return total ? result : jaiTypeOptional(result);
}

/* ------------------------------------------------------------------ */
/* Declarations of values                                               */
/* ------------------------------------------------------------------ */

static Symbol *declSymbol(AstNode *decl) {
    if (decl->as.varDecl.symbol != NULL) return decl->as.varDecl.symbol;
    AstNode *pattern = decl->as.varDecl.pattern;
    if (pattern != NULL && pattern->kind == AST_PAT_BIND) return pattern->as.patBind.symbol;
    return NULL;
}

/* An unannotated `var` initialised to null would otherwise be pinned to the
 * null type and reject every later assignment; the binding is dynamic. */
static JaiType *widenInferred(VarDeclKind kind, JaiType *t) {
    if (t == NULL) return gTypes.tAny;
    if (kind == VD_VAR && (t->kind == TY_NULL || t->kind == TY_NEVER)) return gTypes.tAny;
    if (t->kind == TY_NEVER) return gTypes.tAny;
    return t;
}

static void foldConstDecl(Checker *c, AstNode *node, Symbol *sym) {
    AstNode *init = node->as.varDecl.init;
    if (init == NULL) {
        ERR(c, E0304_MISSING_INITIALIZER, node->span,
            "a `const` declaration needs an initialiser");
        return;
    }

    int before = c->errorCount;
    gJaiCheck.constRequired = true;
    ConstValue v = jaiConstEval(c, init);
    gJaiCheck.constRequired = false;

    if (v.kind == CONST_NONE) {
        if (c->errorCount == before) {
            JaiDiag *d = ERR(c, E0303_NON_CONSTANT_CONST, init->span,
                             "the initialiser of a `const` must be a constant expression");
            jaiDiagAddHelp(d, "use `let` if the value is only known at run time");
        }
        return;
    }

    /* Keep the folded value on the symbol so every use can inline it. */
    AstNode *literal = jaiAstNew(c->ast, AST_NULL_LIT, init->span);
    jaiConstReplace(c->ast, literal, v);
    jaiConstReplace(c->ast, init, v);
    if (sym != NULL) {
        sym->isConstFolded = true;
        sym->constValue = literal;
    }
}

static void checkVarDecl(Checker *c, AstNode *node) {
    AstType *annotation = node->as.varDecl.declaredType;
    JaiType *declared = (annotation != NULL && annotation->kind != TYPE_INFER)
                            ? jaiChkResolveAstType(c, annotation)
                            : NULL;
    AstNode *init = node->as.varDecl.init;
    JaiType *bound = declared;

    if (init != NULL) {
        if (declared != NULL) jaiChkApplyContext(c, init, declared);
        JaiType *got = jaiChkValue(c, init);
        if (declared != NULL)
            jaiChkRequireAssignable(c, init, got, declared, E0400_TYPE_MISMATCH,
                              "mismatched types in this binding");
        else
            bound = widenInferred(node->as.varDecl.kind, got);
    }

    if (bound == NULL) bound = gTypes.tAny;
    node->type = bound;

    Symbol *sym = declSymbol(node);
    if (sym != NULL) sym->type = bound;
    if (node->as.varDecl.pattern != NULL)
        jaiChkPattern(c, node->as.varDecl.pattern, bound);

    if (node->as.varDecl.kind == VD_CONST) foldConstDecl(c, node, sym);
}

/* ------------------------------------------------------------------ */
/* Assignment                                                           */
/* ------------------------------------------------------------------ */

/* Store the right-hand side into a known destination type: the literal
 * conversion has to happen before the value is typed, or an int literal in a
 * float field would be reported and then silently converted. */
static void checkStoredValue(Checker *c, AstNode *value, JaiType *want) {
    if (want != NULL && !jaiChkIsAny(want)) jaiChkApplyContext(c, value, want);
    JaiType *got = jaiChkValue(c, value);
    if (want != NULL)
        jaiChkRequireAssignable(c, value, got, want, E0400_TYPE_MISMATCH,
                          "mismatched types in this assignment");
}

static void checkFieldAssign(Checker *c, AstNode *node, AstNode *target) {
    AstNode *object = target->as.member.object;
    AstNode *value = node->as.assign.value;
    const char *name = target->as.member.name;

    TypeDecl *decl = jaiChkStaticTargetDecl(c, object);
    bool staticAccess = decl != NULL;
    if (decl == NULL) {
        JaiType *objectType = jaiChkValue(c, object);
        decl = jaiChkTypeDeclOf(objectType);
        if (decl == NULL) {
            /* An `any` receiver, a module, or a builtin: the store is checked
             * at run time. */
            target->type = gTypes.tAny;
            jaiChkValue(c, value);
            return;
        }
    }
    jaiChkLayoutDecl(c, jaiChkDeclEntry(decl));

    int fi = jaiChkFindFieldIndex(decl, name);
    if (fi >= 0) {
        const TypeDecl *owner = jaiChkOwnerOfField(decl, name);
        if (!jaiChkVisibleFrom(c, owner, decl->fields[fi].visibility)) {
            JaiDiag *d = ERR(c, E0701_PRIVATE_ACCESS, target->span,
                             "`%s` is %s to `%s`", name,
                             jaiChkVisibilityWord(decl->fields[fi].visibility), owner->name);
            jaiDiagAddLabel(d, owner->span, "declared here");
        }
        /* A `let` field is settable from inside its own class — that is what
         * `init` is for — and nowhere else. */
        if (decl->fields[fi].isLet && !staticAccess && c->currentClass != decl) {
            JaiDiag *d = ERR(c, E0301_ASSIGN_TO_IMMUTABLE, target->span,
                             "`%s` is declared `let` and cannot be reassigned", name);
            jaiDiagAddHelp(d, "declare it `var` if it needs to change after `init`");
        }

        target->type = jaiChkOrAny(decl->fields[fi].type);
        checkStoredValue(c, value, target->type);
        return;
    }

    int si = jaiChkFindAccessorIndex(decl, name, true);
    if (si >= 0) {
        target->type = jaiChkOrAny(decl->members[si].type);
        checkStoredValue(c, value, target->type);
        return;
    }

    target->type = gTypes.tAny;
    if (jaiChkFindMemberIndex(decl, name) >= 0) {
        ERR(c, E0702_UNDECLARED_FIELD, target->span,
            "`%s` is a method of `%s` and cannot be assigned", name, decl->name);
    } else {
        JaiDiag *d = ERR(c, E0702_UNDECLARED_FIELD, target->span,
                         "`%s` has no field `%s`", decl->name, name);
        jaiDiagAddLabel(d, decl->span, "`%s` is declared here", decl->name);
        jaiChkSuggestMember(d, decl, name);
        jaiDiagAddHelp(d, "declare it in the class body; fields are not created "
                          "by assignment");
    }
    jaiChkValue(c, value);
}

static void checkAssign(Checker *c, AstNode *node) {
    AstNode *target = node->as.assign.target;
    AstNode *value = node->as.assign.value;
    if (target == NULL || value == NULL) return;

    /* A compound assignment reads the target first, so it is typed as an
     * operand of the operator before it is typed as a destination. */
    if (node->as.assign.isCompound) {
        JaiType *current = jaiChkExpr(c, target);
        JaiType *rhs = jaiChkValue(c, value);
        if (!jaiChkIsAny(current) && !jaiChkIsAny(rhs) && !jaiChkIsNever(current) && !jaiChkIsNever(rhs)) {
            OpKind op = node->as.assign.op;
            /* `x += n` on a float target with an int operand widens the
             * operand, exactly as the `x = x + n` it stands for would. */
            if (jaiChkWidensOperands(op) && current->kind == TY_FLOAT &&
                rhs->kind == TY_INT) {
                jaiChkWidenToFloat(c, value, gTypes.tFloat);
                rhs = gTypes.tFloat;
            }

            /* The other direction cannot: a compound assignment writes back
             * into its target, and a float result does not fit an int. */
            if (jaiChkWidensOperands(op) && current->kind == TY_INT &&
                rhs->kind == TY_FLOAT) {
                JaiDiag *d = ERR(c, E0411_INT_FLOAT_MIX, node->span,
                                 "cannot apply `%s=` to `int` and `float`",
                                 jaiOpKindText(op));
                jaiDiagAddHelp(d,
                               "the result is `float` and `%s` is `int`; declare "
                               "it `float`, or truncate with `int(...)`",
                               jaiChkSubjectOf(target, "the target"));
                return;
            }

            JaiType *result = jaiTypeBinaryResult((int)op, current, rhs);
            if (result == NULL) {
                char a[TYPE_BUF], b[TYPE_BUF];
                jaiChkRenderType(current, a, sizeof a);
                jaiChkRenderType(rhs, b, sizeof b);
                ERR(c, E0406_BAD_OPERAND_TYPES, node->span,
                    "operator `%s=` does not apply to `%s` and `%s`",
                    jaiOpKindText(op), a, b);
                return;
            }
            jaiChkRequireAssignable(c, NULL, result, current, E0400_TYPE_MISMATCH,
                              "the result of this compound assignment does not fit");
        }
        return;
    }

    switch (target->kind) {
    case AST_IDENT: {
        JaiType *want = jaiChkExpr(c, target);
        Symbol *sym = target->as.ident.symbol;
        /* The destination is the declaration, not the flow-narrowed view of
         * it: inside `if v != null { ... }` a `v = maybeNull()` is still a
         * legal store into a `T?`. */
        if (sym != NULL && sym->type != NULL) want = jaiChkDeclaredType(sym);
        target->type = jaiChkOrAny(want);
        checkStoredValue(c, value, want);
        jaiChkNarrowInvalidate(sym);
        break;
    }

    case AST_MEMBER:
    case AST_OPT_MEMBER:
        checkFieldAssign(c, node, target);
        break;

    case AST_INDEX: {
        JaiType *container = jaiChkValue(c, target->as.index.object);
        JaiType *index = jaiChkValue(c, target->as.index.index);
        JaiType *wantIndex = NULL, *element = NULL;
        if (!jaiChkIsAny(container) && jaiTypeIsIndexable(container, &wantIndex, &element)) {
            if (wantIndex != NULL && !jaiChkIsAny(index))
                jaiChkRequireAssignable(c, target->as.index.index, index, wantIndex,
                                  E0400_TYPE_MISMATCH, "invalid index type");
            if (element != NULL) jaiChkApplyContext(c, value, element);
        } else if (!jaiChkIsAny(container) && jaiChkTypeDeclOf(container) == NULL) {
            char got[TYPE_BUF];
            jaiChkRenderType(container, got, sizeof got);
            ERR(c, E0404_NOT_INDEXABLE, target->span,
                "`%s` does not support indexed assignment", got);
        }
        JaiType *got = jaiChkValue(c, value);
        if (element != NULL)
            jaiChkRequireAssignable(c, value, got, element, E0400_TYPE_MISMATCH,
                              "mismatched types in this assignment");
        target->type = jaiChkOrAny(element);
        break;
    }

    case AST_SLICE: {
        jaiChkSlice(c, target);
        jaiChkValue(c, value);
        break;
    }

    default:
        /* Destructuring assignment: `(a, b) = point`. */
        if (jaiAstIsPattern(target->kind)) {
            JaiType *got = jaiChkValue(c, value);
            jaiChkPattern(c, target, got);
        } else {
            jaiChkExpr(c, target);
            jaiChkValue(c, value);
        }
        break;
    }
}

/* ------------------------------------------------------------------ */
/* Statements                                                           */
/* ------------------------------------------------------------------ */

static bool isDeclaration(AstKind kind) {
    switch (kind) {
    case AST_FN_DECL: case AST_CLASS_DECL: case AST_TRAIT_DECL: case AST_ENUM_DECL:
    case AST_TYPE_DECL: case AST_IMPORT: case AST_FROM_IMPORT: case AST_EXPORT:
    case AST_MODULE_DECL:
        return true;
    default:
        return false;
    }
}

void jaiChkBlock(Checker *c, AstNode *node) {
    int mark = jaiChkNarrowMark();
    bool diverged = false;
    bool warned = false;

    for (int i = 0; i < node->as.block.count; i++) {
        AstNode *stmt = node->as.block.stmts == NULL ? NULL : node->as.block.stmts[i];
        if (stmt == NULL) continue;

        /* Declarations are hoisted by the resolver, so only real statements
         * after a diverging one are dead. */
        if (diverged && !warned && !isDeclaration(stmt->kind)) {
            WARN(W0102_UNREACHABLE_CODE, stmt->span,
                 "unreachable code: the previous statement always transfers control");
            warned = true;
        }
        jaiChkStmt(c, stmt);
        if (!diverged && jaiAstAlwaysDiverges(stmt)) diverged = true;
    }

    jaiChkNarrowRestore(mark);
}

static void checkIf(Checker *c, AstNode *node) {
    jaiChkCondition(c, node->as.conditional.cond, "an `if`");

    NarrowFact facts[MAX_FACTS];
    int count = jaiChkCollectFacts(c, node->as.conditional.cond, facts, MAX_FACTS, false);

    /* The false edge is collected separately rather than by inverting the true
     * edge: `a or b` yields nothing when true, but `not (a or b)` yields both
     * facts, and that is exactly the `if x is null or x <= 0 { return }` guard
     * the library is written with. */
    NarrowFact elseFacts[MAX_FACTS];
    int elseCount = jaiChkCollectFacts(c, node->as.conditional.cond, elseFacts, MAX_FACTS, true);

    int mark = jaiChkNarrowMark();
    jaiChkNarrowApply(facts, count, false);
    if (node->as.conditional.thenBranch != NULL) jaiChkStmt(c, node->as.conditional.thenBranch);
    jaiChkNarrowRestore(mark);

    if (node->as.conditional.elseBranch != NULL) {
        mark = jaiChkNarrowMark();
        jaiChkNarrowApply(elseFacts, elseCount, false);
        jaiChkStmt(c, node->as.conditional.elseBranch);
        jaiChkNarrowRestore(mark);
        return;
    }

    /* `if x == null { return }` narrows everything after it: the facts stay
     * applied and the enclosing block restores them when it ends. */
    if (jaiAstAlwaysDiverges(node->as.conditional.thenBranch))
        jaiChkNarrowApply(elseFacts, elseCount, false);
}

static void checkLoopStmt(Checker *c, AstNode *node) {
    if (node->kind == AST_WHILE && node->as.loop.cond != NULL) {
        jaiChkCondition(c, node->as.loop.cond, "a `while`");
        NarrowFact facts[MAX_FACTS];
        int count = jaiChkCollectFacts(c, node->as.loop.cond, facts, MAX_FACTS, false);
        int mark = jaiChkNarrowMark();
        jaiChkNarrowApply(facts, count, false);
        if (node->as.loop.body != NULL) jaiChkStmt(c, node->as.loop.body);
        jaiChkNarrowRestore(mark);
        return;
    }
    if (node->as.loop.body != NULL) jaiChkStmt(c, node->as.loop.body);
}

static void checkFor(Checker *c, AstNode *node) {
    JaiType *source = jaiChkValue(c, node->as.forLoop.iterable);
    JaiType *elem = jaiChkIterableElement(c, node->as.forLoop.iterable, source);

    int mark = jaiChkNarrowMark();
    jaiChkPattern(c, node->as.forLoop.pattern, elem);

    Symbol *iter = node->as.forLoop.iterSymbol;
    if (iter != NULL && iter->type == NULL) {
        AstNode *pattern = node->as.forLoop.pattern;
        bool isLoopVariable = pattern != NULL && pattern->kind == AST_PAT_BIND &&
                              pattern->as.patBind.symbol == iter;
        iter->type = isLoopVariable ? elem : gTypes.tAny;
    }

    if (node->as.forLoop.body != NULL) jaiChkStmt(c, node->as.forLoop.body);
    jaiChkNarrowRestore(mark);
}

static void checkReturn(Checker *c, AstNode *node) {
    JaiType *want = c->currentReturnType;
    AstNode *value = node->as.ret.value;

    if (value == NULL) {
        if (want != NULL && !jaiChkIsAny(want) && !jaiChkIsVoid(want) && want->kind != TY_NULL &&
            !jaiTypeIsOptional(want)) {
            char expected[TYPE_BUF];
            jaiChkRenderType(want, expected, sizeof expected);
            JaiDiag *d = ERR(c, E0408_RETURN_TYPE_MISMATCH, node->span,
                             "this function must return `%s`", expected);
            jaiDiagAddHelp(d, "add a value: `return <%s>`", expected);
        }
        return;
    }

    if (want != NULL && jaiChkIsVoid(want)) {
        jaiChkValue(c, value);
        ERR(c, E0408_RETURN_TYPE_MISMATCH, node->span,
            "a `-> void` function must not return a value");
        return;
    }

    if (want != NULL) jaiChkApplyContext(c, value, want);
    JaiType *got = jaiChkValue(c, value);
    if (want != NULL)
        jaiChkRequireAssignable(c, value, got, want, E0408_RETURN_TYPE_MISMATCH,
                          "mismatched return type");
}

static void checkThrow(Checker *c, AstNode *node) {
    if (node->as.ret.value == NULL) return;
    JaiType *t = jaiChkValue(c, node->as.ret.value);
    if (jaiChkIsAny(t) || jaiChkIsNever(t)) return;
    if (t->kind == TY_CLASS || t->kind == TY_TRAIT || t->kind == TY_GENERIC_PARAM) return;

    char got[TYPE_BUF];
    jaiChkRenderType(t, got, sizeof got);
    JaiDiag *d = ERR(c, E0400_TYPE_MISMATCH, node->span,
                     "only an `Error` instance can be thrown, found `%s`", got);
    jaiDiagAddHelp(d, "throw a class deriving from `Error`, e.g. "
                      "`ValueError(\"...\")`");
}

static void checkTry(Checker *c, AstNode *node) {
    if (node->as.tryStmt.body != NULL) jaiChkStmt(c, node->as.tryStmt.body);

    for (int i = 0; i < node->as.tryStmt.catchCount; i++) {
        AstCatch *handler = &node->as.tryStmt.catches[i];
        JaiType *caught = NULL;
        if (handler->typeCount > 0) {
            JaiType **members = jaiChkTypeArray(c, handler->typeCount);
            for (int j = 0; j < handler->typeCount; j++)
                members[j] = jaiChkResolveAstType(c, handler->types[j]);
            caught = handler->typeCount == 1 ? members[0]
                                             : jaiTypeUnion(members, handler->typeCount);
        } else {
            caught = jaiChkOrAny(jaiChkLookupTypeName(c, "Error"));
        }
        if (handler->symbol != NULL) handler->symbol->type = jaiChkOrAny(caught);
        if (handler->body != NULL) jaiChkStmt(c, handler->body);
    }

    if (node->as.tryStmt.finallyBlock != NULL) jaiChkStmt(c, node->as.tryStmt.finallyBlock);
}

static void checkAssert(Checker *c, AstNode *node) {
    jaiChkCondition(c, node->as.assertStmt.cond, "an `assert`");
    if (node->as.assertStmt.message == NULL) return;
    JaiType *t = jaiChkValue(c, node->as.assertStmt.message);
    if (!jaiChkIsAny(t) && t->kind != TY_STR)
        jaiChkRequireAssignable(c, node->as.assertStmt.message, t, gTypes.tStr,
                          E0400_TYPE_MISMATCH, "an assertion message must be a `str`");
}

void jaiChkStmt(Checker *c, AstNode *node) {
    if (node == NULL) return;

    switch (node->kind) {
    case AST_BLOCK:
    case AST_PROGRAM:
        jaiChkBlock(c, node);
        break;

    case AST_EXPR_STMT:
        /* The only place a `void` value is allowed: nothing consumes it. */
        if (node->as.exprStmt.expr != NULL) jaiChkExpr(c, node->as.exprStmt.expr);
        break;

    case AST_VAR_DECL: checkVarDecl(c, node);   break;
    case AST_ASSIGN:   checkAssign(c, node);    break;
    case AST_IF:       checkIf(c, node);        break;

    case AST_WHILE:
    case AST_LOOP:
        checkLoopStmt(c, node);
        break;

    case AST_FOR:      checkFor(c, node);       break;

    case AST_MATCH:
        node->type = jaiChkMatchExpr(c, node, true);
        break;

    case AST_RETURN:   checkReturn(c, node);    break;
    case AST_THROW:    checkThrow(c, node);     break;
    case AST_TRY:      checkTry(c, node);       break;
    case AST_ASSERT:   checkAssert(c, node);    break;

    case AST_DEFER:
        if (node->as.defer.body != NULL) jaiChkStmt(c, node->as.defer.body);
        break;

    case AST_BREAK:
    case AST_CONTINUE:
        break;

    default:
        if (isDeclaration(node->kind)) { jaiChkDecl(c, node); break; }
        if (jaiAstIsExpression(node->kind)) { jaiChkExpr(c, node); break; }
        break;
    }
}

/* ------------------------------------------------------------------ */
/* Functions                                                            */
/* ------------------------------------------------------------------ */

void jaiChkPushGenerics(Checker *c, AstGeneric *generics, int count) {
    if (generics == NULL) return;
    for (int i = 0; i < count; i++)
        jaiChkPushTypeName(generics[i].name, jaiTypeGenericParam(generics[i].name));
    /* Bounds are resolved after the parameters are in scope so that
     * `[T: Comparable[T]]` sees itself. */
    for (int i = 0; i < count; i++)
        if (generics[i].bound != NULL) (void)jaiChkResolveAstType(c, generics[i].bound);
}

bool jaiChkIsInitMethod(const AstNode *fn) {
    return fn->kind == AST_FN_DECL && jaiChkSameName(fn->as.fn.name, "init");
}

/* The declared return type, with the two places where omission means `void`
 * rather than `any`: constructors and property setters. */
JaiType *jaiChkDeclaredReturnType(Checker *c, AstNode *fn, bool isMethod) {
    if (fn->as.fn.returnType != NULL && fn->as.fn.returnType->kind != TYPE_INFER)
        return jaiChkResolveAstType(c, fn->as.fn.returnType);
    if (isMethod && jaiChkIsInitMethod(fn)) return gTypes.tVoid;
    return gTypes.tAny;
}

JaiType *jaiChkFunctionType(Checker *c, AstNode *fn, bool isMethod) {
    int mark = jaiChkTypeNameMark();
    jaiChkPushGenerics(c, fn->as.fn.generics, fn->as.fn.genericCount);

    int skip = isMethod ? jaiChkSelfSkipOf(fn) : 0;
    int count = fn->as.fn.paramCount - skip;
    if (count < 0) count = 0;

    uint8_t flags = 0;
    JaiType **params = jaiChkTypeArray(c, count);
    for (int i = 0; i < count; i++) {
        AstParam *p = &fn->as.fn.params[skip + i];
        if (p->isVariadic) flags |= FN_FLAG_VARIADIC;
        if (p->isKwRest) flags |= FN_FLAG_KWREST;
        params[i] = p->type != NULL ? jaiChkResolveAstType(c, p->type) : gTypes.tAny;
    }

    JaiType *ret = jaiChkDeclaredReturnType(c, fn, isMethod);
    jaiChkTypeNameRestore(mark);
    return jaiTypeFn(params, count, ret, flags);
}

/* The type a parameter's binding has inside the body, which is not the type in
 * the signature for `...xs` and `**opts`. */
static JaiType *parameterBindingType(Checker *c, AstParam *p) {
    JaiType *declared = p->type != NULL ? jaiChkResolveAstType(c, p->type) : gTypes.tAny;
    if (p->isVariadic) return jaiTypeList(declared);
    if (p->isKwRest) return jaiTypeDict(gTypes.tStr, declared);
    return declared;
}

void jaiChkFunction(Checker *c, AstNode *fn, TypeDecl *owner, bool isMethod) {
    int nameMark = jaiChkTypeNameMark();
    jaiChkPushGenerics(c, fn->as.fn.generics, fn->as.fn.genericCount);

    int narrow = jaiChkNarrowMark();
    JaiType *savedReturn = c->currentReturnType;
    TypeDecl *savedClass = c->currentClass;
    if (owner != NULL) c->currentClass = owner;

    int skip = isMethod ? jaiChkSelfSkipOf(fn) : 0;
    for (int i = 0; i < fn->as.fn.paramCount; i++) {
        AstParam *p = &fn->as.fn.params[i];
        JaiType *bound;
        if (i < skip) {
            bound = owner != NULL ? jaiChkDeclType(owner) : gTypes.tAny;
        } else {
            bound = parameterBindingType(c, p);
            if (c->strict && p->type == NULL && !p->isKwRest) {
                JaiDiag *d = ERR(c, E0400_TYPE_MISMATCH, p->span,
                                 "parameter `%s` has no type annotation", p->name);
                jaiDiagAddHelp(d, "--strict requires every parameter to be annotated");
            }
        }
        if (p->symbol != NULL) p->symbol->type = bound;
        if (p->defaultValue != NULL) {
            jaiChkApplyContext(c, p->defaultValue, bound);
            JaiType *got = jaiChkValue(c, p->defaultValue);
            jaiChkRequireAssignable(c, p->defaultValue, got, bound, E0400_TYPE_MISMATCH,
                              "mismatched type for a default value");
        }
    }

    JaiType *ret = jaiChkDeclaredReturnType(c, fn, isMethod);
    c->currentReturnType = ret;

    AstNode *body = fn->as.fn.body;
    if (body != NULL) {
        if (body->kind == AST_BLOCK) {
            jaiChkBlock(c, body);
        } else {
            /* A lambda's body is its return value. */
            if (!jaiChkIsAny(ret)) jaiChkApplyContext(c, body, ret);
            JaiType *got = jaiChkValue(c, body);
            if (fn->as.fn.returnType != NULL && fn->as.fn.returnType->kind != TYPE_INFER)
                jaiChkRequireAssignable(c, body, got, ret, E0408_RETURN_TYPE_MISMATCH,
                                  "mismatched return type");
            else
                ret = got;
        }
    }

    /* E0601: an annotated non-void function must not fall off the end. A
     * generator returns through `yield`, so it is exempt. */
    if (body != NULL && body->kind == AST_BLOCK && !fn->as.fn.isGenerator &&
        fn->as.fn.returnType != NULL && fn->as.fn.returnType->kind != TYPE_INFER &&
        !jaiChkIsVoid(ret) && !jaiChkIsAny(ret) && !jaiChkIsNever(ret) && ret->kind != TY_NULL &&
        !jaiTypeIsOptional(ret) && !jaiAstAlwaysDiverges(body)) {
        char want[TYPE_BUF];
        jaiChkRenderType(ret, want, sizeof want);
        JaiDiag *d = ERR(c, E0601_MISSING_RETURN, fn->span,
                         "`%s` must return `%s` on every path",
                         fn->as.fn.name != NULL ? fn->as.fn.name : "this function", want);
        jaiDiagAddLabel(d, fn->as.fn.returnType->span, "declared to return `%s`", want);
        jaiDiagAddHelp(d, "add a `return` at the end, or make the last branch "
                          "an `else`");
    }

    c->currentReturnType = savedReturn;
    c->currentClass = savedClass;
    jaiChkNarrowRestore(narrow);
    jaiChkTypeNameRestore(nameMark);
}
