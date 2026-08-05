/* ast.c — AST construction, traversal, printing, and JSON serialisation.
 *
 * jaiAstWalk is the load-bearing routine: every later pass (resolver, checker,
 * const folder, code generator) trusts that reaching a node means reaching all
 * of its children. Its switch is therefore exhaustive over AstKind and panics
 * on an unknown kind instead of quietly skipping a subtree. Add a node kind to
 * ast.h and this file stops compiling clean or stops running clean — by design.
 */
#include "ast.h"

#include <math.h>

/* ------------------------------------------------------------------ */
/* Kind metadata                                                        */
/* ------------------------------------------------------------------ */

/* Positional, in AstKind declaration order. The static assert on the length is
 * what keeps this honest when a kind is added. */
static const char *const kKindNames[] = {
    "AST_INT_LIT", "AST_FLOAT_LIT", "AST_STR_LIT", "AST_BOOL_LIT", "AST_NULL_LIT",
    "AST_FSTRING",
    "AST_IDENT",
    "AST_SELF", "AST_SUPER",
    "AST_LIST_LIT", "AST_DICT_LIT", "AST_SET_LIT", "AST_TUPLE_LIT",
    "AST_UNARY", "AST_BINARY", "AST_LOGICAL", "AST_COMPARE_CHAIN",
    "AST_TERNARY", "AST_COALESCE",
    "AST_CALL", "AST_INDEX", "AST_SLICE", "AST_MEMBER", "AST_OPT_MEMBER",
    "AST_LAMBDA", "AST_ANON_FN",
    "AST_COMPREHENSION",
    "AST_RANGE",
    "AST_IF_EXPR", "AST_MATCH_EXPR",
    "AST_CAST",
    "AST_YIELD", "AST_AWAIT",
    "AST_THROW_EXPR",

    "AST_BLOCK",
    "AST_EXPR_STMT",
    "AST_VAR_DECL",
    "AST_ASSIGN",
    "AST_IF", "AST_WHILE", "AST_LOOP", "AST_FOR", "AST_MATCH",
    "AST_BREAK", "AST_CONTINUE", "AST_RETURN", "AST_THROW",
    "AST_TRY", "AST_DEFER", "AST_ASSERT",

    "AST_FN_DECL", "AST_CLASS_DECL", "AST_TRAIT_DECL", "AST_ENUM_DECL",
    "AST_TYPE_DECL", "AST_IMPORT", "AST_FROM_IMPORT", "AST_EXPORT",
    "AST_MODULE_DECL",

    "AST_PAT_WILDCARD", "AST_PAT_BIND", "AST_PAT_LITERAL", "AST_PAT_RANGE",
    "AST_PAT_TUPLE", "AST_PAT_LIST", "AST_PAT_CLASS", "AST_PAT_ENUM",
    "AST_PAT_OR",

    "AST_PROGRAM",
};
_Static_assert(sizeof kKindNames / sizeof kKindNames[0] == AST_KIND_COUNT,
               "kKindNames is out of sync with AstKind");

enum {
    CAT_EXPR = 1u << 0,
    CAT_STMT = 1u << 1,
    CAT_PAT  = 1u << 2,
    CAT_DECL = 1u << 3,   /* also a statement: declarations live in blocks */
    CAT_ROOT = 1u << 4,   /* AST_PROGRAM: neither expression nor statement */
};

static const uint8_t kKindCategory[] = {
    CAT_EXPR, CAT_EXPR, CAT_EXPR, CAT_EXPR, CAT_EXPR,   /* int float str bool null */
    CAT_EXPR,                                            /* fstring */
    CAT_EXPR,                                            /* ident */
    CAT_EXPR, CAT_EXPR,                                  /* self super */
    CAT_EXPR, CAT_EXPR, CAT_EXPR, CAT_EXPR,              /* list dict set tuple */
    CAT_EXPR, CAT_EXPR, CAT_EXPR, CAT_EXPR,              /* unary binary logical chain */
    CAT_EXPR, CAT_EXPR,                                  /* ternary coalesce */
    CAT_EXPR, CAT_EXPR, CAT_EXPR, CAT_EXPR, CAT_EXPR,    /* call index slice member optmember */
    CAT_EXPR, CAT_EXPR,                                  /* lambda anonfn */
    CAT_EXPR,                                            /* comprehension */
    CAT_EXPR,                                            /* range */
    CAT_EXPR, CAT_EXPR,                                  /* if-expr match-expr */
    CAT_EXPR,                                            /* cast */
    CAT_EXPR, CAT_EXPR,                                  /* yield await */
    CAT_EXPR,                                            /* throw-expr */

    CAT_STMT,                                            /* block */
    CAT_STMT,                                            /* expr-stmt */
    CAT_STMT,                                            /* var-decl */
    CAT_STMT,                                            /* assign */
    CAT_STMT, CAT_STMT, CAT_STMT, CAT_STMT, CAT_STMT,    /* if while loop for match */
    CAT_STMT, CAT_STMT, CAT_STMT, CAT_STMT,              /* break continue return throw */
    CAT_STMT, CAT_STMT, CAT_STMT,                        /* try defer assert */

    CAT_DECL, CAT_DECL, CAT_DECL, CAT_DECL,              /* fn class trait enum */
    CAT_DECL, CAT_DECL, CAT_DECL, CAT_DECL, CAT_DECL,    /* type import from export module */

    CAT_PAT, CAT_PAT, CAT_PAT, CAT_PAT,                  /* wildcard bind literal range */
    CAT_PAT, CAT_PAT, CAT_PAT, CAT_PAT, CAT_PAT,         /* tuple list class enum or */

    CAT_ROOT,                                            /* program */
};
_Static_assert(sizeof kKindCategory / sizeof kKindCategory[0] == AST_KIND_COUNT,
               "kKindCategory is out of sync with AstKind");

static uint8_t categoryOf(AstKind kind) {
    if ((unsigned)kind >= (unsigned)AST_KIND_COUNT) return 0;
    return kKindCategory[kind];
}

const char *jaiAstKindName(AstKind kind) {
    if ((unsigned)kind >= (unsigned)AST_KIND_COUNT) return "<invalid-ast-kind>";
    return kKindNames[kind];
}

bool jaiAstIsExpression(AstKind kind) { return (categoryOf(kind) & CAT_EXPR) != 0; }
bool jaiAstIsPattern(AstKind kind)    { return (categoryOf(kind) & CAT_PAT) != 0; }

bool jaiAstIsStatement(AstKind kind) {
    return (categoryOf(kind) & (CAT_STMT | CAT_DECL)) != 0;
}

/* ------------------------------------------------------------------ */
/* Operators                                                            */
/* ------------------------------------------------------------------ */

static const char *const kOpText[] = {
    "+", "-", "*", "/", "//", "%", "**",
    "+%", "-%", "*%",
    "&", "|", "^", "<<", ">>", "@",
    "==", "!=", "<", "<=", ">", ">=",
    "is", "is not", "in", "not in",
    "and", "or",
    "-", "+", "not", "~",
};
_Static_assert(sizeof kOpText / sizeof kOpText[0] == OPK_COUNT,
               "kOpText is out of sync with OpKind");

static const char *const kOpNames[] = {
    "OPK_ADD", "OPK_SUB", "OPK_MUL", "OPK_DIV", "OPK_FLOORDIV", "OPK_MOD", "OPK_POW",
    "OPK_ADD_WRAP", "OPK_SUB_WRAP", "OPK_MUL_WRAP",
    "OPK_BAND", "OPK_BOR", "OPK_BXOR", "OPK_SHL", "OPK_SHR", "OPK_MATMUL",
    "OPK_EQ", "OPK_NE", "OPK_LT", "OPK_LE", "OPK_GT", "OPK_GE",
    "OPK_IS", "OPK_IS_NOT", "OPK_IN", "OPK_NOT_IN",
    "OPK_AND", "OPK_OR",
    "OPK_NEG", "OPK_POS", "OPK_NOT", "OPK_BNOT",
};
_Static_assert(sizeof kOpNames / sizeof kOpNames[0] == OPK_COUNT,
               "kOpNames is out of sync with OpKind");

const char *jaiOpKindText(OpKind op) {
    if ((unsigned)op >= (unsigned)OPK_COUNT) return "<invalid-op>";
    return kOpText[op];
}

static const char *opEnumName(OpKind op) {
    if ((unsigned)op >= (unsigned)OPK_COUNT) return "<invalid-op>";
    return kOpNames[op];
}

bool jaiOpKindIsComparison(OpKind op) {
    return op >= OPK_EQ && op <= OPK_NOT_IN;
}

static const char *varDeclKindName(VarDeclKind kind) {
    switch (kind) {
    case VD_LET:   return "VD_LET";
    case VD_VAR:   return "VD_VAR";
    case VD_CONST: return "VD_CONST";
    }
    return "<invalid-vardecl-kind>";
}

static const char *visibilityName(AstVisibility vis) {
    switch (vis) {
    case AST_VIS_PRIVATE:   return "AST_VIS_PRIVATE";
    case AST_VIS_PROTECTED: return "AST_VIS_PROTECTED";
    case AST_VIS_PUBLIC:    return "AST_VIS_PUBLIC";
    }
    return "<invalid-visibility>";
}

static const char *compKindName(CompKind kind) {
    switch (kind) {
    case COMP_LIST:      return "COMP_LIST";
    case COMP_DICT:      return "COMP_DICT";
    case COMP_SET:       return "COMP_SET";
    case COMP_GENERATOR: return "COMP_GENERATOR";
    }
    return "<invalid-comp-kind>";
}

static const char *typeKindName(AstTypeKind kind) {
    switch (kind) {
    case TYPE_NAME:     return "TYPE_NAME";
    case TYPE_GENERIC:  return "TYPE_GENERIC";
    case TYPE_OPTIONAL: return "TYPE_OPTIONAL";
    case TYPE_UNION:    return "TYPE_UNION";
    case TYPE_FN:       return "TYPE_FN";
    case TYPE_TUPLE:    return "TYPE_TUPLE";
    case TYPE_INFER:    return "TYPE_INFER";
    }
    return "<invalid-type-kind>";
}

/* ------------------------------------------------------------------ */
/* Construction                                                         */
/* ------------------------------------------------------------------ */

#define AST_ARENA_BLOCK ((size_t)(64 * 1024))

void jaiAstContextInit(AstContext *ctx) {
    if (ctx == NULL) JAI_PANIC("jaiAstContextInit: NULL context");
    jaiArenaInit(&ctx->arena, AST_ARENA_BLOCK);
    ctx->nodeCount = 0;
}

void jaiAstContextFree(AstContext *ctx) {
    if (ctx == NULL) return;
    jaiArenaFree(&ctx->arena);
    ctx->nodeCount = 0;
}

AstNode *jaiAstNew(AstContext *ctx, AstKind kind, JaiSpan span) {
    if (ctx == NULL) JAI_PANIC("jaiAstNew: NULL context");
    if ((unsigned)kind >= (unsigned)AST_KIND_COUNT)
        JAI_PANIC("jaiAstNew: invalid AST kind %d", (int)kind);

    AstNode *node = JAI_ARENA_NEW(&ctx->arena, AstNode);
    node->kind = kind;
    node->span = span;
    ctx->nodeCount++;
    return node;
}

AstType *jaiAstTypeNew(AstContext *ctx, AstTypeKind kind, JaiSpan span) {
    if (ctx == NULL) JAI_PANIC("jaiAstTypeNew: NULL context");
    AstType *type = JAI_ARENA_NEW(&ctx->arena, AstType);
    type->kind = kind;
    type->span = span;
    return type;
}

AstNode **jaiAstNodeArray(AstContext *ctx, int count) {
    if (ctx == NULL) JAI_PANIC("jaiAstNodeArray: NULL context");
    if (count <= 0) return NULL;   /* an empty array has no storage to hand out */
    return JAI_ARENA_NEW_ARRAY(&ctx->arena, AstNode *, count);
}

/* ------------------------------------------------------------------ */
/* Traversal                                                            */
/* ------------------------------------------------------------------ */

static void walkChildren(AstNode *node, AstVisitor pre, AstVisitor post,
                         void *userData);

/* Walk order is pre(node), children left to right, post(node). `pre` returning
 * false prunes the children only — `post` still runs for the node itself, so a
 * visitor may keep balanced enter/leave state across a pruned subtree. */
void jaiAstWalk(AstNode *node, AstVisitor pre, AstVisitor post, void *userData) {
    if (node == NULL) return;

    bool descend = true;
    if (pre != NULL) descend = pre(node, userData);
    if (descend) walkChildren(node, pre, post, userData);
    if (post != NULL) (void)post(node, userData);
}

static void walkArray(AstNode **nodes, int count, AstVisitor pre,
                      AstVisitor post, void *userData) {
    if (nodes == NULL) return;
    for (int i = 0; i < count; i++) jaiAstWalk(nodes[i], pre, post, userData);
}

static void walkChildren(AstNode *node, AstVisitor pre, AstVisitor post,
                         void *userData) {
#define WALK(child) jaiAstWalk((child), pre, post, userData)
#define WALK_N(arr, n) walkArray((arr), (n), pre, post, userData)

    switch (node->kind) {
    case AST_INT_LIT:
    case AST_FLOAT_LIT:
    case AST_STR_LIT:
    case AST_BOOL_LIT:
    case AST_NULL_LIT:
    case AST_IDENT:
    case AST_SELF:
    case AST_SUPER:
        break;

    case AST_FSTRING:
        WALK_N(node->as.fstring.parts, node->as.fstring.partCount);
        break;

    case AST_LIST_LIT:
    case AST_SET_LIT:
    case AST_TUPLE_LIT:
        WALK_N(node->as.sequence.items, node->as.sequence.count);
        break;

    case AST_DICT_LIT:
        for (int i = 0; i < node->as.dict.count; i++) {
            if (node->as.dict.keys != NULL) WALK(node->as.dict.keys[i]);
            if (node->as.dict.values != NULL) WALK(node->as.dict.values[i]);
        }
        break;

    case AST_UNARY:
        WALK(node->as.unary.operand);
        break;

    case AST_BINARY:
    case AST_LOGICAL:
        WALK(node->as.binary.left);
        WALK(node->as.binary.right);
        break;

    /* A chain of n operators has n + 1 operands. */
    case AST_COMPARE_CHAIN:
        WALK_N(node->as.chain.operands, node->as.chain.opCount + 1);
        break;

    case AST_TERNARY:
        WALK(node->as.ternary.cond);
        WALK(node->as.ternary.thenExpr);
        WALK(node->as.ternary.elseExpr);
        break;

    case AST_COALESCE:
        WALK(node->as.coalesce.left);
        WALK(node->as.coalesce.right);
        break;

    case AST_CALL:
        WALK(node->as.call.callee);
        if (node->as.call.args != NULL)
            for (int i = 0; i < node->as.call.argCount; i++)
                WALK(node->as.call.args[i].value);
        break;

    case AST_INDEX:
        WALK(node->as.index.object);
        WALK(node->as.index.index);
        break;

    case AST_SLICE:
        WALK(node->as.slice.object);
        WALK(node->as.slice.start);
        WALK(node->as.slice.stop);
        WALK(node->as.slice.step);
        break;

    case AST_MEMBER:
    case AST_OPT_MEMBER:
        WALK(node->as.member.object);
        break;

    case AST_LAMBDA:
    case AST_ANON_FN:
    case AST_FN_DECL:
        if (node->as.fn.params != NULL)
            for (int i = 0; i < node->as.fn.paramCount; i++)
                WALK(node->as.fn.params[i].defaultValue);
        WALK(node->as.fn.body);
        break;

    case AST_COMPREHENSION:
        WALK(node->as.comp.keyExpr);
        WALK(node->as.comp.element);
        if (node->as.comp.clauses != NULL) {
            for (int i = 0; i < node->as.comp.clauseCount; i++) {
                AstCompClause *clause = &node->as.comp.clauses[i];
                WALK(clause->pattern);
                WALK(clause->iterable);
                WALK_N(clause->conditions, clause->conditionCount);
            }
        }
        break;

    case AST_RANGE:
        WALK(node->as.range.start);
        WALK(node->as.range.stop);
        break;

    case AST_IF_EXPR:
    case AST_IF:
        WALK(node->as.conditional.cond);
        WALK(node->as.conditional.thenBranch);
        WALK(node->as.conditional.elseBranch);
        break;

    case AST_MATCH:
    case AST_MATCH_EXPR:
        WALK(node->as.match.subject);
        if (node->as.match.arms != NULL) {
            for (int i = 0; i < node->as.match.armCount; i++) {
                AstMatchArm *arm = &node->as.match.arms[i];
                WALK(arm->pattern);
                WALK(arm->guard);
                WALK(arm->body);
            }
        }
        break;

    case AST_CAST:
        WALK(node->as.cast.operand);
        break;

    /* AST_THROW_EXPR has no union member of its own; it borrows `wrap`. That is
     * safe whichever single-pointer member the parser writes through, because
     * `wrap` and `ret` share a common initial sequence of one AstNode*. */
    case AST_YIELD:
    case AST_AWAIT:
    case AST_THROW_EXPR:
        WALK(node->as.wrap.operand);
        break;

    case AST_BLOCK:
    case AST_PROGRAM:
        WALK_N(node->as.block.stmts, node->as.block.count);
        break;

    case AST_EXPR_STMT:
        WALK(node->as.exprStmt.expr);
        break;

    case AST_VAR_DECL:
        WALK(node->as.varDecl.pattern);
        WALK(node->as.varDecl.init);
        break;

    case AST_ASSIGN:
        WALK(node->as.assign.target);
        WALK(node->as.assign.value);
        break;

    case AST_WHILE:
    case AST_LOOP:
        WALK(node->as.loop.cond);      /* NULL for `loop` */
        WALK(node->as.loop.body);
        break;

    case AST_FOR:
        WALK(node->as.forLoop.pattern);
        WALK(node->as.forLoop.iterable);
        WALK(node->as.forLoop.body);
        break;

    case AST_BREAK:
    case AST_CONTINUE:
        break;

    case AST_RETURN:
    case AST_THROW:
        WALK(node->as.ret.value);
        break;

    case AST_TRY:
        WALK(node->as.tryStmt.body);
        if (node->as.tryStmt.catches != NULL)
            for (int i = 0; i < node->as.tryStmt.catchCount; i++)
                WALK(node->as.tryStmt.catches[i].body);
        WALK(node->as.tryStmt.finallyBlock);
        break;

    case AST_DEFER:
        WALK(node->as.defer.body);
        break;

    case AST_ASSERT:
        WALK(node->as.assertStmt.cond);
        WALK(node->as.assertStmt.message);
        break;

    case AST_CLASS_DECL:
        if (node->as.classDecl.fields != NULL)
            for (int i = 0; i < node->as.classDecl.fieldCount; i++)
                WALK(node->as.classDecl.fields[i].defaultValue);
        WALK_N(node->as.classDecl.methods, node->as.classDecl.methodCount);
        WALK_N(node->as.classDecl.getters, node->as.classDecl.getterCount);
        WALK_N(node->as.classDecl.setters, node->as.classDecl.setterCount);
        break;

    case AST_TRAIT_DECL:
        WALK_N(node->as.traitDecl.methods, node->as.traitDecl.methodCount);
        break;

    case AST_ENUM_DECL:
        if (node->as.enumDecl.variants != NULL) {
            for (int i = 0; i < node->as.enumDecl.variantCount; i++) {
                AstVariant *variant = &node->as.enumDecl.variants[i];
                if (variant->params == NULL) continue;
                for (int j = 0; j < variant->paramCount; j++)
                    WALK(variant->params[j].defaultValue);
            }
        }
        WALK_N(node->as.enumDecl.methods, node->as.enumDecl.methodCount);
        break;

    case AST_TYPE_DECL:
    case AST_IMPORT:
    case AST_FROM_IMPORT:
    case AST_EXPORT:
    case AST_MODULE_DECL:
    case AST_PAT_WILDCARD:
    case AST_PAT_BIND:
        break;

    case AST_PAT_LITERAL:
        WALK(node->as.patLiteral.value);
        break;

    case AST_PAT_RANGE:
        WALK(node->as.patRange.lo);
        WALK(node->as.patRange.hi);
        break;

    case AST_PAT_TUPLE:
    case AST_PAT_LIST:
    case AST_PAT_OR:
        WALK_N(node->as.patSeq.elems, node->as.patSeq.count);
        break;

    case AST_PAT_CLASS:
    case AST_PAT_ENUM:
        WALK_N(node->as.patClass.subPatterns, node->as.patClass.count);
        break;

    default:
        JAI_PANIC("jaiAstWalk: unhandled AST kind %d (%s)", (int)node->kind,
                  jaiAstKindName(node->kind));
    }

#undef WALK
#undef WALK_N
}

/* ------------------------------------------------------------------ */
/* Predicates                                                           */
/* ------------------------------------------------------------------ */

bool jaiAstIsAssignable(const AstNode *node) {
    if (node == NULL) return false;

    switch (node->kind) {
    case AST_IDENT:
    case AST_MEMBER:
    case AST_INDEX:
    case AST_SLICE:
        return true;

    /* `x?.y = v` has no meaning: the whole point of `?.` is that it may not
     * reach an object at all. */
    case AST_OPT_MEMBER:
        return false;

    /* A bind or wildcard is a name being written to, which is what assignability
     * means; they appear as the leaves of a destructuring target. */
    case AST_PAT_BIND:
    case AST_PAT_WILDCARD:
        return true;

    case AST_TUPLE_LIT:
    case AST_LIST_LIT:
        if (node->as.sequence.count <= 0 || node->as.sequence.items == NULL)
            return false;
        for (int i = 0; i < node->as.sequence.count; i++)
            if (!jaiAstIsAssignable(node->as.sequence.items[i])) return false;
        return true;

    case AST_PAT_TUPLE:
    case AST_PAT_LIST:
        if (node->as.patSeq.count <= 0 || node->as.patSeq.elems == NULL)
            return false;
        for (int i = 0; i < node->as.patSeq.count; i++)
            if (!jaiAstIsAssignable(node->as.patSeq.elems[i])) return false;
        return true;

    default:
        return false;
    }
}

typedef struct {
    const char *label;     /* label of the loop under test; may be NULL */
    int         loopDepth; /* nested loops entered since the search began */
    bool        found;
} BreakSearch;

static bool isLoopKind(AstKind kind) {
    return kind == AST_WHILE || kind == AST_LOOP || kind == AST_FOR;
}

static bool isFunctionKind(AstKind kind) {
    return kind == AST_FN_DECL || kind == AST_LAMBDA || kind == AST_ANON_FN;
}

static bool breakSearchPre(AstNode *node, void *userData) {
    BreakSearch *search = (BreakSearch *)userData;

    /* Control never crosses a function or type boundary, so a `break` down
     * there cannot be ours. */
    if (isFunctionKind(node->kind) || node->kind == AST_CLASS_DECL ||
        node->kind == AST_TRAIT_DECL || node->kind == AST_ENUM_DECL) {
        return false;
    }

    if (isLoopKind(node->kind)) {
        search->loopDepth++;      /* paired with the decrement in the post hook */
        return !search->found;
    }

    if (node->kind == AST_BREAK) {
        const char *target = node->as.jump.label;
        if (target == NULL) {
            /* Unlabelled: binds to the innermost enclosing loop. */
            if (search->loopDepth == 0) search->found = true;
        } else if (search->label != NULL && strcmp(target, search->label) == 0) {
            search->found = true;
        }
    }
    return !search->found;
}

static bool breakSearchPost(AstNode *node, void *userData) {
    BreakSearch *search = (BreakSearch *)userData;
    if (isLoopKind(node->kind)) search->loopDepth--;
    return true;
}

/* Conservative: any `break` binding to this loop counts, whether or not it is
 * reachable. Claiming a loop diverges when it can in fact exit would silence a
 * real missing-return error, so we only ever err towards "does not diverge". */
static bool loopHasBreak(const AstNode *node) {
    BreakSearch search;
    search.label = (node->kind == AST_FOR) ? node->as.forLoop.label
                                           : node->as.loop.label;
    search.loopDepth = 0;
    search.found = false;

    AstNode *body = (node->kind == AST_FOR) ? node->as.forLoop.body
                                            : node->as.loop.body;
    jaiAstWalk(body, breakSearchPre, breakSearchPost, &search);
    return search.found;
}

static bool isTrueLiteral(const AstNode *node) {
    return node != NULL && node->kind == AST_BOOL_LIT && node->as.boolLit;
}

bool jaiAstAlwaysDiverges(const AstNode *node) {
    if (node == NULL) return false;

    switch (node->kind) {
    case AST_RETURN:
    case AST_THROW:
    case AST_THROW_EXPR:
    case AST_BREAK:
    case AST_CONTINUE:
        return true;

    /* Once one statement diverges the rest of the block is unreachable, so the
     * block as a whole diverges — this subsumes "the last statement diverges". */
    case AST_BLOCK:
    case AST_PROGRAM:
        for (int i = 0; i < node->as.block.count; i++) {
            if (node->as.block.stmts == NULL) break;
            if (jaiAstAlwaysDiverges(node->as.block.stmts[i])) return true;
        }
        return false;

    case AST_EXPR_STMT:
        return jaiAstAlwaysDiverges(node->as.exprStmt.expr);

    case AST_IF:
    case AST_IF_EXPR:
        return node->as.conditional.elseBranch != NULL &&
               jaiAstAlwaysDiverges(node->as.conditional.thenBranch) &&
               jaiAstAlwaysDiverges(node->as.conditional.elseBranch);

    /* `loop` and `while true` run forever unless something breaks out. */
    case AST_LOOP:
        return !loopHasBreak(node);
    case AST_WHILE:
        return isTrueLiteral(node->as.loop.cond) && !loopHasBreak(node);

    /* A `for` may iterate zero times, so it can always fall through. */
    case AST_FOR:
        return false;

    case AST_MATCH:
    case AST_MATCH_EXPR:
        if (node->as.match.armCount <= 0 || node->as.match.arms == NULL)
            return false;
        for (int i = 0; i < node->as.match.armCount; i++)
            if (!jaiAstAlwaysDiverges(node->as.match.arms[i].body)) return false;
        return true;

    /* A `finally` that diverges wins outright; otherwise every path — the body
     * and each handler — has to diverge. */
    case AST_TRY:
        if (jaiAstAlwaysDiverges(node->as.tryStmt.finallyBlock)) return true;
        if (!jaiAstAlwaysDiverges(node->as.tryStmt.body)) return false;
        for (int i = 0; i < node->as.tryStmt.catchCount; i++) {
            if (node->as.tryStmt.catches == NULL) break;
            if (!jaiAstAlwaysDiverges(node->as.tryStmt.catches[i].body))
                return false;
        }
        return true;

    default:
        return false;
    }
}

/* ------------------------------------------------------------------ */
/* S-expression printing                                                */
/* ------------------------------------------------------------------ */

static void printNode(FILE *out, const AstNode *node, int indent);

static void printIndent(FILE *out, int indent) {
    for (int i = 0; i < indent; i++) fputs("  ", out);
}

/* Kind tag without the AST_ prefix, lowercased: AST_INT_LIT -> int_lit. */
static void printKindTag(FILE *out, AstKind kind) {
    const char *name = jaiAstKindName(kind);
    if (strncmp(name, "AST_", 4) == 0) name += 4;
    for (const char *p = name; *p != '\0'; p++) {
        int c = (unsigned char)*p;
        fputc(c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c, out);
    }
}

static void printQuoted(FILE *out, const char *chars, size_t length) {
    fputc('"', out);
    for (size_t i = 0; i < length; i++) {
        unsigned char c = (unsigned char)chars[i];
        switch (c) {
        case '"':  fputs("\\\"", out); break;
        case '\\': fputs("\\\\", out); break;
        case '\n': fputs("\\n", out); break;
        case '\r': fputs("\\r", out); break;
        case '\t': fputs("\\t", out); break;
        default:
            if (c < 0x20) fprintf(out, "\\x%02x", c);
            else fputc((int)c, out);
        }
    }
    fputc('"', out);
}

static void printFloat(FILE *out, double value) {
    if (isnan(value))      fputs("nan", out);
    else if (isinf(value)) fputs(value > 0 ? "inf" : "-inf", out);
    else                   fprintf(out, "%.17g", value);
}

static void printName(FILE *out, const char *key, const char *name) {
    if (name == NULL) return;
    fprintf(out, " %s=%s", key, name);
}

static void printType(FILE *out, const AstType *type) {
    if (type == NULL) { fputs("_", out); return; }

    switch (type->kind) {
    case TYPE_NAME:
        fputs(type->name != NULL ? type->name : "_", out);
        break;
    case TYPE_GENERIC:
        fputs(type->name != NULL ? type->name : "_", out);
        fputc('[', out);
        for (int i = 0; i < type->argCount; i++) {
            if (i > 0) fputs(", ", out);
            printType(out, type->args != NULL ? type->args[i] : NULL);
        }
        fputc(']', out);
        break;
    case TYPE_OPTIONAL:
        printType(out, type->inner);
        fputc('?', out);
        break;
    case TYPE_UNION:
        for (int i = 0; i < type->argCount; i++) {
            if (i > 0) fputs(" | ", out);
            printType(out, type->args != NULL ? type->args[i] : NULL);
        }
        break;
    case TYPE_FN:
        fputs("fn(", out);
        for (int i = 0; i < type->argCount; i++) {
            if (i > 0) fputs(", ", out);
            printType(out, type->args != NULL ? type->args[i] : NULL);
        }
        fputs(") -> ", out);
        printType(out, type->inner);
        break;
    case TYPE_TUPLE:
        fputc('(', out);
        for (int i = 0; i < type->argCount; i++) {
            if (i > 0) fputs(", ", out);
            printType(out, type->args != NULL ? type->args[i] : NULL);
        }
        fputc(')', out);
        break;
    case TYPE_INFER:
        fputc('_', out);
        break;
    }
}

static void printTypeAttr(FILE *out, const char *key, const AstType *type) {
    if (type == NULL) return;
    fprintf(out, " %s=", key);
    printType(out, type);
}

/* Every child slot gets its own line; an absent optional slot prints `nil` so
 * that positional slots stay unambiguous in golden output. */
static void printChild(FILE *out, const AstNode *child, int indent) {
    fputc('\n', out);
    if (child == NULL) {
        printIndent(out, indent);
        fputs("nil", out);
    } else {
        printNode(out, child, indent);
    }
}

static void printChildren(FILE *out, AstNode **nodes, int count, int indent) {
    for (int i = 0; i < count; i++)
        printChild(out, nodes != NULL ? nodes[i] : NULL, indent);
}

static void printParams(FILE *out, const AstParam *params, int count, int indent) {
    if (params == NULL) return;
    for (int i = 0; i < count; i++) {
        const AstParam *param = &params[i];
        fputc('\n', out);
        printIndent(out, indent);
        fputs("(param", out);
        printName(out, "name", param->name);
        printTypeAttr(out, "type", param->type);
        if (param->isVariadic) fputs(" variadic", out);
        if (param->isKwRest) fputs(" kwrest", out);
        if (param->defaultValue != NULL) printChild(out, param->defaultValue, indent + 1);
        fputc(')', out);
    }
}

static void printGenerics(FILE *out, const AstGeneric *generics, int count,
                          int indent) {
    if (generics == NULL) return;
    for (int i = 0; i < count; i++) {
        fputc('\n', out);
        printIndent(out, indent);
        fputs("(generic", out);
        printName(out, "name", generics[i].name);
        printTypeAttr(out, "bound", generics[i].bound);
        fputc(')', out);
    }
}

static void printNode(FILE *out, const AstNode *node, int indent) {
    printIndent(out, indent);
    fputc('(', out);
    printKindTag(out, node->kind);

    int child = indent + 1;

    switch (node->kind) {
    case AST_INT_LIT:
        fprintf(out, " %lld", (long long)node->as.intLit);
        break;

    case AST_FLOAT_LIT:
        fputc(' ', out);
        printFloat(out, node->as.floatLit);
        break;

    case AST_STR_LIT:
        fputc(' ', out);
        printQuoted(out, node->as.strLit.chars != NULL ? node->as.strLit.chars : "",
                    node->as.strLit.chars != NULL ? node->as.strLit.length : 0);
        break;

    case AST_BOOL_LIT:
        fputs(node->as.boolLit ? " true" : " false", out);
        break;

    case AST_NULL_LIT:
    case AST_SELF:
    case AST_SUPER:
        break;

    case AST_FSTRING:
        printChildren(out, node->as.fstring.parts, node->as.fstring.partCount, child);
        break;

    case AST_IDENT:
        printName(out, "name", node->as.ident.name);
        break;

    case AST_LIST_LIT:
    case AST_SET_LIT:
    case AST_TUPLE_LIT:
        printChildren(out, node->as.sequence.items, node->as.sequence.count, child);
        break;

    case AST_DICT_LIT:
        for (int i = 0; i < node->as.dict.count; i++) {
            printChild(out, node->as.dict.keys != NULL ? node->as.dict.keys[i] : NULL, child);
            printChild(out, node->as.dict.values != NULL ? node->as.dict.values[i] : NULL, child);
        }
        break;

    case AST_UNARY:
        fprintf(out, " op=%s", jaiOpKindText(node->as.unary.op));
        printChild(out, node->as.unary.operand, child);
        break;

    case AST_BINARY:
    case AST_LOGICAL:
        fprintf(out, " op=%s", jaiOpKindText(node->as.binary.op));
        printChild(out, node->as.binary.left, child);
        printChild(out, node->as.binary.right, child);
        break;

    case AST_COMPARE_CHAIN:
        fputs(" ops=", out);
        for (int i = 0; i < node->as.chain.opCount; i++) {
            if (i > 0) fputc(',', out);
            fputs(node->as.chain.ops != NULL ? jaiOpKindText(node->as.chain.ops[i])
                                             : "?", out);
        }
        printChildren(out, node->as.chain.operands, node->as.chain.opCount + 1, child);
        break;

    case AST_TERNARY:
        printChild(out, node->as.ternary.cond, child);
        printChild(out, node->as.ternary.thenExpr, child);
        printChild(out, node->as.ternary.elseExpr, child);
        break;

    case AST_COALESCE:
        printChild(out, node->as.coalesce.left, child);
        printChild(out, node->as.coalesce.right, child);
        break;

    case AST_CALL:
        printChild(out, node->as.call.callee, child);
        for (int i = 0; i < node->as.call.argCount; i++) {
            const AstArg *arg = &node->as.call.args[i];
            fputc('\n', out);
            printIndent(out, child);
            fputs("(arg", out);
            printName(out, "name", arg->name);
            if (arg->isSpread) fputs(" spread", out);
            printChild(out, arg->value, child + 1);
            fputc(')', out);
        }
        break;

    case AST_INDEX:
        printChild(out, node->as.index.object, child);
        printChild(out, node->as.index.index, child);
        break;

    case AST_SLICE:
        printChild(out, node->as.slice.object, child);
        printChild(out, node->as.slice.start, child);
        printChild(out, node->as.slice.stop, child);
        printChild(out, node->as.slice.step, child);
        break;

    case AST_MEMBER:
    case AST_OPT_MEMBER:
        printName(out, "name", node->as.member.name);
        printChild(out, node->as.member.object, child);
        break;

    case AST_LAMBDA:
    case AST_ANON_FN:
    case AST_FN_DECL:
        printName(out, "name", node->as.fn.name);
        printTypeAttr(out, "returns", node->as.fn.returnType);
        if (node->as.fn.visibility != AST_VIS_PRIVATE)
            fprintf(out, " vis=%s", visibilityName(node->as.fn.visibility));
        if (node->as.fn.isStatic) fputs(" static", out);
        if (node->as.fn.isGenerator) fputs(" generator", out);
        if (node->as.fn.isAsync) fputs(" async", out);
        printGenerics(out, node->as.fn.generics, node->as.fn.genericCount, child);
        printParams(out, node->as.fn.params, node->as.fn.paramCount, child);
        printChild(out, node->as.fn.body, child);
        break;

    case AST_COMPREHENSION:
        fprintf(out, " kind=%s", compKindName(node->as.comp.kind));
        printChild(out, node->as.comp.element, child);
        printChild(out, node->as.comp.keyExpr, child);   /* dict comprehensions only */
        for (int i = 0; i < node->as.comp.clauseCount; i++) {
            const AstCompClause *clause = &node->as.comp.clauses[i];
            fputc('\n', out);
            printIndent(out, child);
            fputs("(clause", out);
            printChild(out, clause->pattern, child + 1);
            printChild(out, clause->iterable, child + 1);
            printChildren(out, clause->conditions, clause->conditionCount, child + 1);
            fputc(')', out);
        }
        break;

    case AST_RANGE:
        if (node->as.range.inclusive) fputs(" inclusive", out);
        printChild(out, node->as.range.start, child);
        printChild(out, node->as.range.stop, child);
        break;

    case AST_IF_EXPR:
    case AST_IF:
        printChild(out, node->as.conditional.cond, child);
        printChild(out, node->as.conditional.thenBranch, child);
        printChild(out, node->as.conditional.elseBranch, child);
        break;

    case AST_MATCH:
    case AST_MATCH_EXPR:
        printChild(out, node->as.match.subject, child);
        for (int i = 0; i < node->as.match.armCount; i++) {
            const AstMatchArm *arm = &node->as.match.arms[i];
            fputc('\n', out);
            printIndent(out, child);
            fputs("(arm", out);
            printChild(out, arm->pattern, child + 1);
            printChild(out, arm->guard, child + 1);
            printChild(out, arm->body, child + 1);
            fputc(')', out);
        }
        break;

    case AST_CAST:
        printTypeAttr(out, "target", node->as.cast.target);
        printChild(out, node->as.cast.operand, child);
        break;

    case AST_YIELD:
    case AST_AWAIT:
    case AST_THROW_EXPR:
        printChild(out, node->as.wrap.operand, child);
        break;

    case AST_BLOCK:
    case AST_PROGRAM:
        printChildren(out, node->as.block.stmts, node->as.block.count, child);
        break;

    case AST_EXPR_STMT:
        printChild(out, node->as.exprStmt.expr, child);
        break;

    case AST_VAR_DECL:
        fprintf(out, " kind=%s", varDeclKindName(node->as.varDecl.kind));
        printTypeAttr(out, "type", node->as.varDecl.declaredType);
        if (node->as.varDecl.visibility != AST_VIS_PRIVATE)
            fprintf(out, " vis=%s", visibilityName(node->as.varDecl.visibility));
        printChild(out, node->as.varDecl.pattern, child);
        printChild(out, node->as.varDecl.init, child);
        break;

    case AST_ASSIGN:
        if (node->as.assign.isCompound)
            fprintf(out, " op=%s=", jaiOpKindText(node->as.assign.op));
        printChild(out, node->as.assign.target, child);
        printChild(out, node->as.assign.value, child);
        break;

    case AST_WHILE:
    case AST_LOOP:
        printName(out, "label", node->as.loop.label);
        printChild(out, node->as.loop.cond, child);
        printChild(out, node->as.loop.body, child);
        break;

    case AST_FOR:
        printName(out, "label", node->as.forLoop.label);
        printChild(out, node->as.forLoop.pattern, child);
        printChild(out, node->as.forLoop.iterable, child);
        printChild(out, node->as.forLoop.body, child);
        break;

    case AST_BREAK:
    case AST_CONTINUE:
        printName(out, "label", node->as.jump.label);
        break;

    case AST_RETURN:
    case AST_THROW:
        printChild(out, node->as.ret.value, child);
        break;

    case AST_TRY:
        printChild(out, node->as.tryStmt.body, child);
        for (int i = 0; i < node->as.tryStmt.catchCount; i++) {
            const AstCatch *handler = &node->as.tryStmt.catches[i];
            fputc('\n', out);
            printIndent(out, child);
            fputs("(catch", out);
            printName(out, "name", handler->name);
            for (int t = 0; t < handler->typeCount; t++) {
                fputs(t == 0 ? " types=" : " | ", out);
                printType(out, handler->types != NULL ? handler->types[t] : NULL);
            }
            printChild(out, handler->body, child + 1);
            fputc(')', out);
        }
        printChild(out, node->as.tryStmt.finallyBlock, child);
        break;

    case AST_DEFER:
        printChild(out, node->as.defer.body, child);
        break;

    case AST_ASSERT:
        printChild(out, node->as.assertStmt.cond, child);
        printChild(out, node->as.assertStmt.message, child);
        break;

    case AST_CLASS_DECL:
        printName(out, "name", node->as.classDecl.name);
        printTypeAttr(out, "extends", node->as.classDecl.superclass);
        if (node->as.classDecl.visibility != AST_VIS_PRIVATE)
            fprintf(out, " vis=%s", visibilityName(node->as.classDecl.visibility));
        if (node->as.classDecl.isAbstract) fputs(" abstract", out);
        for (int i = 0; i < node->as.classDecl.traitCount; i++) {
            fputs(i == 0 ? " traits=" : ",", out);
            printType(out, node->as.classDecl.traits != NULL
                               ? node->as.classDecl.traits[i] : NULL);
        }
        printGenerics(out, node->as.classDecl.generics,
                      node->as.classDecl.genericCount, child);
        for (int i = 0; i < node->as.classDecl.fieldCount; i++) {
            const AstField *field = &node->as.classDecl.fields[i];
            fputc('\n', out);
            printIndent(out, child);
            fputs("(field", out);
            printName(out, "name", field->name);
            printTypeAttr(out, "type", field->type);
            fprintf(out, " vis=%s", visibilityName(field->visibility));
            if (field->isStatic) fputs(" static", out);
            fputs(field->isLet ? " let" : " var", out);
            if (field->defaultValue != NULL)
                printChild(out, field->defaultValue, child + 1);
            fputc(')', out);
        }
        printChildren(out, node->as.classDecl.methods,
                      node->as.classDecl.methodCount, child);
        printChildren(out, node->as.classDecl.getters,
                      node->as.classDecl.getterCount, child);
        printChildren(out, node->as.classDecl.setters,
                      node->as.classDecl.setterCount, child);
        break;

    case AST_TRAIT_DECL:
        printName(out, "name", node->as.traitDecl.name);
        if (node->as.traitDecl.visibility != AST_VIS_PRIVATE)
            fprintf(out, " vis=%s", visibilityName(node->as.traitDecl.visibility));
        for (int i = 0; i < node->as.traitDecl.superCount; i++) {
            fputs(i == 0 ? " supers=" : ",", out);
            printType(out, node->as.traitDecl.supers != NULL
                               ? node->as.traitDecl.supers[i] : NULL);
        }
        printGenerics(out, node->as.traitDecl.generics,
                      node->as.traitDecl.genericCount, child);
        printChildren(out, node->as.traitDecl.methods,
                      node->as.traitDecl.methodCount, child);
        break;

    case AST_ENUM_DECL:
        printName(out, "name", node->as.enumDecl.name);
        if (node->as.enumDecl.visibility != AST_VIS_PRIVATE)
            fprintf(out, " vis=%s", visibilityName(node->as.enumDecl.visibility));
        printGenerics(out, node->as.enumDecl.generics,
                      node->as.enumDecl.genericCount, child);
        for (int i = 0; i < node->as.enumDecl.variantCount; i++) {
            const AstVariant *variant = &node->as.enumDecl.variants[i];
            fputc('\n', out);
            printIndent(out, child);
            fputs("(variant", out);
            printName(out, "name", variant->name);
            printParams(out, variant->params, variant->paramCount, child + 1);
            fputc(')', out);
        }
        printChildren(out, node->as.enumDecl.methods,
                      node->as.enumDecl.methodCount, child);
        break;

    case AST_TYPE_DECL:
        printName(out, "name", node->as.typeDecl.name);
        printTypeAttr(out, "aliased", node->as.typeDecl.aliased);
        if (node->as.typeDecl.visibility != AST_VIS_PRIVATE)
            fprintf(out, " vis=%s", visibilityName(node->as.typeDecl.visibility));
        printGenerics(out, node->as.typeDecl.generics,
                      node->as.typeDecl.genericCount, child);
        break;

    case AST_IMPORT:
        printName(out, "path", node->as.import.path);
        printName(out, "alias", node->as.import.alias);
        break;

    case AST_FROM_IMPORT:
        printName(out, "path", node->as.fromImport.path);
        if (node->as.fromImport.isWildcard) fputs(" wildcard", out);
        for (int i = 0; i < node->as.fromImport.itemCount; i++) {
            const AstImportItem *item = &node->as.fromImport.items[i];
            fputc('\n', out);
            printIndent(out, child);
            fputs("(item", out);
            printName(out, "name", item->name);
            printName(out, "alias", item->alias);
            fputc(')', out);
        }
        break;

    case AST_EXPORT:
        for (int i = 0; i < node->as.exportDecl.count; i++) {
            fputs(i == 0 ? " names=" : ",", out);
            fputs(node->as.exportDecl.names != NULL &&
                          node->as.exportDecl.names[i] != NULL
                      ? node->as.exportDecl.names[i] : "?", out);
        }
        break;

    case AST_MODULE_DECL:
        printName(out, "name", node->as.moduleDecl.name);
        break;

    case AST_PAT_WILDCARD:
        break;

    case AST_PAT_BIND:
        printName(out, "name", node->as.patBind.name);
        printTypeAttr(out, "type", node->as.patBind.type);
        break;

    case AST_PAT_LITERAL:
        printChild(out, node->as.patLiteral.value, child);
        break;

    case AST_PAT_RANGE:
        if (node->as.patRange.inclusive) fputs(" inclusive", out);
        printChild(out, node->as.patRange.lo, child);
        printChild(out, node->as.patRange.hi, child);
        break;

    case AST_PAT_TUPLE:
    case AST_PAT_LIST:
    case AST_PAT_OR:
        if (node->as.patSeq.restIndex >= 0)
            fprintf(out, " rest=%d", node->as.patSeq.restIndex);
        printChildren(out, node->as.patSeq.elems, node->as.patSeq.count, child);
        break;

    case AST_PAT_CLASS:
    case AST_PAT_ENUM:
        printName(out, "typeName", node->as.patClass.typeName);
        printName(out, "variantName", node->as.patClass.variantName);
        for (int i = 0; i < node->as.patClass.count; i++) {
            const char *fieldName = node->as.patClass.fieldNames != NULL
                                        ? node->as.patClass.fieldNames[i] : NULL;
            if (fieldName != NULL) {
                fputc('\n', out);
                printIndent(out, child);
                fprintf(out, "(field name=%s", fieldName);
                printChild(out, node->as.patClass.subPatterns != NULL
                                    ? node->as.patClass.subPatterns[i] : NULL,
                           child + 1);
                fputc(')', out);
            } else {
                printChild(out, node->as.patClass.subPatterns != NULL
                                    ? node->as.patClass.subPatterns[i] : NULL,
                           child);
            }
        }
        break;

    default:
        JAI_PANIC("jaiAstPrint: unhandled AST kind %d", (int)node->kind);
    }

    fputc(')', out);
}

void jaiAstPrint(FILE *out, const AstNode *node, int indent) {
    if (out == NULL) return;
    if (indent < 0) indent = 0;
    if (node == NULL) {
        printIndent(out, indent);
        fputs("nil\n", out);
        return;
    }
    printNode(out, node, indent);
    fputc('\n', out);
}

/* ------------------------------------------------------------------ */
/* JSON                                                                 */
/* ------------------------------------------------------------------ */

/* Field names below mirror the C struct field names exactly so that the
 * self-hosted jaithon.ast can decode this without a translation table. Resolver
 * and code-generator scratch (symbol, resolveInfo, scope, cacheSlot, slot
 * counts) is deliberately omitted: it is not source information.
 *
 * Two payload fields are spelled differently from their C names because the
 * union member and AstNode itself both call the field `kind`, and a repeated
 * JSON key would silently overwrite the node kind: varDecl.kind is emitted as
 * "varDeclKind" and comp.kind as "compKind". */

static void jsonNode(FILE *out, const AstNode *node);

static void jsonText(FILE *out, const char *chars, size_t length) {
    fputc('"', out);
    for (size_t i = 0; i < length; i++) {
        unsigned char c = (unsigned char)chars[i];
        switch (c) {
        case '"':  fputs("\\\"", out); break;
        case '\\': fputs("\\\\", out); break;
        case '\n': fputs("\\n", out); break;
        case '\r': fputs("\\r", out); break;
        case '\t': fputs("\\t", out); break;
        case '\b': fputs("\\b", out); break;
        case '\f': fputs("\\f", out); break;
        default:
            if (c < 0x20) fprintf(out, "\\u%04x", (unsigned)c);
            else fputc((int)c, out);
        }
    }
    fputc('"', out);
}

/* A NULL C string is a real distinction in this AST (no alias, positional
 * argument, anonymous function), so it encodes as JSON null, not "". */
static void jsonStr(FILE *out, const char *s) {
    if (s == NULL) { fputs("null", out); return; }
    jsonText(out, s, strlen(s));
}

static void jsonKey(FILE *out, const char *key) {
    fputc(',', out);
    jsonText(out, key, strlen(key));
    fputc(':', out);
}

static void jsonBool(FILE *out, bool value) {
    fputs(value ? "true" : "false", out);
}

static void jsonFloat(FILE *out, double value) {
    /* JSON has no infinity or NaN; encode them as strings rather than emit a
     * document no parser will accept. */
    if (isnan(value))      { fputs("\"NaN\"", out); return; }
    if (isinf(value))      { fputs(value > 0 ? "\"Infinity\"" : "\"-Infinity\"", out); return; }
    fprintf(out, "%.17g", value);
}

static void jsonSpan(FILE *out, JaiSpan span) {
    fprintf(out, "{\"start\":%u,\"end\":%u,\"file\":%d}", (unsigned)span.start,
            (unsigned)span.end, (int)span.file);
}

static void jsonType(FILE *out, const AstType *type) {
    if (type == NULL) { fputs("null", out); return; }

    fputs("{\"kind\":", out);
    jsonStr(out, typeKindName(type->kind));
    jsonKey(out, "span");
    jsonSpan(out, type->span);
    jsonKey(out, "name");
    jsonStr(out, type->name);
    jsonKey(out, "args");
    fputc('[', out);
    for (int i = 0; i < type->argCount; i++) {
        if (i > 0) fputc(',', out);
        jsonType(out, type->args != NULL ? type->args[i] : NULL);
    }
    fputc(']', out);
    jsonKey(out, "inner");
    jsonType(out, type->inner);
    fputc('}', out);
}

static void jsonTypeArray(FILE *out, AstType **types, int count) {
    fputc('[', out);
    for (int i = 0; i < count; i++) {
        if (i > 0) fputc(',', out);
        jsonType(out, types != NULL ? types[i] : NULL);
    }
    fputc(']', out);
}

static void jsonNodeArray(FILE *out, AstNode **nodes, int count) {
    fputc('[', out);
    for (int i = 0; i < count; i++) {
        if (i > 0) fputc(',', out);
        jsonNode(out, nodes != NULL ? nodes[i] : NULL);
    }
    fputc(']', out);
}

static void jsonStrArray(FILE *out, const char **names, int count) {
    fputc('[', out);
    for (int i = 0; i < count; i++) {
        if (i > 0) fputc(',', out);
        jsonStr(out, names != NULL ? names[i] : NULL);
    }
    fputc(']', out);
}

static void jsonParams(FILE *out, const AstParam *params, int count) {
    fputc('[', out);
    for (int i = 0; i < count; i++) {
        const AstParam *param = &params[i];
        if (i > 0) fputc(',', out);
        fputs("{\"name\":", out);
        jsonStr(out, param->name);
        jsonKey(out, "type");
        jsonType(out, param->type);
        jsonKey(out, "defaultValue");
        jsonNode(out, param->defaultValue);
        jsonKey(out, "span");
        jsonSpan(out, param->span);
        jsonKey(out, "isVariadic");
        jsonBool(out, param->isVariadic);
        jsonKey(out, "isKwRest");
        jsonBool(out, param->isKwRest);
        fputc('}', out);
    }
    fputc(']', out);
}

static void jsonGenerics(FILE *out, const AstGeneric *generics, int count) {
    fputc('[', out);
    for (int i = 0; i < count; i++) {
        if (i > 0) fputc(',', out);
        fputs("{\"name\":", out);
        jsonStr(out, generics[i].name);
        jsonKey(out, "bound");
        jsonType(out, generics[i].bound);
        jsonKey(out, "span");
        jsonSpan(out, generics[i].span);
        fputc('}', out);
    }
    fputc(']', out);
}

static void jsonNode(FILE *out, const AstNode *node) {
    if (node == NULL) { fputs("null", out); return; }

    fputs("{\"kind\":", out);
    jsonStr(out, jaiAstKindName(node->kind));
    jsonKey(out, "span");
    jsonSpan(out, node->span);

    switch (node->kind) {
    case AST_INT_LIT:
        jsonKey(out, "intLit");
        fprintf(out, "%lld", (long long)node->as.intLit);
        break;

    case AST_FLOAT_LIT:
        jsonKey(out, "floatLit");
        jsonFloat(out, node->as.floatLit);
        break;

    case AST_STR_LIT:
        jsonKey(out, "chars");
        if (node->as.strLit.chars == NULL) fputs("null", out);
        else jsonText(out, node->as.strLit.chars, node->as.strLit.length);
        jsonKey(out, "length");
        fprintf(out, "%zu", node->as.strLit.length);
        break;

    case AST_BOOL_LIT:
        jsonKey(out, "boolLit");
        jsonBool(out, node->as.boolLit);
        break;

    case AST_NULL_LIT:
    case AST_SELF:
    case AST_SUPER:
        break;

    case AST_FSTRING:
        jsonKey(out, "parts");
        jsonNodeArray(out, node->as.fstring.parts, node->as.fstring.partCount);
        break;

    case AST_IDENT:
        jsonKey(out, "name");
        jsonStr(out, node->as.ident.name);
        break;

    case AST_LIST_LIT:
    case AST_SET_LIT:
    case AST_TUPLE_LIT:
        jsonKey(out, "items");
        jsonNodeArray(out, node->as.sequence.items, node->as.sequence.count);
        break;

    case AST_DICT_LIT:
        jsonKey(out, "keys");
        jsonNodeArray(out, node->as.dict.keys, node->as.dict.count);
        jsonKey(out, "values");
        jsonNodeArray(out, node->as.dict.values, node->as.dict.count);
        break;

    case AST_UNARY:
        jsonKey(out, "op");
        jsonStr(out, opEnumName(node->as.unary.op));
        jsonKey(out, "operand");
        jsonNode(out, node->as.unary.operand);
        break;

    case AST_BINARY:
    case AST_LOGICAL:
        jsonKey(out, "op");
        jsonStr(out, opEnumName(node->as.binary.op));
        jsonKey(out, "left");
        jsonNode(out, node->as.binary.left);
        jsonKey(out, "right");
        jsonNode(out, node->as.binary.right);
        break;

    case AST_COMPARE_CHAIN:
        jsonKey(out, "operands");
        jsonNodeArray(out, node->as.chain.operands, node->as.chain.opCount + 1);
        jsonKey(out, "ops");
        fputc('[', out);
        for (int i = 0; i < node->as.chain.opCount; i++) {
            if (i > 0) fputc(',', out);
            jsonStr(out, node->as.chain.ops != NULL
                             ? opEnumName(node->as.chain.ops[i]) : NULL);
        }
        fputc(']', out);
        break;

    case AST_TERNARY:
        jsonKey(out, "cond");
        jsonNode(out, node->as.ternary.cond);
        jsonKey(out, "thenExpr");
        jsonNode(out, node->as.ternary.thenExpr);
        jsonKey(out, "elseExpr");
        jsonNode(out, node->as.ternary.elseExpr);
        break;

    case AST_COALESCE:
        jsonKey(out, "left");
        jsonNode(out, node->as.coalesce.left);
        jsonKey(out, "right");
        jsonNode(out, node->as.coalesce.right);
        break;

    case AST_CALL:
        jsonKey(out, "callee");
        jsonNode(out, node->as.call.callee);
        jsonKey(out, "args");
        fputc('[', out);
        for (int i = 0; i < node->as.call.argCount; i++) {
            const AstArg *arg = &node->as.call.args[i];
            if (i > 0) fputc(',', out);
            fputs("{\"name\":", out);
            jsonStr(out, arg->name);
            jsonKey(out, "value");
            jsonNode(out, arg->value);
            jsonKey(out, "isSpread");
            jsonBool(out, arg->isSpread);
            jsonKey(out, "span");
            jsonSpan(out, arg->span);
            fputc('}', out);
        }
        fputc(']', out);
        break;

    case AST_INDEX:
        jsonKey(out, "object");
        jsonNode(out, node->as.index.object);
        jsonKey(out, "index");
        jsonNode(out, node->as.index.index);
        break;

    case AST_SLICE:
        jsonKey(out, "object");
        jsonNode(out, node->as.slice.object);
        jsonKey(out, "start");
        jsonNode(out, node->as.slice.start);
        jsonKey(out, "stop");
        jsonNode(out, node->as.slice.stop);
        jsonKey(out, "step");
        jsonNode(out, node->as.slice.step);
        break;

    case AST_MEMBER:
    case AST_OPT_MEMBER:
        jsonKey(out, "object");
        jsonNode(out, node->as.member.object);
        jsonKey(out, "name");
        jsonStr(out, node->as.member.name);
        break;

    case AST_LAMBDA:
    case AST_ANON_FN:
    case AST_FN_DECL:
        jsonKey(out, "name");
        jsonStr(out, node->as.fn.name);
        jsonKey(out, "generics");
        jsonGenerics(out, node->as.fn.generics, node->as.fn.genericCount);
        jsonKey(out, "params");
        jsonParams(out, node->as.fn.params, node->as.fn.paramCount);
        jsonKey(out, "returnType");
        jsonType(out, node->as.fn.returnType);
        jsonKey(out, "body");
        jsonNode(out, node->as.fn.body);
        jsonKey(out, "visibility");
        jsonStr(out, visibilityName(node->as.fn.visibility));
        jsonKey(out, "isStatic");
        jsonBool(out, node->as.fn.isStatic);
        jsonKey(out, "isExprBody");
        jsonBool(out, node->as.fn.isExprBody);
        jsonKey(out, "isGenerator");
        jsonBool(out, node->as.fn.isGenerator);
        jsonKey(out, "isAsync");
        jsonBool(out, node->as.fn.isAsync);
        break;

    case AST_COMPREHENSION:
        jsonKey(out, "compKind");
        jsonStr(out, compKindName(node->as.comp.kind));
        jsonKey(out, "element");
        jsonNode(out, node->as.comp.element);
        jsonKey(out, "keyExpr");
        jsonNode(out, node->as.comp.keyExpr);
        jsonKey(out, "clauses");
        fputc('[', out);
        for (int i = 0; i < node->as.comp.clauseCount; i++) {
            const AstCompClause *clause = &node->as.comp.clauses[i];
            if (i > 0) fputc(',', out);
            fputs("{\"pattern\":", out);
            jsonNode(out, clause->pattern);
            jsonKey(out, "iterable");
            jsonNode(out, clause->iterable);
            jsonKey(out, "conditions");
            jsonNodeArray(out, clause->conditions, clause->conditionCount);
            jsonKey(out, "span");
            jsonSpan(out, clause->span);
            fputc('}', out);
        }
        fputc(']', out);
        break;

    case AST_RANGE:
        jsonKey(out, "start");
        jsonNode(out, node->as.range.start);
        jsonKey(out, "stop");
        jsonNode(out, node->as.range.stop);
        jsonKey(out, "inclusive");
        jsonBool(out, node->as.range.inclusive);
        break;

    case AST_IF_EXPR:
    case AST_IF:
        jsonKey(out, "cond");
        jsonNode(out, node->as.conditional.cond);
        jsonKey(out, "thenBranch");
        jsonNode(out, node->as.conditional.thenBranch);
        jsonKey(out, "elseBranch");
        jsonNode(out, node->as.conditional.elseBranch);
        break;

    case AST_MATCH:
    case AST_MATCH_EXPR:
        jsonKey(out, "subject");
        jsonNode(out, node->as.match.subject);
        jsonKey(out, "arms");
        fputc('[', out);
        for (int i = 0; i < node->as.match.armCount; i++) {
            const AstMatchArm *arm = &node->as.match.arms[i];
            if (i > 0) fputc(',', out);
            fputs("{\"pattern\":", out);
            jsonNode(out, arm->pattern);
            jsonKey(out, "guard");
            jsonNode(out, arm->guard);
            jsonKey(out, "body");
            jsonNode(out, arm->body);
            jsonKey(out, "span");
            jsonSpan(out, arm->span);
            fputc('}', out);
        }
        fputc(']', out);
        break;

    case AST_CAST:
        jsonKey(out, "operand");
        jsonNode(out, node->as.cast.operand);
        jsonKey(out, "target");
        jsonType(out, node->as.cast.target);
        break;

    case AST_YIELD:
    case AST_AWAIT:
    case AST_THROW_EXPR:
        jsonKey(out, "operand");
        jsonNode(out, node->as.wrap.operand);
        break;

    case AST_BLOCK:
    case AST_PROGRAM:
        jsonKey(out, "stmts");
        jsonNodeArray(out, node->as.block.stmts, node->as.block.count);
        break;

    case AST_EXPR_STMT:
        jsonKey(out, "expr");
        jsonNode(out, node->as.exprStmt.expr);
        break;

    case AST_VAR_DECL:
        jsonKey(out, "varDeclKind");
        jsonStr(out, varDeclKindName(node->as.varDecl.kind));
        jsonKey(out, "pattern");
        jsonNode(out, node->as.varDecl.pattern);
        jsonKey(out, "declaredType");
        jsonType(out, node->as.varDecl.declaredType);
        jsonKey(out, "init");
        jsonNode(out, node->as.varDecl.init);
        jsonKey(out, "visibility");
        jsonStr(out, visibilityName(node->as.varDecl.visibility));
        break;

    case AST_ASSIGN:
        jsonKey(out, "target");
        jsonNode(out, node->as.assign.target);
        jsonKey(out, "value");
        jsonNode(out, node->as.assign.value);
        jsonKey(out, "op");
        jsonStr(out, opEnumName(node->as.assign.op));
        jsonKey(out, "isCompound");
        jsonBool(out, node->as.assign.isCompound);
        break;

    case AST_WHILE:
    case AST_LOOP:
        jsonKey(out, "cond");
        jsonNode(out, node->as.loop.cond);
        jsonKey(out, "body");
        jsonNode(out, node->as.loop.body);
        jsonKey(out, "label");
        jsonStr(out, node->as.loop.label);
        break;

    case AST_FOR:
        jsonKey(out, "pattern");
        jsonNode(out, node->as.forLoop.pattern);
        jsonKey(out, "iterable");
        jsonNode(out, node->as.forLoop.iterable);
        jsonKey(out, "body");
        jsonNode(out, node->as.forLoop.body);
        jsonKey(out, "label");
        jsonStr(out, node->as.forLoop.label);
        break;

    case AST_BREAK:
    case AST_CONTINUE:
        jsonKey(out, "label");
        jsonStr(out, node->as.jump.label);
        break;

    case AST_RETURN:
    case AST_THROW:
        jsonKey(out, "value");
        jsonNode(out, node->as.ret.value);
        break;

    case AST_TRY:
        jsonKey(out, "body");
        jsonNode(out, node->as.tryStmt.body);
        jsonKey(out, "catches");
        fputc('[', out);
        for (int i = 0; i < node->as.tryStmt.catchCount; i++) {
            const AstCatch *handler = &node->as.tryStmt.catches[i];
            if (i > 0) fputc(',', out);
            fputs("{\"name\":", out);
            jsonStr(out, handler->name);
            jsonKey(out, "types");
            jsonTypeArray(out, handler->types, handler->typeCount);
            jsonKey(out, "body");
            jsonNode(out, handler->body);
            jsonKey(out, "span");
            jsonSpan(out, handler->span);
            fputc('}', out);
        }
        fputc(']', out);
        jsonKey(out, "finallyBlock");
        jsonNode(out, node->as.tryStmt.finallyBlock);
        break;

    case AST_DEFER:
        jsonKey(out, "body");
        jsonNode(out, node->as.defer.body);
        break;

    case AST_ASSERT:
        jsonKey(out, "cond");
        jsonNode(out, node->as.assertStmt.cond);
        jsonKey(out, "message");
        jsonNode(out, node->as.assertStmt.message);
        break;

    case AST_CLASS_DECL:
        jsonKey(out, "name");
        jsonStr(out, node->as.classDecl.name);
        jsonKey(out, "generics");
        jsonGenerics(out, node->as.classDecl.generics,
                     node->as.classDecl.genericCount);
        jsonKey(out, "superclass");
        jsonType(out, node->as.classDecl.superclass);
        jsonKey(out, "traits");
        jsonTypeArray(out, node->as.classDecl.traits, node->as.classDecl.traitCount);
        jsonKey(out, "fields");
        fputc('[', out);
        for (int i = 0; i < node->as.classDecl.fieldCount; i++) {
            const AstField *field = &node->as.classDecl.fields[i];
            if (i > 0) fputc(',', out);
            fputs("{\"name\":", out);
            jsonStr(out, field->name);
            jsonKey(out, "type");
            jsonType(out, field->type);
            jsonKey(out, "defaultValue");
            jsonNode(out, field->defaultValue);
            jsonKey(out, "visibility");
            jsonStr(out, visibilityName(field->visibility));
            jsonKey(out, "isStatic");
            jsonBool(out, field->isStatic);
            jsonKey(out, "isLet");
            jsonBool(out, field->isLet);
            jsonKey(out, "span");
            jsonSpan(out, field->span);
            fputc('}', out);
        }
        fputc(']', out);
        jsonKey(out, "methods");
        jsonNodeArray(out, node->as.classDecl.methods, node->as.classDecl.methodCount);
        jsonKey(out, "getters");
        jsonNodeArray(out, node->as.classDecl.getters, node->as.classDecl.getterCount);
        jsonKey(out, "setters");
        jsonNodeArray(out, node->as.classDecl.setters, node->as.classDecl.setterCount);
        jsonKey(out, "visibility");
        jsonStr(out, visibilityName(node->as.classDecl.visibility));
        jsonKey(out, "isAbstract");
        jsonBool(out, node->as.classDecl.isAbstract);
        break;

    case AST_TRAIT_DECL:
        jsonKey(out, "name");
        jsonStr(out, node->as.traitDecl.name);
        jsonKey(out, "generics");
        jsonGenerics(out, node->as.traitDecl.generics,
                     node->as.traitDecl.genericCount);
        jsonKey(out, "supers");
        jsonTypeArray(out, node->as.traitDecl.supers, node->as.traitDecl.superCount);
        jsonKey(out, "methods");
        jsonNodeArray(out, node->as.traitDecl.methods, node->as.traitDecl.methodCount);
        jsonKey(out, "visibility");
        jsonStr(out, visibilityName(node->as.traitDecl.visibility));
        break;

    case AST_ENUM_DECL:
        jsonKey(out, "name");
        jsonStr(out, node->as.enumDecl.name);
        jsonKey(out, "generics");
        jsonGenerics(out, node->as.enumDecl.generics, node->as.enumDecl.genericCount);
        jsonKey(out, "variants");
        fputc('[', out);
        for (int i = 0; i < node->as.enumDecl.variantCount; i++) {
            const AstVariant *variant = &node->as.enumDecl.variants[i];
            if (i > 0) fputc(',', out);
            fputs("{\"name\":", out);
            jsonStr(out, variant->name);
            jsonKey(out, "params");
            jsonParams(out, variant->params, variant->paramCount);
            jsonKey(out, "span");
            jsonSpan(out, variant->span);
            fputc('}', out);
        }
        fputc(']', out);
        jsonKey(out, "methods");
        jsonNodeArray(out, node->as.enumDecl.methods, node->as.enumDecl.methodCount);
        jsonKey(out, "visibility");
        jsonStr(out, visibilityName(node->as.enumDecl.visibility));
        break;

    case AST_TYPE_DECL:
        jsonKey(out, "name");
        jsonStr(out, node->as.typeDecl.name);
        jsonKey(out, "generics");
        jsonGenerics(out, node->as.typeDecl.generics, node->as.typeDecl.genericCount);
        jsonKey(out, "aliased");
        jsonType(out, node->as.typeDecl.aliased);
        jsonKey(out, "visibility");
        jsonStr(out, visibilityName(node->as.typeDecl.visibility));
        break;

    case AST_IMPORT:
        jsonKey(out, "path");
        jsonStr(out, node->as.import.path);
        jsonKey(out, "alias");
        jsonStr(out, node->as.import.alias);
        break;

    case AST_FROM_IMPORT:
        jsonKey(out, "path");
        jsonStr(out, node->as.fromImport.path);
        jsonKey(out, "items");
        fputc('[', out);
        for (int i = 0; i < node->as.fromImport.itemCount; i++) {
            const AstImportItem *item = &node->as.fromImport.items[i];
            if (i > 0) fputc(',', out);
            fputs("{\"name\":", out);
            jsonStr(out, item->name);
            jsonKey(out, "alias");
            jsonStr(out, item->alias);
            jsonKey(out, "span");
            jsonSpan(out, item->span);
            fputc('}', out);
        }
        fputc(']', out);
        jsonKey(out, "isWildcard");
        jsonBool(out, node->as.fromImport.isWildcard);
        break;

    case AST_EXPORT:
        jsonKey(out, "names");
        jsonStrArray(out, node->as.exportDecl.names, node->as.exportDecl.count);
        break;

    case AST_MODULE_DECL:
        jsonKey(out, "name");
        jsonStr(out, node->as.moduleDecl.name);
        break;

    case AST_PAT_WILDCARD:
        break;

    case AST_PAT_BIND:
        jsonKey(out, "name");
        jsonStr(out, node->as.patBind.name);
        jsonKey(out, "type");
        jsonType(out, node->as.patBind.type);
        break;

    case AST_PAT_LITERAL:
        jsonKey(out, "value");
        jsonNode(out, node->as.patLiteral.value);
        break;

    case AST_PAT_RANGE:
        jsonKey(out, "lo");
        jsonNode(out, node->as.patRange.lo);
        jsonKey(out, "hi");
        jsonNode(out, node->as.patRange.hi);
        jsonKey(out, "inclusive");
        jsonBool(out, node->as.patRange.inclusive);
        break;

    case AST_PAT_TUPLE:
    case AST_PAT_LIST:
    case AST_PAT_OR:
        jsonKey(out, "elems");
        jsonNodeArray(out, node->as.patSeq.elems, node->as.patSeq.count);
        jsonKey(out, "restIndex");
        fprintf(out, "%d", node->as.patSeq.restIndex);
        break;

    case AST_PAT_CLASS:
    case AST_PAT_ENUM:
        jsonKey(out, "typeName");
        jsonStr(out, node->as.patClass.typeName);
        jsonKey(out, "variantName");
        jsonStr(out, node->as.patClass.variantName);
        jsonKey(out, "fieldNames");
        jsonStrArray(out, node->as.patClass.fieldNames, node->as.patClass.count);
        jsonKey(out, "subPatterns");
        jsonNodeArray(out, node->as.patClass.subPatterns, node->as.patClass.count);
        break;

    default:
        JAI_PANIC("jaiAstToJson: unhandled AST kind %d", (int)node->kind);
    }

    fputc('}', out);
}

void jaiAstToJson(FILE *out, const AstNode *node) {
    if (out == NULL) return;
    jsonNode(out, node);
    fputc('\n', out);
}
