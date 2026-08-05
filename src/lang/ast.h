/* ast.h — the Jaithon abstract syntax tree.
 *
 * Every node is arena-allocated and carries a source span. The AST is the sole
 * interface between the parser and everything downstream (resolver, type
 * checker, code generator, formatter, documentation generator, and the
 * self-hosted front end, which mirrors this shape in jaithon.ast).
 */
#ifndef JAI_AST_H
#define JAI_AST_H

#include "../common/common.h"
#include "../common/diag.h"
#include "token.h"

typedef struct AstNode  AstNode;
typedef struct AstType  AstType;
typedef struct Symbol   Symbol;   /* filled in by the resolver */
typedef struct JaiType  JaiType;  /* filled in by the type checker */

typedef enum {
    /* --- expressions --- */
    AST_INT_LIT, AST_FLOAT_LIT, AST_STR_LIT, AST_BOOL_LIT, AST_NULL_LIT,
    AST_FSTRING,          /* interpolated string: alternating parts */
    AST_IDENT,
    AST_SELF, AST_SUPER,
    AST_LIST_LIT, AST_DICT_LIT, AST_SET_LIT, AST_TUPLE_LIT,
    AST_UNARY, AST_BINARY, AST_LOGICAL, AST_COMPARE_CHAIN,
    AST_TERNARY, AST_COALESCE,
    AST_CALL, AST_INDEX, AST_SLICE, AST_MEMBER, AST_OPT_MEMBER,
    AST_LAMBDA, AST_ANON_FN,
    AST_COMPREHENSION,
    AST_RANGE,
    AST_IF_EXPR, AST_MATCH_EXPR,
    AST_CAST,             /* inserted by the type checker for any->T guards */
    AST_YIELD, AST_AWAIT,
    AST_THROW_EXPR,       /* `throw e` in expression position; type `never` */

    /* --- statements --- */
    AST_BLOCK,
    AST_EXPR_STMT,
    AST_VAR_DECL,
    AST_ASSIGN,
    AST_IF, AST_WHILE, AST_LOOP, AST_FOR, AST_MATCH,
    AST_BREAK, AST_CONTINUE, AST_RETURN, AST_THROW,
    AST_TRY, AST_DEFER, AST_ASSERT,

    /* --- declarations --- */
    AST_FN_DECL, AST_CLASS_DECL, AST_TRAIT_DECL, AST_ENUM_DECL,
    AST_TYPE_DECL, AST_IMPORT, AST_FROM_IMPORT, AST_EXPORT, AST_MODULE_DECL,

    /* --- patterns --- */
    AST_PAT_WILDCARD, AST_PAT_BIND, AST_PAT_LITERAL, AST_PAT_RANGE,
    AST_PAT_TUPLE, AST_PAT_LIST, AST_PAT_CLASS, AST_PAT_ENUM, AST_PAT_OR,

    AST_PROGRAM,
    AST_KIND_COUNT
} AstKind;

/* Binary/unary operator codes, shared by the checker and code generator. */
typedef enum {
    OPK_ADD, OPK_SUB, OPK_MUL, OPK_DIV, OPK_FLOORDIV, OPK_MOD, OPK_POW,
    OPK_ADD_WRAP, OPK_SUB_WRAP, OPK_MUL_WRAP,
    OPK_BAND, OPK_BOR, OPK_BXOR, OPK_SHL, OPK_SHR, OPK_MATMUL,
    OPK_EQ, OPK_NE, OPK_LT, OPK_LE, OPK_GT, OPK_GE,
    OPK_IS, OPK_IS_NOT, OPK_IN, OPK_NOT_IN,
    OPK_AND, OPK_OR,
    OPK_NEG, OPK_POS, OPK_NOT, OPK_BNOT,
    OPK_COUNT
} OpKind;

const char *jaiOpKindText(OpKind op);      /* "+", "and", ... */
bool        jaiOpKindIsComparison(OpKind op);

typedef enum { VD_LET, VD_VAR, VD_CONST } VarDeclKind;
typedef enum { AST_VIS_PRIVATE, AST_VIS_PROTECTED, AST_VIS_PUBLIC } AstVisibility;
typedef enum { COMP_LIST, COMP_DICT, COMP_SET, COMP_GENERATOR } CompKind;

/* ------------------------------------------------------------------ */
/* Type expressions (syntax, not resolved semantics)                    */
/* ------------------------------------------------------------------ */

typedef enum {
    TYPE_NAME,       /* int, str, Account, T                    */
    TYPE_GENERIC,    /* list[int], dict[str, V]                 */
    TYPE_OPTIONAL,   /* T?                                      */
    TYPE_UNION,      /* A | B                                   */
    TYPE_FN,         /* fn(A, B) -> C                           */
    TYPE_TUPLE,      /* (A, B)                                  */
    TYPE_INFER,      /* absent annotation                       */
} AstTypeKind;

struct AstType {
    AstTypeKind kind;
    JaiSpan     span;
    const char *name;          /* TYPE_NAME: interned identifier text */
    AstType   **args;          /* GENERIC/UNION/FN params/TUPLE members */
    int         argCount;
    AstType    *inner;         /* OPTIONAL: T; FN: return type */
    JaiType    *resolved;      /* filled by the type checker */
};

/* ------------------------------------------------------------------ */
/* Sub-structures                                                       */
/* ------------------------------------------------------------------ */

typedef struct {
    const char *name;
    AstType    *type;
    AstNode    *defaultValue;   /* NULL if none */
    JaiSpan     span;
    bool        isVariadic;     /* ...xs */
    bool        isKwRest;       /* **opts */
    Symbol     *symbol;         /* resolver output */
} AstParam;

typedef struct {
    const char *name;           /* NULL for positional */
    AstNode    *value;
    bool        isSpread;       /* ...expr */
    JaiSpan     span;
} AstArg;

typedef struct {
    AstNode *pattern;           /* AST_PAT_* */
    AstNode *guard;             /* may be NULL */
    AstNode *body;              /* expression or AST_BLOCK */
    JaiSpan  span;
} AstMatchArm;

typedef struct {
    AstNode  *pattern;          /* binding pattern for the loop variable */
    AstNode  *iterable;
    AstNode **conditions;       /* `if` filters */
    int       conditionCount;
    JaiSpan   span;
} AstCompClause;

typedef struct {
    const char *name;           /* exception binding name, may be NULL */
    AstType   **types;          /* caught types; empty = catch-all */
    int         typeCount;
    AstNode    *body;
    Symbol     *symbol;
    JaiSpan     span;
} AstCatch;

typedef struct {
    const char   *name;
    AstType      *type;
    AstNode      *defaultValue;
    AstVisibility visibility;
    bool          isStatic;
    bool          isLet;
    JaiSpan       span;
} AstField;

typedef struct {
    const char *name;
    AstParam   *params;         /* payload fields; empty for unit variants */
    int         paramCount;
    JaiSpan     span;
} AstVariant;

typedef struct {
    const char *name;
    AstType    *bound;          /* trait bound, may be NULL */
    JaiSpan     span;
} AstGeneric;

typedef struct {
    const char *name;           /* imported name */
    const char *alias;          /* may be NULL */
    JaiSpan     span;
    Symbol     *symbol;         /* resolver output: the binding this item makes */
} AstImportItem;

/* ------------------------------------------------------------------ */
/* The node                                                             */
/* ------------------------------------------------------------------ */

struct AstNode {
    AstKind  kind;
    JaiSpan  span;
    JaiType *type;          /* inferred/checked type; NULL before typecheck */

    union {
        /* literals */
        int64_t     intLit;
        double      floatLit;
        struct { const char *chars; size_t length; } strLit;
        bool        boolLit;

        /* AST_FSTRING: alternating literal/expression parts. */
        struct { AstNode **parts; int partCount; } fstring;

        /* AST_IDENT */
        struct { const char *name; Symbol *symbol; } ident;

        /* AST_LIST_LIT / AST_SET_LIT / AST_TUPLE_LIT */
        struct { AstNode **items; int count; } sequence;

        /* AST_DICT_LIT */
        struct { AstNode **keys; AstNode **values; int count; } dict;

        /* AST_UNARY */
        struct { OpKind op; AstNode *operand; } unary;

        /* AST_BINARY / AST_LOGICAL */
        struct { OpKind op; AstNode *left, *right; } binary;

        /* AST_COMPARE_CHAIN: a < b <= c */
        struct { AstNode **operands; OpKind *ops; int opCount; } chain;

        /* AST_TERNARY */
        struct { AstNode *cond, *thenExpr, *elseExpr; } ternary;

        /* AST_COALESCE */
        struct { AstNode *left, *right; } coalesce;

        /* AST_CALL */
        struct { AstNode *callee; AstArg *args; int argCount; } call;

        /* AST_INDEX. `typeArgs` is set by the resolver when the brackets held
         * type arguments (`Box[int]`) rather than a subscript; generics are
         * erased (spec §6.1), so nothing is emitted for them. */
        struct { AstNode *object, *index; bool typeArgs; } index;

        /* AST_SLICE */
        struct { AstNode *object, *start, *stop, *step; } slice;

        /* AST_MEMBER / AST_OPT_MEMBER */
        struct { AstNode *object; const char *name; int cacheSlot; } member;

        /* AST_LAMBDA / AST_ANON_FN / AST_FN_DECL */
        struct {
            const char   *name;             /* NULL for anonymous */
            AstGeneric   *generics;
            int           genericCount;
            AstParam     *params;
            int           paramCount;
            AstType      *returnType;
            AstNode      *body;             /* AST_BLOCK, or expression for lambdas */
            AstVisibility visibility;
            bool          isStatic;
            bool          isExprBody;
            bool          isGenerator;      /* contains yield */
            bool          isAsync;
            Symbol       *symbol;
            /* Resolver output: */
            int           localCount;
            int           upvalueCount;
            void         *resolveInfo;      /* opaque FunctionScope* */
        } fn;

        /* AST_COMPREHENSION */
        struct {
            CompKind       kind;
            AstNode       *element;         /* value expression */
            AstNode       *keyExpr;         /* dict comprehensions only */
            AstCompClause *clauses;
            int            clauseCount;
        } comp;

        /* AST_RANGE */
        struct { AstNode *start, *stop; bool inclusive; } range;

        /* AST_IF_EXPR / AST_IF */
        struct { AstNode *cond, *thenBranch, *elseBranch; } conditional;

        /* AST_MATCH / AST_MATCH_EXPR */
        struct { AstNode *subject; AstMatchArm *arms; int armCount; } match;

        /* AST_CAST. `widen` marks §2.2's int-to-float conversion, which always
         * succeeds; without it the node is an any->T guard, which is a runtime
         * test that can fail. Both are inserted by the checker — no syntax
         * produces an AST_CAST — so the flag is never parsed. */
        struct { AstNode *operand; AstType *target; bool widen; } cast;

        /* AST_YIELD / AST_AWAIT */
        struct { AstNode *operand; } wrap;

        /* AST_BLOCK / AST_PROGRAM.
         *
         * `captureBase` is the block's half of the per-scope freshness rule
         * that `loop` below describes: the lowest slot *declared by this very
         * block* that a closure captures by reference, or -1. A block's
         * bindings die when it exits and the next sibling block reuses their
         * slots, so an escaped closure that still aliased one would read the
         * unrelated binding that landed there. Only the declaring block
         * closes: an inner block exiting leaves an outer block's bindings
         * alive. AST_PROGRAM never sets it — a module body outlives nothing. */
        struct { AstNode **stmts; int count; void *scope; int captureBase; } block;

        /* AST_EXPR_STMT */
        struct { AstNode *expr; } exprStmt;

        /* AST_VAR_DECL */
        struct {
            VarDeclKind   kind;
            AstNode      *pattern;          /* AST_PAT_BIND or a destructuring pattern */
            AstType      *declaredType;
            AstNode      *init;             /* may be NULL */
            AstVisibility visibility;
            Symbol       *symbol;           /* for the simple single-name case */
        } varDecl;

        /* AST_ASSIGN */
        struct { AstNode *target, *value; OpKind op; bool isCompound; } assign;

        /* AST_WHILE / AST_LOOP.
         *
         * `captureBase` is the resolver's answer to "does an iteration of this
         * loop need its bindings closed?": the lowest slot of a binding
         * declared inside the loop that some closure captures *by reference*,
         * or -1 when there is none. Codegen closes at that slot on every path
         * that ends an iteration, so the next one gets fresh cells (spec §5.2).
         * A `let` capture is by value (spec §6) and needs no close, which is
         * why the common `for i in ... { || i }` costs nothing. */
        struct {
            AstNode    *cond;
            AstNode    *body;
            const char *label;
            int         captureBase;
        } loop;

        /* AST_FOR */
        struct {
            AstNode    *pattern;
            AstNode    *iterable;
            AstNode    *body;
            const char *label;
            Symbol     *iterSymbol;
            int         captureBase;   /* see `loop` above */
        } forLoop;

        /* AST_BREAK / AST_CONTINUE */
        struct { const char *label; } jump;

        /* AST_RETURN / AST_THROW */
        struct { AstNode *value; } ret;

        /* AST_TRY */
        struct {
            AstNode  *body;
            AstCatch *catches;
            int       catchCount;
            AstNode  *finallyBlock;
        } tryStmt;

        /* AST_DEFER */
        struct { AstNode *body; } defer;

        /* AST_ASSERT */
        struct { AstNode *cond, *message; } assertStmt;

        /* AST_CLASS_DECL */
        struct {
            const char   *name;
            AstGeneric   *generics;
            int           genericCount;
            AstType      *superclass;       /* may be NULL */
            AstType     **traits;
            int           traitCount;
            AstField     *fields;
            int           fieldCount;
            AstNode     **methods;          /* AST_FN_DECL */
            int           methodCount;
            AstNode     **getters;
            int           getterCount;
            AstNode     **setters;
            int           setterCount;
            AstVisibility visibility;
            bool          isAbstract;
            Symbol       *symbol;
        } classDecl;

        /* AST_TRAIT_DECL */
        struct {
            const char   *name;
            AstGeneric   *generics;
            int           genericCount;
            AstType     **supers;
            int           superCount;
            AstNode     **methods;          /* AST_FN_DECL; body may be NULL */
            int           methodCount;
            AstVisibility visibility;
            Symbol       *symbol;
        } traitDecl;

        /* AST_ENUM_DECL */
        struct {
            const char   *name;
            AstGeneric   *generics;
            int           genericCount;
            AstVariant   *variants;
            int           variantCount;
            AstNode     **methods;
            int           methodCount;
            AstVisibility visibility;
            Symbol       *symbol;
        } enumDecl;

        /* AST_TYPE_DECL */
        struct {
            const char   *name;
            AstGeneric   *generics;
            int           genericCount;
            AstType      *aliased;
            AstVisibility visibility;
            Symbol       *symbol;
        } typeDecl;

        /* AST_IMPORT */
        struct { const char *path; const char *alias; Symbol *symbol; } import;

        /* AST_FROM_IMPORT */
        struct {
            const char    *path;
            AstImportItem *items;
            int            itemCount;
            bool           isWildcard;
        } fromImport;

        /* AST_EXPORT */
        struct { const char **names; int count; } exportDecl;

        /* AST_MODULE_DECL */
        struct { const char *name; } moduleDecl;

        /* --- patterns --- */
        /* AST_PAT_BIND */
        struct { const char *name; AstType *type; Symbol *symbol; } patBind;
        /* AST_PAT_LITERAL */
        struct { AstNode *value; } patLiteral;
        /* AST_PAT_RANGE */
        struct { AstNode *lo, *hi; bool inclusive; } patRange;
        /* AST_PAT_TUPLE / AST_PAT_LIST / AST_PAT_OR */
        struct { AstNode **elems; int count; int restIndex; } patSeq;
        /* AST_PAT_CLASS / AST_PAT_ENUM */
        struct {
            const char  *typeName;
            const char  *variantName;    /* enums only */
            const char **fieldNames;     /* NULL entries = positional */
            AstNode    **subPatterns;
            int          count;
        } patClass;
    } as;
};

/* ------------------------------------------------------------------ */
/* Construction                                                         */
/* ------------------------------------------------------------------ */

typedef struct {
    JaiArena arena;
    int      nodeCount;
} AstContext;

void     jaiAstContextInit(AstContext *ctx);
void     jaiAstContextFree(AstContext *ctx);
AstNode *jaiAstNew(AstContext *ctx, AstKind kind, JaiSpan span);
AstType *jaiAstTypeNew(AstContext *ctx, AstTypeKind kind, JaiSpan span);
AstNode **jaiAstNodeArray(AstContext *ctx, int count);

/* ------------------------------------------------------------------ */
/* Traversal and output                                                 */
/* ------------------------------------------------------------------ */

/* Visit every child of `node`. Return false from `fn` to stop descending into
 * that subtree. */
typedef bool (*AstVisitor)(AstNode *node, void *userData);
void jaiAstWalk(AstNode *node, AstVisitor pre, AstVisitor post, void *userData);

const char *jaiAstKindName(AstKind kind);
bool        jaiAstIsExpression(AstKind kind);
bool        jaiAstIsStatement(AstKind kind);
bool        jaiAstIsPattern(AstKind kind);
/* Can this expression appear on the left of `=`? */
bool        jaiAstIsAssignable(const AstNode *node);
/* Does this statement always transfer control (return/throw/break/continue)? */
bool        jaiAstAlwaysDiverges(const AstNode *node);

/* Pretty-printed S-expression form, for --emit=ast and golden tests. */
void jaiAstPrint(FILE *out, const AstNode *node, int indent);
/* JSON form consumed by jaithon.ast and external tools. */
void jaiAstToJson(FILE *out, const AstNode *node);

#endif /* JAI_AST_H */
