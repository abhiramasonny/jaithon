/* check.h — the gradual type checker and constant folder.
 *
 * Runs after the resolver. Annotates every expression node with a JaiType,
 * reports E04xx/E05xx/E06xx/E07xx diagnostics, inserts AST_CAST nodes where a
 * runtime guard is needed at an any->T boundary, and folds constant
 * subexpressions in place.
 */
#ifndef JAI_CHECK_H
#define JAI_CHECK_H

#include "modsig.h"
#include "resolve.h"
#include "types.h"

typedef struct {
    Resolver *resolver;
    AstContext *ast;
    JaiType  *currentReturnType;
    struct TypeDecl *currentClass;
    int       errorCount;
    bool      strict;        /* --strict: unannotated params are an error */
    bool      foldConstants;
} Checker;

void jaiCheckerInit(Checker *c, Resolver *r, AstContext *ast);
void jaiCheckerFree(Checker *c);

/* Type-check a resolved program. Returns false if any error was reported. */
bool jaiCheckProgram(Checker *c, AstNode *program);

/* Check a single expression (REPL, const contexts). */
JaiType *jaiCheckExpr(Checker *c, AstNode *expr);

/* ------------------------------------------------------------------ */
/* Constant folding                                                     */
/* ------------------------------------------------------------------ */

typedef enum { CONST_NONE, CONST_INT, CONST_FLOAT, CONST_BOOL, CONST_STR, CONST_NULL } ConstKind;

typedef struct {
    ConstKind kind;
    union {
        int64_t     i;
        double      f;
        bool        b;
        struct { const char *chars; size_t length; } s;
    } as;
} ConstValue;

/* Attempt to evaluate `node` at compile time. Returns CONST_NONE when the
 * expression is not constant. Reports E0900/E0901 on overflow or /0. */
ConstValue jaiConstEval(Checker *c, AstNode *node);
/* Replace `node` in place with a literal node for `v`. */
void       jaiConstReplace(AstContext *ast, AstNode *node, ConstValue v);

/* ------------------------------------------------------------------ */
/* Class metadata built during checking                                 */
/* ------------------------------------------------------------------ */

/* How one `: Trait[...]` clause binds that trait's generic parameters.
 * Opaque here; defined in check.c, where substitution happens. */
typedef struct TraitBinding TraitBinding;

struct TypeDecl {
    const char   *name;
    JaiSpan       span;
    AstNode      *decl;              /* AST_CLASS_DECL / TRAIT / ENUM */

    /* NULL for a declaration written in the file being compiled. Otherwise the
     * module whose source it was read out of: its body is never checked here,
     * its annotations are resolved in that module's namespace rather than this
     * one's, and every diagnostic its layout produces is dropped, because it
     * belongs to that module's own compilation. */
    ModuleSig    *origin;
    struct TypeDecl *superclass;
    struct TypeDecl **traits;
    /* Parallel to `traits`: the arguments each was written with, so a
     * requirement can be substituted before it is compared against the
     * implementation. Empty for a non-generic trait. */
    TraitBinding *traitBindings;
    int           traitCount;

    /* Traits named in the header whose declaration this compilation unit
     * cannot see — they arrived through the prelude or an import as a runtime
     * object. Only the name is known, so `C` implements one of these exactly
     * when it named it; the requirements themselves are enforced at run time. */
    const char  **opaqueTraits;
    int           opaqueTraitCount;

    /* The superclass named in the header when its declaration is not visible
     * here either — same story as `opaqueTraits`, one parent instead of many.
     * Without it a subclass of an imported class looked like a class with no
     * parent, and every inherited member was reported as E0410. */
    const char   *opaqueSuper;

    /* Fields, parents first, with slots assigned. */
    struct {
        const char   *name;
        JaiType      *type;
        AstVisibility visibility;
        int           slot;
        bool          isStatic;
        bool          isLet;
        bool          hasDefault;
    } *fields;
    int fieldCount;

    struct {
        const char   *name;
        JaiType      *type;          /* TY_FN */
        AstVisibility visibility;
        bool          isStatic;
        bool          isGetter;
        bool          isSetter;
        bool          isAbstract;
        AstNode      *decl;
    } *members;
    int memberCount;

    bool isTrait;
    bool isEnum;
    bool isAbstract;
    int  variantCount;               /* enums */
};

TypeDecl *jaiTypeDeclFind(const char *name);
/* The name a class, trait or enum type answers to *in the module being
 * compiled* — which is what the VM resolves a type operand against. It is not
 * always `JaiType.name`: an imported declaration's type is interned under its
 * qualified name (`std.time.Duration`) so that two modules declaring a `Node`
 * are two types, while the binding the run time will see is the bare name, or
 * the alias it was imported under. NULL when `t` is not a declared type. */
const char *jaiTypeDeclBindingName(const JaiType *t);
const void *jaiTypeDeclFindField(const TypeDecl *d, const char *name);
const void *jaiTypeDeclFindMember(const TypeDecl *d, const char *name);
bool        jaiTypeDeclIsSubclassOf(const TypeDecl *sub, const TypeDecl *super);

#endif /* JAI_CHECK_H */
