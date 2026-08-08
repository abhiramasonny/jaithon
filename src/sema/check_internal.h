/* check_internal.h — the interface between the checker's five source files.
 *
 * The checker is a tree walk over a recursive grammar, so its parts call each
 * other the way the grammar nests: a statement checks expressions, an
 * expression checks a block, a class body checks functions. That coupling
 * cannot be made one-way. What it can be is *named*, and this header is the
 * whole of it — everything else in each file is static to it.
 *
 *   check.c        per-run state, the declaration registry, the type-name
 *                  scope stack, narrowing, guard insertion, and the entry
 *                  points jaiCheckProgram and jaiCheckExpr
 *   check_fold.c   the constant folder and evaluator (spec §2.2, §8)
 *   check_expr.c   expressions, member access, calls, comprehensions
 *   check_stmt.c   patterns, match, bindings, assignment, statements, functions
 *   check_decl.c   class/trait/enum layout, trait conformance, class bodies
 *
 * Internal names are prefixed `jaiChk` so that they cannot be confused with
 * the public `jaiCheck*` surface in check.h, which is what the rest of the
 * tree calls.
 */
#ifndef JAI_CHECK_INTERNAL_H
#define JAI_CHECK_INTERNAL_H

#include "check.h"
#include "semadump.h"

typedef struct {
    TypeDecl *decl;
    JaiType  *type;
    uint8_t   status;      /* 0 = registered, 1 = laying out, 2 = laid out */
    /* An imported module's declarations are all registered, because one of them
     * may name another in an annotation, but only the ones this file actually
     * imported answer to a bare name here. `alias` is the name it was imported
     * under when that differs from the name it was declared with. */
    bool        visible;
    const char *alias;
} DeclEntry;

/* A name that resolves to a type: an alias, or a generic parameter that is in
 * scope only for the declaration that introduced it. */
typedef struct {
    const char *name;
    JaiType    *type;
} NamedType;

typedef struct {
    Symbol  *sym;
    JaiType *saved;
} NarrowSave;

/* A module bound by `import m`, and what its source says it declares. */
typedef struct {
    Symbol    *sym;
    ModuleSig *sig;
} ImportedModule;

/* The per-run bookkeeping. One object rather than a member of Checker because
 * jaiTypeDeclFind takes no context and must answer from anywhere, including
 * the code generator; defined in check.c. */
typedef struct {
    /* Entries are arena-allocated and held by pointer, not by value: a
     * declaration's layout can discover another module's type and register it,
     * and a `DeclEntry *` that a layout in progress is holding must not move
     * under it. */
    JAI_VEC(DeclEntry *) decls;
    JAI_VEC(NamedType)  names;
    JAI_VEC(NarrowSave) narrow;
    /* The module scope, kept alive by the program node after the resolver
     * popped it: the only place a top-level import binding can still be
     * found once resolution is over. */
    Scope *moduleScope;
    /* Set while a declaration read out of another module's source is being laid
     * out. A type name inside it means whatever it meant *there*: a builtin, a
     * sibling of the same module, or — since this file cannot see that module's
     * own imports — nothing, which is `any`. Never this file's names, or two
     * modules that both declare a `Node` would silently share one. */
    ModuleSig *foreignOrigin;
    /* Modules bound by `import m`, so that `m.f(...)` can be checked. */
    JAI_VEC(ImportedModule) modules;
    /* E0900/E0901 are reported only where a constant is required; elsewhere a
     * failed fold just leaves the expression for the VM, which raises the
     * matching exception at run time. */
    bool constRequired;
} JaiCheckState;

extern JaiCheckState gJaiCheck;

/* Reading a declaration that belongs to another module runs the ordinary
 * machinery over a tree this compilation does not own, so its two side channels
 * have to be redirected. Names resolve in that module's namespace — always.
 * Diagnostics are dropped — when `quiet`, which is every path that is only
 * *reading* a signature: a method without `self` or an override that does not
 * fit is that module's own compilation to report, and reporting it here would
 * fail `check` on a file with nothing wrong with it. `errorCount` is restored
 * with the bag, or the failure would be silent instead of absent. */
typedef struct {
    ModuleSig *origin;
    JaiDiagBag bag;
    int        errorCount;    /* -1 when this level did not swap the bag */
} ForeignCtx;

typedef struct {
    Symbol  *sym;
    JaiType *by;
    bool     positive;
} NarrowFact;

/* §7.2: `Self` is the implementing type. Static storage, because interned
 * types keep the name pointer rather than copying it. */
extern const char *jaiChkSelfName;

/* Bits of JaiType.fnFlags, set when a signature is built (check_stmt.c) and
 * read when a call is bound against it (check_expr.c). */
enum { FN_FLAG_VARIADIC = 1, FN_FLAG_KWREST = 2 };

#define TYPE_BUF 160
#define MAX_FACTS 8

/* jaiDiagError never returns NULL, so the result is always safe to decorate. */
#define ERR(c, code, span, ...) \
    (jaiChkCountError(c), jaiDiagError((code), (span), __VA_ARGS__))
#define WARN(code, span, ...)   jaiDiagWarn((code), (span), __VA_ARGS__)

/* --- check.c — the checker's state and shared machinery --------------- */

void jaiChkCountError(Checker *c);
const char *jaiChkRenderType(JaiType *t, char *buf, size_t size);
bool jaiChkIsAny(JaiType *t);
bool jaiChkIsVoid(JaiType *t);
bool jaiChkIsNever(JaiType *t);
JaiType *jaiChkOrAny(JaiType *t);
bool jaiChkSameName(const char *a, const char *b);
JaiType **jaiChkTypeArray(Checker *c, int count);
bool jaiChkCloseEnough(const char *name, const char *candidate, int *best);
DeclEntry *jaiChkDeclEntry(const TypeDecl *d);
TypeDecl *jaiChkDeclForNode(const AstNode *node);
TypeDecl *jaiChkIdentTypeDecl(const Symbol *s);
ModuleSig *jaiChkOriginOfNode(const AstNode *node);

void jaiChkForeignBegin(Checker *c, ModuleSig *origin, bool quiet,
                        ForeignCtx *saved);

void jaiChkForeignEnd(Checker *c, ForeignCtx *saved);
int jaiChkFindFieldIndex(const TypeDecl *d, const char *name);
int jaiChkFindMemberIndex(const TypeDecl *d, const char *name);
int jaiChkFindAccessorIndex(const TypeDecl *d, const char *name, bool wantSetter);
const TypeDecl *jaiChkOwnerOfField(const TypeDecl *d, const char *name);
const TypeDecl *jaiChkOwnerOfMember(const TypeDecl *d, const char *name);
bool jaiChkInheritsOpaquely(const TypeDecl *d);
JaiType *jaiChkDeclType(const TypeDecl *d);
TypeDecl *jaiChkTypeDeclOf(JaiType *t);
void jaiChkPushTypeName(const char *name, JaiType *type);
void jaiChkPushSelf(JaiType *type);
int jaiChkTypeNameMark(void);
void jaiChkTypeNameRestore(int mark);
JaiType *jaiChkLookupTypeName(Checker *c, const char *name);
JaiType *jaiChkResolveAstType(Checker *c, AstType *t);
int jaiChkNarrowMark(void);
void jaiChkNarrowRestore(int mark);
void jaiChkNarrowApply(const NarrowFact *facts, int count, bool invert);
JaiType *jaiChkDeclaredType(Symbol *sym);
void jaiChkNarrowInvalidate(Symbol *sym);
int jaiChkCollectFacts(Checker *c, AstNode *cond, NarrowFact *out, int max, bool negate);
bool jaiChkRetypeIntLiteral(AstNode *node);
void jaiChkApplyContext(Checker *c, AstNode *node, JaiType *expected);

bool jaiChkRequireAssignable(Checker *c, AstNode *node, JaiType *from, JaiType *to,
                             JaiDiagCode code, const char *what);
/* Rewrite `node`, an expression of type int, so that it produces `to` (a
 * float): §2.2's widening, made explicit in the tree. */
void jaiChkWidenToFloat(Checker *c, AstNode *node, JaiType *to);

/* --- check_fold.c — constant folding and evaluation ------------------- */

void jaiChkTryFold(Checker *c, AstNode *node);

/* --- check_expr.c — expression checking ------------------------------- */

const char *jaiChkSubjectOf(const AstNode *n, const char *fallback);
void jaiChkCondition(Checker *c, AstNode *node, const char *where);
JaiType *jaiChkIterableElement(Checker *c, AstNode *node, JaiType *t);
JaiType *jaiChkValue(Checker *c, AstNode *node);
bool jaiChkIsArithmetic(OpKind op);
bool jaiChkWidensOperands(OpKind op);
JaiType *jaiChkSlice(Checker *c, AstNode *node);
bool jaiChkVisibleFrom(Checker *c, const TypeDecl *owner, AstVisibility vis);
const char *jaiChkVisibilityWord(AstVisibility vis);
void jaiChkSuggestMember(JaiDiag *d, const TypeDecl *decl, const char *name);
TypeDecl *jaiChkStaticTargetDecl(Checker *c, AstNode *object);
JaiType *jaiChkForeignFunctionType(Checker *c, ModuleSig *origin, AstNode *fn);
JaiType *jaiChkMember(Checker *c, AstNode *node, AstNode **outDecl);
int jaiChkSelfSkipOf(const AstNode *fn);
AstNode *jaiChkFindInitDecl(TypeDecl *decl, TypeDecl **owner);
JaiType *jaiChkExpr(Checker *c, AstNode *node);

/* --- check_stmt.c — statement and control-flow checking --------------- */

void jaiChkPattern(Checker *c, AstNode *pat, JaiType *subject);
JaiType *jaiChkMatchExpr(Checker *c, AstNode *node, bool isStatement);
void jaiChkBlock(Checker *c, AstNode *node);
void jaiChkStmt(Checker *c, AstNode *node);
void jaiChkPushGenerics(Checker *c, AstGeneric *generics, int count);
bool jaiChkIsInitMethod(const AstNode *fn);
JaiType *jaiChkDeclaredReturnType(Checker *c, AstNode *fn, bool isMethod);
JaiType *jaiChkFunctionType(Checker *c, AstNode *fn, bool isMethod);
void jaiChkFunction(Checker *c, AstNode *fn, TypeDecl *owner, bool isMethod);

/* --- check_decl.c — declaration checking ------------------------------ */

TypeDecl *jaiChkRegisterDecl(Checker *c, AstNode *node);
void jaiChkLayoutDecl(Checker *c, DeclEntry *entry);
void jaiChkDecl(Checker *c, AstNode *node);

#endif /* JAI_CHECK_INTERNAL_H */
