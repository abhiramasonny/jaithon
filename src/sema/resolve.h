/* resolve.h — scopes, symbols, name resolution, and slot assignment.
 *
 * The resolver runs after parsing and before type checking. It:
 *   - builds the scope tree and binds every AST_IDENT to a Symbol
 *   - assigns local slots and upvalue indices (the code generator does no
 *     name lookup at all)
 *   - classifies each binding as local / upvalue / global / member
 *   - enforces let-immutability, use-before-declaration, and shadowing rules
 *   - records which functions capture, so non-capturing functions skip closure
 *     allocation entirely
 */
#ifndef JAI_RESOLVE_H
#define JAI_RESOLVE_H

#include "../lang/ast.h"
#include "types.h"

typedef enum {
    SYM_LOCAL,
    SYM_UPVALUE,
    SYM_GLOBAL,
    SYM_PARAM,
    SYM_FIELD,
    SYM_METHOD,
    SYM_CLASS,
    SYM_TRAIT,
    SYM_ENUM,
    SYM_ENUM_VARIANT,
    SYM_MODULE,
    SYM_TYPE_ALIAS,
    SYM_GENERIC_PARAM,
    SYM_BUILTIN,
} SymbolKind;

struct Symbol {
    SymbolKind    kind;
    const char   *name;          /* interned */
    JaiSpan       declSpan;
    JaiType      *type;          /* filled by the type checker */

    int           slot;          /* local slot / upvalue index / field slot */
    int           depth;         /* scope depth of the declaration */
    VarDeclKind   mutability;    /* VD_LET / VD_VAR / VD_CONST */
    AstVisibility visibility;

    bool          isCaptured;    /* some inner function closes over it */
    bool          isUsed;
    bool          isInitialized;
    bool          isConstFolded;
    struct AstNode *constValue;  /* for VD_CONST after folding */

    struct Symbol  *next;        /* hash-chain within a Scope */
    struct AstNode *decl;        /* declaring node */
    /* For a binding made by `from m import f`: the declaration of `f` in `m`,
     * read out of that module's source by the checker (src/sema/modsig.c).
     * `decl` above stays the import statement, which is what says the binding
     * is an import at all; this is what says what it names. NULL when the
     * module could not be read, or when the name is not one it declares. */
    struct AstNode *importedDecl;
};

typedef enum { SCOPE_MODULE, SCOPE_FUNCTION, SCOPE_BLOCK, SCOPE_CLASS, SCOPE_LOOP } ScopeKind;

typedef struct Scope Scope;

typedef struct {
    int  index;      /* local slot in the enclosing function, or upvalue index */
    bool isLocal;
    /* Spec §6: a `var` is captured by reference and a `let` by value. Only the
     * innermost capture of a `let` local snapshots it; the re-captures along a
     * chain then copy that snapshot, which is the same value. */
    bool byValue;
} UpvalueRef;

/* Per-function resolution state; reachable from AstNode.as.fn.resolveInfo. */
typedef struct FunctionScope {
    struct FunctionScope *enclosing;
    AstNode    *decl;
    int         localCount;      /* peak slot usage */
    int         nextSlot;
    UpvalueRef  upvalues[JAI_MAX_UPVALUES];
    int         upvalueCount;
    bool        capturesAnything;
    bool        hasDefer;
    bool        isGenerator;
    int         loopDepth;
} FunctionScope;

typedef struct {
    AstContext   *ast;
    Scope        *current;
    FunctionScope *currentFn;
    struct ObjModule *module;    /* for globals, may be NULL when checking only */
    JAI_VEC(Symbol *) allSymbols;
    JaiArena      arena;
    int           errorCount;
    bool          replMode;      /* undefined globals are deferred to runtime */
} Resolver;

void jaiResolverInit(Resolver *r, AstContext *ast);
void jaiResolverFree(Resolver *r);

/* Resolve a whole program. Returns false if any error was reported. */
bool jaiResolveProgram(Resolver *r, AstNode *program);

/* Scope operations, exposed for the type checker's narrowing scopes. */
Scope  *jaiScopePush(Resolver *r, ScopeKind kind);
void    jaiScopePop(Resolver *r);
Symbol *jaiScopeDeclare(Resolver *r, const char *name, SymbolKind kind,
                        JaiSpan span, VarDeclKind mut);
Symbol *jaiScopeLookup(Resolver *r, const char *name);
Symbol *jaiScopeLookupLocal(Scope *scope, const char *name);

/* Registry of the built-in globals (print, len, range, ...) so that the
 * resolver can bind them without a module. The symbol is returned so that a
 * caller who knows more than a name and a type — the runtime, which can see
 * that the value is a class, a trait or an enum — can refine `kind`; the
 * registry owns `name`, so a type interned against it outlives the caller. */
Symbol *jaiResolverRegisterBuiltin(const char *name, JaiType *type);
Symbol *jaiResolverFindBuiltin(const char *name);

#endif /* JAI_RESOLVE_H */
