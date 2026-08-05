/* resolve.c — scope tree construction, name binding, and slot assignment.
 *
 * Two passes per block. Declarations that may be mutually recursive (fn, class,
 * trait, enum, type alias, import) are hoisted first; then the statements are
 * walked in order so that let/var/const bind at their declaration point. That
 * split is exactly what makes `fn a() { b() } fn b() { a() }` legal while
 * `print(x); let x = 1` is E0201.
 *
 * Every AST_IDENT leaves this pass pointing at a Symbol whose kind and slot say
 * which instruction the code generator must emit. No later pass looks a name up
 * again.
 */
#include "resolve.h"

/* ------------------------------------------------------------------ */
/* Scope                                                                */
/* ------------------------------------------------------------------ */

/* Declaration-ordered list of a scope's symbols; used for the unused-binding
 * warnings, which must be reported in source order. */
typedef struct SymNode {
    Symbol         *sym;
    struct SymNode *next;
} SymNode;

/* A name the hoist pass saw declared later in this block. Reading it before the
 * declaration statement is reached is E0201, not a silent bind to an outer
 * binding of the same name. */
typedef struct PendingDecl {
    const char         *name;
    JaiSpan             span;
    bool                cleared;
    struct PendingDecl *next;
} PendingDecl;

struct Scope {
    /* `fn` is deliberately first: AST_PROGRAM's opaque `scope` pointer is the
     * only handle the code generator has on the module body's FunctionScope,
     * and it can read it as *(FunctionScope **)scope. See the report. */
    FunctionScope *fn;

    Scope         *parent;
    Scope         *funcRoot;     /* innermost enclosing FUNCTION/MODULE scope */
    ScopeKind      kind;
    int            depth;
    int            savedSlot;    /* fn->nextSlot on entry; restored on pop */

    Symbol       **buckets;
    int            bucketCount;  /* power of two */
    int            count;

    SymNode       *first, *last;
    Symbol        *upvalues;     /* SYM_UPVALUE aliases, chained by ->next */
    PendingDecl   *pending;

    const char    *label;        /* SCOPE_LOOP */
    AstNode       *owner;        /* loop / class / function node, may be NULL */

    /* Lowest slot whose binding this scope must close on the way out, or -1.
     * SCOPE_LOOP: anything declared anywhere inside the loop, closed at every
     * iteration edge (markCapturedInLoop). SCOPE_BLOCK: only what the block
     * declares itself, closed when the block exits (markCapturedInBlock). */
    int            captureBase;
};

static const char kSelfName[] = "self";

static uint64_t hashName(const char *name) {
    return jaiHashBytes(name, strlen(name));
}

static bool nameEq(const char *a, const char *b) {
    return a == b || (a != NULL && b != NULL && strcmp(a, b) == 0);
}

/* Names starting with '_' opt out of the unused/shadowing warnings. */
static bool isIgnoredName(const char *name) {
    return name == NULL || name[0] == '_';
}

/* ------------------------------------------------------------------ */
/* Built-in registry                                                    */
/* ------------------------------------------------------------------ */

/* Built-ins outlive any single Resolver (the REPL creates one per line), so
 * they are heap-allocated rather than arena-allocated. */
static JAI_VEC(Symbol *) gBuiltins;

Symbol *jaiResolverRegisterBuiltin(const char *name, JaiType *type) {
    if (name == NULL) return NULL;

    Symbol *existing = jaiResolverFindBuiltin(name);
    if (existing != NULL) {
        existing->type = type;   /* re-registration updates the signature */
        return existing;
    }

    Symbol *sym = JAI_ALLOC_ZEROED(Symbol, 1);
    sym->kind = SYM_BUILTIN;
    sym->name = jaiStrdup(name);
    sym->declSpan = JAI_SPAN_NONE;
    sym->type = type;
    sym->slot = -1;
    sym->depth = -1;
    sym->mutability = VD_CONST;
    sym->visibility = AST_VIS_PUBLIC;
    sym->isInitialized = true;
    sym->isUsed = true;          /* never warn about an unused built-in */
    JAI_VEC_PUSH(Symbol *, &gBuiltins, sym);
    return sym;
}

Symbol *jaiResolverFindBuiltin(const char *name) {
    if (name == NULL) return NULL;
    for (int i = 0; i < gBuiltins.count; i++) {
        if (nameEq(gBuiltins.data[i]->name, name)) return gBuiltins.data[i];
    }
    return NULL;
}

/* ------------------------------------------------------------------ */
/* Resolver lifecycle                                                   */
/* ------------------------------------------------------------------ */

#define RESOLVE_ARENA_BLOCK ((size_t)(32 * 1024))

void jaiResolverInit(Resolver *r, AstContext *ast) {
    if (r == NULL) return;
    memset(r, 0, sizeof *r);
    r->ast = ast;
    JAI_VEC_INIT(&r->allSymbols);
    jaiArenaInit(&r->arena, RESOLVE_ARENA_BLOCK);
}

void jaiResolverFree(Resolver *r) {
    if (r == NULL) return;
    JAI_VEC_FREE(Symbol *, &r->allSymbols);
    jaiArenaFree(&r->arena);
    r->current = NULL;
    r->currentFn = NULL;
    r->module = NULL;
}

static void resolverError(Resolver *r, JaiDiag *d) {
    (void)d;
    if (r != NULL) r->errorCount++;
}

/* ------------------------------------------------------------------ */
/* Scope operations                                                     */
/* ------------------------------------------------------------------ */

static void scopeRehash(Resolver *r, Scope *s) {
    int newCap = s->bucketCount < 8 ? 8 : s->bucketCount * 2;
    Symbol **buckets = JAI_ARENA_NEW_ARRAY(&r->arena, Symbol *, newCap);
    for (int i = 0; i < s->bucketCount; i++) {
        Symbol *sym = s->buckets[i];
        while (sym != NULL) {
            Symbol *next = sym->next;
            int b = (int)(hashName(sym->name) & (uint64_t)(newCap - 1));
            sym->next = buckets[b];
            buckets[b] = sym;
            sym = next;
        }
    }
    s->buckets = buckets;
    s->bucketCount = newCap;
}

Symbol *jaiScopeLookupLocal(Scope *scope, const char *name) {
    if (scope == NULL || name == NULL || scope->bucketCount == 0) return NULL;
    int b = (int)(hashName(name) & (uint64_t)(scope->bucketCount - 1));
    for (Symbol *sym = scope->buckets[b]; sym != NULL; sym = sym->next) {
        if (nameEq(sym->name, name)) return sym;
    }
    return NULL;
}

Symbol *jaiScopeLookup(Resolver *r, const char *name) {
    if (r == NULL) return NULL;
    for (Scope *s = r->current; s != NULL; s = s->parent) {
        Symbol *sym = jaiScopeLookupLocal(s, name);
        if (sym != NULL) return sym;
    }
    return jaiResolverFindBuiltin(name);
}

/* A binding that lives in a frame slot, and so can be captured. */
static bool kindIsSlotted(SymbolKind kind) {
    return kind == SYM_LOCAL || kind == SYM_PARAM;
}

static bool symbolIsSlotted(const Symbol *sym) {
    return kindIsSlotted(sym->kind);
}

/* Which unused-binding warning, if any, this symbol deserves at scope exit. */
static void reportUnusedSymbol(const Symbol *sym) {
    if (sym->isUsed || isIgnoredName(sym->name)) return;
    if (sym->visibility == AST_VIS_PUBLIC) return;   /* part of the interface */
    if (sym->decl == NULL) return;

    switch (sym->decl->kind) {
    case AST_IMPORT:
    case AST_FROM_IMPORT: {
        JaiDiag *d = jaiDiagWarn(W0100_UNUSED_IMPORT, sym->declSpan,
                                 "unused import `%s`", sym->name);
        jaiDiagAddHelp(d, "remove the import, or prefix the binding with `_`");
        break;
    }
    case AST_VAR_DECL:
    case AST_FOR:
    case AST_TRY:
    case AST_MATCH:
    case AST_MATCH_EXPR:
    case AST_COMPREHENSION: {
        JaiDiag *d = jaiDiagWarn(W0101_UNUSED_BINDING, sym->declSpan,
                                 "unused binding `%s`", sym->name);
        jaiDiagAddHelp(d, "prefix the name with `_` if this is intentional");
        break;
    }
    default:
        break;   /* functions, classes, traits, enums, parameters: never warn */
    }
}

Scope *jaiScopePush(Resolver *r, ScopeKind kind) {
    if (r == NULL) return NULL;
    Scope *s = JAI_ARENA_NEW(&r->arena, Scope);
    s->parent = r->current;
    s->kind = kind;
    s->depth = r->current != NULL ? r->current->depth + 1 : 0;
    s->fn = r->currentFn;
    s->savedSlot = r->currentFn != NULL ? r->currentFn->nextSlot : 0;
    s->captureBase = -1;
    s->funcRoot = (r->current == NULL || kind == SCOPE_FUNCTION || kind == SCOPE_MODULE)
                      ? s
                      : r->current->funcRoot;
    r->current = s;
    return s;
}

void jaiScopePop(Resolver *r) {
    if (r == NULL || r->current == NULL) return;
    Scope *s = r->current;

    for (SymNode *n = s->first; n != NULL; n = n->next) {
        reportUnusedSymbol(n->sym);
    }

    if (s->fn != NULL) {
        if (s->fn->nextSlot > s->fn->localCount) s->fn->localCount = s->fn->nextSlot;
        /* Sibling blocks reuse the slots this one held. */
        if (s->kind != SCOPE_FUNCTION && s->kind != SCOPE_MODULE) {
            s->fn->nextSlot = s->savedSlot;
        }
    }
    r->current = s->parent;
}

/* Next free slot in the enclosing function frame. Slot 0 is callee/self. */
static int allocSlot(Resolver *r, JaiSpan span) {
    FunctionScope *fn = r->currentFn;
    if (fn == NULL) return -1;
    if (fn->nextSlot >= JAI_MAX_LOCALS) {
        /* Report once, on the declaration that crosses the line, then keep
         * handing out the same slot so the rest of the pass still runs. */
        if (fn->nextSlot == JAI_MAX_LOCALS) {
            JaiDiag *d = jaiDiagError(E0208_TOO_MANY_LOCALS, span,
                                      "too many local variables in one function (max %d)",
                                      JAI_MAX_LOCALS);
            resolverError(r, d);
            fn->nextSlot++;
        }
        return JAI_MAX_LOCALS - 1;
    }
    int slot = fn->nextSlot++;
    if (fn->nextSlot > fn->localCount) fn->localCount = fn->nextSlot;
    return slot;
}

/* Innermost slotted binding of this name outside `scope`, ignoring class
 * bodies. Only used to decide whether a declaration shadows (W0103). */
static Symbol *findShadowed(Scope *scope, const char *name) {
    for (Scope *s = scope != NULL ? scope->parent : NULL; s != NULL; s = s->parent) {
        if (s->kind == SCOPE_CLASS) continue;
        Symbol *sym = jaiScopeLookupLocal(s, name);
        if (sym != NULL) return symbolIsSlotted(sym) ? sym : NULL;
    }
    return NULL;
}

Symbol *jaiScopeDeclare(Resolver *r, const char *name, SymbolKind kind,
                        JaiSpan span, VarDeclKind mut) {
    if (r == NULL || r->current == NULL || name == NULL) return NULL;
    Scope *scope = r->current;

    /* `_` is a deliberate discard: redeclaring it is legal and never warns. */
    bool isDiscard = (name[0] == '_' && name[1] == '\0');

    Symbol *previous = jaiScopeLookupLocal(scope, name);
    bool duplicate = previous != NULL && !isDiscard;
    if (duplicate) {
        JaiDiagCode code = scope->kind == SCOPE_CLASS ? E0709_DUPLICATE_MEMBER
                                                      : E0302_DUPLICATE_DECLARATION;
        JaiDiag *d = jaiDiagError(code, span,
                                  scope->kind == SCOPE_CLASS
                                      ? "duplicate member `%s`"
                                      : "`%s` is already declared in this scope",
                                  name);
        jaiDiagAddLabel(d, previous->declSpan, "previous declaration of `%s`", name);
        resolverError(r, d);
    } else if (kindIsSlotted(kind) && !isIgnoredName(name)) {
        Symbol *shadowed = findShadowed(scope, name);
        if (shadowed != NULL) {
            JaiDiag *d = jaiDiagWarn(W0103_SHADOWED_BINDING, span,
                                     "declaration of `%s` shadows an outer binding", name);
            jaiDiagAddLabel(d, shadowed->declSpan, "`%s` first declared here", name);
        }
    }

    Symbol *sym = JAI_ARENA_NEW(&r->arena, Symbol);
    sym->kind = kind;
    sym->name = name;
    sym->declSpan = span;
    sym->slot = -1;
    sym->depth = scope->depth;
    sym->mutability = mut;
    sym->visibility = AST_VIS_PRIVATE;
    if (symbolIsSlotted(sym)) sym->slot = allocSlot(r, span);

    if (scope->bucketCount == 0 || scope->count + 1 > scope->bucketCount) {
        scopeRehash(r, scope);
    }
    int b = (int)(hashName(name) & (uint64_t)(scope->bucketCount - 1));
    sym->next = scope->buckets[b];
    scope->buckets[b] = sym;
    scope->count++;

    /* A duplicate replaces the earlier binding for lookup purposes but is not
     * listed twice for the unused-binding walk. */
    if (!duplicate && !isDiscard) {
        SymNode *node = JAI_ARENA_NEW(&r->arena, SymNode);
        node->sym = sym;
        if (scope->last != NULL) scope->last->next = node; else scope->first = node;
        scope->last = node;
    }

    JAI_VEC_PUSH(Symbol *, &r->allSymbols, sym);
    return sym;
}

/* ------------------------------------------------------------------ */
/* Pending declarations (E0201)                                         */
/* ------------------------------------------------------------------ */

static void pendingAdd(Resolver *r, const char *name, JaiSpan span) {
    if (r->current == NULL || name == NULL) return;
    for (PendingDecl *p = r->current->pending; p != NULL; p = p->next) {
        if (nameEq(p->name, name)) return;   /* first declaration wins */
    }
    PendingDecl *p = JAI_ARENA_NEW(&r->arena, PendingDecl);
    p->name = name;
    p->span = span;
    p->next = r->current->pending;
    r->current->pending = p;
}

static PendingDecl *pendingFind(Scope *s, const char *name) {
    for (PendingDecl *p = s->pending; p != NULL; p = p->next) {
        if (!p->cleared && nameEq(p->name, name)) return p;
    }
    return NULL;
}

static void pendingClear(Scope *s, const char *name) {
    for (PendingDecl *p = s->pending; p != NULL; p = p->next) {
        if (nameEq(p->name, name)) p->cleared = true;
    }
}

/* ------------------------------------------------------------------ */
/* Upvalues                                                             */
/* ------------------------------------------------------------------ */

static int addUpvalue(Resolver *r, FunctionScope *fn, int idx, bool isLocal,
                      bool byValue, JaiSpan span) {
    for (int i = 0; i < fn->upvalueCount; i++) {
        if (fn->upvalues[i].index == idx && fn->upvalues[i].isLocal == isLocal &&
            fn->upvalues[i].byValue == byValue) {
            return i;
        }
    }
    if (fn->upvalueCount >= JAI_MAX_UPVALUES) {
        JaiDiag *d = jaiDiagError(E0209_TOO_MANY_UPVALUES, span,
                                  "too many captured variables in one function (max %d)",
                                  JAI_MAX_UPVALUES);
        resolverError(r, d);
        return 0;
    }
    fn->upvalues[fn->upvalueCount].index = idx;
    fn->upvalues[fn->upvalueCount].isLocal = isLocal;
    fn->upvalues[fn->upvalueCount].byValue = byValue;
    fn->capturesAnything = true;
    return fn->upvalueCount++;
}

/* Crafting Interpreters' resolveUpvalue, threaded through the FunctionScope
 * chain: the innermost function captures the owner's local directly, and every
 * function in between re-captures the upvalue of the one outside it. */
static int captureThrough(Resolver *r, FunctionScope *fn, FunctionScope *owner,
                          int localSlot, bool byValue, JaiSpan span) {
    if (fn == NULL || fn->enclosing == NULL) return -1;
    if (fn->enclosing == owner)
        return addUpvalue(r, fn, localSlot, true, byValue, span);
    int outer = captureThrough(r, fn->enclosing, owner, localSlot, byValue, span);
    if (outer < 0) return -1;
    return addUpvalue(r, fn, outer, false, byValue, span);
}

/* Spec §6: `var` is captured by reference, `let` by value. A nested `fn`
 * declaration is neither — it is a name bound to the closure being built, and
 * snapshotting it would capture the null that is there a moment before the
 * assignment, breaking recursion. */
static bool capturesByValue(const Symbol *sym) {
    if (sym->mutability != VD_LET && sym->mutability != VD_CONST) return false;
    return sym->decl == NULL || sym->decl->kind != AST_FN_DECL;
}

/* Spec §5.2: each iteration of a loop declares its bindings afresh, so two
 * closures made in two iterations must not share a cell. A by-value capture is
 * already a snapshot; a by-reference one aliases the frame slot, and the only
 * way to hand the next iteration a new cell is to close the open upvalue at the
 * iteration boundary. Record the lowest such slot on the *innermost* enclosing
 * loop: its close runs before any outer iteration boundary is reached, so
 * marking the outer loops as well would only buy duplicate work.
 *
 * `declScope` is where the binding lives, which is what decides which loop owns
 * it — a capture found three functions deep still belongs to the loop that
 * declared the name. */
static void markCapturedInLoop(Scope *declScope, int slot) {
    if (slot < 0) return;
    for (Scope *s = declScope; s != NULL; s = s->parent) {
        if (s->kind == SCOPE_LOOP) {
            if (s->captureBase < 0 || slot < s->captureBase) s->captureBase = slot;
            return;
        }
        /* Parameters and function-level locals outlive no iteration. */
        if (s->kind == SCOPE_FUNCTION || s->kind == SCOPE_MODULE) return;
    }
}

/* The same argument one level down. A plain block's bindings die when the block
 * exits and the next sibling block is handed their slots (jaiScopePop rewinds
 * `nextSlot`), so a closure that escaped the block would go on aliasing a slot
 * that now holds something else entirely — the defect is invisible until the
 * closure is called.
 *
 * Only the scope that *declares* the slot records it: an inner block exiting
 * must not close an outer block's still-live bindings, and the innermost close
 * runs first on every path out, so marking the enclosing blocks too would only
 * buy duplicate work. A slot declared straight in a loop or function scope is
 * handled by the loop's iteration close and by OP_RETURN respectively. */
static void markCapturedInBlock(Scope *declScope, int slot) {
    if (slot < 0 || declScope == NULL || declScope->kind != SCOPE_BLOCK) return;
    if (declScope->captureBase < 0 || slot < declScope->captureBase) {
        declScope->captureBase = slot;
    }
}

/* One SYM_UPVALUE alias per (function, upvalue index), so that codegen can emit
 * GET_UPVALUE from the symbol alone and the checker types it once. */
static Symbol *upvalueAlias(Resolver *r, Symbol *sym, int idx) {
    Scope *root = r->current != NULL ? r->current->funcRoot : NULL;
    if (root == NULL) return sym;

    for (Symbol *u = root->upvalues; u != NULL; u = u->next) {
        if (u->slot == idx && nameEq(u->name, sym->name)) return u;
    }

    Symbol *u = JAI_ARENA_NEW(&r->arena, Symbol);
    u->kind = SYM_UPVALUE;
    u->name = sym->name;
    u->declSpan = sym->declSpan;
    u->slot = idx;
    u->depth = root->depth;
    u->mutability = sym->mutability;
    u->visibility = sym->visibility;
    u->isInitialized = true;
    u->isUsed = true;
    u->decl = sym->decl;
    u->next = root->upvalues;
    root->upvalues = u;
    JAI_VEC_PUSH(Symbol *, &r->allSymbols, u);
    return u;
}

/* ------------------------------------------------------------------ */
/* "did you mean" suggestions                                           */
/* ------------------------------------------------------------------ */

static void considerSuggestion(const char *name, const char *candidate,
                               const char **best, int *bestDist) {
    if (candidate == NULL || nameEq(candidate, name)) return;
    if (jaiNameIsCloser(name, candidate, bestDist)) *best = candidate;
}

/* Closest visible name, by the shared tolerance in diag.h. */
static const char *suggestName(Resolver *r, const char *name) {
    const char *best = NULL;
    int bestDist = JAI_SUGGEST_NO_MATCH;

    for (Scope *s = r->current; s != NULL; s = s->parent) {
        for (int i = 0; i < s->bucketCount; i++) {
            for (Symbol *sym = s->buckets[i]; sym != NULL; sym = sym->next) {
                considerSuggestion(name, sym->name, &best, &bestDist);
            }
        }
    }
    for (int i = 0; i < gBuiltins.count; i++) {
        considerSuggestion(name, gBuiltins.data[i]->name, &best, &bestDist);
    }
    return best;
}

/* ------------------------------------------------------------------ */
/* Walk context                                                         */
/* ------------------------------------------------------------------ */

/* One per class/trait/enum body, so that field slots can be assigned in
 * inheritance order once every class in the program has been seen. */
typedef struct ClassInfo {
    AstNode          *decl;
    Symbol           *sym;
    Scope            *scope;
    Symbol           *superSym;      /* resolved superclass symbol, may be NULL */
    struct ClassInfo *super;         /* linked in the field-slot pass */
    int               base;          /* first own instance-field slot */
    int               total;         /* base + own instance field count */
    uint8_t           state;         /* 0 unvisited, 1 visiting, 2 done */
} ClassInfo;

typedef struct LoopCtx {
    struct LoopCtx *enclosing;
    const char     *label;
    FunctionScope  *fn;
} LoopCtx;

typedef struct ClassCtx {
    struct ClassCtx *enclosing;
    ClassInfo       *info;
    bool             hasSuper;
} ClassCtx;

typedef struct {
    Resolver *r;
    LoopCtx  *loops;
    ClassCtx *classes;
    JAI_VEC(ClassInfo *) classList;
    bool      selfOk;          /* directly inside a non-static method body */
    bool      superOk;
    bool      wildcardImport;  /* `from m import *` seen: unknown names defer */
} Rz;

typedef struct {
    bool      isMethod;
    bool      isStatic;
    bool      requireSelf;     /* plain methods; getters/setters take none */
    ClassCtx *cls;
} MethodInfo;

static void resolveNode(Rz *rz, AstNode *node);
static void resolveBlockContents(Rz *rz, AstNode **stmts, int count);
static void resolveFunction(Rz *rz, AstNode *node, const MethodInfo *mi);
static void resolvePattern(Rz *rz, AstNode *pat, bool declare, VarDeclKind mut,
                           AstNode *declNode, bool orAlternative);
static void resolveTypeExpr(Rz *rz, AstType *type);

static void resolveEach(Rz *rz, AstNode **nodes, int count) {
    if (nodes == NULL) return;
    for (int i = 0; i < count; i++) resolveNode(rz, nodes[i]);
}

/* Is the current function the module body rather than a real `fn`? */
static bool inFunctionBody(const Rz *rz) {
    FunctionScope *fn = rz->r->currentFn;
    return fn != NULL && fn->decl != NULL && fn->decl->kind != AST_PROGRAM;
}

/* ------------------------------------------------------------------ */
/* Identifier resolution                                                */
/* ------------------------------------------------------------------ */

/* A name that cannot be resolved statically but is still legal — REPL globals
 * and anything a wildcard import may have brought in. Declared once in the
 * module scope so that repeated uses share a symbol. */
static Symbol *deferredGlobal(Rz *rz, const char *name, JaiSpan span) {
    Resolver *r = rz->r;
    Scope *root = r->current;
    while (root != NULL && root->parent != NULL) root = root->parent;
    if (root == NULL) return NULL;

    Symbol *sym = jaiScopeLookupLocal(root, name);
    if (sym != NULL) return sym;

    Scope *saved = r->current;
    r->current = root;
    sym = jaiScopeDeclare(r, name, SYM_GLOBAL, span, VD_VAR);
    r->current = saved;
    if (sym != NULL) {
        sym->isInitialized = true;
        sym->isUsed = true;
    }
    return sym;
}

static void reportUseBeforeDecl(Rz *rz, const char *name, JaiSpan use,
                                const PendingDecl *pending) {
    JaiDiag *d = jaiDiagError(E0201_USE_BEFORE_DECLARATION, use,
                              "cannot use `%s` before it is declared", name);
    jaiDiagAddLabel(d, pending->span, "`%s` is declared here, later in this scope", name);
    jaiDiagAddHelp(d, "move the declaration above this use");
    resolverError(rz->r, d);
}

/* Bind `name` to a symbol. Class bodies are skipped: fields and methods are
 * reachable only through `self.x` / `Class.x`, never as bare names. */
static Symbol *resolveName(Rz *rz, const char *name, JaiSpan span, bool markUsed) {
    Resolver *r = rz->r;
    if (name == NULL) return NULL;

    PendingDecl *crossFn = NULL;
    bool crossModule = false;

    for (Scope *s = r->current; s != NULL; s = s->parent) {
        if (s->kind != SCOPE_CLASS) {
            Symbol *sym = jaiScopeLookupLocal(s, name);
            if (sym != NULL) {
                if (markUsed) sym->isUsed = true;
                if (symbolIsSlotted(sym) && s->fn != r->currentFn) {
                    sym->isCaptured = true;
                    bool byValue = capturesByValue(sym);
                    if (!byValue) {
                        markCapturedInBlock(s, sym->slot);
                        markCapturedInLoop(s, sym->slot);
                    }
                    int idx = captureThrough(r, r->currentFn, s->fn, sym->slot,
                                             byValue, span);
                    if (idx >= 0) return upvalueAlias(r, sym, idx);
                }
                return sym;
            }
        }
        PendingDecl *p = pendingFind(s, name);
        if (p != NULL) {
            /* Within the current function this is unambiguously a read of a
             * slot that does not exist yet. Across a function boundary the
             * closure might legitimately run later, so keep looking and only
             * fall back to this if nothing else matches. */
            if (s->fn == r->currentFn) {
                reportUseBeforeDecl(rz, name, span, p);
                return NULL;
            }
            /* A module-level binding is initialised by the module body, which
             * runs before anything can call a function declared above it. Only
             * a read from the same block is premature (spec §3), so this one is
             * left unresolved and the code generator emits a global load. */
            if (s->kind == SCOPE_MODULE) crossModule = true;
            else if (crossFn == NULL) crossFn = p;
        }
    }

    Symbol *builtin = jaiResolverFindBuiltin(name);
    if (builtin != NULL) return builtin;

    if (crossFn != NULL) {
        reportUseBeforeDecl(rz, name, span, crossFn);
        return NULL;
    }
    if (crossModule) return NULL;

    if (r->replMode || rz->wildcardImport) return deferredGlobal(rz, name, span);

    JaiDiag *d = jaiDiagError(E0200_UNDEFINED_NAME, span, "undefined name `%s`", name);
    const char *suggestion = suggestName(r, name);
    if (suggestion != NULL) jaiDiagAddHelp(d, "did you mean `%s`?", suggestion);
    resolverError(r, d);
    return NULL;
}

static void resolveIdent(Rz *rz, AstNode *node, bool markUsed) {
    node->as.ident.symbol = resolveName(rz, node->as.ident.name, node->span, markUsed);
}

/* Type annotations do not resolve to bindings here — that is the checker's job
 * — but a name used only in a type must not be reported as unused. */
static void resolveTypeExpr(Rz *rz, AstType *type) {
    if (type == NULL) return;
    /* TYPE_GENERIC carries a name too: `Iterable[T]` uses `Iterable` just as
     * much as a bare `Iterable` does, and missing it made every imported
     * generic trait look like an unused import. */
    if ((type->kind == TYPE_NAME || type->kind == TYPE_GENERIC) &&
        type->name != NULL) {
        for (Scope *s = rz->r->current; s != NULL; s = s->parent) {
            Symbol *sym = jaiScopeLookupLocal(s, type->name);
            if (sym != NULL) { sym->isUsed = true; break; }
        }
    }
    for (int i = 0; i < type->argCount; i++) resolveTypeExpr(rz, type->args[i]);
    resolveTypeExpr(rz, type->inner);
}

/* ------------------------------------------------------------------ */
/* Patterns                                                             */
/* ------------------------------------------------------------------ */

static void collectPatternNames(Rz *rz, AstNode *pat) {
    if (pat == NULL) return;
    switch (pat->kind) {
    case AST_PAT_BIND:
        pendingAdd(rz->r, pat->as.patBind.name, pat->span);
        break;
    case AST_PAT_TUPLE:
    case AST_PAT_LIST:
    case AST_PAT_OR:
        for (int i = 0; i < pat->as.patSeq.count; i++) {
            collectPatternNames(rz, pat->as.patSeq.elems[i]);
        }
        break;
    case AST_PAT_CLASS:
    case AST_PAT_ENUM:
        for (int i = 0; i < pat->as.patClass.count; i++) {
            collectPatternNames(rz, pat->as.patClass.subPatterns[i]);
        }
        break;
    default:
        break;
    }
}

static void clearPatternPending(Rz *rz, AstNode *pat) {
    if (pat == NULL || rz->r->current == NULL) return;
    switch (pat->kind) {
    case AST_PAT_BIND:
        pendingClear(rz->r->current, pat->as.patBind.name);
        break;
    case AST_PAT_TUPLE:
    case AST_PAT_LIST:
    case AST_PAT_OR:
        for (int i = 0; i < pat->as.patSeq.count; i++) {
            clearPatternPending(rz, pat->as.patSeq.elems[i]);
        }
        break;
    case AST_PAT_CLASS:
    case AST_PAT_ENUM:
        for (int i = 0; i < pat->as.patClass.count; i++) {
            clearPatternPending(rz, pat->as.patClass.subPatterns[i]);
        }
        break;
    default:
        break;
    }
}

/* Module-level bindings are globals (spec §8); everything else gets a slot. */
static SymbolKind bindingKind(const Rz *rz) {
    return rz->r->current != NULL && rz->r->current->kind == SCOPE_MODULE
               ? SYM_GLOBAL
               : SYM_LOCAL;
}

static Symbol *declareBinding(Rz *rz, const char *name, JaiSpan span,
                              VarDeclKind mut, AstNode *declNode,
                              bool orAlternative) {
    Resolver *r = rz->r;
    /* Every arm of an or-pattern binds the same names; the first arm declares
     * them and the rest reuse those symbols. */
    if (orAlternative && r->current != NULL) {
        Symbol *existing = jaiScopeLookupLocal(r->current, name);
        if (existing != NULL) return existing;
    }
    Symbol *sym = jaiScopeDeclare(r, name, bindingKind(rz), span, mut);
    if (sym == NULL) return NULL;
    sym->decl = declNode;
    sym->isInitialized = true;
    return sym;
}

static void resolvePattern(Rz *rz, AstNode *pat, bool declare, VarDeclKind mut,
                           AstNode *declNode, bool orAlternative) {
    if (pat == NULL) return;
    switch (pat->kind) {
    case AST_PAT_WILDCARD:
        break;

    case AST_PAT_BIND:
        resolveTypeExpr(rz, pat->as.patBind.type);
        if (declare) {
            pat->as.patBind.symbol = declareBinding(rz, pat->as.patBind.name,
                                                    pat->span, mut, declNode,
                                                    orAlternative);
        } else {
            pat->as.patBind.symbol = resolveName(rz, pat->as.patBind.name,
                                                 pat->span, true);
        }
        break;

    case AST_PAT_LITERAL:
        resolveNode(rz, pat->as.patLiteral.value);
        break;

    case AST_PAT_RANGE:
        resolveNode(rz, pat->as.patRange.lo);
        resolveNode(rz, pat->as.patRange.hi);
        break;

    case AST_PAT_TUPLE:
    case AST_PAT_LIST:
        for (int i = 0; i < pat->as.patSeq.count; i++) {
            resolvePattern(rz, pat->as.patSeq.elems[i], declare, mut, declNode,
                           orAlternative);
        }
        break;

    case AST_PAT_OR:
        for (int i = 0; i < pat->as.patSeq.count; i++) {
            resolvePattern(rz, pat->as.patSeq.elems[i], declare, mut, declNode,
                           i > 0 || orAlternative);
        }
        break;

    case AST_PAT_CLASS:
    case AST_PAT_ENUM: {
        /* The type name must exist; whether the variant does is E0700's job. */
        if (pat->as.patClass.typeName != NULL) {
            for (Scope *s = rz->r->current; s != NULL; s = s->parent) {
                Symbol *sym = jaiScopeLookupLocal(s, pat->as.patClass.typeName);
                if (sym != NULL) { sym->isUsed = true; break; }
            }
        }
        for (int i = 0; i < pat->as.patClass.count; i++) {
            resolvePattern(rz, pat->as.patClass.subPatterns[i], declare, mut,
                           declNode, orAlternative);
        }
        break;
    }

    default:
        /* Not a pattern node: the parser put an expression here. */
        resolveNode(rz, pat);
        break;
    }
}

/* ------------------------------------------------------------------ */
/* Hoisting                                                             */
/* ------------------------------------------------------------------ */

/* Give a declaration that lives in a frame (a local `fn`, `class`, ...) its
 * slot at hoist time: a mutually recursive sibling may capture it before the
 * declaration statement is reached, and the capture records the slot. */
static void slotHoistedDecl(Rz *rz, Symbol *sym, JaiSpan span) {
    if (sym == NULL) return;
    if (rz->r->current != NULL && rz->r->current->kind == SCOPE_MODULE) return;
    sym->slot = allocSlot(rz->r, span);
}

static Symbol *hoistNamed(Rz *rz, const char *name, SymbolKind kind,
                          JaiSpan span, AstNode *decl, AstVisibility vis) {
    if (name == NULL) return NULL;
    Symbol *sym = jaiScopeDeclare(rz->r, name, kind, span, VD_LET);
    if (sym == NULL) return NULL;
    sym->decl = decl;
    sym->visibility = vis;
    sym->isInitialized = true;
    return sym;
}

static const char *importBindingName(const AstNode *node) {
    if (node->as.import.alias != NULL) return node->as.import.alias;
    const char *path = node->as.import.path;
    if (path == NULL) return NULL;
    const char *dot = strrchr(path, '.');
    return dot != NULL && dot[1] != '\0' ? dot + 1 : path;
}

static void hoistDecls(Rz *rz, AstNode **stmts, int count) {
    if (stmts == NULL) return;
    for (int i = 0; i < count; i++) {
        AstNode *n = stmts[i];
        if (n == NULL) continue;
        switch (n->kind) {
        case AST_FN_DECL:
            /* SYM_LOCAL is slotted by jaiScopeDeclare, at hoist time, so a
             * mutually recursive sibling can capture the slot. */
            n->as.fn.symbol = hoistNamed(rz, n->as.fn.name, bindingKind(rz), n->span, n,
                                         n->as.fn.visibility);
            break;
        case AST_CLASS_DECL: {
            Symbol *sym = hoistNamed(rz, n->as.classDecl.name, SYM_CLASS, n->span, n,
                                     n->as.classDecl.visibility);
            slotHoistedDecl(rz, sym, n->span);
            n->as.classDecl.symbol = sym;
            break;
        }
        case AST_TRAIT_DECL: {
            Symbol *sym = hoistNamed(rz, n->as.traitDecl.name, SYM_TRAIT, n->span, n,
                                     n->as.traitDecl.visibility);
            n->as.traitDecl.symbol = sym;
            break;
        }
        case AST_ENUM_DECL: {
            Symbol *sym = hoistNamed(rz, n->as.enumDecl.name, SYM_ENUM, n->span, n,
                                     n->as.enumDecl.visibility);
            slotHoistedDecl(rz, sym, n->span);
            n->as.enumDecl.symbol = sym;
            break;
        }
        case AST_TYPE_DECL: {
            Symbol *sym = hoistNamed(rz, n->as.typeDecl.name, SYM_TYPE_ALIAS, n->span,
                                     n, n->as.typeDecl.visibility);
            n->as.typeDecl.symbol = sym;
            break;
        }
        case AST_IMPORT: {
            Symbol *sym = hoistNamed(rz, importBindingName(n), SYM_MODULE, n->span, n,
                                     AST_VIS_PRIVATE);
            slotHoistedDecl(rz, sym, n->span);
            n->as.import.symbol = sym;
            break;
        }
        case AST_FROM_IMPORT: {
            if (n->as.fromImport.isWildcard) {
                rz->wildcardImport = true;
                break;
            }
            for (int j = 0; j < n->as.fromImport.itemCount; j++) {
                AstImportItem *item = &n->as.fromImport.items[j];
                const char *bound = item->alias != NULL ? item->alias : item->name;
                /* Kept, not discarded: the checker reaches the imported
                 * declaration's parameter list through this binding. */
                item->symbol = hoistNamed(rz, bound, bindingKind(rz), item->span, n,
                                          AST_VIS_PRIVATE);
            }
            break;
        }
        case AST_VAR_DECL:
            /* Not hoisted — only remembered, so that a read above the
             * declaration is E0201 instead of a bind to an outer name. */
            collectPatternNames(rz, n->as.varDecl.pattern);
            break;
        default:
            break;
        }
    }
}

/* ------------------------------------------------------------------ */
/* Functions                                                            */
/* ------------------------------------------------------------------ */

static void declareGenerics(Rz *rz, AstGeneric *generics, int count, AstNode *decl) {
    for (int i = 0; i < count; i++) {
        Symbol *sym = jaiScopeDeclare(rz->r, generics[i].name, SYM_GENERIC_PARAM,
                                      generics[i].span, VD_CONST);
        if (sym != NULL) {
            sym->decl = decl;
            sym->isInitialized = true;
            sym->isUsed = true;
        }
        resolveTypeExpr(rz, generics[i].bound);
    }
}

static void resolveFunction(Rz *rz, AstNode *node, const MethodInfo *mi) {
    Resolver *r = rz->r;

    FunctionScope *fs = JAI_ARENA_NEW(&r->arena, FunctionScope);
    fs->enclosing = r->currentFn;
    fs->decl = node;
    fs->nextSlot = 1;                /* slot 0 is the callee / self */
    fs->localCount = 1;
    fs->isGenerator = node->as.fn.isGenerator;
    node->as.fn.resolveInfo = fs;

    FunctionScope *savedFn = r->currentFn;
    LoopCtx *savedLoops = rz->loops;
    bool savedSelf = rz->selfOk;
    bool savedSuper = rz->superOk;

    r->currentFn = fs;
    rz->loops = NULL;                /* break/continue never cross a function */
    Scope *scope = jaiScopePush(r, SCOPE_FUNCTION);
    scope->owner = node;

    declareGenerics(rz, node->as.fn.generics, node->as.fn.genericCount, node);

    bool isMethod = mi != NULL && mi->isMethod;
    bool isStatic = node->as.fn.isStatic || (mi != NULL && mi->isStatic);
    int firstParam = 0;

    if (isMethod && !isStatic) {
        Symbol *selfSym = jaiScopeDeclare(r, kSelfName, SYM_PARAM, node->span, VD_LET);
        if (selfSym != NULL) {
            selfSym->slot = 0;
            selfSym->decl = node;
            selfSym->isInitialized = true;
            selfSym->isUsed = true;
        }
        /* `self` occupies the frame's slot 0, so the declared parameters still
         * start at 1: the receiver is not an extra slot. */
        fs->nextSlot = 1;
        fs->localCount = 1;

        rz->selfOk = true;
        rz->superOk = mi->cls != NULL && mi->cls->hasSuper;

        bool hasSelfParam = node->as.fn.paramCount > 0 &&
                            nameEq(node->as.fn.params[0].name, kSelfName);
        if (hasSelfParam) {
            node->as.fn.params[0].symbol = selfSym;
            firstParam = 1;
        } else if (mi->requireSelf) {
            JaiDiag *d = jaiDiagError(E0703_MISSING_SELF, node->span,
                                      "instance method `%s` must take `self` as its "
                                      "first parameter",
                                      node->as.fn.name != NULL ? node->as.fn.name : "?");
            jaiDiagAddHelp(d, "write `fn %s(self, ...)`, or mark it `static`",
                           node->as.fn.name != NULL ? node->as.fn.name : "m");
            resolverError(r, d);
        }
    } else {
        rz->selfOk = false;
        rz->superOk = false;
        if (isStatic && node->as.fn.paramCount > 0 &&
            nameEq(node->as.fn.params[0].name, kSelfName)) {
            JaiDiag *d = jaiDiagError(E0707_STATIC_WITH_SELF, node->as.fn.params[0].span,
                                      "static method `%s` must not take `self`",
                                      node->as.fn.name != NULL ? node->as.fn.name : "?");
            resolverError(r, d);
            firstParam = 1;
        }
    }

    for (int i = firstParam; i < node->as.fn.paramCount; i++) {
        AstParam *p = &node->as.fn.params[i];
        resolveTypeExpr(rz, p->type);
        /* Defaults are re-evaluated per call inside the callee frame, so they
         * see the parameters declared before them. */
        if (p->defaultValue != NULL) resolveNode(rz, p->defaultValue);
        Symbol *sym = jaiScopeDeclare(r, p->name, SYM_PARAM, p->span, VD_VAR);
        if (sym != NULL) {
            sym->decl = node;
            sym->isInitialized = true;
        }
        p->symbol = sym;
    }

    resolveTypeExpr(rz, node->as.fn.returnType);

    AstNode *body = node->as.fn.body;
    if (body != NULL) {
        if (node->as.fn.isExprBody || body->kind != AST_BLOCK) {
            resolveNode(rz, body);
        } else {
            /* The body shares the function scope: parameters and top-level
             * body locals are one contiguous run of slots. */
            body->as.block.scope = scope;
            resolveBlockContents(rz, body->as.block.stmts, body->as.block.count);
        }
    }

    if (fs->nextSlot > fs->localCount) fs->localCount = fs->nextSlot;
    jaiScopePop(r);

    node->as.fn.localCount = fs->localCount;
    node->as.fn.upvalueCount = fs->upvalueCount;
    node->as.fn.isGenerator = fs->isGenerator;

    r->currentFn = savedFn;
    rz->loops = savedLoops;
    rz->selfOk = savedSelf;
    rz->superOk = savedSuper;
}

/* ------------------------------------------------------------------ */
/* Classes, traits, enums                                               */
/* ------------------------------------------------------------------ */

static Symbol *lookupTypeName(Rz *rz, const AstType *type) {
    if (type == NULL || type->kind != TYPE_NAME || type->name == NULL) return NULL;
    for (Scope *s = rz->r->current; s != NULL; s = s->parent) {
        Symbol *sym = jaiScopeLookupLocal(s, type->name);
        if (sym != NULL) {
            sym->isUsed = true;
            return sym;
        }
    }
    return NULL;
}

static void declareMember(Rz *rz, AstNode *method, bool reuseExisting) {
    const char *name = method->as.fn.name;
    if (name == NULL) return;
    if (reuseExisting && rz->r->current != NULL) {
        Symbol *existing = jaiScopeLookupLocal(rz->r->current, name);
        if (existing != NULL) {
            method->as.fn.symbol = existing;
            return;
        }
    }
    Symbol *sym = jaiScopeDeclare(rz->r, name, SYM_METHOD, method->span, VD_LET);
    if (sym != NULL) {
        sym->decl = method;
        sym->visibility = method->as.fn.visibility;
        sym->isInitialized = true;
        sym->isUsed = true;      /* members are reached through the class, not
                                  * by name, so "unused" means nothing here */
    }
    method->as.fn.symbol = sym;
}

static void resolveMethods(Rz *rz, AstNode **methods, int count, ClassCtx *ctx,
                           bool requireSelf) {
    for (int i = 0; i < count; i++) {
        AstNode *m = methods[i];
        if (m == NULL || m->kind != AST_FN_DECL) continue;
        MethodInfo mi;
        mi.isMethod = true;
        mi.isStatic = m->as.fn.isStatic;
        mi.requireSelf = requireSelf && !m->as.fn.isStatic;
        mi.cls = ctx;
        resolveFunction(rz, m, &mi);
    }
}

static void resolveClassDecl(Rz *rz, AstNode *node) {
    Resolver *r = rz->r;

    if (node->as.classDecl.symbol == NULL) {
        node->as.classDecl.symbol = hoistNamed(rz, node->as.classDecl.name, SYM_CLASS,
                                               node->span, node,
                                               node->as.classDecl.visibility);
    }

    /* The superclass and the trait list live in the enclosing scope. */
    Symbol *superSym = lookupTypeName(rz, node->as.classDecl.superclass);
    resolveTypeExpr(rz, node->as.classDecl.superclass);
    for (int i = 0; i < node->as.classDecl.traitCount; i++) {
        (void)lookupTypeName(rz, node->as.classDecl.traits[i]);
        resolveTypeExpr(rz, node->as.classDecl.traits[i]);
    }

    ClassInfo *info = JAI_ARENA_NEW(&r->arena, ClassInfo);
    info->decl = node;
    info->sym = node->as.classDecl.symbol;
    info->superSym = superSym;
    JAI_VEC_PUSH(ClassInfo *, &rz->classList, info);

    Scope *scope = jaiScopePush(r, SCOPE_CLASS);
    scope->owner = node;
    info->scope = scope;

    declareGenerics(rz, node->as.classDecl.generics, node->as.classDecl.genericCount,
                    node);

    for (int i = 0; i < node->as.classDecl.fieldCount; i++) {
        AstField *f = &node->as.classDecl.fields[i];
        Symbol *sym = jaiScopeDeclare(r, f->name, SYM_FIELD, f->span,
                                      f->isLet ? VD_LET : VD_VAR);
        if (sym != NULL) {
            sym->decl = node;
            sym->visibility = f->visibility;
            sym->isInitialized = f->defaultValue != NULL;
            sym->isUsed = true;
        }
        resolveTypeExpr(rz, f->type);
    }

    for (int i = 0; i < node->as.classDecl.methodCount; i++) {
        if (node->as.classDecl.methods[i] != NULL) {
            declareMember(rz, node->as.classDecl.methods[i], false);
        }
    }
    for (int i = 0; i < node->as.classDecl.getterCount; i++) {
        if (node->as.classDecl.getters[i] != NULL) {
            declareMember(rz, node->as.classDecl.getters[i], false);
        }
    }
    for (int i = 0; i < node->as.classDecl.setterCount; i++) {
        /* A property's getter and setter share one member name. */
        if (node->as.classDecl.setters[i] != NULL) {
            declareMember(rz, node->as.classDecl.setters[i], true);
        }
    }

    ClassCtx ctx;
    ctx.enclosing = rz->classes;
    ctx.info = info;
    ctx.hasSuper = node->as.classDecl.superclass != NULL;
    rz->classes = &ctx;

    /* Field defaults run before any method body, and without `self`. */
    bool savedSelf = rz->selfOk;
    rz->selfOk = false;
    for (int i = 0; i < node->as.classDecl.fieldCount; i++) {
        resolveNode(rz, node->as.classDecl.fields[i].defaultValue);
    }
    rz->selfOk = savedSelf;

    resolveMethods(rz, node->as.classDecl.methods, node->as.classDecl.methodCount,
                   &ctx, true);
    resolveMethods(rz, node->as.classDecl.getters, node->as.classDecl.getterCount,
                   &ctx, false);
    resolveMethods(rz, node->as.classDecl.setters, node->as.classDecl.setterCount,
                   &ctx, false);

    rz->classes = ctx.enclosing;
    jaiScopePop(r);
}

static void resolveTraitDecl(Rz *rz, AstNode *node) {
    Resolver *r = rz->r;
    if (node->as.traitDecl.symbol == NULL) {
        node->as.traitDecl.symbol = hoistNamed(rz, node->as.traitDecl.name, SYM_TRAIT,
                                               node->span, node,
                                               node->as.traitDecl.visibility);
    }
    for (int i = 0; i < node->as.traitDecl.superCount; i++) {
        (void)lookupTypeName(rz, node->as.traitDecl.supers[i]);
        resolveTypeExpr(rz, node->as.traitDecl.supers[i]);
    }

    Scope *scope = jaiScopePush(r, SCOPE_CLASS);
    scope->owner = node;
    declareGenerics(rz, node->as.traitDecl.generics, node->as.traitDecl.genericCount,
                    node);

    for (int i = 0; i < node->as.traitDecl.methodCount; i++) {
        if (node->as.traitDecl.methods[i] != NULL) {
            declareMember(rz, node->as.traitDecl.methods[i], false);
        }
    }

    ClassCtx ctx;
    ctx.enclosing = rz->classes;
    ctx.info = NULL;
    ctx.hasSuper = false;
    rz->classes = &ctx;
    resolveMethods(rz, node->as.traitDecl.methods, node->as.traitDecl.methodCount,
                   &ctx, true);
    rz->classes = ctx.enclosing;

    jaiScopePop(r);
}

static void resolveEnumDecl(Rz *rz, AstNode *node) {
    Resolver *r = rz->r;
    if (node->as.enumDecl.symbol == NULL) {
        node->as.enumDecl.symbol = hoistNamed(rz, node->as.enumDecl.name, SYM_ENUM,
                                              node->span, node,
                                              node->as.enumDecl.visibility);
    }

    Scope *scope = jaiScopePush(r, SCOPE_CLASS);
    scope->owner = node;
    declareGenerics(rz, node->as.enumDecl.generics, node->as.enumDecl.genericCount,
                    node);

    for (int i = 0; i < node->as.enumDecl.variantCount; i++) {
        AstVariant *v = &node->as.enumDecl.variants[i];
        Symbol *sym = jaiScopeDeclare(r, v->name, SYM_ENUM_VARIANT, v->span, VD_CONST);
        if (sym != NULL) {
            sym->decl = node;
            sym->slot = i;               /* variant tag */
            sym->visibility = node->as.enumDecl.visibility;
            sym->isInitialized = true;
            sym->isUsed = true;
        }
        for (int j = 0; j < v->paramCount; j++) {
            resolveTypeExpr(rz, v->params[j].type);
        }
    }

    for (int i = 0; i < node->as.enumDecl.methodCount; i++) {
        if (node->as.enumDecl.methods[i] != NULL) {
            declareMember(rz, node->as.enumDecl.methods[i], false);
        }
    }

    ClassCtx ctx;
    ctx.enclosing = rz->classes;
    ctx.info = NULL;
    ctx.hasSuper = false;
    rz->classes = &ctx;
    resolveMethods(rz, node->as.enumDecl.methods, node->as.enumDecl.methodCount,
                   &ctx, true);
    rz->classes = ctx.enclosing;

    jaiScopePop(r);
}

/* ------------------------------------------------------------------ */
/* Field slots: parents first, in dependency order                      */
/* ------------------------------------------------------------------ */

static ClassInfo *findClassInfo(Rz *rz, const Symbol *sym) {
    if (sym == NULL) return NULL;
    for (int i = 0; i < rz->classList.count; i++) {
        if (rz->classList.data[i]->sym == sym) return rz->classList.data[i];
    }
    return NULL;
}

static void assignOwnFieldSlots(ClassInfo *info, int base) {
    info->base = base;
    int n = 0;
    for (int i = 0; i < info->decl->as.classDecl.fieldCount; i++) {
        AstField *f = &info->decl->as.classDecl.fields[i];
        if (f->isStatic) continue;                 /* class-level, no instance slot */
        Symbol *sym = jaiScopeLookupLocal(info->scope, f->name);
        if (sym != NULL && sym->kind == SYM_FIELD) sym->slot = base + n;
        n++;
    }
    info->total = base + n;
    info->state = 2;
}

static void assignFieldSlots(Rz *rz, ClassInfo *info) {
    if (info->state == 2) return;
    if (info->state == 1) {
        JaiDiag *d = jaiDiagError(E0706_CYCLIC_INHERITANCE, info->decl->span,
                                  "cyclic inheritance involving class `%s`",
                                  info->decl->as.classDecl.name != NULL
                                      ? info->decl->as.classDecl.name
                                      : "?");
        jaiDiagAddNote(d, "a class may not be, directly or indirectly, its own "
                          "superclass");
        resolverError(rz->r, d);
        assignOwnFieldSlots(info, 0);
        return;
    }

    info->state = 1;
    int base = 0;
    info->super = findClassInfo(rz, info->superSym);
    if (info->super != NULL) {
        assignFieldSlots(rz, info->super);
        /* The recursion may have closed this class out through a cycle. */
        if (info->state == 2) return;
        base = info->super->total;
    }
    assignOwnFieldSlots(info, base);
}

/* ------------------------------------------------------------------ */
/* Statements                                                           */
/* ------------------------------------------------------------------ */

static void resolveBlockContents(Rz *rz, AstNode **stmts, int count) {
    hoistDecls(rz, stmts, count);
    resolveEach(rz, stmts, count);
}

static void resolveBlockNode(Rz *rz, AstNode *node) {
    Scope *scope = jaiScopePush(rz->r, SCOPE_BLOCK);
    scope->owner = node;
    node->as.block.scope = scope;
    resolveBlockContents(rz, node->as.block.stmts, node->as.block.count);
    node->as.block.captureBase = scope->captureBase;
    jaiScopePop(rz->r);
}

static const char *varDeclWord(VarDeclKind kind) {
    switch (kind) {
    case VD_LET:   return "let";
    case VD_VAR:   return "var";
    case VD_CONST: return "const";
    }
    return "let";
}

static void resolveVarDecl(Rz *rz, AstNode *node) {
    resolveTypeExpr(rz, node->as.varDecl.declaredType);

    if (node->as.varDecl.init != NULL) {
        resolveNode(rz, node->as.varDecl.init);
    } else if (node->as.varDecl.kind != VD_VAR) {
        JaiDiag *d = jaiDiagError(E0304_MISSING_INITIALIZER, node->span,
                                  "`%s` declaration requires an initialiser",
                                  varDeclWord(node->as.varDecl.kind));
        jaiDiagAddHelp(d, "give it a value, or declare it with `var`");
        resolverError(rz->r, d);
    }

    /* From here on the name is declared, so a later read is a normal bind. */
    clearPatternPending(rz, node->as.varDecl.pattern);
    resolvePattern(rz, node->as.varDecl.pattern, true, node->as.varDecl.kind, node,
                   false);

    AstNode *pattern = node->as.varDecl.pattern;
    if (pattern != NULL && pattern->kind == AST_PAT_BIND) {
        Symbol *sym = pattern->as.patBind.symbol;
        node->as.varDecl.symbol = sym;
        if (sym != NULL) {
            sym->visibility = node->as.varDecl.visibility;
            sym->isInitialized = node->as.varDecl.init != NULL;
        }
    }
}

/* `stmt` is the span of the whole assignment. Spec §13 underlines the store,
 * not just the name it stores into — `x` on its own is legal, writing to it is
 * what is being rejected — so the caret run covers `x = 2` and the declaration
 * is the secondary label. */
static void resolveAssignTarget(Rz *rz, AstNode *node, bool compound, JaiSpan stmt) {
    if (node == NULL) return;

    if (node->kind != AST_IDENT) {
        /* Member, index and slice targets: only the object side binds here. */
        resolveNode(rz, node);
        return;
    }

    resolveIdent(rz, node, compound);
    Symbol *sym = node->as.ident.symbol;
    if (sym == NULL) return;

    if (!jaiSpanValid(stmt)) stmt = node->span;

    if (sym->kind == SYM_BUILTIN) {
        JaiDiag *d = jaiDiagError(E0301_ASSIGN_TO_IMMUTABLE, stmt,
                                  "cannot assign to built-in `%s`", sym->name);
        jaiDiagAddLabel(d, stmt, "assignment to a built-in name");
        jaiDiagAddHelp(d, "pick a different name for the binding");
        resolverError(rz->r, d);
        return;
    }
    if (sym->mutability == VD_CONST && sym->kind != SYM_FIELD) {
        JaiDiag *d = jaiDiagError(E0305_ASSIGN_TO_CONST, stmt,
                                  "cannot assign to constant `%s`", sym->name);
        jaiDiagAddLabel(d, stmt, "assignment to a constant");
        if (jaiSpanValid(sym->declSpan)) {
            jaiDiagAddLabel(d, sym->declSpan, "`%s` declared here", sym->name);
        }
        jaiDiagAddHelp(d, "declare `%s` with `var` if it has to change", sym->name);
        resolverError(rz->r, d);
    } else if (sym->mutability == VD_LET) {
        JaiDiag *d = jaiDiagError(E0301_ASSIGN_TO_IMMUTABLE, stmt,
                                  "cannot assign to immutable binding `%s`", sym->name);
        jaiDiagAddLabel(d, stmt, "assignment to immutable binding");
        if (jaiSpanValid(sym->declSpan)) {
            jaiDiagAddLabel(d, sym->declSpan, "`%s` declared immutable here", sym->name);
        }
        jaiDiagAddHelp(d, "change the declaration to `var %s`", sym->name);
        resolverError(rz->r, d);
    }
}

static void resolveLoop(Rz *rz, AstNode *node) {
    Resolver *r = rz->r;
    resolveNode(rz, node->as.loop.cond);          /* NULL for `loop` */

    LoopCtx lc;
    lc.enclosing = rz->loops;
    lc.label = node->as.loop.label;
    lc.fn = r->currentFn;
    rz->loops = &lc;

    Scope *scope = jaiScopePush(r, SCOPE_LOOP);
    scope->label = node->as.loop.label;
    scope->owner = node;
    if (r->currentFn != NULL) r->currentFn->loopDepth++;

    resolveNode(rz, node->as.loop.body);

    node->as.loop.captureBase = scope->captureBase;
    if (r->currentFn != NULL) r->currentFn->loopDepth--;
    jaiScopePop(r);
    rz->loops = lc.enclosing;
}

static void resolveFor(Rz *rz, AstNode *node) {
    Resolver *r = rz->r;
    /* The iterable is evaluated in the enclosing scope, before the loop
     * variable exists. */
    resolveNode(rz, node->as.forLoop.iterable);

    LoopCtx lc;
    lc.enclosing = rz->loops;
    lc.label = node->as.forLoop.label;
    lc.fn = r->currentFn;
    rz->loops = &lc;

    Scope *scope = jaiScopePush(r, SCOPE_LOOP);
    scope->label = node->as.forLoop.label;
    scope->owner = node;
    if (r->currentFn != NULL) r->currentFn->loopDepth++;

    resolvePattern(rz, node->as.forLoop.pattern, true, VD_LET, node, false);
    if (node->as.forLoop.pattern != NULL &&
        node->as.forLoop.pattern->kind == AST_PAT_BIND) {
        node->as.forLoop.iterSymbol = node->as.forLoop.pattern->as.patBind.symbol;
    }

    resolveNode(rz, node->as.forLoop.body);

    node->as.forLoop.captureBase = scope->captureBase;
    if (r->currentFn != NULL) r->currentFn->loopDepth--;
    jaiScopePop(r);
    rz->loops = lc.enclosing;
}

static void resolveJump(Rz *rz, AstNode *node, bool isBreak) {
    const char *label = node->as.jump.label;

    if (rz->loops == NULL) {
        JaiDiag *d = jaiDiagError(isBreak ? E0203_BREAK_OUTSIDE_LOOP
                                          : E0204_CONTINUE_OUTSIDE_LOOP,
                                  node->span, "`%s` outside of a loop",
                                  isBreak ? "break" : "continue");
        resolverError(rz->r, d);
        return;
    }
    if (label == NULL) return;

    for (LoopCtx *l = rz->loops; l != NULL; l = l->enclosing) {
        if (l->label != NULL && nameEq(l->label, label)) return;
    }
    JaiDiag *d = jaiDiagError(E0202_UNDEFINED_LABEL, node->span,
                              "undefined loop label `%s`", label);
    jaiDiagAddHelp(d, "label the target loop, e.g. `%s: while ...`", label);
    resolverError(rz->r, d);
}

static void resolveTry(Rz *rz, AstNode *node) {
    Resolver *r = rz->r;
    resolveNode(rz, node->as.tryStmt.body);

    for (int i = 0; i < node->as.tryStmt.catchCount; i++) {
        AstCatch *c = &node->as.tryStmt.catches[i];
        Scope *scope = jaiScopePush(r, SCOPE_BLOCK);
        scope->owner = node;
        for (int j = 0; j < c->typeCount; j++) {
            (void)lookupTypeName(rz, c->types[j]);
            resolveTypeExpr(rz, c->types[j]);
        }
        if (c->name != NULL) {
            Symbol *sym = jaiScopeDeclare(r, c->name, SYM_LOCAL, c->span, VD_LET);
            if (sym != NULL) {
                sym->decl = node;
                sym->isInitialized = true;
            }
            c->symbol = sym;
        }
        resolveNode(rz, c->body);
        jaiScopePop(r);
    }

    resolveNode(rz, node->as.tryStmt.finallyBlock);
}

static void resolveMatch(Rz *rz, AstNode *node) {
    Resolver *r = rz->r;
    resolveNode(rz, node->as.match.subject);

    for (int i = 0; i < node->as.match.armCount; i++) {
        AstMatchArm *arm = &node->as.match.arms[i];
        Scope *scope = jaiScopePush(r, SCOPE_BLOCK);
        scope->owner = node;
        resolvePattern(rz, arm->pattern, true, VD_LET, node, false);
        resolveNode(rz, arm->guard);
        resolveNode(rz, arm->body);
        jaiScopePop(r);
    }
}

static void resolveComprehension(Rz *rz, AstNode *node) {
    Resolver *r = rz->r;
    Scope *scope = jaiScopePush(r, SCOPE_BLOCK);
    scope->owner = node;

    for (int i = 0; i < node->as.comp.clauseCount; i++) {
        AstCompClause *clause = &node->as.comp.clauses[i];
        resolveNode(rz, clause->iterable);
        resolvePattern(rz, clause->pattern, true, VD_LET, node, false);
        resolveEach(rz, clause->conditions, clause->conditionCount);
    }
    resolveNode(rz, node->as.comp.keyExpr);
    resolveNode(rz, node->as.comp.element);

    jaiScopePop(r);
}

static void resolveExport(Rz *rz, AstNode *node) {
    for (int i = 0; i < node->as.exportDecl.count; i++) {
        const char *name = node->as.exportDecl.names[i];
        if (name == NULL) continue;
        Symbol *sym = NULL;
        for (Scope *s = rz->r->current; s != NULL && sym == NULL; s = s->parent) {
            sym = jaiScopeLookupLocal(s, name);
        }
        if (sym == NULL) {
            JaiDiag *d = jaiDiagError(E0200_UNDEFINED_NAME, node->span,
                                      "cannot export `%s`: no such declaration in this "
                                      "module", name);
            const char *suggestion = suggestName(rz->r, name);
            if (suggestion != NULL) jaiDiagAddHelp(d, "did you mean `%s`?", suggestion);
            resolverError(rz->r, d);
            continue;
        }
        sym->visibility = AST_VIS_PUBLIC;
        sym->isUsed = true;
    }
}

/* Was `sym` bound by an import? The C front end resolves imports at run time
 * (spec §8), so such a name has no declaration here and its kind says nothing
 * about whether it denotes a type. */
static bool isImportBinding(const Symbol *sym) {
    return sym != NULL && sym->decl != NULL &&
           (sym->decl->kind == AST_IMPORT || sym->decl->kind == AST_FROM_IMPORT);
}

/* Class and trait scopes hold members, which a bare name never reaches, so
 * resolveName walks past them — and past the generic parameters declared
 * alongside. A type argument does need to see them. */
static const Symbol *lookupThroughClassScopes(Rz *rz, const char *name) {
    if (name == NULL) return NULL;
    for (Scope *s = rz->r->current; s != NULL; s = s->parent) {
        Symbol *sym = jaiScopeLookupLocal(s, name);
        if (sym != NULL) return sym;
    }
    return jaiResolverFindBuiltin(name);
}

/* Could `n` be a type argument and nothing else? Only names are accepted, and
 * only names that denote types: a generic parameter, a declared or imported
 * class/trait/enum/alias, or a builtin type spelling. `xs[i]` and `xs[0]` fail
 * here, which is what keeps a real subscript a subscript. */
static bool typeArgumentLike(Rz *rz, const AstNode *n, int depth) {
    if (n == NULL || depth > 8) return false;
    switch (n->kind) {
    case AST_IDENT: {
        static const char *const kTypeWords[] = {
            "any", "void", "null", "bool", "int", "float", "str", "bytes",
            "range", "never", "list", "set", "dict", "tuple",
        };
        const char *name = n->as.ident.name;
        if (name == NULL) return false;
        for (size_t i = 0; i < sizeof kTypeWords / sizeof kTypeWords[0]; i++)
            if (nameEq(name, kTypeWords[i])) return true;
        const Symbol *sym = lookupThroughClassScopes(rz, name);
        if (sym == NULL) return false;
        switch (sym->kind) {
        case SYM_CLASS: case SYM_TRAIT: case SYM_ENUM:
        case SYM_TYPE_ALIAS: case SYM_GENERIC_PARAM:
            return true;
        default:
            return isImportBinding(sym);
        }
    }
    case AST_INDEX:      /* list[T], tuple[K, V] */
        return typeArgumentLike(rz, n->as.index.object, depth + 1) &&
               typeArgumentLike(rz, n->as.index.index, depth + 1);
    case AST_TUPLE_LIT:  /* the parser's shape for `Pair[A, B]` */
        for (int i = 0; i < n->as.sequence.count; i++)
            if (!typeArgumentLike(rz, n->as.sequence.items[i], depth + 1)) return false;
        return n->as.sequence.count > 0;
    default:
        return false;
    }
}

/* Does this AST_INDEX actually denote `Type[args]` rather than `value[i]`?
 * The object must already be resolved. Only a type-denoting symbol qualifies;
 * `xs[0]` on a list variable stays an index. An imported name is the hard case:
 * nothing here knows whether it names a generic class, so the brackets decide —
 * a subscript whose every element denotes a type is a type application. */
static bool isGenericInstantiation(Rz *rz, const AstNode *node) {
    const AstNode *object = node->as.index.object;
    if (object == NULL) return false;

    /* `m.Box[int]` after `import m`: a module member is resolved by name at
     * run time, so nothing here knows whether it is a generic class. The
     * brackets decide, exactly as they do for a name imported directly —
     * otherwise the qualified spelling of a call the unqualified spelling
     * accepts fails with "'class' value is not indexable". */
    if (object->kind == AST_MEMBER) {
        const AstNode *base = object->as.member.object;
        if (base == NULL || base->kind != AST_IDENT) return false;
        return isImportBinding(base->as.ident.symbol) &&
               typeArgumentLike(rz, node->as.index.index, 0);
    }

    if (object->kind != AST_IDENT) return false;

    const Symbol *sym = object->as.ident.symbol;
    if (sym == NULL) return false;

    switch (sym->kind) {
    case SYM_CLASS:
    case SYM_TRAIT:
    case SYM_ENUM:
    case SYM_TYPE_ALIAS:
    case SYM_GENERIC_PARAM:
        return true;
    default:
        return isImportBinding(sym) &&
               typeArgumentLike(rz, node->as.index.index, 0);
    }
}

/* ------------------------------------------------------------------ */
/* The walk                                                             */
/* ------------------------------------------------------------------ */

static void resolveNode(Rz *rz, AstNode *node) {
    if (node == NULL) return;
    Resolver *r = rz->r;

    switch (node->kind) {
    case AST_INT_LIT:
    case AST_FLOAT_LIT:
    case AST_STR_LIT:
    case AST_BOOL_LIT:
    case AST_NULL_LIT:
    case AST_MODULE_DECL:
        break;

    case AST_FSTRING:
        resolveEach(rz, node->as.fstring.parts, node->as.fstring.partCount);
        break;

    case AST_IDENT:
        resolveIdent(rz, node, true);
        break;

    case AST_SELF:
        if (!rz->selfOk) {
            /* A nested function has no receiver in slot 0, but the enclosing
             * method's `self` is an ordinary `let` binding and closures capture
             * those (spec §6). Rewriting the node into the name it already is
             * hands it to the upvalue machinery instead of inventing a second
             * one; only a `self` with no enclosing method is an error. */
            if (jaiScopeLookup(r, kSelfName) != NULL) {
                node->kind = AST_IDENT;
                memset(&node->as, 0, sizeof node->as);
                node->as.ident.name = kSelfName;
                resolveIdent(rz, node, true);
                break;
            }
            JaiDiag *d = jaiDiagError(E0206_SELF_OUTSIDE_METHOD, node->span,
                                      "`self` is only valid inside a method");
            resolverError(r, d);
        }
        break;

    case AST_SUPER:
        if (!rz->superOk) {
            JaiDiag *d = jaiDiagError(E0207_SUPER_OUTSIDE_SUBCLASS, node->span,
                                      "`super` is only valid inside a method of a class "
                                      "that has a superclass");
            resolverError(r, d);
        }
        break;

    case AST_LIST_LIT:
    case AST_SET_LIT:
    case AST_TUPLE_LIT:
        resolveEach(rz, node->as.sequence.items, node->as.sequence.count);
        break;

    case AST_DICT_LIT:
        for (int i = 0; i < node->as.dict.count; i++) {
            if (node->as.dict.keys != NULL) resolveNode(rz, node->as.dict.keys[i]);
            if (node->as.dict.values != NULL) resolveNode(rz, node->as.dict.values[i]);
        }
        break;

    case AST_UNARY:
        resolveNode(rz, node->as.unary.operand);
        break;

    case AST_BINARY:
    case AST_LOGICAL:
        resolveNode(rz, node->as.binary.left);
        resolveNode(rz, node->as.binary.right);
        break;

    case AST_COMPARE_CHAIN:
        resolveEach(rz, node->as.chain.operands, node->as.chain.opCount + 1);
        break;

    case AST_TERNARY:
        resolveNode(rz, node->as.ternary.cond);
        resolveNode(rz, node->as.ternary.thenExpr);
        resolveNode(rz, node->as.ternary.elseExpr);
        break;

    case AST_COALESCE:
        resolveNode(rz, node->as.coalesce.left);
        resolveNode(rz, node->as.coalesce.right);
        break;

    case AST_CALL:
        resolveNode(rz, node->as.call.callee);
        for (int i = 0; i < node->as.call.argCount; i++) {
            resolveNode(rz, node->as.call.args[i].value);
        }
        break;

    case AST_INDEX:
        resolveNode(rz, node->as.index.object);
        /* `Box[int](0)` and `Pair[A, B](a, b)` parse as an index because the
         * parser cannot know whether `Box` names a generic type — only the
         * symbol table can. Now that the object is resolved, a class, trait,
         * enum, or alias receiver means the brackets held TYPE arguments, so
         * they must not be resolved as value expressions. Generics are erased
         * at run time (spec §6.1), so the arguments carry no runtime meaning
         * and the checker reads them straight off the tree. */
        node->as.index.typeArgs = isGenericInstantiation(rz, node);
        if (!node->as.index.typeArgs) {
            resolveNode(rz, node->as.index.index);
        }
        break;

    case AST_SLICE:
        resolveNode(rz, node->as.slice.object);
        resolveNode(rz, node->as.slice.start);
        resolveNode(rz, node->as.slice.stop);
        resolveNode(rz, node->as.slice.step);
        break;

    case AST_MEMBER:
    case AST_OPT_MEMBER:
        /* The member name is checked against the object's type later. */
        resolveNode(rz, node->as.member.object);
        break;

    case AST_LAMBDA:
    case AST_ANON_FN:
        resolveFunction(rz, node, NULL);
        break;

    case AST_COMPREHENSION:
        resolveComprehension(rz, node);
        break;

    case AST_RANGE:
        resolveNode(rz, node->as.range.start);
        resolveNode(rz, node->as.range.stop);
        break;

    case AST_IF_EXPR:
    case AST_IF:
        resolveNode(rz, node->as.conditional.cond);
        resolveNode(rz, node->as.conditional.thenBranch);
        resolveNode(rz, node->as.conditional.elseBranch);
        break;

    case AST_MATCH:
    case AST_MATCH_EXPR:
        resolveMatch(rz, node);
        break;

    case AST_CAST:
        resolveNode(rz, node->as.cast.operand);
        resolveTypeExpr(rz, node->as.cast.target);
        break;

    case AST_YIELD:
        if (!inFunctionBody(rz)) {
            JaiDiag *d = jaiDiagError(E0210_YIELD_OUTSIDE_GENERATOR, node->span,
                                      "`yield` is only valid inside a function");
            resolverError(r, d);
        } else if (r->currentFn != NULL) {
            r->currentFn->isGenerator = true;
        }
        resolveNode(rz, node->as.wrap.operand);
        break;

    case AST_AWAIT:
    case AST_THROW_EXPR:
        resolveNode(rz, node->as.wrap.operand);
        break;

    case AST_BLOCK:
        resolveBlockNode(rz, node);
        break;

    case AST_PROGRAM:
        /* Only jaiResolveProgram opens the module scope. */
        resolveBlockContents(rz, node->as.block.stmts, node->as.block.count);
        break;

    case AST_EXPR_STMT:
        resolveNode(rz, node->as.exprStmt.expr);
        break;

    case AST_VAR_DECL:
        resolveVarDecl(rz, node);
        break;

    case AST_ASSIGN:
        resolveNode(rz, node->as.assign.value);
        resolveAssignTarget(rz, node->as.assign.target, node->as.assign.isCompound,
                            node->span);
        break;

    case AST_WHILE:
    case AST_LOOP:
        resolveLoop(rz, node);
        break;

    case AST_FOR:
        resolveFor(rz, node);
        break;

    case AST_BREAK:
        resolveJump(rz, node, true);
        break;

    case AST_CONTINUE:
        resolveJump(rz, node, false);
        break;

    case AST_RETURN:
        if (!inFunctionBody(rz)) {
            JaiDiag *d = jaiDiagError(E0205_RETURN_OUTSIDE_FUNCTION, node->span,
                                      "`return` outside of a function");
            resolverError(r, d);
        }
        resolveNode(rz, node->as.ret.value);
        break;

    case AST_THROW:
        resolveNode(rz, node->as.ret.value);
        break;

    case AST_TRY:
        resolveTry(rz, node);
        break;

    case AST_DEFER:
        if (!inFunctionBody(rz)) {
            JaiDiag *d = jaiDiagError(E0211_DEFER_OUTSIDE_FUNCTION, node->span,
                                      "`defer` is only valid inside a function");
            resolverError(r, d);
        } else if (r->currentFn != NULL) {
            r->currentFn->hasDefer = true;
        }
        resolveNode(rz, node->as.defer.body);
        break;

    case AST_ASSERT:
        resolveNode(rz, node->as.assertStmt.cond);
        resolveNode(rz, node->as.assertStmt.message);
        break;

    case AST_FN_DECL:
        if (node->as.fn.symbol == NULL) {
            node->as.fn.symbol = hoistNamed(rz, node->as.fn.name, bindingKind(rz),
                                            node->span, node, node->as.fn.visibility);
        }
        resolveFunction(rz, node, NULL);
        break;

    case AST_CLASS_DECL:
        resolveClassDecl(rz, node);
        break;

    case AST_TRAIT_DECL:
        resolveTraitDecl(rz, node);
        break;

    case AST_ENUM_DECL:
        resolveEnumDecl(rz, node);
        break;

    case AST_TYPE_DECL:
        resolveTypeExpr(rz, node->as.typeDecl.aliased);
        break;

    case AST_IMPORT:
    case AST_FROM_IMPORT:
        break;                      /* bound by the hoist pass */

    case AST_EXPORT:
        resolveExport(rz, node);
        break;

    case AST_PAT_WILDCARD:
    case AST_PAT_BIND:
    case AST_PAT_LITERAL:
    case AST_PAT_RANGE:
    case AST_PAT_TUPLE:
    case AST_PAT_LIST:
    case AST_PAT_CLASS:
    case AST_PAT_ENUM:
    case AST_PAT_OR:
        /* A pattern reached through the generic walk is a read, not a binding. */
        resolvePattern(rz, node, false, VD_LET, NULL, false);
        break;

    case AST_KIND_COUNT:
        break;
    }
}

/* ------------------------------------------------------------------ */
/* Entry point                                                          */
/* ------------------------------------------------------------------ */

bool jaiResolveProgram(Resolver *r, AstNode *program) {
    if (r == NULL || program == NULL) return false;
    if (program->kind != AST_PROGRAM && program->kind != AST_BLOCK) {
        JaiDiag *d = jaiDiagError(E0902_INTERNAL_ERROR, program->span,
                                  "the resolver expects a program node, got %s",
                                  jaiAstKindName(program->kind));
        resolverError(r, d);
        return false;
    }

    Rz rz;
    memset(&rz, 0, sizeof rz);
    rz.r = r;
    JAI_VEC_INIT(&rz.classList);

    /* The module body is a function too (spec/BYTECODE.md §5): it has a frame,
     * so blocks at top level still allocate slots. Its top-level names are
     * globals, not slots. */
    FunctionScope *moduleFn = JAI_ARENA_NEW(&r->arena, FunctionScope);
    moduleFn->decl = program;
    moduleFn->nextSlot = 1;
    moduleFn->localCount = 1;

    FunctionScope *savedFn = r->currentFn;
    Scope *savedScope = r->current;
    r->currentFn = moduleFn;
    r->current = NULL;

    Scope *scope = jaiScopePush(r, SCOPE_MODULE);
    scope->owner = program;
    program->as.block.scope = scope;

    resolveBlockContents(&rz, program->as.block.stmts, program->as.block.count);

    /* Field slots need every class in the program, including those whose
     * superclass was declared later. */
    for (int i = 0; i < rz.classList.count; i++) {
        assignFieldSlots(&rz, rz.classList.data[i]);
    }

    jaiScopePop(r);
    JAI_VEC_FREE(ClassInfo *, &rz.classList);

    r->currentFn = savedFn;
    r->current = savedScope;
    return r->errorCount == 0;
}
