/* check.c — the checker's per-run state and the machinery its four sibling
 * files share.
 *
 * The checker runs after the resolver and before code generation. Three things
 * leave it for the phases downstream:
 *
 *   - every expression node carries a JaiType in AstNode.type
 *   - every any->T boundary is wrapped in an AST_CAST whose target is already
 *     resolved, so the code generator emits OP_TYPE_GUARD without re-deriving
 *     anything
 *   - every class/trait/enum has a TypeDecl with fields laid out parents-first
 *     and slots assigned
 *
 * Narrowing is deliberately small: `x is C`, `x != null`, and the early-return
 * form. Anything more ambitious needs a real dataflow pass, and a checker that
 * is occasionally clever is worse than one that is predictably simple.
 *
 * The per-run bookkeeping (declaration registry, scoped type names, narrowing
 * stack) is one object rather than being hung off Checker because
 * jaiTypeDeclFind takes no context and must answer from anywhere, including
 * the code generator.
 *
 * The walk itself is next door — check_expr.c, check_stmt.c, check_decl.c and
 * check_fold.c — over the interface in check_internal.h.
 */
#include "check_internal.h"


/* ------------------------------------------------------------------ */
/* Per-run state                                                        */
/* ------------------------------------------------------------------ */

JaiCheckState gJaiCheck;

/* ------------------------------------------------------------------ */
/* Small utilities                                                      */
/* ------------------------------------------------------------------ */

void jaiChkCountError(Checker *c) {
    if (c != NULL) c->errorCount++;
}

/* jaiTypeToString hands back a buffer that the next call overwrites, so a
 * diagnostic naming two types must copy the first one out immediately. */
const char *jaiChkRenderType(JaiType *t, char *buf, size_t size) {
    const char *s = (t == NULL) ? NULL : jaiTypeToString(t);
    if (s == NULL) s = "<unknown>";
    size_t len = strlen(s);
    if (len >= size) len = size - 1;
    memcpy(buf, s, len);
    buf[len] = '\0';
    return buf;
}

/* Statically unknown. `any` by construction, and an unsubstituted generic
 * parameter because generics are erased (spec §6.1): inside `fn f[T](x: T)`
 * nothing is known about `T` beyond its name, so every operation on it is
 * checked at run time exactly as it would be on `any`. */
bool jaiChkIsAny(JaiType *t) {
    return t == NULL || t->kind == TY_ANY || t->kind == TY_GENERIC_PARAM;
}

/* `void` is its own singleton; compare by pointer, but stay correct if the
 * type universe ever collapses it onto null. */
bool jaiChkIsVoid(JaiType *t) {
    return t != NULL && t == gTypes.tVoid && gTypes.tVoid != gTypes.tNull;
}

bool jaiChkIsNever(JaiType *t) { return t != NULL && t->kind == TY_NEVER; }

JaiType *jaiChkOrAny(JaiType *t) { return t == NULL ? gTypes.tAny : t; }

bool jaiChkSameName(const char *a, const char *b) {
    if (a == b) return true;
    if (a == NULL || b == NULL) return false;
    return strcmp(a, b) == 0;
}

JaiType **jaiChkTypeArray(Checker *c, int count) {
    if (count <= 0) return NULL;
    return (JaiType **)jaiArenaAllocZeroed(&c->ast->arena,
                                           sizeof(JaiType *) * (size_t)count);
}

/* "did you mean" uses the one tolerance defined in diag.h, so a suggestion the
 * resolver would refuse is not offered here under a looser rule. */
bool jaiChkCloseEnough(const char *name, const char *candidate, int *best) {
    if (candidate == NULL) return false;
    return jaiNameIsCloser(name, candidate, best);
}

/* ------------------------------------------------------------------ */
/* Declaration registry                                                 */
/* ------------------------------------------------------------------ */

DeclEntry *jaiChkDeclEntry(const TypeDecl *d) {
    for (int i = 0; i < gJaiCheck.decls.count; i++)
        if (gJaiCheck.decls.data[i]->decl == d) return gJaiCheck.decls.data[i];
    return NULL;
}

/* The name an entry answers to in the file being compiled. */
static const char *entryName(const DeclEntry *e) {
    return e->alias != NULL ? e->alias : e->decl->name;
}

TypeDecl *jaiTypeDeclFind(const char *name) {
    if (name == NULL) return NULL;
    for (int i = 0; i < gJaiCheck.decls.count; i++) {
        if (!gJaiCheck.decls.data[i]->visible) continue;
        if (jaiChkSameName(entryName(gJaiCheck.decls.data[i]), name))
            return gJaiCheck.decls.data[i]->decl;
    }
    return NULL;
}

const char *jaiTypeDeclBindingName(const JaiType *t) {
    if (t == NULL) return NULL;
    if (t->kind != TY_CLASS && t->kind != TY_TRAIT && t->kind != TY_ENUM) return NULL;
    for (int i = 0; i < gJaiCheck.decls.count; i++)
        if (gJaiCheck.decls.data[i]->type == t) return entryName(gJaiCheck.decls.data[i]);
    if (t->name == NULL) return NULL;
    /* Not in the registry: a type interned by an earlier compilation in this
     * process, or a builtin error class. A declared name has no dot in it, so
     * the last component is the name it was declared with. */
    const char *dot = strrchr(t->name, '.');
    return dot != NULL && dot[1] != '\0' ? dot + 1 : t->name;
}

/* The registered declaration for an AST node, wherever it was declared. */
TypeDecl *jaiChkDeclForNode(const AstNode *node) {
    if (node == NULL) return NULL;
    for (int i = 0; i < gJaiCheck.decls.count; i++)
        if (gJaiCheck.decls.data[i]->decl->decl == node) return gJaiCheck.decls.data[i]->decl;
    return NULL;
}

/* The class, trait or enum a name denotes, declared here or imported. An
 * imported binding keeps the kind the resolver gave it — it occupies a local or
 * a global slot and the code generator has to go on believing that — so what it
 * names is reached through the declaration attached to the symbol instead. */
TypeDecl *jaiChkIdentTypeDecl(const Symbol *s) {
    if (s == NULL) return NULL;
    if (s->kind == SYM_CLASS || s->kind == SYM_TRAIT || s->kind == SYM_ENUM)
        return jaiTypeDeclFind(s->name);
    if (s->importedDecl == NULL) return NULL;
    switch (s->importedDecl->kind) {
    case AST_CLASS_DECL:
    case AST_TRAIT_DECL:
    case AST_ENUM_DECL:
        return jaiChkDeclForNode(s->importedDecl);
    default:
        return NULL;
    }
}

/* The module a declaration node was parsed from, or NULL for one written in the
 * file being compiled. */
ModuleSig *jaiChkOriginOfNode(const AstNode *node) {
    if (node == NULL) return NULL;
    return jaiModuleSigForFile(node->span.file);
}

void jaiChkForeignBegin(Checker *c, ModuleSig *origin, bool quiet,
                         ForeignCtx *saved) {
    saved->origin = gJaiCheck.foreignOrigin;
    saved->errorCount = -1;
    gJaiCheck.foreignOrigin = origin;
    if (origin == NULL || !quiet) return;
    saved->bag = gDiags;
    saved->errorCount = c != NULL ? c->errorCount : 0;
    jaiDiagInit(&gDiags);
}

void jaiChkForeignEnd(Checker *c, ForeignCtx *saved) {
    if (saved->errorCount >= 0) {
        jaiDiagFree(&gDiags);
        gDiags = saved->bag;
        if (c != NULL) c->errorCount = saved->errorCount;
    }
    gJaiCheck.foreignOrigin = saved->origin;
}

int jaiChkFindFieldIndex(const TypeDecl *d, const char *name) {
    if (d == NULL || d->fields == NULL) return -1;
    for (int i = 0; i < d->fieldCount; i++)
        if (jaiChkSameName(d->fields[i].name, name)) return i;
    return -1;
}

int jaiChkFindMemberIndex(const TypeDecl *d, const char *name) {
    if (d == NULL || d->members == NULL) return -1;
    for (int i = 0; i < d->memberCount; i++)
        if (jaiChkSameName(d->members[i].name, name)) return i;
    return -1;
}

/* A getter and a setter share a name, so member lookup has to say which one it
 * wants; plain lookup prefers the getter because that is the read side. */
int jaiChkFindAccessorIndex(const TypeDecl *d, const char *name, bool wantSetter) {
    if (d == NULL || d->members == NULL) return -1;
    for (int i = 0; i < d->memberCount; i++) {
        if (!jaiChkSameName(d->members[i].name, name)) continue;
        if (wantSetter ? d->members[i].isSetter : !d->members[i].isSetter) return i;
    }
    return -1;
}

const void *jaiTypeDeclFindField(const TypeDecl *d, const char *name) {
    int i = jaiChkFindFieldIndex(d, name);
    return i < 0 ? NULL : (const void *)&d->fields[i];
}

const void *jaiTypeDeclFindMember(const TypeDecl *d, const char *name) {
    int i = jaiChkFindMemberIndex(d, name);
    return i < 0 ? NULL : (const void *)&d->members[i];
}

bool jaiTypeDeclIsSubclassOf(const TypeDecl *sub, const TypeDecl *super) {
    if (sub == NULL || super == NULL) return false;
    for (const TypeDecl *p = sub; p != NULL; p = p->superclass) {
        if (p == super) return true;
        for (int i = 0; i < p->traitCount; i++) {
            if (p->traits == NULL) break;
            if (p->traits[i] == super) return true;
            /* A trait may itself extend traits; those were flattened into the
             * trait's own list when it was laid out. */
            if (jaiTypeDeclIsSubclassOf(p->traits[i], super)) return true;
        }
    }
    return false;
}

/* The class that first declared `name`: the topmost ancestor that carries it.
 * Fields and members are copied downwards during layout, so "has it" walking
 * up the chain identifies the declarer. */
const TypeDecl *jaiChkOwnerOfField(const TypeDecl *d, const char *name) {
    const TypeDecl *owner = d;
    for (const TypeDecl *p = d->superclass; p != NULL; p = p->superclass)
        if (jaiChkFindFieldIndex(p, name) >= 0) owner = p;
    return owner;
}

const TypeDecl *jaiChkOwnerOfMember(const TypeDecl *d, const char *name) {
    const TypeDecl *owner = d;
    for (const TypeDecl *p = d->superclass; p != NULL; p = p->superclass)
        if (jaiChkFindMemberIndex(p, name) >= 0) owner = p;
    return owner;
}

/* True when some link of this class's ancestry reached the unit as a runtime
 * object rather than a declaration — an imported superclass, or a trait whose
 * defaults are equally invisible. Their members exist and this unit cannot
 * enumerate them, so it may not claim a name is missing. */
bool jaiChkInheritsOpaquely(const TypeDecl *d) {
    for (const TypeDecl *p = d; p != NULL; p = p->superclass)
        if (p->opaqueSuper != NULL || p->opaqueTraitCount > 0) return true;
    return false;
}

JaiType *jaiChkDeclType(const TypeDecl *d) {
    DeclEntry *e = jaiChkDeclEntry(d);
    return e == NULL ? gTypes.tAny : e->type;
}

/* The TypeDecl behind a class/trait/enum type, if it has one.
 *
 * Types are interned in a universe that outlives any one AST — a REPL line, a
 * second module — so JaiType.decl can point at a declaration that is gone. The
 * registry is the authority; the pointer is only a fast path. */
TypeDecl *jaiChkTypeDeclOf(JaiType *t) {
    if (t == NULL) return NULL;
    if (t->kind != TY_CLASS && t->kind != TY_TRAIT && t->kind != TY_ENUM) return NULL;
    for (int i = 0; i < gJaiCheck.decls.count; i++) {
        if (gJaiCheck.decls.data[i]->type == t) return gJaiCheck.decls.data[i]->decl;
        if (gJaiCheck.decls.data[i]->decl == t->decl) return t->decl;
    }
    return jaiTypeDeclFind(t->name);
}

/* ------------------------------------------------------------------ */
/* Names that denote types                                              */
/* ------------------------------------------------------------------ */

/* The exception hierarchy of spec §5.4 is part of the language, not of any
 * module, so `catch e: ValueError` must resolve even with no std in sight.
 * They carry no TypeDecl, which is what keeps member access on them dynamic. */
static const char *const kBuiltinClasses[] = {
    "Error", "AssertionError", "ArithmeticError", "DivisionByZeroError",
    "OverflowError", "LookupError", "IndexError", "KeyError", "NameError",
    "TypeError", "ValueError", "ParseError", "AttributeError", "IOError",
    "FileNotFoundError", "PermissionError", "OSError", "RuntimeError",
    "RecursionError", "StopIteration", "ImportError",
};

static JaiType *builtinScalarType(const char *name) {
    if (jaiChkSameName(name, "any"))   return gTypes.tAny;
    if (jaiChkSameName(name, "void"))  return gTypes.tVoid;
    if (jaiChkSameName(name, "null"))  return gTypes.tNull;
    if (jaiChkSameName(name, "bool"))  return gTypes.tBool;
    if (jaiChkSameName(name, "int"))   return gTypes.tInt;
    if (jaiChkSameName(name, "float")) return gTypes.tFloat;
    if (jaiChkSameName(name, "str"))   return gTypes.tStr;
    if (jaiChkSameName(name, "bytes")) return gTypes.tBytes;
    if (jaiChkSameName(name, "range")) return gTypes.tRange;
    if (jaiChkSameName(name, "never")) return gTypes.tNever;
    /* Bare container names are the fully dynamic form of the generic: `list`
     * is `list[any]`. `tuple` has its arity in the type, so "a tuple of any
     * shape" is not in the lattice at all — `any` is the honest answer. */
    if (jaiChkSameName(name, "list"))  return jaiTypeList(gTypes.tAny);
    if (jaiChkSameName(name, "set"))   return jaiTypeSet(gTypes.tAny);
    if (jaiChkSameName(name, "dict"))  return jaiTypeDict(gTypes.tAny, gTypes.tAny);
    if (jaiChkSameName(name, "tuple")) return gTypes.tAny;
    return NULL;
}

static JaiType *builtinClassType(const char *name) {
    for (size_t i = 0; i < sizeof kBuiltinClasses / sizeof kBuiltinClasses[0]; i++)
        if (jaiChkSameName(kBuiltinClasses[i], name))
            return jaiTypeNamed(TY_CLASS, kBuiltinClasses[i], NULL);
    return NULL;
}

void jaiChkPushTypeName(const char *name, JaiType *type) {
    NamedType entry = { name, type };
    JAI_VEC_PUSH(NamedType, &gJaiCheck.names, entry);
}

/* §7.2: `Self` is the implementing type. Static storage, because interned
 * types keep the name pointer rather than copying it. */
const char *jaiChkSelfName = "Self";

/* In a trait, `Self` is not yet any one type: it stands for whichever class
 * ends up implementing the trait, and is substituted away when the trait's
 * members are installed on that class. A generic parameter is exactly that —
 * erased at run time, accepted by every value, and already understood by
 * `jaiTypeSubstitute` — so `Self` is one, spelled with a name no declaration
 * is allowed to introduce (E0114 in the parser). */
static JaiType *selfPlaceholder(void) { return jaiTypeGenericParam(jaiChkSelfName); }

/* Bind `Self` for the body of one declaration. A class or enum knows its own
 * type, so `Self` there is exact; a trait gets the placeholder. */
void jaiChkPushSelf(JaiType *type) {
    jaiChkPushTypeName(jaiChkSelfName, type != NULL ? type : selfPlaceholder());
}

int jaiChkTypeNameMark(void) { return gJaiCheck.names.count; }

void jaiChkTypeNameRestore(int mark) {
    if (mark >= 0 && mark <= gJaiCheck.names.count) gJaiCheck.names.count = mark;
}

static TypeDecl *registerForeign(Checker *c, ModuleSig *origin, AstNode *node,
                                 const char *alias, bool visible);

/* A name written in another module's source. It means what it means *there*:
 * one of that module's own declarations, or one it imported in turn — followed
 * through, because otherwise `fn draw(w: Window)` in `std.gui.canvas` would
 * check nothing, `Deque[T]: Iterable[T]` would carry none of the trait's
 * defaults, and the difference between a local call and a call one module away
 * would be exactly the thing this is here to remove. Never one of *this* file's
 * names: two modules that each declare a `Node` must stay two types. */
static JaiType *lookupForeignTypeName(Checker *c, const char *name) {
    for (int i = 0; i < gJaiCheck.decls.count; i++) {
        const DeclEntry *e = gJaiCheck.decls.data[i];
        if (e->decl->origin != gJaiCheck.foreignOrigin) continue;
        if (jaiChkSameName(e->decl->name, name)) return e->type;
    }

    ModuleSig *owner = NULL;
    AstNode *decl = jaiModuleSigFind(gJaiCheck.foreignOrigin, name, &owner);
    if (decl == NULL) return NULL;
    switch (decl->kind) {
    case AST_CLASS_DECL:
    case AST_TRAIT_DECL:
    case AST_ENUM_DECL:
        break;
    default:
        return NULL;
    }
    /* Invisible: this file did not import it, so it may not be named here — it
     * is registered only so that the module that *did* import it can. */
    TypeDecl *d = registerForeign(c, owner, decl, NULL, false);
    return d != NULL ? jaiChkDeclType(d) : NULL;
}

JaiType *jaiChkLookupTypeName(Checker *c, const char *name) {
    if (name == NULL) return NULL;
    /* Innermost binding wins: a generic parameter shadows a class of the same
     * name, which shadows a module-level alias. */
    for (int i = gJaiCheck.names.count - 1; i >= 0; i--)
        if (jaiChkSameName(gJaiCheck.names.data[i].name, name)) return gJaiCheck.names.data[i].type;

    JaiType *t = builtinScalarType(name);
    if (t != NULL) return t;

    if (gJaiCheck.foreignOrigin != NULL) {
        JaiType *foreign = lookupForeignTypeName(c, name);
        if (foreign != NULL) return foreign;
        JaiType *builtinClass = builtinClassType(name);
        /* Never NULL: an unresolvable name in a foreign signature is a name
         * this file cannot see, not a mistake it may report. */
        return builtinClass != NULL ? builtinClass : gTypes.tAny;
    }

    TypeDecl *d = jaiTypeDeclFind(name);
    if (d != NULL) return jaiChkDeclType(d);

    /* An imported class or a builtin registered by the runtime carries its
     * type on the symbol. */
    if (c != NULL && c->resolver != NULL) {
        Symbol *s = jaiScopeLookup(c->resolver, name);
        if (s == NULL) s = jaiScopeLookupLocal(gJaiCheck.moduleScope, name);
        if (s == NULL) s = jaiResolverFindBuiltin(name);
        if (s != NULL && s->type != NULL &&
            (s->kind == SYM_CLASS || s->kind == SYM_TRAIT || s->kind == SYM_ENUM ||
             s->kind == SYM_TYPE_ALIAS || s->kind == SYM_GENERIC_PARAM))
            return s->type;
        /* A name this module imported: the C front end resolves imports at run
         * time (spec §8), so no declaration for it exists here. It is a real
         * name, not a typo, and gradual typing already has a word for a type
         * whose shape is unknown — `any`. Reporting E0402 would make every
         * `from std.core import Iterable` unusable in an annotation. */
        if (s != NULL && s->decl != NULL &&
            (s->decl->kind == AST_FROM_IMPORT || s->decl->kind == AST_IMPORT))
            return gTypes.tAny;
    }

    return builtinClassType(name);
}

static void suggestTypeName(Checker *c, JaiDiag *d, const char *name) {
    if (jaiChkSameName(name, jaiChkSelfName)) {
        /* Reaching here means `Self` was not pushed, so there is no enclosing
         * declaration for it to name (spec §7.2). */
        jaiDiagAddHelp(d, "`Self` names the implementing type and is only in "
                          "scope inside a class, trait or enum body");
        return;
    }
    static const char *const kScalars[] = { "any", "void", "null", "bool", "int",
                                            "float", "str", "bytes", "range",
                                            "list", "dict", "set", "tuple" };
    const char *best = NULL;
    int bestDistance = JAI_SUGGEST_NO_MATCH;
    for (size_t i = 0; i < sizeof kScalars / sizeof kScalars[0]; i++)
        if (jaiChkCloseEnough(name, kScalars[i], &bestDistance)) best = kScalars[i];
    for (int i = 0; i < gJaiCheck.decls.count; i++)
        if (jaiChkCloseEnough(name, gJaiCheck.decls.data[i]->decl->name, &bestDistance))
            best = gJaiCheck.decls.data[i]->decl->name;
    for (int i = gJaiCheck.names.count - 1; i >= 0; i--)
        if (jaiChkCloseEnough(name, gJaiCheck.names.data[i].name, &bestDistance))
            best = gJaiCheck.names.data[i].name;
    (void)c;
    if (best != NULL) jaiDiagAddHelp(d, "did you mean `%s`?", best);
}

JaiType *jaiChkResolveAstType(Checker *c, AstType *t);

/* list/dict/set/tuple are spelled as generic applications but are structural,
 * not declared, so their arity is checked here. */
static JaiType *resolveBuiltinGeneric(Checker *c, AstType *t, bool *handled) {
    const char *name = t->name;
    int want = -1;
    if (jaiChkSameName(name, "list") || jaiChkSameName(name, "set")) want = 1;
    else if (jaiChkSameName(name, "dict")) want = 2;
    else if (jaiChkSameName(name, "tuple")) want = -2;
    else { *handled = false; return NULL; }

    *handled = true;
    int given = t->argCount;
    if (want >= 0 && given != want) {
        JaiDiag *d = ERR(c, E0407_GENERIC_ARITY, t->span,
                         "`%s` takes %d type argument%s but %d %s given",
                         name, want, want == 1 ? "" : "s", given,
                         given == 1 ? "was" : "were");
        jaiDiagAddHelp(d, "write `%s[%s]`", name,
                       want == 1 ? "T" : "K, V");
        return gTypes.tAny;
    }

    if (want == 1) {
        JaiType *elem = jaiChkResolveAstType(c, t->args[0]);
        return jaiChkSameName(name, "list") ? jaiTypeList(elem) : jaiTypeSet(elem);
    }
    if (want == 2)
        return jaiTypeDict(jaiChkResolveAstType(c, t->args[0]),
                           jaiChkResolveAstType(c, t->args[1]));

    JaiType **members = jaiChkTypeArray(c, given);
    for (int i = 0; i < given; i++) members[i] = jaiChkResolveAstType(c, t->args[i]);
    return jaiTypeTuple(members, given);
}

static int genericArityOf(const TypeDecl *d) {
    if (d == NULL || d->decl == NULL) return 0;
    switch (d->decl->kind) {
    case AST_CLASS_DECL: return d->decl->as.classDecl.genericCount;
    case AST_TRAIT_DECL: return d->decl->as.traitDecl.genericCount;
    case AST_ENUM_DECL:  return d->decl->as.enumDecl.genericCount;
    default:             return 0;
    }
}

JaiType *jaiChkResolveAstType(Checker *c, AstType *t) {
    if (t == NULL) return gTypes.tAny;
    if (t->resolved != NULL) return t->resolved;

    JaiType *result = gTypes.tAny;
    switch (t->kind) {
    case TYPE_INFER:
        result = gTypes.tAny;
        break;

    case TYPE_NAME: {
        JaiType *named = jaiChkLookupTypeName(c, t->name);
        if (named == NULL) {
            JaiDiag *d = ERR(c, E0402_UNKNOWN_TYPE, t->span,
                             "unknown type `%s`", t->name == NULL ? "?" : t->name);
            if (t->name != NULL) suggestTypeName(c, d, t->name);
            named = gTypes.tAny;
        }
        result = named;
        break;
    }

    case TYPE_GENERIC: {
        bool handled = false;
        JaiType *builtin = resolveBuiltinGeneric(c, t, &handled);
        if (handled) { result = builtin; break; }

        /* Generics are erased, so a declared generic type checks its arity and
         * then stands for the bare declaration. */
        for (int i = 0; i < t->argCount; i++) (void)jaiChkResolveAstType(c, t->args[i]);
        JaiType *base = jaiChkLookupTypeName(c, t->name);
        if (base == NULL) {
            JaiDiag *d = ERR(c, E0402_UNKNOWN_TYPE, t->span,
                             "unknown type `%s`", t->name == NULL ? "?" : t->name);
            if (t->name != NULL) suggestTypeName(c, d, t->name);
            result = gTypes.tAny;
            break;
        }
        TypeDecl *decl = jaiChkTypeDeclOf(base);
        int want = genericArityOf(decl);
        if (decl != NULL && want != t->argCount) {
            ERR(c, E0407_GENERIC_ARITY, t->span,
                "`%s` takes %d type argument%s but %d %s given",
                t->name, want, want == 1 ? "" : "s", t->argCount,
                t->argCount == 1 ? "was" : "were");
        }
        result = base;
        break;
    }

    case TYPE_OPTIONAL:
        result = jaiTypeOptional(jaiChkResolveAstType(c, t->inner));
        break;

    case TYPE_UNION: {
        JaiType **members = jaiChkTypeArray(c, t->argCount);
        for (int i = 0; i < t->argCount; i++) members[i] = jaiChkResolveAstType(c, t->args[i]);
        result = t->argCount > 0 ? jaiTypeUnion(members, t->argCount) : gTypes.tAny;
        break;
    }

    case TYPE_FN: {
        JaiType **params = jaiChkTypeArray(c, t->argCount);
        for (int i = 0; i < t->argCount; i++) params[i] = jaiChkResolveAstType(c, t->args[i]);
        result = jaiTypeFn(params, t->argCount, jaiChkResolveAstType(c, t->inner), 0);
        break;
    }

    case TYPE_TUPLE: {
        JaiType **members = jaiChkTypeArray(c, t->argCount);
        for (int i = 0; i < t->argCount; i++) members[i] = jaiChkResolveAstType(c, t->args[i]);
        result = jaiTypeTuple(members, t->argCount);
        break;
    }
    }

    t->resolved = jaiChkOrAny(result);
    return t->resolved;
}

/* ------------------------------------------------------------------ */
/* Flow-sensitive narrowing                                             */
/* ------------------------------------------------------------------ */

int jaiChkNarrowMark(void) { return gJaiCheck.narrow.count; }

void jaiChkNarrowRestore(int mark) {
    while (gJaiCheck.narrow.count > mark) {
        NarrowSave s = JAI_VEC_POP(&gJaiCheck.narrow);
        s.sym->type = s.saved;
    }
}

void jaiChkNarrowApply(const NarrowFact *facts, int count, bool invert) {
    for (int i = 0; i < count; i++) {
        Symbol *sym = facts[i].sym;
        if (sym == NULL) continue;
        JaiType *current = sym->type;
        if (current == NULL) continue;
        JaiType *next = jaiTypeNarrow(current, facts[i].by,
                                      invert ? !facts[i].positive : facts[i].positive);
        if (next == NULL || next == current) continue;
        NarrowSave save = { sym, current };
        JAI_VEC_PUSH(NarrowSave, &gJaiCheck.narrow, save);
        sym->type = next;
    }
}

/* The type a binding was declared with, before any live fact rewrote it. A
 * fact only ever replaces an already-saved type, so the oldest save for the
 * symbol holds the declaration. */
JaiType *jaiChkDeclaredType(Symbol *sym) {
    if (sym == NULL) return NULL;
    for (int i = 0; i < gJaiCheck.narrow.count; i++)
        if (gJaiCheck.narrow.data[i].sym == sym) return gJaiCheck.narrow.data[i].saved;
    return sym->type;
}

/* A store to a narrowed binding ends the fact: what it holds afterwards is
 * whatever the declaration admits. The save stack is left untouched so the
 * enclosing jaiChkNarrowRestore still unwinds in order. */
void jaiChkNarrowInvalidate(Symbol *sym) {
    JaiType *declared = jaiChkDeclaredType(sym);
    if (declared != NULL) sym->type = declared;
}

static Symbol *narrowTarget(AstNode *node) {
    if (node == NULL || node->kind != AST_IDENT) return NULL;
    Symbol *s = node->as.ident.symbol;
    if (s == NULL || s->type == NULL) return NULL;
    /* Narrowing a `var` is still sound within a scope: an assignment to it is
     * checked against the declared type and then ends the fact, which is what
     * jaiChkNarrowInvalidate above does. */
    return s;
}

/* The right operand of `is` names a type when it is a bare identifier bound to
 * a class, trait or enum; anything else is an identity comparison. */
static JaiType *typeOperand(Checker *c, AstNode *node) {
    if (node == NULL || node->kind != AST_IDENT) return NULL;
    Symbol *s = node->as.ident.symbol;
    if (s != NULL && (s->kind == SYM_CLASS || s->kind == SYM_TRAIT || s->kind == SYM_ENUM))
        return jaiChkOrAny(s->type != NULL ? s->type : jaiChkLookupTypeName(c, node->as.ident.name));
    /* `str`, `int`, `list`... are bound as builtin conversion functions as well
     * as naming types (spec §9). Comparing a value against a function object is
     * never what `x is str` means, so the type reading wins. */
    if (s != NULL && s->kind != SYM_BUILTIN) return NULL;
    return jaiChkLookupTypeName(c, node->as.ident.name);
}

static bool isNullLiteral(const AstNode *n) {
    return n != NULL && n->kind == AST_NULL_LIT;
}

int jaiChkCollectFacts(Checker *c, AstNode *cond, NarrowFact *out, int max, bool negate);

static int collectComparisonFact(Checker *c, OpKind op, AstNode *lhs, AstNode *rhs,
                                 NarrowFact *out, int max, bool negate) {
    if (max <= 0) return 0;

    if (op == OPK_IS || op == OPK_IS_NOT) {
        /* `null` is a unit type with one value (§2.1), so `x is null` is the
         * null identity test and narrows exactly as `x == null` does. It is
         * not an AST_IDENT, so typeOperand cannot see it. */
        JaiType *target = isNullLiteral(rhs) ? gTypes.tNull : typeOperand(c, rhs);
        Symbol *sym = narrowTarget(lhs);
        if (target == NULL || sym == NULL) return 0;
        out[0].sym = sym;
        out[0].by = target;
        out[0].positive = (op == OPK_IS) != negate;
        return 1;
    }

    if (op == OPK_EQ || op == OPK_NE) {
        Symbol *sym = NULL;
        if (isNullLiteral(rhs)) sym = narrowTarget(lhs);
        else if (isNullLiteral(lhs)) sym = narrowTarget(rhs);
        if (sym == NULL) return 0;
        out[0].sym = sym;
        out[0].by = gTypes.tNull;
        out[0].positive = (op == OPK_EQ) != negate;
        return 1;
    }

    return 0;
}

int jaiChkCollectFacts(Checker *c, AstNode *cond, NarrowFact *out, int max, bool negate) {
    if (cond == NULL || max <= 0) return 0;

    switch (cond->kind) {
    case AST_UNARY:
        if (cond->as.unary.op == OPK_NOT)
            return jaiChkCollectFacts(c, cond->as.unary.operand, out, max, !negate);
        return 0;

    case AST_LOGICAL: {
        /* `a and b` gives both facts on the true edge; `not (a or b)` gives
         * both on the false edge. Every other combination is a disjunction,
         * where no single fact is guaranteed. */
        bool conjunction = (cond->as.binary.op == OPK_AND) != negate;
        if (!conjunction) return 0;
        int n = jaiChkCollectFacts(c, cond->as.binary.left, out, max, negate);
        n += jaiChkCollectFacts(c, cond->as.binary.right, out + n, max - n, negate);
        return n;
    }

    case AST_CALL: {
        /* `isinstance(x, T)` is the call form of `x is T` and the one the
         * library uses on unions; it narrows the same way. */
        AstNode *callee = cond->as.call.callee;
        if (callee == NULL || callee->kind != AST_IDENT ||
            !jaiChkSameName(callee->as.ident.name, "isinstance") ||
            cond->as.call.argCount != 2)
            return 0;
        Symbol *sym = narrowTarget(cond->as.call.args[0].value);
        JaiType *target = typeOperand(c, cond->as.call.args[1].value);
        if (sym == NULL || target == NULL) return 0;
        out[0].sym = sym;
        out[0].by = target;
        out[0].positive = !negate;
        return 1;
    }

    case AST_BINARY:
        return collectComparisonFact(c, cond->as.binary.op, cond->as.binary.left,
                                     cond->as.binary.right, out, max, negate);

    case AST_COMPARE_CHAIN: {
        if (cond->as.chain.opCount != 1 || cond->as.chain.operands == NULL ||
            cond->as.chain.ops == NULL)
            return 0;
        return collectComparisonFact(c, cond->as.chain.ops[0], cond->as.chain.operands[0],
                                     cond->as.chain.operands[1], out, max, negate);
    }

    default:
        return 0;
    }
}
/* ------------------------------------------------------------------ */
/* int literals in float context                                        */
/* ------------------------------------------------------------------ */

/* One half of §2.2's implicit conversion: the half that costs nothing at run
 * time, because an int literal can simply be respelled as a float literal.
 *
 * This is not the assignment rule, and it must not be read as one. `let n = 1;
 * let x: float = n` is accepted and yields 1.0 — jaiChkRequireAssignable
 * consults jaiTypeWidenTarget and calls jaiChkWidenToFloat, which falls back
 * to an AST_CAST{widen} for every shape this function refuses. What this
 * function decides is only whether a conversion node is needed, never whether
 * a conversion happens. */
bool jaiChkRetypeIntLiteral(AstNode *node) {
    if (node == NULL) return false;
    switch (node->kind) {
    case AST_INT_LIT: {
        double value = (double)node->as.intLit;
        node->kind = AST_FLOAT_LIT;
        node->as.floatLit = value;
        node->type = gTypes.tFloat;
        return true;
    }
    case AST_UNARY:
        if (node->as.unary.op != OPK_NEG && node->as.unary.op != OPK_POS) return false;
        if (!jaiChkRetypeIntLiteral(node->as.unary.operand)) return false;
        node->type = gTypes.tFloat;
        return true;
    default:
        return false;
    }
}

static bool unionWantsFloat(JaiType *t) {
    if (t == NULL) return false;
    if (t->kind == TY_FLOAT) return true;
    if (t->kind != TY_UNION) return false;
    bool hasFloat = false;
    for (int i = 0; i < t->argCount; i++) {
        if (t->args[i]->kind == TY_INT) return false;   /* int is a better fit */
        if (t->args[i]->kind == TY_FLOAT) hasFloat = true;
    }
    return hasFloat;
}

/* Push an expected type down into a literal so that the check below sees the
 * already-converted form. Only literal shapes are rewritten; a variable keeps
 * its type and produces E0411 as it should. */
void jaiChkApplyContext(Checker *c, AstNode *node, JaiType *expected) {
    if (node == NULL || expected == NULL) return;

    if (unionWantsFloat(expected)) {
        if (jaiChkRetypeIntLiteral(node)) return;
        /* Arithmetic over literals is still a literal expression as far as the
         * conversion is concerned: `let x: float = 1 + 2`. */
        if (node->kind == AST_BINARY) {
            switch (node->as.binary.op) {
            case OPK_ADD: case OPK_SUB: case OPK_MUL:
            case OPK_DIV: case OPK_FLOORDIV: case OPK_MOD: case OPK_POW:
                jaiChkApplyContext(c, node->as.binary.left, expected);
                jaiChkApplyContext(c, node->as.binary.right, expected);
                break;
            default:
                break;
            }
        } else if (node->kind == AST_TERNARY) {
            jaiChkApplyContext(c, node->as.ternary.thenExpr, expected);
            jaiChkApplyContext(c, node->as.ternary.elseExpr, expected);
        } else if (node->kind == AST_IF_EXPR) {
            jaiChkApplyContext(c, node->as.conditional.thenBranch, expected);
            jaiChkApplyContext(c, node->as.conditional.elseBranch, expected);
        }
        return;
    }

    /* Container literals additionally adopt the expected type itself, not just
     * the element context. Without this an empty `[]` has no items to recurse
     * into and infers list[any], and a populated one infers the JOIN of its
     * elements — so `let s: list[Shape] = [Circle(), Square()]` infers
     * list[Circle | Square] and is then rejected against its own annotation.
     * The literal checkers read this back and verify each element against the
     * target instead of against its siblings. */
    switch (expected->kind) {
    case TY_LIST:
    case TY_SET:
        /* A comprehension builds a fresh container, so it takes the annotation
         * the same way a literal does. */
        if (node->kind == AST_COMPREHENSION) { node->type = expected; break; }
        if ((node->kind == AST_LIST_LIT || node->kind == AST_SET_LIT) &&
            expected->argCount == 1) {
            for (int i = 0; i < node->as.sequence.count; i++)
                jaiChkApplyContext(c, node->as.sequence.items[i], expected->args[0]);
            node->type = expected;
        }
        break;

    case TY_DICT:
        if (node->kind == AST_COMPREHENSION) { node->type = expected; break; }
        if (node->kind == AST_DICT_LIT && expected->argCount == 2) {
            for (int i = 0; i < node->as.dict.count; i++) {
                jaiChkApplyContext(c, node->as.dict.keys[i], expected->args[0]);
                jaiChkApplyContext(c, node->as.dict.values[i], expected->args[1]);
            }
            node->type = expected;
        }
        break;

    case TY_TUPLE:
        if (node->kind == AST_TUPLE_LIT && node->as.sequence.count == expected->argCount) {
            for (int i = 0; i < node->as.sequence.count; i++)
                jaiChkApplyContext(c, node->as.sequence.items[i], expected->args[i]);
            /* Stamped for the same reason a list literal is: each member is
             * then checked against the position it is going into, so an int
             * widens into a `float` member and a mismatch names the member
             * rather than failing the whole tuple against its annotation. */
            node->type = expected;
        }
        break;

    default:
        break;
    }
}

/* ------------------------------------------------------------------ */
/* Guard insertion and the assignability report                         */
/* ------------------------------------------------------------------ */

/* Does `t` mention an unsubstituted generic parameter anywhere? Such a type
 * has no runtime shape after erasure, so it cannot be guarded against. */
static bool mentionsGenericParam(JaiType *t) {
    if (t == NULL) return false;
    if (t->kind == TY_GENERIC_PARAM) return true;
    for (int i = 0; i < t->argCount; i++)
        if (mentionsGenericParam(t->args[i])) return true;
    return t->kind == TY_FN && mentionsGenericParam(t->ret);
}

/* Wrap `node` in an AST_CAST in place: the original is copied into a fresh
 * node that becomes the cast's operand. Rewriting in place means no caller has
 * to hand over the slot that points at the node. */
static void wrapInCast(Checker *c, AstNode *node, JaiType *to, bool widen) {
    if (node == NULL || to == NULL || node->kind == AST_CAST) return;
    /* Erased: OP_TYPE_GUARD would look up a class named `T`. */
    if (!widen && mentionsGenericParam(to)) return;

    AstNode *inner = jaiAstNew(c->ast, node->kind, node->span);
    *inner = *node;

    AstType *target = jaiAstTypeNew(c->ast, TYPE_NAME, node->span);
    const char *rendered = jaiTypeToString(to);
    target->name = rendered == NULL
                       ? NULL
                       : jaiArenaMemdup(&c->ast->arena, rendered, strlen(rendered));
    target->resolved = to;

    node->kind = AST_CAST;
    memset(&node->as, 0, sizeof node->as);
    node->as.cast.operand = inner;
    node->as.cast.target = target;
    node->as.cast.widen = widen;
    node->type = to;
}

static void wrapInGuard(Checker *c, AstNode *node, JaiType *to) {
    wrapInCast(c, node, to, false);
}

void jaiChkWidenToFloat(Checker *c, AstNode *node, JaiType *to) {
    /* An int literal becomes a float literal instead, which costs nothing at
     * run time; anything else gets the conversion node §2.2 calls for, so the
     * code generator emits one OP_TO_FLOAT rather than leaving the VM to
     * discover the mismatch on every evaluation. */
    if (jaiChkRetypeIntLiteral(node)) return;
    wrapInCast(c, node, to, true);
}

/* Check that `from` fits `to`, inserting the any->T guard when one is needed.
 * `what` reads into the message: "cannot assign `str` to `int`" style. */
bool jaiChkRequireAssignable(Checker *c, AstNode *node, JaiType *from, JaiType *to,
                              JaiDiagCode code, const char *what) {
    if (to == NULL || from == NULL) return true;
    if (jaiChkIsNever(from)) return true;

    /* §2.2's widening, before the general rule, because it is the one
     * assignment whose value changes on the way in and so the one that has to
     * put a conversion into the tree. */
    JaiType *widen = jaiTypeWidenTarget(from, to);
    if (widen != NULL) {
        if (node != NULL) jaiChkWidenToFloat(c, node, widen);
        return true;
    }

    bool needsGuard = false;
    if (jaiTypeAssignable(from, to, &needsGuard)) {
        if (needsGuard && node != NULL) wrapInGuard(c, node, to);
        return true;
    }

    char want[TYPE_BUF], got[TYPE_BUF];
    jaiChkRenderType(to, want, sizeof want);
    jaiChkRenderType(from, got, sizeof got);
    /* float to int is the direction that loses information, so it is the
     * direction that stays an error, and it keeps E0411 to itself. */
    bool narrowing = to->kind == TY_INT && from->kind == TY_FLOAT;
    JaiDiag *d = ERR(c, narrowing ? E0411_INT_FLOAT_MIX : code,
                     node != NULL ? node->span : JAI_SPAN_NONE,
                     "%s: expected `%s`, found `%s`", what, want, got);
    if (narrowing)
        jaiDiagAddHelp(d, "float does not narrow to int, because it would lose "
                          "the fraction; write `int(%s)` to truncate",
                       jaiChkSubjectOf(node, "x"));
    else if (jaiTypeIsOptional(to) && from->kind == TY_NULL)
        jaiDiagAddHelp(d, "`null` is only assignable to an optional type");
    return false;
}

/* ------------------------------------------------------------------ */
/* Program-level passes                                                 */
/* ------------------------------------------------------------------ */

/* ------------------------------------------------------------------ */
/* Imported declarations                                                */
/* ------------------------------------------------------------------ */

/* Register one class, trait or enum read out of `origin`.
 *
 * Every top-level type of an imported module gets an entry, whether or not this
 * file named it, because a declaration it *did* name can mention a sibling in
 * an annotation and that sibling has to mean the sibling. Only what this file
 * imported is `visible`, i.e. reachable by a bare name here.
 *
 * The type is interned under its qualified name, so `std.gui.Window` and a
 * `Window` declared in this file are two types and can never be confused for
 * one another — and neither can two modules that both declare a `Node`. */
static TypeDecl *registerForeign(Checker *c, ModuleSig *origin, AstNode *node,
                                 const char *alias, bool visible) {
    const char *name;
    TypeKind kind;
    switch (node->kind) {
    case AST_CLASS_DECL: name = node->as.classDecl.name; kind = TY_CLASS; break;
    case AST_TRAIT_DECL: name = node->as.traitDecl.name; kind = TY_TRAIT; break;
    case AST_ENUM_DECL:  name = node->as.enumDecl.name;  kind = TY_ENUM;  break;
    default: return NULL;
    }
    if (name == NULL) return NULL;

    /* A name this file already declares wins; the resolver has its own word for
     * a genuine clash (E0302) and this is not the place for a second one. */
    if (visible && jaiTypeDeclFind(alias != NULL ? alias : name) != NULL)
        visible = false;

    for (int i = 0; i < gJaiCheck.decls.count; i++) {
        DeclEntry *e = gJaiCheck.decls.data[i];
        if (e->decl->decl != node) continue;
        if (visible && !e->visible) {
            e->visible = true;
            e->alias = alias;
        }
        return e->decl;
    }

    TypeDecl *d = JAI_ARENA_NEW(&c->ast->arena, TypeDecl);
    d->name = name;
    d->span = node->span;
    d->decl = node;
    d->origin = origin;
    d->isTrait = node->kind == AST_TRAIT_DECL;
    d->isEnum = node->kind == AST_ENUM_DECL;
    d->isAbstract = node->kind == AST_CLASS_DECL && node->as.classDecl.isAbstract;

    DeclEntry *entry = JAI_ARENA_NEW(&c->ast->arena, DeclEntry);
    entry->decl = d;
    entry->type = jaiTypeNamed(kind, jaiModuleSigQualify(origin, name), d);
    entry->visible = visible;
    entry->alias = alias;
    JAI_VEC_PUSH(DeclEntry *, &gJaiCheck.decls, entry);
    return d;
}

static void registerModuleTypes(Checker *c, ModuleSig *sig) {
    int count = jaiModuleSigTypeCount(sig);
    for (int i = 0; i < count; i++)
        (void)registerForeign(c, sig, jaiModuleSigTypeAt(sig, i), NULL, false);
}

static ModuleSig *loadSigFor(Checker *c, const char *dotted, JaiSpan at) {
    ModuleSig *sig = jaiModuleSigLoad(dotted, at.file);
    if (sig != NULL) registerModuleTypes(c, sig);
    return sig;
}

/* Attach what `m` declares for `name` to the binding `from m import name`
 * made. A name the module does not declare — one its body defines at run time,
 * or one this loader cannot follow — is left exactly as it was: `any`, which is
 * what every imported name used to be. */
static void bindImportedName(Checker *c, ModuleSig *sig, const char *name,
                             const char *alias, Symbol *sym) {
    if (sig == NULL || name == NULL) return;
    ModuleSig *owner = sig;
    AstNode *decl = jaiModuleSigFind(sig, name, &owner);
    if (decl == NULL) return;
    if (owner != sig) registerModuleTypes(c, owner);

    switch (decl->kind) {
    case AST_FN_DECL:
        if (sym != NULL) {
            sym->importedDecl = decl;
            if (sym->type == NULL) sym->type = jaiChkForeignFunctionType(c, owner, decl);
        }
        break;

    case AST_CLASS_DECL:
    case AST_TRAIT_DECL:
    case AST_ENUM_DECL: {
        TypeDecl *d = registerForeign(c, owner, decl, alias, true);
        if (d == NULL) break;
        if (sym != NULL) {
            sym->importedDecl = decl;
            if (sym->type == NULL) sym->type = jaiChkDeclType(d);
        }
        break;
    }

    case AST_TYPE_DECL: {
        ForeignCtx saved;
        jaiChkForeignBegin(c, owner, true, &saved);
        int mark = jaiChkTypeNameMark();
        jaiChkPushGenerics(c, decl->as.typeDecl.generics, decl->as.typeDecl.genericCount);
        JaiType *target = jaiChkOrAny(jaiChkResolveAstType(c, decl->as.typeDecl.aliased));
        jaiChkTypeNameRestore(mark);
        jaiChkForeignEnd(c, &saved);
        /* Pushed before this file's own aliases, so a local one still wins. */
        jaiChkPushTypeName(alias != NULL ? alias : name, target);
        if (sym != NULL && sym->type == NULL) sym->type = target;
        break;
    }

    default:
        break;
    }
}

static bool visitImports(AstNode *node, void *userData) {
    Checker *c = (Checker *)userData;
    switch (node->kind) {
    case AST_IMPORT: {
        ModuleSig *sig = loadSigFor(c, node->as.import.path, node->span);
        Symbol *sym = node->as.import.symbol;
        if (sig != NULL && sym != NULL) {
            ImportedModule bound = { sym, sig };
            JAI_VEC_PUSH(ImportedModule, &gJaiCheck.modules, bound);
        }
        break;
    }
    case AST_FROM_IMPORT: {
        /* `from m import *` binds no symbols here at all — the resolver defers
         * every unknown name to the runtime — so there is nothing to attach. */
        if (node->as.fromImport.isWildcard) break;
        ModuleSig *sig = loadSigFor(c, node->as.fromImport.path, node->span);
        if (sig == NULL) break;
        for (int i = 0; i < node->as.fromImport.itemCount; i++) {
            AstImportItem *item = &node->as.fromImport.items[i];
            bindImportedName(c, sig, item->name, item->alias, item->symbol);
        }
        break;
    }
    default:
        break;
    }
    return true;
}

static bool visitRegister(AstNode *node, void *userData) {
    Checker *c = (Checker *)userData;
    switch (node->kind) {
    case AST_CLASS_DECL:
    case AST_TRAIT_DECL:
    case AST_ENUM_DECL:
        jaiChkRegisterDecl(c, node);
        break;
    default:
        break;
    }
    return true;
}

static bool visitAlias(AstNode *node, void *userData) {
    Checker *c = (Checker *)userData;
    if (node->kind != AST_TYPE_DECL) return true;

    int mark = jaiChkTypeNameMark();
    jaiChkPushGenerics(c, node->as.typeDecl.generics, node->as.typeDecl.genericCount);
    JaiType *target = jaiChkResolveAstType(c, node->as.typeDecl.aliased);
    jaiChkTypeNameRestore(mark);

    jaiChkPushTypeName(node->as.typeDecl.name, jaiChkOrAny(target));
    if (node->as.typeDecl.symbol != NULL) node->as.typeDecl.symbol->type = jaiChkOrAny(target);
    node->type = jaiChkOrAny(target);
    return true;
}

/* Signatures are typed before any body is checked so that mutually recursive
 * functions and forward references see a real type rather than `any`. */
static bool visitSignature(AstNode *node, void *userData) {
    Checker *c = (Checker *)userData;
    if (node->kind != AST_FN_DECL) return true;
    Symbol *sym = node->as.fn.symbol;
    if (sym != NULL && sym->type == NULL) sym->type = jaiChkFunctionType(c, node, false);
    if (node->type == NULL && sym != NULL) node->type = sym->type;
    return true;
}

void jaiCheckerInit(Checker *c, Resolver *r, AstContext *ast) {
    if (c == NULL) return;
    memset(c, 0, sizeof *c);
    c->resolver = r;
    c->ast = ast;
    c->foldConstants = true;

    gJaiCheck.decls.count = 0;
    gJaiCheck.names.count = 0;
    gJaiCheck.narrow.count = 0;
    gJaiCheck.modules.count = 0;
    gJaiCheck.foreignOrigin = NULL;
    gJaiCheck.constRequired = false;
}

void jaiCheckerFree(Checker *c) {
    /* TypeDecls and their arrays live in the AST arena, so only the tracking
     * vectors are owned here — but the interned types outlive that arena, and
     * JaiType.decl points into it. A second module compiled in the same process
     * would otherwise find `Iterator` carrying a back-pointer to the freed
     * declaration of the first and read it. Drop the pointers; jaiChkTypeDeclOf falls
     * back on the registry, which is the authority anyway. */
    for (int i = 0; i < gJaiCheck.decls.count; i++)
        if (gJaiCheck.decls.data[i]->type != NULL) gJaiCheck.decls.data[i]->type->decl = NULL;
    JAI_VEC_FREE(DeclEntry *, &gJaiCheck.decls);
    JAI_VEC_FREE(NamedType, &gJaiCheck.names);
    JAI_VEC_FREE(NarrowSave, &gJaiCheck.narrow);
    JAI_VEC_FREE(ImportedModule, &gJaiCheck.modules);
    gJaiCheck.foreignOrigin = NULL;
    gJaiCheck.constRequired = false;
    if (c != NULL) {
        c->currentReturnType = NULL;
        c->currentClass = NULL;
    }
}

bool jaiCheckProgram(Checker *c, AstNode *program) {
    if (c == NULL || program == NULL || c->ast == NULL) return false;

    gJaiCheck.moduleScope = (program->kind == AST_PROGRAM || program->kind == AST_BLOCK)
                             ? program->as.block.scope
                             : NULL;

    /* This file's own declarations first, so that a name it declares always
     * wins over one it imports, and only then what the imports bring in. */
    jaiAstWalk(program, visitRegister, NULL, c);
    jaiAstWalk(program, visitImports, NULL, c);
    jaiAstWalk(program, visitAlias, NULL, c);
    /* Everything registered, imported declarations included. Laying those out
     * on demand does not work: `Iter` is assignable to `Iterator` only if its
     * trait list says so, and half the questions the checker asks about a type
     * — is it iterable, does it implement that trait, is it callable — are
     * asked in code with no declaration in hand to lay out first. A laid-out
     * declaration answers them all; an empty one silently answers "no".
     *
     * The loop re-reads `count` because a layout can discover a type in a
     * third module and register it, which is also why entries are held by
     * pointer: the array under them moves. */
    for (int i = 0; i < gJaiCheck.decls.count; i++) jaiChkLayoutDecl(c, gJaiCheck.decls.data[i]);
    jaiAstWalk(program, visitSignature, NULL, c);

    jaiChkStmt(c, program);
    program->type = gTypes.tVoid;

    jaiChkNarrowRestore(0);
    /* After everything, not during: folding rewrites nodes and a cast replaces
     * one in place, so a type recorded as it was assigned can be a type the
     * finished tree does not hold. */
    return c->errorCount == 0;
}
