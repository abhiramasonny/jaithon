/* check_decl.c — turning a class, trait or enum declaration into a TypeDecl,
 * and checking its body.
 *
 * One job in three stages, which is why they are one file: layout assigns
 * field slots parents-first and installs inherited and trait-default members;
 * conformance checks that every requirement a trait names is implemented,
 * substituting the trait's generic arguments and `Self` before comparing
 * signatures; and the body pass checks each method against the type the first
 * two stages built. The middle stage cannot be separated from either
 * neighbour — installing a trait default *is* substitution, and a requirement
 * is compared against a member the layout put there.
 */
#include "check_internal.h"

/* ------------------------------------------------------------------ */
/* Class, trait, and enum layout                                        */
/* ------------------------------------------------------------------ */

TypeDecl *jaiChkRegisterDecl(Checker *c, AstNode *node) {
    const char *name;
    TypeKind kind;
    AstVisibility visibility;
    switch (node->kind) {
    case AST_CLASS_DECL:
        name = node->as.classDecl.name;
        kind = TY_CLASS;
        visibility = node->as.classDecl.visibility;
        break;
    case AST_TRAIT_DECL:
        name = node->as.traitDecl.name;
        kind = TY_TRAIT;
        visibility = node->as.traitDecl.visibility;
        break;
    case AST_ENUM_DECL:
        name = node->as.enumDecl.name;
        kind = TY_ENUM;
        visibility = node->as.enumDecl.visibility;
        break;
    default:
        return NULL;
    }
    (void)visibility;
    if (name == NULL) return NULL;

    TypeDecl *existing = jaiTypeDeclFind(name);
    if (existing != NULL) return existing;   /* E0302 is the resolver's to report */

    TypeDecl *d = JAI_ARENA_NEW(&c->ast->arena, TypeDecl);
    d->name = name;
    d->span = node->span;
    d->decl = node;
    d->isTrait = node->kind == AST_TRAIT_DECL;
    d->isEnum = node->kind == AST_ENUM_DECL;
    d->isAbstract = node->kind == AST_CLASS_DECL && node->as.classDecl.isAbstract;

    DeclEntry *entry = JAI_ARENA_NEW(&c->ast->arena, DeclEntry);
    entry->decl = d;
    entry->type = jaiTypeNamed(kind, name, d);
    entry->visible = true;
    JAI_VEC_PUSH(DeclEntry *, &gJaiCheck.decls, entry);

    Symbol *sym = NULL;
    if (node->kind == AST_CLASS_DECL) sym = node->as.classDecl.symbol;
    else if (node->kind == AST_TRAIT_DECL) sym = node->as.traitDecl.symbol;
    else sym = node->as.enumDecl.symbol;
    if (sym != NULL) sym->type = entry->type;
    node->type = entry->type;
    return d;
}

static void addField(Checker *c, TypeDecl *d, const char *name, JaiType *type,
                     AstVisibility visibility, bool isStatic, bool isLet,
                     bool hasDefault, int *nextSlot, JaiSpan span) {
    if (jaiChkFindFieldIndex(d, name) >= 0) {
        JaiDiag *diag = ERR(c, E0709_DUPLICATE_MEMBER, span,
                            "`%s` is already declared in `%s`", name, d->name);
        jaiDiagAddLabel(diag, d->span, "in this declaration");
        return;
    }
    int i = d->fieldCount;
    d->fields[i].name = name;
    d->fields[i].type = jaiChkOrAny(type);
    d->fields[i].visibility = visibility;
    d->fields[i].isStatic = isStatic;
    d->fields[i].isLet = isLet;
    d->fields[i].hasDefault = hasDefault;
    d->fields[i].slot = isStatic ? -1 : (*nextSlot)++;
    d->fieldCount = i + 1;
}

static void layoutClassFields(Checker *c, TypeDecl *d, TypeDecl *parent,
                              AstField *fields, int count) {
    int parentCount = parent != NULL ? parent->fieldCount : 0;
    int total = parentCount + count;
    if (total <= 0) return;

    d->fields = jaiArenaAllocZeroed(&c->ast->arena, sizeof *d->fields * (size_t)total);
    int nextSlot = 0;
    for (int i = 0; i < parentCount; i++) {
        d->fields[i] = parent->fields[i];
        if (!d->fields[i].isStatic && d->fields[i].slot >= nextSlot)
            nextSlot = d->fields[i].slot + 1;
    }
    d->fieldCount = parentCount;

    for (int i = 0; i < count; i++) {
        AstField *f = &fields[i];
        addField(c, d, f->name, jaiChkResolveAstType(c, f->type), f->visibility, f->isStatic,
                 f->isLet, f->defaultValue != NULL, &nextSlot, f->span);
    }
}

static void memberTypeOf(Checker *c, AstNode *fn, bool isGetter, bool isSetter,
                         JaiType **outType) {
    if (isGetter) {
        *outType = jaiChkDeclaredReturnType(c, fn, true);
        return;
    }
    if (isSetter) {
        *outType = fn->as.fn.paramCount > 0 && fn->as.fn.params[0].type != NULL
                       ? jaiChkResolveAstType(c, fn->as.fn.params[0].type)
                       : gTypes.tAny;
        return;
    }
    *outType = jaiChkFunctionType(c, fn, true);
}

/* An override may widen what it accepts and narrow what it returns. */
static bool typeFits(JaiType *sub, JaiType *super) {
    if (sub == NULL || super == NULL) return true;
    if (jaiChkIsAny(sub) || jaiChkIsAny(super)) return true;
    return jaiTypeEquals(sub, super) || jaiTypeIsSubtype(sub, super);
}

static bool signatureCompatible(JaiType *sub, JaiType *super, const char **why) {
    if (sub == NULL || super == NULL) return true;
    if (jaiChkIsAny(sub) || jaiChkIsAny(super)) return true;
    if (sub->kind != TY_FN || super->kind != TY_FN) {
        if (typeFits(sub, super)) return true;
        *why = "the types differ";
        return false;
    }
    if (sub->argCount != super->argCount) {
        *why = "the parameter counts differ";
        return false;
    }
    for (int i = 0; i < sub->argCount; i++)
        if (!typeFits(super->args[i], sub->args[i])) {
            *why = "a parameter type is not general enough";
            return false;
        }
    if (!typeFits(sub->ret, super->ret)) {
        *why = "the return type is not compatible";
        return false;
    }
    return true;
}

static int addMember(Checker *c, TypeDecl *d, const char *name, JaiType *type,
                     AstVisibility visibility, bool isStatic, bool isGetter,
                     bool isSetter, bool isAbstract, AstNode *decl) {
    int i = d->memberCount;
    d->members[i].name = name;
    d->members[i].type = jaiChkOrAny(type);
    d->members[i].visibility = visibility;
    d->members[i].isStatic = isStatic;
    d->members[i].isGetter = isGetter;
    d->members[i].isSetter = isSetter;
    d->members[i].isAbstract = isAbstract;
    d->members[i].decl = decl;
    d->memberCount = i + 1;
    (void)c;
    return i;
}

typedef struct {
    int parentEnd;    /* members [0, parentEnd) came from the superclass */
    int traitEnd;     /* members [parentEnd, traitEnd) came from a trait   */
} MemberRegions;

/* Install one declared member, either replacing an inherited entry (an
 * override, checked for compatibility) or appending a new one. */
static void installMember(Checker *c, TypeDecl *d, AstNode *fn, bool isGetter,
                          bool isSetter, const MemberRegions *regions) {
    const char *name = fn->as.fn.name;
    if (name == NULL) return;

    JaiType *type = NULL;
    memberTypeOf(c, fn, isGetter, isSetter, &type);
    bool isStatic = fn->as.fn.isStatic;

    if (fn->as.fn.symbol != NULL && fn->as.fn.symbol->type == NULL)
        fn->as.fn.symbol->type = isGetter || isSetter ? type : jaiChkFunctionType(c, fn, true);
    fn->type = type;

    /* self is mandatory on an instance method and forbidden on a static one
     * (spec §7.1). Properties take no self at all. */
    if (!isGetter && !isSetter) {
        int skip = jaiChkSelfSkipOf(fn);
        if (!isStatic && skip == 0)
            ERR(c, E0703_MISSING_SELF, fn->span,
                "instance method `%s` must take `self` as its first parameter", name);
        if (isStatic && fn->as.fn.paramCount > 0 &&
            jaiChkSameName(fn->as.fn.params[0].name, "self"))
            ERR(c, E0707_STATIC_WITH_SELF, fn->span,
                "`static fn %s` must not take `self`", name);
    }

    int existing = jaiChkFindAccessorIndex(d, name, isSetter);
    if (existing < 0 && !isSetter) existing = jaiChkFindMemberIndex(d, name);

    if (existing >= regions->traitEnd) {
        JaiDiag *diag = ERR(c, E0709_DUPLICATE_MEMBER, fn->span,
                            "`%s` is already declared in `%s`", name, d->name);
        jaiDiagAddLabel(diag, d->span, "in this declaration");
        return;
    }
    if (existing < 0 && jaiChkFindFieldIndex(d, name) >= 0) {
        ERR(c, E0709_DUPLICATE_MEMBER, fn->span,
            "`%s` is already declared as a field of `%s`", name, d->name);
        return;
    }

    /* `init` is not an override: a constructor belongs to its own class and is
     * reached through `super(...)`, never through dynamic dispatch, so a
     * subclass is free to take different arguments. */
    if (existing >= 0 && existing < regions->parentEnd && !jaiChkIsInitMethod(fn)) {
        const char *why = NULL;
        if (!signatureCompatible(type, d->members[existing].type, &why)) {
            char sub[TYPE_BUF], super[TYPE_BUF];
            jaiChkRenderType(type, sub, sizeof sub);
            jaiChkRenderType(d->members[existing].type, super, sizeof super);
            JaiDiag *diag = ERR(c, E0704_INCOMPATIBLE_OVERRIDE, fn->span,
                                "`%s` overrides an incompatible signature: %s",
                                name, why);
            jaiDiagAddNote(diag, "overriding `%s` with `%s`", super, sub);
            if (d->members[existing].decl != NULL)
                jaiDiagAddLabel(diag, d->members[existing].decl->span,
                                "the overridden declaration");
            jaiDiagAddHelp(diag, "parameters may only widen and the return type "
                                 "may only narrow");
        }
    }

    /* A trait body declares an interface, so its members are public whether or
     * not `pub` is written: §7.1's "private unless pub" governs classes, and
     * §7.2 makes traits types, through which a private member could never be
     * reached. The spec's own `trait Shape` example relies on this. */
    AstVisibility vis = d->isTrait ? AST_VIS_PUBLIC : fn->as.fn.visibility;

    if (existing >= 0) {
        d->members[existing].type = jaiChkOrAny(type);
        d->members[existing].visibility = vis;
        d->members[existing].isStatic = isStatic;
        d->members[existing].isGetter = isGetter;
        d->members[existing].isSetter = isSetter;
        d->members[existing].isAbstract = fn->as.fn.body == NULL;
        d->members[existing].decl = fn;
        return;
    }

    addMember(c, d, name, type, vis, isStatic, isGetter, isSetter,
              fn->as.fn.body == NULL, fn);
}

/* ------------------------------------------------------------------ */
/* Trait generic arguments                                              */
/* ------------------------------------------------------------------ */

/* `class MapIter[T, U]: Iterator[U]` on `trait Iterator[T]` binds the trait's
 * own T to U, so its `fn next(self) -> T?` requires `-> U?` here. */
struct TraitBinding {
    const char **names;
    JaiType    **values;
    int          count;
    /* What `Self` becomes for this implementer, or NULL to leave it abstract —
     * which is what a trait extending another trait wants, since the
     * implementing type is still unknown one level up. */
    JaiType     *selfType;
};

static TraitBinding traitBindingOf(Checker *c, TypeDecl *trait, AstType *ref,
                                   JaiType *selfType) {
    TraitBinding b = { NULL, NULL, 0, selfType };
    if (trait == NULL || trait->decl == NULL ||
        trait->decl->kind != AST_TRAIT_DECL || ref == NULL ||
        ref->kind != TYPE_GENERIC)
        return b;
    int count = trait->decl->as.traitDecl.genericCount;
    /* A wrong argument count is already E0407 from jaiChkResolveAstType; binding a
     * prefix of it would only add a second, more confusing diagnostic. */
    if (count <= 0 || count != ref->argCount) return b;

    b.names = jaiArenaAllocZeroed(&c->ast->arena,
                                  sizeof(const char *) * (size_t)count);
    b.values = jaiChkTypeArray(c, count);
    b.count = count;
    for (int i = 0; i < count; i++) {
        b.names[i] = trait->decl->as.traitDecl.generics[i].name;
        b.values[i] = jaiChkResolveAstType(c, ref->args[i]);
    }
    return b;
}

static JaiType *applyTraitBinding(const TraitBinding *b, JaiType *t) {
    if (b == NULL) return t;
    /* Self first: a requirement written `-> Self` becomes the implementer's own
     * type, so an implementation may promise to return itself (spec §7.2). */
    if (b->selfType != NULL) {
        JaiType *value = b->selfType;
        t = jaiTypeSubstitute(t, &jaiChkSelfName, &value, 1);
    }
    if (b->count == 0) return t;
    return jaiTypeSubstitute(t, b->names, b->values, b->count);
}

/* "Iterator[(int, T)]" — the trait as this implementer wrote it, so a note can
 * point back at the declaration the requirement came from. */
static const char *renderTraitRef(const TypeDecl *trait, const TraitBinding *b,
                                  char *buf, size_t size) {
    size_t n = (size_t)snprintf(buf, size, "%s", trait->name);
    if (b == NULL || b->count == 0) return buf;
    for (int i = 0; i < b->count && n < size; i++) {
        char arg[TYPE_BUF];
        jaiChkRenderType(b->values[i], arg, sizeof arg);
        n += (size_t)snprintf(buf + n, size - n, "%s%s",
                              i == 0 ? "[" : ", ", arg);
    }
    if (n < size) snprintf(buf + n, size - n, "]");
    return buf;
}

/* The declaration the author has to write, not merely its type. Spec §7 makes
 * `self` an explicit first parameter, so a help quoting the bare `fn` type
 * would send the reader straight into E0703. Parameter names come from the
 * trait, but the types come from `required`, which has the implementer's trait
 * arguments already substituted in. */
static const char *renderRequiredMethod(const char *name, const AstNode *traitDecl,
                                        const JaiType *required, char *buf,
                                        size_t size) {
    const AstParam *params = NULL;
    int paramCount = 0, skip = 0;
    bool isStatic = false;
    if (traitDecl != NULL && traitDecl->kind == AST_FN_DECL) {
        params = traitDecl->as.fn.params;
        paramCount = traitDecl->as.fn.paramCount;
        skip = jaiChkSelfSkipOf(traitDecl);
        isStatic = traitDecl->as.fn.isStatic;
    }

    size_t n = 0;
    /* snprintf returns what it *would* have written, so every step clamps. */
#define APPEND(...)                                                     \
    do {                                                                \
        if (n >= size) break;                                           \
        int w = snprintf(buf + n, size - n, __VA_ARGS__);               \
        n = (w < 0 || (size_t)w >= size - n) ? size : n + (size_t)w;    \
    } while (0)

    APPEND("%sfn %s(", isStatic ? "static " : "", name);
    bool wroteParam = false;
    if (skip > 0) { APPEND("self"); wroteParam = true; }

    int argCount = (required != NULL && required->kind == TY_FN) ? required->argCount : 0;
    for (int i = 0; i < argCount; i++) {
        char ty[TYPE_BUF];
        jaiChkRenderType(required->args[i], ty, sizeof ty);
        const char *pname = (params != NULL && skip + i < paramCount)
                                ? params[skip + i].name
                                : NULL;
        if (pname == NULL) pname = "arg";
        APPEND("%s%s: %s", wroteParam ? ", " : "", pname, ty);
        wroteParam = true;
    }

    char ret[TYPE_BUF];
    jaiChkRenderType(required != NULL ? required->ret : NULL, ret, sizeof ret);
    APPEND(") -> %s", ret);
#undef APPEND
    return buf;
}

static void verifyTraits(Checker *c, TypeDecl *d) {
    if (d->isAbstract || d->isTrait) return;

    for (int t = 0; t < d->traitCount; t++) {
        TypeDecl *trait = d->traits == NULL ? NULL : d->traits[t];
        if (trait == NULL) continue;
        const TraitBinding *bind =
            d->traitBindings == NULL ? NULL : &d->traitBindings[t];

        for (int m = 0; m < trait->memberCount; m++) {
            if (!trait->members[m].isAbstract) continue;   /* has a default body */
            const char *name = trait->members[m].name;
            /* The requirement as this implementer wrote it: a generic trait
             * demands its arguments substituted in, not its own parameters. */
            JaiType *required = applyTraitBinding(bind, trait->members[m].type);
            int mi = jaiChkFindAccessorIndex(d, name, trait->members[m].isSetter);
            if (mi < 0) mi = jaiChkFindMemberIndex(d, name);

            if (mi < 0 || d->members[mi].isAbstract) {
                JaiDiag *diag = ERR(c, E0705_TRAIT_NOT_IMPLEMENTED, d->span,
                                    "`%s` does not implement `%s` required by trait `%s`",
                                    d->name, name, trait->name);
                if (trait->members[m].decl != NULL)
                    jaiDiagAddLabel(diag, trait->members[m].decl->span,
                                    "required here");
                char sig[TYPE_BUF * 2];
                jaiDiagAddHelp(diag, "add `%s` to `%s`",
                               renderRequiredMethod(name, trait->members[m].decl,
                                                    required, sig, sizeof sig),
                               d->name);
                continue;
            }

            const char *why = NULL;
            if (!signatureCompatible(d->members[mi].type, required, &why)) {
                char got[TYPE_BUF], want[TYPE_BUF];
                jaiChkRenderType(d->members[mi].type, got, sizeof got);
                jaiChkRenderType(required, want, sizeof want);
                JaiDiag *diag = ERR(c, E0705_TRAIT_NOT_IMPLEMENTED,
                                    d->members[mi].decl != NULL
                                        ? d->members[mi].decl->span
                                        : d->span,
                                    "`%s` implements `%s` as `%s`, but trait "
                                    "`%s` requires `%s`: %s",
                                    d->name, name, got, trait->name, want, why);
                if (trait->members[m].decl != NULL)
                    jaiDiagAddLabel(diag, trait->members[m].decl->span,
                                    "required here");
                /* After substitution the requirement no longer reads like the
                 * trait body, so say where the substituted form came from. */
                if (required != trait->members[m].type) {
                    char declared[TYPE_BUF], ref[TYPE_BUF];
                    jaiChkRenderType(trait->members[m].type, declared, sizeof declared);
                    /* With no trait arguments the only substitution that can
                     * have fired is `Self`, and naming the trait again would
                     * say nothing ("which `Bad` implements as `Doubler`"). */
                    if (bind == NULL || bind->count == 0)
                        jaiDiagAddNote(diag, "`%s` is declared `%s` in `%s`, "
                                             "where `Self` is `%s`",
                                       name, declared, trait->name, d->name);
                    else
                        jaiDiagAddNote(diag, "`%s` is declared `%s` in `%s`, which "
                                             "`%s` implements as `%s`",
                                       name, declared, trait->name, d->name,
                                       renderTraitRef(trait, bind, ref, sizeof ref));
                }
            }
        }
    }
}

/* `outOpaque` receives the name of a trait or class that resolved to a type but
 * to no declaration this unit can see. */
static TypeDecl *resolveDeclReference(Checker *c, AstType *ref, bool wantTrait,
                                      DeclEntry *self, const char **outOpaque) {
    if (outOpaque != NULL) *outOpaque = NULL;
    JaiType *t = jaiChkResolveAstType(c, ref);
    TypeDecl *target = jaiChkTypeDeclOf(t);
    if (target == NULL) {
        /* A trait or class that reached this module through the prelude or an
         * import has a type but no declaration to lay out here. It is opaque,
         * not wrong: its contract was checked where it was declared and is
         * enforced at run time, so `class P: Printable` must be accepted with
         * only spec §9's re-export in scope. */
        bool opaque = t != NULL &&
                      (wantTrait ? t->kind == TY_TRAIT : t->kind == TY_CLASS);
        if (opaque && outOpaque != NULL) *outOpaque = t->name;
        if (!jaiChkIsAny(t) && !opaque)
            ERR(c, E0700_UNKNOWN_CLASS, ref->span, "`%s` is not a %s",
                ref->name != NULL ? ref->name : "this type",
                wantTrait ? "trait" : "class");
        return NULL;
    }
    if (wantTrait != target->isTrait || (!wantTrait && target->isEnum)) {
        ERR(c, E0700_UNKNOWN_CLASS, ref->span, "`%s` is not a %s", target->name,
            wantTrait ? "trait" : "class");
        return NULL;
    }

    DeclEntry *entry = jaiChkDeclEntry(target);
    if (entry == self || (entry != NULL && entry->status == 1)) {
        JaiDiag *d = ERR(c, E0706_CYCLIC_INHERITANCE, ref->span,
                         "`%s` would inherit from itself", target->name);
        jaiDiagAddLabel(d, target->span, "the cycle passes through here");
        return NULL;
    }
    jaiChkLayoutDecl(c, entry);
    return target;
}

static void layoutClass(Checker *c, DeclEntry *entry) {
    TypeDecl *d = entry->decl;
    AstNode *node = d->decl;

    int mark = jaiChkTypeNameMark();
    jaiChkPushGenerics(c, node->as.classDecl.generics, node->as.classDecl.genericCount);
    jaiChkPushSelf(entry->type);

    TypeDecl *parent = NULL;
    const char *opaqueParent = NULL;
    if (node->as.classDecl.superclass != NULL) {
        parent = resolveDeclReference(c, node->as.classDecl.superclass, false,
                                      entry, &opaqueParent);
        /* A named parent that produced no declaration is a parent this unit
         * cannot enumerate, whatever the reason — imported as a runtime object,
         * or already reported as unknown. Either way the members it might carry
         * are not this unit's to deny. */
        if (parent == NULL && opaqueParent == NULL)
            opaqueParent = node->as.classDecl.superclass->name;
    }
    d->superclass = parent;
    d->opaqueSuper = opaqueParent;

    int traitCount = node->as.classDecl.traitCount;
    if (traitCount > 0) {
        d->traits = jaiArenaAllocZeroed(&c->ast->arena,
                                        sizeof(TypeDecl *) * (size_t)traitCount);
        d->traitBindings = jaiArenaAllocZeroed(
            &c->ast->arena, sizeof(TraitBinding) * (size_t)traitCount);
        d->opaqueTraits = jaiArenaAllocZeroed(
            &c->ast->arena, sizeof(const char *) * (size_t)traitCount);
        for (int i = 0; i < traitCount; i++) {
            AstType *ref = node->as.classDecl.traits[i];
            const char *opaque = NULL;
            TypeDecl *trait = resolveDeclReference(c, ref, true, entry, &opaque);
            /* Same rule as the superclass above, and for the same reason: a
             * named trait that produced no declaration is a trait this unit
             * cannot enumerate. Its defaults are members of the class whatever
             * the reason, so they are not this unit's to deny. */
            if (trait == NULL && opaque == NULL) opaque = ref->name;
            if (opaque != NULL) d->opaqueTraits[d->opaqueTraitCount++] = opaque;
            if (trait == NULL) continue;
            d->traitBindings[d->traitCount] =
                traitBindingOf(c, trait, ref, entry->type);
            d->traits[d->traitCount++] = trait;
        }
    }

    layoutClassFields(c, d, parent, node->as.classDecl.fields,
                      node->as.classDecl.fieldCount);

    int parentMembers = parent != NULL ? parent->memberCount : 0;
    int traitMembers = 0;
    for (int i = 0; i < d->traitCount; i++) traitMembers += d->traits[i]->memberCount;
    int own = node->as.classDecl.methodCount + node->as.classDecl.getterCount +
              node->as.classDecl.setterCount;
    int capacity = parentMembers + traitMembers + own;

    MemberRegions regions = { 0, 0 };
    if (capacity > 0) {
        d->members = jaiArenaAllocZeroed(&c->ast->arena,
                                         sizeof *d->members * (size_t)capacity);
        for (int i = 0; i < parentMembers; i++) d->members[i] = parent->members[i];
        d->memberCount = parentMembers;
        regions.parentEnd = parentMembers;

        /* Trait members come next so that a default implementation is visible
         * on the class, and an unimplemented requirement stays abstract. */
        for (int t = 0; t < d->traitCount; t++) {
            TypeDecl *trait = d->traits[t];
            const TraitBinding *bind = &d->traitBindings[t];
            for (int m = 0; m < trait->memberCount; m++) {
                if (jaiChkFindAccessorIndex(d, trait->members[m].name,
                                      trait->members[m].isSetter) >= 0)
                    continue;
                /* A member arrives bound the way the class wrote the trait, so
                 * an inherited default reads in the class's own parameters. */
                addMember(c, d, trait->members[m].name,
                          applyTraitBinding(bind, trait->members[m].type),
                          trait->members[m].visibility, trait->members[m].isStatic,
                          trait->members[m].isGetter, trait->members[m].isSetter,
                          trait->members[m].isAbstract, trait->members[m].decl);
            }
        }
        regions.traitEnd = d->memberCount;

        for (int i = 0; i < node->as.classDecl.methodCount; i++)
            installMember(c, d, node->as.classDecl.methods[i], false, false, &regions);
        for (int i = 0; i < node->as.classDecl.getterCount; i++)
            installMember(c, d, node->as.classDecl.getters[i], true, false, &regions);
        for (int i = 0; i < node->as.classDecl.setterCount; i++)
            installMember(c, d, node->as.classDecl.setters[i], false, true, &regions);
    }

    /* Trait satisfaction is checked before anything can conclude that the
     * class is abstract, or an unimplemented requirement would silently make
     * the class abstract instead of reporting E0705. */
    verifyTraits(c, d);
    jaiChkTypeNameRestore(mark);
}

static void layoutTrait(Checker *c, DeclEntry *entry) {
    TypeDecl *d = entry->decl;
    AstNode *node = d->decl;

    int mark = jaiChkTypeNameMark();
    jaiChkPushGenerics(c, node->as.traitDecl.generics, node->as.traitDecl.genericCount);
    jaiChkPushSelf(NULL);

    int superCount = node->as.traitDecl.superCount;
    if (superCount > 0) {
        d->traits = jaiArenaAllocZeroed(&c->ast->arena,
                                        sizeof(TypeDecl *) * (size_t)superCount);
        d->traitBindings = jaiArenaAllocZeroed(
            &c->ast->arena, sizeof(TraitBinding) * (size_t)superCount);
        d->opaqueTraits = jaiArenaAllocZeroed(
            &c->ast->arena, sizeof(const char *) * (size_t)superCount);
        for (int i = 0; i < superCount; i++) {
            AstType *ref = node->as.traitDecl.supers[i];
            const char *opaque = NULL;
            TypeDecl *super = resolveDeclReference(c, ref, true, entry, &opaque);
            if (super == NULL && opaque == NULL) opaque = ref->name;
            if (opaque != NULL) d->opaqueTraits[d->opaqueTraitCount++] = opaque;
            if (super == NULL) continue;
            /* NULL Self: a supertrait's `Self` stays abstract here, because
             * this trait is not the implementing type either. */
            d->traitBindings[d->traitCount] = traitBindingOf(c, super, ref, NULL);
            d->traits[d->traitCount++] = super;
        }
    }

    int inherited = 0;
    for (int i = 0; i < d->traitCount; i++) inherited += d->traits[i]->memberCount;
    int capacity = inherited + node->as.traitDecl.methodCount;
    if (capacity > 0) {
        d->members = jaiArenaAllocZeroed(&c->ast->arena,
                                         sizeof *d->members * (size_t)capacity);
        MemberRegions regions = { 0, 0 };
        for (int t = 0; t < d->traitCount; t++) {
            TypeDecl *super = d->traits[t];
            const TraitBinding *bind = &d->traitBindings[t];
            for (int m = 0; m < super->memberCount; m++) {
                if (jaiChkFindAccessorIndex(d, super->members[m].name,
                                      super->members[m].isSetter) >= 0)
                    continue;
                /* An inherited requirement is restated in this trait's own
                 * parameters: `trait Sorted[T]: Comparable[T]` requires
                 * `compare(T)`, not the supertrait's bare parameter. */
                addMember(c, d, super->members[m].name,
                          applyTraitBinding(bind, super->members[m].type),
                          super->members[m].visibility, super->members[m].isStatic,
                          super->members[m].isGetter, super->members[m].isSetter,
                          super->members[m].isAbstract, super->members[m].decl);
            }
        }
        regions.traitEnd = d->memberCount;
        for (int i = 0; i < node->as.traitDecl.methodCount; i++)
            installMember(c, d, node->as.traitDecl.methods[i], false, false, &regions);
    }

    jaiChkTypeNameRestore(mark);
}

static void layoutEnum(Checker *c, DeclEntry *entry) {
    TypeDecl *d = entry->decl;
    AstNode *node = d->decl;

    int mark = jaiChkTypeNameMark();
    jaiChkPushGenerics(c, node->as.enumDecl.generics, node->as.enumDecl.genericCount);
    jaiChkPushSelf(entry->type);

    int variantCount = node->as.enumDecl.variantCount;
    if (variantCount > 0) {
        d->fields = jaiArenaAllocZeroed(&c->ast->arena,
                                        sizeof *d->fields * (size_t)variantCount);
        for (int i = 0; i < variantCount; i++) {
            AstVariant *v = &node->as.enumDecl.variants[i];
            /* A unit variant is a value of the enum; a payload variant is the
             * constructor for one. `slot` carries the tag. */
            JaiType *type = entry->type;
            if (v->paramCount > 0) {
                JaiType **payload = jaiChkTypeArray(c, v->paramCount);
                for (int p = 0; p < v->paramCount; p++)
                    payload[p] = v->params[p].type != NULL
                                     ? jaiChkResolveAstType(c, v->params[p].type)
                                     : gTypes.tAny;
                type = jaiTypeFn(payload, v->paramCount, entry->type, 0);
            }
            if (jaiChkFindFieldIndex(d, v->name) >= 0) {
                ERR(c, E0709_DUPLICATE_MEMBER, v->span,
                    "`%s` is already a variant of `%s`", v->name, d->name);
                continue;
            }
            int index = d->fieldCount;
            d->fields[index].name = v->name;
            d->fields[index].type = type;
            d->fields[index].visibility = AST_VIS_PUBLIC;
            d->fields[index].isStatic = true;
            d->fields[index].slot = index;
            d->fieldCount = index + 1;
        }
    }
    d->variantCount = d->fieldCount;

    if (node->as.enumDecl.methodCount > 0) {
        d->members = jaiArenaAllocZeroed(
            &c->ast->arena, sizeof *d->members * (size_t)node->as.enumDecl.methodCount);
        MemberRegions regions = { 0, 0 };
        for (int i = 0; i < node->as.enumDecl.methodCount; i++)
            installMember(c, d, node->as.enumDecl.methods[i], false, false, &regions);
    }

    jaiChkTypeNameRestore(mark);
}

void jaiChkLayoutDecl(Checker *c, DeclEntry *entry) {
    if (entry == NULL || entry->status != 0) return;
    entry->status = 1;

    ForeignCtx saved;
    jaiChkForeignBegin(c, entry->decl->origin, true, &saved);

    switch (entry->decl->decl->kind) {
    case AST_CLASS_DECL: layoutClass(c, entry); break;
    case AST_TRAIT_DECL: layoutTrait(c, entry); break;
    case AST_ENUM_DECL:  layoutEnum(c, entry);  break;
    default: break;
    }

    jaiChkForeignEnd(c, &saved);
    entry->status = 2;
}

/* ------------------------------------------------------------------ */
/* Class bodies                                                         */
/* ------------------------------------------------------------------ */

typedef struct {
    AstNode *first;   /* the first super(...) call found anywhere in the body */
} SuperScan;

static bool findSuperCall(AstNode *node, void *userData) {
    SuperScan *scan = (SuperScan *)userData;
    if (scan->first == NULL && node->kind == AST_CALL &&
        node->as.call.callee != NULL && node->as.call.callee->kind == AST_SUPER)
        scan->first = node;
    return true;
}

/* super(...) must open the constructor when the parent declares one, so that a
 * subclass can never observe a half-initialised parent (spec §7.1). */
static void checkSuperInitPosition(Checker *c, TypeDecl *d, AstNode *init) {
    if (d->superclass == NULL || jaiChkFindInitDecl(d->superclass, NULL) == NULL) return;

    AstNode *body = init->as.fn.body;
    if (body == NULL || body->kind != AST_BLOCK) return;

    SuperScan scan = { NULL };
    jaiAstWalk(body, findSuperCall, NULL, &scan);
    if (scan.first == NULL) return;

    AstNode *first = body->as.block.count > 0 && body->as.block.stmts != NULL
                         ? body->as.block.stmts[0]
                         : NULL;
    if (first != NULL && first->kind == AST_EXPR_STMT &&
        first->as.exprStmt.expr == scan.first)
        return;

    JaiDiag *diag = ERR(c, E0708_SUPER_INIT_NOT_FIRST, scan.first->span,
                        "`super(...)` must be the first statement of `init`");
    jaiDiagAddLabel(diag, d->superclass->span, "`%s` declares an `init`",
                    d->superclass->name);
    jaiDiagAddHelp(diag, "move the call to the top of `init`");
}

static void checkClassBody(Checker *c, AstNode *node) {
    TypeDecl *d = jaiTypeDeclFind(node->as.classDecl.name);
    if (d == NULL || d->decl != node) return;
    jaiChkLayoutDecl(c, jaiChkDeclEntry(d));

    int mark = jaiChkTypeNameMark();
    jaiChkPushGenerics(c, node->as.classDecl.generics, node->as.classDecl.genericCount);
    jaiChkPushSelf(jaiChkDeclType(d));
    TypeDecl *savedClass = c->currentClass;
    c->currentClass = d;

    for (int i = 0; i < node->as.classDecl.fieldCount; i++) {
        AstField *f = &node->as.classDecl.fields[i];
        if (f->defaultValue == NULL) continue;
        JaiType *want = jaiChkResolveAstType(c, f->type);
        jaiChkApplyContext(c, f->defaultValue, want);
        JaiType *got = jaiChkValue(c, f->defaultValue);
        jaiChkRequireAssignable(c, f->defaultValue, got, want, E0400_TYPE_MISMATCH,
                          "mismatched type for a field default");
    }

    for (int i = 0; i < node->as.classDecl.methodCount; i++) {
        AstNode *method = node->as.classDecl.methods[i];
        jaiChkFunction(c, method, d, true);
        if (jaiChkIsInitMethod(method)) checkSuperInitPosition(c, d, method);
    }
    for (int i = 0; i < node->as.classDecl.getterCount; i++)
        jaiChkFunction(c, node->as.classDecl.getters[i], d, true);
    for (int i = 0; i < node->as.classDecl.setterCount; i++)
        jaiChkFunction(c, node->as.classDecl.setters[i], d, true);

    c->currentClass = savedClass;
    jaiChkTypeNameRestore(mark);
}

static void checkTraitBody(Checker *c, AstNode *node) {
    TypeDecl *d = jaiTypeDeclFind(node->as.traitDecl.name);
    if (d == NULL || d->decl != node) return;
    jaiChkLayoutDecl(c, jaiChkDeclEntry(d));

    int mark = jaiChkTypeNameMark();
    jaiChkPushGenerics(c, node->as.traitDecl.generics, node->as.traitDecl.genericCount);
    jaiChkPushSelf(NULL);
    TypeDecl *savedClass = c->currentClass;
    c->currentClass = d;

    for (int i = 0; i < node->as.traitDecl.methodCount; i++) {
        AstNode *method = node->as.traitDecl.methods[i];
        if (method->as.fn.body == NULL) continue;   /* a requirement, not a body */
        jaiChkFunction(c, method, d, true);
    }

    c->currentClass = savedClass;
    jaiChkTypeNameRestore(mark);
}

static void checkEnumBody(Checker *c, AstNode *node) {
    TypeDecl *d = jaiTypeDeclFind(node->as.enumDecl.name);
    if (d == NULL || d->decl != node) return;
    jaiChkLayoutDecl(c, jaiChkDeclEntry(d));

    int mark = jaiChkTypeNameMark();
    jaiChkPushGenerics(c, node->as.enumDecl.generics, node->as.enumDecl.genericCount);
    jaiChkPushSelf(jaiChkDeclType(d));
    TypeDecl *savedClass = c->currentClass;
    c->currentClass = d;

    for (int i = 0; i < node->as.enumDecl.variantCount; i++) {
        AstVariant *v = &node->as.enumDecl.variants[i];
        for (int p = 0; p < v->paramCount; p++)
            if (v->params[p].defaultValue != NULL) jaiChkValue(c, v->params[p].defaultValue);
    }
    for (int i = 0; i < node->as.enumDecl.methodCount; i++)
        jaiChkFunction(c, node->as.enumDecl.methods[i], d, true);

    c->currentClass = savedClass;
    jaiChkTypeNameRestore(mark);
}

void jaiChkDecl(Checker *c, AstNode *node) {
    switch (node->kind) {
    case AST_FN_DECL: {
        Symbol *sym = node->as.fn.symbol;
        if (sym != NULL && sym->type == NULL) sym->type = jaiChkFunctionType(c, node, false);
        if (node->type == NULL)
            node->type = sym != NULL ? sym->type : jaiChkFunctionType(c, node, false);
        jaiChkFunction(c, node, NULL, false);
        break;
    }

    case AST_CLASS_DECL: checkClassBody(c, node); break;
    case AST_TRAIT_DECL: checkTraitBody(c, node); break;
    case AST_ENUM_DECL:  checkEnumBody(c, node);  break;

    case AST_TYPE_DECL:
        if (node->as.typeDecl.symbol != NULL && node->as.typeDecl.symbol->type == NULL)
            node->as.typeDecl.symbol->type = jaiChkResolveAstType(c, node->as.typeDecl.aliased);
        break;

    case AST_IMPORT:
        if (node->as.import.symbol != NULL && node->as.import.symbol->type == NULL)
            node->as.import.symbol->type =
                jaiTypeNamed(TY_MODULE, node->as.import.path, NULL);
        break;

    case AST_FROM_IMPORT:
    case AST_EXPORT:
    case AST_MODULE_DECL:
        break;

    default:
        break;
    }
}
