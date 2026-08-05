/* types.c — the interned type universe (spec §2).
 *
 * Every type is built by internType(), which hashes the structure it is asked
 * for (kind, name, arguments, return type, fn flags) and hands back the
 * existing type whenever one matches. Structural equality is therefore pointer
 * equality, which is what makes jaiTypeEquals a single compare and what makes
 * `A | B` and `B | A` the same type: jaiTypeUnion sorts its members with a
 * total structural order before it interns them.
 *
 * Types are arena-allocated and never freed individually; the whole universe
 * goes away in jaiTypesFree. `decl` is the one mutable field — a class type is
 * often interned by name before its declaration exists, and a later
 * jaiTypeNamed with the declaration fills the back-pointer in.
 *
 * Nothing here reports a diagnostic. A relation that fails returns false or
 * NULL and the caller, which owns the span, decides which E04xx code applies:
 * this module cannot tell E0400 from E0411 without knowing the syntax that
 * produced the operands.
 */

#include "types.h"

/* For the layout of TypeDecl: the subclass/trait walks and the dunder lookup
 * that gives operator overloads a result type. */
#include "check.h"

TypeUniverse gTypes;

/* Set before the singletons are built so that the constructors called during
 * jaiTypesInit do not try to initialise the universe again. */
static bool gInitialized = false;

/* Open-addressed index over gTypes.interned. Power-of-two capacity, linear
 * probing, no deletions (a type is never removed), grown at 3/4 load. */
static JaiType **gBuckets   = NULL;
static int       gBucketCap = 0;

/* TY_FN flag bits, matching the function-record flags of BYTECODE.md §5. */
#define FN_FLAG_VARIADIC 0x01u
#define FN_FLAG_KWREST   0x02u

/* A join wider than this stops being useful in a diagnostic and usually means
 * the code really is dynamic, so it collapses to `any` (spec §2.2). */
#define JOIN_MAX_MEMBERS 8

/* Alias and declaration chains are checked for cycles elsewhere (E0706); the
 * caps here only stop a malformed chain from hanging the compiler. */
#define MAX_ALIAS_HOPS 32
#define MAX_DECL_DEPTH 32

/* Structural recursion depth for rendering. Only nesting can reach this —
 * named types are leaves, so a recursive class cannot loop. */
#define MAX_RENDER_DEPTH 32

typedef JAI_VEC(JaiType *) TypeVec;

static void      renderType(JaiBuf *b, JaiType *t, int depth);
static int       typeCompare(const JaiType *a, const JaiType *b);
static bool      assignableInto(JaiType *from, JaiType *to, bool *guard);

/* ------------------------------------------------------------------ */
/* Interning                                                            */
/* ------------------------------------------------------------------ */

static uint64_t mixHash(uint64_t h, uint64_t x) {
    return jaiHashU64(h ^ jaiHashU64(x + 0x9e3779b97f4a7c15ull));
}

/* Hash of the intern key. `decl` is deliberately not part of it: a named type
 * is identified by kind and name so that a forward reference and the eventual
 * declaration land on the same type. */
static uint64_t typeHash(TypeKind kind, const char *name, JaiType *const *args,
                         int argCount, JaiType *ret, uint8_t flags) {
    uint64_t h = jaiHashU64((uint64_t)kind + 1u);
    if (name != NULL) h = mixHash(h, jaiHashBytes(name, strlen(name)));
    h = mixHash(h, (uint64_t)argCount);
    for (int i = 0; i < argCount; i++) {
        /* args are already interned, so their hashes are stable identities. */
        h = mixHash(h, args[i] != NULL ? args[i]->hash : 0);
    }
    if (ret != NULL) h = mixHash(h, ret->hash);
    return mixHash(h, flags);
}

static bool nameEquals(const char *a, const char *b) {
    if (a == b) return true;
    if (a == NULL || b == NULL) return false;
    return strcmp(a, b) == 0;
}

static bool typeMatches(const JaiType *t, TypeKind kind, const char *name,
                        JaiType *const *args, int argCount, JaiType *ret,
                        uint8_t flags) {
    if (t->kind != kind || t->argCount != argCount || t->ret != ret ||
        t->fnFlags != flags) {
        return false;
    }
    if (!nameEquals(t->name, name)) return false;
    for (int i = 0; i < argCount; i++) {
        if (t->args[i] != args[i]) return false;
    }
    return true;
}

static void bucketInsert(JaiType *t) {
    uint64_t mask = (uint64_t)gBucketCap - 1u;
    uint64_t i = t->hash & mask;
    while (gBuckets[i] != NULL) i = (i + 1u) & mask;
    gBuckets[i] = t;
}

static void bucketsGrow(void) {
    int oldCap = gBucketCap;
    JaiType **old = gBuckets;

    gBucketCap = oldCap < 256 ? 256 : oldCap * 2;
    gBuckets = JAI_ALLOC_ZEROED(JaiType *, gBucketCap);
    for (int i = 0; i < gTypes.interned.count; i++) {
        bucketInsert(gTypes.interned.data[i]);
    }
    JAI_FREE_ARRAY(JaiType *, old, oldCap);
}

/* A type outlives the syntax that named it. Every named type takes its `name`
 * from an identifier the parser interned into the AST arena (jaiPInternText),
 * and that arena is freed at the end of each module's front-end pass
 * (jaiAstContextFree, called from frontEnd in module.c). The type universe is
 * global and lives for the process, so a type interned while compiling one
 * module kept a pointer into memory the next import had already released, and
 * the next internType compared against it with strcmp. ASan caught it as eleven
 * heap-use-after-frees, all with the same stack.
 *
 * Copying into the universe's own arena is what makes `name` genuinely interned
 * rather than borrowed from whichever module happened to mention the type
 * first. The copy is per distinct type, not per lookup: the lookup path below
 * still compares against the caller's string and never allocates. */
static const char *internName(const char *name) {
    if (name == NULL) return NULL;
    return jaiArenaMemdup(&gTypes.arena, name, strlen(name) + 1);
}

static JaiType *internType(TypeKind kind, const char *name, JaiType **args,
                           int argCount, JaiType *ret, uint8_t flags,
                           TypeDecl *decl) {
    if (argCount < 0 || args == NULL) argCount = 0;

    uint64_t hash = typeHash(kind, name, args, argCount, ret, flags);
    if (gBucketCap > 0) {
        uint64_t mask = (uint64_t)gBucketCap - 1u;
        for (uint64_t i = hash & mask; gBuckets[i] != NULL; i = (i + 1u) & mask) {
            JaiType *candidate = gBuckets[i];
            if (candidate->hash != hash) continue;
            if (!typeMatches(candidate, kind, name, args, argCount, ret, flags)) {
                continue;
            }
            /* The declaration may arrive after the type: a use site can name a
             * class before the checker has built its TypeDecl. */
            if (decl != NULL) candidate->decl = decl;
            return candidate;
        }
    }

    if (gBucketCap == 0 || (gTypes.interned.count + 1) * 4 > gBucketCap * 3) {
        bucketsGrow();
    }

    JaiType *t = JAI_ARENA_NEW(&gTypes.arena, JaiType);
    t->kind = kind;
    t->name = internName(name);
    t->argCount = argCount;
    t->ret = ret;
    t->decl = decl;
    t->fnFlags = flags;
    t->hash = hash;
    if (argCount > 0) {
        t->args = JAI_ARENA_NEW_ARRAY(&gTypes.arena, JaiType *, argCount);
        memcpy(t->args, args, sizeof(JaiType *) * (size_t)argCount);
    } else {
        t->args = NULL;
    }

    JAI_VEC_PUSH(JaiType *, &gTypes.interned, t);
    bucketInsert(t);
    return t;
}

static void ensureInit(void) {
    if (!gInitialized) jaiTypesInit();
}

/* ------------------------------------------------------------------ */
/* The universe                                                         */
/* ------------------------------------------------------------------ */

void jaiTypesInit(void) {
    /* Idempotent: a second init must not invalidate types the caller is still
     * holding. Use jaiTypesFree to actually tear the universe down. */
    if (gInitialized) return;
    gInitialized = true;

    jaiArenaInit(&gTypes.arena, 16 * 1024);
    JAI_VEC_INIT(&gTypes.interned);
    gBuckets = NULL;
    gBucketCap = 0;

    gTypes.tAny   = internType(TY_ANY, NULL, NULL, 0, NULL, 0, NULL);
    gTypes.tNever = internType(TY_NEVER, NULL, NULL, 0, NULL, 0, NULL);
    gTypes.tNull  = internType(TY_NULL, NULL, NULL, 0, NULL, 0, NULL);
    gTypes.tBool  = internType(TY_BOOL, NULL, NULL, 0, NULL, 0, NULL);
    gTypes.tInt   = internType(TY_INT, NULL, NULL, 0, NULL, 0, NULL);
    gTypes.tFloat = internType(TY_FLOAT, NULL, NULL, 0, NULL, 0, NULL);
    gTypes.tStr   = internType(TY_STR, NULL, NULL, 0, NULL, 0, NULL);
    gTypes.tBytes = internType(TY_BYTES, NULL, NULL, 0, NULL, 0, NULL);
    gTypes.tRange = internType(TY_RANGE, NULL, NULL, 0, NULL, 0, NULL);
    /* `void` is not a separate type: a void function returns null (spec §6).
     * It exists as a name so return positions can render as `-> void`. */
    gTypes.tVoid  = gTypes.tNull;
}

/* The rendering ring, defined here so jaiTypesFree can release it. */
#define STRING_RING_SIZE 8
static JaiBuf gStringRing[STRING_RING_SIZE];
static int    gStringRingNext = 0;

void jaiTypesFree(void) {
    if (!gInitialized) return;

    for (int i = 0; i < STRING_RING_SIZE; i++) jaiBufFree(&gStringRing[i]);
    gStringRingNext = 0;

    JAI_FREE_ARRAY(JaiType *, gBuckets, gBucketCap);
    gBuckets = NULL;
    gBucketCap = 0;

    JAI_VEC_FREE(JaiType *, &gTypes.interned);
    jaiArenaFree(&gTypes.arena);
    memset(&gTypes, 0, sizeof gTypes);
    gInitialized = false;
}

/* ------------------------------------------------------------------ */
/* Alias resolution                                                     */
/* ------------------------------------------------------------------ */

/* Every relation works on resolved types: an alias must behave exactly like
 * its target. Only rendering keeps the alias, because the name is what the
 * programmer wrote. */
static JaiType *resolveAlias(JaiType *t) {
    for (int hops = 0; t != NULL && t->kind == TY_ALIAS && t->ret != NULL &&
                       hops < MAX_ALIAS_HOPS;
         hops++) {
        t = t->ret;
    }
    return t;
}

/* ------------------------------------------------------------------ */
/* Constructors                                                         */
/* ------------------------------------------------------------------ */

JaiType *jaiTypeList(JaiType *elem) {
    ensureInit();
    if (elem == NULL) elem = gTypes.tAny;
    return internType(TY_LIST, NULL, &elem, 1, NULL, 0, NULL);
}

JaiType *jaiTypeDict(JaiType *key, JaiType *value) {
    ensureInit();
    JaiType *args[2];
    args[0] = key != NULL ? key : gTypes.tAny;
    args[1] = value != NULL ? value : gTypes.tAny;
    return internType(TY_DICT, NULL, args, 2, NULL, 0, NULL);
}

JaiType *jaiTypeSet(JaiType *elem) {
    ensureInit();
    if (elem == NULL) elem = gTypes.tAny;
    return internType(TY_SET, NULL, &elem, 1, NULL, 0, NULL);
}

JaiType *jaiTypeTuple(JaiType **members, int count) {
    ensureInit();
    if (members == NULL || count < 0) count = 0;
    /* A missing member type is `any`, never a NULL slot: every later walk over
     * args would have to test for it otherwise. */
    bool needsFix = false;
    for (int i = 0; i < count; i++) {
        if (members[i] == NULL) { needsFix = true; break; }
    }
    if (!needsFix) return internType(TY_TUPLE, NULL, members, count, NULL, 0, NULL);

    TypeVec fixed;
    JAI_VEC_INIT(&fixed);
    for (int i = 0; i < count; i++) {
        JAI_VEC_PUSH(JaiType *, &fixed, members[i] != NULL ? members[i] : gTypes.tAny);
    }
    JaiType *t = internType(TY_TUPLE, NULL, fixed.data, fixed.count, NULL, 0, NULL);
    JAI_VEC_FREE(JaiType *, &fixed);
    return t;
}

JaiType *jaiTypeFn(JaiType **params, int count, JaiType *ret, uint8_t flags) {
    ensureInit();
    if (params == NULL || count < 0) count = 0;
    /* Return type omitted means `any` (spec §6). */
    if (ret == NULL) ret = gTypes.tAny;

    bool needsFix = false;
    for (int i = 0; i < count; i++) {
        if (params[i] == NULL) { needsFix = true; break; }
    }
    if (!needsFix) return internType(TY_FN, NULL, params, count, ret, flags, NULL);

    TypeVec fixed;
    JAI_VEC_INIT(&fixed);
    for (int i = 0; i < count; i++) {
        JAI_VEC_PUSH(JaiType *, &fixed, params[i] != NULL ? params[i] : gTypes.tAny);
    }
    JaiType *t = internType(TY_FN, NULL, fixed.data, fixed.count, ret, flags, NULL);
    JAI_VEC_FREE(JaiType *, &fixed);
    return t;
}

JaiType *jaiTypeOptional(JaiType *inner) {
    ensureInit();
    if (inner == NULL) return gTypes.tAny;
    JaiType *members[2] = {inner, gTypes.tNull};
    return jaiTypeUnion(members, 2);
}

JaiType *jaiTypeNamed(TypeKind kind, const char *name, TypeDecl *decl) {
    ensureInit();
    /* A named type without a name has no identity to intern on; the caller has
     * already reported whatever went wrong upstream (E0402, E0200). */
    if (name == NULL) return gTypes.tAny;
    return internType(kind, name, NULL, 0, NULL, 0, decl);
}

JaiType *jaiTypeGenericParam(const char *name) {
    return jaiTypeNamed(TY_GENERIC_PARAM, name, NULL);
}

/* ------------------------------------------------------------------ */
/* Union normalisation                                                  */
/* ------------------------------------------------------------------ */

/* Total order over types. It exists only to give union members one canonical
 * arrangement, so its exact shape does not matter — but it must be total, or
 * `A | B` and `B | A` would intern to two different types. Named types compare
 * by kind and name alone, which is also why a recursive class cannot make this
 * recurse forever. */
static int typeCompare(const JaiType *a, const JaiType *b) {
    if (a == b) return 0;
    if (a == NULL) return -1;
    if (b == NULL) return 1;
    if (a->kind != b->kind) return a->kind < b->kind ? -1 : 1;

    if (a->name != b->name) {
        if (a->name == NULL) return -1;
        if (b->name == NULL) return 1;
        int c = strcmp(a->name, b->name);
        if (c != 0) return c < 0 ? -1 : 1;
    }
    if (a->argCount != b->argCount) return a->argCount < b->argCount ? -1 : 1;
    for (int i = 0; i < a->argCount; i++) {
        int c = typeCompare(a->args[i], b->args[i]);
        if (c != 0) return c;
    }
    if (a->ret != b->ret) {
        int c = typeCompare(a->ret, b->ret);
        if (c != 0) return c;
    }
    if (a->fnFlags != b->fnFlags) return a->fnFlags < b->fnFlags ? -1 : 1;
    return 0;
}

static void unionCollect(TypeVec *out, JaiType *t) {
    t = resolveAlias(t);
    if (t == NULL) return;
    if (t->kind == TY_UNION) {
        for (int i = 0; i < t->argCount; i++) unionCollect(out, t->args[i]);
        return;
    }
    JAI_VEC_PUSH(JaiType *, out, t);
}

JaiType *jaiTypeUnion(JaiType **members, int count) {
    ensureInit();
    if (members == NULL || count <= 0) return gTypes.tNever;

    TypeVec flat;
    JAI_VEC_INIT(&flat);
    for (int i = 0; i < count; i++) unionCollect(&flat, members[i]);

    /* `any` absorbs: a union that can hold anything is `any`, and dropping
     * `never` keeps the bottom type from widening a union it cannot inhabit. */
    int live = 0;
    for (int i = 0; i < flat.count; i++) {
        JaiType *m = flat.data[i];
        if (m->kind == TY_ANY) {
            JAI_VEC_FREE(JaiType *, &flat);
            return gTypes.tAny;
        }
        if (m->kind == TY_NEVER) continue;
        flat.data[live++] = m;
    }
    flat.count = live;

    /* Insertion sort: union members are few, and this is deterministic. */
    for (int i = 1; i < flat.count; i++) {
        JaiType *key = flat.data[i];
        int j = i - 1;
        while (j >= 0 && typeCompare(flat.data[j], key) > 0) {
            flat.data[j + 1] = flat.data[j];
            j--;
        }
        flat.data[j + 1] = key;
    }

    int unique = 0;
    for (int i = 0; i < flat.count; i++) {
        if (unique > 0 && flat.data[unique - 1] == flat.data[i]) continue;
        flat.data[unique++] = flat.data[i];
    }
    flat.count = unique;

    JaiType *result;
    if (flat.count == 0) {
        result = gTypes.tNever;
    } else if (flat.count == 1) {
        result = flat.data[0];
    } else {
        result = internType(TY_UNION, NULL, flat.data, flat.count, NULL, 0, NULL);
    }
    JAI_VEC_FREE(JaiType *, &flat);
    return result;
}

/* ------------------------------------------------------------------ */
/* Declaration walks                                                    */
/* ------------------------------------------------------------------ */

static bool declIsSubclassOf(const TypeDecl *sub, const TypeDecl *super) {
    if (sub == NULL || super == NULL) return false;
    for (int depth = 0; sub != NULL && depth < MAX_DECL_DEPTH;
         sub = sub->superclass, depth++) {
        if (sub == super) return true;
    }
    return false;
}

/* Does `d` (a class, enum, or trait) satisfy trait `trait`? Traits are
 * inherited through the superclass chain and through super-traits. */
static bool declImplements(const TypeDecl *d, const TypeDecl *trait, int depth) {
    if (d == NULL || trait == NULL || depth > MAX_DECL_DEPTH) return false;
    if (d == trait) return true;
    for (int i = 0; d->traits != NULL && i < d->traitCount; i++) {
        if (declImplements(d->traits[i], trait, depth + 1)) return true;
    }
    return declImplements(d->superclass, trait, depth + 1);
}

/* Does `d` name a trait this unit cannot see the declaration of? Matching is by
 * name because a name is all there is: the trait came in as a runtime object,
 * and nothing built a TypeDecl for it. The class still had to write the trait
 * in its header, so this stays narrower than "any class satisfies it". */
static bool declNamesOpaqueTrait(const TypeDecl *d, const char *name, int depth) {
    if (d == NULL || name == NULL || depth > MAX_DECL_DEPTH) return false;
    for (int i = 0; d->opaqueTraits != NULL && i < d->opaqueTraitCount; i++) {
        if (nameEquals(d->opaqueTraits[i], name)) return true;
    }
    for (int i = 0; d->traits != NULL && i < d->traitCount; i++) {
        if (declNamesOpaqueTrait(d->traits[i], name, depth + 1)) return true;
    }
    return declNamesOpaqueTrait(d->superclass, name, depth + 1);
}

/* Find a method or field by name, walking traits and the superclass chain.
 * *outType may come back NULL when the member exists but has not been checked
 * yet; that is distinct from the member being absent. */
static bool declFindMember(const TypeDecl *d, const char *name,
                           JaiType **outType, int depth) {
    if (d == NULL || name == NULL || depth > MAX_DECL_DEPTH) return false;
    for (int i = 0; d->members != NULL && i < d->memberCount; i++) {
        if (d->members[i].name != NULL && strcmp(d->members[i].name, name) == 0) {
            if (outType != NULL) *outType = d->members[i].type;
            return true;
        }
    }
    for (int i = 0; d->traits != NULL && i < d->traitCount; i++) {
        if (declFindMember(d->traits[i], name, outType, depth + 1)) return true;
    }
    return declFindMember(d->superclass, name, outType, depth + 1);
}

static bool typeIsUserDefined(const JaiType *t) {
    return t->kind == TY_CLASS || t->kind == TY_TRAIT || t->kind == TY_ENUM;
}

/* ------------------------------------------------------------------ */
/* Assignability (spec §2.2)                                            */
/* ------------------------------------------------------------------ */

static bool assignableInto(JaiType *from, JaiType *to, bool *guard) {
    /* An unknown operand means an error was already reported upstream;
     * accepting it keeps one mistake from producing a second diagnostic. */
    if (from == NULL || to == NULL) return true;
    if (from == to) return true;                 /* interning: T -> T */
    if (to->kind == TY_ANY) return true;         /* everything -> any */
    if (from->kind == TY_NEVER) return true;     /* never -> everything */
    if (from->kind == TY_ANY) {
        /* `to` is concrete here: the dynamic-to-static boundary, checked at
         * run time by a TYPE_GUARD. */
        *guard = true;
        return true;
    }

    /* Generics are erased (spec §6.1), so an unsubstituted parameter names no
     * runtime shape and accepts every value. Without this every call to a
     * generic function would be an arity-shaped type error. */
    if (to->kind == TY_GENERIC_PARAM) return true;

    if (from->kind == TY_UNION) {
        bool g = false;
        for (int i = 0; i < from->argCount; i++) {
            if (!assignableInto(from->args[i], to, &g)) return false;
        }
        if (g) *guard = true;
        return true;
    }
    if (to->kind == TY_UNION) {
        /* Also the null -> T? rule, since T? is T | null. A member that fits
         * without a guard wins over one that only fits behind a run-time check:
         * `U` reaching `U | null` must not be reported as a dynamic conversion
         * just because the null member was tried first. */
        bool guarded = false;
        for (int i = 0; i < to->argCount; i++) {
            bool g = false;
            if (!assignableInto(from, to->args[i], &g)) continue;
            if (!g) return true;
            guarded = true;
        }
        if (guarded) *guard = true;
        return guarded;
    }

    /* The mirror of the rule above, but only once the unions have been taken
     * apart: `U` reaching `T | null` is `U` reaching `T`, which is free, while
     * `U` reaching `int` is the erased-to-concrete boundary `any` also crosses,
     * and that is checked at run time. */
    if (from->kind == TY_GENERIC_PARAM) {
        *guard = true;
        return true;
    }

    switch (from->kind) {
    case TY_CLASS:
    case TY_ENUM:
        if (to->kind == TY_CLASS) return declIsSubclassOf(from->decl, to->decl);
        if (to->kind == TY_TRAIT)
            return to->decl != NULL
                       ? declImplements(from->decl, to->decl, 0)
                       : declNamesOpaqueTrait(from->decl, to->name, 0);
        return false;

    case TY_TRAIT:
        /* A trait value satisfies the traits that trait extends. */
        if (to->kind != TY_TRAIT) return false;
        return to->decl != NULL ? declImplements(from->decl, to->decl, 0)
                                : declNamesOpaqueTrait(from->decl, to->name, 0);

    case TY_LIST:
    case TY_SET:
    case TY_DICT: {
        /* Invariant between concrete elements, not covariant. These containers
         * are mutable: if list[Circle] were assignable to list[Shape], the
         * callee could push a Square through the widened reference and the
         * caller would read it back as a Circle.
         *
         * A dynamic element is the exception, because `any` is the one type
         * that converts both ways (spec §2.2): `list[any]` — what an untyped
         * comprehension or a `.enumerate()` produces — flows into `list[str]`
         * behind the same guard a bare `any` gets, and `list[str]` flows back
         * into `list[any]` freely. Equal element types are the same pointer,
         * so the identity test above already covered them. */
        if (to->kind != from->kind || from->argCount != to->argCount) return false;
        bool g = false;
        for (int i = 0; i < from->argCount; i++) {
            JaiType *a = from->args[i], *b = to->args[i];
            if (a == b) continue;
            if (a->kind == TY_ANY || b->kind == TY_ANY ||
                a->kind == TY_GENERIC_PARAM || b->kind == TY_GENERIC_PARAM) {
                if (a->kind == TY_ANY && b->kind != TY_ANY) g = true;
                continue;
            }
            /* Structural elements answer the same question one level down:
             * `list[(any, any)]` is what an untyped comprehension of pairs
             * produces, and it has to reach `list[(K, V)]`. Nominal elements
             * do not recurse — that would be the covariance ruled out above. */
            if (a->kind == b->kind &&
                (a->kind == TY_LIST || a->kind == TY_SET || a->kind == TY_DICT ||
                 a->kind == TY_TUPLE || a->kind == TY_FN) &&
                assignableInto(a, b, &g))
                continue;
            return false;
        }
        if (g) *guard = true;
        return true;
    }

    case TY_TUPLE: {
        /* Tuples are immutable, so element-wise covariance is sound. */
        if (to->kind != TY_TUPLE || from->argCount != to->argCount) return false;
        bool g = false;
        for (int i = 0; i < from->argCount; i++) {
            if (!assignableInto(from->args[i], to->args[i], &g)) return false;
        }
        if (g) *guard = true;
        return true;
    }

    case TY_FN: {
        if (to->kind != TY_FN) return false;
        if (from->argCount != to->argCount || from->fnFlags != to->fnFlags) {
            return false;
        }
        bool g = false;
        /* Contravariant in parameters: the target's caller passes `to`'s
         * argument types, so `from` must accept them. */
        for (int i = 0; i < from->argCount; i++) {
            if (!assignableInto(to->args[i], from->args[i], &g)) return false;
        }
        /* Covariant in the return type. */
        if (!assignableInto(from->ret, to->ret, &g)) return false;
        if (g) *guard = true;
        return true;
    }

    default:
        /* int -> float is not here: it is a conversion rather than a subtype
         * relation, so it belongs to jaiTypeWidenTarget below and not to this
         * recursion. See the comment there. */
        return false;
    }
}

JaiType *jaiTypeWidenTarget(JaiType *from, JaiType *to) {
    ensureInit();
    from = resolveAlias(from);
    to = resolveAlias(to);
    if (from == NULL || to == NULL || from->kind != TY_INT) return NULL;
    if (to->kind == TY_FLOAT) return to;

    /* `float?` and other unions with a float arm but no int arm: the value has
     * to become a float to fit, and that is the same conversion. A union that
     * already admits int (or `any`) needs no conversion and never reaches
     * here, because plain assignability accepted it. */
    if (to->kind != TY_UNION) return NULL;
    JaiType *found = NULL;
    for (int i = 0; i < to->argCount; i++) {
        JaiType *m = resolveAlias(to->args[i]);
        if (m->kind == TY_INT || m->kind == TY_ANY) return NULL;
        if (m->kind == TY_FLOAT) found = m;
    }
    return found;
}

bool jaiTypeAssignable(JaiType *from, JaiType *to, bool *needsGuard) {
    ensureInit();
    if (needsGuard != NULL) *needsGuard = false;

    bool guard = false;
    if (!assignableInto(resolveAlias(from), resolveAlias(to), &guard)) {
        /* int widens to float on assignment (spec §2.2). Reported through
         * needsGuard because the caller must insert the conversion: unlike
         * every other assignment, the bits that arrive are not the bits that
         * are stored.
         *
         * It is applied here, at the top level, rather than inside
         * assignableInto, so that it stays a conversion and does not leak into
         * the subtype relation. If `int` were a subtype of `float` then
         * `list[int]` would pass where a `list[float]` was wanted with no way
         * to convert the elements, and a `fn() -> int` would pass as a
         * `fn() -> float` with no way to convert a result that does not exist
         * yet. Those two are still exactly why the relation stops here.
         *
         * `[1, 2.0]` used to be the third item on that list. It is not any
         * more: the checker fuses such a literal before the join runs (see
         * fuseNumeric in check_expr.c), converting each int element in place,
         * so the `list[float]` that jaiTypeJoin reports is one the elements
         * actually agree with. That is a rewrite of the tree, performed where
         * the per-element nodes exist; it is still not a subtype rule, and
         * nothing in this file knows about it. */
        if (jaiTypeWidenTarget(from, to) == NULL) return false;
        guard = true;
    }
    if (needsGuard != NULL) *needsGuard = guard;
    return true;
}

bool jaiTypeEquals(JaiType *a, JaiType *b) {
    if (a == b) return true;
    ensureInit();
    return resolveAlias(a) == resolveAlias(b);
}

bool jaiTypeIsSubtype(JaiType *sub, JaiType *super) {
    /* Subtyping is assignability minus the dynamic escape hatch: `any` is not
     * a subtype of a concrete type, it merely converts into one. */
    bool guard = false;
    return jaiTypeAssignable(sub, super, &guard) && !guard;
}

/* ------------------------------------------------------------------ */
/* Join and narrowing                                                   */
/* ------------------------------------------------------------------ */

JaiType *jaiTypeJoin(JaiType *a, JaiType *b) {
    ensureInit();
    a = resolveAlias(a);
    b = resolveAlias(b);
    if (a == NULL) return b != NULL ? b : gTypes.tAny;
    if (b == NULL) return a;
    if (a == b) return a;

    if (a->kind == TY_NEVER) return b;
    if (b->kind == TY_NEVER) return a;
    if (a->kind == TY_ANY || b->kind == TY_ANY) return gTypes.tAny;

    /* Covers a class and its ancestor, a class and a trait it implements, and
     * T with T?. */
    if (jaiTypeIsSubtype(a, b)) return b;
    if (jaiTypeIsSubtype(b, a)) return a;

    JaiType *members[2] = {a, b};
    JaiType *u = jaiTypeUnion(members, 2);
    if (u->kind == TY_UNION && u->argCount > JOIN_MAX_MEMBERS) return gTypes.tAny;
    return u;
}

/* What survives of `m` when the value is known to be a `by`? Either `m` itself
 * (already at least as specific), or `by` (a downcast: `x is Circle` where x is
 * a Shape), or nothing (the test can never succeed for this member). */
static JaiType *narrowMember(JaiType *m, JaiType *by) {
    if (jaiTypeIsSubtype(m, by)) return m;
    if (jaiTypeIsSubtype(by, m)) return by;
    /* `any` reaches here only as `by`; `m` being `any` is handled by the
     * downcast case above, which is what makes `if x is Circle` on a dynamic
     * binding narrow to Circle. */
    return NULL;
}

JaiType *jaiTypeNarrow(JaiType *t, JaiType *by, bool positive) {
    ensureInit();
    t = resolveAlias(t);
    by = resolveAlias(by);
    if (t == NULL || by == NULL) return t;

    if (t->kind == TY_UNION) {
        TypeVec keep;
        JAI_VEC_INIT(&keep);
        for (int i = 0; i < t->argCount; i++) {
            JaiType *m = t->args[i];
            if (positive) {
                JaiType *n = narrowMember(m, by);
                if (n != NULL) JAI_VEC_PUSH(JaiType *, &keep, n);
            } else if (!jaiTypeIsSubtype(m, by)) {
                /* Subtract: only a member that definitely *is* a `by` goes
                 * away. This is `x != null` removing null from int | null. */
                JAI_VEC_PUSH(JaiType *, &keep, m);
            }
        }
        JaiType *result =
            keep.count == 0 ? gTypes.tNever : jaiTypeUnion(keep.data, keep.count);
        JAI_VEC_FREE(JaiType *, &keep);
        return result;
    }

    if (positive) {
        JaiType *n = narrowMember(t, by);
        /* No overlap: the branch is unreachable, so the binding has no value. */
        return n != NULL ? n : gTypes.tNever;
    }
    return jaiTypeIsSubtype(t, by) ? gTypes.tNever : t;
}

/* ------------------------------------------------------------------ */
/* Predicates                                                           */
/* ------------------------------------------------------------------ */

bool jaiTypeIsNumeric(JaiType *t) {
    t = resolveAlias(t);
    return t != NULL && (t->kind == TY_INT || t->kind == TY_FLOAT);
}

bool jaiTypeIsOptional(JaiType *t) {
    t = resolveAlias(t);
    if (t == NULL) return false;
    /* `any` and `null` both admit null without a union wrapper. */
    if (t->kind == TY_NULL || t->kind == TY_ANY) return true;
    if (t->kind != TY_UNION) return false;
    for (int i = 0; i < t->argCount; i++) {
        if (t->args[i]->kind == TY_NULL) return true;
    }
    return false;
}

bool jaiTypeIsCallable(JaiType *t) {
    t = resolveAlias(t);
    if (t == NULL) return false;
    switch (t->kind) {
    case TY_FN:
    case TY_ANY:
    case TY_CLASS:   /* the class itself is the constructor (spec §7.1) */
    case TY_ENUM:    /* variant construction */
        return true;
    case TY_TRAIT:
        return declFindMember(t->decl, "__call__", NULL, 0);
    default:
        return false;
    }
}

static JaiType *joinAll(JaiType **types, int count) {
    JaiType *result = gTypes.tNever;
    for (int i = 0; i < count; i++) result = jaiTypeJoin(result, types[i]);
    return result;
}

bool jaiTypeIsIterable(JaiType *t, JaiType **outElem) {
    ensureInit();
    t = resolveAlias(t);
    if (t == NULL) return false;

    JaiType *elem = NULL;
    switch (t->kind) {
    case TY_LIST:
    case TY_SET:
        elem = t->args[0];
        break;
    case TY_DICT:
        elem = t->args[0];   /* iterating a dict yields its keys */
        break;
    case TY_TUPLE:
        elem = joinAll(t->args, t->argCount);
        break;
    case TY_STR:
        elem = gTypes.tStr;  /* one-scalar strings, not bytes */
        break;
    case TY_BYTES:
        elem = gTypes.tInt;
        break;
    case TY_RANGE:
        elem = gTypes.tInt;
        break;
    case TY_ANY:
        elem = gTypes.tAny;
        break;
    case TY_CLASS:
    case TY_TRAIT:
    case TY_ENUM:
        /* A user iterable satisfies trait Iterable (spec §5.2). Its element
         * type hides inside Iterator[T], which this module cannot open, so the
         * loop variable is dynamic. */
        if (!declFindMember(t->decl, "iter", NULL, 0) &&
            !declFindMember(t->decl, "__iter__", NULL, 0)) {
            return false;
        }
        elem = gTypes.tAny;
        break;
    default:
        return false;
    }

    if (outElem != NULL) *outElem = elem;
    return true;
}

bool jaiTypeIsIndexable(JaiType *t, JaiType **outIndex, JaiType **outResult) {
    ensureInit();
    t = resolveAlias(t);
    if (t == NULL) return false;

    JaiType *index = gTypes.tInt;
    JaiType *result = NULL;
    switch (t->kind) {
    case TY_LIST:
        result = t->args[0];
        break;
    case TY_DICT:
        index = t->args[0];
        result = t->args[1];
        break;
    case TY_TUPLE:
        /* Without a constant index the best static answer is the join of the
         * members; a checker that knows the index can do better itself. */
        result = joinAll(t->args, t->argCount);
        break;
    case TY_STR:
        result = gTypes.tStr;
        break;
    case TY_BYTES:
    case TY_RANGE:
        result = gTypes.tInt;
        break;
    case TY_ANY:
        index = gTypes.tAny;
        result = gTypes.tAny;
        break;
    case TY_CLASS:
    case TY_TRAIT:
    case TY_ENUM:
        if (!declFindMember(t->decl, "__getitem__", NULL, 0)) return false;
        index = gTypes.tAny;
        result = gTypes.tAny;
        break;
    default:
        return false;   /* sets are not indexable */
    }

    if (outIndex != NULL) *outIndex = index;
    if (outResult != NULL) *outResult = result;
    return true;
}

/* ------------------------------------------------------------------ */
/* Operators                                                            */
/* ------------------------------------------------------------------ */

/* The dunder that implements `op` for a user type (spec §7.1). Operators
 * without one are simply not overloadable. */
static const char *arithDunder(OpKind op) {
    switch (op) {
    case OPK_ADD: return "__add__";
    case OPK_SUB: return "__sub__";
    case OPK_MUL: return "__mul__";
    case OPK_DIV: return "__div__";
    case OPK_MOD: return "__mod__";
    case OPK_POW: return "__pow__";
    default:      return NULL;
    }
}

static JaiType *overloadResult(JaiType *t, const char *dunder) {
    if (dunder == NULL || t->decl == NULL) return NULL;
    JaiType *member = NULL;
    if (!declFindMember(t->decl, dunder, &member, 0)) return NULL;
    /* The method exists but its signature has not been checked yet, or it is
     * not a function: `any` keeps the expression usable instead of raising a
     * bogus E0406 that would vanish on the next pass. */
    if (member == NULL || member->kind != TY_FN || member->ret == NULL) {
        return gTypes.tAny;
    }
    return member->ret;
}

static bool isKind(const JaiType *t, TypeKind k) { return t->kind == k; }

JaiType *jaiTypeBinaryResult(int opKind, JaiType *left, JaiType *right) {
    ensureInit();
    JaiType *l = resolveAlias(left);
    JaiType *r = resolveAlias(right);
    if (l == NULL) l = gTypes.tAny;
    if (r == NULL) r = gTypes.tAny;
    OpKind op = (OpKind)opKind;

    switch (op) {
    /* Comparisons always produce bool. Whether the operands are comparable at
     * all depends on __eq__/__lt__, which is a member lookup the caller does;
     * this table only fixes the result type. */
    case OPK_EQ: case OPK_NE:
    case OPK_LT: case OPK_LE: case OPK_GT: case OPK_GE:
    case OPK_IS: case OPK_IS_NOT:
    case OPK_IN: case OPK_NOT_IN:
        return gTypes.tBool;

    /* No truthiness (spec §5.1): `and`/`or` take bools and yield a bool. */
    case OPK_AND: case OPK_OR: {
        bool lok = isKind(l, TY_BOOL) || isKind(l, TY_ANY);
        bool rok = isKind(r, TY_BOOL) || isKind(r, TY_ANY);
        return (lok && rok) ? gTypes.tBool : NULL;
    }
    default:
        break;
    }

    /* Anything mixed with `any` stays dynamic. */
    if (isKind(l, TY_ANY) || isKind(r, TY_ANY)) return gTypes.tAny;

    bool bothInt = isKind(l, TY_INT) && isKind(r, TY_INT);
    bool bothFloat = isKind(l, TY_FLOAT) && isKind(r, TY_FLOAT);
    /* One int and one float: the int widens and the result is float (spec
     * §2.5). `anyFloat` is what the arithmetic cases below test, so a mixed
     * pair lands on the same row of the table as two floats. The checker
     * inserts the conversion into the tree before it gets here; this function
     * only says what the result type is.
     *
     * Deliberately not `mixed` for the wrapping forms or for the bitwise
     * operators: both are 64-bit integer operations with no float meaning, so
     * they stay int-only and a float operand is still E0406. */
    bool mixed = (isKind(l, TY_INT) && isKind(r, TY_FLOAT)) ||
                 (isKind(l, TY_FLOAT) && isKind(r, TY_INT));
    bool anyFloat = bothFloat || mixed;

    switch (op) {
    case OPK_ADD:
        if (bothInt) return gTypes.tInt;
        if (anyFloat) return gTypes.tFloat;
        if (isKind(l, TY_STR) && isKind(r, TY_STR)) return gTypes.tStr;
        if (isKind(l, TY_BYTES) && isKind(r, TY_BYTES)) return gTypes.tBytes;
        if (isKind(l, TY_LIST) && isKind(r, TY_LIST)) {
            return jaiTypeList(jaiTypeJoin(l->args[0], r->args[0]));
        }
        break;

    case OPK_SUB:
        if (bothInt) return gTypes.tInt;
        if (anyFloat) return gTypes.tFloat;
        break;

    case OPK_MUL:
        if (bothInt) return gTypes.tInt;
        if (anyFloat) return gTypes.tFloat;
        /* Repetition. int * str is the same operation with the operands the
         * other way round. */
        if (isKind(l, TY_STR) && isKind(r, TY_INT)) return gTypes.tStr;
        if (isKind(l, TY_INT) && isKind(r, TY_STR)) return gTypes.tStr;
        if (isKind(l, TY_BYTES) && isKind(r, TY_INT)) return gTypes.tBytes;
        if (isKind(l, TY_INT) && isKind(r, TY_BYTES)) return gTypes.tBytes;
        if (isKind(l, TY_LIST) && isKind(r, TY_INT)) return l;
        if (isKind(l, TY_INT) && isKind(r, TY_LIST)) return r;
        break;

    case OPK_DIV:
        /* int / int is float (BYTECODE.md §3.3); use // for the int result. */
        if (bothInt || anyFloat) return gTypes.tFloat;
        break;

    case OPK_FLOORDIV:
    case OPK_MOD:
        if (bothInt) return gTypes.tInt;
        if (anyFloat) return gTypes.tFloat;
        break;

    case OPK_POW:
        /* int ** int stays int. A negative exponent has no int result, so it
         * raises ValueError at run time rather than silently producing a
         * float: widening the static type to float for every power would make
         * `2 ** 8` unusable where an int is required. */
        if (bothInt) return gTypes.tInt;
        if (anyFloat) return gTypes.tFloat;
        break;

    case OPK_ADD_WRAP:
    case OPK_SUB_WRAP:
    case OPK_MUL_WRAP:
        if (bothInt) return gTypes.tInt;
        break;

    case OPK_BAND:
    case OPK_BOR:
    case OPK_BXOR:
    case OPK_SHL:
    case OPK_SHR:
        /* Bitwise operators are int-only; there is no bool arithmetic. */
        return bothInt ? gTypes.tInt : NULL;

    case OPK_MATMUL:
        break;   /* no core type defines @ */

    default:
        return NULL;
    }

    /* Fall back to an operator overload on the left operand. */
    if (typeIsUserDefined(l)) return overloadResult(l, arithDunder(op));
    return NULL;
}

JaiType *jaiTypeUnaryResult(int opKind, JaiType *operand) {
    ensureInit();
    JaiType *t = resolveAlias(operand);
    if (t == NULL) t = gTypes.tAny;

    switch ((OpKind)opKind) {
    case OPK_NEG:
        if (isKind(t, TY_INT) || isKind(t, TY_FLOAT) || isKind(t, TY_ANY)) return t;
        if (typeIsUserDefined(t)) return overloadResult(t, "__neg__");
        return NULL;

    case OPK_POS:
        if (isKind(t, TY_INT) || isKind(t, TY_FLOAT) || isKind(t, TY_ANY)) return t;
        return NULL;

    case OPK_NOT:
        /* No truthiness: `not` takes a bool and gives a bool. */
        if (isKind(t, TY_BOOL) || isKind(t, TY_ANY)) return gTypes.tBool;
        return NULL;

    case OPK_BNOT:
        if (isKind(t, TY_INT) || isKind(t, TY_ANY)) return t;
        return NULL;

    default:
        return NULL;
    }
}

/* ------------------------------------------------------------------ */
/* Generic substitution                                                 */
/* ------------------------------------------------------------------ */

JaiType *jaiTypeSubstitute(JaiType *t, const char **names, JaiType **values,
                           int count) {
    ensureInit();
    if (t == NULL || names == NULL || values == NULL || count <= 0) return t;

    switch (t->kind) {
    case TY_GENERIC_PARAM:
        for (int i = 0; i < count; i++) {
            if (nameEquals(t->name, names[i])) {
                return values[i] != NULL ? values[i] : t;
            }
        }
        return t;

    case TY_LIST: {
        JaiType *e = jaiTypeSubstitute(t->args[0], names, values, count);
        return e == t->args[0] ? t : jaiTypeList(e);
    }
    case TY_SET: {
        JaiType *e = jaiTypeSubstitute(t->args[0], names, values, count);
        return e == t->args[0] ? t : jaiTypeSet(e);
    }
    case TY_DICT: {
        JaiType *k = jaiTypeSubstitute(t->args[0], names, values, count);
        JaiType *v = jaiTypeSubstitute(t->args[1], names, values, count);
        return (k == t->args[0] && v == t->args[1]) ? t : jaiTypeDict(k, v);
    }

    case TY_TUPLE:
    case TY_FN:
    case TY_UNION:
    case TY_CLASS:
    case TY_TRAIT:
    case TY_ENUM: {
        TypeVec args;
        JAI_VEC_INIT(&args);
        bool changed = false;
        for (int i = 0; i < t->argCount; i++) {
            JaiType *a = jaiTypeSubstitute(t->args[i], names, values, count);
            if (a != t->args[i]) changed = true;
            JAI_VEC_PUSH(JaiType *, &args, a);
        }
        JaiType *ret = t->ret;
        if (t->kind == TY_FN) {
            ret = jaiTypeSubstitute(t->ret, names, values, count);
            if (ret != t->ret) changed = true;
        }
        JaiType *result = t;
        if (changed) {
            if (t->kind == TY_TUPLE) {
                result = jaiTypeTuple(args.data, args.count);
            } else if (t->kind == TY_FN) {
                result = jaiTypeFn(args.data, args.count, ret, t->fnFlags);
            } else if (t->kind == TY_UNION) {
                result = jaiTypeUnion(args.data, args.count);
            } else {
                /* A generic class/trait/enum instantiation keeps its name and
                 * declaration; only the arguments change. */
                result = internType(t->kind, t->name, args.data, args.count,
                                    t->ret, t->fnFlags, t->decl);
            }
        }
        JAI_VEC_FREE(JaiType *, &args);
        return result;
    }

    case TY_ALIAS: {
        JaiType *target = resolveAlias(t);
        if (target == NULL || target == t) return t;
        JaiType *sub = jaiTypeSubstitute(target, names, values, count);
        /* Substituting inside an alias yields the substituted target: the name
         * no longer describes the result. */
        return sub == target ? t : sub;
    }

    default:
        return t;
    }
}

/* ------------------------------------------------------------------ */
/* Rendering                                                            */
/* ------------------------------------------------------------------ */

static void renderUnion(JaiBuf *b, JaiType *t, int depth) {
    /* T | null renders as the T? sugar, except when T is a function type:
     * `fn() -> int?` would read back as a function returning int?. */
    if (t->argCount == 2) {
        JaiType *other = NULL;
        if (t->args[0]->kind == TY_NULL) other = t->args[1];
        else if (t->args[1]->kind == TY_NULL) other = t->args[0];
        if (other != NULL && other->kind != TY_FN && other->kind != TY_UNION) {
            renderType(b, other, depth + 1);
            jaiBufAppendStr(b, "?");
            return;
        }
    }

    /* Members are in intern order, which puts null first; a trailing null
     * reads better and parses the same. */
    bool first = true;
    bool hasNull = false;
    for (int i = 0; i < t->argCount; i++) {
        if (t->args[i]->kind == TY_NULL) {
            hasNull = true;
            continue;
        }
        if (!first) jaiBufAppendStr(b, " | ");
        renderType(b, t->args[i], depth + 1);
        first = false;
    }
    if (hasNull) {
        if (!first) jaiBufAppendStr(b, " | ");
        jaiBufAppendStr(b, "null");
    }
}

static void renderFn(JaiBuf *b, JaiType *t, int depth) {
    int kwIndex = (t->fnFlags & FN_FLAG_KWREST) ? t->argCount - 1 : -1;
    int varIndex = -1;
    if (t->fnFlags & FN_FLAG_VARIADIC) {
        varIndex = t->argCount - 1 - (kwIndex >= 0 ? 1 : 0);
    }

    jaiBufAppendStr(b, "fn(");
    for (int i = 0; i < t->argCount; i++) {
        if (i > 0) jaiBufAppendStr(b, ", ");
        if (i == kwIndex) jaiBufAppendStr(b, "**");
        else if (i == varIndex) jaiBufAppendStr(b, "...");
        renderType(b, t->args[i], depth + 1);
    }
    jaiBufAppendStr(b, ") -> ");
    if (t->ret == NULL) {
        jaiBufAppendStr(b, "any");
    } else if (t->ret->kind == TY_NULL) {
        /* A function returning null is what `-> void` declares (spec §6). */
        jaiBufAppendStr(b, "void");
    } else {
        renderType(b, t->ret, depth + 1);
    }
}

static void renderType(JaiBuf *b, JaiType *t, int depth) {
    if (t == NULL) {
        jaiBufAppendStr(b, "<unknown>");
        return;
    }
    if (depth > MAX_RENDER_DEPTH) {
        jaiBufAppendStr(b, "...");
        return;
    }

    switch (t->kind) {
    case TY_ANY:   jaiBufAppendStr(b, "any"); return;
    case TY_NEVER: jaiBufAppendStr(b, "never"); return;
    case TY_NULL:  jaiBufAppendStr(b, "null"); return;
    case TY_BOOL:  jaiBufAppendStr(b, "bool"); return;
    case TY_INT:   jaiBufAppendStr(b, "int"); return;
    case TY_FLOAT: jaiBufAppendStr(b, "float"); return;
    case TY_STR:   jaiBufAppendStr(b, "str"); return;
    case TY_BYTES: jaiBufAppendStr(b, "bytes"); return;
    case TY_RANGE: jaiBufAppendStr(b, "range"); return;

    case TY_LIST:
        jaiBufAppendStr(b, "list[");
        renderType(b, t->args[0], depth + 1);
        jaiBufAppendStr(b, "]");
        return;

    case TY_SET:
        jaiBufAppendStr(b, "set[");
        renderType(b, t->args[0], depth + 1);
        jaiBufAppendStr(b, "]");
        return;

    case TY_DICT:
        jaiBufAppendStr(b, "dict[");
        renderType(b, t->args[0], depth + 1);
        jaiBufAppendStr(b, ", ");
        renderType(b, t->args[1], depth + 1);
        jaiBufAppendStr(b, "]");
        return;

    case TY_TUPLE:
        jaiBufAppendStr(b, "(");
        for (int i = 0; i < t->argCount; i++) {
            if (i > 0) jaiBufAppendStr(b, ", ");
            renderType(b, t->args[i], depth + 1);
        }
        jaiBufAppendStr(b, ")");
        return;

    case TY_FN:
        renderFn(b, t, depth);
        return;

    case TY_UNION:
        renderUnion(b, t, depth);
        return;

    case TY_MODULE:
        /* Modules have no type syntax; name them so a mismatch reads clearly. */
        jaiBufAppendStr(b, "module ");
        jaiBufAppendStr(b, t->name != NULL ? t->name : "<anonymous>");
        return;

    case TY_CLASS:
    case TY_TRAIT:
    case TY_ENUM:
    case TY_GENERIC_PARAM:
    case TY_ALIAS:
        jaiBufAppendStr(b, t->name != NULL ? t->name : "<anonymous>");
        if (t->argCount > 0) {
            jaiBufAppendStr(b, "[");
            for (int i = 0; i < t->argCount; i++) {
                if (i > 0) jaiBufAppendStr(b, ", ");
                renderType(b, t->args[i], depth + 1);
            }
            jaiBufAppendStr(b, "]");
        }
        return;
    }

    jaiBufAppendStr(b, "<invalid type>");
}

const char *jaiTypeToString(JaiType *t) {
    ensureInit();

    /* A ring, so that two or three %s arguments in one printf all still point
     * at live text. Each buffer grows to fit, so nothing is ever truncated. */
    JaiBuf *b = &gStringRing[gStringRingNext];
    gStringRingNext = (gStringRingNext + 1) % STRING_RING_SIZE;

    b->count = 0;
    renderType(b, t, 0);
    jaiBufPush(b, '\0');
    b->count--;   /* keep the terminator out of the length so the next use of
                   * this slot overwrites it */
    return (const char *)b->data;
}
