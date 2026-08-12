/* object_class.c — the object model's nominal type system: ObjClass,
 * ObjTrait, ObjInstance, and ObjEnum/ObjEnumVal/ObjEnumCtor.
 *
 * These four kinds are grouped because they are the checker-known, "shaped"
 * types — the ones with a shapeId feeding an inline cache — as opposed to the
 * built-in value containers in object_collection.c. A class and an enum
 * define what a value of the type looks like (field slots vs. tag+payload);
 * an instance and an enum value are what one of those looks like at runtime.
 * jaiFreshShapeId is shared between jaiClassNew and jaiEnumNew for exactly
 * this reason: one counter, so a class shape and an enum shape can never
 * collide in a cache keyed by shapeId alone.
 *
 * The dunder cache (kDunders, setDunderField, DUNDER_COUNT) is the one piece
 * of real machinery here: jaiClassAddMethod, jaiClassInherit and
 * jaiClassRefreshDunders all have to agree on the same fixed name table, so
 * it stays a single static array rather than three copies.
 */

#include "object.h"
#include "object_internal.h"   /* pushObjRoot */

#include "gc.h"
#include "table.h"
#include "vm.h"

/* ------------------------------------------------------------------ */
/* Classes                                                              */
/* ------------------------------------------------------------------ */

/* Shape ids are the inline-cache key; 0 means "no shape", so ids start at 1
 * and are never reused. */
static uint32_t nextShapeId = 1;

uint32_t jaiFreshShapeId(void) {
    uint32_t id = nextShapeId++;
    if (nextShapeId == 0) nextShapeId = 1;   /* 0 is reserved */
    return id;
}

/* The dunder cache mirrors these method names into fixed ObjClass fields so
 * that operator dispatch never touches a hash table. */
typedef struct {
    const char *name;
    uint8_t     length;
    size_t      offset;       /* of the Value field inside ObjClass */
} DunderEntry;

#define DUNDER(field, text) \
    {(text), (uint8_t)(sizeof(text) - 1u), offsetof(ObjClass, field)}

static const DunderEntry kDunders[] = {
    DUNDER(dunderStr,      "__str__"),
    DUNDER(dunderRepr,     "__repr__"),
    DUNDER(dunderEq,       "__eq__"),
    DUNDER(dunderLt,       "__lt__"),
    DUNDER(dunderHash,     "__hash__"),
    DUNDER(dunderAdd,      "__add__"),
    DUNDER(dunderSub,      "__sub__"),
    DUNDER(dunderMul,      "__mul__"),
    DUNDER(dunderDiv,      "__div__"),
    DUNDER(dunderMod,      "__mod__"),
    DUNDER(dunderPow,      "__pow__"),
    DUNDER(dunderNeg,      "__neg__"),
    DUNDER(dunderLen,      "__len__"),
    DUNDER(dunderGetItem,  "__getitem__"),
    DUNDER(dunderSetItem,  "__setitem__"),
    DUNDER(dunderContains, "__contains__"),
    DUNDER(dunderIter,     "__iter__"),
    DUNDER(dunderNext,     "__next__"),
    DUNDER(dunderCall,     "__call__"),
};

#define DUNDER_COUNT ((int)(sizeof(kDunders) / sizeof(kDunders[0])))

static inline void setDunderField(ObjClass *c, size_t offset, Value v) {
    memcpy((char *)c + offset, &v, sizeof v);
}

ObjClass *jaiClassNew(ObjString *name, ObjClass *superclass) {
    pushObjRoot(name);
    pushObjRoot(superclass);
    ObjClass *c = JAI_ALLOCATE_OBJ(ObjClass, OBJ_CLASS);
    jaiGCPopRoots(2);

    c->name = name;
    c->qualifiedName = name;
    c->superclass = superclass;
    c->shapeId = jaiFreshShapeId();
    jaiTableInit(&c->methods);
    jaiTableInit(&c->statics);
    jaiTableInit(&c->getters);
    jaiTableInit(&c->setters);
    jaiTableInit(&c->restricted);
    c->initializer = NULL_VAL;

    for (int i = 0; i < DUNDER_COUNT; ++i)
        setDunderField(c, kDunders[i].offset, NULL_VAL);

    return c;
}

void jaiClassInherit(ObjClass *sub, ObjClass *super) {
    if (sub == NULL || super == NULL) return;
    sub->superclass = super;

    const uint16_t superCount = super->fieldCount;
    const uint16_t ownCount = sub->fieldCount;
    FieldInfo *const own = sub->fields;
    const uint32_t wide = (uint32_t)superCount + ownCount;

    if (wide > UINT16_MAX) {
        jaiThrow(vm.cRuntimeError,
                 "class '%s' would have %u fields, the limit is %u",
                 sub->name != NULL ? sub->name->chars : "?",
                 wide, UINT16_MAX);
        return;
    }

    const uint16_t total = (uint16_t)wide;
    FieldInfo *merged = NULL;

    if (total != 0) {
        merged = JAI_ALLOC(FieldInfo, total);

        if (superCount != 0)
            memcpy(merged, super->fields,
                   sizeof(FieldInfo) * (size_t)superCount);

        if (ownCount != 0) {
            memcpy(merged + superCount, own,
                   sizeof(FieldInfo) * (size_t)ownCount);

            for (uint16_t i = 0; i < ownCount; ++i)
                merged[superCount + i].slot = (uint16_t)(superCount + i);
        }
    }

    JAI_FREE_ARRAY(FieldInfo, own, ownCount);
    sub->fields = merged;
    sub->fieldCount = total;

    jaiTableAddAll(&super->methods, &sub->methods);
    jaiTableAddAll(&super->statics, &sub->statics);
    jaiTableAddAll(&super->getters, &sub->getters);
    jaiTableAddAll(&super->setters, &sub->setters);
    jaiTableAddAll(&super->restricted, &sub->restricted);

    /* Inheritance runs before the subclass declares its own methods, so the
     * fixed dunder cache can be copied directly instead of re-interning and
     * probing every dunder name. Later jaiClassAddMethod calls overwrite the
     * individual slots when the subclass declares an override. */
    for (int i = 0; i < DUNDER_COUNT; ++i) {
        Value inherited;
        memcpy(&inherited,
            (const char *)super + kDunders[i].offset,
            sizeof inherited);
        setDunderField(sub, kDunders[i].offset, inherited);
    }

    if (IS_NULL(sub->initializer))
        sub->initializer = super->initializer;
}

void jaiClassAddMethod(ObjClass *c, ObjString *name, Value method,
                       Visibility vis, uint32_t flags) {
    JaiTable *target;
    if (flags & FN_GETTER) target = &c->getters;
    else if (flags & FN_SETTER) target = &c->setters;
    else if (flags & FN_STATIC) target = &c->statics;
    else target = &c->methods;

    const uint32_t nameLength = name != NULL ? name->length : 0;
    const bool isDunder =
        name != NULL && nameLength > 4 &&
        name->chars[0] == '_' && name->chars[1] == '_' &&
        name->chars[nameLength - 2] == '_' &&
        name->chars[nameLength - 1] == '_';

    jaiGCPushRoot(OBJ_VAL(c));
    jaiGCPushRoot(method);
    pushObjRoot(name);
    (void)jaiTableSetInterned(target, name, method);

    if (vis == VIS_PUBLIC || isDunder) {
        if (c->restricted.count > 0)
            (void)jaiTableDelete(&c->restricted, OBJ_VAL(name));
    } else {
        (void)jaiTableSetInterned(
            &c->restricted, name,
            INT_VAL((int64_t)vis |
                    ((int64_t)(flags & 0xFFFFu) << 8) |
                    ((int64_t)c->shapeId << 24)));
    }

    jaiGCPopRoots(3);

    if (target != &c->methods) return;

    if ((flags & FN_INIT) ||
        (nameLength == 4 && memcmp(name->chars, "init", 4) == 0)) {
        c->initializer = method;
        return;
    }

    for (int i = 0; i < DUNDER_COUNT; ++i) {
        const DunderEntry *const dunder = kDunders + i;
        if (nameLength != dunder->length) continue;
        if (memcmp(name->chars, dunder->name, dunder->length) != 0) continue;
        setDunderField(c, dunder->offset, method);
        return;
    }
}

bool jaiClassRestrictedMethod(ObjClass *c, ObjString *name, MethodInfo *out) {
    /* The whole point of the side table: a class with no non-public method
     * answers here, before any hashing. */
    if (c == NULL || name == NULL || c->restricted.count == 0) return false;
    Value packed;
    if (!jaiTableGetInterned(&c->restricted, name, &packed)) return false;
    if (!IS_INT(packed)) return false;

    int64_t bits = AS_INT(packed);
    out->name = name;
    out->visibility = (Visibility)(bits & 0xFF);
    out->flags = (uint32_t)((bits >> 8) & 0xFFFF);

    /* `private` is private to the *declaring* class, not to whoever inherited
     * the entry, so the verdict needs that class. Its shapeId was packed in
     * when the method was added; recovering the pointer is a walk up the
     * superclass chain with no hashing at all. */
    uint32_t ownerShape = (uint32_t)((uint64_t)bits >> 24);
    out->owner = c;
    for (ObjClass *k = c; k != NULL; k = k->superclass) {
        if (k->shapeId == ownerShape) {
            out->owner = k;
            break;
        }
    }
    return true;
}

int jaiClassFieldSlot(ObjClass *c, ObjString *name) {
    const FieldInfo *info = jaiClassFieldInfo(c, name);
    return info == NULL ? -1 : (int)info->slot;
}

const FieldInfo *jaiClassFieldInfo(ObjClass *c, ObjString *name) {
    if (c == NULL || name == NULL) return NULL;

    const uint16_t count = c->fieldCount;
    const FieldInfo *const fields = c->fields;

    /* Compiler/deserializer names are canonical. For an interned lookup name,
     * equal field names must therefore be pointer-identical. */
    if (JAI_STR_INTERNED(name)) {
        for (uint16_t i = 0; i < count; ++i) {
            if (fields[i].name == name)
                return fields + i;
        }
        return NULL;
    }

    for (uint16_t i = 0; i < count; ++i) {
        const FieldInfo *const field = fields + i;
        if (field->name == name || jaiStringEquals(field->name, name))
            return field;
    }

    return NULL;
}

bool jaiClassIsSubclassOf(const ObjClass *sub, const ObjClass *super) {
    if (super == NULL) return false;
    for (const ObjClass *c = sub; c != NULL; c = c->superclass) {
        if (c == super) return true;
    }
    return false;
}

/* Depth-limited walk of a trait's supertrait DAG. The limit stops a cyclic
 * trait graph — which the checker rejects, but the VM must not hang on. */
static bool traitSatisfies(const ObjTrait *have, const ObjTrait *want,
                           int depth) {
    if (have == NULL || depth > 32) return false;
    if (have == want) return true;
    for (uint16_t i = 0; i < have->superCount; i++) {
        if (traitSatisfies(have->supers[i], want, depth + 1)) return true;
    }
    return false;
}

bool jaiClassImplements(const ObjClass *c, const ObjTrait *t) {
    if (t == NULL) return false;
    for (const ObjClass *k = c; k != NULL; k = k->superclass) {
        for (uint16_t i = 0; i < k->traitCount; i++) {
            if (traitSatisfies(k->traits[i], t, 0)) return true;
        }
    }
    return false;
}

void jaiClassRefreshDunders(ObjClass *c) {
    if (c == NULL) return;
    jaiGCPushRoot(OBJ_VAL(c));
    for (int i = 0; i < DUNDER_COUNT; i++) {
        /* Interning is idempotent: a dunder present in the table is keyed by
         * exactly this pointer, so the interned lookup is a pointer compare. */
        ObjString *name = jaiStringInternC(kDunders[i].name);
        Value m;
        if (name != NULL && jaiTableGetInterned(&c->methods, name, &m)) {
            setDunderField(c, kDunders[i].offset, m);
        } else {
            setDunderField(c, kDunders[i].offset, NULL_VAL);
        }
    }
    ObjString *init = jaiStringInternC("init");
    Value initFn;
    if (init != NULL && jaiTableGetInterned(&c->methods, init, &initFn)) {
        c->initializer = initFn;
    }
    jaiGCPopRoot();
}

/* ------------------------------------------------------------------ */
/* Traits                                                               */
/* ------------------------------------------------------------------ */

ObjTrait *jaiTraitNew(ObjString *name) {
    pushObjRoot(name);
    ObjTrait *trait = JAI_ALLOCATE_OBJ(ObjTrait, OBJ_TRAIT);
    jaiGCPopRoot();

    trait->name = name;
    jaiTableInit(&trait->required);
    jaiTableInit(&trait->defaults);
    return trait;
}

/* ------------------------------------------------------------------ */
/* Instances                                                            */
/* ------------------------------------------------------------------ */

ObjInstance *jaiInstanceNew(ObjClass *klass) {
    const uint16_t count = klass == NULL ? 0 : klass->fieldCount;

    pushObjRoot(klass);
    ObjInstance *inst = (ObjInstance *)jaiAllocateObjectRaw(
        sizeof(ObjInstance) + sizeof(Value) * (size_t)count,
        OBJ_INSTANCE);
    jaiGCPopRoot();

    inst->klass = klass;
    inst->fieldCount = count;

    for (uint16_t i = 0; i < count; ++i)
        inst->fields[i] = NULL_VAL;

    return inst;
}

/* ------------------------------------------------------------------ */
/* Enums                                                                */
/* ------------------------------------------------------------------ */

ObjEnum *jaiEnumNew(ObjString *name) {
    pushObjRoot(name);
    ObjEnum *e = JAI_ALLOCATE_OBJ(ObjEnum, OBJ_ENUM);
    jaiGCPopRoot();

    e->name = name;
    jaiTableInit(&e->methods);
    e->shapeId = jaiFreshShapeId();
    return e;
}

ObjEnumCtor *jaiEnumCtorNew(ObjEnum *e, uint16_t tag) {
    pushObjRoot(e);
    ObjEnumCtor *ctor = JAI_ALLOCATE_OBJ(ObjEnumCtor, OBJ_ENUM_CTOR);
    jaiGCPopRoot();

    ctor->type = e;
    ctor->tag = tag;
    return ctor;
}

ObjEnumVal *jaiEnumValNew(ObjEnum *e, uint16_t tag,
                          const Value *payload, int count) {
    if (count < 0) count = 0;
    if (count > 255) count = 255;

    pushObjRoot(e);
    ObjEnumVal *value = (ObjEnumVal *)jaiAllocateObjectRaw(
        sizeof(ObjEnumVal) + sizeof(Value) * (size_t)count,
        OBJ_ENUM_VAL);
    jaiGCPopRoot();

    value->type = e;
    value->tag = tag;
    value->count = (uint8_t)count;

    if (count != 0) {
        if (payload != NULL) {
            memcpy(value->payload, payload,
                   sizeof(Value) * (size_t)count);
        } else {
            for (int i = 0; i < count; ++i)
                value->payload[i] = NULL_VAL;
        }
    }

    return value;
}
