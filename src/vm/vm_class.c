/* vm_class.c — the object model as the interpreter sees it: visibility,
 * method and property lookup on every receiver shape, the runtime type tests,
 * and building a class, trait or enum from its spec constant.
 */
#include "vm/vm_internal.h"
#include "runtime/runtime.h"

/* ------------------------------------------------------------------ */
/* Visibility                                                           */
/*                                                                      */
/* The checker rejects what it can prove (E0701); this is the dynamic    */
/* half, for `any`-typed receivers.                                      */
/* ------------------------------------------------------------------ */

static bool accessPermitted(const ObjClass *owner, Visibility vis) {
    if (vis == VIS_PUBLIC) return true;
    CallFrame *frame = topFrame();
    if (frame == NULL || owner == NULL) return false;

    ObjClass *selfClass = NULL;
    if (frame->closure != NULL && frame->closure->fn != NULL)
        selfClass = frame->closure->fn->owner;
    if (selfClass == NULL) {
        Value self = frame->slots[0];
        if (IS_INSTANCE(self))   selfClass = AS_INSTANCE(self)->klass;
        else if (IS_CLASS(self)) selfClass = AS_CLASS(self);
    }
    if (selfClass == NULL) return false;

    if (vis == VIS_PROTECTED) {
        return jaiClassIsSubclassOf(selfClass, owner) ||
               jaiClassIsSubclassOf(owner, selfClass);
    }
    /* Private: the executing method must belong to the declaring class or to
     * a subclass that inherited the field slot. */
    return jaiClassIsSubclassOf(selfClass, owner);
}

/* The class a field was declared on, for the visibility test above. Walks up
 * from `klass` to the topmost class that still declares the slot. */
/* Callers must gate this on the field being non-public. It walks the whole
 * superclass chain running a linear field scan at every level, and
 * accessPermitted discards it for VIS_PUBLIC -- but C evaluates arguments
 * before the call, so writing it as an argument paid for it on every public
 * field read. That was ~11% of a compile in getPropertyInto. */
static const ObjClass *fieldOwner(ObjClass *klass, ObjString *name) {
    const ObjClass *owner = klass;
    for (ObjClass *k = klass; k != NULL; k = k->superclass) {
        if (jaiClassFieldInfo(k, name) != NULL) owner = k;
    }
    return owner;
}

/* The dynamic half of spec §7.1 for methods. Returns true when `name` may be
 * reached from the running frame; when it may not, raises AttributeError (or
 * stays quiet if `raise` is false) and returns false. Public and unknown names
 * — everything, in a class that declares no non-public method — cost one load
 * inside jaiClassRestrictedMethod. */
bool methodPermitted(ObjClass *klass, ObjString *name, bool raise) {
    MethodInfo mi;
    if (!jaiClassRestrictedMethod(klass, name, &mi)) return true;
    if (accessPermitted(mi.owner, mi.visibility)) return true;
    if (!raise) return false;
    return jaiThrow(vm.cAttributeError, "method '%.*s' of class '%s' is %s",
                    (int)name->length, name->chars,
                    mi.owner != NULL && mi.owner->name != NULL
                        ? mi.owner->name->chars : "?",
                    mi.visibility == VIS_PROTECTED ? "protected" : "private");
}

/* ------------------------------------------------------------------ */
/* Method and property lookup                                           */
/* ------------------------------------------------------------------ */

/* Trait default implementations are copied into the class at OP_IMPL_TRAIT;
 * this walk is the safety net for classes built by other means. */
static bool findTraitDefault(const ObjClass *klass, ObjString *name, Value *out) {
    for (const ObjClass *k = klass; k != NULL; k = k->superclass) {
        for (uint16_t i = 0; i < k->traitCount; i++) {
            ObjTrait *t = k->traits[i];
            if (t == NULL) continue;
            if (jaiTableGetInterned(&t->defaults, name, out)) return true;
            for (uint16_t j = 0; j < t->superCount; j++) {
                if (t->supers[j] != NULL &&
                    jaiTableGetInterned(&t->supers[j]->defaults, name, out)) {
                    return true;
                }
            }
        }
    }
    return false;
}

/* The raw (unbound) method for `name` on `klass`, including inherited and
 * trait-default methods. */
bool findMethod(ObjClass *klass, ObjString *name, Value *out) {
    if (klass == NULL || name == NULL) return false;
    if (jaiTableGetInterned(&klass->methods, name, out)) return true;
    if (jaiTableGetInterned(&klass->statics, name, out)) return true;
    return findTraitDefault(klass, name, out);
}

/* Index of the variant named `name`, or -1. */
int jaiEnumVariantIndex(const ObjEnum *e, const ObjString *name) {
    if (e == NULL || name == NULL) return -1;
    for (uint16_t i = 0; i < e->variantCount; i++) {
        ObjString *have = e->variants[i].name;
        if (have == name || jaiStringEquals(have, name)) return (int)i;
    }
    return -1;
}

/* Zero-arity enum variant, or the enum's own method, for `Color.Red`.
 *
 * The variant's one value is cached on the variant. Building a fresh one per
 * mention made `x is Token.LParen` false for an `x` that *was* `Token.LParen`
 * — `is` is identity (spec §4.2) — and allocated on a path as hot as a parser's
 * token test. A payload-less variant has no state to distinguish instances by,
 * so one value is all there can be. */
bool enumMember(ObjEnum *e, ObjString *name, Value *out) {
    if (jaiTableGetInterned(&e->methods, name, out)) return true;
    for (uint16_t i = 0; i < e->variantCount; i++) {
        EnumVariant *v = &e->variants[i];
        if (v->name != name && !jaiStringEquals(v->name, name)) continue;
        if (v->arity != 0) {
            /* Not a value yet, but a function of its payload — and the checker
             * types it as exactly that, so `let make = Shape.Circle` has to
             * work rather than report a variant that plainly exists as
             * missing. The direct-call sites could special-case this, and one
             * of them did; a tail call could not, because the callee is a
             * value by then. */
            if (v->ctor == NULL) v->ctor = jaiEnumCtorNew(e, i);
            *out = OBJ_VAL(v->ctor);
            return true;
        }
        if (v->unit == NULL) v->unit = jaiEnumValNew(e, i, NULL, 0);
        *out = OBJ_VAL(v->unit);
        return true;
    }
    return false;
}

/* A module member, honouring the export list. A module with no explicit
 * exports (no `export` block and no `pub`) exposes everything: restricting it
 * would make an un-annotated module unusable rather than encapsulated. */
bool moduleMember(ObjModule *m, ObjString *name, Value *out,
                         bool *outHidden) {
    *outHidden = false;
    if (!jaiModuleGet(m, name, out)) return false;
    if (m->exports.count > 0 && !jaiModuleIsExported(m, name)) {
        *outHidden = true;
        return false;
    }
    return true;
}

/* Shared implementation of `receiver.name` for reads. `ic`, when non-NULL, is
 * filled on a plain instance-field hit so the next execution skips all of
 * this. `raise` selects between "AttributeError" and "quietly false".
 *
 * `name` here is not always the compiler's own interned identifier: get_field
 * and get_method (builtins_prim.c) call in with whatever ObjString the
 * program passed as the field name, which can be a view into a growing
 * concatenation buffer whose NUL a later, unrelated append moved past
 * (object_string.c). Every message below takes name->length instead of
 * trusting %s to find where it ends. */
bool getPropertyInto(Value receiver, ObjString *name, Value *out,
                            bool raise, InlineCache *ic) {
    if (name == NULL) return false;
#ifdef JAI_PROP_STATS
    { extern uint64_t jaiPropRecv[]; jaiPropRecv[IS_OBJ(receiver) ? 8 + (int)AS_OBJ(receiver)->type : (int)receiver.type]++; }
#endif

    if (IS_INSTANCE(receiver)) {
        ObjInstance *inst = AS_INSTANCE(receiver);
        ObjClass *klass = inst->klass;

        /* Getters win over the raw field: a property exists precisely to
         * intercept the access (spec §7.1). */
        Value getter;
        if (klass != NULL && jaiTableGetInterned(&klass->getters, name, &getter)) {
            if (!methodPermitted(klass, name, raise)) return false;
            return jaiCallValue(OBJ_VAL(jaiBoundNew(receiver, getter)), 0, NULL,
                                out);
        }

        const FieldInfo *info = jaiClassFieldInfo(klass, name);
        if (info != NULL) {
            if (info->visibility != VIS_PUBLIC &&
                !accessPermitted(fieldOwner(klass, name), info->visibility)) {
                if (!raise) return false;
                return jaiThrow(vm.cAttributeError,
                                "field '%.*s' of class '%s' is private",
                                (int)name->length, name->chars,
                                klass->name != NULL ? klass->name->chars : "?");
            }
            if (info->slot >= inst->fieldCount) {
                if (!raise) return false;
                return jaiThrow(vm.cAttributeError,
                                "field '%.*s' is not present on this instance",
                                (int)name->length, name->chars);
            }
            if (ic != NULL && klass != NULL) {
                /* Only monomorphic-to-polymorphic growth; a megamorphic site
                 * stops caching rather than thrashing four ways forever. */
                if (ic->state == IC_MEGA) {
                    /* leave it alone */
                } else if (ic->count < JAI_IC_WAYS) {
                    ic->shapeId[ic->count] = klass->shapeId;
                    ic->payload[ic->count] = info->slot;
                    ic->cached[ic->count] = NULL_VAL;
                    ic->count++;
                    ic->state = (ic->count == 1) ? IC_MONO : IC_POLY;
                } else {
                    ic->state = IC_MEGA;
                }
            }
            *out = inst->fields[info->slot];
            return true;
        }

        Value method;
        if (findMethod(klass, name, &method)) {
            if (!methodPermitted(klass, name, raise)) return false;
            *out = OBJ_VAL(jaiBoundNew(receiver, method));
            return true;
        }
        if (!raise) return false;
        return jaiThrow(vm.cAttributeError, "'%s' object has no attribute '%.*s'",
                        klass != NULL && klass->name != NULL ? klass->name->chars
                                                             : "instance",
                        (int)name->length, name->chars);
    }

    if (IS_CLASS(receiver)) {
        ObjClass *klass = AS_CLASS(receiver);
        if (jaiTableGetInterned(&klass->statics, name, out) ||
            jaiTableGetInterned(&klass->methods, name, out)) {
            return methodPermitted(klass, name, raise);
        }
        if (!raise) return false;
        return jaiThrow(vm.cAttributeError, "class '%s' has no member '%.*s'",
                        klass->name != NULL ? klass->name->chars : "?",
                        (int)name->length, name->chars);
    }

    if (IS_MODULE(receiver)) {
        ObjModule *m = AS_MODULE(receiver);
        bool hidden = false;
        if (moduleMember(m, name, out, &hidden)) return true;
        if (!raise) return false;
        if (hidden) {
            return jaiThrow(vm.cImportError,
                            "'%.*s' is not exported by module '%s'",
                            (int)name->length, name->chars,
                            m->name != NULL ? m->name->chars : "?");
        }
        /* Reading `mod.members` without calling it has to find the same
         * introspection helpers `mod.members()` does. */
        if (jaiBuiltinMethod(receiver, name, out)) return true;
        return jaiThrow(vm.cAttributeError, "module '%s' has no member '%.*s'",
                        m->name != NULL ? m->name->chars : "?",
                        (int)name->length, name->chars);
    }

    if (IS_ENUM(receiver)) {
        if (enumMember(AS_ENUM(receiver), name, out)) {
            /* Safe to memoise: a variant's unit value and ctor are created
             * once and kept on the variant, the methods table is filled only
             * while the enum is being defined, and any addition bumps shapeId.
             * The GC marks ic->cached (gc.c:201), so this is a strong
             * reference and cannot dangle. */
            if (ic != NULL && ic->state != IC_MEGA) {
                if (ic->count < JAI_IC_WAYS) {
                    ic->shapeId[ic->count] = AS_ENUM(receiver)->shapeId;
                    ic->payload[ic->count] = 0;
                    ic->cached[ic->count] = *out;
                    ic->count++;
                    ic->state = (ic->count == 1) ? IC_MONO : IC_POLY;
                } else {
                    ic->state = IC_MEGA;
                }
            }
            return true;
        }
        if (!raise) return false;
        return jaiThrow(vm.cAttributeError, "enum '%s' has no variant '%.*s'",
                        AS_ENUM(receiver)->name != NULL
                            ? AS_ENUM(receiver)->name->chars : "?",
                        (int)name->length, name->chars);
    }

    if (IS_ENUM_VAL(receiver)) {
        ObjEnumVal *ev = AS_ENUM_VAL(receiver);
        if (ev->type != NULL && ev->tag < ev->type->variantCount) {
            EnumVariant *variant = &ev->type->variants[ev->tag];
            for (uint8_t i = 0; i < variant->arity && i < ev->count; i++) {
                if (variant->fieldNames == NULL) break;
                if (variant->fieldNames[i] == name ||
                    jaiStringEquals(variant->fieldNames[i], name)) {
                    *out = ev->payload[i];
                    return true;
                }
            }
        }
        if (ev->type != NULL && jaiTableGetInterned(&ev->type->methods, name, out)) {
            *out = OBJ_VAL(jaiBoundNew(receiver, *out));
            return true;
        }
        /* After the enum's own members, so a user method of the same name
         * wins: the builtin table is what every enum value has by default. */
        if (jaiBuiltinMethod(receiver, name, out)) return true;
        if (!raise) return false;
        return jaiThrow(vm.cAttributeError, "'%s' has no member '%.*s'",
                        jaiTypeNameStatic(receiver), (int)name->length, name->chars);
    }

    /* str, list, dict, ... : the built-in method tables hand back a bound
     * native, so the call path downstream is identical to a user method. */
    if (jaiBuiltinMethod(receiver, name, out)) return true;
    if (!raise) return false;
    return jaiThrow(vm.cAttributeError, "'%s' object has no attribute '%.*s'",
                    jaiTypeNameStatic(receiver), (int)name->length, name->chars);
}

bool jaiGetProperty(Value receiver, ObjString *name, Value *out) {
    return getPropertyInto(receiver, name, out, true, NULL);
}

bool throwFieldKind(const FieldInfo *info, Value v) {
    return jaiThrow(vm.cTypeError,
                    "cannot assign %s to field '%s' declared %s",
                    jaiTypeNameStatic(v), info->name->chars,
                    jaiFieldKindName(info->typeId));
}

/* Mirrors getPropertyInto's note: set_field (builtins_prim.c) reaches here
 * with a program-computed name too, so every message below takes
 * name->length rather than trust %s to find where it ends. */
bool jaiSetProperty(Value receiver, ObjString *name, Value value) {
    if (name == NULL) return false;

    if (IS_INSTANCE(receiver)) {
        ObjInstance *inst = AS_INSTANCE(receiver);
        ObjClass *klass = inst->klass;

        Value setter;
        if (klass != NULL && jaiTableGetInterned(&klass->setters, name, &setter)) {
            /* A property is a method; a private one is no more assignable from
             * outside than a private field is. */
            if (!methodPermitted(klass, name, true)) return false;
            Value ignored;
            Value arg = value;
            return jaiCallValue(OBJ_VAL(jaiBoundNew(receiver, setter)), 1, &arg,
                                &ignored);
        }

        const FieldInfo *info = jaiClassFieldInfo(klass, name);
        if (info == NULL) {
            /* Spec §7.1: fields are declared, never conjured by assignment. */
            return jaiThrow(vm.cAttributeError,
                            "'%s' object has no field '%.*s'; fields must be "
                            "declared in the class body",
                            klass != NULL && klass->name != NULL
                                ? klass->name->chars : "instance",
                            (int)name->length, name->chars);
        }
        if (info->visibility != VIS_PUBLIC &&
                !accessPermitted(fieldOwner(klass, name), info->visibility)) {
            return jaiThrow(vm.cAttributeError,
                            "field '%.*s' of class '%s' is private",
                            (int)name->length, name->chars,
                            klass->name != NULL ? klass->name->chars : "?");
        }
        if (info->slot >= inst->fieldCount) {
            return jaiThrow(vm.cAttributeError,
                            "field '%.*s' is not present on this instance",
                            (int)name->length, name->chars);
        }
        /* The checker could not have caught this: reaching a field through an
         * `any` receiver means it did not know the class, so it did not know
         * what the field was declared as. FieldInfo is the only place that
         * knows, so the runtime is the only place this can be rejected. */
        if (!jaiKindAccepts(info->typeId, value)) {
            return throwFieldKind(info, value);
        }
        inst->fields[info->slot] = value;
        return true;
    }

    if (IS_CLASS(receiver)) {
        ObjClass *klass = AS_CLASS(receiver);
        Value existing;
        if (!jaiTableGetInterned(&klass->statics, name, &existing)) {
            return jaiThrow(vm.cAttributeError,
                            "class '%s' has no static member '%.*s'",
                            klass->name != NULL ? klass->name->chars : "?",
                            (int)name->length, name->chars);
        }
        jaiGCPushRoot(receiver);
        jaiGCPushRoot(value);
        (void)jaiTableSetInterned(&klass->statics, name, value);
        jaiGCPopRoots(2);
        return true;
    }

    if (IS_MODULE(receiver)) {
        ObjModule *m = AS_MODULE(receiver);
        Value existing;
        if (!jaiModuleGet(m, name, &existing)) {
            return jaiThrow(vm.cAttributeError, "module '%s' has no member '%.*s'",
                            m->name != NULL ? m->name->chars : "?",
                            (int)name->length, name->chars);
        }
        jaiModuleSet(m, name, value);
        return true;
    }

    return jaiThrow(vm.cAttributeError,
                    "cannot set attribute '%.*s' on a '%s' value",
                    (int)name->length, name->chars,
                    jaiTypeNameStatic(receiver));
}

/* ------------------------------------------------------------------ */
/* Type tests                                                           */
/* ------------------------------------------------------------------ */

/* Does `value` satisfy the type named by a constant? The constant is a class,
 * a trait, an enum, or — when the type was not resolvable at compile time —
 * the spelling of a primitive type. */
bool valueMatchesType(Value value, Value typeConstant) {
    if (IS_CLASS(typeConstant)) {
        ObjClass *expected = AS_CLASS(typeConstant);
        if (IS_INSTANCE(value)) {
            return jaiClassIsSubclassOf(AS_INSTANCE(value)->klass, expected);
        }
        return IS_CLASS(value) && jaiClassIsSubclassOf(AS_CLASS(value), expected);
    }
    if (IS_TRAIT(typeConstant)) {
        return IS_INSTANCE(value) &&
               jaiClassImplements(AS_INSTANCE(value)->klass, AS_TRAIT(typeConstant));
    }
    if (IS_ENUM(typeConstant)) {
        return IS_ENUM_VAL(value) && AS_ENUM_VAL(value)->type == AS_ENUM(typeConstant);
    }
    if (IS_STRING(typeConstant)) {
        ObjString *name = AS_STRING(typeConstant);
        if (name->length == 3 && memcmp(name->chars, "any", 3) == 0) return true;
        /* `Enum.Variant`: codegen writes this when the enum came from another
         * module, so its tag numbering was not visible where the pattern was
         * compiled. The variant list is on the runtime enum, so resolve it
         * here. */
        const char *dot = memchr(name->chars, '.', name->length);
        if (dot != NULL) {
            if (!IS_ENUM_VAL(value)) return false;
            ObjEnumVal *ev = AS_ENUM_VAL(value);
            size_t typeLen = (size_t)(dot - name->chars);
            if (ev->type == NULL || ev->type->name == NULL ||
                ev->type->name->length != typeLen ||
                memcmp(ev->type->name->chars, name->chars, typeLen) != 0)
                return false;
            const char *want = dot + 1;
            size_t wantLen = name->length - typeLen - 1;
            if (ev->tag >= ev->type->variantCount) return false;
            ObjString *have = ev->type->variants[ev->tag].name;
            return have != NULL && have->length == wantLen &&
                   memcmp(have->chars, want, wantLen) == 0;
        }
        /* A declared type is a name here because codegen cannot see the
         * runtime object; resolving it now is what makes a guard accept a
         * subclass or a trait implementer rather than only the exact name. */
        Value declared;
        CallFrame *frame = topFrame();
        if ((frame != NULL && frame->module != NULL &&
             jaiModuleGet(frame->module, name, &declared)) ||
            (vm.builtins != NULL && jaiModuleGet(vm.builtins, name, &declared))) {
            if (IS_CLASS(declared) || IS_TRAIT(declared) || IS_ENUM(declared)) {
                return valueMatchesType(value, declared);
            }
        }
        return strcmp(jaiTypeNameStatic(value), name->chars) == 0;
    }
    if (IS_TUPLE(typeConstant)) {          /* a union type */
        ObjTuple *members = AS_TUPLE(typeConstant);
        for (uint32_t i = 0; i < members->count; i++) {
            if (valueMatchesType(value, members->items[i])) return true;
        }
        return false;
    }
    if (IS_NULL(typeConstant)) return IS_NULL(value);
    return false;
}

/* The builtin type names (spec §9) are bound as conversion *functions*, so
 * `x is int` has an ObjNative on its right and there is no runtime type object
 * to compare against. Recognised by identity against the binding in
 * `vm.builtins`, not by name alone, so a user value that happens to be called
 * `int` is still just a value. */
static bool nativeNamesType(ObjNative *native) {
    static const char *const kTypeNames[] = {
        "int", "float", "str", "bool", "bytes",
        "list", "dict", "set", "tuple", "range",
    };
    if (native->name == NULL || vm.builtins == NULL) return false;
    for (size_t i = 0; i < sizeof kTypeNames / sizeof kTypeNames[0]; i++) {
        if (strcmp(native->name->chars, kTypeNames[i]) != 0) continue;
        Value bound;
        return jaiModuleGet(vm.builtins, native->name, &bound) &&
               IS_NATIVE(bound) && AS_NATIVE(bound) == native;
    }
    return false;
}

/* `is` compares identity (spec §4.2) *except* when its right operand denotes a
 * type, which is the reading the checker narrows on — see `typeOperand` in
 * sema/check.c, whose comment says the type reading wins over the value one.
 * Without this the two disagree: every `if x is C` guard narrowed statically
 * and then took the false branch at run time. Deciding it here rather than in
 * codegen keeps a class reached through an import, an alias or a variable on
 * the same path as one named directly, and keeps both front ends honest. */
bool valueIsTest(Value subject, Value target) {
    if (IS_CLASS(target) || IS_TRAIT(target) || IS_ENUM(target)) {
        return valueMatchesType(subject, target);
    }
    if (IS_NATIVE(target) && nativeNamesType(AS_NATIVE(target))) {
        return valueMatchesType(subject, OBJ_VAL(AS_NATIVE(target)->name));
    }
    return jaiValuesIdentical(subject, target);
}

const char *typeConstantName(Value typeConstant) {
    if (IS_CLASS(typeConstant) && AS_CLASS(typeConstant)->name != NULL) {
        return AS_CLASS(typeConstant)->name->chars;
    }
    if (IS_TRAIT(typeConstant) && AS_TRAIT(typeConstant)->name != NULL) {
        return AS_TRAIT(typeConstant)->name->chars;
    }
    if (IS_ENUM(typeConstant) && AS_ENUM(typeConstant)->name != NULL) {
        return AS_ENUM(typeConstant)->name->chars;
    }
    if (IS_STRING(typeConstant)) return AS_CSTRING(typeConstant);
    return jaiTypeNameStatic(typeConstant);
}

bool classDeclareField(ObjClass *klass, ObjString *name, uint8_t info) {
    if (jaiClassFieldInfo(klass, name) != NULL) return true;   /* redeclared */
    if (klass->fieldCount == UINT16_MAX) {
        return jaiThrow(vm.cRuntimeError, "class '%s' has too many fields",
                        klass->name != NULL ? klass->name->chars : "?");
    }
    uint16_t oldCount = klass->fieldCount;
    klass->fields = JAI_GROW_ARRAY(FieldInfo, klass->fields, oldCount, oldCount + 1);
    FieldInfo *field = &klass->fields[oldCount];
    field->name = name;
    field->slot = oldCount;
    field->visibility = (Visibility)(info & 0x3);
    field->isStatic = (info & 0x4) != 0;
    field->isLet = (info & 0x8) != 0;
    /* Bits 4-7 are the declared kind (spec §3.7). Zero -- which is every image
     * written before this encoding existed -- is FIELD_KIND_ANY. */
    field->typeId = (uint32_t)((info >> 4) & 0xF);
    klass->fieldCount = (uint16_t)(oldCount + 1);

    /* A static field lives on the class, not in an instance window, so it needs
     * an entry in `statics` before anything can read or assign it: both
     * jaiSetProperty and the class-member read refuse a name that is not
     * already there. Codegen stores its initialiser right after this. */
    if (field->isStatic) {
        jaiGCPushRoot(OBJ_VAL(klass));
        (void)jaiTableSetInterned(&klass->statics, name, NULL_VAL);
        jaiGCPopRoot();
    }
    return true;
}

/* The class-spec constant that codegen emits for OP_CLASS is a 6-tuple
 *     (INT kind, STR name, STR|NULL superName, BOOL isAbstract,
 *      TUPLE required, TUPLE variants)
 * where kind selects class/trait/enum (spec/BYTECODE.md §6). `required` and
 * `variants` are flattened pair lists; superName is informational only,
 * because the link is made by the OP_INHERIT that codegen emits right after.
 *
 * This tuple is the whole spec. There is no separate on-disk class record: it
 * serialises as an ordinary tuple constant, and the fields, methods and traits
 * it does not carry arrive as FIELD_DEF, METHOD and IMPL_TRAIT in the class
 * body, so a class is built by running code. */
#define SPEC_KIND_CLASS 0
#define SPEC_KIND_TRAIT 1
#define SPEC_KIND_ENUM  2

/* Builds the runtime object a class spec describes. Returns NULL and leaves an
 * exception pending on a malformed spec. */
Obj *classSpecInstantiate(Value spec) {
    if (!IS_TUPLE(spec) || AS_TUPLE(spec)->count != 6) {
        jaiThrow(vm.cRuntimeError, "CLASS operand is not a class spec");
        return NULL;
    }
    const Value *items = AS_TUPLE(spec)->items;
    if (!IS_INT(items[0]) || !IS_STRING(items[1])) {
        jaiThrow(vm.cRuntimeError, "CLASS operand is not a class spec");
        return NULL;
    }
    int64_t kind = AS_INT(items[0]);
    ObjString *name = AS_STRING(items[1]);

    switch (kind) {
    case SPEC_KIND_CLASS: {
        ObjClass *klass = jaiClassNew(name, NULL);
        klass->isAbstract = IS_BOOL(items[3]) && AS_BOOL(items[3]);
        return (Obj *)klass;
    }

    case SPEC_KIND_TRAIT: {
        if (!IS_TUPLE(items[4])) {
            jaiThrow(vm.cRuntimeError, "trait spec has no requirement list");
            return NULL;
        }
        ObjTrait *trait = jaiTraitNew(name);
        jaiGCPushRoot(OBJ_VAL(trait));
        ObjTuple *required = AS_TUPLE(items[4]);
        for (uint32_t i = 0; i + 1 < required->count; i += 2) {
            if (!IS_STRING(required->items[i]) ||
                !IS_INT(required->items[i + 1])) {
                continue;
            }
            jaiTableSet(&trait->required, required->items[i],
                        required->items[i + 1]);
        }
        jaiGCPopRoot();
        return (Obj *)trait;
    }

    case SPEC_KIND_ENUM: {
        if (!IS_TUPLE(items[5])) {
            jaiThrow(vm.cRuntimeError, "enum spec has no variant list");
            return NULL;
        }
        ObjTuple *variants = AS_TUPLE(items[5]);
        uint32_t count = variants->count / 2;
        if (count > UINT16_MAX) {
            jaiThrow(vm.cRuntimeError, "enum '%s' has too many variants",
                     name->chars);
            return NULL;
        }
        ObjEnum *e = jaiEnumNew(name);
        jaiGCPushRoot(OBJ_VAL(e));
        /* Variant names and field names are already interned constants held
         * live by the chunk, so only the arrays need allocating here. */
        e->variants = count > 0 ? JAI_ALLOC(EnumVariant, count) : NULL;
        e->variantCount = (uint16_t)count;
        for (uint32_t i = 0; i < count; i++) {
            EnumVariant *v = &e->variants[i];
            Value vname = variants->items[i * 2];
            Value fields = variants->items[i * 2 + 1];
            v->name = IS_STRING(vname) ? AS_STRING(vname) : name;
            uint32_t arity = IS_TUPLE(fields) ? AS_TUPLE(fields)->count : 0;
            if (arity > 255) arity = 255;
            v->arity = (uint8_t)arity;
            v->unit = NULL;   /* both filled on the variant's first mention */
            v->ctor = NULL;
            v->fieldNames = arity > 0 ? JAI_ALLOC(ObjString *, arity) : NULL;
            for (uint32_t p = 0; p < arity; p++) {
                Value fname = AS_TUPLE(fields)->items[p];
                v->fieldNames[p] = IS_STRING(fname) ? AS_STRING(fname) : NULL;
            }
        }
        jaiGCPopRoot();
        return (Obj *)e;
    }

    default:
        jaiThrow(vm.cRuntimeError, "unknown class-spec kind %lld",
                 (long long)kind);
        return NULL;
    }
}
