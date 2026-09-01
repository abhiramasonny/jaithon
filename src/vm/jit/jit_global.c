/* jit_global.c -- resolving globals, static fields and module members, the guards
 * that keep those resolutions honest, and the slot-kind inference built on them. */

#include "vm/jit/jit_arm64.h"
/* For jaiJitFieldReadFor: which builtins are one load from their receiver. */
#include "vm/jit/jit_field_read.h"
/* For jaiBuiltinMethod: resolving `xs.len()` to a native needs the runtime's name table. */
/* For jaiOpBranchOperandAt: says which opcodes carry a branch target. */
#include "vm/vm.h"

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "vm/jit/jit_internal.h"

#if (defined(__aarch64__) || defined(__arm64__))

ObjClass *globalClass(ObjClosure *closure, uint32_t nameIdx) {
    ObjFunction *fn = closure->fn;
    if (fn->module == NULL) return NULL;
    if (nameIdx >= (uint32_t)fn->chunk.constants.count) return NULL;
    Value name = fn->chunk.constants.data[nameIdx];
    if (!IS_STRING(name)) return NULL;
    Value bound;
    if (!jaiModuleGet(fn->module, AS_STRING(name), &bound)) return NULL;
    return IS_CLASS(bound) ? AS_CLASS(bound) : NULL;
}

/* The first entry a dict walk would yield, for the component kinds the compiled
 * pair head specialises on -- the dict equivalent of taking items[0] off a list.
 * False for a dict with nothing live in it, which declines rather than guessing.
 * Reads the order array exactly as jaiTableNext does, so "first" here and
 * "first" at run time are the same entry. */
bool firstLiveEntry(const JaiTable *t, Value *key, Value *value) {
    if (t->entries == NULL) return false;
    for (int i = 0; i < t->orderCount; i++) {
        const int32_t slot = t->order[i];
        if (slot < 0) continue;
        *key   = t->entries[slot].key;
        *value = t->entries[slot].value;
        return true;
    }
    return false;
}

ObjFunction *globalFunction(ObjClosure *closure, uint32_t nameIdx,
                                   Value *out) {
    ObjFunction *fn = closure->fn;
    if (fn->module == NULL) return NULL;
    if (nameIdx >= (uint32_t)fn->chunk.constants.count) return NULL;
    Value name = fn->chunk.constants.data[nameIdx];
    if (!IS_STRING(name)) return NULL;
    Value bound;
    if (!jaiModuleGet(fn->module, AS_STRING(name), &bound)) return NULL;
    if (!IS_CLOSURE(bound)) return NULL;
    *out = bound;
    return AS_CLOSURE(bound)->fn;
}

/* Resolved the way the interpreter resolves a builtin: the module first, `vm.builtins` only when the
 * module has no such name -- a module-level binding is never mistaken for it, and if one appears later the module's version retires this compiled form. */
ObjNative *globalNative(ObjClosure *closure, uint32_t nameIdx,
                               Value *out) {
    ObjFunction *fn = closure->fn;
    if (fn->module == NULL || vm.builtins == NULL) return NULL;
    if (nameIdx >= (uint32_t)fn->chunk.constants.count) return NULL;
    Value name = fn->chunk.constants.data[nameIdx];
    if (!IS_STRING(name)) return NULL;
    Value shadow;
    if (jaiModuleGet(fn->module, AS_STRING(name), &shadow)) return NULL;
    Value bound;
    if (!jaiModuleGet(vm.builtins, AS_STRING(name), &bound)) return NULL;
    if (!IS_NATIVE(bound)) return NULL;
    *out = bound;
    return AS_NATIVE(bound);
}

/* `__prim__`, resolved the same way globalNative resolves a bare builtin --
 * the module first, so a real user binding is never mistaken for it -- except
 * what sits at `vm.builtins`'s name is an ObjModule (a native namespace:
 * jaiDefineNative's dotted names build one the first time a "ns.leaf" name
 * registers, in namespaceFor/makeNamespace, builtins.c), not an ObjNative.
 * `math.sqrt`'s own body is `__prim__.f64_sqrt(x)` -- lib/std/math.jai:218 --
 * so this is not a hypothetical name, it is the one OP_GET_GLOBAL's own
 * refusal message already names as "not a compiled global function" on every
 * `__prim__.f64_*` body in the tree.
 *
 * Resolved BY VALUE, same as globalNative: nothing here is re-checked at run
 * time beyond fn->module->version at entry (guards a later shadow), so this
 * carries exactly the soundness globalNative already carries for a bare
 * builtin -- no better, no worse. What DOES need a per-call guard is the
 * MEMBER read through this namespace afterward, because `mod.attr = v` is
 * real syntax for any module receiver (jaiSetProperty's IS_MODULE arm) and
 * `__prim__` is reachable as a bare identifier -- see emitModuleNativeCall's
 * m->version check for that half. */
ObjModule *globalNamespace(ObjClosure *closure, uint32_t nameIdx) {
    ObjFunction *fn = closure->fn;
    if (fn->module == NULL || vm.builtins == NULL) return NULL;
    if (nameIdx >= (uint32_t)fn->chunk.constants.count) return NULL;
    Value name = fn->chunk.constants.data[nameIdx];
    if (!IS_STRING(name)) return NULL;
    Value shadow;
    if (jaiModuleGet(fn->module, AS_STRING(name), &shadow)) return NULL;
    Value bound;
    if (!jaiModuleGet(vm.builtins, AS_STRING(name), &bound)) return NULL;
    if (!IS_MODULE(bound)) return NULL;
    return AS_MODULE(bound);
}

bool globalIsSelf(ObjClosure *closure, uint32_t nameIdx) {
    ObjFunction *fn = closure->fn;
    if (fn->module == NULL) return false;
    if (nameIdx >= (uint32_t)fn->chunk.constants.count) return false;
    Value name = fn->chunk.constants.data[nameIdx];
    if (!IS_STRING(name)) return false;

    Value bound;
    if (!jaiModuleGet(fn->module, AS_STRING(name), &bound)) return false;
    return IS_CLOSURE(bound) && AS_CLOSURE(bound)->fn == fn;
}

/* ------------------------------------------------------------------ */
/* Module globals                                                       */
/* ------------------------------------------------------------------ */

/* A JaiEntry's address is stable as long as the table doesn't rehash/delete/clear -- overwriting an
 * existing global never moves it (ensureRoom only runs for a NEW key) -- which is what lets a compiled load be one `ldr` from a baked pointer. `keyVersion` counts every event that breaks this; emitGlobalsGuard checks it. */
JaiEntry *globalSlot(Emit *e, ObjClosure *closure, uint32_t nameIdx,
                            Value *out) {
    ObjFunction *fn = closure->fn;
    if (fn->module == NULL) return NULL;
    if (nameIdx >= (uint32_t)fn->chunk.constants.count) return NULL;
    Value name = fn->chunk.constants.data[nameIdx];
    if (!IS_STRING(name)) return NULL;
    JaiTable *t = &fn->module->globals;
    JaiEntry *slot = jaiTableFindEntryInterned(t, AS_STRING(name));
    if (slot == NULL) return NULL;
    /* The whole body reads one table, so one guard covers every slot. */
    if (e->globalsTable == NULL) {
        e->globalsTable = t;
        e->globalsKeyVersion = t->keyVersion;
    } else if (e->globalsTable != t) {
        return NULL;
    }
    if (out != NULL) *out = slot->value;
    return slot;
}

/* Emitted before EVERY access, not hoisted: hoisting is sound only given a control-flow claim (no
 * call-out between a guard and a later access on a back edge) -- exactly the kind of reasoning this file has been bitten by before. Costs four instructions on a predictable branch. */
void emitGlobalsGuard(Emit *e) {
    uint32_t at = e->globalsKeyVersion;
    emitConst64(e, JIT_SCRATCH_D,
                (int64_t)(uintptr_t)&e->globalsTable->keyVersion);
    emit(e, jaiA64LdrW(JIT_SCRATCH_C, JIT_SCRATCH_D, 0));
    if (at <= 0xfffu) {
        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_C, at));
    } else {
        emitConst64(e, JIT_SCRATCH_B, (int64_t)at);
        emit(e, jaiA64SubsX(31, JIT_SCRATCH_C, JIT_SCRATCH_B));
    }
    branchOnDeopt(e, JAI_A64_NE);
}

/* JAITHON_JIT_STATIC_FIELD=0 turns the SLOT_CLASS arm of OP_GET_FIELD off, so
 * the same binary can be A/B'd around it without a rebuild -- same idiom as
 * jaiListUnboxOn's JAITHON_LIST_UNBOX (object_collection.c) and
 * jitPicEnabled's JAITHON_JIT_PIC. Read once: OP_GET_FIELD is hot enough that
 * an uncached getenv on every static access would be its own cost. */
bool jitStaticFieldEnabled(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("JAITHON_JIT_STATIC_FIELD");
        on = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return on != 0;
}

/* `klass->statics` is a JaiTable exactly like a module's globals table (same
 * struct, same keyVersion), so a static field is read the same way a module
 * global is read BY ADDRESS above: bake the JaiEntry*, not the value. A
 * static is reassignable at runtime -- jaiSetProperty's IS_CLASS arm (vm.c)
 * takes any value with no isLet check, and the checker's own
 * _check_field_assign skips its immutability error whenever `static_access`
 * is true, so `let` buys no promise here that the interpreter or the checker
 * actually keeps. Nothing below may bake the VALUE, only its address, behind
 * the same two guards a module global stands on. One table per body, same
 * plan as globalsTable -- see that field's comment on the struct. */
JaiEntry *staticFieldSlot(Emit *e, ObjClass *klass, ObjString *name) {
    JaiTable *t = &klass->statics;
    JaiEntry *slot = jaiTableFindEntryInterned(t, name);
    if (slot == NULL) return NULL;
    if (e->staticsTable == NULL) {
        e->staticsTable = t;
        e->staticsKeyVersion = t->keyVersion;
    } else if (e->staticsTable != t) {
        return NULL;
    }
    return slot;
}

/* A member of an imported module, resolved to the entry's address. The VALUE
 * is never baked, only the address, behind the same keyVersion guard a global
 * stands on -- a module global is assignable, so the tag is re-checked at every
 * read and a rebind deoptimises. */
JaiEntry *moduleMemberSlot(Emit *e, ObjModule *m, ObjString *name) {
    JaiTable *t = &m->globals;
    JaiEntry *slot = jaiTableFindEntryInterned(t, name);
    if (slot == NULL) return NULL;
    if (e->modTable == NULL) {
        e->modTable = t;
        e->modKeyVersion = t->keyVersion;
    } else if (e->modTable != t) {
        return NULL;
    }
    return slot;
}

void emitModuleGuard(Emit *e) {
    uint32_t at = e->modKeyVersion;
    emitConst64(e, JIT_SCRATCH_D,
                (int64_t)(uintptr_t)&e->modTable->keyVersion);
    emit(e, jaiA64LdrW(JIT_SCRATCH_C, JIT_SCRATCH_D, 0));
    if (at <= 0xfffu) {
        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_C, at));
    } else {
        emitConst64(e, JIT_SCRATCH_B, (int64_t)at);
        emit(e, jaiA64SubsX(31, JIT_SCRATCH_C, JIT_SCRATCH_B));
    }
    branchOnDeopt(e, JAI_A64_NE);
}

/* Off leaves an observed list return as SLOT_OBJ, so the promotion can be
 * measured apart from the probe that precedes it. */
static bool retListKindOn(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("JAITHON_JIT_RET_LIST");
        on = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return on != 0;
}

/* Off refuses a predicted-list receiver again, so the probe and the plumbing
 * that feeds it can be measured apart. */
bool listProbeOn(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("JAITHON_JIT_LIST_PROBE");
        on = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return on != 0;
}

/* Off drops the callee's observed object type again, so the difference is
 * measurable in one binary. */
bool retObjTypeOn(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("JAITHON_JIT_RET_OBJTYPE");
        on = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return on != 0;
}

/* Off puts the refusal back, so a body reading `math.PI` can be measured both
 * ways in one binary. */
bool moduleFieldOn(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("JAITHON_JIT_MODULE_FIELD");
        on = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return on != 0;
}

/* emitGlobalsGuard's counterpart for e->staticsTable: not hoisted, for the
 * same reason. */
void emitStaticsGuard(Emit *e) {
    uint32_t at = e->staticsKeyVersion;
    emitConst64(e, JIT_SCRATCH_D,
                (int64_t)(uintptr_t)&e->staticsTable->keyVersion);
    emit(e, jaiA64LdrW(JIT_SCRATCH_C, JIT_SCRATCH_D, 0));
    if (at <= 0xfffu) {
        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_C, at));
    } else {
        emitConst64(e, JIT_SCRATCH_B, (int64_t)at);
        emit(e, jaiA64SubsX(31, JIT_SCRATCH_C, JIT_SCRATCH_B));
    }
    branchOnDeopt(e, JAI_A64_NE);
}

/* `m->version` retires every cache that resolved a NAME to a heap object or presence (interpreter's
 * global inline cache, OP_FORMAT's builtin-str check, this tier's baked classes/closures/natives) -- all of which required the bound value to BE a heap object. A store replacing a non-object with a non-object changes none of them and may skip the bump; getting this wrong made `total = total + 1` at module scope retire every compiled function in the module. */
void emitVersionBump(Emit *e, ObjModule *m) {
    emitConst64(e, JIT_SCRATCH_A, (int64_t)(uintptr_t)&m->version);
    emit(e, jaiA64LdrW(JIT_SCRATCH_B, JIT_SCRATCH_A, 0));
    emit(e, jaiA64AddXImm(JIT_SCRATCH_B, JIT_SCRATCH_B, 1));
    emit(e, jaiA64StrW(JIT_SCRATCH_B, JIT_SCRATCH_A, 0));
}

/* Predicted stack kind + guard tag for a recorded OP_INVOKE result; false when the tier has no use
 * for the byte. SLOT_INST is deliberately excluded: it carries a class shape this byte can't encode, and admitting it would silently lose the shape every field offset was resolved against. Class/closure/native excluded too -- they have register-free stack kinds of their own. */
/* What an InlineCache::resultKind byte says, for JAI_JIT_WHY. A site the tier
 * refuses for want of a result kind is refused for one of three quite different
 * reasons, and the fix differs for each. */
const char *jaiFeedbackName(uint8_t fb) {
    if (fb == JAI_FB_NONE) return "never observed";
    if (fb == JAI_FB_MIXED) return "mixed";
    if (fb >= JAI_FB_OBJ && fb < JAI_FB_OBJ + (unsigned)OBJ_TYPE_COUNT) {
        return jaiObjTypeName((ObjType)(fb - JAI_FB_OBJ));
    }
    switch ((ValueType)(fb - 1u)) {
    case VAL_NULL:  return "null";
    case VAL_BOOL:  return "bool";
    case VAL_INT:   return "int";
    case VAL_FLOAT: return "float";
    default: break;
    }
    return "something the tier does not name";
}

/* `objType` reports the ObjType the feedback named, as ObjType + 1, or 0 when
 * the result is not an object. See Emit::stackObjType for what it is for. */
bool feedbackSlotKind(uint8_t fb, SlotKind *k, unsigned *tag,
                             uint8_t *objType) {
    *objType = 0;
    switch (fb) {
    case 1u + VAL_INT:   *k = SLOT_INT;   *tag = VAL_INT;   return true;
    case 1u + VAL_FLOAT: *k = SLOT_FLOAT; *tag = VAL_FLOAT; return true;
    case 1u + VAL_BOOL:  *k = SLOT_BOOL;  *tag = VAL_BOOL;  return true;
    default: break;
    }
    if (fb < JAI_FB_OBJ || fb >= JAI_FB_OBJ + (unsigned)OBJ_TYPE_COUNT) {
        return false;
    }
    switch ((ObjType)(fb - JAI_FB_OBJ)) {
    case OBJ_INSTANCE: case OBJ_CLASS: case OBJ_TRAIT:
    case OBJ_CLOSURE:  case OBJ_FUNCTION: case OBJ_NATIVE:
    case OBJ_BOUND:    case OBJ_ENUM: case OBJ_ENUM_CTOR:
        return false;
    default: break;
    }
    /* Anything else on the heap can be loaded, passed and stored and nothing
     * else, which is exactly SLOT_OBJ. The tag guard below is the whole of
     * what makes that sound: whatever object comes back, it is an object. */
    *k = SLOT_OBJ; *tag = VAL_OBJ;
    *objType = (uint8_t)(fb - JAI_FB_OBJ + 1u);
    return true;
}

/* The store half of `list.push` and of a comprehension's append: they differ
 * only in where the list sits on the stack and in what is left behind, so the
 * bounds check, the grow fixup and the two stores live here. Returns false
 * with `whyNot` set when the value's kind has no tag to store.
 *
 * Appending is a bounds check and two stores -- a descriptor+native round trip
 * costs far more than the work itself (list_ops spent all its time on the
 * call). A full list goes out to the `grow` stubs' realloc helper and comes
 * straight back; see there for why this used to be a deopt and what it cost. */
bool emitListStore(Emit *e, SlotKind vk, unsigned rList, unsigned rVal,
                          int slot) {
    unsigned vtag = vk == SLOT_INT   ? VAL_INT
                  : vk == SLOT_FLOAT ? VAL_FLOAT
                  : vk == SLOT_BOOL  ? VAL_BOOL
                  : (vk == SLOT_INST || vk == SLOT_LIST ||
                     vk == SLOT_OBJ)  ? VAL_OBJ
                                      : 0xffffffffu;
    if (vtag == 0xffffffffu) {
        e->whyNot = "pushing a kind the tier cannot store";
        return false;
    }

    /* jitListGrow only reserves, and jaiListReserve is width-aware, so the
     * growth half of this needs nothing; it is the store below that has to
     * know how wide an element is. */
    /* jitListGrow only reserves, and jaiListReserve is width-aware, so the
     * growth half of this needs nothing; it is the store below that has to
     * know how wide an element is. */
    ListAccess pAcc = listAccessFor(e, rList, slot, vk, JIT_SCRATCH_A);
    if (!pAcc.dynamic && pAcc.stg != LIST_STORE_BOXED &&
        vk != listStgKind(pAcc.stg)) {
        return subWhy(e, "pushing kind %d onto storage %u", (int)vk, pAcc.stg);
    }

    emit(e, jaiA64LdrW(JIT_SCRATCH_A, rList,
                       (unsigned)offsetof(ObjList, count)));
    emit(e, jaiA64LdrW(JIT_SCRATCH_B, rList,
                       (unsigned)offsetof(ObjList, capacity)));
    emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
    if (e->growCount >= JIT_MAX_GROW) {
        e->whyNot = "more list pushes than the tier tracks";
        return false;
    }
    /* jitListGrow can raise, and the stub routes that to the exception exit
     * (see emitGrowStubs). */
    if (!raiseExitAllowed(e, "a list growth inside a try")) return false;

    noteScratchClobber(e);
    unsigned gi = e->growCount++;
    e->grow[gi].listReg  = rList;
    e->grow[gi].valReg   = rVal;
    e->grow[gi].tag      = vtag;
    e->grow[gi].countReg = JIT_SCRATCH_A;
    e->grow[gi].stub     = -1;
    if (e->fixupCount >= JIT_MAX_FIXUPS) { e->failed = true; return false; }
    e->fixups[e->fixupCount].instIndex    = (int)e->count;
    e->fixups[e->fixupCount].targetOffset = FIXUP_GROW - gi;
    e->fixups[e->fixupCount].conditional  = true;
    e->fixups[e->fixupCount].depth        = -1;
    e->fixupCount++;
    emit(e, jaiA64BCond(JAI_A64_GE, 0));
    e->grow[gi].returnTo = (int)e->count;

    emit(e, jaiA64LdrX(JIT_SCRATCH_C, rList,
                       (unsigned)offsetof(ObjList, items)));
    /* JIT_SCRATCH_C is the items pointer and JIT_SCRATCH_A the index; every
     * arm below starts from those two, so the test costs a load, a compare and
     * two branches and touches nothing else. */
    int pSkip = listDispatchBegin(e, &pAcc, rList, JIT_SCRATCH_D);
    emitListElemStore(e, pAcc.stg, vtag, rVal);
    if (pSkip >= 0) {
        int pJoin = listDispatchElse(e, pSkip);
        emitListElemStore(e, pAcc.alt, vtag, rVal);
        listDispatchEnd(e, pJoin);
    }
    emit(e, jaiA64AddXImm(JIT_SCRATCH_A, JIT_SCRATCH_A, 1));
    emit(e, jaiA64StrW(JIT_SCRATCH_A, rList,
                       (unsigned)offsetof(ObjList, count)));
    emit(e, jaiA64LdrW(JIT_SCRATCH_A, rList,
                       (unsigned)offsetof(ObjList, version)));
    emit(e, jaiA64AddXImm(JIT_SCRATCH_A, JIT_SCRATCH_A, 1));
    emit(e, jaiA64StrW(JIT_SCRATCH_A, rList,
                       (unsigned)offsetof(ObjList, version)));
    e->wroteHeap = true;
    return true;
}

/* What an OP_INVOKE site has been observed to return, merged over every way its
 * inline cache holds.
 *
 * For a site whose receiver class the model cannot pin there is no callee to
 * ask -- but the interpreter watched the same site run, and eight
 * implementations of one trait method that all return an int agree on that
 * much. Ways that recorded nothing are skipped rather than merged: NONE
 * against a real kind is MIXED, which would throw away the evidence the other
 * ways did gather.
 *
 * A PREDICTION. The tag guard the caller emits after the call is the whole of
 * what makes it sound; a way that later returns something else deoptimises. */
bool siteInvokeResultKind(const Chunk *chunk, uint16_t cacheIdx,
                                 SlotKind *k, unsigned *tag) {
    if (chunk->caches == NULL || (int)cacheIdx >= chunk->cacheCount) {
        return false;
    }
    const InlineCache *ic = &chunk->caches[cacheIdx];
    uint8_t merged = JAI_FB_NONE;
    for (int w = 0; w < ic->count && w < JAI_IC_WAYS; w++) {
        if (ic->resultKind[w] == JAI_FB_NONE) continue;
        merged = jaiFeedbackMerge(merged, ic->resultKind[w]);
    }
    if (merged == JAI_FB_NONE || merged == JAI_FB_MIXED) return false;
    uint8_t objType;
    return feedbackSlotKind(merged, k, tag, &objType);
}

/* The same test feedbackSlotKind applies to a call's result, as a predicate on a Value: is this a heap
 * object with no stack kind of its own, so SLOT_OBJ's "read, pass, store, root and nothing else" describes it exactly? Excludes the kinds a later arm resolves to a register-free entry (class/closure/native/...), which SLOT_OBJ would silently outrank. */
bool rawObjValue(Value v) {
    if (!IS_OBJ(v) || AS_OBJ(v) == NULL) return false;
    switch (OBJ_TYPE(v)) {
    case OBJ_INSTANCE: case OBJ_CLASS: case OBJ_TRAIT:
    case OBJ_CLOSURE:  case OBJ_FUNCTION: case OBJ_NATIVE:
    case OBJ_BOUND:    case OBJ_ENUM: case OBJ_ENUM_CTOR:
        return false;
    default: break;
    }
    return true;
}

/* Predicted stack kind for a callee that has not compiled, from what it has been observed to return
 * (ObjFunction::obsReturnKind). Same contract as feedbackSlotKind's: a prediction the caller must guard.
 * SLOT_INST is admissible here where it is not there, because a per-callee record can carry the class
 * shape a one-byte-per-way cache cannot -- and the shape is guarded after the call like the tag. */
/* `objType` may be NULL. It is ObjType + 1 when the record says the callee
 * returns a particular heap object, and 0 otherwise.
 *
 * It used to be computed and dropped on the floor -- feedbackSlotKind filled a
 * local that nothing read. So a callee OBSERVED to return a list handed its
 * caller SLOT_OBJ and NO type, and `w.len()` on the result then refused with
 * "an object with no sample and no known type" even though the record said
 * plainly that it was a list. That refusal was 18.4% of the interpreted work
 * on the self-hosted compiler, the largest by a factor of seven. */
/* Same bargain the enum-`match` arms struck: believing the band lets a walk
 * past a refusal it used to stop at, and what it reaches there can decline the
 * WHOLE body where a prefix used to compile. Measured: 339 bodies compiled
 * before, 321 after -- eighteen lost. The retry in compileFunc puts them back.
 * See gMatchUsed. */
bool gNullableFbUsed;
bool gNoNullableFb;

bool observedReturnKind(const ObjFunction *cfn, SlotKind *k,
                               uint32_t *shape, uint8_t *objType) {
    uint8_t fb = cfn->obsReturnKind;
    *shape = 0;
    if (objType != NULL) *objType = 0;
    if (fb == JAI_FB_NULL) { *k = SLOT_NULL; return true; }
    /* "null and one object type", the band jaiFeedbackMerge records instead of
     * collapsing to MIXED. An instance becomes the kind the tier has spoken all
     * along; anything else it can hold at all becomes the weaker
     * SLOT_MAYBE_OBJ. Both take their tag off the payload, so neither needs a
     * representation this tier does not already emit. */
    if (jaiFeedbackIsNullable(fb) && !gNoNullableFb && jitNullableFbOn()) {
        gNullableFbUsed = true;
        ObjType ot = (ObjType)(fb - JAI_FB_NULLABLE);
        if (ot == OBJ_INSTANCE) {
            if (cfn->obsReturnShape == 0) return false;
            *k = SLOT_MAYBE_INST;
            *shape = cfn->obsReturnShape;
            return true;
        }
        /* Deliberately nothing for the other object types. SLOT_MAYBE_OBJ is
         * legal as a RETURN kind but not on an operand stack, so handing one
         * back here does not compile a caller -- it only moves the refusal
         * from "observed returns disagree" to "a method returning
         * object-or-null", 97 of them on lib/std. Widening the callers to
         * accept it means auditing every site that reads a stack entry's
         * kind, which is its own change. */
        return false;
    }
    if (fb == JAI_FB_OBJ + (unsigned)OBJ_INSTANCE) {
        if (cfn->obsReturnShape == 0) return false;
        *k = SLOT_INST;
        *shape = cfn->obsReturnShape;
        return true;
    }
    unsigned tag;
    uint8_t seenType = 0;
    if (!feedbackSlotKind(fb, k, &tag, &seenType)) return false;
    if (objType != NULL) *objType = seenType;
    /* A list earns the stronger kind here, where a per-way cache's byte cannot:
     * this is the per-callee record, the same reason SLOT_INST is admissible
     * above. It costs nothing to guard -- emitCallOutResult already emits the
     * OBJ_LIST check for SLOT_LIST -- and it is the difference between `w[2]`
     * compiling and refusing, since a subscript wants a list and SLOT_OBJ is
     * only "some object".
     *
     * `_fuse_at` is `let w = window_of(...)` then `w.len()` and `w[1]`,
     * `w[2]`, `w[3]`. Answering the method lookup alone left every subscript
     * still refusing; this is the other half of that pair. */
    if (retListKindOn() && *k == SLOT_OBJ &&
        seenType == (uint8_t)(OBJ_LIST + 1)) {
        *k = SLOT_LIST;
    }
    return true;
}

bool globalKind(Value v, SlotKind *k, uint32_t *shape, ObjClass **kls) {
    *shape = 0; *kls = NULL;
    if (IS_INT(v))      { *k = SLOT_INT;   return true; }
    if (IS_FLOAT(v))    { *k = SLOT_FLOAT; return true; }
    if (IS_BOOL(v))     { *k = SLOT_BOOL;  return true; }
    if (IS_LIST(v))     { *k = SLOT_LIST;  return true; }
    if (IS_INSTANCE(v)) {
        ObjInstance *inst = AS_INSTANCE(v);
        if (inst->klass == NULL) return false;
        *k = SLOT_INST; *kls = inst->klass; *shape = inst->klass->shapeId;
        return true;
    }
    /* Anything else on the heap (dict, string, closure) can be loaded, passed and stored, nothing else.
 * A class, function or native never reach here: OP_GET_GLOBAL resolves those to their own register-free stack kinds first. */
    if (IS_OBJ(v) && AS_OBJ(v) != NULL && !IS_CLASS(v) && !IS_CLOSURE(v) &&
        !IS_NATIVE(v)) {
        *k = SLOT_OBJ; return true;
    }
    return false;
}

/* A scalar field's DECLARED kind (FieldInfo::typeId, OP_FIELD_DEF's bits 4-7,
 * spec Sec3.7), for a receiver with a pinned class but no sample Value to read
 * `inst->fields[slot]` off -- OP_GET_FIELD's twin of OP_ELEM_KIND's own use of
 * the same bits (see that case) rather than a sampled container.
 *
 * Not a new promise: OP_SET_FIELD's guard (jaiKindAccepts, vm.c) already
 * refuses any store that disagrees with this field's declared kind, so a
 * caller here is only reading a fact the runtime enforces on every write, and
 * the tag is checked again at the load below regardless -- a stale or wrong
 * record still deopts rather than answers.
 *
 * INT/FLOAT/BOOL only. FIELD_KIND_LIST names the box but not the element, so
 * admitting it here would still leave a consumer that iterates the field with
 * nothing to look at (the "iterating a list with nothing to look at" refusal,
 * unresolved either way); FIELD_KIND_INSTANCE names no specific class to guard
 * against; FIELD_KIND_ANY and FIELD_KIND_STR/DICT promise nothing scalar. All
 * four are left to the sampled path, unchanged. */
bool declaredScalarFieldKind(uint32_t typeId, SlotKind *k, unsigned *tag) {
    switch (typeId) {
    case FIELD_KIND_INT:   *k = SLOT_INT;   *tag = VAL_INT;   return true;
    case FIELD_KIND_FLOAT: *k = SLOT_FLOAT; *tag = VAL_FLOAT; return true;
    case FIELD_KIND_BOOL:  *k = SLOT_BOOL;  *tag = VAL_BOOL;  return true;
    /* A declared `list[T]` field. VAL_OBJ is every heap object, so the caller
     * must ALSO prove OBJ_LIST before anything reads ObjList's header off it --
     * the same hazard that segfaulted the VM through the dict-index arm. It is
     * separated from the three scalars above because it is the only kind here
     * that needs a second guard.
     *
     * Worth predicting because a list field is where a chain bottoms out:
     * `code.data[i]` is a field read the tier could not classify, and behind
     * that one refusal sat eleven functions and 14.27% of one file's
     * interpreted work. */
    /* Predicts that the field IS a list, and cannot predict what is IN it: the
     * index arm below wants an element exemplar and a prediction has no value
     * to take one from. So `d.items` compiles and `d.items[i]` still declines,
     * which is why clearing this link alone did not free `_fuse_at`. */
    case FIELD_KIND_LIST:  *k = SLOT_LIST;  *tag = VAL_OBJ;   return true;
    /* A declared `str`. Same two-guard shape as the list above, and worth its
     * own row because a string entry carrying a SAMPLE unlocks the arms below
     * it -- `.len()`, `==` on interned pointers, the ordering leaf call -- all
     * of which ask `stringOperand`, which asks the sample and not the kind. */
    case FIELD_KIND_STR:   *k = SLOT_OBJ;   *tag = VAL_OBJ;   return true;
    /* A user type name. VAL_OBJ is exactly what it promises and exactly what
     * the tag guard proves, so no second guard is wanted -- unlike the list and
     * str rows above, this one does not claim a particular ObjType.
     *
     * A field holding NULL deopts here, which is the honest cost: a field
     * declared `Foo` is null only before its init assigns it, and a field never
     * assigned at all would deopt on every read rather than answer wrongly. */
    case FIELD_KIND_DECLARED: *k = SLOT_OBJ; *tag = VAL_OBJ;   return true;
    default: return false;
    }
}

bool emitGetGlobal(Emit *e, ObjFunction *fn, ObjClosure *closure,
                   const uint8_t *code, int *offp, int stop) {
    int off = *offp;
    do {
        /* Two resolution paths with different obligations. BY VALUE (globalIsSelf/globalClass/globalFunction/
         * globalNative): resolved and baked at compile time, nothing rechecked at run time, so ObjModule::version must retire the whole form whenever such a binding could change (jaiModuleSet's jaiValueIsInertGlobal decides that) -- teaching any of the four a new value kind, or constant-folding a module int here, means updating jaiValueIsInertGlobal too. BY ADDRESS (the value case below): bakes the JaiEntry*, not the value, and re-loads it behind a tag guard (+ Obj.type/class-shape guards where needed) on EVERY access -- depends on JaiTable::keyVersion, not on ObjModule::version. */
        uint32_t nameIdx = jaiReadU24(code + off + 1);
        if (globalIsSelf(closure, nameIdx)) {
            if (!pushSelf(e)) return false;
            off += 6;
            break;
        }
        /* A class, resolved now and pinned by the module version check at
         * entry: rebinding the name retires the compiled form. It occupies
         * no register -- it is baked into the call sequence. */
        ObjClass *cls = globalClass(closure, nameIdx);
        if (cls != NULL) {
            if (e->depth >= JIT_MAX_STACK) return false;
            e->stackShape[e->depth] = cls->shapeId;
            e->stackClass[e->depth] = cls;
            e->stackSeen[e->depth]  = NULL_VAL;
            e->stackLocal[e->depth] = -1;
            e->stackAscii[e->depth] = false;
            e->stackNullLit[e->depth] = false;
            e->stackUnit[e->depth]  = false;
            e->stack[e->depth++]    = SLOT_CLASS;
            off += 6;
            break;
        }
        /* A plain function, resolved now and pinned by the same
         * module-version check. Only one that has itself compiled -- and
         * the reason is NOT soundness, which is what this comment used to
         * say and what two separate investigations went looking for.
         *
         * The miscompile it used to warn about is real and is FIXED. It was
         * never a property of the callee: a descriptor `result` holding a
         * bool was loaded eight bytes wide where BOOL_VAL writes one, so
         * `false` came back with dirty high bytes and read as true. That is
         * why the corruption was one-directional and looked deterministic
         * in a lexer -- delta-debugging 63 admitted callees reduced it to
         * `_is_alpha` alone. Both sites are closed: the ordinary return in
         * 5036099, the verdict-4 slow stub in emitSelfSlowStubs above.
         * docs/agents/uncompiled-callee.md has the reduction.
         *
         * With both fixed, dropping the check is SOUND -- 1274 green plain,
         * under JAITHON_NO_JIT, deopt stress and split stress -- and it
         * does let the mutually recursive groups in: json_parse's `value`,
         * `object`, `array` and `integer` all reach the tier.
         *
         * It is kept because removing it does not pay, which nobody had
         * measured. Net on `check --no-cache parser.jai`, stable across
         * three runs each: 284 function-tier bodies with the check, 275
         * without -- five gained, fourteen lost. An independent measurement
         * at a different commit found the same direction (265 to 258, seven
         * gained and thirteen lost). json_parse's own wall clock does not
         * move (0.05s either way) and neither does the compiler's.
         *
         * TWO things make it come out that way, and the second is the one
         * that is easy to miss. Declining here falls into emitUnarmedDeopt
         * just below, which interprets from this instruction ONWARD rather
         * than giving the whole body up; admitting the callee skips that
         * escape hatch and the body walks on to a refusal that costs all of
         * it -- `value` itself then stops at "callee's return kind not
         * usable". AND: admitting callees earlier changes how much
         * interpreted work happens before OTHER, unrelated functions cross
         * their own call-count hotness threshold in a fixed workload, so
         * some fall short of a threshold they used to clear and others
         * clear one they used to miss. The name sets differ in both
         * directions for that reason, and neither count is a general
         * verdict about the tier.
         *
         * So: removable, and not worth removing. If the unarmed-deopt path
         * ever stops being the better half of that trade, this is one line.
         */
        Value gv;
        ObjFunction *gfn = globalFunction(closure, nameIdx, &gv);
        if (gfn == NULL || gfn->jitFunc == NULL) {
            Value nv;
            ObjNative *nat = globalNative(closure, nameIdx, &nv);
            if (nat == NULL) {
                /* `__prim__` and any other native namespace: not a native
                 * itself (globalNative already said so, correctly -- it
                 * is an ObjModule), and not found by globalSlot below
                 * either, because it lives in vm.builtins, not in this
                 * function's own module -- that lookup would always come
                 * back NULL for it, which is the whole reason every
                 * `__prim__.f64_*` body used to stop here. Pushed as
                 * SLOT_OBJ, register-resident, exactly the shape
                 * `math.sqrt`'s own receiver (an IMPORTED module) already
                 * reaches OP_INVOKE in -- except loaded as a bare
                 * constant rather than through a JaiEntry, since nothing
                 * re-binds the name `__prim__` itself the way an import
                 * can be reassigned. See emitModuleNativeCall for the
                 * guard that matters here: not this identity, but whether
                 * the MEMBER OP_INVOKE goes on to read has been rebound
                 * since compile. */
                ObjModule *ns =
                    (jitModuleNativeCalls() &&
                     primInvokePairFits(e, &fn->chunk, (uint32_t)off))
                        ? globalNamespace(closure, nameIdx)
                        : NULL;
                if (ns != NULL) {
                    if (e->depth >= JIT_MAX_STACK) return false;
                    unsigned dst = valueXReg(e, e->valueDepth);
                    if (!pushValue3(e, SLOT_OBJ, 0, NULL,
                                    OBJ_VAL((Obj *)ns), -1)) {
                        return false;
                    }
                    emitConst64(e, dst, (int64_t)(uintptr_t)ns);
                    off += 6;
                    break;
                }
                /* A global holding a plain value: storage is a JaiEntry whose address is fixed once it exists, so the
                 * load is one `ldr` behind two guards (table hasn't moved, value still has the compiled-for kind). Refusing this declined every loop reading a module-level variable -- most benchmarks, once they moved to module scope. */
                Value gvv = NULL_VAL;
                JaiEntry *gslot = globalSlot(e, closure, nameIdx, &gvv);
                SlotKind gk = SLOT_OPAQUE;
                uint32_t gshape = 0;
                ObjClass *gcls = NULL;
                if (gslot != NULL && globalKind(gvv, &gk, &gshape, &gcls)) {
                    /* The register the push BELOW will land in, named
                     * ahead of it because the guards have to run against
                     * the model as it is now. `pushReg` is one past the
                     * CURRENT top, which is the same register only while
                     * the bank is one contiguous run -- at a split
                     * boundary it is the last callee-saved one and the
                     * push goes to x0. That mismatch loaded a global into
                     * a register nothing then read, and bitops printed
                     * 68720029766 for 999625. */
                    unsigned dst = valueXReg(e, e->valueDepth);
                    unsigned tag = gk == SLOT_INT   ? VAL_INT
                                 : gk == SLOT_FLOAT ? VAL_FLOAT
                                 : gk == SLOT_BOOL  ? VAL_BOOL
                                                    : VAL_OBJ;
                    /* Every guard runs against the model as it is BEFORE
                     * the value is pushed, so a deopt here resumes at this
                     * instruction with the operand stack the interpreter
                     * expects. */
                    emitGlobalsGuard(e);
                    emitConst64(e, JIT_SCRATCH_D,
                                (int64_t)(uintptr_t)gslot);
                    emit(e, jaiA64LdrW(JIT_SCRATCH_C, JIT_SCRATCH_D,
                                       (unsigned)offsetof(JaiEntry, value)));
                    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_C, tag));
                    branchOnDeopt(e, JAI_A64_NE);
                    if (gk == SLOT_BOOL) {
                        /* A byte, not a word. BOOL_VAL writes the union's
                         * one-byte `bool` member and leaves the other seven
                         * bytes of the payload indeterminate, so an 8-byte
                         * load brings back whatever the slot held before.
                         * Every consumer tests the whole register against
                         * zero, so those bytes decide the answer.
                         *
                         * This read a module-level `var` as FALSE once the
                         * body compiled: `std.fmt`'s `_colors_enabled` is
                         * true, and `green()` returned uncoloured text from
                         * the 65th call onward -- exactly
                         * JAI_JIT_THRESHOLD. A silent wrong answer, not a
                         * crash, and it had been in the tree long enough
                         * that a parallel test run turned it up by
                         * accident. emitCallOutResult's SLOT_BOOL arm has
                         * always split the two; this site never did. */
                        emit(e, jaiA64LdrByte(JIT_SCRATCH_C, JIT_SCRATCH_D,
                                              (unsigned)offsetof(JaiEntry, value) + 8u));
                    } else {
                        emit(e, jaiA64LdrX(JIT_SCRATCH_C, JIT_SCRATCH_D,
                                           (unsigned)offsetof(JaiEntry, value) + 8u));
                    }
                    if (gk == SLOT_INST || gk == SLOT_LIST) {
                        /* VAL_OBJ is every heap object; the tag alone doesn't confirm the specific type, so it's checked before the class pointer is read. */
                        emit(e, jaiA64LdrByte(JIT_SCRATCH_B, JIT_SCRATCH_C,
                                              (unsigned)offsetof(Obj, type)));
                        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_B,
                                               gk == SLOT_INST ? OBJ_INSTANCE
                                                               : OBJ_LIST));
                        branchOnDeopt(e, JAI_A64_NE);
                    }
                    if (gk == SLOT_INST) {
                        emit(e, jaiA64LdrX(JIT_SCRATCH_B, JIT_SCRATCH_C,
                                           (unsigned)offsetof(ObjInstance, klass)));
                        emit(e, jaiA64LdrW(JIT_SCRATCH_B, JIT_SCRATCH_B,
                                           (unsigned)offsetof(ObjClass, shapeId)));
                        emitConst64(e, JIT_SCRATCH_A, (int64_t)gshape);
                        emit(e, jaiA64SubsX(31, JIT_SCRATCH_B, JIT_SCRATCH_A));
                        branchOnDeopt(e, JAI_A64_NE);
                    }
                    if (!pushValue3(e, gk, gshape, gcls, gvv, -1)) return false;
                    emit(e, jaiA64MovX(dst, JIT_SCRATCH_C));
                    off += 6;
                    break;
                }
                /* Distinguish the two, because the census reads these and
                 * "a global of a kind the tier has no slot for" is a very
                 * different backlog item from an uncompiled callee. */
                /* Interpreted from here rather than the whole function
                 * declined. The case that matters is a global read on a
                 * branch that never runs: `throw ValueError(...)` in an
                 * argument check is the shape, there are hundreds of them
                 * in lib/std alone, and declining for one cost 110 ms
                 * against 14 ms for the same function guarded by a plain
                 * `return`. The unarmed path already exists for exactly
                 * this -- see emitUnarmedDeopt. */
                if (!e->osr && emitUnarmedDeopt(e, &fn->chunk, &off, stop)) {
                    break;
                }
                if (e->failed) return false;
                /* Named, because this is the top state-level refusal by
                 * attempt count -- 406 in one file, 82 of them retries of
                 * a single loop -- and "callee is not a compiled global
                 * function" gave the reader nothing to act on. Which
                 * callee it is decides whether this is a tier-ordering
                 * problem that resolves itself or a function that never
                 * compiles at all. */
                Value gNameVal = nameIdx < (uint32_t)fn->chunk.constants.count
                                     ? fn->chunk.constants.data[nameIdx]
                                     : NULL_VAL;
                const char *gName = IS_STRING(gNameVal)
                                        ? AS_STRING(gNameVal)->chars
                                        : "?";
                if (gslot != NULL && !IS_CLOSURE(gvv) && !IS_CLASS(gvv) &&
                    !IS_NATIVE(gvv)) {
                    return subWhy(e, "`%s` is a global of a kind the tier "
                                  "has no slot for", gName);
                }
                return subWhy(e, "`%s` is not a compiled global function",
                              gName);
            }
            if (e->depth >= JIT_MAX_STACK) return false;
            e->stackShape[e->depth] = 0;
            e->stackClass[e->depth] = (ObjClass *)(void *)AS_OBJ(nv);
            e->stackSeen[e->depth]  = nv;
            e->stackLocal[e->depth] = -1;
            e->stackAscii[e->depth] = false;
            e->stackNullLit[e->depth] = false;
            e->stackUnit[e->depth]  = false;
            e->stack[e->depth++]    = SLOT_NATIVE;
            off += 6;
            break;
        }
        if (e->depth >= JIT_MAX_STACK) return false;
        e->stackShape[e->depth] = 0;
        /* The stub that writes a deopt record materialises a callee from
         * here, so it has to be the object and not NULL. */
        e->stackClass[e->depth] = (ObjClass *)(void *)AS_OBJ(gv);
        e->stackSeen[e->depth]  = gv;
        e->stackLocal[e->depth] = -1;
        e->stackAscii[e->depth] = false;
        e->stackNullLit[e->depth] = false;
        e->stackUnit[e->depth]  = false;
        e->stack[e->depth++]    = SLOT_FUNC;
        off += 6;
        break;
    } while (0);
    *offp = off;
    return true;
}

bool emitSetGlobal(Emit *e, ObjClosure *closure, const uint8_t *code,
                   int *offp) {
    int off = *offp;
    do {
        /* Assigns without popping, exactly as the interpreter does: the
         * value is the statement's result and an OP_POP follows. */
        uint32_t nameIdx = jaiReadU24(code + off + 1);
        Value gvv;
        JaiEntry *gslot = globalSlot(e, closure, nameIdx, &gvv);
        if (gslot == NULL) {
            /* The interpreter raises NameError for a name with no binding;
             * there is nothing to store into and nothing to compile. */
            e->whyNot = "a global with no storage to store into";
            return false;
        }
        if (e->depth == 0 || !holdsRegister(e->stack[e->depth - 1])) {
            e->whyNot = "nothing on the stack to store into a global";
            return false;
        }
        SlotKind sk = e->stack[e->depth - 1];
        if (sk != SLOT_INT && sk != SLOT_FLOAT && sk != SLOT_BOOL &&
            sk != SLOT_LIST && sk != SLOT_INST && sk != SLOT_MAYBE_INST) {
            /* SLOT_OBJ deliberately excluded: it covers a closure too, and storing a closure into a global
             * rebinds a callee some compiled form may have already baked BY VALUE -- ObjModule::version only retires that form at its next ENTRY, not mid-body, so this would be a silently wrong answer, not a decline. Every remaining kind is inert per jaiValueIsInertGlobal, which is what makes the bump rule below sound. */
            e->whyNot = "a global store of a kind that has no Value form";
            return false;
        }
        emitGlobalsGuard(e);
        {
            unsigned src = pushReg(e) - 1;
            emitConst64(e, JIT_SCRATCH_D, (int64_t)(uintptr_t)gslot);
            /* Version only needs to move when a class/closure/native leaves or arrives (jaiValueIsInertGlobal is
             * the authority). The incoming value is already provably inert (SLOT_OBJ refused above), so only what's overwritten matters -- this checks a STRICT SUBSET of the inert types (non-object, ObjInstance, ObjList); anything else conservatively bumps. Widening jaiValueIsInertGlobal needs no change here; narrowing it (removing OBJ_INSTANCE/OBJ_LIST) does. */
            unsigned skip[3];
            unsigned nskip = 0;
            emit(e, jaiA64LdrW(JIT_SCRATCH_B, JIT_SCRATCH_D,
                               (unsigned)offsetof(JaiEntry, value)));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_B, VAL_OBJ));
            skip[nskip++] = e->count;
            emit(e, jaiA64BCond(JAI_A64_NE, 0));
            emit(e, jaiA64LdrX(JIT_SCRATCH_B, JIT_SCRATCH_D,
                               (unsigned)offsetof(JaiEntry, value) + 8u));
            emit(e, jaiA64LdrByte(JIT_SCRATCH_C, JIT_SCRATCH_B,
                                  (unsigned)offsetof(Obj, type)));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_C, OBJ_INSTANCE));
            skip[nskip++] = e->count;
            emit(e, jaiA64BCond(JAI_A64_EQ, 0));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_C, OBJ_LIST));
            skip[nskip++] = e->count;
            emit(e, jaiA64BCond(JAI_A64_EQ, 0));
            emitVersionBump(e, closure->fn->module);
            for (unsigned si = 0; si < nskip; si++) {
                unsigned at = skip[si];
                if (at < JIT_MAX_INSTS && e->count > at) {
                    e->code[at] = jaiA64BCond(
                        si == 0 ? JAI_A64_NE : JAI_A64_EQ,
                        (int32_t)(e->count - at));
                }
            }
            /* No write barrier: mark-sweep traces module globals as a root every collection, so a raw store is
             * enough. Nothing between the tag and payload writes can allocate, so no collection can ever see them disagree. */
            emitTagFor(e, sk, src, JIT_SCRATCH_B, JIT_SCRATCH_A);
            emit(e, jaiA64StrW(JIT_SCRATCH_B, JIT_SCRATCH_D,
                               (unsigned)offsetof(JaiEntry, value)));
            emit(e, jaiA64StrX(src, JIT_SCRATCH_D,
                               (unsigned)offsetof(JaiEntry, value) + 8u));
        }
        /* Re-running this call interpreted would apply the store twice, so
         * from here a bail is no longer free. */
        e->wroteHeap = true;
        off += 6;
        break;
    } while (0);
    *offp = off;
    return true;
}

#else

#endif
