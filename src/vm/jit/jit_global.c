/* jit_global.c -- resolving globals, static fields and module members, the guards
 * that keep those resolutions honest, and the slot-kind inference built on them. */
#include "vm/jit/jit.h"

#include "vm/jit/jit_arm64.h"
/* For jaiJitFieldReadFor: which builtins are one load from their receiver. */
#include "vm/jit/jit_field_read.h"
#include "vm/gc.h"
/* For jaiBuiltinMethod: resolving `xs.len()` to a native needs the runtime's name table. */
#include "runtime/runtime.h"
/* For jaiOpBranchOperandAt: says which opcodes carry a branch target. */
#include "vm/bytecode/verify.h"
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
bool observedReturnKind(const ObjFunction *cfn, SlotKind *k,
                               uint32_t *shape, uint8_t *objType) {
    uint8_t fb = cfn->obsReturnKind;
    *shape = 0;
    if (objType != NULL) *objType = 0;
    if (fb == 1u + (unsigned)VAL_NULL) { *k = SLOT_NULL; return true; }
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

#else

#endif
