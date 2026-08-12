/* object_module.c — ObjModule: a loaded source file's global namespace and
 * export set, plus the version counter compiled code checks before trusting
 * a binding it baked in at JIT time.
 *
 * A one-kind file — nothing else in the object model depends on ObjModule's
 * internals, and it depends on nothing beyond JaiTable and pushObjRoot — but
 * jaiModuleSet's counter-bump logic is subtle enough (see the comment inline)
 * that it earns being read on its own rather than folded in beside a bigger,
 * busier group.
 */

#include "object.h"
#include "object_internal.h"   /* pushObjRoot */

#include "gc.h"
#include "table.h"
#include "vm.h"

/* ------------------------------------------------------------------ */
/* Modules                                                              */
/* ------------------------------------------------------------------ */

ObjModule *jaiModuleNew(ObjString *name, ObjString *path) {
    pushObjRoot(name);
    pushObjRoot(path);
    ObjModule *module = JAI_ALLOCATE_OBJ(ObjModule, OBJ_MODULE);
    jaiGCPopRoots(2);

    module->name = name;
    module->path = path;
    jaiTableInit(&module->globals);
    jaiTableInit(&module->exports);
    module->state = MOD_UNLOADED;
    module->sourceFileId = -1;
    return module;
}

bool jaiModuleGet(ObjModule *m, ObjString *name, Value *out) {
    return jaiTableGetInterned(&m->globals, name, out);
}

void jaiModuleSet(ObjModule *m, ObjString *name, Value v) {
    jaiGCPushRoot(OBJ_VAL(m));
    jaiGCPushRoot(v);
    Value prev = NULL_VAL;
    const bool added = jaiTableSetInternedPrev(&m->globals, name, v, &prev);
    jaiGCPopRoots(2);

    /* ObjModule::version retires compiled code, and compiled code resolves a
     * global by VALUE exactly four ways -- globalClass, globalFunction,
     * globalNative and globalIsSelf in jit_func.c -- each of which demands
     * IS_CLASS, IS_CLOSURE or IS_NATIVE. Everything else it resolves by
     * ADDRESS, re-loading the value behind a tag guard on every access, so an
     * update to an inert value needs no invalidation at all.
     *
     * jaiValueIsInertGlobal is the test, and it is written the safe way round:
     * it lists the types compiled code provably cannot bake and answers false
     * for everything else, so a new ObjType falls into the conservative arm.
     *
     * The key set and the table layout are NOT tracked here -- JaiTable bumps
     * keyVersion itself, so no writer can forget to. */
    if (added || ((IS_OBJ(prev) || IS_OBJ(v)) &&
                  (!jaiValueIsInertGlobal(prev) || !jaiValueIsInertGlobal(v)))) {
        m->version++;
    }
}

bool jaiModuleIsExported(ObjModule *m, ObjString *name) {
    Value ignored;
    return jaiTableGetInterned(&m->exports, name, &ignored);
}
