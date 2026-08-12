/* module_methods.c — native methods on module objects (mod.get, .has, ...).
 *
 * Independent of module.c's loading/import machinery: everything here runs
 * *after* a module already exists, dispatched by the VM's member-access path
 * for a receiver that is a module rather than an instance (jaiModuleMethod is
 * the entry point, declared in methods.h alongside the other builtin method
 * tables). None of it touches the search path, the cache, or the bootstrap
 * window's state — it only reads the ObjModule the caller handed it, so it
 * shares no private state with the rest of module loading and belongs on its
 * own.
 *
 * `mod.thing` is resolved by the VM directly, so what reaches here is the
 * reflective path: dir(), and any C caller that asks the built-in method
 * tables about a module receiver. Both need the same answer the VM would
 * give, which is why the export rule is repeated rather than approximated: a
 * module with no explicit exports exposes everything, because restricting an
 * un-annotated module would make it unusable rather than encapsulated
 * (spec §8.2).
 */

#include <stdlib.h>

#include "runtime.h"
#include "methods.h"

static bool moduleExposes(ObjModule *m, ObjString *name, Value *out) {
    /* Reflective: `name` came from the caller, and a module table is keyed by
     * pointer. */
    name = jaiStringCanonical(name);
    if (!jaiModuleGet(m, name, out)) return false;
    return m->exports.count == 0 || jaiModuleIsExported(m, name);
}

static bool requireModuleArg(Value v, const char *fnName, ObjModule **out) {
    if (!IS_MODULE(v))
        return jaiThrow(vm.cTypeError, "%s() expected a module, not %s", fnName,
                        jaiTypeNameStatic(v));
    *out = AS_MODULE(v);
    return true;
}

static bool nModuleName(int argc, Value *args, Value *out) {
    (void)argc;
    ObjModule *m;
    if (!requireModuleArg(args[0], "name", &m)) return false;
    *out = m->name != NULL ? OBJ_VAL(m->name) : OBJ_VAL(jaiStringIntern("", 0));
    return true;
}

static bool nModulePath(int argc, Value *args, Value *out) {
    (void)argc;
    ObjModule *m;
    if (!requireModuleArg(args[0], "path", &m)) return false;
    *out = m->path != NULL ? OBJ_VAL(m->path) : OBJ_VAL(jaiStringIntern("", 0));
    return true;
}

static bool nModuleHas(int argc, Value *args, Value *out) {
    (void)argc;
    ObjModule *m;
    ObjString *name;
    if (!requireModuleArg(args[0], "has", &m)) return false;
    if (!jaiArgString(args[1], 2, "has", &name)) return false;
    Value ignored;
    *out = BOOL_VAL(moduleExposes(m, name, &ignored));
    return true;
}

/* get(name, default = null): the reflective read, so a missing name is a value
 * rather than an exception — a caller who wants the raise uses `mod.name`. */
static bool nModuleGet(int argc, Value *args, Value *out) {
    ObjModule *m;
    ObjString *name;
    if (!requireModuleArg(args[0], "get", &m)) return false;
    if (!jaiArgString(args[1], 2, "get", &name)) return false;

    if (moduleExposes(m, name, out)) return true;
    *out = argc >= 3 ? args[2] : NULL_VAL;
    return true;
}

static int compareNames(const void *a, const void *b) {
    ObjString *left = *(ObjString *const *)a;
    ObjString *right = *(ObjString *const *)b;
    uint32_t shortest = left->length < right->length ? left->length : right->length;
    int order = memcmp(left->chars, right->chars, shortest);
    if (order != 0) return order;
    return left->length < right->length ? -1
         : left->length > right->length ? 1 : 0;
}

/* Sorted, because a hash table's order is an implementation detail and a
 * program that prints `members()` should not change output between runs. */
static bool nModuleMembers(int argc, Value *args, Value *out) {
    (void)argc;
    ObjModule *m;
    if (!requireModuleArg(args[0], "members", &m)) return false;

    int slot = 0;
    Value key, value;
    int found = 0;
    while (jaiTableNext(&m->globals, &slot, &key, &value)) {
        if (IS_STRING(key) && moduleExposes(m, AS_STRING(key), &value)) found++;
    }

    ObjString **names = found > 0 ? JAI_ALLOC(ObjString *, found) : NULL;
    int written = 0;
    slot = 0;
    while (jaiTableNext(&m->globals, &slot, &key, &value) && written < found) {
        if (IS_STRING(key) && moduleExposes(m, AS_STRING(key), &value))
            names[written++] = AS_STRING(key);
    }
    if (written > 1) qsort(names, (size_t)written, sizeof *names, compareNames);

    ObjList *list = jaiListNew(written);
    jaiPushRoot(OBJ_VAL(list));
    for (int i = 0; i < written; i++) jaiListPush(list, OBJ_VAL(names[i]));
    jaiPopRoot();
    if (names != NULL) JAI_FREE_ARRAY(ObjString *, names, found);

    *out = OBJ_VAL(list);
    return true;
}

bool jaiModuleMethod(Value receiver, ObjString *name, Value *out) {
    if (!IS_MODULE(receiver) || name == NULL) return false;
    ObjModule *m = AS_MODULE(receiver);

    /* A member the module exports outranks the introspection helpers: a module
     * defining `get` means its own `get` everywhere it is named. */
    if (moduleExposes(m, name, out)) return true;

    const char *text = name->chars;

/* Arities count the receiver, which the VM passes as args[0]. */
#define MODULE_METHOD(label, fn, minArity, maxArity)                           \
    if (strcmp(text, (label)) == 0) {                                          \
        *out = jaiBindNative(receiver, (label), (fn), (minArity), (maxArity),  \
                             NULL);                                            \
        return true;                                                           \
    }

    MODULE_METHOD("get",     nModuleGet,     2, 3)
    MODULE_METHOD("has",     nModuleHas,     2, 2)
    MODULE_METHOD("members", nModuleMembers, 1, 1)
    MODULE_METHOD("name",    nModuleName,    1, 1)
    MODULE_METHOD("path",    nModulePath,    1, 1)

#undef MODULE_METHOD

    /* Present but private. Saying "no such member" would send the reader
     * looking for a typo, so this is E0802's wording, the same one the VM and
     * `from ... import` raise. */
    Value hidden;
    if (jaiModuleGet(m, name, &hidden))
        return jaiThrow(vm.cImportError, "'%s' is not exported by module '%s'",
                        name->chars, m->name != NULL ? m->name->chars : "?");
    return false;
}
