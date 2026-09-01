/* Compile, eval, exec, globals, and build-id reflection primitives. */

#include "runtime/runtime.h"

#include "vm/gc.h"
#include "vm/bytecode/serialize.h"

/* The binding eval() compiles into and then removes. It is not of the form
 * `__x__`, which the parser reserves (§2.1), and it starts with an underscore,
 * which is what stops the checker warning that it is never read. */
static const char kEvalSlot[] = "__jai_eval";

static ObjModule *callerModule(void) {
    if (vm.frameCount > 0 && vm.frames[vm.frameCount - 1].module != NULL)
        return vm.frames[vm.frameCount - 1].module;
    if (vm.mainModule != NULL) return vm.mainModule;
    return vm.builtins;
}

static CodegenOptions reflectOptions(void) {
    CodegenOptions opts = jaiCodegenDefaults();
    opts.optLevel = vm.optLevel;
    opts.stripAsserts = vm.releaseMode;
    return opts;
}

static bool throwCompileError(const char *fnName) {
    char message[512];
    const JaiDiag *first = NULL;
    for (int i = 0; i < gDiags.diags.count && first == NULL; i++) {
        if (gDiags.diags.data[i].severity == JAI_SEV_ERROR)
            first = &gDiags.diags.data[i];
    }
    if (first == NULL && gDiags.diags.count > 0) first = &gDiags.diags.data[0];
    if (first != NULL) {
        snprintf(message, sizeof message, "%s(): %s: %s", fnName,
                 jaiDiagCodeString(first->code),
                 first->message != NULL ? first->message : "compilation failed");
    } else {
        snprintf(message, sizeof message, "%s(): compilation failed", fnName);
    }
    jaiDiagReset(&gDiags);
    return jaiThrow(vm.cParseError, "%s", message);
}

static ObjFunction *compileFragment(const char *source, size_t length,
                                    const char *path, ObjModule *module,
                                    const char *fnName) {
    CodegenOptions opts = reflectOptions();
    ObjFunction *fn = jaiCompileSource(source, length, path, module, &opts);
    if (fn == NULL) {
        if (!vm.hasException) (void)throwCompileError(fnName);
        return NULL;
    }
    fn->module = module;
    return fn;
}

static bool runFragment(ObjFunction *fn, Value *out) {
    Value ignored;
    if (out == NULL) out = &ignored;

    jaiGCPushRoot(OBJ_VAL(fn));
    ObjClosure *closure = jaiClosureNew(fn);
    jaiGCPopRoot();

    jaiGCPushRoot(OBJ_VAL(closure));
    bool ok = jaiCallValue(OBJ_VAL(closure), 0, NULL, out);
    jaiGCPopRoot();
    return ok;
}

static bool nReflectCompile(int argc, Value *args, Value *out) {
    ObjString *source;
    if (!jaiArgString(args[0], 1, "compile", &source)) return false;

    const char *path = "<compiled>";
    if (argc >= 2 && !IS_NULL(args[1])) {
        ObjString *name;
        if (!jaiArgString(args[1], 2, "compile", &name)) return false;
        if (name->length > 0) path = name->chars;
    }

    ObjModule *module = callerModule();
    ObjFunction *fn = compileFragment(source->chars, source->length, path,
                                      module, "compile");
    if (fn == NULL) return false;

    jaiGCPushRoot(OBJ_VAL(fn));
    ObjClosure *closure = jaiClosureNew(fn);
    jaiGCPopRoot();
    *out = OBJ_VAL(closure);
    return true;
}

static bool nReflectEval(int argc, Value *args, Value *out) {
    (void)argc;
    ObjString *source;
    if (!jaiArgString(args[0], 1, "eval", &source)) return false;

    JaiBuf text;
    jaiBufInit(&text);
    jaiBufAppendStr(&text, "let ");
    jaiBufAppendStr(&text, kEvalSlot);
    /* The expression gets a line of its own inside the parentheses so that a
     * trailing comment cannot swallow the closing one. */
    jaiBufAppendStr(&text, " = (\n");
    jaiBufAppend(&text, source->chars, source->length);
    jaiBufAppendStr(&text, "\n)\n");

    size_t length = 0;
    char *wrapped = jaiBufTakeCString(&text, &length);
    if (wrapped == NULL) return jaiThrow(vm.cRuntimeError, "out of memory in eval");

    ObjModule *module = callerModule();
    ObjFunction *fn = compileFragment(wrapped, length, "<eval>", module, "eval");
    JAI_FREE_ARRAY(char, wrapped, length + 1);
    if (fn == NULL) return false;

    if (!runFragment(fn, NULL)) return false;

    ObjString *slot = jaiStringInternC(kEvalSlot);
    Value value = NULL_VAL;
    if (module != NULL) {
        (void)jaiModuleGet(module, slot, &value);
        if (jaiTableDelete(&module->globals, OBJ_VAL(slot))) module->version++;
    }
    *out = value;
    return true;
}

static ObjModule *execNamespace(ObjString *name) {
    name = jaiStringCanonical(name);
    Value existing;
    if (jaiTableGetInterned(&vm.modules, name, &existing) && IS_MODULE(existing))
        return AS_MODULE(existing);

    ObjModule *module = jaiModuleNew(name, name);
    if (module == NULL) {
        (void)jaiThrow(vm.cRuntimeError, "out of memory in exec");
        return NULL;
    }
    module->state = MOD_LOADED;      /* nothing may try to load it from disk */
    jaiGCPushRoot(OBJ_VAL(module));
    (void)jaiTableSetInterned(&vm.modules, name, OBJ_VAL(module));
    jaiGCPopRoot();
    return module;
}

static bool nReflectExec(int argc, Value *args, Value *out) {
    ObjString *source;
    if (!jaiArgString(args[0], 1, "exec", &source)) return false;

    ObjModule *module = callerModule();
    if (argc >= 2 && !IS_NULL(args[1])) {
        ObjString *space;
        if (!jaiArgString(args[1], 2, "exec", &space)) return false;
        module = execNamespace(space);
        if (module == NULL) return false;
    }

    ObjFunction *fn = compileFragment(source->chars, source->length, "<exec>",
                                      module, "exec");
    if (fn == NULL) return false;
    if (!runFragment(fn, NULL)) return false;
    *out = NULL_VAL;
    return true;
}

static bool nReflectGlobals(int argc, Value *args, Value *out) {
    (void)argc;
    (void)args;
    ObjModule *module = callerModule();

    ObjDict *snapshot = jaiDictNew();
    jaiGCPushRoot(OBJ_VAL(snapshot));
    if (module != NULL) {
        int slot = 0;
        Value key, value;
        while (jaiTableNext(&module->globals, &slot, &key, &value)) {
            jaiGCPushRoot(key);
            jaiGCPushRoot(value);
            (void)jaiDictSet(snapshot, key, value);
            jaiGCPopRoots(2);
        }
    }
    jaiGCPopRoot();

    *out = OBJ_VAL(snapshot);
    return true;
}

static bool nReflectBuildId(int argc, Value *args, Value *out) {
    (void)argc;
    (void)args;
    *out = INT_VAL((int64_t)jaiBuildId());
    return true;
}

void jaiRegisterReflectPrimitives(void) {
    jaiDefineNative("__prim__.jaic_build_id", nReflectBuildId,   0, 0);
    jaiDefineNative("__prim__.compile",     nReflectCompile,     1, 2);
    jaiDefineNative("__prim__.eval",        nReflectEval,        1, 1);
    jaiDefineNative("__prim__.exec",        nReflectExec,        1, 2);
    jaiDefineNative("__prim__.globals",     nReflectGlobals,     0, 0);
}
