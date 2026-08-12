/* builtins_reflect.c — compile, eval, exec, globals, and the build-id
 * primitive eval needs to stamp what it compiles (spec Appendix C).
 *
 * Split out of builtins_io.c, which explains what else moved and why. This
 * group is self-contained: it shares no helper with the files that stayed or
 * moved alongside it, because it is the one primitive family here that talks
 * to the front end rather than to the operating system.
 *
 * The reflection primitives all go through jaiCompileSource, so `eval`, the
 * REPL and lib/jaithon/compile see exactly one front end.
 */

#include "runtime.h"

#include "../vm/gc.h"
#include "../vm/serialize.h"

/* The binding eval() compiles into and then removes. It is not of the form
 * `__x__`, which the parser reserves (§2.1), and it starts with an underscore,
 * which is what stops the checker warning that it is never read. */
static const char kEvalSlot[] = "__jai_eval";

/* A native runs on the caller's frame, so the caller's module is the top one.
 * That is what makes eval() and globals() see the names the caller sees. */
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

/* A compile error inside a running program cannot be printed — the caller is
 * mid-expression and expects an exception — so the first diagnostic becomes
 * the message and the bag is emptied rather than left to leak into whatever
 * flushes it next. */
static bool throwCompileError(const char *fnName) {
    char message[512];
    const JaiDiag *first = NULL;
    for (int i = 0; i < gDiags.diags.count && first == NULL; i++) {
        if (gDiags.diags.data[i].severity == JAI_SEV_ERROR)
            first = &gDiags.diags.data[i];
    }
    /* With --warnings-as-errors the failure may be recorded as a warning; it
     * is still the reason the compile produced nothing. */
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

/* Compile a fragment as a module body of `module`. Returns NULL with the
 * exception already raised. */
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

/* Runs a fragment by calling it like any other closure. jaiVMRunModule is not
 * an option here: it resets the value stack, which the native's own caller is
 * still standing on. */
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

/* An expression is evaluated by binding it to a module-level name and reading
 * that name back: the module body is the only thing the code generator emits,
 * and a body discards the value of an expression statement. The binding is
 * removed afterwards so eval leaves no trace in the module. */
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
        /* Deleting a name can take a class or a callable away, so compiled
         * forms have to be retired. The table's own keyVersion covers the
         * cached slot addresses and OP_FORMAT's negative cache; removeEntry
         * bumps it, so there is nothing to do for those here. */
        if (jaiTableDelete(&module->globals, OBJ_VAL(slot))) module->version++;
    }
    *out = value;
    return true;
}

/* The namespace a two-argument `exec` runs in: a module of its own, made on
 * first use and kept in vm.modules so that a later call sees what an earlier
 * one bound. jaithon.tool.test needs this — running a test file in the runner's
 * own namespace lets an `import std.str as str` in that file shadow the
 * runner's `str`, and the runner then fails inside its own reporting code. */
static ObjModule *execNamespace(ObjString *name) {
    /* vm.modules is keyed by pointer and `name` is whatever the caller passed. */
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

/* The §7 `buildId` this binary writes into every .jaic and demands back out of
 * one. It is a hash of the C sources, so the self-hosted front end cannot
 * derive it and has to be told; a `.jaic` it writes without this stamp is
 * rejected by the reader with the header otherwise perfectly well formed. */
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
