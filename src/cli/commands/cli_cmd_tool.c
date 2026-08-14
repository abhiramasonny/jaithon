/* cli_cmd_tool.c — fmt, test, doc and bench, which are written in Jaithon.
 *
 * There is deliberately no C implementation of these four here: the formatter
 * is canonical (spec §14) and having two of it is exactly the Jaithon 2
 * mistake. Instead runJaithonTool imports the `jaithon.tool.*` module and
 * calls its `main`, forwarding the command line the user actually wrote.
 */
#include "cli/cli_internal.h"

static inline void toolPush(ObjList *args, const char *text) {
    ObjString *const s = jaiStringInternC(text);

    JAI_ASSERT(args->count < args->capacity,
               "tool argument list exceeded its reserved capacity");

    args->items[args->count++] = OBJ_VAL(s);
    args->version++;
}

static ObjList *toolArguments(const JaiCliOptions *opts) {
    ObjList *args = jaiListNew(opts->toolArgCount + 4);
    jaiPushRoot(OBJ_VAL(args));

    if (opts->fmtCheck)   toolPush(args, "--check");
    if (opts->jsonOutput) toolPush(args, "--json");
    if (opts->output != NULL) {
        toolPush(args, "--out");
        toolPush(args, opts->output);
    }
    for (int i = 0; i < opts->toolArgCount; i++) toolPush(args, opts->toolArgs[i]);
    return args;   /* still rooted; the caller pops */
}

/* True when the tool's `main` should be called with no arguments at all. */
static inline bool toolMainTakesNoArgs(Value entry) {
    const ObjFunction *fn = NULL;
    if (IS_CLOSURE(entry)) fn = AS_CLOSURE(entry)->fn;
    else if (IS_FUNCTION(entry)) fn = AS_FUNCTION(entry);
    else if (IS_NATIVE(entry)) return AS_NATIVE(entry)->maxArity == 0;
    if (fn == NULL) return false;
    return fn->arity == 0 && fn->paramCount == 0 &&
           (fn->flags & (FN_VARIADIC | FN_KWREST)) == 0;
}

int runJaithonTool(const JaiCliOptions *opts, const char *tool) {
    static const char prefix[] = "jaithon.tool.";
    char moduleName[64];

    const size_t toolLen = strlen(tool);
    JAI_ASSERT(sizeof prefix + toolLen <= sizeof moduleName,
               "tool module name exceeds local buffer");

    memcpy(moduleName, prefix, sizeof prefix - 1);
    memcpy(moduleName + sizeof prefix - 1, tool, toolLen + 1);

    ObjModule *module = jaiImportModule(moduleName, NULL);
    if (module == NULL) {
        jaiClearException();
        JaiDiag *d = jaiDiagError(E0800_MODULE_NOT_FOUND, JAI_SPAN_NONE,
                                  "`%s` is written in Jaithon and needs the "
                                  "standard library module `%s`",
                                  tool, moduleName);
        jaiDiagAddHelp(d, "install the standard library, or point JAITHON_PATH "
                          "at the directory that contains `std`");
        (void)cliFlush();
        return 1;
    }
    jaiPushRoot(OBJ_VAL(module));

    Value entry;
    if (!jaiModuleGet(module, jaiStringInternC("main"), &entry)) {
        (void)jaiDiagError(E0802_NOT_EXPORTED, JAI_SPAN_NONE,
                           "module `%s` does not define `main`", moduleName);
        (void)cliFlush();
        jaiPopRoot();
        return 1;
    }

    bool ok;
    Value result = NULL_VAL;
    if (toolMainTakesNoArgs(entry)) {
        ok = jaiCallValue(entry, 0, NULL, &result);
    } else {
        ObjList *args = toolArguments(opts);
        Value argument = OBJ_VAL(args);
        ok = jaiCallValue(entry, 1, &argument, &result);
        jaiPopRoot();
    }
    jaiPopRoot();

    if (!ok) {
        jaiReportUncaught(vm.pendingException);
        jaiClearException();
        return 1;
    }
    if (IS_INT(result)) {
        int64_t code = AS_INT(result);
        return (int)(code & 0xff);
    }
    return 0;
}
