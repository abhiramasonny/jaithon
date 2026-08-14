#include "runtime/modules/frontend.h"

#include <string.h>

#include "runtime/runtime.h"
#include "common/diag.h"
#include "vm/gc.h"
#include "vm/bytecode/serialize.h"

#define JAI_REPL_MODULE "jaithon.compile.repl"

bool jaiFrontEndField(Value v, const char *name, Value *out) {
    if (!IS_INSTANCE(v)) return false;
    ObjInstance *inst = AS_INSTANCE(v);
    int slot = jaiClassFieldSlot(inst->klass, jaiStringInternC(name));
    if (slot < 0 || slot >= (int)inst->fieldCount) return false;
    *out = inst->fields[slot];
    return true;
}

bool jaiFrontEndCall0(Value v, const char *name, Value *out) {
    if (!IS_INSTANCE(v)) return false;
    ObjInstance *inst = AS_INSTANCE(v);

    Value method;
    if (!jaiTableGetInterned(&inst->klass->methods, jaiStringInternC(name),
                             &method)) {
        return false;
    }

    jaiPushRoot(v);
    Value bound = OBJ_VAL(jaiBoundNew(v, method));
    jaiPushRoot(bound);
    bool ok = jaiCallValue(bound, 0, NULL, out);
    jaiPopRoots(2);

    if (!ok) jaiClearException();
    return ok;
}

bool jaiFrontEndInvoke(const char *module, const char *name, int argc,
                       Value *args, Value *out) {
    ObjModule *owner = jaiImportFrontEndModule(module);
    if (owner == NULL) {
        jaiClearException();
        return false;
    }
    jaiPushRoot(OBJ_VAL(owner));

    Value entry;
    if (!jaiModuleGet(owner, jaiStringInternC(name), &entry)) {
        jaiPopRoot();
        return false;
    }

    bool ok = jaiCallValue(entry, argc, args, out);
    jaiPopRoot();

    if (!ok) {
        jaiReportUncaught(vm.pendingException);
        jaiClearException();
    }
    return ok;
}

static char firstChar(Value v) {
    if (!IS_STRING(v)) return '\0';
    ObjString *s = AS_STRING(v);
    return s->length > 0 ? s->chars[0] : '\0';
}

static int intField(Value scan, const char *name) {
    Value field;
    if (!jaiFrontEndField(scan, name, &field) || !IS_INT(field)) return 0;
    return (int)AS_INT(field);
}

static bool boolField(Value scan, const char *name) {
    Value field;
    if (!jaiFrontEndField(scan, name, &field)) return false;
    return IS_BOOL(field) && AS_BOOL(field);
}

static char charField(Value scan, const char *name) {
    Value field;
    if (!jaiFrontEndField(scan, name, &field)) return '\0';
    return firstChar(field);
}

bool jaiFrontEndReplScan(const char *source, size_t length, JaiReplScan *out) {
    memset(out, 0, sizeof *out);
    if (source == NULL) return false;

    Value arg = OBJ_VAL(jaiStringNew(source, length));
    jaiPushRoot(arg);
    Value produced = NULL_VAL;
    bool called = jaiFrontEndInvoke(JAI_REPL_MODULE, "scan", 1, &arg, &produced);
    jaiPopRoot();
    if (!called || !IS_INSTANCE(produced)) return false;

    jaiPushRoot(produced);
    out->incomplete = boolField(produced, "incomplete");
    out->mismatched = boolField(produced, "mismatched");
    out->opener     = charField(produced, "opener");
    out->openerLine = intField(produced, "opener_line");
    out->closer     = charField(produced, "closer");
    out->closerLine = intField(produced, "closer_line");
    out->closerCol  = intField(produced, "closer_col");

    Value ordinal = NULL_VAL;
    bool gotOrdinal = jaiFrontEndCall0(produced, "open_ordinal", &ordinal) &&
                      IS_INT(ordinal);
    if (gotOrdinal) out->open = (JaiReplOpenKind)AS_INT(ordinal);
    jaiPopRoot();

    if (!gotOrdinal) {
        memset(out, 0, sizeof *out);
        return false;
    }
    return true;
}

/* ------------------------------------------------------------------ */
/* The prompt's compile path                                            */
/* ------------------------------------------------------------------ */

#define JAI_FRONT_END_MODULE "jaithon.compile"

static ObjBytes *imageBytes(Value compiled) {
    Value field;
    if (!jaiFrontEndField(compiled, "image", &field) || !IS_LIST(field)) {
        return NULL;
    }
    ObjList *list = AS_LIST(field);
    if (list->count == 0) return NULL;

    int count = list->count;
    uint8_t *raw = JAI_ALLOC(uint8_t, count);
    for (int i = 0; i < count; i++) {
        Value item = list->items[i];
        if (!IS_INT(item)) {
            JAI_FREE_ARRAY(uint8_t, raw, count);
            return NULL;
        }
        raw[i] = (uint8_t)(AS_INT(item) & 0xFF);
    }
    ObjBytes *bytes = jaiBytesNew(raw, count);
    JAI_FREE_ARRAY(uint8_t, raw, count);
    return bytes;
}

static Value sSession = NULL_VAL;
static bool  sSessionTried;

static bool sessionReady(void) {
    if (!IS_NULL(sSession)) return true;
    if (sSessionTried) return false;
    sSessionTried = true;

    ObjModule *owner = jaiImportFrontEndModule(JAI_FRONT_END_MODULE);
    if (owner == NULL) { jaiClearException(); return false; }
    jaiPushRoot(OBJ_VAL(owner));

    Value klass;
    bool got = jaiModuleGet(owner, jaiStringInternC("ReplSession"), &klass);
    jaiPopRoot();
    if (!got) return false;

    Value made = NULL_VAL;
    if (!jaiCallValue(klass, 0, NULL, &made)) {
        jaiClearException();
        return false;
    }
    sSession = made;
    jaiGCAddPermanentRoot(sSession);
    return true;
}

void jaiFrontEndReplForget(void) {
    if (IS_NULL(sSession)) return;
    Value ignored = NULL_VAL;
    (void)jaiFrontEndCall0(sSession, "reset", &ignored);
}

ObjFunction *jaiFrontEndReplCompile(const char *source, size_t length,
                                    const JaiReplCompileOptions *opts,
                                    ObjModule *module, bool *outWasExpression) {
    if (outWasExpression != NULL) *outWasExpression = false;
    if (!sessionReady()) {
        (void)jaiDiagError(E0800_MODULE_NOT_FOUND, JAI_SPAN_NONE,
                           "the prompt needs the self-hosted front end in `%s`, "
                           "which could not be imported", JAI_FRONT_END_MODULE);
        return NULL;
    }

    Value args[11];
    args[0] = sSession;
    args[1] = OBJ_VAL(jaiStringNew(source, length));
    jaiPushRoot(args[1]);
    args[2] = OBJ_VAL(jaiStringInternC(opts->path));
    jaiPushRoot(args[2]);
    args[3] = INT_VAL(opts->fileId);
    args[4] = INT_VAL(opts->optLevel);
    args[5] = OBJ_VAL(jaiStringInternC(opts->echo != NULL ? opts->echo : ""));
    jaiPushRoot(args[5]);
    args[6] = BOOL_VAL(opts->wholeFile);
    args[7] = BOOL_VAL(opts->record);
    args[8] = OBJ_VAL(jaiStringInternC(opts->sourceDir != NULL ? opts->sourceDir
                                                               : ""));
    jaiPushRoot(args[8]);
    args[9] = BOOL_VAL(opts->strict);
    args[10] = BOOL_VAL(opts->lateGlobals);

    Value produced = NULL_VAL;
    bool called = jaiFrontEndInvoke(JAI_FRONT_END_MODULE, "compile_repl", 11,
                                    args, &produced);
    jaiPopRoots(4);
    if (!called || !IS_INSTANCE(produced)) return NULL;

    jaiPushRoot(produced);
    if (outWasExpression != NULL) {
        *outWasExpression = boolField(produced, "repl_expression");
    }

    (void)jaiFrontEndTransferDiagnostics(produced);

    ObjBytes *image = imageBytes(produced);
    if (image == NULL) { jaiPopRoot(); return NULL; }
    jaiPushRoot(OBJ_VAL(image));

    Value hash;
    uint64_t expected = 0;
    if (jaiFrontEndField(produced, "hash", &hash) && IS_INT(hash)) {
        expected = (uint64_t)AS_INT(hash);
    }

    ObjFunction *body = jaiDeserializeModule(image->data, image->length, module,
                                             expected);
    if (body == NULL) {
        (void)jaiDiagError(E0902_INTERNAL_ERROR, JAI_SPAN_NONE,
                           "%s: the prompt's front end produced a .jaic image "
                           "this build cannot load", opts->path);
    }
    jaiPopRoots(2);
    return body;
}

/* ------------------------------------------------------------------ */
/* Diagnostics                                                          */
/* ------------------------------------------------------------------ */

static JaiSpan spanFrom(Value v) {
    JaiSpan span = JAI_SPAN_NONE;
    if (!IS_INSTANCE(v)) return span;

    Value file, start, end;
    if (!jaiFrontEndField(v, "file", &file) || !IS_INT(file)) return span;
    if (!jaiFrontEndField(v, "start", &start) || !IS_INT(start)) return span;
    if (!jaiFrontEndField(v, "end", &end) || !IS_INT(end)) return span;

    span.file  = (int32_t)AS_INT(file);
    span.start = (uint32_t)AS_INT(start);
    span.end   = (uint32_t)AS_INT(end);
    return span;
}

static const char *stringOrNull(Value v) {
    return IS_STRING(v) ? AS_STRING(v)->chars : NULL;
}

static void transferOne(Value diagnostic) {
    Value codeValue, messageValue;
    if (!jaiFrontEndField(diagnostic, "code", &codeValue)) return;
    if (!jaiFrontEndField(diagnostic, "message", &messageValue)) return;

    const char *codeText = stringOrNull(codeValue);
    const char *message = stringOrNull(messageValue);
    if (message == NULL) return;

    JaiDiagCode code = jaiDiagCodeFromString(codeText);
    JaiSpan primary = JAI_SPAN_NONE;
    Value spanValue;
    if (jaiFrontEndField(diagnostic, "span", &spanValue)) {
        primary = spanFrom(spanValue);
    }

    JaiDiag *d = (codeText != NULL && codeText[0] == 'W')
                     ? jaiDiagWarn(code, primary, "%s", message)
                     : jaiDiagError(code, primary, "%s", message);
    if (d == NULL) return;

    Value labels;
    if (jaiFrontEndField(diagnostic, "labels", &labels) && IS_LIST(labels)) {
        ObjList *list = AS_LIST(labels);
        for (int i = 0; i < list->count; i++) {
            if (!IS_TUPLE(list->items[i])) continue;
            ObjTuple *pair = AS_TUPLE(list->items[i]);
            if (pair->count < 2) continue;
            const char *text = stringOrNull(pair->items[1]);
            jaiDiagAddLabel(d, spanFrom(pair->items[0]), "%s",
                            text != NULL ? text : "");
        }
    }

    Value help, note;
    if (jaiFrontEndField(diagnostic, "help", &help) && IS_STRING(help)) {
        jaiDiagAddHelp(d, "%.*s", (int)AS_STRING(help)->length,
                       AS_STRING(help)->chars);
    }
    if (jaiFrontEndField(diagnostic, "note", &note) && IS_STRING(note)) {
        jaiDiagAddNote(d, "%.*s", (int)AS_STRING(note)->length,
                       AS_STRING(note)->chars);
    }
}

bool jaiFrontEndTransferDiagnostics(Value compiled) {
    /* `Compiled` carries a bag; `import_cycles` answers one directly. Both are
     * accepted so a caller need not know which shape its entry point returns. */
    Value bag;
    if (!jaiFrontEndField(compiled, "diagnostics", &bag)) bag = compiled;
    Value items;
    if (!jaiFrontEndField(bag, "items", &items) || !IS_LIST(items)) return false;

    ObjList *list = AS_LIST(items);
    jaiPushRoot(items);
    for (int i = 0; i < list->count; i++) transferOne(list->items[i]);
    jaiPopRoot();
    return list->count > 0;
}

/* ------------------------------------------------------------------ */
/* Dumps                                                                */
/* ------------------------------------------------------------------ */

static ObjString *dumpText(const char *module, const char *entry,
                           const char *source, size_t length, int fileId,
                           bool wantsJson) {
    Value args[3];
    args[0] = OBJ_VAL(jaiStringNew(source, length));
    jaiPushRoot(args[0]);
    args[1] = INT_VAL(fileId);
    args[2] = BOOL_VAL(wantsJson);

    Value produced = NULL_VAL;
    bool called = jaiFrontEndInvoke(module, entry, wantsJson ? 3 : 2, args,
                                    &produced);
    jaiPopRoot();
    if (!called || !IS_STRING(produced)) return NULL;
    return AS_STRING(produced);
}

ObjString *jaiFrontEndAstText(const char *source, size_t length,
                              const char *path, int fileId, bool json) {
    (void)path;
    return dumpText(JAI_FRONT_END_MODULE, "dump_ast", source, length, fileId,
                    json);
}

ObjString *jaiFrontEndTokenText(const char *source, size_t length, int fileId) {
    return dumpText("jaithon.compile.lexer", "dump_tokens", source, length,
                    fileId, false);
}
