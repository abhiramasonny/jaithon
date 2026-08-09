#include "frontend.h"

#include <string.h>

#include "runtime.h"
#include "../common/diag.h"
#include "../vm/gc.h"
#include "../vm/serialize.h"

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
    /* Through the front-end door, not the ordinary one: this module is part of
     * the compiler, so the seed may serve it. Only the import needs the
     * window; the call below is ordinary Jaithon code. */
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
        /* A front end that raises is a bug, and the traceback is the only
         * thing that can say where. */
        jaiReportUncaught(vm.pendingException);
        jaiClearException();
    }
    return ok;
}

/* The first character of a Jaithon string, or '\0' for an empty one. Every
 * opener and closer the scan reports is one ASCII byte. */
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

    /* `open` is a Jaithon enum, which has no C spelling; the module answers its
     * ordinal instead. A verdict the driver cannot read is worse than none, so
     * a missing accessor fails the whole scan rather than silently reporting
     * JAI_REPL_OPEN_NONE for a bracket. */
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

/* `Compiled.image` is a list of byte-sized ints: Jaithon has no writable bytes
 * type, so the front end returns the container that way. */
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

/* The session object, made once and held for the process. A prompt is one
 * conversation; nothing about it is per-input. */
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

    Value args[10];
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

    Value produced = NULL_VAL;
    bool called = jaiFrontEndInvoke(JAI_FRONT_END_MODULE, "compile_repl", 10,
                                    args, &produced);
    jaiPopRoots(4);
    if (!called || !IS_INSTANCE(produced)) return NULL;

    jaiPushRoot(produced);
    if (outWasExpression != NULL) {
        *outWasExpression = boolField(produced, "repl_expression");
    }

    /* Diagnostics first: an input can be rejected with an image already empty,
     * and they are the whole of what the prompt has to say about it. Into the
     * bag rather than onto stderr, so the driver renders them the way it
     * renders everything else. */
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

/* The front end reports into a Jaithon `DiagnosticBag`, and rendering one as
 * text is all `Compiled.report` can do: a flat line per diagnostic, with no
 * source excerpt, no caret and no help.
 *
 * That is not what a diagnostic is supposed to look like, and printing it that
 * way is what the driver did between the day the self-hosted front end became
 * the default and the day this was written. `tests/errors` never noticed --
 * it greps the output for the code and nothing else -- and only the REPL
 * goldens, which pin the rendered form, ever failed.
 *
 * So the diagnostics are rebuilt as `JaiDiag` and put in the bag the driver
 * already flushes. `jaiDiagRender` then produces the same output for both
 * front ends, which is the only way the two can be told apart by what they
 * accept rather than by how they complain. */

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

    /* Severity comes off the code rather than the record's own field: the
     * letter and the numeric range say the same thing here, and reading a
     * Jaithon enum from C costs a call. */
    JaiDiag *d = (codeText != NULL && codeText[0] == 'W')
                     ? jaiDiagWarn(code, primary, "%s", message)
                     : jaiDiagError(code, primary, "%s", message);
    if (d == NULL) return;   /* the bag is full, or warnings are suppressed */

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
        jaiDiagAddHelp(d, "%s", AS_STRING(help)->chars);
    }
    if (jaiFrontEndField(diagnostic, "note", &note) && IS_STRING(note)) {
        jaiDiagAddNote(d, "%s", AS_STRING(note)->chars);
    }
}

bool jaiFrontEndTransferDiagnostics(Value compiled) {
    Value bag;
    if (!jaiFrontEndField(compiled, "diagnostics", &bag)) return false;
    Value items;
    if (!jaiFrontEndField(bag, "items", &items) || !IS_LIST(items)) return false;

    ObjList *list = AS_LIST(items);
    jaiPushRoot(items);
    for (int i = 0; i < list->count; i++) transferOne(list->items[i]);
    jaiPopRoot();
    return list->count > 0;
}
