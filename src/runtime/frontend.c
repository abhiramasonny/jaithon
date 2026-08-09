#include "frontend.h"

#include <string.h>

#include "runtime.h"

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
