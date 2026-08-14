/* errors.c — the built-in exception hierarchy of spec §7.2.
 * Built in C because the runtime throws before any Jaithon source loads.
 * jaiClassInherit pins Error's two fields at the same slots in every
 * subclass: slot 0 message, slot 1 traceback (VM-written only, no setter). */

#include "runtime/runtime.h"

#include "vm/gc.h"

#define FIELD_MESSAGE   0
#define FIELD_TRACEBACK 1

/* ------------------------------------------------------------------ */
/* Instance helpers                                                     */
/* ------------------------------------------------------------------ */

static Value fieldAt(ObjInstance *inst, uint16_t slot) {
    return slot < inst->fieldCount ? inst->fields[slot] : NULL_VAL;
}

static void setFieldAt(ObjInstance *inst, uint16_t slot, Value v) {
    if (slot < inst->fieldCount) inst->fields[slot] = v;
}

static const char *classNameOf(const ObjInstance *inst) {
    if (inst == NULL || inst->klass == NULL || inst->klass->name == NULL)
        return "Error";
    return inst->klass->name->chars;
}

/* Receiver is args[0] (spec: frame slot 0 is callee/self), argc counts it;
 * everything below reads its arguments from args[1] onward. */
static bool errorSelf(int argc, Value *args, const char *method,
                      ObjInstance **out) {
    if (argc >= 1 && args != NULL && IS_INSTANCE(args[0]) &&
        jaiClassIsSubclassOf(AS_INSTANCE(args[0])->klass, vm.cError)) {
        *out = AS_INSTANCE(args[0]);
        return true;
    }
    return jaiThrow(vm.cTypeError, "%s() needs an Error instance as its receiver",
                    method);
}

static bool appendText(JaiBuf *buf, Value v) {
    if (IS_STRING(v)) {
        jaiBufAppend(buf, AS_STRING(v)->chars, AS_STRING(v)->length);
        return true;
    }
    ObjString *text = jaiValueToStr(v);
    if (text == NULL) return false;
    jaiBufAppend(buf, text->chars, text->length);
    return true;
}

static bool takeString(JaiBuf *buf, Value *out) {
    size_t length = 0;
    char *chars = jaiBufTakeCString(buf, &length);
    if (chars == NULL) return jaiThrow(vm.cRuntimeError, "out of memory rendering an exception");
    ObjString *text = jaiStringTake(chars, length);
    if (text == NULL) return false;
    *out = OBJ_VAL(text);
    return true;
}

/* ------------------------------------------------------------------ */
/* Natives on Error                                                     */
/* ------------------------------------------------------------------ */

static bool errorInit(int argc, Value *args, Value *out) {
    ObjInstance *self;
    if (!errorSelf(argc, args, "Error.init", &self)) return false;

    Value message = (argc >= 2) ? args[1] : NULL_VAL;
    jaiGCPushRoot(OBJ_VAL(self));
    ObjString *text;
    if (IS_STRING(message)) {
        text = AS_STRING(message);
    } else if (IS_NULL(message)) {
        text = jaiStringIntern("", 0);
    } else {
        text = jaiValueToStr(message);
    }
    jaiGCPopRoot();
    if (text == NULL) return false;

    setFieldAt(self, FIELD_MESSAGE, OBJ_VAL(text));
    *out = OBJ_VAL(self);
    return true;
}

static bool errorStr(int argc, Value *args, Value *out) {
    ObjInstance *self;
    if (!errorSelf(argc, args, "Error.__str__", &self)) return false;

    Value message = fieldAt(self, FIELD_MESSAGE);
    if (IS_STRING(message)) {
        *out = message;
        return true;
    }
    jaiGCPushRoot(OBJ_VAL(self));
    ObjString *text = IS_NULL(message) ? jaiStringIntern("", 0)
                                       : jaiValueToStr(message);
    jaiGCPopRoot();
    if (text == NULL) return false;
    *out = OBJ_VAL(text);
    return true;
}

static bool errorRepr(int argc, Value *args, Value *out) {
    ObjInstance *self;
    if (!errorSelf(argc, args, "Error.__repr__", &self)) return false;

    jaiGCPushRoot(OBJ_VAL(self));
    Value message = fieldAt(self, FIELD_MESSAGE);
    if (IS_NULL(message)) message = OBJ_VAL(jaiStringIntern("", 0));
    ObjString *quoted = jaiValueToRepr(message);
    if (quoted == NULL) {
        jaiGCPopRoot();
        return false;
    }

    JaiBuf buf;
    jaiBufInit(&buf);
    jaiBufAppendStr(&buf, classNameOf(self));
    jaiBufPush(&buf, '(');
    jaiBufAppend(&buf, quoted->chars, quoted->length);
    jaiBufPush(&buf, ')');
    jaiGCPopRoot();
    return takeString(&buf, out);
}

static bool errorTraceback(int argc, Value *args, Value *out) {
    ObjInstance *self;
    if (!errorSelf(argc, args, "Error.__traceback__", &self)) return false;
    *out = fieldAt(self, FIELD_TRACEBACK);
    return true;
}

static bool errorTracebackString(int argc, Value *args, Value *out) {
    ObjInstance *self;
    if (!errorSelf(argc, args, "Error.traceback_string", &self)) return false;

    jaiGCPushRoot(OBJ_VAL(self));
    Value tb = fieldAt(self, FIELD_TRACEBACK);

    JaiBuf buf;
    jaiBufInit(&buf);
    bool ok = true;
    if (IS_LIST(tb)) {
        ObjList *frames = AS_LIST(tb);
        for (int i = 0; ok && i < frames->count; i++) {
            if (i > 0) jaiBufPush(&buf, '\n');
            ok = appendText(&buf, frames->items[i]);
        }
    } else if (!IS_NULL(tb)) {
        ok = appendText(&buf, tb);
    }
    jaiGCPopRoot();

    if (!ok) {
        jaiBufFree(&buf);
        return false;
    }
    return takeString(&buf, out);
}

/* ------------------------------------------------------------------ */
/* Building the hierarchy                                               */
/* ------------------------------------------------------------------ */

static void addNativeMethod(ObjClass *c, const char *name, JaiNativeFn fn,
                            int minArity, int maxArity, uint32_t flags) {
    ObjNative *native = jaiNativeNew(fn, name, minArity, maxArity, NULL);
    jaiGCPushRoot(OBJ_VAL(native));
    ObjString *methodName = jaiStringInternC(name);
    if (methodName != NULL) {
        jaiClassAddMethod(c, methodName, OBJ_VAL(native), VIS_PUBLIC, flags);
    }
    jaiGCPopRoot();
}

static void addErrorFields(ObjClass *error) {
    ObjString *messageName =
        (vm.strMessage != NULL) ? vm.strMessage : jaiStringInternC("message");
    jaiGCPushRoot(OBJ_VAL(messageName));
    ObjString *tracebackName = jaiStringInternC("traceback");
    jaiGCPushRoot(OBJ_VAL(tracebackName));

    FieldInfo *fields = JAI_ALLOC(FieldInfo, 2);
    fields[FIELD_MESSAGE] = (FieldInfo){
        .name = messageName, .slot = FIELD_MESSAGE, .visibility = VIS_PUBLIC,
        .isStatic = false, .isLet = false, .typeId = 0,
    };
    fields[FIELD_TRACEBACK] = (FieldInfo){
        .name = tracebackName, .slot = FIELD_TRACEBACK, .visibility = VIS_PUBLIC,
        .isStatic = false, .isLet = false, .typeId = 0,
    };
    error->fields = fields;
    error->fieldCount = 2;

    jaiGCPopRoots(2);
}

static ObjClass *defineErrorClass(const char *name, ObjClass *super) {
    ObjString *className = jaiStringInternC(name);
    jaiGCPushRoot(OBJ_VAL(className));
    ObjClass *c = jaiClassNew(className, super);
    jaiGCPushRoot(OBJ_VAL(c));

    if (super != NULL) jaiClassInherit(c, super);
    jaiDefineGlobal(name, OBJ_VAL(c));

    jaiGCPopRoots(2);
    return c;
}

void jaiRegisterErrorClasses(void) {
    ObjClass *error = defineErrorClass("Error", NULL);
    vm.cError = error;                    /* also makes it a GC root */
    addErrorFields(error);
    addNativeMethod(error, "init", errorInit, 1, 2, FN_METHOD | FN_INIT);
    addNativeMethod(error, "__str__", errorStr, 1, 1, FN_METHOD);
    addNativeMethod(error, "__repr__", errorRepr, 1, 1, FN_METHOD);
    addNativeMethod(error, "__traceback__", errorTraceback, 1, 1, FN_METHOD);
    addNativeMethod(error, "traceback_string", errorTracebackString, 1, 1,
                    FN_METHOD);

    vm.cAssertionError    = defineErrorClass("AssertionError", error);
    vm.cArithmeticError   = defineErrorClass("ArithmeticError", error);
    vm.cDivisionByZeroError =
        defineErrorClass("DivisionByZeroError", vm.cArithmeticError);
    vm.cOverflowError     = defineErrorClass("OverflowError", vm.cArithmeticError);
    vm.cLookupError       = defineErrorClass("LookupError", error);
    vm.cIndexError        = defineErrorClass("IndexError", vm.cLookupError);
    vm.cKeyError          = defineErrorClass("KeyError", vm.cLookupError);
    vm.cNameError         = defineErrorClass("NameError", error);
    vm.cTypeError         = defineErrorClass("TypeError", error);
    vm.cValueError        = defineErrorClass("ValueError", error);
    vm.cParseError        = defineErrorClass("ParseError", vm.cValueError);
    vm.cAttributeError    = defineErrorClass("AttributeError", error);
    vm.cIOError           = defineErrorClass("IOError", error);
    vm.cFileNotFoundError = defineErrorClass("FileNotFoundError", vm.cIOError);
    vm.cPermissionError   = defineErrorClass("PermissionError", vm.cIOError);
    vm.cOSError           = defineErrorClass("OSError", error);
    vm.cRuntimeError      = defineErrorClass("RuntimeError", error);
    vm.cRecursionError    = defineErrorClass("RecursionError", vm.cRuntimeError);
    vm.cStopIteration     = defineErrorClass("StopIteration", vm.cRuntimeError);
    vm.cImportError       = defineErrorClass("ImportError", error);
}

/* ------------------------------------------------------------------ */
/* Constructing an exception from C                                     */
/* ------------------------------------------------------------------ */

Value jaiMakeException(ObjClass *klass, const char *message) {
    ObjClass *k = (klass != NULL) ? klass : vm.cError;
    if (k == NULL) {
        JAI_PANIC("jaiMakeException called before jaiRegisterErrorClasses");
    }

    jaiGCPushRoot(OBJ_VAL(k));
    ObjInstance *inst = jaiInstanceNew(k);
    jaiGCPushRoot(OBJ_VAL(inst));

    ObjString *text = jaiStringNew(message != NULL ? message : "",
                                   message != NULL ? strlen(message) : 0);
    if (text == NULL) {
        /* Only a >4 GiB message gets here, and jaiStringNew already threw;
         * that exception must not displace the one being constructed. */
        jaiClearException();
        text = jaiStringIntern("", 0);
    }
    jaiGCPushRoot(text != NULL ? OBJ_VAL(text) : NULL_VAL);

    ObjString *name = (vm.strMessage != NULL) ? vm.strMessage
                                              : jaiStringInternC("message");
    int slot = (name != NULL) ? jaiClassFieldSlot(k, name) : -1;
    if (slot >= 0 && text != NULL) setFieldAt(inst, (uint16_t)slot, OBJ_VAL(text));

    jaiGCPopRoots(3);
    return OBJ_VAL(inst);
}
