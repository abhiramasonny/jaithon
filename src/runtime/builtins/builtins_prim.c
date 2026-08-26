/* builtins_prim.c — the `__prim__` operator surface (spec Appendix C); reachable only through std, never by name from user code. */

#include <math.h>
#include <stdlib.h>

#include "runtime/builtins/builtins.h"
#include "runtime/methods.h"
#include "runtime/runtime.h"

#include "vm/gc.h"
#include "vm/object/object.h"
#include "vm/trace/trace.h"

bool jaiBuiltinAddI64(int64_t a, int64_t b, int64_t *out) {
    if ((b > 0 && a > INT64_MAX - b) || (b < 0 && a < INT64_MIN - b)) return false;
    *out = a + b;
    return true;
}

static bool subI64(int64_t a, int64_t b, int64_t *out) {
    if ((b < 0 && a > INT64_MAX + b) || (b > 0 && a < INT64_MIN + b)) return false;
    *out = a - b;
    return true;
}

static bool mulI64(int64_t a, int64_t b, int64_t *out) {
    if (a == 0 || b == 0) { *out = 0; return true; }
    if (a == -1) { if (b == INT64_MIN) return false; *out = -b; return true; }
    if (b == -1) { if (a == INT64_MIN) return false; *out = -a; return true; }
    if (a > 0) {
        if (b > 0) { if (a > INT64_MAX / b) return false; }
        else       { if (b < INT64_MIN / a) return false; }
    } else {
        if (b > 0) { if (a < INT64_MIN / b) return false; }
        else       { if (b < INT64_MAX / a) return false; }
    }
    *out = a * b;
    return true;
}

bool jaiBuiltinOverflowError(const char *op) {
    return jaiThrow(vm.cOverflowError, "integer overflow in %s", op);
}

static bool floorDivI64(int64_t a, int64_t b, int64_t *out) {
    if (b == 0) return jaiThrow(vm.cDivisionByZeroError, "integer division by zero");
    if (a == INT64_MIN && b == -1) return jaiBuiltinOverflowError("//");
    int64_t q = a / b;
    int64_t r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) q--;
    *out = q;
    return true;
}

static bool modI64(int64_t a, int64_t b, int64_t *out) {
    if (b == 0) return jaiThrow(vm.cDivisionByZeroError, "integer modulo by zero");
    if (a == INT64_MIN && b == -1) { *out = 0; return true; }
    int64_t r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) r += b;
    *out = r;
    return true;
}

static double floorDivF64(double a, double b) { return floor(a / b); }

static double modF64(double a, double b) {
    double r = fmod(a, b);
    if (r != 0.0 && ((r < 0.0) != (b < 0.0))) r += b;
    return r;
}

static bool powI64(int64_t base, int64_t exp, int64_t *out) {
    int64_t result = 1;
    int64_t b = base;
    int64_t e = exp;
    while (e > 0) {
        if ((e & 1) != 0 && !mulI64(result, b, &result)) return false;
        e >>= 1;
        if (e > 0 && !mulI64(b, b, &b)) return false;
    }
    *out = result;
    return true;
}

static bool primOperands(Value a, Value b, const char *name) {
    if (!IS_NUMBER(a)) return jaiBuiltinArgTypeError(1, name, "int or float", a);
    if (!IS_NUMBER(b)) return jaiBuiltinArgTypeError(2, name, "int or float", b);
    return true;
}

static bool nPrimAdd(int argc, Value *args, Value *out) {
    (void)argc;
    if (!primOperands(args[0], args[1], "add")) return false;
    if (IS_INT(args[0]) && IS_INT(args[1])) {
        int64_t sum;
        if (!jaiBuiltinAddI64(AS_INT(args[0]), AS_INT(args[1]), &sum)) return jaiBuiltinOverflowError("+");
        *out = INT_VAL(sum);
        return true;
    }
    *out = FLOAT_VAL(jaiAsDouble(args[0]) + jaiAsDouble(args[1]));
    return true;
}

static bool nPrimSub(int argc, Value *args, Value *out) {
    (void)argc;
    if (!primOperands(args[0], args[1], "sub")) return false;
    if (IS_INT(args[0]) && IS_INT(args[1])) {
        int64_t diff;
        if (!subI64(AS_INT(args[0]), AS_INT(args[1]), &diff)) return jaiBuiltinOverflowError("-");
        *out = INT_VAL(diff);
        return true;
    }
    *out = FLOAT_VAL(jaiAsDouble(args[0]) - jaiAsDouble(args[1]));
    return true;
}

static bool nPrimMul(int argc, Value *args, Value *out) {
    (void)argc;
    if (!primOperands(args[0], args[1], "mul")) return false;
    if (IS_INT(args[0]) && IS_INT(args[1])) {
        int64_t product;
        if (!mulI64(AS_INT(args[0]), AS_INT(args[1]), &product)) return jaiBuiltinOverflowError("*");
        *out = INT_VAL(product);
        return true;
    }
    *out = FLOAT_VAL(jaiAsDouble(args[0]) * jaiAsDouble(args[1]));
    return true;
}

static bool nPrimDiv(int argc, Value *args, Value *out) {
    (void)argc;
    if (!primOperands(args[0], args[1], "div")) return false;
    double divisor = jaiAsDouble(args[1]);
    if (divisor == 0.0) return jaiThrow(vm.cDivisionByZeroError, "division by zero");
    *out = FLOAT_VAL(jaiAsDouble(args[0]) / divisor);
    return true;
}

static bool nPrimFloorDiv(int argc, Value *args, Value *out) {
    (void)argc;
    if (!primOperands(args[0], args[1], "floordiv")) return false;
    if (IS_INT(args[0]) && IS_INT(args[1])) {
        int64_t quotient;
        if (!floorDivI64(AS_INT(args[0]), AS_INT(args[1]), &quotient)) return false;
        *out = INT_VAL(quotient);
        return true;
    }
    double divisor = jaiAsDouble(args[1]);
    if (divisor == 0.0) return jaiThrow(vm.cDivisionByZeroError, "float floor division by zero");
    *out = FLOAT_VAL(floorDivF64(jaiAsDouble(args[0]), divisor));
    return true;
}

static bool nPrimMod(int argc, Value *args, Value *out) {
    (void)argc;
    if (!primOperands(args[0], args[1], "mod")) return false;
    if (IS_INT(args[0]) && IS_INT(args[1])) {
        int64_t rest;
        if (!modI64(AS_INT(args[0]), AS_INT(args[1]), &rest)) return false;
        *out = INT_VAL(rest);
        return true;
    }
    double divisor = jaiAsDouble(args[1]);
    if (divisor == 0.0) return jaiThrow(vm.cDivisionByZeroError, "float modulo by zero");
    *out = FLOAT_VAL(modF64(jaiAsDouble(args[0]), divisor));
    return true;
}

static bool nPrimPow(int argc, Value *args, Value *out) {
    (void)argc;
    if (!primOperands(args[0], args[1], "pow")) return false;
    if (IS_INT(args[0]) && IS_INT(args[1])) {
        int64_t base = AS_INT(args[0]), exponent = AS_INT(args[1]);
        if (exponent >= 0) {
            int64_t power;
            if (!powI64(base, exponent, &power)) return jaiBuiltinOverflowError("**");
            *out = INT_VAL(power);
            return true;
        }
        if (base == 0) return jaiThrow(vm.cDivisionByZeroError,
                                       "zero to a negative power");
        *out = FLOAT_VAL(pow((double)base, (double)exponent));
        return true;
    }
    *out = FLOAT_VAL(pow(jaiAsDouble(args[0]), jaiAsDouble(args[1])));
    return true;
}

static bool nPrimNeg(int argc, Value *args, Value *out) {
    (void)argc;
    Value v = args[0];
    if (IS_INT(v)) {
        if (AS_INT(v) == INT64_MIN) return jaiBuiltinOverflowError("unary -");
        *out = INT_VAL(-AS_INT(v));
        return true;
    }
    if (IS_FLOAT(v)) { *out = FLOAT_VAL(-AS_FLOAT(v)); return true; }
    return jaiBuiltinArgTypeError(1, "neg", "int or float", v);
}

static bool bitOperands(Value a, Value b, const char *name, int64_t *x, int64_t *y) {
    if (!jaiArgInt(a, 1, name, x)) return false;
    if (!jaiArgInt(b, 2, name, y)) return false;
    return true;
}

static bool nPrimBand(int argc, Value *args, Value *out) {
    (void)argc;
    int64_t a, b;
    if (!bitOperands(args[0], args[1], "band", &a, &b)) return false;
    *out = INT_VAL(a & b);
    return true;
}

static bool nPrimBor(int argc, Value *args, Value *out) {
    (void)argc;
    int64_t a, b;
    if (!bitOperands(args[0], args[1], "bor", &a, &b)) return false;
    *out = INT_VAL(a | b);
    return true;
}

static bool nPrimBxor(int argc, Value *args, Value *out) {
    (void)argc;
    int64_t a, b;
    if (!bitOperands(args[0], args[1], "bxor", &a, &b)) return false;
    *out = INT_VAL(a ^ b);
    return true;
}

static bool nPrimBnot(int argc, Value *args, Value *out) {
    (void)argc;
    int64_t a;
    if (!jaiArgInt(args[0], 1, "bnot", &a)) return false;
    *out = INT_VAL(~a);
    return true;
}

static bool nPrimShl(int argc, Value *args, Value *out) {
    (void)argc;
    int64_t a, n;
    if (!bitOperands(args[0], args[1], "shl", &a, &n)) return false;
    if (n < 0) return jaiThrow(vm.cValueError, "negative shift count");
    if (n >= 64) { *out = INT_VAL(0); return true; }
    *out = INT_VAL((int64_t)((uint64_t)a << (unsigned)n));
    return true;
}

static bool nPrimShr(int argc, Value *args, Value *out) {
    (void)argc;
    int64_t a, n;
    if (!bitOperands(args[0], args[1], "shr", &a, &n)) return false;
    if (n < 0) return jaiThrow(vm.cValueError, "negative shift count");
    if (n >= 64) { *out = INT_VAL(a < 0 ? -1 : 0); return true; }
    if (a >= 0) {
        *out = INT_VAL((int64_t)((uint64_t)a >> (unsigned)n));
    } else {
        uint64_t magnitude = ~(uint64_t)a;            /* -a - 1, no overflow */
        *out = INT_VAL(-(int64_t)(magnitude >> (unsigned)n) - 1);
    }
    return true;
}

static bool nPrimEq(int argc, Value *args, Value *out) {
    (void)argc;
    bool equal = jaiValuesEqual(args[0], args[1]);
    if (vm.hasException) return false;
    *out = BOOL_VAL(equal);
    return true;
}

static bool nPrimNe(int argc, Value *args, Value *out) {
    (void)argc;
    bool equal = jaiValuesEqual(args[0], args[1]);
    if (vm.hasException) return false;
    *out = BOOL_VAL(!equal);
    return true;
}

static bool primOrder(Value a, Value b, const char *name, int wantLow, int wantHigh,
                      Value *out) {
    int order;
    if (!jaiValueCompare(a, b, &order)) {
        if (vm.hasException) return false;
        if (IS_NUMBER(a) && IS_NUMBER(b)) { *out = BOOL_VAL(false); return true; }
        return jaiThrow(vm.cTypeError, "%s(): cannot compare %s with %s", name,
                        jaiTypeNameStatic(a), jaiTypeNameStatic(b));
    }
    *out = BOOL_VAL(order == wantLow || order == wantHigh);
    return true;
}

static bool nPrimLt(int argc, Value *args, Value *out) {
    (void)argc;
    return primOrder(args[0], args[1], "lt", -1, -1, out);
}

static bool nPrimLe(int argc, Value *args, Value *out) {
    (void)argc;
    return primOrder(args[0], args[1], "le", -1, 0, out);
}

static bool nPrimGt(int argc, Value *args, Value *out) {
    (void)argc;
    return primOrder(args[0], args[1], "gt", 1, 1, out);
}

static bool nPrimGe(int argc, Value *args, Value *out) {
    (void)argc;
    return primOrder(args[0], args[1], "ge", 0, 1, out);
}

static bool nPrimTypeOf(int argc, Value *args, Value *out) {
    (void)argc;
    *out = OBJ_VAL(jaiTypeName(args[0]));
    return true;
}

static bool nPrimCast(int argc, Value *args, Value *out) {
    (void)argc;
    bool matched = false;
    if (!jaiBuiltinMatchesType(args[0], args[1], &matched)) return false;
    if (!matched) {
        ObjString *want = jaiValueToStr(args[1]);
        if (want == NULL) return false;
        jaiGCPushRoot(OBJ_VAL(want));
        (void)jaiThrow(vm.cTypeError, "cannot cast %s to %s",
                       jaiTypeNameStatic(args[0]), want->chars);
        jaiGCPopRoot();
        return false;
    }
    *out = args[0];
    return true;
}

static bool nPrimObjNew(int argc, Value *args, Value *out) {
    (void)argc;
    if (!IS_CLASS(args[0])) return jaiBuiltinArgTypeError(1, "obj_new", "class", args[0]);
    *out = OBJ_VAL(jaiInstanceNew(AS_CLASS(args[0])));
    return true;
}

static bool nPrimGetField(int argc, Value *args, Value *out) {
    (void)argc;
    ObjString *name;
    if (!jaiArgString(args[1], 2, "get_field", &name)) return false;
    return jaiGetProperty(args[0], jaiStringCanonical(name), out);
}

static bool nPrimSetField(int argc, Value *args, Value *out) {
    (void)argc;
    ObjString *name;
    if (!jaiArgString(args[1], 2, "set_field", &name)) return false;
    if (!jaiSetProperty(args[0], jaiStringCanonical(name), args[2])) return false;
    *out = NULL_VAL;
    return true;
}

static bool nPrimGetMethod(int argc, Value *args, Value *out) {
    (void)argc;
    ObjString *name;
    if (!jaiArgString(args[1], 2, "get_method", &name)) return false;
    name = jaiStringCanonical(name);
    Value member;
    if (!jaiGetProperty(args[0], name, &member)) return false;
    if (!jaiBuiltinIsCallable(member))
        return jaiThrow(vm.cAttributeError, "'%s' has no method '%s'",
                        jaiTypeNameStatic(args[0]), name->chars);
    *out = member;
    return true;
}

static bool nPrimRaise(int argc, Value *args, Value *out) {
    (void)argc;
    (void)out;
    return jaiThrowValue(args[0]);
}

static bool nPrimCurrentException(int argc, Value *args, Value *out) {
    (void)argc;
    (void)args;
    *out = vm.hasException ? vm.pendingException : NULL_VAL;
    return true;
}

static bool nPrimTraceback(int argc, Value *args, Value *out) {
    (void)argc;
    (void)args;
    int count = 0;
    JaiFrameInfo *frames = jaiBuildTraceback(&count);
    if (frames == NULL) count = 0;

    ObjList *lines = jaiListNew(count);
    jaiGCPushRoot(OBJ_VAL(lines));
    for (int i = 0; i < count; i++) {
        JaiBuf text;
        jaiBufInit(&text);
        jaiBufPrintf(&text, "%s",
                     frames[i].functionName != NULL ? frames[i].functionName : "<anonymous>");
        if (frames[i].modulePath != NULL) {
            int line = 0, col = 0;
            if (jaiSpanValid(frames[i].span))
                jaiSourceLineCol(frames[i].span.file, frames[i].span.start, &line, &col);
            jaiBufPrintf(&text, " (%s:%d:%d)", frames[i].modulePath, line, col);
        }
        ObjString *entry = jaiStringNew(text.data != NULL ? (const char *)text.data : "",
                                        text.count);
        jaiBufFree(&text);
        if (entry == NULL) break;
        jaiGCPushRoot(OBJ_VAL(entry));
        jaiListPush(lines, OBJ_VAL(entry));
        jaiGCPopRoot();
    }
    jaiGCPopRoot();

    if (frames != NULL) JAI_FREE_ARRAY(JaiFrameInfo, frames, count);
    *out = OBJ_VAL(lines);
    return true;
}

static bool nPrimTraceActive(int argc, Value *args, Value *out) {
    (void)argc;
    (void)args;
    *out = BOOL_VAL(jaiTraceIsActive());
    return true;
}

static bool nPrimTraceReplay(int argc, Value *args, Value *out) {
    (void)argc;
    (void)args;
    *out = BOOL_VAL(jaiTraceReplay());
    return true;
}

static bool nPrimTraceRunId(int argc, Value *args, Value *out) {
    (void)argc;
    (void)args;
    *out = INT_VAL((int64_t)jaiTraceRunId());
    return true;
}

static bool nPrimTraceRecordOp(int argc, Value *args, Value *out) {
    (void)argc;
    ObjString *name;
    if (!jaiArgString(args[0], 1, "trace_record_op", &name)) return false;

    int shape[4];
    int shape_len = 0;
    if (argc >= 2 && IS_LIST(args[1])) {
        ObjList *list = AS_LIST(args[1]);
        shape_len = (int)list->count;
        if (shape_len > 4) shape_len = 4;
        for (int i = 0; i < shape_len; i++) {
            if (!IS_INT(jaiListGet(list, i))) {
                return jaiThrow(vm.cTypeError,
                                "trace_record_op(): shape entries must be int");
            }
            shape[i] = (int)AS_INT(jaiListGet(list, i));
        }
    }
    *out = BOOL_VAL(jaiTraceRecordOp(name->chars, shape, shape_len));
    return true;
}

static bool nPrimFunctionHasFlag(int argc, Value *args, Value *out) {
    (void)argc;
    ObjFunction *fn = NULL;
    Value target = args[0];
    while (IS_BOUND(target)) target = AS_BOUND(target)->method;
    if (IS_CLOSURE(target)) fn = AS_CLOSURE(target)->fn;
    else if (IS_FUNCTION(target)) fn = AS_FUNCTION(target);
    else {
        return jaiThrow(vm.cTypeError,
                        "function_has_flag(): expected a function");
    }

    int64_t bit;
    if (!jaiArgInt(args[1], 2, "function_has_flag", &bit)) return false;
    *out = BOOL_VAL((fn->flags & (uint32_t)bit) != 0);
    return true;
}

static bool nPrimFunctionFlags(int argc, Value *args, Value *out) {
    (void)argc;
    ObjFunction *fn = NULL;
    Value target = args[0];
    while (IS_BOUND(target)) target = AS_BOUND(target)->method;
    if (IS_CLOSURE(target)) fn = AS_CLOSURE(target)->fn;
    else if (IS_FUNCTION(target)) fn = AS_FUNCTION(target);
    else {
        return jaiThrow(vm.cTypeError,
                        "function_flags(): expected a function");
    }
    *out = INT_VAL((int64_t)fn->flags);
    return true;
}

void jaiRegisterOperatorPrimitives(void) {
    jaiDefineNative("__prim__.add",      nPrimAdd,      2, 2);
    jaiDefineNative("__prim__.sub",      nPrimSub,      2, 2);
    jaiDefineNative("__prim__.mul",      nPrimMul,      2, 2);
    jaiDefineNative("__prim__.div",      nPrimDiv,      2, 2);
    jaiDefineNative("__prim__.floordiv", nPrimFloorDiv, 2, 2);
    jaiDefineNative("__prim__.mod",      nPrimMod,      2, 2);
    jaiDefineNative("__prim__.pow",      nPrimPow,      2, 2);
    jaiDefineNative("__prim__.neg",      nPrimNeg,      1, 1);

    jaiDefineNative("__prim__.band", nPrimBand, 2, 2);
    jaiDefineNative("__prim__.bor",  nPrimBor,  2, 2);
    jaiDefineNative("__prim__.bxor", nPrimBxor, 2, 2);
    jaiDefineNative("__prim__.bnot", nPrimBnot, 1, 1);
    jaiDefineNative("__prim__.shl",  nPrimShl,  2, 2);
    jaiDefineNative("__prim__.shr",  nPrimShr,  2, 2);

    jaiDefineNative("__prim__.eq", nPrimEq, 2, 2);
    jaiDefineNative("__prim__.ne", nPrimNe, 2, 2);
    jaiDefineNative("__prim__.lt", nPrimLt, 2, 2);
    jaiDefineNative("__prim__.le", nPrimLe, 2, 2);
    jaiDefineNative("__prim__.gt", nPrimGt, 2, 2);
    jaiDefineNative("__prim__.ge", nPrimGe, 2, 2);

    jaiDefineNative("__prim__.type_of",     nPrimTypeOf,    1, 1);
    jaiDefineNative("__prim__.cast",        nPrimCast,      2, 2);

    jaiDefineNative("__prim__.obj_new",        nPrimObjNew,    1, 1);
    jaiDefineNative("__prim__.obj_get_field",  nPrimGetField,  2, 2);
    jaiDefineNative("__prim__.obj_set_field",  nPrimSetField,  3, 3);
    jaiDefineNative("__prim__.obj_get_method", nPrimGetMethod, 2, 2);

    jaiDefineNative("__prim__.raise",             nPrimRaise,            1, 1);
    jaiDefineNative("__prim__.current_exception", nPrimCurrentException, 0, 0);
    jaiDefineNative("__prim__.traceback",         nPrimTraceback,        0, 0);
    jaiDefineNative("__prim__.trace_active",      nPrimTraceActive,      0, 0);
    jaiDefineNative("__prim__.trace_replay",      nPrimTraceReplay,      0, 0);
    jaiDefineNative("__prim__.trace_run_id",      nPrimTraceRunId,       0, 0);
    jaiDefineNative("__prim__.trace_record_op",   nPrimTraceRecordOp,    1, 2);
    jaiDefineNative("__prim__.function_has_flag", nPrimFunctionHasFlag,    2, 2);
    jaiDefineNative("__prim__.function_flags",    nPrimFunctionFlags,     1, 1);
}
