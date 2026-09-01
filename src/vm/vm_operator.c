/* vm_operator.c — the operator helpers behind the dispatch arms: arithmetic
 * and its overflow rules, bitwise, ordered comparison, containment, indexing
 * and slicing.
 */
#include <inttypes.h>
#include <math.h>

#include "vm/vm_internal.h"

/* ------------------------------------------------------------------ */
/* Operators                                                            */
/* ------------------------------------------------------------------ */

static const char *opSymbol(OpCode op) {
    switch (op) {
    case OP_ADD: case OP_ADD_WRAP:  return op == OP_ADD ? "+" : "+%";
    case OP_SUB: case OP_SUB_WRAP:  return op == OP_SUB ? "-" : "-%";
    case OP_MUL: case OP_MUL_WRAP:  return op == OP_MUL ? "*" : "*%";
    case OP_DIV:                    return "/";
    case OP_FLOORDIV:               return "//";
    case OP_MOD:                    return "%";
    case OP_POW:                    return "**";
    case OP_BAND:                   return "&";
    case OP_BOR:                    return "|";
    case OP_BXOR:                   return "^";
    case OP_SHL:                    return "<<";
    case OP_SHR:                    return ">>";
    case OP_LT:                     return "<";
    case OP_LE:                     return "<=";
    case OP_GT:                     return ">";
    case OP_GE:                     return ">=";
    case OP_CONCAT:                 return "+";
    default:                        return "?";
    }
}

static bool badOperands(OpCode op, Value a, Value b) {
    return jaiThrow(vm.cTypeError,
                    "unsupported operand types for '%s': '%s' and '%s'",
                    opSymbol(op), jaiTypeNameStatic(a), jaiTypeNameStatic(b));
}

static bool intOverflow(OpCode op) {
    /* Only `+ - *` have a wrapping spelling (spec §2.5). Suggesting `**%` or
     * `//%` sends the reader looking for an operator the grammar does not have. */
    switch (op) {
    case OP_ADD: case OP_SUB: case OP_MUL:
        return jaiThrow(vm.cOverflowError,
                        "integer overflow in '%s'; use '%s%%' to wrap",
                        opSymbol(op), opSymbol(op));
    default:
        return jaiThrow(vm.cOverflowError, "integer overflow in '%s'", opSymbol(op));
    }
}

/* Python's floor semantics: the result takes the sign of the divisor, so
 * -7 // 2 is -4 and -7 % 3 is 2. C truncates toward zero, which is why both
 * results are corrected here rather than used directly. */
static bool intFloorDivMod(int64_t a, int64_t b, int64_t *outDiv,
                           int64_t *outMod) {
    if (b == 0) {
        return jaiThrow(vm.cDivisionByZeroError, "integer division by zero");
    }
    if (a == INT64_MIN && b == -1) {
        *outDiv = INT64_MIN;   /* only reached by MOD, which wants 0 */
        *outMod = 0;
        return true;
    }
    int64_t q = a / b;
    int64_t r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) {
        q -= 1;
        r += b;
    }
    *outDiv = q;
    *outMod = r;
    return true;
}

static bool floatFloorDivMod(double a, double b, double *outDiv, double *outMod) {
    if (b == 0.0) {
        return jaiThrow(vm.cDivisionByZeroError, "float division by zero");
    }
    double r = fmod(a, b);
    if (r != 0.0 && ((r < 0.0) != (b < 0.0))) r += b;
    *outDiv = floor(a / b);
    *outMod = r;
    return true;
}

static bool intPow(int64_t base, int64_t exp, int64_t *out) {
    if (exp < 0) {
        return jaiThrow(vm.cValueError,
                        "negative exponent on int '**'; convert to float first");
    }
    int64_t result = 1;
    int64_t acc = base;
    while (exp > 0) {
        if (exp & 1) {
            if (__builtin_mul_overflow(result, acc, &result)) {
                return intOverflow(OP_POW);
            }
        }
        exp >>= 1;
        if (exp == 0) break;
        if (__builtin_mul_overflow(acc, acc, &acc)) return intOverflow(OP_POW);
    }
    *out = result;
    return true;
}

/* Repeat a sequence: "ab" * 3 and [0] * 4. */
static bool repeatString(ObjString *s, int64_t times, Value *out) {
    if (times <= 0) {
        ObjString *empty = jaiStringIntern("", 0);
        if (empty == NULL) return false;
        *out = OBJ_VAL(empty);
        return true;
    }
    if ((uint64_t)times * (uint64_t)s->length > (uint64_t)UINT32_MAX) {
        return jaiThrow(vm.cOverflowError,
                        "repeated string exceeds the maximum length");
    }
    size_t total = (size_t)times * (size_t)s->length;
    char *buf = JAI_ALLOC(char, total + 1);
    for (int64_t i = 0; i < times; i++) {
        memcpy(buf + (size_t)i * s->length, s->chars, s->length);
    }
    buf[total] = '\0';
    ObjString *result = jaiStringTake(buf, total);
    if (result == NULL) return false;
    *out = OBJ_VAL(result);
    return true;
}

static bool repeatList(ObjList *list, int64_t times, Value *out) {
    if (times < 0) times = 0;
    /* Divide rather than multiply: the product itself can overflow. */
    if (list->count > 0 && times > (int64_t)INT32_MAX / list->count) {
        return jaiThrow(vm.cRuntimeError, "list cannot grow beyond %d items",
                        INT32_MAX);
    }
    jaiGCPushRoot(OBJ_VAL(list));
    ObjList *result = jaiListNew((int)(times * list->count));
    for (int64_t i = 0; i < times; i++) {
        for (int j = 0; j < list->count; j++) {
            jaiListPut(result, result->count++, jaiListGet(list, j));
        }
    }
    jaiGCPopRoot();
    *out = OBJ_VAL(result);
    return true;
}

/* The dunder method for a binary operator on an instance receiver, or
 * NULL_VAL when the class does not overload it. */
static Value operatorDunder(ObjClass *klass, OpCode op) {
    if (klass == NULL) return NULL_VAL;
    switch (op) {
    case OP_ADD: case OP_ADD_WRAP: case OP_CONCAT: return klass->dunderAdd;
    case OP_SUB: case OP_SUB_WRAP:                 return klass->dunderSub;
    case OP_MUL: case OP_MUL_WRAP:                 return klass->dunderMul;
    case OP_DIV:                                   return klass->dunderDiv;
    case OP_MOD:                                   return klass->dunderMod;
    case OP_POW:                                   return klass->dunderPow;
    default:                                       return NULL_VAL;
    }
}

/* Instance fallback. Only `__add__`-style forward dispatch exists — the spec's
 * dunder list has no reflected forms — so when only the right operand is an
 * instance its own dunder is tried with the operands as written. */
static bool binaryDunder(OpCode op, Value a, Value b, Value *out, bool *handled) {
    *handled = false;
    Value method = NULL_VAL;
    Value receiver = a, arg = b;

    if (IS_INSTANCE(a)) {
        method = operatorDunder(AS_INSTANCE(a)->klass, op);
    }
    if (IS_NULL(method) && IS_INSTANCE(b)) {
        method = operatorDunder(AS_INSTANCE(b)->klass, op);
        receiver = b;
        arg = a;
    }
    if (IS_NULL(method)) return true;

    *handled = true;
    Value bound = OBJ_VAL(jaiBoundNew(receiver, method));
    return jaiCallValue(bound, 1, &arg, out);
}

/* The float half of `arithmetic`, split out because two operand shapes reach
 * it: two floats, and an int widened against a float. `a` and `b` are carried
 * only so that an operator with no float form names its real operands. */
static bool floatArithmetic(OpCode op, double x, double y, Value a, Value b,
                            Value *out) {
    switch (op) {
    case OP_ADD: *out = FLOAT_VAL(x + y); return true;
    case OP_SUB: *out = FLOAT_VAL(x - y); return true;
    case OP_MUL: *out = FLOAT_VAL(x * y); return true;
    case OP_DIV:
        if (y == 0.0) {
            return jaiThrow(vm.cDivisionByZeroError, "float division by zero");
        }
        *out = FLOAT_VAL(x / y);
        return true;
    case OP_FLOORDIV: {
        double q, m;
        if (!floatFloorDivMod(x, y, &q, &m)) return false;
        *out = FLOAT_VAL(q);
        return true;
    }
    case OP_MOD: {
        double q, m;
        if (!floatFloorDivMod(x, y, &q, &m)) return false;
        *out = FLOAT_VAL(m);
        return true;
    }
    case OP_POW: *out = FLOAT_VAL(pow(x, y)); return true;
    default:     return badOperands(op, a, b);
    }
}

bool arithmetic(OpCode op, Value a, Value b, Value *out) {
    if (IS_INT(a) && IS_INT(b)) {
        int64_t x = AS_INT(a), y = AS_INT(b), r = 0;
        switch (op) {
        case OP_ADD:
            if (__builtin_add_overflow(x, y, &r)) return intOverflow(op);
            break;
        case OP_SUB:
            if (__builtin_sub_overflow(x, y, &r)) return intOverflow(op);
            break;
        case OP_MUL:
            if (__builtin_mul_overflow(x, y, &r)) return intOverflow(op);
            break;
        case OP_ADD_WRAP: r = (int64_t)((uint64_t)x + (uint64_t)y); break;
        case OP_SUB_WRAP: r = (int64_t)((uint64_t)x - (uint64_t)y); break;
        case OP_MUL_WRAP: r = (int64_t)((uint64_t)x * (uint64_t)y); break;
        case OP_DIV:
            /* int / int is float (spec §3.3); use // for an integer result. */
            if (y == 0) {
                return jaiThrow(vm.cDivisionByZeroError, "division by zero");
            }
            *out = FLOAT_VAL((double)x / (double)y);
            return true;
        case OP_FLOORDIV: {
            int64_t q, m;
            if (!intFloorDivMod(x, y, &q, &m)) return false;
            if (x == INT64_MIN && y == -1) return intOverflow(op);
            r = q;
            break;
        }
        case OP_MOD: {
            int64_t q, m;
            if (!intFloorDivMod(x, y, &q, &m)) return false;
            r = m;
            break;
        }
        case OP_POW:
            if (!intPow(x, y, &r)) return false;
            break;
        default:
            return badOperands(op, a, b);
        }
        *out = INT_VAL(r);
        return true;
    }

    if (IS_FLOAT(a) && IS_FLOAT(b)) {
        return floatArithmetic(op, AS_FLOAT(a), AS_FLOAT(b), a, b, out);
    }

    /* One int and one float: the int widens and the result is a float (spec
     * §2.5). Where both types are known the checker has already inserted the
     * conversion, so what reaches here came through `any`.
     *
     * The wrapping forms are the exception. They are 64-bit integer
     * operations, `float` has nothing to wrap, and quietly widening `+%` into
     * ordinary float addition would answer a question about overflow with a
     * number that cannot overflow. */
    if (IS_NUMBER(a) && IS_NUMBER(b)) {
        if (op == OP_ADD_WRAP || op == OP_SUB_WRAP || op == OP_MUL_WRAP) {
            return jaiThrow(vm.cTypeError,
                            "'%s' wraps a 64-bit integer and does not apply to "
                            "'float'", opSymbol(op));
        }
        double x = IS_INT(a) ? (double)AS_INT(a) : AS_FLOAT(a);
        double y = IS_INT(b) ? (double)AS_INT(b) : AS_FLOAT(b);
        return floatArithmetic(op, x, y, a, b, out);
    }

    if (op == OP_ADD || op == OP_CONCAT) {
        if (IS_STRING(a) && IS_STRING(b)) {
            ObjString *s = jaiStringConcat(AS_STRING(a), AS_STRING(b));
            if (s == NULL) return false;
            *out = OBJ_VAL(s);
            return true;
        }
        if (IS_LIST(a) && IS_LIST(b)) {
            ObjList *l = jaiListConcat(AS_LIST(a), AS_LIST(b));
            if (l == NULL) return false;
            *out = OBJ_VAL(l);
            return true;
        }
        if (IS_TUPLE(a) && IS_TUPLE(b)) {
            ObjTuple *ta = AS_TUPLE(a), *tb = AS_TUPLE(b);
            jaiGCPushRoot(a);
            jaiGCPushRoot(b);
            ObjTuple *result = jaiTupleNew(NULL, (int)(ta->count + tb->count));
            for (uint32_t i = 0; i < ta->count; i++) result->items[i] = ta->items[i];
            for (uint32_t i = 0; i < tb->count; i++) {
                result->items[ta->count + i] = tb->items[i];
            }
            jaiGCPopRoots(2);
            *out = OBJ_VAL(result);
            return true;
        }
    }

    if (op == OP_MUL) {
        if (IS_STRING(a) && IS_INT(b)) return repeatString(AS_STRING(a), AS_INT(b), out);
        if (IS_INT(a) && IS_STRING(b)) return repeatString(AS_STRING(b), AS_INT(a), out);
        if (IS_LIST(a) && IS_INT(b))   return repeatList(AS_LIST(a), AS_INT(b), out);
        if (IS_INT(a) && IS_LIST(b))   return repeatList(AS_LIST(b), AS_INT(a), out);
    }

    bool handled = false;
    if (!binaryDunder(op, a, b, out, &handled)) return false;
    if (handled) return true;

    return badOperands(op, a, b);
}

bool bitwise(OpCode op, Value a, Value b, Value *out) {
    if (!IS_INT(a) || !IS_INT(b)) {
        bool handled = false;
        if (!binaryDunder(op, a, b, out, &handled)) return false;
        if (handled) return true;
        return badOperands(op, a, b);
    }
    int64_t x = AS_INT(a), y = AS_INT(b);

    switch (op) {
    case OP_BAND: *out = INT_VAL(x & y); return true;
    case OP_BOR:  *out = INT_VAL(x | y); return true;
    case OP_BXOR: *out = INT_VAL(x ^ y); return true;
    case OP_SHL:
        if (y < 0) return jaiThrow(vm.cValueError, "negative shift count");
        /* Bits shifted off the top are discarded, not an overflow: spec §2.5
         * scopes OverflowError to `+ - *`, and `__prim__.shl` has always
         * wrapped. Shifting through the unsigned domain also avoids the C UB
         * of a signed left shift that crosses the sign bit. */
        *out = INT_VAL(y >= 64 ? 0 : (int64_t)((uint64_t)x << (uint64_t)y));
        return true;
    case OP_SHR:
        if (y < 0) return jaiThrow(vm.cValueError, "negative shift count");
        /* Arithmetic shift, saturating: >> 64 or more leaves the sign. */
        *out = INT_VAL(y >= 64 ? (x < 0 ? -1 : 0) : (x >> y));
        return true;
    default:
        return badOperands(op, a, b);
    }
}

bool compareOp(OpCode op, Value a, Value b, Value *out) {
    int cmp = 0;
    if (!jaiValueCompare(a, b, &cmp)) {
        if (vm.hasException) return false;
        return jaiThrow(vm.cTypeError,
                        "'%s' is not supported between '%s' and '%s'",
                        opSymbol(op), jaiTypeNameStatic(a), jaiTypeNameStatic(b));
    }
    switch (op) {
    case OP_LT: *out = BOOL_VAL(cmp < 0);  return true;
    case OP_LE: *out = BOOL_VAL(cmp <= 0); return true;
    case OP_GT: *out = BOOL_VAL(cmp > 0);  return true;
    case OP_GE: *out = BOOL_VAL(cmp >= 0); return true;
    default:    return badOperands(op, a, b);
    }
}

/* `element in container`. */
bool jaiContainsOp(Value container, Value element, bool *out) {
    if (IS_STRING(container)) {
        if (!IS_STRING(element)) {
            return jaiThrow(vm.cTypeError,
                            "'in' on a str requires a str, not '%s'",
                            jaiTypeNameStatic(element));
        }
        ObjString *hay = AS_STRING(container), *needle = AS_STRING(element);
        if (needle->length == 0) { *out = true; return true; }
        if (needle->length > hay->length) { *out = false; return true; }
        for (uint32_t i = 0; i + needle->length <= hay->length; i++) {
            if (memcmp(hay->chars + i, needle->chars, needle->length) == 0) {
                *out = true;
                return true;
            }
        }
        *out = false;
        return true;
    }
    if (IS_LIST(container)) {
        ObjList *list = AS_LIST(container);
        for (int i = 0; i < list->count; i++) {
            if (jaiValuesEqual(jaiListGet(list, i), element)) { *out = true; return true; }
            if (vm.hasException) return false;
        }
        *out = false;
        return true;
    }
    if (IS_TUPLE(container)) {
        ObjTuple *t = AS_TUPLE(container);
        for (uint32_t i = 0; i < t->count; i++) {
            if (jaiValuesEqual(t->items[i], element)) { *out = true; return true; }
            if (vm.hasException) return false;
        }
        *out = false;
        return true;
    }
    if (IS_DICT(container)) {
        Value ignored;
        *out = jaiDictGet(AS_DICT(container), element, &ignored);
        return !vm.hasException;
    }
    if (IS_SET(container)) {
        *out = jaiSetHas(AS_SET(container), element);
        return !vm.hasException;
    }
    if (IS_RANGE(container)) {
        if (!IS_INT(element)) { *out = false; return true; }
        ObjRange *r = AS_RANGE(container);
        int64_t v = AS_INT(element), n = jaiRangeLength(r);
        if (n <= 0) { *out = false; return true; }
        int64_t delta = v - r->start;
        if (r->step == 0 || delta % r->step != 0) { *out = false; return true; }
        int64_t index = delta / r->step;
        *out = index >= 0 && index < n;
        return true;
    }
    if (IS_INSTANCE(container)) {
        ObjClass *klass = AS_INSTANCE(container)->klass;
        if (klass != NULL && !IS_NULL(klass->dunderContains)) {
            Value result, arg = element;
            Value bound = OBJ_VAL(jaiBoundNew(container, klass->dunderContains));
            if (!jaiCallValue(bound, 1, &arg, &result)) return false;
            if (!IS_BOOL(result)) {
                return jaiThrow(vm.cTypeError,
                                "__contains__ must return bool, not %s",
                                jaiTypeNameStatic(result));
            }
            *out = AS_BOOL(result);
            return true;
        }
    }
    return jaiThrow(vm.cTypeError, "'in' is not supported for '%s'",
                    jaiTypeNameStatic(container));
}

bool unaryNegate(Value v, Value *out) {
    if (IS_INT(v)) {
        if (AS_INT(v) == INT64_MIN) {
            return jaiThrow(vm.cOverflowError, "integer overflow in unary '-'");
        }
        *out = INT_VAL(-AS_INT(v));
        return true;
    }
    if (IS_FLOAT(v)) {
        *out = FLOAT_VAL(-AS_FLOAT(v));
        return true;
    }
    if (IS_INSTANCE(v)) {
        ObjClass *klass = AS_INSTANCE(v)->klass;
        if (klass != NULL && !IS_NULL(klass->dunderNeg)) {
            return jaiCallValue(OBJ_VAL(jaiBoundNew(v, klass->dunderNeg)), 0,
                                NULL, out);
        }
    }
    return jaiThrow(vm.cTypeError, "unary '-' is not supported for '%s'",
                    jaiTypeNameStatic(v));
}

/* ------------------------------------------------------------------ */
/* Indexing and slicing                                                 */
/* ------------------------------------------------------------------ */

bool jaiIndexGet(Value container, Value index, Value *out) {
    if (IS_LIST(container)) {
        if (!IS_INT(index)) {
            return jaiThrow(vm.cTypeError, "list indices must be int, not '%s'",
                            jaiTypeNameStatic(index));
        }
        ObjList *list = AS_LIST(container);
        int at;
        if (!jaiNormalizeIndex(AS_INT(index), list->count, &at)) {
            return jaiThrow(vm.cIndexError,
                            "list index %" PRId64 " out of range for length %d",
                            AS_INT(index), list->count);
        }
        *out = jaiListGet(list, at);
        return true;
    }
    if (IS_TUPLE(container)) {
        if (!IS_INT(index)) {
            return jaiThrow(vm.cTypeError, "tuple indices must be int, not '%s'",
                            jaiTypeNameStatic(index));
        }
        ObjTuple *t = AS_TUPLE(container);
        int at;
        if (!jaiNormalizeIndex(AS_INT(index), (int)t->count, &at)) {
            return jaiThrow(vm.cIndexError,
                            "tuple index %" PRId64 " out of range for length %u",
                            AS_INT(index), (unsigned)t->count);
        }
        *out = t->items[at];
        return true;
    }
    if (IS_STRING(container)) {
        if (!IS_INT(index)) {
            return jaiThrow(vm.cTypeError, "str indices must be int, not '%s'",
                            jaiTypeNameStatic(index));
        }
        ObjString *s = AS_STRING(container);
        int at;
        if (!jaiNormalizeIndex(AS_INT(index), (int)jaiStringScalarCount(s), &at)) {
            return jaiThrow(vm.cIndexError,
                            "str index %" PRId64 " out of range for length %u",
                            AS_INT(index), (unsigned)jaiStringScalarCount(s));
        }
        ObjString *scalar = jaiStringSlice(s, at, at + 1, 1);
        if (scalar == NULL) return false;
        *out = OBJ_VAL(scalar);
        return true;
    }
    if (IS_BYTES(container)) {
        if (!IS_INT(index)) {
            return jaiThrow(vm.cTypeError, "bytes indices must be int, not '%s'",
                            jaiTypeNameStatic(index));
        }
        ObjBytes *b = AS_BYTES(container);
        int at;
        if (!jaiNormalizeIndex(AS_INT(index), (int)b->length, &at)) {
            return jaiThrow(vm.cIndexError,
                            "bytes index %" PRId64 " out of range for length %u",
                            AS_INT(index), (unsigned)b->length);
        }
        *out = INT_VAL(b->data[at]);
        return true;
    }
    if (IS_DICT(container)) {
        if (jaiDictGet(AS_DICT(container), index, out)) return true;
        if (vm.hasException) return false;
        char buf[96];
        describeValue(index, buf, sizeof buf);
        return jaiThrow(vm.cKeyError, "key %s not found", buf);
    }
    if (IS_RANGE(container)) {
        if (!IS_INT(index)) {
            return jaiThrow(vm.cTypeError, "range indices must be int, not '%s'",
                            jaiTypeNameStatic(index));
        }
        ObjRange *r = AS_RANGE(container);
        int64_t n = jaiRangeLength(r);
        int at;
        if (n > INT32_MAX || !jaiNormalizeIndex(AS_INT(index), (int)n, &at)) {
            return jaiThrow(vm.cIndexError, "range index %" PRId64 " out of range",
                            AS_INT(index));
        }
        *out = INT_VAL(r->start + (int64_t)at * r->step);
        return true;
    }
    if (IS_INSTANCE(container)) {
        ObjClass *klass = AS_INSTANCE(container)->klass;
        if (klass != NULL && !IS_NULL(klass->dunderGetItem)) {
            Value arg = index;
            return jaiCallValue(OBJ_VAL(jaiBoundNew(container, klass->dunderGetItem)),
                                1, &arg, out);
        }
    }
    return jaiThrow(vm.cTypeError, "'%s' value is not indexable",
                    jaiTypeNameStatic(container));
}

bool indexSet(Value container, Value index, Value value) {
    if (IS_LIST(container)) {
        if (!IS_INT(index)) {
            return jaiThrow(vm.cTypeError, "list indices must be int, not '%s'",
                            jaiTypeNameStatic(index));
        }
        ObjList *list = AS_LIST(container);
        int at;
        if (!jaiNormalizeIndex(AS_INT(index), list->count, &at)) {
            return jaiThrow(vm.cIndexError,
                            "list index %" PRId64 " out of range for length %d",
                            AS_INT(index), list->count);
        }
        if (!jaiCheckKind(list->elemKind, value, "an element")) return false;
        jaiListPut(list, at, value);
        jaiListTouch(list);      /* the count is unchanged; only the version tells */
        return true;
    }
    if (IS_DICT(container)) {
        (void)jaiDictSet(AS_DICT(container), index, value);
        return !vm.hasException;
    }
    if (IS_SET(container)) {
        return jaiThrow(vm.cTypeError, "set does not support index assignment");
    }
    if (IS_INSTANCE(container)) {
        ObjClass *klass = AS_INSTANCE(container)->klass;
        if (klass != NULL && !IS_NULL(klass->dunderSetItem)) {
            Value args[2] = {index, value};
            Value ignored;
            return jaiCallValue(OBJ_VAL(jaiBoundNew(container, klass->dunderSetItem)),
                                2, args, &ignored);
        }
    }
    return jaiThrow(vm.cTypeError, "'%s' value does not support item assignment",
                    jaiTypeNameStatic(container));
}

/* Slice bounds from the optional start/stop/step pushed under the flags. */
bool sliceBounds(Value startV, Value stopV, Value stepV, bool hasStart,
                        bool hasStop, bool hasStep, int64_t length,
                        int64_t *start, int64_t *stop, int64_t *step) {
    *step = 1;
    if (hasStep) {
        if (!IS_INT(stepV)) {
            return jaiThrow(vm.cTypeError, "slice step must be int, not '%s'",
                            jaiTypeNameStatic(stepV));
        }
        *step = AS_INT(stepV);
        if (*step == 0) {
            return jaiThrow(vm.cValueError, "slice step cannot be zero");
        }
    }
    *start = (*step > 0) ? 0 : length - 1;
    *stop = (*step > 0) ? length : -length - 1;

    if (hasStart) {
        if (!IS_INT(startV)) {
            return jaiThrow(vm.cTypeError, "slice bounds must be int, not '%s'",
                            jaiTypeNameStatic(startV));
        }
        *start = AS_INT(startV);
    }
    if (hasStop) {
        if (!IS_INT(stopV)) {
            return jaiThrow(vm.cTypeError, "slice bounds must be int, not '%s'",
                            jaiTypeNameStatic(stopV));
        }
        *stop = AS_INT(stopV);
    }
    return true;
}

/* The whole of OP_GET_SLICE, so the compiled tier runs this rather than a copy
 * of it. The caller must have saved VM state; returns false with an exception
 * pending. */
bool jaiSliceGet(Value container, Value startValue, Value stopValue,
                 Value stepValue, bool hasStart, bool hasStop, bool hasStep,
                 Value *out) {
    int64_t length;
    if (IS_LIST(container))        length = AS_LIST(container)->count;
    else if (IS_STRING(container)) length = jaiStringScalarCount(AS_STRING(container));
    else if (IS_TUPLE(container))  length = AS_TUPLE(container)->count;
    else {
        return jaiThrow(vm.cTypeError, "'%s' value does not support slicing",
                        jaiTypeNameStatic(container));
    }

    int64_t start, stop, step;
    if (!sliceBounds(startValue, stopValue, stepValue, hasStart, hasStop,
                     hasStep, length, &start, &stop, &step)) {
        return false;
    }
    if (IS_LIST(container)) {
        ObjList *sliced = jaiListSlice(AS_LIST(container), start, stop, step);
        if (sliced == NULL) return false;
        *out = OBJ_VAL(sliced);
        return true;
    }
    if (IS_STRING(container)) {
        ObjString *sliced = jaiStringSlice(AS_STRING(container), start, stop, step);
        if (sliced == NULL) return false;
        *out = OBJ_VAL(sliced);
        return true;
    }
    {
        ObjTuple *tuple = AS_TUPLE(container);
        /* Reuse the list slicer for index arithmetic, then re-wrap. */
        ObjList *temp = jaiListNew((int)tuple->count);
        jaiGCPushRoot(OBJ_VAL(temp));
        for (uint32_t i = 0; i < tuple->count; i++) {
            jaiListBox(temp)[temp->count++] = tuple->items[i];
        }
        ObjList *sliced = jaiListSlice(temp, start, stop, step);
        jaiGCPopRoot();
        if (sliced == NULL) return false;
        jaiGCPushRoot(OBJ_VAL(sliced));
        ObjTuple *outT = jaiTupleNew(jaiListBox(sliced), sliced->count);
        jaiGCPopRoot();
        *out = OBJ_VAL(outT);
        return true;
    }
}
