/* builtins_numeric.c — the int and float method tables, and the checked
 * integer arithmetic behind them. */

#include <math.h>

#include "runtime/builtins/builtins_math.h"

/* |v| as an unsigned magnitude. Negating through uint64_t is the only way to
 * express |INT64_MIN|, which has no int64_t representation. */
static inline uint64_t magnitudeOf(int64_t v) {
    return v < 0 ? -(uint64_t)v : (uint64_t)v;
}

static inline bool signedFromMagnitude(uint64_t magnitude, bool negative,
                                       int64_t *out) {
    if (negative) {
        const uint64_t limit = (uint64_t)INT64_MAX + 1u;
        if (magnitude > limit) return false;
        *out = magnitude == limit ? INT64_MIN : -(int64_t)magnitude;
        return true;
    }

    if (magnitude > (uint64_t)INT64_MAX) return false;
    *out = (int64_t)magnitude;
    return true;
}

static inline bool mulChecked(int64_t a, int64_t b, int64_t *out) {
#if defined(__clang__) || defined(__GNUC__)
    return !__builtin_mul_overflow(a, b, out);
#else
    if (a == 0 || b == 0) {
        *out = 0;
        return true;
    }
    if (a == -1) {
        if (b == INT64_MIN) return false;
        *out = -b;
        return true;
    }
    if (b == -1) {
        if (a == INT64_MIN) return false;
        *out = -a;
        return true;
    }
    if (a > 0) {
        if (b > 0) {
            if (a > INT64_MAX / b) return false;
        } else {
            if (b < INT64_MIN / a) return false;
        }
    } else {
        if (b > 0) {
            if (a < INT64_MIN / b) return false;
        } else {
            if (b < INT64_MAX / a) return false;
        }
    }
    *out = a * b;
    return true;
#endif
}

static inline bool powChecked(int64_t base, int64_t exponent, int64_t *out) {
    int64_t result = 1;
    int64_t square = base;
    uint64_t remaining = (uint64_t)exponent;

    while (remaining) {
        if ((remaining & 1u) && !mulChecked(result, square, &result))
            return false;

        remaining >>= 1;
        if (remaining && !mulChecked(square, square, &square))
            return false;
    }

    *out = result;
    return true;
}

/* Truncate a float towards zero into an int64, refusing what cannot be held. */
static inline bool floatToInt(double d, const char *fnName, int64_t *out) {
    if (isnan(d))
        return jaiThrow(vm.cValueError,
                        "%s(): cannot convert NaN to int", fnName);

    if (isinf(d))
        return jaiThrow(vm.cOverflowError,
                        "%s(): cannot convert infinity to int", fnName);

    if (d >= kTwoPow63 || d < -kTwoPow63)
        return jaiThrow(vm.cOverflowError,
                        "%s(): %g is out of range for int", fnName, d);

    /* C conversion already truncates toward zero. */
    *out = (int64_t)d;
    return true;
}

/* ------------------------------------------------------------------ */
/* Integer helpers behind the int methods                               */
/* ------------------------------------------------------------------ */

/* All 64 bits are counted, so bit_count(-1) is 64. */
static inline int bitCountU64(uint64_t x) {
#if defined(__clang__) || defined(__GNUC__)
    return __builtin_popcountll((unsigned long long)x);
#else
    int count = 0;
    while (x) {
        x &= x - 1;
        ++count;
    }
    return count;
#endif
}


/* Position of the highest set bit of |n|: 0 for zero, 64 for INT_MIN. */
static inline int bitLengthI64(int64_t n) {
    const uint64_t x = magnitudeOf(n);
    if (x == 0) return 0;

#if defined(__clang__) || defined(__GNUC__)
    return 64 - __builtin_clzll((unsigned long long)x);
#else
    uint64_t v = x;
    int length = 0;
    while (v) {
        ++length;
        v >>= 1;
    }
    return length;
#endif
}


/* Portable fallback helpers for platforms without a native 128-bit integer. */
#if !defined(__SIZEOF_INT128__)
static inline uint64_t addMod(uint64_t a, uint64_t b, uint64_t m) {
    return a >= m - b ? a - (m - b) : a + b;
}
#endif

/* a * b mod m. Clang/GCC use one 128-bit product; other compilers keep every
 * intermediate inside 64 bits. Callers maintain a < m and b < m. */
static inline uint64_t mulMod(uint64_t a, uint64_t b, uint64_t m) {
#if defined(__SIZEOF_INT128__)
    return (uint64_t)(((__uint128_t)a * (__uint128_t)b) % (__uint128_t)m);
#else
    if (m <= 0xFFFFFFFFULL)
        return (a * b) % m;

    uint64_t result = 0;
    while (b) {
        if (b & 1u) result = addMod(result, a, m);
        b >>= 1;
        if (b) a = addMod(a, a, m);
    }
    return result;
#endif
}

/* Modular exponentiation. The result takes the sign of the modulus, exactly as
 * `%` does, so pow_mod agrees with `(base ** exponent) % modulus`. */
static bool powModI64(int64_t base, int64_t exponent, int64_t modulus,
                      int64_t *out) {
    const uint64_t m = magnitudeOf(modulus);
    if (m == 1) {
        *out = 0;
        return true;
    }

    uint64_t reduced = magnitudeOf(base) % m;
    if (base < 0 && reduced)
        reduced = m - reduced;

    uint64_t result = 1;
    uint64_t remaining = (uint64_t)exponent;

    while (remaining) {
        if (remaining & 1u)
            result = mulMod(result, reduced, m);

        remaining >>= 1;
        if (remaining)
            reduced = mulMod(reduced, reduced, m);
    }

    if (modulus < 0 && result)
        return signedFromMagnitude(m - result, true, out);

    return signedFromMagnitude(result, false, out);
}

static inline bool powModGuarded(int64_t base, int64_t exponent,
                                 int64_t modulus, const char *fnName,
                                 int64_t *out) {
    if (modulus == 0)
        return jaiThrow(vm.cDivisionByZeroError,
                        "%s(): modulus must not be zero", fnName);

    if (exponent < 0)
        return jaiThrow(vm.cValueError,
                        "%s() expects a non-negative exponent, got %lld",
                        fnName, (long long)exponent);

    if (!powModI64(base, exponent, modulus, out))
        return jaiThrow(vm.cOverflowError,
                        "%s(): the result does not fit in an int", fnName);

    return true;
}

/* ------------------------------------------------------------------ */
/* int and float methods                                                */
/* ------------------------------------------------------------------ */

/* A bound native gets the receiver in args[0] (argc counts it), so methods
 * read their own args from args[1] on; table arities include the receiver. */

static inline bool intSelf(Value *args, const char *method, int64_t *out) {
    const Value self = args[0];

    if (IS_INT(self)) {
        *out = AS_INT(self);
        return true;
    }

    return jaiThrow(vm.cTypeError,
                    "int.%s() needs an int as its receiver, got %s",
                    method, jaiTypeNameStatic(self));
}

static inline bool floatSelf(Value *args, const char *method, double *out) {
    const Value self = args[0];

    if (IS_FLOAT(self)) {
        *out = AS_FLOAT(self);
        return true;
    }

    return jaiThrow(vm.cTypeError,
                    "float.%s() needs a float as its receiver, got %s",
                    method, jaiTypeNameStatic(self));
}

static ObjString *intToString(int64_t value, int base) {
    static const char digits[] = "0123456789abcdefghijklmnopqrstuvwxyz";
    char buffer[70];
    char *p = buffer + sizeof buffer;
    uint64_t magnitude = magnitudeOf(value);
    const uint64_t radix = (uint64_t)base;

    do {
        *--p = digits[magnitude % radix];
        magnitude /= radix;
    } while (magnitude);

    if (value < 0)
        *--p = '-';

    return jaiStringIntern(p, (size_t)((buffer + sizeof buffer) - p));
}

static bool mIntAbs(int argc, Value *args, Value *out) {
    (void)argc;
    int64_t n;
    if (!intSelf(args, "abs", &n)) return false;
    if (n == INT64_MIN)
        return jaiThrow(vm.cOverflowError,
                        "abs(INT_MIN) is 2**63, which does not fit in an int");
    *out = INT_VAL(n < 0 ? -n : n);
    return true;
}

static bool mIntToStr(int argc, Value *args, Value *out) {
    int64_t n;
    if (!intSelf(args, "to_str", &n)) return false;

    int64_t base = 10;
    if (argc >= 2 && !IS_NULL(args[1]) &&
        !argIntFast(args[1], 1, "int.to_str", &base))
        return false;
    if (base < 2 || base > 36)
        return jaiThrow(vm.cValueError,
                        "int.to_str() base must be between 2 and 36, got %lld",
                        (long long)base);

    ObjString *text = intToString(n, (int)base);
    if (text == NULL) return false;
    *out = OBJ_VAL(text);
    return true;
}

static bool mIntToFloat(int argc, Value *args, Value *out) {
    (void)argc;
    int64_t n;
    if (!intSelf(args, "to_float", &n)) return false;
    *out = FLOAT_VAL((double)n);
    return true;
}

static bool mIntToInt(int argc, Value *args, Value *out) {
    (void)argc;
    int64_t ignored;
    if (!intSelf(args, "to_int", &ignored)) return false;
    *out = args[0];
    return true;
}

static bool mIntBitCount(int argc, Value *args, Value *out) {
    (void)argc;
    int64_t n;
    if (!intSelf(args, "bit_count", &n)) return false;
    *out = INT_VAL(bitCountU64((uint64_t)n));
    return true;
}

static bool mIntBitLength(int argc, Value *args, Value *out) {
    (void)argc;
    int64_t n;
    if (!intSelf(args, "bit_length", &n)) return false;
    *out = INT_VAL(bitLengthI64(n));
    return true;
}

static bool mIntMin(int argc, Value *args, Value *out) {
    (void)argc;
    int64_t n, other;
    if (!intSelf(args, "min", &n)) return false;
    if (!argIntFast(args[1], 1, "int.min", &other)) return false;
    *out = INT_VAL(n < other ? n : other);
    return true;
}

static bool mIntMax(int argc, Value *args, Value *out) {
    (void)argc;
    int64_t n, other;
    if (!intSelf(args, "max", &n)) return false;
    if (!argIntFast(args[1], 1, "int.max", &other)) return false;
    *out = INT_VAL(n > other ? n : other);
    return true;
}

static bool mIntClamp(int argc, Value *args, Value *out) {
    (void)argc;
    int64_t n, low, high;
    if (!intSelf(args, "clamp", &n)) return false;
    if (!argIntFast(args[1], 1, "int.clamp", &low)) return false;
    if (!argIntFast(args[2], 2, "int.clamp", &high)) return false;
    if (low > high)
        return jaiThrow(vm.cValueError,
                        "int.clamp() expects low <= high, got low=%lld, high=%lld",
                        (long long)low, (long long)high);
    *out = INT_VAL(n < low ? low : (n > high ? high : n));
    return true;
}

static bool mIntSign(int argc, Value *args, Value *out) {
    (void)argc;
    int64_t n;
    if (!intSelf(args, "sign", &n)) return false;
    *out = INT_VAL(n > 0 ? 1 : (n < 0 ? -1 : 0));
    return true;
}

static bool mIntIsEven(int argc, Value *args, Value *out) {
    (void)argc;
    int64_t n;
    if (!intSelf(args, "is_even", &n)) return false;
    /* Masking rather than `% 2 == 0`: the remainder of a negative odd number is
     * -1, which the naive test gets wrong. */
    *out = BOOL_VAL(((uint64_t)n & 1u) == 0);
    return true;
}

static bool mIntIsOdd(int argc, Value *args, Value *out) {
    (void)argc;
    int64_t n;
    if (!intSelf(args, "is_odd", &n)) return false;
    *out = BOOL_VAL(((uint64_t)n & 1u) != 0);
    return true;
}

static bool mIntPow(int argc, Value *args, Value *out) {
    (void)argc;
    int64_t n, exponent;
    if (!intSelf(args, "pow", &n)) return false;
    if (!argIntFast(args[1], 1, "int.pow", &exponent)) return false;
    if (exponent < 0)
        return jaiThrow(vm.cValueError,
                        "int.pow() expects a non-negative exponent, got %lld; "
                        "use ** for a float result", (long long)exponent);
    int64_t result;
    if (!powChecked(n, exponent, &result))
        return jaiThrow(vm.cOverflowError, "int.pow(): %lld ** %lld overflows",
                        (long long)n, (long long)exponent);
    *out = INT_VAL(result);
    return true;
}

static bool mIntPowMod(int argc, Value *args, Value *out) {
    (void)argc;
    int64_t n, exponent, modulus;
    if (!intSelf(args, "pow_mod", &n)) return false;
    if (!argIntFast(args[1], 1, "int.pow_mod", &exponent)) return false;
    if (!argIntFast(args[2], 2, "int.pow_mod", &modulus)) return false;
    int64_t result;
    if (!powModGuarded(n, exponent, modulus, "int.pow_mod", &result)) return false;
    *out = INT_VAL(result);
    return true;
}

static bool mNumberHash(int argc, Value *args, Value *out) {
    (void)argc;
    bool ok = true;
    uint64_t hash = jaiValueHash(args[0], &ok);
    if (!ok) {
        if (vm.hasException) return false;
        return jaiThrow(vm.cTypeError, "unhashable type: '%s'",
                        jaiTypeNameStatic(args[0]));
    }
    *out = INT_VAL((int64_t)hash);
    return true;
}

static bool mFloatAbs(int argc, Value *args, Value *out) {
    (void)argc;
    double x;
    if (!floatSelf(args, "abs", &x)) return false;
    *out = FLOAT_VAL(fabs(x));
    return true;
}

static bool mFloatToInt(int argc, Value *args, Value *out) {
    (void)argc;
    double x;
    if (!floatSelf(args, "to_int", &x)) return false;
    int64_t truncated;
    if (!floatToInt(x, "float.to_int", &truncated)) return false;
    *out = INT_VAL(truncated);
    return true;
}

static bool mFloatToFloat(int argc, Value *args, Value *out) {
    (void)argc;
    double ignored;
    if (!floatSelf(args, "to_float", &ignored)) return false;
    *out = args[0];
    return true;
}

static bool mFloatRound(int argc, Value *args, Value *out) {
    double x;
    if (!floatSelf(args, "round", &x)) return false;

    int64_t digits = 0;
    if (argc >= 2 && !IS_NULL(args[1]) &&
        !argIntFast(args[1], 1, "float.round", &digits))
        return false;

    if (digits == 0 || !isfinite(x)) {
        *out = FLOAT_VAL(round(x));
        return true;
    }
    /* Scaling past the format's range cannot round to anything the value does
     * not already equal, so the argument comes back untouched. */
    double scale = pow(10.0, (double)digits);
    double scaled = x * scale;
    if (!isfinite(scale) || scale == 0.0 || !isfinite(scaled)) {
        *out = FLOAT_VAL(x);
        return true;
    }
    *out = FLOAT_VAL(round(scaled) / scale);
    return true;
}

static bool mFloatFloor(int argc, Value *args, Value *out) {
    (void)argc;
    double x;
    if (!floatSelf(args, "floor", &x)) return false;
    *out = FLOAT_VAL(floor(x));
    return true;
}

static bool mFloatCeil(int argc, Value *args, Value *out) {
    (void)argc;
    double x;
    if (!floatSelf(args, "ceil", &x)) return false;
    *out = FLOAT_VAL(ceil(x));
    return true;
}

static bool mFloatTrunc(int argc, Value *args, Value *out) {
    (void)argc;
    double x;
    if (!floatSelf(args, "trunc", &x)) return false;
    *out = FLOAT_VAL(trunc(x));
    return true;
}

static bool mFloatSqrt(int argc, Value *args, Value *out) {
    (void)argc;
    double x;
    if (!floatSelf(args, "sqrt", &x)) return false;
    if (x < 0.0) return domainError("float.sqrt", "a non-negative receiver", x);
    *out = FLOAT_VAL(sqrt(x));
    return true;
}

static bool mFloatSign(int argc, Value *args, Value *out) {
    (void)argc;
    double x;
    if (!floatSelf(args, "sign", &x)) return false;
    /* A zero is returned as it came in, so -0.0 keeps its sign; NaN has no
     * sign to report and answers NaN. */
    *out = FLOAT_VAL(x > 0.0 ? 1.0 : (x < 0.0 ? -1.0 : x));
    return true;
}

static bool mFloatClamp(int argc, Value *args, Value *out) {
    (void)argc;
    double x, low, high;
    if (!floatSelf(args, "clamp", &x)) return false;
    if (!argNumberFast(args[1], 1, "float.clamp", &low)) return false;
    if (!argNumberFast(args[2], 2, "float.clamp", &high)) return false;
    if (low > high)
        return jaiThrow(vm.cValueError,
                        "float.clamp() expects low <= high, got low=%g, high=%g",
                        low, high);
    *out = FLOAT_VAL(x < low ? low : (x > high ? high : x));
    return true;
}

static bool mFloatIsNan(int argc, Value *args, Value *out) {
    (void)argc;
    double x;
    if (!floatSelf(args, "is_nan", &x)) return false;
    *out = BOOL_VAL(isnan(x) != 0);
    return true;
}

static bool mFloatIsInf(int argc, Value *args, Value *out) {
    (void)argc;
    double x;
    if (!floatSelf(args, "is_inf", &x)) return false;
    *out = BOOL_VAL(isinf(x) != 0);
    return true;
}

static bool mFloatIsFinite(int argc, Value *args, Value *out) {
    (void)argc;
    double x;
    if (!floatSelf(args, "is_finite", &x)) return false;
    *out = BOOL_VAL(isfinite(x) != 0);
    return true;
}

/* to_str goes through jaiValueToStr so that a number renders identically
 * whether it is printed, interpolated, or converted by hand. */
static bool mNumberToStr(int argc, Value *args, Value *out) {
    (void)argc;
    ObjString *text = jaiValueToStr(args[0]);
    if (text == NULL) return false;
    *out = OBJ_VAL(text);
    return true;
}

typedef struct {
    const char *name;
    uint8_t     length;
    JaiNativeFn fn;
    int8_t      minArity;
    int8_t      maxArity;
} MethodEntry;

#define METHOD_ENTRY(name_, fn_, min_, max_) \
    { (name_), (uint8_t)(sizeof(name_) - 1), (fn_), (min_), (max_) }

static const MethodEntry kIntMethods[] = {
    METHOD_ENTRY("abs",        mIntAbs,       1, 1),
    METHOD_ENTRY("bit_count",  mIntBitCount,  1, 1),
    METHOD_ENTRY("bit_length", mIntBitLength, 1, 1),
    METHOD_ENTRY("clamp",      mIntClamp,     3, 3),
    METHOD_ENTRY("hash",       mNumberHash,   1, 1),
    METHOD_ENTRY("is_even",    mIntIsEven,    1, 1),
    METHOD_ENTRY("is_odd",     mIntIsOdd,     1, 1),
    METHOD_ENTRY("max",        mIntMax,       2, 2),
    METHOD_ENTRY("min",        mIntMin,       2, 2),
    METHOD_ENTRY("pow",        mIntPow,       2, 2),
    METHOD_ENTRY("pow_mod",    mIntPowMod,    3, 3),
    METHOD_ENTRY("sign",       mIntSign,      1, 1),
    METHOD_ENTRY("to_float",   mIntToFloat,   1, 1),
    METHOD_ENTRY("to_int",     mIntToInt,     1, 1),
    METHOD_ENTRY("to_str",     mIntToStr,     1, 2),
};

static const MethodEntry kFloatMethods[] = {
    METHOD_ENTRY("abs",       mFloatAbs,      1, 1),
    METHOD_ENTRY("ceil",      mFloatCeil,     1, 1),
    METHOD_ENTRY("clamp",     mFloatClamp,    3, 3),
    METHOD_ENTRY("floor",     mFloatFloor,    1, 1),
    METHOD_ENTRY("hash",      mNumberHash,    1, 1),
    METHOD_ENTRY("is_finite", mFloatIsFinite, 1, 1),
    METHOD_ENTRY("is_inf",    mFloatIsInf,    1, 1),
    METHOD_ENTRY("is_nan",    mFloatIsNan,    1, 1),
    METHOD_ENTRY("round",     mFloatRound,    1, 2),
    METHOD_ENTRY("sign",      mFloatSign,     1, 1),
    METHOD_ENTRY("sqrt",      mFloatSqrt,     1, 1),
    METHOD_ENTRY("to_float",  mFloatToFloat,  1, 1),
    METHOD_ENTRY("to_int",    mFloatToInt,    1, 1),
    METHOD_ENTRY("to_str",    mNumberToStr,   1, 1),
    METHOD_ENTRY("trunc",     mFloatTrunc,    1, 1),
};

#undef METHOD_ENTRY

static inline bool bindFrom(const MethodEntry *table, size_t count,
                            Value receiver, ObjString *name, Value *out) {
    const size_t length = (size_t)name->length;
    if (length == 0) return false;

    const unsigned char first = (unsigned char)name->chars[0];

    for (size_t i = 0; i < count; ++i) {
        const MethodEntry *const e = table + i;

        if ((size_t)e->length != length ||
            (unsigned char)e->name[0] != first)
            continue;

        if (length > 1 &&
            memcmp(e->name + 1, name->chars + 1, length - 1) != 0)
            continue;

        *out = jaiBindNative(receiver, e->name, e->fn,
                             e->minArity, e->maxArity, NULL);
        return true;
    }

    return false;
}

bool jaiIntMethod(Value receiver, ObjString *name, Value *out) {
    if (!IS_INT(receiver) || name == NULL) return false;
    return bindFrom(kIntMethods, sizeof kIntMethods / sizeof kIntMethods[0],
                    receiver, name, out);
}

bool jaiFloatMethod(Value receiver, ObjString *name, Value *out) {
    if (!IS_FLOAT(receiver) || name == NULL) return false;
    return bindFrom(kFloatMethods, sizeof kFloatMethods / sizeof kFloatMethods[0],
                    receiver, name, out);
}
