/* builtins_math.c — the __prim__.f64_* primitives.
 *
 * A domain error (sqrt of a negative, log of a non-positive, ...) raises
 * ValueError instead of returning NaN, which would corrupt a result silently
 * downstream; a NaN *argument*, though, passes through untouched. */

#include <math.h>

#include "runtime/builtins/builtins_math.h"

/* ------------------------------------------------------------------ */
/* Shared helpers                                                       */
/* ------------------------------------------------------------------ */

/* sin, cos and tan of an infinity have no value: libm answers NaN and sets a
 * domain error. A NaN argument is left alone (see the file comment). */
static inline bool requireFinite(double x, const char *fnName) {
    if (!isinf(x)) return true;
    return domainError(fnName, "a finite argument", x);
}

/* ------------------------------------------------------------------ */
/* __prim__.f64_*                                                       */
/* ------------------------------------------------------------------ */

/* Unrestricted-domain unaries share this body via macro; the ones with a
 * domain error are written out below so each can say what it expected. */
#define F64_UNARY(cName, jaiName, expr)                                        \
    static bool cName(int argc, Value *args, Value *out) {                     \
        (void)argc;                                                            \
        double x;                                                              \
        if (!argNumberFast(args[0], 1, jaiName, &x)) return false;              \
        *out = FLOAT_VAL(expr);                                                \
        return true;                                                           \
    }

F64_UNARY(nF64Exp,   "f64_exp",   exp(x))
F64_UNARY(nF64Atan,  "f64_atan",  atan(x))
F64_UNARY(nF64Sinh,  "f64_sinh",  sinh(x))
F64_UNARY(nF64Cosh,  "f64_cosh",  cosh(x))
F64_UNARY(nF64Tanh,  "f64_tanh",  tanh(x))
F64_UNARY(nF64Asinh, "f64_asinh", asinh(x))
F64_UNARY(nF64Floor, "f64_floor", floor(x))
F64_UNARY(nF64Ceil,  "f64_ceil",  ceil(x))
F64_UNARY(nF64Trunc, "f64_trunc", trunc(x))
F64_UNARY(nF64Round, "f64_round", round(x))
F64_UNARY(nF64Erf,   "f64_erf",   erf(x))

#undef F64_UNARY

static bool nF64Sqrt(int argc, Value *args, Value *out) {
    (void)argc;
    double x;
    if (!argNumberFast(args[0], 1, "f64_sqrt", &x)) return false;
    if (x < 0.0) return domainError("f64_sqrt", "a non-negative argument", x);
    *out = FLOAT_VAL(sqrt(x));
    return true;
}

/* The three logarithms share their domain: zero is a pole, not a value. */
static bool logOf(Value arg, const char *fnName, double (*compute)(double),
                  Value *out) {
    double x;
    if (!argNumberFast(arg, 1, fnName, &x)) return false;
    if (x < 0.0) return domainError(fnName, "a positive argument", x);
    if (x == 0.0)
        return jaiThrow(vm.cValueError, "%s(0.0) is undefined; the limit is -inf",
                        fnName);
    *out = FLOAT_VAL(compute(x));
    return true;
}

static bool nF64Log(int argc, Value *args, Value *out) {
    (void)argc;
    return logOf(args[0], "f64_log", log, out);
}

static bool nF64Log2(int argc, Value *args, Value *out) {
    (void)argc;
    return logOf(args[0], "f64_log2", log2, out);
}

static bool nF64Log10(int argc, Value *args, Value *out) {
    (void)argc;
    return logOf(args[0], "f64_log10", log10, out);
}

static bool nF64Sin(int argc, Value *args, Value *out) {
    (void)argc;
    double x;
    if (!argNumberFast(args[0], 1, "f64_sin", &x)) return false;
    if (!requireFinite(x, "f64_sin")) return false;
    *out = FLOAT_VAL(sin(x));
    return true;
}

static bool nF64Cos(int argc, Value *args, Value *out) {
    (void)argc;
    double x;
    if (!argNumberFast(args[0], 1, "f64_cos", &x)) return false;
    if (!requireFinite(x, "f64_cos")) return false;
    *out = FLOAT_VAL(cos(x));
    return true;
}

static bool nF64Tan(int argc, Value *args, Value *out) {
    (void)argc;
    double x;
    if (!argNumberFast(args[0], 1, "f64_tan", &x)) return false;
    if (!requireFinite(x, "f64_tan")) return false;
    *out = FLOAT_VAL(tan(x));
    return true;
}

static bool nF64Asin(int argc, Value *args, Value *out) {
    (void)argc;
    double x;
    if (!argNumberFast(args[0], 1, "f64_asin", &x)) return false;
    if (x < -1.0 || x > 1.0)
        return domainError("f64_asin", "an argument in [-1.0, 1.0]", x);
    *out = FLOAT_VAL(asin(x));
    return true;
}

static bool nF64Acos(int argc, Value *args, Value *out) {
    (void)argc;
    double x;
    if (!argNumberFast(args[0], 1, "f64_acos", &x)) return false;
    if (x < -1.0 || x > 1.0)
        return domainError("f64_acos", "an argument in [-1.0, 1.0]", x);
    *out = FLOAT_VAL(acos(x));
    return true;
}

static bool nF64Acosh(int argc, Value *args, Value *out) {
    (void)argc;
    double x;
    if (!argNumberFast(args[0], 1, "f64_acosh", &x)) return false;
    if (x < 1.0) return domainError("f64_acosh", "an argument >= 1.0", x);
    *out = FLOAT_VAL(acosh(x));
    return true;
}

static bool nF64Atanh(int argc, Value *args, Value *out) {
    (void)argc;
    double x;
    if (!argNumberFast(args[0], 1, "f64_atanh", &x)) return false;
    /* ±1 are poles, so the open interval is the whole domain. */
    if (x <= -1.0 || x >= 1.0)
        return domainError("f64_atanh", "an argument in (-1.0, 1.0)", x);
    *out = FLOAT_VAL(atanh(x));
    return true;
}

static bool nF64Atan2(int argc, Value *args, Value *out) {
    (void)argc;
    double y, x;
    if (!argNumberFast(args[0], 1, "f64_atan2", &y)) return false;
    if (!argNumberFast(args[1], 2, "f64_atan2", &x)) return false;
    *out = FLOAT_VAL(atan2(y, x));
    return true;
}

static bool nF64Fmod(int argc, Value *args, Value *out) {
    (void)argc;
    double x, y;
    if (!argNumberFast(args[0], 1, "f64_fmod", &x)) return false;
    if (!argNumberFast(args[1], 2, "f64_fmod", &y)) return false;
    if (y == 0.0)
        return jaiThrow(vm.cValueError, "f64_fmod() expects a non-zero divisor");
    *out = FLOAT_VAL(fmod(x, y));
    return true;
}

static bool nF64Pow(int argc, Value *args, Value *out) {
    (void)argc;
    double base, exponent;
    if (!argNumberFast(args[0], 1, "f64_pow", &base)) return false;
    if (!argNumberFast(args[1], 2, "f64_pow", &exponent)) return false;

    /* Two cases where pow() would answer NaN/inf uninformatively: a negative
     * base with a non-integral exponent, and zero to a negative power. */
    if (base < 0.0 && isfinite(exponent) && exponent != trunc(exponent))
        return jaiThrow(vm.cValueError,
                        "f64_pow() expects an integral exponent for a negative "
                        "base, got %g ** %g", base, exponent);
    if (base == 0.0 && exponent < 0.0)
        return jaiThrow(vm.cDivisionByZeroError, "zero to a negative power");

    *out = FLOAT_VAL(pow(base, exponent));
    return true;
}

static bool nF64Hypot(int argc, Value *args, Value *out) {
    (void)argc;
    double x, y;
    if (!argNumberFast(args[0], 1, "f64_hypot", &x)) return false;
    if (!argNumberFast(args[1], 2, "f64_hypot", &y)) return false;
    *out = FLOAT_VAL(hypot(x, y));
    return true;
}

static bool nF64Copysign(int argc, Value *args, Value *out) {
    (void)argc;
    double magnitude, source;
    if (!argNumberFast(args[0], 1, "f64_copysign", &magnitude)) return false;
    if (!argNumberFast(args[1], 2, "f64_copysign", &source)) return false;
    *out = FLOAT_VAL(copysign(magnitude, source));
    return true;
}

static bool nF64Frexp(int argc, Value *args, Value *out) {
    (void)argc;
    double x;
    if (!argNumberFast(args[0], 1, "f64_frexp", &x)) return false;
    int exponent = 0;
    double mantissa = frexp(x, &exponent);
    Value pair[2] = {FLOAT_VAL(mantissa), INT_VAL(exponent)};
    *out = OBJ_VAL(jaiTupleNew(pair, 2));
    return true;
}

static bool nF64Ldexp(int argc, Value *args, Value *out) {
    (void)argc;
    double mantissa;
    int64_t exponent;
    if (!argNumberFast(args[0], 1, "f64_ldexp", &mantissa)) return false;
    if (!argIntFast(args[1], 2, "f64_ldexp", &exponent)) return false;
    /* ldexp takes an int; an exponent past the format's range saturates to zero
     * or infinity, which is what clamping to INT_MIN/INT_MAX also produces. */
    int clamped = exponent > 100000 ? 100000 : (exponent < -100000 ? -100000
                                                                   : (int)exponent);
    *out = FLOAT_VAL(ldexp(mantissa, clamped));
    return true;
}

static bool nF64Modf(int argc, Value *args, Value *out) {
    (void)argc;
    double x;
    if (!argNumberFast(args[0], 1, "f64_modf", &x)) return false;
    double integral = 0.0;
    double fractional = modf(x, &integral);
    Value pair[2] = {FLOAT_VAL(fractional), FLOAT_VAL(integral)};
    *out = OBJ_VAL(jaiTupleNew(pair, 2));
    return true;
}

/* tgamma and lgamma have poles at zero and every negative integer. */
static inline bool gammaDomain(double x, const char *fnName) {
    if (x > 0.0 || isnan(x)) return true;

    if (isinf(x) && x < 0.0)
        return domainError(fnName, "an argument other than -inf", x);

    if (x == trunc(x))
        return jaiThrow(vm.cValueError,
                        "%s() has a pole at %g; the gamma function is undefined "
                        "at zero and the negative integers",
                        fnName, x);

    return true;
}

static bool nF64Gamma(int argc, Value *args, Value *out) {
    (void)argc;
    double x;
    if (!argNumberFast(args[0], 1, "f64_gamma", &x)) return false;
    if (!gammaDomain(x, "f64_gamma")) return false;
    *out = FLOAT_VAL(tgamma(x));
    return true;
}

static bool nF64Lgamma(int argc, Value *args, Value *out) {
    (void)argc;
    double x;
    if (!argNumberFast(args[0], 1, "f64_lgamma", &x)) return false;
    if (!gammaDomain(x, "f64_lgamma")) return false;
    *out = FLOAT_VAL(lgamma(x));
    return true;
}

static bool nF64IsNan(int argc, Value *args, Value *out) {
    (void)argc;
    double x;
    if (!argNumberFast(args[0], 1, "f64_is_nan", &x)) return false;
    *out = BOOL_VAL(isnan(x) != 0);
    return true;
}

static bool nF64IsInf(int argc, Value *args, Value *out) {
    (void)argc;
    double x;
    if (!argNumberFast(args[0], 1, "f64_is_inf", &x)) return false;
    *out = BOOL_VAL(isinf(x) != 0);
    return true;
}

static bool nF64IsFinite(int argc, Value *args, Value *out) {
    (void)argc;
    double x;
    if (!argNumberFast(args[0], 1, "f64_is_finite", &x)) return false;
    *out = BOOL_VAL(isfinite(x) != 0);
    return true;
}

/* ------------------------------------------------------------------ */
/* Registration                                                         */
/* ------------------------------------------------------------------ */

void jaiRegisterMathPrimitives(void) {
    if (vm.builtins == NULL) return;

    jaiDefineNative("__prim__.f64_sqrt",  nF64Sqrt,  1, 1);
    jaiDefineNative("__prim__.f64_exp",   nF64Exp,   1, 1);
    jaiDefineNative("__prim__.f64_log",   nF64Log,   1, 1);
    jaiDefineNative("__prim__.f64_log2",  nF64Log2,  1, 1);
    jaiDefineNative("__prim__.f64_log10", nF64Log10, 1, 1);

    jaiDefineNative("__prim__.f64_sin",   nF64Sin,   1, 1);
    jaiDefineNative("__prim__.f64_cos",   nF64Cos,   1, 1);
    jaiDefineNative("__prim__.f64_tan",   nF64Tan,   1, 1);
    jaiDefineNative("__prim__.f64_asin",  nF64Asin,  1, 1);
    jaiDefineNative("__prim__.f64_acos",  nF64Acos,  1, 1);
    jaiDefineNative("__prim__.f64_atan",  nF64Atan,  1, 1);
    jaiDefineNative("__prim__.f64_atan2", nF64Atan2, 2, 2);

    jaiDefineNative("__prim__.f64_sinh",  nF64Sinh,  1, 1);
    jaiDefineNative("__prim__.f64_cosh",  nF64Cosh,  1, 1);
    jaiDefineNative("__prim__.f64_tanh",  nF64Tanh,  1, 1);
    jaiDefineNative("__prim__.f64_asinh", nF64Asinh, 1, 1);
    jaiDefineNative("__prim__.f64_acosh", nF64Acosh, 1, 1);
    jaiDefineNative("__prim__.f64_atanh", nF64Atanh, 1, 1);

    jaiDefineNative("__prim__.f64_floor", nF64Floor, 1, 1);
    jaiDefineNative("__prim__.f64_ceil",  nF64Ceil,  1, 1);
    jaiDefineNative("__prim__.f64_trunc", nF64Trunc, 1, 1);
    jaiDefineNative("__prim__.f64_round", nF64Round, 1, 1);

    jaiDefineNative("__prim__.f64_fmod",     nF64Fmod,     2, 2);
    jaiDefineNative("__prim__.f64_pow",      nF64Pow,      2, 2);
    jaiDefineNative("__prim__.f64_hypot",    nF64Hypot,    2, 2);
    jaiDefineNative("__prim__.f64_copysign", nF64Copysign, 2, 2);
    jaiDefineNative("__prim__.f64_frexp",    nF64Frexp,    1, 1);
    jaiDefineNative("__prim__.f64_ldexp",    nF64Ldexp,    2, 2);
    jaiDefineNative("__prim__.f64_modf",     nF64Modf,     1, 1);

    jaiDefineNative("__prim__.f64_erf",    nF64Erf,    1, 1);
    jaiDefineNative("__prim__.f64_gamma",  nF64Gamma,  1, 1);
    jaiDefineNative("__prim__.f64_lgamma", nF64Lgamma, 1, 1);

    jaiDefineNative("__prim__.f64_is_nan",    nF64IsNan,    1, 1);
    jaiDefineNative("__prim__.f64_is_inf",    nF64IsInf,    1, 1);
    jaiDefineNative("__prim__.f64_is_finite", nF64IsFinite, 1, 1);
}
