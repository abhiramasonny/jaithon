/* builtins_math.c — the numeric, random and clock primitives, plus the method
 * tables for the two numeric types.
 *
 * Three groups of `__prim__` natives live here (spec Appendix C):
 *
 *   - `__prim__.f64_*`: the C math library, which Jaithon source cannot
 *     reproduce to the same accuracy;
 *   - `__prim__.random_u64` and `random_seed`: one xoshiro256** stream shared
 *     by the process, which std.random draws every distribution from;
 *   - `__prim__.time_*`: the two clocks, sleeping, and calendar text.
 *
 * The integer helpers further down are not primitives. They back the `int`
 * methods (`abs`, `bit_count`, `bit_length`, `pow_mod`); std.math writes gcd,
 * lcm and isqrt in Jaithon, which is where anything expressible belongs.
 *
 * A domain error — sqrt of a negative, log of a non-positive, asin outside
 * [-1, 1] — raises ValueError. It does not return NaN. A NaN that escapes a
 * math call propagates silently through every arithmetic operation downstream
 * and finally surfaces as a wrong number in a place that has nothing to do with
 * the call that produced it; by then there is nothing left to debug. Jaithon
 * fails where the mistake was made. A NaN *argument* is different: it is passed
 * through untouched, because the caller already had one and reporting it here
 * would hide where it came from.
 */

/* Feature macros must precede every include: -std=c11 alone does not expose
 * nanosleep, clock_gettime, strptime, localtime_r or getentropy. */
#if !defined(_POSIX_C_SOURCE)
#  define _POSIX_C_SOURCE 200809L
#endif
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#  define _DARWIN_C_SOURCE
#endif

#include <errno.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

#if defined(__APPLE__)
#  include <sys/random.h>
#  define JAI_HAVE_GETENTROPY 1
#elif defined(__GLIBC__) &&                                                    \
    (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 25))
#  include <sys/random.h>
#  define JAI_HAVE_GETENTROPY 1
#endif

#include "runtime.h"
#include "methods.h"
#include "../vm/gc.h"

/* ------------------------------------------------------------------ */
/* Constants                                                            */
/* ------------------------------------------------------------------ */

/* 2^63 as a double, exactly representable; the first value an int64 cannot
 * hold. Every float-to-int bound below is a strict comparison against it. */
static const double kTwoPow63 = 9223372036854775808.0;


/* Fast paths for the overwhelmingly common already-correct primitive argument.
 * Fall back to the canonical helpers so diagnostics/coercion semantics stay
 * exactly where the runtime defines them. */
static inline bool argNumberFast(Value v, int position, const char *fnName,
                                 double *out) {
    if (IS_FLOAT(v)) {
        *out = AS_FLOAT(v);
        return true;
    }
    if (IS_INT(v)) {
        *out = (double)AS_INT(v);
        return true;
    }
    return jaiArgNumber(v, position, fnName, out);
}

static inline bool argIntFast(Value v, int position, const char *fnName,
                              int64_t *out) {
    if (IS_INT(v)) {
        *out = AS_INT(v);
        return true;
    }
    return jaiArgInt(v, position, fnName, out);
}

/* ------------------------------------------------------------------ */
/* Shared helpers                                                       */
/* ------------------------------------------------------------------ */

/* |v| as an unsigned magnitude. Negating through uint64_t is the only way to
 * express |INT64_MIN|, which has no int64_t representation. */
static inline uint64_t magnitudeOf(int64_t v) {
    return v < 0 ? -(uint64_t)v : (uint64_t)v;
}

/* Turn an unsigned magnitude back into a signed value, given the sign it
 * should carry. Returns false when the magnitude does not fit. */
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

static inline bool domainError(const char *fnName, const char *expected,
                               double got) {
    return jaiThrow(vm.cValueError, "%s() expects %s, got %g",
                    fnName, expected, got);
}

/* sin, cos and tan of an infinity have no value: libm answers NaN and sets a
 * domain error. A NaN argument is left alone (see the file comment). */
static inline bool requireFinite(double x, const char *fnName) {
    if (!isinf(x)) return true;
    return domainError(fnName, "a finite argument", x);
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
/* __prim__.f64_*                                                       */
/* ------------------------------------------------------------------ */

/* The unary primitives with no domain restriction all have the same body; the
 * ones that can be handed an argument they have no value for are written out
 * below so that each can say what it expected. */
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

    /* Two cases where pow() answers NaN or infinity for a reason the caller can
     * act on: a negative base has no real fractional power, and zero has no
     * negative power at all. */
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
/* Random — xoshiro256**                                                */
/* ------------------------------------------------------------------ */

/* One stream for the process. std.random layers distributions on top of it;
 * nothing here is thread-safe, which matches the rest of the runtime. */
static uint64_t gRngState[4];
static bool     gRngSeeded;

static inline uint64_t splitMix64(uint64_t *state) {
    *state += 0x9E3779B97F4A7C15ULL;
    return jaiHashU64(*state);
}

/* An all-zero state is a fixed point of xoshiro, so the seed is expanded
 * through splitmix64, which cannot produce four zero words in a row. */
static void seedFromWord(uint64_t seed) {
    uint64_t state = seed;
    for (int i = 0; i < 4; i++) gRngState[i] = splitMix64(&state);
    gRngSeeded = true;
}

static bool readSystemEntropy(unsigned char *bytes, size_t length) {
#if defined(JAI_HAVE_GETENTROPY)
    /* getentropy is capped at 256 bytes per call, which is far above what the
     * 32-byte state needs. */
    if (getentropy(bytes, length) == 0) return true;
#endif
    FILE *urandom = fopen("/dev/urandom", "rb");
    if (urandom == NULL) return false;
    size_t got = fread(bytes, 1, length, urandom);
    (void)fclose(urandom);
    return got == length;
}

static void seedFromEntropy(void) {
    unsigned char bytes[sizeof gRngState];
    if (readSystemEntropy(bytes, sizeof bytes)) {
        memcpy(gRngState, bytes, sizeof gRngState);
        gRngSeeded = true;
        if ((gRngState[0] | gRngState[1] | gRngState[2] | gRngState[3]) == 0)
            seedFromWord(0x2545F4914F6CDD1DULL);
        return;
    }
    /* No entropy source. A fixed seed would replay the same stream on every
     * run, so mix in whatever this process can say about itself instead. */
    const void *here = (const void *)&bytes;
    uint64_t mixed = (uint64_t)(jaiClockMonotonic() * 1e9);
    mixed ^= jaiHashBytes(&here, sizeof here);
    seedFromWord(mixed);
}

static inline void ensureSeeded(void) {
    if (!gRngSeeded) seedFromEntropy();
}

static inline uint64_t rotateLeft(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static inline uint64_t randomNext(void) {
    uint64_t s0 = gRngState[0];
    uint64_t s1 = gRngState[1];
    uint64_t s2 = gRngState[2];
    uint64_t s3 = gRngState[3];

    const uint64_t result = rotateLeft(s1 * 5u, 7) * 9u;
    const uint64_t t = s1 << 17;

    s2 ^= s0;
    s3 ^= s1;
    s1 ^= s2;
    s0 ^= s3;
    s2 ^= t;
    s3 = rotateLeft(s3, 45);

    gRngState[0] = s0;
    gRngState[1] = s1;
    gRngState[2] = s2;
    gRngState[3] = s3;

    return result;
}

static bool nRandomU64(int argc, Value *args, Value *out) {
    (void)argc;
    (void)args;
    ensureSeeded();
    /* The full 64-bit pattern is handed over as an int; the top bit reads as a
     * sign, which is what `& INT_MAX` in std.random strips off. */
    *out = INT_VAL((int64_t)randomNext());
    return true;
}

static bool nRandomSeed(int argc, Value *args, Value *out) {
    if (argc == 0 || IS_NULL(args[0])) {
        seedFromEntropy();
        *out = NULL_VAL;
        return true;
    }
    int64_t seed;
    if (!argIntFast(args[0], 1, "random_seed", &seed)) return false;
    seedFromWord((uint64_t)seed);
    *out = NULL_VAL;
    return true;
}

/* ------------------------------------------------------------------ */
/* Time                                                                 */
/* ------------------------------------------------------------------ */

static inline bool secondsToNanos(double seconds, const char *fnName,
                                  int64_t *out) {
    const double nanos = seconds * 1e9;

    if (isnan(nanos) || nanos >= kTwoPow63 || nanos < -kTwoPow63)
        return jaiThrow(vm.cOverflowError,
                        "%s(): %g seconds does not fit in an int of nanoseconds",
                        fnName, seconds);

    *out = (int64_t)nanos;
    return true;
}

/* Both clocks report integer nanoseconds: the monotonic one from an arbitrary
 * origin, the wall one from the Unix epoch. std.time builds its Duration and
 * Instant directly on those counts. */
static bool nTimeMono(int argc, Value *args, Value *out) {
    (void)argc;
    (void)args;
    int64_t nanos;
    if (!secondsToNanos(jaiClockMonotonic(), "time_mono", &nanos)) return false;
    *out = INT_VAL(nanos);
    return true;
}

static bool nTimeMonoSeconds(int argc, Value *args, Value *out) {
    (void)argc;
    (void)args;
    *out = FLOAT_VAL(jaiClockMonotonic());
    return true;
}

static bool wallClock(const char *fnName, struct timespec *ts) {
    if (clock_gettime(CLOCK_REALTIME, ts) == 0) return true;
    return jaiThrow(vm.cOSError, "%s(): the system clock is unavailable: %s",
                    fnName, strerror(errno));
}

static bool nTimeWall(int argc, Value *args, Value *out) {
    (void)argc;
    (void)args;
    struct timespec ts;
    if (!wallClock("time_wall", &ts)) return false;
    if ((int64_t)ts.tv_sec > INT64_MAX / 1000000000)
        return jaiThrow(vm.cOverflowError,
                        "time_wall(): the clock is past the range of an int of "
                        "nanoseconds");
    *out = INT_VAL((int64_t)ts.tv_sec * 1000000000 + (int64_t)ts.tv_nsec);
    return true;
}

static bool nSleep(int argc, Value *args, Value *out) {
    (void)argc;
    double seconds;
    if (!argNumberFast(args[0], 1, "sleep", &seconds)) return false;
    if (isnan(seconds) || isinf(seconds))
        return domainError("sleep", "a finite duration in seconds", seconds);
    if (seconds < 0.0)
        return domainError("sleep", "a non-negative duration in seconds", seconds);

    double whole = floor(seconds);
    if (whole >= kTwoPow63)
        return jaiThrow(vm.cOverflowError, "sleep(): %g seconds is too long",
                        seconds);

    struct timespec request;
    request.tv_sec = (time_t)whole;
    long fraction = (long)((seconds - whole) * 1e9);
    request.tv_nsec = fraction > 999999999L ? 999999999L
                                            : (fraction < 0 ? 0 : fraction);

    /* A signal cuts the sleep short and reports what is left; the request is
     * restarted so that the caller waits for the duration it asked for. */
    struct timespec remaining;
    while (nanosleep(&request, &remaining) != 0) {
        if (errno != EINTR)
            return jaiThrow(vm.cOSError, "sleep(): %s", strerror(errno));
        request = remaining;
    }
    *out = NULL_VAL;
    return true;
}

/* Days from 1970-01-01 to the given proleptic Gregorian date (Hinnant's
 * days_from_civil). Written out rather than calling timegm, which is not in any
 * standard, and mktime, which would apply the local time zone. */
static inline int64_t daysFromCivil(int64_t year, int64_t month, int64_t day) {
    year -= month <= 2;
    const int64_t era = (year >= 0 ? year : year - 399) / 400;
    const int64_t yearOfEra = year - era * 400;
    const int64_t dayOfYear =
        (153 * (month + (month > 2 ? -3 : 9)) + 2) / 5 + day - 1;
    const int64_t dayOfEra =
        yearOfEra * 365 + yearOfEra / 4 - yearOfEra / 100 + dayOfYear;

    return era * 146097 + dayOfEra - 719468;
}

static inline bool tmToUnix(const struct tm *parts, int64_t *out) {
    const int64_t days =
        daysFromCivil((int64_t)parts->tm_year + 1900,
                      (int64_t)parts->tm_mon + 1,
                      (int64_t)parts->tm_mday);

    if (days > 106751991167LL || days < -106751991167LL)
        return false;

    *out = days * 86400 +
           (int64_t)parts->tm_hour * 3600 +
           (int64_t)parts->tm_min * 60 +
           (int64_t)parts->tm_sec;

    return true;
}

static bool nTimeFormat(int argc, Value *args, Value *out) {
    double seconds;
    ObjString *format;
    bool utc = false;

    if (!argNumberFast(args[0], 1, "time_format", &seconds)) return false;
    if (!jaiArgString(args[1], 2, "time_format", &format)) return false;
    if (argc >= 3 && !jaiArgBool(args[2], 3, "time_format", &utc)) return false;

    if (isnan(seconds) || isinf(seconds))
        return domainError("time_format", "a finite Unix timestamp", seconds);

    const double whole = floor(seconds);
    if (whole >= kTwoPow63 || whole < -kTwoPow63)
        return jaiThrow(vm.cOverflowError,
                        "time_format(): %g is out of range for a timestamp",
                        seconds);

    const time_t when = (time_t)whole;
    struct tm parts;
    struct tm *filled =
        utc ? gmtime_r(&when, &parts) : localtime_r(&when, &parts);

    if (filled == NULL)
        return jaiThrow(vm.cValueError,
                        "time_format(): %g is not a representable date",
                        seconds);

    if (format->length == 0) {
        ObjString *empty = jaiStringIntern("", 0);
        if (empty == NULL) return false;
        *out = OBJ_VAL(empty);
        return true;
    }

    /* Most formatted timestamps fit here: avoid allocator traffic entirely. */
    char stackBuffer[128];
    size_t written =
        strftime(stackBuffer, sizeof stackBuffer, format->chars, &parts);

    if (written > 0) {
        ObjString *text = jaiStringNew(stackBuffer, written);
        if (text == NULL) return false;
        *out = OBJ_VAL(text);
        return true;
    }

    size_t capacity = 256;
    for (;;) {
        char *buffer = JAI_ALLOC(char, capacity);
        written = strftime(buffer, capacity, format->chars, &parts);

        if (written > 0) {
            ObjString *text = jaiStringNew(buffer, written);
            JAI_FREE_ARRAY(char, buffer, capacity);
            if (text == NULL) return false;
            *out = OBJ_VAL(text);
            return true;
        }

        JAI_FREE_ARRAY(char, buffer, capacity);

        if (capacity >= 65536)
            return jaiThrow(vm.cValueError,
                            "time_format(): '%s' produces more than 64 KiB",
                            format->chars);

        capacity <<= 1;
    }
}

static bool nTimeParse(int argc, Value *args, Value *out) {
    ObjString *text, *format;
    bool utc = false;
    if (!jaiArgString(args[0], 1, "time_parse", &text)) return false;
    if (!jaiArgString(args[1], 2, "time_parse", &format)) return false;
    if (argc >= 3 && !jaiArgBool(args[2], 3, "time_parse", &utc)) return false;

    struct tm parts;
    memset(&parts, 0, sizeof parts);
    /* strptime leaves untouched fields alone, and a zero day-of-month is not a
     * date; -1 lets mktime work out whether daylight saving is in effect. */
    parts.tm_mday = 1;
    parts.tm_isdst = -1;

    if (strptime(text->chars, format->chars, &parts) == NULL)
        return jaiThrow(vm.cValueError, "time_parse(): '%s' does not match '%s'",
                        text->chars, format->chars);

    int64_t seconds;
    if (utc) {
        if (!tmToUnix(&parts, &seconds))
            return jaiThrow(vm.cOverflowError,
                            "time_parse(): '%s' is out of range for a timestamp",
                            text->chars);
    } else {
        time_t local = mktime(&parts);
        if (local == (time_t)-1)
            return jaiThrow(vm.cValueError,
                            "time_parse(): '%s' is not a valid local time",
                            text->chars);
        seconds = (int64_t)local;
    }
    *out = INT_VAL(seconds);
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

void jaiRegisterRandomPrimitives(void) {
    if (vm.builtins == NULL) return;

    jaiDefineNative("__prim__.random_u64",  nRandomU64,  0, 0);
    jaiDefineNative("__prim__.random_seed", nRandomSeed, 0, 1);
}

void jaiRegisterTimePrimitives(void) {
    if (vm.builtins == NULL) return;

    jaiDefineNative("__prim__.time_mono", nTimeMono, 0, 0);
    jaiDefineNative("__prim__.time_wall", nTimeWall, 0, 0);
    jaiDefineNative("__prim__.time_mono_seconds", nTimeMonoSeconds, 0, 0);

    /* Appendix C names this `sleep`. `time_sleep` is the same native under a
     * second name and exists only because lib/std/time.jai still calls it; it
     * goes when that one call is repointed. */
    jaiDefineNative("__prim__.sleep",      nSleep, 1, 1);
    jaiDefineNative("__prim__.time_sleep", nSleep, 1, 1);

    jaiDefineNative("__prim__.time_format", nTimeFormat, 2, 3);
    jaiDefineNative("__prim__.time_parse",  nTimeParse,  2, 3);
}

/* ------------------------------------------------------------------ */
/* int and float methods                                                */
/* ------------------------------------------------------------------ */

/* A bound native is called with the receiver in args[0] and argc counting it,
 * so every method below reads its own arguments from args[1] onwards and every
 * arity in the tables includes the receiver. */

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
