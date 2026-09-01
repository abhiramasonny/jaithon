/* builtins_math.h — what the three builtins_math translation units share.
 *
 * builtins_math.c keeps the __prim__.f64_* surface, builtins_time.c the RNG
 * stream and the two clocks, builtins_numeric.c the int and float method
 * tables. Only those three include this header; it is not a public interface.
 */
#ifndef JAI_BUILTINS_MATH_H
#define JAI_BUILTINS_MATH_H

#include "runtime/runtime.h"
#include "runtime/methods.h"

/* ------------------------------------------------------------------ */
/* Constants                                                            */
/* ------------------------------------------------------------------ */

/* 2^63 as a double, exactly representable; the first value an int64 cannot
 * hold. Every float-to-int bound below is a strict comparison against it. */
static const double kTwoPow63 = 9223372036854775808.0;


/* Fast path for an already-correct primitive argument; anything else falls
 * back to the canonical helper so diagnostics/coercion stay centralized. */
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

static inline bool domainError(const char *fnName, const char *expected,
                               double got) {
    return jaiThrow(vm.cValueError, "%s() expects %s, got %g",
                    fnName, expected, got);
}

#endif /* JAI_BUILTINS_MATH_H */
