/* builtins.h — what the three halves of the builtin namespace share.
 *
 * builtins.c owns registration, the argument helpers and the method-table
 * dispatcher; builtins_core.c owns the functions a program can name
 * (print, len, sorted, the conversions); builtins_prim.c owns the `__prim__`
 * operators that lib/std is written over. Five things straddle that split and
 * this header is all of them.
 *
 * Not a public interface: runtime.h declares what the VM and the registrar
 * call, and nothing below belongs there.
 */
#ifndef JAI_BUILTINS_H
#define JAI_BUILTINS_H

#include "runtime.h"

/* --- defined in builtins.c ----------------------------------------- */

/* Callable in the sense the call path means it: anything jaiCallValue accepts. */
bool jaiBuiltinIsCallable(Value v);

/* One wording for every argument mismatch, so that a native reads as a
 * language error rather than as a C function that was handed the wrong thing.
 * `index` is 0-based over the user-visible arguments. */
bool jaiBuiltinArgTypeError(int index, const char *fnName, const char *expected,
                            Value got);

/* --- defined in builtins_core.c ------------------------------------ */

/* Does `v` match the type token `t`? The builtin type names are the native
 * conversion functions, so the test is by identity against those. */
bool jaiBuiltinMatchesType(Value v, Value t, bool *matched);

/* --- defined in builtins_prim.c ------------------------------------ */

/* Plain + is checked: spec §2.5 says it raises OverflowError rather than
 * wrapping, so sum() and the `__prim__` operators share one implementation. */
bool jaiBuiltinAddI64(int64_t a, int64_t b, int64_t *out);
bool jaiBuiltinOverflowError(const char *op);

/* The `__prim__` operator surface (spec Appendix C), registered from
 * jaiRegisterCoreBuiltins so that there is still one entry point. */
void jaiRegisterOperatorPrimitives(void);

#endif /* JAI_BUILTINS_H */
