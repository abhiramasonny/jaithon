/* builtins.h — what builtins.c, builtins_core.c and builtins_prim.c share.
 * Not a public interface: runtime.h declares what the VM and registrar call. */
#ifndef JAI_BUILTINS_H
#define JAI_BUILTINS_H

#include "runtime/runtime.h"

/* --- defined in builtins.c ----------------------------------------- */

/* Callable in the sense the call path means it: anything jaiCallValue accepts. */
bool jaiBuiltinIsCallable(Value v);

/* Consistent wording for every argument mismatch. `index` is 0-based over the
 * user-visible arguments. */
bool jaiBuiltinArgTypeError(int index, const char *fnName, const char *expected,
                            Value got);

/* --- defined in builtins_core.c ------------------------------------ */

/* Builtin type tokens are the native conversion functions, so this tests
 * identity against those. */
bool jaiBuiltinMatchesType(Value v, Value t, bool *matched);

/* --- defined in builtins_prim.c ------------------------------------ */

/* Plain + is checked: spec §2.5 says it raises OverflowError rather than
 * wrapping, so sum() and the `__prim__` operators share one implementation. */
bool jaiBuiltinAddI64(int64_t a, int64_t b, int64_t *out);
bool jaiBuiltinOverflowError(const char *op);

/* The __prim__ operator surface (spec Appendix C); registered from
 * jaiRegisterCoreBuiltins for one entry point. */
void jaiRegisterOperatorPrimitives(void);

#endif /* JAI_BUILTINS_H */
