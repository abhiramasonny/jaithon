/* builtins.h — what builtins.c, builtins_core.c and builtins_prim.c share.
 * Not a public interface: runtime.h declares what the VM and registrar call. */
#ifndef JAI_BUILTINS_H
#define JAI_BUILTINS_H

#include "runtime/runtime.h"

/* --- defined in builtins.c ----------------------------------------- */

bool jaiBuiltinIsCallable(Value v);

bool jaiBuiltinArgTypeError(int index, const char *fnName, const char *expected,
                            Value got);

/* --- defined in builtins_core.c ------------------------------------ */

bool jaiBuiltinMatchesType(Value v, Value t, bool *matched);

/* --- defined in builtins_prim.c ------------------------------------ */

bool jaiBuiltinAddI64(int64_t a, int64_t b, int64_t *out);
bool jaiBuiltinOverflowError(const char *op);

void jaiRegisterOperatorPrimitives(void);

#endif /* JAI_BUILTINS_H */
