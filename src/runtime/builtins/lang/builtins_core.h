/* builtins_core.h — what the three builtins_core translation units share.
 *
 * The builtins of spec §9 are one registration table, built in
 * builtins_core.c, whose implementations are split by family:
 * builtins_core.c itself keeps the printing/identity/reflection builtins and
 * the helpers below, builtins_convert.c the type constructors, and
 * builtins_iter.c the builtins that consume an iterable. This header carries
 * the handful of helpers shared between them and every JaiNativeFn
 * jaiRegisterCoreBuiltins binds from outside builtins_core.c. Not a public
 * interface: builtins.h is the contract with the rest of the runtime.
 */
#ifndef JAI_BUILTINS_CORE_H
#define JAI_BUILTINS_CORE_H

#include "runtime/builtins/builtins.h"

/* --- shared helpers (defined in builtins_core.c) -------------------- */

bool valueLength(Value v, int64_t *out);

ObjList *collectIterable(Value v);

/* builtins_list.c has a file-local static of this name and shape for the
 * list.sort() path; that one keeps internal linkage and binds to itself. */
ObjList *sortedCopy(ObjList *items, ObjList *keys, bool reverse,
                    const char *fnName);

/* --- type constructors defined in builtins_convert.c ---------------- */

bool nIntConv(int argc, Value *args, Value *out);
bool nFloatConv(int argc, Value *args, Value *out);
bool nBoolConv(int argc, Value *args, Value *out);
bool nListConv(int argc, Value *args, Value *out);
bool nDictConv(int argc, Value *args, Value *out);
bool nSetConv(int argc, Value *args, Value *out);
bool nTupleConv(int argc, Value *args, Value *out);
bool nBytesConv(int argc, Value *args, Value *out);

/* --- iterable builtins defined in builtins_iter.c ------------------- */

bool nMin(int argc, Value *args, Value *out);
bool nMax(int argc, Value *args, Value *out);
bool nSum(int argc, Value *args, Value *out);
bool nSorted(int argc, Value *args, Value *out);
bool nReversed(int argc, Value *args, Value *out);
bool nEnumerate(int argc, Value *args, Value *out);
bool nZip(int argc, Value *args, Value *out);
bool nMap(int argc, Value *args, Value *out);
bool nFilter(int argc, Value *args, Value *out);
bool nAny(int argc, Value *args, Value *out);
bool nAll(int argc, Value *args, Value *out);

#endif /* JAI_BUILTINS_CORE_H */
