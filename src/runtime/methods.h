/* methods.h — built-in method tables for the primitive types.
 *
 * `"abc".upper()` and `xs.map(f)` do not go through a class: the VM asks
 * jaiBuiltinMethod for a bound native. Each primitive type contributes one
 * lookup function here; builtins.c dispatches on the receiver's type.
 *
 * Every lookup returns a *bound* value (ObjBound wrapping an ObjNative) so the
 * call path is identical to a user method call.
 */
#ifndef JAI_METHODS_H
#define JAI_METHODS_H

#include "../vm/object.h"

/* Each returns false when `name` is not a method of that type. */
bool jaiStrMethod(Value receiver, ObjString *name, Value *out);
bool jaiListMethod(Value receiver, ObjString *name, Value *out);
bool jaiDictMethod(Value receiver, ObjString *name, Value *out);
bool jaiSetMethod(Value receiver, ObjString *name, Value *out);
bool jaiTupleMethod(Value receiver, ObjString *name, Value *out);
bool jaiRangeMethod(Value receiver, ObjString *name, Value *out);
bool jaiBytesMethod(Value receiver, ObjString *name, Value *out);
bool jaiIntMethod(Value receiver, ObjString *name, Value *out);
bool jaiFloatMethod(Value receiver, ObjString *name, Value *out);
bool jaiFileMethod(Value receiver, ObjString *name, Value *out);
bool jaiModuleMethod(Value receiver, ObjString *name, Value *out);
bool jaiIterMethod(Value receiver, ObjString *name, Value *out);

/* Not a per-type table: the parser lowers every f-string hole that carries a
 * format spec to `value.__format__(spec)`, so every receiver must answer. */
bool jaiValueFormatMethod(Value receiver, ObjString *name, Value *out);

/* Registration helper used by each of the above: builds (and memoises) an
 * ObjNative for `name`, then wraps it in an ObjBound around `receiver`.
 * `paramNames` names args[0..maxArity-1] — args[0] is the receiver — so that
 * `xs.sorted(reverse: true)` can bind; NULL means positional only. */
Value jaiBindNative(Value receiver, const char *name, JaiNativeFn fn,
                    int minArity, int maxArity, const char *const *paramNames);

/* Method-table registries are initialised once at startup. */
void jaiMethodTablesInit(void);

/* Introspection for `dir()` and the documentation generator. */
ObjList *jaiBuiltinMethodNames(Value receiver);

#endif /* JAI_METHODS_H */
