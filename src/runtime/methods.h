/* methods.h — built-in method tables for the primitive types. */
#ifndef JAI_METHODS_H
#define JAI_METHODS_H

#include "vm/object/object.h"

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

bool jaiValueFormatMethod(Value receiver, ObjString *name, Value *out);

Value jaiBindNative(Value receiver, const char *name, JaiNativeFn fn,
                    int minArity, int maxArity, const char *const *paramNames);

void jaiMethodTablesInit(void);

ObjList *jaiBuiltinMethodNames(Value receiver);

#endif /* JAI_METHODS_H */
