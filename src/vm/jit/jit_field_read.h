/* jit_field_read.h — builtin methods whose entire body is a load from the
 * receiver.
 *
 * `k.len()` on a string compiled to 39 arm64 instructions and a `blr` into
 * jitInvokeNative: a callee Value, an argument Value, a root for every live
 * object the body holds, an argc, an nroots, a call, a status test and a tag
 * test. All of it marshalling, to read one 32-bit field. This table names the
 * field instead, so the invoke arm can emit the load and nothing else.
 *
 * It is deliberately a class and not a case. `len` is a field on a string, a
 * list, a dict, a set, a tuple and a bytes; the shape "the receiver's type
 * decides which field, and the type is already guarded" is what the entries
 * have in common, not the name.
 *
 * The key is (receiver ObjType, name, argc) because that is exactly what the
 * invoke arm holds: which builtin a name means is a function of the receiver's
 * type, and the type guard is emitted before the load. Keying on a name alone
 * would be wrong, and keying on it at all is only safe because
 * `tests/vm/field_natives.c` calls every one of these natives for real and
 * checks the field agrees with what it returns -- which is also the gate that
 * pins "a builtin named len returns an int", recorded in roadmap.md §6 as an
 * invariant that was true everywhere and checked nowhere.
 *
 * Every field named here is a count that cannot be negative, which is what
 * makes the zero-extending `ldr w` the arm emits equivalent to the native's
 * `INT_VAL((int64_t)field)`. The gate checks that too, on a real object.
 *
 * `sentinel` covers a field computed lazily: ObjString::scalars holds
 * UINT32_MAX until someone asks for it. The arm tests for it and falls into
 * the ordinary descriptor call, so the native itself fills the memo and every
 * later call takes the load. A *guard* would be wrong there -- a loop over
 * freshly built strings would deopt on every iteration, which is worse than
 * never compiling.
 */
#ifndef JAI_VM_JIT_FIELD_READ_H
#define JAI_VM_JIT_FIELD_READ_H

#include "vm/object/object.h"
#include "vm/table.h"
#include "vm/value.h"

#include <stddef.h>
#include <string.h>

typedef struct {
    ObjType     type;      /* receiver type this entry describes */
    const char *name;      /* method name, as the invoke arm sees it */
    uint8_t     argc;      /* arguments past the receiver; only 0 so far */
    uint16_t    offset;    /* byte offset of the field within the object */
    uint8_t     width;     /* 4 or 8 */
    ValueType   tag;       /* tag of the Value the native builds from it */
    bool        lazy;      /* `sentinel` means "not computed yet" */
    uint32_t    sentinel;
} JaiJitFieldRead;

/* One entry per native whose body is `*out = INT_VAL(self-><field>)`. Adding
 * one means adding a row to the gate's expectation table as well; it refuses
 * an entry it has no case for, so a new row cannot arrive unchecked. */
static const JaiJitFieldRead JAI_JIT_FIELD_READS[] = {
    {OBJ_STRING, "len", 0, (uint16_t)offsetof(ObjString, scalars), 4, VAL_INT,
     true, UINT32_MAX},
    {OBJ_LIST, "len", 0, (uint16_t)offsetof(ObjList, count), 4, VAL_INT,
     false, 0},
    {OBJ_DICT, "len", 0,
     (uint16_t)(offsetof(ObjDict, table) + offsetof(JaiTable, count)), 4,
     VAL_INT, false, 0},
    {OBJ_SET, "len", 0,
     (uint16_t)(offsetof(ObjSet, table) + offsetof(JaiTable, count)), 4,
     VAL_INT, false, 0},
    {OBJ_TUPLE, "len", 0, (uint16_t)offsetof(ObjTuple, count), 4, VAL_INT,
     false, 0},
    {OBJ_BYTES, "len", 0, (uint16_t)offsetof(ObjBytes, length), 4, VAL_INT,
     false, 0},
};

#define JAI_JIT_FIELD_READ_COUNT \
    ((int)(sizeof(JAI_JIT_FIELD_READS) / sizeof(JAI_JIT_FIELD_READS[0])))

/* NULL when this receiver type and name are not one. */
static inline const JaiJitFieldRead *jaiJitFieldReadFor(ObjType type,
                                                        const char *name,
                                                        size_t nameLen,
                                                        unsigned argc) {
    for (int i = 0; i < JAI_JIT_FIELD_READ_COUNT; i++) {
        const JaiJitFieldRead *r = &JAI_JIT_FIELD_READS[i];
        if (r->type != type || r->argc != argc) continue;
        if (strlen(r->name) != nameLen) continue;
        if (memcmp(r->name, name, nameLen) != 0) continue;
        return r;
    }
    return NULL;
}

#endif /* JAI_VM_JIT_FIELD_READ_H */
