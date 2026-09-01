/* builtins_core.c — the functions a Jaithon program can name without importing anything (spec §9). */

#include <math.h>
#include <stdlib.h>

#include "runtime/builtins/lang/builtins_core.h"
#include "runtime/builtins/collections/builtins_seq.h"
#include "runtime/methods.h"
#include "runtime/runtime.h"

#include "vm/gc.h"

// Small shared helpers

static ObjString *dunderName(ObjString *cached, const char *text) {
    return cached != NULL ? cached : jaiStringInternC(text);
}

bool valueLength(Value v, int64_t *out) {
    if (!IS_OBJ(v)) return false;
    switch (OBJ_TYPE(v)) {
    case OBJ_STRING: *out = (int64_t)jaiStringScalarCount(AS_STRING(v)); return true;
    case OBJ_BYTES:  *out = (int64_t)AS_BYTES(v)->length;                return true;
    case OBJ_LIST:   *out = (int64_t)AS_LIST(v)->count;                  return true;
    case OBJ_TUPLE:  *out = (int64_t)AS_TUPLE(v)->count;                 return true;
    case OBJ_DICT:   *out = (int64_t)AS_DICT(v)->table.count;            return true;
    case OBJ_SET:    *out = (int64_t)AS_SET(v)->table.count;             return true;
    case OBJ_RANGE:  *out = jaiRangeLength(AS_RANGE(v));                 return true;
    case OBJ_INSTANCE: {
        ObjInstance *inst = AS_INSTANCE(v);
        if (inst->klass == NULL || IS_NULL(inst->klass->dunderLen)) return false;
        Value result;
        if (!jaiInvokeMethod(v, dunderName(vm.strLen, "__len__"), 0, NULL, &result))
            return false;
        if (!IS_INT(result)) {
            jaiThrow(vm.cTypeError, "__len__ must return int, not %s",
                     jaiTypeNameStatic(result));
            return false;
        }
        if (AS_INT(result) < 0) {
            jaiThrow(vm.cValueError, "__len__ must return a non-negative int");
            return false;
        }
        *out = AS_INT(result);
        return true;
    }
    default:
        return false;
    }
}

ObjList *collectIterable(Value v) {
    Value iterVal;
    if (!jaiGetIter(v, &iterVal)) return NULL;
    if (!IS_ITER(iterVal)) {
        jaiThrow(vm.cTypeError, "'%s' object is not iterable", jaiTypeNameStatic(v));
        return NULL;
    }

    jaiGCPushRoot(iterVal);
    ObjList *out = jaiListNew(0);
    jaiGCPushRoot(OBJ_VAL(out));

    Value item;
    while (jaiIterNext(AS_ITER(iterVal), &item)) {
        jaiGCPushRoot(item);
        jaiListPush(out, item);
        jaiGCPopRoot();
    }
    jaiGCPopRoots(2);

    return vm.hasException ? NULL : out;
}

// The copy a sort hands back

ObjList *sortedCopy(ObjList *items, ObjList *keys, bool reverse,
                    const char *fnName) {
    int n = items->count;
    ObjList *result = jaiListNew(n);
    jaiGCPushRoot(OBJ_VAL(result));
    if (n <= 1) {
        for (int i = 0; i < n; i++) jaiListPush(result, jaiListGet(items, i));
        jaiGCPopRoot();
        return result;
    }

    int *idx = JAI_ALLOC(int, n);
    int *scratch = JAI_ALLOC(int, n);
    for (int i = 0; i < n; i++) idx[i] = i;

    bool ok = jaiSeqSortIndices(keys, idx, scratch, n, reverse, fnName);
    if (ok) {
        for (int i = 0; i < n; i++) jaiListPush(result, jaiListGet(items, idx[i]));
    }

    JAI_FREE_ARRAY(int, idx, n);
    JAI_FREE_ARRAY(int, scratch, n);
    jaiGCPopRoot();
    return ok ? result : NULL;
}

// Core builtins

static bool nPrint(int argc, Value *args, Value *out) {
    for (int i = 0; i < argc; i++) {
        ObjString *text = jaiValueToStr(args[i]);
        if (text == NULL) return false;
        if (i > 0) fputc(' ', stdout);
        (void)fwrite(text->chars, 1, text->length, stdout);
    }
    fputc('\n', stdout);
    *out = NULL_VAL;
    return true;
}

static bool nInput(int argc, Value *args, Value *out) {
    if (argc >= 1 && !IS_NULL(args[0])) {
        ObjString *prompt = jaiValueToStr(args[0]);
        if (prompt == NULL) return false;
        (void)fwrite(prompt->chars, 1, prompt->length, stdout);
    }
    fflush(stdout);

    JaiBuf line;
    jaiBufInit(&line);
    int c;
    bool sawAny = false;
    while ((c = fgetc(stdin)) != EOF) {
        sawAny = true;
        if (c == '\n') break;
        jaiBufPush(&line, (uint8_t)c);
    }
    if (!sawAny) {
        jaiBufFree(&line);
        return jaiThrow(vm.cIOError, "input(): end of input");
    }
    if (line.count > 0 && line.data[line.count - 1] == '\r') line.count--;

    ObjString *text = jaiStringNew(line.data != NULL ? (const char *)line.data : "",
                                   line.count);
    jaiBufFree(&line);
    if (text == NULL) return false;
    *out = OBJ_VAL(text);
    return true;
}

static bool nLen(int argc, Value *args, Value *out) {
    (void)argc;
    int64_t length;
    if (!valueLength(args[0], &length)) {
        if (vm.hasException) return false;
        return jaiThrow(vm.cTypeError, "object of type '%s' has no len()",
                        jaiTypeNameStatic(args[0]));
    }
    *out = INT_VAL(length);
    return true;
}

static bool nRange(int argc, Value *args, Value *out) {
    int64_t start = 0, stop = 0, step = 1;
    if (argc == 1) {
        if (!jaiArgInt(args[0], 1, "range", &stop)) return false;
    } else {
        if (!jaiArgInt(args[0], 1, "range", &start)) return false;
        if (!jaiArgInt(args[1], 2, "range", &stop)) return false;
        if (argc >= 3 && !jaiArgInt(args[2], 3, "range", &step)) return false;
    }
    if (step == 0) return jaiThrow(vm.cValueError, "range() step must not be zero");
    *out = OBJ_VAL(jaiRangeNew(start, stop, step, false));
    return true;
}

static bool nType(int argc, Value *args, Value *out) {
    (void)argc;
    *out = OBJ_VAL(jaiTypeName(args[0]));
    return true;
}

static bool nRepr(int argc, Value *args, Value *out) {
    (void)argc;
    ObjString *text = jaiValueToRepr(args[0]);
    if (text == NULL) return false;
    *out = OBJ_VAL(text);
    return true;
}

static bool nStr(int argc, Value *args, Value *out) {
    (void)argc;
    ObjString *text = jaiValueToStr(args[0]);
    if (text == NULL) return false;
    *out = OBJ_VAL(text);
    return true;
}

static bool nId(int argc, Value *args, Value *out) {
    (void)argc;
    Value v = args[0];
    switch (jaiValueType(v)) {
    case VAL_NULL:  *out = INT_VAL(0); return true;
    case VAL_BOOL:  *out = INT_VAL(AS_BOOL(v) ? 1 : 2); return true;
    case VAL_INT:   *out = INT_VAL(AS_INT(v)); return true;
    case VAL_FLOAT: {
        double d = AS_FLOAT(v);
        uint64_t bits;
        memcpy(&bits, &d, sizeof bits);
        *out = INT_VAL((int64_t)bits);
        return true;
    }
    case VAL_OBJ: {
        Obj *o = AS_OBJ(v);
        uint64_t address = 0;
        memcpy(&address, &o, sizeof o < sizeof address ? sizeof o : sizeof address);
        *out = INT_VAL((int64_t)address);
        return true;
    }
    }
    *out = INT_VAL(0);
    return true;
}

static bool nHash(int argc, Value *args, Value *out) {
    (void)argc;
    bool ok = true;
    uint64_t h = jaiValueHash(args[0], &ok);
    if (!ok) {
        if (vm.hasException) return false;
        return jaiThrow(vm.cTypeError, "unhashable type: '%s'",
                        jaiTypeNameStatic(args[0]));
    }
    *out = INT_VAL((int64_t)h);
    return true;
}

static bool nAbs(int argc, Value *args, Value *out) {
    (void)argc;
    Value v = args[0];
    if (IS_INT(v)) {
        int64_t n = AS_INT(v);
        if (n == INT64_MIN) return jaiBuiltinOverflowError("abs()");
        *out = INT_VAL(n < 0 ? -n : n);
        return true;
    }
    if (IS_FLOAT(v)) {
        *out = FLOAT_VAL(fabs(AS_FLOAT(v)));
        return true;
    }
    return jaiBuiltinArgTypeError(1, "abs", "int or float", v);
}

static bool nChr(int argc, Value *args, Value *out) {
    (void)argc;
    int64_t cp;
    if (!jaiArgInt(args[0], 1, "chr", &cp)) return false;
    if (cp < 0 || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF))
        return jaiThrow(vm.cValueError, "chr() argument out of range: %lld",
                        (long long)cp);

    char buf[4];
    int n = jaiUtf8Encode((int32_t)cp, buf);
    if (n == 0)
        return jaiThrow(vm.cValueError, "chr() argument is not a scalar value: %lld",
                        (long long)cp);
    *out = OBJ_VAL(jaiStringIntern(buf, (size_t)n));
    return true;
}

static bool nOrd(int argc, Value *args, Value *out) {
    (void)argc;
    ObjString *s;
    if (!jaiArgString(args[0], 1, "ord", &s)) return false;
    if (s->length == 0)
        return jaiThrow(vm.cValueError, "ord() expected one character, got \"\"");

    int len = 1;
    int32_t cp = jaiUtf8Decode(s->chars, s->chars + s->length, &len);
    if (cp < 0) return jaiThrow(vm.cValueError, "ord(): invalid UTF-8");
    if ((uint32_t)len != s->length)
        return jaiThrow(vm.cValueError,
                        "ord() expected one character, got a string of length %u",
                        jaiStringScalarCount(s));
    *out = INT_VAL(cp);
    return true;
}

bool jaiBuiltinMatchesType(Value v, Value t, bool *matched) {
    if (IS_CLASS(t)) {
        *matched = IS_INSTANCE(v) && jaiClassIsSubclassOf(AS_INSTANCE(v)->klass, AS_CLASS(t));
        return true;
    }
    if (IS_TRAIT(t)) {
        *matched = IS_INSTANCE(v) && jaiClassImplements(AS_INSTANCE(v)->klass, AS_TRAIT(t));
        return true;
    }
    if (IS_ENUM(t)) {
        *matched = IS_ENUM_VAL(v) && AS_ENUM_VAL(v)->type == AS_ENUM(t);
        return true;
    }
    if (IS_NATIVE(t) || IS_STRING(t)) {
        const char *want = IS_NATIVE(t) ? AS_NATIVE(t)->name->chars : AS_CSTRING(t);
        *matched = strcmp(jaiTypeNameStatic(v), want) == 0;
        if (!*matched && strcmp(want, "any") == 0) *matched = true;
        return true;
    }
    if (IS_LIST(t) || IS_TUPLE(t)) {
        int count = IS_LIST(t) ? AS_LIST(t)->count : (int)AS_TUPLE(t)->count;
        const Value *entries = IS_LIST(t) ? jaiListBox(AS_LIST(t))
                                          : AS_TUPLE(t)->items;
        for (int i = 0; i < count; i++) {
            if (!jaiBuiltinMatchesType(v, entries[i], matched)) return false;
            if (*matched) return true;
        }
        *matched = false;
        return true;
    }
    return jaiThrow(vm.cTypeError,
                    "isinstance() argument 2 must be a class, trait or type name, got %s",
                    jaiTypeNameStatic(t));
}

static bool nIsInstance(int argc, Value *args, Value *out) {
    (void)argc;
    bool matched = false;
    if (!jaiBuiltinMatchesType(args[0], args[1], &matched)) return false;
    *out = BOOL_VAL(matched);
    return true;
}

static bool nCallable(int argc, Value *args, Value *out) {
    (void)argc;
    *out = BOOL_VAL(jaiBuiltinIsCallable(args[0]));
    return true;
}

static void collectTableKeys(const JaiTable *table, ObjList *into) {
    int slot = 0;
    Value key, ignored;
    while (jaiTableNext(table, &slot, &key, &ignored)) {
        if (!IS_STRING(key)) continue;
        jaiGCPushRoot(key);
        jaiListPush(into, key);
        jaiGCPopRoot();
    }
}

static ObjList *memberNames(Value v) {
    if (IS_MODULE(v)) {
        ObjList *names = jaiListNew(0);
        jaiGCPushRoot(OBJ_VAL(names));
        collectTableKeys(&AS_MODULE(v)->globals, names);
        jaiGCPopRoot();
        return names;
    }
    if (IS_CLASS(v)) {
        ObjClass *k = AS_CLASS(v);
        ObjList *names = jaiListNew(0);
        jaiGCPushRoot(OBJ_VAL(names));
        collectTableKeys(&k->methods, names);
        collectTableKeys(&k->statics, names);
        collectTableKeys(&k->getters, names);
        collectTableKeys(&k->setters, names);
        jaiGCPopRoot();
        return names;
    }
    if (IS_INSTANCE(v)) {
        ObjClass *k = AS_INSTANCE(v)->klass;
        ObjList *names = jaiListNew(0);
        if (k == NULL) return names;
        jaiGCPushRoot(OBJ_VAL(names));
        collectTableKeys(&k->methods, names);
        collectTableKeys(&k->getters, names);
        collectTableKeys(&k->setters, names);
        for (uint16_t i = 0; i < k->fieldCount; i++) {
            ObjString *fieldName = k->fields[i].name;
            if (fieldName == NULL) continue;
            jaiGCPushRoot(OBJ_VAL(fieldName));
            jaiListPush(names, OBJ_VAL(fieldName));
            jaiGCPopRoot();
        }
        jaiGCPopRoot();
        return names;
    }
    if (IS_ENUM(v)) {
        ObjEnum *e = AS_ENUM(v);
        ObjList *names = jaiListNew(0);
        jaiGCPushRoot(OBJ_VAL(names));
        for (uint16_t i = 0; i < e->variantCount; i++) {
            ObjString *variant = e->variants[i].name;
            if (variant == NULL) continue;
            jaiGCPushRoot(OBJ_VAL(variant));
            jaiListPush(names, OBJ_VAL(variant));
            jaiGCPopRoot();
        }
        collectTableKeys(&e->methods, names);
        jaiGCPopRoot();
        return names;
    }
    return jaiBuiltinMethodNames(v);
}

static bool nDir(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *names = memberNames(args[0]);
    if (names == NULL || vm.hasException) return false;
    jaiGCPushRoot(OBJ_VAL(names));

    ObjList *sorted = sortedCopy(names, names, false, "dir");
    jaiGCPopRoot();
    if (sorted == NULL) return false;

    jaiGCPushRoot(OBJ_VAL(sorted));
    ObjList *unique = jaiListNew(sorted->count);
    jaiGCPushRoot(OBJ_VAL(unique));
    for (int i = 0; i < sorted->count; i++) {
        if (i > 0 && IS_STRING(jaiListGet(sorted, i)) && IS_STRING(jaiListGet(sorted, i - 1)) &&
            jaiStringEquals(AS_STRING(jaiListGet(sorted, i)), AS_STRING(jaiListGet(sorted, i - 1))))
            continue;
        jaiListPush(unique, jaiListGet(sorted, i));
    }
    jaiGCPopRoots(2);

    *out = OBJ_VAL(unique);
    return true;
}

static bool nAssertEq(int argc, Value *args, Value *out) {
    bool equal = jaiValuesEqual(args[0], args[1]);
    if (vm.hasException) return false;
    if (equal) {
        *out = NULL_VAL;
        return true;
    }

    ObjString *actual = jaiValueToRepr(args[0]);
    if (actual == NULL) return false;
    jaiGCPushRoot(OBJ_VAL(actual));

    ObjString *expected = jaiValueToRepr(args[1]);
    if (expected == NULL) { jaiGCPopRoot(); return false; }
    jaiGCPushRoot(OBJ_VAL(expected));

    ObjString *message = NULL;
    if (argc >= 3 && !IS_NULL(args[2])) {
        message = jaiValueToStr(args[2]);
        if (message == NULL) { jaiGCPopRoots(2); return false; }
        jaiGCPushRoot(OBJ_VAL(message));
    }

    if (message != NULL) {
        (void)jaiThrow(vm.cAssertionError, "%.*s: expected %.*s, got %.*s",
                       (int)message->length, message->chars,
                       (int)expected->length, expected->chars,
                       (int)actual->length, actual->chars);
    } else {
        (void)jaiThrow(vm.cAssertionError, "assert_eq failed: expected %.*s, got %.*s",
                       (int)expected->length, expected->chars,
                       (int)actual->length, actual->chars);
    }
    jaiGCPopRoots(message != NULL ? 3 : 2);
    return false;
}

static bool nExit(int argc, Value *args, Value *out) {
    (void)out;
    int64_t code = 0;
    if (argc >= 1 && !IS_NULL(args[0]) && !jaiArgInt(args[0], 1, "exit", &code))
        return false;
    fflush(stdout);
    fflush(stderr);
    exit((int)(code & 0xFF));
}

/* ------------------------------------------------------------------ */
/* ------------------------------------------------------------------ */
/* ------------------------------------------------------------------ */
/* Registration                                                         */
/* ------------------------------------------------------------------ */

void jaiRegisterCoreBuiltins(void) {
    jaiDefineNative("print",      nPrint,      0, -1);
    jaiDefineNative("input",      nInput,      0,  1);
    jaiDefineNative("len",        nLen,        1,  1);
    jaiDefineNative("range",      nRange,      1,  3);
    jaiDefineNative("type_of",    nType,       1,  1);
    jaiDefineNative("repr",       nRepr,       1,  1);
    jaiDefineNative("str",        nStr,        1,  1);
    jaiDefineNative("int",        nIntConv,    1,  2);
    jaiDefineNative("float",      nFloatConv,  1,  1);
    jaiDefineNative("bool",       nBoolConv,   1,  1);
    jaiDefineNative("id",         nId,         1,  1);
    jaiDefineNative("hash",       nHash,       1,  1);
    jaiDefineNative("abs",        nAbs,        1,  1);
    jaiDefineNative("min",        nMin,        1, -1);
    jaiDefineNative("max",        nMax,        1, -1);
    jaiDefineNative("sum",        nSum,        1,  2);
    jaiDefineNative("sorted",     nSorted,     1,  3);
    jaiDefineNative("reversed",   nReversed,   1,  1);
    jaiDefineNative("enumerate",  nEnumerate,  1,  2);
    jaiDefineNative("zip",        nZip,        1, -1);
    jaiDefineNative("map",        nMap,        2, -1);
    jaiDefineNative("filter",     nFilter,     2,  2);
    jaiDefineNative("any",        nAny,        1,  1);
    jaiDefineNative("all",        nAll,        1,  1);
    jaiDefineNative("chr",        nChr,        1,  1);
    jaiDefineNative("ord",        nOrd,        1,  1);
    jaiDefineNative("isinstance", nIsInstance, 2,  2);

    jaiDefineNative("__prim__.is_instance", nIsInstance, 2, 2);

    jaiDefineNative("callable",   nCallable,   1,  1);
    jaiDefineNative("dir",        nDir,        1,  1);
    jaiDefineNative("assert_eq",  nAssertEq,   2,  3);
    jaiDefineNative("exit",       nExit,       0,  1);

    jaiDefineNative("list",  nListConv,  0, 1);
    jaiDefineNative("dict",  nDictConv,  0, 1);
    jaiDefineNative("set",   nSetConv,   0, 1);
    jaiDefineNative("tuple", nTupleConv, 0, 1);
    jaiDefineNative("bytes", nBytesConv, 0, 1);

    jaiRegisterOperatorPrimitives();
}
