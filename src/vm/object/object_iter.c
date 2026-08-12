/* object_iter.c — ObjIter: an iterator over any of the built-in sequence
 * kinds, plus the two protocols by which a user-defined class supplies its
 * own iteration (spec §7.1's `__next__`/StopIteration and std.core's `trait
 * Iterator`/`next`-returns-null).
 *
 * This is the one file in the split with real control flow rather than plain
 * data-shuffling: jaiGetIter has to decide, per source value, which of five
 * built-in kinds it is or which of two user protocols its class implements,
 * and jaiIterNext's switch is the resulting per-kind advance step. It is also
 * the only file that calls back into the interpreter proper (jaiInvokeMethod,
 * for a user `__iter__`/`__next__`/`next`) rather than just jaiThrow — see
 * object.c's file comment for why that line is drawn here.
 *
 * The ITER_STRING case in jaiIterNext reads the ASCII fast-path table that
 * object_string.c owns, through the `static inline` jaiAsciiCharTable()
 * accessor in object.h — see that file's header comment for why this is the
 * one field shared directly, rather than through a normal function call,
 * between object_*.c files.
 */

#include "vm/object/object.h"

#include "vm/gc.h"
#include "vm/table.h"
#include "vm/vm.h"

/* ------------------------------------------------------------------ */
/* Iterators                                                            */
/* ------------------------------------------------------------------ */

/* Interned dunder names are cached on the VM at startup; fall back to
 * interning on demand so object.c works before jaiVMInit has run. */
static inline ObjString *iterName(void) {
    return vm.strIter != NULL ? vm.strIter : jaiStringInternC("__iter__");
}

static inline ObjString *nextName(void) {
    return vm.strNext != NULL ? vm.strNext : jaiStringInternC("__next__");
}

/* The trait spellings of the same two methods (std.core `Iterable`/`Iterator`).
 * They are ordinary names, so there is no cached intern for them. */
static inline ObjString *traitIterName(void) {
    return jaiStringInternC("iter");
}

static inline ObjString *traitNextName(void) {
    return jaiStringInternC("next");
}

/* True when instances of `v`'s class answer `name`. Inherited methods are
 * copied down at class creation, so one table lookup is the whole answer. */
ObjIter *jaiIterNew(IterKind kind, Value source) {
    const bool rootSource = IS_OBJ(source);
    if (rootSource) jaiGCPushRoot(source);

    ObjIter *it = JAI_ALLOCATE_OBJ(ObjIter, OBJ_ITER);

    if (rootSource) jaiGCPopRoot();

    it->kind = kind;
    it->source = source;

    switch (kind) {
        case ITER_LIST:
            if (IS_LIST(source)) {
                ObjList *const list = AS_LIST(source);
                it->limit = list->count;
                it->version = list->version;
            }
            break;

        case ITER_TUPLE:
            if (IS_TUPLE(source))
                it->limit = (int64_t)AS_TUPLE(source)->count;
            break;

        case ITER_STRING:
            if (IS_STRING(source))
                it->limit = (int64_t)AS_STRING(source)->length;
            break;

        case ITER_DICT_KEYS:
        case ITER_DICT_ITEMS:
            if (IS_DICT(source)) {
                JaiTable *const table = &AS_DICT(source)->table;
                it->limit = table->count;
                it->version = table->version;
            }
            break;

        case ITER_SET:
            if (IS_SET(source)) {
                JaiTable *const table = &AS_SET(source)->table;
                it->limit = table->count;
                it->version = table->version;
            }
            break;

        case ITER_RANGE:
            if (IS_RANGE(source))
                it->limit = jaiRangeLength(AS_RANGE(source));
            break;

        case ITER_USER:
        case ITER_TRAIT:
        case ITER_GENERATOR:
            break;
    }

    return it;
}

/* Both forms are fatal to the traversal, but they are reported apart because
 * the reader's next question differs: a resize invalidates the iterator's
 * bounds, while an in-place store leaves them valid and silently changes what
 * the loop sees. */
static inline bool iterMutated(bool resized) {
    return jaiThrow(vm.cRuntimeError,
                    resized ? "container changed size during iteration"
                            : "container was modified during iteration");
}

/* True when the pending exception is a StopIteration, i.e. an ordinary end of
 * a user-defined iterator rather than a failure. */
static inline bool pendingIsStopIteration(void) {
    if (!vm.hasException || vm.cStopIteration == NULL)
        return false;

    const Value exception = vm.pendingException;

    if (IS_INSTANCE(exception))
        return jaiClassIsSubclassOf(AS_INSTANCE(exception)->klass,
                                    vm.cStopIteration);

    if (IS_CLASS(exception))
        return jaiClassIsSubclassOf(AS_CLASS(exception), vm.cStopIteration);

    return false;
}

/* Two exhaustion protocols meet here. `__next__` (spec §7.1) ends by raising
 * StopIteration; `trait Iterator.next` (spec §9), which the whole standard
 * library is written against, ends by returning null. Which one applies is
 * decided by which method the object actually has, so neither has to know
 * about the other. */
static bool iterUserNext(ObjIter *it, Value *out) {
    Value result;
    if (!jaiInvokeMethod(it->source, nextName(), 0, NULL, &result)) {
        if (pendingIsStopIteration()) {
            jaiClearException();
            return false;
        }
        if (!vm.hasException) {
            jaiThrow(vm.cTypeError, "'%s' object has no __next__ method",
                     jaiTypeNameStatic(it->source));
        }
        return false;
    }
    *out = result;
    return true;
}

/* std.core's contract: `next` returns null at the end and must keep returning
 * null afterwards. An iterator over values that can themselves be null wraps
 * them, which is why null is safe to read as "done" here. */
static bool iterTraitNext(ObjIter *it, Value *out) {
    Value result;
    if (!jaiInvokeMethod(it->source, traitNextName(), 0, NULL, &result)) {
        if (pendingIsStopIteration()) {
            jaiClearException();
            return false;
        }
        if (!vm.hasException) {
            jaiThrow(vm.cTypeError, "'%s' object has no `next` method",
                     jaiTypeNameStatic(it->source));
        }
        return false;
    }
    if (IS_NULL(result)) return false;
    *out = result;
    return true;
}

bool jaiIterNext(ObjIter *it, Value *out) {
    switch (it->kind) {
        case ITER_LIST: {
            ObjList *const list = AS_LIST(it->source);
            if (JAI_UNLIKELY(list->version != it->version))
                return iterMutated((int64_t)list->count != it->limit);

            const int64_t index = it->index;
            if (index >= it->limit) return false;

            *out = list->items[index];
            it->index = index + 1;
            return true;
        }

        case ITER_TUPLE: {
            const int64_t index = it->index;
            if (index >= it->limit) return false;

            *out = AS_TUPLE(it->source)->items[index];
            it->index = index + 1;
            return true;
        }

        case ITER_STRING: {
            ObjString *const string = AS_STRING(it->source);
            const int64_t index = it->index;
            if (index >= it->limit) return false;

            const char *const p = string->chars + index;
            const unsigned char first = (unsigned char)*p;

            if (first < 0x80u) {
                ObjString *scalar = jaiAsciiCharTable()[first];
                if (scalar == NULL) {
                    jaiGCPushRoot(OBJ_VAL(it));
                    scalar = jaiStringChar(first);
                    jaiGCPopRoot();
                    if (scalar == NULL) return false;
                }

                it->index = index + 1;
                *out = OBJ_VAL(scalar);
                return true;
            }

            int len = 1;
            (void)jaiUtf8Decode(p, string->chars + string->length, &len);
            if (index + len > it->limit)
                len = (int)(it->limit - index);

            jaiGCPushRoot(OBJ_VAL(it));
            ObjString *scalar = jaiStringNew(p, (size_t)len);
            jaiGCPopRoot();
            if (scalar == NULL) return false;

            it->index = index + len;
            *out = OBJ_VAL(scalar);
            return true;
        }

        case ITER_DICT_KEYS:
        case ITER_DICT_ITEMS: {
            ObjDict *const dict = AS_DICT(it->source);
            JaiTable *const table = &dict->table;

            if (JAI_UNLIKELY(table->version != it->version))
                return iterMutated((int64_t)table->count != it->limit);

            int slot = (int)it->index;
            Value key, value;
            if (!jaiTableNext(table, &slot, &key, &value)) {
                it->index = slot;
                return false;
            }
            it->index = slot;

            if (it->kind == ITER_DICT_KEYS) {
                *out = key;
                return true;
            }

            Value pair[2] = {key, value};
            jaiGCPushRoot(OBJ_VAL(it));
            ObjTuple *tuple = jaiTupleNew(pair, 2);
            jaiGCPopRoot();
            *out = OBJ_VAL(tuple);
            return true;
        }

        case ITER_SET: {
            ObjSet *const set = AS_SET(it->source);
            JaiTable *const table = &set->table;

            if (JAI_UNLIKELY(table->version != it->version))
                return iterMutated((int64_t)table->count != it->limit);

            int slot = (int)it->index;
            Value key, ignored;
            if (!jaiTableNext(table, &slot, &key, &ignored)) {
                it->index = slot;
                return false;
            }

            it->index = slot;
            *out = key;
            return true;
        }

        case ITER_RANGE: {
            const int64_t index = it->index;
            if (index >= it->limit) return false;

            ObjRange *const range = AS_RANGE(it->source);
            uint64_t value;

            if (range->step == 1) {
                value = (uint64_t)range->start + (uint64_t)index;
            } else if (range->step == -1) {
                value = (uint64_t)range->start - (uint64_t)index;
            } else {
                value = (uint64_t)range->start +
                        (uint64_t)index * (uint64_t)range->step;
            }

            *out = INT_VAL((int64_t)value);
            it->index = index + 1;
            return true;
        }

        case ITER_USER:
        case ITER_GENERATOR:
            return iterUserNext(it, out);

        case ITER_TRAIT:
            return iterTraitNext(it, out);
    }

    return false;
}

bool jaiGetIter(Value v, Value *out) {
    if (IS_OBJ(v)) {
        switch (OBJ_TYPE(v)) {
            case OBJ_ITER:
                *out = v;
                return true;

            case OBJ_LIST:
                *out = OBJ_VAL(jaiIterNew(ITER_LIST, v));
                return true;

            case OBJ_TUPLE:
                *out = OBJ_VAL(jaiIterNew(ITER_TUPLE, v));
                return true;

            case OBJ_STRING:
                *out = OBJ_VAL(jaiIterNew(ITER_STRING, v));
                return true;

            case OBJ_DICT:
                *out = OBJ_VAL(jaiIterNew(ITER_DICT_KEYS, v));
                return true;

            case OBJ_SET:
                *out = OBJ_VAL(jaiIterNew(ITER_SET, v));
                return true;

            case OBJ_RANGE:
                *out = OBJ_VAL(jaiIterNew(ITER_RANGE, v));
                return true;

            case OBJ_INSTANCE: {
                ObjInstance *const instance = AS_INSTANCE(v);
                ObjClass *const klass = instance->klass;

                const bool hasDunderIter =
                    klass != NULL && !IS_NULL(klass->dunderIter);
                ObjString *const method =
                    hasDunderIter ? iterName() : traitIterName();

                Value result;
                if (!jaiInvokeMethod(v, method, 0, NULL, &result)) {
                    if (!vm.hasException)
                        jaiThrow(vm.cTypeError,
                                 "'%s' object is not iterable",
                                 jaiTypeNameStatic(v));
                    return false;
                }

                if (IS_ITER(result)) {
                    *out = result;
                    return true;
                }

                if (IS_INSTANCE(result)) {
                    ObjInstance *const iterator = AS_INSTANCE(result);
                    ObjClass *const iteratorClass = iterator->klass;
                    const IterKind kind =
                        iteratorClass != NULL &&
                        !IS_NULL(iteratorClass->dunderNext)
                            ? ITER_USER
                            : ITER_TRAIT;

                    *out = OBJ_VAL(jaiIterNew(kind, result));
                    return true;
                }

                return jaiGetIter(result, out);
            }

            default:
                break;
        }
    }

    return jaiThrow(vm.cTypeError, "'%s' object is not iterable",
                    jaiTypeNameStatic(v));
}
