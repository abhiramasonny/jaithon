/* object.c — the heap object model's shared core: allocation, the free-list
 * dispatch every object kind's destructor funnels through, and the
 * human-readable type name used in error messages.
 *
 * The other pieces of what used to be one 2200-line file now live one kind
 * (or a small family of related kinds) per sibling source file:
 *
 *   object_string.c       ObjString, ObjStrBuf — allocation, interning
 *                         policy, concatenation, Python-semantics slicing
 *   object_collection.c   ObjBytes, ObjList, ObjTuple, ObjDict, ObjSet,
 *                         ObjRange — the pure-data containers
 *   object_function.c     ObjFunction, ObjUpvalue, ObjClosure, ObjNative,
 *                         ObjBound — the callable kinds
 *   object_class.c        ObjClass, ObjTrait, ObjInstance, ObjEnum,
 *                         ObjEnumVal, ObjEnumCtor — the nominal type system
 *   object_iter.c         ObjIter — iteration over every built-in kind, plus
 *                         both user-iterator protocols
 *   object_module.c       ObjModule — a loaded source file's namespace
 *   object_file.c         ObjFile — an open OS file handle
 *
 * object_internal.h holds the two pieces of plumbing more than one of those
 * files needs but that have no business in the public object.h: pushObjRoot
 * (root-then-allocate for a constructor whose argument may be NULL) and
 * sliceCount (the [start,stop)/step clamp shared by jaiStringSlice and
 * jaiListSlice).
 *
 * What stayed here did so because it is the one part of the object model
 * that is not really about any single kind: allocObj is the allocation path
 * every kind's constructor funnels through regardless of which file that
 * constructor now lives in, and jaiFreeObject/jaiObjTypeName are exhaustive
 * switches over ObjType that have to see every kind at once to be checked
 * for completeness at all — splitting either of those by kind would mean
 * splitting one function's `switch`, which helps no one reading it.
 *
 * Nothing here interprets bytecode; the only call back into the VM is
 * jaiThrow for runtime errors.
 */

#include "vm/object/object.h"

#include "vm/gc.h"
#include "vm/table.h"
#include "vm/vm.h"

/* ------------------------------------------------------------------ */
/* Allocation and lifetime                                              */
/* ------------------------------------------------------------------ */

static inline Obj *allocObj(size_t size, ObjType type) {
    /* Collect *before* the allocation: afterwards the new object is not yet
     * reachable from any root and a collection would free it. Guarding this
     * with the inlined jaiGCWanted() test used to measure a wash, because the
     * allocation dwarfed the call; once small objects came from a free list
     * that stopped being true. */
    if (JAI_UNLIKELY(jaiGCWanted())) jaiGCMaybeCollect();

    /* jaiSmallNew is the bins-and-slab half of jaiRealloc with the size class
     * already known, so this is the same block from the same place minus the
     * call and five branches sorting resize/free/unserved classes. Measured
     * 10% of alloc_churn for two calls per object. */
    Obj *obj = (Obj *)(JAI_LIKELY(jaiSmallServes(size)) ? jaiSmallNew(size)
                                                        : jaiRealloc(NULL, 0, size));
    obj->type = type;
    obj->isMarked = false;
    obj->next = NULL;
    jaiGCTrackObject(obj);
    vm.allocCount++;
    return obj;
}

/* Hands back `size` bytes with only the Obj header set and everything after it
 * uninitialised. Worth it only where the payload is about to be overwritten
 * wholesale, which means the variable-length types: zeroing a string's
 * characters and then memcpying over them turned the memset into a call into
 * libc with a runtime length, and that was 11% of tests/bench/string_build. */
Obj *jaiAllocateObjectRaw(size_t size, ObjType type) {
    return allocObj(size, type);
}

Obj *jaiAllocateObject(size_t size, ObjType type) {
    Obj *obj = allocObj(size, type);
    /* Everything past the header starts zeroed, so a constructor that does not
     * mention a field leaves it NULL or 0 rather than garbage. */
    memset((char *)obj + sizeof(Obj), 0, size - sizeof(Obj));
    return obj;
}

void jaiFreeObject(Obj *obj) {
    if (obj == NULL) return;

    switch (obj->type) {
    case OBJ_STRING: {
        /* Flexible array: header and payload are one block. The intern table
         * holds only weak references and is purged by the sweep, so there is
         * nothing to unlink here. */
        ObjString *s = (ObjString *)obj;
        /* A slice carries no bytes of its own; the buffer is swept separately
         * once nothing views it. */
        (void)jaiRealloc(obj, s->owner != NULL ? sizeof(ObjString)
                                               : JAI_STRING_ALLOC(s->length), 0);
        return;
    }
    case OBJ_STRBUF: {
        ObjStrBuf *b = (ObjStrBuf *)obj;
        (void)jaiRealloc(obj, sizeof(ObjStrBuf) + b->capacity + 1, 0);
        return;
    }
    case OBJ_BYTES: {
        ObjBytes *b = (ObjBytes *)obj;
        (void)jaiRealloc(obj, sizeof(ObjBytes) + b->length, 0);
        return;
    }
    case OBJ_TUPLE: {
        ObjTuple *t = (ObjTuple *)obj;
        (void)jaiRealloc(obj, sizeof(ObjTuple) + sizeof(Value) * (size_t)t->count, 0);
        return;
    }
    case OBJ_INSTANCE: {
        ObjInstance *inst = (ObjInstance *)obj;
        (void)jaiRealloc(obj, sizeof(ObjInstance) + sizeof(Value) * (size_t)inst->fieldCount, 0);
        return;
    }
    case OBJ_ENUM_VAL: {
        ObjEnumVal *ev = (ObjEnumVal *)obj;
        (void)jaiRealloc(obj, sizeof(ObjEnumVal) + sizeof(Value) * (size_t)ev->count, 0);
        return;
    }
    case OBJ_LIST: {
        ObjList *l = (ObjList *)obj;
        JAI_FREE_ARRAY(Value, l->items, l->capacity);
        JAI_FREE(ObjList, obj);
        return;
    }
    case OBJ_DICT: {
        ObjDict *d = (ObjDict *)obj;
        jaiTableFree(&d->table);
        JAI_FREE(ObjDict, obj);
        return;
    }
    case OBJ_SET: {
        ObjSet *s = (ObjSet *)obj;
        jaiTableFree(&s->table);
        JAI_FREE(ObjSet, obj);
        return;
    }
    case OBJ_RANGE:
        JAI_FREE(ObjRange, obj);
        return;
    case OBJ_FUNCTION: {
        ObjFunction *fn = (ObjFunction *)obj;
        jaiChunkFree(&fn->chunk);
        JAI_FREE_ARRAY(ObjString *, fn->paramNames, fn->paramCount);
        JAI_FREE_ARRAY(ExceptionEntry, fn->exceptions, fn->exceptionCount);
        JAI_FREE_ARRAY(uint32_t, fn->defaultOffsets, fn->defaultCount);
        JAI_FREE(ObjFunction, obj);
        return;
    }
    case OBJ_CLOSURE: {
        ObjClosure *c = (ObjClosure *)obj;
        JAI_FREE_ARRAY(ObjUpvalue *, c->upvalues, c->upvalueCount);
        JAI_FREE(ObjClosure, obj);
        return;
    }
    case OBJ_UPVALUE:
        JAI_FREE(ObjUpvalue, obj);
        return;
    case OBJ_NATIVE:
        JAI_FREE(ObjNative, obj);
        return;
    case OBJ_BOUND:
        JAI_FREE(ObjBound, obj);
        return;
    case OBJ_CLASS: {
        ObjClass *c = (ObjClass *)obj;
        JAI_FREE_ARRAY(FieldInfo, c->fields, c->fieldCount);
        JAI_FREE_ARRAY(ObjTrait *, c->traits, c->traitCount);
        jaiTableFree(&c->methods);
        jaiTableFree(&c->statics);
        jaiTableFree(&c->getters);
        jaiTableFree(&c->setters);
        jaiTableFree(&c->restricted);
        JAI_FREE(ObjClass, obj);
        return;
    }
    case OBJ_TRAIT: {
        ObjTrait *t = (ObjTrait *)obj;
        jaiTableFree(&t->required);
        jaiTableFree(&t->defaults);
        JAI_FREE_ARRAY(ObjTrait *, t->supers, t->superCount);
        JAI_FREE(ObjTrait, obj);
        return;
    }
    case OBJ_MODULE: {
        ObjModule *m = (ObjModule *)obj;
        jaiTableFree(&m->globals);
        jaiTableFree(&m->exports);
        JAI_FREE(ObjModule, obj);
        return;
    }
    case OBJ_ENUM: {
        ObjEnum *e = (ObjEnum *)obj;
        for (uint16_t i = 0; i < e->variantCount; i++) {
            JAI_FREE_ARRAY(ObjString *, e->variants[i].fieldNames,
                           e->variants[i].arity);
        }
        JAI_FREE_ARRAY(EnumVariant, e->variants, e->variantCount);
        jaiTableFree(&e->methods);
        JAI_FREE(ObjEnum, obj);
        return;
    }
    case OBJ_ITER:
        JAI_FREE(ObjIter, obj);
        return;
    case OBJ_FILE: {
        ObjFile *f = (ObjFile *)obj;
        /* A file reaching the collector without being closed is closed here;
         * the standard streams are never ours to close. */
        if (f->handle != NULL && !f->closed && f->handle != stdin &&
            f->handle != stdout && f->handle != stderr) {
            (void)fclose(f->handle);
        }
        f->handle = NULL;
        f->closed = true;
        JAI_FREE(ObjFile, obj);
        return;
    }
    case OBJ_ENUM_CTOR:
        JAI_FREE(ObjEnumCtor, obj);
        return;

    case OBJ_TYPE_COUNT:
        break;   /* not a real tag; fall through to the panic below */
    }

    /* Every real tag returns from its own case. Reaching here means obj->type
     * is corrupt, and there is no size with which to free it correctly. */
    JAI_PANIC("jaiFreeObject: object with invalid type tag %d", (int)obj->type);
}

const char *jaiObjTypeName(ObjType t) {
    switch (t) {
    case OBJ_STRING:    return "str";
    case OBJ_STRBUF:    return "str";   /* never user-visible */
    case OBJ_BYTES:     return "bytes";
    case OBJ_LIST:      return "list";
    case OBJ_DICT:      return "dict";
    case OBJ_SET:       return "set";
    case OBJ_TUPLE:     return "tuple";
    case OBJ_RANGE:     return "range";
    case OBJ_FUNCTION:  return "function";
    case OBJ_CLOSURE:   return "function";
    case OBJ_UPVALUE:   return "upvalue";
    case OBJ_NATIVE:    return "native function";
    case OBJ_BOUND:     return "method";
    case OBJ_CLASS:     return "class";
    case OBJ_TRAIT:     return "trait";
    case OBJ_INSTANCE:  return "instance";
    case OBJ_MODULE:    return "module";
    case OBJ_ENUM:      return "enum";
    case OBJ_ENUM_VAL:  return "enum value";
    case OBJ_ENUM_CTOR: return "function";
    case OBJ_ITER:      return "iterator";
    case OBJ_FILE:      return "file";
    case OBJ_TYPE_COUNT: break;
    }
    return "object";
}

