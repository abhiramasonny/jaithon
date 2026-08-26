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

#ifdef JAI_ALLOC_CENSUS
#  include <inttypes.h>
#endif

#include "vm/gc.h"
#include "vm/table.h"
#include "vm/vm.h"

/* ------------------------------------------------------------------ */
/* Allocation and lifetime                                              */
/* ------------------------------------------------------------------ */

/* The allocation census: how many objects of each kind a program made, and how
 * many bytes that was. Off unless the whole tree is built with it, because the
 * two increments belong on the hottest path there is:
 *
 *   make -j8 BUILD_ROOT=build-census TARGET=jaithon-census \
 *        EXTRA_CFLAGS=-DJAI_ALLOC_CENSUS
 *   ./jaithon-census --stats run prog.jai
 *
 * `--stats` already reported a single allocation total, which says a program
 * allocates a lot but never what. Splitting it by kind is what identified 4.5M
 * of nbody's 4.5M allocations as the ObjRange and ObjIter that `for i in 0..n`
 * builds per loop ENTRY -- a benchmark whose own header says it measures float
 * math. The GC half of the same picture is in jaiGCPrintStats. */
#ifdef JAI_ALLOC_CENSUS
uint64_t jaiAllocByType[OBJ_TYPE_COUNT];
uint64_t jaiAllocBytesByType[OBJ_TYPE_COUNT];

void jaiAllocPrintCensus(FILE *out) {
    if (out == NULL) return;
    for (int i = 0; i < OBJ_TYPE_COUNT; i++) {
        if (jaiAllocByType[i] == 0) continue;
        fprintf(out, "alloc %-14s %12" PRIu64 " %14" PRIu64 " bytes\n",
                jaiObjTypeName((ObjType)i), jaiAllocByType[i],
                jaiAllocBytesByType[i]);
    }
}
#endif

static inline Obj *allocObj(size_t size, ObjType type) {
#ifdef JAI_ALLOC_CENSUS
    jaiAllocByType[type]++;
    jaiAllocBytesByType[type] += size;
#endif
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

    /* Every kind whose whole footprint is its own block -- string, instance,
     * tuple, iterator, range and the rest -- is sized in one place, and the
     * sweep uses the same answer without coming through here at all. The
     * intern table holds only weak references and is purged by the sweep, so a
     * string has nothing to unlink either. */
    size_t sole = jaiObjSoleBlock(obj);
    if (sole != 0) {
        (void)jaiRealloc(obj, sole, 0);
        return;
    }

    switch (obj->type) {
    case OBJ_LIST: {
        ObjList *l = (ObjList *)obj;
        JAI_FREE_ARRAY(char, l->items,
                       (size_t)l->capacity * jaiListStoreWidth(l->stg));
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
    /* Freed above, off jaiObjSoleBlock's size. Named rather than defaulted so
     * that -Wswitch still forces a new kind to be classified in both places. */
    case OBJ_STRING:
    case OBJ_STRBUF:
    case OBJ_BYTES:
    case OBJ_TUPLE:
    case OBJ_INSTANCE:
    case OBJ_ENUM_VAL:
    case OBJ_RANGE:
    case OBJ_UPVALUE:
    case OBJ_NATIVE:
    case OBJ_BOUND:
    case OBJ_ITER:
    case OBJ_ENUM_CTOR:
    case OBJ_TYPE_COUNT:
        break;   /* fall through to the panic below */
    }

    /* Every compound tag returns from its own case and every sole-block tag
     * returned above. Reaching here means obj->type is corrupt, or that a kind
     * is classified as sole-block by one of the two and compound by the other
     * -- and there is no size with which to free it correctly either way. */
    JAI_PANIC("jaiFreeObject: object with invalid type tag %d", (int)obj->type);
}

/* Does `v` satisfy a declared kind?
 *
 * One vocabulary for every place a declared type meets a runtime value: a
 * class field (OP_FIELD_DEF's spare bits) and a container element
 * (OP_ELEM_KIND). FIELD_KIND_ANY accepts everything, which is what an
 * undeclared field, a nullable, a generic parameter, an unstamped container
 * and any kind this encoding cannot name all carry -- claiming nothing is
 * always safe, and a false TypeError on correct code is worse than a missed
 * one. An unrecognised code is a newer encoding read by an older binary and is
 * treated the same way.
 *
 * A float accepts an int, per LANGUAGE.md §2.5, and stores it as an int: a
 * write arriving through `any` carries no OP_TO_FLOAT, because the checker
 * never saw a float on the target side. Rejecting it would break code that
 * works today. */
bool jaiKindAccepts(uint32_t kind, Value v) {
    switch ((FieldKind)kind) {
    case FIELD_KIND_ANY:      return true;
    case FIELD_KIND_INT:      return IS_INT(v);
    case FIELD_KIND_FLOAT:    return IS_FLOAT(v) || IS_INT(v);
    case FIELD_KIND_BOOL:     return IS_BOOL(v);
    case FIELD_KIND_STR:      return IS_STRING(v);
    case FIELD_KIND_LIST:     return IS_LIST(v);
    case FIELD_KIND_DICT:     return IS_DICT(v);
    case FIELD_KIND_INSTANCE: return IS_INSTANCE(v) || IS_NULL(v);
    }
    return true;
}

/* Throws TypeError and returns false when `v` violates a declared kind.
 *
 * `what` names the thing for the message: "element of list[int]" reads better
 * at a push than "field 'x'". A kind of FIELD_KIND_ANY is the overwhelmingly
 * common case and is one compare away, so an unstamped container pays a
 * predictable not-taken branch and nothing else. */
bool jaiCheckKind(uint32_t kind, Value v, const char *what) {
    if (kind == FIELD_KIND_ANY || jaiKindAccepts(kind, v)) return true;
    return jaiThrow(vm.cTypeError, "cannot put %s into %s of %s",
                    jaiTypeNameStatic(v), what, jaiFieldKindName(kind));
}

const char *jaiFieldKindName(uint32_t kind) {
    switch ((FieldKind)kind) {
    case FIELD_KIND_INT:      return "int";
    case FIELD_KIND_FLOAT:    return "float";
    case FIELD_KIND_BOOL:     return "bool";
    case FIELD_KIND_STR:      return "str";
    case FIELD_KIND_LIST:     return "list";
    case FIELD_KIND_DICT:     return "dict";
    case FIELD_KIND_INSTANCE: return "instance";
    case FIELD_KIND_ANY:      break;
    }
    /* An unrecognised code is a newer encoding read by an older binary, and
     * "any" is the reading that claims nothing. */
    return "any";
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

