/* jit_runtime.c -- the C helpers compiled code calls back into, and the state it
 * shares with the collector and the deopt path. */
#include "vm/jit/jit.h"

/* For jaiJitFieldReadFor: which builtins are one load from their receiver. */
#include "vm/jit/jit_field_read.h"
#include "vm/gc.h"
/* For jaiBuiltinMethod: resolving `xs.len()` to a native needs the runtime's name table. */
/* For jaiOpBranchOperandAt: says which opcodes carry a branch target. */
#include "vm/vm.h"

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "vm/jit/jit_internal.h"

#if (defined(__aarch64__) || defined(__arm64__))

/* A descriptor call pushes its roots via a C helper; a self-call (bare `bl`) does not, so callee-saved
 * registers are invisible to a collection inside the callee -- the tier refuses allocate-then-self-call for exactly this reason. */
JitCallDesc *gJitFrames;

void jaiJitMarkFrames(void) {
    for (JitCallDesc *f = gJitFrames; f != NULL; f = f->link) {
        for (int64_t i = 0; i < f->nroots; i++) jaiGCMarkValue(f->roots[i]);
    }
}

/* Roots go in as a RANGE (jaiGCPushRootRange/Pop), not copied one at a time -- copying individually
 * was O(roots) per call-out; a bulk jaiGCPushRoots variant was tried and reverted, since it still copied every value. Returns 0 on success, 1 with an exception pending. */
/* Not a deopt: overflow raises directly, since the interpreter would also throw here. A bail is only
 * sound before any write; raising stays sound after a field store has already happened. */
void jitThrowOverflow(int64_t which) {
    static const char *ops[3]  = { "+",  "-",  "*"  };
    static const char *wrap[3] = { "+%", "-%", "*%" };
    int i = (which >= 0 && which < 3) ? (int)which : 0;
    (void)jaiThrow(vm.cOverflowError,
                   "integer overflow in '%s'; use '%s' to wrap", ops[i], wrap[i]);
}

JitDeoptRecord gDeopt;

/* Cached: getenv is O(environ) and this sits on the hot deopt path, so leaving it uncached let ambient
 * shell-exported variable count perturb benchmarks (sort_merge moved 70ms->100ms on padding alone). Same idiom as jaiJitEnabled. */
static bool jitReconTrace(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAI_JIT_RECON");
        cached = (v != NULL && v[0] != '\0') ? 1 : 0;
    }
    return cached != 0;
}

bool jaiJitApplyDeopt(ObjClosure *closure, Value *slotBase) {
    ObjFunction *fn = closure->fn;
    if (gDeopt.ip < 0 || gDeopt.ip >= fn->chunk.count) return false;

    for (int64_t i = 0; i < gDeopt.nlocals; i++) {
        if (i < 64 && (((uint64_t)gDeopt.skipLocals >> i) & 1u) != 0) continue;
        slotBase[gDeopt.base + i] = gDeopt.locals[i];
    }
    /* The operand stack sits above the frame's window, which bindCallArgs has
     * already set vm.stackTop to. */
    for (int64_t i = 0; i < gDeopt.nstack; i++) {
        *vm.stackTop++ = gDeopt.stack[i];
    }
    CallFrame *frame = &vm.frames[vm.frameCount - 1];
    frame->ip = fn->chunk.code + gDeopt.ip;
    if (jitReconTrace()) {
        fprintf(stderr, "[deopt] %s ip=%lld base=%lld nlocals=%lld nstack=%lld\n",
                jitFnLabel(fn), (long long)gDeopt.ip,
                (long long)gDeopt.base, (long long)gDeopt.nlocals,
                (long long)gDeopt.nstack);
    }
    return true;
}

/* Receiver is args[0], exactly where callNativeAt wants it, so no bound wrapper is made. Roots as jitCallOut does, since push and its kin allocate. */
int jitInvokeMethod(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    bool ok = jaiCallMethodWithReceiver(d->callee, d->args, (int)d->argc,
                                        &d->result);
    jaiGCPopRootRange();
    return ok ? 0 : 1;
}

/* Receiver whose class the model could not pin. The method NAME travels in the
 * descriptor's callee slot -- there is no method Value to put there -- and the
 * resolve happens per call against the shared megamorphic table. */
int jitInvokeByName(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    bool ok = jaiInvokeMethodByName(AS_STRING(d->callee), d->args,
                                    (int)d->argc, &d->result);
    jaiGCPopRootRange();
    return ok ? 0 : 1;
}

int jitInvokeNative(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    bool ok = jaiInvokeNativeWithReceiver(d->callee, d->args, (int)d->argc,
                                          &d->result);
    jaiGCPopRootRange();
    return ok ? 0 : 1;
}

/* Allocates (jaiListNew), so roots go down first as for any call out of compiled code. */
int jitBuildList(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    ObjList *list = jaiListNew((int)d->argc);
    for (int64_t i = 0; i < d->argc; i++) jaiListPut(list, (int)i, d->args[i]);
    list->count = (int)d->argc;
    d->result = OBJ_VAL(list);
    jaiGCPopRootRange();
    return 0;
}

/* `x in c`. The needle is args[0] and the container args[1], matching the
 * operand order the interpreter peeks. The answer is stored as a Value so the
 * one-byte BOOL_VAL member is the thing the caller's LdrByte reads -- see the
 * SLOT_BOOL result convention in emitDescriptorStatus.
 *
 * Roots go down because a container's __contains__ can run Jaithon code and
 * therefore collect. Returns 1 having thrown. */
int jitContains(JitCallDesc *d) {
    bool contains = false;
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    bool ok = jaiContainsOp(d->args[1], d->args[0], &contains);
    jaiGCPopRootRange();
    if (!ok) return 1;
    d->result = BOOL_VAL(contains);
    return 0;
}

int jitNotContains(JitCallDesc *d) {
    if (jitContains(d) != 0) return 1;
    d->result = BOOL_VAL(!AS_BOOL(d->result));
    return 0;
}

/* `{a: 1, b: 2}`. Keys and values alternate in the operand list, which is the
 * order the interpreter reads them in, so the args array is used as-is.
 *
 * jaiDictSet hashes the key and can raise on an unhashable one, so the root
 * range goes down and a raise comes back as 1 for the descriptor's threw
 * branch. */
int jitBuildDict(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    ObjDict *dict = jaiDictNew();
    jaiGCPushRoot(OBJ_VAL(dict));
    for (int64_t i = 0; i + 1 < d->argc; i += 2) {
        (void)jaiDictSet(dict, d->args[i], d->args[i + 1]);
        if (vm.hasException) {
            jaiGCPopRoot();
            jaiGCPopRootRange();
            return 1;
        }
    }
    jaiGCPopRoot();
    jaiGCPopRootRange();
    d->result = OBJ_VAL(dict);
    return 0;
}

/* `{a, b}`. Same shape, one operand per element. */
int jitBuildSet(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    ObjSet *set = jaiSetNew();
    jaiGCPushRoot(OBJ_VAL(set));
    for (int64_t i = 0; i < d->argc; i++) {
        (void)jaiSetAdd(set, d->args[i]);
        if (vm.hasException) {
            jaiGCPopRoot();
            jaiGCPopRootRange();
            return 1;
        }
    }
    jaiGCPopRoot();
    jaiGCPopRootRange();
    d->result = OBJ_VAL(set);
    return 0;
}

int jitBuildTuple(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    ObjTuple *tuple = jaiTupleNew(d->args, (int)d->argc);
    jaiGCPopRootRange();
    d->result = OBJ_VAL(tuple);
    return 0;
}

/* `a == b` on two heap objects the tier can say nothing else about.
 *
 * The interpreter's own equality, so an `__eq__` behaves and a comparison that
 * raises raises the same message. It can therefore allocate and throw: roots go
 * down, and 1 comes back for the descriptor's threw branch.
 *
 * This is the LAST arm in each equality chain on purpose. Every arm above it
 * answers without a call -- interned string pointers, a folded enum unit, two
 * registers -- and each is strictly better where it applies. */
int jitValuesEqual(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    bool equal = jaiValuesEqual(d->args[0], d->args[1]);
    jaiGCPopRootRange();
    if (vm.hasException) return 1;
    d->result = BOOL_VAL(equal);
    return 0;
}

bool jitObjEquality(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_OBJ_EQ");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

/* `a + b` on two strings. Both operands are guarded for OBJ_STRING at the call
 * site, so this is jaiStringConcat and nothing else -- the general arithmetic()
 * fallback would have to be answered with a tag test on the result, and there
 * is no shape of `+` other than this one that reaches here.
 *
 * Allocates (and appends into an existing ObjStrBuf), so roots go down first as
 * for any call out of compiled code. NULL back means the concatenation
 * overflowed UINT32_MAX and already threw. */
int jitStringConcat(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    ObjString *s = jaiStringConcat(AS_STRING(d->args[0]), AS_STRING(d->args[1]));
    jaiGCPopRootRange();
    if (s == NULL) return 1;
    d->result = OBJ_VAL(s);
    return 0;
}

/* args: start, stop, inclusive-flag. Allocates twice (range + iterator), so roots go down first as usual. */
int jitMakeRangeIter(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    ObjRange *r = jaiRangeNew(AS_INT(d->args[0]), AS_INT(d->args[1]), 1,
                              AS_INT(d->args[2]) != 0);
    jaiGCPushRoot(OBJ_VAL(r));
    ObjIter *it = jaiIterNew(ITER_RANGE, OBJ_VAL(r));
    jaiGCPopRoot();
    d->result = OBJ_VAL(it);
    jaiGCPopRootRange();
    return 0;
}

/* The iterator for a `for` whose source the caller has already proved. A string
 * yields one-CHARACTER strings and a list its elements; both are the same
 * ObjIter the interpreter builds, so nothing here is a second implementation.
 *
 * The fallthrough RAISES rather than returning 1 quietly. Returning 1 without
 * setting an exception is what the caller reads as "the callee threw", and the
 * VM then reports `internal error: failed operation raised nothing` -- 104 test
 * failures with no useful message, which is how this arm's first draft
 * announced that it had reached here with a string. */
int jitMakeIter(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    Value src = d->args[0];
    IterKind k;
    if (IS_LIST(src)) {
        k = ITER_LIST;
    } else if (IS_STRING(src)) {
        k = ITER_STRING;
    } else {
        jaiGCPopRootRange();
        (void)jaiThrow(vm.cTypeError, "'%s' is not iterable",
                       jaiTypeNameStatic(src));
        return 1;
    }
    ObjIter *it = jaiIterNew(k, src);
    d->result = OBJ_VAL(it);
    jaiGCPopRootRange();
    return 0;
}

/* OP_GET_ITER_ITEMS' dict case: the lazy view `for (k, v) in d.items()` walks,
 * built once per loop entry rather than the N 2-tuples the eager `items()`
 * materialises. Allocates, so roots go down first as for any call out of
 * compiled code. The emitted guard has already proved the receiver is a dict;
 * the test here is the same belt-and-braces jitMakeIter carries. */
int jitMakeItemsIter(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    Value src = d->args[0];
    ObjIter *it = IS_DICT(src) ? jaiIterNew(ITER_DICT_ITEMS, src) : NULL;
    if (it != NULL) d->result = OBJ_VAL(it);
    jaiGCPopRootRange();
    return it != NULL ? 0 : 1;
}

/* f-string: the interpreter's parts, read off the operand stack, land here contiguously in args[].
 * Builtin path only -- compiler checks at compile time that the module hasn't rebound `str`; a rebind retires this form. */
int jitFormat(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    ObjString *formatted = jaiValueFormat(d->args, (int)d->argc);
    jaiGCPopRootRange();
    if (formatted == NULL) return 1;
    d->result = OBJ_VAL(formatted);
    return 0;
}

/* No descriptor/roots: growing a list cannot collect -- jaiListReserve->jaiRealloc never triggers the
 * marker (gc.c: collections only happen at jaiGCMaybeCollect safepoints). Returns 1 if it raised (list past INT32_MAX). */
int jitListGrow(ObjList *list, uint64_t tag, int64_t payload) {
    Value pending;
    pending.type = (ValueType)tag;
    pending.as.integer = payload;
    jaiGCPushRoot(OBJ_VAL(list));
    jaiGCPushRoot(pending);
    if (list->capacity > INT32_MAX / 2) {
        jaiGCPopRoots(2);
        (void)jaiThrow(vm.cRuntimeError,
                       "list cannot grow beyond %d items", INT32_MAX);
        return 1;
    }
    jaiListReserve(list, JAI_GROW_CAP(list->capacity));
    jaiGCPopRoots(2);
    return 0;
}

/* Safe because jaiGCWanted()==false is gc.c's own proof no collection can happen here (collections begin
 * only in jaiGCMaybeCollect), and the object is fully built before being linked in. NULL means "declined": caller falls back to the descriptor path, which roots and may collect. */
ObjInstance *jitInstanceAlloc(ObjClass *cls) {
    if (JAI_UNLIKELY(jaiGCWanted())) return NULL;

    GCState *g = jaiGCActive;
    if (JAI_UNLIKELY(g == NULL || cls == NULL)) return NULL;

    const uint16_t count = cls->fieldCount;
    const size_t size = sizeof(ObjInstance) + sizeof(Value) * (size_t)count;
    if (JAI_UNLIKELY(!jaiSmallServes(size))) return NULL;

    ObjInstance *inst = (ObjInstance *)jaiSmallNew(size);
    Obj *obj = (Obj *)inst;

    obj->type = OBJ_INSTANCE;
    obj->isMarked = false;
    obj->subFlag = false;
    obj->subFlag2 = false;

    inst->klass = cls;
    inst->fieldCount = count;
    /* Zeroes every field, not just the ones about to be overwritten: the marker reads all `count` of them,
 * and an unwritten field is whatever the last occupant of this bin left behind. */
    for (uint16_t i = 0; i < count; ++i) inst->fields[i] = NULL_VAL;

    /* Linked last: a mid-collection isMarked state can't arise here, since a collection in progress makes jaiGCLimit zero and the first line above would already have declined. */
    obj->next = g->objects;
    g->objects = obj;

    vm.allocCount++;
    return inst;
}

int jitNewInstance(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    ObjInstance *inst = jaiInstanceNew((ObjClass *)(uintptr_t)AS_OBJ(d->callee));
    d->result = OBJ_VAL(inst);
    jaiGCPopRootRange();
    return 0;
}

int jitGetSlice(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    uint8_t flags = (uint8_t)d->aux;
    bool hasStart = (flags & 1) != 0, hasStop = (flags & 2) != 0,
         hasStep = (flags & 4) != 0;
    int at = 1;
    Value startV = hasStart ? d->args[at++] : NULL_VAL;
    Value stopV  = hasStop  ? d->args[at++] : NULL_VAL;
    Value stepV  = hasStep  ? d->args[at++] : NULL_VAL;
    bool ok = jaiSliceGet(d->args[0], startV, stopV, stepV,
                          hasStart, hasStop, hasStep, &d->result);
    jaiGCPopRootRange();
    return ok ? 0 : 1;
}

/* Dict read. Calls the interpreter's own OP_GET_INDEX so a missing key raises
 * the KeyError it would have raised, with the message it would have used --
 * naming the key -- rather than a second spelling of the same error that has to
 * be kept in step with it. Pure apart from that throw, but it can allocate the
 * exception, so the roots go down first. */
int jitGetIndexDict(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    bool ok = jaiIndexGet(d->args[0], d->args[1], &d->result);
    jaiGCPopRootRange();
    return ok ? 0 : 1;
}

/* Dict store: unlike a list store there's no offset to normalise, it's a table probe either way --
 * this only saves the dispatch and indexSet's type ladder, not the probe itself. */
int jitSetIndexDict(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    (void)jaiDictSet(AS_DICT(d->args[0]), d->args[1], d->args[2]);
    jaiGCPopRootRange();
    return vm.hasException ? 1 : 0;
}

int jitCallOut(JitCallDesc *d) {
    jaiGCPushRootRange(d->roots, (int)d->nroots);
    bool ok = jaiCallValue(d->callee, (int)d->argc, d->args, &d->result);
    jaiGCPopRootRange();
    return ok ? 0 : 1;
}

#else

/* Nothing compiles here, so there is never a record to resume from and never a
 * compiled frame to mark. */
bool jaiJitApplyDeopt(ObjClosure *closure, Value *slotBase) {
    (void)closure; (void)slotBase; return false;
}
void jaiJitMarkFrames(void) {
}

#endif
