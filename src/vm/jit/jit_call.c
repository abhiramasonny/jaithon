/* jit_call.c -- the constructor call, string concatenation, the one-load field
 * read, and the small predicates the call arms ask before emitting. */

#include "vm/jit/jit_arm64.h"
/* For jaiJitFieldReadFor: which builtins are one load from their receiver. */
#include "vm/jit/jit_field_read.h"
/* For jaiBuiltinMethod: resolving `xs.len()` to a native needs the runtime's name table. */
/* For jaiOpBranchOperandAt: says which opcodes carry a branch target. */
#include "vm/vm.h"

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "vm/jit/jit_internal.h"

#if (defined(__aarch64__) || defined(__arm64__))

/* JAITHON_JIT_FIELD_DECL_KIND=0 turns declaredScalarFieldKind's OP_GET_FIELD
 * arm off, reproducing the pre-fix decline for an A/B inside one binary --
 * same cached-getenv idiom as jitDeoptStress above, but default ON since this
 * is a fix, not a stress knob. */

bool jitDeclaredFieldKindEnabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_FIELD_DECL_KIND");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

/* A callee that is not this function: only a class, whose result is an
 * instance of a shape known here. Anything else would need a guard on a return
 * value nothing can predict. */
/* Recognises an initializer that does nothing but store its arguments into fields in order
 * (`GET_LOCAL2 0 k; SET_FIELD f` repeated, then RETURN_NULL) -- anything else (a default, a computed field, a call, a branch) goes the long way. Lets `Point(a, b)` become an allocation and two stores instead of a descriptor + jaiCallValue + invokeCallable's type switch + a compiled init. */
static bool simpleInitFields(ObjClass *cls, unsigned argc, uint16_t *slots) {
    Value initv;
    if (cls == NULL) return false;
    if (!jaiClassFindMethod(cls, vm.strInit, &initv)) return false;
    if (!IS_CLOSURE(initv)) return false;
    ObjFunction *ifn = AS_CLOSURE(initv)->fn;
    if (ifn->arity != argc || ifn->defaultCount != 0) return false;
    if (ifn->flags & (FN_VARIADIC | FN_KWREST)) return false;
    if (ifn->upvalueCount != 0) return false;

    const uint8_t *c = ifn->chunk.code;
    int n = ifn->chunk.count;
    int off = 0;
    for (unsigned i = 0; i < argc; i++) {
        if (off + 5 > n || c[off] != OP_GET_LOCAL2) return false;
        if (jaiReadU16(c + off + 1) != 0) return false;
        if (jaiReadU16(c + off + 3) != i + 1) return false;
        off += 5;
        if (off + 6 > n || c[off] != OP_SET_FIELD) return false;
        uint32_t nameIdx = jaiReadU24(c + off + 1);
        if (nameIdx >= (uint32_t)ifn->chunk.constants.count) return false;
        Value nm = ifn->chunk.constants.data[nameIdx];
        if (!IS_STRING(nm)) return false;
        const FieldInfo *fi = jaiClassFieldInfo(cls, AS_STRING(nm));
        if (fi == NULL || fi->isStatic) return false;
        slots[i] = fi->slot;
        off += 6;
    }
    return off < n && c[off] == OP_RETURN_NULL;
}

bool isClassCallee(const Emit *e, unsigned argc) {
    return e->depth >= argc + 1u &&
           e->stack[e->depth - argc - 1] == SLOT_CLASS;
}

/* JAITHON_JIT_SHAPE_LIMIT=8 puts the OSR instance-shape cap back where it was,
 * for a one-binary A/B. The array in JaiOsrForm is always the wider one, so
 * only the refusal moves. */
unsigned jitShapeLimit(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_SHAPE_LIMIT");
        cached = (v != NULL) ? atoi(v) : (int)JAI_OSR_SHAPES;
        if (cached < 1 || cached > (int)JAI_OSR_SHAPES) cached = (int)JAI_OSR_SHAPES;
    }
    return (unsigned)cached;
}

/* Is the pair on top of the stack `str + str`?
 *
 * A string is SLOT_OBJ here, which pins nothing, so the sample is what makes
 * this worth emitting and the guards below are what make it sound. Only ONE
 * side needs a string sample: OP_FORMAT pushes its result without one (there
 * is no Value to carry at compile time), and `text = text + f"..."` -- the
 * shape word_freq's whole hot loop is -- has exactly that on the right. */
bool concatOperands(const Emit *e, Value *sample) {
    /* Not inside an inlined body. OP_ADD is on inlinableBody's whitelist
     * because every arm it had emitted straight-line code; this one calls, and
     * an inlined body that calls breaks two things at once. Its entries live in
     * x0..x8 when the caller's bank is callee-saved (inlineOwnBank), which is
     * the register file the call destroys, and emitDescriptor reads its
     * arguments through valueBankReg -- which does not know about that bank, so
     * the descriptor is filled from callee-saved registers holding something
     * else entirely. `fn cat(a: any) -> any { return a + "x" }` in a loop
     * segfaults without this line. Declining costs nothing: the inline fails,
     * and both tiers retry the whole body with inlining off, where this arm
     * fires normally. */
    if (e->inlining) return false;
    if (e->depth < 2) return false;
    if (e->stack[e->depth - 1] != SLOT_OBJ) return false;
    if (e->stack[e->depth - 2] != SLOT_OBJ) return false;
    Value sa = e->stackSeen[e->depth - 2], sb = e->stackSeen[e->depth - 1];
    if (!IS_STRING(sa) && !IS_STRING(sb)) return false;
    *sample = IS_STRING(sa) ? sa : sb;
    return true;
}

/* `a + b` out to jaiStringConcat. Concatenation allocates, so there is nothing
 * to inline; the point is that the rest of the loop body stops being given up.
 * word_freq declined its whole `main` loop on the ADD_BIND this replaces -- one
 * refusal costing an LCG step, an f-string and the append around it.
 *
 * Measured (best of five, alternating builds, under the GPU lock): a 3M-step
 * `text = text + f"w{n} "` loop 204.4ms -> 128.4ms, 1.59x; word_freq at 3M
 * words 392.8ms -> 313.9ms end to end, 1.25x; word_freq as shipped 35.2ms ->
 * 28.2ms, 1.85x once the ~20ms startup floor is taken off. Ruled out: a
 * self-hosted `check` of compile/parser.jai does NOT move (775ms -> 767ms,
 * noise) even though OP_ADD is a few percent of its interpreted work -- the
 * bodies holding it decline for other reasons anyway, so freeing this one
 * changes nothing there.
 *
 * Both operands are guarded, not just the sample-less one: SLOT_OBJ says
 * "heap object" and no more, and a guard here resumes at this instruction with
 * both operands still on the interpreter's stack, so a miss costs nothing --
 * a loop alternating `str + str` with `list + list` agrees with the
 * interpreter under both --gc-stress and JAITHON_JIT_DEOPT_STRESS.
 *
 * Deliberately NOT the general `arithmetic()` fallback: that answers int, float
 * and string alike, so the result would need a tag test after the call to know
 * whether the register holds a pointer. Guarding the two operands instead
 * settles the result kind before the call is made, which is the same argument
 * OP_GET_SLICE makes for guarding its container. */
bool emitStringConcat(Emit *e, Value sample) {
    settleAll(e);                    /* this path guards */
    unsigned ra = valueXReg(e, e->valueDepth - 2);
    unsigned rb = valueXReg(e, e->valueDepth - 1);

    emit(e, jaiA64LdrW(JIT_SCRATCH_A, ra, (unsigned)offsetof(Obj, type)));
    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_STRING));
    branchOnDeopt(e, JAI_A64_NE);
    emit(e, jaiA64LdrW(JIT_SCRATCH_A, rb, (unsigned)offsetof(Obj, type)));
    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_STRING));
    branchOnDeopt(e, JAI_A64_NE);

    if (!emitDescriptor(e, NULL_VAL, e->depth - 2, 2,
                        (void *)&jitStringConcat)) {
        return false;
    }
    unsigned drop;
    if (!popValue(e, &drop, NULL)) return false;
    if (!popValue(e, &drop, NULL)) return false;
    /* Carry a sample so the next instruction still knows this is a string --
     * the same reason the string-index arm carries its receiver's. */
    if (!pushValue3(e, SLOT_OBJ, 0, NULL, sample, -1)) return false;
    emit(e, jaiA64LdrX(pushReg(e) - 1, 31,
                       e->descOffset +
                           (unsigned)offsetof(JitCallDesc, result) + 8));
    e->wroteHeap = true;
    return true;
}

/* Put a local on the model stack, the way OP_GET_LOCAL does, for an arm that
 * needs stack operands but was handed slot numbers. Only the general X path --
 * a float local wants OP_GET_LOCAL's FP handling and no caller here has one. */
/* `x == null` where the model calls x an object -- a string, a list, a dict.
 *
 * The arm already mixes SLOT_INST with SLOT_MAYBE_INST because both are a
 * pointer or a zero in a register. A SLOT_OBJ is the same shape, and stronger:
 * emitTagFor gives it VAL_OBJ unconditionally, so its register never holds a
 * zero and the answer is always "not null" -- which is exactly what comparing
 * it against the null literal's zero register produces.
 *
 * EQUALITY ONLY. `x < null` is a TypeError in the interpreter, and compiling it
 * as a register compare would answer where it should raise.
 *
 * SLOT_OBJ ONLY, and that is the load-bearing half. A SLOT_INT is also "never
 * null", but its register holds a NUMBER -- and zero is a perfectly good int,
 * so `0 == null` would compare equal and answer true. Only a kind whose
 * register holds a pointer may be compared against the null literal's zero.
 *
 * 108 declines across four compiler files, in `_parse_postfix`,
 * `_parse_decorators`, `lookup_type_name` and their kin -- the shape is
 * `if tok == null` on something the model did not name more precisely. */
static bool jitNullPair(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_NULL_PAIR");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

bool nullLiteralPair(const Emit *e, uint8_t op, SlotKind ka, SlotKind kb) {
    if (!jitNullPair()) return false;
    if (op != OP_EQ && op != OP_NE) return false;
    if (e->depth < 2) return false;
    if (ka == SLOT_OBJ && kb == SLOT_MAYBE_INST &&
        e->stackNullLit[e->depth - 1]) {
        return true;
    }
    return kb == SLOT_OBJ && ka == SLOT_MAYBE_INST &&
           e->stackNullLit[e->depth - 2];
}

/* Does the instruction after a call throw its result away?
 *
 * `OP_POP` is the obvious spelling and the only one this used to test for. But
 * `lib/jaithon/compile/opt/peephole.jai` fuses `Pop; ReturnNull` into
 * OP_POP_RETURN_NULL at the default -O2, so a `-> void` method called as the
 * LAST STATEMENT of a branch never matched -- and that is the commonest place
 * such a call appears. The escape hatch was written against un-optimised
 * bytecode.
 *
 * It cost more than any other single miss in the tier. `_scan_token`'s
 * `self._line_continuation()` -- a `-> void` method on a backslash branch no
 * source file in this tree even takes -- declined the whole lexer entry point:
 * 2,607,687 interpreted instructions, 7.6% of one file, from five compile
 * attempts before the retry budget ran out.
 *
 * `discardedAfter` reports WHICH, because the fused form has to emit the
 * return half itself rather than simply stepping over the pop. */
static bool jitFusedDiscard(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_FUSED_DISCARD");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

DiscardKind discardedAfter(const uint8_t *code, int at, int count) {
    if (at >= count) return DISCARD_NO;
    if (code[at] == OP_POP) return DISCARD_POP;
    if (code[at] == OP_POP_RETURN_NULL && jitFusedDiscard()) {
        return DISCARD_POP_RETURN;
    }
    return DISCARD_NO;
}

/* The RETURN_NULL half of an OP_POP_RETURN_NULL that followed a call whose
 * result nothing observes.
 *
 * The fused opcode carries the function's return with it, so it cannot be
 * stepped over the way a bare OP_POP is -- doing that would drop the return
 * entirely. This is OP_RETURN_NULL's arm, unchanged, factored out because the
 * same escape hatch appears at four sites in the invoke family and duplicating
 * it four times is how three of them came to be missing the fix in the first
 * place.
 *
 * Returns false having set whyNot; the caller returns false. On true the caller
 * must advance by 8 and set `afterUncond`, because an invoke falls through and
 * a return does not. */
bool emitFusedReturnNull(Emit *e, ObjFunction *fn) {
    if (e->osr) {
        e->whyNot = "a return inside an OSR loop";
        return false;
    }
    if ((fn->flags & FN_INIT) != 0) {
        /* An initialiser's return yields the object, not null, and wants
         * slot 0 -- a different arm entirely. */
        e->whyNot = "a discarded call fused with an initialiser's return";
        return false;
    }
    if (e->sawReturn && e->returnKind != SLOT_NULL) {
        /* One more return site for mergeReturnKind to join, so a body whose
         * other edges return values still reaches SLOT_DYNAMIC. */
        if (!jitDynamicReturn() || !mergeReturnKind(e, SLOT_NULL, 0)) {
            e->whyNot = "two different return kinds";
            return false;
        }
    } else {
        e->sawReturn  = true;
        e->returnKind = SLOT_NULL;
    }
    emit(e, jaiA64MovzX(0, 0, 0));
    emitReturnLeave(e, SLOT_NULL);
    return true;
}

bool jitStrIter(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_STR_ITER");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

/* JAITHON_JIT_ITER_STG=0 puts the nested `for x in <list>` arm back on
 * `emitListBoxedGuard` alone, so the storage dispatch can be A/B'd in one
 * binary. Off, an unboxed list deoptimises on every loop entry and the inner
 * loop runs interpreted -- 3.6x on a nested-loop probe, 2x on graph_bfs. */
bool jitIterStorage(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_ITER_STG");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

/* JAITHON_JIT_STR_HEAD=0 refuses ITER_STRING at an OSR loop head, which is what
 * the tier did until 2026-09-07 ("an iterator kind with no loop-head arm"), so
 * the head can be A/B'd in one binary. Off, a per-character loop long enough to
 * reach the OSR tier runs entirely interpreted: 290ms against 30ms. */
bool jitStringHead(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_STR_HEAD");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

bool pushLocalAsValue(Emit *e, unsigned slot) {
    if (!pushValue3(e, e->localKind[slot], e->localShape[slot],
                    e->localClass[slot],
                    seenLocal(e, slot),
                    (int)slot)) {
        return false;
    }
    unsigned home = localHomeX(e, slot);
    if (home != 0) {
        xBorrowLocal(e, e->valueDepth - 1, home);
    } else {
        unsigned dst = pushReg(e) - 1;
        unsigned src = localIn(e, slot, dst);
        if (src != dst) emit(e, jaiA64MovX(dst, src));
    }
    return true;
}

bool jitConcatLocals(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_CONCAT_LOCALS");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

/* A builtin whose whole body is a load from its receiver (jit_field_read.h).
 * The caller has already guarded that the receiver is `fr->type`, so the load
 * is the entire call: no callee Value, no argument Value, no root fill, no
 * status test, no result tag test. `k.len()` was 39 instructions and a `blr`
 * into jitInvokeNative for a 32-bit field.
 *
 * A lazily-computed field (ObjString::scalars) keeps the descriptor call as a
 * slow path *inline*, reached only when the memo is empty, so the native fills
 * it and every later call takes the load. Deopting there instead would be
 * wrong: a loop over freshly built strings would leave the compiled body on
 * every iteration. Both paths land on the same register, and the span over the
 * slow path is measured rather than counted -- emitDescriptor's length moves
 * with the number of roots the body holds. */
bool emitFieldRead(Emit *e, const JaiJitFieldRead *fr, Value nativeVal,
                          unsigned ridx, unsigned argc, uint32_t afterIp) {
    if (fr->tag != VAL_INT) {
        e->whyNot = "a field-reading builtin whose result is not an int";
        return false;
    }
    unsigned rRecv   = pushReg(e) - argc - 1;
    unsigned skipSlow = 0;

    if (fr->width == 4) {
        emit(e, jaiA64LdrW(JIT_SCRATCH_A, rRecv, fr->offset));
    } else {
        emit(e, jaiA64LdrX(JIT_SCRATCH_A, rRecv, fr->offset));
    }

    if (fr->lazy) {
        emitConst64(e, JIT_SCRATCH_B, (int64_t)(uint64_t)fr->sentinel);
        emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
        skipSlow = e->count;
        emit(e, jaiA64BCond(JAI_A64_NE, 0));      /* patched below */
        if (!emitDescriptor(e, nativeVal, ridx, argc + 1,
                            (void *)&jitInvokeNative)) {
            return false;
        }
    }

    for (unsigned i = 0; i <= argc; i++) {
        unsigned r;
        if (!popValue(e, &r, NULL)) return false;
    }
    if (!pushValue(e, SLOT_INT, 0, NULL)) return false;

    if (fr->lazy) {
        unsigned rat = e->descOffset +
                       (unsigned)offsetof(JitCallDesc, result);
        emit(e, jaiA64LdrW(JIT_SCRATCH_B, 31, rat));
        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_B, VAL_INT));
        branchOnDeoptAt(e, JAI_A64_NE, afterIp, true);
        emit(e, jaiA64LdrX(JIT_SCRATCH_A, 31, rat + 8));
        if (skipSlow < e->count && e->count <= JIT_MAX_INSTS) {
            e->code[skipSlow] =
                jaiA64BCond(JAI_A64_NE, (int32_t)(e->count - skipSlow));
        }
        /* The slow path calls out, and a native may write. */
        e->wroteHeap = true;
    }
    emit(e, jaiA64MovX(pushReg(e) - 1, JIT_SCRATCH_A));
    return true;
}

bool emitCallOut(Emit *e, unsigned argc) {
    ObjClass *cls = e->stackClass[e->depth - argc - 1];
    if (cls == NULL) { e->whyNot = "callee class"; return false; }

    uint16_t fslots[JIT_MAX_ARGS_OUT];
    if (argc <= JIT_MAX_ARGS_OUT && simpleInitFields(cls, argc, fslots)) {
        unsigned first = e->depth - argc;
        SlotKind kinds[JIT_MAX_ARGS_OUT];
        unsigned regs[JIT_MAX_ARGS_OUT];
        for (unsigned i = 0; i < argc; i++) {
            kinds[i] = e->stack[first + i];
            if (kinds[i] != SLOT_INT && kinds[i] != SLOT_FLOAT &&
                kinds[i] != SLOT_BOOL && kinds[i] != SLOT_INST &&
                kinds[i] != SLOT_LIST && kinds[i] != SLOT_OBJ &&
                kinds[i] != SLOT_MAYBE_INST) {
                e->whyNot = "an argument kind a field cannot take";
                return false;
            }
            regs[i] = valueBankReg(e, first + i - (e->depth - e->valueDepth));
        }
        /* Fast path (jitInstanceAlloc) is a leaf that cannot collect -- it declines whenever jaiGCWanted(),
     * exactly when jaiInstanceNew would have collected -- so needs no descriptor/roots, versus the descriptor's dozen stores plus a root push/pop just to allocate 64 bytes. NULL means it did nothing, so falling into the descriptor path is always correct; both paths land at the load below with the instance in SCRATCH_C. Skipped for a class the small-object bins can't serve. */
        const size_t instBytes =
            sizeof(ObjInstance) + sizeof(Value) * (size_t)cls->fieldCount;
        unsigned skipSlow = 0;
        bool haveFast = jaiSmallServes(instBytes);
        if (haveFast) {
            emitConst64(e, 0, (int64_t)(uintptr_t)cls);
            emitConst64(e, JIT_SCRATCH_A,
                        (int64_t)(uintptr_t)&jitInstanceAlloc);
            noteScratchClobber(e);
            emit(e, jaiA64Blr(JIT_SCRATCH_A));
            emit(e, jaiA64MovX(JIT_SCRATCH_C, 0));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_C, 0));
            skipSlow = e->count;
            emit(e, jaiA64BCond(JAI_A64_NE, 0));   /* patched below */
        }
        if (!emitDescriptor(e, OBJ_VAL((Obj *)cls), first, 0,
                            (void *)&jitNewInstance)) {
            return false;
        }
        emit(e, jaiA64LdrX(JIT_SCRATCH_C, 31,
                           e->descOffset +
                               (unsigned)offsetof(JitCallDesc, result) + 8));
        /* The span is measured rather than counted: emitDescriptor's length
         * moves with the number of roots this body holds and with how many
         * halfwords the class pointer needs. */
        if (haveFast && skipSlow < e->count && e->count <= JIT_MAX_INSTS) {
            e->code[skipSlow] =
                jaiA64BCond(JAI_A64_NE, (int32_t)(e->count - skipSlow));
        }
        for (unsigned i = 0; i < argc; i++) {
            unsigned r;
            if (!popValue(e, &r, NULL)) return false;
        }
        if (e->depth == 0 || e->stack[e->depth - 1] != SLOT_CLASS) return false;
        e->depth--;
        if (!pushValue(e, SLOT_INST, cls->shapeId, cls)) return false;
        /* Instance held in a scratch (SCRATCH_C) until every field is stored, since the result reuses the
     * first argument's register: loading it into its final register first overwrote the argument about to be stored into it -- alloc_churn came back in 5ms with a wrong answer. */
        unsigned rinst = pushReg(e) - 1;
        for (unsigned i = 0; i < argc; i++) {
            unsigned at = (unsigned)offsetof(ObjInstance, fields) +
                          (unsigned)fslots[i] * (unsigned)sizeof(Value);
            /* SCRATCH_C holds the instance, so the dynamic tag needs two
             * other scratches. */
            emitTagFor(e, kinds[i], regs[i], JIT_SCRATCH_A, JIT_SCRATCH_B);
            emit(e, jaiA64StrW(JIT_SCRATCH_A, JIT_SCRATCH_C, at));
            emit(e, jaiA64StrX(regs[i], JIT_SCRATCH_C, at + 8));
        }
        emit(e, jaiA64MovX(rinst, JIT_SCRATCH_C));
        e->wroteHeap = true;
        return true;
    }

    if (!emitDescriptor(e, OBJ_VAL((Obj *)cls), e->depth - argc, argc,
                        (void *)&jitCallOut)) {
        return false;
    }

    for (unsigned i = 0; i < argc; i++) {
        unsigned r;
        if (!popValue(e, &r, NULL)) { e->whyNot = "call argument"; return false; }
    }
    if (e->depth == 0 || e->stack[e->depth - 1] != SLOT_CLASS) {
        e->whyNot = "callee was not where it should be";
        return false;
    }
    e->depth--;

    if (!pushValue(e, SLOT_INST, cls->shapeId, cls)) return false;
    emit(e, jaiA64LdrX(pushReg(e) - 1, 31,
                       e->descOffset + (unsigned)offsetof(JitCallDesc, result) + 8));
    /* A call is an effect: no bail may follow it, for the same reason no bail
     * may follow a store. */
    e->wroteHeap = true;
    return true;
}

#else

#endif
