/* jit_call.c -- call emission: descriptors, direct calls, the monomorphic invoke
 * cache, inlining, and the native-call arms. */
#include "vm/jit/jit.h"

#include "vm/jit/jit_arm64.h"
/* For jaiJitFieldReadFor: which builtins are one load from their receiver. */
#include "vm/jit/jit_field_read.h"
#include "vm/gc.h"
/* For jaiBuiltinMethod: resolving `xs.len()` to a native needs the runtime's name table. */
#include "runtime/runtime.h"
/* For jaiOpBranchOperandAt: says which opcodes carry a branch target. */
#include "vm/bytecode/verify.h"
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

/* JAI_JIT_CHAIN=1: print the whole chain of refusals a body would hit, not just
 * the first one.
 *
 * "What would this body stop at NEXT?" is the question that decides whether an
 * arm is worth building, and until now it was answered by BUILDING the arm and
 * re-running -- a day per link, and how three separate changes came to measure
 * exactly zero after clearing one link of a longer chain.
 *
 * The mechanism is deliberately dumb: recompile the body with the offending
 * offset forced onto the unarmed path, and see what it says next. That reuses a
 * path the tier already exercises constantly, rather than continuing a walk
 * whose model has gone inconsistent -- which was tried, and segfaults.
 *
 * Diagnostic only. Each link costs one extra compile of one body, and nothing
 * here runs unless the env var is set. */
bool jitChainOn(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAI_JIT_CHAIN");
        cached = (v != NULL && v[0] != '\0' && v[0] != '0') ? 1 : 0;
    }
    return cached != 0;
}

void reportChain(const Emit *proto, Emit *first, ObjClosure *closure,
                        ObjFunction *fn) {
    static Emit probe;
    uint32_t skips[JIT_MAX_CHAIN];
    unsigned n = 0;
    const char *name = fn->name != NULL ? fn->name->chars : "<anon>";

    fprintf(stderr, "[jit] chain %s:\n", name);
    /* Link 1 is a fact. Everything below it is a PROBE, and the probe is not
     * the same thing as a fix.
     *
     * Stepping over an instruction takes the unarmed path, which abandons the
     * rest of that straight-line block and can only resume at a later
     * independently-reachable jump target. If a bind lived in the abandoned
     * part, the walk reaches the next link with that local never bound at all
     * -- and then reports a refusal ("local N has kind int, not instance")
     * that a genuinely fixed link 1 would never have produced. One night's
     * `.len()` chain read that way and the link 2 it named was an artefact.
     *
     * So: chase link 1. Treat the rest as a hint about where to look next,
     * never as a list of things that must all be cleared. */
    fprintf(stderr, "[jit]   (link 1 is measured; the links below are probed "
                    "by forcing it unarmed,\n[jit]    which skips the rest of "
                    "its block -- treat them as hints, not facts)\n");
    fprintf(stderr, "[jit]   1. %s  (at %u)\n", declineReason(first),
            first->curOffset);
    skips[n++] = first->curOffset;

    for (unsigned link = 2; link <= JIT_MAX_CHAIN; link++) {
        memcpy(&probe, proto, sizeof probe);
        memcpy(probe.chainSkip, skips, n * sizeof skips[0]);
        probe.chainSkipCount = n;
        if (compileBody(&probe, closure)) {
            fprintf(stderr, "[jit]   %u. compiles, once the %u above %s "
                            "cleared\n", link, n, n == 1 ? "is" : "are");
            return;
        }
        /* Refusing again at the SAME offset means the unarmed path cannot step
         * over that instruction: deoptSite has nowhere to resume, which is a
         * real property of the instruction and not an artefact of this probe.
         * Say so rather than numbering it as the next link, because it is not
         * one -- it is where the walk stops being able to look. */
        if (probe.curOffset == skips[n - 1]) {
            fprintf(stderr,
                    "[jit]   ... cannot look past link %u: stepping over it "
                    "gives \"%s\"\n", n, declineReason(&probe));
            return;
        }
        fprintf(stderr, "[jit]   %u. %s  (at %u)\n", link,
                declineReason(&probe), probe.curOffset);
        if (n >= JIT_MAX_CHAIN) {
            fprintf(stderr, "[jit]   ... and the chain runs longer than %u\n",
                    JIT_MAX_CHAIN);
            return;
        }
        skips[n++] = probe.curOffset;
    }
}

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

/* Which local the tier could not settle on one kind for.
 *
 * The reason on its own says a body has such a local but not which, and a
 * body with twenty of them then has to be read line by line to find it. The
 * slot number is what the disassembly labels its locals with, so the two can
 * be put side by side.
 */
/* Record which unnamed refusal an arm took, and return false so the call sites
 * read as `return subWhy(e, "...")`. See Emit::whySub. */
bool subWhy(Emit *e, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(e->whySub, sizeof e->whySub, fmt, ap);
    va_end(ap);
    return false;
}

/* What `JAI_JIT_WHY` prints: the named reason when there is one, otherwise the
 * opcode with whatever the arm noted about it. */
const char *declineReason(Emit *e) {
    if (e->whyNot != NULL) return e->whyNot;
    const char *name = jaiOpName((OpCode)e->lastOp);
    if (e->whySub[0] == '\0') return name;
    snprintf(e->whyBuf, sizeof e->whyBuf, "%s: %s", name, e->whySub);
    return e->whyBuf;
}

bool jitCollectClashes(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_COLLECT_CLASHES");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

const char *kindClash(Emit *e, unsigned slot) {
    /* WHICH two kinds, not just which slot. The slot number says where to look
     * and the pair says what to do about it: an int meeting a float is a
     * widening the tier could learn, an instance meeting a list is a genuinely
     * polymorphic local and nothing will help it. Without the pair, ~290 of
     * these across four compiler files were one undifferentiated heap. */
    snprintf(e->whyBuf, sizeof e->whyBuf,
             "local %u was given two kinds, %s and %s", slot,
             slotKindName(e->localKind[slot]), slotKindName(e->clashKind));
    return e->whyBuf;
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
        e->whyNot = "two different return kinds";
        return false;
    }
    e->sawReturn  = true;
    e->returnKind = SLOT_NULL;
    emit(e, jaiA64MovzX(0, 0, 0));
    emitEpilogue(e, 0);
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

/* What an unarmed opcode was ABOUT, when the opcode alone does not say.
 *
 * `walked only to OP_GET_GLOBAL at 53` names the instruction and not the
 * question. The question is WHICH global, because that is what decides whether
 * the stop is the deliberate one on a cold `throw` path or a callee that could
 * have compiled: `sqrt` and `floor` each stop at one, and together they are
 * 4.2% of the jaicv benchmark's interpreted work. */
const char *unarmedDetail(const ObjFunction *fn, uint8_t op,
                                 uint32_t at) {
    if (op != OP_GET_GLOBAL) return "";
    if ((size_t)at + 4 > (size_t)fn->chunk.count) return "";
    uint32_t idx = jaiReadU24(fn->chunk.code + at + 1);
    if (idx >= (uint32_t)fn->chunk.constants.count) return "";
    Value name = fn->chunk.constants.data[idx];
    if (!IS_STRING(name)) return "";
    static char buf[96];
    snprintf(buf, sizeof buf, " (`%s`)", AS_STRING(name)->chars);
    return buf;
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


/* ---- literal operands -------------------------------------------------- */

/* Whether anything but fall-through can reach `off`: reading the previous instruction's literal
 * without this check is a miscompile, not a decline -- `x // (if c {2} else {4})` puts OP_INT right before OP_FLOORDIV *and* a jump from the other arm onto it. Scans the WHOLE chunk, not just fixups already emitted (a back edge compiles after its target is walked, so the fixup list would miss loop tops) -- and handler/finally addresses too, since the unwinder can resume there with a stack this walk never saw. */
bool offsetIsBranchTarget(const Chunk *c, uint32_t off) {
    for (int at = 0; at < c->count;) {
        int len = instructionLength(c, at);
        if (len <= 0) return true;      /* undecodable: assume the worst */
        int rel = jaiOpBranchOperandAt(c->code[at]);
        if (rel >= 0) {
            int16_t jump = jaiReadI16(c->code + at + 1 + rel);
            /* Every branch operand is measured from the end of the
             * instruction, which is what `at + len` is. */
            if ((int32_t)(at + len) + jump == (int32_t)off) return true;
        }
        at += len;
    }
    return false;
}

/* OP_INT carries its value inline, OP_CONST names a pool entry -- the only two ways a literal reaches
 * the stack. The adjacency check is belt-and-braces (the walk is linear) but a real bug: OP_FORMAT once advanced `off` by nine instead of ten, and this is what would have caught it. */
bool literalIntOperand(const ObjFunction *fn, int prevOff, int off,
                              int64_t *out) {
    if (prevOff < 0 || prevOff >= off) return false;
    const Chunk *c = &fn->chunk;
    if (prevOff + instructionLength(c, prevOff) != off) return false;
    uint8_t prev = c->code[prevOff];
    if (prev == OP_INT) {
        *out = jaiReadI16(c->code + prevOff + 1);
    } else if (prev == OP_CONST) {
        uint32_t idx = jaiReadU24(c->code + prevOff + 1);
        if (idx >= (uint32_t)c->constants.count) return false;
        Value k = c->constants.data[idx];
        if (!IS_INT(k)) return false;
        *out = AS_INT(k);
    } else {
        return false;
    }
    return !offsetIsBranchTarget(c, (uint32_t)off);
}

/* `k` is 2^shift, for a shift this can name. Positive only: floor division by
 * a negative power of two is not a shift, and `k` is at most 2^62 because 2^63
 * does not fit in a positive int64. */
/* Collapses the general "add divisor back if remainder is non-zero and signs differ" (7 instructions)
 * to 2 when the divisor's sign is known at compile time: msub leaves |r| < |d| with r's sign following the dividend's, so for a positive divisor the whole test is "r < 0" (one bit), and for a negative one "r > 0". `r + d` cannot overflow since |r| < |d| puts the sum strictly between -|d| and |d|. */
void emitFloorFixup(Emit *e, unsigned rrem, unsigned rd,
                           bool signKnown, int64_t divisor, uint32_t fixup) {
    if (signKnown && divisor > 0) {
        emit(e, jaiA64Tbz(rrem, 63u, 2));
        emit(e, fixup);
        return;
    }
    if (signKnown) {
        emit(e, jaiA64SubsXImm(31, rrem, 0));
        emit(e, jaiA64BCond(JAI_A64_LE, 2));
        emit(e, fixup);
        return;
    }
    emit(e, jaiA64SubsXImm(31, rrem, 0));
    emit(e, jaiA64BCond(JAI_A64_EQ, 5));
    emit(e, jaiA64EorX(JIT_SCRATCH_D, rrem, rd));
    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_D, 0));
    emit(e, jaiA64BCond(JAI_A64_GE, 2));
    emit(e, fixup);
}

bool powerOfTwoShift(int64_t k, unsigned *shift) {
    if (k <= 0) return false;
    uint64_t u = (uint64_t)k;
    if ((u & (u - 1u)) != 0u) return false;
    unsigned s = 0;
    while ((u >> s) != 1u) s++;
    *shift = s;
    return true;
}

/* The kind a container's elements may be held as, read off ONE live element.
 *
 * The sample specialises and the tag guard at the read site confirms: a
 * container that later holds something else deoptimises rather than being
 * answered wrongly. An instance carries its class too, since a tag check alone
 * cannot tell two shapes apart. */
bool exemplarKind(Value elem, SlotKind *kind, unsigned *tag,
                         ObjClass **cls, uint32_t *shape) {
    *cls = NULL;
    *shape = 0;
    if (IS_INT(elem))        { *kind = SLOT_INT;   *tag = VAL_INT;   return true; }
    if (IS_FLOAT(elem))      { *kind = SLOT_FLOAT; *tag = VAL_FLOAT; return true; }
    if (IS_BOOL(elem))       { *kind = SLOT_BOOL;  *tag = VAL_BOOL;  return true; }
    if (IS_LIST(elem))       { *kind = SLOT_LIST;  *tag = VAL_OBJ;   return true; }
    if (rawObjValue(elem))   { *kind = SLOT_OBJ;   *tag = VAL_OBJ;   return true; }
    if (IS_INSTANCE(elem) && AS_INSTANCE(elem)->klass != NULL) {
        *kind  = SLOT_INST;
        *tag   = VAL_OBJ;
        *cls   = AS_INSTANCE(elem)->klass;
        *shape = (*cls)->shapeId;
        return true;
    }
    return false;
}

/* An exemplar for the elements a freshly-built list is about to hold.
 *
 * `[1, 2, 3]` has no live list to sample -- the list does not exist until the
 * compiled code runs -- but the model knows the kind of every entry that went
 * into it. For a scalar that is enough: a synthesised zero of the right kind
 * answers every question the iterate arm asks of a sample, and being an
 * immediate it has no lifetime to worry about. For an object the element's own
 * sample is reused, which was already being held on the operand stack.
 *
 * All `n` must agree, and an instance must agree on its class too, for the same
 * reason the dict walk insists on it: a mispredicted element kind deoptimises on
 * every read, which is worse than not compiling at all.
 *
 * Returns false when the list is empty (a comprehension's accumulator) or the
 * elements disagree -- in both cases there is simply nothing to predict. */
bool buildListExemplar(const Emit *e, unsigned first, unsigned n,
                              Value *out) {
    if (n == 0) return false;
    Value chosen = NULL_VAL;
    for (unsigned i = 0; i < n; i++) {
        unsigned idx = first + i;
        Value here;
        switch (e->stack[idx]) {
        case SLOT_INT:   here = INT_VAL(0);        break;
        case SLOT_FLOAT: here = FLOAT_VAL(0.0);    break;
        case SLOT_BOOL:  here = BOOL_VAL(false);   break;
        case SLOT_OBJ:
        case SLOT_LIST:
        case SLOT_INST:
            here = e->stackSeen[idx];
            if (!IS_OBJ(here) || AS_OBJ(here) == NULL) return false;
            break;
        default:
            return false;
        }
        if (i == 0) { chosen = here; continue; }
        if (jaiValueType(here) != jaiValueType(chosen)) return false;
        if (IS_OBJ(here) && OBJ_TYPE(here) != OBJ_TYPE(chosen)) return false;
        if (IS_INSTANCE(here) &&
            AS_INSTANCE(here)->klass != AS_INSTANCE(chosen)->klass) {
            return false;
        }
    }
    *out = chosen;
    return true;
}

/* One value out of a live dict, and only if every value in it agrees.
 *
 * A LIST is sampled at index 0 alone, because a list's elements are usually
 * built by one loop and a wrong guess costs a deopt per read. A dict is not:
 * `{"name": "x", "count": 3}` is an ordinary dict and its values disagree, so
 * predicting off the first entry would deoptimise every iteration -- measured
 * elsewhere at 5.7x worse than declining outright (chunk.h states the same for
 * an invoke's result). Refusing a mixed dict is the point of the walk.
 *
 * Capped, so compiling a body that indexes a large dict does not walk it. Past
 * the cap the guard still holds; only the prediction is made on a prefix. */
#define JIT_DICT_SAMPLE_MAX 256u

bool dictUniformValue(ObjDict *dict, Value *out) {
    const JaiTable *t = &dict->table;
    if (t->entries == NULL || t->count <= 0) return false;
    bool have = false;
    Value first = NULL_VAL;
    unsigned seen = 0;
    for (int i = 0; i < t->capacity && seen < JIT_DICT_SAMPLE_MAX; i++) {
        /* A live entry is one with a nonnegative order; empty and tombstoned
         * slots both carry a negative one (see table.c's entryIsLive). */
        if (t->entries[i].order < 0) continue;
        Value v = t->entries[i].value;
        seen++;
        if (!have) { first = v; have = true; continue; }
        if (jaiValueType(v) != jaiValueType(first)) return false;
        if (IS_OBJ(v) && OBJ_TYPE(v) != OBJ_TYPE(first)) return false;
        if (IS_INSTANCE(v) && AS_INSTANCE(v)->klass != AS_INSTANCE(first)->klass) {
            return false;
        }
    }
    if (!have) return false;
    *out = first;
    return true;
}

/* Builtins whose result kind is a property of the FUNCTION and not of its
 * arguments: `str(x)` is a string whatever x is, `len(x)` is an int, `bool(x)`
 * is a bool. That is the only thing the surrounding body needs to know, so the
 * call can be an ordinary call out and everything around it stays compiled.
 *
 * Worth having because the alternative was not a slower call but no compiled
 * body at all -- one `str()` in a loop declined the whole enclosing function.
 * A probe doing `str(i % 10_000)` per iteration ran 90,000,323 interpreted
 * instructions against 1,231 for the f-string spelling of the same thing.
 *
 * The kinds are read off the natives in builtins_core.c, and the returned tag
 * is guarded regardless: a wrong row here costs a deopt, never an answer. */
typedef struct {
    const char *name;
    unsigned    argc;
    SlotKind    kind;
    uint8_t     tag;
} NativeResult;

static const NativeResult kNativeResults[] = {
    { "str",        1, SLOT_OBJ,  VAL_OBJ  },
    { "repr",       1, SLOT_OBJ,  VAL_OBJ  },
    { "chr",        1, SLOT_OBJ,  VAL_OBJ  },
    { "type_of",    1, SLOT_OBJ,  VAL_OBJ  },
    { "len",        1, SLOT_INT,  VAL_INT  },
    { "hash",       1, SLOT_INT,  VAL_INT  },
    { "id",         1, SLOT_INT,  VAL_INT  },
    { "ord",        1, SLOT_INT,  VAL_INT  },
    { "int",        1, SLOT_INT,  VAL_INT  },
    { "int",        2, SLOT_INT,  VAL_INT  },
    { "bool",       1, SLOT_BOOL, VAL_BOOL },
    { "callable",   1, SLOT_BOOL, VAL_BOOL },
    { "isinstance", 2, SLOT_BOOL, VAL_BOOL },
    /* No `range` row. It would emit, but the loop that consumes the result
     * declines one instruction later ("iterating something other than a list
     * or range") because SLOT_OBJ does not say `range` -- so the body is
     * refused either way and the row only buys an allocation. */
};

/* `sum`, `min` and `max` over a LIST answer with the element's own kind: an int
 * list sums to an int and a float list to a float (checked against `type_of`,
 * not assumed). So unlike every row in kNativeResults their result is a
 * property of the ARGUMENT, and the table cannot state it.
 *
 * Worth the separate arm because they are cheap next to the loop around them,
 * which is the test a builtin row has to pass -- `sorted` is not, and its rows
 * were built and discarded for measuring zero
 * (docs/research/FALSIFIED-list-returning-builtins.md). A five-element `sum` is
 * five adds; `docs/probes/p20_sum_min_max.jai` ran 17,000,327 interpreted
 * instructions, i.e. the whole loop, on account of this one refusal.
 *
 * The exemplar comes from a live list if the model has one and otherwise from
 * `stackElem`, which is what OP_BUILD_LIST recorded -- the same two sources the
 * iterate arm reads, in the same order. */
/* JAITHON_JIT_LIST_SCALAR=0 turns the arm below off, so it can be A/B'd inside
 * ONE binary. Not a nicety: two-binary comparisons are where this tree's
 * measurements go wrong -- each switch invalidates __jaicache__, and three
 * people measuring one change tonight two-binary got 3.9x, 100x and 6%
 * SLOWER for what a switch settled in one command. */
/* JAITHON_JIT_NEGATE=0 turns off the OP_NEG arm, for a one-binary A/B. */
bool jitSoftField(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_SOFT_FIELD");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

bool jitMembership(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_MEMBERSHIP");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

bool jitTuple(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_TUPLE");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

bool jitNegate(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_NEGATE");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

/* JAITHON_JIT_LIST_RESULT=0 turns off the predicted result for a list method
 * that is neither a field read nor discarded, for a one-binary A/B. */
bool jitListResult(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_LIST_RESULT");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

static bool jitListScalarResult(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_LIST_SCALAR");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

static bool listScalarResult(const Emit *e, const char *nm, unsigned argc,
                             SlotKind *kind, uint8_t *tag) {
    if (!jitListScalarResult()) return false;
    if (argc != 1 && !(argc == 2 && strcmp(nm, "sum") == 0)) return false;
    if (strcmp(nm, "sum") != 0 && strcmp(nm, "min") != 0 &&
        strcmp(nm, "max") != 0) {
        return false;
    }
    unsigned idx = e->depth - argc;
    if (e->stack[idx] != SLOT_LIST) return false;

    Value elem = NULL_VAL;
    Value seen = e->stackSeen[idx];
    if (IS_LIST(seen) && AS_LIST(seen)->count > 0) {
        elem = jaiListGet(AS_LIST(seen), 0);
    }
    if (IS_NULL(elem)) elem = e->stackElem[idx];

    if (IS_INT(elem))   { *kind = SLOT_INT;   *tag = VAL_INT;   return true; }
    if (IS_FLOAT(elem)) { *kind = SLOT_FLOAT; *tag = VAL_FLOAT; return true; }
    return false;
}

/* 1 emitted, 0 no row for this builtin, -1 the emit failed. */
int emitNativeResultCall(Emit *e, Value cv, const char *nm,
                                unsigned argc, uint32_t afterIp) {
    /* inlinableBody admits OP_CALL only for the two builtins the tier emits as
     * a single instruction, on the grounds that an inlined body cannot call.
     * Refusing here keeps that true even if a constant string reaches an
     * `int()` inside one. */
    if (e->inlining) return 0;

    NativeResult derived;
    const NativeResult *nr = NULL;
    for (size_t i = 0; i < sizeof kNativeResults / sizeof kNativeResults[0]; i++) {
        if (kNativeResults[i].argc == argc &&
            strcmp(kNativeResults[i].name, nm) == 0) {
            nr = &kNativeResults[i];
            break;
        }
    }
    if (nr == NULL) {
        SlotKind dk;
        uint8_t dtag;
        if (!listScalarResult(e, nm, argc, &dk, &dtag)) return 0;
        derived.name = nm;
        derived.argc = argc;
        derived.kind = dk;
        derived.tag  = dtag;
        nr = &derived;
    }

    if (!emitDescriptor(e, cv, e->depth - argc, argc, (void *)&jitCallOut)) {
        return -1;
    }
    for (unsigned i = 0; i < argc; i++) {
        /* A class argument occupies no register, so there is nothing to pop --
         * only the entry to drop. */
        if (e->depth > 0 && !holdsRegister(e->stack[e->depth - 1])) {
            e->depth--;
            continue;
        }
        unsigned r;
        if (!popValue(e, &r, NULL)) { e->whyNot = "call argument"; return -1; }
    }
    if (e->depth == 0 || e->stack[e->depth - 1] != SLOT_NATIVE) {
        e->whyNot = "callee was not where it should be";
        return -1;
    }
    e->depth--;
    if (!pushValue(e, nr->kind, 0, NULL)) return -1;

    unsigned at = e->descOffset + (unsigned)offsetof(JitCallDesc, result);
    emit(e, jaiA64LdrW(JIT_SCRATCH_A, 31, at));
    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, nr->tag));
    /* Resumes AFTER the call, taking the result from the descriptor: the
     * native has already run and may have written, so re-running it is not on
     * offer. `lastFromDesc` is what hands the interpreter the Value the native
     * actually produced, whatever tag it turned out to have. */
    branchOnDeoptAt(e, JAI_A64_NE, afterIp, true);
    /* A bool is ONE byte of the Value union; reading eight would carry the
     * neighbouring bytes of the result slot into the register. */
    if (nr->kind == SLOT_BOOL) {
        emit(e, jaiA64LdrByte(pushReg(e) - 1, 31, at + 8));
    } else {
        emit(e, jaiA64LdrX(pushReg(e) - 1, 31, at + 8));
    }
    /* A call is an effect: no bail may follow it. */
    e->wroteHeap = true;
    return 1;
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
