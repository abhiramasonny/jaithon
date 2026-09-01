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


/* ownStatus: caller decodes the helper's return itself, skipping the built-in "nonzero means raised"
 * test. Written for the iterator step (0 yielded, 1 exhausted, 2 raised), whose call-out the list arm of
 * OP_FOR_ITER_BIND no longer makes -- the default test sent `exhausted` to the throw stub, which found no pending exception and died on "internal error: failed operation raised nothing". Kept because any helper with a three-way answer needs it, and because the lesson is not rediscoverable from the code. */
/* Root-fills the descriptor: shared by the descriptor path (a C helper pushes them) and the self-call
 * path (the emitted code links the descriptor onto the collector's frame chain instead, since a bare `bl` pushes nothing). */
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

/* JAITHON_JIT_ROOT_LIMIT=10 puts the root cap back where it was when it shared
 * the register budget, for a one-binary A/B. The ARRAY is always the wider one,
 * so only the refusal moves. */
static unsigned jitRootLimit(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_ROOT_LIMIT");
        cached = (v != NULL) ? atoi(v) : (int)JIT_MAX_ROOTS;
        if (cached < 1 || cached > (int)JIT_MAX_ROOTS) cached = (int)JIT_MAX_ROOTS;
    }
    return (unsigned)cached;
}

bool emitRootFill(Emit *e, unsigned d, unsigned *nrootsOut) {
    unsigned nroots = 0;
    for (unsigned slot = e->base; slot < e->base + e->locals; slot++) {
        if (e->localKind[slot] != SLOT_INST &&
            e->localKind[slot] != SLOT_LIST &&
            e->localKind[slot] != SLOT_OBJ &&
            e->localKind[slot] != SLOT_ITER &&
            e->localKind[slot] != SLOT_MAYBE_INST) {
            continue;
        }
        if (nroots >= jitRootLimit()) {
            e->whyNot = "too many roots"; return false;
        }
        unsigned at = d + (unsigned)offsetof(JitCallDesc, roots) +
                      nroots * (unsigned)sizeof(Value);
        unsigned rslot = localIn(e, slot, JIT_SCRATCH_C);
        emitTagFor(e, e->localKind[slot], rslot, JIT_SCRATCH_B, JIT_SCRATCH_A);
        emit(e, jaiA64StrW(JIT_SCRATCH_B, 31, at));
        emit(e, jaiA64StrX(rslot, 31, at + 8));
        nroots++;
    }

    /* Locals alone weren't enough once OP_GET_ITER started leaving an ObjIter live in a register across
 * a call that might collect -- only visible under --gc-stress, and only once such a loop could compile at all. */
    /* Counts register-holding entries from the bottom, not by assuming they're the top `valueDepth` --
 * a no-register entry (class/function/builtin/self) can sit in the middle of the stack, e.g. `join(f(a), f(b))` pushes a callee before its arguments. Subtracting valueDepth would name the wrong register above it and skip entries that still need rooting; the deopt stub has always counted this way. */
    unsigned seen = 0;
    for (unsigned idx = 0; idx < e->depth; idx++) {
        SlotKind k = e->stack[idx];
        if (!holdsRegister(k)) continue;
        unsigned reg = valueBankReg(e, seen);
        seen++;
        if (k != SLOT_INST && k != SLOT_LIST && k != SLOT_OBJ &&
            k != SLOT_ITER && k != SLOT_MAYBE_INST) {
            continue;
        }
        if (nroots >= jitRootLimit()) {
            e->whyNot = "too many roots"; return false;
        }
        unsigned at = d + (unsigned)offsetof(JitCallDesc, roots) +
                      nroots * (unsigned)sizeof(Value);
        emitTagFor(e, k, reg, JIT_SCRATCH_B, JIT_SCRATCH_A);
        emit(e, jaiA64StrW(JIT_SCRATCH_B, 31, at));
        emit(e, jaiA64StrX(reg, 31, at + 8));
        nroots++;
    }

    *nrootsOut = nroots;
    return true;
}

bool emitDescriptorStatus(Emit *e, Value calleeVal, unsigned first,
                                 unsigned nargs, void *helper, bool ownStatus,
                                 int calleeReg) {
    if (nargs > JIT_MAX_ARGS_OUT) { e->whyNot = "call argc"; return false; }
    if (!e->callsOut) { e->whyNot = "callsOut off"; return false; }

    unsigned d = e->descOffset;

    /* Callee as a whole Value, from a register when only known at run time (a closure held in a local):
     * baking the compile-time-live closure would freeze its upvalues -- `closure_calls` builds a fresh closure over a different `step` every outer iteration. */
    if (calleeReg >= 0) {
        emit(e, jaiA64MovzX(JIT_SCRATCH_A, VAL_OBJ, 0));
        emit(e, jaiA64StrW(JIT_SCRATCH_A, 31,
                           d + (unsigned)offsetof(JitCallDesc, callee)));
        emit(e, jaiA64StrX((unsigned)calleeReg, 31,
                           d + (unsigned)offsetof(JitCallDesc, callee) + 8));
    } else {
        emit(e, jaiA64MovzX(JIT_SCRATCH_A, (unsigned)calleeVal.type, 0));
        emit(e, jaiA64StrW(JIT_SCRATCH_A, 31,
                           d + (unsigned)offsetof(JitCallDesc, callee)));
        emitConst64(e, JIT_SCRATCH_A, (int64_t)(uintptr_t)calleeVal.as.obj);
        emit(e, jaiA64StrX(JIT_SCRATCH_A, 31,
                           d + (unsigned)offsetof(JitCallDesc, callee) + 8));
    }

    /* The value index of the first argument. Counted rather than derived from
     * `depth - valueDepth`, which is the number of register-free entries below
     * `depth` ANYWHERE: a class argument is one of those, and every argument
     * above it would then be read out of the wrong register. */
    unsigned vidx = e->valueDepth;
    for (unsigned idx = first; idx < e->depth; idx++) {
        if (holdsRegister(e->stack[idx])) vidx--;
    }

    /* The arguments, which for an invoke begin with the receiver. */
    for (unsigned i = 0; i < nargs; i++) {
        unsigned idx = first + i;
        SlotKind k = e->stack[idx];
        unsigned at = d + (unsigned)offsetof(JitCallDesc, args) +
                      i * (unsigned)sizeof(Value);
        /* A class occupies no register -- it is a constant of the module, and
         * the class the interpreter would have found is the one the model
         * recorded. Baking it is the same trust the callee slot above already
         * takes, and it is what lets `isinstance(x, T)` be called at all. */
        if (k == SLOT_CLASS) {
            ObjClass *argCls = e->stackClass[idx];
            if (argCls == NULL) {
                e->whyNot = "a class argument the model did not pin";
                return false;
            }
            emit(e, jaiA64MovzX(JIT_SCRATCH_A, VAL_OBJ, 0));
            emit(e, jaiA64StrW(JIT_SCRATCH_A, 31, at));
            emitConst64(e, JIT_SCRATCH_A, (int64_t)(uintptr_t)argCls);
            emit(e, jaiA64StrX(JIT_SCRATCH_A, 31, at + 8));
            continue;
        }
        if (k != SLOT_INT && k != SLOT_FLOAT && k != SLOT_BOOL &&
            k != SLOT_INST && k != SLOT_LIST && k != SLOT_OBJ &&
            k != SLOT_ITER && k != SLOT_MAYBE_INST) {
            e->whyNot = "an argument kind this call cannot pass";
            return false;
        }
        unsigned reg = valueBankReg(e, vidx);
        vidx++;
        /* A maybe-instance's tag is not a property of its kind, and this Value
         * reaches jaiCallValue: writing VAL_OBJ over a zero payload would hand
         * the interpreter a null pointer dressed as an object. */
        emitTagFor(e, k, reg, JIT_SCRATCH_B, JIT_SCRATCH_A);
        emit(e, jaiA64StrW(JIT_SCRATCH_B, 31, at));
        emit(e, jaiA64StrX(reg, 31, at + 8));
    }

    unsigned nroots = 0;
    if (!emitRootFill(e, d, &nroots)) return false;
    emit(e, jaiA64MovzX(JIT_SCRATCH_A, nargs, 0));
    emit(e, jaiA64StrX(JIT_SCRATCH_A, 31,
                       d + (unsigned)offsetof(JitCallDesc, argc)));
    emit(e, jaiA64MovzX(JIT_SCRATCH_A, nroots, 0));
    emit(e, jaiA64StrX(JIT_SCRATCH_A, 31,
                       d + (unsigned)offsetof(JitCallDesc, nroots)));

    emit(e, jaiA64AddXImm(0, 31, d));
    emitConst64(e, JIT_SCRATCH_A, (int64_t)(uintptr_t)helper);
    noteScratchClobber(e);
    emit(e, jaiA64Blr(JIT_SCRATCH_A));

    if (ownStatus) return true;

    /* Nonzero means the callee raised; the interpreter owns it from here. */
    if (!raiseExitAllowed(e, "a call that can raise inside a try")) return false;
    emit(e, jaiA64SubsXImm(31, 0, 0));
    if (e->fixupCount >= JIT_MAX_FIXUPS) { e->failed = true; return false; }
    e->fixups[e->fixupCount].instIndex    = (int)e->count;
    e->fixups[e->fixupCount].targetOffset = FIXUP_THREW;
    e->fixups[e->fixupCount].conditional  = true;
    e->fixups[e->fixupCount].depth        = -1;
    e->fixupCount++;
    emit(e, jaiA64BCond(JAI_A64_NE, 0));
    return true;
}

bool emitDescriptor(Emit *e, Value calleeVal, unsigned first,
                           unsigned nargs, void *helper) {
    return emitDescriptorStatus(e, calleeVal, first, nargs, helper, false, -1);
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

/* Direct-branch arguments arrive as a raw payload, so the caller's kind must match what the callee
 * was specialised for -- and for an instance, the same class shape, since every field offset was resolved against it. jaiJitEnterFunc is the only check standing between a float's bits and a body that treats them as a pointer. `firstIdx`: the entry above the callee for a plain call, or the receiver (callee's slot 0) for a method. */
bool directCallArgsMatch(Emit *e, const ObjFunction *cfn,
                                unsigned firstIdx, unsigned argc) {
    for (unsigned i = 0; i < argc; i++) {
        unsigned idx = firstIdx + i;
        SlotKind have = e->stack[idx];
        SlotKind want = (SlotKind)cfn->jitParamKind[i];
        if (!holdsRegister(have)) {
            e->whyNot = "a direct call argument that is not in a register";
            return false;
        }
        if (want == SLOT_OPAQUE) continue;   /* never read; see seedLocals */
        if (want == SLOT_MAYBE_INST) {
            if (have != SLOT_INST && have != SLOT_MAYBE_INST) {
                e->whyNot = "a direct call argument is not the parameter's kind";
                return false;
            }
        } else if (have != want) {
            e->whyNot = "a direct call argument is not the parameter's kind";
            return false;
        }
        if ((want == SLOT_INST || want == SLOT_MAYBE_INST) &&
            e->stackShape[idx] != cfn->jitParamShape[i]) {
            e->whyNot = "a direct call passing a different class";
            return false;
        }
    }
    return true;
}

/* Branches straight to a compiled callee's entry, skipping the descriptor/jaiCallValue/interpreter-
 * frame path. Convention: raw payloads in x0.., closure in the last arg register if the callee reads an upvalue, x0/x1 = value/verdict on return -- the same one jaiJitEnterFunc checks and a self-call already uses, so skipping that entry means answering its checks here instead: module version (why the callee must live in the caller's module), every parameter's kind+shape (by the caller's model), and the verdict (below). Nonzero verdict: a callee that writes NOTHING can have the whole call abandoned and re-executed from the pre-call stack (two compares, no stub); a callee that WRITES cannot be re-run -- verdict 4 means it deoptimised part-way and is FINISHED in the interpreter from its own record, sharing the `selfSlow` machinery a recursive self-call already uses. A raised exception goes to the throw exit instead, since its effects already happened. `calleeReg`: the ObjClosure register, or -1 if baked in. `cidx`: operand-stack index of the callee entry (the RECEIVER for a method, i.e. its slot 0). `after`: offset of the fall-through instruction. */
/* A `-> T?` result read back out of a call descriptor.
 *
 * Two tags are acceptable where every other kind has exactly one, so the single
 * compare the other arms use cannot serve. VAL_NULL is zero, which is what lets
 * "object or null" be two csels and a compare rather than a branch: afterwards
 * D holds the payload or a defined zero, and B is zero exactly when the tag was
 * one of the two.
 *
 * The class checks apply only to a non-null. The field arm at
 * OP_GET_FIELD_LOCAL keeps its equivalents branch-free by redirecting the loads
 * at its live receiver; a call result has no such pointer to borrow, so they
 * sit behind a forward branch instead. A deopt inside that span is sound for
 * the reason emitBoundsNormalise's is: the record is written from the
 * descriptor, which holds the true Value whichever way the branch went.
 *
 * Worth having because after this the refusal it removes was the tier's single
 * largest on the self-hosted compiler: one `check --no-cache` of
 * compile/parser.jai stopped 64 bodies at "callee's return kind not usable"
 * with the kind being exactly SLOT_MAYBE_INST, and every one of them had a
 * resolvable class. `-> Node?` is what a parser's methods return. */
void emitMaybeInstResult(Emit *e, unsigned dst, unsigned rat,
                                uint32_t rshape, uint32_t deoptIp) {
    emit(e, jaiA64LdrW(JIT_SCRATCH_A, 31, rat));
    emit(e, jaiA64LdrX(JIT_SCRATCH_D, 31, rat + 8));
    emit(e, jaiA64MovzX(JIT_SCRATCH_B, 0, 0));
    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, VAL_OBJ));
    emit(e, jaiA64CselX(JIT_SCRATCH_D, JIT_SCRATCH_D, JIT_SCRATCH_B,
                        JAI_A64_EQ));
    emit(e, jaiA64CselX(JIT_SCRATCH_B, JIT_SCRATCH_B, JIT_SCRATCH_A,
                        JAI_A64_EQ));
    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_B, 0));
    branchOnDeoptAt(e, JAI_A64_NE, deoptIp, true);

    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_D, 0));
    unsigned skip = e->count;
    emit(e, jaiA64BCond(JAI_A64_EQ, 0));
    emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_D,
                       (unsigned)offsetof(Obj, type)));
    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_INSTANCE));
    branchOnDeoptAt(e, JAI_A64_NE, deoptIp, true);
    emit(e, jaiA64LdrX(JIT_SCRATCH_A, JIT_SCRATCH_D,
                       (unsigned)offsetof(ObjInstance, klass)));
    emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_A,
                       (unsigned)offsetof(ObjClass, shapeId)));
    emitConst64(e, JIT_SCRATCH_B, (int64_t)rshape);
    emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
    branchOnDeoptAt(e, JAI_A64_NE, deoptIp, true);
    /* Nothing to jump over means the arena filled mid-sequence; leaving the
     * branch unpatched would run the class loads on a null pointer. */
    if (skip < e->count && e->count <= JIT_MAX_INSTS) {
        e->code[skip] = jaiA64BCond(JAI_A64_EQ, (int32_t)(e->count - skip));
    } else {
        e->failed = true;
    }
    emit(e, jaiA64MovX(dst, JIT_SCRATCH_D));
}

bool jitAnyGuard(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_ANY_GUARD");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

static bool jitReturnKnownOn(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_RETURN_KNOWN");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

bool emitDirectCall(Emit *e, ObjFunction *caller, ObjFunction *cfn,
                           Value calleeVal, int calleeReg, unsigned cidx,
                           unsigned argc, uint32_t callOff, uint32_t after,
                           bool method) {
    /* Either verdict path below can come back with an exception pending. */
    if (!raiseExitAllowed(e, "a call that can raise inside a try")) return false;
    if (cfn->module != caller->module) {
        e->whyNot = "a direct callee from another module";
        return false;
    }
    /* Slot 0 is the closure for a plain function and the receiver for a
     * method, so which of the two this is decides where the arguments start
     * and whether the receiver is one of them. */
    if (cfn->jitArgBase != (method ? 0u : 1u)) {
        e->whyNot = method
            ? "a direct method that does not take its receiver in slot 0"
            : "a direct callee that is not a plain function";
        return false;
    }
    /* The callee's baked classes/closures/natives are pinned by ITS jitFuncModuleVersion, checked at the
     * entry this call skips -- valid only if the caller's own check agrees NOW: a global rebound after the callee compiled would leave jitFuncModuleVersion stale but still reachable via a direct call, and a caller compiling afterwards would silently pin the newer version. */
    if (caller->module == NULL ||
        cfn->jitFuncModuleVersion != caller->module->version) {
        e->whyNot = "a direct callee compiled against an older module";
        return false;
    }
    /* A callee that writes is finished in the interpreter on verdict 4, and
     * that needs its ObjClosure -- which only a callee baked in at compile
     * time provides. */
    bool writes = !cfn->jitFuncNoWrite;
    if (writes && (calleeReg >= 0 || !IS_CLOSURE(calleeVal))) {
        e->whyNot = "a direct callee that writes and is not known here";
        return false;
    }
    /* `nargs` is how many registers the branch fills. A method's receiver is
     * one of them; a plain call's callee entry holds no register at all. */
    unsigned nargs = method ? argc + 1u : argc;
    unsigned firstIdx = method ? cidx : cidx + 1u;
    unsigned calleeArgs = (unsigned)cfn->jitArgCount;
    bool wantsClosure = calleeArgs == nargs + 1u;
    if (!wantsClosure && calleeArgs != nargs) {
        e->whyNot = "a direct callee with a different arity";
        return false;
    }
    if (calleeArgs > JIT_MAX_ARITY) {
        e->whyNot = "a direct callee with too many arguments";
        return false;
    }

    if (!directCallArgsMatch(e, cfn, firstIdx, nargs)) return false;
    if (wantsClosure &&
        (SlotKind)cfn->jitParamKind[nargs] != SLOT_CLOSURE) {
        e->whyNot = "a direct callee whose trailing argument is not its closure";
        return false;
    }
    if (wantsClosure && calleeReg < 0 && !IS_CLOSURE(calleeVal)) {
        e->whyNot = "a direct callee that wants a closure it has not got";
        return false;
    }
    /* SLOT_NULL: a `-> void` function. Epilogue leaves x0 zero, so the pushed entry has a fixed tag and
     * zero payload -- same treatment a self-call to a void function gets. Refusing it declined every caller of a procedure, which in nbody is the whole of `main`. */
    /* A callee whose walk never reached an OP_RETURN has no return kind to be
     * the contract, only the SLOT_INT of a zeroed Emit. `_is_ident_start` in
     * the lexer walks only to OP_GET_GLOBAL at offset 0 -- a cold `throw` on
     * its first instruction -- and still claimed to return an int, which made
     * `_is_ident_start(c) or _is_digit(c)` decline on "a branch on a int, not
     * a bool" and left `_ident_run_end`'s OSR loop retrying it eighty times.
     *
     * Declining here is not a coverage loss: the caller falls back to the
     * guarded emitGlobalCall path, which asks the interpreter's own
     * observation first and gets the right answer. */
    if (!cfn->jitReturnKnown && jitReturnKnownOn()) {
        e->whyNot = "a direct callee whose walk never reached a return";
        return false;
    }
    SlotKind rk = (SlotKind)cfn->jitReturnKind;
    ObjClass *rcls = NULL;
    /* SLOT_MAYBE_INST rides with SLOT_INST here and needs no guard of its own:
     * a direct branch takes the callee's raw payload, and for a nullable
     * instance that payload IS the pointer or a zero -- the same
     * representation this caller will hold. The callee's declared return kind
     * is the contract, exactly as it is for every other kind at this arm. */
    if (rk != SLOT_INT && rk != SLOT_FLOAT && rk != SLOT_BOOL &&
        rk != SLOT_INST && rk != SLOT_MAYBE_INST && rk != SLOT_LIST &&
        rk != SLOT_OBJ && rk != SLOT_NULL) {
        e->whyNot = "callee's return kind not usable";
        return false;
    }
    if ((rk == SLOT_INST || rk == SLOT_MAYBE_INST) &&
        (cfn->jitReturnShape == 0 ||
         !jaiClassForShape(cfn->jitReturnShape, &rcls) || rcls == NULL)) {
        e->whyNot = "callee's return class not on record";
        return false;
    }
    /* Asked here rather than where the slot is taken: below this point the
     * root fill has been emitted and the descriptor linked onto the collector's
     * chain, so there is no falling back to the descriptor path any more. */
    if (writes && e->selfSlowCount >= JIT_MAX_SELF_SLOW) {
        e->whyNot = "more slow call sites than the tier tracks";
        return false;
    }

    /* Past here everything is settled and this call is happening: a failure
     * below is the emitter running out of room, not a decision, so it stops
     * the compile rather than falling back to the descriptor path onto a
     * half-written instruction stream. */

    /* Roots before the branch: a `bl` pushes none, and the callee may
     * allocate -- OP_GET_SLICE builds a fresh list without ever counting as a
     * heap write. */
    unsigned callRoots = 0;
    if (!emitRootFill(e, e->descOffset, &callRoots)) { e->failed = true; return false; }
    if (callRoots > 0) {
        unsigned dd = e->descOffset;
        emit(e, jaiA64MovzX(JIT_SCRATCH_A, callRoots, 0));
        emit(e, jaiA64StrX(JIT_SCRATCH_A, 31,
                           dd + (unsigned)offsetof(JitCallDesc, nroots)));
        emitConst64(e, JIT_SCRATCH_A, (int64_t)(uintptr_t)&gJitFrames);
        emit(e, jaiA64LdrX(JIT_SCRATCH_B, JIT_SCRATCH_A, 0));
        emit(e, jaiA64AddXImm(JIT_SCRATCH_C, 31, dd));
        emit(e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_C,
                           (unsigned)offsetof(JitCallDesc, link)));
        emit(e, jaiA64StrX(JIT_SCRATCH_C, JIT_SCRATCH_A, 0));
    }

    unsigned firstArg = firstIdx - (e->depth - e->valueDepth);
    for (unsigned i = 0; i < nargs; i++) {
        emit(e, jaiA64MovX(i, valueXReg(e, firstArg + i)));
    }
    if (wantsClosure) {
        if (calleeReg >= 0) emit(e, jaiA64MovX(nargs, (unsigned)calleeReg));
        else emitConst64(e, nargs, (int64_t)(uintptr_t)AS_OBJ(calleeVal));
    }
    emitConst64(e, JIT_SCRATCH_D, (int64_t)(uintptr_t)cfn->jitFunc);
    noteScratchClobber(e);
    emit(e, jaiA64Blr(JIT_SCRATCH_D));

    if (callRoots > 0) {
        unsigned dd = e->descOffset;
        emitConst64(e, JIT_SCRATCH_A, (int64_t)(uintptr_t)&gJitFrames);
        emit(e, jaiA64AddXImm(JIT_SCRATCH_C, 31, dd));
        emit(e, jaiA64LdrX(JIT_SCRATCH_B, JIT_SCRATCH_C,
                           (unsigned)offsetof(JitCallDesc, link)));
        emit(e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_A, 0));
    }

    unsigned si = 0;
    if (writes) {
        /* One compare and one not-taken branch on the fast path; every other
         * answer is the shared cold block. The record has to be taken here,
         * where the model still holds the receiver and the arguments, even
         * though the block is emitted with the stubs. */
        if (e->selfSlowCount >= JIT_MAX_SELF_SLOW) { e->failed = true; return false; }
        si = e->selfSlowCount++;
        e->selfSlow[si].roots    = callRoots;
        e->selfSlow[si].stub     = -1;
        e->selfSlow[si].callee   = AS_CLOSURE(calleeVal);
        e->selfSlow[si].retShape = rk == SLOT_INST ? cfn->jitReturnShape : 0;
        e->selfSlow[si].retType  = rk == SLOT_INST ? (int)OBJ_INSTANCE
                                 : rk == SLOT_LIST ? (int)OBJ_LIST
                                                   : -1;
        if (!deoptRecordAt(e, callOff, false, &e->selfSlow[si].deoptBail)) {
            e->failed = true;
            return false;
        }
        emit(e, jaiA64SubsXImm(31, 1, 0));
        if (e->fixupCount >= JIT_MAX_FIXUPS) { e->failed = true; return false; }
        e->fixups[e->fixupCount].instIndex    = (int)e->count;
        e->fixups[e->fixupCount].targetOffset = FIXUP_SELFSLOW - si;
        e->fixups[e->fixupCount].conditional  = true;
        e->fixups[e->fixupCount].depth        = -1;
        e->fixupCount++;
        emit(e, jaiA64BCond(JAI_A64_NE, 0));
    } else {
        /* Verdict 2 is a pending exception: the interpreter owns it and this
         * call must not run again. */
        emit(e, jaiA64SubsXImm(31, 1, 2));
        if (e->fixupCount >= JIT_MAX_FIXUPS) { e->failed = true; return false; }
        e->fixups[e->fixupCount].instIndex    = (int)e->count;
        e->fixups[e->fixupCount].targetOffset = FIXUP_THREW;
        e->fixups[e->fixupCount].conditional  = true;
        e->fixups[e->fixupCount].depth        = -1;
        e->fixupCount++;
        emit(e, jaiA64BCond(JAI_A64_EQ, 0));
        /* Anything else -- a bail, or a guard that failed inside the callee --
         * hands the whole call back. The record is taken with the callee and
         * its arguments still on the model's stack, which is what the
         * interpreter expects to find at this offset. */
        emit(e, jaiA64SubsXImm(31, 1, 0));
        branchOnDeoptAt(e, JAI_A64_NE, callOff, false);
    }

    for (unsigned i = 0; i < nargs; i++) {
        unsigned r;
        if (!popValue(e, &r, NULL)) { e->failed = true; return false; }
    }
    if (!method) {
        if (e->depth == 0) { e->failed = true; return false; }
        e->depth--;
    }
    if (!pushValue(e, rk, cfn->jitReturnShape, rcls)) { e->failed = true; return false; }
    emit(e, jaiA64MovX(pushReg(e) - 1, 0));
    if (writes) {
        e->selfSlow[si].resultReg = pushReg(e) - 1;
        e->selfSlow[si].returnTo  = (int)e->count;
        e->selfSlow[si].tag = rk == SLOT_INT   ? VAL_INT
                            : rk == SLOT_FLOAT ? VAL_FLOAT
                            : rk == SLOT_BOOL  ? VAL_BOOL
                                               : VAL_OBJ;
        /* The interpreted continuation is typed by nothing this compiled for,
         * so what it hands back is checked and a surprise resumes AFTER the
         * call -- which has happened and must not happen twice. */
        if (!deoptRecordAt(e, after, true, &e->selfSlow[si].deoptKind)) {
            e->failed = true;
            return false;
        }
        /* A call that writes is an effect, so no bail may follow it -- the
         * same rule the descriptor path lives under. */
        e->wroteHeap = true;
    }
    /* Deliberately NOT wroteHeap for a non-writing callee, unlike the descriptor path: it stores and
     * calls nothing, so a re-run repeats no visible effect. It may still allocate (OP_GET_SLICE does) -- a fresh object isn't an observable effect. */
    return true;
}

/* JAITHON_JIT_PIC=0 turns the one-way inline cache below off, so the same
 * binary can be A/B'd around it without a rebuild -- same idiom as
 * jaiListUnboxOn's JAITHON_LIST_UNBOX (object_collection.c) and
 * jaiJitEnabled's JAITHON_NO_JIT. Read once: this sits on every unpinned
 * OP_INVOKE, and an uncached getenv there is its own cost (see
 * jitReconTrace above). */
bool jitPicEnabled(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("JAITHON_JIT_PIC");
        on = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return on != 0;
}

/* Mirrors every decision emitDirectCall makes before it commits to emitting,
 * for the one way this cache is about to speculate on. By the time the shape
 * compare below is in the instruction stream a decline has nowhere to fall
 * back to but e->failed, so this runs FIRST and answers instead of finding
 * out the hard way. */
static bool jitPic1Admissible(Emit *e, ObjFunction *caller, ObjFunction *cfn,
                              unsigned ridx, unsigned argc, uint32_t shape) {
    if (cfn->module != caller->module) return false;
    if (caller->module == NULL ||
        cfn->jitFuncModuleVersion != caller->module->version) {
        return false;
    }
    if (cfn->jitArgBase != 0u) return false;
    unsigned nargs = argc + 1u;
    unsigned calleeArgs = (unsigned)cfn->jitArgCount;
    bool wantsClosure = calleeArgs == nargs + 1u;
    if (!wantsClosure && calleeArgs != nargs) return false;
    if (calleeArgs > JIT_MAX_ARITY) return false;
    if (wantsClosure &&
        (SlotKind)cfn->jitParamKind[nargs] != SLOT_CLOSURE) {
        return false;
    }
    /* The receiver is the callee's slot 0, and the branch about to be
     * emitted is the proof of its class -- so the parameter the callee was
     * specialised for has to be that same class, not merely some instance. */
    if ((SlotKind)cfn->jitParamKind[0] != SLOT_INST) return false;
    if (cfn->jitParamShape[0] != shape) return false;
    for (unsigned i = 1; i < nargs; i++) {
        unsigned idx = ridx + i;
        SlotKind have = e->stack[idx];
        SlotKind want = (SlotKind)cfn->jitParamKind[i];
        if (!holdsRegister(have)) return false;
        if (want == SLOT_OPAQUE) continue;   /* never read; see seedLocals */
        if (want == SLOT_MAYBE_INST) {
            if (have != SLOT_INST && have != SLOT_MAYBE_INST) return false;
        } else if (have != want) {
            return false;
        }
        if ((want == SLOT_INST || want == SLOT_MAYBE_INST) &&
            e->stackShape[idx] != cfn->jitParamShape[i]) {
            return false;
        }
    }
    /* A callee that writes is finished in the interpreter from a selfSlow
     * record, and this arm takes one of its own -- the same budget a pinned
     * direct call draws from. */
    if (!cfn->jitFuncNoWrite && e->selfSlowCount >= JIT_MAX_SELF_SLOW) {
        return false;
    }
    return true;
}

/* The compile-time half of a ONE-WAY inline cache: what the model could not
 * pin about a receiver, read off the site's own InlineCache instead.
 *
 * One way only, and only the FIRST shape this site ever recorded --
 * ic->cached[0] / ic->shapeId[0] -- not the whole walk over every way a
 * polymorphic cache holds. The receiver's class is loaded once and compared
 * against that one shape; a hit branches straight into the recorded callee's
 * compiled entry exactly as a pinned receiver already does (emitDirectCall).
 * A miss -- every OTHER class at a polymorphic site, and the common case at a
 * megamorphic one -- falls through to jitInvokeByName, which is exactly what
 * the site emits today and is unchanged by any of this.
 *
 * The compare is not a guard and a miss is not a deopt: both sides of it run
 * BEFORE the call, so nothing has happened yet that must not happen twice.
 * That is the whole reason this needs no deopt record of its own, unlike the
 * call inside it (emitDirectCall still takes one, for what the CALL can do
 * after it starts).
 *
 * `havePrediction`/`rkind` is the fall-through's own prediction
 * (siteInvokeResultKind), which the one way must also return: only a
 * register-only kind (SLOT_INT/FLOAT/BOOL) is admitted, because those are the
 * only ones that carry no class shape for the two paths to disagree about --
 * see feedbackSlotKind's exclusion of SLOT_INST. `*toEnd` comes back holding
 * the arm's own branch to the merge point, for the caller to patch once it
 * has emitted the fall-through after it.
 *
 * Returns false having emitted nothing whenever there was something to
 * cleanly decline, so the caller can still take the descriptor path alone --
 * UNLESS e->failed, which means the shape compare is already in the stream
 * and there is nowhere left to fall back to (mirrors emitDirectCall's own
 * rule, since this arm ends by calling into it). */
bool emitInvokePic1(Emit *e, ObjFunction *fn, unsigned ridx,
                           unsigned argc, uint32_t callOff, uint32_t after,
                           int siteCache, bool havePrediction, SlotKind rkind,
                           int *toEnd) {
    if (!havePrediction) {
        return subWhy(e, "no predicted result kind to join the arm on");
    }
    if (rkind != SLOT_INT && rkind != SLOT_FLOAT && rkind != SLOT_BOOL) {
        return subWhy(e, "an unpinned receiver returning something with a shape");
    }
    if (siteCache < 0 || fn->chunk.caches == NULL ||
        siteCache >= fn->chunk.cacheCount) {
        return subWhy(e, "no inline cache recorded for this site");
    }
    const InlineCache *ic = &fn->chunk.caches[siteCache];
    /* IC_MEGA is admitted, and it is the case that matters. A site that runs
     * out of ways stops caching ALTOGETHER and re-resolves every call through
     * sMegaCache -- see the comment on that table -- which is exactly the
     * shape a trait with eight implementations makes, and exactly the shape
     * this arm exists for. The ways it recorded before it gave up are still
     * there and still true: a way is (shapeId, method), shapeIds come from a
     * monotonic counter that would need four billion classes to repeat, and
     * every way is re-checked below for a live class and a compiled callee.
     * Four ways against eight classes is a partial cache, not a complete one,
     * and a partial cache is the whole point -- the misses cost one compare
     * each and then do exactly what the site does today. */
    if (ic->state != IC_MONO && ic->state != IC_POLY &&
        ic->state != IC_MEGA) {
        return subWhy(e, "the site's cache is empty");
    }
    if (ic->count == 0) return subWhy(e, "the site's cache has no way filled");
    if (ridx + argc + 1u > JIT_MAX_STACK) {
        return subWhy(e, "past the stack depth the model can describe");
    }
    /* Every way the cache holds is tried, in the order it recorded them, so
     * a site that warmed up on its second-most-common class no longer
     * speculates on the wrong one. The compares chain: way w's compare falls
     * through to way w+1's, and the last falls through to the descriptor the
     * site emits today. A miss therefore costs one compare per way and then
     * does exactly what it did before. */
    /* Collect the ways this compile can actually take. A way is dropped, not
     * fatal: the site keeps its remaining arms and the dropped class simply
     * goes round the descriptor as it does today. */
    unsigned    wayShape[JAI_IC_WAYS];
    Value       wayVal  [JAI_IC_WAYS];
    ObjFunction *wayFn  [JAI_IC_WAYS];
    ObjClass    *wayCls [JAI_IC_WAYS];
    unsigned    ways = 0;

    for (int w = 0; w < ic->count && w < JAI_IC_WAYS; w++) {
        /* What the cache settles is which method a shape resolves to, not
         * whether THIS caller may call it -- one site can present as two
         * classes at different visibilities, so the interpreter re-decides
         * that on every hit and nothing emitted here can. */
        if (ic->payload[w] != 0) continue;
        Value cv = ic->cached[w];
        if (!IS_CLOSURE(cv)) continue;
        ObjFunction *cf = AS_CLOSURE(cv)->fn;
        if (cf->jitFunc == NULL) continue;
        if ((SlotKind)cf->jitReturnKind != rkind || cf->jitReturnShape != 0) {
            continue;
        }
        ObjClass *cc = NULL;
        if (!jaiClassForShape(ic->shapeId[w], &cc) || cc == NULL) continue;
        if (!jitPic1Admissible(e, fn, cf, ridx, argc, ic->shapeId[w])) continue;
        wayShape[ways] = ic->shapeId[w];
        wayVal[ways]   = cv;
        wayFn[ways]    = cf;
        wayCls[ways]   = cc;
        ways++;
    }
    if (ways == 0) return subWhy(e, "no way of this site's cache is usable");

    settleAll(e);
    fpReleaseAll(e);
    if (e->fpLive != 0) {
        return subWhy(e, "an unpinned receiver with a value in the float bank");
    }

    /* Past here the shape compare is in the stream and the site is
     * committed: a failure below stops the compile rather than falling
     * back. */
    unsigned rreg = valueXReg(e, ridx - (e->depth - e->valueDepth));
    /* The receiver's shape is loaded ONCE and every way compares against it. */
    emit(e, jaiA64LdrX(JIT_SCRATCH_A, rreg,
                       (unsigned)offsetof(ObjInstance, klass)));
    emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_A,
                       (unsigned)offsetof(ObjClass, shapeId)));

    /* The model as the fall-through must find it again: emitDirectCall
     * consumes the receiver and the arguments and pushes a result, and the
     * fall-through's own descriptor call has to see the SAME depth and
     * valueDepth it would have without any of this, or the two paths
     * disagree about where the result lands. */
    unsigned  saveDepth      = e->depth;
    unsigned  saveValueDepth = e->valueDepth;
    SlotKind  saveKind [JIT_MAX_STACK];
    uint32_t  saveShape[JIT_MAX_STACK];
    ObjClass *saveClass[JIT_MAX_STACK];
    Value     saveSeen [JIT_MAX_STACK];
    memcpy(saveKind,  e->stack,      sizeof saveKind);
    memcpy(saveShape, e->stackShape, sizeof saveShape);
    memcpy(saveClass, e->stackClass, sizeof saveClass);
    memcpy(saveSeen,  e->stackSeen,  sizeof saveSeen);

    /* One arm a way. Each is: prove the shape, call directly, jump to the
     * merge. A way that does not match falls into the next way's compare, and
     * the last falls into the descriptor path the caller emits -- which is
     * what this site did for every receiver before any of this. */
    int armMiss[JAI_IC_WAYS];
    for (unsigned w = 0; w < ways; w++) {
        emitConst64(e, JIT_SCRATCH_B, (int64_t)wayShape[w]);
        emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
        armMiss[w] = (int)e->count;
        emit(e, jaiA64BCond(JAI_A64_NE, 0));

        /* True on this arm alone, which is why it is also what the CALL's own
         * deopt records (inside emitDirectCall, for what happens after the
         * call starts -- not for this compare) should say: the branch just
         * above proved it. */
        e->depth      = saveDepth;
        e->valueDepth = saveValueDepth;
        memcpy(e->stack,      saveKind,  sizeof saveKind);
        memcpy(e->stackShape, saveShape, sizeof saveShape);
        memcpy(e->stackClass, saveClass, sizeof saveClass);
        memcpy(e->stackSeen,  saveSeen,  sizeof saveSeen);
        e->stack[ridx]      = SLOT_INST;
        e->stackShape[ridx] = wayShape[w];
        e->stackClass[ridx] = wayCls[w];

        if (!emitDirectCall(e, fn, wayFn[w], wayVal[w], -1, ridx, argc,
                            callOff, after, true)) {
            /* jitPic1Admissible said it would take this and it did not: the
             * branch into it is already emitted, so there is nowhere left to
             * fall back to. */
            e->failed = true;
            return false;
        }
        if (e->picExitCount >= JIT_MAX_PIC_EXITS) {
            e->failed = true;
            return false;
        }
        e->picExits[e->picExitCount++] = (int)e->count;
        emit(e, jaiA64B(0));

        /* Same condition as the placeholder (NE: skip the call on a shape
         * that doesn't match), now with the real offset -- flipping it to EQ
         * would call this way's callee on every receiver whose shape did NOT
         * match, reading its fields at the wrong class's layout. That is what
         * test_mixed_list_runs_every_class caught: a wrong answer, not a
         * crash, because the read lands inside the instance's own allocation.
         *
         * Guarded because emit() silently DROPS the word once e->count reaches
         * JIT_MAX_INSTS and only sets e->failed -- so the slot may never have
         * been written, and Emit::code is immediately followed by `count` with
         * no padding between them. */
        if (armMiss[w] < (int)e->count && e->count <= JIT_MAX_INSTS) {
            e->code[armMiss[w]] =
                jaiA64BCond(JAI_A64_NE, (int32_t)((int)e->count - armMiss[w]));
        }
    }

    /* The model the caller's descriptor path must find, restored exactly. */
    e->depth      = saveDepth;
    e->valueDepth = saveValueDepth;
    memcpy(e->stack,      saveKind,  sizeof saveKind);
    memcpy(e->stackShape, saveShape, sizeof saveShape);
    memcpy(e->stackClass, saveClass, sizeof saveClass);
    memcpy(e->stackSeen,  saveSeen,  sizeof saveSeen);

    if (getenv("JAI_JIT_WHY")) {
        fprintf(stderr, "[jit] pic %u-way (of %u recorded, state %d) at %u\n",
                ways, (unsigned)ic->count, (int)ic->state, callOff);
    }

    /* The fall-through runs with the receiver and the arguments untouched,
     * so the caller emits its descriptor path against the model as it was
     * before any of this. */
    e->depth      = saveDepth;
    e->valueDepth = saveValueDepth;
    memcpy(e->stack,      saveKind,  sizeof saveKind);
    memcpy(e->stackShape, saveShape, sizeof saveShape);
    memcpy(e->stackClass, saveClass, sizeof saveClass);
    memcpy(e->stackSeen,  saveSeen,  sizeof saveSeen);
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

/* Something inside an inlined body could not be emitted, so the whole compile
 * is worth retrying with inlining off rather than declining: the same call
 * through the descriptor still compiles, and a compiled form with a real call
 * in it beats none at all. A file static for the same reason the Emit buffers
 * are -- compilation is not reentrant, nothing it calls compiles anything. */
bool gInlineFailed;

/* Structural check, answered before anything is emitted (a half-inlined body can't be taken back):
 * no branches (no offset map, no join, no fixup naming a callee offset in the caller's table); exactly one RETURN, last; locals only via the four opcodes the inline frame understands, and only slots this callee actually has; globals only for the two builtins the tier emits inline (else a global VALUE load would bake a JaiEntry from the callee's own table, needing its own guard); nothing that stores (a guard inside re-executes the WHOLE call, so an earlier store would run twice). What's left is straight-line register arithmetic -- the main walker already speaks it, so no second emitter is needed. `evalA` in spectral is fifteen instructions of exactly this shape. */
static bool inlinableBody(ObjClosure *callee, unsigned argc,
                          unsigned *maxSlotOut, bool *readsUpvalueOut) {
    ObjFunction *cfn = callee->fn;
    const Chunk *c = &cfn->chunk;
    if (cfn->arity != argc || cfn->defaultCount != 0) return false;
    if (cfn->flags & (FN_VARIADIC | FN_KWREST | FN_INIT)) return false;
    if (c->count <= 0 || c->count > 128) return false;

    unsigned maxSlot = argc;
    bool sawReturn = false;
    bool readsUpvalue = false;
    for (int off = 0; off < c->count;) {
        uint8_t op = c->code[off];
        int len = instructionLength(c, off);
        if (len <= 0) return false;
        unsigned slot = 0, slot2 = 0;
        switch (op) {
        /* The local frame the caller builds understands exactly these. Any
         * other opcode naming a slot -- a field read off one, a compare
         * against one, an in-place update -- would read the CALLER's local of
         * that number, which is a different variable entirely. */
        case OP_GET_LOCAL:
        case OP_BIND:
            slot = jaiReadU16(c->code + off + 1);
            if (slot > maxSlot) maxSlot = slot;
            break;
        case OP_GET_LOCAL2:
        case OP_ADD_LOCALS:
            slot  = jaiReadU16(c->code + off + 1);
            slot2 = jaiReadU16(c->code + off + 3);
            if (slot > maxSlot) maxSlot = slot;
            if (slot2 > maxSlot) maxSlot = slot2;
            break;
        /* An upvalue is reached through the closure that is actually being
         * called, which is a register the call site has to supply -- so this
         * is only inlinable where that register exists. OP_SET_UPVALUE is not
         * here and falls to `default`: a store would have to be undone if a
         * later guard in the same body deoptimised to the call. */
        case OP_GET_UPVALUE:
            if ((unsigned)c->code[off + 1] >= (unsigned)cfn->upvalueCount) {
                return false;
            }
            readsUpvalue = true;
            break;
        case OP_GET_GLOBAL: {
            uint32_t nameIdx = jaiReadU24(c->code + off + 1);
            Value nv;
            if (globalNative(callee, nameIdx, &nv) == NULL) return false;
            ObjNative *nat = AS_NATIVE(nv);
            const char *nm = nat->name != NULL ? nat->name->chars : "";
            if (strcmp(nm, "float") != 0 && strcmp(nm, "int") != 0) return false;
            break;
        }
        case OP_CALL:
            /* The only callee that can be on the stack here is one of the two
             * builtins above, and the tier emits those as one instruction. */
            if (c->code[off + 1] != 1) return false;
            break;
        case OP_CONST: case OP_INT: case OP_TRUE: case OP_FALSE:
        case OP_ADD: case OP_SUB: case OP_MUL: case OP_DIV:
        case OP_FLOORDIV: case OP_MOD: case OP_POW: case OP_NEG:
        case OP_BAND: case OP_BOR: case OP_BXOR:
        case OP_SHL: case OP_SHR: case OP_BNOT:
        case OP_TYPE_GUARD:
            break;
        case OP_RETURN:
            if (off + len != c->count) return false;
            sawReturn = true;
            break;
        default:
            return false;
        }
        off += len;
    }
    if (!sawReturn) return false;
    if (maxSlot > JIT_MAX_SLOTS) return false;
    *maxSlotOut = maxSlot;
    *readsUpvalueOut = readsUpvalue;
    return true;
}

/* Inlines the callee's body: slot 1+i IS entry cidx+1+i already on the stack, so nothing is copied in;
 * a bound slot pins one more entry underneath what's pushed after it, sound only because the body is straight-line. Callee's module must be the caller's, and its baked builtins are retired by the CALLER's own module-version check (the callee's is never run). `calleeReg`: needed only if the body reads an upvalue, since `callee` is a SAMPLE closure at an indirect site -- one ObjFunction, many closures (`|x| x + step`), so its captured cells aren't necessarily the next call's. Constants/globals are safe from the sample since they belong to the function/module, not the closure. */
bool inlineGlobalCall(Emit *e, ObjFunction *caller, ObjClosure *callee,
                             unsigned argc, uint32_t callOff, int calleeReg) {
    if (e->noInline) return false;
    /* An inlined body's entries want x0..x8 (inlineOwnBank) and a split bank
     * is already using them, so the plan withholds the split from a body the
     * measuring pass saw inline. Refusing here as well is what makes that a
     * fact rather than an agreement between two passes: the worst this can do
     * is decline an inline the probe never took. */
    if (e->splitAt != 0) return false;
    ObjFunction *cfn = callee->fn;
    if (cfn->module != caller->module) return false;
    if (e->inlining) return false;             /* one level, no recursion */
    /* The inlined body's offsets are the callee's, so `inProtected` describes
     * the CALLER's regions throughout -- a `try` of the callee's own would go
     * unseen. inlinableBody's whitelist already refuses every opcode a handler
     * needs; this says so rather than relying on it. */
    if (cfn->exceptionCount > 0) return false;
    unsigned cidx = e->depth - argc - 1;
    unsigned maxSlot = 0;
    bool readsUpvalue = false;
    if (!inlinableBody(callee, argc, &maxSlot, &readsUpvalue)) return false;
    if (readsUpvalue && calleeReg < 0) return false;
    if (getenv("JAI_JIT_WHY")) {
        fprintf(stderr, "[jit] inlining %s\n",
                cfn->name ? cfn->name->chars : "<anon>");
    }

    /* Every argument has to be in a register, since that is where the body
     * will read its parameters from. */
    for (unsigned i = 0; i < argc; i++) {
        if (!holdsRegister(e->stack[cidx + 1u + i])) return false;
    }

    int savedSlot[JIT_MAX_SLOTS + 1];
    memcpy(savedSlot, e->inlSlot, sizeof savedSlot);
    for (unsigned i = 0; i <= JIT_MAX_SLOTS; i++) e->inlSlot[i] = -1;
    for (unsigned i = 0; i < argc; i++) e->inlSlot[1u + i] = (int)(cidx + 1u + i);

    /* No noteScratchClobber here. An inlined body cannot call -- inlinableBody
     * admits nothing that does -- so it destroys x0..x8 only by USING them,
     * which is not a clobber but an allocation: when the caller already owns
     * that bank the two share one numbering (see inlineOwnBank), and when it
     * does not, the inlined entries have x0..x8 to themselves as before.
     * Anything inside that really does call still reaches noteScratchClobber
     * on its own, and under scratchValues that declines the compile. */
    e->inlining     = true;
    e->inlDepth     = cidx + 1u + argc;
    e->inlPinned    = 0;
    e->inlValueBase = e->valueDepth;
    e->inlIp        = callOff;
    e->inlClosureReg = calleeReg;

    /* The callee's own offset map, so its offsets cannot land in the
     * caller's. Nothing reads it back -- there are no branches -- but
     * compileBody writes one entry per instruction either way. */
    int cmap[129], cdepths[129];
    for (int i = 0; i <= cfn->chunk.count; i++) { cmap[i] = -1; cdepths[i] = -1; }
    int *savedMap = e->offsetToInst, *savedDepths = e->offsetToDepth;
    unsigned savedCarry = e->fpCarryCount;
    uint32_t savedCurOffset = e->curOffset;
    unsigned savedInstDepth = e->instDepth;
    unsigned savedInstValue = e->instValueDepth;
    e->offsetToInst = cmap;
    e->offsetToDepth = cdepths;

    bool ok = compileBody(e, callee);

    e->offsetToInst = savedMap;
    e->offsetToDepth = savedDepths;
    e->fpCarryCount = savedCarry;
    e->curOffset = savedCurOffset;
    e->instDepth = savedInstDepth;
    e->instValueDepth = savedInstValue;

    if (!ok || e->failed) {
        /* Instructions have been written; there is no taking them back. The
         * whole compile is retried with inlining off, which is the same answer
         * the register budget already gets. */
        e->inlining = false;
        memcpy(e->inlSlot, savedSlot, sizeof savedSlot);
        gInlineFailed = true;
        e->failed = true;
        return false;
    }

    /* OP_RETURN left the result on top and everything the body pinned beneath
     * it. Both are read while `inlining` is still set, because that is what
     * says which bank they are in; only the result's new home belongs to the
     * caller. */
    unsigned rres;
    SlotKind kres;
    uint32_t rshape;
    ObjClass *rcls;
    if (e->depth <= cidx) { e->failed = true; return false; }
    rshape = e->stackShape[e->depth - 1];
    rcls   = e->stackClass[e->depth - 1];
    /* Read while `inlining` is still set, so this names the inlined bank's d
     * register; the caller's own is taken after it is cleared. */
    bool rfp = (e->fpLive & (1u << (e->valueDepth - 1))) != 0;
    unsigned rfpReg = rfp ? fpHeldIn(e, e->valueDepth - 1) : 0;
    if (rfp) {
        if (!popValueRaw(e, &rres, &kres)) { e->failed = true; return false; }
    } else if (!popValue(e, &rres, &kres)) { e->failed = true; return false; }
    /* Raw, because nothing reads these again: the body is over and its pinned
     * locals go with it, so materialising one costs an instruction whose
     * destination is dead. */
    while (e->depth > cidx) {
        if (holdsRegister(e->stack[e->depth - 1])) {
            unsigned r;
            if (!popValueRaw(e, &r, NULL)) { e->failed = true; return false; }
        } else {
            e->depth--;
        }
    }
    e->inlining = false;
    memcpy(e->inlSlot, savedSlot, sizeof savedSlot);

    if (!pushValue(e, kres, rshape, rcls)) { e->failed = true; return false; }
    if (rfp) {
        unsigned dd = fpRegAt(e, e->valueDepth - 1);
        if (dd != rfpReg) emit(e, jaiA64FmovDD(dd, rfpReg));
        fpClaim(e, e->valueDepth - 1);
    } else {
        unsigned dst = pushReg(e) - 1;
        if (dst != rres) emit(e, jaiA64MovX(dst, rres));
    }
    e->inlined = true;
    return true;
}

/* What a descriptor call does with its result, once the arguments are consumed:
 * the predicted kind types the entry pushed for it, the tag that actually comes
 * back is checked, and a surprise deopts to the instruction AFTER the call --
 * which has happened and must not happen twice.
 *
 * Shared by the global-call and module-call arms below. They differ only in
 * what they consume before it and in how the callee was resolved; from the
 * descriptor's `result` onwards there is nothing to tell them apart. */
static bool emitCallOutResult(Emit *e, SlotKind rk, uint32_t rshape,
                              ObjClass *rcls, uint32_t after, uint8_t robj) {
    if (!pushValue(e, rk, rshape, rcls)) return false;
    /* A PREDICTION recorded on the entry, not a guard: every consumer of a
     * SLOT_OBJ checks Obj.type for itself before it reads anything, so this
     * only ever chooses which guard to emit and never deletes one. */
    if (retObjTypeOn() && rk == SLOT_OBJ && e->depth > 0) {
        e->stackObjType[e->depth - 1] = robj;
    }

    unsigned rat = e->descOffset + (unsigned)offsetof(JitCallDesc, result);
    if (rk == SLOT_MAYBE_INST) {
        emitMaybeInstResult(e, pushReg(e) - 1, rat, rshape, after);
        return true;
    }
    unsigned wantTag = rk == SLOT_INT   ? VAL_INT
                     : rk == SLOT_FLOAT ? VAL_FLOAT
                     : rk == SLOT_BOOL  ? VAL_BOOL
                     : rk == SLOT_NULL  ? VAL_NULL
                                        : VAL_OBJ;
    emit(e, jaiA64LdrW(JIT_SCRATCH_A, 31, rat));
    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, wantTag));
    branchOnDeoptAt(e, JAI_A64_NE, after, true);
    /* A null carries no payload worth loading, but the register still stands
     * for the entry and a deopt materialises it, so it gets a defined zero
     * rather than whatever the descriptor happened to leave behind. */
    if (rk == SLOT_NULL) emit(e, jaiA64MovzX(pushReg(e) - 1, 0, 0));
    /* A byte, not a word: BOOL_VAL writes the union's `bool` member and leaves
     * the other seven bytes of the payload indeterminate, so a 64-bit load
     * brings back whatever the slot held before. The register stands for a
     * bool from here on and everything downstream tests it against zero, so
     * those bytes read as true -- `values.map(|v| is_nan(v))` came back all
     * true over a list with no NaN in it. Every other descriptor return site
     * already splits the two; this one did not. */
    else if (rk == SLOT_BOOL) emit(e, jaiA64LdrByte(pushReg(e) - 1, 31, rat + 8));
    else emit(e, jaiA64LdrX(pushReg(e) - 1, 31, rat + 8));
    if (rk == SLOT_INST) {
        /* The tag says "an object", which is not "an instance of this class",
         * and every field offset resolved against the entry below assumes it
         * is. The object type is checked before `klass` is read for the same
         * reason it is at the invoke arm: VAL_OBJ covers every heap object and
         * a returned string's header is shorter than an instance's. */
        emit(e, jaiA64LdrW(JIT_SCRATCH_A, pushReg(e) - 1,
                           (unsigned)offsetof(Obj, type)));
        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_INSTANCE));
        branchOnDeoptAt(e, JAI_A64_NE, after, true);
        emit(e, jaiA64LdrX(JIT_SCRATCH_A, pushReg(e) - 1,
                           (unsigned)offsetof(ObjInstance, klass)));
        emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_A,
                           (unsigned)offsetof(ObjClass, shapeId)));
        emitConst64(e, JIT_SCRATCH_B, (int64_t)rshape);
        emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
        branchOnDeoptAt(e, JAI_A64_NE, after, true);
    } else if (rk == SLOT_LIST) {
        /* Same hazard as SLOT_INST above: a callee entered with another
         * specialisation runs interpreted and may return any type, so
         * VAL_OBJ alone does not prove the payload is a list before
         * downstream code reads ObjList's fields off it unguarded. */
        emit(e, jaiA64LdrW(JIT_SCRATCH_A, pushReg(e) - 1,
                           (unsigned)offsetof(Obj, type)));
        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_LIST));
        branchOnDeoptAt(e, JAI_A64_NE, after, true);
    }
    e->wroteHeap = true;
    return true;
}

/* A call to a global function that has itself compiled. Its return kind types
 * the result; the tag that actually comes back is checked, and a surprise
 * deopts to the instruction after the call, since the call has happened. */
bool emitGlobalCall(Emit *e, ObjFunction *caller, unsigned argc,
                           uint32_t callOff, uint32_t after) {
    unsigned cidx = e->depth - argc - 1;
    Value cv = e->stackSeen[cidx];
    if (!IS_CLOSURE(cv)) { e->whyNot = "callee vanished"; return false; }
    ObjFunction *cfn = AS_CLOSURE(cv)->fn;

    /* Straight to the callee's entry when everything jaiJitEnterFunc would
     * have checked can be checked here instead. Falling back rather than
     * declining matters: the descriptor path speaks a much wider language --
     * any argument kind, any module, a callee that writes -- and a call
     * through jaiCallValue still beats no compiled loop at all. */
    if (cfn->jitFunc != NULL) {
        const char *saved = e->whyNot;
        if (emitDirectCall(e, caller, cfn, cv, -1, cidx, argc, callOff,
                           after, false)) {
            return true;
        }
        if (e->failed) return false;   /* it had started emitting */
        e->whyNot = saved;
    }

    /* A callee with no compiled form of its own still knows what it has been
     * returning; see ObjFunction::obsReturnKind and the twin case at OP_INVOKE.
     * A recursive function is the ordinary way to reach this -- the loop being
     * compiled is inside the very function the call names, so there is nothing
     * for `jitReturnKind` to have been written by yet. */
    /* Observed first, compiled kind second -- the order the sibling site at
     * emitGlobalCall spells out, and which this one did not have.
     *
     * `jitFunc != NULL` does not make `jitReturnKind` a fact; it is stored
     * unconditionally at the end of a compile, so a body that compiled a
     * PREFIX and took the unarmed path before any OP_RETURN advertises the
     * SLOT_INT a zeroed Emit starts on. That is not hypothetical here:
     * `_is_ident_start` in the lexer walks only to OP_GET_GLOBAL at offset 0
     * -- a cold `throw` on its first instruction -- so it compiles nothing and
     * still claims to return an int. `_is_ident_cont`, which is
     * `_is_ident_start(c) or _is_digit(c)`, then declined on "a branch on a
     * int, not a bool", and `_ident_run_end`'s OSR loop retried it eighty
     * times waiting for a function that could never compile.
     *
     * A refusal is a chain, and this was three links of one. */
    SlotKind rk = SLOT_NULL;
    uint32_t rshape = 0;
    ObjClass *rcls = NULL;
    uint8_t robj = 0;
    bool haveKind = observedReturnKind(cfn, &rk, &rshape, &robj);
    if (!haveKind && cfn->jitFunc != NULL) {
        rk = (SlotKind)cfn->jitReturnKind;
        rshape = cfn->jitReturnShape;
        haveKind = true;
    }
    if (haveKind && (rk == SLOT_INST || rk == SLOT_MAYBE_INST) &&
        (rshape == 0 || !jaiClassForShape(rshape, &rcls) || rcls == NULL)) {
        e->whyNot = "callee's return class not on record";
        return false;
    }
    if (!haveKind ||
        (rk != SLOT_INT && rk != SLOT_FLOAT && rk != SLOT_BOOL &&
         rk != SLOT_INST && rk != SLOT_MAYBE_INST && rk != SLOT_LIST &&
         rk != SLOT_OBJ && rk != SLOT_NULL)) {
        e->whyNot = "callee's return kind not usable";
        return false;
    }

    if (!emitDescriptor(e, cv, e->depth - argc, argc, (void *)&jitCallOut)) {
        return false;
    }
    for (unsigned i = 0; i < argc; i++) {
        unsigned r;
        if (!popValue(e, &r, NULL)) return false;
    }
    if (e->depth == 0 || e->stack[e->depth - 1] != SLOT_FUNC) return false;
    e->depth--;
    return emitCallOutResult(e, rk, rshape, rcls, after, robj);
}

/* `math.sqrt(x)` is not a method call. It is a global call whose callee is
 * resolved through ANOTHER module, so what was missing was never the call
 * machinery -- it is the pinning.
 *
 * Two guards, in this order, both before anything is consumed so a miss resumes
 * at the invoke with the receiver and the arguments untouched:
 *
 *   THIS module. OP_GET_GLOBAL loads `math` by address behind a VAL_OBJ tag
 *   guard, and the arm's own object-type guard would only prove "some module",
 *   so the receiver is compared against the ObjModule this compiled against.
 *   Everything below -- the version word's address, the resolved closure --
 *   belongs to that one module.
 *
 *   STILL this binding. ObjModule::version is the counter for a memoised
 *   global VALUE or a resolved callee, and it is the one that moves when
 *   `math.sqrt = f` overwrites the member (jaiModuleSet bumps it because a
 *   closure is not inert). `globals.keyVersion` is the WRONG counter here and
 *   would be silent: overwriting an existing key never moves its entry, which
 *   is the whole reason keyVersion exists. Guarding it per call rather than
 *   at entry also covers a rebinding from inside the loop, which the entry
 *   check by itself does not -- see the note at OP_SET_GLOBAL.
 *
 * The callee must have compiled. The standing warning at OP_GET_GLOBAL says
 * admitting a callee whose jitFunc is NULL MISCOMPILES for a reason nobody has
 * written down; this arm does not cross it, and pays nothing for that --
 * `math.sqrt` compiles long before any loop calling it does.
 *
 * The receiver is dropped rather than passed: a module is not an argument. */
bool emitModuleCall(Emit *e, ObjModule *m, Value calleeVal,
                           unsigned ridx, unsigned argc, uint32_t after) {
    ObjFunction *cfn = AS_CLOSURE(calleeVal)->fn;
    if (cfn->jitFunc == NULL) {
        return subWhy(e, "a module member that has not compiled");
    }
    /* WHICH RECORD SAYS WHAT COMES BACK, and it is not the obvious one.
     * `jitFunc != NULL` does NOT make `jitReturnKind` a fact: it is stored
     * unconditionally at the end of a compile, so a body whose walk never
     * reached an OP_RETURN -- one that compiled a prefix and bails -- leaves it
     * at the SLOT_INT a zeroed Emit starts on. math.sqrt and math.sin are both
     * exactly that (`sawReturn=0`, `obsReturnKind=float`), so trusting the
     * compiled kind here predicted int, the tag guard below failed on the first
     * call, and the loop left compiled code every iteration: p27 did not move at
     * all. The interpreter's own per-callee record is a measured fact and is
     * asked first; the compiled kind stands in only when there is none.
     *
     * Either way it is a prediction, and the tag guard after the call is what
     * makes it sound -- the same contract emitGlobalCall states. */
    SlotKind rk = SLOT_NULL;
    uint32_t rshape = 0;
    ObjClass *rcls = NULL;
    uint8_t robj = 0;
    if (!observedReturnKind(cfn, &rk, &rshape, &robj)) {
        rk = (SlotKind)cfn->jitReturnKind;
        rshape = cfn->jitReturnShape;
    }
    if (rk != SLOT_INT && rk != SLOT_FLOAT && rk != SLOT_BOOL &&
        rk != SLOT_INST && rk != SLOT_MAYBE_INST && rk != SLOT_LIST &&
        rk != SLOT_OBJ && rk != SLOT_NULL) {
        return subWhy(e, "a module member's return kind (%d)", (int)rk);
    }
    if ((rk == SLOT_INST || rk == SLOT_MAYBE_INST) &&
        (rshape == 0 || !jaiClassForShape(rshape, &rcls) || rcls == NULL)) {
        return subWhy(e, "a module member's return class is not on record");
    }
    /* Every entry this consumes, receiver included, has to be one the
     * descriptor can pass -- asked HERE rather than left to emitDescriptor,
     * which refuses only after the guards below have been emitted and would
     * turn a graceful decline into a whole-body one. It also settles the
     * register arithmetic: the receiver is named by counting back from the top
     * of the value bank, which is the same thing as counting back from the top
     * of the model only while every entry between them holds a register, and
     * every kind admitted here does. */
    for (unsigned i = 0; i <= argc; i++) {
        SlotKind k = e->stack[ridx + i];
        if (k != SLOT_INT && k != SLOT_FLOAT && k != SLOT_BOOL &&
            k != SLOT_INST && k != SLOT_LIST && k != SLOT_OBJ &&
            k != SLOT_ITER && k != SLOT_MAYBE_INST) {
            return subWhy(e, "a module call over an entry of kind %d", (int)k);
        }
    }
    /* Past here the guards are emitted, so a later refusal stops the compile
     * rather than falling back onto a half-written instruction stream. */

    settleAll(e);
    unsigned rRecv = valueXReg(e, e->valueDepth - argc - 1);
    emitConst64(e, JIT_SCRATCH_C, (int64_t)(uintptr_t)m);
    emit(e, jaiA64SubsXReg(31, rRecv, JIT_SCRATCH_C));
    branchOnDeopt(e, JAI_A64_NE);

    emitConst64(e, JIT_SCRATCH_D, (int64_t)(uintptr_t)&m->version);
    emit(e, jaiA64LdrW(JIT_SCRATCH_C, JIT_SCRATCH_D, 0));
    emitConst64(e, JIT_SCRATCH_B, (int64_t)(uint32_t)m->version);
    emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_C, JIT_SCRATCH_B));
    branchOnDeopt(e, JAI_A64_NE);

    if (!emitDescriptor(e, calleeVal, ridx + 1, argc, (void *)&jitCallOut)) {
        return false;
    }
    for (unsigned i = 0; i <= argc; i++) {
        unsigned r;
        if (!popValue(e, &r, NULL)) return false;
    }
    return emitCallOutResult(e, rk, rshape, rcls, after, robj);
}

/* What `__prim__.f64_sqrt` and its kin answer, since unlike a compiled
 * closure a native carries no jitReturnKind/observedReturnKind to ask --
 * emitModuleCall's whole "which record says what comes back" problem does not
 * apply, because there is no record. Scoped to natives reached ONLY through a
 * MODULE receiver (emitModuleNativeCall's one caller): every row here is one
 * of lib/std/math.jai's own `__prim__.f64_*` callees, registered in
 * builtins_math.c, none of which allocates or can throw past a domain check
 * the CALLER (math.jai) already made before reaching the primitive -- see
 * emitModuleNativeCall's own comment for how `sqrt`'s negative-argument raise
 * stays correct despite that.
 *
 * A row missing here is a decline, not a wrong answer -- the tag guard
 * downstream only matters for a row that IS present, same contract
 * kNativeResults states above. frexp/modf are deliberately absent: both
 * return `tuple[float, int]`, a kind this arm has no row shape for. */
typedef struct {
    const char *name;
    unsigned    argc;
    SlotKind    kind;
} PrimNativeResult;

static const PrimNativeResult kPrimNativeResults[] = {
    { "f64_sqrt",     1, SLOT_FLOAT },
    { "f64_exp",      1, SLOT_FLOAT },
    { "f64_log",      1, SLOT_FLOAT },
    { "f64_log2",     1, SLOT_FLOAT },
    { "f64_log10",    1, SLOT_FLOAT },
    { "f64_sin",      1, SLOT_FLOAT },
    { "f64_cos",      1, SLOT_FLOAT },
    { "f64_tan",      1, SLOT_FLOAT },
    { "f64_asin",     1, SLOT_FLOAT },
    { "f64_acos",     1, SLOT_FLOAT },
    { "f64_atan",     1, SLOT_FLOAT },
    { "f64_atan2",    2, SLOT_FLOAT },
    { "f64_sinh",     1, SLOT_FLOAT },
    { "f64_cosh",     1, SLOT_FLOAT },
    { "f64_tanh",     1, SLOT_FLOAT },
    { "f64_asinh",    1, SLOT_FLOAT },
    { "f64_acosh",    1, SLOT_FLOAT },
    { "f64_atanh",    1, SLOT_FLOAT },
    { "f64_floor",    1, SLOT_FLOAT },
    { "f64_ceil",     1, SLOT_FLOAT },
    { "f64_trunc",    1, SLOT_FLOAT },
    { "f64_round",    1, SLOT_FLOAT },
    { "f64_fmod",     2, SLOT_FLOAT },
    { "f64_pow",      2, SLOT_FLOAT },
    { "f64_hypot",    2, SLOT_FLOAT },
    { "f64_copysign", 2, SLOT_FLOAT },
    { "f64_ldexp",    2, SLOT_FLOAT },
    { "f64_erf",      1, SLOT_FLOAT },
    { "f64_gamma",    1, SLOT_FLOAT },
    { "f64_lgamma",   1, SLOT_FLOAT },
    { "f64_is_nan",    1, SLOT_BOOL },
    { "f64_is_inf",    1, SLOT_BOOL },
    { "f64_is_finite", 1, SLOT_BOOL },
};

static bool primNativeResultKind(const char *nm, unsigned argc, SlotKind *k) {
    for (size_t i = 0; i < sizeof kPrimNativeResults / sizeof kPrimNativeResults[0]; i++) {
        if (kPrimNativeResults[i].argc == argc &&
            strcmp(kPrimNativeResults[i].name, nm) == 0) {
            *k = kPrimNativeResults[i].kind;
            return true;
        }
    }
    return false;
}

/* `__prim__.f64_sqrt(x)` and its kin -- a NATIVE reached through a MODULE
 * receiver, the other half of what emitModuleCall does for a Jaithon-written
 * one (`math.sqrt`, wrapping this very call). Same shape, same reason: a
 * member of another namespace, resolved at compile time, called with the
 * receiver DROPPED -- resolveInvokeTarget's IS_MODULE arm (vm.c) hands back
 * jaiBuiltinMethod's raw result with `isMethod` left false, so invokeCallable
 * runs it as a PLAIN call, not a method call. jaiModuleMethod (module_methods.c)
 * returns the ObjNative straight out of `m->globals` with no bound wrapper --
 * `moduleExposes` succeeds on the first check because `__prim__`'s
 * `exports.count` is 0 (nothing ever declares an export list for a namespace
 * jaiDefineNative built, so every member reads as exposed) -- so `onative` at
 * the call site already IS the plain native, never a bound one to unwrap.
 *
 * jitCallOut's jaiCallValue -> invokeCallable dispatches on OBJ_NATIVE with
 * `args[0]` as the first REAL argument and no receiver slot at all (vm.c's
 * `case OBJ_NATIVE:` in invokeCallable) -- exactly the descriptor
 * emitDescriptor already builds from `ridx + 1, argc` for emitModuleCall's
 * closures, so the same call-out helper reaches a native correctly with no
 * changes of its own. The only work here is specific to a NATIVE: what it
 * returns (kPrimNativeResults, since there is no jitReturnKind to ask) and the
 * guards, which are NOT the same two as emitModuleCall's.
 *
 * THE RECEIVER IDENTITY GUARD emitModuleCall emits is *provably* redundant
 * here and is skipped: `math`'s receiver register is loaded from a JaiEntry
 * (globalSlot's "value case" arm), a genuinely runtime-variable location an
 * import could rebind, so comparing it against the ObjModule compiled against
 * is live work. `__prim__`'s register is instead loaded by
 * `emitConst64(e, dst, ...)` directly in OP_GET_GLOBAL's globalNamespace
 * branch -- a compile-time CONSTANT baked into the instruction stream, which
 * cannot hold anything else at run time by construction, so re-checking it
 * against itself would prove nothing a bug in the emitter could not also get
 * wrong in the omitted check.
 *
 * `m->version` IS kept, and unlike the identity check it is NOT redundant:
 * `mod.attr = v` is real syntax for any module receiver (jaiSetProperty's
 * IS_MODULE arm, vm.c) and `__prim__` is reachable as a bare identifier --
 * lib/std/math.jai names it in the open -- so `__prim__.f64_sqrt = something`
 * is something a running program could actually do. jaiModuleSet bumps
 * `m->version` on exactly that kind of write (ObjNative is not
 * jaiValueIsInertGlobal), which is what retires this compiled form if it
 * happens after the bake.
 *
 * `sqrt`'s own domain check (`if x < 0.0 { throw ValueError(...) }`,
 * lib/std/math.jai:217) is ordinary Jaithon ahead of this call and compiles
 * on its own merits -- ints/floats/branches/throw are all arms this tier
 * already has -- so it is not this arm's problem to solve; declining THIS
 * call alone (a too-wide argc, an unknown name) still leaves the raise
 * compiled, and only the call after it falls back to the interpreter via
 * emitUnarmedDeopt. `f64_sqrt` raising its OWN domain error for a negative
 * input it should never see is still reachable and still correct either way:
 * callNativeAt raises through the ordinary exception path jitCallOut's
 * `raiseExitAllowed` branch already handles, message and all -- nothing about
 * going through a descriptor changes what the native itself decides to
 * raise. */
/* Whether the `__prim__` read at `off` is paired with an OP_INVOKE this tier
 * can actually compile. If it is not, the namespace is left unresolved so the
 * read falls to the unarmed path exactly as it did before this arm existed --
 * which for a call the tier cannot emit is strictly better than resolving it.
 *
 * WHY THIS EXISTS. `span` ends in `__prim__.fill_span(...)` with ten arguments
 * and `fill_convex` with eleven. While `__prim__` had no arm the walk skipped
 * receiver, pushes and invoke as ONE unarmed block and span's prologue
 * compiled; resolving the namespace made the walk continue into those pushes
 * and hit a wall it cannot pass -- the argc cap, then the result-kind
 * whitelist, then the register budget, each a hard whole-body decline. span's
 * interpreted work DOUBLED, 5,821,736 to 10,867,344, from arming a
 * neighbouring opcode.
 *
 * PAIRING IS THE WHOLE DIFFICULTY, and an earlier attempt got it wrong: it
 * took the FIRST OP_INVOKE after the read, which is not the paired one
 * whenever an argument expression contains its own method call. When that
 * inner invoke happened to be whitelisted the lookahead said yes and the OUTER
 * one declined the body -- the very thing it was written to prevent.
 *
 * `chunkDepth` settles it exactly. It is the operand-stack depth BEFORE each
 * instruction, so the invoke paired with this read is the one that consumes
 * back to the depth the read started from: `chunkDepth[p] - (argc + 1) ==
 * chunkDepth[off]`. A nested invoke inside an argument sits deeper and cannot
 * match. Where the table is unavailable -- OSR, inlining -- the answer is no,
 * which keeps the old unarmed behaviour rather than guessing. */
bool primInvokePairFits(const Emit *e, const Chunk *chunk, uint32_t off) {
    if (e->chunkDepth == NULL || e->osr || e->inlining) return false;
    if (off >= (uint32_t)e->chunkDepthCount) return false;
    const int base = e->chunkDepth[off];
    const uint8_t *code = chunk->code;
    uint32_t p = off + 6;
    while (p < (uint32_t)chunk->count) {
        if (p >= (uint32_t)e->chunkDepthCount) return false;
        if (code[p] == OP_INVOKE) {
            unsigned argc = code[p + 4];
            if (e->chunkDepth[p] - (int)(argc + 1u) == base) {
                if (argc + 1u > (unsigned)JIT_MAX_ARGS_OUT) return false;
                uint32_t nameIdx = jaiReadU24(code + p + 1);
                if (nameIdx >= (uint32_t)chunk->constants.count) return false;
                Value nm = chunk->constants.data[nameIdx];
                if (!IS_STRING(nm)) return false;
                SlotKind rk;
                return primNativeResultKind(AS_STRING(nm)->chars, argc, &rk);
            }
        }
        unsigned len = instructionLength(chunk, p);
        if (len == 0) return false;
        p += len;
    }
    return false;
}

bool emitModuleNativeCall(Emit *e, ObjModule *m, Value calleeVal,
                                 unsigned ridx, unsigned argc, uint32_t after) {
    ObjNative *nat = AS_NATIVE(calleeVal);
    SlotKind rk;
    if (!primNativeResultKind(nat->name != NULL ? nat->name->chars : "",
                              argc, &rk)) {
        return subWhy(e, "`%s.%s`'s result kind is not on record",
                      m->name != NULL ? m->name->chars : "?",
                      nat->name != NULL ? nat->name->chars : "?");
    }
    for (unsigned i = 0; i <= argc; i++) {
        SlotKind k = e->stack[ridx + i];
        if (k != SLOT_INT && k != SLOT_FLOAT && k != SLOT_BOOL &&
            k != SLOT_INST && k != SLOT_LIST && k != SLOT_OBJ &&
            k != SLOT_ITER && k != SLOT_MAYBE_INST) {
            return subWhy(e, "a module call over an entry of kind %d", (int)k);
        }
    }
    /* Past here the guard is emitted, so a later refusal stops the compile
     * rather than falling back onto a half-written instruction stream. */

    settleAll(e);
    emitConst64(e, JIT_SCRATCH_D, (int64_t)(uintptr_t)&m->version);
    emit(e, jaiA64LdrW(JIT_SCRATCH_C, JIT_SCRATCH_D, 0));
    emitConst64(e, JIT_SCRATCH_B, (int64_t)(uint32_t)m->version);
    emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_C, JIT_SCRATCH_B));
    branchOnDeopt(e, JAI_A64_NE);

    if (!emitDescriptor(e, calleeVal, ridx + 1, argc, (void *)&jitCallOut)) {
        return false;
    }
    for (unsigned i = 0; i <= argc; i++) {
        unsigned r;
        if (!popValue(e, &r, NULL)) return false;
    }
    /* A __prim__ native's result kind comes from the whitelist, not from a
     * callee record, so there is no observed object type to carry. */
    return emitCallOutResult(e, rk, 0, NULL, after, 0);
}

/* `Klass.static_method(args)` -- an OP_INVOKE whose receiver is a CLASS. The
 * same job emitModuleCall does for `math.sqrt(x)`, and for the same reason: a
 * member of another namespace, resolved at compile time, called with the
 * receiver DROPPED. The interpreter drops it too. resolveInvokeTarget (vm.c)
 * leaves the class in slot 0 and reports `isMethod == false`, so the call goes
 * through invokeCallable and the closure's first parameter is the first
 * ARGUMENT, not the class -- which is exactly what jitCallOut does with a
 * descriptor holding argc arguments and no receiver.
 *
 * Three things differ from the module case, and only the third is machinery.
 *
 *   THE RECEIVER NEEDS NO GUARD. A module receiver is loaded from a global by
 *   address behind a bare VAL_OBJ tag check, so emitModuleCall has to compare
 *   it against the one ObjModule it compiled against. A class receiver is not
 *   in a register at all: SLOT_CLASS holds nothing (holdsRegister says so) and
 *   the ObjClass came from OP_GET_GLOBAL's `globalClass` arm, which resolves it
 *   BY VALUE and is retired wholesale by the module version check at entry if
 *   the name is rebound. There is nothing here that could be a different class
 *   at run time than it was at compile time.
 *
 *   THE MEMBER IS RESOLVED OUT OF `klass->statics`, DIRECTLY. Not through
 *   jaiBuiltinMethod, and not through anything that can hand back a BoundMethod
 *   to unwrap: unwrapping one and then dropping the receiver as this arm does
 *   loses both halves and calls an unbound closure with the first real argument
 *   sitting where `self` belongs. Only an IS_CLOSURE value straight out of the
 *   table is admitted. `klass->methods` is deliberately NOT consulted -- see
 *   the call site, which declines an instance method named through the class.
 *
 *   WHAT RETIRES THE BAKED CALLEE is the BINDING itself, re-read from the
 *   statics entry on every call and compared against the closure this site was
 *   compiled against. `Klass.name = v` reaches jaiSetProperty's IS_CLASS arm,
 *   which requires the key to be present already and then overwrites in place,
 *   so the entry never moves and reading it back asks the direct question: is
 *   this still what I compiled for.
 *
 *   It was a COUNTER first -- `klass->statics.version`, which tableSetHashed
 *   bumps on every value write -- and that was unsound. The field is uint32_t
 *   (table.h) and nothing filters the bump the way jaiModuleSet filters
 *   ObjModule::version through jaiValueIsInertGlobal, so an ordinary
 *   `Klass.counter = n` loop drives it at 51M writes/sec, measured. Land the
 *   count exactly 2^32 on from the bake and the guard reads the value it baked
 *   while the binding has changed: `Box.make` rebound after 2^32 writes
 *   answered 400008, against 2000000 from the interpreter, from the arm
 *   switched off, and from a control one write short. That is about 84 seconds
 *   of a loop any program might contain, not an unreachable corner.
 *
 *   Reading the binding is also strictly less trigger-happy than the counter,
 *   which retired the callee whenever any OTHER static of the same class was
 *   written. It costs 1.8% of the win (0.551s -> 0.561s on a 32M-call probe
 *   against 1.130s with the arm off).
 *
 * `keyVersion` is still needed, for a different job: it makes the entry ADDRESS
 * trustworthy. It moves on rehash, delete and clear -- never on an overwrite
 * (table.c: insertAt bumps it only for a NEW key) -- so unlike `version` a
 * running program cannot drive it. It is the guard the static-FIELD arm stands
 * on, baked per site here rather than through e->staticsTable so that a body
 * naming two classes still compiles both. */
bool emitClassCall(Emit *e, ObjClass *klass, JaiEntry *slot,
                          Value calleeVal, unsigned ridx, unsigned argc,
                          uint32_t after) {
    ObjFunction *cfn = AS_CLOSURE(calleeVal)->fn;
    if (cfn->jitFunc == NULL) {
        return subWhy(e, "a static that has not compiled");
    }
    /* Which record says what comes back: emitModuleCall's finding, and it is
     * not the obvious one. `jitFunc != NULL` does NOT make `jitReturnKind` a
     * fact -- it is stored unconditionally at the end of a compile, so a body
     * whose walk never reached an OP_RETURN leaves it at the SLOT_INT a zeroed
     * Emit starts on. The interpreter's own per-callee record is a measured
     * fact and is asked first; the compiled kind stands in only when there is
     * none. Either way it is a prediction, and emitCallOutResult's tag guard
     * after the call is what makes it sound. */
    SlotKind rk = SLOT_NULL;
    uint32_t rshape = 0;
    ObjClass *rcls = NULL;
    uint8_t robj = 0;
    if (!observedReturnKind(cfn, &rk, &rshape, &robj)) {
        rk = (SlotKind)cfn->jitReturnKind;
        rshape = cfn->jitReturnShape;
    }
    if (rk != SLOT_INT && rk != SLOT_FLOAT && rk != SLOT_BOOL &&
        rk != SLOT_INST && rk != SLOT_MAYBE_INST && rk != SLOT_LIST &&
        rk != SLOT_OBJ && rk != SLOT_NULL) {
        return subWhy(e, "a static's return kind (%s)", slotKindName(rk));
    }
    if ((rk == SLOT_INST || rk == SLOT_MAYBE_INST) &&
        (rshape == 0 || !jaiClassForShape(rshape, &rcls) || rcls == NULL)) {
        return subWhy(e, "a static's return class is not on record");
    }
    /* The ARGUMENTS only. The receiver is skipped where emitModuleCall checks
     * it, because a class entry holds no register and emitDescriptorStatus
     * already knows how to bake one -- but it is never passed here, so even
     * that does not arise. Asked HERE rather than left to emitDescriptor, which
     * refuses only after the guard below has been emitted and would turn a
     * graceful decline into a whole-body one. */
    for (unsigned i = 1; i <= argc; i++) {
        SlotKind k = e->stack[ridx + i];
        if (k != SLOT_INT && k != SLOT_FLOAT && k != SLOT_BOOL &&
            k != SLOT_INST && k != SLOT_LIST && k != SLOT_OBJ &&
            k != SLOT_ITER && k != SLOT_MAYBE_INST) {
            return subWhy(e, "a static call over an argument of kind %s",
                          slotKindName(k));
        }
    }
    /* Past here the guard is emitted, so a later refusal stops the compile
     * rather than falling back onto a half-written instruction stream. */

    settleAll(e);
    emitConst64(e, JIT_SCRATCH_D,
                (int64_t)(uintptr_t)&klass->statics.keyVersion);
    emit(e, jaiA64LdrW(JIT_SCRATCH_C, JIT_SCRATCH_D, 0));
    emitConst64(e, JIT_SCRATCH_B, (int64_t)(uint32_t)klass->statics.keyVersion);
    emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_C, JIT_SCRATCH_B));
    branchOnDeopt(e, JAI_A64_NE);

    /* The tag before the pointer, so a static rebound to an int whose payload
     * happened to equal the closure's address is not called as if it were the
     * closure.
     *
     * Comparing against the LIVE binding is also what makes address recycling
     * harmless rather than dangerous. If the old closure were collected and a
     * new object took its address, the object bound NOW is the one at that
     * address, and emitDescriptor's callee slot goes through jaiCallValue,
     * which dispatches on the value dynamically -- including raising, if what
     * is bound there is not callable. */
    emitConst64(e, JIT_SCRATCH_D, (int64_t)(uintptr_t)slot);
    emit(e, jaiA64LdrW(JIT_SCRATCH_C, JIT_SCRATCH_D,
                       (unsigned)offsetof(JaiEntry, value)));
    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_C, VAL_OBJ));
    branchOnDeopt(e, JAI_A64_NE);
    emit(e, jaiA64LdrX(JIT_SCRATCH_C, JIT_SCRATCH_D,
                       (unsigned)offsetof(JaiEntry, value) + 8u));
    emitConst64(e, JIT_SCRATCH_B, (int64_t)(uintptr_t)AS_OBJ(calleeVal));
    emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_C, JIT_SCRATCH_B));
    branchOnDeopt(e, JAI_A64_NE);

    if (!emitDescriptor(e, calleeVal, ridx + 1, argc, (void *)&jitCallOut)) {
        return false;
    }
    for (unsigned i = 0; i < argc; i++) {
        unsigned r;
        if (!popValue(e, &r, NULL)) return false;
    }
    /* The receiver held no register, so dropping it is the whole of popping it
     * -- popValue would refuse it via holdsRegister. Same as the static-field
     * arm at OP_GET_FIELD. */
    if (e->depth == 0 || e->stack[e->depth - 1] != SLOT_CLASS) return false;
    e->depth--;
    return emitCallOutResult(e, rk, rshape, rcls, after, robj);
}

/* Emit a method's body directly, when that body is one expression.
 *
 * Deliberately narrow: no jumps, no stores, no calls, only field reads of its
 * own parameters and int or float arithmetic. Those restrictions are what make
 * a second walker over the callee's bytecode safe to write -- with no branches
 * there is no offset map to keep, and with no stores there is nothing to undo
 * if a guard inside it deoptimises to the call site.
 *
 * Reached through inlineMethod, which is what puts the model back when this
 * declines -- see there. */
static bool inlineMethodWalk(Emit *e, ObjClosure *closure, uint32_t nameIdx,
                             unsigned argc, int callOff) {
    if (e->noInline) return false;
    if (e->splitAt != 0) return false;   /* see inlineGlobalCall */
    unsigned ridx = e->depth - argc - 1;
    ObjClass *rcls = e->stackClass[ridx];
    if (rcls == NULL) return false;
    ObjFunction *cfn = closure->fn;
    if (nameIdx >= (uint32_t)cfn->chunk.constants.count) return false;
    Value mname = cfn->chunk.constants.data[nameIdx];
    if (!IS_STRING(mname)) return false;
    Value method;
    if (!jaiClassFindMethod(rcls, AS_STRING(mname), &method)) return false;
    if (!IS_CLOSURE(method)) return false;
    ObjFunction *mfn = AS_CLOSURE(method)->fn;
    if (mfn->exceptionCount > 0) return false;   /* see inlineGlobalCall */
    if (mfn->arity != argc || mfn->defaultCount != 0) return false;
    if (mfn->flags & (FN_VARIADIC | FN_KWREST | FN_INIT)) return false;
    if (mfn->upvalueCount != 0) return false;
    if (mfn->chunk.count > 96) return false;

    unsigned inReg[JIT_MAX_ARGS_OUT + 1];
    Value    inSeen[JIT_MAX_ARGS_OUT + 1];
    ObjClass *inCls[JIT_MAX_ARGS_OUT + 1];
    for (unsigned i = 0; i <= argc; i++) {
        unsigned idx = ridx + i;
        if (!holdsRegister(e->stack[idx])) return false;
        inReg[i]  = valueBankReg(e, idx - (e->depth - e->valueDepth));
        inSeen[i] = e->stackSeen[idx];
        inCls[i]  = e->stackClass[idx];
    }

    /* A dry walk first: nothing is emitted until the whole body is known to
     * be expressible, because a half-inlined body cannot be taken back. */
    const uint8_t *c = mfn->chunk.code;
    int n = mfn->chunk.count;
    for (int pass = 0; pass < 2; pass++) {
        int depth0 = (int)e->depth;
        for (int o = 0; o < n;) {
            uint8_t op = c[o];
            if (op == OP_GET_FIELD_LOCAL) {
                unsigned slot = jaiReadU16(c + o + 1);
                uint32_t nidx = jaiReadU24(c + o + 3);
                if (slot > argc) return false;
                if (e->stack[ridx + slot] != SLOT_INST) return false;
                if (nidx >= (uint32_t)mfn->chunk.constants.count) return false;
                Value fname = mfn->chunk.constants.data[nidx];
                if (!IS_STRING(fname)) return false;
                const FieldInfo *fi =
                    jaiClassFieldInfo(inCls[slot], AS_STRING(fname));
                if (fi == NULL || fi->isStatic) return false;
                if (!IS_INSTANCE(inSeen[slot])) return false;
                ObjInstance *si = AS_INSTANCE(inSeen[slot]);
                if (fi->slot >= si->fieldCount) return false;
                Value fv = si->fields[fi->slot];
                SlotKind fk; unsigned ftag;
                if (IS_INT(fv))        { fk = SLOT_INT;   ftag = VAL_INT; }
                else if (IS_FLOAT(fv)) { fk = SLOT_FLOAT; ftag = VAL_FLOAT; }
                else return false;
                unsigned fbase = (unsigned)offsetof(ObjInstance, fields) +
                                 (unsigned)fi->slot * (unsigned)sizeof(Value);
                if (pass == 1) {
                    emit(e, jaiA64LdrW(JIT_SCRATCH_A, inReg[slot], fbase));
                    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, ftag));
                    branchOnDeoptAt(e, JAI_A64_NE, (uint32_t)callOff, false);
                }
                if (!pushValue(e, fk, 0, NULL)) return false;
                if (pass == 1) {
                    emit(e, jaiA64LdrX(pushReg(e) - 1, inReg[slot], fbase + 8));
                }
                o += 8;
                continue;
            }
            if (op == OP_ADD || op == OP_SUB || op == OP_MUL) {
                unsigned rb, ra; SlotKind kb, ka;
                if (!popValue(e, &rb, &kb)) return false;
                if (!popValue(e, &ra, &ka)) return false;
                if (ka != kb) return false;
                if (ka != SLOT_INT && ka != SLOT_FLOAT) return false;
                if (!pushValue(e, ka, 0, NULL)) return false;
                unsigned rd = pushReg(e) - 1;
                if (pass == 1) {
                    if (ka == SLOT_FLOAT) {
                        emit(e, jaiA64FmovDX(JIT_FSCRATCH_A, ra));
                        emit(e, jaiA64FmovDX(JIT_FSCRATCH_B, rb));
                        emit(e, op == OP_ADD
                                 ? jaiA64FaddD(JIT_FSCRATCH_A, JIT_FSCRATCH_A, JIT_FSCRATCH_B)
                             : op == OP_SUB
                                 ? jaiA64FsubD(JIT_FSCRATCH_A, JIT_FSCRATCH_A, JIT_FSCRATCH_B)
                                 : jaiA64FmulD(JIT_FSCRATCH_A, JIT_FSCRATCH_A, JIT_FSCRATCH_B));
                        emit(e, jaiA64FmovXD(rd, JIT_FSCRATCH_A));
                    } else if (op == OP_MUL) {
                        emit(e, jaiA64SmulhX(JIT_SCRATCH_A, ra, rb));
                        emit(e, jaiA64MulX(rd, ra, rb));
                        emit(e, jaiA64SubsXAsr(31, JIT_SCRATCH_A, rd, 63));
                        branchOnOverflow(e, 2u, JAI_A64_NE);
                    } else {
                        emit(e, op == OP_ADD ? jaiA64AddsX(rd, ra, rb)
                                             : jaiA64SubsXReg(rd, ra, rb));
                        branchOnOverflow(e, op == OP_ADD ? 0u : 1u, JAI_A64_VS);
                    }
                }
                o += 1;
                continue;
            }
            if (op == OP_RETURN) {
                if ((int)e->depth != depth0 + 1) return false;
                o += 1;
                if (o != n) return false;
                break;
            }
            return false;
        }
        if (pass == 0) {
            while ((int)e->depth > depth0) {
                unsigned r; if (!popValue(e, &r, NULL)) return false;
            }
        }
    }

    unsigned rres;
    SlotKind kres;
    if (!popValue(e, &rres, &kres)) return false;
    for (unsigned i = 0; i <= argc; i++) {
        unsigned r; if (!popValue(e, &r, NULL)) return false;
    }
    if (!pushValue(e, kres, 0, NULL)) return false;
    unsigned dst = pushReg(e) - 1;
    if (dst != rres) emit(e, jaiA64MovX(dst, rres));
    e->inlined = true;
    return true;
}

/* The model must be exactly where it was if the inline did not happen.
 *
 * inlineMethodWalk's dry pass pushes and pops as it reads the callee, and every
 * one of its two dozen refusals returns from the middle of that -- so on its
 * own it leaves the model as deep as the walk got. Its caller does NOT decline
 * when it declines: the OP_INVOKE arm falls through to the descriptor path,
 * which then names every later entry's register from an index that is too high
 * and, far worse, writes deopt records describing an operand stack the
 * interpreter does not have. `_crossings` in lib/std/gui/path.jai is the shape
 * that found this: `edge.crossing(y)` gets three instructions into `crossing`
 * before an OP_BIND stops the walk, so every deopt after it handed the
 * interpreter the receiver and the argument a second time and the next
 * instruction read a float where a list belonged.
 *
 * Unwinding here rather than at each `return false` is deliberate: there are
 * far too many of them to keep right by hand, and the dry pass's own tail
 * already pops back to its starting depth in exactly this way.
 *
 * A failure that has already emitted cannot be unwound at all -- the caller
 * would stack a second call sequence on top of half of this one -- so that
 * declines the compile instead. */
bool inlineMethod(Emit *e, ObjClosure *closure, uint32_t nameIdx,
                         unsigned argc, int callOff) {
    unsigned depth0 = e->depth;
    unsigned count0 = e->count;
    if (inlineMethodWalk(e, closure, nameIdx, argc, callOff)) return true;
    if (e->failed) return false;
    if (e->count != count0) { e->failed = true; return false; }
    while (e->depth > depth0) {
        unsigned r;
        if (!popValue(e, &r, NULL)) { e->failed = true; return false; }
    }
    /* Below where it started is not something an unwind can repair: the
     * entries are the caller's and their registers are gone. */
    if (e->depth != depth0) { e->failed = true; return false; }
    return false;
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
