/* jit_body.c -- the opcode walk: compileBody and the helpers only it uses. */
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

/* After an unconditional OP_LOOP/OP_JUMP there's no fall-through, so the linear walk can't carry the
 * preceding instruction's stack across the gap -- reconciles from a branch that targets this offset instead, accepting only a pure truncation (register entries are the top `valueDepth`, so popping from the top preserves every index below the join). */
/* The depth an emitted branch to `off` would reconcile the model to, or -1 for
 * none. Split out because emitUnarmedDeopt has to ask the question WITHOUT
 * answering it: a resume point it cannot reconcile is one it must not stop at.
 * A "no" here is usually the kinds disagreeing, not the depth -- the signature
 * carries both -- and that is exactly a join this tier cannot compile. */
static int reconcileDepth(const Emit *e, uint32_t off) {
    for (unsigned i = 0; i < e->fixupCount; i++) {
        if (e->fixups[i].targetOffset != off) continue;
        int want = e->fixups[i].depth;
        if (want < 0) continue;
        unsigned d = (unsigned)want & 0xfu;
        if (d > e->depth) continue;
        if ((int)stackSignatureAt(e, d) != want) continue;
        if (e->depth - d > e->valueDepth) continue;
        return (int)d;
    }
    return -1;
}

static void reconcileAfterUncond(Emit *e, uint32_t off) {
    int d = reconcileDepth(e, off);
    if (d < 0) return;
    e->valueDepth -= e->depth - (unsigned)d;
    e->depth = (unsigned)d;
}

/* `build` in binary_trees returns `null` on one path and `Node(..)` on another -- an instance merged
 * with a maybe-instance becomes a maybe-instance (shape survives only if both sides agree). Written once before and reverted when it made binary_trees 11x slower -- not this merge's fault: at the time a self-call rooted nothing and emitRootFill's operand-stack-to-register mapping was wrong, and `build` was the first body to hold a fresh allocation across an allocating self-call. Both bugs are now fixed. */
static bool mergeReturnKind(Emit *e, SlotKind k, uint32_t shape) {
    if (!e->sawReturn) {
        e->sawReturn = true; e->returnKind = k; e->returnShape = shape;
        return true;
    }
    if (e->returnKind == k) {
        if (e->returnShape != shape) e->returnShape = 0;
        return true;
    }
    bool nullable = (e->returnKind == SLOT_INST && k == SLOT_MAYBE_INST) ||
                    (e->returnKind == SLOT_MAYBE_INST && k == SLOT_INST);
    /* Named, because bare this was the whole of what a census said about
     * OP_RETURN: which two kinds a body cannot agree on is the entire question,
     * and an instance meeting a nullable instance is already merged above. */
    if (!nullable) {
        return subWhy(e, "a body returning both %s and %s",
                      slotKindName(e->returnKind), slotKindName(k));
    }
    if (e->returnShape != shape) e->returnShape = 0;
    e->returnKind = SLOT_MAYBE_INST;
    return true;
}

/* Opcodes that pull a float operand straight out of the FP bank; every other opcode sees the model
 * materialised as before. Adding an opcode here without also teaching it fpOperand is a MISCOMPILE, not a decline -- the whole risk of this design, and why the list stays short. */
static bool fpFastOp(uint8_t op) {
    switch (op) {
    case OP_ADD: case OP_SUB: case OP_MUL: case OP_DIV:
    case OP_ADD_BIND: case OP_SUB_BIND: case OP_MUL_BIND:
    case OP_GET_LOCAL: case OP_GET_LOCAL2:
    case OP_ADD_LOCALS: case OP_BIND: case OP_SET_LOCAL:
    /* These three only ever write the entry they push; a `float`-boundary type guard on something
     * already float emits nothing -- it sat between the ADD and the BIND in mandelbrot's `x = x2 - y2 + x0` and made the whole expression materialise for nothing. */
    case OP_CONST: case OP_INT: case OP_TYPE_GUARD:
    case OP_LT: case OP_LE: case OP_GT: case OP_GE:
    case OP_EQ: case OP_NE:
    case OP_JUMP_IF_CMP_FALSE:
    /* Reading an element writes only the entry it pushes plus the four scratch X registers -- no v
     * register, no call. Until this was listed, an FP-resident accumulator materialised the moment a subscript appeared, i.e. every array-summing loop (`sum += ai[k] * b[k][j]`, all of matrix_mul). */
    case OP_GET_INDEX:
    /* Same argument as reading an element; nbody wanted it: `bi.vx -= dx * bj.mass * mag` reads three
     * fields between the local it multiplies and the multiply, so without this every field-expression operand materialised into X and came back through an fmov. Neither opcode writes a v register or reads an X register whose FP copy could be the live one. */
    case OP_GET_FIELD: case OP_GET_FIELD_LOCAL:
    /* Both take their operand out of the bank (see fpConsumer), or the dispatch loop syncs it back out
     * first and the arm never sees a live entry. `**0.5` (square root) sat between every `d2` and the divide that used it. */
    case OP_SET_FIELD: case OP_POW:
        return true;
    default:
        return false;
    }
}

/* Used to be the whole of `fpWorthLoading`, tested against just the very next opcode -- true only
 * while OP_GET_INDEX still materialised everything; fpWorthLoading below now walks to the consumer instead. */
static bool fpConsumer(uint8_t op) {
    switch (op) {
    case OP_ADD: case OP_SUB: case OP_MUL: case OP_DIV:
    case OP_ADD_BIND: case OP_SUB_BIND: case OP_MUL_BIND:
    case OP_BIND: case OP_SET_LOCAL:
    case OP_LT: case OP_LE: case OP_GT: case OP_GE:
    case OP_EQ: case OP_NE:
    case OP_JUMP_IF_CMP_FALSE:
    case OP_SET_FIELD: case OP_POW:
        return true;
    default:
        return false;
    }
}

/* Counted rather than subtracted from the top: a no-register entry (class/function/builtin/self) can
 * sit in the middle of the stack -- exactly the shape an inlined call site has. */
static unsigned valueIndexOf(const Emit *e, unsigned idx) {
    unsigned seen = 0;
    for (unsigned i = 0; i < idx; i++) {
        if (holdsRegister(e->stack[i])) seen++;
    }
    return seen;
}

static bool pushCopyOfEntry(Emit *e, unsigned idx) {
    if (idx >= e->depth || !holdsRegister(e->stack[idx])) return false;
    unsigned vi = valueIndexOf(e, idx);
    fpSyncOne(e, vi);
    unsigned src = xHeldIn(e, vi);
    if (!pushValue3(e, e->stack[idx], e->stackShape[idx], e->stackClass[idx],
                    e->stackSeen[idx], -1)) {
        return false;
    }
    unsigned dst = pushReg(e) - 1;
    if (dst != src) emit(e, jaiA64MovX(dst, src));
    return true;
}

/* Answered against the inlined body's own frame -- the CALLER's operand stack: a parameter is the
 * argument entry already sitting there, a bind pins whatever's on top. Reading these through the main switch would read the CALLER's local of the same number, a different variable entirely. */
static bool inlineLocalOp(Emit *e, const uint8_t *code, int off) {
    uint8_t op = code[off];
    unsigned a = jaiReadU16(code + off + 1);
    if (a > JIT_MAX_SLOTS) return false;

    if (op == OP_BIND) {
        if (e->inlSlot[a] >= 0) {
            /* Straight-line code binds each `let` once. A second bind would
             * have to move the value into the pinned entry's register, and
             * that register may sit below something live. */
            e->whyNot = "an inlined body binding a local twice";
            return false;
        }
        /* Sound only when the bound value is the ONLY thing above the already-pinned region (`let x = expr`
     * as a statement) -- `let a, b = ..` would leave this entry the top for both binds, aliasing two slots to one register. Caught by measuring depth, not by trusting the shape. */
        if (e->depth != e->inlDepth + e->inlPinned + 1u) {
            e->whyNot = "an inlined body binding with an expression under it";
            return false;
        }
        if (!holdsRegister(e->stack[e->depth - 1])) return false;
        fpSyncOne(e, e->valueDepth - 1);
        e->inlSlot[a] = (int)(e->depth - 1);
        e->inlPinned++;
        return true;
    }

    if (e->inlSlot[a] < 0) {
        e->whyNot = "an inlined body reading a local it never bound";
        return false;
    }
    if (op == OP_GET_LOCAL) return pushCopyOfEntry(e, (unsigned)e->inlSlot[a]);

    unsigned b = jaiReadU16(code + off + 3);
    if (b > JIT_MAX_SLOTS || e->inlSlot[b] < 0) {
        e->whyNot = "an inlined body reading a local it never bound";
        return false;
    }
    if (op == OP_GET_LOCAL2) {
        return pushCopyOfEntry(e, (unsigned)e->inlSlot[a]) &&
               pushCopyOfEntry(e, (unsigned)e->inlSlot[b]);
    }

    unsigned ia = (unsigned)e->inlSlot[a], ib = (unsigned)e->inlSlot[b];
    SlotKind k = e->stack[ia];
    if (k != e->stack[ib] || (k != SLOT_INT && k != SLOT_FLOAT)) return false;
    unsigned via = valueIndexOf(e, ia), vib = valueIndexOf(e, ib);
    fpSyncOne(e, via);
    fpSyncOne(e, vib);
    unsigned ra = valueXReg(e, via), rb = valueXReg(e, vib);
    if (!pushValue(e, k, 0, NULL)) return false;
    unsigned rd = pushReg(e) - 1;
    if (k == SLOT_FLOAT) {
        emit(e, jaiA64FmovDX(JIT_FSCRATCH_A, ra));
        emit(e, jaiA64FmovDX(JIT_FSCRATCH_B, rb));
        emit(e, jaiA64FaddD(JIT_FSCRATCH_A, JIT_FSCRATCH_A, JIT_FSCRATCH_B));
        emit(e, jaiA64FmovXD(rd, JIT_FSCRATCH_A));
    } else {
        emit(e, jaiA64AddsX(rd, ra, rb));
        branchOnOverflow(e, 0u, JAI_A64_VS);
    }
    return true;
}

/* A *consumer* must be reachable, not merely the next instruction: the original rule looked only at
 * the very next opcode, which said no for `sum` in matrix_mul (followed by OP_GET_LOCAL2 then all of `ai[k] * b[k][j]` before any float operator) and cost the loop its accumulator to a cross-register-file bounce every iteration. Walks forward instead through fpFastOp's opcodes and answers on the first one that isn't; bounded since none of them is variable length. */
/* Deliberately a short whitelist: every one of these reads float entries through fpOperand/fpHeldIn
 * and none records a deopt while one is live (an int overflow inside OP_ADD goes through fpSyncAll first). Anything else releases at the top of the instruction; deoptRecordAt declines if one gets through anyway, so a wrong entry here costs a decline, not a miscompile. */
static bool fpBorrowSurvives(uint8_t op) {
    switch (op) {
    case OP_GET_LOCAL: case OP_GET_LOCAL2:
    case OP_CONST: case OP_INT:
    case OP_ADD: case OP_SUB: case OP_MUL: case OP_DIV:
    case OP_ADD_BIND: case OP_SUB_BIND: case OP_MUL_BIND:
    case OP_BIND: case OP_SET_LOCAL:
    case OP_LT: case OP_LE: case OP_GT: case OP_GE:
    case OP_EQ: case OP_NE:
    case OP_JUMP_IF_CMP_FALSE:
    /* A settled type guard emits/records nothing, but sits on this list because it stands between an
     * unfused float op and its OP_BIND (`y = 2.0*xy+y0` is ADD,TYPE_GUARD,BIND) -- releasing the borrow at the guard would reintroduce the exact `fmov` the lookahead removed. */
    case OP_TYPE_GUARD:
        return true;
    default:
        return false;
    }
}

/* Same shape as fpBorrowSurvives, just as deliberately short: each reads its operands via popValue/
 * xHeldIn, computes into `pushReg(e)-1`, and takes no deopt record with a deferred entry live. OP_INT/OP_CONST earn their place standing between `GET_LOCAL n` and the `SUB` consuming it. */
static bool deferSurvives(uint8_t op) {
    switch (op) {
    case OP_GET_LOCAL: case OP_GET_LOCAL2:
    case OP_INT: case OP_CONST:
    case OP_ADD: case OP_SUB: case OP_MUL: case OP_DIV:
    case OP_SHL: case OP_SHR:
    /* Its arm settles both operands itself on every path that reads a register
     * for them, and takes no deopt record before doing so. */
    case OP_FLOORDIV:
    case OP_LT: case OP_LE: case OP_GT: case OP_GE:
    case OP_EQ: case OP_NE:
    case OP_JUMP_IF_CMP_FALSE:
    case OP_BIND: case OP_SET_LOCAL:
    case OP_POP:
    case OP_RETURN:
        return true;
    default:
        return false;
    }
}

/* The opcodes that can spell an integer literal as an immediate, which is the
 * whole reason OP_INT is ever allowed to push nothing. Kept in step with the
 * arms below by hand, and harmless if it says yes too often: the consumer
 * settles what it cannot fold. */
static bool foldsIntLiteral(uint8_t op) {
    switch (op) {
    case OP_ADD: case OP_SUB:
    case OP_LT: case OP_LE: case OP_GT: case OP_GE:
    case OP_EQ: case OP_NE:
    case OP_JUMP_IF_CMP_FALSE:
    /* Not because a shift takes an imm12 (it doesn't) -- the shift count becomes part of the instruction
     * encoding when it's a literal 0..63, so the register that used to hold it was written and never read. bitops paid a `movz` for every one of its shifts before this. */
    case OP_SHL: case OP_SHR:
    /* Same again: a power-of-two divisor is spelt by the `asr`'s own shift field. Says yes for divisors
     * that are not powers of two too, which is harmless -- that path settles what it cannot fold, exactly as this list's contract allows. spectral's `s * (s + 1) // 2` paid a `movz x3,#2` into a register no instruction read, once per inner iteration. */
    case OP_FLOORDIV:
        return true;
    default:
        return false;
    }
}

static bool fpWorthLoading(const Emit *e, const uint8_t *code, int next,
                           int stop) {
    if (e->fpOff) return false;
    for (unsigned step = 0; step < 24u && next < stop; step++) {
        uint8_t op = code[next];
        if (fpConsumer(op)) return true;
        if (!fpFastOp(op)) return false;
        int size = jaiOpOperandSize((OpCode)op);
        if (size < 0) return false;
        next += 1 + size;
    }
    return false;
}

/* Inside a `try`: an entry of the function's own static exception table covers
 * this offset. Linear over the table because a function has one or two entries,
 * never a table worth indexing. */
static bool offsetIsProtected(const ObjFunction *fn, uint32_t off) {
    for (uint16_t i = 0; i < fn->exceptionCount; i++) {
        const ExceptionEntry *x = &fn->exceptions[i];
        if (off >= x->start && off < x->end) return true;
    }
    return false;
}

/* An opcode with no arm in the switch below.
 *
 * It used to `return false`, which declines the WHOLE enclosing function -- so
 * a `try` whose catch block holds OP_GET_EXC, or a `{}` literal on a path that
 * never runs, evicted every function containing it from the compiled tier for
 * good. 66 of 137 opcodes were in that position (tests/vm/jit_unarmed.baseline)
 * and `tests/bench/error_paths`' eight-million-iteration counted loop was one
 * of them: 124ms interpreted against 12ms compiled, decided by a catch block
 * the opcode histogram says runs zero times.
 *
 * Instead, deoptimise unconditionally here. The interpreter resumes at this
 * exact instruction holding exactly what the model says, which is what every
 * failed guard already does -- the cold path is interpreted and the hot path
 * around it still compiles. It declines only when a deopt site cannot be built
 * at this offset, which is deoptSite's existing "a guard resumes an instruction
 * whose operands it has already consumed" test and exactly the right question.
 *
 * Nothing on the fall-through edge past an unconditional deopt can execute, so
 * the walk does not model it: it skips ahead. Advancing the model by the
 * opcode's static stack effect instead was the other option and buys nothing --
 * every instruction it would emit is unreachable -- while needing a kind for
 * whatever the opcode produced, which the tier by definition does not have for
 * an opcode it cannot compile. A skipped offset keeps offsetToInst == -1, so a
 * branch that does target one declines when the fixups are resolved rather than
 * jumping into nothing.
 *
 * WHERE it may resume is the part that has to be earned. Skipping to the next
 * offset anything in the chunk can branch to was not enough: the branch that
 * reaches it may itself have been inside the skipped run, in which case nothing
 * establishes the model there and the walk carries on with the one it had at
 * the deopt. That is silent -- the registers all shift by the same amount, so
 * the body stays self-consistent -- right up until a deopt record names an
 * operand stack the interpreter has not got. `_is_any` in
 * lib/jaithon/compile/check/decl.jai is the shape: its `t is null` is unarmed,
 * and the only branch to the code after it is the `if`'s own jump, three bytes
 * into the skipped run.
 *
 * So it resumes only where the model can be stated:
 *   - an offset an already-emitted branch targets, which reconcileAfterUncond
 *     trims to that branch's own depth on the next iteration; or
 *   - an offset the BYTECODE says has an empty operand stack, where there are
 *     no entries and therefore no kinds to invent. This is what keeps a `while`
 *     whose head has not been compiled yet -- a back edge is a branch target
 *     the walk has not reached -- from being lost.
 * Anything else keeps skipping, and running out means the rest of the function
 * is not compiled: the last thing emitted is the unconditional branch to the
 * deopt stub, so control never falls off the end. */
static bool skipResumeOk(const Emit *e, uint32_t at) {
    if (reconcileDepth(e, at) >= 0) return true;
    return e->chunkDepth != NULL && at < (uint32_t)e->chunkDepthCount &&
           e->chunkDepth[at] == 0;
}

static bool emitUnarmedDeopt(Emit *e, const Chunk *c, int *off, int stop) {
    /* Remembered for the fixup pass. When this stops the walk before a forward
     * branch's target, that branch cannot be resolved and the error reported is
     * "a branch to an offset this walk never emitted" -- which names the
     * SYMPTOM. The opcode that actually stopped the walk is the cause, and
     * without it the reader has a bytecode offset and no idea why. */
    e->unarmedOp = e->lastOp;
    e->unarmedAt = e->curOffset;
    if (e->inlining) {
        /* Half an inlined body cannot be taken back, and the caller reads the
         * result out of the model -- past a deopt there is none. */
        e->whyNot = "an inlined body reaching an opcode this tier cannot speak";
        return false;
    }
    unsigned k;
    if (!deoptRecordAt(e, e->curOffset, false, &k)) return false;
    if (e->fixupCount >= JIT_MAX_FIXUPS) { e->failed = true; return false; }
    e->fixups[e->fixupCount].instIndex    = (int)e->count;
    e->fixups[e->fixupCount].targetOffset = FIXUP_DEOPT - k;
    e->fixups[e->fixupCount].conditional  = false;
    e->fixups[e->fixupCount].depth        = -1;
    e->fixupCount++;
    emit(e, jaiA64B(0));

    int at = *off;
    for (;;) {
        int len = instructionLength(c, at);
        if (len <= 0) return false;      /* undecodable: the walk is lost */
        at += len;
        if (at >= stop) break;
        if (offsetIsBranchTarget(c, (uint32_t)at) &&
            skipResumeOk(e, (uint32_t)at)) {
            /* Nothing an emitted branch reaches, so reconcileAfterUncond has
             * nothing to trim to and the empty stack the bytecode promises is
             * the whole model. Cleared raw rather than through popValue: this
             * sits immediately after an unconditional branch, so an fmov
             * settling an entry nobody will read is dead code. */
            if (reconcileDepth(e, (uint32_t)at) < 0) {
                e->depth      = 0;
                e->valueDepth = 0;
                e->fpLive     = 0;
                e->fpBorrow   = 0;
                e->kPend      = 0;
                e->xBorrow    = 0;
            }
            break;
        }
    }
    if (at > stop) at = stop;
    *off = at;
    return true;
}

/* Why an arm jumps to `unarmedOpcode` instead of returning a reason.
 *
 * Every opcode added to the switch converts what used to be the `default`
 * case's partial walk into an outright refusal, and that is a regression: a
 * body that compiled a prefix and interpreted the rest now compiles none of
 * it. Adding `not` alone cost 80 whole-body declines in the resolver, all of
 * them the measuring pass seeing SLOT_INT where a bool would later be.
 *
 * It also makes each arm's kill switch honest. With a hard refusal the "off"
 * side of an A/B declines the body, which is worse than the no-arm behaviour
 * it stands in for, and every ratio measured against it is inflated.
 *
 * The jump has to land in `default` rather than in a helper: the unarmed path
 * emits an unconditional branch, so the fall-through edge is gone and the walk
 * has to be told (`afterUncond`). A helper returning "I handled it" cannot say
 * that. */


/* Where the body proper starts. A defaulted parameter compiles to a thunk of
 * its own at the TOP of the chunk with the body hopped to over an
 * unconditional jump -- `fn matches(x: int, want: int = -1)` is exactly
 * `0000 OP_JUMP +4 -> 0007 / 0003 OP_INT -1 / 0006 OP_RETURN / 0007 body`.
 *
 * Walking those three dead instructions cost far more than emitting them: a
 * thunk ends in OP_RETURN like any other return, so mergeReturnKind recorded
 * the DEFAULT's kind and every real return of a different kind then clashed
 * with it and declined the whole function. `matches` above is INT-then-BOOL
 * and stopped at `OP_RETURN` for it; in a self-hosted compile the same clash
 * is what stopped window_clean, Lexer._push, Universe.intern, parser._type and
 * modsig.load.
 *
 * MEASURED, best of nine alternating runs under scripts/gpu_lock.sh, the two
 * sample sets not overlapping at all in either workload: `check --no-cache`
 * over four compiler files 2196ms -> 2106ms (4.3%), and over four others
 * 1469ms -> 1404ms (4.6%). A 3M-call probe on `matches` itself is 137ms ->
 * 29ms wall, ~13x on the loop once the ~20ms process floor is taken off both
 * sides -- the callee's body is too small for the OSR tier to rescue, which is
 * the shape that pays most. Where the loop is INSIDE the defaulted function,
 * OSR already had it and the same change is only 1.36x. The durable figure is
 * the decline count: 66 stops at OP_RETURN in one `check --no-cache` of
 * check/expr.jai, and zero after.
 *
 * SAFE because the skipped region is unreachable from the whole-function
 * entry, not merely unwalked:
 *   - The interpreter runs a thunk by ENTERING at its own offset
 *     (evalDefaultThunk, off fn->defaultOffsets), never by falling into it,
 *     and bindCallArgsSlow does all of that BEFORE the frame runs -- so by the
 *     time a compiled instruction executes the defaults are already in slots.
 *   - Every path into the whole-function form checks argc == arity first
 *     (callClosure, jaiCallValue1, run()'s tail call, and a compiled self-call,
 *     which declines outright unless argc == arity).
 *   - Nothing branches into the region: every offset in it is checked against
 *     offsetIsBranchTarget, whose operand table includes OP_PUSH_HANDLER and
 *     OP_PUSH_FINALLY, so a default expression holding a try edge fails the
 *     test and keeps the old behaviour. (It would be safe anyway -- the tier
 *     has no arm for OP_PUSH_HANDLER, so only the interpreter ever runs that
 *     thunk -- but the check does not have to know that.)
 *
 * Ruled out as a narrower fix: having mergeReturnKind ignore returns inside the
 * region. It would recover the return kind but still emit the thunks and still
 * charge them to the instruction budget, and it leaves the walk modelling code
 * the entry cannot reach. */
static int bodyEntryOffset(const ObjFunction *fn) {
    const Chunk *c = &fn->chunk;
    if (fn->defaultCount == 0 || fn->defaultOffsets == NULL) return 0;
    if (c->count < 3 || c->code[0] != OP_JUMP) return 0;
    int target = 3 + (int)jaiReadI16(c->code + 1);
    if (target <= 3 || target >= c->count) return 0;
    for (unsigned d = 0; d < fn->defaultCount; d++) {
        uint32_t at = fn->defaultOffsets[d];
        if (at < 3u || at >= (uint32_t)target) return 0;
    }
    for (int at = 3; at < target;) {
        if (offsetIsBranchTarget(c, (uint32_t)at)) return 0;
        int len = instructionLength(c, at);
        if (len <= 0) return 0;
        at += len;
    }
    return target;
}

bool compileBody(Emit *e, ObjClosure *closure) {
    ObjFunction *fn = closure->fn;
    const uint8_t *code = fn->chunk.code;
    int count = fn->chunk.count;

    /* An inlined body is walked whole; the OSR window belongs to the caller. */
    int start = (!e->inlining && e->osr) ? (int)e->osrTop : bodyEntryOffset(fn);
    int stop  = (!e->inlining && e->osr) ? (int)e->osrEnd : count;
    bool afterUncond = false;
    /* The offset the walk visited before this one, for the arms that want to
     * know whether the value on top of the stack came from a literal. Updated
     * at the top rather than the bottom because arms leave the loop body by
     * `break`, by `continue` and by `return` alike. */
    int prevOff = -1, thisOff = -1;
    for (int off = start; off < stop && !e->failed;) {
        prevOff = thisOff;
        thisOff = off;
        bool fellIn = !afterUncond;
        if (afterUncond) reconcileAfterUncond(e, (uint32_t)off);
        /* The one place the model is asked to agree with anything. Which of the
         * two answers is right depends on how the walk got here: along a
         * fall-through edge a disagreement means an arm moved the model by the
         * wrong amount, and there is nothing to do but decline; arriving
         * without one means this offset is reached only by a branch, and if
         * reconcileAfterUncond could not restate the model from that branch
         * then this is code the compiled body has no way in to. Stopping is
         * right there and declining would be a coverage loss for nothing: the
         * last thing emitted is unconditional, so control never falls off the
         * end, and a branch that does target a skipped offset declines when the
         * fixups are resolved. */
        if (!modelAgreesWithChunk(e, (uint32_t)off)) {
            if (fellIn) {
                e->whyNot = "the operand model disagrees with the bytecode";
                if (getenv("JAI_JIT_WHY")) {
                    fprintf(stderr,
                            "[jit] at %d: model depth %u, the bytecode says %d\n",
                            off, e->depth, e->chunkDepth[off]);
                }
                return false;
            }
            break;
        }
        uint8_t op = code[off];
        afterUncond = !jaiOpFallsThrough(op);
        /* Walked out of the way for the chain diagnostic; see Emit::chainSkip.
         * Never set in an ordinary compile. */
        for (unsigned ci = 0; ci < e->chainSkipCount; ci++) {
            if (e->chainSkip[ci] == (uint32_t)off) {
                e->curOffset = (uint32_t)off;
                e->lastOp = op;
                /* Settled first, as a branch join is. The ordinary unarmed path
                 * is only ever reached from an arm that declined BEFORE
                 * touching the model, whereas this one steps over an
                 * instruction whose arm may have left a value deferred or in
                 * the FP bank -- and the deopt record cannot describe those.
                 * Without this the diagnostic reported "a deferred value
                 * reached a guard" as the second link of every chain, which is
                 * an artefact of the skip and not a fact about the program. */
                fpSyncAll(e);
                settleAll(e);
                goto unarmedOpcode;
            }
        }
        /* Whose regions these are matters: inside an inline the offsets are the
         * callee's while every guard resumes at the CALLER's call site, so the
         * caller's answer is the one that stands. inlineGlobalCall/inlineMethod
         * refuse a callee with a table of its own. */
        if (!e->inlining) {
            e->inProtected = offsetIsProtected(fn, (uint32_t)off);
        }
        /* A stack proof is only good along the fall-through edge this walk is
         * following. offsetIsBranchTarget scans the whole chunk, so it catches
         * a back edge whose branch has not been emitted yet -- and it is only
         * asked while a proof is actually live, which is the two or three
         * instructions between `s[i]` and whatever consumes it, or the single
         * instruction between OP_NULL and OP_IS. */
        if (anyStackProof(e) &&
            offsetIsBranchTarget(&fn->chunk, (uint32_t)off)) {
            clearStackProofs(e);
        }
        /* A field-kind memo is good along the same one edge, and goes for the
         * same reason -- see forgetFieldKinds. `fn` is whichever body is being
         * walked, so an inlined one is measured against its own chunk. */
        if (e->knownCount != 0 &&
            offsetIsBranchTarget(&fn->chunk, (uint32_t)off)) {
            forgetFieldKinds(e);
        }
        /* Settles any deferred entry BEFORE the offset map records the instruction start, so a branch landing
 * here (arriving with everything in its own register) skips the settle and only the fall-through pays -- both paths then agree, which a join requires. Forward branches are known here; backward ones checked at the end. */
        if (anyDeferred(e)) {
            bool joinsHere = false;
            if (!e->inlining) {
                for (unsigned f = 0; f < e->fixupCount && !joinsHere; f++) {
                    joinsHere = (e->fixups[f].targetOffset == (uint32_t)off);
                }
            }
            /* Inside a `try` nothing may cross an instruction boundary
             * unmaterialised: branchOnDeoptInstStart describes the model as of
             * the instruction's START, which means describing entries this
             * instruction has already popped -- and a popped entry that was
             * only ever a borrow of a local's register never wrote its own.
             * Settling here is the same conservative branch a non-whitelisted
             * opcode already takes, and it costs `mov`s that §5 prices at zero.
             */
            if (joinsHere || e->inProtected || !deferSurvives(op) ||
                e->deferCarryCount >= 64) {
                settleAll(e);
            } else {
                e->deferCarry[e->deferCarryCount++] = (uint32_t)off;
            }
        }
        /* Above the offset map for the same reason the deferred settle is:
         * branchTo empties the FP bank before it records a fixup, so an edge
         * arriving here holds every entry in its own X register -- and must
         * not run the fall-through's `fmov x, d` over a d register it never
         * wrote. `if flag { a[i] } else { a[0] }` is the shape: one arm ends
         * at the jump with its value in X, the other flows into the OP_ADD
         * and leaves it in the bank. Settled here, both edges agree. */
        if (e->fpLive != 0 && !e->inlining) {
            for (unsigned f = 0; f < e->fixupCount; f++) {
                if (e->fixups[f].targetOffset != (uint32_t)off) continue;
                fpSyncAll(e);
                break;
            }
        }
        /* Above the offset map on purpose: a back edge to `off` must land on
         * the loop head, not on the loads that were hoisted out of it. */
        emitHoistsAt(e, (uint32_t)off);
        e->offsetToInst[off]  = (int)e->count;
        e->offsetToDepth[off] = (int)stackSignature(e);
        e->lastOp = op;
        e->whySub[0] = '\0';
        /* A borrow ends here unless the instruction is one of the few it is
         * allowed to live across. The top of an instruction is the one place
         * the release is guaranteed to be on the executed path. */
        if (e->fpBorrow != 0 && (e->inProtected || !fpBorrowSurvives(op))) {
            fpReleaseAll(e);
        }
        e->curOffset = (uint32_t)off;
        e->instDepth = e->depth;
        e->instValueDepth = e->valueDepth;
        /* Every entry is in its own register right now, so a record taken of
         * this instruction's START stays readable however far the arm below
         * gets. An assertion, not the mechanism: the settles above are what
         * make it true inside a protected region, and branchOnDeoptInstStart
         * declines rather than describing a register nothing wrote. */
        e->instClean = (e->kPend == 0 && e->xBorrow == 0 && e->fpBorrow == 0);

        if (e->fpLive != 0) {
            /* A join already settled above, so anything still live here is on
             * a single edge. An inlined OP_RETURN is not a sync point either:
             * the only entry that outlives it is the result, and
             * inlineGlobalCall carries that one across in the bank. Everything
             * under it is discarded unread. */
            if (e->inlining && op == OP_RETURN) {
                /* handled below */
            } else if (!fpFastOp(op) || e->inProtected) {
                fpSyncAll(e);
            } else if (e->fpCarryCount < 64) {
                e->fpCarry[e->fpCarryCount++] = (uint32_t)off;
            } else {
                fpSyncAll(e);
            }
        }

        /* The four local opcodes an inlined body is allowed, answered against its own frame, before the main
         * switch reads them as the caller's slot numbers. */
        if (e->inlining && (op == OP_GET_LOCAL || op == OP_GET_LOCAL2 ||
                            op == OP_ADD_LOCALS || op == OP_BIND)) {
            if (!inlineLocalOp(e, code, off)) return false;
            off += instructionLength(&fn->chunk, off);
            continue;
        }
        if (e->inlining && op == OP_RETURN) {
            /* The result is on top and stays there; the caller's driver takes
             * it from the model. */
            if (e->depth == 0 || !holdsRegister(e->stack[e->depth - 1])) {
                e->whyNot = "an inlined body returning a value with no register";
                return false;
            }
            /* A float result stays in its d register: inlineGlobalCall moves it into the caller's bank with
             * one FP-to-FP move, where syncing here spent an `fmov x,d` and then made the caller's first float operator pay an `fmov d,x` to undo it. settleAll still runs -- it is X-side only (kPend/xBorrow), and those bits are never set on an entry the bank holds. */
            if (!(e->fpLive & (1u << (e->valueDepth - 1)))) {
                fpSyncOne(e, e->valueDepth - 1);
            }
            settleAll(e);   /* the caller reads the result out of valueXReg */
            break;
        }

        switch (op) {
        case OP_GET_LOCAL: {
            unsigned slot = jaiReadU16(code + off + 1);
            /* Both refusals used to be silent, so the census said only
             * "OP_GET_LOCAL" for two unrelated causes -- one a window the OSR
             * form does not cover, the other a slot whose kind is not known
             * yet. They want different fixes; they should not share a line. */
            if (!localInRange(e, slot)) {
                e->whyNot = "a local outside the compiled window";
                return false;
            }
            if (e->localKind[slot] == SLOT_OPAQUE) {
                e->whyNot = "a local of no known kind";
                return false;
            }
            if (slot == 0) e->usesSlot0 = true;
            if (!pushValue3(e, e->localKind[slot], e->localShape[slot],
                            e->localClass[slot],
                            seenLocal(e, slot),
                            (int)slot)) {
                return false;
            }
            /* The seed of the index shape: this entry IS this local, offset
             * zero. See Emit::idxKnown. */
            if (e->localKind[slot] == SLOT_INT && slot <= UINT8_MAX) {
                unsigned at = e->valueDepth - 1;
                e->idxKnown |= 1u << at;
                e->idxBase[at] = (uint8_t)slot;
                e->idxOff[at]  = 0;
            }
            if (e->localKind[slot] == SLOT_FLOAT && !e->dynamicLocal[slot] &&
                fpWorthLoading(e, code, off + 3, stop)) {
                unsigned idx = e->valueDepth - 1;
                if (e->slotFpReg[slot] != 0) {
                    fpBorrowLocal(e, idx, e->slotFpReg[slot]);
                } else {
                    localInFp(e, slot, fpRegAt(e, idx));
                    fpClaim(e, idx);
                }
            } else {
                unsigned home = localHomeX(e, slot);
                if (home != 0) {
                    /* The copy this used to always emit is the whole cost of reading a local (six of them in `fib`) --
                     * borrowing defers it; if nothing consumes the value before an instruction that can't read a borrow, the settle there emits exactly the same mov, so this never costs more. */
                    xBorrowLocal(e, e->valueDepth - 1, home);
                } else {
                    unsigned dst = pushReg(e) - 1;
                    unsigned src = localIn(e, slot, dst);
                    if (src != dst) emit(e, jaiA64MovX(dst, src));
                }
            }
            off += 3;
            break;
        }

        case OP_INT: {
            int16_t k = jaiReadI16(code + off + 1);
            if (!pushValue(e, SLOT_INT, 0, NULL)) return false;
            e->kKnown |= 1u << (e->valueDepth - 1);
            e->kKnownVal[e->valueDepth - 1] = k;
            /* Push nothing when the next instruction can say the literal as an
             * immediate: `n - 1` is one `subs`, not a `movz` and a `subs`. The
             * consumer settles it if the rest of its shape turns out not to
             * allow the fold, so being wrong here costs the instruction it
             * would have spent anyway. */
            if (off + 3 < stop && foldsIntLiteral(code[off + 3]) &&
                k >= -4095 && k <= 4095) {
                kPendLocal(e, e->valueDepth - 1, k);
            } else {
                emitConst64(e, pushReg(e) - 1, k);
            }
            off += 3;
            break;
        }

        case OP_TRUE:
        case OP_FALSE: {
            /* A bool is a payload of 1 or 0 in a register, the same as any
             * other value here. Their absence declined `queens` and `sieve`
             * outright -- `return false` and a list of flags are not exotic. */
            if (!pushValue(e, SLOT_BOOL, 0, NULL)) return false;
            emitConst64(e, pushReg(e) - 1, op == OP_TRUE ? 1 : 0);
            off += 1;
            break;
        }

        case OP_SET_LOCAL: {
            /* Assigns without popping: the value stays as the statement's
             * result, which is what the interpreter does. */
            unsigned slot = jaiReadU16(code + off + 1);
            if (!localInRange(e, slot)) return false;
            if (slot == 0) e->usesSlot0 = true;
            if (e->depth == 0 || !holdsRegister(e->stack[e->depth - 1])) return false;
            /* A local keeps one kind for the whole function. Two kinds would
             * mean the reads of it cannot be compiled to one instruction, and
             * the join check works on the operand stack, not on locals. */
            if (!adoptLocalKind(e, slot, e->stack[e->depth - 1],
                                e->stackShape[e->depth - 1],
                                e->stackClass[e->depth - 1])) {
                e->whyNot = kindClash(e, slot);
                return false;
            }
            if (!e->fpOff && !e->dynamicLocal[slot] &&
                e->stack[e->depth - 1] == SLOT_FLOAT &&
                (e->fpLive & (1u << (e->valueDepth - 1)))) {
                localOutFp(e, slot, fpHeldIn(e, e->valueDepth - 1));
            } else {
                localOut(e, slot, xHeldIn(e, e->valueDepth - 1));
            }
            off += 3;
            break;
        }

        case OP_ADD_LOCALS: {
            unsigned a = jaiReadU16(code + off + 1);
            unsigned b = jaiReadU16(code + off + 3);
            if (!localInRange(e, a) || !localInRange(e, b)) {
                return subWhy(e, "a fused add of a local the model does not "
                              "cover");
            }
            SlotKind ka2 = e->localKind[a];
            /* Named: this was 148 declines across four compiler files reading
             * only "OP_ADD_LOCALS", with nothing in them to act on. */
            if (ka2 != e->localKind[b]) {
                return subWhy(e, "a fused add of a %s and a %s",
                              slotKindName(ka2), slotKindName(e->localKind[b]));
            }
            if (ka2 == SLOT_OBJ && jitConcatLocals() && e->callsOut &&
                !e->inlining) {
                /* `out = out + piece` -- string building, and the fused form is
                 * the one real code emits. The stack `+` already had a concat
                 * arm; this opcode did not, so an OSR loop doing the commonest
                 * thing a lexer does declined WHOLE: the probe is 4,001,920
                 * interpreted instructions and the census showed 28 of these in
                 * parser.jai alone reading only "a fused add of two objects".
                 *
                 * The operands are slot numbers here and emitStringConcat wants
                 * stack entries, so they are pushed first. That is the cost the
                 * fusion existed to avoid, and it is nothing next to a call
                 * that allocates a string. Both operands are guarded inside
                 * emitStringConcat, so a sample that turns out wrong deopts at
                 * this instruction with both locals untouched. */
                Value sa = seenLocal(e, a);
                Value sb = seenLocal(e, b);
                if (IS_STRING(sa) && IS_STRING(sb)) {
                    if (a == 0 || b == 0) e->usesSlot0 = true;
                    if (!pushLocalAsValue(e, a)) return false;
                    if (!pushLocalAsValue(e, b)) return false;
                    if (!emitStringConcat(e, sa)) return false;
                    off += 5;
                    break;
                }
            }
            if (ka2 != SLOT_INT && ka2 != SLOT_FLOAT) {
                return subWhy(e, "a fused add of two %ss", slotKindName(ka2));
            }
            if (a == 0 || b == 0) e->usesSlot0 = true;
            if (!pushValue(e, ka2, 0, NULL)) return false;
            if (ka2 == SLOT_FLOAT && !e->dynamicLocal[a] &&
                !e->dynamicLocal[b] && !e->fpOff) {
                unsigned idx = e->valueDepth - 1;
                /* Either operand already in a d register of its own is read
                 * from there. `x2 + y2` was two fmovs and an add. */
                unsigned da, db;
                if (e->slotFpReg[a] != 0) {
                    da = e->slotFpReg[a];
                } else {
                    localInFp(e, a, fpRegAt(e, idx));
                    da = fpRegAt(e, idx);
                }
                if (e->slotFpReg[b] != 0) {
                    db = e->slotFpReg[b];
                } else {
                    localInFp(e, b, JIT_FP_BANK + JIT_MAX_SAVED);
                    db = JIT_FP_BANK + JIT_MAX_SAVED;
                }
                emit(e, jaiA64FaddD(fpRegAt(e, idx), da, db));
                fpClaim(e, idx);
                off += 5;
                break;
            }
            {
                unsigned ra2 = localIn(e, a, JIT_SCRATCH_C);
                unsigned rb2 = localIn(e, b, JIT_SCRATCH_D);
                if (ka2 == SLOT_FLOAT) {
                    emit(e, jaiA64FmovDX(JIT_FSCRATCH_A, ra2));
                    emit(e, jaiA64FmovDX(JIT_FSCRATCH_B, rb2));
                    emit(e, jaiA64FaddD(JIT_FSCRATCH_A, JIT_FSCRATCH_A,
                                        JIT_FSCRATCH_B));
                    emit(e, jaiA64FmovXD(pushReg(e) - 1, JIT_FSCRATCH_A));
                } else {
                    emit(e, jaiA64AddsX(pushReg(e) - 1, ra2, rb2));
                    branchOnOverflow(e, 0u, JAI_A64_VS);
                }
            }
            off += 5;
            break;
        }

        case OP_ADD_BIND: {
            /* `ADD; BIND a` fused. Floats go through the same fmov pair the
             * plain add uses; ints keep the overflow check. */
            unsigned slot = jaiReadU16(code + off + 1);
            if (!localInRange(e, slot)) return false;
            if (slot == 0) e->usesSlot0 = true;
            if (!e->fpOff && e->depth >= 2 && !e->dynamicLocal[slot] &&
                e->stack[e->depth - 1] == SLOT_FLOAT &&
                e->stack[e->depth - 2] == SLOT_FLOAT) {
                /* The whole operation stays in the FP bank: two operands that
                 * are already there, one instruction, and a store straight out
                 * of a d register. */
                if (!adoptLocalKind(e, slot, SLOT_FLOAT, 0, NULL)) {
                    e->whyNot = kindClash(e, slot);
                    return false;
                }
                unsigned ib = e->valueDepth - 1, ia = e->valueDepth - 2;
                unsigned db = fpOperand(e, ib), da = fpOperand(e, ia);
                unsigned rx; SlotKind kx;
                if (!popValueRaw(e, &rx, &kx)) return false;
                if (!popValueRaw(e, &rx, &kx)) return false;
                unsigned dd = fpBindDest(e, slot, fpRegAt(e, ia));
                emit(e, jaiA64FaddD(dd, da, db));
                localOutFp(e, slot, dd);
                off += 3;
                break;
            }
            /* `text = text + piece` on strings. Out to jaiStringConcat behind a
             * pair of type guards -- see emitStringConcat. Before this arm the
             * refusal below gave up word_freq's entire `main` loop, which is
             * this one instruction plus the LCG arithmetic around it. */
            {
                Value csample;
                if (concatOperands(e, &csample)) {
                    /* The kind is adopted AFTER the call, not before: the
                     * descriptor's root fill reads every object-kinded local,
                     * and until the store below this slot still holds the old
                     * string -- claiming the new kind first would describe a
                     * slot the call has not written yet. */
                    if (!emitStringConcat(e, csample)) return false;
                    if (!adoptLocalKindSeen(e, slot, SLOT_OBJ, 0, NULL,
                                            csample)) {
                        e->whyNot = kindClash(e, slot);
                        return false;
                    }
                    unsigned rc;
                    if (!popValue(e, &rc, NULL)) return false;
                    localOut(e, slot, rc);
                    off += 3;
                    break;
                }
            }
            unsigned rb, ra;
            SlotKind kb, ka;
            if (!popValue(e, &rb, &kb)) return false;
            if (!popValue(e, &ra, &ka)) return false;
            if (ka != kb) return false;
            if (!adoptLocalKind(e, slot, ka, 0, NULL)) {
                e->whyNot = kindClash(e, slot);
                return false;
            }
            unsigned rd = localDest(e, slot);
            if (ka == SLOT_FLOAT) {
                emit(e, jaiA64FmovDX(JIT_FSCRATCH_A, ra));
                emit(e, jaiA64FmovDX(JIT_FSCRATCH_B, rb));
                emit(e, jaiA64FaddD(JIT_FSCRATCH_A, JIT_FSCRATCH_A,
                                    JIT_FSCRATCH_B));
                emit(e, jaiA64FmovXD(rd, JIT_FSCRATCH_A));
            } else if (ka == SLOT_INT) {
                rd = ovfDest(e, rd);   /* localOut copies it home below */
                emit(e, jaiA64AddsX(rd, ra, rb));
                branchOnOverflow(e, 0u, JAI_A64_VS);
            } else {
                e->whyNot = "add-bind of a kind that is neither int nor float";
                return false;
            }
            localOut(e, slot, rd);
            off += 3;
            break;
        }

        /* `slots[S] < imm`, pushing the bool.
         *
         * This arm was missing until 2026-08-12, so any function containing the
         * opcode declined WHOLE -- which is the failure the peephole's own
         * history records: OP_CMP_LOCAL_CONST_LT shipped without a JIT arm once
         * before and cost 14 distinct declines. Nothing in the benchmark suite
         * happened to hit it this time, which is exactly why it went unnoticed;
         * `make jit-fusion-check` now refuses a build where a fused opcode has
         * no arm here. */
        /* Stamp the declared element kind onto the container on top.
         *
         * Needs an arm rather than a decline: it sits right after a container
         * literal, so `var out: list[int] = []` inside a hot function put it in
         * the middle of one. Without this, sort_merge's `merge` -- whose whole
         * body is pushes onto exactly such a list -- declined and ran
         * interpreted: 270ms to 510ms. A new opcode in ordinary code is a JIT
         * admission question before it is anything else. */
        case OP_ELEM_KIND: {
            uint8_t packed = code[off + 1];
            if (e->depth == 0) return false;
            /* The STATIC kind, not the sampled value: the measuring pass runs
             * with no sample, so keying on stackSeen declined every time and
             * cost the whole function. OP_BUILD_LIST pushes SLOT_LIST, which is
             * exactly what the emitter puts this opcode after. */
            if (e->stack[e->depth - 1] != SLOT_LIST) {
                /* The interpreter stamps a dict's two nibbles as well, and
                 * does NOTHING for any other container -- "an unstamped
                 * container is simply unguarded". Both of those are arms.
                 *
                 * They became reachable the day the dict and set literals got
                 * arms of their own: before that the walk stopped AT the
                 * literal, so this opcode was never reached with a non-list on
                 * top. `var d: dict[str, int] = {}` in a hot body then
                 * declined the WHOLE function, which is strictly worse than
                 * the partial walk it replaced. */
                uint8_t built = e->stackObjType[e->depth - 1];
                if (built == (uint8_t)(OBJ_SET + 1) ||
                    built == (uint8_t)(OBJ_TUPLE + 1)) {
                    off += 2;
                    break;
                }
                if (built != (uint8_t)(OBJ_DICT + 1)) {
                    goto unarmedOpcode;
                }
                unsigned dr = valueXReg(e, e->valueDepth - 1);
                /* The prediction came from the build instruction just below,
                 * but a guard costs two instructions and does not depend on
                 * the emitter keeping them adjacent. */
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, dr,
                                   (unsigned)offsetof(Obj, type)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_DICT));
                branchOnDeoptInstStart(e, JAI_A64_NE);
                emitConst64(e, JIT_SCRATCH_A, (int64_t)((packed >> 4) & 0xFu));
                emit(e, jaiA64StrByte(JIT_SCRATCH_A, dr,
                                      (unsigned)offsetof(ObjDict, keyKind)));
                emitConst64(e, JIT_SCRATCH_A, (int64_t)(packed & 0xFu));
                emit(e, jaiA64StrByte(JIT_SCRATCH_A, dr,
                                      (unsigned)offsetof(ObjDict, valKind)));
                e->wroteHeap = true;
                off += 2;
                break;
            }
            unsigned r = valueXReg(e, e->valueDepth - 1);
            /* The arm was already computing this byte and throwing it away.
             * Keeping it is what lets a subscript of this list choose a load
             * when no sample of it can exist. */
            e->stackElemDecl[e->depth - 1] = (uint8_t)((packed & 0xFu) + 1u);
            emitConst64(e, JIT_SCRATCH_A, (int64_t)(packed & 0xFu));
            emit(e, jaiA64StrByte(JIT_SCRATCH_A, r,
                                  (unsigned)offsetof(ObjList, elemKind)));
            /* And the storage, on the same terms jaiListSpecialise takes: an
             * empty list with nothing reserved, which is what a `[]` literal
             * is. Six instructions rather than a call, and no allocation --
             * that is the whole reason the interpreter's half refuses a
             * non-empty list too. Without this the two tiers build the same
             * literal at different widths and a pinned loop form is denied
             * entry for half the lists it meets; see jaiListSpecialise. */
            uint8_t kStg = listAltFor(
                (packed & 0xFu) == FIELD_KIND_INT   ? SLOT_INT
              : (packed & 0xFu) == FIELD_KIND_FLOAT ? SLOT_FLOAT
              : (packed & 0xFu) == FIELD_KIND_BOOL  ? SLOT_BOOL
                                                    : SLOT_OPAQUE);
            if (kStg != LIST_STORE_BOXED && jaiListUnboxOn()) {
                /* JIT_SCRATCH_A only: this arm has always used one scratch,
                 * and e->scratchRoom is what says how many the body actually
                 * reserved -- reaching for a second clobbered a live value
                 * register and miscompiled the self-hosted emitter. */
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, r,
                                   (unsigned)offsetof(ObjList, count)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, 0));
                int kA = (int)e->count;
                emit(e, jaiA64BCond(JAI_A64_NE, 0));
                emit(e, jaiA64LdrX(JIT_SCRATCH_A, r,
                                   (unsigned)offsetof(ObjList, items)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, 0));
                int kB = (int)e->count;
                emit(e, jaiA64BCond(JAI_A64_NE, 0));
                emitConst64(e, JIT_SCRATCH_A, (int64_t)kStg);
                emit(e, jaiA64StrByte(JIT_SCRATCH_A, r,
                                      (unsigned)offsetof(ObjList, stg)));
                e->code[kA] = jaiA64BCond(JAI_A64_NE,
                                          (int32_t)((int)e->count - kA));
                e->code[kB] = jaiA64BCond(JAI_A64_NE,
                                          (int32_t)((int)e->count - kB));
            }
            e->wroteHeap = true;
            off += 2;
            break;
        }

        case OP_CMP_LOCAL_CONST_LT: {
            unsigned slot = jaiReadU16(code + off + 1);
            int16_t  imm  = jaiReadI16(code + off + 3);
            if (!localInRange(e, slot)) return false;
            if (e->localKind[slot] != SLOT_INT) return false;
            if (slot == 0) e->usesSlot0 = true;

            /* Compare before pushing: the compare reads the local, and the
             * pushed entry is only the bool the flags produce. */
            if (imm >= -4095 && imm <= 4095) {
                emitCmpImm(e, localIn(e, slot, JIT_SCRATCH_C), imm);
            } else {
                emitConst64(e, JIT_SCRATCH_A, imm);
                emit(e, jaiA64SubsXReg(31, localIn(e, slot, JIT_SCRATCH_C),
                                       JIT_SCRATCH_A));
            }
            if (!pushValue(e, SLOT_BOOL, 0, NULL)) return false;
            emit(e, jaiA64CsetX(pushReg(e) - 1, JAI_A64_LT));
            off += 5;
            break;
        }

        case OP_ADD_INT_CONST: {
            unsigned slot = jaiReadU16(code + off + 1);
            int16_t  imm  = jaiReadI16(code + off + 3);
            if (!localInRange(e, slot)) return false;
            if (e->localKind[slot] != SLOT_INT) return false;
            if (slot == 0) e->usesSlot0 = true;
            if (!pushValue(e, SLOT_INT, 0, NULL)) return false;
            {
                unsigned dst = pushReg(e) - 1;
                unsigned cur = localIn(e, slot, JIT_SCRATCH_C);
                /* imm12, so a step outside +/-4095 still goes through a
                 * register. See OP_INC_LOCAL for why `subs` is the negative
                 * arm rather than a negated constant. */
                if (imm >= 0 && imm <= 4095) {
                    emit(e, jaiA64AddsXImm(dst, cur, (unsigned)imm));
                } else if (imm < 0 && imm >= -4095) {
                    emit(e, jaiA64SubsXImm(dst, cur, (unsigned)(-(int)imm)));
                } else {
                    emitConst64(e, JIT_SCRATCH_A, imm);
                    emit(e, jaiA64AddsX(dst, cur, JIT_SCRATCH_A));
                }
            }
            branchOnOverflow(e, 0u, JAI_A64_VS);
            off += 5;
            break;
        }

        case OP_SUB_INT_CONST: {
            unsigned slot = jaiReadU16(code + off + 1);
            int16_t  imm  = jaiReadI16(code + off + 3);
            if (!localInRange(e, slot)) return false;
            if (e->localKind[slot] != SLOT_INT) return false;
            if (slot == 0) e->usesSlot0 = true;
            if (!pushValue(e, SLOT_INT, 0, NULL)) return false;
            {
                unsigned dst = pushReg(e) - 1;
                unsigned cur = localIn(e, slot, JIT_SCRATCH_C);
                if (imm >= 0 && imm <= 4095) {
                    emit(e, jaiA64SubsXImm(dst, cur, (unsigned)imm));
                } else if (imm < 0 && imm >= -4095) {
                    emit(e, jaiA64AddsXImm(dst, cur, (unsigned)(-(int)imm)));
                } else {
                    emitConst64(e, JIT_SCRATCH_A, imm);
                    emit(e, jaiA64SubsXReg(dst, cur, JIT_SCRATCH_A));
                }
            }
            branchOnOverflow(e, 1u, JAI_A64_VS);
            off += 5;
            break;
        }

        case OP_MUL_INT_CONST: {
            unsigned slot = jaiReadU16(code + off + 1);
            int16_t  imm  = jaiReadI16(code + off + 3);
            if (!localInRange(e, slot)) return false;
            if (e->localKind[slot] != SLOT_INT) return false;
            if (slot == 0) e->usesSlot0 = true;
            if (!pushValue(e, SLOT_INT, 0, NULL)) return false;
            unsigned dst = pushReg(e) - 1;
            unsigned cur = localIn(e, slot, JIT_SCRATCH_C);
            unsigned rt = ovfDest(e, dst);
            /* MUL has no immediate form on this encoder, so the constant goes
             * into JIT_SCRATCH_D -- left free by `cur` and `rt` above, so it
             * cannot collide with either even when ovfDest hands back
             * JIT_SCRATCH_B inside a `try`. Same overflow test as plain
             * OP_MUL: the product overflows exactly when smulh's high half is
             * not the low half's sign bit replicated, and it shares that
             * arm's overflow-stub slot (2, the `*` message) since it is the
             * same operator. */
            emitConst64(e, JIT_SCRATCH_D, imm);
            emit(e, jaiA64SmulhX(JIT_SCRATCH_A, cur, JIT_SCRATCH_D));
            emit(e, jaiA64MulX(rt, cur, JIT_SCRATCH_D));
            emit(e, jaiA64SubsXAsr(31, JIT_SCRATCH_A, rt, 63));
            branchOnOverflow(e, 2u, JAI_A64_NE);
            if (rt != dst) emit(e, jaiA64MovX(dst, rt));
            off += 5;
            break;
        }

        case OP_MUL_BIND: {
            unsigned slot = jaiReadU16(code + off + 1);
            if (!localInRange(e, slot)) return false;
            if (slot == 0) e->usesSlot0 = true;
            if (!e->fpOff && e->depth >= 2 && !e->dynamicLocal[slot] &&
                e->stack[e->depth - 1] == SLOT_FLOAT &&
                e->stack[e->depth - 2] == SLOT_FLOAT) {
                if (!adoptLocalKind(e, slot, SLOT_FLOAT, 0, NULL)) {
                    e->whyNot = kindClash(e, slot);
                    return false;
                }
                unsigned ib = e->valueDepth - 1, ia = e->valueDepth - 2;
                unsigned db = fpOperand(e, ib), da = fpOperand(e, ia);
                unsigned rx; SlotKind kx;
                if (!popValueRaw(e, &rx, &kx)) return false;
                if (!popValueRaw(e, &rx, &kx)) return false;
                unsigned dd = fpBindDest(e, slot, fpRegAt(e, ia));
                emit(e, jaiA64FmulD(dd, da, db));
                localOutFp(e, slot, dd);
                off += 3;
                break;
            }
            unsigned rb, ra;
            SlotKind kb, ka;
            if (!popValue(e, &rb, &kb)) return false;
            if (!popValue(e, &ra, &ka)) return false;
            if (ka != kb) return false;
            if (!adoptLocalKind(e, slot, ka, 0, NULL)) {
                e->whyNot = kindClash(e, slot);
                return false;
            }
            unsigned rd = localDest(e, slot);
            if (ka == SLOT_FLOAT) {
                emit(e, jaiA64FmovDX(JIT_FSCRATCH_A, ra));
                emit(e, jaiA64FmovDX(JIT_FSCRATCH_B, rb));
                emit(e, jaiA64FmulD(JIT_FSCRATCH_A, JIT_FSCRATCH_A,
                                    JIT_FSCRATCH_B));
                emit(e, jaiA64FmovXD(rd, JIT_FSCRATCH_A));
            } else if (ka == SLOT_INT) {
                rd = ovfDest(e, rd);   /* localOut copies it home below */
                emit(e, jaiA64SmulhX(JIT_SCRATCH_A, ra, rb));
                emit(e, jaiA64MulX(rd, ra, rb));
                emit(e, jaiA64SubsXAsr(31, JIT_SCRATCH_A, rd, 63));
                branchOnOverflow(e, 2u, JAI_A64_NE);
            } else {
                return subWhy(e, "arithmetic on a %s", slotKindName(ka));
            }
            localOut(e, slot, rd);
            off += 3;
            break;
        }

        case OP_SUB_BIND: {
            unsigned slot = jaiReadU16(code + off + 1);
            if (!localInRange(e, slot)) return false;
            if (slot == 0) e->usesSlot0 = true;
            if (!e->fpOff && e->depth >= 2 && !e->dynamicLocal[slot] &&
                e->stack[e->depth - 1] == SLOT_FLOAT &&
                e->stack[e->depth - 2] == SLOT_FLOAT) {
                if (!adoptLocalKind(e, slot, SLOT_FLOAT, 0, NULL)) {
                    e->whyNot = kindClash(e, slot);
                    return false;
                }
                unsigned ib = e->valueDepth - 1, ia = e->valueDepth - 2;
                unsigned db = fpOperand(e, ib), da = fpOperand(e, ia);
                unsigned rx; SlotKind kx;
                if (!popValueRaw(e, &rx, &kx)) return false;
                if (!popValueRaw(e, &rx, &kx)) return false;
                unsigned dd = fpBindDest(e, slot, fpRegAt(e, ia));
                emit(e, jaiA64FsubD(dd, da, db));
                localOutFp(e, slot, dd);
                off += 3;
                break;
            }
            unsigned rb, ra;
            SlotKind kb, ka;
            if (!popValue(e, &rb, &kb)) return false;
            if (!popValue(e, &ra, &ka)) return false;
            if (ka != kb) return false;
            if (!adoptLocalKind(e, slot, ka, 0, NULL)) {
                e->whyNot = kindClash(e, slot);
                return false;
            }
            unsigned rd = localDest(e, slot);
            if (ka == SLOT_FLOAT) {
                emit(e, jaiA64FmovDX(JIT_FSCRATCH_A, ra));
                emit(e, jaiA64FmovDX(JIT_FSCRATCH_B, rb));
                emit(e, jaiA64FsubD(JIT_FSCRATCH_A, JIT_FSCRATCH_A,
                                    JIT_FSCRATCH_B));
                emit(e, jaiA64FmovXD(rd, JIT_FSCRATCH_A));
            } else if (ka == SLOT_INT) {
                rd = ovfDest(e, rd);   /* localOut copies it home below */
                emit(e, jaiA64SubsXReg(rd, ra, rb));
                branchOnOverflow(e, 1u, JAI_A64_VS);
            } else {
                return subWhy(e, "arithmetic on a %s", slotKindName(ka));
            }
            localOut(e, slot, rd);
            off += 3;
            break;
        }

        case OP_BIND: {
            unsigned slot = jaiReadU16(code + off + 1);
            if (!localInRange(e, slot)) return false;
            if (slot == 0) e->usesSlot0 = true;
            if (e->depth == 0 || !holdsRegister(e->stack[e->depth - 1])) return false;
            if (!adoptLocalKindSeen(e, slot, e->stack[e->depth - 1],
                                    e->stackShape[e->depth - 1],
                                    e->stackClass[e->depth - 1],
                                    e->stackSeen[e->depth - 1])) {
                e->whyNot = kindClash(e, slot);
                return false;
            }
            if (e->stackElemDecl[e->depth - 1] != 0) {
                e->localElemDecl[slot] = e->stackElemDecl[e->depth - 1];
            }
            e->localObjType[slot] = e->stackObjType[e->depth - 1];
            if (!e->fpOff && !e->dynamicLocal[slot] &&
                e->stack[e->depth - 1] == SLOT_FLOAT &&
                (e->fpLive & (1u << (e->valueDepth - 1)))) {
                unsigned idx = e->valueDepth - 1;
                unsigned held = fpHeldIn(e, idx);
                unsigned r2; SlotKind k2;
                if (!popValueRaw(e, &r2, &k2)) return false;
                localOutFp(e, slot, held);
                off += 3;
                break;
            }
            unsigned r;
            if (!popValue(e, &r, NULL)) return false;
            localOut(e, slot, r);
            off += 3;
            break;
        }

        case OP_INC_LOCAL: {
            unsigned slot = jaiReadU16(code + off + 1);
            int8_t   imm  = (int8_t)code[off + 3];
            if (!localInRange(e, slot)) return false;
            if (e->localKind[slot] != SLOT_INT) return false;
            if (slot == 0) e->usesSlot0 = true;
            {
                unsigned cur = localIn(e, slot, JIT_SCRATCH_C);
                unsigned dst = ovfDest(e, localDest(e, slot));
                /* Step is an i8, always fits imm12, so the constant never needs a register of its own. `subs` for a
                 * negative step rather than a negated `adds`: both set V for the operation actually performed, which is what the overflow guard below reads. */
                if (imm >= 0) {
                    emit(e, jaiA64AddsXImm(dst, cur, (unsigned)imm));
                } else {
                    emit(e, jaiA64SubsXImm(dst, cur, (unsigned)(-(int)imm)));
                }
                /* The guard is taken before the home is written, not after: it
                 * resumes at this instruction inside a `try` (ovfDest), and
                 * neither fpSyncAll nor a b.cond disturbs V or `dst`. */
                branchOnOverflow(e, 0u, JAI_A64_VS);
                localOut(e, slot, dst);
            }
            off += 4;
            break;
        }

        case OP_EQ: case OP_NE:
        case OP_LT: case OP_LE: case OP_GT: case OP_GE:
            if (!emitCompare(e, op, &off)) return false;
            break;

        /* `x is null` and `x is not null`, which is how every optional in
         * this language is tested. `valueIsTest` against a null target is
         * plain identity, so the whole question is whether the subject's tag
         * is VAL_NULL -- and the tier already represents a null instance as a
         * zero in the register, so for the kind that actually occurs it is a
         * compare and a `cset`.
         *
         * Worth an arm on its own: without one the operator fell to the
         * unarmed-opcode deopt a few instructions into the body, and a body
         * that compiles and then immediately bails is SLOWER than one that was
         * never compiled -- measured at 212 ms against the interpreter's
         * 162 ms on a guard doing nothing else. It was the only construct
         * found that made the tier a net loss. */
        case OP_IS:
        case OP_IS_NOT:
            if (!emitIsTest(e, op, &off)) return false;
            break;

        case OP_JUMP_IF_FALSE:
        case OP_JUMP_IF_TRUE:
        case OP_JUMP_IF_FALSE_KEEP:
        case OP_JUMP_IF_TRUE_KEEP: {
            int16_t jump = jaiReadI16(code + off + 1);
            bool keep = (op == OP_JUMP_IF_FALSE_KEEP ||
                         op == OP_JUMP_IF_TRUE_KEEP);
            bool wantTrue = (op == OP_JUMP_IF_TRUE ||
                             op == OP_JUMP_IF_TRUE_KEEP);
            if (e->depth == 0 || e->stack[e->depth - 1] != SLOT_BOOL) {
                /* Named because the bare refusal was unreadable in a census:
                 * `_is_ident_cont` -- `c.is_alnum() or c == "_"`, called per
                 * character by the lexer -- reported only "OP_JUMP_IF_TRUE_KEEP"
                 * while three OSR loops retried it 80 times each waiting for it
                 * to compile. */
                return subWhy(e, "a branch on a %s, not a bool",
                              e->depth > 0
                                  ? slotKindName(e->stack[e->depth - 1])
                                  : "empty stack");
            }
            unsigned r = pushReg(e) - 1;
            if (!keep) {
                unsigned popped;
                if (!popValue(e, &popped, NULL)) return false;
            }
            emit(e, jaiA64SubsXImm(31, r, 0));
            branchTo(e, (uint32_t)((int32_t)(off + 3) + jump), true,
                     wantTrue ? JAI_A64_NE : JAI_A64_EQ);
            off += 3;
            break;
        }

        case OP_JUMP_IF_CMP_FALSE:
            if (!emitJumpIfCmpFalse(e, code, &off)) return false;
            break;

        case OP_POP: {
            unsigned r;
            /* Nothing reads a value that is being thrown away, so a deferred
             * entry is forgotten rather than settled. Without this, `a = b`
             * paid at the POP the copy the borrow had just saved. */
            if (e->depth > 0 && holdsRegister(e->stack[e->depth - 1]) &&
                e->valueDepth > 0) {
                unsigned idx = e->valueDepth - 1;
                e->kPend   &= ~(1u << idx);
                e->xBorrow &= ~(1u << idx);
            }
            if (!popValue(e, &r, NULL)) return false;
            off += 1;
            break;
        }

        case OP_NULL: {
            /* Zero, which is what a null instance is in a register. */
            if (!pushValue(e, SLOT_MAYBE_INST, 0, NULL)) return false;
            emit(e, jaiA64MovzX(pushReg(e) - 1, 0, 0));
            /* The kind says "may be null"; this says "IS null", which is what
             * the `is` arm needs and cannot recover from the kind. */
            e->stackNullLit[e->depth - 1] = true;
            off += 1;
            break;
        }

        case OP_CONST: {
            uint32_t idx = jaiReadU24(code + off + 1);
            if (idx >= (uint32_t)fn->chunk.constants.count) return false;
            Value k = fn->chunk.constants.data[idx];
            if (IS_INT(k)) {
                if (!pushValue3(e, SLOT_INT, 0, NULL, k, -1)) return false;
                int64_t kv = AS_INT(k);
                if (off + 4 < stop && foldsIntLiteral(code[off + 4]) &&
                    kv >= -4095 && kv <= 4095) {   /* see OP_INT */
                    kPendLocal(e, e->valueDepth - 1, kv);
                } else {
                    emitConst64(e, pushReg(e) - 1, kv);
                }
            } else if (IS_FLOAT(k)) {
                /* The bits, not the number: a float lives in an X register exactly as in a Value's payload -- unless
                 * the value is one FMOV's 8-bit immediate can name, in which case it goes straight to the FP bank in one instruction instead of two plus an fmov. */
                double d = AS_FLOAT(k);
                int64_t bits;
                memcpy(&bits, &d, sizeof bits);
                if (!pushValue3(e, SLOT_FLOAT, 0, NULL, k, -1)) return false;
                unsigned imm8;
                /* The lookahead, not just the encodability: a constant with no float consumer ahead of it is
                 * synced back out to X at the next ordinary opcode, so the bank form costs `fmov d,#imm` plus that `fmov x,d` where one `movz` would have done. Every value FMOV's imm8 can name has its whole payload in the top 16 bits, so emitConst64 is exactly one instruction for all of them. spectral's `1.0 / float(..)` is the shape: OP_GET_GLOBAL stands between the constant and its divide. */
                if (!e->fpOff && jaiA64FpImm8(d, &imm8) &&
                    fpWorthLoading(e, code, off + 4, stop)) {
                    /* Not `idx`: that is this instruction's constant index, and
                     * shadowing it here was the tree's only build warning. This
                     * one is a position on the operand stack. */
                    unsigned at = e->valueDepth - 1;
                    emit(e, jaiA64FmovDImm(fpRegAt(e, at), imm8));
                    fpClaim(e, at);
                } else {
                    emitConst64(e, pushReg(e) - 1, bits);
                }
            } else if (IS_BOOL(k)) {
                if (!pushValue3(e, SLOT_BOOL, 0, NULL, k, -1)) return false;
                emitConst64(e, pushReg(e) - 1, AS_BOOL(k) ? 1 : 0);
            } else if (IS_STRING(k)) {
                /* Pointer to the constant pool's own string, safe to hold raw for the same reason resolved globals
                 * are: the pool belongs to the chunk, the chunk to the function, and the caller holds the closure for the whole call. Was the commonest reason this tier declined a body -- thirty refusals across the benchmark suite. */
                if (!pushValue3(e, SLOT_OBJ, 0, NULL, k, -1)) return false;
                emitConst64(e, pushReg(e) - 1, (int64_t)(uintptr_t)AS_OBJ(k));
            } else {
                return subWhy(e, "a constant of a kind the tier cannot hold");
            }
            off += 4;
            break;
        }

        case OP_TYPE_GUARD: {
            /* A declared boundary (parameter or return type). The kind is already known here, so the guard is
             * either nothing at all or the int-to-float widening the interpreter does at the same place (spec 2.2). Anything the kinds can't settle is declined, not guessed: `evalA` in spectral was refused outright for want of this. */
            uint32_t idx = jaiReadU24(code + off + 1);
            if (idx >= (uint32_t)fn->chunk.constants.count) return false;
            Value t = fn->chunk.constants.data[idx];
            if (!IS_STRING(t)) return false;
            if (e->depth == 0) return false;
            const char *tn = AS_STRING(t)->chars;
            SlotKind k = e->stack[e->depth - 1];
            if (strcmp(tn, "float") == 0) {
                if (k == SLOT_INT) {
                    unsigned r = pushReg(e) - 1;
                    emit(e, jaiA64ScvtfDX(JIT_FSCRATCH_A, r));
                    emit(e, jaiA64FmovXD(r, JIT_FSCRATCH_A));
                    e->stack[e->depth - 1]      = SLOT_FLOAT;
                    e->stackShape[e->depth - 1] = 0;
                    e->stackClass[e->depth - 1] = NULL;
                    e->stackSeen[e->depth - 1]  = NULL_VAL;
                    e->stackLocal[e->depth - 1] = -1;
                    e->stackAscii[e->depth - 1] = false;
                    e->stackNullLit[e->depth - 1] = false;
                    e->stackUnit[e->depth - 1]  = false;
                    e->stackObjType[e->depth - 1] = 0;
                    e->stackElem[e->depth - 1] = NULL_VAL;
                } else if (k != SLOT_FLOAT) {
                    e->whyNot = "a type guard the kinds cannot settle";
                    return false;
                }
            } else if ((strcmp(tn, "int") == 0 && k == SLOT_INT) ||
                       (strcmp(tn, "bool") == 0 && k == SLOT_BOOL)) {
            } else if (jitAnyGuard() && strcmp(tn, "list") == 0 &&
                       k == SLOT_OBJ) {
                /* A declared `list` boundary on a value the model has only as
                 * "some heap object". Unlike the cases below this one is not a
                 * no-op: it NARROWS. Proving OBJ_LIST here turns the entry into
                 * a SLOT_LIST, and every list arm downstream -- indexing,
                 * iteration, `len` -- asks for exactly that.
                 *
                 * The guard is what makes the narrowing honest. VAL_OBJ covers
                 * every heap object, so without the Obj.type check the next arm
                 * would read ObjList's header off whatever this is, which is
                 * the hole that segfaulted the VM through the dict-index arm.
                 *
                 * A deopt here resumes AT this instruction with the operand
                 * still on the interpreter's stack, so a miss costs nothing
                 * beyond the exit.
                 *
                 * WORTH ZERO ON ITS OWN, and recorded as such: it takes the
                 * "a `list` guard on a object" refusal in the jaicv benchmark
                 * from four to none, and `min_enclosing_circle`'s attributed
                 * interpreted work does not move -- the body goes from
                 * declining HERE to declining one instruction later, at
                 * `OP_GET_INDEX: the container has kind int, not list`, after
                 * which JAI_JIT_CHAIN says it compiles. Kept because it is six
                 * guarded lines that narrow rather than guess, and because the
                 * link it exposes is the one being worked next. */
                unsigned gr = valueXReg(e, e->valueDepth - 1);
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, gr,
                                   (unsigned)offsetof(Obj, type)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_LIST));
                branchOnDeopt(e, JAI_A64_NE);
                e->stack[e->depth - 1] = SLOT_LIST;
            } else if (jitAnyGuard() && strcmp(tn, "list") == 0 &&
                       k == SLOT_LIST) {
                /* SLOT_LIST is a guarded fact, not a hope: every arm that puts
                 * one on the stack has already proved OBJ_LIST, because VAL_OBJ
                 * covers every heap object and the shared return path says so
                 * in as many words. `jaiTypeNameStatic` calls an ObjList "list"
                 * (value.c), which is the name this guard compares against, so
                 * there is nothing left to check.
                 *
                 * Same exposure as the `float`/`int`/`bool` cases above and no
                 * more: all four reason from the type NAME, and a module global
                 * shadowing one of those names with a class would change what
                 * the interpreter does. That is the arm's existing contract. */
            } else if (jitAnyGuard() && k == SLOT_INST &&
                       e->stackClass[e->depth - 1] != NULL &&
                       e->stackClass[e->depth - 1]->name != NULL &&
                       strcmp(e->stackClass[e->depth - 1]->name->chars, tn) == 0) {
                /* The pinned class IS the one the boundary names. Like
                 * SLOT_LIST this is a guarded fact -- every arm that puts a
                 * SLOT_INST on the stack with a class pinned has emitted the
                 * shapeId check that proves it -- so the guard has nothing left
                 * to do.
                 *
                 * An exact name match is sufficient, not necessary:
                 * jaiValueMatchesType also accepts a subclass and a trait
                 * implementer, so anything this does not recognise still
                 * declines rather than being waved through. The names that
                 * turn up are `Tensor`, `Mat` and `NDArray` -- the ML packages
                 * annotate their boundaries, so one of these sat in the middle
                 * of a hot body and declined all of it. */
            } else if (jitAnyGuard() && strcmp(tn, "any") == 0) {
                /* `any` is satisfied by every value, so this guard is a no-op
                 * for every kind -- not a narrowing the tier is guessing at.
                 * jaiValueMatchesType returns true for the name "any" before
                 * looking at the subject at all (vm.c), which is what makes
                 * emitting nothing here the same thing the interpreter does.
                 *
                 * It is common enough to matter because `dict[str, any]` is how
                 * this compiler carries AST records: a parameter or return
                 * declared `any` put one of these in the middle of a body and
                 * declined all of it. */
            } else {
                return subWhy(e, "a `%s` guard on a %s", tn, slotKindName(k));
            }
            off += 4;
            break;
        }

        case OP_FORMAT: {
            /* Largest single refusal reason across the benchmark census (ninety) -- every f-string is one, and
             * dict_ops, word_freq and string_build all build their keys with one. */
            unsigned parts = code[off + 1];
            if (parts == 0 || parts > JIT_MAX_ARGS_OUT) {
                e->whyNot = "an f-string with more parts than the descriptor holds";
                return false;
            }
            if (e->depth < parts) return false;
            /* `str` bound in the module means every part goes through it
             * instead, which is a call this does not make. */
            {
                ObjModule *fmod = closure->fn->module;
                Value bound;
                ObjString *sname = jaiStringIntern("str", 3);
                if (fmod == NULL || sname == NULL ||
                    jaiTableGetInterned(&fmod->globals, sname, &bound)) {
                    e->whyNot = "the module binds its own str";
                    return false;
                }
            }
            if (!emitDescriptor(e, NULL_VAL, e->depth - parts, parts,
                                (void *)&jitFormat)) {
                return false;
            }
            for (unsigned i = 0; i < parts; i++) {
                unsigned drop;
                if (!popValue(e, &drop, NULL)) return false;
            }
            if (!pushValue(e, SLOT_OBJ, 0, NULL)) return false;
            /* jitFormat always builds a string, and there is no Value to carry
             * as a sample, so the expectation is recorded instead: without it
             * `f"{a}-{b}".len()` declined the loop around it at the very next
             * instruction ("an invoke on an object with nothing to look at"). */
            e->stackObjType[e->depth - 1] = (uint8_t)(OBJ_STRING + 1);
            emit(e, jaiA64LdrX(pushReg(e) - 1, 31,
                               e->descOffset +
                                   (unsigned)offsetof(JitCallDesc, result) + 8));
            e->wroteHeap = true;
            /* count u8, litmask u24, name u24, cache u16 -- nine after the
             * opcode. */
            off += 10;
            break;
        }

        case OP_JUMP: {
            int16_t jump = jaiReadI16(code + off + 1);
            branchTo(e, (uint32_t)((int32_t)(off + 3) + jump), false, 0);
            off += 3;
            break;
        }

        case OP_LOOP: {
            /* Compiled code has no safepoint on the back edge (uninterruptible, unsampled) -- acceptable only
             * because this tier bails on anything unbounded: the loop is over ints, can't allocate, and the stack guard still catches runaway recursion. Ctrl-C during a long compiled loop waits for the loop to end. */
            int16_t jump = jaiReadI16(code + off + 1);
            branchTo(e, (uint32_t)((int32_t)(off + 3) + jump), false, 0);
            off += 3;
            break;
        }

        case OP_MUL: {
            unsigned rb, ra;
            SlotKind kb, ka;
            if (e->depth >= 2 && e->stack[e->depth - 1] == SLOT_FLOAT &&
                e->stack[e->depth - 2] == SLOT_FLOAT) {
                unsigned ib = e->valueDepth - 1, ia = e->valueDepth - 2;
                unsigned db = fpOperand(e, ib), da = fpOperand(e, ia);
                if (!popValueRaw(e, &rb, &kb)) return false;
                if (!popValueRaw(e, &ra, &ka)) return false;
                if (!pushValue(e, SLOT_FLOAT, 0, NULL)) return false;
                unsigned dm = fpRegAt(e, ia);
                uint32_t bindOffM = 0;
                unsigned homeM = fpBindLookahead(e, code, off + 1, stop, fn,
                                                 &bindOffM);
                if (homeM != 0) {
                    fpReleaseHome(e, homeM);
                    dm = homeM;
                }
                emit(e, jaiA64FmulD(dm, da, db));
                if (homeM != 0) {
                    fpBorrowLocal(e, ia, homeM);
                    e->homeEarly[e->homeEarlyCount++] = bindOffM;
                } else {
                    fpClaim(e, ia);
                }
                off += 1;
                break;
            }
            if (!popValue(e, &rb, &kb)) return false;
            if (!popValue(e, &ra, &ka)) return false;
            if (ka != kb) return false;
            if (ka != SLOT_INT) return false;
            if (!pushValue(e, SLOT_INT, 0, NULL)) return false;
            unsigned rd = pushReg(e) - 1;
            unsigned rt = ovfDest(e, rd);
            /* The product overflows exactly when the high half is not the low
             * half's sign bit replicated, so smulh and one shifted compare
             * decide it. mul must come after smulh reads its inputs, since rd
             * may be one of them -- which is also why `rt` differs inside a
             * `try`: rd IS the first operand's register, and the guard resumes
             * at an instruction whose operands the interpreter still needs. */
            emit(e, jaiA64SmulhX(JIT_SCRATCH_A, ra, rb));
            emit(e, jaiA64MulX(rt, ra, rb));
            emit(e, jaiA64SubsXAsr(31, JIT_SCRATCH_A, rt, 63));
            branchOnOverflow(e, 2u, JAI_A64_NE);
            if (rt != rd) emit(e, jaiA64MovX(rd, rt));
            off += 1;
            break;
        }

        case OP_BAND:
        case OP_BOR:
        case OP_BXOR:
        case OP_SHL:
        case OP_SHR: {
            /* None of `& | ^` can fail, so they're one instruction. Shifts have two edges hardware doesn't share:
             * jaithon throws on a negative count and saturates at >=64, while arm64's LSLV/ASRV wrap on the low six bits of the count. Both edges are guarded and deopt; the guards precede the pops, so a deopt resumes at an instruction that hasn't happened yet. */
            unsigned rb, ra;
            SlotKind kb, ka;
            if (e->depth < 2) return false;
            if (e->stack[e->depth - 1] != SLOT_INT) return false;
            if (e->stack[e->depth - 2] != SLOT_INT) return false;

            /* A literal count settles both edges here rather than at run time: the immediate-form shift then IS
             * the interpreter's rule for that count. bitops shifts by 7, 3, 11, 1 and 31 and paid five instructions and two deopt sites for each before this. */
            int64_t kcount;
            bool kcountUsable =
                (op == OP_SHL || op == OP_SHR) &&
                literalIntOperand(fn, prevOff, off, &kcount) &&
                kcount >= 0 && kcount <= 63;

            if ((op == OP_SHL || op == OP_SHR) && !kcountUsable) {
                /* The count reaches a register and a guard, so it has to be
                 * in one. OP_INT may have deferred it on the strength of the
                 * opcode alone -- a count of 100 is a literal this arm cannot
                 * fold -- and this is where that is put right. */
                settleAll(e);
            }
            if (op != OP_SHL && op != OP_SHR) settleAll(e);
            if (kcountUsable) e->kPend &= ~(1u << (e->valueDepth - 1));

            if ((op == OP_SHL || op == OP_SHR) && !kcountUsable) {
                unsigned rCount = pushReg(e) - 1;
                emit(e, jaiA64SubsXImm(31, rCount, 0));
                branchOnDeopt(e, JAI_A64_LT);            /* negative count */
                emitConst64(e, JIT_SCRATCH_A, 64);
                emit(e, jaiA64SubsXReg(31, rCount, JIT_SCRATCH_A));
                branchOnDeopt(e, JAI_A64_GE);            /* 64 or more */
            }

            if (!popValue(e, &rb, &kb)) return false;
            if (!popValue(e, &ra, &ka)) return false;
            if (!pushValue(e, SLOT_INT, 0, NULL)) return false;
            unsigned rd = pushReg(e) - 1;
            if (kcountUsable) {
                emit(e, op == OP_SHL ? jaiA64LslX(rd, ra, (unsigned)kcount)
                                     : jaiA64AsrX(rd, ra, (unsigned)kcount));
                off += 1;
                break;
            }
            emit(e, op == OP_BAND ? jaiA64AndX(rd, ra, rb)
                  : op == OP_BOR  ? jaiA64OrrX(rd, ra, rb)
                  : op == OP_BXOR ? jaiA64EorX(rd, ra, rb)
                  : op == OP_SHL  ? jaiA64LslvX(rd, ra, rb)
                                  : jaiA64AsrvX(rd, ra, rb));
            off += 1;
            break;
        }

        case OP_ADD:
        case OP_SUB:
        case OP_DIV: {
            unsigned rb, ra;
            SlotKind kb, ka;
            /* `head + "/" + right`: two strings, out to jaiStringConcat, the
             * unfused half of the ADD_BIND arm above. Measured on a self-hosted
             * `check`, where OP_ADD is a few percent of the interpreted work:
             * no change (775ms -> 767ms, noise). The bodies holding it decline
             * for other reasons anyway, so this arm earns its place on the
             * ADD_BIND shape and not on the compiler. */
            if (op == OP_ADD) {
                Value csample;
                if (concatOperands(e, &csample)) {
                    if (!emitStringConcat(e, csample)) return false;
                    off += 1;
                    break;
                }
            }
            if (e->depth >= 2 && e->stack[e->depth - 1] == SLOT_FLOAT &&
                e->stack[e->depth - 2] == SLOT_FLOAT) {
                unsigned ib = e->valueDepth - 1, ia = e->valueDepth - 2;
                unsigned db = fpOperand(e, ib), da = fpOperand(e, ia);
                if (!popValueRaw(e, &rb, &kb)) return false;
                if (!popValueRaw(e, &ra, &ka)) return false;
                if (!pushValue(e, SLOT_FLOAT, 0, NULL)) return false;
                unsigned dd = fpRegAt(e, ia);
                uint32_t bindOff = 0;
                unsigned home = fpBindLookahead(e, code, off + 1, stop, fn,
                                                &bindOff);
                if (home != 0) {
                    fpReleaseHome(e, home);
                    dd = home;
                }
                emit(e, op == OP_ADD ? jaiA64FaddD(dd, da, db)
                     : op == OP_SUB  ? jaiA64FsubD(dd, da, db)
                                     : jaiA64FdivD(dd, da, db));
                if (home != 0) {
                    fpBorrowLocal(e, ia, home);
                    e->homeEarly[e->homeEarlyCount++] = bindOff;
                } else {
                    fpClaim(e, ia);
                }
                off += 1;
                break;
            }
            /* A literal right operand becomes the immediate `adds`/`subs`
             * already takes, so it never reaches a register. Decided before
             * the pops, which is where the entry index still names it. */
            int64_t kimm = 0;
            bool foldK = op != OP_DIV && e->depth >= 2 &&
                         e->stack[e->depth - 1] == SLOT_INT &&
                         e->stack[e->depth - 2] == SLOT_INT &&
                         pendingImm12(e, e->valueDepth - 1, &kimm);
            if (foldK) e->kPend &= ~(1u << (e->valueDepth - 1));

            /* Carried through the arithmetic: see Emit::idxKnown. Only the
             * folded-literal form, which is exactly what `xs[j - 1]` and
             * `xs[j + 1]` compile to, and only while the offset stays small --
             * the loop head's guard has to add it to a count without
             * overflowing, and a bound keeps that argument short. The overflow
             * branch below deoptimises, so a shape that survives to the
             * subscript describes arithmetic that actually happened. */
            bool    idxCarry     = false;
            uint8_t idxCarryBase = 0;
            int32_t idxCarryOff  = 0;
            if (foldK && e->valueDepth >= 2 &&
                (e->idxKnown & (1u << (e->valueDepth - 2))) != 0 &&
                kimm >= -4096 && kimm <= 4096) {
                unsigned at = e->valueDepth - 2;
                int64_t sum = (int64_t)e->idxOff[at] +
                              (op == OP_SUB ? -kimm : kimm);
                if (sum >= -4096 && sum <= 4096) {
                    idxCarry     = true;
                    idxCarryBase = e->idxBase[at];
                    idxCarryOff  = (int32_t)sum;
                }
            }

            if (!popValue(e, &rb, &kb)) return false;
            if (!popValue(e, &ra, &ka)) return false;
            if (ka != kb) return false;   /* no implicit widening here */

            /* Integer division is not here: it has a zero-divisor error and a
             * truncation rule of its own, and getting either wrong would be a
             * wrong answer rather than a decline. */
            if (ka != SLOT_INT || op == OP_DIV) return false;
            if (!pushValue(e, SLOT_INT, 0, NULL)) return false;
            if (idxCarry) {
                unsigned at = e->valueDepth - 1;
                e->idxKnown |= 1u << at;
                e->idxBase[at] = idxCarryBase;
                e->idxOff[at]  = idxCarryOff;
            }
            unsigned rd = pushReg(e) - 1;
            /* rd is the first operand's own register (two off, one on), so
             * inside a `try` the sum computes elsewhere: the overflow guard
             * resumes at this instruction and the interpreter reads the
             * operands back off its stack. See ovfDest. */
            unsigned rt = ovfDest(e, rd);
            if (foldK) emitAddSubImm(e, rt, ra, kimm, op == OP_SUB);
            else emit(e, op == OP_ADD ? jaiA64AddsX(rt, ra, rb)
                                      : jaiA64SubsXReg(rt, ra, rb));
            branchOnOverflow(e, op == OP_ADD ? 0u : 1u, JAI_A64_VS);
            if (rt != rd) emit(e, jaiA64MovX(rd, rt));
            off += 1;
            break;
        }

        case OP_JUMP_IF_CMP_LOCAL_K:
            if (!emitJumpIfCmpLocalK(e, fn, code, &off)) return false;
            break;

        case OP_GET_FIELD_LOCAL: {
            unsigned slot    = jaiReadU16(code + off + 1);
            uint32_t nameIdx = jaiReadU24(code + off + 3);

            /* A maybe-instance reads like an instance once known not-null. The program has usually just tested
             * it (`if node == null { return 0 }`) but the tier doesn't track that, so the guard stands and costs one compare against zero; a null arriving for real just deopts. */
            if (e->localKind[slot] != SLOT_INST &&
                e->localKind[slot] != SLOT_MAYBE_INST) {
                return subWhy(e, "receiver local %u has kind %s, not instance",
                              slot, slotKindName(e->localKind[slot]));
            }
            if (e->localClass[slot] == NULL) {
                return subWhy(e, "receiver local %u has no pinned class", slot);
            }
            /* The name is resolved BEFORE the receiver guard is emitted, so
             * that a name this arm cannot read reaches `unarmedOpcode` with
             * nothing half-emitted -- the unarmed path takes over at the start
             * of an instruction. */
            if (nameIdx >= (uint32_t)fn->chunk.constants.count) {
                return subWhy(e, "the field name is not in the pool");
            }
            Value nameVal = fn->chunk.constants.data[nameIdx];
            if (!IS_STRING(nameVal)) return subWhy(e, "the field name is not a string");

            const FieldInfo *info =
                jaiClassFieldInfo(e->localClass[slot], AS_STRING(nameVal));
            /* The commonest of the four by a wide margin, and the one that
             * reads as a puzzle without being named: `self.method` is a METHOD,
             * and jaiClassFieldInfo only knows fields, so every
             * `return self.m(x)` written as a value lands here.
             *
             * It is 88 of resolve.jai's 104 OP_GET_FIELD_LOCAL declines, and
             * every one is the same shape: a keyword call to a private method,
             * `self._error(code, start, end, msg, help: "...")`. A keyword
             * argument rules out OP_INVOKE, so codegen has to materialise the
             * callee, and reading a method name as a value lands here.
             *
             * Those calls are on ERROR paths -- the lexer reaching a malformed
             * token -- which is what makes the unarmed path the right answer
             * rather than a decline, and the same reason OP_GET_GLOBAL takes it
             * for `throw ValueError(...)`. Softening a HOT refusal is a loss:
             * the same change on OP_GET_INDEX's `d[k]` measured 18% slower,
             * because a body that deopts every iteration costs more than one
             * that is simply interpreted. Cold is the whole condition.
             *
             * `!e->osr` is the same rule again, and measured too: inside a loop
             * nothing is cold. Dropping that guard compiles six MORE bodies in
             * resolve.jai (277 -> 283) and cancels the win -- 0.272s becomes
             * 0.274-0.294s against 0.274-0.284s for the hard refusal, where the
             * non-OSR-only form was a clean 3%. A loop that deopts inside
             * itself pays the entry and the exit every iteration. */
            if (info == NULL) {
                if (!e->osr && jitSoftField()) goto unarmedOpcode;
                return subWhy(e, "`%s` is not a field of %s",
                              AS_STRING(nameVal)->chars,
                              e->localClass[slot]->name
                                  ? e->localClass[slot]->name->chars : "?");
            }
            if (e->localKind[slot] == SLOT_MAYBE_INST) {
                unsigned rcv = localIn(e, slot, JIT_SCRATCH_A);
                emit(e, jaiA64SubsXImm(31, rcv, 0));
                branchOnDeopt(e, JAI_A64_EQ);
            }
            if (slot == 0) e->usesSlot0 = true;
            if (info->isStatic) {
                return subWhy(e, "`%s` is a static", AS_STRING(nameVal)->chars);
            }

            /* Field type is read off the LIVE receiver, so the tier specialises to what the program actually
             * stores rather than a declaration -- only possible for a parameter (hence the arity cap above): a local assigned further in has no value yet to look at. */
            Value seen = seenLocal(e, slot);
            if (!IS_INSTANCE(seen)) {
                /* No sample -- but the DECLARATION may still say enough, and
                 * the receiver is a proven instance either way: localKind is
                 * SLOT_INST or SLOT_MAYBE_INST with a pinned class, and the
                 * null check above has already run. Only the field's own kind
                 * was missing.
                 *
                 * The sibling arm at OP_GET_FIELD has had this since the
                 * accessor-then-field-read chain was fixed; this one did not,
                 * and the asymmetry was worth **5.2% of parser.jai's
                 * interpreted work** by exact attribution -- the largest single
                 * reason left once the declared-ANY fields were given a code.
                 *
                 * Everything below is the same shape as there: the tag is
                 * guarded, a list or str is proved by Obj.type as well because
                 * VAL_OBJ is every heap object, and a str carries an interned
                 * empty string as a sample so the arms downstream that ask
                 * `stringOperand` can fire. */
                SlotKind lkind;
                unsigned ltag;
                if (!jitDeclaredFieldKindEnabled() ||
                    !declaredScalarFieldKind(info->typeId, &lkind, &ltag)) {
                    return subWhy(e,
                        "no live receiver to read local %u's field off, and its "
                        "declared kind (%u) is not one this predicts",
                        slot, (unsigned)info->typeId);
                }
                unsigned lbase = (unsigned)offsetof(ObjInstance, fields) +
                                 (unsigned)info->slot * (unsigned)sizeof(Value);
                unsigned lrr = localIn(e, slot, JIT_SCRATCH_C);
                /* Guarded before anything is pushed, so a deopt resumes at this
                 * instruction with the interpreter's stack untouched. */
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, lrr, lbase));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, ltag));
                branchOnDeopt(e, JAI_A64_NE);
                Value lprobe = NULL_VAL;
                if (lkind == SLOT_LIST || info->typeId == FIELD_KIND_STR) {
                    unsigned want = lkind == SLOT_LIST ? (unsigned)OBJ_LIST
                                                       : (unsigned)OBJ_STRING;
                    emit(e, jaiA64LdrX(JIT_SCRATCH_B, lrr, lbase + 8));
                    emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_B,
                                       (unsigned)offsetof(Obj, type)));
                    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, want));
                    branchOnDeopt(e, JAI_A64_NE);
                    if (want == (unsigned)OBJ_STRING) {
                        ObjString *empty = jaiStringIntern("", 0);
                        if (empty == NULL) return false;
                        lprobe = OBJ_VAL((Obj *)empty);
                    }
                }
                if (!pushValue3(e, lkind, 0, NULL, lprobe, -1)) return false;
                if (!IS_NULL(lprobe)) {
                    e->stackObjType[e->depth - 1] = (uint8_t)(OBJ_STRING + 1);
                }
                if (lkind == SLOT_BOOL) {
                    /* One byte: BOOL_VAL writes only the union's bool member. */
                    emit(e, jaiA64LdrByte(pushReg(e) - 1, lrr, lbase + 8));
                } else {
                    emit(e, jaiA64LdrX(pushReg(e) - 1, lrr, lbase + 8));
                }
                off += 8;
                break;
            }
            ObjInstance *inst = AS_INSTANCE(seen);
            if (info->slot >= inst->fieldCount) {
                return subWhy(e, "field slot %u is past the live receiver's %d",
                              (unsigned)info->slot, inst->fieldCount);
            }
            Value fieldVal = inst->fields[info->slot];

            SlotKind kind;
            unsigned tag;
            ObjClass *fcls = NULL;
            if (IS_INT(fieldVal))        { kind = SLOT_INT;   tag = VAL_INT; }
            else if (IS_FLOAT(fieldVal)) { kind = SLOT_FLOAT; tag = VAL_FLOAT; }
            else if (IS_INSTANCE(fieldVal) || IS_NULL(fieldVal)) {
                /* A tree's `left` is a Node on one call and null on the next; a leaf has no class to read, so the
                 * receiver's own class is the guess (a nullable instance field is usually a linked structure) -- safe because the class guard below just deopts if wrong, never miscompiles. */
                kind = SLOT_MAYBE_INST;
                tag  = VAL_OBJ;
                fcls = IS_INSTANCE(fieldVal) ? AS_INSTANCE(fieldVal)->klass
                                             : e->localClass[slot];
                if (fcls == NULL) return false;
            }
            /* A list field earns the stronger kind: SLOT_OBJ can be passed and stored but not iterated,
             * indexed or pushed to, and `for x in self.items` / `self.items.push(v)` is the commonest thing
             * a class holding a list does. Paid for with the OBJ_LIST check below, since VAL_OBJ alone
             * would let a str reach a header read. */
            else if (IS_LIST(fieldVal)) { kind = SLOT_LIST; tag = VAL_OBJ; }
            /* A str/dict/set field, held raw -- the same contract as a SLOT_OBJ global or list element:
             * the tag guard below says "an object", the sample says which type it was, and every consumer
             * (index, invoke, compare) re-checks Obj.type for itself before it does anything type-specific.
             * Refusing this declined the whole enclosing FUNCTION, which is most object-oriented code. */
            /* A bool field. Most classes carry one, and without this arm the
             * whole enclosing function declined -- `_check_live` in jaicv's
             * cascade is a method whose only field read is a bool, and it cost
             * 107 ms against 11.8 ms for the byte-identical method over an int
             * field. The load below is a `ldrb`, because BOOL_VAL writes only
             * the union's one-byte member. */
            else if (IS_BOOL(fieldVal)) { kind = SLOT_BOOL; tag = VAL_BOOL; }
            else if (rawObjValue(fieldVal)) { kind = SLOT_OBJ; tag = VAL_OBJ; }
            else return false;

            unsigned base = (unsigned)offsetof(ObjInstance, fields) +
                            (unsigned)info->slot * (unsigned)sizeof(Value);
            /* The tag is checked every time, unless this body wrote the field
             * itself. A field is not typed by the runtime, so a later int
             * where a float was seen must bail rather than be read as one. */
            unsigned recv = localIn(e, slot, JIT_SCRATCH_C);
            SlotKind already = knownFieldKind(e, (int)slot, info->slot);
            if (kind == SLOT_MAYBE_INST) {
                /* Three guards, branch-free via a trick: when the field is null, the loads below are redirected at
                 * the receiver instead (already a live instance, per the entry guard/null check above) -- a real object of the right type, so nothing ever dereferences zero. */
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, recv, base));       /* tag */
                emit(e, jaiA64LdrX(JIT_SCRATCH_D, recv, base + 8));   /* value */
                emit(e, jaiA64MovzX(JIT_SCRATCH_B, 0, 0));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, VAL_OBJ));
                emit(e, jaiA64CselX(JIT_SCRATCH_D, JIT_SCRATCH_D,
                                    JIT_SCRATCH_B, JAI_A64_EQ));
                emit(e, jaiA64CselX(JIT_SCRATCH_B, JIT_SCRATCH_B,
                                    JIT_SCRATCH_A, JAI_A64_EQ));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_B, 0));
                branchOnDeopt(e, JAI_A64_NE);      /* neither object nor null */

                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_D, 0));
                emit(e, jaiA64CselX(JIT_SCRATCH_B, recv, JIT_SCRATCH_D,
                                    JAI_A64_EQ));
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_B,
                                   (unsigned)offsetof(Obj, type)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_INSTANCE));
                branchOnDeopt(e, JAI_A64_NE);

                emit(e, jaiA64LdrX(JIT_SCRATCH_A, JIT_SCRATCH_B,
                                   (unsigned)offsetof(ObjInstance, klass)));
                emitConst64(e, JIT_SCRATCH_B, (int64_t)(uintptr_t)fcls);
                if (fcls != e->localClass[slot]) {
                    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_D, 0));
                    emit(e, jaiA64CselX(JIT_SCRATCH_A, JIT_SCRATCH_B,
                                        JIT_SCRATCH_A, JAI_A64_EQ));
                }
                emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
                branchOnDeopt(e, JAI_A64_NE);
            } else if (already != SLOT_SELF) {
                kind = already;
            } else {
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, recv, base));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, tag));
                branchOnDeopt(e, JAI_A64_NE);
                /* "an object" is not "a list": every SLOT_LIST consumer reads the header with no check of
                 * its own, so the object type is confirmed here, once, before the kind is handed out.
                 * `already` is never SLOT_LIST (see recordFieldStore's caller), so this arm sees them all. */
                if (kind == SLOT_LIST) {
                    emit(e, jaiA64LdrX(JIT_SCRATCH_D, recv, base + 8));
                    emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_D,
                                       (unsigned)offsetof(Obj, type)));
                    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_LIST));
                    branchOnDeopt(e, JAI_A64_NE);
                }
            }

            if (kind == SLOT_MAYBE_INST || kind == SLOT_LIST) {
                if (!pushValue3(e, kind,
                                kind == SLOT_MAYBE_INST ? fcls->shapeId : 0,
                                kind == SLOT_MAYBE_INST ? fcls : NULL,
                                kind == SLOT_LIST ? fieldVal
                                : IS_INSTANCE(fieldVal) ? fieldVal : NULL_VAL,
                                -1)) {
                    return false;
                }
                emit(e, jaiA64MovX(pushReg(e) - 1, JIT_SCRATCH_D));
            } else {
                /* SLOT_OBJ needs the sample carried: it is the only record of which object type this was,
                 * and a consumer with nothing to look at declines. NULL_VAL when `already` overrode the
                 * observation, since then the stored kind and the sampled value need not agree. */
                if (!pushValue3(e, kind, 0, NULL,
                                kind == SLOT_OBJ && rawObjValue(fieldVal)
                                    ? fieldVal : NULL_VAL,
                                -1)) {
                    return false;
                }
                /* A float field goes straight to the FP bank when something
                 * downstream will read it there: `ldr d` instead of `ldr x`
                 * followed by the `fmov` its consumer would emit. */
                if (kind == SLOT_FLOAT &&
                    fpWorthLoading(e, code, off + 8, stop)) {
                    unsigned idx = e->valueDepth - 1;
                    emit(e, jaiA64LdrD(fpRegAt(e, idx), recv, base + 8));
                    fpClaim(e, idx);
                } else if (kind == SLOT_BOOL) {
                    /* One byte: BOOL_VAL writes only the union's `boolean`
                     * member, so the seven above it are whatever the field
                     * held before, and every SLOT_BOOL consumer tests the
                     * whole register. */
                    emit(e, jaiA64LdrByte(pushReg(e) - 1, recv, base + 8));
                } else {
                    emit(e, jaiA64LdrX(pushReg(e) - 1, recv, base + 8));
                }
            }
            off += 8;
            break;
        }

        case OP_GET_LOCAL2: {
            unsigned a = jaiReadU16(code + off + 1);
            unsigned b = jaiReadU16(code + off + 3);
            /* The second local may guard (if dynamic), and a guard can't be reached with a borrow live -- the
             * deopt stub writes float entries out of fpRegAt, where a borrowed one isn't. So the FIRST local takes a copy instead of a borrow whenever the second is going to guard: `dt * b.vx` in nbody's second loop is exactly this shape (`dt` has a home, `b` is dynamic). */
            bool guardFollows = b <= JIT_MAX_SLOTS && e->dynamicLocal[b];
            for (unsigned k = 0; k < 2; k++) {
                unsigned slot = k == 0 ? a : b;
                /* Named, for the reason given at OP_GET_LOCAL. */
                if (!localInRange(e, slot)) {
                    e->whyNot = "a local outside the compiled window";
                    return false;
                }
                if (e->localKind[slot] == SLOT_OPAQUE) {
                    e->whyNot = "a local of no known kind";
                    return false;
                }
                if (slot == 0) e->usesSlot0 = true;
                if (!pushValue3(e, e->localKind[slot], e->localShape[slot],
                                e->localClass[slot],
                                seenLocal(e, slot),
                                (int)slot)) {
                    return false;
                }
                /* Same seed as OP_GET_LOCAL, and this is the arm that matters:
                 * the emitter fuses `xs[j]` into GET_LOCAL2, so every subscript
                 * in a stencil arrives here and nowhere else. */
                if (e->localKind[slot] == SLOT_INT && slot <= UINT8_MAX) {
                    unsigned at = e->valueDepth - 1;
                    e->idxKnown |= 1u << at;
                    e->idxBase[at] = (uint8_t)slot;
                    e->idxOff[at]  = 0;
                }
                if (e->localKind[slot] == SLOT_FLOAT &&
                    !e->dynamicLocal[slot] &&
                    fpWorthLoading(e, code, off + 5, stop)) {
                    unsigned idx = e->valueDepth - 1;
                    if (e->slotFpReg[slot] != 0 && !(k == 0 && guardFollows)) {
                        fpBorrowLocal(e, idx, e->slotFpReg[slot]);
                    } else {
                        localInFp(e, slot, fpRegAt(e, idx));
                        fpClaim(e, idx);
                    }
                } else {
                    unsigned home = localHomeX(e, slot);
                    if (home != 0) {            /* see OP_GET_LOCAL */
                        xBorrowLocal(e, e->valueDepth - 1, home);
                    } else {
                        unsigned dst = pushReg(e) - 1;
                        unsigned src = localIn(e, slot, dst);
                        if (src != dst) emit(e, jaiA64MovX(dst, src));
                    }
                }
            }
            off += 5;
            break;
        }

        case OP_SET_FIELD: {
            uint32_t nameIdx = jaiReadU24(code + off + 1);
            /* Receiver then value, both dropped. The receiver's class comes from its STACK entry, not guessed
             * from the locals: two instance locals of different classes would make any guess a silently wrong field offset. */
            if (e->depth < 2) return false;
            ObjClass *klass = e->stackClass[e->depth - 2];
            int recvLocal = e->stackLocal[e->depth - 2];

            unsigned rv, rr;
            SlotKind kv, kr;
            /* A float already in the FP bank is stored from there (`str d`, not popValue's `fmov x,d` + `str x`)
             * -- captured before the pop, since popping is what clears the bit and renames the index. */
            bool vIsFp = e->depth >= 1 && e->valueDepth >= 1 &&
                         e->stack[e->depth - 1] == SLOT_FLOAT &&
                         (e->fpLive & (1u << (e->valueDepth - 1))) != 0;
            unsigned dv = vIsFp ? fpHeldIn(e, e->valueDepth - 1) : 0u;
            if (vIsFp) {
                if (!popValueRaw(e, &rv, &kv)) return false;
            } else if (!popValue(e, &rv, &kv)) {
                return false;
            }
            if (!popValue(e, &rr, &kr)) return false;
            if (kr != SLOT_INST) return false;
            /* An object goes in as readily as a number. The collector is a plain mark-sweep with no write
             * barrier and nothing moves, so the only question is reachability: the receiver is rooted (it
             * is a live SLOT_INST here), and after the store the value hangs off it, while before the store
             * it was rooted in its own right by emitRootFill. What is refused is a kind with no payload
             * register to store (class/function/native/self) and SLOT_ITER, whose index lives in memory. */
            if (kv != SLOT_INT && kv != SLOT_FLOAT && kv != SLOT_BOOL &&
                kv != SLOT_OBJ && kv != SLOT_LIST && kv != SLOT_INST &&
                kv != SLOT_MAYBE_INST) {
                e->whyNot = "storing a field kind this tier cannot write";
                return false;
            }
            if (nameIdx >= (uint32_t)fn->chunk.constants.count) return false;
            Value nameVal = fn->chunk.constants.data[nameIdx];
            if (!IS_STRING(nameVal)) return false;

            if (klass == NULL) return false;
            const FieldInfo *info = jaiClassFieldInfo(klass, AS_STRING(nameVal));
            if (info == NULL || info->isStatic) return false;

            unsigned base = (unsigned)offsetof(ObjInstance, fields) +
                            (unsigned)info->slot * (unsigned)sizeof(Value);
            /* Not a constant tag any more: a maybe-instance's is null-or-object,
             * read off the payload, which is exactly what emitTagFor does. */
            emitTagFor(e, kv, rv, JIT_SCRATCH_A, JIT_SCRATCH_B);
            emit(e, jaiA64StrW(JIT_SCRATCH_A, rr, base));
            if (vIsFp) emit(e, jaiA64StrD(dv, rr, base + 8));
            else       emit(e, jaiA64StrX(rv, rr, base + 8));
            /* Only a kind a later read can replay EXACTLY is remembered; the rest merely retire what was
             * known of this field (local -1), because the read side would otherwise take SLOT_INST with no
             * class behind it -- a shape every field offset it then resolved would be resolved against. */
            bool replayable = kv == SLOT_INT || kv == SLOT_FLOAT ||
                              kv == SLOT_BOOL || kv == SLOT_OBJ;
            recordFieldStore(e, replayable ? recvLocal : -1, info->slot, kv);
            e->wroteHeap = true;
            off += 6;
            break;
        }

        case OP_RETURN_NULL: {
            /* An OSR form's x0 is the resume bytecode offset (see jaiJitEnterOsr's `*resumeAt = at`), but this
             * return sequence is the function tier's and leaves the VALUE in x0 -- a `return` inside a compiled loop once handed the interpreter an int/pointer as an instruction offset instead, miscompiling `for i in 0..n { if .. { return x } }` (14/20 wrong in released builds, 10/10 under --gc-stress). */
            if (e->osr) {
                e->whyNot = "a return inside an OSR loop";
                return false;
            }
            if ((fn->flags & FN_INIT) == 0) {
                /* A function with nothing to return. No register carries the
                 * answer; the entry point builds a null from the kind alone. */
                if (e->sawReturn && e->returnKind != SLOT_NULL) return false;
                e->sawReturn  = true;
                e->returnKind = SLOT_NULL;
                emit(e, jaiA64MovzX(0, 0, 0));
                emitEpilogue(e, 0);
                off += 1;
                break;
            }
            /* An initializer yields the object it initialised, which is what
             * makes `Point(1, 2)` an expression. */
            if (!localInRange(e, 0) || e->localKind[0] != SLOT_INST) return false;
            e->usesSlot0 = true;
            if (e->sawReturn && e->returnKind != SLOT_INST) return false;
            e->sawReturn  = true;
            e->returnKind = SLOT_INST;
            e->returnShape = e->localShape[0];
            {
                unsigned src = localIn(e, 0, 0);
                if (src != 0) emit(e, jaiA64MovX(0, src));
            }
            emitEpilogue(e, 0);
            off += 1;
            break;
        }

        case OP_POP_RETURN_NULL: {
            /* POP's half: forget a deferred/borrowed entry rather than settle
             * it, same as plain OP_POP -- nothing reads a value being thrown
             * away. */
            unsigned r;
            if (e->depth > 0 && holdsRegister(e->stack[e->depth - 1]) &&
                e->valueDepth > 0) {
                unsigned idx = e->valueDepth - 1;
                e->kPend   &= ~(1u << idx);
                e->xBorrow &= ~(1u << idx);
            }
            if (!popValue(e, &r, NULL)) return false;

            /* RETURN_NULL's half, unchanged. */
            if (e->osr) {
                e->whyNot = "a return inside an OSR loop";
                return false;
            }
            if ((fn->flags & FN_INIT) == 0) {
                if (e->sawReturn && e->returnKind != SLOT_NULL) return false;
                e->sawReturn  = true;
                e->returnKind = SLOT_NULL;
                emit(e, jaiA64MovzX(0, 0, 0));
                emitEpilogue(e, 0);
                off += 1;
                break;
            }
            if (!localInRange(e, 0) || e->localKind[0] != SLOT_INST) return false;
            e->usesSlot0 = true;
            if (e->sawReturn && e->returnKind != SLOT_INST) return false;
            e->sawReturn  = true;
            e->returnKind = SLOT_INST;
            e->returnShape = e->localShape[0];
            {
                unsigned src = localIn(e, 0, 0);
                if (src != 0) emit(e, jaiA64MovX(0, src));
            }
            emitEpilogue(e, 0);
            off += 1;
            break;
        }

        case OP_GET_UPVALUE: {
            unsigned index = code[off + 1];
            if (index >= (unsigned)fn->upvalueCount) return false;
            /* Whose closure. An inlined body's is in the register the call
             * site guarded, not in the caller's own closure register: the
             * caller may have no upvalues at all and still be inlining a body
             * that has them. */
            unsigned creg;
            if (e->inlining) {
                if (e->inlClosureReg < 0) return false;
                creg = (unsigned)e->inlClosureReg;
            } else {
                if (!e->usesUpvalues) return false; /* decided before this pass */
                creg = closureReg(e);
            }

            /* closure->upvalues[index]->location, then the Value there. The
             * upvalue may still be open, pointing into the VM stack, so the
             * location is followed rather than assumed closed. */
            emit(e, jaiA64LdrX(JIT_SCRATCH_A, creg,
                               (unsigned)offsetof(ObjClosure, upvalues)));
            emit(e, jaiA64LdrX(JIT_SCRATCH_A, JIT_SCRATCH_A, index * 8u));
            emit(e, jaiA64LdrX(JIT_SCRATCH_A, JIT_SCRATCH_A,
                               (unsigned)offsetof(ObjUpvalue, location)));
            emit(e, jaiA64LdrW(JIT_SCRATCH_B, JIT_SCRATCH_A, 0));

            /* An upvalue's type is whatever the capture put there, so it is
             * read once here and checked on every entry into the loop. */
            Value seen = NULL_VAL;
            ObjClosure *cl = closure;
            if (index < (unsigned)cl->upvalueCount && cl->upvalues[index] != NULL) {
                seen = *cl->upvalues[index]->location;
            }
            SlotKind kind;
            unsigned tag;
            uint32_t seenShape = 0;
            ObjClass *seenClass = NULL;
            if (IS_INT(seen))        { kind = SLOT_INT;   tag = VAL_INT; }
            else if (IS_FLOAT(seen)) { kind = SLOT_FLOAT; tag = VAL_FLOAT; }
            else if (IS_BOOL(seen))  { kind = SLOT_BOOL;  tag = VAL_BOOL; }
            else if (IS_LIST(seen))  { kind = SLOT_LIST;  tag = VAL_OBJ; }
            else if (rawObjValue(seen)) { kind = SLOT_OBJ; tag = VAL_OBJ; }
            else if (IS_INSTANCE(seen) && AS_INSTANCE(seen)->klass != NULL) {
                /* Same reasoning as every other raw-object read site in this
                 * tier: nothing about an upvalue makes its capture special,
                 * it is read the same way a global or a field is. A closure
                 * over a str/list/dict/instance never compiled before this. */
                kind = SLOT_INST; tag = VAL_OBJ;
                seenClass = AS_INSTANCE(seen)->klass;
                seenShape = seenClass->shapeId;
            } else return false;

            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_B, tag));
            branchOnDeopt(e, JAI_A64_NE);
            if (kind == SLOT_LIST) {
                /* "an object" is not "a list": same contract as every other
                 * SLOT_LIST arm in this tier. JIT_SCRATCH_A holds the
                 * upvalue's location and must survive to the load below. */
                emit(e, jaiA64LdrX(JIT_SCRATCH_D, JIT_SCRATCH_A, 8));
                emit(e, jaiA64LdrW(JIT_SCRATCH_B, JIT_SCRATCH_D,
                                   (unsigned)offsetof(Obj, type)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_B, OBJ_LIST));
                branchOnDeopt(e, JAI_A64_NE);
            } else if (kind == SLOT_INST) {
                emit(e, jaiA64LdrX(JIT_SCRATCH_D, JIT_SCRATCH_A, 8));
                emit(e, jaiA64LdrW(JIT_SCRATCH_B, JIT_SCRATCH_D,
                                   (unsigned)offsetof(Obj, type)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_B, OBJ_INSTANCE));
                branchOnDeopt(e, JAI_A64_NE);
                emit(e, jaiA64LdrX(JIT_SCRATCH_D, JIT_SCRATCH_D,
                                   (unsigned)offsetof(ObjInstance, klass)));
                emit(e, jaiA64LdrW(JIT_SCRATCH_D, JIT_SCRATCH_D,
                                   (unsigned)offsetof(ObjClass, shapeId)));
                emitConst64(e, JIT_SCRATCH_B, (int64_t)seenShape);
                emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_D, JIT_SCRATCH_B));
                branchOnDeopt(e, JAI_A64_NE);
            }
            if (!pushValue3(e, kind, seenShape, seenClass, seen, -1)) return false;
            if (kind == SLOT_BOOL) {
                emit(e, jaiA64LdrByte(pushReg(e) - 1, JIT_SCRATCH_A, 8));
            } else {
                emit(e, jaiA64LdrX(pushReg(e) - 1, JIT_SCRATCH_A, 8));
            }
            off += 2;
            break;
        }

        case OP_GET_FIELD: {
            /* Same as OP_GET_FIELD_LOCAL, but the receiver arrives on the
             * operand stack -- which is what a compound assignment emits,
             * since it pushes the receiver twice. */
            uint32_t nameIdx = jaiReadU24(code + off + 1);
            if (e->depth == 0) return false;
            ObjClass *klass = e->stackClass[e->depth - 1];
            Value seen = e->stackSeen[e->depth - 1];
            int fromLocal = e->stackLocal[e->depth - 1];

            /* `Enum.Variant` for a payload-less variant, folded to the one
             * value that variant will ever have (EnumVariant::unit, made on
             * first mention and shared from then on so that `is` keeps
             * meaning identity). Without it the receiver is a global enum,
             * which globalKind classifies SLOT_OBJ, and the SLOT_INST
             * requirement below declined the whole enclosing function.
             *
             * The guard is the enum's shapeId, not its address: shapeId comes
             * from a monotonic counter precisely so that a freed enum whose
             * address is reused cannot be mistaken for the original, which is
             * what makes baking `unit` as a constant safe. `rcv` holds this
             * iteration's load of the global, so reading through it touches a
             * live object.
             *
             * The methods table is consulted first because enumMember does:
             * a method may shadow a variant name, and folding past that would
             * change what the program means.
             *
             * MEASURED. A loop dispatching on `k == Kind.X` over a token list
             * went 654 ms to 62 ms (10.5x, best of five alternating), and a
             * bare `Color.Red`/`Color.Green` compare loop 208 ms to 31 ms:
             * before this the whole loop was interpreted, and this was the
             * tier's single largest refusal on the self-hosted front end --
             * 189 distinct declining sites against 90 for the next reason,
             * the declined list being the whole of the lexer and the parser.
             *
             * RULED OUT: the front end itself does not get faster. It drops
             * 594 distinct declines to 540 and 469M interpreted instructions
             * to 459M (-2.0%), and 56 parser functions newly compile, but
             * `check` over ten large sources measured 3253 ms against 3253 ms
             * and `fmt --check lib` 3514 against 3471 -- both inside the
             * run-to-run spread. The functions that hold the time
             * (_parse_expression, _scan_punct, _punct_kind, keyword_lookup)
             * clear this refusal only to stop at the next one. Kept for the
             * loops it does unblock and as the prerequisite for that work,
             * not as a win on the compiler.
             *
             * REACH. Only an operand-stack enum receiver, which in practice
             * means a global: `Kind.X` written against a module-level enum.
             * The same mention through a LOCAL holding the enum arrives at
             * OP_GET_FIELD_LOCAL, which has no fold and still declines. That
             * is the next site, and the arm here ports to it unchanged. */
            if (e->stack[e->depth - 1] == SLOT_OBJ &&
                IS_OBJ_TYPE(seen, OBJ_ENUM) &&
                nameIdx < (uint32_t)fn->chunk.constants.count &&
                IS_STRING(fn->chunk.constants.data[nameIdx])) {
                ObjEnum *en = (ObjEnum *)AS_OBJ(seen);
                ObjString *vname = AS_STRING(fn->chunk.constants.data[nameIdx]);
                Value shadow;
                int tag = -1;
                if (!jaiTableGetInterned(&en->methods, vname, &shadow)) {
                    for (uint16_t vi = 0; vi < en->variantCount; vi++) {
                        if (en->variants[vi].name == vname ||
                            jaiStringEquals(en->variants[vi].name, vname)) {
                            tag = (int)vi;
                            break;
                        }
                    }
                }
                if (tag >= 0 && en->variants[tag].arity == 0 &&
                    en->variants[tag].unit != NULL) {
                    ObjEnumVal *unit = en->variants[tag].unit;
                    /* This path guards, so nothing may still be deferred. */
                    settleAll(e);
                    unsigned rcv = valueXReg(e, e->valueDepth - 1);
                    emit(e, jaiA64LdrW(JIT_SCRATCH_A, rcv,
                                       (unsigned)offsetof(Obj, type)));
                    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_ENUM));
                    branchOnDeopt(e, JAI_A64_NE);
                    emit(e, jaiA64LdrW(JIT_SCRATCH_A, rcv,
                                       (unsigned)offsetof(ObjEnum, shapeId)));
                    emitConst64(e, JIT_SCRATCH_B, (int64_t)en->shapeId);
                    emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
                    branchOnDeopt(e, JAI_A64_NE);
                    unsigned droppedEnum;
                    if (!popValue(e, &droppedEnum, NULL)) return false;
                    if (!pushValue3(e, SLOT_OBJ, 0, NULL, OBJ_VAL(unit), -1)) {
                        return false;
                    }
                    e->stackUnit[e->depth - 1] = true;
                    emitConst64(e, pushReg(e) - 1, (int64_t)(uintptr_t)unit);
                    off += 6;
                    break;
                }
            }

            /* `Klass.STATIC_NAME`: a static field read off a class object
             * known at compile time. The receiver came from OP_GET_GLOBAL's
             * `globalClass` arm (the only producer of SLOT_CLASS -- see its
             * comment: "resolved now and pinned by the module version check
             * at entry" and "occupies no register, baked into the call
             * sequence"), so `klass` here is not a guess.
             *
             * What is NOT known at compile time is the static's VALUE: it
             * lives in `klass->statics`, a plain JaiTable exactly like a
             * module's globals table, reassignable at any point after class
             * definition (see staticFieldSlot's comment -- `isLet` is not
             * enforced by the interpreter or the checker for a static, only
             * for an INSTANCE field written through `self`). So this reads it
             * the same way OP_GET_GLOBAL reads a plain module global BY
             * ADDRESS: the JaiEntry* is baked, and two guards (the table
             * hasn't rehashed, the value still has the kind observed at
             * compile time) stand between the load and trusting it. A wrong
             * guess, or a rebind between compile and this call, deopts; nothing
             * here can return a stale or mistyped answer.
             *
             * REACH. `TokenFlags.AFTER_NEWLINE` in Lexer._push
             * (lib/jaithon/compile/lexer.jai) -- 10/10 attempts hit this
             * exact gap with no secondary reason (docs/agents/fix-1.md). */
            if (jitStaticFieldEnabled() && e->stack[e->depth - 1] == SLOT_CLASS) {
                if (klass == NULL) {
                    return subWhy(e, "a static receiver with no class pinned");
                }
                if (nameIdx >= (uint32_t)fn->chunk.constants.count) {
                    return subWhy(e, "the field name is not in the pool");
                }
                Value sNameVal = fn->chunk.constants.data[nameIdx];
                if (!IS_STRING(sNameVal)) {
                    return subWhy(e, "the field name is not a string");
                }
                ObjString *sname = AS_STRING(sNameVal);

                const FieldInfo *sinfo = jaiClassFieldInfo(klass, sname);
                if (sinfo == NULL) {
                    return subWhy(e, "`%s` is not a field of %s", sname->chars,
                                  klass->name ? klass->name->chars : "?");
                }
                if (!sinfo->isStatic) {
                    return subWhy(e, "`%s` is an instance field, not a static",
                                  sname->chars);
                }

                JaiEntry *sslot = staticFieldSlot(e, klass, sname);
                if (sslot == NULL) {
                    return subWhy(e, "`%s` has no static storage, or is a "
                                  "second class's statics in one body",
                                  sname->chars);
                }
                Value sseen = sslot->value;

                SlotKind sk = SLOT_OPAQUE;
                uint32_t sshape = 0;
                ObjClass *sfcls = NULL;
                if (!globalKind(sseen, &sk, &sshape, &sfcls)) {
                    return subWhy(e, "static `%s` is a kind this tier cannot "
                                  "hold", sname->chars);
                }

                /* Named ahead of the guards, same reason as OP_GET_GLOBAL's
                 * BY-ADDRESS arm: they run against the model as it is now. */
                unsigned dst = valueXReg(e, e->valueDepth);
                unsigned stag = sk == SLOT_INT   ? VAL_INT
                              : sk == SLOT_FLOAT ? VAL_FLOAT
                              : sk == SLOT_BOOL  ? VAL_BOOL
                                                 : VAL_OBJ;

                emitStaticsGuard(e);
                emitConst64(e, JIT_SCRATCH_D, (int64_t)(uintptr_t)sslot);
                emit(e, jaiA64LdrW(JIT_SCRATCH_C, JIT_SCRATCH_D,
                                   (unsigned)offsetof(JaiEntry, value)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_C, stag));
                branchOnDeopt(e, JAI_A64_NE);
                emit(e, jaiA64LdrX(JIT_SCRATCH_C, JIT_SCRATCH_D,
                                   (unsigned)offsetof(JaiEntry, value) + 8u));
                if (sk == SLOT_INST || sk == SLOT_LIST) {
                    emit(e, jaiA64LdrByte(JIT_SCRATCH_B, JIT_SCRATCH_C,
                                          (unsigned)offsetof(Obj, type)));
                    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_B,
                                           sk == SLOT_INST ? OBJ_INSTANCE
                                                            : OBJ_LIST));
                    branchOnDeopt(e, JAI_A64_NE);
                }
                if (sk == SLOT_INST) {
                    emit(e, jaiA64LdrX(JIT_SCRATCH_B, JIT_SCRATCH_C,
                                       (unsigned)offsetof(ObjInstance, klass)));
                    emit(e, jaiA64LdrW(JIT_SCRATCH_B, JIT_SCRATCH_B,
                                       (unsigned)offsetof(ObjClass, shapeId)));
                    emitConst64(e, JIT_SCRATCH_A, (int64_t)sshape);
                    emit(e, jaiA64SubsX(31, JIT_SCRATCH_B, JIT_SCRATCH_A));
                    branchOnDeopt(e, JAI_A64_NE);
                }

                /* The receiver held no register (SLOT_CLASS -- see its
                 * comment on the enum): dropping it is the whole of popping
                 * it. popValue would refuse it via holdsRegister. */
                e->depth--;
                if (!pushValue3(e, sk, sshape, sfcls, sseen, -1)) return false;
                emit(e, jaiA64MovX(dst, JIT_SCRATCH_C));
                off += 6;
                break;
            }

            /* And, as at the local-receiver arm, a maybe-instance reads like an
             * instance once known not-null -- which is what `a.b.c` needs, since
             * `.b` came back SLOT_MAYBE_INST and `.c` wants a receiver.
             *
             * This arm and the SLOT_MAYBE_INST field arm below were the two
             * halves of the same hole: the whole chain `a.b.c.d.v` declined its
             * enclosing loop, and it now compiles. Measured on a loop reading
             * one four-deep chain per iteration, alternating old/new binaries
             * and warmed by the clock, best of five each: 0.4356 s -> 0.1107 s,
             * 3.9x. What did NOT move is a self-hosted compile (`check` over
             * parser.jai + emit.jai): 1.4253 s -> 1.4353 s, inside the 2% a
             * base-against-base control shows -- only eleven of the compiler's
             * bodies decline for these two reasons. The enum receiver that stops
             * 110 more of them is the arm directly above, and it did not move
             * the compile either. */
            /* `module.MEMBER`. The receiver is a module the walk has a live
             * sample of, which is every imported module: OP_GET_GLOBAL just
             * read it out of this body's own globals. Reading a constant off
             * one used to decline the whole function. */
            if (moduleFieldOn() && IS_MODULE(seen) &&
                nameIdx < (uint32_t)fn->chunk.constants.count &&
                IS_STRING(fn->chunk.constants.data[nameIdx])) {
                ObjString *mname = AS_STRING(fn->chunk.constants.data[nameIdx]);
                JaiEntry *mslot = moduleMemberSlot(e, AS_MODULE(seen), mname);
                Value held = (mslot != NULL) ? mslot->value : NULL_VAL;
                SlotKind mk = SLOT_OPAQUE;
                unsigned mtag = VAL_OBJ;
                if (mslot != NULL) {
                    if (IS_INT(held))        { mk = SLOT_INT;   mtag = VAL_INT; }
                    else if (IS_FLOAT(held)) { mk = SLOT_FLOAT; mtag = VAL_FLOAT; }
                    else if (IS_BOOL(held))  { mk = SLOT_BOOL;  mtag = VAL_BOOL; }
                    /* Objects are deliberately left out. A module member that
                     * is a closure, class or native is a CALLEE, and the arms
                     * that bake one by value rely on ObjModule::version, which
                     * only retires a form at its next entry -- handing one out
                     * through this path would be a wrong answer rather than a
                     * decline. Scalars are inert and have no such problem. */
                }
                if (mk != SLOT_OPAQUE) {
                    emitModuleGuard(e);
                    unsigned mdrop;
                    if (!popValue(e, &mdrop, NULL)) return false;
                    if (!pushValue3(e, mk, 0, NULL, held, -1)) return false;
                    unsigned mrd = pushReg(e) - 1;
                    emitConst64(e, JIT_SCRATCH_D, (int64_t)(uintptr_t)mslot);
                    /* The tag is re-checked every read: a module global is
                     * assignable, so `math.PI = 3` must deoptimise rather than
                     * hand back the old kind's payload. */
                    emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_D,
                                       (unsigned)offsetof(JaiEntry, value)));
                    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, mtag));
                    branchOnDeopt(e, JAI_A64_NE);
                    if (mk == SLOT_BOOL) {
                        emit(e, jaiA64LdrByte(mrd, JIT_SCRATCH_D,
                                              (unsigned)offsetof(JaiEntry, value) + 8u));
                    } else {
                        emit(e, jaiA64LdrX(mrd, JIT_SCRATCH_D,
                                           (unsigned)offsetof(JaiEntry, value) + 8u));
                    }
                    off += 6;
                    break;
                }
            }
            if (e->stack[e->depth - 1] != SLOT_INST &&
                e->stack[e->depth - 1] != SLOT_MAYBE_INST) {
                return subWhy(e, "a receiver of kind %s, not an instance",
                              slotKindName(e->stack[e->depth - 1]));
            }
            if (klass == NULL) {
                return subWhy(e, "an instance receiver with no class pinned");
            }
            if (e->stack[e->depth - 1] == SLOT_MAYBE_INST) {
                emit(e, jaiA64SubsXImm(31, valueBankReg(e, e->valueDepth - 1), 0));
                branchOnDeopt(e, JAI_A64_EQ);
            }
            if (nameIdx >= (uint32_t)fn->chunk.constants.count) {
                return subWhy(e, "the field name is not in the pool");
            }
            Value nameVal = fn->chunk.constants.data[nameIdx];
            if (!IS_STRING(nameVal)) {
                return subWhy(e, "the field name is not a string");
            }

            const FieldInfo *info = jaiClassFieldInfo(klass, AS_STRING(nameVal));
            if (info == NULL) {
                return subWhy(e, "`%s` is not a field of %s",
                              AS_STRING(nameVal)->chars,
                              klass->name ? klass->name->chars : "that class");
            }
            if (info->isStatic) {
                return subWhy(e, "`%s` is a static, not an instance field",
                              AS_STRING(nameVal)->chars);
            }
            if (!IS_INSTANCE(seen)) {
                /* No sample to classify the field from. The commonest cause
                 * is a receiver OP_INVOKE itself predicted: `klass` came from
                 * observedReturnKind's shape (or a compiled callee's
                 * jitReturnShape) by way of jaiClassForShape, which pins the
                 * class exactly, but pushValue leaves `seen` at NULL_VAL
                 * because there is no actual instance behind a prediction,
                 * only a shape id -- `self._peek().kind`, `self._chunk().depth`
                 * and every other zero-arg-accessor-then-field-read chain in
                 * the self-hosted front end is this shape, and it declined
                 * outright before this arm existed. See declaredScalarFieldKind
                 * for why only a scalar can be predicted from here. */
                SlotKind dkind;
                unsigned dtag;
                if (!jitDeclaredFieldKindEnabled() ||
                    !declaredScalarFieldKind(info->typeId, &dkind, &dtag)) {
                    /* The declared kind is NAMED, because which one it is
                     * decides the work: a scalar wants a row in
                     * declaredScalarFieldKind, an ANY wants nothing at all,
                     * and the census could not tell them apart. */
                    return subWhy(e,
                        "no live receiver to read `%s` off, and its declared "
                        "kind (%u) is not one this predicts",
                        AS_STRING(nameVal)->chars, (unsigned)info->typeId);
                }
                unsigned dbase = (unsigned)offsetof(ObjInstance, fields) +
                                 (unsigned)info->slot * (unsigned)sizeof(Value);
                unsigned drr = valueBankReg(e, e->valueDepth - 1);
                /* Guard BEFORE the receiver comes off the model, as the
                 * sampled path below does: a deopt here resumes at this
                 * instruction, and the interpreter's stack still has the
                 * receiver on it. */
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, drr, dbase));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, dtag));
                branchOnDeopt(e, JAI_A64_NE);
                Value dprobe = NULL_VAL;
                if (dkind == SLOT_LIST || info->typeId == FIELD_KIND_STR) {
                    /* VAL_OBJ said "a heap object" and no more. Prove the
                     * actual type before the entry claims to be one, or the
                     * next arm reads ObjList's count out of a string's header
                     * -- the hole that segfaulted the VM through the
                     * dict-index arm. */
                    unsigned want = dkind == SLOT_LIST ? (unsigned)OBJ_LIST
                                                       : (unsigned)OBJ_STRING;
                    emit(e, jaiA64LdrX(JIT_SCRATCH_B, drr, dbase + 8));
                    emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_B,
                                       (unsigned)offsetof(Obj, type)));
                    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, want));
                    branchOnDeopt(e, JAI_A64_NE);
                    if (want == (unsigned)OBJ_STRING) {
                        /* A probe, not an observation: the arms below ask
                         * `stringOperand`, which reads the SAMPLE rather than
                         * the kind, so a string entry without one is a string
                         * nothing can do anything with. The guard just emitted
                         * is what makes the probe honest -- it chooses an arm
                         * and never removes a check. Same device the invoke
                         * arm uses for a receiver it knows only the type of. */
                        ObjString *empty = jaiStringIntern("", 0);
                        if (empty == NULL) return false;
                        dprobe = OBJ_VAL((Obj *)empty);
                    }
                }
                unsigned dpopped;
                SlotKind dkr;
                if (!popValue(e, &dpopped, &dkr)) return false;
                if (!pushValue3(e, dkind, 0, NULL, dprobe, -1)) return false;
                if (!IS_NULL(dprobe)) {
                    e->stackObjType[e->depth - 1] =
                        (uint8_t)(OBJ_STRING + 1);
                }
                if (dkind == SLOT_FLOAT &&
                    fpWorthLoading(e, code, off + 6, stop)) {
                    unsigned idx = e->valueDepth - 1;
                    emit(e, jaiA64LdrD(fpRegAt(e, idx), drr, dbase + 8));
                    fpClaim(e, idx);
                } else if (dkind == SLOT_BOOL) {
                    /* One byte, for the reason given at the local-receiver
                     * arm: BOOL_VAL writes only the union's bool member. */
                    emit(e, jaiA64LdrByte(pushReg(e) - 1, drr, dbase + 8));
                } else {
                    emit(e, jaiA64LdrX(pushReg(e) - 1, drr, dbase + 8));
                }
                off += 6;
                break;
            }
            ObjInstance *inst = AS_INSTANCE(seen);
            if (info->slot >= inst->fieldCount) {
                return subWhy(e, "the sampled instance has fewer fields than "
                                 "its class declares");
            }
            Value fieldVal = inst->fields[info->slot];

            SlotKind kind;
            unsigned tag;
            ObjClass *fcls = NULL;
            if (IS_INT(fieldVal))        { kind = SLOT_INT;   tag = VAL_INT; }
            else if (IS_FLOAT(fieldVal)) { kind = SLOT_FLOAT; tag = VAL_FLOAT; }
            /* A nullable instance field, same guess and same guards as the
             * local-receiver arm: a leaf's null has no class to read, so the
             * receiver's own class stands in and the class guard below deopts
             * if that was wrong. Without this arm `rawObjValue` (which excludes
             * OBJ_INSTANCE) let an instance-valued field fall off the end of
             * the chain and decline the whole function. */
            else if (IS_INSTANCE(fieldVal) || IS_NULL(fieldVal)) {
                kind = SLOT_MAYBE_INST;
                tag  = VAL_OBJ;
                fcls = IS_INSTANCE(fieldVal) ? AS_INSTANCE(fieldVal)->klass
                                             : klass;
                if (fcls == NULL) {
                    return subWhy(e, "a nullable field with no class to guard");
                }
            }
            /* As in OP_GET_FIELD_LOCAL: an object-typed field is held raw
             * rather than declining the enclosing function, and a list earns
             * the stronger kind at the price of an OBJ_LIST check. */
            else if (IS_LIST(fieldVal))     { kind = SLOT_LIST; tag = VAL_OBJ; }
            /* A bool field. Most classes carry one, and without this arm the
             * whole enclosing function declined -- `_check_live` in jaicv's
             * cascade is a method whose only field read is a bool, and it cost
             * 107 ms against 11.8 ms for the byte-identical method over an int
             * field. The load below is a `ldrb`, because BOOL_VAL writes only
             * the union's one-byte member. */
            else if (IS_BOOL(fieldVal)) { kind = SLOT_BOOL; tag = VAL_BOOL; }
            else if (rawObjValue(fieldVal)) { kind = SLOT_OBJ;  tag = VAL_OBJ; }
            else {
                return subWhy(e, "a field holding %s, which has no arm",
                              jaiTypeNameStatic(fieldVal));
            }

            unsigned fbase = (unsigned)offsetof(ObjInstance, fields) +
                             (unsigned)info->slot * (unsigned)sizeof(Value);
            unsigned rr = valueBankReg(e, e->valueDepth - 1);
            SlotKind already = knownFieldKind(e, fromLocal, info->slot);
            if (kind == SLOT_MAYBE_INST) {
                /* The same three guards the local arm emits, branch-free by the
                 * same trick: a null payload redirects the loads at the receiver,
                 * which the entry check above has already proved is a live
                 * instance, so nothing ever dereferences zero. All of it stands
                 * before the receiver comes off the model, since every deopt
                 * here resumes at this instruction with it still on the stack. */
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, rr, fbase));       /* tag */
                emit(e, jaiA64LdrX(JIT_SCRATCH_D, rr, fbase + 8));   /* value */
                emit(e, jaiA64MovzX(JIT_SCRATCH_B, 0, 0));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, VAL_OBJ));
                emit(e, jaiA64CselX(JIT_SCRATCH_D, JIT_SCRATCH_D,
                                    JIT_SCRATCH_B, JAI_A64_EQ));
                emit(e, jaiA64CselX(JIT_SCRATCH_B, JIT_SCRATCH_B,
                                    JIT_SCRATCH_A, JAI_A64_EQ));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_B, 0));
                branchOnDeopt(e, JAI_A64_NE);      /* neither object nor null */

                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_D, 0));
                emit(e, jaiA64CselX(JIT_SCRATCH_B, rr, JIT_SCRATCH_D,
                                    JAI_A64_EQ));
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_B,
                                   (unsigned)offsetof(Obj, type)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_INSTANCE));
                branchOnDeopt(e, JAI_A64_NE);

                emit(e, jaiA64LdrX(JIT_SCRATCH_A, JIT_SCRATCH_B,
                                   (unsigned)offsetof(ObjInstance, klass)));
                emitConst64(e, JIT_SCRATCH_B, (int64_t)(uintptr_t)fcls);
                if (fcls != klass) {
                    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_D, 0));
                    emit(e, jaiA64CselX(JIT_SCRATCH_A, JIT_SCRATCH_B,
                                        JIT_SCRATCH_A, JAI_A64_EQ));
                }
                emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
                branchOnDeopt(e, JAI_A64_NE);
            } else if (already != SLOT_SELF) {
                kind = already;
            } else {
                /* Guard BEFORE the receiver comes off the model: a deopt here
                 * resumes at this instruction, and the interpreter's stack
                 * still has the receiver on it. */
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, rr, fbase));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, tag));
                branchOnDeopt(e, JAI_A64_NE);
                /* And that it is a list, not merely an object -- same reason as
                 * in OP_GET_FIELD_LOCAL, and likewise while the receiver is
                 * still on the model, since this guard resumes here too. */
                if (kind == SLOT_LIST) {
                    emit(e, jaiA64LdrX(JIT_SCRATCH_D, rr, fbase + 8));
                    emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_D,
                                       (unsigned)offsetof(Obj, type)));
                    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_LIST));
                    branchOnDeopt(e, JAI_A64_NE);
                }
            }
            unsigned popped;
            SlotKind kr;
            if (!popValue(e, &popped, &kr)) return false;
            if (kind == SLOT_MAYBE_INST) {
                if (!pushValue3(e, kind, fcls->shapeId, fcls,
                                IS_INSTANCE(fieldVal) ? fieldVal : NULL_VAL,
                                -1)) {
                    return false;
                }
                /* Already in hand: the guards above loaded the payload (or a
                 * zero) into D, so there is nothing left to read. */
                emit(e, jaiA64MovX(pushReg(e) - 1, JIT_SCRATCH_D));
            } else {
                if (!pushValue3(e, kind, 0, NULL,
                                (kind == SLOT_OBJ && rawObjValue(fieldVal)) ||
                                        (kind == SLOT_LIST && IS_LIST(fieldVal))
                                    ? fieldVal : NULL_VAL,
                                -1)) {
                    return false;
                }
                if (kind == SLOT_FLOAT && fpWorthLoading(e, code, off + 6, stop)) {
                    unsigned idx = e->valueDepth - 1;
                    emit(e, jaiA64LdrD(fpRegAt(e, idx), rr, fbase + 8));
                    fpClaim(e, idx);
                } else if (kind == SLOT_BOOL) {
                    /* One byte, for the reason given at the local-receiver arm. */
                    emit(e, jaiA64LdrByte(pushReg(e) - 1, rr, fbase + 8));
                } else {
                    emit(e, jaiA64LdrX(pushReg(e) - 1, rr, fbase + 8));
                }
            }
            off += 6;
            break;
        }

        case OP_BUILD_LIST: {
            unsigned n = jaiReadU16(code + off + 1);
            if (n > JIT_MAX_ARGS_OUT) return false;
            if (!e->callsOut) return false;
            if (e->depth < n) return false;
            Value elemSeen = NULL_VAL;
            if (!buildListExemplar(e, e->depth - n, n, &elemSeen)) {
                elemSeen = NULL_VAL;
            }
            if (!emitDescriptor(e, NULL_VAL, e->depth - n, n,
                                (void *)&jitBuildList)) {
                return false;
            }
            for (unsigned i = 0; i < n; i++) {
                unsigned r;
                if (!popValue(e, &r, NULL)) return false;
            }
            if (!pushValue(e, SLOT_LIST, 0, NULL)) return false;
            /* Computed BEFORE the pops above, since it reads the entries they
             * remove. See buildListExemplar for why the list itself cannot
             * answer this. */
            e->stackElem[e->depth - 1] = elemSeen;
            /* The same exemplar as a KIND, which survives a bind into a local
             * where the Value does not -- there is nowhere to root a Value per
             * local, and a byte needs no rooting.
             *
             * A prediction, not a fact, unlike the OP_ELEM_KIND route: an
             * undeclared literal gets boxed storage, so a later append may put
             * anything in it. That is safe on the same terms as stackElem
             * itself -- a boxed element is tag-checked at every read, so a
             * changed kind deoptimises. `min_area_rect` builds
             * `[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]` with no declared type and then
             * subscripts it, which is the case that needed this. */
            e->stackElemDecl[e->depth - 1] =
                IS_INT(elemSeen)   ? (uint8_t)(FIELD_KIND_INT   + 1)
              : IS_FLOAT(elemSeen) ? (uint8_t)(FIELD_KIND_FLOAT + 1)
              : IS_BOOL(elemSeen)  ? (uint8_t)(FIELD_KIND_BOOL  + 1)
                                   : (uint8_t)0;
            emit(e, jaiA64LdrX(pushReg(e) - 1, 31,
                               e->descOffset +
                                   (unsigned)offsetof(JitCallDesc, result) + 8));
            e->wroteHeap = true;
            off += 3;
            break;
        }

        case OP_IN:
        case OP_NOT_IN: {
            /* `x in c`. No arm existed, so a membership test ENDED THE WALK:
             * `if k in seen` is the shape of every dedup loop in the corpus and
             * everything after it ran interpreted.
             *
             * The containment itself is not made faster -- it is the same
             * jaiContainsOp the interpreter runs, called out to. What the arm
             * buys is the body around it, which is the whole point of a row
             * over a call that is cheap next to its loop.
             *
             * `not in` is the same call with the sense flipped, in its own
             * entry point rather than an argc flag -- a wider descriptor would
             * name a stack entry past the operands. */
            if (!jitMembership() || !e->callsOut || e->depth < 2) {
                goto unarmedOpcode;
            }
            if (!emitDescriptor(e, NULL_VAL, e->depth - 2, 2,
                                code[off] == OP_IN ? (void *)&jitContains
                                                   : (void *)&jitNotContains)) {
                return false;
            }
            for (unsigned i = 0; i < 2; i++) {
                unsigned r;
                if (!popValue(e, &r, NULL)) return false;
            }
            if (!pushValue(e, SLOT_BOOL, 0, NULL)) return false;
            emit(e, jaiA64LdrByte(pushReg(e) - 1, 31,
                                  e->descOffset +
                                      (unsigned)offsetof(JitCallDesc, result) +
                                      8));
            /* Containment is not pure: a class can define __contains__, so the
             * call may run Jaithon code that writes. Leaving this unset marked
             * every body holding an `in` jitFuncNoWrite, which lets a direct
             * caller finish the callee by RE-RUNNING it from the start on a
             * bail -- and re-running the writes with it. */
            e->wroteHeap = true;
            off += 1;
            break;
        }

        case OP_BUILD_DICT:
        case OP_BUILD_SET: {
            /* The remaining container literals, on the OP_BUILD_LIST template.
             *
             * Both were top of the partial-walk census over the self-hosted
             * parser: `_node(kind, span, fields: dict = {})` builds a dict for
             * every AST node, and the walk stopped there sixteen times in one
             * file. */
            bool isDict = code[off] == OP_BUILD_DICT;
            unsigned n = jaiReadU16(code + off + 1);
            unsigned operands = isDict ? n * 2u : n;
            if (!jitTuple() || operands > JIT_MAX_ARGS_OUT || !e->callsOut ||
                e->depth < operands) {
                goto unarmedOpcode;
            }
            if (!emitDescriptor(e, NULL_VAL, e->depth - operands, operands,
                                isDict ? (void *)&jitBuildDict
                                       : (void *)&jitBuildSet)) {
                return false;
            }
            for (unsigned i = 0; i < operands; i++) {
                unsigned r;
                if (!popValue(e, &r, NULL)) return false;
            }
            if (!pushValue(e, SLOT_OBJ, 0, NULL)) return false;
            /* SLOT_OBJ does not say which container this is, and OP_ELEM_KIND
             * comes straight after a literal and has to know. */
            e->stackObjType[e->depth - 1] =
                (uint8_t)((isDict ? OBJ_DICT : OBJ_SET) + 1);
            emit(e, jaiA64LdrX(pushReg(e) - 1, 31,
                               e->descOffset +
                                   (unsigned)offsetof(JitCallDesc, result) + 8));
            e->wroteHeap = true;
            off += 3;
            break;
        }

        case OP_BUILD_TUPLE: {
            /* Same shape as OP_BUILD_LIST above, and simpler: jaiTupleNew
             * copies the operands itself and cannot throw. No exemplar is
             * kept -- a tuple has no element arm to feed, so the entry is a
             * plain SLOT_OBJ.
             *
             * Worth an arm only because there was none: a tuple build ENDED
             * THE WALK, and `let p = (x, y)` in a loop body is common enough
             * that the whole body after it ran interpreted. */
            unsigned n = jaiReadU16(code + off + 1);
            if (!jitTuple() || n > JIT_MAX_ARGS_OUT || !e->callsOut ||
                e->depth < n) {
                goto unarmedOpcode;
            }
            if (!emitDescriptor(e, NULL_VAL, e->depth - n, n,
                                (void *)&jitBuildTuple)) {
                return false;
            }
            for (unsigned i = 0; i < n; i++) {
                unsigned r;
                if (!popValue(e, &r, NULL)) return false;
            }
            if (!pushValue(e, SLOT_OBJ, 0, NULL)) return false;
            e->stackObjType[e->depth - 1] = (uint8_t)(OBJ_TUPLE + 1);
            emit(e, jaiA64LdrX(pushReg(e) - 1, 31,
                               e->descOffset +
                                   (unsigned)offsetof(JitCallDesc, result) + 8));
            e->wroteHeap = true;
            off += 3;
            break;
        }

        case OP_NOT: {
            /* `not x`. The interpreter REQUIREs a bool here, and SLOT_BOOL's
             * contract is "0 or 1 in a register", so the flip is an xor with
             * one and there is nothing to guard. Anything else is not a
             * narrowing this tier declines to do -- it is a program the
             * interpreter would throw on, and it reaches the throw by the
             * unarmed path.
             *
             * No arm existed, so `not` ENDED THE WALK the way OP_NEG did:
             * `if not a` in a loop was 23,252,579 interpreted instructions. */
            if (!jitNegate() || e->depth < 1 ||
                e->stack[e->depth - 1] != SLOT_BOOL) {
                goto unarmedOpcode;
            }
            {
                unsigned nr = pushReg(e) - 1;
                emitConst64(e, JIT_SCRATCH_A, 1);
                emit(e, jaiA64EorX(nr, nr, JIT_SCRATCH_A));
            }
            off += 1;
            break;
        }

        case OP_BNOT: {
            /* `~x` is `x ^ -1`, and the model has already proved the int. */
            if (!jitNegate() || e->depth < 1 ||
                e->stack[e->depth - 1] != SLOT_INT) {
                goto unarmedOpcode;
            }
            {
                unsigned nr = pushReg(e) - 1;
                emitConst64(e, JIT_SCRATCH_A, -1);
                emit(e, jaiA64EorX(nr, nr, JIT_SCRATCH_A));
            }
            off += 1;
            break;
        }

        case OP_POS: {
            /* Unary `+` on a number is the identity -- the interpreter checks
             * the type and does nothing else. The model has already proved it,
             * so this emits nothing at all; the point is only that the walk
             * does not stop here. */
            if (!jitNegate() || e->depth < 1 ||
                (e->stack[e->depth - 1] != SLOT_INT &&
                 e->stack[e->depth - 1] != SLOT_FLOAT)) {
                goto unarmedOpcode;
            }
            off += 1;
            break;
        }

        case OP_NEG: {
            /* `-x`. There was no arm at all, for either kind: OP_NEG appeared
             * only in inlinableBody's whitelist, so in an ordinary body it fell
             * to `default` and ENDED THE WALK -- everything after a negation
             * ran interpreted. `-(i & 255)` in a loop was 28,394,773
             * interpreted instructions and the float form 18,034,610.
             *
             * It hid because emitUnarmedDeopt is silent by design (the point of
             * it is to interpret from here rather than decline the body), so
             * nothing was ever printed. What surfaced it was a DIFFERENT
             * message: `is_inf`'s `x == INF or x == -INF` records a forward
             * branch to its OP_RETURN before reaching the negation, and once
             * the walk stops that branch has nowhere to land. The fixup pass
             * now names the opcode that ended the walk, which is how a
             * confusing "branch to offset 25" became "OP_NEG at 23". Every
             * guarded function in std.math goes through `_require_finite` and
             * so through `is_inf`. */
            if (!jitNegate()) {
                return subWhy(e, "the negate arm is switched off");
            }
            if (e->depth < 1) return subWhy(e, "nothing to negate");
            SlotKind nk = e->stack[e->depth - 1];
            if (nk == SLOT_INT) {
                /* `subs` off zero both negates and reports the one input that
                 * cannot be: INT64_MIN overflows, and the interpreter raises
                 * the OverflowError on re-entry.
                 *
                 * Into a SCRATCH first, and moved only once the guard has
                 * passed -- the same discipline the `abs` arm states, and the
                 * reason is this instruction resumes at its own START. Writing
                 * the operand's register before the guard hands the deopt
                 * record a value that has ALREADY been negated, and the
                 * interpreter negates it again: `neg_int(5)` returned 5.
                 * Only JAITHON_JIT_DEOPT_STRESS=1 finds it, because in
                 * ordinary running the guard fires only on INT64_MIN, which is
                 * its own negation and so hides the mistake. */
                unsigned nr = pushReg(e) - 1;
                emit(e, jaiA64SubsX(JIT_SCRATCH_A, JAI_A64_XZR, nr));
                branchOnDeoptInstStart(e, JAI_A64_VS);
                emit(e, jaiA64MovX(nr, JIT_SCRATCH_A));
                off += 1;
                break;
            }
            if (nk == SLOT_FLOAT) {
                /* Straight in the bank. IEEE negation is the sign bit and
                 * cannot fail, so there is no guard and nothing to raise --
                 * -0.0 and NaN both come out of `fneg` the way the interpreter
                 * produces them. */
                unsigned nd = fpOperand(e, e->valueDepth - 1);
                unsigned drop;
                if (!popValueRaw(e, &drop, NULL)) return false;
                if (!pushValue(e, SLOT_FLOAT, 0, NULL)) return false;
                unsigned nidx = e->valueDepth - 1;
                emit(e, jaiA64FnegD(fpRegAt(e, nidx), nd));
                fpClaim(e, nidx);
                off += 1;
                break;
            }
            return subWhy(e, "negating a %s", slotKindName(nk));
        }

        case OP_POW: {
            /* Only `** 0.5` (square root): C's pow(x,0.5) is sqrt(x) for x >= +0, differing only at -0.0 (pow
             * gives +0.0, sqrt gives -0.0) -- so a negative sign bit sends this back to the interpreter. */
            if (e->depth < 2) return false;
            if (e->stack[e->depth - 1] != SLOT_FLOAT) return false;
            if (e->stack[e->depth - 2] != SLOT_FLOAT) return false;
            Value expv = e->stackSeen[e->depth - 1];
            if (!IS_FLOAT(expv) || AS_FLOAT(expv) != 0.5) {
                e->whyNot = "an exponent other than 0.5";
                return false;
            }
            /* Wholly in the FP bank: the sign bit is the only thing needing an integer register, and an `fmov`
             * out of d costs one instruction where routing the operand through X cost four (two for an exponent constant nothing reads, two more around the fsqrt). */
            unsigned ia = e->valueDepth - 2;
            unsigned da = fpOperand(e, ia);
            emit(e, jaiA64FmovXD(JIT_SCRATCH_A, da));
            emit(e, jaiA64LsrX(JIT_SCRATCH_A, JIT_SCRATCH_A, 63));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, 0));
            branchOnDeopt(e, JAI_A64_NE);

            unsigned dp1, dp2;
            if (!popValueRaw(e, &dp1, NULL)) return false;
            if (!popValueRaw(e, &dp2, NULL)) return false;
            if (!pushValue(e, SLOT_FLOAT, 0, NULL)) return false;
            {
                unsigned idx = e->valueDepth - 1;
                emit(e, jaiA64FsqrtD(fpRegAt(e, idx), da));
                fpClaim(e, idx);
            }
            off += 1;
            break;
        }

        case OP_MOD: {
            /* Floor remainder, the same rule as the fused form but with the
             * divisor in a register, so zero and -1 both have to be checked. */
            if (e->depth < 2) return false;
            if (e->stack[e->depth - 1] != SLOT_INT) return false;
            if (e->stack[e->depth - 2] != SLOT_INT) return false;
            unsigned ry = pushReg(e) - 1, rx = valueXReg(e, e->valueDepth - 2);

            /* A literal power-of-two modulus decides the whole thing: floor remainder by 2^s is exactly the low
             * s bits, which two's complement already holds -- exact even for negative dividends, unlike the truncating `msub` path below (whose correction exists to fix exactly that). */
            int64_t kmod = 0;
            bool kmodKnown = literalIntOperand(fn, prevOff, off, &kmod) &&
                             kmod != 0 && kmod != -1;
            unsigned mshift;
            if (kmodKnown && powerOfTwoShift(kmod, &mshift)) {
                unsigned q1, q2;
                if (!popValue(e, &q1, NULL)) return false;
                if (!popValue(e, &q2, NULL)) return false;
                if (!pushValue(e, SLOT_INT, 0, NULL)) return false;
                unsigned rd = pushReg(e) - 1;
                if (mshift == 0) emitConst64(e, rd, 0);   /* x %% 1 is 0 */
                else emit(e, jaiA64AndXOnes(rd, rx, mshift));
                off += 1;
                break;
            }

            if (!kmodKnown) {
                emit(e, jaiA64SubsXImm(31, ry, 0));
                branchOnDeopt(e, JAI_A64_EQ);
                emit(e, jaiA64AddXImm(JIT_SCRATCH_A, ry, 1));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, 0));
                branchOnDeopt(e, JAI_A64_EQ);
            }

            /* Remainder lands in `rx`, where the result belongs: two entries come off, one goes on, so the
             * surviving (lower) entry keeps its register -- removes a trailing `mov`. Safe because every guard this arm emits is above this line: nothing below can deopt and find the dividend gone. */
            emit(e, jaiA64SdivX(JIT_SCRATCH_B, rx, ry));
            emit(e, jaiA64MsubX(rx, JIT_SCRATCH_B, ry, rx));
            emitFloorFixup(e, rx, ry, kmodKnown, kmod,
                           jaiA64AddX(rx, rx, ry));

            unsigned dm1, dm2;
            if (!popValue(e, &dm1, NULL)) return false;
            if (!popValue(e, &dm2, NULL)) return false;
            if (!pushValue(e, SLOT_INT, 0, NULL)) return false;
            off += 1;
            break;
        }

        case OP_MOD_INT_CONST: {
            /* `<int k>; MOD` fused: k is known non-zero (fusion requires it); -1 goes back to the interpreter so
             * INT64_MIN %% -1 stays its problem. */
            int16_t imm = jaiReadI16(code + off + 1);
            if (imm == 0 || imm == -1) return false;
            if (e->depth == 0 || e->stack[e->depth - 1] != SLOT_INT) return false;
            unsigned rx = pushReg(e) - 1;

            /* A power of two is the low bits and nothing else -- see OP_MOD. */
            unsigned kshift;
            if (powerOfTwoShift(imm, &kshift)) {
                if (kshift == 0) emitConst64(e, rx, 0);
                else emit(e, jaiA64AndXOnes(rx, rx, kshift));
                off += 3;
                break;
            }

            emitConst64(e, JIT_SCRATCH_A, imm);
            emit(e, jaiA64SdivX(JIT_SCRATCH_B, rx, JIT_SCRATCH_A));
            emit(e, jaiA64MsubX(rx, JIT_SCRATCH_B, JIT_SCRATCH_A, rx));
            emitFloorFixup(e, rx, JIT_SCRATCH_A, true, imm,
                           jaiA64AddX(rx, rx, JIT_SCRATCH_A));
            off += 3;
            break;
        }

        case OP_FLOORDIV: {
            unsigned rb, ra;
            SlotKind kb, ka;
            if (e->depth < 2) return false;
            ka = e->stack[e->depth - 2]; kb = e->stack[e->depth - 1];
            if (ka != SLOT_INT || kb != SLOT_INT) return false;

            /* A literal power-of-two divisor decides the whole thing: floor(x / 2^s) is exactly `asr x, #s` for
             * every int64 x (negative included), since asr already rounds toward minus infinity -- what the correction below exists to reproduce for the general case. */
            int64_t kdiv = 0;
            bool kdivKnown = literalIntOperand(fn, prevOff, off, &kdiv) &&
                             kdiv != 0 && kdiv != -1;
            unsigned dshift;
            if (kdivKnown && powerOfTwoShift(kdiv, &dshift)) {
                /* The divisor is spelt by the shift field, so drop its deferral rather than settle it -- the
                 * same move OP_ADD makes for an imm12 -- popValueRaw drops it. The dividend comes back from popValue, not from pushReg, because a borrowed entry lives in the local's register and not in its own. */
                unsigned p1, p2;
                if (!popValueRaw(e, &p1, NULL)) return false;
                if (!popValue(e, &p2, NULL)) return false;
                if (!pushValue(e, SLOT_INT, 0, NULL)) return false;
                emit(e, jaiA64AsrX(pushReg(e) - 1, p2, dshift));
                off += 1;
                break;
            }
            /* Every path below reads both operands out of their own registers,
             * and two of them guard. */
            settleAll(e);
            rb = pushReg(e) - 1; ra = valueXReg(e, e->valueDepth - 2);

            /* Zero and -1 both decline: the interpreter reports division-by-zero, and INT64_MIN / -1 is the one
             * quotient that doesn't fit -- both rare enough that declining -1 outright costs nothing. A literal divisor has already answered both. */
            if (!kdivKnown) {
                emit(e, jaiA64SubsXImm(31, rb, 0));
                branchOnDeopt(e, JAI_A64_EQ);
                emit(e, jaiA64AddXImm(JIT_SCRATCH_A, rb, 1));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, 0));
                branchOnDeopt(e, JAI_A64_EQ);
            }

            unsigned d1, d2;
            if (!popValue(e, &d1, NULL)) return false;
            if (!popValue(e, &d2, NULL)) return false;
            if (!pushValue(e, SLOT_INT, 0, NULL)) return false;
            unsigned rq = pushReg(e) - 1;

            /* Quotient lands in the register the dividend was in (the push reuses it), so the dividend is copied
             * out first -- without that, msub read a value sdiv had already overwritten and 7 // 2 came out 2. */
            emit(e, jaiA64MovX(JIT_SCRATCH_C, ra));

            /* Truncating quotient, then one down when the remainder is nonzero
             * and its sign differs from the divisor's -- which is what makes
             * this floor division rather than C's. */
            emit(e, jaiA64SdivX(rq, JIT_SCRATCH_C, rb));
            emit(e, jaiA64MsubX(JIT_SCRATCH_A, rq, rb, JIT_SCRATCH_C));
            /* A literal divisor makes the correction a single sign test on the remainder (see emitFloorFixup).
             * JIT_SCRATCH_B is free again here -- the quotient is in rq, not in it. */
            emitFloorFixup(e, JIT_SCRATCH_A, rb, kdivKnown, kdiv,
                           jaiA64SubXImm(rq, rq, 1));
            off += 1;
            break;
        }

        /* A list comprehension's append. It is the same two stores `push`
         * makes, but it was the one list write with no arm at all, so every
         * comprehension in the language -- `[0 for _i in 0..n]`, the way this
         * codebase preallocates -- left its loop running interpreted. */
        case OP_LIST_APPEND: {
            unsigned back = jaiReadU16(code + off + 1);
            if (e->depth < back + 1u) {
                e->whyNot = "an append reaching past the model"; return false;
            }
            unsigned lidx = e->depth - 1u - back;
            if (e->stack[lidx] != SLOT_LIST) {
                e->whyNot = "an append to an entry the tier does not know is a list";
                return false;
            }
            if (!emitListStore(e, e->stack[e->depth - 1],
                               valueXReg(e, valueIndexOf(e, lidx)),
                               pushReg(e) - 1, e->stackLocal[lidx])) {
                return false;
            }
            /* The value is consumed; the target stays where it was. */
            unsigned dropAppend;
            if (!popValue(e, &dropAppend, NULL)) return false;
            off += 3;
            break;
        }

        case OP_INVOKE: {
            uint32_t nameIdx = jaiReadU24(code + off + 1);
            unsigned argc    = code[off + 4];
            /* The argc cap is a fixed limit rather than a property of this
             * call, and it was indistinguishable in the census from every
             * other reason an invoke declines -- which is how it went a long
             * time without anyone knowing how many sites it costs.
             *
             * NOT softened, despite OP_GET_GLOBAL's own refusal being exactly
             * this shape -- tried, and it does not buy what it looks like it
             * should. `span`/`fill` (packages/jaicv/imgproc/drawing.jai) each
             * call `__prim__.fill_span`/`fill_convex` at 10 and 11 arguments,
             * past this cap; before globalNamespace existed that call never
             * reached here (OP_GET_GLOBAL "__prim__" stopped the walk first,
             * softly, and the body still compiled everything ahead of it).
             * Once __prim__ resolves, softening THIS refusal the same way
             * only trades it for an EARLIER one: the measuring pass that
             * computes `body.maxValue` now walks all the way through the ten
             * argument pushes before reaching (and skipping) this check,
             * which is what overflows JIT_MAX_SAVED and fails "the operand
             * stack alone exceeds the registers" instead -- a harder, whole-
             * function-declining wall this file has no per-instruction answer
             * for (raising JIT_MAX_SAVED or teaching the allocator to spill
             * more of an argument list is a different, larger piece of work).
             * Measured: softening this arm moved `span` from 12.71% to
             * 13.60% of jaicv's interpreted work (BENCH_LEVEL=easy), not
             * down -- so it is left hard, on purpose, rather than landed for
             * a win it does not actually produce. */
            if (argc > JIT_MAX_ARGS_OUT - 1) {
                return subWhy(e, "%u arguments, past the cap of %d",
                              argc, JIT_MAX_ARGS_OUT - 1);
            }
            if (!e->callsOut) return subWhy(e, "this region may not call out");
            if (e->depth < argc + 1) {
                return subWhy(e, "the model is %u deep for %u arguments",
                              e->depth, argc);
            }

            unsigned ridx = e->depth - argc - 1;
            SlotKind rk = e->stack[ridx];

            /* A maybe-instance receiver is an instance once it is known not to
             * be null, and the whole arm below already resolves against the
             * class the entry carries -- which a maybe-instance carries too.
             * So one compare buys it: prove it here, then let it read as
             * SLOT_INST for the rest of the instruction.
             *
             * Order matters. The guard's record is taken while the entry is
             * still MAYBE_INST, because that is what the interpreter has on
             * its stack if the guard fires; the narrowing happens only after,
             * on the path where the proof holds. Both kinds hold a bare
             * pointer in a register, so nothing about the allocation moves.
             *
             * This is what `var at = head` / `while at is not null` /
             * `at = at.follow()` needs -- the loop shape every list and tree
             * walk has. Without it the OSR retry that widens `at` to
             * MAYBE_INST only moved the refusal from the assignment to the
             * call. */
            if (rk == SLOT_MAYBE_INST && e->stackClass[ridx] != NULL) {
                settleAll(e);
                emit(e, jaiA64SubsXImm(31, valueXReg(e, ridx - (e->depth - e->valueDepth)), 0));
                branchOnDeopt(e, JAI_A64_EQ);
                e->stack[ridx] = SLOT_INST;
                rk = SLOT_INST;
            }

            if (rk == SLOT_LIST && argc == 1 &&
                nameIdx < (uint32_t)fn->chunk.constants.count &&
                IS_STRING(fn->chunk.constants.data[nameIdx]) &&
                strcmp(AS_STRING(fn->chunk.constants.data[nameIdx])->chars,
                       "push") == 0) {
                /* Appending to a list is a bounds check and two stores -- a descriptor+native round trip costs far
                 * more than the work itself (list_ops spent all its time on the call). A full list goes out to the `grow` stubs' realloc helper and comes straight back; see there for why this used to be a deopt and what that cost. */
                if (!emitListStore(e, e->stack[e->depth - 1],
                                   valueXReg(e, e->valueDepth - 2),
                                   pushReg(e) - 1, e->stackLocal[ridx])) {
                    return false;
                }
                /* push returns the list, which is the receiver entry already
                 * sitting under the argument. */
                unsigned drop;
                if (!popValue(e, &drop, NULL)) return false;
                off += 7;
                break;
            }

            /* `Klass.static_method(x)`. NOT a field read plus a call, which is
             * what an earlier attempt at this assumed and why it measured
             * nothing: the compiler emits OP_GET_GLOBAL "Box" and then an
             * OP_INVOKE whose receiver is the class, so an arm in OP_GET_FIELD
             * never sees it. See emitClassCall. */
            if (rk == SLOT_CLASS && jitClassCalls()) {
                ObjClass *scls = e->stackClass[ridx];
                if (scls == NULL) {
                    return subWhy(e, "a class receiver the model did not pin");
                }
                if (nameIdx >= (uint32_t)fn->chunk.constants.count) {
                    return subWhy(e, "the member name is not in the pool");
                }
                Value sname = fn->chunk.constants.data[nameIdx];
                if (!IS_STRING(sname)) {
                    return subWhy(e, "the member name is not a string");
                }
                Value smember;
                if (!jaiTableGetInterned(&scls->statics, AS_STRING(sname),
                                         &smember)) {
                    /* The interpreter's own order: statics first, then
                     * `methods`. An instance method IS reachable this way and
                     * must not be admitted -- resolveInvokeTarget hands the
                     * bare closure to invokeCallable with the CLASS sitting in
                     * slot 0, so `self` is the class and the body raises the
                     * moment it touches a field ("class 'Box' has no member
                     * 'v'", measured). Compiling that faithfully would take a
                     * receiver this arm exists to drop, and the reward would be
                     * a faster way to reach the same exception. */
                    Value smeth;
                    if (jaiTableGetInterned(&scls->methods, AS_STRING(sname),
                                            &smeth)) {
                        return subWhy(e, "`%s.%s` is an instance method reached "
                                         "through the class",
                                      scls->name != NULL ? scls->name->chars
                                                         : "?",
                                      AS_STRING(sname)->chars);
                    }
                    return subWhy(e, "`%s` is not a static of %s",
                                  AS_STRING(sname)->chars,
                                  scls->name != NULL ? scls->name->chars : "?");
                }
                /* Visibility is the interpreter's methodPermitted, and it is
                 * NOT a compile-time property in general: accessPermitted reads
                 * the running frame's owner class. A non-public static DOES
                 * reach here -- the checker rejects `Box.hidden(i)` from
                 * outside with E0701, but a `static fn` with no `pub` called
                 * from a method of its own class type-checks and runs. This arm
                 * resolves once and never calls methodPermitted again, so
                 * admitting one would be skipping a check the interpreter makes
                 * on every call. Restricted means non-public, which is the
                 * whole condition.
                 *
                 * Reached, not hypothetical: `Hidden.walk` in
                 * tests/lang/test_jit_class_call.jai stops here, and the
                 * targeted suites hit it nine times. */
                MethodInfo smi;
                if (jaiClassRestrictedMethod(scls, AS_STRING(sname), &smi)) {
                    return subWhy(e, "`%s.%s` is not public",
                                  scls->name != NULL ? scls->name->chars : "?",
                                  AS_STRING(sname)->chars);
                }
                /* A closure straight out of the table, never a bound method and
                 * never a native: this arm drops the receiver, and both of the
                 * others want one. See emitClassCall's second paragraph. */
                if (!IS_CLOSURE(smember)) {
                    return subWhy(e, "`%s.%s` is a %s, not a function",
                                  scls->name != NULL ? scls->name->chars : "?",
                                  AS_STRING(sname)->chars,
                                  jaiTypeNameStatic(smember));
                }
                /* The ENTRY, not just the value: the guard reads the
                 * binding back out of this slot on every call. */
                JaiEntry *sslot =
                    jaiTableFindEntryInterned(&scls->statics, AS_STRING(sname));
                if (sslot == NULL) {
                    return subWhy(e, "a static with no table entry");
                }
                if (!emitClassCall(e, scls, sslot, smember, ridx, argc,
                                   (uint32_t)(off + 7))) {
                    return false;
                }
                if (getenv("JAI_JIT_WHY")) {
                    fprintf(stderr, "[jit] static call %s.%s at %d\n",
                            scls->name != NULL ? scls->name->chars : "?",
                            AS_STRING(sname)->chars, off);
                }
                off += 7;
                break;
            }

            if (rk == SLOT_INST) {
                /* A method whose whole body is one arithmetic expression over receiver+arguments is worth inlining:
                 * the call costs more than the expression. Vec2.dot is four field reads, two multiplies and an add, reached through a descriptor, a helper and a compiled entry. */
                if (inlineMethod(e, closure, nameIdx, argc, off)) {
                    off += 7;
                    break;
                }

                /* A method on an instance. The class is fixed here, so the
                 * method is resolved now; what it returns is not knowable, so
                 * the tag that comes back is checked and a surprise deopts to
                 * the instruction AFTER the call -- the call has happened and
                 * must not happen twice. */
                ObjClass *rcls = e->stackClass[ridx];
                if (nameIdx >= (uint32_t)fn->chunk.constants.count) return false;
                Value mname = fn->chunk.constants.data[nameIdx];
                if (!IS_STRING(mname)) return false;

                /* An instance whose class the model could not pin -- what
                 * `for op in ops` produces when the list holds more than one
                 * implementation of a trait. There is no method to resolve
                 * here, so the name goes out instead and jitInvokeByName
                 * resolves per call. Declining instead was worth 2.4x on the
                 * two-class form of tests/bench/poly_dispatch, entirely
                 * because the loop around it then ran interpreted. */
                if (rcls == NULL) {
                    SlotKind rkind;
                    unsigned rtag;
                    const DiscardKind udisc = discardedAfter(code, off + 7, count);
                    const bool discarded = udisc != DISCARD_NO;
                    const uint16_t invokeCache = jaiReadU16(code + off + 5);
                    const bool havePrediction =
                        siteInvokeResultKind(&fn->chunk, invokeCache,
                                             &rkind, &rtag);
                    if (!havePrediction && !discarded) {
                        e->whyNot = "an unpinned receiver's result kind";
                        return false;
                    }

                    /* The site's own inline cache first: the ONE class it
                     * saw first gets a compare and a direct branch into its
                     * compiled entry, and only a receiver that isn't that
                     * class pays for the descriptor below -- which, absent
                     * this, is every receiver. See emitInvokePic1. */
                    int picToEnd = -1;
                    if (jitPicEnabled()) {
                        if (emitInvokePic1(e, fn, ridx, argc, (uint32_t)off,
                                          (uint32_t)(off + 7),
                                          (int)invokeCache, havePrediction,
                                          havePrediction ? rkind : SLOT_NULL,
                                          &picToEnd)) {
                            vm.jitPicAdmits++;
                        } else {
                            if (e->failed) return false;
                            vm.jitPicRefusals++;
                        }
                    }

                    if (!emitDescriptor(e, mname, ridx, argc + 1,
                                        (void *)&jitInvokeByName)) {
                        return false;
                    }
                    for (unsigned i = 0; i <= argc; i++) {
                        unsigned r;
                        if (!popValue(e, &r, NULL)) return false;
                    }
                    e->wroteHeap = true;
                    if (!havePrediction) {
                        if (udisc == DISCARD_POP_RETURN) {
                            if (!emitFusedReturnNull(e, fn)) return false;
                            off += 8;
                            afterUncond = true;
                            continue;
                        }
                        off += 8;          /* the OP_POP this consumed */
                        break;
                    }
                    if (!pushValue(e, rkind, 0, NULL)) return false;
                    unsigned rat = e->descOffset +
                                   (unsigned)offsetof(JitCallDesc, result);
                    emit(e, jaiA64LdrW(JIT_SCRATCH_A, 31, rat));
                    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, rtag));
                    branchOnDeoptAt(e, JAI_A64_NE, (uint32_t)(off + 7), true);
                    if (rkind == SLOT_BOOL) {
                        emit(e, jaiA64LdrByte(pushReg(e) - 1, 31, rat + 8));
                    } else {
                        emit(e, jaiA64LdrX(pushReg(e) - 1, 31, rat + 8));
                    }
                    /* The merge: the one way branched here rather than round
                     * the descriptor, arriving with the same entry in the
                     * same register this path just loaded -- the whole of
                     * what the two paths had to agree about for the arm to
                     * be emitted at all. */
                    for (unsigned pw = 0; pw < e->picExitCount; pw++) {
                        const int at = e->picExits[pw];
                        if (at >= 0 && at < (int)e->count &&
                            e->count <= JIT_MAX_INSTS) {
                            e->code[at] =
                                jaiA64B((int32_t)((int)e->count - at));
                        }
                    }
                    e->picExitCount = 0;
                    off += 7;
                    break;
                }
                Value method;
                if (!jaiClassFindMethod(rcls, AS_STRING(mname), &method)) {
                    return false;
                }
                if (!IS_CLOSURE(method)) return false;
                ObjFunction *mfn = AS_CLOSURE(method)->fn;
                /* The callee's own compiled form states its return kind exactly; without one the
                 * interpreter's record of what it has been returning stands in. That case is not exotic --
                 * two methods that call each other can never take turns being the first to compile, so
                 * neither ever has a jitFunc to ask, and json_parse's whole parser is that shape. Either
                 * way it is only a prediction: the tag guard below is what makes it sound. */
                SlotKind rkind = SLOT_NULL;
                uint32_t rshape = 0;
                ObjClass *rrcls = NULL;
                bool haveKind;
                if (mfn->jitFunc != NULL) {
                    rkind = (SlotKind)mfn->jitReturnKind;
                    rshape = mfn->jitReturnShape;
                    haveKind = true;
                } else {
                    haveKind = observedReturnKind(mfn, &rkind, &rshape, NULL);
                }
                if (haveKind &&
                    (rkind == SLOT_INST || rkind == SLOT_MAYBE_INST) &&
                    (rshape == 0 || !jaiClassForShape(rshape, &rrcls) ||
                     rrcls == NULL)) {
                    haveKind = false;
                    rrcls = NULL;
                }
                if (haveKind && rkind != SLOT_INT && rkind != SLOT_FLOAT &&
                    rkind != SLOT_BOOL && rkind != SLOT_INST &&
                    rkind != SLOT_MAYBE_INST && rkind != SLOT_LIST &&
                    rkind != SLOT_OBJ && rkind != SLOT_NULL) {
                    haveKind = false;
                }
                /* A result the very next instruction pops needs no kind at all -- which is every `-> void`
                 * method called as a statement, and the reason a parser's `self.skip()` used to decline the
                 * function around it. Same relaxation the builtin arm below already makes. */
                DiscardKind mdisc = discardedAfter(code, off + 7, count);
                bool mdiscarded = mdisc != DISCARD_NO;
                if (!haveKind && !mdiscarded) {
                    e->whyNot = "callee's return kind not usable";
                    return false;
                }

                /* Straight to the method's compiled entry since the receiver's class is fixed here (SLOT_INST
                 * carries a shape the caller already guarded) -- the "inline cache" is the model itself, free at run time. Skips the whole jitInvokeMethod -> jaiCallMethodWithReceiver -> invokeCallable -> callClosure -> jaiJitEnterFunc -> jitArgIn chain of C glue between two compiled bodies, which was a large fraction of object_dispatch. Falls back rather than declines, for the same reason a global call does: the descriptor path speaks a wider language. */
                if (mfn->jitFunc != NULL) {
                    const char *saved = e->whyNot;
                    if (emitDirectCall(e, fn, mfn, method, -1, ridx, argc,
                                       (uint32_t)off, (uint32_t)(off + 7),
                                       true)) {
                        if (getenv("JAI_JIT_WHY")) {
                            fprintf(stderr, "[jit] direct method %s.%s at %d\n",
                                    rcls->name ? rcls->name->chars : "?",
                                    AS_STRING(mname)->chars, off);
                        }
                        off += 7;
                        break;
                    }
                    if (e->failed) return false;   /* it had started emitting */
                    e->whyNot = saved;
                }

                if (!emitDescriptor(e, method, ridx, argc + 1,
                                    (void *)&jitInvokeMethod)) {
                    return false;
                }
                for (unsigned i = 0; i <= argc; i++) {
                    unsigned r;
                    if (!popValue(e, &r, NULL)) return false;
                }
                if (!haveKind) {
                    /* Nothing observes the result, so nothing has to be
                     * predicted or guarded about it. */
                    e->wroteHeap = true;
                    if (mdisc == DISCARD_POP_RETURN) {
                        if (!emitFusedReturnNull(e, fn)) return false;
                        off += 8;
                        afterUncond = true;
                        continue;
                    }
                    off += 8;          /* the OP_POP this consumed */
                    break;
                }
                if (!pushValue(e, rkind, rshape, rrcls)) return false;

                unsigned rat = e->descOffset +
                               (unsigned)offsetof(JitCallDesc, result);
                if (rkind == SLOT_MAYBE_INST) {
                    emitMaybeInstResult(e, pushReg(e) - 1, rat, rshape,
                                        (uint32_t)(off + 7));
                    off += 7;
                    break;
                }
                unsigned wantTag = rkind == SLOT_INT   ? VAL_INT
                                 : rkind == SLOT_FLOAT ? VAL_FLOAT
                                 : rkind == SLOT_BOOL  ? VAL_BOOL
                                 : rkind == SLOT_NULL  ? VAL_NULL
                                                       : VAL_OBJ;
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, 31, rat));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, wantTag));
                branchOnDeoptAt(e, JAI_A64_NE, (uint32_t)(off + 7), true);
                /* Bool result is one byte, not eight (see the SLOT_OBJ arm below for why) -- latent here since this
                 * was written: needs a bool-returning method emitDirectCall declines, which nothing in the suite exercises. */
                if (rkind == SLOT_BOOL) {
                    emit(e, jaiA64LdrByte(pushReg(e) - 1, 31, rat + 8));
                } else if (rkind == SLOT_NULL) {
                    /* A null Value's payload word is not written by NULL_VAL, so
                     * there is nothing to load; SLOT_NULL is a defined zero. */
                    emit(e, jaiA64MovzX(pushReg(e) - 1, 0, 0));
                } else {
                    emit(e, jaiA64LdrX(pushReg(e) - 1, 31, rat + 8));
                }
                if (rkind == SLOT_INST) {
                    /* "an object" is not "an object of this class", and a
                     * method entered with another specialisation runs
                     * interpreted and may return either. The object TYPE goes
                     * first: VAL_OBJ is every heap object, so a method that
                     * returned a string here would have `klass` read one word
                     * past its header and the shape loaded through whatever was
                     * there -- a segfault, reachable from ordinary code. */
                    emit(e, jaiA64LdrW(JIT_SCRATCH_A, pushReg(e) - 1,
                                       (unsigned)offsetof(Obj, type)));
                    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_INSTANCE));
                    branchOnDeoptAt(e, JAI_A64_NE, (uint32_t)(off + 7), true);
                    emit(e, jaiA64LdrX(JIT_SCRATCH_A, pushReg(e) - 1,
                                       (unsigned)offsetof(ObjInstance, klass)));
                    emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_A,
                                       (unsigned)offsetof(ObjClass, shapeId)));
                    emitConst64(e, JIT_SCRATCH_B, (int64_t)rshape);
                    emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
                    branchOnDeoptAt(e, JAI_A64_NE, (uint32_t)(off + 7), true);
                } else if (rkind == SLOT_LIST) {
                    /* Same hazard as SLOT_INST above: a method entered with
                     * another specialisation runs interpreted and may
                     * return any type, so VAL_OBJ alone does not prove the
                     * payload is a list before ObjList's fields get read
                     * off it unguarded downstream. */
                    emit(e, jaiA64LdrW(JIT_SCRATCH_A, pushReg(e) - 1,
                                       (unsigned)offsetof(Obj, type)));
                    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_LIST));
                    branchOnDeoptAt(e, JAI_A64_NE, (uint32_t)(off + 7), true);
                }

                e->wroteHeap = true;
                off += 7;
                break;
            }

            if (rk == SLOT_OBJ) {
                /* Built-in method on a receiver typed only as "some object" (dict/string/set/tuple). Three things:
                 * WHICH METHOD -- from the observed receiver, like the SLOT_LIST arm below (a builtin is a function of receiver-type + name). THAT IT'S STILL THAT TYPE -- SLOT_OBJ pins nothing (`for x in [d, "s"]` mixes types), so the object type is guarded before anything is consumed; a miss resumes with receiver+args untouched. WHAT COMES BACK -- predicted via InlineCache::resultKind (no per-call-site feedback existed before), and the tag guard after the call is what makes the prediction sound, deopting to the instruction AFTER the call since it already happened. */
                Value oseen = e->stackSeen[ridx];
                bool  oProbe = false;
                /* Read before the refusal below can fire, so the refusal can
                 * name the method. Without the name it said only "an invoke on
                 * an object with nothing to look at", which is the top
                 * STATE refusal by attributed cost -- 9.2% of the interpreted
                 * work in one file sat behind it with nothing to act on. */
                Value oNameEarly =
                    nameIdx < (uint32_t)fn->chunk.constants.count
                        ? fn->chunk.constants.data[nameIdx] : NULL_VAL;
                const char *oNameChars =
                    IS_STRING(oNameEarly) ? AS_STRING(oNameEarly)->chars : "?";
                if (!IS_OBJ(oseen)) {
                    /* No sample, but the model may still know the TYPE -- an
                     * f-string's result, or an earlier invoke's predicted from
                     * its site's feedback. A probe of that type answers the
                     * method lookup just as well, exactly as the SLOT_LIST arm
                     * below builds an empty list for the same reason. Only
                     * strings are probed: they are the types this tier
                     * currently learns without a Value, and a probe has to be
                     * something cheap and permanent to make.
                     *
                     * The object type is guarded at run time below either way,
                     * so the probe chooses a guard and never deletes one. */
                    if (e->stackObjType[ridx] == (uint8_t)(OBJ_STRING + 1)) {
                        ObjString *empty = jaiStringIntern("", 0);
                        if (empty == NULL) return false;
                        oseen = OBJ_VAL((Obj *)empty);
                        oProbe = true;
                    } else if (listProbeOn() &&
                               e->stackObjType[ridx] == (uint8_t)(OBJ_LIST + 1)) {
                        /* A predicted list. The type comes from the callee's
                         * own observed return record, which is what
                         * `let w = window_of(...)` gives -- `window_of` is
                         * declared `-> list[int]`, compiles, and has always
                         * had OBJ_LIST on record; the caller simply threw it
                         * away (see observedReturnKind).
                         *
                         * An empty list rather than an interned constant: the
                         * probe only has to answer WHICH builtin `.len` is,
                         * and it is rooted across the lookup exactly as the
                         * string probe is, with only the resolved native kept.
                         * The receiver's object type is guarded at run time
                         * below either way, so this chooses a guard and never
                         * deletes one. */
                        ObjList *probe = jaiListNew(0);
                        if (probe == NULL) return false;
                        oseen = OBJ_VAL((Obj *)probe);
                        oProbe = true;
                    } else {
                        /* Naming the method is what priced this: the census
                         * showed `.len()` and nothing else, and the attribution
                         * instrument put **14.27% of lexer.jai's interpreted
                         * work** in eleven functions behind it.
                         *
                         * THE CHAIN, re-derived 2026-08-30 by actually clearing
                         * the links rather than probing them (see the memory
                         * note on why a probed link can be an artefact). For
                         * `_fuse_at`, which is `let w = window_of(...)` then
                         * `w.len()` and `w[1]`, `w[2]`, `w[3]`:
                         *
                         *   1. `.len()` on a receiver with no sample. Cleared:
                         *      the callee's OBSERVED return record already said
                         *      OBJ_LIST and observedReturnKind was dropping it
                         *      on the floor. An empty-list probe answers the
                         *      lookup, as the string probe already did.
                         *   2. `w[2]`: the container has kind object, not list.
                         *      Cleared: promote an observed list return to
                         *      SLOT_LIST, which emitCallOutResult was already
                         *      emitting the OBJ_LIST guard for.
                         *   3. `w[2]`: no live list to read an element kind
                         *      off. NOT cleared. A call result has no declared
                         *      element type on record anywhere -- `window_of`
                         *      is declared `-> list[int]` and nothing carries
                         *      the `int`. That needs a new record, either the
                         *      declared element kind (a SEEDED emit.jai change)
                         *      or an observed one on ObjFunction.
                         *
                         * MEASURED, links 1 and 2 together, one binary through
                         * JAITHON_JIT_LIST_PROBE and JAITHON_JIT_RET_LIST, four
                         * paired runs of `check --no-cache parser.jai`:
                         *   off 58,069,566  58,188,736  58,038,914  58,503,867
                         *   on  58,929,750  58,075,894  57,560,941  57,301,824
                         * Flat. Three of four favour it and the mean moves
                         * 0.4%, which is inside the spread -- because link 3
                         * still declines the body.
                         *
                         * KEPT anyway, on the same reasoning as the enum fold
                         * above: the refusal is now accurate rather than
                         * misleading, and links 1 and 2 are prerequisites for
                         * link 3 being worth anything. Do not re-measure these
                         * two alone and expect a number. */
                        /* Say WHICH type was predicted, if any. "no known
                         * type" hid the difference between a receiver the
                         * model knows nothing about and one it knows is a
                         * list -- and those want opposite fixes. */
                        if (e->stackObjType[ridx] != 0) {
                            return subWhy(e,
                                "`.%s()` on a predicted %s with no sample to "
                                "probe", oNameChars,
                                jaiObjTypeName((ObjType)(e->stackObjType[ridx] - 1)));
                        }
                        return subWhy(e, "`.%s()` on an object with no sample "
                                      "and no known type", oNameChars);
                    }
                }
                if (nameIdx >= (uint32_t)fn->chunk.constants.count) return false;
                Value oname = fn->chunk.constants.data[nameIdx];
                if (!IS_STRING(oname)) return false;
                Value obound;
                /* Rooted across the lookup: resolving allocates the bound
                 * wrapper, and a probe interned a moment ago is otherwise
                 * unreachable. Only the native is kept, and that outlives it. */
                if (oProbe) jaiGCPushRoot(oseen);
                bool oFound = jaiBuiltinMethod(oseen, AS_STRING(oname), &obound);
                if (oProbe) jaiGCPopRoot();
                if (!oFound) {
                    /* Names the pair, since which builtin is missing is the
                     * whole question and the bare reason only says that one
                     * was. */
                    return subWhy(e, "`%s.%s` is not a builtin of the observed "
                                     "receiver's type",
                                  jaiTypeNameStatic(oseen),
                                  AS_STRING(oname)->chars);
                }
                Value onative = IS_BOUND(obound) ? AS_BOUND(obound)->method
                                                 : obound;
                /* `__prim__.f64_sqrt(x)` and its kin: a member reached
                 * through a MODULE receiver that IS already a native --
                 * jaiModuleMethod hands one back unwrapped (see
                 * emitModuleNativeCall's comment for why), so `onative` here
                 * skips the `!IS_NATIVE` block below entirely and would
                 * otherwise fall into the generic native-with-a-receiver path
                 * two arms down, which passes the MODULE as args[0] -- wrong,
                 * since the interpreter drops it (resolveInvokeTarget leaves
                 * `isMethod` false for a module receiver). Caught here, before
                 * that path, for the same reason the closure arm below is
                 * caught before it: on the same `IS_MODULE(oseen) &&
                 * !IS_BOUND(obound)` condition, one kill switch for both
                 * halves (jitModuleNativeCalls, OP_GET_GLOBAL's
                 * globalNamespace side). */
                if (IS_MODULE(oseen) && !IS_BOUND(obound) &&
                    IS_NATIVE(onative) && jitModuleNativeCalls()) {
                    /* Refuse SOFTLY when nothing was emitted and the model is
                     * untouched -- this is a NEW arm, so a hard refusal here
                     * is worse than having no arm at all.
                     *
                     * Concretely: `span` ends in `__prim__.fill_span(...)`,
                     * which is not on the result-kind whitelist. While
                     * `__prim__` had no arm the walk took the unarmed path,
                     * skipped the whole call as one block, and span's prologue
                     * compiled. Declining hard here instead threw the body
                     * away and DOUBLED span's interpreted work, 5,821,736 to
                     * 10,867,344 -- a regression caused entirely by arming a
                     * neighbouring opcode.
                     *
                     * Guarded on emitting nothing rather than on which refusal
                     * fired, so a later refusal that has already written code
                     * or moved the stack still declines hard, where resuming
                     * would be unsound. */
                    unsigned mnCount = e->count;
                    unsigned mnDepth = e->depth;
                    if (!emitModuleNativeCall(e, AS_MODULE(oseen), onative,
                                              ridx, argc,
                                              (uint32_t)(off + 7))) {
                        if (!e->failed && e->count == mnCount &&
                            e->depth == mnDepth) {
                            goto unarmedOpcode;
                        }
                        return false;
                    }
                    if (getenv("JAI_JIT_WHY")) {
                        fprintf(stderr, "[jit] module native call %s.%s at %d\n",
                                AS_MODULE(oseen)->name != NULL
                                    ? AS_MODULE(oseen)->name->chars : "?",
                                AS_STRING(oname)->chars, off);
                    }
                    off += 7;
                    break;
                }
                if (!IS_NATIVE(onative)) {
                    /* A module member written in Jaithon lands here: `math.sqrt`
                     * is a CLOSURE wrapping __prim__.f64_sqrt, not a builtin,
                     * and so is every other one. See emitModuleCall -- it is a
                     * global call through another module, not a method call.
                     *
                     * NOT a bound one. `onative` above has already thrown the
                     * receiver away, which is right for the builtin arm below
                     * (callNativeAt reads args[0], and the receiver is already
                     * in the callee slot) and wrong here, because emitModuleCall
                     * DROPS the receiver as well: nothing is left to be `self`.
                     * A module member whose value is a bound method --
                     * `pub let handler = obj.method` -- then called the unbound
                     * closure with the first argument where `self` belongs, and
                     * `self.base` raised "'fn' object has no attribute 'base'"
                     * on code the interpreter answers correctly. Declining is
                     * enough: the shape is rare and the descriptor would have to
                     * carry `obound`, not `onative`, to do better. */
                    if (IS_MODULE(oseen) && !IS_BOUND(obound) &&
                        IS_CLOSURE(onative) && jitModuleCalls()) {
                        if (!emitModuleCall(e, AS_MODULE(oseen), onative, ridx,
                                            argc, (uint32_t)(off + 7))) {
                            return false;
                        }
                        if (getenv("JAI_JIT_WHY")) {
                            fprintf(stderr, "[jit] module call %s.%s at %d\n",
                                    AS_MODULE(oseen)->name != NULL
                                        ? AS_MODULE(oseen)->name->chars : "?",
                                    AS_STRING(oname)->chars, off);
                        }
                        off += 7;
                        break;
                    }
                    /* Named, because bare this was invisible: the census only
                     * ever said "OP_INVOKE", and p27_float_math cost a night to
                     * trace to `math.sqrt` being an ordinary function. */
                    return subWhy(e, "%s.%s is a %s, not a builtin",
                                  jaiTypeNameStatic(oseen),
                                  AS_STRING(oname)->chars,
                                  jaiTypeNameStatic(onative));
                }

                /* A builtin that only reads a field of its receiver needs
                 * neither the feedback nor the call: the type guard below is
                 * already the whole precondition, and the table states what
                 * comes back. See jit_field_read.h. */
                const JaiJitFieldRead *ofr = jaiJitFieldReadFor(
                    OBJ_TYPE(oseen), AS_STRING(oname)->chars,
                    AS_STRING(oname)->length, argc);

                DiscardKind odisc = discardedAfter(code, off + 7, count);
                bool odiscarded = odisc != DISCARD_NO;
                SlotKind orkind = SLOT_INT;
                unsigned owantTag = VAL_INT;
                uint8_t  orObjType = 0;
                if (ofr == NULL && !odiscarded) {
                    uint8_t fb = jaiInvokeResultFeedback(
                        &fn->chunk, jaiReadU16(code + off + 5), oseen);
                    if (!feedbackSlotKind(fb, &orkind, &owantTag, &orObjType)) {
                        /* Which of the three it is decides what to do about
                         * it: nothing recorded means the window never opened,
                         * mixed means the site really does return two things,
                         * and a named type means the tier has no slot for it. */
                        return subWhy(e, "`%s.%s` has no usable result kind (%s)",
                                      jaiTypeNameStatic(oseen),
                                      AS_STRING(oname)->chars,
                                      jaiFeedbackName(fb));
                    }
                }

                emit(e, jaiA64LdrW(JIT_SCRATCH_A, pushReg(e) - argc - 1,
                                   (unsigned)offsetof(Obj, type)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A,
                                       (unsigned)OBJ_TYPE(oseen)));
                branchOnDeopt(e, JAI_A64_NE);

                if (ofr != NULL) {
                    if (!emitFieldRead(e, ofr, onative, ridx, argc,
                                       (uint32_t)(off + 7))) {
                        return false;
                    }
                    off += 7;   /* a discarded result is the next OP_POP's */
                    break;
                }

                if (!emitDescriptor(e, onative, ridx, argc + 1,
                                    (void *)&jitInvokeNative)) {
                    return false;
                }
                for (unsigned i = 0; i <= argc; i++) {
                    unsigned r;
                    if (!popValue(e, &r, NULL)) return false;
                }
                e->wroteHeap = true;
                if (odiscarded) {
                    if (odisc == DISCARD_POP_RETURN) {
                        if (!emitFusedReturnNull(e, fn)) return false;
                        off += 8;
                        afterUncond = true;
                        continue;
                    }
                    off += 8;          /* the OP_POP this consumed */
                    break;
                }
                if (!pushValue(e, orkind, 0, NULL)) return false;
                /* The feedback named the object type, and nothing else will:
                 * the result does not exist until run time, so there is no
                 * sample. `s.lower().len()` chains two invokes and the second
                 * declined the loop for want of exactly this. */
                if (orkind == SLOT_OBJ) {
                    e->stackObjType[e->depth - 1] = orObjType;
                }
                unsigned orat = e->descOffset +
                                (unsigned)offsetof(JitCallDesc, result);
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, 31, orat));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, owantTag));
                branchOnDeoptAt(e, JAI_A64_NE, (uint32_t)(off + 7), true);
                /* One byte for a bool: BOOL_VAL writes only the union's `boolean` member, so an 8-byte load would
                 * pull in whatever garbage was above it in that Value's slot, and every SLOT_BOOL consumer does `cbnz` on the whole word. Cost a day: the (self-hosted) front end's own `flags.get(name, false)` came back true from garbage bits, misreporting a non-variadic parameter as `list[fn(T) -> bool]`. */
                if (orkind == SLOT_BOOL) {
                    emit(e, jaiA64LdrByte(pushReg(e) - 1, 31, orat + 8));
                } else {
                    emit(e, jaiA64LdrX(pushReg(e) - 1, 31, orat + 8));
                }
                off += 7;
                break;
            }

            if (rk != SLOT_LIST) {
                /* The last arm. Naming the kind is what turns a bare
                 * "OP_INVOKE" in the census into something actionable: every
                 * receiver kind the arms above do not take lands here, and a
                 * class is the biggest of them -- `Klass.static_method(x)` is
                 * an invoke on a SLOT_CLASS receiver, not the field read plus
                 * call it looks like, and docs/probes/p33_static_call.jai runs
                 * 72,434,801 interpreted instructions on account of it. */
                return subWhy(e, "a receiver of kind %s", slotKindName(rk));
            }
            if (nameIdx >= (uint32_t)fn->chunk.constants.count) return false;
            Value nameVal = fn->chunk.constants.data[nameIdx];
            if (!IS_STRING(nameVal)) return false;

            /* Which method a name means depends on the receiver's type, not what it holds -- so for a list this
             * body built with no sample to look at, an empty probe list answers just as well. Rooted across the lookup since resolving allocates the bound wrapper; only the native is kept, and that outlives it. */
            Value probe = e->stackSeen[ridx];
            bool madeProbe = false;
            if (!IS_LIST(probe)) {
                ObjList *tmp = jaiListNew(0);
                probe = OBJ_VAL(tmp);
                jaiGCPushRoot(probe);
                madeProbe = true;
            }
            Value bound;
            bool found = jaiBuiltinMethod(probe, AS_STRING(nameVal), &bound);
            if (madeProbe) jaiGCPopRoot();
            if (!found) return false;
            Value nativeVal = IS_BOUND(bound) ? AS_BOUND(bound)->method : bound;
            if (!IS_NATIVE(nativeVal)) return false;

            /* What comes back: a field-reading builtin says so itself (`len`
             * is the list's count); anything whose result is dropped on the
             * next instruction needs no kind at all.
             *
             * Nothing else -- and the stated reason for that, "a wrong guess
             * has nowhere to go", is no longer true. The SLOT_OBJ arm below
             * predicts from InlineCache::resultKind and resumes AFTER the call
             * with the result read out of the descriptor
             * (branchOnDeoptAt's `lastFromDesc`), which is exactly the
             * somewhere. This arm never got the same treatment, so every list
             * method that is neither a field read nor discarded declines the
             * whole body: `xs.enumerate()` costs
             * docs/probes/p25_enumerate.jai its 12,800,326 interpreted
             * instructions here.
             *
             * Worth doing only for the methods that are CHEAP next to the loop
             * around them -- `contains`, `index`, `count` on a short list.
             * `enumerate`, `map`, `filter` and `sorted` all allocate a list per
             * call, and rows for that shape were built and measured at zero
             * (docs/research/FALSIFIED-list-returning-builtins.md). */
            DiscardKind ldisc = discardedAfter(code, off + 7, count);
            bool discarded = ldisc != DISCARD_NO;
            const JaiJitFieldRead *lfr = jaiJitFieldReadFor(
                OBJ_LIST, AS_STRING(nameVal)->chars, AS_STRING(nameVal)->length,
                argc);
            SlotKind lrkind = SLOT_INT;
            unsigned lrtag = VAL_INT;
            uint8_t lrObjType = 0;
            if (lfr == NULL && !discarded) {
                /* Predicted from the site's own feedback and guarded after the
                 * call, exactly as the SLOT_OBJ arm below does it. */
                if (!jitListResult()) {
                    return subWhy(e, "`list.%s` is neither a field read nor "
                                     "discarded", AS_STRING(nameVal)->chars);
                }
                uint8_t lfb = jaiInvokeResultFeedback(
                    &fn->chunk, jaiReadU16(code + off + 5), probe);
                if (!feedbackSlotKind(lfb, &lrkind, &lrtag, &lrObjType)) {
                    return subWhy(e, "`list.%s` has no usable result kind (%s)",
                                  AS_STRING(nameVal)->chars,
                                  jaiFeedbackName(lfb));
                }
            }

            /* `xs.len()` on a list is one field read. Through the descriptor it meant a GC root push/pop, a
             * bound-method resolve, an arity check and a native call just to read a 32-bit count -- paid every iteration of the ordinary `while i < xs.len()` loop. SLOT_LIST is the type guard, already made. */
            if (lfr != NULL) {
                if (!emitFieldRead(e, lfr, nativeVal, ridx, argc,
                                   (uint32_t)(off + 7))) {
                    return false;
                }
                off += 7;   /* a discarded result is the next OP_POP's */
                break;
            }

            if (!emitDescriptor(e, nativeVal, ridx, argc + 1,
                                (void *)&jitInvokeNative)) {
                return false;
            }

            for (unsigned i = 0; i <= argc; i++) {
                unsigned r;
                if (!popValue(e, &r, NULL)) return false;
            }
            e->wroteHeap = true;
            if (discarded) {
                if (ldisc == DISCARD_POP_RETURN) {
                    if (!emitFusedReturnNull(e, fn)) return false;
                    off += 8;
                    afterUncond = true;
                    continue;
                }
                off += 8;      /* the OP_POP this consumed */
                break;
            }
            if (!pushValue(e, lrkind, 0, NULL)) return false;
            if (lrkind == SLOT_OBJ) e->stackObjType[e->depth - 1] = lrObjType;
            unsigned lrat = e->descOffset +
                            (unsigned)offsetof(JitCallDesc, result);
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, 31, lrat));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, lrtag));
            branchOnDeoptAt(e, JAI_A64_NE, (uint32_t)(off + 7), true);
            /* One byte for a bool: BOOL_VAL writes only the union's `boolean`
             * member, and every SLOT_BOOL consumer tests the whole word. */
            if (lrkind == SLOT_BOOL) {
                emit(e, jaiA64LdrByte(pushReg(e) - 1, 31, lrat + 8));
            } else {
                emit(e, jaiA64LdrX(pushReg(e) - 1, 31, lrat + 8));
            }
            off += 7;
            break;
        }

        case OP_BUILD_RANGE: {
            /* Deferred: the range is only worth building alongside its
             * iterator, which the next instruction asks for. */
            if (e->depth < 2) return false;
            if (e->stack[e->depth - 1] != SLOT_INT) return false;
            if (e->stack[e->depth - 2] != SLOT_INT) return false;
            /* OP_BUILD_RANGE carries one operand byte, so the next opcode is
             * two along. */
            if (off + 2 >= count || code[off + 2] != OP_GET_ITER) {
                e->whyNot = "a range that is not immediately iterated";
                return false;
            }
            e->rangeInclusive = code[off + 1] != 0;
            e->pendingRange = true;
            e->rangeBuildIp = (uint32_t)off;
            /* Both ends hold registers, so the low end is the entry one below
             * the top in the value bank as well as on the stack. */
            {
                unsigned lo = e->valueDepth - 2;
                e->rangeStartKnown = (e->kKnown & (1u << lo)) != 0;
                e->rangeStartVal   = e->rangeStartKnown ? e->kKnownVal[lo] : 0;
            }
            off += 2;
            break;
        }

        case OP_GET_ITER: {
            if (!e->pendingRange) {
                if (e->depth == 0) return false;
                Value itSeen = e->stackSeen[e->depth - 1];
                /* `for c in text`. The iterator itself is the same call the
                 * list path makes -- jaiGetIter takes anything iterable -- so
                 * the only new thing a string needs is an element exemplar and
                 * a proof that it IS a string.
                 *
                 * Worth an arm because it is a ONE-LINK chain on a hot body:
                 * `JAI_JIT_CHAIN=1` says `_scan_identifier` compiles the moment
                 * this is cleared, and character loops are what a lexer is
                 * made of. */
                const bool strIter = jitStrIter() &&
                                     e->stack[e->depth - 1] == SLOT_OBJ &&
                                     IS_STRING(itSeen);
                if (e->stack[e->depth - 1] != SLOT_LIST && !strIter) {
                    /* Named: 4.3% of parser.jai's interpreted work sat behind
                     * this and it said nothing about WHAT was being iterated,
                     * which is the entire question. */
                    return subWhy(e, "iterating a %s, not a list or a range",
                                  IS_OBJ(itSeen)
                                      ? jaiTypeNameStatic(itSeen)
                                      : slotKindName(e->stack[e->depth - 1]));
                }
                if (!e->callsOut) return false;
                /* Carry one element forward: the loop variable's kind comes
                 * from it, and the iterator itself says nothing about what it
                 * will yield. */
                Value srcv = itSeen;
                Value sample = NULL_VAL;
                if (IS_LIST(srcv) && AS_LIST(srcv)->count > 0) {
                    sample = jaiListGet(AS_LIST(srcv), 0);
                }
                if (strIter) {
                    /* Prove OBJ_STRING before the exemplar claims the elements
                     * are strings. VAL_OBJ is every heap object, and an
                     * exemplar the value does not match is how the next arm
                     * comes to read the wrong header.
                     *
                     * The exemplar is an interned EMPTY string, not the first
                     * character: a string yields one-CHARACTER elements, and a
                     * character is not always one byte. Claiming one byte would
                     * send multi-byte text down the one-byte arm, which guards
                     * the length and would therefore deopt once per iteration.
                     * Empty says "a string" and claims nothing else. */
                    emit(e, jaiA64LdrW(JIT_SCRATCH_A,
                                       valueXReg(e, e->valueDepth - 1),
                                       (unsigned)offsetof(Obj, type)));
                    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_STRING));
                    branchOnDeopt(e, JAI_A64_NE);
                    ObjString *empty = jaiStringIntern("", 0);
                    if (empty == NULL) return false;
                    sample = OBJ_VAL((Obj *)empty);
                }
                /* A list this body built has no live sample to read an element
                 * off -- it does not exist yet -- but OP_BUILD_LIST recorded
                 * what went into it. `[expr for x in [a, b, c]]` is the shape
                 * that wanted this: the source list is a literal built one
                 * instruction earlier, and without it every comprehension over
                 * one ran interpreted. */
                if (IS_NULL(sample)) sample = e->stackElem[e->depth - 1];
                if (IS_NULL(sample)) {
                    e->whyNot = "iterating a list with nothing to look at";
                    return false;
                }
                if (!emitDescriptor(e, NULL_VAL, e->depth - 1, 1,
                                    (void *)&jitMakeIter)) {
                    return false;
                }
                unsigned rdrop;
                if (!popValue(e, &rdrop, NULL)) return false;
                /* Shape 1 marks an iterator the runtime has to step; a range is 0 and gets the inline path. Rides on
                 * the stack entry, not the Emit, since a function can build both -- nbody's `advance` runs two range loops then a list loop, and one whole-compile flag made the path taken depend on what came before it. */
                if (!pushValue3(e, SLOT_ITER, 1, NULL, sample, -1)) return false;
                emit(e, jaiA64LdrX(pushReg(e) - 1, 31,
                                   e->descOffset +
                                       (unsigned)offsetof(JitCallDesc, result) + 8));
                e->wroteHeap = true;
                off += 1;
                break;
            }
            if (!e->callsOut) return false;
            if (!pushValue(e, SLOT_INT, 0, NULL)) return false;
            emitConst64(e, pushReg(e) - 1, e->rangeInclusive ? 1 : 0);
            if (!emitDescriptor(e, NULL_VAL, e->depth - 3, 3,
                                (void *)&jitMakeRangeIter)) {
                return false;
            }
            for (unsigned i = 0; i < 3; i++) {
                unsigned r;
                if (!popValue(e, &r, NULL)) return false;
            }
            /* Shape 2 says this body built the range itself, so its step is 1
             * by construction; shape 3 adds a start the emitter knows, carried
             * as the entry's sample. Shape 0 stays the general form, for an
             * ObjIter that arrived from anywhere else. */
            if (!pushValue3(e, SLOT_ITER, e->rangeStartKnown ? 3u : 2u, NULL,
                            e->rangeStartKnown ? INT_VAL(e->rangeStartVal)
                                               : NULL_VAL,
                            -1)) {
                return false;
            }
            emit(e, jaiA64LdrX(pushReg(e) - 1, 31,
                               e->descOffset +
                                   (unsigned)offsetof(JitCallDesc, result) + 8));
            e->wroteHeap = true;
            e->pendingRange = false;
            off += 1;
            break;
        }

        case OP_ITER_RANGE: {
            /* `for x in a..b` opened, with neither object built. The two ints
             * this writes are the whole loop, so there is no descriptor, no
             * root fill and no call out -- and, being ordinary int locals, they
             * compete for registers on the same terms as everything else
             * instead of reserving four the way an ObjIter head does.
             *
             * `end` is one past the last value, WRAPPING, which is what makes
             * `a..=INT64_MAX` terminate: the counter meets INT64_MIN there.
             * Same arithmetic as the interpreter's, deliberately -- see
             * OP_ITER_RANGE in chunk.h. */
            bool     inclusive = code[off + 1] != 0;
            unsigned curSlot   = jaiReadU16(code + off + 2);
            unsigned endSlot   = jaiReadU16(code + off + 4);
            if (e->depth < 2) return false;
            if (e->stack[e->depth - 1] != SLOT_INT) return false;
            if (e->stack[e->depth - 2] != SLOT_INT) return false;
            if (!localInRange(e, curSlot) || !localInRange(e, endSlot)) {
                return false;
            }
            if (!adoptLocalKind(e, curSlot, SLOT_INT, 0, NULL)) return false;
            if (!adoptLocalKind(e, endSlot, SLOT_INT, 0, NULL)) return false;
            if (curSlot == 0 || endSlot == 0) e->usesSlot0 = true;

            unsigned rHi, rLo;
            if (!popValue(e, &rHi, NULL)) return false;
            if (!popValue(e, &rLo, NULL)) return false;
            /* Both ends stay read-only: a popped register may be a local's own,
             * borrowed, and writing it would rewrite the local. */
            if (inclusive) {
                emit(e, jaiA64AddXImm(JIT_SCRATCH_A, rHi, 1));
            } else {
                emit(e, jaiA64MovX(JIT_SCRATCH_A, rHi));
            }
            emit(e, jaiA64SubsXReg(31, rHi, rLo));
            /* An empty range ends where it begins, so the first test already
             * fails and the body never runs. */
            emit(e, jaiA64CselX(JIT_SCRATCH_B, rLo, JIT_SCRATCH_A, JAI_A64_LT));
            localOut(e, endSlot, JIT_SCRATCH_B);
            localOut(e, curSlot, rLo);
            off += 6;
            break;
        }

        case OP_FOR_RANGE_BIND: {
            /* One step of that loop: a compare, a branch, a bind and an add,
             * with nothing to load from the heap and nothing to guard. The two
             * slots are written by OP_ITER_RANGE and by this instruction and by
             * nothing else -- the emitter hands out fresh temporaries for them
             * -- so their kind is a fact of the shape rather than a sample, and
             * this arm has no deopt of its own. A bail lands on this
             * instruction with the counter unadvanced. */
            int16_t  jump    = jaiReadI16(code + off + 1);
            unsigned slot    = jaiReadU16(code + off + 3);
            unsigned curSlot = jaiReadU16(code + off + 5);
            unsigned endSlot = jaiReadU16(code + off + 7);
            if (!localInRange(e, slot)) return false;
            if (!localInRange(e, curSlot) || !localInRange(e, endSlot)) {
                return false;
            }
            if (e->localKind[curSlot] != SLOT_INT ||
                e->localKind[endSlot] != SLOT_INT) {
                return false;
            }
            if (!adoptLocalKind(e, slot, SLOT_INT, 0, NULL)) {
                return subWhy(e, "loop variable in local %u has kind %s, "
                                 "not int", slot,
                              slotKindName(e->localKind[slot]));
            }
            if (slot == 0) e->usesSlot0 = true;

            unsigned rCur = localIn(e, curSlot, JIT_SCRATCH_A);
            unsigned rEnd = localIn(e, endSlot, JIT_SCRATCH_B);
            emit(e, jaiA64SubsXReg(31, rCur, rEnd));
            /* Nothing of this loop's is on the operand stack, so the exit is
             * reached at exactly the depth this branch leaves from. */
            branchTo(e, (uint32_t)((int32_t)(off + 9) + jump), true,
                     JAI_A64_EQ);
            /* Bind before stepping: in register mode the counter's home IS
             * rCur, so the add would destroy the value about to be bound. */
            localOut(e, slot, rCur);
            {
                unsigned dst = localDest(e, curSlot);
                emit(e, jaiA64AddXImm(dst, rCur, 1));
                localOut(e, curSlot, dst);
            }
            off += 9;
            break;
        }

        case OP_GET_ITER_ITEMS: {
            /* `for (a, b) in X.items()`. The emitter cannot know X's type, so
             * it plants this ahead of an ordinary `INVOKE items; GET_ITER` and
             * lets the opcode jump over the pair when X turns out to be a dict,
             * building a lazy ITER_DICT_ITEMS instead. Unarmed, it declined the
             * whole enclosing function -- which is why dict_iter's two loops
             * ran interpreted end to end.
             *
             * The dict case is specialised and the branch is resolved HERE, by
             * walking on at the target rather than by emitting a jump: the
             * skipped `INVOKE items` never executes in this form, so its inline
             * cache is empty and compiling it as dead code would decline. That
             * is only sound because the region really is the emitter's own, so
             * nothing branches into it -- checked below, and backstopped by the
             * fixup resolver, which declines a branch to an offset the walk
             * never reached rather than mis-resolving it.
             *
             * Specialising rather than falling through to the eager `items()`
             * is required, not merely faster: the lazy view raises when the
             * dict changes under the loop and the materialised list does not,
             * so a compiled body that took the other path would answer
             * differently from the interpreter. */
            if (e->depth == 0) return false;
            unsigned sidx = e->depth - 1;
            if (e->stack[sidx] != SLOT_OBJ || !IS_DICT(e->stackSeen[sidx])) {
                e->whyNot = "items() on something that is not a dict";
                return false;
            }
            if (!e->callsOut) return false;
            /* Read before the entry is popped below, not through the model
             * afterwards: the push that replaces it overwrites this cell. */
            Value    itemsDict = e->stackSeen[sidx];
            int16_t  ijump = jaiReadI16(code + off + 1);
            int32_t  after = (int32_t)(off + 3) + ijump;
            /* The emitter's shape exactly: OP_INVOKE (7 bytes) then
             * OP_GET_ITER (1), and the head that follows must be the pair form,
             * since that is the only one shape 4 has an arm for. */
            if (after != off + 11 || after >= stop ||
                code[off + 3] != OP_INVOKE || code[off + 10] != OP_GET_ITER ||
                code[after] != OP_FOR_ITER_PAIR) {
                e->whyNot = "an items() head this tier does not recognise";
                return false;
            }

            emit(e, jaiA64LdrW(JIT_SCRATCH_A, pushReg(e) - 1,
                               (unsigned)offsetof(Obj, type)));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_DICT));
            branchOnDeopt(e, JAI_A64_NE);

            if (!emitDescriptor(e, NULL_VAL, sidx, 1,
                                (void *)&jitMakeItemsIter)) {
                return false;
            }
            unsigned rdrop;
            if (!popValue(e, &rdrop, NULL)) return false;
            /* Shape 4 is an ITER_DICT_ITEMS, and it carries the DICT as its
             * sample rather than an element: the pair head reads the first live
             * entry off it for the component kinds, exactly as the list form
             * reads items[0]. */
            if (!pushValue3(e, SLOT_ITER, 4, NULL, itemsDict, -1)) {
                return false;
            }
            emit(e, jaiA64LdrX(pushReg(e) - 1, 31,
                               e->descOffset +
                                   (unsigned)offsetof(JitCallDesc, result) + 8));
            e->wroteHeap = true;
            off = after;
            break;
        }

        case OP_FOR_ITER_BIND: {
            /* A loop this body built the iterator for: the index lives in the
             * iterator, so every iteration loads and stores it, and a deopt
             * needs nothing -- what is on the stack is already current. */
            /* Only the loop at the OSR entry point owns the reserved iterator registers -- a nested FOR_ITER_BIND
             * built its own iterator and is an ordinary one. Refusing it stopped the outer loops of spectral, mandelbrot, matrix_mul and life from compiling at all, while their inner loops compiled fine. */
            if (!e->osr || !e->hasIter || (uint32_t)off != e->osrTop) {
                if (e->depth == 0 || e->stack[e->depth - 1] != SLOT_ITER) {
                    return false;
                }
                int16_t  fjump = jaiReadI16(code + off + 1);
                unsigned fslot = jaiReadU16(code + off + 3);
                if (!localInRange(e, fslot)) return false;
                unsigned rIt = pushReg(e) - 1;

                /* 0 is a range, 1 an iterator the runtime has to step, 2 and 3
                 * ranges this body built (see OP_GET_ITER). Anything else is
                 * not something the inline range form may assume, so it keeps
                 * the stepped path this test has always sent it down. */
                uint32_t iterShape = e->stackShape[e->depth - 1];
                /* Shape 4 is a dict-items view, and the arm below would read
                 * ObjList's offsets out of an ObjDict were the kind guard not
                 * there to stop it. OP_GET_ITER_ITEMS only makes one when a
                 * pair head follows, so this is unreachable -- kept because the
                 * fact lives in another arm and a decline is the cheap side. */
                if (iterShape == 4) {
                    e->whyNot = "a non-destructuring loop over dict items";
                    return false;
                }
                if (iterShape != 0 && iterShape != 2 && iterShape != 3) {
                    /* A list iterator, stepped inline (jaiIterNext's ITER_LIST
                     * case, instruction for instruction) rather than through
                     * jitIterStep. The kind still comes from the element the
                     * list is holding now, but the tag of what the step
                     * actually produces is GUARDED, and every guard runs
                     * BEFORE the index advances.
                     *
                     * Calling out cannot be made sound: jitIterStep advances
                     * the iterator before it returns, so a guard on its result
                     * has nowhere to resume -- this instruction would re-run
                     * and SKIP an element, and under
                     * JAITHON_JIT_DEOPT_STRESS (where branchOnDeopt is
                     * unconditional) it would skip one on every iteration of
                     * every list loop. Reading the payload out of the
                     * descriptor with no tag check at all was worse: a list
                     * sampled as int and later pushed a str bound the string's
                     * POINTER as an integer (probe: 41080394656 where the
                     * interpreter raises TypeError), and a float's IEEE bits
                     * likewise. Inline, nothing has happened when a guard
                     * fires, so the resume point is this instruction and the
                     * interpreter does the raise. */
                    Value sample = e->stackSeen[e->depth - 1];
                    SlotKind ek; unsigned etag; uint32_t esh = 0;
                    ObjClass *ecl = NULL;
                    if (IS_INT(sample))        { ek = SLOT_INT;   etag = VAL_INT; }
                    else if (IS_FLOAT(sample)) { ek = SLOT_FLOAT; etag = VAL_FLOAT; }
                    else if (IS_BOOL(sample))  { ek = SLOT_BOOL;  etag = VAL_BOOL; }
                    else if (IS_LIST(sample))  { ek = SLOT_LIST;  etag = VAL_OBJ; }
                    else if (rawObjValue(sample)) { ek = SLOT_OBJ; etag = VAL_OBJ; }
                    else if (IS_INSTANCE(sample) && AS_INSTANCE(sample)->klass) {
                        ek = SLOT_INST; etag = VAL_OBJ;
                        ecl = AS_INSTANCE(sample)->klass;
                        esh = ecl->shapeId;
                    } else { e->whyNot = "element kind unknown"; return false; }

                    if (!adoptLocalKindSeen(e, fslot, ek, esh, ecl, sample)) {
                        return subWhy(e, "loop variable in local %u has kind "
                                         "%s, not %s", fslot,
                                      slotKindName(e->localKind[fslot]),
                                      slotKindName(ek));
                    }

                    /* Only OP_GET_ITER's list arm makes a shape-nonzero
                     * SLOT_ITER, and SLOT_ITER is never adopted into a local,
                     * so this is an ITER_LIST over an ObjList. Checked anyway,
                     * one load: the alternative is reading an ObjString or an
                     * ObjDict through ObjList's offsets if another iterator
                     * shape is ever added above, and a wrong answer is the one
                     * failure mode this tier is not allowed. */
                    emit(e, jaiA64LdrW(JIT_SCRATCH_A, rIt,
                                       (unsigned)offsetof(ObjIter, kind)));
                    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, ITER_LIST));
                    branchOnDeopt(e, JAI_A64_NE);

                    emit(e, jaiA64LdrX(JIT_SCRATCH_C, rIt,
                                       (unsigned)offsetof(ObjIter, source) + 8));
                    /* Nothing names this list -- it is whatever the iterator
                     * was built over -- so the storage is proved rather than
                     * pinned. */
                    emitListBoxedGuard(e, JIT_SCRATCH_C, JIT_SCRATCH_A);

                    /* Mutation first, as jaiIterNext tests it: a list that grew
                     * or shrank under the loop must raise, and the version is
                     * the only thing that says so. */
                    emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_C,
                                       (unsigned)offsetof(ObjList, version)));
                    emit(e, jaiA64LdrW(JIT_SCRATCH_B, rIt,
                                       (unsigned)offsetof(ObjIter, version)));
                    emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
                    branchOnDeopt(e, JAI_A64_NE);

                    /* `limit` is the snapshot count, not the live one -- the
                     * version guard above owns any disagreement between them. */
                    emit(e, jaiA64LdrX(JIT_SCRATCH_A, rIt,
                                       (unsigned)offsetof(ObjIter, index)));
                    emit(e, jaiA64LdrX(JIT_SCRATCH_B, rIt,
                                       (unsigned)offsetof(ObjIter, limit)));
                    emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
                    /* The exhausted arm drops the iterator, so the target is
                     * reached one entry shallower than this branch leaves
                     * from. */
                    branchToDepth(e, (uint32_t)((int32_t)(off + 5) + fjump),
                                  JAI_A64_GE,
                                  (int)stackSignatureAt(e, e->depth - 1));

                    /* Reload items rather than hoisting: a reallocation bumps
                     * the version, which the guard above covers. */
                    emit(e, jaiA64LdrX(JIT_SCRATCH_C, JIT_SCRATCH_C,
                                       (unsigned)offsetof(ObjList, items)));
                    emit(e, jaiA64AddXLsl(JIT_SCRATCH_C, JIT_SCRATCH_C,
                                          JIT_SCRATCH_A, 4));

                    emit(e, jaiA64LdrW(JIT_SCRATCH_B, JIT_SCRATCH_C, 0));
                    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_B, etag));
                    branchOnDeopt(e, JAI_A64_NE);

                    if (ek == SLOT_INST) {
                        /* VAL_OBJ is every heap object, so the object type is
                         * checked before `klass` is read -- otherwise a list
                         * that gained a string reads `klass` one word past an
                         * ObjString's header. JIT_SCRATCH_A still holds the
                         * index and must survive to the store below. */
                        emit(e, jaiA64LdrX(JIT_SCRATCH_B, JIT_SCRATCH_C, 8));
                        emit(e, jaiA64LdrW(JIT_SCRATCH_D, JIT_SCRATCH_B,
                                           (unsigned)offsetof(Obj, type)));
                        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_D, OBJ_INSTANCE));
                        branchOnDeopt(e, JAI_A64_NE);
                        emit(e, jaiA64LdrX(JIT_SCRATCH_B, JIT_SCRATCH_B,
                                           (unsigned)offsetof(ObjInstance, klass)));
                        emit(e, jaiA64LdrW(JIT_SCRATCH_B, JIT_SCRATCH_B,
                                           (unsigned)offsetof(ObjClass, shapeId)));
                        emitConst64(e, JIT_SCRATCH_D, (int64_t)esh);
                        emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_B, JIT_SCRATCH_D));
                        branchOnDeopt(e, JAI_A64_NE);
                    } else if (ek == SLOT_LIST) {
                        /* Same contract as OP_GET_INDEX's own SLOT_LIST arm:
                         * VAL_OBJ is every heap object, not specifically a
                         * list, so the object type is confirmed here, once,
                         * before a SLOT_LIST consumer trusts it with no check
                         * of its own. JIT_SCRATCH_A still holds the index and
                         * must survive to the store below. */
                        emit(e, jaiA64LdrX(JIT_SCRATCH_B, JIT_SCRATCH_C, 8));
                        emit(e, jaiA64LdrW(JIT_SCRATCH_D, JIT_SCRATCH_B,
                                           (unsigned)offsetof(Obj, type)));
                        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_D, OBJ_LIST));
                        branchOnDeopt(e, JAI_A64_NE);
                    }

                    /* Past the last guard: advance, then bind. The advance goes
                     * first because localOut may use JIT_SCRATCH_C/D for the
                     * tag and the index has to be stored out of a register the
                     * write cannot touch. One byte for a bool: see the note in
                     * OP_GET_INDEX -- `strb` is what BOOL_VAL compiles to, so
                     * the rest of the payload word is stale. */
                    emit(e, jaiA64AddXImm(JIT_SCRATCH_B, JIT_SCRATCH_A, 1));
                    emit(e, jaiA64StrX(JIT_SCRATCH_B, rIt,
                                       (unsigned)offsetof(ObjIter, index)));
                    if (ek == SLOT_BOOL) {
                        emit(e, jaiA64LdrByte(JIT_SCRATCH_A, JIT_SCRATCH_C, 8));
                    } else {
                        emit(e, jaiA64LdrX(JIT_SCRATCH_A, JIT_SCRATCH_C, 8));
                    }
                    localOut(e, fslot, JIT_SCRATCH_A);
                    /* The index store is a heap write, as the call-out this
                     * replaced was: a bail after it would re-run the loop from
                     * the top with the iterator already advanced. */
                    e->wroteHeap = true;
                    off += 5;
                    break;
                }

                emit(e, jaiA64LdrX(JIT_SCRATCH_A, rIt,
                                   (unsigned)offsetof(ObjIter, index)));
                emit(e, jaiA64LdrX(JIT_SCRATCH_B, rIt,
                                   (unsigned)offsetof(ObjIter, limit)));
                if (!adoptLocalKind(e, fslot, SLOT_INT, 0, NULL)) return false;
                emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
                /* The exhausted arm drops the iterator, so the target is
                 * reached one entry shallower than this branch leaves from. */
                branchToDepth(e, (uint32_t)((int32_t)(off + 5) + fjump),
                              JAI_A64_GE,
                              (int)stackSignatureAt(e, e->depth - 1));
                /* A range yields `start + index * step`, not the index itself (see jaiIterNext's ITER_RANGE case) --
                 * the index is always zero-based, so using it directly is only right for `0..n` in unit steps. `for j in i + 1..n` once counted from zero instead of i+1, a plausible wrong answer (nested loops summed the wrong pairs), not a crash. The dead-after-compare limit register carries the index across to the increment. */
                emit(e, jaiA64MovX(JIT_SCRATCH_B, JIT_SCRATCH_A));
                /* Both halves of that map are loop-invariant, and for a range
                 * this body built (shape 2) the step is 1 by construction --
                 * jitMakeRangeIter has no step argument. With the start a
                 * literal too (shape 3) nothing about the ObjRange has to be
                 * read at all, which is five loads and a multiply off the back
                 * of every nested `for k in 0..n`: matrix_mul spends fourteen
                 * of its innermost forty-nine instructions on this counter. */
                /* Shape 3's constant travels in the entry's sample, and an
                 * entry can reach here having been through a local, where the
                 * sample need not have come with it. No sample, no shortcut:
                 * the general form below is right for any range. */
                if (iterShape == 3 && !IS_INT(e->stackSeen[e->depth - 1])) {
                    iterShape = 2;
                }
                if (iterShape == 3) {
                    int64_t k = AS_INT(e->stackSeen[e->depth - 1]);
                    if (k > 0 && k <= 4095) {
                        emit(e, jaiA64AddXImm(JIT_SCRATCH_A, JIT_SCRATCH_A,
                                              (unsigned)k));
                    } else if (k < 0 && k >= -4095) {
                        emit(e, jaiA64SubXImm(JIT_SCRATCH_A, JIT_SCRATCH_A,
                                              (unsigned)(-k)));
                    } else if (k != 0) {
                        emitConst64(e, JIT_SCRATCH_D, k);
                        emit(e, jaiA64AddX(JIT_SCRATCH_A, JIT_SCRATCH_D,
                                           JIT_SCRATCH_A));
                    }
                } else {
                    emit(e, jaiA64LdrX(JIT_SCRATCH_C, rIt,
                                       (unsigned)offsetof(ObjIter, source) + 8));
                    if (iterShape != 2) {
                        emit(e, jaiA64LdrX(JIT_SCRATCH_D, JIT_SCRATCH_C,
                                           (unsigned)offsetof(ObjRange, step)));
                        emit(e, jaiA64MulX(JIT_SCRATCH_A, JIT_SCRATCH_A,
                                           JIT_SCRATCH_D));
                    }
                    emit(e, jaiA64LdrX(JIT_SCRATCH_D, JIT_SCRATCH_C,
                                       (unsigned)offsetof(ObjRange, start)));
                    emit(e, jaiA64AddX(JIT_SCRATCH_A, JIT_SCRATCH_D,
                                       JIT_SCRATCH_A));
                }
                localOut(e, fslot, JIT_SCRATCH_A);
                emit(e, jaiA64AddXImm(JIT_SCRATCH_B, JIT_SCRATCH_B, 1));
                emit(e, jaiA64StrX(JIT_SCRATCH_B, rIt,
                                   (unsigned)offsetof(ObjIter, index)));
                off += 5;
                break;
            }
            /* Only as the head of the loop being compiled, and only for a range from zero in unit steps -- that's
             * what makes the yielded value the index itself. Everything else about the iterator is a runtime fact, checked at entry. */
            if (!e->osr || !e->hasIter) return false;
            if ((uint32_t)off != e->osrTop) return false;
            int16_t  jump = jaiReadI16(code + off + 1);
            unsigned slot = jaiReadU16(code + off + 3);
            if (!localInRange(e, slot)) return false;

            if (e->iterKind == 2) {
                /* A list at the loop head: reserved registers mean what they do for a range, except JIT_START_REG
                 * holds the ObjList instead of a first value. Without this, a top-level `for x in xs` was never even attempted -- the gate refused anything not a range before compileOsr ran, so not even a decline was recorded. */
                Value sample = e->elemSample;
                SlotKind ek;
                unsigned etag;
                uint32_t esh = 0;
                ObjClass *ecl = NULL;
                if (IS_INT(sample))        { ek = SLOT_INT;   etag = VAL_INT; }
                else if (IS_FLOAT(sample)) { ek = SLOT_FLOAT; etag = VAL_FLOAT; }
                else if (IS_BOOL(sample))  { ek = SLOT_BOOL;  etag = VAL_BOOL; }
                else if (IS_LIST(sample))  { ek = SLOT_LIST;  etag = VAL_OBJ; }
                else if (rawObjValue(sample)) { ek = SLOT_OBJ; etag = VAL_OBJ; }
                else if (IS_INSTANCE(sample) &&
                         AS_INSTANCE(sample)->klass != NULL) {
                    /* The object type is checked before the class is read,
                     * because VAL_OBJ is every heap object and a list holding
                     * a string beside the sampled instance would otherwise
                     * read `klass` one word past an ObjString's header. */
                    ek = SLOT_INST; etag = VAL_OBJ;
                    ecl = AS_INSTANCE(sample)->klass;
                    esh = ecl->shapeId;
                } else {
                    e->whyNot = "list element kind unknown";
                    return false;
                }
                /* The list holds more than one class, so the loop variable is
                 * an instance of no particular one. Two classes are not two
                 * KINDS -- the representation is the same untagged pointer --
                 * so the slot widens rather than the compile failing, and the
                 * call sites inside dispatch by name. Only sound at THIS
                 * instruction: an OSR body is walked from its loop head, so
                 * nothing has been emitted against the class being dropped. */
                if (ek == SLOT_INST && e->elemMixed) {
                    esh = 0;
                    ecl = NULL;
                    e->localTyped[slot] = false;
                }
                if (!adoptLocalKindSeen(e, slot, ek, esh, ecl, sample)) {
                    return subWhy(e, "loop variable in local %u has kind %s, "
                                     "not %s", slot,
                                  slotKindName(e->localKind[slot]),
                                  slotKindName(ek));
                }
                e->iterSlot = slot;
                e->iterExit = (uint32_t)((int32_t)(off + 5) + jump);

                /* Mutation first: a list that grew or shrank under the loop
                 * must raise, and the version is the only thing that says so.
                 * Nothing has happened yet, so this resumes at this very
                 * instruction and the interpreter raises it properly. */
                /* Storage is pinned per form, not checked here: jaiJitEnterOsr
                 * matches JaiOsrForm::iterStg against the list this head is
                 * about to walk, so by the time the body runs the stride below
                 * is already the right one. */
                /* Unpinned is unproved, and the form records LIST_STG_ANY
                 * for the head -- but a deopt guard here is not the answer.
                 * `e->elemStgPin` is false for any body that calls out, and a
                 * `push` is a call, so `for x in xs { out.push(f(x)) }` over a
                 * `list[int]` would fail that guard on its FIRST element and
                 * on every entry after: 11x slower than boxed, and the head's
                 * give-up counter never fires because a bail is not a decline.
                 * So the head dispatches like every other site. */
                ListAccess iAcc;
                iAcc.stg = e->elemStgPin ? e->elemStg
                                         : (uint8_t)LIST_STORE_BOXED;
                iAcc.alt = e->elemStgPin ? iAcc.stg : listAltFor(ek);
                iAcc.dynamic = !e->elemStgPin && iAcc.alt != LIST_STORE_BOXED;
                uint8_t iStg = iAcc.stg;
                if (iStg != LIST_STORE_BOXED && ek != listStgKind(iStg)) {
                    return subWhy(e, "element kind %d is not storage %u's",
                                  (int)ek, iStg);
                }

                emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_START_REG,
                                   (unsigned)offsetof(ObjList, version)));
                emit(e, jaiA64LdrW(JIT_SCRATCH_B, JIT_ITER_REG,
                                   (unsigned)offsetof(ObjIter, version)));
                emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
                branchOnDeopt(e, JAI_A64_NE);

                emit(e, jaiA64SubsXReg(31, JIT_IDX_REG, JIT_LIM_REG));
                branchTo(e, e->iterExit, true, JAI_A64_GE);

                /* Reload items each time rather than hoisting: a reallocation
                 * bumps the version so the guard above covers it, and one ldr
                 * removes the question entirely. */
                int iSkip = listDispatchBegin(e, &iAcc, JIT_START_REG,
                                              JIT_SCRATCH_A);
                emit(e, jaiA64LdrX(JIT_SCRATCH_C, JIT_START_REG,
                                   (unsigned)offsetof(ObjList, items)));
                emit(e, jaiA64AddXLsl(JIT_SCRATCH_C, JIT_SCRATCH_C,
                                      JIT_IDX_REG, listStgShift(iStg)));

                /* Nothing to check on an unboxed element: no tag, and no
                 * object behind it whose type could surprise the arms below. */
                if (iStg == LIST_STORE_BOXED) {
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_C, 0));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, etag));
                branchOnDeopt(e, JAI_A64_NE);

                if (ek == SLOT_INST) {
                    emit(e, jaiA64LdrX(JIT_SCRATCH_D, JIT_SCRATCH_C, 8));
                    /* Same hazard as above: VAL_OBJ covers every heap object, so the type is checked before `klass` is read. */
                    emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_D,
                                       (unsigned)offsetof(Obj, type)));
                    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_INSTANCE));
                    branchOnDeopt(e, JAI_A64_NE);
                    /* The class is checked only when one was pinned. A widened
                     * slot has no class to check against, and that it is an
                     * instance at all -- which the guard above settles -- is
                     * everything a by-name call needs of it. */
                    if (esh != 0) {
                        emit(e, jaiA64LdrX(JIT_SCRATCH_D, JIT_SCRATCH_D,
                                           (unsigned)offsetof(ObjInstance, klass)));
                        emit(e, jaiA64LdrW(JIT_SCRATCH_D, JIT_SCRATCH_D,
                                           (unsigned)offsetof(ObjClass, shapeId)));
                        emitConst64(e, JIT_SCRATCH_A, (int64_t)esh);
                        emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_D, JIT_SCRATCH_A));
                        branchOnDeopt(e, JAI_A64_NE);
                    }
                } else if (ek == SLOT_LIST) {
                    /* Same contract as OP_GET_INDEX's own SLOT_LIST arm: VAL_OBJ
                     * is every heap object, not specifically a list, so the
                     * object type is confirmed here, once, before a SLOT_LIST
                     * consumer trusts it with no check of its own. */
                    emit(e, jaiA64LdrX(JIT_SCRATCH_D, JIT_SCRATCH_C, 8));
                    emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_D,
                                       (unsigned)offsetof(Obj, type)));
                    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_LIST));
                    branchOnDeopt(e, JAI_A64_NE);
                }
                }

                /* Both arms leave JIT_SCRATCH_C on the payload, as
                 * OP_GET_INDEX's do, so the load below serves either. */
                if (iStg == LIST_STORE_BOXED) {
                    emit(e, jaiA64AddXImm(JIT_SCRATCH_C, JIT_SCRATCH_C, 8));
                }
                if (iSkip >= 0) {
                    int iJoin = listDispatchElse(e, iSkip);
                    emit(e, jaiA64LdrX(JIT_SCRATCH_C, JIT_START_REG,
                                       (unsigned)offsetof(ObjList, items)));
                    emit(e, jaiA64AddXLsl(JIT_SCRATCH_C, JIT_SCRATCH_C,
                                          JIT_IDX_REG,
                                          listStgShift(iAcc.alt)));
                    listDispatchEnd(e, iJoin);
                }

                /* One byte for a bool: see the note in OP_GET_INDEX. `strb` is
                 * what BOOL_VAL compiles to, so the rest of the payload word is
                 * stale, and a SLOT_BOOL register must hold 0 or 1. */
                unsigned eAt = 0;
                if (ek == SLOT_BOOL) {
                    emit(e, jaiA64LdrByte(JIT_SCRATCH_A, JIT_SCRATCH_C, eAt));
                } else {
                    emit(e, jaiA64LdrX(JIT_SCRATCH_A, JIT_SCRATCH_C, eAt));
                }
                localOut(e, slot, JIT_SCRATCH_A);
                emit(e, jaiA64AddXImm(JIT_IDX_REG, JIT_IDX_REG, 1));
                off += 5;
                break;
            }

            if (!adoptLocalKind(e, slot, SLOT_INT, 0, NULL)) return false;
            e->iterSlot = slot;
            e->iterExit = (uint32_t)((int32_t)(off + 5) + jump);

            emit(e, jaiA64SubsXReg(31, JIT_IDX_REG, JIT_LIM_REG));
            branchTo(e, e->iterExit, true, JAI_A64_GE);
            /* Value is start + index, not the index (`for j in i + 1..n` is nbody advance's inner loop): both
             * registers were biased by the start in the prologue, so the value IS the index register and the add that used to be here is gone. */
            localOut(e, slot, JIT_IDX_REG);
            emit(e, jaiA64AddXImm(JIT_IDX_REG, JIT_IDX_REG, 1));
            off += 5;
            break;
        }

        case OP_FOR_ITER_PAIR: {
            /* `for (a, b) in xs` over a list of 2-tuples, stepped inline.
             *
             * Deliberately not a call-out to a helper that steps the iterator.
             * OP_FOR_ITER_BIND's list arm used to be one, and it is inline for
             * the same reason this is: such a helper advances the iterator
             * before returning, so a component-kind guard after it would deopt
             * to an instruction that has already happened -- and under
             * JAITHON_JIT_DEOPT_STRESS every guard is turned into an
             * unconditional branch, which would then skip an element on every
             * pass. Every guard here is placed BEFORE anything is written, so
             * resuming at this very instruction is exact whether the guard
             * failed for a real reason or because the stress flag forced it.
             *
             * The reachable iterator is always a list one -- shape != 0 comes
             * only from OP_GET_ITER's jitMakeIter, over a SLOT_LIST -- but the
             * kind is guarded anyway rather than assumed, because that fact
             * lives two arms away. */
            /* As the head of an OSR loop the iterator is not on the modelled
             * operand stack at all -- it arrives in a reserved register and
             * stays on the interpreter's stack, which is what lets an exit
             * leave without unwinding anything. Only iterKind 3 gets here: a
             * range or list head is an OP_FOR_ITER_BIND. */
            bool pairHead = e->osr && e->hasIter && e->iterKind == 3 &&
                            (uint32_t)off == e->osrTop;
            if (!pairHead &&
                (e->depth == 0 || e->stack[e->depth - 1] != SLOT_ITER)) {
                return false;
            }
            if (!pairHead && e->stackShape[e->depth - 1] == 0) {
                /* A range head yields ints, which never destructure. */
                e->whyNot = "destructuring what a range yields";
                return false;
            }
            int16_t  pjump = jaiReadI16(code + off + 1);
            unsigned pslotA = jaiReadU16(code + off + 3);
            unsigned pslotB = jaiReadU16(code + off + 5);
            if (!localInRange(e, pslotA) || !localInRange(e, pslotB)) {
                return false;
            }
            if (pslotA == pslotB) {
                /* `for (x, x) in …`: legal, and the second write wins. Not
                 * worth a special case; the interpreter keeps it. */
                e->whyNot = "a pair loop binding one slot twice";
                return false;
            }

            /* Component kinds come from the pair the source was holding when
             * the iterator was built (OP_GET_ITER / OP_GET_ITER_ITEMS carries
             * it forward), and the guards below are what make that a
             * specialisation rather than an assumption. */
            bool pairIsDict = pairHead || e->stackShape[e->depth - 1] == 4;
            Value psample = pairHead ? e->elemSample : e->stackSeen[e->depth - 1];
            SlotKind pk[2];
            unsigned ptag[2];
            Value pseen[2];
            if (pairIsDict) {
                /* Shape 4 carries the dict itself, so the sample is its first
                 * live entry -- the one the loop is about to yield. */
                if (!IS_DICT(psample) ||
                    !firstLiveEntry(&AS_DICT(psample)->table,
                                    &pseen[0], &pseen[1])) {
                    e->whyNot = "iterating a dict with nothing to look at";
                    return false;
                }
            } else {
                if (!IS_TUPLE(psample) || AS_TUPLE(psample)->count != 2) {
                    e->whyNot = "pair element is not a 2-tuple";
                    return false;
                }
                pseen[0] = AS_TUPLE(psample)->items[0];
                pseen[1] = AS_TUPLE(psample)->items[1];
            }
            for (unsigned i = 0; i < 2; i++) {
                Value v = pseen[i];
                if (IS_INT(v))        { pk[i] = SLOT_INT;   ptag[i] = VAL_INT; }
                else if (IS_FLOAT(v)) { pk[i] = SLOT_FLOAT; ptag[i] = VAL_FLOAT; }
                else if (IS_BOOL(v))  { pk[i] = SLOT_BOOL;  ptag[i] = VAL_BOOL; }
                else if (IS_OBJ(v) && AS_OBJ(v) != NULL) {
                    /* Held raw, like any other SLOT_OBJ: the tag guard is the
                     * whole of what this promises, and an arm that wants to
                     * know WHICH object type checks that itself. */
                    pk[i] = SLOT_OBJ; ptag[i] = VAL_OBJ;
                } else {
                    e->whyNot = "pair component kind unknown";
                    return false;
                }
            }
            if (!adoptLocalKindSeen(e, pslotA, pk[0], 0, NULL, pseen[0]) ||
                !adoptLocalKindSeen(e, pslotB, pk[1], 0, NULL, pseen[1])) {
                return subWhy(e, "a pair's loop variables (locals %u and %u) "
                                 "have kinds %s and %s", pslotA, pslotB,
                              slotKindName(e->localKind[pslotA]),
                              slotKindName(e->localKind[pslotB]));
            }

            unsigned rIter = pairHead ? JIT_PAIR_ITER_REG : pushReg(e) - 1;
            uint32_t pairExit = (uint32_t)((int32_t)(off + 7) + pjump);
            /* A head's exit leaves the model at the depth it is already at --
             * the iterator it drops was never in the model. Registering it as
             * iterExit is what makes the exit stub tell the interpreter to pop
             * the exhausted iterator off its own stack. */
            int pairExitDepth = pairHead
                                    ? (int)stackSignature(e)
                                    : (int)stackSignatureAt(e, e->depth - 1);
            if (pairHead) e->iterExit = pairExit;

            if (pairIsDict) {
                /* iterStepPairFast's ITER_DICT_ITEMS case plus the jaiTableNext
                 * it calls, inline. Same discipline as the list arm below:
                 * every guard, and the whole scan, runs before the index is
                 * written back, so a deopt -- forced or real -- resumes at this
                 * instruction with the iterator exactly as the interpreter left
                 * it and re-does the scan. */
                _Static_assert(sizeof(JaiEntry) == 48,
                               "the dict-items step scales the order index by "
                               "hand: slot * 16 * 3");
                const unsigned tOff = (unsigned)offsetof(ObjDict, table);

                emit(e, jaiA64LdrW(JIT_SCRATCH_A, rIter,
                                   (unsigned)offsetof(ObjIter, kind)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, ITER_DICT_ITEMS));
                branchOnDeopt(e, JAI_A64_NE);
                emit(e, jaiA64LdrX(JIT_SCRATCH_B, rIter,
                                   (unsigned)offsetof(ObjIter, source) + 8));
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_B,
                                   (unsigned)offsetof(Obj, type)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_DICT));
                branchOnDeopt(e, JAI_A64_NE);

                /* A dict that changed under the loop must raise, and only the
                 * version says so. jaiIterNext owns that message, so the guard
                 * hands the whole instruction back unadvanced. */
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_B,
                                   tOff + (unsigned)offsetof(JaiTable, version)));
                emit(e, jaiA64LdrW(JIT_SCRATCH_C, rIter,
                                   (unsigned)offsetof(ObjIter, version)));
                emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_C));
                branchOnDeopt(e, JAI_A64_NE);

                emit(e, jaiA64LdrX(JIT_SCRATCH_C, rIter,
                                   (unsigned)offsetof(ObjIter, index)));
                emit(e, jaiA64LdrW(JIT_SCRATCH_D, JIT_SCRATCH_B,
                                   tOff +
                                       (unsigned)offsetof(JaiTable, orderCount)));

                /* The scan. `order` holds an entry index per insertion
                 * position, negative where a delete left a hole, so a dict with
                 * deletions in it costs one extra pass per hole and nothing
                 * else. orderCount is hoisted because only a mutation can move
                 * it and the version guard above has already excluded one.
                 *
                 * branchToDepth inside a loop is sound only because it settles
                 * nothing here: the branchOnDeopt three lines up fails the
                 * compile outright if a deferred value is live, so the settle
                 * it performs is a no-op and cannot be re-executed. */
                unsigned scanTop = e->count;
                emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_C, JIT_SCRATCH_D));
                /* The exhausted arm drops the iterator, so the target is
                 * reached one entry shallower than this branch leaves from. */
                branchToDepth(e, pairExit, JAI_A64_GE, pairExitDepth);
                emit(e, jaiA64LdrX(JIT_SCRATCH_A, JIT_SCRATCH_B,
                                   tOff + (unsigned)offsetof(JaiTable, order)));
                emit(e, jaiA64AddXLsl(JIT_SCRATCH_A, JIT_SCRATCH_A,
                                      JIT_SCRATCH_C, 2));
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_A, 0));
                emit(e, jaiA64AddXImm(JIT_SCRATCH_C, JIT_SCRATCH_C, 1));
                /* A hole: the slot is int32 and negative, which after the
                 * zero-extending load is bit 31 set. Measured against e->count
                 * so an instruction added above cannot rot the distance. */
                emit(e, jaiA64Tbnz(JIT_SCRATCH_A, 31,
                                   (int32_t)scanTop - (int32_t)e->count));

                /* entries + slot * sizeof(JaiEntry): slot << 4, then + itself
                 * twice over, which is the 48 the assert above pins. */
                emit(e, jaiA64LdrX(JIT_SCRATCH_D, JIT_SCRATCH_B,
                                   tOff + (unsigned)offsetof(JaiTable, entries)));
                emit(e, jaiA64LslX(JIT_SCRATCH_A, JIT_SCRATCH_A, 4));
                emit(e, jaiA64AddXLsl(JIT_SCRATCH_A, JIT_SCRATCH_A,
                                      JIT_SCRATCH_A, 1));
                emit(e, jaiA64AddX(JIT_SCRATCH_B, JIT_SCRATCH_D, JIT_SCRATCH_A));

                /* Key and value carry the kinds sampled off the first live
                 * entry; a dict that later holds another kind fails here with
                 * nothing written. */
                for (unsigned i = 0; i < 2; i++) {
                    unsigned at = i == 0 ? (unsigned)offsetof(JaiEntry, key)
                                         : (unsigned)offsetof(JaiEntry, value);
                    emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_B, at));
                    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, ptag[i]));
                    branchOnDeopt(e, JAI_A64_NE);
                }

                /* Past the last guard. The index goes first because localOut
                 * spends JIT_SCRATCH_C and JIT_SCRATCH_D on a frame-resident
                 * slot's tag; only JIT_SCRATCH_B survives it. */
                emit(e, jaiA64StrX(JIT_SCRATCH_C, rIter,
                                   (unsigned)offsetof(ObjIter, index)));
                for (unsigned i = 0; i < 2; i++) {
                    unsigned at = (i == 0 ? (unsigned)offsetof(JaiEntry, key)
                                          : (unsigned)offsetof(JaiEntry, value))
                                  + 8u;
                    /* A bool is one byte (see OP_GET_INDEX) -- the rest of its
                     * payload word is stale, and a SLOT_BOOL register must hold
                     * 0 or 1. */
                    if (pk[i] == SLOT_BOOL) {
                        emit(e, jaiA64LdrByte(JIT_SCRATCH_A, JIT_SCRATCH_B, at));
                    } else {
                        emit(e, jaiA64LdrX(JIT_SCRATCH_A, JIT_SCRATCH_B, at));
                    }
                    localOut(e, i == 0 ? pslotA : pslotB, JIT_SCRATCH_A);
                }
                e->wroteHeap = true;
                off += 7;
                break;
            }

            /* Really a list iterator, and its source really a list. */
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, rIter,
                               (unsigned)offsetof(ObjIter, kind)));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, ITER_LIST));
            branchOnDeopt(e, JAI_A64_NE);
            emit(e, jaiA64LdrX(JIT_SCRATCH_B, rIter,
                               (unsigned)offsetof(ObjIter, source) + 8));
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_B,
                               (unsigned)offsetof(Obj, type)));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_LIST));
            branchOnDeopt(e, JAI_A64_NE);

            /* A list that grew or shrank under the loop must raise, and the
             * version is the only thing that says so. Nothing has happened
             * yet, so the interpreter raises it from this instruction. */
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_B,
                               (unsigned)offsetof(ObjList, version)));
            emit(e, jaiA64LdrW(JIT_SCRATCH_C, rIter,
                               (unsigned)offsetof(ObjIter, version)));
            emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_C));
            branchOnDeopt(e, JAI_A64_NE);

            emit(e, jaiA64LdrX(JIT_SCRATCH_C, rIter,
                               (unsigned)offsetof(ObjIter, index)));
            emit(e, jaiA64LdrX(JIT_SCRATCH_D, rIter,
                               (unsigned)offsetof(ObjIter, limit)));
            emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_C, JIT_SCRATCH_D));
            /* The exhausted arm drops the iterator, so the target is reached
             * one entry shallower than this branch leaves from. */
            branchToDepth(e, pairExit, JAI_A64_GE, pairExitDepth);

            /* items is reloaded rather than hoisted: a reallocation bumps the
             * version, which the guard above covers, and one ldr removes the
             * question. */
            emitListBoxedGuard(e, JIT_SCRATCH_B, JIT_SCRATCH_A);
            emit(e, jaiA64LdrX(JIT_SCRATCH_B, JIT_SCRATCH_B,
                               (unsigned)offsetof(ObjList, items)));
            emit(e, jaiA64AddXLsl(JIT_SCRATCH_B, JIT_SCRATCH_B,
                                  JIT_SCRATCH_C, 4));

            /* The element is a 2-tuple whose components have the sampled
             * kinds. Object type is checked before `count` is read: VAL_OBJ
             * covers every heap object, so a string beside the sampled tuple
             * would otherwise have `count` read out of an ObjString. */
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_B, 0));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, VAL_OBJ));
            branchOnDeopt(e, JAI_A64_NE);
            emit(e, jaiA64LdrX(JIT_SCRATCH_B, JIT_SCRATCH_B, 8));
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_B,
                               (unsigned)offsetof(Obj, type)));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_TUPLE));
            branchOnDeopt(e, JAI_A64_NE);
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_B,
                               (unsigned)offsetof(ObjTuple, count)));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, 2));
            branchOnDeopt(e, JAI_A64_NE);
            for (unsigned i = 0; i < 2; i++) {
                unsigned at = (unsigned)offsetof(ObjTuple, items) +
                              i * (unsigned)sizeof(Value);
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_B, at));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, ptag[i]));
                branchOnDeopt(e, JAI_A64_NE);
            }

            /* Past the last guard: from here nothing may fail, and every write
             * below is what the interpreter would have left behind. The index
             * goes first because localOut spends JIT_SCRATCH_C and
             * JIT_SCRATCH_D on the tag of a slot that lives in the frame; only
             * the element pointer in JIT_SCRATCH_B survives it. */
            emit(e, jaiA64AddXImm(JIT_SCRATCH_C, JIT_SCRATCH_C, 1));
            emit(e, jaiA64StrX(JIT_SCRATCH_C, rIter,
                               (unsigned)offsetof(ObjIter, index)));
            /* A bool is one byte (see OP_GET_INDEX) -- the rest of its payload
             * word is stale, and a SLOT_BOOL register must hold 0 or 1. */
            for (unsigned i = 0; i < 2; i++) {
                unsigned at = (unsigned)offsetof(ObjTuple, items) +
                              i * (unsigned)sizeof(Value) + 8u;
                if (pk[i] == SLOT_BOOL) {
                    emit(e, jaiA64LdrByte(JIT_SCRATCH_A, JIT_SCRATCH_B, at));
                } else {
                    emit(e, jaiA64LdrX(JIT_SCRATCH_A, JIT_SCRATCH_B, at));
                }
                localOut(e, i == 0 ? pslotA : pslotB, JIT_SCRATCH_A);
            }
            e->wroteHeap = true;
            off += 7;
            break;
        }

        case OP_GET_INDEX: {
            /* `s[i]` on a string: every guard is a load+compare, and the result is a table lookup, not an
             * allocation (the 128 one-byte strings are made once and shared). Without this the whole loop around a character scan declines -- why `str_search` ran interpreted end to end, and every lexer scans one byte at a time. */
            if (e->depth >= 2 && e->stack[e->depth - 1] == SLOT_INT &&
                e->stack[e->depth - 2] == SLOT_OBJ &&
                IS_STRING(e->stackSeen[e->depth - 2])) {
                unsigned rIdx = pushReg(e) - 1;
                unsigned rStr = valueXReg(e, e->valueDepth - 2);

                /* Really a string, and not something else this object slot
                 * happened to hold when the loop was compiled. */
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, rStr,
                                   (unsigned)offsetof(Obj, type)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_STRING));
                branchOnDeopt(e, JAI_A64_NE);

                /* ASCII only: one scalar is one byte, so indexing is indexing.
                 * `scalars` is UINT32_MAX until something asks, so the first
                 * time through deopts and the interpreter fills it in. */
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, rStr,
                                   (unsigned)offsetof(ObjString, length)));
                emit(e, jaiA64LdrW(JIT_SCRATCH_B, rStr,
                                   (unsigned)offsetof(ObjString, scalars)));
                emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
                branchOnDeopt(e, JAI_A64_NE);

                /* jaiNormalizeIndex, then one unsigned compare for both ends.
                 * `length` came from an `ldr w`, so it is already the whole
                 * register. */
                emitBoundsNormalise(e, rIdx, JIT_SCRATCH_A, JIT_SCRATCH_B,
                                    false);

                emit(e, jaiA64LdrX(JIT_SCRATCH_C, rStr,
                                   (unsigned)offsetof(ObjString, chars)));
                emit(e, jaiA64AddX(JIT_SCRATCH_C, JIT_SCRATCH_C, JIT_SCRATCH_B));
                emit(e, jaiA64LdrByte(JIT_SCRATCH_A, JIT_SCRATCH_C, 0));
                /* 128 is an imm12, so the compare needs no register: a
                 * materialised constant on a body this hot is not free the way
                 * a register copy is. */
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, 128));
                branchOnDeopt(e, JAI_A64_HS);

                /* The shared one-byte string. jaiVMInit fills all 128 slots, so
                 * this is a load and not a load plus a null test -- see
                 * jaiAsciiCharsFill. The scaled add folds the shift in. */
                emitConst64(e, JIT_SCRATCH_C,
                            (int64_t)(uintptr_t)jaiAsciiCharTable());
                emit(e, jaiA64AddXLsl(JIT_SCRATCH_C, JIT_SCRATCH_C,
                                      JIT_SCRATCH_A, 3));
                emit(e, jaiA64LdrX(JIT_SCRATCH_C, JIT_SCRATCH_C, 0));

                /* Carry a sample so later instructions know this is a
                 * string: the receiver serves, since only its type is read.
                 * Without one the interned-equality path below cannot tell
                 * what it is holding and declines. */
                Value strSample = e->stackSeen[e->depth - 2];
                unsigned d1, d2;
                if (!popValue(e, &d1, NULL)) return false;
                if (!popValue(e, &d2, NULL)) return false;
                if (!pushValue3(e, SLOT_OBJ, 0, NULL, strSample, -1)) {
                    return false;
                }
                /* What the table holds is interned by construction -- see
                 * jaiStringChar -- so a consumer that would guard this for
                 * being a string, and for being interned, need not. */
                e->stackAscii[e->depth - 1] = true;
                emit(e, jaiA64MovX(pushReg(e) - 1, JIT_SCRATCH_C));
                off += 1;
                break;
            }
            /* `buf[i]` on a `bytes`: a length-guarded byte load, and the
             * result is a plain int, so nothing is allocated. Every binary
             * format in the language is read one byte at a time through this
             * -- the JPEG bit reader is a `bytes` index and nothing else --
             * and without it the whole function around one declined. */
            if (e->depth >= 2 && e->stack[e->depth - 1] == SLOT_INT &&
                e->stack[e->depth - 2] == SLOT_OBJ &&
                IS_BYTES(e->stackSeen[e->depth - 2])) {
                unsigned rIdx = pushReg(e) - 1;
                unsigned rBuf = valueXReg(e, e->valueDepth - 2);

                /* Really a bytes, and not something else this object slot
                 * happened to hold when the body was compiled. */
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, rBuf,
                                   (unsigned)offsetof(Obj, type)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_BYTES));
                branchOnDeopt(e, JAI_A64_NE);

                emit(e, jaiA64LdrW(JIT_SCRATCH_A, rBuf,
                                   (unsigned)offsetof(ObjBytes, length)));
                emitBoundsNormalise(e, rIdx, JIT_SCRATCH_A, JIT_SCRATCH_B,
                                    false);

                /* The payload is inline after the header, so the base needs no
                 * load of its own -- unlike a string, which holds a pointer. */
                emit(e, jaiA64AddX(JIT_SCRATCH_C, rBuf, JIT_SCRATCH_B));
                emit(e, jaiA64LdrByte(JIT_SCRATCH_A, JIT_SCRATCH_C,
                                      (unsigned)offsetof(ObjBytes, data)));

                unsigned dByte1, dByte2;
                if (!popValue(e, &dByte1, NULL)) return false;
                if (!popValue(e, &dByte2, NULL)) return false;
                if (!pushValue(e, SLOT_INT, 0, NULL)) return false;
                emit(e, jaiA64MovX(pushReg(e) - 1, JIT_SCRATCH_A));
                off += 1;
                break;
            }
            /* Index normalised as jaiNormalizeIndex does it, one unsigned compare covering both ends. Out of
             * range, or an element not the kind seen at compile time, goes back to the interpreter -- reading an element has no effect, so resuming at this instruction is always sound. */
            if (e->depth < 2) return subWhy(e, "the model is only %u deep", e->depth);
            if (e->stack[e->depth - 2] == SLOT_OBJ &&
                IS_DICT(e->stackSeen[e->depth - 2])) {
                /* `d[k]`, the read half of the OP_SET_INDEX dict arm below.
                 * Without it a loop that reads a dict ran interpreted end to
                 * end: `t += d["a"]` two million times was 16,280,472
                 * interpreted instructions and 1,684 once this landed.
                 *
                 * Predicted off a live sample and guarded, as the list arm is,
                 * except that the sample must be UNIFORM across the dict --
                 * see dictUniformValue for why a dict is not a list here. */
                unsigned dsidx = e->depth - 2;
                Value dsample;
                if (!dictUniformValue(AS_DICT(e->stackSeen[dsidx]), &dsample)) {
                    return subWhy(e, "the live dict is empty or holds more than "
                                     "one kind of value");
                }
                SlotKind dkind;
                unsigned dtag;
                ObjClass *dcls;
                uint32_t dshape;
                if (!exemplarKind(dsample, &dkind, &dtag, &dcls, &dshape)) {
                    return subWhy(e, "a dict value of a kind the tier cannot hold");
                }
                /* SLOT_OBJ pins nothing, so the container is proved to be a
                 * dict before anything is consumed: a miss resumes with the
                 * dict and the key both still on the interpreter's stack. */
                emit(e, jaiA64LdrW(JIT_SCRATCH_A,
                                   valueXReg(e, e->valueDepth - 2),
                                   (unsigned)offsetof(Obj, type)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_DICT));
                branchOnDeopt(e, JAI_A64_NE);

                if (!emitDescriptor(e, NULL_VAL, dsidx, 2,
                                    (void *)&jitGetIndexDict)) {
                    return false;
                }
                for (unsigned i = 0; i < 2; i++) {
                    unsigned drop;
                    if (!popValue(e, &drop, NULL)) return false;
                }
                /* The sample travels with the entry, as the list arm's does:
                 * without it `names["first"].len()` is an invoke on an object
                 * the model cannot name, and the body declines one instruction
                 * after the read it just learned to make. */
                if (!pushValue3(e, dkind, dshape, dcls, dsample, -1)) {
                    return false;
                }

                unsigned drat = e->descOffset +
                                (unsigned)offsetof(JitCallDesc, result);
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, 31, drat));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, dtag));
                /* Resumes AFTER the read. The lookup itself is pure, but it may
                 * have raised and been caught, and re-running it would be a
                 * second probe of a table the handler could have changed. */
                branchOnDeoptAt(e, JAI_A64_NE, (uint32_t)(off + 1), true);
                unsigned drd = pushReg(e) - 1;
                if (dkind == SLOT_BOOL) {
                    emit(e, jaiA64LdrByte(drd, 31, drat + 8));
                } else {
                    emit(e, jaiA64LdrX(drd, 31, drat + 8));
                }
                if (dkind == SLOT_INST) {
                    /* Two shapes in one dict cannot be told apart by the tag,
                     * and the walk above only sampled a prefix.
                     *
                     * The object type comes first, for the reason the shared
                     * return path gives: VAL_OBJ covers every heap object, and
                     * reading `klass` off a string lands in its length/hash and
                     * dereferences it. A dict holding a Box under one key and a
                     * str under another SEGFAULTED the VM from ordinary code --
                     * `d[k]` in any body hot enough to compile.
                     *
                     * The tag test above cannot stand in for this: it is the
                     * same test the sampled prefix already passed. */
                    emit(e, jaiA64LdrW(JIT_SCRATCH_A, drd,
                                       (unsigned)offsetof(Obj, type)));
                    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_INSTANCE));
                    branchOnDeoptAt(e, JAI_A64_NE, (uint32_t)(off + 1), true);
                    emit(e, jaiA64LdrX(JIT_SCRATCH_A, drd,
                                       (unsigned)offsetof(ObjInstance, klass)));
                    emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_A,
                                       (unsigned)offsetof(ObjClass, shapeId)));
                    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, dshape));
                    branchOnDeoptAt(e, JAI_A64_NE, (uint32_t)(off + 1), false);
                }
                e->wroteHeap = true;
                off += 1;
                break;
            }
            if (e->stack[e->depth - 2] != SLOT_LIST) {
                return subWhy(e, "the container has kind %s, not list",
                              slotKindName(e->stack[e->depth - 2]));
            }
            if (e->stack[e->depth - 1] != SLOT_INT) {
                return subWhy(e, "the subscript is kind %d, not an int",
                              (int)e->stack[e->depth - 1]);
            }
            unsigned rIdx = pushReg(e) - 1;
            unsigned rList = valueXReg(e, e->valueDepth - 2);
            bool gHoisted = false;

            Value seenList = e->stackSeen[e->depth - 2];
            SlotKind kind = SLOT_OPAQUE;
            unsigned tag = VAL_OBJ;
            ObjClass *elemClass = NULL;
            uint32_t  elemShape = 0;
            /* NULL_VAL on the declared route: there is no exemplar to carry,
             * which is the whole reason that route exists. */
            Value elem = NULL_VAL;
            /* No sample, but the list was DECLARED. See Emit::stackElemDecl:
             * for a body-local list filled through an alias there is nothing
             * to sample and never will be, so the declaration is the only
             * fact available -- and it is a fact, not a guess, because the
             * same byte pins ObjList::stg while the list is still empty.
             *
             * Safe even if it were wrong: listAccessFor rejects a kind the
             * pinned storage contradicts at compile time, and a dispatched
             * access still tag-checks the boxed arm at run time, so a bad
             * declaration deoptimises rather than misreading memory. */
            if (!IS_LIST(seenList) && elemDeclOn()) {
                SlotKind dk = SLOT_OPAQUE;
                switch ((unsigned)e->stackElemDecl[e->depth - 2]) {
                case FIELD_KIND_INT   + 1u: dk = SLOT_INT;   tag = VAL_INT;   break;
                case FIELD_KIND_FLOAT + 1u: dk = SLOT_FLOAT; tag = VAL_FLOAT; break;
                case FIELD_KIND_BOOL  + 1u: dk = SLOT_BOOL;  tag = VAL_BOOL;  break;
                default: break;
                }
                if (dk != SLOT_OPAQUE) {
                    kind = dk;
                    goto haveElemKind;
                }
            }
            if (!IS_LIST(seenList)) {
                return subWhy(e, "no live list to read an element kind off");
            }
            {
            ObjList *sl = AS_LIST(seenList);
            if (sl->count <= 0) {
                return subWhy(e, "the live list is empty, so there is no exemplar");
            }
            elem = jaiListGet(sl, 0);
            if (IS_INT(elem))        { kind = SLOT_INT;   tag = VAL_INT; }
            else if (IS_FLOAT(elem)) { kind = SLOT_FLOAT; tag = VAL_FLOAT; }
            else if (IS_BOOL(elem))  { kind = SLOT_BOOL;  tag = VAL_BOOL; }
            else if (IS_LIST(elem)) {
                /* A list of lists. `matrix_mul` is `b[k][j]` in its innermost
                 * loop and could not compile the outer half of it. */
                kind = SLOT_LIST;
                tag = VAL_OBJ;
            }
            else if (rawObjValue(elem)) {
                /* A list of strings, dicts, sets, or tuples, held raw: the same contract as a SLOT_OBJ
                 * global or field (sample specialises, the tag guard below confirms, every consumer
                 * re-checks Obj.type for itself). `str_search` builds text out of `chunks[seed %% 8]` and
                 * declined that whole loop forty times over before the string case alone was admitted;
                 * widened from IS_STRING to rawObjValue so every other raw-holdable element kind gets the
                 * same treatment rather than only strings. */
                kind = SLOT_OBJ;
                tag = VAL_OBJ;
            }
            else if (IS_INSTANCE(elem) && AS_INSTANCE(elem)->klass != NULL) {
                /* A list of instances, all of one shape -- which the per-read
                 * tag check cannot confirm on its own, so the class is checked
                 * too. A list holding two shapes deoptimises on the second. */
                kind = SLOT_INST;
                tag = VAL_OBJ;
                elemClass = AS_INSTANCE(elem)->klass;
                elemShape = elemClass->shapeId;
            } else return false;
            }
        haveElemKind:

            /* One `ldp` for both header fields: `items` at +16, `count`/`capacity` the adjacent int32s at +24, so
             * the pair's second half is `count | capacity << 32` and the bounds test reads it with uxtw -- one instruction per element read (life does nine per cell). */
            noteSlotIndexed(e, e->stackLocal[e->depth - 2]);
            {
                int32_t gOff = 0;
                uint8_t gBase = 0;
                bool gShaped = boundsCoveredAtHead(e, e->stackLocal[e->depth - 2],
                                                   e->valueDepth - 1, &gOff,
                                                   &gBase);
                noteIndexSpan(e, e->stackLocal[e->depth - 2], gShaped, gOff,
                              gBase);
                gHoisted = gShaped;
            }
            ListAccess gAcc = listAccessFor(e, rList,
                                            e->stackLocal[e->depth - 2],
                                            kind, JIT_SCRATCH_D);
            /* The sampled element and a PINNED storage cannot disagree -- an
             * I64 store holds ints and nothing else -- but the kind is what
             * the loads below are emitted for, so it is checked rather than
             * assumed. A dispatched access picks its second arm from the kind,
             * so it cannot disagree by construction. */
            if (!gAcc.dynamic && gAcc.stg != LIST_STORE_BOXED &&
                kind != listStgKind(gAcc.stg)) {
                return subWhy(e, "element kind %d is not storage %u's",
                              (int)kind, gAcc.stg);
            }
            unsigned gItems = JIT_SCRATCH_C, gCount = JIT_SCRATCH_A;
            int gh = hoistFor(e, e->stackLocal[e->depth - 2]);
            if (gh >= 0) {
                gItems = e->hoist[gh].itemsReg;
                gCount = e->hoist[gh].countReg;
            } else {
                emitListHeader(e, rList, gItems, gCount);
            }
            if (gHoisted) {
                /* The head proved it. Only the normalisation copy is left, and
                 * a shaped index is non-negative by that same proof, so even
                 * that is just a move. */
                emit(e, jaiA64MovX(JIT_SCRATCH_B, rIdx));
            } else {
                emitBoundsNormalise(e, rIdx, gCount, JIT_SCRATCH_B, true);
            }

            /* Both arms below leave JIT_SCRATCH_C on the PAYLOAD rather than
             * on the element, which is what lets one load serve them: a boxed
             * element's payload is eight bytes into it, an unboxed element IS
             * its payload. */
            int gSkip = listDispatchBegin(e, &gAcc, rList, JIT_SCRATCH_D);

            emit(e, jaiA64AddXLsl(JIT_SCRATCH_C, gItems,
                                  JIT_SCRATCH_B, listStgShift(gAcc.stg)));
            /* An unboxed element has no tag to check, and no object behind it
             * to confirm the type of: the storage already said what it is. */
            if (gAcc.stg == LIST_STORE_BOXED) {
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_C, 0));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, tag));
            branchOnDeopt(e, JAI_A64_NE);
            emit(e, jaiA64AddXImm(JIT_SCRATCH_C, JIT_SCRATCH_C, 8));
            if (kind == SLOT_INST) {
                /* The tag says "an object", not "an object of this class" --
                 * and not even "an instance" yet: VAL_OBJ is every heap
                 * object, so a list holding an instance beside a string must
                 * have its object type confirmed before `klass` is read,
                 * exactly as OP_FOR_ITER_BIND's SLOT_INST arms already do.
                 * Without this, a list whose sampled element is an instance
                 * but a later element is (say) a string reads that string's
                 * header bytes as an ObjInstance's `klass` pointer and
                 * segfaults dereferencing it -- not merely a wrong answer. */
                emit(e, jaiA64LdrX(JIT_SCRATCH_D, JIT_SCRATCH_C, 0));
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_D,
                                   (unsigned)offsetof(Obj, type)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_INSTANCE));
                branchOnDeopt(e, JAI_A64_NE);
                emit(e, jaiA64LdrX(JIT_SCRATCH_D, JIT_SCRATCH_D,
                                   (unsigned)offsetof(ObjInstance, klass)));
                emit(e, jaiA64LdrW(JIT_SCRATCH_D, JIT_SCRATCH_D,
                                   (unsigned)offsetof(ObjClass, shapeId)));
                emitConst64(e, JIT_SCRATCH_A, (int64_t)elemShape);
                emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_D, JIT_SCRATCH_A));
                branchOnDeopt(e, JAI_A64_NE);
            } else if (kind == SLOT_LIST) {
                /* "an object" is not "a list": every SLOT_LIST consumer reads the header with no check of
                 * its own, so the object type is confirmed here, once, before the kind is handed out --
                 * same contract, same check, as OP_GET_FIELD_LOCAL's SLOT_LIST arm. Without this a
                 * heterogeneous list (`[[1, 2], "not a list"]`) passes the generic VAL_OBJ tag check on
                 * either element and reads the second one's bytes through ObjList's field offsets. */
                emit(e, jaiA64LdrX(JIT_SCRATCH_D, JIT_SCRATCH_C, 0));
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_D,
                                   (unsigned)offsetof(Obj, type)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_LIST));
                branchOnDeopt(e, JAI_A64_NE);
            }
            }
            if (gSkip >= 0) {
                int gJoin = listDispatchElse(e, gSkip);
                emit(e, jaiA64AddXLsl(JIT_SCRATCH_C, gItems, JIT_SCRATCH_B,
                                      listStgShift(gAcc.alt)));
                listDispatchEnd(e, gJoin);
            }

            unsigned d1, d2;
            if (!popValue(e, &d1, NULL)) return false;
            if (!popValue(e, &d2, NULL)) return false;
            if (!pushValue3(e, kind, elemShape, elemClass, elem, -1)) return false;
            /* Bool payload is one byte (`BOOL_VAL` compiles to `strb`), so the other seven bytes are stale --
             * an 8-byte load would hand a SLOT_BOOL register (required to hold exactly 0 or 1, since every consumer does `cbnz` on the whole word) garbage. */
            if (kind == SLOT_BOOL) {
                emit(e, jaiA64LdrByte(pushReg(e) - 1, JIT_SCRATCH_C, 0));
            } else if (kind == SLOT_FLOAT &&
                       fpWorthLoading(e, code, off + 1, stop)) {
                /* Straight into the FP bank, for the same reason a float local
                 * goes there: `ldr x` followed by `fmov d, x` puts a
                 * cross-register-file move between the load and the multiply
                 * that wants it, and `ai[k] * b[k][j]` had two of them. */
                unsigned idx = e->valueDepth - 1;
                emit(e, jaiA64LdrD(fpRegAt(e, idx), JIT_SCRATCH_C, 0));
                fpClaim(e, idx);
            } else {
                emit(e, jaiA64LdrX(pushReg(e) - 1, JIT_SCRATCH_C, 0));
            }
            off += 1;
            break;
        }

        case OP_SET_INDEX: {
            /* Write half of OP_GET_INDEX, normalised the same way; every guard runs before the store, so a deopt
             * here still resumes at an instruction that hasn't happened yet. Sixteen refusals across the benchmarks came from its absence -- `queens` couldn't compile the function that does the work. */
            if (e->depth < 3) return false;
            if (e->stack[e->depth - 3] == SLOT_OBJ) {
                /* `d[k] = v`: a dict is as ordinary a container here as a list -- without this, dict_ops' loop just
                 * moved its decline from `get` to this store (a loop that declines anywhere runs interpreted end to end). Object type guarded before anything is consumed, so a miss resumes with container/key/value all still on the interpreter's stack. */
                unsigned sidx = e->depth - 3;
                if (!IS_DICT(e->stackSeen[sidx])) {
                    e->whyNot = "an index store into an object that is not a dict";
                    return false;
                }
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, valueXReg(e, e->valueDepth - 3),
                                   (unsigned)offsetof(Obj, type)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_DICT));
                branchOnDeopt(e, JAI_A64_NE);
                if (!emitDescriptor(e, NULL_VAL, sidx, 3,
                                    (void *)&jitSetIndexDict)) {
                    return false;
                }
                for (unsigned i = 0; i < 3; i++) {
                    unsigned r;
                    if (!popValue(e, &r, NULL)) return false;
                }
                e->wroteHeap = true;
                off += 1;
                break;
            }
            if (e->stack[e->depth - 3] != SLOT_LIST) return false;
            if (e->stack[e->depth - 2] != SLOT_INT) return false;
            SlotKind vk = e->stack[e->depth - 1];
            unsigned vtag = vk == SLOT_INT   ? VAL_INT
                          : vk == SLOT_FLOAT ? VAL_FLOAT
                          : vk == SLOT_BOOL  ? VAL_BOOL
                          : (vk == SLOT_INST || vk == SLOT_LIST ||
                             vk == SLOT_OBJ)  ? VAL_OBJ
                                              : 0xffffffffu;
            if (vtag == 0xffffffffu) return false;
            unsigned rVal = pushReg(e) - 1;
            unsigned rIdx = valueXReg(e, e->valueDepth - 2);
            unsigned rList = valueXReg(e, e->valueDepth - 3);

            noteSlotIndexed(e, e->stackLocal[e->depth - 3]);
            bool sHoisted;
            {
                int32_t sOff = 0;
                uint8_t sBase = 0;
                sHoisted = boundsCoveredAtHead(e, e->stackLocal[e->depth - 3],
                                               e->valueDepth - 2, &sOff,
                                               &sBase);
                noteIndexSpan(e, e->stackLocal[e->depth - 3], sHoisted, sOff,
                              sBase);
            }
            ListAccess sAcc = listAccessFor(e, rList, e->stackLocal[e->depth - 3],
                                            vk, JIT_SCRATCH_D);
            /* Exactly what jaiListStoreAccepts allows, and for its reason: an
             * int written into a `list[float]` de-specialises the list in the
             * interpreter, which is not something this can do inline. The
             * dispatched form cannot hit it -- its second arm is the storage
             * that holds a `vk` and no other. */
            if (!sAcc.dynamic && sAcc.stg != LIST_STORE_BOXED &&
                vk != listStgKind(sAcc.stg)) {
                return subWhy(e, "storing kind %d into storage %u",
                              (int)vk, sAcc.stg);
            }
            unsigned sItems = JIT_SCRATCH_C, sCount = JIT_SCRATCH_A;
            int sh = hoistFor(e, e->stackLocal[e->depth - 3]);
            if (sh >= 0) {
                sItems = e->hoist[sh].itemsReg;
                sCount = e->hoist[sh].countReg;
            } else {
                emitListHeader(e, rList, sItems, sCount);
            }
            if (sHoisted) {
                emit(e, jaiA64MovX(JIT_SCRATCH_B, rIdx));
            } else {
                emitBoundsNormalise(e, rIdx, sCount, JIT_SCRATCH_B, true);
            }

            int sSkip = listDispatchBegin(e, &sAcc, rList, JIT_SCRATCH_A);
            emitElemStoreAt(e, sAcc.stg, sItems, JIT_SCRATCH_B, vtag, rVal);
            if (sSkip >= 0) {
                int sJoin = listDispatchElse(e, sSkip);
                emitElemStoreAt(e, sAcc.alt, sItems, JIT_SCRATCH_B, vtag, rVal);
                listDispatchEnd(e, sJoin);
            }
            /* jaiListTouch: the count has not changed, so only the version
             * tells an iterator that the list moved under it. */
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, rList,
                               (unsigned)offsetof(ObjList, version)));
            emit(e, jaiA64AddXImm(JIT_SCRATCH_A, JIT_SCRATCH_A, 1));
            emit(e, jaiA64StrW(JIT_SCRATCH_A, rList,
                               (unsigned)offsetof(ObjList, version)));
            e->wroteHeap = true;

            unsigned d1, d2, d3;
            if (!popValue(e, &d1, NULL)) return false;
            if (!popValue(e, &d2, NULL)) return false;
            if (!popValue(e, &d3, NULL)) return false;
            off += 1;
            break;
        }

        case OP_GET_SLICE: {
            /* `xs[a:b]` out to the runtime: the clamp arithmetic has three
             * throwing exits and lives in one place, and the work itself is an
             * O(n) copy against which the descriptor's stores are noise. */
            unsigned flags = code[off + 1];
            unsigned nops  = ((flags & 1u) != 0) + ((flags & 2u) != 0) +
                             ((flags & 4u) != 0);
            unsigned nargs = 1u + nops;
            if (e->depth < nargs) return false;
            unsigned cidx = e->depth - nargs;
            Value cseen = e->stackSeen[cidx];
            /* A string slices as readily as a list -- same runtime call, same
             * "the guard pins the type so the result kind follows" argument --
             * and `s[a:b]` is what every hand-written scanner cuts tokens with.
             * Held as SLOT_OBJ, since that is what a string is here. */
            unsigned cType;
            SlotKind sliceKind;
            if (e->stack[cidx] == SLOT_LIST) {
                cType = OBJ_LIST; sliceKind = SLOT_LIST;
            } else if (e->stack[cidx] == SLOT_OBJ && IS_STRING(cseen)) {
                cType = OBJ_STRING; sliceKind = SLOT_OBJ;
            } else if (e->stack[cidx] == SLOT_OBJ && IS_TUPLE(cseen)) {
                /* `jitGetSlice` is a thin wrapper over `jaiSliceGet`, which
                 * already handles a tuple container exactly like a list or a
                 * string -- only this arm's own type guard was narrower than
                 * what the call it makes actually supports. */
                cType = OBJ_TUPLE; sliceKind = SLOT_OBJ;
            } else {
                e->whyNot = "slicing a container this tier does not model";
                return false;
            }

            /* Guard the container, not the result: with its object type pinned
             * the arm jaiSliceGet takes is settled, so the result's kind
             * follows. Before the descriptor and before any pop, so a miss
             * resumes here with everything still on the interpreter's stack. */
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, pushReg(e) - nargs,
                               (unsigned)offsetof(Obj, type)));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, cType));
            branchOnDeopt(e, JAI_A64_NE);

            emit(e, jaiA64MovzX(JIT_SCRATCH_A, flags, 0));
            emit(e, jaiA64StrX(JIT_SCRATCH_A, 31,
                               e->descOffset +
                                   (unsigned)offsetof(JitCallDesc, aux)));
            if (!emitDescriptorStatus(e, NULL_VAL, cidx, nargs,
                                      (void *)&jitGetSlice, false, -1)) {
                return false;
            }
            for (unsigned i = 0; i < nargs; i++) {
                unsigned r;
                if (!popValue(e, &r, NULL)) return false;
            }
            /* The container's own sample types the slice: a slice of a list of
             * ints is a list of ints, a slice of a string is a string, and
             * every element read re-checks its own tag, so this is a hint and
             * not an assumption. */
            if (!pushValue3(e, sliceKind, 0, NULL, cseen, -1)) return false;
            emit(e, jaiA64LdrX(pushReg(e) - 1, 31,
                               e->descOffset +
                                   (unsigned)offsetof(JitCallDesc, result) + 8));
            /* Deliberately not e->wroteHeap: the only effect is a fresh object
             * and an interpreted re-run would make another. Setting it would
             * decline the next self-call, which is the shape `sort` has. */
            off += 2;
            break;
        }

        case OP_GET_GLOBAL: {
            /* Two resolution paths with different obligations. BY VALUE (globalIsSelf/globalClass/globalFunction/
             * globalNative): resolved and baked at compile time, nothing rechecked at run time, so ObjModule::version must retire the whole form whenever such a binding could change (jaiModuleSet's jaiValueIsInertGlobal decides that) -- teaching any of the four a new value kind, or constant-folding a module int here, means updating jaiValueIsInertGlobal too. BY ADDRESS (the value case below): bakes the JaiEntry*, not the value, and re-loads it behind a tag guard (+ Obj.type/class-shape guards where needed) on EVERY access -- depends on JaiTable::keyVersion, not on ObjModule::version. */
            uint32_t nameIdx = jaiReadU24(code + off + 1);
            if (globalIsSelf(closure, nameIdx)) {
                if (!pushSelf(e)) return false;
                off += 6;
                break;
            }
            /* A class, resolved now and pinned by the module version check at
             * entry: rebinding the name retires the compiled form. It occupies
             * no register -- it is baked into the call sequence. */
            ObjClass *cls = globalClass(closure, nameIdx);
            if (cls != NULL) {
                if (e->depth >= JIT_MAX_STACK) return false;
                e->stackShape[e->depth] = cls->shapeId;
                e->stackClass[e->depth] = cls;
                e->stackSeen[e->depth]  = NULL_VAL;
                e->stackLocal[e->depth] = -1;
                e->stackAscii[e->depth] = false;
                e->stackNullLit[e->depth] = false;
                e->stackUnit[e->depth]  = false;
                e->stack[e->depth++]    = SLOT_CLASS;
                off += 6;
                break;
            }
            /* A plain function, resolved now and pinned by the same
             * module-version check. Only one that has itself compiled -- and
             * the reason is NOT soundness, which is what this comment used to
             * say and what two separate investigations went looking for.
             *
             * The miscompile it used to warn about is real and is FIXED. It was
             * never a property of the callee: a descriptor `result` holding a
             * bool was loaded eight bytes wide where BOOL_VAL writes one, so
             * `false` came back with dirty high bytes and read as true. That is
             * why the corruption was one-directional and looked deterministic
             * in a lexer -- delta-debugging 63 admitted callees reduced it to
             * `_is_alpha` alone. Both sites are closed: the ordinary return in
             * 5036099, the verdict-4 slow stub in emitSelfSlowStubs above.
             * docs/agents/uncompiled-callee.md has the reduction.
             *
             * With both fixed, dropping the check is SOUND -- 1274 green plain,
             * under JAITHON_NO_JIT, deopt stress and split stress -- and it
             * does let the mutually recursive groups in: json_parse's `value`,
             * `object`, `array` and `integer` all reach the tier.
             *
             * It is kept because removing it does not pay, which nobody had
             * measured. Net on `check --no-cache parser.jai`, stable across
             * three runs each: 284 function-tier bodies with the check, 275
             * without -- five gained, fourteen lost. An independent measurement
             * at a different commit found the same direction (265 to 258, seven
             * gained and thirteen lost). json_parse's own wall clock does not
             * move (0.05s either way) and neither does the compiler's.
             *
             * TWO things make it come out that way, and the second is the one
             * that is easy to miss. Declining here falls into emitUnarmedDeopt
             * just below, which interprets from this instruction ONWARD rather
             * than giving the whole body up; admitting the callee skips that
             * escape hatch and the body walks on to a refusal that costs all of
             * it -- `value` itself then stops at "callee's return kind not
             * usable". AND: admitting callees earlier changes how much
             * interpreted work happens before OTHER, unrelated functions cross
             * their own call-count hotness threshold in a fixed workload, so
             * some fall short of a threshold they used to clear and others
             * clear one they used to miss. The name sets differ in both
             * directions for that reason, and neither count is a general
             * verdict about the tier.
             *
             * So: removable, and not worth removing. If the unarmed-deopt path
             * ever stops being the better half of that trade, this is one line.
             */
            Value gv;
            ObjFunction *gfn = globalFunction(closure, nameIdx, &gv);
            if (gfn == NULL || gfn->jitFunc == NULL) {
                Value nv;
                ObjNative *nat = globalNative(closure, nameIdx, &nv);
                if (nat == NULL) {
                    /* `__prim__` and any other native namespace: not a native
                     * itself (globalNative already said so, correctly -- it
                     * is an ObjModule), and not found by globalSlot below
                     * either, because it lives in vm.builtins, not in this
                     * function's own module -- that lookup would always come
                     * back NULL for it, which is the whole reason every
                     * `__prim__.f64_*` body used to stop here. Pushed as
                     * SLOT_OBJ, register-resident, exactly the shape
                     * `math.sqrt`'s own receiver (an IMPORTED module) already
                     * reaches OP_INVOKE in -- except loaded as a bare
                     * constant rather than through a JaiEntry, since nothing
                     * re-binds the name `__prim__` itself the way an import
                     * can be reassigned. See emitModuleNativeCall for the
                     * guard that matters here: not this identity, but whether
                     * the MEMBER OP_INVOKE goes on to read has been rebound
                     * since compile. */
                    ObjModule *ns =
                        (jitModuleNativeCalls() &&
                         primInvokePairFits(e, &fn->chunk, (uint32_t)off))
                            ? globalNamespace(closure, nameIdx)
                            : NULL;
                    if (ns != NULL) {
                        if (e->depth >= JIT_MAX_STACK) return false;
                        unsigned dst = valueXReg(e, e->valueDepth);
                        if (!pushValue3(e, SLOT_OBJ, 0, NULL,
                                        OBJ_VAL((Obj *)ns), -1)) {
                            return false;
                        }
                        emitConst64(e, dst, (int64_t)(uintptr_t)ns);
                        off += 6;
                        break;
                    }
                    /* A global holding a plain value: storage is a JaiEntry whose address is fixed once it exists, so the
                     * load is one `ldr` behind two guards (table hasn't moved, value still has the compiled-for kind). Refusing this declined every loop reading a module-level variable -- most benchmarks, once they moved to module scope. */
                    Value gvv = NULL_VAL;
                    JaiEntry *gslot = globalSlot(e, closure, nameIdx, &gvv);
                    SlotKind gk = SLOT_OPAQUE;
                    uint32_t gshape = 0;
                    ObjClass *gcls = NULL;
                    if (gslot != NULL && globalKind(gvv, &gk, &gshape, &gcls)) {
                        /* The register the push BELOW will land in, named
                         * ahead of it because the guards have to run against
                         * the model as it is now. `pushReg` is one past the
                         * CURRENT top, which is the same register only while
                         * the bank is one contiguous run -- at a split
                         * boundary it is the last callee-saved one and the
                         * push goes to x0. That mismatch loaded a global into
                         * a register nothing then read, and bitops printed
                         * 68720029766 for 999625. */
                        unsigned dst = valueXReg(e, e->valueDepth);
                        unsigned tag = gk == SLOT_INT   ? VAL_INT
                                     : gk == SLOT_FLOAT ? VAL_FLOAT
                                     : gk == SLOT_BOOL  ? VAL_BOOL
                                                        : VAL_OBJ;
                        /* Every guard runs against the model as it is BEFORE
                         * the value is pushed, so a deopt here resumes at this
                         * instruction with the operand stack the interpreter
                         * expects. */
                        emitGlobalsGuard(e);
                        emitConst64(e, JIT_SCRATCH_D,
                                    (int64_t)(uintptr_t)gslot);
                        emit(e, jaiA64LdrW(JIT_SCRATCH_C, JIT_SCRATCH_D,
                                           (unsigned)offsetof(JaiEntry, value)));
                        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_C, tag));
                        branchOnDeopt(e, JAI_A64_NE);
                        if (gk == SLOT_BOOL) {
                            /* A byte, not a word. BOOL_VAL writes the union's
                             * one-byte `bool` member and leaves the other seven
                             * bytes of the payload indeterminate, so an 8-byte
                             * load brings back whatever the slot held before.
                             * Every consumer tests the whole register against
                             * zero, so those bytes decide the answer.
                             *
                             * This read a module-level `var` as FALSE once the
                             * body compiled: `std.fmt`'s `_colors_enabled` is
                             * true, and `green()` returned uncoloured text from
                             * the 65th call onward -- exactly
                             * JAI_JIT_THRESHOLD. A silent wrong answer, not a
                             * crash, and it had been in the tree long enough
                             * that a parallel test run turned it up by
                             * accident. emitCallOutResult's SLOT_BOOL arm has
                             * always split the two; this site never did. */
                            emit(e, jaiA64LdrByte(JIT_SCRATCH_C, JIT_SCRATCH_D,
                                                  (unsigned)offsetof(JaiEntry, value) + 8u));
                        } else {
                            emit(e, jaiA64LdrX(JIT_SCRATCH_C, JIT_SCRATCH_D,
                                               (unsigned)offsetof(JaiEntry, value) + 8u));
                        }
                        if (gk == SLOT_INST || gk == SLOT_LIST) {
                            /* VAL_OBJ is every heap object; the tag alone doesn't confirm the specific type, so it's checked before the class pointer is read. */
                            emit(e, jaiA64LdrByte(JIT_SCRATCH_B, JIT_SCRATCH_C,
                                                  (unsigned)offsetof(Obj, type)));
                            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_B,
                                                   gk == SLOT_INST ? OBJ_INSTANCE
                                                                   : OBJ_LIST));
                            branchOnDeopt(e, JAI_A64_NE);
                        }
                        if (gk == SLOT_INST) {
                            emit(e, jaiA64LdrX(JIT_SCRATCH_B, JIT_SCRATCH_C,
                                               (unsigned)offsetof(ObjInstance, klass)));
                            emit(e, jaiA64LdrW(JIT_SCRATCH_B, JIT_SCRATCH_B,
                                               (unsigned)offsetof(ObjClass, shapeId)));
                            emitConst64(e, JIT_SCRATCH_A, (int64_t)gshape);
                            emit(e, jaiA64SubsX(31, JIT_SCRATCH_B, JIT_SCRATCH_A));
                            branchOnDeopt(e, JAI_A64_NE);
                        }
                        if (!pushValue3(e, gk, gshape, gcls, gvv, -1)) return false;
                        emit(e, jaiA64MovX(dst, JIT_SCRATCH_C));
                        off += 6;
                        break;
                    }
                    /* Distinguish the two, because the census reads these and
                     * "a global of a kind the tier has no slot for" is a very
                     * different backlog item from an uncompiled callee. */
                    /* Interpreted from here rather than the whole function
                     * declined. The case that matters is a global read on a
                     * branch that never runs: `throw ValueError(...)` in an
                     * argument check is the shape, there are hundreds of them
                     * in lib/std alone, and declining for one cost 110 ms
                     * against 14 ms for the same function guarded by a plain
                     * `return`. The unarmed path already exists for exactly
                     * this -- see emitUnarmedDeopt. */
                    if (!e->osr && emitUnarmedDeopt(e, &fn->chunk, &off, stop)) {
                        break;
                    }
                    if (e->failed) return false;
                    /* Named, because this is the top state-level refusal by
                     * attempt count -- 406 in one file, 82 of them retries of
                     * a single loop -- and "callee is not a compiled global
                     * function" gave the reader nothing to act on. Which
                     * callee it is decides whether this is a tier-ordering
                     * problem that resolves itself or a function that never
                     * compiles at all. */
                    Value gNameVal = nameIdx < (uint32_t)fn->chunk.constants.count
                                         ? fn->chunk.constants.data[nameIdx]
                                         : NULL_VAL;
                    const char *gName = IS_STRING(gNameVal)
                                            ? AS_STRING(gNameVal)->chars
                                            : "?";
                    if (gslot != NULL && !IS_CLOSURE(gvv) && !IS_CLASS(gvv) &&
                        !IS_NATIVE(gvv)) {
                        return subWhy(e, "`%s` is a global of a kind the tier "
                                      "has no slot for", gName);
                    }
                    return subWhy(e, "`%s` is not a compiled global function",
                                  gName);
                }
                if (e->depth >= JIT_MAX_STACK) return false;
                e->stackShape[e->depth] = 0;
                e->stackClass[e->depth] = (ObjClass *)(void *)AS_OBJ(nv);
                e->stackSeen[e->depth]  = nv;
                e->stackLocal[e->depth] = -1;
                e->stackAscii[e->depth] = false;
                e->stackNullLit[e->depth] = false;
                e->stackUnit[e->depth]  = false;
                e->stack[e->depth++]    = SLOT_NATIVE;
                off += 6;
                break;
            }
            if (e->depth >= JIT_MAX_STACK) return false;
            e->stackShape[e->depth] = 0;
            /* The stub that writes a deopt record materialises a callee from
             * here, so it has to be the object and not NULL. */
            e->stackClass[e->depth] = (ObjClass *)(void *)AS_OBJ(gv);
            e->stackSeen[e->depth]  = gv;
            e->stackLocal[e->depth] = -1;
            e->stackAscii[e->depth] = false;
            e->stackNullLit[e->depth] = false;
            e->stackUnit[e->depth]  = false;
            e->stack[e->depth++]    = SLOT_FUNC;
            off += 6;
            break;
        }

        case OP_SET_GLOBAL: {
            /* Assigns without popping, exactly as the interpreter does: the
             * value is the statement's result and an OP_POP follows. */
            uint32_t nameIdx = jaiReadU24(code + off + 1);
            Value gvv;
            JaiEntry *gslot = globalSlot(e, closure, nameIdx, &gvv);
            if (gslot == NULL) {
                /* The interpreter raises NameError for a name with no binding;
                 * there is nothing to store into and nothing to compile. */
                e->whyNot = "a global with no storage to store into";
                return false;
            }
            if (e->depth == 0 || !holdsRegister(e->stack[e->depth - 1])) {
                e->whyNot = "nothing on the stack to store into a global";
                return false;
            }
            SlotKind sk = e->stack[e->depth - 1];
            if (sk != SLOT_INT && sk != SLOT_FLOAT && sk != SLOT_BOOL &&
                sk != SLOT_LIST && sk != SLOT_INST && sk != SLOT_MAYBE_INST) {
                /* SLOT_OBJ deliberately excluded: it covers a closure too, and storing a closure into a global
                 * rebinds a callee some compiled form may have already baked BY VALUE -- ObjModule::version only retires that form at its next ENTRY, not mid-body, so this would be a silently wrong answer, not a decline. Every remaining kind is inert per jaiValueIsInertGlobal, which is what makes the bump rule below sound. */
                e->whyNot = "a global store of a kind that has no Value form";
                return false;
            }
            emitGlobalsGuard(e);
            {
                unsigned src = pushReg(e) - 1;
                emitConst64(e, JIT_SCRATCH_D, (int64_t)(uintptr_t)gslot);
                /* Version only needs to move when a class/closure/native leaves or arrives (jaiValueIsInertGlobal is
                 * the authority). The incoming value is already provably inert (SLOT_OBJ refused above), so only what's overwritten matters -- this checks a STRICT SUBSET of the inert types (non-object, ObjInstance, ObjList); anything else conservatively bumps. Widening jaiValueIsInertGlobal needs no change here; narrowing it (removing OBJ_INSTANCE/OBJ_LIST) does. */
                unsigned skip[3];
                unsigned nskip = 0;
                emit(e, jaiA64LdrW(JIT_SCRATCH_B, JIT_SCRATCH_D,
                                   (unsigned)offsetof(JaiEntry, value)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_B, VAL_OBJ));
                skip[nskip++] = e->count;
                emit(e, jaiA64BCond(JAI_A64_NE, 0));
                emit(e, jaiA64LdrX(JIT_SCRATCH_B, JIT_SCRATCH_D,
                                   (unsigned)offsetof(JaiEntry, value) + 8u));
                emit(e, jaiA64LdrByte(JIT_SCRATCH_C, JIT_SCRATCH_B,
                                      (unsigned)offsetof(Obj, type)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_C, OBJ_INSTANCE));
                skip[nskip++] = e->count;
                emit(e, jaiA64BCond(JAI_A64_EQ, 0));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_C, OBJ_LIST));
                skip[nskip++] = e->count;
                emit(e, jaiA64BCond(JAI_A64_EQ, 0));
                emitVersionBump(e, closure->fn->module);
                for (unsigned si = 0; si < nskip; si++) {
                    unsigned at = skip[si];
                    if (at < JIT_MAX_INSTS && e->count > at) {
                        e->code[at] = jaiA64BCond(
                            si == 0 ? JAI_A64_NE : JAI_A64_EQ,
                            (int32_t)(e->count - at));
                    }
                }
                /* No write barrier: mark-sweep traces module globals as a root every collection, so a raw store is
                 * enough. Nothing between the tag and payload writes can allocate, so no collection can ever see them disagree. */
                emitTagFor(e, sk, src, JIT_SCRATCH_B, JIT_SCRATCH_A);
                emit(e, jaiA64StrW(JIT_SCRATCH_B, JIT_SCRATCH_D,
                                   (unsigned)offsetof(JaiEntry, value)));
                emit(e, jaiA64StrX(src, JIT_SCRATCH_D,
                                   (unsigned)offsetof(JaiEntry, value) + 8u));
            }
            /* Re-running this call interpreted would apply the store twice, so
             * from here a bail is no longer free. */
            e->wroteHeap = true;
            off += 6;
            break;
        }

        case OP_CALL: {
            unsigned argc = code[off + 1];

            if (isClassCallee(e, argc)) {
                if (!emitCallOut(e, argc)) return false;
                off += 2;
                break;
            }
            if (e->depth >= argc + 1u &&
                e->stack[e->depth - argc - 1] == SLOT_FUNC) {
                /* Cheapest first: a body small enough to stand where the call
                 * is costs neither the frame nor the argument shuffle. */
                Value cvv = e->stackSeen[e->depth - argc - 1];
                if (IS_CLOSURE(cvv) &&
                    inlineGlobalCall(e, fn, AS_CLOSURE(cvv), argc,
                                     (uint32_t)off, -1)) {
                    off += 2;
                    break;
                }
                if (e->failed) return false;
                if (!emitGlobalCall(e, fn, argc, (uint32_t)off,
                                    (uint32_t)(off + 2))) return false;
                off += 2;
                break;
            }
            if (e->depth >= argc + 1u &&
                e->stack[e->depth - argc - 1] == SLOT_NATIVE) {
                /* `float(i)`/`int(x)` are one instruction each, so they're emitted rather than called -- what
                 * `spectral`/`matrix_mul` have in their inner loops, and the whole body was declined for want of them. Every other builtin still declines: a call needs a result kind known without running anything, and only these two have one. */
                if (argc != 1 && argc != 2) {
                    e->whyNot = "builtin arity"; return false;
                }
                Value cv = e->stackSeen[e->depth - argc - 1];
                ObjNative *nat = IS_NATIVE(cv) ? AS_NATIVE(cv) : NULL;
                const char *nm = nat != NULL && nat->name != NULL
                                     ? nat->name->chars : "";
                /* `min`/`max` of two ints are a compare and a csel, and image
                 * code reaches for them per pixel -- the JPEG block reader
                 * calls `min` twice a sample, so declining them left the whole
                 * body interpreted. Only the two-argument integer form is
                 * emitted: the float form throws on a NaN operand (see
                 * compareDoubles), which a bare fcsel would silently swallow,
                 * and the one-argument form takes an iterable. */
                bool isMin = strcmp(nm, "min") == 0;
                bool isMax = strcmp(nm, "max") == 0;
                if (argc == 2) {
                    if (!isMin && !isMax) {
                        int done = emitNativeResultCall(e, cv, nm, argc,
                                                        (uint32_t)(off + 2));
                        if (done < 0) return false;
                        if (done > 0) { off += 2; break; }
                        return subWhy(e, "`%s` has no known result kind", nm);
                    }
                    if (e->stack[e->depth - 1] != SLOT_INT ||
                        e->stack[e->depth - 2] != SLOT_INT) {
                        e->whyNot = "a builtin with no known result kind";
                        return false;
                    }
                    unsigned rb, ra2;
                    if (!popValue(e, &rb, NULL)) return false;
                    if (!popValue(e, &ra2, NULL)) return false;
                    if (!pushValue(e, SLOT_INT, 0, NULL)) return false;
                    unsigned rd = pushReg(e) - 1;
                    /* `min(a, b)` keeps `a` unless `b` is STRICTLY smaller
                     * (extremum() only replaces on a nonzero order), so the
                     * compare is of b against a and the condition is strict. */
                    emit(e, jaiA64SubsX(JAI_A64_XZR, rb, ra2));
                    emit(e, jaiA64CselX(rd, rb, ra2,
                                        isMax ? JAI_A64_GT : JAI_A64_LT));
                    dropCalleeEntry(e);
                    off += 2;
                    break;
                }
                SlotKind ak = e->stack[e->depth - 1];
                unsigned ar = pushReg(e) - 1;
                if (strcmp(nm, "abs") == 0 && ak == SLOT_INT) {
                    /* `subs` off zero both negates and reports the one input
                     * abs() rejects: only INT64_MIN overflows the negation, and
                     * the interpreter raises the OverflowError on re-entry. */
                    emit(e, jaiA64SubsX(JIT_SCRATCH_A, JAI_A64_XZR, ar));
                    branchOnDeoptInstStart(e, JAI_A64_VS);
                    emit(e, jaiA64CselX(ar, ar, JIT_SCRATCH_A, JAI_A64_MI));
                    dropCalleeEntry(e);
                    off += 2;
                    break;
                }
                /* `ord(c)` on a one-byte ASCII string is that byte, which is
                 * the inverse of the string-index arm above and restricted the
                 * same way: a multi-byte scalar needs decoding, and an empty
                 * or longer string is an error nOrd raises. All three deopt to
                 * the interpreter, which finishes the call properly.
                 *
                 * Worth an arm because it is the ONLY builtin json_parse
                 * declines on -- 141 of its 73 declined sites name it, and
                 * every one takes the whole enclosing loop down with it. A
                 * scanner reads its input a character at a time and asks what
                 * that character is; that is the shape. */
                if (strcmp(nm, "ord") == 0 && ak == SLOT_OBJ &&
                    IS_STRING(e->stackSeen[e->depth - 1])) {
                    emit(e, jaiA64LdrW(JIT_SCRATCH_A, ar,
                                       (unsigned)offsetof(Obj, type)));
                    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_STRING));
                    branchOnDeoptInstStart(e, JAI_A64_NE);
                    emit(e, jaiA64LdrW(JIT_SCRATCH_A, ar,
                                       (unsigned)offsetof(ObjString, length)));
                    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, 1));
                    branchOnDeoptInstStart(e, JAI_A64_NE);
                    emit(e, jaiA64LdrX(JIT_SCRATCH_A, ar,
                                       (unsigned)offsetof(ObjString, chars)));
                    emit(e, jaiA64LdrByte(JIT_SCRATCH_A, JIT_SCRATCH_A, 0));
                    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, 128));
                    branchOnDeoptInstStart(e, JAI_A64_HS);
                    /* The argument's register is not written until every guard
                     * has passed, so a bail still reconstructs the call with
                     * the string the program passed. */
                    emit(e, jaiA64MovX(ar, JIT_SCRATCH_A));
                    e->stack[e->depth - 1] = SLOT_INT;
                    /* The entry keeps its register and changes kind, so the
                     * claims made about the STRING that was in it go with it. */
                    e->stackObjType[e->depth - 1] = 0;
                    e->stackElem[e->depth - 1] = NULL_VAL;
                    dropCalleeEntry(e);
                    off += 2;
                    break;
                }
                bool toFloat = strcmp(nm, "float") == 0;
                bool toInt   = strcmp(nm, "int") == 0;
                if (toFloat && ak == SLOT_INT) {
                    /* Straight into the bank, not out through X: what consumes a `float(i)` is a float
                     * operator, and one reads the bank through fpOperand. Computing into d0 and storing the bits to X cost spectral's inner loop two cross-register-file fmovs per iteration -- the store, and the load the very next instruction made of it. */
                    unsigned fidx = e->valueDepth - 1;
                    if (!e->fpOff) {
                        emit(e, jaiA64ScvtfDX(fpRegAt(e, fidx), ar));
                        fpClaim(e, fidx);
                    } else {
                        emit(e, jaiA64ScvtfDX(JIT_FSCRATCH_A, ar));
                        emit(e, jaiA64FmovXD(ar, JIT_FSCRATCH_A));
                    }
                } else if (toInt && ak == SLOT_FLOAT) {
                    emit(e, jaiA64FcvtzsXD(ar, fpOperand(e, e->valueDepth - 1)));
                } else if (!((toFloat && ak == SLOT_FLOAT) ||
                             (toInt && ak == SLOT_INT))) {
                    int done = emitNativeResultCall(e, cv, nm, argc,
                                                    (uint32_t)(off + 2));
                    if (done < 0) return false;
                    if (done > 0) { off += 2; break; }
                    /* Names it: the table in kNativeResults is what would have
                     * to grow, and the reason alone never said which row. */
                    return subWhy(e, "`%s` has no known result kind for an "
                                     "argument of kind %s", nm,
                                  slotKindName(ak));
                }
                /* The result stays in the argument's register, so the claims
                 * made about what WAS in it go with it. */
                e->stack[e->depth - 1] = toFloat ? SLOT_FLOAT : SLOT_INT;
                e->stackObjType[e->depth - 1] = 0;
                e->stackElem[e->depth - 1] = NULL_VAL;
                dropCalleeEntry(e);
                off += 2;
                break;
            }

            if (e->depth >= argc + 1u &&
                e->stack[e->depth - argc - 1] == SLOT_OBJ) {
                /* Closure held in a local (`apply_n(f, ..)` doing `acc = f(acc)`, the shape most library code takes).
                 * Guard is on the FUNCTION, not the closure: `closure_calls` builds a fresh closure over a different `step` every outer iteration, so guarding the closure itself would deoptimise every time -- all share one ObjFunction (monomorphic) and differ only by upvalue. The callee Value still comes from the register, which is what keeps the upvalues right. */
                unsigned cidx = e->depth - argc - 1;
                Value cv = e->stackSeen[cidx];
                if (!IS_CLOSURE(cv)) {
                    e->whyNot = "an indirect callee that is not a closure";
                    return false;
                }
                ObjFunction *cfn = AS_CLOSURE(cv)->fn;
                unsigned rCallee0 =
                    valueBankReg(e, cidx - (e->depth - e->valueDepth));

                /* The guard comes first now, because what follows it is a
                 * choice between two ways of making the call and both need it:
                 * once this register is known to name `cfn`, the body behind
                 * it is known too, and that is the whole licence to inline. */
                emit(e, jaiA64LdrX(JIT_SCRATCH_A, rCallee0,
                                   (unsigned)offsetof(ObjClosure, fn)));
                emitConst64(e, JIT_SCRATCH_B, (int64_t)(uintptr_t)cfn);
                emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
                branchOnDeopt(e, JAI_A64_NE);

                /* Cheapest first, as the direct call does it: a body small enough to stand where the call is costs
                 * neither frame, argument shuffle, nor root fill. An indirect site needs NONE of the checks below to inline -- no argument kind has to match a specialisation, and the callee need not have compiled at all, since arguments stay in the caller's own registers with the caller's own kinds. */
                if (inlineGlobalCall(e, fn, AS_CLOSURE(cv), argc,
                                     (uint32_t)off, (int)rCallee0)) {
                    off += 2;
                    break;
                }
                if (e->failed) return false;

                /* The same two things the direct global call checks, and for
                 * the same reasons: a raw payload is only sound if the callee
                 * was specialised to that kind and shape, and the caller's
                 * module-version guard only stands in for the callee's entry
                 * check if the two were compiled against the same version. */
                if (!directCallArgsMatch(e, cfn, cidx + 1u, argc)) return false;
                if (fn->module == NULL ||
                    cfn->jitFuncModuleVersion != fn->module->version) {
                    e->whyNot = "an indirect callee compiled against an older module";
                    return false;
                }
                if (cfn->jitFunc == NULL || cfn->arity != argc) {
                    e->whyNot = "the closure this calls has not compiled";
                    return false;
                }
                SlotKind rkind = (SlotKind)cfn->jitReturnKind;
                if (rkind != SLOT_INT && rkind != SLOT_FLOAT &&
                    rkind != SLOT_BOOL) {
                    e->whyNot = "an indirect call whose result kind is not scalar";
                    return false;
                }
                unsigned rCallee = rCallee0;   /* guarded above */

                /* Straight to the callee's compiled entry, not through jaiCallValue/an interpreter frame -- same
                 * convention a self-call and jaiJitEnterFunc use. Callee must live in this module since the caller's module-version guard stands in for the callee's own entry check. The baked jitFunc address can't go stale: the arena is never freed and jitFunc is written once; only the ObjFunction identity needs guarding (done above). */
                unsigned calleeArgs = (unsigned)cfn->jitArgCount;
                bool wantsClosure = calleeArgs == argc + 1u;
                if (cfn->module != fn->module || cfn->jitArgBase != 1u ||
                    (!wantsClosure && calleeArgs != argc)) {
                    e->whyNot = "an indirect callee this tier cannot enter directly";
                    return false;
                }
                if (calleeArgs > JIT_MAX_ARITY) {
                    e->whyNot = "an indirect callee with too many arguments";
                    return false;
                }

                /* Roots before the branch: a `blr` pushes none, and the callee
                 * may allocate. */
                unsigned callRoots = 0;
                if (!emitRootFill(e, e->descOffset, &callRoots)) return false;
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

                unsigned firstArg = cidx + 1u - (e->depth - e->valueDepth);
                for (unsigned i = 0; i < argc; i++) {
                    emit(e, jaiA64MovX(i, valueBankReg(e, firstArg + i)));
                }
                if (wantsClosure) emit(e, jaiA64MovX(argc, rCallee));
                emitConst64(e, JIT_SCRATCH_D,
                            (int64_t)(uintptr_t)cfn->jitFunc);
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
                /* x1 carries the callee's verdict. It used to BAIL here,
                 * which is sound only where partial execution is invisible --
                 * true of the function tier, whose locals are registers and
                 * whose frame is entered fresh, and false of the OSR tier,
                 * whose locals ARE the interpreter's frame slots and whose
                 * every way out syncs them back. An OSR bail mid-iteration
                 * therefore kept the loop variable already advanced and the
                 * locals already written, and the interpreter resumed at the
                 * LOOP HEAD -- which advanced the iterator again and dropped
                 * the rest of that iteration's work.
                 *
                 * That is the whole of the "closure call loses one count per
                 * collection" bug: `for v in xs { count += rule(v) }` came
                 * back short by one per bail, and under --gc-stress the callee
                 * deopts often enough to bail regularly. Measured on
                 * docs/repro/osr-closure-loses-counts.jai at --gc-stress=200:
                 * 170 of 77027 lost before, 0 after.
                 *
                 * A deopt at the CALL OFFSET instead, which is exactly what
                 * emitDirectCall does at the same point and for the same
                 * reason: the record is taken with the callee and its
                 * arguments still on the model's stack, so the interpreter
                 * re-runs the call from the top. Re-running is sound because
                 * the callee is a compiled body that bailed, and the function
                 * tier declines any body whose bail can follow a heap write.
                 *
                 * It costs nothing measurable -- two instructions on the path
                 * and a deopt record. Alternating binaries under
                 * scripts/gpu_lock.sh, best of five each: closure_calls
                 * 123/123, poly_dispatch 138/138, json_parse 110/110,
                 * object_dispatch 151/150, sort_merge 353/358. */
                emit(e, jaiA64SubsXImm(31, 1, 2));
                if (e->fixupCount >= JIT_MAX_FIXUPS) { e->failed = true; return false; }
                e->fixups[e->fixupCount].instIndex    = (int)e->count;
                e->fixups[e->fixupCount].targetOffset = FIXUP_THREW;
                e->fixups[e->fixupCount].conditional  = true;
                e->fixups[e->fixupCount].depth        = -1;
                e->fixupCount++;
                emit(e, jaiA64BCond(JAI_A64_EQ, 0));
                emit(e, jaiA64SubsXImm(31, 1, 0));
                branchOnDeoptAt(e, JAI_A64_NE, (uint32_t)off, false);

                for (unsigned i = 0; i <= argc; i++) {
                    unsigned r;
                    if (!popValue(e, &r, NULL)) return false;
                }
                if (!pushValue(e, rkind, 0, NULL)) return false;
                emit(e, jaiA64MovX(pushReg(e) - 1, 0));
                off += 2;
                break;
            }

            if (argc != e->arity) return false;
            /* Recorded, not rejected, here: the measuring pass always runs with slot 0 available, so testing the
             * base in THIS pass silently aborted every recursive function's compile (fib_recursive: 8.8ms back to 83ms). The decision belongs after the base is actually chosen. */
            e->hasSelfCall = true;
            if (e->depth < argc + 1) return false;
            if (e->stack[e->depth - argc - 1] != SLOT_SELF) return false;

            /* A self-call branches past the entry guard, so nothing else checks what it hands over -- passing a
             * maybe-instance into a slot typed as a plain instance would let a later field read dereference zero. Recorded per-slot so the retry seeds only that slot as nullable, costing nothing for parameters that are never null. */
            {
                bool retry = false;
                for (unsigned i = 0; i < argc; i++) {
                    unsigned pslot = 1u + i;   /* parameters are slots 1..arity */
                    unsigned aidx  = e->depth - argc + i;
                    SlotKind ak = e->stack[aidx];
                    SlotKind pk = e->localKind[pslot];
                    if (ak == SLOT_MAYBE_INST && pk == SLOT_INST) {
                        e->needNullable[pslot] = true;
                        retry = true;
                        continue;
                    }
                    if (ak != pk) {
                        e->whyNot = "a self-call argument is not the parameter's kind";
                        return false;
                    }
                    if ((pk == SLOT_INST || pk == SLOT_MAYBE_INST) &&
                        e->stackClass[aidx] != e->localClass[pslot]) {
                        e->whyNot = "a self-call passing a different class";
                        return false;
                    }
                }
                if (retry) {
                    e->whyNot = "a parameter is sometimes null";
                    return false;
                }
            }

            /* A `bl` pushes no roots, so anything this body holds in a
             * callee-saved register is invisible to a collection inside the
             * callee. Link the descriptor onto the collector's frame chain
             * around the call instead. Costs six instructions and only when
             * there is something to root. */
            unsigned selfRoots = 0;
            if (!emitRootFill(e, e->descOffset, &selfRoots)) return false;
            if (selfRoots > 0) {
                unsigned d = e->descOffset;
                emit(e, jaiA64MovzX(JIT_SCRATCH_A, selfRoots, 0));
                emit(e, jaiA64StrX(JIT_SCRATCH_A, 31,
                                   d + (unsigned)offsetof(JitCallDesc, nroots)));
                emitConst64(e, JIT_SCRATCH_A, (int64_t)(uintptr_t)&gJitFrames);
                emit(e, jaiA64LdrX(JIT_SCRATCH_B, JIT_SCRATCH_A, 0));
                emit(e, jaiA64AddXImm(JIT_SCRATCH_C, 31, d));
                emit(e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_C,
                                   (unsigned)offsetof(JitCallDesc, link)));
                emit(e, jaiA64StrX(JIT_SCRATCH_C, JIT_SCRATCH_A, 0));
            }

            /* The arguments sit in the top `argc` value registers, in order.
             * They move to x0.. which nothing else is using. */
            unsigned first = e->valueDepth - argc;
            for (unsigned i = 0; i < argc; i++) {
                emit(e, jaiA64MovX(i, valueBankReg(e, first + i)));
            }

            /* To instruction 0, the prologue -- NOT to the first instruction
             * of the body. A recursive call that skipped the prologue would
             * not save x19 upward, so the callee would overwrite the caller's
             * locals and the recursion would never terminate. */
            if (!e->sawReturn) e->assumedIntReturn = true;

            noteScratchClobber(e);
            branchTo(e, FIXUP_ENTRY, false, 0);
            e->code[e->count - 1] = jaiA64Bl(0);
            /* Unlink before anything can leave: the bail branch below and every
             * later exit go through the epilogue, and a frame still on the
             * chain then points at a stack slot that no longer exists. x0 and
             * x1 carry the callee's answer, so only the scratches are free. */
            if (selfRoots > 0) {
                unsigned d = e->descOffset;
                emitConst64(e, JIT_SCRATCH_A, (int64_t)(uintptr_t)&gJitFrames);
                emit(e, jaiA64AddXImm(JIT_SCRATCH_C, 31, d));
                emit(e, jaiA64LdrX(JIT_SCRATCH_B, JIT_SCRATCH_C,
                                   (unsigned)offsetof(JitCallDesc, link)));
                emit(e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_A, 0));
            }
            /* x1 is the callee's verdict, each needing a different response here (this used to be one `bail`,
             * which is why queens' `place` never compiled -- a bail re-runs the whole caller, unsound above `cols[row] = col`):
             *   0  value is in x0. Costs one compare, one not-taken branch.
             *   2  callee raised; the interpreter owns it, leave.
             *   1  callee bailed (only possible having written nothing) -- NOT a bail here, but a deopt at
             *      this instruction, re-executing just the call with the caller's own earlier writes intact.
             *   4  callee deoptimised part-way and may have written -- can't be re-executed or recorded over
             *      (gDeopt is one global), so it's FINISHED in the interpreter from its own record, value handed back here. One record suffices at any recursion depth since it's consumed at the innermost frame that sees it.
             * A kind the body wasn't compiled for lands in the descriptor's result slot with its real tag,
             * written out by the existing `lastFromDesc` deopt. */
            /* The fast path is what a recursive body pays per call: one
             * compare and one branch that is not taken. Every other answer is
             * a jump to a block emitted with the stubs -- inline, the two call
             * sites in `fib` had thirty instructions of cold code between them
             * and the benchmark lost 25%.
             *
             * The records have to be taken HERE even though the code is
             * emitted later, because they are of the model as it stands at
             * this instruction, and the model has moved on by then. */
            if (e->selfSlowCount >= JIT_MAX_SELF_SLOW) {
                e->whyNot = "more self-calls than the tier tracks";
                return false;
            }
            /* Its cold block routes a raised exception to the exception exit
             * (see emitSelfSlowStubs). */
            if (!raiseExitAllowed(e, "a call that can raise inside a try")) {
                return false;
            }
            unsigned si = e->selfSlowCount++;
            unsigned resultReg = valueXReg(e, e->valueDepth - argc);
            e->selfSlow[si].roots     = selfRoots;
            e->selfSlow[si].resultReg = resultReg;
            e->selfSlow[si].stub      = -1;
            /* This body is its own callee, but its own returns are not the
             * only way the deopt continuation can answer: `emitUnarmedDeopt`
             * can skip a run of instructions whose own `return`s never reach
             * `mergeReturnKind`'s walk, so `e->returnKind` is not proven to
             * bound every value this path can actually hand back. VAL_OBJ is
             * every heap object, so trusting a SLOT_LIST/SLOT_INST tag alone
             * risks the same wrong-object-shape read `emitDirectCall`'s
             * sibling stub (just above) already guards against; -1 is
             * "no type check", used for every other kind, and said explicitly
             * rather than left to the zeroed struct, where 0 is OBJ_STRING. */
            e->selfSlow[si].callee    = NULL;
            e->selfSlow[si].retType   = e->returnKind == SLOT_INST ? (int)OBJ_INSTANCE
                                       : e->returnKind == SLOT_LIST ? (int)OBJ_LIST
                                                                     : -1;
            e->selfSlow[si].retShape  = e->returnKind == SLOT_INST ? e->returnShape : 0;
            if (!deoptRecordAt(e, (uint32_t)off, false,
                               &e->selfSlow[si].deoptBail)) {
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
            emit(e, jaiA64MovX(resultReg, 0));
            e->selfSlow[si].returnTo = (int)e->count;

            /* The call has happened, so the model moves on before the second
             * record: an unexpected result kind resumes AFTER the call,
             * holding what the descriptor carries. */
            e->depth      -= argc + 1;
            e->valueDepth -= argc;
            if (!pushValue(e, e->returnKind, e->returnShape, NULL)) return false;
            e->selfSlow[si].tag =
                  e->returnKind == SLOT_INT   ? VAL_INT
                : e->returnKind == SLOT_FLOAT ? VAL_FLOAT
                : e->returnKind == SLOT_BOOL  ? VAL_BOOL
                : e->returnKind == SLOT_NULL  ? VAL_NULL
                                              : VAL_OBJ;
            if (!deoptRecordAt(e, (uint32_t)off + 2u, true,
                               &e->selfSlow[si].deoptKind)) {
                return false;
            }
            off += 2;
            break;
        }

        case OP_TAIL_CALL: {
            /* Same OSR resume-offset hazard as OP_RETURN_NULL -- see there. */
            if (e->osr) {
                e->whyNot = "a return inside an OSR loop";
                return false;
            }
            /* `return C(...)` compiles to this. The call is made exactly as
             * OP_CALL makes it and its result is returned. */
            unsigned argc = code[off + 1];
            if (isClassCallee(e, argc)) {
                if (!emitCallOut(e, argc)) return false;
            } else if (e->depth >= argc + 1u &&
                       e->stack[e->depth - argc - 1] == SLOT_FUNC) {
                if (!emitGlobalCall(e, fn, argc, (uint32_t)off,
                                    (uint32_t)(off + 2))) return false;
            } else {
                e->whyNot = "tail callee is neither a class nor a compiled function";
                return false;
            }
            /* Which class came back, not just that an object did: a caller
             * binding this result to a local cannot compile without it. */
            uint32_t tshape = e->depth > 0 ? e->stackShape[e->depth - 1] : 0;
            unsigned r;
            SlotKind k;
            if (!popValue(e, &r, &k)) return false;
            if (!mergeReturnKind(e, k, tshape)) return false;
            emit(e, jaiA64MovX(0, r));
            emitEpilogue(e, 0);
            off += 2;
            /* The compiler emits OP_RETURN after a tail call and the
             * interpreter never reaches it, because the tail call returned.
             * Walking into it would try to pop from an empty stack. */
            if (off < count && code[off] == OP_RETURN) off += 1;
            break;
        }

        case OP_RETURN: {
            /* Same OSR resume-offset hazard as OP_RETURN_NULL -- see there. */
            if (e->osr) {
                e->whyNot = "a return inside an OSR loop";
                return false;
            }
            uint32_t rsh = e->depth > 0 ? e->stackShape[e->depth - 1] : 0;
            unsigned r;
            SlotKind k;
            if (e->depth == 0) return subWhy(e, "a return with an empty stack");
            if (!popValue(e, &r, &k)) {
                return subWhy(e, "returning a %s, which holds no register",
                              slotKindName(e->stack[e->depth - 1]));
            }
            /* One return kind per function: the entry point rebuilds a Value
             * from it, and it cannot rebuild two. */
            if (!mergeReturnKind(e, k, rsh)) return false;
            emit(e, jaiA64MovX(0, r));
            emitEpilogue(e, 0);
            off += 1;
            break;
        }

        default:
        unarmedOpcode:
            /* An opcode this tier does not speak -- or an arm that cannot emit
             * for the shape in front of it: interpreted from here, rather than
             * the whole function interpreted. See emitUnarmedDeopt. */
            if (!emitUnarmedDeopt(e, &fn->chunk, &off, stop)) {
                if (getenv("JAI_JIT_WHY")) {
                    fprintf(stderr, "[jit] %s declined at %s\n",
                            fn->name ? fn->name->chars : "<anon>",
                            jaiOpName((OpCode)op));
                }
                return false;
            }
            afterUncond = true;   /* the fall-through edge is gone */
            continue;
        }
    }
    /* An inlined body's offsets are the callee's; matching them against the
     * caller's fixups compares two different numbering schemes. There is
     * nothing to check either way -- it has no branches. */
    if (e->inlining) {
        /* All but the result, which OP_RETURN deliberately left in the bank for
         * inlineGlobalCall to carry across. Everything under it is either the
         * caller's (settled before the call) or about to be discarded. */
        unsigned keep = e->valueDepth > 0 ? e->valueDepth - 1 : 32u;
        for (unsigned i = 0; i < 32u; i++) {
            if (i != keep) fpSyncOne(e, i);
        }
        settleAll(e);
        return !e->failed;
    }
    fpSyncAll(e);
    settleAll(e);
    /* Nothing may branch to an offset this walk carried a float into (see fpCarry) -- declines, and the caller retries with the FP bank off. */
    for (unsigned i = 0; i < e->fpCarryCount; i++) {
        for (unsigned f = 0; f < e->fixupCount; f++) {
            if (e->fixups[f].targetOffset != e->fpCarry[i]) continue;
            e->whyNot = "a branch lands inside a float expression";
            return false;
        }
    }
    /* Mirror image for homeEarly (see fpBindLookahead): a back edge to such a bind is only visible here, after the whole walk. */
    for (unsigned i = 0; i < e->homeEarlyCount; i++) {
        for (unsigned f = 0; f < e->fixupCount; f++) {
            if (e->fixups[f].targetOffset != e->homeEarly[i]) continue;
            e->whyNot = "a branch lands on a bind whose local was written early";
            return false;
        }
    }
    /* Same for a deferred X entry: forward branches were settled during the walk, this catches a
     * backward one (a loop head sitting between an OP_INT and the operator consuming it). Nothing in the suite reaches it -- the point is that it costs a decline, not a register nothing wrote. */
    for (unsigned i = 0; i < e->deferCarryCount; i++) {
        for (unsigned f = 0; f < e->fixupCount; f++) {
            if (e->fixups[f].targetOffset != e->deferCarry[i]) continue;
            e->whyNot = "a branch lands where a value was deferred";
            return false;
        }
    }
    return !e->failed;
}

#else

#endif
