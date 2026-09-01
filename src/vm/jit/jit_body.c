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
bool mergeReturnKind(Emit *e, SlotKind k, uint32_t shape) {
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

bool fpWorthLoading(const Emit *e, const uint8_t *code, int next,
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

bool emitUnarmedDeopt(Emit *e, const Chunk *c, int *off, int stop) {
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

        case OP_TYPE_GUARD:
            if (!emitTypeGuard(e, fn, code, &off)) return false;
            break;

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
            JitArmResult r = emitGetFieldLocal(e, fn, code, &off, stop);
            if (r == JIT_ARM_REFUSED) return false;
            if (r == JIT_ARM_UNARMED) goto unarmedOpcode;
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

        case OP_GET_FIELD:
            if (!emitGetField(e, fn, code, &off, stop)) return false;
            break;

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

        case OP_GET_ITER:
            if (!emitGetIter(e, &off)) return false;
            break;

        case OP_ITER_RANGE:
            if (!emitIterRange(e, code, &off)) return false;
            break;

        case OP_FOR_RANGE_BIND:
            if (!emitForRangeBind(e, code, &off)) return false;
            break;

        case OP_GET_ITER_ITEMS:
            if (!emitGetIterItems(e, code, &off, stop)) return false;
            break;

        case OP_FOR_ITER_BIND:
            if (!emitForIterBind(e, code, &off)) return false;
            break;

        case OP_FOR_ITER_PAIR:
            if (!emitForIterPair(e, code, &off)) return false;
            break;

        case OP_GET_INDEX:
            if (!emitGetIndex(e, code, &off, stop)) return false;
            break;

        case OP_SET_INDEX:
            if (!emitSetIndex(e, &off)) return false;
            break;

        case OP_GET_SLICE:
            if (!emitGetSlice(e, code, &off)) return false;
            break;

        case OP_GET_GLOBAL:
            if (!emitGetGlobal(e, fn, closure, code, &off, stop)) return false;
            break;

        case OP_SET_GLOBAL:
            if (!emitSetGlobal(e, closure, code, &off)) return false;
            break;

        case OP_CALL:
            if (!emitCall(e, fn, code, &off)) return false;
            break;

        case OP_TAIL_CALL:
            if (!emitTailCall(e, fn, code, &off, count)) return false;
            break;

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
