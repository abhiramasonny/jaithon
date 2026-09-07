/* jit_body.c -- the opcode walk: compileBody and the helpers only it uses. */

#include "vm/jit/jit_arm64.h"
/* For jaiJitFieldReadFor: which builtins are one load from their receiver. */
#include "vm/jit/jit_field_read.h"
/* For jaiBuiltinMethod: resolving `xs.len()` to a native needs the runtime's name table. */
/* For jaiOpBranchOperandAt: says which opcodes carry a branch target. */
#include "vm/bytecode/verify.h"

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
        int64_t want = e->fixups[i].depth;
        if (want < 0) continue;
        unsigned d = jitJoinMode() == 0 ? (unsigned)want & 0xfu
                                        : (unsigned)want & 0x1fu;
        if (d > e->depth) continue;
        if (stackSignatureAt(e, d) != want) continue;
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
        /* popSkipTarget is the other half of the same question: the `match`
         * arms retarget a miss edge one instruction past where the bytecode
         * sends it, so this offset can be reached by something the chunk scan
         * cannot see. See matchMissResume. */
        if (anyStackProof(e) &&
            (offsetIsBranchTarget(&fn->chunk, (uint32_t)off) ||
             popSkipTarget(e, (uint32_t)off))) {
            clearStackProofs(e);
        }
        /* A field-kind memo is good along the same one edge, and goes for the
         * same reason -- see forgetFieldKinds. `fn` is whichever body is being
         * walked, so an inlined one is measured against its own chunk. */
        if (e->knownCount != 0 &&
            (offsetIsBranchTarget(&fn->chunk, (uint32_t)off) ||
             popSkipTarget(e, (uint32_t)off))) {
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
        e->offsetToDepth[off] = stackSignature(e);
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
        case OP_GET_LOCAL:
            if (!emitGetLocal(e, code, &off, stop)) return false;
            break;

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

        case OP_SET_LOCAL:
            if (!emitSetLocal(e, code, &off)) return false;
            break;

        case OP_ADD_LOCALS:
            if (!emitAddLocals(e, code, &off)) return false;
            break;

        case OP_ADD_BIND:
            if (!emitAddBind(e, code, &off)) return false;
            break;

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
            JitArmResult r = emitElemKind(e, code, &off);
            if (r == JIT_ARM_REFUSED) return false;
            if (r == JIT_ARM_UNARMED) goto unarmedOpcode;
            break;
        }

        case OP_CMP_LOCAL_CONST_LT:
            if (!emitCmpLocalConstLt(e, code, &off)) return false;
            break;

        case OP_ADD_INT_CONST:
            if (!emitAddIntConst(e, code, &off)) return false;
            break;

        case OP_SUB_INT_CONST:
            if (!emitSubIntConst(e, code, &off)) return false;
            break;

        case OP_MUL_INT_CONST:
            if (!emitMulIntConst(e, code, &off)) return false;
            break;

        case OP_MUL_BIND:
            if (!emitMulBind(e, code, &off)) return false;
            break;

        case OP_SUB_BIND:
            if (!emitSubBind(e, code, &off)) return false;
            break;

        case OP_BIND:
            if (!emitBind(e, code, &off)) return false;
            break;

        case OP_INC_LOCAL:
            if (!emitIncLocal(e, code, &off)) return false;
            break;

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

        /* The four opcodes an enum-variant `match` is made of. Arming any ONE
         * of them buys exactly nothing -- the next link of the chain is two
         * instructions later -- so they land together. See jit_body_match.c. */
        case OP_MATCH_TYPE_POP: {
            JitArmResult r = emitMatchTypePop(e, closure, code, &off);
            if (r == JIT_ARM_REFUSED) return false;
            if (r == JIT_ARM_UNARMED) goto unarmedOpcode;
            break;
        }

        case OP_ENUM_TAG: {
            JitArmResult r = emitEnumTag(e, &off);
            if (r == JIT_ARM_REFUSED) return false;
            if (r == JIT_ARM_UNARMED) goto unarmedOpcode;
            break;
        }

        case OP_SWAP_POP: {
            JitArmResult r = emitSwapPop(e, &off);
            if (r == JIT_ARM_REFUSED) return false;
            if (r == JIT_ARM_UNARMED) goto unarmedOpcode;
            break;
        }

        case OP_MATCH_CONST_POP: {
            JitArmResult r = emitMatchConstPop(e, fn, code, &off, &afterUncond);
            if (r == JIT_ARM_REFUSED) return false;
            if (r == JIT_ARM_UNARMED) goto unarmedOpcode;
            break;
        }

        case OP_FORMAT:
            if (!emitFormat(e, closure, code, &off)) return false;
            break;

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

        case OP_MUL:
            if (!emitMul(e, fn, code, &off, stop)) return false;
            break;

        case OP_BAND:
        case OP_BOR:
        case OP_BXOR:
        case OP_SHL:
        case OP_SHR:
            if (!emitBitOp(e, fn, op, prevOff, &off)) return false;
            break;

        case OP_ADD:
        case OP_SUB:
        case OP_DIV:
            if (!emitAddSubDiv(e, fn, op, code, &off, stop)) return false;
            break;

        case OP_JUMP_IF_CMP_LOCAL_K:
            if (!emitJumpIfCmpLocalK(e, fn, code, &off)) return false;
            break;

        case OP_GET_FIELD_LOCAL: {
            JitArmResult r = emitGetFieldLocal(e, fn, code, &off, stop);
            if (r == JIT_ARM_REFUSED) return false;
            if (r == JIT_ARM_UNARMED) goto unarmedOpcode;
            break;
        }

        case OP_GET_LOCAL2:
            if (!emitGetLocal2(e, code, &off, stop)) return false;
            break;

        case OP_SET_FIELD:
            if (!emitSetField(e, fn, code, &off)) return false;
            break;

        case OP_RETURN_NULL:
            if (!emitReturnNull(e, fn, &off)) return false;
            break;

        case OP_POP_RETURN_NULL:
            if (!emitPopReturnNull(e, fn, &off)) return false;
            break;

        case OP_GET_UPVALUE:
            if (!emitGetUpvalue(e, fn, closure, code, &off)) return false;
            break;

        case OP_GET_FIELD:
            if (!emitGetField(e, fn, code, &off, stop)) return false;
            break;

        case OP_BUILD_LIST:
            if (!emitBuildList(e, code, &off)) return false;
            break;

        case OP_IN:
        case OP_NOT_IN: {
            JitArmResult r = emitMembership(e, code, &off);
            if (r == JIT_ARM_REFUSED) return false;
            if (r == JIT_ARM_UNARMED) goto unarmedOpcode;
            break;
        }

        case OP_BUILD_DICT:
        case OP_BUILD_SET: {
            JitArmResult r = emitBuildDictSet(e, code, &off);
            if (r == JIT_ARM_REFUSED) return false;
            if (r == JIT_ARM_UNARMED) goto unarmedOpcode;
            break;
        }

        case OP_BUILD_TUPLE: {
            JitArmResult r = emitBuildTuple(e, code, &off);
            if (r == JIT_ARM_REFUSED) return false;
            if (r == JIT_ARM_UNARMED) goto unarmedOpcode;
            break;
        }

        case OP_ADD_WRAP:
        case OP_SUB_WRAP:
        case OP_MUL_WRAP: {
            JitArmResult r = emitWrapArith(e, op, &off);
            if (r == JIT_ARM_REFUSED) return false;
            if (r == JIT_ARM_UNARMED) goto unarmedOpcode;
            break;
        }

        case OP_NOT: {
            JitArmResult r = emitNot(e, &off);
            if (r == JIT_ARM_REFUSED) return false;
            if (r == JIT_ARM_UNARMED) goto unarmedOpcode;
            break;
        }

        case OP_BNOT: {
            JitArmResult r = emitBitNot(e, &off);
            if (r == JIT_ARM_REFUSED) return false;
            if (r == JIT_ARM_UNARMED) goto unarmedOpcode;
            break;
        }

        case OP_POS: {
            JitArmResult r = emitUnaryPlus(e, &off);
            if (r == JIT_ARM_REFUSED) return false;
            if (r == JIT_ARM_UNARMED) goto unarmedOpcode;
            break;
        }

        case OP_NEG:
            if (!emitNegate(e, &off)) return false;
            break;

        case OP_POW:
            if (!emitPow(e, &off)) return false;
            break;

        case OP_MOD:
            if (!emitMod(e, fn, prevOff, &off)) return false;
            break;

        case OP_MOD_INT_CONST:
            if (!emitModIntConst(e, code, &off)) return false;
            break;

        case OP_FLOORDIV:
            if (!emitFloorDiv(e, fn, prevOff, &off)) return false;
            break;

        /* A list comprehension's append. It is the same two stores `push`
         * makes, but it was the one list write with no arm at all, so every
         * comprehension in the language -- `[0 for _i in 0..n]`, the way this
         * codebase preallocates -- left its loop running interpreted. */
        case OP_LIST_APPEND: {
            unsigned back = jaiReadU16(code + off + 1);
            if (e->depth < back + 1u) {
                /* A comprehension: the accumulator was pushed before the loop
                 * head, so the model never saw it. compileOsr worked out which
                 * frame slot holds it (findComprehensionAcc) and reserved a
                 * callee-saved register, because emitListStore needs the
                 * container somewhere that survives its four loads and the grow
                 * stub's call -- every scratch is spoken for inside it, and a
                 * hoist register can never be placed in a loop that appends,
                 * since an append IS a call as far as regionCalls is concerned.
                 *
                 * Reloaded from the frame at every append rather than hoisted
                 * once: the register is then a pure cache of a slot the
                 * interpreter still owns, so a deopt needs no write-back, and
                 * nothing has to assume the slot's contents are stable. The
                 * guard is what makes a wrong slot a deopt instead of a store
                 * through a bad pointer. */
                if (!e->osr || !e->accWanted) {
                    e->whyNot = "an append reaching past the model";
                    return false;
                }
                unsigned rAcc = osrAccReg(e);
                unsigned at = (unsigned)e->accSlot * 16u;
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SLOTS_REG, at));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, VAL_OBJ));
                branchOnDeopt(e, JAI_A64_NE);
                emit(e, jaiA64LdrX(rAcc, JIT_SLOTS_REG, at + 8u));
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, rAcc,
                                   (unsigned)offsetof(Obj, type)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_LIST));
                branchOnDeopt(e, JAI_A64_NE);
                if (!emitListStore(e, e->stack[e->depth - 1], rAcc,
                                   pushReg(e) - 1, -1)) {
                    return false;
                }
                unsigned dropAcc;
                if (!popValue(e, &dropAcc, NULL)) return false;
                off += 3;
                break;
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
            JitArmResult r = emitInvoke(e, fn, closure, code, &off, count,
                                        &afterUncond);
            if (r == JIT_ARM_REFUSED) return false;
            if (r == JIT_ARM_UNARMED) goto unarmedOpcode;
            break;
        }

        case OP_BUILD_RANGE:
            if (!emitBuildRange(e, code, &off, count)) return false;
            break;

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

        case OP_RETURN:
            if (!emitReturn(e, &off)) return false;
            break;

        default:
        unarmedOpcode:
            /* An opcode this tier does not speak -- or an arm that cannot emit
             * for the shape in front of it: interpreted from here, rather than
             * the whole function interpreted. See emitUnarmedDeopt. */
            if (!emitUnarmedDeopt(e, &fn->chunk, &off, stop)) {
                if (getenv("JAI_JIT_WHY")) {
                    fprintf(stderr, "[jit] %s declined at %s\n",
                            jitFnLabel(fn),
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
