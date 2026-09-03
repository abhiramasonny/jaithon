/* jit_body_call.c -- the call and invoke arms of the opcode walk. */

#include "vm/jit/jit_arm64.h"
#include "vm/jit/jit_field_read.h"
#include "vm/gc.h"
#include "runtime/runtime.h"
#include "vm/vm.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vm/jit/jit_internal.h"

#if (defined(__aarch64__) || defined(__arm64__))

bool emitCall(Emit *e, ObjFunction *fn, const uint8_t *code, int *offp) {
    int off = *offp;
    do {
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
             * convention a self-call and jaiJitEnterFunc use. Callee must live in this module since the caller's module-version guard stands in for the callee's own entry check. The baked jitFunc address can't go stale: the arena is never freed, so the words this branches to stay exactly what they were. jitFunc is NOT written once -- a body blocked on a cold callee is recompiled when that callee compiles (ObjFunction::jitBlockedOn) -- but the address and every field read beside it here are taken in the same instant and baked as immediates, so this site keeps calling the form it was told about rather than reading a newer one's metadata. Only the ObjFunction identity needs guarding (done above). */
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
        /* The site below reads x1 as a bare verdict and x0 as one settled
         * kind; a SLOT_DYNAMIC body answers with neither. mergeReturnKind
         * refuses the widening once this flag is set, and this refuses the
         * call once the widening is decided, so the two can never coexist. */
        if (e->dynamicReturn || e->returnKind == SLOT_DYNAMIC) {
            return subWhy(e, "a self-call in a body returning dynamic");
        }
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
    } while (0);
    *offp = off;
    return true;
}

bool emitTailCall(Emit *e, ObjFunction *fn, const uint8_t *code, int *offp,
                  int count) {
    int off = *offp;
    do {
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
        emitReturnLeave(e, k);
        off += 2;
        /* The compiler emits OP_RETURN after a tail call and the
         * interpreter never reaches it, because the tail call returned.
         * Walking into it would try to pop from an empty stack. */
        if (off < count && code[off] == OP_RETURN) off += 1;
        break;
    } while (0);
    *offp = off;
    return true;
}

JitArmResult emitInvoke(Emit *e, ObjFunction *fn, ObjClosure *closure,
                        const uint8_t *code, int *offp, int count,
                        bool *afterUncondp) {
    int off = *offp;
    bool afterUncond = *afterUncondp;
    do {
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
            /* Three different refusals used to share one string, and the
             * kind it printed was this variable's INITIALISER whenever the
             * first of them fired -- reading as "a method returning null" for
             * a method whose return kind was simply never recorded. Which one
             * it is decides what would fix it, so they are kept apart. */
            const char *mwhy = NULL;
            if (!haveKind) {
                /* Split because the two want opposite fixes: nothing recorded
                 * at all means the method has not returned while the
                 * interpreter watched (a mutually-recursive pair never can),
                 * while a recorded-but-unusable one means the feedback saw
                 * more than one kind and no widening covers them. */
                if (mfn->obsReturnKind == JAI_FB_NONE) e->coldCallee = true;
                mwhy = mfn->obsReturnKind == JAI_FB_NONE
                     ? "a method that has not returned yet"
                     : mfn->obsReturnKind == JAI_FB_MIXED
                     ? "a method whose observed returns disagree"
                     : jaiFeedbackIsNullable(mfn->obsReturnKind)
                     ? "a method returning a nullable object, which only "
                       "reaches the stack as an instance"
                     : "a method returning an object kind the tier does not model";
            } else if ((rkind == SLOT_INST || rkind == SLOT_MAYBE_INST) &&
                       (rshape == 0 || !jaiClassForShape(rshape, &rrcls) ||
                        rrcls == NULL)) {
                haveKind = false;
                rrcls = NULL;
                mwhy = "a method whose return class is not on record";
            } else if (rkind != SLOT_INT && rkind != SLOT_FLOAT &&
                       rkind != SLOT_BOOL && rkind != SLOT_INST &&
                       rkind != SLOT_MAYBE_INST && rkind != SLOT_LIST &&
                       rkind != SLOT_OBJ && rkind != SLOT_NULL &&
                       !(rkind == SLOT_MAYBE_OBJ && jitMaybeObjStackOn())) {
                haveKind = false;
                mwhy = "a method returning %s";
            }
            /* A result the very next instruction pops needs no kind at all -- which is every `-> void`
             * method called as a statement, and the reason a parser's `self.skip()` used to decline the
             * function around it. Same relaxation the builtin arm below already makes. */
            DiscardKind mdisc = discardedAfter(code, off + 7, count);
            bool mdiscarded = mdisc != DISCARD_NO;
            if (!haveKind && !mdiscarded) {
                return subWhy(e, mwhy, slotKindName(rkind));
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

        /* `for (i, x) in xs.enumerate()`. OP_INVOKE (vm.c) swaps the eager
         * native for an ITER_LIST_ENUM snapshot when the next instruction is
         * the GET_ITER, and this is the same swap made on the same bytes, so
         * the loop walks the same thing under either tier: the descriptor
         * calls jitMakeEnumIter, the GET_ITER is skipped, and what is pushed
         * is a SLOT_ITER of shape 5 carrying an element sample for the pair
         * head to specialise on -- the head arm in emitForIterPair. That 5 is
         * this tier's own SLOT_ITER shape numbering and is not the OSR
         * iterKind, which is 5 for this head by coincidence: shape 4 here is
         * the dict view, whose head is iterKind 3.
         *
         * Before this the site declined the whole body twice over: the
         * result of `list.enumerate` is neither a field read nor discarded,
         * and even predicted it reached OP_GET_ITER as an object with no
         * element to look at. 96 of the 106 pair loops in lib/jaithon are
         * this shape.
         *
         * The sample and its census come off the receiver's live list,
         * exactly as the OSR list head takes them, including the
         * null-density refusal: a form pinned to the sampled class bails once
         * per null element, and past one in 64 that is slower than not
         * compiling. A list holding several classes is refused too -- the
         * head cannot widen a nested loop's variable the way the OSR entry
         * can, and pinning one class of several deopts on every other. */
        if (jaiLazyEnumerateOn() && argc == 0 &&
            off + 8 < count && code[off + 7] == OP_GET_ITER &&
            code[off + 8] == OP_FOR_ITER_PAIR &&
            AS_STRING(nameVal) == vm.strEnumerate) {
            Value probe = e->stackSeen[ridx];
            Value sample = NULL_VAL;
            if (IS_LIST(probe) && AS_LIST(probe)->count > 0) {
                bool mixed = false;
                if (!jitListHeadSample(AS_LIST(probe), 0, &sample, &mixed)) {
                    return subWhy(e, "enumerating a list whose elements are "
                                     "too often null");
                }
                if (mixed) {
                    return subWhy(e, "enumerating a list holding instances "
                                     "of several classes");
                }
            }
            if (IS_NULL(sample)) sample = e->stackElem[ridx];
            if (IS_NULL(sample)) {
                return subWhy(e, "enumerating a list with nothing to look at");
            }
            if (!emitDescriptor(e, NULL_VAL, ridx, 1,
                                (void *)&jitMakeEnumIter)) {
                return false;
            }
            unsigned edrop;
            if (!popValue(e, &edrop, NULL)) return false;
            if (!pushValue3(e, SLOT_ITER, 5, NULL, sample, -1)) return false;
            emit(e, jaiA64LdrX(pushReg(e) - 1, 31,
                               e->descOffset +
                                   (unsigned)offsetof(JitCallDesc, result) + 8));
            e->wroteHeap = true;
            off += 8;              /* the invoke and the GET_ITER it absorbed */
            break;
        }

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
    } while (0);
    *offp = off;
    *afterUncondp = afterUncond;
    return JIT_ARM_OK;
unarmedOpcode:
    *offp = off;
    *afterUncondp = afterUncond;
    return JIT_ARM_UNARMED;
}

#endif /* __aarch64__ || __arm64__ */
