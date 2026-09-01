/* jit_body_call.c -- the call and invoke arms of the opcode walk. */
#include "vm/jit/jit.h"

#include "vm/jit/jit_arm64.h"
#include "vm/jit/jit_field_read.h"
#include "vm/gc.h"
#include "runtime/runtime.h"
#include "vm/bytecode/verify.h"
#include "vm/vm.h"

#include <stddef.h>
#include <stdint.h>
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
        emitEpilogue(e, 0);
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

#endif /* __aarch64__ || __arm64__ */
