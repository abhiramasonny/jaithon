/* jit_call_callee.c -- one emitter per kind of callee a descriptor call can
 * reach: a global function, a module member, a module native, a class. */

#include "vm/jit/jit_arm64.h"
#include "vm/jit/jit_field_read.h"
#include "vm/vm.h"

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "vm/jit/jit_internal.h"

#if (defined(__aarch64__) || defined(__arm64__))

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
    if (!haveKind) {
        return subWhy(e, "a callee whose return kind is not on record");
    }
    if (rk != SLOT_INT && rk != SLOT_FLOAT && rk != SLOT_BOOL &&
        rk != SLOT_INST && rk != SLOT_MAYBE_INST && rk != SLOT_LIST &&
        rk != SLOT_OBJ && rk != SLOT_NULL) {
        return subWhy(e, "a callee returning %s", slotKindName(rk));
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

#endif /* __aarch64__ || __arm64__ */
