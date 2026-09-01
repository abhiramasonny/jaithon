/* vm.c — the bytecode interpreter (spec/BYTECODE.md).
 *
 * Three invariants hold everywhere below:
 *   1. ip/stackTop/slots/constants/frame are C locals throughout runLoop();
 *      SAVE_STATE()/LOAD_STATE() sync them with the VM around anything that
 *      can allocate, call, or throw -- the GC scans up to vm.stackTop, so a
 *      stale one is a collected live object, not just a slow path.
 *   2. Every re-entrant helper works off vm.stackTop, not the loop's local,
 *      and restores it after (plus whatever it pushed) -- this is what lets
 *      jaiCallValue and friends be called from native code at any depth.
 *   3. No setjmp/longjmp: a raise sets vm.pendingException and jumps to
 *      `vmThrow`, which walks handlers and frames explicitly.
 */
#include <inttypes.h>
#include <math.h>
#include <signal.h>
#include <stdlib.h>

#include "vm/vm.h"
#include "vm/vm_internal.h"
#include "vm/jit/jit.h"
#include "vm/trace/trace.h"

#include "vm/gc.h"
#include "vm/object/object.h"
#include "vm/table.h"
#include "runtime/runtime.h"

VM vm;

/* Nested runLoop() invocations, one per re-entry from native code. Each costs
 * a C stack frame, so it is bounded well below the interpreter frame limit. */
#define JAI_MAX_NESTED_RUN 128
int sRunDepth;

/* An exception suspended by a `finally` that was reached while unwinding.
 * OP_END_FINALLY resumes the unwind when this is nonzero. */
int sFinallyPending;

/* Frame index of the default-value thunk currently running, or -1. A thunk
 * shares its function record with the method it belongs to, so OP_RETURN has
 * to know not to apply the initializer's "return self" rule to it. Thunks
 * nest strictly, so one saved index is enough. */
int sThunkFrame = -1;

/* Set from SIGINT so a runaway program can be stopped at the LOOP safepoint. */
/* 0 = nothing, 1 = Ctrl-C, 2 = a sampling tick from the JIT's timer.
 *
 * One flag with three states rather than two flags, so a tick rides the back
 * edge's existing test instead of adding one. Measured: a new branch in OP_LOOP
 * costs 11% even when it is never taken. */
volatile sig_atomic_t jaiInterrupted;

/* ------------------------------------------------------------------ */
/* GC roots                                                             */
/* ------------------------------------------------------------------ */

void jaiPushRoot(Value v)   { jaiGCPushRoot(v); }
void jaiPopRoot(void)       { jaiGCPopRoot(); }
void jaiPopRoots(int n)     { jaiGCPopRoots(n); }

/* ------------------------------------------------------------------ */
/* Safepoint                                                            */
/* ------------------------------------------------------------------ */

/* The LOOP safepoint (spec/BYTECODE.md §10): the one place a long-running
 * program can be interrupted and the one place a collection is guaranteed to
 * be able to run. */
static bool safepoint(void) {
    if (JAI_UNLIKELY(jaiInterrupted == 2)) {
        jaiInterrupted = 0;
        if (vm.frameCount > 0) {
            CallFrame *top = &vm.frames[vm.frameCount - 1];
            if (!jaiJitSample(top->closure,
                              (uint32_t)(top->ip -
                                         top->closure->fn->chunk.code))) {
                return false;   /* the compiled loop raised */
            }
        }
    } else if (vm.frameCount > 0) {
        /* A loop that already has a compiled form enters it here, on the back
         * edge, rather than waiting for the next timer tick.
         *
         * Waiting was the whole reason the compiled form did not matter. A
         * tick arrives at 4kHz and only counts when it lands on a back edge,
         * so a loop that runs a hundred iterations and exits is almost never
         * entered: matrix_mul's innermost body runs 1.7 million times and the
         * compiled form was entered fewer than twenty thousand. Reaching it
         * from the back edge instead costs one load and a compare on a path
         * that has no compiled form, measured at about 1% of the suite. */
        CallFrame *top = &vm.frames[vm.frameCount - 1];
        ObjFunction *f = top->closure->fn;
        if (f->osrHot) {
            uint32_t at = (uint32_t)(top->ip - f->chunk.code);
            JaiOsrForm *form = NULL;
            for (unsigned i = 0; i < f->osrCount; i++) {
                if (f->osrForms[i].top == at) { form = &f->osrForms[i]; break; }
            }
            /* Given up on per HEAD: a guard that keeps failing on one loop
             * says nothing about the other loops in the same body. */
            if (form != NULL && form->declines < JAI_OSR_GIVE_UP) {
                uint32_t resumeAt = 0;
                int outcome = jaiJitEnterOsr(top->closure, at, &resumeAt);
                if (outcome == 2) return false;
                if (outcome == 1) {
                    top->ip = f->chunk.code + resumeAt;
                    form->declines = 0;
                } else if (++form->declines >= JAI_OSR_GIVE_UP) {
                    /* Stop paying the back-edge check only once every head has
                     * been given up on. */
                    bool any = false;
                    for (unsigned i = 0; i < f->osrCount; i++) {
                        if (f->osrForms[i].declines < JAI_OSR_GIVE_UP) { any = true; break; }
                    }
                    f->osrHot = any;
                }
            }
        }
    }
    /* Only 1 is Ctrl-C. A tick that arrived while compiled code was running
     * leaves 2 here, and treating any non-zero value as an interrupt turned
     * every long compiled loop into a spurious RuntimeError. */
    if (JAI_UNLIKELY(jaiInterrupted == 1)) {
        jaiInterrupted = 0;
        return jaiThrow(vm.cRuntimeError, "interrupted");
    }
    jaiGCMaybeCollect();
    return true;
}

#ifdef JAI_OPCODE_STATS
uint64_t jaiOpCounts[OP_COUNT];
/* Successor pairs: jaiOpPairs[a][b] is how often opcode b was dispatched
 * immediately after a. Dumped as `pair A B count share` beside the per-opcode
 * rows, where share is B's fraction of everything that followed A.
 *
 * The per-opcode counts say which instructions are hot; only this says which
 * DISPATCH is predictable, which is the question VM_NEXT_HINT below exists to
 * answer and the five hint sites in this file were picked by hand without.
 *
 * MEASURED AND FOUND NOT TO MATTER, 2026-08-31, which is the useful part.
 * Checking parser.jai the census names a chain the `match` lowering walks
 * 1.35M times each at ~100% -- OP_ENUM_TAG -> OP_SWAP_POP ->
 * OP_MATCH_CONST_POP -> OP_POP -- plus OP_MATCH_TYPE_POP -> OP_GET_LOCAL
 * (1,354,485, 100%), OP_LIST_APPEND -> OP_LOOP (379,584, 100%) and
 * OP_IS -> OP_JUMP_IF_FALSE (332,845, 94.4%). Hinting all seven, about 7.7M
 * of the run's 43M dispatches, measured FLAT: ten iterations per arm in one
 * binary behind a runtime gate, 5.25/5.16/5.14 s against 5.26/5.15/5.16 s.
 *
 * So the interpreter is not branch-bound here. `runLoop` is 26.7% of a
 * sampling profile, and that is the opcodes doing their work, not the shared
 * indirect branch mispredicting. Do not spend another session on dispatch
 * threading without a measurement that contradicts this. */
uint64_t jaiOpPairs[OP_COUNT][OP_COUNT];
uint8_t  jaiPrevOp;
#endif

#ifdef JAI_PROP_STATS
uint64_t jaiPropRecv[32];
#endif

/* ------------------------------------------------------------------ */
/* The interpreter loop                                                 */
/* ------------------------------------------------------------------ */

#if JAI_COMPUTED_GOTO
#  define VM_CASE(name)  L_##name
#  ifdef JAI_OPCODE_STATS
#    define VM_NEXT()      do { DISPATCH_TRACE(); instStart = ip;                \
                                jaiOpCounts[*ip]++;                              \
                                jaiOpPairs[jaiPrevOp][*ip]++;                    \
                                jaiPrevOp = *ip;                                 \
                                goto *jaiDispatchTable[*ip++]; } while (0)
#  else
#    define VM_NEXT()      do { DISPATCH_TRACE(); instStart = ip;                \
                              goto *jaiDispatchTable[*ip++]; } while (0)
#  endif
/* Clang emits exactly ONE indirect branch per function for computed goto --
 * CodeGenFunction caches a single indirect-goto block and every `goto *p`
 * branches to it -- so all 132 opcodes share one entry in the branch-target
 * predictor no matter how this is written. VM_NEXT_HINT names the successor
 * the census says is overwhelmingly likely and reaches it by a direct branch,
 * which the predictor keys on its own address. A wrong guess costs one
 * compare and falls through to the shared branch, so the only hints worth
 * taking are the lopsided ones: a 50/50 hint replaces a predictable indirect
 * branch with an unpredictable direct one. */
#  ifdef JAI_OPCODE_STATS
/* Counted on both arms, and before the compare, so a hinted dispatch is not
 * invisible to the census. It used to be: the five hint sites cover OP_LOOP,
 * OP_INC_LOCAL and OP_BIND, and loop_sum's OP_LOOP -- 14.28% of that run --
 * reported as zero. sum(jaiOpCounts) == vm.instructionCount is the invariant
 * that says no dispatch path is missing, and scripts/opstats_check.sh is what
 * asserts it. */
#    define VM_NEXT_HINT(nextOp)                                               \
        do { DISPATCH_TRACE(); instStart = ip;                                 \
             jaiOpCounts[*ip]++;                                               \
             jaiOpPairs[jaiPrevOp][*ip]++;                                     \
             jaiPrevOp = *ip;                                                  \
             if (JAI_LIKELY(*ip == (nextOp))) { ip++; goto L_##nextOp; }       \
             goto *jaiDispatchTable[*ip++]; } while (0)
#  else
#    define VM_NEXT_HINT(nextOp)                                               \
        do { DISPATCH_TRACE(); instStart = ip;                                 \
             if (JAI_LIKELY(*ip == (nextOp))) { ip++; goto L_##nextOp; }       \
             goto *jaiDispatchTable[*ip++]; } while (0)
#  endif
/* The trailing `;` lets the case labels that follow live in a plain block;
 * labels have function scope, so the nesting costs nothing. */
#  define VM_DISPATCH()  VM_NEXT();
#else
#  define VM_CASE(name)  case name
#  define VM_NEXT()      goto vmDispatch
#  define VM_NEXT_HINT(nextOp)  ((void)(nextOp), VM_NEXT())
#  define VM_DISPATCH()  vmDispatch:                                           \
                         DISPATCH_TRACE();                                     \
                         instStart = ip;                                       \
                         switch ((OpCode)*ip++)
#endif

/* Hot state lives in locals; these two macros are the only bridge back to the
 * VM struct. Anything that can allocate, call, or throw must be bracketed by
 * them — the collector reads vm.stackTop, and a stale one frees live values. */
#define SAVE_STATE()                                                           \
    do {                                                                       \
        frame->ip = ip;                                                        \
        vm.stackTop = stackTop;                                                \
    } while (0)

/* After a call that completed without pushing a frame, the frame, its ip, its
 * slots and its constants are all exactly as they were -- the frame array and
 * the value stack are both fixed-capacity, so neither can have moved. Only the
 * stack top changed. LOAD_STATE's six loads, one of them a three-deep chase to
 * the constant pool, buy nothing there, and every native call was paying
 * them. instStart needs no restoring because VM_NEXT sets it. */
#define LOAD_STACK_ONLY()  (stackTop = vm.stackTop)

#define LOAD_STATE()                                                           \
    do {                                                                       \
        frame = &vm.frames[vm.frameCount - 1];                                 \
        ip = frame->ip;                                                        \
        instStart = ip;                                                        \
        stackTop = vm.stackTop;                                                \
        slots = frame->slots;                                                  \
        constants = frame->closure->fn->chunk.constants.data;                  \
    } while (0)

/* Word for word what `UNPACK 2 255` raises for `item`, because that is the
 * sequence this replaces and an optimisation may not change a message. */
static bool pairSplitFail(Value item) {
    if (!IS_LIST(item) && !IS_TUPLE(item)) {
        return jaiThrow(vm.cTypeError, "cannot destructure a '%s' value",
                        jaiTypeNameStatic(item));
    }
    const int available = IS_LIST(item) ? AS_LIST(item)->count
                                        : (int)AS_TUPLE(item)->count;
    return jaiThrow(vm.cValueError, "cannot unpack %d value%s into 2 targets",
                    available, available == 1 ? "" : "s");
}

#define READ_BYTE()  (*ip++)
#define READ_I8()    ((int8_t)*ip++)
#define READ_U16()   (ip += 2, jaiReadU16(ip - 2))
#define READ_U24()   (ip += 3, jaiReadU24(ip - 3))
#define READ_I16()   (ip += 2, jaiReadI16(ip - 2))
#define READ_CONST() (constants[READ_U24()])

#define PUSH(v)  (*stackTop++ = (v))
#define POP()    (*(--stackTop))
#define PEEK(d)  (stackTop[-1 - (d)])
#define DROP(n)  (stackTop -= (n))

/* Raise and unwind. The state must be current: the unwinder inspects the
 * frame's ip to find the protected region the fault occurred in. */
#define THROW(...)                                                             \
    do {                                                                       \
        SAVE_STATE();                                                          \
        (void)jaiThrow(__VA_ARGS__);                                           \
        goto vmThrow;                                                          \
    } while (0)

#define DISPATCH_TRACE()                                                       \
    do {                                                                       \
        if (JAI_UNLIKELY(countInsts)) {                                        \
            vm.instructionCount++;                                             \
            if (JAI_UNLIKELY(vm.attributeInstructions)) {                      \
                attributeInstruction(frame);                                   \
            }                                                                  \
            if (vm.debugTrace) {                                               \
                vm.stackTop = stackTop;                                        \
                traceInstruction(frame, ip);                                   \
            }                                                                  \
        }                                                                      \
    } while (0)

/* The program rebound the global `str`, so OP_FORMAT owes its holes a call to
 * whatever that name means now — and owes it only to the holes: the literal
 * runs between them are source text the old lowering never converted, which is
 * what `litmask` records. Converts the top `count` stack slots in place; they
 * are re-addressed through vm.stackTop on every iteration because a call can
 * grow the value stack and move it. */
static bool formatViaUserStr(ObjModule *module, ObjString *name, int count,
                             uint32_t litmask) {
    Value strFn;
    if (module == NULL || !jaiTableGetInterned(&module->globals, name, &strFn)) {
        return true;                     /* vanished again; the builtin applies */
    }
    jaiGCPushRoot(strFn);                /* the call may rebind the name */
    for (int i = 0; i < count; i++) {
        if (litmask & (1u << i)) continue;
        Value arg = vm.stackTop[i - count];
        Value converted;
        if (!jaiCallValue(strFn, 1, &arg, &converted)) {
            jaiGCPopRoot();
            return false;
        }
        vm.stackTop[i - count] = converted;
    }
    jaiGCPopRoot();
    return true;
}

/* Both binary helpers share this shape: pop two, push one, with the slow path
 * free to re-enter the VM through a dunder method. */
#define BINARY(fn, opcode)                                                     \
    do {                                                                       \
        SAVE_STATE();                                                          \
        Value _result;                                                         \
        if (!fn((opcode), stackTop[-2], stackTop[-1], &_result)) goto vmThrow;  \
        LOAD_STATE();                                                          \
        stackTop -= 2;                                                         \
        PUSH(_result);                                                         \
        VM_NEXT();                                                             \
    } while (0)

/* Ordered comparison of two same-typed numbers, decided in the dispatch arm.
 * The generic path calls jaiValueCompare, which was 7.2% of loop_sum's whole
 * run just to answer `int < int`. Falls through to BINARY on anything else.
 * NaN is deliberately excluded: an unordered pair is a TypeError here (spec
 * §3.3), not a false, and compareDoubles is the one that reports it. */
#define CMP_FAST(cop)                                                          \
    do {                                                                       \
        Value _b = stackTop[-1], _a = stackTop[-2];                            \
        if (JAI_LIKELY(IS_INT(_a) && IS_INT(_b))) {                            \
            bool _r = AS_INT(_a) cop AS_INT(_b);                               \
            DROP(1);                                                           \
            stackTop[-1] = BOOL_VAL(_r);                                       \
            VM_NEXT();                                                         \
        }                                                                      \
        if (IS_FLOAT(_a) && IS_FLOAT(_b)) {                                    \
            double _x = AS_FLOAT(_a), _y = AS_FLOAT(_b);                       \
            if (JAI_LIKELY(!isnan(_x) && !isnan(_y))) {                        \
                bool _r = _x cop _y;                                           \
                DROP(1);                                                       \
                stackTop[-1] = BOOL_VAL(_r);                                   \
                VM_NEXT();                                                     \
            }                                                                  \
        }                                                                      \
    } while (0)

/* A jump condition must be a bool. There is no truthiness in Jaithon (spec
 * §5.1) and `any` values reach here unchecked, so the VM is the last line of
 * defence for the rule. */
#define REQUIRE_BOOL(v, what)                                                  \
    do {                                                                       \
        if (JAI_UNLIKELY(!IS_BOOL(v))) {                                       \
            THROW(vm.cTypeError,                                               \
                  "%s must be bool, not '%s'; there is no truthiness in "      \
                  "Jaithon", (what), jaiTypeNameStatic(v));                    \
        }                                                                      \
    } while (0)

static JaiRunResult runLoop(int baseFrameCount) {
#if JAI_COMPUTED_GOTO
    static const void *const jaiDispatchTable[] = {
        [OP_NOP]                = &&L_OP_NOP,
        [OP_CONST]              = &&L_OP_CONST,
        [OP_NULL]               = &&L_OP_NULL,
        [OP_TRUE]               = &&L_OP_TRUE,
        [OP_FALSE]              = &&L_OP_FALSE,
        [OP_INT]                = &&L_OP_INT,
        [OP_POP]                = &&L_OP_POP,
        [OP_POPN]               = &&L_OP_POPN,
        [OP_DUP]                = &&L_OP_DUP,
        [OP_DUP2]               = &&L_OP_DUP2,
        [OP_SWAP]               = &&L_OP_SWAP,
        [OP_ROT3]               = &&L_OP_ROT3,
        [OP_GET_LOCAL]          = &&L_OP_GET_LOCAL,
        [OP_SET_LOCAL]          = &&L_OP_SET_LOCAL,
        [OP_GET_UPVALUE]        = &&L_OP_GET_UPVALUE,
        [OP_SET_UPVALUE]        = &&L_OP_SET_UPVALUE,
        [OP_CLOSE_UPVALUE]      = &&L_OP_CLOSE_UPVALUE,
        [OP_GET_GLOBAL]         = &&L_OP_GET_GLOBAL,
        [OP_SET_GLOBAL]         = &&L_OP_SET_GLOBAL,
        [OP_DEF_GLOBAL]         = &&L_OP_DEF_GLOBAL,
        [OP_GET_MODULE]         = &&L_OP_GET_MODULE,
        [OP_ADD]                = &&L_OP_ADD,
        [OP_SUB]                = &&L_OP_SUB,
        [OP_MUL]                = &&L_OP_MUL,
        [OP_DIV]                = &&L_OP_DIV,
        [OP_FLOORDIV]           = &&L_OP_FLOORDIV,
        [OP_MOD]                = &&L_OP_MOD,
        [OP_POW]                = &&L_OP_POW,
        [OP_ADD_WRAP]           = &&L_OP_ADD_WRAP,
        [OP_SUB_WRAP]           = &&L_OP_SUB_WRAP,
        [OP_MUL_WRAP]           = &&L_OP_MUL_WRAP,
        [OP_NEG]                = &&L_OP_NEG,
        [OP_POS]                = &&L_OP_POS,
        [OP_BAND]               = &&L_OP_BAND,
        [OP_BOR]                = &&L_OP_BOR,
        [OP_BXOR]               = &&L_OP_BXOR,
        [OP_SHL]                = &&L_OP_SHL,
        [OP_SHR]                = &&L_OP_SHR,
        [OP_BNOT]               = &&L_OP_BNOT,
        [OP_EQ]                 = &&L_OP_EQ,
        [OP_NE]                 = &&L_OP_NE,
        [OP_LT]                 = &&L_OP_LT,
        [OP_LE]                 = &&L_OP_LE,
        [OP_GT]                 = &&L_OP_GT,
        [OP_GE]                 = &&L_OP_GE,
        [OP_IS]                 = &&L_OP_IS,
        [OP_IS_NOT]             = &&L_OP_IS_NOT,
        [OP_IN]                 = &&L_OP_IN,
        [OP_NOT_IN]             = &&L_OP_NOT_IN,
        [OP_NOT]                = &&L_OP_NOT,
        [OP_CONCAT]             = &&L_OP_CONCAT,
        [OP_ADD_INT_CONST]      = &&L_OP_ADD_INT_CONST,
        [OP_SUB_INT_CONST]      = &&L_OP_SUB_INT_CONST,
        [OP_MOD_INT_CONST]      = &&L_OP_MOD_INT_CONST,
        [OP_ADD_BIND]           = &&L_OP_ADD_BIND,
        [OP_SUB_BIND]           = &&L_OP_SUB_BIND,
        [OP_MUL_BIND]           = &&L_OP_MUL_BIND,
        [OP_ELEM_KIND]          = &&L_OP_ELEM_KIND,
        [OP_GET_ITER_ITEMS]     = &&L_OP_GET_ITER_ITEMS,
        [OP_FOR_ITER_PAIR]      = &&L_OP_FOR_ITER_PAIR,
        [OP_ITER_RANGE]         = &&L_OP_ITER_RANGE,
        [OP_FOR_RANGE_BIND]     = &&L_OP_FOR_RANGE_BIND,
        [OP_INC_LOCAL]          = &&L_OP_INC_LOCAL,
        [OP_CMP_LOCAL_CONST_LT] = &&L_OP_CMP_LOCAL_CONST_LT,
        [OP_GET_LOCAL2]         = &&L_OP_GET_LOCAL2,
        [OP_ADD_LOCALS]         = &&L_OP_ADD_LOCALS,
        [OP_JUMP]               = &&L_OP_JUMP,
        [OP_JUMP_IF_FALSE]      = &&L_OP_JUMP_IF_FALSE,
        [OP_JUMP_IF_TRUE]       = &&L_OP_JUMP_IF_TRUE,
        [OP_JUMP_IF_FALSE_KEEP] = &&L_OP_JUMP_IF_FALSE_KEEP,
        [OP_JUMP_IF_TRUE_KEEP]  = &&L_OP_JUMP_IF_TRUE_KEEP,
        [OP_JUMP_IF_NULL]       = &&L_OP_JUMP_IF_NULL,
        [OP_LOOP]               = &&L_OP_LOOP,
        [OP_GET_ITER]           = &&L_OP_GET_ITER,
        [OP_FOR_ITER]           = &&L_OP_FOR_ITER,
        [OP_CALL]               = &&L_OP_CALL,
        [OP_CALL_KW]            = &&L_OP_CALL_KW,
        [OP_CALL_SPREAD]        = &&L_OP_CALL_SPREAD,
        [OP_INVOKE]             = &&L_OP_INVOKE,
        [OP_SUPER_INVOKE]       = &&L_OP_SUPER_INVOKE,
        [OP_TAIL_CALL]          = &&L_OP_TAIL_CALL,
        [OP_RETURN]             = &&L_OP_RETURN,
        [OP_RETURN_NULL]        = &&L_OP_RETURN_NULL,
        [OP_POP_RETURN_NULL]    = &&L_OP_POP_RETURN_NULL,
        [OP_CLOSURE]            = &&L_OP_CLOSURE,
        [OP_BUILD_LIST]         = &&L_OP_BUILD_LIST,
        [OP_BUILD_DICT]         = &&L_OP_BUILD_DICT,
        [OP_BUILD_SET]          = &&L_OP_BUILD_SET,
        [OP_BUILD_TUPLE]        = &&L_OP_BUILD_TUPLE,
        [OP_BUILD_RANGE]        = &&L_OP_BUILD_RANGE,
        [OP_LIST_APPEND]        = &&L_OP_LIST_APPEND,
        [OP_DICT_INSERT]        = &&L_OP_DICT_INSERT,
        [OP_SET_ADD]            = &&L_OP_SET_ADD,
        [OP_GET_INDEX]          = &&L_OP_GET_INDEX,
        [OP_SET_INDEX]          = &&L_OP_SET_INDEX,
        [OP_GET_SLICE]          = &&L_OP_GET_SLICE,
        [OP_SET_SLICE]          = &&L_OP_SET_SLICE,
        [OP_UNPACK]             = &&L_OP_UNPACK,
        [OP_CLASS]              = &&L_OP_CLASS,
        [OP_INHERIT]            = &&L_OP_INHERIT,
        [OP_IMPL_TRAIT]         = &&L_OP_IMPL_TRAIT,
        [OP_METHOD]             = &&L_OP_METHOD,
        [OP_FIELD_DEF]          = &&L_OP_FIELD_DEF,
        [OP_GET_FIELD]          = &&L_OP_GET_FIELD,
        [OP_SET_FIELD]          = &&L_OP_SET_FIELD,
        [OP_GET_SUPER]          = &&L_OP_GET_SUPER,
        [OP_NEW]                = &&L_OP_NEW,
        [OP_ENUM_NEW]           = &&L_OP_ENUM_NEW,
        [OP_ENUM_TAG]           = &&L_OP_ENUM_TAG,
        [OP_ENUM_FIELD]         = &&L_OP_ENUM_FIELD,
        [OP_IS_INSTANCE]        = &&L_OP_IS_INSTANCE,
        [OP_THROW]              = &&L_OP_THROW,
        [OP_RERAISE]            = &&L_OP_RERAISE,
        [OP_PUSH_HANDLER]       = &&L_OP_PUSH_HANDLER,
        [OP_POP_HANDLER]        = &&L_OP_POP_HANDLER,
        [OP_PUSH_FINALLY]       = &&L_OP_PUSH_FINALLY,
        [OP_END_FINALLY]        = &&L_OP_END_FINALLY,
        [OP_PUSH_DEFER]         = &&L_OP_PUSH_DEFER,
        [OP_RUN_DEFERS]         = &&L_OP_RUN_DEFERS,
        [OP_MATCH_EXC]          = &&L_OP_MATCH_EXC,
        [OP_GET_EXC]            = &&L_OP_GET_EXC,
        [OP_MATCH_CONST]        = &&L_OP_MATCH_CONST,
        [OP_MATCH_CONST_POP]    = &&L_OP_MATCH_CONST_POP,
        [OP_SWAP_POP]           = &&L_OP_SWAP_POP,
        [OP_MATCH_TYPE_POP]     = &&L_OP_MATCH_TYPE_POP,
        [OP_MATCH_RANGE_POP]    = &&L_OP_MATCH_RANGE_POP,
        [OP_MATCH_SEQ_POP]      = &&L_OP_MATCH_SEQ_POP,
        [OP_MUL_INT_CONST]      = &&L_OP_MUL_INT_CONST,
        [OP_MATCH_RANGE]        = &&L_OP_MATCH_RANGE,
        [OP_MATCH_TYPE]         = &&L_OP_MATCH_TYPE,
        [OP_MATCH_SEQ]          = &&L_OP_MATCH_SEQ,
        [OP_MATCH_FIELDS]       = &&L_OP_MATCH_FIELDS,
        [OP_BIND]               = &&L_OP_BIND,
        [OP_IMPORT]             = &&L_OP_IMPORT,
        [OP_IMPORT_FROM]        = &&L_OP_IMPORT_FROM,
        [OP_EXPORT]             = &&L_OP_EXPORT,
        [OP_ASSERT_FAIL]        = &&L_OP_ASSERT_FAIL,
        [OP_TYPE_GUARD]         = &&L_OP_TYPE_GUARD,
        [OP_HALT]               = &&L_OP_HALT,
        [OP_GET_FIELD_LOCAL]    = &&L_OP_GET_FIELD_LOCAL,
        [OP_FOR_ITER_BIND]      = &&L_OP_FOR_ITER_BIND,
        [OP_JUMP_IF_CMP_FALSE]  = &&L_OP_JUMP_IF_CMP_FALSE,
        [OP_FORMAT]             = &&L_OP_FORMAT,
        [OP_JUMP_IF_CMP_LOCAL_K] = &&L_OP_JUMP_IF_CMP_LOCAL_K,
        [OP_TO_FLOAT]           = &&L_OP_TO_FLOAT,
    };
    /* A designated initialiser leaves a hole as NULL rather than failing to
     * compile, so the count is what catches an opcode added without a case. */
    _Static_assert(sizeof(jaiDispatchTable) / sizeof(jaiDispatchTable[0]) ==
                       OP_COUNT,
                   "dispatch table is out of sync with enum OpCode");
#endif

    CallFrame *frame;
    uint8_t *ip;
    uint8_t *instStart;
    Value *stackTop;
    Value *slots;
    Value *constants;
    Value retval = NULL_VAL;
    /* vm.countInstructions is written once, by the CLI, before anything runs.
     * Read from the struct it is an adrp, a load, a compare and a branch on
     * every single dispatch -- four of the fourteen instructions OP_NOP costs.
     * In a local it is a load off the frame and a cbnz. */
    const bool countInsts = vm.countInstructions;

    LOAD_STATE();

    VM_DISPATCH() {

    /* --- constants and stack (spec §3.1) --- */

    VM_CASE(OP_NOP):
        VM_NEXT();

    VM_CASE(OP_CONST):
        PUSH(READ_CONST());
        VM_NEXT();

    VM_CASE(OP_NULL):
        PUSH(NULL_VAL);
        VM_NEXT();

    VM_CASE(OP_TRUE):
        PUSH(BOOL_VAL(true));
        VM_NEXT();

    VM_CASE(OP_FALSE):
        PUSH(BOOL_VAL(false));
        VM_NEXT();

    VM_CASE(OP_INT):
        PUSH(INT_VAL(READ_I16()));
        VM_NEXT();

    VM_CASE(OP_POP):
        DROP(1);
        VM_NEXT_HINT(OP_LOOP);

    VM_CASE(OP_POPN):
        DROP(READ_BYTE());
        VM_NEXT();

    VM_CASE(OP_DUP): {
        Value top = PEEK(0);
        PUSH(top);
        VM_NEXT();
    }

    VM_CASE(OP_DUP2): {
        Value a = PEEK(1), b = PEEK(0);
        PUSH(a);
        PUSH(b);
        VM_NEXT();
    }

    VM_CASE(OP_SWAP): {
        Value top = stackTop[-1];
        stackTop[-1] = stackTop[-2];
        stackTop[-2] = top;
        VM_NEXT();
    }

    VM_CASE(OP_SWAP_POP): {
        stackTop[-2] = stackTop[-1];
        stackTop--;
        VM_NEXT();
    }

    VM_CASE(OP_ROT3): {
        /* a b c -> c a b */
        Value c = stackTop[-1], b = stackTop[-2], a = stackTop[-3];
        stackTop[-3] = c;
        stackTop[-2] = a;
        stackTop[-1] = b;
        VM_NEXT();
    }

    /* --- variables (spec §3.2) --- */

    VM_CASE(OP_GET_LOCAL):
        PUSH(slots[READ_U16()]);
        VM_NEXT();

    VM_CASE(OP_SET_LOCAL): {
        uint16_t slot = READ_U16();
        slots[slot] = PEEK(0);
        VM_NEXT();
    }

    VM_CASE(OP_GET_UPVALUE): {
        uint8_t index = READ_BYTE();
        ObjUpvalue *upvalue = frame->closure->upvalues[index];
        PUSH(upvalue != NULL ? *upvalue->location : NULL_VAL);
        VM_NEXT();
    }

    VM_CASE(OP_SET_UPVALUE): {
        uint8_t index = READ_BYTE();
        ObjUpvalue *upvalue = frame->closure->upvalues[index];
        if (upvalue != NULL) *upvalue->location = PEEK(0);
        VM_NEXT();
    }

    /* Locals live in a slot window, not on the operand stack, so the scope
     * being closed is named by its lowest slot: everything at or above it
     * belongs to that scope. Emitted at each iteration boundary of a loop
     * whose body has a by-reference capture, so the next iteration's
     * OP_CLOSURE allocates a fresh cell (spec §5.2). No SAVE_STATE: closing
     * neither allocates nor throws, and it never reads vm.stackTop. */
    VM_CASE(OP_CLOSE_UPVALUE): {
        uint16_t slot = READ_U16();
        closeUpvalues(frame->slots + slot);
        VM_NEXT();
    }

    VM_CASE(OP_GET_GLOBAL): {
        uint32_t nameIdx = READ_U24();
        uint16_t cacheIdx = READ_U16();
        ObjModule *module = frame->module;
        InlineCache *ic = cacheAt(frameChunk(frame), cacheIdx);

        /* Validated by IDENTITY, not by a version.
         *
         * Global names are interned -- the globals table itself probes by
         * pointer (findEntryInterned) -- so the entry that holds this very
         * ObjString IS this name's binding, whatever has happened to the table
         * since the index was cached. That covers every way the index can go
         * stale at once, with no counter to keep in step:
         *   - a rehash moved things: the slot holds a different key, or none;
         *   - the name was deleted: the key is JAI_TOMBSTONE, VAL_OBJ carrying
         *     a NULL Obj*, which the old `!IS_NULL(entry->key)` test let
         *     through and then returned NULL_VAL instead of raising NameError;
         *   - the chunk is running against a module other than the one it was
         *     cached against (pushFrame falls back to the caller's module for
         *     a function that has none of its own).
         * And, the point of it: assigning to a global that already exists
         * changes JaiEntry::value and nothing else, so it can no longer miss. */
        Value nameVal = constants[nameIdx];
        if (ic != NULL && ic->state == IC_MONO && module != NULL) {
            int index = (int)ic->payload[0];
            if (index >= 0 && index < module->globals.capacity) {
                JaiEntry *entry = &module->globals.entries[index];
                if (IS_OBJ(entry->key) && AS_OBJ(entry->key) == AS_OBJ(nameVal)) {
                    vm.icHits++;
                    PUSH(entry->value);
                    VM_NEXT();
                }
            }
        }

        ObjString *name = AS_STRING(nameVal);
        if (module != NULL) {
            int index = jaiTableFindIndex(&module->globals, OBJ_VAL(name));
            if (index >= 0) {
                vm.icMisses++;
                if (ic != NULL) {
                    ic->state = IC_MONO;
                    ic->count = 1;
                    ic->shapeId[0] = 0;   /* unused: the key check validates */
                    ic->payload[0] = (uint32_t)index;
                    ic->cached[0] = NULL_VAL;
                }
                PUSH(module->globals.entries[index].value);
                VM_NEXT();
            }
        }
        Value value;
        if (vm.builtins != NULL && jaiModuleGet(vm.builtins, name, &value)) {
            PUSH(value);
            VM_NEXT();
        }
        THROW(vm.cNameError, "undefined name '%s'", name->chars);
    }

    VM_CASE(OP_SET_GLOBAL): {
        uint32_t nameIdx = READ_U24();
        (void)READ_U16();                 /* the cache is read-side only */
        ObjString *name = AS_STRING(constants[nameIdx]);
        ObjModule *module = frame->module;
        Value existing;
        if (module == NULL || !jaiModuleGet(module, name, &existing)) {
            THROW(vm.cNameError, "undefined name '%s'", name->chars);
        }
        SAVE_STATE();
        jaiModuleSet(module, name, PEEK(0));
        VM_NEXT();
    }

    VM_CASE(OP_DEF_GLOBAL): {
        ObjString *name = AS_STRING(READ_CONST());
        SAVE_STATE();
        if (frame->module == NULL) {
            THROW(vm.cRuntimeError, "no module in scope to define '%s' in",
                  name->chars);
        }
        jaiModuleSet(frame->module, name, PEEK(0));
        DROP(1);
        VM_NEXT();
    }

    VM_CASE(OP_GET_MODULE): {
        Value moduleRef = READ_CONST();
        ObjString *member = AS_STRING(READ_CONST());
        SAVE_STATE();

        ObjModule *module = NULL;
        if (IS_MODULE(moduleRef)) {
            module = AS_MODULE(moduleRef);
        } else if (IS_STRING(moduleRef)) {
            Value found;
            if (frame->module != NULL &&
                jaiModuleGet(frame->module, AS_STRING(moduleRef), &found) &&
                IS_MODULE(found)) {
                module = AS_MODULE(found);
            } else if (jaiTableGetInterned(&vm.modules, AS_STRING(moduleRef),
                                           &found) && IS_MODULE(found)) {
                module = AS_MODULE(found);
            }
        }
        if (module == NULL) {
            THROW(vm.cNameError, "unknown module in member access '%s'",
                  member->chars);
        }
        Value value;
        if (!getPropertyInto(OBJ_VAL(module), member, &value, true, NULL)) {
            goto vmThrow;
        }
        LOAD_STATE();
        PUSH(value);
        VM_NEXT();
    }

    /* --- arithmetic and logic (spec §3.3) --- */

    VM_CASE(OP_ADD): {
        if (IS_INT(stackTop[-1]) && IS_INT(stackTop[-2])) {
            int64_t r;
            if (JAI_UNLIKELY(__builtin_add_overflow(AS_INT(stackTop[-2]),
                                                    AS_INT(stackTop[-1]), &r))) {
                THROW(vm.cOverflowError,
                      "integer overflow in '+'; use '+%%' to wrap");
            }
            DROP(1);
            stackTop[-1] = INT_VAL(r);
            VM_NEXT();
        }
        if (IS_FLOAT(stackTop[-1]) && IS_FLOAT(stackTop[-2])) {
            double r = AS_FLOAT(stackTop[-2]) + AS_FLOAT(stackTop[-1]);
            DROP(1);
            stackTop[-1] = FLOAT_VAL(r);
            VM_NEXT();
        }
        BINARY(arithmetic, OP_ADD);
    }

    VM_CASE(OP_SUB): {
        if (IS_INT(stackTop[-1]) && IS_INT(stackTop[-2])) {
            int64_t r;
            if (JAI_UNLIKELY(__builtin_sub_overflow(AS_INT(stackTop[-2]),
                                                    AS_INT(stackTop[-1]), &r))) {
                THROW(vm.cOverflowError,
                      "integer overflow in '-'; use '-%%' to wrap");
            }
            DROP(1);
            stackTop[-1] = INT_VAL(r);
            VM_NEXT();
        }
        if (IS_FLOAT(stackTop[-1]) && IS_FLOAT(stackTop[-2])) {
            double r = AS_FLOAT(stackTop[-2]) - AS_FLOAT(stackTop[-1]);
            DROP(1);
            stackTop[-1] = FLOAT_VAL(r);
            VM_NEXT();
        }
        BINARY(arithmetic, OP_SUB);
    }

    VM_CASE(OP_MUL): {
        if (IS_INT(stackTop[-1]) && IS_INT(stackTop[-2])) {
            int64_t r;
            if (JAI_UNLIKELY(__builtin_mul_overflow(AS_INT(stackTop[-2]),
                                                    AS_INT(stackTop[-1]), &r))) {
                THROW(vm.cOverflowError,
                      "integer overflow in '*'; use '*%%' to wrap");
            }
            DROP(1);
            stackTop[-1] = INT_VAL(r);
            VM_NEXT();
        }
        if (IS_FLOAT(stackTop[-1]) && IS_FLOAT(stackTop[-2])) {
            double r = AS_FLOAT(stackTop[-2]) * AS_FLOAT(stackTop[-1]);
            DROP(1);
            stackTop[-1] = FLOAT_VAL(r);
            VM_NEXT();
        }
        BINARY(arithmetic, OP_MUL);
    }

    VM_CASE(OP_DIV):       BINARY(arithmetic, OP_DIV);
    VM_CASE(OP_FLOORDIV):  BINARY(arithmetic, OP_FLOORDIV);

    VM_CASE(OP_MOD): {
        /* Floor remainder, inlined for int %% int: it is 8.3% of loop_sum and
         * reaching it through `arithmetic` cost that benchmark 13% of its run.
         * y == 0 and INT64_MIN %% -1 (C UB) go the slow way, which reports the
         * division-by-zero and computes the 0. */
        if (JAI_LIKELY(IS_INT(stackTop[-1]) && IS_INT(stackTop[-2]))) {
            int64_t y = AS_INT(stackTop[-1]), x = AS_INT(stackTop[-2]);
            if (JAI_LIKELY(y != 0 && !(x == INT64_MIN && y == -1))) {
                int64_t r = x % y;
                if (r != 0 && ((r < 0) != (y < 0))) r += y;
                DROP(1);
                stackTop[-1] = INT_VAL(r);
                VM_NEXT();
            }
        }
        BINARY(arithmetic, OP_MOD);
    }

    VM_CASE(OP_POW):       BINARY(arithmetic, OP_POW);
    VM_CASE(OP_ADD_WRAP):  BINARY(arithmetic, OP_ADD_WRAP);
    VM_CASE(OP_SUB_WRAP):  BINARY(arithmetic, OP_SUB_WRAP);
    VM_CASE(OP_MUL_WRAP):  BINARY(arithmetic, OP_MUL_WRAP);
    VM_CASE(OP_CONCAT):    BINARY(arithmetic, OP_CONCAT);
    VM_CASE(OP_BAND):      BINARY(bitwise, OP_BAND);
    VM_CASE(OP_BOR):       BINARY(bitwise, OP_BOR);
    VM_CASE(OP_BXOR):      BINARY(bitwise, OP_BXOR);
    VM_CASE(OP_SHL):       BINARY(bitwise, OP_SHL);
    VM_CASE(OP_SHR):       BINARY(bitwise, OP_SHR);
    VM_CASE(OP_LT): CMP_FAST(<);  BINARY(compareOp, OP_LT);
    VM_CASE(OP_LE): CMP_FAST(<=); BINARY(compareOp, OP_LE);
    VM_CASE(OP_GT): CMP_FAST(>);  BINARY(compareOp, OP_GT);
    VM_CASE(OP_GE): CMP_FAST(>=); BINARY(compareOp, OP_GE);

    VM_CASE(OP_NEG): {
        if (IS_INT(stackTop[-1]) && AS_INT(stackTop[-1]) != INT64_MIN) {
            stackTop[-1] = INT_VAL(-AS_INT(stackTop[-1]));
            VM_NEXT();
        }
        if (IS_FLOAT(stackTop[-1])) {
            stackTop[-1] = FLOAT_VAL(-AS_FLOAT(stackTop[-1]));
            VM_NEXT();
        }
        SAVE_STATE();
        Value result;
        if (!unaryNegate(stackTop[-1], &result)) goto vmThrow;
        LOAD_STATE();
        stackTop[-1] = result;
        VM_NEXT();
    }

    VM_CASE(OP_POS):
        if (!IS_NUMBER(PEEK(0))) {
            THROW(vm.cTypeError, "unary '+' is not supported for '%s'",
                  jaiTypeNameStatic(PEEK(0)));
        }
        VM_NEXT();

    VM_CASE(OP_BNOT):
        if (!IS_INT(PEEK(0))) {
            THROW(vm.cTypeError, "'~' requires an int, not '%s'",
                  jaiTypeNameStatic(PEEK(0)));
        }
        stackTop[-1] = INT_VAL(~AS_INT(stackTop[-1]));
        VM_NEXT();

    VM_CASE(OP_EQ):
    VM_CASE(OP_NE): {
        bool wantEqual = (instStart[0] == OP_EQ);
        bool fast;
        if (JAI_LIKELY(valuesEqualFast(stackTop[-2], stackTop[-1], &fast))) {
            DROP(2);
            PUSH(BOOL_VAL(fast == wantEqual));
            VM_NEXT();
        }
        SAVE_STATE();
        bool equal = jaiValuesEqual(stackTop[-2], stackTop[-1]);
        if (vm.hasException) goto vmThrow;
        LOAD_STATE();
        DROP(2);
        PUSH(BOOL_VAL(equal == wantEqual));
        VM_NEXT();
    }

    VM_CASE(OP_IS): {
        bool same = valueIsTest(stackTop[-2], stackTop[-1]);
        DROP(2);
        PUSH(BOOL_VAL(same));
        VM_NEXT();
    }

    VM_CASE(OP_IS_NOT): {
        bool same = valueIsTest(stackTop[-2], stackTop[-1]);
        DROP(2);
        PUSH(BOOL_VAL(!same));
        VM_NEXT();
    }

    VM_CASE(OP_IN):
    VM_CASE(OP_NOT_IN): {
        bool wantIn = (instStart[0] == OP_IN);
        SAVE_STATE();
        bool contains = false;
        if (!jaiContainsOp(stackTop[-1], stackTop[-2], &contains)) goto vmThrow;
        LOAD_STATE();
        DROP(2);
        PUSH(BOOL_VAL(contains == wantIn));
        VM_NEXT();
    }

    VM_CASE(OP_NOT):
        REQUIRE_BOOL(PEEK(0), "operand of 'not'");
        stackTop[-1] = BOOL_VAL(!AS_BOOL(stackTop[-1]));
        VM_NEXT();

    /* The checker emits this only where it proved the value is an int, so the
     * float case is not a coercion but the harmless identity that keeps a
     * hand-written or optimised chunk from throwing. Above 2^53 the double
     * cannot hold the integer exactly and the low bits are lost, which is what
     * every language with this rule does and what spec §2.5 records. */
    VM_CASE(OP_TO_FLOAT):
        if (IS_INT(PEEK(0))) {
            stackTop[-1] = FLOAT_VAL((double)AS_INT(stackTop[-1]));
        } else if (!IS_FLOAT(PEEK(0))) {
            THROW(vm.cTypeError, "expected 'int' but got '%s'",
                  jaiTypeNameStatic(PEEK(0)));
        }
        VM_NEXT();

    /* --- peephole-fused forms ---
     *
     * Each is exactly the sequence it replaced (spec §3.3), so the int case is
     * a fast path and never a restriction: the peephole reads bytecode and has
     * no types, so it fuses reads of an `any` local just as readily as of an
     * `int` one. Anything but two ints goes to the same helper the unfused
     * opcode would have called — a float compares, a str concatenates, a class
     * gets its dunder, and the diagnostic on a real mismatch is the one the
     * user would have seen at -O0. */

    VM_CASE(OP_ADD_BIND): {
        /* `ADD; BIND a` fused: the int path stores straight into the slot,
         * skipping the push-then-pop the pair performed. */
        uint16_t slot = READ_U16();
        if (JAI_LIKELY(IS_INT(stackTop[-1]) && IS_INT(stackTop[-2]))) {
            int64_t r;
            if (JAI_LIKELY(!__builtin_add_overflow(AS_INT(stackTop[-2]),
                                                   AS_INT(stackTop[-1]), &r))) {
                DROP(2);
                slots[slot] = INT_VAL(r);
                VM_NEXT();
            }
        }
        SAVE_STATE();
        Value sum;
        if (!arithmetic(OP_ADD, stackTop[-2], stackTop[-1], &sum)) goto vmThrow;
        LOAD_STATE();
        DROP(2);
        slots[slot] = sum;
        VM_NEXT();
    }

    VM_CASE(OP_SUB_BIND): {
        uint16_t slot = READ_U16();
        if (JAI_LIKELY(IS_INT(stackTop[-1]) && IS_INT(stackTop[-2]))) {
            int64_t r;
            if (JAI_LIKELY(!__builtin_sub_overflow(AS_INT(stackTop[-2]),
                                                   AS_INT(stackTop[-1]), &r))) {
                DROP(2);
                slots[slot] = INT_VAL(r);
                VM_NEXT();
            }
        }
        SAVE_STATE();
        Value out;
        if (!arithmetic(OP_SUB, stackTop[-2], stackTop[-1], &out)) goto vmThrow;
        LOAD_STATE();
        DROP(2);
        slots[slot] = out;
        VM_NEXT();
    }

    /* Stamp `list[T]` / `dict[K, V]` onto the container the literal just built,
     * so the mutation guards below have something to check. Peeks rather than
     * pops: the container is still the expression's value. */
    VM_CASE(OP_ELEM_KIND): {
        uint8_t packed = READ_BYTE();
        Value target = PEEK(0);
        if (IS_LIST(target)) {
            AS_LIST(target)->elemKind = (uint8_t)(packed & 0xFu);
            /* The literal is still empty here -- the stamp is emitted the
             * instant it is built -- which is the only moment the backing
             * array can change width for free. */
            jaiListSpecialise(AS_LIST(target), (uint8_t)(packed & 0xFu));
        } else if (IS_DICT(target)) {
            AS_DICT(target)->keyKind = (uint8_t)((packed >> 4) & 0xFu);
            AS_DICT(target)->valKind = (uint8_t)(packed & 0xFu);
        }
        /* Anything else: the annotation named a shape this does not model, and
         * an unstamped container is simply unguarded. */
        VM_NEXT();
    }

    VM_CASE(OP_MUL_BIND): {
        uint16_t slot = READ_U16();
        if (JAI_LIKELY(IS_INT(stackTop[-1]) && IS_INT(stackTop[-2]))) {
            int64_t r;
            if (JAI_LIKELY(!__builtin_mul_overflow(AS_INT(stackTop[-2]),
                                                   AS_INT(stackTop[-1]), &r))) {
                DROP(2);
                slots[slot] = INT_VAL(r);
                VM_NEXT();
            }
        }
        SAVE_STATE();
        Value out;
        if (!arithmetic(OP_MUL, stackTop[-2], stackTop[-1], &out)) goto vmThrow;
        LOAD_STATE();
        DROP(2);
        slots[slot] = out;
        VM_NEXT();
    }

    VM_CASE(OP_MOD_INT_CONST): {
        /* `<int k>; MOD` fused. The same floor-remainder rule as OP_MOD with an
         * immediate divisor; k is known non-zero at fusion time, so only
         * INT64_MIN % -1 has to reach the slow path. */
        int16_t imm = READ_I16();
        if (JAI_LIKELY(IS_INT(stackTop[-1]))) {
            int64_t y = (int64_t)imm, x = AS_INT(stackTop[-1]);
            if (JAI_LIKELY(!(x == INT64_MIN && y == -1))) {
                int64_t r = x % y;
                if (r != 0 && ((r < 0) != (y < 0))) r += y;
                stackTop[-1] = INT_VAL(r);
                VM_NEXT();
            }
        }
        SAVE_STATE();
        Value result;
        if (!arithmetic(OP_MOD, stackTop[-1], INT_VAL(imm), &result)) goto vmThrow;
        LOAD_STATE();
        stackTop[-1] = result;
        VM_NEXT();
    }

    VM_CASE(OP_ADD_INT_CONST): {
        uint16_t slot = READ_U16();
        int16_t imm = READ_I16();
        Value local = slots[slot];
        if (JAI_LIKELY(IS_INT(local))) {
            int64_t r;
            if (JAI_UNLIKELY(__builtin_add_overflow(AS_INT(local), (int64_t)imm, &r))) {
                THROW(vm.cOverflowError, "integer overflow in '+'; use '+%%' to wrap");
            }
            PUSH(INT_VAL(r));
            VM_NEXT();
        }
        SAVE_STATE();
        Value result;
        if (!arithmetic(OP_ADD, local, INT_VAL(imm), &result)) goto vmThrow;
        LOAD_STATE();
        PUSH(result);
        VM_NEXT();
    }

    VM_CASE(OP_SUB_INT_CONST): {
        uint16_t slot = READ_U16();
        int16_t imm = READ_I16();
        Value local = slots[slot];
        if (JAI_LIKELY(IS_INT(local))) {
            int64_t r;
            if (JAI_UNLIKELY(__builtin_sub_overflow(AS_INT(local),
                                                    (int64_t)imm, &r))) {
                THROW(vm.cOverflowError,
                      "integer overflow in '-'; use '-%%' to wrap");
            }
            PUSH(INT_VAL(r));
            VM_NEXT();
        }
        SAVE_STATE();
        Value result;
        if (!arithmetic(OP_SUB, local, INT_VAL(imm), &result)) goto vmThrow;
        LOAD_STATE();
        PUSH(result);
        VM_NEXT();
    }

    VM_CASE(OP_MUL_INT_CONST): {
        uint16_t slot = READ_U16();
        int16_t imm = READ_I16();
        Value local = slots[slot];
        if (JAI_LIKELY(IS_INT(local))) {
            int64_t r;
            if (JAI_UNLIKELY(__builtin_mul_overflow(AS_INT(local), (int64_t)imm, &r))) {
                THROW(vm.cOverflowError, "integer overflow in '*'; use '*%%' to wrap");
            }
            PUSH(INT_VAL(r));
            VM_NEXT();
        }
        SAVE_STATE();
        Value result;
        if (!arithmetic(OP_MUL, local, INT_VAL(imm), &result)) goto vmThrow;
        LOAD_STATE();
        PUSH(result);
        VM_NEXT();
    }

    VM_CASE(OP_INC_LOCAL): {
        uint16_t slot = READ_U16();
        int8_t imm = READ_I8();
        Value local = slots[slot];
        if (JAI_LIKELY(IS_INT(local))) {
            int64_t r;
            if (JAI_UNLIKELY(__builtin_add_overflow(AS_INT(local), (int64_t)imm, &r))) {
                THROW(vm.cOverflowError, "integer overflow in '+'; use '+%%' to wrap");
            }
            slots[slot] = INT_VAL(r);
            VM_NEXT_HINT(OP_LOOP);
        }
        SAVE_STATE();
        Value result;
        if (!arithmetic(OP_ADD, local, INT_VAL(imm), &result)) goto vmThrow;
        LOAD_STATE();
        /* `slots` is reloaded above: the generic path can call into Jaithon and
         * grow the value stack, which moves every frame's window. */
        slots[slot] = result;
        VM_NEXT();
    }

    VM_CASE(OP_CMP_LOCAL_CONST_LT): {
        uint16_t slot = READ_U16();
        int16_t imm = READ_I16();
        Value local = slots[slot];
        if (JAI_LIKELY(IS_INT(local))) {
            PUSH(BOOL_VAL(AS_INT(local) < (int64_t)imm));
            VM_NEXT();
        }
        SAVE_STATE();
        Value result;
        if (!compareOp(OP_LT, local, INT_VAL(imm), &result)) goto vmThrow;
        LOAD_STATE();
        PUSH(result);
        VM_NEXT();
    }

    VM_CASE(OP_GET_LOCAL2): {
        uint16_t a = READ_U16(), b = READ_U16();
        PUSH(slots[a]);
        PUSH(slots[b]);
        VM_NEXT();
    }

    VM_CASE(OP_ADD_LOCALS): {
        uint16_t a = READ_U16(), b = READ_U16();
        Value x = slots[a], y = slots[b];
        if (IS_INT(x) && IS_INT(y)) {
            int64_t r;
            if (JAI_UNLIKELY(__builtin_add_overflow(AS_INT(x), AS_INT(y), &r))) {
                THROW(vm.cOverflowError,
                      "integer overflow in '+'; use '+%%' to wrap");
            }
            PUSH(INT_VAL(r));
            VM_NEXT();
        }
        SAVE_STATE();
        Value result;
        if (!arithmetic(OP_ADD, x, y, &result)) goto vmThrow;
        LOAD_STATE();
        PUSH(result);
        VM_NEXT();
    }

    /* --- control flow (spec §3.4) --- */

    VM_CASE(OP_JUMP): {
        int16_t offset = READ_I16();
        ip += offset;
        VM_NEXT();
    }

    VM_CASE(OP_JUMP_IF_FALSE): {
        int16_t offset = READ_I16();
        Value condition = POP();
        REQUIRE_BOOL(condition, "condition");
        if (!AS_BOOL(condition)) ip += offset;
        VM_NEXT_HINT(OP_INC_LOCAL);
    }

    VM_CASE(OP_JUMP_IF_TRUE): {
        int16_t offset = READ_I16();
        Value condition = POP();
        REQUIRE_BOOL(condition, "condition");
        if (AS_BOOL(condition)) ip += offset;
        VM_NEXT();
    }

    VM_CASE(OP_JUMP_IF_FALSE_KEEP): {
        int16_t offset = READ_I16();
        REQUIRE_BOOL(PEEK(0), "operand of 'and'");
        if (!AS_BOOL(PEEK(0))) ip += offset;
        VM_NEXT();
    }

    VM_CASE(OP_JUMP_IF_TRUE_KEEP): {
        int16_t offset = READ_I16();
        REQUIRE_BOOL(PEEK(0), "operand of 'or'");
        if (AS_BOOL(PEEK(0))) ip += offset;
        VM_NEXT();
    }

    VM_CASE(OP_JUMP_IF_NULL): {
        int16_t offset = READ_I16();
        if (IS_NULL(PEEK(0))) ip += offset;
        VM_NEXT();
    }

    /* <cmp>; JUMP_IF_FALSE fused (spec §3.3). The bool never reaches the stack,
     * and with it goes JUMP_IF_FALSE's REQUIRE_BOOL: every producer this pass
     * accepts is bool by construction, which is the same fact `pushesBool` in
     * optimize.c already relies on to fold OP_NOT into a jump. Only int/int is
     * inlined; float and everything else take the identical slow path the
     * unfused pair would have, so NaN and __lt__ behave exactly as before. */
    VM_CASE(OP_JUMP_IF_CMP_FALSE): {
        uint8_t cmp = READ_BYTE();
        int16_t offset = READ_I16();
        Value a = stackTop[-2], b = stackTop[-1];

        if (JAI_LIKELY(IS_INT(a) && IS_INT(b))) {
            int64_t x = AS_INT(a), y = AS_INT(b);
            bool taken;
            switch (cmp) {
            case OP_EQ: taken = x == y; break;
            case OP_NE: taken = x != y; break;
            case OP_LT: taken = x <  y; break;
            case OP_LE: taken = x <= y; break;
            case OP_GT: taken = x >  y; break;
            case OP_GE: taken = x >= y; break;
            default:
                THROW(vm.cRuntimeError, "JUMP_IF_CMP_FALSE has opcode %u, "
                      "which is not a comparison", (unsigned)cmp);
            }
            stackTop -= 2;
            if (!taken) ip += offset;
            VM_NEXT();
        }

        if (cmp == OP_EQ || cmp == OP_NE) {
            bool fast;
            if (JAI_LIKELY(valuesEqualFast(a, b, &fast))) {
                stackTop -= 2;
                if (fast != (cmp == OP_EQ)) ip += offset;
                VM_NEXT();
            }
        }

        SAVE_STATE();
        bool condition;
        if (cmp == OP_EQ || cmp == OP_NE) {
            bool equal = jaiValuesEqual(a, b);
            if (vm.hasException) goto vmThrow;
            condition = equal == (cmp == OP_EQ);
        } else {
            Value result;
            if (!compareOp((OpCode)cmp, a, b, &result)) goto vmThrow;
            condition = AS_BOOL(result);
        }
        LOAD_STATE();
        stackTop -= 2;
        if (!condition) ip += offset;
        VM_NEXT();
    }

    /* GET_LOCAL S; CONST K; JUMP_IF_CMP_FALSE cmp fused (spec §3.3): a loop
     * guard or a recursion base case in one instruction. Neither operand is
     * ever pushed, so unlike JUMP_IF_CMP_FALSE there is no stack to unwind on
     * either path — but both are still rooted while the slow path runs, the
     * local by the frame it lives in and the constant by the chunk. */
    VM_CASE(OP_JUMP_IF_CMP_LOCAL_K): {
        uint8_t cmp = READ_BYTE();
        Value a = slots[READ_U16()];
        Value b = constants[READ_U24()];
        int16_t offset = READ_I16();

        if (JAI_LIKELY(IS_INT(a) && IS_INT(b))) {
            int64_t x = AS_INT(a), y = AS_INT(b);
            bool taken;
            switch (cmp) {
            case OP_EQ: taken = x == y; break;
            case OP_NE: taken = x != y; break;
            case OP_LT: taken = x <  y; break;
            case OP_LE: taken = x <= y; break;
            case OP_GT: taken = x >  y; break;
            case OP_GE: taken = x >= y; break;
            default:
                THROW(vm.cRuntimeError, "JUMP_IF_CMP_LOCAL_K has opcode %u, "
                      "which is not a comparison", (unsigned)cmp);
            }
            if (!taken) ip += offset;
            VM_NEXT();
        }

        if (cmp == OP_EQ || cmp == OP_NE) {
            bool fast;
            if (JAI_LIKELY(valuesEqualFast(a, b, &fast))) {
                if (fast != (cmp == OP_EQ)) ip += offset;
                VM_NEXT();
            }
        }

        SAVE_STATE();
        bool condition;
        if (cmp == OP_EQ || cmp == OP_NE) {
            bool equal = jaiValuesEqual(a, b);
            if (vm.hasException) goto vmThrow;
            condition = equal == (cmp == OP_EQ);
        } else {
            Value result;
            if (!compareOp((OpCode)cmp, a, b, &result)) goto vmThrow;
            condition = AS_BOOL(result);
        }
        LOAD_STATE();
        if (!condition) ip += offset;
        VM_NEXT();
    }

    /* FOR_ITER J; BIND S fused (spec §3.3): `for x in xs` in one instruction.
     * The produced item goes straight to its slot instead of being pushed and
     * popped back off. */
    VM_CASE(OP_FOR_ITER_BIND): {
        int16_t offset = READ_I16();
        uint16_t slot = READ_U16();
        Value iterator = PEEK(0);
        if (!IS_ITER(iterator)) {
            THROW(vm.cTypeError, "for-loop expected an iterator, not '%s'",
                  jaiTypeNameStatic(iterator));
        }
        Value item;
        IterStep step = iterStepFast(AS_ITER(iterator), &item);
        if (JAI_LIKELY(step == ITER_STEP_VALUE)) {
            slots[slot] = item;
            VM_NEXT();
        }
        if (step == ITER_STEP_DONE) {
            DROP(1);
            ip += offset;
            VM_NEXT();
        }
        SAVE_STATE();
        bool advanced = jaiIterNext(AS_ITER(iterator), &item);
        if (!advanced && vm.hasException) goto vmThrow;
        LOAD_STATE();
        if (!advanced) {
            DROP(1);            /* exhausted: the iterator goes with the loop */
            ip += offset;
            VM_NEXT();
        }
        /* `slots` is reloaded above: jaiIterNext can re-enter the VM through a
         * user __next__ and grow the value stack, moving every frame window. */
        slots[slot] = item;
        VM_NEXT();
    }

    /* FOR_ITER J; UNPACK 2 255; BIND A; BIND B fused (spec §3.3):
     * `for (a, b) in …` in one instruction and, on a dict, with no pair object
     * built at all. See OP_FOR_ITER_PAIR in chunk.h for what that is worth. */
    VM_CASE(OP_FOR_ITER_PAIR): {
        int16_t  offset = READ_I16();
        uint16_t slotA  = READ_U16();
        uint16_t slotB  = READ_U16();
        Value iterator = PEEK(0);
        if (!IS_ITER(iterator)) {
            THROW(vm.cTypeError, "for-loop expected an iterator, not '%s'",
                  jaiTypeNameStatic(iterator));
        }
        Value a, b;
        PairStep step = iterStepPairFast(AS_ITER(iterator), &a, &b);
        if (JAI_LIKELY(step == PAIR_STEP_VALUE)) {
            slots[slotA] = a;
            slots[slotB] = b;
            VM_NEXT();
        }
        if (step == PAIR_STEP_DONE) {
            DROP(1);
            ip += offset;
            VM_NEXT();
        }
        if (step == PAIR_STEP_BAD) {
            SAVE_STATE();
            (void)pairSplitFail(a);
            goto vmThrow;
        }
        SAVE_STATE();
        Value item;
        bool advanced = jaiIterNext(AS_ITER(iterator), &item);
        if (!advanced && vm.hasException) goto vmThrow;
        LOAD_STATE();
        if (!advanced) {
            DROP(1);            /* exhausted: the iterator goes with the loop */
            ip += offset;
            VM_NEXT();
        }
        if (!pairSplit(item, &a, &b)) {
            SAVE_STATE();
            (void)pairSplitFail(item);
            goto vmThrow;
        }
        /* `slots` is reloaded above: jaiIterNext can re-enter the VM through a
         * user __next__ and grow the value stack, moving every frame window. */
        slots[slotA] = a;
        slots[slotB] = b;
        VM_NEXT();
    }

    /* `for x in a..b`, opened (spec §3.4). Where BUILD_RANGE; GET_ITER built an
     * ObjRange and an ObjIter per loop ENTRY, this writes the loop's whole
     * state into two int frame slots and allocates nothing at all.
     *
     * `end` is one past the last value, computed WRAPPING, so the counter test
     * downstream is `!=` rather than `<` and `a..=INT64_MAX` still terminates:
     * the counter wraps to INT64_MIN and meets it there. See OP_ITER_RANGE in
     * chunk.h for the single range where that disagrees with ObjIter's
     * saturating limit, and why it does not matter. */
    VM_CASE(OP_ITER_RANGE): {
        bool     inclusive = READ_BYTE() != 0;
        uint16_t curSlot   = READ_U16();
        uint16_t endSlot   = READ_U16();
        Value stopValue = PEEK(0), startValue = PEEK(1);
        /* The same check, and the same wording, OP_BUILD_RANGE makes: this is a
         * fast path for one loop shape, never a second set of rules. */
        if (!IS_INT(startValue) || !IS_INT(stopValue)) {
            THROW(vm.cTypeError, "range bounds must be int, not '%s' and '%s'",
                  jaiTypeNameStatic(startValue), jaiTypeNameStatic(stopValue));
        }
        int64_t start = AS_INT(startValue), stop = AS_INT(stopValue);
        int64_t end = stop < start
                          ? start          /* empty: end where it begins */
                          : (inclusive
                                 ? (int64_t)((uint64_t)stop + 1u)
                                 : stop);
        DROP(2);
        slots[curSlot] = INT_VAL(start);
        slots[endSlot] = INT_VAL(end);
        VM_NEXT();
    }

    /* One step of that loop (spec §3.4). Both slots were written by the
     * OP_ITER_RANGE above and by this instruction and by nothing else — the
     * emitter hands out fresh temporaries for them — so the payloads are read
     * without re-testing a tag the loop itself is the only writer of. */
    VM_CASE(OP_FOR_RANGE_BIND): {
        int16_t  offset  = READ_I16();
        uint16_t slot    = READ_U16();
        uint16_t curSlot = READ_U16();
        uint16_t endSlot = READ_U16();
        int64_t cur = AS_INT(slots[curSlot]);
        if (cur == AS_INT(slots[endSlot])) {
            ip += offset;
            VM_NEXT();
        }
        slots[slot]    = INT_VAL(cur);
        slots[curSlot] = INT_VAL((int64_t)((uint64_t)cur + 1u));
        VM_NEXT();
    }

    /* An f-string, whole (spec §3.6). The parts are on the stack in order and
     * the result replaces them.
     *
     * The lowering this replaces was, per hole, a global lookup of `str`, a
     * native call and an OP_CONCAT: two heap strings, two hashes and two
     * intern probes to interpolate one integer. jaiValueFormat measures the
     * parts and allocates once.
     *
     * The cache answers one question — is `str` still the builtin? — and is
     * keyed on the globals table's KEY version, which moves when a name is
     * added or removed and not when one is merely assigned to. That is exactly
     * the fact being memoised, so a module-scope loop that writes globals no
     * longer throws this cache away on every iteration. */
    VM_CASE(OP_FORMAT): {
        uint8_t  count    = READ_BYTE();
        uint32_t litmask  = READ_U24();
        uint32_t nameIdx  = READ_U24();
        uint16_t cacheIdx = READ_U16();

        ObjModule *module = frame->module;
        InlineCache *ic = cacheAt(frameChunk(frame), cacheIdx);
        bool builtin;
        if (JAI_LIKELY(ic != NULL && ic->state == IC_MONO && module != NULL &&
                       ic->shapeId[0] == module->globals.keyVersion &&
                       ic->payload[1] ==
                           (uint32_t)((uintptr_t)module >> 4))) {
            builtin = ic->payload[0] != 0;
        } else {
            ObjString *name = AS_STRING(constants[nameIdx]);
            Value shadow;
            builtin = module == NULL ||
                      !jaiTableGetInterned(&module->globals, name, &shadow);
            if (ic != NULL && module != NULL) {
                ic->state = IC_MONO;
                ic->count = 1;
                ic->shapeId[0] = module->globals.keyVersion;
                /* A chunk whose function has no module of its own runs against
                 * the caller's (pushFrame), so one cache slot can be reached
                 * with two ObjModule*. keyVersion is small and dense, so pin
                 * the module too -- no deref, so no rooting question. */
                ic->payload[1] = (uint32_t)((uintptr_t)module >> 4);
                ic->payload[0] = builtin ? 1u : 0u;
                ic->cached[0] = NULL_VAL;
            }
        }

        SAVE_STATE();
        if (JAI_UNLIKELY(!builtin) &&
            !formatViaUserStr(module, AS_STRING(constants[nameIdx]), count,
                              litmask)) {
            goto vmThrow;
        }
        ObjString *formatted = jaiValueFormat(vm.stackTop - count, count);
        if (formatted == NULL) goto vmThrow;
        LOAD_STATE();
        stackTop -= count;
        PUSH(OBJ_VAL(formatted));
        VM_NEXT_HINT(OP_BIND);
    }

    VM_CASE(OP_LOOP): {
        int16_t offset = READ_I16();
        ip += offset;
        /* Both halves of the safepoint are almost always "nothing to do", so
         * only the taken case pays for writing the frame state back: the
         * common back edge is one predictable branch, not a
         * SAVE_STATE/call/LOAD_STATE round trip. */
        {
            const ObjFunction *lf = frame->closure->fn;
            if (JAI_UNLIKELY(jaiInterrupted || jaiGCWanted() ||
                             lf->osrHot)) {
                SAVE_STATE();
                if (!safepoint()) goto vmThrow;
                LOAD_STATE();
            }
        }
        VM_NEXT();
    }

    VM_CASE(OP_GET_ITER): {
        SAVE_STATE();
        Value iterator;
        if (!jaiGetIter(stackTop[-1], &iterator)) goto vmThrow;
        LOAD_STATE();
        stackTop[-1] = iterator;
        VM_NEXT();
    }

    /* `for … in X.items()`. On a dict, iterate the table directly instead of
     * building the whole list of pairs first: jaiDictItems allocates an
     * N-element list of N fresh 2-tuples per call and keeps all N alive at
     * once, so every collection during the loop marks the lot. The lazy form
     * keeps one tuple alive at a time. Measured on tests/bench/dict_iter:
     * peak RSS 37.1 MB -> 30.3 MB, instructions -22%.
     *
     * Anything else is LEFT ALONE and the ordinary `INVOKE items; GET_ITER`
     * that follows handles it, so a user class defining `items()` still has its
     * own method called. Invoking that method from here instead was the first
     * attempt and --gc-stress rejected it: a Jaithon call needs the frame
     * machinery an opcode body does not have. */
    VM_CASE(OP_GET_ITER_ITEMS): {
        int16_t offset = READ_I16();
        Value target = PEEK(0);
        if (IS_DICT(target)) {
            SAVE_STATE();
            ObjIter *it = jaiIterNew(ITER_DICT_ITEMS, target);
            if (it == NULL) goto vmThrow;
            LOAD_STATE();
            stackTop[-1] = OBJ_VAL(it);
            ip += offset;
        }
        VM_NEXT();
    }

    VM_CASE(OP_FOR_ITER): {
        int16_t offset = READ_I16();
        Value iterator = PEEK(0);
        if (!IS_ITER(iterator)) {
            THROW(vm.cTypeError, "for-loop expected an iterator, not '%s'",
                  jaiTypeNameStatic(iterator));
        }
        Value item;
        IterStep step = iterStepFast(AS_ITER(iterator), &item);
        if (JAI_LIKELY(step == ITER_STEP_VALUE)) {
            PUSH(item);
            VM_NEXT();
        }
        if (step == ITER_STEP_DONE) {
            DROP(1);
            ip += offset;
            VM_NEXT();
        }
        SAVE_STATE();
        bool advanced = jaiIterNext(AS_ITER(iterator), &item);
        if (!advanced && vm.hasException) goto vmThrow;
        LOAD_STATE();
        if (!advanced) {
            DROP(1);            /* exhausted: the iterator goes with the loop */
            ip += offset;
            VM_NEXT();
        }
        PUSH(item);
        VM_NEXT();
    }

    /* --- calls (spec §3.5) --- */

    VM_CASE(OP_CALL): {
        int argc = READ_BYTE();
        SAVE_STATE();
        /* A closure is what almost every call site holds, and reaching
         * callClosure through callValueOnStack costs an out-of-line call and
         * invokeCallable's IS_OBJ test and type switch on the way. A call is
         * 15ns of overhead measured against the same loop written inline, and
         * this is the part of it that buys nothing on the common path.
         *
         * Anything else -- natives, bound methods, classes, functions without a
         * closure -- takes the general path unchanged. */
        Value callee = vm.stackTop[-argc - 1];
        CallOutcome outcome = JAI_LIKELY(IS_CLOSURE(callee))
                                  ? callClosure(AS_CLOSURE(callee), argc)
                                  : callValueOnStack(argc);
        if (outcome == CALL_ERROR) goto vmThrow;
        if (outcome == CALL_DONE) { LOAD_STACK_ONLY(); VM_NEXT(); }
        LOAD_STATE();
        VM_NEXT();
    }

    VM_CASE(OP_CALL_KW): {
        int posCount = READ_BYTE();
        Value nameTuple = READ_CONST();
        SAVE_STATE();
        if (!IS_TUPLE(nameTuple)) {
            THROW(vm.cRuntimeError,
                  "CALL_KW expected a tuple of keyword names");
        }
        ObjDict *kwRest = NULL;
        int argc = prepareKeywordCall(posCount, AS_TUPLE(nameTuple), &kwRest);
        if (argc < 0) goto vmThrow;

        Value callee = vm.stackTop[-argc - 1];
        if (kwRest != NULL) jaiGCPushRoot(OBJ_VAL(kwRest));
        CallOutcome outcome = invokeCallable(callee, argc);
        if (outcome == CALL_ERROR) {
            if (kwRest != NULL) jaiGCPopRoot();
            goto vmThrow;
        }
        if (kwRest != NULL) {
            if (outcome == CALL_FRAME) {
                CallFrame *calleeFrame = &vm.frames[vm.frameCount - 1];
                int slot = kwRestSlotOf(calleeFrame->closure->fn);
                calleeFrame->slots[slot] = OBJ_VAL(kwRest);
            }
            jaiGCPopRoot();
        }
        LOAD_STATE();
        VM_NEXT();
    }

    VM_CASE(OP_CALL_SPREAD): {
        int argc = READ_BYTE();
        SAVE_STATE();
        if (argc < 1) THROW(vm.cRuntimeError, "CALL_SPREAD with no arguments");
        Value spread = stackTop[-1];
        if (!IS_LIST(spread)) {
            THROW(vm.cTypeError, "spread argument must be a list, not '%s'",
                  jaiTypeNameStatic(spread));
        }
        ObjList *list = AS_LIST(spread);
        DROP(1);
        SAVE_STATE();
        if (!ensureStack(list->count + 1)) goto vmThrow;
        for (int i = 0; i < list->count; i++) PUSH(jaiListGet(list, i));
        int total = argc - 1 + list->count;
        SAVE_STATE();
        if (callValueOnStack(total) == CALL_ERROR) goto vmThrow;
        LOAD_STATE();
        VM_NEXT();
    }

    VM_CASE(OP_INVOKE): {
        uint32_t nameIdx = READ_U24();
        int argc = READ_BYTE();
        uint16_t cacheIdx = READ_U16();
        Value receiver = stackTop[-argc - 1];
        uint32_t builtinTag = 0;
        /* Set when this pass has a way whose result kind is still being
         * observed and the call is one that finishes here (a builtin, or an
         * instance method the compiled tier completed). See
         * InlineCache::resultKind. */
        InlineCache *fbCache = NULL;
        int fbWay = 0;
        /* True when the record is also armed in sResultSite, so the return will
         * make it unless the call turns out to have finished here. */
        bool fbFromFrame = false;

        /* Fast path: the receiver's class is one the cache has already seen,
         * so the method is known without touching a hash table. */
        if (IS_INSTANCE(receiver)) {
            InlineCache *ic = cacheAt(frameChunk(frame), cacheIdx);
            ObjClass *klass = AS_INSTANCE(receiver)->klass;
            if (ic != NULL && klass != NULL && ic->state != IC_EMPTY &&
                ic->state != IC_MEGA) {
                for (int w = 0; w < ic->count; w++) {
                    if (ic->shapeId[w] != klass->shapeId) continue;
                    vm.icHits++;
                    SAVE_STATE();
                    /* The payload marks a way whose method is not public. What
                     * the cache settles is which method a shape resolves to;
                     * whether this frame may call it depends on the *caller*,
                     * which one site can present as two classes, so that half
                     * is re-decided on every hit. */
                    if (ic->payload[w] != 0 &&
                        !methodPermitted(klass, AS_STRING(constants[nameIdx]),
                                         true)) {
                        goto vmThrow;
                    }
                    /* The cache is only filled when slot 0 is the receiver
                     * (see the fill site below), so this is a method call. */
                    bool armed = false;
                    if (JAI_UNLIKELY(ic->obsBudget != 0)) {
                        armInvokeResult(ic, (unsigned)w, frame->ip);
                        armed = true;
                    }
                    CallOutcome outcome = invokeMethodOnStack(ic->cached[w],
                                                              argc);
                    if (outcome == CALL_ERROR) goto vmThrow;
                    /* CALL_DONE means no frame was pushed -- a native method in
                     * the class's table, or a closure the compiled tier ran
                     * outright -- so no OP_RETURN will ever answer the arming,
                     * and the result is on the stack right here instead. */
                    if (JAI_UNLIKELY(armed) && outcome == CALL_DONE) {
                        recordInvokeResult(ic, (unsigned)w, vm.stackTop[-1]);
                        sResultSite.ic = NULL;
                    }
                    LOAD_STATE();
                    VM_NEXT();
                }
            } else if (ic != NULL && klass != NULL && ic->state == IC_MEGA) {
                /* Out of ways: the shared table above answers instead. */
                ObjString *mname = AS_STRING(constants[nameIdx]);
                MegaEntry *me = &sMegaCache[megaSlot(klass, mname)];
                Value found;
                if (me->klass == klass && me->name == mname &&
                    me->tableVersion == klass->methods.version) {
                    vm.icHits++;
                    SAVE_STATE();
                    if (me->recheck && !methodPermitted(klass, mname, true)) {
                        goto vmThrow;
                    }
                    if (invokeMethodOnStack(me->method, argc) == CALL_ERROR) {
                        goto vmThrow;
                    }
                    LOAD_STATE();
                    VM_NEXT();
                }
                if (jaiTableGetInterned(&klass->methods, mname, &found)) {
                    MethodInfo mi;
                    vm.icMisses++;
                    SAVE_STATE();
                    me->klass = klass;
                    me->name = mname;
                    me->method = found;
                    me->tableVersion = klass->methods.version;
                    me->recheck = jaiClassRestrictedMethod(klass, mname, &mi);
                    if (me->recheck && !methodPermitted(klass, mname, true)) {
                        goto vmThrow;
                    }
                    if (invokeMethodOnStack(found, argc) == CALL_ERROR) {
                        goto vmThrow;
                    }
                    LOAD_STATE();
                    VM_NEXT();
                }
            }
        } else if ((builtinTag = builtinShapeTag(receiver)) != 0) {
            /* Same idea for `xs.push(v)`. The cached value is the ObjNative
             * itself, not a bound wrapper: the receiver is already sitting in
             * the callee slot, which is exactly where a built-in method wants
             * its args[0], so the call needs no intermediate object at all. */
            InlineCache *ic = cacheAt(frameChunk(frame), cacheIdx);
            if (ic != NULL && ic->state != IC_EMPTY && ic->state != IC_MEGA) {
                for (int w = 0; w < ic->count; w++) {
                    if (ic->shapeId[w] != builtinTag) continue;
                    vm.icHits++;
                    SAVE_STATE();
                    Value *slot = vm.stackTop - argc - 1;
                    Value result;
                    if (!callNativeAt(AS_NATIVE(ic->cached[w]), slot, argc + 1,
                                      &result)) {
                        goto vmThrow;
                    }
                    vm.stackTop = slot;
                    *vm.stackTop++ = result;
                    /* The result is right here, so a builtin needs none of the
                     * frame machinery an instance method does -- but it does
                     * need the same WINDOW. Recorded only at the fill, this was
                     * a single observation, which roadmap §6 records as an open
                     * bug: `d.get(k)` over a dict holding two kinds predicted
                     * whichever key came first, and every later call deopted. */
                    if (JAI_UNLIKELY(ic->obsBudget != 0)) {
                        ic->obsBudget--;
                        recordInvokeResult(ic, (unsigned)w, result);
                    }
                    /* A built-in method pushes no frame: `xs.push(v)` in a hot
                     * loop reaches here, and LOAD_STATE's constant-pool chase
                     * was the largest thing left in it. */
                    LOAD_STACK_ONLY();
                    VM_NEXT();
                }
            }
        }

        SAVE_STATE();
        ObjString *name = AS_STRING(constants[nameIdx]);

        /* `Shape.Circle(2.0)`: a variant with a payload is not a value, it is
         * a constructor, and this is the only place the arguments and the
         * variant name are both in hand. Zero-arity variants are values and
         * `enumMember` already produced one. */
        if (IS_ENUM(receiver)) {
            ObjEnum *enumType = AS_ENUM(receiver);
            int tag = jaiEnumVariantIndex(enumType, name);
            if (tag >= 0 && enumType->variants[tag].arity > 0) {
                if (argc != enumType->variants[tag].arity) {
                    THROW(vm.cTypeError,
                          "%s.%s() takes %d argument%s but %d were given",
                          enumType->name != NULL ? enumType->name->chars : "?",
                          name->chars, (int)enumType->variants[tag].arity,
                          enumType->variants[tag].arity == 1 ? "" : "s", argc);
                }
                ObjEnumVal *built = jaiEnumValNew(enumType, (uint16_t)tag,
                                                  vm.stackTop - argc, argc);
                LOAD_STATE();
                DROP(argc + 1);            /* the arguments and the enum */
                PUSH(OBJ_VAL(built));
                VM_NEXT();
            }
        }

        Value method, slotZero;
        bool isMethod;
        if (!resolveInvokeTarget(receiver, name, &method, &slotZero, &isMethod)) {
            /* The lookup may already have raised something more precise than
             * "no such method" — E0802 for a module member that exists but is
             * private. Overwriting it would send the reader hunting a typo. */
            if (vm.hasException) goto vmThrow;
            THROW(vm.cAttributeError, "'%s' object has no method '%s'",
                  jaiTypeNameStatic(receiver), name->chars);
        }
        vm.stackTop[-argc - 1] = slotZero;

        if (IS_INSTANCE(receiver)) {
            vm.icMisses++;
            InlineCache *ic = cacheAt(frameChunk(frame), cacheIdx);
            ObjClass *klass = AS_INSTANCE(receiver)->klass;
            /* Only a real method is cacheable: a callable field is per
             * instance, not per shape. A non-public one is cached with the
             * payload set, which tells the fast path above to re-run the
             * visibility test it cannot cache. */
            MethodInfo restricted;
            if (ic != NULL && klass != NULL && AS_OBJ(slotZero) == AS_OBJ(receiver) &&
                ic->state != IC_MEGA) {
                /* Inside the guard, not above it: a site that has gone
                 * megamorphic reaches here on every call and has nothing to
                 * fill, so the walk was pure cost exactly where calls are
                 * dearest. */
                uint32_t recheck =
                    jaiClassRestrictedMethod(klass, name, &restricted) ? 1u : 0u;
                if (ic->count < JAI_IC_WAYS) {
                    ic->shapeId[ic->count] = klass->shapeId;
                    ic->payload[ic->count] = recheck;
                    ic->cached[ic->count] = method;
                    ic->resultKind[ic->count] = JAI_FB_NONE;
                    ic->count++;
                    ic->state = (ic->count == 1) ? IC_MONO : IC_POLY;
                    /* The way this pass just filled is the way this call's
                     * result belongs to, and the interpreter is about to make
                     * that call: arm it here rather than waiting for the next
                     * hit, so a site called exactly once is still recorded. */
                    if (ic->obsBudget != 0) {
                        armInvokeResult(ic, (unsigned)(ic->count - 1),
                                        frame->ip);
                        fbCache = ic;
                        fbWay   = ic->count - 1;
                        fbFromFrame = true;
                    }
                } else {
                    ic->state = IC_MEGA;
                }
            }
        } else if (builtinTag != 0 && IS_BOUND(method) &&
                   IS_NATIVE(AS_BOUND(method)->method)) {
            /* jaiBuiltinMethod hands back a bound native whose receiver is the
             * one we passed; cache the native and drop the wrapper. Anything
             * shaped differently (a module member, a __format__ on a value the
             * tag does not cover) simply is not cached. */
            vm.icMisses++;
            InlineCache *ic = cacheAt(frameChunk(frame), cacheIdx);
            if (ic != NULL && ic->state != IC_MEGA) {
                if (ic->count < JAI_IC_WAYS) {
                    ic->shapeId[ic->count] = builtinTag;
                    ic->payload[ic->count] = 0;
                    ic->cached[ic->count] = AS_BOUND(method)->method;
                    ic->resultKind[ic->count] = JAI_FB_NONE;
                    ic->count++;
                    ic->state = (ic->count == 1) ? IC_MONO : IC_POLY;
                    if (ic->obsBudget != 0) {
                        ic->obsBudget--;
                        fbCache = ic;
                        fbWay   = ic->count - 1;
                    }
                } else {
                    ic->state = IC_MEGA;
                }
            }
        }
        CallOutcome fillOutcome = isMethod ? invokeMethodOnStack(method, argc)
                                           : invokeCallable(method, argc);
        if (fillOutcome == CALL_ERROR) goto vmThrow;
        /* Only reached on the way that just filled the cache -- once per site
         * and receiver type, never on the path a loop repeats.
         *
         * A bound native pushed no frame, so its result is on the stack now. An
         * instance method usually did push one, and then the record is already
         * armed and OP_RETURN will make it; CALL_DONE says otherwise and this
         * is the only place that can tell. */
        if (fbCache != NULL && (!fbFromFrame || fillOutcome == CALL_DONE)) {
            recordInvokeResult(fbCache, (unsigned)fbWay, vm.stackTop[-1]);
            if (fbFromFrame) sResultSite.ic = NULL;
        }
        LOAD_STATE();
        VM_NEXT();
    }

    VM_CASE(OP_SUPER_INVOKE): {
        uint32_t nameIdx = READ_U24();
        int argc = READ_BYTE();
        SAVE_STATE();
        Value receiver = stackTop[-argc - 1];
        ObjString *name = AS_STRING(constants[nameIdx]);

        ObjClass *start = NULL;
        if (IS_INSTANCE(receiver)) start = AS_INSTANCE(receiver)->klass;
        else if (IS_CLASS(receiver)) start = AS_CLASS(receiver);
        if (start == NULL || start->superclass == NULL) {
            THROW(vm.cRuntimeError, "'super.%s' has no superclass to dispatch to",
                  name->chars);
        }
        Value method;
        if (!findMethod(start->superclass, name, &method)) {
            THROW(vm.cAttributeError, "superclass '%s' has no method '%s'",
                  start->superclass->name != NULL ? start->superclass->name->chars
                                                  : "?",
                  name->chars);
        }
        /* Slot 0 is the receiver here whatever the parent's method turns out
         * to be, so `super.f()` reaches a native parent method with `self`. */
        if (invokeMethodOnStack(method, argc) == CALL_ERROR) goto vmThrow;
        LOAD_STATE();
        VM_NEXT();
    }

    VM_CASE(OP_TAIL_CALL): {
        int argc = READ_BYTE();
        SAVE_STATE();
        Value callee = stackTop[-argc - 1];
        ObjFunction *fn = frame->closure->fn;

        /* Reuse the window only for the straightforward shape; anything with
         * defaults, a variadic tail, or a non-closure callee goes through the
         * ordinary path, which is correct if less frugal.
         *
         * `@trace` callers also take the slow path: reusing this frame would
         * skip OP_RETURN's jaiTraceLeave, and leaving before the tail callee
         * runs would drop the session while `return trace.is_replay()` still
         * needs it. */
        if (IS_CLOSURE(callee) && AS_CLOSURE(callee)->fn->defaultCount == 0 &&
            !(AS_CLOSURE(callee)->fn->flags & (FN_VARIADIC | FN_KWREST)) &&
            AS_CLOSURE(callee)->fn->arity == argc &&
            !(fn->flags & FN_INIT) &&
            !(fn->flags & FN_TRACE)) {
            ObjClosure *target = AS_CLOSURE(callee);

            /* A tail call reuses this frame instead of going through
             * callClosure, so without this the compiled tier never saw the
             * callee at all -- not even to count it as hot. `sort` tail-calls
             * `merge`, which is the whole of that benchmark's inner loop, and
             * it was invisible. The compiled form finishes the call outright,
             * and finishing a tail call is returning from this frame. */
            ObjFunction *tfn = target->fn;
            Value *tailBase = vm.stackTop - argc - 1;
            if (tfn->jitFunc != NULL) {
                JaiJitOutcome outcome = jaiJitEnterFunc(target, tailBase);
                if (outcome == JAI_JIT_ERROR) goto vmThrow;
                if (outcome == JAI_JIT_DEOPT) {
                    /* Push a real frame rather than reusing this one: the
                     * callee is resuming part-way through its own body. The
                     * OP_RETURN the compiler puts after a tail call then
                     * returns its result, which is what a tail call means. */
                    if (!bindCallArgs(target, argc, tailBase)) goto vmThrow;
                    if (!pushFrame(target, tailBase)) goto vmThrow;
                    if (!jaiJitApplyDeopt(target, tailBase)) goto vmThrow;
                    LOAD_STATE();
                    VM_NEXT();
                }
                if (outcome == JAI_JIT_DONE) {
                    retval = tailBase[0];
                    stackTop = vm.stackTop;   /* opReturn saves this back */
                    goto opReturn;
                }
            } else if (tfn->entryCount < jaiJitThreshold(tfn)) {
                tfn->entryCount++;
            } else if (!tfn->jitRefused && jaiJitEnabled()) {
                /* Same three-way answer the jitFunc branch above already
                 * handles: on the call that compiles it, the body may run and
                 * then deoptimise, and reusing this frame for the tail call
                 * would re-run the callee from the top. */
                JaiJitOutcome outcome = jaiJitEnter(target, tailBase);
                if (outcome == JAI_JIT_ERROR) goto vmThrow;
                if (outcome == JAI_JIT_DEOPT) {
                    if (!bindCallArgs(target, argc, tailBase)) goto vmThrow;
                    if (!pushFrame(target, tailBase)) goto vmThrow;
                    if (!jaiJitApplyDeopt(target, tailBase)) goto vmThrow;
                    LOAD_STATE();
                    VM_NEXT();
                }
                if (outcome == JAI_JIT_DONE) {
                    retval = tailBase[0];
                    stackTop = vm.stackTop;
                    goto opReturn;
                }
            }

            if (FRAME_HAS_DEFERS(frame) && !runFrameDefers(frame)) goto vmThrow;
            if (vm.hasException) goto vmThrow;
            closeUpvalues(frame->slots);

            Value *window = frame->slots;
            memmove(window, vm.stackTop - argc - 1,
                    sizeof(Value) * (size_t)(argc + 1));
            vm.stackTop = window + argc + 1;

            int newWindow = frameWindowSize(target->fn);
            for (int i = argc + 1; i < newWindow; i++) window[i] = NULL_VAL;
            vm.stackTop = window + newWindow;

            vm.handlers.count = frame->handlerBase;
            vm.defers.count = frame->deferBase;
            frame->closure = target;
            frame->ip = target->fn->chunk.code;
            if (target->fn->module != NULL) frame->module = target->fn->module;
            vm.callCount++;
            LOAD_STATE();
            VM_NEXT();
        }
        if (callValueOnStack(argc) == CALL_ERROR) goto vmThrow;
        LOAD_STATE();
        VM_NEXT();
    }

    VM_CASE(OP_RETURN):
        retval = POP();
        goto opReturn;

    VM_CASE(OP_RETURN_NULL):
        retval = NULL_VAL;
        goto opReturn;

    VM_CASE(OP_POP_RETURN_NULL):
        DROP(1);
        retval = NULL_VAL;
        goto opReturn;

    opReturn: {
        SAVE_STATE();
        ObjFunction *fn = frame->closure->fn;
        /* An initializer yields the object it initialised, whatever its body
         * returned; that is what makes `Point(1, 2)` an expression. A default
         * thunk borrows the same function record, so it is excluded. */
        if (((fn->flags & FN_INIT) || (fn->name != NULL && fn->name == vm.strInit)) &&
            vm.frameCount - 1 != sThunkFrame) {
            retval = frame->slots[0];
        }
        if (FRAME_HAS_DEFERS(frame)) (void)runFrameDefers(frame);
        if (vm.hasException) goto vmThrow;

        /* Record what this function returns, for a compiled caller that cannot
         * ask its compiled form because it has none yet. Costs the steady state
         * one already-hot load and a not-taken branch: entryCount saturates at
         * the compile threshold and never moves again. See
         * ObjFunction::obsReturnKind for why this is per callee, not per site. */
        if (JAI_UNLIKELY(fn->entryCount < jaiJitThreshold(fn))) {
            uint32_t shape = 0;
            if (IS_INSTANCE(retval) && AS_INSTANCE(retval)->klass != NULL) {
                shape = AS_INSTANCE(retval)->klass->shapeId;
                /* A shape is only usable by the tier if the shape->class table
                 * can answer it, and until now the only thing that filled that
                 * table was a function COMPILING with an instance return -- so
                 * a callee that has not compiled recorded a shape nothing could
                 * resolve, which is exactly the callee this record is for. */
                jaiClassRememberShape(AS_INSTANCE(retval)->klass);
            }
            /* A null says nothing about the shape, and must not be allowed
             * to answer for one: the old rule zeroed the shape the moment a
             * nullable-instance function returned its null, so the very case
             * the nullable band exists to record arrived with no class and was
             * refused anyway. The first NON-null return sets it; later ones
             * confirm it or zero it, and that zero stays sticky because
             * `firstReal` can never come back true afterwards. */
            if (!IS_NULL(retval)) {
                uint8_t prevfb = fn->obsReturnKind;
                bool firstReal = prevfb == JAI_FB_NONE || prevfb == JAI_FB_NULL;
                if (firstReal) fn->obsReturnShape = shape;
                else if (fn->obsReturnShape != shape) fn->obsReturnShape = 0;
            }
            fn->obsReturnKind = jaiFeedbackMerge(fn->obsReturnKind,
                                                 jaiFeedbackKind(retval));
        }

        /* And the same question asked per SITE: whichever OP_INVOKE armed
         * itself for this frame gets what that frame returned. The two are not
         * redundant -- a caller that has pinned nothing about its receiver
         * cannot name the callee, so it has no obsReturnKind to read.
         *
         * Two tests, not one. The depth says the return belongs to the frame
         * the arming was for; the resume address says it belongs to the SITE it
         * was for, which is what an intervening unwind and a re-entry at the
         * same depth would otherwise get wrong. Costs the steady state one load
         * and a not-taken branch: every site freezes after JAI_IC_OBS_BUDGET
         * invokes and nothing arms again. */
        if (JAI_UNLIKELY(sResultSite.ic != NULL) &&
            sResultSite.depth == vm.frameCount && vm.frameCount >= 2 &&
            vm.frames[vm.frameCount - 2].ip == sResultSite.resumeIp) {
            recordInvokeResult(sResultSite.ic, sResultSite.way, retval);
            sResultSite.ic = NULL;
        }

        closeUpvalues(frame->base);
        vm.handlers.count = frame->handlerBase;
        vm.defers.count = frame->deferBase;
        if (fn->flags & FN_TRACE) jaiTraceLeave(fn);
        vm.frameCount--;
        vm.stackTop = frame->base;
        *vm.stackTop++ = retval;

        if (vm.frameCount <= baseFrameCount) return JAI_RUN_OK;
        LOAD_STATE();
        VM_NEXT();
    }

    VM_CASE(OP_CLOSURE): {
        Value fnValue = READ_CONST();
        SAVE_STATE();
        if (!IS_FUNCTION(fnValue)) {
            THROW(vm.cRuntimeError, "CLOSURE operand is not a function");
        }
        ObjFunction *fn = AS_FUNCTION(fnValue);
        ObjClosure *closure = jaiClosureNew(fn);
        LOAD_STATE();
        /* Read the upvalue descriptors before pushing: the closure is only
         * reachable from this local until then. */
        jaiGCPushRoot(OBJ_VAL(closure));
        for (int i = 0; i < closure->upvalueCount; i++) {
            uint8_t how = READ_BYTE();
            uint16_t index = READ_U16();
            bool isLocal = (how & 1u) != 0;
            if ((how & 2u) != 0) {
                /* By value (spec §6's `let`): the closure gets the value the
                 * binding holds now, so a loop that builds one closure per
                 * iteration keeps one value per iteration. */
                Value snapshot = isLocal
                                     ? frame->slots[index]
                                     : *frame->closure->upvalues[index]->location;
                SAVE_STATE();
                closure->upvalues[i] = jaiUpvalueClosed(snapshot);
                LOAD_STATE();
            } else if (isLocal) {
                SAVE_STATE();
                closure->upvalues[i] = captureUpvalue(frame->slots + index);
                LOAD_STATE();
            } else {
                closure->upvalues[i] = frame->closure->upvalues[index];
            }
        }
        jaiGCPopRoot();
        PUSH(OBJ_VAL(closure));
        VM_NEXT();
    }

    /* --- data structures (spec §3.6) --- */

    VM_CASE(OP_BUILD_LIST): {
        int count = READ_U16();
        SAVE_STATE();
        ObjList *list = jaiListNew(count);
        LOAD_STATE();
        for (int i = 0; i < count; i++) jaiListPut(list, i, stackTop[-count + i]);
        list->count = count;
        DROP(count);
        PUSH(OBJ_VAL(list));
        VM_NEXT();
    }

    VM_CASE(OP_BUILD_DICT): {
        int count = READ_U16();
        SAVE_STATE();
        ObjDict *dict = jaiDictNew();
        jaiGCPushRoot(OBJ_VAL(dict));
        for (int i = 0; i < count; i++) {
            Value key = vm.stackTop[-2 * count + 2 * i];
            Value value = vm.stackTop[-2 * count + 2 * i + 1];
            (void)jaiDictSet(dict, key, value);
            if (vm.hasException) { jaiGCPopRoot(); goto vmThrow; }
        }
        jaiGCPopRoot();
        LOAD_STATE();
        DROP(2 * count);
        PUSH(OBJ_VAL(dict));
        VM_NEXT();
    }

    VM_CASE(OP_BUILD_SET): {
        int count = READ_U16();
        SAVE_STATE();
        ObjSet *set = jaiSetNew();
        jaiGCPushRoot(OBJ_VAL(set));
        for (int i = 0; i < count; i++) {
            (void)jaiSetAdd(set, vm.stackTop[-count + i]);
            if (vm.hasException) { jaiGCPopRoot(); goto vmThrow; }
        }
        jaiGCPopRoot();
        LOAD_STATE();
        DROP(count);
        PUSH(OBJ_VAL(set));
        VM_NEXT();
    }

    VM_CASE(OP_BUILD_TUPLE): {
        int count = READ_U16();
        SAVE_STATE();
        ObjTuple *tuple = jaiTupleNew(vm.stackTop - count, count);
        LOAD_STATE();
        DROP(count);
        PUSH(OBJ_VAL(tuple));
        VM_NEXT();
    }

    VM_CASE(OP_BUILD_RANGE): {
        bool inclusive = READ_BYTE() != 0;
        Value stopValue = PEEK(0), startValue = PEEK(1);
        if (!IS_INT(startValue) || !IS_INT(stopValue)) {
            THROW(vm.cTypeError, "range bounds must be int, not '%s' and '%s'",
                  jaiTypeNameStatic(startValue), jaiTypeNameStatic(stopValue));
        }
        SAVE_STATE();
        ObjRange *range = jaiRangeNew(AS_INT(startValue), AS_INT(stopValue), 1,
                                      inclusive);
        LOAD_STATE();
        DROP(2);
        PUSH(OBJ_VAL(range));
        VM_NEXT();
    }

    VM_CASE(OP_LIST_APPEND): {
        int depth = READ_U16();
        /* The operand is the container's peek index outright (spec §3.6). */
        Value target = PEEK(depth);
        if (!IS_LIST(target)) {
            THROW(vm.cTypeError, "comprehension target is not a list");
        }
        SAVE_STATE();
        jaiListPush(AS_LIST(target), PEEK(0));
        if (vm.hasException) goto vmThrow;
        LOAD_STATE();
        DROP(1);
        VM_NEXT();
    }

    VM_CASE(OP_DICT_INSERT): {
        int depth = READ_U16();
        /* The operand is the container's peek index outright (spec §3.6). */
        Value target = PEEK(depth);
        if (!IS_DICT(target)) {
            THROW(vm.cTypeError, "comprehension target is not a dict");
        }
        SAVE_STATE();
        (void)jaiDictSet(AS_DICT(target), PEEK(1), PEEK(0));
        if (vm.hasException) goto vmThrow;
        LOAD_STATE();
        DROP(2);
        VM_NEXT();
    }

    VM_CASE(OP_SET_ADD): {
        int depth = READ_U16();
        /* The operand is the container's peek index outright (spec §3.6). */
        Value target = PEEK(depth);
        if (!IS_SET(target)) {
            THROW(vm.cTypeError, "comprehension target is not a set");
        }
        SAVE_STATE();
        (void)jaiSetAdd(AS_SET(target), PEEK(0));
        if (vm.hasException) goto vmThrow;
        LOAD_STATE();
        DROP(1);
        VM_NEXT();
    }

    VM_CASE(OP_GET_INDEX): {
        Value result;
        if (JAI_LIKELY(indexGetFast(stackTop[-2], stackTop[-1], &result))) {
            DROP(2);
            PUSH(result);
            VM_NEXT();
        }
        SAVE_STATE();
        if (!jaiIndexGet(stackTop[-2], stackTop[-1], &result)) goto vmThrow;
        LOAD_STATE();
        DROP(2);
        PUSH(result);
        VM_NEXT();
    }

    VM_CASE(OP_SET_INDEX): {
        SAVE_STATE();
        if (!indexSet(stackTop[-3], stackTop[-2], stackTop[-1])) goto vmThrow;
        LOAD_STATE();
        DROP(3);
        VM_NEXT_HINT(OP_LOOP);
    }

    VM_CASE(OP_GET_SLICE): {
        uint8_t flags = READ_BYTE();
        bool hasStart = (flags & 1) != 0;
        bool hasStop = (flags & 2) != 0;
        bool hasStep = (flags & 4) != 0;
        int operands = (hasStart ? 1 : 0) + (hasStop ? 1 : 0) + (hasStep ? 1 : 0);

        SAVE_STATE();
        Value *args = stackTop - operands;
        int at = 0;
        Value startValue = hasStart ? args[at++] : NULL_VAL;
        Value stopValue = hasStop ? args[at++] : NULL_VAL;
        Value stepValue = hasStep ? args[at++] : NULL_VAL;
        Value result;
        if (!jaiSliceGet(args[-1], startValue, stopValue, stepValue,
                         hasStart, hasStop, hasStep, &result)) {
            goto vmThrow;
        }
        LOAD_STATE();
        DROP(operands + 1);
        PUSH(result);
        VM_NEXT();
    }

    VM_CASE(OP_SET_SLICE): {
        uint8_t flags = READ_BYTE();
        bool hasStart = (flags & 1) != 0;
        bool hasStop = (flags & 2) != 0;
        bool hasStep = (flags & 4) != 0;
        int operands = (hasStart ? 1 : 0) + (hasStop ? 1 : 0) + (hasStep ? 1 : 0);

        SAVE_STATE();
        Value assigned = stackTop[-1];
        Value *args = stackTop - 1 - operands;
        Value container = args[-1];
        if (!IS_LIST(container)) {
            THROW(vm.cTypeError, "'%s' value does not support slice assignment",
                  jaiTypeNameStatic(container));
        }
        int at = 0;
        Value startValue = hasStart ? args[at++] : NULL_VAL;
        Value stopValue = hasStop ? args[at++] : NULL_VAL;
        Value stepValue = hasStep ? args[at++] : NULL_VAL;

        ObjList *list = AS_LIST(container);
        int64_t start, stop, step;
        if (!sliceBounds(startValue, stopValue, stepValue, hasStart, hasStop,
                         hasStep, list->count, &start, &stop, &step)) {
            goto vmThrow;
        }

        const Value *source = NULL;
        int sourceCount = 0;
        if (IS_LIST(assigned)) {
            source = jaiListBox(AS_LIST(assigned));
            sourceCount = AS_LIST(assigned)->count;
        } else if (IS_TUPLE(assigned)) {
            source = AS_TUPLE(assigned)->items;
            sourceCount = (int)AS_TUPLE(assigned)->count;
        } else {
            THROW(vm.cTypeError,
                  "slice assignment requires a list or tuple, not '%s'",
                  jaiTypeNameStatic(assigned));
        }

        /* Only same-length replacement: resizing through a slice would make
         * the surrounding indices silently wrong. */
        int64_t selected = 0;
        for (int64_t i = start; (step > 0) ? (i < stop) : (i > stop); i += step) {
            if (i < 0 || i >= list->count) break;
            selected++;
        }
        if (selected != sourceCount) {
            THROW(vm.cValueError,
                  "slice assignment needs %" PRId64 " values but %d were given",
                  selected, sourceCount);
        }
        int64_t written = 0;
        for (int64_t i = start; (step > 0) ? (i < stop) : (i > stop); i += step) {
            if (i < 0 || i >= list->count) break;
            jaiListPut(list, i, source[written++]);
        }
        if (written > 0) jaiListTouch(list);
        LOAD_STATE();
        DROP(operands + 2);
        VM_NEXT();
    }

    VM_CASE(OP_UNPACK): {
        int count = READ_BYTE();
        int restIndex = READ_BYTE();
        SAVE_STATE();
        Value source = stackTop[-1];

        const Value *items = NULL;
        int available = 0;
        if (IS_LIST(source)) {
            items = jaiListBox(AS_LIST(source));
            available = AS_LIST(source)->count;
        } else if (IS_TUPLE(source)) {
            items = AS_TUPLE(source)->items;
            available = (int)AS_TUPLE(source)->count;
        } else {
            THROW(vm.cTypeError, "cannot destructure a '%s' value",
                  jaiTypeNameStatic(source));
        }

        bool hasRest = (restIndex != 255);
        int fixed = hasRest ? count - 1 : count;
        if ((hasRest && available < fixed) || (!hasRest && available != count)) {
            THROW(vm.cValueError,
                  "cannot unpack %d value%s into %d target%s", available,
                  available == 1 ? "" : "s", count, count == 1 ? "" : "s");
        }

        Value unpacked[JAI_MAX_ARGS];
        if (count > JAI_MAX_ARGS) {
            THROW(vm.cRuntimeError, "too many destructuring targets");
        }
        int restCount = available - fixed;
        int read = 0;
        for (int i = 0; i < count; i++) {
            if (hasRest && i == restIndex) {
                ObjList *rest = jaiListNew(restCount);
                for (int j = 0; j < restCount; j++) jaiListPut(rest, j, items[read + j]);
                rest->count = restCount;
                read += restCount;
                unpacked[i] = OBJ_VAL(rest);
                /* jaiListNew can collect; re-read the (still rooted) source. */
                items = IS_LIST(source) ? jaiListBox(AS_LIST(source))
                                        : AS_TUPLE(source)->items;
            } else {
                unpacked[i] = items[read++];
            }
        }
        LOAD_STATE();
        DROP(1);
        /* Right to left, so the leftmost target is on top and a run of
         * SET_LOCAL/POP assigns in source order. */
        for (int i = count - 1; i >= 0; i--) PUSH(unpacked[i]);
        VM_NEXT();
    }

    /* --- objects, classes, traits (spec §3.7) --- */

    VM_CASE(OP_CLASS): {
        Value spec = READ_CONST();
        SAVE_STATE();
        Obj *created = classSpecInstantiate(spec);
        if (created == NULL) goto vmThrow;
        LOAD_STATE();
        PUSH(OBJ_VAL(created));
        VM_NEXT();
    }

    VM_CASE(OP_INHERIT): {
        Value subValue = PEEK(0), superValue = PEEK(1);
        if (!IS_CLASS(superValue)) {
            THROW(vm.cTypeError, "a superclass must be a class, not '%s'",
                  jaiTypeNameStatic(superValue));
        }
        if (!IS_CLASS(subValue)) {
            THROW(vm.cTypeError, "INHERIT expected a class on top of the stack");
        }
        if (jaiClassIsSubclassOf(AS_CLASS(superValue), AS_CLASS(subValue))) {
            THROW(vm.cRuntimeError, "cyclic inheritance involving class '%s'",
                  AS_CLASS(subValue)->name != NULL
                      ? AS_CLASS(subValue)->name->chars : "?");
        }
        SAVE_STATE();
        jaiClassInherit(AS_CLASS(subValue), AS_CLASS(superValue));
        if (vm.hasException) goto vmThrow;
        LOAD_STATE();
        VM_NEXT();
    }

    VM_CASE(OP_IMPL_TRAIT): {
        Value traitRef = READ_CONST();
        SAVE_STATE();
        Value implementor = PEEK(0);
        if (!IS_CLASS(implementor) && !IS_TRAIT(implementor)) {
            THROW(vm.cTypeError,
                  "IMPL_TRAIT expected a class or trait on the stack");
        }
        ObjClass *klass = IS_CLASS(implementor) ? AS_CLASS(implementor) : NULL;

        ObjTrait *trait = NULL;
        if (IS_TRAIT(traitRef)) {
            trait = AS_TRAIT(traitRef);
        } else if (IS_STRING(traitRef)) {
            Value found;
            if ((frame->module != NULL &&
                 jaiModuleGet(frame->module, AS_STRING(traitRef), &found) &&
                 IS_TRAIT(found)) ||
                (vm.builtins != NULL &&
                 jaiModuleGet(vm.builtins, AS_STRING(traitRef), &found) &&
                 IS_TRAIT(found))) {
                trait = AS_TRAIT(found);
            }
        }
        if (trait == NULL) {
            THROW(vm.cTypeError, "unknown trait in `impl` for '%s'",
                  jaiTypeNameStatic(implementor));
        }

        if (klass == NULL) {
            /* `trait Sub: Super` — record the supertrait and inherit both its
             * requirements and its defaults, so a class implementing Sub is
             * checked against, and inherits, the whole chain. */
            ObjTrait *sub = AS_TRAIT(implementor);
            uint16_t oldSupers = sub->superCount;
            sub->supers = JAI_GROW_ARRAY(ObjTrait *, sub->supers, oldSupers,
                                         oldSupers + 1);
            sub->supers[oldSupers] = trait;
            sub->superCount = (uint16_t)(oldSupers + 1);

            int si = 0;
            Value skey, svalue, sexisting;
            while (jaiTableNext(&trait->required, &si, &skey, &svalue)) {
                if (!jaiTableGet(&sub->required, skey, &sexisting))
                    jaiTableSet(&sub->required, skey, svalue);
            }
            si = 0;
            while (jaiTableNext(&trait->defaults, &si, &skey, &svalue)) {
                if (!jaiTableGet(&sub->defaults, skey, &sexisting))
                    jaiTableSet(&sub->defaults, skey, svalue);
            }
            LOAD_STATE();
            VM_NEXT();
        }

        uint16_t oldCount = klass->traitCount;
        klass->traits = JAI_GROW_ARRAY(ObjTrait *, klass->traits, oldCount,
                                       oldCount + 1);
        klass->traits[oldCount] = trait;
        klass->traitCount = (uint16_t)(oldCount + 1);

        /* Copy the trait's default implementations in without overwriting
         * anything the class declares, so the two possible emission orders
         * (methods before or after the impl) both come out right. */
        int i = 0;
        Value key, value;
        while (jaiTableNext(&trait->defaults, &i, &key, &value)) {
            if (!IS_STRING(key)) continue;
            Value existing;
            if (jaiTableGetInterned(&klass->methods, AS_STRING(key), &existing)) {
                continue;
            }
            jaiClassAddMethod(klass, AS_STRING(key), value, VIS_PUBLIC, 0);
        }
        jaiClassRefreshDunders(klass);
        LOAD_STATE();
        VM_NEXT();
    }

    VM_CASE(OP_METHOD): {
        ObjString *name = AS_STRING(READ_CONST());
        uint8_t flags = READ_BYTE();
        uint8_t visByte = READ_BYTE();
        SAVE_STATE();
        Value method = PEEK(0);
        Value owner = PEEK(1);
        /* The flags byte is full of FunctionFlags, so visibility rides in a
         * byte of its own; a value from a corrupt chunk is read as private,
         * which can only ever deny an access. */
        Visibility vis = visByte <= (uint8_t)VIS_PUBLIC ? (Visibility)visByte
                                                        : VIS_PRIVATE;
        /* Record where the function was declared, for the visibility test:
         * "private to the declaring class" is a property of the code, and slot
         * 0 only reports it for a call that came through OP_INVOKE. */
        if (IS_CLASS(owner)) {
            ObjFunction *fn = IS_CLOSURE(method)    ? AS_CLOSURE(method)->fn
                              : IS_FUNCTION(method) ? AS_FUNCTION(method)
                                                    : NULL;
            if (fn != NULL) fn->owner = AS_CLASS(owner);
        }
        if (IS_CLASS(owner)) {
            jaiClassAddMethod(AS_CLASS(owner), name, method, vis, flags);
        } else if (IS_TRAIT(owner)) {
            /* A trait's only bodied methods are its defaults; the requirements
             * travelled in the spec constant. */
            jaiTableSet(&AS_TRAIT(owner)->defaults, OBJ_VAL(name), method);
        } else if (IS_ENUM(owner)) {
            jaiTableSet(&AS_ENUM(owner)->methods, OBJ_VAL(name), method);
            /* A member cache keyed on the old id can no longer be reached, so
             * anything memoised before this method existed is stale by
             * construction. Only runs while the enum is being defined. */
            AS_ENUM(owner)->shapeId = jaiFreshShapeId();
        } else {
            THROW(vm.cTypeError,
                  "METHOD expected a class, trait, or enum under the closure");
        }
        LOAD_STATE();
        DROP(1);
        VM_NEXT();
    }

    VM_CASE(OP_FIELD_DEF): {
        ObjString *name = AS_STRING(READ_CONST());
        uint8_t info = READ_BYTE();
        SAVE_STATE();
        Value classValue = PEEK(0);
        if (!IS_CLASS(classValue)) {
            THROW(vm.cTypeError, "FIELD_DEF expected a class on the stack");
        }
        if (!classDeclareField(AS_CLASS(classValue), name, info)) goto vmThrow;
        LOAD_STATE();
        VM_NEXT();
    }

    VM_CASE(OP_GET_FIELD): {
        uint32_t nameIdx = READ_U24();
        uint16_t cacheIdx = READ_U16();
        Value receiver = PEEK(0);

        if (IS_INSTANCE(receiver)) {
            ObjInstance *instance = AS_INSTANCE(receiver);
            InlineCache *ic = cacheAt(frameChunk(frame), cacheIdx);
            if (ic != NULL && instance->klass != NULL && ic->state != IC_EMPTY) {
                for (int w = 0; w < ic->count; w++) {
                    if (ic->shapeId[w] != instance->klass->shapeId) continue;
                    if (ic->payload[w] >= instance->fieldCount) break;
                    vm.icHits++;
                    stackTop[-1] = instance->fields[ic->payload[w]];
                    VM_NEXT();
                }
            }
            vm.icMisses++;
        }

        if (IS_ENUM(receiver)) {
            InlineCache *ic = cacheAt(frameChunk(frame), cacheIdx);
            if (ic != NULL && ic->state != IC_EMPTY) {
                ObjEnum *en = AS_ENUM(receiver);
                for (int w = 0; w < ic->count; w++) {
                    /* Shape ids are unique across classes and enums alike, so
                     * an instance way can never answer here by accident. */
                    if (ic->shapeId[w] != en->shapeId) continue;
                    vm.icHits++;
                    stackTop[-1] = ic->cached[w];
                    VM_NEXT();
                }
            }
            vm.icMisses++;
        }

        SAVE_STATE();
        ObjString *name = AS_STRING(constants[nameIdx]);
        Value result;
        if (!getPropertyInto(receiver, name, &result, true,
                             cacheAt(frameChunk(frame), cacheIdx))) {
            goto vmThrow;
        }
        LOAD_STATE();
        stackTop[-1] = result;
        VM_NEXT();
    }

    /* GET_LOCAL S; GET_FIELD K,C fused (spec §3.3). The receiver never reaches
     * the stack, which is the point: `self.x` is the single most frequent pair
     * the VM executes. It is still a GC root — a frame's slots are part of the
     * value stack — so the slow path below is as safe as OP_GET_FIELD's. */
    VM_CASE(OP_GET_FIELD_LOCAL): {
        uint16_t slot = READ_U16();
        uint32_t nameIdx = READ_U24();
        uint16_t cacheIdx = READ_U16();
        Value receiver = slots[slot];

        if (IS_INSTANCE(receiver)) {
            ObjInstance *instance = AS_INSTANCE(receiver);
            InlineCache *ic = cacheAt(frameChunk(frame), cacheIdx);
            if (ic != NULL && instance->klass != NULL && ic->state != IC_EMPTY) {
                for (int w = 0; w < ic->count; w++) {
                    if (ic->shapeId[w] != instance->klass->shapeId) continue;
                    if (ic->payload[w] >= instance->fieldCount) break;
                    vm.icHits++;
                    PUSH(instance->fields[ic->payload[w]]);
                    VM_NEXT();
                }
            }
            vm.icMisses++;
        }

        if (IS_ENUM(receiver)) {
            InlineCache *ic = cacheAt(frameChunk(frame), cacheIdx);
            if (ic != NULL && ic->state != IC_EMPTY) {
                ObjEnum *en = AS_ENUM(receiver);
                for (int w = 0; w < ic->count; w++) {
                    /* Shape ids are unique across classes and enums alike, so
                     * an instance way can never answer here by accident. */
                    if (ic->shapeId[w] != en->shapeId) continue;
                    vm.icHits++;
                    PUSH(ic->cached[w]);
                    VM_NEXT();
                }
            }
            vm.icMisses++;
        }

        SAVE_STATE();
        ObjString *name = AS_STRING(constants[nameIdx]);
        Value result;
        if (!getPropertyInto(receiver, name, &result, true,
                             cacheAt(frameChunk(frame), cacheIdx))) {
            goto vmThrow;
        }
        LOAD_STATE();
        PUSH(result);
        VM_NEXT();
    }

    VM_CASE(OP_SET_FIELD): {
        uint32_t nameIdx = READ_U24();
        uint16_t cacheIdx = READ_U16();
        Value receiver = PEEK(1);
        Value value = PEEK(0);

        if (IS_INSTANCE(receiver)) {
            ObjInstance *instance = AS_INSTANCE(receiver);
            InlineCache *ic = cacheAt(frameChunk(frame), cacheIdx);
            if (ic != NULL && instance->klass != NULL && ic->state != IC_EMPTY) {
                for (int w = 0; w < ic->count; w++) {
                    if (ic->shapeId[w] != instance->klass->shapeId) continue;
                    if (ic->payload[w] >= instance->fieldCount) break;
                    /* The kind guard has to be here too, not only in
                     * jaiSetProperty: once a site's cache is warm every later
                     * write takes this path and never reaches it. A field's
                     * slot is its index in klass->fields by construction
                     * (classDeclareField assigns slot = fieldCount as it
                     * appends), so this is one indexed load, not a lookup. */
                    if (ic->payload[w] < instance->klass->fieldCount) {
                        const FieldInfo *fi =
                            &instance->klass->fields[ic->payload[w]];
                        if (!jaiKindAccepts(fi->typeId, value)) {
                            SAVE_STATE();
                            (void)throwFieldKind(fi, value);
                            goto vmThrow;
                        }
                    }
                    vm.icHits++;
                    instance->fields[ic->payload[w]] = value;
                    DROP(2);
                    VM_NEXT();
                }
            }
            vm.icMisses++;
        }

        SAVE_STATE();
        ObjString *name = AS_STRING(constants[nameIdx]);
        if (!jaiSetProperty(receiver, name, value)) goto vmThrow;

        /* Cache only a plain field write: a setter has to keep running. */
        if (IS_INSTANCE(receiver)) {
            ObjInstance *instance = AS_INSTANCE(receiver);
            ObjClass *klass = instance->klass;
            Value setter;
            InlineCache *ic = cacheAt(frameChunk(frame), cacheIdx);
            const FieldInfo *field = jaiClassFieldInfo(klass, name);
            if (ic != NULL && field != NULL && klass != NULL &&
                !jaiTableGetInterned(&klass->setters, name, &setter) &&
                ic->state != IC_MEGA) {
                if (ic->count < JAI_IC_WAYS) {
                    ic->shapeId[ic->count] = klass->shapeId;
                    ic->payload[ic->count] = field->slot;
                    ic->cached[ic->count] = NULL_VAL;
                    ic->count++;
                    ic->state = (ic->count == 1) ? IC_MONO : IC_POLY;
                } else {
                    ic->state = IC_MEGA;
                }
            }
        }
        LOAD_STATE();
        DROP(2);
        VM_NEXT();
    }

    VM_CASE(OP_GET_SUPER): {
        ObjString *name = AS_STRING(READ_CONST());
        SAVE_STATE();
        Value receiver = PEEK(0);
        ObjClass *start = NULL;
        if (IS_INSTANCE(receiver)) start = AS_INSTANCE(receiver)->klass;
        else if (IS_CLASS(receiver)) start = AS_CLASS(receiver);
        if (start == NULL || start->superclass == NULL) {
            THROW(vm.cRuntimeError, "'super.%s' has no superclass", name->chars);
        }
        Value method;
        if (!findMethod(start->superclass, name, &method)) {
            THROW(vm.cAttributeError, "superclass '%s' has no member '%s'",
                  start->superclass->name != NULL ? start->superclass->name->chars
                                                  : "?",
                  name->chars);
        }
        ObjBound *bound = jaiBoundNew(receiver, method);
        LOAD_STATE();
        stackTop[-1] = OBJ_VAL(bound);
        VM_NEXT();
    }

    VM_CASE(OP_NEW): {
        Value classRef = READ_CONST();
        int argc = READ_BYTE();
        SAVE_STATE();

        Value classValue = classRef;
        if (IS_STRING(classRef)) {
            Value found;
            if ((frame->module != NULL &&
                 jaiModuleGet(frame->module, AS_STRING(classRef), &found)) ||
                (vm.builtins != NULL &&
                 jaiModuleGet(vm.builtins, AS_STRING(classRef), &found))) {
                classValue = found;
            } else {
                THROW(vm.cNameError, "undefined class '%s'",
                      AS_CSTRING(classRef));
            }
        }
        if (!IS_CLASS(classValue)) {
            THROW(vm.cTypeError, "NEW operand is not a class");
        }
        /* The arguments are already on the stack but the callee slot is not,
         * so open one below them for the instance. */
        if (!ensureStack(1)) goto vmThrow;
        Value *args = vm.stackTop - argc;
        memmove(args + 1, args, sizeof(Value) * (size_t)argc);
        args[0] = classValue;
        vm.stackTop++;
        if (invokeCallable(classValue, argc) == CALL_ERROR) goto vmThrow;
        LOAD_STATE();
        VM_NEXT();
    }

    VM_CASE(OP_ENUM_NEW): {
        Value enumRef = READ_CONST();
        int tag = READ_BYTE();
        int argc = READ_BYTE();
        SAVE_STATE();

        Value enumValue = enumRef;
        if (IS_STRING(enumRef)) {
            Value found;
            if ((frame->module != NULL &&
                 jaiModuleGet(frame->module, AS_STRING(enumRef), &found)) ||
                (vm.builtins != NULL &&
                 jaiModuleGet(vm.builtins, AS_STRING(enumRef), &found))) {
                enumValue = found;
            }
        }
        if (!IS_ENUM(enumValue)) {
            THROW(vm.cTypeError, "ENUM_NEW operand is not an enum");
        }
        ObjEnum *enumType = AS_ENUM(enumValue);
        if (tag >= (int)enumType->variantCount) {
            THROW(vm.cRuntimeError, "enum '%s' has no variant %d",
                  enumType->name != NULL ? enumType->name->chars : "?", tag);
        }
        /* Share the one value of a payload-less variant, exactly as
         * `enumMember` does: two spellings of `Color.Red` must be the same
         * object or `is` stops meaning identity (spec §4.2). */
        EnumVariant *variant = &enumType->variants[tag];
        ObjEnumVal *value;
        if (variant->arity == 0 && argc == 0) {
            if (variant->unit == NULL) {
                variant->unit = jaiEnumValNew(enumType, (uint16_t)tag, NULL, 0);
            }
            value = variant->unit;
        } else {
            value = jaiEnumValNew(enumType, (uint16_t)tag, vm.stackTop - argc, argc);
        }
        LOAD_STATE();
        DROP(argc);
        PUSH(OBJ_VAL(value));
        VM_NEXT();
    }

    VM_CASE(OP_ENUM_TAG): {
        Value value = PEEK(0);
        if (!IS_ENUM_VAL(value)) {
            THROW(vm.cTypeError, "ENUM_TAG expected an enum value, not '%s'",
                  jaiTypeNameStatic(value));
        }
        PUSH(INT_VAL(AS_ENUM_VAL(value)->tag));
        VM_NEXT();
    }

    VM_CASE(OP_ENUM_FIELD): {
        uint8_t index = READ_BYTE();
        Value value = PEEK(0);
        if (!IS_ENUM_VAL(value) || index >= AS_ENUM_VAL(value)->count) {
            THROW(vm.cIndexError, "enum payload field %u is out of range",
                  (unsigned)index);
        }
        PUSH(AS_ENUM_VAL(value)->payload[index]);
        VM_NEXT();
    }

    VM_CASE(OP_IS_INSTANCE): {
        Value typeConstant = READ_CONST();
        stackTop[-1] = BOOL_VAL(valueMatchesType(stackTop[-1], typeConstant));
        VM_NEXT();
    }

    /* --- exceptions and defer (spec §3.8) --- */

    VM_CASE(OP_THROW): {
        Value exception = POP();
        SAVE_STATE();
        if (IS_CLASS(exception)) {
            /* `throw SomeError` without a call: construct it here so handlers
             * always see an instance. */
            Value constructed;
            if (!jaiCallValue(exception, 0, NULL, &constructed)) goto vmThrow;
            exception = constructed;
        }
        (void)jaiThrowValue(exception);
        goto vmThrow;
    }

    VM_CASE(OP_RERAISE):
        SAVE_STATE();
        if (IS_NULL(vm.pendingException)) {
            THROW(vm.cRuntimeError, "no exception to re-raise");
        }
        (void)jaiThrowValue(vm.pendingException);
        goto vmThrow;

    VM_CASE(OP_PUSH_HANDLER): {
        int16_t offset = READ_I16();
        uint32_t typeConst = READ_U24();
        ExcHandler handler;
        handler.handlerOffset = (uint32_t)((ip + offset) - frameChunk(frame)->code);
        handler.typeConst = typeConst;
        handler.frameIndex = vm.frameCount - 1;
        handler.stackTop = stackTop;
        SAVE_STATE();
        JAI_VEC_PUSH(ExcHandler, &vm.handlers, handler);
        VM_NEXT();
    }

    VM_CASE(OP_POP_HANDLER):
        if (vm.handlers.count > frame->handlerBase) vm.handlers.count--;
        VM_NEXT();

    VM_CASE(OP_PUSH_FINALLY): {
        int16_t offset = READ_I16();
        ExcHandler handler;
        handler.handlerOffset = (uint32_t)((ip + offset) - frameChunk(frame)->code);
        handler.typeConst = JAI_HANDLER_FINALLY;
        handler.frameIndex = vm.frameCount - 1;
        handler.stackTop = stackTop;
        SAVE_STATE();
        JAI_VEC_PUSH(ExcHandler, &vm.handlers, handler);
        VM_NEXT();
    }

    VM_CASE(OP_END_FINALLY):
        /* Reached either by falling out of the try (nothing to do) or by the
         * unwinder, which recorded that the exception is still in flight. */
        if (sFinallyPending > 0) {
            sFinallyPending--;
            SAVE_STATE();
            (void)jaiThrowValue(vm.pendingException);
            goto vmThrow;
        }
        VM_NEXT();

    VM_CASE(OP_PUSH_DEFER): {
        Value deferred = READ_CONST();
        SAVE_STATE();
        if (IS_FUNCTION(deferred)) {
            ObjClosure *thunk = jaiClosureNew(AS_FUNCTION(deferred));
            /* emitDefer gave the thunk the enclosing function's upvalue count
             * and indices, so the whole array is shared rather than captured. */
            for (int i = 0; i < thunk->upvalueCount &&
                            i < frame->closure->upvalueCount; i++) {
                thunk->upvalues[i] = frame->closure->upvalues[i];
            }
            deferred = OBJ_VAL(thunk);
        } else if (!valueIsCallable(deferred)) {
            /* The constant is a placeholder; the closure was built by a
             * preceding OP_CLOSURE and is on the stack. */
            deferred = vm.stackTop > vm.stack ? vm.stackTop[-1] : NULL_VAL;
        }
        if (!valueIsCallable(deferred)) {
            THROW(vm.cTypeError, "defer requires a callable");
        }
        JAI_VEC_PUSH(Value, &vm.defers, deferred);
        LOAD_STATE();
        VM_NEXT();
    }

    VM_CASE(OP_RUN_DEFERS):
        SAVE_STATE();
        (void)runFrameDefers(frame);
        if (vm.hasException) goto vmThrow;
        LOAD_STATE();
        VM_NEXT();

    VM_CASE(OP_MATCH_EXC): {
        Value typeConstant = READ_CONST();
        PUSH(BOOL_VAL(valueMatchesType(vm.pendingException, typeConstant)));
        VM_NEXT();
    }

    VM_CASE(OP_GET_EXC):
        PUSH(vm.pendingException);
        VM_NEXT();

    /* --- pattern matching (spec §3.9) --- */

    VM_CASE(OP_MATCH_CONST): {
        Value expected = READ_CONST();
        int16_t offset = READ_I16();
        SAVE_STATE();
        bool equal = jaiValuesEqual(PEEK(0), expected);
        if (vm.hasException) goto vmThrow;
        LOAD_STATE();
        if (!equal) ip += offset;
        VM_NEXT();
    }

    VM_CASE(OP_MATCH_CONST_POP): {
        Value expected = READ_CONST();
        int16_t offset = READ_I16();
        SAVE_STATE();
        bool equal = jaiValuesEqual(PEEK(0), expected);
        if (vm.hasException) goto vmThrow;
        LOAD_STATE();
        if (equal) {
            DROP(1);
        } else {
            ip += offset;
        }
        VM_NEXT();
    }

    VM_CASE(OP_MATCH_RANGE): {
        Value low = READ_CONST();
        Value high = READ_CONST();
        bool inclusive = READ_BYTE() != 0;
        int16_t offset = READ_I16();
        Value subject = PEEK(0);

        int cmpLow = 0, cmpHigh = 0;
        SAVE_STATE();
        bool ordered = jaiValueCompare(subject, low, &cmpLow) &&
                       jaiValueCompare(subject, high, &cmpHigh);
        if (vm.hasException) goto vmThrow;
        LOAD_STATE();
        bool inside = ordered && cmpLow >= 0 &&
                      (inclusive ? cmpHigh <= 0 : cmpHigh < 0);
        if (!inside) ip += offset;
        VM_NEXT();
    }

    VM_CASE(OP_MATCH_RANGE_POP): {
        Value low = READ_CONST();
        Value high = READ_CONST();
        bool inclusive = READ_BYTE() != 0;
        int16_t offset = READ_I16();
        Value subject = PEEK(0);

        int cmpLow = 0, cmpHigh = 0;
        SAVE_STATE();
        bool ordered = jaiValueCompare(subject, low, &cmpLow) &&
                       jaiValueCompare(subject, high, &cmpHigh);
        if (vm.hasException) goto vmThrow;
        LOAD_STATE();
        bool inside = ordered && cmpLow >= 0 &&
                      (inclusive ? cmpHigh <= 0 : cmpHigh < 0);
        if (inside) {
            DROP(1);
        } else {
            ip += offset;
        }
        VM_NEXT();
    }

    VM_CASE(OP_MATCH_TYPE): {
        Value typeConstant = READ_CONST();
        int16_t offset = READ_I16();
        if (!valueMatchesType(PEEK(0), typeConstant)) ip += offset;
        VM_NEXT();
    }

    VM_CASE(OP_MATCH_TYPE_POP): {
        Value typeConstant = READ_CONST();
        int16_t offset = READ_I16();
        if (valueMatchesType(PEEK(0), typeConstant)) {
            DROP(1);
        } else {
            ip += offset;
        }
        VM_NEXT();
    }

    VM_CASE(OP_MATCH_SEQ): {
        int count = READ_BYTE();
        bool hasRest = READ_BYTE() != 0;
        int16_t offset = READ_I16();
        Value subject = PEEK(0);

        int length = -1;
        if (IS_LIST(subject))       length = AS_LIST(subject)->count;
        else if (IS_TUPLE(subject)) length = (int)AS_TUPLE(subject)->count;

        bool matched = length >= 0 &&
                       (hasRest ? (length >= count) : (length == count));
        if (!matched) ip += offset;
        VM_NEXT();
    }

    VM_CASE(OP_MATCH_SEQ_POP): {
        int count = READ_BYTE();
        bool hasRest = READ_BYTE() != 0;
        int16_t offset = READ_I16();
        Value subject = PEEK(0);

        int length = -1;
        if (IS_LIST(subject))       length = AS_LIST(subject)->count;
        else if (IS_TUPLE(subject)) length = (int)AS_TUPLE(subject)->count;

        bool matched = length >= 0 &&
                       (hasRest ? (length >= count) : (length == count));
        if (matched) {
            DROP(1);
        } else {
            ip += offset;
        }
        VM_NEXT();
    }

    VM_CASE(OP_MATCH_FIELDS): {
        Value fieldNames = READ_CONST();
        int16_t offset = READ_I16();
        Value subject = PEEK(0);
        SAVE_STATE();

        const Value *names = NULL;
        int count = 0;
        if (IS_TUPLE(fieldNames)) {
            names = AS_TUPLE(fieldNames)->items;
            count = (int)AS_TUPLE(fieldNames)->count;
        } else if (IS_STRING(fieldNames)) {
            names = &fieldNames;
            count = 1;
        }
        if (!IS_INSTANCE(subject) || names == NULL) {
            ip += offset;
            VM_NEXT();
        }

        ObjInstance *instance = AS_INSTANCE(subject);
        Value extracted[JAI_MAX_ARGS];
        if (count > JAI_MAX_ARGS) {
            THROW(vm.cRuntimeError, "too many fields in a class pattern");
        }
        bool ok = true;
        for (int i = 0; i < count && ok; i++) {
            if (!IS_STRING(names[i])) { ok = false; break; }
            const FieldInfo *field = jaiClassFieldInfo(instance->klass,
                                                       AS_STRING(names[i]));
            if (field == NULL || field->slot >= instance->fieldCount) {
                ok = false;
                break;
            }
            extracted[i] = instance->fields[field->slot];
        }
        if (!ok) {
            ip += offset;
            VM_NEXT();
        }
        if (!ensureStack(count)) goto vmThrow;
        LOAD_STATE();
        for (int i = 0; i < count; i++) PUSH(extracted[i]);
        VM_NEXT();
    }

    VM_CASE(OP_BIND): {
        uint16_t slot = READ_U16();
        slots[slot] = POP();
        VM_NEXT();
    }

    /* --- modules and misc (spec §3.10) --- */

    VM_CASE(OP_IMPORT): {
        ObjString *path = AS_STRING(READ_CONST());
        SAVE_STATE();
        char dirBuffer[JAI_MAX_PATH];
        const char *fromDir = NULL;
        if (frame->module != NULL && frame->module->path != NULL &&
            frame->module->path->length > 0) {
            jaiPathDirname(dirBuffer, sizeof dirBuffer,
                           frame->module->path->chars);
            fromDir = dirBuffer;
        }
        ObjModule *imported = jaiImportModule(path->chars, fromDir);
        if (imported == NULL) {
            if (!vm.hasException) {
                (void)jaiThrow(vm.cImportError, "cannot import module '%s'",
                               path->chars);
            }
            goto vmThrow;
        }
        LOAD_STATE();
        PUSH(OBJ_VAL(imported));
        VM_NEXT();
    }

    VM_CASE(OP_IMPORT_FROM): {
        ObjString *name = AS_STRING(READ_CONST());
        SAVE_STATE();
        Value moduleValue = PEEK(0);
        if (!IS_MODULE(moduleValue)) {
            THROW(vm.cImportError, "'from' import expected a module, not '%s'",
                  jaiTypeNameStatic(moduleValue));
        }
        ObjModule *module = AS_MODULE(moduleValue);
        Value member;
        bool hidden = false;
        if (!moduleMember(module, name, &member, &hidden)) {
            /* E0802: the name exists but is not part of the module's surface,
             * or it does not exist at all. */
            THROW(vm.cImportError, "'%s' is not exported by module '%s'",
                  name->chars, module->name != NULL ? module->name->chars : "?");
        }
        LOAD_STATE();
        PUSH(member);
        VM_NEXT();
    }

    VM_CASE(OP_EXPORT): {
        ObjString *name = AS_STRING(READ_CONST());
        SAVE_STATE();
        if (frame->module == NULL) {
            THROW(vm.cRuntimeError, "no module in scope to export '%s' from",
                  name->chars);
        }
        jaiGCPushRoot(OBJ_VAL(frame->module));
        (void)jaiTableSetInterned(&frame->module->exports, name, BOOL_VAL(true));
        jaiGCPopRoot();
        LOAD_STATE();
        VM_NEXT();
    }

    VM_CASE(OP_ASSERT_FAIL): {
        Value message = READ_CONST();
        SAVE_STATE();
        const char *text = "assertion failed";
        if (IS_STRING(message)) {
            text = AS_CSTRING(message);
        } else if (stackTop > slots && IS_STRING(PEEK(0))) {
            text = AS_CSTRING(PEEK(0));
        }
        THROW(vm.cAssertionError, "%s", text);
    }

    VM_CASE(OP_TYPE_GUARD): {
        Value typeConstant = READ_CONST();
        Value subject = PEEK(0);
        if (!valueMatchesType(subject, typeConstant)) {
            /* An int arriving at a float boundary widens (spec §2.2), the same
             * as it would have where the checker could see it. Only the bare
             * `float` guard converts: `x is float` and `match` ask what a value
             * *is*, and an int is not a float, so valueMatchesType above is
             * left alone. */
            if (IS_INT(subject) && IS_STRING(typeConstant) &&
                AS_STRING(typeConstant)->length == 5 &&
                memcmp(AS_STRING(typeConstant)->chars, "float", 5) == 0) {
                stackTop[-1] = FLOAT_VAL((double)AS_INT(subject));
                VM_NEXT();
            }
            THROW(vm.cTypeError, "expected '%s' but got '%s'",
                  typeConstantName(typeConstant), jaiTypeNameStatic(subject));
        }
        VM_NEXT();
    }

    VM_CASE(OP_HALT):
        SAVE_STATE();
        while (vm.frameCount > baseFrameCount) {
            CallFrame *halting = &vm.frames[vm.frameCount - 1];
            closeUpvalues(halting->base);
            vm.stackTop = halting->base;
            vm.frameCount--;
        }
        vm.handlers.count = 0;
        vm.defers.count = 0;
        *vm.stackTop++ = NULL_VAL;
        return JAI_RUN_OK;

#if !JAI_COMPUTED_GOTO
    default:
        SAVE_STATE();
        (void)jaiThrow(vm.cRuntimeError, "unknown opcode 0x%02x",
                       (unsigned)instStart[0]);
        goto vmThrow;
#endif

    }   /* VM_DISPATCH */

vmThrow: {
        if (!vm.hasException) {
            /* A helper failed without raising; make the failure visible rather
             * than resuming with a corrupt stack. */
            (void)jaiThrow(vm.cRuntimeError,
                           "internal error: failed operation raised nothing");
        }
        CallFrame *faulting = &vm.frames[vm.frameCount - 1];
        Chunk *chunk = frameChunk(faulting);
        ptrdiff_t at = instStart - chunk->code;
        if (at < 0 || at > chunk->count) at = 0;

        if (!unwindToHandler(baseFrameCount, (uint32_t)at)) {
            return JAI_RUN_RUNTIME_ERROR;
        }
        LOAD_STATE();
        VM_NEXT();
    }
}

JaiRunResult run(int baseFrameCount) {
    if (sRunDepth >= JAI_MAX_NESTED_RUN) {
        (void)jaiThrow(vm.cRecursionError,
                       "maximum native re-entry depth exceeded (%d)",
                       JAI_MAX_NESTED_RUN);
        while (vm.frameCount > baseFrameCount) popFrameForUnwind();
        return JAI_RUN_RUNTIME_ERROR;
    }
    sRunDepth++;
    JaiRunResult result = runLoop(baseFrameCount);
    sRunDepth--;
    return result;
}
