/* jit_func.c — compiling a whole function, calls and all.
 *
 * The loop tier in jit_loop.c compiles a counted loop out of a function that
 * the interpreter still owns. This one compiles the function itself, into an
 * ordinary arm64 routine with a native calling convention, so that a call it
 * makes to itself is a `bl` and not a CallFrame. That is the entire point: at
 * 15ns per interpreted call, `fib(30)` spends nearly all of its time in
 * pushFrame and OP_RETURN, and no amount of faster dispatch reaches C.
 *
 * The language it accepts is small and the restrictions are what make it
 * sound rather than what make it easy:
 *
 *   - integers only, and every operand is an int by construction: the entry
 *     guard checks the arguments, and the only values the body can make are
 *     results of int arithmetic or of a call to this same function
 *   - no allocation, no stores, no globals but this function's own name.
 *     A body that cannot write anything is a body whose partial execution is
 *     invisible, which is what lets a bail simply throw the work away and
 *     re-run the call interpreted
 *   - self-recursion only. A general call would need a compiled callee and an
 *     agreed convention between them; that comes later
 *
 * Values live in registers for the whole call. Locals get x19 upward and the
 * operand stack continues from there, all callee-saved, so a value that is
 * live across a recursive call survives it without a spill. That caps the
 * function at nine live values, which is generous for anything this tier can
 * compile and is checked rather than assumed.
 *
 * Two things can go wrong at run time, and both take the same exit: signed
 * overflow, and the native stack running low. Compiled code sets a flag and
 * returns; the entry point sees the flag, discards the result, and hands the
 * call back to the interpreter, which then produces the real OverflowError or
 * RecursionError with a traceback. The function is refused from then on --
 * once a body has bailed, compiling it again only buys another bail.
 */
#include "jit.h"

#include "jit_arm64.h"
#include "vm.h"

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>

#if (defined(__aarch64__) || defined(__arm64__))

#include <pthread.h>

/* ------------------------------------------------------------------ */
/* Run-time state shared with compiled code                             */
/* ------------------------------------------------------------------ */

/* Compiled code returns two words: the value in x0 and whether it gave up in
 * x1. AAPCS64 returns a 16-byte struct of integers in exactly that pair, so
 * the flag costs no memory traffic at all -- it used to be a global, which was
 * a store and a load on every single call. A recursive call propagates it:
 * a callee that bailed sends its caller straight to its own bail block, so the
 * whole recursion unwinds at once instead of computing on with a junk value. */
typedef struct { int64_t value; int64_t bailed; } JitResult;

/* The lowest stack address compiled code may use, computed from the thread's
 * real bounds rather than from wherever the compile happened to run. Deriving
 * it from the compiling frame's sp looked simpler and is wrong: a function
 * compiled near the top of the stack and later entered from deep inside the
 * interpreter would bail on entry every time, and silently refuse itself. */
static uintptr_t stackLimit(void) {
    pthread_t self = pthread_self();
    void  *top  = pthread_get_stackaddr_np(self);   /* high address */
    size_t size = pthread_get_stacksize_np(self);
    if (top == NULL || size == 0) return 0;
    /* The margin has to cover the deepest single compiled frame plus whatever
     * the interpreter needs to unwind and report the error afterwards. */
    return (uintptr_t)top - size + (256u * 1024u);
}

/* ------------------------------------------------------------------ */
/* Register plan                                                        */
/* ------------------------------------------------------------------ */

#define JIT_FIRST_SAVED 19u   /* x19..x27 are callee-saved and ours */
#define JIT_MAX_SAVED    9u
#define JIT_MAX_ARITY    4u   /* arguments arrive in x0..x3 */
#define JIT_MAX_INSTS  512u
#define JIT_SCRATCH_A    9u
#define JIT_SCRATCH_B   10u

/* One entry of the compile-time operand stack. A `self` entry is the callee of
 * a recursive call: it names this function and occupies no register, which is
 * why register numbers are derived from the count of value entries below an
 * entry rather than from its depth. */
typedef enum {
    SLOT_INT,     /* int64 payload, in an X register */
    SLOT_FLOAT,   /* double, held as raw bits in an X register */
    SLOT_INST,    /* ObjInstance *, raw, of a class fixed at compile time */
    SLOT_SELF,    /* this function, as a callee; occupies no register */
    SLOT_OPAQUE,  /* present in a register, but nothing may be done with it */
    SLOT_CLOSURE  /* the running closure, for reaching its upvalues */
} SlotKind;

/* Floats live in X registers and visit d0/d1 only for the arithmetic itself.
 * Three extra fmovs per operation, against a second register bank with its own
 * allocator, its own save set and its own spill rules -- for a tier this young
 * the simpler thing that is obviously correct is worth more than the three
 * instructions. Nothing calls between the fmovs, so the scratch pair is safe. */
#define JIT_FSCRATCH_A 0u
#define JIT_FSCRATCH_B 1u

static bool holdsRegister(SlotKind k) { return k != SLOT_SELF; }

/* Two targets are not bytecode offsets at all. */
#define FIXUP_BAIL   UINT32_MAX
#define FIXUP_ENTRY  (UINT32_MAX - 1u)

typedef struct {
    int      instIndex;    /* which emitted instruction to patch */
    uint32_t targetOffset; /* bytecode offset, or FIXUP_BAIL / FIXUP_ENTRY */
    bool     conditional;  /* b.cond rather than b */
    int      depth;        /* value-stack depth where the branch leaves from */
} Fixup;

typedef struct {
    uint32_t  code[JIT_MAX_INSTS];
    unsigned  count;

    SlotKind  stack[JIT_MAX_SAVED];
    uint32_t  stackShape[JIT_MAX_SAVED];  /* class shapeId for SLOT_INST */
    ObjClass *stackClass[JIT_MAX_SAVED];
    SlotKind  localKind[JIT_MAX_SAVED + 1];
    uint32_t  localShape[JIT_MAX_SAVED + 1];
    ObjClass *localClass[JIT_MAX_SAVED + 1];
    Value    *observed;      /* the live arguments, for field-type feedback */
    bool      assumedIntReturn;
    unsigned  depth;       /* entries on the operand stack */
    unsigned  valueDepth;  /* of those, how many hold a register */
    unsigned  maxValue;    /* high-water mark, for the save set */

    Fixup     fixups[JIT_MAX_INSTS];
    unsigned  fixupCount;

    int      *offsetToInst;  /* bytecode offset -> instruction index, or -1 */
    int      *offsetToDepth; /* bytecode offset -> stack signature, or -1 */

    unsigned  arity;
    /* Slot 0 is the callee for a plain call and the RECEIVER for a method, so
     * a method's `self.x` reads it. When the body touches it, it becomes an
     * ordinary local and an extra incoming argument; when it does not -- every
     * plain function -- it costs nothing. */
    unsigned  base;        /* first slot held in a register: 0 or 1 */
    bool      usesSlot0;
    /* A body that reads an upvalue needs the closure itself, which is not any
     * slot: for a method slot 0 is the receiver, not the callee. It arrives as
     * one extra argument and takes the register just past the locals. */
    bool      usesUpvalues;
    bool      hasSelfCall;
    unsigned  locals;      /* slots base..base+locals-1 live in registers */
    unsigned  frameBytes;
    unsigned  savedCount;

    int       limitLiteral;  /* instruction index of the stack-limit word */
    int       bailBlock;     /* instruction index of the bail sequence */

    SlotKind  returnKind;
    uint32_t  returnShape;
    bool      sawReturn;

    bool      failed;
} Emit;

static void emit(Emit *e, uint32_t word) {
    if (e->count >= JIT_MAX_INSTS) { e->failed = true; return; }
    e->code[e->count++] = word;
}

static unsigned localReg(const Emit *e, unsigned slot) {
    return JIT_FIRST_SAVED + (slot - e->base);
}

/* Slots the body may name at all. */
static bool localInRange(const Emit *e, unsigned slot) {
    return slot >= e->base && slot < e->base + e->locals;
}

/* Slots whose value this call can be looked at, for type feedback. */
static bool localObserved(const Emit *e, unsigned slot) {
    return slot >= e->base && slot <= e->arity;
}

/* The register a new value entry would occupy. */
static unsigned closureReg(const Emit *e) {
    return JIT_FIRST_SAVED + e->locals;
}

static unsigned pushReg(const Emit *e) {
    return JIT_FIRST_SAVED + e->locals + (e->usesUpvalues ? 1u : 0u) +
           e->valueDepth;
}

static bool pushValue(Emit *e, SlotKind kind, uint32_t shape, ObjClass *klass) {
    if (e->depth >= JIT_MAX_SAVED) return false;
    if (e->locals + (e->usesUpvalues ? 1u : 0u) + e->valueDepth + 1 >
        JIT_MAX_SAVED) {
        return false;
    }
    e->stackShape[e->depth] = shape;
    e->stackClass[e->depth] = klass;
    e->stack[e->depth++] = kind;
    e->valueDepth++;
    if (e->valueDepth > e->maxValue) e->maxValue = e->valueDepth;
    return true;
}

static bool pushSelf(Emit *e) {
    if (e->depth >= JIT_MAX_SAVED) return false;
    e->stack[e->depth++] = SLOT_SELF;
    return true;
}

/* Pop a value entry, reporting the register it was in and what it held. */
static bool popValue(Emit *e, unsigned *reg, SlotKind *kind) {
    if (e->depth == 0 || !holdsRegister(e->stack[e->depth - 1])) return false;
    e->depth--;
    e->valueDepth--;
    if (kind != NULL) *kind = e->stack[e->depth];
    *reg = JIT_FIRST_SAVED + e->locals + (e->usesUpvalues ? 1u : 0u) +
           e->valueDepth;
    return true;
}

/* Depth and the kind of every entry, in one word. Registers are assigned from
 * the depth and instructions are chosen from the kinds, so a join reached with
 * either one different is a join this tier cannot compile. */
static uint32_t stackSignature(const Emit *e) {
    uint32_t sig = e->depth & 0xfu;
    for (unsigned i = 0; i < e->depth && i < 9; i++) {
        sig |= ((uint32_t)e->stack[i] & 3u) << (4 + 2 * i);
    }
    return sig;
}

/* ------------------------------------------------------------------ */
/* Constants                                                            */
/* ------------------------------------------------------------------ */

/* Materialise a full 64-bit constant. One instruction for the small cases
 * that matter (`n < 2`, `n - 1`), up to four for anything else. */
static void emitConst64(Emit *e, unsigned rd, int64_t value) {
    if (value >= 0 && value <= 0xffff) {
        emit(e, jaiA64MovzX(rd, (unsigned)value, 0));
        return;
    }
    if (value < 0 && value >= -0x10000) {
        emit(e, jaiA64MovnX(rd, (unsigned)(~(uint64_t)value & 0xffffu)));
        return;
    }
    uint64_t bits = (uint64_t)value;
    emit(e, jaiA64MovzX(rd, (unsigned)(bits & 0xffffu), 0));
    for (unsigned shift = 1; shift < 4; shift++) {
        unsigned part = (unsigned)((bits >> (16 * shift)) & 0xffffu);
        if (part != 0) emit(e, jaiA64MovkX(rd, part, shift));
    }
}

/* ------------------------------------------------------------------ */
/* Prologue and epilogue                                                */
/* ------------------------------------------------------------------ */

static void emitSaveRestore(Emit *e, bool save) {
    for (unsigned i = 0; i < e->savedCount; i += 2) {
        unsigned r1 = JIT_FIRST_SAVED + i;
        int32_t at = (int32_t)(16 + 8 * i);
        if (i + 1 < e->savedCount) {
            emit(e, save ? jaiA64StpOff(r1, r1 + 1, 31, at)
                         : jaiA64LdpOff(r1, r1 + 1, 31, at));
        } else {
            emit(e, save ? jaiA64StrX(r1, 31, (unsigned)at)
                         : jaiA64LdrX(r1, 31, (unsigned)at));
        }
    }
}

static void emitEpilogue(Emit *e, unsigned bailed) {
    emit(e, jaiA64MovzX(1, bailed, 0));
    emitSaveRestore(e, false);
    emit(e, jaiA64LdpPost(29, 30, 31, (int32_t)e->frameBytes));
    emit(e, jaiA64Ret());
}

/* ------------------------------------------------------------------ */
/* The body                                                            */
/* ------------------------------------------------------------------ */

static void branchTo(Emit *e, uint32_t targetOffset, bool conditional,
                     unsigned cond) {
    if (e->fixupCount >= JIT_MAX_INSTS) { e->failed = true; return; }
    e->fixups[e->fixupCount].instIndex    = (int)e->count;
    e->fixups[e->fixupCount].targetOffset = targetOffset;
    e->fixups[e->fixupCount].conditional  = conditional;
    e->fixups[e->fixupCount].depth        = (int)stackSignature(e);
    e->fixupCount++;
    emit(e, conditional ? jaiA64BCond(cond, 0) : jaiA64B(0));
}

/* Branch to the bail block on `cond`. The block's index is not known yet, so
 * it is patched with the rest. */
static void branchOnCondition(Emit *e, unsigned cond) {
    if (e->fixupCount >= JIT_MAX_INSTS) { e->failed = true; return; }
    e->fixups[e->fixupCount].instIndex    = (int)e->count;
    e->fixups[e->fixupCount].targetOffset = FIXUP_BAIL;
    e->fixups[e->fixupCount].conditional  = true;
    e->fixups[e->fixupCount].depth        = -1;   /* bails do not join */
    e->fixupCount++;
    emit(e, jaiA64BCond(cond, 0));
}

static void branchOnOverflow(Emit *e) { branchOnCondition(e, JAI_A64_VS); }

/* The condition to branch on when the comparison is FALSE: the opcode jumps
 * over the taken side, `if (!taken) ip += offset`. */
static bool negatedCondition(uint8_t cmp, unsigned *out) {
    switch (cmp) {
    case OP_EQ: *out = JAI_A64_NE; return true;
    case OP_NE: *out = JAI_A64_EQ; return true;
    case OP_LT: *out = JAI_A64_GE; return true;
    case OP_LE: *out = JAI_A64_GT; return true;
    case OP_GT: *out = JAI_A64_LE; return true;
    case OP_GE: *out = JAI_A64_LT; return true;
    default: return false;
    }
}

/* Does this OP_GET_GLOBAL name the function being compiled? */
static bool globalIsSelf(ObjClosure *closure, uint32_t nameIdx) {
    ObjFunction *fn = closure->fn;
    if (fn->module == NULL) return false;
    if (nameIdx >= (uint32_t)fn->chunk.constants.count) return false;
    Value name = fn->chunk.constants.data[nameIdx];
    if (!IS_STRING(name)) return false;

    Value bound;
    if (!jaiModuleGet(fn->module, AS_STRING(name), &bound)) return false;
    return IS_CLOSURE(bound) && AS_CLOSURE(bound)->fn == fn;
}

static bool compileBody(Emit *e, ObjClosure *closure) {
    ObjFunction *fn = closure->fn;
    const uint8_t *code = fn->chunk.code;
    int count = fn->chunk.count;

    for (int off = 0; off < count && !e->failed;) {
        e->offsetToInst[off]  = (int)e->count;
        e->offsetToDepth[off] = (int)stackSignature(e);
        uint8_t op = code[off];

        switch (op) {
        case OP_GET_LOCAL: {
            unsigned slot = jaiReadU16(code + off + 1);
            if (!localInRange(e, slot)) return false;
            if (e->localKind[slot] == SLOT_OPAQUE) return false;
            if (slot == 0) e->usesSlot0 = true;
            if (!pushValue(e, e->localKind[slot], e->localShape[slot], e->localClass[slot])) return false;
            emit(e, jaiA64MovX(pushReg(e) - 1, localReg(e, slot)));
            off += 3;
            break;
        }

        case OP_INT: {
            int16_t k = jaiReadI16(code + off + 1);
            if (!pushValue(e, SLOT_INT, 0, NULL)) return false;
            emitConst64(e, pushReg(e) - 1, k);
            off += 3;
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
            if (e->localKind[slot] != e->stack[e->depth - 1]) return false;
            if (e->localShape[slot] != e->stackShape[e->depth - 1]) return false;
            emit(e, jaiA64MovX(localReg(e, slot), pushReg(e) - 1));
            off += 3;
            break;
        }

        case OP_POP: {
            unsigned r;
            if (!popValue(e, &r, NULL)) return false;
            off += 1;
            break;
        }

        case OP_CONST: {
            uint32_t idx = jaiReadU24(code + off + 1);
            if (idx >= (uint32_t)fn->chunk.constants.count) return false;
            Value k = fn->chunk.constants.data[idx];
            if (IS_INT(k)) {
                if (!pushValue(e, SLOT_INT, 0, NULL)) return false;
                emitConst64(e, pushReg(e) - 1, AS_INT(k));
            } else if (IS_FLOAT(k)) {
                /* The bits, not the number: a float lives in an X register
                 * exactly as it does in a Value's payload. */
                double d = AS_FLOAT(k);
                int64_t bits;
                memcpy(&bits, &d, sizeof bits);
                if (!pushValue(e, SLOT_FLOAT, 0, NULL)) return false;
                emitConst64(e, pushReg(e) - 1, bits);
            } else {
                return false;
            }
            off += 4;
            break;
        }

        case OP_JUMP: {
            int16_t jump = jaiReadI16(code + off + 1);
            branchTo(e, (uint32_t)((int32_t)(off + 3) + jump), false, 0);
            off += 3;
            break;
        }

        case OP_LOOP: {
            /* The interpreter runs a safepoint on the back edge; compiled code
             * cannot, so a compiled loop is not interruptible and does not get
             * sampled. Both are acceptable only because this tier bails on any
             * unbounded construct: the loop is over ints, cannot allocate, and
             * the stack guard still catches runaway recursion. Ctrl-C during a
             * long compiled loop waits for the loop, which is a real cost and
             * the reason the trip count is not unbounded in practice. */
            int16_t jump = jaiReadI16(code + off + 1);
            branchTo(e, (uint32_t)((int32_t)(off + 3) + jump), false, 0);
            off += 3;
            break;
        }

        case OP_MUL: {
            unsigned rb, ra;
            SlotKind kb, ka;
            if (!popValue(e, &rb, &kb)) return false;
            if (!popValue(e, &ra, &ka)) return false;
            if (ka != kb) return false;

            if (ka == SLOT_FLOAT) {
                if (!pushValue(e, SLOT_FLOAT, 0, NULL)) return false;
                unsigned rf = pushReg(e) - 1;
                emit(e, jaiA64FmovDX(JIT_FSCRATCH_A, ra));
                emit(e, jaiA64FmovDX(JIT_FSCRATCH_B, rb));
                emit(e, jaiA64FmulD(JIT_FSCRATCH_A, JIT_FSCRATCH_A,
                                    JIT_FSCRATCH_B));
                emit(e, jaiA64FmovXD(rf, JIT_FSCRATCH_A));
                off += 1;
                break;
            }
            if (ka != SLOT_INT) return false;
            if (!pushValue(e, SLOT_INT, 0, NULL)) return false;
            unsigned rd = pushReg(e) - 1;
            /* The product overflows exactly when the high half is not the low
             * half's sign bit replicated, so smulh and one shifted compare
             * decide it. mul must come after smulh reads its inputs, since rd
             * may be one of them. */
            emit(e, jaiA64SmulhX(JIT_SCRATCH_A, ra, rb));
            emit(e, jaiA64MulX(rd, ra, rb));
            emit(e, jaiA64SubsXAsr(31, JIT_SCRATCH_A, rd, 63));
            branchOnCondition(e, JAI_A64_NE);
            off += 1;
            break;
        }

        case OP_ADD:
        case OP_SUB:
        case OP_DIV: {
            unsigned rb, ra;
            SlotKind kb, ka;
            if (!popValue(e, &rb, &kb)) return false;
            if (!popValue(e, &ra, &ka)) return false;
            if (ka != kb) return false;   /* no implicit widening here */

            if (ka == SLOT_FLOAT) {
                if (!pushValue(e, SLOT_FLOAT, 0, NULL)) return false;
                unsigned rd = pushReg(e) - 1;
                emit(e, jaiA64FmovDX(JIT_FSCRATCH_A, ra));
                emit(e, jaiA64FmovDX(JIT_FSCRATCH_B, rb));
                emit(e, op == OP_ADD
                            ? jaiA64FaddD(JIT_FSCRATCH_A, JIT_FSCRATCH_A,
                                          JIT_FSCRATCH_B)
                        : op == OP_SUB
                            ? jaiA64FsubD(JIT_FSCRATCH_A, JIT_FSCRATCH_A,
                                          JIT_FSCRATCH_B)
                            : jaiA64FdivD(JIT_FSCRATCH_A, JIT_FSCRATCH_A,
                                          JIT_FSCRATCH_B));
                emit(e, jaiA64FmovXD(rd, JIT_FSCRATCH_A));
                off += 1;
                break;
            }

            /* Integer division is not here: it has a zero-divisor error and a
             * truncation rule of its own, and getting either wrong would be a
             * wrong answer rather than a decline. */
            if (ka != SLOT_INT || op == OP_DIV) return false;
            if (!pushValue(e, SLOT_INT, 0, NULL)) return false;
            unsigned rd = pushReg(e) - 1;
            emit(e, op == OP_ADD ? jaiA64AddsX(rd, ra, rb)
                                 : jaiA64SubsXReg(rd, ra, rb));
            branchOnOverflow(e);
            off += 1;
            break;
        }

        case OP_JUMP_IF_CMP_LOCAL_K: {
            uint8_t  cmp  = code[off + 1];
            unsigned slot = jaiReadU16(code + off + 2);
            uint32_t kIdx = jaiReadU24(code + off + 4);
            int16_t  jump = jaiReadI16(code + off + 7);
            uint32_t next = (uint32_t)(off + 9);

            unsigned cond;
            if (!negatedCondition(cmp, &cond)) return false;
            if (!localInRange(e, slot)) return false;
            if (e->localKind[slot] != SLOT_INT) return false;
            if (slot == 0) e->usesSlot0 = true;
            if (kIdx >= (uint32_t)fn->chunk.constants.count) return false;
            Value k = fn->chunk.constants.data[kIdx];
            if (!IS_INT(k)) return false;

            emitConst64(e, JIT_SCRATCH_A, AS_INT(k));
            emit(e, jaiA64SubsXReg(31, localReg(e, slot), JIT_SCRATCH_A));
            branchTo(e, (uint32_t)((int32_t)next + jump), true, cond);
            off += 9;
            break;
        }

        case OP_GET_FIELD_LOCAL: {
            unsigned slot    = jaiReadU16(code + off + 1);
            uint32_t nameIdx = jaiReadU24(code + off + 3);

            if (!localObserved(e, slot)) return false;   /* see below */
            if (e->localKind[slot] != SLOT_INST) return false;
            if (slot == 0) e->usesSlot0 = true;
            if (nameIdx >= (uint32_t)fn->chunk.constants.count) return false;
            Value nameVal = fn->chunk.constants.data[nameIdx];
            if (!IS_STRING(nameVal)) return false;

            const FieldInfo *info =
                jaiClassFieldInfo(e->localClass[slot], AS_STRING(nameVal));
            if (info == NULL || info->isStatic) return false;

            /* Which type this field holds is read off the live receiver, so
             * the tier specialises to what the program actually stores rather
             * than to a declaration. That is only possible for a parameter,
             * which is why the slot is capped at the arity above: a local
             * assigned further in has no value to look at yet. */
            Value seen = e->observed[slot];
            if (!IS_INSTANCE(seen)) return false;
            ObjInstance *inst = AS_INSTANCE(seen);
            if (info->slot >= inst->fieldCount) return false;
            Value fieldVal = inst->fields[info->slot];

            SlotKind kind;
            unsigned tag;
            if (IS_INT(fieldVal))        { kind = SLOT_INT;   tag = VAL_INT; }
            else if (IS_FLOAT(fieldVal)) { kind = SLOT_FLOAT; tag = VAL_FLOAT; }
            else return false;

            unsigned base = (unsigned)offsetof(ObjInstance, fields) +
                            (unsigned)info->slot * (unsigned)sizeof(Value);
            /* The tag is checked every time. A field is not typed by the
             * runtime, so a later int where a float was seen must bail rather
             * than be read as one. */
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, localReg(e, slot), base));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, tag));
            branchOnCondition(e, JAI_A64_NE);

            if (!pushValue(e, kind, 0, NULL)) return false;
            emit(e, jaiA64LdrX(pushReg(e) - 1, localReg(e, slot), base + 8));
            off += 8;
            break;
        }

        case OP_GET_LOCAL2: {
            unsigned a = jaiReadU16(code + off + 1);
            unsigned b = jaiReadU16(code + off + 3);
            for (unsigned k = 0; k < 2; k++) {
                unsigned slot = k == 0 ? a : b;
                if (!localInRange(e, slot)) return false;
                if (e->localKind[slot] == SLOT_OPAQUE) return false;
                if (slot == 0) e->usesSlot0 = true;
                if (!pushValue(e, e->localKind[slot], e->localShape[slot], e->localClass[slot])) {
                    return false;
                }
                emit(e, jaiA64MovX(pushReg(e) - 1, localReg(e, slot)));
            }
            off += 5;
            break;
        }

        case OP_SET_FIELD: {
            uint32_t nameIdx = jaiReadU24(code + off + 1);
            /* The stack is receiver then value, and both are dropped. The
             * receiver's class is read off its stack entry, not guessed from
             * the locals: two instance locals of different classes would make
             * any guess a silently wrong field offset. */
            if (e->depth < 2) return false;
            ObjClass *klass = e->stackClass[e->depth - 2];

            unsigned rv, rr;
            SlotKind kv, kr;
            if (!popValue(e, &rv, &kv)) return false;
            if (!popValue(e, &rr, &kr)) return false;
            if (kr != SLOT_INST) return false;
            /* Only a number goes in. Storing an object would put a pointer the
             * collector has not seen into a field, and while this tier cannot
             * allocate -- so nothing can move or be swept while it runs -- the
             * value would still have to be one the caller already had rooted.
             * Not worth the argument for what it buys. */
            if (kv != SLOT_INT && kv != SLOT_FLOAT) return false;
            if (nameIdx >= (uint32_t)fn->chunk.constants.count) return false;
            Value nameVal = fn->chunk.constants.data[nameIdx];
            if (!IS_STRING(nameVal)) return false;

            if (klass == NULL) return false;
            const FieldInfo *info = jaiClassFieldInfo(klass, AS_STRING(nameVal));
            if (info == NULL || info->isStatic) return false;

            unsigned base = (unsigned)offsetof(ObjInstance, fields) +
                            (unsigned)info->slot * (unsigned)sizeof(Value);
            emit(e, jaiA64MovzX(JIT_SCRATCH_A,
                                kv == SLOT_INT ? VAL_INT : VAL_FLOAT, 0));
            emit(e, jaiA64StrW(JIT_SCRATCH_A, rr, base));
            emit(e, jaiA64StrX(rv, rr, base + 8));
            off += 6;
            break;
        }

        case OP_RETURN_NULL: {
            /* Only an initializer, which yields the object it initialised --
             * that is what makes `Point(1, 2)` an expression. A plain function
             * returning null has no int64 to hand back. */
            if ((fn->flags & FN_INIT) == 0) return false;
            if (!localInRange(e, 0) || e->localKind[0] != SLOT_INST) return false;
            e->usesSlot0 = true;
            if (e->sawReturn && e->returnKind != SLOT_INST) return false;
            e->sawReturn  = true;
            e->returnKind = SLOT_INST;
            e->returnShape = e->localShape[0];
            emit(e, jaiA64MovX(0, localReg(e, 0)));
            emitEpilogue(e, 0);
            off += 1;
            break;
        }

        case OP_GET_UPVALUE: {
            unsigned index = code[off + 1];
            if (index >= (unsigned)fn->upvalueCount) return false;
            if (!e->usesUpvalues) return false;   /* decided before this pass */

            /* closure->upvalues[index]->location, then the Value there. The
             * upvalue may still be open, pointing into the VM stack, so the
             * location is followed rather than assumed closed. */
            emit(e, jaiA64LdrX(JIT_SCRATCH_A, closureReg(e),
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
            if (IS_INT(seen))        { kind = SLOT_INT;   tag = VAL_INT; }
            else if (IS_FLOAT(seen)) { kind = SLOT_FLOAT; tag = VAL_FLOAT; }
            else return false;

            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_B, tag));
            branchOnCondition(e, JAI_A64_NE);
            if (!pushValue(e, kind, 0, NULL)) return false;
            emit(e, jaiA64LdrX(pushReg(e) - 1, JIT_SCRATCH_A, 8));
            off += 2;
            break;
        }

        case OP_GET_GLOBAL: {
            uint32_t nameIdx = jaiReadU24(code + off + 1);
            if (!globalIsSelf(closure, nameIdx)) return false;
            if (!pushSelf(e)) return false;
            off += 6;
            break;
        }

        case OP_CALL: {
            unsigned argc = code[off + 1];
            if (argc != e->arity) return false;
            /* Recorded, not rejected here: the measuring pass always runs
             * with slot 0 available, so testing the base in this pass aborted
             * every recursive function and quietly cost fib_recursive its
             * compiled form -- 8.8ms back to 83ms. The decision belongs after
             * the base is chosen. */
            e->hasSelfCall = true;
            if (e->depth < argc + 1) return false;
            if (e->stack[e->depth - argc - 1] != SLOT_SELF) return false;

            /* The arguments sit in the top `argc` value registers, in order.
             * They move to x0.. which nothing else is using. */
            unsigned first = JIT_FIRST_SAVED + e->locals +
                             (e->usesUpvalues ? 1u : 0u) + e->valueDepth - argc;
            for (unsigned i = 0; i < argc; i++) {
                emit(e, jaiA64MovX(i, first + i));
            }

            /* To instruction 0, the prologue -- NOT to the first instruction
             * of the body. A recursive call that skipped the prologue would
             * not save x19 upward, so the callee would overwrite the caller's
             * locals and the recursion would never terminate. */
            if (!e->sawReturn) e->assumedIntReturn = true;

            branchTo(e, FIXUP_ENTRY, false, 0);
            e->code[e->count - 1] = jaiA64Bl(0);
            /* x1 carries the callee's verdict; a bail there is a bail here. */
            emit(e, jaiA64SubsXImm(31, 1, 0));
            branchOnCondition(e, JAI_A64_NE);

            e->depth      -= argc + 1;
            e->valueDepth -= argc;
            if (!pushValue(e, e->returnKind, e->returnShape, NULL)) return false;
            emit(e, jaiA64MovX(pushReg(e) - 1, 0));
            off += 2;
            break;
        }

        case OP_RETURN: {
            unsigned r;
            SlotKind k;
            if (!popValue(e, &r, &k)) return false;
            /* One return kind per function: the entry point rebuilds a Value
             * from it, and it cannot rebuild two. */
            if (e->sawReturn && e->returnKind != k) return false;
            e->sawReturn = true;
            e->returnKind = k;
            emit(e, jaiA64MovX(0, r));
            emitEpilogue(e, 0);
            off += 1;
            break;
        }

        default:
            if (getenv("JAI_JIT_WHY")) {
                fprintf(stderr, "[jit] %s declined at %s\n",
                        fn->name ? fn->name->chars : "<anon>",
                        jaiOpName((OpCode)op));
            }
            return false;   /* an opcode this tier does not speak */
        }
    }
    return !e->failed;
}

/* ------------------------------------------------------------------ */
/* Assembly                                                             */
/* ------------------------------------------------------------------ */


/* The kinds of the parameters, read off the arguments this call was made with.
 * Everything downstream is specialised to them, and the entry guard re-checks
 * them on every later call. */
static bool seedLocals(Emit *e, Value *slotBase) {
    e->observed = slotBase;
    for (unsigned i = 0; i < e->base + e->locals; i++) {
        e->localKind[i]  = SLOT_INT;
        e->localShape[i] = 0;
        e->localClass[i] = NULL;
    }
    for (unsigned i = e->base; i <= e->arity; i++) {
        Value v = slotBase[i];
        if (IS_INT(v)) {
            e->localKind[i] = SLOT_INT;
        } else if (IS_FLOAT(v)) {
            e->localKind[i] = SLOT_FLOAT;
        } else if (IS_INSTANCE(v) && AS_INSTANCE(v)->klass != NULL) {
            e->localKind[i]  = SLOT_INST;
            e->localClass[i] = AS_INSTANCE(v)->klass;
            e->localShape[i] = AS_INSTANCE(v)->klass->shapeId;
        } else if (i == 0) {
            /* A plain function's slot 0 is the closure being called. Nothing
             * can be done with it, but the body has no reason to read it. */
            e->localKind[i] = SLOT_OPAQUE;
        } else {
            return false;
        }
    }
    return true;
}

static void jitFree(int *map, int *depths, int count) {
    JAI_FREE_ARRAY(int, map, count);
    JAI_FREE_ARRAY(int, depths, count);
}

static bool eligible(ObjFunction *fn) {
    const char *why = NULL;
    if (fn->arity == 0 || fn->arity > JIT_MAX_ARITY) why = "arity";
    else if (fn->maxSlots < 1 || (unsigned)fn->maxSlots - 1 > JIT_MAX_SAVED) why = "maxSlots";
    else if (fn->defaultCount != 0) why = "defaults";
    else if (fn->flags & (FN_VARIADIC | FN_KWREST)) why = "flags";

    else if (fn->module == NULL) why = "no module";
    else if (fn->chunk.count <= 0) why = "empty";
    if (why != NULL) {
        if (getenv("JAI_JIT_WHY")) {
            fprintf(stderr, "[jit] %s ineligible: %s (arity=%d maxSlots=%d)\n",
                    fn->name ? fn->name->chars : "<anon>", why, (int)fn->arity,
                    (int)fn->maxSlots);
        }
        return false;
    }
    return true;
}

bool jaiJitCompileFunc(ObjClosure *closure, Value *slotBase) {
    ObjFunction *fn = closure->fn;
    if (!eligible(fn)) return false;

    JaiCodeArena *arena = jaiJitArena();
    if (arena == NULL) return false;

    int *map = JAI_ALLOC(int, fn->chunk.count + 1);
    int *depths = JAI_ALLOC(int, fn->chunk.count + 1);
    for (int i = 0; i <= fn->chunk.count; i++) { map[i] = -1; depths[i] = -1; }


    Emit e;
    memset(&e, 0, sizeof e);
    e.arity        = fn->arity;
    e.offsetToInst  = map;
    e.offsetToDepth = depths;
    e.limitLiteral = -1;
    e.bailBlock    = -1;

    /* The prologue cannot be emitted first: its save set depends on how deep
     * the operand stack gets, which only the body knows. So the body goes into
     * the buffer at a fixed offset and the prologue is written in front of it
     * afterwards, with every instruction index shifted by the same amount. */
    Emit body;
    memset(&body, 0, sizeof body);
    body.arity        = fn->arity;
    /* The measuring pass runs with slot 0 available, purely to find out
     * whether the body reads it; the real pass then drops it if not. */
    body.base         = 0;
    body.locals       = (unsigned)fn->maxSlots;
    body.usesUpvalues = fn->upvalueCount > 0;
    body.offsetToInst = map;
    body.offsetToDepth = depths;
    if (!seedLocals(&body, slotBase)) {
        jitFree(map, depths, fn->chunk.count + 1);
        return false;
    }

    /* A first pass with a provisional frame, only to learn maxValue. The
     * emitted words are thrown away: frameBytes appears in the epilogue, so
     * they would be wrong. */
    body.savedCount = JIT_MAX_SAVED;
    body.frameBytes = 16 + 8 * JIT_MAX_SAVED + 8;   /* 16-aligned below */
    body.frameBytes = (body.frameBytes + 15u) & ~15u;
    if (!compileBody(&body, closure)) { jitFree(map, depths, fn->chunk.count + 1); return false; }

    /* A self-call cannot reproduce slot 0: no register holds the callee. A
     * body that both recurses and reads slot 0 is not compiled. */
    if (body.usesSlot0 && body.hasSelfCall) {
        jitFree(map, depths, fn->chunk.count + 1);
        return false;
    }
    e.base         = body.usesSlot0 ? 0u : 1u;
    e.locals       = (unsigned)fn->maxSlots - e.base;
    e.usesUpvalues = fn->upvalueCount > 0;

    unsigned argCount = fn->arity + 1u - e.base;
    unsigned closureArg = argCount;
    if (e.usesUpvalues) argCount++;
    if (argCount > JIT_MAX_ARITY) { jitFree(map, depths, fn->chunk.count + 1); return false; }

    unsigned saved = e.locals + (e.usesUpvalues ? 1u : 0u) + body.maxValue;
    if (saved > JIT_MAX_SAVED) { jitFree(map, depths, fn->chunk.count + 1); return false; }

    e.savedCount = saved;
    e.frameBytes = (16u + 8u * saved + 15u) & ~15u;

    /* Prologue. */
    emit(&e, jaiA64StpPre(29, 30, 31, -(int32_t)e.frameBytes));
    emitSaveRestore(&e, true);
    /* The real arguments land in the local registers in order; the closure,
     * when there is one, is the last incoming register but lives just past the
     * locals, where closureReg expects it. Placing it by argument index
     * instead put it three registers low and the first upvalue read
     * dereferenced whatever was there. */
    unsigned realArgs = e.usesUpvalues ? argCount - 1u : argCount;
    for (unsigned i = 0; i < realArgs; i++) {
        emit(&e, jaiA64MovX(JIT_FIRST_SAVED + i, i));
    }
    if (e.usesUpvalues) {
        emit(&e, jaiA64MovX(closureReg(&e), realArgs));
    }
    /* A local the interpreter would have left as NULL_VAL starts at zero here.
     * The checker guarantees definite assignment before any read, so this is
     * belt and braces -- but a register holding the last call's value would be
     * a bug that only shows up under recursion. */
    for (unsigned i = realArgs; i < e.locals; i++) {
        emit(&e, jaiA64MovzX(JIT_FIRST_SAVED + i, 0, 0));
    }
    /* Stack guard: bail rather than run off the end of the thread's stack,
     * so that runaway recursion still becomes a RecursionError. */
    int guardLoad = (int)e.count;
    emit(&e, jaiA64LdrLit(JIT_SCRATCH_A, 0));         /* patched below */
    emit(&e, jaiA64AddXImm(JIT_SCRATCH_B, 31, 0));    /* mov x10, sp */
    emit(&e, jaiA64SubsXReg(31, JIT_SCRATCH_B, JIT_SCRATCH_A));
    unsigned guardBranch = e.count;
    emit(&e, jaiA64BCond(JAI_A64_LO, 0));             /* patched below */

    unsigned prologue = e.count;

    for (int i = 0; i <= fn->chunk.count; i++) { map[i] = -1; depths[i] = -1; }
    e.offsetToInst  = map;
    e.offsetToDepth = depths;
    if (!seedLocals(&e, slotBase)) {
        jitFree(map, depths, fn->chunk.count + 1);
        return false;
    }
    if (!compileBody(&e, closure)) { jitFree(map, depths, fn->chunk.count + 1); return false; }
    if (e.failed) { jitFree(map, depths, fn->chunk.count + 1); return false; }
    (void)prologue;

    /* The bail block: say so, return anything, and let the caller throw the
     * whole computation away. */
    e.bailBlock = (int)e.count;
    emit(&e, jaiA64MovzX(0, 0, 0));
    emitEpilogue(&e, 1);

    /* Literal pool, 8-byte aligned so the 64-bit loads are aligned. */
    if ((e.count & 1u) != 0) emit(&e, jaiA64Nop());
    e.limitLiteral = (int)e.count;
    uintptr_t limit = stackLimit();
    if (limit == 0) { jitFree(map, depths, fn->chunk.count + 1); return false; }
    emit(&e, (uint32_t)(uint64_t)limit);
    emit(&e, (uint32_t)((uint64_t)limit >> 32));

    if (e.failed) { jitFree(map, depths, fn->chunk.count + 1); return false; }

    /* Patch the two literal loads and the guard branch. */
    e.code[guardLoad] = jaiA64LdrLit(JIT_SCRATCH_A, e.limitLiteral - guardLoad);
    e.code[guardBranch] =
        jaiA64BCond(JAI_A64_LO, e.bailBlock - (int)guardBranch);

    /* Patch every branch and the recursive calls. */
    for (unsigned i = 0; i < e.fixupCount; i++) {
        const Fixup *f = &e.fixups[i];
        int target;
        if (f->targetOffset == FIXUP_BAIL) {
            target = e.bailBlock;
        } else if (f->targetOffset == FIXUP_ENTRY) {
            target = 0;
        } else {
            if (f->targetOffset > (uint32_t)fn->chunk.count) {
                jitFree(map, depths, fn->chunk.count + 1);
                return false;
            }
            target = map[f->targetOffset];
            if (target < 0) {
                jitFree(map, depths, fn->chunk.count + 1);
                return false;
            }
            /* Registers are assigned from the operand-stack depth at each
             * point, so a join reached at two different depths would read a
             * value out of a register that holds something else. The walk
             * through the bytecode is linear and cannot see that, so it is
             * checked here and the function is declined rather than
             * mis-compiled. */
            if (f->depth >= 0 && depths[f->targetOffset] != f->depth) {
                jitFree(map, depths, fn->chunk.count + 1);
                return false;
            }
        }
        int rel = target - f->instIndex;
        uint32_t word = e.code[f->instIndex];
        if ((word & 0xfc000000u) == 0x94000000u) {
            e.code[f->instIndex] = jaiA64Bl(rel);
        } else if (f->conditional) {
            e.code[f->instIndex] = jaiA64BCond(word & 0xfu, rel);
        } else {
            e.code[f->instIndex] = jaiA64B(rel);
        }
    }
    jitFree(map, depths, fn->chunk.count + 1);

    /* A `bl` at instruction i must reach instruction 0 of this function, so
     * the recursive-call fixups above are relative to the function's own
     * start, which is where the arena is about to place it. */
    if (!jaiCodeArenaUnseal(arena)) return false;
    /* 8-align the entry so the literal pool's alignment is what it looks. */
    while ((arena->used & 7u) != 0) {
        uint32_t pad = jaiA64Nop();
        if (jaiCodeArenaWrite(arena, &pad, sizeof pad) == NULL) return false;
    }
    uint8_t *entry = jaiCodeArenaWrite(arena, e.code, e.count * sizeof e.code[0]);
    if (entry == NULL) return false;
    if (!jaiCodeArenaSeal(arena)) return false;

    if (e.assumedIntReturn && e.returnKind != SLOT_INT) {
        /* A self-call before the first return had to guess, and guessed
         * wrong. */
        return false;
    }
    for (unsigned i = 0; i < argCount; i++) {
        if (e.usesUpvalues && i == closureArg) {
            fn->jitParamKind[i]  = (uint8_t)SLOT_CLOSURE;
            fn->jitParamShape[i] = 0;
            continue;
        }
        fn->jitParamKind[i]  = (uint8_t)e.localKind[i + e.base];
        fn->jitParamShape[i] = e.localShape[i + e.base];
    }
    fn->jitReturnKind = (uint8_t)e.returnKind;
    fn->jitArgBase    = (uint8_t)e.base;
    fn->jitArgCount   = (uint8_t)argCount;

    if (getenv("JAI_JIT_WHY")) {
        fprintf(stderr, "[jit] compiled %s  arity=%u locals=%u insts=%u\n",
                fn->name ? fn->name->chars : "<anon>", e.arity, e.locals,
                e.count);
    }
    fn->jitFunc = entry;
    return true;
}

/* ------------------------------------------------------------------ */
/* Entry from the interpreter                                          */
/* ------------------------------------------------------------------ */

typedef JitResult (*Fn1)(int64_t);
typedef JitResult (*Fn2)(int64_t, int64_t);
typedef JitResult (*Fn3)(int64_t, int64_t, int64_t);
typedef JitResult (*Fn4)(int64_t, int64_t, int64_t, int64_t);

bool jaiJitEnterFunc(ObjClosure *closure, Value *slotBase) {
    ObjFunction *fn = closure->fn;
    if (fn->jitFunc == NULL) return false;

    /* Compiled code reads the global that names this function exactly once,
     * at compile time, and then calls it directly. Rebinding the name has to
     * invalidate that, and a module's version counter moves on every global
     * mutation, so one comparison covers it. It is conservative -- any global
     * write in the module retires the compiled form -- and conservative is the
     * safe direction. */
    if (fn->module == NULL || fn->module->version != fn->jitModuleVersion) {
        return false;
    }

    unsigned arity = fn->jitArgCount;
    int64_t a[JIT_MAX_ARITY];
    for (unsigned i = 0; i < arity; i++) {
        Value v = slotBase[fn->jitArgBase + i];
        switch ((SlotKind)fn->jitParamKind[i]) {
        case SLOT_INT:
            if (!IS_INT(v)) return false;
            a[i] = AS_INT(v);
            break;
        case SLOT_FLOAT: {
            if (!IS_FLOAT(v)) return false;
            double d = AS_FLOAT(v);
            memcpy(&a[i], &d, sizeof a[i]);
            break;
        }
        case SLOT_INST: {
            /* The class as well as the type: every field offset in the body
             * was resolved against this one shape. Compiled code holds the
             * raw pointer, which is safe only because the body cannot
             * allocate -- no collection can run while it does -- and the
             * argument slots keep the instance reachable meanwhile. */
            if (!IS_INSTANCE(v)) return false;
            ObjInstance *inst = AS_INSTANCE(v);
            if (inst->klass == NULL ||
                inst->klass->shapeId != fn->jitParamShape[i]) {
                return false;
            }
            a[i] = (int64_t)(uintptr_t)inst;
            break;
        }
        case SLOT_OPAQUE:
            a[i] = 0;   /* never read; see seedLocals */
            break;
        case SLOT_CLOSURE:
            /* Not a slot at all: the closure the interpreter is calling. Safe
             * to hold raw for the same reason every other pointer here is --
             * the body cannot allocate, and the caller holds this closure. */
            a[i] = (int64_t)(uintptr_t)closure;
            break;
        default:
            return false;
        }
    }

    JitResult r;
    switch (arity) {
    case 1: r = ((Fn1)(uintptr_t)fn->jitFunc)(a[0]); break;
    case 2: r = ((Fn2)(uintptr_t)fn->jitFunc)(a[0], a[1]); break;
    case 3: r = ((Fn3)(uintptr_t)fn->jitFunc)(a[0], a[1], a[2]); break;
    case 4: r = ((Fn4)(uintptr_t)fn->jitFunc)(a[0], a[1], a[2], a[3]); break;
    default: return false;
    }

    if (r.bailed) {
        /* Overflow or a stack that ran low. Nothing was written -- the body
         * cannot write -- so handing the call back to the interpreter is
         * enough, and it will raise the error with a traceback. Refusing the
         * function permanently keeps a bailing body from being re-entered on
         * every call only to bail again. */
        fn->jitRefused = true;
        fn->jitFunc = NULL;
        return false;
    }

    switch ((SlotKind)fn->jitReturnKind) {
    case SLOT_INT:
        slotBase[0] = INT_VAL(r.value);
        break;
    case SLOT_FLOAT: {
        double d;
        memcpy(&d, &r.value, sizeof d);
        slotBase[0] = FLOAT_VAL(d);
        break;
    }
    case SLOT_INST:
        slotBase[0] = OBJ_VAL((Obj *)(uintptr_t)r.value);
        break;
    default:
        return false;
    }
    vm.stackTop = slotBase + 1;
    return true;
}

#else

bool jaiJitCompileFunc(ObjClosure *closure, Value *slotBase) {
    (void)closure; (void)slotBase; return false;
}
bool jaiJitEnterFunc(ObjClosure *closure, Value *slotBase) {
    (void)closure; (void)slotBase; return false;
}

#endif
