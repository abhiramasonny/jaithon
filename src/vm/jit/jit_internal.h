/* jit_internal.h -- types, constants and cross-file declarations shared by the
 * whole-function JIT tier's translation units (jit_func.c and its siblings). */
#ifndef JAI_VM_JIT_INTERNAL_H
#define JAI_VM_JIT_INTERNAL_H

#include "vm/jit/jit.h"
#include "vm/vm.h"

#include <stdbool.h>
#include <stdint.h>

#if (defined(__aarch64__) || defined(__arm64__))

/* Two words in x0/x1 per AAPCS64 struct return (no store/load like the old global flag). A bailed
 * callee sends its caller straight to its own bail block, so recursion unwinds in one shot. */
typedef struct { int64_t value; int64_t bailed; } JitResult;

/* ------------------------------------------------------------------ */
/* Register plan                                                        */
/* ------------------------------------------------------------------ */

#define JIT_FIRST_SAVED 19u   /* x19..x28 are callee-saved and ours */
#define JIT_MAX_SAVED   10u
#define JIT_MAX_ROOTS   24u
/* Model entries, not register count -- an inlined body's operand-stack entries live in their own bank, wider than x19..x28. */
#define JIT_MAX_STACK   20u
/* Slots the compile-time model can describe, not the register budget: a declared frame can be wider
 * than what it touches; only the second (measuring) pass is held to JIT_MAX_SAVED. */
#define JIT_MAX_SLOTS   64u
#define JIT_MAX_ARITY    8u   /* arguments arrive in x0..x7 */
/* Deopt stubs dominate this size: each writes out every local and live stack entry. `merge` silently needed 512 -- hence the diagnostics. */
#define JIT_MAX_INSTS 20000u
/* Distinct pooled constants, and loads of them, one body may carry. Past
 * either a constant is simply materialised in registers instead. */
#define JIT_MAX_LITS     64u
#define JIT_MAX_LIT_USES 256u
#define JIT_MAX_FIXUPS 6000u
/* How many links of a refusal chain JAI_JIT_CHAIN will walk out. Each costs one
 * extra compile of the body, and a chain longer than this is not a backlog item
 * anybody is going to clear in one go. */
#define JIT_MAX_CHAIN 8u
#define JIT_SCRATCH_A    9u
#define JIT_SCRATCH_B   10u
#define JIT_SCRATCH_C   11u
#define JIT_SCRATCH_D   12u
/* x13..x17 are caller-saved and the tier names none of them, so a body that
 * calls nothing owns them outright. x18 is Darwin's platform register and is
 * NOT in the range. They are the only registers a loop-invariant hoist can
 * spend without taking one from the locals -- see planHoists. */
#define JIT_FREE_FIRST  13u
#define JIT_FREE_COUNT   5u
/* Two registers each; four list headers is every stencil seen so far. */
#define JIT_MAX_PIC_EXITS JAI_IC_WAYS
#define JIT_MAX_HOIST    4u
/* Register-free string facts (see Emit::strFact). Each costs a guard or two
 * at its loop head and nothing inside the loop. */
#define JIT_MAX_STR_FACTS 8u
/* How many offsets one walk may carry an FP-resident or deferred value into,
 * and how many float operators may write a local's home early. Past the
 * limit every instruction settles instead, so a large float body ran its
 * first sixty-four expressions in the FP bank and the rest through X. See
 * jitCarryLimit and JAITHON_JIT_WIDE_CARRY. */
#define JIT_MAX_CARRY      1024u
#define JIT_MAX_HOME_EARLY  256u
/* How many distinct clobber SITES the measuring pass will remember, so that
 * "does x0..x8 survive across this bytecode range" can be asked of a range
 * rather than of the whole body. Past this the body answers yes everywhere,
 * which is the same answer it gave before the range existed. */
#define JIT_MAX_CLOBBER 24u
/* Appends and storage stamps the measuring pass records per body; see
 * Emit::pushOff. Running out answers "unknown", which only costs a hoist. */
#define JIT_MAX_PUSHREC 24u
/* A hoisted header is proved distinct from at most this many append targets
 * at the hoist; a loop appending to more lists hoists nothing. */
#define JIT_MAX_HOIST_ALIAS 4u
/* Indirect call sites whose closure the measuring pass inlined, and the
 * closure hoists planned from them (see planClosureHoists). Each hoist keeps
 * at most JIT_MAX_CLOS_UP of the callee's upvalues in registers. */
#define JIT_MAX_CLOS_SITES 8u
#define JIT_MAX_CLOS_HOIST 2u
#define JIT_MAX_CLOS_UP    2u
/* Module-global accesses the measuring pass records, and the loops whose
 * keyVersion guard is proved once at the head (see planGuardHoists). */
#define JIT_MAX_GLOBAL_SITES 64u
#define JIT_MAX_GUARD_HOIST   4u
#define JIT_MAX_GLOBAL_ACC   64u
#define JIT_MAX_TAG_PROOF     8u
#define JIT_PUSH_UNKNOWN (-1)
#define JIT_PUSH_FRESH   (-2)
/* Distinct offsets the `match` arms may branch to across a discarded OP_POP
 * (see matchMissResume). One per alternative of one `match`, and `_is_operator`
 * in the lexer -- the largest in the tree -- has 40. Past this the arm refuses,
 * which costs coverage and never an unrecorded join. */
#define JIT_MAX_POPSKIP 128u
/* Repeated from the register plan below, which cannot be declared this early:
 * x0..x8, the bank a call-free body's operand stack uses. */
#define JIT_SCRATCH_BANK_COUNT 9u
#define JIT_MAX_ARGS_OUT 12

/* Values first in JitCallDesc so every field is 8-aligned and the emitted stores can use scaled forms. */
typedef struct JitCallDesc {
    struct JitCallDesc *link;
    int64_t nroots;
    Value   roots[JIT_MAX_ROOTS];
    Value   callee;
    Value   args[JIT_MAX_ARGS_OUT];
    Value   result;
    int64_t argc;
    int64_t aux;
} JitCallDesc;

typedef struct {
    int64_t ip;
    int64_t base;
    int64_t nlocals;
    int64_t nstack;
    int64_t skipLocals;
    Value   locals[JIT_MAX_SLOTS + 1];
    Value   stack[JIT_MAX_STACK + 1];
} JitDeoptRecord;

_Static_assert(JIT_MAX_SLOTS <= 64,
               "skipLocals is one bit per local slot");

/* A `self` entry (the callee of a recursive call) occupies no register; register numbers are
 * derived from the count of value entries below an entry, not from its depth. */
typedef enum {
    SLOT_INT,
    SLOT_FLOAT,
    SLOT_INST,
    /* Fixed class or null, held as the pointer or zero (`x == null` is then a compare against zero);
 * refusing this stopped six hundred stdlib bodies. Cost: materialising picks VAL_NULL/VAL_OBJ off the register at run time -- the tag isn't a static property of the kind. */
    SLOT_MAYBE_INST,
    SLOT_SELF,
    SLOT_OPAQUE,  /* present in a register, but nothing may be done with it */
    SLOT_CLOSURE,
    SLOT_CLASS,
    SLOT_FUNC,
    SLOT_NATIVE,
    SLOT_ITER,    /* ObjIter this body built, held raw; its index stays in memory (not a register) -- costs a load/store
                   * per iteration but means a deopt needs no write-back, since the stack's iterator is always current. */
    SLOT_BOOL,    /* 0 or 1 in a register -- a Value's boolean member is its low byte, so the same word serves both. */
    SLOT_NULL,    /* What `-> void` returns: a defined zero in a register (droppable, or written out by a deopt) whose
                   * tag is VAL_NULL rather than the VAL_OBJ every other kind chain in this file falls through to. */
    SLOT_OBJ,     /* Heap object of a type this tier doesn't model, held raw: may only be read, passed, stored and rooted. */
    SLOT_LIST,    /* ObjList *, raw -- safe for the same reason an instance is: nothing moves, and a call spills it as a root first. */
    SLOT_MAYBE_OBJ,
    SLOT_DYNAMIC
} SlotKind;

#define JIT_RET_TAG_SHIFT 8u

_Static_assert(SLOT_MAYBE_OBJ <= 15,
               "a SlotKind must fit the four bits stackSignatureAt packs it into");
_Static_assert(SLOT_DYNAMIC == SLOT_MAYBE_OBJ + 1 && SLOT_DYNAMIC == 16,
               "SLOT_DYNAMIC is a return kind only: it is never packed into a "
               "stack signature, so it alone may sit past the four-bit limit");


/* Floats live in X registers, visiting d0/d1 only for the arithmetic itself -- simpler-but-correct beats
 * a second register bank (own allocator/save-set/spill rules) for a tier this young. Nothing calls between the fmovs, so the scratch pair is safe. */
#define JIT_FSCRATCH_A 0u
#define JIT_FSCRATCH_B 1u

/* A SLOT_FLOAT entry may live in an FP register between the instruction that computes it and the one
 * that consumes it: entry i's canonical home is x(19+regBase+i), but `fpLive` bit i means that copy is stale and v(16+i) holds the value instead. Bank v16..v25 is caller-saved on purpose (nothing here survives a call); the index is shared with the X bank so two live entries can never collide. */
#define JIT_FP_BANK 16u

/* Same argument for a float LOCAL: v8..v15 is ABI-preserved across a call, nothing else here names it,
 * and only fmov/ldr d/str d ever touch a local's home -- never arithmetic -- so a slot parked here is a bit-exact 64-bit home (a kind the allocator guessed wrong costs an fmov, never wrong bits). */
#define JIT_FP_FIRST_SAVED 8u
#define JIT_FP_MAX_SAVED   8u

#define JIT_MAX_DEOPT     160u
#define JIT_MAX_EXIT      8u
/* Self-calls and direct calls to a callee that writes share this table, so it
 * is sized for a body with several of each rather than for recursion alone. */
#define JIT_MAX_SELF_SLOW 32u
#define JIT_MAX_GROW      16u

#define FIXUP_BAIL   UINT32_MAX
#define FIXUP_ENTRY  (UINT32_MAX - 1u)
#define FIXUP_THREW  (UINT32_MAX - 2u)
#define FIXUP_OVF    (UINT32_MAX - 3u)   /* minus 0,1,2 for the three operators */
#define FIXUP_DEOPT    (UINT32_MAX - 7u)                     /* minus a deopt index */
#define FIXUP_EXIT     (FIXUP_DEOPT - JIT_MAX_DEOPT)         /* minus an exit index */
#define FIXUP_SELFSLOW (FIXUP_EXIT - JIT_MAX_EXIT)           /* minus a self-call index */
#define FIXUP_GROW     (FIXUP_SELFSLOW - JIT_MAX_SELF_SLOW)  /* minus a growth index */
#define JIT_MAX_COLD      48u
#define FIXUP_COLD     (FIXUP_GROW - JIT_MAX_GROW)           /* minus a cold index */

/* Every sentinel range must stay above any offset a real chunk can have. A
 * chunk that large is not representable long before this matters, so half the
 * u32 range is an enormous margin -- the point is that the build fails if the
 * tables ever grow enough to reach down into bytecode-offset territory. */
_Static_assert(FIXUP_COLD - JIT_MAX_COLD > UINT32_MAX / 2u,
               "jit fixup sentinels have grown down into bytecode offsets");
_Static_assert(FIXUP_COLD < FIXUP_GROW - (JIT_MAX_GROW - 1u),
               "jit growth and cold fixup ranges overlap");
_Static_assert(FIXUP_EXIT < FIXUP_DEOPT - (JIT_MAX_DEOPT - 1u),
               "jit deopt and exit fixup ranges overlap");
_Static_assert(FIXUP_SELFSLOW < FIXUP_EXIT - (JIT_MAX_EXIT - 1u),
               "jit exit and self-call fixup ranges overlap");
_Static_assert(FIXUP_GROW < FIXUP_SELFSLOW - (JIT_MAX_SELF_SLOW - 1u),
               "jit self-call and growth fixup ranges overlap");

typedef struct {
    int      instIndex;
    uint32_t targetOffset; /* bytecode offset, or FIXUP_BAIL / FIXUP_ENTRY */
    bool     conditional;
    int64_t  depth;
} Fixup;

typedef struct {
    uint32_t  code[JIT_MAX_INSTS];
    unsigned  count;

    SlotKind  stack[JIT_MAX_STACK];
    uint32_t  stackShape[JIT_MAX_STACK];
    ObjClass *stackClass[JIT_MAX_STACK];
    Value     stackSeen[JIT_MAX_STACK];
    /* The object type this SLOT_OBJ entry is expected to have, as ObjType + 1,
     * or 0 for "no expectation". For results the tier produces itself and so
     * has no Value to sample -- an f-string's, a builtin method's predicted
     * from its site's feedback, a `str()` call's -- where the type is known
     * even though the object does not exist until run time.
     *
     * A PREDICTION, not a proof: every consumer already re-checks Obj.type and
     * deoptimises on a miss, so this only chooses which guard to emit. That is
     * why it survives a branch merge and is not listed in clearStackProofs --
     * see the contrast drawn at stackAscii below. */
    /* The kind that could not be adopted into a local already typed otherwise;
     * read only by kindClash, to say WHICH two kinds disagreed. */
    SlotKind  clashKind;
    /* A slot asked to be widened or made dynamic, with the walk carrying on
     * anyway so the REST of them are found in the same pass. The attempt is
     * already lost when this is set; it exists so ONE retry fixes every
     * clashing slot instead of one per attempt.
     *
     * The budget is five attempts and each clash used to cost one, so a body
     * with four clashing locals spent four of them before it could even try:
     * `lead_with_extremes` in jaicv has exactly that, and `thick_contours`
     * carries 13% of the jaicv benchmark's interpreted work behind the same
     * thing.
     *
     * MEASURING PASS ONLY. Carrying on after a clash leaves the model and the
     * emitted code disagreeing, which is fine when the output is discarded and
     * segfaults when it is not -- that was measured, not guessed. */
    bool      pendingRetry;
    /* See emitUnarmedDeopt: the opcode and offset the walk stopped at, so the
     * fixup pass can name the cause and not just the symptom. */
    uint8_t   unarmedOp;
    /* The FIRST unarmed stop of the walk (unarmedAt is the last), and
     * whether there was one. See earlyUnarmedDecline. */
    uint32_t  firstUnarmedAt;
    bool      haveFirstUnarmed;
    uint32_t  unarmedAt;
    /* Offsets this walk branched to across an OP_POP it never emitted; see
     * matchMissResume. Nothing in the BYTECODE branches there, so every test in
     * the walk that asks "can something else reach this offset" by scanning the
     * chunk has to be told about them separately -- the ones that scan the
     * fixup list instead already see them. */
    uint32_t  popSkip[JIT_MAX_POPSKIP];
    unsigned  popSkipCount;
    uint8_t   stackObjType[JIT_MAX_STACK];
    /* An exemplar of what THIS list entry's elements are, for a list the body
     * built itself and so has no live sample of. `OP_BUILD_LIST` knows the kind
     * of everything it just popped; nothing downstream does, because the list
     * does not exist until run time.
     *
     * A prediction of the same standing as stackSeen: whatever reads it emits
     * the guard it would have emitted anyway. For an int, float or bool the
     * exemplar is a synthesised immediate and so has no lifetime at all; for an
     * object it is the element's own live sample, which was already being held
     * here and is reachable for the same reasons it was. */
    Value     stackElem[JIT_MAX_STACK];
    uint8_t   stackElemDecl[JIT_MAX_STACK];
    int       stackLocal[JIT_MAX_STACK];
    /* localEpoch when the entry was pushed from its local. stackLocal says
     * which local an entry was READ from; this says whether that local still
     * holds it, which is what a memo keyed on the local needs (see
     * stackLocalCurrent). */
    uint32_t  stackLocalEpoch[JIT_MAX_STACK];
    bool      stackAscii[JIT_MAX_STACK];
    bool      stackUnit[JIT_MAX_STACK];
    /* This entry is a compile-time constant resolved out of the module, not a
     * value loaded at run time -- so its type is a FACT and the arms that
     * consume it owe no guard. Only OP_GET_GLOBAL's by-value arms set it, and
     * ObjModule::version retires the whole form if the binding could change
     * (jaiValueIsInertGlobal decides which kinds qualify). Rides alongside
     * SLOT_OBJ rather than taking a SlotKind of its own: the kind is 4 bits
     * and full. */
    bool      stackPinned[JIT_MAX_STACK];
    /* On a list iterator (SLOT_ITER shape 1) only, written by OP_GET_ITER's
     * list arm and read by OP_FOR_ITER_BIND's: the live list holds instances
     * of more than one class, so the loop variable is bound unpinned and its
     * calls dispatch through the site's cache. Meaningless on any other
     * entry, and nothing else reads it. */
    bool      stackMixed[JIT_MAX_STACK];
    bool      stackNullLit[JIT_MAX_STACK];
    struct { int local; uint16_t field; SlotKind kind; } known[16];
    unsigned  knownCount;
    SlotKind  localKind[JIT_MAX_SLOTS + 1];
    uint32_t  localShape[JIT_MAX_SLOTS + 1];
    ObjClass *localClass[JIT_MAX_SLOTS + 1];
    bool      localTyped[JIT_MAX_SLOTS + 1];
    /* Kept only so a field read on this local has something to read the field's type off; a local bound
     * from a list element has no argument to look at (why nbody's `advance`'s `bi` couldn't have its fields read). */
    Value     localSeen[JIT_MAX_SLOTS + 1];
    /* stackElemDecl carried across the bind that named this local. */
    uint8_t   localElemDecl[JIT_MAX_SLOTS + 1];
    /* And stackObjType, for the same reason and by the same route. Without it
     * a callee's observed return type survives exactly as long as the value
     * stays on the operand stack: `let w = window_of(...)` then `w.len()`
     * loses it at the bind, which is every use that matters. */
    uint8_t   localObjType[JIT_MAX_SLOTS + 1];
    Value    *observed;
    bool      assumedIntReturn;
    unsigned  depth;
    unsigned  valueDepth;
    unsigned  maxValue;
    unsigned  maxValueAll;

    Fixup     fixups[JIT_MAX_FIXUPS];
    unsigned  fixupCount;

    int      *offsetToInst;
    int64_t  *offsetToDepth;
    /* What the BYTECODE says the operand stack is at each offset, from
     * jaiChunkStackDepths -- an oracle this file did not write, checked against
     * every deopt record. NULL only when the chunk would not verify, which is
     * already impossible by the time anything is compiled. See modelAgreesWithChunk. */
    const int *chunkDepth;
    int        chunkDepthCount;

    unsigned  arity;
    /* Slot 0 is the callee for a plain call, the RECEIVER for a method (so `self.x` reads it). Touching
     * it turns it into an ordinary local plus an extra incoming argument; untouched (every plain function), it costs nothing. */
    unsigned  base;
    /* A list comprehension's accumulator. `OP_BUILD_LIST` pushes it BEFORE the
     * loop head, so it sits below the window the OSR model tracks and
     * `OP_LIST_APPEND` used to refuse outright -- which left every comprehension
     * in the language running interpreted. It lives at a fixed FRAME SLOT for
     * the whole loop, so the append arm reloads it from there through
     * JIT_SLOTS_REG rather than trusting a hoist, and guards that what it found
     * is really a list. `accWanted` costs one reserved callee-saved register
     * (see osrReserved), because emitListStore needs the container in a
     * register that survives its four loads and the grow stub's call. */
    int       accSlot;
    bool      accWanted;
    bool      usesSlot0;
    /* Highest slot the body actually names -- not maxSlots, the (routinely larger) frame window the
     * interpreter reserves. Every slot here costs one of the ten callee-saved registers. */
    unsigned  maxSlotUsed;
    /* The first pass exists to find maxValue and maxSlotUsed, so it must not
     * stop at a budget computed from a slot count it is still discovering. */
    bool      measuring;
    /* Locals spill to the frame instead of registers when the body has more live values than callee-saved
     * registers (nbody's `advance` wants nineteen); the operand stack always stays in registers. Flags that the PER-SLOT plan (slotXReg/slotFpReg) is in force: busiest slots keep a register, the rest live in the frame, as the OSR tier does. */
    bool      spilled;
    unsigned  localsFrameOffset;
    /* OSR: locals ARE the interpreter's frame slots via a pointer handed to the entry -- nothing is copied
     * either way, which is also what makes a deopt cheap here: only the operand stack needs rebuilding. */
    bool      osr;
    uint32_t  osrTop;
    uint32_t  osrEnd;
    /* `for i in a..b` compiled as a counted loop: the iterator object stays on the interpreter's stack
     * untouched, only its index rides in a register, and every exit writes it back. */
    bool      hasIter;
    /* 1 a unit-step range, 2 a list, 3 a dict-items view, 4 a list of 2-tuples
     * at a pair head, 5 an enumerate snapshot (ITER_LIST_ENUM) at a pair head.
     * 3 and 4 share a prologue and reserve the same single register, because
     * both read the index out of the ObjIter every iteration and write it
     * straight back -- they differ only in where emitForIterPair finds the
     * pair, and so in what elemSample holds: the DICT itself for 3, the
     * sampled TUPLE ELEMENT for 4. 5 shares neither of those: it is a LIST
     * head that happens to bind two slots, so it takes 2's prologue, 2's
     * reserved registers and 2's index write-back, and elemSample holds the
     * sampled list ELEMENT. Nothing to do with the function tier's separate
     * SLOT_ITER shape numbering, where 4 is the dict view and 5 is this
     * snapshot only by coincidence of the next free digit. */
    uint8_t   iterKind;
    Value     elemSample;
    /* The sampled element's class is one of several the list holds, so the
     * loop variable is an instance of no particular class. Set by the driver,
     * which is the only place the whole list is in hand. */
    bool      elemMixed;
    /* The ListStore each list local is backed by, pinned at compile time and
     * re-checked by the entry guard -- the element loads are emitted at one
     * width, so this is as much a commitment as an instance slot's class. The
     * function tier samples nothing and leaves every entry BOXED, which is the
     * storage a list built by compiled code always has. */
    /* Storage of the list an iterKind 2 head walks, from the same sample. */
    uint8_t   elemStg;
    /* Which of those pins the emission is allowed to BELIEVE. Decided once,
     * between the measuring pass and the real one, because the measuring pass
     * is what says where a slot is written and where the body calls out --
     * planHoists settles its own question in the same place for the same
     * reason. False everywhere in the probe, which therefore emits the boxed
     * form; the two passes already differ over hoisting. */
    bool      localStgPin[JIT_MAX_SLOTS + 1];
    bool      elemStgPin;
    /* Per-slot register assignment. Historically one flag gated the whole loop on a budget check that
     * almost nothing passed, so most loops ran fully in memory. Three things fixed it: only NAMED slots take a register; a float slot takes v8..v15 instead of an X register; overflow slots stay in the frame (busiest slots win, weighted by loop nesting) rather than all-or-nothing. */
    uint8_t   slotXReg[JIT_MAX_SLOTS + 1];   /* x19..x28, or 0 for none */
    uint8_t   slotFpReg[JIT_MAX_SLOTS + 1];  /* d8..d15, or 0 for none */
    unsigned  xLocals;
    unsigned  fpLocals;
    uint32_t  slotUse[JIT_MAX_SLOTS + 1];
    /* Same idea for the function tier, but counted in INSTRUCTIONS SAVED, not sites (see noteSlotCost):
     * a float read through the FP bank costs one `ldr d` plus one `fmov d,x` from an X register, so a flat per-use count would wrongly credit an X register for such a slot. Two banks, two ledgers. */
    uint32_t  slotSaveX[JIT_MAX_SLOTS + 1];
    uint32_t  slotSaveFp[JIT_MAX_SLOTS + 1];
    const uint8_t *loopDepth;
    unsigned  loopDepthCount;
    unsigned  fpSaveOffset;
    /* Where a range loop parks the ObjIter, since it holds no register for it -- written once in the prologue, read once per exit stub. */
    unsigned  iterFrameOffset;
    /* Inlining widens the live range of everything the callee reads, so a loop that fit the registers as
     * a call may not fit as an expression; the compile retries with this set when that's what went wrong. */
    bool      noInline;
    bool      inlined;
    /* A map kernel (jaiJitCompileMapKernel): the body wrapped in a loop over
     * gJitMapRun, each return storing its value and taking the next element
     * instead of leaving. `mapKernelHead` is the loop's first instruction and
     * `mapKernelExit` the `b.ge` out of it, patched once the exit exists. */
    bool      mapKernel;
    uint8_t   mapKernelKind;
    int       mapKernelHead;
    int       mapKernelExit;
    /* x(reg) the index, x(reg+1) the bound, x(reg+2) &gJitMapRun, and for
     * a filter kernel (mapKernelKind SLOT_BOOL) x(reg+3) the count kept. */
    unsigned  mapKernelReg;
    /* Inlined callee: its locals are operand-stack entries of the CALLER's frame (slots 1..n are the
     * already-present argument entries); nothing is copied, no frame appears -- but the interpreter has no idea, so every guard inside deoptimises to `inlIp` (the caller's OP_CALL) with the model as of `inlDepth`. */
    bool      inlining;
    unsigned  inlDepth;
    unsigned  inlPinned;
    unsigned  inlValueBase;
    uint32_t  inlIp;
    /* Register holding the ObjClosure being inlined (-1 if none) -- the CALLEE's closure, not the caller's;
     * only meaningful while `inlining`, the only thing that writes it, so a zeroed Emit never reads stale state. */
    int       inlClosureReg;
    int       inlSlot[JIT_MAX_SLOTS + 1];
    /* A local whose kind differs across paths into some point: lives in the frame with its tag, and every
     * read guards. Not exotic -- the compiler reuses one slot for non-overlapping loop induction variables (nbody's advance). */
    bool      dynamicLocal[JIT_MAX_SLOTS + 1];
    bool      needDynamic[JIT_MAX_SLOTS + 1];
    /* A slot holding an instance on one path, null on another: unlike a dynamic local it stays in a register
     * (pointer or zero), tag built at materialisation. Per-slot, since widening every nullable-mentioning parameter cost object_dispatch 2x. */
    bool      nullableLocal[JIT_MAX_SLOTS + 1];
    bool      needNullable[JIT_MAX_SLOTS + 1];
    bool      pendingRange;
    bool      rangeInclusive;
    /* Whether the pending range's low end was an integer literal, and which
     * one. A range this body builds always steps by 1 (jitMakeRangeIter takes
     * no step), so with the start known too the value a nested FOR_ITER_BIND
     * yields is `index + K` -- no ObjRange to reach through at all. */
    bool      rangeStartKnown;
    int64_t   rangeStartVal;
    /* Where the deferred OP_BUILD_RANGE was, so a guard inside the header it
     * folded into can resume at an offset the model still describes. */
    uint32_t  rangeBuildIp;
    unsigned  iterSlot;
    uint32_t  iterExit;
    int       exitStub[JIT_MAX_EXIT];
    uint32_t  exitOffset[8];
    unsigned  exitCount;
    /* A body that reads an upvalue needs the closure itself -- not any slot (a method's slot 0 is the
     * receiver, not the callee) -- so it arrives as one extra argument, in the register just past the locals. */
    bool      usesUpvalues;
    /* Set once the body has written to the heap. A deopt after that is still
     * fine -- it resumes AT an instruction, so the writes before it are not
     * re-run -- which is what makes the retired bail path unnecessary. */
    bool      wroteHeap;
    /* The instruction being compiled is inside a protected region of the
     * function's static exception table (spec §3.8) -- i.e. inside a `try`.
     *
     * A raise from compiled code unwinds from a frame the interpreter never
     * pushed (function tier) or from the loop head rather than the faulting
     * instruction (OSR tier), so vmThrow consults the wrong offset and the
     * function's OWN handler is skipped. Nothing used to check this because no
     * function containing a `try` could compile at all -- every catch block
     * holds OP_GET_EXC, which had no arm and declined the whole function. The
     * unarmed-opcode deopt below removes that accident, so the rule is now
     * explicit: inside a protected region an overflow resumes at its
     * instruction (branchOnDeoptInstStart) and lets the interpreter raise it,
     * and anything else that can raise -- a call, a list growth -- declines. */
    bool      inProtected;
    /* Every operand-stack entry was in its own register at the top of the
     * instruction being compiled -- nothing deferred, nothing borrowed. See
     * branchOnDeoptInstStart, whose record describes entries the arm may
     * already have popped. */
    bool      instClean;
    /* Baked globals share the defining module's table, so one `keyVersion` guard covers all of them --
     * it changes only when a live entry's address or key could move (new key, rehash, delete, clear). ObjModule::version (used by the function-tier entry check) is neither necessary nor sufficient here. */
    JaiTable *globalsTable;
    uint32_t  globalsKeyVersion;
    /* Same plan, one class's `statics` table (OP_GET_FIELD's SLOT_CLASS arm):
     * a body that reads static fields off two different classes declines the
     * second rather than pretend one table's keyVersion stands for both. */
    JaiTable *staticsTable;
    uint32_t  staticsKeyVersion;
    /* And one imported module's globals, for `module.MEMBER`. Same plan again:
     * one table per body, and a second module declines rather than pretend one
     * keyVersion stands for both.
     *
     * `math.PI` was the single largest refusal left on the jaicv benchmark
     * once the loop tier stopped throwing bodies away -- 4.6% of interpreted
     * work in `min_area_rect` alone, which reads it once. Before this the
     * receiver was a module, OP_GET_FIELD wanted an instance, and the whole
     * function declined at a constant. */
    JaiTable *modTable;
    uint32_t  modKeyVersion;
    bool      callsOut;
    /* Set the moment anything is emitted that can destroy x0..x8 while the
     * body is still running: a call out, or one of the two stubs that call and
     * then branch BACK into the body (list grow, self-call slow path). It is
     * what `scratchValues` below is decided from, and the measuring pass is
     * what observes it. */
    bool      clobbersScratch;
    /* WHERE each of those sites was, in bytecode offsets, so the same question
     * can be asked of one loop instead of the whole body: a nest whose inner
     * loop calls nothing still owns x0..x8 and x13..x17 *inside* that loop,
     * however many times the outer one calls. Recorded by the measuring pass
     * and copied into the real one. Overflowing the array sets `clobberSpill`,
     * which answers "yes, everywhere" -- the whole-body answer, so running out
     * of room costs a hoist and never a wrong one. */
    uint32_t  clobberOff[JIT_MAX_CLOBBER];
    /* The site is an instance allocation of a class with a simple init
     * (emitCallOut's whole path): it may collect, but runs no Jaithon code
     * and so cannot add, remove or rehash a module global. */
    bool      clobberAlloc[JIT_MAX_CLOBBER];
    bool      allocOnlyCall;
    unsigned  clobberCount;
    bool      clobberSpill;
    /* Where the measuring pass emitted a module-globals keyVersion guard. */
    uint32_t  globalOff[JIT_MAX_GLOBAL_SITES];
    unsigned  globalSiteCount;
    bool      globalSiteSpill;
    /* JAITHON_JIT_GLOBAL_GUARD_HOIST: loops whose every globals guard is
     * proved once at the head instead. See planGuardHoists. */
    struct { uint32_t top, end; } guardHoist[JIT_MAX_GUARD_HOIST];
    unsigned  guardHoistCount;
    /* Every scalar module-global read and write the measuring pass emitted:
     * where, which entry, of what kind. Overflow only costs proofs. */
    struct {
        uint32_t  off;
        JaiEntry *slot;
        uint8_t   kind;
        bool      write;
    } globalAcc[JIT_MAX_GLOBAL_ACC];
    unsigned  globalAccCount;
    bool      globalAccSpill;
    /* JAITHON_JIT_GLOBAL_TAG_PROOF: a global whose every access inside a
     * guard-hoisted loop is of one scalar kind has that tag checked once at
     * the head; reads skip their tag check, writes their old-tag check and
     * tag store. See planTagProofs. */
    struct {
        uint32_t  top, end;
        JaiEntry *slot;
        uint8_t   kind;
        /* JAITHON_JIT_GLOBAL_PROMOTE: over a loop that calls nothing, the
         * global's value also lives in this hoist register (0: it does not).
         * Loaded at the head, read as a borrow, and written THROUGH: every
         * store still goes to the entry, so a deopt or an exit finds memory
         * current and nothing is ever written back. */
        uint8_t   reg;
        /* The loop allocates (an allocation-only call, Emit::clobberAlloc):
         * `reg` is one of x13..x17, which the inline allocation never names,
         * and each allocation's slow path reloads it from the entry after
         * its call (emitPromotedReload). */
        bool      allocs;
    } tagProof[JIT_MAX_TAG_PROOF];
    unsigned  tagProofCount;
    /* Every list append the body makes, when the grow stub keeps the
     * registers (jitGrowKeeps): such an append is no longer a clobber, so
     * regionCalls stops seeing it -- but it still moves ONE list's `items`
     * and `count`, and a header hoisted over it must not be that list. So the
     * measuring pass records, per append, the local the target was read out
     * of: a slot, JIT_PUSH_UNKNOWN for a target with no local behind it (a
     * field, a subscript, a call's result), or JIT_PUSH_FRESH for a
     * comprehension's accumulator, which no local can name while its loop
     * runs. planHoists asks this before hoisting anything over an append.
     * Overflow sets `pushSpill`, which reads as an unknown target everywhere. */
    uint32_t  pushOff[JIT_MAX_PUSHREC];
    int16_t   pushSlot[JIT_MAX_PUSHREC];
    unsigned  pushCount;
    bool      pushSpill;
    /* Every OP_ELEM_KIND that may stamp an unboxed storage onto an empty list.
     * It is the one thing compiled code does that changes a list's `stg`
     * without calling out, and with appends no longer counting as calls a
     * loop could otherwise pin a slot's storage over one -- `var b: list[int]
     * = a` re-stamping the empty list `a` pinned BOXED, and the append that
     * follows storing at the wrong width. A region holding one pins nothing. */
    uint32_t  stampOff[JIT_MAX_PUSHREC];
    unsigned  stampCount;
    bool      stampSpill;
    /* The deepest the operand stack ever was at one of those sites. Every entry
     * live when a helper runs is below it, so every entry AT or ABOVE it is
     * provably never live across a call -- which is the whole condition for
     * putting one in a caller-saved register. Unlike a region, this is an
     * index, so where an entry lives stays a function of its index alone and
     * every join, deopt record and stub keeps agreeing about it for free. */
    unsigned  clobberDepth;
    /* The operand stack lives in x0..x8 rather than above the locals in the
     * callee-saved bank. Sound exactly when nothing can clobber a caller-saved
     * register between a push and its use, i.e. `!clobbersScratch` and the
     * whole stack -- an inlined body's entries included, which is what
     * maxValueAll counts -- fits the nine. Worth having because
     * the two banks were competing for the same ten registers: a stencil whose
     * expression is seven deep left NOTHING for its four row pointers and its
     * index, and reloaded all five from the frame every iteration. */
    bool      scratchValues;
    /* The same idea one granularity down, for a body that DOES call. Entries
     * below `splitAt` stay in the callee-saved bank; entries at or above it
     * live in x0..x8. `splitAt` is the deepest the operand stack ever is at a
     * clobber site (Emit::clobberDepth), so an entry at or above it is one no
     * helper can ever be running underneath -- which is the whole soundness
     * condition, stated about an entry's live range rather than about the
     * body. Zero means no split.
     *
     * A split INDEX rather than a region is what keeps this cheap: where an
     * entry lives stays a function of its index alone, so joins, deopt records
     * and stubs all keep agreeing about it without being told anything. The
     * price is that the operand stack is no longer one run of registers, and
     * every site that used to add an index to a base has to say `valueBankReg`
     * instead. */
    unsigned  splitAt;
    /* probe.clobbersScratch, carried into the real pass. `no call anywhere in
     * this body` is a stronger statement than `the values may live in x0..x8`
     * -- scratchValues also wants the stack to fit nine -- and the hoist below
     * needs the stronger one. */
    bool      bodyCalls;
    /* Where the measuring pass saw each slot written, and where it saw each one
     * used as the base of a subscript. Bounds rather than a set: a loop is a
     * contiguous range of offsets, so "written inside this loop" is a range
     * overlap, and widening a range can only lose a hoist, never make a wrong
     * one. lo > hi means "never". */
    uint32_t  slotWriteLo[JIT_MAX_SLOTS + 1];
    uint32_t  slotWriteHi[JIT_MAX_SLOTS + 1];
    uint32_t  slotIndexLo[JIT_MAX_SLOTS + 1];
    uint32_t  slotIndexHi[JIT_MAX_SLOTS + 1];
    /* The same bounds for the subscripts that STORE into the slot's list --
     * the ones a hoist could give a version register. */
    uint32_t  slotStoreLo[JIT_MAX_SLOTS + 1];
    uint32_t  slotStoreHi[JIT_MAX_SLOTS + 1];
    unsigned  slotIndexUse[JIT_MAX_SLOTS + 1];
    /* Where the measuring pass saw each slot read as one side of a string
     * identity compare -- the arm that guards "a string, and interned" on
     * each side. Same bounds-not-a-set form as slotIndexLo/Hi. */
    uint32_t  slotEqLo[JIT_MAX_SLOTS + 1];
    uint32_t  slotEqHi[JIT_MAX_SLOTS + 1];
    unsigned  slotEqUse[JIT_MAX_SLOTS + 1];
    /* Loop-invariant list headers. See planHoists. */
    struct {
        uint32_t top, end;
        uint8_t  slot, itemsReg, countReg;
        /* The counted-range head of the loop this hoist covers, when it has
         * one. Per hoist rather than per form because the loop that matters is
         * often INNER: a stencil's outer `for i` assigns the row locals, so
         * nothing hoists there, while the inner `for j` hoists all four and is
         * where every subscript actually happens. */
        bool     rangeOk;
        uint16_t rVar, rCur, rEnd;
        /* A STRING's header rather than a list's: itemsReg holds `chars` and
         * countReg `length`, and the head has already proved the local is a
         * string whose every scalar is one byte. See planHoists. */
        bool     str;
        /* Every subscript of the slot lies inside this loop, so the slot's
         * span is this loop's and the head may prove its bounds. */
        bool     inside;
        /* The locals the loop appends to. Not this slot (planHoists refused
         * that), but possibly the same LIST under another name, so the hoist
         * proves the pointers differ before the loop runs. */
        uint8_t  aliasCount;
        uint8_t  aliasSlot[JIT_MAX_HOIST_ALIAS];
        /* The storage this hoist proves at the head, so every access inside
         * the loop is emitted at it with no test (see jitHoistPinOn). */
        bool     stgPin;
        uint8_t  stg;
        /* Whether countReg holds the count. A lean hoist (JAITHON_JIT_HOIST_LEAN)
         * keeps only `items`: every subscript the head's guard covers needs no
         * count, and one it does not cover loads its own header. */
        bool     hasCount;
        /* The version the list's stores in [top, end) write, already bumped:
         * loaded once where the header is, so a store is one `str` instead of
         * a load-add-store whose load waits on the previous iteration's store.
         * Sound because the region is call-free -- no iterator can be made in
         * it, so every snapshot was taken before the load, and any store moves
         * the version past all of them, which is all an iterator asks. */
        bool     hasVer;
        uint8_t  verReg;
    } hoist[JIT_MAX_HOIST];
    unsigned  hoistCount;
    /* Loop-invariant FACTS about a string local, proved once at a loop head
     * and spending no register: that it is a string whose scalars are all one
     * byte (`ascii`, for `s[i]`), or an interned string (`interned`, for the
     * identity compare). Unlike a header these survive a call -- a string
     * never changes, the collector never moves it, interning is never undone,
     * and the local is one the loop does not write and nothing else can. See
     * planStrFacts. */
    struct {
        uint32_t top, end;
        uint8_t  slot;
        bool     ascii, interned;
    } strFact[JIT_MAX_STR_FACTS];
    unsigned  strFactCount;
    /* Set by an iterate arm that found no element to sample and emitted
     * nothing, asking the walk to interpret from that instruction rather
     * than decline the body. Function tier only; see jitIterSoft. */
    bool      iterUnarmed;
    /* The push whose list's count and version live in registers for a loop
     * (JAITHON_JIT_PUSH_REG): loaded where the headers are, written back to
     * the list after every push -- the stores stay, the loads that chained
     * each push to the previous one through memory go. Planned only over a
     * loop whose one push is a keeping grow site (jitGrowKeeps), so the stub
     * puts the two registers back. */
    struct {
        uint32_t top, end;
        uint8_t  slot, countReg, verReg;
    } pushHoist[2];
    unsigned  pushHoistCount;
    /* A for-in over a list whose iterator this body built keeps the
     * iterator's index in a register for the loop (JAITHON_JIT_ITER_IDX_REG):
     * loaded where the headers are, stored back after every step, never
     * loaded again -- the store keeps the iterator current for a deopt, and
     * the load it replaces was the step's dependency on the step before.
     * `live` says the load was emitted: only over a list iterator. */
    struct {
        uint32_t top, end;
        uint8_t  reg;
        bool     live;
    } iterHoist[2];
    unsigned  iterHoistCount;
    /* Indirect call sites where the measuring pass inlined a closure read
     * straight out of a local: the call's offset, the local, and the sample
     * closure the inline was made against. Overflow only costs hoists. */
    struct {
        uint32_t off;
        uint8_t  slot;
        ObjClosure *sample;
    } closSite[JIT_MAX_CLOS_SITES];
    unsigned  closSiteCount;
    /* JAITHON_JIT_CLOSURE_HOIST: over a loop that calls nothing and never
     * writes the local, an inlined closure's function guard is proved once
     * at the loop head, and the scalar upvalues its body reads are loaded
     * there into registers. A closure's `fn` never changes, and with no call
     * and no OP_SET_UPVALUE in compiled code nothing can write the cell while
     * the loop runs. See planClosureHoists. */
    struct {
        uint32_t top, end;
        uint8_t  slot;
        ObjFunction *fn;
        uint8_t  upCount;
        uint8_t  upIdx[JIT_MAX_CLOS_UP];
        uint8_t  upReg[JIT_MAX_CLOS_UP];
        uint8_t  upKind[JIT_MAX_CLOS_UP];
    } closHoist[JIT_MAX_CLOS_HOIST];
    unsigned  closHoistCount;
    /* 1 + the closure hoist covering the call being inlined, 0 for none;
     * read by emitGetUpvalue while `inlining`. */
    unsigned  inlClosHoist;
    /* The caller stores an inlined call's result straight into a local (the
     * next instruction is OP_SET_LOCAL or OP_BIND, and nothing branches to
     * it), so the result may stay a borrow of the inlined bank's register
     * for that one instruction instead of being copied into the caller's. */
    bool      inlBorrowResult;
    /* The function tier's walk starts here instead of at offset 0: the code
     * before it is an early-return arm already compiled ahead of the frame
     * (emitEarlyReturnArm), whose test the entry has therefore passed. */
    uint32_t  walkFrom;
    /* The body being inlined ends by constructing an instance, which calls
     * out to allocate: its entries continue the caller's callee-saved bank
     * (and count towards its save set) instead of taking x0..x8, which the
     * allocator's slow path destroys. See inlinableBody. */
    bool      inlShared;
    uint8_t   hoistPool[JIT_FREE_COUNT + JIT_SCRATCH_BANK_COUNT];
    unsigned  hoistPoolCount;
    unsigned  hoistTaken;
    /* How many entries the scratch bank may still hold once the hoists have
     * taken theirs off the top. The probe said the body never goes deeper, but
     * "said" is not "cannot": lowering the room here is what turns a walk that
     * disagreed with the probe into a decline instead of a value read out of a
     * register a hoisted header owns. */
    unsigned  scratchRoom;
    const char *whyNot;
    /* Somewhere to build a reason that names a number. `whyNot` is otherwise
     * always a literal, and "a local was given two different kinds" without
     * saying which one left the reader to guess -- a poor trade for ninety
     * bytes that live exactly as long as the emitter does. */
    char        whyBuf[96];
    /* WHICH of an arm's unnamed refusals fired, and on what value.
     *
     * `whyNot` is a decision: an arm that names one has told the reader
     * something durable. Several arms instead return false from a dozen places
     * without naming any of them, and `JAI_JIT_WHY` then prints the bare
     * opcode -- so one census row like "OP_GET_FIELD_LOCAL 129" is a merge of
     * eleven unrelated causes wearing one label, and the fixes they want are
     * nothing alike. This is that missing half: printed only when `whyNot` is
     * NULL, never read by control flow, and cleared at the top of every
     * instruction so a note cannot outlive the arm that wrote it.
     *
     * Inert by construction and checked as such: with it set at every site
     * below, the decline TOTAL is unchanged run for run. */
    char        whySub[80];
    uint8_t     lastOp;
    unsigned  descOffset;
    int       exceptionExit;
    bool      overflowUsed[3];
    int       overflowStub[3];

    /* One per guard: the bytecode offset to resume at, and a snapshot of the
     * compile-time model there. The stub that writes them is emitted after the
     * body, so the hot path keeps a single not-taken branch. */
    struct {
        uint32_t ip;
        unsigned depth;
        unsigned valueDepth;
        SlotKind kinds[JIT_MAX_STACK + 1];
        ObjClass *classes[JIT_MAX_STACK + 1];
        /* The topmost entry is the result of a call that already happened, so
         * it lives in the descriptor rather than a register. */
        bool     lastFromDesc;
        uint32_t fpLive;
        /* Live entries that are still a borrow of a local's d home, and
         * which home: the stub writes those out of the home, since nothing
         * was ever written to their own bank register (jitBorrowGuardsOn). */
        uint32_t fpBorrow;
        uint8_t  fpBorrowReg[32];
        int      stub;
    } deopt[JIT_MAX_DEOPT];
    unsigned  deoptCount;
    /* Self-call cold path, emitted with the stubs, not inline: the fast path is three instructions plus a
     * not-taken branch; interleaving thirty instructions of cold code between recursive call sites cost fib_recursive 25% (measured, same code, two layouts). */
    struct {
        int      returnTo;
        int      stub;
        unsigned roots;      /* the descriptor is on the frame chain when >0 */
        unsigned deoptBail;  /* record for verdict 1: re-execute the call */
        unsigned deoptKind;  /* record for a result of an unexpected kind */
        unsigned tag;
        unsigned resultReg;
        /* Closure to finish on verdict 4: NULL means this one (a self-call knows its own); a direct call to
         * another compiled function that writes names it here instead. */
        ObjClosure *callee;
        /* Continuation's required object type/shape, when the fast path's kind says more than "a heap object":
         * VAL_OBJ covers every heap object, and reading `klass` off an ObjString answers wrongly rather than faulting (the list-element-head bug) -- so the type is checked first. */
        int      retType;
        uint32_t retShape;
    } selfSlow[JIT_MAX_SELF_SLOW];
    unsigned  selfSlowCount;
    /* Out-of-line list-grow path (like the self-call cold half, to keep the store's 9 instructions and a
     * realloc call from sitting in the loop). Before this existed, list-full was a full deopt -- handing the WHOLE rest of the function to the interpreter -- which made compiled list-building bodies nearly worthless (merge in sort_merge deopted on every single push, 62434 times for 62500 items). */
    struct {
        int      returnTo;
        int      stub;
        unsigned listReg;
        unsigned valReg;
        unsigned tag;
        unsigned countReg;
        /* The stub saves and restores every caller-saved register the body
         * can hold a value in, so the append is not a clobber (see
         * jitGrowKeeps). Decided per site, at the site, so the stub can never
         * disagree with what the walk assumed. */
        bool     keeps;
        /* The site dispatches on storage after the grow, so jitListGrow may
         * unbox an untyped list it grows past JAI_LIST_SHAPE_AT. Cleared by
         * emitGrowStubs when the body also appends an object to the same
         * local (see jitShapeSiblingsOn). */
        bool     shape;
        /* The local appended to, or JIT_PUSH_UNKNOWN (also for an inlined
         * body, whose slots are the callee's). */
        int16_t  target;
    } grow[JIT_MAX_GROW];
    unsigned  growCount;
    /* One instruction a rare branch takes out of line and comes straight
     * back from (see emitColdFixup), so the common path falls through. */
    struct {
        int      stub;
        int      returnTo;
        uint32_t insn;
        /* 0: `insn` and back. 1: a bounds check's negative-index arm (see
         * emitBoundsNormalise): add the count, re-check, deopt to record
         * `deoptK` or come back. */
        uint8_t  kind;
        uint8_t  rOut, rCount;
        bool     countW;
        int      deoptK;
        /* kind 2: an element read's BOXED arm (see OP_GET_INDEX): the items
         * register and the tag the element must carry. */
        uint8_t  rItems;
        uint32_t tag;
    } cold[JIT_MAX_COLD];
    unsigned  coldCount;
    uint32_t  curOffset;
    uint32_t  chainSkip[JIT_MAX_CHAIN];
    unsigned  chainSkipCount;
    /* Model as it stood at the start of `curOffset`, before that instruction's own pushes -- a guard fires
     * mid-instruction and the interpreter resumes at its start, so this is what's live there (see deoptSite). */
    unsigned  instDepth;
    unsigned  instValueDepth;
    bool      hasSelfCall;
    unsigned  locals;
    unsigned  frameBytes;
    unsigned  savedCount;

    int       limitLiteral;
    int       bailBlock;

    /* 64-bit constants loaded from a pool after the code (emitConst64), and
     * every `ldr` that names one. Only a tier that lays the pool out
     * (emitLiteralPool) sets `litPool`; without it every constant is
     * materialised with movz/movk as before. */
    bool      litPool;
    unsigned  litCount;
    uint64_t  litVal[JIT_MAX_LITS];
    unsigned  litUseCount;
    int       litUseInst[JIT_MAX_LIT_USES];
    uint8_t   litUseIdx[JIT_MAX_LIT_USES];

    SlotKind  returnKind;
    uint32_t  returnShape;
    bool      sawReturn;
    /* Set for the real pass when the measuring pass merged the return kinds
     * to SLOT_DYNAMIC: every return site then leaves its own Value tag in
     * x1 (emitReturnLeave) instead of the bare zero verdict. Decided before
     * the first return is emitted, because a site cannot be re-tagged after
     * the fact and the x1 convention must be one thing for the whole body. */
    bool      dynamicReturn;
    /* The walk declined on a callee that has never returned while the
     * interpreter watched -- `obsReturnKind == JAI_FB_NONE` and no compiled
     * form. jaiJitEnter does not charge such a decline to the attempt budget;
     * see JAI_JIT_COLD_RETRIES. */
    bool      coldCallee;

    bool      failed;

    /* Value entries whose payload is in v(16 + index) rather than in their X
     * register. See JIT_FP_BANK. */
    uint32_t  fpLive;
    bool      fpOff;
    /* Offsets this walk carried an FP-resident value INTO: a branch landing on one would arrive with the
     * value only in X, so the compile declines and retries with fpOff. Safety net, not a real path -- straight-line float expressions are never branch targets; JAI_JIT_WHY reports it when it fires. */
    uint32_t  fpCarry[JIT_MAX_CARRY];
    unsigned  fpCarryCount;
    /* Offsets of an OP_BIND whose local's d register was written EARLY by the float operator just above it,
     * so the bind itself emits nothing -- nothing may branch here, since an arriving path skipped the operator. Checked against fixups at the end of the walk, since a back-edge isn't in the list yet mid-walk (same reason as fpCarry). */
    uint32_t  homeEarly[JIT_MAX_HOME_EARLY];
    unsigned  homeEarlyCount;
    /* Value entries that are a plain read of a float local, held in that LOCAL's own d register instead of
     * copied into the bank (`x * x` becomes one multiply instead of two fmovs + a multiply). Read-only borrow: only localOut/localOutFp ever write a local's d home, and both release the borrow first. Cleared on push/pop/claim/before any deopt record. */
    uint32_t  fpBorrow;
    uint8_t   fpBorrowReg[32];

    /* Two deferred-materialisation flavors, one mechanism: kPend is an int literal (OP_INT/OP_CONST) left
     * unmaterialised so a consumer can fold it as an imm12 (`n - 1` becomes just `subs`); xBorrow is a plain read of a register-resident local held in THAT register (X-twin of fpBorrow, `fib(n-1)` loses its `mov`). Both settle at the TOP of the next instruction (the one point unconditionally on the executed path -- a guard is not) unless whitelisted by deferSurvives. Nothing deferred may reach a deopt record: deoptRecordAt/branchOnDeopt assert it, so a wrong whitelist entry costs a decline, not a wrong answer. */
    uint32_t  kPend;
    int64_t   kPendVal[32];
    /* Which entries are known to BE a given integer literal, whether or not
     * they were materialised. kPend answers "may I fold this into the next
     * instruction" and is gone the moment the literal reaches a register;
     * this answers "what is this value", which stays true afterwards. Set
     * where a literal is pushed, cleared with every other per-entry mask on
     * push and pop, so it cannot outlive the entry it describes. */
    uint32_t  kKnown;
    int64_t   kKnownVal[32];
    /* Entries that are the float literal 2.0, pushed by OP_CONST. `2.0 * x`
     * is then `x + x` -- bit for bit the same double for every x, NaN, the
     * infinities and -0.0 included, and an add is a cycle shorter than a
     * multiply on a dependency chain. See JAITHON_JIT_FTWO. */
    uint32_t  fTwo;
    uint32_t  xBorrow;
    uint8_t   xBorrowReg[32];
    /* Dynamic locals (slots 0..63) whose tag -- and class, for an instance --
     * this edge has already guarded since the slot was last written: a second
     * read of the same value along the same straight line needs no second
     * guard. Retired exactly where a field-kind memo is (a join, a call out, a
     * write of the slot). See JAITHON_JIT_DYN_MEMO. */
    uint64_t  dynGuarded;
    /* Entries that are an X-side copy of a float LOCAL, taken by OP_GET_LOCAL when no float consumer was
     * in reach (`sum += f(i) * v[j]` pushes `sum` before the call). While no local has been written and no
     * join crossed since the read (localEpoch unchanged), the local's own d home still holds exactly those
     * bits, so the float operator that finally consumes the entry reads the home instead of moving the copy
     * back across register files -- which is what took the loop-carried `fmov d,x; fadd; fmov x,d` off the
     * accumulator's chain. See fpOperandReread and JAITHON_JIT_FP_REREAD. Cleared with every other
     * per-entry mask on push and pop, and by fpClaim and the in-place rewrites. */
    uint32_t  fpSrc;
    uint8_t   fpSrcSlot[32];
    uint32_t  fpSrcEpoch[32];
    uint32_t  localEpoch;
    /* Which entries are known to be a LOCAL plus a constant, and which local
     * and what constant. Not a value like kKnown -- a shape. It exists so a
     * subscript can say "this index is the loop counter, minus one" and have
     * its bounds check hoisted to the loop head, where one compare covers
     * every iteration instead of two instructions per element.
     *
     * Only int-kinded entries carry it, and only while the arithmetic stays
     * exact: an add or subtract that could overflow drops the shape rather
     * than describing a value the loop head's guard would not cover. Cleared
     * with every other per-entry mask on push and pop. */
    /* Where each admitted PIC arm jumps to reach the merge. One per way, and
     * the caller patches them all once the descriptor path it emits after the
     * chain is behind it. */
    int       picExits[JAI_IC_WAYS];
    unsigned  picExitCount;

    uint32_t  idxKnown;
    uint8_t   idxBase[32];
    int32_t   idxOff[32];
    /* The span of offsets each list slot is subscripted at, over the sites
     * whose index turned out to be the loop counter plus a constant. The
     * measuring pass fills it; the real pass turns it into ONE compare at the
     * loop head and lets those sites skip their own. A slot with no shaped
     * site keeps `spanLo > spanHi`, which is how "nothing to hoist" reads. */
    int32_t   spanLo[JIT_MAX_SLOTS + 1];
    int32_t   spanHi[JIT_MAX_SLOTS + 1];
    bool      spanOk[JIT_MAX_SLOTS + 1];
    /* Which loop variable the span is measured against. Two loops subscripting
     * the same list off different variables cannot share one span, so the
     * second one seen turns the slot off rather than widening it. */
    uint8_t   spanBase[JIT_MAX_SLOTS + 1];
    bool      spanSeen[JIT_MAX_SLOTS + 1];
    /* The OSR loop is a counted range whose head is OP_FOR_RANGE_BIND, and
     * these are its three slots: the variable the body reads, the counter, and
     * the end. The last two are fresh temporaries the emitter hands out per
     * loop and nothing else writes -- see the arm for OP_FOR_RANGE_BIND --
     * which is what makes the end a loop invariant a single compare can use.
     * Decoded once at setup because the guard is emitted ABOVE the head, before
     * the walk reaches the instruction that would otherwise name them. */
    bool      rangeHead;
    uint16_t  rangeVar, rangeCur, rangeEnd;
    /* Offsets this walk carried a deferred entry into (same reason as fpCarry): a BACKWARD branch there
     * would arrive with the value in-register while the instruction reads the borrow. Forward branches are caught during the walk; this catches the rest. */
    uint32_t  deferCarry[JIT_MAX_CARRY];
    unsigned  deferCarryCount;
} Emit;

/* Registers the operand stack starts after: none of them, once locals live in
 * the frame. */
/* x19 carries the slots pointer in OSR mode, so the operand stack starts one
 * register later. */
#define JIT_SLOTS_REG (JIT_FIRST_SAVED)
#define JIT_IDX_REG   (JIT_FIRST_SAVED + 1u)
#define JIT_LIM_REG   (JIT_FIRST_SAVED + 2u)
/* The ObjList a list head is walking. A range head has no use for a third register (see osrReserved),
 * so this is numbered after the two a range keeps, and a range's locals begin where it would have been. */
#define JIT_START_REG (JIT_FIRST_SAVED + 3u)
#define JIT_ITER_REG  (JIT_FIRST_SAVED + 4u)
/* The ObjIter a dict-items head is walking, and the ONLY register that head
 * reserves: it keeps no index and no limit, because the step reads both out of
 * the iterator and writes the index straight back, so nothing has to be
 * unwound at an exit. Numbered second so a dict head's locals start where a
 * range head's index would have been -- see osrReserved. */
#define JIT_PAIR_ITER_REG (JIT_FIRST_SAVED + 1u)

/* Counts from the bottom of the operand stack, not the top (unlike pushReg). An inlined body gets its
 * own bank (x0..x8, minus the emitter's x9..x12 scratches) whenever the caller's stack is NOT already
 * there: it can't call anything, so every caller-saved register is free and costs the caller nothing -- `evalA` inlined into spectral's inner loop needs eight live values where the OSR form had only six left, so without this it wouldn't fit. When the caller IS on that bank the two share one numbering instead; see inlineOwnBank. */
#define JIT_INL_BANK   0u    /* x0..x8, all caller-saved */
#define JIT_INL_COUNT  JIT_SCRATCH_BANK_COUNT

/* v16.. for the ordinary bank, v2..v7 for an inlined body -- caller-saved, and
 * far enough from v16+JIT_MAX_SAVED, which the local-add path uses as a temp. */
#define JIT_INL_FP_BANK 2u

/* How an element access should be emitted.
 *
 * `stg` is the storage the first arm is emitted at. When `dynamic` is set a
 * second arm at `alt` follows it, with a runtime test on ObjList::stg picking
 * between the two -- see listDispatchBegin. */
typedef struct {
    uint8_t stg;
    uint8_t alt;
    bool    dynamic;
} ListAccess;

typedef enum { DISCARD_NO, DISCARD_POP, DISCARD_POP_RETURN } DiscardKind;

/* Defined in jit_runtime.c. */
extern JitCallDesc *gJitFrames;
extern JitDeoptRecord gDeopt;

void jitThrowOverflow(int64_t which);
int jitInvokeMethod(JitCallDesc *d);
int jitInvokeByName(JitCallDesc *d);
int jitInvokeByNameLearn(JitCallDesc *d);
int jitInvokeNative(JitCallDesc *d);
int jitBuildList(JitCallDesc *d);
int jitContains(JitCallDesc *d);
int jitNotContains(JitCallDesc *d);
int jitBuildDict(JitCallDesc *d);
int jitBuildSet(JitCallDesc *d);
int jitBuildTuple(JitCallDesc *d);
int jitValuesEqual(JitCallDesc *d);
bool jitObjEquality(void);
int jitStringConcat(JitCallDesc *d);
int jitMakeRangeIter(JitCallDesc *d);
int jitMakeIter(JitCallDesc *d);
int jitMakeItemsIter(JitCallDesc *d);
int jitMakeEnumIter(JitCallDesc *d);
/* The element a list head is about to bind, from `src[at]`, and whether the
 * list's first 1024 elements hold instances of more than one class. Answers
 * false when nulls are dense enough -- past one in 64 -- that a form pinned to
 * the sampled class would bail once per null; see the table in
 * jaiJitEnterOsr. Shared by the list head, the enumerate head and the
 * function tier's enumerate arm, so all three refuse on the same data. */
bool jitListHeadSample(const ObjList *src, int at, Value *sample, bool *mixed);
int jitFormat(JitCallDesc *d);
int jitListGrow(ObjList *list, uint64_t tag, int64_t payload,
                int64_t mayShape);
bool jitListShapeable(Value v);
ObjInstance *jitInstanceAlloc(ObjClass *cls);
ObjIter *jitIterAlloc(Obj *source);
bool jitIterAllocOn(void);
ObjInstance *jitInstanceAllocBare(ObjClass *cls);
int jitNewInstance(JitCallDesc *d);
int jitGetSlice(JitCallDesc *d);
int jitGetIndexDict(JitCallDesc *d);
int jitMakeDictKeysIter(JitCallDesc *d);
int jitSetIndexDict(JitCallDesc *d);

/* The string-keyed dict leaves and the arm that calls them; see jitDictGetStr. */
#define JIT_DICT_ABSENT_SLOW 0xFFFFu
bool jitDictLeaf(void);
bool jitDictProbeOn(void);
ObjDict *jitDictProbe(void);
/* A dict the model has a sample of, or knows by its predicted type alone. */
#define JIT_SEEN_OR_PREDICTED_DICT(e, idx) \
    (IS_DICT((e)->stackSeen[idx]) ||       \
     (jitDictProbeOn() &&                  \
      (e)->stackObjType[idx] == (uint8_t)(OBJ_DICT + 1)))
int64_t jitDictGetStr(ObjDict *d, ObjString *key, Value *result,
                      uint64_t defTag, int64_t defPayload);
int64_t jitDictSetStr(ObjDict *d, ObjString *key, uint64_t tag,
                      int64_t payload);
int64_t jitDictHasStr(Obj *container, Obj *needle, Value *result,
                      int64_t negate);
int64_t jitDictAddStr(Obj *dictObj, Obj *keyObj, int64_t defPayload,
                      int64_t addend, int64_t absentSlow);
int64_t jitDictGetInt(ObjDict *d, int64_t key, Value *result, uint64_t defTag,
                      int64_t defPayload);
int64_t jitDictSetInt(ObjDict *d, int64_t key, uint64_t tag, int64_t payload);
int64_t jitDictHasInt(Obj *container, int64_t key, Value *result,
                      int64_t negate);
int64_t jitDictAddInt(Obj *dictObj, int64_t key, int64_t defPayload,
                      int64_t addend, int64_t absentSlow);
typedef struct {
    int      slow[2];   /* the branches to the descriptor path, or -1 */
    unsigned cond[2];   /* the condition each of them branches on */
    int      done;      /* the branch over it */
    bool     on;        /* a leaf was emitted at all */
} LeafFix;
void emitDictLeafGet(Emit *e, unsigned rDict, unsigned rKey, SlotKind keyKind,
                     int defIdx, SlotKind defKind, bool absentSlow,
                     LeafFix *fx);
void emitDictLeafHas(Emit *e, unsigned rDict, unsigned rKey, SlotKind keyKind,
                     bool negate, LeafFix *fx);
/* Whether the dict leaves take the key at a stack entry: an int, or an object
 * not seen to be anything but a string. */
bool dictLeafKeyAt(const Emit *e, unsigned idx);
bool emitDictAddFused(Emit *e, const Chunk *chunk, int off, int count,
                      unsigned ridx);
bool emitDictAugAddFused(Emit *e, const uint8_t *code, int off, int count);
bool leafRegOk(unsigned r);
/* Defined in jit_body_build.c: the inline probe of a site's f-string memo.
 * `hit` is the probe's b.eq, -1 when the site has no memo. */
typedef struct {
    int         hit;
    JaiFmtSite *site;
} FmtMemoFix;
void emitFmtMemoProbe(Emit *e, unsigned rPre, unsigned rN, unsigned rPost,
                      FmtMemoFix *fix);
void emitFmtMemoCall(Emit *e, const FmtMemoFix *fix);
void fmtMemoHitHere(Emit *e, int at);
int emitFmtMemoFillTest(Emit *e, const FmtMemoFix *fix);
void emitFmtMemoFillStub(Emit *e, int test, int answered,
                         const FmtMemoFix *fix, unsigned rPre, unsigned rN,
                         unsigned rPost);
bool jitLeafInReg(void);
void leafSlowHere(Emit *e, LeafFix *fx);
void leafDoneHere(Emit *e, LeafFix *fx);
int jitCallOut(JitCallDesc *d);

/* Defined in jit_osr.c. */
int instructionLength(const Chunk *c, int off);
/* Filled by loopDepthTable, which lives with the OSR entry below. */
const uint8_t *loopDepthFor(const Chunk *c, unsigned *count);

/* Defined in jit_func.c. */
const char *slotKindName(SlotKind k);
void emit(Emit *e, uint32_t word);
unsigned osrReserved(const Emit *e);
unsigned osrAccReg(const Emit *e);
void emitTagFor(Emit *e, SlotKind kind, unsigned payloadReg,
                       unsigned tagReg, unsigned spare);
uint8_t localStgOf(const Emit *e, unsigned slot);
unsigned valueBankReg(const Emit *e, unsigned idx);
unsigned fpRegAt(const Emit *e, unsigned idx);
void emitConst64(Emit *e, unsigned rd, int64_t value);
bool jitLitPoolOn(void);
void emitConstCmp(Emit *e, unsigned rd, int64_t value);
/* Lays out the constants emitConst64 pooled, after everything else, and
 * points each load at its constant. False if the code buffer filled. */
bool emitLiteralPool(Emit *e);
void emitSaveRestore(Emit *e, bool save);
void emitFrameEnter(Emit *e);
void emitFpSaveRestore(Emit *e, bool save);
void emitEpilogue(Emit *e, unsigned bailed);
void emitEpilogueKeepX1(Emit *e);
void emitReturnLeave(Emit *e, SlotKind k);
bool jitSplitStress(void);
bool jitGrowKeeps(void);
bool jitHoistPinOn(void);
bool jitSplitHoistsOn(void);
bool jitHoistPartialOn(void);
bool jitIndexedLoadOn(void);
void notePushTarget(Emit *e, int slot);
void noteStorageStamp(Emit *e);
bool regionStamps(const Emit *e, uint32_t lo, uint32_t hi);
bool regionCalls(const Emit *e, uint32_t lo, uint32_t hi);
void planHoists(Emit *e, ObjFunction *fn, const SlotKind *kinds);
const char *declineReason(Emit *e);
unsigned jitShapeLimit(void);
const char *unarmedDetail(const ObjFunction *fn, uint8_t op,
                                 uint32_t at);
void emitSelfSlowStubs(Emit *e, ObjClosure *closure);
void emitGrowStubs(Emit *e);
bool compileBody(Emit *e, ObjClosure *closure);
void jitFree(int *map, int64_t *depths, int *chunkDepth, int count);
int *chunkDepthTable(const ObjFunction *fn);
void planSlotRegisters(Emit *e, const Emit *m, unsigned availX,
                              const bool *skip, unsigned *strandedOut);
uint8_t *arenaEmit(JaiCodeArena *arena, const uint32_t *code,
                          unsigned count);

/* Defined in jit_func.c. */
extern bool gInlineFailed;
/* What a map kernel loops over: `src` and `dst` are the two lists' element
 * arrays, both eight bytes wide, and the kernel runs `i` up to `n` in a
 * register, writing it back here before each element so that every way out
 * leaves it naming the element it stopped on. One run at a time -- the VM runs one
 * Jaithon thread, and a kernel calls nothing. */
typedef struct {
    const void *src;
    void       *dst;
    int64_t     i;
    int64_t     n;
    /* A filter kernel's count of elements kept, in a register as `i` is and
     * written back beside it. */
    int64_t     j;
} JitMapRun;
extern JitMapRun gJitMapRun;
void emitMapKernelNext(Emit *e, SlotKind k);
/* An `if`/`elif` chain of int assignments at *offp emitted as csel; see
 * jit_body_cmp.c. False, having emitted nothing, when it does not apply. */
bool jitTryIfConvert(Emit *e, ObjFunction *fn, int *offp);

/* Defined in jit_compile.c. */
bool adoptLocalKindSeen(Emit *e, unsigned slot, SlotKind kind,
                               uint32_t shape, ObjClass *klass, Value seen);
bool adoptLocalKind(Emit *e, unsigned slot, SlotKind kind,
                           uint32_t shape, ObjClass *klass);

/* Defined in jit_func.c. */
unsigned localFrameOff(const Emit *e, unsigned slot);
bool localTagInFrame(const Emit *e, unsigned slot);
unsigned localTagFor(const Emit *e, unsigned slot);
unsigned closureReg(const Emit *e);
bool jitChainOn(void);
void reportChain(const Emit *proto, Emit *first, ObjClosure *closure,
                        ObjFunction *fn);
bool jitCollectClashes(void);

/* Defined in jit_body.c. */

/* Defined in jit_func.c. */
bool elemDeclOn(void);
bool holdsRegister(SlotKind k);
unsigned localIn(Emit *e, unsigned slot, unsigned scratch);
unsigned localDest(const Emit *e, unsigned slot);
void forgetFieldKinds(Emit *e);
void jitEmitReset(Emit *e);
void noteIndexSpan(Emit *e, int slot, bool shaped, int32_t off,
                          uint8_t base);
void noteSlotIndexed(Emit *e, int slot);
void localOut(Emit *e, unsigned slot, unsigned src);
void localInFp(Emit *e, unsigned slot, unsigned dst);
void localOutFp(Emit *e, unsigned slot, unsigned src);
bool localInRange(Emit *e, unsigned slot);
Value seenLocal(Emit *e, unsigned slot);
void noteScratchClobber(Emit *e);
unsigned valueXReg(const Emit *e, unsigned idx);
unsigned pushReg(const Emit *e);
unsigned valueBankRoom(const Emit *e);
unsigned fpHeldIn(const Emit *e, unsigned idx);
void fpSyncOne(Emit *e, unsigned idx);
void fpReleaseHome(Emit *e, unsigned reg);
void fpReleaseAll(Emit *e);
bool jitBorrowGuardsOn(void);
unsigned deoptFpSource(const Emit *e, unsigned k, unsigned valueIdx);
void fpSyncAll(Emit *e);
unsigned fpOperand(Emit *e, unsigned idx);
bool jitFTwo(void);
void fpSrcNote(Emit *e, unsigned idx, unsigned slot);
unsigned fpOperandReread(Emit *e, unsigned idx);
void fpClaim(Emit *e, unsigned idx);
void fpBorrowLocal(Emit *e, unsigned idx, unsigned reg);
unsigned fpBindDest(Emit *e, unsigned slot, unsigned bank);
unsigned fpBindLookahead(Emit *e, const uint8_t *code, int next,
                                int stop, const ObjFunction *fn,
                                uint32_t *bindOffOut);
bool anyDeferred(const Emit *e);
void settleAll(Emit *e);
/* d register holding entry `idx`, loaded from X if that's where it still lives. Leaves fpLive
 * untouched -- the two copies now agree, and marking the X copy stale when it isn't would cost a needless sync. */
unsigned xHeldIn(Emit *e, unsigned idx);
unsigned localHomeX(const Emit *e, unsigned slot);
void noteLocalBorrowed(Emit *e, unsigned slot);
void xBorrowLocal(Emit *e, unsigned idx, unsigned reg);
void kPendLocal(Emit *e, unsigned idx, int64_t k);
bool pendingImm12(const Emit *e, unsigned idx, int64_t *out);
void emitCmpImm(Emit *e, unsigned rn, int64_t k);
void emitAddSubImm(Emit *e, unsigned rd, unsigned rn, int64_t k,
                          bool subtract);
bool pushValue3(Emit *e, SlotKind kind, uint32_t shape, ObjClass *klass,
                       Value seen, int fromLocal);
bool pushValue(Emit *e, SlotKind kind, uint32_t shape, ObjClass *klass);
SlotKind knownFieldKind(const Emit *e, int local, uint16_t field);
void recordFieldStore(Emit *e, int local, uint16_t field, SlotKind kind);
void recordFieldRead(Emit *e, int local, uint16_t field, SlotKind kind);
bool stackLocalCurrent(const Emit *e, unsigned depthIdx);
bool jitFieldReadMemoOn(void);
bool pushSelf(Emit *e);
void clearStackProofs(Emit *e);
bool anyStackProof(const Emit *e);
bool popValueRaw(Emit *e, unsigned *reg, SlotKind *kind);
bool popValue(Emit *e, unsigned *reg, SlotKind *kind);
void dropCalleeEntry(Emit *e);
int64_t stackSignatureAt(const Emit *e, unsigned depth);
int64_t stackSignature(const Emit *e);
unsigned jitJoinMode(void);
void branchTo(Emit *e, uint32_t targetOffset, bool conditional,
                     unsigned cond);
void branchToDepth(Emit *e, uint32_t targetOffset, unsigned cond,
                          int64_t depthOverride);
bool jitModuleCalls(void);
bool jitClassCalls(void);
bool jitModuleNativeCalls(void);
extern bool gNullableFbUsed;
extern bool gNoNullableFb;
bool jitNullableFbOn(void);
bool jitMaybeObjStackOn(void);
bool jitMaybeObjOn(void);
bool jitMatchArm(void);
bool jitKeepRetryNeeds(void);
bool jitDynamicReturn(void);
bool jitStaticMethodOn(void);

bool jitPairListOn(void);
bool modelAgreesWithChunk(const Emit *e, uint32_t off);
bool deoptRecordAt(Emit *e, uint32_t ip, bool lastFromDesc,
                          unsigned *out);
void branchOnDeoptAt(Emit *e, unsigned cond, uint32_t ip,
                            bool lastFromDesc);
void branchOnDeopt(Emit *e, unsigned cond);
void branchOnDeoptInstStart(Emit *e, unsigned cond);
void nanToDeopt(Emit *e);
bool isOrdering(uint8_t op);
bool stringOperand(const Emit *e, unsigned at);
bool oneBytePair(const Emit *e, unsigned a, unsigned b);
void emitOneByteString(Emit *e, unsigned at, unsigned reg, unsigned dst);
bool jitStrCmpOn(void);
bool jitStrCmpEqOn(void);
bool jitConstAscii(void);
bool jitOneByteLocalK(void);
bool preferLeafEquality(const Emit *e, unsigned da, unsigned db);
void emitStringOrder(Emit *e);
void emitListBoxedGuard(Emit *e, unsigned rList, unsigned scratch);
uint8_t listAltFor(SlotKind vk);
ListAccess listAccessFor(Emit *e, unsigned rList, int slot,
                                SlotKind vk, unsigned scratch);
int listDispatchBegin(Emit *e, const ListAccess *a, unsigned rList,
                             unsigned scratch);
int listDispatchElse(Emit *e, int skip);
void listDispatchEnd(Emit *e, int join);
unsigned listStgShift(uint8_t stg);
SlotKind listStgKind(uint8_t stg);
void emitElemStoreAt(Emit *e, uint8_t stg, unsigned rItems,
                            unsigned rIdx, unsigned vtag, unsigned rVal);
void emitListHeader(Emit *e, unsigned rList, unsigned rItems,
                           unsigned rCount);
int hoistFor(const Emit *e, int slot);
int pushHoistFor(const Emit *e, int slot);
int iterHoistAt(const Emit *e, uint32_t top);
int closHoistFor(const Emit *e, int slot, uint32_t at, const ObjFunction *fn);
bool jitInlineBorrow(void);
void noteGlobalAccess(Emit *e, JaiEntry *slot, SlotKind kind, bool write);
bool globalTagProven(const Emit *e, JaiEntry *slot, SlotKind kind);
int globalPromotedReg(const Emit *e, JaiEntry *slot, SlotKind kind);
void emitPromotedReload(Emit *e);
bool jitSimpleInitClass(ObjClass *cls, unsigned argc);
bool closUpHoisted(const Emit *e, ObjClosure *closure, unsigned index);
void noteClosureSite(Emit *e, int slot, uint32_t off, ObjClosure *sample);
int hoistForStr(const Emit *e, int slot);
ObjDict *jitDictExemplar(void);
bool jitFieldDict(void);
bool jitIterSoft(void);
bool jitIterEmptySkip(void);
bool jitPairInstance(void);
void jitBranchTargetsReset(void);
bool jitOsrSelfGlobal(void);
void planStrFacts(Emit *e, ObjFunction *fn);
bool strFactAscii(const Emit *e, int slot);
bool strFactInterned(const Emit *e, int slot);
void noteSlotStrEq(Emit *e, int slot);
bool jitIterRecycle(void);
extern ObjIter *gJitIterSpare;
void noteSlotStored(Emit *e, int slot);
unsigned jitCarryLimit(void);
bool boundsCoveredAtHead(const Emit *e, int slot, unsigned vidx,
                                int32_t *offOut, uint8_t *baseOut);
void emitHoistsAt(Emit *e, uint32_t off);
void emitBoundsNormalise(Emit *e, unsigned rIdx, unsigned rCount,
                                unsigned rOut, bool countW);
void branchOnOverflow(Emit *e, unsigned which, unsigned cond);
unsigned ovfDest(const Emit *e, unsigned home);
bool raiseExitAllowed(Emit *e, const char *what);
bool negatedCondition(uint8_t cmp, unsigned *out);
ObjClass *globalClass(ObjClosure *closure, uint32_t nameIdx);
bool jitGlobalEnum(void);
bool jitOsrGlobalSoft(void);
bool firstLiveEntry(const JaiTable *t, Value *key, Value *value);
ObjFunction *globalFunction(ObjClosure *closure, uint32_t nameIdx,
                                   Value *out);
ObjNative *globalNative(ObjClosure *closure, uint32_t nameIdx,
                               Value *out);
ObjModule *globalNamespace(ObjClosure *closure, uint32_t nameIdx);
bool globalIsSelf(ObjClosure *closure, uint32_t nameIdx);
JaiEntry *globalSlot(Emit *e, ObjClosure *closure, uint32_t nameIdx,
                            Value *out);
void emitGlobalsGuard(Emit *e);
bool jitStaticFieldEnabled(void);
JaiEntry *staticFieldSlot(Emit *e, ObjClass *klass, ObjString *name);
JaiEntry *moduleMemberSlot(Emit *e, ObjModule *m, ObjString *name);
void emitModuleGuard(Emit *e);
bool listProbeOn(void);
bool moduleFieldOn(void);
void emitStaticsGuard(Emit *e);
void emitVersionBump(Emit *e, ObjModule *m);
const char *jaiFeedbackName(uint8_t fb);
bool feedbackSlotKind(uint8_t fb, SlotKind *k, unsigned *tag,
                             uint8_t *objType);
bool emitListStore(Emit *e, SlotKind vk, unsigned rList, unsigned rVal,
                          int slot, int target);
bool siteInvokeResultKind(const Chunk *chunk, uint16_t cacheIdx,
                                 SlotKind *k, unsigned *tag);
bool rawObjValue(Value v);
bool observedReturnKind(const ObjFunction *cfn, SlotKind *k,
                               uint32_t *shape, uint8_t *objType);
bool globalKind(Value v, SlotKind *k, uint32_t *shape, ObjClass **kls);
bool declaredScalarFieldKind(uint32_t typeId, SlotKind *k, unsigned *tag);
bool jitDeclaredFieldKindEnabled(void);
bool isClassCallee(const Emit *e, unsigned argc);
bool subWhy(Emit *e, const char *fmt, ...);
const char *kindClash(Emit *e, unsigned slot);
bool emitRootFill(Emit *e, unsigned d, unsigned *nrootsOut);
bool emitDescriptorStatus(Emit *e, Value calleeVal, unsigned first,
                                 unsigned nargs, void *helper, bool ownStatus,
                                 int calleeReg);
bool emitDescriptor(Emit *e, Value calleeVal, unsigned first,
                           unsigned nargs, void *helper);
bool emitDescriptorFull(Emit *e, Value calleeVal, unsigned first,
                        unsigned nargs, void *helper, bool ownStatus,
                        int calleeReg, bool noRoots);
bool concatOperands(const Emit *e, Value *sample);
bool emitStringConcat(Emit *e, Value sample);
bool nullLiteralPair(const Emit *e, uint8_t op, SlotKind ka, SlotKind kb);
DiscardKind discardedAfter(const uint8_t *code, int at, int count);
bool emitFusedReturnNull(Emit *e, ObjFunction *fn);
bool jitStrIter(void);
bool jitIterStorage(void);
bool jitStringHead(void);
bool jitCompAcc(void);
bool jitBuiltinClass(void);
bool jitInvokeSoftRecv(void);
bool jitInvokeSoftCold(void);
#ifdef JAI_ALLOC_CENSUS
void jaiDeoptHitEmit(Emit *e, const char *fn, uint32_t top, unsigned ord,
                     uint32_t ip);
#endif
bool pushLocalAsValue(Emit *e, unsigned slot);
bool jitConcatLocals(void);
bool emitFieldRead(Emit *e, const JaiJitFieldRead *fr, Value nativeVal,
                          unsigned ridx, unsigned argc, uint32_t afterIp);
bool directCallArgsMatch(Emit *e, const ObjFunction *cfn,
                                unsigned firstIdx, unsigned argc);
void emitMaybeInstResult(Emit *e, unsigned dst, unsigned rat,
                                uint32_t rshape, uint32_t deoptIp);
bool jitAnyGuard(void);
bool jitReturnKnownOn(void);
/* The float divide-by-zero guard in emitAddSubDiv. ON by default -- it is a
 * correctness fix, not an optimisation. JAITHON_JIT_FDIV_GUARD=0 exists only so
 * its cost (one fcmp and one not-taken branch per compiled float division) can
 * be A/B'd inside one binary, which is the only honest way to price it. */
bool jitFdivGuard(void);
bool emitDirectCall(Emit *e, ObjFunction *caller, ObjFunction *cfn,
                           Value calleeVal, int calleeReg, unsigned cidx,
                           unsigned argc, uint32_t callOff, uint32_t after,
                           bool method);
bool jitPicEnabled(void);
bool emitInvokePic1(Emit *e, ObjFunction *fn, unsigned ridx,
                           unsigned argc, uint32_t callOff, uint32_t after,
                           int siteCache, bool havePrediction, SlotKind rkind,
                           int *toEnd);
/* An OSR compile's polymorphic site took `ways` of the ways its cache holds,
 * short because a way's callee had not compiled yet (or its class was not on
 * record). jit_osr.c attaches the note to the form it installs, and
 * re-compiles that form once more of the site's ways have become usable.
 * JAITHON_JIT_PIC_UPGRADE=0 turns the re-compile off. */
void jitOsrPicShort(int siteCache, unsigned ways, SlotKind rkind);
/* An OSR compile met a call to `cfn`, which has no compiled form yet but is
 * being called and will reach the threshold soon. True means: decline this
 * compile and look again on a later tick, rather than build a form that calls
 * it the slow way for the rest of the run (jit_osr.c). */
bool jitOsrColdWait(Emit *e, const ObjFunction *cfn);
bool jitPicUpgradeOn(void);
/* Parameters of `fn` the whole-function tier should bind unpinned -- an
 * instance of no class in particular -- because its pinned form kept
 * declining calls that passed another class there (jit_entry.c). Bit i is
 * slot i. */
uint32_t jitPolyParamMask(const ObjFunction *fn);

bool offsetIsBranchTarget(const Chunk *c, uint32_t off);
bool literalIntOperand(const ObjFunction *fn, int prevOff, int off,
                              int64_t *out);
void emitFloorFixup(Emit *e, unsigned rrem, unsigned rd,
                           bool signKnown, int64_t divisor, uint32_t fixup);
bool powerOfTwoShift(int64_t k, unsigned *shift);
bool inlineMethodCall(Emit *e, ObjFunction *caller, ObjClosure *method,
                      unsigned argc, uint32_t callOff);
bool emitColdFixup(Emit *e, unsigned rrem, uint32_t insn);
int deoptRecordNow(Emit *e);
bool jitDeoptStressOn(void);
bool jitBoundsColdOn(void);
bool jitDispatchColdOn(void);
bool inlineGlobalCall(Emit *e, ObjFunction *caller, ObjClosure *callee,
                             unsigned argc, uint32_t callOff, int calleeReg);
bool emitGlobalCall(Emit *e, ObjFunction *caller, unsigned argc,
                           uint32_t callOff, uint32_t after);
bool emitModuleCall(Emit *e, ObjModule *m, Value calleeVal,
                           unsigned ridx, unsigned argc, uint32_t after);
bool primInvokePairFits(const Emit *e, const Chunk *chunk, uint32_t off);
bool emitModuleNativeCall(Emit *e, ObjModule *m, Value calleeVal,
                                 unsigned ridx, unsigned argc, uint32_t after);
bool emitClassCall(Emit *e, ObjClass *klass, JaiEntry *slot,
                          Value calleeVal, unsigned ridx, unsigned argc,
                          uint32_t after);
bool inlineMethod(Emit *e, ObjClosure *closure, uint32_t nameIdx,
                         unsigned argc, int callOff);
bool exemplarKind(Value elem, SlotKind *kind, unsigned *tag,
                         ObjClass **cls, uint32_t *shape);
bool buildListExemplar(const Emit *e, unsigned first, unsigned n,
                              Value *out);
bool dictUniformValue(ObjDict *dict, Value *out);
bool jitSoftField(void);
bool jitMembership(void);
bool jitTuple(void);
bool jitNegate(void);
bool jitWrapArith(void);
bool jitListResult(void);
int emitNativeResultCall(Emit *e, Value cv, const char *nm,
                                unsigned argc, uint32_t afterIp);
bool emitCallOut(Emit *e, unsigned argc);

/* Defined in jit_list.c. */
void emitListElemStore(Emit *e, uint8_t stg, unsigned vtag,
                              unsigned rVal);

/* Defined in jit_global.c. */
bool retObjTypeOn(void);
bool emitGetGlobal(Emit *e, ObjFunction *fn, ObjClosure *closure,
                   const uint8_t *code, int *offp, int stop);
bool emitSetGlobal(Emit *e, ObjClosure *closure, const uint8_t *code,
                   int *offp);

/* Defined in jit_frame.c. */

/* Defined in jit_call.c. */

/* An arm's three outcomes. REFUSED is zero and OK is one so that an arm body's
 * own `return false` and `return subWhy(...)` still mean "declined". */
typedef enum { JIT_ARM_REFUSED = 0, JIT_ARM_OK = 1, JIT_ARM_UNARMED = 2 }
    JitArmResult;

/* Defined in jit_body.c. */
bool fpWorthLoading(const Emit *e, const uint8_t *code, int next, int stop);
bool emitUnarmedDeopt(Emit *e, const Chunk *c, int *off, int stop);

/* Defined in jit_body_return.c. */
bool mergeReturnKind(Emit *e, SlotKind k, uint32_t shape);
bool emitReturn(Emit *e, int *offp);
bool emitReturnNull(Emit *e, ObjFunction *fn, int *offp);
bool emitPopReturnNull(Emit *e, ObjFunction *fn, int *offp);

/* Defined in jit_body_build.c. */
bool emitBuildList(Emit *e, const uint8_t *code, int *offp);
JitArmResult emitBuildDictSet(Emit *e, const uint8_t *code, int *offp);
JitArmResult emitBuildTuple(Emit *e, const uint8_t *code, int *offp);
JitArmResult emitElemKind(Emit *e, const uint8_t *code, int *offp);
bool emitFormat(Emit *e, ObjClosure *closure, const uint8_t *code, int *offp);

/* Defined in jit_body_local.c. */
bool emitGetLocal(Emit *e, const uint8_t *code, int *offp, int stop);
bool emitGetLocal2(Emit *e, const uint8_t *code, int *offp, int stop);
bool emitSetLocal(Emit *e, const uint8_t *code, int *offp);
bool emitBind(Emit *e, const uint8_t *code, int *offp);
bool emitGetUpvalue(Emit *e, ObjFunction *fn, ObjClosure *closure,
                    const uint8_t *code, int *offp);

/* Defined in jit_body_cmp.c. */
bool emitCompare(Emit *e, uint8_t op, int *offp);
bool emitIsTest(Emit *e, uint8_t op, int *offp);
bool emitJumpIfCmpFalse(Emit *e, const uint8_t *code, int *offp);
bool emitJumpIfCmpLocalK(Emit *e, ObjFunction *fn, const uint8_t *code,
                         int *offp);
JitArmResult emitMembership(Emit *e, const uint8_t *code, int *offp);

/* Defined in jit_body_match.c. */
extern bool gMatchUsed;
extern bool gNoMatchArm;
bool popSkipTarget(const Emit *e, uint32_t at);
JitArmResult emitMatchTypePop(Emit *e, ObjClosure *closure,
                              const uint8_t *code, int *offp);
JitArmResult emitEnumTag(Emit *e, int *offp);
JitArmResult emitSwapPop(Emit *e, int *offp);
JitArmResult emitMatchConstPop(Emit *e, ObjFunction *fn, const uint8_t *code,
                               int *offp, bool *afterUncondp);

/* Defined in jit_body_field.c. */
bool emitTypeGuard(Emit *e, ObjFunction *fn, const uint8_t *code, int *offp);
JitArmResult emitGetFieldLocal(Emit *e, ObjFunction *fn, const uint8_t *code,
                               int *offp, int stop);
bool emitGetField(Emit *e, ObjFunction *fn, const uint8_t *code, int *offp,
                  int stop);
bool emitSetField(Emit *e, ObjFunction *fn, const uint8_t *code, int *offp);

/* Defined in jit_body_iter.c. */
bool emitGetIter(Emit *e, int *offp);
bool emitIterRange(Emit *e, const uint8_t *code, int *offp);
bool emitForRangeBind(Emit *e, const uint8_t *code, int *offp);
bool emitGetIterItems(Emit *e, const uint8_t *code, int *offp, int stop);
bool emitForIterBind(Emit *e, const uint8_t *code, int *offp);
bool emitForIterPair(Emit *e, const uint8_t *code, int *offp);
bool emitBuildRange(Emit *e, const uint8_t *code, int *offp, int count);

/* Defined in jit_body_index.c. */
bool emitGetIndex(Emit *e, const uint8_t *code, int *offp, int stop);
bool emitSetIndex(Emit *e, int *offp);
bool emitGetSlice(Emit *e, const uint8_t *code, int *offp);

/* Defined in jit_body_call.c. */
bool emitCall(Emit *e, ObjFunction *fn, const uint8_t *code, int *offp);
bool emitTailCall(Emit *e, ObjFunction *fn, const uint8_t *code, int *offp,
                  int count);
JitArmResult emitInvoke(Emit *e, ObjFunction *fn, ObjClosure *closure,
                        const uint8_t *code, int *offp, int count,
                        bool *afterUncondp);

/* Defined in jit_body_arith.c. */
bool emitMul(Emit *e, ObjFunction *fn, const uint8_t *code, int *offp,
             int stop);
bool emitBitOp(Emit *e, ObjFunction *fn, uint8_t op, int prevOff, int *offp);
bool emitAddSubDiv(Emit *e, ObjFunction *fn, uint8_t op, const uint8_t *code,
                   int *offp, int stop);
bool emitNegate(Emit *e, int *offp);
bool emitPow(Emit *e, int *offp);
bool emitMod(Emit *e, ObjFunction *fn, int prevOff, int *offp);
bool emitFloorDiv(Emit *e, ObjFunction *fn, int prevOff, int *offp);
JitArmResult emitNot(Emit *e, int *offp);
JitArmResult emitBitNot(Emit *e, int *offp);
JitArmResult emitWrapArith(Emit *e, uint8_t op, int *offp);
JitArmResult emitUnaryPlus(Emit *e, int *offp);

/* Defined in jit_body_fused.c. */
bool emitAddLocals(Emit *e, const uint8_t *code, int *offp);
bool emitAddBind(Emit *e, const uint8_t *code, int *offp);
bool emitCmpLocalConstLt(Emit *e, const uint8_t *code, int *offp);
bool emitAddIntConst(Emit *e, const ObjFunction *fn, const uint8_t *code, int *offp);
bool emitSubIntConst(Emit *e, const ObjFunction *fn, const uint8_t *code, int *offp);
bool emitMulIntConst(Emit *e, const uint8_t *code, int *offp);
bool emitMulBind(Emit *e, const uint8_t *code, int *offp);
bool emitSubBind(Emit *e, const uint8_t *code, int *offp);
bool emitIncLocal(Emit *e, const ObjFunction *fn, const uint8_t *code, int *offp);
/* jit_range.c: whether a comparison every path to offset `q` made proves
 * `slot + k` cannot overflow there. */
bool jitSlotAddSafe(const Emit *e, const ObjFunction *fn, uint32_t q,
                    unsigned slot, int64_t k);
/* Forget the control-flow graph jit_range.c keeps for the compile in
 * progress; called as each compile starts. */
void jitRangeReset(void);
/* jit_vector.c: a two-lane NEON run of a `for j in a..b` whose body is one
 * float store, emitted at the loop head on entry only (JAITHON_JIT_VECTOR). */
void emitVectorHead(Emit *e, ObjFunction *fn, uint32_t off);
/* jit_range.c: whether the instruction at `off` may write local `slot`. */
bool jitOpWritesSlot(const Chunk *c, int off, unsigned slot);
bool emitModIntConst(Emit *e, const uint8_t *code, int *offp);

#endif /* arm64 */

#endif /* JAI_VM_JIT_INTERNAL_H */
