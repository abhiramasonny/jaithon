/* chunk.h — bytecode chunks and the opcode enumeration. Normative reference:
 * spec/BYTECODE.md. OpCode's order is the on-disk format: append new opcodes
 * at the end and bump JAI_COMPILER_VERSION. */
#ifndef JAI_CHUNK_H
#define JAI_CHUNK_H

#include "common/common.h"
#include "vm/table.h"
#include "vm/value.h"

typedef enum {
    /* --- constants and stack (spec §3.1) --- */
    OP_NOP,
    OP_CONST,            /* u24 K */
    OP_NULL,
    OP_TRUE,
    OP_FALSE,
    OP_INT,              /* i16 */
    OP_POP,
    OP_POPN,             /* u8 */
    OP_DUP,
    OP_DUP2,
    OP_SWAP,
    OP_ROT3,

    /* --- variables (spec §3.2) --- */
    OP_GET_LOCAL,        /* u16 S */
    OP_SET_LOCAL,        /* u16 S */
    OP_GET_UPVALUE,      /* u8 U */
    OP_SET_UPVALUE,      /* u8 U */
    OP_CLOSE_UPVALUE,
    OP_GET_GLOBAL,       /* u24 K, u16 C */
    OP_SET_GLOBAL,       /* u24 K, u16 C */
    OP_DEF_GLOBAL,       /* u24 K */
    OP_GET_MODULE,       /* u24 K(module), u24 K(member) */

    /* --- arithmetic and logic (spec §3.3) --- */
    OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_FLOORDIV, OP_MOD, OP_POW,
    OP_ADD_WRAP, OP_SUB_WRAP, OP_MUL_WRAP,
    OP_NEG, OP_POS,
    OP_BAND, OP_BOR, OP_BXOR, OP_SHL, OP_SHR, OP_BNOT,
    OP_EQ, OP_NE, OP_LT, OP_LE, OP_GT, OP_GE,
    OP_IS, OP_IS_NOT, OP_IN, OP_NOT_IN,
    OP_NOT,
    OP_CONCAT,

    /* fused forms emitted by the peephole pass */
    OP_ADD_INT_CONST,        /* u16 S, i16 */
    OP_INC_LOCAL,            /* u16 S, i8  */
    OP_CMP_LOCAL_CONST_LT,   /* u16 S, i16 */
    OP_GET_LOCAL2,           /* u16 S, u16 S */
    OP_ADD_LOCALS,           /* u16 S, u16 S */

    /* --- control flow (spec §3.4) --- */
    OP_JUMP,                 /* i16 J */
    OP_JUMP_IF_FALSE,        /* i16 J */
    OP_JUMP_IF_TRUE,         /* i16 J */
    OP_JUMP_IF_FALSE_KEEP,   /* i16 J */
    OP_JUMP_IF_TRUE_KEEP,    /* i16 J */
    OP_JUMP_IF_NULL,         /* i16 J */
    OP_LOOP,                 /* i16 J (backward) */
    OP_GET_ITER,
    OP_FOR_ITER,             /* i16 J */

    /* --- calls (spec §3.5) --- */
    OP_CALL,                 /* u8 A */
    OP_CALL_KW,              /* u8 A, u24 K */
    OP_CALL_SPREAD,          /* u8 A */
    OP_INVOKE,               /* u24 K, u8 A, u16 C */
    OP_SUPER_INVOKE,         /* u24 K, u8 A */
    OP_TAIL_CALL,            /* u8 A */
    OP_RETURN,
    OP_RETURN_NULL,
    OP_CLOSURE,              /* u24 K, then upvalueCount * (u8 isLocal, u16 index) */

    /* --- data structures (spec §3.6) --- */
    OP_BUILD_LIST,           /* u16 */
    OP_BUILD_DICT,           /* u16 */
    OP_BUILD_SET,            /* u16 */
    OP_BUILD_TUPLE,          /* u16 */
    OP_BUILD_RANGE,          /* u8 inclusive */
    OP_LIST_APPEND,          /* u16 depth */
    OP_DICT_INSERT,          /* u16 depth */
    OP_SET_ADD,              /* u16 depth */
    OP_GET_INDEX,
    OP_SET_INDEX,
    OP_GET_SLICE,            /* u8 flags */
    OP_SET_SLICE,            /* u8 flags */
    OP_UNPACK,               /* u8 count, u8 restIndex (255 = no rest) */

    /* --- objects and classes (spec §3.7) --- */
    OP_CLASS,                /* u24 K */
    OP_INHERIT,
    OP_IMPL_TRAIT,           /* u24 K */
    OP_METHOD,               /* u24 K, u8 fnFlags, u8 visibility */
    OP_FIELD_DEF,            /* u24 K, u8: 0-1 visibility, 2 static, 3 let */
    OP_GET_FIELD,            /* u24 K, u16 C */
    OP_SET_FIELD,            /* u24 K, u16 C */
    OP_GET_SUPER,            /* u24 K */
    OP_NEW,                  /* u24 K, u8 A */
    OP_ENUM_NEW,             /* u24 K, u8 tag, u8 A */
    OP_ENUM_TAG,
    OP_ENUM_FIELD,           /* u8 */
    OP_IS_INSTANCE,          /* u24 K */

    /* --- exceptions and defer (spec §3.8) --- */
    OP_THROW,
    OP_RERAISE,
    OP_PUSH_HANDLER,         /* i16 J, u24 K */
    OP_POP_HANDLER,
    OP_PUSH_FINALLY,         /* i16 J */
    OP_END_FINALLY,
    OP_PUSH_DEFER,           /* u24 K */
    OP_RUN_DEFERS,
    OP_MATCH_EXC,            /* u24 K */
    OP_GET_EXC,

    /* --- pattern matching (spec §3.9) --- */
    OP_MATCH_CONST,          /* u24 K, i16 J */
    OP_MATCH_RANGE,          /* u24 K, u24 K, u8 inclusive, i16 J */
    OP_MATCH_TYPE,           /* u24 K, i16 J */
    OP_MATCH_SEQ,            /* u8 count, u8 restFlag, i16 J */
    OP_MATCH_FIELDS,         /* u24 K, i16 J */
    OP_BIND,                 /* u16 S */

    /* --- modules and misc (spec §3.10) --- */
    OP_IMPORT,               /* u24 K */
    OP_IMPORT_FROM,          /* u24 K */
    OP_EXPORT,               /* u24 K */
    OP_ASSERT_FAIL,          /* u24 K */
    OP_TYPE_GUARD,           /* u24 K */
    OP_HALT,

    /* --- fused forms appended after compiler version 13 (spec §3.3) ---
     * Appended here, not beside the older fused forms, because the
     * enumeration order is the on-disk encoding: inserting one in the middle
     * renumbers every opcode above it and silently misreads every cached
     * .jaic. Each replaces a pair measured executing millions of times per
     * benchmark run. */
    OP_GET_FIELD_LOCAL,      /* u16 S, u24 K, u16 C */
    OP_FOR_ITER_BIND,        /* i16 J, u16 S */
    OP_JUMP_IF_CMP_FALSE,    /* u8 cmp (an OP_EQ..OP_GE byte), i16 J */

    /* --- string interpolation (spec §3.6) --- */
    OP_FORMAT,               /* u8 N, u24 litmask, u24 K("str"), u16 C */

    /* --- fused form appended after compiler version 15 (spec §3.3) ---
     * `while i < N` / `if n < 2` -- a loop guard and a base case, the two most
     * frequent conditionals -- both lower to push-local/push-const/compare-
     * branch; this is all three. The constant is a pool index rather than an
     * immediate so a guard against a large literal fuses too. */
    OP_JUMP_IF_CMP_LOCAL_K,  /* u8 cmp (an OP_EQ..OP_GE byte), u16 S, u24 K, i16 J */

    /* --- numeric widening, appended after compiler version 17 (spec §3.3) ---
     * int-on-stack -> float. The checker emits it wherever §2.2's widening
     * applies and both types are known, so typed code pays a conversion, not
     * the type test the `any` path in arithmetic opcodes pays. */
    OP_TO_FLOAT,

    /* `<int k>; MOD` fused (§3.3): pops the dividend, pushes `x % k`. An i16
     * immediate modulus, so the pair collapses wherever it appears. */
    OP_MOD_INT_CONST,        /* i16 */

    /* `ADD; BIND a` fused (§3.3): pops both operands, adds, stores into slot a.
     * Net stack effect matches the pair, so operand provenance doesn't matter. */
    OP_ADD_BIND,             /* u16 S */
    OP_SUB_BIND,             /* u16 S */
    OP_MUL_BIND,             /* u16 S */

    /* Stamp a declared element type onto the container on top of the stack
     * (spec §3.6). `list[int]` and `dict[str, int]` are otherwise compile-time
     * facts only, and a container reaching a function through `any` can be
     * mutated with no diagnostic at all -- the checker cannot see it, because
     * with an `any` receiver it does not know what the container holds. Two
     * FieldKind nibbles: element/value in the low four bits, dict key in the
     * high four. Emitted only where a type was actually written. */
    OP_ELEM_KIND,            /* u8: low nibble value kind, high nibble key kind */

    /* `for … in X.items()`: replace X on the stack with an iterator over its
     * pairs (spec §3.4). On a dict this is ITER_DICT_ITEMS, which walks the
     * table and yields one tuple at a time; on anything else it calls the
     * receiver's own `items()` and iterates the result, so a user class that
     * defines `items()` behaves exactly as before.
     *
     * The dispatch is at RUNTIME on purpose: emit.jai has no checker types
     * (see its note), so the emitter matches only the syntactic shape and this
     * opcode decides what the receiver actually is.
     *
     * It does NOT call `items()` itself -- invoking a Jaithon method from
     * inside an opcode is not safe here, and --gc-stress caught it. It only
     * converts a dict and jumps past the ordinary sequence; anything else is
     * left on the stack for that sequence to handle:
     *
     *     <receiver>
     *     GET_ITER_ITEMS J   ; dict: replace with a lazy iterator, jump to J
     *     INVOKE items 0     ; anything else: its own items(), unchanged
     *     GET_ITER
     *   J:
     */
    OP_GET_ITER_ITEMS,       /* i16 J */

    /* `for (a, b) in …` fused (spec §3.3): the whole of
     *
     *     FOR_ITER J; UNPACK 2 255; BIND A; BIND B
     *
     * in one instruction, and — this is the point of it — with no pair object
     * in between. The unfused sequence makes the iterator materialise a pair,
     * pushes it, takes it apart and drops it again, so a dict walked by
     * `for (k, v) in d.items()` allocated one 2-tuple per entry per pass:
     * 4.8M tuples and 306 MB freed on tests/bench/dict_iter, where the
     * collector was 30% of the run.
     *
     * Deliberately fused from the BYTECODE and not emitted from the pattern:
     * every `for (a, b) in <anything>` lowers to exactly the four instructions
     * above, whatever the receiver is, so matching them covers dicts, lists of
     * pairs, `enumerate` and a user iterator alike with no type knowledge at
     * the emitter — which has none to offer (see emit.jai's note at :2853).
     *
     * A dict-items iterator yields its key and value straight into the two
     * slots. Anything else produces its item as usual and the item is split in
     * place; an item that is not a 2-element list or tuple raises exactly what
     * OP_UNPACK raised for it. */
    OP_FOR_ITER_PAIR,        /* i16 J, u16 A, u16 B */

    /* `for x in a..b` with neither object built (spec §3.4). The pair below
     * replaces the whole of
     *
     *     <a>; <b>; BUILD_RANGE incl; GET_ITER
     *   L: FOR_ITER_BIND J, S
     *
     * which allocated an ObjRange (48B) AND an ObjIter (64B) on every LOOP
     * ENTRY — not per program, per entry. An allocation census put that at
     * 99.95% of everything tests/bench/nbody allocates (4.5M objects, 243 MB,
     * 114 collections) for a benchmark whose own header says it measures float
     * math, 98.5% of matrix_mul's and 88.6% of graph_bfs's.
     *
     * Nothing is on the heap and nothing is on the operand stack: the loop's
     * whole state is two int frame slots, which the register allocator can
     * treat like any other int local and a deopt writes back like any other.
     *
     *   ITER_RANGE incl, C, E   pops the two bounds, seeds slots[C] with the
     *                           first value and slots[E] with the value one
     *                           past the last
     *   FOR_RANGE_BIND J,S,C,E  slots[C] == slots[E] ends the loop; otherwise
     *                           bind slots[S] = slots[C] and step slots[C]
     *
     * `E` is an END, not a count, so the test is one compare and the step one
     * add — the same shape the compiled range loop already ran on. It is
     * computed WRAPPING, which is what makes `a..=INT64_MAX` terminate: the
     * counter wraps to INT64_MIN, which is exactly what E holds. The one range
     * that does not survive the round trip is the full-width inclusive
     * `INT64_MIN..=INT64_MAX`, which iterates 2^64 times here and INT64_MAX
     * times through ObjIter's saturating `limit`. Both tiers agree, and
     * neither terminates this side of the heat death of the universe.
     *
     * Emitted from the loop, not fused from bytecode, because the operand
     * stack shrinks: `break` unwinds to a depth the emitter alone knows. The
     * shape it matches is the syntactic one (`for <name> in <range>`), which
     * needs no type knowledge — a range literal is a range literal. Every
     * other iterable, and a range that reaches the loop through a variable,
     * still lowers to BUILD_RANGE/GET_ITER/FOR_ITER_BIND unchanged. */
    OP_ITER_RANGE,           /* u8 inclusive, u16 C, u16 E */
    OP_FOR_RANGE_BIND,       /* i16 J, u16 S, u16 C, u16 E */

    /* `GET_LOCAL S; <int k>; SUB` fused (§3.3). Kept distinct from
     * ADD_INT_CONST so overflow diagnostics and user-defined subtraction keep
     * naming `-`, rather than being rewritten as addition by a negative. */
    OP_SUB_INT_CONST,        /* u16 S, i16 */

    /* `POP; RETURN_NULL` fused (§3.3): a discarded expression-statement right
     * before an implicit void return. RETURN_NULL contributes no stack effect
     * of its own (it never touches this frame's operand stack), so the fused
     * effect is exactly POP's. */
    OP_POP_RETURN_NULL,

    /* `MATCH_CONST; POP` fused (§3.3): a literal match-arm's own success path
     * (subject equals the arm's constant), emitted directly by the match
     * compiler rather than synthesised by the peephole -- the pattern is
     * always exactly this shape, not something that only sometimes lands
     * adjacent. The no-match path is unchanged: MATCH_CONST's own jump,
     * still peeking, still leaving the subject for the next arm's test. */
    OP_MATCH_CONST_POP,      /* u24 K, i16 J */

    OP_COUNT
} OpCode;

/* Length in bytes of the operands following `op` (not counting the opcode
 * byte). OP_CLOSURE is variable-length; callers must special-case it. */
int  jaiOpOperandSize(OpCode op);
const char *jaiOpName(OpCode op);
/* Net stack effect, or INT32_MIN when it depends on operands. */
int  jaiOpStackEffect(OpCode op);
/* Byte offset of the u16 inline-cache operand inside `op`'s operand run, or -1
 * when it has none. One definition, not two (verifier + .jaic reader): the
 * old split let OP_GET_FIELD_LOCAL be added to one and not the other,
 * silently under-sizing every deserialised chunk's cache array. */
int  jaiOpCacheOperand(OpCode op);

/* ------------------------------------------------------------------ */
/* Line tables (spec §5): source *spans*, not line numbers, so a diagnostic
 * lands a caret under one expression rather than pointing at a line.
 *
 * Stored as LTV1 -- a delta+LEB128 byte stream, not an array of records. Three
 * absolute u32s per entry was 55.5% of every .jaic and 54.4% of boot/seed.bin
 * to carry values whose deltas encode in about three bytes; LTV1 measures
 * ~3.1 bytes per entry against 12.
 *
 * The stream is also the RUNTIME form: it is never expanded. jaiChunkSpanAt
 * decodes it forward, which costs O(entries) instead of a binary search on a
 * path with exactly one runtime caller -- frameSpan in vm.c, reached only when
 * a diagnostic or traceback is actually printed. Expanding at load would give
 * back the file-size win and none of the memory or load-time win.
 *
 * Per entry, deltas against the previous one:
 *     uleb128  offset - prevOffset        (offsets are non-decreasing)
 *     sleb128  span   - prevSpan
 *     sleb128  spanEnd - span             (against its OWN span: a span's
 *                                          length is small and local)
 * The first entry encodes against zero. */
/* ------------------------------------------------------------------ */

typedef struct {
    uint32_t offset;   /* first code offset covered */
    uint32_t span;     /* source span start (byte offset in the source file) */
    uint32_t spanEnd;
} LineEntry;

/* LTV1 codec, shared with the serialiser and with tests/vm/linetable_ltv1.c.
 * jaiLtv1Encode appends one entry to `buf`; jaiLtv1Decode walks the stream and
 * returns the entry covering `codeOffset`. */
size_t jaiLtv1EncodeEntry(uint8_t *out, size_t cap, const LineEntry *prev,
                          const LineEntry *e);
bool   jaiLtv1Lookup(const uint8_t *stream, size_t len, uint32_t codeOffset,
                     uint32_t *span, uint32_t *spanEnd);
int    jaiLtv1Count(const uint8_t *stream, size_t len);

/* ------------------------------------------------------------------ */
/* Inline caches: allocated per chunk, indexed by a u16 operand baked into  */
/* the instruction, so a lookup is one array index with no hashing. The    */
/* code generator hands out slots with jaiChunkAddCache.                   */
/* ------------------------------------------------------------------ */

typedef enum { IC_EMPTY = 0, IC_MONO, IC_POLY, IC_MEGA } ICState;

#define JAI_IC_WAYS 4

/* What an OP_INVOKE site was observed to RETURN, one byte per way. Only ever a
 * PREDICTION -- the tag guard emitted alongside it is what makes it sound, so a
 * method returning something else deoptimises rather than miscompiles.
 * Encoding: 0 never observed, 1+ValueType for a non-object, JAI_FB_OBJ+ObjType
 * for an object, 255 mixed. Sits in `shapeId`'s former alignment padding, so
 * sizeof(InlineCache) is unchanged.
 *
 * Per WAY, so one site with three receiver classes keeps three answers: merging
 * them would say MIXED for a site that is perfectly predictable once the
 * receiver's class is known.
 *
 * The record is MERGED over a window rather than taken from one call. A first
 * observation is what this used to be, and roadmap §6 records the consequence:
 * a site whose result kind later changes then deoptimises every iteration, and
 * a wrong prediction at a polymorphic site measured 5.7x slower than declining.
 * A merge that disagrees goes to JAI_FB_MIXED, which no consumer can turn into
 * a slot kind -- "unstable, predict nothing" is the answer, and it is a better
 * one than a lucky guess. */
#define JAI_FB_NONE   0u
#define JAI_FB_OBJ    32u
#define JAI_FB_MIXED  255u

/* How many INVOKEs one site observes before it stops recording.
 *
 * The window has to close: recording forever is a store on the hit path, which
 * measured 2.4% of dict_ops when it was unconditional. Once the budget is spent
 * the steady state is one already-hot load and a not-taken branch, at the
 * invoke and at the return.
 *
 * 64 is JAI_JIT_THRESHOLD -- the point at which the tier first looks at a
 * function -- so a site inside a loop is settled well before anything reads it,
 * and a site reached once per call is settled on the entry that compiles.
 * chunk.h cannot include jit.h, so this is checked against it in jit.h. */
#define JAI_IC_OBS_BUDGET 64u

JAI_INLINE uint8_t jaiFeedbackKind(Value v) {
    return IS_OBJ(v) ? (uint8_t)(JAI_FB_OBJ + (unsigned)OBJ_TYPE(v))
                     : (uint8_t)(1u + (unsigned)jaiValueType(v));
}

JAI_INLINE uint8_t jaiFeedbackMerge(uint8_t prev, uint8_t seen) {
    if (prev == JAI_FB_NONE) return seen;
    return prev == seen ? prev : (uint8_t)JAI_FB_MIXED;
}

typedef struct {
    uint8_t  state;
    uint8_t  count;
    uint8_t  resultKind[JAI_IC_WAYS];  /* see above; in former padding */
    uint8_t  obsBudget;   /* invokes left to observe; 0 = frozen. Also padding. */
    uint8_t  obsPad;      /* the last byte before shapeId's alignment */
    uint32_t shapeId[JAI_IC_WAYS];   /* ObjClass.shapeId or ObjModule.version */
    uint32_t payload[JAI_IC_WAYS];   /* field slot or global table index */
    Value    cached[JAI_IC_WAYS];    /* bound method for INVOKE sites */
} InlineCache;

/* The two observation bytes were meant to be free. If a field is ever added
 * that pushes shapeId past offset 8, every chunk in the program pays for it and
 * nothing else says so. */
_Static_assert(offsetof(InlineCache, shapeId) == 8,
               "InlineCache observation bytes must fit shapeId's padding");

typedef struct {
    uint8_t   *code;
    int        count;
    int        capacity;

    ValueArray constants;

    /* LTV1 stream (see above), appended to as code is emitted and decoded in
     * place when a diagnostic asks. `lineLast` is the encoder's running state:
     * the deltas are against it, and it is also what collapses a run of
     * instructions sharing one span into a single entry. */
    uint8_t   *lineStream;
    int        lineStreamLen;
    int        lineStreamCap;
    LineEntry  lineLast;
    bool       lineHasLast;

    InlineCache *caches;
    int          cacheCount;
    int          cacheCapacity;

    /* Constant-pool dedup index: hash of a constant -> its pool index. Built
     * lazily once the pool outgrows a linear scan, so NULL means "not built
     * yet". Not serialised (derived from `constants`) and not GC-traced
     * (every key/value is an int). */
    JaiTable  *constIndex;

    int        sourceFileId;
} Chunk;

void jaiChunkInit(Chunk *chunk, int sourceFileId);
/* Reserve a fresh inline-cache slot; returns its index (fits in u16). */
uint16_t jaiChunkAddCache(Chunk *chunk);
void jaiChunkFree(Chunk *chunk);

/* Allocate exactly `count` cache slots in one go, initialised as
 * jaiChunkAddCache initialises them. The deserialiser knows the count before it
 * allocates, and jaiChunkAddCache's JAI_GROW_CAP rounding left 1.19 MB of slack
 * across the seed's images. Refuses a chunk that already has caches: sizing one
 * array twice is a bug, not a resize. */
bool jaiChunkReserveCaches(Chunk *chunk, int count);

void jaiChunkWrite(Chunk *chunk, uint8_t byte, uint32_t spanStart, uint32_t spanEnd);
void jaiChunkWriteU16(Chunk *chunk, uint16_t v, uint32_t s, uint32_t e);
void jaiChunkWriteU24(Chunk *chunk, uint32_t v, uint32_t s, uint32_t e);
void jaiChunkWriteI16(Chunk *chunk, int16_t v, uint32_t s, uint32_t e);
void jaiChunkPatchU16(Chunk *chunk, int offset, uint16_t v);
void jaiChunkPatchI16(Chunk *chunk, int offset, int16_t v);

/* Appends `v` and returns its index, deduplicating equal constants. */
uint32_t jaiChunkAddConstant(Chunk *chunk, Value v);
/* Source span covering the instruction at `codeOffset`. */
void     jaiChunkSpanAt(const Chunk *chunk, int codeOffset, uint32_t *start,
                        uint32_t *end);

/* Operands are little-endian by construction. Read as one unaligned load
 * (both arm64 and x86-64 permit it) instead of byte-assembled, this costs one
 * instruction for a u16 and two for a u24, against four and six. */
#if defined(__LITTLE_ENDIAN__) ||                                              \
    (defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
#  define JAI_OPERANDS_ARE_NATIVE 1
#else
#  define JAI_OPERANDS_ARE_NATIVE 0
#endif

JAI_INLINE uint16_t jaiReadU16(const uint8_t *p) {
#if JAI_OPERANDS_ARE_NATIVE
    uint16_t v;
    __builtin_memcpy(&v, p, sizeof v);
    return v;
#else
    return (uint16_t)(p[0] | ((uint16_t)p[1] << 8));
#endif
}
JAI_INLINE uint32_t jaiReadU24(const uint8_t *p) {
    return (uint32_t)jaiReadU16(p) | ((uint32_t)p[2] << 16);
}
JAI_INLINE int16_t jaiReadI16(const uint8_t *p) {
    return (int16_t)jaiReadU16(p);
}

/* ------------------------------------------------------------------ */
/* Disassembly                                                          */
/* ------------------------------------------------------------------ */

void jaiDisassembleChunk(FILE *out, const Chunk *chunk, const char *name);
int  jaiDisassembleInstruction(FILE *out, const Chunk *chunk, int offset);

#endif /* JAI_CHUNK_H */
