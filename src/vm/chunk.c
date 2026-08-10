/* chunk.c — bytecode chunks, the opcode metadata tables, and the disassembler.
 *
 * Normative reference: spec/BYTECODE.md §2 (operand encoding), §3 (opcode
 * table). The single source of truth for opcode metadata is JAI_OPCODES below;
 * jaiOpName, jaiOpOperandSize and jaiOpStackEffect are all generated from it so
 * they cannot drift apart, and a static assert ties the row count to OP_COUNT.
 */
#include "chunk.h"

#include "../common/diag.h"
#include "object.h"
#include "table.h"

/* ------------------------------------------------------------------ */
/* Opcode metadata                                                      */
/* ------------------------------------------------------------------ */

/* Stack effect of an instruction whose net change depends on its operands
 * (argument counts, element counts) or on which way it branches. */
#define SE_VAR INT32_MIN

/* OP_CLOSURE is the one variable-length instruction: its operand run is a u24
 * constant index followed by upvalueCount * (u8 isLocal, u16 index), and
 * upvalueCount lives in the referenced ObjFunction. -1 is returned so that a
 * caller that forgets to special-case it fails loudly (a zero-length
 * instruction hangs its decode loop) instead of silently decoding the upvalue
 * bytes as opcodes. */
#define OPERANDS_VARIABLE (-1)

/* X(opcode, operandBytes, stackEffect) — order must match enum OpCode. */
#define JAI_OPCODES(X)                                                         \
    /* constants and stack (§3.1) */                                           \
    X(OP_NOP,                  0,  0)                                          \
    X(OP_CONST,                3, +1)                                          \
    X(OP_NULL,                 0, +1)                                          \
    X(OP_TRUE,                 0, +1)                                          \
    X(OP_FALSE,                0, +1)                                          \
    X(OP_INT,                  2, +1)                                          \
    X(OP_POP,                  0, -1)                                          \
    X(OP_POPN,                 1, SE_VAR)                                      \
    X(OP_DUP,                  0, +1)                                          \
    X(OP_DUP2,                 0, +2)                                          \
    X(OP_SWAP,                 0,  0)                                          \
    X(OP_ROT3,                 0,  0)                                          \
    /* variables (§3.2) */                                                     \
    X(OP_GET_LOCAL,            2, +1)                                          \
    X(OP_SET_LOCAL,            2,  0)                                          \
    X(OP_GET_UPVALUE,          1, +1)                                          \
    X(OP_SET_UPVALUE,          1,  0)                                          \
    X(OP_CLOSE_UPVALUE,        2,  0)                                          \
    X(OP_GET_GLOBAL,           5, +1)                                          \
    X(OP_SET_GLOBAL,           5,  0)                                          \
    X(OP_DEF_GLOBAL,           3, -1)                                          \
    X(OP_GET_MODULE,           6, +1)                                          \
    /* arithmetic and logic (§3.3) */                                          \
    X(OP_ADD,                  0, -1)                                          \
    X(OP_SUB,                  0, -1)                                          \
    X(OP_MUL,                  0, -1)                                          \
    X(OP_DIV,                  0, -1)                                          \
    X(OP_FLOORDIV,             0, -1)                                          \
    X(OP_MOD,                  0, -1)                                          \
    X(OP_POW,                  0, -1)                                          \
    X(OP_ADD_WRAP,             0, -1)                                          \
    X(OP_SUB_WRAP,             0, -1)                                          \
    X(OP_MUL_WRAP,             0, -1)                                          \
    X(OP_NEG,                  0,  0)                                          \
    X(OP_POS,                  0,  0)                                          \
    X(OP_BAND,                 0, -1)                                          \
    X(OP_BOR,                  0, -1)                                          \
    X(OP_BXOR,                 0, -1)                                          \
    X(OP_SHL,                  0, -1)                                          \
    X(OP_SHR,                  0, -1)                                          \
    X(OP_BNOT,                 0,  0)                                          \
    X(OP_EQ,                   0, -1)                                          \
    X(OP_NE,                   0, -1)                                          \
    X(OP_LT,                   0, -1)                                          \
    X(OP_LE,                   0, -1)                                          \
    X(OP_GT,                   0, -1)                                          \
    X(OP_GE,                   0, -1)                                          \
    X(OP_IS,                   0, -1)                                          \
    X(OP_IS_NOT,               0, -1)                                          \
    X(OP_IN,                   0, -1)                                          \
    X(OP_NOT_IN,               0, -1)                                          \
    X(OP_NOT,                  0,  0)                                          \
    X(OP_CONCAT,               0, -1)                                          \
    /* peephole-fused forms */                                                 \
    X(OP_ADD_INT_CONST,        4, +1)                                          \
    X(OP_INC_LOCAL,            3,  0)                                          \
    X(OP_CMP_LOCAL_CONST_LT,   4, +1)                                          \
    X(OP_GET_LOCAL2,           4, +2)                                          \
    X(OP_ADD_LOCALS,           4, +1)                                          \
    /* control flow (§3.4) */                                                  \
    X(OP_JUMP,                 2,  0)                                          \
    X(OP_JUMP_IF_FALSE,        2, -1)                                          \
    X(OP_JUMP_IF_TRUE,         2, -1)                                          \
    X(OP_JUMP_IF_FALSE_KEEP,   2,  0)                                          \
    X(OP_JUMP_IF_TRUE_KEEP,    2,  0)                                          \
    X(OP_JUMP_IF_NULL,         2,  0)                                          \
    X(OP_LOOP,                 2,  0)                                          \
    X(OP_GET_ITER,             0,  0)                                          \
    X(OP_FOR_ITER,             2, SE_VAR)                                      \
    /* calls (§3.5) */                                                         \
    X(OP_CALL,                 1, SE_VAR)                                      \
    X(OP_CALL_KW,              4, SE_VAR)                                      \
    X(OP_CALL_SPREAD,          1, SE_VAR)                                      \
    X(OP_INVOKE,               6, SE_VAR)                                      \
    X(OP_SUPER_INVOKE,         4, SE_VAR)                                      \
    X(OP_TAIL_CALL,            1, SE_VAR)                                      \
    X(OP_RETURN,               0, -1)                                          \
    X(OP_RETURN_NULL,          0,  0)                                          \
    X(OP_CLOSURE,   OPERANDS_VARIABLE, +1)                                     \
    /* data structures (§3.6) */                                               \
    X(OP_BUILD_LIST,           2, SE_VAR)                                      \
    X(OP_BUILD_DICT,           2, SE_VAR)                                      \
    X(OP_BUILD_SET,            2, SE_VAR)                                      \
    X(OP_BUILD_TUPLE,          2, SE_VAR)                                      \
    X(OP_BUILD_RANGE,          1, -1)                                          \
    X(OP_LIST_APPEND,          2, -1)                                          \
    X(OP_DICT_INSERT,          2, -2)                                          \
    X(OP_SET_ADD,              2, -1)                                          \
    X(OP_GET_INDEX,            0, -1)                                          \
    X(OP_SET_INDEX,            0, -3)                                          \
    X(OP_GET_SLICE,            1, SE_VAR)                                      \
    X(OP_SET_SLICE,            1, SE_VAR)                                      \
    X(OP_UNPACK,               2, SE_VAR)                                      \
    /* objects, classes, traits (§3.7) */                                      \
    X(OP_CLASS,                3, +1)                                          \
    X(OP_INHERIT,              0,  0)                                          \
    X(OP_IMPL_TRAIT,           3,  0)                                          \
    X(OP_METHOD,               5, -1)                                          \
    X(OP_FIELD_DEF,            4,  0)                                          \
    X(OP_GET_FIELD,            5,  0)                                          \
    X(OP_SET_FIELD,            5, -2)                                          \
    X(OP_GET_SUPER,            3,  0)                                          \
    X(OP_NEW,                  4, SE_VAR)                                      \
    X(OP_ENUM_NEW,             5, SE_VAR)                                      \
    X(OP_ENUM_TAG,             0, +1)                                          \
    X(OP_ENUM_FIELD,           1, +1)                                          \
    X(OP_IS_INSTANCE,          3,  0)                                          \
    /* exceptions and defer (§3.8) */                                          \
    X(OP_THROW,                0, -1)                                          \
    X(OP_RERAISE,              0,  0)                                          \
    X(OP_PUSH_HANDLER,         5,  0)                                          \
    X(OP_POP_HANDLER,          0,  0)                                          \
    X(OP_PUSH_FINALLY,         2,  0)                                          \
    X(OP_END_FINALLY,          0,  0)                                          \
    X(OP_PUSH_DEFER,           3,  0)                                          \
    X(OP_RUN_DEFERS,           0,  0)                                          \
    X(OP_MATCH_EXC,            3, +1)                                          \
    X(OP_GET_EXC,              0, +1)                                          \
    /* pattern matching (§3.9) */                                              \
    X(OP_MATCH_CONST,          5,  0)                                          \
    X(OP_MATCH_RANGE,          9,  0)                                          \
    X(OP_MATCH_TYPE,           5,  0)                                          \
    X(OP_MATCH_SEQ,            4,  0)                                          \
    X(OP_MATCH_FIELDS,         5, SE_VAR)                                      \
    X(OP_BIND,                 2, -1)                                          \
    /* modules and misc (§3.10) */                                             \
    X(OP_IMPORT,               3, +1)                                          \
    X(OP_IMPORT_FROM,          3, +1)                                          \
    X(OP_EXPORT,               3,  0)                                          \
    X(OP_ASSERT_FAIL,          3,  0)                                          \
    X(OP_TYPE_GUARD,           3,  0)                                          \
    X(OP_HALT,                 0,  0)                                          \
    /* peephole-fused forms appended after compiler version 13 (§3.3) */       \
    X(OP_GET_FIELD_LOCAL,      7, +1)                                          \
    X(OP_FOR_ITER_BIND,        4,  0)                                          \
    X(OP_JUMP_IF_CMP_FALSE,    3, -2)                                          \
    /* string interpolation (§3.6) */                                          \
    X(OP_FORMAT,               9, SE_VAR)                                      \
    /* peephole-fused form appended after compiler version 15 (§3.3) */        \
    X(OP_JUMP_IF_CMP_LOCAL_K,  8,  0)                                          \
    /* numeric widening appended after compiler version 17 (§3.3) */           \
    X(OP_TO_FLOAT,             0,  0)                                          \
    /* `<int k>; MOD` fused, appended after compiler version 18 (§3.3) */      \
    X(OP_MOD_INT_CONST,        2,  0)                                          \
    X(OP_ADD_BIND,             2, -2)                                          \
    X(OP_SUB_BIND,             2, -2)                                          \
    X(OP_MUL_BIND,             2, -2)

#define X_NAME(op, operands, effect)     #op,
#define X_OPERANDS(op, operands, effect) (int8_t)(operands),
#define X_EFFECT(op, operands, effect)   (int32_t)(effect),

static const char *const kOpNames[]  = { JAI_OPCODES(X_NAME) };
static const int8_t      kOpOperands[] = { JAI_OPCODES(X_OPERANDS) };
static const int32_t     kOpEffects[]  = { JAI_OPCODES(X_EFFECT) };

#undef X_NAME
#undef X_OPERANDS
#undef X_EFFECT

_Static_assert(sizeof(kOpNames) / sizeof(kOpNames[0]) == OP_COUNT,
               "opcode name table is out of sync with enum OpCode");
_Static_assert(sizeof(kOpOperands) / sizeof(kOpOperands[0]) == OP_COUNT,
               "operand size table is out of sync with enum OpCode");
_Static_assert(sizeof(kOpEffects) / sizeof(kOpEffects[0]) == OP_COUNT,
               "stack effect table is out of sync with enum OpCode");

/* Bytes of the OP_CLOSURE header (the u24 constant index) and of one trailing
 * upvalue descriptor (u8 isLocal, u16 index). */
#define CLOSURE_HEADER_BYTES 3
#define CLOSURE_UPVALUE_BYTES 3

static bool opInRange(OpCode op) { return (unsigned)op < (unsigned)OP_COUNT; }

const char *jaiOpName(OpCode op) {
    return opInRange(op) ? kOpNames[op] : "OP_UNKNOWN";
}

int jaiOpOperandSize(OpCode op) {
    return opInRange(op) ? kOpOperands[op] : 0;
}

int jaiOpStackEffect(OpCode op) {
    return opInRange(op) ? kOpEffects[op] : SE_VAR;
}

int jaiOpCacheOperand(OpCode op) {
    switch (op) {
    case OP_GET_GLOBAL:
    case OP_SET_GLOBAL:
    case OP_GET_FIELD:
    case OP_SET_FIELD:
        return 3;
    case OP_INVOKE:
        return 4;
    case OP_GET_FIELD_LOCAL:
        return 5;   /* after the u16 slot and the u24 name */
    case OP_FORMAT:
        return 7;   /* after the u8 count, the u24 mask and the u24 name */
    default:
        return -1;
    }
}

/* ------------------------------------------------------------------ */
/* Constant deduplication                                              */
/*                                                                      */
/* Small pools are scanned linearly. Past CONST_LINEAR_MAX the chunk's   */
/* Chunk.constIndex table is built, mapping a 64-bit hash of a constant  */
/* to its pool index.                                                    */
/*                                                                      */
/* The table is keyed by hash rather than by the Value itself because    */
/* jaiValueHash/jaiValuesEqual compare numerically: they would collapse  */
/* INT_VAL(1) and FLOAT_VAL(1.0) into one constant, and int and float    */
/* are distinct types (spec §2.2). Every hash hit is therefore verified  */
/* with constEqual, which is exact; a hash collision costs one duplicate */
/* pool entry and never a wrong one.                                     */
/* ------------------------------------------------------------------ */

#define CONST_LINEAR_MAX 64
#define CONST_MAX_COUNT  (1u << 24)   /* u24 operand */
#define CONST_MAX_DEPTH  8            /* nested tuple recursion bound */

static bool constDedupable(Value v) {
    switch (v.type) {
    case VAL_NULL:
    case VAL_BOOL:
    case VAL_INT:
    case VAL_FLOAT:
        return true;
    case VAL_OBJ:
        if (AS_OBJ(v) == NULL) return false;
        return OBJ_TYPE(v) == OBJ_STRING || OBJ_TYPE(v) == OBJ_BYTES ||
               OBJ_TYPE(v) == OBJ_TUPLE;
    }
    return false;
}

/* Exact structural equality for constant-pool entries. Deliberately not
 * jaiValuesEqual: this must never conflate int with float, must keep -0.0
 * distinct from 0.0, and must never run user code (__eq__). */
static bool constEqual(Value a, Value b, int depth) {
    if (a.type != b.type) return false;
    switch (a.type) {
    case VAL_NULL:  return true;
    case VAL_BOOL:  return AS_BOOL(a) == AS_BOOL(b);
    case VAL_INT:   return AS_INT(a) == AS_INT(b);
    case VAL_FLOAT: {
        double x = AS_FLOAT(a), y = AS_FLOAT(b);
        return memcmp(&x, &y, sizeof x) == 0;   /* bitwise: -0.0 != 0.0 */
    }
    case VAL_OBJ:   break;
    }

    Obj *x = AS_OBJ(a), *y = AS_OBJ(b);
    if (x == y) return true;
    if (x == NULL || y == NULL || x->type != y->type) return false;

    switch (x->type) {
    case OBJ_STRING: {
        const ObjString *sa = AS_STRING(a), *sb = AS_STRING(b);
        return sa->length == sb->length && sa->hash == sb->hash &&
               memcmp(sa->chars, sb->chars, sa->length) == 0;
    }
    case OBJ_BYTES: {
        const ObjBytes *ba = AS_BYTES(a), *bb = AS_BYTES(b);
        return ba->length == bb->length &&
               memcmp(ba->data, bb->data, ba->length) == 0;
    }
    case OBJ_TUPLE: {
        if (depth >= CONST_MAX_DEPTH) return false;   /* bound the recursion */
        const ObjTuple *ta = AS_TUPLE(a), *tb = AS_TUPLE(b);
        if (ta->count != tb->count) return false;
        for (uint32_t i = 0; i < ta->count; i++) {
            if (!constEqual(ta->items[i], tb->items[i], depth + 1)) return false;
        }
        return true;
    }
    default:
        return false;   /* functions, class specs: identity only */
    }
}

static uint64_t constHash(Value v, int depth) {
    switch (v.type) {
    case VAL_NULL:  return 0x9e3779b97f4a7c15ull;
    case VAL_BOOL:  return AS_BOOL(v) ? 0x2545f4914f6cdd1dull : 0xff51afd7ed558ccdull;
    case VAL_INT:   return jaiHashU64((uint64_t)AS_INT(v)) ^ 1u;
    case VAL_FLOAT: {
        uint64_t bits;
        double d = AS_FLOAT(v);
        memcpy(&bits, &d, sizeof bits);
        return jaiHashU64(bits) ^ 2u;
    }
    case VAL_OBJ:   break;
    }

    Obj *o = AS_OBJ(v);
    if (o == NULL) return 0;
    switch (o->type) {
    case OBJ_STRING:
        return jaiHashU64(AS_STRING(v)->hash) ^ 3u;
    case OBJ_BYTES:
        return jaiHashBytes(AS_BYTES(v)->data, AS_BYTES(v)->length) ^ 4u;
    case OBJ_TUPLE:
        if (depth < CONST_MAX_DEPTH) {
            const ObjTuple *t = AS_TUPLE(v);
            uint64_t h = 0x27d4eb2f165667c5ull ^ t->count;
            for (uint32_t i = 0; i < t->count; i++) {
                h = jaiHashU64(h ^ constHash(t->items[i], depth + 1));
            }
            return h;
        }
        break;
    default:
        break;
    }
    return jaiHashU64((uint64_t)(uintptr_t)o);
}

/* The chunk's dedup table, created and back-filled from the existing pool on
 * first use. Deserialised chunks arrive with a full pool and no index, so the
 * back-fill is what keeps their indices correct if anything appends later. */
static JaiTable *indexMapFor(Chunk *chunk) {
    if (chunk->constIndex != NULL) return chunk->constIndex;

    JaiTable *map = JAI_ALLOC(JaiTable, 1);
    jaiTableInit(map);
    chunk->constIndex = map;

    for (int i = 0; i < chunk->constants.count; i++) {
        Value existing = chunk->constants.data[i];
        if (!constDedupable(existing)) continue;
        Value key = INT_VAL((int64_t)constHash(existing, 0)), ignored;
        /* Keep the earliest index on a collision so indices stay stable. */
        if (!jaiTableGet(map, key, &ignored)) {
            jaiTableSet(map, key, INT_VAL(i));
        }
    }
    return map;
}

/* ------------------------------------------------------------------ */
/* Chunk lifetime                                                       */
/* ------------------------------------------------------------------ */

void jaiChunkInit(Chunk *chunk, int sourceFileId) {
    if (chunk == NULL) return;
    /* Init only zeroes: it is also called on raw memory (jaiFunctionNew), so
     * it must never dereference what the fields happen to contain. Releasing
     * a populated chunk is jaiChunkFree's job. */
    chunk->code = NULL;
    chunk->count = 0;
    chunk->capacity = 0;
    jaiValueArrayInit(&chunk->constants);
    chunk->lines = NULL;
    chunk->lineCount = 0;
    chunk->lineCapacity = 0;
    chunk->caches = NULL;
    chunk->cacheCount = 0;
    chunk->cacheCapacity = 0;
    chunk->constIndex = NULL;
    chunk->sourceFileId = sourceFileId;
}

void jaiChunkFree(Chunk *chunk) {
    if (chunk == NULL) return;

    JAI_FREE_ARRAY(uint8_t, chunk->code, chunk->capacity);
    jaiValueArrayFree(&chunk->constants);
    JAI_FREE_ARRAY(LineEntry, chunk->lines, chunk->lineCapacity);
    JAI_FREE_ARRAY(InlineCache, chunk->caches, chunk->cacheCapacity);
    if (chunk->constIndex != NULL) {
        jaiTableFree(chunk->constIndex);
        JAI_FREE(JaiTable, chunk->constIndex);
    }

    int sourceFileId = chunk->sourceFileId;
    jaiChunkInit(chunk, sourceFileId);
}

uint16_t jaiChunkAddCache(Chunk *chunk) {
    if (chunk == NULL) return 0;
    if (chunk->cacheCount >= UINT16_MAX + 1) {
        jaiDiagError(E0902_INTERNAL_ERROR,
                     (JaiSpan){0, 0, chunk->sourceFileId},
                     "too many inline caches in one function (limit %d)",
                     UINT16_MAX + 1);
        return 0;
    }
    if (chunk->cacheCapacity < chunk->cacheCount + 1) {
        int oldCapacity = chunk->cacheCapacity;
        chunk->cacheCapacity = JAI_GROW_CAP(oldCapacity);
        chunk->caches = JAI_GROW_ARRAY(InlineCache, chunk->caches, oldCapacity,
                                       chunk->cacheCapacity);
    }

    InlineCache *ic = &chunk->caches[chunk->cacheCount];
    memset(ic, 0, sizeof *ic);
    ic->state = IC_EMPTY;
    for (int i = 0; i < JAI_IC_WAYS; i++) ic->cached[i] = NULL_VAL;
    return (uint16_t)chunk->cacheCount++;
}

/* ------------------------------------------------------------------ */
/* Emission                                                            */
/* ------------------------------------------------------------------ */

/* One LineEntry per *run* of instructions sharing a source span: operand bytes
 * and consecutive instructions from the same expression collapse into one
 * entry, so the table stays tiny and jaiChunkSpanAt can binary-search it. */
static void recordSpan(Chunk *chunk, uint32_t start, uint32_t end) {
    if (chunk->lineCount > 0) {
        const LineEntry *last = &chunk->lines[chunk->lineCount - 1];
        if (last->span == start && last->spanEnd == end) return;
    }
    if (chunk->lineCapacity < chunk->lineCount + 1) {
        int oldCapacity = chunk->lineCapacity;
        chunk->lineCapacity = JAI_GROW_CAP(oldCapacity);
        chunk->lines = JAI_GROW_ARRAY(LineEntry, chunk->lines, oldCapacity,
                                      chunk->lineCapacity);
    }
    chunk->lines[chunk->lineCount].offset = (uint32_t)chunk->count;
    chunk->lines[chunk->lineCount].span = start;
    chunk->lines[chunk->lineCount].spanEnd = end;
    chunk->lineCount++;
}

void jaiChunkWrite(Chunk *chunk, uint8_t byte, uint32_t spanStart,
                   uint32_t spanEnd) {
    if (chunk == NULL) return;
    recordSpan(chunk, spanStart, spanEnd);   /* before count changes */

    if (chunk->capacity < chunk->count + 1) {
        int oldCapacity = chunk->capacity;
        chunk->capacity = JAI_GROW_CAP(oldCapacity);
        chunk->code = JAI_GROW_ARRAY(uint8_t, chunk->code, oldCapacity,
                                     chunk->capacity);
    }
    chunk->code[chunk->count++] = byte;
}

void jaiChunkWriteU16(Chunk *chunk, uint16_t v, uint32_t s, uint32_t e) {
    jaiChunkWrite(chunk, (uint8_t)(v & 0xff), s, e);
    jaiChunkWrite(chunk, (uint8_t)((v >> 8) & 0xff), s, e);
}

void jaiChunkWriteU24(Chunk *chunk, uint32_t v, uint32_t s, uint32_t e) {
    jaiChunkWrite(chunk, (uint8_t)(v & 0xff), s, e);
    jaiChunkWrite(chunk, (uint8_t)((v >> 8) & 0xff), s, e);
    jaiChunkWrite(chunk, (uint8_t)((v >> 16) & 0xff), s, e);
}

void jaiChunkWriteI16(Chunk *chunk, int16_t v, uint32_t s, uint32_t e) {
    jaiChunkWriteU16(chunk, (uint16_t)v, s, e);
}

/* Patching a slot outside the emitted code means the code generator computed a
 * bad jump address; there is no way to continue emitting correct bytecode. */
static void checkPatchRange(const Chunk *chunk, int offset, const char *what) {
    if (chunk == NULL || offset < 0 || offset + 1 >= chunk->count) {
        JAI_PANIC("chunk %s patch at offset %d is outside the code (%d bytes)",
                  what, offset, chunk == NULL ? 0 : chunk->count);
    }
}

void jaiChunkPatchU16(Chunk *chunk, int offset, uint16_t v) {
    checkPatchRange(chunk, offset, "u16");
    chunk->code[offset]     = (uint8_t)(v & 0xff);
    chunk->code[offset + 1] = (uint8_t)((v >> 8) & 0xff);
}

void jaiChunkPatchI16(Chunk *chunk, int offset, int16_t v) {
    checkPatchRange(chunk, offset, "i16");
    jaiChunkPatchU16(chunk, offset, (uint16_t)v);
}

uint32_t jaiChunkAddConstant(Chunk *chunk, Value v) {
    if (chunk == NULL) return 0;

    int count = chunk->constants.count;
    JaiTable *map = NULL;      /* non-NULL only when `hash` must be recorded */
    uint64_t hash = 0;

    if (count <= CONST_LINEAR_MAX) {
        for (int i = 0; i < count; i++) {
            if (constEqual(chunk->constants.data[i], v, 0)) return (uint32_t)i;
        }
    } else if (constDedupable(v)) {
        hash = constHash(v, 0);
        map = indexMapFor(chunk);
        Value found;
        if (jaiTableGet(map, INT_VAL((int64_t)hash), &found) && IS_INT(found)) {
            int64_t index = AS_INT(found);
            if (index >= 0 && index < count &&
                constEqual(chunk->constants.data[index], v, 0)) {
                return (uint32_t)index;
            }
            /* Hash collision with a different constant: fall through and append
             * a second entry rather than aliasing them. The earlier index keeps
             * the slot, so leave the table alone. */
            map = NULL;
        }
    }
    /* Beyond CONST_LINEAR_MAX, values that cannot be hashed structurally
     * (functions, class specs) are appended without a dedup search; they are
     * unique per emission site anyway. */

    if ((unsigned)count >= CONST_MAX_COUNT) {
        jaiDiagError(E0902_INTERNAL_ERROR,
                     (JaiSpan){0, 0, chunk->sourceFileId},
                     "constant pool overflow: a function may hold at most %u "
                     "constants", CONST_MAX_COUNT);
        return 0;
    }

    jaiValueArrayPush(&chunk->constants, v);
    if (map != NULL) {
        jaiTableSet(map, INT_VAL((int64_t)hash), INT_VAL(count));
    }
    return (uint32_t)count;
}

void jaiChunkSpanAt(const Chunk *chunk, int codeOffset, uint32_t *start,
                    uint32_t *end) {
    uint32_t s = 0, e = 0;
    if (chunk != NULL && chunk->lineCount > 0 && codeOffset >= 0) {
        int lo = 0, hi = chunk->lineCount - 1, best = -1;
        while (lo <= hi) {
            int mid = lo + (hi - lo) / 2;
            if (chunk->lines[mid].offset <= (uint32_t)codeOffset) {
                best = mid;
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        if (best >= 0) {
            s = chunk->lines[best].span;
            e = chunk->lines[best].spanEnd;
        }
    }
    if (start != NULL) *start = s;
    if (end != NULL) *end = e;
}

/* ------------------------------------------------------------------ */
/* Disassembly                                                          */
/* ------------------------------------------------------------------ */

/* 1-based source line covering `codeOffset`, or -1 when the chunk carries no
 * usable debug info. */
static int lineAt(const Chunk *chunk, int codeOffset) {
    if (chunk->sourceFileId < 0 || jaiSourceGet(chunk->sourceFileId) == NULL) {
        return -1;
    }
    uint32_t start = 0, end = 0;
    jaiChunkSpanAt(chunk, codeOffset, &start, &end);
    int line = -1, col = 0;
    jaiSourceLineCol(chunk->sourceFileId, start, &line, &col);
    return line;
}

static void printConstant(FILE *out, const Chunk *chunk, uint32_t index) {
    if (index < (uint32_t)chunk->constants.count) {
        jaiPrintValue(out, chunk->constants.data[index], true);
    } else {
        fprintf(out, "<invalid constant %u>", (unsigned)index);
    }
}

/* "operands  ; note", with the note column aligned. */
static void emitOperands(FILE *out, const char *operands, const char *note) {
    if (note != NULL && note[0] != '\0') {
        fprintf(out, "%-18s ; %s\n", operands, note);
    } else if (operands[0] != '\0') {
        fprintf(out, "%s\n", operands);
    } else {
        fputc('\n', out);
    }
}

/* As emitOperands, but the note is the constant's value followed by `suffix`. */
static void emitConstOperands(FILE *out, const Chunk *chunk,
                              const char *operands, uint32_t index,
                              const char *suffix) {
    fprintf(out, "%-18s ; ", operands);
    printConstant(out, chunk, index);
    if (suffix != NULL) fputs(suffix, out);
    fputc('\n', out);
}

/* "OFFSET  LINE  OP_NAME", where LINE is "|" when this instruction shares a
 * source line with the previous one. `pad` right-pads the mnemonic for the
 * operand column; it is off for operand-less instructions so that they do not
 * end in a run of spaces. */
static void printPrefix(FILE *out, const Chunk *chunk, int offset,
                        const char *name, bool pad) {
    char line[8];
    int here = lineAt(chunk, offset);
    if (offset > 0 && here == lineAt(chunk, offset - 1)) {
        snprintf(line, sizeof line, "%4s", "|");
    } else if (here < 0) {
        snprintf(line, sizeof line, "%4s", "?");
    } else {
        snprintf(line, sizeof line, "%4d", here);
    }
    fprintf(out, "%04d  %s  ", offset, line);
    fprintf(out, pad ? "%-23s" : "%s", name);
}

/* Jump operands are relative to the byte after the whole instruction, so the
 * absolute target is offset + 1 + operandSize + j (spec §2). */
static int jumpTarget(int offset, int size, int16_t j) {
    return offset + 1 + size + (int)j;
}

static int disassembleClosure(FILE *out, const Chunk *chunk, int offset) {
    int next = offset + 1 + CLOSURE_HEADER_BYTES;
    if (next > chunk->count) {
        emitOperands(out, "", "<truncated>");
        return chunk->count;
    }

    uint32_t index = jaiReadU24(chunk->code + offset + 1);
    char operands[32];
    snprintf(operands, sizeof operands, "%u", (unsigned)index);

    int upvalues = 0;
    bool known = false;
    if (index < (uint32_t)chunk->constants.count) {
        Value fn = chunk->constants.data[index];
        if (IS_FUNCTION(fn)) {
            upvalues = (int)AS_FUNCTION(fn)->upvalueCount;
            known = true;
        }
    }

    fprintf(out, "%-18s ; ", operands);
    printConstant(out, chunk, index);
    if (!known) {
        fputs("  <not a function: upvalue list unknown>", out);
        fputc('\n', out);
        return next;
    }
    fprintf(out, "  %d upvalue%s\n", upvalues, upvalues == 1 ? "" : "s");

    for (int i = 0; i < upvalues; i++) {
        if (next + CLOSURE_UPVALUE_BYTES > chunk->count) {
            fprintf(out, "%04d  %4s  %-23s %s\n", next, "|", "|",
                    "; <truncated upvalue list>");
            return chunk->count;
        }
        /* bit 0: source is a local of the enclosing frame; bit 1: captured by
         * value. Printing only bit 0 made a `let` capture indistinguishable
         * from a `var` one, which is the difference the reader is looking for. */
        uint8_t how = chunk->code[next];
        uint16_t slot = jaiReadU16(chunk->code + next + 1);
        fprintf(out, "%04d  %4s  %-23s %s %u%s\n", next, "|", "|",
                (how & 1u) ? "local" : "upvalue", (unsigned)slot,
                (how & 2u) ? " by value" : "");
        next += CLOSURE_UPVALUE_BYTES;
    }
    return next;
}

int jaiDisassembleInstruction(FILE *out, const Chunk *chunk, int offset) {
    if (out == NULL || chunk == NULL) return offset + 1;
    if (offset < 0 || offset >= chunk->count) return offset + 1;

    uint8_t raw = chunk->code[offset];
    OpCode op = (OpCode)raw;
    if (!opInRange(op)) {
        printPrefix(out, chunk, offset, "OP_UNKNOWN", true);
        char note[32];
        snprintf(note, sizeof note, "0x%02x", (unsigned)raw);
        emitOperands(out, "", note);
        return offset + 1;
    }

    int size = jaiOpOperandSize(op);
    printPrefix(out, chunk, offset, jaiOpName(op), size != 0);

    if (op == OP_CLOSURE) return disassembleClosure(out, chunk, offset);

    if (offset + 1 + size > chunk->count) {
        emitOperands(out, "", "<truncated>");
        return chunk->count;
    }

    const uint8_t *a = chunk->code + offset + 1;
    int next = offset + 1 + size;
    char operands[64];
    char note[128];
    operands[0] = '\0';
    note[0] = '\0';

    switch (op) {
    /* --- u24 constant index --- */
    case OP_CONST:
    case OP_DEF_GLOBAL:
    case OP_CLASS:
    case OP_IMPL_TRAIT:
    case OP_GET_SUPER:
    case OP_IS_INSTANCE:
    case OP_PUSH_DEFER:
    case OP_MATCH_EXC:
    case OP_IMPORT:
    case OP_IMPORT_FROM:
    case OP_EXPORT:
    case OP_ASSERT_FAIL:
    case OP_TYPE_GUARD: {
        uint32_t k = jaiReadU24(a);
        snprintf(operands, sizeof operands, "%u", (unsigned)k);
        emitConstOperands(out, chunk, operands, k, NULL);
        return next;
    }

    /* --- u24 constant + u16 inline cache --- */
    case OP_GET_GLOBAL:
    case OP_SET_GLOBAL:
    case OP_GET_FIELD:
    case OP_SET_FIELD: {
        uint32_t k = jaiReadU24(a);
        uint16_t cache = jaiReadU16(a + 3);
        char suffix[32];
        snprintf(operands, sizeof operands, "%u c%u", (unsigned)k,
                 (unsigned)cache);
        snprintf(suffix, sizeof suffix, "  cache %u", (unsigned)cache);
        emitConstOperands(out, chunk, operands, k, suffix);
        return next;
    }

    /* --- u24 constant + u24 constant --- */
    case OP_GET_MODULE: {
        uint32_t mod = jaiReadU24(a), member = jaiReadU24(a + 3);
        snprintf(operands, sizeof operands, "%u %u", (unsigned)mod,
                 (unsigned)member);
        fprintf(out, "%-18s ; ", operands);
        printConstant(out, chunk, mod);
        fputc('.', out);
        printConstant(out, chunk, member);
        fputc('\n', out);
        return next;
    }

    /* --- u24 constant + u8 flags + u8 visibility --- */
    case OP_METHOD: {
        uint32_t k = jaiReadU24(a);
        unsigned flags = a[3], vis = a[4];
        char suffix[64];
        snprintf(operands, sizeof operands, "%u %u %u", (unsigned)k, flags, vis);
        snprintf(suffix, sizeof suffix, "  flags 0x%02x, visibility %u", flags,
                 vis);
        emitConstOperands(out, chunk, operands, k, suffix);
        return next;
    }

    /* --- u24 constant + u8 --- */
    case OP_FIELD_DEF:
    case OP_SUPER_INVOKE:
    case OP_NEW: {
        uint32_t k = jaiReadU24(a);
        unsigned b = a[3];
        char suffix[48];
        snprintf(operands, sizeof operands, "%u %u", (unsigned)k, b);
        if (op == OP_FIELD_DEF) {
            /* Packed, not a plain number (spec §3.7): printing it raw read as
             * "visibility 10" for a public `let`. */
            static const char *const kVis[4] = { "private", "protected",
                                                 "public", "?" };
            snprintf(suffix, sizeof suffix, "  %s%s%s", kVis[b & 0x3],
                     (b & 0x4) ? " static" : "", (b & 0x8) ? " let" : "");
        } else {
            snprintf(suffix, sizeof suffix, "  %u arg%s", b, b == 1 ? "" : "s");
        }
        emitConstOperands(out, chunk, operands, k, suffix);
        return next;
    }

    /* --- u24 constant + u8 argc + u16 inline cache --- */
    case OP_INVOKE: {
        uint32_t k = jaiReadU24(a);
        unsigned argc = a[3];
        uint16_t cache = jaiReadU16(a + 4);
        char suffix[64];
        snprintf(operands, sizeof operands, "%u %u c%u", (unsigned)k, argc,
                 (unsigned)cache);
        snprintf(suffix, sizeof suffix, "  %u arg%s, cache %u", argc,
                 argc == 1 ? "" : "s", (unsigned)cache);
        emitConstOperands(out, chunk, operands, k, suffix);
        return next;
    }

    /* --- u8 argc + u24 keyword-name tuple --- */
    case OP_CALL_KW: {
        unsigned argc = a[0];
        uint32_t k = jaiReadU24(a + 1);
        char suffix[48];
        snprintf(operands, sizeof operands, "%u %u", argc, (unsigned)k);
        snprintf(suffix, sizeof suffix, "  %u positional", argc);
        emitConstOperands(out, chunk, operands, k, suffix);
        return next;
    }

    /* --- u24 constant + u8 tag + u8 argc --- */
    case OP_ENUM_NEW: {
        uint32_t k = jaiReadU24(a);
        unsigned tag = a[3], argc = a[4];
        char suffix[64];
        snprintf(operands, sizeof operands, "%u %u %u", (unsigned)k, tag, argc);
        snprintf(suffix, sizeof suffix, "  tag %u, %u arg%s", tag, argc,
                 argc == 1 ? "" : "s");
        emitConstOperands(out, chunk, operands, k, suffix);
        return next;
    }

    /* --- i16 jump --- */
    case OP_JUMP:
    case OP_JUMP_IF_FALSE:
    case OP_JUMP_IF_TRUE:
    case OP_JUMP_IF_FALSE_KEEP:
    case OP_JUMP_IF_TRUE_KEEP:
    case OP_JUMP_IF_NULL:
    case OP_LOOP:
    case OP_FOR_ITER:
    case OP_PUSH_FINALLY: {
        int16_t j = jaiReadI16(a);
        snprintf(operands, sizeof operands, "%+d", (int)j);
        snprintf(note, sizeof note, "-> %04d", jumpTarget(offset, size, j));
        break;
    }

    /* --- i16 jump + u24 exception class --- */
    case OP_PUSH_HANDLER: {
        int16_t j = jaiReadI16(a);
        uint32_t k = jaiReadU24(a + 2);
        char suffix[48];
        snprintf(operands, sizeof operands, "%+d %u", (int)j, (unsigned)k);
        snprintf(suffix, sizeof suffix, ", handler -> %04d",
                 jumpTarget(offset, size, j));
        emitConstOperands(out, chunk, operands, k, suffix);
        return next;
    }

    /* --- u24 constant + i16 jump --- */
    case OP_MATCH_CONST:
    case OP_MATCH_TYPE:
    case OP_MATCH_FIELDS: {
        uint32_t k = jaiReadU24(a);
        int16_t j = jaiReadI16(a + 3);
        char suffix[48];
        snprintf(operands, sizeof operands, "%u %+d", (unsigned)k, (int)j);
        snprintf(suffix, sizeof suffix, ", no match -> %04d",
                 jumpTarget(offset, size, j));
        emitConstOperands(out, chunk, operands, k, suffix);
        return next;
    }

    /* --- u24 lo, u24 hi, u8 inclusive, i16 jump --- */
    case OP_MATCH_RANGE: {
        uint32_t lo = jaiReadU24(a), hi = jaiReadU24(a + 3);
        unsigned inclusive = a[6];
        int16_t j = jaiReadI16(a + 7);
        snprintf(operands, sizeof operands, "%u %u %u %+d", (unsigned)lo,
                 (unsigned)hi, inclusive, (int)j);
        fprintf(out, "%-18s ; ", operands);
        printConstant(out, chunk, lo);
        fputs(inclusive ? " ..= " : " ..< ", out);
        printConstant(out, chunk, hi);
        fprintf(out, ", no match -> %04d\n", jumpTarget(offset, size, j));
        return next;
    }

    /* --- u8 count, u8 rest flag, i16 jump --- */
    case OP_MATCH_SEQ: {
        unsigned count = a[0], rest = a[1];
        int16_t j = jaiReadI16(a + 2);
        snprintf(operands, sizeof operands, "%u %u %+d", count, rest, (int)j);
        snprintf(note, sizeof note, "%u element%s%s, no match -> %04d", count,
                 count == 1 ? "" : "s", rest ? " + rest" : "",
                 jumpTarget(offset, size, j));
        break;
    }

    /* --- u16 local slot --- */
    case OP_GET_LOCAL:
    case OP_SET_LOCAL:
    case OP_BIND: {
        snprintf(operands, sizeof operands, "%u", (unsigned)jaiReadU16(a));
        snprintf(note, sizeof note, "slot %u", (unsigned)jaiReadU16(a));
        break;
    }

    /* --- u16 local slot, closed together with everything above it --- */
    case OP_CLOSE_UPVALUE: {
        unsigned slot = (unsigned)jaiReadU16(a);
        snprintf(operands, sizeof operands, "%u", slot);
        snprintf(note, sizeof note, "close slot %u and above", slot);
        break;
    }

    /* --- u16 element count --- */
    case OP_BUILD_LIST:
    case OP_BUILD_DICT:
    case OP_BUILD_SET:
    case OP_BUILD_TUPLE: {
        unsigned n = jaiReadU16(a);
        snprintf(operands, sizeof operands, "%u", n);
        snprintf(note, sizeof note, "%u element%s", n, n == 1 ? "" : "s");
        break;
    }

    /* --- u16 stack depth of the target container --- */
    case OP_LIST_APPEND:
    case OP_DICT_INSERT:
    case OP_SET_ADD: {
        unsigned depth = jaiReadU16(a);
        snprintf(operands, sizeof operands, "%u", depth);
        snprintf(note, sizeof note, "container at peek(%u)", depth);
        break;
    }

    /* --- u8 upvalue index --- */
    case OP_GET_UPVALUE:
    case OP_SET_UPVALUE: {
        snprintf(operands, sizeof operands, "%u", (unsigned)a[0]);
        snprintf(note, sizeof note, "upvalue %u", (unsigned)a[0]);
        break;
    }

    /* --- u8 argument count --- */
    case OP_CALL:
    case OP_CALL_SPREAD:
    case OP_TAIL_CALL: {
        unsigned argc = a[0];
        snprintf(operands, sizeof operands, "%u", argc);
        snprintf(note, sizeof note, "%u arg%s", argc, argc == 1 ? "" : "s");
        break;
    }

    case OP_POPN: {
        snprintf(operands, sizeof operands, "%u", (unsigned)a[0]);
        snprintf(note, sizeof note, "pop %u", (unsigned)a[0]);
        break;
    }

    case OP_BUILD_RANGE: {
        snprintf(operands, sizeof operands, "%u", (unsigned)a[0]);
        snprintf(note, sizeof note, "%s", a[0] ? "inclusive" : "exclusive");
        break;
    }

    case OP_GET_SLICE:
    case OP_SET_SLICE: {
        unsigned flags = a[0];
        snprintf(operands, sizeof operands, "0x%02x", flags);
        snprintf(note, sizeof note, "%s%s%s",
                 (flags & 1) ? "start " : "", (flags & 2) ? "stop " : "",
                 (flags & 4) ? "step" : "");
        break;
    }

    case OP_UNPACK: {
        unsigned count = a[0], rest = a[1];
        snprintf(operands, sizeof operands, "%u %u", count, rest);
        if (rest == 255) {
            snprintf(note, sizeof note, "%u value%s, no rest", count,
                     count == 1 ? "" : "s");
        } else {
            snprintf(note, sizeof note, "%u value%s, rest at %u", count,
                     count == 1 ? "" : "s", rest);
        }
        break;
    }

    case OP_ENUM_FIELD: {
        snprintf(operands, sizeof operands, "%u", (unsigned)a[0]);
        snprintf(note, sizeof note, "payload field %u", (unsigned)a[0]);
        break;
    }

    /* --- immediates and fused local forms --- */
    case OP_INT: {
        int16_t imm = jaiReadI16(a);
        snprintf(operands, sizeof operands, "%d", (int)imm);
        break;
    }

    case OP_ADD_INT_CONST:
    case OP_CMP_LOCAL_CONST_LT: {
        unsigned slot = jaiReadU16(a);
        int16_t imm = jaiReadI16(a + 2);
        snprintf(operands, sizeof operands, "%u %d", slot, (int)imm);
        snprintf(note, sizeof note, "slot %u %s %d", slot,
                 op == OP_ADD_INT_CONST ? "+" : "<", (int)imm);
        break;
    }

    case OP_INC_LOCAL: {
        unsigned slot = jaiReadU16(a);
        int imm = (int)(int8_t)a[2];
        snprintf(operands, sizeof operands, "%u %d", slot, imm);
        snprintf(note, sizeof note, "slot %u += %d", slot, imm);
        break;
    }

    case OP_GET_LOCAL2:
    case OP_ADD_LOCALS: {
        unsigned x = jaiReadU16(a), y = jaiReadU16(a + 2);
        snprintf(operands, sizeof operands, "%u %u", x, y);
        snprintf(note, sizeof note, "slot %u, slot %u", x, y);
        break;
    }

    /* --- u16 local slot + u24 constant + u16 inline cache --- */
    case OP_GET_FIELD_LOCAL: {
        unsigned slot = jaiReadU16(a);
        uint32_t k = jaiReadU24(a + 2);
        unsigned cache = jaiReadU16(a + 5);
        char suffix[48];
        snprintf(operands, sizeof operands, "%u %u c%u", slot, (unsigned)k,
                 cache);
        snprintf(suffix, sizeof suffix, " of slot %u, cache %u", slot, cache);
        emitConstOperands(out, chunk, operands, k, suffix);
        return next;
    }

    /* --- i16 jump + u16 local slot --- */
    case OP_FOR_ITER_BIND: {
        int16_t j = jaiReadI16(a);
        unsigned slot = jaiReadU16(a + 2);
        snprintf(operands, sizeof operands, "%+d %u", (int)j, slot);
        snprintf(note, sizeof note, "-> slot %u, exhausted -> %04d", slot,
                 jumpTarget(offset, size, j));
        break;
    }

    /* --- u8 comparison opcode + i16 jump --- */
    case OP_JUMP_IF_CMP_FALSE: {
        int16_t j = jaiReadI16(a + 1);
        snprintf(operands, sizeof operands, "%s %+d", jaiOpName((OpCode)a[0]),
                 (int)j);
        snprintf(note, sizeof note, "false -> %04d",
                 jumpTarget(offset, size, j));
        break;
    }

    /* --- u8 comparison opcode + u16 local slot + u24 constant + i16 jump --- */
    case OP_JUMP_IF_CMP_LOCAL_K: {
        unsigned slot = jaiReadU16(a + 1);
        uint32_t k = jaiReadU24(a + 3);
        int16_t j = jaiReadI16(a + 6);
        snprintf(operands, sizeof operands, "%s %u %u %+d",
                 jaiOpName((OpCode)a[0]), slot, (unsigned)k, (int)j);
        char suffix[64];
        snprintf(suffix, sizeof suffix, "  slot %u, false -> %04d", slot,
                 jumpTarget(offset, size, j));
        emitConstOperands(out, chunk, operands, k, suffix);
        return next;
    }

    /* --- u8 part count, u24 literal mask, u24 name, u16 cache --- */
    case OP_FORMAT: {
        unsigned n = a[0];
        snprintf(operands, sizeof operands, "%u %06x", n,
                 (unsigned)jaiReadU24(a + 1));
        snprintf(note, sizeof note, "%u part%s, literals %06x", n,
                 n == 1 ? "" : "s", (unsigned)jaiReadU24(a + 1));
        break;
    }

    default:
        /* No operands, or an opcode whose operands need no annotation. */
        if (size > 0) {
            int written = 0;
            for (int i = 0; i < size && written >= 0 &&
                            (size_t)written < sizeof operands - 4; i++) {
                written += snprintf(operands + written,
                                    sizeof operands - (size_t)written,
                                    i == 0 ? "%02x" : " %02x", (unsigned)a[i]);
            }
        }
        break;
    }

    emitOperands(out, operands, note);
    return next;
}

void jaiDisassembleChunk(FILE *out, const Chunk *chunk, const char *name) {
    if (out == NULL || chunk == NULL) return;

    fprintf(out, "== %s ==\n", name != NULL ? name : "<chunk>");
    fprintf(out, "; %d byte%s, %d constant%s, %d cache%s\n", chunk->count,
            chunk->count == 1 ? "" : "s", chunk->constants.count,
            chunk->constants.count == 1 ? "" : "s", chunk->cacheCount,
            chunk->cacheCount == 1 ? "" : "s");

    for (int offset = 0; offset < chunk->count;) {
        int next = jaiDisassembleInstruction(out, chunk, offset);
        if (next <= offset) break;   /* malformed chunk: never spin */
        offset = next;
    }
}
