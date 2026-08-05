/* codegen_internal.h — the interface between the code generator's five files.
 *
 * Lowering is a structural walk of a mutually recursive tree: a declaration
 * holds statements, a statement holds expressions, an expression can hold a
 * `match` or a block, a pattern can hold an expression. No split of codegen.c
 * can be a one-way dependency, so what this header fixes is the *shape* of the
 * coupling:
 *
 *   the plumbing  Emitter, FnCtx and the byte-level emitters. Small enough to
 *                 be `static inline` here rather than calls across a file, and
 *                 unprefixed because they are this header's whole vocabulary.
 *   jaiCg*        the rest of the shared machinery — symbols, operators,
 *                 literals, type guards. Defined in codegen.c.
 *   jaiEmit*      one construct each, defined in the file named after its
 *                 syntactic category and called from wherever it can appear.
 *
 * Nothing else crosses a file boundary: every other helper is static in the
 * file that uses it. Not a public interface — codegen.h is what the rest of
 * the tree calls.
 *
 *   codegen.c          the emitter's shared machinery and the entry points
 *   codegen_expr.c     expressions, calls, closures, comprehensions (spec §4)
 *   codegen_stmt.c     statements and control flow (spec §5)
 *   codegen_decl.c     functions, classes, traits, enums (spec §6, §7)
 *   codegen_pattern.c  pattern lowering and `match` (spec §5.3)
 */
#ifndef JAI_CODEGEN_INTERNAL_H
#define JAI_CODEGEN_INTERNAL_H

#include "codegen.h"

#include "../common/diag.h"
#include "../sema/resolve.h"
#include "../vm/chunk.h"
#include "../vm/gc.h"
#include "../vm/object.h"

/* Emitter state                                                        */
/* ------------------------------------------------------------------ */

/* A jump whose target is not known yet. `end` is the offset one past the whole
 * instruction, which is what spec §2 measures the i16 displacement from. */
typedef struct {
    int operand;
    int end;
} JumpSite;

typedef JAI_VEC(JumpSite) JumpList;

typedef struct LoopCtx {
    struct LoopCtx *enclosing;
    const char     *label;         /* NULL for an unlabelled loop */
    int             continueTarget;/* code offset to LOOP back to */
    int             continueDepth; /* stack depth at continueTarget */
    int             breakDepth;    /* stack depth once the loop has exited */
    int             finallyDepth;  /* active finally blocks at loop entry */
    int             captureBase;   /* resolver's AstNode captureBase, or -1 */
    JumpList        breaks;
} LoopCtx;

/* A `finally` block that a non-local exit has to run on its way out. The
 * exception path goes through the exception table; return/break/continue get an
 * inline copy, which is why the block AST is kept here. */
typedef struct FinallyCtx {
    struct FinallyCtx *enclosing;
    AstNode           *block;
    int                loopDepth;  /* loops on the stack when the try started */
} FinallyCtx;

typedef struct FnCtx {
    struct FnCtx  *enclosing;
    ObjFunction   *fn;
    FunctionScope *scope;          /* resolver output; NULL when unavailable */
    AstNode       *decl;

    int stackDepth;
    int maxStackDepth;
    int nextTemp;                  /* first slot above the resolver's locals */
    int maxSlot;

    LoopCtx    *loops;
    int         loopDepth;

    /* Lowest by-reference capture base of the blocks currently open around the
     * cursor, or -1. `return` leaves all of them at once and has to close them
     * before the inlined `finally` copies and the defer thunks run, since those
     * share this frame's slots and would otherwise overwrite a slot an escaping
     * closure still points at. Saved and restored around each block. */
    int         closeBase;
    /* Watermark, not a stack: the lowest base of any block or loop *opened*
     * since it was last reset. emitTry resets it around the protected region to
     * learn what the unwind path has to close, because an exception skips every
     * block close inside that region. */
    int         deepCloseBase;

    FinallyCtx *finallys;
    int         finallyDepth;
    int         protectDepth;      /* >0 inside try/catch/finally/defer */
    bool        hasDefer;
    bool        isGenerator;

    JAI_VEC(ExceptionEntry) exceptions;
} FnCtx;

struct Emitter {
    const CodegenOptions *opts;
    ObjModule            *module;
    FnCtx                *fn;
    int                   fileId;
    int                   errors;
};

/* Codegen only ever reports internal errors: everything a user can get wrong
 * was already rejected by the parser, resolver, or checker. */
#define CG_ERROR(e, span, code, ...)                                           \
    do {                                                                       \
        JaiSpan _cgSpan = (span);                                              \
        jaiDiagError((code), _cgSpan, __VA_ARGS__);                            \
        (e)->errors++;                                                         \
    } while (0)

/* Fallback span for diagnostics about the chunk rather than about a node. */
static inline JaiSpan fileSpan(const Emitter *e) {
    JaiSpan span = {0, 0, e->fileId};
    return span;
}

/* ------------------------------------------------------------------ */
/* Primitive emission                                                   */
/* ------------------------------------------------------------------ */

static inline Chunk *chunkOf(Emitter *e) { return &e->fn->fn->chunk; }
static inline int    here(Emitter *e)    { return chunkOf(e)->count; }

static inline void adjust(Emitter *e, int delta) {
    FnCtx *f = e->fn;
    f->stackDepth += delta;
    if (f->stackDepth < 0) f->stackDepth = 0;   /* only reachable after an error */
    if (f->stackDepth > f->maxStackDepth) f->maxStackDepth = f->stackDepth;
}

/* Re-synchronise the tracked depth at a merge point (a patched label). */
static inline void setDepth(Emitter *e, int d) {
    FnCtx *f = e->fn;
    f->stackDepth = d;
    if (d > f->maxStackDepth) f->maxStackDepth = d;
}

static inline int depthOf(Emitter *e) { return e->fn->stackDepth; }

static inline void emitByte(Emitter *e, uint8_t b, JaiSpan span) {
    jaiChunkWrite(chunkOf(e), b, span.start, span.end);
}

static inline void emitU16(Emitter *e, uint16_t v, JaiSpan span) {
    jaiChunkWriteU16(chunkOf(e), v, span.start, span.end);
}

static inline void emitU24(Emitter *e, uint32_t v, JaiSpan span) {
    jaiChunkWriteU24(chunkOf(e), v, span.start, span.end);
}

static inline void emitI16(Emitter *e, int16_t v, JaiSpan span) {
    jaiChunkWriteI16(chunkOf(e), v, span.start, span.end);
}

/* Emits an opcode and applies its fixed stack effect. Opcodes whose effect
 * depends on their operands (SE_VAR, reported as INT32_MIN) are left to the
 * caller, which knows the count. */
static inline void emitOp(Emitter *e, OpCode op, JaiSpan span) {
    emitByte(e, (uint8_t)op, span);
    int effect = jaiOpStackEffect(op);
    if (effect != INT32_MIN) adjust(e, effect);
}

static inline void emitPop(Emitter *e, JaiSpan span) { emitOp(e, OP_POP, span); }

static inline void emitPopN(Emitter *e, int n, JaiSpan span) {
    while (n > 0) {
        if (n == 1) {
            emitPop(e, span);
            return;
        }
        int chunkSize = n > 255 ? 255 : n;
        emitOp(e, OP_POPN, span);
        emitByte(e, (uint8_t)chunkSize, span);
        adjust(e, -chunkSize);
        n -= chunkSize;
    }
}

/* ------------------------------------------------------------------ */
/* Constants                                                            */
/* ------------------------------------------------------------------ */

/* Every constant is rooted across jaiChunkAddConstant: the pool grows through
 * jaiRealloc, which can trigger a collection, and the chunk is not yet
 * reachable from anything the collector scans. */
static inline uint32_t addConst(Emitter *e, Value v) {
    jaiGCPushRoot(v);
    uint32_t index = jaiChunkAddConstant(chunkOf(e), v);
    jaiGCPopRoot();
    return index;
}

/* Strings and identifiers are interned so the VM compares them by pointer. */
static inline uint32_t strConst(Emitter *e, const char *chars, size_t length) {
    ObjString *s = jaiStringIntern(chars != NULL ? chars : "", length);
    return addConst(e, OBJ_VAL(s));
}

static inline uint32_t nameConst(Emitter *e, const char *name) {
    return strConst(e, name, name != NULL ? strlen(name) : 0);
}

static inline uint16_t newCache(Emitter *e) { return jaiChunkAddCache(chunkOf(e)); }

/* ------------------------------------------------------------------ */
/* Jumps and labels                                                     */
/* ------------------------------------------------------------------ */

static inline void jumpListInit(JumpList *list) { JAI_VEC_INIT(list); }
static inline void jumpListFree(JumpList *list) { JAI_VEC_FREE(JumpSite, list); }

static inline JumpSite emitJump(Emitter *e, OpCode op, JaiSpan span) {
    emitOp(e, op, span);
    JumpSite site;
    site.operand = here(e);
    emitI16(e, 0, span);
    site.end = here(e);
    return site;
}

static inline void patchJump(Emitter *e, JumpSite site) {
    long delta = (long)here(e) - (long)site.end;
    if (delta < INT16_MIN || delta > INT16_MAX) {
        CG_ERROR(e, fileSpan(e), E0902_INTERNAL_ERROR,
                 "jump distance %ld exceeds the 16-bit branch range; "
                 "split the enclosing function", delta);
        delta = 0;
    }
    jaiChunkPatchI16(chunkOf(e), site.operand, (int16_t)delta);
}

static inline void patchAll(Emitter *e, JumpList *list) {
    for (int i = 0; i < list->count; i++) patchJump(e, list->data[i]);
    list->count = 0;
}

static inline void emitLoopBack(Emitter *e, int target, JaiSpan span) {
    emitOp(e, OP_LOOP, span);
    int operand = here(e);
    long delta = (long)target - (long)(operand + 2);
    if (delta < INT16_MIN || delta > INT16_MAX) {
        CG_ERROR(e, span, E0902_INTERNAL_ERROR,
                 "loop body is too large for a 16-bit backward branch");
        delta = 0;
    }
    emitI16(e, (int16_t)delta, span);
}

/* ------------------------------------------------------------------ */
/* Frame slots                                                          */
/* ------------------------------------------------------------------ */

/* Temporaries (comparison-chain operands, match subjects, destructuring
 * scratch) are allocated above the resolver's peak slot usage rather than on
 * the operand stack. Keeping them in slots is what makes the pattern-matching
 * miss invariant hold: a test never has to unwind a variable number of
 * partially destructured values. */
static inline int allocTemp(Emitter *e) {
    FnCtx *f = e->fn;
    if (f->nextTemp >= JAI_MAX_LOCALS) {
        CG_ERROR(e, fileSpan(e), E0208_TOO_MANY_LOCALS,
                 "too many temporary slots in one function (max %d)",
                 JAI_MAX_LOCALS);
        return JAI_MAX_LOCALS - 1;
    }
    int slot = f->nextTemp++;
    if (f->nextTemp > f->maxSlot) f->maxSlot = f->nextTemp;
    return slot;
}

static inline void freeTemps(Emitter *e, int n) {
    e->fn->nextTemp -= n;
    if (e->fn->nextTemp < 0) e->fn->nextTemp = 0;
}

static inline void emitGetLocal(Emitter *e, int slot, JaiSpan span) {
    emitOp(e, OP_GET_LOCAL, span);
    emitU16(e, (uint16_t)(slot < 0 ? 0 : slot), span);
}

/* Store-and-pop into a frame slot. */
static inline void emitBindLocal(Emitter *e, int slot, JaiSpan span) {
    emitOp(e, OP_BIND, span);
    emitU16(e, (uint16_t)(slot < 0 ? 0 : slot), span);
}

/* Detach every open upvalue at or above `base` from the frame, so that the
 * closures already made over those slots keep their values while the slots
 * themselves are handed to whatever is declared next (spec §5.2, §6). A scope
 * is named by its lowest slot because locals live in a slot window rather than
 * on the operand stack: everything at or above it belongs to that scope.
 *
 * base < 0 — nothing in the scope was captured by reference, which is the
 * overwhelmingly common case — emits nothing at all, so an ordinary block or
 * loop pays no instruction for the rule. */
static inline void emitCloseScope(Emitter *e, int base, JaiSpan span) {
    if (base < 0) return;
    emitOp(e, OP_CLOSE_UPVALUE, span);
    emitU16(e, (uint16_t)base, span);
}

/* Record a scope's base on the enclosing FnCtx: `closeBase` is the running
 * minimum over the blocks open right here, `deepCloseBase` the minimum over
 * everything opened since the last reset. Returns the previous `closeBase`,
 * which the caller restores when the scope ends. */
static inline int enterCloseScope(Emitter *e, int base) {
    FnCtx *f = e->fn;
    int saved = f->closeBase;
    if (base >= 0) {
        if (f->closeBase < 0 || base < f->closeBase) f->closeBase = base;
        if (f->deepCloseBase < 0 || base < f->deepCloseBase) f->deepCloseBase = base;
    }
    return saved;
}

/* ------------------------------------------------------------------ */
/* codegen.c — shared machinery                                         */
/* ------------------------------------------------------------------ */

void jaiCgGetGlobal(Emitter *e, const char *name, JaiSpan span);
void jaiCgLoadSymbol(Emitter *e, Symbol *sym, const char *name, JaiSpan span);
void jaiCgStoreSymbol(Emitter *e, Symbol *sym, const char *name, JaiSpan span,
               bool declaring);

OpCode jaiCgOpcodeForOp(OpKind op);
void   jaiCgInvokeName(Emitter *e, const char *name, int argc, JaiSpan span);
void   jaiCgBinaryOp(Emitter *e, OpKind op, JaiSpan span);

void jaiCgInt(Emitter *e, int64_t value, JaiSpan span);
void jaiCgConstValue(Emitter *e, Value v, JaiSpan span);
/* True when `node` is a literal the emitter can fold to a Value outright. */
bool jaiCgLiteralValue(AstNode *node, Value *out);

/* The spelling a runtime type guard tests against, as a constant index. */
uint32_t    jaiCgTypeGuardConst(Emitter *e, const AstType *target);
const char *jaiCgTypeNameOf(const AstType *type);

/* Close out a function body: emit the implicit return, run the optimiser and
 * the verifier, and pop the FnCtx. */
void jaiCgFinishFunction(Emitter *e, ObjFunction *fn);

/* ------------------------------------------------------------------ */
/* codegen_expr.c — expressions                                         */
/* ------------------------------------------------------------------ */

/* Leaves exactly one value on the operand stack. */
void jaiEmitExpr(Emitter *e, AstNode *node);
void jaiEmitBlockValue(Emitter *e, AstNode *node);
void jaiEmitCall(Emitter *e, AstNode *node, bool tail);
void jaiEmitAssign(Emitter *e, AstNode *node);

/* The function object for a `fn` node, as a constant index, and the CLOSURE
 * that captures its upvalues. */
uint32_t jaiEmitFunctionConstant(Emitter *e, AstNode *node, uint32_t extraFlags,
                          const char *ownerName,
                          const AstField *initFields, int initFieldCount);
void     jaiEmitClosure(Emitter *e, AstNode *node, uint32_t fnConst,
                 JaiSpan span);

/* ------------------------------------------------------------------ */
/* codegen_stmt.c — statements and control flow                         */
/* ------------------------------------------------------------------ */

/* Leaves nothing on the operand stack. */
void jaiEmitStmt(Emitter *e, AstNode *node);

/* ------------------------------------------------------------------ */
/* codegen_decl.c — declarations                                        */
/* ------------------------------------------------------------------ */

void jaiEmitClassDecl(Emitter *e, AstNode *node);
void jaiEmitTraitDecl(Emitter *e, AstNode *node);
void jaiEmitEnumDecl(Emitter *e, AstNode *node);

/* Compile a `fn` node into its own ObjFunction, in a nested FnCtx. */
ObjFunction *jaiEmitFunctionNode(Emitter *e, AstNode *node,
                          uint32_t extraFlags, const char *ownerName,
                          const AstField *initFields,
                          int initFieldCount);

/* ------------------------------------------------------------------ */
/* codegen_pattern.c — patterns and match                               */
/* ------------------------------------------------------------------ */

/* Destructure the value on top of the stack into the pattern's bindings. The
 * pattern is irrefutable here: the checker proved it (spec §5.3). */
void jaiEmitStorePattern(Emitter *e, AstNode *pattern, bool declaring);

/* `match` as a statement leaves nothing; as an expression, one value. */
void jaiEmitMatch(Emitter *e, AstNode *node, bool asExpression);

#endif /* JAI_CODEGEN_INTERNAL_H */
