/* codegen_expr.c — expressions (spec §4).
 *
 * Every emitter here leaves exactly one value on the operand stack, which is
 * what the depth tracking in codegen_internal.h relies on. Calls and closures
 * are here because a call and a lambda are expressions; the `fn` declaration
 * that shares their lowering is in codegen_decl.c.
 */
#include "codegen_internal.h"

/* `if` in expression position, which emitBranchValue reaches before the
 * definition below. */
static void emitTernary(Emitter *e, AstNode *cond, AstNode *thenExpr,
                        AstNode *elseExpr, JaiSpan span);

/* Calls                                                                */
/* ------------------------------------------------------------------ */

static void argKinds(const AstArg *args, int count, int *outSpreads,
                     int *outKeywords) {
    int spreads = 0, keywords = 0;
    for (int i = 0; i < count; i++) {
        if (args[i].isSpread) spreads++;
        else if (args[i].name != NULL) keywords++;
    }
    *outSpreads = spreads;
    *outKeywords = keywords;
}

/* Builds the spread tail as a single list on the stack: runs of plain
 * arguments become BUILD_LIST, spread operands are concatenated with ADD
 * (list + list is list concatenation). */
static void emitSpreadTail(Emitter *e, const AstArg *args, int from, int count,
                           JaiSpan span) {
    bool haveList = false;
    int pending = 0;

    for (int i = from; i < count; i++) {
        if (!args[i].isSpread) {
            jaiEmitExpr(e, args[i].value);
            pending++;
            continue;
        }
        if (pending > 0) {
            emitOp(e, OP_BUILD_LIST, span);
            emitU16(e, (uint16_t)pending, span);
            adjust(e, -pending + 1);
            if (haveList) emitOp(e, OP_ADD, span);
            haveList = true;
            pending = 0;
        }
        jaiEmitExpr(e, args[i].value);
        if (haveList) emitOp(e, OP_ADD, span);
        haveList = true;
    }

    if (pending > 0) {
        emitOp(e, OP_BUILD_LIST, span);
        emitU16(e, (uint16_t)pending, span);
        adjust(e, -pending + 1);
        if (haveList) emitOp(e, OP_ADD, span);
        haveList = true;
    }
    if (!haveList) {
        emitOp(e, OP_BUILD_LIST, span);
        emitU16(e, 0, span);
        adjust(e, 1);
    }
}

/* Keyword names as a constant tuple, in the order their values were pushed. */
static uint32_t keywordNameTuple(Emitter *e, const AstArg *args, int count) {
    Value items[JAI_MAX_ARGS];
    int n = 0;
    int rooted = 0;

    for (int i = 0; i < count && n < JAI_MAX_ARGS; i++) {
        if (args[i].isSpread || args[i].name == NULL) continue;
        items[n] = OBJ_VAL(jaiStringIntern(args[i].name, strlen(args[i].name)));
        jaiGCPushRoot(items[n]);
        rooted++;
        n++;
    }

    ObjTuple *tuple = jaiTupleNew(items, n);
    uint32_t k = addConst(e, OBJ_VAL(tuple));
    jaiGCPopRoots(rooted);
    return k;
}

/* Pushes positional arguments and returns how many were emitted. */
static int emitPositionalArgs(Emitter *e, const AstArg *args, int count) {
    int n = 0;
    for (int i = 0; i < count; i++) {
        if (args[i].isSpread || args[i].name != NULL) continue;
        jaiEmitExpr(e, args[i].value);
        n++;
    }
    return n;
}

static int emitKeywordArgs(Emitter *e, const AstArg *args, int count) {
    int n = 0;
    for (int i = 0; i < count; i++) {
        if (args[i].isSpread || args[i].name == NULL) continue;
        jaiEmitExpr(e, args[i].value);
        n++;
    }
    return n;
}

/* `super.m(...)` and `super(...)`; the receiver is always slot 0. */
static void emitSuperCall(Emitter *e, AstNode *node, const char *method) {
    JaiSpan span = node->span;
    const AstArg *args = node->as.call.args;
    int count = node->as.call.argCount;

    int spreads = 0, keywords = 0;
    argKinds(args, count, &spreads, &keywords);
    if (spreads > 0 || keywords > 0) {
        CG_ERROR(e, span, E0902_INTERNAL_ERROR,
                 "`super` calls take positional arguments only");
    }

    emitGetLocal(e, 0, span);
    int argc = emitPositionalArgs(e, args, count);
    uint32_t k = nameConst(e, method);
    emitOp(e, OP_SUPER_INVOKE, span);
    emitU24(e, k, span);
    emitByte(e, (uint8_t)argc, span);
    adjust(e, -argc);
}

/* `tail` selects OP_TAIL_CALL for the plain positional case. */
void jaiEmitCall(Emitter *e, AstNode *node, bool tail) {
    JaiSpan span = node->span;
    AstNode *callee = node->as.call.callee;
    const AstArg *args = node->as.call.args;
    int count = node->as.call.argCount;

    if (callee != NULL && callee->kind == AST_SUPER) {
        emitSuperCall(e, node, "init");
        return;
    }
    if (callee != NULL && callee->kind == AST_MEMBER &&
        callee->as.member.object != NULL &&
        callee->as.member.object->kind == AST_SUPER) {
        emitSuperCall(e, node, callee->as.member.name);
        return;
    }

    int spreads = 0, keywords = 0;
    argKinds(args, count, &spreads, &keywords);

    if (count > JAI_MAX_ARGS) {
        CG_ERROR(e, span, E0602_BAD_CALL_ARITY,
                 "a call may pass at most %d arguments", JAI_MAX_ARGS);
        count = JAI_MAX_ARGS;
    }

    /* obj.m(args) fuses into INVOKE, which carries the method cache. Optional
     * chaining and argument forms that need their own opcode do not fuse. */
    bool fuseInvoke = callee != NULL && callee->kind == AST_MEMBER &&
                      spreads == 0 && keywords == 0 && !tail;
    if (fuseInvoke) {
        jaiEmitExpr(e, callee->as.member.object);
        int argc = emitPositionalArgs(e, args, count);
        jaiCgInvokeName(e, callee->as.member.name, argc, span);
        return;
    }

    /* `a?.m(x)` yields null *without calling* when `a` is null (spec §4.2), so
     * the guard has to clear the arguments and the call as well as the member
     * load. emitOptMember only jumps past the load, which would leave the null
     * in the callee slot and then call it. */
    JumpSite optSkip = {0, 0};
    bool optChain = callee != NULL && callee->kind == AST_OPT_MEMBER;
    if (optChain) {
        jaiEmitExpr(e, callee->as.member.object);
        optSkip = emitJump(e, OP_JUMP_IF_NULL, span);
        uint32_t nameK = nameConst(e, callee->as.member.name);
        uint16_t nameCache = newCache(e);
        emitOp(e, OP_GET_FIELD, span);
        emitU24(e, nameK, span);
        emitU16(e, nameCache, span);
        /* The frame has to survive to the patch target below. */
        tail = false;
    } else {
        jaiEmitExpr(e, callee);
    }

    if (spreads > 0) {
        if (keywords > 0) {
            CG_ERROR(e, span, E0902_INTERNAL_ERROR,
                     "spread and keyword arguments cannot be combined");
        }
        int firstSpread = 0;
        while (firstSpread < count && !args[firstSpread].isSpread) firstSpread++;
        int leading = 0;
        for (int i = 0; i < firstSpread; i++) {
            jaiEmitExpr(e, args[i].value);
            leading++;
        }
        emitSpreadTail(e, args, firstSpread, count, span);
        emitOp(e, OP_CALL_SPREAD, span);
        /* The operand counts every stack slot the call consumes as an argument,
         * the splat list included (BYTECODE.md §3.5) — the VM reads it as
         * `argc - 1 + list.count` and the verifier as `pops = A + 1`. */
        emitByte(e, (uint8_t)(leading + 1), span);
        adjust(e, -(leading + 1));
        if (optChain) patchJump(e, optSkip);
        return;
    }

    int argc = emitPositionalArgs(e, args, count);

    if (keywords > 0) {
        int kwCount = emitKeywordArgs(e, args, count);
        uint32_t names = keywordNameTuple(e, args, count);
        emitOp(e, OP_CALL_KW, span);
        emitByte(e, (uint8_t)argc, span);
        emitU24(e, names, span);
        adjust(e, -(argc + kwCount));
        if (optChain) patchJump(e, optSkip);
        return;
    }

    emitOp(e, tail ? OP_TAIL_CALL : OP_CALL, span);
    emitByte(e, (uint8_t)argc, span);
    adjust(e, -argc);
    if (optChain) patchJump(e, optSkip);
}

/* Closures                                                             */
/* ------------------------------------------------------------------ */

/* Every function is created with OP_CLOSURE, even one that captures nothing:
 * upvalueCount == 0 costs the VM one branch and keeps the call path uniform. */
void jaiEmitClosure(Emitter *e, AstNode *node, uint32_t fnConst,
                    JaiSpan span) {
    FunctionScope *scope = (FunctionScope *)node->as.fn.resolveInfo;
    emitOp(e, OP_CLOSURE, span);
    emitU24(e, fnConst, span);

    int count = scope != NULL ? scope->upvalueCount : 0;
    for (int i = 0; i < count; i++) {
        /* bit 0: the source is a local of the enclosing frame rather than one
         * of its upvalues. bit 1: capture by value (spec §6's `let` rule). */
        uint8_t how = (uint8_t)((scope->upvalues[i].isLocal ? 1u : 0u) |
                                (scope->upvalues[i].byValue ? 2u : 0u));
        emitByte(e, how, span);
        emitU16(e, (uint16_t)scope->upvalues[i].index, span);
    }
}

/* Compiles `node` and appends the ObjFunction to the enclosing chunk's pool. */
uint32_t jaiEmitFunctionConstant(Emitter *e, AstNode *node, uint32_t extraFlags,
                                 const char *ownerName,
                                 const AstField *initFields, int initFieldCount) {
    ObjFunction *fn = jaiEmitFunctionNode(e, node, extraFlags, ownerName,
                                          initFields, initFieldCount);
    jaiCgFinishFunction(e, fn);   /* every function is optimised, not just the body */
    uint32_t k = addConst(e, OBJ_VAL(fn));
    jaiGCPopRoot();   /* jaiEmitFunctionNode leaves the function rooted */
    return k;
}

/* Comprehensions                                                       */
/* ------------------------------------------------------------------ */

static void emitComprehensionClause(Emitter *e, AstNode *node, int clause,
                                    int containerDepth);

/* The innermost body: evaluate the element(s) and fold them into the
 * accumulator sitting at `containerDepth`. */
static void emitComprehensionBody(Emitter *e, AstNode *node, int containerDepth) {
    JaiSpan span = node->span;
    CompKind kind = node->as.comp.kind;

    if (kind == COMP_DICT) {
        jaiEmitExpr(e, node->as.comp.keyExpr);
        jaiEmitExpr(e, node->as.comp.element);
        /* The operand is the peek index of the accumulator: one iterator per
         * enclosing clause sits between it and the pushed key/value. */
        int n = depthOf(e) - 1 - containerDepth;
        emitOp(e, OP_DICT_INSERT, span);
        emitU16(e, (uint16_t)n, span);
        return;
    }

    jaiEmitExpr(e, node->as.comp.element);
    int n = depthOf(e) - 1 - containerDepth;
    emitOp(e, kind == COMP_SET ? OP_SET_ADD : OP_LIST_APPEND, span);
    emitU16(e, (uint16_t)n, span);
}

static void emitComprehensionClause(Emitter *e, AstNode *node, int clause,
                                    int containerDepth) {
    if (clause >= node->as.comp.clauseCount) {
        emitComprehensionBody(e, node, containerDepth);
        return;
    }

    AstCompClause *c = &node->as.comp.clauses[clause];
    JaiSpan span = c->span;

    int beforeIter = depthOf(e);
    jaiEmitExpr(e, c->iterable);
    emitOp(e, OP_GET_ITER, span);

    int start = here(e);
    JumpSite exit = emitJump(e, OP_FOR_ITER, span);
    adjust(e, 1);   /* the loop value, on the fall-through path */

    jaiEmitStorePattern(e, c->pattern, true);

    JumpList skips;
    jumpListInit(&skips);
    for (int i = 0; i < c->conditionCount; i++) {
        jaiEmitExpr(e, c->conditions[i]);
        JAI_VEC_PUSH(JumpSite, &skips,
                     emitJump(e, OP_JUMP_IF_FALSE, c->conditions[i]->span));
    }

    emitComprehensionClause(e, node, clause + 1, containerDepth);

    patchAll(e, &skips);
    jumpListFree(&skips);
    emitLoopBack(e, start, span);

    setDepth(e, beforeIter);   /* FOR_ITER drops the iterator on exhaustion */
    patchJump(e, exit);
}

static void emitComprehension(Emitter *e, AstNode *node) {
    JaiSpan span = node->span;
    CompKind kind = node->as.comp.kind;

    OpCode build = OP_BUILD_LIST;
    if (kind == COMP_DICT)     build = OP_BUILD_DICT;
    else if (kind == COMP_SET) build = OP_BUILD_SET;

    emitOp(e, build, span);
    emitU16(e, 0, span);
    adjust(e, 1);

    int containerDepth = depthOf(e) - 1;
    emitComprehensionClause(e, node, 0, containerDepth);

    /* A generator expression should build a lazy closure; the AST carries no
     * FunctionScope for one, so it is materialised eagerly and iterated. See
     * the note in the header of this file's report. */
    if (kind == COMP_GENERATOR) emitOp(e, OP_GET_ITER, span);
}

/* Expressions                                                          */
/* ------------------------------------------------------------------ */

static void emitLogical(Emitter *e, AstNode *node) {
    JaiSpan span = node->span;
    OpKind op = node->as.binary.op;

    jaiEmitExpr(e, node->as.binary.left);
    /* The KEEP forms leave the operand in place, so a short-circuited `and`/`or`
     * yields the operand itself rather than a synthesised bool. */
    JumpSite skip = emitJump(e, op == OPK_AND ? OP_JUMP_IF_FALSE_KEEP
                                              : OP_JUMP_IF_TRUE_KEEP, span);
    emitPop(e, span);
    jaiEmitExpr(e, node->as.binary.right);
    patchJump(e, skip);
}

/* `a < b <= c` evaluates b once: it is stashed in a temporary between the two
 * comparisons instead of being duplicated on the stack, which keeps the depth
 * at every short-circuit edge equal to "one bool". */
static void emitCompareChain(Emitter *e, AstNode *node) {
    JaiSpan span = node->span;
    int opCount = node->as.chain.opCount;
    if (opCount <= 0) {
        jaiEmitExpr(e, node->as.chain.operands[0]);
        return;
    }

    int temp = opCount > 1 ? allocTemp(e) : -1;
    JumpList shortCircuit;
    jumpListInit(&shortCircuit);
    int base = depthOf(e);

    jaiEmitExpr(e, node->as.chain.operands[0]);
    for (int i = 0; i < opCount; i++) {
        jaiEmitExpr(e, node->as.chain.operands[i + 1]);
        if (i + 1 < opCount) {
            emitOp(e, OP_SET_LOCAL, span);   /* peek: keeps the operand */
            emitU16(e, (uint16_t)temp, span);
        }
        jaiCgBinaryOp(e, node->as.chain.ops[i], span);
        if (i + 1 < opCount) {
            JAI_VEC_PUSH(JumpSite, &shortCircuit,
                         emitJump(e, OP_JUMP_IF_FALSE_KEEP, span));
            emitPop(e, span);
            emitGetLocal(e, temp, span);
        }
    }

    setDepth(e, base + 1);
    patchAll(e, &shortCircuit);
    jumpListFree(&shortCircuit);
    if (temp >= 0) freeTemps(e, 1);
}

static void emitFStringPart(Emitter *e, AstNode *part) {
    if (part != NULL && part->kind == AST_STR_LIT) {
        jaiEmitExpr(e, part);
        return;
    }
    /* Interpolation is a `str(...)` call, never a runtime eval (spec §1.7). */
    JaiSpan span = part != NULL ? part->span : fileSpan(e);
    jaiCgGetGlobal(e, "str", span);
    jaiEmitExpr(e, part);
    emitOp(e, OP_CALL, span);
    emitByte(e, 1, span);
    adjust(e, -1);
}

/* One OP_FORMAT instead of a `str` global lookup, a native call and a concat
 * per hole. The old lowering allocated two strings per interpolation — one for
 * str(x), one for the concatenation — and each carried a hash and an intern
 * probe; OP_FORMAT measures the parts and allocates the result once.
 *
 * The literal mask records which parts came from the source text rather than a
 * hole, because a program that rebinds the global `str` still gets its own
 * function called on the holes and only on the holes. */
static void emitFString(Emitter *e, AstNode *node) {
    JaiSpan span = node->span;
    int count = node->as.fstring.partCount;
    AstNode **parts = node->as.fstring.parts;

    if (count == 0) {
        jaiCgConstValue(e, OBJ_VAL(jaiStringIntern("", 0)), span);
        return;
    }
    if (count == 1 && parts[0] != NULL && parts[0]->kind == AST_STR_LIT) {
        jaiEmitExpr(e, parts[0]);
        return;
    }

    if (count <= JAI_FMT_MAX_PARTS) {
        uint32_t litmask = 0;
        for (int i = 0; i < count; i++) {
            if (parts[i] != NULL && parts[i]->kind == AST_STR_LIT) {
                litmask |= 1u << i;
            }
            jaiEmitExpr(e, parts[i]);
        }
        emitOp(e, OP_FORMAT, span);
        emitByte(e, (uint8_t)count, span);
        emitU24(e, litmask, span);
        emitU24(e, nameConst(e, "str"), span);
        emitU16(e, newCache(e), span);
        adjust(e, -count + 1);
        return;
    }

    /* More holes than the mask holds: fall back to one join. */
    jaiCgConstValue(e, OBJ_VAL(jaiStringIntern("", 0)), span);
    for (int i = 0; i < count; i++) emitFStringPart(e, parts[i]);
    emitOp(e, OP_BUILD_LIST, span);
    emitU16(e, (uint16_t)count, span);
    adjust(e, -count + 1);
    jaiCgInvokeName(e, "join", 1, span);
}

static void emitSequenceLiteral(Emitter *e, AstNode *node, OpCode build) {
    JaiSpan span = node->span;
    int count = node->as.sequence.count;
    if (count > UINT16_MAX) {
        CG_ERROR(e, span, E0902_INTERNAL_ERROR,
                 "a literal may hold at most %d elements", UINT16_MAX);
        count = UINT16_MAX;
    }
    for (int i = 0; i < count; i++) jaiEmitExpr(e, node->as.sequence.items[i]);
    emitOp(e, build, span);
    emitU16(e, (uint16_t)count, span);
    adjust(e, -count + 1);
}

static void emitMember(Emitter *e, AstNode *node) {
    JaiSpan span = node->span;
    AstNode *object = node->as.member.object;

    if (object != NULL && object->kind == AST_SUPER) {
        emitGetLocal(e, 0, span);
        emitOp(e, OP_GET_SUPER, span);
        emitU24(e, nameConst(e, node->as.member.name), span);
        return;
    }

    jaiEmitExpr(e, object);
    uint32_t k = nameConst(e, node->as.member.name);
    uint16_t cache = newCache(e);
    emitOp(e, OP_GET_FIELD, span);
    emitU24(e, k, span);
    emitU16(e, cache, span);
}

static void emitOptMember(Emitter *e, AstNode *node) {
    JaiSpan span = node->span;
    jaiEmitExpr(e, node->as.member.object);

    /* On null the jump lands past the access with the null still in place,
     * which is exactly the value `x?.y` yields. */
    JumpSite isNull = emitJump(e, OP_JUMP_IF_NULL, span);
    uint32_t k = nameConst(e, node->as.member.name);
    uint16_t cache = newCache(e);
    emitOp(e, OP_GET_FIELD, span);
    emitU24(e, k, span);
    emitU16(e, cache, span);
    patchJump(e, isNull);
}

static void emitSlice(Emitter *e, AstNode *node) {
    JaiSpan span = node->span;
    jaiEmitExpr(e, node->as.slice.object);

    uint8_t flags = 0;
    int pushed = 0;
    if (node->as.slice.start != NULL) {
        jaiEmitExpr(e, node->as.slice.start);
        flags |= 1;
        pushed++;
    }
    if (node->as.slice.stop != NULL) {
        jaiEmitExpr(e, node->as.slice.stop);
        flags |= 2;
        pushed++;
    }
    if (node->as.slice.step != NULL) {
        jaiEmitExpr(e, node->as.slice.step);
        flags |= 4;
        pushed++;
    }
    emitOp(e, OP_GET_SLICE, span);
    emitByte(e, flags, span);
    adjust(e, -pushed);
}

/* One arm of a conditional, emitted for its value.
 *
 * A branch is not always an expression node: a block yields its tail, and an
 * `elif` chain that the parser built in statement position arrives as AST_IF.
 * `match` can appear here for the same reason. */
static void emitBranchValue(Emitter *e, AstNode *branch, JaiSpan span) {
    if (branch == NULL) {
        emitOp(e, OP_NULL, span);
        return;
    }
    switch (branch->kind) {
    case AST_BLOCK:
        jaiEmitBlockValue(e, branch);
        return;
    case AST_MATCH:
        jaiEmitMatch(e, branch, true);
        return;
    case AST_IF:
        /* A trailing `elif` with no `else` has no value on the path not taken;
         * emitTernary supplies null for it. */
        emitTernary(e, branch->as.conditional.cond,
                    branch->as.conditional.thenBranch,
                    branch->as.conditional.elseBranch, branch->span);
        return;
    default:
        jaiEmitExpr(e, branch);
        return;
    }
}

static void emitTernary(Emitter *e, AstNode *cond, AstNode *thenExpr,
                        AstNode *elseExpr, JaiSpan span) {
    jaiEmitExpr(e, cond);
    JumpSite toElse = emitJump(e, OP_JUMP_IF_FALSE, span);

    int base = depthOf(e);
    emitBranchValue(e, thenExpr, span);
    JumpSite toEnd = emitJump(e, OP_JUMP, span);

    setDepth(e, base);
    patchJump(e, toElse);
    emitBranchValue(e, elseExpr, span);

    setDepth(e, base + 1);
    patchJump(e, toEnd);
}

static void emitCoalesce(Emitter *e, AstNode *node) {
    JaiSpan span = node->span;
    jaiEmitExpr(e, node->as.coalesce.left);

    JumpSite isNull = emitJump(e, OP_JUMP_IF_NULL, span);
    JumpSite done = emitJump(e, OP_JUMP, span);

    int base = depthOf(e) - 1;
    setDepth(e, base + 1);
    patchJump(e, isNull);
    emitPop(e, span);
    jaiEmitExpr(e, node->as.coalesce.right);

    setDepth(e, base + 1);
    patchJump(e, done);
}

static void emitBlockValueContents(Emitter *e, AstNode *node);

/* A block used for its value. The exit close comes after the value is on the
 * stack: the value itself may be a closure over one of the block's bindings. */
void jaiEmitBlockValue(Emitter *e, AstNode *node) {
    int base = node->as.block.captureBase;
    int saved = enterCloseScope(e, base);
    emitBlockValueContents(e, node);
    emitCloseScope(e, base, node->span);
    e->fn->closeBase = saved;
}

/* Every statement but the last is a statement, and a trailing expression
 * statement is the block's result. */
static void emitBlockValueContents(Emitter *e, AstNode *node) {
    JaiSpan span = node->span;
    int count = node->as.block.count;
    int last = count - 1;

    while (last >= 0 && node->as.block.stmts[last] == NULL) last--;

    for (int i = 0; i < last; i++) jaiEmitStmt(e, node->as.block.stmts[i]);

    if (last < 0) {
        emitOp(e, OP_NULL, span);
        return;
    }
    AstNode *tail = node->as.block.stmts[last];
    if (tail->kind == AST_EXPR_STMT) {
        jaiEmitExpr(e, tail->as.exprStmt.expr);
        return;
    }

    /* `if` and `match` in statement position parse as AST_IF / AST_MATCH, but
     * at the end of a block that is being read for its value they *are* that
     * value — the checker already types them so. Emitting them as statements
     * and pushing null is why a match arm written
     *     => { let d = n * 2; if gate { d } else { null } }
     * evaluated to null on every path. An `if` with no `else` is left alone:
     * it has no value on the path not taken, so the block yields null as
     * before. */
    if ((tail->kind == AST_IF && tail->as.conditional.elseBranch != NULL) ||
        tail->kind == AST_MATCH) {
        emitBranchValue(e, tail, tail->span);
        return;
    }

    jaiEmitStmt(e, tail);
    emitOp(e, OP_NULL, span);
}

void jaiEmitExpr(Emitter *e, AstNode *node) {
    if (node == NULL) {
        emitOp(e, OP_NULL, fileSpan(e));
        return;
    }
    JaiSpan span = node->span;

    switch (node->kind) {
    case AST_INT_LIT:
        jaiCgInt(e, node->as.intLit, span);
        break;
    case AST_FLOAT_LIT:
        jaiCgConstValue(e, FLOAT_VAL(node->as.floatLit), span);
        break;
    case AST_STR_LIT:
        jaiCgConstValue(e, OBJ_VAL(jaiStringIntern(node->as.strLit.chars,
                                                   node->as.strLit.length)),
                        span);
        break;
    case AST_BOOL_LIT:
        emitOp(e, node->as.boolLit ? OP_TRUE : OP_FALSE, span);
        break;
    case AST_NULL_LIT:
        emitOp(e, OP_NULL, span);
        break;

    case AST_FSTRING:
        emitFString(e, node);
        break;

    case AST_IDENT: {
        Symbol *sym = node->as.ident.symbol;
        if (sym != NULL && sym->isConstFolded && sym->constValue != NULL) {
            jaiEmitExpr(e, sym->constValue);
            break;
        }
        jaiCgLoadSymbol(e, sym, node->as.ident.name, span);
        break;
    }

    case AST_SELF:
        emitGetLocal(e, 0, span);
        break;

    case AST_SUPER:
        /* Bare `super` only reaches here after a resolver error; slot 0 keeps
         * the chunk well-formed. */
        emitGetLocal(e, 0, span);
        break;

    case AST_LIST_LIT:
        emitSequenceLiteral(e, node, OP_BUILD_LIST);
        break;
    case AST_SET_LIT:
        emitSequenceLiteral(e, node, OP_BUILD_SET);
        break;
    case AST_TUPLE_LIT:
        emitSequenceLiteral(e, node, OP_BUILD_TUPLE);
        break;

    case AST_DICT_LIT: {
        int count = node->as.dict.count;
        for (int i = 0; i < count; i++) {
            jaiEmitExpr(e, node->as.dict.keys[i]);
            jaiEmitExpr(e, node->as.dict.values[i]);
        }
        emitOp(e, OP_BUILD_DICT, span);
        emitU16(e, (uint16_t)count, span);
        adjust(e, -2 * count + 1);
        break;
    }

    case AST_UNARY: {
        jaiEmitExpr(e, node->as.unary.operand);
        OpCode code = jaiCgOpcodeForOp(node->as.unary.op);
        if (code == OP_NOP) {
            CG_ERROR(e, span, E0902_INTERNAL_ERROR,
                     "no opcode for unary operator `%s`",
                     jaiOpKindText(node->as.unary.op));
            break;
        }
        emitOp(e, code, span);
        break;
    }

    case AST_BINARY:
        jaiEmitExpr(e, node->as.binary.left);
        jaiEmitExpr(e, node->as.binary.right);
        jaiCgBinaryOp(e, node->as.binary.op, span);
        break;

    case AST_LOGICAL:
        emitLogical(e, node);
        break;

    case AST_COMPARE_CHAIN:
        emitCompareChain(e, node);
        break;

    case AST_TERNARY:
        emitTernary(e, node->as.ternary.cond, node->as.ternary.thenExpr,
                    node->as.ternary.elseExpr, span);
        break;

    case AST_IF_EXPR:
        emitTernary(e, node->as.conditional.cond, node->as.conditional.thenBranch,
                    node->as.conditional.elseBranch, span);
        break;

    case AST_COALESCE:
        emitCoalesce(e, node);
        break;

    case AST_CALL:
        jaiEmitCall(e, node, false);
        break;

    case AST_INDEX:
        jaiEmitExpr(e, node->as.index.object);
        /* Generics are erased (spec §6.1): `Box[int]` *is* `Box`, and the type
         * arguments were never resolved as values, so there is nothing to
         * evaluate and nothing to subscript. */
        if (node->as.index.typeArgs) break;
        jaiEmitExpr(e, node->as.index.index);
        emitOp(e, OP_GET_INDEX, span);
        break;

    case AST_SLICE:
        emitSlice(e, node);
        break;

    case AST_MEMBER:
        emitMember(e, node);
        break;

    case AST_OPT_MEMBER:
        emitOptMember(e, node);
        break;

    case AST_LAMBDA:
    case AST_ANON_FN: {
        uint32_t k = jaiEmitFunctionConstant(e, node, 0, NULL, NULL, 0);
        jaiEmitClosure(e, node, k, span);
        break;
    }

    case AST_COMPREHENSION:
        emitComprehension(e, node);
        break;

    case AST_RANGE:
        jaiEmitExpr(e, node->as.range.start);
        jaiEmitExpr(e, node->as.range.stop);
        emitOp(e, OP_BUILD_RANGE, span);
        emitByte(e, node->as.range.inclusive ? 1 : 0, span);
        break;

    case AST_MATCH_EXPR:
        jaiEmitMatch(e, node, true);
        break;

    case AST_CAST:
        jaiEmitExpr(e, node->as.cast.operand);
        if (node->as.cast.widen) {
            /* An int the checker proved is an int, on its way into a float:
             * one conversion, no test (spec §2.2). */
            emitOp(e, OP_TO_FLOAT, span);
        } else {
            emitOp(e, OP_TYPE_GUARD, span);
            emitU24(e, jaiCgTypeGuardConst(e, node->as.cast.target), span);
        }
        break;

    case AST_AWAIT:
        /* `await` is a suspension point in std.async, dispatched through the
         * awaitable's __await__ method; there is no dedicated opcode. */
        jaiEmitExpr(e, node->as.wrap.operand);
        jaiCgInvokeName(e, "__await__", 0, span);
        break;

    case AST_YIELD:
        CG_ERROR(e, span, E0902_INTERNAL_ERROR,
                 "`yield` has no opcode in this bytecode version");
        jaiEmitExpr(e, node->as.wrap.operand);
        break;

    case AST_THROW_EXPR:
        jaiEmitExpr(e, node->as.ret.value);
        emitOp(e, OP_THROW, span);
        /* `throw` has type `never`; nothing after this is reachable, but the
         * join arithmetic downstream still expects a value here. */
        adjust(e, 1);
        break;

    case AST_BLOCK:
        jaiEmitBlockValue(e, node);
        break;

    default:
        CG_ERROR(e, span, E0902_INTERNAL_ERROR,
                 "`%s` is not an expression", jaiAstKindName(node->kind));
        emitOp(e, OP_NULL, span);
        break;
    }
}

/* Assignment                                                           */
/* ------------------------------------------------------------------ */

void jaiEmitAssign(Emitter *e, AstNode *node) {
    JaiSpan span = node->span;
    AstNode *target = node->as.assign.target;
    AstNode *value = node->as.assign.value;
    bool compound = node->as.assign.isCompound;
    OpKind op = node->as.assign.op;

    if (target == NULL) {
        CG_ERROR(e, span, E0109_INVALID_ASSIGN_TARGET, "missing assignment target");
        return;
    }

    switch (target->kind) {
    case AST_IDENT: {
        Symbol *sym = target->as.ident.symbol;
        if (compound) {
            jaiCgLoadSymbol(e, sym, target->as.ident.name, span);
            jaiEmitExpr(e, value);
            jaiCgBinaryOp(e, op, span);
        } else {
            jaiEmitExpr(e, value);
        }
        jaiCgStoreSymbol(e, sym, target->as.ident.name, span, false);
        break;
    }

    case AST_MEMBER: {
        uint32_t k = nameConst(e, target->as.member.name);
        jaiEmitExpr(e, target->as.member.object);
        if (compound) {
            emitOp(e, OP_DUP, span);
            uint16_t readCache = newCache(e);
            emitOp(e, OP_GET_FIELD, span);
            emitU24(e, k, span);
            emitU16(e, readCache, span);
            jaiEmitExpr(e, value);
            jaiCgBinaryOp(e, op, span);
        } else {
            jaiEmitExpr(e, value);
        }
        uint16_t writeCache = newCache(e);
        emitOp(e, OP_SET_FIELD, span);
        emitU24(e, k, span);
        emitU16(e, writeCache, span);
        break;
    }

    case AST_INDEX:
        jaiEmitExpr(e, target->as.index.object);
        jaiEmitExpr(e, target->as.index.index);
        if (compound) {
            emitOp(e, OP_DUP2, span);
            emitOp(e, OP_GET_INDEX, span);
            jaiEmitExpr(e, value);
            jaiCgBinaryOp(e, op, span);
        } else {
            jaiEmitExpr(e, value);
        }
        emitOp(e, OP_SET_INDEX, span);
        break;

    case AST_SLICE: {
        if (compound) {
            CG_ERROR(e, span, E0109_INVALID_ASSIGN_TARGET,
                     "compound assignment to a slice is not supported");
        }
        jaiEmitExpr(e, target->as.slice.object);
        uint8_t flags = 0;
        int pushed = 0;
        if (target->as.slice.start != NULL) {
            jaiEmitExpr(e, target->as.slice.start);
            flags |= 1;
            pushed++;
        }
        if (target->as.slice.stop != NULL) {
            jaiEmitExpr(e, target->as.slice.stop);
            flags |= 2;
            pushed++;
        }
        if (target->as.slice.step != NULL) {
            jaiEmitExpr(e, target->as.slice.step);
            flags |= 4;
            pushed++;
        }
        jaiEmitExpr(e, value);
        emitOp(e, OP_SET_SLICE, span);
        emitByte(e, flags, span);
        adjust(e, -(pushed + 2));
        break;
    }

    default:
        if (compound) {
            CG_ERROR(e, span, E0109_INVALID_ASSIGN_TARGET,
                     "compound assignment needs a single target");
        }
        jaiEmitExpr(e, value);
        jaiEmitStorePattern(e, target, false);
        break;
    }
}
