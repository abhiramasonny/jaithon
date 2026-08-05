/* codegen_stmt.c — statements and control flow (spec §5).
 *
 * Every emitter here leaves the operand stack as it found it. The hard part is
 * not the shapes but the exits: `break`, `continue` and `return` all have to
 * unwind whatever the cursor is standing inside — open blocks whose by-
 * reference captures must be closed, and `finally` blocks that get an inline
 * copy on every non-local path out. LoopCtx and FinallyCtx below are what
 * makes that decidable at emission time.
 */
#include "codegen_internal.h"

/* Loops                                                                */
/* ------------------------------------------------------------------ */

static LoopCtx *findLoop(Emitter *e, const char *label, JaiSpan span) {
    LoopCtx *loop = e->fn->loops;
    if (label == NULL) {
        if (loop == NULL) {
            CG_ERROR(e, span, E0203_BREAK_OUTSIDE_LOOP,
                     "`break`/`continue` outside a loop");
        }
        return loop;
    }
    for (; loop != NULL; loop = loop->enclosing) {
        if (loop->label != NULL && strcmp(loop->label, label) == 0) return loop;
    }
    CG_ERROR(e, span, E0202_UNDEFINED_LABEL, "no enclosing loop labelled `%s`",
             label);
    return NULL;
}

/* Inline copies of every `finally` entered inside `loop`, innermost first.
 * The exception path uses the exception table instead; only the explicit
 * break/continue/return edges need the duplication. */
static void emitFinallyChainForLoop(Emitter *e, const LoopCtx *loop) {
    FnCtx *f = e->fn;
    int limit = loop != NULL ? loop->finallyDepth : 0;
    int index = f->finallyDepth;
    for (FinallyCtx *fin = f->finallys; fin != NULL && index > limit;
         fin = fin->enclosing, index--) {
        jaiEmitStmt(e, fin->block);
    }
}

static void emitFinallyChainForReturn(Emitter *e) {
    for (FinallyCtx *fin = e->fn->finallys; fin != NULL; fin = fin->enclosing) {
        jaiEmitStmt(e, fin->block);
    }
}

/* try / catch / finally                                                */
/* ------------------------------------------------------------------ */

static void addException(Emitter *e, uint32_t start, uint32_t end,
                         uint32_t handler, uint32_t typeConst) {
    ExceptionEntry entry;
    entry.start = start;
    entry.end = end;
    entry.handler = handler;
    entry.typeConst = typeConst;
    JAI_VEC_PUSH(ExceptionEntry, &e->fn->exceptions, entry);
}

/* OP_PUSH_HANDLER carries the branch offset *and* a type constant, so its jump
 * site ends after both operands — that is where the VM resolves it from. */
static JumpSite emitPushHandler(Emitter *e, uint32_t typeConst, JaiSpan span) {
    emitOp(e, OP_PUSH_HANDLER, span);
    JumpSite site;
    site.operand = here(e);
    emitI16(e, 0, span);
    emitU24(e, typeConst, span);
    site.end = here(e);
    return site;
}

/* More clauses than this on one `try` cannot be dispatched dynamically; the
 * parser has no such limit, so the tail is reported rather than mis-compiled. */
#define MAX_DYNAMIC_CATCH_CLAUSES 32

/* Dispatch to the catch clause whose type matches, or re-raise. Entered by the
 * unwinder with the stack restored to `base`; the clause bodies follow. */
static void emitCatchDispatch(Emitter *e, AstNode *node, int base,
                              JumpList *done) {
    FnCtx *f = e->fn;
    int catchCount = node->as.tryStmt.catchCount;
    JumpList entries[MAX_DYNAMIC_CATCH_CLAUSES];
    int tested = catchCount;
    if (tested > MAX_DYNAMIC_CATCH_CLAUSES) {
        CG_ERROR(e, node->span, E0902_INTERNAL_ERROR,
                 "a `try` inside a loop may have at most %d catch clauses",
                 MAX_DYNAMIC_CATCH_CLAUSES);
        tested = MAX_DYNAMIC_CATCH_CLAUSES;
    }

    setDepth(e, base);
    for (int i = 0; i < tested; i++) {
        AstCatch *clause = &node->as.tryStmt.catches[i];
        jumpListInit(&entries[i]);
        if (clause->typeCount == 0) {
            JAI_VEC_PUSH(JumpSite, &entries[i], emitJump(e, OP_JUMP, clause->span));
            continue;
        }
        for (int t = 0; t < clause->typeCount; t++) {
            emitOp(e, OP_MATCH_EXC, clause->span);
            emitU24(e, jaiCgTypeGuardConst(e, clause->types[t]), clause->span);
            JAI_VEC_PUSH(JumpSite, &entries[i],
                         emitJump(e, OP_JUMP_IF_TRUE, clause->span));
        }
    }
    /* No clause claimed it: the exception keeps going up. */
    emitOp(e, OP_RERAISE, node->span);

    for (int i = 0; i < tested; i++) {
        AstCatch *clause = &node->as.tryStmt.catches[i];
        setDepth(e, base);
        patchAll(e, &entries[i]);
        jumpListFree(&entries[i]);

        emitOp(e, OP_GET_EXC, clause->span);
        if (clause->symbol != NULL || clause->name != NULL) {
            jaiCgStoreSymbol(e, clause->symbol, clause->name, clause->span,
                             clause->symbol == NULL);
        } else {
            emitPop(e, clause->span);
        }

        f->protectDepth++;
        jaiEmitStmt(e, clause->body);
        f->protectDepth--;
        JAI_VEC_PUSH(JumpSite, done, emitJump(e, OP_JUMP, clause->span));
    }
}

static void emitTry(Emitter *e, AstNode *node) {
    JaiSpan span = node->span;
    FnCtx *f = e->fn;
    AstNode *finallyBlock = node->as.tryStmt.finallyBlock;
    int base = depthOf(e);
    /* An exception-table handler is entered with every temporary of the frame
     * already discarded, which is only right when the try sits at depth 0.
     * Inside a for loop the iterator is a live temporary and dropping it would
     * leave the loop reading garbage, so there the handler is registered with
     * PUSH_HANDLER, which records the stack top to restore.
     *
     * A `finally` forces the same choice for a different reason: it registers
     * a catch-all dynamic handler, and the unwinder tries dynamic handlers
     * before the table, so a table-based catch clause would never be reached.
     * Pushing the catch dispatcher on top of the finally restores the order
     * the language asks for — catch first, finally after. */
    bool dynamic = node->as.tryStmt.catchCount > 0 &&
                   (base > 0 || finallyBlock != NULL);

    JumpSite finallyEntry = {0, 0};
    if (finallyBlock != NULL) {
        /* Registers the finally target for the unwinder; dispatch itself is
         * driven by the exception table below. */
        finallyEntry = emitJump(e, OP_PUSH_FINALLY, span);
    }

    FinallyCtx finCtx;
    if (finallyBlock != NULL) {
        finCtx.enclosing = f->finallys;
        finCtx.block = finallyBlock;
        finCtx.loopDepth = f->loopDepth;
        f->finallys = &finCtx;
        f->finallyDepth++;
    }

    JumpSite dispatchEntry = {0, 0};
    if (dynamic) dispatchEntry = emitPushHandler(e, nameConst(e, "any"), span);

    /* An exception skips every block close inside the protected region: the
     * unwinder jumps straight to a handler in this same frame, and the catch
     * binding and the handler's own locals are then handed the slots the body's
     * bindings held. The watermark reports what the region opened, and closing
     * at it on entry to every handler is the unwind path's version of the block
     * close. Slots below it are left alone — they are still live. */
    int savedDeep = f->deepCloseBase;
    f->deepCloseBase = -1;

    int protectedStart = here(e);
    f->protectDepth++;
    jaiEmitStmt(e, node->as.tryStmt.body);
    f->protectDepth--;
    if (dynamic) emitOp(e, OP_POP_HANDLER, span);
    int protectedEnd = here(e);
    int bodyDeep = f->deepCloseBase;

    JumpList done;
    jumpListInit(&done);
    JAI_VEC_PUSH(JumpSite, &done, emitJump(e, OP_JUMP, span));

    if (dynamic) {
        patchJump(e, dispatchEntry);
        emitCloseScope(e, bodyDeep, span);
        emitCatchDispatch(e, node, base, &done);
    }

    for (int i = 0; !dynamic && i < node->as.tryStmt.catchCount; i++) {
        AstCatch *clause = &node->as.tryStmt.catches[i];
        int handler = here(e);
        setDepth(e, base);
        emitCloseScope(e, bodyDeep, clause->span);

        if (clause->typeCount == 0) {
            addException(e, (uint32_t)protectedStart, (uint32_t)protectedEnd,
                         (uint32_t)handler, UINT32_MAX);
        } else {
            for (int t = 0; t < clause->typeCount; t++) {
                uint32_t k = jaiCgTypeGuardConst(e, clause->types[t]);
                addException(e, (uint32_t)protectedStart, (uint32_t)protectedEnd,
                             (uint32_t)handler, k);
            }
        }

        emitOp(e, OP_GET_EXC, clause->span);
        if (clause->symbol != NULL || clause->name != NULL) {
            jaiCgStoreSymbol(e, clause->symbol, clause->name, clause->span,
                             clause->symbol == NULL);
        } else {
            emitPop(e, clause->span);
        }

        f->protectDepth++;
        jaiEmitStmt(e, clause->body);
        f->protectDepth--;
        JAI_VEC_PUSH(JumpSite, &done, emitJump(e, OP_JUMP, clause->span));
    }

    int catchesEnd = here(e);

    if (finallyBlock != NULL) {
        f->finallys = finCtx.enclosing;
        f->finallyDepth--;

        setDepth(e, base);
        patchAll(e, &done);
        patchJump(e, finallyEntry);

        /* One catch-all entry spanning the body *and* the catch handlers routes
         * every propagating exception here; it is registered after the typed
         * catch entries so the VM's first-match scan still prefers them. */
        int finallyStart = here(e);
        addException(e, (uint32_t)protectedStart, (uint32_t)catchesEnd,
                     (uint32_t)finallyStart, UINT32_MAX);

        /* The entry covers the catch handlers too, so the close here is the
         * whole region's, not just the body's. On the normal edge it is a
         * no-op: those scopes closed themselves on the way out. */
        emitCloseScope(e, f->deepCloseBase, span);

        /* Both the normal edge above and the unwinder land here. END_FINALLY
         * resumes the unwind when one is pending and otherwise falls through. */
        f->protectDepth++;
        jaiEmitStmt(e, finallyBlock);
        f->protectDepth--;
        emitOp(e, OP_END_FINALLY, span);
    } else {
        setDepth(e, base);
        patchAll(e, &done);
    }

    /* Fold the region back into the enclosing watermark: an outer `try` has to
     * close what this one opened as well. */
    if (savedDeep >= 0 && (f->deepCloseBase < 0 || savedDeep < f->deepCloseBase)) {
        f->deepCloseBase = savedDeep;
    }

    jumpListFree(&done);
    setDepth(e, base);
}

/* defer                                                                */
/* ------------------------------------------------------------------ */

/* A deferred block runs later but in *this* frame: the VM pushes it with the
 * defining frame's slots and closure, so its body keeps the enclosing
 * function's slot numbering and upvalue indices and needs no capture list. */
static void emitDefer(Emitter *e, AstNode *node) {
    JaiSpan span = node->span;
    FnCtx *parent = e->fn;

    ObjFunction *thunk = jaiFunctionNew();
    jaiGCPushRoot(OBJ_VAL(thunk));

    thunk->module = e->module;
    thunk->name = jaiStringIntern("defer", 5);
    thunk->qualifiedName = thunk->name;
    thunk->upvalueCount = parent->fn->upvalueCount;
    jaiChunkInit(&thunk->chunk, e->fileId);

    FnCtx ctx;
    memset(&ctx, 0, sizeof ctx);
    ctx.closeBase = -1;       /* memset would read as "close at slot 0" */
    ctx.deepCloseBase = -1;
    ctx.enclosing = parent;
    ctx.fn = thunk;
    ctx.scope = parent->scope;
    ctx.decl = parent->decl;
    ctx.nextTemp = parent->nextTemp;
    ctx.maxSlot = parent->maxSlot;
    ctx.protectDepth = 1;   /* never tail-call out of a deferred block */
    JAI_VEC_INIT(&ctx.exceptions);

    e->fn = &ctx;
    jaiEmitStmt(e, node->as.defer.body);
    emitOp(e, OP_RETURN_NULL, span);
    e->fn = parent;

    /* The thunk borrows the parent's window, so the parent must be wide enough
     * for whatever the thunk touches. */
    if (ctx.maxSlot > parent->maxSlot) parent->maxSlot = ctx.maxSlot;
    if (ctx.maxStackDepth > parent->maxStackDepth) {
        parent->maxStackDepth = ctx.maxStackDepth;
    }

    thunk->maxSlots = (uint16_t)(ctx.maxSlot + ctx.maxStackDepth);
    if (ctx.exceptions.count > 0) {
        thunk->exceptionCount = (uint16_t)ctx.exceptions.count;
        thunk->exceptions = JAI_ALLOC(ExceptionEntry, ctx.exceptions.count);
        memcpy(thunk->exceptions, ctx.exceptions.data,
               sizeof(ExceptionEntry) * (size_t)ctx.exceptions.count);
    }
    JAI_VEC_FREE(ExceptionEntry, &ctx.exceptions);

    jaiCgFinishFunction(e, thunk);
    uint32_t k = addConst(e, OBJ_VAL(thunk));
    jaiGCPopRoot();

    emitOp(e, OP_PUSH_DEFER, span);
    emitU24(e, k, span);
    parent->hasDefer = true;
}

/* Loops                                                                */
/* ------------------------------------------------------------------ */

static void enterLoop(Emitter *e, LoopCtx *loop, const char *label,
                      int continueTarget, int continueDepth, int breakDepth,
                      int captureBase) {
    loop->enclosing = e->fn->loops;
    loop->label = label;
    loop->continueTarget = continueTarget;
    loop->continueDepth = continueDepth;
    loop->breakDepth = breakDepth;
    loop->finallyDepth = e->fn->finallyDepth;
    loop->captureBase = captureBase;
    jumpListInit(&loop->breaks);
    /* A `return` out of the loop and an exception leaving an enclosing `try`
     * both have to close what the loop opened; the loop chain answers the
     * first, the watermark the second. */
    if (captureBase >= 0 &&
        (e->fn->deepCloseBase < 0 || captureBase < e->fn->deepCloseBase)) {
        e->fn->deepCloseBase = captureBase;
    }
    e->fn->loops = loop;
    e->fn->loopDepth++;
}

static void leaveLoop(Emitter *e, LoopCtx *loop) {
    setDepth(e, loop->breakDepth);
    patchAll(e, &loop->breaks);
    jumpListFree(&loop->breaks);
    e->fn->loops = loop->enclosing;
    e->fn->loopDepth--;
}

/* Per-iteration bindings (spec §5.2). The resolver hands over the lowest slot
 * that a closure captures *by reference* anywhere inside the loop; closing at
 * it detaches every open upvalue of that iteration from the frame, so the next
 * iteration's OP_CLOSURE allocates fresh cells instead of aliasing the same
 * ones. Slots above it belong to blocks nested in the body, which are equally
 * per-iteration — which is also why the body block does not emit its own close
 * here: the loop base is never higher, so the two would close the same slots
 * twice at the same offset. See emitLoopBody. */
static void emitCloseIteration(Emitter *e, int captureBase, JaiSpan span) {
    emitCloseScope(e, captureBase, span);
}

/* The statements of a loop body, without the block's own exit close: every path
 * out of the body — fall-through, `break`, `continue` — already ends an
 * iteration and closes at the loop's base, which covers the body's. */
static void emitLoopBody(Emitter *e, AstNode *body) {
    if (body == NULL || body->kind != AST_BLOCK) {
        jaiEmitStmt(e, body);
        return;
    }
    for (int i = 0; i < body->as.block.count; i++) {
        jaiEmitStmt(e, body->as.block.stmts[i]);
    }
}

/* A labelled break/continue leaves several loop levels at once, ending an
 * iteration of each of them; one close at the lowest of their bases covers the
 * lot, since closing is "this slot and everything above". */
static int closeBaseThrough(Emitter *e, const LoopCtx *target) {
    int base = -1;
    for (const LoopCtx *l = e->fn->loops; l != NULL; l = l->enclosing) {
        if (l->captureBase >= 0 && (base < 0 || l->captureBase < base)) {
            base = l->captureBase;
        }
        if (l == target) break;
    }
    return base;
}

static void emitWhile(Emitter *e, AstNode *node) {
    JaiSpan span = node->span;
    int base = depthOf(e);
    int start = here(e);

    jaiEmitExpr(e, node->as.loop.cond);
    JumpSite exit = emitJump(e, OP_JUMP_IF_FALSE, span);

    LoopCtx loop;
    enterLoop(e, &loop, node->as.loop.label, start, base, base,
              node->as.loop.captureBase);
    emitLoopBody(e, node->as.loop.body);
    emitCloseIteration(e, loop.captureBase, span);
    emitLoopBack(e, start, span);

    setDepth(e, base);
    patchJump(e, exit);
    leaveLoop(e, &loop);
}

static void emitInfiniteLoop(Emitter *e, AstNode *node) {
    JaiSpan span = node->span;
    int base = depthOf(e);
    int start = here(e);

    LoopCtx loop;
    enterLoop(e, &loop, node->as.loop.label, start, base, base,
              node->as.loop.captureBase);
    emitLoopBody(e, node->as.loop.body);
    emitCloseIteration(e, loop.captureBase, span);
    emitLoopBack(e, start, span);
    leaveLoop(e, &loop);
}

static void emitFor(Emitter *e, AstNode *node) {
    JaiSpan span = node->span;
    int base = depthOf(e);

    jaiEmitExpr(e, node->as.forLoop.iterable);
    emitOp(e, OP_GET_ITER, span);

    int start = here(e);
    JumpSite exit = emitJump(e, OP_FOR_ITER, span);
    adjust(e, 1);   /* the element, on the fall-through path */

    /* continue keeps the iterator (depth base+1); break drops it (depth base). */
    LoopCtx loop;
    enterLoop(e, &loop, node->as.forLoop.label, start, base + 1, base,
              node->as.forLoop.captureBase);

    jaiEmitStorePattern(e, node->as.forLoop.pattern, true);
    emitLoopBody(e, node->as.forLoop.body);
    emitCloseIteration(e, loop.captureBase, span);
    emitLoopBack(e, start, span);

    setDepth(e, base);   /* FOR_ITER pops the iterator when it is exhausted */
    patchJump(e, exit);
    leaveLoop(e, &loop);
}

static void emitBreak(Emitter *e, AstNode *node) {
    JaiSpan span = node->span;
    LoopCtx *loop = findLoop(e, node->as.jump.label, span);
    if (loop == NULL) return;

    emitFinallyChainForLoop(e, loop);
    int saved = depthOf(e);
    emitPopN(e, saved - loop->breakDepth, span);
    /* A closure made in the final iteration escapes the loop; without this its
     * upvalue would still alias the slot, which the next declaration in the
     * enclosing scope reuses. */
    emitCloseIteration(e, closeBaseThrough(e, loop), span);
    JAI_VEC_PUSH(JumpSite, &loop->breaks, emitJump(e, OP_JUMP, span));
    setDepth(e, saved);   /* code after `break` is unreachable but must parse */
}

static void emitContinue(Emitter *e, AstNode *node) {
    JaiSpan span = node->span;
    LoopCtx *loop = findLoop(e, node->as.jump.label, span);
    if (loop == NULL) return;

    emitFinallyChainForLoop(e, loop);
    int saved = depthOf(e);
    emitPopN(e, saved - loop->continueDepth, span);
    emitCloseIteration(e, closeBaseThrough(e, loop), span);
    emitLoopBack(e, loop->continueTarget, span);
    setDepth(e, saved);
}

/* return                                                               */
/* ------------------------------------------------------------------ */

/* A call in tail position reuses the frame, but only when nothing in this frame
 * still has to run afterwards: defers, an enclosing try/finally, and generator
 * frames all rule it out. */
static bool canTailCall(Emitter *e, AstNode *value) {
    if (!e->opts->emitTailCalls) return false;
    if (value == NULL || value->kind != AST_CALL) return false;
    FnCtx *f = e->fn;
    if (f->protectDepth > 0 || f->hasDefer || f->isGenerator) return false;
    if (f->finallys != NULL) return false;

    AstNode *callee = value->as.call.callee;
    if (callee != NULL && (callee->kind == AST_SUPER ||
                           (callee->kind == AST_MEMBER &&
                            callee->as.member.object != NULL &&
                            callee->as.member.object->kind == AST_SUPER))) {
        return false;
    }
    for (int i = 0; i < value->as.call.argCount; i++) {
        if (value->as.call.args[i].isSpread) return false;
        if (value->as.call.args[i].name != NULL) return false;
    }
    return true;
}

/* `return` leaves every block and every iteration of this frame at once, so one
 * close at the lowest of their bases covers the lot. OP_RETURN closes the whole
 * frame anyway — but only *after* the inlined `finally` copies and the defer
 * thunks have run, and those borrow this frame's slots: a `var` declared in a
 * finally body lands in the slot a sibling block's escaping closure still
 * points at, and the frame close would then capture that value. Closing first
 * is what makes the escaped closure read what it captured. */
static int returnCloseBase(Emitter *e) {
    int base = e->fn->closeBase;
    int loops = closeBaseThrough(e, NULL);   /* NULL: every enclosing loop */
    if (loops >= 0 && (base < 0 || loops < base)) base = loops;
    return base;
}

static void emitReturn(Emitter *e, AstNode *node) {
    JaiSpan span = node->span;
    AstNode *value = node->as.ret.value;
    FnCtx *f = e->fn;

    if (value == NULL) {
        emitCloseScope(e, returnCloseBase(e), span);
        emitFinallyChainForReturn(e);
        if (f->hasDefer) emitOp(e, OP_RUN_DEFERS, span);
        emitOp(e, OP_RETURN_NULL, span);
        return;
    }

    if (canTailCall(e, value)) {
        /* No close: a tail call has no finally chain and no defers to run —
         * canTailCall rules both out — and OP_TAIL_CALL closes the whole frame
         * before it reuses the window. */
        jaiEmitCall(e, value, true);
        /* TAIL_CALL reuses the frame for a closure callee; for a native one it
         * leaves the result behind, so the RETURN is still needed. */
        emitOp(e, OP_RETURN, span);
        return;
    }

    /* After the value: it may itself be a closure over a slot being closed. */
    jaiEmitExpr(e, value);
    emitCloseScope(e, returnCloseBase(e), span);
    emitFinallyChainForReturn(e);
    if (f->hasDefer) emitOp(e, OP_RUN_DEFERS, span);
    emitOp(e, OP_RETURN, span);
}

/* assert                                                               */
/* ------------------------------------------------------------------ */

/* The message baked into ASSERT_FAIL: an explicit string literal when there is
 * one, otherwise the condition's own source text. */
static uint32_t assertMessage(Emitter *e, AstNode *node) {
    AstNode *message = node->as.assertStmt.message;
    if (message != NULL && message->kind == AST_STR_LIT) {
        return strConst(e, message->as.strLit.chars, message->as.strLit.length);
    }

    AstNode *cond = node->as.assertStmt.cond;
    if (cond != NULL && jaiSpanValid(cond->span)) {
        const JaiSourceFile *file = jaiSourceGet(cond->span.file);
        if (file != NULL && file->source != NULL &&
            cond->span.end <= file->length && cond->span.end > cond->span.start) {
            char buffer[256];
            size_t length = cond->span.end - cond->span.start;
            if (length > sizeof buffer - 32) length = sizeof buffer - 32;
            int written = snprintf(buffer, sizeof buffer, "assertion failed: %.*s",
                                   (int)length, file->source + cond->span.start);
            if (written > 0) {
                size_t total = (size_t)written < sizeof buffer
                                   ? (size_t)written
                                   : sizeof buffer - 1;
                return strConst(e, buffer, total);
            }
        }
    }
    return strConst(e, "assertion failed", 16);
}

static void emitAssert(Emitter *e, AstNode *node) {
    if (e->opts->stripAsserts) return;
    JaiSpan span = node->span;

    jaiEmitExpr(e, node->as.assertStmt.cond);
    JumpSite ok = emitJump(e, OP_JUMP_IF_TRUE, span);
    emitOp(e, OP_ASSERT_FAIL, span);
    emitU24(e, assertMessage(e, node), span);
    patchJump(e, ok);
}

/* Imports and exports                                                  */
/* ------------------------------------------------------------------ */

/* The bound name of `import a.b.c` is its last component. */
static const char *lastPathComponent(const char *path) {
    if (path == NULL) return "";
    const char *dot = strrchr(path, '.');
    return dot != NULL ? dot + 1 : path;
}

static void emitImport(Emitter *e, AstNode *node) {
    JaiSpan span = node->span;
    const char *path = node->as.import.path;
    const char *alias = node->as.import.alias != NULL
                            ? node->as.import.alias
                            : lastPathComponent(path);

    emitOp(e, OP_IMPORT, span);
    emitU24(e, nameConst(e, path), span);
    jaiCgStoreSymbol(e, node->as.import.symbol, alias, span, true);
}

static void emitFromImport(Emitter *e, AstNode *node) {
    JaiSpan span = node->span;

    emitOp(e, OP_IMPORT, span);
    emitU24(e, nameConst(e, node->as.fromImport.path), span);

    if (node->as.fromImport.isWildcard) {
        CG_ERROR(e, span, E0902_INTERNAL_ERROR,
                 "`from ... import *` has no bytecode form; list the names");
        emitPop(e, span);
        return;
    }

    for (int i = 0; i < node->as.fromImport.itemCount; i++) {
        const AstImportItem *item = &node->as.fromImport.items[i];
        emitOp(e, OP_IMPORT_FROM, item->span);
        emitU24(e, nameConst(e, item->name), item->span);
        uint32_t bound = nameConst(e, item->alias != NULL ? item->alias
                                                          : item->name);
        emitOp(e, OP_DEF_GLOBAL, item->span);
        emitU24(e, bound, item->span);
    }
    emitPop(e, span);
}

/* Statements                                                           */
/* ------------------------------------------------------------------ */

static void emitVarDecl(Emitter *e, AstNode *node) {
    JaiSpan span = node->span;
    AstNode *init = node->as.varDecl.init;

    if (init != NULL) jaiEmitExpr(e, init);
    else emitOp(e, OP_NULL, span);

    AstNode *pattern = node->as.varDecl.pattern;
    if (pattern != NULL && pattern->kind == AST_PAT_BIND) {
        Symbol *sym = pattern->as.patBind.symbol != NULL
                          ? pattern->as.patBind.symbol
                          : node->as.varDecl.symbol;
        jaiCgStoreSymbol(e, sym, pattern->as.patBind.name, span, true);
    } else {
        jaiEmitStorePattern(e, pattern, true);
    }

    if (node->as.varDecl.visibility == AST_VIS_PUBLIC &&
        pattern != NULL && pattern->kind == AST_PAT_BIND) {
        emitOp(e, OP_EXPORT, span);
        emitU24(e, nameConst(e, pattern->as.patBind.name), span);
    }
}

static void emitIf(Emitter *e, AstNode *node) {
    JaiSpan span = node->span;
    int base = depthOf(e);

    jaiEmitExpr(e, node->as.conditional.cond);
    JumpSite toElse = emitJump(e, OP_JUMP_IF_FALSE, span);
    jaiEmitStmt(e, node->as.conditional.thenBranch);

    if (node->as.conditional.elseBranch == NULL) {
        setDepth(e, base);
        patchJump(e, toElse);
        return;
    }

    JumpSite toEnd = emitJump(e, OP_JUMP, span);
    setDepth(e, base);
    patchJump(e, toElse);
    jaiEmitStmt(e, node->as.conditional.elseBranch);
    setDepth(e, base);
    patchJump(e, toEnd);
}

void jaiEmitStmt(Emitter *e, AstNode *node) {
    if (node == NULL) return;
    JaiSpan span = node->span;

    switch (node->kind) {
    /* The block's bindings die here and the next sibling block gets their
     * slots, so anything a closure captured by reference has to be detached on
     * the way out (spec §5.2). The resolver reports the lowest such slot and
     * -1 when there is none, which is nearly always. */
    case AST_BLOCK: {
        int closeAt = node->as.block.captureBase;
        int saved = enterCloseScope(e, closeAt);
        for (int i = 0; i < node->as.block.count; i++) {
            jaiEmitStmt(e, node->as.block.stmts[i]);
        }
        emitCloseScope(e, closeAt, span);
        e->fn->closeBase = saved;
        break;
    }

    /* Top-level `fn` declarations are bound before any top-level statement
     * runs, because the resolver already treats the whole module body as one
     * scope: it accepts a call to a function declared further down, and
     * without this the VM would then raise NameError on a program the checker
     * passed. Hoisting is only creating the closure and storing the name —
     * default-value expressions live in thunks run per call, and a module-level
     * binding is a global looked up by name rather than a captured upvalue, so
     * nothing user-visible is evaluated early. This is what lets STYLE §8 put
     * private helpers at the bottom of a module that also has top-level
     * initialisers. Nested blocks keep source order: a `fn` inside a block is
     * an ordinary local binding and may shadow. */
    case AST_PROGRAM:
        for (int i = 0; i < node->as.block.count; i++) {
            if (node->as.block.stmts[i] != NULL &&
                node->as.block.stmts[i]->kind == AST_FN_DECL) {
                jaiEmitStmt(e, node->as.block.stmts[i]);
            }
        }
        for (int i = 0; i < node->as.block.count; i++) {
            if (node->as.block.stmts[i] == NULL ||
                node->as.block.stmts[i]->kind != AST_FN_DECL) {
                jaiEmitStmt(e, node->as.block.stmts[i]);
            }
        }
        break;

    case AST_EXPR_STMT:
        jaiEmitExpr(e, node->as.exprStmt.expr);
        emitPop(e, span);
        break;

    case AST_VAR_DECL:
        emitVarDecl(e, node);
        break;

    case AST_ASSIGN:
        jaiEmitAssign(e, node);
        break;

    case AST_IF:
        emitIf(e, node);
        break;

    case AST_WHILE:
        emitWhile(e, node);
        break;

    case AST_LOOP:
        emitInfiniteLoop(e, node);
        break;

    case AST_FOR:
        emitFor(e, node);
        break;

    case AST_MATCH:
        jaiEmitMatch(e, node, false);
        break;

    case AST_BREAK:
        emitBreak(e, node);
        break;

    case AST_CONTINUE:
        emitContinue(e, node);
        break;

    case AST_RETURN:
        emitReturn(e, node);
        break;

    case AST_THROW:
        jaiEmitExpr(e, node->as.ret.value);
        emitOp(e, OP_THROW, span);
        break;

    case AST_TRY:
        emitTry(e, node);
        break;

    case AST_DEFER:
        emitDefer(e, node);
        break;

    case AST_ASSERT:
        emitAssert(e, node);
        break;

    case AST_FN_DECL: {
        uint32_t flags = 0;
        if (node->as.fn.isGenerator) flags |= FN_GENERATOR;
        if (node->as.fn.isAsync)     flags |= FN_ASYNC;
        uint32_t k = jaiEmitFunctionConstant(e, node, flags, NULL, NULL, 0);
        jaiEmitClosure(e, node, k, span);
        jaiCgStoreSymbol(e, node->as.fn.symbol, node->as.fn.name, span, true);
        if (node->as.fn.visibility == AST_VIS_PUBLIC) {
            emitOp(e, OP_EXPORT, span);
            emitU24(e, nameConst(e, node->as.fn.name), span);
        }
        break;
    }

    case AST_CLASS_DECL:
        jaiEmitClassDecl(e, node);
        if (node->as.classDecl.visibility == AST_VIS_PUBLIC) {
            emitOp(e, OP_EXPORT, span);
            emitU24(e, nameConst(e, node->as.classDecl.name), span);
        }
        break;

    case AST_TRAIT_DECL:
        jaiEmitTraitDecl(e, node);
        if (node->as.traitDecl.visibility == AST_VIS_PUBLIC) {
            emitOp(e, OP_EXPORT, span);
            emitU24(e, nameConst(e, node->as.traitDecl.name), span);
        }
        break;

    case AST_ENUM_DECL:
        jaiEmitEnumDecl(e, node);
        if (node->as.enumDecl.visibility == AST_VIS_PUBLIC) {
            emitOp(e, OP_EXPORT, span);
            emitU24(e, nameConst(e, node->as.enumDecl.name), span);
        }
        break;

    case AST_IMPORT:
        emitImport(e, node);
        break;

    case AST_FROM_IMPORT:
        emitFromImport(e, node);
        break;

    case AST_EXPORT:
        for (int i = 0; i < node->as.exportDecl.count; i++) {
            emitOp(e, OP_EXPORT, span);
            emitU24(e, nameConst(e, node->as.exportDecl.names[i]), span);
        }
        break;

    /* An alias is erased wherever it is *used* (spec §2.3): a type is not a
     * value. Its declaration is not free, though — `from M import Alias` is
     * resolved against M's globals and export set at run time, so a public
     * alias needs a binding there or the import fails with E0802. The binding
     * is null because nothing may read it; only the export mark matters. */
    case AST_TYPE_DECL:
        if (node->as.typeDecl.visibility == AST_VIS_PUBLIC) {
            uint32_t k = nameConst(e, node->as.typeDecl.name);
            emitOp(e, OP_NULL, span);
            emitOp(e, OP_DEF_GLOBAL, span);
            emitU24(e, k, span);
            emitOp(e, OP_EXPORT, span);
            emitU24(e, k, span);
        }
        break;

    /* A `module` header is erased outright. */
    case AST_MODULE_DECL:
        break;

    default:
        if (jaiAstIsExpression(node->kind)) {
            jaiEmitExpr(e, node);
            emitPop(e, span);
            break;
        }
        CG_ERROR(e, span, E0902_INTERNAL_ERROR,
                 "no lowering for `%s` in statement position",
                 jaiAstKindName(node->kind));
        break;
    }
}
