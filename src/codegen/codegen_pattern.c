/* codegen_pattern.c — pattern lowering and `match` (spec §5.3).
 *
 * One invariant governs everything here: a pattern never leaves partial
 * results on the operand stack. The candidate value lives in a frame slot and
 * every test is the triple `GET_LOCAL slot; MATCH_x -> miss; POP`, so at
 * *every* miss edge the stack holds exactly one extra value, and one POP at
 * the miss label undoes it whatever the pattern's shape. Nested patterns
 * recurse into fresh slots and obey the same rule, which is what lets an arm
 * fail halfway through and still fall cleanly into the next one.
 */
#include "codegen_internal.h"

/* Patterns                                                             */
/* ------------------------------------------------------------------ */

/* Records a branch to patch once the whole pattern has been emitted. Callers
 * lower a test as `GET_LOCAL slot; <test> -> miss; POP`, so every site landing
 * here is reached with exactly one extra value on the stack. */
static void recordMiss(JumpList *misses, JumpSite site) {
    JAI_VEC_PUSH(JumpSite, misses, site);
}

static void emitPatternInSlot(Emitter *e, AstNode *pattern, int slot,
                              JumpList *misses);

/* Loads a sub-value of the candidate into a fresh temporary and recurses. */
static void emitSubPattern(Emitter *e, AstNode *sub, JumpList *misses,
                           JaiSpan span) {
    if (sub == NULL) {
        emitPop(e, span);
        return;
    }
    if (sub->kind == AST_PAT_WILDCARD) {
        emitPop(e, span);
        return;
    }
    if (sub->kind == AST_PAT_BIND && sub->as.patBind.type == NULL) {
        jaiEmitStorePattern(e, sub, true);
        return;
    }
    int temp = allocTemp(e);
    emitBindLocal(e, temp, span);
    emitPatternInSlot(e, sub, temp, misses);
    freeTemps(e, 1);
}

/* The declared field order of a class pattern's positional sub-patterns. */
static const char *positionalFieldName(Emitter *e, const char *typeName,
                                       int index, JaiSpan span) {
    const TypeDecl *decl = jaiTypeDeclFind(typeName);
    if (decl == NULL || index >= decl->fieldCount) {
        CG_ERROR(e, span, E0700_UNKNOWN_CLASS,
                 "cannot resolve positional field %d of `%s`", index,
                 typeName != NULL ? typeName : "?");
        return "__unknown__";
    }
    return decl->fields[index].name;
}

/* The tag of `variantName` within enum `typeName`, or -1. */
static int enumVariantTag(const char *typeName, const char *variantName) {
    const TypeDecl *decl = jaiTypeDeclFind(typeName);
    if (decl == NULL || decl->decl == NULL ||
        decl->decl->kind != AST_ENUM_DECL) {
        return -1;
    }
    const AstNode *enumDecl = decl->decl;
    for (int i = 0; i < enumDecl->as.enumDecl.variantCount; i++) {
        const char *name = enumDecl->as.enumDecl.variants[i].name;
        if (name != NULL && variantName != NULL &&
            strcmp(name, variantName) == 0) {
            return i;
        }
    }
    return -1;
}

static void emitClassPattern(Emitter *e, AstNode *pattern, int slot,
                             JumpList *misses) {
    JaiSpan span = pattern->span;
    const char *typeName = pattern->as.patClass.typeName;
    int count = pattern->as.patClass.count;

    emitGetLocal(e, slot, span);
    emitOp(e, OP_MATCH_TYPE, span);
    emitU24(e, nameConst(e, typeName), span);
    JumpSite typeMiss = {here(e), 0};
    emitI16(e, 0, span);
    typeMiss.end = here(e);
    recordMiss(misses, typeMiss);
    emitPop(e, span);

    if (count == 0) return;

    /* MATCH_FIELDS destructures in one step: its constant is a tuple of field
     * names, it peeks the object, and on success pushes one value per name. */
    Value names[JAI_MAX_ARGS];
    int rooted = 0;
    int n = count < JAI_MAX_ARGS ? count : JAI_MAX_ARGS;
    for (int i = 0; i < n; i++) {
        const char *field = pattern->as.patClass.fieldNames != NULL
                                ? pattern->as.patClass.fieldNames[i]
                                : NULL;
        if (field == NULL) field = positionalFieldName(e, typeName, i, span);
        names[i] = OBJ_VAL(jaiStringIntern(field, strlen(field)));
        jaiGCPushRoot(names[i]);
        rooted++;
    }
    ObjTuple *tuple = jaiTupleNew(names, n);
    uint32_t k = addConst(e, OBJ_VAL(tuple));
    jaiGCPopRoots(rooted);

    emitGetLocal(e, slot, span);
    emitOp(e, OP_MATCH_FIELDS, span);
    emitU24(e, k, span);
    JumpSite fieldMiss = {here(e), 0};
    emitI16(e, 0, span);
    fieldMiss.end = here(e);
    recordMiss(misses, fieldMiss);
    adjust(e, n);   /* success path pushes one value per field */

    /* Fields come off the stack last-first; park them in temporaries so the
     * sub-patterns can be compiled in declaration order. */
    int temps[JAI_MAX_ARGS];
    for (int i = n - 1; i >= 0; i--) {
        temps[i] = allocTemp(e);
        emitBindLocal(e, temps[i], span);
    }
    emitPop(e, span);   /* the object MATCH_FIELDS left underneath */

    for (int i = 0; i < n; i++) {
        emitPatternInSlot(e, pattern->as.patClass.subPatterns[i], temps[i],
                          misses);
    }
    freeTemps(e, n);
}

static void emitEnumPattern(Emitter *e, AstNode *pattern, int slot,
                            JumpList *misses) {
    JaiSpan span = pattern->span;
    const char *typeName = pattern->as.patClass.typeName;
    const char *variant = pattern->as.patClass.variantName;
    int count = pattern->as.patClass.count;

    int tag = enumVariantTag(typeName, variant);

    emitGetLocal(e, slot, span);
    emitOp(e, OP_MATCH_TYPE, span);
    if (tag < 0) {
        /* The enum was declared in another module, so its tag numbering is not
         * visible to this compilation unit. MATCH_TYPE already resolves its
         * name against the running module (spec/BYTECODE.md §3.9); a dotted
         * `Enum.Variant` asks it to check the tag too, which is exactly the
         * information codegen is missing. Payload extraction below is by index
         * and needs no tag at all. */
        char dotted[256];
        snprintf(dotted, sizeof dotted, "%s.%s",
                 typeName != NULL ? typeName : "?",
                 variant != NULL ? variant : "?");
        emitU24(e, strConst(e, dotted, strlen(dotted)), span);
    } else {
        emitU24(e, nameConst(e, typeName), span);
    }
    JumpSite typeMiss = {here(e), 0};
    emitI16(e, 0, span);
    typeMiss.end = here(e);
    recordMiss(misses, typeMiss);
    emitPop(e, span);

    if (tag >= 0) {
        /* ENUM_TAG peeks, so drop the enum value and test the tag alone: that
         * keeps the one-extra-value miss invariant. */
        emitGetLocal(e, slot, span);
        emitOp(e, OP_ENUM_TAG, span);
        emitOp(e, OP_SWAP, span);
        emitPop(e, span);
        emitOp(e, OP_MATCH_CONST, span);
        emitU24(e, addConst(e, INT_VAL(tag)), span);
        JumpSite tagMiss = {here(e), 0};
        emitI16(e, 0, span);
        tagMiss.end = here(e);
        recordMiss(misses, tagMiss);
        emitPop(e, span);
    }

    for (int i = 0; i < count; i++) {
        emitGetLocal(e, slot, span);
        emitOp(e, OP_ENUM_FIELD, span);
        emitByte(e, (uint8_t)i, span);
        emitOp(e, OP_SWAP, span);
        emitPop(e, span);
        emitSubPattern(e, pattern->as.patClass.subPatterns[i], misses, span);
    }
}

static void emitSeqPattern(Emitter *e, AstNode *pattern, int slot,
                           JumpList *misses) {
    JaiSpan span = pattern->span;
    int count = pattern->as.patSeq.count;
    int restIndex = pattern->as.patSeq.restIndex;
    bool hasRest = restIndex >= 0;
    int fixed = hasRest ? count - 1 : count;
    int tail = hasRest ? count - restIndex - 1 : 0;

    emitGetLocal(e, slot, span);
    emitOp(e, OP_MATCH_SEQ, span);
    emitByte(e, (uint8_t)(fixed > 255 ? 255 : fixed), span);
    emitByte(e, hasRest ? 1 : 0, span);
    JumpSite miss = {here(e), 0};
    emitI16(e, 0, span);
    miss.end = here(e);
    recordMiss(misses, miss);
    emitPop(e, span);

    for (int i = 0; i < count; i++) {
        AstNode *sub = pattern->as.patSeq.elems[i];
        if (hasRest && i == restIndex) {
            /* rest = value[restIndex : -tail], or [restIndex:] when it is last */
            emitGetLocal(e, slot, span);
            jaiCgInt(e, restIndex, span);
            uint8_t flags = 1;
            int pushed = 1;
            if (tail > 0) {
                jaiCgInt(e, -tail, span);
                flags |= 2;
                pushed++;
            }
            emitOp(e, OP_GET_SLICE, span);
            emitByte(e, flags, span);
            adjust(e, -pushed);
        } else {
            int index = (hasRest && i > restIndex) ? -(count - i) : i;
            emitGetLocal(e, slot, span);
            jaiCgInt(e, index, span);
            emitOp(e, OP_GET_INDEX, span);
        }
        emitSubPattern(e, sub, misses, span);
    }
}

/* An or-pattern tries each alternative in turn; only the last one's failures
 * escape to the caller's miss list. */
static void emitOrPattern(Emitter *e, AstNode *pattern, int slot,
                          JumpList *misses) {
    JaiSpan span = pattern->span;
    int count = pattern->as.patSeq.count;
    if (count == 0) return;

    JumpList matched;
    jumpListInit(&matched);
    int base = depthOf(e);

    for (int i = 0; i < count; i++) {
        if (i == count - 1) {
            emitPatternInSlot(e, pattern->as.patSeq.elems[i], slot, misses);
            break;
        }
        JumpList altMiss;
        jumpListInit(&altMiss);
        emitPatternInSlot(e, pattern->as.patSeq.elems[i], slot, &altMiss);
        JAI_VEC_PUSH(JumpSite, &matched, emitJump(e, OP_JUMP, span));

        setDepth(e, base + 1);   /* the failing candidate is still on the stack */
        patchAll(e, &altMiss);
        jumpListFree(&altMiss);
        emitPop(e, span);
    }

    setDepth(e, base);
    patchAll(e, &matched);
    jumpListFree(&matched);
}

static void emitPatternInSlot(Emitter *e, AstNode *pattern, int slot,
                              JumpList *misses) {
    if (pattern == NULL) return;
    JaiSpan span = pattern->span;

    switch (pattern->kind) {
    case AST_PAT_WILDCARD:
        break;

    case AST_PAT_BIND: {
        if (pattern->as.patBind.type != NULL) {
            emitGetLocal(e, slot, span);
            emitOp(e, OP_MATCH_TYPE, span);
            emitU24(e, nameConst(e, jaiCgTypeNameOf(pattern->as.patBind.type)), span);
            JumpSite miss = {here(e), 0};
            emitI16(e, 0, span);
            miss.end = here(e);
            recordMiss(misses, miss);
            emitPop(e, span);
        }
        Symbol *sym = pattern->as.patBind.symbol;
        if (sym == NULL && pattern->as.patBind.name == NULL) break;
        emitGetLocal(e, slot, span);
        jaiCgStoreSymbol(e, sym, pattern->as.patBind.name, span, sym == NULL);
        break;
    }

    case AST_PAT_LITERAL: {
        Value v;
        if (!jaiCgLiteralValue(pattern->as.patLiteral.value, &v)) {
            CG_ERROR(e, span, E0110_INVALID_PATTERN,
                     "literal pattern is not a compile-time constant");
            break;
        }
        emitGetLocal(e, slot, span);
        emitOp(e, OP_MATCH_CONST, span);
        emitU24(e, addConst(e, v), span);
        JumpSite miss = {here(e), 0};
        emitI16(e, 0, span);
        miss.end = here(e);
        recordMiss(misses, miss);
        emitPop(e, span);
        break;
    }

    case AST_PAT_RANGE: {
        Value lo, hi;
        if (!jaiCgLiteralValue(pattern->as.patRange.lo, &lo) ||
            !jaiCgLiteralValue(pattern->as.patRange.hi, &hi)) {
            CG_ERROR(e, span, E0110_INVALID_PATTERN,
                     "range pattern bounds must be compile-time constants");
            break;
        }
        emitGetLocal(e, slot, span);
        emitOp(e, OP_MATCH_RANGE, span);
        emitU24(e, addConst(e, lo), span);
        emitU24(e, addConst(e, hi), span);
        emitByte(e, pattern->as.patRange.inclusive ? 1 : 0, span);
        JumpSite miss = {here(e), 0};
        emitI16(e, 0, span);
        miss.end = here(e);
        recordMiss(misses, miss);
        emitPop(e, span);
        break;
    }

    case AST_PAT_TUPLE:
    case AST_PAT_LIST:
        emitSeqPattern(e, pattern, slot, misses);
        break;

    case AST_PAT_CLASS:
        emitClassPattern(e, pattern, slot, misses);
        break;

    case AST_PAT_ENUM:
        emitEnumPattern(e, pattern, slot, misses);
        break;

    case AST_PAT_OR:
        emitOrPattern(e, pattern, slot, misses);
        break;

    default:
        CG_ERROR(e, span, E0110_INVALID_PATTERN, "unsupported pattern `%s`",
                 jaiAstKindName(pattern->kind));
        break;
    }
}

/* Irrefutable binding: `let (a, b) = p`, `for x in xs`, `(a, b) = p`.
 * Consumes the value on top of the stack. `declaring` distinguishes the first
 * two — which introduce their names and so must DEF_GLOBAL at module level —
 * from the third, which stores into names that already exist. */
void jaiEmitStorePattern(Emitter *e, AstNode *pattern, bool declaring) {
    if (pattern == NULL) {
        emitPop(e, fileSpan(e));
        return;
    }
    JaiSpan span = pattern->span;

    switch (pattern->kind) {
    case AST_PAT_WILDCARD:
        emitPop(e, span);
        break;

    case AST_PAT_BIND:
        jaiCgStoreSymbol(e, pattern->as.patBind.symbol, pattern->as.patBind.name,
                         span, declaring || pattern->as.patBind.symbol == NULL);
        break;

    case AST_PAT_TUPLE:
    case AST_PAT_LIST: {
        int count = pattern->as.patSeq.count;
        int restIndex = pattern->as.patSeq.restIndex;
        if (count > 255) {
            CG_ERROR(e, span, E0110_INVALID_PATTERN,
                     "a destructuring pattern may bind at most 255 elements");
            emitPop(e, span);
            break;
        }
        emitOp(e, OP_UNPACK, span);
        emitByte(e, (uint8_t)count, span);
        emitByte(e, (uint8_t)(restIndex < 0 ? 255 : restIndex), span);
        adjust(e, count - 1);
        /* UNPACK pushes right to left, so element 0 is on top: storing in
         * source order pops them in the order they were written. */
        for (int i = 0; i < count; i++) {
            jaiEmitStorePattern(e, pattern->as.patSeq.elems[i], declaring);
        }
        break;
    }

    /* An assignment target can be a plain expression rather than a pattern. */
    case AST_IDENT:
        jaiCgStoreSymbol(e, pattern->as.ident.symbol, pattern->as.ident.name, span,
                         declaring);
        break;

    default:
        CG_ERROR(e, span, E0109_INVALID_ASSIGN_TARGET,
                 "`%s` is not a binding pattern",
                 jaiAstKindName(pattern->kind));
        emitPop(e, span);
        break;
    }
}

/* Match                                                                */
/* ------------------------------------------------------------------ */

void jaiEmitMatch(Emitter *e, AstNode *node, bool asExpression) {
    JaiSpan span = node->span;
    int base = depthOf(e);

    jaiEmitExpr(e, node->as.match.subject);
    int subject = allocTemp(e);
    emitBindLocal(e, subject, span);

    JumpList done;
    jumpListInit(&done);

    for (int i = 0; i < node->as.match.armCount; i++) {
        AstMatchArm *arm = &node->as.match.arms[i];
        JumpList misses, guardMiss;
        jumpListInit(&misses);
        jumpListInit(&guardMiss);

        setDepth(e, base);
        emitPatternInSlot(e, arm->pattern, subject, &misses);

        if (arm->guard != NULL) {
            jaiEmitExpr(e, arm->guard);
            JAI_VEC_PUSH(JumpSite, &guardMiss,
                         emitJump(e, OP_JUMP_IF_FALSE, arm->guard->span));
        }

        if (asExpression) {
            if (arm->body != NULL && arm->body->kind == AST_BLOCK) {
                jaiEmitBlockValue(e, arm->body);
            } else {
                jaiEmitExpr(e, arm->body);
            }
        } else if (arm->body != NULL) {
            if (arm->body->kind == AST_BLOCK || jaiAstIsStatement(arm->body->kind)) {
                jaiEmitStmt(e, arm->body);
            } else {
                jaiEmitExpr(e, arm->body);
                emitPop(e, arm->span);
            }
        }
        JAI_VEC_PUSH(JumpSite, &done, emitJump(e, OP_JUMP, arm->span));

        /* Pattern misses arrive with the failed candidate still pushed. */
        setDepth(e, base + 1);
        patchAll(e, &misses);
        emitPop(e, arm->span);
        setDepth(e, base);
        patchAll(e, &guardMiss);

        jumpListFree(&misses);
        jumpListFree(&guardMiss);
    }

    setDepth(e, base);
    if (asExpression) {
        /* The checker proves exhaustiveness; this is the belt-and-braces path
         * for `any` subjects, and it keeps the join depth honest. */
        emitOp(e, OP_ASSERT_FAIL, span);
        emitU24(e, strConst(e, "no match arm matched", 20), span);
        emitOp(e, OP_NULL, span);
    }

    setDepth(e, base + (asExpression ? 1 : 0));
    patchAll(e, &done);
    jumpListFree(&done);
    freeTemps(e, 1);
}
