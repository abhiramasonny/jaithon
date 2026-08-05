/* check_expr.c — expression checking: the operators, member access, calls,
 * indexing and slicing, comprehensions and function expressions.
 *
 * Every function here answers the same question — what type does this node
 * have — and has the same two side effects: it writes that type into
 * AstNode.type, and where an any->T boundary needs a runtime check it wraps
 * the node in an AST_CAST. Nothing here decides control flow; `if` and `match`
 * in expression position hand back to check_stmt.c.
 */
#include "check_internal.h"


/* ------------------------------------------------------------------ */
/* Conditions — there is no truthiness                                  */
/* ------------------------------------------------------------------ */

/* The subject of a help that shows the fix. Only a bare name can be quoted
 * back verbatim; for anything larger a placeholder is honest, where echoing a
 * guess at the user's expression would not be. */
const char *jaiChkSubjectOf(const AstNode *n, const char *fallback) {
    if (n != NULL && n->kind == AST_IDENT && n->as.ident.name != NULL)
        return n->as.ident.name;
    return fallback;
}

/* Written into the diagnostic with the condition's own spelling, so the help
 * is a line the reader can paste rather than a template to translate. */
static const char *conditionHelp(JaiType *t, const AstNode *node, char *buf,
                                 size_t bufLen) {
    if (t == NULL) return NULL;
    if (jaiTypeIsOptional(t) || t->kind == TY_NULL) {
        (void)snprintf(buf, bufLen, "test for absence explicitly: `%s != null`",
                       jaiChkSubjectOf(node, "x"));
        return buf;
    }
    switch (t->kind) {
    case TY_LIST: case TY_DICT: case TY_SET: case TY_STR:
    case TY_BYTES: case TY_TUPLE:
        (void)snprintf(buf, bufLen, "test the length explicitly: `%s.len() > 0`",
                       jaiChkSubjectOf(node, "xs"));
        return buf;
    case TY_INT: case TY_FLOAT:
        (void)snprintf(buf, bufLen, "compare explicitly: `%s != 0`",
                       jaiChkSubjectOf(node, "n"));
        return buf;
    default:
        return "Jaithon has no truthiness; a condition must be a `bool`";
    }
}

/* Every boolean position in the language funnels through here: if, elif,
 * while, and, or, not, ternary, match guards, comprehension filters, assert. */
void jaiChkCondition(Checker *c, AstNode *node, const char *where) {
    if (node == NULL) return;
    JaiType *t = jaiChkValue(c, node);
    if (jaiChkIsAny(t) || jaiChkIsNever(t) || t->kind == TY_BOOL) return;

    char got[TYPE_BUF];
    jaiChkRenderType(t, got, sizeof got);
    JaiDiag *d = ERR(c, E0401_CONDITION_NOT_BOOL, node->span,
                     "the condition of %s must be `bool`, found `%s`", where, got);
    char helpBuf[160];
    const char *help = conditionHelp(t, node, helpBuf, sizeof helpBuf);
    if (help != NULL) jaiDiagAddHelp(d, "%s", help);
}

/* ------------------------------------------------------------------ */
/* Iteration and indexing                                               */
/* ------------------------------------------------------------------ */

static bool classIsIterable(const TypeDecl *d) {
    if (d == NULL) return true;   /* unknown shape: assume the author knows */
    if (jaiChkFindMemberIndex(d, "__iter__") >= 0 || jaiChkFindMemberIndex(d, "__next__") >= 0)
        return true;
    for (int i = 0; i < d->traitCount; i++)
        if (d->traits != NULL && d->traits[i] != NULL &&
            jaiChkSameName(d->traits[i]->name, "Iterable"))
            return true;
    return false;
}

JaiType *jaiChkIterableElement(Checker *c, AstNode *node, JaiType *t) {
    if (jaiChkIsAny(t) || jaiChkIsNever(t)) return gTypes.tAny;

    /* Whether a declared type is iterable is a question about its members and
     * its traits, and a declaration read from another module is laid out on
     * demand — so demand it before asking, or every imported iterable would
     * answer "no members, no traits" and be reported as not iterable. */
    TypeDecl *iterDecl = jaiChkTypeDeclOf(t);
    if (iterDecl != NULL) jaiChkLayoutDecl(c, jaiChkDeclEntry(iterDecl));

    JaiType *elem = NULL;
    if (jaiTypeIsIterable(t, &elem)) return jaiChkOrAny(elem);

    /* A user type is iterable when it implements Iterable (spec §5.2); the
     * type system cannot see that through a bare TY_CLASS, so ask the decl. */
    if ((t->kind == TY_CLASS || t->kind == TY_TRAIT) && classIsIterable(jaiChkTypeDeclOf(t)))
        return gTypes.tAny;

    char got[TYPE_BUF];
    jaiChkRenderType(t, got, sizeof got);
    JaiDiag *d = ERR(c, E0405_NOT_ITERABLE, node->span, "`%s` is not iterable", got);
    if (t->kind == TY_CLASS)
        jaiDiagAddHelp(d, "implement `trait Iterable` on `%s` to loop over it", got);
    else if (t->kind == TY_INT)
        jaiDiagAddHelp(d, "loop over a range: `for i in 0..%s`", "n");
    return gTypes.tAny;
}

/* ------------------------------------------------------------------ */
/* Expressions                                                          */
/* ------------------------------------------------------------------ */

JaiType *jaiChkValue(Checker *c, AstNode *node) {
    JaiType *t = jaiChkExpr(c, node);
    if (jaiChkIsVoid(t)) {
        ERR(c, E0409_VOID_VALUE_USED, node->span,
            "this expression has type `void` and produces no value");
        return gTypes.tAny;
    }
    return t;
}

static JaiType *checkIdent(Checker *c, AstNode *node) {
    Symbol *s = node->as.ident.symbol;
    if (s == NULL) {
        /* The resolver already reported the undefined name; do not pile on. */
        return gTypes.tAny;
    }

    /* A folded `const` is inlined at its use site (spec §3). */
    if (c->foldConstants && s->mutability == VD_CONST && s->isConstFolded &&
        s->constValue != NULL) {
        ConstValue v = jaiConstEval(c, s->constValue);
        if (v.kind != CONST_NONE) {
            jaiConstReplace(c->ast, node, v);
            if (s->type != NULL) node->type = s->type;
            return jaiChkOrAny(node->type);
        }
    }

    if (s->type == NULL && s->kind != SYM_LOCAL && s->kind != SYM_PARAM) {
        /* Classes and enums may be referenced before their layout pass ran. */
        JaiType *named = jaiChkLookupTypeName(c, s->name);
        if (named != NULL &&
            (s->kind == SYM_CLASS || s->kind == SYM_TRAIT || s->kind == SYM_ENUM))
            s->type = named;
    }
    return jaiChkOrAny(s->type);
}

/* The contextual type jaiChkApplyContext stamped on a container literal, if it is
 * the right shape for this literal. Consumed so a stale hint cannot leak into
 * a later, unannotated literal that happens to reuse the node. */
static JaiType *takeContainerContext(AstNode *node, TypeKind want, int arity) {
    JaiType *hint = node->type;
    node->type = NULL;
    if (hint == NULL || hint->kind != want || hint->argCount != arity) return NULL;
    return hint;
}

/* The type an already-checked element contributes to a container's join.
 * jaiChkValue reports `any` for a void expression after diagnosing it, but
 * leaves node->type as `void`; re-reading the tree has to normalise the same
 * way or an error cascade would change shape. */
static JaiType *elementType(AstNode *node) {
    if (node == NULL) return gTypes.tAny;
    if (jaiChkIsVoid(node->type)) return gTypes.tAny;
    return jaiChkOrAny(node->type);
}

/* Numeric fusion (spec §2.2). `nodes` are the already-checked elements of an
 * unannotated literal position. When the position mixes an exact `int` with an
 * exact `float`, every int node is converted in place — retyped if it is a
 * literal, wrapped in AST_CAST{widen} otherwise — so that the element really
 * is a float at run time and not merely called one.
 *
 * Nothing is returned, and that is the point. The caller joins by re-reading
 * node->type, so the type it computes is the type the tree actually holds: a
 * node that failed to convert stays an int, the join stays a union, and the
 * failure mode of a bug here is today's behaviour rather than a lie. */
static void fuseNumeric(Checker *c, AstNode **nodes, int count) {
    bool hasInt = false, hasFloat = false;
    for (int i = 0; i < count; i++) {
        JaiType *t = nodes[i] != NULL ? nodes[i]->type : NULL;
        /* jaiTypeWidenTarget is the exact-int test: it resolves aliases and
         * answers NULL for `int?`, `any`, `never` and `list[int]` alike. */
        if (jaiTypeWidenTarget(t, gTypes.tFloat) != NULL) hasInt = true;
        if (jaiTypeEquals(t, gTypes.tFloat)) hasFloat = true;
    }
    if (!hasInt || !hasFloat) return;

    for (int i = 0; i < count; i++) {
        AstNode *n = nodes[i];
        if (n == NULL || jaiTypeWidenTarget(n->type, gTypes.tFloat) == NULL) continue;
        jaiChkWidenToFloat(c, n, gTypes.tFloat);
        JAI_ASSERT(jaiTypeEquals(n->type, gTypes.tFloat) &&
                       (n->kind == AST_FLOAT_LIT ||
                        (n->kind == AST_CAST && n->as.cast.widen)),
                   "numeric fusion retyped a node without converting it");
    }
}

static JaiType *checkSequenceLiteral(Checker *c, AstNode *node) {
    int count = node->as.sequence.count;

    if (node->kind == AST_TUPLE_LIT) {
        JaiType *wantTuple = takeContainerContext(node, TY_TUPLE, count);
        if (wantTuple != NULL) {
            for (int i = 0; i < count; i++) {
                AstNode *item = node->as.sequence.items[i];
                jaiChkRequireAssignable(c, item, jaiChkValue(c, item), wantTuple->args[i],
                                        E0400_TYPE_MISMATCH, "mismatched member type");
            }
            return wantTuple;
        }
        for (int i = 0; i < count; i++) jaiChkValue(c, node->as.sequence.items[i]);
        JaiType **members = jaiChkTypeArray(c, count);
        for (int i = 0; i < count; i++) members[i] = jaiChkOrAny(node->as.sequence.items[i]->type);
        return jaiTypeTuple(members, count);
    }

    JaiType *want = takeContainerContext(
        node, node->kind == AST_SET_LIT ? TY_SET : TY_LIST, 1);

    if (want != NULL) {
        /* Annotated: every element is checked against the declared element
         * type, so a heterogeneous literal reports the offending element
         * rather than the whole container failing against its annotation. */
        for (int i = 0; i < count; i++) {
            AstNode *item = node->as.sequence.items[i];
            JaiType *t = jaiChkValue(c, item);
            jaiChkRequireAssignable(c, item, t, want->args[0], E0400_TYPE_MISMATCH,
                              "mismatched element type");
        }
        return want;
    }

    /* Unannotated. Check every element first and discard the reported types,
     * then fuse, then join by re-reading the tree. Folding the join as we went
     * would make the result order-dependent — `[1, "a", 2.0]` would compute
     * join(join(int, str), float) and never see the int and the float as a
     * pair — as well as leaving no point at which the whole literal is known.
     *
     * A set literal is deliberately not fused. A number in a hash position
     * decides which elements are the same element, so converting one can merge
     * two members that were distinct above 2^53 and silently shrink the set.
     * An annotation is an instruction and may do that; an inference is a guess
     * and must never change how many elements a container has. */
    for (int i = 0; i < count; i++) jaiChkValue(c, node->as.sequence.items[i]);
    if (node->kind == AST_LIST_LIT) fuseNumeric(c, node->as.sequence.items, count);

    JaiType *elem = NULL;
    for (int i = 0; i < count; i++) {
        JaiType *t = elementType(node->as.sequence.items[i]);
        elem = (elem == NULL) ? t : jaiTypeJoin(elem, t);
    }
    if (elem == NULL) elem = gTypes.tAny;
    return node->kind == AST_SET_LIT ? jaiTypeSet(elem) : jaiTypeList(elem);
}

static JaiType *checkDictLiteral(Checker *c, AstNode *node) {
    JaiType *want = takeContainerContext(node, TY_DICT, 2);

    if (want != NULL) {
        for (int i = 0; i < node->as.dict.count; i++) {
            AstNode *k = node->as.dict.keys[i];
            AstNode *v = node->as.dict.values[i];
            jaiChkRequireAssignable(c, k, jaiChkValue(c, k), want->args[0],
                              E0400_TYPE_MISMATCH, "mismatched key type");
            jaiChkRequireAssignable(c, v, jaiChkValue(c, v), want->args[1],
                              E0400_TYPE_MISMATCH, "mismatched value type");
        }
        return want;
    }

    /* Unannotated: check, then fuse the values, then join by re-reading the
     * tree — see checkSequenceLiteral for why the join cannot be folded as we
     * go. Keys are left alone for the same reason a set literal is: they are a
     * hash position, and an inferred conversion there could merge two entries
     * the programmer wrote separately. */
    for (int i = 0; i < node->as.dict.count; i++) {
        jaiChkValue(c, node->as.dict.keys[i]);
        jaiChkValue(c, node->as.dict.values[i]);
    }
    fuseNumeric(c, node->as.dict.values, node->as.dict.count);

    JaiType *key = NULL, *value = NULL;
    for (int i = 0; i < node->as.dict.count; i++) {
        JaiType *k = elementType(node->as.dict.keys[i]);
        JaiType *v = elementType(node->as.dict.values[i]);
        key = (key == NULL) ? k : jaiTypeJoin(key, k);
        value = (value == NULL) ? v : jaiTypeJoin(value, v);
    }
    return jaiTypeDict(jaiChkOrAny(key), jaiChkOrAny(value));
}

bool jaiChkIsArithmetic(OpKind op) {
    switch (op) {
    case OPK_ADD: case OPK_SUB: case OPK_MUL: case OPK_DIV:
    case OPK_FLOORDIV: case OPK_MOD: case OPK_POW:
    case OPK_ADD_WRAP: case OPK_SUB_WRAP: case OPK_MUL_WRAP:
        return true;
    default:
        return false;
    }
}

/* Which operators widen an int operand against a float one (spec §2.5).
 *
 * The wrapping forms are excluded along with the bitwise operators: both are
 * 64-bit integer operations with no float form, so a float operand there is
 * still E0406 rather than a conversion. */
bool jaiChkWidensOperands(OpKind op) {
    switch (op) {
    case OPK_ADD_WRAP: case OPK_SUB_WRAP: case OPK_MUL_WRAP:
        return false;
    default:
        return jaiChkIsArithmetic(op);
    }
}

/* One int, one float: the int side widens and the expression is float. The
 * conversion goes into the tree here rather than being left to the VM, so
 * `n * 1.5` costs a conversion and not a type test. */
static void widenNumericMix(Checker *c, AstNode *node, OpKind op,
                            JaiType **left, JaiType **right) {
    if (!jaiChkWidensOperands(op)) return;
    bool leftInt = (*left)->kind == TY_INT && (*right)->kind == TY_FLOAT;
    bool rightInt = (*right)->kind == TY_INT && (*left)->kind == TY_FLOAT;
    if (!leftInt && !rightInt) return;

    AstNode *intSide = leftInt ? node->as.binary.left : node->as.binary.right;
    jaiChkWidenToFloat(c, intSide, gTypes.tFloat);
    if (leftInt) *left = gTypes.tFloat; else *right = gTypes.tFloat;
}

static JaiType *checkBinary(Checker *c, AstNode *node) {
    OpKind op = node->as.binary.op;
    JaiType *left = jaiChkValue(c, node->as.binary.left);
    JaiType *right = jaiChkValue(c, node->as.binary.right);

    /* Identity never constrains its operands: `x is Circle` compares a value
     * against a class, and `x is y` compares two references. */
    if (op == OPK_IS || op == OPK_IS_NOT) return gTypes.tBool;

    widenNumericMix(c, node, op, &left, &right);
    if (jaiChkIsNever(left) || jaiChkIsNever(right)) return gTypes.tNever;
    if (jaiChkIsAny(left) || jaiChkIsAny(right))
        return jaiOpKindIsComparison(op) ? gTypes.tBool : gTypes.tAny;

    JaiType *result = jaiTypeBinaryResult((int)op, left, right);
    if (result != NULL) return result;

    char a[TYPE_BUF], b[TYPE_BUF];
    jaiChkRenderType(left, a, sizeof a);
    jaiChkRenderType(right, b, sizeof b);
    JaiDiag *d = ERR(c, E0406_BAD_OPERAND_TYPES, node->span,
                     "operator `%s` does not apply to `%s` and `%s`",
                     jaiOpKindText(op), a, b);
    if (jaiChkTypeDeclOf(left) != NULL)
        jaiDiagAddHelp(d, "define the matching dunder method on `%s`", a);
    return gTypes.tAny;
}

static JaiType *checkLogical(Checker *c, AstNode *node) {
    const char *what = node->as.binary.op == OPK_AND ? "`and`" : "`or`";
    jaiChkCondition(c, node->as.binary.left, what);

    /* The right operand of `and` sees the facts the left operand established. */
    int mark = jaiChkNarrowMark();
    NarrowFact facts[MAX_FACTS];
    int count = jaiChkCollectFacts(c, node->as.binary.left, facts, MAX_FACTS,
                             node->as.binary.op == OPK_OR);
    jaiChkNarrowApply(facts, count, false);
    jaiChkCondition(c, node->as.binary.right, what);
    jaiChkNarrowRestore(mark);
    return gTypes.tBool;
}

static JaiType *checkChain(Checker *c, AstNode *node) {
    int opCount = node->as.chain.opCount;
    if (opCount <= 0 || node->as.chain.operands == NULL || node->as.chain.ops == NULL)
        return gTypes.tBool;

    for (int i = 0; i <= opCount; i++) jaiChkValue(c, node->as.chain.operands[i]);

    for (int i = 0; i < opCount; i++) {
        OpKind op = node->as.chain.ops[i];
        if (op == OPK_IS || op == OPK_IS_NOT) continue;
        JaiType *left = jaiChkOrAny(node->as.chain.operands[i]->type);
        JaiType *right = jaiChkOrAny(node->as.chain.operands[i + 1]->type);
        if (jaiChkIsAny(left) || jaiChkIsAny(right) || jaiChkIsNever(left) || jaiChkIsNever(right)) continue;
        if (jaiTypeBinaryResult((int)op, left, right) != NULL) continue;

        char a[TYPE_BUF], b[TYPE_BUF];
        jaiChkRenderType(left, a, sizeof a);
        jaiChkRenderType(right, b, sizeof b);
        ERR(c, E0406_BAD_OPERAND_TYPES, node->span,
            "operator `%s` does not apply to `%s` and `%s`", jaiOpKindText(op), a, b);
    }
    return gTypes.tBool;
}

static JaiType *checkUnary(Checker *c, AstNode *node) {
    OpKind op = node->as.unary.op;
    if (op == OPK_NOT) {
        jaiChkCondition(c, node->as.unary.operand, "`not`");
        return gTypes.tBool;
    }

    JaiType *operand = jaiChkValue(c, node->as.unary.operand);
    if (jaiChkIsAny(operand) || jaiChkIsNever(operand)) return operand;

    JaiType *result = jaiTypeUnaryResult((int)op, operand);
    if (result != NULL) return result;

    char got[TYPE_BUF];
    jaiChkRenderType(operand, got, sizeof got);
    ERR(c, E0406_BAD_OPERAND_TYPES, node->span,
        "operator `%s` does not apply to `%s`", jaiOpKindText(op), got);
    return gTypes.tAny;
}

static JaiType *checkTernary(Checker *c, AstNode *node) {
    jaiChkCondition(c, node->as.ternary.cond, "a ternary");

    int mark = jaiChkNarrowMark();
    NarrowFact facts[MAX_FACTS];
    int count = jaiChkCollectFacts(c, node->as.ternary.cond, facts, MAX_FACTS, false);
    jaiChkNarrowApply(facts, count, false);
    JaiType *thenType = jaiChkValue(c, node->as.ternary.thenExpr);
    jaiChkNarrowRestore(mark);

    mark = jaiChkNarrowMark();
    jaiChkNarrowApply(facts, count, true);
    JaiType *elseType = jaiChkValue(c, node->as.ternary.elseExpr);
    jaiChkNarrowRestore(mark);

    return jaiTypeJoin(thenType, elseType);
}

static JaiType *checkCoalesce(Checker *c, AstNode *node) {
    JaiType *left = jaiChkValue(c, node->as.coalesce.left);
    JaiType *right = jaiChkValue(c, node->as.coalesce.right);
    if (jaiChkIsAny(left)) return gTypes.tAny;

    JaiType *present = jaiTypeNarrow(left, gTypes.tNull, false);
    if (present == NULL) present = left;
    if (left->kind != TY_NULL && !jaiTypeIsOptional(left) && !jaiChkIsAny(left)) {
        /* Not an error — `a ?? b` on a non-optional is merely dead — but the
         * author almost certainly meant something else. */
        char got[TYPE_BUF];
        jaiChkRenderType(left, got, sizeof got);
        JaiDiag *d = WARN(W0102_UNREACHABLE_CODE, node->as.coalesce.right->span,
                          "the right operand of `??` is unreachable: `%s` is never null",
                          got);
        jaiDiagAddHelp(d, "remove the `??`, or make the left operand optional");
    }
    return jaiTypeJoin(present, right);
}

static JaiType *checkIndex(Checker *c, AstNode *node) {
    AstNode *object = node->as.index.object;

    /* `Box[int](21)` applies type arguments to a generic declaration; erasure
     * makes the result the bare declared type. The resolver settled which
     * reading the brackets have — it is the only pass that can, since an
     * imported name carries no kind here (spec §8). */
    if (node->as.index.typeArgs) {
        JaiType *base = jaiChkValue(c, object);
        node->as.index.index->type = gTypes.tAny;
        return jaiChkOrAny(base);
    }
    if (object != NULL && object->kind == AST_IDENT) {
        Symbol *s = object->as.ident.symbol;
        if (s != NULL && (s->kind == SYM_CLASS || s->kind == SYM_TRAIT ||
                          s->kind == SYM_ENUM || s->kind == SYM_TYPE_ALIAS ||
                          jaiChkIdentTypeDecl(s) != NULL)) {
            JaiType *base = jaiChkOrAny(s->type != NULL ? s->type
                                                  : jaiChkLookupTypeName(c, s->name));
            object->type = base;
            node->as.index.index->type = gTypes.tAny;
            return base;
        }
    }

    JaiType *target = jaiChkValue(c, object);
    JaiType *index = jaiChkValue(c, node->as.index.index);
    if (jaiChkIsAny(target) || jaiChkIsNever(target)) return gTypes.tAny;
    /* `__getitem__` is a member, so the declaration has to be laid out first. */
    TypeDecl *indexDecl = jaiChkTypeDeclOf(target);
    if (indexDecl != NULL) jaiChkLayoutDecl(c, jaiChkDeclEntry(indexDecl));

    /* A tuple's members have different types, so a constant subscript answers
     * with the one it actually selects: `pair[0]` on `tuple[str, int]` is a
     * `str`, not `str | int`. Only a literal index can be resolved this way —
     * a computed one keeps the join. */
    if (target->kind == TY_TUPLE) {
        ConstValue k = jaiConstEval(c, node->as.index.index);
        if (k.kind == CONST_INT) {
            int64_t at = k.as.i < 0 ? k.as.i + target->argCount : k.as.i;
            if (at < 0 || at >= target->argCount) {
                ERR(c, E0404_NOT_INDEXABLE, node->span,
                    "index %lld is out of range for a %d-member tuple",
                    (long long)k.as.i, target->argCount);
                return gTypes.tAny;
            }
            return target->args[at];
        }
    }

    JaiType *wantIndex = NULL, *result = NULL;
    if (jaiTypeIsIndexable(target, &wantIndex, &result)) {
        if (wantIndex != NULL && !jaiChkIsAny(index))
            jaiChkRequireAssignable(c, node->as.index.index, index, wantIndex,
                              E0400_TYPE_MISMATCH, "invalid index type");
        return jaiChkOrAny(result);
    }

    /* `__getitem__` makes a user type indexable. */
    TypeDecl *decl = jaiChkTypeDeclOf(target);
    if (decl != NULL && jaiChkFindMemberIndex(decl, "__getitem__") >= 0) return gTypes.tAny;
    if (decl == NULL && (target->kind == TY_CLASS || target->kind == TY_TRAIT))
        return gTypes.tAny;

    char got[TYPE_BUF];
    jaiChkRenderType(target, got, sizeof got);
    JaiDiag *d = ERR(c, E0404_NOT_INDEXABLE, node->span, "`%s` cannot be indexed", got);
    if (decl != NULL) jaiDiagAddHelp(d, "define `__getitem__` on `%s`", got);
    return gTypes.tAny;
}

JaiType *jaiChkSlice(Checker *c, AstNode *node) {
    JaiType *target = jaiChkValue(c, node->as.slice.object);
    AstNode *parts[3] = { node->as.slice.start, node->as.slice.stop, node->as.slice.step };
    for (int i = 0; i < 3; i++) {
        if (parts[i] == NULL) continue;
        JaiType *t = jaiChkValue(c, parts[i]);
        if (!jaiChkIsAny(t) && t->kind != TY_INT && t->kind != TY_NULL)
            jaiChkRequireAssignable(c, parts[i], t, gTypes.tInt, E0400_TYPE_MISMATCH,
                              "a slice bound must be an int");
    }

    if (jaiChkIsAny(target) || jaiChkIsNever(target)) return gTypes.tAny;
    switch (target->kind) {
    case TY_LIST: case TY_STR: case TY_BYTES: case TY_TUPLE:
        return target;
    default:
        break;
    }
    TypeDecl *decl = jaiChkTypeDeclOf(target);
    if (decl != NULL && jaiChkFindMemberIndex(decl, "__getitem__") >= 0) return gTypes.tAny;
    if (decl == NULL && target->kind == TY_CLASS) return gTypes.tAny;

    char got[TYPE_BUF];
    jaiChkRenderType(target, got, sizeof got);
    ERR(c, E0404_NOT_INDEXABLE, node->span, "`%s` cannot be sliced", got);
    return gTypes.tAny;
}

static JaiType *checkRange(Checker *c, AstNode *node) {
    AstNode *ends[2] = { node->as.range.start, node->as.range.stop };
    for (int i = 0; i < 2; i++) {
        if (ends[i] == NULL) continue;
        JaiType *t = jaiChkValue(c, ends[i]);
        if (!jaiChkIsAny(t) && t->kind != TY_INT)
            jaiChkRequireAssignable(c, ends[i], t, gTypes.tInt, E0400_TYPE_MISMATCH,
                              "a range bound must be an int");
    }
    return gTypes.tRange;
}

static JaiType *checkFString(Checker *c, AstNode *node) {
    for (int i = 0; i < node->as.fstring.partCount; i++) {
        AstNode *part = node->as.fstring.parts[i];
        if (part == NULL) continue;
        if (part->kind == AST_STR_LIT) { part->type = gTypes.tStr; continue; }
        jaiChkValue(c, part);
    }
    return gTypes.tStr;
}

/* ------------------------------------------------------------------ */
/* Member access                                                        */
/* ------------------------------------------------------------------ */

bool jaiChkVisibleFrom(Checker *c, const TypeDecl *owner, AstVisibility vis) {
    if (vis == AST_VIS_PUBLIC || owner == NULL) return true;
    const TypeDecl *from = c->currentClass;
    if (from == NULL) return false;
    if (vis == AST_VIS_PROTECTED) return jaiTypeDeclIsSubclassOf(from, owner);
    return from == owner;
}

const char *jaiChkVisibilityWord(AstVisibility vis) {
    return vis == AST_VIS_PROTECTED ? "protected" : "private";
}

void jaiChkSuggestMember(JaiDiag *d, const TypeDecl *decl, const char *name) {
    const char *best = NULL;
    int bestDistance = JAI_SUGGEST_NO_MATCH;
    for (int i = 0; i < decl->fieldCount; i++)
        if (jaiChkCloseEnough(name, decl->fields[i].name, &bestDistance))
            best = decl->fields[i].name;
    for (int i = 0; i < decl->memberCount; i++)
        if (jaiChkCloseEnough(name, decl->members[i].name, &bestDistance))
            best = decl->members[i].name;
    if (best != NULL) jaiDiagAddHelp(d, "did you mean `%s`?", best);
}

/* Resolve `name` against a declared type. `staticAccess` distinguishes
 * `Account.count` from `account.count`; enum variants live in the field array
 * with their tag in `slot`. */
static JaiType *lookupMember(Checker *c, AstNode *site, TypeDecl *decl, const char *name,
                             bool staticAccess, AstNode **outDecl) {
    int fi = jaiChkFindFieldIndex(decl, name);
    if (fi >= 0) {
        const TypeDecl *owner = jaiChkOwnerOfField(decl, name);
        if (!jaiChkVisibleFrom(c, owner, decl->fields[fi].visibility)) {
            JaiDiag *d = ERR(c, E0701_PRIVATE_ACCESS, site->span,
                             "`%s` is %s to `%s`", name,
                             jaiChkVisibilityWord(decl->fields[fi].visibility), owner->name);
            jaiDiagAddLabel(d, owner->span, "declared here");
            jaiDiagAddHelp(d, "mark it `pub` or add an accessor method");
        }
        return jaiChkOrAny(decl->fields[fi].type);
    }

    int mi = jaiChkFindAccessorIndex(decl, name, false);
    if (mi < 0) mi = jaiChkFindMemberIndex(decl, name);
    if (mi >= 0) {
        const TypeDecl *owner = jaiChkOwnerOfMember(decl, name);
        if (!jaiChkVisibleFrom(c, owner, decl->members[mi].visibility)) {
            JaiDiag *d = ERR(c, E0701_PRIVATE_ACCESS, site->span,
                             "`%s` is %s to `%s`", name,
                             jaiChkVisibilityWord(decl->members[mi].visibility), owner->name);
            jaiDiagAddLabel(d, owner->span, "declared here");
        }
        if (outDecl != NULL) *outDecl = decl->members[mi].decl;
        return jaiChkOrAny(decl->members[mi].type);
    }

    /* A parent this unit cannot see may well declare the member. Saying "no
     * member" would be a guess, and it was a wrong one: every subclass of an
     * imported class lost the whole inherited surface. Defer to the runtime.
     *
     * An imported declaration is only read for its signatures, so the same
     * caution applies one step further out: it may carry a static extension or
     * a member its own imports contribute, none of which this reading sees. It
     * answers for what it does declare and stays quiet about the rest. */
    if (jaiChkInheritsOpaquely(decl)) return gTypes.tAny;

    char got[TYPE_BUF];
    jaiChkRenderType(jaiChkDeclType(decl), got, sizeof got);
    JaiDiag *d = ERR(c, E0410_UNKNOWN_MEMBER, site->span,
                     "`%s` has no member `%s`", got, name);
    jaiDiagAddLabel(d, decl->span, "`%s` is declared here", decl->name);
    jaiChkSuggestMember(d, decl, name);
    if (staticAccess && decl->isEnum)
        jaiDiagAddHelp(d, "`%s` has %d variant%s", decl->name, decl->variantCount,
                       decl->variantCount == 1 ? "" : "s");
    return gTypes.tAny;
}

/* The declaration behind `Account` in `Account.total()`, as opposed to an
 * instance of it. */
TypeDecl *jaiChkStaticTargetDecl(Checker *c, AstNode *object) {
    if (object == NULL || object->kind != AST_IDENT) return NULL;
    TypeDecl *decl = jaiChkIdentTypeDecl(object->as.ident.symbol);
    if (decl != NULL) object->type = jaiChkDeclType(decl);
    (void)c;
    return decl;
}

/* `import m` binds a module object, and `m.f` is resolved by name at run time.
 * The signature loader knows what `m` declares, so a call through one is
 * checked exactly like a call through `from m import f`. */
static ModuleSig *moduleSigOfIdent(const AstNode *object) {
    if (object == NULL || object->kind != AST_IDENT) return NULL;
    const Symbol *s = object->as.ident.symbol;
    if (s == NULL || s->kind != SYM_MODULE) return NULL;
    for (int i = 0; i < gJaiCheck.modules.count; i++)
        if (gJaiCheck.modules.data[i].sym == s) return gJaiCheck.modules.data[i].sig;
    return NULL;
}

JaiType *jaiChkForeignFunctionType(Checker *c, ModuleSig *origin, AstNode *fn) {
    ForeignCtx saved;
    jaiChkForeignBegin(c, origin, true, &saved);
    JaiType *t = jaiChkFunctionType(c, fn, false);
    jaiChkForeignEnd(c, &saved);
    return t;
}

static JaiType *moduleMemberType(Checker *c, ModuleSig *sig, const char *name,
                                 AstNode **outDecl) {
    ModuleSig *owner = sig;
    AstNode *decl = jaiModuleSigFind(sig, name, &owner);
    if (decl == NULL) return gTypes.tAny;   /* defined at run time, or private */
    switch (decl->kind) {
    case AST_FN_DECL:
        if (outDecl != NULL) *outDecl = decl;
        return jaiChkForeignFunctionType(c, owner, decl);
    case AST_CLASS_DECL:
    case AST_TRAIT_DECL:
    case AST_ENUM_DECL: {
        TypeDecl *d = jaiChkDeclForNode(decl);
        if (d == NULL) return gTypes.tAny;
        jaiChkLayoutDecl(c, jaiChkDeclEntry(d));
        return jaiChkDeclType(d);
    }
    default:
        return gTypes.tAny;
    }
}

JaiType *jaiChkMember(Checker *c, AstNode *node, AstNode **outDecl) {
    AstNode *object = node->as.member.object;
    const char *name = node->as.member.name;
    bool optional = node->kind == AST_OPT_MEMBER;

    TypeDecl *staticDecl = jaiChkStaticTargetDecl(c, object);
    if (staticDecl != NULL) {
        jaiChkLayoutDecl(c, jaiChkDeclEntry(staticDecl));
        return lookupMember(c, node, staticDecl, name, true, outDecl);
    }

    /* `super.method()` resolves in the parent, not in the current class. */
    if (object != NULL && object->kind == AST_SUPER) {
        TypeDecl *parent = c->currentClass != NULL ? c->currentClass->superclass : NULL;
        object->type = parent != NULL ? jaiChkDeclType(parent) : gTypes.tAny;
        if (parent == NULL) return gTypes.tAny;
        return lookupMember(c, node, parent, name, false, outDecl);
    }

    JaiType *target = jaiChkValue(c, object);

    ModuleSig *sig = moduleSigOfIdent(object);
    if (sig != NULL) {
        JaiType *result = moduleMemberType(c, sig, name, outDecl);
        return optional ? jaiTypeOptional(result) : result;
    }

    if (optional) {
        JaiType *present = jaiTypeNarrow(target, gTypes.tNull, false);
        if (present != NULL) target = present;
    }
    if (jaiChkIsAny(target) || jaiChkIsNever(target)) return gTypes.tAny;

    TypeDecl *decl = jaiChkTypeDeclOf(target);
    if (decl == NULL) {
        /* Builtin containers, modules, and the builtin Error classes carry
         * their methods in the runtime, not in a TypeDecl. */
        return gTypes.tAny;
    }
    jaiChkLayoutDecl(c, jaiChkDeclEntry(decl));

    if (jaiTypeIsOptional(target) && !optional) {
        char got[TYPE_BUF];
        jaiChkRenderType(target, got, sizeof got);
        JaiDiag *d = ERR(c, E0400_TYPE_MISMATCH, node->span,
                         "`%s` may be null, so `.%s` is not safe", got, name);
        jaiDiagAddHelp(d, "use `?.%s`, or test with `!= null` first", name);
    }

    JaiType *result = lookupMember(c, node, decl, name, false, outDecl);
    return optional ? jaiTypeOptional(result) : result;
}

/* ------------------------------------------------------------------ */
/* Calls                                                                */
/* ------------------------------------------------------------------ */

typedef struct {
    AstNode    *decl;        /* AST_FN_DECL when the callee is statically known */
    JaiType    *type;        /* TY_FN, may be NULL */
    const char *name;
    int         selfSkip;    /* 1 when the declaration writes `self` first */
    bool        named;       /* a named callee reports E0602 rather than E0600 */
    /* When the callee is a value rather than a function declaration there is
     * no `fn` to point at, and an arity error that only states a number leaves
     * the reader guessing which `name` was checked. These say where the value
     * was bound and what kind of binding it is. */
    JaiSpan     bindingSpan;
    const char *bindingWhat;
    /* Set when that binding hides a method of the same name on the enclosing
     * class: the arity then looks wrong only because the reader is reading the
     * method's signature, which is the whole confusion worth naming. */
    const char *shadowedIn;
} CalleeInfo;

/* What to call a binding in prose. */
static const char *bindingWord(SymbolKind kind) {
    switch (kind) {
    case SYM_PARAM:         return "parameter";
    case SYM_LOCAL:         return "local binding";
    case SYM_UPVALUE:       return "captured binding";
    case SYM_GLOBAL:        return "global binding";
    case SYM_FIELD:         return "field";
    case SYM_METHOD:        return "method";
    case SYM_CLASS:         return "class";
    case SYM_TRAIT:         return "trait";
    case SYM_ENUM:          return "enum";
    case SYM_ENUM_VARIANT:  return "enum variant";
    case SYM_MODULE:        return "module";
    case SYM_TYPE_ALIAS:    return "type alias";
    case SYM_GENERIC_PARAM: return "generic parameter";
    case SYM_BUILTIN:       return "builtin";
    }
    return "binding";
}

/* The function declaration this name *denotes*, or NULL. `Symbol.decl` is the
 * node that DECLARES the symbol, which for a parameter, for `self`, and for a
 * generic parameter is the enclosing `fn` — so `decl->kind == AST_FN_DECL`
 * alone makes every parameter masquerade as the function it belongs to, and a
 * call through an `fn`-typed parameter gets bound to its own function's
 * parameter list. Only the declaration's back-link, written wherever a `fn` is
 * bound to a name, says which binding actually names the function. An upvalue
 * inherits the captured symbol's `decl` and so is correctly rejected here: an
 * alias is not the declaration. */
static AstNode *denotedFunctionDecl(const Symbol *sym) {
    if (sym == NULL) return NULL;
    /* `from m import f` binds a name whose declaring node is the import
     * statement. What it *denotes* is the `fn` the signature loader found for
     * it in `m`, which is the only reason a call across a module boundary can
     * be checked at all. */
    if (sym->importedDecl != NULL)
        return sym->importedDecl->kind == AST_FN_DECL ? sym->importedDecl : NULL;
    if (sym->decl == NULL || sym->decl->kind != AST_FN_DECL) return NULL;
    return sym->decl->as.fn.symbol == sym ? sym->decl : NULL;
}

int jaiChkSelfSkipOf(const AstNode *fn) {
    if (fn == NULL || fn->kind != AST_FN_DECL) return 0;
    if (fn->as.fn.isStatic || fn->as.fn.paramCount <= 0 || fn->as.fn.params == NULL) return 0;
    return jaiChkSameName(fn->as.fn.params[0].name, "self") ? 1 : 0;
}

static void checkArgExpr(Checker *c, AstArg *arg, JaiType *want, const char *what) {
    if (want != NULL) jaiChkApplyContext(c, arg->value, want);
    JaiType *got = jaiChkValue(c, arg->value);
    if (want != NULL) jaiChkRequireAssignable(c, arg->value, got, want, E0400_TYPE_MISMATCH, what);
}

/* Bind arguments to parameters and type-check each one. Every argument is
 * checked exactly once, whether or not it found a home. */
static void checkArgs(Checker *c, AstNode *call, const CalleeInfo *info) {
    AstArg *args = call->as.call.args;
    int argCount = call->as.call.argCount;
    JaiDiagCode arityCode = info->named ? E0602_BAD_CALL_ARITY : E0600_ARITY_MISMATCH;
    const char *name = info->name != NULL ? info->name : "this callable";

    /* Per-argument expected type and the parameter name it bound to, so the
     * final pass can check each argument once with a precise label. */
    JaiType *targets[JAI_MAX_ARGS];
    const char *bound[JAI_MAX_ARGS];
    int tracked = argCount < JAI_MAX_ARGS ? argCount : JAI_MAX_ARGS;
    for (int i = 0; i < tracked; i++) {
        targets[i] = NULL;
        bound[i] = NULL;
    }

    bool sawKeyword = false, sawSpread = false;
    for (int i = 0; i < argCount; i++) {
        if (args[i].isSpread) { sawSpread = true; continue; }
        if (args[i].name == NULL) {
            if (sawKeyword)
                ERR(c, E0605_POSITIONAL_AFTER_KEYWORD, args[i].span,
                    "positional argument follows a keyword argument");
            continue;
        }
        sawKeyword = true;
        for (int j = 0; j < i; j++)
            if (args[j].name != NULL && jaiChkSameName(args[j].name, args[i].name)) {
                JaiDiag *d = ERR(c, E0604_DUPLICATE_ARGUMENT, args[i].span,
                                 "argument `%s` is given twice", args[i].name);
                jaiDiagAddLabel(d, args[j].span, "first given here");
                break;
            }
    }

    AstParam *params = NULL;
    int paramCount = 0;
    bool declared = info->decl != NULL && info->decl->kind == AST_FN_DECL;
    if (declared) {
        paramCount = info->decl->as.fn.paramCount - info->selfSkip;
        if (paramCount < 0) paramCount = 0;
        if (paramCount > 0) params = info->decl->as.fn.params + info->selfSkip;
    }

    /* Resolved once, up front, because a declaration in another module has to
     * be read in *its* namespace and the loop below reports this file's errors
     * in this file's. `int` and `list[str]` mean the same thing everywhere; a
     * name that module imported and this one did not is `any`. */
    JaiType *paramTypes[JAI_MAX_ARGS];
    if (declared && paramCount <= JAI_MAX_ARGS) {
        ForeignCtx saved;
        jaiChkForeignBegin(c, jaiChkOriginOfNode(info->decl), true, &saved);
        for (int i = 0; i < paramCount; i++)
            paramTypes[i] = jaiChkOrAny(params[i].type == NULL
                                      ? gTypes.tAny
                                      : jaiChkResolveAstType(c, params[i].type));
        jaiChkForeignEnd(c, &saved);
    }

    if (declared && paramCount <= JAI_MAX_ARGS) {
        unsigned char filled[JAI_MAX_ARGS];
        memset(filled, 0, sizeof filled);
        int variadic = -1, kwRest = -1;
        for (int i = 0; i < paramCount; i++) {
            if (params[i].isVariadic) variadic = i;
            if (params[i].isKwRest) kwRest = i;
        }

        int positional = 0;
        bool reportedOverflow = false;
        for (int i = 0; i < argCount; i++) {
            if (args[i].isSpread) continue;
            if (args[i].name == NULL) {
                int slot = -1;
                if (variadic >= 0 && positional >= variadic) slot = variadic;
                else if (positional < paramCount && !params[positional].isKwRest)
                    slot = positional;

                if (slot < 0) {
                    if (!sawSpread && !reportedOverflow) {
                        JaiDiag *d = ERR(c, arityCode, args[i].span,
                                         "`%s` takes %d argument%s but %d %s given",
                                         name, paramCount, paramCount == 1 ? "" : "s",
                                         argCount, argCount == 1 ? "was" : "were");
                        if (info->decl != NULL)
                            jaiDiagAddLabel(d, info->decl->span,
                                            "`%s` is declared here", name);
                        reportedOverflow = true;
                    }
                } else {
                    if (slot != variadic) filled[slot] = 1;
                    if (i < tracked) {
                        targets[i] = paramTypes[slot];
                        bound[i] = params[slot].name;
                    }
                }
                positional++;
                continue;
            }

            int slot = -1;
            for (int p = 0; p < paramCount; p++) {
                if (params[p].isVariadic || params[p].isKwRest) continue;
                if (jaiChkSameName(params[p].name, args[i].name)) { slot = p; break; }
            }
            if (slot < 0) {
                if (kwRest < 0) {
                    JaiDiag *d = ERR(c, E0603_UNKNOWN_KEYWORD_ARG, args[i].span,
                                     "`%s` has no parameter named `%s`", name,
                                     args[i].name);
                    const char *best = NULL;
                    int bestDistance = JAI_SUGGEST_NO_MATCH;
                    for (int p = 0; p < paramCount; p++)
                        if (jaiChkCloseEnough(args[i].name, params[p].name, &bestDistance))
                            best = params[p].name;
                    if (best != NULL) jaiDiagAddHelp(d, "did you mean `%s`?", best);
                }
                continue;
            }
            if (filled[slot] != 0)
                ERR(c, E0604_DUPLICATE_ARGUMENT, args[i].span,
                    "argument `%s` is given twice", args[i].name);
            filled[slot] = 1;
            if (i < tracked) {
                targets[i] = paramTypes[slot];
                bound[i] = params[slot].name;
            }
        }

        if (!sawSpread) {
            int missing = 0;
            const char *firstMissing = NULL;
            for (int p = 0; p < paramCount; p++) {
                if (filled[p] != 0 || params[p].isVariadic || params[p].isKwRest) continue;
                if (params[p].defaultValue != NULL) continue;
                if (firstMissing == NULL) firstMissing = params[p].name;
                missing++;
            }
            if (missing > 0) {
                JaiDiag *d = ERR(c, arityCode, call->span,
                                 "`%s` is missing %d required argument%s", name,
                                 missing, missing == 1 ? "" : "s");
                if (firstMissing != NULL)
                    jaiDiagAddHelp(d, "the first one is `%s`", firstMissing);
                if (info->decl != NULL)
                    jaiDiagAddLabel(d, info->decl->span, "`%s` is declared here", name);
            }
        }
    } else if (info->type != NULL && info->type->kind == TY_FN && !sawSpread &&
               !sawKeyword) {
        int want = info->type->argCount;
        bool variadic = (info->type->fnFlags & FN_FLAG_VARIADIC) != 0;
        if (argCount != want && !(variadic && argCount >= want - 1)) {
            JaiDiag *d = ERR(c, arityCode, call->span,
                             "`%s` takes %d argument%s but %d %s given", name, want,
                             want == 1 ? "" : "s", argCount,
                             argCount == 1 ? "was" : "were");
            /* Show the signature that was actually checked, at its binding. */
            if (info->bindingWhat != NULL && jaiSpanValid(info->bindingSpan)) {
                char sig[TYPE_BUF];
                jaiChkRenderType(info->type, sig, sizeof sig);
                jaiDiagAddLabel(d, info->bindingSpan,
                                "`%s` is the %s declared here, of type `%s`", name,
                                info->bindingWhat, sig);
            }
            if (info->shadowedIn != NULL)
                jaiDiagAddHelp(d,
                               "this `%s` is the %s, which shadows the method "
                               "`%s.%s`; write `self.%s(...)` to call the method",
                               name, info->bindingWhat, info->shadowedIn, name, name);
        }
        for (int i = 0; i < tracked && i < want; i++) targets[i] = info->type->args[i];
    }

    for (int i = 0; i < argCount; i++) {
        if (i >= tracked) { jaiChkValue(c, args[i].value); continue; }
        char label[96];
        if (bound[i] != NULL)
            snprintf(label, sizeof label, "argument `%s` of `%s`", bound[i], name);
        else
            snprintf(label, sizeof label, "argument %d of `%s`", i + 1, name);
        checkArgExpr(c, &args[i], args[i].isSpread ? NULL : targets[i], label);
    }
}

/* The init of `decl` or of the nearest ancestor that declares one. */
AstNode *jaiChkFindInitDecl(TypeDecl *decl, TypeDecl **owner) {
    for (TypeDecl *d = decl; d != NULL; d = d->superclass) {
        int mi = jaiChkFindMemberIndex(d, "init");
        if (mi >= 0 && d->members[mi].decl != NULL) {
            if (owner != NULL) *owner = d;
            return d->members[mi].decl;
        }
    }
    return NULL;
}

static JaiType *checkConstruct(Checker *c, AstNode *call, TypeDecl *decl) {
    jaiChkLayoutDecl(c, jaiChkDeclEntry(decl));

    if (decl->isTrait) {
        ERR(c, E0710_ABSTRACT_INSTANTIATION, call->span,
            "`%s` is a trait and cannot be instantiated", decl->name);
    } else if (decl->isAbstract) {
        JaiDiag *d = ERR(c, E0710_ABSTRACT_INSTANTIATION, call->span,
                         "`%s` is abstract and cannot be instantiated", decl->name);
        jaiDiagAddLabel(d, decl->span, "declared abstract here");
        for (int i = 0; i < decl->memberCount; i++)
            if (decl->members[i].isAbstract) {
                jaiDiagAddHelp(d, "`%s` has no implementation", decl->members[i].name);
                break;
            }
    }

    CalleeInfo info;
    memset(&info, 0, sizeof info);
    info.name = decl->name;
    info.named = true;
    info.decl = jaiChkFindInitDecl(decl, NULL);
    info.selfSkip = jaiChkSelfSkipOf(info.decl);
    if (info.decl == NULL && call->as.call.argCount > 0) {
        ERR(c, E0602_BAD_CALL_ARITY, call->span,
            "`%s` has no `init`, so it takes no arguments", decl->name);
    }
    checkArgs(c, call, &info);
    return jaiChkDeclType(decl);
}

static JaiType *checkCall(Checker *c, AstNode *node) {
    AstNode *callee = node->as.call.callee;
    if (callee == NULL) return gTypes.tAny;

    /* ClassName(...) constructs; there is no `new`. */
    if (callee->kind == AST_IDENT) {
        TypeDecl *decl = jaiChkIdentTypeDecl(callee->as.ident.symbol);
        if (decl != NULL && !decl->isEnum) {
            callee->type = jaiChkDeclType(decl);
            return checkConstruct(c, node, decl);
        }
    }

    /* super(...) runs the parent constructor. */
    if (callee->kind == AST_SUPER) {
        TypeDecl *parent = c->currentClass != NULL ? c->currentClass->superclass : NULL;
        callee->type = parent != NULL ? jaiChkDeclType(parent) : gTypes.tAny;
        CalleeInfo info;
        memset(&info, 0, sizeof info);
        info.name = "super";
        info.named = true;
        if (parent != NULL) {
            info.decl = jaiChkFindInitDecl(parent, NULL);
            info.selfSkip = jaiChkSelfSkipOf(info.decl);
            info.name = parent->name;
        }
        checkArgs(c, node, &info);
        return gTypes.tVoid;
    }

    AstNode *calleeDecl = NULL;
    JaiType *fnType;
    /* `a?.m(x)` short-circuits the whole call, not just the member load (spec
     * §4.2): when `a` is null nothing is called and the call yields null. So
     * the optional jaiChkMember put on the member is taken off the callee and
     * put back on the result. */
    bool optChain = callee->kind == AST_OPT_MEMBER;
    if (callee->kind == AST_MEMBER || callee->kind == AST_OPT_MEMBER) {
        fnType = jaiChkMember(c, callee, &calleeDecl);
        if (optChain) {
            JaiType *present = jaiTypeNarrow(fnType, gTypes.tNull, false);
            if (present != NULL) fnType = present;
        }
        callee->type = fnType;
        /* `m.Thing(...)` after `import m` names a class, not a value: the
         * member load yielded the declaration itself, so this is construction
         * and not a call of something that happens to be a class. */
        if (moduleSigOfIdent(callee->as.member.object) != NULL) {
            TypeDecl *classDecl = jaiChkTypeDeclOf(fnType);
            if (classDecl != NULL && !classDecl->isEnum)
                return checkConstruct(c, node, classDecl);
        }
    } else {
        fnType = jaiChkValue(c, callee);
        if (callee->kind == AST_IDENT)
            calleeDecl = denotedFunctionDecl(callee->as.ident.symbol);
    }

    CalleeInfo info;
    memset(&info, 0, sizeof info);
    info.decl = calleeDecl;
    info.type = (fnType != NULL && fnType->kind == TY_FN) ? fnType : NULL;
    info.selfSkip = jaiChkSelfSkipOf(calleeDecl);
    info.bindingSpan = JAI_SPAN_NONE;   /* memset leaves file 0, a real file */
    if (callee->kind == AST_IDENT) {
        info.name = callee->as.ident.name;
        info.named = true;
        /* A name that denotes no declaration is a value being called. Record
         * its binding so an arity error can show the signature it checked. */
        Symbol *sym = callee->as.ident.symbol;
        if (calleeDecl == NULL && sym != NULL) {
            info.bindingSpan = sym->declSpan;
            info.bindingWhat = bindingWord(sym->kind);
            /* Members are copied down during layout, so one lookup covers
             * inherited and trait methods too. */
            if (c->currentClass != NULL &&
                (sym->kind == SYM_PARAM || sym->kind == SYM_LOCAL ||
                 sym->kind == SYM_UPVALUE) &&
                jaiChkFindMemberIndex(c->currentClass, sym->name) >= 0) {
                const TypeDecl *owner = jaiChkOwnerOfMember(c->currentClass, sym->name);
                if (owner != NULL) info.shadowedIn = owner->name;
            }
        }
    } else if (callee->kind == AST_MEMBER || callee->kind == AST_OPT_MEMBER) {
        info.name = callee->as.member.name;
        info.named = calleeDecl != NULL;
    }

    /* An enum variant with a payload is spelled as a call. */
    if (fnType != NULL && fnType->kind == TY_FN) {
        checkArgs(c, node, &info);
        JaiType *result = jaiChkOrAny(fnType->ret);
        return optChain ? jaiTypeOptional(result) : result;
    }

    if (jaiChkIsAny(fnType) || jaiChkIsNever(fnType)) {
        info.decl = NULL;
        info.type = NULL;
        checkArgs(c, node, &info);
        return gTypes.tAny;
    }

    TypeDecl *decl = jaiChkTypeDeclOf(fnType);
    if (decl != NULL) jaiChkLayoutDecl(c, jaiChkDeclEntry(decl));
    if (decl != NULL && jaiChkFindMemberIndex(decl, "__call__") >= 0) {
        info.decl = NULL;
        info.type = NULL;
        checkArgs(c, node, &info);
        return gTypes.tAny;
    }
    if (jaiTypeIsCallable(fnType)) {
        info.decl = NULL;
        info.type = NULL;
        checkArgs(c, node, &info);
        return gTypes.tAny;
    }

    char got[TYPE_BUF];
    jaiChkRenderType(fnType, got, sizeof got);
    JaiDiag *d = ERR(c, E0403_NOT_CALLABLE, node->span, "`%s` is not callable", got);
    if (decl != NULL) jaiDiagAddHelp(d, "define `__call__` on `%s`", got);
    info.decl = NULL;
    info.type = NULL;
    checkArgs(c, node, &info);
    return gTypes.tAny;
}

/* ------------------------------------------------------------------ */
/* Comprehensions and function expressions                              */
/* ------------------------------------------------------------------ */

static JaiType *checkComprehension(Checker *c, AstNode *node) {
    TypeKind shape = node->as.comp.kind == COMP_SET    ? TY_SET
                     : node->as.comp.kind == COMP_DICT ? TY_DICT
                                                       : TY_LIST;
    JaiType *want = takeContainerContext(node, shape, shape == TY_DICT ? 2 : 1);
    int mark = jaiChkNarrowMark();
    for (int i = 0; i < node->as.comp.clauseCount; i++) {
        AstCompClause *clause = &node->as.comp.clauses[i];
        JaiType *source = jaiChkValue(c, clause->iterable);
        JaiType *elem = jaiChkIterableElement(c, clause->iterable, source);
        jaiChkPattern(c, clause->pattern, elem);
        for (int j = 0; j < clause->conditionCount; j++)
            jaiChkCondition(c, clause->conditions[j], "a comprehension filter");
    }

    JaiType *value = node->as.comp.element != NULL
                         ? jaiChkValue(c, node->as.comp.element)
                         : gTypes.tAny;
    JaiType *key = node->as.comp.keyExpr != NULL ? jaiChkValue(c, node->as.comp.keyExpr)
                                                 : NULL;
    jaiChkNarrowRestore(mark);

    if (want != NULL) {
        /* Annotated: report the element that does not fit rather than the whole
         * comprehension failing against its annotation. */
        if (node->as.comp.keyExpr != NULL)
            jaiChkRequireAssignable(c, node->as.comp.keyExpr, jaiChkOrAny(key), want->args[0],
                              E0400_TYPE_MISMATCH, "mismatched key type");
        jaiChkRequireAssignable(c, node->as.comp.element, value,
                          want->args[want->argCount - 1], E0400_TYPE_MISMATCH,
                          "mismatched element type");
        return want;
    }

    switch (node->as.comp.kind) {
    case COMP_LIST: return jaiTypeList(value);
    case COMP_SET:  return jaiTypeSet(value);
    case COMP_DICT: return jaiTypeDict(jaiChkOrAny(key), value);
    case COMP_GENERATOR:
    default:        return gTypes.tAny;
    }
}

static JaiType *checkFunctionExpr(Checker *c, AstNode *node) {
    jaiChkFunction(c, node, NULL, false);
    JaiType *t = jaiChkFunctionType(c, node, false);
    if (node->as.fn.symbol != NULL) node->as.fn.symbol->type = t;
    return t;
}

static JaiType *checkIfExpr(Checker *c, AstNode *node) {
    jaiChkCondition(c, node->as.conditional.cond, "an `if`");

    NarrowFact facts[MAX_FACTS];
    int count = jaiChkCollectFacts(c, node->as.conditional.cond, facts, MAX_FACTS, false);

    int mark = jaiChkNarrowMark();
    jaiChkNarrowApply(facts, count, false);
    JaiType *thenType = jaiChkExpr(c, node->as.conditional.thenBranch);
    jaiChkNarrowRestore(mark);

    if (node->as.conditional.elseBranch == NULL) return jaiTypeOptional(jaiChkOrAny(thenType));

    NarrowFact elseFacts[MAX_FACTS];
    int elseCount = jaiChkCollectFacts(c, node->as.conditional.cond, elseFacts, MAX_FACTS, true);
    mark = jaiChkNarrowMark();
    jaiChkNarrowApply(elseFacts, elseCount, false);
    JaiType *elseType = jaiChkExpr(c, node->as.conditional.elseBranch);
    jaiChkNarrowRestore(mark);
    return jaiTypeJoin(jaiChkOrAny(thenType), jaiChkOrAny(elseType));
}

JaiType *jaiChkMatchExpr(Checker *c, AstNode *node, bool isStatement);

JaiType *jaiChkExpr(Checker *c, AstNode *node) {
    if (node == NULL) return gTypes.tAny;

    JaiType *t;
    switch (node->kind) {
    case AST_INT_LIT:   t = gTypes.tInt;   break;
    case AST_FLOAT_LIT: t = gTypes.tFloat; break;
    case AST_STR_LIT:   t = gTypes.tStr;   break;
    case AST_BOOL_LIT:  t = gTypes.tBool;  break;
    case AST_NULL_LIT:  t = gTypes.tNull;  break;
    case AST_FSTRING:   t = checkFString(c, node); break;

    case AST_IDENT:
        t = checkIdent(c, node);
        /* checkIdent may have folded the node into a literal. */
        if (node->kind != AST_IDENT) return jaiChkOrAny(node->type);
        break;

    case AST_SELF:
        t = c->currentClass != NULL ? jaiChkDeclType(c->currentClass) : gTypes.tAny;
        break;

    case AST_SUPER:
        t = (c->currentClass != NULL && c->currentClass->superclass != NULL)
                ? jaiChkDeclType(c->currentClass->superclass)
                : gTypes.tAny;
        break;

    case AST_LIST_LIT:
    case AST_SET_LIT:
    case AST_TUPLE_LIT:
        t = checkSequenceLiteral(c, node);
        break;

    case AST_DICT_LIT:      t = checkDictLiteral(c, node); break;
    case AST_UNARY:         t = checkUnary(c, node);       break;
    case AST_BINARY:        t = checkBinary(c, node);      break;
    case AST_LOGICAL:       t = checkLogical(c, node);     break;
    case AST_COMPARE_CHAIN: t = checkChain(c, node);       break;
    case AST_TERNARY:       t = checkTernary(c, node);     break;
    case AST_COALESCE:      t = checkCoalesce(c, node);    break;
    case AST_CALL:          t = checkCall(c, node);        break;
    case AST_INDEX:         t = checkIndex(c, node);       break;
    case AST_SLICE:         t = jaiChkSlice(c, node);       break;

    case AST_MEMBER:
    case AST_OPT_MEMBER:
        t = jaiChkMember(c, node, NULL);
        break;

    case AST_LAMBDA:
    case AST_ANON_FN:
        t = checkFunctionExpr(c, node);
        break;

    case AST_COMPREHENSION: t = checkComprehension(c, node); break;
    case AST_RANGE:         t = checkRange(c, node);         break;
    case AST_IF_EXPR:       t = checkIfExpr(c, node);        break;
    case AST_MATCH_EXPR:    t = jaiChkMatchExpr(c, node, false); break;

    case AST_CAST: {
        JaiType *want = jaiChkResolveAstType(c, node->as.cast.target);
        jaiChkValue(c, node->as.cast.operand);
        t = want;
        break;
    }

    case AST_YIELD:
    case AST_AWAIT:
        if (node->as.wrap.operand != NULL) jaiChkValue(c, node->as.wrap.operand);
        t = gTypes.tAny;
        break;

    /* A block in expression position is the body of a lambda or of an `if`
     * expression arm; its value is the value of its last expression. */
    case AST_BLOCK:
        jaiChkBlock(c, node);
        t = gTypes.tAny;
        break;

    default:
        if (jaiAstIsStatement(node->kind)) { jaiChkStmt(c, node); t = gTypes.tVoid; break; }
        t = gTypes.tAny;
        break;
    }

    node->type = jaiChkOrAny(t);
    jaiChkTryFold(c, node);
    return jaiChkOrAny(node->type);
}

JaiType *jaiCheckExpr(Checker *c, AstNode *expr) {
    if (c == NULL || expr == NULL) return gTypes.tAny;
    return jaiChkExpr(c, expr);
}
