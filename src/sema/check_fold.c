/* check_fold.c — the constant folder and evaluator.
 *
 * Two jobs that are one implementation. `jaiConstEval` answers "what is this
 * expression's value, if it has one at compile time", which spec §2.2 needs
 * for a `const` initialiser and §5.3 for a match arm; `jaiChkTryFold` is the
 * same walk run for its side effect, replacing a node in place with the
 * literal it evaluates to.
 *
 * Arithmetic here is overflow-checked rather than wrapping: folding
 * `2 ** 70` must report E0900 and not quietly produce a different program
 * from the one the VM would have run.
 */
#include "check_internal.h"

#include <math.h>

/* ------------------------------------------------------------------ */
/* Overflow-checked integer arithmetic                                  */
/* ------------------------------------------------------------------ */

#if defined(__GNUC__) || defined(__clang__)
#  define ADD_OVF(a, b, r) __builtin_add_overflow((a), (b), (r))
#  define SUB_OVF(a, b, r) __builtin_sub_overflow((a), (b), (r))
#  define MUL_OVF(a, b, r) __builtin_mul_overflow((a), (b), (r))
#else
/* Division-based fallback: every test is performed before the operation, so
 * signed overflow is never executed. */
static bool addOverflow(int64_t a, int64_t b, int64_t *r) {
    if ((b > 0 && a > INT64_MAX - b) || (b < 0 && a < INT64_MIN - b)) return true;
    *r = a + b;
    return false;
}
static bool subOverflow(int64_t a, int64_t b, int64_t *r) {
    if ((b < 0 && a > INT64_MAX + b) || (b > 0 && a < INT64_MIN + b)) return true;
    *r = a - b;
    return false;
}
static bool mulOverflow(int64_t a, int64_t b, int64_t *r) {
    if (a == 0 || b == 0) { *r = 0; return false; }
    if (a == -1) { if (b == INT64_MIN) return true; *r = -b; return false; }
    if (b == -1) { if (a == INT64_MIN) return true; *r = -a; return false; }
    bool bad = a > 0 ? (b > 0 ? a > INT64_MAX / b : b < INT64_MIN / a)
                     : (b > 0 ? a < INT64_MIN / b : a < INT64_MAX / b);
    if (bad) return true;
    *r = a * b;
    return false;
}
#  define ADD_OVF(a, b, r) addOverflow((a), (b), (r))
#  define SUB_OVF(a, b, r) subOverflow((a), (b), (r))
#  define MUL_OVF(a, b, r) mulOverflow((a), (b), (r))
#endif

/* ------------------------------------------------------------------ */
/* Constant folding                                                     */
/* ------------------------------------------------------------------ */

static ConstValue constNone(void) {
    ConstValue v;
    memset(&v, 0, sizeof v);
    v.kind = CONST_NONE;
    return v;
}

static ConstValue constInt(int64_t i) {
    ConstValue v = constNone();
    v.kind = CONST_INT;
    v.as.i = i;
    return v;
}

static ConstValue constFloat(double f) {
    ConstValue v = constNone();
    v.kind = CONST_FLOAT;
    v.as.f = f;
    return v;
}

static ConstValue constBool(bool b) {
    ConstValue v = constNone();
    v.kind = CONST_BOOL;
    v.as.b = b;
    return v;
}

static ConstValue constStr(const char *chars, size_t length) {
    ConstValue v = constNone();
    v.kind = CONST_STR;
    v.as.s.chars = chars;
    v.as.s.length = length;
    return v;
}

static bool constNumeric(const ConstValue *v) {
    return v->kind == CONST_INT || v->kind == CONST_FLOAT;
}

static double constAsFloat(const ConstValue *v) {
    return v->kind == CONST_INT ? (double)v->as.i : v->as.f;
}

/* Overflow and division by zero are only errors where a constant is required.
 * Everywhere else the fold simply gives up and the VM raises OverflowError or
 * DivisionByZeroError at run time, which is what the language promises. */
static ConstValue constFail(Checker *c, JaiDiagCode code, JaiSpan span,
                            const char *fmt, const char *detail) {
    if (gJaiCheck.constRequired) {
        jaiChkCountError(c);
        jaiDiagError(code, span, fmt, detail);
    }
    return constNone();
}

static bool ipow(int64_t base, int64_t exp, int64_t *out) {
    int64_t result = 1;
    while (exp > 0) {
        if ((exp & 1) != 0 && MUL_OVF(result, base, &result)) return false;
        exp >>= 1;
        if (exp > 0 && MUL_OVF(base, base, &base)) return false;
    }
    *out = result;
    return true;
}

static int64_t ifloordiv(int64_t a, int64_t b) {
    int64_t q = a / b;
    if ((a % b) != 0 && ((a < 0) != (b < 0))) q--;
    return q;
}

static int64_t imod(int64_t a, int64_t b) {
    int64_t r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) r += b;
    return r;
}

static ConstValue foldIntBinary(Checker *c, OpKind op, int64_t a, int64_t b, JaiSpan span) {
    int64_t r = 0;
    switch (op) {
    case OPK_ADD:
        if (ADD_OVF(a, b, &r)) return constFail(c, E0900_CONST_OVERFLOW, span,
                                                "integer overflow in constant `%s`", "+");
        return constInt(r);
    case OPK_SUB:
        if (SUB_OVF(a, b, &r)) return constFail(c, E0900_CONST_OVERFLOW, span,
                                                "integer overflow in constant `%s`", "-");
        return constInt(r);
    case OPK_MUL:
        if (MUL_OVF(a, b, &r)) return constFail(c, E0900_CONST_OVERFLOW, span,
                                                "integer overflow in constant `%s`", "*");
        return constInt(r);
    case OPK_ADD_WRAP: return constInt((int64_t)((uint64_t)a + (uint64_t)b));
    case OPK_SUB_WRAP: return constInt((int64_t)((uint64_t)a - (uint64_t)b));
    case OPK_MUL_WRAP: return constInt((int64_t)((uint64_t)a * (uint64_t)b));

    /* `/` on two ints yields float (BYTECODE §3.3), so it folds as one. */
    case OPK_DIV:
        if (b == 0) return constFail(c, E0901_CONST_DIVIDE_BY_ZERO, span,
                                     "division by zero in constant `%s`", "/");
        return constFloat((double)a / (double)b);

    case OPK_FLOORDIV:
        if (b == 0) return constFail(c, E0901_CONST_DIVIDE_BY_ZERO, span,
                                     "division by zero in constant `%s`", "//");
        if (a == INT64_MIN && b == -1)
            return constFail(c, E0900_CONST_OVERFLOW, span,
                             "integer overflow in constant `%s`", "//");
        return constInt(ifloordiv(a, b));

    case OPK_MOD:
        if (b == 0) return constFail(c, E0901_CONST_DIVIDE_BY_ZERO, span,
                                     "division by zero in constant `%s`", "%");
        if (a == INT64_MIN && b == -1) return constInt(0);
        return constInt(imod(a, b));

    case OPK_POW:
        if (b < 0) return constNone();   /* result is a float; leave it to the VM */
        if (!ipow(a, b, &r)) return constFail(c, E0900_CONST_OVERFLOW, span,
                                              "integer overflow in constant `%s`", "**");
        return constInt(r);

    case OPK_BAND: return constInt(a & b);
    case OPK_BOR:  return constInt(a | b);
    case OPK_BXOR: return constInt(a ^ b);

    /* These must fold to exactly what `bitwise()` in the VM computes, or a
     * constant-folded expression means something different from the same
     * expression behind a variable. Discarded bits are not an overflow; only a
     * negative count is an error, and it is one the VM raises too. */
    case OPK_SHL:
        if (b < 0) return constFail(c, E0900_CONST_OVERFLOW, span,
                                    "negative shift count in constant `%s`", "<<");
        return constInt(b >= 64 ? 0 : (int64_t)((uint64_t)a << (uint64_t)b));

    case OPK_SHR:
        if (b < 0) return constFail(c, E0900_CONST_OVERFLOW, span,
                                    "negative shift count in constant `%s`", ">>");
        return constInt(b >= 64 ? (a < 0 ? -1 : 0) : (a >> b));

    case OPK_EQ: return constBool(a == b);
    case OPK_NE: return constBool(a != b);
    case OPK_LT: return constBool(a < b);
    case OPK_LE: return constBool(a <= b);
    case OPK_GT: return constBool(a > b);
    case OPK_GE: return constBool(a >= b);
    default:     return constNone();
    }
}

static ConstValue foldFloatBinary(Checker *c, OpKind op, double a, double b, JaiSpan span) {
    switch (op) {
    case OPK_ADD: return constFloat(a + b);
    case OPK_SUB: return constFloat(a - b);
    case OPK_MUL: return constFloat(a * b);
    case OPK_DIV:
        if (b == 0.0) return constFail(c, E0901_CONST_DIVIDE_BY_ZERO, span,
                                       "division by zero in constant `%s`", "/");
        return constFloat(a / b);
    case OPK_FLOORDIV:
        if (b == 0.0) return constFail(c, E0901_CONST_DIVIDE_BY_ZERO, span,
                                       "division by zero in constant `%s`", "//");
        return constFloat(floor(a / b));
    case OPK_MOD: {
        if (b == 0.0) return constFail(c, E0901_CONST_DIVIDE_BY_ZERO, span,
                                       "division by zero in constant `%s`", "%");
        double r = fmod(a, b);
        if (r != 0.0 && ((r < 0.0) != (b < 0.0))) r += b;
        return constFloat(r);
    }
    case OPK_POW: return constFloat(pow(a, b));
    case OPK_EQ:  return constBool(a == b);
    case OPK_NE:  return constBool(a != b);
    case OPK_LT:  return constBool(a < b);
    case OPK_LE:  return constBool(a <= b);
    case OPK_GT:  return constBool(a > b);
    case OPK_GE:  return constBool(a >= b);
    default:      return constNone();
    }
}

static ConstValue foldStrBinary(Checker *c, OpKind op, const ConstValue *a,
                                const ConstValue *b) {
    if (op == OPK_ADD && b->kind == CONST_STR) {
        size_t len = a->as.s.length + b->as.s.length;
        char *buf = (char *)jaiArenaAlloc(&c->ast->arena, len + 1);
        memcpy(buf, a->as.s.chars, a->as.s.length);
        memcpy(buf + a->as.s.length, b->as.s.chars, b->as.s.length);
        buf[len] = '\0';
        return constStr(buf, len);
    }
    if (b->kind != CONST_STR) return constNone();

    size_t n = a->as.s.length < b->as.s.length ? a->as.s.length : b->as.s.length;
    int cmp = n == 0 ? 0 : memcmp(a->as.s.chars, b->as.s.chars, n);
    if (cmp == 0) cmp = a->as.s.length == b->as.s.length
                            ? 0
                            : (a->as.s.length < b->as.s.length ? -1 : 1);
    switch (op) {
    case OPK_EQ: return constBool(cmp == 0);
    case OPK_NE: return constBool(cmp != 0);
    case OPK_LT: return constBool(cmp < 0);
    case OPK_LE: return constBool(cmp <= 0);
    case OPK_GT: return constBool(cmp > 0);
    case OPK_GE: return constBool(cmp >= 0);
    default:     return constNone();
    }
}

ConstValue jaiConstEval(Checker *c, AstNode *node) {
    if (c == NULL || node == NULL) return constNone();

    switch (node->kind) {
    case AST_INT_LIT:   return constInt(node->as.intLit);
    case AST_FLOAT_LIT: return constFloat(node->as.floatLit);
    case AST_BOOL_LIT:  return constBool(node->as.boolLit);
    case AST_STR_LIT:   return constStr(node->as.strLit.chars, node->as.strLit.length);
    case AST_NULL_LIT: {
        ConstValue v = constNone();
        v.kind = CONST_NULL;
        return v;
    }

    /* A `const` binding was folded at its declaration; its uses inline it. */
    case AST_IDENT: {
        Symbol *s = node->as.ident.symbol;
        if (s == NULL || !s->isConstFolded || s->constValue == NULL) return constNone();
        return jaiConstEval(c, s->constValue);
    }

    case AST_CAST:
        return jaiConstEval(c, node->as.cast.operand);

    case AST_UNARY: {
        ConstValue v = jaiConstEval(c, node->as.unary.operand);
        if (v.kind == CONST_NONE) return v;
        switch (node->as.unary.op) {
        case OPK_NEG:
            if (v.kind == CONST_INT) {
                if (v.as.i == INT64_MIN)
                    return constFail(c, E0900_CONST_OVERFLOW, node->span,
                                     "integer overflow in constant `%s`", "-");
                return constInt(-v.as.i);
            }
            if (v.kind == CONST_FLOAT) return constFloat(-v.as.f);
            return constNone();
        case OPK_POS:
            return constNumeric(&v) ? v : constNone();
        case OPK_NOT:
            return v.kind == CONST_BOOL ? constBool(!v.as.b) : constNone();
        case OPK_BNOT:
            return v.kind == CONST_INT ? constInt(~v.as.i) : constNone();
        default:
            return constNone();
        }
    }

    case AST_BINARY: {
        ConstValue a = jaiConstEval(c, node->as.binary.left);
        if (a.kind == CONST_NONE) return a;
        ConstValue b = jaiConstEval(c, node->as.binary.right);
        if (b.kind == CONST_NONE) return b;
        OpKind op = node->as.binary.op;

        if (a.kind == CONST_STR) return foldStrBinary(c, op, &a, &b);
        if (a.kind == CONST_BOOL && b.kind == CONST_BOOL) {
            if (op == OPK_EQ) return constBool(a.as.b == b.as.b);
            if (op == OPK_NE) return constBool(a.as.b != b.as.b);
            return constNone();
        }
        if (!constNumeric(&a) || !constNumeric(&b)) return constNone();
        if (a.kind == CONST_INT && b.kind == CONST_INT)
            return foldIntBinary(c, op, a.as.i, b.as.i, node->span);
        /* A mixed pair widens to float, which is what §2.5 says arithmetic
         * across the two numeric types produces and what comparison across
         * them already did. */
        return foldFloatBinary(c, op, constAsFloat(&a), constAsFloat(&b), node->span);
    }

    case AST_LOGICAL: {
        ConstValue a = jaiConstEval(c, node->as.binary.left);
        if (a.kind != CONST_BOOL) return constNone();
        if (node->as.binary.op == OPK_AND && !a.as.b) return constBool(false);
        if (node->as.binary.op == OPK_OR && a.as.b) return constBool(true);
        ConstValue b = jaiConstEval(c, node->as.binary.right);
        return b.kind == CONST_BOOL ? b : constNone();
    }

    case AST_TERNARY: {
        ConstValue cond = jaiConstEval(c, node->as.ternary.cond);
        if (cond.kind != CONST_BOOL) return constNone();
        return jaiConstEval(c, cond.as.b ? node->as.ternary.thenExpr
                                         : node->as.ternary.elseExpr);
    }

    case AST_COALESCE: {
        ConstValue a = jaiConstEval(c, node->as.coalesce.left);
        if (a.kind == CONST_NONE) return a;
        return a.kind == CONST_NULL ? jaiConstEval(c, node->as.coalesce.right) : a;
    }

    case AST_COMPARE_CHAIN: {
        if (node->as.chain.opCount < 1 || node->as.chain.operands == NULL) return constNone();
        for (int i = 0; i < node->as.chain.opCount; i++) {
            ConstValue a = jaiConstEval(c, node->as.chain.operands[i]);
            ConstValue b = jaiConstEval(c, node->as.chain.operands[i + 1]);
            if (a.kind == CONST_NONE || b.kind == CONST_NONE) return constNone();
            ConstValue r;
            if (a.kind == CONST_STR) r = foldStrBinary(c, node->as.chain.ops[i], &a, &b);
            else if (constNumeric(&a) && constNumeric(&b))
                r = (a.kind == CONST_INT && b.kind == CONST_INT)
                        ? foldIntBinary(c, node->as.chain.ops[i], a.as.i, b.as.i, node->span)
                        : foldFloatBinary(c, node->as.chain.ops[i], constAsFloat(&a),
                                          constAsFloat(&b), node->span);
            else return constNone();
            if (r.kind != CONST_BOOL) return constNone();
            if (!r.as.b) return constBool(false);
        }
        return constBool(true);
    }

    default:
        return constNone();
    }
}

void jaiConstReplace(AstContext *ast, AstNode *node, ConstValue v) {
    if (ast == NULL || node == NULL || v.kind == CONST_NONE) return;

    JaiSpan span = node->span;
    memset(&node->as, 0, sizeof node->as);
    node->span = span;

    switch (v.kind) {
    case CONST_INT:
        node->kind = AST_INT_LIT;
        node->as.intLit = v.as.i;
        node->type = gTypes.tInt;
        break;
    case CONST_FLOAT:
        node->kind = AST_FLOAT_LIT;
        node->as.floatLit = v.as.f;
        node->type = gTypes.tFloat;
        break;
    case CONST_BOOL:
        node->kind = AST_BOOL_LIT;
        node->as.boolLit = v.as.b;
        node->type = gTypes.tBool;
        break;
    case CONST_STR: {
        /* The characters may point into a scratch buffer, so take a copy that
         * lives as long as the AST does. */
        const char *chars = v.as.s.chars == NULL ? "" : v.as.s.chars;
        node->kind = AST_STR_LIT;
        node->as.strLit.chars = jaiArenaMemdup(&ast->arena, chars, v.as.s.length);
        node->as.strLit.length = v.as.s.length;
        node->type = gTypes.tStr;
        break;
    }
    case CONST_NULL:
        node->kind = AST_NULL_LIT;
        node->type = gTypes.tNull;
        break;
    case CONST_NONE:
        break;
    }
}

/* Fold an operator node in place once its operands have been checked. */
void jaiChkTryFold(Checker *c, AstNode *node) {
    if (!c->foldConstants) return;
    switch (node->kind) {
    case AST_UNARY: case AST_BINARY: case AST_LOGICAL:
    case AST_TERNARY: case AST_COMPARE_CHAIN: case AST_COALESCE:
        break;
    default:
        return;
    }
    ConstValue v = jaiConstEval(c, node);
    if (v.kind == CONST_NONE) return;
    JaiType *checked = node->type;
    jaiConstReplace(c->ast, node, v);
    /* Keep the type the checker derived: it may be a union or an alias that is
     * more informative than the literal's own type. */
    bool guard = false;
    if (checked != NULL && jaiTypeAssignable(node->type, checked, &guard) && !guard)
        node->type = checked;
}
