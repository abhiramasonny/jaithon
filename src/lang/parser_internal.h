/* parser_internal.h — the interface between the parser's five source files.
 *
 * The grammar is mutually recursive — a declaration holds statements, a
 * statement holds expressions, an expression can hold an `if` or a `match`, a
 * pattern can hold a literal — so no split of parser.c can be a one-way
 * dependency. What can be one-way is the *shape* of the coupling, and this
 * header is it:
 *
 *   jaiP*      the plumbing every file needs — token access, spans, interning,
 *              diagnostics, node construction, recovery. Defined in parser.c.
 *   jaiParse*  one production each. Defined in the file named after its
 *              syntactic category, called from wherever the grammar says.
 *
 * Nothing else crosses a file boundary: every other helper is static in the
 * file that uses it. Not a public interface — parser.h is what the rest of the
 * tree calls.
 *
 *   parser.c        plumbing, recovery, and the public entry points
 *   parser_expr.c   the Pratt ladder (spec §4)
 *   parser_stmt.c   statements and control flow (spec §5)
 *   parser_decl.c   declarations (spec §6, §7)
 *   parser_type.c   type syntax (spec §2) and patterns (spec §5.3)
 */
#ifndef JAI_PARSER_INTERNAL_H
#define JAI_PARSER_INTERNAL_H

#include "parser.h"

/* --- scratch vectors ----------------------------------------------- */

/* Grown on the heap while a production runs, then copied into the AST arena
 * and freed. The arena has no free, so a list whose length is not known in
 * advance cannot be built in it directly. */
typedef JAI_VEC(AstNode *)     NodeVec;
typedef JAI_VEC(AstType *)     TypeVec;
typedef JAI_VEC(AstParam)      ParamVec;
typedef JAI_VEC(AstArg)        ArgVec;
typedef JAI_VEC(AstMatchArm)   ArmVec;
typedef JAI_VEC(AstCompClause) ClauseVec;
typedef JAI_VEC(AstCatch)      CatchVec;
typedef JAI_VEC(AstField)      FieldVec;
typedef JAI_VEC(AstVariant)    VariantVec;
typedef JAI_VEC(AstGeneric)    GenericVec;
typedef JAI_VEC(AstImportItem) ItemVec;
typedef JAI_VEC(const char *)  NameVec;
typedef JAI_VEC(OpKind)        OpVec;

void *jaiPVecTake(Parser *p, void *data, int count, int capacity,
                  size_t elemSize);

/* Copy a scratch vector into the arena and free it. The vector must not be
 * used afterwards. */
#define JAI_VEC_TAKE(p, v, type)                                               \
    ((type *)jaiPVecTake((p), (v)->data, (v)->count, (v)->capacity,            \
                         sizeof(type)))

/* Yield tracking: a function is a generator iff its own body contains `yield`,
 * so the counter is saved and restored around every nested function body. */
extern int jaiPYieldCount;

/* --- token access -------------------------------------------------- */

Token     *jaiPTokAt(Parser *p, int idx);
Token     *jaiPCur(Parser *p);
TokenKind  jaiPCurKind(Parser *p);
TokenKind  jaiPKindAt(Parser *p, int offset);
bool       jaiPCheck(Parser *p, TokenKind kind);
Token     *jaiPAdvance(Parser *p);
bool       jaiPMatch(Parser *p, TokenKind kind);
void       jaiPSkipNewlines(Parser *p);

JaiSpan jaiPSpanOf(Parser *p, const Token *t);
JaiSpan jaiPCurSpan(Parser *p);
JaiSpan jaiPSpanSince(Parser *p, int startIdx);

const char *jaiPRawText(Parser *p, const Token *t, size_t *outLen);
bool        jaiPTokenTextIs(Parser *p, const Token *t, const char *text);
bool        jaiPCheckIdentText(Parser *p, int offset, const char *text);
const char *jaiPInternText(Parser *p, const char *s, size_t len);
const char *jaiPInternToken(Parser *p, const Token *t);
const char *jaiPInternLabel(Parser *p, const Token *t);

/* --- diagnostics and recovery -------------------------------------- */

/* How a token reads in a message: a keyword or operator by its spelling,
 * anything else by its category. */
void jaiPDescribeToken(Parser *p, const Token *t, char *buf, size_t bufSize);

JaiDiag *jaiPErrorAt(Parser *p, JaiSpan span, JaiDiagCode code,
                     const char *fmt, ...);
JaiDiag *jaiPErrorAtCur(Parser *p, JaiDiagCode code, const char *fmt, ...);

/* Reported without entering panic mode, so the caller keeps parsing and can
 * hang a note or a fix-it off the returned diagnostic. */
JaiDiag *jaiPErrorRecoveredCode(Parser *p, JaiSpan span, JaiDiagCode code,
                                const char *fmt, ...);
JaiDiag *jaiPErrorRecovered(Parser *p, JaiSpan span, const char *fmt, ...);

/* Every parser warning goes through here rather than calling jaiDiagWarn, for
 * the same reason the errors above do: the REPL reads a line as a trial parse
 * before it reads it for real, and a warning left behind by the trial is
 * rendered twice. */
JaiDiag *jaiPWarnAt(Parser *p, JaiSpan span, JaiDiagCode code,
                    const char *fmt, ...);

bool        jaiPExpect(Parser *p, TokenKind kind, const char *what);
void        jaiPUnclosed(Parser *p, int openIdx, const char *closeText,
                         const char *what);
const char *jaiPExpectIdentName(Parser *p, const char *what);

/* Skip to the next plausible statement boundary and leave panic mode. */
void jaiPSynchronize(Parser *p);

/* Consume the newline that ends a statement, reporting E0101 if what follows
 * is something else. */
void jaiPEndStatement(Parser *p);

/* --- node construction --------------------------------------------- */

AstNode *jaiPNewNode(Parser *p, AstKind kind, JaiSpan span);

/* False once the recursion guard has fired; the caller must unwind. */
bool jaiPEnterRecursion(Parser *p);

/* --- productions --------------------------------------------------- */

AstType *jaiParseType(Parser *p);
/* Append `t` to `out`, splicing a tuple type in as its members. */
void     jaiParsePushTypeFlattened(TypeVec *out, AstType *t);

bool     jaiParseStartsExpression(Parser *p);
AstNode *jaiParseExpression(Parser *p);
AstNode *jaiParsePrimary(Parser *p);
/* `-<int literal>` folded before the range check, so INT64_MIN is writable. */
AstNode *jaiParseNegMagnitude(Parser *p);

AstNode *jaiParsePattern(Parser *p);
/* Reinterpret an already-parsed expression as the pattern it must have been. */
AstNode *jaiParseExprToPattern(Parser *p, AstNode *e);

AstNode *jaiParseStatement(Parser *p);
AstNode *jaiParseExpectBlock(Parser *p, const char *what);
AstNode *jaiParseIf(Parser *p, bool asExpr);
AstNode *jaiParseMatch(Parser *p, bool asExpr);

AstNode *jaiParseDeclaration(Parser *p);
/* Everything after the `fn` keyword: name, generics, parameters, body. */
AstNode *jaiParseFnRest(Parser *p, AstVisibility vis, bool isStatic,
                        int startIdx, bool named, bool bodyOptional);

#endif /* JAI_PARSER_INTERNAL_H */
