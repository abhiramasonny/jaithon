/* semadump.h — a span-keyed record of every decision the checker made.
 *
 * The self-hosted front end is being written against this one, and a bytecode
 * offset is a poor oracle for a type bug: `OP_TYPE_GUARD vs OP_CALL` does not
 * say which type was wrong, and a type computed correctly but not yet used by
 * any emission produces no signal at all. This writes the decisions
 * themselves, so two checkers can be diffed in the vocabulary the bug is in.
 *
 * Records are keyed on source span rather than on a node index. Two
 * independently written front ends do not allocate AST nodes in the same
 * order; they do parse the same text into nodes covering the same byte ranges,
 * so a span is the only identity both sides can agree on without sharing an
 * implementation detail.
 *
 * Inactive unless jaiSemaDumpBegin was given a path, which is what keeps a
 * normal build paying one predictable branch per recorded decision.
 */
#ifndef JAI_SEMADUMP_H
#define JAI_SEMADUMP_H

#include "../lang/ast.h"
#include "types.h"

#define JAI_SEMADUMP_VERSION 1

/* Where a dump will be written when one is started. Set once from the command
 * line; a NULL path disables everything below. Armed and begun are two steps
 * because the driver knows the destination and the front end knows which file
 * is the one the user named — imports run through the same checker and must
 * not each overwrite the dump. */
void jaiSemaDumpArm(const char *outPath);
bool jaiSemaDumpArmed(void);

/* Start collecting for `sourcePath`, which goes in the header. No-op when
 * nothing was armed. */
void jaiSemaDumpBegin(const char *sourcePath);
bool jaiSemaDumpActive(void);

/* The final type of every node that has one. Walks the tree, so it is called
 * once at the end of checking rather than at each of the twenty-odd places
 * that assign a type. */
void jaiSemaDumpTypes(AstNode *program);

/* A constant folded in place. `kind` is what the node was *before* it became a
 * literal, which is the interesting half: that `2 + 3` folded says more than
 * that a node is now `5`. */
void jaiSemaDumpFold(JaiSpan span, AstKind kind, const char *rendered);

/* A cast the checker inserted: a runtime guard at an any->T boundary, or an
 * int->float widening. */
void jaiSemaDumpCast(JaiSpan span, AstKind kind, const JaiType *from,
                     const JaiType *to, bool widen);

/* Sort, write, release. False means the file could not be written, which is a
 * failure of the run and not a warning: a dump that silently did not appear
 * turns a comparison into a pass. */
bool jaiSemaDumpEnd(void);

#endif /* JAI_SEMADUMP_H */
