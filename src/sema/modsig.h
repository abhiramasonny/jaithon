/* modsig.h — the signatures of imported modules, for the type checker.
 *
 * A Jaithon module is resolved at *run* time (spec §8): `from m import f` emits
 * an import instruction and binds a value, and nothing about `m` reaches the
 * front end. That left every call across a module boundary unchecked — arity,
 * keyword-argument names and parameter types all stopped at the import.
 *
 * This is the missing half. Given a dotted module name and the file that names
 * it, the loader finds the module's source on the same search path the runtime
 * uses, parses it once, and keeps the tree alive for the rest of the process so
 * that the checker can read a declaration's parameter list out of it. Only the
 * *signatures* are used: no body is resolved, no body is checked, and no
 * diagnostic from the parsed file escapes — whatever is wrong with a module is
 * that module's own compilation to report.
 *
 * Parsing rather than reading a `.jaic` is deliberate. A module may have no
 * cache entry, may have one written by a different compiler build, and when it
 * does have a warm one the run time skips the front end entirely — so the cache
 * is exactly the state in which no AST exists. One source of truth, one code
 * path, and a warm cache changes nothing about what is checked.
 */
#ifndef JAI_MODSIG_H
#define JAI_MODSIG_H

#include "../lang/ast.h"

typedef struct ModuleSig ModuleSig;

/* Load (or return the cached) signature of the module `dotted` names, resolved
 * relative to the file `fromFileId`, exactly as the runtime would resolve it.
 * NULL when the module cannot be found or does not parse; neither case is
 * reported here. */
ModuleSig *jaiModuleSigLoad(const char *dotted, int fromFileId);

/* The top-level declaration `name` denotes in `sig`: an AST_FN_DECL,
 * AST_CLASS_DECL, AST_TRAIT_DECL, AST_ENUM_DECL or AST_TYPE_DECL. A name the
 * module itself imported and re-exports is followed to the module that declares
 * it, and `*outOwner` (when given) receives that module — the annotations in
 * the declaration must be read in its namespace, not in the re-exporter's.
 * NULL when the module declares no such name. */
AstNode *jaiModuleSigFind(ModuleSig *sig, const char *name, ModuleSig **outOwner);

/* The module's own top-level class/trait/enum declarations, in source order.
 * The checker registers all of them, so that one imported declaration's
 * annotations can name a sibling the importer never asked for. */
int      jaiModuleSigTypeCount(const ModuleSig *sig);
AstNode *jaiModuleSigTypeAt(const ModuleSig *sig, int index);

/* The dotted name the module was first loaded under, e.g. "std.gui". */
const char *jaiModuleSigDotted(const ModuleSig *sig);

/* The signature parsed from source file `fileId`, or NULL when that file is not
 * one — in particular when it is the file being compiled. Every AST node
 * carries the file it was parsed from, so this is how a declaration reached
 * indirectly, through an imported class's member table, is traced back to the
 * module whose namespace its annotations must be read in. */
ModuleSig *jaiModuleSigForFile(int fileId);

/* "std.gui.Window": the name an imported type is interned under, so that two
 * modules declaring a `Node` are two types and neither collides with a `Node`
 * declared in the importing file. Allocated in the signature's own arena. */
const char *jaiModuleSigQualify(ModuleSig *sig, const char *name);

/* Release every parsed signature. Only a shutdown hook: the trees have to
 * outlive every compilation in the process, because interned type names point
 * into them. */
void jaiModuleSigFreeAll(void);

#endif /* JAI_MODSIG_H */
