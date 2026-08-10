/* codegen.h — AST -> bytecode.
 *
 * There is exactly one code generator and exactly one execution path. Jaithon 2
 * had a token-stream tree-walker and a separate bytecode compiler that
 * disagreed with each other; that whole class of bug is gone.
 */
#ifndef JAI_CODEGEN_H
#define JAI_CODEGEN_H

#include "../lang/ast.h"
#include "../sema/check.h"
#include "../vm/object.h"

typedef struct Emitter Emitter;

typedef struct {
    int  optLevel;        /* 0 = none, 2 = default, 3 = + inlining */
    bool debugInfo;       /* keep local names and full line tables */
    bool stripAsserts;    /* --release */
    bool emitTailCalls;
} CodegenOptions;

CodegenOptions jaiCodegenDefaults(void);

/* Compile a resolved, checked program into the module body function.
 * Returns NULL and reports diagnostics on failure. */
ObjFunction *jaiCompileProgram(AstNode *program, ObjModule *module,
                               const CodegenOptions *opts);

/* Compile a single function declaration (used for lazily compiled bodies and
 * by the REPL). */
ObjFunction *jaiCompileFunction(AstNode *fnDecl, ObjModule *module,
                                const CodegenOptions *opts);

/* ------------------------------------------------------------------ */
/* Optimisation passes (spec/BYTECODE.md §8)                            */
/* ------------------------------------------------------------------ */

/* Each pass rewrites `fn->chunk` in place and returns the number of changes.
 * They are safe to run in any order but the pipeline below is what
 * jaiCompileProgram uses. */
int jaiOptDeadCode(ObjFunction *fn);
int jaiOptPeephole(ObjFunction *fn);
int jaiOptSuperinstructions(ObjFunction *fn);
int jaiOptCoalesceSlots(ObjFunction *fn);
int jaiOptHoistGlobals(ObjFunction *fn);
void jaiOptimize(ObjFunction *fn, const CodegenOptions *opts);

/* Verify that a chunk is well-formed: every jump lands on an instruction
 * boundary, the stack depth is consistent at every join point, and no opcode
 * reads past the end. Used by the test suite and after deserialisation. */

#endif /* JAI_CODEGEN_H */
