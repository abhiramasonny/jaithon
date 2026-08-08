/* runtime.h — builtins, the import machinery, and the top-level driver. */
#ifndef JAI_RUNTIME_H
#define JAI_RUNTIME_H

#include "../vm/vm.h"
#include "../codegen/codegen.h"

/* ------------------------------------------------------------------ */
/* Builtin registration                                                 */
/* ------------------------------------------------------------------ */

/* Register one native under the given name in the builtins module.
 * maxArity of -1 means variadic. */
void jaiDefineNative(const char *name, JaiNativeFn fn, int minArity, int maxArity);
void jaiDefineGlobal(const char *name, Value value);

/* Argument helpers for natives. Each raises TypeError and returns false on a
 * mismatch, so a native body reads as a straight line of guards. */
bool jaiArgInt(Value v, int index, const char *fnName, int64_t *out);
bool jaiArgFloat(Value v, int index, const char *fnName, double *out);
bool jaiArgNumber(Value v, int index, const char *fnName, double *out);
bool jaiArgBool(Value v, int index, const char *fnName, bool *out);
bool jaiArgString(Value v, int index, const char *fnName, ObjString **out);
bool jaiArgList(Value v, int index, const char *fnName, ObjList **out);
bool jaiArgDict(Value v, int index, const char *fnName, ObjDict **out);
bool jaiArgCallable(Value v, int index, const char *fnName);

/* Each of these installs one group of natives into the builtins module.
 * They are the entire native surface of Jaithon (spec Appendix C). */
void jaiRegisterCoreBuiltins(void);      /* print, len, range, type_of, repr, ... */
void jaiRegisterMathPrimitives(void);    /* __prim__.f64_*                      */
void jaiRegisterStringPrimitives(void);
void jaiRegisterListPrimitives(void);
void jaiRegisterDictPrimitives(void);
void jaiRegisterIOPrimitives(void);
void jaiRegisterOSPrimitives(void);
void jaiRegisterTimePrimitives(void);
void jaiRegisterRandomPrimitives(void);
void jaiRegisterThreadPrimitives(void);
void jaiRegisterGCPrimitives(void);
void jaiRegisterReflectPrimitives(void); /* compile, eval, exec, ast           */
void jaiRegisterGuiPrimitives(void);     /* no-ops on non-macOS                 */
void jaiRegisterGpuPrimitives(void);
void jaiRegisterAllBuiltins(void);

/* Built-in method tables for the primitive types: "abc".upper(), xs.map(f).
 * Looked up by the VM when a member access hits a non-instance receiver. */
bool jaiBuiltinMethod(Value receiver, ObjString *name, Value *out);

/* ------------------------------------------------------------------ */
/* Exception classes                                                    */
/* ------------------------------------------------------------------ */

void jaiRegisterErrorClasses(void);
Value jaiMakeException(ObjClass *klass, const char *message);

/* ------------------------------------------------------------------ */
/* Modules and imports                                                  */
/* ------------------------------------------------------------------ */

typedef struct {
    const char    *entryPath;
    CodegenOptions codegen;
    bool           useCache;
    bool           writeCache;
    bool           selfHosted;      /* --front=jai */
    bool           checkOnly;
    bool           verbose;
} JaiRunOptions;

JaiRunOptions jaiRunDefaults(void);

void       jaiModulePathInit(const char *execDir);
void       jaiModulePathAdd(const char *dir);
/* Resolve a dotted module name to a filesystem path. `fromDir` is the directory
 * of the importing file, used for relative imports (paths starting with '.'). */
bool       jaiResolveModulePath(const char *dottedName, const char *fromDir,
                                char *out, size_t outSize);
/* The same search with neither channel used: a miss leaves no diagnostic and no
 * pending exception. For callers who are asking rather than importing — the
 * type checker's signature loader (src/sema/modsig.c). */
bool       jaiResolveModulePathQuiet(const char *dottedName, const char *fromDir,
                                     char *out, size_t outSize);
/* Import (loading and executing if necessary). Returns NULL and sets the
 * pending exception on failure. */
ObjModule *jaiImportModule(const char *dottedName, const char *fromDir);

/* Compile a source string into a module body without running it. */
ObjFunction *jaiCompileSource(const char *source, size_t length,
                              const char *path, ObjModule *module,
                              const CodegenOptions *opts);

/* The module name a path carries: its basename with `.jai` removed, or
 * `__main__` when nothing is left. The self-hosted front end has its own copy
 * of this rule in `module_name_for` (lib/jaithon/compile/mod.jai) and the two
 * must agree: --bootstrap-verify compiles one file with both front ends and
 * the name is a constant in the record's pool. */
void jaiModuleNameFor(const char *path, char *out, size_t outSize);

/* Compile a source string with the *self-hosted* front end (lib/jaithon/compile)
 * and load the .jaic image it returns into `module`. This runs the VM:
 * `jaithon.compile` is imported and called, so the C front end has to have compiled
 * it first. `hash` is jaiSourceHash of the same text, which the container
 * records and the loader checks.
 *
 * NULL means the reason is already in gDiags, naming the stage that failed. It
 * never means "the C front end was used instead": --front=jai must not silently
 * become --front=c. Callers own the flush. */
ObjFunction *jaiSelfHostedCompileInto(const char *source, size_t length,
                                      const char *path, ObjModule *module,
                                      uint64_t hash, int optLevel);

/* Full pipeline for the CLI: read, compile (or load cache), run, call main().
 * Returns the process exit code. */
int jaiRunFile(const char *path, const JaiRunOptions *opts, int argc,
               char **argv);
int jaiCheckFile(const char *path, const JaiRunOptions *opts);
/* Release the import graph jaiCheckFile builds to detect E0801. Checking more
 * files reconstructs whatever it needs, so this is only a teardown hook. */
void jaiImportGraphFree(void);

/* Load the standard library prelude into the builtins module. */
bool jaiLoadPrelude(void);

#endif /* JAI_RUNTIME_H */
