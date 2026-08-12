/* runtime.h — builtins, the import machinery, and the top-level driver. */
#ifndef JAI_RUNTIME_H
#define JAI_RUNTIME_H

#include "../vm/vm.h"

/* What the front end is asked to produce; moved here from
 * `src/codegen/codegen.h`. */
typedef struct {
    int  optLevel;        /* 0 = none, 2 = default, 3 = + inlining */
    bool debugInfo;       /* keep local names and full line tables */
    bool stripAsserts;    /* --release */
    bool emitTailCalls;
} CodegenOptions;

CodegenOptions jaiCodegenDefaults(void);

/* ------------------------------------------------------------------ */
/* Builtin registration                                                 */
/* ------------------------------------------------------------------ */

/* maxArity of -1 means variadic. */
void jaiDefineNative(const char *name, JaiNativeFn fn, int minArity, int maxArity);
void jaiDefineGlobal(const char *name, Value value);

/* Each raises TypeError and returns false on mismatch, so a native body reads
 * as a straight line of guards. */
bool jaiArgInt(Value v, int index, const char *fnName, int64_t *out);
bool jaiArgFloat(Value v, int index, const char *fnName, double *out);
bool jaiArgNumber(Value v, int index, const char *fnName, double *out);
bool jaiArgBool(Value v, int index, const char *fnName, bool *out);
bool jaiArgString(Value v, int index, const char *fnName, ObjString **out);
bool jaiArgList(Value v, int index, const char *fnName, ObjList **out);
bool jaiArgDict(Value v, int index, const char *fnName, ObjDict **out);
bool jaiArgCallable(Value v, int index, const char *fnName);

/* The entire native surface of Jaithon (spec Appendix C). */
void jaiRegisterCoreBuiltins(void);      /* print, len, range, type_of, repr, ... */
void jaiRegisterMathPrimitives(void);    /* __prim__.f64_*                      */
void jaiRegisterStringPrimitives(void);
void jaiRegisterListPrimitives(void);
void jaiRegisterDictPrimitives(void);
void jaiRegisterIOPrimitives(void);
void jaiRegisterOSPrimitives(void);
void jaiRegisterCanvasPrimitives(void);  /* __prim__.canvas_*                   */
void jaiRegisterCompressPrimitives(void); /* __prim__.deflate, crc32, adler32   */
void jaiRegisterTimePrimitives(void);
void jaiRegisterRandomPrimitives(void);
void jaiRegisterThreadPrimitives(void);
void jaiRegisterGCPrimitives(void);
void jaiRegisterReflectPrimitives(void); /* compile, eval, exec, ast           */
void jaiRegisterGuiPrimitives(void);     /* no-ops on non-macOS                 */
void jaiRegisterGpuPrimitives(void);
void jaiRegisterAllBuiltins(void);

/* Looked up by the VM when member access hits a non-instance receiver, e.g.
 * "abc".upper(), xs.map(f). */
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
/* `fromDir` enables relative imports (paths starting with '.'). */
bool       jaiResolveModulePath(const char *dottedName, const char *fromDir,
                                char *out, size_t outSize);
/* Same search, but silent: no diagnostic, no pending exception on a miss
 * (used by src/sema/modsig.c, which is asking rather than importing). */
bool       jaiResolveModulePathQuiet(const char *dottedName, const char *fromDir,
                                     char *out, size_t outSize);
ObjModule *jaiImportModule(const char *dottedName, const char *fromDir);

/* Seed consulted only inside the bootstrap window: outside it, a seeded image
 * would shadow source the tree has moved past. Getting it wrong costs time,
 * not correctness — under --gc-stress a cold-cache compile goes from 0.6s to
 * minutes. */
ObjModule *jaiImportFrontEndModule(const char *dottedName);

ObjFunction *jaiCompileSource(const char *source, size_t length,
                              const char *path, ObjModule *module,
                              const CodegenOptions *opts);

/* Basename with `.jai` removed, or `__main__`. Must agree with
 * `module_name_for` in lib/jaithon/compile/mod.jai — the name is pooled as a
 * constant, and disagreement breaks a cached module's own imports. */
void jaiModuleNameFor(const char *path, char *out, size_t outSize);

/* Runs `jaithon.compile` via the VM, so the C front end must have compiled it
 * first. NULL means gDiags has the reason — never a silent fallback to
 * --front=c. Callers own the flush. */
ObjFunction *jaiSelfHostedCompileInto(const char *source, size_t length,
                                      const char *path, ObjModule *module,
                                      uint64_t hash, int optLevel);

int jaiRunFile(const char *path, const JaiRunOptions *opts, int argc,
               char **argv);
int jaiCheckFile(const char *path, const JaiRunOptions *opts);
/* Frees the import graph jaiCheckFile builds for E0801 detection; more checks
 * just reconstruct it. */
void jaiImportGraphFree(void);

bool jaiLoadPrelude(void);

#endif /* JAI_RUNTIME_H */
