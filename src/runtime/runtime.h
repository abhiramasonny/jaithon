/* runtime.h — builtins, the import machinery, and the top-level driver. */
#ifndef JAI_RUNTIME_H
#define JAI_RUNTIME_H

#include "vm/vm.h"

typedef struct {
    int  optLevel;        /* 0 = none, 2 = default, 3 = + inlining */
    bool debugInfo;
    bool stripAsserts;
    bool emitTailCalls;
} CodegenOptions;

CodegenOptions jaiCodegenDefaults(void);

/* ------------------------------------------------------------------ */
/* Builtin registration                                                 */
/* ------------------------------------------------------------------ */

/* maxArity of -1 means variadic */
void jaiDefineNative(const char *name, JaiNativeFn fn, int minArity, int maxArity);
void jaiDefineGlobal(const char *name, Value value);

bool jaiArgInt(Value v, int index, const char *fnName, int64_t *out);
bool jaiArgFloat(Value v, int index, const char *fnName, double *out);
bool jaiArgNumber(Value v, int index, const char *fnName, double *out);
bool jaiArgBool(Value v, int index, const char *fnName, bool *out);
bool jaiArgString(Value v, int index, const char *fnName, ObjString **out);
bool jaiArgList(Value v, int index, const char *fnName, ObjList **out);
bool jaiArgDict(Value v, int index, const char *fnName, ObjDict **out);
bool jaiArgCallable(Value v, int index, const char *fnName);

void jaiRegisterCoreBuiltins(void);
void jaiRegisterMathPrimitives(void);
void jaiRegisterStringPrimitives(void);
void jaiRegisterListPrimitives(void);
void jaiRegisterDictPrimitives(void);
void jaiRegisterIOPrimitives(void);
void jaiRegisterOSPrimitives(void);
void jaiRegisterProcessPrimitives(void);
void jaiRegisterCanvasPrimitives(void);
void jaiRegisterCompressPrimitives(void);
void jaiRegisterTimePrimitives(void);
void jaiRegisterRandomPrimitives(void);
void jaiRegisterThreadPrimitives(void);
void jaiRegisterGCPrimitives(void);
void jaiRegisterReflectPrimitives(void); /* compile, eval, exec, ast           */
void jaiRegisterGuiPrimitives(void);
void jaiRegisterGpuPrimitives(void);
void jaiRegisterAllBuiltins(void);

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
bool       jaiResolveModulePath(const char *dottedName, const char *fromDir,
                                char *out, size_t outSize);
bool       jaiResolveModulePathQuiet(const char *dottedName, const char *fromDir,
                                     char *out, size_t outSize);
ObjModule *jaiImportModule(const char *dottedName, const char *fromDir);

ObjModule *jaiImportFrontEndModule(const char *dottedName);

ObjFunction *jaiCompileSource(const char *source, size_t length,
                              const char *path, ObjModule *module,
                              const CodegenOptions *opts);

void jaiModuleNameFor(const char *path, char *out, size_t outSize);

ObjFunction *jaiSelfHostedCompileInto(const char *source, size_t length,
                                      const char *path, ObjModule *module,
                                      uint64_t hash, int optLevel);

int jaiRunFile(const char *path, const JaiRunOptions *opts, int argc,
               char **argv);
int jaiCheckFile(const char *path, const JaiRunOptions *opts);
void jaiImportGraphFree(void);

bool jaiLoadPrelude(void);

#endif /* JAI_RUNTIME_H */
