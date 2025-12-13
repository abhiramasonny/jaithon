#ifndef PARALLEL_H
#define PARALLEL_H

#include "runtime.h"
#include <pthread.h>
#include <dispatch/dispatch.h>
#include <stdatomic.h>
#include <stdbool.h>


#define PAR_MAX_THREADS 16
#define PAR_MIN_PARALLEL_WORK 500       
#define PAR_MIN_SIMD_WORK 8             
#define PAR_MIN_GPU_WORK 5000           
#define PAR_MAX_TRACKED_VARS 64         
#define PAR_MAX_SIDE_EFFECTS 32         


typedef enum {
    EXEC_SERIAL,        
    EXEC_SIMD,          
    EXEC_PARALLEL,      
    EXEC_GPU,           
    EXEC_HYBRID         
} ExecutionBackend;

typedef enum {
    PAR_MODE_AUTO,      
    PAR_MODE_SERIAL,    
    PAR_MODE_PARALLEL,  
    PAR_MODE_GPU        
} ParallelMode;


typedef enum {
    ACCESS_NONE,
    ACCESS_READ,
    ACCESS_WRITE,
    ACCESS_READ_WRITE
} AccessType;

typedef struct {
    char name[MAX_NAME_LEN];
    AccessType access;
    bool isLoopInvariant;   
    bool isReduction;       
    char reductionOp;       
    bool isArrayAccess;     
    bool indexDependsOnIter;
} VarAccess;


typedef enum {
    EFFECT_NONE,
    EFFECT_PRINT,           
    EFFECT_FILE_IO,         
    EFFECT_NETWORK,         
    EFFECT_GUI,             
    EFFECT_RANDOM,          
    EFFECT_GLOBAL_WRITE,    
    EFFECT_FUNCTION_CALL    
} SideEffectType;

typedef struct {
    SideEffectType type;
    char detail[MAX_NAME_LEN];
    bool isOrderDependent;  
} SideEffect;


typedef struct {
    
    VarAccess variables[PAR_MAX_TRACKED_VARS];
    int varCount;
    
    
    SideEffect sideEffects[PAR_MAX_SIDE_EFFECTS];
    int effectCount;
    
    
    char iteratorVar[MAX_NAME_LEN];
    bool hasIterator;
    bool iteratorModifiedInBody;
    int64_t estimatedIterations;
    
    
    bool hasReduction;
    char reductionVar[MAX_NAME_LEN];
    char reductionOp;
    
    
    bool hasDataDependencies;   
    bool hasControlDependencies;
    bool hasOrderDependentEffects;
    
    
    bool canParallelize;
    bool canVectorize;
    bool canUseGPU;
    ExecutionBackend recommendedBackend;
    
    
    double estimatedWorkPerIter;
    bool isMemoryBound;
    bool isComputeBound;
} AnalysisResult;


typedef struct {
    
    const char* source;
    int sourceLen;
    
    
    int64_t rangeStart;
    int64_t rangeEnd;
    int64_t step;
    char iteratorVar[MAX_NAME_LEN];
    
    
    double* partialResults;
    int numPartitions;
    char reductionOp;
    
    
    pthread_mutex_t mutex;
    atomic_int completedThreads;
    atomic_bool hasError;
    char errorMessage[256];
    
    
    Value* results;
    Value finalResult;
} ParallelExecContext;


typedef struct {
    ParallelMode mode;
    int maxThreads;
    bool enableGPU;
    bool enableSIMD;
    bool verboseAnalysis;   
    
    
    int minParallelWork;
    int minSIMDWork;
    int minGPUWork;
    
    
    int64_t totalLoopsAnalyzed;
    int64_t loopsParallelized;
    int64_t loopsVectorized;
    int64_t loopsGPUOffloaded;
} ParallelConfig;

extern ParallelConfig parallelConfig;


void parallelInit(void);
void parallelShutdown(void);


void parallelSetMode(ParallelMode mode);
void parallelSetMaxThreads(int n);
void parallelEnableGPU(bool enable);
void parallelEnableSIMD(bool enable);


AnalysisResult analyzeCode(const char* source, const char* iteratorVar);


AnalysisResult analyzeExpression(const char* expr);


bool isFunctionPure(const char* funcName);


bool areBlocksIndependent(const char* block1, const char* block2);


Value executeWithStrategy(const char* condSrc, const char* bodySrc, 
                         AnalysisResult* analysis);


Value executeRangeLoop(int64_t start, int64_t end, int64_t step,
                      const char* iteratorVar, const char* bodySrc);


Value executeWhileLoop(const char* condSrc, const char* bodySrc);


Value executeParallelStatements(const char** statements, int count);


Value executeSmartReduction(int64_t start, int64_t end,
                           const char* iteratorVar, const char* bodySrc,
                           const char* reductionVar, char reductionOp,
                           double initialValue);


void arrayAdd(double* result, const double* a, const double* b, int n);
void arraySub(double* result, const double* a, const double* b, int n);
void arrayMul(double* result, const double* a, const double* b, int n);
void arrayDiv(double* result, const double* a, const double* b, int n);
void arrayScale(double* result, const double* a, double scalar, int n);
double arraySum(const double* a, int n);
double arrayProduct(const double* a, int n);
double arrayMin(const double* a, int n);
double arrayMax(const double* a, int n);
double arrayDot(const double* a, const double* b, int n);


typedef double (*MapFunc)(double x);
void arrayMap(double* result, const double* a, int n, MapFunc f);


#ifdef __APPLE__
bool gpuIsAvailable(void);
bool gpuCanHandleWorkload(const AnalysisResult* analysis);
Value gpuExecuteReduction(const double* data, int n, char op);
Value gpuExecuteMap(const double* input, int n, const char* operation);
Value gpuExecuteLoop(int64_t start, int64_t end, const char* bodySrc,
                    const AnalysisResult* analysis);
#endif


typedef void (*ParallelBody)(int64_t index, void* context);
void parallelForRange(int64_t start, int64_t end, ParallelBody body, void* context);


typedef double (*ReduceFunc)(double a, double b);
double parallelReduce(const double* data, int n, double identity, ReduceFunc op);


int getAvailableCores(void);


int estimateOptimalThreads(int64_t workSize, double workPerItem);


bool simdIsAvailable(void);


void printAnalysisResult(const AnalysisResult* result);

#endif 
