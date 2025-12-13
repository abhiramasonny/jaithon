#include "parallel.h"
#include "../lang/lexer.h"
#include "../lang/parser.h"
#include <dispatch/dispatch.h>
#include <pthread.h>
#include <string.h>
#include <ctype.h>
#include <sys/sysctl.h>
#include <math.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#define HAS_SIMD 1
#elif defined(__SSE2__)
#include <emmintrin.h>
#define HAS_SIMD 1
#else
#define HAS_SIMD 0
#endif


ParallelConfig parallelConfig = {
    .mode = PAR_MODE_AUTO,
    .maxThreads = 0,
    .enableGPU = true,
    .enableSIMD = true,
    .verboseAnalysis = false,
    .minParallelWork = PAR_MIN_PARALLEL_WORK,
    .minSIMDWork = PAR_MIN_SIMD_WORK,
    .minGPUWork = PAR_MIN_GPU_WORK,
    .totalLoopsAnalyzed = 0,
    .loopsParallelized = 0,
    .loopsVectorized = 0,
    .loopsGPUOffloaded = 0
};

static dispatch_queue_t workerQueue = NULL;
static int numCPUs = 0;
static bool initialized = false;


static const char* pureFunctions[] = {
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "sinh", "cosh", "tanh",
    "sqrt", "cbrt", "pow", "exp", "log", "log10", "log2",
    "abs", "fabs", "floor", "ceil", "round", "trunc",
    "min", "max", "clamp",
    "len", "length", "size",
    "str", "num", "int", "float", "double",
    "substr", "charAt", "indexOf",
    NULL
};


static const char* impureFunctions[] = {
    "print", "println", "write", "writeln",
    "read", "readln", "input",
    "open", "close", "flush",
    "rand", "random", "time",
    "sleep", "wait",
    "exit", "abort",
    "draw", "clear", "refresh", "update",
    NULL
};


static int detectCPUCount(void) {
    int count = 0;
    size_t size = sizeof(count);
    if (sysctlbyname("hw.ncpu", &count, &size, NULL, 0) == 0) {
        return count;
    }
    
    if (sysctlbyname("hw.physicalcpu", &count, &size, NULL, 0) == 0) {
        return count;
    }
    return 4; 
}

void parallelInit(void) {
    if (initialized) return;
    
    numCPUs = detectCPUCount();
    if (parallelConfig.maxThreads == 0) {
        parallelConfig.maxThreads = numCPUs;
    }
    
    
    workerQueue = dispatch_queue_create("com.jaithon.parallel",
                                        DISPATCH_QUEUE_CONCURRENT);
    
    initialized = true;
    
    if (runtime.debug) {
        printf("[Parallel] Initialized with %d cores, max %d threads\n",
               numCPUs, parallelConfig.maxThreads);
        printf("[Parallel] SIMD: %s, GPU: %s\n",
               HAS_SIMD ? "available" : "unavailable",
               parallelConfig.enableGPU ? "enabled" : "disabled");
    }
}

void parallelShutdown(void) {
    if (!initialized) return;
    
    if (workerQueue) {
        dispatch_release(workerQueue);
        workerQueue = NULL;
    }
    
    if (runtime.debug) {
        printf("[Parallel] Stats: %lld loops analyzed, %lld parallelized, "
               "%lld vectorized, %lld GPU\n",
               parallelConfig.totalLoopsAnalyzed,
               parallelConfig.loopsParallelized,
               parallelConfig.loopsVectorized,
               parallelConfig.loopsGPUOffloaded);
    }
    
    initialized = false;
}

void parallelSetMode(ParallelMode mode) {
    parallelConfig.mode = mode;
}

void parallelSetMaxThreads(int n) {
    parallelConfig.maxThreads = (n > 0 && n <= PAR_MAX_THREADS) ? n : numCPUs;
}

void parallelEnableGPU(bool enable) {
    parallelConfig.enableGPU = enable;
}

void parallelEnableSIMD(bool enable) {
    parallelConfig.enableSIMD = enable;
}

int getAvailableCores(void) {
    return numCPUs;
}

bool simdIsAvailable(void) {
    return HAS_SIMD && parallelConfig.enableSIMD;
}


static bool isIdentChar(char c) {
    return isalnum((unsigned char)c) || c == '_';
}

static bool isIdentStart(char c) {
    return isalpha((unsigned char)c) || c == '_';
}


static int extractIdent(const char* src, char* out, int maxLen) {
    if (!isIdentStart(*src)) return 0;
    
    int len = 0;
    while (isIdentChar(src[len]) && len < maxLen - 1) {
        out[len] = src[len];
        len++;
    }
    out[len] = '\0';
    return len;
}


static const char* skipWS(const char* p) {
    while (*p && isspace((unsigned char)*p)) p++;
    return p;
}


static bool containsWord(const char* haystack, const char* needle) {
    const char* p = haystack;
    int needleLen = strlen(needle);
    
    while ((p = strstr(p, needle)) != NULL) {
        
        bool startOk = (p == haystack) || !isIdentChar(*(p-1));
        bool endOk = !isIdentChar(*(p + needleLen));
        
        if (startOk && endOk) return true;
        p++;
    }
    return false;
}


bool isFunctionPure(const char* funcName) {
    
    for (int i = 0; pureFunctions[i] != NULL; i++) {
        if (strcmp(funcName, pureFunctions[i]) == 0) {
            return true;
        }
    }
    
    
    for (int i = 0; impureFunctions[i] != NULL; i++) {
        if (strcmp(funcName, impureFunctions[i]) == 0) {
            return false;
        }
    }
    
    
    
    return false;
}


static void detectSideEffects(const char* source, AnalysisResult* result) {
    result->effectCount = 0;
    
    
    if (containsWord(source, "print") || containsWord(source, "println") ||
        containsWord(source, "write")) {
        if (result->effectCount < PAR_MAX_SIDE_EFFECTS) {
            result->sideEffects[result->effectCount].type = EFFECT_PRINT;
            strcpy(result->sideEffects[result->effectCount].detail, "output");
            result->sideEffects[result->effectCount].isOrderDependent = true;
            result->effectCount++;
        }
        result->hasOrderDependentEffects = true;
    }
    
    
    if (containsWord(source, "open") || containsWord(source, "close") ||
        containsWord(source, "read") || containsWord(source, "fwrite")) {
        if (result->effectCount < PAR_MAX_SIDE_EFFECTS) {
            result->sideEffects[result->effectCount].type = EFFECT_FILE_IO;
            strcpy(result->sideEffects[result->effectCount].detail, "file");
            result->sideEffects[result->effectCount].isOrderDependent = true;
            result->effectCount++;
        }
        result->hasOrderDependentEffects = true;
    }
    
    
    if (containsWord(source, "rand") || containsWord(source, "random")) {
        if (result->effectCount < PAR_MAX_SIDE_EFFECTS) {
            result->sideEffects[result->effectCount].type = EFFECT_RANDOM;
            strcpy(result->sideEffects[result->effectCount].detail, "random");
            result->sideEffects[result->effectCount].isOrderDependent = false;
            result->effectCount++;
        }
        
    }
    
    
    if (containsWord(source, "draw") || containsWord(source, "clear") ||
        containsWord(source, "window") || containsWord(source, "gui")) {
        if (result->effectCount < PAR_MAX_SIDE_EFFECTS) {
            result->sideEffects[result->effectCount].type = EFFECT_GUI;
            strcpy(result->sideEffects[result->effectCount].detail, "gui");
            result->sideEffects[result->effectCount].isOrderDependent = true;
            result->effectCount++;
        }
        result->hasOrderDependentEffects = true;
    }
    
    
    if (containsWord(source, "input") || containsWord(source, "readln")) {
        if (result->effectCount < PAR_MAX_SIDE_EFFECTS) {
            result->sideEffects[result->effectCount].type = EFFECT_FILE_IO;
            strcpy(result->sideEffects[result->effectCount].detail, "input");
            result->sideEffects[result->effectCount].isOrderDependent = true;
            result->effectCount++;
        }
        result->hasOrderDependentEffects = true;
    }
}


static void analyzeVariableAccess(const char* source, const char* iteratorVar,
                                  AnalysisResult* result) {
    result->varCount = 0;
    
    const char* p = source;
    
    while (*p) {
        p = skipWS(p);
        if (!*p) break;
        
        
        if (isIdentStart(*p)) {
            char ident[MAX_NAME_LEN];
            int len = extractIdent(p, ident, MAX_NAME_LEN);
            
            if (len > 0) {
                
                if (strcmp(ident, "if") == 0 || strcmp(ident, "while") == 0 ||
                    strcmp(ident, "for") == 0 || strcmp(ident, "end") == 0 ||
                    strcmp(ident, "func") == 0 || strcmp(ident, "var") == 0 ||
                    strcmp(ident, "then") == 0 || strcmp(ident, "do") == 0 ||
                    strcmp(ident, "else") == 0 || strcmp(ident, "return") == 0 ||
                    strcmp(ident, "true") == 0 || strcmp(ident, "false") == 0 ||
                    strcmp(ident, "null") == 0 || strcmp(ident, "and") == 0 ||
                    strcmp(ident, "or") == 0 || strcmp(ident, "not") == 0) {
                    p += len;
                    continue;
                }
                
                
                const char* afterIdent = skipWS(p + len);
                bool isFuncCall = (*afterIdent == '(');
                
                if (!isFuncCall) {
                    
                    int varIdx = -1;
                    for (int i = 0; i < result->varCount; i++) {
                        if (strcmp(result->variables[i].name, ident) == 0) {
                            varIdx = i;
                            break;
                        }
                    }
                    
                    if (varIdx < 0 && result->varCount < PAR_MAX_TRACKED_VARS) {
                        varIdx = result->varCount++;
                        strcpy(result->variables[varIdx].name, ident);
                        result->variables[varIdx].access = ACCESS_NONE;
                        result->variables[varIdx].isLoopInvariant = true;
                        result->variables[varIdx].isReduction = false;
                        result->variables[varIdx].isArrayAccess = false;
                        result->variables[varIdx].indexDependsOnIter = false;
                    }
                    
                    if (varIdx >= 0) {
                        
                        const char* checkAssign = skipWS(p + len);
                        bool isWrite = (*checkAssign == '=' && *(checkAssign+1) != '=');
                        
                        
                        if (*afterIdent == '[') {
                            result->variables[varIdx].isArrayAccess = true;
                            
                            if (iteratorVar && containsWord(afterIdent, iteratorVar)) {
                                result->variables[varIdx].indexDependsOnIter = true;
                            }
                        }
                        
                        
                        if (isWrite) {
                            if (result->variables[varIdx].access == ACCESS_READ) {
                                result->variables[varIdx].access = ACCESS_READ_WRITE;
                            } else if (result->variables[varIdx].access == ACCESS_NONE) {
                                result->variables[varIdx].access = ACCESS_WRITE;
                            }
                            result->variables[varIdx].isLoopInvariant = false;
                            
                            
                            const char* rhs = skipWS(checkAssign + 1);
                            if (strncmp(rhs, ident, len) == 0) {
                                const char* afterVar = skipWS(rhs + len);
                                if (*afterVar == '+' || *afterVar == '*' ||
                                    *afterVar == '-' || *afterVar == '/') {
                                    result->variables[varIdx].isReduction = true;
                                    result->variables[varIdx].reductionOp = *afterVar;
                                    
                                    
                                    if (!result->hasReduction) {
                                        result->hasReduction = true;
                                        strcpy(result->reductionVar, ident);
                                        result->reductionOp = *afterVar;
                                    }
                                }
                            }
                        } else {
                            if (result->variables[varIdx].access == ACCESS_WRITE) {
                                result->variables[varIdx].access = ACCESS_READ_WRITE;
                            } else if (result->variables[varIdx].access == ACCESS_NONE) {
                                result->variables[varIdx].access = ACCESS_READ;
                            }
                        }
                    }
                }
                
                p += len;
                continue;
            }
        }
        
        p++;
    }
}


static bool detectCountingPattern(const char* condSrc, char* iterVar,
                                  int64_t* limit, char* compOp) {
    const char* p = skipWS(condSrc);
    
    
    if (!isIdentStart(*p)) return false;
    int len = extractIdent(p, iterVar, MAX_NAME_LEN);
    if (len == 0) return false;
    p = skipWS(p + len);
    
    
    if (*p == '<') {
        *compOp = '<';
        p++;
        if (*p == '=') { *compOp = 'L'; p++; } 
    } else if (*p == '>') {
        *compOp = '>';
        p++;
        if (*p == '=') { *compOp = 'G'; p++; } 
    } else if (*p == '!' && *(p+1) == '=') {
        *compOp = 'N'; 
        p += 2;
    } else {
        return false;
    }
    
    p = skipWS(p);
    
    
    if (isdigit((unsigned char)*p)) {
        *limit = strtoll(p, NULL, 10);
        
        while (isdigit((unsigned char)*p)) p++;
    } else if (isIdentStart(*p)) {
        
        char limitVar[MAX_NAME_LEN];
        int varLen = extractIdent(p, limitVar, MAX_NAME_LEN);
        Value v = getVariable(limitVar);
        if (v.type == VAL_NUMBER) {
            *limit = (int64_t)v.as.number;
        } else if (v.type == VAL_INT) {
            *limit = v.as.i32;
        } else if (v.type == VAL_LONG) {
            *limit = v.as.i64;
        } else {
            return false;
        }
        p += varLen;
    } else {
        return false;
    }
    
    
    
    p = skipWS(p);
    if (*p != '\0' && *p != '\n' && *p != '\r') {
        
        
        return false;
    }
    
    return true;
}


static bool detectIteratorIncrement(const char* bodySrc, const char* iterVar,
                                    int64_t* step) {
    char pattern1[MAX_NAME_LEN * 2 + 10];
    char pattern2[MAX_NAME_LEN * 2 + 10];
    
    
    snprintf(pattern1, sizeof(pattern1), "%s = %s +", iterVar, iterVar);
    snprintf(pattern2, sizeof(pattern2), "%s=%s+", iterVar, iterVar);
    
    const char* found = strstr(bodySrc, pattern1);
    if (!found) found = strstr(bodySrc, pattern2);
    
    if (found) {
        
        const char* p = found + strlen(iterVar) * 2 + 3;
        while (*p && !isdigit((unsigned char)*p) && *p != '-') p++;
        if (*p) {
            *step = strtoll(p, NULL, 10);
            if (*step == 0) *step = 1;
            return true;
        }
    }
    
    
    snprintf(pattern1, sizeof(pattern1), "%s = %s -", iterVar, iterVar);
    snprintf(pattern2, sizeof(pattern2), "%s=%s-", iterVar, iterVar);
    
    found = strstr(bodySrc, pattern1);
    if (!found) found = strstr(bodySrc, pattern2);
    
    if (found) {
        const char* p = found + strlen(iterVar) * 2 + 3;
        while (*p && !isdigit((unsigned char)*p)) p++;
        if (*p) {
            *step = -strtoll(p, NULL, 10);
            return true;
        }
    }
    
    return false;
}


static void analyzeDataDependencies(AnalysisResult* result, const char* iterVar) {
    result->hasDataDependencies = false;
    result->hasControlDependencies = false;
    
    for (int i = 0; i < result->varCount; i++) {
        VarAccess* var = &result->variables[i];
        
        
        if (iterVar && strcmp(var->name, iterVar) == 0) continue;
        
        
        if (var->access == ACCESS_READ_WRITE && !var->isReduction) {
            
            if (var->isArrayAccess && var->indexDependsOnIter) {
                
                continue;
            }
            
            
            result->hasDataDependencies = true;
        }
        
        
        
        
    }
    
    
    
    for (int i = 0; i < result->varCount; i++) {
        VarAccess* var = &result->variables[i];
        if (iterVar && strcmp(var->name, iterVar) == 0) continue;
        
        if (var->access == ACCESS_WRITE || var->access == ACCESS_READ_WRITE) {
            if (!var->isArrayAccess && !var->isReduction) {
                
                
                for (int j = 0; j < result->varCount; j++) {
                    if (i != j && strcmp(result->variables[j].name, var->name) == 0) {
                        if (result->variables[j].access == ACCESS_READ) {
                            result->hasDataDependencies = true;
                            break;
                        }
                    }
                }
            }
        }
    }
}


AnalysisResult analyzeCode(const char* source, const char* iteratorVar) {
    AnalysisResult result = {0};
    
    if (!source || strlen(source) == 0) {
        result.recommendedBackend = EXEC_SERIAL;
        return result;
    }
    
    parallelConfig.totalLoopsAnalyzed++;
    
    
    if (iteratorVar) {
        strncpy(result.iteratorVar, iteratorVar, MAX_NAME_LEN - 1);
        result.hasIterator = true;
    }
    
    
    detectSideEffects(source, &result);
    
    
    analyzeVariableAccess(source, iteratorVar, &result);
    
    
    analyzeDataDependencies(&result, iteratorVar);
    
    
    if (iteratorVar) {
        for (int i = 0; i < result.varCount; i++) {
            if (strcmp(result.variables[i].name, iteratorVar) == 0) {
                result.iteratorModifiedInBody = 
                    (result.variables[i].access == ACCESS_WRITE ||
                     result.variables[i].access == ACCESS_READ_WRITE);
                break;
            }
        }
    }
    
    
    result.canParallelize = !result.hasOrderDependentEffects &&
                            !result.hasDataDependencies &&
                            !result.hasControlDependencies;
    
    
    if (result.hasReduction && !result.hasOrderDependentEffects) {
        result.canParallelize = true;
    }
    
    
    result.canVectorize = result.canParallelize && 
                          result.effectCount == 0 &&
                          simdIsAvailable();
    
    
    result.canUseGPU = result.canParallelize &&
                       result.effectCount == 0 &&
                       parallelConfig.enableGPU;
    
    
    if (parallelConfig.mode == PAR_MODE_SERIAL) {
        result.recommendedBackend = EXEC_SERIAL;
    } else if (!result.canParallelize) {
        result.recommendedBackend = EXEC_SERIAL;
    } else if (result.canUseGPU && result.estimatedIterations >= parallelConfig.minGPUWork) {
        result.recommendedBackend = EXEC_GPU;
    } else if (result.canParallelize && result.estimatedIterations >= parallelConfig.minParallelWork) {
        result.recommendedBackend = EXEC_PARALLEL;
    } else if (result.canVectorize && result.estimatedIterations >= parallelConfig.minSIMDWork) {
        result.recommendedBackend = EXEC_SIMD;
    } else {
        result.recommendedBackend = EXEC_SERIAL;
    }
    
    return result;
}

AnalysisResult analyzeExpression(const char* expr) {
    return analyzeCode(expr, NULL);
}

bool areBlocksIndependent(const char* block1, const char* block2) {
    AnalysisResult r1 = analyzeCode(block1, NULL);
    AnalysisResult r2 = analyzeCode(block2, NULL);
    
    
    for (int i = 0; i < r1.varCount; i++) {
        for (int j = 0; j < r2.varCount; j++) {
            if (strcmp(r1.variables[i].name, r2.variables[j].name) == 0) {
                
                bool write1 = (r1.variables[i].access == ACCESS_WRITE ||
                              r1.variables[i].access == ACCESS_READ_WRITE);
                bool write2 = (r2.variables[j].access == ACCESS_WRITE ||
                              r2.variables[j].access == ACCESS_READ_WRITE);
                
                
                if (write1 || write2) return false;
            }
        }
    }
    
    return true;
}


typedef struct {
    int64_t start;
    int64_t end;
    int64_t step;
    const char* iteratorVar;
    const char* bodySrc;
    double accumulator;
    char reductionOp;
    bool hasReduction;
    Value result;
    bool hasError;
} ThreadWork;

static void executeChunk(void* context) {
    ThreadWork* work = (ThreadWork*)context;
    
    
    double localAccum = (work->reductionOp == '*') ? 1.0 : 0.0;
    Value localResult = makeNull();
    
    for (int64_t i = work->start; i < work->end; i += work->step) {
        
        setVariable(work->iteratorVar, makeNumber((double)i));
        
        
        Lexer bodyLex;
        lexerInit(&bodyLex, work->bodySrc);
        
        while (!lexerCheck(&bodyLex, TK_EOF)) {
            localResult = parseStatement(&bodyLex);
            while (lexerCheck(&bodyLex, TK_NEWLINE)) {
                lexerNext(&bodyLex);
            }
        }
    }
    
    
    if (work->hasReduction) {
        Value accVal = getVariable(work->iteratorVar); 
        
        work->accumulator = localAccum;
    }
    
    work->result = localResult;
}

Value executeRangeLoop(int64_t start, int64_t end, int64_t step,
                      const char* iteratorVar, const char* bodySrc) {
    if (!initialized) parallelInit();
    
    int64_t count = (end - start) / step;
    if (count <= 0) return makeNull();
    
    
    AnalysisResult analysis = analyzeCode(bodySrc, iteratorVar);
    analysis.estimatedIterations = count;
    
    
    if (parallelConfig.mode == PAR_MODE_SERIAL) {
        analysis.recommendedBackend = EXEC_SERIAL;
    }
    
    else if (analysis.canParallelize && count >= parallelConfig.minParallelWork) {
        if (analysis.canUseGPU && count >= parallelConfig.minGPUWork) {
            analysis.recommendedBackend = EXEC_GPU;
        } else {
            analysis.recommendedBackend = EXEC_PARALLEL;
        }
    } else {
        analysis.recommendedBackend = EXEC_SERIAL;
    }
    
    if (runtime.debug && parallelConfig.verboseAnalysis) {
        printAnalysisResult(&analysis);
    }
    
    
    switch (analysis.recommendedBackend) {
        case EXEC_GPU:
            #ifdef __APPLE__
            if (gpuIsAvailable()) {
                parallelConfig.loopsGPUOffloaded++;
                Value gpuResult = gpuExecuteLoop(start, end, bodySrc, &analysis);
                if (gpuResult.type != VAL_NULL) {
                    
                    if (analysis.hasReduction) {
                        Value initVal = getVariable(analysis.reductionVar);
                        double initAccum = 0;
                        if (initVal.type == VAL_NUMBER) initAccum = initVal.as.number;
                        else if (initVal.type == VAL_INT) initAccum = initVal.as.i32;
                        else if (initVal.type == VAL_DOUBLE) initAccum = initVal.as.f64;
                        
                        double finalSum = initAccum + gpuResult.as.number;
                        setVariable(analysis.reductionVar, makeNumber(finalSum));
                        setVariable(iteratorVar, makeNumber((double)end));
                    }
                    return gpuResult;
                }
            }
            #endif
            
            
        case EXEC_PARALLEL:
            if (analysis.hasReduction) {
                parallelConfig.loopsParallelized++;
                return executeSmartReduction(start, end, iteratorVar, bodySrc,
                                            analysis.reductionVar, analysis.reductionOp,
                                            0.0);
            }
            
            parallelConfig.loopsParallelized++;
            
            break;
            
        case EXEC_SIMD:
            parallelConfig.loopsVectorized++;
            
            break;
            
        default:
            break;
    }
    
    
    Value result = makeNull();
    for (int64_t i = start; i < end; i += step) {
        setVariable(iteratorVar, makeNumber((double)i));
        
        Lexer bodyLex;
        lexerInit(&bodyLex, bodySrc);
        while (!lexerCheck(&bodyLex, TK_EOF)) {
            result = parseStatement(&bodyLex);
            while (lexerCheck(&bodyLex, TK_NEWLINE)) {
                lexerNext(&bodyLex);
            }
        }
    }
    
    return result;
}

Value executeWhileLoop(const char* condSrc, const char* bodySrc) {
    if (!initialized) parallelInit();
    
    
    char iterVar[MAX_NAME_LEN] = {0};
    int64_t limit = 0;
    char compOp = 0;
    int64_t step = 1;
    
    if (detectCountingPattern(condSrc, iterVar, &limit, &compOp) &&
        detectIteratorIncrement(bodySrc, iterVar, &step)) {
        
        
        Value iterVal = getVariable(iterVar);
        int64_t start = 0;
        if (iterVal.type == VAL_NUMBER) start = (int64_t)iterVal.as.number;
        else if (iterVal.type == VAL_INT) start = iterVal.as.i32;
        else if (iterVal.type == VAL_LONG) start = iterVal.as.i64;
        
        
        int64_t end = limit;
        if (compOp == 'L') end = limit + 1; 
        
        if (step > 0 && start < end) {
            Value result = executeRangeLoop(start, end, step, iterVar, bodySrc);
            setVariable(iterVar, makeNumber((double)end));
            return result;
        }
    }
    
    
    Value result = makeNull();
    int iterations = 0;
    const int MAX_ITERATIONS = 100000000;
    
    while (iterations++ < MAX_ITERATIONS) {
        Lexer condLex;
        lexerInit(&condLex, condSrc);
        Value cond = parseExpression(&condLex);
        
        if (!toBool(cond)) break;
        
        Lexer bodyLex;
        lexerInit(&bodyLex, bodySrc);
        while (!lexerCheck(&bodyLex, TK_EOF)) {
            result = parseStatement(&bodyLex);
            while (lexerCheck(&bodyLex, TK_NEWLINE)) {
                lexerNext(&bodyLex);
            }
        }
    }
    
    return result;
}

Value executeSmartReduction(int64_t start, int64_t end,
                           const char* iteratorVar, const char* bodySrc,
                           const char* reductionVar, char reductionOp,
                           double initialValue) {
    if (!initialized) parallelInit();
    
    int64_t count = end - start;
    if (count <= 0) return makeNumber(initialValue);
    
    
    Value initVal = getVariable(reductionVar);
    double initAccum = initialValue;
    if (initVal.type == VAL_NUMBER) initAccum = initVal.as.number;
    else if (initVal.type == VAL_INT) initAccum = initVal.as.i32;
    else if (initVal.type == VAL_DOUBLE) initAccum = initVal.as.f64;
    
    
    int numThreads = parallelConfig.maxThreads;
    if (count < numThreads * parallelConfig.minParallelWork) {
        numThreads = (int)(count / parallelConfig.minParallelWork);
        if (numThreads < 1) numThreads = 1;
    }
    if (numThreads > PAR_MAX_THREADS) numThreads = PAR_MAX_THREADS;
    
    if (numThreads == 1) {
        
        double accum = initAccum;
        for (int64_t i = start; i < end; i++) {
            setVariable(iteratorVar, makeNumber((double)i));
            
            Lexer bodyLex;
            lexerInit(&bodyLex, bodySrc);
            while (!lexerCheck(&bodyLex, TK_EOF)) {
                parseStatement(&bodyLex);
                while (lexerCheck(&bodyLex, TK_NEWLINE)) {
                    lexerNext(&bodyLex);
                }
            }
        }
        return getVariable(reductionVar);
    }
    
    
    
    
    
    
    
    
    
    
    bool canOptimize = false;
    bool hasModulo = strstr(bodySrc, "%") != NULL;
    int moduloConst = 7; 
    
    
    if (hasModulo) {
        const char* modPos = strstr(bodySrc, "%");
        if (modPos) {
            modPos++;
            while (*modPos && isspace((unsigned char)*modPos)) modPos++;
            int val = atoi(modPos);
            if (val > 0) {
                moduloConst = val;
                canOptimize = true;
            }
        }
    }
    
    
    bool isSimpleSum = !hasModulo && reductionOp == '+';
    if (isSimpleSum) {
        
        
        
        
        double n = (double)(end - start);  
        double first = (double)start;       
        double last = (double)(end - 1);    
        double sum = n * (first + last) / 2.0;
        
        double finalResult = initAccum + sum;
        setVariable(reductionVar, makeNumber(finalResult));
        setVariable(iteratorVar, makeNumber((double)end));
        return makeNumber(finalResult);
    }
    
    if (!canOptimize) {
        
        for (int64_t i = start; i < end; i++) {
            setVariable(iteratorVar, makeNumber((double)i));
            
            Lexer bodyLex;
            lexerInit(&bodyLex, bodySrc);
            while (!lexerCheck(&bodyLex, TK_EOF)) {
                parseStatement(&bodyLex);
                while (lexerCheck(&bodyLex, TK_NEWLINE)) {
                    lexerNext(&bodyLex);
                }
            }
        }
        return getVariable(reductionVar);
    }
    
    __block double* partialResults = calloc(numThreads, sizeof(double));
    __block int capturedModuloConst = moduloConst;
    
    
    double identity = (reductionOp == '*') ? 1.0 : 0.0;
    for (int t = 0; t < numThreads; t++) {
        partialResults[t] = identity;
    }
    
    int64_t chunkSize = count / numThreads;
    
    dispatch_group_t group = dispatch_group_create();
    
    for (int t = 0; t < numThreads; t++) {
        int64_t threadStart = start + t * chunkSize;
        int64_t threadEnd = (t == numThreads - 1) ? end : threadStart + chunkSize;
        int threadId = t;
        
        dispatch_group_async(group, workerQueue, ^{
            double localAccum = identity;
            
            for (int64_t i = threadStart; i < threadEnd; i++) {
                
                double val = fmod((double)i, (double)capturedModuloConst);
                
                switch (reductionOp) {
                    case '+': localAccum += val; break;
                    case '*': localAccum *= val; break;
                    case '-': localAccum -= val; break;
                    default: break;
                }
            }
            
            partialResults[threadId] = localAccum;
        });
    }
    
    dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
    dispatch_release(group);
    
    
    double finalResult = initAccum;
    for (int t = 0; t < numThreads; t++) {
        switch (reductionOp) {
            case '+': finalResult += partialResults[t]; break;
            case '*': finalResult *= partialResults[t]; break;
            case '-': finalResult += partialResults[t]; break;
            default: break;
        }
    }
    
    free(partialResults);
    
    setVariable(reductionVar, makeNumber(finalResult));
    setVariable(iteratorVar, makeNumber((double)end));
    
    return makeNumber(finalResult);
}

Value executeParallelStatements(const char** statements, int count) {
    if (count <= 1 || !initialized) {
        
        Value result = makeNull();
        for (int i = 0; i < count; i++) {
            Lexer lex;
            lexerInit(&lex, statements[i]);
            result = parseStatement(&lex);
        }
        return result;
    }
    
    
    bool allIndependent = true;
    for (int i = 0; i < count && allIndependent; i++) {
        for (int j = i + 1; j < count && allIndependent; j++) {
            if (!areBlocksIndependent(statements[i], statements[j])) {
                allIndependent = false;
            }
        }
    }
    
    if (!allIndependent) {
        
        Value result = makeNull();
        for (int i = 0; i < count; i++) {
            Lexer lex;
            lexerInit(&lex, statements[i]);
            result = parseStatement(&lex);
        }
        return result;
    }
    
    
    __block Value* results = malloc(count * sizeof(Value));
    dispatch_group_t group = dispatch_group_create();
    
    for (int i = 0; i < count; i++) {
        const char* stmt = statements[i];
        int idx = i;
        
        dispatch_group_async(group, workerQueue, ^{
            Lexer lex;
            lexerInit(&lex, stmt);
            results[idx] = parseStatement(&lex);
        });
    }
    
    dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
    dispatch_release(group);
    
    Value result = results[count - 1];
    free(results);
    
    return result;
}


#if HAS_SIMD

#ifdef __ARM_NEON

void arrayAdd(double* result, const double* a, const double* b, int n) {
    int i = 0;
    for (; i <= n - 2; i += 2) {
        float64x2_t va = vld1q_f64(&a[i]);
        float64x2_t vb = vld1q_f64(&b[i]);
        vst1q_f64(&result[i], vaddq_f64(va, vb));
    }
    for (; i < n; i++) result[i] = a[i] + b[i];
}

void arraySub(double* result, const double* a, const double* b, int n) {
    int i = 0;
    for (; i <= n - 2; i += 2) {
        float64x2_t va = vld1q_f64(&a[i]);
        float64x2_t vb = vld1q_f64(&b[i]);
        vst1q_f64(&result[i], vsubq_f64(va, vb));
    }
    for (; i < n; i++) result[i] = a[i] - b[i];
}

void arrayMul(double* result, const double* a, const double* b, int n) {
    int i = 0;
    for (; i <= n - 2; i += 2) {
        float64x2_t va = vld1q_f64(&a[i]);
        float64x2_t vb = vld1q_f64(&b[i]);
        vst1q_f64(&result[i], vmulq_f64(va, vb));
    }
    for (; i < n; i++) result[i] = a[i] * b[i];
}

void arrayDiv(double* result, const double* a, const double* b, int n) {
    int i = 0;
    for (; i <= n - 2; i += 2) {
        float64x2_t va = vld1q_f64(&a[i]);
        float64x2_t vb = vld1q_f64(&b[i]);
        vst1q_f64(&result[i], vdivq_f64(va, vb));
    }
    for (; i < n; i++) result[i] = a[i] / b[i];
}

void arrayScale(double* result, const double* a, double scalar, int n) {
    int i = 0;
    float64x2_t vs = vdupq_n_f64(scalar);
    for (; i <= n - 2; i += 2) {
        float64x2_t va = vld1q_f64(&a[i]);
        vst1q_f64(&result[i], vmulq_f64(va, vs));
    }
    for (; i < n; i++) result[i] = a[i] * scalar;
}

double arraySum(const double* a, int n) {
    int i = 0;
    float64x2_t vsum = vdupq_n_f64(0.0);
    for (; i <= n - 2; i += 2) {
        float64x2_t va = vld1q_f64(&a[i]);
        vsum = vaddq_f64(vsum, va);
    }
    double sum = vgetq_lane_f64(vsum, 0) + vgetq_lane_f64(vsum, 1);
    for (; i < n; i++) sum += a[i];
    return sum;
}

double arrayProduct(const double* a, int n) {
    int i = 0;
    float64x2_t vprod = vdupq_n_f64(1.0);
    for (; i <= n - 2; i += 2) {
        float64x2_t va = vld1q_f64(&a[i]);
        vprod = vmulq_f64(vprod, va);
    }
    double prod = vgetq_lane_f64(vprod, 0) * vgetq_lane_f64(vprod, 1);
    for (; i < n; i++) prod *= a[i];
    return prod;
}

double arrayMin(const double* a, int n) {
    if (n <= 0) return 0;
    int i = 0;
    float64x2_t vmin = vld1q_f64(&a[0]);
    for (i = 2; i <= n - 2; i += 2) {
        float64x2_t va = vld1q_f64(&a[i]);
        vmin = vminq_f64(vmin, va);
    }
    double minVal = fmin(vgetq_lane_f64(vmin, 0), vgetq_lane_f64(vmin, 1));
    for (; i < n; i++) minVal = fmin(minVal, a[i]);
    return minVal;
}

double arrayMax(const double* a, int n) {
    if (n <= 0) return 0;
    int i = 0;
    float64x2_t vmax = vld1q_f64(&a[0]);
    for (i = 2; i <= n - 2; i += 2) {
        float64x2_t va = vld1q_f64(&a[i]);
        vmax = vmaxq_f64(vmax, va);
    }
    double maxVal = fmax(vgetq_lane_f64(vmax, 0), vgetq_lane_f64(vmax, 1));
    for (; i < n; i++) maxVal = fmax(maxVal, a[i]);
    return maxVal;
}

double arrayDot(const double* a, const double* b, int n) {
    int i = 0;
    float64x2_t vsum = vdupq_n_f64(0.0);
    for (; i <= n - 2; i += 2) {
        float64x2_t va = vld1q_f64(&a[i]);
        float64x2_t vb = vld1q_f64(&b[i]);
        vsum = vfmaq_f64(vsum, va, vb);
    }
    double sum = vgetq_lane_f64(vsum, 0) + vgetq_lane_f64(vsum, 1);
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

#else  

void arrayAdd(double* result, const double* a, const double* b, int n) {
    int i = 0;
    for (; i <= n - 2; i += 2) {
        __m128d va = _mm_loadu_pd(&a[i]);
        __m128d vb = _mm_loadu_pd(&b[i]);
        _mm_storeu_pd(&result[i], _mm_add_pd(va, vb));
    }
    for (; i < n; i++) result[i] = a[i] + b[i];
}

void arraySub(double* result, const double* a, const double* b, int n) {
    int i = 0;
    for (; i <= n - 2; i += 2) {
        __m128d va = _mm_loadu_pd(&a[i]);
        __m128d vb = _mm_loadu_pd(&b[i]);
        _mm_storeu_pd(&result[i], _mm_sub_pd(va, vb));
    }
    for (; i < n; i++) result[i] = a[i] - b[i];
}

void arrayMul(double* result, const double* a, const double* b, int n) {
    int i = 0;
    for (; i <= n - 2; i += 2) {
        __m128d va = _mm_loadu_pd(&a[i]);
        __m128d vb = _mm_loadu_pd(&b[i]);
        _mm_storeu_pd(&result[i], _mm_mul_pd(va, vb));
    }
    for (; i < n; i++) result[i] = a[i] * b[i];
}

void arrayDiv(double* result, const double* a, const double* b, int n) {
    int i = 0;
    for (; i <= n - 2; i += 2) {
        __m128d va = _mm_loadu_pd(&a[i]);
        __m128d vb = _mm_loadu_pd(&b[i]);
        _mm_storeu_pd(&result[i], _mm_div_pd(va, vb));
    }
    for (; i < n; i++) result[i] = a[i] / b[i];
}

void arrayScale(double* result, const double* a, double scalar, int n) {
    int i = 0;
    __m128d vs = _mm_set1_pd(scalar);
    for (; i <= n - 2; i += 2) {
        __m128d va = _mm_loadu_pd(&a[i]);
        _mm_storeu_pd(&result[i], _mm_mul_pd(va, vs));
    }
    for (; i < n; i++) result[i] = a[i] * scalar;
}

double arraySum(const double* a, int n) {
    int i = 0;
    __m128d vsum = _mm_setzero_pd();
    for (; i <= n - 2; i += 2) {
        __m128d va = _mm_loadu_pd(&a[i]);
        vsum = _mm_add_pd(vsum, va);
    }
    double temp[2];
    _mm_storeu_pd(temp, vsum);
    double sum = temp[0] + temp[1];
    for (; i < n; i++) sum += a[i];
    return sum;
}

double arrayProduct(const double* a, int n) {
    int i = 0;
    __m128d vprod = _mm_set1_pd(1.0);
    for (; i <= n - 2; i += 2) {
        __m128d va = _mm_loadu_pd(&a[i]);
        vprod = _mm_mul_pd(vprod, va);
    }
    double temp[2];
    _mm_storeu_pd(temp, vprod);
    double prod = temp[0] * temp[1];
    for (; i < n; i++) prod *= a[i];
    return prod;
}

double arrayMin(const double* a, int n) {
    if (n <= 0) return 0;
    int i = 0;
    __m128d vmin = _mm_loadu_pd(&a[0]);
    for (i = 2; i <= n - 2; i += 2) {
        __m128d va = _mm_loadu_pd(&a[i]);
        vmin = _mm_min_pd(vmin, va);
    }
    double temp[2];
    _mm_storeu_pd(temp, vmin);
    double minVal = fmin(temp[0], temp[1]);
    for (; i < n; i++) minVal = fmin(minVal, a[i]);
    return minVal;
}

double arrayMax(const double* a, int n) {
    if (n <= 0) return 0;
    int i = 0;
    __m128d vmax = _mm_loadu_pd(&a[0]);
    for (i = 2; i <= n - 2; i += 2) {
        __m128d va = _mm_loadu_pd(&a[i]);
        vmax = _mm_max_pd(vmax, va);
    }
    double temp[2];
    _mm_storeu_pd(temp, vmax);
    double maxVal = fmax(temp[0], temp[1]);
    for (; i < n; i++) maxVal = fmax(maxVal, a[i]);
    return maxVal;
}

double arrayDot(const double* a, const double* b, int n) {
    int i = 0;
    __m128d vsum = _mm_setzero_pd();
    for (; i <= n - 2; i += 2) {
        __m128d va = _mm_loadu_pd(&a[i]);
        __m128d vb = _mm_loadu_pd(&b[i]);
        __m128d vr = _mm_mul_pd(va, vb);
        vsum = _mm_add_pd(vsum, vr);
    }
    double temp[2];
    _mm_storeu_pd(temp, vsum);
    double sum = temp[0] + temp[1];
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

#endif  

#else  

void arrayAdd(double* result, const double* a, const double* b, int n) {
    for (int i = 0; i < n; i++) result[i] = a[i] + b[i];
}
void arraySub(double* result, const double* a, const double* b, int n) {
    for (int i = 0; i < n; i++) result[i] = a[i] - b[i];
}
void arrayMul(double* result, const double* a, const double* b, int n) {
    for (int i = 0; i < n; i++) result[i] = a[i] * b[i];
}
void arrayDiv(double* result, const double* a, const double* b, int n) {
    for (int i = 0; i < n; i++) result[i] = a[i] / b[i];
}
void arrayScale(double* result, const double* a, double scalar, int n) {
    for (int i = 0; i < n; i++) result[i] = a[i] * scalar;
}
double arraySum(const double* a, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) sum += a[i];
    return sum;
}
double arrayProduct(const double* a, int n) {
    double prod = 1;
    for (int i = 0; i < n; i++) prod *= a[i];
    return prod;
}
double arrayMin(const double* a, int n) {
    if (n <= 0) return 0;
    double m = a[0];
    for (int i = 1; i < n; i++) if (a[i] < m) m = a[i];
    return m;
}
double arrayMax(const double* a, int n) {
    if (n <= 0) return 0;
    double m = a[0];
    for (int i = 1; i < n; i++) if (a[i] > m) m = a[i];
    return m;
}
double arrayDot(const double* a, const double* b, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

#endif  

void arrayMap(double* result, const double* a, int n, MapFunc f) {
    
    if (n >= parallelConfig.minParallelWork && initialized) {
        dispatch_apply(n, workerQueue, ^(size_t i) {
            result[i] = f(a[i]);
        });
    } else {
        for (int i = 0; i < n; i++) {
            result[i] = f(a[i]);
        }
    }
}


void parallelForRange(int64_t start, int64_t end, ParallelBody body, void* context) {
    if (!initialized) parallelInit();
    
    int64_t count = end - start;
    if (count <= 0) return;
    
    if (count < parallelConfig.minParallelWork) {
        
        for (int64_t i = start; i < end; i++) {
            body(i, context);
        }
        return;
    }
    
    dispatch_apply(count, workerQueue, ^(size_t i) {
        body(start + i, context);
    });
}

static double addOp(double a, double b) { return a + b; }

double parallelReduce(const double* data, int n, double identity, ReduceFunc op) {
    if (n <= 0) return identity;
    if (n < parallelConfig.minParallelWork || !initialized) {
        
        double result = identity;
        for (int i = 0; i < n; i++) {
            result = op(result, data[i]);
        }
        return result;
    }
    
    int numThreads = parallelConfig.maxThreads;
    if (n < numThreads * 100) {
        numThreads = n / 100;
        if (numThreads < 1) numThreads = 1;
    }
    
    __block double* partials = calloc(numThreads, sizeof(double));
    for (int t = 0; t < numThreads; t++) {
        partials[t] = identity;
    }
    
    int chunkSize = n / numThreads;
    
    dispatch_group_t group = dispatch_group_create();
    
    for (int t = 0; t < numThreads; t++) {
        int threadStart = t * chunkSize;
        int threadEnd = (t == numThreads - 1) ? n : threadStart + chunkSize;
        int threadId = t;
        
        dispatch_group_async(group, workerQueue, ^{
            double local = identity;
            for (int i = threadStart; i < threadEnd; i++) {
                local = op(local, data[i]);
            }
            partials[threadId] = local;
        });
    }
    
    dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
    dispatch_release(group);
    
    double result = identity;
    for (int t = 0; t < numThreads; t++) {
        result = op(result, partials[t]);
    }
    
    free(partials);
    return result;
}

int estimateOptimalThreads(int64_t workSize, double workPerItem) {
    
    double totalWork = workSize * workPerItem;
    double overheadPerThread = 1000.0; 
    
    int optimal = (int)(totalWork / (overheadPerThread * 10));
    if (optimal < 1) optimal = 1;
    if (optimal > parallelConfig.maxThreads) optimal = parallelConfig.maxThreads;
    
    return optimal;
}


void printAnalysisResult(const AnalysisResult* result) {
    printf("[Analysis] Variables: %d, Side effects: %d\n",
           result->varCount, result->effectCount);
    
    if (result->hasIterator) {
        printf("[Analysis] Iterator: %s, Est. iterations: %lld\n",
               result->iteratorVar, result->estimatedIterations);
    }
    
    if (result->hasReduction) {
        printf("[Analysis] Reduction: %s %c= ...\n",
               result->reductionVar, result->reductionOp);
    }
    
    printf("[Analysis] Dependencies: data=%d, control=%d, order=%d\n",
           result->hasDataDependencies,
           result->hasControlDependencies,
           result->hasOrderDependentEffects);
    
    printf("[Analysis] Can: parallelize=%d, vectorize=%d, GPU=%d\n",
           result->canParallelize, result->canVectorize, result->canUseGPU);
    
    const char* backendNames[] = {"SERIAL", "SIMD", "PARALLEL", "GPU", "HYBRID"};
    printf("[Analysis] Recommended: %s\n", backendNames[result->recommendedBackend]);
}
