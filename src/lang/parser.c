#include "parser.h"
#include <time.h>
#include <sys/time.h>
#include <ctype.h>
#include <strings.h>
#include "../vm/vm.h"
#include "../vm/compiler.h"
#include "../vm/bytecode.h"
#include "../core/parallel.h"
#include <dirent.h>
#include <sys/stat.h>

#define MAX_STATEMENTS 64
#define MAX_INFIXES 64
#define MAX_CALL_ARGS 64

#define MAX_COMPILED_FUNCS 256
typedef struct {
    char name[MAX_NAME_LEN];
    int paramCount;
    bool isVariadic;
    uint64_t bodyHash;
    CompiledFunc* compiled;
} CompiledFuncEntry;

static CompiledFuncEntry compiledFuncs[MAX_COMPILED_FUNCS];
static int compiledFuncCount = 0;


static int statsVMCalls = 0;
static int statsInterpretCalls = 0;
static int statsCacheHits = 0;
static int statsDiskCacheHits = 0;
static int statsDiskCacheSaves = 0;
static bool statsEnabled = false;

void printCompilationStats(void) {
    if (!statsEnabled) return;
    int total = statsVMCalls + statsInterpretCalls;
    if (total == 0) return;
    fprintf(stderr, "\n=== Jaithon Compilation Stats ===\n");
    fprintf(stderr, "VM bytecode executions:    %d (%.1f%%)\n", statsVMCalls, 
            total > 0 ? 100.0 * statsVMCalls / total : 0.0);
    fprintf(stderr, "Interpreted executions:    %d (%.1f%%)\n", statsInterpretCalls,
            total > 0 ? 100.0 * statsInterpretCalls / total : 0.0);
    fprintf(stderr, "Memory cache hits:         %d\n", statsCacheHits);
    fprintf(stderr, "Disk cache hits (.jaic):   %d\n", statsDiskCacheHits);
    fprintf(stderr, "Disk cache saves:          %d\n", statsDiskCacheSaves);
    fprintf(stderr, "Functions compiled:        %d\n", compiledFuncCount);
    fprintf(stderr, "=================================\n");
}

#define MAX_FAILED_FUNCS 256
typedef struct {
    char name[MAX_NAME_LEN];
    int paramCount;
    bool isVariadic;
    uint64_t bodyHash;
} FailedFuncEntry;

static FailedFuncEntry failedFuncs[MAX_FAILED_FUNCS];
static int failedFuncCount = 0;

uint64_t functionBodyHash(const JaiFunction* f) {
    if (!f) return 0;
    if (f->hasBodyHash) return f->bodyHash;
    
    uint64_t h = hashSource(f->body ? f->body : "");
    h ^= (uint64_t)f->paramCount + 0x9e3779b97f4a7c15ULL;
    if (f->isVariadic) {
        h ^= 0xfeedfacecafebeefULL;
    }
    for (int i = 0; i < f->paramCount; i++) {
        const char* p = f->params[i];
        while (p && *p) {
            h ^= (uint8_t)(*p++);
            h *= 1099511628211ULL;
        }
    }
    
    JaiFunction* mutable = (JaiFunction*)f;
    mutable->bodyHash = h;
    mutable->hasBodyHash = true;
    return h;
}

CompiledFunc* getCompiledFunc(JaiFunction* f) {
    if (!f) return NULL;
    
    static bool checkedEnv = false;
    static bool enableVMCompile = true;
    if (!checkedEnv) {
        const char* env = getenv("JAITHON_DISABLE_VM");
        if (env && (strcmp(env, "1") == 0 || strcasecmp(env, "true") == 0)) {
            enableVMCompile = false;
        }
        const char* envOn = getenv("JAITHON_ENABLE_VM");
        if (envOn && (strcmp(envOn, "1") == 0 || strcasecmp(envOn, "true") == 0)) {
            enableVMCompile = true;
        }
        const char* envStats = getenv("JAITHON_STATS");
        if (envStats && (strcmp(envStats, "1") == 0 || strcasecmp(envStats, "true") == 0)) {
            statsEnabled = true;
        }
        checkedEnv = true;
    }
    if (!enableVMCompile) return NULL;
    
    uint64_t hash = functionBodyHash(f);

    for (int i = 0; i < failedFuncCount; i++) {
        if (strcmp(failedFuncs[i].name, f->name) == 0 &&
            failedFuncs[i].paramCount == f->paramCount &&
            failedFuncs[i].isVariadic == f->isVariadic &&
            failedFuncs[i].bodyHash == hash) {
            return NULL;
        }
    }
    
    
    for (int i = 0; i < compiledFuncCount; i++) {
        if (strcmp(compiledFuncs[i].name, f->name) == 0 &&
            compiledFuncs[i].paramCount == f->paramCount &&
            compiledFuncs[i].isVariadic == f->isVariadic &&
            compiledFuncs[i].bodyHash == hash) {
            statsCacheHits++;
            return compiledFuncs[i].compiled;
        }
    }
    
    if (!f->body) {
        return NULL;
    }
    
    
    static bool diskCacheEnabled = true;
    static bool checkedDiskCache = false;
    if (!checkedDiskCache) {
        const char* env = getenv("JAITHON_NO_DISK_CACHE");
        if (env && (strcmp(env, "1") == 0 || strcasecmp(env, "true") == 0)) {
            diskCacheEnabled = false;
        }
        checkedDiskCache = true;
    }
    
    CompiledFunc* compiled = NULL;
    bool loadedFromDisk = false;
    
    if (diskCacheEnabled && runtime.currentSourceFile[0] != '\0') {
        compiled = cacheLoad(f->name, runtime.currentSourceFile, f->body);
        if (compiled) {
            loadedFromDisk = true;
            statsDiskCacheHits++;
            
            compiled->arity = f->paramCount;
            compiled->isVariadic = f->isVariadic;
            if (f->paramCount > 0 && !compiled->paramNames) {
                compiled->paramNames = malloc(sizeof(char*) * f->paramCount);
                for (int i = 0; i < f->paramCount; i++) {
                    compiled->paramNames[i] = strdup(f->params[i]);
                }
            }
        }
    }
    
    
    if (!compiled) {
        Token* tokens = NULL;
        int tokenCount = tokenizeSource(f->body, &tokens);
        static bool tokenDebug = false;
        static bool tokenChecked = false;
        if (!tokenChecked) {
            const char* env = getenv("JAITHON_TOKEN_DEBUG");
            if (env && (strcmp(env, "1") == 0 || strcasecmp(env, "true") == 0)) {
                tokenDebug = true;
            }
            tokenChecked = true;
        }
        if (tokenDebug) {
            fprintf(stderr, "[TOKEN_DEBUG] function %s tokenCount=%d\n", f->name, tokenCount);
            for (int i = 0; i < tokenCount; i++) {
                fprintf(stderr, "  %d: kind=%d line=%d text=%s\n", i, tokens[i].kind, tokens[i].line, tokens[i].strValue);
            }
        }
        if (!tokens || tokenCount == 0) {
            if (tokens) free(tokens);
            return NULL;
        }
        
        compiled = compileFunction(f, tokens, tokenCount);
        free(tokens);
    }
    
    if (!compiled) {
        static bool compileDebug = false;
        static bool checkedDebug = false;
        if (!checkedDebug) {
            const char* env = getenv("JAITHON_COMPILE_DEBUG");
            if (env && (strcmp(env, "1") == 0 || strcasecmp(env, "true") == 0)) {
                compileDebug = true;
            }
            checkedDebug = true;
        }
        if (compileDebug) {
            fprintf(stderr, "[COMPILE_DEBUG] Failed to compile: %s (params=%d, variadic=%d)\n", 
                    f->name, f->paramCount, f->isVariadic);
        }
        if (failedFuncCount < MAX_FAILED_FUNCS) {
            strncpy(failedFuncs[failedFuncCount].name, f->name, MAX_NAME_LEN - 1);
            failedFuncs[failedFuncCount].paramCount = f->paramCount;
            failedFuncs[failedFuncCount].isVariadic = f->isVariadic;
            failedFuncs[failedFuncCount].bodyHash = hash;
            failedFuncCount++;
        }
        return NULL;
    }
    
    if (!loadedFromDisk) {
        compiled->arity = f->paramCount;
        compiled->isVariadic = f->isVariadic;
        if (f->paramCount > 0) {
            compiled->paramNames = malloc(sizeof(char*) * f->paramCount);
            for (int i = 0; i < f->paramCount; i++) {
                compiled->paramNames[i] = strdup(f->params[i]);
            }
        }
        
        
        if (diskCacheEnabled && runtime.currentSourceFile[0] != '\0') {
            if (cacheSave(f->name, runtime.currentSourceFile, compiled, f->body)) {
                statsDiskCacheSaves++;
            }
        }
    }
    
    
    if (compiledFuncCount < MAX_COMPILED_FUNCS) {
        strncpy(compiledFuncs[compiledFuncCount].name, f->name, MAX_NAME_LEN - 1);
        compiledFuncs[compiledFuncCount].paramCount = f->paramCount;
        compiledFuncs[compiledFuncCount].isVariadic = f->isVariadic;
        compiledFuncs[compiledFuncCount].bodyHash = hash;
        compiledFuncs[compiledFuncCount].compiled = compiled;
        compiledFuncCount++;
    }
    
    return compiled;
}

bool registerCompiledFunction(JaiFunction* f, CompiledFunc* compiled, uint64_t bodyHash) {
    if (!f || !compiled) return false;
    
    f->bodyHash = bodyHash;
    f->hasBodyHash = true;
    
    for (int i = 0; i < compiledFuncCount; i++) {
        if (strcmp(compiledFuncs[i].name, f->name) == 0 &&
            compiledFuncs[i].paramCount == f->paramCount &&
            compiledFuncs[i].isVariadic == f->isVariadic) {
            compiledFuncs[i].compiled = compiled;
            compiledFuncs[i].bodyHash = bodyHash;
            return true;
        }
    }
    
    if (compiledFuncCount >= MAX_COMPILED_FUNCS) {
        return false;
    }
    
    compiled->arity = f->paramCount;
    compiled->isVariadic = f->isVariadic;
    if (f->paramCount > 0 && !compiled->paramNames) {
        compiled->paramNames = malloc(sizeof(char*) * f->paramCount);
        for (int i = 0; i < f->paramCount; i++) {
            compiled->paramNames[i] = strdup(f->params[i]);
        }
    }
    
    strncpy(compiledFuncs[compiledFuncCount].name, f->name, MAX_NAME_LEN - 1);
    compiledFuncs[compiledFuncCount].paramCount = f->paramCount;
    compiledFuncs[compiledFuncCount].isVariadic = f->isVariadic;
    compiledFuncs[compiledFuncCount].bodyHash = bodyHash;
    compiledFuncs[compiledFuncCount].compiled = compiled;
    compiledFuncCount++;
    return true;
}

static StatementEntry statements[MAX_STATEMENTS];
static int statementCount = 0;

static InfixEntry infixes[MAX_INFIXES];
static int infixCount = 0;

static Value returnValue;
static bool hasReturn = false;
static bool eagerCompile = true;
static bool eagerStrict = false;
static bool eagerInit = false;

static bool startsWithJavaStyleDecl(Lexer* lex);
static Value convertToTypeName(Value v, const char* typeName);
static Value defaultValueForType(const char* typeName);

static bool isDefinitionStart(Lexer* lex) {
    int k = lex->currentToken.kind;
    if (k == getKW_VAR()) return true;
    if (startsWithJavaStyleDecl(lex)) return true;
    if (k == getKW_FUNC() || k == getKW_CLASS() || k == getKW_NAMESPACE() || k == getKW_IMPORT()) return true;
    if (k == getKW_PUBLIC() || k == getKW_PRIVATE() || k == getKW_PROTECTED() || k == getKW_STATIC()) return true;
    return false;
}

static void skipStatementNoExec(Lexer* lex) {
    int depth = 0;
    while (!lexerCheck(lex, TK_EOF)) {
        int k = lex->currentToken.kind;
        if (k == getKW_IF() || k == getKW_WHILE()) {
            depth++;
        } else if (k == getKW_END()) {
            if (depth == 0) {
                lexerNext(lex);
                break;
            }
            depth = depth > 0 ? depth - 1 : 0;
        }
        if (k == TK_NEWLINE && depth == 0) {
            lexerNext(lex);
            break;
        }
        lexerNext(lex);
    }
}

void registerStatement(int keyword, StatementHandler handler) {
    if (statementCount >= MAX_STATEMENTS) {
        runtimeError("Too many statement handlers");
        return;
    }
    statements[statementCount].keyword = keyword;
    statements[statementCount].handler = handler;
    statementCount++;
}

void registerInfix(int tokenKind, int precedence, ExprHandler handler) {
    if (infixCount >= MAX_INFIXES) {
        runtimeError("Too many infix handlers");
        return;
    }
    infixes[infixCount].tokenKind = tokenKind;
    infixes[infixCount].precedence = precedence;
    infixes[infixCount].infixHandler = handler;
    infixCount++;
}

static InfixEntry* findInfix(int kind) {
    for (int i = 0; i < infixCount; i++) {
        if (infixes[i].tokenKind == kind) {
            return &infixes[i];
        }
    }
    return NULL;
}

static StatementHandler findStatement(int keyword) {
    for (int i = 0; i < statementCount; i++) {
        if (statements[i].keyword == keyword) {
            return statements[i].handler;
        }
    }
    return NULL;
}

static void skipNewlines(Lexer* lex) {
    while (lexerCheck(lex, TK_NEWLINE)) {
        lexerNext(lex);
    }
}

static bool isAccessModifier(int kind) {
    return kind == getKW_PUBLIC() || kind == getKW_PRIVATE() || kind == getKW_PROTECTED();
}

static bool isModifierToken(int kind) {
    return isAccessModifier(kind) || kind == getKW_STATIC();
}

static bool isTypeToken(int kind) {
    return kind == getKW_VAR() ||
           kind == getKW_VOID() ||
           kind == getKW_INT() ||
           kind == getKW_DOUBLE() ||
           kind == getKW_FLOAT() ||
           kind == getKW_STRING() ||
           kind == getKW_CHAR() ||
           kind == getKW_LONG() ||
           kind == getKW_SHORT() ||
           kind == getKW_BYTE() ||
           kind == getKW_BOOL();
}

static bool startsWithJavaStyleDecl(Lexer* lex) {
    Lexer look = *lex;
    bool sawModifier = false;
    
    while (isModifierToken(look.currentToken.kind)) {
        sawModifier = true;
        lexerNext(&look);
    }
    
    if (!isTypeToken(look.currentToken.kind)) return false;
    if (!sawModifier && look.currentToken.kind == getKW_VAR()) return false;
    
    lexerNext(&look);
    return lexerCheck(&look, TK_IDENTIFIER);
}

static bool startsWithJavaStyleFuncDecl(Lexer* lex) {
    Lexer look = *lex;
    while (isModifierToken(look.currentToken.kind)) {
        lexerNext(&look);
    }
    if (!isTypeToken(look.currentToken.kind)) return false;
    lexerNext(&look);
    if (!lexerCheck(&look, TK_IDENTIFIER)) return false;
    lexerNext(&look);
    return lexerCheck(&look, TK_LPAREN);
}

static JaiNamespace* resolveNamespaceTarget(const char* name) {
    if (hasVariable(name)) {
        Value existing = getVariable(name);
        if (existing.type != VAL_NAMESPACE) {
            runtimeError("'%s' is not a namespace", name);
            return NULL;
        }
        return existing.as.namespace;
    }
    Value nsVal = makeNamespace(name);
    setVariable(name, nsVal);
    return nsVal.as.namespace;
}

static void namespaceSetVariable(JaiNamespace* ns, const char* name, Value val, const char* typeName) {
    for (int i = 0; i < ns->varCount; i++) {
        if (strcmp(ns->variables[i].name, name) == 0) {
            if (typeName && typeName[0] != '\0') {
                strncpy(ns->variables[i].declaredType, typeName, MAX_NAME_LEN - 1);
                ns->variables[i].declaredType[MAX_NAME_LEN - 1] = '\0';
            }
            if (ns->variables[i].declaredType[0] != '\0') {
                ns->variables[i].value = convertToTypeName(val, ns->variables[i].declaredType);
            } else {
                ns->variables[i].value = val;
            }
            return;
        }
    }
    
    if (ns->varCount >= ns->varCapacity) {
        int oldCap = ns->varCapacity;
        ns->varCapacity *= GROWTH_FACTOR;
        ns->variables = realloc(ns->variables, sizeof(Variable) * ns->varCapacity);
        for (int i = oldCap; i < ns->varCapacity; i++) {
            ns->variables[i].name[0] = '\0';
            ns->variables[i].declaredType[0] = '\0';
            ns->variables[i].value = makeNull();
        }
    }
    strncpy(ns->variables[ns->varCount].name, name, MAX_NAME_LEN - 1);
    ns->variables[ns->varCount].name[MAX_NAME_LEN - 1] = '\0';
    ns->variables[ns->varCount].declaredType[0] = '\0';
    if (typeName) {
        strncpy(ns->variables[ns->varCount].declaredType, typeName, MAX_NAME_LEN - 1);
        ns->variables[ns->varCount].declaredType[MAX_NAME_LEN - 1] = '\0';
    }
    if (ns->variables[ns->varCount].declaredType[0] != '\0') {
        ns->variables[ns->varCount].value = convertToTypeName(val, ns->variables[ns->varCount].declaredType);
    } else {
        ns->variables[ns->varCount].value = val;
    }
    ns->varCount++;
}

static void namespaceAddFunction(JaiNamespace* ns, JaiFunction* f) {
    if (!ns || !f) return;
    if (ns->funcCount >= ns->funcCapacity) {
        ns->funcCapacity *= 2;
        ns->functions = realloc(ns->functions, sizeof(JaiFunction*) * ns->funcCapacity);
    }
    ns->functions[ns->funcCount++] = f;
    f->namespace = ns;
}

double toNumber(Value v) {
    switch (v.type) {
        case VAL_NUMBER: return v.as.number;
        case VAL_DOUBLE: return v.as.f64;
        case VAL_FLOAT: return v.as.f32;
        case VAL_INT: return v.as.i32;
        case VAL_LONG: return (double)v.as.i64;
        case VAL_SHORT: return v.as.i16;
        case VAL_BYTE: return v.as.i8;
        case VAL_CHAR: return (unsigned char)v.as.ch;
        case VAL_BOOL: return v.as.boolean ? 1.0 : 0.0;
        case VAL_STRING: return v.as.string ? strtod(v.as.string, NULL) : 0.0;
        default: return 0.0;
    }
}

bool toBool(Value v) {
    switch (v.type) {
        case VAL_NUMBER: return v.as.number != 0;
        case VAL_DOUBLE: return v.as.f64 != 0;
        case VAL_FLOAT: return v.as.f32 != 0;
        case VAL_INT: return v.as.i32 != 0;
        case VAL_LONG: return v.as.i64 != 0;
        case VAL_SHORT: return v.as.i16 != 0;
        case VAL_BYTE: return v.as.i8 != 0;
        case VAL_CHAR: return v.as.ch != 0;
        case VAL_BOOL: return v.as.boolean;
        case VAL_STRING: return strlen(v.as.string) > 0;
        case VAL_NULL: return false;
        default: return true;
    }
}

static bool isNumericValueType(ValueType t) {
    return t == VAL_NUMBER || t == VAL_DOUBLE || t == VAL_FLOAT || t == VAL_INT ||
           t == VAL_LONG || t == VAL_SHORT || t == VAL_BYTE || t == VAL_CHAR;
}

static char* valueToString(Value v) {
    char buf[256];
    switch (v.type) {
        case VAL_STRING:
            return v.as.string ? strdup(v.as.string) : strdup("");
        case VAL_CHAR:
            snprintf(buf, sizeof(buf), "%c", v.as.ch);
            break;
        case VAL_NUMBER:
            snprintf(buf, sizeof(buf), "%g", v.as.number);
            break;
        case VAL_DOUBLE:
            snprintf(buf, sizeof(buf), "%g", v.as.f64);
            break;
        case VAL_FLOAT:
            snprintf(buf, sizeof(buf), "%g", v.as.f32);
            break;
        case VAL_INT:
            snprintf(buf, sizeof(buf), "%d", v.as.i32);
            break;
        case VAL_LONG:
            snprintf(buf, sizeof(buf), "%lld", (long long)v.as.i64);
            break;
        case VAL_SHORT:
            snprintf(buf, sizeof(buf), "%d", v.as.i16);
            break;
        case VAL_BYTE:
            snprintf(buf, sizeof(buf), "%d", v.as.i8);
            break;
        case VAL_BOOL:
            snprintf(buf, sizeof(buf), "%s", v.as.boolean ? "true" : "false");
            break;
        default:
            snprintf(buf, sizeof(buf), "null");
            break;
    }
    return strdup(buf);
}

static Value convertToTypeName(Value v, const char* typeName) {
    if (!typeName || typeName[0] == '\0' || strcasecmp(typeName, "var") == 0) {
        return v;
    }
    if (strcasecmp(typeName, "int") == 0) return makeInt((int32_t)toNumber(v));
    if (strcasecmp(typeName, "long") == 0 || strcasecmp(typeName, "long long") == 0) return makeLong((int64_t)toNumber(v));
    if (strcasecmp(typeName, "short") == 0) return makeShort((int16_t)toNumber(v));
    if (strcasecmp(typeName, "byte") == 0) return makeByte((int8_t)toNumber(v));
    if (strcasecmp(typeName, "float") == 0) return makeFloat((float)toNumber(v));
    if (strcasecmp(typeName, "double") == 0 || strcasecmp(typeName, "number") == 0) return makeDouble(toNumber(v));
    if (strcasecmp(typeName, "char") == 0) {
        if (v.type == VAL_STRING && v.as.string && strlen(v.as.string) > 0) {
            return makeChar(v.as.string[0]);
        }
        return makeChar((char)(int)toNumber(v));
    }
    if (strcasecmp(typeName, "bool") == 0) return makeBool(toBool(v));
    if (strcasecmp(typeName, "string") == 0) {
        if (v.type == VAL_STRING) return v;
        char buf[64];
        snprintf(buf, sizeof(buf), "%g", toNumber(v));
        return makeString(buf);
    }
    return v;
}

static Value defaultValueForType(const char* typeName) {
    if (!typeName || typeName[0] == '\0' || strcasecmp(typeName, "var") == 0) return makeNull();
    if (strcasecmp(typeName, "int") == 0) return makeInt(0);
    if (strcasecmp(typeName, "long") == 0 || strcasecmp(typeName, "long long") == 0) return makeLong(0);
    if (strcasecmp(typeName, "short") == 0) return makeShort(0);
    if (strcasecmp(typeName, "byte") == 0) return makeByte(0);
    if (strcasecmp(typeName, "float") == 0) return makeFloat(0.0f);
    if (strcasecmp(typeName, "double") == 0 || strcasecmp(typeName, "number") == 0) return makeDouble(0.0);
    if (strcasecmp(typeName, "char") == 0) return makeChar('\0');
    if (strcasecmp(typeName, "bool") == 0) return makeBool(false);
    if (strcasecmp(typeName, "string") == 0) return makeString("");
    return makeNull();
}

static bool methodExpectsSelf(JaiFunction* m) {
    return m && m->paramCount > 0 && strcmp(m->params[0], "self") == 0;
}

static Value stubGuiMousePos(Value* args, int argc) {
    (void)args; (void)argc;
    Value arr = makeArray(2);
    arrayPush(arr.as.array, makeNumber(0));
    arrayPush(arr.as.array, makeNumber(0));
    arr.as.array->length = 2;
    return arr;
}

static Value stubGuiMouseDown(Value* args, int argc) {
    (void)args; (void)argc;
    return makeBool(false);
}

static Value stubGuiKeyDown(Value* args, int argc) {
    (void)args; (void)argc;
    return makeBool(false);
}

static Value stubGuiPoll(Value* args, int argc) {
    (void)args; (void)argc;
    return makeNull();
}

Value parsePrimary(Lexer* lex) {
    Token t = lex->currentToken;
    
    if (t.kind == TK_NUMBER) {
        lexerNext(lex);
        return makeNumber(t.numValue);
    }
    
    if (t.kind == TK_STRING) {
        lexerNext(lex);
        return makeString(t.strValue);
    }
    
    if (t.kind == getKW_TRUE()) {
        lexerNext(lex);
        return makeBool(true);
    }
    
    if (t.kind == getKW_FALSE()) {
        lexerNext(lex);
        return makeBool(false);
    }
    
    if (t.kind == getKW_NULL()) {
        lexerNext(lex);
        return makeNull();
    }
    
    if (t.kind == getKW_NEW()) {
        lexerNext(lex);
        
        if (!lexerCheck(lex, TK_IDENTIFIER)) {
            runtimeError("Expected class name after 'new'");
            return makeNull();
        }
        char className[MAX_NAME_LEN];
        strcpy(className, lex->currentToken.strValue);
        lexerNext(lex);
        
        JaiClass* cls = findClass(className);
        if (!cls) {
            runtimeError("Class not found: %s", className);
            return makeNull();
        }
        
        Value obj = makeObject(cls);
        
        if (lexerMatch(lex, TK_LPAREN)) {
            Value args[MAX_CALL_ARGS];
            args[0] = obj;
            int argc = 1;
            bool userPassedArgs = false;
            
            if (!lexerCheck(lex, TK_RPAREN)) {
                do {
                    if (argc >= MAX_CALL_ARGS) {
                        runtimeError("Too many constructor arguments");
                        return makeNull();
                    }
                    args[argc++] = parseExpression(lex);
                    userPassedArgs = true;
                } while (lexerMatch(lex, TK_COMMA));
            }
            lexerExpect(lex, TK_RPAREN);
            
            if (cls->constructor) {
                if (cls->constructor->paramCount == argc ||
                    (!userPassedArgs && cls->constructor->paramCount == 1)) {
                    callValue(makeFunction(cls->constructor), args, argc);
                }
            }
        }
        
        return obj;
    }
    
    if (t.kind == TK_LPAREN) {
        lexerNext(lex);
        Value v = parseExpression(lex);
        lexerExpect(lex, TK_RPAREN);
        return v;
    }
    
    if (t.kind == TK_IDENTIFIER || t.kind == getKW_SELF()) {
        char name[MAX_NAME_LEN];
        strcpy(name, t.strValue);
        lexerNext(lex);
        
        Value result;
        
        if (lexerCheck(lex, TK_LPAREN)) {
            lexerNext(lex);
            Value args[MAX_CALL_ARGS];
            int argc = 0;
            
            if (!lexerCheck(lex, TK_RPAREN)) {
                do {
                    if (argc >= MAX_CALL_ARGS) {
                        runtimeError("Too many arguments");
                        return makeNull();
                    }
                    args[argc++] = parseExpression(lex);
                } while (lexerMatch(lex, TK_COMMA));
            }
            lexerExpect(lex, TK_RPAREN);
            
            Value callee = getVariable(name);
            result = callValue(callee, args, argc);
        } else {
            result = getVariable(name);
        }
        
        while (lexerCheck(lex, TK_LBRACKET) || lexerCheck(lex, TK_DOT)) {
            if (lexerMatch(lex, TK_LBRACKET)) {
                Value index = parseExpression(lex);
                lexerExpect(lex, TK_RBRACKET);
                
                if (lexerCheck(lex, TK_EQUALS)) {
                    lexerNext(lex);
                    Value val = parseExpression(lex);
                    if (result.type == VAL_ARRAY) {
                        arraySet(result.as.array, (int)toNumber(index), val);
                        return val;
                    }
                    runtimeError("Cannot assign to index of non-array");
                    return makeNull();
                }
                
                if (result.type == VAL_ARRAY) {
                    result = arrayGet(result.as.array, (int)toNumber(index));
                } else if (result.type == VAL_STRING) {
                    int idx = (int)toNumber(index);
                    const char* s = result.as.string;
                    int slen = strlen(s);
                    if (idx >= 0 && idx < slen) {
                        char buf[2] = {s[idx], '\0'};
                        result = makeString(buf);
                    } else {
                        result = makeString("");
                    }
                } else {
                    runtimeError("Cannot index non-array/string value");
                    return makeNull();
                }
            } else if (lexerMatch(lex, TK_DOT)) {
                if (!lexerCheck(lex, TK_IDENTIFIER)) {
                    runtimeError("Expected field name after '.'");
                    return makeNull();
                }
                char fieldName[MAX_NAME_LEN];
                strcpy(fieldName, lex->currentToken.strValue);
                lexerNext(lex);
                
                if (result.type == VAL_NAMESPACE) {
                    JaiNamespace* ns = result.as.namespace;
                    if (lexerCheck(lex, TK_LPAREN)) {
                         JaiFunction* func = NULL;
                         for(int i=0; i<ns->funcCount; i++) {
                            if(strcmp(ns->functions[i]->name, fieldName) == 0) {
                                func = ns->functions[i];
                                break;
                            }
                         }
                         if (!func) {
                             for(int i=0; i<ns->varCount; i++) {
                                if(strcmp(ns->variables[i].name, fieldName) == 0) {
                                    if (ns->variables[i].value.type == VAL_FUNCTION) {
                                        func = ns->variables[i].value.as.function;
                                    }
                                    break;
                                }
                             }
                         }
                         if (!func) {
                             runtimeError("Namespace '%s' has no function '%s'", ns->name, fieldName);
                             return makeNull();
                         }
                         lexerNext(lex);
                         Value args[MAX_CALL_ARGS];
                         int argc = 0;
                         if (!lexerCheck(lex, TK_RPAREN)) {
                            do {
                                args[argc++] = parseExpression(lex);
                            } while (lexerMatch(lex, TK_COMMA));
                         }
                         lexerExpect(lex, TK_RPAREN);
                         result = callValue(makeFunction(func), args, argc);
                    }
                    else if (lexerCheck(lex, TK_EQUALS)) {
                        lexerNext(lex);
                        Value val = parseExpression(lex);
                        bool found = false;
                        for(int i=0; i<ns->varCount; i++) {
                            if(strcmp(ns->variables[i].name, fieldName) == 0) {
                                ns->variables[i].value = val;
                                found = true;
                                break;
                            }
                        }
                        if (!found) {
                            if (ns->varCount >= ns->varCapacity) {
                                ns->varCapacity *= 2;
                                ns->variables = realloc(ns->variables, sizeof(Variable) * ns->varCapacity);
                            }
                            strcpy(ns->variables[ns->varCount].name, fieldName);
                            ns->variables[ns->varCount].value = val;
                            ns->varCount++;
                        }
                        return val;
                    }
                    else {
                        bool found = false;
                        for(int i=0; i<ns->varCount; i++) {
                            if(strcmp(ns->variables[i].name, fieldName) == 0) {
                                result = ns->variables[i].value;
                                found = true;
                                break;
                            }
                        }
                        if (!found) {
                            for(int i=0; i<ns->funcCount; i++) {
                                if(strcmp(ns->functions[i]->name, fieldName) == 0) {
                                    result = makeFunction(ns->functions[i]);
                                    found = true;
                                    break;
                                }
                            }
                        }
                        if (!found) {
                            runtimeError("Namespace '%s' has no member '%s'", ns->name, fieldName);
                            return makeNull();
                        }
                    }
                } else if (result.type == VAL_OBJECT) {
                    if (lexerCheck(lex, TK_LPAREN)) {
                        JaiFunction* method = objectGetMethod(result.as.object, fieldName);
                        if (!method) {
                            runtimeError("Object has no method: %s", fieldName);
                            return makeNull();
                        }
                        lexerNext(lex);
                        Value args[MAX_CALL_ARGS];
                        int argc = 0;
                        if (methodExpectsSelf(method)) {
                            args[argc++] = result;
                        }
                        if (!lexerCheck(lex, TK_RPAREN)) {
                            do {
                                if (argc >= MAX_CALL_ARGS) {
                                    runtimeError("Too many arguments");
                                    return makeNull();
                                }
                                args[argc++] = parseExpression(lex);
                            } while (lexerMatch(lex, TK_COMMA));
                        }
                        lexerExpect(lex, TK_RPAREN);
                        result = callValue(makeFunction(method), args, argc);
                    } else if (lexerCheck(lex, TK_EQUALS)) {
                        lexerNext(lex);
                        Value val = parseExpression(lex);
                        objectSetField(result.as.object, fieldName, val);
                        return val;
                    } else {
                        result = objectGetField(result.as.object, fieldName);
                    }
                } else {
                    runtimeError("Cannot access field '%s' of non-object", fieldName);
                    return makeNull();
                }
            }
        }

        if (lexerCheck(lex, TK_EQUALS)) {
            lexerNext(lex);
            Value val = parseExpression(lex);
            setVariable(name, val);
            return val;
        }

        return result;
    }
    
    if (t.kind == TK_MINUS) {
        lexerNext(lex);
        Value v = parsePrimary(lex);
        return makeNumber(-toNumber(v));
    }
    
    if (t.kind == getKW_NOT()) {
        lexerNext(lex);
        Value v = parsePrimary(lex);
        return makeBool(!toBool(v));
    }

    if (t.kind == TK_LBRACKET) {
        lexerNext(lex);
        Value arr = makeArray(INITIAL_CAPACITY);
        
        if (!lexerCheck(lex, TK_RBRACKET)) {
            do {
                Value elem = parseExpression(lex);
                arrayPush(arr.as.array, elem);
            } while (lexerMatch(lex, TK_COMMA));
        }
        lexerExpect(lex, TK_RBRACKET);
        return arr;
    }
    
    runtimeError("Unexpected token: %s", tokenKindName(t.kind));
    lexerNext(lex);
    return makeNull();
}

Value parseExpressionPrec(Lexer* lex, int minPrec) {
    Value left = parsePrimary(lex);
    
    for (;;) {
        InfixEntry* infix = findInfix(lex->currentToken.kind);
        if (!infix || infix->precedence < minPrec) {
            break;
        }
        
        int kind = lex->currentToken.kind;
        lexerNext(lex);
        left = infix->infixHandler(lex, left);
        (void)kind;
    }
    
    return left;
}

Value parseExpression(Lexer* lex) {
    return parseExpressionPrec(lex, 1);
}

static Value handleAdd(Lexer* lex, Value left) {
    Value right = parseExpressionPrec(lex, 7);
    if (left.type == VAL_STRING || right.type == VAL_STRING ||
        left.type == VAL_CHAR || right.type == VAL_CHAR) {
        char* lstr = valueToString(left);
        char* rstr = valueToString(right);
        size_t len = strlen(lstr) + strlen(rstr);
        char* buf = malloc(len + 1);
        memcpy(buf, lstr, strlen(lstr));
        memcpy(buf + strlen(lstr), rstr, strlen(rstr) + 1);
        Value v;
        v.type = VAL_STRING;
        v.as.string = buf;
        free(lstr);
        free(rstr);
        return v;
    }
    return makeDouble(toNumber(left) + toNumber(right));
}

static Value handleSub(Lexer* lex, Value left) {
    Value right = parseExpressionPrec(lex, 7);
    return makeNumber(toNumber(left) - toNumber(right));
}

static Value handleMul(Lexer* lex, Value left) {
    Value right = parseExpressionPrec(lex, 8);
    return makeNumber(toNumber(left) * toNumber(right));
}

static Value handleDiv(Lexer* lex, Value left) {
    Value right = parseExpressionPrec(lex, 8);
    double d = toNumber(right);
    if (d == 0) {
        runtimeError("Division by zero");
        return makeNull();
    }
    return makeNumber(toNumber(left) / d);
}

static Value handleMod(Lexer* lex, Value left) {
    Value right = parseExpressionPrec(lex, 8);
    return makeNumber(fmod(toNumber(left), toNumber(right)));
}

static Value handleDot(Lexer* lex, Value left) {
    if (!lexerCheck(lex, TK_IDENTIFIER)) {
        runtimeError("Expected identifier after '.'");
        return makeNull();
    }
    char fieldName[MAX_NAME_LEN];
    strcpy(fieldName, lex->currentToken.strValue);
    lexerNext(lex);

    if (left.type == VAL_NAMESPACE) {
        JaiNamespace* ns = left.as.namespace;
        
        if (lexerCheck(lex, TK_LPAREN)) {
             JaiFunction* func = NULL;
             for(int i=0; i<ns->funcCount; i++) {
                if(strcmp(ns->functions[i]->name, fieldName) == 0) {
                    func = ns->functions[i];
                    break;
                }
             }
             if (!func) {
                 for(int i=0; i<ns->varCount; i++) {
                    if(strcmp(ns->variables[i].name, fieldName) == 0) {
                        if (ns->variables[i].value.type == VAL_FUNCTION) {
                            func = ns->variables[i].value.as.function;
                        }
                        break;
                    }
                 }
             }
             
             if (!func) {
                 runtimeError("Namespace '%s' has no function '%s'", ns->name, fieldName);
                 return makeNull();
             }
             
             lexerNext(lex);
             Value args[MAX_CALL_ARGS];
             int argc = 0;
             if (!lexerCheck(lex, TK_RPAREN)) {
                do {
                    args[argc++] = parseExpression(lex);
                } while (lexerMatch(lex, TK_COMMA));
             }
             lexerExpect(lex, TK_RPAREN);
             return callValue(makeFunction(func), args, argc);
        }
        
        for(int i=0; i<ns->varCount; i++) {
            if(strcmp(ns->variables[i].name, fieldName) == 0) {
                return ns->variables[i].value;
            }
        }
        for(int i=0; i<ns->funcCount; i++) {
            if(strcmp(ns->functions[i]->name, fieldName) == 0) {
                return makeFunction(ns->functions[i]);
            }
        }
        
        runtimeError("Namespace '%s' has no member '%s'", ns->name, fieldName);
        return makeNull();
    }

    if (lexerCheck(lex, TK_LPAREN)) {
        lexerNext(lex);
        Value args[MAX_CALL_ARGS];
        int argc = 0;

        if (left.type != VAL_OBJECT) {
            runtimeError("Cannot call method on non-object");
            return makeNull();
        }
        JaiFunction* method = objectGetMethod(left.as.object, fieldName);
        if (!method) {
            runtimeError("Object has no method: %s", fieldName);
            return makeNull();
        }

        if (methodExpectsSelf(method)) {
            args[argc++] = left;
        }
        if (!lexerCheck(lex, TK_RPAREN)) {
            do {
                if (argc >= MAX_CALL_ARGS) {
                    runtimeError("Too many arguments");
                    return makeNull();
                }
                args[argc++] = parseExpression(lex);
            } while (lexerMatch(lex, TK_COMMA));
        }
        lexerExpect(lex, TK_RPAREN);
        
        return callValue(makeFunction(method), args, argc);
    }

    if (lexerCheck(lex, TK_EQUALS)) {
        lexerNext(lex);
        Value val = parseExpression(lex);
        if (left.type != VAL_OBJECT) {
            runtimeError("Cannot set field on non-object");
            return makeNull();
        }
        objectSetField(left.as.object, fieldName, val);
        return val;
    }

    if (left.type == VAL_OBJECT) {
        return objectGetField(left.as.object, fieldName);
    }
    runtimeError("Cannot access field '%s' of non-object", fieldName);
    return makeNull();
}

static Value handlePow(Lexer* lex, Value left) {
    Value right = parseExpressionPrec(lex, 9);
    return makeNumber(pow(toNumber(left), toNumber(right)));
}

static Value handleFactorial(Lexer* lex, Value left) {
    (void)lex;
    int n = (int)toNumber(left);
    int result = 1;
    for (int i = 2; i <= n; i++) result *= i;
    return makeNumber(result);
}

static Value handleGt(Lexer* lex, Value left) {
    Value right = parseExpressionPrec(lex, 6);
    return makeBool(toNumber(left) > toNumber(right));
}

static Value handleLt(Lexer* lex, Value left) {
    Value right = parseExpressionPrec(lex, 6);
    return makeBool(toNumber(left) < toNumber(right));
}

static Value handleGe(Lexer* lex, Value left) {
    Value right = parseExpressionPrec(lex, 6);
    return makeBool(toNumber(left) >= toNumber(right));
}

static Value handleLe(Lexer* lex, Value left) {
    Value right = parseExpressionPrec(lex, 6);
    return makeBool(toNumber(left) <= toNumber(right));
}

static Value handleEq(Lexer* lex, Value left) {
    Value right = parseExpressionPrec(lex, 5);
    if (isNumericValueType(left.type) && isNumericValueType(right.type)) {
        return makeBool(toNumber(left) == toNumber(right));
    }
    if (left.type != right.type) return makeBool(false);
    switch (left.type) {
        case VAL_NUMBER: return makeBool(left.as.number == right.as.number);
        case VAL_DOUBLE: return makeBool(left.as.f64 == right.as.f64);
        case VAL_FLOAT: return makeBool(left.as.f32 == right.as.f32);
        case VAL_INT: return makeBool(left.as.i32 == right.as.i32);
        case VAL_LONG: return makeBool(left.as.i64 == right.as.i64);
        case VAL_SHORT: return makeBool(left.as.i16 == right.as.i16);
        case VAL_BYTE: return makeBool(left.as.i8 == right.as.i8);
        case VAL_CHAR: return makeBool(left.as.ch == right.as.ch);
        case VAL_BOOL: return makeBool(left.as.boolean == right.as.boolean);
        case VAL_STRING: return makeBool(strcmp(left.as.string, right.as.string) == 0);
        case VAL_NULL: return makeBool(true);
        default: return makeBool(false);
    }
}

static Value handleNe(Lexer* lex, Value left) {
    Value right = parseExpressionPrec(lex, 5);
    if (isNumericValueType(left.type) && isNumericValueType(right.type)) {
        return makeBool(toNumber(left) != toNumber(right));
    }
    if (left.type != right.type) return makeBool(true);
    switch (left.type) {
        case VAL_NUMBER: return makeBool(left.as.number != right.as.number);
        case VAL_DOUBLE: return makeBool(left.as.f64 != right.as.f64);
        case VAL_FLOAT: return makeBool(left.as.f32 != right.as.f32);
        case VAL_INT: return makeBool(left.as.i32 != right.as.i32);
        case VAL_LONG: return makeBool(left.as.i64 != right.as.i64);
        case VAL_SHORT: return makeBool(left.as.i16 != right.as.i16);
        case VAL_BYTE: return makeBool(left.as.i8 != right.as.i8);
        case VAL_CHAR: return makeBool(left.as.ch != right.as.ch);
        case VAL_BOOL: return makeBool(left.as.boolean != right.as.boolean);
        case VAL_STRING: return makeBool(strcmp(left.as.string, right.as.string) != 0);
        case VAL_NULL: return makeBool(false);
        default: return makeBool(true);
    }
}

static Value handleAnd(Lexer* lex, Value left) {
    Value right = parseExpressionPrec(lex, 4);
    if (!toBool(left)) return makeBool(false);
    return makeBool(toBool(right));
}

static Value handleOr(Lexer* lex, Value left) {
    Value right = parseExpressionPrec(lex, 3);
    if (toBool(left)) return makeBool(true);
    return makeBool(toBool(right));
}

static Value stmtVar(Lexer* lex) {
    lexerExpect(lex, getKW_VAR());
    
    if (!lexerCheck(lex, TK_IDENTIFIER)) {
        runtimeError("Expected variable name");
        return makeNull();
    }
    char name[MAX_NAME_LEN];
    strcpy(name, lex->currentToken.strValue);
    lexerNext(lex);
    
    JaiNamespace* targetNs = NULL;
    if (lexerMatch(lex, getKW_IN())) {
        if (!lexerCheck(lex, TK_IDENTIFIER)) {
            runtimeError("Expected namespace name after 'in'");
            return makeNull();
        }
        targetNs = resolveNamespaceTarget(lex->currentToken.strValue);
        lexerNext(lex);
    }
    
    Value val = makeNull();
    if (lexerMatch(lex, TK_EQUALS)) {
        val = parseExpression(lex);
    }
    
    if (targetNs) {
        namespaceSetVariable(targetNs, name, val, "");
    } else {
        setVariable(name, val);
    }
    
    if (runtime.debug) {
        printf("Set %s = ", name);
        switch (val.type) {
            case VAL_NUMBER: printf("%g\n", val.as.number); break;
            case VAL_DOUBLE: printf("%g\n", val.as.f64); break;
            case VAL_FLOAT: printf("%g\n", val.as.f32); break;
            case VAL_INT: printf("%d\n", val.as.i32); break;
            case VAL_LONG: printf("%lld\n", (long long)val.as.i64); break;
            case VAL_SHORT: printf("%d\n", val.as.i16); break;
            case VAL_BYTE: printf("%d\n", val.as.i8); break;
            case VAL_CHAR: printf("%c\n", val.as.ch); break;
            case VAL_STRING: printf("\"%s\"\n", val.as.string); break;
            case VAL_BOOL: printf("%s\n", val.as.boolean ? "true" : "false"); break;
            default: printf("null\n"); break;
        }
    }
    
    return val;
}

static Value stmtPrint(Lexer* lex) {
    lexerExpect(lex, getKW_PRINT());
    Value val = parseExpression(lex);
    
    switch (val.type) {
        case VAL_NUMBER: {
            double n = val.as.number;
            if (n == (int)n) {
                printf("%d\n", (int)n);
            } else {
                printf("%g\n", n);
            }
            break;
        }
        case VAL_DOUBLE: {
            double n = val.as.f64;
            if (n == (int)n) printf("%d\n", (int)n);
            else printf("%g\n", n);
            break;
        }
        case VAL_FLOAT: {
            double n = val.as.f32;
            if (n == (int)n) printf("%d\n", (int)n);
            else printf("%g\n", n);
            break;
        }
        case VAL_INT:
            printf("%d\n", val.as.i32);
            break;
        case VAL_LONG:
            printf("%lld\n", (long long)val.as.i64);
            break;
        case VAL_SHORT:
            printf("%d\n", val.as.i16);
            break;
        case VAL_BYTE:
            printf("%d\n", val.as.i8);
            break;
        case VAL_CHAR:
            printf("%c\n", val.as.ch);
            break;
        case VAL_STRING:
            printf("%s\n", val.as.string);
            break;
        case VAL_BOOL:
            printf("%d\n", val.as.boolean ? 1 : 0);
            break;
        case VAL_NULL:
            printf("null\n");
            break;
        case VAL_FUNCTION:
            printf("<function %s>\n", val.as.function->name);
            break;
        case VAL_NATIVE_FUNC:
            printf("<native function>\n");
            break;
        default:
            printf("<unknown>\n");
    }
    
    return val;
}

static Value stmtIf(Lexer* lex) {
    lexerExpect(lex, getKW_IF());
    Value cond = parseExpression(lex);
    
    lexerMatch(lex, getKW_THEN()); 
    lexerMatch(lex, getKW_DO());
    skipNewlines(lex);
    
    Value result = makeNull();
    
    if (toBool(cond)) {
        while (!lexerCheck(lex, getKW_END()) && !lexerCheck(lex, getKW_ELSE()) && !lexerCheck(lex, TK_EOF)) {
            result = parseStatement(lex);
            skipNewlines(lex);
            if (hasReturn) break;
        }
        if (lexerMatch(lex, getKW_ELSE())) {
            skipNewlines(lex);
            int depth = 1;
            while (depth > 0 && !lexerCheck(lex, TK_EOF)) {
                int kind = lex->currentToken.kind;
                if (kind == getKW_IF() || kind == getKW_WHILE() || kind == getKW_FUNC()) {
                    depth++;
                }
                if (kind == getKW_END()) {
                    depth--;
                }
                if (depth > 0) lexerNext(lex);
            }
        }
    } else {
        int depth = 1;
        while (depth > 0 && !lexerCheck(lex, TK_EOF)) {
            int kind = lex->currentToken.kind;
            if (kind == getKW_IF() || kind == getKW_WHILE() || kind == getKW_FUNC()) {
                depth++;
            }
            if (kind == getKW_END()) {
                depth--;
            }
            if (depth == 1 && kind == getKW_ELSE()) {
                lexerNext(lex);
                skipNewlines(lex);
                while (!lexerCheck(lex, getKW_END()) && !lexerCheck(lex, TK_EOF)) {
                    result = parseStatement(lex);
                    skipNewlines(lex);
                    if (hasReturn) break;
                }
                break;
            }
            if (depth > 0) lexerNext(lex);
        }
    }
    
    lexerExpect(lex, getKW_END());
    return result;
}

static Value stmtWhileVM(Lexer* lex) {
    
    lexerExpect(lex, getKW_WHILE());
    
    
    const char* condStart = lex->start;
    
    int parenDepth = 0;
    while (!lexerCheck(lex, TK_EOF)) {
        int kind = lex->currentToken.kind;
        if (kind == getKW_DO() && parenDepth == 0) break;
        if (kind == TK_NEWLINE && parenDepth == 0) break;
        if (kind == TK_LPAREN) parenDepth++;
        if (kind == TK_RPAREN) parenDepth--;
        lexerNext(lex);
    }
    const char* condEnd = lex->start;
    
    lexerMatch(lex, getKW_DO());
    skipNewlines(lex);
    
    
    const char* bodyStart = lex->start;
    int depth = 1;
    while (depth > 0 && !lexerCheck(lex, TK_EOF)) {
        int kind = lex->currentToken.kind;
        if (kind == getKW_WHILE() || kind == getKW_IF() || kind == getKW_FUNC() || kind == getKW_CLASS()) {
            depth++;
        }
        if (kind == getKW_END()) {
            depth--;
            if (depth == 0) break;
        }
        lexerNext(lex);
    }
    const char* bodyEnd = lex->start;
    
    
    int bodyLen = (int)(bodyEnd - bodyStart);
    char* bodySrc = malloc(bodyLen + 1);
    memcpy(bodySrc, bodyStart, bodyLen);
    bodySrc[bodyLen] = '\0';
    
    int condLen = (int)(condEnd - condStart);
    char* condSrc = malloc(condLen + 1);
    memcpy(condSrc, condStart, condLen);
    condSrc[condLen] = '\0';
    
    
    
    
    
    
    Value result = executeWhileLoop(condSrc, bodySrc);
    
    free(bodySrc);
    free(condSrc);
    
    lexerExpect(lex, getKW_END());
    return result;
}

static Value stmtWhile(Lexer* lex) {
    return stmtWhileVM(lex);
}

static Value stmtJavaStyleDecl(Lexer* lex) {
    
    while (isModifierToken(lex->currentToken.kind)) {
        lexerNext(lex);
    }
    
    if (!isTypeToken(lex->currentToken.kind)) {
        runtimeError("Expected type or 'var' after modifiers");
        return makeNull();
    }
    
    char returnType[MAX_NAME_LEN];
    strncpy(returnType, lex->currentToken.strValue, MAX_NAME_LEN - 1);
    returnType[MAX_NAME_LEN - 1] = '\0';
    lexerNext(lex);
    
    if (!lexerCheck(lex, TK_IDENTIFIER)) {
        runtimeError("Expected name after type");
        return makeNull();
    }
    
    char name[MAX_NAME_LEN];
    strncpy(name, lex->currentToken.strValue, MAX_NAME_LEN - 1);
    name[MAX_NAME_LEN - 1] = '\0';
    lexerNext(lex);
    
    JaiNamespace* targetNs = NULL;
    if (lexerMatch(lex, getKW_IN())) {
        if (!lexerCheck(lex, TK_IDENTIFIER)) {
            runtimeError("Expected namespace name after 'in'");
            return makeNull();
        }
        targetNs = resolveNamespaceTarget(lex->currentToken.strValue);
        lexerNext(lex);
    }
    
    if (!lexerCheck(lex, TK_LPAREN)) {
        Value val = defaultValueForType(returnType);
        if (lexerMatch(lex, TK_EQUALS)) {
            val = parseExpression(lex);
        }
        val = convertToTypeName(val, returnType);
        
        if (targetNs) {
            namespaceSetVariable(targetNs, name, val, returnType);
        } else {
            setTypedVariable(name, val, returnType);
        }
        return val;
    }
    
    lexerExpect(lex, TK_LPAREN);
    
    int paramCapacity = 8;
    char** params = malloc(sizeof(char*) * paramCapacity);
    char** paramTypes = malloc(sizeof(char*) * paramCapacity);
    int paramCount = 0;
    bool isVariadic = false;
    
    if (!lexerCheck(lex, TK_RPAREN)) {
        do {
            if (lexerMatch(lex, TK_STAR)) {
                isVariadic = true;
            }
            if (!isTypeToken(lex->currentToken.kind)) {
                runtimeError("Expected parameter type");
                for (int i = 0; i < paramCount; i++) {
                    free(params[i]);
                    free(paramTypes[i]);
                }
                free(params);
                free(paramTypes);
                return makeNull();
            }
            
            char typeName[MAX_NAME_LEN];
            strncpy(typeName, lex->currentToken.strValue, MAX_NAME_LEN - 1);
            typeName[MAX_NAME_LEN - 1] = '\0';
            lexerNext(lex);
            
            if (!lexerCheck(lex, TK_IDENTIFIER)) {
                runtimeError("Expected parameter name");
                for (int i = 0; i < paramCount; i++) {
                    free(params[i]);
                    free(paramTypes[i]);
                }
                free(params);
                free(paramTypes);
                return makeNull();
            }
            if (paramCount >= paramCapacity) {
                paramCapacity *= 2;
                params = realloc(params, sizeof(char*) * paramCapacity);
                paramTypes = realloc(paramTypes, sizeof(char*) * paramCapacity);
            }
            params[paramCount] = strdup(lex->currentToken.strValue);
            paramTypes[paramCount] = strdup(typeName);
            paramCount++;
            lexerNext(lex);
        } while (lexerMatch(lex, TK_COMMA));
    }
    lexerExpect(lex, TK_RPAREN);
    
    if (!targetNs && lexerMatch(lex, getKW_IN())) {
        if (!lexerCheck(lex, TK_IDENTIFIER)) {
            runtimeError("Expected namespace name after 'in'");
            for (int i = 0; i < paramCount; i++) {
                free(params[i]);
                free(paramTypes[i]);
            }
            free(params);
            free(paramTypes);
            return makeNull();
        }
        targetNs = resolveNamespaceTarget(lex->currentToken.strValue);
        lexerNext(lex);
    }
    
    skipNewlines(lex);
    
    const char* bodyStart = lex->start;
    const char* bodyEnd = NULL;
    int depth = 1;
    while (depth > 0 && !lexerCheck(lex, TK_EOF)) {
        if (lexerCheck(lex, getKW_END())) {
            depth--;
        } else if (lexerCheck(lex, getKW_IF()) || lexerCheck(lex, getKW_WHILE()) || lexerCheck(lex, getKW_FUNC()) || startsWithJavaStyleFuncDecl(lex)) {
            depth++;
        }
        if (depth > 0) lexerNext(lex);
    }
    bodyEnd = lex->start;
    
    int bodyLen = (int)(bodyEnd - bodyStart);
    char* body = malloc(bodyLen + 1);
    strncpy(body, bodyStart, bodyLen);
    body[bodyLen] = '\0';
    
    Module* owner = runtime.currentModule;
    JaiFunction* f = defineFunction(name, params, paramCount, isVariadic, body);
    if (f) {
        strncpy(f->returnType, returnType, MAX_NAME_LEN - 1);
        f->returnType[MAX_NAME_LEN - 1] = '\0';
        f->paramTypes = paramTypes;
    } else {
        for (int i = 0; i < paramCount; i++) {
            free(paramTypes[i]);
        }
        free(paramTypes);
    }
    
    if (f && targetNs) {
        namespaceAddFunction(targetNs, f);
        namespaceSetVariable(targetNs, name, makeFunction(f), returnType);
        if (owner) {
            for (int i = 0; i < owner->funcCount; i++) {
                if (owner->functions[i] == f) {
                    owner->functions[i] = owner->functions[owner->funcCount - 1];
                    owner->functions[owner->funcCount - 1] = NULL;
                    owner->funcCount--;
                    break;
                }
            }
        }
    }
    
    for (int i = 0; i < paramCount; i++) free(params[i]);
    free(params);
    free(body);
    
    lexerExpect(lex, getKW_END());
    
    return makeFunction(f);
}

static Value stmtFunc(Lexer* lex) {
    lexerExpect(lex, getKW_FUNC());
    
    if (!lexerCheck(lex, TK_IDENTIFIER)) {
        runtimeError("Expected function name");
        return makeNull();
    }
    char name[MAX_NAME_LEN];
    strcpy(name, lex->currentToken.strValue);
    lexerNext(lex);
    
    lexerExpect(lex, TK_LPAREN);
    
    int paramCapacity = 8;
    char** params = malloc(sizeof(char*) * paramCapacity);
    int paramCount = 0;
    bool isVariadic = false;
    
    if (!lexerCheck(lex, TK_RPAREN)) {
        do {
            if (lexerMatch(lex, TK_STAR)) {
                isVariadic = true;
            }
            if (!lexerCheck(lex, TK_IDENTIFIER)) {
                runtimeError("Expected parameter name");
                for (int i = 0; i < paramCount; i++) free(params[i]);
                free(params);
                return makeNull();
            }
            if (paramCount >= paramCapacity) {
                paramCapacity *= 2;
                params = realloc(params, sizeof(char*) * paramCapacity);
            }
            params[paramCount++] = strdup(lex->currentToken.strValue);
            lexerNext(lex);
        } while (lexerMatch(lex, TK_COMMA));
    }
    lexerExpect(lex, TK_RPAREN);
    skipNewlines(lex);
    
    const char* bodyStart = lex->start;
    const char* bodyEnd = NULL;
    int depth = 1;
    while (depth > 0 && !lexerCheck(lex, TK_EOF)) {
        int kind = lex->currentToken.kind;
        if (kind == getKW_FUNC() || kind == getKW_IF() || kind == getKW_WHILE()) {
            depth++;
        }
        if (kind == getKW_END()) {
            depth--;
        }
        if (depth > 0) lexerNext(lex);
    }
    bodyEnd = lex->start;
    
    int bodyLen = (int)(bodyEnd - bodyStart);
    char* body = malloc(bodyLen + 1);
    strncpy(body, bodyStart, bodyLen);
    body[bodyLen] = '\0';
    
    JaiFunction* f = defineFunction(name, params, paramCount, isVariadic, body);
    
    for (int i = 0; i < paramCount; i++) free(params[i]);
    free(params);
    free(body);
    
    lexerExpect(lex, getKW_END());
    
    if (runtime.debug) {
        printf("Defined function %s with %d params%s\n", name, paramCount, isVariadic ? " (variadic)" : "");
    }
    
    return makeFunction(f);
}


static Value stmtClass(Lexer* lex) {
    lexerExpect(lex, getKW_CLASS());
    
    if (!lexerCheck(lex, TK_IDENTIFIER)) {
        runtimeError("Expected class name");
        return makeNull();
    }
    char className[MAX_NAME_LEN];
    strcpy(className, lex->currentToken.strValue);
    lexerNext(lex);
    
    JaiClass* parent = NULL;
    if (lexerMatch(lex, getKW_EXTENDS())) {
        if (!lexerCheck(lex, TK_IDENTIFIER)) {
            runtimeError("Expected parent class name");
            return makeNull();
        }
        parent = findClass(lex->currentToken.strValue);
        if (!parent) {
            runtimeError("Parent class not found: %s", lex->currentToken.strValue);
            return makeNull();
        }
        lexerNext(lex);
    }
    
    skipNewlines(lex);
    
    JaiClass* cls = defineClass(className, parent);
    
    while (!lexerCheck(lex, getKW_END()) && !lexerCheck(lex, TK_EOF)) {
        skipNewlines(lex);
        
        if (lexerCheck(lex, getKW_END())) break;
        
        if (lexerMatch(lex, getKW_VAR())) {
            if (!lexerCheck(lex, TK_IDENTIFIER)) {
                runtimeError("Expected field name");
                return makeNull();
            }
            classAddField(cls, lex->currentToken.strValue);
            lexerNext(lex);
            skipNewlines(lex);
            continue;
        }
        
        if (lexerCheck(lex, getKW_FUNC())) {
            lexerNext(lex);
            
            if (!lexerCheck(lex, TK_IDENTIFIER)) {
                runtimeError("Expected method name");
                return makeNull();
            }
            char methodName[MAX_NAME_LEN];
            strcpy(methodName, lex->currentToken.strValue);
            lexerNext(lex);
            
            lexerExpect(lex, TK_LPAREN);
            
            int paramCapacity = 8;
            char** params = malloc(sizeof(char*) * paramCapacity);
            int paramCount = 0;
            
            if (!lexerCheck(lex, TK_RPAREN)) {
                do {
                    if (!lexerCheck(lex, TK_IDENTIFIER) && lex->currentToken.kind != getKW_SELF()) {
                        runtimeError("Expected parameter name");
                        for (int i = 0; i < paramCount; i++) free(params[i]);
                        free(params);
                        return makeNull();
                    }
                    if (paramCount >= paramCapacity) {
                        paramCapacity *= 2;
                        params = realloc(params, sizeof(char*) * paramCapacity);
                    }
                    if (lex->currentToken.kind == getKW_SELF()) {
                        params[paramCount++] = strdup("self");
                    } else {
                        params[paramCount++] = strdup(lex->currentToken.strValue);
                    }
                    lexerNext(lex);
                } while (lexerMatch(lex, TK_COMMA));
            }
            lexerExpect(lex, TK_RPAREN);
            skipNewlines(lex);
            
            const char* bodyStart = lex->start;
            const char* bodyEnd = NULL;
            int depth = 1;
            while (depth > 0 && !lexerCheck(lex, TK_EOF)) {
                int kind = lex->currentToken.kind;
                if (kind == getKW_FUNC() || kind == getKW_IF() || kind == getKW_WHILE() || kind == getKW_CLASS()) {
                    depth++;
                }
                if (kind == getKW_END()) {
                    depth--;
                }
                if (depth > 0) lexerNext(lex);
            }
            bodyEnd = lex->start;
            
            int bodyLen = (int)(bodyEnd - bodyStart);
            char* body = malloc(bodyLen + 1);
            strncpy(body, bodyStart, bodyLen);
            body[bodyLen] = '\0';
            
            
            JaiFunction* method = defineFunction(methodName, params, paramCount, false, body);
            classAddMethod(cls, methodName, method);
            
            for (int i = 0; i < paramCount; i++) free(params[i]);
            free(params);
            free(body);
            
            lexerExpect(lex, getKW_END());
            skipNewlines(lex);
            continue;
        }
        
        lexerNext(lex);
    }
    
    lexerExpect(lex, getKW_END());
    
    if (runtime.debug) {
        printf("Defined class %s with %d fields, %d methods\n", 
               className, cls->fieldCount, cls->methodCount);
    }
    
    return makeNull();
}

static Value stmtNamespace(Lexer* lex) {
    lexerExpect(lex, getKW_NAMESPACE());
    
    if (!lexerCheck(lex, TK_IDENTIFIER)) {
        runtimeError("Expected namespace name");
        return makeNull();
    }
    
    char nsName[MAX_NAME_LEN];
    strcpy(nsName, lex->currentToken.strValue);
    lexerNext(lex);
    
    skipNewlines(lex);
    
    Value nsVal = makeNamespace(nsName);
    JaiNamespace* ns = nsVal.as.namespace;
    
    Module* caller = runtime.currentModule;
    Module* tempMod = createModule(nsName, caller->path);
    runtime.currentModule = tempMod;
    
    while (!lexerCheck(lex, getKW_END()) && !lexerCheck(lex, TK_EOF)) {
        parseStatement(lex);
        skipNewlines(lex);
    }
    
    lexerExpect(lex, getKW_END());
    
    ns->varCount = tempMod->varCount;
    if (ns->varCount > ns->varCapacity) {
        ns->varCapacity = ns->varCount;
        ns->variables = realloc(ns->variables, sizeof(Variable) * ns->varCapacity);
    }
    
    for(int i=0; i<tempMod->varCount; i++) {
        ns->variables[i] = tempMod->variables[i];
        tempMod->variables[i].value = makeNull();
    }
    
    ns->funcCount = tempMod->funcCount;
    if (ns->funcCount > ns->funcCapacity) {
        ns->funcCapacity = ns->funcCount;
        ns->functions = realloc(ns->functions, sizeof(JaiFunction*) * ns->funcCapacity);
    }
    
    for(int i=0; i<tempMod->funcCount; i++) {
        ns->functions[i] = tempMod->functions[i];
        ns->functions[i]->namespace = ns;
        tempMod->functions[i] = NULL;
    }
    
    runtime.currentModule = caller;
    setVariable(nsName, nsVal);
    
    return makeNull();
}

static Value stmtReturn(Lexer* lex) {
    lexerExpect(lex, getKW_RETURN());
    
    if (!lexerCheck(lex, TK_NEWLINE) && !lexerCheck(lex, TK_EOF)) {
        returnValue = parseExpression(lex);
    } else {
        returnValue = makeNull();
    }
    hasReturn = true;
    return returnValue;
}

static bool isNameToken(Lexer* lex) {
    int k = lex->currentToken.kind;
    return k == TK_IDENTIFIER || k >= TK_KEYWORD;
}

static bool pathExists(const char* path) {
    struct stat st;
    return stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

static bool findModuleRecursive(const char* base, const char* targetFile, char* out, size_t outSize, int depth) {
    if (depth > 4) return false;
    DIR* dir = opendir(base);
    if (!dir) return false;
    struct dirent* ent;
    bool found = false;
    while (!found && (ent = readdir(dir))) {
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) continue;
        char candidate[1024];
        snprintf(candidate, sizeof(candidate), "%s/%s", base, ent->d_name);
        struct stat st;
        if (stat(candidate, &st) != 0) continue;
        if (S_ISDIR(st.st_mode)) {
            found = findModuleRecursive(candidate, targetFile, out, outSize, depth + 1);
        } else if (S_ISREG(st.st_mode)) {
            if (strcmp(ent->d_name, targetFile) == 0) {
                strncpy(out, candidate, outSize - 1);
                out[outSize - 1] = '\0';
                found = true;
            }
        }
    }
    closedir(dir);
    return found;
}

static bool resolveModulePath(const char* modulePath, char* outPath, size_t outSize) {
    static bool importDebug = false;
    static bool importDebugChecked = false;
    if (!importDebugChecked) {
        const char* env = getenv("JAITHON_IMPORT_DEBUG");
        if (env && (strcmp(env, "1") == 0 || strcasecmp(env, "true") == 0)) {
            importDebug = true;
        }
        importDebugChecked = true;
    }
    if (importDebug) {
        fprintf(stderr, "[IMPORT] resolving %s (execDir=%s)\n", modulePath, gExecDir);
    }
    const char* envLib = getenv("JAITHON_LIB");
    const char* envBase = (envLib && strlen(envLib) > 0) ? envLib : "";
    const char* execBase = strlen(gExecDir) > 0 ? gExecDir : "";
    const char* bases[] = {
        "",
        envBase,
        execBase,
        "lib/modules",
        "/usr/local/share/jaithon",
        "/usr/local/lib/jaithon",
        "/Library/Jaithon",
        "/opt/homebrew/share/jaithon",
        NULL
    };
    
    char targetFile[512];
    snprintf(targetFile, sizeof(targetFile), "%s.jai", modulePath);
    char targetShort[512];
    const char* lastSlash = strrchr(modulePath, '/');
    const char* tail = lastSlash ? lastSlash + 1 : modulePath;
    snprintf(targetShort, sizeof(targetShort), "%s.jai", tail);
    bool hasSlash = strchr(modulePath, '/') != NULL;
    
    for (int i = 0; bases[i]; i++) {
        const char* base = bases[i];
        if (!base) continue;
        bool baseEmpty = strlen(base) == 0;
        const char* altBase = base;
        char altBuf[1024];
        if (!baseEmpty && base[0] != '/' && strlen(gExecDir) > 0) {
            snprintf(altBuf, sizeof(altBuf), "%s/%s", gExecDir, base);
            altBase = altBuf;
        }
        if (importDebug) {
            fprintf(stderr, "[IMPORT] base='%s' alt='%s' module='%s'\n", base, altBase, modulePath);
        }
        
        if (hasSlash) {
            if (baseEmpty) {
                if (snprintf(outPath, outSize, "%s.jai", modulePath) < (int)outSize && pathExists(outPath)) {
                    return true;
                }
                if (snprintf(outPath, outSize, "%s/index.jai", modulePath) < (int)outSize && pathExists(outPath)) {
                    return true;
                }
            } else {
                if (snprintf(outPath, outSize, "%s/%s.jai", base, modulePath) < (int)outSize &&
                    pathExists(outPath)) {
                    if (importDebug) fprintf(stderr, "[IMPORT] %s -> %s\n", modulePath, outPath);
                    return true;
                }
                if (snprintf(outPath, outSize, "%s/%s/index.jai", base, modulePath) < (int)outSize &&
                    pathExists(outPath)) {
                    if (importDebug) fprintf(stderr, "[IMPORT] %s -> %s\n", modulePath, outPath);
                    return true;
                }
                if (snprintf(outPath, outSize, "%s/%s.jai", altBase, modulePath) < (int)outSize &&
                    pathExists(outPath)) {
                    if (importDebug) fprintf(stderr, "[IMPORT] %s -> %s\n", modulePath, outPath);
                    return true;
                }
                if (snprintf(outPath, outSize, "%s/%s/index.jai", altBase, modulePath) < (int)outSize &&
                    pathExists(outPath)) {
                    if (importDebug) fprintf(stderr, "[IMPORT] %s -> %s\n", modulePath, outPath);
                    return true;
                }
            }
        } else {
            if (baseEmpty) {
                if (snprintf(outPath, outSize, "%s", targetFile) < (int)outSize && pathExists(outPath)) {
                    if (importDebug) fprintf(stderr, "[IMPORT] %s -> %s\n", modulePath, outPath);
                    return true;
                }
                if (snprintf(outPath, outSize, "%s/index.jai", modulePath) < (int)outSize && pathExists(outPath)) {
                    if (importDebug) fprintf(stderr, "[IMPORT] %s -> %s\n", modulePath, outPath);
                    return true;
                }
                if (findModuleRecursive(".", targetFile, outPath, outSize, 0)) {
                    if (importDebug) fprintf(stderr, "[IMPORT] %s -> %s\n", modulePath, outPath);
                    return true;
                }
            } else {
                if (snprintf(outPath, outSize, "%s/%s", base, targetFile) < (int)outSize &&
                    pathExists(outPath)) {
                    if (importDebug) fprintf(stderr, "[IMPORT] %s -> %s\n", modulePath, outPath);
                    return true;
                }
                if (snprintf(outPath, outSize, "%s/%s/index.jai", base, modulePath) < (int)outSize &&
                    pathExists(outPath)) {
                    if (importDebug) fprintf(stderr, "[IMPORT] %s -> %s\n", modulePath, outPath);
                    return true;
                }
                if (findModuleRecursive(base, targetFile, outPath, outSize, 0)) {
                    if (importDebug) fprintf(stderr, "[IMPORT] %s -> %s\n", modulePath, outPath);
                    return true;
                }
                if (snprintf(outPath, outSize, "%s/%s", altBase, targetFile) < (int)outSize &&
                    pathExists(outPath)) {
                    if (importDebug) fprintf(stderr, "[IMPORT] %s -> %s\n", modulePath, outPath);
                    return true;
                }
                if (snprintf(outPath, outSize, "%s/%s/index.jai", altBase, modulePath) < (int)outSize &&
                    pathExists(outPath)) {
                    if (importDebug) fprintf(stderr, "[IMPORT] %s -> %s\n", modulePath, outPath);
                    return true;
                }
                if (findModuleRecursive(altBase, targetFile, outPath, outSize, 0)) {
                    if (importDebug) fprintf(stderr, "[IMPORT] %s -> %s\n", modulePath, outPath);
                    return true;
                }
            }
        }
    }
    if (strlen(gExecDir) > 0) {
        char start[1024];
        snprintf(start, sizeof(start), "%s/lib/modules", gExecDir);
        if (findModuleRecursive(start, hasSlash ? targetShort : targetFile, outPath, outSize, 0)) {
            if (importDebug) fprintf(stderr, "[IMPORT] %s -> %s\n", modulePath, outPath);
            return true;
        }
        if (findModuleRecursive(gExecDir, hasSlash ? targetShort : targetFile, outPath, outSize, 0)) {
            if (importDebug) fprintf(stderr, "[IMPORT] %s -> %s\n", modulePath, outPath);
            return true;
        }
    }
    if (importDebug) fprintf(stderr, "[IMPORT] %s -> not found\n", modulePath);
    
    return false;
}

static Value stmtImport(Lexer* lex) {
    lexerExpect(lex, getKW_IMPORT());
    
    if (!isNameToken(lex)) {
        runtimeError("Expected module name");
        return makeNull();
    }
    
    char modulePath[1024] = {0};
    size_t len = 0;
    
    while (true) {
        const char* segment = lex->currentToken.strValue;
        size_t segLen = strlen(segment);
        
        if (len + (len > 0) + segLen + 1 >= sizeof(modulePath)) {
            runtimeError("Module path too long");
            return makeNull();
        }
        
        if (len > 0) {
            modulePath[len++] = '/';
        }
        memcpy(modulePath + len, segment, segLen);
        len += segLen;
        modulePath[len] = '\0';
        
        lexerNext(lex);
        if (!lexerMatch(lex, TK_SLASH)) break;
        if (!isNameToken(lex)) {
            runtimeError("Expected module name after '/'");
            return makeNull();
        }
    }
    
    char path[1024];
    if (!resolveModulePath(modulePath, path, sizeof(path))) {
        runtimeError("Cannot open module: %s.jai", modulePath);
        return makeNull();
    }
    FILE* f = fopen(path, "r");
    if (!f) {
        runtimeError("Cannot open module: %s", path);
        return makeNull();
    }
    
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    char* code = malloc(size + 1);
    fread(code, 1, size, f);
    code[size] = '\0';
    fclose(f);
    


    Module* caller = runtime.currentModule;
    Module* newMod = createModule(modulePath, path);
    runtime.currentModule = newMod;
    
    Lexer modLex;
    lexerInit(&modLex, code);
    parseProgram(&modLex);
    if (eagerCompileEnabled()) {
        compileModuleFunctions(newMod, eagerCompileStrict());
    }
    
    free(code);
    
    runtime.currentModule = caller;
    
    for (int i = 0; i < newMod->varCount; i++) {
        bool exists = false;
        for (int j = 0; j < caller->varCount; j++) {
            if (strcmp(caller->variables[j].name, newMod->variables[i].name) == 0) {
                exists = true;
                break;
            }
        }
        if (!exists) {
            setVariable(newMod->variables[i].name, newMod->variables[i].value);
        }
    }
    
    for (int i = 0; i < newMod->funcCount; i++) {
        JaiFunction* f = newMod->functions[i];
        if (!f) continue;
        if (!hasVariable(f->name)) {
            setVariable(f->name, makeFunction(f));
        }
    }
    
    return makeNull();
}

static Value stmtInput(Lexer* lex) {
    lexerExpect(lex, getKW_INPUT());
    
    if (!lexerCheck(lex, TK_IDENTIFIER)) {
        runtimeError("Expected variable name");
        return makeNull();
    }
    char name[MAX_NAME_LEN];
    strcpy(name, lex->currentToken.strValue);
    lexerNext(lex);
    
    printf("Enter a value for %s: ", name);
    char buf[1024];
    if (fgets(buf, sizeof(buf), stdin)) {
        buf[strcspn(buf, "\n")] = 0;
        
        char* end;
        double num = strtod(buf, &end);
        if (*end == '\0' && end != buf) {
            setVariable(name, makeNumber(num));
        } else {
            setVariable(name, makeString(buf));
        }
    }
    
    return getVariable(name);
}

static Value stmtBreak(Lexer* lex) {
    lexerExpect(lex, getKW_BREAK());
    if (!runtime.shellMode) {
        exit(0);
    }
    return makeNull();
}

static Value stmtSystem(Lexer* lex) {
    lexerExpect(lex, getKW_SYSTEM());
    
    if (!lexerCheck(lex, TK_IDENTIFIER) && !lexerCheck(lex, TK_STRING)) {
        runtimeError("Expected command");
        return makeNull();
    }
    
    char cmd[MAX_NAME_LEN];
    strcpy(cmd, lex->currentToken.strValue);
    lexerNext(lex);
    
    int ret = system(cmd);
    return makeNumber(ret);
}

bool eagerCompileEnabled(void) {
    if (!eagerInit) {
        const char* envOff = getenv("JAITHON_NO_EAGER");
        const char* envOn = getenv("JAITHON_EAGER");
        const char* envStrict = getenv("JAITHON_EAGER_STRICT");
        if (envOff && (strcmp(envOff, "1") == 0 || strcasecmp(envOff, "true") == 0)) {
            eagerCompile = false;
        }
        if (envOn && (strcmp(envOn, "1") == 0 || strcasecmp(envOn, "true") == 0)) {
            eagerCompile = true;
        }
        if (envStrict && (strcmp(envStrict, "1") == 0 || strcasecmp(envStrict, "true") == 0)) {
            eagerStrict = true;
        }
        eagerInit = true;
    }
    return eagerCompile;
}

bool eagerCompileStrict(void) {
    eagerCompileEnabled();
    return eagerStrict;
}

bool compileModuleFunctions(Module* mod, bool strict) {
    if (!mod) return true;
    static bool eagerDebug = false;
    static bool debugInit = false;
    if (!debugInit) {
        const char* env = getenv("JAITHON_EAGER_DEBUG");
        if (env && (strcmp(env, "1") == 0 || strcasecmp(env, "true") == 0)) {
            eagerDebug = true;
        }
        debugInit = true;
    }
    for (int i = 0; i < mod->funcCount; i++) {
        JaiFunction* f = mod->functions[i];
        if (!f || !f->body) continue;
        CompiledFunc* c = getCompiledFunc(f);
        if (!c) {
            if (strict) {
                runtimeError("Failed to compile '%s' in module '%s'", f->name, mod->name);
                return false;
            }
            if (eagerDebug) {
                fprintf(stderr, "[EAGER] fallback to interpreter: %s\n", f->name);
            }
        } else if (eagerDebug) {
            fprintf(stderr, "[EAGER] compiled: %s\n", f->name);
        }
    }
    return true;
}

static Value stmtDel(Lexer* lex) {
    lexerExpect(lex, getKW_DEL());
    if (!lexerCheck(lex, TK_IDENTIFIER)) {
        runtimeError("Expected identifier after 'del'");
        return makeNull();
    }
    char name[MAX_NAME_LEN];
    strcpy(name, lex->currentToken.strValue);
    lexerNext(lex);
    
    if (lexerMatch(lex, TK_LBRACKET)) {
        Value index = parseExpression(lex);
        lexerExpect(lex, TK_RBRACKET);
        
        Value arr = getVariable(name);
        if (arr.type != VAL_ARRAY) {
            runtimeError("Cannot delete index of non-array '%s'", name);
            return makeNull();
        }
        arrayDelete(arr.as.array, (int)toNumber(index));
        return makeNull();
    }
    
    if (!deleteVariable(name)) {
        runtimeError("Name '%s' not found for deletion", name);
    }
    return makeNull();
}

Value parseStatement(Lexer* lex) {
    skipNewlines(lex);
    
    if (lexerCheck(lex, TK_EOF)) {
        return makeNull();
    }

    if (runtime.compileOnly && !isDefinitionStart(lex)) {
        skipStatementNoExec(lex);
        return makeNull();
    }

    if (startsWithJavaStyleDecl(lex)) {
        return stmtJavaStyleDecl(lex);
    }
    
    int kind = lex->currentToken.kind;
    StatementHandler handler = findStatement(kind);
    
    if (handler) {
        return handler(lex);
    }
    
    if (kind == TK_IDENTIFIER || kind == getKW_SELF()) {
        char name[MAX_NAME_LEN];
        strcpy(name, lex->currentToken.strValue);
        lexerNext(lex);
        
        if (lexerCheck(lex, TK_DOT) || lexerCheck(lex, TK_LBRACKET)) {
            Value result = getVariable(name);
            
            while (lexerCheck(lex, TK_DOT) || lexerCheck(lex, TK_LBRACKET)) {
                if (lexerMatch(lex, TK_DOT)) {
                    if (!lexerCheck(lex, TK_IDENTIFIER)) {
                        runtimeError("Expected field/method name after '.'");
                        return makeNull();
                    }
                    char fieldName[MAX_NAME_LEN];
                    strcpy(fieldName, lex->currentToken.strValue);
                    lexerNext(lex);
                    
                    if (result.type == VAL_NAMESPACE) {
                        JaiNamespace* ns = result.as.namespace;
                        if (lexerCheck(lex, TK_LPAREN)) {
                             JaiFunction* func = NULL;
                             for(int i=0; i<ns->funcCount; i++) {
                                if(strcmp(ns->functions[i]->name, fieldName) == 0) {
                                    func = ns->functions[i];
                                    break;
                                }
                             }
                             if (!func) {
                                 for(int i=0; i<ns->varCount; i++) {
                                    if(strcmp(ns->variables[i].name, fieldName) == 0) {
                                        if (ns->variables[i].value.type == VAL_FUNCTION) {
                                            func = ns->variables[i].value.as.function;
                                        }
                                        break;
                                    }
                                 }
                             }
                             if (!func) {
                                 runtimeError("Namespace '%s' has no function '%s'", ns->name, fieldName);
                                 return makeNull();
                             }
                             lexerNext(lex);
                             Value args[MAX_CALL_ARGS];
                             int argc = 0;
                             if (!lexerCheck(lex, TK_RPAREN)) {
                                do {
                                    args[argc++] = parseExpression(lex);
                                } while (lexerMatch(lex, TK_COMMA));
                             }
                             lexerExpect(lex, TK_RPAREN);
                             result = callValue(makeFunction(func), args, argc);
                        }
                        else if (lexerCheck(lex, TK_EQUALS)) {
                            lexerNext(lex);
                            Value val = parseExpression(lex);
                            bool found = false;
                            for(int i=0; i<ns->varCount; i++) {
                                if(strcmp(ns->variables[i].name, fieldName) == 0) {
                                    ns->variables[i].value = val;
                                    found = true;
                                    break;
                                }
                            }
                            if (!found) {
                                if (ns->varCount >= ns->varCapacity) {
                                    ns->varCapacity *= 2;
                                    ns->variables = realloc(ns->variables, sizeof(Variable) * ns->varCapacity);
                                }
                                strcpy(ns->variables[ns->varCount].name, fieldName);
                                ns->variables[ns->varCount].value = val;
                                ns->varCount++;
                            }
                            return val;
                        }
                        else {
                            bool found = false;
                            for(int i=0; i<ns->varCount; i++) {
                                if(strcmp(ns->variables[i].name, fieldName) == 0) {
                                    result = ns->variables[i].value;
                                    found = true;
                                    break;
                                }
                            }
                            if (!found) {
                                for(int i=0; i<ns->funcCount; i++) {
                                    if(strcmp(ns->functions[i]->name, fieldName) == 0) {
                                        result = makeFunction(ns->functions[i]);
                                        found = true;
                                        break;
                                    }
                                }
                            }
                            if (!found) {
                                runtimeError("Namespace '%s' has no member '%s'", ns->name, fieldName);
                                return makeNull();
                            }
                        }
                        continue;
                    }

                    if (result.type != VAL_OBJECT) {
                        runtimeError("Cannot access field '%s' of non-object", fieldName);
                        return makeNull();
                    }
                    
                    if (lexerCheck(lex, TK_LPAREN)) {
                        JaiFunction* method = objectGetMethod(result.as.object, fieldName);
                        if (!method) {
                            runtimeError("Object has no method: %s", fieldName);
                            return makeNull();
                        }
                        lexerNext(lex);
                        Value args[MAX_CALL_ARGS];
                        int argc = 0;
                        if (methodExpectsSelf(method)) {
                            args[argc++] = result;
                        }
                        if (!lexerCheck(lex, TK_RPAREN)) {
                            do {
                                args[argc++] = parseExpression(lex);
                            } while (lexerMatch(lex, TK_COMMA));
                        }
                        lexerExpect(lex, TK_RPAREN);
                        result = callValue(makeFunction(method), args, argc);
                    }
                    else if (lexerCheck(lex, TK_EQUALS)) {
                        lexerNext(lex);
                        Value val = parseExpression(lex);
                        objectSetField(result.as.object, fieldName, val);
                        return val;
                    }
                    else {
                        result = objectGetField(result.as.object, fieldName);
                    }
                } else if (lexerMatch(lex, TK_LBRACKET)) {
                    Value index = parseExpression(lex);
                    lexerExpect(lex, TK_RBRACKET);
                    
                    if (lexerCheck(lex, TK_EQUALS)) {
                        lexerNext(lex);
                        Value val = parseExpression(lex);
                        if (result.type == VAL_ARRAY) {
                            arraySet(result.as.array, (int)toNumber(index), val);
                        }
                        return val;
                    }
                    
                    if (result.type == VAL_ARRAY) {
                        result = arrayGet(result.as.array, (int)toNumber(index));
                    }
                }
            }
            return result;
        }
        
        if (lexerCheck(lex, TK_EQUALS)) {
            lexerNext(lex);
            Value val = parseExpression(lex);
            setVariable(name, val);
            return val;
        }
        
        if (lexerCheck(lex, TK_LPAREN)) {
            lexerNext(lex);
            Value args[MAX_CALL_ARGS];
            int argc = 0;
            
            if (!lexerCheck(lex, TK_RPAREN)) {
                do {
                    args[argc++] = parseExpression(lex);
                } while (lexerMatch(lex, TK_COMMA));
            }
            lexerExpect(lex, TK_RPAREN);
            
            Value callee = getVariable(name);
            return callValue(callee, args, argc);
        }
        
        runtimeError("Unexpected identifier: %s", name);
        return makeNull();
    }
    
    runtimeError("Unexpected token: %s", tokenKindName(kind));
    lexerNext(lex);
    return makeNull();
}

Value parseProgram(Lexer* lex) {
    Value last = makeNull();
    
    while (!lexerCheck(lex, TK_EOF)) {
        last = parseStatement(lex);
        skipNewlines(lex);
        if (hasReturn) break;
    }
    
    return last;
}

Value callValue(Value callee, Value* args, int argc) {
    static int callDepth = 0;
    callDepth++;
    if (callDepth > MAX_CALL_STACK) {
        callDepth--;
        runtimeError("Call stack overflow");
        return makeNull();
    }

    if (callee.type == VAL_NATIVE_FUNC) {
        Value r = callee.as.nativeFunc(args, argc);
        callDepth--;
        return r;
    }
    
    if (callee.type == VAL_FUNCTION) {
        JaiFunction* f = callee.as.function;
        
        if (runtime.callStackSize < MAX_CALL_STACK) {
            strncpy(runtime.callStack[runtime.callStackSize++], f->name, MAX_NAME_LEN - 1);
        }
        
        if (f->isVariadic) {
            int minArgs = f->paramCount - 1;
            if (argc < minArgs) {
                runtimeError("Expected at least %d arguments, got %d", minArgs, argc);
                callDepth--;
                return makeNull();
            }
        } else if (argc != f->paramCount) {
            runtimeError("Expected %d arguments, got %d", f->paramCount, argc);
            callDepth--;
            return makeNull();
        }
        
    Module* oldMod = runtime.currentModule;
    
    CompiledFunc* compiled = getCompiledFunc(f);
    if (compiled) {
        statsVMCalls++;
        VM* vm = malloc(sizeof(VM));
        vmInit(vm);
        runtime.currentModule = oldMod;
        for (int i = 0; i < compiled->arity && i < argc; i++) {
            vmPush(vm, args[i]);
        }
        InterpretResult vmResult = vmRun(vm, compiled);
        Value result = vmResult == INTERPRET_OK ? vm->result : makeNull();
        vmFree(vm);
        free(vm);
        
        runtime.currentModule = oldMod;
        if (runtime.callStackSize > 0) runtime.callStackSize--;
        callDepth--;
        
        if (vmResult != INTERPRET_OK) {
            statsVMCalls--;  
            goto INTERPRET_FALLBACK;
        }
        return result;
    }
    
INTERPRET_FALLBACK: ;
    statsInterpretCalls++;
    Module* funcMod = createModule("__call__", "");
    runtime.currentModule = funcMod;
    
    if (f->namespace) {
        for (int i = 0; i < f->namespace->varCount; i++) {
            setVariable(f->namespace->variables[i].name, f->namespace->variables[i].value);
        }
        for (int i = 0; i < f->namespace->funcCount; i++) {
            if (!hasVariable(f->namespace->functions[i]->name)) {
                setVariable(f->namespace->functions[i]->name, makeFunction(f->namespace->functions[i]));
            }
        }
    }
    
    if (f->isVariadic) {
        int regularParams = f->paramCount - 1;
        for (int i = 0; i < regularParams; i++) {
            const char* tname = (f->paramTypes && f->paramTypes[i]) ? f->paramTypes[i] : "";
            setTypedVariable(f->params[i], args[i], tname);
        }
        Value variadicArray = makeArray(argc - regularParams);
        for (int i = regularParams; i < argc; i++) {
            arrayPush(variadicArray.as.array, args[i]);
        }
        setTypedVariable(f->params[regularParams], variadicArray, "var");
    } else {
        for (int i = 0; i < argc; i++) {
            const char* tname = (f->paramTypes && f->paramTypes[i]) ? f->paramTypes[i] : "";
            setTypedVariable(f->params[i], args[i], tname);
        }
    }
    
    hasReturn = false;
    Value result = makeNull();
    
    Lexer bodyLex;
    lexerInit(&bodyLex, f->body);
    result = parseProgram(&bodyLex);
    
    if (hasReturn) {
        result = returnValue;
        hasReturn = false;
    }
    
    if (f->namespace) {
        for (int i = 0; i < f->namespace->varCount; i++) {
            for (int j = 0; j < funcMod->varCount; j++) {
                if (strcmp(funcMod->variables[j].name, f->namespace->variables[i].name) == 0) {
                    f->namespace->variables[i].value = funcMod->variables[j].value;
                    break;
                }
            }
        }
    }
    
    runtime.currentModule = oldMod;
    runtime.moduleCount--;
    
    if (runtime.callStackSize > 0) {
        runtime.callStackSize--;
    }
    
    callDepth--;
    return result;
}
    
    runtimeError("Cannot call non-function value");
    callDepth--;
    return makeNull();
}

static Value nativeSin(Value* args, int argc) {
    if (argc < 1) return makeNull();
    return makeNumber(sin(toNumber(args[0])));
}

static Value nativeCos(Value* args, int argc) {
    if (argc < 1) return makeNull();
    return makeNumber(cos(toNumber(args[0])));
}

static Value nativeTan(Value* args, int argc) {
    if (argc < 1) return makeNull();
    return makeNumber(tan(toNumber(args[0])));
}

static Value nativeSqrt(Value* args, int argc) {
    if (argc < 1) return makeNull();
    return makeNumber(sqrt(toNumber(args[0])));
}

static Value nativeLog(Value* args, int argc) {
    if (argc < 1) return makeNull();
    return makeNumber(log(toNumber(args[0])));
}

static Value nativeExp(Value* args, int argc) {
    if (argc < 1) return makeNull();
    return makeNumber(exp(toNumber(args[0])));
}

static Value nativeTime(Value* args, int argc) {
    (void)args; (void)argc;
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return makeNumber(tv.tv_sec + tv.tv_usec / 1000000.0);
}

static Value nativeRand(Value* args, int argc) {
    (void)args; (void)argc;
    return makeNumber((double)rand() / RAND_MAX);
}

static Value nativeLen(Value* args, int argc) {
    if (argc < 1) return makeNumber(0);
    if (args[0].type == VAL_STRING) {
        return makeNumber(strlen(args[0].as.string));
    }
    return makeNumber(0);
}

static Value nativeStr(Value* args, int argc) {
    if (argc < 1) return makeString("");
    char buf[256];
    switch (args[0].type) {
        case VAL_NUMBER: snprintf(buf, sizeof(buf), "%g", args[0].as.number); break;
        case VAL_DOUBLE: snprintf(buf, sizeof(buf), "%g", args[0].as.f64); break;
        case VAL_FLOAT: snprintf(buf, sizeof(buf), "%g", args[0].as.f32); break;
        case VAL_INT: snprintf(buf, sizeof(buf), "%d", args[0].as.i32); break;
        case VAL_LONG: snprintf(buf, sizeof(buf), "%lld", (long long)args[0].as.i64); break;
        case VAL_SHORT: snprintf(buf, sizeof(buf), "%d", args[0].as.i16); break;
        case VAL_BYTE: snprintf(buf, sizeof(buf), "%d", args[0].as.i8); break;
        case VAL_CHAR: snprintf(buf, sizeof(buf), "%c", args[0].as.ch); break;
        case VAL_BOOL: snprintf(buf, sizeof(buf), "%d", args[0].as.boolean ? 1 : 0); break;
        case VAL_STRING: return args[0];
        default: strcpy(buf, "null");
    }
    return makeString(buf);
}

static Value nativeNum(Value* args, int argc) {
    if (argc < 1) return makeNumber(0);
    if (args[0].type == VAL_STRING) {
        return makeNumber(strtod(args[0].as.string, NULL));
    }
    return makeNumber(toNumber(args[0]));
}


static Value nativeInt(Value* args, int argc) {
    if (argc < 1) return makeInt(0);
    if (args[0].type == VAL_STRING) {
        return makeInt((int32_t)strtol(args[0].as.string, NULL, 10));
    }
    return makeInt((int32_t)toNumber(args[0]));
}

static Value nativeFloat(Value* args, int argc) {
    if (argc < 1) return makeFloat(0.0f);
    if (args[0].type == VAL_STRING) {
        return makeFloat((float)strtod(args[0].as.string, NULL));
    }
    return makeFloat((float)toNumber(args[0]));
}

static Value nativeDouble(Value* args, int argc) {
    if (argc < 1) return makeDouble(0.0);
    if (args[0].type == VAL_STRING) {
        return makeDouble(strtod(args[0].as.string, NULL));
    }
    return makeDouble(toNumber(args[0]));
}

static Value nativeBoolCast(Value* args, int argc) {
    if (argc < 1) return makeBool(false);
    return makeBool(toBool(args[0]));
}

static Value nativeCharCast(Value* args, int argc) {
    if (argc < 1) return makeChar('\0');
    if (args[0].type == VAL_STRING && args[0].as.string && strlen(args[0].as.string) > 0) {
        return makeChar(args[0].as.string[0]);
    }
    return makeChar((char)(int)toNumber(args[0]));
}

static Value nativeLong(Value* args, int argc) {
    if (argc < 1) return makeLong(0);
    if (args[0].type == VAL_STRING) {
        return makeLong((int64_t)strtoll(args[0].as.string, NULL, 10));
    }
    return makeLong((int64_t)toNumber(args[0]));
}

static Value nativeShort(Value* args, int argc) {
    if (argc < 1) return makeShort(0);
    if (args[0].type == VAL_STRING) {
        return makeShort((int16_t)strtol(args[0].as.string, NULL, 10));
    }
    return makeShort((int16_t)toNumber(args[0]));
}

static Value nativeByteCast(Value* args, int argc) {
    if (argc < 1) return makeByte(0);
    if (args[0].type == VAL_STRING) {
        return makeByte((int8_t)strtol(args[0].as.string, NULL, 10));
    }
    return makeByte((int8_t)toNumber(args[0]));
}

static Value nativeType(Value* args, int argc) {
    if (argc < 1) return makeString("null");
    switch (args[0].type) {
        case VAL_NUMBER: return makeString("number");
        case VAL_DOUBLE: return makeString("double");
        case VAL_FLOAT: return makeString("float");
        case VAL_INT: return makeString("int");
        case VAL_LONG: return makeString("long");
        case VAL_SHORT: return makeString("short");
        case VAL_BYTE: return makeString("byte");
        case VAL_CHAR: return makeString("char");
        case VAL_STRING: return makeString("string");
        case VAL_BOOL: return makeString("bool");
        case VAL_FUNCTION: return makeString("function");
        case VAL_NATIVE_FUNC: return makeString("native");
        case VAL_CELL: return makeString("cell");
        case VAL_FILE: return makeString("file");
        case VAL_ARRAY: return makeString("array");
        case VAL_OBJECT: return makeString("object");
        case VAL_NAMESPACE: return makeString("namespace");
        default: return makeString("null");
    }
}

static Value nativeCell(Value* args, int argc) {
    Value cell = makeCell();
    if (argc > 0) {
        *cell.as.cell->car = args[0];
    }
    if (argc > 1) {
        *cell.as.cell->cdr = args[1];
    }
    return cell;
}

static Value nativeCar(Value* args, int argc) {
    if (argc < 1) return makeNull();
    if (args[0].type != VAL_CELL) return makeNull();
    return *args[0].as.cell->car;
}

static Value nativeCdr(Value* args, int argc) {
    if (argc < 1) return makeNull();
    if (args[0].type != VAL_CELL) return makeNull();
    return *args[0].as.cell->cdr;
}

static Value nativeSetCar(Value* args, int argc) {
    if (argc < 2) return makeNull();
    if (args[0].type != VAL_CELL) return makeNull();
    *args[0].as.cell->car = args[1];
    return args[1];
}

static Value nativeSetCdr(Value* args, int argc) {
    if (argc < 2) return makeNull();
    if (args[0].type != VAL_CELL) return makeNull();
    *args[0].as.cell->cdr = args[1];
    return args[1];
}

static Value nativeCharAt(Value* args, int argc) {
    if (argc < 2) return makeString("");
    if (args[0].type != VAL_STRING) return makeString("");
    int idx = (int)toNumber(args[1]);
    const char* s = args[0].as.string;
    int slen = strlen(s);
    if (idx < 0 || idx >= slen) return makeString("");
    char buf[2] = {s[idx], '\0'};
    return makeString(buf);
}

static Value nativeSubstr(Value* args, int argc) {
    if (argc < 3) return makeString("");
    if (args[0].type != VAL_STRING) return makeString("");
    const char* s = args[0].as.string;
    int slen = strlen(s);
    int start = (int)toNumber(args[1]);
    int count = (int)toNumber(args[2]);
    if (start < 0) start = 0;
    if (start >= slen) return makeString("");
    if (start + count > slen) count = slen - start;
    char* buf = malloc(count + 1);
    strncpy(buf, s + start, count);
    buf[count] = '\0';
    Value result = makeString(buf);
    free(buf);
    return result;
}

static Value nativeConcat(Value* args, int argc) {
    if (argc < 2) return makeString("");
    char buf[4096] = "";
    for (int i = 0; i < argc; i++) {
        if (args[i].type == VAL_STRING) {
            strncat(buf, args[i].as.string, sizeof(buf) - strlen(buf) - 1);
        }
    }
    return makeString(buf);
}

static Value nativeArray(Value* args, int argc) {
    int initialSize = argc > 0 ? (int)toNumber(args[0]) : 0;
    Value arr = makeArray(initialSize > 0 ? initialSize : INITIAL_CAPACITY);
    return arr;
}

static Value nativePush(Value* args, int argc) {
    if (argc < 2 || args[0].type != VAL_ARRAY) {
        runtimeError("_push requires array and value");
        return makeNull();
    }
    arrayPush(args[0].as.array, args[1]);
    return args[1];
}

static Value nativePop(Value* args, int argc) {
    if (argc < 1 || args[0].type != VAL_ARRAY) {
        runtimeError("_pop requires array");
        return makeNull();
    }
    return arrayPop(args[0].as.array);
}

static Value nativeGet(Value* args, int argc) {
    if (argc < 2 || args[0].type != VAL_ARRAY) {
        runtimeError("_get requires array and index");
        return makeNull();
    }
    return arrayGet(args[0].as.array, (int)toNumber(args[1]));
}

static Value nativeSet(Value* args, int argc) {
    if (argc < 3 || args[0].type != VAL_ARRAY) {
        runtimeError("_set requires array, index, and value");
        return makeNull();
    }
    arraySet(args[0].as.array, (int)toNumber(args[1]), args[2]);
    return args[2];
}

static Value nativeAlen(Value* args, int argc) {
    if (argc < 1 || args[0].type != VAL_ARRAY) {
        return makeNumber(0);
    }
    return makeNumber(arrayLen(args[0].as.array));
}

static Value nativeFopen(Value* args, int argc) {
    if (argc < 2) return makeNull();
    if (args[0].type != VAL_STRING || args[1].type != VAL_STRING) return makeNull();
    FILE* f = fopen(args[0].as.string, args[1].as.string);
    if (!f) return makeNull();
    return makeFile(f);
}

static Value nativeFclose(Value* args, int argc) {
    if (argc < 1 || args[0].type != VAL_FILE) return makeNull();
    if (args[0].as.file) {
        fclose(args[0].as.file);
        args[0].as.file = NULL;
    }
    return makeNull();
}

static Value nativeFread(Value* args, int argc) {
    if (argc < 1 || args[0].type != VAL_FILE) return makeString("");
    FILE* f = args[0].as.file;
    if (!f) return makeString("");
    
    fseek(f, 0, SEEK_END);
    long length = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    char* buffer = malloc(length + 1);
    if (!buffer) return makeString("");
    
    size_t read = fread(buffer, 1, length, f);
    buffer[read] = '\0';
    
    Value v = makeString(buffer);
    free(buffer);
    return v;
}

static Value nativeFwrite(Value* args, int argc) {
    if (argc < 2 || args[0].type != VAL_FILE || args[1].type != VAL_STRING) return makeNumber(0);
    FILE* f = args[0].as.file;
    if (!f) return makeNumber(0);
    
    int written = fprintf(f, "%s", args[1].as.string);
    return makeNumber(written);
}

static Value nativeInput(Value* args, int argc) {
    if (argc > 0 && args[0].type == VAL_STRING) {
        printf("%s", args[0].as.string);
    }
    
    char buf[1024];
    if (fgets(buf, sizeof(buf), stdin)) {
        buf[strcspn(buf, "\n")] = 0;
        return makeString(buf);
    }
    return makeString("");
}

static Value nativeSystem(Value* args, int argc) {
    if (argc < 1 || args[0].type != VAL_STRING) return makeNumber(-1);
    int ret = system(args[0].as.string);
    return makeNumber(ret);
}

void initParser(void) {
    statementCount = 0;
    infixCount = 0;
    hasReturn = false;
    
    registerStatement(getKW_VAR(), stmtVar);
    registerStatement(getKW_PRINT(), stmtPrint);
    registerStatement(getKW_IF(), stmtIf);
    registerStatement(getKW_WHILE(), stmtWhile);
    registerStatement(getKW_FUNC(), stmtFunc);
    registerStatement(getKW_RETURN(), stmtReturn);
    registerStatement(getKW_IMPORT(), stmtImport);
    registerStatement(getKW_INPUT(), stmtInput);
    registerStatement(getKW_BREAK(), stmtBreak);
    registerStatement(getKW_SYSTEM(), stmtSystem);
    registerStatement(getKW_DEL(), stmtDel);
    registerStatement(getKW_CLASS(), stmtClass);
    registerStatement(getKW_NAMESPACE(), stmtNamespace);
    registerStatement(getKW_PUBLIC(), stmtJavaStyleDecl);
    registerStatement(getKW_PRIVATE(), stmtJavaStyleDecl);
    registerStatement(getKW_PROTECTED(), stmtJavaStyleDecl);
    registerStatement(getKW_STATIC(), stmtJavaStyleDecl);
    
    registerInfix(TK_PLUS, 6, handleAdd);
    registerInfix(TK_MINUS, 6, handleSub);
    registerInfix(TK_STAR, 7, handleMul);
    registerInfix(TK_SLASH, 7, handleDiv);
    registerInfix(TK_PERCENT, 7, handleMod);
    registerInfix(TK_DOT, 9, handleDot);
    registerInfix(TK_CARET, 8, handlePow);
    registerInfix(TK_BANG, 9, handleFactorial);
    registerInfix(TK_GT, 5, handleGt);
    registerInfix(TK_LT, 5, handleLt);
    registerInfix(TK_GE, 5, handleGe);
    registerInfix(TK_LE, 5, handleLe);
    registerInfix(TK_EQ_EQ, 4, handleEq);
    registerInfix(TK_NE, 4, handleNe);
    registerInfix(getKW_AND(), 3, handleAnd);
    registerInfix(getKW_OR(), 2, handleOr);
    
    srand(time(NULL));
    
    if (!hasVariable("gui_mouse_pos")) {
        setVariable("gui_mouse_pos", makeNativeFunc(stubGuiMousePos));
        setVariable("gui_mouse_down", makeNativeFunc(stubGuiMouseDown));
        setVariable("gui_key_down", makeNativeFunc(stubGuiKeyDown));
        setVariable("gui_poll", makeNativeFunc(stubGuiPoll));
        setVariable("gui_get_keys", makeNativeFunc(stubGuiPoll)); 
    }
    
    setVariable("_sin", makeNativeFunc(nativeSin));
    setVariable("_cos", makeNativeFunc(nativeCos));
    setVariable("_tan", makeNativeFunc(nativeTan));
    setVariable("_sqrt", makeNativeFunc(nativeSqrt));
    setVariable("_log", makeNativeFunc(nativeLog));
    setVariable("_exp", makeNativeFunc(nativeExp));
    setVariable("_time", makeNativeFunc(nativeTime));
    setVariable("_rand", makeNativeFunc(nativeRand));
    setVariable("_len", makeNativeFunc(nativeLen));
    setVariable("_str", makeNativeFunc(nativeStr));
    setVariable("_num", makeNativeFunc(nativeNum));
    setVariable("_int", makeNativeFunc(nativeInt));
    setVariable("_float", makeNativeFunc(nativeFloat));
    setVariable("_double", makeNativeFunc(nativeDouble));
    setVariable("_bool", makeNativeFunc(nativeBoolCast));
    setVariable("_char", makeNativeFunc(nativeCharCast));
    setVariable("_long", makeNativeFunc(nativeLong));
    setVariable("_short", makeNativeFunc(nativeShort));
    setVariable("_byte", makeNativeFunc(nativeByteCast));
    setVariable("_type", makeNativeFunc(nativeType));
    setVariable("_cell", makeNativeFunc(nativeCell));
    setVariable("_car", makeNativeFunc(nativeCar));
    setVariable("_cdr", makeNativeFunc(nativeCdr));
    setVariable("_setcar", makeNativeFunc(nativeSetCar));
    setVariable("_setcdr", makeNativeFunc(nativeSetCdr));
    setVariable("_charAt", makeNativeFunc(nativeCharAt));
    setVariable("_substr", makeNativeFunc(nativeSubstr));
    setVariable("_concat", makeNativeFunc(nativeConcat));
    
    setVariable("_array", makeNativeFunc(nativeArray));
    setVariable("_push", makeNativeFunc(nativePush));
    setVariable("_apush", makeNativeFunc(nativePush));
    setVariable("_pop", makeNativeFunc(nativePop));
    setVariable("_get", makeNativeFunc(nativeGet));
    setVariable("_set", makeNativeFunc(nativeSet));
    setVariable("_alen", makeNativeFunc(nativeAlen));

    setVariable("_fopen", makeNativeFunc(nativeFopen));
    setVariable("_fclose", makeNativeFunc(nativeFclose));
    setVariable("_fread", makeNativeFunc(nativeFread));
    setVariable("_fwrite", makeNativeFunc(nativeFwrite));
    setVariable("_input", makeNativeFunc(nativeInput));
    setVariable("_system", makeNativeFunc(nativeSystem));
}
