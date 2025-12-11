#ifndef JAI_COMPILER_H
#define JAI_COMPILER_H

#include "vm.h"
#include "../lang/lexer.h"

#define MAX_LOCALS 256
#define MAX_LOOP_DEPTH 32

typedef struct {
    char name[MAX_NAME_LEN];
    int depth;
} Local;

typedef struct {
    int loopStart;
    int breakJumps[64];
    int breakCount;
} LoopInfo;

typedef struct {
    CompiledFunc* function;
    Local locals[MAX_LOCALS];
    int localCount;
    int scopeDepth;
    LoopInfo loops[MAX_LOOP_DEPTH];
    int loopDepth;
    bool hadError;
    bool panicMode;
    
    char currentClass[MAX_NAME_LEN];
    bool inMethod;
} Compiler;

void compilerInit(Compiler* compiler, const char* functionName);
void compilerFree(Compiler* compiler);

Chunk* currentChunk(Compiler* compiler);

void emitByte(Compiler* compiler, uint8_t byte, int line);
void emitBytes(Compiler* compiler, uint8_t byte1, uint8_t byte2, int line);
void emitConstant(Compiler* compiler, Value value, int line);
int emitJump(Compiler* compiler, uint8_t instruction, int line);
void patchJump(Compiler* compiler, int offset);
void emitLoop(Compiler* compiler, int loopStart, int line);
void emitReturn(Compiler* compiler, int line);

int resolveLocal(Compiler* compiler, const char* name);
void addLocal(Compiler* compiler, const char* name);
void beginScope(Compiler* compiler);
void endScope(Compiler* compiler);

CompiledFunc* compileFunction(const JaiFunction* func, Token* tokens, int tokenCount);
bool compileStmts(Compiler* compiler, Token* tokens, int* pos, int end);
bool compileExpr(Compiler* compiler, Token* tokens, int* pos, int end);

void compileError(Compiler* compiler, const char* message, int line);

#endif
