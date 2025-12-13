#ifndef VM_H
#define VM_H

#include "../core/runtime.h"
#include <stdint.h>

typedef enum {
    
    OP_CONST,
    OP_POP,
    OP_DUP,
    
    
    OP_GET_LOCAL,
    OP_SET_LOCAL,
    OP_GET_GLOBAL,
    OP_SET_GLOBAL,
    
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_MOD,
    OP_POW,
    OP_NEG,

    OP_EQ,
    OP_NE,
    OP_LT,
    OP_LE,
    OP_GT,
    OP_GE,
    
    OP_NOT,
    OP_AND,
    OP_OR,
    
    OP_JUMP,
    OP_JUMP_IF_FALSE,   
    OP_LOOP,            
    
    OP_CALL,
    OP_RETURN,
    
    OP_NEW_ARRAY,    
    OP_ARRAY_GET,    
    OP_ARRAY_SET,    
    OP_ARRAY_PUSH,  
    OP_ARRAY_LEN,    
    
    OP_NEW_OBJECT,   
    OP_GET_FIELD,    
    OP_SET_FIELD,    
    OP_CALL_METHOD,  
    
    OP_PRINT,           
    OP_HALT,            
} OpCode;

typedef struct {
    uint8_t* code;
    int count;
    int capacity;
    
    Value* constants;
    int constCount;
    int constCapacity;
    
    int* lines;
} Chunk;

void chunkInit(Chunk* chunk);
void chunkFree(Chunk* chunk);
void chunkWrite(Chunk* chunk, uint8_t byte, int line);
int chunkAddConstant(Chunk* chunk, Value value);

typedef struct {
    char name[MAX_NAME_LEN];
    int arity;
    bool isVariadic;
    Chunk chunk;
    char** paramNames;
} CompiledFunc;

CompiledFunc* compiledFuncNew(const char* name, int arity);
void compiledFuncFree(CompiledFunc* func);

#define FRAMES_MAX 256
#define STACK_MAX (FRAMES_MAX * 256)

typedef struct {
    CompiledFunc* function;
    uint8_t* ip; 
    Value* slots;
} CallFrame;

typedef struct {
    CallFrame frames[FRAMES_MAX];
    int frameCount;
    
    Value stack[STACK_MAX];
    Value* stackTop;
    
    Value result;
    
    bool running;
} VM;

void vmInit(VM* vm);
void vmFree(VM* vm);
void vmPush(VM* vm, Value value);
Value vmPop(VM* vm);
Value vmPeek(VM* vm, int distance);
Value vmGetResult(VM* vm);

typedef enum {
    INTERPRET_OK,
    INTERPRET_COMPILE_ERROR,
    INTERPRET_RUNTIME_ERROR,
} InterpretResult;

InterpretResult vmRun(VM* vm, CompiledFunc* main);
InterpretResult vmExecute(VM* vm);

#endif
