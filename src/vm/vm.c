#include "vm.h"
#include <stdio.h>
#include <string.h>

void chunkInit(Chunk* chunk) {
    chunk->count = 0;
    chunk->capacity = 0;
    chunk->code = NULL;
    chunk->lines = NULL;
    chunk->constCount = 0;
    chunk->constCapacity = 0;
    chunk->constants = NULL;
}

void chunkFree(Chunk* chunk) {
    if (chunk->code) free(chunk->code);
    if (chunk->lines) free(chunk->lines);
    if (chunk->constants) free(chunk->constants);
    chunkInit(chunk);
}

void chunkWrite(Chunk* chunk, uint8_t byte, int line) {
    if (chunk->capacity < chunk->count + 1) {
        int oldCap = chunk->capacity;
        chunk->capacity = oldCap < 8 ? 8 : oldCap * 2;
        chunk->code = realloc(chunk->code, chunk->capacity);
        chunk->lines = realloc(chunk->lines, chunk->capacity * sizeof(int));
    }
    chunk->code[chunk->count] = byte;
    chunk->lines[chunk->count] = line;
    chunk->count++;
}

int chunkAddConstant(Chunk* chunk, Value value) {
    if (chunk->constCapacity < chunk->constCount + 1) {
        int oldCap = chunk->constCapacity;
        chunk->constCapacity = oldCap < 8 ? 8 : oldCap * 2;
        chunk->constants = realloc(chunk->constants, chunk->constCapacity * sizeof(Value));
    }
    chunk->constants[chunk->constCount] = value;
    return chunk->constCount++;
}

CompiledFunc* compiledFuncNew(const char* name, int arity) {
    CompiledFunc* func = malloc(sizeof(CompiledFunc));
    strncpy(func->name, name, MAX_NAME_LEN - 1);
    func->name[MAX_NAME_LEN - 1] = '\0';
    func->arity = arity;
    func->isVariadic = false;
    func->paramNames = NULL;
    chunkInit(&func->chunk);
    return func;
}

void compiledFuncFree(CompiledFunc* func) {
    if (!func) return;
    chunkFree(&func->chunk);
    if (func->paramNames) {
        for (int i = 0; i < func->arity; i++) {
            if (func->paramNames[i]) free(func->paramNames[i]);
        }
        free(func->paramNames);
    }
    free(func);
}

void vmInit(VM* vm) {
    vm->stackTop = vm->stack;
    vm->frameCount = 0;
    vm->running = false;
    vm->result = makeNull();
}

void vmFree(VM* vm) {
    vmInit(vm);
}

void vmPush(VM* vm, Value value) {
    if (vm->stackTop - vm->stack >= STACK_MAX) {
        fprintf(stderr, "VM Error: Stack overflow\n");
        return;
    }
    *vm->stackTop = value;
    vm->stackTop++;
}

Value vmPop(VM* vm) {
    if (vm->stackTop == vm->stack) {
        fprintf(stderr, "VM Error: Stack underflow\n");
        return makeNull();
    }
    vm->stackTop--;
    return *vm->stackTop;
}

Value vmPeek(VM* vm, int distance) {
    return vm->stackTop[-1 - distance];
}

static inline uint8_t readByte(CallFrame* frame) {
    return *frame->ip++;
}

static inline uint16_t readShort(CallFrame* frame) {
    uint16_t result = (frame->ip[0] << 8) | frame->ip[1];
    frame->ip += 2;
    return result;
}

static inline Value readConstant(CallFrame* frame) {
    return frame->function->chunk.constants[readByte(frame)];
}

#define BINARY_OP(vm, op) \
    do { \
        Value b = vmPop(vm); \
        Value a = vmPop(vm); \
        if (a.type != VAL_NUMBER || b.type != VAL_NUMBER) { \
            fprintf(stderr, "VM Error: Operands must be numbers\n"); \
            return INTERPRET_RUNTIME_ERROR; \
        } \
        vmPush(vm, makeNumber(a.as.number op b.as.number)); \
    } while (0)

InterpretResult vmExecute(VM* vm) {
    CallFrame* frame = &vm->frames[vm->frameCount - 1];
    
    vm->running = true;
    
    while (vm->running) {
        #ifdef DEBUG_TRACE
        printf("          ");
        for (Value* slot = vm->stack; slot < vm->stackTop; slot++) {
            printf("[ ");
            printf(" ]");
        }
        printf("\n");
        #endif
        
        uint8_t instruction = readByte(frame);
        
        switch (instruction) {
            case OP_CONST: {
                Value constant = readConstant(frame);
                vmPush(vm, constant);
                break;
            }
            
            case OP_POP: {
                vmPop(vm);
                break;
            }
            
            case OP_DUP: {
                vmPush(vm, vmPeek(vm, 0));
                break;
            }
            
            case OP_GET_LOCAL: {
                uint8_t slot = readByte(frame);
                vmPush(vm, frame->slots[slot]);
                break;
            }
            
            case OP_SET_LOCAL: {
                uint8_t slot = readByte(frame);
                frame->slots[slot] = vmPeek(vm, 0);
                break;
            }
            
            case OP_GET_GLOBAL: {
                Value nameVal = readConstant(frame);
                if (nameVal.type != VAL_STRING) {
                    fprintf(stderr, "VM Error: Global name must be string\n");
                    return INTERPRET_RUNTIME_ERROR;
                }
                Value value = getVariable(nameVal.as.string);
                vmPush(vm, value);
                break;
            }
            
            case OP_SET_GLOBAL: {
                Value nameVal = readConstant(frame);
                if (nameVal.type != VAL_STRING) {
                    fprintf(stderr, "VM Error: Global name must be string\n");
                    return INTERPRET_RUNTIME_ERROR;
                }
                setVariable(nameVal.as.string, vmPeek(vm, 0));
                break;
            }
            
            case OP_ADD: {
                Value b = vmPop(vm);
                Value a = vmPop(vm);
                if (a.type == VAL_NUMBER && b.type == VAL_NUMBER) {
                    vmPush(vm, makeNumber(a.as.number + b.as.number));
                } else if (a.type == VAL_STRING && b.type == VAL_STRING) {
                    size_t len = strlen(a.as.string) + strlen(b.as.string) + 1;
                    char* result = malloc(len);
                    strcpy(result, a.as.string);
                    strcat(result, b.as.string);
                    vmPush(vm, makeString(result));
                    free(result);
                } else {
                    fprintf(stderr, "VM Error: Cannot add these types\n");
                    return INTERPRET_RUNTIME_ERROR;
                }
                break;
            }
            
            case OP_SUB: BINARY_OP(vm, -); break;
            case OP_MUL: BINARY_OP(vm, *); break;
            case OP_DIV: BINARY_OP(vm, /); break;
            
            case OP_MOD: {
                Value b = vmPop(vm);
                Value a = vmPop(vm);
                if (a.type != VAL_NUMBER || b.type != VAL_NUMBER) {
                    fprintf(stderr, "VM Error: Operands must be numbers\n");
                    return INTERPRET_RUNTIME_ERROR;
                }
                vmPush(vm, makeNumber(fmod(a.as.number, b.as.number)));
                break;
            }
            
            case OP_POW: {
                Value b = vmPop(vm);
                Value a = vmPop(vm);
                if (a.type != VAL_NUMBER || b.type != VAL_NUMBER) {
                    fprintf(stderr, "VM Error: Operands must be numbers\n");
                    return INTERPRET_RUNTIME_ERROR;
                }
                vmPush(vm, makeNumber(pow(a.as.number, b.as.number)));
                break;
            }
            
            case OP_NEG: {
                Value a = vmPop(vm);
                if (a.type != VAL_NUMBER) {
                    fprintf(stderr, "VM Error: Operand must be number\n");
                    return INTERPRET_RUNTIME_ERROR;
                }
                vmPush(vm, makeNumber(-a.as.number));
                break;
            }
            
            case OP_EQ: {
                Value b = vmPop(vm);
                Value a = vmPop(vm);
                bool eq = false;
                if (a.type != b.type) {
                    eq = false;
                } else if (a.type == VAL_NUMBER) {
                    eq = a.as.number == b.as.number;
                } else if (a.type == VAL_BOOL) {
                    eq = a.as.boolean == b.as.boolean;
                } else if (a.type == VAL_NULL) {
                    eq = true;
                } else if (a.type == VAL_STRING) {
                    eq = strcmp(a.as.string, b.as.string) == 0;
                }
                vmPush(vm, makeBool(eq));
                break;
            }
            
            case OP_NE: {
                Value b = vmPop(vm);
                Value a = vmPop(vm);
                bool eq = false;
                if (a.type != b.type) {
                    eq = false;
                } else if (a.type == VAL_NUMBER) {
                    eq = a.as.number == b.as.number;
                } else if (a.type == VAL_BOOL) {
                    eq = a.as.boolean == b.as.boolean;
                } else if (a.type == VAL_NULL) {
                    eq = true;
                } else if (a.type == VAL_STRING) {
                    eq = strcmp(a.as.string, b.as.string) == 0;
                }
                vmPush(vm, makeBool(!eq));
                break;
            }
            
            case OP_LT: {
                Value b = vmPop(vm);
                Value a = vmPop(vm);
                if (a.type != VAL_NUMBER || b.type != VAL_NUMBER) {
                    fprintf(stderr, "VM Error: Operands must be numbers\n");
                    return INTERPRET_RUNTIME_ERROR;
                }
                vmPush(vm, makeBool(a.as.number < b.as.number));
                break;
            }
            
            case OP_LE: {
                Value b = vmPop(vm);
                Value a = vmPop(vm);
                if (a.type != VAL_NUMBER || b.type != VAL_NUMBER) {
                    fprintf(stderr, "VM Error: Operands must be numbers\n");
                    return INTERPRET_RUNTIME_ERROR;
                }
                vmPush(vm, makeBool(a.as.number <= b.as.number));
                break;
            }
            
            case OP_GT: {
                Value b = vmPop(vm);
                Value a = vmPop(vm);
                if (a.type != VAL_NUMBER || b.type != VAL_NUMBER) {
                    fprintf(stderr, "VM Error: Operands must be numbers\n");
                    return INTERPRET_RUNTIME_ERROR;
                }
                vmPush(vm, makeBool(a.as.number > b.as.number));
                break;
            }
            
            case OP_GE: {
                Value b = vmPop(vm);
                Value a = vmPop(vm);
                if (a.type != VAL_NUMBER || b.type != VAL_NUMBER) {
                    fprintf(stderr, "VM Error: Operands must be numbers\n");
                    return INTERPRET_RUNTIME_ERROR;
                }
                vmPush(vm, makeBool(a.as.number >= b.as.number));
                break;
            }
            
            case OP_NOT: {
                Value a = vmPop(vm);
                bool isFalsy = (a.type == VAL_NULL) ||
                               (a.type == VAL_BOOL && !a.as.boolean) ||
                               (a.type == VAL_NUMBER && a.as.number == 0);
                vmPush(vm, makeBool(isFalsy));
                break;
            }
            
            case OP_AND: {
                Value b = vmPop(vm);
                Value a = vmPop(vm);
                bool aTrue = !(a.type == VAL_NULL || (a.type == VAL_BOOL && !a.as.boolean));
                bool bTrue = !(b.type == VAL_NULL || (b.type == VAL_BOOL && !b.as.boolean));
                vmPush(vm, makeBool(aTrue && bTrue));
                break;
            }
            
            case OP_OR: {
                Value b = vmPop(vm);
                Value a = vmPop(vm);
                bool aTrue = !(a.type == VAL_NULL || (a.type == VAL_BOOL && !a.as.boolean));
                bool bTrue = !(b.type == VAL_NULL || (b.type == VAL_BOOL && !b.as.boolean));
                vmPush(vm, makeBool(aTrue || bTrue));
                break;
            }
            
            case OP_JUMP: {
                uint16_t offset = readShort(frame);
                frame->ip += offset;
                break;
            }
            
            case OP_JUMP_IF_FALSE: {
                uint16_t offset = readShort(frame);
                Value condition = vmPeek(vm, 0);
                bool isFalsy = (condition.type == VAL_NULL) ||
                               (condition.type == VAL_BOOL && !condition.as.boolean) ||
                               (condition.type == VAL_NUMBER && condition.as.number == 0);
                if (isFalsy) {
                    frame->ip += offset;
                }
                break;
            }
            
            case OP_LOOP: {
                uint16_t offset = readShort(frame);
                frame->ip -= offset;
                break;
            }
            
            case OP_CALL: {
                uint8_t argCount = readByte(frame);
                Value callee = vmPeek(vm, argCount);
                
                if (callee.type == VAL_NATIVE_FUNC) {
                    Value* args = vm->stackTop - argCount;
                    Value result = callee.as.nativeFunc(args, argCount);
                    vm->stackTop -= argCount + 1;
                    vmPush(vm, result);
                } else if (callee.type == VAL_FUNCTION) {
                    //TODO: compile functions to bytecode
                    JaiFunction* f = callee.as.function;
                    Value* args = vm->stackTop - argCount;
                    extern Value callValue(Value callee, Value* args, int argc);
                    Value result = callValue(callee, args, argCount);
                    vm->stackTop -= argCount + 1;
                    vmPush(vm, result);
                } else {
                    fprintf(stderr, "VM Error: Can only call functions\n");
                    return INTERPRET_RUNTIME_ERROR;
                }
                break;
            }
            
            case OP_RETURN: {
                Value result = vmPop(vm);
                vm->result = result;
                vm->frameCount--;
                if (vm->frameCount == 0) {
                    vm->running = false;
                    return INTERPRET_OK;
                }
                vm->stackTop = frame->slots;
                vmPush(vm, result);
                frame = &vm->frames[vm->frameCount - 1];
                break;
            }
            
            case OP_NEW_ARRAY: {
                uint8_t size = readByte(frame);
                Value arr = makeArray(size);
                for (int i = size - 1; i >= 0; i--) {
                    Value val = vmPop(vm);
                    arraySet(arr.as.array, i, val);
                }
                arr.as.array->length = size;
                vmPush(vm, arr);
                break;
            }
            
            case OP_ARRAY_GET: {
                Value index = vmPop(vm);
                Value arr = vmPop(vm);
                if (arr.type != VAL_ARRAY) {
                    fprintf(stderr, "VM Error: Cannot index non-array\n");
                    return INTERPRET_RUNTIME_ERROR;
                }
                if (index.type != VAL_NUMBER) {
                    fprintf(stderr, "VM Error: Array index must be number\n");
                    return INTERPRET_RUNTIME_ERROR;
                }
                vmPush(vm, arrayGet(arr.as.array, (int)index.as.number));
                break;
            }
            
            case OP_ARRAY_SET: {
                Value value = vmPop(vm);
                Value index = vmPop(vm);
                Value arr = vmPop(vm);
                if (arr.type != VAL_ARRAY) {
                    fprintf(stderr, "VM Error: Cannot index non-array\n");
                    return INTERPRET_RUNTIME_ERROR;
                }
                arraySet(arr.as.array, (int)index.as.number, value);
                vmPush(vm, value);
                break;
            }
            
            case OP_ARRAY_PUSH: {
                Value value = vmPop(vm);
                Value arr = vmPop(vm);
                if (arr.type != VAL_ARRAY) {
                    fprintf(stderr, "VM Error: Cannot push to non-array\n");
                    return INTERPRET_RUNTIME_ERROR;
                }
                arrayPush(arr.as.array, value);
                vmPush(vm, arr);
                break;
            }
            
            case OP_ARRAY_LEN: {
                Value arr = vmPop(vm);
                if (arr.type != VAL_ARRAY) {
                    fprintf(stderr, "VM Error: Cannot get length of non-array\n");
                    return INTERPRET_RUNTIME_ERROR;
                }
                vmPush(vm, makeNumber(arr.as.array->length));
                break;
            }

            case OP_NEW_OBJECT: {
                Value classNameVal = readConstant(frame);
                uint8_t argCount = readByte(frame);
                if (classNameVal.type != VAL_STRING) {
                    fprintf(stderr, "VM Error: Class name must be string\n");
                    return INTERPRET_RUNTIME_ERROR;
                }
                JaiClass* cls = findClass(classNameVal.as.string);
                if (!cls) {
                    fprintf(stderr, "VM Error: Class not found: %s\n", classNameVal.as.string);
                    return INTERPRET_RUNTIME_ERROR;
                }
                Value obj = makeObject(cls);

                Value args[256];
                args[0] = obj;
                for (int i = argCount - 1; i >= 0; i--) {
                    args[i + 1] = vmPop(vm);
                }

                if (cls->constructor) {
                    extern Value callValue(Value callee, Value* args, int argc);
                    callValue(makeFunction(cls->constructor), args, argCount + 1);
                }

                vmPush(vm, obj);
                break;
            }
            
            case OP_GET_FIELD: {
                Value nameVal = readConstant(frame);
                Value obj = vmPop(vm);
                
                if (obj.type == VAL_OBJECT) {
                    Value field = objectGetField(obj.as.object, nameVal.as.string);
                    vmPush(vm, field);
                } else {
                    fprintf(stderr, "VM Error: Cannot get field of non-object\n");
                    return INTERPRET_RUNTIME_ERROR;
                }
                break;
            }
            
            case OP_SET_FIELD: {
                Value nameVal = readConstant(frame);
                Value value = vmPop(vm);
                Value obj = vmPop(vm);
                
                if (obj.type == VAL_OBJECT) {
                    objectSetField(obj.as.object, nameVal.as.string, value);
                    vmPush(vm, value);
                } else {
                    fprintf(stderr, "VM Error: Cannot set field of non-object\n");
                    return INTERPRET_RUNTIME_ERROR;
                }
                break;
            }
            
            case OP_CALL_METHOD: {
                Value nameVal = readConstant(frame);
                uint8_t argCount = readByte(frame);
                
                Value obj = vmPeek(vm, argCount);
                
                if (obj.type == VAL_OBJECT) {
                    JaiFunction* method = objectGetMethod(obj.as.object, nameVal.as.string);
                    if (!method) {
                        fprintf(stderr, "VM Error: Object has no method '%s'\n", nameVal.as.string);
                        return INTERPRET_RUNTIME_ERROR;
                    }
                    
                    Value args[256];
                    args[0] = obj;
                    for (int i = argCount - 1; i >= 0; i--) {
                        args[argCount - i] = vmPop(vm);
                    }
                    vmPop(vm);
                    
                    extern Value callValue(Value callee, Value* args, int argc);
                    Value result = callValue(makeFunction(method), args, argCount + 1);
                    vmPush(vm, result);
                } else {
                    fprintf(stderr, "VM Error: Cannot call method on non-object\n");
                    return INTERPRET_RUNTIME_ERROR;
                }
                break;
            }
            
            case OP_PRINT: {
                Value val = vmPop(vm);
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
                    case VAL_STRING:
                        printf("%s\n", val.as.string);
                        break;
                    case VAL_BOOL:
                        printf("%d\n", val.as.boolean ? 1 : 0);
                        break;
                    case VAL_NULL:
                        printf("null\n");
                        break;
                    default:
                        printf("<value>\n");
                        break;
                }
                break;
            }
            
            case OP_HALT: {
                vm->running = false;
                return INTERPRET_OK;
            }
            
            default:
                fprintf(stderr, "VM Error: Unknown opcode %d\n", instruction);
                return INTERPRET_RUNTIME_ERROR;
        }
    }
    
    return INTERPRET_OK;
}

Value vmGetResult(VM* vm) {
    if (vm->stackTop > vm->stack) {
        return vm->stackTop[-1];
    }
    return makeNull();
}

InterpretResult vmRun(VM* vm, CompiledFunc* main) {
    CallFrame* frame = &vm->frames[vm->frameCount++];
    frame->function = main;
    frame->ip = main->chunk.code;
    frame->slots = vm->stack;
    
    return vmExecute(vm);
}
