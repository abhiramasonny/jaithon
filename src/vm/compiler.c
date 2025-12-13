#include "compiler.h"
#include <stdio.h>
#include <string.h>

void compilerInit(Compiler* compiler, const char* functionName) {
    compiler->function = compiledFuncNew(functionName, 0);
    compiler->localCount = 0;
    compiler->scopeDepth = 0;
    compiler->loopDepth = 0;
    compiler->hadError = false;
    compiler->panicMode = false;
    compiler->currentClass[0] = '\0';
    compiler->inMethod = false;
}

void compilerFree(Compiler* compiler) {
}

Chunk* currentChunk(Compiler* compiler) {
    return &compiler->function->chunk;
}

void emitByte(Compiler* compiler, uint8_t byte, int line) {
    chunkWrite(currentChunk(compiler), byte, line);
}

void emitBytes(Compiler* compiler, uint8_t byte1, uint8_t byte2, int line) {
    emitByte(compiler, byte1, line);
    emitByte(compiler, byte2, line);
}

void emitConstant(Compiler* compiler, Value value, int line) {
    int idx = chunkAddConstant(currentChunk(compiler), value);
    if (idx > 255) {
        compileError(compiler, "Too many constants in one chunk", line);
        return;
    }
    emitBytes(compiler, OP_CONST, (uint8_t)idx, line);
}

int emitJump(Compiler* compiler, uint8_t instruction, int line) {
    emitByte(compiler, instruction, line);
    emitByte(compiler, 0xff, line);
    emitByte(compiler, 0xff, line);
    return currentChunk(compiler)->count - 2;
}

void patchJump(Compiler* compiler, int offset) {
    Chunk* chunk = currentChunk(compiler);
    int jump = chunk->count - offset - 2;
    
    if (jump > 65535) {
        compileError(compiler, "Jump too large", 0);
        return;
    }
    
    chunk->code[offset] = (jump >> 8) & 0xff;
    chunk->code[offset + 1] = jump & 0xff;
}

void emitLoop(Compiler* compiler, int loopStart, int line) {
    emitByte(compiler, OP_LOOP, line);
    
    int offset = currentChunk(compiler)->count - loopStart + 2;
    if (offset > 65535) {
        compileError(compiler, "Loop body too large", line);
        return;
    }
    
    emitByte(compiler, (offset >> 8) & 0xff, line);
    emitByte(compiler, offset & 0xff, line);
}

void emitReturn(Compiler* compiler, int line) {
    emitByte(compiler, OP_CONST, line);
    emitByte(compiler, chunkAddConstant(currentChunk(compiler), makeNull()), line);
    emitByte(compiler, OP_RETURN, line);
}

int resolveLocal(Compiler* compiler, const char* name) {
    for (int i = compiler->localCount - 1; i >= 0; i--) {
        if (strcmp(compiler->locals[i].name, name) == 0) {
            return i;
        }
    }
    return -1; 
}

void addLocal(Compiler* compiler, const char* name) {
    if (compiler->localCount >= MAX_LOCALS) {
        compileError(compiler, "Too many local variables", 0);
        return;
    }
    Local* local = &compiler->locals[compiler->localCount++];
    strncpy(local->name, name, MAX_NAME_LEN - 1);
    local->name[MAX_NAME_LEN - 1] = '\0';
    local->depth = compiler->scopeDepth;
}

void beginScope(Compiler* compiler) {
    compiler->scopeDepth++;
}

void endScope(Compiler* compiler) {
    compiler->scopeDepth--;
    
    while (compiler->localCount > 0 &&
           compiler->locals[compiler->localCount - 1].depth > compiler->scopeDepth) {
        emitByte(compiler, OP_POP, 0);
        compiler->localCount--;
    }
}

void compileError(Compiler* compiler, const char* message, int line) {
    if (compiler->panicMode) return;
    compiler->panicMode = true;
    compiler->hadError = true;
    
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
        fprintf(stderr, "[COMPILE_ERROR] %s at line %d in function '%s'\n", 
                message, line, compiler->function->name);
    }
}

static bool isAtEnd(Token* tokens, int* pos, int end) {
    return *pos >= end || tokens[*pos].kind == TK_EOF;
}

static Token current(Token* tokens, int pos) {
    return tokens[pos];
}

static Token advance(Token* tokens, int* pos) {
    return tokens[(*pos)++];
}

static bool check(Token* tokens, int pos, int kind) {
    return tokens[pos].kind == kind;
}

static bool match(Token* tokens, int* pos, int kind) {
    if (tokens[*pos].kind == kind) {
        (*pos)++;
        return true;
    }
    return false;
}

static void skipNewlinesTokens(Token* tokens, int* pos, int end) {
    while (*pos < end && tokens[*pos].kind == TK_NEWLINE) {
        (*pos)++;
    }
}

static bool compileUnary(Compiler* compiler, Token* tokens, int* pos, int end);
static bool compileBinary(Compiler* compiler, Token* tokens, int* pos, int end, int minPrec);

static int getPrecedence(int tokenKind) {
    
    
    if (tokenKind == getKW_OR()) return 4;
    if (tokenKind == getKW_AND()) return 5;
    
    switch (tokenKind) {
        case TK_EQ_EQ:
        case TK_NE:
            return 9;
        case TK_LT:
        case TK_GT:
        case TK_LE:
        case TK_GE:
            return 10;
        case TK_PLUS:
        case TK_MINUS:
            return 12;
        case TK_STAR:
        case TK_SLASH:
        case TK_PERCENT:
            return 13;
        case TK_CARET:
            return 14;  
        default:
            return 0;
    }
}

static OpCode binaryOp(int tokenKind) {
    switch (tokenKind) {
        case TK_PLUS: return OP_ADD;
        case TK_MINUS: return OP_SUB;
        case TK_STAR: return OP_MUL;
        case TK_SLASH: return OP_DIV;
        case TK_PERCENT: return OP_MOD;
        case TK_CARET: return OP_POW;
        case TK_EQ_EQ: return OP_EQ;
        case TK_NE: return OP_NE;
        case TK_LT: return OP_LT;
        case TK_LE: return OP_LE;
        case TK_GT: return OP_GT;
        case TK_GE: return OP_GE;
        default: return OP_HALT;
    }
}

static bool compilePrimary(Compiler* compiler, Token* tokens, int* pos, int end) {
    if (isAtEnd(tokens, pos, end)) {
        compileError(compiler, "Unexpected end of expression", 0);
        return false;
    }
    
    Token t = advance(tokens, pos);
    int line = t.line;
    
    if (t.kind == TK_NUMBER) {
        emitConstant(compiler, makeNumber(t.numValue), line);
        return true;
    }
    
    if (t.kind == TK_STRING) {
        emitConstant(compiler, makeString(t.strValue), line);
        return true;
    }
    
    if (t.kind == getKW_TRUE()) {
        emitConstant(compiler, makeBool(true), line);
        return true;
    }
    if (t.kind == getKW_FALSE()) {
        emitConstant(compiler, makeBool(false), line);
        return true;
    }
    if (t.kind == getKW_NULL()) {
        emitConstant(compiler, makeNull(), line);
        return true;
    }
    
    if (t.kind == TK_LPAREN) {
        if (!compileExpr(compiler, tokens, pos, end)) return false;
        if (!match(tokens, pos, TK_RPAREN)) {
            compileError(compiler, "Expected ')' after expression", line);
            return false;
        }
        return true;
    }

    if (t.kind == TK_LBRACKET) {
        int count = 0;
        skipNewlinesTokens(tokens, pos, end);
        if (!check(tokens, *pos, TK_RBRACKET)) {
            do {
                skipNewlinesTokens(tokens, pos, end);
                if (!compileExpr(compiler, tokens, pos, end)) return false;
                count++;
                skipNewlinesTokens(tokens, pos, end);
            } while (match(tokens, pos, TK_COMMA));
        }
        if (!match(tokens, pos, TK_RBRACKET)) {
            compileError(compiler, "Expected ']' after array elements", line);
            return false;
        }
        emitBytes(compiler, OP_NEW_ARRAY, count, line);
        return true;
    }

    if (t.kind == getKW_NEW()) {
        if (!check(tokens, *pos, TK_IDENTIFIER)) {
            compileError(compiler, "Expected class name after 'new'", line);
            return false;
        }
        char className[MAX_NAME_LEN];
        strcpy(className, tokens[*pos].strValue);
        advance(tokens, pos);

        int argc = 0;
        if (match(tokens, pos, TK_LPAREN)) {
            if (!check(tokens, *pos, TK_RPAREN)) {
                do {
                    if (!compileExpr(compiler, tokens, pos, end)) return false;
                    argc++;
                } while (match(tokens, pos, TK_COMMA));
            }
            if (!match(tokens, pos, TK_RPAREN)) {
                compileError(compiler, "Expected ')' after constructor args", line);
                return false;
            }
        }

        int nameIdx = chunkAddConstant(currentChunk(compiler), makeString(className));
        emitByte(compiler, OP_NEW_OBJECT, line);
        emitByte(compiler, (uint8_t)nameIdx, line);
        emitByte(compiler, (uint8_t)argc, line);
        return true;
    }
    
    if (t.kind == TK_MINUS) {
        if (!compilePrimary(compiler, tokens, pos, end)) return false;
        emitByte(compiler, OP_NEG, line);
        return true;
    }

    if (t.kind == getKW_NOT() || t.kind == TK_BANG) {
        if (!compilePrimary(compiler, tokens, pos, end)) return false;
        emitByte(compiler, OP_NOT, line);
        return true;
    }
    
    if (t.kind == TK_IDENTIFIER || t.kind == getKW_SELF()) {
        char name[MAX_NAME_LEN];
        strcpy(name, t.strValue);
        
        int slotForName = resolveLocal(compiler, name);
        
        if (check(tokens, *pos, TK_LPAREN)) {
            if (slotForName != -1) {
                emitBytes(compiler, OP_GET_LOCAL, slotForName, line);
            } else {
                int constIdx = chunkAddConstant(currentChunk(compiler), makeString(name));
                emitBytes(compiler, OP_GET_GLOBAL, constIdx, line);
            }
            
            advance(tokens, pos);
            int argc = 0;
            if (!check(tokens, *pos, TK_RPAREN)) {
                do {
                    if (!compileExpr(compiler, tokens, pos, end)) return false;
                    argc++;
                } while (match(tokens, pos, TK_COMMA));
            }
            if (!match(tokens, pos, TK_RPAREN)) {
                compileError(compiler, "Expected ')' after arguments", line);
                return false;
            }
            emitBytes(compiler, OP_CALL, argc, line);
            return true;
        }

        if (slotForName != -1) {
            emitBytes(compiler, OP_GET_LOCAL, slotForName, line);
        } else {
            int constIdx = chunkAddConstant(currentChunk(compiler), makeString(name));
            emitBytes(compiler, OP_GET_GLOBAL, constIdx, line);
        }
        
        while (check(tokens, *pos, TK_LBRACKET) || check(tokens, *pos, TK_DOT)) {
            if (check(tokens, *pos, TK_LBRACKET)) {
                advance(tokens, pos);
                if (!compileExpr(compiler, tokens, pos, end)) return false;
                if (!match(tokens, pos, TK_RBRACKET)) {
                    compileError(compiler, "Expected ']' after index", line);
                    return false;
                }
                emitByte(compiler, OP_ARRAY_GET, line);
            } else if (check(tokens, *pos, TK_DOT)) {
                advance(tokens, pos);
                if (!check(tokens, *pos, TK_IDENTIFIER)) {
                    compileError(compiler, "Expected field name after '.'", line);
                    return false;
                }
                char fieldName[MAX_NAME_LEN];
                strcpy(fieldName, tokens[*pos].strValue);
                advance(tokens, pos);
                
                if (check(tokens, *pos, TK_LPAREN)) {
                    advance(tokens, pos);
                    int argc = 0;
                    if (!check(tokens, *pos, TK_RPAREN)) {
                        do {
                            if (!compileExpr(compiler, tokens, pos, end)) return false;
                            argc++;
                        } while (match(tokens, pos, TK_COMMA));
                    }
                    if (!match(tokens, pos, TK_RPAREN)) {
                        compileError(compiler, "Expected ')' after method args", line);
                        return false;
                    }
                    int nameIdx = chunkAddConstant(currentChunk(compiler), makeString(fieldName));
                    emitBytes(compiler, OP_CALL_METHOD, nameIdx, line);
                    emitByte(compiler, argc, line);
                } else {
                    int nameIdx = chunkAddConstant(currentChunk(compiler), makeString(fieldName));
                    emitBytes(compiler, OP_GET_FIELD, nameIdx, line);
                }
            }
        }
        
        return true;
    }
    
    compileError(compiler, "Unexpected token in expression", t.line);
    return false;
}

bool compileExpr(Compiler* compiler, Token* tokens, int* pos, int end) {
    return compileBinary(compiler, tokens, pos, end, 1);
}

static bool compileBinary(Compiler* compiler, Token* tokens, int* pos, int end, int minPrec) {
    if (!compilePrimary(compiler, tokens, pos, end)) return false;
    
    while (!isAtEnd(tokens, pos, end)) {
        int prec = getPrecedence(tokens[*pos].kind);
        if (prec < minPrec) break;
        
        Token op = advance(tokens, pos);
        OpCode opCode = binaryOp(op.kind);
        
        if (op.kind == getKW_AND()) {
            int jump = emitJump(compiler, OP_JUMP_IF_FALSE, op.line);
            emitByte(compiler, OP_POP, op.line);
            if (!compileBinary(compiler, tokens, pos, end, prec + 1)) return false;
            patchJump(compiler, jump);
            continue;
        }
        if (op.kind == getKW_OR()) {
            int elseJump = emitJump(compiler, OP_JUMP_IF_FALSE, op.line);
            int endJump = emitJump(compiler, OP_JUMP, op.line);
            patchJump(compiler, elseJump);
            emitByte(compiler, OP_POP, op.line);
            if (!compileBinary(compiler, tokens, pos, end, prec + 1)) return false;
            patchJump(compiler, endJump);
            continue;
        }
        
        if (!compileBinary(compiler, tokens, pos, end, prec + 1)) return false;
        emitByte(compiler, opCode, op.line);
    }
    
    return true;
}

static bool compileVarDecl(Compiler* compiler, Token* tokens, int* pos, int end) {
    if (!check(tokens, *pos, TK_IDENTIFIER)) {
        compileError(compiler, "Expected variable name", tokens[*pos].line);
        return false;
    }
    
    Token name = advance(tokens, pos);
    
    if (match(tokens, pos, TK_EQUALS)) {
        if (!compileExpr(compiler, tokens, pos, end)) return false;
    } else {
        emitConstant(compiler, makeNull(), name.line);
    }

    addLocal(compiler, name.strValue);
    
    return true;
}

static bool compilePrint(Compiler* compiler, Token* tokens, int* pos, int end) {
    int line = tokens[*pos - 1].line;
    if (!compileExpr(compiler, tokens, pos, end)) return false;
    emitByte(compiler, OP_PRINT, line);
    return true;
}

static bool compileIf(Compiler* compiler, Token* tokens, int* pos, int end) {
    int line = tokens[*pos - 1].line;
    
    if (!compileExpr(compiler, tokens, pos, end)) return false;
    
    skipNewlinesTokens(tokens, pos, end);
    if (check(tokens, *pos, getKW_THEN())) {
        advance(tokens, pos);
    }
    
    int thenJump = emitJump(compiler, OP_JUMP_IF_FALSE, line);
    emitByte(compiler, OP_POP, line);
    
    beginScope(compiler);
    skipNewlinesTokens(tokens, pos, end);
    while (!isAtEnd(tokens, pos, end) &&
           !check(tokens, *pos, getKW_ELSE()) &&
           !check(tokens, *pos, getKW_END())) {
        if (!compileStmts(compiler, tokens, pos, end)) return false;
        skipNewlinesTokens(tokens, pos, end);
    }
    endScope(compiler);
    
    int elseJump = emitJump(compiler, OP_JUMP, line);
    patchJump(compiler, thenJump);
    emitByte(compiler, OP_POP, line);
    
    if (check(tokens, *pos, getKW_ELSE())) {
        advance(tokens, pos);
        beginScope(compiler);
        skipNewlinesTokens(tokens, pos, end);
        while (!isAtEnd(tokens, pos, end) && !check(tokens, *pos, getKW_END())) {
            if (!compileStmts(compiler, tokens, pos, end)) return false;
            skipNewlinesTokens(tokens, pos, end);
        }
        endScope(compiler);
    }
    patchJump(compiler, elseJump);
    
    if (!match(tokens, pos, getKW_END())) {
        compileError(compiler, "Expected 'end' after if statement", line);
        return false;
    }
    
    return true;
}

static bool compileWhile(Compiler* compiler, Token* tokens, int* pos, int end) {
    int line = tokens[*pos - 1].line;
    int loopStart = currentChunk(compiler)->count;
    
    if (!compileExpr(compiler, tokens, pos, end)) return false;
    
    skipNewlinesTokens(tokens, pos, end);
    if (check(tokens, *pos, getKW_DO())) {
        advance(tokens, pos);
    }
    
    int exitJump = emitJump(compiler, OP_JUMP_IF_FALSE, line);
    emitByte(compiler, OP_POP, line);
    
    if (compiler->loopDepth >= MAX_LOOP_DEPTH) {
        compileError(compiler, "Too many nested loops", line);
        return false;
    }
    LoopInfo* loop = &compiler->loops[compiler->loopDepth++];
    loop->loopStart = loopStart;
    loop->breakCount = 0;
    
    beginScope(compiler);
    skipNewlinesTokens(tokens, pos, end);
    while (!isAtEnd(tokens, pos, end) && !check(tokens, *pos, getKW_END())) {
        if (!compileStmts(compiler, tokens, pos, end)) return false;
        skipNewlinesTokens(tokens, pos, end);
    }
    endScope(compiler);
    
    emitLoop(compiler, loopStart, line);
    
    patchJump(compiler, exitJump);
    emitByte(compiler, OP_POP, line);
    
    for (int i = 0; i < loop->breakCount; i++) {
        patchJump(compiler, loop->breakJumps[i]);
    }
    compiler->loopDepth--;
    
    if (!match(tokens, pos, getKW_END())) {
        compileError(compiler, "Expected 'end' after while loop", line);
        return false;
    }
    
    return true;
}

static bool compileReturn(Compiler* compiler, Token* tokens, int* pos, int end) {
    int line = tokens[*pos - 1].line;
    
    if (check(tokens, *pos, TK_NEWLINE) || check(tokens, *pos, TK_EOF) ||
        check(tokens, *pos, getKW_END())) {
        emitConstant(compiler, makeNull(), line);
    } else {
        if (!compileExpr(compiler, tokens, pos, end)) return false;
    }
    
    emitByte(compiler, OP_RETURN, line);
    return true;
}

static bool compileBreak(Compiler* compiler, Token* tokens, int* pos, int end) {
    int line = tokens[*pos - 1].line;
    
    if (compiler->loopDepth == 0) {
        compileError(compiler, "'break' outside of loop", line);
        return false;
    }
    
    LoopInfo* loop = &compiler->loops[compiler->loopDepth - 1];
    if (loop->breakCount >= 64) {
        compileError(compiler, "Too many breaks in loop", line);
        return false;
    }
    
    loop->breakJumps[loop->breakCount++] = emitJump(compiler, OP_JUMP, line);
    return true;
}

static bool compileAssignment(Compiler* compiler, Token* tokens, int* pos, int end) {
    Token name = tokens[*pos - 1];
    int line = name.line;
    
    if (check(tokens, *pos, TK_LBRACKET)) {
        int slot = resolveLocal(compiler, name.strValue);
        if (slot != -1) {
            emitBytes(compiler, OP_GET_LOCAL, slot, line);
        } else {
            int constIdx = chunkAddConstant(currentChunk(compiler), makeString(name.strValue));
            emitBytes(compiler, OP_GET_GLOBAL, constIdx, line);
        }
        
        advance(tokens, pos);  
        if (!compileExpr(compiler, tokens, pos, end)) return false;
        if (!match(tokens, pos, TK_RBRACKET)) {
            compileError(compiler, "Expected ']'", line);
            return false;
        }
        if (!match(tokens, pos, TK_EQUALS)) {
            emitByte(compiler, OP_ARRAY_GET, line);
            return true;
        }
        if (!compileExpr(compiler, tokens, pos, end)) return false;
        emitByte(compiler, OP_ARRAY_SET, line);
        return true;
    }
    
    if (check(tokens, *pos, TK_DOT)) {
        
        int slotObj = resolveLocal(compiler, name.strValue);
        if (slotObj != -1) {
            emitBytes(compiler, OP_GET_LOCAL, slotObj, line);
        } else {
            int constIdx = chunkAddConstant(currentChunk(compiler), makeString(name.strValue));
            emitBytes(compiler, OP_GET_GLOBAL, constIdx, line);
        }
        advance(tokens, pos); 
        if (!check(tokens, *pos, TK_IDENTIFIER)) {
            compileError(compiler, "Expected field name", line);
            return false;
        }
        char fieldName[MAX_NAME_LEN];
        strcpy(fieldName, tokens[*pos].strValue);
        advance(tokens, pos);
        if (!match(tokens, pos, TK_EQUALS)) {
            compileError(compiler, "Expected '='", line);
            return false;
        }
        if (!compileExpr(compiler, tokens, pos, end)) return false;
        int nameIdx = chunkAddConstant(currentChunk(compiler), makeString(fieldName));
        emitBytes(compiler, OP_SET_FIELD, nameIdx, line);
        emitByte(compiler, OP_POP, line);
        return true;
    }

    if (!match(tokens, pos, TK_EQUALS)) {
        (*pos)--;
        if (!compileExpr(compiler, tokens, pos, end)) return false;
        emitByte(compiler, OP_POP, line);
        return true;
    }
    
    if (!compileExpr(compiler, tokens, pos, end)) return false;
    
    int slot = resolveLocal(compiler, name.strValue);
    if (slot != -1) {
        emitBytes(compiler, OP_SET_LOCAL, slot, line);
    } else {
        int constIdx = chunkAddConstant(currentChunk(compiler), makeString(name.strValue));
        emitBytes(compiler, OP_SET_GLOBAL, constIdx, line);
    }
    emitByte(compiler, OP_POP, line);
    
    return true;
}

bool compileStmts(Compiler* compiler, Token* tokens, int* pos, int end) {
    skipNewlinesTokens(tokens, pos, end);
    if (isAtEnd(tokens, pos, end)) return true;
    
    Token t = advance(tokens, pos);
    int line = t.line;
    
    if (t.kind == getKW_VAR()) {
        return compileVarDecl(compiler, tokens, pos, end);
    }
    
    if (t.kind == getKW_PRINT()) {
        return compilePrint(compiler, tokens, pos, end);
    }
    
    if (t.kind == getKW_IF()) {
        return compileIf(compiler, tokens, pos, end);
    }
    
    if (t.kind == getKW_WHILE()) {
        return compileWhile(compiler, tokens, pos, end);
    }
    
    if (t.kind == getKW_RETURN()) {
        return compileReturn(compiler, tokens, pos, end);
    }
    
    if (t.kind == getKW_BREAK()) {
        return compileBreak(compiler, tokens, pos, end);
    }
    
    if (t.kind == TK_IDENTIFIER || t.kind == getKW_SELF()) {
        return compileAssignment(compiler, tokens, pos, end);
    }
    
    (*pos)--;
    if (!compileExpr(compiler, tokens, pos, end)) return false;
    emitByte(compiler, OP_POP, line);
    return true;
}

CompiledFunc* compileFunction(const JaiFunction* func, Token* tokens, int tokenCount) {
    Compiler compiler;
    const char* fname = func ? func->name : "<main>";
    compilerInit(&compiler, fname);
    
    for (int i = 0; i < tokenCount; i++) {
        int k = tokens[i].kind;
        if (k == getKW_CLASS() || k == getKW_NAMESPACE() || k == getKW_IMPORT()) {
            return NULL;
        }
    }
    
    if (func) {
        if (func->isVariadic) {
            return NULL;
        }
        for (int i = 0; i < func->paramCount; i++) {
            addLocal(&compiler, func->params[i]);
        }
        compiler.function->arity = func->paramCount;
        compiler.function->isVariadic = func->isVariadic;
        if (func->paramCount > 0) {
            compiler.function->paramNames = malloc(sizeof(char*) * func->paramCount);
            for (int i = 0; i < func->paramCount; i++) {
                compiler.function->paramNames[i] = strdup(func->params[i]);
            }
        }
    }
    
    int pos = 0;
    while (pos < tokenCount && tokens[pos].kind != TK_EOF) {
        if (!compileStmts(&compiler, tokens, &pos, tokenCount)) {
            break;
        }
    }
    
    emitByte(&compiler, OP_HALT, 0);
    
    if (compiler.hadError) {
        compiledFuncFree(compiler.function);
        return NULL;
    }
    
    return compiler.function;
}
