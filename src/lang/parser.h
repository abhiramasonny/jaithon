#ifndef PARSER_H
#define PARSER_H

#include "../core/runtime.h"
#include "lexer.h"
#include "../vm/vm.h"

typedef Value (*StatementHandler)(Lexer* lex);
typedef Value (*ExprHandler)(Lexer* lex, Value left);

typedef struct {
    int keyword;
    StatementHandler handler;
} StatementEntry;

typedef struct {
    int tokenKind;
    int precedence;
    ExprHandler infixHandler;
} InfixEntry;

void registerStatement(int keyword, StatementHandler handler);
void registerInfix(int tokenKind, int precedence, ExprHandler handler);

Value parseProgram(Lexer* lex);
Value parseStatement(Lexer* lex);
Value parseExpression(Lexer* lex);
Value parseExpressionPrec(Lexer* lex, int minPrec);
Value parsePrimary(Lexer* lex);

Value callValue(Value callee, Value* args, int argc);


bool toBool(Value v);
double toNumber(Value v);


void printCompilationStats(void);

void initParser(void);

uint64_t functionBodyHash(const JaiFunction* f);
bool registerCompiledFunction(JaiFunction* f, CompiledFunc* compiled, uint64_t bodyHash);
CompiledFunc* getCompiledFunc(JaiFunction* f);
bool compileModuleFunctions(Module* mod, bool strict);
bool eagerCompileEnabled(void);
bool eagerCompileStrict(void);

#endif
