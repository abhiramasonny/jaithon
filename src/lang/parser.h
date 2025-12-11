#ifndef PARSER_H
#define PARSER_H

#include "../core/runtime.h"
#include "lexer.h"

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

void initParser(void);

#endif
