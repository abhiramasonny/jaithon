#ifndef LEXER_H
#define LEXER_H

#include "../core/runtime.h"

typedef enum {
    TK_EOF = 0,
    TK_NUMBER,
    TK_STRING,
    TK_IDENTIFIER,
    TK_PLUS,
    TK_MINUS,
    TK_STAR,
    TK_SLASH,
    TK_PERCENT,
    TK_CARET,
    TK_BANG,
    TK_LPAREN,
    TK_RPAREN,
    TK_LBRACKET,
    TK_RBRACKET,
    TK_LBRACE,
    TK_RBRACE,
    TK_COMMA,
    TK_DOT,
    TK_COLON,
    TK_EQUALS,
    TK_EQ_EQ,
    TK_NE,
    TK_GT,
    TK_LT,
    TK_GE,
    TK_LE,
    TK_NEWLINE,
    TK_KEYWORD = 100
} TokenKind;

typedef struct {
    int kind;
    double numValue;
    char strValue[MAX_NAME_LEN];
    int line;
} Token;

typedef struct {
    const char* source;
    const char* current;
    const char* start;
    int line;
    Token currentToken;
    Token peekToken;
    bool hasPeek;
} Lexer;

void lexerInit(Lexer* lex, const char* source);
Token lexerNext(Lexer* lex);
Token lexerPeek(Lexer* lex);
void lexerExpect(Lexer* lex, int kind);
bool lexerMatch(Lexer* lex, int kind);
bool lexerCheck(Lexer* lex, int kind);
const char* tokenKindName(int kind);

void registerBuiltinKeywords(void);

int getKW_VAR(void);
int getKW_PRINT(void);
int getKW_IF(void);
int getKW_THEN(void);
int getKW_ELSE(void);
int getKW_DO(void);
int getKW_WHILE(void);
int getKW_LOOP(void);
int getKW_FUNC(void);
int getKW_RETURN(void);
int getKW_END(void);
int getKW_IMPORT(void);
int getKW_FROM(void);
int getKW_AS(void);
int getKW_AND(void);
int getKW_OR(void);
int getKW_NOT(void);
int getKW_XOR(void);
int getKW_TRUE(void);
int getKW_FALSE(void);
int getKW_NULL(void);
int getKW_INPUT(void);
int getKW_BREAK(void);
int getKW_SYSTEM(void);
int getKW_CLASS(void);
int getKW_NEW(void);
int getKW_EXTENDS(void);
int getKW_SELF(void);
int getKW_NAMESPACE(void);

#endif
