#include "lexer.h"
#include <ctype.h>
#include <stdio.h>

static bool isAtEnd(Lexer* lex) {
    return *lex->current == '\0';
}

static char advance(Lexer* lex) {
    return *lex->current++;
}

static char peek(Lexer* lex) {
    return *lex->current;
}

static char peekNext(Lexer* lex) {
    if (isAtEnd(lex)) return '\0';
    return lex->current[1];
}

static void skipWhitespace(Lexer* lex) {
    for (;;) {
        char c = peek(lex);
        switch (c) {
            case ' ':
            case '\t':
            case '\r':
                advance(lex);
                break;
            case '\n':
                return;
            case '#':
                while (peek(lex) != '\n' && !isAtEnd(lex)) {
                    advance(lex);
                }
                break;
            default:
                return;
        }
    }
}

static Token makeToken(Lexer* lex, int kind) {
    Token t;
    t.kind = kind;
    t.numValue = 0;
    t.strValue[0] = '\0';
    t.line = lex->line;
    
    int len = (int)(lex->current - lex->start);
    if (len >= MAX_NAME_LEN) len = MAX_NAME_LEN - 1;
    strncpy(t.strValue, lex->start, len);
    t.strValue[len] = '\0';
    
    return t;
}

static Token makeNumberToken(Lexer* lex, double value) {
    Token t = makeToken(lex, TK_NUMBER);
    t.numValue = value;
    return t;
}

static Token makeStringToken(Lexer* lex, const char* str) {
    Token t = makeToken(lex, TK_STRING);
    strncpy(t.strValue, str, MAX_NAME_LEN - 1);
    return t;
}

static Token errorToken(Lexer* lex, const char* msg) {
    Token t;
    t.kind = TK_EOF;
    t.numValue = 0;
    strncpy(t.strValue, msg, MAX_NAME_LEN - 1);
    t.line = lex->line;
    return t;
}

static Token scanNumber(Lexer* lex) {
    while (isdigit(peek(lex))) advance(lex);
    
    if (peek(lex) == '.' && isdigit(peekNext(lex))) {
        advance(lex);
        while (isdigit(peek(lex))) advance(lex);
    }

    if (peek(lex) == 'e' || peek(lex) == 'E') {
        char next = peekNext(lex);
        if (isdigit(next)) {
            advance(lex);
            while (isdigit(peek(lex))) advance(lex);
        } else if (next == '+' || next == '-') {
            char nextNext = lex->current[2];
            if (isdigit(nextNext)) {
                advance(lex);
                advance(lex);
                while (isdigit(peek(lex))) advance(lex);
            }
        }
    }
    
    char buf[64];
    int len = (int)(lex->current - lex->start);
    if (len >= 64) len = 63;
    strncpy(buf, lex->start, len);
    buf[len] = '\0';
    
    return makeNumberToken(lex, strtod(buf, NULL));
}

static Token scanString(Lexer* lex) {
    char buf[MAX_NAME_LEN];
    int i = 0;
    
    while (peek(lex) != '"' && !isAtEnd(lex)) {
        if (peek(lex) == '\n') lex->line++;
        if (peek(lex) == '\\' && peekNext(lex) != '\0') {
            advance(lex);
            switch (peek(lex)) {
                case 'n': buf[i++] = '\n'; break;
                case 't': buf[i++] = '\t'; break;
                case '\\': buf[i++] = '\\'; break;
                case '"': buf[i++] = '"'; break;
                default: buf[i++] = peek(lex);
            }
            advance(lex);
        } else {
            if (i < MAX_NAME_LEN - 1) buf[i++] = peek(lex);
            advance(lex);
        }
    }
    
    if (isAtEnd(lex)) {
        return errorToken(lex, "Unterminated string");
    }
    
    advance(lex);
    buf[i] = '\0';
    return makeStringToken(lex, buf);
}

static Token scanIdentifier(Lexer* lex) {
    while (isalnum(peek(lex)) || peek(lex) == '_') {
        advance(lex);
    }
    
    char buf[MAX_NAME_LEN];
    int len = (int)(lex->current - lex->start);
    if (len >= MAX_NAME_LEN) len = MAX_NAME_LEN - 1;
    strncpy(buf, lex->start, len);
    buf[len] = '\0';
    
    int kwType = lookupKeyword(buf);
    if (kwType >= 0) {
        Token t = makeToken(lex, kwType);
        strcpy(t.strValue, buf);
        return t;
    }
    
    return makeToken(lex, TK_IDENTIFIER);
}

static Token scanToken(Lexer* lex) {
    skipWhitespace(lex);
    lex->start = lex->current;
    
    if (isAtEnd(lex)) {
        return makeToken(lex, TK_EOF);
    }
    
    char c = advance(lex);
    
    if (isdigit(c)) return scanNumber(lex);
    if (isalpha(c) || c == '_') return scanIdentifier(lex);
    
    switch (c) {
        case '(': return makeToken(lex, TK_LPAREN);
        case ')': return makeToken(lex, TK_RPAREN);
        case '[': return makeToken(lex, TK_LBRACKET);
        case ']': return makeToken(lex, TK_RBRACKET);
        case '{': return makeToken(lex, TK_LBRACE);
        case '}': return makeToken(lex, TK_RBRACE);
        case ',': return makeToken(lex, TK_COMMA);
        case '.': return makeToken(lex, TK_DOT);
        case ':': return makeToken(lex, TK_COLON);
        case '+': return makeToken(lex, TK_PLUS);
        case '-': return makeToken(lex, TK_MINUS);
        case '*': return makeToken(lex, TK_STAR);
        case '/': return makeToken(lex, TK_SLASH);
        case '%': return makeToken(lex, TK_PERCENT);
        case '^': return makeToken(lex, TK_CARET);
        case '!': 
            if (peek(lex) == '=') {
                advance(lex);
                return makeToken(lex, TK_NE);
            }
            return makeToken(lex, TK_BANG);
        case '=': 
            if (peek(lex) == '=') {
                advance(lex);
                return makeToken(lex, TK_EQ_EQ);
            }
            return makeToken(lex, TK_EQUALS);
        case '>':  
            if (peek(lex) == '=') {
                advance(lex);
                return makeToken(lex, TK_GE);
            }
            return makeToken(lex, TK_GT);
        case '<':
            if (peek(lex) == '=') {
                advance(lex);
                return makeToken(lex, TK_LE);
            }
            return makeToken(lex, TK_LT);
        case '\n':
            lex->line++;
            return makeToken(lex, TK_NEWLINE);
        case '"': return scanString(lex);
    }
    
    return errorToken(lex, "Unexpected character");
}

void lexerInit(Lexer* lex, const char* source) {
    lex->source = source;
    lex->current = source;
    lex->start = source;
    lex->line = 1;
    lex->hasPeek = false;
    lex->currentToken = scanToken(lex);
}

Token lexerNext(Lexer* lex) {
    if (lex->hasPeek) {
        lex->currentToken = lex->peekToken;
        lex->hasPeek = false;
    } else {
        lex->currentToken = scanToken(lex);
    }
    runtime.lineNumber = lex->currentToken.line;
    return lex->currentToken;
}

Token lexerPeek(Lexer* lex) {
    if (!lex->hasPeek) {
        lex->peekToken = scanToken(lex);
        lex->hasPeek = true;
    }
    return lex->peekToken;
}

void lexerExpect(Lexer* lex, int kind) {
    if (lex->currentToken.kind != kind) {
        runtimeError("Expected %s, got %s", tokenKindName(kind), tokenKindName(lex->currentToken.kind));
    }
    lexerNext(lex);
}

bool lexerMatch(Lexer* lex, int kind) {
    if (lex->currentToken.kind == kind) {
        lexerNext(lex);
        return true;
    }
    return false;
}

bool lexerCheck(Lexer* lex, int kind) {
    return lex->currentToken.kind == kind;
}

const char* tokenKindName(int kind) {
    switch (kind) {
        case TK_EOF: return "EOF";
        case TK_NUMBER: return "number";
        case TK_STRING: return "string";
        case TK_IDENTIFIER: return "identifier";
        case TK_PLUS: return "+";
        case TK_MINUS: return "-";
        case TK_STAR: return "*";
        case TK_SLASH: return "/";
        case TK_PERCENT: return "%";
        case TK_CARET: return "^";
        case TK_BANG: return "!";
        case TK_LPAREN: return "(";
        case TK_RPAREN: return ")";
        case TK_LBRACKET: return "[";
        case TK_RBRACKET: return "]";
        case TK_LBRACE: return "{";
        case TK_RBRACE: return "}";
        case TK_COMMA: return ",";
        case TK_DOT: return ".";
        case TK_COLON: return ":";
        case TK_EQUALS: return "=";
        case TK_EQ_EQ: return "==";
        case TK_NE: return "!=";
        case TK_GT: return ">";
        case TK_LT: return "<";
        case TK_NEWLINE: return "newline";
        default: 
            if (kind >= TK_KEYWORD) {
                for (int i = 0; i < runtime.keywords.count; i++) {
                    if (runtime.keywords.entries[i].tokenType == kind) {
                        return runtime.keywords.entries[i].keyword;
                    }
                }
            }
            return "unknown";
    }
}

static int KW_VAR, KW_PRINT, KW_IF, KW_THEN, KW_ELSE, KW_DO, KW_WHILE, KW_LOOP;
static int KW_FUNC, KW_RETURN, KW_END, KW_IMPORT, KW_FROM, KW_AS;
static int KW_AND, KW_OR, KW_NOT, KW_XOR;
static int KW_TRUE, KW_FALSE, KW_NULL;
static int KW_INPUT, KW_BREAK, KW_SYSTEM;
static int KW_CLASS, KW_NEW, KW_EXTENDS, KW_SELF;
static int KW_NAMESPACE;
static int KW_PUBLIC, KW_PRIVATE, KW_PROTECTED, KW_STATIC, KW_IN, KW_VOID;
static int KW_INT, KW_DOUBLE, KW_FLOAT, KW_STRING, KW_CHAR, KW_LONG, KW_SHORT, KW_BYTE, KW_BOOL;
static int KW_DEL;

void registerBuiltinKeywords(void) {
    KW_VAR = registerKeyword("var");
    KW_PRINT = registerKeyword("print");
    KW_IF = registerKeyword("if");
    KW_THEN = registerKeyword("then");
    KW_ELSE = registerKeyword("else");
    KW_DO = registerKeyword("do");
    KW_WHILE = registerKeyword("while");
    KW_LOOP = registerKeyword("loop");
    KW_FUNC = registerKeyword("func");
    KW_RETURN = registerKeyword("return");
    KW_END = registerKeyword("end");
    KW_IMPORT = registerKeyword("import");
    KW_FROM = registerKeyword("from");
    KW_AS = registerKeyword("as");
    KW_AND = registerKeyword("and");
    KW_OR = registerKeyword("or");
    KW_NOT = registerKeyword("not");
    KW_XOR = registerKeyword("xor");
    KW_TRUE = registerKeyword("true");
    KW_FALSE = registerKeyword("false");
    KW_NULL = registerKeyword("null");
    KW_INPUT = registerKeyword("input");
    KW_BREAK = registerKeyword("break");
    KW_SYSTEM = registerKeyword("system");
    KW_CLASS = registerKeyword("class");
    KW_NEW = registerKeyword("new");
    KW_EXTENDS = registerKeyword("extends");
    KW_SELF = registerKeyword("self");
    KW_NAMESPACE = registerKeyword("namespace");
    KW_PUBLIC = registerKeyword("public");
    KW_PRIVATE = registerKeyword("private");
    KW_PROTECTED = registerKeyword("protected");
    KW_STATIC = registerKeyword("static");
    KW_IN = registerKeyword("in");
    KW_VOID = registerKeyword("void");
    KW_INT = registerKeyword("int");
    KW_DOUBLE = registerKeyword("double");
    KW_FLOAT = registerKeyword("float");
    KW_STRING = registerKeyword("string");
    KW_CHAR = registerKeyword("char");
    KW_LONG = registerKeyword("long");
    KW_SHORT = registerKeyword("short");
    KW_BYTE = registerKeyword("byte");
    KW_BOOL = registerKeyword("bool");
    KW_DEL = registerKeyword("del");
}

int getKW_VAR(void) { return KW_VAR; }
int getKW_PRINT(void) { return KW_PRINT; }
int getKW_IF(void) { return KW_IF; }
int getKW_THEN(void) { return KW_THEN; }
int getKW_ELSE(void) { return KW_ELSE; }
int getKW_DO(void) { return KW_DO; }
int getKW_WHILE(void) { return KW_WHILE; }
int getKW_LOOP(void) { return KW_LOOP; }
int getKW_FUNC(void) { return KW_FUNC; }
int getKW_RETURN(void) { return KW_RETURN; }
int getKW_END(void) { return KW_END; }
int getKW_IMPORT(void) { return KW_IMPORT; }
int getKW_FROM(void) { return KW_FROM; }
int getKW_AS(void) { return KW_AS; }
int getKW_AND(void) { return KW_AND; }
int getKW_OR(void) { return KW_OR; }
int getKW_NOT(void) { return KW_NOT; }
int getKW_XOR(void) { return KW_XOR; }
int getKW_TRUE(void) { return KW_TRUE; }
int getKW_FALSE(void) { return KW_FALSE; }
int getKW_NULL(void) { return KW_NULL; }
int getKW_INPUT(void) { return KW_INPUT; }
int getKW_BREAK(void) { return KW_BREAK; }
int getKW_SYSTEM(void) { return KW_SYSTEM; }
int getKW_CLASS(void) { return KW_CLASS; }
int getKW_NEW(void) { return KW_NEW; }
int getKW_EXTENDS(void) { return KW_EXTENDS; }
int getKW_SELF(void) { return KW_SELF; }
int getKW_NAMESPACE(void) { return KW_NAMESPACE; }
int getKW_PUBLIC(void) { return KW_PUBLIC; }
int getKW_PRIVATE(void) { return KW_PRIVATE; }
int getKW_PROTECTED(void) { return KW_PROTECTED; }
int getKW_STATIC(void) { return KW_STATIC; }
int getKW_IN(void) { return KW_IN; }
int getKW_VOID(void) { return KW_VOID; }
int getKW_INT(void) { return KW_INT; }
int getKW_DOUBLE(void) { return KW_DOUBLE; }
int getKW_FLOAT(void) { return KW_FLOAT; }
int getKW_STRING(void) { return KW_STRING; }
int getKW_CHAR(void) { return KW_CHAR; }
int getKW_LONG(void) { return KW_LONG; }
int getKW_SHORT(void) { return KW_SHORT; }
int getKW_BYTE(void) { return KW_BYTE; }
int getKW_BOOL(void) { return KW_BOOL; }
int getKW_DEL(void) { return KW_DEL; }

int tokenizeSource(const char* source, Token** outTokens) {
    Lexer lex;
    lexerInit(&lex, source);
    
    int capacity = 256;
    Token* tokens = malloc(sizeof(Token) * capacity);
    int count = 0;
    
    tokens[count++] = lex.currentToken;
    
    if (lex.currentToken.kind == TK_EOF) {
        *outTokens = tokens;
        return count;
    }
    
    while (true) {
        Token t = lexerNext(&lex);
        
        if (count >= capacity) {
            capacity *= 2;
            tokens = realloc(tokens, sizeof(Token) * capacity);
        }
        tokens[count++] = t;
        
        if (t.kind == TK_EOF) break;
    }
    
    *outTokens = tokens;
    return count;
}
