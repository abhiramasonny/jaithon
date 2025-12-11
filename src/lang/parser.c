#include "parser.h"
#include <time.h>
#include <sys/time.h>

#define MAX_STATEMENTS 64
#define MAX_INFIXES 64
#define MAX_CALL_ARGS 256

static StatementEntry statements[MAX_STATEMENTS];
static int statementCount = 0;

static InfixEntry infixes[MAX_INFIXES];
static int infixCount = 0;

static Value returnValue;
static bool hasReturn = false;

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

static double toNumber(Value v) {
    switch (v.type) {
        case VAL_NUMBER: return v.as.number;
        case VAL_BOOL: return v.as.boolean ? 1.0 : 0.0;
        case VAL_STRING: return strlen(v.as.string) > 0 ? 1.0 : 0.0;
        default: return 0.0;
    }
}

static bool toBool(Value v) {
    switch (v.type) {
        case VAL_NUMBER: return v.as.number != 0;
        case VAL_BOOL: return v.as.boolean;
        case VAL_STRING: return strlen(v.as.string) > 0;
        case VAL_NULL: return false;
        default: return true;
    }
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
            
            if (!lexerCheck(lex, TK_RPAREN)) {
                do {
                    if (argc >= MAX_CALL_ARGS) {
                        runtimeError("Too many constructor arguments");
                        return makeNull();
                    }
                    args[argc++] = parseExpression(lex);
                } while (lexerMatch(lex, TK_COMMA));
            }
            lexerExpect(lex, TK_RPAREN);
            
            if (cls->constructor) {
                callValue(makeFunction(cls->constructor), args, argc);
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
                
                if (result.type == VAL_OBJECT) {
                    if (lexerCheck(lex, TK_LPAREN)) {
                        JaiFunction* method = objectGetMethod(result.as.object, fieldName);
                        if (!method) {
                            runtimeError("Object has no method: %s", fieldName);
                            return makeNull();
                        }
                        lexerNext(lex);
                        Value args[MAX_CALL_ARGS];
                        args[0] = result;
                        int argc = 1;
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
    if (left.type == VAL_STRING || right.type == VAL_STRING) {
        char buf[MAX_NAME_LEN * 2];
        if (left.type == VAL_STRING) {
            strcpy(buf, left.as.string);
        } else {
            snprintf(buf, sizeof(buf), "%g", left.as.number);
        }
        if (right.type == VAL_STRING) {
            strcat(buf, right.as.string);
        } else {
            char tmp[64];
            snprintf(tmp, sizeof(tmp), "%g", right.as.number);
            strcat(buf, tmp);
        }
        return makeString(buf);
    }
    return makeNumber(toNumber(left) + toNumber(right));
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

    if (lexerCheck(lex, TK_LPAREN)) {
        lexerNext(lex);
        Value args[MAX_CALL_ARGS];
        int argc = 0;

        args[argc++] = left;
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
        
        if (left.type != VAL_OBJECT) {
            runtimeError("Cannot call method on non-object");
            return makeNull();
        }
        JaiFunction* method = objectGetMethod(left.as.object, fieldName);
        if (!method) {
            runtimeError("Object has no method: %s", fieldName);
            return makeNull();
        }
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
    if (left.type != right.type) return makeBool(false);
    switch (left.type) {
        case VAL_NUMBER: return makeBool(left.as.number == right.as.number);
        case VAL_BOOL: return makeBool(left.as.boolean == right.as.boolean);
        case VAL_STRING: return makeBool(strcmp(left.as.string, right.as.string) == 0);
        case VAL_NULL: return makeBool(true);
        default: return makeBool(false);
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
    
    lexerExpect(lex, TK_EQUALS);
    
    Value val = parseExpression(lex);
    setVariable(name, val);
    
    if (runtime.debug) {
        printf("Set %s = ", name);
        if (val.type == VAL_NUMBER) printf("%g\n", val.as.number);
        else if (val.type == VAL_STRING) printf("\"%s\"\n", val.as.string);
        else if (val.type == VAL_BOOL) printf("%s\n", val.as.boolean ? "true" : "false");
        else printf("null\n");
    }
    
    return val;
}

static Value stmtPrint(Lexer* lex) {
    lexerExpect(lex, getKW_PRINT());
    if (runtime.debug) {
        printf("DEBUG print expr starts with token %s\n", tokenKindName(lex->currentToken.kind));
    }
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
        case VAL_STRING:
            printf("%s\n", val.as.string);
            break;
        case VAL_BOOL:
            printf("%s\n", val.as.boolean ? "true" : "false");
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
    
    lexerMatch(lex, getKW_THEN()); //not required (should prolly remove ltr)
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

static Value stmtWhile(Lexer* lex) {
    const char* loopStart = lex->current;
    Token startToken = lex->currentToken;
    int startLine = lex->line;
    
    lexerExpect(lex, getKW_WHILE());
    
    for (;;) {
        Value cond = parseExpression(lex);
        
        lexerMatch(lex, getKW_DO());
        skipNewlines(lex);
        
        if (!toBool(cond)) {
            int depth = 1;
            while (depth > 0 && !lexerCheck(lex, TK_EOF)) {
                if (lex->currentToken.kind == getKW_WHILE() || 
                    lex->currentToken.kind == getKW_IF() ||
                    lex->currentToken.kind == getKW_FUNC()) {
                    depth++;
                }
                if (lex->currentToken.kind == getKW_END()) {
                    depth--;
                    if (depth == 0) break;
                }
                lexerNext(lex);
            }
            lexerExpect(lex, getKW_END());
            break;
        }
        
        while (!lexerCheck(lex, getKW_END()) && !lexerCheck(lex, TK_EOF)) {
            parseStatement(lex);
            skipNewlines(lex);
            if (hasReturn) return returnValue;
        }
        
        lex->current = loopStart;
        lex->line = startLine;
        lex->currentToken = startToken;
        lex->hasPeek = false;
        lexerNext(lex);
    }
    
    return makeNull();
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

// class ClassName [extends ParentClass]
//     var field1
//     var field2
//     
//     func init(self, args)
//         self.field1 = value
//     end
//     
//     func method(self, args)
//         ...
//     end
// end
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

static Value stmtImport(Lexer* lex) {
    lexerExpect(lex, getKW_IMPORT());
    
    if (!lexerCheck(lex, TK_IDENTIFIER)) {
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
        if (!lexerCheck(lex, TK_IDENTIFIER)) {
            runtimeError("Expected module name after '/'");
            return makeNull();
        }
    }
    
    char path[1024];
    if (snprintf(path, sizeof(path), "%s.jai", modulePath) >= (int)sizeof(path)) {
        runtimeError("Module path too long");
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
    lexerNext(&modLex);
    parseProgram(&modLex);
    
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

Value parseStatement(Lexer* lex) {
    skipNewlines(lex);
    
    if (lexerCheck(lex, TK_EOF)) {
        return makeNull();
    }
    
    if (runtime.debug) {
        printf("DEBUG statement token: %s (line %d)\n", tokenKindName(lex->currentToken.kind), lex->currentToken.line);
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
                        args[0] = result;
                        int argc = 1;
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
    if (callee.type == VAL_NATIVE_FUNC) {
        return callee.as.nativeFunc(args, argc);
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
                return makeNull();
            }
        } else if (argc != f->paramCount) {
            runtimeError("Expected %d arguments, got %d", f->paramCount, argc);
            return makeNull();
        }
        
        Module* oldMod = runtime.currentModule;
        Module* funcMod = createModule("__call__", "");
        runtime.currentModule = funcMod;
        
        if (f->isVariadic) {
            int regularParams = f->paramCount - 1;
            for (int i = 0; i < regularParams; i++) {
                setVariable(f->params[i], args[i]);
            }
            Value variadicArray = makeArray(argc - regularParams);
            for (int i = regularParams; i < argc; i++) {
                arrayPush(variadicArray.as.array, args[i]);
            }
            setVariable(f->params[regularParams], variadicArray);
        } else {
            for (int i = 0; i < argc; i++) {
                setVariable(f->params[i], args[i]);
            }
        }
        
        Lexer bodyLex;
        lexerInit(&bodyLex, f->body);
        
        hasReturn = false;
        Value result = parseProgram(&bodyLex);
        
        if (hasReturn) {
            result = returnValue;
            hasReturn = false;
        }
        
        runtime.currentModule = oldMod;
        runtime.moduleCount--;
        
        if (runtime.callStackSize > 0) {
            runtime.callStackSize--;
        }
        
        return result;
    }
    
    runtimeError("Cannot call non-function value");
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
        case VAL_BOOL: strcpy(buf, args[0].as.boolean ? "true" : "false"); break;
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

static Value nativeType(Value* args, int argc) {
    if (argc < 1) return makeString("null");
    switch (args[0].type) {
        case VAL_NUMBER: return makeString("number");
        case VAL_STRING: return makeString("string");
        case VAL_BOOL: return makeString("bool");
        case VAL_FUNCTION: return makeString("function");
        case VAL_NATIVE_FUNC: return makeString("native");
        case VAL_CELL: return makeString("cell");
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
    registerStatement(getKW_CLASS(), stmtClass);
    
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
    registerInfix(getKW_EQ(), 4, handleEq);
    registerInfix(getKW_AND(), 3, handleAnd);
    registerInfix(getKW_OR(), 2, handleOr);
    
    srand(time(NULL));
    
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
    setVariable("_pop", makeNativeFunc(nativePop));
    setVariable("_get", makeNativeFunc(nativeGet));
    setVariable("_set", makeNativeFunc(nativeSet));
    setVariable("_alen", makeNativeFunc(nativeAlen));
}
