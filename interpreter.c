#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>

// Token types
typedef enum {
    TOKEN_INT,
    TOKEN_PLUS,
    TOKEN_MINUS,
    TOKEN_MULTIPLY,
    TOKEN_DIVIDE,
    TOKEN_LPAREN,
    TOKEN_RPAREN,
    TOKEN_PRINT,
    TOKEN_VAR,
    TOKEN_ASSIGN,
    TOKEN_IDENTIFIER,
    TOKEN_EOF,
    TOKEN_OR,
    TOKEN_AND,
    TOKEN_NOT,
    TOKEN_LT,
    TOKEN_GT
} TokenType;

// Token structure
typedef struct {
    TokenType type;
    int value;
    char identifier[256];
} Token;

// Global variables
Token currentToken;
char *input;

// Symbol table for variables
typedef struct {
    char name[256];
    int value;
} Variable;

Variable variables[256];
int numVariables = 0;

// Function declarations
void advance();
void error(const char *message, const char *errorToken);
void eat(TokenType type);
int factor();
int term();
int expression();
bool comparison();
bool logicalOr();
bool logicalAnd();
bool logicalNot();
void assignment();
void printStatement();
void statement();
void program();

// Lexer: Converts input string to tokens
void lexer(char *code) {
    input = code;
    advance();
}

// Advance to the next token
void advance() {
    while (isspace(*input)) {
        input++;
    }

    if (*input == '\0') {
        currentToken.type = TOKEN_EOF;
        return;
    }

    if (*input >= '0' && *input <= '9') {
        currentToken.type = TOKEN_INT;
        currentToken.value = strtol(input, &input, 10);
    } else if (*input == '+') {
        currentToken.type = TOKEN_PLUS;
        input++;
    } else if (*input == '-') {
        currentToken.type = TOKEN_MINUS;
        input++;
    } else if (*input == '*') {
        currentToken.type = TOKEN_MULTIPLY;
        input++;
    } else if (*input == '/') {
        currentToken.type = TOKEN_DIVIDE;
        input++;
    } else if (*input == '(') {
        currentToken.type = TOKEN_LPAREN;
        input++;
    } else if (*input == ')') {
        currentToken.type = TOKEN_RPAREN;
        input++;
    } else if (strncmp(input, "print", 5) == 0) {
        currentToken.type = TOKEN_PRINT;
        input += 5;
    } else if (strncmp(input, "var", 3) == 0) {
        currentToken.type = TOKEN_VAR;
        input += 3;
    } else if (strncmp(input, "or", 2) == 0) {
        currentToken.type = TOKEN_OR;
        input += 2;
    } else if (strncmp(input, "and", 3) == 0) {
        currentToken.type = TOKEN_AND;
        input += 3;
    } else if (strncmp(input, "not", 3) == 0) {
        currentToken.type = TOKEN_NOT;
        input += 3;
    } else if (strncmp(input, "<", 1) == 0) {
        currentToken.type = TOKEN_LT;
        input++;
    } else if (strncmp(input, ">", 1) == 0) {
        currentToken.type = TOKEN_GT;
        input++;
    } else if (isalpha(*input)) {
        currentToken.type = TOKEN_IDENTIFIER;
        int i = 0;
        while (isalpha(*input) || isdigit(*input)) {
            currentToken.identifier[i++] = *input;
            input++;
        }
        currentToken.identifier[i] = '\0';
    } else if (*input == '=') {
        currentToken.type = TOKEN_ASSIGN;
        input++;
    } else {
        error("Invalid token", input);
    }
}

// Report an error
void error(const char *message, const char *errorToken) {
    fprintf(stderr, "Error: %s. Found: %s\n", message, errorToken);
    exit(1);
}

// Ensure the current token matches the expected type and advance to the next token
void eat(TokenType type) {
    if (currentToken.type == type) {
        advance();
    } else {
        char tokenName[256];
        snprintf(tokenName, sizeof(tokenName), "%d", currentToken.type);
        error("Unexpected token", tokenName);
    }
}

// Get the value of a variable
int getVariableValue(const char *name) {
    for (int i = 0; i < numVariables; i++) {
        if (strcmp(name, variables[i].name) == 0) {
            return variables[i].value;
        }
    }
    error("Variable not found", name);
    return 0;
}

// Set the value of a variable
void setVariableValue(const char *name, int value) {
    for (int i = 0; i < numVariables; i++) {
        if (strcmp(name, variables[i].name) == 0) {
            variables[i].value = value;
            return;
        }
    }
    strcpy(variables[numVariables].name, name);
    variables[numVariables].value = value;
    numVariables++;
}

// Parse a factor: a number, variable, or expression within parentheses
int factor() {
    if (currentToken.type == TOKEN_INT) {
        int value = currentToken.value;
        eat(TOKEN_INT);
        return value;
    } else if (currentToken.type == TOKEN_IDENTIFIER) {
        char identifier[256];
        strcpy(identifier, currentToken.identifier);
        eat(TOKEN_IDENTIFIER);
        return getVariableValue(identifier);
    } else if (currentToken.type == TOKEN_LPAREN) {
        eat(TOKEN_LPAREN);
        int result = expression();
        eat(TOKEN_RPAREN);
        return result;
    } else if (currentToken.type == TOKEN_NOT) {
        eat(TOKEN_NOT);
        return logicalNot();
    } else {
        error("Invalid factor", currentToken.identifier);
        return 0;
    }
}

// Parse a term: multiplication or division of factors
int term() {
    int result = factor();

    while (currentToken.type == TOKEN_MULTIPLY || currentToken.type == TOKEN_DIVIDE) {
        if (currentToken.type == TOKEN_MULTIPLY) {
            eat(TOKEN_MULTIPLY);
            result *= factor();
        } else if (currentToken.type == TOKEN_DIVIDE) {
            eat(TOKEN_DIVIDE);
            result /= factor();
        }
    }

    return result;
}

// Parse an expression: addition or subtraction of terms
int expression() {
    int result = term();

    while (currentToken.type == TOKEN_PLUS || currentToken.type == TOKEN_MINUS) {
        if (currentToken.type == TOKEN_PLUS) {
            eat(TOKEN_PLUS);
            result += term();
        } else if (currentToken.type == TOKEN_MINUS) {
            eat(TOKEN_MINUS);
            result -= term();
        }
    }

    return result;
}

// Parse a comparison expression
bool comparison() {
    int left = expression();
    TokenType comparisonType = currentToken.type;
    if (comparisonType == TOKEN_LT || comparisonType == TOKEN_GT) {
        eat(comparisonType);
        int right = expression();
        if (comparisonType == TOKEN_LT) {
            return left < right;
        } else if (comparisonType == TOKEN_GT) {
            return left > right;
        }
    }
    return false;
}

// Parse a logical OR expression
bool logicalOr() {
    bool result = comparison();
    while (currentToken.type == TOKEN_OR) {
        eat(TOKEN_OR);
        result = result || comparison();
    }
    return result;
}

// Parse a logical AND expression
bool logicalAnd() {
    bool result = logicalOr();
    while (currentToken.type == TOKEN_AND) {
        eat(TOKEN_AND);
        result = result && logicalOr();
    }
    return result;
}

// Parse a logical NOT expression
bool logicalNot() {
    if (currentToken.type == TOKEN_NOT) {
        eat(TOKEN_NOT);
        return !logicalNot();
    } else {
        return logicalAnd();
    }
}

// Parse an assignment statement
void assignment() {
    char identifier[256];
    strcpy(identifier, currentToken.identifier);
    eat(TOKEN_IDENTIFIER);
    eat(TOKEN_ASSIGN);
    int value = expression();
    setVariableValue(identifier, value);
}

// Parse a print statement
void printStatement() {
    eat(TOKEN_PRINT);
    int value = expression();
    printf("%d\n", value);
}

// Parse a statement: assignment or print statement
void statement() {
    if (currentToken.type == TOKEN_VAR) {
        eat(TOKEN_VAR);
        assignment();
    } else if (currentToken.type == TOKEN_PRINT) {
        printStatement();
    } else {
        error("Invalid statement", currentToken.identifier);
    }
}

// Parse a program: multiple statements
void program() {
    while (currentToken.type != TOKEN_EOF) {
        statement();
    }
}

int main() {
    char code[] = "";
    lexer(code);
    program();
    return 0;
}
