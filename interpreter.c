#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

// Token types
typedef enum {
    TOKEN_INT,
    TOKEN_FLOAT,
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
    TOKEN_EOF
} TokenType;

// Token structure
typedef struct {
    TokenType type;
    double value;
    char identifier[256];
} Token;

// Global variables
Token currentToken;
char *input;

// Symbol table for variables
typedef struct {
    char name[256];
    double value;
} Variable;

Variable variables[256];
int numVariables = 0;

// Function declarations
void advance();
void error(const char *message, const char *errorToken);
void eat(TokenType type);
double factor();
double term();
double expression();
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
        int isFloat = 0;
        char *start = input;

        // Check for negative sign
        if (*start == '-') {
            start++;
        }

        while (*input && ((*input >= '0' && *input <= '9') || *input == '.')) {
            if (*input == '.') {
                isFloat = 1;
            }
            input++;
        }
        
        if (isFloat) {
            currentToken.type = TOKEN_FLOAT;
            currentToken.value = strtod(start, NULL);
        } else {
            currentToken.type = TOKEN_INT;
            currentToken.value = strtol(start, NULL, 10);
        }
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
double getVariableValue(const char *name) {
    for (int i = 0; i < numVariables; i++) {
        if (strcmp(name, variables[i].name) == 0) {
            return variables[i].value;
        }
    }
    error("Variable not found", name);
    return 0;
}

// Set the value of a variable
void setVariableValue(const char *name, double value) {
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
double factor() {
    if (currentToken.type == TOKEN_INT || currentToken.type == TOKEN_FLOAT) {
        double value = currentToken.value;
        eat(currentToken.type);
        return value;
    } else if (currentToken.type == TOKEN_MINUS) {
        eat(TOKEN_MINUS);
        return -factor();
    } else if (currentToken.type == TOKEN_IDENTIFIER) {
        char identifier[256];
        strcpy(identifier, currentToken.identifier);
        eat(TOKEN_IDENTIFIER);
        return getVariableValue(identifier);
    } else if (currentToken.type == TOKEN_LPAREN) {
        eat(TOKEN_LPAREN);
        double result = expression();
        eat(TOKEN_RPAREN);
        return result;
    } else {
        error("Invalid factor", currentToken.identifier);
        return 0;
    }
}

// Parse a term: multiplication or division of factors
double term() {
    double result = factor();

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
double expression() {
    double result = term();

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

// Parse an assignment statement
void assignment() {
    char identifier[256];
    strcpy(identifier, currentToken.identifier);
    eat(TOKEN_IDENTIFIER);
    eat(TOKEN_ASSIGN);
    double value = expression();
    setVariableValue(identifier, value);
}

// Parse a print statement
void printStatement() {
    eat(TOKEN_PRINT);
    double value = expression();
    printf("%lf\n", value);
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
    char code[] = "var x = -1\n print x*3.14";

    lexer(code);
    program();
    return 0;
}
