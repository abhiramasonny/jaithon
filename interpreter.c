#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
    TOKEN_FOR,
    TOKEN_TO,
    TOKEN_DO,
    TOKEN_EOF
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
void error(const char *message);
void eat(TokenType type);
int factor();
int term();
int expression();
void assignment();
void printStatement();
void forLoop();
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
    } else if (strncmp(input, "for", 3) == 0) {
        currentToken.type = TOKEN_FOR;
        input += 3;
    } else if (strncmp(input, "to", 2) == 0) {
        currentToken.type = TOKEN_TO;
        input += 2;
    } else if (strncmp(input, "do", 2) == 0) {
        currentToken.type = TOKEN_DO;
        input += 2;
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
        error("Invalid token");
    }
}

// Report an error
void error(const char *message) {
    fprintf(stderr, "Error: %s\n", message);
    exit(1);
}

// Ensure the current token matches the expected type and advance to the next token
void eat(TokenType type) {
    if (currentToken.type == type) {
        advance();
    } else {
        error("Unexpected token");
    }
}

// Get the value of a variable
int getVariableValue(const char *name) {
    for (int i = 0; i < numVariables; i++) {
        if (strcmp(name, variables[i].name) == 0) {
            return variables[i].value;
        }
    }
    error("Variable not found");
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
    } else {
        error("Invalid factor");
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
    eat(TOKEN_LPAREN);
    if (currentToken.type == TOKEN_IDENTIFIER) {
        char identifier[256];
        strcpy(identifier, currentToken.identifier);
        eat(TOKEN_IDENTIFIER);
        int value = getVariableValue(identifier);
        printf("%d\n", value);
    } else {
        int value = expression();
        printf("%d\n", value);
    }
    eat(TOKEN_RPAREN);
}

// Parse a for loop
void forLoop() {
    eat(TOKEN_FOR);
    char identifier[256];
    strcpy(identifier, currentToken.identifier);
    eat(TOKEN_IDENTIFIER);
    eat(TOKEN_ASSIGN);
    int start = expression();
    setVariableValue(identifier, start);
    eat(TOKEN_TO);
    int end = expression();
    eat(TOKEN_DO);
    while (getVariableValue(identifier) <= end) {
        program();
        setVariableValue(identifier, getVariableValue(identifier) + 1);
    }
}

// Parse the program
void program() {
    while (currentToken.type != TOKEN_EOF) {
        if (currentToken.type == TOKEN_PRINT) {
            printStatement();
        } else if (currentToken.type == TOKEN_VAR) {
            eat(TOKEN_VAR);
            assignment();
        } else if (currentToken.type == TOKEN_FOR) {
            forLoop();
        } else {
            error("Invalid statement");
        }
    }
}

// Entry point
int main() {
    char code[] = "for i = 1 to 100 do\n  print(i)\n";
    lexer(code);
    program();
    return 0;
}
