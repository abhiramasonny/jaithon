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
    TOKEN_EOF
} TokenType;

// Token structure
typedef struct {
    TokenType type;
    int value;
} Token;

// Global variables
Token currentToken;
char *input;

// Function declarations
void advance();
void error(const char *message);
void eat(TokenType type);
int factor();
int term();
int expression();
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

// Parse a factor: a number or expression within parentheses
int factor() {
    if (currentToken.type == TOKEN_INT) {
        int value = currentToken.value;
        eat(TOKEN_INT);
        return value;
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

// Parse a print statement
void printStatement() {
    eat(TOKEN_PRINT);
    eat(TOKEN_LPAREN);
    int value = expression();
    printf("%d\n", value);
    eat(TOKEN_RPAREN);
}

// Parse the program
void program() {
    while (currentToken.type != TOKEN_EOF) {
        if (currentToken.type == TOKEN_PRINT) {
            printStatement();
        } else {
            error("Invalid statement");
        }
    }
}

// Entry point
int main() {
    char code[] = "print(2 + 3 * 4)\nprint(10 / 2 - 3)";
    lexer(code);
    program();
    return 0;
}
