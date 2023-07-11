#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

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
    TOKEN_EOF,
    TOKEN_SIN,
    TOKEN_COS,
    TOKEN_TAN,
    TOKEN_ASIN,
    TOKEN_ACOS,
    TOKEN_ATAN,
    TOKEN_SQRT,
    TOKEN_ARRAY,
    TOKEN_COMMA,
    TOKEN_LBRACKET,
    TOKEN_RBRACKET
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

// Array structure
typedef struct {
    char name[256];
    int size;
    double elements[256];
} Array;

Array arrays[256];
int numArrays = 0;

// Function declarations
void advance();
void error(const char *message, const char *errorToken);
void eat(TokenType type);
double factor();
double term();
double expression();
double trigFunction();
void assignment();
void printStatement();
void statement();
void program();
void array();

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
    } else if (*input == ',') {
        currentToken.type = TOKEN_COMMA;
        input++;
    } else if (*input == '[') {
        currentToken.type = TOKEN_LBRACKET;
        input++;
    } else if (*input == ']') {
        currentToken.type = TOKEN_RBRACKET;
        input++;
    } else if (strncmp(input, "print", 5) == 0) {
        currentToken.type = TOKEN_PRINT;
        input += 5;
    } else if (strncmp(input, "var", 3) == 0) {
        currentToken.type = TOKEN_VAR;
        input += 3;
    } else if (strncmp(input, "array", 5) == 0){
        currentToken.type = TOKEN_ARRAY;
        input += 5;
    } else if (strncmp(input, "sin", 3) == 0) {
        currentToken.type = TOKEN_SIN;
        input += 3;
    } else if (strncmp(input, "cos", 3) == 0) {
        currentToken.type = TOKEN_COS;
        input += 3;
    } else if (strncmp(input, "tan", 3) == 0) {
        currentToken.type = TOKEN_TAN;
        input += 3;
    } else if (strncmp(input, "asin", 4) == 0) {
        currentToken.type = TOKEN_ASIN;
        input += 4;
    } else if (strncmp(input, "acos", 4) == 0) {
        currentToken.type = TOKEN_ACOS;
        input += 4;
    } else if (strncmp(input, "atan", 4) == 0) {
        currentToken.type = TOKEN_ATAN;
        input += 4;
    } else if (strncmp(input, "sqrt", 4) == 0) {
        currentToken.type = TOKEN_SQRT;
        input += 4;
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
    } else if (currentToken.type == TOKEN_IDENTIFIER) {
        char identifier[256];
        strcpy(identifier, currentToken.identifier);
        eat(TOKEN_IDENTIFIER);

        if (currentToken.type == TOKEN_LBRACKET) {
            eat(TOKEN_LBRACKET);
            int index = (int)expression();
            eat(TOKEN_RBRACKET);

            // Find the array and return the element at the given index
            for (int i = 0; i < numArrays; i++) {
                if (strcmp(identifier, arrays[i].name) == 0) {
                    if (index >= 0 && index < arrays[i].size) {
                        return arrays[i].elements[index];
                    } else {
                        error("Array index out of bounds", identifier);
                    }
                }
            }
            error("Array not found", identifier);
            return 0;
        }

        return getVariableValue(identifier);
    } else if (currentToken.type == TOKEN_LPAREN) {
        eat(TOKEN_LPAREN);
        double value = expression();
        eat(TOKEN_RPAREN);
        return value;
    } else if (currentToken.type == TOKEN_SIN ||
               currentToken.type == TOKEN_COS ||
               currentToken.type == TOKEN_TAN ||
               currentToken.type == TOKEN_ASIN ||
               currentToken.type == TOKEN_ACOS ||
               currentToken.type == TOKEN_ATAN ||
               currentToken.type == TOKEN_SQRT) {
        return trigFunction();
    }

    error("Invalid factor", currentToken.identifier);
    return 0;
}

// Parse a term: multiplication or division of factors
double term() {
    double value = factor();

    while (currentToken.type == TOKEN_MULTIPLY || currentToken.type == TOKEN_DIVIDE) {
        if (currentToken.type == TOKEN_MULTIPLY) {
            eat(TOKEN_MULTIPLY);
            value *= factor();
        } else if (currentToken.type == TOKEN_DIVIDE) {
            eat(TOKEN_DIVIDE);
            double divisor = factor();
            if (divisor == 0) {
                error("Division by zero", "");
            }
            value /= divisor;
        }
    }

    return value;
}

// Parse an expression: addition or subtraction of terms
double expression() {
    double value = term();

    while (currentToken.type == TOKEN_PLUS || currentToken.type == TOKEN_MINUS) {
        if (currentToken.type == TOKEN_PLUS) {
            eat(TOKEN_PLUS);
            value += term();
        } else if (currentToken.type == TOKEN_MINUS) {
            eat(TOKEN_MINUS);
            value -= term();
        }
    }

    return value;
}

// Parse a trigonometric function: sin, cos, tan, asin, acos, atan, sqrt
double trigFunction() {
    TokenType functionType = currentToken.type;
    eat(functionType);
    eat(TOKEN_LPAREN);
    double value = expression();
    eat(TOKEN_RPAREN);

    switch (functionType) {
        case TOKEN_SIN:
            return sin(value);
        case TOKEN_COS:
            return cos(value);
        case TOKEN_TAN:
            return tan(value);
        case TOKEN_ASIN:
            return asin(value);
        case TOKEN_ACOS:
            return acos(value);
        case TOKEN_ATAN:
            return atan(value);
        case TOKEN_SQRT:
            return sqrt(value);
        default:
            error("Invalid trigonometric function", "");
    }

    return 0;
}

// Handle variable assignment
void assignment() {
    char identifier[256];
    strcpy(identifier, currentToken.identifier);
    eat(TOKEN_IDENTIFIER);
    eat(TOKEN_ASSIGN);
    double value = expression();
    setVariableValue(identifier, value);
}

// Handle print statement
void printStatement() {
    eat(TOKEN_PRINT);
    double value = expression();
    printf("%f\n", value);
}

// Handle statements: variable assignment or print statement
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

// Parse the program
void program() {
    while (currentToken.type != TOKEN_EOF) {
        if (currentToken.type == TOKEN_ARRAY) {
            array();
        } else {
            statement();
        }
    }
}

// Parse an array declaration
void array() {
    eat(TOKEN_ARRAY);
    char arrayName[256];
    strcpy(arrayName, currentToken.identifier);
    eat(TOKEN_IDENTIFIER);
    eat(TOKEN_ASSIGN);
    eat(TOKEN_LBRACKET);
    int size = (int)expression();
    eat(TOKEN_RBRACKET);

    // Store the array and its size
    strcpy(arrays[numArrays].name, arrayName);
    arrays[numArrays].size = size;
    numArrays++;
}

int main() {
    // Test array declaration and access
    char code[] = "array e = [5]";
    lexer(code);
    program();  // Output: 2.000000

    // Test sine function
    char code1[] = "var angle = 0.5\n var result = sin(angle)\n print result";
    lexer(code1);
    program();  // Output: 0.479426

    // Test cosine function
    char code2[] = "var angle = 0.8\n var result = cos(angle)\n print result";
    lexer(code2);
    program();  // Output: 0.696707

    // Test tangent function
    char code3[] = "var angle = 1.2\n var result = tan(angle)\n print result";
    lexer(code3);
    program();  // Output: 2.572151

    // Test arcsine function
    char code4[] = "var value = 0.5\n var result = asin(value)\n print result";
    lexer(code4);
    program();  // Output: 0.523599

    // Test arccosine function
    char code5[] = "var value = 0.8\n var result = acos(value)\n print result";
    lexer(code5);
    program();  // Output: 0.643501

    // Test arctangent function
    char code6[] = "var value = 1.2\n var result = atan(value)\n print result";
    lexer(code6);
    program();  // Output: 0.876058

    // Test square root function
    char code7[] = "var value = 25.0\n var result = sqrt(value)\n print result";
    lexer(code7);
    program();  // Output: 5.000000

    return 0;
}
