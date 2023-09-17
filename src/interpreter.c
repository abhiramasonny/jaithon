#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <sys/time.h>
#include <unistd.h>
#include <getopt.h>
#include <readline/readline.h>
#include <readline/history.h>
#include <stdlib.h>

#define MAX_IMPORTED_FILES 2048
#define MAX_FILENAME_LEN 256
#define FILE_EXTENSION ".jai"
#define LOG_FILE "config/log.txt"
#define VERSION_FILE "config/version.txt"

char importedFiles[MAX_IMPORTED_FILES][256];
int numImportedFiles = 0;
int lines = 1;
bool debug = false;
int auto_extension = 1;
bool log_enabled = false;
bool shell_mode = false;

// Big list of tokens
typedef enum {
    TOKEN_EOF,
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
    TOKEN_STRING,
    TOKEN_ASSIGN,
    TOKEN_IDENTIFIER,
    TOKEN_MATH,
    TOKEN_SIN,
    TOKEN_COS,
    TOKEN_TAN,
    TOKEN_ASIN,
    TOKEN_ACOS,
    TOKEN_ATAN,
    TOKEN_SQRT,
    TOKEN_DEGREES,
    TOKEN_QUADRATIC,
    TOKEN_PYTHAGOREAN,
    TOKEN_FACTORIAL,
    TOKEN_EXP,
    TOKEN_ROOT,
    TOKEN_BINARY,
    TOKEN_CONV,
    TOKEN_BADD,
    TOKEN_COMMA,
    TOKEN_DOT,
    TOKEN_INPUT,
    TOKEN_TIME,
    TOKEN_IMPORT,
    TOKEN_GREATER_THAN,
    TOKEN_LESS_THAN,
    TOKEN_EQ,
    TOKEN_IF,
    TOKEN_THEN,
    TOKEN_DO,
    TOKEN_AND,
    TOKEN_OR,
    TOKEN_NOT,
    TOKEN_XOR,
    TOKEN_DIST,
    TOKEN_WHILE,
    TOKEN_DONE,
    TOKEN_BREAK,
    TOKEN_ROUND,
    TOKEN_COMP,
    TOKEN_MOD,
    TOKEN_RAND,
    TOKEN_UNIFORM,
    TOKEN_WRITE,
    TOKEN_READ,
    TOKEN_SYSTEM,
} TokenType;

typedef struct {
    TokenType type;
    double value;
    char identifier[0];
    char string[0];
} Token;

Token currentToken;
char *input;
typedef struct {
    char name[2048];
    double value;
    char stringValue[2048]; 
} Variable;

Variable variables[2048];
int numVariables = 0;

//predefined functions
void lexer(char *code);
void skipToEndOfInput();
int isDelimiter(char c);
void advance();
void error(const char *message, const char *errorToken);
void eat(TokenType type);
void skipToEnd();
void ifStatement();
void whileStatement();
void setStringValue(const char *name, const char *value);
const char* getStringValue(const char *name);
int areEqual(const char *str1, const char *str2);
void importFile(const char *filename);
void writeToFile();
char* readFromFile();
double getVariableValue(const char *name);
void setVariableValue(const char *name, double value);
double getTimeInSeconds();
double timeFunction();
double quadFunction();
int factorial(int n);
double expon(double base, double p);
double nthRoot();
double distance(double ax, double ay, double bx, double by);
double pythagoreanTheorem();
double decimalToBinary(double n);
double binaryToTen(double n);
double binaryAdd();
double binaryConversion();
double factor();
double term();
double expression();
double argsMath();
void assignment();
void printStatement();
void varInputStatement();
void statement();
void program();
void executeFile(const char *filename);
void shellMode();
void displayVersion();
void displayHelp();
void writeLog(const char *message);


void lexer(char *code) {
    input = code;
    advance();
}

void skipToEndOfInput() {
    while (currentToken.type != TOKEN_EOF) {
        advance();
    }
}
int isDelimiter(char c) {
    const char validDelimiters[] = " \t\n(),+-*/%<>=!^.#";
    for (int i = 0; validDelimiters[i]; i++) {
        if (c == validDelimiters[i]) {
            return 1;
        }
    }
    return 0;
}

void advance() {
    while (isspace(*input)) {
        if (*input == '\n') {
            lines++;
        }
        input++;
    } 
    if (*input == '#') {
        while (*input && *input != '\n') {
            input++;
        }
        advance();
        return;
    }
    if (*input == '\0') {
        currentToken.type = TOKEN_EOF;
        return;
    }

    if (*input >= '0' && *input <= '9') {
        int isFloat = 0;
        char *start = input;

        if (*start == '-') {start++;}
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
    } else if (*input == '%') {
        currentToken.type = TOKEN_MOD;
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
    } else if (*input == '.') {
        currentToken.type = TOKEN_DOT;
        input++;
    } else if (*input == '>') {
        currentToken.type = TOKEN_GREATER_THAN;
        input++;
    } else if (*input == '<') {
        currentToken.type = TOKEN_LESS_THAN;
        input++;
    } else if (*input == '=') {
        currentToken.type = TOKEN_ASSIGN;
        input++;
    } else if (*input == '!') {
        currentToken.type = TOKEN_FACTORIAL;
        input++;
    } else if (*input == '^') {
        currentToken.type = TOKEN_EXP;
        input++;
    } else if ((strncmp(input, "if", 2) == 0) && isDelimiter(input[2])) {
        currentToken.type = TOKEN_IF;
        input += 2;
    } else if ((strncmp(input, "do", 2) == 0) && isDelimiter(input[2])) {
        currentToken.type = TOKEN_DO;
        input += 2;
    } else if ((strncmp(input, "then", 4) == 0) && isDelimiter(input[4])) {
        currentToken.type = TOKEN_THEN;
        input += 4;
    } else if ((strncmp(input, "eq", 2) == 0)) {
        currentToken.type = TOKEN_EQ;
        input += 2;
    } else if ((strncmp(input, "or", 2) == 0)) {
        currentToken.type = TOKEN_OR;
        input += 2;
    } else if ((strncmp(input, "system", 6) == 0) && isDelimiter(input[6])) {
        currentToken.type = TOKEN_SYSTEM;
        input += 6;
    } else if (strncmp(input, "write", 5) == 0) {
        currentToken.type = TOKEN_WRITE;
        input += 5;
    }  else if (strncmp(input, "read", 4) == 0) {
        currentToken.type = TOKEN_READ;
        input += 4;
    } else if (strncmp(input, "while", 5) == 0) {
        currentToken.type = TOKEN_WHILE;
        input += 5;
    } else if ((strncmp(input, "break", 5) == 0) && isDelimiter(input[5])) {
        currentToken.type = TOKEN_BREAK;
        input += 5;
    } else if ((strncmp(input, "print", 5) == 0) && isDelimiter(input[5])) {
        currentToken.type = TOKEN_PRINT;
        input += 5;
    } else if ((strncmp(input, "var", 3) == 0)&& isDelimiter(input[3])) {
        currentToken.type = TOKEN_VAR;
        input += 3;
    } else if ((strncmp(input, "loop", 4) == 0)) {
        currentToken.type = TOKEN_DONE;
        input += 4;
    } else if ((strncmp(input, "compare", 7) == 0) && isDelimiter(input[7])){
        currentToken.type = TOKEN_COMP;
        input += 7;
    } else if ((strncmp(input, "deg", 3) == 0) && isDelimiter(input[3])){
        currentToken.type = TOKEN_DEGREES;
        input += 3;
    } else if ((strncmp(input, "bin", 3) == 0) && isDelimiter(input[3])) {
        currentToken.type = TOKEN_BINARY;
        input += 3;
    } else if ((strncmp(input, "badd", 4) == 0) && isDelimiter(input[4])) {
        currentToken.type = TOKEN_BADD;
        input += 4;
    } else if ((strncmp(input, "dist", 4) == 0) && isDelimiter(input[4])) {
        currentToken.type = TOKEN_DIST;
        input += 4;
    } else if ((strncmp(input, "root", 4) == 0) && isDelimiter(input[4])){
        currentToken.type = TOKEN_ROOT;
        input += 4;
    } else if ((strncmp(input, "rand", 4) == 0) && isDelimiter(input[4])){
        currentToken.type = TOKEN_RAND;
        input += 4;
    } else if ((strncmp(input, "uniform", 7) == 0) && isDelimiter(input[7])){
        currentToken.type = TOKEN_UNIFORM;
        input += 7;
    } else if ((strncmp(input, "sin", 3) == 0)&& isDelimiter(input[3])) {
        currentToken.type = TOKEN_SIN;
        input += 3;
    } else if ((strncmp(input, "cos", 3) == 0)&& isDelimiter(input[3])) {
        currentToken.type = TOKEN_COS;
        input += 3;
    } else if ((strncmp(input, "tan", 3) == 0)&& isDelimiter(input[3])) {
        currentToken.type = TOKEN_TAN;
        input += 3;
    } else if ((strncmp(input, "asin", 4) == 0) && isDelimiter(input[4])) {
        currentToken.type = TOKEN_ASIN;
        input += 4;
    } else if ((strncmp(input, "acos", 4) == 0) && isDelimiter(input[4])) {
        currentToken.type = TOKEN_ACOS;
        input += 4;
    } else if ((strncmp(input, "atan", 4) == 0) && isDelimiter(input[4])) {
        currentToken.type = TOKEN_ATAN;
        input += 4;
    } else if ((strncmp(input, "sqrt", 4) == 0) && isDelimiter(input[4])) {
        currentToken.type = TOKEN_SQRT;
        input += 4;
    } else if ((strncmp(input, "quad", 4) == 0) && isDelimiter(input[4])) {
        currentToken.type = TOKEN_QUADRATIC;
        input += 4;
    } else if ((strncmp(input, "math", 4) == 0) && isDelimiter(input[4])) {
        currentToken.type = TOKEN_MATH;
        input += 4;
    } else if ((strncmp(input, "conv", 4) == 0) && isDelimiter(input[4])) {
        currentToken.type = TOKEN_CONV;
        input += 4;
    } else if ((strncmp(input, "pyth", 4) == 0) && isDelimiter(input[4])) {
        currentToken.type = TOKEN_PYTHAGOREAN;
        input += 4;
    } else if ((strncmp(input, "round", 5) == 0) && isDelimiter(input[5])) {
        currentToken.type = TOKEN_ROUND;
        input += 5;
    } else if ((strncmp(input, "input", 5) == 0)  && isDelimiter(input[5]) ){
        currentToken.type = TOKEN_INPUT;
        input += 5;
    } else if ((strncmp(input, "import", 6) == 0)  && isDelimiter(input[6]) ){
        currentToken.type = TOKEN_IMPORT;
        input += 6;
    } else if ((strncmp(input, "time", 4) == 0)  && isDelimiter(input[4])){
        currentToken.type = TOKEN_TIME;
        input += 4;
    } else if (strncmp(input, "not", 3) == 0) {
        currentToken.type = TOKEN_NOT;
        input += 3;
    } else if (strncmp(input, "and", 3) == 0) {
        currentToken.type = TOKEN_AND;
        input += 3;
    } else if (strncmp(input, "xor", 3)== 0){
        currentToken.type = TOKEN_XOR;
        input += 3;
    } else if (isalpha(*input)) {
        currentToken.type = TOKEN_IDENTIFIER;
        int i = 0;
        while (isalpha(*input) || isdigit(*input)) {
            currentToken.identifier[i++] = *input;
            input++;
        }
        currentToken.identifier[i] = '\0';
    } else if (*input == '\"') {
        input++;
        currentToken.type = TOKEN_STRING;
        int i = 0;
        while (*input && *input != '\"') {
            currentToken.string[i++] = *input;
            input++;
        }
        currentToken.string[i] = '\0';
        input++;
    } else {
        error("This token is not yet supported. Invalid Token", input);
    }
}

void error(const char *message, const char *errorToken) {
    fprintf(stderr, "Error: %s. Found: %s\n", message, errorToken);
    if(shell_mode == false){
        exit(1);
    }
    if(log_enabled == 1){
        writeLog(message);
    }
}

void eat(TokenType type) {
    if (currentToken.type == type) {
        advance();

    } else {
        char tokenName[256];
        snprintf(tokenName, sizeof(tokenName), "%d", currentToken.type);
        error("Unexpected token", tokenName);
        if(shell_mode == false){
            exit(1);
        }
    }
}

void skipToEnd(){
    while (*input && *input != '\n') {
            input++;
    }
    advance();
    return;
}

void ifStatement() {
    eat(TOKEN_IF);
    double conditionValue = expression();
    eat(TOKEN_THEN);
    eat(TOKEN_DO);
    if (conditionValue != 0) {  //Remember, 0 is false cus im too lazy to code in actuall boolean logic
        if(debug){
            printf("\033[1;32mRunning if statement loop.\033[0m\n");
        }
        statement();
    } else{
        skipToEnd();
    }
}

void whileStatement() {
    char *start = input;
    Token startToken = currentToken;
    eat(TOKEN_WHILE);
    double conditionValue = expression();
    eat(TOKEN_DO);
    if (conditionValue != 0) {
        if(debug){
            printf("\033[1;32mRunning while loop.\033[0m\n");
        }
        while(currentToken.type != TOKEN_DONE){
            statement();
        }
        input = start;
        currentToken = startToken;
        whileStatement();
    }
    skipToEnd();
}

void setStringValue(const char *name, const char *value) {
    if(debug){
        printf("\033[1;31mSetting String %s to\033[0m \033[1;34m%s\033[0m\n", name, value);
    }
    for (int i = 0; i < numVariables; i++) {
        if (strcmp(name, variables[i].name) == 0) {
            strcpy(variables[i].stringValue, value);
            return;
        }
    }
    strcpy(variables[numVariables].name, name);
    strcpy(variables[numVariables].stringValue, value);
    numVariables++;
}

const char* getStringValue(const char *name) {
    for (int i = 0; i < numVariables; i++) {
        if (strcmp(name, variables[i].name) == 0) {
            return variables[i].stringValue;
        }
    }
    error("Variable not found", name);
    return 0;
}

int areEqual(const char *str1, const char *str2) {
    while(*str1 && *str2) {
        if(*str1 != *str2)
            return 0;
        str1++;
        str2++;
    }
    
    // If both strings have reached the end, they are equal.
    if(*str1 == '\0' && *str2 == '\0')
        return 1;
        
    return 0;
}

void importFile(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }

    fseek(file, 0, SEEK_END);
    long fileSize = ftell(file);
    rewind(file);

    char *importedCode = malloc(fileSize + 1);
    if (importedCode == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(file);
        exit(1);
    }

    size_t bytesRead = fread(importedCode, 1, fileSize, file);
    if (bytesRead < fileSize) {
        fprintf(stderr, "Error reading file: %s\n", filename);
        fclose(file);
        free(importedCode);
        exit(1);
    }

    importedCode[fileSize] = '\0';
    fclose(file);

    char *tempInput = input;
    Token tempCurrentToken = currentToken;

    lexer(importedCode);
    program();

    input = tempInput;
    currentToken = tempCurrentToken;

    free(importedCode);
}

void writeToFile() {
    eat(TOKEN_WRITE);
    char strName[2048];
    strcpy(strName, currentToken.identifier);
    eat(TOKEN_IDENTIFIER);
    eat(TOKEN_COMMA);
    char contentVarName[2048];
    strcpy(contentVarName, currentToken.identifier);
    eat(TOKEN_IDENTIFIER);
    const char *filename = getStringValue(strName);
    const char *content = getStringValue(contentVarName);
    FILE *file = fopen(filename, "w");
    
    if (file == NULL) {
        printf("Error opening file for writing.\n");
        return;
    }
    
    fprintf(file, "%s", content);
    fclose(file);
    if (debug){
        printf("\033[1;31mWriting %s to file\033[0m \033[1;34m%s\033[0m\n", content, filename);
    }
}

char* readFromFile() {
    eat(TOKEN_READ);
    char strname[2048];
    strcpy(strname, currentToken.identifier);
    eat(TOKEN_IDENTIFIER);
    const char *filename = getStringValue(strname);
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        return NULL;
    }
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    char* content = (char*)malloc(file_size + 1);
    if (content == NULL) {
        fclose(file);
        perror("Memory allocation error");
        return NULL;
    }

    size_t bytes_read = fread(content, 1, file_size, file);
    if (bytes_read != file_size) {
        fclose(file);
        free(content);
        perror("Error reading file");
        return NULL;
    }

    content[file_size] = '\0';

    fclose(file);
    return content;
}

double getVariableValue(const char *name) {
    for (int i = 0; i < numVariables; i++) {
        if (strcmp(name, variables[i].name) == 0) {
            return variables[i].value;
        }
    }
    error("Variable not found", name);
    return 0;
}

void setVariableValue(const char *name, double value) {
    if(debug==true){
        printf("\033[1;31mSetting variable %s to\033[0m \033[1;34m%f\033[0m\n", name, value);
    }
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

double getTimeInSeconds() {
    time_t currentTime = time(NULL);
    return difftime(currentTime, 0);
}

double timeFunction() {
    eat(TOKEN_TIME);
    return getTimeInSeconds();
}

double quadFunction() {
    double a, b, c, discriminant, root1, root2;
    eat(TOKEN_QUADRATIC);
    eat(TOKEN_LPAREN);
    a = expression();
    eat(TOKEN_COMMA);
    b = expression();
    eat(TOKEN_COMMA);
    c = expression();
    eat(TOKEN_RPAREN);
    discriminant = b * b - 4 * a * c;
    if (discriminant > 0) {
        root1 = (-b + sqrt(discriminant)) / (2 * a);
        root2 = (-b - sqrt(discriminant)) / (2 * a);
        if (root1>root2){
            return root1;
        } else {
            return root2;
        }
    } else if (discriminant == 0) {
        root1 = root2 = -b / (2 * a);
        return root1;
    } else {
        error("Imaginary nums not supported", "");
        return 1;
    }
}

int factorial(int n) {
    int fact = 1;
    for(int i = 1; i <= n; i++) {
        fact *= i;
    }
    return fact;
}

double expon(double base, double p) {
    if(base == 0 && p <= 0){
        return NAN;
    }
    return pow(base, p);
}

double nthRoot(){
    eat(TOKEN_ROOT);
    eat(TOKEN_LPAREN);
    double n = expression();
    eat(TOKEN_COMMA);
    double value = expression();
    eat(TOKEN_RPAREN);
    if(value < 0 || n <= 0) {
        return NAN;
    }
    if(n == 2){
        return sqrt(value);
    }
    
    double x = 1;
    double eps = 0.001; 

    while(1) {
        double diff = value - pow(x, n);
        if(fabs(diff) <= eps) {
            break;
        }

        x = x + diff / (n * pow(x, n - 1));
    }

    return x;
}

double distance(double ax, double ay, double bx, double by){
     return sqrtf((double)((bx-ax)*(bx-ax)+(by-ay)*(by-ay)));
}

double pythagoreanTheorem() {
    double a, b, c;
    eat(TOKEN_PYTHAGOREAN);
    eat(TOKEN_LPAREN);
    a = expression();
    eat(TOKEN_COMMA);
    b = expression();
    eat(TOKEN_RPAREN);
    c = sqrt(a*a+b*b);
    return c;
}

double decimalToBinary(double n) {
    int isNegative = 0;
    if (n < 0) {
        isNegative = 1;
        n = -n;
    }

    int integerPart = (int)n;
    double fractionalPart = n - integerPart;

    long long binaryIntegerPart = 0;
    int i = 0;
    while (integerPart > 0) {
        binaryIntegerPart += (integerPart % 2) * pow(10, i);
        integerPart /= 2;
        ++i;
    }

    double binaryFractionalPart = 0;
    i = -1;
    while (fractionalPart > 0 && i > -6) {
        fractionalPart *= 2;
        int bit = (int)fractionalPart;
        if (bit == 1) {
            fractionalPart -= bit;
            binaryFractionalPart += pow(10, i);
        }
        --i;
    }

    double result = binaryIntegerPart + binaryFractionalPart;
    return isNegative ? -result : result;
}

double binaryToTen(double n) {
    int isNegative = 0;
    if (n < 0) {
        isNegative = 1;
        n = -n;
    }

    long long integerPart = (long long)n;
    double fractionalPart = n - integerPart;

    double decimalIntegerPart = 0;
    int i = 0;
    while (integerPart > 0) {
        if (integerPart % 10 == 1) {
            decimalIntegerPart += pow(2, i);
        }
        integerPart /= 10;
        ++i;
    }

    double decimalFractionalPart = 0;
    i = -1;
    while (fractionalPart > 0 && i > -6) {
        double bit = fractionalPart * 10;
        if (bit >= 1) {
            fractionalPart = bit - (int)bit;
            decimalFractionalPart += pow(2, i);
        } else {
            fractionalPart = bit;
        }
        --i;
    }

    double result = decimalIntegerPart + decimalFractionalPart;
    return isNegative ? -result : result;
}

double binaryAdd(){
    double b1, b2, dec1, dec2;
    eat(TOKEN_LPAREN);
    b1=(double)expression();
    eat(TOKEN_COMMA);
    b2=(double)expression();
    eat(TOKEN_RPAREN);
    dec1 = binaryToTen(b1);
    dec2 = binaryToTen(b2);
    double n = dec1+dec2;
    return decimalToBinary(n);
}

double binaryConversion(){
    double n;
    eat(TOKEN_LPAREN);
    n=(double)expression();
    eat(TOKEN_RPAREN);
    double ans = decimalToBinary(n);
    return ans;
}

double factor() {
    if (currentToken.type == TOKEN_INT || currentToken.type == TOKEN_FLOAT) {
        double value = currentToken.value;
        eat(currentToken.type);
        return value;
    } else if (currentToken.type == TOKEN_IDENTIFIER) {
        char identifier[256];
        strcpy(identifier, currentToken.identifier);
        eat(TOKEN_IDENTIFIER);
        return getVariableValue(identifier);
    } else if (currentToken.type == TOKEN_LPAREN) {
        eat(TOKEN_LPAREN);
        double value = expression();
        eat(TOKEN_RPAREN);
        return value;
    } else if (currentToken.type == TOKEN_TIME) {
        return timeFunction();
    } else if (currentToken.type == TOKEN_COMP){
            eat(TOKEN_COMP);
            char identifier[2048];
            strcpy(identifier, currentToken.identifier);
            eat(TOKEN_IDENTIFIER);
            eat(TOKEN_COMMA);
            char identifier1[2048];
            strcpy(identifier1, currentToken.identifier);
            eat(TOKEN_IDENTIFIER);
            int compare = strcmp(getStringValue(identifier), getStringValue(identifier1));
            if(compare == 0){
                return 1;
            } else{
                return 0;
            }
    }
    else if (currentToken.type == TOKEN_MATH) {
        eat(TOKEN_MATH);
        eat(TOKEN_DOT);
        if (currentToken.type == TOKEN_QUADRATIC){
            return quadFunction();
        } else if (currentToken.type == TOKEN_SIN ||
                   currentToken.type == TOKEN_COS ||
                   currentToken.type == TOKEN_TAN ||
                   currentToken.type == TOKEN_ASIN||
                   currentToken.type == TOKEN_ACOS||
                   currentToken.type == TOKEN_ATAN||
                   currentToken.type == TOKEN_SQRT) {
            return argsMath();
        } else if(currentToken.type == TOKEN_PYTHAGOREAN){
            return pythagoreanTheorem();
        } else if(currentToken.type == TOKEN_ROOT){
            return nthRoot();
        } else if (currentToken.type == TOKEN_DIST){
            eat(TOKEN_DIST);
            eat(TOKEN_LPAREN);
            double ax = expression();
            eat(TOKEN_COMMA);
            double ay = expression();
            eat(TOKEN_COMMA);
            double bx = expression();
            eat(TOKEN_COMMA);
            double by = expression();
            eat(TOKEN_RPAREN);
            return distance(ax, ay, bx, by);
        } else if (currentToken.type == TOKEN_ROUND){
            eat(TOKEN_ROUND);
            eat(TOKEN_LPAREN);
            double thingToRound = expression();
            eat(TOKEN_RPAREN);
            return round(thingToRound);
        } else if (currentToken.type == TOKEN_RAND){
            eat(TOKEN_RAND);
            if (currentToken.type == TOKEN_LPAREN ){
                struct timeval tv;
                gettimeofday(&tv, NULL);
                unsigned int seed = (int)((tv.tv_sec * 1000000 + tv.tv_usec / 1000000 )/lines + cos(lines));
                srand(seed);
                eat(TOKEN_LPAREN);
                eat(TOKEN_RPAREN);
                return rand();
            } else if (currentToken.type == TOKEN_DOT){
                eat(TOKEN_DOT);
                eat(TOKEN_UNIFORM);
                eat(TOKEN_LPAREN);
                eat(TOKEN_RPAREN);
                struct timeval tv;
                gettimeofday(&tv, NULL);
                unsigned int seed = (int)((tv.tv_sec * 1000000 + tv.tv_usec / 1000000 )/lines + cos(lines));
                srand(seed);
                return ((double)rand() / (double)RAND_MAX);
            }
        } else if(currentToken.type == TOKEN_BINARY){
            eat(TOKEN_BINARY);
            eat(TOKEN_DOT);
            if(currentToken.type == TOKEN_CONV){
                eat(TOKEN_CONV);
                return binaryConversion();
            } else if(currentToken.type == TOKEN_BADD){
                eat(TOKEN_BADD);
                return binaryAdd();

            }
        }
    }
    

    error("Invalid factor", currentToken.identifier);
    return 0;
}

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

double expression() {
    double value = term();

    while (currentToken.type == TOKEN_PLUS || currentToken.type == TOKEN_MINUS ||
           currentToken.type == TOKEN_GREATER_THAN || currentToken.type == TOKEN_LESS_THAN ||
           currentToken.type == TOKEN_EQ || currentToken.type == TOKEN_NOT || currentToken.type == TOKEN_AND || 
           currentToken.type == TOKEN_OR || currentToken.type == TOKEN_XOR || currentToken.type == TOKEN_FACTORIAL || 
           currentToken.type == TOKEN_EXP || currentToken.type == TOKEN_MOD) {
        if (currentToken.type == TOKEN_PLUS) {
            eat(TOKEN_PLUS);
            value += term();
        } else if (currentToken.type == TOKEN_MINUS) {
            eat(TOKEN_MINUS);
            value -= term();
        } else if (currentToken.type == TOKEN_MOD) {
            eat(TOKEN_MOD);
            value = (int)value % (int)term();
        } else if (currentToken.type == TOKEN_GREATER_THAN) {
            eat(TOKEN_GREATER_THAN);
            value = (value > term()) ? 1.0 : 0.0;
        } else if (currentToken.type == TOKEN_EQ) {
            eat(TOKEN_EQ);
            if(value == term()){
                value = 1;
            } else {
                value = 0;
            }
        } else if (currentToken.type == TOKEN_LESS_THAN) {
            eat(TOKEN_LESS_THAN);
            value = (value < term()) ? 1.0 : 0.0;
        } else if (currentToken.type == TOKEN_FACTORIAL) {
            eat(TOKEN_FACTORIAL);
            value = factorial(value);
        } else if (currentToken.type == TOKEN_EXP) {
            eat(TOKEN_EXP);
            double value2 = term();
            value = expon(value, value2);

        } else if (currentToken.type == TOKEN_NOT) {
            eat(TOKEN_NOT);
            if(value==1){
                value = 0.0;
            } else if(value == 0){
                value = 1.0;
            } else{
                char str[2048];
                sprintf(str, "%f", value);
                error("The not keyword is a boolean operation supporting 1s and 0s", str);
            }
        } else if (currentToken.type == TOKEN_AND) {
            eat(TOKEN_AND);
            double value2 = term();
            if(value&&value2){
                value = 1.0;
            } else{
                value = 0.0;
            }
        } else if (currentToken.type == TOKEN_OR) {
            eat(TOKEN_OR);
            double value2 = term();
            if(value||value2){
                value = 1.0;
            } else{
                value = 0.0;
            }
        } else if(currentToken.type == TOKEN_XOR) {
            eat(TOKEN_XOR);
            double value2 = term();
            if((value && value2) || !(value && value2)){
                value = 0;
            } else if (value || value2){
                value = 1;
            }
        }
    }

    return value;
}

double argsMath() {
    TokenType functionType = currentToken.type;
    eat(functionType);
    eat(TOKEN_LPAREN);

    double value = expression();
    if (currentToken.type != TOKEN_RPAREN && functionType != TOKEN_SQRT){
        eat(TOKEN_COMMA);
        if (currentToken.type == TOKEN_DEGREES){
            value *= 3.1415926535/180;
            eat(TOKEN_DEGREES);
        } else{
            error("Invalid trig setting, Either degrees or radians.", "");
        }
    }
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

void assignment() {
    char identifier[2048];
    strcpy(identifier, currentToken.identifier);
    eat(TOKEN_IDENTIFIER);
    eat(TOKEN_ASSIGN);
    
    if (currentToken.type == TOKEN_STRING) {
        setStringValue(identifier, currentToken.string);
        eat(TOKEN_STRING);
    } else if(currentToken.type == TOKEN_READ){
        char *value = readFromFile();
        setStringValue(identifier, value);
    }else {
        double value = expression();
        setVariableValue(identifier, value);
    }
}

void printStatement() {
    eat(TOKEN_PRINT);
    if (currentToken.type == TOKEN_IDENTIFIER) {
        char identifier[2048];
        strcpy(identifier, currentToken.identifier);
        eat(TOKEN_IDENTIFIER);
        int isString = 0;
        for (int i = 0; i < numVariables; i++) {
            if (strcmp(identifier, variables[i].name) == 0) {
                isString = strlen(variables[i].stringValue) > 0;
                break;
            }
        }
        if (isString) {
            printf("%s\n", getStringValue(identifier));
            return;
        } 
        double value = getVariableValue(identifier);
        int int_value = (int)value;
        if (int_value == value){
            printf("%d\n", int_value);
        }else{
        printf("%f\n", value);
        }
    } else {
        double value = expression();
        printf("%f\n", value);
    }
}

void varInputStatement() {
    eat(TOKEN_INPUT);
    char identifier[2048];
    strcpy(identifier, currentToken.identifier);
    eat(TOKEN_IDENTIFIER);
    if(currentToken.type == TOKEN_COMMA){
        char userInput[2048];
        eat(TOKEN_COMMA);
        eat(TOKEN_IDENTIFIER);
        printf("Enter a value for %s: ", identifier);
        scanf("%s", userInput);
        setStringValue(identifier, userInput);
    } else{
        printf("Enter a value for %s: ", identifier);
        double value;
        scanf("%lf", &value);
        setVariableValue(identifier, value);
    }
}

void statement() {
    if (currentToken.type == TOKEN_IF) {
        ifStatement();
    } else if (currentToken.type == TOKEN_VAR) {
        eat(TOKEN_VAR);
        assignment();
    } else if (currentToken.type == TOKEN_PRINT) {
        printStatement();
    } else if (currentToken.type == TOKEN_INPUT) {
        varInputStatement();
    } else if (currentToken.type == TOKEN_WHILE) {
        whileStatement();
    } else if (currentToken.type == TOKEN_BREAK) {
        if(shell_mode == false){
            exit(1);
        }
    } else if (currentToken.type == TOKEN_WRITE){
        writeToFile();
    } else if (currentToken.type == TOKEN_SYSTEM){
        eat(TOKEN_SYSTEM);
        char cmd[2048];
        strcpy(cmd, currentToken.identifier);
        eat(TOKEN_IDENTIFIER);
        system(cmd);
    }
        
    else {
        error("Invalid statement", currentToken.identifier);
    }
}

void program() {
    while (currentToken.type != TOKEN_EOF) {
        if (currentToken.type == TOKEN_IMPORT) {
            eat(TOKEN_IMPORT);
            char filename[2048];
            strcpy(filename, currentToken.identifier);
            char extension[4] = ".jai";
            strcat(filename, extension);
            eat(TOKEN_IDENTIFIER);
            numVariables = 0;
            numImportedFiles = 0;

            importFile(filename);
            

        } else {
            statement();
        }
    }
}

void executeFile(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "err opening\n");
        exit(1);
    }

    fseek(file, 0, SEEK_END);
    long fileSize = ftell(file);
    rewind(file);

    char *code = malloc(fileSize + 1);
    if (code == NULL) {
        fprintf(stderr, "mem allocation err\n");
        fclose(file);
        exit(1);
    }

    size_t bytesRead = fread(code, 1, fileSize, file);
    if (bytesRead < fileSize) {
        fprintf(stderr, "err reading\n");
        fclose(file);
        free(code);
        exit(1);
    }

    code[fileSize] = '\0';
    fclose(file);

    lexer(code);
    program();
    
    free(code);
}

void shellMode() {
    char *input;
    shell_mode = true;
    printf("SHELL MODE!!!. Type 'exit' to quit.\n");
    while (1) {
        input = readline("> ");
        if (!input) {
            break;
        }
        
        if (strcmp(input, "exit") == 0) {
            free(input);
            break;
        }

        if (input && *input) {
            add_history(input);
        }
        
        lexer(input);
        program();

        free(input);
    }
}

void displayVersion() {
    FILE *file = fopen(VERSION_FILE, "r");
    if (!file) {
        fprintf(stderr, "Error: Unable to open version file.\n");
        return;
    }

    char line[32];
    while (fgets(line, sizeof(line), file)) {
        printf("%s", line);
    }

    fclose(file);
}

void displayHelp() {
    printf("Usage: %s [options] [filename]\n", "JAITHON");
    printf("\nOptions:\n");
    printf("  -d               Turn debug mode on\n");
    printf("  -s               Enter shell mode\n");
    printf("  -v, --version    Program Version\n");
    printf("  -h, --help       You are seeking Help!\n");
    printf("  --no-extension   Do not append .jai extension to filename\n");
}

void writeLog(const char *message) {
    if (!log_enabled) {
        return;
    }

    FILE *file = fopen(LOG_FILE, "a");
    if (file) {
        time_t current_time = time(NULL);
        char *time_str = ctime(&current_time);
        
        size_t len = strlen(time_str);
        if (len > 0 && time_str[len - 1] == '\n') {
            time_str[len - 1] = '\0';
        }

        fprintf(file, "[%s] %s\n", time_str, message);
        fclose(file);
    } else {
        fprintf(stderr, "Failed to write to log file.\n");
    }
}

int main(int argc, char *argv[]) {
    struct timeval stop, start;
    int opt;

    static struct option long_options[] = {
        {"version", no_argument, 0, 'v'},
        {"help", no_argument, 0, 'h'},
        {"no-extension", no_argument, 0, 1000},
        {"log", no_argument, 0, 'l'},
        {0, 0, 0, 0}
    };

    int long_index = 0;
    while ((opt = getopt_long(argc, argv, "dsvhlp", long_options, &long_index)) != -1) {
        switch (opt) {
            case 'd':
                debug = 1;
                break;
            case 's':
                shellMode();
                return 0;
            case 'v':
                displayVersion();
                return 0;
            case 'h':
                displayHelp();
                return 0;
            case 'l':
                log_enabled = true;
                writeLog("Logging enabled.");
                break;
            case 1000: // --no-extension
                auto_extension = 0;
                break;
            default:
                displayHelp();
                return 1;
        }
    }

    if (debug) {
        printf("====================YOU ARE IN DEBUG MODE====================\n");
        writeLog("Debug mode activated.");
    }

    gettimeofday(&start, NULL);
    writeLog("Program execution started.");

    char filename[MAX_FILENAME_LEN] = {0};
    if (optind < argc) {
        strncpy(filename, argv[optind], sizeof(filename) - 1);
        filename[sizeof(filename) - 1] = '\0';
    }

    if (strlen(filename) > 0) {
        if (auto_extension && !strstr(filename, FILE_EXTENSION)) {
            strncat(filename, FILE_EXTENSION, sizeof(filename) - strlen(filename) - 1);
        }
        writeLog("Executing provided file.");
        executeFile(filename);
    } else {
        writeLog("Entering shell mode.");
        shellMode();
    }

    gettimeofday(&stop, NULL);

    if (debug) {
        double elapsed = ((stop.tv_sec - start.tv_sec) * 1000000.0 + stop.tv_usec - start.tv_usec) * 0.000001;
        printf("\033[1;31mTook %f seconds\033[0m\n", elapsed);
        char log_message[100];
        snprintf(log_message, sizeof(log_message), "Execution took %f seconds.", elapsed);
        writeLog(log_message);
    }

    writeLog("Program execution completed.");
    return 0;
}
