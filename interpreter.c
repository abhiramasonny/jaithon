#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

typedef enum {
    TOKEN_EOF,
    TOKEN_INT,
    TOKEN_IDENTIFIER,
    TOKEN_ASSIGN,
    TOKEN_SEMICOLON,
    TOKEN_PLUS,
    TOKEN_MINUS,
    TOKEN_STAR,
    TOKEN_SLASH,
    TOKEN_LOOP
} TokenType;

typedef struct {
    TokenType type;
    union {
        int value;
        char identifier;
    };
} Token;

typedef struct ASTNode {
    TokenType type;
    struct ASTNode* left;
    struct ASTNode* right;
    int value;
    char identifier;
    struct ASTNode* expression;
    struct ASTNode* body;
} ASTNode;

typedef struct {
    int value;
} Variable;

Variable variables[26];

// Tokenization logic
Token* lex(const char* source) {
    int length = strlen(source);
    Token* tokens = (Token*)malloc((length + 1) * sizeof(Token));
    int tokenCount = 0;

    int i = 0;
    while (i < length) {
        if (source[i] == ' ') {
            i++; // Skip whitespace
            continue;
        }

        Token token;
        if (isdigit(source[i])) {
            token.type = TOKEN_INT;
            token.value = atoi(source + i);
            while (isdigit(source[i]))
                i++; // Consume digits
        } else if (isalpha(source[i])) {
            token.type = TOKEN_IDENTIFIER;
            token.identifier = source[i];
            i++; // Consume identifier
        } else {
            switch (source[i]) {
                case '=':
                    token.type = TOKEN_ASSIGN;
                    break;
                case ';':
                    token.type = TOKEN_SEMICOLON;
                    break;
                case '+':
                    token.type = TOKEN_PLUS;
                    break;
                case '-':
                    token.type = TOKEN_MINUS;
                    break;
                case '*':
                    token.type = TOKEN_STAR;
                    break;
                case '/':
                    token.type = TOKEN_SLASH;
                    break;
                case 'l':
                    if (strncmp(source + i, "loop", 4) == 0) {
                        token.type = TOKEN_LOOP;
                        i += 3; // Consume "loop"
                    } else {
                        printf("Error: Unexpected token at position %d\n", i);
                        exit(1);
                    }
                    break;
                default:
                    printf("Error: Unexpected token at position %d\n", i);
                    exit(1);
            }
            i++; // Consume operator or delimiter
        }

        tokens[tokenCount++] = token;
    }

    // Add EOF token
    Token eofToken;
    eofToken.type = TOKEN_EOF;
    tokens[tokenCount] = eofToken;

    return tokens;
}

ASTNode* parseExpression(Token** tokens);

ASTNode* parseFactor(Token** tokens) {
    Token* currentToken = *tokens;

    if (currentToken->type == TOKEN_INT) {
        ASTNode* factor = (ASTNode*)malloc(sizeof(ASTNode));
        factor->type = currentToken->type;
        factor->value = currentToken->value;
        (*tokens)++;
        return factor;
    } else if (currentToken->type == TOKEN_IDENTIFIER) {
        ASTNode* factor = (ASTNode*)malloc(sizeof(ASTNode));
        factor->type = currentToken->type;
        factor->identifier = currentToken->identifier;
        (*tokens)++;
        return factor;
    } else {
        printf("Error: Unexpected token\n");
        exit(1);
    }
}

ASTNode* parseTerm(Token** tokens) {
    ASTNode* factor = parseFactor(tokens);
    Token* currentToken = *tokens;

    while (currentToken->type == TOKEN_STAR || currentToken->type == TOKEN_SLASH) {
        (*tokens)++; // Consume operator

        ASTNode* nextFactor = parseFactor(tokens);

        ASTNode* term = (ASTNode*)malloc(sizeof(ASTNode));
        term->type = currentToken->type;
        term->left = factor;
        term->right = nextFactor;

        factor = term;
        currentToken = *tokens;
    }

    return factor;
}

ASTNode* parseExpression(Token** tokens) {
    ASTNode* term = parseTerm(tokens);
    Token* currentToken = *tokens;

    while (currentToken->type == TOKEN_PLUS || currentToken->type == TOKEN_MINUS) {
        (*tokens)++; // Consume operator

        ASTNode* nextTerm = parseTerm(tokens);

        ASTNode* expression = (ASTNode*)malloc(sizeof(ASTNode));
        expression->type = currentToken->type;
        expression->left = term;
        expression->right = nextTerm;

        term = expression;
        currentToken = *tokens;
    }

    return term;
}

ASTNode* parseAssignment(Token** tokens) {
    Token* currentToken = *tokens;

    if (currentToken->type == TOKEN_IDENTIFIER) {
        (*tokens)++; // Consume identifier

        if ((*tokens)->type == TOKEN_ASSIGN) {
            (*tokens)++; // Consume assignment operator

            ASTNode* expression = parseExpression(tokens);

            ASTNode* assignment = (ASTNode*)malloc(sizeof(ASTNode));
            assignment->type = TOKEN_ASSIGN;
            assignment->left = (ASTNode*)malloc(sizeof(ASTNode));
            assignment->left->type = TOKEN_IDENTIFIER;
            assignment->left->identifier = currentToken->identifier;
            assignment->right = expression;

            return assignment;
        }
    }

    printf("Error: Invalid assignment\n");
    exit(1);
}

ASTNode* parseLoop(Token** tokens) {
    Token* currentToken = *tokens;

    if (currentToken->type == TOKEN_LOOP) {
        (*tokens)++; // Consume loop keyword

        ASTNode* startValue = parseExpression(tokens);
        ASTNode* endValue = parseExpression(tokens);
        ASTNode* increment = parseExpression(tokens);
        ASTNode* body = parseExpression(tokens);

        ASTNode* loopNode = (ASTNode*)malloc(sizeof(ASTNode));
        loopNode->type = TOKEN_LOOP;
        loopNode->left = startValue;
        loopNode->right = endValue;
        loopNode->expression = increment;
        loopNode->body = body;

        return loopNode;
    }

    printf("Error: Invalid loop\n");
    exit(1);
}

int evaluate(ASTNode* node) {
    if (node->type == TOKEN_INT) {
        return node->value;
    } else if (node->type == TOKEN_IDENTIFIER) {
        int index = node->identifier - 'A';
        return variables[index].value;
    } else if (node->type == TOKEN_PLUS) {
        return evaluate(node->left) + evaluate(node->right);
    } else if (node->type == TOKEN_MINUS) {
        return evaluate(node->left) - evaluate(node->right);
    } else if (node->type == TOKEN_STAR) {
        return evaluate(node->left) * evaluate(node->right);
    } else if (node->type == TOKEN_SLASH) {
        return evaluate(node->left) / evaluate(node->right);
    } else if (node->type == TOKEN_LOOP) {
        int startValue = evaluate(node->left);
        int endValue = evaluate(node->right);
        int increment = evaluate(node->expression);

        int result = 0;
        for (int i = startValue; i <= endValue; i += increment) {
            variables[node->identifier - 'A'].value = i;
            result += evaluate(node->body);
        }

        return result;
    }

    printf("Error: Invalid expression\n");
    exit(1);
}

void execute(ASTNode* node) {
    if (node->type == TOKEN_ASSIGN) {
        int index = node->left->identifier - 'A';
        variables[index].value = evaluate(node->right);
    } else if (node->type == TOKEN_LOOP) {
        int index = node->left->identifier - 'A';

        int startValue = evaluate(node->left);
        int endValue = evaluate(node->right);
        int increment = evaluate(node->expression);

        for (int i = startValue; i <= endValue; i += increment) {
            variables[index].value = i;
            execute(node->body);
        }
    } else {
        printf("%d\n", evaluate(node));
    }
}

int main() {
    const char* source = "A = 10; B = 5; loop A 1 B 1 A = A - 1;";


    Token* tokens = lex(source);

    Token* currentToken = tokens;
    while (currentToken->type != TOKEN_EOF) {
        ASTNode* node;
        if (currentToken->type == TOKEN_ASSIGN) {
            node = parseAssignment(&currentToken);
        } else if (currentToken->type == TOKEN_LOOP) {
            node = parseLoop(&currentToken);
        } else {
            node = parseExpression(&currentToken);
        }

        execute(node);
        currentToken++; // Consume semicolon
    }

    free(tokens);

    return 0;
}
