#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

// Token types
typedef enum {
    TOKEN_INT,
    TOKEN_STRING,
    TOKEN_IDENTIFIER,
    TOKEN_ASSIGN,
    TOKEN_PLUS,
    TOKEN_MINUS,
    TOKEN_STAR,
    TOKEN_SLASH,
    TOKEN_LPAREN,
    TOKEN_RPAREN,
    TOKEN_SEMICOLON,
    TOKEN_EOF
} TokenType;

// Token structure
typedef struct {
    TokenType type;
    char* value;
} Token;

// Lexer
Token* lex(char* source) {
    // Tokenize the source code
    // Example implementation (tokenizes only integers and operators)

    int length = strlen(source);
    Token* tokens = (Token*)malloc((length + 1) * sizeof(Token));
    int tokenCount = 0;

    int i = 0;
    while (i < length) {
        if (isdigit(source[i])) {
            int j = i;
            while (isdigit(source[j]))
                j++;

            char* value = (char*)malloc((j - i + 1) * sizeof(char));
            strncpy(value, source + i, j - i);
            value[j - i] = '\0';

            Token token;
            token.type = TOKEN_INT;
            token.value = value;

            tokens[tokenCount] = token;
            tokenCount++;

            i = j;
        } else if (source[i] == '+') {
            Token token;
            token.type = TOKEN_PLUS;
            token.value = "+";

            tokens[tokenCount] = token;
            tokenCount++;

            i++;
        } else if (source[i] == '-') {
            Token token;
            token.type = TOKEN_MINUS;
            token.value = "-";

            tokens[tokenCount] = token;
            tokenCount++;

            i++;
        } else if (source[i] == '*') {
            Token token;
            token.type = TOKEN_STAR;
            token.value = "*";

            tokens[tokenCount] = token;
            tokenCount++;

            i++;
        } else if (source[i] == '/') {
            Token token;
            token.type = TOKEN_SLASH;
            token.value = "/";

            tokens[tokenCount] = token;
            tokenCount++;

            i++;
        } else {
            // Skip unrecognized characters
            i++;
        }
    }

    Token eofToken;
    eofToken.type = TOKEN_EOF;
    eofToken.value = "";

    tokens[tokenCount] = eofToken;
    tokenCount++;

    return tokens;
}

// Parser
typedef struct ASTNode {
    TokenType type;
    char* value;
    struct ASTNode* left;
    struct ASTNode* right;
} ASTNode;

ASTNode* parseExpression(Token** tokens);

ASTNode* parseFactor(Token** tokens) {
    if ((*tokens)->type == TOKEN_INT) {
        Token* token = *tokens;
        (*tokens)++;
        
        ASTNode* node = (ASTNode*)malloc(sizeof(ASTNode));
        node->type = TOKEN_INT;
        node->value = token->value;
        node->left = NULL;
        node->right = NULL;

        return node;
    }
    else if ((*tokens)->type == TOKEN_LPAREN) {
        (*tokens)++; // Consume '('
        
        ASTNode* node = parseExpression(tokens);
        
        if ((*tokens)->type != TOKEN_RPAREN) {
            printf("Error: Expected ')' after expression\n");
            exit(1);
        }
        
        (*tokens)++; // Consume ')'
        
        return node;
    }
    else {
        printf("Error: Invalid expression\n");
        exit(1);
    }
}

ASTNode* parseTerm(Token** tokens) {
    ASTNode* factor = parseFactor(tokens);

    while ((*tokens)->type == TOKEN_STAR || (*tokens)->type == TOKEN_SLASH) {
        TokenType operatorType = (*tokens)->type;
        (*tokens)++;

        ASTNode* nextFactor = parseFactor(tokens);

        ASTNode* newNode = (ASTNode*)malloc(sizeof(ASTNode));
        newNode->type = operatorType;
        newNode->value = NULL;
        newNode->left = factor;
        newNode->right = nextFactor;

        factor = newNode;
    }

    return factor;
}

ASTNode* parseExpression(Token** tokens) {
    ASTNode* term = parseTerm(tokens);

    while ((*tokens)->type == TOKEN_PLUS || (*tokens)->type == TOKEN_MINUS) {
        TokenType operatorType = (*tokens)->type;
        (*tokens)++;

        ASTNode* nextTerm = parseTerm(tokens);

        ASTNode* newNode = (ASTNode*)malloc(sizeof(ASTNode));
        newNode->type = operatorType;
        newNode->value = NULL;
        newNode->left = term;
        newNode->right = nextTerm;

        term = newNode;
    }

    return term;
}

ASTNode* parse(Token** tokens) {
    return parseExpression(tokens);
}

// Interpreter
int evaluate(ASTNode* node) {
    if (node->type == TOKEN_INT) {
        return atoi(node->value);
    } else if (node->type == TOKEN_PLUS) {
        return evaluate(node->left) + evaluate(node->right);
    } else if (node->type == TOKEN_MINUS) {
        return evaluate(node->left) - evaluate(node->right);
    } else if (node->type == TOKEN_STAR) {
        return evaluate(node->left) * evaluate(node->right);
    } else if (node->type == TOKEN_SLASH) {
        return evaluate(node->left) / evaluate(node->right);
    } else {
        // Error: Unsupported node type
        return 0;
    }
}

void freeTokens(Token* tokens) {
    for (int i = 0; tokens[i].type != TOKEN_EOF; i++) {
        free(tokens[i].value);
    }
    free(tokens);
}

void freeAST(ASTNode* node) {
    if (node == NULL) {
        return;
    }

    freeAST(node->left);
    freeAST(node->right);
    free(node);
}

void interpret(char* source) {
    Token* tokens = lex(source);
    ASTNode* ast = parse(&tokens);
    int result = evaluate(ast);

    printf("Result: %d\n", result);

    freeTokens(tokens);
    freeAST(ast);
}

int main() {
    char source[] = "10 + 5 * 2";

    interpret(source);

    return 0;
}
