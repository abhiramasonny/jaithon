/* token.h — token kinds and the compact token representation.
 *
 * A Token is 16 bytes and holds no copied text: it points into the source
 * buffer. Jaithon 2's 280-byte token (with a 256-byte inline name field) is
 * gone.
 */
#ifndef JAI_TOKEN_H
#define JAI_TOKEN_H

#include "../common/common.h"
#include "../common/diag.h"

typedef enum {
    TOK_EOF = 0,
    TOK_ERROR,
    TOK_NEWLINE,

    /* literals */
    TOK_INT,            /* value in Token.intValue    */
    TOK_FLOAT,          /* value in Token.floatValue  */
    TOK_STRING,         /* cooked text via jaiTokenStringValue */
    TOK_FSTRING_START,  /* f" ... up to the first {   */
    TOK_FSTRING_MID,    /* } ... {                    */
    TOK_FSTRING_END,    /* } ... "                    */
    TOK_IDENT,
    TOK_LABEL,          /* 'name  */

    /* punctuation */
    TOK_LPAREN, TOK_RPAREN, TOK_LBRACKET, TOK_RBRACKET, TOK_LBRACE, TOK_RBRACE,
    TOK_COMMA, TOK_DOT, TOK_DOTDOT, TOK_DOTDOT_EQ, TOK_ELLIPSIS,
    TOK_COLON, TOK_SEMICOLON, TOK_ARROW, TOK_FAT_ARROW, TOK_AT,
    TOK_QUESTION, TOK_QUESTION_DOT, TOK_QUESTION_QUESTION, TOK_UNDERSCORE,

    /* operators */
    TOK_PLUS, TOK_MINUS, TOK_STAR, TOK_SLASH, TOK_SLASH_SLASH, TOK_PERCENT,
    TOK_STAR_STAR, TOK_PLUS_PERCENT, TOK_MINUS_PERCENT, TOK_STAR_PERCENT,
    TOK_AMP, TOK_PIPE, TOK_CARET, TOK_TILDE, TOK_SHL, TOK_SHR,
    TOK_EQ, TOK_EQ_EQ, TOK_BANG_EQ, TOK_LT, TOK_LE, TOK_GT, TOK_GE,
    TOK_PLUS_EQ, TOK_MINUS_EQ, TOK_STAR_EQ, TOK_SLASH_EQ, TOK_SLASH_SLASH_EQ,
    TOK_PERCENT_EQ, TOK_STAR_STAR_EQ, TOK_AMP_EQ, TOK_PIPE_EQ, TOK_CARET_EQ,
    TOK_SHL_EQ, TOK_SHR_EQ,

    /* keywords — must stay contiguous and in sync with jaiKeywordTable */
    TOK_KW_AND, TOK_KW_AS, TOK_KW_ASSERT, TOK_KW_BREAK, TOK_KW_CASE,
    TOK_KW_CATCH, TOK_KW_CLASS, TOK_KW_CONST, TOK_KW_CONTINUE, TOK_KW_DEFER,
    TOK_KW_ELIF, TOK_KW_ELSE, TOK_KW_ENUM, TOK_KW_EXPORT, TOK_KW_EXTENDS,
    TOK_KW_FALSE, TOK_KW_FINALLY, TOK_KW_FOR, TOK_KW_FN, TOK_KW_FROM,
    TOK_KW_IF, TOK_KW_IMPL, TOK_KW_IMPORT, TOK_KW_IN, TOK_KW_IS, TOK_KW_LET,
    TOK_KW_LOOP, TOK_KW_MATCH, TOK_KW_MODULE, TOK_KW_MUT, TOK_KW_NOT,
    TOK_KW_NULL, TOK_KW_OR, TOK_KW_PROT, TOK_KW_PUB, TOK_KW_RETURN,
    TOK_KW_SELF, TOK_KW_STATIC, TOK_KW_SUPER, TOK_KW_THROW, TOK_KW_TRAIT,
    TOK_KW_TRUE, TOK_KW_TRY, TOK_KW_TYPE, TOK_KW_VAR, TOK_KW_WHILE,
    TOK_KW_YIELD,

    TOK_COUNT
} TokenKind;

#define TOK_FIRST_KEYWORD TOK_KW_AND
#define TOK_LAST_KEYWORD  TOK_KW_YIELD

typedef struct {
    uint16_t kind;        /* TokenKind */
    uint16_t flags;       /* TOKF_* */
    uint32_t start;       /* byte offset into the source */
    uint32_t length;
    union {
        int64_t  intValue;
        double   floatValue;
        uint32_t strIndex;   /* index into Lexer.strings for cooked text */
    } v;
} Token;

enum {
    TOKF_AFTER_NEWLINE = 1 << 0,  /* first token on its line */
    TOKF_HAS_ESCAPE    = 1 << 1,  /* string literal needed unescaping */
    TOKF_RAW_STRING    = 1 << 2,
    TOKF_TRIPLE_STRING = 1 << 3,
    /* TOK_INT holding 9223372036854775808, the magnitude of INT64_MIN. The
     * lexer only accepts it directly behind a prefix `-`, and stores it as the
     * already-negated INT64_MIN, so the parser must swallow that one `-`
     * instead of emitting a negation the VM could not perform. */
    TOKF_NEG_MAGNITUDE = 1 << 4,
};

const char *jaiTokenKindName(TokenKind kind);   /* "TOK_PLUS" */
const char *jaiTokenKindText(TokenKind kind);   /* "+", "fn", "<identifier>" */
bool        jaiTokenKindIsKeyword(TokenKind kind);
/* Look up a keyword by text; returns TOK_IDENT when not a keyword. */
TokenKind   jaiKeywordLookup(const char *text, size_t length);

#endif /* JAI_TOKEN_H */
