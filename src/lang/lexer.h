/* lexer.h — tokenizer.
 *
 * The lexer runs to completion up front and produces a token array. This keeps
 * the parser's lookahead trivial, makes error recovery cheap, and lets the
 * formatter reuse the same token stream with trivia attached.
 */
#ifndef JAI_LEXER_H
#define JAI_LEXER_H

#include "token.h"

typedef struct {
    const char *source;
    size_t      length;
    int         fileId;

    size_t      pos;
    int         line;

    JAI_VEC(Token)  tokens;
    /* Cooked text of string literals, indexed by Token.v.strIndex.
     * Owned by the lexer; jaiLexerTakeStrings transfers ownership. */
    JAI_VEC(char *) strings;
    /* Allocated size of strings.data[i], parallel to it. A cooked literal may
     * contain an embedded NUL (`"\0"`), so strlen is not the size that was
     * handed out and freeing by strlen+1 misreports oldSize to the
     * allocator. */
    JAI_VEC(size_t) stringSizes;

    /* Bracket nesting stack: tracks whether a '{' opened a block or an
     * expression, which decides whether newlines are significant. */
    JAI_VEC(uint8_t) brackets;
    /* Depth of f-string interpolation currently being lexed. */
    int         interpDepth;

    bool        hadError;
} Lexer;

void jaiLexerInit(Lexer *lex, const char *source, size_t length, int fileId);
void jaiLexerFree(Lexer *lex);

/* Tokenize the whole input. Returns false if any lexical error was reported
 * (errors go to gDiags; lexing continues so the parser still sees a stream
 * terminated by TOK_EOF). */
bool jaiLexerRun(Lexer *lex);

/* Cooked text of a string-literal token, NUL-terminated. */
const char *jaiTokenStringValue(const Lexer *lex, const Token *tok, size_t *outLen);
/* Raw source text of any token (points into the source buffer). */
const char *jaiTokenText(const Lexer *lex, const Token *tok, size_t *outLen);
/* Span for diagnostics. */
JaiSpan     jaiTokenSpan(const Lexer *lex, const Token *tok);

/* Transfer the cooked-string arena to the caller (the parser keeps it alive
 * for as long as the AST references it). */
char      **jaiLexerTakeStrings(Lexer *lex, int *outCount);

/* Diagnostic helper used by tools: dump the token stream. */
void jaiLexerDump(FILE *out, const Lexer *lex);

#endif /* JAI_LEXER_H */
