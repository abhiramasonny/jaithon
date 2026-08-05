/* token.c — token kind tables and keyword lookup.
 *
 * Both tables use designated initialisers so that a kind added to the enum but
 * forgotten here reports as "TOK_INVALID" instead of silently shifting every
 * later entry.
 */
#include "token.h"

_Static_assert(TOK_LAST_KEYWORD + 1 == TOK_COUNT,
               "keywords must be the last token kinds");

static const char *const kKindName[TOK_COUNT] = {
    [TOK_EOF]             = "TOK_EOF",
    [TOK_ERROR]           = "TOK_ERROR",
    [TOK_NEWLINE]         = "TOK_NEWLINE",

    [TOK_INT]             = "TOK_INT",
    [TOK_FLOAT]           = "TOK_FLOAT",
    [TOK_STRING]          = "TOK_STRING",
    [TOK_FSTRING_START]   = "TOK_FSTRING_START",
    [TOK_FSTRING_MID]     = "TOK_FSTRING_MID",
    [TOK_FSTRING_END]     = "TOK_FSTRING_END",
    [TOK_IDENT]           = "TOK_IDENT",
    [TOK_LABEL]           = "TOK_LABEL",

    [TOK_LPAREN]          = "TOK_LPAREN",
    [TOK_RPAREN]          = "TOK_RPAREN",
    [TOK_LBRACKET]        = "TOK_LBRACKET",
    [TOK_RBRACKET]        = "TOK_RBRACKET",
    [TOK_LBRACE]          = "TOK_LBRACE",
    [TOK_RBRACE]          = "TOK_RBRACE",
    [TOK_COMMA]           = "TOK_COMMA",
    [TOK_DOT]             = "TOK_DOT",
    [TOK_DOTDOT]          = "TOK_DOTDOT",
    [TOK_DOTDOT_EQ]       = "TOK_DOTDOT_EQ",
    [TOK_ELLIPSIS]        = "TOK_ELLIPSIS",
    [TOK_COLON]           = "TOK_COLON",
    [TOK_SEMICOLON]       = "TOK_SEMICOLON",
    [TOK_ARROW]           = "TOK_ARROW",
    [TOK_FAT_ARROW]       = "TOK_FAT_ARROW",
    [TOK_AT]              = "TOK_AT",
    [TOK_QUESTION]        = "TOK_QUESTION",
    [TOK_QUESTION_DOT]    = "TOK_QUESTION_DOT",
    [TOK_QUESTION_QUESTION] = "TOK_QUESTION_QUESTION",
    [TOK_UNDERSCORE]      = "TOK_UNDERSCORE",

    [TOK_PLUS]            = "TOK_PLUS",
    [TOK_MINUS]           = "TOK_MINUS",
    [TOK_STAR]            = "TOK_STAR",
    [TOK_SLASH]           = "TOK_SLASH",
    [TOK_SLASH_SLASH]     = "TOK_SLASH_SLASH",
    [TOK_PERCENT]         = "TOK_PERCENT",
    [TOK_STAR_STAR]       = "TOK_STAR_STAR",
    [TOK_PLUS_PERCENT]    = "TOK_PLUS_PERCENT",
    [TOK_MINUS_PERCENT]   = "TOK_MINUS_PERCENT",
    [TOK_STAR_PERCENT]    = "TOK_STAR_PERCENT",
    [TOK_AMP]             = "TOK_AMP",
    [TOK_PIPE]            = "TOK_PIPE",
    [TOK_CARET]           = "TOK_CARET",
    [TOK_TILDE]           = "TOK_TILDE",
    [TOK_SHL]             = "TOK_SHL",
    [TOK_SHR]             = "TOK_SHR",
    [TOK_EQ]              = "TOK_EQ",
    [TOK_EQ_EQ]           = "TOK_EQ_EQ",
    [TOK_BANG_EQ]         = "TOK_BANG_EQ",
    [TOK_LT]              = "TOK_LT",
    [TOK_LE]              = "TOK_LE",
    [TOK_GT]              = "TOK_GT",
    [TOK_GE]              = "TOK_GE",
    [TOK_PLUS_EQ]         = "TOK_PLUS_EQ",
    [TOK_MINUS_EQ]        = "TOK_MINUS_EQ",
    [TOK_STAR_EQ]         = "TOK_STAR_EQ",
    [TOK_SLASH_EQ]        = "TOK_SLASH_EQ",
    [TOK_SLASH_SLASH_EQ]  = "TOK_SLASH_SLASH_EQ",
    [TOK_PERCENT_EQ]      = "TOK_PERCENT_EQ",
    [TOK_STAR_STAR_EQ]    = "TOK_STAR_STAR_EQ",
    [TOK_AMP_EQ]          = "TOK_AMP_EQ",
    [TOK_PIPE_EQ]         = "TOK_PIPE_EQ",
    [TOK_CARET_EQ]        = "TOK_CARET_EQ",
    [TOK_SHL_EQ]          = "TOK_SHL_EQ",
    [TOK_SHR_EQ]          = "TOK_SHR_EQ",

    [TOK_KW_AND]          = "TOK_KW_AND",
    [TOK_KW_AS]           = "TOK_KW_AS",
    [TOK_KW_ASSERT]       = "TOK_KW_ASSERT",
    [TOK_KW_BREAK]        = "TOK_KW_BREAK",
    [TOK_KW_CASE]         = "TOK_KW_CASE",
    [TOK_KW_CATCH]        = "TOK_KW_CATCH",
    [TOK_KW_CLASS]        = "TOK_KW_CLASS",
    [TOK_KW_CONST]        = "TOK_KW_CONST",
    [TOK_KW_CONTINUE]     = "TOK_KW_CONTINUE",
    [TOK_KW_DEFER]        = "TOK_KW_DEFER",
    [TOK_KW_ELIF]         = "TOK_KW_ELIF",
    [TOK_KW_ELSE]         = "TOK_KW_ELSE",
    [TOK_KW_ENUM]         = "TOK_KW_ENUM",
    [TOK_KW_EXPORT]       = "TOK_KW_EXPORT",
    [TOK_KW_EXTENDS]      = "TOK_KW_EXTENDS",
    [TOK_KW_FALSE]        = "TOK_KW_FALSE",
    [TOK_KW_FINALLY]      = "TOK_KW_FINALLY",
    [TOK_KW_FOR]          = "TOK_KW_FOR",
    [TOK_KW_FN]           = "TOK_KW_FN",
    [TOK_KW_FROM]         = "TOK_KW_FROM",
    [TOK_KW_IF]           = "TOK_KW_IF",
    [TOK_KW_IMPL]         = "TOK_KW_IMPL",
    [TOK_KW_IMPORT]       = "TOK_KW_IMPORT",
    [TOK_KW_IN]           = "TOK_KW_IN",
    [TOK_KW_IS]           = "TOK_KW_IS",
    [TOK_KW_LET]          = "TOK_KW_LET",
    [TOK_KW_LOOP]         = "TOK_KW_LOOP",
    [TOK_KW_MATCH]        = "TOK_KW_MATCH",
    [TOK_KW_MODULE]       = "TOK_KW_MODULE",
    [TOK_KW_MUT]          = "TOK_KW_MUT",
    [TOK_KW_NOT]          = "TOK_KW_NOT",
    [TOK_KW_NULL]         = "TOK_KW_NULL",
    [TOK_KW_OR]           = "TOK_KW_OR",
    [TOK_KW_PROT]         = "TOK_KW_PROT",
    [TOK_KW_PUB]          = "TOK_KW_PUB",
    [TOK_KW_RETURN]       = "TOK_KW_RETURN",
    [TOK_KW_SELF]         = "TOK_KW_SELF",
    [TOK_KW_STATIC]       = "TOK_KW_STATIC",
    [TOK_KW_SUPER]        = "TOK_KW_SUPER",
    [TOK_KW_THROW]        = "TOK_KW_THROW",
    [TOK_KW_TRAIT]        = "TOK_KW_TRAIT",
    [TOK_KW_TRUE]         = "TOK_KW_TRUE",
    [TOK_KW_TRY]          = "TOK_KW_TRY",
    [TOK_KW_TYPE]         = "TOK_KW_TYPE",
    [TOK_KW_VAR]          = "TOK_KW_VAR",
    [TOK_KW_WHILE]        = "TOK_KW_WHILE",
    [TOK_KW_YIELD]        = "TOK_KW_YIELD",
};

/* Source spelling, or a bracketed placeholder for kinds whose text varies.
 * These strings land verbatim in diagnostics ("expected `,`, found `}`"). */
static const char *const kKindText[TOK_COUNT] = {
    [TOK_EOF]             = "<end of file>",
    [TOK_ERROR]           = "<error>",
    [TOK_NEWLINE]         = "<newline>",

    [TOK_INT]             = "<integer literal>",
    [TOK_FLOAT]           = "<float literal>",
    [TOK_STRING]          = "<string literal>",
    [TOK_FSTRING_START]   = "<f-string>",
    [TOK_FSTRING_MID]     = "<f-string part>",
    [TOK_FSTRING_END]     = "<f-string end>",
    [TOK_IDENT]           = "<identifier>",
    [TOK_LABEL]           = "<label>",

    [TOK_LPAREN]          = "(",
    [TOK_RPAREN]          = ")",
    [TOK_LBRACKET]        = "[",
    [TOK_RBRACKET]        = "]",
    [TOK_LBRACE]          = "{",
    [TOK_RBRACE]          = "}",
    [TOK_COMMA]           = ",",
    [TOK_DOT]             = ".",
    [TOK_DOTDOT]          = "..",
    [TOK_DOTDOT_EQ]       = "..=",
    [TOK_ELLIPSIS]        = "...",
    [TOK_COLON]           = ":",
    [TOK_SEMICOLON]       = ";",
    [TOK_ARROW]           = "->",
    [TOK_FAT_ARROW]       = "=>",
    [TOK_AT]              = "@",
    [TOK_QUESTION]        = "?",
    [TOK_QUESTION_DOT]    = "?.",
    [TOK_QUESTION_QUESTION] = "??",
    [TOK_UNDERSCORE]      = "_",

    [TOK_PLUS]            = "+",
    [TOK_MINUS]           = "-",
    [TOK_STAR]            = "*",
    [TOK_SLASH]           = "/",
    [TOK_SLASH_SLASH]     = "//",
    [TOK_PERCENT]         = "%",
    [TOK_STAR_STAR]       = "**",
    [TOK_PLUS_PERCENT]    = "+%",
    [TOK_MINUS_PERCENT]   = "-%",
    [TOK_STAR_PERCENT]    = "*%",
    [TOK_AMP]             = "&",
    [TOK_PIPE]            = "|",
    [TOK_CARET]           = "^",
    [TOK_TILDE]           = "~",
    [TOK_SHL]             = "<<",
    [TOK_SHR]             = ">>",
    [TOK_EQ]              = "=",
    [TOK_EQ_EQ]           = "==",
    [TOK_BANG_EQ]         = "!=",
    [TOK_LT]              = "<",
    [TOK_LE]              = "<=",
    [TOK_GT]              = ">",
    [TOK_GE]              = ">=",
    [TOK_PLUS_EQ]         = "+=",
    [TOK_MINUS_EQ]        = "-=",
    [TOK_STAR_EQ]         = "*=",
    [TOK_SLASH_EQ]        = "/=",
    [TOK_SLASH_SLASH_EQ]  = "//=",
    [TOK_PERCENT_EQ]      = "%=",
    [TOK_STAR_STAR_EQ]    = "**=",
    [TOK_AMP_EQ]          = "&=",
    [TOK_PIPE_EQ]         = "|=",
    [TOK_CARET_EQ]        = "^=",
    [TOK_SHL_EQ]          = "<<=",
    [TOK_SHR_EQ]          = ">>=",

    [TOK_KW_AND]          = "and",
    [TOK_KW_AS]           = "as",
    [TOK_KW_ASSERT]       = "assert",
    [TOK_KW_BREAK]        = "break",
    [TOK_KW_CASE]         = "case",
    [TOK_KW_CATCH]        = "catch",
    [TOK_KW_CLASS]        = "class",
    [TOK_KW_CONST]        = "const",
    [TOK_KW_CONTINUE]     = "continue",
    [TOK_KW_DEFER]        = "defer",
    [TOK_KW_ELIF]         = "elif",
    [TOK_KW_ELSE]         = "else",
    [TOK_KW_ENUM]         = "enum",
    [TOK_KW_EXPORT]       = "export",
    [TOK_KW_EXTENDS]      = "extends",
    [TOK_KW_FALSE]        = "false",
    [TOK_KW_FINALLY]      = "finally",
    [TOK_KW_FOR]          = "for",
    [TOK_KW_FN]           = "fn",
    [TOK_KW_FROM]         = "from",
    [TOK_KW_IF]           = "if",
    [TOK_KW_IMPL]         = "impl",
    [TOK_KW_IMPORT]       = "import",
    [TOK_KW_IN]           = "in",
    [TOK_KW_IS]           = "is",
    [TOK_KW_LET]          = "let",
    [TOK_KW_LOOP]         = "loop",
    [TOK_KW_MATCH]        = "match",
    [TOK_KW_MODULE]       = "module",
    [TOK_KW_MUT]          = "mut",
    [TOK_KW_NOT]          = "not",
    [TOK_KW_NULL]         = "null",
    [TOK_KW_OR]           = "or",
    [TOK_KW_PROT]         = "prot",
    [TOK_KW_PUB]          = "pub",
    [TOK_KW_RETURN]       = "return",
    [TOK_KW_SELF]         = "self",
    [TOK_KW_STATIC]       = "static",
    [TOK_KW_SUPER]        = "super",
    [TOK_KW_THROW]        = "throw",
    [TOK_KW_TRAIT]        = "trait",
    [TOK_KW_TRUE]         = "true",
    [TOK_KW_TRY]          = "try",
    [TOK_KW_TYPE]         = "type",
    [TOK_KW_VAR]          = "var",
    [TOK_KW_WHILE]        = "while",
    [TOK_KW_YIELD]        = "yield",
};

const char *jaiTokenKindName(TokenKind kind) {
    if ((unsigned)kind >= (unsigned)TOK_COUNT || kKindName[kind] == NULL)
        return "TOK_INVALID";
    return kKindName[kind];
}

const char *jaiTokenKindText(TokenKind kind) {
    if ((unsigned)kind >= (unsigned)TOK_COUNT || kKindText[kind] == NULL)
        return "<invalid>";
    return kKindText[kind];
}

bool jaiTokenKindIsKeyword(TokenKind kind) {
    return kind >= TOK_FIRST_KEYWORD && kind <= TOK_LAST_KEYWORD;
}

/* The first byte is already known to match, so only the tail is compared. The
 * caller guarantees strlen(kw) == length. */
JAI_INLINE bool kwTail(const char *text, size_t length, const char *kw) {
    return memcmp(text + 1, kw + 1, length - 1) == 0;
}

/* Dispatch on (length, first byte). Every keyword bucket then needs at most
 * one memcmp of the tail, so no input ever walks a chain of comparisons. */
TokenKind jaiKeywordLookup(const char *text, size_t length) {
    if (text == NULL || length < 2 || length > 8) return TOK_IDENT;

    switch (length) {
    case 2:
        switch (text[0]) {
        case 'a': return kwTail(text, length, "as") ? TOK_KW_AS : TOK_IDENT;
        case 'f': return kwTail(text, length, "fn") ? TOK_KW_FN : TOK_IDENT;
        case 'i':
            if (text[1] == 'f') return TOK_KW_IF;
            if (text[1] == 'n') return TOK_KW_IN;
            if (text[1] == 's') return TOK_KW_IS;
            return TOK_IDENT;
        case 'o': return kwTail(text, length, "or") ? TOK_KW_OR : TOK_IDENT;
        default:  return TOK_IDENT;
        }
    case 3:
        switch (text[0]) {
        case 'a': return kwTail(text, length, "and") ? TOK_KW_AND : TOK_IDENT;
        case 'f': return kwTail(text, length, "for") ? TOK_KW_FOR : TOK_IDENT;
        case 'l': return kwTail(text, length, "let") ? TOK_KW_LET : TOK_IDENT;
        case 'm': return kwTail(text, length, "mut") ? TOK_KW_MUT : TOK_IDENT;
        case 'n': return kwTail(text, length, "not") ? TOK_KW_NOT : TOK_IDENT;
        case 'p': return kwTail(text, length, "pub") ? TOK_KW_PUB : TOK_IDENT;
        case 't': return kwTail(text, length, "try") ? TOK_KW_TRY : TOK_IDENT;
        case 'v': return kwTail(text, length, "var") ? TOK_KW_VAR : TOK_IDENT;
        default:  return TOK_IDENT;
        }
    case 4:
        switch (text[0]) {
        case 'c': return kwTail(text, length, "case") ? TOK_KW_CASE : TOK_IDENT;
        case 'e':
            if (kwTail(text, length, "elif")) return TOK_KW_ELIF;
            if (kwTail(text, length, "else")) return TOK_KW_ELSE;
            if (kwTail(text, length, "enum")) return TOK_KW_ENUM;
            return TOK_IDENT;
        case 'f': return kwTail(text, length, "from") ? TOK_KW_FROM : TOK_IDENT;
        case 'i': return kwTail(text, length, "impl") ? TOK_KW_IMPL : TOK_IDENT;
        case 'l': return kwTail(text, length, "loop") ? TOK_KW_LOOP : TOK_IDENT;
        case 'n': return kwTail(text, length, "null") ? TOK_KW_NULL : TOK_IDENT;
        case 'p': return kwTail(text, length, "prot") ? TOK_KW_PROT : TOK_IDENT;
        case 's': return kwTail(text, length, "self") ? TOK_KW_SELF : TOK_IDENT;
        case 't':
            if (kwTail(text, length, "true")) return TOK_KW_TRUE;
            if (kwTail(text, length, "type")) return TOK_KW_TYPE;
            return TOK_IDENT;
        default:  return TOK_IDENT;
        }
    case 5:
        switch (text[0]) {
        case 'b': return kwTail(text, length, "break") ? TOK_KW_BREAK : TOK_IDENT;
        case 'c':
            if (kwTail(text, length, "catch")) return TOK_KW_CATCH;
            if (kwTail(text, length, "class")) return TOK_KW_CLASS;
            if (kwTail(text, length, "const")) return TOK_KW_CONST;
            return TOK_IDENT;
        case 'd': return kwTail(text, length, "defer") ? TOK_KW_DEFER : TOK_IDENT;
        case 'f': return kwTail(text, length, "false") ? TOK_KW_FALSE : TOK_IDENT;
        case 'm': return kwTail(text, length, "match") ? TOK_KW_MATCH : TOK_IDENT;
        case 's': return kwTail(text, length, "super") ? TOK_KW_SUPER : TOK_IDENT;
        case 't':
            if (kwTail(text, length, "throw")) return TOK_KW_THROW;
            if (kwTail(text, length, "trait")) return TOK_KW_TRAIT;
            return TOK_IDENT;
        case 'w': return kwTail(text, length, "while") ? TOK_KW_WHILE : TOK_IDENT;
        case 'y': return kwTail(text, length, "yield") ? TOK_KW_YIELD : TOK_IDENT;
        default:  return TOK_IDENT;
        }
    case 6:
        switch (text[0]) {
        case 'a': return kwTail(text, length, "assert") ? TOK_KW_ASSERT : TOK_IDENT;
        case 'e': return kwTail(text, length, "export") ? TOK_KW_EXPORT : TOK_IDENT;
        case 'i': return kwTail(text, length, "import") ? TOK_KW_IMPORT : TOK_IDENT;
        case 'm': return kwTail(text, length, "module") ? TOK_KW_MODULE : TOK_IDENT;
        case 'r': return kwTail(text, length, "return") ? TOK_KW_RETURN : TOK_IDENT;
        case 's': return kwTail(text, length, "static") ? TOK_KW_STATIC : TOK_IDENT;
        default:  return TOK_IDENT;
        }
    case 7:
        switch (text[0]) {
        case 'e': return kwTail(text, length, "extends") ? TOK_KW_EXTENDS : TOK_IDENT;
        case 'f': return kwTail(text, length, "finally") ? TOK_KW_FINALLY : TOK_IDENT;
        default:  return TOK_IDENT;
        }
    case 8:
        if (text[0] == 'c' && kwTail(text, length, "continue")) return TOK_KW_CONTINUE;
        return TOK_IDENT;
    default:
        return TOK_IDENT;
    }
}
