/* lexer.c — the Jaithon tokenizer. Normative reference: spec/LANGUAGE.md §1.
 *
 * The whole input is tokenized up front. Lexical errors are reported to gDiags
 * and recovered from locally, so the parser always receives a well-formed
 * stream terminated by exactly one TOK_EOF.
 */
#include "lexer.h"

#include <errno.h>
#include <stdlib.h>

#define JAI_MAX_INTERP_DEPTH 8

/* ------------------------------------------------------------------ */
/* Bracket stack                                                       */
/* ------------------------------------------------------------------ */

/* Entries of Lexer.brackets. Every entry except BRK_BRACE_BLOCK suppresses
 * newlines: those brackets opened an expression, and an expression may span
 * lines freely (§1.3).
 *
 * An f-string interpolation pushes BRK_INTERP plus the flavour bits of the
 * string it belongs to, so the '}' that closes the interpolation knows how to
 * resume scanning the tail. Keeping that state on the bracket stack is what
 * lets a nested f-string inside an interpolation be lexed by the same loop
 * with no recursion and no side stack. */
enum {
    BRK_PAREN       = 1,
    BRK_BRACKET     = 2,
    BRK_BRACE_BLOCK = 3,
    BRK_BRACE_EXPR  = 4,

    BRK_INTERP      = 8,   /* or-ed with the flavour bits below */
    BRK_F_SINGLE    = 1,   /* delimiter is ' rather than " */
    BRK_F_TRIPLE    = 2,
    BRK_F_RAW       = 4,
};

typedef struct {
    char quote;
    bool triple;
    bool raw;
    bool interp;
} StrFlavor;

/* Scanner state that does not fit in the (frozen) Lexer struct. */
typedef struct {
    Lexer *lex;
    bool   atLineStart;   /* the next token emitted is the first on its line */
} Scan;

typedef enum { CHUNK_CLOSED, CHUNK_INTERP, CHUNK_BROKEN } ChunkStop;

static void scanStringLiteral(Scan *s, size_t tokStart, bool raw, bool interp);

/* ------------------------------------------------------------------ */
/* Character classification                                            */
/* ------------------------------------------------------------------ */

static bool isAsciiDigit(char c) { return c >= '0' && c <= '9'; }

static bool isAsciiAlpha(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

static bool isAsciiIdentStart(char c) { return isAsciiAlpha(c) || c == '_'; }

static bool isAsciiIdentCont(char c) {
    return isAsciiIdentStart(c) || isAsciiDigit(c);
}

static int hexValue(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static bool digitInBase(char c, int base) {
    int v = hexValue(c);
    return v >= 0 && v < base;
}

/* Non-ASCII scalars that behave as separators rather than letters. Source is
 * not allowed to smuggle these into identifiers, where they would be
 * invisible. */
static bool isUnicodeSpaceScalar(int32_t cp) {
    return cp == 0x00A0 || cp == 0x1680 || cp == 0x3000 || cp == 0xFEFF ||
           (cp >= 0x2000 && cp <= 0x200D) || cp == 0x2028 || cp == 0x2029 ||
           cp == 0x202F || cp == 0x205F;
}

/* Pragmatic approximation of XID_Start/XID_Continue (§1.4): any well-formed
 * scalar >= 0x80 is an identifier character unless it is space-like or lives
 * in the General Punctuation block. Real XID tables are ~10 KB of data and
 * buy nothing the compiler currently needs; this accepts a superset that is
 * stable across Unicode versions and still rejects the confusable spaces and
 * the smart quotes that people paste in by accident. */
static bool isUnicodeIdentScalar(int32_t cp) {
    if (cp < 0x80) return false;
    if (isUnicodeSpaceScalar(cp)) return false;
    if (cp >= 0x2000 && cp <= 0x206F) return false;   /* general punctuation */
    if (cp >= 0xFFF9 && cp <= 0xFFFB) return false;   /* interlinear annotation */
    return true;
}

/* ------------------------------------------------------------------ */
/* Cursor helpers                                                      */
/* ------------------------------------------------------------------ */

static bool atEnd(const Lexer *lex) { return lex->pos >= lex->length; }

static char peekAt(const Lexer *lex, size_t offset) {
    size_t i = lex->pos + offset;
    return i < lex->length ? lex->source[i] : '\0';
}

static char peekc(const Lexer *lex) { return peekAt(lex, 0); }

static bool eatChar(Lexer *lex, char c) {
    if (peekc(lex) == c) {
        lex->pos++;
        return true;
    }
    return false;
}

static JaiSpan spanOf(const Lexer *lex, size_t start, size_t end) {
    JaiSpan span;
    span.start = (uint32_t)start;
    span.end   = (uint32_t)end;
    span.file  = lex->fileId;
    return span;
}

static TokenKind lastKind(const Lexer *lex) {
    if (lex->tokens.count == 0) return TOK_EOF;
    return (TokenKind)lex->tokens.data[lex->tokens.count - 1].kind;
}

static Token *pushToken(Scan *s, TokenKind kind, size_t start, size_t end) {
    Lexer *lex = s->lex;
    Token t;
    t.kind        = (uint16_t)kind;
    t.flags       = (uint16_t)(s->atLineStart ? TOKF_AFTER_NEWLINE : 0);
    t.start       = (uint32_t)start;
    t.length      = (uint32_t)(end - start);
    t.v.intValue  = 0;
    JAI_VEC_PUSH(Token, &lex->tokens, t);
    s->atLineStart = false;
    return &JAI_VEC_LAST(&lex->tokens);
}

/* Copies the cooked bytes into the lexer's string arena and releases `buf`. */
static uint32_t internString(Lexer *lex, JaiBuf *buf) {
    size_t n = buf->count;
    char *copy = JAI_ALLOC(char, n + 1);
    if (n > 0) memcpy(copy, buf->data, n);
    copy[n] = '\0';
    jaiBufFree(buf);
    uint32_t index = (uint32_t)lex->strings.count;
    JAI_VEC_PUSH(char *, &lex->strings, copy);
    JAI_VEC_PUSH(size_t, &lex->stringSizes, n + 1);
    return index;
}

/* ------------------------------------------------------------------ */
/* Newline significance (§1.3)                                         */
/* ------------------------------------------------------------------ */

/* Tokens after which a line break cannot end a statement. This is the list in
 * §1.3 plus the wrapping operators (added to the token set after that list was
 * written), `..`/`..=`, and `{`. Suppressing the newline directly after `{`
 * costs nothing for a block — the grammar already allows `"{" { NEWLINE }` —
 * and it keeps multi-line dict/set literals working even when the brace was
 * classified as a block. `?` is deliberately absent: `let x: int?` ends a
 * statement with it. */
static bool isContinuationToken(TokenKind k) {
    if (k >= TOK_PLUS && k <= TOK_SHR_EQ) return true;   /* every operator */
    switch (k) {
    case TOK_COMMA:
    case TOK_DOT:
    case TOK_DOTDOT:
    case TOK_DOTDOT_EQ:
    case TOK_COLON:
    case TOK_ARROW:
    case TOK_FAT_ARROW:
    case TOK_LPAREN:
    case TOK_LBRACKET:
    case TOK_LBRACE:
    case TOK_QUESTION_DOT:
    case TOK_QUESTION_QUESTION:
    case TOK_KW_AND:
    case TOK_KW_OR:
    case TOK_KW_NOT:
    case TOK_FSTRING_START:
    case TOK_FSTRING_MID:
        return true;
    default:
        return false;
    }
}

static bool bracketTopIsInterp(const Lexer *lex) {
    return lex->brackets.count > 0 &&
           JAI_VEC_LAST(&lex->brackets) >= BRK_INTERP;
}

static bool bracketTopIsExpression(const Lexer *lex) {
    if (lex->brackets.count == 0) return false;
    uint8_t top = JAI_VEC_LAST(&lex->brackets);
    return top != BRK_BRACE_BLOCK;
}

/* Decides whether a '{' opens a block or an expression. Ambiguity resolves to
 * BLOCK: mistaking a block for an expression swallows the newlines that
 * separate its statements, whereas the reverse only leaves NEWLINE tokens
 * inside a literal, which the parser skips. So only an unmistakable
 * expression position — right after an operator, a comma, an opening bracket,
 * or a keyword that must be followed by a value — opens an expression.
 *
 * `?` is not in the set even though a ternary may yield a dict, because
 * `fn f() -> int? {` would then swallow the whole body. */
static bool braceOpensExpression(const Lexer *lex) {
    TokenKind prev = lastKind(lex);
    if (prev >= TOK_PLUS && prev <= TOK_SHR_EQ) return true;
    switch (prev) {
    case TOK_COMMA:
    case TOK_COLON:
    case TOK_LPAREN:
    case TOK_LBRACKET:
    case TOK_QUESTION_QUESTION:
    case TOK_KW_RETURN:
    case TOK_KW_THROW:
    case TOK_KW_YIELD:
    case TOK_KW_IN:
    case TOK_KW_IS:
    case TOK_KW_AND:
    case TOK_KW_OR:
    case TOK_KW_NOT:
    case TOK_FSTRING_START:
    case TOK_FSTRING_MID:
        return true;
    case TOK_LBRACE:
        /* `{{` — inherit from the brace we are already inside. */
        return bracketTopIsExpression(lex);
    default:
        return false;
    }
}

static void handleNewline(Scan *s) {
    Lexer *lex = s->lex;
    size_t start = lex->pos;
    lex->pos++;
    lex->line++;

    bool significant = lex->tokens.count > 0 &&
                       !bracketTopIsExpression(lex) &&
                       lastKind(lex) != TOK_NEWLINE &&
                       !isContinuationToken(lastKind(lex));
    if (significant) pushToken(s, TOK_NEWLINE, start, start + 1);
    s->atLineStart = true;
}

/* ------------------------------------------------------------------ */
/* Comments                                                            */
/* ------------------------------------------------------------------ */

static void skipComment(Scan *s) {
    Lexer *lex = s->lex;
    size_t start = lex->pos;

    if (peekAt(lex, 1) != '*') {
        while (!atEnd(lex) && peekc(lex) != '\n') lex->pos++;
        return;
    }

    lex->pos += 2;
    int depth = 1;
    while (!atEnd(lex)) {
        if (peekc(lex) == '#' && peekAt(lex, 1) == '*') {
            depth++;
            lex->pos += 2;
            continue;
        }
        if (peekc(lex) == '*' && peekAt(lex, 1) == '#') {
            lex->pos += 2;
            if (--depth == 0) return;
            continue;
        }
        if (peekc(lex) == '\n') lex->line++;
        lex->pos++;
    }

    lex->hadError = true;
    JaiDiag *d = jaiDiagError(E0002_UNTERMINATED_COMMENT,
                              spanOf(lex, start, start + 2),
                              "unterminated block comment");
    if (d != NULL) {
        if (depth > 1)
            jaiDiagAddNote(d, "%d nested `#*` comments are still open", depth);
        jaiDiagAddHelp(d, "close it with `*#`");
    }
}

/* ------------------------------------------------------------------ */
/* Numbers (§1.6)                                                      */
/* ------------------------------------------------------------------ */

/* Consumes a run of base-`base` digits with `_` separators, appending only the
 * digits to `out`. Separators may not lead, trail, or double up (E0005). */
static bool scanDigits(Scan *s, int base, JaiBuf *out, int *outCount) {
    Lexer *lex = s->lex;
    bool prevUnderscore = false;
    bool any = false;
    int count = 0;

    for (;;) {
        char c = peekc(lex);
        if (c == '_') {
            if (!any || prevUnderscore) {
                lex->hadError = true;
                JaiDiag *d = jaiDiagError(
                    E0005_INVALID_NUMBER, spanOf(lex, lex->pos, lex->pos + 1),
                    prevUnderscore ? "repeated `_` in numeric literal"
                                   : "numeric literal may not start with `_`");
                if (d != NULL)
                    jaiDiagAddHelp(d, "digit separators must sit between digits");
                while (!atEnd(lex) &&
                       (peekc(lex) == '_' || digitInBase(peekc(lex), base)))
                    lex->pos++;
                *outCount = count;
                return false;
            }
            prevUnderscore = true;
            lex->pos++;
            continue;
        }
        if (!digitInBase(c, base) || atEnd(lex)) break;
        jaiBufPush(out, (uint8_t)c);
        count++;
        any = true;
        prevUnderscore = false;
        lex->pos++;
    }

    *outCount = count;
    if (prevUnderscore) {
        lex->hadError = true;
        JaiDiag *d = jaiDiagError(E0005_INVALID_NUMBER,
                                  spanOf(lex, lex->pos - 1, lex->pos),
                                  "numeric literal may not end with `_`");
        if (d != NULL)
            jaiDiagAddHelp(d, "digit separators must sit between digits");
        return false;
    }
    return true;
}

/* Whether a token can be the end of an operand. Used to tell a prefix `-` from
 * a subtraction. */
static bool endsOperand(TokenKind k) {
    switch (k) {
    case TOK_IDENT: case TOK_INT: case TOK_FLOAT: case TOK_STRING:
    case TOK_FSTRING_END: case TOK_LABEL: case TOK_UNDERSCORE:
    case TOK_RPAREN: case TOK_RBRACKET: case TOK_RBRACE:
    case TOK_KW_TRUE: case TOK_KW_FALSE: case TOK_KW_NULL:
    case TOK_KW_SELF: case TOK_KW_SUPER:
        return true;
    default:
        return false;
    }
}

/* 9223372036854775808 is one past INT64_MAX, so it only exists as the operand
 * of a prefix `-`, where the pair spells INT64_MIN. Deciding this here, from
 * the preceding token rather than from whitespace, is what keeps `a -1` a
 * subtraction (`a` ends an operand) and `- x` a negation (no literal follows):
 * the magnitude is admitted only where a *literal* could start.
 *
 * A second `-` in front disqualifies it: `--9223372036854775808` asks for
 * `-INT64_MIN`, which is not an `int`, so the magnitude is rejected as E0005
 * exactly like the bare literal. */
static bool isNegatedLiteral(const Lexer *lex) {
    int n = lex->tokens.count;
    if (n == 0 || lex->tokens.data[n - 1].kind != TOK_MINUS) return false;
    if (n == 1) return true;
    TokenKind before = (TokenKind)lex->tokens.data[n - 2].kind;
    return before != TOK_MINUS && !endsOperand(before);
}

/* A letter or digit glued to the end of a literal is never valid: there are no
 * literal suffixes, so `2f` and `0b102` are typos, not two tokens. */
static bool rejectNumberSuffix(Scan *s, int base, size_t start) {
    Lexer *lex = s->lex;
    char c = peekc(lex);
    if (atEnd(lex) || (!isAsciiIdentStart(c) && !isAsciiDigit(c) &&
                       (unsigned char)c < 0x80))
        return true;

    size_t bad = lex->pos;
    while (!atEnd(lex) &&
           (isAsciiIdentCont(peekc(lex)) || (unsigned char)peekc(lex) >= 0x80))
        lex->pos++;

    lex->hadError = true;
    JaiDiag *d = jaiDiagError(E0005_INVALID_NUMBER, spanOf(lex, start, lex->pos),
                              "invalid suffix on numeric literal");
    if (d != NULL) {
        jaiDiagAddLabel(d, spanOf(lex, bad, lex->pos), "not part of a number");
        if (isAsciiDigit(c))
            jaiDiagAddHelp(d, "`%c` is not a base-%d digit", c, base);
        else
            jaiDiagAddHelp(d, "Jaithon has no literal suffixes; separate the "
                              "number from the name");
    }
    return false;
}

static void scanNumber(Scan *s) {
    Lexer *lex = s->lex;
    size_t start = lex->pos;
    JaiBuf text;
    jaiBufInit(&text);

    int base = 10;
    bool isFloat = false;
    bool ok = true;
    int count = 0;

    char prefix = peekAt(lex, 1);
    if (peekc(lex) == '0' && (prefix == 'x' || prefix == 'X' || prefix == 'o' ||
                              prefix == 'O' || prefix == 'b' || prefix == 'B')) {
        base = (prefix == 'x' || prefix == 'X') ? 16
             : (prefix == 'o' || prefix == 'O') ? 8
                                                : 2;
        lex->pos += 2;
        ok = scanDigits(s, base, &text, &count);
        if (ok && count == 0) {
            ok = false;
            lex->hadError = true;
            JaiDiag *d = jaiDiagError(E0005_INVALID_NUMBER,
                                      spanOf(lex, start, lex->pos),
                                      "missing digits after `0%c`", prefix);
            if (d != NULL) jaiDiagAddHelp(d, "write at least one base-%d digit", base);
        }
    } else {
        ok = scanDigits(s, 10, &text, &count);

        /* A '.' belongs to the number only when a digit follows it, so `0..10`
         * is INT DOTDOT INT and `1.foo` is INT DOT IDENT. */
        if (ok && peekc(lex) == '.' && isAsciiDigit(peekAt(lex, 1))) {
            isFloat = true;
            jaiBufPush(&text, '.');
            lex->pos++;
            ok = scanDigits(s, 10, &text, &count);
        }

        if (ok && (peekc(lex) == 'e' || peekc(lex) == 'E')) {
            char sign = peekAt(lex, 1);
            size_t digitOffset = (sign == '+' || sign == '-') ? 2 : 1;
            if (isAsciiDigit(peekAt(lex, digitOffset))) {
                isFloat = true;
                jaiBufPush(&text, 'e');
                if (digitOffset == 2) jaiBufPush(&text, (uint8_t)sign);
                lex->pos += digitOffset;
                ok = scanDigits(s, 10, &text, &count);
            }
        }
    }

    if (ok) ok = rejectNumberSuffix(s, base, start);

    int64_t intValue = 0;
    double  floatValue = 0.0;
    bool minMagnitude = false;
    if (ok) {
        jaiBufPush(&text, '\0');
        const char *digits = (const char *)text.data;
        if (isFloat) {
            errno = 0;
            floatValue = strtod(digits, NULL);
            /* ERANGE here means the literal saturates to inf or flushes to
             * zero, which is exactly IEEE-754 behaviour, so it is not an
             * error. */
        } else {
            errno = 0;
            unsigned long long raw = strtoull(digits, NULL, base);
            /* Decimal literals are signed and must fit in int64, the single
             * exception being INT64_MIN's magnitude under a prefix `-`. Hex,
             * octal and binary literals are bit patterns, so the whole
             * unsigned range is accepted and reinterpreted two's complement. */
            /* Hex, octal and binary reach INT64_MIN's bit pattern on their
             * own, but the `-` in front of one is still the sign of a single
             * literal, so all four bases are marked the same way. */
            minMagnitude = raw == (unsigned long long)INT64_MAX + 1 &&
                           isNegatedLiteral(lex);
            bool overflow = (errno == ERANGE) ||
                            (base == 10 && raw > (unsigned long long)INT64_MAX &&
                             !minMagnitude);
            if (overflow) {
                ok = false;
                lex->hadError = true;
                JaiDiag *d = jaiDiagError(E0005_INVALID_NUMBER,
                                          spanOf(lex, start, lex->pos),
                                          "integer literal does not fit in `int`");
                if (d != NULL)
                    jaiDiagAddNote(d, "`int` is 64-bit signed: %lld to %lld",
                                   (long long)INT64_MIN, (long long)INT64_MAX);
            } else {
                intValue = (int64_t)raw;
            }
        }
    }
    jaiBufFree(&text);

    Token *t = pushToken(s, isFloat ? TOK_FLOAT : TOK_INT, start, lex->pos);
    if (isFloat) t->v.floatValue = ok ? floatValue : 0.0;
    else         t->v.intValue   = ok ? intValue : 0;
    if (ok && minMagnitude) t->flags |= TOKF_NEG_MAGNITUDE;
}

/* ------------------------------------------------------------------ */
/* Strings (§1.6, §1.7)                                                */
/* ------------------------------------------------------------------ */

static void pushScalar(JaiBuf *out, int32_t cp) {
    if (cp == 0) {
        jaiBufPush(out, 0);          /* jaiUtf8Encode rejects NUL */
        return;
    }
    char utf8[4];
    int n = jaiUtf8Encode(cp, utf8);
    if (n > 0) jaiBufAppend(out, utf8, (size_t)n);
    /* n == 0 cannot happen: every caller validates the scalar first. */
}

/* Consumes one escape sequence, `lex->pos` sitting on the backslash. Returns
 * false when the escape was rejected; the backslash is always consumed so the
 * caller still makes progress. */
static bool scanEscape(Scan *s, JaiBuf *out) {
    Lexer *lex = s->lex;
    size_t esc = lex->pos;
    lex->pos++;
    if (atEnd(lex)) return false;   /* the caller reports the unterminated literal */

    char c = lex->source[lex->pos++];
    switch (c) {
    case 'n':  jaiBufPush(out, '\n'); return true;
    case 't':  jaiBufPush(out, '\t'); return true;
    case 'r':  jaiBufPush(out, '\r'); return true;
    case '\\': jaiBufPush(out, '\\'); return true;
    case '"':  jaiBufPush(out, '"');  return true;
    case '\'': jaiBufPush(out, '\''); return true;
    case '0':  jaiBufPush(out, 0);    return true;
    case 'x': {
        int32_t value = 0;
        for (int i = 0; i < 2; i++) {
            int digit = hexValue(peekc(lex));
            if (digit < 0 || atEnd(lex)) {
                lex->hadError = true;
                JaiDiag *d = jaiDiagError(E0004_INVALID_ESCAPE,
                                          spanOf(lex, esc, lex->pos),
                                          "`\\x` needs exactly two hex digits");
                if (d != NULL) jaiDiagAddHelp(d, "for example `\\x1b`");
                return false;
            }
            value = value * 16 + digit;
            lex->pos++;
        }
        pushScalar(out, value);   /* \xNN names a scalar, encoded as UTF-8 */
        return true;
    }
    case 'u': {
        if (peekc(lex) != '{') {
            lex->hadError = true;
            JaiDiag *d = jaiDiagError(E0004_INVALID_ESCAPE,
                                      spanOf(lex, esc, lex->pos),
                                      "`\\u` must be followed by `{...}`");
            if (d != NULL) jaiDiagAddHelp(d, "for example `\\u{1F600}`");
            return false;
        }
        lex->pos++;
        int32_t value = 0;
        int digits = 0;
        while (hexValue(peekc(lex)) >= 0 && !atEnd(lex)) {
            if (digits < 6) value = value * 16 + hexValue(peekc(lex));
            digits++;
            lex->pos++;
        }
        bool closed = eatChar(lex, '}');
        if (digits == 0 || digits > 6 || !closed) {
            lex->hadError = true;
            JaiDiag *d = jaiDiagError(E0004_INVALID_ESCAPE,
                                      spanOf(lex, esc, lex->pos),
                                      "malformed `\\u{...}` escape");
            if (d != NULL)
                jaiDiagAddHelp(d, "write one to six hex digits inside the braces");
            return false;
        }
        if (value > 0x10FFFF || (value >= 0xD800 && value <= 0xDFFF)) {
            lex->hadError = true;
            JaiDiag *d = jaiDiagError(E0004_INVALID_ESCAPE,
                                      spanOf(lex, esc, lex->pos),
                                      "U+%04X is not a Unicode scalar value", value);
            if (d != NULL)
                jaiDiagAddNote(d, "scalars are U+0000..U+D7FF and U+E000..U+10FFFF");
            return false;
        }
        pushScalar(out, value);
        return true;
    }
    default: {
        lex->hadError = true;
        JaiDiag *d = jaiDiagError(E0004_INVALID_ESCAPE, spanOf(lex, esc, lex->pos),
                                  "unknown escape sequence `\\%c`", c);
        if (d != NULL)
            jaiDiagAddHelp(d, "valid escapes are \\n \\t \\r \\\\ \\\" \\' \\0 "
                              "\\xNN \\u{...}");
        jaiBufPush(out, (uint8_t)c);   /* recover as if the backslash were absent */
        return false;
    }
    }
}

/* Cooks characters up to the closing delimiter or, in an f-string, up to the
 * '{' that opens an interpolation (which is consumed). */
static ChunkStop scanStringChunk(Scan *s, StrFlavor f, JaiBuf *out,
                                 size_t openStart, bool *sawEscape) {
    Lexer *lex = s->lex;

    for (;;) {
        if (atEnd(lex)) {
            lex->hadError = true;
            JaiDiag *d = jaiDiagError(
                f.interp ? E0008_UNTERMINATED_INTERPOLATION
                         : E0001_UNTERMINATED_STRING,
                spanOf(lex, openStart, lex->pos),
                f.interp ? "unterminated f-string" : "unterminated string literal");
            if (d != NULL)
                jaiDiagAddHelp(d, "add a closing %s",
                               f.triple ? (f.quote == '"' ? "\"\"\"" : "'''")
                                        : (f.quote == '"' ? "\"" : "'"));
            return CHUNK_BROKEN;
        }

        char c = lex->source[lex->pos];

        if (c == f.quote) {
            if (!f.triple) {
                lex->pos++;
                return CHUNK_CLOSED;
            }
            if (peekAt(lex, 1) == f.quote && peekAt(lex, 2) == f.quote) {
                lex->pos += 3;
                return CHUNK_CLOSED;
            }
            jaiBufPush(out, (uint8_t)c);
            lex->pos++;
            continue;
        }

        if (c == '\r' && peekAt(lex, 1) == '\n') {
            lex->pos++;              /* §1.1: the \r of a CRLF is discarded */
            continue;
        }

        if (c == '\n') {
            if (!f.triple) {
                /* Leave the newline for the main loop so the next line still
                 * lexes as statements. */
                lex->hadError = true;
                JaiDiag *d = jaiDiagError(
                    f.interp ? E0008_UNTERMINATED_INTERPOLATION
                             : E0001_UNTERMINATED_STRING,
                    spanOf(lex, openStart, lex->pos),
                    "unterminated %s at end of line",
                    f.interp ? "f-string" : "string literal");
                if (d != NULL)
                    jaiDiagAddHelp(d, "use a triple-quoted string to span lines");
                return CHUNK_BROKEN;
            }
            jaiBufPush(out, '\n');
            lex->pos++;
            lex->line++;
            continue;
        }

        if (c == '\\') {
            if (f.raw) {
                jaiBufPush(out, '\\');   /* raw strings keep the backslash */
                lex->pos++;
                continue;
            }
            if (scanEscape(s, out)) *sawEscape = true;
            continue;
        }

        if (f.interp && c == '{') {
            if (peekAt(lex, 1) == '{') {
                jaiBufPush(out, '{');
                lex->pos += 2;
                continue;
            }
            lex->pos++;
            return CHUNK_INTERP;
        }

        if (f.interp && c == '}') {
            if (peekAt(lex, 1) == '}') {
                jaiBufPush(out, '}');
                lex->pos += 2;
                continue;
            }
            lex->hadError = true;
            JaiDiag *d = jaiDiagError(E0003_INVALID_CHARACTER,
                                      spanOf(lex, lex->pos, lex->pos + 1),
                                      "single `}` in f-string text");
            if (d != NULL) jaiDiagAddHelp(d, "write `}}` for a literal brace");
            jaiBufPush(out, '}');
            lex->pos++;
            continue;
        }

        jaiBufPush(out, (uint8_t)c);   /* UTF-8 passes through byte by byte */
        lex->pos++;
    }
}

static StrFlavor flavorFromMarker(uint8_t marker) {
    StrFlavor f;
    f.quote  = (marker & BRK_F_SINGLE) ? '\'' : '"';
    f.triple = (marker & BRK_F_TRIPLE) != 0;
    f.raw    = (marker & BRK_F_RAW) != 0;
    f.interp = true;
    return f;
}

static void openInterpolation(Scan *s, StrFlavor f) {
    Lexer *lex = s->lex;
    if (lex->interpDepth >= JAI_MAX_INTERP_DEPTH) {
        lex->hadError = true;
        JaiDiag *d = jaiDiagError(E0009_NESTED_INTERPOLATION_TOO_DEEP,
                                  spanOf(lex, lex->pos - 1, lex->pos),
                                  "f-string interpolation nested more than %d deep",
                                  JAI_MAX_INTERP_DEPTH);
        if (d != NULL)
            jaiDiagAddHelp(d, "compute the value into a binding before formatting it");
    }
    lex->interpDepth++;
    uint8_t marker = (uint8_t)(BRK_INTERP |
                               (f.quote == '\'' ? BRK_F_SINGLE : 0) |
                               (f.triple ? BRK_F_TRIPLE : 0) |
                               (f.raw ? BRK_F_RAW : 0));
    JAI_VEC_PUSH(uint8_t, &lex->brackets, marker);
}

static uint16_t stringFlags(StrFlavor f, bool sawEscape) {
    return (uint16_t)((sawEscape ? TOKF_HAS_ESCAPE : 0) |
                      (f.raw ? TOKF_RAW_STRING : 0) |
                      (f.triple ? TOKF_TRIPLE_STRING : 0));
}

/* Emits an empty cooked-text token; used to keep an f-string balanced. */
static void pushEmptyString(Scan *s, TokenKind kind, size_t at) {
    JaiBuf empty;
    jaiBufInit(&empty);
    uint32_t index = internString(s->lex, &empty);
    Token *t = pushToken(s, kind, at, at);
    t->v.strIndex = index;
}

/* Scans the tail of an f-string after the '}' that closed an interpolation. */
static void resumeFString(Scan *s, StrFlavor f, size_t tokStart) {
    Lexer *lex = s->lex;
    JaiBuf buf;
    jaiBufInit(&buf);
    bool sawEscape = false;
    ChunkStop stop = scanStringChunk(s, f, &buf, tokStart, &sawEscape);
    uint32_t index = internString(lex, &buf);

    /* A broken chunk still ends the f-string, so the parser sees a balanced
     * START ... END sequence and can keep going. */
    TokenKind kind = (stop == CHUNK_INTERP) ? TOK_FSTRING_MID : TOK_FSTRING_END;
    Token *t = pushToken(s, kind, tokStart, lex->pos);
    t->flags |= stringFlags(f, sawEscape);
    t->v.strIndex = index;

    if (stop == CHUNK_INTERP) openInterpolation(s, f);
}

/* `lex->pos` sits on the opening quote; `tokStart` is the start of the token,
 * which for a prefixed literal is the `r`/`f`. */
static void scanStringLiteral(Scan *s, size_t tokStart, bool raw, bool interp) {
    Lexer *lex = s->lex;
    char quote = lex->source[lex->pos];
    StrFlavor f;
    f.quote  = quote;
    f.triple = peekAt(lex, 1) == quote && peekAt(lex, 2) == quote;
    f.raw    = raw;
    f.interp = interp;
    lex->pos += f.triple ? 3 : 1;

    JaiBuf buf;
    jaiBufInit(&buf);
    bool sawEscape = false;
    ChunkStop stop = scanStringChunk(s, f, &buf, tokStart, &sawEscape);
    uint32_t index = internString(lex, &buf);

    if (!interp) {
        Token *t = pushToken(s, TOK_STRING, tokStart, lex->pos);
        t->flags |= stringFlags(f, sawEscape);
        t->v.strIndex = index;
        return;
    }

    Token *t = pushToken(s, TOK_FSTRING_START, tokStart, lex->pos);
    t->flags |= stringFlags(f, sawEscape);
    t->v.strIndex = index;

    if (stop == CHUNK_INTERP) openInterpolation(s, f);
    else pushEmptyString(s, TOK_FSTRING_END, lex->pos);   /* no interpolation */
}

/* A `:` at the top level of an interpolation introduces a format spec (§1.7).
 * The spec is data, not Jaithon source — `.2f` would not lex as a number — so
 * it is captured whole as a TOK_STRING for std.fmt to interpret. */
static void scanFormatSpec(Scan *s) {
    Lexer *lex = s->lex;
    size_t start = lex->pos;
    JaiBuf buf;
    jaiBufInit(&buf);

    int depth = 0;
    while (!atEnd(lex)) {
        char c = lex->source[lex->pos];
        if (c == '\n') break;              /* unterminated; reported at the '}' */
        if (c == '{') {
            depth++;
        } else if (c == '}') {
            if (depth == 0) break;
            depth--;
        }
        jaiBufPush(&buf, (uint8_t)c);
        lex->pos++;
    }

    uint32_t index = internString(lex, &buf);
    Token *t = pushToken(s, TOK_STRING, start, lex->pos);
    t->v.strIndex = index;
}

/* ------------------------------------------------------------------ */
/* Identifiers and labels                                              */
/* ------------------------------------------------------------------ */

/* End of the identifier run starting at `p`, without moving the cursor. */
static size_t identRunEnd(const Lexer *lex, size_t p) {
    while (p < lex->length) {
        unsigned char c = (unsigned char)lex->source[p];
        if (c < 0x80) {
            if (!isAsciiIdentCont((char)c)) break;
            p++;
            continue;
        }
        int len = 1;
        int32_t cp = jaiUtf8Decode(lex->source + p, lex->source + lex->length, &len);
        if (cp < 0 || !isUnicodeIdentScalar(cp)) break;
        p += (size_t)(len > 0 ? len : 1);
    }
    return p;
}

static void scanIdentifier(Scan *s) {
    Lexer *lex = s->lex;
    size_t start = lex->pos;
    lex->pos = identRunEnd(lex, start);
    size_t length = lex->pos - start;
    const char *text = lex->source + start;

    /* String prefixes `r`, `f` and their combination, only when the quote is
     * adjacent, so `r "x"` stays an identifier followed by a string. */
    if (length <= 2) {
        bool raw = false, interp = false, isPrefix = true;
        for (size_t i = 0; i < length; i++) {
            if (text[i] == 'r' && !raw) raw = true;
            else if (text[i] == 'f' && !interp) interp = true;
            else { isPrefix = false; break; }
        }
        if (isPrefix && (raw || interp) &&
            (peekc(lex) == '"' || peekc(lex) == '\'')) {
            scanStringLiteral(s, start, raw, interp);
            return;
        }
    }

    /* `__foo__` is reserved for the implementation (§1.4), but rejecting it is
     * the parser's call (E0114): the lexer emits a plain identifier because
     * the standard library legitimately declares such names. */
    TokenKind kind = jaiKeywordLookup(text, length);
    if (kind == TOK_IDENT && length == 1 && text[0] == '_') kind = TOK_UNDERSCORE;
    pushToken(s, kind, start, lex->pos);
}

/* True for the tokens a statement can begin after. Used to recognise the one
 * position where a label may be *declared*. */
static bool atStatementStart(const Lexer *lex) {
    switch (lastKind(lex)) {
    case TOK_EOF:        /* nothing emitted yet */
    case TOK_NEWLINE:
    case TOK_SEMICOLON:
    case TOK_LBRACE:
    case TOK_RBRACE:
        return true;
    default:
        return false;
    }
}

/* `'name` is a loop label, `'text'` is a string, and the grammar is what tells
 * them apart: a label is only ever declared as `'name:` at the start of a
 * statement, or used right after `break`/`continue`. Everything else that
 * starts with a quote is a string — including `'single quotes'`, whose first
 * word would otherwise read as a label. */
static void scanQuoteOrLabel(Scan *s) {
    Lexer *lex = s->lex;
    size_t start = lex->pos;
    size_t after = start + 1;

    if (after < lex->length) {
        unsigned char c = (unsigned char)lex->source[after];
        bool identStart;
        if (c < 0x80) {
            identStart = isAsciiIdentStart((char)c);
        } else {
            int len = 1;
            int32_t cp = jaiUtf8Decode(lex->source + after,
                                       lex->source + lex->length, &len);
            identStart = cp >= 0 && isUnicodeIdentScalar(cp);
        }

        if (identStart) {
            size_t end = identRunEnd(lex, after);
            bool closedByQuote = end < lex->length && lex->source[end] == '\'';
            size_t probe = end;
            while (probe < lex->length &&
                   (lex->source[probe] == ' ' || lex->source[probe] == '\t'))
                probe++;
            bool declaration = probe < lex->length && lex->source[probe] == ':' &&
                               atStatementStart(lex);
            bool usage = lastKind(lex) == TOK_KW_BREAK ||
                         lastKind(lex) == TOK_KW_CONTINUE;

            if (!closedByQuote && (declaration || usage)) {
                lex->pos = end;
                pushToken(s, TOK_LABEL, start, end);
                return;
            }
        }
    }

    scanStringLiteral(s, start, false, false);
}

/* ------------------------------------------------------------------ */
/* Punctuation and operators                                           */
/* ------------------------------------------------------------------ */

static void pushBracket(Lexer *lex, uint8_t kind) {
    JAI_VEC_PUSH(uint8_t, &lex->brackets, kind);
}

/* Pops only on a match: a stray closer is the parser's problem (E0107), and
 * popping blindly would corrupt the newline state of every enclosing
 * bracket. */
static void popBracket(Lexer *lex, uint8_t expected) {
    if (lex->brackets.count > 0 && JAI_VEC_LAST(&lex->brackets) == expected)
        lex->brackets.count--;
}

static void popBraceBracket(Lexer *lex) {
    if (lex->brackets.count == 0) return;
    uint8_t top = JAI_VEC_LAST(&lex->brackets);
    if (top == BRK_BRACE_BLOCK || top == BRK_BRACE_EXPR) lex->brackets.count--;
}

static void scanPunct(Scan *s) {
    Lexer *lex = s->lex;
    size_t start = lex->pos;
    char c = lex->source[lex->pos++];
    TokenKind kind;

    switch (c) {
    case '(': pushBracket(lex, BRK_PAREN);   kind = TOK_LPAREN; break;
    case '[': pushBracket(lex, BRK_BRACKET); kind = TOK_LBRACKET; break;
    case '{':
        pushBracket(lex, (uint8_t)(braceOpensExpression(lex) ? BRK_BRACE_EXPR
                                                             : BRK_BRACE_BLOCK));
        kind = TOK_LBRACE;
        break;
    case ')': popBracket(lex, BRK_PAREN);   kind = TOK_RPAREN; break;
    case ']': popBracket(lex, BRK_BRACKET); kind = TOK_RBRACKET; break;
    case '}':
        if (bracketTopIsInterp(lex)) {
            /* Closes an interpolation: no token, the f-string tail resumes.
             * `f"{}"` has to be caught here — it produces the same token shape
             * as an f-string with no interpolation at all, so the parser could
             * never tell the two apart. */
            if (lastKind(lex) == TOK_FSTRING_START ||
                lastKind(lex) == TOK_FSTRING_MID) {
                lex->hadError = true;
                JaiDiag *d = jaiDiagError(E0102_EXPECTED_EXPRESSION,
                                          spanOf(lex, start, lex->pos),
                                          "empty interpolation in f-string");
                if (d != NULL)
                    jaiDiagAddHelp(d, "write `{{}}` for literal braces");
            }
            uint8_t marker = JAI_VEC_POP(&lex->brackets);
            if (lex->interpDepth > 0) lex->interpDepth--;
            resumeFString(s, flavorFromMarker(marker), start);
            return;
        }
        /* §1.3: a NEWLINE immediately before `}` is discarded. */
        if (lex->tokens.count > 0 && lastKind(lex) == TOK_NEWLINE)
            lex->tokens.count--;
        popBraceBracket(lex);
        kind = TOK_RBRACE;
        break;

    case ',': kind = TOK_COMMA; break;
    case ';': kind = TOK_SEMICOLON; break;
    case '@': kind = TOK_AT; break;
    case '~': kind = TOK_TILDE; break;
    case ':':
        pushToken(s, TOK_COLON, start, lex->pos);
        if (bracketTopIsInterp(lex)) scanFormatSpec(s);
        return;

    case '.':
        if (eatChar(lex, '.')) {
            if (eatChar(lex, '.'))      kind = TOK_ELLIPSIS;
            else if (eatChar(lex, '=')) kind = TOK_DOTDOT_EQ;
            else                        kind = TOK_DOTDOT;
        } else {
            kind = TOK_DOT;
        }
        break;

    case '+':
        if (eatChar(lex, '='))      kind = TOK_PLUS_EQ;
        else if (eatChar(lex, '%')) kind = TOK_PLUS_PERCENT;
        else                        kind = TOK_PLUS;
        break;
    case '-':
        if (eatChar(lex, '>'))      kind = TOK_ARROW;
        else if (eatChar(lex, '=')) kind = TOK_MINUS_EQ;
        else if (eatChar(lex, '%')) kind = TOK_MINUS_PERCENT;
        else                        kind = TOK_MINUS;
        break;
    case '*':
        if (eatChar(lex, '*'))      kind = eatChar(lex, '=') ? TOK_STAR_STAR_EQ
                                                             : TOK_STAR_STAR;
        else if (eatChar(lex, '=')) kind = TOK_STAR_EQ;
        else if (eatChar(lex, '%')) kind = TOK_STAR_PERCENT;
        else                        kind = TOK_STAR;
        break;
    case '/':
        if (eatChar(lex, '/'))      kind = eatChar(lex, '=') ? TOK_SLASH_SLASH_EQ
                                                             : TOK_SLASH_SLASH;
        else if (eatChar(lex, '=')) kind = TOK_SLASH_EQ;
        else                        kind = TOK_SLASH;
        break;
    case '%': kind = eatChar(lex, '=') ? TOK_PERCENT_EQ : TOK_PERCENT; break;
    case '&': kind = eatChar(lex, '=') ? TOK_AMP_EQ : TOK_AMP; break;
    case '|': kind = eatChar(lex, '=') ? TOK_PIPE_EQ : TOK_PIPE; break;
    case '^': kind = eatChar(lex, '=') ? TOK_CARET_EQ : TOK_CARET; break;

    case '<':
        if (eatChar(lex, '<'))      kind = eatChar(lex, '=') ? TOK_SHL_EQ : TOK_SHL;
        else if (eatChar(lex, '=')) kind = TOK_LE;
        else                        kind = TOK_LT;
        break;
    case '>':
        if (eatChar(lex, '>'))      kind = eatChar(lex, '=') ? TOK_SHR_EQ : TOK_SHR;
        else if (eatChar(lex, '=')) kind = TOK_GE;
        else                        kind = TOK_GT;
        break;
    case '=':
        if (eatChar(lex, '='))      kind = TOK_EQ_EQ;
        else if (eatChar(lex, '>')) kind = TOK_FAT_ARROW;
        else                        kind = TOK_EQ;
        break;
    case '!':
        if (eatChar(lex, '=')) {
            kind = TOK_BANG_EQ;
            break;
        }
        lex->hadError = true;
        {
            JaiDiag *d = jaiDiagError(E0003_INVALID_CHARACTER,
                                      spanOf(lex, start, lex->pos),
                                      "unexpected character `!`");
            if (d != NULL)
                jaiDiagAddHelp(d, "Jaithon spells logical negation `not` and "
                                  "inequality `!=`");
        }
        return;
    case '?':
        if (eatChar(lex, '?'))      kind = TOK_QUESTION_QUESTION;
        else if (eatChar(lex, '.')) kind = TOK_QUESTION_DOT;
        else                        kind = TOK_QUESTION;
        break;

    default:
        lex->hadError = true;
        {
            JaiDiag *d;
            if ((unsigned char)c < 0x20 || (unsigned char)c == 0x7F)
                d = jaiDiagError(E0003_INVALID_CHARACTER,
                                 spanOf(lex, start, lex->pos),
                                 "unexpected control character U+%04X",
                                 (unsigned)(unsigned char)c);
            else
                d = jaiDiagError(E0003_INVALID_CHARACTER,
                                 spanOf(lex, start, lex->pos),
                                 "unexpected character `%c`", c);
            (void)d;
        }
        return;
    }

    pushToken(s, kind, start, lex->pos);
}

/* ------------------------------------------------------------------ */
/* Main loop                                                           */
/* ------------------------------------------------------------------ */

static void scanToken(Scan *s) {
    Lexer *lex = s->lex;
    char c = lex->source[lex->pos];
    unsigned char uc = (unsigned char)c;

    switch (c) {
    case ' ': case '\t': case '\v': case '\f':
        lex->pos++;
        return;
    case '\r':
        lex->pos++;              /* §1.1: discarded, the '\n' terminates */
        return;
    case '\n':
        handleNewline(s);
        return;
    case '#':
        skipComment(s);
        return;
    case '\\': {
        /* Explicit line continuation. */
        size_t start = lex->pos;
        size_t next = start + 1;
        if (next < lex->length && lex->source[next] == '\r') next++;
        if (next < lex->length && lex->source[next] == '\n') {
            lex->pos = next + 1;
            lex->line++;
            return;
        }
        lex->pos++;
        lex->hadError = true;
        JaiDiag *d = jaiDiagError(E0003_INVALID_CHARACTER,
                                  spanOf(lex, start, lex->pos),
                                  "stray `\\` outside a string literal");
        if (d != NULL)
            jaiDiagAddHelp(d, "a backslash continues a line only at end of line");
        return;
    }
    default:
        break;
    }

    if (uc >= 0x80) {
        int len = 1;
        int32_t cp = jaiUtf8Decode(lex->source + lex->pos,
                                   lex->source + lex->length, &len);
        if (len < 1) len = 1;
        if (cp < 0) {
            lex->hadError = true;
            JaiDiag *d = jaiDiagError(E0006_INVALID_UTF8,
                                      spanOf(lex, lex->pos, lex->pos + (size_t)len),
                                      "invalid UTF-8 byte 0x%02X", uc);
            if (d != NULL) jaiDiagAddNote(d, "Jaithon source must be valid UTF-8");
            lex->pos += (size_t)len;
            return;
        }
        if (isUnicodeSpaceScalar(cp)) {
            lex->pos += (size_t)len;
            return;
        }
        if (!isUnicodeIdentScalar(cp)) {
            lex->hadError = true;
            JaiDiag *d = jaiDiagError(E0003_INVALID_CHARACTER,
                                      spanOf(lex, lex->pos, lex->pos + (size_t)len),
                                      "unexpected character U+%04X", cp);
            if (d != NULL)
                jaiDiagAddHelp(d, "only ASCII punctuation is meaningful outside "
                                  "strings and comments");
            lex->pos += (size_t)len;
            return;
        }
        scanIdentifier(s);
        return;
    }

    if (isAsciiDigit(c))      { scanNumber(s); return; }
    if (isAsciiIdentStart(c)) { scanIdentifier(s); return; }
    if (c == '"')             { scanStringLiteral(s, lex->pos, false, false); return; }
    if (c == '\'')            { scanQuoteOrLabel(s); return; }

    scanPunct(s);
}

/* ------------------------------------------------------------------ */
/* Public API                                                          */
/* ------------------------------------------------------------------ */

void jaiLexerInit(Lexer *lex, const char *source, size_t length, int fileId) {
    if (lex == NULL) return;
    lex->source = source != NULL ? source : "";
    lex->length = source != NULL ? length : 0;
    lex->fileId = fileId;
    lex->pos    = 0;
    lex->line   = 1;
    JAI_VEC_INIT(&lex->tokens);
    JAI_VEC_INIT(&lex->strings);
    JAI_VEC_INIT(&lex->stringSizes);
    JAI_VEC_INIT(&lex->brackets);
    lex->interpDepth = 0;
    lex->hadError    = false;
}

/* Cooked literals are freed with the size internString handed out, not
 * strlen+1: `"\0"` cooks to one NUL byte, whose strlen is zero. The allocator
 * bins blocks by the size it is told, so an under-reported free corrupts a
 * bin rather than merely skewing a counter. */
static void freeCookedStrings(Lexer *lex) {
    for (int i = 0; i < lex->strings.count; i++) {
        char *str = lex->strings.data[i];
        if (str != NULL) JAI_FREE_ARRAY(char, str, lex->stringSizes.data[i]);
    }
}

void jaiLexerFree(Lexer *lex) {
    if (lex == NULL) return;
    freeCookedStrings(lex);
    JAI_VEC_FREE(char *, &lex->strings);
    JAI_VEC_FREE(size_t, &lex->stringSizes);
    JAI_VEC_FREE(Token, &lex->tokens);
    JAI_VEC_FREE(uint8_t, &lex->brackets);
    lex->interpDepth = 0;
}

bool jaiLexerRun(Lexer *lex) {
    if (lex == NULL) return false;

    /* A rerun starts from scratch; the previous token stream and its cooked
     * strings are dropped. */
    freeCookedStrings(lex);
    lex->strings.count  = 0;
    lex->stringSizes.count = 0;
    lex->tokens.count   = 0;
    lex->brackets.count = 0;
    lex->interpDepth    = 0;
    lex->hadError       = false;
    lex->pos            = 0;
    lex->line           = 1;

    Scan s;
    s.lex = lex;
    s.atLineStart = true;

    if (lex->length >= 3 && (unsigned char)lex->source[0] == 0xEF &&
        (unsigned char)lex->source[1] == 0xBB &&
        (unsigned char)lex->source[2] == 0xBF)
        lex->pos = 3;   /* UTF-8 BOM */

    while (!atEnd(lex)) scanToken(&s);

    /* An interpolation still open at EOF means the f-string never closed. Each
     * open one gets an empty FSTRING_END so the parser sees a balanced
     * sequence instead of a truncated one. */
    if (lex->interpDepth > 0) {
        lex->hadError = true;
        JaiDiag *d = jaiDiagError(E0008_UNTERMINATED_INTERPOLATION,
                                  spanOf(lex, lex->length, lex->length),
                                  "unterminated f-string interpolation at end of file");
        if (d != NULL) jaiDiagAddHelp(d, "close it with `}` and the string's quote");
        while (lex->brackets.count > 0) {
            uint8_t top = JAI_VEC_POP(&lex->brackets);
            if (top >= BRK_INTERP) pushEmptyString(&s, TOK_FSTRING_END, lex->length);
        }
        lex->interpDepth = 0;
    }

    pushToken(&s, TOK_EOF, lex->length, lex->length);
    return !lex->hadError;
}

const char *jaiTokenText(const Lexer *lex, const Token *tok, size_t *outLen) {
    if (lex == NULL || tok == NULL || lex->source == NULL ||
        tok->start > lex->length) {
        if (outLen != NULL) *outLen = 0;
        return "";
    }
    size_t length = tok->length;
    if (tok->start + length > lex->length) length = lex->length - tok->start;
    if (outLen != NULL) *outLen = length;
    return lex->source + tok->start;
}

const char *jaiTokenStringValue(const Lexer *lex, const Token *tok, size_t *outLen) {
    if (lex == NULL || tok == NULL) {
        if (outLen != NULL) *outLen = 0;
        return "";
    }
    switch (tok->kind) {
    case TOK_STRING:
    case TOK_FSTRING_START:
    case TOK_FSTRING_MID:
    case TOK_FSTRING_END: {
        /* Cooked text is only reachable while the lexer still owns the arena;
         * after jaiLexerTakeStrings the caller indexes it directly. */
        if (lex->strings.data == NULL ||
            tok->v.strIndex >= (uint32_t)lex->strings.count) {
            if (outLen != NULL) *outLen = 0;
            return "";
        }
        const char *str = lex->strings.data[tok->v.strIndex];
        if (str == NULL) {
            if (outLen != NULL) *outLen = 0;
            return "";
        }
        /* NOT strlen: `"a\0b"` cooks to three bytes whose strlen is one, and
         * every length downstream — the constant, `len()`, indexing, slicing,
         * concatenation — is derived from what is returned here. `stringSizes`
         * is the size internString handed out, which is the cooked length plus
         * the terminator it appends. */
        if (outLen != NULL) {
            size_t size = 0;
            if (tok->v.strIndex < (uint32_t)lex->stringSizes.count)
                size = lex->stringSizes.data[tok->v.strIndex];
            *outLen = size > 0 ? size - 1 : 0;
        }
        return str;
    }
    default:
        return jaiTokenText(lex, tok, outLen);
    }
}

JaiSpan jaiTokenSpan(const Lexer *lex, const Token *tok) {
    if (lex == NULL || tok == NULL) return JAI_SPAN_NONE;
    return spanOf(lex, tok->start, (size_t)tok->start + tok->length);
}

char **jaiLexerTakeStrings(Lexer *lex, int *outCount) {
    if (lex == NULL) {
        if (outCount != NULL) *outCount = 0;
        return NULL;
    }
    char **data = lex->strings.data;
    if (outCount != NULL) *outCount = lex->strings.count;
    JAI_VEC_INIT(&lex->strings);
    return data;
}

void jaiLexerDump(FILE *out, const Lexer *lex) {
    if (out == NULL || lex == NULL) return;

    /* Tokens are in source order, so one cursor walk computes every position;
     * columns count scalars, not bytes (§1.1). */
    size_t offset = 0;
    int line = 1, col = 1;

    for (int i = 0; i < lex->tokens.count; i++) {
        const Token *tok = &lex->tokens.data[i];
        while (offset < tok->start && offset < lex->length) {
            if (lex->source[offset] == '\n') {
                line++;
                col = 1;
                offset++;
                continue;
            }
            int len = 1;
            if ((unsigned char)lex->source[offset] >= 0x80) {
                jaiUtf8Decode(lex->source + offset, lex->source + lex->length, &len);
                if (len < 1) len = 1;
            }
            offset += (size_t)len;
            col++;
        }

        fprintf(out, "%5d  %-22s %4d:%-4d ", i, jaiTokenKindName((TokenKind)tok->kind),
                line, col);

        size_t textLen = 0;
        const char *text = jaiTokenText(lex, tok, &textLen);
        if (textLen == 0) {
            fprintf(out, "%s", jaiTokenKindText((TokenKind)tok->kind));
        } else {
            for (size_t j = 0; j < textLen; j++) {
                char ch = text[j];
                if (ch == '\n')      fputs("\\n", out);
                else if (ch == '\t') fputs("\\t", out);
                else if (ch == '\r') fputs("\\r", out);
                else                 fputc(ch, out);
            }
        }
        fputc('\n', out);
    }
}
