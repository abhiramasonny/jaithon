/* parser.h — recursive-descent + Pratt parser producing an AST.
 *
 * The parser never executes anything. It never reads a file. It never touches
 * the runtime. Given tokens, it produces an AST or reports diagnostics. This
 * separation is what Jaithon 2 lacked.
 */
#ifndef JAI_PARSER_H
#define JAI_PARSER_H

#include "ast.h"
#include "lexer.h"

typedef struct {
    Lexer      *lex;
    AstContext *ast;
    int         current;      /* index into lex->tokens */
    bool        hadError;
    bool        panicMode;
    int         loopDepth;
    int         fnDepth;
    int         exprDepth;    /* recursion guard */
} Parser;

void jaiParserInit(Parser *p, Lexer *lex, AstContext *ast);

/* Parse a whole module. Returns an AST_PROGRAM node, or NULL if parsing failed
 * so badly that no tree could be produced. Diagnostics land in gDiags. */
AstNode *jaiParseProgram(Parser *p);

/* Parse a single statement or expression — used by the REPL, which needs to
 * distinguish "incomplete input" (keep reading) from "syntax error". */
AstNode *jaiParseREPLLine(Parser *p, bool *outIncomplete);

/* Parse a standalone expression; used by const-expression contexts and tests. */
AstNode *jaiParseExpressionOnly(Parser *p);

/* Convenience: source text -> AST, running the lexer internally.
 * `fileId` must already be registered with jaiSourceAdd. Returns NULL on error. */
AstNode *jaiParseSource(AstContext *ast, Lexer *lexOut, const char *source,
                        size_t length, int fileId);

/* ------------------------------------------------------------------ */
/* Precedence — mirrors spec §4.1                                       */
/* ------------------------------------------------------------------ */

typedef enum {
    PREC_NONE = 0,
    PREC_TERNARY,      /*  ?:            */
    PREC_COALESCE,     /*  ??            */
    PREC_OR,           /*  or            */
    PREC_AND,          /*  and           */
    PREC_NOT,          /*  not           */
    PREC_COMPARISON,   /*  < <= > >= == != is in */
    PREC_RANGE,        /*  .. ..=        */
    PREC_BITOR,        /*  |             */
    PREC_BITXOR,       /*  ^             */
    PREC_BITAND,       /*  &             */
    PREC_SHIFT,        /*  << >>         */
    PREC_TERM,         /*  + -           */
    PREC_FACTOR,       /*  * / // % @    */
    PREC_UNARY,        /*  - + ~         */
    PREC_POWER,        /*  **            */
    PREC_POSTFIX,      /*  . () [] ?.    */
    PREC_PRIMARY
} Precedence;

#endif /* JAI_PARSER_H */
