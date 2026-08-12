#ifndef JAI_RUNTIME_FRONTEND_H
#define JAI_RUNTIME_FRONTEND_H

#include "../vm/value.h"

/* The C side of the boundary with the self-hosted front end: import, call,
 * and read results, in one place so no driver needs to know it twice. */

/* False when `v` isn't an instance or has no such field; `*out` is left
 * untouched. */
bool jaiFrontEndField(Value v, const char *name, Value *out);

/* False if there's no such method or the call raised; a raise is cleared, not
 * reported. */
bool jaiFrontEndCall0(Value v, const char *name, Value *out);

/* False if the module can't be imported/doesn't export `name`, or the call
 * raised — here the raise is reported (not just cleared): a front end that
 * throws is a bug worth a traceback. */
bool jaiFrontEndInvoke(const char *module, const char *name, int argc,
                       Value *args, Value *out);

/* ------------------------------------------------------------------ */
/* The prompt                                                           */
/* ------------------------------------------------------------------ */

/* Mirrors `OpenKind.open_ordinal` in lib/jaithon/compile/repl.jai; change both
 * together. */
typedef enum {
    JAI_REPL_OPEN_NONE = 0,
    JAI_REPL_OPEN_BRACKET,   /* `(`, `[` or `{` is still waiting for its closer */
    JAI_REPL_OPEN_STRING,    /* a triple-quoted string runs past the last line  */
    JAI_REPL_OPEN_COMMENT,   /* a `#*` block comment has no `*#` yet            */
    JAI_REPL_OPEN_OPERATOR,  /* the last line ends on something that binds on   */
} JaiReplOpenKind;

/* The verdict on one buffer: run it, keep reading, or reject it now. */
typedef struct {
    bool            incomplete;
    bool            mismatched;  /* a closer nothing opened: more input cannot help */
    JaiReplOpenKind open;
    char            opener;      /* the bracket still open, or wrongly closed */
    int             openerLine;
    char            closer;      /* the offending closer, when mismatched */
    int             closerLine;
    int             closerCol;
} JaiReplScan;

/* False (no front end on the path) leaves `*out` zeroed, which reads as
 * "complete" — the compile that follows reports the real reason. */
bool jaiFrontEndReplScan(const char *source, size_t length, JaiReplScan *out);

/* The session (type names declared so far) persists across calls, so a class
 * declared on one line is nameable in an annotation on the next. NULL when
 * the input was rejected — diagnostics already printed. */
typedef struct {
    const char *path;       /* the label the input is registered under */
    int         fileId;
    int         optLevel;
    const char *echo;       /* native a bare expression's value goes to, or NULL */
    bool        wholeFile;  /* read as a file rather than as one prompt line */
    bool        record;     /* let what it declares outlive the input */
    const char *sourceDir;  /* where `from m import T` starts looking */
    bool        strict;     /* --strict: every parameter must be annotated */
    bool        lateGlobals; /* a name a running module defined is deferred */
} JaiReplCompileOptions;

ObjFunction *jaiFrontEndReplCompile(const char *source, size_t length,
                                    const JaiReplCompileOptions *opts,
                                    ObjModule *module, bool *outWasExpression);

/* Rebuilds the diagnostics as JaiDiag in `gDiags`, so they render exactly
 * like the C front end's (source excerpt, caret, help) rather than as
 * `Compiled.report()`'s flat line per diagnostic. True when there were any. */
bool jaiFrontEndTransferDiagnostics(Value compiled);

/* ------------------------------------------------------------------ */
/* Dumps                                                                */
/* ------------------------------------------------------------------ */

/* NULL when the front end couldn't be reached or refused the input (reason in
 * gDiags, or thrown). The returned string is only guaranteed live until the
 * next allocation — callers write it out and drop it. */
ObjString *jaiFrontEndAstText(const char *source, size_t length,
                              const char *path, int fileId, bool json);
ObjString *jaiFrontEndTokenText(const char *source, size_t length, int fileId);

/* Forget every declaration the session recorded, for `:reset`. */
void jaiFrontEndReplForget(void);

#endif /* JAI_RUNTIME_FRONTEND_H */
