#ifndef JAI_RUNTIME_FRONTEND_H
#define JAI_RUNTIME_FRONTEND_H

#include "../vm/value.h"

/* The C side of the boundary with the self-hosted front end.
 *
 * The compiler is a Jaithon program, so every question the driver asks it is a
 * VM call that returns a Jaithon object. The mechanics of that -- import the
 * module, find the entry, root the arguments, read a field back off the result
 * -- are the same for every question, and they live here so that no driver has
 * to know them twice. */

/* Read `name` off a Jaithon instance. False when `v` is not an instance or has
 * no such field, in which case `*out` is untouched. */
bool jaiFrontEndField(Value v, const char *name, Value *out);

/* Call the zero-argument method `name` on `v`. False when there is no such
 * method or the call raised; a raise is cleared, not reported. */
bool jaiFrontEndCall0(Value v, const char *name, Value *out);

/* Call `module.name(args...)`. False when the module cannot be imported, does
 * not export `name`, or the call raised -- the raise is reported and cleared,
 * because a front end that throws is a bug worth a traceback. */
bool jaiFrontEndInvoke(const char *module, const char *name, int argc,
                       Value *args, Value *out);

/* ------------------------------------------------------------------ */
/* The prompt                                                           */
/* ------------------------------------------------------------------ */

/* What is holding an input open. Mirrors `OpenKind.open_ordinal` in
 * lib/jaithon/compile/repl.jai; the two are changed together. */
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

/* Ask the front end whether `source` is a whole input.
 *
 * False means the question could not be asked -- no front end on the path --
 * and leaves `*out` zeroed, which reads as "complete": the compile that
 * follows will report the real reason rather than the prompt hanging on a line
 * nothing can finish. */
bool jaiFrontEndReplScan(const char *source, size_t length, JaiReplScan *out);

/* Compile one prompt input into `module`.
 *
 * The session -- the type names earlier inputs declared -- lives on the Jaithon
 * side and is held for the process, so a class declared on one line is
 * nameable in an annotation on the next.
 *
 * `echo` names the native a bare expression's value is handed to, or NULL to
 * discard it. `wholeFile` reads the input as a file rather than as one prompt
 * line, which is what `:load` wants.
 *
 * NULL when the input was rejected; its diagnostics have been printed. */
typedef struct {
    const char *path;       /* the label the input is registered under */
    int         fileId;
    int         optLevel;
    const char *echo;       /* native a bare expression's value goes to, or NULL */
    bool        wholeFile;  /* read as a file rather than as one prompt line */
    bool        record;     /* let what it declares outlive the input */
} JaiReplCompileOptions;

ObjFunction *jaiFrontEndReplCompile(const char *source, size_t length,
                                    const JaiReplCompileOptions *opts,
                                    ObjModule *module, bool *outWasExpression);

/* Rebuild the front end's own diagnostics as JaiDiag in `gDiags`, so the
 * driver renders them exactly as it renders the C's -- with the source
 * excerpt, the caret and the help. True when there were any.
 *
 * `Compiled.report()` is the alternative and it is a flat line per diagnostic;
 * printing that was a real loss of quality that no gate caught. */
bool jaiFrontEndTransferDiagnostics(Value compiled);

/* Forget every declaration the session recorded, for `:reset`. */
void jaiFrontEndReplForget(void);

#endif /* JAI_RUNTIME_FRONTEND_H */
