#ifndef JAI_RUNTIME_FRONTEND_H
#define JAI_RUNTIME_FRONTEND_H

#include "vm/value.h"

/* The C side of the boundary with the self-hosted front end. */

bool jaiFrontEndField(Value v, const char *name, Value *out);

bool jaiFrontEndCall0(Value v, const char *name, Value *out);

bool jaiFrontEndInvoke(const char *module, const char *name, int argc,
                       Value *args, Value *out);

/* ------------------------------------------------------------------ */
/* The prompt                                                           */
/* ------------------------------------------------------------------ */

typedef enum {
    JAI_REPL_OPEN_NONE = 0,
    JAI_REPL_OPEN_BRACKET,
    JAI_REPL_OPEN_STRING,
    JAI_REPL_OPEN_COMMENT,
    JAI_REPL_OPEN_OPERATOR,
} JaiReplOpenKind;

typedef struct {
    bool            incomplete;
    bool            mismatched;
    JaiReplOpenKind open;
    char            opener;
    int             openerLine;
    char            closer;
    int             closerLine;
    int             closerCol;
} JaiReplScan;

bool jaiFrontEndReplScan(const char *source, size_t length, JaiReplScan *out);

typedef struct {
    const char *path;
    int         fileId;
    int         optLevel;
    const char *echo;
    bool        wholeFile;
    bool        record;
    const char *sourceDir;
    bool        strict;
    bool        lateGlobals;
} JaiReplCompileOptions;

ObjFunction *jaiFrontEndReplCompile(const char *source, size_t length,
                                    const JaiReplCompileOptions *opts,
                                    ObjModule *module, bool *outWasExpression);

bool jaiFrontEndTransferDiagnostics(Value compiled);

/* ------------------------------------------------------------------ */
/* Dumps                                                                */
/* ------------------------------------------------------------------ */

ObjString *jaiFrontEndAstText(const char *source, size_t length,
                              const char *path, int fileId, bool json);
ObjString *jaiFrontEndTokenText(const char *source, size_t length, int fileId);

void jaiFrontEndReplForget(void);

#endif /* JAI_RUNTIME_FRONTEND_H */
