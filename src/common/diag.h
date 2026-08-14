#ifndef JAI_DIAG_H
#define JAI_DIAG_H

#include "common/common.h"

typedef struct {
    int         id;
    char       *path;
    char       *source;
    size_t      length;
    uint32_t   *lineStarts;
    int         lineCount;
} JaiSourceFile;

typedef struct {
    uint32_t start;
    uint32_t end;
    int32_t  file;
} JaiSpan;

#define JAI_SPAN_NONE ((JaiSpan){0, 0, -1})

JAI_INLINE bool jaiSpanValid(JaiSpan s) { return s.file >= 0 && s.end >= s.start; }
JaiSpan jaiSpanJoin(JaiSpan a, JaiSpan b);

int             jaiSourceAdd(const char *path, char *source, size_t length);
JaiSourceFile  *jaiSourceGet(int id);
void            jaiSourceFreeAll(void);
void            jaiSourceLineCol(int fileId, uint32_t offset, int *line, int *col);
const char     *jaiSourceLineText(int fileId, uint32_t offset, size_t *outLen,
                                  int *outLine);


// error codes
typedef enum {
    JAI_OK = 0,

    /* E00xx lexical */
    E0001_UNTERMINATED_STRING = 1,
    E0002_UNTERMINATED_COMMENT,
    E0003_INVALID_CHARACTER,
    E0004_INVALID_ESCAPE,
    E0005_INVALID_NUMBER,
    E0006_INVALID_UTF8,
    E0007_TAB_IN_INDENT,
    E0008_UNTERMINATED_INTERPOLATION,
    E0009_NESTED_INTERPOLATION_TOO_DEEP,

    /* E01xx syntax */
    E0100_UNEXPECTED_TOKEN = 100,
    E0101_EXPECTED_TOKEN,
    E0102_EXPECTED_EXPRESSION,
    E0103_EXPECTED_STATEMENT,
    E0104_EXPECTED_BLOCK,
    E0105_EXPECTED_TYPE,
    E0106_EXPECTED_IDENTIFIER,
    E0107_UNCLOSED_DELIMITER,
    E0108_TRAILING_TOKENS,
    E0109_INVALID_ASSIGN_TARGET,
    E0110_INVALID_PATTERN,
    E0111_DUPLICATE_PARAMETER,
    E0112_PARAM_AFTER_VARIADIC,
    E0113_DEFAULT_BEFORE_REQUIRED,
    E0114_RESERVED_IDENTIFIER,
    E0115_LABEL_NOT_ON_LOOP,

    /* E02xx name resolution */
    E0200_UNDEFINED_NAME = 200,
    E0201_USE_BEFORE_DECLARATION,
    E0202_UNDEFINED_LABEL,
    E0203_BREAK_OUTSIDE_LOOP,
    E0204_CONTINUE_OUTSIDE_LOOP,
    E0205_RETURN_OUTSIDE_FUNCTION,
    E0206_SELF_OUTSIDE_METHOD,
    E0207_SUPER_OUTSIDE_SUBCLASS,
    E0208_TOO_MANY_LOCALS,
    E0209_TOO_MANY_UPVALUES,
    E0210_YIELD_OUTSIDE_GENERATOR,
    E0211_DEFER_OUTSIDE_FUNCTION,

    /* E03xx bindings */
    E0300_UNDEFINED_VARIABLE = 300,
    E0301_ASSIGN_TO_IMMUTABLE,
    E0302_DUPLICATE_DECLARATION,
    E0303_NON_CONSTANT_CONST,
    E0304_MISSING_INITIALIZER,
    E0305_ASSIGN_TO_CONST,

    /* E04xx types */
    E0400_TYPE_MISMATCH = 400,
    E0401_CONDITION_NOT_BOOL,
    E0402_UNKNOWN_TYPE,
    E0403_NOT_CALLABLE,
    E0404_NOT_INDEXABLE,
    E0405_NOT_ITERABLE,
    E0406_BAD_OPERAND_TYPES,
    E0407_GENERIC_ARITY,
    E0408_RETURN_TYPE_MISMATCH,
    E0409_VOID_VALUE_USED,
    E0410_UNKNOWN_MEMBER,
    E0411_INT_FLOAT_MIX,

    /* E05xx match */
    E0500_UNREACHABLE_ARM = 500,
    E0501_NON_EXHAUSTIVE_MATCH,
    E0502_PATTERN_ARITY,
    E0503_DUPLICATE_BINDING_IN_PATTERN,

    /* E06xx functions */
    E0600_ARITY_MISMATCH = 600,
    E0601_MISSING_RETURN,
    E0602_BAD_CALL_ARITY,
    E0603_UNKNOWN_KEYWORD_ARG,
    E0604_DUPLICATE_ARGUMENT,
    E0605_POSITIONAL_AFTER_KEYWORD,

    /* E07xx classes */
    E0700_UNKNOWN_CLASS = 700,
    E0701_PRIVATE_ACCESS,
    E0702_UNDECLARED_FIELD,
    E0703_MISSING_SELF,
    E0704_INCOMPATIBLE_OVERRIDE,
    E0705_TRAIT_NOT_IMPLEMENTED,
    E0706_CYCLIC_INHERITANCE,
    E0707_STATIC_WITH_SELF,
    E0708_SUPER_INIT_NOT_FIRST,
    E0709_DUPLICATE_MEMBER,
    E0710_ABSTRACT_INSTANTIATION,

    /* E08xx modules */
    E0800_MODULE_NOT_FOUND = 800,
    E0801_CIRCULAR_IMPORT,
    E0802_NOT_EXPORTED,
    E0803_DUPLICATE_IMPORT,
    E0804_INVALID_MODULE_PATH,

    /* E09xx const evaluation and internal */
    E0900_CONST_OVERFLOW = 900,
    E0901_CONST_DIVIDE_BY_ZERO,
    E0902_INTERNAL_ERROR,

    /* W01xx warnings */
    W0100_UNUSED_IMPORT = 10000,
    W0101_UNUSED_BINDING,
    W0102_UNREACHABLE_CODE,
    W0103_SHADOWED_BINDING,
    W0104_EMPTY_BLOCK,
    W0105_DEPRECATED,
    W0106_UNUSED_PARAMETER,
} JaiDiagCode;

const char *jaiDiagCodeString(JaiDiagCode code);
JaiDiagCode jaiDiagCodeFromString(const char *text);
bool        jaiDiagCodeIsWarning(JaiDiagCode code);

#define JAI_SUGGEST_MAX_LEN 64

int  jaiNameDistance(const char *a, const char *b, int max);

#define JAI_SUGGEST_NO_MATCH 4
bool jaiNameIsCloser(const char *name, const char *candidate, int *best);

typedef enum { JAI_SEV_ERROR, JAI_SEV_WARNING, JAI_SEV_NOTE, JAI_SEV_HELP } JaiSeverity;

typedef struct {
    JaiSpan  span;
    char    *label;
} JaiDiagLabel;

typedef struct {
    JaiDiagCode  code;
    JaiSeverity  severity;
    char        *message;
    JaiSpan      primary;
    JaiDiagLabel secondary[4];
    int          secondaryCount;
    char        *help;
    char        *note;
} JaiDiag;

typedef struct {
    JAI_VEC(JaiDiag) diags;
    int   errorCount;
    int   warningCount;
    int   maxErrors;
    bool  warningsAsErrors;
    bool  colorOutput;
    bool  quiet;
} JaiDiagBag;

extern JaiDiagBag gDiags;

void jaiDiagInit(JaiDiagBag *bag);
void jaiDiagFree(JaiDiagBag *bag);
void jaiDiagReset(JaiDiagBag *bag);

JaiDiag *jaiDiagError(JaiDiagCode code, JaiSpan span, const char *fmt, ...)
    JAI_PRINTF(3, 4);
JaiDiag *jaiDiagWarn(JaiDiagCode code, JaiSpan span, const char *fmt, ...)
    JAI_PRINTF(3, 4);

void jaiDiagAddLabel(JaiDiag *d, JaiSpan span, const char *fmt, ...) JAI_PRINTF(3, 4);
void jaiDiagAddHelp(JaiDiag *d, const char *fmt, ...) JAI_PRINTF(2, 3);
void jaiDiagAddNote(JaiDiag *d, const char *fmt, ...) JAI_PRINTF(2, 3);

bool jaiDiagFlush(JaiDiagBag *bag, FILE *out);
bool jaiDiagHasErrors(const JaiDiagBag *bag);

void jaiDiagRender(const JaiDiag *d, FILE *out, bool color);

typedef struct {
    const char *functionName;
    const char *modulePath;
    JaiSpan     span;
} JaiFrameInfo;

void jaiPrintTraceback(FILE *out, const JaiFrameInfo *frames, int count,
                       const char *excType, const char *excMessage, bool color);

#endif
