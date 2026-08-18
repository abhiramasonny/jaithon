#ifndef JAI_TRACE_H
#define JAI_TRACE_H

#include <stdbool.h>
#include <stdint.h>

struct ObjFunction;

#define JAI_TRACE_MAX_OPS 256

typedef struct {
    char     name[32];
    int      shape[4];
    int      shape_len;
} JaiTraceOp;

typedef struct {
    struct ObjFunction *fn;
    uint32_t            run_id;
    int                 op_count;
    int                 replay_at;
    JaiTraceOp          ops[JAI_TRACE_MAX_OPS];
    bool                shapes_match;
    bool                replaying;
} JaiTraceSession;

void     jaiTraceInit(void);
void     jaiTraceEnter(struct ObjFunction *fn);
void     jaiTraceLeave(struct ObjFunction *fn);
bool     jaiTraceIsActive(void);
uint32_t jaiTraceRunId(void);
bool     jaiTraceRecordOp(const char *name, const int *shape, int shape_len);
int      jaiTraceOpCount(void);
bool     jaiTraceReplay(void);

#endif
