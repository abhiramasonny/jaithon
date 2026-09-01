/* builtins_gpu_train.c — the fused MLP training primitives and the label
 * check that guards them. */

#include "runtime/builtins/platform/builtins_gpu.h"

#include "native/native.h"

bool nGpuMlpSgdStep(int argc, Value *args, Value *out) {
    (void)argc;
    GpuBuffer *x, *w1, *b1, *w2, *b2, *labels, *lossAcc, *correctAcc;
    if (!requireBuffer(args[0], 1, "gpu_mlp_sgd_step", &x)) return false;
    if (!requireBuffer(args[1], 2, "gpu_mlp_sgd_step", &w1)) return false;
    if (!requireBuffer(args[2], 3, "gpu_mlp_sgd_step", &b1)) return false;
    if (!requireBuffer(args[3], 4, "gpu_mlp_sgd_step", &w2)) return false;
    if (!requireBuffer(args[4], 5, "gpu_mlp_sgd_step", &b2)) return false;
    if (!requireBuffer(args[5], 6, "gpu_mlp_sgd_step", &labels)) return false;
    if (!requireBuffer(args[6], 7, "gpu_mlp_sgd_step", &lossAcc)) return false;
    correctAcc = NULL;
    if (!IS_NULL(args[7])) {
        if (!requireBuffer(args[7], 8, "gpu_mlp_sgd_step", &correctAcc)) return false;
    }

    int64_t batch, inputs, hidden, classes;
    double lr;
    if (!jaiArgInt(args[8], 9, "gpu_mlp_sgd_step", &batch)) return false;
    if (!jaiArgInt(args[9], 10, "gpu_mlp_sgd_step", &inputs)) return false;
    if (!jaiArgInt(args[10], 11, "gpu_mlp_sgd_step", &hidden)) return false;
    if (!jaiArgInt(args[11], 12, "gpu_mlp_sgd_step", &classes)) return false;
    if (!jaiArgNumber(args[12], 13, "gpu_mlp_sgd_step", &lr)) return false;
    if (batch <= 0 || inputs <= 0 || hidden <= 0 || classes <= 0 ||
        batch > UINT32_MAX || inputs > UINT32_MAX || hidden > UINT32_MAX ||
        classes > UINT32_MAX)
        return jaiThrow(vm.cValueError,
                        "gpu_mlp_sgd_step(): dimensions must be positive uint32, "
                        "got %lldx%lldx%lldx%lld",
                        (long long)batch, (long long)inputs, (long long)hidden,
                        (long long)classes);
    if (!(lr > 0.0) || lr > 1e6)
        return jaiThrow(vm.cValueError, "gpu_mlp_sgd_step(): learning rate is invalid");

    bool ok = jaiGpuMlpSgdStep(
        x->buffer, (size_t)x->origin * sizeof(float),
        w1->buffer, (size_t)w1->origin * sizeof(float),
        b1->buffer, (size_t)b1->origin * sizeof(float),
        w2->buffer, (size_t)w2->origin * sizeof(float),
        b2->buffer, (size_t)b2->origin * sizeof(float),
        labels->buffer, (size_t)labels->origin * sizeof(float),
        lossAcc->buffer, (size_t)lossAcc->origin * sizeof(float),
        correctAcc != NULL ? correctAcc->buffer : NULL,
        correctAcc != NULL ? (size_t)correctAcc->origin * sizeof(float) : 0,
        (uint32_t)batch, (uint32_t)inputs, (uint32_t)hidden, (uint32_t)classes,
        (float)lr);
    if (!ok)
        return jaiThrow(vm.cRuntimeError, "gpu_mlp_sgd_step(): the kernel failed");
    *out = NULL_VAL;
    return true;
}

bool nGpuMlpSgdEpoch(int argc, Value *args, Value *out) {
    (void)argc;
    GpuBuffer *x, *w1, *b1, *w2, *b2, *labels, *lossAcc, *correctAcc;
    if (!requireBuffer(args[0], 1, "gpu_mlp_sgd_epoch", &x)) return false;
    if (!requireBuffer(args[1], 2, "gpu_mlp_sgd_epoch", &w1)) return false;
    if (!requireBuffer(args[2], 3, "gpu_mlp_sgd_epoch", &b1)) return false;
    if (!requireBuffer(args[3], 4, "gpu_mlp_sgd_epoch", &w2)) return false;
    if (!requireBuffer(args[4], 5, "gpu_mlp_sgd_epoch", &b2)) return false;
    if (!requireBuffer(args[5], 6, "gpu_mlp_sgd_epoch", &labels)) return false;
    if (!requireBuffer(args[6], 7, "gpu_mlp_sgd_epoch", &lossAcc)) return false;
    correctAcc = NULL;
    if (!IS_NULL(args[7])) {
        if (!requireBuffer(args[7], 8, "gpu_mlp_sgd_epoch", &correctAcc)) return false;
    }
    int64_t samples, batch, inputs, hidden, classes, flushEvery;
    double lr;
    if (!jaiArgInt(args[8], 9, "gpu_mlp_sgd_epoch", &samples)) return false;
    if (!jaiArgInt(args[9], 10, "gpu_mlp_sgd_epoch", &batch)) return false;
    if (!jaiArgInt(args[10], 11, "gpu_mlp_sgd_epoch", &inputs)) return false;
    if (!jaiArgInt(args[11], 12, "gpu_mlp_sgd_epoch", &hidden)) return false;
    if (!jaiArgInt(args[12], 13, "gpu_mlp_sgd_epoch", &classes)) return false;
    if (!jaiArgNumber(args[13], 14, "gpu_mlp_sgd_epoch", &lr)) return false;
    if (!jaiArgInt(args[14], 15, "gpu_mlp_sgd_epoch", &flushEvery)) return false;
    if (samples <= 0 || batch <= 0 || inputs <= 0 || hidden <= 0 || classes <= 0 ||
        flushEvery <= 0 || samples > UINT32_MAX || batch > UINT32_MAX ||
        inputs > UINT32_MAX || hidden > UINT32_MAX || classes > UINT32_MAX ||
        flushEvery > UINT32_MAX)
        return jaiThrow(vm.cValueError, "gpu_mlp_sgd_epoch(): dimensions are invalid");
    if (!(lr > 0.0) || lr > 1e6)
        return jaiThrow(vm.cValueError, "gpu_mlp_sgd_epoch(): learning rate is invalid");
    uint32_t processed = 0;
    bool ok = jaiGpuMlpSgdEpoch(
        x->buffer, (size_t)x->origin * sizeof(float),
        w1->buffer, (size_t)w1->origin * sizeof(float),
        b1->buffer, (size_t)b1->origin * sizeof(float),
        w2->buffer, (size_t)w2->origin * sizeof(float),
        b2->buffer, (size_t)b2->origin * sizeof(float),
        labels->buffer, (size_t)labels->origin * sizeof(float),
        lossAcc->buffer, (size_t)lossAcc->origin * sizeof(float),
        correctAcc != NULL ? correctAcc->buffer : NULL,
        correctAcc != NULL ? (size_t)correctAcc->origin * sizeof(float) : 0,
        (uint32_t)samples, (uint32_t)batch, (uint32_t)inputs, (uint32_t)hidden,
        (uint32_t)classes, (float)lr, (uint32_t)flushEvery, &processed);
    if (!ok)
        return jaiThrow(vm.cRuntimeError, "gpu_mlp_sgd_epoch(): the kernel failed");
    *out = INT_VAL((int64_t)processed);
    return true;
}

bool nGpuMlpBwdStep(int argc, Value *args, Value *out) {
    (void)argc;
    GpuBuffer *x, *w1, *b1, *w2, *b2, *labels, *gW1, *gB1, *gW2, *gB2, *lossAcc, *correctAcc;
    if (!requireBuffer(args[0], 1, "gpu_mlp_bwd_step", &x)) return false;
    if (!requireBuffer(args[1], 2, "gpu_mlp_bwd_step", &w1)) return false;
    if (!requireBuffer(args[2], 3, "gpu_mlp_bwd_step", &b1)) return false;
    if (!requireBuffer(args[3], 4, "gpu_mlp_bwd_step", &w2)) return false;
    if (!requireBuffer(args[4], 5, "gpu_mlp_bwd_step", &b2)) return false;
    if (!requireBuffer(args[5], 6, "gpu_mlp_bwd_step", &labels)) return false;
    if (!requireBuffer(args[6], 7, "gpu_mlp_bwd_step", &gW1)) return false;
    if (!requireBuffer(args[7], 8, "gpu_mlp_bwd_step", &gB1)) return false;
    if (!requireBuffer(args[8], 9, "gpu_mlp_bwd_step", &gW2)) return false;
    if (!requireBuffer(args[9], 10, "gpu_mlp_bwd_step", &gB2)) return false;
    if (!requireBuffer(args[10], 11, "gpu_mlp_bwd_step", &lossAcc)) return false;
    correctAcc = NULL;
    if (!IS_NULL(args[11])) {
        if (!requireBuffer(args[11], 12, "gpu_mlp_bwd_step", &correctAcc)) return false;
    }

    int64_t batch, inputs, hidden, classes;
    if (!jaiArgInt(args[12], 13, "gpu_mlp_bwd_step", &batch)) return false;
    if (!jaiArgInt(args[13], 14, "gpu_mlp_bwd_step", &inputs)) return false;
    if (!jaiArgInt(args[14], 15, "gpu_mlp_bwd_step", &hidden)) return false;
    if (!jaiArgInt(args[15], 16, "gpu_mlp_bwd_step", &classes)) return false;
    if (batch <= 0 || inputs <= 0 || hidden <= 0 || classes <= 0 ||
        batch > UINT32_MAX || inputs > UINT32_MAX || hidden > UINT32_MAX ||
        classes > UINT32_MAX)
        return jaiThrow(vm.cValueError,
                        "gpu_mlp_bwd_step(): dimensions must be positive uint32, "
                        "got %lldx%lldx%lldx%lld",
                        (long long)batch, (long long)inputs, (long long)hidden,
                        (long long)classes);

    bool ok = jaiGpuMlpBwdStep(
        x->buffer, (size_t)x->origin * sizeof(float),
        w1->buffer, (size_t)w1->origin * sizeof(float),
        b1->buffer, (size_t)b1->origin * sizeof(float),
        w2->buffer, (size_t)w2->origin * sizeof(float),
        b2->buffer, (size_t)b2->origin * sizeof(float),
        labels->buffer, (size_t)labels->origin * sizeof(float),
        gW1->buffer, (size_t)gW1->origin * sizeof(float),
        gB1->buffer, (size_t)gB1->origin * sizeof(float),
        gW2->buffer, (size_t)gW2->origin * sizeof(float),
        gB2->buffer, (size_t)gB2->origin * sizeof(float),
        lossAcc->buffer, (size_t)lossAcc->origin * sizeof(float),
        correctAcc != NULL ? correctAcc->buffer : NULL,
        correctAcc != NULL ? (size_t)correctAcc->origin * sizeof(float) : 0,
        (uint32_t)batch, (uint32_t)inputs, (uint32_t)hidden, (uint32_t)classes);
    if (!ok)
        return jaiThrow(vm.cRuntimeError, "gpu_mlp_bwd_step(): the kernel failed");
    *out = NULL_VAL;
    return true;
}

bool nGpuMlp3SgdStep(int argc, Value *args, Value *out) {
    (void)argc;
    GpuBuffer *x, *w1, *b1, *w2, *b2, *w3, *b3, *w4, *b4, *labels, *lossAcc, *correctAcc;
    if (!requireBuffer(args[0], 1, "gpu_mlp3_sgd_step", &x)) return false;
    if (!requireBuffer(args[1], 2, "gpu_mlp3_sgd_step", &w1)) return false;
    if (!requireBuffer(args[2], 3, "gpu_mlp3_sgd_step", &b1)) return false;
    if (!requireBuffer(args[3], 4, "gpu_mlp3_sgd_step", &w2)) return false;
    if (!requireBuffer(args[4], 5, "gpu_mlp3_sgd_step", &b2)) return false;
    if (!requireBuffer(args[5], 6, "gpu_mlp3_sgd_step", &w3)) return false;
    if (!requireBuffer(args[6], 7, "gpu_mlp3_sgd_step", &b3)) return false;
    if (!requireBuffer(args[7], 8, "gpu_mlp3_sgd_step", &w4)) return false;
    if (!requireBuffer(args[8], 9, "gpu_mlp3_sgd_step", &b4)) return false;
    if (!requireBuffer(args[9], 10, "gpu_mlp3_sgd_step", &labels)) return false;
    if (!requireBuffer(args[10], 11, "gpu_mlp3_sgd_step", &lossAcc)) return false;
    correctAcc = NULL;
    if (!IS_NULL(args[11])) {
        if (!requireBuffer(args[11], 12, "gpu_mlp3_sgd_step", &correctAcc)) return false;
    }
    int64_t batch, inputs, hidden1, hidden2, hidden3, classes;
    double lr;
    if (!jaiArgInt(args[12], 13, "gpu_mlp3_sgd_step", &batch)) return false;
    if (!jaiArgInt(args[13], 14, "gpu_mlp3_sgd_step", &inputs)) return false;
    if (!jaiArgInt(args[14], 15, "gpu_mlp3_sgd_step", &hidden1)) return false;
    if (!jaiArgInt(args[15], 16, "gpu_mlp3_sgd_step", &hidden2)) return false;
    if (!jaiArgInt(args[16], 17, "gpu_mlp3_sgd_step", &hidden3)) return false;
    if (!jaiArgInt(args[17], 18, "gpu_mlp3_sgd_step", &classes)) return false;
    if (!jaiArgNumber(args[18], 19, "gpu_mlp3_sgd_step", &lr)) return false;
    if (batch <= 0 || inputs <= 0 || hidden1 <= 0 || hidden2 <= 0 || hidden3 <= 0 ||
        classes <= 0 || batch > UINT32_MAX || inputs > UINT32_MAX ||
        hidden1 > UINT32_MAX || hidden2 > UINT32_MAX || hidden3 > UINT32_MAX ||
        classes > UINT32_MAX)
        return jaiThrow(vm.cValueError,
                        "gpu_mlp3_sgd_step(): dimensions must be positive uint32");
    if (!(lr > 0.0) || lr > 1e6)
        return jaiThrow(vm.cValueError, "gpu_mlp3_sgd_step(): learning rate is invalid");
    bool ok = jaiGpuMlp3SgdStep(
        x->buffer, (size_t)x->origin * sizeof(float),
        w1->buffer, (size_t)w1->origin * sizeof(float),
        b1->buffer, (size_t)b1->origin * sizeof(float),
        w2->buffer, (size_t)w2->origin * sizeof(float),
        b2->buffer, (size_t)b2->origin * sizeof(float),
        w3->buffer, (size_t)w3->origin * sizeof(float),
        b3->buffer, (size_t)b3->origin * sizeof(float),
        w4->buffer, (size_t)w4->origin * sizeof(float),
        b4->buffer, (size_t)b4->origin * sizeof(float),
        labels->buffer, (size_t)labels->origin * sizeof(float),
        lossAcc->buffer, (size_t)lossAcc->origin * sizeof(float),
        correctAcc != NULL ? correctAcc->buffer : NULL,
        correctAcc != NULL ? (size_t)correctAcc->origin * sizeof(float) : 0,
        (uint32_t)batch, (uint32_t)inputs, (uint32_t)hidden1, (uint32_t)hidden2,
        (uint32_t)hidden3, (uint32_t)classes, (float)lr);
    if (!ok)
        return jaiThrow(vm.cRuntimeError, "gpu_mlp3_sgd_step(): the kernel failed");
    *out = NULL_VAL;
    return true;
}

bool nGpuMlp3SgdEpoch(int argc, Value *args, Value *out) {
    (void)argc;
    GpuBuffer *x, *w1, *b1, *w2, *b2, *w3, *b3, *w4, *b4, *labels, *lossAcc, *correctAcc;
    if (!requireBuffer(args[0], 1, "gpu_mlp3_sgd_epoch", &x)) return false;
    if (!requireBuffer(args[1], 2, "gpu_mlp3_sgd_epoch", &w1)) return false;
    if (!requireBuffer(args[2], 3, "gpu_mlp3_sgd_epoch", &b1)) return false;
    if (!requireBuffer(args[3], 4, "gpu_mlp3_sgd_epoch", &w2)) return false;
    if (!requireBuffer(args[4], 5, "gpu_mlp3_sgd_epoch", &b2)) return false;
    if (!requireBuffer(args[5], 6, "gpu_mlp3_sgd_epoch", &w3)) return false;
    if (!requireBuffer(args[6], 7, "gpu_mlp3_sgd_epoch", &b3)) return false;
    if (!requireBuffer(args[7], 8, "gpu_mlp3_sgd_epoch", &w4)) return false;
    if (!requireBuffer(args[8], 9, "gpu_mlp3_sgd_epoch", &b4)) return false;
    if (!requireBuffer(args[9], 10, "gpu_mlp3_sgd_epoch", &labels)) return false;
    if (!requireBuffer(args[10], 11, "gpu_mlp3_sgd_epoch", &lossAcc)) return false;
    correctAcc = NULL;
    if (!IS_NULL(args[11])) {
        if (!requireBuffer(args[11], 12, "gpu_mlp3_sgd_epoch", &correctAcc)) return false;
    }
    int64_t samples, batch, inputs, hidden1, hidden2, hidden3, classes, flushEvery;
    double lr;
    if (!jaiArgInt(args[12], 13, "gpu_mlp3_sgd_epoch", &samples)) return false;
    if (!jaiArgInt(args[13], 14, "gpu_mlp3_sgd_epoch", &batch)) return false;
    if (!jaiArgInt(args[14], 15, "gpu_mlp3_sgd_epoch", &inputs)) return false;
    if (!jaiArgInt(args[15], 16, "gpu_mlp3_sgd_epoch", &hidden1)) return false;
    if (!jaiArgInt(args[16], 17, "gpu_mlp3_sgd_epoch", &hidden2)) return false;
    if (!jaiArgInt(args[17], 18, "gpu_mlp3_sgd_epoch", &hidden3)) return false;
    if (!jaiArgInt(args[18], 19, "gpu_mlp3_sgd_epoch", &classes)) return false;
    if (!jaiArgNumber(args[19], 20, "gpu_mlp3_sgd_epoch", &lr)) return false;
    if (!jaiArgInt(args[20], 21, "gpu_mlp3_sgd_epoch", &flushEvery)) return false;
    if (samples <= 0 || batch <= 0 || inputs <= 0 || hidden1 <= 0 || hidden2 <= 0 ||
        hidden3 <= 0 || classes <= 0 || flushEvery <= 0 || samples > UINT32_MAX ||
        batch > UINT32_MAX || inputs > UINT32_MAX || hidden1 > UINT32_MAX ||
        hidden2 > UINT32_MAX || hidden3 > UINT32_MAX || classes > UINT32_MAX ||
        flushEvery > UINT32_MAX)
        return jaiThrow(vm.cValueError, "gpu_mlp3_sgd_epoch(): dimensions are invalid");
    if (!(lr > 0.0) || lr > 1e6)
        return jaiThrow(vm.cValueError, "gpu_mlp3_sgd_epoch(): learning rate is invalid");
    uint32_t processed = 0;
    bool ok = jaiGpuMlp3SgdEpoch(
        x->buffer, (size_t)x->origin * sizeof(float),
        w1->buffer, (size_t)w1->origin * sizeof(float),
        b1->buffer, (size_t)b1->origin * sizeof(float),
        w2->buffer, (size_t)w2->origin * sizeof(float),
        b2->buffer, (size_t)b2->origin * sizeof(float),
        w3->buffer, (size_t)w3->origin * sizeof(float),
        b3->buffer, (size_t)b3->origin * sizeof(float),
        w4->buffer, (size_t)w4->origin * sizeof(float),
        b4->buffer, (size_t)b4->origin * sizeof(float),
        labels->buffer, (size_t)labels->origin * sizeof(float),
        lossAcc->buffer, (size_t)lossAcc->origin * sizeof(float),
        correctAcc != NULL ? correctAcc->buffer : NULL,
        correctAcc != NULL ? (size_t)correctAcc->origin * sizeof(float) : 0,
        (uint32_t)samples, (uint32_t)batch, (uint32_t)inputs, (uint32_t)hidden1,
        (uint32_t)hidden2, (uint32_t)hidden3, (uint32_t)classes, (float)lr,
        (uint32_t)flushEvery, &processed);
    if (!ok)
        return jaiThrow(vm.cRuntimeError, "gpu_mlp3_sgd_epoch(): the kernel failed");
    *out = INT_VAL((int64_t)processed);
    return true;
}

bool nGpuMlp3BwdStep(int argc, Value *args, Value *out) {
    (void)argc;
    GpuBuffer *x, *w1, *b1, *w2, *b2, *w3, *b3, *w4, *b4, *labels;
    GpuBuffer *gW1, *gB1, *gW2, *gB2, *gW3, *gB3, *gW4, *gB4, *lossAcc, *correctAcc;
    if (!requireBuffer(args[0], 1, "gpu_mlp3_bwd_step", &x)) return false;
    if (!requireBuffer(args[1], 2, "gpu_mlp3_bwd_step", &w1)) return false;
    if (!requireBuffer(args[2], 3, "gpu_mlp3_bwd_step", &b1)) return false;
    if (!requireBuffer(args[3], 4, "gpu_mlp3_bwd_step", &w2)) return false;
    if (!requireBuffer(args[4], 5, "gpu_mlp3_bwd_step", &b2)) return false;
    if (!requireBuffer(args[5], 6, "gpu_mlp3_bwd_step", &w3)) return false;
    if (!requireBuffer(args[6], 7, "gpu_mlp3_bwd_step", &b3)) return false;
    if (!requireBuffer(args[7], 8, "gpu_mlp3_bwd_step", &w4)) return false;
    if (!requireBuffer(args[8], 9, "gpu_mlp3_bwd_step", &b4)) return false;
    if (!requireBuffer(args[9], 10, "gpu_mlp3_bwd_step", &labels)) return false;
    if (!requireBuffer(args[10], 11, "gpu_mlp3_bwd_step", &gW1)) return false;
    if (!requireBuffer(args[11], 12, "gpu_mlp3_bwd_step", &gB1)) return false;
    if (!requireBuffer(args[12], 13, "gpu_mlp3_bwd_step", &gW2)) return false;
    if (!requireBuffer(args[13], 14, "gpu_mlp3_bwd_step", &gB2)) return false;
    if (!requireBuffer(args[14], 15, "gpu_mlp3_bwd_step", &gW3)) return false;
    if (!requireBuffer(args[15], 16, "gpu_mlp3_bwd_step", &gB3)) return false;
    if (!requireBuffer(args[16], 17, "gpu_mlp3_bwd_step", &gW4)) return false;
    if (!requireBuffer(args[17], 18, "gpu_mlp3_bwd_step", &gB4)) return false;
    if (!requireBuffer(args[18], 19, "gpu_mlp3_bwd_step", &lossAcc)) return false;
    correctAcc = NULL;
    if (!IS_NULL(args[19])) {
        if (!requireBuffer(args[19], 20, "gpu_mlp3_bwd_step", &correctAcc)) return false;
    }
    int64_t batch, inputs, hidden1, hidden2, hidden3, classes;
    if (!jaiArgInt(args[20], 21, "gpu_mlp3_bwd_step", &batch)) return false;
    if (!jaiArgInt(args[21], 22, "gpu_mlp3_bwd_step", &inputs)) return false;
    if (!jaiArgInt(args[22], 23, "gpu_mlp3_bwd_step", &hidden1)) return false;
    if (!jaiArgInt(args[23], 24, "gpu_mlp3_bwd_step", &hidden2)) return false;
    if (!jaiArgInt(args[24], 25, "gpu_mlp3_bwd_step", &hidden3)) return false;
    if (!jaiArgInt(args[25], 26, "gpu_mlp3_bwd_step", &classes)) return false;
    if (batch <= 0 || inputs <= 0 || hidden1 <= 0 || hidden2 <= 0 || hidden3 <= 0 ||
        classes <= 0 || batch > UINT32_MAX || inputs > UINT32_MAX ||
        hidden1 > UINT32_MAX || hidden2 > UINT32_MAX || hidden3 > UINT32_MAX ||
        classes > UINT32_MAX)
        return jaiThrow(vm.cValueError,
                        "gpu_mlp3_bwd_step(): dimensions must be positive uint32");
    bool ok = jaiGpuMlp3BwdStep(
        x->buffer, (size_t)x->origin * sizeof(float),
        w1->buffer, (size_t)w1->origin * sizeof(float),
        b1->buffer, (size_t)b1->origin * sizeof(float),
        w2->buffer, (size_t)w2->origin * sizeof(float),
        b2->buffer, (size_t)b2->origin * sizeof(float),
        w3->buffer, (size_t)w3->origin * sizeof(float),
        b3->buffer, (size_t)b3->origin * sizeof(float),
        w4->buffer, (size_t)w4->origin * sizeof(float),
        b4->buffer, (size_t)b4->origin * sizeof(float),
        labels->buffer, (size_t)labels->origin * sizeof(float),
        gW1->buffer, (size_t)gW1->origin * sizeof(float),
        gB1->buffer, (size_t)gB1->origin * sizeof(float),
        gW2->buffer, (size_t)gW2->origin * sizeof(float),
        gB2->buffer, (size_t)gB2->origin * sizeof(float),
        gW3->buffer, (size_t)gW3->origin * sizeof(float),
        gB3->buffer, (size_t)gB3->origin * sizeof(float),
        gW4->buffer, (size_t)gW4->origin * sizeof(float),
        gB4->buffer, (size_t)gB4->origin * sizeof(float),
        lossAcc->buffer, (size_t)lossAcc->origin * sizeof(float),
        correctAcc != NULL ? correctAcc->buffer : NULL,
        correctAcc != NULL ? (size_t)correctAcc->origin * sizeof(float) : 0,
        (uint32_t)batch, (uint32_t)inputs, (uint32_t)hidden1, (uint32_t)hidden2,
        (uint32_t)hidden3, (uint32_t)classes);
    if (!ok)
        return jaiThrow(vm.cRuntimeError, "gpu_mlp3_bwd_step(): the kernel failed");
    *out = NULL_VAL;
    return true;
}

bool nGpuLabelsValid(int argc, Value *args, Value *out) {
    (void)argc;
    GpuBuffer *labels;
    if (!requireBuffer(args[0], 1, "gpu_labels_valid", &labels)) return false;
    int64_t count, classes;
    if (!jaiArgInt(args[1], 2, "gpu_labels_valid", &count)) return false;
    if (!jaiArgInt(args[2], 3, "gpu_labels_valid", &classes)) return false;
    if (count < 0 || classes <= 0 || count > UINT32_MAX || classes > UINT32_MAX)
        return jaiThrow(vm.cValueError,
                        "gpu_labels_valid(): count and classes must fit in uint32");
    *out = BOOL_VAL(jaiGpuLabelsValid(
        labels->buffer, (size_t)labels->origin * sizeof(float),
        (uint32_t)count, (uint32_t)classes));
    return true;
}
