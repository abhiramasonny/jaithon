#: Metal kernels used by dense classification models.
import std.gpu as gpu
from std.gpu import Kernel

let SOURCE = """#include <metal_stdlib>
using namespace metal;

kernel void dense_forward(device const float* input [[buffer(0)]],
                          device const float* weights [[buffer(1)]],
                          device const float* bias [[buffer(2)]],
                          device float* output [[buffer(3)]],
                          constant uint& batch [[buffer(4)]],
                          constant uint& inputs [[buffer(5)]],
                          constant uint& outputs [[buffer(6)]],
                          constant uint& input_offset [[buffer(7)]],
                          constant uint& activation [[buffer(8)]],
                          constant uint& output_offset [[buffer(9)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= batch * outputs) return;
    uint sample = id / outputs;
    uint unit = id % outputs;
    uint input_base = (sample + input_offset) * inputs;
    float sum = bias[unit];
    for (uint feature = 0; feature < inputs; ++feature)
        sum += input[input_base + feature] * weights[feature * outputs + unit];
    if (activation == 1) sum = max(sum, 0.0f);
    else if (activation == 2) sum = 1.0f / (1.0f + exp(-sum));
    else if (activation == 3) sum = tanh(sum);
    output[(sample + output_offset) * outputs + unit] = sum;
}

kernel void softmax_loss(device const float* logits [[buffer(0)]],
                         device const float* labels [[buffer(1)]],
                         device float* delta [[buffer(2)]],
                         device float* metrics [[buffer(3)]],
                         constant uint& batch [[buffer(4)]],
                         constant uint& classes [[buffer(5)]],
                         constant uint& label_offset [[buffer(6)]],
                         uint sample [[thread_position_in_grid]]) {
    if (sample >= batch) return;
    uint base = sample * classes;
    float peak = logits[base];
    uint prediction = 0;
    for (uint digit = 1; digit < classes; ++digit) {
        float value = logits[base + digit];
        if (value > peak) { peak = value; prediction = digit; }
    }
    float total = 0.0f;
    for (uint digit = 0; digit < classes; ++digit)
        total += exp(logits[base + digit] - peak);
    uint answer = uint(labels[label_offset + sample]);
    for (uint digit = 0; digit < classes; ++digit) {
        float probability = exp(logits[base + digit] - peak) / total;
        delta[base + digit] = probability - (digit == answer ? 1.0f : 0.0f);
    }
    uint metric = (label_offset + sample) * 2;
    metrics[metric] = peak + log(total) - logits[base + answer];
    metrics[metric + 1] = prediction == answer ? 1.0f : 0.0f;
}

kernel void dense_backward(device const float* next_delta [[buffer(0)]],
                           device const float* weights [[buffer(1)]],
                           device const float* activation_value [[buffer(2)]],
                           device float* delta [[buffer(3)]],
                           constant uint& batch [[buffer(4)]],
                           constant uint& inputs [[buffer(5)]],
                           constant uint& outputs [[buffer(6)]],
                           constant uint& activation [[buffer(7)]],
                           uint id [[thread_position_in_grid]]) {
    if (id >= batch * inputs) return;
    uint sample = id / inputs;
    uint feature = id % inputs;
    float sum = 0.0f;
    for (uint unit = 0; unit < outputs; ++unit)
        sum += next_delta[sample * outputs + unit] * weights[feature * outputs + unit];
    float value = activation_value[id];
    if (activation == 1 && value <= 0.0f) sum = 0.0f;
    else if (activation == 2) sum *= value * (1.0f - value);
    else if (activation == 3) sum *= 1.0f - value * value;
    delta[id] = sum;
}

kernel void optimizer_update(device const float* input [[buffer(0)]],
                             device const float* delta [[buffer(1)]],
                             device float* weights [[buffer(2)]],
                             device float* bias [[buffer(3)]],
                             device float* weight_first [[buffer(4)]],
                             device float* weight_second [[buffer(5)]],
                             device float* bias_first [[buffer(6)]],
                             device float* bias_second [[buffer(7)]],
                             device const float* settings [[buffer(8)]],
                             constant uint& batch [[buffer(9)]],
                             constant uint& inputs [[buffer(10)]],
                             constant uint& outputs [[buffer(11)]],
                             constant uint& input_offset [[buffer(12)]],
                             constant uint& optimizer [[buffer(13)]],
                             constant uint& step [[buffer(14)]],
                             uint id [[thread_position_in_grid]]) {
    uint weight_count = inputs * outputs;
    if (id < weight_count) {
        uint feature = id / outputs;
        uint unit = id % outputs;
        float gradient = 0.0f;
        for (uint sample = 0; sample < batch; ++sample)
            gradient += input[(sample + input_offset) * inputs + feature]
                      * delta[sample * outputs + unit];
        gradient /= float(batch);
        if (optimizer == 0) {
            float velocity = settings[1] * weight_first[id] + gradient;
            weight_first[id] = velocity;
            float update = settings[2] > 0.5f ? settings[1] * velocity + gradient : velocity;
            weights[id] -= settings[0] * update;
        } else {
            float first = settings[1] * weight_first[id] + (1.0f - settings[1]) * gradient;
            float second = settings[2] * weight_second[id]
                         + (1.0f - settings[2]) * gradient * gradient;
            weight_first[id] = first;
            weight_second[id] = second;
            float corrected_first = first / (1.0f - pow(settings[1], float(step)));
            float corrected_second = second / (1.0f - pow(settings[2], float(step)));
            weights[id] -= settings[0] * corrected_first
                         / (sqrt(corrected_second) + settings[3]);
        }
    }
    if (id < outputs) {
        float gradient = 0.0f;
        for (uint sample = 0; sample < batch; ++sample)
            gradient += delta[sample * outputs + id];
        gradient /= float(batch);
        if (optimizer == 0) {
            float velocity = settings[1] * bias_first[id] + gradient;
            bias_first[id] = velocity;
            float update = settings[2] > 0.5f ? settings[1] * velocity + gradient : velocity;
            bias[id] -= settings[0] * update;
        } else {
            float first = settings[1] * bias_first[id] + (1.0f - settings[1]) * gradient;
            float second = settings[2] * bias_second[id]
                         + (1.0f - settings[2]) * gradient * gradient;
            bias_first[id] = first;
            bias_second[id] = second;
            float corrected_first = first / (1.0f - pow(settings[1], float(step)));
            float corrected_second = second / (1.0f - pow(settings[2], float(step)));
            bias[id] -= settings[0] * corrected_first
                      / (sqrt(corrected_second) + settings[3]);
        }
    }
}

kernel void row_softmax(device const float* logits [[buffer(0)]],
                        device float* probabilities [[buffer(1)]],
                        constant uint& rows [[buffer(2)]],
                        constant uint& classes [[buffer(3)]],
                        uint row [[thread_position_in_grid]]) {
    if (row >= rows) return;
    uint base = row * classes;
    float peak = logits[base];
    for (uint column = 1; column < classes; ++column)
        peak = max(peak, logits[base + column]);
    float total = 0.0f;
    for (uint column = 0; column < classes; ++column)
        total += exp(logits[base + column] - peak);
    for (uint column = 0; column < classes; ++column)
        probabilities[base + column] = exp(logits[base + column] - peak) / total;
}
"""

#: Compiled pipelines shared by a `Sequential` model.
pub class Kernels {
    #: Dense affine transform and activation.
    pub let forward: Kernel
    #: Sparse softmax cross-entropy and output gradient.
    pub let loss: Kernel
    #: Gradient propagation into the preceding layer.
    pub let backward: Kernel
    #: Dense weight and bias update for SGD or Adam.
    pub let update: Kernel
    #: Row-wise softmax used by prediction.
    pub let softmax: Kernel
    var freed: bool

    #: Compile the training and inference pipelines from the shared Metal source.
    pub fn init(self) {
        self.forward = gpu.Kernel.compile(SOURCE, "dense_forward")
        self.loss = gpu.Kernel.compile(SOURCE, "softmax_loss")
        self.backward = gpu.Kernel.compile(SOURCE, "dense_backward")
        self.update = gpu.Kernel.compile(SOURCE, "optimizer_update")
        self.softmax = gpu.Kernel.compile(SOURCE, "row_softmax")
        self.freed = false
    }

    #: Release each compiled pipeline after queued work finishes.
    pub fn free(self) -> void {
        if self.freed { return }
        gpu.synchronize()
        self.forward.free()
        self.loss.free()
        self.backward.free()
        self.update.free()
        self.softmax.free()
        self.freed = true
    }
}
