#: GPU tensor operations that keep their inputs and results on the device.
import std.gpu as gpu
from std.gpu import Buffer, Kernel
from .tensor import GradNode, Tensor, element_count

let SOURCE = """#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void binary_op(device const float* a [[buffer(0)]],
                      device const float* b [[buffer(1)]],
                      device float* output [[buffer(2)]],
                      constant uint& count [[buffer(3)]],
                      constant uint& operation [[buffer(4)]],
                      uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    if (operation == 0) output[id] = a[id] + b[id];
    else if (operation == 1) output[id] = a[id] - b[id];
    else if (operation == 2) output[id] = a[id] * b[id];
    else output[id] = a[id] / b[id];
}

kernel void unary_op(device const float* input [[buffer(0)]],
                     device float* output [[buffer(1)]],
                     constant uint& count [[buffer(2)]],
                     constant uint& operation [[buffer(3)]],
                     uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    float value = input[id];
    if (operation == 0) output[id] = value;
    else if (operation == 1) output[id] = max(value, 0.0f);
    else if (operation == 2) output[id] = 1.0f / (1.0f + exp(-value));
    else if (operation == 3) output[id] = tanh(value);
    else if (operation == 4) output[id] = exp(value);
    else output[id] = log(value);
}

kernel void scalar_mul(device const float* input [[buffer(0)]],
                       device const float* scalar [[buffer(1)]],
                       device float* output [[buffer(2)]],
                       constant uint& count [[buffer(3)]],
                       uint id [[thread_position_in_grid]]) {
    if (id < count) output[id] = input[id] * scalar[0];
}

kernel void matrix_mul(device const float* a [[buffer(0)]],
                       device const float* b [[buffer(1)]],
                       device float* output [[buffer(2)]],
                       constant uint& rows [[buffer(3)]],
                       constant uint& inner [[buffer(4)]],
                       constant uint& columns [[buffer(5)]],
                       uint id [[thread_position_in_grid]]) {
    if (id >= rows * columns) return;
    uint row = id / columns;
    uint column = id % columns;
    float total = 0.0f;
    for (uint k = 0; k < inner; ++k)
        total += a[row * inner + k] * b[k * columns + column];
    output[id] = total;
}

kernel void matrix_transpose(device const float* input [[buffer(0)]],
                             device float* output [[buffer(1)]],
                             constant uint& rows [[buffer(2)]],
                             constant uint& columns [[buffer(3)]],
                             uint id [[thread_position_in_grid]]) {
    if (id >= rows * columns) return;
    uint row = id / columns;
    uint column = id % columns;
    output[column * rows + row] = input[id];
}

kernel void row_softmax(device const float* input [[buffer(0)]],
                        device float* output [[buffer(1)]],
                        constant uint& rows [[buffer(2)]],
                        constant uint& columns [[buffer(3)]],
                        uint row [[thread_position_in_grid]]) {
    if (row >= rows) return;
    uint base = row * columns;
    float peak = input[base];
    for (uint column = 1; column < columns; ++column)
        peak = max(peak, input[base + column]);
    float total = 0.0f;
    for (uint column = 0; column < columns; ++column)
        total += exp(input[base + column] - peak);
    for (uint column = 0; column < columns; ++column)
        output[base + column] = exp(input[base + column] - peak) / total;
}

kernel void row_argmax(device const float* input [[buffer(0)]],
                       device float* output [[buffer(1)]],
                       constant uint& rows [[buffer(2)]],
                       constant uint& columns [[buffer(3)]],
                       uint row [[thread_position_in_grid]]) {
    if (row >= rows) return;
    uint base = row * columns;
    float peak = input[base];
    uint best = 0;
    for (uint column = 1; column < columns; ++column) {
        float value = input[base + column];
        if (value > peak) { peak = value; best = column; }
    }
    output[row] = float(best);
}

kernel void one_hot_rows(device const float* indices [[buffer(0)]],
                         device float* output [[buffer(1)]],
                         constant uint& rows [[buffer(2)]],
                         constant uint& depth [[buffer(3)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= rows * depth) return;
    uint row = id / depth;
    uint column = id % depth;
    output[id] = int(indices[row]) == int(column) ? 1.0f : 0.0f;
}

kernel void broadcast_add(device const float* a [[buffer(0)]],
                          device const float* b [[buffer(1)]],
                          device float* output [[buffer(2)]],
                          constant uint& inner [[buffer(3)]],
                          constant uint& total [[buffer(4)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= total) return;
    output[id] = a[id] + b[id % inner];
}

kernel void binary_backward_op(device const float* a [[buffer(0)]],
                               device const float* b [[buffer(1)]],
                               device const float* upstream [[buffer(2)]],
                               device float* grad_a [[buffer(3)]],
                               device float* grad_b [[buffer(4)]],
                               constant uint& count [[buffer(5)]],
                               constant uint& operation [[buffer(6)]],
                               uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    float u = upstream[id];
    if (operation == 0) { grad_a[id] = u; grad_b[id] = u; }
    else if (operation == 1) { grad_a[id] = u; grad_b[id] = -u; }
    else if (operation == 2) { grad_a[id] = u * b[id]; grad_b[id] = u * a[id]; }
    else {
        float bv = b[id];
        grad_a[id] = u / bv;
        grad_b[id] = -u * a[id] / (bv * bv);
    }
}

kernel void reduce_leading(device const float* upstream [[buffer(0)]],
                           device float* grad_small [[buffer(1)]],
                           constant uint& outer [[buffer(2)]],
                           constant uint& inner [[buffer(3)]],
                           uint id [[thread_position_in_grid]]) {
    if (id >= inner) return;
    float total = 0.0f;
    for (uint row = 0; row < outer; ++row) total += upstream[row * inner + id];
    grad_small[id] = total;
}

kernel void reduce_leading4(device const float* upstream [[buffer(0)]],
                            device float* grad_small [[buffer(1)]],
                            constant uint& outer [[buffer(2)]],
                            constant uint& inner [[buffer(3)]],
                            uint id [[thread_position_in_grid]]) {
    uint col = id * 4;
    if (col >= inner) return;
    float4 total = float4(0.0f);
    for (uint row = 0; row < outer; ++row)
        total += *((device const float4*)(upstream + row * inner + col));
    *((device float4*)(grad_small + col)) = total;
}

kernel void unary_backward_op(device const float* saved [[buffer(0)]],
                              device const float* upstream [[buffer(1)]],
                              device float* grad_input [[buffer(2)]],
                              constant uint& count [[buffer(3)]],
                              constant uint& operation [[buffer(4)]],
                              uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    float s = saved[id];
    float u = upstream[id];
    if (operation == 0) grad_input[id] = u;
    else if (operation == 1) grad_input[id] = s > 0.0f ? u : 0.0f;
    else if (operation == 2) grad_input[id] = u * s * (1.0f - s);
    else if (operation == 3) grad_input[id] = u * (1.0f - s * s);
    else if (operation == 4) grad_input[id] = u * s;
    else grad_input[id] = u / s;
}

kernel void unary_backward_relu4(device const float* saved [[buffer(0)]],
                                 device const float* upstream [[buffer(1)]],
                                 device float* grad_input [[buffer(2)]],
                                 constant uint& count [[buffer(3)]],
                                 uint id [[thread_position_in_grid]]) {
    uint i = id * 4;
    if (i >= count) return;
    float4 s = *((device const float4*)(saved + i));
    float4 u = *((device const float4*)(upstream + i));
    *((device float4*)(grad_input + i)) = select(float4(0.0f), u, s > 0.0f);
}

kernel void softmax_backward(device const float* output [[buffer(0)]],
                             device const float* upstream [[buffer(1)]],
                             device float* grad_input [[buffer(2)]],
                             constant uint& rows [[buffer(3)]],
                             constant uint& columns [[buffer(4)]],
                             uint row [[thread_position_in_grid]]) {
    if (row >= rows) return;
    uint base = row * columns;
    float dot = 0.0f;
    for (uint column = 0; column < columns; ++column) dot += upstream[base + column] * output[base + column];
    for (uint column = 0; column < columns; ++column)
        grad_input[base + column] = output[base + column] * (upstream[base + column] - dot);
}

kernel void copy_offset(device const float* source [[buffer(0)]],
                        device float* output [[buffer(1)]],
                        constant uint& source_offset [[buffer(2)]],
                        constant uint& count [[buffer(3)]],
                        uint id [[thread_position_in_grid]]) {
    if (id < count) output[id] = source[source_offset + id];
}

constexpr constant uint MATMUL_TILE = 32;
constexpr constant uint MATMUL_GROUP = 128;

kernel void matrix_mul_mma(device const float* a [[buffer(0)]],
                           device const float* b [[buffer(1)]],
                           device float* output [[buffer(2)]],
                           device const float* bias [[buffer(3)]],
                           constant uint& rows [[buffer(4)]],
                           constant uint& inner [[buffer(5)]],
                           constant uint& columns [[buffer(6)]],
                           constant uint& trans_a [[buffer(7)]],
                           constant uint& trans_b [[buffer(8)]],
                           constant uint& epilogue [[buffer(9)]],
                           uint wid [[threadgroup_position_in_grid]],
                           uint sgitg [[simdgroup_index_in_threadgroup]]) {
    uint groups_x = columns / MATMUL_TILE;
    uint gx = wid % groups_x;
    uint gy = wid / groups_x;
    uint sg_m = sgitg / 2;
    uint sg_n = sgitg % 2;
    uint row0 = gy * MATMUL_TILE + sg_m * 16;
    uint col0 = gx * MATMUL_TILE + sg_n * 16;
    simdgroup_float8x8 acc00 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc01 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc10 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc11 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k = 0; k < inner; k += 8) {
        simdgroup_float8x8 matA0;
        simdgroup_float8x8 matA1;
        simdgroup_float8x8 matB0;
        simdgroup_float8x8 matB1;
        if (trans_a) {
            simdgroup_load(matA0, a + k * rows + row0, rows, ulong2(0, 0), true);
            simdgroup_load(matA1, a + k * rows + row0 + 8, rows, ulong2(0, 0), true);
        } else {
            simdgroup_load(matA0, a + row0 * inner + k, inner);
            simdgroup_load(matA1, a + (row0 + 8) * inner + k, inner);
        }
        if (trans_b) {
            simdgroup_load(matB0, b + col0 * inner + k, inner, ulong2(0, 0), true);
            simdgroup_load(matB1, b + (col0 + 8) * inner + k, inner, ulong2(0, 0), true);
        } else {
            simdgroup_load(matB0, b + k * columns + col0, columns);
            simdgroup_load(matB1, b + k * columns + col0 + 8, columns);
        }
        simdgroup_multiply_accumulate(acc00, matA0, matB0, acc00);
        simdgroup_multiply_accumulate(acc01, matA0, matB1, acc01);
        simdgroup_multiply_accumulate(acc10, matA1, matB0, acc10);
        simdgroup_multiply_accumulate(acc11, matA1, matB1, acc11);
    }
    (void)bias;
    (void)epilogue;
    simdgroup_store(acc00, output + row0 * columns + col0, columns);
    simdgroup_store(acc01, output + row0 * columns + col0 + 8, columns);
    simdgroup_store(acc10, output + (row0 + 8) * columns + col0, columns);
    simdgroup_store(acc11, output + (row0 + 8) * columns + col0 + 8, columns);
}

kernel void matrix_mul_mma_k(device const float* a [[buffer(0)]],
                             device const float* b [[buffer(1)]],
                             device float* output [[buffer(2)]],
                             device const float* bias [[buffer(3)]],
                             constant uint& rows [[buffer(4)]],
                             constant uint& inner [[buffer(5)]],
                             constant uint& columns [[buffer(6)]],
                             constant uint& trans_a [[buffer(7)]],
                             constant uint& trans_b [[buffer(8)]],
                             constant uint& epilogue [[buffer(9)]],
                             uint lid [[thread_index_in_threadgroup]],
                             uint wid [[threadgroup_position_in_grid]],
                             uint sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float padA[MATMUL_TILE * 8];
    threadgroup float padB[8 * MATMUL_TILE];
    uint groups_x = columns / MATMUL_TILE;
    uint gx = wid % groups_x;
    uint gy = wid / groups_x;
    uint sg_m = sgitg / 2;
    uint sg_n = sgitg % 2;
    uint row0 = gy * MATMUL_TILE + sg_m * 16;
    uint col0 = gx * MATMUL_TILE + sg_n * 16;
    simdgroup_float8x8 acc00 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc01 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc10 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc11 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    uint k = 0;
    for (; k + 8 <= inner; k += 8) {
        simdgroup_float8x8 matA0;
        simdgroup_float8x8 matA1;
        simdgroup_float8x8 matB0;
        simdgroup_float8x8 matB1;
        if (trans_a) {
            simdgroup_load(matA0, a + k * rows + row0, rows, ulong2(0, 0), true);
            simdgroup_load(matA1, a + k * rows + row0 + 8, rows, ulong2(0, 0), true);
        } else {
            simdgroup_load(matA0, a + row0 * inner + k, inner);
            simdgroup_load(matA1, a + (row0 + 8) * inner + k, inner);
        }
        if (trans_b) {
            simdgroup_load(matB0, b + col0 * inner + k, inner, ulong2(0, 0), true);
            simdgroup_load(matB1, b + (col0 + 8) * inner + k, inner, ulong2(0, 0), true);
        } else {
            simdgroup_load(matB0, b + k * columns + col0, columns);
            simdgroup_load(matB1, b + k * columns + col0 + 8, columns);
        }
        simdgroup_multiply_accumulate(acc00, matA0, matB0, acc00);
        simdgroup_multiply_accumulate(acc01, matA0, matB1, acc01);
        simdgroup_multiply_accumulate(acc10, matA1, matB0, acc10);
        simdgroup_multiply_accumulate(acc11, matA1, matB1, acc11);
    }
    if (k < inner) {
        uint rem = inner - k;
        for (uint t = 0; t < 2; ++t) {
            uint idx = lid + t * MATMUL_GROUP;
            uint ar = idx / 8;
            uint ac = idx % 8;
            float av = 0.0f;
            if (ac < rem) {
                uint am = gy * MATMUL_TILE + ar;
                uint ak = k + ac;
                av = trans_a ? a[ak * rows + am] : a[am * inner + ak];
            }
            padA[idx] = av;
            uint br = idx / MATMUL_TILE;
            uint bc = idx % MATMUL_TILE;
            float bv = 0.0f;
            if (br < rem) {
                uint bk = k + br;
                uint bn = gx * MATMUL_TILE + bc;
                bv = trans_b ? b[bn * inner + bk] : b[bk * columns + bn];
            }
            padB[idx] = bv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 matA0;
        simdgroup_float8x8 matA1;
        simdgroup_float8x8 matB0;
        simdgroup_float8x8 matB1;
        simdgroup_load(matA0, padA + (sg_m * 16) * 8, 8);
        simdgroup_load(matA1, padA + (sg_m * 16 + 8) * 8, 8);
        simdgroup_load(matB0, padB + sg_n * 16, MATMUL_TILE);
        simdgroup_load(matB1, padB + sg_n * 16 + 8, MATMUL_TILE);
        simdgroup_multiply_accumulate(acc00, matA0, matB0, acc00);
        simdgroup_multiply_accumulate(acc01, matA0, matB1, acc01);
        simdgroup_multiply_accumulate(acc10, matA1, matB0, acc10);
        simdgroup_multiply_accumulate(acc11, matA1, matB1, acc11);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    (void)bias;
    (void)epilogue;
    simdgroup_store(acc00, output + row0 * columns + col0, columns);
    simdgroup_store(acc01, output + row0 * columns + col0 + 8, columns);
    simdgroup_store(acc10, output + (row0 + 8) * columns + col0, columns);
    simdgroup_store(acc11, output + (row0 + 8) * columns + col0 + 8, columns);
}

kernel void matrix_mul_mma_bias(device const float* a [[buffer(0)]],
                                device const float* b [[buffer(1)]],
                                device float* output [[buffer(2)]],
                                device const float* bias [[buffer(3)]],
                                constant uint& rows [[buffer(4)]],
                                constant uint& inner [[buffer(5)]],
                                constant uint& columns [[buffer(6)]],
                                constant uint& trans_a [[buffer(7)]],
                                constant uint& trans_b [[buffer(8)]],
                                constant uint& epilogue [[buffer(9)]],
                                uint lid [[thread_index_in_threadgroup]],
                                uint wid [[threadgroup_position_in_grid]],
                                uint sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float tileC[MATMUL_TILE * MATMUL_TILE];
    uint groups_x = columns / MATMUL_TILE;
    uint gx = wid % groups_x;
    uint gy = wid / groups_x;
    uint sg_m = sgitg / 2;
    uint sg_n = sgitg % 2;
    uint row0 = gy * MATMUL_TILE + sg_m * 16;
    uint col0 = gx * MATMUL_TILE + sg_n * 16;
    simdgroup_float8x8 acc00 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc01 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc10 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc11 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k = 0; k < inner; k += 8) {
        simdgroup_float8x8 matA0;
        simdgroup_float8x8 matA1;
        simdgroup_float8x8 matB0;
        simdgroup_float8x8 matB1;
        if (trans_a) {
            simdgroup_load(matA0, a + k * rows + row0, rows, ulong2(0, 0), true);
            simdgroup_load(matA1, a + k * rows + row0 + 8, rows, ulong2(0, 0), true);
        } else {
            simdgroup_load(matA0, a + row0 * inner + k, inner);
            simdgroup_load(matA1, a + (row0 + 8) * inner + k, inner);
        }
        if (trans_b) {
            simdgroup_load(matB0, b + col0 * inner + k, inner, ulong2(0, 0), true);
            simdgroup_load(matB1, b + (col0 + 8) * inner + k, inner, ulong2(0, 0), true);
        } else {
            simdgroup_load(matB0, b + k * columns + col0, columns);
            simdgroup_load(matB1, b + k * columns + col0 + 8, columns);
        }
        simdgroup_multiply_accumulate(acc00, matA0, matB0, acc00);
        simdgroup_multiply_accumulate(acc01, matA0, matB1, acc01);
        simdgroup_multiply_accumulate(acc10, matA1, matB0, acc10);
        simdgroup_multiply_accumulate(acc11, matA1, matB1, acc11);
    }
    simdgroup_store(acc00, tileC + (sg_m * 16) * MATMUL_TILE + sg_n * 16, MATMUL_TILE);
    simdgroup_store(acc01, tileC + (sg_m * 16) * MATMUL_TILE + sg_n * 16 + 8, MATMUL_TILE);
    simdgroup_store(acc10, tileC + (sg_m * 16 + 8) * MATMUL_TILE + sg_n * 16, MATMUL_TILE);
    simdgroup_store(acc11, tileC + (sg_m * 16 + 8) * MATMUL_TILE + sg_n * 16 + 8, MATMUL_TILE);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = 0; t < 8; ++t) {
        uint idx = lid + t * MATMUL_GROUP;
        uint r = gy * MATMUL_TILE + idx / MATMUL_TILE;
        uint c = gx * MATMUL_TILE + idx % MATMUL_TILE;
        float total = tileC[idx] + bias[c];
        if (epilogue == 2) total = max(total, 0.0f);
        else if (epilogue == 3) total = 1.0f / (1.0f + exp(-total));
        else if (epilogue == 4) total = tanh(total);
        output[r * columns + c] = total;
    }
}

constexpr constant uint MATMUL64 = 64;
constexpr constant uint MATMUL64_K = 32;
constexpr constant uint MATMUL64_GROUP = 256;

kernel void matrix_mul_mma64(device const float* a [[buffer(0)]],
                             device const float* b [[buffer(1)]],
                             device float* output [[buffer(2)]],
                             device const float* bias [[buffer(3)]],
                             constant uint& rows [[buffer(4)]],
                             constant uint& inner [[buffer(5)]],
                             constant uint& columns [[buffer(6)]],
                             constant uint& trans_a [[buffer(7)]],
                             constant uint& trans_b [[buffer(8)]],
                             constant uint& epilogue [[buffer(9)]],
                             uint lid [[thread_index_in_threadgroup]],
                             uint wid [[threadgroup_position_in_grid]],
                             uint sgitg [[simdgroup_index_in_threadgroup]]) {
    uint groups_x = columns / MATMUL64;
    uint gx = wid % groups_x;
    uint gy = wid / groups_x;
    uint sg_m = sgitg / 2;
    uint sg_n = sgitg % 2;
    uint row0 = gy * MATMUL64 + sg_m * 16;
    uint col0 = gx * MATMUL64 + sg_n * 32;
    simdgroup_float8x8 acc[2][4];
    for (uint i = 0; i < 2; ++i)
        for (uint j = 0; j < 4; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    (void)trans_a;
    (void)trans_b;
    for (uint k = 0; k < inner; k += 8) {
        simdgroup_float8x8 matA0;
        simdgroup_float8x8 matA1;
        simdgroup_float8x8 matB0;
        simdgroup_float8x8 matB1;
        simdgroup_float8x8 matB2;
        simdgroup_float8x8 matB3;
        simdgroup_load(matA0, a + row0 * inner + k, inner);
        simdgroup_load(matA1, a + (row0 + 8) * inner + k, inner);
        simdgroup_load(matB0, b + k * columns + col0, columns);
        simdgroup_load(matB1, b + k * columns + col0 + 8, columns);
        simdgroup_load(matB2, b + k * columns + col0 + 16, columns);
        simdgroup_load(matB3, b + k * columns + col0 + 24, columns);
        simdgroup_multiply_accumulate(acc[0][0], matA0, matB0, acc[0][0]);
        simdgroup_multiply_accumulate(acc[0][1], matA0, matB1, acc[0][1]);
        simdgroup_multiply_accumulate(acc[0][2], matA0, matB2, acc[0][2]);
        simdgroup_multiply_accumulate(acc[0][3], matA0, matB3, acc[0][3]);
        simdgroup_multiply_accumulate(acc[1][0], matA1, matB0, acc[1][0]);
        simdgroup_multiply_accumulate(acc[1][1], matA1, matB1, acc[1][1]);
        simdgroup_multiply_accumulate(acc[1][2], matA1, matB2, acc[1][2]);
        simdgroup_multiply_accumulate(acc[1][3], matA1, matB3, acc[1][3]);
    }
    for (uint i = 0; i < 2; ++i)
        for (uint j = 0; j < 4; ++j)
            simdgroup_store(acc[i][j], output + (row0 + i * 8) * columns + col0 + j * 8, columns);
    if (epilogue == 0) return;
    threadgroup_barrier(mem_flags::mem_device);
    for (uint t = 0; t < 4; ++t) {
        uint elem = (lid + t * MATMUL64_GROUP) * 4;
        uint r = gy * MATMUL64 + elem / MATMUL64;
        uint c = gx * MATMUL64 + elem % MATMUL64;
        device float* slot = output + r * columns + c;
        float4 total = *((device float4*)slot) + *((device const float4*)(bias + c));
        if (epilogue == 2) total = max(total, 0.0f);
        else if (epilogue == 3) total = 1.0f / (1.0f + exp(-total));
        else if (epilogue == 4) total = tanh(total);
        *((device float4*)slot) = total;
    }
}

kernel void matrix_mul_mma64_tn(device const float* a [[buffer(0)]],
                                device const float* b [[buffer(1)]],
                                device float* output [[buffer(2)]],
                                device const float* bias [[buffer(3)]],
                                constant uint& rows [[buffer(4)]],
                                constant uint& inner [[buffer(5)]],
                                constant uint& columns [[buffer(6)]],
                                constant uint& trans_a [[buffer(7)]],
                                constant uint& trans_b [[buffer(8)]],
                                constant uint& epilogue [[buffer(9)]],
                                uint lid [[thread_index_in_threadgroup]],
                                uint wid [[threadgroup_position_in_grid]],
                                uint sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float tileA[MATMUL64 * MATMUL64_K];
    threadgroup float tileB[MATMUL64_K * MATMUL64];
    uint groups_x = columns / MATMUL64;
    uint gx = wid % groups_x;
    uint gy = wid / groups_x;
    uint sg_m = sgitg / 2;
    uint sg_n = sgitg % 2;
    uint row0 = gy * MATMUL64 + sg_m * 16;
    uint col0 = gx * MATMUL64 + sg_n * 32;
    simdgroup_float8x8 acc[2][4];
    for (uint i = 0; i < 2; ++i)
        for (uint j = 0; j < 4; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    (void)trans_a;
    (void)trans_b;
    for (uint k0 = 0; k0 < inner; k0 += MATMUL64_K) {
        uint k_local = lid / 8;
        uint m_base = (lid % 8) * 8;
        uint am = gy * MATMUL64 + m_base;
        uint ak = k0 + k_local;
        uint b_elem = lid * 8;
        uint b_row = b_elem / MATMUL64;
        uint b_col = b_elem % MATMUL64;
        uint bk = k0 + b_row;
        uint bn = gx * MATMUL64 + b_col;
        if (k0 + MATMUL64_K <= inner) {
            float4 av0 = *((device const float4*)(a + ak * rows + am));
            float4 av1 = *((device const float4*)(a + ak * rows + am + 4));
            tileA[(m_base + 0) * MATMUL64_K + k_local] = av0.x;
            tileA[(m_base + 1) * MATMUL64_K + k_local] = av0.y;
            tileA[(m_base + 2) * MATMUL64_K + k_local] = av0.z;
            tileA[(m_base + 3) * MATMUL64_K + k_local] = av0.w;
            tileA[(m_base + 4) * MATMUL64_K + k_local] = av1.x;
            tileA[(m_base + 5) * MATMUL64_K + k_local] = av1.y;
            tileA[(m_base + 6) * MATMUL64_K + k_local] = av1.z;
            tileA[(m_base + 7) * MATMUL64_K + k_local] = av1.w;
            float4 bv0 = *((device const float4*)(b + bk * columns + bn));
            float4 bv1 = *((device const float4*)(b + bk * columns + bn + 4));
            *((threadgroup float4*)(tileB + b_elem)) = bv0;
            *((threadgroup float4*)(tileB + b_elem + 4)) = bv1;
        } else {
            if (ak < inner) {
                float4 av0 = *((device const float4*)(a + ak * rows + am));
                float4 av1 = *((device const float4*)(a + ak * rows + am + 4));
                tileA[(m_base + 0) * MATMUL64_K + k_local] = av0.x;
                tileA[(m_base + 1) * MATMUL64_K + k_local] = av0.y;
                tileA[(m_base + 2) * MATMUL64_K + k_local] = av0.z;
                tileA[(m_base + 3) * MATMUL64_K + k_local] = av0.w;
                tileA[(m_base + 4) * MATMUL64_K + k_local] = av1.x;
                tileA[(m_base + 5) * MATMUL64_K + k_local] = av1.y;
                tileA[(m_base + 6) * MATMUL64_K + k_local] = av1.z;
                tileA[(m_base + 7) * MATMUL64_K + k_local] = av1.w;
            } else {
                for (uint t = 0; t < 8; ++t)
                    tileA[(m_base + t) * MATMUL64_K + k_local] = 0.0f;
            }
            if (bk < inner) {
                float4 bv0 = *((device const float4*)(b + bk * columns + bn));
                float4 bv1 = *((device const float4*)(b + bk * columns + bn + 4));
                *((threadgroup float4*)(tileB + b_elem)) = bv0;
                *((threadgroup float4*)(tileB + b_elem + 4)) = bv1;
            } else {
                *((threadgroup float4*)(tileB + b_elem)) = float4(0.0f);
                *((threadgroup float4*)(tileB + b_elem + 4)) = float4(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (uint kk = 0; kk < MATMUL64_K; kk += 8) {
            simdgroup_float8x8 matA0;
            simdgroup_float8x8 matA1;
            simdgroup_float8x8 matB0;
            simdgroup_float8x8 matB1;
            simdgroup_float8x8 matB2;
            simdgroup_float8x8 matB3;
            simdgroup_load(matA0, tileA + (sg_m * 16) * MATMUL64_K + kk, MATMUL64_K);
            simdgroup_load(matA1, tileA + (sg_m * 16 + 8) * MATMUL64_K + kk, MATMUL64_K);
            simdgroup_load(matB0, tileB + kk * MATMUL64 + sg_n * 32, MATMUL64);
            simdgroup_load(matB1, tileB + kk * MATMUL64 + sg_n * 32 + 8, MATMUL64);
            simdgroup_load(matB2, tileB + kk * MATMUL64 + sg_n * 32 + 16, MATMUL64);
            simdgroup_load(matB3, tileB + kk * MATMUL64 + sg_n * 32 + 24, MATMUL64);
            simdgroup_multiply_accumulate(acc[0][0], matA0, matB0, acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1], matA0, matB1, acc[0][1]);
            simdgroup_multiply_accumulate(acc[0][2], matA0, matB2, acc[0][2]);
            simdgroup_multiply_accumulate(acc[0][3], matA0, matB3, acc[0][3]);
            simdgroup_multiply_accumulate(acc[1][0], matA1, matB0, acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1], matA1, matB1, acc[1][1]);
            simdgroup_multiply_accumulate(acc[1][2], matA1, matB2, acc[1][2]);
            simdgroup_multiply_accumulate(acc[1][3], matA1, matB3, acc[1][3]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < 2; ++i)
        for (uint j = 0; j < 4; ++j)
            simdgroup_store(acc[i][j], output + (row0 + i * 8) * columns + col0 + j * 8, columns);
    if (epilogue == 0) return;
    threadgroup_barrier(mem_flags::mem_device);
    for (uint t = 0; t < 4; ++t) {
        uint elem = (lid + t * MATMUL64_GROUP) * 4;
        uint r = gy * MATMUL64 + elem / MATMUL64;
        uint c = gx * MATMUL64 + elem % MATMUL64;
        device float* slot = output + r * columns + c;
        float4 total = *((device float4*)slot) + *((device const float4*)(bias + c));
        if (epilogue == 2) total = max(total, 0.0f);
        else if (epilogue == 3) total = 1.0f / (1.0f + exp(-total));
        else if (epilogue == 4) total = tanh(total);
        *((device float4*)slot) = total;
    }
}

kernel void matrix_mul_mma64_nt(device const float* a [[buffer(0)]],
                                device const float* b [[buffer(1)]],
                                device float* output [[buffer(2)]],
                                device const float* bias [[buffer(3)]],
                                constant uint& rows [[buffer(4)]],
                                constant uint& inner [[buffer(5)]],
                                constant uint& columns [[buffer(6)]],
                                constant uint& trans_a [[buffer(7)]],
                                constant uint& trans_b [[buffer(8)]],
                                constant uint& epilogue [[buffer(9)]],
                                uint lid [[thread_index_in_threadgroup]],
                                uint wid [[threadgroup_position_in_grid]],
                                uint sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float tileA[MATMUL64 * MATMUL64_K];
    threadgroup float tileB[MATMUL64_K * MATMUL64];
    uint groups_x = columns / MATMUL64;
    uint gx = wid % groups_x;
    uint gy = wid / groups_x;
    uint sg_m = sgitg / 2;
    uint sg_n = sgitg % 2;
    uint row0 = gy * MATMUL64 + sg_m * 16;
    uint col0 = gx * MATMUL64 + sg_n * 32;
    simdgroup_float8x8 acc[2][4];
    for (uint i = 0; i < 2; ++i)
        for (uint j = 0; j < 4; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    (void)trans_a;
    (void)trans_b;
    for (uint k0 = 0; k0 < inner; k0 += MATMUL64_K) {
        uint a_elem = lid * 8;
        uint a_row = a_elem / MATMUL64_K;
        uint a_col = a_elem % MATMUL64_K;
        uint am = gy * MATMUL64 + a_row;
        uint ak = k0 + a_col;
        uint n_local = lid % MATMUL64;
        uint k_base = (lid / MATMUL64) * 8;
        uint bn = gx * MATMUL64 + n_local;
        uint bk = k0 + k_base;
        if (k0 + MATMUL64_K <= inner) {
            float4 av0 = *((device const float4*)(a + am * inner + ak));
            float4 av1 = *((device const float4*)(a + am * inner + ak + 4));
            *((threadgroup float4*)(tileA + a_elem)) = av0;
            *((threadgroup float4*)(tileA + a_elem + 4)) = av1;
            float4 bv0 = *((device const float4*)(b + bn * inner + bk));
            float4 bv1 = *((device const float4*)(b + bn * inner + bk + 4));
            tileB[(k_base + 0) * MATMUL64 + n_local] = bv0.x;
            tileB[(k_base + 1) * MATMUL64 + n_local] = bv0.y;
            tileB[(k_base + 2) * MATMUL64 + n_local] = bv0.z;
            tileB[(k_base + 3) * MATMUL64 + n_local] = bv0.w;
            tileB[(k_base + 4) * MATMUL64 + n_local] = bv1.x;
            tileB[(k_base + 5) * MATMUL64 + n_local] = bv1.y;
            tileB[(k_base + 6) * MATMUL64 + n_local] = bv1.z;
            tileB[(k_base + 7) * MATMUL64 + n_local] = bv1.w;
        } else {
            for (uint t = 0; t < 8; ++t)
                tileA[a_elem + t] = (ak + t < inner) ? a[am * inner + ak + t] : 0.0f;
            for (uint t = 0; t < 8; ++t)
                tileB[(k_base + t) * MATMUL64 + n_local] = (bk + t < inner) ? b[bn * inner + bk + t] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (uint kk = 0; kk < MATMUL64_K; kk += 8) {
            simdgroup_float8x8 matA0;
            simdgroup_float8x8 matA1;
            simdgroup_float8x8 matB0;
            simdgroup_float8x8 matB1;
            simdgroup_float8x8 matB2;
            simdgroup_float8x8 matB3;
            simdgroup_load(matA0, tileA + (sg_m * 16) * MATMUL64_K + kk, MATMUL64_K);
            simdgroup_load(matA1, tileA + (sg_m * 16 + 8) * MATMUL64_K + kk, MATMUL64_K);
            simdgroup_load(matB0, tileB + kk * MATMUL64 + sg_n * 32, MATMUL64);
            simdgroup_load(matB1, tileB + kk * MATMUL64 + sg_n * 32 + 8, MATMUL64);
            simdgroup_load(matB2, tileB + kk * MATMUL64 + sg_n * 32 + 16, MATMUL64);
            simdgroup_load(matB3, tileB + kk * MATMUL64 + sg_n * 32 + 24, MATMUL64);
            simdgroup_multiply_accumulate(acc[0][0], matA0, matB0, acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1], matA0, matB1, acc[0][1]);
            simdgroup_multiply_accumulate(acc[0][2], matA0, matB2, acc[0][2]);
            simdgroup_multiply_accumulate(acc[0][3], matA0, matB3, acc[0][3]);
            simdgroup_multiply_accumulate(acc[1][0], matA1, matB0, acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1], matA1, matB1, acc[1][1]);
            simdgroup_multiply_accumulate(acc[1][2], matA1, matB2, acc[1][2]);
            simdgroup_multiply_accumulate(acc[1][3], matA1, matB3, acc[1][3]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < 2; ++i)
        for (uint j = 0; j < 4; ++j)
            simdgroup_store(acc[i][j], output + (row0 + i * 8) * columns + col0 + j * 8, columns);
    if (epilogue == 0) return;
    threadgroup_barrier(mem_flags::mem_device);
    for (uint t = 0; t < 4; ++t) {
        uint elem = (lid + t * MATMUL64_GROUP) * 4;
        uint r = gy * MATMUL64 + elem / MATMUL64;
        uint c = gx * MATMUL64 + elem % MATMUL64;
        device float* slot = output + r * columns + c;
        float4 total = *((device float4*)slot) + *((device const float4*)(bias + c));
        if (epilogue == 2) total = max(total, 0.0f);
        else if (epilogue == 3) total = 1.0f / (1.0f + exp(-total));
        else if (epilogue == 4) total = tanh(total);
        *((device float4*)slot) = total;
    }
}

kernel void matrix_mul_tiled(device const float* a [[buffer(0)]],
                             device const float* b [[buffer(1)]],
                             device float* output [[buffer(2)]],
                             device const float* bias [[buffer(3)]],
                             constant uint& rows [[buffer(4)]],
                             constant uint& inner [[buffer(5)]],
                             constant uint& columns [[buffer(6)]],
                             constant uint& trans_a [[buffer(7)]],
                             constant uint& trans_b [[buffer(8)]],
                             constant uint& epilogue [[buffer(9)]],
                             constant uint& skip_aligned [[buffer(10)]],
                             constant uint& row_origin [[buffer(11)]],
                             constant uint& col_origin [[buffer(12)]],
                             uint lid [[thread_index_in_threadgroup]],
                             uint wid [[threadgroup_position_in_grid]],
                             uint sgitg [[simdgroup_index_in_threadgroup]]) {
    uint groups_x = (columns - col_origin + MATMUL_TILE - 1) / MATMUL_TILE;
    uint gx = wid % groups_x;
    uint gy = wid / groups_x;
    uint tile_row = row_origin + gy * MATMUL_TILE;
    uint tile_col = col_origin + gx * MATMUL_TILE;
    if (skip_aligned && (tile_row + MATMUL_TILE <= rows) && (tile_col + MATMUL_TILE <= columns))
        return;
    threadgroup float tileA[MATMUL_TILE * MATMUL_TILE];
    threadgroup float tileB[MATMUL_TILE * MATMUL_TILE];
    threadgroup float tileC[MATMUL_TILE * MATMUL_TILE];
    uint sg_m = sgitg / 2;
    uint sg_n = sgitg % 2;
    simdgroup_float8x8 acc00 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc01 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc10 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc11 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k0 = 0; k0 < inner; k0 += MATMUL_TILE) {
        for (uint t = 0; t < 8; ++t) {
            uint idx = lid + t * MATMUL_GROUP;
            uint tr = idx / MATMUL_TILE;
            uint tc = idx % MATMUL_TILE;
            uint am = tile_row + tr;
            uint ak = k0 + tc;
            float av = 0.0f;
            if (am < rows && ak < inner)
                av = trans_a ? a[ak * rows + am] : a[am * inner + ak];
            tileA[idx] = av;
            uint bk = k0 + tr;
            uint bn = tile_col + tc;
            float bv = 0.0f;
            if (bk < inner && bn < columns)
                bv = trans_b ? b[bn * inner + bk] : b[bk * columns + bn];
            tileB[idx] = bv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < MATMUL_TILE; kk += 8) {
            simdgroup_float8x8 matA0;
            simdgroup_float8x8 matA1;
            simdgroup_float8x8 matB0;
            simdgroup_float8x8 matB1;
            simdgroup_load(matA0, tileA + (sg_m * 16) * MATMUL_TILE + kk, MATMUL_TILE);
            simdgroup_load(matA1, tileA + (sg_m * 16 + 8) * MATMUL_TILE + kk, MATMUL_TILE);
            simdgroup_load(matB0, tileB + kk * MATMUL_TILE + sg_n * 16, MATMUL_TILE);
            simdgroup_load(matB1, tileB + kk * MATMUL_TILE + sg_n * 16 + 8, MATMUL_TILE);
            simdgroup_multiply_accumulate(acc00, matA0, matB0, acc00);
            simdgroup_multiply_accumulate(acc01, matA0, matB1, acc01);
            simdgroup_multiply_accumulate(acc10, matA1, matB0, acc10);
            simdgroup_multiply_accumulate(acc11, matA1, matB1, acc11);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(acc00, tileC + (sg_m * 16) * MATMUL_TILE + sg_n * 16, MATMUL_TILE);
    simdgroup_store(acc01, tileC + (sg_m * 16) * MATMUL_TILE + sg_n * 16 + 8, MATMUL_TILE);
    simdgroup_store(acc10, tileC + (sg_m * 16 + 8) * MATMUL_TILE + sg_n * 16, MATMUL_TILE);
    simdgroup_store(acc11, tileC + (sg_m * 16 + 8) * MATMUL_TILE + sg_n * 16 + 8, MATMUL_TILE);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = 0; t < 8; ++t) {
        uint idx = lid + t * MATMUL_GROUP;
        uint tr = idx / MATMUL_TILE;
        uint tc = idx % MATMUL_TILE;
        uint r = tile_row + tr;
        uint c = tile_col + tc;
        if (r < rows && c < columns) {
            float total = tileC[idx];
            if (epilogue != 0) total += bias[c];
            if (epilogue == 2) total = max(total, 0.0f);
            else if (epilogue == 3) total = 1.0f / (1.0f + exp(-total));
            else if (epilogue == 4) total = tanh(total);
            output[r * columns + c] = total;
        }
    }
}

kernel void add_scaled_into(device const float* value [[buffer(0)]],
                            device float* total [[buffer(1)]],
                            constant uint& weight [[buffer(2)]],
                            uint id [[thread_position_in_grid]]) {
    if (id == 0) total[0] += value[0] * float(weight);
}

kernel void add_correct_into(device const float* logits [[buffer(0)]],
                             device const float* labels [[buffer(1)]],
                             device float* total [[buffer(2)]],
                             constant uint& batch [[buffer(3)]],
                             constant uint& classes [[buffer(4)]],
                             uint sample [[thread_position_in_grid]]) {
    if (sample >= batch) return;
    uint base = sample * classes;
    uint best = 0;
    float peak = logits[base];
    for (uint column = 1; column < classes; ++column) {
        float value = logits[base + column];
        if (value > peak) { peak = value; best = column; }
    }
    if (uint(labels[sample]) == best)
        atomic_fetch_add_explicit((device atomic<float>*)total, 1.0f, memory_order_relaxed);
}
"""

var _kernel_cache: dict[str, Kernel] = {}

fn cached_kernel(source: str, entry: str) -> Kernel {
    let key = entry + "\n" + source
    let existing = _kernel_cache.get(key, null)
    if existing is not null { return existing }
    let kernel = Kernel.compile(source, entry)
    _kernel_cache[key] = kernel
    return kernel
}

fn binary(a: Tensor, b: Tensor, operation: int) -> Tensor {
    if not a.same_shape(b) { throw ValueError(f"tensor shapes differ: {a.shape} and {b.shape}") }
    let output = Tensor(a.shape)
    let kernel = cached_kernel(SOURCE, "binary_op")
    kernel.dispatch_async([a.buffer(), b.buffer(), output.buffer()], a.size, [a.size, operation])
    return output
}

fn unary(input: Tensor, operation: int) -> Tensor {
    let output = Tensor(input.shape)
    let kernel = cached_kernel(SOURCE, "unary_op")
    kernel.dispatch_async([input.buffer(), output.buffer()], input.size, [input.size, operation])
    return output
}

fn is_trailing_suffix(small: list[int], large: list[int]) -> bool {
    if small.len() > large.len() { return false }
    let offset = large.len() - small.len()
    for index in 0..small.len() {
        if small[index] != large[offset + index] { return false }
    }
    return true
}

fn broadcast_add(a: Tensor, b: Tensor) -> Tensor {
    let output = Tensor(a.shape)
    let kernel = cached_kernel(SOURCE, "broadcast_add")
    kernel.dispatch_async([a.buffer(), b.buffer(), output.buffer()], a.size, [b.size, a.size])
    return output
}

fn binary_backward(a: Tensor, b: Tensor, upstream: Tensor, operation: int) -> list[Tensor] {
    let grad_a = Tensor(a.shape)
    let grad_b = Tensor(b.shape)
    let kernel = cached_kernel(SOURCE, "binary_backward_op")
    kernel.dispatch_async(
        [a.buffer(), b.buffer(), upstream.buffer(), grad_a.buffer(), grad_b.buffer()],
        a.size,
        [a.size, operation]
    )
    return [grad_a, grad_b]
}

fn reduce_leading(upstream: Tensor, small_shape: list[int], out: Tensor? = null) -> Tensor {
    let inner = element_count(small_shape)
    let outer = upstream.size // inner
    let output = if out is not null {
        if out.size != inner {
            throw ValueError(f"reduce_leading out size {out.size} does not match {inner}")
        }
        out
    } else {
        Tensor(small_shape)
    }
    if inner % 4 == 0 {
        let kernel = cached_kernel(SOURCE, "reduce_leading4")
        kernel.dispatch_async([upstream.buffer(), output.buffer()], inner // 4, [outer, inner])
    } else {
        let kernel = cached_kernel(SOURCE, "reduce_leading")
        kernel.dispatch_async([upstream.buffer(), output.buffer()], inner, [outer, inner])
    }
    return output
}

#: Elementwise sum of equal-shaped tensors, or of `a` with a `b` whose shape
#: is a trailing suffix of `a`'s (for example `[units]` against `[batch,
#: units]`).
pub fn add(a: Tensor, b: Tensor) -> Tensor {
    let broadcasting = not a.same_shape(b)
    if broadcasting and not is_trailing_suffix(b.shape, a.shape) {
        throw ValueError(f"cannot broadcast {b.shape} against {a.shape}")
    }
    let output = if broadcasting { broadcast_add(a, b) } else { binary(a, b, 0) }
    if a.requires_grad or b.requires_grad {
        output.attach_creator(GradNode([a, b], |upstream| if broadcasting {
            [unary(upstream, 0), reduce_leading(upstream, b.shape)]
        } else {
            binary_backward(a, b, upstream, 0)
        }))
    }
    return output
}

#: Elementwise difference of equal-shaped tensors.
pub fn subtract(a: Tensor, b: Tensor) -> Tensor {
    let output = binary(a, b, 1)
    if a.requires_grad or b.requires_grad {
        output.attach_creator(GradNode([a, b], |upstream| binary_backward(a, b, upstream, 1)))
    }
    return output
}

#: Elementwise product of equal-shaped tensors.
pub fn multiply(a: Tensor, b: Tensor) -> Tensor {
    let output = binary(a, b, 2)
    if a.requires_grad or b.requires_grad {
        output.attach_creator(GradNode([a, b], |upstream| binary_backward(a, b, upstream, 2)))
    }
    return output
}

#: Elementwise quotient of equal-shaped tensors.
pub fn divide(a: Tensor, b: Tensor) -> Tensor {
    let output = binary(a, b, 3)
    if a.requires_grad or b.requires_grad {
        output.attach_creator(GradNode([a, b], |upstream| binary_backward(a, b, upstream, 3)))
    }
    return output
}

#: Multiply every element by one host scalar.
pub fn scale(input: Tensor, value: float) -> Tensor {
    let output = Tensor(input.shape)
    let scalar = gpu.Buffer.of([value])
    defer { scalar.free() }
    let kernel = cached_kernel(SOURCE, "scalar_mul")
    kernel.dispatch_async([input.buffer(), scalar, output.buffer()], input.size, [input.size])
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [scale(upstream, value)]))
    }
    return output
}

#: Copy a tensor while giving the result a new shape with the same size.
pub fn reshape(input: Tensor, shape: list[int]) -> Tensor {
    if element_count(shape) != input.size {
        throw ValueError(f"cannot reshape {input.shape} into {shape}")
    }
    let output = Tensor(shape)
    let kernel = cached_kernel(SOURCE, "unary_op")
    kernel.dispatch_async([input.buffer(), output.buffer()], input.size, [input.size, 0])
    if input.requires_grad {
        let original_shape = input.shape
        output.attach_creator(GradNode([input], |upstream| [reshape(upstream, original_shape)]))
    }
    return output
}

fn uses_mma64(rows: int, inner: int, columns: int, trans_a: bool, trans_b: bool) -> bool {
    if trans_a and trans_b { return false }
    if rows < 128 or columns < 128 { return false }
    if columns // 64 == 0 or rows // 64 == 0 or inner % 8 != 0 { return false }
    if rows >= 1024 or columns >= 1024 or inner >= 1024 { return true }
    return rows >= 512 and
        columns >= 512 and
        inner >= 512 and
        rows % 64 == 0 and
        columns % 64 == 0 and
        inner % 64 == 0
}

fn raw_matmul(
    a: Tensor,
    b: Tensor,
    trans_a: bool = false,
    trans_b: bool = false,
    bias: Tensor? = null,
    epilogue: int = 0,
    out: Tensor? = null
) -> Tensor {
    let rows = if trans_a { a.shape[1] } else { a.shape[0] }
    let inner = if trans_a { a.shape[0] } else { a.shape[1] }
    let inner_b = if trans_b { b.shape[1] } else { b.shape[0] }
    let columns = if trans_b { b.shape[0] } else { b.shape[1] }
    if inner != inner_b {
        throw ValueError(f"matmul inner dimensions differ: {a.shape} and {b.shape}")
    }
    let output = if out is not null {
        if out.shape != [rows, columns] {
            throw ValueError(f"matmul out shape {out.shape} does not match [{rows}, {columns}]")
        }
        out
    } else {
        Tensor([rows, columns])
    }
    let tile = 32
    let tile64 = 64
    let group = 128
    let group64 = 256
    let aligned_x = columns // tile
    let aligned_y = rows // tile
    let aligned64_x = columns // tile64
    let aligned64_y = rows // tile64
    let bias_buffer = if bias is not null { bias.buffer() } else { output.buffer() }
    let trans_a_flag = if trans_a { 1 } else { 0 }
    let trans_b_flag = if trans_b { 1 } else { 0 }
    let use64 = uses_mma64(rows, inner, columns, trans_a, trans_b)
    let use_mma = not use64 and
        aligned_x > 0 and
        aligned_y > 0 and
        (epilogue == 0 or inner % 8 == 0)
    if use64 {
        let entry = if trans_a {
            "matrix_mul_mma64_tn"
        } elif trans_b {
            "matrix_mul_mma64_nt"
        } else {
            "matrix_mul_mma64"
        }
        let kernel = cached_kernel(SOURCE, entry)
        kernel.dispatch_async(
            [a.buffer(), b.buffer(), output.buffer(), bias_buffer],
            aligned64_x * aligned64_y * group64,
            [rows, inner, columns, trans_a_flag, trans_b_flag, epilogue],
            group_size: group64
        )
        dispatch_tiled_region(
            a,
            b,
            output,
            bias_buffer,
            rows,
            inner,
            columns,
            trans_a_flag,
            trans_b_flag,
            epilogue,
            aligned64_y * tile64,
            0,
            rows - aligned64_y * tile64,
            columns
        )
        dispatch_tiled_region(
            a,
            b,
            output,
            bias_buffer,
            rows,
            inner,
            columns,
            trans_a_flag,
            trans_b_flag,
            epilogue,
            0,
            aligned64_x * tile64,
            aligned64_y * tile64,
            columns - aligned64_x * tile64
        )
    } elif use_mma {
        let entry = if epilogue != 0 {
            "matrix_mul_mma_bias"
        } elif inner % 8 == 0 {
            "matrix_mul_mma"
        } else {
            "matrix_mul_mma_k"
        }
        let kernel = cached_kernel(SOURCE, entry)
        kernel.dispatch_async(
            [a.buffer(), b.buffer(), output.buffer(), bias_buffer],
            aligned_x * aligned_y * group,
            [rows, inner, columns, trans_a_flag, trans_b_flag, epilogue],
            group_size: group
        )
        dispatch_tiled_region(
            a,
            b,
            output,
            bias_buffer,
            rows,
            inner,
            columns,
            trans_a_flag,
            trans_b_flag,
            epilogue,
            aligned_y * tile,
            0,
            rows - aligned_y * tile,
            columns
        )
        dispatch_tiled_region(
            a,
            b,
            output,
            bias_buffer,
            rows,
            inner,
            columns,
            trans_a_flag,
            trans_b_flag,
            epilogue,
            0,
            aligned_x * tile,
            aligned_y * tile,
            columns - aligned_x * tile
        )
    } else {
        dispatch_tiled_region(
            a,
            b,
            output,
            bias_buffer,
            rows,
            inner,
            columns,
            trans_a_flag,
            trans_b_flag,
            epilogue,
            0,
            0,
            rows,
            columns
        )
    }
    return output
}

fn dispatch_tiled_region(
    a: Tensor,
    b: Tensor,
    output: Tensor,
    bias_buffer: Buffer,
    rows: int,
    inner: int,
    columns: int,
    trans_a_flag: int,
    trans_b_flag: int,
    epilogue: int,
    row_origin: int,
    col_origin: int,
    region_rows: int,
    region_cols: int
) -> void {
    if region_rows <= 0 or region_cols <= 0 { return }
    let tile = 32
    let group = 128
    let groups_x = (region_cols + tile - 1) // tile
    let groups_y = (region_rows + tile - 1) // tile
    let kernel = cached_kernel(SOURCE, "matrix_mul_tiled")
    kernel.dispatch_async(
        [a.buffer(), b.buffer(), output.buffer(), bias_buffer],
        groups_x * groups_y * group,
        [rows, inner, columns, trans_a_flag, trans_b_flag, epilogue, 0, row_origin, col_origin],
        group_size: group
    )
}

fn raw_transpose(input: Tensor) -> Tensor {
    let rows = input.shape[0]
    let columns = input.shape[1]
    let output = Tensor([columns, rows])
    let kernel = cached_kernel(SOURCE, "matrix_transpose")
    kernel.dispatch_async([input.buffer(), output.buffer()], input.size, [rows, columns])
    return output
}

#: Matrix product of two rank-two tensors.
#:
#: `trans_a` / `trans_b` read `a` or `b` as if transposed without copying.
#: Gradient graphs are only attached for the untransposed product.
pub fn matmul(a: Tensor, b: Tensor, trans_a: bool = false, trans_b: bool = false) -> Tensor {
    if a.rank() != 2 or b.rank() != 2 {
        throw ValueError(f"matmul needs rank-two tensors, got {a.shape} and {b.shape}")
    }
    let a_inner = if trans_a { a.shape[0] } else { a.shape[1] }
    let b_inner = if trans_b { b.shape[1] } else { b.shape[0] }
    if a_inner != b_inner {
        throw ValueError(f"matmul inner dimensions differ: {a.shape} and {b.shape}")
    }
    let output = raw_matmul(a, b, trans_a: trans_a, trans_b: trans_b)
    if a.requires_grad or b.requires_grad {
        if trans_a or trans_b {
            throw ValueError("transposed matmul does not attach gradients")
        }
        output.attach_creator(GradNode([a, b], fn(upstream: Tensor) -> list[Tensor] {
            return [raw_matmul(upstream, b, trans_b: true), raw_matmul(a, upstream, trans_a: true)]
        }))
    }
    return output
}

#: Transpose a rank-two tensor.
pub fn transpose(input: Tensor) -> Tensor {
    if input.rank() != 2 { throw ValueError(f"transpose needs rank two, got {input.shape}") }
    let output = raw_transpose(input)
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [raw_transpose(upstream)]))
    }
    return output
}

fn unary_backward(saved: Tensor, upstream: Tensor, operation: int) -> Tensor {
    let grad_input = Tensor(saved.shape)
    if operation == 1 and saved.size % 4 == 0 {
        let kernel = cached_kernel(SOURCE, "unary_backward_relu4")
        kernel.dispatch_async(
            [saved.buffer(), upstream.buffer(), grad_input.buffer()],
            saved.size // 4,
            [saved.size]
        )
        return grad_input
    }
    let kernel = cached_kernel(SOURCE, "unary_backward_op")
    kernel.dispatch_async(
        [saved.buffer(), upstream.buffer(), grad_input.buffer()],
        saved.size,
        [saved.size, operation]
    )
    return grad_input
}

#: Elementwise rectified linear activation.
pub fn relu(input: Tensor) -> Tensor {
    let output = unary(input, 1)
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [unary_backward(output, upstream, 1)]))
    }
    return output
}

#: Elementwise logistic sigmoid activation.
pub fn sigmoid(input: Tensor) -> Tensor {
    let output = unary(input, 2)
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [unary_backward(output, upstream, 2)]))
    }
    return output
}

#: Elementwise hyperbolic tangent activation.
pub fn tanh(input: Tensor) -> Tensor {
    let output = unary(input, 3)
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [unary_backward(output, upstream, 3)]))
    }
    return output
}

#: Affine transform `x @ weights + bias` with an optional fused activation.
#:
#: `activation` matches `Dense.activation_code`: 0 linear, 1 ReLU, 2 sigmoid,
#: 3 tanh. Backward skips GEMMs for inputs that do not require a gradient,
#: so a first-layer feature batch never materializes `dX`.
pub fn dense_affine(x: Tensor, weights: Tensor, bias: Tensor, activation: int) -> Tensor {
    if activation < 0 or activation > 3 {
        throw ValueError(f"dense_affine activation must be 0..3, got {activation}")
    }
    let output = raw_matmul(x, weights, bias: bias, epilogue: activation + 1)
    let need_x = x.requires_grad
    let need_w = weights.requires_grad
    let need_b = bias.requires_grad
    if need_x or need_w or need_b {
        var inputs: list[Tensor] = []
        if need_x { inputs.push(x) }
        if need_w { inputs.push(weights) }
        if need_b { inputs.push(bias) }
        output.attach_creator(GradNode(inputs, fn(upstream: Tensor) -> list[Tensor] {
            let affine_grad = if activation == 0 {
                upstream
            } else {
                unary_backward(output, upstream, activation)
            }
            var grads: list[Tensor] = []
            if need_x {
                grads.push(raw_matmul(affine_grad, weights, trans_b: true))
            }
            if need_w {
                grads.push(raw_matmul(x, affine_grad, trans_a: true, out: weights.grad))
            }
            if need_b {
                grads.push(reduce_leading(affine_grad, bias.shape, out: bias.grad))
            }
            if activation != 0 {
                affine_grad.free()
            }
            return grads
        }))
    }
    return output
}

var _warmed = false

#: Compile training kernels and run a tiny dispatch before the first timed epoch.
pub fn warmup_kernels() -> void {
    if _warmed { return }
    cached_kernel(SOURCE, "matrix_mul_mma")
    cached_kernel(SOURCE, "matrix_mul_mma_k")
    cached_kernel(SOURCE, "matrix_mul_mma_bias")
    cached_kernel(SOURCE, "matrix_mul_mma64")
    cached_kernel(SOURCE, "matrix_mul_mma64_tn")
    cached_kernel(SOURCE, "matrix_mul_mma64_nt")
    cached_kernel(SOURCE, "matrix_mul_tiled")
    cached_kernel(SOURCE, "unary_backward_op")
    cached_kernel(SOURCE, "unary_backward_relu4")
    cached_kernel(SOURCE, "reduce_leading")
    cached_kernel(SOURCE, "reduce_leading4")
    cached_kernel(SOURCE, "add_scaled_into")
    cached_kernel(SOURCE, "add_correct_into")
    let wide = Tensor([64, 1024])
    let wide_w = Tensor([1024, 64])
    let wide_out = raw_matmul(wide, wide_w)
    let wide_k = Tensor([1024, 64])
    let wide_tn = raw_matmul(wide_k, wide_k, trans_a: true)
    let wide_nt = raw_matmul(wide, wide, trans_b: true)
    gpu.synchronize()
    wide_nt.free()
    wide_tn.free()
    wide_k.free()
    wide_out.free()
    wide.free()
    wide_w.free()
    let mid = Tensor([256, 256])
    let mid_out = raw_matmul(mid, mid, bias: mid, epilogue: 2)
    let mid_tn = raw_matmul(mid, mid, trans_a: true)
    let mid_nt = raw_matmul(mid, mid, trans_b: true)
    gpu.synchronize()
    mid_nt.free()
    mid_tn.free()
    mid_out.free()
    mid.free()
    let huge = Tensor([64, 4096])
    let huge_w = Tensor([4096, 64])
    let huge_b = Tensor.zeros([64])
    let huge_out = raw_matmul(huge, huge_w, bias: huge_b, epilogue: 2)
    gpu.synchronize()
    huge_out.free()
    huge_b.free()
    huge.free()
    huge_w.free()
    let x = Tensor([512, 784])
    let weights = Tensor([784, 256])
    let bias = Tensor.zeros([256])
    let hidden = raw_matmul(x, weights, bias: bias, epilogue: 2)
    let grad_x = raw_matmul(hidden, weights, trans_b: true)
    let grad_w = raw_matmul(x, hidden, trans_a: true)
    let head_w = Tensor([256, 10])
    let head_b = Tensor.zeros([10])
    let logits = raw_matmul(hidden, head_w, bias: head_b, epilogue: 1)
    gpu.synchronize()
    logits.free()
    head_b.free()
    head_w.free()
    grad_w.free()
    grad_x.free()
    hidden.free()
    x.free()
    weights.free()
    bias.free()
    _warmed = true
}

#: Elementwise exponential.
pub fn exp(input: Tensor) -> Tensor {
    let output = unary(input, 4)
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [unary_backward(output, upstream, 4)]))
    }
    return output
}

#: Elementwise natural logarithm.
pub fn log(input: Tensor) -> Tensor {
    let output = unary(input, 5)
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [unary_backward(input, upstream, 5)]))
    }
    return output
}

#: Normalize each row of a rank-two tensor into a probability distribution.
pub fn softmax(input: Tensor) -> Tensor {
    if input.rank() != 2 { throw ValueError(f"softmax needs rank two, got {input.shape}") }
    let rows = input.shape[0]
    let columns = input.shape[1]
    let output = Tensor(input.shape)
    let kernel = cached_kernel(SOURCE, "row_softmax")
    kernel.dispatch_async([input.buffer(), output.buffer()], rows, [rows, columns])
    if input.requires_grad {
        output.attach_creator(GradNode([input], fn(upstream: Tensor) -> list[Tensor] {
            let grad_input = Tensor(input.shape)
            let backward_kernel = cached_kernel(SOURCE, "softmax_backward")
            backward_kernel.dispatch_async([output.buffer(), upstream.buffer(), grad_input.buffer()], rows, [rows, columns])
            return [grad_input]
        }))
    }
    return output
}

#: Index of the largest value in each row of a rank-two tensor.
#:
#: Indices use float storage because all Metal buffers in `std.gpu` are
#: float32. Downloaded values are exact for practical class counts.
pub fn argmax(input: Tensor) -> Tensor {
    if input.rank() != 2 { throw ValueError(f"argmax needs rank two, got {input.shape}") }
    let rows = input.shape[0]
    let columns = input.shape[1]
    let output = Tensor([rows])
    let kernel = cached_kernel(SOURCE, "row_argmax")
    kernel.dispatch_async([input.buffer(), output.buffer()], rows, [rows, columns])
    return output
}

#: Expand a rank-one tensor of integer-valued indices into one-hot rows.
pub fn one_hot(indices: Tensor, depth: int) -> Tensor {
    if indices.rank() != 1 { throw ValueError(f"one_hot needs rank one, got {indices.shape}") }
    if depth <= 0 { throw ValueError(f"depth must be positive, got {depth}") }
    let rows = indices.shape[0]
    let output = Tensor([rows, depth])
    let kernel = cached_kernel(SOURCE, "one_hot_rows")
    kernel.dispatch_async([indices.buffer(), output.buffer()], rows * depth, [rows, depth])
    return output
}

#: Copy a contiguous range of `source` into a new tensor of `shape`.
#:
#: The copy runs on the device. `source_offset` is in elements, not rows, so a
#: batch starting at row `start` of a `[samples, width]` matrix uses offset
#: `start * width`. The result is a leaf: it does not attach a gradient graph.
pub fn slice_flat(source: Tensor, source_offset: int, shape: list[int]) -> Tensor {
    let output = Tensor(shape)
    if source_offset < 0 or source_offset + output.size > source.size {
        output.free()
        throw ValueError(
            f"slice_flat offset {source_offset} and size {output.size} " +
                f"run past source size {source.size}"
        )
    }
    let kernel = cached_kernel(SOURCE, "copy_offset")
    kernel.dispatch_async(
        [source.buffer(), output.buffer()],
        output.size,
        [source_offset, output.size]
    )
    return output
}

#: Add `value[0] * weight` into a scalar accumulator tensor.
pub fn add_scaled_into(value: Tensor, total: Tensor, weight: int) -> void {
    if value.size != 1 or total.size != 1 {
        throw ValueError("add_scaled_into needs scalar tensors")
    }
    if weight < 0 { throw ValueError(f"weight must be non-negative, got {weight}") }
    let kernel = cached_kernel(SOURCE, "add_scaled_into")
    kernel.dispatch_async([value.buffer(), total.buffer()], 1, [weight])
}

#: Add the number of correct argmax classes into a scalar accumulator.
pub fn add_correct_into(logits: Tensor, labels: Tensor, total: Tensor) -> void {
    if logits.rank() != 2 {
        throw ValueError(f"add_correct_into needs rank-two logits, got {logits.shape}")
    }
    if total.size != 1 { throw ValueError("add_correct_into needs a scalar accumulator") }
    let batch = logits.shape[0]
    let classes = logits.shape[1]
    if labels.size != batch {
        throw ValueError(f"add_correct_into needs {batch} labels, got {labels.size}")
    }
    let kernel = cached_kernel(SOURCE, "add_correct_into")
    kernel.dispatch_async(
        [logits.buffer(), labels.buffer(), total.buffer()],
        batch,
        [batch, classes]
    )
}
