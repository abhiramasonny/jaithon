#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

/* Packed [seq, heads*hd] float → BHSD [heads, seq, hd] half. One pass over
 * Q, K, and V so the prefill kernel streams contiguous half K/V instead of
 * converting fp32 on every tile. */
kernel void jaiPackMhaHalf(device const float *Q [[buffer(0)]],
                           device const float *K [[buffer(1)]],
                           device const float *V [[buffer(2)]],
                           device half *Qh [[buffer(3)]],
                           device half *Kh [[buffer(4)]],
                           device half *Vh [[buffer(5)]],
                           constant uint &seq [[buffer(6)]],
                           constant uint &heads [[buffer(7)]],
                           constant uint &hd [[buffer(8)]],
                           uint id [[thread_position_in_grid]]) {
    uint vecs = hd / 4u;
    uint count = heads * seq * vecs;
    if (id >= count) return;
    uint d4 = id % vecs;
    uint s = (id / vecs) % seq;
    uint h = id / (vecs * seq);
    uint packed = s * (heads * hd) + h * hd + d4 * 4u;
    uint bhsd = (h * seq + s) * hd + d4 * 4u;
    float4 q = *reinterpret_cast<device const float4 *>(Q + packed);
    float4 k = *reinterpret_cast<device const float4 *>(K + packed);
    float4 v = *reinterpret_cast<device const float4 *>(V + packed);
    *reinterpret_cast<device half4 *>(Qh + bhsd) = half4(q);
    *reinterpret_cast<device half4 *>(Kh + bhsd) = half4(k);
    *reinterpret_cast<device half4 *>(Vh + bhsd) = half4(v);
}

inline uint2 frag_coord(uint lane) {
    uint qid = lane / 4u;
    uint row = (qid & 4u) + ((lane / 2u) % 4u);
    uint col = (qid & 2u) * 2u + (lane % 2u) * 2u;
    return uint2(col, row);
}

#define PV_D(ACC, S, KOFF, DOFF) \
    simdgroup_load(vmat, KVs + (KOFF) * LDV + (DOFF), LDV); \
    simdgroup_multiply_accumulate(ACC, S, vmat, ACC);

#define PV_K(S, KOFF, HD_) \
    PV_D(a0, S, KOFF, 0) \
    PV_D(a1, S, KOFF, 8) \
    PV_D(a2, S, KOFF, 16) \
    PV_D(a3, S, KOFF, 24) \
    if ((HD_) > 32) { \
        PV_D(a4, S, KOFF, 32) \
        PV_D(a5, S, KOFF, 40) \
        PV_D(a6, S, KOFF, 48) \
        PV_D(a7, S, KOFF, 56) \
    }

#define PREFILL_KERNEL(NAME, HD) \
kernel void NAME(device const half *Q [[buffer(0)]], \
                 device const half *K [[buffer(1)]], \
                 device const half *V [[buffer(2)]], \
                 device float *Y [[buffer(3)]], \
                 constant uint &seq [[buffer(4)]], \
                 constant uint &heads [[buffer(5)]], \
                 constant float &scale [[buffer(6)]], \
                 uint lid [[thread_index_in_threadgroup]], \
                 uint2 tgpig [[threadgroup_position_in_grid]], \
                 uint sgitg [[simdgroup_index_in_threadgroup]]) { \
    constexpr uint BR = 64; \
    constexpr uint BC = 32; \
    constexpr uint LDQ = HD + 8; \
    constexpr uint LDK = BC + 8; \
    constexpr uint LDV = HD + 8; \
    constexpr uint KV0 = LDK * HD; \
    constexpr uint KV1 = BC * LDV; \
    constexpr uint KVN = KV0 > KV1 ? KV0 : KV1; \
    constexpr uint Q_TCOLS = 4; \
    constexpr uint Q_NREADS = HD / Q_TCOLS; \
    constexpr uint KV_TCOLS = 8; \
    constexpr uint KV_NREADS = HD / KV_TCOLS; \
    const uint q0 = tgpig.x * BR; \
    const uint head = tgpig.y; \
    const uint dim = heads * HD; \
    const uint qbase = head * HD; \
    const uint head_off = head * seq * HD; \
    const uint row0 = sgitg * 8; \
    const uint lane = lid & 31u; \
    const uint2 coord = frag_coord(lane); \
    const float scale2 = scale * 1.4426950408889634f; \
    threadgroup half Qs[BR * LDQ]; \
    threadgroup half KVs[KVN]; \
    const uint qrow_l = lid / Q_TCOLS; \
    const uint qd0 = (lid % Q_TCOLS) * Q_NREADS; \
    const uint kvrow_l = lid / KV_TCOLS; \
    const uint kvd0 = (lid % KV_TCOLS) * KV_NREADS; \
    { \
        uint qrow = q0 + qrow_l; \
        threadgroup half *dst = Qs + qrow_l * LDQ + qd0; \
        if (qrow < seq) { \
            device const half *src = Q + head_off + qrow * HD + qd0; \
            _Pragma("clang loop unroll(full)") \
            for (uint j = 0; j < Q_NREADS; j += 4) { \
                half4 v = *reinterpret_cast<device const half4 *>(src + j); \
                dst[j + 0] = half(float(v.x) * scale2); \
                dst[j + 1] = half(float(v.y) * scale2); \
                dst[j + 2] = half(float(v.z) * scale2); \
                dst[j + 3] = half(float(v.w) * scale2); \
            } \
        } else { \
            _Pragma("clang loop unroll(full)") \
            for (uint j = 0; j < Q_NREADS; ++j) dst[j] = half(0.0f); \
        } \
    } \
    simdgroup_float8x8 a0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \
    simdgroup_float8x8 a1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \
    simdgroup_float8x8 a2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \
    simdgroup_float8x8 a3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \
    simdgroup_float8x8 a4 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \
    simdgroup_float8x8 a5 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \
    simdgroup_float8x8 a6 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \
    simdgroup_float8x8 a7 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \
    float m_i = -INFINITY; \
    float l_i = 0.0f; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    for (uint k0 = 0; k0 < seq; k0 += BC) { \
        { \
            uint kabs = k0 + kvrow_l; \
            if (kabs < seq) { \
                device const half *src = K + head_off + kabs * HD + kvd0; \
                _Pragma("clang loop unroll(full)") \
                for (uint j = 0; j < KV_NREADS; j += 4) { \
                    half4 v = *reinterpret_cast<device const half4 *>(src + j); \
                    KVs[(kvd0 + j + 0) * LDK + kvrow_l] = v.x; \
                    KVs[(kvd0 + j + 1) * LDK + kvrow_l] = v.y; \
                    KVs[(kvd0 + j + 2) * LDK + kvrow_l] = v.z; \
                    KVs[(kvd0 + j + 3) * LDK + kvrow_l] = v.w; \
                } \
            } else { \
                _Pragma("clang loop unroll(full)") \
                for (uint j = 0; j < KV_NREADS; ++j) { \
                    KVs[(kvd0 + j) * LDK + kvrow_l] = half(0.0f); \
                } \
            } \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        simdgroup_float8x8 s0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \
        simdgroup_float8x8 s1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \
        simdgroup_float8x8 s2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \
        simdgroup_float8x8 s3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \
        _Pragma("clang loop unroll(full)") \
        for (uint d = 0; d < HD; d += 8) { \
            simdgroup_half8x8 qmat; \
            simdgroup_load(qmat, Qs + row0 * LDQ + d, LDQ); \
            simdgroup_half8x8 kmat; \
            simdgroup_load(kmat, KVs + d * LDK + 0, LDK); \
            simdgroup_multiply_accumulate(s0, qmat, kmat, s0); \
            simdgroup_load(kmat, KVs + d * LDK + 8, LDK); \
            simdgroup_multiply_accumulate(s1, qmat, kmat, s1); \
            simdgroup_load(kmat, KVs + d * LDK + 16, LDK); \
            simdgroup_multiply_accumulate(s2, qmat, kmat, s2); \
            simdgroup_load(kmat, KVs + d * LDK + 24, LDK); \
            simdgroup_multiply_accumulate(s3, qmat, kmat, s3); \
        } \
        thread auto &e0 = s0.thread_elements(); \
        thread auto &e1 = s1.thread_elements(); \
        thread auto &e2 = s2.thread_elements(); \
        thread auto &e3 = s3.thread_elements(); \
        const uint k_lim = (k0 + BC <= seq) ? BC : (seq - k0); \
        const uint c0 = coord.x; \
        if (c0 + 0 >= k_lim) e0[0] = -INFINITY; \
        if (c0 + 1 >= k_lim) e0[1] = -INFINITY; \
        if (c0 + 8 >= k_lim) e1[0] = -INFINITY; \
        if (c0 + 9 >= k_lim) e1[1] = -INFINITY; \
        if (c0 + 16 >= k_lim) e2[0] = -INFINITY; \
        if (c0 + 17 >= k_lim) e2[1] = -INFINITY; \
        if (c0 + 24 >= k_lim) e3[0] = -INFINITY; \
        if (c0 + 25 >= k_lim) e3[1] = -INFINITY; \
        float tilemax = max(max(e0[0], e0[1]), max(e1[0], e1[1])); \
        tilemax = max(tilemax, max(max(e2[0], e2[1]), max(e3[0], e3[1]))); \
        tilemax = max(tilemax, simd_shuffle_xor(tilemax, 1)); \
        tilemax = max(tilemax, simd_shuffle_xor(tilemax, 8)); \
        float newm = max(m_i, tilemax); \
        float alpha = (newm == -INFINITY) ? 1.0f : fast::exp2(m_i - newm); \
        m_i = newm; \
        e0[0] = (e0[0] == -INFINITY) ? 0.0f : fast::exp2(e0[0] - m_i); \
        e0[1] = (e0[1] == -INFINITY) ? 0.0f : fast::exp2(e0[1] - m_i); \
        e1[0] = (e1[0] == -INFINITY) ? 0.0f : fast::exp2(e1[0] - m_i); \
        e1[1] = (e1[1] == -INFINITY) ? 0.0f : fast::exp2(e1[1] - m_i); \
        e2[0] = (e2[0] == -INFINITY) ? 0.0f : fast::exp2(e2[0] - m_i); \
        e2[1] = (e2[1] == -INFINITY) ? 0.0f : fast::exp2(e2[1] - m_i); \
        e3[0] = (e3[0] == -INFINITY) ? 0.0f : fast::exp2(e3[0] - m_i); \
        e3[1] = (e3[1] == -INFINITY) ? 0.0f : fast::exp2(e3[1] - m_i); \
        float lpart = e0[0] + e0[1] + e1[0] + e1[1] + e2[0] + e2[1] + e3[0] + e3[1]; \
        lpart += simd_shuffle_xor(lpart, 1); \
        lpart += simd_shuffle_xor(lpart, 8); \
        l_i = l_i * alpha + lpart; \
        { thread auto &ae = a0.thread_elements(); ae[0] *= alpha; ae[1] *= alpha; } \
        { thread auto &ae = a1.thread_elements(); ae[0] *= alpha; ae[1] *= alpha; } \
        { thread auto &ae = a2.thread_elements(); ae[0] *= alpha; ae[1] *= alpha; } \
        { thread auto &ae = a3.thread_elements(); ae[0] *= alpha; ae[1] *= alpha; } \
        if (HD > 32) { \
            { thread auto &ae = a4.thread_elements(); ae[0] *= alpha; ae[1] *= alpha; } \
            { thread auto &ae = a5.thread_elements(); ae[0] *= alpha; ae[1] *= alpha; } \
            { thread auto &ae = a6.thread_elements(); ae[0] *= alpha; ae[1] *= alpha; } \
            { thread auto &ae = a7.thread_elements(); ae[0] *= alpha; ae[1] *= alpha; } \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        { \
            uint vabs = k0 + kvrow_l; \
            threadgroup half *dst = KVs + kvrow_l * LDV + kvd0; \
            if (vabs < seq) { \
                device const half *src = V + head_off + vabs * HD + kvd0; \
                _Pragma("clang loop unroll(full)") \
                for (uint j = 0; j < KV_NREADS; j += 4) { \
                    half4 v = *reinterpret_cast<device const half4 *>(src + j); \
                    dst[j + 0] = v.x; \
                    dst[j + 1] = v.y; \
                    dst[j + 2] = v.z; \
                    dst[j + 3] = v.w; \
                } \
            } else { \
                _Pragma("clang loop unroll(full)") \
                for (uint j = 0; j < KV_NREADS; ++j) dst[j] = half(0.0f); \
            } \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        simdgroup_half8x8 vmat; \
        PV_K(s0, 0, HD) \
        PV_K(s1, 8, HD) \
        PV_K(s2, 16, HD) \
        PV_K(s3, 24, HD) \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
    const uint out_row = q0 + row0 + coord.y; \
    const uint out_col = coord.x; \
    const float inv = (l_i > 0.0f && m_i != -INFINITY) ? (1.0f / l_i) : 0.0f; \
    if (out_row < seq) { \
        device float *dst = Y + out_row * dim + qbase + out_col; \
        { thread auto &e = a0.thread_elements(); dst[0] = e[0] * inv; dst[1] = e[1] * inv; } \
        { thread auto &e = a1.thread_elements(); dst[8] = e[0] * inv; dst[9] = e[1] * inv; } \
        { thread auto &e = a2.thread_elements(); dst[16] = e[0] * inv; dst[17] = e[1] * inv; } \
        { thread auto &e = a3.thread_elements(); dst[24] = e[0] * inv; dst[25] = e[1] * inv; } \
        if (HD > 32) { \
            { thread auto &e = a4.thread_elements(); dst[32] = e[0] * inv; dst[33] = e[1] * inv; } \
            { thread auto &e = a5.thread_elements(); dst[40] = e[0] * inv; dst[41] = e[1] * inv; } \
            { thread auto &e = a6.thread_elements(); dst[48] = e[0] * inv; dst[49] = e[1] * inv; } \
            { thread auto &e = a7.thread_elements(); dst[56] = e[0] * inv; dst[57] = e[1] * inv; } \
        } \
    } \
}

PREFILL_KERNEL(jaiPrefill32, 32)
PREFILL_KERNEL(jaiPrefill64, 64)
