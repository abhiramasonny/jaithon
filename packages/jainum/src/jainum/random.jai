#: Counter-based random sampling: whole arrays in one dispatch, and the same
#: numbers whatever the machine does with the grid.
#:
#: Every generator here is Philox 4x32-10 — the Random123 one, verified
#: against its published vectors — keyed on `(seed, stream, call)` with the
#: element's own index as the counter. Sample `i` is therefore a pure
#: function of `i`: it does not move when the threadgroup width changes, when
#: the same count is asked for as `[1000]` instead of `[4, 250]`, or when
#: another array was drawn from the generator first. Nothing is carried from
#: one thread to the next, so there is no sequence to keep in order and no
#: state to write back.
#:
#: Why this and not the two generators already in the tree. `std.random`'s
#: `Random` is a host object: every draw is an interpreted xoshiro step and a
#: boxed float, and a million of them is seconds — which is the exact failure
#: this module exists to prevent, and the reason nothing below ever loops on
#: the host over samples. `std.gpu`'s `Buffer.fill_uniform` has the right
#: shape — one dispatch, one thread per value — but gives thread `id` the
#: state `seed + (id + 1) * K` and runs one xorshift64 over it. Two problems.
#: Neighbouring states differ by a constant, and a single three-shift round
#: does not decorrelate that. And xorshift64 is a GF(2)-linear map with no
#: output scramble, so every bit of the stream is a linear function of the
#: seed's bits — which is what MatrixRank and LinearComp in TestU01 are built
#: to catch, and why xoshiro puts a multiply on the way out. It is fine for
#: scratch data and not fine for anything whose moments are looked at. Philox
#: is the cheapest thing that is: ten rounds of two 32x32->64 multiplies, no
#: state, and it passes BigCrush.
#:
#: ```
#: let g = default_rng(7)
#: let x = g.standard_normal([1024, 1024])
#: let y = g.gamma(2.5, 1.0, [1000])
#: let picked = g.choice(1000, 10, replace: false)
#: ```
#:
#: The closed-form distributions run four samples per thread, because one
#: Philox call produces four 32-bit words and Box-Muller turns four words into
#: four normals. The ones with a rejection loop — gamma, and the two discrete
#: ones — run one sample per thread and spend counters privately: the
#: element's coordinates are folded into a key of its own first, so a thread
#: that retries twenty times cannot walk into its neighbour's numbers.
#:
#: A distribution's parameters are host scalars, not arrays. numpy broadcasts
#: them; doing that here would put a second strided read in the inner loop of
#: every sampler to serve a case that is rare. Two draws selected between by
#: a mask compose the same thing.
#:
#: Not here: reordering an array along an axis by sort keys, which is
#: `sorting.jai`'s `argsort` — `permutation`, `shuffle` and `permuted` all
#: call it rather than carrying a second sort network.
import std.random as host_random
from .array import FLOAT32, GROUP, INT32, MAX_EXACT_INT, NDArray, PRELUDE
from .array import copy_into, element_count, f32_bits, gather_flat, kernel_of
from .array import require_real
from .reduce import cumsum
from .sorting import argsort

#: Rounds of the Philox permutation. Ten is the published default and the
#: point at which the generator passes BigCrush; seven is the cheapest that
#: does, and the three-round margin costs six multiplies out of a dispatch
#: that is memory-bound anyway. The shader's loop bound is this number
#: written a second time, because a shader cannot read a Jaithon constant.
pub const PHILOX_ROUNDS = 10

#: Largest population `choice` and `permutation` will order. Beyond this the
#: bitonic network's dispatch count, not the sampling, is the bill — and the
#: INT32 index it hands back stops being exact at `MAX_EXACT_INT` regardless.
pub let MAX_POPULATION = MAX_EXACT_INT

#: Population below which the weighted search stages its CDF in threadgroup
#: memory. 1024 floats is 4 KB of the 32 KB a group may hold, and the tile has
#: to be a compile-time extent, so this number is in the shader too.
pub const TILED_CDF = 1024

#: Selectors for the two sampling kernels. Each has a twin in the shader
#: below for the same reason `PHILOX_ROUNDS` does, and the two lists have to
#: be changed together.
const _UNIFORM = 0
const _NORMAL = 1
const _LOGNORMAL = 2
const _EXPONENTIAL = 3
const _RAYLEIGH = 4
const _WEIBULL = 5
const _PARETO = 6
const _LOGISTIC = 7
const _LAPLACE = 8
const _INTEGERS = 9

const _GAMMA = 0
const _BETA = 1
const _CHISQUARE = 2
const _STUDENT_T = 3
const _POISSON = 4
const _BINOMIAL = 5

let SOURCE = PRELUDE + """

/* Philox 4x32-10, bit for bit the Random123 generator: this reproduces all
 * three of its published known-answer vectors, and `test_random.jai` pins the
 * first of them through the uniform kernel so a change here cannot quietly
 * become a different generator.
 *
 * The 128-bit counter carries (index, spare, stream, call) and the key is the
 * 64-bit seed, so two draws differ exactly when one of those words does. The
 * spare word stays zero: a dispatch is capped at 2^31 threads, so one word of
 * index is already more than an array can hold.
 *
 * The round is two 32x32->64 multiplies whose halves are crossed into the
 * other lanes. Written through `ulong` rather than `mulhi` because the same
 * source has to build wherever `std.gpu` builds, and `ulong` is what
 * `fill_uniform` already relies on. */
constant uint NR_M0 = 0xD2511F53u;
constant uint NR_M1 = 0xCD9E8D57u;
constant uint NR_W0 = 0x9E3779B9u;
constant uint NR_W1 = 0xBB67AE85u;

/* Threads' worth of CDF the weighted search stages in threadgroup memory.
 * Every thread of a group bisects the same table, so the overlap is total and
 * a 4 KB tile removes log2(n) device round trips per draw. */
constant uint NR_TILE = 1024u;

/* The two distribution selectors, twins of the Jaithon constants above. */
constant uint NR_UNIFORM = 0u;
constant uint NR_NORMAL = 1u;
constant uint NR_LOGNORMAL = 2u;
constant uint NR_EXPONENTIAL = 3u;
constant uint NR_RAYLEIGH = 4u;
constant uint NR_WEIBULL = 5u;
constant uint NR_PARETO = 6u;
constant uint NR_LOGISTIC = 7u;
constant uint NR_INTEGERS = 9u;

constant uint NR_GAMMA = 0u;
constant uint NR_BETA = 1u;
constant uint NR_CHISQUARE = 2u;
constant uint NR_STUDENT_T = 3u;
constant uint NR_POISSON = 4u;

/* log(k!) minus its Stirling estimate, for k below ten where the asymptotic
 * series has not converged. Both discrete samplers need this and neither MSL
 * nor Metal ships an `lgamma`. */
constant float nr_tail_table[10] = {
    0.08106147f, 0.04134070f, 0.02767793f, 0.02079067f, 0.01664469f,
    0.01387613f, 0.01189671f, 0.01041127f, 0.00925546f, 0.00833056f
};

inline uint4 nr_philox(uint4 ctr, uint2 key) {
    for (uint round = 0u; round < 10u; round++) {
        ulong wide0 = (ulong)NR_M0 * (ulong)ctr.x;
        ulong wide1 = (ulong)NR_M1 * (ulong)ctr.z;
        ctr = uint4(uint(wide1 >> 32) ^ ctr.y ^ key.x,
                    uint(wide1 & 0xFFFFFFFFul),
                    uint(wide0 >> 32) ^ ctr.w ^ key.y,
                    uint(wide0 & 0xFFFFFFFFul));
        key.x += NR_W0;
        key.y += NR_W1;
    }
    return ctr;
}

/* The top 24 bits of a word, which is exactly a float32 mantissa: every
 * representable multiple of 2^-24 in the interval is equally likely and no
 * value repeats twice as often as its neighbour, which is what taking the low
 * bits would do. */
inline float nr_unit(uint word) { return float(word >> 8) * 5.9604645e-8f; }

/* The same draw pushed half a step off both ends, for the transforms that
 * take a logarithm or a reciprocal of it. */
inline float nr_open(uint word) { return (float(word >> 8) + 0.5f) * 5.9604645e-8f; }

inline float4 nr_unit4(uint4 w) {
    return float4(nr_unit(w.x), nr_unit(w.y), nr_unit(w.z), nr_unit(w.w));
}

inline float4 nr_open4(uint4 w) {
    return float4(nr_open(w.x), nr_open(w.y), nr_open(w.z), nr_open(w.w));
}

/* Box-Muller: two words in, two independent normals out. The polar method
 * rejects a fifth of its draws and would put a loop where there is none;
 * a sine and a cosine are cheaper than divergence. */
inline float2 nr_gauss(uint a, uint b) {
    float radius = sqrt(-2.0f * log(nr_open(a)));
    float angle = 6.28318531f * nr_unit(b);
    return float2(radius * cos(angle), radius * sin(angle));
}

inline float4 nr_gauss4(uint4 w) {
    float2 low = nr_gauss(w.x, w.y);
    float2 high = nr_gauss(w.z, w.w);
    return float4(low.x, low.y, high.x, high.y);
}

/* A private stream for one output element, for the samplers whose draw count
 * is not known in advance.
 *
 * The element's coordinates are folded through one Philox call into a key,
 * and the counter then starts at zero. A rejection loop can spend 2^32 blocks
 * without reaching what the next element was given, which a shared counter
 * incremented per element could not promise. */
struct nr_state {
    uint2 key;
    uint4 ctr;
    uint4 buf;
    uint pos;
};

inline nr_state nr_begin(uint2 key, uint stream, uint call, uint index) {
    uint4 folded = nr_philox(uint4(index, stream, call, 0x243F6A88u), key);
    nr_state s;
    s.key = uint2(folded.x, folded.y);
    s.ctr = uint4(0u, folded.z, folded.w, 0u);
    s.buf = uint4(0u, 0u, 0u, 0u);
    s.pos = 4u;
    return s;
}

/* A chain of comparisons and not `s.buf[at]`: a dynamic index into a vector
 * forces the whole block to memory, and this is the innermost call of every
 * rejection loop below. */
inline uint nr_word(thread nr_state& s) {
    if (s.pos >= 4u) {
        s.buf = nr_philox(s.ctr, s.key);
        s.ctr.x += 1u;
        s.pos = 0u;
    }
    uint at = s.pos;
    s.pos = at + 1u;
    if (at == 0u) return s.buf.x;
    if (at == 1u) return s.buf.y;
    if (at == 2u) return s.buf.z;
    return s.buf.w;
}

inline float nr_next_unit(thread nr_state& s) { return nr_unit(nr_word(s)); }

inline float nr_next_open(thread nr_state& s) { return nr_open(nr_word(s)); }

/* One normal costs two words and throws the second of the pair away. Every
 * caller is inside a rejection loop that already dwarfs a Philox block. */
inline float nr_next_normal(thread nr_state& s) {
    uint first = nr_word(s);
    return nr_gauss(first, nr_word(s)).x;
}

inline float nr_tail(float k) {
    if (k < 10.0f) return nr_tail_table[int(clamp(k, 0.0f, 9.0f))];
    float square = (k + 1.0f) * (k + 1.0f);
    return (0.083333333f - (0.0027777778f - 0.00079365079f / square) / square) / (k + 1.0f);
}

/* log(k!) for a whole k >= 0, as the Stirling series in (k+1) plus the table
 * above. Exact to a float32 ulp at k = 0 and k = 1, which the two rejection
 * bounds lean on hardest. */
inline float nr_log_factorial(float k) {
    return (k + 0.5f) * log(k + 1.0f) - (k + 1.0f) + 0.91893853f + nr_tail(k);
}

/* Marsaglia-Tsang. Accepts on the squeeze more than 98% of the time, so the
 * loop is one iteration in the overwhelming majority of threads and the
 * divergence it costs a simdgroup is small.
 *
 * A shape below one has no mode to squeeze against, so it is drawn at
 * shape+1 and pulled back by U^(1/shape) -- Marsaglia's own boost, and the
 * reason there is only one loop here rather than two algorithms.
 *
 * The 128 cap is not a correctness knob: at a 0.98 acceptance rate reaching
 * it has probability around 1e-220. It is there so a shape that arrives as a
 * NaN cannot hang a dispatch. */
inline float nr_gamma(thread nr_state& s, float shape) {
    if (!(shape > 0.0f)) return 0.0f;
    float boost = 1.0f;
    float a = shape;
    if (a < 1.0f) {
        boost = pow(nr_next_open(s), 1.0f / a);
        a += 1.0f;
    }
    float d = a - 0.333333333f;
    float c = 1.0f / sqrt(9.0f * d);
    for (uint tries = 0u; tries < 128u; tries++) {
        float x = nr_next_normal(s);
        float v = 1.0f + c * x;
        if (v <= 0.0f) continue;
        v = v * v * v;
        float u = nr_next_open(s);
        float xx = x * x;
        if (u < 1.0f - 0.0331f * xx * xx) return boost * d * v;
        if (log(u) < 0.5f * xx + d * (1.0f - v + log(v))) return boost * d * v;
    }
    return boost * d;
}

/* Knuth's product of uniforms, which needs rate+1 draws on average. That is
 * why it stops at ten and the transformed rejection takes over. */
inline float nr_poisson_small(thread nr_state& s, float rate) {
    float limit = exp(-rate);
    float product = nr_next_open(s);
    float k = 0.0f;
    while (product > limit && k < 512.0f) {
        product *= nr_next_open(s);
        k += 1.0f;
    }
    return k;
}

/* Hoermann's PTRS. Constant expected work in the rate, which the product
 * above is not, and the crossover is where its setup -- a log, a sqrt and
 * two divides -- stops costing more than the draws it saves. */
inline float nr_poisson_large(thread nr_state& s, float rate) {
    float log_rate = log(rate);
    float b = 0.931f + 2.53f * sqrt(rate);
    float a = -0.059f + 0.02483f * b;
    float inv_alpha = 1.1239f + 1.1328f / (b - 3.4f);
    float v_r = 0.9277f - 3.6224f / (b - 2.0f);
    for (uint tries = 0u; tries < 256u; tries++) {
        float u = nr_next_unit(s) - 0.5f;
        float v = nr_next_open(s);
        float us = 0.5f - fabs(u);
        if (us <= 0.0f) continue;
        float k = floor((2.0f * a / us + b) * u + rate + 0.43f);
        if (us >= 0.07f && v <= v_r) return k;
        if (k < 0.0f) continue;
        if (us < 0.013f && v > us) continue;
        float log_v = log(v * inv_alpha / (a / (us * us) + b));
        if (log_v <= -rate + k * log_rate - nr_log_factorial(k)) return k;
    }
    return floor(rate);
}

inline float nr_poisson(thread nr_state& s, float rate) {
    if (!(rate > 0.0f)) return 0.0f;
    return rate < 10.0f ? nr_poisson_small(s, rate) : nr_poisson_large(s, rate);
}

/* Inversion, walking the pmf up from k = 0. Expected n*p+1 steps, which is
 * why the caller only sends it work below n*p = 10.
 *
 * The series and not `log(1 - p)`: at p below 6e-8, which n*p < 10 reaches
 * for any n past 1.6e8, one minus p rounds to one in float32 and the whole
 * distribution collapses onto zero. MSL ships no `log1p`, so `nr_log1m` is
 * the four-term expansion below a hundredth and the library call above it,
 * where the two agree to better than a float32 ulp. */
inline float nr_log1m(float p) {
    if (p < 0.01f) return -p * (1.0f + p * (0.5f + p * (0.333333333f + 0.25f * p)));
    return log(1.0f - p);
}

inline float nr_binomial_small(thread nr_state& s, float n, float p) {
    float ratio = p / (1.0f - p);
    float term = exp(n * nr_log1m(p));
    float u = nr_next_unit(s);
    float k = 0.0f;
    while (u > term && k < n) {
        u -= term;
        k += 1.0f;
        term *= ratio * (n - k + 1.0f) / k;
    }
    return k;
}

/* Hoermann's BTRS, the binomial twin of PTRS: a scaled Cauchy proposal, a
 * cheap squeeze for the body, and a log-pmf bound built out of Stirling
 * tails for the rest. Constant expected work in n. */
inline float nr_binomial_large(thread nr_state& s, float n, float p) {
    float spq = sqrt(n * p * (1.0f - p));
    float b = 1.15f + 2.53f * spq;
    float a = -0.0873f + 0.0248f * b + 0.01f * p;
    float c = n * p + 0.5f;
    float v_r = 0.92f - 4.2f / b;
    float alpha = (2.83f + 5.1f / b) * spq;
    float ratio = p / (1.0f - p);
    float m = floor((n + 1.0f) * p);
    float anchor = (m + 0.5f) * log((m + 1.0f) / (ratio * (n - m + 1.0f)))
                 + nr_tail(m) + nr_tail(n - m);
    for (uint tries = 0u; tries < 256u; tries++) {
        float u = nr_next_unit(s) - 0.5f;
        float v = nr_next_open(s);
        float us = 0.5f - fabs(u);
        if (us <= 0.0f) continue;
        float k = floor((2.0f * a / us + b) * u + c);
        if (k < 0.0f || k > n) continue;
        if (us >= 0.07f && v <= v_r) return k;
        float log_v = log(v * alpha / (a / (us * us) + b));
        float bound = anchor
                    + (n + 1.0f) * log((n - m + 1.0f) / (n - k + 1.0f))
                    + (k + 0.5f) * log(ratio * (n - k + 1.0f) / (k + 1.0f))
                    - nr_tail(k) - nr_tail(n - k);
        if (log_v <= bound) return k;
    }
    return m;
}

/* Reflected through p -> 1-p so the two samplers only ever see the short
 * tail, which is what keeps BINV's step count bounded and BTRS's proposal
 * centred. */
inline float nr_binomial(thread nr_state& s, float n, float p) {
    if (!(n > 0.0f) || !(p > 0.0f)) return 0.0f;
    if (p >= 1.0f) return n;
    float q = min(p, 1.0f - p);
    float draw = n * q < 10.0f ? nr_binomial_small(s, n, q) : nr_binomial_large(s, n, q);
    return p > 0.5f ? n - draw : draw;
}

/* The closed-form distributions, four samples from one Philox block.
 *
 * Both unit forms are computed for every code. That is four shifts and four
 * multiply-adds spent unconditionally to keep the selector out of the vector
 * path; the alternative is a branch per lane in a kernel that is already
 * bound by the store. */
inline float4 nr_closed(uint4 w, uint dist, uint p0, uint p1) {
    float a = as_type<float>(p0);
    float b = as_type<float>(p1);
    if (dist == NR_INTEGERS) {
        /* Multiply-shift rather than a modulo: the device has no integer
         * divide, and the residual bias is span/2^32, which for any span an
         * exact float32 can name is below one part in 256. */
        int low = as_type<int>(p0);
        ulong span = (ulong)p1;
        return float4(
            float(low + int((ulong(w.x) * span) >> 32)),
            float(low + int((ulong(w.y) * span) >> 32)),
            float(low + int((ulong(w.z) * span) >> 32)),
            float(low + int((ulong(w.w) * span) >> 32)));
    }
    if (dist == NR_NORMAL) return a + b * nr_gauss4(w);
    if (dist == NR_LOGNORMAL) return exp(a + b * nr_gauss4(w));
    float4 unit = nr_unit4(w);
    float4 open = nr_open4(w);
    if (dist == NR_UNIFORM) return a + (b - a) * unit;
    if (dist == NR_EXPONENTIAL) return -a * log(open);
    if (dist == NR_RAYLEIGH) return a * sqrt(-2.0f * log(open));
    if (dist == NR_WEIBULL) return pow(-log(open), float4(1.0f / a));
    if (dist == NR_PARETO) return pow(open, float4(-1.0f / a)) - 1.0f;
    if (dist == NR_LOGISTIC) return a + b * log(open / (1.0f - open));
    /* Laplace, folded about the median so the two branches numpy writes
     * become one sign and one logarithm. */
    float4 side = open - 0.5f;
    return a - b * sign(side) * log(1.0f - 2.0f * fabs(side));
}

kernel void nrand_closed(device float* dst [[buffer(0)]],
                         constant uint& count [[buffer(1)]],
                         constant uint& quads [[buffer(2)]],
                         constant uint& dist [[buffer(3)]],
                         constant uint& seed_lo [[buffer(4)]],
                         constant uint& seed_hi [[buffer(5)]],
                         constant uint& stream [[buffer(6)]],
                         constant uint& call [[buffer(7)]],
                         constant uint& p0 [[buffer(8)]],
                         constant uint& p1 [[buffer(9)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= quads) return;
    uint4 w = nr_philox(uint4(id, 0u, stream, call), uint2(seed_lo, seed_hi));
    float4 value = nr_closed(w, dist, p0, p1);
    uint base = id * 4u;
    /* The destination is always a fresh allocation at slot zero, so a quad
     * boundary is a 16-byte boundary and the wide store is legal. Only the
     * last quad of a count that does not divide by four goes scalar. */
    if (base + 4u <= count) {
        *((device float4*)(dst + base)) = value;
        return;
    }
    if (base < count) dst[base] = value.x;
    if (base + 1u < count) dst[base + 1u] = value.y;
    if (base + 2u < count) dst[base + 2u] = value.z;
    if (base + 3u < count) dst[base + 3u] = value.w;
}

/* One sample per thread for the samplers that draw until they accept. */
kernel void nrand_reject(device float* dst [[buffer(0)]],
                         constant uint& count [[buffer(1)]],
                         constant uint& dist [[buffer(2)]],
                         constant uint& seed_lo [[buffer(3)]],
                         constant uint& seed_hi [[buffer(4)]],
                         constant uint& stream [[buffer(5)]],
                         constant uint& call [[buffer(6)]],
                         constant uint& p0 [[buffer(7)]],
                         constant uint& p1 [[buffer(8)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    nr_state s = nr_begin(uint2(seed_lo, seed_hi), stream, call, id);
    float a = as_type<float>(p0);
    float b = as_type<float>(p1);
    float value = 0.0f;
    if (dist == NR_GAMMA) {
        value = b * nr_gamma(s, a);
    } else if (dist == NR_BETA) {
        float x = nr_gamma(s, a);
        float y = nr_gamma(s, b);
        float total = x + y;
        value = total > 0.0f ? x / total : 0.5f;
    } else if (dist == NR_CHISQUARE) {
        value = 2.0f * nr_gamma(s, 0.5f * a);
    } else if (dist == NR_STUDENT_T) {
        float z = nr_next_normal(s);
        value = z / sqrt(2.0f * nr_gamma(s, 0.5f * a) / a);
    } else if (dist == NR_POISSON) {
        value = nr_poisson(s, a);
    } else {
        value = nr_binomial(s, a, b);
    }
    dst[id] = value;
}

/* One row of counts per thread, by the conditional-binomial factorisation:
 * category j is binomial over what is left of the trials with what is left of
 * the mass. Sequential in the categories, which is the whole point -- the k
 * counts of a row are not independent and cannot be drawn in parallel.
 *
 * The row total is summed here rather than passed in so that the host never
 * has to read the weights back to normalise them. */
kernel void nrand_multinomial(device float* dst [[buffer(0)]],
                              device const float* probs [[buffer(1)]],
                              constant uint& rows [[buffer(2)]],
                              constant uint& classes [[buffer(3)]],
                              constant uint& trials [[buffer(4)]],
                              constant uint& seed_lo [[buffer(5)]],
                              constant uint& seed_hi [[buffer(6)]],
                              constant uint& stream [[buffer(7)]],
                              constant uint& call [[buffer(8)]],
                              uint id [[thread_position_in_grid]]) {
    if (id >= rows) return;
    nr_state s = nr_begin(uint2(seed_lo, seed_hi), stream, call, id);
    float left = 0.0f;
    for (uint j = 0u; j < classes; j++) left += probs[j];
    float remaining = float(trials);
    uint base = id * classes;
    for (uint j = 0u; j + 1u < classes; j++) {
        float share = probs[j];
        float take = 0.0f;
        if (remaining > 0.0f && left > 0.0f) {
            take = nr_binomial(s, remaining, min(share / left, 1.0f));
        }
        dst[base + j] = take;
        remaining -= take;
        left -= share;
    }
    if (classes > 0u) dst[base + classes - 1u] = remaining;
}

/* Weighted choice: bisect a cumulative sum for the first entry at or above
 * a uniform draw scaled by the total, which is the inverse-CDF method with
 * the normalisation folded into the scale. */
inline uint nr_bisect_device(device const float* cdf, uint n, float target) {
    uint low = 0u;
    uint high = n - 1u;
    while (low < high) {
        uint mid = (low + high) >> 1;
        if (cdf[mid] < target) low = mid + 1u; else high = mid;
    }
    return low;
}

kernel void nrand_search(device float* dst [[buffer(0)]],
                         device const float* cdf [[buffer(1)]],
                         constant uint& count [[buffer(2)]],
                         constant uint& n [[buffer(3)]],
                         constant uint& seed_lo [[buffer(4)]],
                         constant uint& seed_hi [[buffer(5)]],
                         constant uint& stream [[buffer(6)]],
                         constant uint& call [[buffer(7)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint4 w = nr_philox(uint4(id, 0u, stream, call), uint2(seed_lo, seed_hi));
    dst[id] = float(nr_bisect_device(cdf, n, nr_unit(w.x) * cdf[n - 1u]));
}

/* The same search with the table staged in threadgroup memory. Every thread
 * of the group walks the same log2(n) entries of it, so the tile is read
 * 256 * log2(n) times for the one coalesced pass that filled it. */
kernel void nrand_search_tiled(device float* dst [[buffer(0)]],
                               device const float* cdf [[buffer(1)]],
                               constant uint& count [[buffer(2)]],
                               constant uint& n [[buffer(3)]],
                               constant uint& seed_lo [[buffer(4)]],
                               constant uint& seed_hi [[buffer(5)]],
                               constant uint& stream [[buffer(6)]],
                               constant uint& call [[buffer(7)]],
                               uint id [[thread_position_in_grid]],
                               uint lid [[thread_index_in_threadgroup]]) {
    threadgroup float table[NR_TILE];
    for (uint at = lid; at < n; at += ND_GROUP) table[at] = cdf[at];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (id >= count) return;
    uint4 w = nr_philox(uint4(id, 0u, stream, call), uint2(seed_lo, seed_hi));
    float target = nr_unit(w.x) * table[n - 1u];
    uint low = 0u;
    uint high = n - 1u;
    while (low < high) {
        uint mid = (low + high) >> 1;
        if (table[mid] < target) low = mid + 1u; else high = mid;
    }
    dst[id] = float(low);
}

/* Efraimidis-Spirakis keys: -log(u) / w. The `size` smallest of these are a
 * weighted sample without replacement, which is why the weighted and the
 * unweighted draw are the same one sort and differ only by this kernel.
 *
 * A zero weight gets a key no finite draw can reach, so it sorts last and is
 * never taken -- the sampling equivalent of excluding it. */
kernel void nrand_reservoir(device float* dst [[buffer(0)]],
                            device const float* weight [[buffer(1)]],
                            constant uint& count [[buffer(2)]],
                            constant uint& seed_lo [[buffer(3)]],
                            constant uint& seed_hi [[buffer(4)]],
                            constant uint& stream [[buffer(5)]],
                            constant uint& call [[buffer(6)]],
                            uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint4 w = nr_philox(uint4(id, 0u, stream, call), uint2(seed_lo, seed_hi));
    float mass = weight[id];
    dst[id] = mass > 0.0f ? -log(nr_open(w.x)) / mass : 3.0e38f;
}

/* dst[row] = src[index[row]], whole rows. One division to find the row and a
 * multiply-subtract for the column, because the device has no integer divide
 * and two of them would cost more than the copy. */
kernel void nrand_rows(device float* dst [[buffer(0)]],
                       device const float* src [[buffer(1)]],
                       device const float* index [[buffer(2)]],
                       constant uint& count [[buffer(3)]],
                       constant uint& width [[buffer(4)]],
                       uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint row = id / width;
    uint col = id - row * width;
    dst[id] = src[uint(index[row]) * width + col];
}

/* The same gather where the row length divides by four. Rows are contiguous
 * and both sides start at slot zero, so a row boundary is a quad boundary. */
kernel void nrand_rows4(device float4* dst [[buffer(0)]],
                        device const float4* src [[buffer(1)]],
                        device const float* index [[buffer(2)]],
                        constant uint& count [[buffer(3)]],
                        constant uint& quads [[buffer(4)]],
                        uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint row = id / quads;
    uint col = id - row * quads;
    dst[id] = src[uint(index[row]) * quads + col];
}

/* dst[row, i] = src[row, order[row, i]] -- one independent permutation per
 * row, which is what `permuted` asks of `argsort`'s output. */
kernel void nrand_take_last(device float* dst [[buffer(0)]],
                            device const float* src [[buffer(1)]],
                            device const float* order [[buffer(2)]],
                            constant uint& count [[buffer(3)]],
                            constant uint& width [[buffer(4)]],
                            uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint row = id / width;
    dst[id] = src[row * width + uint(order[id])];
}
"""

#: A counter-based generator: `numpy.random.Generator`, with the counter
#: exposed as a stream rather than hidden in a state vector.
#:
#: Three numbers key every draw. `seed` is the 64-bit Philox key. `stream` is
#: a 32-bit label that selects an independent sequence — two generators with
#: the same seed and different streams share nothing, which is how a worker
#: gets its own randomness without a hand-off. `call` is a counter this object
#: bumps once per draw, so consecutive calls do not repeat; it is the only
#: mutable state, and it is deliberately not part of what "reproducible"
#: means below.
#:
#: What is reproducible: given a seed and a stream, the n-th draw's element i
#: is a fixed number. It does not depend on the shape the draw was asked for,
#: the threadgroup width the device chose, how many elements the previous
#: draws had, or how many other generators exist. `spawn` gives a fresh stream
#: and `Generator(seed, stream)` names one directly.
pub class Generator {
    #: The 64-bit Philox key.
    pub let seed: int
    #: The 32-bit stream label. Independent sequences, not a re-seeding.
    pub let stream: int
    var calls: int

    pub fn init(self, seed: int = 0, stream: int = 0) {
        if stream < 0 or stream > 0xFFFFFFFF {
            throw ValueError(f"a stream is a 32-bit label, got {stream}")
        }
        self.seed = seed
        self.stream = stream
        self.calls = 0
    }

    #: A generator on the same seed and a different stream.
    #:
    #: Independent, and reproducible from the pair alone: `spawn(3)` is the
    #: same sequence whether it was asked for first or thousandth.
    pub fn spawn(self, stream: int) -> Generator { return Generator(self.seed, stream) }

    #: Draws made so far. The call counter that separates them wraps at 2^32,
    #: which is four billion dispatches from this object.
    pub fn draws(self) -> int { return self.calls }

    #: Uniform in `[0, 1)`.
    pub fn random(self, size: any = null) -> NDArray {
        return self._closed(size, _UNIFORM, f32_bits(0.0), f32_bits(1.0), FLOAT32)
    }

    #: Uniform in `[low, high)`.
    pub fn uniform(self, low: float = 0.0, high: float = 1.0, size: any = null) -> NDArray {
        return self._closed(size, _UNIFORM, f32_bits(low), f32_bits(high), FLOAT32)
    }

    #: Standard normal, mean 0 and variance 1.
    pub fn standard_normal(self, size: any = null) -> NDArray {
        return self._closed(size, _NORMAL, f32_bits(0.0), f32_bits(1.0), FLOAT32)
    }

    #: Normal with mean `loc` and standard deviation `scale`.
    pub fn normal(self, loc: float = 0.0, scale: float = 1.0, size: any = null) -> NDArray {
        if scale < 0.0 { throw ValueError(f"a scale cannot be negative, got {scale}") }
        return self._closed(size, _NORMAL, f32_bits(loc), f32_bits(scale), FLOAT32)
    }

    #: Whole numbers in `[low, high)`, or in `[0, low)` when `high` is null.
    #:
    #: The bound is checked against `MAX_EXACT_INT` rather than against 2^31:
    #: storage is float32, so an integer past that point is not a number this
    #: array can hold, and a bound that lets one through is a bound that lies.
    pub fn integers(
        self,
        low: int,
        high: int? = null,
        size: any = null,
        endpoint: bool = false
    ) -> NDArray {
        var first = low
        var last = high ?? low
        if high is null { first = 0 }
        if endpoint { last += 1 }
        if last <= first { throw ValueError(f"integers needs low < high, got [{first}, {last})") }
        if first < -MAX_EXACT_INT or last > MAX_EXACT_INT {
            throw ValueError(f"[{first}, {last}) leaves what float32 stores exactly")
        }
        return self._closed(size, _INTEGERS, first & 0xFFFFFFFF, last - first, INT32)
    }

    #: `integers`, under the name the legacy `RandomState` uses.
    pub fn randint(self, low: int, high: int? = null, size: any = null) -> NDArray {
        return self.integers(low, high, size)
    }

    #: Exponential with mean `scale`.
    pub fn exponential(self, scale: float = 1.0, size: any = null) -> NDArray {
        _require_positive(scale, "scale")
        return self._closed(size, _EXPONENTIAL, f32_bits(scale), 0, FLOAT32)
    }

    #: Rayleigh with mode `scale`; the length of a 2-d standard normal scaled
    #: by it.
    pub fn rayleigh(self, scale: float = 1.0, size: any = null) -> NDArray {
        _require_positive(scale, "scale")
        return self._closed(size, _RAYLEIGH, f32_bits(scale), 0, FLOAT32)
    }

    #: Weibull with shape `a` and unit scale.
    pub fn weibull(self, a: float, size: any = null) -> NDArray {
        _require_positive(a, "a")
        return self._closed(size, _WEIBULL, f32_bits(a), 0, FLOAT32)
    }

    #: Pareto II (Lomax) with shape `a`, which is numpy's `pareto`: the
    #: classical Pareto shifted to start at zero rather than at one.
    pub fn pareto(self, a: float, size: any = null) -> NDArray {
        _require_positive(a, "a")
        return self._closed(size, _PARETO, f32_bits(a), 0, FLOAT32)
    }

    #: Logistic with location `loc` and scale `scale`.
    pub fn logistic(self, loc: float = 0.0, scale: float = 1.0, size: any = null) -> NDArray {
        _require_positive(scale, "scale")
        return self._closed(size, _LOGISTIC, f32_bits(loc), f32_bits(scale), FLOAT32)
    }

    #: Laplace with location `loc` and scale `scale`.
    pub fn laplace(self, loc: float = 0.0, scale: float = 1.0, size: any = null) -> NDArray {
        _require_positive(scale, "scale")
        return self._closed(size, _LAPLACE, f32_bits(loc), f32_bits(scale), FLOAT32)
    }

    #: Log-normal: `exp(normal(mean, sigma))`, so `mean` and `sigma` describe
    #: the logarithm and not the sample.
    pub fn lognormal(self, mean: float = 0.0, sigma: float = 1.0, size: any = null) -> NDArray {
        _require_positive(sigma, "sigma")
        return self._closed(size, _LOGNORMAL, f32_bits(mean), f32_bits(sigma), FLOAT32)
    }

    #: Gamma with shape `shape` and scale `scale`, mean `shape * scale`.
    pub fn gamma(self, shape: float, scale: float = 1.0, size: any = null) -> NDArray {
        _require_positive(shape, "shape")
        _require_positive(scale, "scale")
        return self._reject(size, _GAMMA, f32_bits(shape), f32_bits(scale))
    }

    #: Beta on `[0, 1]`, as the ratio of two gammas.
    pub fn beta(self, a: float, b: float, size: any = null) -> NDArray {
        _require_positive(a, "a")
        _require_positive(b, "b")
        return self._reject(size, _BETA, f32_bits(a), f32_bits(b))
    }

    #: Chi-square with `df` degrees of freedom, mean `df`.
    pub fn chisquare(self, df: float, size: any = null) -> NDArray {
        _require_positive(df, "df")
        return self._reject(size, _CHISQUARE, f32_bits(df), 0)
    }

    #: Student's t with `df` degrees of freedom.
    pub fn standard_t(self, df: float, size: any = null) -> NDArray {
        _require_positive(df, "df")
        return self._reject(size, _STUDENT_T, f32_bits(df), 0)
    }

    #: Poisson counts with mean `lam`.
    pub fn poisson(self, lam: float = 1.0, size: any = null) -> NDArray {
        if lam < 0.0 { throw ValueError(f"a rate cannot be negative, got {lam}") }
        return self._reject(size, _POISSON, f32_bits(lam), 0, INT32)
    }

    #: Binomial counts: successes in `n` trials of probability `p`.
    pub fn binomial(self, n: int, p: float, size: any = null) -> NDArray {
        _require_countable(n)
        if p < 0.0 or p > 1.0 { throw ValueError(f"a probability lies in [0, 1], got {p}") }
        return self._reject(size, _BINOMIAL, f32_bits(float(n)), f32_bits(p), INT32)
    }

    #: Counts over `pvals` categories from `n` trials, one row per draw.
    #:
    #: The last axis of the result is the categories, so `size` names the rows
    #: and the shape is `size + [k]`. `pvals` is a list or a 1-d array and need
    #: not sum to one; the kernel divides by its own total, which is what
    #: keeps the weights on the device.
    pub fn multinomial(self, n: int, pvals: any, size: any = null) -> NDArray {
        _require_countable(n)
        let (weights, owned) = _weight_array(pvals, "pvals")
        defer { if owned { weights.free() } }
        let classes = weights.size()
        if classes == 0 { throw ValueError("multinomial needs at least one category") }
        var shape = _shape_of(size)
        let rows = element_count(shape)
        shape.push(classes)
        let out = NDArray(shape, INT32)
        if rows == 0 { return out }
        let call = self._advance()
        kernel_of(SOURCE, "nrand_multinomial").dispatch_async(
            [out.storage, weights.storage],
            rows,
            [rows, classes, n, self._key(0), self._key(32), self.stream, call]
        )
        return out
    }

    #: Draws from `0..n` when `a` is an int, or from the 1-d array `a`.
    #:
    #: `replace: false` orders the whole population once and takes the front
    #: of it, which is `argsort` on a key array — the same trick with the keys
    #: weighted by `-log(u)/w` when `p` is given, which is the
    #: Efraimidis-Spirakis reservoir. Both are one sort, not `size` passes.
    pub fn choice(self, a: any, size: any = null, replace: bool = true, p: any = null) -> NDArray {
        let population = _population_of(a)
        if population <= 0 { throw ValueError("choice needs a non-empty population") }
        if population > MAX_POPULATION {
            throw ValueError(f"{population} is past what an exact index can name")
        }
        let shape = _shape_of(size)
        let wanted = element_count(shape)
        if not replace and wanted > population {
            throw ValueError(f"{wanted} draws without replacement from {population}")
        }
        let indices = self._indices(population, shape, wanted, replace, p)
        if a is int { return indices }
        defer { indices.free() }
        return gather_flat(a, indices)
    }

    #: A shuffled copy: `0..n` when `a` is an int, or `a` reordered along its
    #: first axis.
    pub fn permutation(self, a: any) -> NDArray {
        if a is int { return self._order(a) }
        if not a is NDArray {
            throw TypeError(f"permutation takes an int or an array, not {type_of(a)}")
        }
        require_real(a, "permutation")
        if a.ndim() == 0 { throw ValueError("a 0-d array has no axis to permute") }
        let order = self._order(a.shape[0])
        defer { order.free() }
        return _take_rows(a, order)
    }

    #: Reorder `a` along its first axis in place.
    pub fn shuffle(self, a: NDArray) -> void {
        let shuffled = self.permutation(a)
        defer { shuffled.free() }
        copy_into(a, shuffled)
    }

    #: A copy with every 1-d slice along `axis` shuffled independently.
    #:
    #: Distinct from `shuffle`, which moves whole rows and keeps each row's
    #: contents together. One `argsort` over a key array of the same shape
    #: does every slice at once.
    pub fn permuted(self, a: NDArray, axis: int = -1) -> NDArray {
        require_real(a, "permuted")
        if a.ndim() == 0 { throw ValueError("a 0-d array has no axis to permute") }
        let at = if axis < 0 { axis + a.ndim() } else { axis }
        if at < 0 or at >= a.ndim() {
            throw ValueError(f"axis {axis} is out of bounds for {a.ndim()} dimensions")
        }
        let out = NDArray(a.shape, a.dtype)
        let width = a.shape[at]
        if a.size() == 0 or width <= 1 {
            copy_into(out, a)
            return out
        }
        let moved = _axis_last(a, at)
        let (work, owned) = _dense(moved)
        defer { if owned { work.free() } }
        let keys = self.random(moved.shape)
        defer { keys.free() }
        let order = argsort(keys, -1)
        defer { order.free() }

        #: When the axis is already last the destination is dense in the same
        #: order the kernel writes, so the permutation lands in `out` directly
        #: and the whole call is one gather. Any other axis pays a strided
        #: copy back through the moved view.
        let direct = at == a.ndim() - 1
        let mixed = if direct { out } else { NDArray(moved.shape, a.dtype) }
        defer { if not direct { mixed.free() } }
        let count = work.size()
        kernel_of(SOURCE, "nrand_take_last").dispatch_async(
            [mixed.storage, work.storage, order.storage],
            count,
            [count, width]
        )
        if not direct { copy_into(_axis_last(out, at), mixed) }
        return out
    }

    #: `random`, under the name the legacy `RandomState` uses.
    pub fn rand(self, shape: list[int] = []) -> NDArray { return self.random(shape) }

    #: `standard_normal`, under the name the legacy `RandomState` uses.
    pub fn randn(self, shape: list[int] = []) -> NDArray { return self.standard_normal(shape) }

    fn _key(self, shift: int) -> int { return self.seed >> shift & 0xFFFFFFFF }

    #: The ordinal of this draw, and the bump that keeps the next one apart.
    fn _advance(self) -> int {
        let at = self.calls & 0xFFFFFFFF
        self.calls += 1
        return at
    }

    fn _closed(self, size: any, dist: int, p0: int, p1: int, dtype: int) -> NDArray {
        let out = NDArray(_shape_of(size), dtype)
        let count = out.size()
        if count == 0 { return out }
        let quads = (count + 3) // 4
        let call = self._advance()
        kernel_of(SOURCE, "nrand_closed").dispatch_async(
            [out.storage],
            quads,
            [count, quads, dist, self._key(0), self._key(32), self.stream, call, p0, p1]
        )
        return out
    }

    fn _reject(self, size: any, dist: int, p0: int, p1: int, dtype: int = FLOAT32) -> NDArray {
        let out = NDArray(_shape_of(size), dtype)
        let count = out.size()
        if count == 0 { return out }
        let call = self._advance()
        kernel_of(SOURCE, "nrand_reject").dispatch_async(
            [out.storage],
            count,
            [count, dist, self._key(0), self._key(32), self.stream, call, p0, p1]
        )
        return out
    }

    #: The four ways to pick indices, split by replacement and weighting.
    fn _indices(
        self,
        population: int,
        shape: list[int],
        wanted: int,
        replace: bool,
        p: any
    ) -> NDArray {
        if p is null {
            if replace { return self.integers(0, population, shape) }
            return self._front(self.random([population]), shape, wanted)
        }
        let (weights, owned) = _weight_array(p, "p")
        defer { if owned { weights.free() } }
        if weights.size() != population {
            throw ValueError(f"{weights.size()} weights for a population of {population}")
        }
        if replace { return self._weighted(weights, population, shape) }
        return self._front(self._reservoir(weights), shape, wanted)
    }

    #: Indices drawn from a weighted population, by bisecting a scanned CDF.
    #:
    #: The scan is float32, so a population whose weights span more than seven
    #: decades loses the small ones into the running total. That is the same
    #: bound every accumulation in this package carries and not a choice made
    #: here.
    fn _weighted(self, weights: NDArray, population: int, shape: list[int]) -> NDArray {
        let cdf = cumsum(weights, 0)
        defer { cdf.free() }
        let out = NDArray(shape, INT32)
        let count = out.size()
        if count == 0 { return out }
        let call = self._advance()
        if population <= TILED_CDF {
            let padded = (count + GROUP - 1) // GROUP * GROUP
            kernel_of(SOURCE, "nrand_search_tiled").dispatch_async(
                [out.storage, cdf.storage],
                padded,
                [count, population, self._key(0), self._key(32), self.stream, call],
                GROUP
            )
            return out
        }
        kernel_of(SOURCE, "nrand_search").dispatch_async(
            [out.storage, cdf.storage],
            count,
            [count, population, self._key(0), self._key(32), self.stream, call]
        )
        return out
    }

    #: One Efraimidis-Spirakis key per member of the population.
    fn _reservoir(self, weights: NDArray) -> NDArray {
        let count = weights.size()
        let keys = NDArray([count], FLOAT32)
        if count == 0 { return keys }
        let call = self._advance()
        kernel_of(SOURCE, "nrand_reservoir").dispatch_async(
            [keys.storage, weights.storage],
            count,
            [count, self._key(0), self._key(32), self.stream, call]
        )
        return keys
    }

    #: The `wanted` smallest keys' positions, shaped like the request. Takes
    #: ownership of `keys`.
    #:
    #: One sort of the whole population and a prefix of it, not `wanted`
    #: passes: a partial sort would win only when `wanted` is a small fraction
    #: of the population, and the network is already the cheapest full
    #: ordering this package has.
    fn _front(self, keys: NDArray, shape: list[int], wanted: int) -> NDArray {
        defer { keys.free() }
        let order = argsort(keys, 0)
        defer { order.free() }
        let out = NDArray(shape, INT32)
        if wanted == 0 { return out }
        copy_into(out.view([wanted], [1], 0), order.view([wanted], [1], 0))
        return out
    }

    #: A random permutation of `0..n`, as the order that sorts random keys.
    fn _order(self, n: int) -> NDArray {
        if n < 0 { throw ValueError(f"a population cannot be negative, got {n}") }
        if n > MAX_POPULATION { throw ValueError(f"{n} is past what an exact index can name") }
        let keys = self.random([n])
        defer { keys.free() }
        return argsort(keys, 0)
    }
}

#: A generator on `seed`, or on host entropy when none is given.
#:
#: The entropy path is the one place this module touches `std.random`: one
#: 64-bit draw from the host generator, which is exactly the workload a host
#: generator is good at.
pub fn default_rng(seed: int? = null, stream: int = 0) -> Generator {
    return Generator(seed ?? host_random.next_int(), stream)
}

var _default = default_rng()

#: The generator the module-level functions draw from.
pub fn default_generator() -> Generator { return _default }

#: Re-seed the module-level generator, as `numpy.random.seed` does.
#:
#: This replaces it rather than rewinding one: a counter-based generator has
#: no state to rewind, and a fresh object is what makes the next call the
#: first call again.
pub fn seed(value: int) -> void { _default = Generator(value) }

#: Uniform in `[0, 1)` from the module-level generator.
pub fn random(size: any = null) -> NDArray { return _default.random(size) }

#: Uniform in `[low, high)` from the module-level generator.
pub fn uniform(low: float = 0.0, high: float = 1.0, size: any = null) -> NDArray {
    return _default.uniform(low, high, size)
}

#: Standard normal from the module-level generator.
pub fn standard_normal(size: any = null) -> NDArray { return _default.standard_normal(size) }

#: Normal from the module-level generator.
pub fn normal(loc: float = 0.0, scale: float = 1.0, size: any = null) -> NDArray {
    return _default.normal(loc, scale, size)
}

#: Whole numbers in `[low, high)` from the module-level generator.
pub fn integers(low: int, high: int? = null, size: any = null, endpoint: bool = false) -> NDArray {
    return _default.integers(low, high, size, endpoint)
}

#: `integers`, under the legacy name.
pub fn randint(low: int, high: int? = null, size: any = null) -> NDArray {
    return _default.integers(low, high, size)
}

#: Exponential from the module-level generator.
pub fn exponential(scale: float = 1.0, size: any = null) -> NDArray {
    return _default.exponential(scale, size)
}

#: Rayleigh from the module-level generator.
pub fn rayleigh(scale: float = 1.0, size: any = null) -> NDArray {
    return _default.rayleigh(scale, size)
}

#: Weibull from the module-level generator.
pub fn weibull(a: float, size: any = null) -> NDArray { return _default.weibull(a, size) }

#: Pareto II from the module-level generator.
pub fn pareto(a: float, size: any = null) -> NDArray { return _default.pareto(a, size) }

#: Logistic from the module-level generator.
pub fn logistic(loc: float = 0.0, scale: float = 1.0, size: any = null) -> NDArray {
    return _default.logistic(loc, scale, size)
}

#: Laplace from the module-level generator.
pub fn laplace(loc: float = 0.0, scale: float = 1.0, size: any = null) -> NDArray {
    return _default.laplace(loc, scale, size)
}

#: Log-normal from the module-level generator.
pub fn lognormal(mean: float = 0.0, sigma: float = 1.0, size: any = null) -> NDArray {
    return _default.lognormal(mean, sigma, size)
}

#: Gamma from the module-level generator.
pub fn gamma(shape: float, scale: float = 1.0, size: any = null) -> NDArray {
    return _default.gamma(shape, scale, size)
}

#: Beta from the module-level generator.
pub fn beta(a: float, b: float, size: any = null) -> NDArray { return _default.beta(a, b, size) }

#: Chi-square from the module-level generator.
pub fn chisquare(df: float, size: any = null) -> NDArray { return _default.chisquare(df, size) }

#: Student's t from the module-level generator.
pub fn standard_t(df: float, size: any = null) -> NDArray { return _default.standard_t(df, size) }

#: Poisson from the module-level generator.
pub fn poisson(lam: float = 1.0, size: any = null) -> NDArray { return _default.poisson(lam, size) }

#: Binomial from the module-level generator.
pub fn binomial(n: int, p: float, size: any = null) -> NDArray {
    return _default.binomial(n, p, size)
}

#: Multinomial from the module-level generator.
pub fn multinomial(n: int, pvals: any, size: any = null) -> NDArray {
    return _default.multinomial(n, pvals, size)
}

#: Choice from the module-level generator.
pub fn choice(a: any, size: any = null, replace: bool = true, p: any = null) -> NDArray {
    return _default.choice(a, size, replace, p)
}

#: Permutation from the module-level generator.
pub fn permutation(a: any) -> NDArray { return _default.permutation(a) }

#: In-place shuffle by the module-level generator.
pub fn shuffle(a: NDArray) -> void { _default.shuffle(a) }

#: Per-slice shuffle by the module-level generator.
pub fn permuted(a: NDArray, axis: int = -1) -> NDArray { return _default.permuted(a, axis) }

#: `random`, under the legacy name.
pub fn rand(shape: list[int] = []) -> NDArray { return _default.random(shape) }

#: `standard_normal`, under the legacy name.
pub fn randn(shape: list[int] = []) -> NDArray { return _default.standard_normal(shape) }

#: The shape a `size` argument asks for.
#:
#: A null size gives a 0-d array rather than a float. Every other array-valued
#: call in this package stays on the device until something asks for a value,
#: and a scalar draw that synchronised would be the one exception.
fn _shape_of(size: any) -> list[int] {
    if size is null { return [] }
    if size is int {
        if size < 0 { throw ValueError(f"a size cannot be negative, got {size}") }
        return [size]
    }
    if size is list { return [extent for extent in size] }
    if size is tuple { return [extent for extent in size] }
    throw TypeError(f"a size is an int, a list of ints or null, not {type_of(size)}")
}

#: A trial count both the kernel's uint scalar and a float32 result can hold.
fn _require_countable(n: int) -> void {
    if n < 0 { throw ValueError(f"a trial count cannot be negative, got {n}") }
    if n > MAX_EXACT_INT { throw ValueError(f"{n} trials leaves what float32 counts exactly") }
}

fn _require_positive(value: float, name: str) -> void {
    if not value > 0.0 { throw ValueError(f"{name} must be positive, got {value}") }
}

#: Weights as a dense 1-d float array, from a list or an existing array.
#:
#: A list is validated as it is converted; an array is trusted, because
#: checking it would mean a download and the caller that built it on the
#: device is exactly the caller who cannot afford one.
fn _weight_array(weights: any, name: str) -> (NDArray, bool) {
    if weights is NDArray {
        if weights.ndim() != 1 { throw ValueError(f"{name} is a 1-d array") }
        require_real(weights, name)
        #: A scanned CDF and a division by a weight both need real float
        #: slots; an integer dtype would round the scan on its way back in.
        if weights.dtype != FLOAT32 or not weights.is_contiguous() or weights.offset != 0 {
            return (weights.astype(FLOAT32), true)
        }
        return (weights, false)
    }
    if not weights is list { throw TypeError(f"{name} is a list of floats or a 1-d array") }
    var values: list[float] = []
    var total = 0.0
    for item in weights {
        let value = float(item)
        if value < 0.0 { throw ValueError(f"{name} cannot hold the negative weight {value}") }
        total += value
        values.push(value)
    }
    if not total > 0.0 { throw ValueError(f"{name} must hold some positive weight") }
    let out = NDArray([values.len()], FLOAT32)
    out.upload(values)
    return (out, true)
}

fn _population_of(a: any) -> int {
    if a is int { return a }
    if not a is NDArray { throw TypeError(f"choice takes an int or an array, not {type_of(a)}") }
    require_real(a, "choice")
    if a.ndim() != 1 { throw ValueError(f"choice draws from a 1-d array, not {a.ndim()} axes") }
    return a.shape[0]
}

#: `a` as a dense array starting at slot zero, and whether that cost an
#: allocation the caller now owns. `array.jai` keeps its own copy of this
#: private, and the flag is the only honest way to say which of the two
#: happened.
fn _dense(a: NDArray) -> (NDArray, bool) {
    if a.offset == 0 and a.is_contiguous() { return (a, false) }
    return (a.copy(), true)
}

#: A view of `a` with `at` moved to the end, so a per-slice kernel always
#: walks a contiguous fast axis. The same move `sorting.jai` makes.
fn _axis_last(a: NDArray, at: int) -> NDArray {
    if at == a.ndim() - 1 { return a }
    var shape: list[int] = []
    var strides: list[int] = []
    for axis in 0..a.ndim() {
        if axis == at { continue }
        shape.push(a.shape[axis])
        strides.push(a.strides[axis])
    }
    shape.push(a.shape[at])
    strides.push(a.strides[at])
    return a.view(shape, strides, a.offset)
}

#: `a` with its first axis reordered by `index`.
#:
#: The wide path needs the row length to divide by four, which is what makes a
#: row boundary a quad boundary on both sides; anything else goes scalar.
fn _take_rows(a: NDArray, index: NDArray) -> NDArray {
    let out = NDArray(a.shape, a.dtype)
    let rows = a.shape[0]
    if rows == 0 or a.size() == 0 { return out }
    let (source, owned) = _dense(a)
    defer { if owned { source.free() } }
    let width = a.size() // rows
    if width % 4 == 0 {
        let quads = width // 4
        kernel_of(SOURCE, "nrand_rows4").dispatch_async(
            [out.storage, source.storage, index.storage],
            rows * quads,
            [rows * quads, quads]
        )
        return out
    }
    kernel_of(SOURCE, "nrand_rows").dispatch_async(
        [out.storage, source.storage, index.storage],
        rows * width,
        [rows * width, width]
    )
    return out
}
