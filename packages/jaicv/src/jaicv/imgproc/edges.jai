#: `canny` — gradient, non-maximum suppression, and hysteresis.
#:
#: The gradient magnitude is the L1 sum of the Sobel derivatives by default,
#: which is what OpenCV computes unless `l2_gradient` asks for the Euclidean
#: one. Hysteresis runs as a fixed-point iteration on the device: a weak edge
#: is promoted while any neighbour is strong, repeated until nothing changes.
#:
#: ```
#: let outline = canny(gray, 80.0, 200.0)
#: ```
import std.math
from std.gpu import Buffer, Kernel
from ..core.mat import Mat, f32_bits
from ..core.types import BORDER_REPLICATE, CV_8U, CV_32F, CV_MAKETYPE
from .common import PRELUDE, saturation_args
from .filter import sobel

#: Rounds fired before the host asks how far they got.
const CANNY_BATCH = 3
const CANNY_BATCH_MAX = 64

let SOURCE = PRELUDE + """
kernel void canny_suppress(device const float* dx [[buffer(0)]],
                           device const float* dy [[buffer(1)]],
                           device float* labels [[buffer(2)]],
                           constant uint& count [[buffer(3)]],
                           constant uint& rows [[buffer(4)]],
                           constant uint& cols [[buffer(5)]],
                           constant uint& dx_off [[buffer(6)]],
                           constant uint& dx_step [[buffer(7)]],
                           constant uint& dy_off [[buffer(8)]],
                           constant uint& dy_step [[buffer(9)]],
                           constant uint& out_off [[buffer(10)]],
                           constant uint& out_step [[buffer(11)]],
                           constant uint& l2 [[buffer(12)]],
                           constant uint& low_bits [[buffer(13)]],
                           constant uint& high_bits [[buffer(14)]],
                           uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint y = id / cols;
    uint x = id - y * cols;

    float gx = dx[dx_off + y * dx_step + x];
    float gy = dy[dy_off + y * dy_step + x];
    float magnitude = l2 != 0u ? sqrt(gx * gx + gy * gy) : fabs(gx) + fabs(gy);
    float high = as_type<float>(high_bits);
    float low = as_type<float>(low_bits);
    if (magnitude <= low) {
        labels[out_off + y * out_step + x] = 0.0f;
        return;
    }

    /* The gradient direction picks which two neighbours the magnitude has to
     * beat. tan(22.5) and tan(67.5) split the four sectors. */
    float ax = fabs(gx);
    float ay = fabs(gy);
    int nx = 0;
    int ny = 0;
    if (ay <= ax * 0.41421356f) { nx = 1; ny = 0; }
    else if (ax <= ay * 0.41421356f) { nx = 0; ny = 1; }
    else if ((gx > 0.0f) == (gy > 0.0f)) { nx = 1; ny = 1; }
    else { nx = 1; ny = -1; }

    float before = 0.0f;
    float after = 0.0f;
    int px = int(x) - nx;
    int py = int(y) - ny;
    if (px >= 0 && py >= 0 && px < int(cols) && py < int(rows)) {
        float bx = dx[dx_off + uint(py) * dx_step + uint(px)];
        float by = dy[dy_off + uint(py) * dy_step + uint(px)];
        before = l2 != 0u ? sqrt(bx * bx + by * by) : fabs(bx) + fabs(by);
    }
    int qx = int(x) + nx;
    int qy = int(y) + ny;
    if (qx >= 0 && qy >= 0 && qx < int(cols) && qy < int(rows)) {
        float cx = dx[dx_off + uint(qy) * dx_step + uint(qx)];
        float cy = dy[dy_off + uint(qy) * dy_step + uint(qx)];
        after = l2 != 0u ? sqrt(cx * cx + cy * cy) : fabs(cx) + fabs(cy);
    }
    if (magnitude < before || magnitude < after) {
        labels[out_off + y * out_step + x] = 0.0f;
        return;
    }
    labels[out_off + y * out_step + x] = magnitude > high ? 2.0f : 1.0f;
}

/* Hysteresis as a joining problem rather than as a spreading one.
 *
 * A weak edge survives when it is connected, through other edges, to a strong
 * one -- so what matters is which edges belong to the same run, and whether
 * that run holds a strong pixel anywhere. Spreading the answer a ring at a
 * time takes as many passes as the longest run is long, and a frame of noisy
 * edges has runs hundreds of pixels long. Naming each run instead, and
 * letting the names collapse into each other, settles it in a handful.
 *
 * Every edge pixel starts as its own run, named for where it is. */
kernel void canny_seed(device const float* labels [[buffer(0)]],
                       device uint* runs [[buffer(1)]],
                       constant uint& count [[buffer(2)]],
                       constant uint& cols [[buffer(3)]],
                       constant uint& off [[buffer(4)]],
                       constant uint& step [[buffer(5)]],
                       uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint y = id / cols;
    uint x = id - y * cols;
    runs[id] = labels[off + y * step + x] != 0.0f ? id + 1u : 0u;
}

/* A pixel that can see a lower-numbered run tells its own run's root. */
/* Which round last had news, not whether any did. Rounds run in order, so a
 * batch of them can go down together and one read says where they settled --
 * one wait for the batch instead of one a round, and a wait costs about what
 * a round does. Every thread with news in a round stores the same value, so
 * the store needs no atomic. */
kernel void canny_join(device atomic_uint* runs [[buffer(0)]],
                       device float* changed [[buffer(1)]],
                       constant uint& count [[buffer(2)]],
                       constant uint& rows [[buffer(3)]],
                       constant uint& cols [[buffer(4)]],
                       constant uint& round [[buffer(5)]],
                       uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint mine = atomic_load_explicit(&runs[id], memory_order_relaxed);
    if (mine == 0u) return;
    int y = int(id / cols);
    int x = int(id - uint(y) * cols);
    uint best = mine;
    for (int dy = -1; dy <= 1; ++dy) {
        int sy = y + dy;
        if (sy < 0 || sy >= int(rows)) continue;
        for (int dx = -1; dx <= 1; ++dx) {
            int sx = x + dx;
            if (sx < 0 || sx >= int(cols)) continue;
            uint other = atomic_load_explicit(&runs[uint(sy) * cols + uint(sx)],
                                              memory_order_relaxed);
            if (other != 0u && other < best) best = other;
        }
    }
    if (best >= mine) return;
    atomic_fetch_min_explicit(&runs[mine - 1u], best, memory_order_relaxed);
    changed[0] = float(round);
}

/* Every chain pointed straight at its end. */
kernel void canny_flatten(device atomic_uint* runs [[buffer(0)]],
                          constant uint& count [[buffer(1)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint mine = atomic_load_explicit(&runs[id], memory_order_relaxed);
    if (mine == 0u) return;
    uint root = mine;
    for (uint guard = 0u; guard < 64u; ++guard) {
        uint up = atomic_load_explicit(&runs[root - 1u], memory_order_relaxed);
        if (up == root || up == 0u) break;
        root = up;
    }
    if (root != mine) atomic_store_explicit(&runs[id], root, memory_order_relaxed);
}

/* Which runs hold a strong pixel. */
kernel void canny_mark(device const float* labels [[buffer(0)]],
                       device const uint* runs [[buffer(1)]],
                       device atomic_uint* strong [[buffer(2)]],
                       constant uint& count [[buffer(3)]],
                       constant uint& cols [[buffer(4)]],
                       constant uint& off [[buffer(5)]],
                       constant uint& step [[buffer(6)]],
                       uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint y = id / cols;
    uint x = id - y * cols;
    if (labels[off + y * step + x] != 2.0f) return;
    uint run = runs[id];
    if (run == 0u) return;
    atomic_store_explicit(&strong[run - 1u], 1u, memory_order_relaxed);
}

/* An edge is kept when the run it belongs to holds a strong pixel. */
kernel void canny_tally(device const uint* value [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        constant uint& count [[buffer(2)]],
                        uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    out[id] = float(value[id]);
}

kernel void canny_keep(device const uint* runs [[buffer(0)]],
                       device const uint* strong [[buffer(1)]],
                       device float* out [[buffer(2)]],
                       constant uint& count [[buffer(3)]],
                       constant uint& cols [[buffer(4)]],
                       constant uint& d_off [[buffer(5)]],
                       constant uint& d_step [[buffer(6)]],
                       uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint y = id / cols;
    uint x = id - y * cols;
    uint run = runs[id];
    out[d_off + y * d_step + x] = (run != 0u && strong[run - 1u] != 0u) ? 255.0f : 0.0f;
}

"""

var _kernels: dict[str, Kernel] = {}

fn kernel_for(entry: str) -> Kernel {
    let existing = _kernels.get(entry, null)
    if existing is not null { return existing }
    let compiled = Kernel.compile(SOURCE, entry)
    _kernels[entry] = compiled
    return compiled
}

#: Find edges by the Canny method.
#:
#: `threshold1` and `threshold2` are the hysteresis pair; the smaller is the
#: level a pixel needs to survive at all, the larger the level that starts a
#: chain. `aperture_size` is the Sobel window.
pub fn canny(
    src: Mat,
    threshold1: float,
    threshold2: float,
    aperture_size: int = 3,
    l2_gradient: bool = false
) -> Mat {
    src.flush()
    if src.channels() != 1 { throw ValueError("canny needs a single-channel Mat") }
    let low = min(threshold1, threshold2)
    let high = max(threshold1, threshold2)
    let dx = sobel(src, CV_32F, 1, 0, ksize: aperture_size, border_mode: BORDER_REPLICATE)
    let dy = sobel(src, CV_32F, 0, 1, ksize: aperture_size, border_mode: BORDER_REPLICATE)
    let labels = Mat.zeros(src.rows, src.cols, CV_MAKETYPE(CV_32F, 1))
    let out = Mat.zeros(src.rows, src.cols, CV_MAKETYPE(CV_8U, 1))
    let count = src.rows * src.cols
    if count == 0 { return out }
    kernel_for("canny_suppress").dispatch_async([dx.storage, dy.storage, labels.storage], count, [count, src.rows, src.cols, dx.offset, dx.step_, dy.offset, dy.step_, labels.offset, labels.step_, if l2_gradient {
        1
    } else {
        0
    }, f32_bits(low), f32_bits(high)])
    labels.invalidate()

    #: Which run of connected edges each pixel belongs to, joined until the
    #: names stop moving.
    #:
    #: Spreading the strong label a ring at a time is the obvious way and was
    #: the way here, with a fixed pass count to bound it. Two things were
    #: wrong with that. A run longer than the pass count never finished, so
    #: six percent of the edges of a noisy frame were simply dropped; and the
    #: passes read and write the same labels, so how far each run had got when
    #: the count ran out depended on the order the threads happened to run in
    #: and the same picture answered differently every time. Joining runs by
    #: their lowest member settles in a handful of rounds instead of hundreds,
    #: and its fixed point does not care about the order.
    let runs = Buffer(count)
    kernel_for("canny_seed").dispatch_async(
        [labels.storage, runs],
        count,
        [count, src.cols, labels.offset, labels.step_]
    )
    let changed = Buffer(1)
    let counted = Buffer(1)
    let join = kernel_for("canny_join")
    let flatten = kernel_for("canny_flatten")
    #: A round with no news is the proof that the rounds before it settled,
    #: so a batch whose last round was quiet is done -- and the batch widens
    #: for a frame whose runs are long enough to need more of them.
    var rounds = 0
    var batch = CANNY_BATCH
    while rounds < count {
        changed.fill_zero()
        for step in 0..batch {
            join.dispatch_async([runs, changed], count, [count, src.rows, src.cols, step + 1])
            flatten.dispatch_async([runs], count, [count])
        }
        rounds += batch
        if changed.download()[0] < float(batch) { break }
        batch = min(batch * 2, CANNY_BATCH_MAX)
    }

    let strong = Buffer(count)
    strong.fill_zero()
    kernel_for("canny_mark").dispatch_async(
        [labels.storage, runs, strong],
        count,
        [count, src.cols, labels.offset, labels.step_]
    )
    kernel_for("canny_keep").dispatch_async(
        [runs, strong, out.storage],
        count,
        [count, src.cols, out.offset, out.step_]
    )
    out.invalidate()
    runs.free()
    strong.free()
    changed.free()
    counted.free()
    return out
}
