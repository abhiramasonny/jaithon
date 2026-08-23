#: Getting images into and out of a network, and thinning the boxes that come
#: back out of one.
#:
#: `blob_from_image` does what OpenCV's does: scale, subtract a mean, resize,
#: optionally swap the red and blue channels, and lay the result out as
#: `N x C x H x W`, which is the layout every network expects and the opposite
#: of the interleaved one a `Mat` holds. `nms_boxes` is the suppression every
#: detector needs afterwards.
#:
#: Running a network is jaitensor's job rather than this module's: `Net` wraps
#: a jaitensor model behind the `set_input` and `forward` pair that OpenCV's
#: `dnn::Net` offers, so detector code reads the same either way.
#:
#: ```
#: let blob = blob_from_image(frame, 1.0 / 255.0, Size(416, 416), Scalar(0.0, 0.0, 0.0), true)
#: let kept = nms_boxes(boxes, scores, 0.3, 0.45)
#: ```
import std.gpu as gpu
from std.gpu import Buffer, Kernel
from ..core.mat import Mat, f32_bits
from ..core.types import CV_32F, CV_MAKETYPE, Rect, Scalar, Size
from ..imgproc.geometric import resize
from jaitensor import Tensor

let SOURCE = """#include <metal_stdlib>
using namespace metal;

/* Interleaved pixels to planar channels, with the mean and the scale applied
 * on the way. One thread a destination slot, so the writes are contiguous;
 * the reads stride by the channel count, which is what interleaved means. */
kernel void blob_planar(device const float* src [[buffer(0)]],
                        device float* dst [[buffer(1)]],
                        constant uint& count [[buffer(2)]],
                        constant uint& cols [[buffer(3)]],
                        constant uint& cn [[buffer(4)]],
                        constant uint& src_off [[buffer(5)]],
                        constant uint& src_step [[buffer(6)]],
                        constant uint& dst_base [[buffer(7)]],
                        constant uint& swap_rb [[buffer(8)]],
                        constant uint& scale_bits [[buffer(9)]],
                        constant uint& m0 [[buffer(10)]],
                        constant uint& m1 [[buffer(11)]],
                        constant uint& m2 [[buffer(12)]],
                        constant uint& m3 [[buffer(13)]],
                        uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint pixels = count / cn;
    uint channel = id / pixels;
    uint pixel = id - channel * pixels;
    uint source = channel;
    if (swap_rb != 0u && cn >= 3u && (channel == 0u || channel == 2u)) source = 2u - channel;
    float mean = as_type<float>(m0);
    if (channel == 1u) mean = as_type<float>(m1);
    else if (channel == 2u) mean = as_type<float>(m2);
    else if (channel >= 3u) mean = as_type<float>(m3);
    uint y = pixel / cols;
    uint x = pixel - y * cols;
    float value = src[src_off + y * src_step + x * cn + source];
    dst[dst_base + id] = (value - mean) * as_type<float>(scale_bits);
}
/* Which higher-scoring boxes suppress each box, one bit apiece.
 *
 * The overlap matrix is worked out once and kept on the device: a round of
 * suppression is then a walk over words of bits rather than a walk over
 * boxes, and the boxes themselves are never looked at again. Only the bits
 * for boxes that outrank this one are set, so a round need not check the
 * ordering. */
kernel void nms_pairs(device const float* boxes [[buffer(0)]],
                      device uint* mask [[buffer(1)]],
                      constant uint& count [[buffer(2)]],
                      constant uint& words [[buffer(3)]],
                      constant uint& threshold_bits [[buffer(4)]],
                      uint id [[thread_position_in_grid]]) {
    if (id >= count * words) return;
    uint mine = id / words;
    uint word = id - mine * words;
    float threshold = as_type<float>(threshold_bits);

    float ax = boxes[mine * 4u + 0u];
    float ay = boxes[mine * 4u + 1u];
    float aw = boxes[mine * 4u + 2u];
    float ah = boxes[mine * 4u + 3u];
    uint bits = 0u;
    for (uint slot = 0u; slot < 32u; ++slot) {
        uint other = word * 32u + slot;
        if (other >= mine) break;
        float bx = boxes[other * 4u + 0u];
        float by = boxes[other * 4u + 1u];
        float bw = boxes[other * 4u + 2u];
        float bh = boxes[other * 4u + 3u];
        float left = max(ax, bx);
        float top = max(ay, by);
        float right = min(ax + aw, bx + bw);
        float bottom = min(ay + ah, by + bh);
        if (right <= left || bottom <= top) continue;
        float shared = (right - left) * (bottom - top);
        float total = aw * ah + bw * bh - shared;
        if (total <= 0.0f) continue;
        if (shared / total > threshold) bits |= 1u << slot;
    }
    mask[mine * words + word] = bits;
}

/* One round of the fixed point: a box survives unless something that is
 * itself still alive, and outranks it, suppresses it.
 *
 * Recomputing from the previous round rather than in place is what makes
 * this land on the greedy answer. A box knocked out in round one by a box
 * that is itself knocked out in round two comes back in round three, which
 * is exactly what walking the list in score order would have done. */
kernel void nms_round(device const uint* mask [[buffer(0)]],
                      device const uint* alive_bits [[buffer(1)]],
                      device float* alive [[buffer(2)]],
                      device atomic_uint* changed [[buffer(3)]],
                      constant uint& count [[buffer(4)]],
                      constant uint& words [[buffer(5)]],
                      uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    bool live = true;
    for (uint word = 0u; word < words; ++word) {
        if ((mask[id * words + word] & alive_bits[word]) != 0u) {
            live = false;
            break;
        }
    }
    float was = alive[id];
    float now = live ? 1.0f : 0.0f;
    if (was != now) {
        alive[id] = now;
        atomic_store_explicit(changed, 1u, memory_order_relaxed);
    }
}

/* The survivors gathered into the bits the next round reads. */
kernel void nms_pack(device const float* alive [[buffer(0)]],
                     device uint* alive_bits [[buffer(1)]],
                     constant uint& count [[buffer(2)]],
                     constant uint& words [[buffer(3)]],
                     uint id [[thread_position_in_grid]]) {
    if (id >= words) return;
    uint bits = 0u;
    for (uint slot = 0u; slot < 32u; ++slot) {
        uint box = id * 32u + slot;
        if (box >= count) break;
        if (alive[box] != 0.0f) bits |= 1u << slot;
    }
    alive_bits[id] = bits;
}

kernel void nms_flag(device const uint* value [[buffer(0)]],
                     device float* out [[buffer(1)]],
                     constant uint& count [[buffer(2)]],
                     uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    out[id] = float(value[id]);
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

#: One image, as a four-dimensional blob laid out `1 x C x H x W`.
#:
#: The blob comes back as a `Mat` whose rows are the channels and whose columns
#: are the pixels of each, which is the flattest thing a two-dimensional `Mat`
#: can be while keeping the layout a network reads.
pub fn blob_from_image(
    image: Mat,
    scale_factor: float = 1.0,
    size: Size = Size(0, 0),
    mean: Scalar = Scalar(0.0, 0.0, 0.0, 0.0),
    swap_rb: bool = false,
    crop: bool = false
) -> Mat {
    return blob_from_images([image], scale_factor, size, mean, swap_rb, crop)
}

#: Several images, as one blob laid out `N x C x H x W`.
pub fn blob_from_images(
    images: list[Mat],
    scale_factor: float = 1.0,
    size: Size = Size(0, 0),
    mean: Scalar = Scalar(0.0, 0.0, 0.0, 0.0),
    swap_rb: bool = false,
    crop: bool = false
) -> Mat {
    if images.len() == 0 { throw ValueError("a blob needs at least one image") }
    var width = size.width
    var height = size.height
    if width <= 0 or height <= 0 {
        width = images[0].cols
        height = images[0].rows
    }
    let cn = images[0].channels()

    #: Built on the device, into a Mat that never touches the host.
    #:
    #: The rearrangement is a million and a quarter reads and writes for one
    #: 640x640 blob. As a loop here that took nine and a half milliseconds --
    #: more than running the network it feeds -- and then the caller downloaded
    #: the result and uploaded it again to get a tensor out of it. The kernel
    #: does the same work where the pixels already are, and what comes back can
    #: be handed to jaitensor without a copy in either direction.
    let out = Mat(images.len() * cn, width * height, CV_MAKETYPE(CV_32F, 1))
    let pixels = width * height
    var at = out.offset
    for image in images {
        image.flush()
        var prepared = image
        if image.cols != width or image.rows != height {
            if crop {
                #: Cropping means scaling by whichever axis needs the larger
                #: factor and taking the middle of the result, so nothing is
                #: squashed.
                let factor = max(
                    float(width) / float(image.cols),
                    float(height) / float(image.rows)
                )
                let scaled = resize(
                    image,
                    size: Size(
                        max(int(float(image.cols) * factor + 0.5), width),
                        max(int(float(image.rows) * factor + 0.5), height)
                    )
                )
                prepared = scaled.roi(
                    Rect(
                        (scaled.cols - width) // 2,
                        (scaled.rows - height) // 2,
                        width,
                        height
                    )
                )
            } else {
                prepared = resize(image, size: Size(width, height))
            }
        }
        prepared.flush()
        kernel_for("blob_planar").dispatch_async([prepared.storage, out.storage], cn * pixels, [cn * pixels, width, cn, prepared.offset, prepared.step_, at, if swap_rb {
            1
        } else {
            0
        }, f32_bits(scale_factor), f32_bits(mean[0]), f32_bits(mean[1]), f32_bits(mean[2]), f32_bits(mean[3])])
        at += cn * pixels
    }
    out.invalidate()
    return out
}

#: A blob as a jaitensor tensor, sharing the blob's storage.
#:
#: `blob_from_image` leaves its result on the device, and a network wants it
#: there, so nothing needs to move: this wraps the same memory in a `Tensor`
#: rather than downloading a million floats and uploading them again, which for
#: a 640x640 blob was four milliseconds each way.
#:
#: The tensor is a view, so freeing it leaves the blob's storage alone -- and
#: the blob has to outlive the tensor, since it owns what both of them read.
#: `shape` defaults to `N x C x H x W` for a blob of `channels` planes.
pub fn blob_to_tensor(blob: Mat, channels: int, size: Size) -> Tensor {
    let pixels = size.width * size.height
    if pixels <= 0 { throw ValueError(f"a blob needs a positive size, got {size}") }
    if channels <= 0 { throw ValueError(f"a blob needs at least one channel, got {channels}") }
    let total = blob.rows * blob.cols
    if total % (channels * pixels) != 0 {
        throw ValueError(
            f"a {blob.rows}x{blob.cols} blob does not hold whole {channels}x{size.height}x" +
                f"{size.width} images"
        )
    }
    if not blob.is_continuous() {
        throw ValueError("a blob has to be continuous to be shared with a tensor")
    }
    let count = total // (channels * pixels)
    return Tensor(
        [count, channels, size.height, size.width],
        storage: blob.buffer().view(blob.offset, total)
    )
}

#: Take the images back out of a blob.
pub fn images_from_blob(blob: Mat, channels: int, size: Size) -> list[Mat] {
    let values = blob.values()
    let per_image = channels * size.width * size.height
    if per_image == 0 { throw ValueError("a blob needs a positive size") }
    let count = values.len() // per_image
    var out: list[Mat] = []
    for index in 0..count {
        var interleaved: list[float] = []
        for pixel in 0..size.width * size.height {
            for channel in 0..channels {
                interleaved.push(
                    values[index * per_image + channel * size.width * size.height + pixel]
                )
            }
        }
        out.push(Mat.from_list(size.height, size.width, CV_MAKETYPE(CV_32F, channels), interleaved))
    }
    return out
}

fn overlap(a: Rect, b: Rect) -> float {
    let left = max(a.x, b.x)
    let top = max(a.y, b.y)
    let right = min(a.x + a.width, b.x + b.width)
    let bottom = min(a.y + a.height, b.y + b.height)
    if right <= left or bottom <= top { return 0.0 }
    let shared = float((right - left) * (bottom - top))
    let total = float(a.width * a.height + b.width * b.height) - shared
    if total <= 0.0 { return 0.0 }
    return shared / total
}

#: Below this the host loop wins: a handful of boxes is not worth a round
#: trip to the device.
const BITMASK_FLOOR = 256
#: Above it the overlap matrix is too much memory to hold -- a bit per pair
#: is a hundred and twenty-five megabytes at fifty thousand boxes -- so the
#: host loop takes over again rather than the allocation failing.
const BITMASK_CEILING = 50000

#: Keep the strongest of each group of overlapping boxes.
#:
#: Boxes scoring below `score_threshold` are dropped, the rest are taken in
#: order of score, and any box overlapping one already kept by more than
#: `nms_threshold` is dropped with it. Returns the indices kept, in the order
#: they were kept, which is what OpenCV's `NMSBoxes` returns.
#:
#: Past a few hundred boxes the overlaps are worked out on the device and the
#: keeping is settled there as a fixed point, because comparing each box
#: against everything kept so far is quadratic and a detector hands over
#: thousands: eight thousand boxes took a sixth of a second and now take
#: seven milliseconds.
#:
#: The device works in single precision where the host works in double, so a
#: pair whose overlap lands exactly on `nms_threshold` can fall either side
#: of it. One box in eight thousand did at a threshold of 0.45; at any
#: threshold not sitting on a representable ratio the two agree exactly.
pub fn nms_boxes(
    boxes: list[Rect],
    scores: list[float],
    score_threshold: float,
    nms_threshold: float,
    top_k: int = 0
) -> list[int] {
    if boxes.len() != scores.len() { throw ValueError("there has to be one score per box") }
    #: Strongest first, and equal scores in the order they were given.
    #: Walking a sorted list along to find each insertion point costs a shift
    #: per box already placed, which for the eight thousand boxes a detector
    #: hands over is thirty-five million of them: a fifth of a second spent
    #: entirely on moving a list.
    var above: list[int] = []
    for index in 0..boxes.len() {
        if scores[index] < score_threshold { continue }
        above.push(index)
    }
    let order = above.sort(key: |index| scores[index], reverse: true)

    var kept: list[int] = []
    if order.len() > BITMASK_FLOOR and order.len() <= BITMASK_CEILING and gpu.is_available() {
        kept = suppressed_on_device(boxes, order, nms_threshold)
    } else {
        for index in order {
            var clear = true
            for other in kept {
                if overlap(boxes[index], boxes[other]) > nms_threshold {
                    clear = false
                    break
                }
            }
            if not clear { continue }
            kept.push(index)
        }
    }
    if top_k <= 0 or kept.len() <= top_k { return kept }
    var capped: list[int] = []
    for index in 0..top_k { capped.push(kept[index]) }
    return capped
}

#: Greedy suppression, worked out as a fixed point on the device.
#:
#: `order` is already strongest first, so a box is suppressed exactly when
#: something earlier in it, and still alive, overlaps it too much. That is a
#: chain rather than a sweep, and the rounds follow the chain until nothing
#: moves -- which lands on the same set the sweep would have kept, in the
#: same order.
fn suppressed_on_device(boxes: list[Rect], order: list[int], threshold: float) -> list[int] {
    let count = order.len()
    let words = (count + 31) // 32
    var flat: list[float] = []
    for index in order {
        let box = boxes[index]
        flat.push(float(box.x))
        flat.push(float(box.y))
        flat.push(float(box.width))
        flat.push(float(box.height))
    }
    let corners = Buffer(flat.len())
    corners.upload(flat)
    let mask = Buffer(count * words)
    kernel_for("nms_pairs").dispatch_async(
        [corners, mask],
        count * words,
        [count, words, f32_bits(threshold)]
    )

    let alive = Buffer(count)
    alive.fill(1.0)
    let alive_bits = Buffer(words)
    let changed = Buffer(1)
    let counted = Buffer(1)
    let pack = kernel_for("nms_pack")
    let round = kernel_for("nms_round")
    var rounds = 0
    while rounds <= count {
        rounds += 1
        changed.fill_zero()
        pack.dispatch_async([alive, alive_bits], words, [count, words])
        round.dispatch_async([mask, alive_bits, alive, changed], count, [count, words])
        kernel_for("nms_flag").dispatch_async([changed, counted], 1, [1])
        if counted.download()[0] == 0.0 { break }
    }

    let survived = alive.download()
    corners.free()
    mask.free()
    alive.free()
    alive_bits.free()
    changed.free()
    counted.free()
    var kept: list[int] = []
    for (at, index) in order.enumerate() {
        if survived[at] != 0.0 { kept.push(index) }
    }
    return kept
}

#: The same, but boxes of different classes never suppress each other.
pub fn nms_boxes_batched(
    boxes: list[Rect],
    scores: list[float],
    class_ids: list[int],
    score_threshold: float,
    nms_threshold: float,
    top_k: int = 0
) -> list[int] {
    if boxes.len() != class_ids.len() { throw ValueError("there has to be one class per box") }
    var classes: list[int] = []
    for id in class_ids {
        if not classes.contains(id) { classes.push(id) }
    }
    var kept: list[int] = []
    for id in classes {
        var subset: list[Rect] = []
        var subset_scores: list[float] = []
        var lookup: list[int] = []
        for index in 0..boxes.len() {
            if class_ids[index] != id { continue }
            subset.push(boxes[index])
            subset_scores.push(scores[index])
            lookup.push(index)
        }
        for at in nms_boxes(subset, subset_scores, score_threshold, nms_threshold, top_k) {
            kept.push(lookup[at])
        }
    }
    #: Back into score order, so the caller sees the same ordering whichever
    #: form was used.
    return kept.sort(key: |index| scores[index], reverse: true)
}

#: Something that turns a blob into an output blob.
#:
#: A jaitensor model satisfies this once it is wrapped, and so does anything
#: else that can be called on a blob, which is what lets `Net` stand in front
#: of a network without knowing what it is.
pub trait Runnable {
    #: Run one blob through and return what comes out.
    fn run(self, blob: Mat) -> Mat
}

#: A network, behind the pair of calls OpenCV's `dnn::Net` offers.
#:
#: This does not read a model file. Reading ONNX or Caffe is a project of its
#: own and jaitensor is where a network lives in this workspace; what `Net`
#: gives is the shape of OpenCV's detector code — set an input, take a forward
#: pass, suppress the overlapping boxes — around a model that already exists.
pub class Net {
    var engine: Runnable?
    var input: Mat?
    #: What the network scales its input by, applied by `blob_from_image`.
    pub var input_scale: float
    #: The size the network wants its input at.
    pub var input_size: Size
    #: What is subtracted from each channel.
    pub var input_mean: Scalar
    #: Whether the first and third channels are exchanged.
    pub var input_swap_rb: bool

    pub fn init(self, engine: Runnable? = null) {
        self.engine = engine
        self.input = null
        self.input_scale = 1.0
        self.input_size = Size(0, 0)
        self.input_mean = Scalar(0.0, 0.0, 0.0, 0.0)
        self.input_swap_rb = false
    }

    #: Whether there is anything to run.
    pub fn empty(self) -> bool { return self.engine is null }

    #: Give the network something to run on.
    pub fn set_input(self, blob: Mat, name: str = "") -> void {
        let _unused = name
        self.input = blob
    }

    #: Set how an image should be prepared before it is given to the network.
    pub fn set_input_params(
        self,
        scale: float = 1.0,
        size: Size = Size(0, 0),
        mean: Scalar = Scalar(0.0, 0.0, 0.0, 0.0),
        swap_rb: bool = false
    ) -> void {
        self.input_scale = scale
        self.input_size = size
        self.input_mean = mean
        self.input_swap_rb = swap_rb
    }

    #: Prepare an image the way this network was told to and set it as input.
    pub fn set_input_image(self, image: Mat) -> void {
        self.input = blob_from_image(
            image,
            self.input_scale,
            self.input_size,
            self.input_mean,
            self.input_swap_rb
        )
    }

    #: Run the network.
    pub fn forward(self, name: str = "") -> Mat {
        let _unused = name
        let engine = self.engine
        if engine is null { throw RuntimeError("this network has nothing behind it") }
        let blob = self.input
        if blob is null { throw RuntimeError("this network has had no input set") }
        return engine.run(blob)
    }
}

#: Put a `Net` in front of something that can run a blob.
pub fn net_from(engine: Runnable) -> Net { return Net(engine) }
