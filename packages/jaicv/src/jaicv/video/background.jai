#: Telling a moving foreground from a background that stays put.
#:
#: Both subtractors here keep a model per pixel and update it as frames arrive:
#: `BackgroundSubtractorMOG2` a small mixture of Gaussians, and
#: `BackgroundSubtractorKNN` a short memory of recent values. A pixel is
#: foreground when the new frame disagrees with what its own model expects.
#:
#: The model is one buffer on the device and every frame updates it in place
#: with one kernel launch, so the cost per frame is a single pass over the
#: image no matter how many components the model has.
#:
#: ```
#: let subtractor = BackgroundSubtractorMOG2()
#: for frame in frames { let mask = subtractor.apply(frame) }
#: ```
import std.math
from std.gpu import Buffer, Kernel
from ..core.mat import Mat, f32_bits
from ..core.types import CV_8U, CV_MAKETYPE

#: What a shadow is marked with in the mask, when shadow detection is on.
pub const SHADOW_VALUE = 127
#: What the foreground is marked with.
pub const FOREGROUND_VALUE = 255

let SOURCE = """#include <metal_stdlib>
using namespace metal;

/* The model for one pixel is `mixtures` entries laid end to end, each of them
 * a weight, a variance, and one mean per channel, kept sorted so the strongest
 * component is first. */
kernel void bg_mog2(device const float* frame [[buffer(0)]],
                    device float* model [[buffer(1)]],
                    device float* mask [[buffer(2)]],
                    constant uint& count [[buffer(3)]],
                    constant uint& cn [[buffer(4)]],
                    constant uint& mixtures [[buffer(5)]],
                    constant uint& frame_off [[buffer(6)]],
                    constant uint& frame_step [[buffer(7)]],
                    constant uint& cols [[buffer(8)]],
                    constant float& learning_rate [[buffer(9)]],
                    constant float& var_threshold [[buffer(10)]],
                    constant float& var_init [[buffer(11)]],
                    constant float& var_min [[buffer(12)]],
                    constant float& var_max [[buffer(13)]],
                    constant float& background_ratio [[buffer(14)]],
                    constant float& complexity_prior [[buffer(15)]],
                    constant uint& detect_shadows [[buffer(16)]],
                    constant float& shadow_threshold [[buffer(17)]],
                    uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint y = id / cols;
    uint x = id - y * cols;

    float pixel[4];
    for (uint c = 0; c < cn; ++c) {
        pixel[c] = frame[frame_off + y * frame_step + x * cn + c];
    }

    uint stride = 2u + cn;
    device float* entry = model + id * mixtures * stride;

    float total_weight = 0.0f;
    int matched = -1;
    for (uint k = 0; k < mixtures; ++k) {
        device float* g = entry + k * stride;
        float weight = g[0];
        /* Every component decays a little each frame, and the prior nudges the
         * weakest towards zero so unused components are recycled rather than
         * lingering forever. */
        weight = weight - learning_rate * (weight + complexity_prior);
        if (matched < 0 && weight > 0.0f) {
            float variance = g[1];
            float distance = 0.0f;
            for (uint c = 0; c < cn; ++c) {
                float gap = pixel[c] - g[2 + c];
                distance += gap * gap;
            }
            if (distance < var_threshold * variance) {
                matched = int(k);
                weight = weight + learning_rate;
                float rate = learning_rate / max(weight, 1e-6f);
                for (uint c = 0; c < cn; ++c) {
                    g[2 + c] = g[2 + c] + rate * (pixel[c] - g[2 + c]);
                }
                float updated = variance + rate * (distance - variance);
                g[1] = clamp(updated, var_min, var_max);
            }
        }
        if (weight < 0.0f) {
            weight = 0.0f;
            g[1] = 0.0f;
        }
        g[0] = weight;
        total_weight += weight;
    }

    if (matched < 0) {
        /* Nothing explained this pixel, so the weakest component is replaced
         * by one centred on what was actually seen. */
        uint weakest = mixtures - 1u;
        for (uint k = 0; k < mixtures; ++k) {
            if (entry[k * stride] < entry[weakest * stride]) weakest = k;
        }
        device float* g = entry + weakest * stride;
        total_weight = total_weight - g[0] + learning_rate;
        g[0] = learning_rate;
        g[1] = var_init;
        for (uint c = 0; c < cn; ++c) g[2 + c] = pixel[c];
        matched = int(weakest);
    }

    if (total_weight > 0.0f) {
        for (uint k = 0; k < mixtures; ++k) entry[k * stride] /= total_weight;
    }

    /* Keep the strongest first with one pass of insertion, which is enough
     * because only the matched component's weight went up. */
    for (int k = matched; k > 0; --k) {
        device float* here = entry + uint(k) * stride;
        device float* above = entry + uint(k - 1) * stride;
        if (above[0] >= here[0]) break;
        for (uint f = 0; f < stride; ++f) {
            float swap = here[f];
            here[f] = above[f];
            above[f] = swap;
        }
        matched = k - 1;
    }

    /* A pixel is background when the component that explained it is inside the
     * heaviest few that together make up `background_ratio` of the model. */
    float running = 0.0f;
    int background_components = 0;
    for (uint k = 0; k < mixtures; ++k) {
        running += entry[k * stride];
        background_components = int(k) + 1;
        if (running > background_ratio) break;
    }

    float result = matched < background_components ? 0.0f : 255.0f;
    if (result > 0.0f && detect_shadows != 0u) {
        /* A shadow is the background seen darker: the same colour, scaled
         * down, and still close to the strongest component once scaled back. */
        float numerator = 0.0f;
        float denominator = 0.0f;
        for (uint c = 0; c < cn; ++c) {
            numerator += pixel[c] * entry[2 + c];
            denominator += entry[2 + c] * entry[2 + c];
        }
        if (denominator > 0.0f) {
            float scale = numerator / denominator;
            if (scale >= shadow_threshold && scale <= 1.0f) {
                float distance = 0.0f;
                for (uint c = 0; c < cn; ++c) {
                    float gap = scale * entry[2 + c] - pixel[c];
                    distance += gap * gap;
                }
                if (distance < var_threshold * entry[1]) result = 127.0f;
            }
        }
    }
    mask[id] = result;
}

/* The KNN model keeps the last `samples` values seen at each pixel, together
 * with a flag saying whether each was itself judged background. */
kernel void bg_knn(device const float* frame [[buffer(0)]],
                   device float* model [[buffer(1)]],
                   device float* mask [[buffer(2)]],
                   device float* cursor [[buffer(3)]],
                   constant uint& count [[buffer(4)]],
                   constant uint& cn [[buffer(5)]],
                   constant uint& samples [[buffer(6)]],
                   constant uint& frame_off [[buffer(7)]],
                   constant uint& frame_step [[buffer(8)]],
                   constant uint& cols [[buffer(9)]],
                   constant float& learning_rate [[buffer(10)]],
                   constant float& distance_threshold [[buffer(11)]],
                   constant uint& neighbours [[buffer(12)]],
                   uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint y = id / cols;
    uint x = id - y * cols;

    float pixel[4];
    for (uint c = 0; c < cn; ++c) {
        pixel[c] = frame[frame_off + y * frame_step + x * cn + c];
    }

    uint stride = cn + 1u;
    device float* entry = model + id * samples * stride;

    uint close = 0;
    for (uint k = 0; k < samples; ++k) {
        device float* s = entry + k * stride;
        if (s[cn] == 0.0f) continue;
        float distance = 0.0f;
        for (uint c = 0; c < cn; ++c) {
            float gap = pixel[c] - s[c];
            distance += gap * gap;
        }
        if (distance < distance_threshold) close += 1u;
    }

    mask[id] = close >= neighbours ? 0.0f : 255.0f;

    /* Only a fraction of frames update the memory, which is what the learning
     * rate means here: a smaller rate remembers further back. */
    float turn = cursor[id];
    float step = learning_rate * float(samples);
    turn += step;
    if (turn >= 1.0f) {
        uint slot = uint(fmod(cursor[id] * float(samples) + float(samples), float(samples)));
        device float* s = entry + (slot % samples) * stride;
        for (uint c = 0; c < cn; ++c) s[c] = pixel[c];
        s[cn] = 1.0f;
        turn -= 1.0f;
    }
    cursor[id] = turn + 1e-6f;
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

#: A background model built from a mixture of Gaussians per pixel.
#:
#: Each pixel keeps a few Gaussians describing what it has looked like. A new
#: value that one of the heavier ones explains is background; anything else is
#: foreground. Because the components are learned rather than fixed, a pixel
#: that flickers between two states — a leaf moving, a screen refreshing —
#: settles into two components and stops being reported as motion.
pub class BackgroundSubtractorMOG2 {
    pub var history: int
    pub var var_threshold: float
    pub var detect_shadows: bool
    pub var n_mixtures: int
    pub var background_ratio: float
    pub var var_init: float
    pub var var_min: float
    pub var var_max: float
    pub var complexity_reduction: float
    pub var shadow_threshold: float
    var model: Buffer?
    var channels: int
    var pixels: int
    var frames: int

    pub fn init(
        self,
        history: int = 500,
        var_threshold: float = 16.0,
        detect_shadows: bool = true
    ) {
        self.history = max(history, 1)
        self.var_threshold = var_threshold
        self.detect_shadows = detect_shadows
        self.n_mixtures = 5
        self.background_ratio = 0.9
        self.var_init = 15.0
        self.var_min = 4.0
        self.var_max = 75.0
        self.complexity_reduction = 0.05
        self.shadow_threshold = 0.5
        self.model = null
        self.channels = 0
        self.pixels = 0
        self.frames = 0
    }

    #: Feed one frame in and get its foreground mask back.
    #:
    #: A negative `learning_rate` lets the model choose one from `history`,
    #: which is what OpenCV does; zero freezes the model and one forgets
    #: everything but the frame just given.
    pub fn apply(self, frame: Mat, learning_rate: float = -1.0) -> Mat {
        frame.flush()
        let cn = frame.channels()
        if cn > 4 { throw ValueError("background subtraction handles at most four channels") }
        let pixels = frame.rows * frame.cols
        let stride = 2 + cn
        if self.model is null or self.channels != cn or self.pixels != pixels {
            let buffer = Buffer(pixels * self.n_mixtures * stride)
            buffer.fill_zero()
            self.model = buffer
            self.channels = cn
            self.pixels = pixels
            self.frames = 0
        }
        let model = self.model
        if model is null { throw RuntimeError("the background model went missing") }

        self.frames += 1
        var rate = learning_rate
        if rate < 0.0 { rate = 1.0 / float(min(self.frames, self.history)) }

        let mask = Mat.zeros(frame.rows, frame.cols, CV_MAKETYPE(CV_8U, 1))
        if pixels == 0 { return mask }
        kernel_for("bg_mog2").dispatch_async([frame.storage, model, mask.storage], pixels, [pixels, cn, self.n_mixtures, frame.offset, frame.step_, frame.cols, f32_bits(rate), f32_bits(self.var_threshold), f32_bits(self.var_init), f32_bits(self.var_min), f32_bits(self.var_max), f32_bits(self.background_ratio), f32_bits(self.complexity_reduction * rate), if self.detect_shadows {
            1
        } else {
            0
        }, f32_bits(self.shadow_threshold)])
        mask.invalidate()
        return mask
    }

    #: What the model currently thinks the scene looks like without anything
    #: moving in it.
    pub fn get_background_image(self) -> Mat? {
        let model = self.model
        if model is null { return null }
        let values = model.download()
        let stride = 2 + self.channels
        var out: list[float] = []
        for index in 0..self.pixels {
            let base = index * self.n_mixtures * stride
            for channel in 0..self.channels { out.push(values[base + 2 + channel]) }
        }
        return Mat.from_list(1, self.pixels, CV_MAKETYPE(CV_8U, self.channels), out)
    }

    #: Throw the model away and start again.
    pub fn clear(self) -> void {
        let model = self.model
        if model is not null { model.free() }
        self.model = null
        self.frames = 0
    }
}

#: A background model built from a short memory of what each pixel has been.
#:
#: Simpler than the mixture: keep the last few values seen, and call a pixel
#: background when enough of them are close to the new one. It copes better
#: with a background that changes shape rather than brightness.
pub class BackgroundSubtractorKNN {
    pub var history: int
    pub var dist2_threshold: float
    pub var detect_shadows: bool
    pub var n_samples: int
    pub var k_nn: int
    var model: Buffer?
    var cursor: Buffer?
    var channels: int
    var pixels: int
    var frames: int

    pub fn init(
        self,
        history: int = 500,
        dist2_threshold: float = 400.0,
        detect_shadows: bool = true
    ) {
        self.history = max(history, 1)
        self.dist2_threshold = dist2_threshold
        self.detect_shadows = detect_shadows
        self.n_samples = 7
        self.k_nn = 3
        self.model = null
        self.cursor = null
        self.channels = 0
        self.pixels = 0
        self.frames = 0
    }

    #: Feed one frame in and get its foreground mask back.
    pub fn apply(self, frame: Mat, learning_rate: float = -1.0) -> Mat {
        frame.flush()
        let cn = frame.channels()
        if cn > 4 { throw ValueError("background subtraction handles at most four channels") }
        let pixels = frame.rows * frame.cols
        let stride = cn + 1
        if self.model is null or self.channels != cn or self.pixels != pixels {
            let buffer = Buffer(pixels * self.n_samples * stride)
            buffer.fill_zero()
            let turns = Buffer(max(pixels, 1))
            turns.fill_zero()
            self.model = buffer
            self.cursor = turns
            self.channels = cn
            self.pixels = pixels
            self.frames = 0
        }
        let model = self.model
        let cursor = self.cursor
        if model is null or cursor is null {
            throw RuntimeError("the background model went missing")
        }

        self.frames += 1
        var rate = learning_rate
        if rate < 0.0 { rate = 1.0 / float(min(self.frames, self.history)) }

        let mask = Mat.zeros(frame.rows, frame.cols, CV_MAKETYPE(CV_8U, 1))
        if pixels == 0 { return mask }
        kernel_for("bg_knn").dispatch_async(
            [frame.storage, model, mask.storage, cursor],
            pixels,
            [
                pixels,
                cn,
                self.n_samples,
                frame.offset,
                frame.step_,
                frame.cols,
                f32_bits(rate),
                f32_bits(self.dist2_threshold),
                self.k_nn,
            ]
        )
        mask.invalidate()
        return mask
    }

    #: Throw the model away and start again.
    pub fn clear(self) -> void {
        let model = self.model
        if model is not null { model.free() }
        let cursor = self.cursor
        if cursor is not null { cursor.free() }
        self.model = null
        self.cursor = null
        self.frames = 0
    }
}

#: Build a mixture-of-Gaussians subtractor, the way OpenCV's factory reads.
pub fn create_background_subtractor_mog2(
    history: int = 500,
    var_threshold: float = 16.0,
    detect_shadows: bool = true
) -> BackgroundSubtractorMOG2 {
    return BackgroundSubtractorMOG2(history, var_threshold, detect_shadows)
}

#: Build a nearest-neighbour subtractor, the way OpenCV's factory reads.
pub fn create_background_subtractor_knn(
    history: int = 500,
    dist2_threshold: float = 400.0,
    detect_shadows: bool = true
) -> BackgroundSubtractorKNN {
    return BackgroundSubtractorKNN(history, dist2_threshold, detect_shadows)
}
