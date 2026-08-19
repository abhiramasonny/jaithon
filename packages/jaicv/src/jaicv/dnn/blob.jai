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
import std.math
from ..core.mat import Mat
from ..core.types import CV_32F, CV_MAKETYPE, Rect, Scalar, Size
from ..imgproc.geometric import resize

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

    var out: list[float] = []
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
        let values = prepared.to_list()
        for channel in 0..cn {
            #: The blob is planar, so one channel is written out whole before
            #: the next begins.
            var source = channel
            if swap_rb and cn >= 3 and (channel == 0 or channel == 2) { source = 2 - channel }
            let offset = mean[min(channel, 3)]
            for index in 0..width * height {
                out.push((values[index * cn + source] - offset) * scale_factor)
            }
        }
    }
    return Mat.from_list(images.len() * cn, width * height, CV_MAKETYPE(CV_32F, 1), out)
}

#: Take the images back out of a blob.
pub fn images_from_blob(blob: Mat, channels: int, size: Size) -> list[Mat] {
    let values = blob.to_list()
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

#: Keep the strongest of each group of overlapping boxes.
#:
#: Boxes scoring below `score_threshold` are dropped, the rest are taken in
#: order of score, and any box overlapping one already kept by more than
#: `nms_threshold` is dropped with it. Returns the indices kept, in the order
#: they were kept, which is what OpenCV's `NMSBoxes` returns.
pub fn nms_boxes(
    boxes: list[Rect],
    scores: list[float],
    score_threshold: float,
    nms_threshold: float,
    top_k: int = 0
) -> list[int] {
    if boxes.len() != scores.len() { throw ValueError("there has to be one score per box") }
    var order: list[int] = []
    for index in 0..boxes.len() {
        if scores[index] < score_threshold { continue }
        var at = order.len()
        while at > 0 and scores[index] > scores[order[at - 1]] { at -= 1 }
        order.insert(at, index)
    }

    var kept: list[int] = []
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
        if top_k > 0 and kept.len() >= top_k { break }
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
    var ordered: list[int] = []
    for index in kept {
        var at = ordered.len()
        while at > 0 and scores[index] > scores[ordered[at - 1]] { at -= 1 }
        ordered.insert(at, index)
    }
    return ordered
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
