#: Everything between a YOLOv8 ONNX graph and boxes drawn on a picture.
#:
#: Split out because the still-image and camera front ends need exactly the
#: same three steps -- letterbox in, decode out, draw -- and the decoding is
#: the part that is easy to get subtly wrong.
#:
#: The output layout is worth knowing before reading `decode`. YOLOv8 gives
#: 8400 candidate boxes, one per cell of three feature-map strides, and 84
#: rows: rows 0-3 are the box as centre-x, centre-y, width and height in
#: LETTERBOXED pixels, and rows 4-83 are one score per COCO class. There is no
#: separate objectness row -- v5 had one, v8 does not -- so the class score IS
#: the confidence, and no sigmoid is needed because the exported graph has
#: already applied it.
import std.gpu as gpu
import jaicv as cv
import jaitensor as jt
from jaitensor import Tensor
from jaicv import Mat, Point, Rect, Scalar, Size
from jaicv.dnn.graph import run_graph
from jaicv.dnn.plan import run_best
from jaicv.dnn.onnx import read_net_from_onnx
from jaicv.dnn.graph import Graph
from std.io import Path

from .coco import COCO_NAMES

pub let INPUT = 640
pub let SCORE_THRESHOLD = 0.25
pub let NMS_THRESHOLD = 0.45

#: A detection in the coordinates of the ORIGINAL image, not the letterboxed
#: one the network saw.
pub class Detection {
    pub var box: Rect
    pub var score: float
    pub var class_id: int

    pub fn init(self, box: Rect, score: float, class_id: int) {
        self.box = box
        self.score = score
        self.class_id = class_id
    }

    pub fn label(self) -> str {
        if self.class_id < 0 or self.class_id >= COCO_NAMES.len() {
            return f"class {self.class_id}"
        }
        return COCO_NAMES[self.class_id]
    }
}

#: Resize to fit inside `INPUT` square and pad the remainder, keeping the
#: aspect ratio. Returns the padded image, the scale it was shrunk by and the
#: padding, because undoing all three is what puts a box back on the original.
#:
#: Squashing to 640x640 instead would work and detect slightly worse: the
#: network was trained on letterboxed images, so a stretched one is off its
#: training distribution.
pub fn letterbox(image: Mat) -> tuple[Mat, float, int, int] {
    let scale = min(float(INPUT) / float(image.cols), float(INPUT) / float(image.rows))
    let width = max(1, int(float(image.cols) * scale))
    let height = max(1, int(float(image.rows) * scale))
    let resized = cv.resize(image, Size(width, height))
    let left = (INPUT - width) // 2
    let top = (INPUT - height) // 2
    #: 114 grey is what Ultralytics pads with, and the network has seen it in
    #: training; black would be a colour it reads as content.
    let padded = cv.copy_make_border(
        resized,
        top,
        INPUT - height - top,
        left,
        INPUT - width - left,
        cv.BORDER_CONSTANT,
        Scalar(114.0, 114.0, 114.0)
    )
    return (padded, scale, left, top)
}

#: Turn the raw `1 x 84 x 8400` output into detections, undoing the letterbox.
#: `1 x 84 x 8400` of raw output into boxes.
#:
#: The best class for each of the 8400 cells is found on the device, not here.
#: Doing it in this loop meant eighty comparisons a cell -- two thirds of a
#: million of them -- and downloading all 705,600 numbers to make them. The
#: reduction leaves two arrays of 8400 and the four rows of box geometry, which
#: is a fourteenth of the numbers and none of the comparisons.
pub fn decode(
    output: Tensor,
    scale: float,
    pad_x: int,
    pad_y: int,
    width: int,
    height: int
) -> list[Detection] {
    let rows = 84
    let cells = output.size // rows
    #: Rows four onward are the class scores, and they are contiguous, so the
    #: view costs nothing. Transposed because the reduction runs along the
    #: last axis and the classes are along the first.
    let per_class = jt.Tensor([80, cells], storage: output.buffer().view(4 * cells, 80 * cells))
    let by_cell = jt.transpose(per_class)
    let best_scores = jt.amax(by_cell, axis: -1).download()
    let best_classes = jt.argmax(by_cell).download()
    let geometry = jt.Tensor([4, cells], storage: output.buffer().view(0, 4 * cells)).download()
    by_cell.free()

    var boxes: list[Rect] = []
    var scores: list[float] = []
    var classes: list[int] = []

    for cell in 0..cells {
        let best_score = best_scores[cell]
        if best_score < SCORE_THRESHOLD { continue }
        let best = int(best_classes[cell])

        let cx = geometry[cell]
        let cy = geometry[cells + cell]
        let w = geometry[2 * cells + cell]
        let h = geometry[3 * cells + cell]

        #: Out of letterboxed pixels, back into the original's.
        let x0 = (cx - w * 0.5 - float(pad_x)) / scale
        let y0 = (cy - h * 0.5 - float(pad_y)) / scale
        let bw = w / scale
        let bh = h / scale

        let left = max(0, int(x0))
        let top = max(0, int(y0))
        let right = min(width, int(x0 + bw))
        let bottom = min(height, int(y0 + bh))
        if right <= left or bottom <= top { continue }

        boxes.push(Rect(left, top, right - left, bottom - top))
        scores.push(best_score)
        classes.push(best)
    }

    #: Non-maximum suppression per class, which is what OpenCV's own YOLO
    #: samples do: two different objects can overlap heavily and both be
    #: right, where two boxes on the same object should not both survive.
    let kept = cv.nms_boxes_batched(boxes, scores, classes, SCORE_THRESHOLD, NMS_THRESHOLD)
    var found: list[Detection] = []
    for index in kept { found.push(Detection(boxes[index], scores[index], classes[index])) }
    return found
}

#: One colour per class, built once.
#:
#: This used to make a 1x1 `Mat` and run `cvt_color` on it for every box of
#: every frame -- a GPU round trip per detection, which measured at most of
#: the 24 ms `draw` was taking. The conversion is eight lines of arithmetic,
#: so it happens here on the CPU, once, into a table.
#:
#: 67 is coprime with the 180 hues OpenCV's HSV uses, so consecutive classes
#: land far apart on the wheel and all eighty get a distinct one. A smaller
#: stride collides: 37 put `person` and `bus` five degrees apart and drew
#: every box the same red.
fn build_palette() -> list[Scalar] {
    var table: list[Scalar] = []
    for class_id in 0..COCO_NAMES.len() {
        let hue = float(class_id * 67 % 180) * 2.0
        let saturation = 0.86
        let value = 255.0
        let sector = hue / 60.0
        let index = int(sector) % 6
        let frac = sector - float(int(sector))
        let p = value * (1.0 - saturation)
        let q = value * (1.0 - saturation * frac)
        let t = value * (1.0 - saturation * (1.0 - frac))
        var red = value
        var green = t
        var blue = p
        if index == 1 {
            red = q
            green = value
            blue = p
        } elif index == 2 {
            red = p
            green = value
            blue = t
        } elif index == 3 {
            red = p
            green = q
            blue = value
        } elif index == 4 {
            red = t
            green = p
            blue = value
        } elif index == 5 {
            red = value
            green = p
            blue = q
        }
        table.push(Scalar(blue, green, red))
    }
    return table
}

let PALETTE = build_palette()

fn colour_for(class_id: int) -> Scalar {
    if class_id < 0 or class_id >= PALETTE.len() { return Scalar(200.0, 200.0, 200.0) }
    return PALETTE[class_id]
}

pub fn draw(image: Mat, found: list[Detection]) -> Mat {
    for hit in found {
        let colour = colour_for(hit.class_id)
        cv.rectangle(
            image,
            Point(hit.box.x, hit.box.y),
            Point(hit.box.x + hit.box.width, hit.box.y + hit.box.height),
            colour,
            2
        )
        let caption = f"{hit.label()} {hit.score:.2f}"
        let (size, baseline) = cv.get_text_size(caption, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        #: The caption sits above the box when there is room and inside it
        #: when the box is against the top edge.
        let text_top = if hit.box.y - size.height - baseline >= 0 {
            hit.box.y - size.height - baseline
        } else {
            hit.box.y
        }
        cv.rectangle(
            image,
            Point(hit.box.x, text_top),
            Point(hit.box.x + size.width + 4, text_top + size.height + baseline),
            colour,
            -1
        )
        cv.put_text(
            image,
            caption,
            Point(hit.box.x + 2, text_top + size.height),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            Scalar(20.0, 20.0, 20.0),
            1
        )
    }
    return image
}

#: A detection that has been started and not yet read.
#:
#: Everything up to the network's output is queued work; nothing waits for the
#: device until `finish` reads the result. Between the two calls the caller can
#: get on with the previous frame -- drawing its boxes, packing it for the
#: window -- and that work runs on the processor while the network runs on the
#: device. A loop written that way costs the slower of the two halves rather
#: than their sum.
pub class Pending {
    pub let output: Tensor
    pub let scale: float
    pub let pad_x: int
    pub let pad_y: int
    pub let width: int
    pub let height: int

    pub fn init(
        self,
        output: Tensor,
        scale: float,
        pad_x: int,
        pad_y: int,
        width: int,
        height: int
    ) {
        self.output = output
        self.scale = scale
        self.pad_x = pad_x
        self.pad_y = pad_y
        self.width = width
        self.height = height
    }
}

#: Queue a frame's network work and come back for it.
#:
#: Nothing here waits: `run_best` hands the graph to the device and returns,
#: and the result is not read until `finish_detect`. A loop that has processor
#: work of its own -- drawing the frame before this one, writing it out -- can
#: do it in between and pay the slower of the two rather than their sum.
pub fn start_detect(graph: Graph, image: Mat) -> Pending {
    let (padded, scale, pad_x, pad_y) = letterbox(image)
    #: `blob_from_image` does the BGR-to-RGB swap and the 1/255 scaling the
    #: network expects, and lays the result out as N x C x H x W.
    let blob = cv.blob_from_image(
        padded,
        1.0 / 255.0,
        Size(INPUT, INPUT),
        Scalar(0.0, 0.0, 0.0),
        true
    )
    #: The blob is already on the device, so the tensor shares its storage
    #: rather than making a round trip through the host to say so.
    let input = cv.blob_to_tensor(blob, 3, Size(INPUT, INPUT))
    #: `run_best` interprets the first frame and compiles the model after,
    #: which for a run of more than one frame is most of the speed.
    let outputs = run_best(graph, {graph.inputs[0]: input})
    #: Committed here so the GPU starts on it now rather than whenever the
    #: caller next happens to touch the device. Nothing waits: a flush hands
    #: the queue over and returns, which is the whole point of coming back for
    #: the answer later.
    gpu.flush()
    return Pending(outputs[graph.outputs[0]], scale, pad_x, pad_y, image.cols, image.rows)
}

#: Read what `start_detect` queued. This is where the wait happens.
pub fn finish_detect(pending: Pending) -> list[Detection] {
    return decode(
        pending.output,
        pending.scale,
        pending.pad_x,
        pending.pad_y,
        pending.width,
        pending.height
    )
}

#: One frame, from image to detections. The graph is passed in rather than
#: reopened, because loading the 12 MB of weights costs about two seconds and
#: a camera loop must do it once.
#:
#: This waits for the network before returning. A loop that has other work to
#: do meanwhile should use `start_detect` and `finish_detect` instead.
pub fn detect_frame(graph: Graph, image: Mat) -> list[Detection] {
    return finish_detect(start_detect(graph, image))
}

#: The model, from `YOLO_MODEL` or from beside this file.
pub fn open_model(override: str?) -> Graph? {
    let path = if override is not null {
        Path(override)
    } else {
        Path("examples/yolo_detect/yolov8n.onnx")
    }
    if not path.exists() {
        print(f"no model at {path.to_str()}")
        print("run examples/yolo_detect/fetch_model.sh, or set YOLO_MODEL to one")
        return null
    }
    return read_net_from_onnx(path.to_str())
}
