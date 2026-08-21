#: YOLOv8 object detection on a still image.
#:
#: The network is a real ONNX file exported from Ultralytics, imported by
#: `jaicv.dnn.onnx` and run by `jaicv.dnn.graph` on jaitensor's GPU tensors --
#: nothing here shells out to Python. `detector.jai` holds the part a model
#: file does not contain: turning `1 x 84 x 8400` of raw output into boxes.
#:
#: Get the model first (about 12 MB, deliberately not committed):
#:
#:     examples/yolo_detect/fetch_model.sh
#:
#: then
#:
#:     jaithon run examples/yolo_detect/detect.jai <image> [output.png]
#:
#: For the camera instead, see `live.jai` beside this.
import jaicv as cv
from std.os import argv, env
from std.time import Instant

from .detector import detect_frame, draw, open_model

fn main() -> int {
    let args = argv()
    if args.len() < 2 {
        print("usage: jaithon run examples/yolo_detect/detect.jai <image> [output.png]")
        return 2
    }
    let source = args[1]
    let target = if args.len() > 2 { args[2] } else { "detections.png" }

    var image = cv.imread(source)
    if image.empty() {
        print(f"could not read {source}")
        return 1
    }
    print(f"{source}: {image.cols}x{image.rows}")

    var clock = Instant.now()
    let graph = open_model(env("YOLO_MODEL"))
    if graph is null { return 1 }
    print(
        f"model: {graph.nodes.len()} nodes, {graph.initializers.len()} weights, loaded in {clock.elapsed().to_seconds() * 1000.0:.0f}ms"
    )

    clock = Instant.now()
    let found = detect_frame(graph, image)
    print(f"inference {clock.elapsed().to_seconds() * 1000.0:.0f}ms, {found.len()} detections")
    for hit in found {
        print(
            f"  {hit.label():<16} {hit.score:.3f}  at ({hit.box.x}, {hit.box.y}) {hit.box.width}x{hit.box.height}"
        )
    }

    cv.imwrite(target, draw(image, found))
    print(f"wrote {target}")
    return 0
}
