#: YOLOv8 on the camera, live.
#:
#:     examples/yolo_detect/fetch_model.sh          # once, ~12 MB
#:     jaithon run examples/yolo_detect/live.jai
#:
#: Optional arguments, in any order:
#:
#:     --camera N     which device, default 0
#:     --video FILE   a recorded clip instead of the camera
#:     --every N      run the network every Nth frame, default 1
#:     --record FILE  also write an AVI of what is shown
#:     --headless     do not open a window (for --video with --record)
#:
#: On an M2 Max the network takes about 200 ms a frame, so detecting on every
#: frame gives roughly 5 fps. `--every 3` runs it on one frame in three and
#: keeps drawing the last boxes on the two between, which looks far smoother
#: because the camera keeps delivering at its own rate -- the boxes are then up
#: to two frames stale, which for anything moving at human speed is invisible.
#: Press Esc or q to stop.
import jaicv as cv
from jaicv import Mat, Point, Scalar, Size
from jaicv.videoio.writer import VideoWriter
from std.os import argv, env
from std.time import Instant

from .detector import Detection, detect_frame, draw, open_model

class Options {
    pub var camera: int
    pub var video: str?
    pub var every: int
    pub var record: str?
    pub var headless: bool

    pub fn init(self) {
        self.camera = 0
        self.video = null
        self.every = 1
        self.record = null
        self.headless = false
    }
}

fn parse(args: list[str]) -> Options {
    var options = Options()
    var index = 1
    while index < args.len() {
        let flag = args[index]
        let value = if index + 1 < args.len() { args[index + 1] } else { "" }
        if flag == "--headless" {
            options.headless = true
            index += 1
        } elif flag == "--camera" and value != "" {
            options.camera = int(value)
            index += 2
        } elif flag == "--video" and value != "" {
            options.video = value
            index += 2
        } elif flag == "--every" and value != "" {
            options.every = max(1, int(value))
            index += 2
        } elif flag == "--record" and value != "" {
            options.record = value
            index += 2
        } else {
            throw ValueError(f"unrecognised argument {flag}")
        }
    }
    return options
}

#: A running mean over the last few frames. A single frame's time swings
#: enough that an instantaneous figure is unreadable.
class Smoothed {
    var samples: list[float]
    var at: int

    pub fn init(self, window: int) {
        self.samples = [0.0 for _slot in 0..window]
        self.at = 0
    }

    pub fn push(self, value: float) -> void {
        self.samples[self.at % self.samples.len()] = value
        self.at += 1
    }

    pub fn mean(self) -> float {
        let count = min(self.at, self.samples.len())
        if count == 0 { return 0.0 }
        var total = 0.0
        for index in 0..count { total += self.samples[index] }
        return total / float(count)
    }
}

fn banner(frame: Mat, fps: float, inference: float, found: int, every: int) -> void {
    let text = if every > 1 {
        f"{fps:.1f} fps   net {inference:.0f}ms every {every}   {found} objects"
    } else {
        f"{fps:.1f} fps   net {inference:.0f}ms   {found} objects"
    }
    let (size, baseline) = cv.get_text_size(text, cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv.rectangle(
        frame,
        Point(0, 0),
        Point(size.width + 16, size.height + baseline + 12),
        Scalar(0.0, 0.0, 0.0),
        -1
    )
    cv.put_text(
        frame,
        text,
        Point(8, size.height + 6),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        Scalar(80.0, 240.0, 120.0),
        2
    )
}

fn main() -> int {
    var options = Options()
    try {
        options = parse(argv())
    } catch failure {
        print(failure.message)
        print(
            "usage: jaithon run examples/yolo_detect/live.jai [--camera N] [--every N] [--record FILE]"
        )
        return 2
    }

    let graph = open_model(env("YOLO_MODEL"))
    if graph is null { return 1 }
    print(f"model: {graph.nodes.len()} nodes, {graph.initializers.len()} weights")

    #: `VideoCapture` takes a path as readily as a device index, so a
    #: recorded clip runs the identical loop -- which is how this gets tested
    #: somewhere without a camera attached.
    let clip = options.video
    let camera = if clip is not null {
        cv.VideoCapture(clip)
    } else {
        cv.VideoCapture(options.camera)
    }
    if not camera.is_opened() {
        let what = if clip is not null { clip } else { f"camera {options.camera}" }
        print(f"{what} failed to open: {camera.why()}")
        if clip is null {
            #: On macOS the first run has to be granted access, and the
            #: prompt only appears for an application that asks.
            print(cv.PERMISSION_HELP)
        }
        return 1
    }

    var writer: VideoWriter? = null
    var frames = 0
    var fps = Smoothed(15)
    var net_ms = 0.0
    var found: list[Detection] = []
    var last = Instant.now()

    print("running -- Esc or q to stop")
    loop {
        let (ok, frame) = camera.read()
        if not ok { break }

        if frames % options.every == 0 {
            let clock = Instant.now()
            found = detect_frame(graph, frame)
            net_ms = clock.elapsed().to_seconds() * 1000.0
        }
        draw(frame, found)

        let elapsed = last.elapsed().to_seconds()
        last = Instant.now()
        if elapsed > 0.0 { fps.push(1.0 / elapsed) }
        banner(frame, fps.mean(), net_ms, found.len(), options.every)

        let target = options.record
        if target is not null {
            if writer is null {
                #: Opened from the first frame, because that is when the
                #: camera's real size is known -- asking before it delivers
                #: one gives whatever default the device reports.
                writer = VideoWriter(
                    target,
                    cv.fourcc("M", "J", "P", "G"),
                    15.0,
                    Size(frame.cols, frame.rows)
                )
            }
            writer.write(frame)
        }

        if not options.headless {
            cv.imshow("yolo", frame)
            let key = cv.wait_key(1)
            if key == 27 or key == 113 { break }
        }
        frames += 1
    }

    if writer is not null {
        writer.release()
        print(f"wrote {options.record}")
    }
    camera.release()
    if not options.headless { cv.destroy_all_windows() }
    print(f"{frames} frames, {fps.mean():.1f} fps")
    return 0
}
