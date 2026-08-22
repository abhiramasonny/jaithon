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
#: On an M2 Max a 720p frame costs about 5.6 ms all in, so detecting on every
#: frame runs at around 178 fps and the camera, not the detector, sets the
#: pace. The model is interpreted on the first frame and compiled after, so the
#: first one is slower than the rest by a wide margin.
#:
#: That number is the network and the imaging running at the same time rather
#: than one after the other: the loop hands the GPU frame N and then draws
#: frame N-1 on the processor while it works, so a frame costs the slower of
#: the two halves instead of their sum. The same loop written one step after
#: another measures about 7 ms. What is on screen is therefore one frame
#: behind the camera, which at this rate is under six milliseconds.
#:
#: `--every N` runs the network on one frame in N and keeps drawing the last
#: boxes on the ones between. It is not needed at this speed, but it is there
#: for a slower machine or a larger model: the boxes go up to N-1 frames stale,
#: which for anything moving at human speed is invisible.
#: Press Esc or q to stop.
import jaicv as cv
from jaicv import Mat, Point, Scalar, Size
from jaicv.videoio.writer import VideoWriter
from std.os import argv, env
from std.time import Instant

from .detector import Detection, Pending, draw, finish_detect, open_model, start_detect

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

    #: The frame the network is working on is not the frame being drawn. The
    #: one being drawn came from the read before, and its boxes are already in
    #: `found`.
    var showing: Mat? = null
    var stopped = false

    #: Draw, record and display one frame. Returns false when the viewer asked
    #: to stop.
    fn present(frame: Mat, boxes: list[Detection]) -> bool {
        draw(frame, boxes)

        let elapsed = last.elapsed().to_seconds()
        last = Instant.now()
        if elapsed > 0.0 { fps.push(1.0 / elapsed) }
        banner(frame, fps.mean(), net_ms, boxes.len(), options.every)

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

        if options.headless { return true }
        cv.imshow("yolo", frame)
        let key = cv.wait_key(1)
        return key != 27 and key != 113
    }

    print("running -- Esc or q to stop")
    loop {
        let (ok, frame) = camera.read()
        if not ok { break }

        #: Queued, not waited for. Everything between here and `finish_detect`
        #: runs on the processor while the GPU runs the network.
        var pending: Pending? = null
        var clock = Instant.now()
        if frames % options.every == 0 {
            pending = start_detect(graph, frame)
            net_ms = clock.elapsed().to_seconds() * 1000.0
        }

        let earlier = showing
        if earlier is not null {
            if not present(earlier, found) {
                stopped = true
                break
            }
        }

        if pending is not null {
            clock = Instant.now()
            found = finish_detect(pending)
            net_ms += clock.elapsed().to_seconds() * 1000.0
        }
        showing = frame
        frames += 1
    }

    #: One frame is always still in hand when the loop ends, and a recording
    #: that dropped it would be a frame short of what the camera gave.
    let leftover = showing
    if not stopped and leftover is not null { present(leftover, found) }

    if writer is not null {
        writer.release()
        print(f"wrote {options.record}")
    }
    camera.release()
    if not options.headless { cv.destroy_all_windows() }
    print(f"{frames} frames, {fps.mean():.1f} fps")
    return 0
}
