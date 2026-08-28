#: A frame pipeline, timed end to end rather than one operation at a time.
#:
#: Holdout for OCCUPANCY, not for any single kernel. Every other row in this
#: directory calls `gpu.synchronize()` on both sides of the thing it times, so
#: the host and the card take strict turns by construction and overlap has
#: nowhere to appear. Measured with the meter in gpu.m, `imgproc.jai` runs the
#: card at 0.45 occupancy with a pipeline depth of exactly 1.00 -- never two
#: command buffers in flight, whatever JAI_GPU_MAX_IN_FLIGHT says.
#:
#: This row is the opposite shape: many frames, no synchronise between stages,
#: and a host-side stage in the middle that the card cannot help with. If
#: anything in this tree can show the two working at once, it is this.
import jaicv as cv
import std.gpu as gpu
from jaicv import Mat
from jaicv.imgproc.color import COLOR_BGR2GRAY, cvt_color
from jaicv.imgproc.filter import gaussian_blur, sobel
from jaicv.imgproc.threshold import THRESH_BINARY, threshold
from jaicv.core.stats import sum_mat
from std.os import env_or
from std.time import Instant

let LEVEL = env_or("BENCH_LEVEL", "hard")
let SHRINK = if LEVEL == "easy" { 8 } elif LEVEL == "medium" { 4 } else { 1 }
let WIDTH = 1920 // SHRINK
let HEIGHT = 1080 // SHRINK
let FRAMES = max(120 // SHRINK, 4)

#: A small pool of distinct frames, cycled. A frame a piece would be six
#: gigabytes at 1080p; the operations allocate a fresh Mat per call and cache
#: nothing on the input, so cycling costs the benchmark nothing.
#:
#: Blocky rather than smooth: the point of the pipeline is that `sobel` finds
#: edges and `threshold` keeps some of them. A smooth ramp differentiates to
#: nearly zero, the mask comes out empty, and the checksum can no longer tell
#: a broken run from a working one -- which is exactly what the first version
#: of this file did.
let POOL = 8

fn frames() -> list[Mat] {
    var out: list[Mat] = []
    for index in 0..POOL {
        var values: list[float] = []
        for cell in 0..WIDTH * HEIGHT * 3 {
            let pixel = cell // 3
            let y = pixel // WIDTH
            let x = pixel - y * WIDTH
            let block = ((x + index * 7) // 24 + (y + index * 5) // 24) % 2
            values.push(if block == 0 { 0.15 } else { 0.85 })
        }
        out.push(Mat.from_list(HEIGHT, WIDTH, cv.CV_32FC3, values))
    }
    return out
}

#: Four GPU stages with no wait between them. Returns the mask rather than a
#: number, so the caller decides WHEN to make the host look at it -- which is
#: the whole variable this benchmark exists to measure.
fn issue(frame: Mat) -> Mat {
    let gray = cvt_color(frame, COLOR_BGR2GRAY)
    let soft = gaussian_blur(gray, cv.Size(5, 5), 1.2)
    let edge = sobel(soft, -1, 1, 0, 3)
    let (_used, mask) = threshold(edge, 0.02, 1.0, THRESH_BINARY)
    return mask
}

#: The only synchronisation point in either variant: reading a device buffer
#: back to the host.
fn collect(mask: Mat) -> float { return sum_mat(mask).v0 }

fn stage(frame: Mat) -> float { return collect(issue(frame)) }

#: The ceiling. One frame of slack: issue frame N + 1 before reading frame N,
#: so the card has work queued for the whole time the host spends on the
#: previous result. This is what examples/yolo_detect/live.jai does by hand
#: with start_detect/finish_detect, measured there at 5.1 ms pipelined against
#: 8.4 ms serial. It is NOT a shippable result -- it is the number an
#: orchestrator would have to earn automatically.
fn pipelined(batch: list[Mat], count: int) -> float {
    var total = 0.0
    var inflight = issue(batch[0])
    for index in 1..count {
        let next = issue(batch[index % POOL])
        total = total + collect(inflight)
        inflight = next
    }
    return total + collect(inflight)
}

fn main() -> int {
    let batch = frames()

    #: Warm the pipeline caches and let the clock ramp, as the other rows do.
    for index in 0..min(4, POOL) { let _ = stage(batch[index]) }
    gpu.synchronize()

    #: One variant a process. Running both in one process made the first one
    #: absorb whatever warm-up was left and the ceiling read 1.42x, 0.96x and
    #: 1.04x on three consecutive runs of identical code. imgproc.jai splits
    #: its rows across processes for the same reason, and says so.
    let which = env_or("PIPELINE_VARIANT", "both")

    if which == "pipelined" {
        let at = Instant.now()
        let total = pipelined(batch, FRAMES)
        gpu.synchronize()
        let took = at.elapsed().to_seconds()
        print(f"jtb\tpipeline-2deep\t{took / float(FRAMES)}\t0\t{WIDTH * HEIGHT * 3}\tok\t{total}")
        return 0
    }
    if which == "serial" {
        let at = Instant.now()
        var total = 0.0
        for index in 0..FRAMES { total = total + stage(batch[index % POOL]) }
        gpu.synchronize()
        let took = at.elapsed().to_seconds()
        print(f"jtb\tpipeline\t{took / float(FRAMES)}\t0\t{WIDTH * HEIGHT * 3}\tok\t{total}")
        return 0
    }

    let serial_at = Instant.now()
    var serial_sum = 0.0
    for index in 0..FRAMES { serial_sum = serial_sum + stage(batch[index % POOL]) }
    gpu.synchronize()
    let serial = serial_at.elapsed().to_seconds()

    let piped_at = Instant.now()
    let piped_sum = pipelined(batch, FRAMES)
    gpu.synchronize()
    let piped = piped_at.elapsed().to_seconds()

    #: The two variants must agree, or the pipelined one is reading a buffer
    #: the card has not finished writing.
    if serial_sum != piped_sum {
        print(f"MISMATCH serial={serial_sum} pipelined={piped_sum}")
        return 1
    }
    print(f"jtb\tpipeline\t{serial / float(FRAMES)}\t0\t{WIDTH * HEIGHT * 3}\tok\t{serial_sum}")
    print(f"jtb\tpipeline-2deep\t{piped / float(FRAMES)}\t0\t{WIDTH * HEIGHT * 3}\tok\t{piped_sum}")
    print(f"#  ceiling {serial / piped}x")
    return 0
}
