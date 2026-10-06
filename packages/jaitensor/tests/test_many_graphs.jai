import jaitensor as jt
from jaitensor import Tensor
from std.test import assert_near

#: From macOS 27 every MPSGraph executable encoded into a command buffer
#: attaches residency sets to it, and Metal aborts the process past 32 of them
#: ("command buffer residency set limit of 32 exceeded"). Forty convolutions of
#: forty shapes -- forty cached executables -- queued before anything is read is
#: the shape a layer-per-executable network such as YOLOv8 makes every frame.
fn test_forty_distinct_graphs_queue_without_a_read() -> void {
    if not jt.is_available() { return }
    var ws: list[float] = []
    for _i in 0..3 * 3 * 4 * 4 { ws.push(0.1) }
    let w = jt.Tensor([3, 3, 4, 4], ws)
    defer { w.free() }
    var outs: list[Tensor] = []
    for s in 4..44 {
        var xs: list[float] = []
        for _i in 0..s * s * 4 { xs.push(1.0) }
        let x = jt.Tensor([1, s, s, 4], xs)
        outs.push(jt.conv2d(x, w, padding: 1))
        x.free()
    }
    # An interior pixel sees nine taps of four channels at 0.1 each.
    let last = outs[39].download()
    let centre = (20 * 43 + 20) * 4
    assert_near(last[centre], 3.6, 1e-4)
    for t in outs { t.free() }
}
