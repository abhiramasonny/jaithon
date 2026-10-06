#: The command queue and the buffer pool under work that never waits.
#:
#: Three mechanisms meet here. The pool hands a small buffer back while its
#: previous owner's work may still be queued, because GPU work into it lands
#: behind that work anyway; a first host write into one that nothing has
#: touched trades it for a fresh MTLBuffer instead of waiting. A batch that
#: holds only kernels is committed every so many dispatches, and any batch is
#: committed once it passes a size cap. And a wait is bounded: a batch that
#: never finishes is reported as a RuntimeError carrying the reason rather than
#: sat in for ever.
import std.gpu
from std.io import Path, write_file
from std.os import Completed, argv, pid, run, temp_dir
from std.test import assert_eq, assert_true

let ADD_ONE = """#include <metal_stdlib>
using namespace metal;
kernel void add_one(device float* values [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    values[i] += 1.0f;
}
kernel void next_of(device const float* src [[buffer(0)]],
                    device float* dst [[buffer(1)]],
                    uint i [[thread_position_in_grid]]) {
    dst[i] = src[i] + 1.0f;
}
kernel void set_seven(device float* values [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    values[i] = 7.0f;
}
"""

#: A chain that never waits, each step writing a buffer the pool hands back
#: while the work of two steps earlier may still be reading it.
#:
#: If queue order ever stopped covering that reuse, a step would read a value
#: a later step had already overwritten and the chain would come out short.
fn test_a_chain_of_recycled_buffers_keeps_queue_order() -> void {
    if not gpu.is_available() { return }
    let next_of = gpu.Kernel.compile(ADD_ONE, "next_of")
    defer { next_of.free() }
    var current = gpu.Buffer.of([0.0, 10.0, 20.0, 30.0])
    for _step in 0..300 {
        let following = gpu.Buffer(4)
        next_of.dispatch_async([current, following], 4)
        current.free()
        current = following
    }
    gpu.synchronize()
    assert_eq(current.download(), [300.0, 310.0, 320.0, 330.0])
    current.free()
}

#: A host write into a recycled buffer whose previous owner's kernel has not
#: run yet. Nothing has touched it since it came back, so the write may land
#: on a fresh MTLBuffer instead of waiting -- either way the old kernel must
#: not land on top of it.
fn test_a_host_write_into_a_busy_recycled_buffer_is_not_overwritten() -> void {
    if not gpu.is_available() { return }
    let add_one = gpu.Kernel.compile(ADD_ONE, "add_one")
    defer { add_one.free() }
    for _round in 0..20 {
        let old = gpu.Buffer.of([1.0, 2.0, 3.0, 4.0, 5.0])
        add_one.dispatch_async([old], 5)
        old.free()
        let reused = gpu.Buffer(5)
        reused.upload([9.0, 9.0, 9.0, 9.0, 9.0])
        gpu.synchronize()
        assert_eq(reused.download(), [9.0, 9.0, 9.0, 9.0, 9.0])
        reused.free()
    }
}

#: Once the new owner has queued work against a recycled buffer, its bytes
#: are that work's, and a partial host write must keep the rest of them. A
#: buffer traded for a fresh one at that point would read back zero outside
#: the write.
fn test_a_partial_host_write_keeps_what_queued_work_wrote() -> void {
    if not gpu.is_available() { return }
    let add_one = gpu.Kernel.compile(ADD_ONE, "add_one")
    let set_seven = gpu.Kernel.compile(ADD_ONE, "set_seven")
    defer { add_one.free() }
    defer { set_seven.free() }
    for _round in 0..20 {
        let old = gpu.Buffer.of([1.0, 2.0, 3.0])
        add_one.dispatch_async([old], 3)
        old.free()
        let reused = gpu.Buffer(3)
        set_seven.dispatch_async([reused], 3)
        reused.upload([0.0], 1)
        gpu.synchronize()
        assert_eq(reused.download(), [7.0, 0.0, 7.0])
        reused.free()
    }
}

fn run_child(source: str, settings: str, tag: str) -> Completed {
    let path = f"{temp_dir()}/jaithon_gpu_queue_{tag}_{pid()}.jai"
    write_file(path, source)
    defer { Path(path).remove() }
    return run(["sh", "-c", f"{settings} exec '{argv()[0]}' run --no-cache '{path}'"])
}

fn child_program(body: list[str]) -> str {
    var src = "import std.gpu\n"
    src += "let SOURCE = \"#include <metal_stdlib>\\nusing namespace metal;\\n"
    src += "kernel void add_one(device float* v [[buffer(0)]], uint i [[thread_position_in_grid]]) { v[i] += 1.0f; }\\n\"\n"
    src += "fn main() -> int {\n"
    src += "    let kernel = gpu.Kernel.compile(SOURCE, \"add_one\")\n"
    for line in body { src += f"    {line}\n" }
    src += "    return 0\n}\n"
    return src
}

#: A wait that never ends is reported, not sat in. The child's first batch
#: waits on an event nobody signals (JAITHON_GPU_DEBUG_STALL), so its
#: synchronize has to give up after the timeout with a RuntimeError that says
#: why -- the shape of the run that once sat in waitUntilCompleted at 0% CPU
#: for five minutes.
fn test_a_stalled_queue_raises_after_the_timeout_instead_of_hanging() -> void {
    if not gpu.is_available() { return }
    let src = child_program([
        "let values = gpu.Buffer.of([1.0, 2.0])",
        "kernel.dispatch_async([values], 2)",
        "try { gpu.synchronize() } catch e: RuntimeError { print(f\"caught: {e.message}\") }",
    ])
    let ran = run_child(src, "JAITHON_GPU_DEBUG_STALL=1 JAITHON_GPU_WAIT_TIMEOUT=0.3", "stall")
    assert_eq(ran.exit_code, 0, ran.stderr)
    assert_true(ran.stdout.contains("caught: gpu_synchronize(): queued GPU work did not complete"),
                ran.stdout)
    assert_true(ran.stdout.contains("did not finish within 0.3 s"), ran.stdout)
}

#: A loop that never reads still has its work committed in bounded pieces.
#: With the kernel-only split off, only the size cap can commit, so the meter
#: counts one buffer per hundred dispatches -- and the answer is still exact
#: across every boundary.
fn test_the_batch_cap_commits_a_loop_that_never_waits() -> void {
    if not gpu.is_available() { return }
    let src = child_program([
        "let values = gpu.Buffer.of([0.0, 5.0])",
        "for _i in 0..1000 { kernel.dispatch_async([values], 2) }",
        "gpu.synchronize()",
        "print(values.download())",
    ])
    let ran = run_child(src, "JAITHON_GPU_METER=1 JAITHON_GPU_KERNEL_COMMIT=0 JAITHON_GPU_BATCH_CAP=100",
                        "cap")
    assert_eq(ran.exit_code, 0, ran.stderr)
    assert_true(ran.stdout.contains("[1000.0, 1005.0]"), ran.stdout)
    assert_true(ran.stderr.contains("buffers 10 "), ran.stderr)
}

#: The same loop with nothing capping it is one buffer, which is what the cap
#: is measured against -- and what the old default did to every unread loop.
fn test_without_the_cap_an_unread_loop_is_one_buffer() -> void {
    if not gpu.is_available() { return }
    let src = child_program([
        "let values = gpu.Buffer.of([0.0, 5.0])",
        "for _i in 0..1000 { kernel.dispatch_async([values], 2) }",
        "gpu.synchronize()",
        "print(values.download())",
    ])
    let ran = run_child(src, "JAITHON_GPU_METER=1 JAITHON_GPU_KERNEL_COMMIT=0 JAITHON_GPU_BATCH_CAP=0",
                        "nocap")
    assert_eq(ran.exit_code, 0, ran.stderr)
    assert_true(ran.stdout.contains("[1000.0, 1005.0]"), ran.stdout)
    assert_true(ran.stderr.contains("buffers 1 "), ran.stderr)
}

#: Kernel-only work is committed every 64 dispatches by default, so the host
#: and the device overlap and freed buffers settle for the pool.
fn test_a_kernel_only_loop_commits_every_sixty_four_dispatches() -> void {
    if not gpu.is_available() { return }
    let src = child_program([
        "let values = gpu.Buffer.of([0.0, 5.0])",
        "for _i in 0..640 { kernel.dispatch_async([values], 2) }",
        "gpu.synchronize()",
        "print(values.download())",
    ])
    let ran = run_child(src, "JAITHON_GPU_METER=1", "kernel")
    assert_eq(ran.exit_code, 0, ran.stderr)
    assert_true(ran.stdout.contains("[640.0, 645.0]"), ran.stdout)
    assert_true(ran.stderr.contains("buffers 10 "), ran.stderr)
}
