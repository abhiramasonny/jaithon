import std.trace as trace
from std.test import assert_eq
from jaithon.compile.emit import FN_TRACE

@trace
fn traced_add(a: int, b: int) -> int {
    return a + b
}

fn untraced_add(a: int, b: int) -> int {
    return a + b
}

@trace
fn traced_run_id() -> int {
    return trace.run_id()
}

@trace
fn traced_shaped_match(width: int) -> bool {
    trace.record_op("op", [width])
    return trace.is_replay()
}

@trace
fn traced_shaped_change(width: int) -> bool {
    trace.record_op("op", [width])
    return trace.is_replay()
}

class TracedCell {
    @trace
    pub fn value(self) -> int {
        return 7
    }
}

fn test_trace_decorator_sets_fn_trace_flag() -> void {
    assert trace.is_traced(traced_add)
    assert not trace.is_traced(untraced_add)
    assert_eq(traced_add(2, 3), 5)
    assert_eq(FN_TRACE, 512)
}

fn test_trace_run_id_increments() -> void {
    let first = traced_run_id()
    let second = traced_run_id()
    assert second >= first
}

fn test_trace_replays_matching_shapes() -> void {
    let first = traced_shaped_match(8)
    let second = traced_shaped_match(8)
    assert not first
    assert second
}

fn test_trace_invalidates_on_shape_change() -> void {
    assert not traced_shaped_change(3)
    assert not traced_shaped_change(5)
    assert not traced_shaped_change(5)
    assert traced_shaped_change(5)
}

fn test_trace_decorator_on_method() -> void {
    let cell = TracedCell()
    assert trace.is_traced(cell.value)
    assert_eq(cell.value(), 7)
}
