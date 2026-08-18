#: Tensor-op recording for `@trace` / `std.trace` graph capture.
import std.trace as trace

fn _record(name: str, shape: list[int]) -> void {
    if trace.is_active() { trace.record_op(name, shape) }
}

#: Record a dense affine forward while a traced function is running.
pub fn record_dense_forward(x: list[int], weights: list[int]) -> void {
    _record("dense_forward", [x[0], x[1], weights[1]])
}

#: Record a dense backward while a traced function is running.
pub fn record_dense_backward(x: list[int], weights: list[int]) -> void {
    _record("dense_backward", [x[0], x[1], weights[1]])
}

#: Record sparse CE while a traced function is running.
pub fn record_sparse_ce(batch: int, classes: int) -> void {
    _record("sparse_ce", [batch, classes])
}
