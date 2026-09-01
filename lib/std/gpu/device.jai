#: Which compute device the process talks to, and the queue its dispatches
#: land on.

#: Report whether a compute device was found.
pub fn is_available() -> bool { return __prim__.gpu_available() }

#: Return the device name, or "none" when there is no device.
#:
#: Meant for logs and benchmark headers, never for a decision — match on
#: `is_available` instead of on a name.
pub fn device_name() -> str {
    if not is_available() { return "none" }
    return __prim__.gpu_device_name()
}

#: How many Metal devices the process can see.
pub fn device_count() -> int { return __prim__.gpu_device_count() }

#: Select the Metal device used by later allocations. Must be called before
#: the first buffer or kernel is created.
pub fn set_device(index: int) -> void {
    __prim__.gpu_set_device(index)
}

#: Compute GEMM and fused MLP graphs in float16, storing results in float32.
pub fn set_mixed_precision(enabled: bool) -> void {
    __prim__.gpu_set_mixed_precision(enabled)
}

#: Whether mixed-precision compute is currently enabled.
pub fn mixed_precision() -> bool { return __prim__.gpu_mixed_precision() }

#: `@gpu_kernel` / `@mps_graph` function flag (see `jaithon.compile.emit`).
pub const FN_GPU_KERNEL: int = 1024

#: Whether `target` was compiled with `@gpu_kernel` or `@mps_graph`.
pub fn is_gpu_kernel(target: any) -> bool {
    return __prim__.function_has_flag(target, FN_GPU_KERNEL)
}

pub fn _require_device(action: str) -> void {
    if not is_available() {
        throw RuntimeError(f"cannot {action}: no GPU device is available")
    }
}

#: Commit queued async kernels so the GPU can start them, without waiting.
#:
#: Returns immediately when no work is queued. A bounded number of command
#: buffers may stay in flight; this call waits only when that bound is hit,
#: and only for the oldest already-running buffer.
pub fn flush() -> void {
    _require_device("flush")
    __prim__.gpu_flush()
}

#: Wait until every asynchronously dispatched kernel has completed.
#:
#: Returns immediately when no work is queued. A failed command raises
#: `RuntimeError`; after this returns, its output buffers are safe to read.
pub fn synchronize() -> void {
    _require_device("synchronize")
    __prim__.gpu_synchronize()
}
