import std.gpu as gpu
from std.test import assert_eq

@gpu_kernel
fn marked_kernel() -> int {
    return 1
}

@mps_graph
fn marked_graph() -> int {
    return 2
}

fn plain() -> int {
    return 3
}

fn test_gpu_kernel_decorator_sets_flag() -> void {
    assert gpu.is_gpu_kernel(marked_kernel)
    assert gpu.is_gpu_kernel(marked_graph)
    assert not gpu.is_gpu_kernel(plain)
    assert_eq(gpu.FN_GPU_KERNEL, 1024)
    assert_eq(marked_kernel(), 1)
    assert_eq(marked_graph(), 2)
}
