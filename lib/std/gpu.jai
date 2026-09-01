#: Compute on the GPU — device buffers, Metal kernels, and the handful of
#: vector operations most programs actually want — with a correct CPU
#: implementation of every operation for when there is no device.
#:
#: `is_available` reports whether a device was found; `Buffer` and `Kernel`
#: raise `RuntimeError` without one, but every operation below picks CPU or
#: GPU itself and never does.
#:
#: A device buffer holds 32-bit floats. Mixed precision (`set_mixed_precision`)
#: runs GEMM and fused MLP graphs in float16 and writes float32; storage is
#: still float32. `upload` and `download` round through float32. The CPU path
#: computes in double precision with Neumaier compensation and is *more*
#: accurate, not just different — never compare the two paths for exact
#: equality, only with a tolerance such as `math.approx_eq`.
#:
#: Call `set_device` before the first buffer or kernel. Device memory is not
#: garbage collected; `free` a `Buffer` when you are done.
#:
#: The CPU/GPU crossover is roughly a hundred thousand elements for elementwise
#: work and far lower for `matmul`, whose CPU cost is cubic. `gpu_dispatch` runs
#: its command buffer to completion before returning, so nothing here needs a
#: separate fence.
#:
#: ```
#: vector_add([1.0, 2.0], [3.0, 4.0])    # [4.0, 6.0]
#: dot([1.0, 2.0], [3.0, 4.0])           # 11.0
#: reduce_sum([1.0, 2.0, 3.0])           # 6.0
#: ```
#:
#: Facade: imports and re-exports every sibling module in `std/gpu/`.
from std.gpu.device import FN_GPU_KERNEL, device_count, device_name, flush, is_available
from std.gpu.device import is_gpu_kernel, mixed_precision, set_device, set_mixed_precision
from std.gpu.device import synchronize
from std.gpu.kernel import Buffer, Kernel
from std.gpu.ops import conv2d_async, conv2d_data_grad_async, conv2d_weights_grad_async
from std.gpu.ops import labels_valid, matmul_async, mha_async, mlp3_bwd_step, mlp3_sgd_epoch
from std.gpu.ops import mlp3_sgd_step, mlp_bwd_step, mlp_sgd_epoch, mlp_sgd_step
from std.gpu.vector import MIN_GPU_ELEMENTS, dot, map_kernel, matmul, reduce_sum, vector_add
from std.gpu.vector import vector_mul

export { FN_GPU_KERNEL, device_count, device_name, flush, is_available }
export { is_gpu_kernel, mixed_precision, set_device, set_mixed_precision, synchronize }
export { Buffer, Kernel }
export { conv2d_async, conv2d_data_grad_async, conv2d_weights_grad_async, labels_valid }
export { matmul_async, mha_async, mlp3_bwd_step, mlp3_sgd_epoch, mlp3_sgd_step }
export { mlp_bwd_step, mlp_sgd_epoch, mlp_sgd_step }
export { MIN_GPU_ELEMENTS, dot, map_kernel, matmul, reduce_sum, vector_add, vector_mul }
