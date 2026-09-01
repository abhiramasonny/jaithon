#: The generator an estimator's random_state names, and k-means++ seeding.
import jaitensor as jt

from jaitensor import Tensor
from std.random import Random

from .kernels import base_kernel
from .params import check_matrix
from .reduce import copy_into

#: `k` starting centres spread out by k-means++, as `[k, n_features]`.
#:
#: The first is uniform; each after it is drawn with probability proportional
#: to its squared distance from the nearest centre already chosen. Distances
#: and draw are both dispatches, so the host loop is over `k` and nothing else.
#: Clustering and the Gaussian mixtures both start here.
pub fn kmeanspp_seed(x: Tensor, k: int, seed: int = 0) -> Tensor {
    check_matrix(x)
    let rows = x.shape[0]
    let width = x.shape[1]
    if k <= 0 or k > rows { throw ValueError(f"{rows} samples cannot make {k} centres") }
    let centres = Tensor([k, width])
    let rng = rng_for(seed)
    copy_row(x, centres, rng.int_range(0, rows), 0, width)
    let nearest = Tensor([rows])
    let keys = Tensor([rows])
    let distance = base_kernel("base_seed_distance")
    let key_step = base_kernel("base_seed_keys")
    let salt = device_seed(seed, 0x5DEECE66)
    distance.dispatch_async(
        [x.buffer(), centres.buffer(), nearest.buffer()],
        rows,
        [rows, width, 0, 1]
    )
    for centre in 1..k {
        key_step.dispatch_async([nearest.buffer(), keys.buffer()], rows, [rows, salt, centre])
        let chosen = int(peak_index(keys) + 0.5)
        copy_row(x, centres, chosen, centre, width)
        distance.dispatch_async(
            [x.buffer(), centres.buffer(), nearest.buffer()],
            rows,
            [rows, width, centre, 0]
        )
    }
    nearest.free()
    keys.free()
    return centres
}

#: Where the largest value of a vector sits.
#:
#: Through jaitensor's row reduction, which puts a threadgroup on the vector:
#: `row_extreme` over a one-row matrix would put a single thread on it, and a
#: single thread walking a hundred thousand samples is the mistake this module
#: exists to keep out of the estimators.
fn peak_index(v: Tensor) -> float {
    let row = v.view(0, [1, v.size])
    let (values, indices) = jt.max_with_index(row)
    let at = indices.download()[0]
    values.free()
    indices.free()
    row.free()
    return at
}

fn copy_row(source: Tensor, destination: Tensor, taken: int, placed: int, width: int) -> void {
    let src = source.view(taken * width, [width])
    let dst = destination.view(placed * width, [width])
    copy_into(src, dst, width)
    src.free()
    dst.free()
}

#: The generator an estimator's `random_state` names.
#:
#: Seed 0 means unseeded, and it means that everywhere in the package: the
#: device fills below draw their own seed from a fresh generator when they get
#: a 0, rather than inheriting `Buffer.fill_uniform`'s habit of turning it into
#: a constant and handing back the same numbers every run.
pub fn rng_for(seed: int) -> Random {
    if seed == 0 { return Random() }
    return Random.with_seed(seed)
}

fn mix_seed(seed: int, salt: int) -> int {
    return seed *% 6364136223846793 +% salt *% 1442695040888963 & 0xFFFFFFFF
}

fn device_seed(seed: int, salt: int) -> int {
    return mix_seed(if seed == 0 { Random().next_int() } else { seed }, salt)
}

#: A tensor of uniform draws in `[lo, hi)`, filled on the device.
pub fn random_uniform(shape: list[int], lo: float = 0.0, hi: float = 1.0, seed: int = 0) -> Tensor {
    let out = Tensor(shape)
    out.buffer().fill_uniform(lo, hi, device_seed(seed, 0x27D4EB2F))
    return out
}

#: A tensor of standard normal draws, Box-Muller over two uniform fills.
#:
#: The first fill starts just above zero rather than at it: `log(0)` is an
#: infinity, and one of those in an initial weight matrix is a model that never
#: trains.
pub fn random_normal(shape: list[int], seed: int = 0) -> Tensor {
    let out = Tensor(shape)
    let first = Tensor(shape)
    let second = Tensor(shape)
    first.buffer().fill_uniform(1.0e-7, 1.0, device_seed(seed, 0x9E3779B9))
    second.buffer().fill_uniform(0.0, 1.0, device_seed(seed, 0x85EBCA6B))
    base_kernel("base_box_muller").dispatch_async(
        [first.buffer(), second.buffer(), out.buffer()],
        out.size,
        [out.size]
    )
    first.free()
    second.free()
    return out
}
