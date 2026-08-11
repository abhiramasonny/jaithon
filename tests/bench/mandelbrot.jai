#: Mandelbrot membership over a grid, counting the points that never escape.
#:
#: `nbody` and `spectral` run their float work to a trip count fixed before the
#: loop starts. This one leaves as soon as a point escapes, so nearly every
#: pixel costs a different number of iterations and the branch that ends them
#: is unpredictable. Each step of the recurrence is written to one intermediate
#: at a time so that every port performs the same rounding in the same order.
from std.os import env_or

# The grid is width * height work, so each level halves both sides rather than
# dividing one of them by the whole scale.
let LEVEL = env_or("BENCH_LEVEL", "hard")
let SHRINK = if LEVEL == "easy" { 4 } elif LEVEL == "medium" { 2 } else { 1 }

fn main() -> int {
    let width = 3200 // SHRINK
    let height = 240 // SHRINK
    let limit = 100

    var inside = 0
    for py in 0..height {
        let y0 = float(py) * 2.0 / float(height) - 1.0
        for px in 0..width {
            let x0 = float(px) * 3.0 / float(width) - 2.0
            var x = 0.0
            var y = 0.0
            var i = 0
            while i < limit {
                let x2 = x * x
                let y2 = y * y
                if x2 + y2 > 4.0 { break }
                let xy = x * y
                y = 2.0 * xy + y0
                x = x2 - y2 + x0
                i += 1
            }
            if i == limit { inside += 1 }
        }
    }
    print(limit)
    print(inside)
    return 0
}
