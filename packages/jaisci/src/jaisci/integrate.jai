#: Quadrature and initial-value problems under scipy's names —
#: `scipy.integrate`.
#:
#: Host f64 throughout, for the reason `core.jai` states: an adaptive
#: quadrature is a sequential refinement over a handful of intervals and an
#: ODE step is a sequential recurrence, so neither has the width a device
#: wants. The INTEGRAND may be a device pass; what these promise is a bounded
#: number of calls to it.
#:
#: Sampled-data rules (`trapezoid`, `simpson`) take `list[float]` for the same
#: reason the rest of the package does: they are the small end. Bulk data that
#: is already an `NDArray` should be reduced there rather than brought back.
#:
#: ```
#: import jaisci as sci
#:
#: let (area, error) = sci.quad(fn(x: float) -> float { return math.exp(-x * x) }, 0.0, 1.0)
#: let path = sci.solve_ivp(decay, (0.0, 5.0), [1.0])
#: ```
import std.math
from std.core import Printable

#: The 15-point Kronrod extension of the 7-point Gauss rule, which is what
#: QUADPACK's QK15 uses and what `quad`'s error estimate is built on.
#:
#: The nodes are symmetric about zero, so only the non-negative half is
#: written; the last entry is the node at zero and is not mirrored. The Gauss
#: rule's nodes are every other Kronrod node, which is the whole point of the
#: construction: fifteen evaluations buy both rules, and their difference is
#: the error estimate. A separate 7-point rule would cost seven more calls for
#: the same information.
let KRONROD_X: list[float] = [
    0.991455371120813,
    0.949107912342759,
    0.864864423359769,
    0.741531185599394,
    0.586087235467691,
    0.405845151377397,
    0.207784955007898,
    0.0,
]

let KRONROD_W: list[float] = [
    0.022935322010529,
    0.063092092629979,
    0.104790010322250,
    0.140653259715525,
    0.169004726639267,
    0.190350578064785,
    0.204432940075298,
    0.209482141084728,
]

#: The Gauss weights, for the odd-indexed Kronrod nodes and the centre.
let GAUSS_W: list[float] = [
    0.129484966168870,
    0.279705391489277,
    0.381830050505119,
    0.417959183673469,
]

#: One QK15 pass over `[a, b]`: the value, the error estimate, and the
#: integral of `|f|` that the estimate is scaled against.
fn qk15(f: fn(float) -> float, a: float, b: float) -> tuple[float, float, float] {
    let centre = 0.5 * (a + b)
    let half = 0.5 * (b - a)

    let f_centre = f(centre)
    var kronrod = KRONROD_W[7] * f_centre
    var gauss = GAUSS_W[3] * f_centre
    var absolute = KRONROD_W[7] * math.fabs(f_centre)

    for index in 0..7 {
        let offset = half * KRONROD_X[index]
        let left = f(centre - offset)
        let right = f(centre + offset)
        let pair = left + right
        kronrod += KRONROD_W[index] * pair
        absolute += KRONROD_W[index] * (math.fabs(left) + math.fabs(right))
        #: Every other Kronrod node IS a Gauss node: 1, 3, 5 of the eight.
        if index % 2 == 1 { gauss += GAUSS_W[index // 2] * pair }
    }

    let value = kronrod * half
    let scale = absolute * half
    #: QUADPACK's estimate, which is the raw difference raised to the power
    #: 1.5 and scaled -- a heuristic that reflects how much faster the Kronrod
    #: rule converges than the difference alone suggests, and which is why a
    #: `quad` error is usually far smaller than `|K - G|`.
    var error = math.fabs((kronrod - gauss) * half)
    if scale != 0.0 and error != 0.0 {
        let ratio = 200.0 * error / scale
        let raised = if ratio < 1.0 { scale * ratio * math.sqrt(ratio) } else { scale }
        error = min(error, raised)
    }
    return (value, error, scale)
}

#: Adaptive Gauss–Kronrod quadrature, as `scipy.integrate.quad` does it.
#:
#: Global adaptive bisection: the interval with the largest error estimate is
#: split, both halves are re-estimated, and the loop stops when the total
#: estimate falls under `max(epsabs, epsrel * |value|)`. That is QUADPACK's
#: QAGS without its epsilon-algorithm extrapolation, which is the part that
#: handles endpoint singularities — a `1/sqrt(x)` on `[0, 1]` converges here
#: but takes far more subdivisions than scipy needs.
#:
#: Returns the value and the error estimate, which is scipy's `(y, abserr)`.
pub fn quad(
    f: fn(float) -> float,
    a: float,
    b: float,
    epsabs: float = 1.49e-08,
    epsrel: float = 1.49e-08,
    limit: int = 50
) -> tuple[float, float] {
    if a == b { return (0.0, 0.0) }
    #: A reversed interval is the negated integral, which is what every
    #: calculus convention and scipy both say.
    if a > b {
        let (flipped, error) = quad(f, b, a, epsabs: epsabs, epsrel: epsrel, limit: limit)
        return (-flipped, error)
    }

    var lows = [a]
    var highs = [b]
    let (first_value, first_error, _scale) = qk15(f, a, b)
    var values = [first_value]
    var errors = [first_error]
    var total = first_value
    var total_error = first_error

    var round = 0
    while round < limit {
        if total_error <= max(epsabs, epsrel * math.fabs(total)) { break }

        #: The worst interval, found by a scan: `limit` is fifty, and a heap
        #: over fifty entries costs more to maintain than the scan costs to
        #: run.
        var worst = 0
        for index in 1..errors.len() {
            if errors[index] > errors[worst] { worst = index }
        }

        let low = lows[worst]
        let high = highs[worst]
        let middle = 0.5 * (low + high)
        #: The interval has shrunk to where the endpoints are the same float;
        #: no split can improve it and the loop would spin.
        if middle <= low or middle >= high { break }

        let (left_value, left_error, _left_scale) = qk15(f, low, middle)
        let (right_value, right_error, _right_scale) = qk15(f, middle, high)

        total += left_value + right_value - values[worst]
        total_error += left_error + right_error - errors[worst]

        highs[worst] = middle
        values[worst] = left_value
        errors[worst] = left_error
        lows.push(middle)
        highs.push(high)
        values.push(right_value)
        errors.push(right_error)
        round += 1
    }
    return (total, total_error)
}

#: A fixed `n`-point Gauss–Legendre rule, as `scipy.integrate.fixed_quad` does.
#:
#: Only the orders whose nodes are tabulated here, which are the ones worth
#: having: 2 through 5 for a cheap smooth integrand and 7 for the rule `quad`
#: is built on. Anything else should use `quad`, which chooses its own
#: subdivision — a high fixed order is almost never the right answer.
pub fn fixed_quad(f: fn(float) -> float, a: float, b: float, n: int = 5) -> float {
    var nodes: list[float] = []
    var weights: list[float] = []
    if n == 2 {
        nodes = [-0.5773502691896257, 0.5773502691896257]
        weights = [1.0, 1.0]
    } elif n == 3 {
        nodes = [-0.7745966692414834, 0.0, 0.7745966692414834]
        weights = [0.5555555555555556, 0.8888888888888888, 0.5555555555555556]
    } elif n == 4 {
        nodes = [-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526]
        weights = [0.34785484513745385, 0.6521451548625462, 0.6521451548625462, 0.34785484513745385]
    } elif n == 5 {
        nodes = [
            -0.906179845938664,
            -0.5384693101056831,
            0.0,
            0.5384693101056831,
            0.906179845938664,
        ]
        weights = [
            0.23692688505618942,
            0.4786286704993664,
            0.568888888888889,
            0.4786286704993664,
            0.23692688505618942,
        ]
    } elif n == 7 {
        nodes = [
            -0.9491079123427585,
            -0.7415311855993945,
            -0.4058451513773972,
            0.0,
            0.4058451513773972,
            0.7415311855993945,
            0.9491079123427585,
        ]
        weights = [
            0.12948496616887065,
            0.2797053914892766,
            0.3818300505051185,
            0.41795918367346885,
            0.3818300505051185,
            0.2797053914892766,
            0.12948496616887065,
        ]
    } else {
        throw ValueError(f"fixed_quad has no {n}-point rule; use quad")
    }

    let centre = 0.5 * (a + b)
    let half = 0.5 * (b - a)
    var total = 0.0
    for index in 0..nodes.len() {
        total += weights[index] * f(centre + half * nodes[index])
    }
    return total * half
}

#: The trapezoid rule over sampled data, as `numpy.trapezoid` does it.
#:
#: `x` is optional and defaults to unit spacing, which is scipy's `dx=1`.
pub fn trapezoid(y: list[float], x: list[float]? = null) -> float {
    if y.len() < 2 { return 0.0 }
    if x is null {
        var total = 0.0
        for index in 1..y.len() { total += 0.5 * (y[index] + y[index - 1]) }
        return total
    }
    let positions = x
    if positions.len() != y.len() {
        throw ValueError(f"trapezoid got {positions.len()} x values and {y.len()} y values")
    }
    var total = 0.0
    for index in 1..y.len() {
        total += 0.5 * (y[index] + y[index - 1]) * (positions[index] - positions[index - 1])
    }
    return total
}

#: Simpson's rule over sampled data, as `scipy.integrate.simpson` does it.
#:
#: Simpson needs pairs of intervals, so an even number of points leaves one
#: over, and what is done with it is the whole difference between
#: implementations. scipy closes it with a three-point Newton–Cotes correction,
#: `(h/12)(5 y[n-1] + 8 y[n-2] - y[n-3])`, which is exact for a quadratic and
#: therefore exact wherever Simpson's own rule is; a trapezoid there is not,
#: and on `[0, 1, 4, 9, 16, 25]` it gives 41.833 against the true 125/3.
#: Matching scipy costs one extra term and is worth it.
#:
#: Uniform spacing only. scipy's non-uniform Simpson is a different formula
#: with per-interval weights; a non-uniform sample should use `trapezoid`,
#: whose error on uneven data is honest rather than optimistic.
pub fn simpson(y: list[float], dx: float = 1.0) -> float {
    let n = y.len()
    if n < 2 { return 0.0 }
    if n == 2 { return 0.5 * dx * (y[0] + y[1]) }

    #: The largest even count of intervals, which is `n - 1` rounded down.
    let pairs = (n - 1) // 2
    var total = 0.0
    for index in 0..pairs {
        let base = index * 2
        total += y[base] + 4.0 * y[base + 1] + y[base + 2]
    }
    total = total * dx / 3.0
    if (n - 1) % 2 == 1 {
        total += dx * (5.0 * y[n - 1] + 8.0 * y[n - 2] - y[n - 3]) / 12.0
    }
    return total
}

#: The running trapezoid integral, as `scipy.integrate.cumulative_trapezoid`
#: with `initial=0` does it: `n` values, the first of them zero.
pub fn cumulative_trapezoid(y: list[float], x: list[float]? = null) -> list[float] {
    var out = [0.0]
    if y.len() < 2 { return out }
    let positions = x
    if positions is not null and positions.len() != y.len() {
        throw ValueError(
            f"cumulative_trapezoid got {positions.len()} x values and {y.len()} y values"
        )
    }
    var running = 0.0
    for index in 1..y.len() {
        let step = if positions is null { 1.0 } else { positions[index] - positions[index - 1] }
        running += 0.5 * (y[index] + y[index - 1]) * step
        out.push(running)
    }
    return out
}

# ------------------------------------------------------------------
# Initial value problems
# ------------------------------------------------------------------

#: What `solve_ivp` hands back: the times it stopped at and the state there.
pub class IVPResult: Printable {
    #: The times, in order, starting at `t_span[0]`.
    pub var t: list[float]
    #: The state at each of them, one list per time.
    pub var y: list[list[float]]
    #: Whether the integration reached the end of the span.
    pub var success: bool
    #: A short reason: "reached the end", "step underflowed", "max steps".
    pub var message: str
    #: Derivative evaluations taken.
    pub var nfev: int

    pub fn init(
        self,
        t: list[float],
        y: list[list[float]],
        success: bool = true,
        message: str = "reached the end",
        nfev: int = 0
    ) {
        self.t = t
        self.y = y
        self.success = success
        self.message = message
        self.nfev = nfev
    }

    pub fn to_str(self) -> str {
        return f"IVPResult(steps={self.t.len()}, success={self.success})"
    }

    fn __str__(self) -> str { return self.to_str() }
}

#: Dormand–Prince 5(4), which is `scipy.integrate.solve_ivp`'s `RK45`.
#:
#: Seven stages, of which the last is the next step's first (the FSAL
#: property), so a step costs six evaluations rather than seven. The fifth
#: order result is what propagates and the fourth order one is only there to
#: size the error, which is what "local extrapolation" means and what makes
#: the method fifth order in practice.
let DP_C: list[float] = [0.0, 0.2, 0.3, 0.8, 0.8888888888888888, 1.0, 1.0]

let DP_A: list[list[float]] = [
    [],
    [0.2],
    [0.075, 0.225],
    [0.9777777777777777, -3.7333333333333334, 3.5555555555555554],
    [
        2.9525986892242035,
        -11.595793324188385,
        9.822892851699436,
        -0.2908093278463649,
    ],
    [
        2.8462752525252526,
        -10.757575757575758,
        8.906422717743473,
        0.2784090909090909,
        -0.2735313036020583,
    ],
    [
        0.09114583333333333,
        0.0,
        0.44923629829290207,
        0.6510416666666666,
        -0.322376179245283,
        0.13095238095238096,
    ],
]

#: The fifth order weights, which are row six of `DP_A` -- the FSAL property
#: again: the propagating solution IS the last stage's argument.
let DP_B5: list[float] = [
    0.09114583333333333,
    0.0,
    0.44923629829290207,
    0.6510416666666666,
    -0.322376179245283,
    0.13095238095238096,
    0.0,
]

let DP_B4: list[float] = [
    0.08991319444444444,
    0.0,
    0.4534890685834082,
    0.6140625,
    -0.2715123820754717,
    0.08904761904761904,
    0.025,
]

fn state_axpy(y: list[float], scale: float, k: list[float]) -> list[float] {
    var out: list[float] = []
    for index in 0..y.len() { out.push(y[index] + scale * k[index]) }
    return out
}

#: Integrate `y' = f(t, y)` from `t_span[0]` to `t_span[1]`.
#:
#: The step controller is scipy's: an error norm scaled by
#: `atol + rtol * |y|` per component, a step multiplier of
#: `0.9 * err^(-1/5)` clamped to [0.2, 10], and a rejected step retried
#: without advancing. `max_step` bounds it above, which is what a caller wants
#: when the derivative is cheap and the output should be dense.
pub fn solve_ivp(
    f: fn(float, list[float]) -> list[float],
    t_span: tuple[float, float],
    y0: list[float],
    rtol: float = 1e-03,
    atol: float = 1e-06,
    max_step: float = 0.0,
    max_steps: int = 100000
) -> IVPResult {
    let (start, stop) = t_span
    let n = y0.len()
    if n == 0 { throw ValueError("solve_ivp needs at least one state variable") }

    var t = start
    var y = [value for value in y0]
    var times = [t]
    var states: list[list[float]] = [[value for value in y]]
    var calls = 0

    if stop == start { return IVPResult(times, states, true, "reached the end", calls) }
    let forward = stop > start
    let span = math.fabs(stop - start)
    let ceiling = if max_step > 0.0 { max_step } else { span }

    #: A first step from the derivative's own size, which is what scipy's
    #: `_select_initial_step` amounts to for the common case: big enough not to
    #: waste a hundred steps crawling, small enough that the controller can
    #: recover in one rejection.
    var k1 = f(t, y)
    calls += 1
    var scale = 0.0
    for index in 0..n {
        let size = math.fabs(k1[index]) / (atol + rtol * math.fabs(y[index]))
        if size > scale { scale = size }
    }
    var h = if scale > 0.0 { min(0.01 * span, 0.1 / scale) } else { 0.01 * span }
    if h <= 0.0 { h = span }
    h = min(h, ceiling)

    var steps = 0
    while steps < max_steps {
        let remaining = math.fabs(stop - t)
        if remaining <= 0.0 { break }
        if h > remaining { h = remaining }
        let signed = if forward { h } else { -h }

        var stages: list[list[float]] = [k1]
        for stage in 1..7 {
            var argument = [value for value in y]
            let row = DP_A[stage]
            for previous in 0..row.len() {
                argument = state_axpy(argument, signed * row[previous], stages[previous])
            }
            stages.push(f(t + signed * DP_C[stage], argument))
            calls += 1
        }

        var fifth = [value for value in y]
        var fourth = [value for value in y]
        for stage in 0..7 {
            fifth = state_axpy(fifth, signed * DP_B5[stage], stages[stage])
            fourth = state_axpy(fourth, signed * DP_B4[stage], stages[stage])
        }

        var error = 0.0
        for index in 0..n {
            let tolerance = atol + rtol * max(math.fabs(y[index]), math.fabs(fifth[index]))
            let size = math.fabs(fifth[index] - fourth[index]) / tolerance
            if size > error { error = size }
        }

        if error <= 1.0 {
            t += signed
            y = fifth
            times.push(t)
            states.push([value for value in y])
            #: FSAL: the seventh stage was evaluated at the step's end with the
            #: propagating solution, so it is the next step's first stage.
            k1 = stages[6]
            steps += 1
        }

        let factor = if error > 0.0 { max(0.2, min(10.0, 0.9 * error ** -0.2)) } else { 10.0 }
        h = min(h * factor, ceiling)
        if h <= 0.0 or h < 1e-14 * (1.0 + math.fabs(t)) {
            return IVPResult(times, states, false, "step underflowed", calls)
        }
    }

    let reached = math.fabs(stop - t) <= 1e-12 * (1.0 + math.fabs(stop))
    if reached { return IVPResult(times, states, true, "reached the end", calls) }
    return IVPResult(times, states, false, "max steps", calls)
}
