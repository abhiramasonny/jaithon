#: The earth mover's distance: `emd`.
#:
#: Two weighted point sets, and the cheapest schedule of mass that turns one
#: into the other. That is the transportation problem, and this solves it the
#: way OpenCV's `EMD` does — Russell's approximation for a first basic feasible
#: solution, then the transportation simplex on the dual variables until no
#: reduced cost is negative.
#:
#: It agrees with OpenCV to float32, and the gap is OpenCV's rather than
#: shared. Over 1794 random signature pairs of up to forty clusters, each also
#: solved exactly in float64 by an independent linear programme, this lands
#: within 3.9e-16 of that optimum and OpenCV 5.0.0 within 9.1e-6 — OpenCV
#: carries the costs and the flows in float32 and this carries them in float64,
#: and that is the whole of the difference. The two therefore disagree with
#: each other by at most 9.1e-6 relative, 2.8e-6 absolute, on the same set.
#:
#: The flow matrix is a different matter and is not compared against OpenCV's:
#: a transportation problem routinely has many optimal schedules, and which one
#: a simplex lands on depends on the pivot order. What is checked of the flow
#: here is what is actually defined — that it is feasible, and that its cost is
#: the distance.
#:
#: Three of OpenCV's conventions are worth naming because they surprise people:
#:
#: - The distance divides by the *larger* of the two total weights, not by the
#:   mass actually moved. Two signatures of one cluster each come out as the
#:   ground distance between them whatever they weigh.
#: - `cost` is read only when `dist_type` is `DIST_USER`. Passing one alongside
#:   `DIST_L2` is ignored, as it is by OpenCV 5.
#: - `lower_bound` of zero means the caller did not ask for the check. OpenCV
#:   compares the centre-of-mass distance against the caller's bound with `<=`,
#:   so a bound of zero would short-circuit every call; its C API distinguishes
#:   the two by a null pointer, and zero is that null here.
#:
#: ```
#: let first = Mat.from_list(2, 2, CV_32FC1, [0.4, 0.0, 0.6, 3.0])
#: let second = Mat.from_list(2, 2, CV_32FC1, [0.5, 1.0, 0.5, 2.0])
#: let (distance, _bound, flow) = emd(first, second, DIST_L2)
#: ```
import std.math
from ..core.mat import Mat
from ..core.types import CV_32F, CV_MAKETYPE
from .segmentation import DIST_C, DIST_L1, DIST_L2

#: The caller supplies the whole cost matrix, so there is no ground distance.
pub const DIST_USER = -1

#: How far apart two total weights may be and still count as equal. This is
#: OpenCV's `CV_EMD_EPS`, and it is loose because it is sized for the float32
#: the signatures arrive in.
let WEIGHT_EPSILON = 1e-5

#: A reduced cost this far below zero, relative to the largest cost, is real
#: rather than the residue of the dual variables being chained along a path of
#: the basis tree.
let REDUCED_EPSILON = 1e-12

#: Degenerate pivots move no flow, so the objective cannot separate the bases
#: they visit and the simplex is free to revisit one — the classic way a
#: transportation solver hangs. After this many in a row the pivot rule changes
#: to Bland's, which cannot cycle, and changes back as soon as flow moves
#: again. Dantzig's rule is kept for the common case because it is much faster.
#:
#: It is worth saying that the fallback has never actually been reached: across
#: the degenerate families the tests build, three quarters of the pivots move
#: no flow and Dantzig's rule still escapes every stall inside a handful of
#: steps. It is here for the input that does not, and what says the branch
#: works at all is that forcing it on for every pivot returns the same answer
#: on every case in the tests and on the random set the doc comment above
#: measures.
let STALL_LIMIT = 32

#: Nothing legitimate takes this many pivots; hitting it means the guard above
#: failed rather than that the problem is hard.
let MAX_PIVOTS = 1000000

#: The ground distance between two points, held as spans of `dims` inside
#: larger lists.
fn ground_distance(
    first: list[float],
    first_at: int,
    second: list[float],
    second_at: int,
    dims: int,
    dist_type: int
) -> float {
    if dist_type == DIST_L1 {
        var total = 0.0
        for k in 0..dims { total += math.fabs(first[first_at + k] - second[second_at + k]) }
        return total
    }
    if dist_type == DIST_L2 {
        var total = 0.0
        for k in 0..dims {
            let delta = first[first_at + k] - second[second_at + k]
            total += delta * delta
        }
        return math.sqrt(total)
    }
    if dist_type == DIST_C {
        var widest = 0.0
        for k in 0..dims {
            let delta = math.fabs(first[first_at + k] - second[second_at + k])
            if delta > widest { widest = delta }
        }
        return widest
    }
    throw ValueError(f"{dist_type} is not a ground distance emd can compute")
}

#: A first basic feasible solution by Russell's method, which is the one
#: Rubner's implementation uses and OpenCV's after it.
#:
#: Each cell is scored against the largest cost still live in its row and in
#: its column, and the cell that loses by the most is filled first — a
#: cheapness measured against what the alternatives would cost rather than in
#: absolute terms, which is why it starts nearer the optimum than the
#: north-west corner does.
#:
#: Exactly one line leaves per step, so exactly `n + m - 1` cells end up basic
#: even when a step exhausts a row and a column together. That matters: the
#: basis has to span the whole bipartite graph for the dual variables to be
#: determined, and a step that dropped both lines would leave it a forest.
fn russell(
    cost: list[float],
    supply: list[float],
    demand: list[float],
    basic: list[bool],
    flow: list[float]
) -> void {
    let n = supply.len()
    let m = demand.len()
    var left = [value for value in supply]
    var want = [value for value in demand]
    var row_live: list[bool] = []
    var col_live: list[bool] = []
    for _i in 0..n { row_live.push(true) }
    for _j in 0..m { col_live.push(true) }
    var rows_left = n
    var cols_left = m

    var row_worst: list[float] = []
    var col_worst: list[float] = []
    for i in 0..n {
        var widest = -math.INF
        for j in 0..m {
            if cost[i * m + j] > widest { widest = cost[i * m + j] }
        }
        row_worst.push(widest)
    }
    for j in 0..m {
        var widest = -math.INF
        for i in 0..n {
            if cost[i * m + j] > widest { widest = cost[i * m + j] }
        }
        col_worst.push(widest)
    }

    loop {
        var best = math.INF
        var best_row = -1
        var best_col = -1
        for i in 0..n {
            if not row_live[i] { continue }
            for j in 0..m {
                if not col_live[j] { continue }
                let score = cost[i * m + j] - row_worst[i] - col_worst[j]
                if score < best {
                    best = score
                    best_row = i
                    best_col = j
                }
            }
        }
        if best_row < 0 { break }

        let take = min(left[best_row], want[best_col])
        basic[best_row * m + best_col] = true
        flow[best_row * m + best_col] = take
        left[best_row] -= take
        want[best_col] -= take
        if rows_left == 1 and cols_left == 1 { break }

        var drop_row = left[best_row] <= want[best_col] and rows_left > 1
        #: With one column left, every row still standing has to be given a
        #: cell of its own or the basis comes out short of a spanning tree and
        #: the dual variables are underdetermined. That only bites when the two
        #: totals were close enough not to earn a dummy without being equal.
        if cols_left == 1 { drop_row = true }
        if drop_row {
            row_live[best_row] = false
            rows_left -= 1
            for j in 0..m {
                if not col_live[j] { continue }
                var widest = -math.INF
                for i in 0..n {
                    if row_live[i] and cost[i * m + j] > widest { widest = cost[i * m + j] }
                }
                col_worst[j] = widest
            }
        } else {
            col_live[best_col] = false
            cols_left -= 1
            for i in 0..n {
                if not row_live[i] { continue }
                var widest = -math.INF
                for j in 0..m {
                    if col_live[j] and cost[i * m + j] > widest { widest = cost[i * m + j] }
                }
                row_worst[i] = widest
            }
        }
    }
}

#: Solve the balanced transportation problem and return the flow, laid out
#: `supply.len()` by `demand.len()` row-major.
fn transport(cost: list[float], supply: list[float], demand: list[float]) -> list[float] {
    let n = supply.len()
    let m = demand.len()
    var basic: list[bool] = []
    var flow: list[float] = []
    for _cell in 0..n * m {
        basic.push(false)
        flow.push(0.0)
    }
    russell(cost, supply, demand, basic, flow)

    var scale = 1.0
    for cell in 0..n * m {
        if math.fabs(cost[cell]) > scale { scale = math.fabs(cost[cell]) }
    }
    let tolerance = REDUCED_EPSILON * scale

    #: Row `i` is node `i` and column `j` is node `n + j`, so one stack and one
    #: visited flag serve both sides of the bipartite basis graph.
    var potential: list[float] = []
    var known: list[bool] = []
    var parent_node: list[int] = []
    var parent_cell: list[int] = []
    for _node in 0..n + m {
        potential.push(0.0)
        known.push(false)
        parent_node.push(-1)
        parent_cell.push(-1)
    }

    var stalled = 0
    var pivots = 0
    loop {
        pivots += 1
        if pivots > MAX_PIVOTS {
            throw ValueError("the transportation simplex did not converge")
        }

        #: The dual variables, read off the basis. Setting the first row's to
        #: zero fixes the one degree of freedom the n + m - 1 equations leave.
        for node in 0..n + m { known[node] = false }
        potential[0] = 0.0
        known[0] = true
        var stack = [0]
        var reached = 1
        while stack.len() > 0 {
            let node = stack.pop()
            if node < n {
                for j in 0..m {
                    if basic[node * m + j] and not known[n + j] {
                        known[n + j] = true
                        potential[n + j] = cost[node * m + j] - potential[node]
                        reached += 1
                        stack.push(n + j)
                    }
                }
            } else {
                let j = node - n
                for i in 0..n {
                    if basic[i * m + j] and not known[i] {
                        known[i] = true
                        potential[i] = cost[i * m + j] - potential[n + j]
                        reached += 1
                        stack.push(i)
                    }
                }
            }
        }
        if reached != n + m {
            throw ValueError("the transportation basis is not a spanning tree")
        }

        #: Bland's rule takes the first improving cell rather than the best
        #: one; taken consistently it cannot revisit a basis, which is what
        #: breaks a degenerate stall.
        let bland = stalled > STALL_LIMIT
        var best = -tolerance
        var enter_row = -1
        var enter_col = -1
        for i in 0..n {
            if bland and enter_row >= 0 { break }
            for j in 0..m {
                if basic[i * m + j] { continue }
                let reduced = cost[i * m + j] - potential[i] - potential[n + j]
                if reduced < best {
                    best = reduced
                    enter_row = i
                    enter_col = j
                    if bland { break }
                }
            }
        }
        if enter_row < 0 { break }

        #: The entering cell closes exactly one cycle, and it is the path
        #: through the basis tree from its row to its column plus the cell
        #: itself.
        for node in 0..n + m {
            known[node] = false
            parent_node[node] = -1
            parent_cell[node] = -1
        }
        known[enter_row] = true
        stack = [enter_row]
        while stack.len() > 0 {
            let node = stack.pop()
            if node < n {
                for j in 0..m {
                    if basic[node * m + j] and not known[n + j] {
                        known[n + j] = true
                        parent_node[n + j] = node
                        parent_cell[n + j] = node * m + j
                        stack.push(n + j)
                    }
                }
            } else {
                let j = node - n
                for i in 0..n {
                    if basic[i * m + j] and not known[i] {
                        known[i] = true
                        parent_node[i] = n + j
                        parent_cell[i] = i * m + j
                        stack.push(i)
                    }
                }
            }
        }
        if not known[n + enter_col] {
            throw ValueError("the transportation basis is not a spanning tree")
        }

        var cycle: list[int] = []
        var walk = n + enter_col
        while walk != enter_row {
            cycle.push(parent_cell[walk])
            walk = parent_node[walk]
        }

        #: The path leaves the column end, so its even positions are the cells
        #: the entering flow comes out of and its odd positions are the ones it
        #: goes into. The path is always odd in length, which is what makes the
        #: alternation close.
        var step = math.INF
        var leaving = -1
        var index = 0
        while index < cycle.len() {
            let cell = cycle[index]
            #: Bland's rule needs the tie broken by index as well, or the
            #: guarantee it buys does not hold.
            var take = flow[cell] < step
            if bland and flow[cell] == step and cell < leaving { take = true }
            if take {
                step = flow[cell]
                leaving = cell
            }
            index += 2
        }
        if leaving < 0 {
            throw ValueError("the transportation simplex found no cycle to move flow round")
        }

        if step <= 0.0 {
            stalled += 1
        } else {
            stalled = 0
        }
        flow[enter_row * m + enter_col] += step
        for position in 0..cycle.len() {
            if position % 2 == 0 {
                flow[cycle[position]] -= step
            } else {
                flow[cycle[position]] += step
            }
        }
        basic[enter_row * m + enter_col] = true
        basic[leaving] = false
        flow[leaving] = 0.0
    }
    return flow
}

#: The earth mover's distance between two signatures.
#:
#: A signature is one cluster a row: the first column is the cluster's weight
#: and the rest are its coordinates. `dist_type` picks the ground distance
#: between coordinates — `DIST_L1`, `DIST_L2` or `DIST_C` — or `DIST_USER`,
#: which takes `cost` as the full `signature1.rows` by `signature2.rows` matrix
#: of ground distances and reads no coordinates at all.
#:
#: Unequal total weights are allowed. The lighter signature gains a dummy
#: cluster that absorbs the difference at no cost, so the surplus stays put
#: rather than being forced somewhere, and the result divides by the heavier
#: total.
#:
#: `lower_bound`, when non-zero, asks for OpenCV's early exit: the distance
#: between the two centres of mass is a lower bound on the answer whenever the
#: totals match, and if it reaches the bound given the function returns it
#: without solving anything. Returns the distance, the bound in force on the
#: way out, and the flow.
pub fn emd(
    signature1: Mat,
    signature2: Mat,
    dist_type: int,
    cost: Mat? = null,
    lower_bound: float = 0.0
) -> tuple[float, float, Mat] {
    if signature1.empty() or signature2.empty() {
        throw ValueError("a signature needs at least one cluster")
    }
    if signature1.channels() != 1 or signature2.channels() != 1 {
        throw ValueError("a signature is single channel")
    }
    let dims = signature1.cols - 1
    #: The caller's table is read and checked here rather than where it is
    #: first indexed, so a wrong shape is refused before any work is done.
    var given: list[float] = []
    var given_width = 0
    if dist_type == DIST_USER {
        let table = cost
        if table is null {
            throw ValueError("DIST_USER needs the cost matrix the caller promised")
        }
        if table.channels() != 1 {
            throw ValueError("a cost matrix is single channel")
        }
        if table.rows != signature1.rows or table.cols != signature2.rows {
            throw ValueError(
                f"a cost matrix for these signatures is {signature1.rows}x{signature2.rows}, not {table.rows}x{table.cols}"
            )
        }
        given = table.values()
        given_width = table.cols
    } else {
        if dims < 1 {
            throw ValueError("a signature needs coordinates as well as a weight")
        }
        if signature2.cols != signature1.cols {
            throw ValueError(
                f"the signatures have {signature1.cols - 1} and {signature2.cols - 1} coordinates"
            )
        }
    }

    let values1 = signature1.values()
    let values2 = signature2.values()
    let width1 = signature1.cols
    let width2 = signature2.cols

    #: A cluster of no weight moves nothing, and carrying it would only add a
    #: row of the basis that no flow can ever leave. OpenCV drops these too,
    #: and the flow returned is indexed against the original rows so the caller
    #: cannot tell.
    var keep1: list[int] = []
    var supply: list[float] = []
    var total1 = 0.0
    for i in 0..signature1.rows {
        let weight = values1[i * width1]
        if weight < 0.0 { throw ValueError("a signature cannot hold a negative weight") }
        if weight == 0.0 { continue }
        keep1.push(i)
        supply.push(weight)
        total1 += weight
    }
    var keep2: list[int] = []
    var demand: list[float] = []
    var total2 = 0.0
    for j in 0..signature2.rows {
        let weight = values2[j * width2]
        if weight < 0.0 { throw ValueError("a signature cannot hold a negative weight") }
        if weight == 0.0 { continue }
        keep2.push(j)
        demand.push(weight)
        total2 += weight
    }
    if supply.len() == 0 or demand.len() == 0 {
        throw ValueError("a signature needs one weight above zero")
    }

    var reported = lower_bound
    let totals_match = math.fabs(total1 - total2) < WEIGHT_EPSILON * total1
    if lower_bound != 0.0 and totals_match {
        if dist_type == DIST_USER {
            throw ValueError("a lower bound is a distance, so it cannot go with DIST_USER")
        }
        #: The centres of mass are as far apart as the cheapest schedule has to
        #: move the average gram, so no transport plan can beat that — which is
        #: what makes this a bound worth testing before paying for the simplex.
        var centre1: list[float] = []
        var centre2: list[float] = []
        for k in 0..dims {
            var moment1 = 0.0
            for i in 0..signature1.rows {
                moment1 += values1[i * width1] * values1[i * width1 + 1 + k]
            }
            centre1.push(moment1 / total1)
            var moment2 = 0.0
            for j in 0..signature2.rows {
                moment2 += values2[j * width2] * values2[j * width2 + 1 + k]
            }
            centre2.push(moment2 / total2)
        }
        let apart = ground_distance(centre1, 0, centre2, 0, dims, dist_type)
        reported = apart
        if lower_bound <= apart {
            let empty = Mat.zeros(signature1.rows, signature2.rows, CV_MAKETYPE(CV_32F, 1))
            return (apart, reported, empty)
        }
    }

    var rows = supply.len()
    var cols = demand.len()
    var costs: list[float] = []
    if dist_type == DIST_USER {
        for i in 0..rows {
            for j in 0..cols { costs.push(given[keep1[i] * given_width + keep2[j]]) }
        }
    } else {
        for i in 0..rows {
            for j in 0..cols {
                costs.push(
                    ground_distance(
                        values1,
                        keep1[i] * width1 + 1,
                        values2,
                        keep2[j] * width2 + 1,
                        dims,
                        dist_type
                    )
                )
            }
        }
    }

    #: The dummy sits at zero cost against every real cluster, so whatever ends
    #: up routed through it is surplus nobody paid to move.
    let difference = total1 - total2
    if math.fabs(difference) >= WEIGHT_EPSILON * total1 {
        if difference < 0.0 {
            supply.push(-difference)
            for _j in 0..cols { costs.push(0.0) }
            rows += 1
        } else {
            demand.push(difference)
            var widened: list[float] = []
            for i in 0..rows {
                for j in 0..cols { widened.push(costs[i * cols + j]) }
                widened.push(0.0)
            }
            costs = widened
            cols += 1
        }
    }

    let flow = transport(costs, supply, demand)

    var moved = 0.0
    var table: list[float] = []
    for _cell in 0..signature1.rows * signature2.rows { table.push(0.0) }
    for i in 0..keep1.len() {
        for j in 0..keep2.len() {
            let amount = flow[i * cols + j]
            moved += amount * costs[i * cols + j]
            table[keep1[i] * signature2.rows + keep2[j]] = amount
        }
    }
    let heavier = max(total1, total2)
    let result = Mat.from_list(signature1.rows, signature2.rows, CV_MAKETYPE(CV_32F, 1), table)
    return (moved / heavier, reported, result)
}
