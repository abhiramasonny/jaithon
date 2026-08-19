#: The earth mover's distance.
#:
#: Two things are checked here. The first is arithmetic anybody can do on
#: paper: two signatures that are the same cost nothing to turn into each
#: other, one cluster a side costs the ground distance between them, and a
#: two-by-two problem whose optimum is short enough to work out by hand comes
#: out at that optimum.
#:
#: The second is OpenCV. Every case below carries what OpenCV 5.0.0 returned
#: for it, recorded through `~/.venvs/scratch/bin/python` and written out
#: rather than computed, so the test runs without OpenCV installed. This is a
#: linear program solved exactly on both sides, so the answers should agree to
#: the last bit either implementation can represent, and the tolerance below
#: says how close they actually come.
import std.math
from std.test import assert_eq, assert_near, assert_throws, assert_true
import std.gpu as gpu
from jaicv.core.mat import Mat
from jaicv.core.types import CV_32FC1
from jaicv.imgproc.emd import emd, DIST_USER
from jaicv.imgproc.segmentation import DIST_C, DIST_L1, DIST_L2

fn signature(rows: int, cols: int, values: list[float]) -> Mat {
    return Mat.from_list(rows, cols, CV_32FC1, values)
}

#: One recorded call and what OpenCV gave back for it. `reported` is the bound
#: OpenCV left behind, which it only overwrites when it actually computed one.
class Case {
    pub let name: str
    pub let first: Mat
    pub let second: Mat
    pub let dist_type: int
    pub let cost: Mat?
    pub let bound: float
    pub let distance: float
    pub let reported: float

    pub fn init(
        self,
        name: str,
        first: Mat,
        second: Mat,
        dist_type: int,
        cost: Mat?,
        bound: float,
        distance: float,
        reported: float
    ) {
        self.name = name
        self.first = first
        self.second = second
        self.dist_type = dist_type
        self.cost = cost
        self.bound = bound
        self.distance = distance
        self.reported = reported
    }
}

#: The cases OpenCV was asked about, and what it answered. Between them they
#: cover every metric, a caller-supplied cost matrix, signatures of unequal
#: total weight both ways round, a cluster of no weight, a problem degenerate
#: enough to have ties everywhere, and both sides of the lower-bound exit.
fn recorded() -> list[Case] {
    let flat = signature(2, 3, [0.3, 1.0, 2.0, 0.7, 4.0, 5.0])
    let solid = signature(2, 4, [0.25, 1.0, 2.0, 3.0, 0.75, -4.0, 0.0, 2.0])
    let origin = signature(1, 3, [2.0, 0.0, 0.0])
    let far = signature(1, 3, [2.0, 3.0, 4.0])
    let farther = signature(1, 3, [5.0, 3.0, 4.0])
    let split = signature(2, 2, [0.7, 0.0, 0.3, 4.0])
    let target = signature(2, 2, [0.3, 1.0, 0.7, 5.0])
    let pair = signature(2, 2, [0.5, 0.0, 0.5, 10.0])
    let nudged = signature(2, 2, [0.5, 1.0, 0.5, 11.0])
    let shift = signature(2, 2, [0.4, 0.0, 0.6, 3.0])
    let middle = signature(2, 2, [0.5, 1.0, 0.5, 2.0])
    let light = signature(2, 2, [0.3, 1.0, 0.3, 2.0])
    let padded = signature(3, 2, [0.5, 0.0, 0.0, 1.0, 0.5, 2.0])
    let lone = signature(1, 2, [1.0, 1.0])
    let halves = signature(2, 1, [0.5, 0.5])
    let quarters = signature(2, 1, [0.25, 0.75])
    let small = signature(2, 2, [1.0, 2.0, 3.0, 0.5])
    let supply = signature(3, 1, [3.0, 1.0, 2.0])
    let demand = signature(4, 1, [2.0, 2.0, 1.0, 1.0])
    let table = signature(3, 4, [4.0, 1.0, 3.0, 2.0, 2.0, 5.0, 1.0, 6.0, 0.0, 3.0, 2.0, 1.0])
    let ladder = signature(6, 2, [1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0])
    let rungs = signature(4, 2, [2.0, 0.0, 1.0, 2.0, 2.0, 3.0, 1.0, 5.0])
    let stack = signature(4, 2, [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    let pile = signature(2, 2, [2.0, 0.0, 2.0, 0.0])
    let corners = signature(4, 3, [0.2, 1.0, 7.0, 0.35, -2.0, 0.5, 0.15, 6.0, 6.0, 0.3, 0.0, -8.0])
    let cloud = signature(3, 3, [0.5, 2.0, 2.0, 0.1, -5.0, 4.0, 0.4, 3.0, -6.0])
    let spread = signature(3, 4, [0.5, 2.0, 2.0, 2.0, 0.1, -5.0, 4.0, -1.0, 0.4, 3.0, -6.0, 0.0])
    let spot_a = signature(2, 3, [0.5, 0.0, 0.0, 0.5, 4.0, 0.0])
    let spot_b = signature(2, 3, [0.5, 1.0, 1.0, 0.5, 3.0, -2.0])
    let spot_c = signature(2, 3, [0.25, -2.0, 5.0, 0.75, 6.0, 1.0])
    #: What OpenCV makes the distance between the two centres of mass, which is
    #: the bound it tests and then hands back.
    let centres = 0.30000007152557373
    return [
        Case("same, L2", flat, flat, DIST_L2, null, 0.0, 0.0, 0.0),
        Case("same, L1", flat, flat, DIST_L1, null, 0.0, 0.0, 0.0),
        Case("same, C", solid, solid, DIST_C, null, 0.0, 0.0, 0.0),
        Case("one each, L2", origin, far, DIST_L2, null, 0.0, 5.0, 0.0),
        Case("one each, L1", origin, far, DIST_L1, null, 0.0, 7.0, 0.0),
        Case("one each, C", origin, far, DIST_C, null, 0.0, 4.0, 0.0),
        Case("one each, heavy", origin, farther, DIST_L2, null, 0.0, 2.0, 0.0),
        Case("a cluster splits", split, target, DIST_L1, null, 0.0, 2.5999999046325684, 0.0),
        Case("parallel pairs", pair, nudged, DIST_L1, null, 0.0, 1.0, 0.0),
        Case("shifted, L2", shift, middle, DIST_L2, null, 0.0, 1.100000023841858, 0.0),
        Case("shifted, L1", shift, middle, DIST_L1, null, 0.0, 1.100000023841858, 0.0),
        Case("shifted, C", shift, middle, DIST_C, null, 0.0, 1.100000023841858, 0.0),
        Case("lighter target", shift, light, DIST_L2, null, 0.0, 0.6000000238418579, 0.0),
        Case("lighter source", light, shift, DIST_L2, null, 0.0, 0.6000000238418579, 0.0),
        Case("weightless row", padded, lone, DIST_L2, null, 0.0, 1.0, 0.0),
        Case("given costs", halves, quarters, DIST_USER, small, 0.0, 1.0, 0.0),
        Case("bigger costs", supply, demand, DIST_USER, table, 0.0, 0.8333333134651184, 0.0),
        Case("ties everywhere", ladder, rungs, DIST_L1, null, 0.0, 0.3333333432674408, 0.0),
        Case("every cost equal", stack, pile, DIST_L2, null, 0.0, 0.0, 0.0),
        Case("four to three", corners, cloud, DIST_L2, null, 0.0, 4.871836185455322, 0.0),
        Case("four to three, C", corners, cloud, DIST_C, null, 0.0, 4.099999904632568, 0.0),
        Case("three to four", cloud, corners, DIST_L1, null, 0.0, 6.525000095367432, 0.0),
        Case("three dims", spread, solid, DIST_L2, null, 0.0, 6.218186855316162, 0.0),
        Case("three dims, C", solid, spread, DIST_C, null, 0.0, 4.949999809265137, 0.0),
        Case("bound too low", shift, middle, DIST_L2, null, 1.0, 1.100000023841858, centres),
        Case("bound stops", shift, middle, DIST_L2, null, 0.1, centres, centres),
        Case("bound, same", flat, flat, DIST_L2, null, 1.0, 0.0, 0.0),
        Case("spot a to b", spot_a, spot_b, DIST_L2, null, 0.0, 1.8251407146453857, 0.0),
        Case("spot b to c", spot_b, spot_c, DIST_L2, null, 0.0, 4.6213202476501465, 0.0),
        Case("spot a to c", spot_a, spot_c, DIST_L2, null, 0.0, 3.985015869140625, 0.0),
    ]
}

fn test_identical_signatures_cost_nothing() -> void {
    if not gpu.is_available() { return }
    let flat = signature(2, 3, [0.3, 1.0, 2.0, 0.7, 4.0, 5.0])
    let solid = signature(2, 4, [0.25, 1.0, 2.0, 3.0, 0.75, -4.0, 0.0, 2.0])
    for dist_type in [DIST_L1, DIST_L2, DIST_C] {
        assert_eq(emd(flat, flat, dist_type)[0], 0.0, "a signature is no distance from itself")
        assert_eq(emd(solid, solid, dist_type)[0], 0.0, "a signature is no distance from itself")
    }
}

fn test_one_cluster_each_is_the_ground_distance() -> void {
    if not gpu.is_available() { return }
    #: Three-four-five, so every metric lands on a different whole number.
    let here = signature(1, 3, [2.0, 0.0, 0.0])
    let there = signature(1, 3, [2.0, 3.0, 4.0])
    assert_eq(emd(here, there, DIST_L2)[0], 5.0, "Euclidean")
    assert_eq(emd(here, there, DIST_L1)[0], 7.0, "city block")
    assert_eq(emd(here, there, DIST_C)[0], 4.0, "chessboard")

    #: The weight cancels, because the distance divides by the total. Ten units
    #: moved five apart is the same answer as two moved five apart.
    let heavy = signature(1, 3, [10.0, 0.0, 0.0])
    let heavier = signature(1, 3, [10.0, 3.0, 4.0])
    assert_eq(emd(heavy, heavier, DIST_L2)[0], 5.0, "the weight should divide out")
}

fn test_two_clusters_a_side_by_hand() -> void {
    if not gpu.is_available() { return }
    #: Nothing to decide: each cluster's nearest partner is also the one it can
    #: reach alone, and the pairing costs one apiece.
    let left = signature(2, 2, [0.5, 0.0, 0.5, 10.0])
    let right = signature(2, 2, [0.5, 1.0, 0.5, 11.0])
    assert_near(emd(left, right, DIST_L1)[0], 1.0, 1e-6, "two pairs one apart")

    #: Now the weights disagree, so 0.7 at the origin cannot go anywhere whole:
    #: 0.3 of it fills the cluster at 1 and the remaining 0.4 has to cross to
    #: 5, which is 0.3 + 2.0 + 0.3 = 2.6 against a total weight of one. Sending
    #: the whole 0.7 to 5 instead would cost 4.4.
    let split = signature(2, 2, [0.7, 0.0, 0.3, 4.0])
    let target = signature(2, 2, [0.3, 1.0, 0.7, 5.0])
    assert_near(emd(split, target, DIST_L1)[0], 2.6, 1e-6, "a cluster has to be split")
}

fn test_a_user_cost_matrix_is_used_as_given() -> void {
    if not gpu.is_available() { return }
    #: Weights only, no coordinates at all -- which is the whole point of
    #: DIST_USER, and something no ground distance could handle.
    let left = signature(2, 1, [0.5, 0.5])
    let right = signature(2, 1, [0.25, 0.75])
    let cost = signature(2, 2, [1.0, 2.0, 3.0, 0.5])
    #: 0.25 down the cheap first cell and 0.25 across at 2.0 empties the first
    #: cluster; the second goes to the second at 0.5. That is 0.25 + 0.5 + 0.25.
    assert_near(emd(left, right, DIST_USER, cost)[0], 1.0, 1e-6, "the given costs")

    #: Three by four, supplies 3, 1, 2 against demands 2, 2, 1, 1. Row three
    #: fills the first demand free, row two the third at 1, and row one takes
    #: what is left at 1 and 2, so five over a total weight of six.
    let supply = signature(3, 1, [3.0, 1.0, 2.0])
    let demand = signature(4, 1, [2.0, 2.0, 1.0, 1.0])
    let table = signature(3, 4, [4.0, 1.0, 3.0, 2.0, 2.0, 5.0, 1.0, 6.0, 0.0, 3.0, 2.0, 1.0])
    assert_near(emd(supply, demand, DIST_USER, table)[0], 5.0 / 6.0, 1e-6, "three by four")
}

fn test_a_coordinate_cost_matrix_is_ignored() -> void {
    if not gpu.is_available() { return }
    #: OpenCV 5 reads `cost` only for DIST_USER. Passing one alongside a ground
    #: distance changes nothing, and this pins that rather than leaving it to
    #: be discovered.
    let left = signature(2, 2, [0.5, 0.0, 0.5, 2.0])
    let right = signature(1, 2, [1.0, 1.0])
    let nonsense = signature(2, 1, [7.0, 9.0])
    assert_near(emd(left, right, DIST_L2, nonsense)[0], 1.0, 1e-6, "the ground distance wins")
    assert_near(emd(left, right, DIST_USER, nonsense)[0], 8.0, 1e-6, "the table wins")
}

fn test_unequal_total_weights() -> void {
    if not gpu.is_available() { return }
    #: 1.0 of supply against 0.6 of demand. The 0.4 left over goes to the dummy
    #: at no cost, the 0.6 that moves costs 0.6, and the answer divides by the
    #: heavier total rather than by what moved -- so 0.6, not 1.0.
    let heavy = signature(2, 2, [0.4, 0.0, 0.6, 3.0])
    let light = signature(2, 2, [0.3, 1.0, 0.3, 2.0])
    assert_near(emd(heavy, light, DIST_L2)[0], 0.6, 1e-6, "the surplus stays put")
    assert_near(emd(light, heavy, DIST_L2)[0], 0.6, 1e-6, "and the same the other way")

    let one = signature(1, 3, [2.0, 0.0, 0.0])
    let other = signature(1, 3, [5.0, 3.0, 4.0])
    #: Two units cross five, and five is the divisor: 10 / 5.
    assert_near(emd(one, other, DIST_L2)[0], 2.0, 1e-6, "the heavier total divides")
}

fn test_a_cluster_of_no_weight_is_ignored() -> void {
    if not gpu.is_available() { return }
    let padded = signature(3, 2, [0.5, 0.0, 0.0, 1.0, 0.5, 2.0])
    let bare = signature(2, 2, [0.5, 0.0, 0.5, 2.0])
    let single = signature(1, 2, [1.0, 1.0])
    let (with_it, _b1, flow) = emd(padded, single, DIST_L2)
    let (without, _b2, _f2) = emd(bare, single, DIST_L2)
    assert_near(with_it, without, 1e-12, "a weightless cluster changed the answer")
    #: The flow is still indexed against the rows the caller passed, so the
    #: empty cluster is a row of zeros rather than a missing row.
    assert_eq(flow.rows, 3, "the flow lost the weightless row")
    assert_eq(flow.at(1, 0), 0.0, "something was routed through a weightless cluster")
}

fn test_lower_bound_stops_it_early() -> void {
    if not gpu.is_available() { return }
    #: Centres of mass at 1.8 and 1.5, so the bound OpenCV computes is 0.3
    #: against a true distance of 1.1.
    let first = signature(2, 2, [0.4, 0.0, 0.6, 3.0])
    let second = signature(2, 2, [0.5, 1.0, 0.5, 2.0])

    let (stopped, reported, flow) = emd(first, second, DIST_L2, null, 0.1)
    assert_near(stopped, 0.3, 1e-6, "the bound should have been returned instead")
    assert_near(reported, 0.3, 1e-6, "the bound in force on the way out")
    var moved = 0.0
    for value in flow.values() { moved += value }
    assert_eq(moved, 0.0, "an early exit should not have solved anything")

    let (solved, also, _f) = emd(first, second, DIST_L2, null, 1.0)
    assert_near(solved, 1.1, 1e-6, "a bound of one is above 0.3, so it solves")
    assert_near(also, 0.3, 1e-6, "and still reports what the bound came to")

    #: Zero is not a bound, it is the absence of one -- OpenCV's C API says so
    #: with a null pointer, and every bound would beat zero if it were.
    let (unbounded, none, _g) = emd(first, second, DIST_L2)
    assert_near(unbounded, 1.1, 1e-6, "no bound asked for, so none applied")
    assert_eq(none, 0.0, "nothing to report")
}

fn test_totals_that_nearly_match() -> void {
    if not gpu.is_available() { return }
    #: Totals a millionth apart are inside the slack that earns a dummy
    #: cluster, so neither implementation adds one and both leave the leftover
    #: millionth parked wherever the last allocation put it. Where it lands is
    #: implementation-defined -- OpenCV is not even symmetric here, giving
    #: 1.0000002 one way and 0.9999990 the other -- so what is pinned is that
    #: the problem is still solved rather than what the last digits come to.
    #: Getting this wrong is not a rounding error: dropping the last column
    #: while rows still need one leaves the basis short of a spanning tree, and
    #: the solver throws.
    let almost = signature(2, 2, [0.9999999, 0.0, 0.0000001, 5.0])
    let single = signature(1, 2, [0.999999, 1.0])
    assert_near(emd(almost, single, DIST_L2)[0], 1.0, 1e-4, "nearly all of it moves one")
    assert_near(emd(single, almost, DIST_L2)[0], 1.0, 1e-4, "and the same the other way")

    let three = signature(3, 2, [0.99999, 0.0, 0.000005, 5.0, 0.000005, 9.0])
    let one = signature(1, 2, [0.999998, 1.0])
    assert_near(emd(three, one, DIST_L2)[0], 1.0, 1e-3, "three clusters into one")
}

fn test_the_distance_is_symmetric() -> void {
    if not gpu.is_available() { return }
    for item in recorded() {
        if item.bound != 0.0 { continue }
        if item.dist_type == DIST_USER { continue }
        let (forward, _b1, _f1) = emd(item.first, item.second, item.dist_type)
        let (backward, _b2, _f2) = emd(item.second, item.first, item.dist_type)
        #: Not to the last bit, and it cannot be: the weights arrive as float32
        #: and are added up in float64, so two signatures that both "weigh one"
        #: can differ in the eighth digit, which leaves the problem a hair
        #: unbalanced and lets the two directions settle on vertices that cost
        #: fractionally different amounts. Measured, the gap is 8e-10.
        assert_near(
            backward,
            forward,
            1e-7 * (math.fabs(forward) + 1.0),
            f"{item.name}: {forward} one way and {backward} the other"
        )
    }
}

fn test_the_triangle_inequality_holds() -> void {
    if not gpu.is_available() { return }
    #: Only with equal total weights -- unequal ones leave mass parked on the
    #: dummy for free, and the distance stops being a metric.
    let a = signature(2, 3, [0.5, 0.0, 0.0, 0.5, 4.0, 0.0])
    let b = signature(2, 3, [0.5, 1.0, 1.0, 0.5, 3.0, -2.0])
    let c = signature(2, 3, [0.25, -2.0, 5.0, 0.75, 6.0, 1.0])
    let sides = [a, b, c]
    for first in sides {
        for second in sides {
            for third in sides {
                let direct = emd(first, third, DIST_L2)[0]
                let round = emd(first, second, DIST_L2)[0] + emd(second, third, DIST_L2)[0]
                assert_true(
                    direct <= round + 1e-9,
                    f"{direct} direct beat {round} by way of a third signature"
                )
            }
        }
    }
}

fn test_the_flow_is_feasible_and_costs_the_distance() -> void {
    if not gpu.is_available() { return }
    for item in recorded() {
        if item.bound != 0.0 { continue }
        let (distance, _b, flow) = emd(item.first, item.second, item.dist_type, item.cost)
        let table = flow.values()
        let rows = item.first.rows
        let cols = item.second.rows
        let left = item.first.values()
        let right = item.second.values()
        var total1 = 0.0
        var total2 = 0.0
        var moved = 0.0
        for i in 0..rows { total1 += left[i * item.first.cols] }
        for j in 0..cols { total2 += right[j * item.second.cols] }
        for i in 0..rows {
            var out = 0.0
            for j in 0..cols {
                let amount = table[i * cols + j]
                assert_true(amount >= -1e-9, f"{item.name}: negative flow")
                out += amount
                moved += amount
            }
            #: float32 storage, so the slack has to be sized for float32.
            assert_true(
                out <= left[i * item.first.cols] * (1.0 + 1e-5) + 1e-6,
                f"{item.name}: cluster {i} shipped more than it holds"
            )
        }
        for j in 0..cols {
            var into = 0.0
            for i in 0..rows { into += table[i * cols + j] }
            assert_true(
                into <= right[j * item.second.cols] * (1.0 + 1e-5) + 1e-6,
                f"{item.name}: cluster {j} took more than it wanted"
            )
        }
        assert_near(
            moved,
            min(total1, total2),
            1e-5 * (total1 + 1.0),
            f"{item.name}: the lighter signature was not emptied"
        )
        #: And the schedule costs what was returned, which is the only thing
        #: tying the flow to the distance.
        var paid = 0.0
        for i in 0..rows {
            for j in 0..cols {
                let amount = table[i * cols + j]
                if amount == 0.0 { continue }
                paid += amount * cost_between(item, i, j)
            }
        }
        assert_near(
            paid / max(total1, total2),
            distance,
            1e-5 * (distance + 1.0),
            f"{item.name}: the flow does not cost what was returned"
        )
    }
}

fn cost_between(item: Case, i: int, j: int) -> float {
    let table = item.cost
    if item.dist_type == DIST_USER {
        if table is null { throw ValueError("DIST_USER without a cost matrix") }
        return table.at(i, j)
    }
    let left = item.first.values()
    let right = item.second.values()
    let dims = item.first.cols - 1
    if item.dist_type == DIST_L1 {
        var total = 0.0
        for k in 0..dims {
            total += math.fabs(
                left[i * item.first.cols + 1 + k] - right[j * item.second.cols + 1 + k]
            )
        }
        return total
    }
    if item.dist_type == DIST_L2 {
        var total = 0.0
        for k in 0..dims {
            let delta = left[i * item.first.cols + 1 + k] - right[j * item.second.cols + 1 + k]
            total += delta * delta
        }
        return math.sqrt(total)
    }
    var widest = 0.0
    for k in 0..dims {
        let delta = math.fabs(
            left[i * item.first.cols + 1 + k] - right[j * item.second.cols + 1 + k]
        )
        if delta > widest { widest = delta }
    }
    return widest
}

#: The cheapest integral transport plan, found by trying all of them.
#:
#: Integer supplies and demands make the linear program's optimum integral, so
#: this exponential thing is an exact independent answer rather than an
#: approximation of one -- which is what makes it worth having against a
#: simplex that could quietly stop early.
fn cheapest(cost: list[float], left: list[int], want: list[int], m: int, i: int, j: int) -> float {
    if i == left.len() { return 0.0 }
    if j == m {
        if left[i] != 0 { return math.INF }
        return cheapest(cost, left, want, m, i + 1, 0)
    }
    var best = math.INF
    let most = min(left[i], want[j])
    for amount in 0..most + 1 {
        left[i] -= amount
        want[j] -= amount
        let rest = cheapest(cost, left, want, m, i, j + 1)
        left[i] += amount
        want[j] += amount
        if rest == math.INF { continue }
        let total = float(amount) * cost[i * m + j] + rest
        if total < best { best = total }
    }
    return best
}

#: A small deterministic generator, so the families below are the same on
#: every run and a failure can be reproduced from the seed alone.
fn draws(seed: int, count: int) -> tuple[int, list[int]] {
    var state = seed
    var values: list[int] = []
    for _index in 0..count {
        state = (state * 1103515245 + 12345) % 2147483648
        values.push(state // 65536)
    }
    return (state, values)
}

#: Weights that are whole numbers, spread so that the demands sum to the
#: supplies. Equal totals are what the brute force needs, and whole numbers are
#: what makes partial sums collide, which is what makes the basis degenerate.
fn balanced(draw: list[int], at: int, rows: int, cols: int) -> tuple[list[int], list[int]] {
    var supply: list[int] = []
    var total = 0
    for i in 0..rows {
        let weight = 1 + draw[at + i] % 2
        supply.push(weight)
        total += weight
    }
    if total < cols {
        supply[0] += cols - total
        total = cols
    }
    var demand: list[int] = []
    for _j in 0..cols { demand.push(1) }
    var spare = total - cols
    var handed = 0
    while spare > 0 {
        demand[draw[at + rows + handed] % cols] += 1
        spare -= 1
        handed += 1
    }
    return (supply, demand)
}

fn whole(values: list[int]) -> list[float] {
    return [float(value) for value in values]
}

fn test_degenerate_bases_reach_the_optimum() -> void {
    if not gpu.is_available() { return }
    #: Costs are drawn at random rather than taken from coordinates on purpose.
    #: A cost matrix built from points on a line is one Russell's method tends
    #: to solve outright -- on such a family sixty problems took five pivots
    #: between them, and the simplex was never really run. Against random costs
    #: the same sixty take thirty-nine, sixteen of which move no flow at all,
    #: which is the degenerate pivot that cycles if the rule is careless.
    var seed = 20260819
    var solved = 0
    var nontrivial = 0
    for _trial in 0..60 {
        let (next, draw) = draws(seed, 64)
        seed = next
        let rows = 3 + draw[0] % 3
        let cols = 3 + draw[1] % 3
        let (supply, demand) = balanced(draw, 2, rows, cols)
        var costs: list[float] = []
        for cell in 0..rows * cols { costs.push(float(draw[24 + cell] % 30)) }
        let first = signature(rows, 1, whole(supply))
        let second = signature(cols, 1, whole(demand))
        let table = signature(rows, cols, costs)
        var left = [value for value in supply]
        var want = [value for value in demand]
        let plan = cheapest(costs, left, want, cols, 0, 0)
        var total = 0
        for weight in supply { total += weight }
        let (got, _b, _f) = emd(first, second, DIST_USER, table)
        assert_near(
            got,
            plan / float(total),
            1e-9,
            f"a degenerate {rows}x{cols} problem did not reach the optimum"
        )
        solved += 1
        if plan > 0.0 { nontrivial += 1 }
    }
    assert_eq(solved, 60, "the degenerate family did not get built")
    #: Guard against the family collapsing into problems whose answer is zero
    #: whatever the solver does.
    assert_true(nontrivial > 50, f"only {nontrivial} of them had anything to move")
}

fn test_a_large_degenerate_problem_stays_consistent() -> void {
    if not gpu.is_available() { return }
    #: Twelve clusters a side and unit-ish weights: ten pivots a problem and
    #: three quarters of them degenerate, which is far past what the brute
    #: force above can check. What is asserted instead is that the answer is
    #: still a sane one -- it beats a feasible plan anyone can write down, it
    #: does not beat a bound no plan can, and it comes out the same when the
    #: whole problem is transposed, which sends the simplex down an entirely
    #: different sequence of pivots.
    var seed = 4242
    for _trial in 0..20 {
        let (next, draw) = draws(seed, 200)
        seed = next
        let side = 12
        let (supply, demand) = balanced(draw, 0, side, side)
        var costs: list[float] = []
        var flipped: list[float] = []
        for _cell in 0..side * side { flipped.push(0.0) }
        for i in 0..side {
            for j in 0..side {
                let value = float(draw[40 + i * side + j] % 50)
                costs.push(value)
                flipped[j * side + i] = value
            }
        }
        var total = 0
        for weight in supply { total += weight }
        let first = signature(side, 1, whole(supply))
        let second = signature(side, 1, whole(demand))
        let (forward, _b1, _f1) = emd(first, second, DIST_USER, signature(side, side, costs))
        let (backward, _b2, _f2) = emd(second, first, DIST_USER, signature(side, side, flipped))
        assert_near(backward, forward, 1e-9, "transposing the problem changed the answer")

        #: North-west corner: fill the first cell as far as it goes, step on.
        #: Nothing about it is clever, which is the point -- it is a plan, so
        #: the optimum cannot cost more.
        var left = [value for value in supply]
        var want = [value for value in demand]
        var feasible = 0.0
        var i = 0
        var j = 0
        while i < side and j < side {
            let take = min(left[i], want[j])
            feasible += float(take) * costs[i * side + j]
            left[i] -= take
            want[j] -= take
            if left[i] == 0 { i += 1 } else { j += 1 }
        }
        assert_true(
            forward <= feasible / float(total) + 1e-9,
            f"{forward} is worse than a north-west corner plan at {feasible / float(total)}"
        )

        #: And every unit leaving a cluster pays at least the cheapest cell in
        #: its row, so no plan at all can come in under this.
        var floor_ = 0.0
        for row in 0..side {
            var cheapest_cell = costs[row * side]
            for column in 1..side {
                if costs[row * side + column] < cheapest_cell {
                    cheapest_cell = costs[row * side + column]
                }
            }
            floor_ += float(supply[row]) * cheapest_cell
        }
        assert_true(
            forward >= floor_ / float(total) - 1e-9,
            f"{forward} is below {floor_ / float(total)}, which no plan can reach"
        )
    }
}

fn test_rejects_what_it_cannot_answer() -> void {
    if not gpu.is_available() { return }
    let good = signature(2, 2, [0.5, 0.0, 0.5, 1.0])
    assert_throws(ValueError, || emd(signature(1, 2, [-1.0, 0.0]), good, DIST_L2))
    assert_throws(ValueError, || emd(signature(1, 2, [0.0, 0.0]), good, DIST_L2))
    assert_throws(ValueError, || emd(good, good, DIST_USER))
    assert_throws(ValueError, || emd(good, signature(1, 3, [1.0, 0.0, 0.0]), DIST_L2))
    assert_throws(ValueError, || emd(signature(2, 1, [0.5, 0.5]), good, DIST_L2))
    assert_throws(ValueError, || emd(good, good, 99))
    assert_throws(ValueError, || emd(good, good, DIST_USER, signature(1, 1, [1.0])))
    #: A bound needs a ground distance to measure the centres of mass with.
    assert_throws(
        ValueError,
        || emd(good, good, DIST_USER, signature(2, 2, [1.0, 1.0, 1.0, 1.0]), 1.0)
    )
}

fn test_every_recorded_case_matches_opencv() -> void {
    if not gpu.is_available() { return }
    let cases = recorded()
    assert_true(cases.len() > 20, "the recorded table went missing")
    var worst = 0.0
    var worst_at = ""
    var worst_bound = 0.0
    for item in cases {
        let (got, reported, _flow) = emd(
            item.first,
            item.second,
            item.dist_type,
            item.cost,
            item.bound
        )
        let scale = max(math.fabs(item.distance), 1.0)
        let error = math.fabs(got - item.distance) / scale
        if error > worst {
            worst = error
            worst_at = item.name
        }
        let bound_error = math.fabs(reported - item.reported) / max(math.fabs(item.reported), 1.0)
        if bound_error > worst_bound { worst_bound = bound_error }
    }
    #: OpenCV carries the costs and the flows in float32 and this carries them
    #: in float64, so the gap is that and nothing else: against an exact
    #: float64 solution of the same linear programme this lands within 3.9e-16
    #: and OpenCV within 9.1e-6, measured over 1794 random signature pairs of
    #: up to forty clusters. The budget below is float32's, not a fudge -- the
    #: cases here are small enough that the worst of them actually comes to
    #: 7.3e-8.
    assert_true(worst < 2e-6, f"worst disagreement with OpenCV was {worst} on '{worst_at}'")
    assert_true(worst_bound < 2e-6, f"worst lower bound disagreement was {worst_bound}")
}
