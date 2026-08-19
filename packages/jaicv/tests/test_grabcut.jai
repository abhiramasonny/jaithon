#: GrabCut: the max-flow solver on its own, then the whole segmentation.
#:
#: `FlowGraph` is checked first, and separately, against small graphs whose
#: answer can be worked out by hand -- a single bottleneck, two disjoint
#: paths, a disconnected pair, CLRS's own textbook example (which needs the
#: reverse-arc trick to reach its optimum of 23), and a graph whose minimum
#: cut is not unique. A wrong min-cut still draws a plausible-looking
#: segmentation, so this is the part that has to be right before the rest of
#: the file means anything.
#:
#: The segmentation itself is checked on a bright rectangle over a dark,
#: noisy ground: `GC_INIT_WITH_RECT` from a loose rectangle, `GC_INIT_WITH_MASK`
#: from a rough mask, and a split call through `GC_EVAL` that has to land on
#: the same answer as running the same iterations in one call, which is what
#: "refines rather than resets" comes down to. Each is graded by intersection
#: over union against the true rectangle, against a threshold stated up front.
#:
#: Last is OpenCV. `EXPECTED_MASK` is what `cv2.grabCut` 5.0.0 returned for
#: the rectangle case below, recorded through `~/.venvs/scratch/bin/python`
#: and written out rather than computed, so this test runs without OpenCV
#: installed. GrabCut is iterative and the two implementations do not share a
#: random seed, so exact agreement is not a given -- what is checked is the
#: intersection over union between the two masks, and it comes out at 1.0 on
#: this image: every one of its two thousand pixels lands on the same side.
import std.math
from std.test import assert_eq, assert_near, assert_throws, assert_true
from jaicv.core.mat import Mat
from jaicv.core.types import CV_64FC1, CV_MAKETYPE, CV_8U, Rect
from jaicv.imgproc.grabcut import FlowGraph, grab_cut
from jaicv.imgproc.grabcut import GC_BGD, GC_FGD, GC_PR_BGD, GC_PR_FGD
from jaicv.imgproc.grabcut import GC_EVAL, GC_EVAL_FREEZE_MODEL, GC_INIT_WITH_MASK, GC_INIT_WITH_RECT

# --- Max-flow, on its own -----------------------------------------------------
fn test_max_flow_single_bottleneck() -> void {
    let graph = FlowGraph(3)
    graph.add_edge(0, 1, 5.0, 0.0)
    graph.add_edge(1, 2, 3.0, 0.0)
    assert_near(graph.max_flow(0, 2), 3.0, message: "a chain caps out at its narrowest link")
    assert_eq(graph.source_side(0), [true, true, false])
}

fn test_max_flow_two_disjoint_paths() -> void {
    let graph = FlowGraph(4)
    graph.add_edge(0, 1, 5.0, 0.0)
    graph.add_edge(1, 3, 5.0, 0.0)
    graph.add_edge(0, 2, 7.0, 0.0)
    graph.add_edge(2, 3, 3.0, 0.0)
    assert_near(graph.max_flow(0, 3), 8.0, message: "min(5,5) + min(7,3)")
    assert_eq(graph.source_side(0), [true, false, true, false])
}

fn test_max_flow_disconnected_graph_is_zero() -> void {
    let graph = FlowGraph(4)
    graph.add_edge(0, 1, 5.0, 0.0)
    graph.add_edge(2, 3, 5.0, 0.0)
    assert_near(graph.max_flow(0, 3), 0.0)
    #: Node 1 stays reachable from the source -- its edge was never touched,
    #: only never led anywhere -- while node 2 and the sink never were.
    assert_eq(graph.source_side(0), [true, true, false, false])
}

fn test_max_flow_classic_textbook_graph() -> void {
    #: Cormen, Leiserson, Rivest and Stein's own max-flow example (figure
    #: 26.6 in the third edition): six nodes, source 0 and sink 5, whose
    #: maximum flow the text works out by hand to be 23. Reaching it needs
    #: the residual graph's reverse arcs: a first augmenting path sends flow
    #: 1->3, and only pushing flow backward along that arc later, through
    #: `3->2->1->...`, reaches the true optimum instead of stopping short.
    let graph = FlowGraph(6)
    graph.add_edge(0, 1, 16.0, 0.0)
    graph.add_edge(0, 2, 13.0, 0.0)
    graph.add_edge(1, 3, 12.0, 0.0)
    graph.add_edge(2, 1, 4.0, 0.0)
    graph.add_edge(2, 4, 14.0, 0.0)
    graph.add_edge(3, 2, 9.0, 0.0)
    graph.add_edge(3, 5, 20.0, 0.0)
    graph.add_edge(4, 3, 7.0, 0.0)
    graph.add_edge(4, 5, 4.0, 0.0)
    assert_near(graph.max_flow(0, 5), 23.0)
    assert_eq(graph.source_side(0), [true, true, true, false, true, false])
}

fn test_max_flow_non_unique_min_cut() -> void {
    #: Two disjoint source-to-sink paths of equal capacity ten. A cut can be
    #: taken at either end, or one edge from each path, and several of those
    #: are worth exactly twenty -- the minimum cut is not unique. What has to
    #: hold regardless is the flow value, and Dinic's algorithm always
    #: saturates both of the source's own edges on the way there, so nothing
    #: past the source stays reachable in the residual graph.
    let graph = FlowGraph(4)
    graph.add_edge(0, 1, 10.0, 0.0)
    graph.add_edge(0, 2, 10.0, 0.0)
    graph.add_edge(1, 3, 10.0, 0.0)
    graph.add_edge(2, 3, 10.0, 0.0)
    assert_near(graph.max_flow(0, 3), 20.0)
    assert_eq(graph.source_side(0), [true, false, false, false])
}

fn test_max_flow_undirected_edge_carries_both_ways() -> void {
    #: `add_edge(u, v, w, w)` is how GrabCut's neighbour links are built --
    #: an undirected edge of weight `w`, which has to carry `w` whichever
    #: direction the flow needs, not `w` one way and nothing back.
    let graph = FlowGraph(3)
    graph.add_edge(0, 1, 4.0, 4.0)
    graph.add_edge(1, 2, 4.0, 4.0)
    assert_near(graph.max_flow(0, 2), 4.0)
    assert_eq(graph.source_side(0), [true, false, false])
}

fn test_flow_graph_rejects_bad_construction() -> void {
    assert_throws(ValueError, || FlowGraph(0))
    assert_throws(ValueError, || FlowGraph(-1))
}

fn test_flow_graph_rejects_terminals_outside_the_graph() -> void {
    let graph = FlowGraph(3)
    graph.add_edge(0, 1, 1.0, 0.0)
    assert_throws(ValueError, || graph.max_flow(-1, 2))
    assert_throws(ValueError, || graph.max_flow(0, 3))
}

# --- A synthetic image: a bright rectangle on a dark, noisy ground ------------
let ROWS = 40
let COLS = 50
let RECT_X = 15
let RECT_Y = 10
let RECT_W = 20
let RECT_H = 20
#: The intersection over union a recovered foreground has to clear against
#: the true rectangle -- comfortably below the 1.0 this actually reaches, so
#: the bound is testing the algorithm rather than restating its answer.
let IOU_THRESHOLD = 0.85

#: A rectangle of `fg` over a ground of `bg`, both dithered by a small,
#: deterministic triangular-ish spread from a linear congruential generator
#: -- enough texture that a flat-colour shortcut could not pass, not so much
#: that the two colour populations overlap.
fn synthetic_rect_image(fg: float, bg: float, noise: int, seed: int) -> Mat {
    var state = seed
    var values: list[float] = []
    for y in 0..ROWS {
        for x in 0..COLS {
            state = state *% 1103515245 +% 12345 & 0x7FFFFFFF
            let jitter = float(state % noise)
            let inside = x >= RECT_X and x < RECT_X + RECT_W and y >= RECT_Y and y < RECT_Y + RECT_H
            let base = if inside { fg } else { bg }
            let value = min(base + jitter, 255.0)
            for _channel in 0..3 { values.push(value) }
        }
    }
    return Mat.from_list(ROWS, COLS, CV_MAKETYPE(CV_8U, 3), values)
}

#: A mask that only roughly agrees with the true rectangle: background
#: outside a margin around it, probable background in a ring just inside
#: that margin, probable foreground through most of the rectangle, and a
#: small hard-foreground core in the middle -- the kind of scribble a rough
#: manual mask leaves rather than a pixel-perfect box.
fn rough_mask() -> Mat {
    var values: list[float] = []
    for y in 0..ROWS {
        for x in 0..COLS {
            var label = GC_BGD
            if x >= RECT_X - 3 and
                x < RECT_X + RECT_W + 3 and
                y >= RECT_Y - 3 and
                y < RECT_Y + RECT_H + 3 {
                label = GC_PR_BGD
            }
            if x >= RECT_X + 2 and
                x < RECT_X + RECT_W - 2 and
                y >= RECT_Y + 2 and
                y < RECT_Y + RECT_H - 2 {
                label = GC_PR_FGD
            }
            if x >= RECT_X + 6 and
                x < RECT_X + RECT_W - 6 and
                y >= RECT_Y + 6 and
                y < RECT_Y + RECT_H - 6 {
                label = GC_FGD
            }
            values.push(float(label))
        }
    }
    return Mat.from_list(ROWS, COLS, CV_MAKETYPE(CV_8U, 1), values)
}

fn is_foreground(label: int) -> bool { return label == GC_FGD or label == GC_PR_FGD }

#: Intersection over union of a mask's foreground against the true rectangle.
fn iou_against_rect(mask_values: list[float]) -> float {
    var intersection = 0
    var union_count = 0
    for y in 0..ROWS {
        for x in 0..COLS {
            let predicted = is_foreground(int(mask_values[y * COLS + x]))
            let truth = x >= RECT_X and x < RECT_X + RECT_W and y >= RECT_Y and y < RECT_Y + RECT_H
            if predicted or truth { union_count += 1 }
            if predicted and truth { intersection += 1 }
        }
    }
    return float(intersection) / float(union_count)
}

fn empty_mat(type_code: int) -> Mat { return Mat(0, 0, type_code) }

fn test_grab_cut_rect_recovers_the_rectangle() -> void {
    let image = synthetic_rect_image(200.0, 20.0, 15, 12345)
    let rect = Rect(RECT_X - 5, RECT_Y - 5, RECT_W + 10, RECT_H + 10)
    let (mask, bgd_model, fgd_model) = grab_cut(
        image,
        empty_mat(CV_MAKETYPE(CV_8U, 1)),
        rect,
        empty_mat(CV_64FC1),
        empty_mat(CV_64FC1),
        4,
        GC_INIT_WITH_RECT
    )
    let iou = iou_against_rect(mask.to_list())
    assert_true(iou >= IOU_THRESHOLD, f"IoU {iou} fell short of {IOU_THRESHOLD}")
    assert_eq(bgd_model.rows, 1)
    assert_eq(bgd_model.cols, 65)
    assert_eq(fgd_model.rows, 1)
    assert_eq(fgd_model.cols, 65)
}

fn test_grab_cut_mask_recovers_the_rectangle() -> void {
    let image = synthetic_rect_image(200.0, 20.0, 15, 12345)
    let (mask, _bgd, _fgd) = grab_cut(
        image,
        rough_mask(),
        Rect(0, 0, 0, 0),
        empty_mat(CV_64FC1),
        empty_mat(CV_64FC1),
        4,
        GC_INIT_WITH_MASK
    )
    let iou = iou_against_rect(mask.to_list())
    assert_true(iou >= IOU_THRESHOLD, f"IoU {iou} fell short of {IOU_THRESHOLD}")
}

fn test_grab_cut_eval_refines_rather_than_resets() -> void {
    let image = synthetic_rect_image(190.0, 30.0, 25, 42)
    let rect = Rect(RECT_X - 5, RECT_Y - 5, RECT_W + 10, RECT_H + 10)

    let (one_shot_mask, one_shot_bgd, _one_shot_fgd) = grab_cut(
        image,
        empty_mat(CV_MAKETYPE(CV_8U, 1)),
        rect,
        empty_mat(CV_64FC1),
        empty_mat(CV_64FC1),
        4,
        GC_INIT_WITH_RECT
    )

    #: One iteration first, so its model is captured partway through --
    #: then three more, fed back in through `GC_EVAL`. If that resets the
    #: mixtures instead of continuing them, this diverges from the one-shot
    #: run above; matching it bit for bit is what "refines" has to mean.
    let (first_mask, first_bgd, first_fgd) = grab_cut(
        image,
        empty_mat(CV_MAKETYPE(CV_8U, 1)),
        rect,
        empty_mat(CV_64FC1),
        empty_mat(CV_64FC1),
        1,
        GC_INIT_WITH_RECT
    )
    let (split_mask, split_bgd, _split_fgd) = grab_cut(
        image,
        first_mask,
        rect,
        first_bgd,
        first_fgd,
        3,
        GC_EVAL
    )

    assert_eq(split_mask.to_list(), one_shot_mask.to_list())
    let split_values = split_bgd.to_list()
    let one_shot_values = one_shot_bgd.to_list()
    for index in 0..split_values.len() {
        assert_near(
            split_values[index],
            one_shot_values[index],
            1e-4,
            f"model entry {index} drifted between the one-shot and split runs"
        )
    }

    #: And the partial, one-iteration model really was partial: it moved on
    #: from there rather than the run above trivially having nothing left to
    #: learn.
    let first_values = first_bgd.to_list()
    var moved = false
    for index in 0..first_values.len() {
        if math.fabs(first_values[index] - one_shot_values[index]) > 1e-3 { moved = true }
    }
    assert_true(moved, "the model should still have been refining after one iteration")
}

fn test_grab_cut_eval_freeze_model_leaves_the_model_untouched() -> void {
    let image = synthetic_rect_image(190.0, 30.0, 25, 42)
    let rect = Rect(RECT_X - 5, RECT_Y - 5, RECT_W + 10, RECT_H + 10)
    let (mask, bgd_model, fgd_model) = grab_cut(
        image,
        empty_mat(CV_MAKETYPE(CV_8U, 1)),
        rect,
        empty_mat(CV_64FC1),
        empty_mat(CV_64FC1),
        1,
        GC_INIT_WITH_RECT
    )
    let (_frozen_mask, frozen_bgd, frozen_fgd) = grab_cut(
        image,
        mask,
        rect,
        bgd_model,
        fgd_model,
        5,
        GC_EVAL_FREEZE_MODEL
    )
    assert_eq(frozen_bgd.to_list(), bgd_model.to_list())
    assert_eq(frozen_fgd.to_list(), fgd_model.to_list())
}

fn test_grab_cut_checks_its_arguments() -> void {
    let image = synthetic_rect_image(200.0, 20.0, 15, 12345)
    let rect = Rect(RECT_X - 5, RECT_Y - 5, RECT_W + 10, RECT_H + 10)
    let one_channel = Mat.zeros(ROWS, COLS, CV_8U)
    assert_throws(
        ValueError,
        || grab_cut(one_channel, empty_mat(CV_8U), rect, empty_mat(CV_64FC1), empty_mat(CV_64FC1), 1)
    )

    let image3 = synthetic_rect_image(200.0, 20.0, 15, 12345)
    assert_throws(
        ValueError,
        || grab_cut(image3, empty_mat(CV_MAKETYPE(CV_8U, 1)), rect, empty_mat(CV_64FC1), empty_mat(CV_64FC1), 1, 7)
    )

    let wrong_size_mask = Mat.zeros(ROWS + 1, COLS, CV_8U)
    assert_throws(
        ValueError,
        || grab_cut(image3, wrong_size_mask, rect, empty_mat(CV_64FC1), empty_mat(CV_64FC1), 1, GC_INIT_WITH_MASK)
    )

    let bad_values_mask = Mat.full(ROWS, COLS, CV_8U, 9.0)
    assert_throws(
        ValueError,
        || grab_cut(image3, bad_values_mask, rect, empty_mat(CV_64FC1), empty_mat(CV_64FC1), 1, GC_INIT_WITH_MASK)
    )

    assert_throws(
        ValueError,
        || grab_cut(image3, rough_mask(), rect, empty_mat(CV_64FC1), empty_mat(CV_64FC1), 1, GC_EVAL)
    )
}

# --- Against OpenCV ------------------------------------------------------------

#: What `cv2.grabCut` 5.0.0 returned for the image and rectangle built by
#: `synthetic_rect_image(200.0, 20.0, 15, 12345)` and
#: `Rect(RECT_X - 5, RECT_Y - 5, RECT_W + 10, RECT_H + 10)`, four iterations,
#: `GC_INIT_WITH_RECT`. `B` is `GC_BGD`, `.` is `GC_PR_BGD`, `f` is
#: `GC_PR_FGD` -- a rectangle-seeded mask never gets a hard `GC_FGD` pixel,
#: since nothing set one going in.
let EXPECTED_MASK = [
    "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB",
    "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB",
    "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB",
    "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB",
    "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB",
    "BBBBBBBBBB..............................BBBBBBBBBB",
    "BBBBBBBBBB..............................BBBBBBBBBB",
    "BBBBBBBBBB..............................BBBBBBBBBB",
    "BBBBBBBBBB..............................BBBBBBBBBB",
    "BBBBBBBBBB..............................BBBBBBBBBB",
    "BBBBBBBBBB.....ffffffffffffffffffff.....BBBBBBBBBB",
    "BBBBBBBBBB.....ffffffffffffffffffff.....BBBBBBBBBB",
    "BBBBBBBBBB.....ffffffffffffffffffff.....BBBBBBBBBB",
    "BBBBBBBBBB.....ffffffffffffffffffff.....BBBBBBBBBB",
    "BBBBBBBBBB.....ffffffffffffffffffff.....BBBBBBBBBB",
    "BBBBBBBBBB.....ffffffffffffffffffff.....BBBBBBBBBB",
    "BBBBBBBBBB.....ffffffffffffffffffff.....BBBBBBBBBB",
    "BBBBBBBBBB.....ffffffffffffffffffff.....BBBBBBBBBB",
    "BBBBBBBBBB.....ffffffffffffffffffff.....BBBBBBBBBB",
    "BBBBBBBBBB.....ffffffffffffffffffff.....BBBBBBBBBB",
    "BBBBBBBBBB.....ffffffffffffffffffff.....BBBBBBBBBB",
    "BBBBBBBBBB.....ffffffffffffffffffff.....BBBBBBBBBB",
    "BBBBBBBBBB.....ffffffffffffffffffff.....BBBBBBBBBB",
    "BBBBBBBBBB.....ffffffffffffffffffff.....BBBBBBBBBB",
    "BBBBBBBBBB.....ffffffffffffffffffff.....BBBBBBBBBB",
    "BBBBBBBBBB.....ffffffffffffffffffff.....BBBBBBBBBB",
    "BBBBBBBBBB.....ffffffffffffffffffff.....BBBBBBBBBB",
    "BBBBBBBBBB.....ffffffffffffffffffff.....BBBBBBBBBB",
    "BBBBBBBBBB.....ffffffffffffffffffff.....BBBBBBBBBB",
    "BBBBBBBBBB.....ffffffffffffffffffff.....BBBBBBBBBB",
    "BBBBBBBBBB..............................BBBBBBBBBB",
    "BBBBBBBBBB..............................BBBBBBBBBB",
    "BBBBBBBBBB..............................BBBBBBBBBB",
    "BBBBBBBBBB..............................BBBBBBBBBB",
    "BBBBBBBBBB..............................BBBBBBBBBB",
    "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB",
    "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB",
    "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB",
    "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB",
    "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB",
]

fn decode_symbol(symbol: str) -> int {
    if symbol == "B" { return GC_BGD }
    if symbol == "F" { return GC_FGD }
    if symbol == "." { return GC_PR_BGD }
    return GC_PR_FGD
}

fn decode_mask(rows: list[str]) -> list[int] {
    var out: list[int] = []
    for row in rows {
        for index in 0..row.len() { out.push(decode_symbol(row[index:index + 1])) }
    }
    return out
}

fn test_grab_cut_matches_opencv() -> void {
    let image = synthetic_rect_image(200.0, 20.0, 15, 12345)
    let rect = Rect(RECT_X - 5, RECT_Y - 5, RECT_W + 10, RECT_H + 10)
    let (mask, _bgd, _fgd) = grab_cut(
        image,
        empty_mat(CV_MAKETYPE(CV_8U, 1)),
        rect,
        empty_mat(CV_64FC1),
        empty_mat(CV_64FC1),
        4,
        GC_INIT_WITH_RECT
    )
    let ours = mask.to_list()
    let theirs = decode_mask(EXPECTED_MASK)
    assert_eq(ours.len(), theirs.len())

    var intersection = 0
    var union_count = 0
    for index in 0..ours.len() {
        let mine = is_foreground(int(ours[index]))
        let cv = is_foreground(theirs[index])
        if mine or cv { union_count += 1 }
        if mine and cv { intersection += 1 }
    }
    let iou = float(intersection) / float(union_count)
    #: Recorded agreement on this image: 1.0 -- every one of the two thousand
    #: pixels lands on the same side as OpenCV's own cut. GrabCut does not
    #: promise that in general, so the bound below is real slack, not the
    #: number this happens to hit.
    assert_true(iou >= 0.95, f"only {iou} IoU against OpenCV's own cut")

    var same = 0
    for index in 0..ours.len() {
        if int(ours[index]) == theirs[index] { same += 1 }
    }
    assert_true(
        float(same) / float(ours.len()) >= 0.95,
        f"only {float(same) / float(ours.len())} of pixels matched OpenCV's mask exactly"
    )
}
