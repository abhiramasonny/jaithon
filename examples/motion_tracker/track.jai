#: Follow points through a moving scene with sparse optical flow.
#:
#:     jaithon run examples/motion_tracker/track.jai
#:
#: The scene is generated rather than filmed, so this runs from a clean
#: checkout with nothing to download: a textured backdrop drifts and turns
#: under the camera while two discs cross it on their own paths. jaicv picks
#: the corners worth following with `good_features_to_track`, then carries
#: them from each frame to the next with `calc_optical_flow_pyr_lk`, and the
#: trails are drawn where the points have been.
#:
#: The generated backdrop matters as much as it does in `digit_trainer`. A
#: flat or repeating background gives the tracker nothing to lock onto and
#: every point drifts; the noise and the blobs are what make the corners
#: distinct enough to be found again a frame later.
import jaicv as cv
from jaicv import Mat, Point, Point2f, Scalar, Size
from std.gpu import is_available
from std.random import Random
from std.time import Instant

let WIDTH = 640
let HEIGHT = 480
let FRAMES = 24
#: How many frames of history each trail keeps.
let TRAIL = 12
#: A point whose windows still differ by more than this has not really been
#: found again, whatever the tracker says.
let RESIDUAL_LIMIT = 30.0

#: A backdrop with enough texture to have corners in it.
fn backdrop(rng: Random) -> Mat {
    #: A grainy base rather than black, so that the corners are spread over
    #: the whole frame instead of clustering on whatever was drawn.
    var values: list[float] = []
    for _index in 0..WIDTH * HEIGHT * 3 { values.push(50.0 + rng.next_float() * 60.0) }
    let canvas = Mat.from_list(HEIGHT, WIDTH, cv.CV_8UC3, values)
    #: Rectangles and discs rather than noise. Noise alone has corners
    #: everywhere and none of them survive a rotation; edges and junctions do.
    for _shape in 0..120 {
        let x = int(rng.next_float() * float(WIDTH))
        let y = int(rng.next_float() * float(HEIGHT))
        let w = 12 + int(rng.next_float() * 70.0)
        let h = 12 + int(rng.next_float() * 70.0)
        let tone = Scalar(
            40.0 + rng.next_float() * 200.0,
            40.0 + rng.next_float() * 200.0,
            40.0 + rng.next_float() * 200.0
        )
        if rng.next_float() < 0.55 {
            cv.rectangle(canvas, Point(x, y), Point(x + w, y + h), tone, -1)
        } else {
            cv.circle(canvas, Point(x, y), w // 2, tone, -1)
        }
    }
    return cv.gaussian_blur(canvas, Size(3, 3), 0.0)
}

#: The scene at time `t`: the backdrop moved under the camera, and two discs
#: crossing it.
fn frame_at(scene: Mat, t: int) -> Mat {
    let turn = float(t) * 0.45
    let centre = Point2f(float(WIDTH) * 0.5, float(HEIGHT) * 0.5)
    let move = cv.get_rotation_matrix_2d(centre, turn, 1.0)
    #: The pan, added straight into the affine's translation column.
    let shifted = move.clone()
    shifted.set_at(0, 2, shifted.at(0, 2) + float(t) * 1.7)
    shifted.set_at(1, 2, shifted.at(1, 2) + float(t) * 0.9)
    let out = cv.warp_affine(scene, shifted, Size(WIDTH, HEIGHT))
    move.release()
    shifted.release()
    let first = Point(90 + t * 9, 140 + t * 4)
    let second = Point(WIDTH - 70 - t * 7, 330 - t * 5)
    cv.circle(out, first, 26, Scalar(250.0, 250.0, 250.0), -1)
    cv.circle(out, second, 20, Scalar(20.0, 20.0, 20.0), -1)
    return out
}

fn main() -> int {
    if not is_available() {
        print("this example needs a GPU")
        return 0
    }
    let rng = Random.with_seed(7)
    let scene = backdrop(rng)
    var flow_seconds = 0.0
    let started = Instant.now()

    var previous = cv.cvt_color(frame_at(scene, 0), cv.COLOR_BGR2GRAY)
    #: The corners are chosen once, on the first frame, and then followed.
    #: Choosing again every frame would be a different program -- it would
    #: never know that this point is that point.
    var points = cv.good_features_to_track(previous, 120, 0.02, 12.0)
    let started_with = points.len()
    var trails: list[list[Point2f]] = []
    for point in points { trails.push([point]) }
    var alive: list[bool] = []
    for _index in 0..points.len() { alive.push(true) }

    var last_colour = frame_at(scene, 0)
    for t in 1..FRAMES {
        let colour = frame_at(scene, t)
        let grey = cv.cvt_color(colour, cv.COLOR_BGR2GRAY)
        let clock = Instant.now()
        let (moved, found, residual) = cv.calc_optical_flow_pyr_lk(previous, grey, points)
        flow_seconds += clock.elapsed().to_seconds()
        var next_points: list[Point2f] = []
        for (index, landed) in moved.enumerate() {
            #: Two ways to lose a point: the tracker says it left the frame,
            #: or it claims a match whose windows do not really agree.
            let kept = alive[index] and found[index] and residual[index] < RESIDUAL_LIMIT
            alive[index] = kept
            if kept {
                trails[index].push(landed)
                if trails[index].len() > TRAIL { trails[index].remove_at(0) }
                next_points.push(landed)
            } else {
                next_points.push(points[index])
            }
        }
        points = next_points
        previous.release()
        previous = grey
        last_colour.release()
        last_colour = colour
    }

    var still_here = 0
    for held in alive {
        if held { still_here += 1 }
    }
    let seconds = started.elapsed().to_seconds()

    #: The last frame with every surviving trail drawn onto it.
    for (index, trail) in trails.enumerate() {
        if not alive[index] { continue }
        for step in 1..trail.len() {
            let start = trail[step - 1]
            let to = trail[step]
            #: Older parts of a trail are drawn fainter, so the direction of
            #: travel reads without an arrowhead.
            let age = float(step) / float(trail.len())
            cv.line(
                last_colour,
                Point(int(start.x), int(start.y)),
                Point(int(to.x), int(to.y)),
                Scalar(60.0 + 80.0 * age, 255.0 * age, 255.0 * (1.0 - age)),
                2
            )
        }
        let head = trail[trail.len() - 1]
        cv.circle(last_colour, Point(int(head.x), int(head.y)), 3, Scalar(0.0, 255.0, 255.0), -1)
    }
    cv.put_text(
        last_colour,
        f"{still_here}/{started_with} tracked over {FRAMES} frames",
        Point(12, 28),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        Scalar(255.0, 255.0, 255.0),
        2
    )

    let out = "motion_trails.png"
    let ok = cv.imwrite(out, last_colour)
    print(f"tracked {still_here} of {started_with} corners across {FRAMES} frames")
    let steps = float(FRAMES - 1)
    print(f"{flow_seconds * 1000.0 / steps:.1f} ms a frame following them")
    print(f"{(seconds - flow_seconds) * 1000.0 / steps:.1f} ms a frame drawing the scene to follow")
    if ok { print(f"wrote {out}") }
    return 0
}
