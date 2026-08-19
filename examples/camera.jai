import jaicv as cv
import std.time as time

fn main() {
    let camera = cv.VideoCapture(0)
    if not camera.is_opened() {
        print(f"camera failed to open: {camera.why()}")
        return
    }

    var last = time.Instant.now()

    loop {
        let (ok, frame) = camera.read()
        if not ok { break }

        let elapsed = last.elapsed().to_seconds()
        last = time.Instant.now()
        let fps = if elapsed > 0.0 { 1.0 / elapsed } else { 0.0 }

        cv.set_window_title("camera", f"camera — {fps:.1f} fps")
        cv.imshow("camera", frame)

        if cv.wait_key(1) == 27 { break }
    }

    camera.release()
    cv.destroy_all_windows()
}
