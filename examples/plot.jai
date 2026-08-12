#:     jaithon run examples/plot.jai            # writes plot.svg and plot.png
#:     jaithon run examples/plot.jai -- --show     # opens a window you can pan
#:     jaithon run examples/plot.jai -- --animate  # a live travelling wave
import std.math
from std.random import Random

from jaiplot import Figure, MarkerShape, linspace, logspace, save
from jaiplot.backend.window import animate, show

fn main(args: list[str]) -> int {
    let figure = Figure(width: 960, height: 680)
    let grid = figure.subplots(rows: 2, cols: 2)

    _waves(grid[0][0])
    _distribution(grid[0][1])
    _growth(grid[1][0])
    _band(grid[1][1])

    if args.contains("--animate") {
        _animation()
        return 0
    }

    if args.contains("--show") {
        let single = Figure(width: 900, height: 600)
        _waves(single.axes())
        show(single, "jaiplot — drag to pan, wheel to zoom, r resets")
        return 0
    }

    save(figure, "plot.svg")
    save(figure, "plot.png")
    print("wrote plot.svg and plot.png")
    return 0
}

fn _waves(axes: any) -> void {
    let x = linspace(0.0, math.TAU, 240)
    axes.line(x, x.map(|v| math.sin(v)), label: "sin")
    axes.line(x, x.map(|v| math.cos(v) * 0.5), label: "cos/2", dash: [7.0, 4.0])
    axes.scatter(
        [math.PI * 0.5, math.PI * 1.5],
        [1.0, -1.0],
        label: "turning points",
        shape: MarkerShape.Diamond,
        size: 8.0
    )
    axes.set_title("waves")
    axes.set_xlabel("radians")
    axes.set_ylabel("amplitude")
    axes.grid(true)
    axes.legend()
}

fn _distribution(axes: any) -> void {
    let rng = Random.with_seed(20260805)
    var sample = []
    for _index in 0..2000 { sample.push(rng.normal(0.0, 4.0)) }
    axes.hist(sample, bins: 32, label: "normal")
    axes.set_title("distribution")
    axes.set_xlabel("value")
    axes.set_ylabel("count")
}

fn _growth(axes: any) -> void {
    let x = logspace(0.0, 4.0, 120)
    axes.line(x, x.map(|v| math.log10(v) + 1.0), label: "log n")
    axes.line(x, x.map(|v| v ** 0.5), label: "sqrt n")
    axes.line(x, x.map(|v| v * 0.02), label: "n/50")
    axes.set_xscale("log")
    axes.set_title("growth")
    axes.set_xlabel("n")
    axes.grid(true)
    axes.legend()
}

fn _band(axes: any) -> void {
    let t = linspace(0.0, 12.0, 160)
    var centre = []
    var lower = []
    var upper = []
    for (index, value) in t.enumerate() {
        if index >= 62 and index < 78 {
            centre.push(math.NAN)
            lower.push(math.NAN)
            upper.push(math.NAN)
        } else {
            let mean = math.sin(value) * math.exp(-value * 0.08)
            let spread = 0.12 + 0.04 * math.cos(value * 2.0)
            centre.push(mean)
            lower.push(mean - spread)
            upper.push(mean + spread)
        }
    }
    axes.fill_between(t, lower, upper, label: "spread")
    axes.line(t, centre, label: "mean")
    axes.set_title("signal with a dropout")
    axes.set_xlabel("seconds")
    axes.legend()
}

fn _animation() -> void {
    let x = linspace(0.0, math.TAU * 2.0, 220)
    let figure = Figure(width: 640, height: 420)
    let axes = figure.axes()
    let _x_limits = axes.set_xlim(0.0, math.TAU * 2.0)
    let _y_limits = axes.set_ylim(-1.4, 1.4)
    let _heading = axes.set_title("travelling wave")
    let _rules = axes.grid(true)
    let wave = axes.line(x, x.map(|v| math.sin(v)), label: "sin(x - t)")
    let envelope = axes.line(x, x.map(|_v| 0.0), label: "envelope", dash: [5.0, 3.0])
    let _key = axes.legend()

    animate(figure, fn(frame: int) -> bool {
        let phase = frame.to_float() * 0.08
        wave.set_data(x, x.map(|v| math.sin(v - phase)))
        envelope.set_data(x, x.map(|v| math.sin(v * 0.25 - phase * 0.5) * 1.2))
        return true
    }, title: "jaiplot — escape to close")
}
