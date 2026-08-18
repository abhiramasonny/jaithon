import std.gpu as gpu
import std.math
from jaitensor import Activation, Adam, Dense, Sequential, Tensor
from jaitensor import device_name, is_available, matmul, multiply, ones, scale, subtract, transpose
from jaiplot import Color, Figure, linspace, save
from std.io import Path
from std.random import Random

fn sine_rows(samples: int, noise: float, seed: int) -> tuple[list[float], list[float]] {
    let rng = Random.with_seed(seed)
    var xs: list[float] = []
    var ys: list[float] = []
    for _index in 0..samples {
        let x = rng.float_range(-math.PI, math.PI)
        xs.push(x)
        ys.push(math.sin(x) + rng.gauss(0.0, noise))
    }
    return (xs, ys)
}

fn regressor() -> Sequential {
    return Sequential(
        [
            Dense(64, activation: Activation.Tanh),
            Dense(64, activation: Activation.Tanh),
            Dense(1, activation: Activation.Linear),
        ]
    )
}

fn parameters_of(model: Sequential) -> list[Tensor] {
    var parameters: list[Tensor] = []
    for layer in model.layers { parameters.extend(layer.parameters()) }
    return parameters
}

fn forward(model: Sequential, x: Tensor) -> Tensor {
    var current = x
    for layer in model.layers { current = layer.forward(current) }
    return current
}

fn mean_squared(prediction: Tensor, target: Tensor) -> tuple[Tensor, Tensor] {
    let error = subtract(prediction, target)
    let squared = multiply(error, error)
    let weights = ones([prediction.shape[0], 1])
    let total = matmul(transpose(weights), squared)
    let loss = scale(total, 1.0 / float(prediction.shape[0]))
    return (loss, weights)
}

fn zero_grad(parameters: list[Tensor]) -> void {
    for parameter in parameters {
        let gradient = parameter.grad
        if gradient is not null {
            gradient.free()
            parameter.grad = null
        }
    }
}

fn draw_report(
    output: Path,
    xs: list[float],
    ys: list[float],
    curve_x: list[float],
    curve_y: list[float],
    losses: list[float]
) -> void {
    let figure = Figure(width: 1100, height: 480)
    let panels = figure.subplots(1, 2)

    let fit = panels[0][0]
    let truth_x = linspace(-math.PI, math.PI, 400)
    let truth_y = [math.sin(value) for value in truth_x]
    fit.scatter(xs, ys, label: "noisy samples", color: Color.from_hex("#2463eb"), size: 4.5)
    fit.line(truth_x, truth_y, label: "sin(x)", color: Color.from_hex("#888888"), dash: [7.0, 4.0])
    fit.line(curve_x, curve_y, label: "network", color: Color.from_hex("#e34b4b"), width: 2.4)
    fit.set_xlim(-math.PI, math.PI)
    fit.set_ylim(-1.8, 1.8)
    fit.set_title("fitted sine")
    fit.set_xlabel("x")
    fit.set_ylabel("y")
    fit.grid(true)
    fit.legend()

    let epochs = [float(index + 1) for index in 0..losses.len()]
    let curves = panels[0][1]
    curves.line(epochs, losses, label: "training loss", color: Color.from_hex("#15956f"))
    curves.set_title("mean squared error")
    curves.set_xlabel("epoch")
    curves.set_ylabel("loss")
    curves.grid(true)
    curves.legend()

    save(figure, output.join("sine_regression.svg").to_str())
    save(figure, output.join("sine_regression.png").to_str())
}

fn main(args: list[str]) -> int {
    let epochs = if args.len() > 1 { int(args[1]) } else { 200 }
    let output = Path(if args.len() > 2 { args[2] } else { "data/sine-regression" })
    output.mkdir(parents: true, exist_ok: true)

    if not is_available() { throw RuntimeError("sine training needs a GPU") }
    let (xs, ys) = sine_rows(256, 0.12, 41)
    let features = Tensor([xs.len(), 1], xs)
    let targets = Tensor([ys.len(), 1], ys)
    defer { features.free() }
    defer { targets.free() }

    print(f"GPU: {device_name()}")
    print(f"fitting sin(x) on {xs.len()} noisy points")
    let model = regressor()
    defer { model.free() }
    model.build(1)
    let optimizer = Adam(learning_rate: 0.01)
    let parameters = parameters_of(model)
    var losses: list[float] = []

    for epoch in 1..=epochs {
        let prediction = forward(model, features)
        let (loss, weights) = mean_squared(prediction, targets)
        let graph = loss.backward()
        optimizer.step(parameters)
        gpu.synchronize()
        let value = loss.download()[0]
        losses.push(value)
        if epoch == 1 or epoch == epochs or epoch % 20 == 0 {
            print(f"epoch {epoch}/{epochs}: loss={value:.5f}")
        }
        loss.free()
        weights.free()
        for tensor in graph { tensor.free() }
        zero_grad(parameters)
    }

    let curve_x = linspace(-math.PI, math.PI, 400)
    let query = Tensor([curve_x.len(), 1], curve_x)
    defer { query.free() }
    let predicted = model.predict(query)
    defer { predicted.free() }
    let curve_y = predicted.download()

    draw_report(output, xs, ys, curve_x, curve_y, losses)
    print(f"wrote {output.join("sine_regression.svg")}")
    print(f"wrote {output.join("sine_regression.png")}")
    return 0
}
