import std.gpu as gpu
from std.gui import Window
from std.gui.color import rgb
from std.gui.input import Input, Key
from std.random import Random

const CELL = 10
const COLS = 40
const ROWS = 30
const WIDTH = COLS * CELL
const HEIGHT = ROWS * CELL

let SHADER = """#include <metal_stdlib>
using namespace metal;
kernel void draw(device const float* cells [[buffer(0)]],
                 device float* frame [[buffer(1)]],
                 constant uint& width [[buffer(2)]],
                 constant uint& cell [[buffer(3)]],
                 constant uint& cols [[buffer(4)]],
                 uint i [[thread_position_in_grid]]) {
    frame[i] = cells[(i / width / cell) * cols + (i % width) / cell];
}
"""

fn random_food(rng: Random) -> tuple[int, int] {
    return (rng.int_range(0, COLS), rng.int_range(0, ROWS))
}

fn main() -> int {
    if not gpu.is_available() { throw RuntimeError("Snake needs a GPU") }
    let window = Window(WIDTH, HEIGHT, "Snake", target_fps: 120)
    defer { window.close() }
    let input = Input(window)
    let rng = Random()
    let kernel = gpu.Kernel.compile(SHADER, "draw")
    let cells = gpu.Buffer(COLS * ROWS)
    let frame = gpu.Buffer(WIDTH * HEIGHT)
    defer { kernel.free() }
    defer { cells.free() }
    defer { frame.free() }
    let pixels = window.pixels()
    let colors = [rgb(15, 15, 20), rgb(80, 200, 120), rgb(220, 80, 80)]

    var snake = [(COLS // 2, ROWS // 2)]
    var food = random_food(rng)
    var dx = 1
    var dy = 0
    var length = 4
    var elapsed = 0.0

    loop {
        if not window.poll() { break }
        input.update()
        if input.just_pressed(Key.Escape) { break }

        var next_dx = dx
        var next_dy = dy
        if input.just_pressed(Key.W) or input.just_pressed(Key.Up) {
            next_dx = 0
            next_dy = -1
        } elif input.just_pressed(Key.S) or input.just_pressed(Key.Down) {
            next_dx = 0
            next_dy = 1
        } elif input.just_pressed(Key.A) or input.just_pressed(Key.Left) {
            next_dx = -1
            next_dy = 0
        } elif input.just_pressed(Key.D) or input.just_pressed(Key.Right) {
            next_dx = 1
            next_dy = 0
        }
        if next_dx != -dx or next_dy != -dy {
            dx = next_dx
            dy = next_dy
        }

        elapsed += window.delta_time()
        if elapsed >= 0.05 {
            elapsed = 0.0
            let (head_x, head_y) = snake[snake.len() - 1]
            let head = ((head_x + dx + COLS) % COLS, (head_y + dy + ROWS) % ROWS)

            if head in snake {
                snake = [(COLS // 2, ROWS // 2)]
                food = random_food(rng)
                dx = 1
                dy = 0
                length = 4
            } else {
                snake.push(head)
                if head == food {
                    length += 1
                    food = random_food(rng)
                }
                if snake.len() > length { snake = snake[1:snake.len()] }
            }
        }

        var board: list[float] = [0.0 for _i in 0..COLS * ROWS]
        let (food_x, food_y) = food
        board[food_y * COLS + food_x] = 2.0
        for (x, y) in snake {
            board[y * COLS + x] = 1.0
        }
        cells.upload(board)
        kernel.dispatch([cells, frame], WIDTH * HEIGHT, [WIDTH, CELL, COLS])
        let rendered = frame.download()
        for i in 0..rendered.len() {
            pixels[i] = colors[int(rendered[i])]
        }
        window.present()
        print(window.fps(), "fps")
    }
    return 0
}
