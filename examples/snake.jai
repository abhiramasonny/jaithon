#: Snake, in a real window at 60 fps.
#:
#: All drawing happens in Jaithon: the native layer hands out a pixel buffer
#: and std.gui.canvas does the rest.
#:
#:     jaithon run examples/snake.jai
from std.gui import Window
from std.gui.canvas import Canvas
from std.gui.color import rgb
from std.gui.input import Input, Key
from std.collections import Deque
from std.random import Random

const CELL = 20
const COLS = 32
const ROWS = 24
const WIDTH = COLS * CELL
const HEIGHT = ROWS * CELL
const TICKS_PER_MOVE = 6

# `let`, not `const`: a `const` initialiser has to fold at compile time
# (spec §3) and `rgb` is an ordinary function. These are still read-only.
let BACKGROUND = rgb(18, 18, 24)
let GRID_LINE = rgb(28, 28, 38)
let SNAKE_HEAD = rgb(120, 220, 140)
let SNAKE_BODY = rgb(70, 170, 100)
let FOOD_COLOR = rgb(230, 90, 90)
let TEXT_COLOR = rgb(235, 235, 240)

enum Direction { Up, Down, Left, Right }

fn delta(d: Direction) -> tuple[int, int] {
    return match d {
        Direction.Up => (0, -1),
        Direction.Down => (0, 1),
        Direction.Left => (-1, 0),
        Direction.Right => (1, 0),
    }
}

fn is_opposite(a: Direction, b: Direction) -> bool {
    let (ax, ay) = delta(a)
    let (bx, by) = delta(b)
    return ax == -bx and ay == -by
}

class Game {
    pub var body: Deque[tuple[int, int]]
    var occupied: set[tuple[int, int]]
    var direction: Direction
    var pending: Direction
    pub var food: tuple[int, int]
    pub var score: int
    pub var over: bool
    var rng: Random

    fn init(self, rng: Random) {
        self.rng = rng
        self.reset()
    }

    pub fn reset(self) -> void {
        let start = (COLS // 2, ROWS // 2)
        self.body = Deque[tuple[int, int]]()
        self.body.push_back(start)
        self.occupied = {start}
        self.direction = Direction.Right
        self.pending = Direction.Right
        self.score = 0
        self.over = false
        self.food = self.spawn_food()
    }

    fn spawn_food(self) -> tuple[int, int] {
        # The board is small, so rejection sampling is fine and stays uniform.
        loop {
            let cell = (self.rng.int_range(0, COLS), self.rng.int_range(0, ROWS))
            if not self.occupied.has(cell) { return cell }
        }
    }

    pub fn turn(self, d: Direction) -> void {
        # Reject a reversal so a single frame cannot kill you.
        if not is_opposite(self.direction, d) { self.pending = d }
    }

    pub fn step(self) -> void {
        if self.over { return }

        self.direction = self.pending
        let (dx, dy) = delta(self.direction)
        let (hx, hy) = self.body.back() ?? (0, 0)
        let next = (hx + dx, hy + dy)
        let (nx, ny) = next

        if nx < 0 or nx >= COLS or ny < 0 or ny >= ROWS {
            self.over = true
            return
        }

        let eating = next == self.food
        # Stepping onto the current tail is legal when the tail moves away.
        let tail = self.body.front() ?? (0, 0)
        if self.occupied.has(next) and not (next == tail and not eating) {
            self.over = true
            return
        }

        self.body.push_back(next)
        self.occupied.add(next)

        if eating {
            self.score += 1
            self.food = self.spawn_food()
        } else {
            let removed = self.body.pop_front()
            self.occupied.remove(removed)
        }
    }
}

fn draw(canvas: Canvas, game: Game) -> void {
    canvas.clear(BACKGROUND)

    for col in 1..COLS { canvas.line(col * CELL, 0, col * CELL, HEIGHT, GRID_LINE) }
    for row in 1..ROWS { canvas.line(0, row * CELL, WIDTH, row * CELL, GRID_LINE) }

    let (fx, fy) = game.food
    canvas.fill_circle(fx * CELL + CELL // 2, fy * CELL + CELL // 2, CELL // 2 - 3, FOOD_COLOR)

    let cells = game.body.to_list()
    for (index, cell) in cells.enumerate() {
        let (x, y) = cell
        let color = if index == cells.len() - 1 { SNAKE_HEAD } else { SNAKE_BODY }
        canvas.fill_rounded_rect(x * CELL + 2, y * CELL + 2, CELL - 4, CELL - 4, 4, color)
    }

    canvas.text(8, 8, f"score {game.score}", TEXT_COLOR)
    if game.over {
        let message = "game over — press R"
        let width = canvas.text_width(message)
        canvas.text((WIDTH - width) // 2, HEIGHT // 2 - 4, message, TEXT_COLOR)
    }
}

fn main() -> int {
    let window = Window(WIDTH, HEIGHT, "Snake", target_fps: 60)
    defer { window.close() }

    let canvas = Canvas(window)
    let input = Input(window)
    let game = Game(Random.with_seed(7))

    var tick = 0
    loop {
        if not window.poll() { break }
        input.update()

        if input.just_pressed(Key.Escape) or input.just_pressed(Key.Q) { break }
        if input.just_pressed(Key.R) { game.reset() }
        if input.just_pressed(Key.Up) or input.just_pressed(Key.W) { game.turn(Direction.Up) }
        if input.just_pressed(Key.Down) or input.just_pressed(Key.S) { game.turn(Direction.Down) }
        if input.just_pressed(Key.Left) or input.just_pressed(Key.A) { game.turn(Direction.Left) }
        if input.just_pressed(Key.Right) or input.just_pressed(Key.D) { game.turn(Direction.Right) }

        tick += 1
        if tick % TICKS_PER_MOVE == 0 { game.step() }

        draw(canvas, game)
        window.present()
    }

    print(f"final score: {game.score}")
    return 0
}
