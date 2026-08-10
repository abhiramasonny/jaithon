#: Train a compact NEAT population to play Snake on a 6x6 grid.
#:
#: The trainer stays responsive while it evaluates genomes. Click
#: "save & watch" to write the champion topology and weights to
#: `neat_snake.weights`, then open a live-play window.
#:
#:     jaithon run examples/neat_snake.jai
from std.gui import Window
from std.gui.canvas import Canvas, rgb
from std.gui.input import Input, Key
from std.gui.widgets import UI, Theme
from std.random import Random
from std.math import tanh
from std.io import write_file
from std.str import StringBuilder

const COLS = 6
const ROWS = 6
const INPUT_COUNT = 11
const OUTPUT_COUNT = 3
const OUTPUT_START = INPUT_COUNT
const POPULATION_SIZE = 48
const EPISODES = 2
const MAX_STEPS = 180
const COMPATIBILITY_THRESHOLD = 2.2
const WEIGHTS_PATH = "neat_snake.weights"

let BACKGROUND = rgb(14, 18, 24)
let SURFACE = rgb(23, 28, 35)
let SURFACE_HOVER = rgb(31, 38, 46)
let SURFACE_ACTIVE = rgb(38, 47, 56)
let BORDER = rgb(49, 59, 69)
let GRID = rgb(34, 40, 48)
let TEXT = rgb(226, 232, 236)
let MUTED = rgb(132, 143, 151)
let GREEN = rgb(92, 211, 137)
let GREEN_DARK = rgb(55, 165, 101)
let RED = rgb(239, 92, 92)

enum Direction { Up, Right, Down, Left }

fn direction_delta(direction: Direction) -> tuple[int, int] {
    return match direction {
        Direction.Up => (0, -1),
        Direction.Right => (1, 0),
        Direction.Down => (0, 1),
        Direction.Left => (-1, 0),
    }
}

fn turn_relative(direction: Direction, choice: int) -> Direction {
    let value = match direction {
        Direction.Up => 0,
        Direction.Right => 1,
        Direction.Down => 2,
        Direction.Left => 3,
    }
    let rotated = if choice == 0 {
        (value + 3) % 4
    } elif choice == 2 {
        (value + 1) % 4
    } else {
        value
    }
    return match rotated {
        0 => Direction.Up,
        1 => Direction.Right,
        2 => Direction.Down,
        _ => Direction.Left,
    }
}

class NodeGene {
    pub let id: int
    pub let kind: int  # 0 input, 1 hidden, 2 output
    pub let layer: float

    pub fn init(self, id: int, kind: int, layer: float) {
        self.id = id
        self.kind = kind
        self.layer = layer
    }

    pub fn copy(self) -> NodeGene { return NodeGene(self.id, self.kind, self.layer) }
}

class ConnectionGene {
    pub let source: int
    pub let target: int
    pub var weight: float
    pub var enabled: bool
    pub let innovation: int

    pub fn init(self, source: int, target: int, weight: float, enabled: bool, innovation: int) {
        self.source = source
        self.target = target
        self.weight = weight
        self.enabled = enabled
        self.innovation = innovation
    }

    pub fn copy(self) -> ConnectionGene {
        return ConnectionGene(self.source, self.target, self.weight, self.enabled, self.innovation)
    }
}

class InnovationBook {
    var next_node: int
    var next_connection: int
    var connections: dict[tuple[int, int], int]
    var splits: dict[int, tuple[int, int, int]]

    pub fn init(self) {
        self.next_node = INPUT_COUNT + OUTPUT_COUNT
        self.next_connection = 0
        self.connections = {}
        self.splits = {}
    }

    pub fn connection(self, source: int, target: int) -> int {
        let key = (source, target)
        if self.connections.has(key) { return self.connections[key] }
        let innovation = self.next_connection
        self.next_connection += 1
        self.connections[key] = innovation
        return innovation
    }

    pub fn split(self, old_innovation: int, source: int, target: int) -> tuple[int, int, int] {
        if self.splits.has(old_innovation) { return self.splits[old_innovation] }
        let node = self.next_node
        self.next_node += 1
        let first = self.connection(source, node)
        let second = self.connection(node, target)
        let result = (node, first, second)
        self.splits[old_innovation] = result
        return result
    }
}

class Genome {
    pub var nodes: list[NodeGene]
    pub var connections: list[ConnectionGene]
    pub var fitness: float

    pub fn init(self) {
        self.nodes = []
        self.connections = []
        self.fitness = 0.0
    }

    pub fn copy(self) -> Genome {
        let result = Genome()
        result.nodes = [node.copy() for node in self.nodes]
        result.connections = [connection.copy() for connection in self.connections]
        result.fitness = self.fitness
        return result
    }

    fn node(self, id: int) -> NodeGene? {
        for node in self.nodes {
            if node.id == id { return node }
        }
        return null
    }

    pub fn connection_with_innovation(self, innovation: int) -> ConnectionGene? {
        for connection in self.connections {
            if connection.innovation == innovation { return connection }
        }
        return null
    }

    fn has_connection(self, source: int, target: int) -> bool {
        for connection in self.connections {
            if connection.source == source and connection.target == target { return true }
        }
        return false
    }

    pub fn activate(self, inputs: list[float]) -> list[float] {
        let values: dict[int, float] = {}
        for index in 0..INPUT_COUNT { values[index] = inputs[index] }

        let complete: set[int] = {index for index in 0..INPUT_COUNT}
        for _pass in 0..self.nodes.len() {
            var progressed = false
            for node in self.nodes {
                if node.kind == 0 or complete.has(node.id) { continue }
                var ready = true
                var total = 0.0
                var incoming = 0
                for connection in self.connections {
                    if not connection.enabled or connection.target != node.id { continue }
                    incoming += 1
                    if not complete.has(connection.source) {
                        ready = false
                    } else {
                        total += values[connection.source] * connection.weight
                    }
                }
                if ready {
                    values[node.id] = if incoming == 0 { 0.0 } else { tanh(total) }
                    complete.add(node.id)
                    progressed = true
                }
            }
            if not progressed { break }
        }

        let output: list[float] = []
        for index in 0..OUTPUT_COUNT {
            let id = OUTPUT_START + index
            output.push(if values.has(id) { values[id] } else { 0.0 })
        }
        return output
    }

    pub fn mutate(self, rng: Random, book: InnovationBook) -> void {
        for connection in self.connections {
            if rng.bool(0.8) { connection.weight += rng.gauss(0.0, 0.35) }
            if rng.bool(0.08) { connection.weight = rng.float_range(-2.0, 2.0) }
        }
        if rng.bool(0.16) { self.add_connection(rng, book) }
        if rng.bool(0.05) { self.add_node(rng, book) }
        if rng.bool(0.02) and self.connections.len() > 0 {
            let connection = rng.choice(self.connections)
            connection.enabled = not connection.enabled
        }
    }

    fn add_connection(self, rng: Random, book: InnovationBook) -> void {
        for _attempt in 0..24 {
            let source = rng.choice(self.nodes)
            let target = rng.choice(self.nodes)
            if source.layer >= target.layer or target.kind == 0 { continue }
            if self.has_connection(source.id, target.id) { continue }
            self.connections.push(
                ConnectionGene(
                    source.id,
                    target.id,
                    rng.float_range(-2.0, 2.0),
                    true,
                    book.connection(source.id, target.id)
                )
            )
            return
        }
    }

    fn add_node(self, rng: Random, book: InnovationBook) -> void {
        let candidates = [connection for connection in self.connections if connection.enabled]
        if candidates.len() == 0 { return }
        let old = rng.choice(candidates)
        let source = self.node(old.source)
        let target = self.node(old.target)
        if source == null or target == null { return }
        old.enabled = false
        let (node_id, first_innovation, second_innovation) = book.split(
            old.innovation,
            old.source,
            old.target
        )
        if self.node(node_id) == null {
            self.nodes.push(NodeGene(node_id, 1, (source.layer + target.layer) / 2.0))
        }
        if not self.has_connection(old.source, node_id) {
            self.connections.push(ConnectionGene(old.source, node_id, 1.0, true, first_innovation))
        }
        if not self.has_connection(node_id, old.target) {
            self.connections.push(
                ConnectionGene(node_id, old.target, old.weight, true, second_innovation)
            )
        }
    }
}

fn crossover(fitter: Genome, other: Genome, rng: Random) -> Genome {
    let child = Genome()
    child.nodes = [node.copy() for node in fitter.nodes]
    for gene in fitter.connections {
        let matching_gene = other.connection_with_innovation(gene.innovation)
        var chosen = gene
        if matching_gene != null and rng.bool() { chosen = matching_gene }
        let copied = chosen.copy()
        if matching_gene != null and (not gene.enabled or not matching_gene.enabled) {
            copied.enabled = not rng.bool(0.75)
        }
        child.connections.push(copied)
    }
    return child
}

fn compatibility(a: Genome, b: Genome) -> float {
    var matching = 0
    var unmatched = 0
    var weight_difference = 0.0
    for connection in a.connections {
        let other = b.connection_with_innovation(connection.innovation)
        if other == null {
            unmatched += 1
        } else {
            matching += 1
            let difference = connection.weight - other.weight
            weight_difference += if difference < 0.0 { -difference } else { difference }
        }
    }
    for connection in b.connections {
        if a.connection_with_innovation(connection.innovation) == null { unmatched += 1 }
    }
    let normalizer = float(max(max(a.connections.len(), b.connections.len()), 1))
    let average_weight = if matching == 0 { 0.0 } else { weight_difference / float(matching) }
    return float(unmatched) / normalizer + 0.4 * average_weight
}

class Species {
    pub let representative: Genome
    pub var members: list[Genome]

    pub fn init(self, representative: Genome) {
        self.representative = representative
        self.members = []
    }
}

class SnakeSim {
    pub var body: list[tuple[int, int]]
    var occupied: set[tuple[int, int]]
    pub var food: tuple[int, int]
    pub var direction: Direction
    pub var score: int
    pub var steps: int
    pub var starvation: int
    pub var dead: bool
    var rng: Random

    pub fn init(self, seed: int) {
        self.rng = Random.with_seed(seed)
        self.reset(seed)
    }

    pub fn reset(self, seed: int) -> void {
        self.rng.reseed(seed)
        self.body = [(1, 3), (2, 3), (3, 3)]
        self.occupied = {(1, 3), (2, 3), (3, 3)}
        self.direction = Direction.Right
        self.score = 0
        self.steps = 0
        self.starvation = 0
        self.dead = false
        self.food = self.spawn_food()
    }

    fn spawn_food(self) -> tuple[int, int] {
        loop {
            let cell = (self.rng.int_range(0, COLS), self.rng.int_range(0, ROWS))
            if not self.occupied.has(cell) { return cell }
        }
    }

    fn blocked(self, direction: Direction) -> bool {
        let (dx, dy) = direction_delta(direction)
        let (head_x, head_y) = self.body[self.body.len() - 1]
        let next_x = head_x + dx
        let next_y = head_y + dy
        if next_x < 0 or next_x >= COLS or next_y < 0 or next_y >= ROWS { return true }
        let next = (next_x, next_y)
        let tail = self.body[0]
        return self.occupied.has(next) and next != tail
    }

    fn inputs(self) -> list[float] {
        let (head_x, head_y) = self.body[self.body.len() - 1]
        let (food_x, food_y) = self.food
        let left = turn_relative(self.direction, 0)
        let right = turn_relative(self.direction, 2)
        let values: list[float] = [if self.blocked(self.direction) {
            1.0
        } else {
            0.0
        }, if self.blocked(left) {
            1.0
        } else {
            0.0
        }, if self.blocked(right) {
            1.0
        } else {
            0.0
        }, float(food_x - head_x) / float(COLS - 1), float(food_y - head_y) / float(ROWS - 1), if self.direction == Direction.Up {
            1.0
        } else {
            0.0
        }, if self.direction == Direction.Right {
            1.0
        } else {
            0.0
        }, if self.direction == Direction.Down {
            1.0
        } else {
            0.0
        }, if self.direction == Direction.Left {
            1.0
        } else {
            0.0
        }, float(self.body.len()) / float(COLS * ROWS), 1.0]
        return values
    }

    pub fn step(self, brain: Genome) -> float {
        if self.dead { return 0.0 }
        let output = brain.activate(self.inputs())
        var choice = 0
        if output[1] > output[choice] { choice = 1 }
        if output[2] > output[choice] { choice = 2 }
        self.direction = turn_relative(self.direction, choice)

        let (old_x, old_y) = self.body[self.body.len() - 1]
        let (food_x, food_y) = self.food
        let old_distance = abs(food_x - old_x) + abs(food_y - old_y)
        let (dx, dy) = direction_delta(self.direction)
        let next_x = old_x + dx
        let next_y = old_y + dy
        if next_x < 0 or next_x >= COLS or next_y < 0 or next_y >= ROWS {
            self.dead = true
            return -2.0
        }
        let next = (next_x, next_y)
        let tail = self.body[0]
        let eating = next == self.food
        if self.occupied.has(next) and not (next == tail and not eating) {
            self.dead = true
            return -2.0
        }

        if not eating {
            self.body = self.body[1:self.body.len()]
            self.occupied.remove(tail)
        }
        self.body.push(next)
        self.occupied.add(next)
        var reward = 0.05
        if eating {
            self.score += 1
            self.starvation = 0
            reward += 25.0
            if self.body.len() == COLS * ROWS {
                self.dead = true
                return reward + 100.0
            }
            self.food = self.spawn_food()
        } else {
            self.starvation += 1
            let (new_x, new_y) = next
            let new_distance = abs(food_x - new_x) + abs(food_y - new_y)
            reward += if new_distance < old_distance { 0.2 } else { -0.04 }
        }
        self.steps += 1
        if self.starvation > 48 + self.score * 8 or self.steps >= MAX_STEPS { self.dead = true }
        return reward
    }
}

fn evaluate(genome: Genome) -> float {
    var fitness = 0.0
    for episode in 0..EPISODES {
        let sim = SnakeSim(9109 + episode * 7919)
        while not sim.dead { fitness += sim.step(genome) }
        fitness += float(sim.score * sim.score) * 30.0
    }
    return max(fitness / float(EPISODES), 0.01)
}

class Trainer {
    pub var population: list[Genome]
    pub var generation: int
    pub var evaluating: int
    pub var species_count: int
    pub var best: Genome
    pub var best_fitness: float
    var rng: Random
    var book: InnovationBook

    pub fn init(self) {
        self.rng = Random.with_seed(20260810)
        self.book = InnovationBook()
        self.population = []
        self.generation = 1
        self.evaluating = 0
        self.species_count = 1

        let base = Genome()
        for index in 0..INPUT_COUNT { base.nodes.push(NodeGene(index, 0, 0.0)) }
        for index in 0..OUTPUT_COUNT { base.nodes.push(NodeGene(OUTPUT_START + index, 2, 1.0)) }
        for source in 0..INPUT_COUNT {
            for target in 0..OUTPUT_COUNT {
                base.connections.push(
                    ConnectionGene(
                        source,
                        OUTPUT_START + target,
                        self.rng.float_range(-1.0, 1.0),
                        true,
                        self.book.connection(source, OUTPUT_START + target)
                    )
                )
            }
        }
        for _index in 0..POPULATION_SIZE {
            let genome = base.copy()
            for connection in genome.connections {
                connection.weight = self.rng.float_range(-1.5, 1.5)
            }
            self.population.push(genome)
        }
        self.best = self.population[0].copy()
        self.best_fitness = 0.0
    }

    pub fn tick(self, count: int = 1) -> void {
        for _work in 0..count {
            let genome = self.population[self.evaluating]
            genome.fitness = evaluate(genome)
            if genome.fitness > self.best_fitness {
                self.best = genome.copy()
                self.best_fitness = genome.fitness
            }
            self.evaluating += 1
            if self.evaluating == self.population.len() { self.evolve() }
        }
    }

    fn species(self) -> list[Species] {
        let groups: list[Species] = []
        for genome in self.population {
            var placed = false
            for group in groups {
                if compatibility(genome, group.representative) < COMPATIBILITY_THRESHOLD {
                    group.members.push(genome)
                    placed = true
                    break
                }
            }
            if not placed {
                let group = Species(genome)
                group.members.push(genome)
                groups.push(group)
            }
        }
        return groups
    }

    fn tournament(self, members: list[Genome]) -> Genome {
        var winner = self.rng.choice(members)
        for _round in 0..2 {
            let candidate = self.rng.choice(members)
            if candidate.fitness > winner.fitness { winner = candidate }
        }
        return winner
    }

    fn evolve(self) -> void {
        let groups = self.species()
        self.species_count = groups.len()
        let next: list[Genome] = []

        # Preserve the best member of every species.
        for group in groups {
            var champion = group.members[0]
            for member in group.members {
                if member.fitness > champion.fitness { champion = member }
            }
            if next.len() < POPULATION_SIZE { next.push(champion.copy()) }
        }

        while next.len() < POPULATION_SIZE {
            let group = self.rng.choice(groups)
            let first = self.tournament(group.members)
            let second = self.tournament(group.members)
            let fitter = if first.fitness >= second.fitness { first } else { second }
            let other = if first.fitness >= second.fitness { second } else { first }
            let child = crossover(fitter, other, self.rng)
            child.mutate(self.rng, self.book)
            next.push(child)
        }
        self.population = next
        self.generation += 1
        self.evaluating = 0
    }
}

fn save_genome(genome: Genome) -> void {
    let text = StringBuilder()
    text.push_line("NEAT-SNAKE 1")
    text.push_line(f"inputs {INPUT_COUNT}")
    text.push_line(f"outputs {OUTPUT_COUNT}")
    text.push_line(f"fitness {genome.fitness:.6f}")
    for node in genome.nodes {
        text.push_line(f"node {node.id} {node.kind} {node.layer:.12f}")
    }
    for connection in genome.connections {
        let enabled = if connection.enabled { 1 } else { 0 }
        text.push_line(
            f"connection {connection.source} {connection.target} {connection.weight:.12f} {enabled} {connection.innovation}"
        )
    }
    write_file(WEIGHTS_PATH, text.build())
}

fn draw_board(canvas: Canvas, sim: SnakeSim, x: int, y: int, cell: int) -> void {
    let size = cell * COLS
    canvas.fill_rect(x, y, size, size, BACKGROUND)
    for col in 1..COLS { canvas.line(x + col * cell, y, x + col * cell, y + size - 1, GRID) }
    for row in 1..ROWS { canvas.line(x, y + row * cell, x + size - 1, y + row * cell, GRID) }

    let (food_x, food_y) = sim.food
    canvas.fill_circle(
        x + food_x * cell + cell // 2,
        y + food_y * cell + cell // 2,
        max(cell // 5, 3),
        RED
    )
    for (index, segment) in sim.body.enumerate() {
        let (segment_x, segment_y) = segment
        let inset = max(cell // 8, 2)
        let color = if index == sim.body.len() - 1 { GREEN } else { GREEN_DARK }
        canvas.fill_rounded_rect(
            x + segment_x * cell + inset,
            y + segment_y * cell + inset,
            cell - inset * 2,
            cell - inset * 2,
            max(cell // 7, 3),
            color
        )
    }
}

fn watch(champion: Genome) -> void {
    const CELL = 80
    const BOARD = CELL * COLS
    const HEADER = 58
    let window = Window(BOARD, BOARD + HEADER, "NEAT Snake - champion", target_fps: 60)
    defer { window.close() }
    let canvas = Canvas(window)
    let input = Input(window)
    let sim = SnakeSim(44041)
    var tick = 0
    var reset_wait = 0

    loop {
        if not window.poll() { break }
        input.update()
        if input.just_pressed(Key.Escape) or input.just_pressed(Key.Q) { break }
        if input.just_pressed(Key.R) { sim.reset(44041 + tick) }

        tick += 1
        if sim.dead {
            reset_wait += 1
            if reset_wait > 75 {
                sim.reset(44041 + tick)
                reset_wait = 0
            }
        } elif tick % 8 == 0 {
            sim.step(champion)
        }

        canvas.clear(BACKGROUND)
        canvas.text(18, 18, "CHAMPION", TEXT, 2)
        canvas.text(365, 21, f"score {sim.score}", MUTED)
        draw_board(canvas, sim, 0, HEADER, CELL)
        if sim.dead {
            let message = "resetting..."
            canvas.fill_rounded_rect(169, HEADER + 218, 142, 44, 10, SURFACE)
            canvas.text((BOARD - canvas.text_width(message)) // 2, HEADER + 237, message, TEXT)
        }
        window.present()
    }
}

fn main() -> int {
    const WIDTH = 720
    const HEIGHT = 500
    let window = Window(WIDTH, HEIGHT, "NEAT Snake - training", target_fps: 60)
    let input = Input(window)
    let theme = Theme(
        BACKGROUND,
        SURFACE,
        SURFACE_HOVER,
        SURFACE_ACTIVE,
        BORDER,
        TEXT,
        MUTED,
        GREEN
    )
    let ui = UI(Canvas(window), input, theme)
    let trainer = Trainer()
    let preview = SnakeSim(7717)
    var preview_tick = 0
    var launch = false

    loop {
        if not window.poll() { break }
        input.update()
        if input.just_pressed(Key.Escape) or input.just_pressed(Key.Q) { break }

        trainer.tick(1)
        preview_tick += 1
        if preview.dead { preview.reset(7717 + trainer.generation) }
        if preview_tick % 7 == 0 { preview.step(trainer.best) }

        ui.begin()
        ui.canvas.clear(BACKGROUND)
        ui.canvas.text(24, 22, "NEAT SNAKE", TEXT, 3)
        ui.canvas.text(24, 52, "evolving a 6x6 snake controller", MUTED)
        draw_board(ui.canvas, preview, 24, 96, 54)

        ui.canvas.text(390, 112, "GENERATION", MUTED)
        ui.canvas.text(390, 134, f"{trainer.generation}", TEXT, 3)
        ui.canvas.text(390, 184, "BEST FITNESS", MUTED)
        ui.canvas.text(390, 206, f"{trainer.best_fitness:.2f}", GREEN, 2)
        ui.canvas.text(390, 250, "SPECIES", MUTED)
        ui.canvas.text(390, 272, f"{trainer.species_count}", TEXT, 2)
        ui.canvas.text(510, 250, "GENOME", MUTED)
        ui.canvas.text(510, 272, f"{trainer.evaluating + 1}/{POPULATION_SIZE}", TEXT, 2)
        ui.canvas.text(
            390,
            318,
            f"nodes {trainer.best.nodes.len()}   links {trainer.best.connections.len()}",
            MUTED
        )

        if ui.button("watch", 390, 370, 238, 42, "save & watch") {
            save_genome(trainer.best)
            launch = true
        }
        ui.canvas.text(390, 430, "writes neat_snake.weights", MUTED)
        ui.end()
        window.present()
        if launch { break }
    }

    window.close()
    if launch { watch(trainer.best) }
    return 0
}
