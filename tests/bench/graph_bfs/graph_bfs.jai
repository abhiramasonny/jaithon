#: Breadth-first search over an adjacency list — pointer chasing, not arithmetic.
#:
#: Holdout. Every other benchmark in this suite walks memory in order: a range,
#: a row, a buffer. This one reads `dist[nb]` at an index the data picked, so no
#: prefetcher helps and the loop is bound by dependent loads. The list it walks
#: is a list *of lists*, and the inner loop is `for nb in g[node]` — a container
#: element that is itself a container, iterated.
from std.os import env_or

let LEVEL = env_or("BENCH_LEVEL", "hard")
let SCALE = if LEVEL == "easy" { 16 } elif LEVEL == "medium" { 4 } else { 1 }
# Building the graph is a fixed cost the level cannot cut without changing the
# shape being measured, so the level cuts the searches over it instead.
let NODES = 80_000
let DEGREE = 6
let REPS = max(6 // SCALE, 1)

fn build(n: int, deg: int) -> list[list[int]] {
    var g: list[list[int]] = []
    var s = 7
    for _i in 0..n {
        var adj: list[int] = []
        for _d in 0..deg {
            s = (s * 1103515245 + 12345) % 2147483648
            adj.push(s % n)
        }
        g.push(adj)
    }
    return g
}

fn bfs(g: list[list[int]], n: int, start: int) -> int {
    var dist: list[int] = []
    for _i in 0..n { dist.push(-1) }
    var queue: list[int] = []
    queue.push(start)
    dist[start] = 0
    var head = 0
    var total = 0
    while head < queue.len() {
        let node = queue[head]
        head += 1
        let d = dist[node]
        total += d
        for nb in g[node] {
            if dist[nb] < 0 {
                dist[nb] = d + 1
                queue.push(nb)
            }
        }
    }
    return total
}

fn main() -> int {
    let n = NODES
    let g = build(n, DEGREE)
    var total = 0
    for _r in 0..REPS { total = bfs(g, n, 0) }
    print(n)
    print(total)
    return 0
}
