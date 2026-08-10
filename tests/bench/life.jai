#: Conway's Game of Life over a list of lists of ints.
#:
#: `matrix_mul` indexes two dimensions along a straight line. This reads eight
#: neighbours around a moving centre, so three rows are live at once and every
#: cell is nine bounds-checked loads and a handful of adds. The board carries a
#: ring of dead cells that is never updated, which keeps the neighbour sum free
#: of edge tests — what is left is the indexing, not the branching.
fn zeros(n: int) -> list[int] {
    var row: list[int] = []
    for _i in 0..n { row.push(0) }
    return row
}

fn seed_board(h: int, w: int, seed: int) -> list[list[int]] {
    var board: list[list[int]] = []
    var s = seed
    board.push(zeros(w + 2))
    for _r in 0..h {
        var row: list[int] = []
        row.push(0)
        for _c in 0..w {
            s = (s * 1103515245 + 12345) % 2147483648
            let bits = s // 65536 % 100
            var alive = 0
            if bits < 35 { alive = 1 }
            row.push(alive)
        }
        row.push(0)
        board.push(row)
    }
    board.push(zeros(w + 2))
    return board
}

fn step(board: list[list[int]], h: int, w: int) -> list[list[int]] {
    var out: list[list[int]] = []
    out.push(zeros(w + 2))
    for r in 1..h + 1 {
        let up = board[r - 1]
        let mid = board[r]
        let down = board[r + 1]
        var row: list[int] = []
        row.push(0)
        for c in 1..w + 1 {
            var n = up[c - 1] + up[c] + up[c + 1]
            n += mid[c - 1] + mid[c + 1]
            n += down[c - 1] + down[c] + down[c + 1]
            var alive = 0
            if n == 3 {
                alive = 1
            } elif n == 2 {
                alive = mid[c]
            }
            row.push(alive)
        }
        row.push(0)
        out.push(row)
    }
    out.push(zeros(w + 2))
    return out
}

fn main() -> int {
    let h = 100
    let w = 100
    let generations = 50

    var board = seed_board(h, w, 7)
    for _g in 0..generations { board = step(board, h, w) }

    var live = 0
    for r in 1..h + 1 {
        let row = board[r]
        for c in 1..w + 1 { live += row[c] }
    }
    print(generations)
    print(live)
    return 0
}
