#: Classic dynamic-programming algorithms, solved bottom-up.
#:
#: Every routine whose answer has a witness — a subsequence, a set of items, a
#: list of cuts — returns that witness beside the optimum, not the optimal
#: value alone.
from std.func import memoize

#: One step of an edit script, as produced by `edit_script`.
#:
#: `index` is a position in the source string: for `Keep`/`Replace`/`Delete`
#: it's the character consumed; for `Insert` it's the position the new
#: character goes before.
pub enum EditOp {
    #: The character at `index` already matches the target and is copied
    #: through; the only operation that does not count towards the distance.
    Keep(index: int, character: str),
    #: The character at `index` is `character` and becomes `replacement`.
    Replace(index: int, character: str, replacement: str),
    #: The character at `index` is dropped, so nothing of it reaches the result.
    Delete(index: int, character: str),
    #: `character` is added before source position `index`, which equals the
    #: length of the source when the insertion goes past its last character.
    Insert(index: int, character: str),
}

#: A contiguous slice `xs[start:stop]` of a list, together with its sum.
pub class Subarray {
    #: Sum of the elements in the slice.
    pub let sum: int
    #: Index of the first element of the slice.
    pub let start: int
    #: Index one past the last element of the slice.
    pub let stop: int

    #: Record a slice; `stop` is exclusive.
    pub fn init(self, sum: int, start: int, stop: int) {
        self.sum = sum
        self.start = start
        self.stop = stop
    }

    fn __str__(self) -> str {
        return f"Subarray(sum: {self.sum}, {self.start}..{self.stop})"
    }
}

#: A rectangular block `grid[top:bottom][left:right]`, together with its sum.
pub class Submatrix {
    #: Sum of every cell in the block.
    pub let sum: int
    #: Index of the first row of the block.
    pub let top: int
    #: Index of the first column of the block.
    pub let left: int
    #: Index one past the last row of the block.
    pub let bottom: int
    #: Index one past the last column of the block.
    pub let right: int

    #: Record a block; `bottom` and `right` are exclusive.
    pub fn init(self, sum: int, top: int, left: int, bottom: int, right: int) {
        self.sum = sum
        self.top = top
        self.left = left
        self.bottom = bottom
        self.right = right
    }

    fn __str__(self) -> str {
        let rows = f"{self.top}..{self.bottom}"
        return f"Submatrix(sum: {self.sum}, rows {rows}, cols {self.left}..{self.right})"
    }
}

#: Return the `n`-th Fibonacci number, counting from `fibonacci(0) == 0`.
#:
#: `F(92)` is the largest term that fits in a 64-bit integer; past it the
#: addition raises `OverflowError`.
#:
#: ```
#: fibonacci(10)    # 55
#: fibonacci(92)    # 7540113804746346429
#: ```
pub fn fibonacci(n: int) -> int {
    if n < 0 { throw ValueError(f"n must be non-negative, got {n}") }
    if n == 0 { return 0 }
    var previous = 0
    var current = 1
    # Stops at F(n), not F(n + 1) — F(93) would overflow.
    for _ in 1..n {
        let next = previous + current
        previous = current
        current = next
    }
    return current
}

#: Return the `n`-th Fibonacci number by raising `[[1, 1], [1, 0]]` to a power.
#:
#: Same domain as `fibonacci` (`n <= 92`, since `M**(n-1)` puts F(n) in the
#: top-left corner). The point isn't speed here — it's the squaring skeleton,
#: which carries any linear recurrence to exponents no linear scan could reach.
#:
#: ```
#: fibonacci_matrix(10)    # 55
#: fibonacci_matrix(92)    # 7540113804746346429
#: ```
pub fn fibonacci_matrix(n: int) -> int {
    if n < 0 { throw ValueError(f"n must be non-negative, got {n}") }
    if n == 0 { return 0 }
    return _mat2_fib_power(n - 1)[0]
}

#: Return one longest common subsequence of `a` and `b`.
#:
#: The full O(n*m) table is kept, not just O(min(n,m)), so the subsequence can
#: be reconstructed from it. Ties prefer earlier elements of `a`.
#:
#: ```
#: longest_common_subsequence([1, 2, 3, 4, 5], [2, 3, 5, 7])    # [2, 3, 5]
#: longest_common_subsequence([1, 2], [3, 4])                   # []
#: ```
pub fn longest_common_subsequence[T](a: list[T], b: list[T]) -> list[T] {
    let n = a.len()
    let m = b.len()
    let table = [[0 for _ in 0..=m] for _ in 0..=n]
    for i in 1..=n {
        for j in 1..=m {
            if a[i - 1] == b[j - 1] {
                table[i][j] = table[i - 1][j - 1] + 1
            } else {
                table[i][j] = max(table[i - 1][j], table[i][j - 1])
            }
        }
    }
    let picked: list[T] = []
    var i = n
    var j = m
    while i > 0 and j > 0 {
        if a[i - 1] == b[j - 1] {
            picked.push(a[i - 1])
            i -= 1
            j -= 1
        } elif table[i - 1][j] >= table[i][j - 1] {
            i -= 1
        } else {
            j -= 1
        }
    }
    return reversed(picked)
}

#: Return one longest increasing subsequence of `xs`.
#:
#: Patience-sorting: each element goes on the leftmost pile whose top isn't
#: smaller than it, and a parent link back to the previous pile's top lets the
#: run be reconstructed without extra table space.
#:
#: ```
#: longest_increasing_subsequence([10, 9, 2, 5, 3, 7, 101, 18])   # [2, 3, 7, 18]
#: longest_increasing_subsequence([3, 3, 3], strict: false)       # [3, 3, 3]
#: ```
pub fn longest_increasing_subsequence(xs: list[int], strict: bool = true) -> list[int] {
    let n = xs.len()
    if n == 0 { return [] }
    # pile_top[k]/pile_values[k]: index/value of the smallest tail among length-(k+1) subsequences.
    let pile_top: list[int] = []
    let pile_values: list[int] = []
    let parent = [-1 for _ in 0..n]
    for i in 0..n {
        let position = _lower_bound(pile_values, xs[i], strict)
        if position > 0 { parent[i] = pile_top[position - 1] }
        if position == pile_values.len() {
            pile_top.push(i)
            pile_values.push(xs[i])
        } else {
            pile_top[position] = i
            pile_values[position] = xs[i]
        }
    }
    let picked: list[int] = []
    var cursor = pile_top[pile_top.len() - 1]
    while cursor >= 0 {
        picked.push(xs[cursor])
        cursor = parent[cursor]
    }
    return reversed(picked)
}

#: Return the Levenshtein distance between `source` and `target`.
#:
#: O(min(n, m)) space — only two rows of the table are ever live. Use
#: `edit_script` when the operations themselves are wanted.
#:
#: ```
#: edit_distance("kitten", "sitting")    # 3
#: edit_distance("", "abc")              # 3
#: ```
pub fn edit_distance(source: str, target: str) -> int {
    if source.len() < target.len() { return edit_distance(target, source) }
    let n = source.len()
    let m = target.len()
    var previous = [j for j in 0..=m]
    for i in 1..=n {
        let current = [i for _ in 0..=m]
        for j in 1..=m {
            let substitution = previous[j - 1] + (source[i - 1] == target[j - 1] ? 0 : 1)
            current[j] = min(substitution, min(previous[j] + 1, current[j - 1] + 1))
        }
        previous = current
    }
    return previous[m]
}

#: Return a shortest sequence of edits turning `source` into `target`.
#:
#: The script is in source order; its cost (non-`Keep` operations) equals
#: `edit_distance(source, target)`. Ties resolve substitution, then deletion,
#: then insertion.
#:
#: ```
#: let script = edit_script("kitten", "sitting")
#: script.len()                            # 7
#: apply_edit_script("kitten", script)     # "sitting"
#: ```
pub fn edit_script(source: str, target: str) -> list[EditOp] {
    let n = source.len()
    let m = target.len()
    let cost = [[0 for _ in 0..=m] for _ in 0..=n]
    for i in 1..=n { cost[i][0] = i }
    for j in 1..=m { cost[0][j] = j }
    for i in 1..=n {
        for j in 1..=m {
            let diagonal = cost[i - 1][j - 1] + (source[i - 1] == target[j - 1] ? 0 : 1)
            cost[i][j] = min(diagonal, min(cost[i - 1][j] + 1, cost[i][j - 1] + 1))
        }
    }
    let script: list[EditOp] = []
    var i = n
    var j = m
    while i > 0 or j > 0 {
        let same = i > 0 and j > 0 and source[i - 1] == target[j - 1]
        if i > 0 and j > 0 and cost[i][j] == cost[i - 1][j - 1] + (same ? 0 : 1) {
            if same {
                script.push(EditOp.Keep(i - 1, source[i - 1]))
            } else {
                script.push(EditOp.Replace(i - 1, source[i - 1], target[j - 1]))
            }
            i -= 1
            j -= 1
        } elif i > 0 and cost[i][j] == cost[i - 1][j] + 1 {
            script.push(EditOp.Delete(i - 1, source[i - 1]))
            i -= 1
        } else {
            script.push(EditOp.Insert(i, target[j - 1]))
            j -= 1
        }
    }
    return reversed(script)
}

#: Apply `script` to `source` and return the resulting string.
#:
#: Raises `ValueError` when an operation names a character that doesn't match
#: `source` at its index — a checked inverse of `edit_script`, not a blind replay.
#:
#: ```
#: apply_edit_script("cat", edit_script("cat", "cart"))    # "cart"
#: ```
pub fn apply_edit_script(source: str, script: list[EditOp]) -> str {
    let pieces: list[str] = []
    for operation in script {
        match operation {
            EditOp.Keep(index, character) => {
                _expect_character(source, index, character)
                pieces.push(character)
            }
            EditOp.Replace(index, character, replacement) => {
                _expect_character(source, index, character)
                pieces.push(replacement)
            }
            EditOp.Delete(index, character) => { _expect_character(source, index, character) }
            EditOp.Insert(_, character) => { pieces.push(character) }
        }
    }
    return "".join(pieces)
}

#: Solve the 0/1 knapsack: return the best value and the indices achieving it.
#:
#: The full O(n*capacity) table is kept so the chosen items can be read back.
#:
#: ```
#: knapsack_01([1, 3, 4, 5], [1, 4, 5, 7], 7)    # (9, [1, 2])
#: ```
pub fn knapsack_01(weights: list[int], values: list[int], capacity: int) -> tuple[int, list[int]] {
    _check_knapsack(weights, values, capacity)
    let n = weights.len()
    let best = [[0 for _ in 0..=capacity] for _ in 0..=n]
    for i in 1..=n {
        for c in 0..=capacity {
            best[i][c] = best[i - 1][c]
            if weights[i - 1] <= c {
                let taken = best[i - 1][c - weights[i - 1]] + values[i - 1]
                if taken > best[i][c] { best[i][c] = taken }
            }
        }
    }
    let items: list[int] = []
    var remaining = capacity
    var i = n
    while i > 0 {
        if best[i][remaining] != best[i - 1][remaining] {
            items.push(i - 1)
            remaining -= weights[i - 1]
        }
        i -= 1
    }
    return (best[n][capacity], reversed(items))
}

#: Solve the unbounded knapsack: return the best value and the items used.
#:
#: The returned index list is a multiset and may repeat an index. Items of
#: weight zero are rejected, since they'd be worth taking infinitely often.
#:
#: ```
#: knapsack_unbounded([3, 4, 5], [4, 5, 8], 10)    # (16, [2, 2])
#: ```
pub fn knapsack_unbounded(
    weights: list[int],
    values: list[int],
    capacity: int
) -> tuple[int, list[int]] {
    _check_knapsack(weights, values, capacity)
    let n = weights.len()
    for i in 0..n {
        if weights[i] == 0 {
            throw ValueError(f"unbounded knapsack needs positive weights, got 0 at index {i}")
        }
    }
    let best = [0 for _ in 0..=capacity]
    let choice = [-1 for _ in 0..=capacity]
    for c in 1..=capacity {
        for i in 0..n {
            if weights[i] <= c {
                let taken = best[c - weights[i]] + values[i]
                if taken > best[c] {
                    best[c] = taken
                    choice[c] = i
                }
            }
        }
    }
    let items: list[int] = []
    var remaining = capacity
    while remaining > 0 and choice[remaining] >= 0 {
        items.push(choice[remaining])
        remaining -= weights[choice[remaining]]
    }
    return (best[capacity], sorted(items))
}

#: Return a smallest multiset of `coins` summing to `amount`, or null.
#:
#: ```
#: coin_change_min([1, 5, 6, 9], 11)    # [5, 6]
#: coin_change_min([5, 7], 3)           # null
#: ```
pub fn coin_change_min(coins: list[int], amount: int) -> list[int]? {
    _check_coins(coins, amount)
    let unreachable = amount + 1
    let best = [unreachable for _ in 0..=amount]
    let choice = [-1 for _ in 0..=amount]
    best[0] = 0
    for value in 1..=amount {
        for coin in coins {
            if coin <= value and best[value - coin] + 1 < best[value] {
                best[value] = best[value - coin] + 1
                choice[value] = coin
            }
        }
    }
    if best[amount] == unreachable { return null }
    let picked: list[int] = []
    var remaining = amount
    while remaining > 0 {
        picked.push(choice[remaining])
        remaining -= choice[remaining]
    }
    return sorted(picked)
}

#: Count the distinct multisets of `coins` that sum to `amount`.
#:
#: Order does not matter (`1 + 2` and `2 + 1` are one way): looping over coins
#: outside and amounts inside is what counts combinations, not permutations.
#:
#: ```
#: coin_change_ways([1, 2, 5], 5)    # 4
#: coin_change_ways([2], 3)          # 0
#: coin_change_ways([2], 0)          # 1
#: ```
pub fn coin_change_ways(coins: list[int], amount: int) -> int {
    _check_coins(coins, amount)
    let ways = [0 for _ in 0..=amount]
    ways[0] = 1
    for coin in coins {
        for value in coin..=amount {
            ways[value] += ways[value - coin]
        }
    }
    return ways[amount]
}

#: Return the indices of a sublist of `xs` summing to `target`, or null.
#:
#: Elements must be non-negative — that's what bounds the O(n*target) table.
#:
#: ```
#: subset_sum([2, 3, 7, 8, 10], 11)          # [1, 3]
#: subset_sum([3, 34, 4, 12, 5, 2], 30)      # null
#: ```
pub fn subset_sum(xs: list[int], target: int) -> list[int]? {
    let reachable = _subset_table(xs, target)
    if not reachable[xs.len()][target] { return null }
    return _subset_indices(reachable, xs, target)
}

#: Split the indices of `xs` into two groups of equal sum, or return null.
#:
#: Looks for a sublist summing to half the total; an odd total is rejected
#: outright. Elements must be non-negative, as for `subset_sum`.
#:
#: ```
#: partition_equal([1, 5, 11, 5])    # ([2], [0, 1, 3])
#: partition_equal([1, 2, 3, 5])     # null
#: ```
pub fn partition_equal(xs: list[int]) -> tuple[list[int], list[int]]? {
    let total = sum(xs)
    if total % 2 != 0 { return null }
    let half = total // 2
    let reachable = _subset_table(xs, half)
    if not reachable[xs.len()][half] { return null }
    let first = _subset_indices(reachable, xs, half)
    let taken = [false for _ in 0..xs.len()]
    for index in first { taken[index] = true }
    let second = [i for i in 0..xs.len() if not taken[i]]
    return (first, second)
}

#: Return the cheapest cost of multiplying a chain of matrices, and its shape.
#:
#: `dimensions` holds n + 1 numbers for n matrices: matrix i is
#: `dimensions[i - 1]` by `dimensions[i]`. The shape string is the
#: parenthesisation, matrices named `A1` upwards.
#:
#: ```
#: matrix_chain_order([40, 20, 30, 10, 30])    # (26000, "((A1(A2A3))A4)")
#: ```
pub fn matrix_chain_order(dimensions: list[int]) -> tuple[int, str] {
    if dimensions.len() < 2 {
        throw ValueError(f"need at least two dimensions, got {dimensions.len()}")
    }
    let n = dimensions.len() - 1
    let cost = [[0 for _ in 0..=n] for _ in 0..=n]
    let split = [[0 for _ in 0..=n] for _ in 0..=n]
    for width in 2..=n {
        for i in 1..=n - width + 1 {
            let j = i + width - 1
            cost[i][j] = -1
            for k in i..j {
                let candidate = cost[i][k] +
                    cost[k + 1][j] +
                    dimensions[i - 1] * dimensions[k] * dimensions[j]
                if cost[i][j] < 0 or candidate < cost[i][j] {
                    cost[i][j] = candidate
                    split[i][j] = k
                }
            }
        }
    }
    return (cost[1][n], _parenthesize(split, 1, n))
}

#: Cut a rod of `length` for the best total price, returning price and pieces.
#:
#: `prices[i]` is what a piece of length `i + 1` sells for.
#:
#: ```
#: rod_cutting([1, 5, 8, 9, 10, 17, 17, 20], 8)    # (22, [2, 6])
#: ```
pub fn rod_cutting(prices: list[int], length: int) -> tuple[int, list[int]] {
    if length < 0 { throw ValueError(f"length must be non-negative, got {length}") }
    let longest = prices.len()
    let best = [0 for _ in 0..=length]
    let cut = [0 for _ in 0..=length]
    for total in 1..=length {
        for piece in 1..=min(total, longest) {
            let candidate = prices[piece - 1] + best[total - piece]
            if candidate > best[total] {
                best[total] = candidate
                cut[total] = piece
            }
        }
    }
    let pieces: list[int] = []
    var remaining = length
    while remaining > 0 and cut[remaining] > 0 {
        pieces.push(cut[remaining])
        remaining -= cut[remaining]
    }
    return (best[length], sorted(pieces))
}

#: Return the maximum-sum contiguous slice of `xs`.
#:
#: Kadane's scan. The slice is never empty, so an all-negative list yields its
#: largest single element.
#:
#: ```
#: max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]).sum      # 6
#: max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]).start    # 3
#: max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]).stop     # 7
#: ```
pub fn max_subarray(xs: list[int]) -> Subarray {
    if xs.len() == 0 { throw ValueError("max_subarray of an empty list is undefined") }
    return _kadane(xs)
}

#: Return the maximum-sum rectangular block of `grid`.
#:
#: Fixes each pair of rows, collapses the columns between them into a running
#: sum, and runs Kadane's scan for the best column span. The block is never
#: empty.
#:
#: ```
#: let grid = [[1, 2, -1, -4, -20], [-8, -3, 4, 2, 1],
#:             [3, 8, 10, 1, 3], [-4, -1, 1, 7, -6]]
#: max_submatrix(grid).sum       # 29
#: max_submatrix(grid).top       # 1
#: max_submatrix(grid).right     # 4
#: ```
pub fn max_submatrix(grid: list[list[int]]) -> Submatrix {
    let rows = grid.len()
    if rows == 0 { throw ValueError("max_submatrix of an empty grid is undefined") }
    let columns = grid[0].len()
    if columns == 0 { throw ValueError("max_submatrix of a grid with no columns is undefined") }
    for r in 0..rows {
        if grid[r].len() != columns {
            throw ValueError(f"row {r} has {grid[r].len()} columns, expected {columns}")
        }
    }
    # Seeded with the top-left cell (a valid block), so every replacement below is a strict improvement.
    var best = Submatrix(grid[0][0], 0, 0, 1, 1)
    for top in 0..rows {
        let column_sums = [0 for _ in 0..columns]
        for bottom in top..rows {
            for c in 0..columns { column_sums[c] += grid[bottom][c] }
            let span = _kadane(column_sums)
            if span.sum > best.sum {
                best = Submatrix(span.sum, top, span.start, bottom + 1, span.stop)
            }
        }
    }
    return best
}

#: Choose non-adjacent houses of maximum total value.
#:
#: O(n) space holds the take/skip decision at each house, for reconstruction.
#:
#: ```
#: house_robber([2, 7, 9, 3, 1])    # (12, [0, 2, 4])
#: house_robber([])                 # (0, [])
#: ```
pub fn house_robber(values: list[int]) -> tuple[int, list[int]] {
    let n = values.len()
    let best = [0 for _ in 0..=n]
    let take = [false for _ in 0..=n]
    for i in 1..=n {
        let skipped = best[i - 1]
        let robbed = best[max(i - 2, 0)] + values[i - 1]
        if robbed > skipped {
            best[i] = robbed
            take[i] = true
        } else {
            best[i] = skipped
        }
    }
    let houses: list[int] = []
    var i = n
    while i > 0 {
        if take[i] {
            houses.push(i - 1)
            i -= 2
        } else {
            i -= 1
        }
    }
    return (best[n], reversed(houses))
}

#: Return the indices of a shortest jump path from the first cell to the last.
#:
#: `steps[i]` is the furthest distance forward that may be jumped from cell
#: `i`. The greedy scan only needs the furthest-reaching cell at each layer —
#: it's always at least as good as any other reachable cell.
#:
#: ```
#: jump_game([2, 3, 1, 1, 4])    # [0, 1, 4]
#: jump_game([3, 2, 1, 0, 4])    # null
#: ```
pub fn jump_game(steps: list[int]) -> list[int]? {
    let n = steps.len()
    if n == 0 { throw ValueError("jump_game of an empty list is undefined") }
    for i in 0..n {
        if steps[i] < 0 {
            throw ValueError(f"steps must be non-negative, got {steps[i]} at index {i}")
        }
    }
    let path = [0]
    var current = 0
    while current < n - 1 {
        if steps[current] == 0 { return null }
        if current + steps[current] >= n - 1 {
            path.push(n - 1)
            return path
        }
        var next = -1
        var furthest = -1
        for candidate in current + 1..=min(current + steps[current], n - 1) {
            if candidate + steps[candidate] > furthest {
                furthest = candidate + steps[candidate]
                next = candidate
            }
        }
        path.push(next)
        current = next
    }
    return path
}

#: Count the paths from the top-left to the bottom-right cell of `grid`.
#:
#: Only steps right and down are allowed; a `true` cell is an obstacle.
#:
#: ```
#: unique_paths([[false, false, false], [false, true, false], [false, false, false]])   # 2
#: unique_paths([[false, false], [false, false]])                                       # 2
#: unique_paths([[true]])                                                               # 0
#: ```
pub fn unique_paths(grid: list[list[bool]]) -> int {
    let rows = grid.len()
    if rows == 0 { throw ValueError("unique_paths of an empty grid is undefined") }
    let columns = grid[0].len()
    if columns == 0 { throw ValueError("unique_paths of a grid with no columns is undefined") }
    let ways = [0 for _ in 0..columns]
    if not grid[0][0] { ways[0] = 1 }
    for r in 0..rows {
        if grid[r].len() != columns {
            throw ValueError(f"row {r} has {grid[r].len()} columns, expected {columns}")
        }
        for c in 0..columns {
            if grid[r][c] {
                ways[c] = 0
            } elif c > 0 {
                ways[c] += ways[c - 1]
            }
        }
    }
    return ways[columns - 1]
}

#: Split `text` into a sequence of `words`, or return null if it cannot be done.
#:
#: Only the last L characters (L = longest word) can start the final word,
#: which bounds the search. The first segmentation found per prefix wins.
#:
#: ```
#: word_break("applepenapple", {"apple", "pen"})                    # ["apple", "pen", "apple"]
#: word_break("catsandog", {"cats", "dog", "sand", "and", "cat"})   # null
#: ```
pub fn word_break(text: str, words: set[str]) -> list[str]? {
    let n = text.len()
    var longest = 0
    for word in words { longest = max(longest, word.len()) }
    let start_of = [-1 for _ in 0..=n]
    let breakable = [false for _ in 0..=n]
    breakable[0] = true
    for stop in 1..=n {
        for start in max(0, stop - longest)..stop {
            if breakable[start] and text[start:stop] in words {
                breakable[stop] = true
                start_of[stop] = start
                break
            }
        }
    }
    if not breakable[n] { return null }
    let pieces: list[str] = []
    var stop = n
    while stop > 0 {
        pieces.push(text[start_of[stop]:stop])
        stop = start_of[stop]
    }
    return reversed(pieces)
}

#: Split `text` into the fewest palindromic pieces possible.
#:
#: A palindrome table is filled first, then a single left-to-right pass finds
#: the minimum-cut split.
#:
#: ```
#: palindrome_partitioning("aab")        # ["aa", "b"]
#: palindrome_partitioning("abcde")      # ["a", "b", "c", "d", "e"]
#: palindrome_partitioning("racecar")    # ["racecar"]
#: ```
pub fn palindrome_partitioning(text: str) -> list[str] {
    let n = text.len()
    if n == 0 { return [] }
    let palindrome = [[false for _ in 0..n] for _ in 0..n]
    var i = n - 1
    while i >= 0 {
        for j in i..n {
            if text[i] == text[j] and (j - i < 2 or palindrome[i + 1][j - 1]) {
                palindrome[i][j] = true
            }
        }
        i -= 1
    }
    let cuts = [0 for _ in 0..=n]
    let start_of = [0 for _ in 0..=n]
    for stop in 1..=n {
        # Seeded with peeling off the last character alone — always a valid (palindrome) split.
        cuts[stop] = cuts[stop - 1] + 1
        start_of[stop] = stop - 1
        for start in 0..stop - 1 {
            if palindrome[start][stop - 1] and cuts[start] + 1 < cuts[stop] {
                cuts[stop] = cuts[start] + 1
                start_of[stop] = start
            }
        }
    }
    let pieces: list[str] = []
    var stop = n
    while stop > 0 {
        pieces.push(text[start_of[stop]:stop])
        stop = start_of[stop]
    }
    return reversed(pieces)
}

#: Return the `n`-th Catalan number, counting from `catalan(0) == 1`.
#:
#: Uses the convolution `C(n) = sum C(i) * C(n-1-i)`, which never forms a value
#: larger than the answer — the closed form `binomial(2n,n)/(n+1)` overflows
#: long before the result does. `C(35)` is the last one that fits a 64-bit int.
#:
#: ```
#: catalan(5)     # 42
#: catalan(35)    # 3116285494907301262
#: ```
pub fn catalan(n: int) -> int {
    if n < 0 { throw ValueError(f"n must be non-negative, got {n}") }
    let table = [0 for _ in 0..=n]
    table[0] = 1
    for i in 1..=n {
        for j in 0..i {
            table[i] += table[j] * table[i - 1 - j]
        }
    }
    return table[n]
}

#: Turn a recursive definition into a memoised function.
#:
#: `body` is called with the memoised function itself as its first argument, so
#: a recursive rule can be written without naming the function it defines. The
#: argument must be hashable — it's the cache key.
#:
#: ```
#: let fib = memoize_recursive(fn(recurse, n) {
#:     if n < 2 { return n }
#:     return recurse(n - 1) + recurse(n - 2)
#: })
#: fib(50)    # 12586269025
#: ```
pub fn memoize_recursive(body: fn(fn(any) -> any, any) -> any) -> fn(any) -> any {
    # `var`, not `let` — the closure below must capture the cell, patched to the real entry point next.
    var recurse: fn(any) -> any = |argument| argument
    recurse = memoize(fn(argument: any) -> any { return body(recurse, argument) })
    return recurse
}

fn _mat2_fib_power(exponent: int) -> list[int] {
    var result = [1, 0, 0, 1]
    var base = [1, 1, 1, 0]
    var remaining = exponent
    while remaining > 0 {
        if remaining & 1 == 1 { result = _mat2_multiply(result, base) }
        remaining >>= 1
        # Skips the final squaring — keeps every intermediate entry at or below F(92).
        if remaining > 0 { base = _mat2_multiply(base, base) }
    }
    return result
}

fn _mat2_multiply(x: list[int], y: list[int]) -> list[int] {
    return [
        x[0] * y[0] + x[1] * y[2],
        x[0] * y[1] + x[1] * y[3],
        x[2] * y[0] + x[3] * y[2],
        x[2] * y[1] + x[3] * y[3],
    ]
}

fn _lower_bound(values: list[int], target: int, strict: bool) -> int {
    var low = 0
    var high = values.len()
    while low < high {
        let middle = (low + high) // 2
        let before = strict ? values[middle] < target : values[middle] <= target
        if before { low = middle + 1 } else { high = middle }
    }
    return low
}

fn _kadane(xs: list[int]) -> Subarray {
    var best_sum = xs[0]
    var best_start = 0
    var best_stop = 1
    var current_sum = xs[0]
    var current_start = 0
    for i in 1..xs.len() {
        if current_sum < 0 {
            current_sum = xs[i]
            current_start = i
        } else {
            current_sum += xs[i]
        }
        if current_sum > best_sum {
            best_sum = current_sum
            best_start = current_start
            best_stop = i + 1
        }
    }
    return Subarray(best_sum, best_start, best_stop)
}

fn _parenthesize(split: list[list[int]], i: int, j: int) -> str {
    if i == j { return f"A{i}" }
    let k = split[i][j]
    return f"({_parenthesize(split, i, k)}{_parenthesize(split, k + 1, j)})"
}

fn _expect_character(source: str, index: int, character: str) -> void {
    if index < 0 or index >= source.len() {
        throw IndexError(f"edit operation names index {index}, outside 0..{source.len()}")
    }
    if source[index] != character {
        throw ValueError(
            f"edit operation expected {character} at index {index}, " +
                f"but the source holds {source[index]}"
        )
    }
}

fn _subset_table(xs: list[int], target: int) -> list[list[bool]] {
    if target < 0 { throw ValueError(f"target must be non-negative, got {target}") }
    let n = xs.len()
    for i in 0..n {
        if xs[i] < 0 {
            throw ValueError(f"elements must be non-negative, got {xs[i]} at index {i}")
        }
    }
    let reachable = [[false for _ in 0..=target] for _ in 0..=n]
    for i in 0..=n { reachable[i][0] = true }
    for i in 1..=n {
        for s in 1..=target {
            reachable[i][s] = reachable[i - 1][s]
            if not reachable[i][s] and xs[i - 1] <= s {
                reachable[i][s] = reachable[i - 1][s - xs[i - 1]]
            }
        }
    }
    return reachable
}

fn _subset_indices(reachable: list[list[bool]], xs: list[int], target: int) -> list[int] {
    let picked: list[int] = []
    var remaining = target
    var i = xs.len()
    while i > 0 {
        # Element i - 1 is needed exactly when the sum was out of reach without it.
        if not reachable[i - 1][remaining] {
            picked.push(i - 1)
            remaining -= xs[i - 1]
        }
        i -= 1
    }
    return reversed(picked)
}

fn _check_knapsack(weights: list[int], values: list[int], capacity: int) -> void {
    if weights.len() != values.len() {
        throw ValueError(f"weights has {weights.len()} entries but values has {values.len()}")
    }
    if capacity < 0 { throw ValueError(f"capacity must be non-negative, got {capacity}") }
    for i in 0..weights.len() {
        if weights[i] < 0 {
            throw ValueError(f"weights must be non-negative, got {weights[i]} at index {i}")
        }
    }
}

fn _check_coins(coins: list[int], amount: int) -> void {
    if amount < 0 { throw ValueError(f"amount must be non-negative, got {amount}") }
    for i in 0..coins.len() {
        if coins[i] <= 0 {
            throw ValueError(f"coin denominations must be positive, got {coins[i]} at index {i}")
        }
    }
}
