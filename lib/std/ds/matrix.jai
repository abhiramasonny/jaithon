#: A dense matrix of floats with the standard decompositions.
#:
#: Rows are stored as `list[list[float]]`, so `m.row(i)` is a copy of a stored
#: row and `m.column(j)` is gathered. Every operation that combines two matrices
#: names both shapes when they disagree.
import std.math
import std.random

const DEFAULT_TOLERANCE = 1e-12

#: A dense, mutable matrix of `float` values.
#:
#: ```
#: let m = Matrix.from_list([[1.0, 2.0], [3.0, 4.0]])
#: m.shape()          # (2, 2)
#: m.determinant()    # -2.0
#: m.trace()          # 5.0
#: m.transpose().row(0)   # [1.0, 3.0]
#: ```
pub class Matrix {
    let rows: int
    let columns: int
    let cells: list[list[float]]

    #: Create a `rows` by `columns` matrix of zeros.
    pub fn init(self, rows: int, columns: int) {
        if rows < 0 or columns < 0 {
            throw ValueError(f"shape must be non-negative, got ({rows}, {columns})")
        }
        self.rows = rows
        self.columns = columns
        let cells: list[list[float]] = []
        for _row in 0..rows {
            let line: list[float] = []
            for _column in 0..columns { line.push(0.0) }
            cells.push(line)
        }
        self.cells = cells
    }

    #: Create a matrix of zeros.
    pub static fn zeros(rows: int, columns: int) -> Matrix {
        return Matrix(rows, columns)
    }

    #: Create a matrix of ones.
    pub static fn ones(rows: int, columns: int) -> Matrix {
        return Matrix.filled(rows, columns, 1.0)
    }

    #: Create a matrix whose every entry is `value`.
    pub static fn filled(rows: int, columns: int, value: float) -> Matrix {
        let m = Matrix(rows, columns)
        for row in 0..rows {
            for column in 0..columns { m.cells[row][column] = value }
        }
        return m
    }

    #: Create the `size` by `size` identity matrix.
    pub static fn identity(size: int) -> Matrix {
        let m = Matrix(size, size)
        for index in 0..size { m.cells[index][index] = 1.0 }
        return m
    }

    #: Create a matrix from a list of equal-length rows.
    #:
    #: Raises `ValueError` when the rows are ragged.
    pub static fn from_list(values: list[list[float]]) -> Matrix {
        let rows = values.len()
        let columns = rows == 0 ? 0 : values[0].len()
        for (index, line) in values.enumerate() {
            if line.len() != columns {
                throw ValueError(f"row {index} has {line.len()} entries, row 0 has {columns}")
            }
        }
        let m = Matrix(rows, columns)
        for (row, line) in values.enumerate() {
            for (column, value) in line.enumerate() { m.cells[row][column] = value }
        }
        return m
    }

    #: Create a matrix whose entries are uniform in `[low, high)`.
    pub static fn random(rows: int, columns: int, low: float = 0.0, high: float = 1.0) -> Matrix {
        if high <= low {
            throw ValueError(f"high must exceed low, got low={low} high={high}")
        }
        let m = Matrix(rows, columns)
        for row in 0..rows {
            for column in 0..columns { m.cells[row][column] = random.uniform(low, high) }
        }
        return m
    }

    #: The (rows, columns) pair.
    pub fn shape(self) -> tuple[int, int] {
        return (self.rows, self.columns)
    }

    #: Number of rows.
    pub fn row_count(self) -> int {
        return self.rows
    }

    #: Number of columns.
    pub fn column_count(self) -> int {
        return self.columns
    }

    #: Report whether the matrix is square.
    pub fn is_square(self) -> bool {
        return self.rows == self.columns
    }

    #: The entry at `(row, column)`.
    pub fn get(self, row: int, column: int) -> float {
        self._check(row, column)
        return self.cells[row][column]
    }

    #: Store `value` at `(row, column)`.
    pub fn set(self, row: int, column: int, value: float) -> void {
        self._check(row, column)
        self.cells[row][column] = value
    }

    #: A copy of row `row`.
    pub fn row(self, row: int) -> list[float] {
        if row < 0 or row >= self.rows {
            throw IndexError(f"row {row} is not within [0, {self.rows})")
        }
        return [value for value in self.cells[row]]
    }

    #: Column `column`, gathered top to bottom.
    pub fn column(self, column: int) -> list[float] {
        if column < 0 or column >= self.columns {
            throw IndexError(f"column {column} is not within [0, {self.columns})")
        }
        return [self.cells[row][column] for row in 0..self.rows]
    }

    #: The rows, as a fresh list of fresh lists.
    pub fn to_list(self) -> list[list[float]] {
        return [self.row(row) for row in 0..self.rows]
    }

    #: An independent copy.
    pub fn copy(self) -> Matrix {
        return Matrix.from_list(self.to_list())
    }

    #: Report whether every entry differs by at most `tolerance`.
    pub fn approx_equals(self, other: Matrix, tolerance: float = 1e-9) -> bool {
        if self.rows != other.rows or self.columns != other.columns { return false }
        for row in 0..self.rows {
            for column in 0..self.columns {
                if abs(self.cells[row][column] - other.cells[row][column]) > tolerance {
                    return false
                }
            }
        }
        return true
    }

    #: The entry-wise sum.
    pub fn add(self, other: Matrix) -> Matrix {
        self._require_same_shape(other, "add")
        let result = Matrix(self.rows, self.columns)
        for row in 0..self.rows {
            for column in 0..self.columns {
                result.cells[row][column] = self.cells[row][column] + other.cells[row][column]
            }
        }
        return result
    }

    #: The entry-wise difference.
    pub fn sub(self, other: Matrix) -> Matrix {
        self._require_same_shape(other, "sub")
        let result = Matrix(self.rows, self.columns)
        for row in 0..self.rows {
            for column in 0..self.columns {
                result.cells[row][column] = self.cells[row][column] - other.cells[row][column]
            }
        }
        return result
    }

    #: Every entry multiplied by `factor`.
    pub fn scale(self, factor: float) -> Matrix {
        let result = Matrix(self.rows, self.columns)
        for row in 0..self.rows {
            for column in 0..self.columns {
                result.cells[row][column] = self.cells[row][column] * factor
            }
        }
        return result
    }

    #: The matrix product `self * other`.
    #:
    #: Raises `ValueError` unless `self.column_count() == other.row_count()`.
    #:
    #: ```
    #: let a = Matrix.from_list([[1.0, 2.0], [3.0, 4.0]])
    #: a.multiply(Matrix.identity(2)).approx_equals(a)   # true
    #: ```
    pub fn multiply(self, other: Matrix) -> Matrix {
        if self.columns != other.rows {
            throw ValueError(
                f"cannot multiply ({self.rows}, {self.columns}) by ({other.rows}, {other.columns})"
            )
        }
        let result = Matrix(self.rows, other.columns)
        for row in 0..self.rows {
            let line = self.cells[row]
            let out = result.cells[row]
            # Accumulating into the output row one source column at a time
            # walks `other` in row order, which is how it is laid out.
            for inner in 0..self.columns {
                let factor = line[inner]
                if factor == 0.0 { continue }
                let source = other.cells[inner]
                for column in 0..other.columns { out[column] += factor * source[column] }
            }
        }
        return result
    }

    #: The product with a column vector, as a list.
    pub fn multiply_vector(self, vector: list[float]) -> list[float] {
        if vector.len() != self.columns {
            throw ValueError(
                f"cannot multiply ({self.rows}, {self.columns}) by a vector of {vector.len()}"
            )
        }
        let result: list[float] = []
        for row in 0..self.rows {
            var total = 0.0
            for column in 0..self.columns { total += self.cells[row][column] * vector[column] }
            result.push(total)
        }
        return result
    }

    #: The transpose.
    pub fn transpose(self) -> Matrix {
        let result = Matrix(self.columns, self.rows)
        for row in 0..self.rows {
            for column in 0..self.columns {
                result.cells[column][row] = self.cells[row][column]
            }
        }
        return result
    }

    #: The sum of the diagonal entries.
    pub fn trace(self) -> float {
        self._require_square("trace")
        var total = 0.0
        for index in 0..self.rows { total += self.cells[index][index] }
        return total
    }

    #: The Frobenius norm, or with `order` 1 or `inf` the induced matrix norm.
    #:
    #: `order` is "fro" (default), "1" for the maximum absolute column sum, or
    #: "inf" for the maximum absolute row sum.
    pub fn norm(self, order: str = "fro") -> float {
        if order == "fro" {
            var total = 0.0
            for line in self.cells {
                for value in line { total += value * value }
            }
            return math.sqrt(total)
        }
        if order == "1" {
            var best = 0.0
            for column in 0..self.columns {
                var total = 0.0
                for row in 0..self.rows { total += abs(self.cells[row][column]) }
                best = max(best, total)
            }
            return best
        }
        if order == "inf" {
            var best = 0.0
            for line in self.cells {
                var total = 0.0
                for value in line { total += abs(value) }
                best = max(best, total)
            }
            return best
        }
        throw ValueError(f"order must be 'fro', '1' or 'inf', got {order}")
    }

    #: The LU decomposition with partial pivoting.
    #:
    #: Returns `(lower, upper, permutation, sign)` where `permutation[i]` is the
    #: original index of the row now in position i, and `sign` is 1.0 or -1.0
    #: according to the parity of the row swaps. `lower` is unit lower
    #: triangular, so `P * self == lower * upper`.
    pub fn lu_decompose(self) -> tuple[Matrix, Matrix, list[int], float] {
        self._require_square("lu_decompose")
        let size = self.rows
        let upper = self.copy()
        let lower = Matrix.identity(size)
        let permutation = [index for index in 0..size]
        var sign = 1.0
        for pivot in 0..size {
            var best = pivot
            for row in pivot + 1..size {
                if abs(upper.cells[row][pivot]) > abs(upper.cells[best][pivot]) { best = row }
            }
            if abs(upper.cells[best][pivot]) <= DEFAULT_TOLERANCE { continue }
            if best != pivot {
                upper._swap_rows(pivot, best)
                sign = -sign
                let carry = permutation[pivot]
                permutation[pivot] = permutation[best]
                permutation[best] = carry
                # The multipliers already written into `lower` belong to the
                # rows that just moved, so they have to move with them.
                for column in 0..pivot {
                    let held = lower.cells[pivot][column]
                    lower.cells[pivot][column] = lower.cells[best][column]
                    lower.cells[best][column] = held
                }
            }
            for row in pivot + 1..size {
                let factor = upper.cells[row][pivot] / upper.cells[pivot][pivot]
                lower.cells[row][pivot] = factor
                if factor == 0.0 { continue }
                for column in pivot..size {
                    upper.cells[row][column] -= factor * upper.cells[pivot][column]
                }
            }
        }
        return (lower, upper, permutation, sign)
    }

    #: The determinant, from the LU factors.
    #:
    #: O(n^3), and numerically far better behaved than cofactor expansion.
    pub fn determinant(self) -> float {
        self._require_square("determinant")
        let factors = self.lu_decompose()
        let upper = factors[1]
        var product = factors[3]
        for index in 0..self.rows { product *= upper.cells[index][index] }
        return product
    }

    #: The rank, from the row echelon form of a copy.
    pub fn rank(self, tolerance: float = 1e-10) -> int {
        let work = self.copy()
        var rank = 0
        var pivot_row = 0
        for column in 0..self.columns {
            if pivot_row >= self.rows { break }
            var best = pivot_row
            for row in pivot_row + 1..self.rows {
                if abs(work.cells[row][column]) > abs(work.cells[best][column]) { best = row }
            }
            if abs(work.cells[best][column]) <= tolerance { continue }
            work._swap_rows(pivot_row, best)
            for row in pivot_row + 1..self.rows {
                let factor = work.cells[row][column] / work.cells[pivot_row][column]
                for inner in column..self.columns {
                    work.cells[row][inner] -= factor * work.cells[pivot_row][inner]
                }
            }
            pivot_row += 1
            rank += 1
        }
        return rank
    }

    #: Solve `self * x = rhs` by Gaussian elimination with partial pivoting.
    #:
    #: Raises `ValueError` when the matrix is singular or the shapes disagree.
    #:
    #: ```
    #: let a = Matrix.from_list([[2.0, 1.0], [1.0, 3.0]])
    #: a.solve([3.0, 4.0])   # [1.0, 1.0]
    #: ```
    pub fn solve(self, rhs: list[float]) -> list[float] {
        self._require_square("solve")
        if rhs.len() != self.rows {
            throw ValueError(
                f"cannot solve ({self.rows}, {self.columns}) against {rhs.len()} values"
            )
        }
        let size = self.rows
        let work = self.copy()
        let values = [value for value in rhs]
        for pivot in 0..size {
            var best = pivot
            for row in pivot + 1..size {
                if abs(work.cells[row][pivot]) > abs(work.cells[best][pivot]) { best = row }
            }
            if abs(work.cells[best][pivot]) <= DEFAULT_TOLERANCE {
                throw ValueError(f"matrix is singular at column {pivot}")
            }
            if best != pivot {
                work._swap_rows(pivot, best)
                let carry = values[pivot]
                values[pivot] = values[best]
                values[best] = carry
            }
            for row in pivot + 1..size {
                let factor = work.cells[row][pivot] / work.cells[pivot][pivot]
                if factor == 0.0 { continue }
                for column in pivot..size {
                    work.cells[row][column] -= factor * work.cells[pivot][column]
                }
                values[row] -= factor * values[pivot]
            }
        }
        let solution: list[float] = []
        for _index in 0..size { solution.push(0.0) }
        var row = size - 1
        while row >= 0 {
            var total = values[row]
            for column in row + 1..size { total -= work.cells[row][column] * solution[column] }
            solution[row] = total / work.cells[row][row]
            row -= 1
        }
        return solution
    }

    #: The inverse, by solving against each identity column.
    #:
    #: Raises `ValueError` when the matrix is singular.
    pub fn inverse(self) -> Matrix {
        self._require_square("inverse")
        let size = self.rows
        let result = Matrix(size, size)
        for column in 0..size {
            let unit: list[float] = []
            for row in 0..size { unit.push(row == column ? 1.0 : 0.0) }
            let solution = self.solve(unit)
            for row in 0..size { result.cells[row][column] = solution[row] }
        }
        return result
    }

    #: The QR decomposition by modified Gram-Schmidt, for a matrix with at
    #: least as many rows as columns.
    #:
    #: Returns `(q, r)` with `q` orthonormal in its columns, `r` upper
    #: triangular, and `q * r == self`. Modified Gram-Schmidt is used over the
    #: classical form because it stays orthogonal on ill-conditioned input.
    pub fn qr_decompose(self) -> tuple[Matrix, Matrix] {
        if self.rows < self.columns {
            throw ValueError(
                f"qr_decompose needs rows >= columns, got ({self.rows}, {self.columns})"
            )
        }
        let q = self.copy()
        let r = Matrix(self.columns, self.columns)
        for column in 0..self.columns {
            var length = 0.0
            for row in 0..self.rows { length += q.cells[row][column] ** 2 }
            length = math.sqrt(length)
            r.cells[column][column] = length
            if length > DEFAULT_TOLERANCE {
                for row in 0..self.rows { q.cells[row][column] /= length }
            }
            for other in column + 1..self.columns {
                var projection = 0.0
                for row in 0..self.rows { projection += q.cells[row][column] * q.cells[row][other] }
                r.cells[column][other] = projection
                for row in 0..self.rows {
                    q.cells[row][other] -= projection * q.cells[row][column]
                }
            }
        }
        return (q, r)
    }

    #: Report whether the matrix equals its own transpose within `tolerance`.
    pub fn is_symmetric(self, tolerance: float = 1e-9) -> bool {
        if not self.is_square() { return false }
        for row in 0..self.rows {
            for column in row + 1..self.columns {
                if abs(self.cells[row][column] - self.cells[column][row]) > tolerance {
                    return false
                }
            }
        }
        return true
    }

    #: The eigenvalues of a symmetric matrix, ascending, by cyclic Jacobi.
    #:
    #: A sweep annihilates each off-diagonal entry in turn with a plane
    #: rotation, and the diagonal converges to the spectrum. Raises `ValueError`
    #: on a non-symmetric matrix, whose eigenvalues may be complex and are out
    #: of scope here.
    #:
    #: ```
    #: let m = Matrix.from_list([[2.0, 1.0], [1.0, 2.0]])
    #: let spectrum = m.eigenvalues()
    #: abs(spectrum[0] - 1.0) < 1e-9   # true
    #: abs(spectrum[1] - 3.0) < 1e-9   # true
    #: ```
    pub fn eigenvalues(self, tolerance: float = 1e-12, max_sweeps: int = 100) -> list[float] {
        if not self.is_symmetric() {
            throw ValueError("eigenvalues requires a symmetric matrix")
        }
        let size = self.rows
        let work = self.copy()
        for _sweep in 0..max_sweeps {
            var off_diagonal = 0.0
            for row in 0..size {
                for column in row + 1..size {
                    off_diagonal += work.cells[row][column] ** 2
                }
            }
            if off_diagonal <= tolerance { break }
            for pivot in 0..size {
                for other in pivot + 1..size {
                    let entry = work.cells[pivot][other]
                    if abs(entry) <= tolerance { continue }
                    # Choose the rotation angle that zeroes cells[pivot][other]
                    # exactly; theta is cot(2 * phi) for that rotation.
                    let spread = work.cells[other][other] - work.cells[pivot][pivot]
                    let theta = spread / (2.0 * entry)
                    var t = 1.0 / (abs(theta) + math.sqrt(theta * theta + 1.0))
                    if theta < 0.0 { t = -t }
                    let cosine = 1.0 / math.sqrt(t * t + 1.0)
                    let sine = t * cosine
                    work._rotate(pivot, other, cosine, sine)
                }
            }
        }
        let spectrum = [work.cells[index][index] for index in 0..size]
        return sorted(spectrum)
    }

    fn _rotate(self, pivot: int, other: int, cosine: float, sine: float) -> void {
        let size = self.rows
        for index in 0..size {
            let left = self.cells[index][pivot]
            let right = self.cells[index][other]
            self.cells[index][pivot] = cosine * left - sine * right
            self.cells[index][other] = sine * left + cosine * right
        }
        for index in 0..size {
            let left = self.cells[pivot][index]
            let right = self.cells[other][index]
            self.cells[pivot][index] = cosine * left - sine * right
            self.cells[other][index] = sine * left + cosine * right
        }
    }

    fn _swap_rows(self, left: int, right: int) -> void {
        if left == right { return }
        let held = self.cells[left]
        self.cells[left] = self.cells[right]
        self.cells[right] = held
    }

    fn _check(self, row: int, column: int) -> void {
        if row < 0 or row >= self.rows {
            throw IndexError(f"row {row} is not within [0, {self.rows})")
        }
        if column < 0 or column >= self.columns {
            throw IndexError(f"column {column} is not within [0, {self.columns})")
        }
    }

    fn _require_square(self, operation: str) -> void {
        if self.rows != self.columns {
            let shape = f"({self.rows}, {self.columns})"
            throw ValueError(f"{operation} requires a square matrix, got {shape}")
        }
    }

    fn _require_same_shape(self, other: Matrix, operation: str) -> void {
        if self.rows != other.rows or self.columns != other.columns {
            let mine = f"({self.rows}, {self.columns})"
            let theirs = f"({other.rows}, {other.columns})"
            throw ValueError(f"{operation} needs equal shapes, got {mine} and {theirs}")
        }
    }

    fn __add__(self, other: Matrix) -> Matrix { return self.add(other) }

    fn __sub__(self, other: Matrix) -> Matrix { return self.sub(other) }

    fn __mul__(self, other: Matrix) -> Matrix { return self.multiply(other) }

    fn __eq__(self, other: Matrix) -> bool { return self.approx_equals(other, 0.0) }

    fn __str__(self) -> str {
        let lines: list[str] = []
        for line in self.cells { lines.push(f"{line}") }
        let separator = ", "
        let body = separator.join(lines)
        return f"Matrix({self.rows}x{self.columns}, [{body}])"
    }
}
