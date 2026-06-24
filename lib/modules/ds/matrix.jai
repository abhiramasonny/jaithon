class Matrix
    var rows
    var cols
    var data

    func init(self, r, c)
        self.rows = r
        self.cols = c
        self.data = []
        var i = 0
        while i < r
            var row = []
            var j = 0
            while j < c
                _push(row, 0)
                j = j + 1
            end
            _push(self.data, row)
            i = i + 1
        end
    end

    func get(self, r, c)
        return self.data[r][c]
    end

    func set(self, r, c, val)
        self.data[r][c] = val
    end

    func add(self, other)
        var result = new Matrix(self.rows, self.cols)
        var i = 0
        while i < self.rows
            var j = 0
            while j < self.cols
                result.set(i, j, self.get(i, j) + other.get(i, j))
                j = j + 1
            end
            i = i + 1
        end
        return result
    end

    func sub(self, other)
        var result = new Matrix(self.rows, self.cols)
        var i = 0
        while i < self.rows
            var j = 0
            while j < self.cols
                result.set(i, j, self.get(i, j) - other.get(i, j))
                j = j + 1
            end
            i = i + 1
        end
        return result
    end

    func multiply(self, other)
        var result = new Matrix(self.rows, other.cols)
        var i = 0
        while i < self.rows
            var j = 0
            while j < other.cols
                var s = 0
                var k = 0
                while k < self.cols
                    s = s + self.get(i, k) * other.get(k, j)
                    k = k + 1
                end
                result.set(i, j, s)
                j = j + 1
            end
            i = i + 1
        end
        return result
    end

    func scale(self, factor)
        var result = new Matrix(self.rows, self.cols)
        var i = 0
        while i < self.rows
            var j = 0
            while j < self.cols
                result.set(i, j, self.get(i, j) * factor)
                j = j + 1
            end
            i = i + 1
        end
        return result
    end

    func transpose(self)
        var result = new Matrix(self.cols, self.rows)
        var i = 0
        while i < self.rows
            var j = 0
            while j < self.cols
                result.set(j, i, self.get(i, j))
                j = j + 1
            end
            i = i + 1
        end
        return result
    end

    func trace(self)
        var s = 0
        var n = min(self.rows, self.cols)
        var i = 0
        while i < n
            s = s + self.get(i, i)
            i = i + 1
        end
        return s
    end

    func frobenius(self)
        var s = 0
        var i = 0
        while i < self.rows
            var j = 0
            while j < self.cols
                s = s + self.get(i, j) * self.get(i, j)
                j = j + 1
            end
            i = i + 1
        end
        return sqrt(s)
    end

    func row(self, r)
        return arrayCopy(self.data[r])
    end

    func col(self, c)
        var result = []
        var i = 0
        while i < self.rows
            _push(result, self.get(i, c))
            i = i + 1
        end
        return result
    end

    func toString(self)
        var result = ""
        var i = 0
        while i < self.rows
            if i > 0
                result = concat(result, "\n")
            end
            var j = 0
            while j < self.cols
                if j > 0
                    result = concat(result, " ")
                end
                result = concat(result, _str(self.get(i, j)))
                j = j + 1
            end
            i = i + 1
        end
        return result
    end
end

func matrixIdentity(n)
    var m = new Matrix(n, n)
    var i = 0
    while i < n
        m.set(i, i, 1)
        i = i + 1
    end
    return m
end

func matrixZeros(rows, cols)
    return new Matrix(rows, cols)
end

func matrixOnes(rows, cols)
    var m = new Matrix(rows, cols)
    var i = 0
    while i < rows
        var j = 0
        while j < cols
            m.set(i, j, 1)
            j = j + 1
        end
        i = i + 1
    end
    return m
end

func matrixFromFlat(arr, rows, cols)
    var m = new Matrix(rows, cols)
    var i = 0
    while i < rows
        var j = 0
        while j < cols
            m.set(i, j, arr[i * cols + j])
            j = j + 1
        end
        i = i + 1
    end
    return m
end

func matrixDiag(arr)
    var n = _alen(arr)
    var m = new Matrix(n, n)
    var i = 0
    while i < n
        m.set(i, i, arr[i])
        i = i + 1
    end
    return m
end
