class UnionFind
    var parent
    var rank
    var sizeArr
    var count

    func init(self, n)
        self.parent = []
        self.rank = []
        self.sizeArr = []
        self.count = n
        var i = 0
        while i < n
            _push(self.parent, i)
            _push(self.rank, 0)
            _push(self.sizeArr, 1)
            i = i + 1
        end
    end

    func find(self, x)
        if self.parent[x] != x
            self.parent[x] = self.find(self.parent[x])
        end
        return self.parent[x]
    end

    func union(self, x, y)
        var px = self.find(x)
        var py = self.find(y)
        if px == py
            return false
        end
        if self.rank[px] < self.rank[py]
            self.parent[px] = py
            self.sizeArr[py] = self.sizeArr[py] + self.sizeArr[px]
        else if self.rank[px] > self.rank[py]
            self.parent[py] = px
            self.sizeArr[px] = self.sizeArr[px] + self.sizeArr[py]
        else
            self.parent[py] = px
            self.sizeArr[px] = self.sizeArr[px] + self.sizeArr[py]
            self.rank[px] = self.rank[px] + 1
        end
        self.count = self.count - 1
        return true
    end

    func connected(self, x, y)
        return self.find(x) == self.find(y)
    end

    func componentSize(self, x)
        return self.sizeArr[self.find(x)]
    end

    func components(self)
        return self.count
    end

    func groups(self)
        var result = _dict()
        var i = 0
        var n = _alen(self.parent)
        while i < n
            var root = _str(self.find(i))
            if not _dict_has(result, root)
                result[root] = []
            end
            _push(result[root], i)
            i = i + 1
        end
        return result
    end
end

func unionFind(n)
    return new UnionFind(n)
end
