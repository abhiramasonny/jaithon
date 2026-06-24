class Graph
    var _adj
    var _directed

    func init(self)
        self._adj = _dict()
        self._directed = false
    end

    func setDirected(self, d)
        self._directed = d
        return self
    end

    func addVertex(self, v)
        var key = _str(v)
        if not _dict_has(self._adj, key)
            self._adj[key] = []
        end
        return self
    end

    func addEdge(self, u, v)
        self.addVertex(u)
        self.addVertex(v)
        _push(self._adj[_str(u)], v)
        if not self._directed
            _push(self._adj[_str(v)], u)
        end
        return self
    end

    func addWeightedEdge(self, u, v, w)
        self.addVertex(u)
        self.addVertex(v)
        _push(self._adj[_str(u)], [v, w])
        if not self._directed
            _push(self._adj[_str(v)], [u, w])
        end
        return self
    end

    func neighbors(self, v)
        var key = _str(v)
        if _dict_has(self._adj, key)
            return self._adj[key]
        end
        return []
    end

    func vertices(self)
        return _dict_keys(self._adj)
    end

    func hasVertex(self, v)
        return _dict_has(self._adj, _str(v))
    end
end

func graph()
    return new Graph()
end

func directedGraph()
    var g = new Graph()
    g.setDirected(true)
    return g
end

func bfs(g, start)
    var visited = _dict()
    var order = []
    var queue = deque()
    queue.pushBack(start)
    visited[_str(start)] = true
    while not queue.isEmpty()
        var v = queue.popFront()
        _push(order, v)
        var neighbors = g.neighbors(v)
        var i = 0
        while i < _alen(neighbors)
            var n = neighbors[i]
            var key = _str(n)
            if not _dict_has(visited, key)
                visited[key] = true
                queue.pushBack(n)
            end
            i = i + 1
        end
    end
    return order
end

func dfs(g, start)
    var visited = _dict()
    var order = []
    _dfsHelper(g, start, visited, order)
    return order
end

func _dfsHelper(g, v, visited, order)
    visited[_str(v)] = true
    _push(order, v)
    var neighbors = g.neighbors(v)
    var i = 0
    while i < _alen(neighbors)
        var n = neighbors[i]
        if not _dict_has(visited, _str(n))
            _dfsHelper(g, n, visited, order)
        end
        i = i + 1
    end
end

func hasCycle(g)
    var visited = _dict()
    var inStack = _dict()
    var verts = g.vertices()
    var i = 0
    while i < _alen(verts)
        var v = verts[i]
        if not _dict_has(visited, v)
            if _dfsHasCycle(g, v, visited, inStack)
                return true
            end
        end
        i = i + 1
    end
    return false
end

func _dfsHasCycle(g, v, visited, inStack)
    visited[v] = true
    inStack[v] = true
    var neighbors = g.neighbors(v)
    var i = 0
    while i < _alen(neighbors)
        var n = _str(neighbors[i])
        if not _dict_has(visited, n)
            if _dfsHasCycle(g, n, visited, inStack)
                return true
            end
        else
            if _dict_has(inStack, n) and inStack[n]
                return true
            end
        end
        i = i + 1
    end
    inStack[v] = false
    return false
end

func topologicalSort(g)
    var visited = _dict()
    var stack = []
    var verts = g.vertices()
    var i = 0
    while i < _alen(verts)
        var v = verts[i]
        if not _dict_has(visited, v)
            _topoHelper(g, v, visited, stack)
        end
        i = i + 1
    end
    return arrayReverse(stack)
end

func _topoHelper(g, v, visited, stack)
    visited[v] = true
    var neighbors = g.neighbors(v)
    var i = 0
    while i < _alen(neighbors)
        var n = _str(neighbors[i])
        if not _dict_has(visited, n)
            _topoHelper(g, n, visited, stack)
        end
        i = i + 1
    end
    _push(stack, v)
end

func dijkstra(g, start)
    var dist = _dict()
    var visited = _dict()
    var h = new MinHeap()
    dist[_str(start)] = 0
    h.push([0, start])
    while not h.isEmpty()
        var top = h.pop()
        var d = top[0]
        var v = top[1]
        var key = _str(v)
        if _dict_has(visited, key)
            continue
        end
        visited[key] = true
        var neighbors = g.neighbors(v)
        var i = 0
        while i < _alen(neighbors)
            var edge = neighbors[i]
            var u = edge[0]
            var w = edge[1]
            var ukey = _str(u)
            var nd = d + w
            if not _dict_has(dist, ukey) or nd < dist[ukey]
                dist[ukey] = nd
                h.push([nd, u])
            end
            i = i + 1
        end
    end
    return dist
end

func bellmanFord(g, start)
    var verts = g.vertices()
    var n = _alen(verts)
    var dist = _dict()
    var prev = _dict()
    var i = 0
    while i < n
        dist[verts[i]] = 999999999
        i = i + 1
    end
    dist[_str(start)] = 0
    var iter = 0
    while iter < n - 1
        var changed = false
        i = 0
        while i < n
            var u = verts[i]
            if dist[u] < 999999999
                var neighbors = g.neighbors(u)
                var j = 0
                while j < _alen(neighbors)
                    var edge = neighbors[j]
                    var v = _str(edge[0])
                    var w = edge[1]
                    if dist[u] + w < dist[v]
                        dist[v] = dist[u] + w
                        prev[v] = u
                        changed = true
                    end
                    j = j + 1
                end
            end
            i = i + 1
        end
        if not changed
            iter = n
        end
        iter = iter + 1
    end
    i = 0
    while i < n
        var u = verts[i]
        if dist[u] < 999999999
            var neighbors = g.neighbors(u)
            var j = 0
            while j < _alen(neighbors)
                var edge = neighbors[j]
                var v = _str(edge[0])
                var w = edge[1]
                if dist[u] + w < dist[v]
                    dist[v] = 0 - 999999999
                end
                j = j + 1
            end
        end
        i = i + 1
    end
    return [dist, prev]
end

func floydWarshall(verts, edges)
    var n = _alen(verts)
    var idx = _dict()
    var i = 0
    while i < n
        idx[_str(verts[i])] = i
        i = i + 1
    end
    var dist = []
    i = 0
    while i < n
        var row = arrayFill(n, 999999999)
        _set(row, i, 0)
        _push(dist, row)
        i = i + 1
    end
    i = 0
    while i < _alen(edges)
        var e = edges[i]
        var u = idx[_str(e[0])]
        var v = idx[_str(e[1])]
        var w = e[2]
        _set(dist[u], v, w)
        i = i + 1
    end
    var k = 0
    while k < n
        i = 0
        while i < n
            var j = 0
            while j < n
                if dist[i][k] + dist[k][j] < dist[i][j]
                    _set(dist[i], j, dist[i][k] + dist[k][j])
                end
                j = j + 1
            end
            i = i + 1
        end
        k = k + 1
    end
    return dist
end

func _kruskalFind(parent, x)
    while parent[x] != x
        _set(parent, x, parent[parent[x]])
        x = parent[x]
    end
    return x
end

func kruskalMST(n, edges)
    var sorted = arrayCopy(edges)
    sortBy(sorted, func(e) return e[2] end)
    var parent = []
    var rank = []
    var i = 0
    while i < n
        _push(parent, i)
        _push(rank, 0)
        i = i + 1
    end
    var mst = []
    var total = 0
    i = 0
    while i < _alen(sorted)
        var e = sorted[i]
        var u = e[0]
        var v = e[1]
        var w = e[2]
        var pu = _kruskalFind(parent, u)
        var pv = _kruskalFind(parent, v)
        if pu != pv
            _push(mst, e)
            total = total + w
            if rank[pu] < rank[pv]
                _set(parent, pu, pv)
            else if rank[pu] > rank[pv]
                _set(parent, pv, pu)
            else
                _set(parent, pv, pu)
                _set(rank, pu, rank[pu] + 1)
            end
        end
        i = i + 1
    end
    return [mst, total]
end

func _sccDFS1(g, v, visited, order)
    visited[v] = true
    var nbrs = g.neighbors(v)
    var i = 0
    while i < _alen(nbrs)
        var n = _str(nbrs[i])
        if not _dict_has(visited, n)
            _sccDFS1(g, n, visited, order)
        end
        i = i + 1
    end
    _push(order, v)
end

func _sccDFS2(gRev, v, visited, component)
    visited[v] = true
    _push(component, v)
    var nbrs = gRev.neighbors(v)
    var i = 0
    while i < _alen(nbrs)
        var n = _str(nbrs[i])
        if not _dict_has(visited, n)
            _sccDFS2(gRev, n, visited, component)
        end
        i = i + 1
    end
end

func stronglyConnectedComponents(g)
    var verts = g.vertices()
    var visited = _dict()
    var order = []
    var i = 0
    while i < _alen(verts)
        var v = verts[i]
        if not _dict_has(visited, v)
            _sccDFS1(g, v, visited, order)
        end
        i = i + 1
    end
    var gRev = directedGraph()
    i = 0
    while i < _alen(verts)
        var u = verts[i]
        var nbrs = g.neighbors(u)
        var j = 0
        while j < _alen(nbrs)
            gRev.addEdge(_str(nbrs[j]), u)
            j = j + 1
        end
        i = i + 1
    end
    visited = _dict()
    var sccs = []
    i = _alen(order) - 1
    while i >= 0
        var v = order[i]
        if not _dict_has(visited, v)
            var component = []
            _sccDFS2(gRev, v, visited, component)
            _push(sccs, component)
        end
        i = i - 1
    end
    return sccs
end

func isBipartite(g)
    var verts = g.vertices()
    var color = _dict()
    var i = 0
    while i < _alen(verts)
        var v = verts[i]
        if not _dict_has(color, v)
            var q = deque()
            q.pushBack(v)
            color[v] = 0
            while not q.isEmpty()
                var u = q.popFront()
                var nbrs = g.neighbors(u)
                var j = 0
                while j < _alen(nbrs)
                    var n = _str(nbrs[j])
                    if not _dict_has(color, n)
                        color[n] = 1 - color[u]
                        q.pushBack(n)
                    else if color[n] == color[u]
                        return false
                    end
                    j = j + 1
                end
            end
        end
        i = i + 1
    end
    return true
end

func connectedComponents(g)
    var verts = g.vertices()
    var visited = _dict()
    var comps = []
    var i = 0
    while i < _alen(verts)
        var v = verts[i]
        if not _dict_has(visited, v)
            var comp = []
            var q = deque()
            q.pushBack(v)
            visited[v] = true
            while not q.isEmpty()
                var u = q.popFront()
                _push(comp, u)
                var nbrs = g.neighbors(u)
                var j = 0
                while j < _alen(nbrs)
                    var n = _str(nbrs[j])
                    if not _dict_has(visited, n)
                        visited[n] = true
                        q.pushBack(n)
                    end
                    j = j + 1
                end
            end
            _push(comps, comp)
        end
        i = i + 1
    end
    return comps
end
