import lib/modules/ds/queue
import lib/modules/core/string

public int findIndex(var arr, var value)
    int i = 0
    while i < _alen(arr)
        if arr[i] == value
            return i
        end
        i = i + 1
    end
    return 0 - 1
end

public var getNeighbors(var graph, var node)
    int i = 0
    while i < _alen(graph)
        var pair = graph[i]
        if pair[0] == node
            return pair[1]
        end
        i = i + 1
    end
    return []
end

public var reconstruct(var visited, var parents, var start, var goal)
    var path = []
    var cur = goal
    bool building = true
    while building
        _push(path, cur)
        if cur == start
            building = false
        else
            int idx = findIndex(visited, cur)
            if idx < 0
                building = false
            else
                cur = parents[idx]
            end
        end
    end
    var out = []
    int i = _alen(path) - 1
    while i >= 0
        _push(out, path[i])
        i = i - 1
    end
    return out
end

public var bfs(var graph, var start, var goal)
    var q = queue()
    var visited = []
    var parents = []
    var path = []
    bool found = false
    q.enqueue(start)
    _push(visited, start)
    _push(parents, null)

    while not q.isEmpty()
        var node = q.dequeue()
        print "Visiting " + node
        if node == goal
            print "found goal"
            path = reconstruct(visited, parents, start, goal)
            found = true
        end

        var neighbors = getNeighbors(graph, node)
        int i = 0
        while i < _alen(neighbors)
            var nb = neighbors[i]
            if findIndex(visited, nb) < 0
                q.enqueue(nb)
                _push(visited, nb)
                _push(parents, node)
            end
            i = i + 1
        end
    end

    if found
        print "Shortest path " + start + "->" + goal + ":"
        int i = 0
        string acc = ""
        while i < _alen(path)
            acc = acc + path[i]
            if i < _alen(path) - 1
                acc = acc + " -> "
            end
            i = i + 1
        end
        print acc
        return path
    end

    print "No path found"
    return []
end

public var makeGraph()
    var g = []
    _push(g, ["A", ["B", "C"]])
    _push(g, ["B", ["D", "E"]])
    _push(g, ["C", ["F"]])
    _push(g, ["D", ["G"]])
    _push(g, ["E", ["G", "H"]])
    _push(g, ["F", ["I"]])
    _push(g, ["G", ["J"]])
    _push(g, ["H", ["J"]])
    _push(g, ["I", ["J"]])
    return g
end

public static void main()
    print "Starting BFS demo"
    var graph = makeGraph()
    bfs(graph, "A", "J")
end

main()
