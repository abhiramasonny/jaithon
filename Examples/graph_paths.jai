#shortest path in an unweighted graph using BFS

import lib/modules/queue
import lib/modules/string

func findIndex(arr, value)
    var i = 0
    while i < _alen(arr)
        if _get(arr, i) == value
            return i
        end
        i = i + 1
    end
    return 0 - 1
end

func getNeighbors(graph, node)
    var i = 0
    while i < _alen(graph)
        var pair = _get(graph, i)   #[name, neighbors]
        if _get(pair, 0) == node
            return _get(pair, 1)
        end
        i = i + 1
    end
    return []
end

func reconstruct(visited, parents, start, goal)
    var path = []
    var cur = goal
    var building = true
    while building
        _push(path, cur)
        if cur == start
            building = false
        else
            var idx = findIndex(visited, cur)
            if idx < 0
                building = false
            else
                cur = _get(parents, idx)
            end
        end
    end
    var out = []
    var i = _alen(path) - 1
    while i >= 0
        _push(out, _get(path, i))
        i = i - 1
    end
    return out
end

func bfs(graph, start, goal)
    var q = queue()
    var visited = []
    var parents = []
    var path = []
    var found = false
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
        var i = 0
        while i < _alen(neighbors)
            var nb = _get(neighbors, i)
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
        var i = 0
        var acc = ""
        while i < _alen(path)
            acc = acc + _get(path, i)
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

func makeGraph()
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

func main()
    print "Starting BFS demo"
    var graph = makeGraph()
    bfs(graph, "A", "J")
end

main()
