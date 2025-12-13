
import ds/queue
import ds/hashmap
import ds/vector

func addNode(graph, indegree, node)
    if graph.get(node) == null
        graph.set(node, vector())
    end
    if indegree.get(node) == null
        indegree.set(node, 0)
    end
end

func addEdge(graph, indegree, src, dst)
    addNode(graph, indegree, src)
    addNode(graph, indegree, dst)
    graph.get(src).push(dst)
    var cur = indegree.get(dst)
    if cur == null
        cur = 0
    end
    indegree.set(dst, cur + 1)
end

func topoSort(graph, indegree)
    var q = queue()
    var keys = graph.keys()
    var i = 0
    while i < _alen(keys)
        var k = _get(keys, i)
        var deg = indegree.get(k)
        if deg == null
            deg = 0
        end
        if deg == 0
            q.enqueue(k)
        end
        i = i + 1
    end

    var order = []
    while not q.isEmpty()
        var node = q.dequeue()
        _push(order, node)

        var neighbors = graph.get(node)
        if not (neighbors == null)
            var j = 0
            while j < neighbors.len()
                var nb = neighbors.get(j)
                var d = indegree.get(nb)
                if d == null
                    d = 0
                end
                var newDeg = d - 1
                indegree.set(nb, newDeg)
                if newDeg == 0
                    q.enqueue(nb)
                end
                j = j + 1
            end
        end
    end
    return order
end

func main()
    var graph = hashmap()
    var indegree = hashmap()

    addEdge(graph, indegree, "Design", "Implement")
    addEdge(graph, indegree, "Design", "WriteTests")
    addEdge(graph, indegree, "Implement", "Deploy")
    addEdge(graph, indegree, "Implement", "PerfTune")
    addEdge(graph, indegree, "WriteTests", "Deploy")
    addEdge(graph, indegree, "PerfTune", "Deploy")

    var order = topoSort(graph, indegree)
    print "One valid schedule:"
    var i = 0
    while i < _alen(order)
        print str(i + 1) + ". " + _get(order, i)
        i = i + 1
    end
end

main()
