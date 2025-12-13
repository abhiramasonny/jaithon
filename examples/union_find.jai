
import ds/vector

class uf
    var parent
    var rank

    func init(self, n)
        self.parent = vector()
        self.rank = vector()
        var i = 0
        while i < n
            self.parent.push(i)
            self.rank.push(0)
            i = i + 1
        end
    end

    func find(self, x)
        var p = self.parent.get(x)
        if p == x
            return x
        end
        var root = self.find(p)
        self.parent.set(x, root)
        return root
    end

    func union(self, a, b)
        var ra = self.find(a)
        var rb = self.find(b)
        if ra == rb
            return
        end

        var rankA = self.rank.get(ra)
        var rankB = self.rank.get(rb)
        if rankA < rankB
            self.parent.set(ra, rb)
        else
            if rankA > rankB
                self.parent.set(rb, ra)
            else
                self.parent.set(rb, ra)
                self.rank.set(ra, rankA + 1)
            end
        end
    end

    func connected(self, a, b)
        return self.find(a) == self.find(b)
    end
end

func showConnected(uf, a, b)
    if uf.connected(a, b)
        print str(a) + " and " + str(b) + " are connected"
    else
        print str(a) + " and " + str(b) + " are NOT connected"
    end
end

func main()
    var uf = new uf(10)

    uf.union(1, 2)
    uf.union(2, 3)
    uf.union(4, 5)
    uf.union(6, 7)
    uf.union(7, 8)
    uf.union(3, 8)

    showConnected(uf, 1, 8)
    showConnected(uf, 1, 4)

    uf.union(4, 8)
    print "After union(4, 8):"
    showConnected(uf, 1, 5)
end

main()
