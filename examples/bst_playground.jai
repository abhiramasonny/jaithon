
import lib/modules/ds/btree
import lib/modules/core/random
import lib/modules/ds/array

func main()
    var tree = bst()
    var i = 0
    while i < 12
        var n = floor(rand() * 100)
        tree.insert(n)
        i = i + 1
    end
    var sorted = tree.toArray()
    print "BST inorder (sorted):"
    var line = ""
    var idx = 0
    while idx < _alen(sorted)
        line = line + str(sorted[idx]) + " "
        idx = idx + 1
    end
    print line
    print "Min: " + str(tree.minVal()) + "  Max: " + str(tree.maxVal())
end

main()
