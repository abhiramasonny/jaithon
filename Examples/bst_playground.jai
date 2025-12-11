#inserts random numbers and prints sorted order

import lib/modules/btree
import lib/modules/random
import lib/modules/array

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
