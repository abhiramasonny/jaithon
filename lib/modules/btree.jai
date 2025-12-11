# Jaithon Binary Search Tree (numbers)

class BNode
    var value
    var left
    var right
end

class BST
    var root
    
    func init(self)
        self.root = null
    end
    
    func insert(self, val)
        self.root = bstInsert(self.root, val)
        return self
    end
    
    func contains(self, val)
        return bstContains(self.root, val)
    end
    
    func minVal(self)
        if self.root eq null
            return null
        end
        var cur = self.root
        while not (cur.left eq null)
            cur = cur.left
        end
        return cur.value
    end
    
    func maxVal(self)
        if self.root eq null
            return null
        end
        var cur = self.root
        while not (cur.right eq null)
            cur = cur.right
        end
        return cur.value
    end
    
    func toArray(self)
        var out = []
        bstInorder(self.root, out)
        return out
    end
end

func bstInsert(node, val)
    if node eq null
        var n = new BNode()
        n.value = val
        n.left = null
        n.right = null
        return n
    end
    if val < node.value
        node.left = bstInsert(node.left, val)
    else
        node.right = bstInsert(node.right, val)
    end
    return node
end

func bstContains(node, val)
    if node eq null
        return false
    end
    if val eq node.value
        return true
    end
    if val < node.value
        return bstContains(node.left, val)
    end
    return bstContains(node.right, val)
end

func bstInorder(node, out)
    if node eq null
        return
    end
    bstInorder(node.left, out)
    _push(out, node.value)
    bstInorder(node.right, out)
end

func bst()
    return new BST()
end
