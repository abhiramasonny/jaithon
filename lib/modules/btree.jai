#bin tree

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
        if self.root == null
            return null
        end
        var cur = self.root
        while not (cur.left == null)
            cur = cur.left
        end
        return cur.value
    end
    
    func maxVal(self)
        if self.root == null
            return null
        end
        var cur = self.root
        while not (cur.right == null)
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
    if node == null
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
    if node == null
        return false
    end
    if val == node.value
        return true
    end
    if val < node.value
        return bstContains(node.left, val)
    end
    return bstContains(node.right, val)
end

func bstInorder(node, out)
    if node == null
        return
    end
    bstInorder(node.left, out)
    _push(out, node.value)
    bstInorder(node.right, out)
end

func bst()
    return new BST()
end
