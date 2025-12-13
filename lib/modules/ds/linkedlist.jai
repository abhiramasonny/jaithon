#doubly linked list

class ListNode
    var value
    var next
    var prev
end

class LinkedList
    var head
    var tail
    var size

    func init(self)
        self.head = null
        self.tail = null
        self.size = 0
    end

func pushFront(self, val)
    var node = new ListNode()
    node.value = val
    node.next = self.head
    node.prev = null
    var h = self.head
    if not (h == null)
        h.prev = node
    else
        self.tail = node
    end
    self.head = node
    self.size = self.size + 1
        return self
    end

func pushBack(self, val)
    var node = new ListNode()
    node.value = val
    node.next = null
    node.prev = self.tail
    var t = self.tail
    if not (t == null)
        t.next = node
    else
        self.head = node
    end
    self.tail = node
    self.size = self.size + 1
        return self
    end

    func popFront(self)
        if self.head == null
            return null
        end
        var h = self.head
        var val = h.value
        self.head = h.next
        if self.head == null
            self.tail = null
        else
            var newHead = self.head
            newHead.prev = null
        end
        self.size = self.size - 1
        return val
    end

    func popBack(self)
        if self.tail == null
            return null
        end
        var t = self.tail
        var val = t.value
        self.tail = t.prev
        if self.tail == null
            self.head = null
        else
            var newTail = self.tail
            newTail.next = null
        end
        self.size = self.size - 1
        return val
    end

    func isEmpty(self)
        return self.size == 0
    end

    func len(self)
        return self.size
    end

    func peekFront(self)
        if self.head == null
            return null
        end
        return self.head.value
    end

    func peekBack(self)
        if self.tail == null
            return null
        end
        return self.tail.value
    end

    func forEach(self, f)
        var cur = self.head
        while not (cur == null)
            f(cur.value)
            cur = cur.next
        end
    end
end

func linkedlist()
    return new LinkedList()
end
