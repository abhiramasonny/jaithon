class Node
    var data
    var next
    
    func init(self, data)
        self.data = data
        self.next = 0
    end
end

class LinkedList
    var head
    var size
    
    func init(self)
        self.head = 0
        self.size = 0
    end
    
    func append(self, data)
        var node = new Node(data)
        if self.head eq 0
            self.head = node
        else
            var current = self.head
            while current.next eq 0 eq 0
                current = current.next
            end
            current.next = node
        end
        self.size = self.size + 1
    end
    
    func prepend(self, data)
        var node = new Node(data)
        node.next = self.head
        self.head = node
        self.size = self.size + 1
    end
    
    func get(self, index)
        if index < 0 or index >= self.size
            print "Index out of bounds"
            return 0
        end
        var current = self.head
        var i = 0
        while i < index
            current = current.next
            i = i + 1
        end
        return current.data
    end
    
    func show(self)
        print "LinkedList contents:"
        var current = self.head
        while not (current eq 0)
            print current.data
            current = current.next
        end
    end
    
    func length(self)
        return self.size
    end
end

var list = new LinkedList()
print "Prepending 10, 20, 30"
list.prepend(30)
list.prepend(20)
list.prepend(10)
list.show()

print "Length:"
print list.length()

print "Element at index 1:"
print list.get(1)
