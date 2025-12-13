class Queue
    var items
    var front
    var rear

    func init(self)
        self.items = []
        self.front = 0
        self.rear = 0
    end

    func enqueue(self, value)
        _push(self.items, value)
        self.rear = self.rear + 1
    end

    func dequeue(self)
        if self.front == self.rear
            print "Queue is empty!"
            return 0
        end
        var val = _get(self.items, self.front)
        self.front = self.front + 1
        return val
    end

    func peek(self)
        if self.front == self.rear
            return 0
        end
        return _get(self.items, self.front)
    end

    func size(self)
        return self.rear - self.front
    end

    func isEmpty(self)
        return self.front == self.rear
    end

    func show(self)
        print "Queue contents (front to rear):"
        var i = self.front
        while i < self.rear
            print _get(self.items, i)
            i = i + 1
        end
    end
end

var q = new Queue()
print "Enqueueing: 1, 2, 3, 4, 5"
q.enqueue(1)
q.enqueue(2)
q.enqueue(3)
q.enqueue(4)
q.enqueue(5)
q.show()

print "Dequeue:"
print q.dequeue()
print "Dequeue:"
print q.dequeue()

q.show()
print "Front element:"
print q.peek()
print "Queue size:"
print q.size()
