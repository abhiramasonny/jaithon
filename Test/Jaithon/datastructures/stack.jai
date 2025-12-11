class Stack
    var items
    var size
    
    func init(self)
        self.items = []
        self.size = 0
    end
    
    func push(self, value)
        _push(self.items, value)
        self.size = self.size + 1
    end
    
    func pop(self)
        if self.size eq 0
            print "Stack is empty!"
            return 0
        end
        self.size = self.size - 1
        return _pop(self.items)
    end
    
    func peek(self)
        if self.size eq 0
            return 0
        end
        return _get(self.items, self.size - 1)
    end
    
    func isEmpty(self)
        return self.size eq 0
    end
    
    func show(self)
        print "Stack contents:"
        var i = 0
        while i < self.size
            print _get(self.items, i)
            i = i + 1
        end
    end
end

var s = new Stack()
s.push(10)
s.push(20)
s.push(30)
s.show()
print "Top: " + str(s.peek())
print "Popped: " + str(s.pop())
s.show()
