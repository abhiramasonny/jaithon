class Queue
    var items
    var front

    func init(self)
        self.items = []
        self.front = 0
    end

    func enqueue(self, val)
        _push(self.items, val)
        return val
    end

    func dequeue(self)
        if self.front >= _alen(self.items)
            return null
        end
        var val = _get(self.items, self.front)
        self.front = self.front + 1
        return val
    end

    func peek(self)
        if self.front >= _alen(self.items)
            return null
        end
        return _get(self.items, self.front)
    end

    func size(self)
        return _alen(self.items) - self.front
    end

    func isEmpty(self)
        return self.size() <= 0
    end

    func clear(self)
        self.items = []
        self.front = 0
        return self
    end
end

func queue()
    return new Queue()
end
