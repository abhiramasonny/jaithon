class Deque
    var _data
    var _head
    var _tail
    var _size

    func init(self)
        self._data = []
        self._size = 0
    end

    func size(self)
        return self._size
    end

    func isEmpty(self)
        return self._size == 0
    end

    func pushBack(self, val)
        _push(self._data, val)
        self._size = self._size + 1
        return self
    end

    func pushFront(self, val)
        var newData = [val]
        var i = 0
        while i < _alen(self._data)
            _push(newData, self._data[i])
            i = i + 1
        end
        self._data = newData
        self._size = self._size + 1
        return self
    end

    func popBack(self)
        if self.isEmpty()
            return null
        end
        var val = self._data[self._size - 1]
        _pop(self._data)
        self._size = self._size - 1
        return val
    end

    func popFront(self)
        if self.isEmpty()
            return null
        end
        var val = self._data[0]
        var newData = []
        var i = 1
        while i < _alen(self._data)
            _push(newData, self._data[i])
            i = i + 1
        end
        self._data = newData
        self._size = self._size - 1
        return val
    end

    func peekFront(self)
        if self.isEmpty()
            return null
        end
        return self._data[0]
    end

    func peekBack(self)
        if self.isEmpty()
            return null
        end
        return self._data[self._size - 1]
    end

    func toArray(self)
        return arrayCopy(self._data)
    end

    func clear(self)
        self._data = []
        self._size = 0
        return self
    end
end

func deque()
    return new Deque()
end

func dequeFrom(arr)
    var d = new Deque()
    var i = 0
    while i < _alen(arr)
        d.pushBack(arr[i])
        i = i + 1
    end
    return d
end
