class MinHeap
    var _data

    func init(self)
        self._data = []
    end

    func size(self)
        return _alen(self._data)
    end

    func isEmpty(self)
        return self.size() == 0
    end

    func peek(self)
        if self.isEmpty()
            return null
        end
        return self._data[0]
    end

    func _swap(self, i, j)
        var tmp = self._data[i]
        _set(self._data, i, self._data[j])
        _set(self._data, j, tmp)
    end

    func _parent(self, i)
        return _int((i - 1) / 2)
    end

    func _left(self, i)
        return 2 * i + 1
    end

    func _right(self, i)
        return 2 * i + 2
    end

    func push(self, val)
        _push(self._data, val)
        var i = self.size() - 1
        while i > 0 and self._data[i] < self._data[self._parent(i)]
            self._swap(i, self._parent(i))
            i = self._parent(i)
        end
        return self
    end

    func pop(self)
        if self.isEmpty()
            return null
        end
        var top = self._data[0]
        var last = _alen(self._data) - 1
        _set(self._data, 0, self._data[last])
        _pop(self._data)
        self._heapifyDown(0)
        return top
    end

    func _heapifyDown(self, i)
        var n = self.size()
        var smallest = i
        var l = self._left(i)
        var r = self._right(i)
        if l < n and self._data[l] < self._data[smallest]
            smallest = l
        end
        if r < n and self._data[r] < self._data[smallest]
            smallest = r
        end
        if smallest != i
            self._swap(i, smallest)
            self._heapifyDown(smallest)
        end
    end

    func toSortedArray(self)
        var result = []
        var copy = new MinHeap()
        var i = 0
        while i < _alen(self._data)
            copy.push(self._data[i])
            i = i + 1
        end
        while not copy.isEmpty()
            _push(result, copy.pop())
        end
        return result
    end
end

class MaxHeap
    var _data

    func init(self)
        self._data = []
    end

    func size(self)
        return _alen(self._data)
    end

    func isEmpty(self)
        return self.size() == 0
    end

    func peek(self)
        if self.isEmpty()
            return null
        end
        return self._data[0]
    end

    func _swap(self, i, j)
        var tmp = self._data[i]
        _set(self._data, i, self._data[j])
        _set(self._data, j, tmp)
    end

    func _parent(self, i)
        return _int((i - 1) / 2)
    end

    func _left(self, i)
        return 2 * i + 1
    end

    func _right(self, i)
        return 2 * i + 2
    end

    func push(self, val)
        _push(self._data, val)
        var i = self.size() - 1
        while i > 0 and self._data[i] > self._data[self._parent(i)]
            self._swap(i, self._parent(i))
            i = self._parent(i)
        end
        return self
    end

    func pop(self)
        if self.isEmpty()
            return null
        end
        var top = self._data[0]
        var last = _alen(self._data) - 1
        _set(self._data, 0, self._data[last])
        _pop(self._data)
        self._heapifyDown(0)
        return top
    end

    func _heapifyDown(self, i)
        var n = self.size()
        var largest = i
        var l = self._left(i)
        var r = self._right(i)
        if l < n and self._data[l] > self._data[largest]
            largest = l
        end
        if r < n and self._data[r] > self._data[largest]
            largest = r
        end
        if largest != i
            self._swap(i, largest)
            self._heapifyDown(largest)
        end
    end
end

func minHeap()
    return new MinHeap()
end

func maxHeap()
    return new MaxHeap()
end

func heapSort(arr)
    var h = new MinHeap()
    var i = 0
    while i < _alen(arr)
        h.push(arr[i])
        i = i + 1
    end
    var result = []
    while not h.isEmpty()
        _push(result, h.pop())
    end
    return result
end

func nSmallest(arr, n)
    var h = new MaxHeap()
    var i = 0
    while i < _alen(arr)
        h.push(arr[i])
        if h.size() > n
            h.pop()
        end
        i = i + 1
    end
    return h._data
end

func nLargest(arr, n)
    var h = new MinHeap()
    var i = 0
    while i < _alen(arr)
        h.push(arr[i])
        if h.size() > n
            h.pop()
        end
        i = i + 1
    end
    return h._data
end
