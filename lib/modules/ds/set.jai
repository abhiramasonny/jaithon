class Set
    var _data

    func init(self)
        self._data = _dict()
    end

    func add(self, val)
        self._data[_str(val)] = val
        return self
    end

    func has(self, val)
        return _dict_has(self._data, _str(val))
    end

    func remove(self, val)
        _dict_del(self._data, _str(val))
        return self
    end

    func size(self)
        return _alen(_dict_keys(self._data))
    end

    func isEmpty(self)
        return self.size() == 0
    end

    func toArray(self)
        return _dict_values(self._data)
    end

    func clear(self)
        self._data = _dict()
        return self
    end

    func union(self, other)
        var result = new Set()
        var a = self.toArray()
        var i = 0
        while i < _alen(a)
            result.add(a[i])
            i = i + 1
        end
        var b = other.toArray()
        i = 0
        while i < _alen(b)
            result.add(b[i])
            i = i + 1
        end
        return result
    end

    func intersection(self, other)
        var result = new Set()
        var a = self.toArray()
        var i = 0
        while i < _alen(a)
            if other.has(a[i])
                result.add(a[i])
            end
            i = i + 1
        end
        return result
    end

    func difference(self, other)
        var result = new Set()
        var a = self.toArray()
        var i = 0
        while i < _alen(a)
            if not other.has(a[i])
                result.add(a[i])
            end
            i = i + 1
        end
        return result
    end

    func isSubsetOf(self, other)
        var a = self.toArray()
        var i = 0
        while i < _alen(a)
            if not other.has(a[i])
                return false
            end
            i = i + 1
        end
        return true
    end

    func equals(self, other)
        if self.size() != other.size()
            return false
        end
        return self.isSubsetOf(other)
    end
end

func set()
    return new Set()
end

func setFrom(arr)
    var s = new Set()
    var i = 0
    while i < _alen(arr)
        s.add(arr[i])
        i = i + 1
    end
    return s
end
