class LRUCache
    var capacity
    var cache
    var order

    func init(self, cap)
        self.capacity = cap
        self.cache = _dict()
        self.order = []
    end

    func _promote(self, key)
        var newOrder = [key]
        var i = 0
        while i < _alen(self.order)
            if self.order[i] != key
                _push(newOrder, self.order[i])
            end
            i = i + 1
        end
        self.order = newOrder
    end

    func get(self, key)
        if not _dict_has(self.cache, key)
            return null
        end
        self._promote(key)
        return self.cache[key]
    end

    func put(self, key, value)
        if _dict_has(self.cache, key)
            self.cache[key] = value
            self._promote(key)
            return
        end
        if _alen(self.order) >= self.capacity
            var lru = self.order[_alen(self.order) - 1]
            _dict_del(self.cache, lru)
            self.order = arraySlice(self.order, 0, _alen(self.order) - 1)
        end
        self.cache[key] = value
        var newOrder = [key]
        var i = 0
        while i < _alen(self.order)
            _push(newOrder, self.order[i])
            i = i + 1
        end
        self.order = newOrder
    end

    func has(self, key)
        return _dict_has(self.cache, key)
    end

    func remove(self, key)
        if not _dict_has(self.cache, key)
            return false
        end
        _dict_del(self.cache, key)
        self.order = arrayFilter(self.order, func(k) return k != key end)
        return true
    end

    func size(self)
        return _alen(self.order)
    end

    func keys(self)
        return arrayCopy(self.order)
    end

    func values(self)
        var result = []
        var i = 0
        while i < _alen(self.order)
            _push(result, self.cache[self.order[i]])
            i = i + 1
        end
        return result
    end

    func clear(self)
        self.cache = _dict()
        self.order = []
    end
end

func lruCache(capacity)
    return new LRUCache(capacity)
end
