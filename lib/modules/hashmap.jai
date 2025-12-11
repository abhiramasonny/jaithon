#simple key/value map (linear search)

class HashMap
    var entries
    
    func init(self)
        self.entries = []
    end
    
    func size(self)
        return _alen(self.entries)
    end
    
    func isEmpty(self)
        return self.size() eq 0
    end
    
    func set(self, key, value)
        var idx = mapFind(self.entries, key)
        if idx >= 0
            _set(self.entries, idx, [key, value])
        else
            _push(self.entries, [key, value])
        end
        return value
    end
    
    func get(self, key)
        var idx = mapFind(self.entries, key)
        if idx >= 0
            return _get(_get(self.entries, idx), 1)
        end
        return null
    end
    
    func has(self, key)
        return mapFind(self.entries, key) >= 0
    end
    
    func delete(self, key)
        var idx = mapFind(self.entries, key)
        if idx < 0
            return false
        end
        var newEntries = []
        var i = 0
        while i < _alen(self.entries)
            if not (i eq idx)
                _push(newEntries, _get(self.entries, i))
            end
            i = i + 1
        end
        self.entries = newEntries
        return true
    end
    
    func keys(self)
        var out = []
        var i = 0
        while i < _alen(self.entries)
            _push(out, _get(_get(self.entries, i), 0))
            i = i + 1
        end
        return out
    end
    
    func values(self)
        var out = []
        var i = 0
        while i < _alen(self.entries)
            _push(out, _get(_get(self.entries, i), 1))
            i = i + 1
        end
        return out
    end
end

func mapFind(entries, key)
    var i = 0
    while i < _alen(entries)
        if _get(_get(entries, i), 0) eq key
            return i
        end
        i = i + 1
    end
    return 0 - 1
end

func hashmap()
    return new HashMap()
end
