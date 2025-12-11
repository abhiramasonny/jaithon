class Vector
    var data
    
    func init(self)
        self.data = []
    end
    
    func fromArray(arr)
        var v = new Vector()
        v.data = arrayCopy(arr)
        return v
    end
    
    func len(self)
        return _alen(self.data)
    end
    
    func get(self, idx)
        return _get(self.data, idx)
    end
    
    func set(self, idx, val)
        return _set(self.data, idx, val)
    end
    
    func push(self, val)
        _push(self.data, val)
        return self
    end
    
    func pop(self)
        if _alen(self.data) == 0
            return null
        end
        return _pop(self.data)
    end
    
    func clear(self)
        self.data = []
        return self
    end
    
    func toArray(self)
        return arrayCopy(self.data)
    end
end

func vector()
    return new Vector()
end
