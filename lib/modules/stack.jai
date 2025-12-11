class Stack
    var items
    
    func init(self)
        self.items = []
    end
    
    func push(self, val)
        _push(self.items, val)
        return val
    end
    
    func pop(self)
        if _alen(self.items) eq 0
            return null
        end
        return _pop(self.items)
    end
    
    func peek(self)
        if _alen(self.items) eq 0
            return null
        end
        return _get(self.items, _alen(self.items) - 1)
    end
    
    func isEmpty(self)
        return _alen(self.items) eq 0
    end
    
    func size(self)
        return _alen(self.items)
    end
    
    func clear(self)
        self.items = []
        return self
    end
end

func stack()
    return new Stack()
end
