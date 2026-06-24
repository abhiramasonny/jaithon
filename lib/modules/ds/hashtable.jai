# Hash table bootstrapped on the _buf memory primitive.
#
# Open addressing + linear probing, djb2 string hash (via _ord), with
# load-factor resize. Average O(1) get/set/has — a real improvement over the
# linear-scan HashMap class in hashmap.jai, and a demonstration that a hash map
# can be built entirely in Jaithon on top of fixed memory buffers.
#
# String keys. (delete is omitted — open addressing needs tombstones.)

func _htHash(s, cap)
    var h = 5381
    var i = 0
    var n = len(s)
    while i < n
        h = (h * 33 + _ord(charAt(s, i))) % 2147483647
        i = i + 1
    end
    if h < 0
        h = 0 - h
    end
    return h % cap
end

func _htInitUsed(buf, cap)
    var i = 0
    while i < cap
        _bset(buf, i, false)
        i = i + 1
    end
    return null
end

func htNew()
    var m = _dict()
    var cap = 8
    m["cap"] = cap
    m["size"] = 0
    m["keys"] = _buf(cap)
    m["vals"] = _buf(cap)
    m["used"] = _buf(cap)
    _htInitUsed(m["used"], cap)
    return m
end

# returns slot index for key: existing slot if present, else first empty slot
func _htFindSlot(keysBuf, usedBuf, cap, key)
    var idx = _htHash(key, cap)
    var probes = 0
    while probes < cap
        if _bget(usedBuf, idx) == false
            return idx
        end
        if _bget(keysBuf, idx) == key
            return idx
        end
        idx = (idx + 1) % cap
        probes = probes + 1
    end
    return 0 - 1
end

func _htResize(m)
    var oldKeys = m["keys"]
    var oldVals = m["vals"]
    var oldUsed = m["used"]
    var oldCap = m["cap"]
    var ncap = oldCap * 2
    m["cap"] = ncap
    m["keys"] = _buf(ncap)
    m["vals"] = _buf(ncap)
    m["used"] = _buf(ncap)
    _htInitUsed(m["used"], ncap)
    m["size"] = 0
    var i = 0
    while i < oldCap
        if _bget(oldUsed, i) == true
            htSet(m, _bget(oldKeys, i), _bget(oldVals, i))
        end
        i = i + 1
    end
    return null
end

func htSet(m, key, val)
    if (m["size"] + 1) * 10 > m["cap"] * 7
        _htResize(m)
    end
    var idx = _htFindSlot(m["keys"], m["used"], m["cap"], key)
    if _bget(m["used"], idx) == false
        _bset(m["used"], idx, true)
        _bset(m["keys"], idx, key)
        m["size"] = m["size"] + 1
    end
    _bset(m["vals"], idx, val)
    return val
end

func htGet(m, key)
    var idx = _htFindSlot(m["keys"], m["used"], m["cap"], key)
    if idx < 0
        return null
    end
    if _bget(m["used"], idx) == true
        return _bget(m["vals"], idx)
    end
    return null
end

func htHas(m, key)
    var idx = _htFindSlot(m["keys"], m["used"], m["cap"], key)
    if idx < 0
        return false
    end
    return _bget(m["used"], idx) == true
end

func htSize(m)
    return m["size"]
end
