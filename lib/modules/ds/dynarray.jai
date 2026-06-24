# Dynamic (growable) array bootstrapped on the fixed-size _buf memory primitive.
#
# This is the answer to "_push shouldn't be a C builtin": here the growth and
# amortized-doubling logic lives entirely in Jaithon. C only provides the
# irreducible fixed-size memory primitive:
#   _buf(n)        -> buffer of n slots
#   _bget(b, i)    -> read slot
#   _bset(b, i, v) -> write slot
#   _blen(b)       -> capacity
#
# The struct is a small dict {buf, len}. (The native [..]/a[i] array stays as
# the perf/syntax primitive; this module shows the same thing built up from
# memory cells.)

func darrNew()
    var d = _dict()
    d["buf"] = _buf(4)
    d["len"] = 0
    return d
end

func darrLen(d)
    return d["len"]
end

func darrGet(d, i)
    if i < 0 or i >= d["len"]
        return null
    end
    return _bget(d["buf"], i)
end

func darrSet(d, i, v)
    if i < 0 or i >= d["len"]
        return null
    end
    _bset(d["buf"], i, v)
    return v
end

# grow to at least `need` capacity by doubling — pure Jaithon
func _darrGrow(d, need)
    var cap = _blen(d["buf"])
    if need <= cap
        return null
    end
    var ncap = cap
    while ncap < need
        ncap = ncap * 2
    end
    var nb = _buf(ncap)
    var i = 0
    var n = d["len"]
    while i < n
        _bset(nb, i, _bget(d["buf"], i))
        i = i + 1
    end
    d["buf"] = nb
    return null
end

func darrPush(d, v)
    var n = d["len"]
    _darrGrow(d, n + 1)
    _bset(d["buf"], n, v)
    d["len"] = n + 1
    return v
end

func darrPop(d)
    var n = d["len"]
    if n == 0
        return null
    end
    var v = _bget(d["buf"], n - 1)
    d["len"] = n - 1
    return v
end

# convert to a native array (for printing / interop)
func darrToArray(d)
    var out = []
    var i = 0
    var n = d["len"]
    while i < n
        _push(out, _bget(d["buf"], i))
        i = i + 1
    end
    return out
end
