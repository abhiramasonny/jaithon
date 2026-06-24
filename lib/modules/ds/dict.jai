func dict_new()
    return _dict()
end

func dict_set(d, key, value)
    d[key] = value
    return value
end

func dict_get(d, key)
    return d[key]
end

func dict_has(d, key)
    return _dict_has(d, key)
end

func dict_del(d, key)
    return _dict_del(d, key)
end

func dict_keys(d)
    return _dict_keys(d)
end

func dict_values(d)
    return _dict_values(d)
end

func dict_size(d)
    return len(_dict_keys(d))
end

func dict_items(d)
    var keys = _dict_keys(d)
    var result = []
    var i = 0
    while i < _alen(keys)
        var k = keys[i]
        _push(result, [k, d[k]])
        i = i + 1
    end
    return result
end

func dict_merge(a, b)
    var result = _dict()
    var ka = _dict_keys(a)
    var i = 0
    while i < _alen(ka)
        result[ka[i]] = a[ka[i]]
        i = i + 1
    end
    var kb = _dict_keys(b)
    i = 0
    while i < _alen(kb)
        result[kb[i]] = b[kb[i]]
        i = i + 1
    end
    return result
end

func dict_copy(d)
    var result = _dict()
    var keys = _dict_keys(d)
    var i = 0
    while i < _alen(keys)
        result[keys[i]] = d[keys[i]]
        i = i + 1
    end
    return result
end

func dict_from_pairs(pairs)
    var result = _dict()
    var i = 0
    while i < _alen(pairs)
        var pair = pairs[i]
        result[pair[0]] = pair[1]
        i = i + 1
    end
    return result
end
