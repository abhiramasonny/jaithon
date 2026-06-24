func identity(x)
    return x
end

func constant(x)
    func f(ignored)
        return x
    end
    return f
end

func negate(f)
    func negated(x)
        return not f(x)
    end
    return negated
end

func compose(f, g)
    func composed(x)
        return f(g(x))
    end
    return composed
end

func pipe(fns)
    func piped(x)
        var result = x
        var i = 0
        while i < _alen(fns)
            result = fns[i](result)
            i = i + 1
        end
        return result
    end
    return piped
end

func memoize(f)
    var cache = _dict()
    func memoized(x)
        var key = _str(x)
        if _dict_has(cache, key)
            return cache[key]
        end
        var result = f(x)
        cache[key] = result
        return result
    end
    return memoized
end

func once(f)
    var called = false
    var result = null
    func onced(*args)
        if not called
            result = f(args)
            called = true
        end
        return result
    end
    return onced
end

func times(n, f)
    var i = 0
    while i < n
        f(i)
        i = i + 1
    end
end

func repeat_fn(n, f, init)
    var result = init
    var i = 0
    while i < n
        result = f(result)
        i = i + 1
    end
    return result
end

func apply(f, args)
    return f(args)
end

func flip(f)
    func flipped(a, b)
        return f(b, a)
    end
    return flipped
end

func mapPairs(pairs, f)
    var result = []
    var i = 0
    while i < _alen(pairs)
        _push(result, f(pairs[i][0], pairs[i][1]))
        i = i + 1
    end
    return result
end

func groupBy(arr, keyFn)
    var groups = _dict()
    var i = 0
    while i < _alen(arr)
        var item = arr[i]
        var key = _str(keyFn(item))
        if not _dict_has(groups, key)
            groups[key] = []
        end
        _push(groups[key], item)
        i = i + 1
    end
    return groups
end

func countBy(arr, keyFn)
    var counts = _dict()
    var i = 0
    while i < _alen(arr)
        var key = _str(keyFn(arr[i]))
        if _dict_has(counts, key)
            counts[key] = counts[key] + 1
        else
            counts[key] = 1
        end
        i = i + 1
    end
    return counts
end

func flatten(arr)
    var result = []
    var i = 0
    while i < _alen(arr)
        var item = arr[i]
        if _type(item) == "array"
            var j = 0
            while j < _alen(item)
                _push(result, item[j])
                j = j + 1
            end
        else
            _push(result, item)
        end
        i = i + 1
    end
    return result
end

func flatMap(arr, f)
    return flatten(arrayMap(arr, f))
end

func zip(a, b)
    var result = []
    var n = min(_alen(a), _alen(b))
    var i = 0
    while i < n
        _push(result, [a[i], b[i]])
        i = i + 1
    end
    return result
end

func unzip(pairs)
    var a = []
    var b = []
    var i = 0
    while i < _alen(pairs)
        _push(a, pairs[i][0])
        _push(b, pairs[i][1])
        i = i + 1
    end
    return [a, b]
end

func enumerate(arr)
    var result = []
    var i = 0
    while i < _alen(arr)
        _push(result, [i, arr[i]])
        i = i + 1
    end
    return result
end

func range(start, stop, step)
    var result = []
    if step == 0
        return result
    end
    var i = start
    if step > 0
        while i < stop
            _push(result, i)
            i = i + step
        end
    else
        while i > stop
            _push(result, i)
            i = i + step
        end
    end
    return result
end

func take(arr, n)
    return arraySlice(arr, 0, n)
end

func drop(arr, n)
    return arraySlice(arr, n, _alen(arr))
end

func takeWhile(arr, pred)
    var result = []
    var i = 0
    while i < _alen(arr) and pred(arr[i])
        _push(result, arr[i])
        i = i + 1
    end
    return result
end

func dropWhile(arr, pred)
    var i = 0
    while i < _alen(arr) and pred(arr[i])
        i = i + 1
    end
    return arraySlice(arr, i, _alen(arr))
end

func partition(arr, pred)
    var yes = []
    var no = []
    var i = 0
    while i < _alen(arr)
        if pred(arr[i])
            _push(yes, arr[i])
        else
            _push(no, arr[i])
        end
        i = i + 1
    end
    return [yes, no]
end

func unique(arr)
    var s = setFrom(arr)
    return s.toArray()
end

func frequencies(arr)
    var counts = _dict()
    var i = 0
    while i < _alen(arr)
        var key = _str(arr[i])
        if _dict_has(counts, key)
            counts[key] = counts[key] + 1
        else
            counts[key] = 1
        end
        i = i + 1
    end
    return counts
end
