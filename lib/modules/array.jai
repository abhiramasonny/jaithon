# Jaithon Array - Dynamic array utilities (using native arrays)

func arrayNew(size)
    return _array(size)
end

func arrayLen(arr)
    return _alen(arr)
end

func arrayGet(arr, idx)
    return _get(arr, idx)
end

func arraySet(arr, idx, val)
    return _set(arr, idx, val)
end

func arrayPush(arr, val)
    return _push(arr, val)
end

func arrayPop(arr)
    return _pop(arr)
end

func arrayFill(size, val)
    var arr = []
    var i = 0
    while i < size
        _push(arr, val)
        i = i + 1
    end
    return arr
end

func arrayRange(start, stop)
    var arr = []
    var i = start
    while i < stop
        _push(arr, i)
        i = i + 1
    end
    return arr
end

func arraySum(arr)
    var total = 0
    var i = 0
    while i < _alen(arr)
        total = total + arr[i]
        i = i + 1
    end
    return total
end

func arrayMax(arr)
    if _alen(arr) eq 0
        return 0
    end
    var m = arr[0]
    var i = 1
    while i < _alen(arr)
        if arr[i] > m
            m = arr[i]
        end
        i = i + 1
    end
    return m
end

func arrayMin(arr)
    if _alen(arr) eq 0
        return 0
    end
    var m = arr[0]
    var i = 1
    while i < _alen(arr)
        if arr[i] < m
            m = arr[i]
        end
        i = i + 1
    end
    return m
end

func arrayReverse(arr)
    var result = []
    var i = _alen(arr) - 1
    while i >= 0
        _push(result, arr[i])
        i = i - 1
    end
    return result
end

func arrayContains(arr, val)
    var i = 0
    while i < _alen(arr)
        if arr[i] eq val
            return true
        end
        i = i + 1
    end
    return false
end

func arrayIndexOf(arr, val)
    var i = 0
    while i < _alen(arr)
        if arr[i] eq val
            return i
        end
        i = i + 1
    end
    return 0 - 1
end

func arrayCopy(arr)
    var result = []
    var i = 0
    while i < _alen(arr)
        _push(result, arr[i])
        i = i + 1
    end
    return result
end

func arraySlice(arr, start, stop)
    var result = []
    var i = start
    while i < stop and i < _alen(arr)
        _push(result, arr[i])
        i = i + 1
    end
    return result
end

func arrayConcat(arr1, arr2)
    var result = arrayCopy(arr1)
    var i = 0
    while i < _alen(arr2)
        _push(result, arr2[i])
        i = i + 1
    end
    return result
end

func arrayMap(arr, f)
    var result = []
    var i = 0
    while i < _alen(arr)
        _push(result, f(arr[i]))
        i = i + 1
    end
    return result
end

func arrayFilter(arr, pred)
    var result = []
    var i = 0
    while i < _alen(arr)
        if pred(arr[i])
            _push(result, arr[i])
        end
        i = i + 1
    end
    return result
end

func arrayReduce(arr, init, f)
    var acc = init
    var i = 0
    while i < _alen(arr)
        acc = f(acc, arr[i])
        i = i + 1
    end
    return acc
end

func arrayForEach(arr, f)
    var i = 0
    while i < _alen(arr)
        f(arr[i])
        i = i + 1
    end
end
