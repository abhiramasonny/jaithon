func insertionSort(arr)
    var n = _alen(arr)
    var i = 1
    while i < n
        var key = arr[i]
        var j = i - 1
        while j >= 0 and arr[j] > key
            _set(arr, j + 1, arr[j])
            j = j - 1
        end
        _set(arr, j + 1, key)
        i = i + 1
    end
    return arr
end

func _merge(arr, lo, mid, hi)
    var left = []
    var right = []
    var i = lo
    while i <= mid
        _push(left, arr[i])
        i = i + 1
    end
    i = mid + 1
    while i <= hi
        _push(right, arr[i])
        i = i + 1
    end
    var li = 0
    var ri = 0
    var k = lo
    while li < _alen(left) and ri < _alen(right)
        if left[li] <= right[ri]
            _set(arr, k, left[li])
            li = li + 1
        else
            _set(arr, k, right[ri])
            ri = ri + 1
        end
        k = k + 1
    end
    while li < _alen(left)
        _set(arr, k, left[li])
        li = li + 1
        k = k + 1
    end
    while ri < _alen(right)
        _set(arr, k, right[ri])
        ri = ri + 1
        k = k + 1
    end
end

func _mergeSortHelper(arr, lo, hi)
    if lo >= hi
        return arr
    end
    var mid = lo + _int((hi - lo) / 2)
    _mergeSortHelper(arr, lo, mid)
    _mergeSortHelper(arr, mid + 1, hi)
    _merge(arr, lo, mid, hi)
    return arr
end

func mergeSort(arr)
    var n = _alen(arr)
    if n <= 1
        return arr
    end
    _mergeSortHelper(arr, 0, n - 1)
    return arr
end

func _partition(arr, lo, hi)
    var pivot = arr[hi]
    var i = lo - 1
    var j = lo
    while j < hi
        if arr[j] <= pivot
            i = i + 1
            var tmp = arr[i]
            _set(arr, i, arr[j])
            _set(arr, j, tmp)
        end
        j = j + 1
    end
    var tmp = arr[i + 1]
    _set(arr, i + 1, arr[hi])
    _set(arr, hi, tmp)
    return i + 1
end

func _quickSortHelper(arr, lo, hi)
    if lo >= hi
        return
    end
    var pi = _partition(arr, lo, hi)
    _quickSortHelper(arr, lo, pi - 1)
    _quickSortHelper(arr, pi + 1, hi)
end

func quickSort(arr)
    var n = _alen(arr)
    if n <= 1
        return arr
    end
    _quickSortHelper(arr, 0, n - 1)
    return arr
end

func sort(arr)
    return mergeSort(arr)
end

func sortBy(arr, keyFn)
    var keys = []
    var i = 0
    while i < _alen(arr)
        _push(keys, keyFn(arr[i]))
        i = i + 1
    end
    var n = _alen(arr)
    i = 1
    while i < n
        var ki = keys[i]
        var vi = arr[i]
        var j = i - 1
        while j >= 0 and keys[j] > ki
            _set(keys, j + 1, keys[j])
            _set(arr, j + 1, arr[j])
            j = j - 1
        end
        _set(keys, j + 1, ki)
        _set(arr, j + 1, vi)
        i = i + 1
    end
    return arr
end

func isSorted(arr)
    var i = 1
    while i < _alen(arr)
        if arr[i] < arr[i - 1]
            return false
        end
        i = i + 1
    end
    return true
end
