func binarySearch(arr, target)
    var lo = 0
    var hi = _alen(arr) - 1
    while lo <= hi
        var mid = lo + _int((hi - lo) / 2)
        if arr[mid] == target
            return mid
        else if arr[mid] < target
            lo = mid + 1
        else
            hi = mid - 1
        end
    end
    return -1
end

func binarySearchLeft(arr, target)
    var lo = 0
    var hi = _alen(arr)
    while lo < hi
        var mid = lo + _int((hi - lo) / 2)
        if arr[mid] < target
            lo = mid + 1
        else
            hi = mid
        end
    end
    return lo
end

func binarySearchRight(arr, target)
    var lo = 0
    var hi = _alen(arr)
    while lo < hi
        var mid = lo + _int((hi - lo) / 2)
        if arr[mid] <= target
            lo = mid + 1
        else
            hi = mid
        end
    end
    return lo
end

func binarySearchBy(arr, target, keyFn)
    var lo = 0
    var hi = _alen(arr) - 1
    while lo <= hi
        var mid = lo + _int((hi - lo) / 2)
        var k = keyFn(arr[mid])
        if k == target
            return mid
        else if k < target
            lo = mid + 1
        else
            hi = mid - 1
        end
    end
    return -1
end

func linearSearch(arr, target)
    var i = 0
    while i < _alen(arr)
        if arr[i] == target
            return i
        end
        i = i + 1
    end
    return -1
end

func searchBy(arr, pred)
    var i = 0
    while i < _alen(arr)
        if pred(arr[i])
            return i
        end
        i = i + 1
    end
    return -1
end

func findAllIndices(arr, target)
    var result = []
    var i = 0
    while i < _alen(arr)
        if arr[i] == target
            _push(result, i)
        end
        i = i + 1
    end
    return result
end

func kthSmallest(arr, k)
    var sorted = arrayCopy(arr)
    sort(sorted)
    return sorted[k - 1]
end

func kthLargest(arr, k)
    var sorted = arrayCopy(arr)
    sort(sorted)
    return sorted[_alen(sorted) - k]
end

func lowerBound(arr, target)
    return binarySearchLeft(arr, target)
end

func upperBound(arr, target)
    return binarySearchRight(arr, target)
end

func interpolationSearch(arr, target)
    var lo = 0
    var hi = _alen(arr) - 1
    while lo <= hi and target >= arr[lo] and target <= arr[hi]
        if lo == hi
            if arr[lo] == target
                return lo
            end
            return -1
        end
        var pos = lo + _int((hi - lo) * (target - arr[lo]) / (arr[hi] - arr[lo]))
        if arr[pos] == target
            return pos
        else if arr[pos] < target
            lo = pos + 1
        else
            hi = pos - 1
        end
    end
    return -1
end
