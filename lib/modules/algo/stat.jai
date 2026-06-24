func statSum(arr)
    var s = 0
    var i = 0
    while i < _alen(arr)
        s = s + arr[i]
        i = i + 1
    end
    return s
end

func statMean(arr)
    if _alen(arr) == 0
        return 0
    end
    return statSum(arr) / _alen(arr)
end

func statVariance(arr, population)
    var n = _alen(arr)
    if n == 0
        return 0
    end
    var m = statMean(arr)
    var s = 0
    var i = 0
    while i < n
        var d = arr[i] - m
        s = s + d * d
        i = i + 1
    end
    if population
        return s / n
    end
    if n == 1
        return 0
    end
    return s / (n - 1)
end

func statStddev(arr, population)
    return sqrt(statVariance(arr, population))
end

func statMedian(arr)
    var n = _alen(arr)
    if n == 0
        return null
    end
    var sorted = arrayCopy(arr)
    sort(sorted)
    var mid = _int(n / 2)
    if n % 2 == 0
        return (sorted[mid - 1] + sorted[mid]) / 2
    end
    return sorted[mid]
end

func statMode(arr)
    var n = _alen(arr)
    if n == 0
        return null
    end
    var counts = _dict()
    var i = 0
    while i < n
        var key = _str(arr[i])
        if _dict_has(counts, key)
            counts[key] = counts[key] + 1
        else
            counts[key] = 1
        end
        i = i + 1
    end
    var bestVal = arr[0]
    var bestCount = 0
    i = 0
    while i < n
        var key = _str(arr[i])
        if counts[key] > bestCount
            bestCount = counts[key]
            bestVal = arr[i]
        end
        i = i + 1
    end
    return bestVal
end

func statRange(arr)
    if _alen(arr) == 0
        return 0
    end
    var lo = arr[0]
    var hi = arr[0]
    var i = 1
    while i < _alen(arr)
        if arr[i] < lo
            lo = arr[i]
        end
        if arr[i] > hi
            hi = arr[i]
        end
        i = i + 1
    end
    return hi - lo
end

func statCovariance(xs, ys)
    var n = _alen(xs)
    if n == 0 or n != _alen(ys)
        return 0
    end
    var mx = statMean(xs)
    var my = statMean(ys)
    var s = 0
    var i = 0
    while i < n
        s = s + (xs[i] - mx) * (ys[i] - my)
        i = i + 1
    end
    return s / n
end

func statCorrelation(xs, ys)
    var cov = statCovariance(xs, ys)
    var sx = statStddev(xs, true)
    var sy = statStddev(ys, true)
    if sx == 0 or sy == 0
        return 0
    end
    return cov / (sx * sy)
end

func statLinearRegression(xs, ys)
    var n = _alen(xs)
    if n == 0
        return [0, 0]
    end
    var mx = statMean(xs)
    var my = statMean(ys)
    var num = 0
    var den = 0
    var i = 0
    while i < n
        num = num + (xs[i] - mx) * (ys[i] - my)
        den = den + (xs[i] - mx) * (xs[i] - mx)
        i = i + 1
    end
    if den == 0
        return [0, my]
    end
    var slope = num / den
    var intercept = my - slope * mx
    return [slope, intercept]
end

func statPercentile(arr, p)
    if _alen(arr) == 0
        return null
    end
    var sorted = arrayCopy(arr)
    sort(sorted)
    var n = _alen(sorted)
    var idx = p / 100 * (n - 1)
    var lo = _int(idx)
    var hi = lo + 1
    if hi >= n
        return sorted[n - 1]
    end
    var frac = idx - lo
    return sorted[lo] + frac * (sorted[hi] - sorted[lo])
end

func statQ1(arr)
    return statPercentile(arr, 25)
end

func statQ3(arr)
    return statPercentile(arr, 75)
end

func statIQR(arr)
    return statQ3(arr) - statQ1(arr)
end

func statZScore(x, arr)
    var m = statMean(arr)
    var s = statStddev(arr, false)
    if s == 0
        return 0
    end
    return (x - m) / s
end

func statNormalize(arr)
    var lo = arrayMin(arr)
    var hi = arrayMax(arr)
    var rng = hi - lo
    if rng == 0
        return arrayFill(_alen(arr), 0)
    end
    var result = []
    var i = 0
    while i < _alen(arr)
        _push(result, (arr[i] - lo) / rng)
        i = i + 1
    end
    return result
end

func statStandardize(arr)
    var m = statMean(arr)
    var s = statStddev(arr, false)
    var result = []
    var i = 0
    while i < _alen(arr)
        if s == 0
            _push(result, 0)
        else
            _push(result, (arr[i] - m) / s)
        end
        i = i + 1
    end
    return result
end

func statMovingAvg(arr, window)
    var n = _alen(arr)
    var result = []
    var i = 0
    while i < n
        var start = i - window + 1
        if start < 0
            start = 0
        end
        var sum = 0
        var j = start
        while j <= i
            sum = sum + arr[j]
            j = j + 1
        end
        _push(result, sum / (i - start + 1))
        i = i + 1
    end
    return result
end

func statOutliers(arr, threshold)
    var m = statMean(arr)
    var s = statStddev(arr, false)
    var result = []
    var i = 0
    while i < _alen(arr)
        if abs(arr[i] - m) > threshold * s
            _push(result, arr[i])
        end
        i = i + 1
    end
    return result
end

func statHistogram(arr, bins)
    if _alen(arr) == 0 or bins <= 0
        return []
    end
    var lo = arrayMin(arr)
    var hi = arrayMax(arr)
    var rng = hi - lo
    if rng == 0
        var counts = arrayFill(bins, 0)
        _set(counts, 0, _alen(arr))
        return counts
    end
    var counts = arrayFill(bins, 0)
    var i = 0
    while i < _alen(arr)
        var bin = _int((arr[i] - lo) / rng * bins)
        if bin >= bins
            bin = bins - 1
        end
        _set(counts, bin, counts[bin] + 1)
        i = i + 1
    end
    return counts
end
