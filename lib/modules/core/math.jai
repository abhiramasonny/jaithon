#math functions

func abs(x)
    if x < 0
        return 0 - x
    end
    return x
end

func floor(x)
    var n = x - (x % 1)
    if x < 0
        if x % 1 < 0
            return n - 1
        end
    end
    return n
end

func ceil(x)
    var n = floor(x)
    if x > n
        return n + 1
    end
    return n
end

func round(x)
    if x - floor(x) < 0.5
        return floor(x)
    end
    return ceil(x)
end

func trunc(x)
    if x < 0
        return ceil(x)
    end
    return floor(x)
end

func sign(x)
    if x > 0
        return 1
    end
    if x < 0
        return 0 - 1
    end
    return 0
end

func max(a, b)
    if a > b
        return a
    end
    return b
end

func min(a, b)
    if a < b
        return a
    end
    return b
end

func clamp(val, lo, hi)
    return max(lo, min(val, hi))
end

func lerp(a, b, t)
    return a + (b - a) * t
end

func pow(base, exponent)
    if exponent == 0
        return 1
    end
    if exponent < 0
        return 1 / pow(base, 0 - exponent)
    end
    var result = 1
    var i = 0
    while i < exponent
        result = result * base
        i = i + 1
    end
    return result
end

func pow2(x)
    return x * x
end

func pow3(x)
    return x * x * x
end

func cbrt(x)
    if x == 0
        return 0
    end
    var guess = x / 3
    var i = 0
    while i < 20
        guess = (2 * guess + x / (guess * guess)) / 3
        i = i + 1
    end
    return guess
end

func hypot(a, b)
    return sqrt(a * a + b * b)
end

func sec(x)
    return 1 / cos(x)
end

func csc(x)
    return 1 / sin(x)
end

func cot(x)
    return cos(x) / sin(x)
end

func sinh(x)
    return (exp(x) - exp(0 - x)) / 2
end

func cosh(x)
    return (exp(x) + exp(0 - x)) / 2
end

func tanh(x)
    return sinh(x) / cosh(x)
end

func degToRad(deg)
    return deg * PI / 180
end

func radToDeg(rad)
    return rad * 180 / PI
end

func isEven(n)
    return n % 2 == 0
end

func isOdd(n)
    return n % 2 == 1 or n % 2 == -1
end

func isPositive(n)
    return n > 0
end

func isNegative(n)
    return n < 0
end

func isZero(n)
    return n == 0
end

func isInteger(n)
    return n == floor(n)
end

func gcd(a, b)
    a = abs(a)
    b = abs(b)
    while b > 0
        var temp = b
        b = a % b
        a = temp
    end
    return a
end

func lcm(a, b)
    return abs(a * b) / gcd(a, b)
end

func isPrime(n)
    if n < 2
        return false
    end
    if n == 2
        return true
    end
    if n % 2 == 0
        return false
    end
    var i = 3
    while i * i <= n
        if n % i == 0
            return false
        end
        i = i + 2
    end
    return true
end

func fact(n)
    if n < 2
        return 1
    end
    var result = 1
    var i = 2
    while i <= n
        result = result * i
        i = i + 1
    end
    return result
end

func nPr(n, r)
    return fact(n) / fact(n - r)
end

func nCr(n, r)
    return fact(n) / (fact(r) * fact(n - r))
end

func fib(n)
    if n < 2
        return n
    end
    var a = 0
    var b = 1
    var i = 2
    while i <= n
        var temp = a + b
        a = b
        b = temp
        i = i + 1
    end
    return b
end

func avg(a, b)
    return (a + b) / 2
end

func geometricMean(a, b)
    return sqrt(a * b)
end

func harmonicMean(a, b)
    return 2 * a * b / (a + b)
end

func log2(x)
    return _log(x) / LN2
end

func log10(x)
    return _log(x) / LN10
end

func logBase(x, base)
    return _log(x) / _log(base)
end

func atan(x)
    if x < 0
        return 0 - atan(0 - x)
    end
    if x > 1
        return PI / 2 - atan(1 / x)
    end
    var result = 0
    var term = x
    var xsq = x * x
    var i = 0
    while i < 20
        if i % 2 == 0
            result = result + term / (2 * i + 1)
        else
            result = result - term / (2 * i + 1)
        end
        term = term * xsq
        i = i + 1
    end
    return result
end

func atan2(y, x)
    if x > 0
        return atan(y / x)
    end
    if x < 0
        if y >= 0
            return atan(y / x) + PI
        end
        return atan(y / x) - PI
    end
    if y > 0
        return PI / 2
    end
    if y < 0
        return 0 - PI / 2
    end
    return 0
end

func asin(x)
    if x < -1 or x > 1
        raise "asin: domain error"
    end
    if x == 1
        return PI / 2
    end
    if x == -1
        return 0 - PI / 2
    end
    return atan(x / sqrt(1 - x * x))
end

func acos(x)
    if x < -1 or x > 1
        raise "acos: domain error"
    end
    return PI / 2 - asin(x)
end

func sumArray(arr)
    var total = 0
    var i = 0
    while i < _alen(arr)
        total = total + arr[i]
        i = i + 1
    end
    return total
end

func productArray(arr)
    var total = 1
    var i = 0
    while i < _alen(arr)
        total = total * arr[i]
        i = i + 1
    end
    return total
end

func mean(arr)
    if _alen(arr) == 0
        return 0
    end
    return sumArray(arr) / _alen(arr)
end

func variance(arr)
    var m = mean(arr)
    var total = 0
    var i = 0
    while i < _alen(arr)
        var diff = arr[i] - m
        total = total + diff * diff
        i = i + 1
    end
    return total / _alen(arr)
end

func stddev(arr)
    return sqrt(variance(arr))
end

func median(arr)
    var sorted = arrayCopy(arr)
    sort(sorted)
    var n = _alen(sorted)
    if n % 2 == 1
        return sorted[_int(n / 2)]
    end
    return (sorted[_int(n / 2) - 1] + sorted[_int(n / 2)]) / 2
end

func mode(arr)
    var freq = frequencies(arr)
    var keys = _dict_keys(freq)
    var bestKey = keys[0]
    var bestCount = freq[keys[0]]
    var i = 1
    while i < _alen(keys)
        if freq[keys[i]] > bestCount
            bestCount = freq[keys[i]]
            bestKey = keys[i]
        end
        i = i + 1
    end
    return _num(bestKey)
end
