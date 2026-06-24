func _hexDigit(d)
    if d == 10
        return "a"
    else if d == 11
        return "b"
    else if d == 12
        return "c"
    else if d == 13
        return "d"
    else if d == 14
        return "e"
    else if d == 15
        return "f"
    end
    return _str(d)
end

func toBinary(n)
    if n == 0
        return "0"
    end
    var result = ""
    var x = abs(n)
    while x > 0
        result = concat(_str(x % 2), result)
        x = _int(x / 2)
    end
    if n < 0
        result = concat("-", result)
    end
    return result
end

func toOctal(n)
    if n == 0
        return "0"
    end
    var result = ""
    var x = abs(n)
    while x > 0
        result = concat(_str(x % 8), result)
        x = _int(x / 8)
    end
    if n < 0
        result = concat("-", result)
    end
    return result
end

func toHex(n)
    if n == 0
        return "0"
    end
    var result = ""
    var x = abs(n)
    while x > 0
        result = concat(_hexDigit(x % 16), result)
        x = _int(x / 16)
    end
    if n < 0
        result = concat("-", result)
    end
    return result
end

func toBase(n, base)
    if n == 0
        return "0"
    end
    var chars = "0123456789abcdefghijklmnopqrstuvwxyz"
    var result = ""
    var x = abs(n)
    while x > 0
        result = concat(charAt(chars, x % base), result)
        x = _int(x / base)
    end
    if n < 0
        result = concat("-", result)
    end
    return result
end

func fromBinary(s)
    var result = 0
    var i = 0
    while i < len(s)
        result = result * 2 + _int(_num(charAt(s, i)))
        i = i + 1
    end
    return result
end

func _hexVal(c)
    c = lower(c)
    if c == "a"
        return 10
    else if c == "b"
        return 11
    else if c == "c"
        return 12
    else if c == "d"
        return 13
    else if c == "e"
        return 14
    else if c == "f"
        return 15
    end
    return _int(_num(c))
end

func fromHex(s)
    var result = 0
    var i = 0
    while i < len(s)
        result = result * 16 + _hexVal(charAt(s, i))
        i = i + 1
    end
    return result
end

func fromBase(s, base)
    var chars = "0123456789abcdefghijklmnopqrstuvwxyz"
    var result = 0
    var i = 0
    while i < len(s)
        var c = lower(charAt(s, i))
        var val = indexOf(chars, c)
        result = result * base + val
        i = i + 1
    end
    return result
end

func toRoman(n)
    var values = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    var symbols = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
    var result = ""
    var i = 0
    while i < _alen(values)
        while n >= values[i]
            result = concat(result, symbols[i])
            n = n - values[i]
        end
        i = i + 1
    end
    return result
end

func fromRoman(s)
    var map = _dict()
    map["I"] = 1
    map["V"] = 5
    map["X"] = 10
    map["L"] = 50
    map["C"] = 100
    map["D"] = 500
    map["M"] = 1000
    var result = 0
    var i = 0
    while i < len(s)
        var curr = map[charAt(s, i)]
        if i + 1 < len(s) and curr < map[charAt(s, i + 1)]
            result = result - curr
        else
            result = result + curr
        end
        i = i + 1
    end
    return result
end

func zfill(n, width)
    var s = _str(abs(n))
    while len(s) < width
        s = concat("0", s)
    end
    if n < 0
        s = concat("-", s)
    end
    return s
end

func digitSum(n)
    var x = abs(n)
    var total = 0
    while x > 0
        total = total + (x % 10)
        x = _int(x / 10)
    end
    return total
end

func reverseDigits(n)
    var neg = n < 0
    var x = abs(n)
    var result = 0
    while x > 0
        result = result * 10 + (x % 10)
        x = _int(x / 10)
    end
    if neg
        return 0 - result
    end
    return result
end

func isPerfect(n)
    if n < 2
        return false
    end
    var s = 1
    var i = 2
    while i * i <= n
        if n % i == 0
            s = s + i
            if i != _int(n / i)
                s = s + _int(n / i)
            end
        end
        i = i + 1
    end
    return s == n
end

func primeFactors(n)
    var factors = []
    var d = 2
    while d * d <= n
        while n % d == 0
            _push(factors, d)
            n = _int(n / d)
        end
        d = d + 1
    end
    if n > 1
        _push(factors, n)
    end
    return factors
end

func sieve(limit)
    var isP = []
    var i = 0
    while i <= limit
        _push(isP, true)
        i = i + 1
    end
    isP[0] = false
    if limit >= 1
        isP[1] = false
    end
    i = 2
    while i * i <= limit
        if isP[i]
            var j = i * i
            while j <= limit
                isP[j] = false
                j = j + i
            end
        end
        i = i + 1
    end
    var primes = []
    i = 2
    while i <= limit
        if isP[i]
            _push(primes, i)
        end
        i = i + 1
    end
    return primes
end

func nthPrime(n)
    var found = []
    var candidate = 2
    while _alen(found) < n
        if isPrime(candidate)
            _push(found, candidate)
        end
        candidate = candidate + 1
    end
    return found[n - 1]
end

func isNumericString(s)
    if len(s) == 0
        return false
    end
    var start = 0
    if charAt(s, 0) == "-" or charAt(s, 0) == "+"
        start = 1
    end
    if start >= len(s)
        return false
    end
    var hasDot = false
    var i = start
    while i < len(s)
        var c = charAt(s, i)
        if c == "."
            if hasDot
                return false
            end
            hasDot = true
        else if not isDigit(c)
            return false
        end
        i = i + 1
    end
    return true
end
