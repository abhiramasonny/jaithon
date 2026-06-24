func extGCD(a, b)
    if b == 0
        return [a, 1, 0]
    end
    var r = extGCD(b, a % b)
    return [r[0], r[2], r[1] - _int(a / b) * r[2]]
end

func modInverse(a, m)
    var r = extGCD(a % m, m)
    if r[0] != 1
        return null
    end
    return (r[1] % m + m) % m
end

func modpow(base, exp, mod)
    if mod == 1
        return 0
    end
    var result = 1
    base = base % mod
    while exp > 0
        if exp % 2 == 1
            result = result * base % mod
        end
        exp = _int(exp / 2)
        base = base * base % mod
    end
    return result
end

func chineseRemainder(remainders, moduli)
    var M = 1
    var i = 0
    while i < _alen(moduli)
        M = M * moduli[i]
        i = i + 1
    end
    var result = 0
    i = 0
    while i < _alen(remainders)
        var mi = moduli[i]
        var ri = remainders[i]
        var Mi = _int(M / mi)
        var inv = modInverse(Mi, mi)
        result = (result + ri * Mi * inv) % M
        i = i + 1
    end
    return result
end

func totient(n)
    var result = n
    var p = 2
    var temp = n
    while p * p <= temp
        if temp % p == 0
            while temp % p == 0
                temp = _int(temp / p)
            end
            result = result - _int(result / p)
        end
        p = p + 1
    end
    if temp > 1
        result = result - _int(result / temp)
    end
    return result
end

func mobius(n)
    if n == 1
        return 1
    end
    var count = 0
    var p = 2
    var temp = n
    while p * p <= temp
        if temp % p == 0
            count = count + 1
            temp = _int(temp / p)
            if temp % p == 0
                return 0
            end
        end
        p = p + 1
    end
    if temp > 1
        count = count + 1
    end
    if count % 2 == 0
        return 1
    end
    return 0 - 1
end

func isPrimeProbabilistic(n, k)
    if n < 2
        return false
    end
    if n == 2 or n == 3
        return true
    end
    if n % 2 == 0
        return false
    end
    var d = n - 1
    var r = 0
    while d % 2 == 0
        d = _int(d / 2)
        r = r + 1
    end
    var i = 0
    while i < k
        var a = 2 + _int(_rand() * (n - 4))
        var x = modpow(a, d, n)
        if x != 1 and x != n - 1
            var j = 0
            var composite = true
            while j < r - 1
                x = modpow(x, 2, n)
                if x == n - 1
                    composite = false
                    j = r
                end
                j = j + 1
            end
            if composite
                return false
            end
        end
        i = i + 1
    end
    return true
end

func discreteLog(g, h, p)
    var m = _int(sqrt(_num(p))) + 1
    var table = _dict()
    var e = 1
    var j = 0
    while j < m
        table[_str(e)] = j
        e = e * g % p
        j = j + 1
    end
    var factor = modInverse(modpow(g, m, p), p)
    var val = h
    j = 0
    while j < m
        var key = _str(val)
        if _dict_has(table, key)
            return j * m + table[key]
        end
        val = val * factor % p
        j = j + 1
    end
    return 0 - 1
end

func primeFactorsUnique(n)
    var factors = []
    var p = 2
    while p * p <= n
        if n % p == 0
            _push(factors, p)
            while n % p == 0
                n = _int(n / p)
            end
        end
        p = p + 1
    end
    if n > 1
        _push(factors, n)
    end
    return factors
end

func legendre(a, p)
    return modpow(a, _int((p - 1) / 2), p)
end

func multiplicativeOrder(a, n)
    if gcd(a, n) != 1
        return 0 - 1
    end
    var order = 1
    var cur = a % n
    while cur != 1
        cur = cur * a % n
        order = order + 1
    end
    return order
end
