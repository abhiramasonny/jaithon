func levenshtein(a, b)
    var la = len(a)
    var lb = len(b)
    var dp = []
    var i = 0
    while i <= la
        var row = []
        var j = 0
        while j <= lb
            _push(row, 0)
            j = j + 1
        end
        _push(dp, row)
        i = i + 1
    end
    i = 0
    while i <= la
        dp[i][0] = i
        i = i + 1
    end
    var j = 0
    while j <= lb
        dp[0][j] = j
        j = j + 1
    end
    i = 1
    while i <= la
        j = 1
        while j <= lb
            if charAt(a, i - 1) == charAt(b, j - 1)
                dp[i][j] = dp[i - 1][j - 1]
            else
                dp[i][j] = 1 + min(dp[i - 1][j], min(dp[i][j - 1], dp[i - 1][j - 1]))
            end
            j = j + 1
        end
        i = i + 1
    end
    return dp[la][lb]
end

func editDistance(a, b)
    return levenshtein(a, b)
end

func lcsLength(a, b)
    var la = len(a)
    var lb = len(b)
    var dp = []
    var i = 0
    while i <= la
        var row = []
        var j = 0
        while j <= lb
            _push(row, 0)
            j = j + 1
        end
        _push(dp, row)
        i = i + 1
    end
    i = 1
    while i <= la
        var j = 1
        while j <= lb
            if charAt(a, i - 1) == charAt(b, j - 1)
                dp[i][j] = dp[i - 1][j - 1] + 1
            else
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
            end
            j = j + 1
        end
        i = i + 1
    end
    return dp[la][lb]
end

func lcsString(a, b)
    var la = len(a)
    var lb = len(b)
    var dp = []
    var i = 0
    while i <= la
        var row = []
        var j = 0
        while j <= lb
            _push(row, 0)
            j = j + 1
        end
        _push(dp, row)
        i = i + 1
    end
    i = 1
    while i <= la
        var j = 1
        while j <= lb
            if charAt(a, i - 1) == charAt(b, j - 1)
                dp[i][j] = dp[i - 1][j - 1] + 1
            else
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
            end
            j = j + 1
        end
        i = i + 1
    end
    var result = ""
    i = la
    j = lb
    while i > 0 and j > 0
        if charAt(a, i - 1) == charAt(b, j - 1)
            result = concat(charAt(a, i - 1), result)
            i = i - 1
            j = j - 1
        else if dp[i - 1][j] > dp[i][j - 1]
            i = i - 1
        else
            j = j - 1
        end
    end
    return result
end

func kmpSearch(text, pattern)
    var n = len(text)
    var m = len(pattern)
    if m == 0
        return []
    end
    var fail = []
    _push(fail, 0)
    var k = 0
    var i = 1
    while i < m
        while k > 0 and charAt(pattern, k) != charAt(pattern, i)
            k = fail[k - 1]
        end
        if charAt(pattern, k) == charAt(pattern, i)
            k = k + 1
        end
        _push(fail, k)
        i = i + 1
    end
    var results = []
    k = 0
    i = 0
    while i < n
        while k > 0 and charAt(pattern, k) != charAt(text, i)
            k = fail[k - 1]
        end
        if charAt(pattern, k) == charAt(text, i)
            k = k + 1
        end
        if k == m
            _push(results, i - m + 1)
            k = fail[k - 1]
        end
        i = i + 1
    end
    return results
end

func isPalindrome(s)
    var lo = 0
    var hi = len(s) - 1
    while lo < hi
        if charAt(s, lo) != charAt(s, hi)
            return false
        end
        lo = lo + 1
        hi = hi - 1
    end
    return true
end

func longestPalindromeLen(s)
    var n = len(s)
    if n == 0
        return 0
    end
    var best = 1
    var start = 0
    while start < n
        var lo = start
        var hi = start
        while hi + 1 < n and charAt(s, hi) == charAt(s, hi + 1)
            hi = hi + 1
        end
        start = hi + 1
        while lo > 0 and hi < n - 1 and charAt(s, lo - 1) == charAt(s, hi + 1)
            lo = lo - 1
            hi = hi + 1
        end
        if hi - lo + 1 > best
            best = hi - lo + 1
        end
    end
    return best
end

func isAnagram(a, b)
    if len(a) != len(b)
        return false
    end
    var counts = _dict()
    var i = 0
    while i < len(a)
        var c = charAt(a, i)
        if _dict_has(counts, c)
            counts[c] = counts[c] + 1
        else
            counts[c] = 1
        end
        i = i + 1
    end
    i = 0
    while i < len(b)
        var c = charAt(b, i)
        if not _dict_has(counts, c)
            return false
        end
        counts[c] = counts[c] - 1
        if counts[c] < 0
            return false
        end
        i = i + 1
    end
    return true
end

func longestCommonSubstring(a, b)
    var la = len(a)
    var lb = len(b)
    var best = 0
    var bestEnd = 0
    var dp = []
    var i = 0
    while i <= la
        var row = []
        var j = 0
        while j <= lb
            _push(row, 0)
            j = j + 1
        end
        _push(dp, row)
        i = i + 1
    end
    i = 1
    while i <= la
        var j = 1
        while j <= lb
            if charAt(a, i - 1) == charAt(b, j - 1)
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > best
                    best = dp[i][j]
                    bestEnd = i
                end
            else
                dp[i][j] = 0
            end
            j = j + 1
        end
        i = i + 1
    end
    return substr(a, bestEnd - best, best)
end

func hammingDistance(a, b)
    var dist = 0
    var n = min(len(a), len(b))
    var i = 0
    while i < n
        if charAt(a, i) != charAt(b, i)
            dist = dist + 1
        end
        i = i + 1
    end
    dist = dist + abs(len(a) - len(b))
    return dist
end
