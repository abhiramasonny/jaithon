func strlen(s)
    return len(s)
end

func strcat(a, b)
    return concat(a, b)
end

func repeat(s, n)
    var result = ""
    var i = 0
    while i < n
        result = concat(result, s)
        i = i + 1
    end
    return result
end

func startsWith(s, prefix)
    if len(prefix) > len(s)
        return false
    end
    return substr(s, 0, len(prefix)) == prefix
end

func endsWith(s, suffix)
    if len(suffix) > len(s)
        return false
    end
    return substr(s, len(s) - len(suffix), len(suffix)) == suffix
end

func contains_str(s, sub)
    var i = 0
    var subLen = len(sub)
    var sLen = len(s)
    while i <= sLen - subLen
        if substr(s, i, subLen) == sub
            return true
        end
        i = i + 1
    end
    return false
end

func indexOf(s, sub)
    var i = 0
    var subLen = len(sub)
    var sLen = len(s)
    while i <= sLen - subLen
        if substr(s, i, subLen) == sub
            return i
        end
        i = i + 1
    end
    return 0 - 1
end

func upper(s)
    var lower = "abcdefghijklmnopqrstuvwxyz"
    var upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    var result = ""
    var i = 0
    while i < len(s)
        var c = charAt(s, i)
        var idx = indexOf(lower, c)
        if idx >= 0
            result = concat(result, charAt(upper, idx))
        else
            result = concat(result, c)
        end
        i = i + 1
    end
    return result
end

func lower(s)
    var lower = "abcdefghijklmnopqrstuvwxyz"
    var upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    var result = ""
    var i = 0
    while i < len(s)
        var c = charAt(s, i)
        var idx = indexOf(upper, c)
        if idx >= 0
            result = concat(result, charAt(lower, idx))
        else
            result = concat(result, c)
        end
        i = i + 1
    end
    return result
end

func replace(s, old, replacement)
    var result = ""
    var i = 0
    var sLen = len(s)
    var oldLen = len(old)
    while i < sLen
        if i <= sLen - oldLen
            if substr(s, i, oldLen) == old
                result = concat(result, replacement)
                i = i + oldLen
            else
                result = concat(result, charAt(s, i))
                i = i + 1
            end
        else
            result = concat(result, charAt(s, i))
            i = i + 1
        end
    end
    return result
end

func reverse_str(s)
    var result = ""
    var i = len(s) - 1
    while i >= 0
        result = concat(result, charAt(s, i))
        i = i - 1
    end
    return result
end

func isDigit(c)
    return c == "0" or c == "1" or c == "2" or c == "3" or c == "4" or c == "5" or c == "6" or c == "7" or c == "8" or c == "9"
end

func isLetter(c)
    var code = c
    return (code >= "a" and code <= "z") or (code >= "A" and code <= "Z")
end

func isSpace(c)
    return c == " " or c == "\t" or c == "\n"
end

func trim(s)
    var start = 0
    var end_idx = len(s) - 1
    while start <= end_idx and isSpace(charAt(s, start))
        start = start + 1
    end
    while end_idx >= start and isSpace(charAt(s, end_idx))
        end_idx = end_idx - 1
    end
    if start > end_idx
        return ""
    end
    return substr(s, start, end_idx - start + 1)
end
