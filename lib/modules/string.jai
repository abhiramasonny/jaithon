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
    return substr(s, 0, len(prefix)) eq prefix
end

func endsWith(s, suffix)
    if len(suffix) > len(s)
        return false
    end
    return substr(s, len(s) - len(suffix), len(suffix)) eq suffix
end

func contains_str(s, sub)
    var i = 0
    var subLen = len(sub)
    var sLen = len(s)
    while i <= sLen - subLen
        if substr(s, i, subLen) eq sub
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
        if substr(s, i, subLen) eq sub
            return i
        end
        i = i + 1
    end
    return 0 - 1
end

func replace(s, old, replacement)
    var result = ""
    var i = 0
    var sLen = len(s)
    var oldLen = len(old)
    while i < sLen
        if i <= sLen - oldLen
            if substr(s, i, oldLen) eq old
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
    return c eq "0" or c eq "1" or c eq "2" or c eq "3" or c eq "4" or c eq "5" or c eq "6" or c eq "7" or c eq "8" or c eq "9"
end

func isLetter(c)
    var code = c
    return (code >= "a" and code <= "z") or (code >= "A" and code <= "Z")
end

func isSpace(c)
    return c eq " " or c eq "\t" or c eq "\n"
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
