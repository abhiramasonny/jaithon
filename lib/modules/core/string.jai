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

func split(s, delim)
    var result = []
    var current = ""
    var dlen = len(delim)
    var slen = len(s)
    var i = 0
    while i < slen
        if i <= slen - dlen and substr(s, i, dlen) == delim
            _push(result, current)
            current = ""
            i = i + dlen
        else
            current = concat(current, charAt(s, i))
            i = i + 1
        end
    end
    _push(result, current)
    return result
end

func join(arr, sep)
    var result = ""
    var i = 0
    while i < _alen(arr)
        if i > 0
            result = concat(result, sep)
        end
        result = concat(result, _str(arr[i]))
        i = i + 1
    end
    return result
end

func strContains(s, sub)
    return contains_str(s, sub)
end

func lpad(s, width, ch)
    var result = s
    while len(result) < width
        result = concat(ch, result)
    end
    return result
end

func rpad(s, width, ch)
    var result = s
    while len(result) < width
        result = concat(result, ch)
    end
    return result
end

func countOccurrences(s, sub)
    var count = 0
    var i = 0
    var subLen = len(sub)
    var sLen = len(s)
    while i <= sLen - subLen
        if substr(s, i, subLen) == sub
            count = count + 1
            i = i + subLen
        else
            i = i + 1
        end
    end
    return count
end

func lines(s)
    return split(s, "\n")
end

func titleCase(s)
    var words = split(s, " ")
    var result = []
    var i = 0
    while i < _alen(words)
        var w = words[i]
        if len(w) > 0
            _push(result, concat(upper(substr(w, 0, 1)), lower(substr(w, 1, len(w) - 1))))
        else
            _push(result, w)
        end
        i = i + 1
    end
    return join(result, " ")
end

func camelCase(s)
    var words = split(replace(s, "_", " "), " ")
    if _alen(words) == 0
        return ""
    end
    var result = lower(words[0])
    var i = 1
    while i < _alen(words)
        var w = words[i]
        if len(w) > 0
            result = concat(result, concat(upper(substr(w, 0, 1)), lower(substr(w, 1, len(w) - 1))))
        end
        i = i + 1
    end
    return result
end

func snakeCase(s)
    var result = ""
    var i = 0
    while i < len(s)
        var c = charAt(s, i)
        if c == " " or c == "-"
            result = concat(result, "_")
        else
            result = concat(result, lower(c))
        end
        i = i + 1
    end
    return result
end

func kebabCase(s)
    var result = ""
    var i = 0
    while i < len(s)
        var c = charAt(s, i)
        if c == " " or c == "_"
            result = concat(result, "-")
        else
            result = concat(result, lower(c))
        end
        i = i + 1
    end
    return result
end

func slugify(s)
    return replace(lower(trim(s)), " ", "-")
end

func stripPrefix(s, prefix)
    if startsWith(s, prefix)
        return substr(s, len(prefix), len(s) - len(prefix))
    end
    return s
end

func stripSuffix(s, suffix)
    if endsWith(s, suffix)
        return substr(s, 0, len(s) - len(suffix))
    end
    return s
end

func wrapText(s, width)
    var words = split(s, " ")
    var ls = []
    var current = ""
    var i = 0
    while i < _alen(words)
        var word = words[i]
        if len(current) == 0
            current = word
        else if len(current) + 1 + len(word) <= width
            current = concat(concat(current, " "), word)
        else
            _push(ls, current)
            current = word
        end
        i = i + 1
    end
    if len(current) > 0
        _push(ls, current)
    end
    return join(ls, "\n")
end

func commonPrefix(a, b)
    var n = min(len(a), len(b))
    var i = 0
    while i < n and charAt(a, i) == charAt(b, i)
        i = i + 1
    end
    return substr(a, 0, i)
end

func commonSuffix(a, b)
    var la = len(a)
    var lb = len(b)
    var n = min(la, lb)
    var i = 0
    while i < n and charAt(a, la - 1 - i) == charAt(b, lb - 1 - i)
        i = i + 1
    end
    return substr(a, la - i, i)
end

func chunk(s, size)
    var result = []
    var i = 0
    while i < len(s)
        var end_idx = i + size
        if end_idx > len(s)
            end_idx = len(s)
        end
        _push(result, substr(s, i, end_idx - i))
        i = i + size
    end
    return result
end

func collapseSpaces(s)
    var result = ""
    var inSpace = false
    var i = 0
    while i < len(s)
        var c = charAt(s, i)
        if isSpace(c)
            if not inSpace
                result = concat(result, " ")
                inSpace = true
            end
        else
            result = concat(result, c)
            inSpace = false
        end
        i = i + 1
    end
    return trim(result)
end

func strMultiply(s, n)
    return repeat(s, n)
end

func isBlank(s)
    return len(trim(s)) == 0
end

func firstN(s, n)
    if n >= len(s)
        return s
    end
    return substr(s, 0, n)
end

func lastN(s, n)
    var l = len(s)
    if n >= l
        return s
    end
    return substr(s, l - n, n)
end

func insertAt(s, pos, ins)
    return concat(concat(substr(s, 0, pos), ins), substr(s, pos, len(s) - pos))
end

func deleteRange(s, start, count)
    return concat(substr(s, 0, start), substr(s, start + count, len(s) - start - count))
end
