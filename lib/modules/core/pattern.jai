func globMatch(pattern, text)
    var pi = 0
    var ti = 0
    var plen = len(pattern)
    var tlen = len(text)
    var starPi = -1
    var starTi = -1
    while ti < tlen
        if pi < plen and (charAt(pattern, pi) == charAt(text, ti) or charAt(pattern, pi) == "?")
            pi = pi + 1
            ti = ti + 1
        else
            if pi < plen and charAt(pattern, pi) == "*"
                starPi = pi
                starTi = ti
                pi = pi + 1
            else
                if starPi >= 0
                    pi = starPi + 1
                    starTi = starTi + 1
                    ti = starTi
                else
                    return false
                end
            end
        end
    end
    while pi < plen and charAt(pattern, pi) == "*"
        pi = pi + 1
    end
    return pi == plen
end

func matchPrefix(text, prefix)
    return startsWith(text, prefix)
end

func matchSuffix(text, suffix)
    return endsWith(text, suffix)
end

func matchAny(text, patterns)
    var i = 0
    while i < _alen(patterns)
        if globMatch(patterns[i], text)
            return true
        end
        i = i + 1
    end
    return false
end

func extractBetween(text, open, close)
    var openIdx = indexOf(text, open)
    if openIdx < 0
        return null
    end
    var start = openIdx + len(open)
    var closeIdx = indexOf(substr(text, start, len(text) - start), close)
    if closeIdx < 0
        return null
    end
    return substr(text, start, closeIdx)
end

func findAll(text, sub)
    var positions = []
    var i = 0
    var subLen = len(sub)
    var textLen = len(text)
    while i <= textLen - subLen
        if substr(text, i, subLen) == sub
            _push(positions, i)
            i = i + subLen
        else
            i = i + 1
        end
    end
    return positions
end

func replaceAll(text, pattern, replacement)
    return replace(text, pattern, replacement)
end

func tokenize(text, delimiters)
    var tokens = []
    var current = ""
    var i = 0
    while i < len(text)
        var c = charAt(text, i)
        var isDelim = false
        var j = 0
        while j < len(delimiters)
            if c == charAt(delimiters, j)
                isDelim = true
            end
            j = j + 1
        end
        if isDelim
            if len(current) > 0
                _push(tokens, current)
                current = ""
            end
        else
            current = concat(current, c)
        end
        i = i + 1
    end
    if len(current) > 0
        _push(tokens, current)
    end
    return tokens
end

func parseKeyValue(text, pairSep, kvSep)
    var result = _dict()
    var pairs = split(text, pairSep)
    var i = 0
    while i < _alen(pairs)
        var pair = pairs[i]
        var kv = split(pair, kvSep)
        if _alen(kv) == 2
            result[trim(kv[0])] = trim(kv[1])
        end
        i = i + 1
    end
    return result
end

func parseCSVLine(line)
    var fields = []
    var current = ""
    var inQuote = false
    var i = 0
    while i < len(line)
        var c = charAt(line, i)
        if c == "\"" and not inQuote
            inQuote = true
        else
            if c == "\"" and inQuote
                inQuote = false
            else
                if c == "," and not inQuote
                    _push(fields, current)
                    current = ""
                else
                    current = concat(current, c)
                end
            end
        end
        i = i + 1
    end
    _push(fields, current)
    return fields
end

func parseCSV(text)
    var lines = split(text, "\n")
    var result = []
    var i = 0
    while i < _alen(lines)
        if len(trim(lines[i])) > 0
            _push(result, parseCSVLine(lines[i]))
        end
        i = i + 1
    end
    return result
end

func escapeHtml(text)
    var result = text
    result = replace(result, "&", "&amp;")
    result = replace(result, "<", "&lt;")
    result = replace(result, ">", "&gt;")
    result = replace(result, "\"", "&quot;")
    return result
end
