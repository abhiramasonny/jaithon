func _formatValue(val, spec)
    if spec == "" or spec == "s"
        return _str(val)
    end
    if spec == "d" or spec == "i"
        return _str(_int(val))
    end
    if spec == "f"
        return _str(val)
    end
    if spec == "b"
        if val
            return "true"
        end
        return "false"
    end
    return _str(val)
end

func format(template, *args)
    var result = ""
    var argIdx = 0
    var i = 0
    var n = len(template)
    while i < n
        var c = charAt(template, i)
        if c == "{"
            if i + 1 < n and charAt(template, i + 1) == "{"
                result = concat(result, "{")
                i = i + 2
            else
                var j = i + 1
                while j < n and charAt(template, j) != "}"
                    j = j + 1
                end
                var spec = substr(template, i + 1, j - i - 1)
                if argIdx < _alen(args)
                    result = concat(result, _formatValue(_get(args, argIdx), spec))
                    argIdx = argIdx + 1
                end
                i = j + 1
            end
        else if c == "}" and i + 1 < n and charAt(template, i + 1) == "}"
            result = concat(result, "}")
            i = i + 2
        else
            result = concat(result, c)
            i = i + 1
        end
    end
    return result
end

func formatNum(n, width, padChar)
    var s = _str(n)
    while len(s) < width
        s = concat(padChar, s)
    end
    return s
end

func formatFixed(n, decimals)
    var factor = pow(10, decimals)
    var rounded = _int(n * factor + 0.5)
    var intPart = _int(rounded / factor)
    var fracPart = rounded - intPart * factor
    var fracStr = _str(fracPart)
    while len(fracStr) < decimals
        fracStr = concat("0", fracStr)
    end
    return concat(concat(_str(intPart), "."), fracStr)
end

func formatPercent(n, decimals)
    return concat(formatFixed(n * 100, decimals), "%")
end

func rightAlign(s, width)
    while len(s) < width
        s = concat(" ", s)
    end
    return s
end

func leftAlign(s, width)
    while len(s) < width
        s = concat(s, " ")
    end
    return s
end

func center(s, width)
    var total = width - len(s)
    var left = _int(total / 2)
    var right = total - left
    var result = s
    var i = 0
    while i < left
        result = concat(" ", result)
        i = i + 1
    end
    i = 0
    while i < right
        result = concat(result, " ")
        i = i + 1
    end
    return result
end
