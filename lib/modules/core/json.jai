func _jsonEscapeStr(s)
    var result = ""
    var i = 0
    var n = len(s)
    while i < n
        var c = charAt(s, i)
        if c == "\""
            result = concat(result, "\\\"")
        else
            if c == "\\"
                result = concat(result, "\\\\")
            else
                if c == "\n"
                    result = concat(result, "\\n")
                else
                    if c == "\t"
                        result = concat(result, "\\t")
                    else
                        result = concat(result, c)
                    end
                end
            end
        end
        i = i + 1
    end
    return result
end

func toJson(val)
    var t = _type(val)
    if t == "null"
        return "null"
    end
    if t == "bool"
        if val
            return "true"
        end
        return "false"
    end
    if t == "number" or t == "int" or t == "long" or t == "float" or t == "double" or t == "short" or t == "byte"
        return _str(val)
    end
    if t == "string"
        return concat(concat("\"", _jsonEscapeStr(val)), "\"")
    end
    if t == "array"
        var parts = []
        var i = 0
        while i < _alen(val)
            _push(parts, toJson(val[i]))
            i = i + 1
        end
        return concat(concat("[", join(parts, ",")), "]")
    end
    if t == "object"
        var keys = _dict_keys(val)
        var parts = []
        var i = 0
        while i < _alen(keys)
            var k = keys[i]
            var kJson = concat(concat("\"", _jsonEscapeStr(k)), "\"")
            var vJson = toJson(val[k])
            _push(parts, concat(concat(kJson, ":"), vJson))
            i = i + 1
        end
        return concat(concat("{", join(parts, ",")), "}")
    end
    return "null"
end

func toPrettyJson(val, indent)
    var pad = ""
    var i = 0
    while i < indent
        pad = concat(pad, "  ")
        i = i + 1
    end
    var innerPad = concat(pad, "  ")
    var t = _type(val)
    if t == "null"
        return "null"
    end
    if t == "bool"
        if val
            return "true"
        end
        return "false"
    end
    if t == "number" or t == "int" or t == "long" or t == "float" or t == "double" or t == "short" or t == "byte"
        return _str(val)
    end
    if t == "string"
        return concat(concat("\"", _jsonEscapeStr(val)), "\"")
    end
    if t == "array"
        if _alen(val) == 0
            return "[]"
        end
        var parts = []
        var j = 0
        while j < _alen(val)
            _push(parts, concat(innerPad, toPrettyJson(val[j], indent + 1)))
            j = j + 1
        end
        return concat(concat(concat("[\n", join(parts, ",\n")), "\n"), concat(pad, "]"))
    end
    if t == "object"
        var keys = _dict_keys(val)
        if _alen(keys) == 0
            return "{}"
        end
        var parts = []
        var j = 0
        while j < _alen(keys)
            var k = keys[j]
            var kJson = concat(concat("\"", _jsonEscapeStr(k)), "\"")
            var vJson = toPrettyJson(val[k], indent + 1)
            _push(parts, concat(concat(concat(innerPad, kJson), ": "), vJson))
            j = j + 1
        end
        return concat(concat(concat("{\n", join(parts, ",\n")), "\n"), concat(pad, "}"))
    end
    return "null"
end

func jsonPrint(val)
    print toPrettyJson(val, 0)
end

func _jSkipWS(s, i, n)
    while i < n
        var c = charAt(s, i)
        if c == " " or c == "\t" or c == "\n" or c == "\r"
            i = i + 1
        else
            return i
        end
    end
    return i
end

func _jParseStr(s, i, n)
    i = i + 1
    var r = ""
    while i < n
        var c = charAt(s, i)
        if c == "\""
            return [r, i + 1]
        end
        if c == "\\"
            i = i + 1
            if i >= n
                return [r, i]
            end
            var e = charAt(s, i)
            if e == "n"
                r = concat(r, "\n")
            else if e == "t"
                r = concat(r, "\t")
            else if e == "r"
                r = concat(r, "\r")
            else if e == "\\"
                r = concat(r, "\\")
            else if e == "\""
                r = concat(r, "\"")
            else
                r = concat(r, e)
            end
        else
            r = concat(r, c)
        end
        i = i + 1
    end
    return [r, i]
end

func _jParseNum(s, i, n)
    var start = i
    if i < n and charAt(s, i) == "-"
        i = i + 1
    end
    while i < n and isDigit(charAt(s, i))
        i = i + 1
    end
    if i < n and charAt(s, i) == "."
        i = i + 1
        while i < n and isDigit(charAt(s, i))
            i = i + 1
        end
    end
    if i < n and (charAt(s, i) == "e" or charAt(s, i) == "E")
        i = i + 1
        if i < n and (charAt(s, i) == "+" or charAt(s, i) == "-")
            i = i + 1
        end
        while i < n and isDigit(charAt(s, i))
            i = i + 1
        end
    end
    return [_num(substr(s, start, i - start)), i]
end

func _jParseArr(s, i, n)
    i = i + 1
    var result = []
    i = _jSkipWS(s, i, n)
    if i < n and charAt(s, i) == "]"
        return [result, i + 1]
    end
    while i < n
        var pair = _jParseVal(s, i, n)
        _push(result, pair[0])
        i = pair[1]
        i = _jSkipWS(s, i, n)
        if i >= n
            return [result, i]
        end
        var c = charAt(s, i)
        if c == "]"
            return [result, i + 1]
        end
        i = i + 1
    end
    return [result, i]
end

func _jParseObj(s, i, n)
    i = i + 1
    var result = _dict()
    i = _jSkipWS(s, i, n)
    if i < n and charAt(s, i) == "}"
        return [result, i + 1]
    end
    while i < n
        i = _jSkipWS(s, i, n)
        var kp = _jParseStr(s, i, n)
        var key = kp[0]
        i = kp[1]
        i = _jSkipWS(s, i, n)
        i = i + 1
        var vp = _jParseVal(s, i, n)
        result[key] = vp[0]
        i = vp[1]
        i = _jSkipWS(s, i, n)
        if i >= n
            return [result, i]
        end
        var c = charAt(s, i)
        if c == "}"
            return [result, i + 1]
        end
        i = i + 1
    end
    return [result, i]
end

func _jParseVal(s, i, n)
    i = _jSkipWS(s, i, n)
    if i >= n
        return [null, i]
    end
    var c = charAt(s, i)
    if c == "n"
        return [null, i + 4]
    end
    if c == "t"
        return [true, i + 4]
    end
    if c == "f"
        return [false, i + 5]
    end
    if c == "\""
        return _jParseStr(s, i, n)
    end
    if c == "["
        return _jParseArr(s, i, n)
    end
    if c == "{"
        return _jParseObj(s, i, n)
    end
    return _jParseNum(s, i, n)
end

func parseJson(s)
    var pair = _jParseVal(s, 0, len(s))
    return pair[0]
end
