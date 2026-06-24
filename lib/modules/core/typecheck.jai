func isNull(x)
    return _type(x) == "null"
end

func isNumber(x)
    var t = _type(x)
    return t == "number" or t == "int" or t == "long" or t == "float" or t == "double" or t == "short" or t == "byte"
end

func isString(x)
    return _type(x) == "string"
end

func isBool(x)
    return _type(x) == "bool"
end

func isArray(x)
    return _type(x) == "array"
end

func isObject(x)
    return _type(x) == "object"
end

func isFunction(x)
    return _type(x) == "function" or _type(x) == "native"
end

func isCell(x)
    return _type(x) == "cell"
end

func isChar(x)
    return _type(x) == "char"
end

func isIterable(x)
    return isArray(x) or isString(x)
end

func isPrimitive(x)
    var t = _type(x)
    return t == "number" or t == "int" or t == "long" or t == "float" or t == "double" or t == "short" or t == "byte" or t == "bool" or t == "string" or t == "char" or t == "null"
end

func requireString(x, name)
    if not isString(x)
        raise concat(concat(name, " must be string, got "), _type(x))
    end
end

func requireNumber(x, name)
    if not isNumber(x)
        raise concat(concat(name, " must be number, got "), _type(x))
    end
end

func requireArray(x, name)
    if not isArray(x)
        raise concat(concat(name, " must be array, got "), _type(x))
    end
end

func coerce(val, typeName)
    if typeName == "string"
        return _str(val)
    end
    if typeName == "number"
        return _num(val)
    end
    if typeName == "int"
        return _int(val)
    end
    if typeName == "bool"
        return _bool(val)
    end
    return val
end

func defaultVal(val, fallback)
    if isNull(val)
        return fallback
    end
    return val
end

func orElse(val, fallback)
    if isNull(val) or val == false or val == 0 or val == ""
        return fallback
    end
    return val
end
