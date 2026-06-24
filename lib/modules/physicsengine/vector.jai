func vec2(x, y)
    return [x, y]
end

func add(a, b)
    return [a[0] + b[0], a[1] + b[1]]
end

func sub(a, b)
    return [a[0] - b[0], a[1] - b[1]]
end

func scale(a, s)
    return [a[0] * s, a[1] * s]
end

func dot(a, b)
    return a[0] * b[0] + a[1] * b[1]
end

func length(a)
    return sqrt(dot(a, a))
end

func normalize(a)
    var len = length(a)
    if len == 0
        return [0, 0]
    end
    return scale(a, 1.0 / len)
end
