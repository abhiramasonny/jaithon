func randInt(min, max)
    return floor(rand() * (max - min + 1)) + min
end

func randFloat(min, max)
    return rand() * (max - min) + min
end

func randBool()
    return rand() > 0.5
end

func randChoice(arr)
    var idx = randInt(0, _alen(arr) - 1)
    return arr[idx]
end

func shuffle(arr)
    var result = arrayCopy(arr)
    var i = _alen(result) - 1
    while i > 0
        var j = randInt(0, i)
        var temp = result[i]
        result[i] = result[j]
        result[j] = temp
        i = i - 1
    end
    return result
end

func randString(length)
    var chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    var result = ""
    var i = 0
    while i < length
        var idx = randInt(0, 61)
        result = concat(result, charAt(chars, idx))
        i = i + 1
    end
    return result
end
