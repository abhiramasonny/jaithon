#cons cell based linked lists

func cons(a, b)
    return _cell(a, b)
end

func car(pair)
    return _car(pair)
end

func cdr(pair)
    return _cdr(pair)
end

func setCar(pair, val)
    return _setcar(pair, val)
end

func setCdr(pair, val)
    return _setcdr(pair, val)
end

func isPair(x)
    return type(x) == "cell"
end

#list ops
func list()
    return null
end

func isEmpty(lst)
    return lst == null
end

func isNull(x)
    return x == null
end

func notNull(x)
    if x == null
        return false
    end
    return true
end

func head(lst)
    return _car(lst)
end

func tail(lst)
    return _cdr(lst)
end

func prepend(elem, lst)
    return _cell(elem, lst)
end

func listLen(lst)
    var count = 0
    var current = lst
    while notNull(current)
        count = count + 1
        current = _cdr(current)
    end
    return count
end

func listGet(lst, idx)
    var current = lst
    var i = 0
    while i < idx
        if current == null
            return null
        end
        current = _cdr(current)
        i = i + 1
    end
    if current == null
        return null
    end
    return _car(current)
end

func listSet(lst, idx, val)
    var current = lst
    var i = 0
    while i < idx
        if current == null
            return null
        end
        current = _cdr(current)
        i = i + 1
    end
    if current == null
        return null
    end
    _setcar(current, val)
    return val
end

func append(lst, elem)
    if lst == null
        return _cell(elem, null)
    end
    var current = lst
    while notNull(_cdr(current))
        current = _cdr(current)
    end
    _setcdr(current, _cell(elem, null))
    return lst
end

func last(lst)
    if lst == null
        return null
    end
    var current = lst
    while notNull(_cdr(current))
        current = _cdr(current)
    end
    return _car(current)
end

func reverse(lst)
    var result = null
    var current = lst
    while notNull(current)
        result = _cell(_car(current), result)
        current = _cdr(current)
    end
    return result
end

#functions
func map(lst, f)
    if lst == null
        return null
    end
    return _cell(f(_car(lst)), map(_cdr(lst), f))
end

func filter(lst, pred)
    if lst == null
        return null
    end
    if pred(_car(lst))
        return _cell(_car(lst), filter(_cdr(lst), pred))
    end
    return filter(_cdr(lst), pred)
end

func foldl(lst, init, f)
    var acc = init
    var current = lst
    while notNull(current)
        acc = f(acc, _car(current))
        current = _cdr(current)
    end
    return acc
end

func contains(lst, elem)
    var current = lst
    while notNull(current)
        if _car(current) == elem
            return true
        end
        current = _cdr(current)
    end
    return false
end

func listConcat(lst1, lst2)
    if lst1 == null
        return lst2
    end
    return _cell(_car(lst1), listConcat(_cdr(lst1), lst2))
end

func take(lst, n)
    if n < 1
        return null
    end
    if lst == null
        return null
    end
    return _cell(_car(lst), take(_cdr(lst), n - 1))
end

func drop(lst, n)
    if n < 1
        return lst
    end
    if lst == null
        return null
    end
    return drop(_cdr(lst), n - 1)
end

func range(start, stop)
    if start >= stop
        return null
    end
    return _cell(start, range(start + 1, stop))
end

func _add(a, b)
    return a + b
end

func _mul(a, b)
    return a * b
end

func sum(lst)
    return foldl(lst, 0, _add)
end

func product(lst)
    return foldl(lst, 1, _mul)
end
