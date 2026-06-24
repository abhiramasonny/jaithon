func tryCall(f)
    try
        return f()
    except err do
        return null
    end
end

func tryCallOr(f, default_val)
    try
        return f()
    except err do
        return default_val
    end
end

func assertEq(a, b, msg)
    if not (a == b)
        raise concat("AssertionError: ", concat(msg, concat(" expected=", concat(_str(a), concat(" got=", _str(b))))))
    end
end

func assertTrue(cond, msg)
    if not cond
        raise concat("AssertionError: ", msg)
    end
end

func assertFalse(cond, msg)
    if cond
        raise concat("AssertionError: expected false but got true — ", msg)
    end
end

func assertNotNull(val, msg)
    if val == null
        raise concat("AssertionError: expected non-null — ", msg)
    end
end
