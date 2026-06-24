var _testPassed = 0
var _testFailed = 0
var _testSuite = ""

func describe(name, fn)
    var prev = _testSuite
    _testSuite = name
    fn()
    _testSuite = prev
end

func it(name, fn)
    try
        fn()
        _testPassed = _testPassed + 1
        print concat(concat("[PASS] ", _testSuite), concat(": ", name))
    except err do
        _testFailed = _testFailed + 1
        print concat(concat("[FAIL] ", _testSuite), concat(": ", name))
        print concat("       ", err)
    end
end

func expect(val)
    var _expectVal = val
    var _expectPass = true

    func toBe(expected)
        if not (_expectVal == expected)
            raise concat(concat("Expected ", _str(expected)), concat(" but got ", _str(_expectVal)))
        end
    end

    func toNotBe(expected)
        if _expectVal == expected
            raise concat("Expected value to not be ", _str(expected))
        end
    end

    func toBeTrue()
        if not _expectVal
            raise concat("Expected true but got ", _str(_expectVal))
        end
    end

    func toBeFalse()
        if _expectVal
            raise concat("Expected false but got ", _str(_expectVal))
        end
    end

    func toBeNull()
        if _expectVal != null
            raise concat("Expected null but got ", _str(_expectVal))
        end
    end

    func toBeGreaterThan(n)
        if not (_expectVal > n)
            raise concat(concat(_str(_expectVal), " is not greater than "), _str(n))
        end
    end

    func toBeLessThan(n)
        if not (_expectVal < n)
            raise concat(concat(_str(_expectVal), " is not less than "), _str(n))
        end
    end

    func toContain(item)
        if _type(_expectVal) == "array"
            if not arrayContains(_expectVal, item)
                raise concat("Array does not contain ", _str(item))
            end
        else
            if _type(_expectVal) == "string"
                if not contains_str(_expectVal, _str(item))
                    raise concat(_str(_expectVal), concat(" does not contain ", _str(item)))
                end
            else
                raise "toContain only works on arrays and strings"
            end
        end
    end

    func toHaveLength(n)
        var l = len(_expectVal)
        if l != n
            raise concat(concat("Expected length ", _str(n)), concat(" but got ", _str(l)))
        end
    end

    var result = _dict()
    result["toBe"] = toBe
    result["toNotBe"] = toNotBe
    result["toBeTrue"] = toBeTrue
    result["toBeFalse"] = toBeFalse
    result["toBeNull"] = toBeNull
    result["toBeGreaterThan"] = toBeGreaterThan
    result["toBeLessThan"] = toBeLessThan
    result["toContain"] = toContain
    result["toHaveLength"] = toHaveLength
    return result
end

func testSummary()
    var total = _testPassed + _testFailed
    print concat(concat(concat("\n=== Test Results: ", _str(_testPassed)), concat("/", _str(total))), " passed ===")
    if _testFailed > 0
        print concat(_str(_testFailed), " tests FAILED")
    else
        print "All tests passed!"
    end
    return _testFailed == 0
end

func resetTests()
    _testPassed = 0
    _testFailed = 0
end
