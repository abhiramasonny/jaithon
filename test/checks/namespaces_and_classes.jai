# Namespaces (Java-style) working with classes and shared state
# Expected output:
# namespace
# 2
# 8
# 8
# 13
# 3
# 12
# 18
# 100

var metrics_scale = 2
var metrics_offset = 0
var metrics_extra = 0

public var mult in metrics(var x)
    return x * metrics_scale
end

public var getScale in metrics()
    return metrics_scale
end

public var setScale in metrics(var v)
    metrics_scale = v
end

public var getOffset in metrics()
    return metrics_offset
end

public var setOffset in metrics(var v)
    metrics_offset = v
end

public var setExtra in metrics(var v)
    metrics_extra = v
end

public var getExtra in metrics()
    return metrics_extra
end

class Accumulator
    var base

    func init(self, base)
        self.base = base
    end

    func add(self, x)
        return self.base + x
    end

    func addScaled(self, x)
        return self.base + (metrics_scale * x) + metrics_offset
    end
end

var acc = new Accumulator(5)
print type(metrics)
print metrics.getScale()
print metrics.mult(4)
print acc.add(3)
print acc.addScaled(4)

metrics.setScale(3)
metrics.setOffset(1)
print metrics.getScale()
print metrics.mult(4)
print acc.addScaled(4)
metrics.setExtra(100)
print metrics.getExtra()
