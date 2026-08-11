class Vec2:
    __slots__ = ("x", "y")

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def add(self, other):
        return Vec2(self.x + other.x, self.y + other.y)

    def dot(self, other):
        return self.x * other.x + self.y * other.y

acc = Vec2(0.0, 0.0)
step = Vec2(1.0, 2.0)
total = 0.0
for _i in range(10_000_000):
    acc = acc.add(step)
    total += acc.dot(step)
print(f"{total:.1f}")
