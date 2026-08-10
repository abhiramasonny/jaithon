class Point:
    __slots__ = ("x", "y")

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def dot(self, other):
        return self.x * other.x + self.y * other.y


total = 0
i = 0
while i < 400_000:
    a = Point(i % 100, i % 37)
    b = Point(i % 53, i % 11)
    total = (total + a.dot(b)) % 1000000007
    i += 1
print(total)
