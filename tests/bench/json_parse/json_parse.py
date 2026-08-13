import os
import sys

_LEVEL = os.environ.get("BENCH_LEVEL", "hard")
SCALE = 16 if _LEVEL == "easy" else 4 if _LEVEL == "medium" else 1
RECORDS = max(2500 // SCALE, 1)
REPS = 6

K_NULL = 0
K_BOOL = 1
K_INT = 2
K_STR = 3
K_LIST = 4
K_DICT = 5


class JNode:
    def __init__(self, kind):
        self.kind = kind
        self.num = 0
        self.text = ""
        self.items = []
        self.fields = {}


class Parser:
    def __init__(self, src):
        self.src = src
        self.at = 0
        self.n = len(src)

    def skip(self):
        while self.at < self.n and self.src[self.at] == " ":
            self.at += 1

    def value(self):
        self.skip()
        c = self.src[self.at]
        if c == "{":
            return self.object()
        if c == "[":
            return self.array()
        if c == '"':
            node = JNode(K_STR)
            node.text = self.text()
            return node
        if c == "t":
            self.at += 4
            node = JNode(K_BOOL)
            node.num = 1
            return node
        if c == "f":
            self.at += 5
            node = JNode(K_BOOL)
            node.num = 0
            return node
        if c == "n":
            self.at += 4
            return JNode(K_NULL)
        node = JNode(K_INT)
        node.num = self.integer()
        return node

    def text(self):
        self.at += 1
        start = self.at
        while self.src[self.at] != '"':
            self.at += 1
        out = self.src[start:self.at]
        self.at += 1
        return out

    def integer(self):
        acc = 0
        while self.at < self.n:
            c = self.src[self.at]
            if c < "0" or c > "9":
                break
            acc = acc * 10 + (ord(c) - 48)
            self.at += 1
        return acc

    def array(self):
        node = JNode(K_LIST)
        self.at += 1
        self.skip()
        if self.src[self.at] == "]":
            self.at += 1
            return node
        while True:
            node.items.append(self.value())
            self.skip()
            c = self.src[self.at]
            self.at += 1
            if c == "]":
                break
        return node

    def object(self):
        node = JNode(K_DICT)
        self.at += 1
        self.skip()
        if self.src[self.at] == "}":
            self.at += 1
            return node
        while True:
            self.skip()
            key = self.text()
            self.skip()
            self.at += 1
            node.fields[key] = self.value()
            self.skip()
            c = self.src[self.at]
            self.at += 1
            if c == "}":
                break
        return node


def walk(node):
    k = node.kind
    if k == K_INT:
        return node.num
    if k == K_BOOL:
        return node.num
    if k == K_STR:
        return len(node.text)
    if k == K_LIST:
        total = 0
        for item in node.items:
            total += walk(item)
        return total
    if k == K_DICT:
        total = 0
        for key, val in node.fields.items():
            total += len(key) + walk(val)
        return total
    return 0


def document(records):
    parts = []
    parts.append("[")
    s = 7
    i = 0
    while i < records:
        if i > 0:
            parts.append(", ")
        s = (s * 1103515245 + 12345) % 2147483648
        a = (s // 65536) % 1000
        b = s % 997
        flag = "true" if a % 2 == 0 else "false"
        parts.append(
            '{"id": %d, "name": "item%d", "tags": [%d, %d, %d], "ok": %s, "note": null}'
            % (i, a, a, b, a + b, flag)
        )
        i += 1
    parts.append("]")
    return "".join(parts)


def main():
    src = document(RECORDS)
    total = 0
    for _r in range(REPS):
        p = Parser(src)
        total = walk(p.value())
    print(len(src))
    print(total)


sys.setrecursionlimit(100000)
main()
