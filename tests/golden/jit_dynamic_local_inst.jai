#: gc-stress-every: 50
#: A dynamic local's read must confirm its specific class shape, not just
#: `Obj.type == OBJ_INSTANCE`.
#:
#: The `SLOT_INST` half of `jit_dynamic_local_list.jai`'s bug: a local
#: reassigned instances of two DIFFERENT classes across its lifetime is a
#: `dynamicLocal`, and `localTagFor` maps every non-scalar kind -- including
#: `SLOT_INST` regardless of WHICH class -- to the same `VAL_OBJ` tag. A read
#: compiled for `Dog`'s field layout therefore passed the tag guard just as
#: easily against a `Bird` instance, and read `Bird.wings` (a different
#: field, at `Dog.legs`'s byte offset) as if it were `Bird.legs`. Confirmed
#: as a real miscompile before this fix: JIT gave 4027584 against the
#: correct 160000 (`Bird.wings` was set to a deliberately distinctive wrong
#: value, 99, to make the corruption unmistakable rather than coincidentally
#: matching).
class Dog {
    pub var name: str
    pub var legs: int

    fn init(self, name: str) -> void {
        self.name = name
        self.legs = 4
    }
}

class Bird {
    pub var name: str
    pub var wings: int
    pub var legs: int

    fn init(self, name: str) -> void {
        self.name = name
        self.wings = 99
        self.legs = 2
    }
}

fn f(n: int, dog: any, bird: any) -> int {
    var a = dog
    var t = 0
    var i = 0
    while i < n {
        t = t + a.legs
        a = bird
        i = i + 1
    }
    return t
}

let d = Dog("Rex")
let b = Bird("Robin")
var total = 0
var k = 0
while k < 20000 {
    total = total + f(3, d, b)
    k = k + 1
}
print(total)
