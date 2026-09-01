#: A layer's weights, out of the `BlobProto`s that carry them.
from .message import Message

#: One `BlobProto`: the numbers, and whatever shape the file put on them.
pub class Blob {
    pub let shape: list[int]
    pub let values: list[float]

    pub fn init(self, shape: list[int], values: list[float]) {
        self.shape = shape
        self.values = values
    }
}

#: Read a blob, in either the modern `shape` form or the four-number one that
#: Caffe deprecated but every older caffemodel still uses.
fn read_blob(message: Message, where: str) -> Blob {
    var values = message.floats_of("data")
    if values.len() == 0 { values = message.floats_of("double_data") }
    var shape: list[int] = []
    let declared = message.child("shape")
    if declared is not null {
        let holder: Message = declared
        shape = holder.ints_of("dim")
    } elif message.has("num") or
        message.has("channels") or
        message.has("height") or
        message.has("width") {
        shape = [
            message.int_of("num", 0),
            message.int_of("channels", 0),
            message.int_of("height", 0),
            message.int_of("width", 0),
        ]
    }
    if shape.len() > 0 {
        var volume = 1
        for dimension in shape { volume *= dimension }
        if volume != values.len() {
            throw ValueError(
                f"{where} says its shape is {shape} and carries {values.len()} numbers"
            )
        }
    }
    return Blob(shape, values)
}

#: A layer's blobs, in the order Caffe numbers them. By the time this runs the
#: caffemodel's blobs have already replaced whatever the prototxt held.
pub fn blobs_of(layer: Message, step_name: str) -> list[Blob] {
    var out: list[Blob] = []
    for (index, held) in layer.children_of("blobs").enumerate() {
        out.push(read_blob(held, f"blob {index} of the layer {step_name}"))
    }
    return out
}
