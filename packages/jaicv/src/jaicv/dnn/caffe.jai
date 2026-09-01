#: Reading a Caffe model.
#:
#: Caffe stores a network twice over. The `.prototxt` is one `NetParameter` in
#: protobuf's *text* format and holds the architecture; the `.caffemodel` is
#: the same message in the *binary* format with the weights filled in. Callers
#: have both, so both are read here and merged by layer name -- the prototxt
#: gives the structure, and a caffemodel layer's blobs replace whatever the
#: prototxt carried for it, which is the order Caffe itself uses.
#:
#: The binary half is `protobuf.jai`'s job. The text half is a parser in this
#: file, and it is the part a real prototxt is most likely to break: they are
#: hand-written files, so they arrive with comments in the middle of a message,
#: angle brackets instead of braces, a colon before a nested message, commas
#: and semicolons between fields, hex and exponent numbers, bare enum
#: identifiers, escapes in a quoted name, and -- most often -- fields this
#: importer has no use for. Unknown fields are skipped by syntax rather than by
#: schema, so a `weight_filler`, a `param { lr_mult: 1 }` or a `transform_param`
#: costs nothing. Both halves land in the same `Message` tree, so everything
#: below this line is written once and does not know which file it came from.
#:
#: Three things about the translation are worth naming.
#:
#: Caffe's `BatchNorm` blobs are not the mean and the variance. They are the
#: mean and the variance each multiplied by the scale factor in the third blob,
#: and the layer divides by it on the way in -- with a zero factor meaning
#: "there are no statistics", not a division by zero. A reader that hands the
#: blobs straight to a normaliser is wrong by whatever that factor happens to
#: be, which for a trained net is the number of batches it saw. Caffe also
#: splits into `BatchNorm` and `Scale` what ONNX fuses into one operator, so an
#: immediately following `Scale` is folded in here.
#:
#: Caffe pools with `ceil` and then, if there is any padding, drops the last
#: window when it would begin inside the padding; and its average pool divides
#: by the window clipped to the *padded* image, which is neither of ONNX's two
#: `count_include_pad` settings when a window hangs off the end. Both are got
#: right by computing the output extent here and handing the executor explicit
#: padding, which needs the input shape -- so this module infers shapes as it
#: goes. Where a shape cannot be inferred it falls back to `ceil_mode`, which
#: differs only by that one trailing window.
#:
#: A Caffe layer may write the blob it reads, and most deploy prototxts do it
#: for every activation. The graph the executor runs needs one producer per
#: name, so each rewrite gets a fresh name -- except the last, which keeps the
#: blob's own name. `graph.inputs` and `graph.outputs` are therefore Caffe's
#: names exactly, and the invented ones never surface.
#:
#: ```
#: let graph = read_net_from_caffe("deploy.prototxt", "weights.caffemodel")
#: defer { free_caffe_weights(graph) }
#: let outputs = run_graph(graph, {"data": blob})
#: ```
import std.io as io
from .graph import Graph
from .caffe.binary import decode_message
from .caffe.layers_filter import layer_conv, layer_inner_product, layer_pool
from .caffe.layers_norm import layer_activation, layer_affine, layer_batch_norm, layer_lrn, layer_prelu
from .caffe.layers_shape import layer_concat, layer_eltwise, layer_flatten, layer_power, layer_reshape, layer_slice, layer_softmax, layer_split
from .caffe.message import Message
from .caffe.reading import Reading, Step
from .caffe.text import parse_net_text

export { Message, parse_net_text }

var _layers: dict[str, fn(Reading, Step) -> void] = {}

#: Every Caffe layer type this importer translates, and what translates it.
fn translators() -> dict[str, fn(Reading, Step) -> void] {
    if _layers.len() > 0 { return _layers }
    _layers["Convolution"] = layer_conv
    _layers["Deconvolution"] = layer_conv
    _layers["Pooling"] = layer_pool
    _layers["InnerProduct"] = layer_inner_product
    _layers["ReLU"] = layer_activation
    _layers["ELU"] = layer_activation
    _layers["Sigmoid"] = layer_activation
    _layers["TanH"] = layer_activation
    _layers["AbsVal"] = layer_activation
    _layers["BNLL"] = layer_activation
    _layers["Dropout"] = layer_activation
    _layers["PReLU"] = layer_prelu
    _layers["LRN"] = layer_lrn
    _layers["BatchNorm"] = layer_batch_norm
    _layers["Scale"] = layer_affine
    _layers["Bias"] = layer_affine
    _layers["Concat"] = layer_concat
    _layers["Slice"] = layer_slice
    _layers["Softmax"] = layer_softmax
    _layers["Eltwise"] = layer_eltwise
    _layers["Flatten"] = layer_flatten
    _layers["Reshape"] = layer_reshape
    _layers["Power"] = layer_power
    _layers["Split"] = layer_split
    return _layers
}

#: The Caffe layer types this importer translates, in alphabetical order.
#:
#: Everything else -- the data layers, the loss layers, and the types a fork of
#: Caffe added to its own copy of the schema -- is refused by name rather than
#: dropped, because a graph quietly missing a step is worse than no graph.
pub fn translated_layers() -> list[str] {
    var names: list[str] = []
    for (name, _handler) in translators().items() { names.push(name) }
    names.sort()
    return names
}

fn convert(reading: Reading, step: Step) -> void {
    let handler = translators().get(step.kind, null)
    if handler is null {
        throw ValueError(f"{step.label()} is of a type this importer does not translate")
    }
    if step.tops.len() == 0 { throw ValueError(f"{step.label()} writes nothing") }
    if step.bottoms.len() == 0 { throw ValueError(f"{step.label()} reads nothing") }
    handler(reading, step)
}

#: Whether a `Scale` may be folded into the `BatchNorm` in front of it.
#:
#: It may when the `Scale` is the learned-parameter form over the channel axis
#: and the value it consumes reaches nobody else. In place -- which is how a
#: deploy prototxt almost always writes the pair -- later layers read the same
#: blob name and so read the scaled value either way, which is what the fused
#: operator produces.
fn foldable(norm: Message, scale: Message, last_read: dict[str, int], position: int) -> bool {
    if scale.str_of("type", "") != "Scale" { return false }
    let tops = norm.strs_of("top")
    let bottoms = scale.strs_of("bottom")
    let written = scale.strs_of("top")
    if tops.len() != 1 or bottoms.len() != 1 or written.len() != 1 { return false }
    if bottoms[0] != tops[0] { return false }
    let param = scale.param("scale_param", "ScaleParameter")
    if param.int_of("axis", 1) != 1 or param.int_of("num_axes", 1) != 1 { return false }
    if written[0] == bottoms[0] { return true }
    return last_read.get(tops[0], -1) <= position + 1
}

#: The layers, with folded pairs collapsed and `Input` layers taken out.
fn plan(layers: list[Message], declared: dict[str, list[int]], order: list[str]) -> list[Step] {
    #: The last layer to read each blob, so that a fold can ask in one lookup
    #: whether the value it would swallow is wanted below the pair.
    var last_read: dict[str, int] = {}
    for (position, layer) in layers.enumerate() {
        for blob in layer.strs_of("bottom") { last_read[blob] = position }
    }

    var steps: list[Step] = []
    var index = 0
    while index < layers.len() {
        let layer = layers[index]
        let kind = layer.str_of("type", "")
        let name = layer.str_of("name", "")
        if kind.len() == 0 {
            throw ValueError(f"the layer {name} does not say what type it is")
        }
        if kind == "Input" {
            let shapes = layer.param("input_param", "InputParameter").children_of("shape")
            for (slot, top) in layer.strs_of("top").enumerate() {
                if not top in order { order.push(top) }
                if shapes.len() == 0 { continue }
                if shapes.len() != 1 and slot >= shapes.len() {
                    throw ValueError(f"the Input layer {name} names more tops than shapes")
                }
                declared[top] = shapes[if shapes.len() == 1 { 0 } else { slot }].ints_of("dim")
            }
            index += 1
            continue
        }
        var scale: Message? = null
        if kind == "BatchNorm" and index + 1 < layers.len() {
            if foldable(layer, layers[index + 1], last_read, index) { scale = layers[index + 1] }
        }
        let tops = if scale is null {
            layer.strs_of("top")
        } else {
            let folded: Message = scale
            folded.strs_of("top")
        }
        steps.push(Step(layer, kind, name, layer.strs_of("bottom"), tops, scale: scale))
        index += if scale is null { 1 } else { 2 }
    }
    return steps
}

#: The inputs a NetParameter declares outside its layers, in either the
#: `input_shape` form or the four-numbers-per-blob one that came before it.
fn declared_inputs(net: Message, declared: dict[str, list[int]], order: list[str]) -> void {
    let names = net.strs_of("input")
    let shapes = net.children_of("input_shape")
    let flat = net.ints_of("input_dim")
    for (index, name) in names.enumerate() {
        order.push(name)
        if index < shapes.len() {
            declared[name] = shapes[index].ints_of("dim")
        } elif flat.len() == 4 * names.len() {
            declared[name] = [flat[4 * index + axis] for axis in 0..4]
        }
    }
}

fn assemble(net: Message) -> Graph {
    if net.children_of("layers").len() > 0 {
        throw ValueError(
            "this file uses the V1 `layers` field, which Caffe replaced with `layer` in " +
                "2015 and this importer does not read"
        )
    }
    var declared: dict[str, list[int]] = {}
    var order: list[str] = []
    declared_inputs(net, declared, order)
    let steps = plan(net.children_of("layer"), declared, order)

    let reading = Reading()
    for name in order {
        reading.taken[name] = true
        reading.declared[name] = true
        reading.live[name] = name
        reading.inputs.push(name)
        let shape = declared.get(name, null)
        if shape is not null {
            let dims: list[int] = shape
            if dims.len() > 0 { reading.shapes[name] = dims }
        }
    }
    for step in steps {
        for blob in step.bottoms { reading.taken[blob] = true }
        for blob in step.tops {
            reading.taken[blob] = true
            reading.expected[blob] = reading.expected.get(blob, 0) + 1
        }
    }

    try {
        for step in steps { convert(reading, step) }
    } catch failure: Error {
        reading.discard()
        throw failure
    }

    #: What comes out is every blob whose last value nothing read. Because the
    #: last write keeps the blob's own name, these are Caffe's names.
    var outputs: list[str] = []
    for name in reading.finished {
        if not reading.consumed.get(name, false) and not name in outputs { outputs.push(name) }
    }
    return Graph(
        nodes: reading.nodes,
        inputs: reading.inputs,
        outputs: outputs,
        initializers: reading.weights
    )
}

#: Put the caffemodel's blobs into the prototxt's layers, by name.
fn merge_weights(net: Message, trained: Message) -> void {
    var byname: dict[str, Message] = {}
    for layer in trained.children_of("layer") { byname[layer.str_of("name", "")] = layer }
    if trained.children_of("layers").len() > 0 {
        throw ValueError(
            "this caffemodel uses the V1 `layers` field, which Caffe replaced with `layer` " +
                "in 2015 and this importer does not read"
        )
    }
    for layer in net.children_of("layer") {
        let name = layer.str_of("name", "")
        let found = byname.get(name, null)
        if found is null { continue }
        let source: Message = found
        let blobs = source.children_of("blobs")
        if blobs.len() == 0 { continue }
        layer.children["blobs"] = blobs
    }
}

#: Free every tensor an imported graph holds.
#:
#: The importer allocates the initializers and nothing else, so this is the
#: whole of a model's GPU footprint.
pub fn free_caffe_weights(graph: Graph) -> void {
    for (_name, tensor) in graph.initializers.items() { tensor.free() }
}

#: Read a Caffe network from a prototxt already in memory, and optionally the
#: caffemodel that goes with it.
#:
#: Without the caffemodel the graph still has every node, and the weights it
#: could not find become names in `graph.inputs` for the caller to feed.
pub fn read_net_from_caffe_bytes(prototxt: bytes, caffemodel: bytes? = null) -> Graph {
    let net = parse_net_text(__prim__.str_decode(prototxt))
    if caffemodel is not null {
        let raw: bytes = caffemodel
        merge_weights(net, decode_message("NetParameter", raw, 0))
    }
    return assemble(net)
}

#: The same, from paths.
pub fn read_net_from_caffe(prototxt: str, caffemodel: str? = null) -> Graph {
    let text = io.read_bytes(prototxt)
    if caffemodel is null { return read_net_from_caffe_bytes(text) }
    let path: str = caffemodel
    return read_net_from_caffe_bytes(text, io.read_bytes(path))
}
