#: Which Caffe layer type each translator answers for.
from .layers_filter import layer_conv, layer_inner_product, layer_pool
from .layers_norm import layer_activation, layer_affine, layer_batch_norm, layer_lrn, layer_prelu
from .layers_shape import layer_concat, layer_eltwise, layer_flatten, layer_power, layer_reshape, layer_slice, layer_softmax, layer_split
from .reading import Reading, Step

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

pub fn convert(reading: Reading, step: Step) -> void {
    let handler = translators().get(step.kind, null)
    if handler is null {
        throw ValueError(f"{step.label()} is of a type this importer does not translate")
    }
    if step.tops.len() == 0 { throw ValueError(f"{step.label()} writes nothing") }
    if step.bottoms.len() == 0 { throw ValueError(f"{step.label()} reads nothing") }
    handler(reading, step)
}
