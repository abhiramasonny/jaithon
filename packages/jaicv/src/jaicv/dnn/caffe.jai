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
from .caffe.layers import translated_layers
from .caffe.message import Message
from .caffe.net import free_caffe_weights, read_net_from_caffe, read_net_from_caffe_bytes
from .caffe.text import parse_net_text

export { Message, free_caffe_weights, parse_net_text, translated_layers }
export { read_net_from_caffe, read_net_from_caffe_bytes }
