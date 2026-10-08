"""Export randomly initialised real architectures to ONNX, record onnxruntime's
CPU answer for each, and census their operators.

The files go OUTSIDE the repository, to $ONNX_ZOO_DIR or ~/.cache/jaithon/onnx_zoo
(about 2.4 GB; a minute or two to regenerate). Nothing here is ever committed.

    VIRTUAL_ENV=~/.venvs/scratch uv run --active --offline --no-project \\
        --with onnx --with onnxruntime --with onnxscript --with torchvision \\
        python examples/onnx_zoo/export.py [--mode legacy|dyn|dyn1|all] [names...]

The overlay on the scratch venv is what supplies transformers (BERT, GPT-2,
Whisper); without it those rows are skipped with a note.

Three exporters per model, because they write different graphs:

- legacy: the TorchScript exporter at opset 17, constant folding on.
- dyn:    `torch.onnx.export(..., dynamo=True)` with every other setting at its
          default. That is what torch >= 2.9 does when nobody asks for anything,
          and it writes the weights to a `.onnx.data` file beside the model.
- dyn1:   dynamo again, with `external_data=False`, so the weights are inline.

For each exported file this writes `<tag>.onnx`, one `<tag>.in.<name>.bin` per
input and one `<tag>.out.<name>.bin` per output (float32, row-major, the values
onnxruntime CPU produced), and `<tag>.meta`, which is what zoo.jai reads.
`census.json` collects the operator counts and the export notes.
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort

warnings.filterwarnings("ignore")
OUT = os.path.expanduser(os.environ.get("ONNX_ZOO_DIR", "~/.cache/jaithon/onnx_zoo"))
torch.backends.mha.set_fastpath_enabled(False)  # legacy vit_b_16 cannot export the fused MHA


class UNet(nn.Module):
    """A reduced-width Ronneberger UNet: conv+BN+ReLU pairs, maxpool down, ConvTranspose (or bilinear) up."""

    def __init__(self, cin=3, cout=2, w=32, bilinear=False):
        super().__init__()

        def dbl(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                 nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        ch = [w, 2 * w, 4 * w, 8 * w, 16 * w]
        self.inc = dbl(cin, ch[0])
        self.downs = nn.ModuleList([dbl(ch[i], ch[i + 1]) for i in range(4)])
        self.ups = nn.ModuleList()
        self.upc = nn.ModuleList()
        for i in range(4, 0, -1):
            if bilinear:
                self.ups.append(nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                                              nn.Conv2d(ch[i], ch[i - 1], 1)))
            else:
                self.ups.append(nn.ConvTranspose2d(ch[i], ch[i - 1], 2, stride=2))
            self.upc.append(dbl(ch[i], ch[i - 1]))
        self.outc = nn.Conv2d(ch[0], cout, 1)

    def forward(self, x):
        xs = [self.inc(x)]
        for d in self.downs:
            xs.append(d(nn.functional.max_pool2d(xs[-1], 2)))
        y = xs.pop()
        for up, c in zip(self.ups, self.upc):
            y = c(torch.cat([xs.pop(), up(y)], dim=1))
        return self.outc(y)


class Pick(nn.Module):
    """One tensor out of a model that returns a dict or a ModelOutput."""

    def __init__(self, m, key="last_hidden_state"):
        super().__init__()
        self.m = m
        self.key = key

    def forward(self, *args):
        o = self.m(*args)
        return o[self.key] if isinstance(o, dict) or hasattr(o, self.key) else o[0]


def tv(name, **kw):
    import torchvision.models as M
    import torchvision.models.segmentation as SG
    import torchvision.models.detection as DT
    for mod in (M, SG, DT):
        if hasattr(mod, name):
            return getattr(mod, name)(weights=None, **kw)
    raise AttributeError(name)


def img(n=1, c=3, h=224, w=224):
    return [("input", np.random.randn(n, c, h, w).astype(np.float32))]


def tokens(vocab, seq=128, mask=True):
    ids = np.random.randint(0, vocab, size=(1, seq)).astype(np.int64)
    out = [("input_ids", ids)]
    if mask:
        out.append(("attention_mask", np.ones((1, seq), np.int64)))
    return out


#: name -> (builder, inputs, exporters). The 25 architectures, plus ssdlite,
#: which carries its own post-processing and is listed to show what is refused.
def specs():
    S = {}
    for n in ["resnet18", "resnet50", "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large",
              "efficientnet_b0", "squeezenet1_1", "shufflenet_v2_x1_0", "densenet121", "regnet_y_400mf",
              "convnext_tiny", "mnasnet1_0"]:
        S[n] = (lambda n=n: tv(n), img)
    S["googlenet"] = (lambda: tv("googlenet", aux_logits=False, init_weights=True), img)
    S["vit_tiny"] = (lambda: __import__("torchvision").models.VisionTransformer(
        image_size=224, patch_size=16, num_layers=12, num_heads=3, hidden_dim=192, mlp_dim=768), img)
    S["vit_b_16"] = (lambda: tv("vit_b_16"), img)
    S["swin_t"] = (lambda: tv("swin_t"), img)
    S["lraspp_mbv3"] = (lambda: Pick(tv("lraspp_mobilenet_v3_large", weights_backbone=None), "out"),
                        lambda: img(h=320, w=320))
    S["deeplabv3_mbv3"] = (lambda: Pick(tv("deeplabv3_mobilenet_v3_large", weights_backbone=None, aux_loss=False),
                                        "out"), lambda: img(h=320, w=320))
    S["fcn_resnet50"] = (lambda: Pick(tv("fcn_resnet50", weights_backbone=None, aux_loss=False), "out"),
                         lambda: img(h=320, w=320))
    S["unet"] = (lambda: UNet(), lambda: img(h=256, w=256))
    S["unet_bilinear"] = (lambda: UNet(bilinear=True), lambda: img(h=256, w=256))

    def bert_tiny():
        from transformers import BertConfig, BertModel
        c = BertConfig(hidden_size=128, num_hidden_layers=2, num_attention_heads=2, intermediate_size=512,
                       attn_implementation="eager")
        return Pick(BertModel(c))
    S["bert_tiny"] = (bert_tiny, lambda: tokens(30522))

    def distilbert():
        from transformers import DistilBertConfig, DistilBertModel
        return Pick(DistilBertModel(DistilBertConfig(attn_implementation="eager")))
    S["distilbert"] = (distilbert, lambda: tokens(30522))

    def gpt2_tiny():
        from transformers import GPT2Config, GPT2Model
        c = GPT2Config(n_layer=4, n_embd=256, n_head=4, n_positions=256, attn_implementation="eager", use_cache=False)
        return Pick(GPT2Model(c))
    S["gpt2_tiny"] = (gpt2_tiny, lambda: tokens(50257, seq=64, mask=False))

    def whisper_enc():
        from transformers import WhisperConfig, WhisperModel
        c = WhisperConfig(d_model=384, encoder_layers=4, encoder_attention_heads=6, decoder_layers=1,
                          decoder_attention_heads=6, encoder_ffn_dim=1536, decoder_ffn_dim=1536,
                          attn_implementation="eager")
        return Pick(WhisperModel(c).encoder)
    S["whisper_tiny_enc"] = (whisper_enc,
                             lambda: [("input_features", np.random.randn(1, 80, 3000).astype(np.float32))])

    S["ssdlite"] = (lambda: tv("ssdlite320_mobilenet_v3_large", weights_backbone=None), lambda: img(h=320, w=320))
    return S


#: Which models also get the two dynamo exports. GPT-2 no longer exports with
#: the legacy exporter (aten::diff), so dynamo is its only route.
DYNAMO = ["resnet18", "convnext_tiny", "efficientnet_b0", "mobilenet_v3_large", "unet_bilinear", "vit_tiny",
          "gpt2_tiny", "bert_tiny", "distilbert", "swin_t"]
LEGACY_SKIP = {"gpt2_tiny"}


def all_ops(graph, acc):
    for n in graph.node:
        key = n.op_type if n.domain in ("", "ai.onnx") else f"{n.domain}::{n.op_type}"
        acc[key] = acc.get(key, 0) + 1
        for a in n.attribute:
            if a.type == onnx.AttributeProto.GRAPH:
                all_ops(a.g, acc)
            for g in a.graphs:
                all_ops(g, acc)
    return acc


def calibrate(m, inputs):
    """Give every BatchNorm statistics measured on random input, and a random affine.

    A fresh BatchNorm has a running mean of 0 and variance of 1, and through a
    deep net that shrinks every activation towards zero: mobilenet_v2's logits
    came out near 3e-9 and googlenet's features near 2e-11, so HardSwish, SiLU
    and sigmoid never left their linear part and agreeing with onnxruntime
    proved little. Measured statistics keep the activations near unit scale.
    A batch of two, because training-mode BatchNorm over a 1x1 map refuses one.
    """
    norms = [x for x in m.modules() if isinstance(x, nn.modules.batchnorm._BatchNorm)]
    fed = [torch.from_numpy(a) for _, a in inputs]
    if not norms or not all(t.is_floating_point() for t in fed):
        return
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for bn in norms:
            bn.reset_running_stats()
            bn.momentum = None
            if bn.affine:
                bn.weight.uniform_(0.5, 1.5, generator=g)
                bn.bias.normal_(0.0, 0.2, generator=g)
        m.train()
        try:
            for _ in range(2):
                m(*[torch.randn((2,) + tuple(t.shape[1:]), generator=g) for t in fed])
        except Exception:
            pass  # a detector wants targets in training mode; its statistics stay at 0 and 1
    m.eval()


def export_one(name, build, inputs_fn, mode):
    tag = name + {"legacy": "", "dyn": "_dyn", "dyn1": "_dyn1"}[mode]
    path = os.path.join(OUT, tag + ".onnx")
    # One seed per model, so a re-export of one row reproduces the same weights and inputs.
    torch.manual_seed(0)
    np.random.seed(0)
    m = build().eval()
    if hasattr(m, "heads"):  # torchvision ViT zero-initialises its head: randomise it so outputs carry signal
        nn.init.normal_(m.heads.head.weight, std=0.02)
        nn.init.normal_(m.heads.head.bias, std=0.02)
    inputs = inputs_fn()
    calibrate(m, inputs)
    targs = tuple(torch.from_numpy(a) for _, a in inputs)
    names = [k for k, _ in inputs]
    rec = {"name": tag, "arch": name, "mode": mode, "params": int(sum(p.numel() for p in m.parameters()))}
    for stale in (path, path + ".data"):
        if os.path.exists(stale):
            os.remove(stale)
    t = time.time()
    try:
        with torch.no_grad():
            if mode == "legacy":
                torch.onnx.export(m, targs, path, input_names=names, output_names=["output"], opset_version=17,
                                  dynamo=False, do_constant_folding=True)
            elif mode == "dyn":
                torch.onnx.export(m, targs, path, input_names=names, output_names=["output"], dynamo=True)
            else:
                torch.onnx.export(m, targs, path, input_names=names, output_names=["output"], dynamo=True,
                                  external_data=False, opset_version=18)
    except Exception as e:
        rec["export_error"] = f"{type(e).__name__}: {str(e)[:300]}"
        return rec
    rec["export_s"] = round(time.time() - t, 2)
    model = onnx.load(path, load_external_data=False)
    rec["bytes"] = os.path.getsize(path) + (os.path.getsize(path + ".data") if os.path.exists(path + ".data") else 0)
    rec["external_data"] = os.path.exists(path + ".data")
    rec["opset"] = [(o.domain, o.version) for o in model.opset_import]
    rec["ops"] = all_ops(model.graph, {})
    rec["nodes"] = sum(rec["ops"].values())
    with torch.no_grad():
        tout = m(*targs)
        if isinstance(tout, (list, tuple)):
            tout = tout[0]
        if isinstance(tout, dict):
            tout = list(tout.values())[0]
        tout = tout.numpy() if hasattr(tout, "numpy") else None
    try:
        t = time.time()
        sess = ort.InferenceSession(path, ort.SessionOptions(), providers=["CPUExecutionProvider"])
        rec["ort_load_ms"] = round((time.time() - t) * 1000, 1)
        outs = sess.run(None, {k: a for k, a in inputs})
        out_names = [o.name for o in sess.get_outputs()]
    except Exception as e:
        rec["ort_error"] = f"{type(e).__name__}: {str(e)[:300]}"
        return rec
    ref = outs[0]
    if tout is not None and tout.shape == ref.shape:
        rec["torch_vs_ort_maxabs"] = float(np.abs(tout - ref).max())
    meta = [f"model {path}"]
    peak = float(np.abs(ref).max())
    if "torch_vs_ort_maxabs" in rec and peak > 0:
        # How far two CPU runtimes already are from each other on this model,
        # as zoo.jai measures it: no third float32 runtime can be held closer.
        rec["torch_vs_ort_rel"] = rec["torch_vs_ort_maxabs"] / peak
        meta.append(f"agreement {rec['torch_vs_ort_rel']!r}")
    for k, a in inputs:
        f = os.path.join(OUT, f"{tag}.in.{k}.bin")
        a.astype(np.float32).tofile(f)
        meta.append(f"input {k} {a.ndim} {' '.join(map(str, a.shape))} {f}")
    for k, a in zip(out_names, outs):
        f = os.path.join(OUT, f"{tag}.out.{k}.bin")
        a.astype(np.float32).tofile(f)
        meta.append(f"output {k} {a.ndim} {' '.join(map(str, a.shape))} {f}")
        rec.setdefault("outputs", []).append([k, list(a.shape), float(np.abs(a).max())])
    with open(os.path.join(OUT, tag + ".meta"), "w") as fh:
        fh.write("\n".join(meta) + "\n")
    return rec


def main(argv):
    mode = "all"
    if argv[:1] == ["--mode"]:
        mode, argv = argv[1], argv[2:]
    modes = ["legacy", "dyn", "dyn1"] if mode == "all" else [mode]
    S = specs()
    want = argv or list(S)
    for n in want:
        if n not in S:
            sys.exit(f"unknown model {n}; known: {' '.join(S)}")
    os.makedirs(OUT, exist_ok=True)
    census_path = os.path.join(OUT, "census.json")
    census = json.load(open(census_path)) if os.path.exists(census_path) else {}
    for n in want:
        for md in modes:
            if md == "legacy" and n in LEGACY_SKIP:
                continue
            if md != "legacy" and n not in DYNAMO:
                continue
            b, i = S[n]
            t = time.time()
            try:
                r = export_one(n, b, i, md)
            except ImportError as e:
                r = {"name": n + {"legacy": "", "dyn": "_dyn", "dyn1": "_dyn1"}[md], "arch": n, "mode": md,
                     "export_error": f"ImportError: {e} (run with the scratch-venv overlay for transformers)"}
            r["wall_s"] = round(time.time() - t, 1)
            census[r["name"]] = r
            print(json.dumps({k: v for k, v in r.items() if k != "ops"}), flush=True)
            json.dump(census, open(census_path, "w"), indent=1)
    # The list zoo.jai walks when asked for every model: every tag with a reference answer.
    tags = sorted(t for t in census if os.path.exists(os.path.join(OUT, t + ".meta")))
    with open(os.path.join(OUT, "models.txt"), "w") as fh:
        fh.write("\n".join(tags) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
