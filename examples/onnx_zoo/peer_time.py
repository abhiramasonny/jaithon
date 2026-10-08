"""The peers, at batch 1: onnxruntime's CPU provider on the exported file, and
eager torch on the MPS device on the same architecture (same seed, so the same
random weights). Warmed by the wall clock, then n samples of enough calls to
fill about 150 ms, median/min/max in ms -- the protocol zoo.jai uses.

    ./scripts/bench/gpu_lock.sh env VIRTUAL_ENV=~/.venvs/scratch uv run --active --offline \\
        --no-project --with onnx --with onnxruntime --with onnxscript --with torchvision \\
        python examples/onnx_zoo/peer_time.py resnet18 vit_tiny_dyn1

Results accumulate in $ONNX_ZOO_DIR/peer.json. `--table zoo.tsv` prints the
README's markdown table from zoo.jai's rows and those results.
"""
import json
import os
import re
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

WARM = float(os.environ.get("WARM", "1.5"))
N = int(os.environ.get("SAMPLES", "7"))


def timeit(fn):
    t0 = time.perf_counter()
    calls = 0
    while time.perf_counter() - t0 < WARM or calls < 3:
        fn()
        calls += 1
    per = (time.perf_counter() - t0) * 1000 / calls
    reps = max(1, int(150 / per)) if per < 150 else 1
    xs = []
    for _ in range(N):
        t = time.perf_counter()
        for _ in range(reps):
            fn()
        xs.append((time.perf_counter() - t) * 1000 / reps)
    xs.sort()
    return {"med": round(xs[len(xs) // 2], 3), "min": round(xs[0], 3), "max": round(xs[-1], 3), "n": N, "reps": reps}


def time_peers(tags):
    import numpy as np
    import torch
    import onnxruntime as ort
    import export as E

    out = os.path.join(E.OUT, "peer.json")
    res = json.load(open(out)) if os.path.exists(out) else {}
    specs = E.specs()
    for tag in tags:
        r = {}
        feeds = {}
        for line in open(os.path.join(E.OUT, tag + ".meta")):
            f = line.split()
            if f[0] == "input":
                rank = int(f[2])
                feeds[f[1]] = np.fromfile(f[3 + rank], np.float32).reshape([int(x) for x in f[3:3 + rank]])
        sess = ort.InferenceSession(os.path.join(E.OUT, tag + ".onnx"), providers=["CPUExecutionProvider"])
        kinds = {i.name: i.type for i in sess.get_inputs()}
        typed = {k: (v.astype(np.int64) if "int64" in kinds[k] else v) for k, v in feeds.items()}
        r["ort_cpu"] = timeit(lambda: sess.run(None, typed))
        del sess
        arch = re.sub(r"_dyn1?$", "", tag)
        if arch in specs:
            try:
                torch.manual_seed(0)
                np.random.seed(0)
                m = specs[arch][0]().eval().to("mps")
                targs = tuple(torch.from_numpy(typed[k]).to("mps") for k in feeds)

                def step():
                    with torch.no_grad():
                        m(*targs)
                    torch.mps.synchronize()
                r["torch_mps"] = timeit(step)
                del m
                torch.mps.empty_cache()
            except Exception as e:
                r["torch_mps"] = f"FAIL {type(e).__name__}: {str(e)[:200]}"
        res[tag] = r
        print(tag, json.dumps(r), flush=True)
        json.dump(res, open(out, "w"), indent=1)


def fmt(v):
    return "-" if v in ("-", None) else f"{float(v):.2f}"


def table(tsv):
    zoo = os.path.expanduser(os.environ.get("ONNX_ZOO_DIR", "~/.cache/jaithon/onnx_zoo"))
    peer_path = os.path.join(zoo, "peer.json")
    peers = json.load(open(peer_path)) if os.path.exists(peer_path) else {}
    print("| model | status | import ms | interp ms | compile ms | plan ms | torch MPS ms | ORT CPU ms | plan vs MPS | plan vs ORT | rel (interp / plan) |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for line in open(tsv):
        f = line.rstrip("\n").split("\t")
        if len(f) < 13:
            continue
        tag, status, imp, _cold, imed, _ilo, _ihi, comp, pmed, plo, phi, irel, prel = f[:13]
        p = peers.get(tag, {})
        mps = p.get("torch_mps", {})
        cpu = p.get("ort_cpu", {})
        mps_med = mps.get("med") if isinstance(mps, dict) else None
        cpu_med = cpu.get("med") if isinstance(cpu, dict) else None
        vs_mps = f"{mps_med / float(pmed):.2f}x" if mps_med and pmed != "-" else "-"
        vs_cpu = f"{cpu_med / float(pmed):.2f}x" if cpu_med and pmed != "-" else "-"
        spread = f" ({fmt(plo)}-{fmt(phi)})" if pmed != "-" else ""
        print(f"| {tag} | {status} | {fmt(imp)} | {fmt(imed)} | {fmt(comp)} | {fmt(pmed)}{spread} | "
              f"{fmt(mps_med)} | {fmt(cpu_med)} | {vs_mps} | {vs_cpu} | {irel} / {prel} |")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--table"]:
        table(sys.argv[2])
    else:
        time_peers(sys.argv[1:])
