"""B13: measure the section 2.8 channel regrouping on the RTL.

Section 2.8's gains are computed with tile_geometry.cycles(), the RTL's cycle
accounting. This runs real operands through the RTL itself. For each case:

  1. take a real slice of a real pruned layer -- A rows from one input, a
     contiguous range of k, a contiguous range of P output channels -- at a
     shape the RTL's buffers hold (M*K <= 256, K*P <= 512);
  2. regroup its output channels with the same greedy_tiles() section 2.8 uses;
  3. run original and regrouped through tb_regroup.sv, skipping off and on.

It checks that every output is correct (in the bench), that the regrouped C is
the original C with its columns permuted, and that the measured compute cycles
equal the accounting up to one fixed offset shared by every case -- which is
what makes section 2.8's arithmetic a measurement rather than a model.

Usage: python experiments/regroup_rtl.py      (~5 min; needs iverilog)
Writes results/regroup_rtl.csv.
"""
import os, subprocess, sys, tempfile
import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import tile_geometry as tg                                    # noqa: E402
from profiler.workload_profiler import _operand_a, _operand_b  # noqa: E402

RTL = os.path.join(ROOT, "idea2_veriflex", "rtl")
SHAPES = [(16, 16, 32), (32, 8, 64)]          # (M, K, P): both fill IBUF / WBUF
SLICES_PER_LAYER = 3
LAYERS = {
    "resnet50_pruned90": ["layer1.0.conv1", "layer2.1.conv1", "layer3.2.conv1",
                          "layer3.4.conv3", "layer4.0.conv1"],
    "bert_pruned90": ["encoder.layer.2.intermediate.dense", "encoder.layer.6.output.dense",
                      "encoder.layer.9.attention.self.value"],
}
torch.set_grad_enabled(False)


def capture_values(model, names, x, call):
    """Real (A, B) float operands for the named layers, from one input."""
    got, hooks = {}, []
    for n, m in model.named_modules():
        if n in names:
            def fn(mod, inp, out, n=n):
                got[n] = (_operand_a(mod, inp[0]).float().numpy(),
                          _operand_b(mod).float().numpy())
            hooks.append(m.register_forward_hook(fn))
    call(model, x)
    for h in hooks:
        h.remove()
    return got


def quantise(x):
    """To int16 with max |x| -> 127, keeping zeros zero and nonzeros nonzero."""
    s = 127.0 / max(np.abs(x).max(), 1e-12)
    q = np.round(x * s)
    q[(x != 0) & (q == 0)] = np.sign(x[(x != 0) & (q == 0)])
    return q.astype(np.int64)


def write_hex(path, m):
    with open(path, "w") as f:
        for v in m.ravel():
            f.write(f"{int(v) & 0xFFFF:04x}\n")


def build(tmp, M, K, P):
    exe = os.path.join(tmp, f"tb_{M}_{K}_{P}")
    subprocess.run(["iverilog", "-g2012", "-o", exe,
                    f'-DCPE_WEIGHT_DIR="{RTL}/weights"',
                    f"-DRG_M={M}", f"-DRG_K={K}", f"-DRG_P={P}",
                    os.path.join(RTL, "common", "versaaccel_pkg.sv"),
                    *sorted(os.path.join(RTL, "versaaccel_rtl", f)
                            for f in os.listdir(os.path.join(RTL, "versaaccel_rtl"))
                            if f.endswith(".sv")),
                    os.path.join(RTL, "testbenches", "tb_regroup.sv")],
                   check=True, capture_output=True)   # iverilog's "sorry:" notes are benign
    return exe


def simulate(exe, tmp, A, B, tag):
    fa, fb = os.path.join(tmp, f"{tag}_A.hex"), os.path.join(tmp, f"{tag}_B.hex")
    write_hex(fa, A); write_hex(fb, B)
    out = subprocess.run(["vvp", "-n", exe, f"+A={fa}", f"+B={fb}"],
                         capture_output=True, text=True, check=True).stdout
    res = [l for l in out.splitlines() if l.startswith("RESULT")][0].split()
    if len(res) != 5:
        raise RuntimeError(out[-2000:])
    c_off, c_on, checks, errors = map(int, res[1:])
    C = np.zeros((A.shape[0], B.shape[1]), np.int64)
    for l in out.splitlines():
        if l.startswith("C "):
            _, r, c, v = l.split()
            C[int(r), int(c)] = int(v)
    return c_off, c_on, checks, errors, C


def main():
    import pandas as pd
    rng = np.random.default_rng(0)
    img = tg.resnet_inputs(1)[0]
    txt = tg.bert_inputs(1)[0]
    models = {n: m for gen in (tg.resnet_models(), tg.bert_models()) for n, m in gen if n in LAYERS}
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        exes = {s: build(tmp, *s) for s in SHAPES}
        for mname, lnames in LAYERS.items():
            call = (lambda m, x: m(x)) if mname.startswith("resnet") else (lambda m, x: m(**x))
            ops = capture_values(models[mname], set(lnames), img if mname.startswith("resnet") else txt, call)
            for ln in lnames:
                Af, Bf = ops[ln]
                for (M, K, P) in SHAPES:
                    for s in range(SLICES_PER_LAYER):
                        r0 = rng.integers(0, Af.shape[0] - M + 1)
                        k0 = rng.integers(0, Af.shape[1] - K + 1)
                        p0 = rng.integers(0, Bf.shape[1] - P + 1)
                        A = quantise(Af[r0:r0 + M, k0:k0 + K])
                        B = quantise(Bf[k0:k0 + K, p0:p0 + P])
                        order = tg.greedy_tiles(B.T != 0, tg.T0)
                        Bg = B[:, order]
                        f0, d0, s0 = tg.cycles(A != 0, B.T != 0, tg.T0)
                        f1, d1, s1 = tg.cycles(A != 0, Bg.T != 0, tg.T0)
                        tag = f"{mname}_{ln}_{M}_{s}".replace(".", "_")
                        o = simulate(exes[(M, K, P)], tmp, A, B, tag + "_o")
                        g = simulate(exes[(M, K, P)], tmp, A, Bg, tag + "_g")
                        rows.append(dict(
                            model=mname, layer=ln, M=M, K=K, P=P, slice=s,
                            w_zero=float((B == 0).mean()), a_zero=float((A == 0).mean()),
                            model_dense=d0, model_skip_orig=s0, model_skip_regroup=s1,
                            rtl_off_orig=o[0], rtl_on_orig=o[1],
                            rtl_off_regroup=g[0], rtl_on_regroup=g[1],
                            checks=o[2] + g[2], errors=o[3] + g[3],
                            permuted_C_equal=bool((g[4] == o[4][:, order]).all())))
                        r = rows[-1]
                        print(f"  {mname:18s} {ln:38s} {M}x{K}x{P} #{s}  "
                              f"on: {r['rtl_on_orig']:4d} -> {r['rtl_on_regroup']:4d}   "
                              f"model skip: {s0:5.0f} -> {s1:5.0f}   "
                              f"err {r['errors']}  C perm {'ok' if r['permuted_C_equal'] else 'BAD'}",
                              flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(ROOT, "results", "regroup_rtl.csv"), index=False)

    off_d = d.rtl_off_orig - d.model_dense
    off_o = d.rtl_on_orig - d.model_skip_orig
    off_g = d.rtl_on_regroup - d.model_skip_regroup
    print("\n=== B13: section 2.8 regrouping measured on the RTL ===")
    print(f"  cases: {len(d)}   output checks: {d.checks.sum():,}   errors: {d.errors.sum()}")
    print(f"  regrouped C == original C with permuted columns: {d.permuted_C_equal.all()}")
    print(f"  skip off, original == regrouped cycles: {(d.rtl_off_orig == d.rtl_off_regroup).all()}")
    print(f"  RTL - accounting offset, skip off: {sorted(off_d.unique())}")
    print(f"  RTL - accounting offset, skip on, original:  {sorted(off_o.unique())}")
    print(f"  RTL - accounting offset, skip on, regrouped: {sorted(off_g.unique())}")
    print(f"  measured cycles saved by regrouping == accounting: "
          f"{((d.rtl_on_orig - d.rtl_on_regroup) == (d.model_skip_orig - d.model_skip_regroup)).all()}")
    for m, x in d.groupby("model"):
        so = 1 - x.rtl_on_orig.sum() / x.rtl_off_orig.sum()
        sg = 1 - x.rtl_on_regroup.sum() / x.rtl_off_orig.sum()
        print(f"  {m:18s} measured compute saving: original {so:6.1%}  regrouped {sg:6.1%}"
              f"  ({(sg - so) * 100:+.1f} pts, {len(x)} cases)")


if __name__ == "__main__":
    main()
