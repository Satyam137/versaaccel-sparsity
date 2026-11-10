"""B16: tile shape under an operand-bandwidth budget, instead of a hard cap.

An NR x NC tile consumes NR + NC operands per cycle (one per row feed and one
per column feed). Section 2.9 handled this by *excluding* shapes above 40
operands/cycle. Here every shape is allowed, but a buffer that delivers only
BW operands per cycle stretches each compute cycle by max(1, (NR+NC)/BW).
Square 16x16 needs 32, the RTL's reshaped tiles 40, the idealised 4x64 / 64x4
68.

Reads results/tile_shape.csv (tile_shape.py) -- no re-profiling.

Usage: python experiments/tile_shape_bandwidth.py
Writes results/tile_shape_bandwidth.csv.
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUDGETS = [32, 40, 48, 64, np.inf]
MODELS = ["bert_pruned90", "resnet50_pruned90", "resnet50_dense", "resnet50_nm24"]


def main():
    d = pd.read_csv(os.path.join(ROOT, "results", "tile_shape.csv"))
    rows = []
    for kind in ("virtual", "fission"):
        x = d[d.kind == kind].copy()
        # operands per cycle: the fission tiles feed 40 distinct values
        x["ops"] = np.where(kind == "fission", np.where(x.NR == x.NC, 32, 40), x.NR + x.NC)
        for bw in BUDGETS:
            f = np.maximum(1.0, x.ops / bw)
            y = x.assign(skip_bw=x.skip * f, cal_skip_bw=x.cal_skip * f)
            for m in MODELS:
                z = y[y.model == m]
                for v in ("original", "wshared"):
                    q = z[z.variant == v]
                    ref = z[(z.NR == 16) & (z.NC == 16) & (z.variant == "original")].skip_bw.sum()
                    pick = q.loc[q.groupby("layer").cal_skip_bw.idxmin()]
                    rows.append(dict(kind=kind, bw=bw, model=m, variant=v,
                                     speedup=ref / pick.skip_bw.sum(),
                                     shapes=", ".join(f"{a}x{b} x{n}" for (a, b), n in
                                                      pick.groupby(["NR", "NC"]).size().items())))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(ROOT, "results", "tile_shape_bandwidth.csv"), index=False)
    print("=== Per-layer shape chosen by structure, under an operand budget (speedup over 16x16) ===")
    for kind in ("fission", "virtual"):
        print(f"\n  {kind} shapes" + (" (the RTL's square / 8x32 / 32x8)" if kind == "fission"
                                        else " (idealised 4x64 ... 64x4)"))
        t = out[(out.kind == kind)].pivot_table(index=["model", "variant"], columns="bw",
                                                 values="speedup")
        t.columns = [("inf" if np.isinf(c) else f"BW={int(c)}") for c in t.columns]
        print(t.to_string(float_format=lambda v: f"{v:.2f}x"))


if __name__ == "__main__":
    main()
