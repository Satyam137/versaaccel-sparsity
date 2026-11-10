"""Does a non-square tile pay on sparse layers? (follow-up to sections 2.7-2.8)

Section 2.7 varied square tiles. But the two operands' structure lives on
different tile edges: a k-step is skippable when it is zero across the tile's
NR rows of A, OR across its NC columns of B. Activation zeros (ReLU) group
along rows, pruned-weight zeros along columns, so the best tile shape should
depend on where a layer's zeros are -- which would make the tile shape a
configuration worth making reconfigurable.

At a fixed 256 PEs this compares shapes NR x NC in {4x64, 8x32, 16x16, 32x8,
64x4}, cycles charged as the RTL charges them (K + NR + NC - 1 per tile pair
dense; active steps + NR + NC skipping), each in original channel order and
with section 2.8's legal regrouping redone for that shape's column width.

Policies, all at 256 PEs, all against fixed 16x16 with skipping:
  fixed S          one shape for every layer of the network
  per-layer dense  each layer takes the shape with fewest DENSE cycles --
                   the shape-fitting gain any reshapeable array gets, with no
                   sparsity involved (prior work: e.g. Planaria, MICRO 2020)
  per-layer skip   each layer takes the shape with fewest SKIP cycles
The skip policy's gain over the dense policy is what sparsity structure adds
to choosing a shape -- the part that would be new. Layer choices are made on
the calibration half of the inputs and scored on the held-out half.

Usage: python experiments/tile_shape.py [--images 16] [--texts 24]
Writes results/tile_shape.csv (per layer x shape x variant, held-out sums).
"""
import argparse, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tile_geometry as tg   # noqa: E402

SHAPES = [(4, 64), (8, 32), (16, 16), (32, 8), (64, 4)]
# B15: the shapes the RTL actually implements (VERSAACCEL_RESHAPE). The logical
# tile is NR x NC, but each half is a physical 8x16 / 16x8 array, so the fill
# is span_r + span_c (24), not NR + NC (40). Entries: (NR, NC, span_r, span_c).
FISSION = [(16, 16, 16, 16), (8, 32, 8, 16), (32, 8, 16, 8)]


def cycles_rect(a_nz, b_nz, NR, NC, span_r=None, span_c=None):
    """tile_geometry.cycles() for an NR x NC tile: A rows by NR, B columns by NC.
    span_r / span_c (default NR / NC) are the physical propagation distances
    that set the fill -- smaller than NR / NC for the RTL's split tiles."""
    span_r = NR if span_r is None else span_r
    span_c = NC if span_c is None else span_c
    A = tg._pad_rows(a_nz, NR)
    B = tg._pad_rows(b_nz, NC)
    K = A.shape[1]
    a_t = A.reshape(-1, NR, K).any(axis=1)
    b_t = B.reshape(-1, NC, K).any(axis=1)
    n_ti, n_tj = a_t.shape[0], b_t.shape[0]
    active = float((a_t.sum(0).astype(np.float64) * b_t.sum(0)).sum())
    dense = n_ti * n_tj * (K + span_r + span_c - 1)
    skip = active + n_ti * n_tj * (span_r + span_c)
    return dense, skip


def run(model_iter, inputs, call, family):
    rows = []
    n_cal = max(1, len(inputs) // 2)
    for name, model in model_iter:
        t = time.time()
        cap = tg.Capture(model)
        for x in inputs:
            call(model, x)
        cap.close()
        orders = {NC: tg.shared_orders(family, cap.layers, NC) for _, NC in SHAPES}
        for lname, rec in cap.layers.items():
            B = rec["b_nz"]
            geoms = [(NR, NC, NR, NC, "virtual") for NR, NC in SHAPES] + \
                    [(NR, NC, sr, sc, "fission") for NR, NC, sr, sc in FISSION]
            for NR, NC, sr, sc, kind in geoms:
                Bs = B[orders[NC][lname]] if lname in orders[NC] else B
                for v, Bv in (("original", B), ("wshared", Bs)):
                    acc = {"cal": [0.0, 0.0], "held": [0.0, 0.0]}
                    for idx, A in enumerate(rec["a_nz"]):
                        d, s = cycles_rect(A, Bv, NR, NC, sr, sc)
                        part = acc["cal" if idx < n_cal else "held"]
                        part[0] += d; part[1] += s
                    rows.append(dict(model=name, layer=lname, kind=kind, NR=NR, NC=NC, variant=v,
                                     M=rec["a_nz"][0].shape[0], K=B.shape[1], P=B.shape[0],
                                     cal_dense=acc["cal"][0], cal_skip=acc["cal"][1],
                                     dense=acc["held"][0], skip=acc["held"][1]))
        print(f"  {name:20s} {time.time()-t:6.1f}s", flush=True)
        del model, cap
    return rows


def report_fission(d):
    """B15: the RTL's three shapes, with their real (half-array) fill."""
    d = d[d.kind == "fission"].copy()
    print("\n=== B15: the RTL's reshapeable array (16x16 / 8x32 / 32x8, fill = half-array spans) ===")
    print("  speedup over fixed 16x16 with skipping; per-layer choice on calibration, scored held-out\n")
    out = []
    for m, x in d.groupby("model"):
        ref = x[(x.NR == 16) & (x.variant == "original")].skip.sum()
        for v in ("original", "wshared"):
            y = x[x.variant == v]
            fixed = {f"{a}x{b}": ref / y[(y.NR == a) & (y.NC == b)].skip.sum()
                     for a, b, _, _ in FISSION}
            fit = ref / y.loc[y.groupby("layer").cal_dense.idxmin()].skip.sum()
            pick = y.loc[y.groupby("layer").cal_skip.idxmin()]
            struct = ref / pick.skip.sum()
            mix = pick.apply(lambda r: f"{int(r.NR)}x{int(r.NC)}", axis=1).value_counts()
            print(f"  {m:18s} {v:8s} fixed: " + "  ".join(f"{k} {x_:.3f}x" for k, x_ in fixed.items())
                  + f"   per-layer fit {fit:.3f}x  structure {struct:.3f}x"
                  f"  (structure adds {(struct / fit - 1) * 100:+.1f}%)   "
                  + ", ".join(f"{k} x{n}" for k, n in mix.items()))
            out.append(dict(model=m, variant=v, **{f"fixed_{k}": x_ for k, x_ in fixed.items()},
                            per_layer_fit=fit, per_layer_structure=struct))
    import pandas as pd
    return pd.DataFrame(out)


def report(d):
    d = d[d.kind == "virtual"].copy()
    shape = lambda r: f"{int(r.NR)}x{int(r.NC)}"
    d["shape"] = d.apply(shape, axis=1)
    print("\n=== Tile shape at 256 PEs: held-out cycles relative to fixed 16x16 with skipping ===")
    print("  (<1.00 is faster; 'dense' = skipping off)\n")
    out = []
    for m, x in d.groupby("model"):
        ref = x[(x["shape"] == "16x16") & (x.variant == "original")].skip.sum()
        print(f"  {m}")
        for v in ("original", "wshared"):
            y = x[x.variant == v]
            cells = []
            for NR, NC in SHAPES:
                z = y[(y.NR == NR) & (y.NC == NC)]
                cells.append(f"{NR}x{NC} {z.skip.sum()/ref:5.3f}")
            print(f"    fixed, {v:8s}  " + "   ".join(cells))
        for v in ("original", "wshared"):
            y = x[x.variant == v]
            # per-layer choices made on calibration inputs, scored on held-out
            pick_dense = y.loc[y.groupby("layer").cal_dense.idxmin()]
            pick_skip = y.loc[y.groupby("layer").cal_skip.idxmin()]
            dense_policy = pick_dense.skip.sum() / ref
            skip_policy = pick_skip.skip.sum() / ref
            oracle = y.groupby("layer").skip.min().sum() / ref
            mix = pick_skip["shape"].value_counts()
            print(f"    per-layer, {v:8s}  by dense cycles {dense_policy:5.3f}   "
                  f"by skip cycles {skip_policy:5.3f}   (held-out oracle {oracle:5.3f})   "
                  f"shapes chosen: " + ", ".join(f"{k} x{n}" for k, n in mix.items()))
            out.append(dict(model=m, variant=v, fixed_16x16=y[y["shape"] == "16x16"].skip.sum() / ref,
                            per_layer_dense=dense_policy, per_layer_skip=skip_policy,
                            oracle=oracle))
    import pandas as pd
    s = pd.DataFrame(out)
    print("\n  Summary: speedup over fixed 16x16 original (ref / cycles)")
    for _, r in s.iterrows():
        print(f"    {r.model:20s} {r.variant:8s}  fixed16 {1/r.fixed_16x16:5.3f}x   "
              f"shape-fit only {1/r.per_layer_dense:5.3f}x   shape by structure "
              f"{1/r.per_layer_skip:5.3f}x   (structure adds "
              f"{(r.per_layer_dense/r.per_layer_skip-1)*100:+.1f}%)")
    return s


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=16)
    ap.add_argument("--texts", type=int, default=24)
    args = ap.parse_args()
    print(f"ResNet-50 on {args.images} images")
    rows = run(tg.resnet_models(), tg.resnet_inputs(args.images), lambda m, x: m(x), "resnet")
    print(f"BERT on {args.texts} sentences")
    rows += run(tg.bert_models(), tg.bert_inputs(args.texts), lambda m, x: m(**x), "bert")
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(tg.RESULTS, "tile_shape.csv"), index=False)
    report(d)
    report_fission(d).to_csv(os.path.join(tg.RESULTS, "tile_shape_fission_summary.csv"), index=False)


if __name__ == "__main__":
    main()
