"""B15: the reshapeable array (VERSAACCEL_RESHAPE) checked and measured on the RTL.

versaaccel_top built with -DVERSAACCEL_RESHAPE runs its 16x16 array as a
square, WIDE (8x32) or TALL (32x8) tile, by cutting the array at a midline
and injecting a second operand feed (section 2.9, B15). Each half is a
physical 8x16 or 16x8 systolic array, so the fill is span_r + span_c, not
NR + NC of the logical tile:

    dense  = n_tiles * (K + span_r + span_c - 1)
    skip   = active steps + n_tiles * (span_r + span_c)

with tiles of tile_r x tile_c and the skip mask taken over the logical tile.

Part 1: random sparse GEMMs at all three shapes -- every output correct, and
RTL cycles = accounting + one fixed offset in both modes.
Part 2: real slices of pruned ResNet-50 / BERT layers through all three shapes
-- measured compute cycles per shape, i.e. the section 2.9 gain on the RTL.

Usage: python experiments/reshape_rtl.py [--tall-only]   (~5 min; needs iverilog)
Writes results/reshape_rtl.csv.
"""
import os, subprocess, sys, tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tile_geometry as tg                                    # noqa: E402
from regroup_rtl import write_hex, quantise, capture_values, RTL, LAYERS  # noqa: E402

NR = NC = 16
SHAPES = {0: ("16x16", NR, NC, NR, NC),            # name, tile_r, tile_c, span_r, span_c
          1: ("8x32 wide", NR // 2, 2 * NC, NR // 2, NC),
          2: ("32x8 tall", 2 * NR, NC // 2, NR, NC // 2)}
GEOMS = [(32, 8, 64), (16, 16, 32)]               # (M, K, P), fit IBUF 256 / WBUF 512


def cycles_reshape(a_nz, b_nz, shape):
    """Accounting for the physical reshape: logical tile for the mask,
    physical half-array spans for the fill."""
    _, tr, tc, sr, sc = SHAPES[shape]
    A = tg._pad_rows(a_nz, tr)
    B = tg._pad_rows(b_nz, tc)
    K = A.shape[1]
    a_t = A.reshape(-1, tr, K).any(axis=1)
    b_t = B.reshape(-1, tc, K).any(axis=1)
    n = a_t.shape[0] * b_t.shape[0]
    active = float((a_t.sum(0).astype(np.float64) * b_t.sum(0)).sum())
    return n * (K + sr + sc - 1), active + n * (sr + sc)


TALL_ONLY = "--tall-only" in sys.argv     # Rev 21: -DVERSAACCEL_RESHAPE_TALL, square + tall
if TALL_ONLY:
    SHAPES = {k: v for k, v in SHAPES.items() if k != 1}


def build(tmp, M, K, P):
    exe = os.path.join(tmp, f"tb_{M}_{K}_{P}")
    srcs = sorted(os.path.join(RTL, "versaaccel_rtl", f)
                  for f in os.listdir(os.path.join(RTL, "versaaccel_rtl")) if f.endswith(".sv"))
    subprocess.run(["iverilog", "-g2012",
                    "-DVERSAACCEL_RESHAPE_TALL" if TALL_ONLY else "-DVERSAACCEL_RESHAPE", "-o", exe,
                    f'-DCPE_WEIGHT_DIR="{RTL}/weights"', f"-DRG_M={M}", f"-DRG_K={K}", f"-DRG_P={P}",
                    os.path.join(RTL, "common", "versaaccel_pkg.sv"), *srcs,
                    os.path.join(RTL, "testbenches", "tb_regroup.sv")],
                   check=True, capture_output=True)
    return exe


def run(exe, tmp, A, B, shape):
    fa, fb = os.path.join(tmp, "a.hex"), os.path.join(tmp, "b.hex")
    write_hex(fa, A); write_hex(fb, B)
    out = subprocess.run(["vvp", "-n", exe, f"+A={fa}", f"+B={fb}", f"+SHAPE={shape}"],
                         capture_output=True, text=True, check=True).stdout
    c_off, c_on, checks, err = map(int, [l for l in out.splitlines()
                                         if l.startswith("RESULT")][0].split()[1:])
    return c_off, c_on, checks, err


def main():
    import pandas as pd
    rng = np.random.default_rng(11)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        exes = {g: build(tmp, *g) for g in GEOMS}
        print("=== Part 1: random sparse GEMMs (correctness + accounting) ===")
        for g in GEOMS:
            M, K, P = g
            for shape, (name, *_rest) in SHAPES.items():
                seen = set(); checks = 0
                for _ in range(8):
                    da, db = rng.choice([0.0, 0.5, 0.8, 0.95], size=2)
                    A = rng.integers(-9, 10, (M, K)); A[rng.random(A.shape) < da] = 0
                    B = rng.integers(-9, 10, (K, P)); B[rng.random(B.shape) < db] = 0
                    c_off, c_on, ch, err = run(exes[g], tmp, A, B, shape)
                    d, s = cycles_reshape(A != 0, B.T != 0, shape)
                    seen.add((c_off - int(d), c_on - int(s), err)); checks += ch
                print(f"  A{M}x{K} B{K}x{P}  {name:9s}  (RTL-acct off, on, errors) {sorted(seen)}"
                      f"   {checks:,} checks")
                rows += [dict(part="random", M=M, K=K, P=P, shape=name, offsets=str(sorted(seen)),
                              checks=checks)]
        print("\n=== Part 2: real pruned-layer slices, measured compute cycles per shape ===")
        img, txt = tg.resnet_inputs(1)[0], tg.bert_inputs(1)[0]
        models = {n: m for gen in (tg.resnet_models(), tg.bert_models())
                  for n, m in gen if n in LAYERS}
        for mname, lnames in LAYERS.items():
            call = (lambda m, x: m(x)) if mname.startswith("resnet") else (lambda m, x: m(**x))
            ops = capture_values(models[mname], set(lnames),
                                 img if mname.startswith("resnet") else txt, call)
            for ln in lnames:
                Af, Bf = ops[ln]
                for g in GEOMS:
                    M, K, P = g
                    for sl in range(3):
                        r0 = rng.integers(0, Af.shape[0] - M + 1)
                        k0 = rng.integers(0, Af.shape[1] - K + 1)
                        p0 = rng.integers(0, Bf.shape[1] - P + 1)
                        A = quantise(Af[r0:r0 + M, k0:k0 + K])
                        B = quantise(Bf[k0:k0 + K, p0:p0 + P])
                        rec = dict(part="real", model=mname, layer=ln, M=M, K=K, P=P, slice=sl)
                        for shape, (name, *_rest) in SHAPES.items():
                            c_off, c_on, ch, err = run(exes[g], tmp, A, B, shape)
                            d, s = cycles_reshape(A != 0, B.T != 0, shape)
                            rec[f"on_{shape}"], rec[f"off_{shape}"] = c_on, c_off
                            rec[f"err_{shape}"] = err
                            rec[f"exact_{shape}"] = (c_on - s == c_off - d)
                        rows.append(rec)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(tg.RESULTS, "reshape_tall_rtl.csv" if TALL_ONLY else "reshape_rtl.csv"),
             index=False)
    r = d[d.part == "real"]
    print(f"  slices: {len(r)}   output errors: {int(r[[f'err_{s}' for s in SHAPES]].sum().sum())}"
          f"   accounting exact (same offset off/on) in every run: "
          f"{bool(r[[f'exact_{s}' for s in SHAPES]].all().all())}")
    for m, x in r.groupby("model"):
        base = x.on_0.sum()
        best = x[[f"on_{s}" for s in SHAPES]].min(axis=1).sum()
        wide = f"wide {base / x.on_1.sum():.3f}x   " if 1 in SHAPES else ""
        print(f"  {m:18s} measured skip-on compute cycles vs 16x16:  "
              f"{wide}tall {base / x.on_2.sum():.3f}x   "
              f"best per slice {base / best:.3f}x   ({len(x)} slices)")


if __name__ == "__main__":
    main()
