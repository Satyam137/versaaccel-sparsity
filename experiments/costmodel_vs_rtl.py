"""
costmodel_vs_rtl.py -- B3: validate the cost model's dense (DMR) term against RTL.

The RTL's tiled GEMM has an exact closed form, verified cycle-for-cycle by
tb_gemm_golden across 11 shapes from K=1 to K=256 (the most the buffers hold):

    load    = M*K + K*P                          (one element per cycle on din)
    compute = ceil(M/NR) * ceil(P/NC) * (K + NR + NC - 1) + NC + 4
                                                 (wide bus, drain hidden, final
                                                  tile's NC-beat drain exposed;
                                                  + 4 once per GEMM since Rev 20:
                                                  three operand-feed stages and
                                                  the output register)

The cost model's DMR term for MM is ceil(M*K*P / (NR*NC)) -- every PE busy on
every cycle: no skew fill, no drain, no partial tiles. That is the ideal
utilisation bound, not a cycle count. The ratio between the two is roughly
(K + NR + NC - 1) / K for full tiles: 6.2x at K=6, 1.02x at K~1,250.

This script answers three questions:
  1. Does the closed form really match the RTL? (recorded measurements)
  2. How wrong is the model's DMR term on the REAL layer shapes profiled?
  3. Does that error change any configuration decision the model makes?

Question 3 is the one that matters. A cost model can be off in absolute cycles
and still pick the right configuration; it is only wrong where it matters if
the error flips an argmin.

Usage:  python experiments/costmodel_vs_rtl.py              # parts 1-3, ~1 min
        python experiments/costmodel_vs_rtl.py --headline   # + part 4, ~20 min:
            the section 3.2 headline speedup with and without the fill term
"""
import os
import sys
import glob
import math
import shutil
import runpy
import tempfile
from contextlib import contextmanager

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from base_simulator.arch_config import ARRAY_ROWS as NR, ARRAY_COLS as NC
from base_simulator.cost_model import CostModel
from base_simulator import config_modes
from base_simulator.layer_cost import get_layer_costs

DATA = os.path.join(REPO, "idea1_adaptaccel", "profiling_data")

# Measured by tb_gemm_golden (wide bus), cycles from `start` to the final
# output beat, operand load included. Rev 6 recorded the first six; Rev 13
# added the five large-K shapes. Every run also passed its element-by-element
# golden check. Reproduce: idea2_veriflex/rtl/scripts/run_gemm_shapes.sh, or
# iverilog with -DGEMM_M/-DGEMM_K/-DGEMM_P on tb_gemm_golden.sv.
RTL_MEASURED = [  # (M, K, P, cycles) -- Rev 20 RTL (pipelined operand feed)
    (32, 6, 32, 552), (17, 5, 20, 349), (16, 1, 16, 84), (8, 12, 40, 725),
    (33, 3, 17, 374), (5, 7, 9, 156),
    (16, 16, 32, 882), (8, 32, 16, 851), (4, 64, 8, 883), (2, 128, 4, 947),
    (1, 256, 2, 1075),
]
# Rev 6-19 RTL (one-cycle operand feed): every entry 4 cycles fewer.
FEED_PIPE = 4   # three operand-feed stages + the output register, once per GEMM


def rtl_load(M, K, P):
    return M * K + K * P


def rtl_compute(M, K, P):
    tiles = math.ceil(M / NR) * math.ceil(P / NC)
    return tiles * (K + NR + NC - 1) + NC + FEED_PIPE


def model_dmr_compute(M, K, P):
    return config_modes.DMRConfig.compute_cycles('MM', M, K, P, M * K, K * P, NR, NC)


@contextmanager
def dmr_with_fill():
    """Temporarily give the model's DMR MM term the RTL's tile cost."""
    orig = config_modes.DMRConfig.compute_cycles

    def patched(op, M, K, P, nnz_A, nnz_B, Nr, Nc, bw=2):
        if op in ('MM', 'SpMM', 'SpMSpM'):     # DMR runs every op as dense MM
            return math.ceil(M / Nr) * math.ceil(P / Nc) * (K + Nr + Nc - 1) + Nc
        return orig(op, M, K, P, nnz_A, nnz_B, Nr, Nc)

    config_modes.DMRConfig.compute_cycles = staticmethod(patched)
    try:
        yield
    finally:
        config_modes.DMRConfig.compute_cycles = orig


def real_traces():
    out = {}
    for f in sorted(glob.glob(os.path.join(DATA, "*_traces.csv"))):
        d = pd.read_csv(f)
        if "data_source" in d.columns and d["data_source"].astype(str).str.contains(
                "synthetic|noise", case=False).any():
            continue
        if "skippable_frac" not in d.columns:      # real sets carry it; stress sets do not
            continue
        out[os.path.basename(f).replace("_traces.csv", "")] = d
    return out


def main():
    print(f"array {NR}x{NC}\n")

    # ---- 1. closed form vs recorded RTL measurements ----------------------------
    print("1. RTL closed form vs measured (tb_gemm_golden, wide bus)")
    print(f"   {'shape':<16}{'measured':>10}{'formula':>10}{'model DMR':>11}{'model/RTL compute':>19}")
    bad = 0
    for M, K, P, meas in RTL_MEASURED:
        f = rtl_load(M, K, P) + rtl_compute(M, K, P)
        bad += f != meas
        ratio = model_dmr_compute(M, K, P) / rtl_compute(M, K, P)
        print(f"   {M:>3}x{K:>3}x{P:<3}     {meas:>10}{f:>10}{model_dmr_compute(M, K, P):>11}{ratio:>18.3f}")
    print(f"   closed form {'matches all' if bad == 0 else f'MISSES {bad} of'} "
          f"{len(RTL_MEASURED)} measurements\n")
    assert bad == 0, "closed form no longer matches the RTL -- re-measure before trusting part 2"

    # ---- 2. real layer shapes ---------------------------------------------------
    traces = real_traces()
    shp = pd.concat([d[["M", "K", "P"]].assign(net=n) for n, d in traces.items()]) \
            .drop_duplicates()
    shp["rtl"] = [rtl_compute(*r) for r in shp[["M", "K", "P"]].itertuples(index=False)]
    shp["model_dmr"] = [model_dmr_compute(*r) for r in shp[["M", "K", "P"]].itertuples(index=False)]
    shp["ratio"] = shp.rtl / shp.model_dmr
    print("2. DMR compute on the real layer shapes (RTL closed form / model)")
    print("   extrapolates the closed form beyond K=256, the largest K the RTL buffers hold")
    print(f"   {'network':<20}{'layers':>7}{'K range':>14}{'total RTL/model':>17}"
          f"{'worst layer':>13}{'layers >1.10x':>15}")
    for n, g in shp.groupby("net"):
        tot = g.rtl.sum() / g.model_dmr.sum()
        print(f"   {n:<20}{len(g):>7}{f'{g.K.min()}..{g.K.max()}':>14}{tot:>16.3f}x"
              f"{g.ratio.max():>12.2f}x{int((g.ratio > 1.10).sum()):>15}")
    print(f"   all unique shapes: {len(shp)}; total RTL/model {shp.rtl.sum() / shp.model_dmr.sum():.3f}x\n")

    # ---- 3. does the error change any decision? ---------------------------------
    cm = CostModel(Nr=NR, Nc=NC, model="nnz")   # section 7.12 checks the published model's terms
    rows = []
    for n, d in traces.items():
        pts = d[["M", "K", "P", "weight_sparsity", "input_sparsity"]].drop_duplicates()
        for r in pts.itertuples(index=False):
            info = {"M": r.M, "K": r.K, "P": r.P, "weight_sparsity": r.weight_sparsity}
            a = get_layer_costs(cm, info, r.input_sparsity)
            with dmr_with_fill():
                b = get_layer_costs(cm, info, r.input_sparsity)
            ca, cb = min(a, key=a.get), min(b, key=b.get)
            rows.append(dict(model=n, M=r.M, K=r.K, P=r.P, best_model=ca,
                             best_fill=cb, flipped=ca != cb,
                             regret=(a[cb] - a[ca]) / a[ca] if ca != cb else 0.0))
    t = pd.DataFrame(rows)
    print("3. Does the missing fill term change the model's chosen configuration?")
    print(f"   {'network':<20}{'points':>8}{'flips':>7}{'flip %':>8}   flips (model -> with fill)")
    for n, g in t.groupby("model"):
        f = g[g.flipped]
        kinds = ", ".join(f"{a}->{b} x{c}" for (a, b), c in
                          f.groupby(["best_model", "best_fill"]).size().items())
        print(f"   {n:<20}{len(g):>8}{len(f):>7}{100 * len(f) / len(g):>7.1f}%   {kinds}")
    nf = int(t.flipped.sum())
    print(f"   ALL                 {len(t):>8}{nf:>7}{100 * nf / len(t):>7.1f}%")
    if nf:
        print(f"   flipped layers have K in {t[t.flipped].K.min()}..{t[t.flipped].K.max()}")

    shp.to_csv(os.path.join(REPO, "results", "costmodel_vs_rtl_shapes.csv"), index=False)
    t.to_csv(os.path.join(REPO, "results", "costmodel_vs_rtl_decisions.csv"), index=False)
    print("\nsaved results/costmodel_vs_rtl_shapes.csv, results/costmodel_vs_rtl_decisions.csv")

    if "--headline" in sys.argv:
        headline()


def headline():
    """Part 4: re-run section 3.2's scheduler with the model as-is and with the fill
    term, and report both speedups. scheduler_variance.py writes a committed results
    file, so it is backed up and restored -- this is a sensitivity run, not a new
    result for section 3.2."""
    csv = os.path.join(REPO, "results", "scheduler_variance.csv")
    backup = os.path.join(tempfile.mkdtemp(), "scheduler_variance.csv")
    shutil.copy(csv, backup)
    script = os.path.join(REPO, "experiments", "scheduler_variance.py")
    argv = sys.argv
    import base_simulator.cost_model as _cm
    default_model, _cm.COST_MODEL = _cm.COST_MODEL, "nnz"   # the published model, as in section 7.12
    try:
        for label, ctx in (("model as-is", None), ("with RTL fill term", dmr_with_fill)):
            print(f"\n4. Section 3.2 headline, {label}")
            sys.argv = [script]
            if ctx is None:
                runpy.run_path(script, run_name="__main__")
            else:
                with ctx():
                    runpy.run_path(script, run_name="__main__")
    finally:
        sys.argv = argv
        _cm.COST_MODEL = default_model
        shutil.copy(backup, csv)


if __name__ == "__main__":
    main()
