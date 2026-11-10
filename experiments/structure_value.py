"""
structure_value.py -- what is the measured k-slice structure actually WORTH?

Section 2.5 measures how structured real sparsity is. This converts that into
cycles using the ceiling section 7.7 derived from the RTL:

    saving = f * K / (K + NR + NC)        f = skippable fraction, NR=NC=16

and compares it against what the cost model's nnz-proportional term credits.
Layers are weighted by M*K*P, their share of the total work, so the result is a
whole-network number rather than an average over layers of wildly different
size.

The point is not that the cost model is optimistic. It is that the cost model
is wrong in BOTH directions -- it over-credits ResNet, whose activation zeros
are only partly k-structured, and it entirely misses pruned BERT, whose benefit
comes from WEIGHT structure that an activation-sparsity term cannot see.

Usage:  python experiments/structure_value.py
"""
import os, glob
import numpy as np, pandas as pd

NR = NC = 16
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'idea1_adaptaccel', 'profiling_data')


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(DATA, '*_traces.csv'))):
        d = pd.read_csv(f)
        if 'skippable_frac' not in d.columns or d['skippable_frac'].isna().all():
            continue
        name = os.path.basename(f).replace('_traces.csv', '')
        work = (d.M * d.K * d.P).astype(float)
        w = work / work.sum()
        rows.append(dict(
            workload=name,
            skippable_frac=float((d.skippable_frac * w).sum()),
            mean_K=float((d.K * w).sum()),
            realisable_saving=float((d.skippable_frac * d.K / (d.K + NR + NC) * w).sum()),
            nnz_model_claims=float((d.input_sparsity.clip(0, 1) * w).sum())))
    if not rows:
        raise SystemExit("no traces carry skippable_frac -- re-profile first (appendix)")
    t = pd.DataFrame(rows)
    t['model_error'] = t.nnz_model_claims - t.realisable_saving
    pd.set_option('display.width', 200)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nmean realisable saving over {len(t)} real trace sets: "
          f"{100 * t.realisable_saving.mean():.1f}%")
    over = t[t.model_error > 0.05]; under = t[t.model_error < -0.05]
    print(f"cost model OVER-credits {len(over)} sets (max {100*over.model_error.max():.1f} pts), "
          f"UNDER-credits {len(under)} (max {-100*under.model_error.min():.1f} pts)")
    out = os.path.join(ROOT, 'results', 'structure_value.csv')
    t.to_csv(out, index=False); print(f"saved {out}")


if __name__ == '__main__':
    main()
