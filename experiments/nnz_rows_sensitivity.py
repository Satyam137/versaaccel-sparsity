"""
nnz_rows_sensitivity.py -- does the nonzero-row estimator change any decision?

FIX_PLAN.md T6 / PROJECT_REPORT.md Q2. Every sparse configuration's cycle count
is gated by `_estimate_nnz_rows`, which answers "how many rows of A hold at
least one nonzero". The original estimator is M*sqrt(density): it reproduces a
single 4x4 example from the paper and has no derivation beyond that.

For zeros that are independent and identically distributed -- which is what
unstructured magnitude pruning produces, and what these traces contain -- the
exact expectation is M*(1-(1-d)^K). The two disagree by up to 9x, always in the
direction that makes the sparse configs look cheaper under sqrt().

An estimator being wrong does not by itself invalidate anything: the cost model
is only ever used through an argmin. This script asks the question that matters
-- how often does swapping the estimator change which configuration wins, and
by how much does the winning cost move?

Usage:
    python experiments/nnz_rows_sensitivity.py
"""

import os
import sys
import glob
import itertools
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from base_simulator import config_modes
from base_simulator.cost_model import CostModel
from base_simulator.memory_model import MemoryModel
from base_simulator.layer_cost import select_operator, is_decision_relevant
from base_simulator.arch_config import ARRAY_ROWS, ARRAY_COLS, describe

MODELS = ('sqrt', 'iid')
MAX_ACT_PER_LAYER = 25          # distinct activation-sparsity values per layer

SHAPES = [
    (32, 768, 768),      # bert attention projection
    (128, 768, 3072),    # bert FFN up
    (512, 768, 2304),    # bert qkv, batched
    (3136, 64, 64),      # resnet layer1 conv
    (12544, 147, 64),    # resnet conv1
    (2048, 2048, 128),   # gcn aggregate
]
ACT_SPARSITY = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]
WEIGHT_SPARSITY = [0.0, 0.5, 0.7, 0.9]


def evaluate(cm, model, op, M, K, P, w, a):
    config_modes.NNZ_ROW_MODEL = model
    r = cm.evaluate(op, M, K, P, sparsity_A=w, sparsity_B=a)
    return r['best_config'], r['total_cost']


def grid_sweep(cm):
    """Synthetic grid: every shape x weight sparsity x activation sparsity."""
    total = flips = 0
    cost_ratio_sum = 0.0
    flip_pairs = {}
    for (M, K, P), w, a in itertools.product(SHAPES, WEIGHT_SPARSITY, ACT_SPARSITY):
        op = select_operator(w, a)
        c0, k0 = evaluate(cm, 'sqrt', op, M, K, P, w, a)
        c1, k1 = evaluate(cm, 'iid',  op, M, K, P, w, a)
        total += 1
        cost_ratio_sum += k1 / k0 if k0 else 1.0
        if c0 != c1:
            flips += 1
            flip_pairs[(c0, c1)] = flip_pairs.get((c0, c1), 0) + 1
    return total, flips, cost_ratio_sum / total, flip_pairs


def trace_sweep(cm):
    """Real traces: every profiled layer at its observed activation sparsities."""
    files = sorted(glob.glob(os.path.join(
        REPO, "idea1_adaptaccel", "profiling_data", "*_traces.csv")))
    rows = []
    for path in files:
        name = os.path.basename(path).replace("_traces.csv", "")
        df = pd.read_csv(path)
        need = {'M', 'K', 'P', 'weight_sparsity', 'input_sparsity', 'layer_name'}
        if not need.issubset(df.columns):
            continue
        total = flips = 0
        cost_ratio_sum = 0.0
        for layer, g in df.groupby('layer_name'):
            first = g.iloc[0]
            M, K, P = int(first['M']), int(first['K']), int(first['P'])
            w = float(first['weight_sparsity'])
            acts = sorted(set(round(float(v), 2) for v in g['input_sparsity']))
            if len(acts) > MAX_ACT_PER_LAYER:
                step = len(acts) / MAX_ACT_PER_LAYER
                acts = [acts[int(i * step)] for i in range(MAX_ACT_PER_LAYER)]
            for a in acts:
                op = select_operator(w, a)
                c0, k0 = evaluate(cm, 'sqrt', op, M, K, P, w, a)
                c1, k1 = evaluate(cm, 'iid',  op, M, K, P, w, a)
                total += 1
                cost_ratio_sum += k1 / k0 if k0 else 1.0
                if c0 != c1:
                    flips += 1
        if total:
            rows.append((name, total, flips, 100.0 * flips / total,
                         cost_ratio_sum / total))
    return rows


def relevance_shift(cm):
    """Does the decision-relevant subset (the section 5 metric) change size?"""
    out = {}
    for model in MODELS:
        config_modes.NNZ_ROW_MODEL = model
        n = sum(1 for (M, K, P), w in itertools.product(SHAPES, WEIGHT_SPARSITY)
                if is_decision_relevant(cm, M, K, P, w))
        out[model] = (n, len(SHAPES) * len(WEIGHT_SPARSITY))
    return out


def main():
    mem = MemoryModel()
    cm = CostModel(Nr=ARRAY_ROWS, Nc=ARRAY_COLS, memory=mem, model="nnz")   # section 2.4 studies the published estimator

    print("=" * 84)
    print("  Q2: nonzero-row estimator sensitivity")
    print(f"  {describe()}")
    print("=" * 84)
    print("\n  sqrt : M*sqrt(density)      -- original, implies ROW-STRUCTURED sparsity")
    print("  iid  : M*(1-(1-d)^K)        -- exact for independent zeros\n")

    total, flips, ratio, pairs = grid_sweep(cm)
    print("  -- synthetic grid (6 shapes x 4 weight x 6 activation sparsities) --")
    print(f"     combinations            : {total}")
    print(f"     argmin changes          : {flips}  ({100.0*flips/total:.1f}%)")
    print(f"     mean best-cost ratio    : {ratio:.3f}x  (iid / sqrt)")
    if pairs:
        print("     flips:")
        for (a, b), n in sorted(pairs.items(), key=lambda kv: -kv[1]):
            print(f"       {a:>6} -> {b:<6}  {n}")

    print("\n  -- real profiled traces --")
    print(f"     {'workload':<26} {'points':>7} {'flips':>7} {'flip %':>8} {'cost ratio':>11}")
    print("     " + "-" * 62)
    agg_t = agg_f = 0
    for name, t, f, pct, cr in trace_sweep(cm):
        print(f"     {name:<26} {t:7d} {f:7d} {pct:7.1f}% {cr:10.3f}x")
        agg_t += t
        agg_f += f
    if agg_t:
        print("     " + "-" * 62)
        print(f"     {'ALL':<26} {agg_t:7d} {agg_f:7d} {100.0*agg_f/agg_t:7.1f}%")

    print("\n  -- decision-relevant subset (the section 5 metric) --")
    for model, (n, tot) in relevance_shift(cm).items():
        print(f"     {model:<6} {n}/{tot} combinations sparsity-sensitive"
              f"  ({100.0*n/tot:.1f}%)")

    config_modes.NNZ_ROW_MODEL = 'sqrt'   # leave the default as we found it
    print("\n" + "=" * 84)


if __name__ == '__main__':
    main()
