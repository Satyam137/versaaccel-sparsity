"""
structure_predictor.py -- B5: did the MLP regress the wrong variable?

Section 5.2 found an 8,961-parameter MLP tying a zero-parameter constant at
selecting configurations, despite predicting sparsity well. Section 11.7
proposes a reason rather than an excuse: its regression target is a SCALAR
DENSITY per layer, and section 7.7 showed density is not what determines
whether the array can skip anything. A shared-operand systolic array skips
only whole k-slices, so the decision-relevant quantity is STRUCTURE.

This script runs the same comparison on both targets, side by side:

    target A   next layer's input density      (what section 5.2 predicted)
    target B   next layer's skippable_frac     (what section 7.7 says matters)

Baselines are identical in both cases: a global constant (0 parameters),
persistence (0 parameters), a PER-LAYER constant, and the same MLP.

The per-layer constant is the null that decides this. It is a compile-time
lookup table -- zero runtime parameters, no inference, no features -- and if it
ties the MLP then whatever the MLP learned is a property of the LAYER, not of
the input, and belongs in a table rather than in a network. Section 5.2 was
missed for nine revisions by comparing against a null that was too weak; the
same mistake is available here one level down, so the strong null is built in
rather than left to a reviewer.

Splits are BY INPUT INDEX, not by row. Two layers of one image share almost
everything, so a row-wise split leaks (section 5.4).

Every comparison is repeated over N_SEEDS seeds, each re-drawing both the input
split and the MLP initialisation, and reported as mean +/- sd. Rev 12 reported
a single seed and concluded the MLP tied the per-layer constant (+1.1%, +4.8%);
Rev 13 found the gain ranges from +8% to +31% across seeds on the SAME data,
so a single seed cannot distinguish a tie from a substantial edge. The verdict
below is therefore based on whether the seeds agree in sign, not on a
threshold applied to one draw.

Usage:  python experiments/structure_predictor.py
"""
import os, math, glob
import numpy as np, pandas as pd
import torch, torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'idea1_adaptaccel', 'profiling_data')
SEED = 0
N_SEEDS = 10

TARGETS = {
    'density  (5.2)': lambda nxt: max(float(nxt['input_sparsity']),
                                      float(nxt.get('attention_sparsity', 0.0) or 0.0)),
    'structure (B5)': lambda nxt: float(nxt['skippable_frac']),
}
# Persistence copies the current layer's own value of the matching quantity:
# input_sparsity (feature 0) for density, a_kslice_zero (feature 8) for structure.
PERSIST_COL = {'density  (5.2)': 0, 'structure (B5)': 8}


def load_real():
    """Only traces carrying the structural columns -- i.e. real model traces."""
    out = {}
    for f in sorted(glob.glob(os.path.join(DATA, '*_traces.csv'))):
        d = pd.read_csv(f)
        if 'skippable_frac' not in d.columns or d['skippable_frac'].isna().all():
            continue
        out[os.path.basename(f).replace('_traces.csv', '')] = d
    return out


def build(dfs, target_fn):
    """One sample per (input, layer) pair; features from layer i, target at i+1."""
    X, y, grp = [], [], []
    for name, d in dfs.items():
        for inp, sub in d.groupby('input_idx'):
            sub = sub.reset_index(drop=True)
            n = len(sub)
            for i in range(n - 1):
                cur, nxt = sub.iloc[i], sub.iloc[i + 1]
                M, K, P = max(1, cur['M']), max(1, cur['K']), max(1, cur['P'])
                X.append([
                    float(cur['input_sparsity']),
                    float(cur['weight_sparsity']),
                    1.0 if cur['layer_type'] == 'Conv2d' else 0.0,
                    math.log2(M) / 20.0, math.log2(K) / 20.0, math.log2(P) / 20.0,
                    i / n,
                    float(cur.get('attention_sparsity', 0.0) or 0.0),
                    # the two structural features -- the only new information
                    float(cur['a_kslice_zero']), float(cur['w_kslice_zero']),
                ])
                y.append(target_fn(nxt))
                grp.append((name, int(inp)))
    return np.array(X, np.float32), np.array(y, np.float32), np.array(grp, object)


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


def run(label, X, y, grp):
    rng = np.random.RandomState(SEED)
    inputs = sorted({int(g[1]) for g in grp})
    rng.shuffle(inputs)
    n = len(inputs)
    tr = set(inputs[:int(.7 * n)]); va = set(inputs[int(.7 * n):int(.85 * n)])
    idx = np.array([int(g[1]) for g in grp])
    m_tr = np.isin(idx, list(tr)); m_va = np.isin(idx, list(va))
    m_te = ~(m_tr | m_va)

    const = float(y[m_tr].mean())
    mae_const = float(np.abs(y[m_te] - const).mean())
    mae_persist = float(np.abs(y[m_te] - X[m_te][:, PERSIST_COL[label]]).mean())

    # per-layer constant: compile-time table keyed on (model, layer position)
    key_tr = [f"{g[0]}|{X[i,6]:.6f}" for i, g in enumerate(grp) if m_tr[i]]
    key_te = [f"{g[0]}|{X[i,6]:.6f}" for i, g in enumerate(grp) if m_te[i]]
    tot, cnt = {}, {}
    for k, v in zip(key_tr, y[m_tr]):
        tot[k] = tot.get(k, 0.0) + float(v); cnt[k] = cnt.get(k, 0) + 1
    tbl = {k: tot[k] / cnt[k] for k in tot}
    pl = np.array([tbl.get(k, const) for k in key_te], np.float32)
    mae_perlayer = float(np.abs(y[m_te] - pl).mean())

    torch.manual_seed(SEED)
    mu, sd = X[m_tr].mean(0), X[m_tr].std(0) + 1e-8
    Z = ((X - mu) / sd).astype(np.float32)
    net = MLP(X.shape[1]); opt = torch.optim.Adam(net.parameters(), 1e-3)
    xt, yt = torch.tensor(Z[m_tr]), torch.tensor(y[m_tr])
    xv, yv = torch.tensor(Z[m_va]), torch.tensor(y[m_va])
    best, best_state, bad = 1e9, None, 0
    for ep in range(200):
        net.train()
        perm = torch.randperm(len(xt))
        for b in range(0, len(xt), 256):
            j = perm[b:b + 256]
            opt.zero_grad(); loss = nn.functional.l1_loss(net(xt[j]), yt[j])
            loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): v = float(nn.functional.l1_loss(net(xv), yv))
        if v < best - 1e-5:
            best, best_state, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 20: break
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        mae_mlp = float(nn.functional.l1_loss(net(torch.tensor(Z[m_te])), torch.tensor(y[m_te])))
    nparam = sum(p.numel() for p in net.parameters())
    return dict(target=label, n_test=int(m_te.sum()), y_std=float(y[m_te].std()),
                mae_const=mae_const, mae_persist=mae_persist,
                mae_perlayer=mae_perlayer, mae_mlp=mae_mlp, mlp_params=nparam,
                gain_vs_global=100 * (mae_const - mae_mlp) / max(mae_const, 1e-12),
                gain_vs_perlayer=100 * (mae_perlayer - mae_mlp) / max(mae_perlayer, 1e-12))


def main():
    global SEED
    dfs = load_real()
    print(f"real trace sets with structural columns: {len(dfs)}, seeds: {N_SEEDS}")
    if not dfs:
        raise SystemExit("no traces carry skippable_frac -- re-profile first (appendix)")
    rows = []
    for label, fn in TARGETS.items():
        X, y, g = build(dfs, fn)
        for s in range(N_SEEDS):
            SEED = s
            rows.append(dict(seed=s, **run(label, X, y, g)))
    t = pd.DataFrame(rows)
    out = os.path.join(ROOT, 'results', 'structure_predictor.csv')
    t.to_csv(out, index=False)

    cols = ['mae_const', 'mae_persist', 'mae_perlayer', 'mae_mlp',
            'gain_vs_global', 'gain_vs_perlayer']
    summ = t.groupby('target', sort=False)[cols].agg(['mean', 'std'])
    pd.set_option('display.width', 220)
    print()
    print(summ.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\nsaved {out} (one row per target x seed)\n")
    for label, grp in t.groupby('target', sort=False):
        v = grp['gain_vs_perlayer']
        if (v > 0).all():
            verdict = "consistent edge -- every seed beats the per-layer table"
        elif (v < 0).all():
            verdict = "consistently worse than the per-layer table"
        else:
            verdict = "not distinguishable from the per-layer table -- seeds straddle zero"
        print(f"  {label}: MLP vs per-layer {v.mean():+.1f}% +/- {v.std():.1f} "
              f"(range {v.min():+.1f} .. {v.max():+.1f})  ->  {verdict}")


if __name__ == '__main__':
    main()
