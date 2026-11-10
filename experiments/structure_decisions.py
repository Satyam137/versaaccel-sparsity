"""B6: a structure-aware cost term, and the decision-level test section 5.3 left open.

The cost model's sparse configurations charge compute in proportion to the
nonzero count -- right for the Gustavson dataflow they describe (section 7.9),
wrong for the shared-operand array the RTL builds, which can only skip
all-zero k-slices (section 7.7). Section 2.6 measured the gap: realisable
saving f * K / (K + NR + NC), with f the skippable fraction.

The structural model keeps each configuration's own dense compute and the
nnz memory term (compressed operands still move less data), and credits the
sparse configurations only the realisable saving:

    DMR            dense, as in the RTL (CFG_DMR ignores skip_en)
    D-MR/DM-R/D-M-R  costcomp_dense(c) * (1 - f*K/(K+NR+NC)) + costmem_nnz(c)

Part 1 -- what it changes: section 3's headline (selection vs best static per
workload, oracle schedule with switch costs) and how many per-layer decisions
differ from the nnz model, on the 10 real trace sets.

Part 2 -- section 5.3's open question. A structure predictor beats a
per-layer table on MAE in every seed, by 0.0016. Does that change any
decision? Under the structural model, each predictor's predicted f for the
next layer picks a configuration; scored on the decision-relevant points
(where sweeping f changes the optimal configuration), over 10 seeds with the
same input split, MLP and early stopping as structure_predictor.py.

Usage: python experiments/structure_decisions.py [--passes 20]
Writes results/structure_decisions_headline.csv and
results/structure_decisions_predictors.csv.
"""
import argparse, glob, math, os, sys
import numpy as np, pandas as pd
import torch, torch.nn as nn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "idea1_adaptaccel"))
sys.path.insert(0, os.path.join(REPO, "experiments"))

from base_simulator.arch_config import ARRAY_ROWS, ARRAY_COLS       # noqa: E402
from base_simulator.cost_model import CostModel                     # noqa: E402
from base_simulator.layer_cost import get_layer_costs, select_operator, CONFIG_NAMES  # noqa: E402
import scheduler.dp_scheduler as sched                               # noqa: E402
import structure_predictor as sp                                     # noqa: E402

FILL = ARRAY_ROWS + ARRAY_COLS
PROBE = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
N_SEEDS = 10
_cache = {}


def act_sp(row):
    return max(float(row["input_sparsity"]), float(row.get("attention_sparsity", 0.0) or 0.0))


MEM = "nnz"   # "nnz": compressed operands still move less data; "dense": as the RTL loads them


def structural_costs(cm, row, f):
    """Per-config cost with the sparse configs credited only realisable structure.
    MEM selects the memory term: the nnz model's (compressed operands) or the
    dense one (the RTL loads operands uncompressed)."""
    M, K, P = int(row["M"]), int(row["K"]), int(row["P"])
    ws, a = float(row["weight_sparsity"]), act_sp(row)
    key = (MEM, M, K, P, round(ws, 6), round(a, 6))
    if key not in _cache:
        op = select_operator(ws, a)
        _cache[key] = {c: (cm.cost_for_config(c, "MM", M, K, P, 0.0, 0.0)["costcomp"],
                           (cm.cost_for_config(c, op, M, K, P, ws, a) if MEM == "nnz" else
                            cm.cost_for_config(c, "MM", M, K, P, 0.0, 0.0))["costmem"])
                       for c in CONFIG_NAMES}
    r = max(0.0, min(1.0, f)) * K / (K + FILL)
    out = {}
    for c, (comp, mem) in _cache[key].items():
        out[c] = (comp if c == "DMR" else comp * (1.0 - r)) + mem
    return out


def best(costs):
    return min(costs, key=costs.get)


def part1(cm, dfs, passes):
    rows = []
    init = sched.CONFIG_IDX["DMR"]
    for name, df in dfs.items():
        for pid in sorted(df["input_idx"].unique())[:passes]:
            g = df[df["input_idx"] == pid].reset_index(drop=True)
            nnz = [get_layer_costs(cm, r, act_sp(r)) for _, r in g.iterrows()]
            st = [structural_costs(cm, r, float(r["skippable_frac"])) for _, r in g.iterrows()]
            rec = dict(model=name, input_idx=pid,
                       flips=sum(best(a) != best(b) for a, b in zip(nnz, st)), layers=len(g))
            for tag, costs in (("nnz", nnz), ("struct", st)):
                orc = sched.rescore_schedule(sched.schedule_oracle(costs)["schedule"], costs)["total_cost"]
                static = min(sum(c[k] for c in costs) + sched.SWITCH_COST[init][sched.CONFIG_IDX[k]]
                             for k in CONFIG_NAMES)
                rec[f"speedup_{tag}"] = static / orc
                rec[f"nsparse_{tag}"] = sum(best(c) != "DMR" for c in costs)
            rows.append(rec)
    return pd.DataFrame(rows)


def build_with_meta(dfs):
    """structure_predictor.build, plus the next layer's row for costing."""
    X, y, grp, meta = [], [], [], []
    for name, d in dfs.items():
        for inp, sub in d.groupby("input_idx"):
            sub = sub.reset_index(drop=True)
            n = len(sub)
            for i in range(n - 1):
                cur, nxt = sub.iloc[i], sub.iloc[i + 1]
                M, K, P = max(1, cur["M"]), max(1, cur["K"]), max(1, cur["P"])
                X.append([float(cur["input_sparsity"]), float(cur["weight_sparsity"]),
                          1.0 if cur["layer_type"] == "Conv2d" else 0.0,
                          math.log2(M) / 20.0, math.log2(K) / 20.0, math.log2(P) / 20.0,
                          i / n, float(cur.get("attention_sparsity", 0.0) or 0.0),
                          float(cur["a_kslice_zero"]), float(cur["w_kslice_zero"])])
                y.append(float(nxt["skippable_frac"]))
                grp.append((name, int(inp)))
                meta.append(nxt)
    return np.array(X, np.float32), np.array(y, np.float32), np.array(grp, object), meta


def fit_predict(X, y, grp, seed):
    """structure_predictor.run's split, table and MLP -- returning predictions."""
    rng = np.random.RandomState(seed)
    inputs = sorted({int(g[1]) for g in grp}); rng.shuffle(inputs)
    n = len(inputs)
    tr = set(inputs[:int(.7 * n)]); va = set(inputs[int(.7 * n):int(.85 * n)])
    idx = np.array([int(g[1]) for g in grp])
    m_tr, m_va = np.isin(idx, list(tr)), np.isin(idx, list(va))
    m_te = ~(m_tr | m_va)
    const = float(y[m_tr].mean())
    keys = np.array([f"{g[0]}|{X[i, 6]:.6f}" for i, g in enumerate(grp)])
    tot, cnt = {}, {}
    for k, v in zip(keys[m_tr], y[m_tr]):
        tot[k] = tot.get(k, 0.0) + float(v); cnt[k] = cnt.get(k, 0) + 1
    table = np.array([tot[k] / cnt[k] if k in tot else const for k in keys], np.float32)
    torch.manual_seed(seed)
    mu, sd = X[m_tr].mean(0), X[m_tr].std(0) + 1e-8
    Z = ((X - mu) / sd).astype(np.float32)
    net = sp.MLP(X.shape[1]); opt = torch.optim.Adam(net.parameters(), 1e-3)
    xt, yt = torch.tensor(Z[m_tr]), torch.tensor(y[m_tr])
    xv, yv = torch.tensor(Z[m_va]), torch.tensor(y[m_va])
    bst, state, bad = 1e9, None, 0
    for _ in range(200):
        net.train(); perm = torch.randperm(len(xt))
        for b in range(0, len(xt), 256):
            j = perm[b:b + 256]
            opt.zero_grad(); nn.functional.l1_loss(net(xt[j]), yt[j]).backward(); opt.step()
        net.eval()
        with torch.no_grad(): v = float(nn.functional.l1_loss(net(xv), yv))
        if v < bst - 1e-5:
            bst, state, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 20: break
    net.load_state_dict(state); net.eval()
    with torch.no_grad(): mlp = net(torch.tensor(Z)).numpy()
    return m_te, {"constant (global mean)": np.full_like(y, const), "per-layer table": table,
                  "persistence": X[:, 8], "MLP": mlp, "oracle (true f)": y}


def part2(cm, dfs):
    X, y, grp, meta = build_with_meta(dfs)
    # decision-relevant: the optimal configuration changes as f sweeps PROBE
    relevant = np.array([len({best(structural_costs(cm, m, f)) for f in PROBE}) > 1 for m in meta])
    true_cfg = [best(structural_costs(cm, m, f)) for m, f in zip(meta, y)]
    true_cost = np.array([structural_costs(cm, m, f)[c] for m, f, c in zip(meta, y, true_cfg)])
    rows = []
    for seed in range(N_SEEDS):
        m_te, preds = fit_predict(X, y, grp, seed)
        sel = m_te & relevant
        for name, p in preds.items():
            cfg = [best(structural_costs(cm, meta[i], float(p[i]))) for i in np.flatnonzero(sel)]
            got = np.array([structural_costs(cm, meta[i], float(y[i]))[c]
                            for i, c in zip(np.flatnonzero(sel), cfg)])
            match = (np.mean([c == true_cfg[i] for i, c in zip(np.flatnonzero(sel), cfg)])
                     if sel.any() else float("nan"))
            rows.append(dict(seed=seed, predictor=name, n_relevant=int(sel.sum()),
                             n_test=int(m_te.sum()), match_relevant=float(match),
                             regret_pct=float((got.sum() / true_cost[sel].sum() - 1) * 100) if sel.any() else 0.0,
                             mae=float(np.abs(p[m_te] - y[m_te]).mean())))
    return pd.DataFrame(rows), int(relevant.sum()), len(relevant)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", type=int, default=20)
    args = ap.parse_args()
    cm = CostModel(Nr=ARRAY_ROWS, Nc=ARRAY_COLS, model="nnz")   # B6 as run in Rev 17: against the published model
    dfs = sp.load_real()
    print(f"B6 on {len(dfs)} real trace sets, {ARRAY_ROWS}x{ARRAY_COLS}\n")

    global MEM
    H, T = [], []
    for MEM in ("nnz", "dense"):
        print(f"\n################ structural model, memory term: {MEM} ################")
        h, t = report(cm, dfs, args.passes)
        H.append(h.assign(mem=MEM)); T.append(t.assign(mem=MEM))
    pd.concat(H).to_csv(os.path.join(REPO, "results", "structure_decisions_headline.csv"), index=False)
    pd.concat(T).to_csv(os.path.join(REPO, "results", "structure_decisions_predictors.csv"), index=False)


def report(cm, dfs, passes):
    h = part1(cm, dfs, passes)
    w = h.groupby("model").mean(numeric_only=True)
    print("=== Part 1: structural vs nnz cost model ===")
    print(f"  {'workload':20s} {'speedup nnz':>12s} {'speedup struct':>15s} {'decisions flipped':>18s}"
          f" {'sparse cfg chosen nnz/struct':>30s}")
    for m, r in w.iterrows():
        print(f"  {m:20s} {r.speedup_nnz:11.3f}x {r.speedup_struct:14.3f}x {r.flips / r.layers:17.1%}"
              f" {r.nsparse_nnz / r.layers:14.1%} / {r.nsparse_struct / r.layers:.1%}")
    print(f"  {'mean':20s} {w.speedup_nnz.mean():11.3f}x {w.speedup_struct.mean():14.3f}x"
          f" {(w.flips / w.layers).mean():17.1%}")

    t, nrel, ntot = part2(cm, dfs)
    print(f"\n=== Part 2: does predicting structure change decisions? (structural model) ===")
    print(f"  decision-relevant (layer, input) points: {nrel:,} of {ntot:,}"
          f" ({nrel / ntot:.1%}); held-out per seed ~{int(t.n_relevant.mean()):,}")
    s = t.groupby("predictor", sort=False)[["match_relevant", "regret_pct", "mae"]].agg(["mean", "std"])
    for name, r in s.iterrows():
        print(f"  {name:24s} match {r[('match_relevant', 'mean')]:6.1%} +/- {r[('match_relevant', 'std')]:.1%}"
              f"   cycle regret {r[('regret_pct', 'mean')]:+6.3f}%   MAE {r[('mae', 'mean')]:.4f}")
    mlp = t[t.predictor == "MLP"].set_index("seed").match_relevant
    tab = t[t.predictor == "per-layer table"].set_index("seed").match_relevant
    d = (mlp - tab) * 100
    print(f"\n  MLP - per-layer table, decision match: {d.mean():+.2f} pts +/- {d.std():.2f}"
          f"  (seeds better / tied / worse: {(d > 0).sum()} / {(d == 0).sum()} / {(d < 0).sum()})")
    return h, t


if __name__ == "__main__":
    main()
