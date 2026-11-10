"""
predictor_seed_robustness.py -- A8: does section 5.2 survive a change of seed?

Section 5.2 withdraws claim 5 on one comparison: on the 1,800 decision-relevant
layers, the trained MLP matches the optimal configuration 96.2% of the time and
so does a zero-parameter constant. That MLP is ONE trained network.
train_predictor.py never seeds torch, so its initialisation and batch order
are whatever the process happened to draw.

Section 5.3's Rev 12 conclusion was wrong for exactly this reason: a single
seed, on a comparison whose seed-to-seed spread was larger than the effect. So
this script retrains the section 5.2 MLP N_SEEDS times -- same data, same split
(test = the pruned70 models, never seen in training), same architecture,
optimiser, validation hold-out and early stopping as train_predictor.py -- and
scores every seed on the same decision-level metric section 5.2 reports.

The committed model is not touched: nothing here writes to trained_models/.

Usage:  python experiments/predictor_seed_robustness.py
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "idea1_adaptaccel"))
sys.path.insert(0, os.path.join(REPO, "experiments"))

from base_simulator.arch_config import ARRAY_ROWS, ARRAY_COLS
from base_simulator.cost_model import CostModel
from predictor.train_predictor import (load_all_traces, build_dataset, DATA_DIR,
                                       SparsityPredictor, HIDDEN_SIZE,
                                       N_HIDDEN_LAYERS, EPOCHS, BATCH_SIZE,
                                       LEARNING_RATE, VAL_PASS_FRACTION)
from predictor_baselines import (decision_relevant_mask, config_match_rate,
                                 load_trained_mlp)

N_SEEDS = 10
PATIENCE = 20   # as train_predictor.py


def split(df):
    """Section 5.2's split, verbatim: test = pruned70, val = last passes of train."""
    test_models = [m for m in df.model_name.unique() if "pruned70" in m]
    train_models = [m for m in df.model_name.unique() if m not in test_models]
    df_train_all = df[df.model_name.isin(train_models)]
    df_test = df[df.model_name.isin(test_models)]
    passes = np.sort(df_train_all.input_idx.unique())
    n_val = max(1, int(round(VAL_PASS_FRACTION * len(passes))))
    val_passes = set(passes[-n_val:])
    return (df_train_all[~df_train_all.input_idx.isin(val_passes)],
            df_train_all[df_train_all.input_idx.isin(val_passes)],
            df_test, test_models)


def train_one(seed, X_tr, y_tr, X_va, y_va):
    """train_predictor.py's loop, with every source of randomness seeded."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
                        batch_size=BATCH_SIZE, shuffle=True, generator=g)
    model = SparsityPredictor(input_dim=X_tr.shape[1], hidden_dim=HIDDEN_SIZE,
                              n_hidden=N_HIDDEN_LAYERS)
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    crit = nn.MSELoss()
    xv = torch.tensor(X_va)
    best, best_state, bad = float("inf"), None, 0
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            mae = float(np.mean(np.abs(model(xv).numpy() - y_va)))
        if mae < best:
            best, best_state, bad = mae, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state); model.eval()
    return model


def main():
    df = load_all_traces(DATA_DIR)
    df_tr, df_va, df_te, test_models = split(df)
    X_tr, y_tr = build_dataset(df_tr)
    X_va, y_va = build_dataset(df_va)
    X, y = build_dataset(df_te)

    info = []
    for _, g in df_te.groupby(["model_name", "input_idx"]):
        g = g.reset_index(drop=True)
        for i in range(len(g) - 1):
            nxt = g.iloc[i + 1]
            info.append({"M": nxt["M"], "K": nxt["K"], "P": nxt["P"],
                         "weight_sparsity": nxt["weight_sparsity"]})
    info = pd.DataFrame(info)
    cm = CostModel(Nr=ARRAY_ROWS, Nc=ARRAY_COLS)
    mask = decision_relevant_mask(info, cm)
    n_rel = int(mask.sum())

    def score(name, seed, pred, params):
        rel = config_match_rate(pred, y, info, cm, mask=mask)
        return dict(predictor=name, seed=seed, params=params,
                    mae=float(np.mean(np.abs(pred - y))),
                    match_all=config_match_rate(pred, y, info, cm),
                    match_relevant=rel,
                    relevant_correct=int(round(rel * n_rel)), n_relevant=n_rel)

    # --- (y_tr for the constant: the full training set, as section 5.2 does) ---
    X_all_tr, y_all_tr = build_dataset(pd.concat([df_tr, df_va]))
    const = float(y_all_tr.mean())
    rows = [score(f"constant-mean ({const:.3f})", -1, np.full_like(y, const, dtype=float), 0)]
    committed, n_params = load_trained_mlp(X)
    if committed is not None:
        rows.append(score("committed MLP (section 5.2)", -1, committed, n_params))

    print(f"test models {test_models}: {len(y):,} samples, {n_rel:,} decision-relevant")
    print(f"training {N_SEEDS} seeds of the section 5.2 MLP ...")
    for s in range(N_SEEDS):
        m = train_one(s, X_tr, y_tr, X_va, y_va)
        with torch.no_grad():
            p = m(torch.tensor(X, dtype=torch.float32)).numpy().astype(float)
        r = score("MLP", s, p, sum(q.numel() for q in m.parameters()))
        rows.append(r)
        print(f"  seed {s}: MAE {r['mae']:.4f}  relevant match {r['match_relevant']:.2%} "
              f"({r['relevant_correct']}/{n_rel})")

    t = pd.DataFrame(rows)
    out = os.path.join(REPO, "results", "predictor_seed_robustness.csv")
    t.to_csv(out, index=False)

    c = t.iloc[0]
    seeds = t[t.predictor == "MLP"]
    d = seeds.relevant_correct - int(c.relevant_correct)
    print("\n" + "=" * 78)
    print(f"  constant-mean        : {c.match_relevant:.2%} ({int(c.relevant_correct)}/{n_rel}), MAE {c.mae:.4f}")
    if committed is not None:
        cr = t.iloc[1]
        print(f"  committed MLP        : {cr.match_relevant:.2%} ({int(cr.relevant_correct)}/{n_rel}), MAE {cr.mae:.4f}")
    print(f"  MLP over {N_SEEDS} seeds   : {seeds.match_relevant.mean():.2%} +/- "
          f"{seeds.match_relevant.std():.2%}  (range {seeds.match_relevant.min():.2%} .. "
          f"{seeds.match_relevant.max():.2%}),  MAE {seeds.mae.mean():.4f} +/- {seeds.mae.std():.4f}")
    print(f"  MLP minus constant   : {d.mean():+.1f} layers of {n_rel} "
          f"(range {d.min():+d} .. {d.max():+d});  "
          f"better in {int((d > 0).sum())}, tie in {int((d == 0).sum())}, "
          f"worse in {int((d < 0).sum())} of {N_SEEDS} seeds")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
