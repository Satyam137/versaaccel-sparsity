"""
predictor_baselines.py -- null baselines for the sparsity predictor.

FIX_PLAN.md M2. Any accuracy claim needs the number a trivial method gets on the
same data. The repo reports "99.2% config match, target >=90%" with no baseline,
which makes the claim unreadable: it is not obvious whether 99.2% reflects a
useful model or a saturated metric.

Baselines
---------
  constant-zero        always predict 0.0 sparsity
  constant-mean        always predict the training-set mean
  persistence          predict the CURRENT layer's sparsity for the next layer
                       (zero parameters, zero training, one array read)
  oracle               the true value (upper bound, sanity check)

`persistence` is the one that matters. If an MLP cannot clear a one-line
heuristic by a margin that changes scheduling decisions, the MLP is not the
contribution.

The script also reports *why* the metric behaves as it does, by measuring how
often the prediction can influence the config choice at all.

Usage:
    python experiments/predictor_baselines.py
"""

import os
import sys
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "idea1_adaptaccel"))

from base_simulator.arch_config import ARRAY_ROWS, ARRAY_COLS, NUM_PES, describe
from base_simulator.cost_model import CostModel
from base_simulator.layer_cost import select_operator, SPARSE_THRESHOLD
from predictor.train_predictor import (load_all_traces, build_dataset, DATA_DIR,
                                       SparsityPredictor)

MODEL_PATH = os.path.join(REPO, "idea1_adaptaccel", "predictor",
                          "trained_models", "sparsity_predictor.pt")


def load_trained_mlp(X):
    """Predictions from the trained MLP, or (None, 0) if it has not been trained.

    The MLP belongs in this table: null baselines are only interesting next to
    the thing they are nulls FOR. Features are already normalised by
    build_dataset, so no scaler is needed.
    """
    if not os.path.exists(MODEL_PATH):
        return None, 0
    import torch
    ck = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model = SparsityPredictor(input_dim=ck["input_dim"],
                              hidden_dim=ck["hidden_dim"],
                              n_hidden=ck["n_hidden"])
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    n_params = sum(q.numel() for q in model.parameters())
    with torch.no_grad():
        pred = model(torch.tensor(X, dtype=torch.float32)).numpy()
    return pred.astype(float), n_params

OUT_DIR = os.path.join(REPO, "results")


def decision_relevant_mask(info, cost_model, probe=(0.0, 0.3, 0.6, 0.99)):
    """
    Which layers can activation sparsity actually influence?

    A layer is decision-relevant if sweeping activation sparsity across `probe`
    yields more than one optimal config. On every other layer the predictor is
    irrelevant by construction: any prediction, however wrong, selects the same
    config. Including those rows in a match rate inflates it toward 100% and
    tells you nothing about the predictor.
    """
    mask = np.zeros(len(info), dtype=bool)
    for i in range(len(info)):
        row = info.iloc[i]
        M, K, P = int(row["M"]), int(row["K"]), int(row["P"])
        w = float(row["weight_sparsity"])
        chosen = {
            cost_model.evaluate(select_operator(w, sp), M, K, P,
                                sparsity_A=w, sparsity_B=sp)["best_config"]
            for sp in probe
        }
        mask[i] = len(chosen) > 1
    return mask


def config_match_rate(pred, actual, info, cost_model, mask=None):
    """
    Fraction of layers where predicted sparsity selects the same config.

    Pass `mask` to restrict to the decision-relevant subset -- the only version
    of this metric that discriminates between predictors.
    """
    idx = range(len(pred)) if mask is None else np.flatnonzero(mask)
    matches = total = 0
    for i in idx:
        row = info.iloc[i]
        M, K, P = int(row["M"]), int(row["K"]), int(row["P"])
        w = float(row["weight_sparsity"])

        best = {}
        for tag, sp in (("pred", float(pred[i])), ("act", float(actual[i]))):
            op = select_operator(w, sp)
            best[tag] = cost_model.evaluate(
                op, M, K, P, sparsity_A=w, sparsity_B=sp
            )["best_config"]
        matches += (best["pred"] == best["act"])
        total += 1
    return matches / total if total else float("nan")


def main():
    print("=" * 84)
    print("  M2: Null baselines for the sparsity predictor")
    print("=" * 84)

    df = load_all_traces(DATA_DIR)
    test_models = [m for m in df.model_name.unique() if "pruned70" in m]
    train_models = [m for m in df.model_name.unique() if m not in test_models]

    X_tr, y_tr = build_dataset(df[df.model_name.isin(train_models)])
    df_test = df[df.model_name.isin(test_models)]
    X, y = build_dataset(df_test)

    info = []
    for _, g in df_test.groupby(["model_name", "input_idx"]):
        g = g.reset_index(drop=True)
        for i in range(len(g) - 1):
            nxt = g.iloc[i + 1]
            info.append({"M": nxt["M"], "K": nxt["K"], "P": nxt["P"],
                         "weight_sparsity": nxt["weight_sparsity"]})
    info = pd.DataFrame(info)

    cm = CostModel(Nr=ARRAY_ROWS, Nc=ARRAY_COLS)
    train_mean = float(y_tr.mean())

    mlp_pred, mlp_params = load_trained_mlp(X)

    baselines = {
        "constant-zero": (np.zeros_like(y), 0),
        f"constant-mean ({train_mean:.3f})": (np.full_like(y, train_mean), 0),
        "persistence (copy current layer)": (X[:, 0].astype(float), 0),
    }
    if mlp_pred is not None:
        baselines["trained MLP"] = (mlp_pred, mlp_params)
    baselines["oracle (true value)"] = (y.astype(float), 0)

    print(f"\n  {describe()}")
    print(f"  Test set: {len(y):,} samples from {test_models}")

    print("\n  Identifying decision-relevant layers...")
    mask = decision_relevant_mask(info, cm)
    n_rel = int(mask.sum())
    print(f"  {n_rel:,} / {len(mask):,} ({n_rel/len(mask):.1%}) of layers can have their "
          f"config changed by activation sparsity.")

    print(f"\n  {'baseline':<38} {'params':>7} {'MAE':>8} {'match% (all)':>13} "
          f"{'match% (relevant)':>18}")
    print("  " + "-" * 88)

    rows = []
    for name, (pred, n_params) in baselines.items():
        mae = float(np.mean(np.abs(pred - y)))
        match_all = config_match_rate(pred, y, info, cm)
        match_rel = config_match_rate(pred, y, info, cm, mask=mask)
        print(f"  {name:<38} {n_params:>7} {mae:>8.4f} {match_all:>12.1%} {match_rel:>17.1%}")
        rows.append({"baseline": name, "params": n_params, "mae": mae,
                     "match_all": match_all, "match_relevant": match_rel,
                     "n_relevant": n_rel, "n_total": len(mask)})

    print(f"\n  The 'all' column is inflated by the {1-n_rel/len(mask):.1%} of layers the")
    print(f"  predictor cannot affect. The 'relevant' column is the metric that")
    print(f"  discriminates, and the one the paper should report.")

    # ---- why the metric saturates ----
    print("\n" + "=" * 84)
    print("  WHY THIS METRIC SATURATES")
    print("=" * 84)

    w = info["weight_sparsity"].values
    forced = (w > SPARSE_THRESHOLD).mean()
    print(f"\n  The config decision takes BOTH operand sparsities, but weight sparsity")
    print(f"  is known exactly at deploy time -- only the activation term is predicted.")
    print(f"\n  Rows where weight sparsity alone already forces a sparse operator: {forced:.1%}")

    # how often does a wrong prediction actually change the chosen config?
    rng = np.random.default_rng(0)
    idx = rng.choice(len(y), size=min(3000, len(y)), replace=False)
    sub_info = info.iloc[idx].reset_index(drop=True)
    y_sub = y[idx]
    flips = 0
    for i in range(len(y_sub)):
        r = sub_info.iloc[i]
        M, K, P = int(r["M"]), int(r["K"]), int(r["P"])
        ws = float(r["weight_sparsity"])
        cfgs = set()
        for sp in (0.0, 0.5, 0.99):  # wildly different activation sparsities
            op = select_operator(ws, sp)
            cfgs.add(cm.evaluate(op, M, K, P, sparsity_A=ws, sparsity_B=sp)["best_config"])
        if len(cfgs) > 1:
            flips += 1
    print(f"  Rows where activation sparsity can change the config AT ALL")
    print(f"  (sweeping it from 0.0 to 0.99): {flips / len(y_sub):.1%}")
    print(f"\n  On the remaining {1 - flips / len(y_sub):.1%} of layers the predictor is")
    print(f"  irrelevant by construction -- any prediction yields the same config.")
    print(f"  A match rate computed over all layers is therefore dominated by rows")
    print(f"  the predictor cannot affect. Report it against `persistence`, or")
    print(f"  restrict the metric to the decision-relevant subset.")

    os.makedirs(OUT_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "predictor_baselines.csv"), index=False)
    print(f"\n  Saved to {OUT_DIR}/predictor_baselines.csv")


if __name__ == "__main__":
    main()
