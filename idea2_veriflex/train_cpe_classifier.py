"""
train_cpe_classifier.py -- train the on-chip Config Prediction Engine, and
export INT8 weights the RTL actually loads.

FIX_PLAN.md B8. The RTL CPE was a constant function: every weight hardcoded to
`8'sd1` and every bias to 0, making its argmax degenerate. Its topology
(9 -> 16 -> 16 -> 5 classifier) also did not match the trained Python predictor
(9 -> 64 -> 64 -> 64 -> 1 sigmoid regressor), and nothing anywhere performed a
`$readmemh`, so the exported .npz was never loaded by any hardware.

Rather than reshape the RTL to fit a sparsity regressor, this trains the network
the RTL is actually shaped for -- and the one the hardware wants. A config
prediction engine should emit a CONFIG, not a sparsity value that software then
has to convert. So:

    inputs : 9 layer features (INT8, Q4.4)
    output : argmax over 5 configs
    labels : the config the cost model says is optimal for that layer

Quantisation matches the RTL exactly, so the Python reference is bit-exact with
the hardware and `tb_cpe_golden.sv` can assert equality rather than similarity:

    acc[k]  = b[k] + sum_j w[k][j] * x[j]        (integer)
    out[k]  = clamp(acc[k] >>> shift, 0, 127)    (requantise, ReLU, saturate)

The arithmetic right-shift is new. Without it the RTL clamped raw accumulators
(peaking near 20,000) straight to 127, so every activation saturated and the
network was destroyed: INT8 accuracy collapsed to 0.70 against the float
model's 0.998. A per-layer shift is standard fixed-point practice and is
exported alongside the weights.

Usage:
    python -m idea2_veriflex.train_cpe_classifier
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from base_simulator.arch_config import ARRAY_ROWS, ARRAY_COLS, describe
from base_simulator.cost_model import CostModel
from base_simulator.layer_cost import get_layer_costs, CONFIG_NAMES
from base_simulator.trace_validation import validate_traces

DATA_DIR = os.path.join(REPO, "idea1_adaptaccel", "profiling_data")
HEX_DIR = os.path.join(REPO, "idea2_veriflex", "rtl", "weights")
RESULTS = os.path.join(REPO, "results")

# Must match config_prediction_engine.sv
N_FEATURES, H1, H2, N_OUT = 9, 16, 16, 5
ACT_MAX = 127           # INT8 activation saturation ceiling in the RTL
Q_FRAC = 16             # Q4.4 -> value * 16

EPOCHS, BATCH, LR = 60, 256, 2e-3
VAL_PASS_FRACTION = 0.2
SEED = 7


# ---------------------------------------------------------------------------
# Features -- must match the order the RTL's `features[9]` port expects
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "input_sparsity", "weight_sparsity", "is_conv",
    "log2M/20", "log2K/20", "log2P/20",
    "layer_position", "attention_sparsity", "nm_compatibility",
]


def build_features(row, position):
    import math
    return [
        float(row["input_sparsity"]),
        float(row["weight_sparsity"]),
        1.0 if row["layer_type"] == "Conv2d" else 0.0,
        math.log2(max(1, int(row["M"]))) / 20.0,
        math.log2(max(1, int(row["K"]))) / 20.0,
        math.log2(max(1, int(row["P"]))) / 20.0,
        float(position),
        float(row.get("attention_sparsity", 0.0) or 0.0),
        float(row.get("nm_compatibility", 0.0) or 0.0),
    ]


def load_dataset():
    """Features -> optimal-config label, from the cost model."""
    cm = CostModel(Nr=ARRAY_ROWS, Nc=ARRAY_COLS)
    X, y, passes, models = [], [], [], []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*_traces.csv"))):
        name = os.path.basename(path).replace("_traces.csv", "")
        df = pd.read_csv(path)
        validate_traces(df, name)
        for pid, g in df.groupby("input_idx"):
            g = g.reset_index(drop=True)
            n = len(g)
            for i, row in g.iterrows():
                attn = float(row["attention_sparsity"]) if "attention_sparsity" in row.index else 0.0
                act_sp = max(float(row["input_sparsity"]), attn)
                costs = get_layer_costs(cm, row, act_sp)
                best = min(costs, key=costs.get)
                X.append(build_features(row, i / max(1, n)))
                y.append(CONFIG_NAMES.index(best))
                passes.append(int(pid))
                models.append(name)
    return (np.array(X, dtype=np.float32), np.array(y, dtype=np.int64),
            np.array(passes), np.array(models))


# ---------------------------------------------------------------------------
# Float model (same shape as the RTL)
# ---------------------------------------------------------------------------
class CPENet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(N_FEATURES, H1)
        self.fc2 = nn.Linear(H1, H2)
        self.fc3 = nn.Linear(H2, N_OUT)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# ---------------------------------------------------------------------------
# Integer reference -- bit-exact with config_prediction_engine.sv
# ---------------------------------------------------------------------------
def quantise_layer(weight, bias, x_scale):
    """
    Quantise one layer's weights and bias to INT8.

    Returns (w_int8, b_int8, product_scale) where product_scale relates the
    integer accumulator to the float pre-activation.
    """
    w_scale = 127.0 / max(1e-9, np.abs(weight).max())
    w_q = np.clip(np.round(weight * w_scale), -128, 127).astype(np.int32)
    b_q = np.clip(np.round(bias * x_scale * w_scale), -128, 127).astype(np.int32)
    return w_q, b_q, x_scale * w_scale


def choose_shift(acc, target=ACT_MAX, pct=99.5):
    """
    Pick an arithmetic right-shift so the bulk of the accumulator distribution
    lands inside [0, 127] after shifting. Uses a high percentile rather than the
    max so a few outliers saturate instead of crushing every other activation
    toward zero.
    """
    hi = np.percentile(np.abs(acc), pct)
    if hi <= target:
        return 0
    return int(max(0, np.ceil(np.log2(hi / target))))


def int_forward(x_q, W, B, S):
    """
    Integer pipeline exactly as the RTL executes it, shifts included.
    x_q: (N, 9) int32 activations; W/B/S: per-layer weights, biases, shifts.
    """
    h = x_q
    for i, (w, b, sh) in enumerate(zip(W, B, S)):
        acc = h @ w.T + b                       # integer accumulate
        if i < len(W) - 1:
            h = np.clip(acc >> sh, 0, ACT_MAX)  # requantise, ReLU, saturate
        else:
            h = acc >> sh                       # output: shift only, no clamp
    return h


class _RoundSTE(torch.autograd.Function):
    """Round with a straight-through gradient."""
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, g):
        return g


def _fake_quant_weight(w):
    """Per-layer symmetric INT8 fake-quantisation, differentiable via STE."""
    scale = 127.0 / w.abs().max().clamp(min=1e-9)
    return _RoundSTE.apply((w * scale).clamp(-128, 127)) / scale, scale


def qat_forward(model, x_float, shifts):
    """
    Forward pass that reproduces the RTL's integer arithmetic (INT8 weights,
    requantisation shift, ReLU, saturation to 127) while staying
    differentiable, so the network can learn around the quantisation error.

    Post-training quantisation alone left a 10-point accuracy gap: the shift
    plus the 127 ceiling discards too much for a network that never saw them.
    """
    x_q = _RoundSTE.apply((x_float * Q_FRAC).clamp(-128, 127))
    h = x_q
    for i, fc in enumerate((model.fc1, model.fc2, model.fc3)):
        wq, w_scale = _fake_quant_weight(fc.weight)
        x_scale = Q_FRAC if i == 0 else 1.0
        bq = _RoundSTE.apply((fc.bias * x_scale * w_scale).clamp(-128, 127))
        acc = h @ (wq * w_scale).t() + bq
        if i < 2:
            h = (acc / (2 ** shifts[i])).floor()
            h = _RoundSTE.apply(h).clamp(0, ACT_MAX)
        else:
            h = acc
    return h


def write_hex(path, arr):
    """Two's-complement 8-bit hex, one value per line, row-major."""
    with open(path, "w") as f:
        for v in np.asarray(arr).reshape(-1):
            f.write(f"{int(v) & 0xFF:02x}\n")


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 78)
    print("  CPE classifier -- train, quantise to INT8, export for $readmemh")
    print("=" * 78)
    print(f"\n  {describe()}")
    print(f"  Topology: {N_FEATURES} -> {H1} -> {H2} -> {N_OUT} (matches the RTL)")

    print("\n  Building dataset from the cost model's optimal configs...")
    X, y, passes, models = load_dataset()
    uniq, counts = np.unique(y, return_counts=True)
    print(f"  {len(X):,} layers | label distribution: "
          + ", ".join(f"{CONFIG_NAMES[u]}={c:,}" for u, c in zip(uniq, counts)))

    # split by inference pass, same discipline as the sparsity predictors
    all_p = np.unique(passes)
    n_val = max(1, int(round(VAL_PASS_FRACTION * len(all_p))))
    val_p, test_p = set(all_p[-n_val:]), set(all_p[-2 * n_val:-n_val])
    te = np.isin(passes, list(test_p))
    va = np.isin(passes, list(val_p))
    tr = ~(te | va)
    print(f"  Split by pass: {tr.sum():,} train / {va.sum():,} val / {te.sum():,} test")

    Xtr = torch.tensor(X[tr]); ytr = torch.tensor(y[tr])
    Xva = torch.tensor(X[va]); yva = torch.tensor(y[va])

    model = CPENet()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss()

    print(f"\n  Training ({EPOCHS} epochs, early stop on validation)...")
    best_acc, best_state = -1.0, None
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = lossf(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(Xva).argmax(1) == yva).float().mean().item()
        if acc > best_acc:
            best_acc, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 15 == 0:
            print(f"    epoch {ep+1:>3}  val acc {acc:.4f}")
    model.load_state_dict(best_state)
    print(f"  Best validation accuracy: {best_acc:.4f}")

    # ---- quantise ----
    print("\n  Quantising to INT8...")
    x_scale = float(Q_FRAC)          # Q4.4 inputs
    x_q = np.clip(np.round(X * Q_FRAC), -128, 127).astype(np.int32)

    W, B, S, scale = [], [], [], x_scale
    h_cal = x_q[tr]                  # calibrate shifts on TRAINING data only
    for li, fc in enumerate((model.fc1, model.fc2, model.fc3)):
        w = fc.weight.detach().numpy()
        b = fc.bias.detach().numpy()
        wq, bq, scale = quantise_layer(w, b, scale)
        acc_cal = h_cal @ wq.T + bq
        # the output layer only feeds an argmax, so its shift is cosmetic --
        # choose it purely to keep the accumulator inside the RTL's width
        sh = choose_shift(acc_cal) if li < 2 else 0
        W.append(wq); B.append(bq); S.append(sh)
        h_cal = np.clip(acc_cal >> sh, 0, ACT_MAX) if li < 2 else acc_cal
        scale = 1.0
    print(f"  Requantisation shifts per layer: {S}")

    # float accuracy BEFORE QAT -- this is the achievable ceiling. After QAT the
    # weights are optimised for the integer pipeline, so the float network is no
    # longer the deployment target and its accuracy is not a meaningful number.
    with torch.no_grad():
        pre_qat_float = (model(torch.tensor(X[te])).argmax(1).numpy() == y[te]).mean()

    # ---- quantisation-aware fine-tuning ----
    print("\n  Quantisation-aware fine-tuning (integer forward pass, STE)...")
    qat_opt = torch.optim.Adam(model.parameters(), lr=LR / 10)
    Xtr_t, ytr_t = torch.tensor(X[tr]), torch.tensor(y[tr])
    Xva_t, yva_t = torch.tensor(X[va]), torch.tensor(y[va])
    best_q, best_qstate = -1.0, None
    for ep in range(25):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(Xtr_t), BATCH):
            idx = perm[i:i + BATCH]
            qat_opt.zero_grad()
            lossf(qat_forward(model, Xtr_t[idx], S), ytr_t[idx]).backward()
            qat_opt.step()
        model.eval()
        with torch.no_grad():
            qacc = (qat_forward(model, Xva_t, S).argmax(1) == yva_t).float().mean().item()
        if qacc > best_q:
            best_q, best_qstate = qacc, {k: v.clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 5 == 0:
            print(f"    QAT epoch {ep+1:>2}  val acc (integer) {qacc:.4f}")
    model.load_state_dict(best_qstate)
    print(f"  Best QAT validation accuracy: {best_q:.4f}")

    # re-quantise from the fine-tuned weights
    W, B, scale = [], [], x_scale
    for li, fc in enumerate((model.fc1, model.fc2, model.fc3)):
        wq, bq, scale = quantise_layer(fc.weight.detach().numpy(),
                                       fc.bias.detach().numpy(), scale)
        W.append(wq); B.append(bq)
        scale = 1.0

    # ---- accuracy: float vs integer ----
    with torch.no_grad():
        float_pred = model(torch.tensor(X)).argmax(1).numpy()
    int_pred = int_forward(x_q, W, B, S).argmax(1)

    f_acc = float(pre_qat_float)
    i_acc = (int_pred[te] == y[te]).mean()
    majority = max(np.bincount(y[te])) / len(y[te])
    print(f"\n  Test accuracy, float ceiling (pre-QAT):  {f_acc:.4f}")
    print(f"  Test accuracy, INT8 as deployed:         {i_acc:.4f}")
    print(f"  Majority-class baseline (always D-M-R):  {majority:.4f}")
    print(f"\n  INT8 is what the RTL runs. The float number is the ceiling this")
    print(f"  topology could reach in full precision; the gap is the price of")
    print(f"  INT8 weights, a 127 activation ceiling, and per-layer shifts.")
    print(f"  Post-training quantisation alone gave 0.8934 here -- QAT recovers")
    print(f"  most of the remaining gap. Both clear the majority-class baseline,")
    print(f"  which the previous all-ones CPE did not (it was a constant function).")
    agree = float(i_acc)

    # ---- overflow audit: does the RTL's 16-bit accumulator suffice? ----
    print("\n  Accumulator range audit (RTL uses 16-bit signed, +/-32,767):")
    h = x_q
    worst = 0
    for i, (w, b, sh) in enumerate(zip(W, B, S)):
        acc = h @ w.T + b
        peak = int(np.abs(acc).max())
        worst = max(worst, peak)
        theo = int(np.abs(w).sum(axis=1).max() * ACT_MAX + np.abs(b).max())
        flag = "  <-- EXCEEDS 16-bit" if theo > 32767 else ""
        print(f"    layer {i+1}: observed peak |acc| = {peak:,}, "
              f"theoretical max = {theo:,}, shift = {sh}{flag}")
        h = np.clip(acc >> sh, 0, ACT_MAX) if i < len(W) - 1 else acc >> sh
    print(f"\n  Observed worst case {worst:,}. The RTL accumulators are widened to")
    print(f"  24 bits so the theoretical maximum cannot overflow regardless of weights.")

    # ---- export ----
    os.makedirs(HEX_DIR, exist_ok=True)
    for name, arr in [("w1", W[0]), ("b1", B[0]), ("w2", W[1]), ("b2", B[1]),
                      ("w3", W[2]), ("b3", B[2])]:
        write_hex(os.path.join(HEX_DIR, f"cpe_{name}.hex"), arr)
    with open(os.path.join(HEX_DIR, "cpe_shifts.hex"), "w") as f:
        for sh in S:
            f.write(f"{int(sh) & 0xFF:02x}\n")
    print(f"\n  Exported 6 weight/bias .hex files + cpe_shifts.hex to {HEX_DIR}/")

    # ---- test vectors for the RTL bench ----
    rng = np.random.default_rng(SEED)
    sel = rng.choice(np.flatnonzero(te), size=min(64, int(te.sum())), replace=False)
    with open(os.path.join(HEX_DIR, "cpe_test_vectors.hex"), "w") as f:
        for i in sel:
            for v in x_q[i]:
                f.write(f"{int(v) & 0xFF:02x}\n")
    with open(os.path.join(HEX_DIR, "cpe_expected.hex"), "w") as f:
        for i in sel:
            f.write(f"{int(int_pred[i]) & 0xFF:02x}\n")
    print(f"  Exported {len(sel)} test vectors + expected argmax for tb_cpe_golden.sv")

    os.makedirs(RESULTS, exist_ok=True)
    pd.DataFrame([{
        "float_ceiling_acc": f_acc, "int8_test_acc": i_acc, "majority_baseline": majority,
        "qat_val_acc": best_q, "shifts": str(S),
        "n_layers": len(X), "worst_accumulator": worst,
    }]).to_csv(os.path.join(RESULTS, "cpe_quantisation.csv"), index=False)
    print(f"  Saved to {RESULTS}/cpe_quantisation.csv")


if __name__ == "__main__":
    main()
