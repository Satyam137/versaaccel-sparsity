"""Ideas 1 and 3: how skippable structure depends on tile geometry.

Idea 1 -- tile height.  A k-step is skippable only when it is zero across every
row of the tile (section 7.7), so taller tiles should expose less structure,
while section 5.1 says larger arrays are where runtime selection matters.  This
measures the first half of that tension on the real models: skippable fraction
and realisable saving at square tiles of T = 4, 8, 16, 32, 64.

Idea 3 -- regrouping.  Which rows share a tile is a free choice of the address
generator (the arithmetic is a permutation of output rows).  If rows whose
zeros coincide are placed together, more k-steps become skippable.  Measured at
the 16x16 operating point, against two controls:

  original       rows in natural order (spatial / token order)
  random         rows shuffled -- the floor; any locality in `original` is lost
  sorted         rows of A regrouped per input by their zero pattern (runtime,
                 needs the previous layer's output before tiling)
  greedy         rows of A regrouped per input by greedy tile construction
                 (keep the most shared zeros) -- an upper-bound-style search
  static         one greedy row order per layer, fixed from calibration inputs
                 and applied unchanged to held-out ones (a fixed address table)
  wsort/wgreedy  columns of B (output channels) regrouped by zero pattern --
                 a compile-time network rewrite, zero runtime cost
  wgreedy_safe   wgreedy restricted to layers whose output channels are private
                 to one block (no residual-stream constraint) -- deployable
  greedy+wgreedy both

Cycle accounting follows the RTL (sections 7.6, 7.7): a tile pair costs
K + 2T - 1 cycles dense, and (active k-steps) + 2T with skipping on.

Usage:  python experiments/tile_geometry.py [--images 16] [--texts 24]
Writes results/tile_geometry_height.csv and results/tile_geometry_regroup.csv.
"""
import argparse, os, re, sys, time
import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "idea1_adaptaccel"))
from profiler.workload_profiler import _operand_a, _operand_b  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
DATA = os.path.join(ROOT, "idea1_adaptaccel", "profiling_data", "_datasets")
HEIGHTS = [4, 8, 16, 32, 64]
T0 = 16
torch.set_grad_enabled(False)


# --------------------------------------------------------------------------- metrics
def _pad_rows(Z, T):
    """Pad a [rows, K] nonzero map to a multiple of T rows with zeros (= skippable)."""
    r = (-Z.shape[0]) % T
    return np.concatenate([Z, np.zeros((r, Z.shape[1]), bool)]) if r else Z


def cycles(a_nz_rows, b_nz_cols, T):
    """Dense and skip cycle counts for A-nonzero [M,K] and B-nonzero [P,K] maps."""
    A = _pad_rows(a_nz_rows, T)
    B = _pad_rows(b_nz_cols, T)
    K = A.shape[1]
    a_t = A.reshape(-1, T, K).any(axis=1)            # [n_ti, K]
    b_t = B.reshape(-1, T, K).any(axis=1)            # [n_tj, K]
    n_ti, n_tj = a_t.shape[0], b_t.shape[0]
    active = float((a_t.sum(0).astype(np.float64) * b_t.sum(0)).sum())
    dense = n_ti * n_tj * (K + 2 * T - 1)
    skip = active + n_ti * n_tj * 2 * T
    f = 1.0 - active / (n_ti * K * n_tj)
    return f, dense, skip


def zero_sort(nz):
    """Order rows so those zero on the same k's are adjacent.

    Columns are ranked by how often they are zero, and rows sorted
    lexicographically on their zero bits in that column order: rows that share
    the commonest zeros land in the same tile.
    """
    z = ~nz
    col_order = np.argsort(-z.mean(axis=0), kind="stable")
    packed = np.packbits(z[:, col_order], axis=1)
    keys = np.ascontiguousarray(packed).view(f"V{packed.shape[1]}").ravel()
    return np.argsort(keys, kind="stable")[::-1]      # most-zero rows first


def greedy_tiles(nz, T=T0):
    """Build tiles greedily so each keeps as many shared zeros as possible.

    Seed a tile with the remaining row that has the most zeros, then add, T-1
    times, the remaining row that keeps the most k's zero across the whole tile
    so far (ties to the row with more zeros). Returns a row order. This is an
    upper-bound-style search, not a proposal for hardware.
    """
    M, K = nz.shape
    z = np.packbits(~nz, axis=1)
    pad = (-z.shape[1]) % 8
    if pad:
        z = np.concatenate([z, np.zeros((M, pad), np.uint8)], axis=1)
    z = np.ascontiguousarray(z).view(np.uint64)             # [M, W]
    nzeros = np.bitwise_count(z).sum(axis=1).astype(np.int64)
    rem = np.ones(M, bool)
    order = []
    while rem.any():
        cand = np.flatnonzero(rem)
        seed = cand[np.argmax(nzeros[cand])]
        order.append(seed); rem[seed] = False
        acc = z[seed].copy()
        for _ in range(T - 1):
            cand = np.flatnonzero(rem)
            if cand.size == 0:
                break
            shared = np.bitwise_count(z[cand] & acc).sum(axis=1).astype(np.int64)
            pick = cand[np.argmax(shared * (K + 1) + nzeros[cand])]
            order.append(pick); rem[pick] = False
            acc &= z[pick]
    return np.array(order)


def freely_permutable(layer):
    """Can this layer's output channels be reordered on their own?

    Reordering a layer's output channels is exact if the consumer's input
    channels are reordered to match. That is always legal for channels private
    to one block: the inner convs of a ResNet bottleneck, BERT's FFN hidden
    layer. Channels on a residual stream are shared by every block that adds
    into it, so they would need one common order -- excluded here, which makes
    `wgreedy_safe` a lower bound and `wgreedy` an upper bound.
    """
    return bool(re.search(r"layer\d\.\d+\.conv[12]$", layer)
                or layer.endswith("intermediate.dense"))


# --------------------------------------------------------------------------- capture
class Capture:
    """Forward hooks that record each Conv2d/Linear layer's A and B nonzero maps."""

    def __init__(self, model):
        self.layers, self.cur, self.hooks = {}, None, []
        for name, m in model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                self.hooks.append(m.register_forward_hook(self._hook(name, m)))

    def _hook(self, name, m):
        def fn(mod, inp, out):
            A = _operand_a(mod, inp[0]).float()
            if A.shape[1] == 0:
                return
            rec = self.layers.setdefault(name, {"b_nz": None, "a_nz": []})
            if rec["b_nz"] is None:
                rec["b_nz"] = (_operand_b(mod).float().t() != 0).numpy()   # [P, K]
            rec["a_nz"].append((A != 0).numpy())                          # [M, K]
        return fn

    def close(self):
        for h in self.hooks:
            h.remove()


# --------------------------------------------------------------------------- shared regrouping
def permutation_groups(family, names):
    """Which layers' output channels must share one order, and in what blocks.

    Returns [(members, block)]: every member's output channels take the same
    order, permuted only within consecutive blocks of `block` channels (None =
    anywhere). Layers in no group keep their order. The groups are exactly the
    constraints that make the rewrite exact -- verify_permutation checks it.

    ResNet-50: each stage's residual stream (every conv3, plus the stage's
      downsample) shares one order; the stem conv, every conv1/conv2 and the
      fc are free.
    BERT: the 768-wide residual stream (every attention-output and FFN-output
      dense) shares one order across all 12 layers; query and key share one
      order per layer, and value has its own, both only within a 64-wide head;
      the FFN hidden layer and the pooler are free.
    """
    groups = []
    if family == "resnet":
        for st in range(1, 5):
            groups.append(([n for n in names if re.fullmatch(rf"layer{st}\.\d+\.conv3", n)
                            or n == f"layer{st}.0.downsample.0"], None))
        groups += [([n], None) for n in names
                   if re.fullmatch(r"layer\d\.\d+\.conv[12]", n) or n in ("conv1", "fc")]
    else:
        groups.append(([n for n in names if re.search(r"layer\.\d+\.(attention\.)?output\.dense$", n)], None))
        for i in range(12):
            pre = f"encoder.layer.{i}."
            groups.append(([pre + "attention.self.query", pre + "attention.self.key"], 64))
            groups.append(([pre + "attention.self.value"], 64))
            groups.append(([pre + "intermediate.dense"], None))
        groups.append((["pooler.dense"], None))
    return [(m, b) for m, b in groups if m and all(n in names for n in m)]


def shared_orders(family, layers, T):
    """One output-channel order per layer, honouring permutation_groups."""
    orders = {}
    for members, block in permutation_groups(family, list(layers)):
        Z = np.concatenate([layers[n]["b_nz"] for n in members], axis=1)   # [P, sum K]
        if block is None:
            order = greedy_tiles(Z, T)
        else:
            order = np.concatenate([b + greedy_tiles(Z[b:b + block], T)
                                    for b in range(0, Z.shape[0], block)])
        for n in members:
            orders[n] = order
    return orders


def verify_permutation(family, model, orders, x, call):
    """Apply the orders to a copy of the network; return max |output change|.

    Each layer's output channels are reordered, and every consumer's input
    channels (and any BN / LayerNorm / bias on the way) follow. If the groups
    were wrong the output would change; this is the proof they are not.
    """
    import copy
    ref = call(model, x)
    m = copy.deepcopy(model)
    P = lambda t, o, d=0: torch.nn.Parameter(t.index_select(d, torch.as_tensor(o)))
    get = lambda n: m.get_submodule(n)
    if family == "resnet":
        def out(conv, bn, o):
            get(conv).weight = P(get(conv).weight, o)
            for a in ("weight", "bias"):
                setattr(get(bn), a, P(getattr(get(bn), a), o))
            for a in ("running_mean", "running_var"):
                setattr(get(bn), a, getattr(get(bn), a)[o])
        def inp(conv, o):
            get(conv).weight = P(get(conv).weight, o, 1)
        if "conv1" in orders:
            out("conv1", "bn1", orders["conv1"]); inp("layer1.0.conv1", orders["conv1"])
            inp("layer1.0.downsample.0", orders["conv1"])
        nblk = {1: 3, 2: 4, 3: 6, 4: 3}
        for st in range(1, 5):
            for b in range(nblk[st]):
                pre = f"layer{st}.{b}."
                for c in (1, 2):
                    if pre + f"conv{c}" in orders:
                        o = orders[pre + f"conv{c}"]
                        out(pre + f"conv{c}", pre + f"bn{c}", o); inp(pre + f"conv{c+1}", o)
            o = orders.get(f"layer{st}.0.conv3")
            if o is None:
                continue
            for b in range(nblk[st]):
                out(f"layer{st}.{b}.conv3", f"layer{st}.{b}.bn3", o)
                if b + 1 < nblk[st]:
                    inp(f"layer{st}.{b+1}.conv1", o)
            out(f"layer{st}.0.downsample.0", f"layer{st}.0.downsample.1", o)
            if st < 4:
                inp(f"layer{st+1}.0.conv1", o); inp(f"layer{st+1}.0.downsample.0", o)
            else:
                get("fc").weight = P(get("fc").weight, o, 1)
        if "fc" in orders:
            o = orders["fc"]
            get("fc").weight = P(get("fc").weight, o); get("fc").bias = P(get("fc").bias, o)
        new = call(m, x)
        o = orders.get("fc", np.arange(ref.shape[-1]))
        return float((new - ref[..., o]).abs().max()), float(ref.abs().max())

    # BERT
    def lin_out(n, o):
        get(n).weight = P(get(n).weight, o); get(n).bias = P(get(n).bias, o)
    def lin_in(n, o):
        get(n).weight = P(get(n).weight, o, 1)
    def ln(n, o):
        get(n).weight = P(get(n).weight, o); get(n).bias = P(get(n).bias, o)
    s = orders.get("encoder.layer.0.output.dense")
    if s is not None:
        for e in ("word_embeddings", "position_embeddings", "token_type_embeddings"):
            get(f"embeddings.{e}").weight = P(get(f"embeddings.{e}").weight, s, 1)
        ln("embeddings.LayerNorm", s)
        lin_in("pooler.dense", s)
    for i in range(12):
        pre = f"encoder.layer.{i}."
        if s is not None:
            for q in ("query", "key", "value"):
                lin_in(pre + f"attention.self.{q}", s)
            lin_out(pre + "attention.output.dense", s); ln(pre + "attention.output.LayerNorm", s)
            lin_in(pre + "intermediate.dense", s)
            lin_out(pre + "output.dense", s); ln(pre + "output.LayerNorm", s)
        o = orders.get(pre + "attention.self.query")
        if o is not None:
            lin_out(pre + "attention.self.query", o); lin_out(pre + "attention.self.key", o)
        o = orders.get(pre + "attention.self.value")
        if o is not None:
            lin_out(pre + "attention.self.value", o); lin_in(pre + "attention.output.dense", o)
        o = orders.get(pre + "intermediate.dense")
        if o is not None:
            lin_out(pre + "intermediate.dense", o); lin_in(pre + "output.dense", o)
    rp = orders.get("pooler.dense")
    if rp is not None:
        lin_out("pooler.dense", rp)
    r, n = ref, call(m, x)
    s = s if s is not None else np.arange(r.last_hidden_state.shape[-1])
    rp = rp if rp is not None else np.arange(r.pooler_output.shape[-1])
    d = max(float((n.last_hidden_state - r.last_hidden_state[..., s]).abs().max()),
            float((n.pooler_output - r.pooler_output[..., rp]).abs().max()))
    return d, float(r.last_hidden_state.abs().max())


def analyse(model_name, layers, rng, shared, idx0=0, n_cal=None, cal=None):
    """Per-layer, per-input metrics for both experiments.

    `shared` maps T -> {layer: output-channel order} from shared_orders.
    Inputs may arrive in chunks: `idx0` is the first input's global index,
    `n_cal` the number of calibration inputs (the first ones), and `cal` a dict
    carried across chunks that accumulates their mean zero map for the
    `static` order. With one chunk this is the same calibration as before.
    """
    h_rows, g_rows = [], []
    cal = {} if cal is None else cal
    for lname, rec in layers.items():
        B = rec["b_nz"]
        ins = rec["a_nz"]
        P, K = B.shape
        if n_cal is None:
            n_cal = max(1, len(ins) // 2)
        c = cal.setdefault(lname, {"sum": None, "n": 0, "order": None})
        Bw = B[zero_sort(B)]
        Bg = {T: B[greedy_tiles(B, T)] for T in HEIGHTS}
        Bsh = {T: B[shared[T][lname]] if lname in shared[T] else B for T in HEIGHTS}
        Bs = Bg[T0] if freely_permutable(lname) else B
        for idx, A in enumerate(ins, idx0):
            M = A.shape[0]
            work = float(M) * K * P
            if idx < n_cal:                            # calibration: accumulate
                if c["sum"] is None:
                    c["sum"] = A.astype(np.float32)
                elif c["sum"].shape == A.shape:
                    c["sum"] += A
                c["n"] += 1
            elif c["order"] is None and c["sum"] is not None:
                c["order"] = greedy_tiles(c["sum"] / c["n"] > 0.5)
            for T in HEIGHTS:
                for v, Bv in (("original", B), ("wshared", Bsh[T]), ("wgreedy", Bg[T])):
                    f, d, s = cycles(A, Bv, T)
                    h_rows.append(dict(model=model_name, layer=lname, input=idx, T=T,
                                       variant=v, M=M, K=K, P=P, work=work, skippable=f,
                                       dense_cycles=d, skip_cycles=s,
                                       ideal_cycles=work / (T * T)))
            if idx < n_cal:
                continue                               # held-out inputs only
            Ag = A[greedy_tiles(A)]
            variants = {
                "original": (A, B),
                "random": (A[rng.permutation(M)], B),
                "sorted": (A[zero_sort(A)], B),
                "wsort": (A, Bw),
                "greedy": (Ag, B),
                "wgreedy_safe": (A, Bs),
                "wshared": (A, Bsh[T0]),
                "wgreedy": (A, Bg[T0]),
                "greedy+wshared": (Ag, Bsh[T0]),
                "greedy+wgreedy": (Ag, Bg[T0]),
            }
            if c["order"] is not None and c["sum"].shape == A.shape:
                variants["static"] = (A[c["order"]], B)
            for v, (Av, Bv) in variants.items():
                f, d, s = cycles(Av, Bv, T0)
                g_rows.append(dict(model=model_name, layer=lname, input=idx, variant=v,
                                   M=M, K=K, P=P, work=work, skippable=f,
                                   dense_cycles=d, skip_cycles=s))
    return h_rows, g_rows


# --------------------------------------------------------------------------- models
def resnet_models():
    from torchvision import models
    sys.path.insert(0, os.path.join(ROOT, "idea1_adaptaccel", "profiler"))
    from profile_resnet import apply_unstructured_pruning
    from idea2_veriflex.profile_nm_models import apply_nm_pruning_to_model

    def base():
        return models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).eval()
    yield "resnet50_dense", base()
    yield "resnet50_pruned90", apply_unstructured_pruning(base(), 0.90)
    yield "resnet50_nm24", apply_nm_pruning_to_model(base())


def resnet_inputs(n, seed=0):
    from torchvision.datasets import Imagenette
    sys.path.insert(0, os.path.join(ROOT, "idea1_adaptaccel", "profiler"))
    from profile_resnet import PREPROCESS
    ds = Imagenette(DATA, split="val", size="320px", download=False, transform=PREPROCESS)
    idx = np.random.default_rng(seed).choice(len(ds), n, replace=False)
    return [ds[int(i)][0].unsqueeze(0) for i in idx]


def bert_models():
    from transformers import BertModel
    sys.path.insert(0, os.path.join(ROOT, "idea1_adaptaccel", "profiler"))
    from profile_bert import apply_unstructured_pruning

    def base():
        return BertModel.from_pretrained("bert-base-uncased",
                                         attn_implementation="eager").eval()
    yield "bert_dense", base()
    yield "bert_pruned90", apply_unstructured_pruning(base(), 0.90)


def bert_inputs(n, seed=0):
    from transformers import BertTokenizerFast
    from datasets import load_dataset
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
    texts = [r["sentence"] for r in load_dataset("nyu-mll/glue", "sst2", split="validation")]
    idx = np.random.default_rng(seed).choice(len(texts), n, replace=False)
    return [tok(texts[int(i)], return_tensors="pt", truncation=True,
                padding="max_length", max_length=128) for i in idx]


def per_input(rows, keys):
    """Collapse per-layer rows to one row per input (sums; work-weighted skippable)."""
    import pandas as pd
    if not rows:
        return []
    d = pd.DataFrame(rows)
    d["sw"] = d.skippable * d.work
    cols = [c for c in ("work", "sw", "dense_cycles", "skip_cycles", "ideal_cycles") if c in d]
    out = d.groupby(keys)[cols].sum().reset_index()
    out["skippable"] = out.sw / out.work
    return out.drop(columns="sw").to_dict("records")


def run(model_iter, inputs, call, rng, family, chunk=None):
    """Capture, verify and analyse each model. With `chunk`, inputs are processed
    `chunk` at a time and results kept per input rather than per layer, so
    memory stays bounded at hundreds of inputs."""
    H, G = [], []
    chunk = chunk or len(inputs)
    n_cal = max(1, len(inputs) // 2)
    for name, model in model_iter:
        t = time.time()
        cal, shared = {}, None
        for i0 in range(0, len(inputs), chunk):
            cap = Capture(model)
            for x in inputs[i0:i0 + chunk]:
                call(model, x)
            cap.close()
            if shared is None:
                shared = {T: shared_orders(family, cap.layers, T) for T in HEIGHTS}
                diff, scale = verify_permutation(family, model, shared[T0], inputs[0], call)
                assert diff <= 1e-4 * max(scale, 1.0), \
                    f"{name}: regrouping changed the output by {diff}"
            h, g = analyse(name, cap.layers, rng, shared, i0, n_cal, cal)
            if chunk < len(inputs):
                h = per_input(h, ["model", "input", "T", "variant"])
                g = per_input(g, ["model", "input", "variant"])
                print(f"    {name}: {min(i0 + chunk, len(inputs))}/{len(inputs)} inputs"
                      f"  {time.time()-t:6.0f}s", flush=True)
            H += h; G += g
            n_layers = len(cap.layers)
            del cap
        print(f"  {name:20s} {n_layers:3d} layers  {time.time()-t:6.1f}s"
              f"   regrouped network max |output change| {diff:.1e} (output scale {scale:.1f})",
              flush=True)
        del model
    return H, G


# --------------------------------------------------------------------------- report
def summarise(H, G, tag=""):
    import pandas as pd
    h, g = pd.DataFrame(H), pd.DataFrame(G)
    suffix = f"_{tag}_by_input" if tag else ""
    h.to_csv(os.path.join(RESULTS, f"tile_geometry_height{suffix}.csv"), index=False)
    g.to_csv(os.path.join(RESULTS, f"tile_geometry_regroup{suffix}.csv"), index=False)

    def agg(df, keys):
        out = df.groupby(keys).apply(lambda d: pd.Series({
            "skippable_w": np.average(d.skippable, weights=d.work),
            "saving": 1 - d.skip_cycles.sum() / d.dense_cycles.sum(),
            "util_dense": (d.ideal_cycles.sum() / d.dense_cycles.sum()) if "ideal_cycles" in d else np.nan,
            "thru_skip": (d.ideal_cycles.sum() / d.skip_cycles.sum()) if "ideal_cycles" in d else np.nan,
        }), include_groups=False)
        return out.reset_index()

    print("\n=== Idea 1: tile height T (square T x T tiles) ===")
    print("  skippable = work-weighted all-zero k-step fraction; saving = skip vs dense cycles;")
    print("  util = ideal MKP/T^2 cycles / actual dense cycles; 'with skip' is the same\n  ratio with skipping on (can exceed 100%: skipping beats the dense ideal)\n")
    a = agg(h[h.variant == "original"], ["model", "T"])
    for m, d in a.groupby("model"):
        print(f"  {m}")
        for _, r in d.iterrows():
            print(f"    T={int(r['T']):3d}  skippable {r.skippable_w:6.3f}   saving {r.saving:6.1%}"
                  f"   util dense {r.util_dense:6.1%}  with skip {r.thru_skip:6.1%}")

    print("\n=== Idea 3: regrouping at 16x16 (held-out inputs) ===")
    b = agg(g, ["model", "variant"])
    order = ["random", "original", "static", "sorted", "greedy", "wsort", "wgreedy_safe",
             "wshared", "wgreedy", "greedy+wshared", "greedy+wgreedy"]
    for m, d in b.groupby("model"):
        d = d.set_index("variant")
        base = d.loc["original", "saving"]
        print(f"  {m}")
        for v in order:
            if v in d.index:
                r = d.loc[v]
                print(f"    {v:15s} skippable {r.skippable_w:6.3f}   saving {r.saving:6.1%}"
                      f"   ({(r.saving-base)*100:+5.1f} pts vs original)")

    print("\n=== Ideas 1 x 3: weight regrouping at every tile height (saving) ===")
    print("  wshared = legal network rewrite (residual / head constraints honoured);")
    print("  wgreedy = every layer reordered on its own (upper bound)\n")
    c = agg(h, ["model", "T", "variant"]).pivot_table(index=["model", "T"], columns="variant",
                                                      values="saving")
    for m in c.index.get_level_values(0).unique():
        print(f"  {m}")
        for T, r in c.loc[m].iterrows():
            print(f"    T={int(T):3d}  original {r.original:6.1%}   wshared {r.wshared:6.1%}"
                  f" ({(r.wshared-r.original)*100:+5.1f})   wgreedy {r.wgreedy:6.1%}"
                  f" ({(r.wgreedy-r.original)*100:+5.1f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=16)
    ap.add_argument("--texts", type=int, default=24)
    ap.add_argument("--skip-bert", action="store_true")
    ap.add_argument("--seed", type=int, default=0,
                    help="selects which images/sentences are drawn (and the random control)")
    ap.add_argument("--chunk", type=int, default=0,
                    help="process inputs this many at a time and write per-input CSVs "
                         "tagged n<images>_s<seed>, for runs at hundreds of inputs")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed + 1)
    tag = f"n{args.images}_s{args.seed}" if args.chunk else ""
    print(f"ResNet-50 on {args.images} Imagenette images (seed {args.seed})")
    H, G = run(resnet_models(), resnet_inputs(args.images, args.seed), lambda m, x: m(x),
               rng, "resnet", args.chunk or None)
    if not args.skip_bert:
        print(f"BERT on {args.texts} SST-2 sentences (seed {args.seed})")
        h, g = run(bert_models(), bert_inputs(args.texts, args.seed), lambda m, x: m(**x),
                   rng, "bert", args.chunk or None)
        H += h; G += g
    summarise(H, G, tag)


if __name__ == "__main__":
    main()
