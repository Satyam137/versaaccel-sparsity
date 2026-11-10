"""
profile_opt.py -- Profile OPT-125m with PRETRAINED weights on REAL text (B9).

The second sequence-model family beside BERT. OPT is a decoder-only
Transformer whose feed-forward block uses ReLU (BERT uses GELU), so its fc2
input carries genuine activation zeros -- the input-dependent structure this
project measures -- and its causal attention is a different sparsity regime
from BERT's bidirectional one.

Same protocol as profile_bert.py: the same SST-2 sentences, padded to 32 / 64 /
128 tokens in rotation, 200 inputs, dense and 50/70/90% global-unstructured
pruning of every Linear layer, eager attention so the probability matrix
exists. attention_sparsity uses BERT's definition (fraction of probabilities
< 0.01 over the whole matrix); under a causal mask the upper triangle is
always zero, so it is also reported over the causally visible entries only
(attention_sparsity_causal).

Usage:
    python -m profiler.profile_opt --dataset sst2 [--out-dir profiling_data/b9]
"""

import argparse
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from profiler.workload_profiler import LayerProfiler                  # noqa: E402
from profiler.profile_bert import (apply_unstructured_pruning,        # noqa: E402
                                   load_texts, N_SAMPLES, OUTPUT_DIR)

MODEL_ID = "facebook/opt-125m"
ATTN_THRESHOLD = 0.01            # as profile_bert.py
SEQ_LENGTHS = [32, 64, 128]


def profile_opt(model, name, tokenizer, texts, provenance, out_dir):
    print(f"\n{'='*60}\n  Profiling: {name}\n{'='*60}")
    profiler = LayerProfiler(model, model_name=name)
    profiler.register_hooks()
    profiler.records = []
    attn = {}                    # (input_idx, layer) -> (whole, causal)
    with torch.no_grad():
        for i in range(N_SAMPLES):
            L = SEQ_LENGTHS[i % len(SEQ_LENGTHS)]
            enc = tokenizer(texts[i % len(texts)], return_tensors="pt", truncation=True,
                            padding="max_length", max_length=L)
            profiler._input_idx = i
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                        output_attentions=True)
            vis = torch.tril(torch.ones(L, L, dtype=torch.bool))
            for li, a in enumerate(out.attentions):            # [1, heads, L, L]
                small = a[0] < ATTN_THRESHOLD
                attn[(i, li)] = (float(small.float().mean()),
                                 float(small[:, vis].float().mean()))
            if (i + 1) % 50 == 0:
                print(f"  Profiled {i + 1}/{N_SAMPLES} inputs...")
    profiler.remove_hooks()

    df = pd.DataFrame(profiler.records)
    whole, causal = [], []
    for _, r in df.iterrows():
        w = c = 0.0
        parts = r["layer_name"].split(".")
        if "self_attn" in parts and "layers" in parts:
            w, c = attn[(int(r["input_idx"]), int(parts[parts.index("layers") + 1]))]
        whole.append(w); causal.append(c)
    df["attention_sparsity"] = whole
    df["attention_sparsity_causal"] = causal
    for k, v in provenance.items():
        df[k] = v
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}_traces.csv")
    df.to_csv(path, index=False)
    print(f"  Layers profiled: {df['layer_name'].nunique()}   records: {len(df)}")
    print(f"  Avg input / weight sparsity: {df['input_sparsity'].mean():.3f} / "
          f"{df['weight_sparsity'].mean():.3f}")
    print(f"  Saved to {path}")


def main():
    torch.manual_seed(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-file", help="real text, one passage per line")
    ap.add_argument("--dataset", help="sst2 | wikitext (needs `datasets`)")
    ap.add_argument("--out-dir", default=os.path.join(OUTPUT_DIR, "b9"))
    args = ap.parse_args()
    args.allow_synthetic = False                 # real text only for this family
    texts, prov = load_texts(args)
    prov["weights"] = MODEL_ID

    from transformers import AutoTokenizer, OPTModel
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    for label, sp in [("dense", None), ("pruned50", 0.50), ("pruned70", 0.70), ("pruned90", 0.90)]:
        # eager: the SDPA path never forms the attention matrix (as profile_bert.py)
        model = OPTModel.from_pretrained(MODEL_ID, attn_implementation="eager").eval()
        if sp is not None:
            apply_unstructured_pruning(model, sp)
            model.eval()
        profile_opt(model, f"opt125m_{label}", tokenizer, texts, dict(prov), args.out_dir)


if __name__ == "__main__":
    main()
