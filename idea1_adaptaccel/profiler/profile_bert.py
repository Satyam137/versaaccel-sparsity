"""
profile_bert.py -- Profile BERT-Base with PRETRAINED weights on REAL text.

FIX_PLAN.md M7. This script previously built BERT from a bare BertConfig
(randomly initialised) and fed it `torch.randint(0, 30522, ...)` -- uniformly
random token IDs. Attention sparsity is the headline input-dependent quantity
in this project, and on a random network fed random tokens the attention
distribution is near-uniform by construction: it measures the softmax, not BERT.

It now refuses to produce traces from random tokens unless you explicitly ask,
and stamps the provenance into every CSV.

Usage:
    # real text, one sentence or paragraph per line
    python -m profiler.profile_bert --text-file corpus.txt

    # or pull a real corpus via the `datasets` package
    python -m profiler.profile_bert --dataset sst2
    python -m profiler.profile_bert --dataset wikitext

    # explicitly reproduce the old, meaningless behaviour
    python -m profiler.profile_bert --allow-synthetic

Requires: pip install transformers        (plus `datasets` for --dataset)
Needs network access on first run.

Note for python.org installs on macOS: if HTTPS downloads fail with
CERTIFICATE_VERIFY_FAILED, that interpreter has no CA bundle wired up. Either
run the "Install Certificates.command" that ships beside it, or:
    export SSL_CERT_FILE=$(python3 -c "import certifi;print(certifi.where())")
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from profiler.workload_profiler import LayerProfiler

N_SAMPLES = 200
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "profiling_data")


def apply_unstructured_pruning(model, sparsity):
    """Apply global unstructured L1 pruning to all Linear layers."""
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parameters_to_prune.append((module, 'weight'))
    if len(parameters_to_prune) == 0:
        return model
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=sparsity,
    )
    for module, _ in parameters_to_prune:
        prune.remove(module, 'weight')
    return model


def load_texts(args):
    """Return (list_of_strings or None, provenance). None means synthetic."""
    if args.text_file:
        with open(args.text_file, encoding="utf-8") as f:
            texts = [ln.strip() for ln in f if ln.strip()]
        if not texts:
            sys.exit(f"{args.text_file} is empty")
        print(f"  Real text: {len(texts)} lines from {args.text_file}")
        return texts, {"data_source": f"textfile:{os.path.basename(args.text_file)}"}

    if args.dataset:
        try:
            from datasets import load_dataset
        except ImportError:
            sys.exit("--dataset needs the `datasets` package: pip install datasets\n"
                     "  Or pass --text-file with your own corpus.")
        # datasets >= 3 requires canonical "namespace/name" ids; the bare
        # "glue" / "wikitext" shorthands were removed and now fail with
        # HfUriError: Repository id must be 'namespace/name'.
        if args.dataset == "sst2":
            ds = load_dataset("nyu-mll/glue", "sst2", split="validation")
            texts = [r["sentence"] for r in ds]
        elif args.dataset == "wikitext":
            ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                              split="validation")
            texts = [r["text"].strip() for r in ds if len(r["text"].strip()) > 64]
        else:
            sys.exit(f"unknown --dataset {args.dataset} (try sst2 or wikitext)")
        print(f"  Real text: {len(texts)} passages from {args.dataset}")
        return texts, {"data_source": args.dataset}

    if args.allow_synthetic:
        print("\n  !! SYNTHETIC INPUTS: uniformly random token IDs, not language.")
        print("  !! Attention sparsity from this run is meaningless and the")
        print("  !! traces are stamped data_source=random-tokens.\n")
        return None, {"data_source": "random-tokens"}

    sys.exit(
        "Refusing to profile on random tokens.\n"
        "  Attention sparsity is this project's headline input-dependent quantity,\n"
        "  and on random tokens it describes the softmax rather than BERT\n"
        "  (FIX_PLAN.md M7).\n\n"
        "  Pick one:\n"
        "    --text-file corpus.txt   your own text, one passage per line\n"
        "    --dataset sst2           fetch a real corpus via `datasets`\n"
        "    --allow-synthetic        reproduce the old behaviour on purpose\n")


def profile_bert(model, model_name, device, texts=None, provenance=None,
                 tokenizer=None, seq_lengths=[32, 64, 128]):
    """Profile a single BERT configuration, including attention-score sparsity."""
    print(f"\n{'='*60}")
    print(f"  Profiling: {model_name}")
    print(f"{'='*60}")

    profiler = LayerProfiler(model, model_name=model_name)
    profiler.register_hooks()
    profiler.records = []

    # --- Attention sparsity hooks ---
    # Capture softmax attention weights: fraction near-zero (< threshold)
    # This is the input-dependent sparsity that varies per sentence/token
    ATTN_THRESHOLD = 0.01  # attention weights below this are "effectively zero"
    attn_sparsities = {}   # layer_idx → list of sparsity values per input
    attn_hooks = []

    def make_attn_hook(layer_idx):
        def hook_fn(mod, inp, out):
            # BertSelfAttention output includes attention_probs when output_attentions=True
            # But we hook the softmax's input/output directly via the dropout layer
            # Instead, access the attention_probs attribute if available
            if hasattr(mod, 'attention_probs_') and mod.attention_probs_ is not None:
                probs = mod.attention_probs_.detach()
                sp = float((probs < ATTN_THRESHOLD).sum().item() / probs.numel())
                attn_sparsities.setdefault(layer_idx, []).append(sp)
        return hook_fn

    # Hook into softmax to capture attention probs
    softmax_hooks = []
    def make_softmax_capture_hook(layer_idx, attn_module):
        """Hook after the matmul(Q, K^T) / sqrt(d) + mask, before/after softmax."""
        original_forward = attn_module.forward

        def patched_forward(*args, **kwargs):
            # Force output_attentions to capture attention weights
            kwargs['output_attentions'] = True
            outputs = original_forward(*args, **kwargs)
            # outputs = (context_layer, attention_probs) when the weights are
            # actually materialised. transformers >= 5 returns None here on the
            # SDPA and flash-attention paths, which never build the full
            # probability matrix -- hence attn_implementation="eager" below.
            if (isinstance(outputs, tuple) and len(outputs) >= 2
                    and outputs[1] is not None):
                attn_probs = outputs[1].detach()
                sp = float((attn_probs < ATTN_THRESHOLD).sum().item() / attn_probs.numel())
                attn_sparsities.setdefault(layer_idx, []).append(sp)
            return outputs

        attn_module.forward = patched_forward
        return original_forward  # return original to restore later

    # Patch attention modules
    original_forwards = []
    try:
        from transformers.models.bert.modeling_bert import BertSelfAttention
        # BertModel: layers at model.encoder.layer
        # BertForXxx: layers at model.bert.encoder.layer
        if hasattr(model, 'encoder'):
            encoder_layers = model.encoder.layer
        elif hasattr(model, 'bert'):
            encoder_layers = model.bert.encoder.layer
        else:
            raise AttributeError("Cannot find encoder layers")
        
        for layer_idx, layer in enumerate(encoder_layers):
            attn = layer.attention.self
            orig = make_softmax_capture_hook(layer_idx, attn)
            original_forwards.append((attn, orig))
        print(f"  Hooked {len(original_forwards)} attention layers for sparsity capture")
    except Exception as e:
        print(f"  Warning: Could not hook attention layers: {e}")

    count = 0
    with torch.no_grad():
        for i in range(N_SAMPLES):
            seq_len = seq_lengths[i % len(seq_lengths)]
            if texts is None:
                # synthetic path, kept only so --allow-synthetic reproduces the
                # original numbers exactly
                input_ids = torch.randint(0, 30522, (1, seq_len)).to(device)
                attention_mask = torch.ones(1, seq_len, dtype=torch.long).to(device)
            else:
                # Pad to exactly seq_len so the profiled M/K/P dimensions match
                # the synthetic run and the two are comparable layer for layer.
                enc = tokenizer(texts[i % len(texts)],
                                return_tensors="pt", truncation=True,
                                padding="max_length", max_length=seq_len)
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc["attention_mask"].to(device)
            profiler._input_idx = count
            model(input_ids=input_ids, attention_mask=attention_mask)
            count += 1
            if count % 50 == 0:
                print(f"  Profiled {count}/{N_SAMPLES} inputs...")

    profiler.remove_hooks()

    # Restore original attention forwards
    for attn, orig in original_forwards:
        attn.forward = orig

    import pandas as pd
    df = pd.DataFrame(profiler.records)

    # Add attention sparsity to relevant layers
    # Map each record to its attention layer (if applicable)
    if attn_sparsities:
        # Each attention layer produces Q, K, V, output projections
        # Assign attention sparsity to the layers in that block
        attn_sp_list = []
        for _, row in df.iterrows():
            layer_name = row['layer_name']
            input_idx = row['input_idx']
            # Extract layer index from name (e.g., "bert.encoder.layer.3.attention...")
            attn_sp = 0.0
            for lid in attn_sparsities:
                if f"layer.{lid}." in layer_name and "attention" in layer_name:
                    idx_in_list = input_idx if input_idx < len(attn_sparsities[lid]) else -1
                    if idx_in_list >= 0 and idx_in_list < len(attn_sparsities[lid]):
                        attn_sp = attn_sparsities[lid][idx_in_list]
                    break
            attn_sp_list.append(attn_sp)
        df['attention_sparsity'] = attn_sp_list
    else:
        df['attention_sparsity'] = 0.0

    if provenance:
        for k, v in provenance.items():
            df[k] = v

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"{model_name}_traces.csv")
    df.to_csv(path, index=False)

    print(f"\n  Summary for {model_name}:")
    print(f"  Layers profiled: {df['layer_name'].nunique()}")
    print(f"  Total records: {len(df)}")
    print(f"  Avg input sparsity:     {df['input_sparsity'].mean():.3f}")
    print(f"  Avg output sparsity:    {df['output_sparsity'].mean():.3f}")
    print(f"  Avg weight sparsity:    {df['weight_sparsity'].mean():.3f}")
    print(f"  Avg attention sparsity: {df['attention_sparsity'].mean():.3f}")
    if attn_sparsities:
        all_attn = [s for v in attn_sparsities.values() for s in v]
        print(f"  Attention sparsity range: [{min(all_attn):.3f}, {max(all_attn):.3f}]")
    print(f"  Saved to {path}")


MODEL_ID = "bert-base-uncased"


def main():
    # R5: only the --allow-synthetic paths draw random inputs; seed them.
    torch.manual_seed(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-file", help="real text, one passage per line")
    ap.add_argument("--dataset", help="sst2 | wikitext (needs `datasets`)")
    ap.add_argument("--allow-synthetic", action="store_true",
                    help="profile on random token IDs (the old behaviour)")
    ap.add_argument("--random-init", action="store_true",
                    help="use untrained weights (the old behaviour)")
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        from transformers import BertModel, BertConfig, BertTokenizerFast
    except ImportError:
        print("ERROR: 'transformers' not found. Run: pip install transformers")
        sys.exit(1)

    texts, prov = load_texts(args)
    pretrained = not args.random_init
    prov["weights"] = MODEL_ID if pretrained else "random-init"
    if not pretrained:
        print("\n  !! RANDOM INIT: sparsity from this run describes no real network.\n")

    tokenizer = None
    if texts is not None:
        tokenizer = BertTokenizerFast.from_pretrained(MODEL_ID)

    def make_bert():
        # attn_implementation="eager" is required, not a preference: the SDPA
        # and flash paths compute attention without ever forming the NxN
        # probability matrix, so attention_sparsity -- the headline
        # input-dependent feature in this project -- cannot be measured on them.
        if pretrained:
            return BertModel.from_pretrained(
                MODEL_ID, attn_implementation="eager").to(device).eval()
        config = BertConfig(
            hidden_size=768, num_hidden_layers=12,
            num_attention_heads=12, intermediate_size=3072,
            attn_implementation="eager",
        )
        return BertModel(config).to(device).eval()

    for label, sparsity in [("dense", None), ("pruned50", 0.50),
                            ("pruned70", 0.70), ("pruned90", 0.90)]:
        model = make_bert()
        if sparsity is not None:
            apply_unstructured_pruning(model, sparsity)
            model.eval()
        profile_bert(model, f"bert_{label}", device,
                     texts=texts, provenance=dict(prov), tokenizer=tokenizer)

    print(f"\nAll BERT profiles saved to {OUTPUT_DIR}")
    print(f"Provenance stamped into every CSV: {prov}")


if __name__ == '__main__':
    main()
