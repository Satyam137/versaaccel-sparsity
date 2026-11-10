"""
profile_nm_models.py — Profile models with N:M structured pruning (2:4).

Unlike Idea 1's unstructured pruning (random zeros), N:M pruning
enforces exactly 2 nonzeros per group of 4 consecutive weights.

This creates workloads where the NM config is competitive because:
  - Sparsity is exactly 50% and STRUCTURED
  - NM hardware can exploit the known pattern with simple MUX routing
  - No Benes network overhead (unlike D-M-R for unstructured sparsity)

These traces feed the N:M result in PROJECT_REPORT.md section 6, so the same
M7 rule applies as to the Idea 1 profilers: a randomly-initialised model fed
Gaussian noise measures ReLU, not ResNet-50. This script reuses profile_resnet's
input loader and likewise REFUSES to run on synthetic data unless told to.

Usage:
    python idea2_veriflex/profile_nm_models.py --download
    python idea2_veriflex/profile_nm_models.py --data-dir /path/to/images
    python idea2_veriflex/profile_nm_models.py --allow-synthetic   # old behaviour

Note for python.org installs on macOS: if HTTPS downloads fail with
CERTIFICATE_VERIFY_FAILED, either run the "Install Certificates.command" that
ships with the interpreter, or set
    export SSL_CERT_FILE=$(python3 -c "import certifi;print(certifi.where())")
"""

import argparse
import itertools
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from idea2_veriflex.nm_sparsity.nm_sparsity import nm_prune_weights, nm_compatibility_score
# Structural sparsity (section 2.5). The N:M sets are the sharpest test of N2:
# a 2:4 pattern is structured along the WRONG axis, so it should yield ~0
# skippable k-slices on a real pruned model, not just on a synthetic pattern.
from idea1_adaptaccel.profiler.workload_profiler import kslice_structure

OUTPUT_DIR = os.path.join(PROJECT_DIR, "idea1_adaptaccel", "profiling_data")
N_SAMPLES = 200


def apply_nm_pruning_to_model(model, N=2, M_ratio=4):
    """
    Apply N:M structured pruning to all Conv2d and Linear layers.

    For each layer: flatten weights, apply 2:4 pruning (keep top-2 per group of 4),
    reshape back.
    """
    total_params = 0
    pruned_params = 0

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight.data
            original_shape = w.shape
            flat = w.detach().cpu().numpy().flatten()
            pruned_flat = nm_prune_weights(flat, N=N, M_ratio=M_ratio)

            total_params += len(flat)
            pruned_params += np.count_nonzero(flat) - np.count_nonzero(pruned_flat)

            module.weight.data = torch.tensor(
                pruned_flat.reshape(original_shape),
                dtype=w.dtype, device=w.device
            )

    actual_sparsity = 1.0 - np.count_nonzero(pruned_flat) / len(pruned_flat) if len(flat) > 0 else 0
    print(f"  Applied 2:4 NM pruning: {total_params:,} params, "
          f"~50% sparsity enforced per group")
    return model


def profile_model_with_hooks(model, model_name, input_fn, device, n_samples=N_SAMPLES):
    """Profile a model using forward hooks, same as Idea 1's profiler."""

    records = []
    hooks = []

    def make_hook(name, module):
        def hook_fn(mod, inp, out):
            x = inp[0] if isinstance(inp, tuple) else inp
            y = out if isinstance(out, torch.Tensor) else (out[0] if isinstance(out, tuple) else None)
            if x is None or not isinstance(x, torch.Tensor):
                return
            if y is None or not isinstance(y, torch.Tensor):
                return

            input_shape = tuple(x.shape)
            output_shape = tuple(y.shape)

            # Dimensions
            if isinstance(mod, nn.Linear):
                K = mod.in_features
                P = mod.out_features
                M = 1
                for d in input_shape[:-1]:
                    M *= d
                op_type = 'MM' if M > 1 else 'MV'
            elif isinstance(mod, nn.Conv2d):
                Cin = mod.in_channels
                Cout = mod.out_channels
                kH, kW = mod.kernel_size
                K = Cin * kH * kW
                P = Cout
                batch = input_shape[0]
                H_out = output_shape[2] if len(output_shape) > 2 else 1
                W_out = output_shape[3] if len(output_shape) > 3 else 1
                M = batch * H_out * W_out
                op_type = 'MM'
            else:
                return

            # Sparsity
            input_sp = float((x == 0).sum().item() / x.numel()) if x.numel() > 0 else 0.0
            output_sp = float((y == 0).sum().item() / y.numel()) if y.numel() > 0 else 0.0
            weight_sp = float((mod.weight == 0).sum().item() / mod.weight.numel()) if hasattr(mod, 'weight') else 0.0

            # NM compatibility
            if hasattr(mod, 'weight'):
                nm_compat = nm_compatibility_score(mod.weight.detach().cpu())
            else:
                nm_compat = 0.0

            _ks = kslice_structure(mod, x)
            records.append({
                'input_idx': profile_model_with_hooks.current_idx,
                'layer_name': name,
                'layer_type': mod.__class__.__name__,
                'op_type': op_type,
                'M': M, 'K': K, 'P': P,
                'input_sparsity': round(input_sp, 4),
                'output_sparsity': round(output_sp, 4),
                'weight_sparsity': round(weight_sp, 4),
                'nm_compatibility': round(nm_compat, 4),
                'a_kslice_zero': _ks[0], 'w_kslice_zero': _ks[1], 'skippable_frac': _ks[2],
                'input_shape': str(input_shape),
                'output_shape': str(output_shape),
            })
        return hook_fn

    # Register hooks
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            h = module.register_forward_hook(make_hook(name, module))
            hooks.append(h)

    # Run inference
    model.eval()
    profile_model_with_hooks.current_idx = 0

    with torch.no_grad():
        for i in range(n_samples):
            profile_model_with_hooks.current_idx = i
            x = input_fn(device)
            model(x)
            if (i + 1) % 50 == 0:
                print(f"  Profiled {i+1}/{n_samples}...")

    # Remove hooks
    for h in hooks:
        h.remove()

    df = pd.DataFrame(records)
    return df


def build_input_fn(args, device):
    """Return (input_fn, provenance). Reuses profile_resnet's loader so the two
    scripts cannot drift apart on what counts as a real input."""
    from idea1_adaptaccel.profiler.profile_resnet import build_loader
    loader, prov = build_loader(args)
    if prov["data_source"] == "gaussian-noise":
        return (lambda dev: torch.randn(1, 3, 224, 224).to(dev)), prov

    # Flatten the loader into a single-image stream, cycling if it runs short.
    def gen():
        while True:
            for images, _ in loader:
                for i in range(images.size(0)):
                    yield images[i:i + 1]
    stream = gen()
    return (lambda dev: next(stream).to(dev)), prov


def main():
    # R5: only the --allow-synthetic paths draw random inputs; seed them.
    torch.manual_seed(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", help="ImageFolder of real images")
    ap.add_argument("--download", action="store_true", help="fetch Imagenette")
    # consumed by profile_bert.load_texts for the BERT half
    ap.add_argument("--text-file", help="real text for BERT, one passage per line")
    ap.add_argument("--dataset", help="sst2 | wikitext, for the BERT half")
    ap.add_argument("--allow-synthetic", action="store_true",
                    help="profile on Gaussian noise (the old behaviour)")
    ap.add_argument("--random-init", action="store_true",
                    help="use untrained weights (the old behaviour)")
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    resnet_input, prov = build_input_fn(args, device)
    pretrained = not args.random_init
    prov["weights"] = "IMAGENET1K_V2" if pretrained else "random-init"
    if not pretrained:
        print("\n  !! RANDOM INIT: sparsity from this run describes no real network.\n")

    # ========================================================================
    # 1) ResNet50 with N:M pruning
    # ========================================================================
    print(f"\n{'='*60}")
    print(f"  Profiling: ResNet50 with 2:4 NM pruning")
    print(f"{'='*60}")

    from torchvision import models
    if pretrained:
        model_resnet = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2).to(device)
    else:
        model_resnet = models.resnet50(weights=None).to(device)
    model_resnet = apply_nm_pruning_to_model(model_resnet)
    model_resnet.eval()

    df_resnet = profile_model_with_hooks(
        model_resnet, "resnet50_nm24", resnet_input, device, n_samples=N_SAMPLES
    )

    for k, v in prov.items():
        df_resnet[k] = v
    path = os.path.join(OUTPUT_DIR, "resnet50_nm24_traces.csv")
    df_resnet.to_csv(path, index=False)

    print(f"\n  Summary for resnet50_nm24:")
    print(f"  Layers: {df_resnet['layer_name'].nunique()}")
    print(f"  Avg input sparsity:     {df_resnet['input_sparsity'].mean():.3f}")
    print(f"  Avg weight sparsity:    {df_resnet['weight_sparsity'].mean():.3f}")
    print(f"  Avg NM compatibility:   {df_resnet['nm_compatibility'].mean():.3f}")
    print(f"  Saved to {path}")

    # ========================================================================
    # 2) BERT with N:M pruning
    # ========================================================================
    print(f"\n{'='*60}")
    print(f"  Profiling: BERT-Base with 2:4 NM pruning")
    print(f"{'='*60}")

    try:
        from transformers import BertModel, BertConfig
    except ImportError:
        print("  WARNING: transformers not installed, skipping BERT NM")
        print(f"\n[DONE] NM profiling complete (ResNet only).")
        return

    # Same M7 rule as the ResNet half: pretrained weights and real text, or
    # nothing. Reuses profile_bert's loader so the two cannot disagree about
    # what a real input is.
    from idea1_adaptaccel.profiler.profile_bert import load_texts, MODEL_ID
    bert_texts, bert_prov = load_texts(args)
    if pretrained:
        from transformers import BertTokenizerFast
        tokenizer = BertTokenizerFast.from_pretrained(MODEL_ID)
        model_bert = BertModel.from_pretrained(
            MODEL_ID, attn_implementation="eager").to(device)
    else:
        tokenizer = None
        config = BertConfig(
            hidden_size=768, num_hidden_layers=12,
            num_attention_heads=12, intermediate_size=3072,
            attn_implementation="eager",
        )
        model_bert = BertModel(config).to(device)
    bert_prov["weights"] = MODEL_ID if pretrained else "random-init"
    model_bert = apply_nm_pruning_to_model(model_bert)
    model_bert.eval()

    # Profile BERT
    records_bert = []
    hooks_bert = []

    def make_bert_hook(name, module):
        def hook_fn(mod, inp, out):
            x = inp[0] if isinstance(inp, tuple) else inp
            y = out if isinstance(out, torch.Tensor) else (out[0] if isinstance(out, tuple) else None)
            if x is None or not isinstance(x, torch.Tensor):
                return
            if y is None or not isinstance(y, torch.Tensor):
                return

            if not isinstance(mod, nn.Linear):
                return

            K = mod.in_features
            P = mod.out_features
            input_shape = tuple(x.shape)
            M = 1
            for d in input_shape[:-1]:
                M *= d

            input_sp = float((x == 0).sum().item() / x.numel())
            output_sp = float((y == 0).sum().item() / y.numel())
            weight_sp = float((mod.weight == 0).sum().item() / mod.weight.numel())
            nm_compat = nm_compatibility_score(mod.weight.detach().cpu())

            _ks = kslice_structure(mod, x)
            records_bert.append({
                'input_idx': main.bert_idx,
                'layer_name': name,
                'layer_type': 'Linear',
                'op_type': 'MM',
                'M': M, 'K': K, 'P': P,
                'input_sparsity': round(input_sp, 4),
                'output_sparsity': round(output_sp, 4),
                'weight_sparsity': round(weight_sp, 4),
                'nm_compatibility': round(nm_compat, 4),
                'a_kslice_zero': _ks[0], 'w_kslice_zero': _ks[1], 'skippable_frac': _ks[2],
                'attention_sparsity': 0.0,
                'input_shape': str(input_shape),
                'output_shape': str(tuple(y.shape)),
            })
        return hook_fn

    for name, module in model_bert.named_modules():
        if isinstance(module, nn.Linear):
            h = module.register_forward_hook(make_bert_hook(name, module))
            hooks_bert.append(h)

    main.bert_idx = 0
    seq_lengths = [32, 64, 128]

    with torch.no_grad():
        for i in range(N_SAMPLES):
            main.bert_idx = i
            seq_len = seq_lengths[i % len(seq_lengths)]
            if bert_texts is None or tokenizer is None:
                input_ids = torch.randint(0, 30522, (1, seq_len)).to(device)
            else:
                input_ids = tokenizer(
                    bert_texts[i % len(bert_texts)], return_tensors="pt",
                    truncation=True, padding="max_length",
                    max_length=seq_len)["input_ids"].to(device)
            attention_mask = torch.ones(1, seq_len, dtype=torch.long).to(device)
            model_bert(input_ids=input_ids, attention_mask=attention_mask)
            if (i + 1) % 50 == 0:
                print(f"  Profiled {i+1}/{N_SAMPLES}...")

    for h in hooks_bert:
        h.remove()

    df_bert = pd.DataFrame(records_bert)
    path_bert = os.path.join(OUTPUT_DIR, "bert_nm24_traces.csv")
    for k, v in bert_prov.items():
        df_bert[k] = v
    df_bert.to_csv(path_bert, index=False)

    print(f"\n  Summary for bert_nm24:")
    print(f"  Layers: {df_bert['layer_name'].nunique()}")
    print(f"  Avg input sparsity:     {df_bert['input_sparsity'].mean():.3f}")
    print(f"  Avg weight sparsity:    {df_bert['weight_sparsity'].mean():.3f}")
    print(f"  Avg NM compatibility:   {df_bert['nm_compatibility'].mean():.3f}")
    print(f"  Saved to {path_bert}")

    print(f"\n[DONE] NM profiling complete!")
    print(f"  Re-run integrated_scheduler.py to see NM config selection.")


if __name__ == '__main__':
    main()
