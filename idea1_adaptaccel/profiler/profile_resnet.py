"""
profile_resnet.py -- Profile ResNet-50 on REAL images with PRETRAINED weights.

FIX_PLAN.md M7. This script previously did:

    models.resnet50(weights=None)          # randomly initialised
    torch.randn(n, 3, 224, 224)            # Gaussian noise

Both halves are fatal to the thing being measured. Activation sparsity after a
ReLU is a property of a *trained* network responding to *natural image
statistics*; a random network fed noise produces roughly half zeros everywhere
by symmetry, which is a fact about ReLU, not about ResNet-50. Every
activation-sparsity number in the paper rests on this file.

It now refuses to produce traces from noise unless you explicitly ask, and it
stamps the provenance into every CSV so a synthetic trace can never be mistaken
for a real one later.

Usage:
    # real photographs you already have, as ImageFolder (class subdirectories)
    python -m profiler.profile_resnet --data-dir /path/to/imagenet/val

    # or let torchvision fetch Imagenette (~1.5 GB, 10 ImageNet classes)
    python -m profiler.profile_resnet --download

    # explicitly reproduce the old, meaningless behaviour
    python -m profiler.profile_resnet --allow-synthetic

    # B9: MobileNetV2 (depthwise-separable, inverted residuals), traces kept
    # apart from the 15 sets sections 3-6 are computed on
    python -m profiler.profile_resnet --download --arch mobilenet_v2 \
        --out-dir profiling_data/b9

Needs network access on first run (pretrained weights, and the dataset if
--download).

Note for python.org installs on macOS: if HTTPS downloads fail with
CERTIFICATE_VERIFY_FAILED, that interpreter has no CA bundle wired up. Either
run the "Install Certificates.command" that ships beside it, or:
    export SSL_CERT_FILE=$(python3 -c "import certifi;print(certifi.where())")
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from profiler.workload_profiler import LayerProfiler

N_SAMPLES = 200
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "profiling_data")

# Standard ImageNet preprocessing. Using the wrong normalisation would shift
# every activation distribution and therefore every sparsity number.
PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def build_loader(args):
    """Return (loader, provenance). Refuses noise unless explicitly allowed."""
    if args.data_dir:
        from torchvision.datasets import ImageFolder
        ds = ImageFolder(args.data_dir, transform=PREPROCESS)
        if len(ds) == 0:
            sys.exit(f"No images under {args.data_dir}. ImageFolder expects "
                     f"class subdirectories, e.g. {args.data_dir}/n01440764/*.JPEG")
        print(f"  Real images: {len(ds)} from {args.data_dir}")
        return (DataLoader(ds, batch_size=16, shuffle=True,
                           generator=torch.Generator().manual_seed(0)),
                {"data_source": f"imagefolder:{os.path.basename(args.data_dir.rstrip('/'))}"})

    if args.download:
        try:
            from torchvision.datasets import Imagenette
        except ImportError:
            sys.exit("This torchvision is too old for Imagenette. Either upgrade "
                     "(pip install -U torchvision) or pass --data-dir.")
        root = os.path.join(OUTPUT_DIR, "_datasets")
        os.makedirs(root, exist_ok=True)
        try:
            ds = Imagenette(root, split="val", size="320px",
                            download=True, transform=PREPROCESS)
        except RuntimeError as e:
            # torchvision raises if the archive is already unpacked
            if "already" in str(e).lower() or "exists" in str(e).lower():
                ds = Imagenette(root, split="val", size="320px",
                                download=False, transform=PREPROCESS)
            else:
                raise
        print(f"  Real images: {len(ds)} from Imagenette (val)")
        return (DataLoader(ds, batch_size=16, shuffle=True,
                           generator=torch.Generator().manual_seed(0)),
                {"data_source": "imagenette-val"})

    if args.allow_synthetic:
        print("\n  !! SYNTHETIC INPUTS: Gaussian noise, not images.")
        print("  !! Activation sparsity from this run is meaningless and the")
        print("  !! traces are stamped data_source=gaussian-noise.\n")
        images = torch.randn(N_SAMPLES, 3, 224, 224)
        labels = torch.randint(0, 1000, (N_SAMPLES,))
        return (DataLoader(TensorDataset(images, labels), batch_size=16),
                {"data_source": "gaussian-noise"})

    sys.exit(
        "Refusing to profile on noise.\n"
        "  Activation sparsity measured on Gaussian noise is a property of ReLU,\n"
        "  not of ResNet-50, and it is what made every sparsity number in the\n"
        "  original draft unusable (FIX_PLAN.md M7).\n\n"
        "  Pick one:\n"
        "    --data-dir /path/to/images   real photographs (ImageFolder layout)\n"
        "    --download                   fetch Imagenette (~1.5 GB)\n"
        "    --allow-synthetic            reproduce the old behaviour on purpose\n")


ARCHS = {   # name -> (constructor, pretrained weights)
    "resnet50":     (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
    "mobilenet_v2": (models.mobilenet_v2, models.MobileNet_V2_Weights.IMAGENET1K_V2),
}


def build_model(pretrained, device, arch="resnet50"):
    ctor, weights = ARCHS[arch]
    if pretrained:
        model = ctor(weights=weights)
        tag = "IMAGENET1K_V2"
    else:
        model = ctor(weights=None)
        tag = "random-init"
    return model.to(device).eval(), tag


def apply_unstructured_pruning(model, sparsity):
    """Global unstructured L1 pruning over all Conv2d and Linear weights.

    Unstructured is deliberate -- it is what the cost model's 'iid' nonzero-row
    estimator assumes (PROJECT_REPORT.md section 2.4). If you switch to
    structured/channel pruning, switch config_modes.NNZ_ROW_MODEL to 'sqrt' as
    well, or the cost model will misestimate in the other direction.
    """
    targets = [(m, 'weight') for m in model.modules()
               if isinstance(m, (nn.Conv2d, nn.Linear))]
    if not targets:
        return model
    prune.global_unstructured(targets, pruning_method=prune.L1Unstructured,
                              amount=sparsity)
    for m, _ in targets:
        prune.remove(m, 'weight')
    return model


def profile_one(name, model, loader, provenance, out_dir=OUTPUT_DIR):
    print(f"\n{'='*60}\n  Profiling: {name}\n{'='*60}")
    profiler = LayerProfiler(model, model_name=name)
    df = profiler.profile(loader, n_samples=N_SAMPLES)
    profiler.save_traces(out_dir, provenance=provenance)
    print(f"\n  Layers profiled:     {df['layer_name'].nunique()}")
    print(f"  Avg input sparsity:  {df['input_sparsity'].mean():.3f}")
    print(f"  Avg output sparsity: {df['output_sparsity'].mean():.3f}")
    print(f"  Avg weight sparsity: {df['weight_sparsity'].mean():.3f}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", help="ImageFolder of real images")
    ap.add_argument("--download", action="store_true",
                    help="fetch Imagenette via torchvision")
    ap.add_argument("--allow-synthetic", action="store_true",
                    help="profile on Gaussian noise (reproduces the old behaviour)")
    ap.add_argument("--random-init", action="store_true",
                    help="use untrained weights (reproduces the old behaviour)")
    ap.add_argument("--arch", default="resnet50", choices=sorted(ARCHS))
    ap.add_argument("--out-dir", default=OUTPUT_DIR,
                    help="where traces go (datasets stay in profiling_data/_datasets)")
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    loader, prov = build_loader(args)
    pretrained = not args.random_init
    prov["weights"] = (("IMAGENET1K_V2" if args.arch == "resnet50" else f"{args.arch}:IMAGENET1K_V2")
                       if pretrained else "random-init")
    if not pretrained:
        print("\n  !! RANDOM INIT: sparsity from this run describes no real network.\n")

    for label, sparsity in [("dense", None), ("pruned50", 0.50),
                            ("pruned70", 0.70), ("pruned90", 0.90)]:
        model, tag = build_model(pretrained, device, args.arch)
        if sparsity is not None:
            apply_unstructured_pruning(model, sparsity)
            model.eval()
        profile_one(f"{args.arch}_{label}", model, loader, dict(prov), args.out_dir)

    print(f"\nAll {args.arch} profiles saved to {args.out_dir}")
    print(f"Provenance stamped into every CSV: {prov}")


if __name__ == '__main__':
    main()
