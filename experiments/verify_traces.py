"""R6: the shipped traces are the ones every number was computed from.

The profiling traces are distributed with the repository (about 45 MB of CSV,
including B9's). They hold per-layer statistics only -- shapes, sparsity fractions,
k-slice structure and provenance stamps -- and no image, text or token from any
dataset, so shipping them redistributes no dataset content. Re-profiling needs
the datasets and pretrained weights (profiler/*.py --download / --dataset).

This checks every trace against MANIFEST.sha256 and checks its provenance:
the real sets must say which dataset and which pretrained weights produced
them, and the synthetic stress sets must not claim to be real.

Usage: python experiments/verify_traces.py [--update]   (--update rewrites the manifest)
Exit status 1 on any mismatch.
"""
import argparse, glob, hashlib, os, sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "idea1_adaptaccel", "profiling_data")
MANIFEST = os.path.join(DATA, "MANIFEST.sha256")
REAL = ("resnet50_", "bert_", "mobilenet_v2_", "opt125m_")


def traces():
    return sorted(glob.glob(os.path.join(DATA, "*_traces.csv")) +
                  glob.glob(os.path.join(DATA, "b9", "*_traces.csv")))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true")
    args = ap.parse_args()
    files = traces()
    rel = {p: os.path.relpath(p, DATA) for p in files}
    if args.update:
        with open(MANIFEST, "w") as f:
            for p in files:
                f.write(f"{sha256(p)}  {rel[p]}\n")
        print(f"  wrote {MANIFEST} ({len(files)} traces)")
    want = dict(reversed(l.split(None, 1)) for l in open(MANIFEST).read().splitlines() if l.strip())
    want = {k.strip(): v for k, v in want.items()}
    bad = 0
    print(f"  {'trace':38s} {'rows':>8s}  {'sha256':8s}  provenance")
    for p in files:
        name = rel[p]
        ok_hash = want.get(name) == sha256(p)
        d = pd.read_csv(p, usecols=lambda c: c in ("data_source", "weights", "input_idx"))
        base = os.path.basename(name)
        if base.startswith(REAL):
            src = d["data_source"].unique().tolist() if "data_source" in d else []
            wts = d["weights"].unique().tolist() if "weights" in d else []
            ok_prov = (len(src) == 1 and len(wts) == 1 and
                       src[0] not in ("gaussian-noise", "random-tokens") and wts[0] != "random-init")
            prov = f"{src[0] if src else '?'} / {wts[0] if wts else '?'}"
        else:
            ok_prov = "data_source" not in d or not str(d["data_source"].iloc[0]).startswith(("imagenette", "sst2"))
            prov = "synthetic stress set (by design)"
        bad += (not ok_hash) + (not ok_prov)
        print(f"  {name:38s} {len(d):8,d}  {'ok' if ok_hash else 'CHANGED':8s}  "
              f"{prov}{'' if ok_prov else '   <-- PROVENANCE'}")
    missing = set(want) - set(rel.values())
    for m in sorted(missing):
        print(f"  {m:38s}  MISSING")
    bad += len(missing)
    print(f"\n  {len(files) - bad if bad <= len(files) else 0} of {len(files)} traces verified"
          if bad else f"\n  all {len(files)} traces match the manifest, provenance intact")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
