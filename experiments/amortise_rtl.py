"""B10 + B12: fill amortisation and streamed activations, checked and measured on the RTL.

B10 (-DVERSAACCEL_AMORTISE) issues every tile of a GEMM back to back through one
pipeline: each schedule entry carries its tile, each PE clears and hands its
result to a two-bank output buffer on a diagonal wavefront, and the skew fill
is paid once per GEMM instead of once per tile. B12 (-DVERSAACCEL_STREAM, needs
AMORTISE) also overlaps the activation load with compute through a two-tile-row
ring buffer. Both are checked here with tb_stream.sv, which runs the loader and
the collector concurrently, obeying din_ready / dout_ready.

Part 1  random sparse GEMMs, default buffers, every build and tile shape.
        Every output element correct, with and without random stalls on both
        ports; and cycles equal the accounting plus ONE constant per build:
          resident, per-tile fill  L + n(K + sr + sc - 1)       skip off
                                   L + sum(active) + n(sr + sc)  skip on
          AMORTISE                 L + pipe_cycles(entries per tile)
        with L = K*P + M*K the load and pipe_cycles() the cycle model of the
        pipelined generator, output banks and drain below.
Part 2  STREAM: GEMMs whose activations do not fit the buffer (only the ring
        makes them possible), with stalls; correctness, and cycles beyond the load.
Part 3  real slices of pruned ResNet-50 / BERT layers on enlarged buffers, every
        build and shape: the zero-skip saving once the fill no longer dilutes it,
        and the end-to-end time once the activation load is overlapped.

Usage: python experiments/amortise_rtl.py [--quick | --real-only]
  (~75 min; --quick: parts 1-2, ~25 min; --real-only: part 3, ~50 min, merged
  into the parts 1-2 rows already in results/amortise_rtl.csv)
Writes results/amortise_rtl.csv (--quick writes amortise_rtl_quick.csv if it exists).
"""
import argparse, os, re, subprocess, sys, tempfile
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tile_geometry as tg                                    # noqa: E402
from regroup_rtl import write_hex, quantise, capture_values, RTL, LAYERS  # noqa: E402

NR = NC = 16
SHAPES = {0: ("16x16", NR, NC, NR, NC),            # name, tile_r, tile_c, span_r, span_c
          1: ("8x32 wide", NR // 2, 2 * NC, NR // 2, NC),
          2: ("32x8 tall", 2 * NR, NC // 2, NR, NC // 2)}
BUILDS = {"resident": ["-DVERSAACCEL_RESHAPE"],
          "amortise": ["-DVERSAACCEL_RESHAPE", "-DVERSAACCEL_AMORTISE"],
          "stream":   ["-DVERSAACCEL_RESHAPE", "-DVERSAACCEL_AMORTISE", "-DVERSAACCEL_STREAM"]}
PART1_GEOMS = [(32, 8, 64), (16, 16, 32), (48, 4, 96), (8, 32, 16)]   # fit IBUF 256 / WBUF 512
PART2_GEOMS = [(100, 4, 20), (70, 8, 32), (200, 2, 64)]               # A exceeds IBUF 256
# Part 3: enlarged buffers; bitmaps bounded to the GEMMs actually run
BIG = dict(IBUF_DEPTH=4096, WBUF_DEPTH=4096, K_MAX=32, M_MAX=128, P_MAX=128)
PART3_GEOMS = {"resnet50_pruned90": (128, 32, 128), "bert_pruned90": (32, 32, 128)}
PART3_LAYERS = {"resnet50_pruned90": ["layer2.1.conv1", "layer3.2.conv1", "layer3.4.conv3",
                                      "layer4.0.conv1"],
                "bert_pruned90": LAYERS["bert_pruned90"]}


# --------------------------------------------------------------------------
# Accounting
# --------------------------------------------------------------------------
def tile_active(a_nz, b_nz, tr, tc):
    """Live k-steps of every logical tr x tc tile, in issue order (tile_j fastest)."""
    A = tg._pad_rows(a_nz, tr)
    B = tg._pad_rows(b_nz, tc)
    K = A.shape[1]
    a_t = A.reshape(-1, tr, K).any(axis=1)
    b_t = B.reshape(-1, tc, K).any(axis=1)
    return (a_t[:, None, :] & b_t[None, :, :]).sum(axis=2).ravel()


FEED_LAT = 3   # versaaccel_top's operand-feed stages (Rev 20)


def resident_cycles(act, K, sr, sc, skip):
    n = len(act)
    return int(act.sum()) + n * (sr + sc) if skip else n * (K + sr + sc - 1)


def pipe_cycles(entries, span_r, nc=NC):
    """Cycle model of ST_GEMM_PIPE with the wide drain and no stalls: cycles
    from the first pipelined cycle to the last output beat. entries[t] is the
    number of schedule entries tile t issues (K dense; its live steps, or 1 if
    none, when skipping). Mirrors versaaccel_top: a tile's first entry (and the
    final terminator) waits until the output bank its wavefront will fill is
    free; that bank becomes drainable when the wavefront reaches index
    span_r-1+FEED_LAT (Rev 20: the operand feed is FEED_LAT stages, so the
    wavefront reaches the array that much later) and is drained in nc beats,
    one bank at a time, in capture order."""
    n = len(entries)
    busy, full = [0, 0], [0, 0]
    cbank = cap_any = dbank = draining = beat = 0
    ti, left, t, drained = 0, 0, 0, 0
    waves = []                                   # (cycle the bank becomes full, bank)
    while drained < n:
        nbusy, nfull = busy[:], full[:]
        ncbank, ncap, ndbank, ndrain, nbeat = cbank, cap_any, dbank, draining, beat
        if ti <= n and left == 0:                # a first entry or the terminator
            if not cap_any or not busy[cbank]:
                if cap_any:
                    nbusy[cbank] = 1
                    ncbank = 1 - cbank
                    waves.append((t + span_r + FEED_LAT, cbank))
                ncap = 1
                if ti == n:
                    ti = n + 1                   # terminator issued
                else:
                    left = entries[ti] - 1
                    if left == 0:
                        ti += 1
        elif left > 0:
            left -= 1
            if left == 0:
                ti += 1
        for c, b in waves:
            if c == t:
                nfull[b] = 1
        if draining:
            if beat == nc - 1:
                drained += 1
                nfull[dbank] = nbusy[dbank] = 0
                ndbank, nbeat = 1 - dbank, 0
                ndrain = 1 if full[1 - dbank] else 0
            else:
                nbeat = beat + 1
        elif full[dbank]:
            ndrain, nbeat = 1, 0
        busy, full, cbank, cap_any, dbank, draining, beat = \
            nbusy, nfull, ncbank, ncap, ndbank, ndrain, nbeat
        t += 1
    return t


def accounting(build, A, B, shape, skip):
    _, tr, tc, sr, sc = SHAPES[shape]
    K = A.shape[1]
    act = tile_active(A != 0, B.T != 0, tr, tc)
    if build == "resident":
        return resident_cycles(act, K, sr, sc, skip)
    entries = np.maximum(act, 1) if skip else np.full(len(act), K)
    return pipe_cycles(entries, sr)


# --------------------------------------------------------------------------
# RTL
# --------------------------------------------------------------------------
def build(tmp, flags, M, K, P, pkg=None):
    exe = os.path.join(tmp, f"tb_{'_'.join(f[13:] for f in flags)}_{M}_{K}_{P}{'_big' if pkg else ''}")
    srcs = sorted(os.path.join(RTL, "versaaccel_rtl", f)
                  for f in os.listdir(os.path.join(RTL, "versaaccel_rtl")) if f.endswith(".sv"))
    subprocess.run(["iverilog", "-g2012", *flags, "-o", exe, f'-DCPE_WEIGHT_DIR="{RTL}/weights"',
                    f"-DST_M={M}", f"-DST_K={K}", f"-DST_P={P}",
                    pkg or os.path.join(RTL, "common", "versaaccel_pkg.sv"), *srcs,
                    os.path.join(RTL, "testbenches", "tb_stream.sv")],
                   check=True, capture_output=True)   # iverilog's "sorry:" notes are benign
    return exe


def big_package(tmp):
    txt = open(os.path.join(RTL, "common", "versaaccel_pkg.sv")).read()
    for name, v in BIG.items():
        txt, n = re.subn(rf"parameter int {name}(\s+)= [A-Z_0-9]+;", rf"parameter int {name}\g<1>= {v};", txt)
        assert n == 1, name
    path = os.path.join(tmp, "pkg_big.sv")
    open(path, "w").write(txt)
    return path


def run(exe, tmp, A, B, shape=0, bp=0):
    fa, fb = os.path.join(tmp, "a.hex"), os.path.join(tmp, "b.hex")
    write_hex(fa, A); write_hex(fb, B)
    out = subprocess.run(["vvp", "-n", exe, f"+A={fa}", f"+B={fb}", f"+SHAPE={shape}", f"+BP={bp}"],
                         capture_output=True, text=True, check=True).stdout
    if "ERROR" in out:
        raise RuntimeError(out[:400])
    c_off, c_on, checks, err = map(int, [l for l in out.splitlines()
                                         if l.startswith("RESULT")][0].split()[1:])
    return c_off, c_on, checks, err


def random_operands(rng, M, K, P, structured):
    da, db = rng.choice([0.0, 0.5, 0.8, 0.95], size=2)
    A = rng.integers(-9, 10, (M, K)); A[rng.random(A.shape) < da] = 0
    B = rng.integers(-9, 10, (K, P)); B[rng.random(B.shape) < db] = 0
    if structured:                                # k-slices zero across all of A
        A[:, rng.random(K) < 0.5] = 0
    return A, B


# --------------------------------------------------------------------------
def part1(tmp, rng, rows):
    print("=== Part 1: random sparse GEMMs -- correctness and accounting ===")
    ok = True
    for M, K, P in PART1_GEOMS:
        exes = {b: build(tmp, f, M, K, P) for b, f in BUILDS.items() if b != "stream"}
        for b, exe in exes.items():
            for shape, (name, *_r) in SHAPES.items():
                seen, checks, errs = set(), 0, 0
                for i in range(4):
                    A, B = random_operands(rng, M, K, P, structured=i % 2 == 1)
                    L = K * P + M * K
                    c_off, c_on, ch, err = run(exe, tmp, A, B, shape)
                    seen.add((c_off - L - accounting(b, A, B, shape, False),
                              c_on - L - accounting(b, A, B, shape, True)))
                    _, _, ch2, err2 = run(exe, tmp, A, B, shape, bp=25)      # stalls: correctness
                    checks += ch + ch2; errs += err + err2
                exact = len(seen) == 1 and len(set(next(iter(seen)))) == 1
                ok &= exact and errs == 0
                rows.append(dict(part="random", build=b, M=M, K=K, P=P, shape=name,
                                 offsets=str(sorted(seen)), checks=checks, errors=errs))
                print(f"  {b:9s} A{M}x{K} B{K}x{P} {name:9s} (RTL-acct off,on) {sorted(seen)}"
                      f"  {checks:,} checks, {errs} errors")
    return ok


def part2(tmp, rng, rows):
    print("\n=== Part 2: STREAM -- activations larger than the buffer ===")
    ok = True
    for M, K, P in PART2_GEOMS:
        exe = build(tmp, BUILDS["stream"], M, K, P)
        for shape, (name, tr, *_r) in SHAPES.items():
            if 2 * tr * K > 256:
                continue                          # ring of two tile-rows must fit IBUF
            for bp in (0, 25):
                A, B = random_operands(rng, M, K, P, structured=True)
                L = K * P + M * K
                c_off, c_on, ch, err = run(exe, tmp, A, B, shape, bp)
                ok &= err == 0
                rows.append(dict(part="stream", build="stream", M=M, K=K, P=P, shape=name, bp=bp,
                                 load=L, total_off=c_off, total_on=c_on, checks=ch, errors=err))
                print(f"  A{M}x{K} ({M * K} > 256 entries) B{K}x{P} {name:9s} stalls {bp:2d}%: "
                      f"{ch:,} checks, {err} errors; cycles {c_off} / {c_on} vs load {L}")
    return ok


def part3(tmp, rng, rows):
    print("\n=== Part 3: real pruned-layer slices, enlarged buffers ===")
    pkg = big_package(tmp)
    img, txt = tg.resnet_inputs(1)[0], tg.bert_inputs(1)[0]
    models = {n: m for gen in (tg.resnet_models(), tg.bert_models())
              for n, m in gen if n in PART3_LAYERS}
    for mname, lnames in PART3_LAYERS.items():
        M, K, P = PART3_GEOMS[mname]
        exes = {b: build(tmp, f, M, K, P, pkg) for b, f in BUILDS.items()}
        call = (lambda m, x: m(x)) if mname.startswith("resnet") else (lambda m, x: m(**x))
        ops = capture_values(models[mname], set(lnames),
                             img if mname.startswith("resnet") else txt, call)
        for ln in lnames:
            Af, Bf = ops[ln]
            for sl in range(2):
                r0 = rng.integers(0, Af.shape[0] - M + 1)
                k0 = rng.integers(0, Af.shape[1] - K + 1)
                p0 = rng.integers(0, Bf.shape[1] - P + 1)
                A = quantise(Af[r0:r0 + M, k0:k0 + K])
                B = quantise(Bf[k0:k0 + K, p0:p0 + P])
                L = K * P + M * K
                for shape, (name, *_r) in SHAPES.items():
                    rec = dict(part="real", model=mname, layer=ln, slice=sl, M=M, K=K, P=P,
                               shape=name, load=L)
                    for b, exe in exes.items():
                        c_off, c_on, ch, err = run(exe, tmp, A, B, shape)
                        rec[f"{b}_off"], rec[f"{b}_on"], rec[f"{b}_err"] = c_off, c_on, err
                        if b != "stream":
                            rec[f"{b}_exact"] = (c_off - L - accounting(b, A, B, shape, False) ==
                                                 c_on - L - accounting(b, A, B, shape, True))
                    rows.append(rec)
                print(f"  {mname} {ln} slice {sl} done", flush=True)


def report(d):
    r = d[d.part == "real"].copy()
    if r.empty:
        return
    errs = int(r[[f"{b}_err" for b in BUILDS]].sum().sum())
    exact = bool(r[["resident_exact", "amortise_exact"]].all().all())
    print(f"\n  real slices: {len(r) // len(SHAPES)}   output errors: {errs}   "
          f"accounting exact in every resident/amortise run: {exact}")
    for b in ("resident", "amortise"):
        r[f"{b}_c_off"] = r[f"{b}_off"] - r.load
        r[f"{b}_c_on"] = r[f"{b}_on"] - r.load
    for (m, s), x in r.groupby(["model", "shape"], sort=False):
        rs = 1 - x.resident_c_on.sum() / x.resident_c_off.sum()
        am = 1 - x.amortise_c_on.sum() / x.amortise_c_off.sum()
        print(f"  {m:18s} {s:9s} compute, skip on vs off:  per-tile fill {rs:6.1%}   "
              f"amortised {am:6.1%}   |  skip-on compute  {x.resident_c_on.sum() / x.amortise_c_on.sum():.2f}x "
              f"faster amortised")
    print()
    for m, x in r.groupby("model", sort=False):
        sq = x[x["shape"] == "16x16"]
        best = {b: x.groupby(["layer", "slice"])[f"{b}_on"].min().sum() for b in BUILDS}
        print(f"  {m:18s} end to end, skip on, best shape per slice:  resident {best['resident']:,}"
              f"   amortise {best['amortise']:,}   stream {best['stream']:,}   (load alone "
              f"{int(sq.load.sum()):,})   resident/stream {best['resident'] / best['stream']:.2f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="parts 1-2 only")
    ap.add_argument("--real-only", action="store_true", help="part 3 only, merged")
    args = ap.parse_args()
    rng = np.random.default_rng(10)
    rows, ok = [], True
    out = os.path.join(tg.RESULTS, "amortise_rtl.csv")
    with tempfile.TemporaryDirectory() as tmp:
        if args.real_only:
            rows = [r for r in pd.read_csv(out).to_dict("records") if r["part"] != "real"]
            ok = bool((pd.DataFrame(rows).errors == 0).all())
        else:
            ok = part1(tmp, rng, rows)
            ok &= part2(tmp, rng, rows)
        if not args.quick:
            part3(tmp, np.random.default_rng(30), rows)
    d = pd.DataFrame(rows)
    if args.quick and os.path.exists(out):
        out = out.replace(".csv", "_quick.csv")
    d.to_csv(out, index=False)
    report(d)
    print("\n  PARTS 1-2: ALL CORRECT AND EXACT" if ok else "\n  PARTS 1-2: MISMATCH PRESENT")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
