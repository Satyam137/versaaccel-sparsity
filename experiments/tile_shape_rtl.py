"""Check tile_shape.cycles_rect() against the RTL at non-square array shapes.

versaaccel_top takes NR and NC from versaaccel_pkg independently. This builds
the RTL at 64x4, 4x64, 8x32, 32x8 and 16x16 (a package copy with NR/NC edited
in a temp dir), runs random sparse GEMMs through tb_regroup.sv with skipping
off and on, and reports RTL cycles minus the accounting. Exact accounting =
one fixed offset per shape (the last tile's exposed drain, NC - 1, plus the
Rev 20 operand-feed pipeline and output register, 4) in both
modes, and zero output errors.

Usage: python experiments/tile_shape_rtl.py      (~1 min; needs iverilog)
"""
import os, re, subprocess, sys, tempfile
import numpy as np

FEED_PIPE = 4   # Rev 20: three operand-feed stages + the output register, once per GEMM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tile_shape as ts                    # noqa: E402
from regroup_rtl import write_hex, RTL    # noqa: E402

# (NR, NC) -> (M, K, P), each filling IBUF (M*K = 256) and WBUF (K*P = 512)
CASES = {(64, 4): (64, 4, 128), (4, 64): (32, 8, 64), (8, 32): (16, 16, 32),
         (32, 8): (32, 8, 64), (16, 16): (16, 16, 32)}


def main():
    rng = np.random.default_rng(5)
    pkg_src = open(os.path.join(RTL, "common", "versaaccel_pkg.sv")).read()
    srcs = sorted(os.path.join(RTL, "versaaccel_rtl", f)
                  for f in os.listdir(os.path.join(RTL, "versaaccel_rtl")) if f.endswith(".sv"))
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        for (nr, nc), (M, K, P) in CASES.items():
            pkg = os.path.join(tmp, f"pkg_{nr}x{nc}.sv")
            txt = re.sub(r"parameter int NR(\s+)= \d+;", rf"parameter int NR\g<1>= {nr};", pkg_src)
            txt = re.sub(r"parameter int NC(\s+)= \d+;", rf"parameter int NC\g<1>= {nc};", txt)
            open(pkg, "w").write(txt)
            exe = os.path.join(tmp, f"tb_{nr}x{nc}")
            subprocess.run(["iverilog", "-g2012", "-o", exe, f'-DCPE_WEIGHT_DIR="{RTL}/weights"',
                            f"-DRG_M={M}", f"-DRG_K={K}", f"-DRG_P={P}", pkg, *srcs,
                            os.path.join(RTL, "testbenches", "tb_regroup.sv")],
                           check=True, capture_output=True)
            seen = set()
            for _ in range(6):
                da, db = rng.choice([0.0, 0.5, 0.8, 0.95], size=2)
                A = rng.integers(-9, 10, (M, K)); A[rng.random(A.shape) < da] = 0
                B = rng.integers(-9, 10, (K, P)); B[rng.random(B.shape) < db] = 0
                fa, fb = os.path.join(tmp, "a.hex"), os.path.join(tmp, "b.hex")
                write_hex(fa, A); write_hex(fb, B)
                out = subprocess.run(["vvp", "-n", exe, f"+A={fa}", f"+B={fb}"],
                                     capture_output=True, text=True, check=True).stdout
                c_off, c_on, _, err = map(int, [l for l in out.splitlines()
                                                if l.startswith("RESULT")][0].split()[1:])
                d, s = ts.cycles_rect(A != 0, B.T != 0, nr, nc)
                seen.add((c_off - int(d), c_on - int(s), err))
            exact = len(seen) == 1 and next(iter(seen)) == (nc - 1 + FEED_PIPE, nc - 1 + FEED_PIPE, 0)
            ok &= exact
            print(f"  {nr:2d}x{nc:<2d}  A{M}x{K} B{K}x{P}  (RTL-acct off, on, errors): "
                  f"{sorted(seen)}  {'exact' if exact else 'MISMATCH'}")
    print("  ALL SHAPES EXACT" if ok else "  MISMATCH PRESENT")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
