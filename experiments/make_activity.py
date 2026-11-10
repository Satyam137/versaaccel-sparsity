"""B1c: switching-activity (VCD) files for Genus power, from real operands.

Genus's default toggle rate makes report_power an order-of-magnitude estimate
(section 7.11). This simulates each design on operands sliced from the SAME
real layer -- ResNet-50 layer3.2.conv1, one Imagenette image, dense and 90%
pruned -- with every output checked as usual, and dumps a VCD of the DUT:

  versaaccel_top  tb_regroup, 32x8x64 GEMM, skipping on   (dense and pruned90)
  systolic_top    tb_systolic_golden, 16x16 weights, 64 compute steps
  sigma_top       tb_sigma_golden, one FULL-mode 16-unit x 32-K run

Caveat for the comparison: systolic_top holds one activation per row for the
whole run (its dataflow), so its activity is lower than a streaming GEMM's
on the same values; that is a property of the baseline, not of the stimulus.

Writes syn/genus/activity/*.vcd (git-ignored) and prints the DUT instance path
and time window Genus needs (syn/genus/power_vcd.tcl).

Usage: python experiments/make_activity.py      (~2 min; needs iverilog)
"""
import os, re, subprocess, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tile_geometry as tg                                     # noqa: E402
from regroup_rtl import write_hex, quantise, capture_values, RTL  # noqa: E402

ROOT = os.path.dirname(RTL.rstrip("/").rsplit("/idea2_veriflex", 1)[0] + "/x")
OUT = os.path.join(tg.ROOT, "syn", "genus", "activity")
LAYER = "layer3.2.conv1"
NR = NC = 16


# Register arrays per design: VCD leaves unpacked arrays out unless each element
# is listed, and these are the flops Genus most needs activity for. Paths are
# relative to the DUT instance. (name, dims)
ARRAYS = {
    "versaaccel": [("u_ibuf.q", [256]), ("u_wbuf.q", [512]), ("u_a_nz.q", [16, 256]),
                   ("u_b_nz.q", [256, 32]), ("u_tbuf.q", [16, 16]), ("khist", [16]),
                   ("kvalid", [16]), ("obuf", [256]), ("cpe_features", [9]),
                   # operand feed pipeline and output register (Rev 20)
                   ("f1_a_q", [16]), ("f1_b_q", [16]), ("f2_a_q", [16]), ("f2_b_q", [16]),
                   ("a_fd_q", [16]), ("f3_b_q", [16]), ("dvec_q", [16]), ("mask_q", [256])],
    "systolic":   [("weight_reg", [16, 16]), ("accum", [16, 16]), ("a_fwd", [16, 17]),
                   ("act_buf", [16])],
    "sigma":      [("weight_store", [16, 32]), ("act_store", [32]), ("acc", [16, 16]),
                   ("dist_q", [16]), ("prod_q", [16, 16])],
}


def array_include(design):
    """SystemVerilog statements dumping every element of the design's arrays."""
    path = os.path.join(OUT, f"dump_{design}.svh")
    with open(path, "w") as f:
        for name, dims in ARRAYS[design]:
            if len(dims) == 1:
                f.write(f"for (int i = 0; i < {dims[0]}; i++) $dumpvars(0, dut.{name}[i]);\n")
            else:   # iverilog rejects variable indices into a 2-D element: unroll
                for i in range(dims[0]):
                    f.write(" ".join(f"$dumpvars(0, dut.{name}[{i}][{j}]);"
                                     for j in range(dims[1])) + "\n")
    return path


def unflatten(vcd, design):
    """iverilog names element [i][j] of a 2-D array by its flat index; restore
    [i][j] so the names match the RTL (and Genus's register names)."""
    two_d = {name.split(".")[-1]: (name.rsplit(".", 1)[0] if "." in name else "", d[1])
             for name, d in ARRAYS[design] if len(d) == 2}
    out, scope = [], []
    pat = re.compile(r"^(\$var \S+ \d+ \S+ )\\?([A-Za-z_][A-Za-z0-9_]*)\[(\d+)\](.*)$")
    for line in open(vcd):
        if line.startswith("$scope"):
            scope.append(line.split()[2])
        elif line.startswith("$upscope"):
            scope.pop()
        else:
            m = pat.match(line)
            if m and m.group(2) in two_d:
                sub, ncol = two_d[m.group(2)]
                here = ".".join(scope[2:])            # path below tb.dut
                if here == sub:
                    k = int(m.group(3))
                    line = f"{m.group(1)}\\{m.group(2)}[{k // ncol}][{k % ncol}]{m.group(4)}\n"
        out.append(line)
    open(vcd, "w").writelines(out)


def sv(*files):
    return [os.path.join(RTL, f) for f in files]


def build_run(name, srcs, defines=(), plusargs=()):
    exe = os.path.join(OUT, f"{name}.vvp")
    subprocess.run(["iverilog", "-g2012", "-DSYNTHESIS", "-I", os.path.join(RTL, "versaaccel_rtl"), "-o", exe,
                    f'-DCPE_WEIGHT_DIR="{RTL}/weights"', *defines,
                    os.path.join(RTL, "common", "versaaccel_pkg.sv"), *srcs],
                   check=True, capture_output=True)
    out = subprocess.run(["vvp", "-n", exe, *plusargs], capture_output=True, text=True,
                         check=True).stdout
    os.remove(exe)
    ok = "[FAIL]" not in out and ("[PASS]" in out or "RESULT" in out)
    if "RESULT" in out:
        ok = ok and [l for l in out.splitlines() if l.startswith("RESULT")][0].split()[-1] == "0"
    return ok


def main():
    os.makedirs(OUT, exist_ok=True)
    img = tg.resnet_inputs(1)[0]
    models = dict(tg.resnet_models())
    full = sorted(os.path.join(RTL, "versaaccel_rtl", f)
                  for f in os.listdir(os.path.join(RTL, "versaaccel_rtl")) if f.endswith(".sv"))
    rng = np.random.default_rng(0)
    print(f"  operands: ResNet-50 {LAYER}, one Imagenette image")
    for tag in ("resnet50_dense", "resnet50_pruned90"):
        Af, Bf = capture_values(models[tag], {LAYER}, img, lambda m, x: m(x))[LAYER]
        r0, k0, p0 = (rng.integers(0, Af.shape[0] - 32), rng.integers(0, Af.shape[1] - 32),
                      rng.integers(0, Bf.shape[1] - 64))
        A = quantise(Af[r0:r0 + 32, k0:k0 + 32]); B = quantise(Bf[k0:k0 + 32, p0:p0 + 64])
        short = tag.replace("resnet50_", "")
        # --- VersaAccel: 32x8x64 GEMM (fills IBUF 256 / WBUF 512), skipping on
        fa, fb = os.path.join(OUT, "a.hex"), os.path.join(OUT, "b.hex")
        write_hex(fa, A[:, :8]); write_hex(fb, B[:8, :])
        v = os.path.join(OUT, f"versaaccel_{short}.vcd")
        ok = build_run("v", full + sv("testbenches/tb_regroup.sv"),
                       ["-DRG_M=32", "-DRG_K=8", "-DRG_P=64",
                        f'-DDUMP_ARRAYS="{array_include("versaaccel")}"'],
                       [f"+A={fa}", f"+B={fb}", f"+VCD={v}"])
        unflatten(v, "versaaccel")
        print(f"  versaaccel_top {short:9s} {'ok' if ok else 'FAIL'}  {os.path.getsize(v)/1e6:6.1f} MB"
              f"  A zeros {np.mean(A[:, :8] == 0):.0%}, B zeros {np.mean(B[:8] == 0):.0%}")
        # --- systolic: 16x16 real weights, 16 real activations, 64 steps
        h = os.path.join(OUT, "sys.hex")
        write_hex(h, np.concatenate([B[:16, :16].T.ravel(), A[0, :16]]))
        s = os.path.join(OUT, f"systolic_{short}.vcd")
        ok = build_run("s", sv("systolic_baseline/systolic_top.sv", "testbenches/tb_systolic_golden.sv"),
                       [f'-DPOWER_HEX="{h}"', f'-DDUMP_VCD="{s}"', "-DPOWER_STEPS=64",
                        f'-DDUMP_ARRAYS="{array_include("systolic")}"'])
        unflatten(s, "systolic")
        print(f"  systolic_top   {short:9s} {'ok' if ok else 'FAIL'}  {os.path.getsize(s)/1e6:6.1f} MB")
        # --- SIGMA: 16 units x K=32 real weights, one real activation vector
        h2 = os.path.join(OUT, "sig.hex")
        write_hex(h2, np.concatenate([B[:32, :16].T.ravel(), A[0, :32]]))
        g = os.path.join(OUT, f"sigma_{short}.vcd")
        ok = build_run("g", sv("versaaccel_rtl/benes_switch.sv", "versaaccel_rtl/distribution_network.sv",
                               "versaaccel_rtl/reduction_network.sv", "sigma_baseline/sigma_top.sv",
                               "testbenches/tb_sigma_golden.sv"),
                       [f'-DPOWER_HEX="{h2}"', f'-DDUMP_VCD="{g}"',
                        f'-DDUMP_ARRAYS="{array_include("sigma")}"'])
        unflatten(g, "sigma")
        print(f"  sigma_top      {short:9s} {'ok' if ok else 'FAIL'}  {os.path.getsize(g)/1e6:6.1f} MB")
    for f in ("a.hex", "b.hex", "sys.hex", "sig.hex",
              "dump_versaaccel.svh", "dump_systolic.svh", "dump_sigma.svh"):
        os.remove(os.path.join(OUT, f))
    print(f"\n  VCDs in {OUT}\n  DUT instances: tb_regroup/dut, tb_systolic_golden/dut, tb_sigma_golden/dut")


if __name__ == "__main__":
    main()
