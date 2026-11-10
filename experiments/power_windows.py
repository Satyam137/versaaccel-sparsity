"""B1c, second pass: activity for the compute phase only, so power compares.

Genus run 3 read the VCDs from make_activity.py whole. Each covers a different
test -- VersaAccel a 32x8x64 GEMM, the systolic array 64 weight-stationary steps,
SIGMA one 16-unit K=32 reduction -- and most of each is operand LOADING, during
which the arrays idle. Average power over such a window measures the testbench,
not the design: run 3 reported 11-18 mW for all three against 150-440 mW
vectorless.

This trims each VCD to its compute window -- from the cycle the operand load
ends (din_valid's last fall) to the last output beat (dout_valid's last fall) --
re-dumping every signal's value at the window start so the trimmed file is
self-contained, and records the window length and the MACs the window performs.
Genus then gives average power over the compute phase only, and

    energy per MAC = average power x window length / MACs

compares across designs whatever their clock or test length (the simulation
clock is 100 MHz; energy per operation does not depend on it, power does).

Writes syn/genus/activity/<design>_<sparsity>_compute.vcd and
syn/genus/activity/power_windows.csv.
Usage: python experiments/power_windows.py     (after make_activity.py; ~10 s)
"""
import csv, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACT = os.path.join(ROOT, "syn", "genus", "activity")

# MACs in each design's test (make_activity.py): the GEMM / steps it simulates.
MACS = {"versaaccel": 32 * 8 * 64,     # 32x8x64 GEMM through the 16x16 array
        "systolic":   16 * 16 * 64,    # 256 PEs x 64 compute steps
        "sigma":      16 * 32}         # 16 dot-product units x K = 32


def dut_ids(path, names):
    """VCD id of each named port in the DUT's own scope (the second scope level)."""
    ids, depth, in_dut = {}, 0, False
    with open(path) as f:
        for line in f:
            if line.startswith("$enddefinitions"):
                break
            if line.startswith("$scope"):
                depth += 1
                in_dut = depth == 2
            elif line.startswith("$upscope"):
                depth -= 1
                in_dut = depth == 2
            elif in_dut and line.startswith("$var"):
                p = line.split()
                if p[4] in names and p[4] not in ids:
                    ids[p[4]] = p[3]
    return ids


def window(path):
    """(start, end) in VCD time units: din_valid's last fall, dout_valid's last fall."""
    ids = dut_ids(path, {"din_valid", "dout_valid"})
    di, do = ids["din_valid"], ids["dout_valid"]
    t, last = 0, {di: None, do: None}
    val = {di: "x", do: "x"}
    with open(path) as f:
        for line in f:
            if line[0] == "#":
                t = int(line[1:])
            elif len(line) > 1 and line[0] in "01xz" and line[1:].strip() in last:
                i = line[1:].strip()
                if val[i] == "1" and line[0] == "0":
                    last[i] = t
                val[i] = line[0]
    return last[di], last[do]


def trim(src, dst, t0, t1):
    """Copy src to dst keeping only [t0, t1], with a full value dump at t0."""
    vals, header = {}, []
    with open(src) as f:
        for line in f:
            header.append(line)
            if line.startswith("$enddefinitions"):
                break
        t, started = 0, False
        with open(dst, "w") as out:
            out.writelines(header)
            for line in f:
                if line[0] == "#":
                    t = int(line[1:])
                    if t > t1:
                        break
                    if not started and t >= t0:
                        out.write(f"#{t0}\n$dumpvars\n")
                        out.writelines(v for v in vals.values())
                        out.write("$end\n")
                        started = True
                    if started and t > t0:
                        out.write(line)
                    continue
                s = line.strip()
                if not s or s.startswith("$"):
                    continue
                key = s.split()[-1] if s[0] in "brBR" else s[1:]
                if started:
                    out.write(line)
                else:
                    vals[key] = line if line.endswith("\n") else line + "\n"
            out.write(f"#{t1}\n")


def main():
    rows = []
    for design in ("versaaccel", "systolic", "sigma"):
        for sp in ("dense", "pruned90"):
            src = os.path.join(ACT, f"{design}_{sp}.vcd")
            t0, t1 = window(src)
            dst = os.path.join(ACT, f"{design}_{sp}_compute.vcd")
            trim(src, dst, t0, t1)
            ns = (t1 - t0) / 1000.0                       # VCD timescale is 1 ps
            rows.append(dict(design=design, sparsity=sp, start_ps=t0, end_ps=t1,
                             window_ns=ns, cycles=round(ns / 10), macs=MACS[design]))
            print(f"  {design:10s} {sp:8s}  window {t0/1000:9.1f} .. {t1/1000:9.1f} ns"
                  f"  = {ns:7.1f} ns ({ns/10:5.0f} cycles at 100 MHz)  MACs {MACS[design]:6d}"
                  f"  -> {os.path.basename(dst)} {os.path.getsize(dst)/1e6:.1f} MB")
    with open(os.path.join(ACT, "power_windows.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  wrote {ACT}/power_windows.csv")


if __name__ == "__main__":
    main()
