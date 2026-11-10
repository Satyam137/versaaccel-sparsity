"""A2: re-derive the report's key numbers from results/ and check the report.

Every number in PROJECT_REPORT.md's claims and headline tables should trace to
a file in results/ (or a fast deterministic rerun) and to the script that
wrote it. This recomputes each one, formats it the way the report prints it,
and checks that exact text is in the report. A stale figure -- the failure the
audit has found five times by hand -- fails here instead.

Usage: python experiments/audit_report_numbers.py      (~15 s)
Exit status 1 if any check fails.
"""
import os, re, subprocess, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda f: pd.read_csv(os.path.join(ROOT, "results", f))
REPORT = open(os.path.join(ROOT, "PROJECT_REPORT.md"), encoding="utf-8").read()


def checks():
    sv = R("scheduler_variance.csv")
    het = sv[~sv.model.str.startswith("bert")]
    yield "3.2", "mean speedup, 15 workloads", f"{sv.speedup_mean.mean():.3f}x", "scheduler_variance.csv"
    yield "3.2", "heterogeneous mean speedup", f"{het.speedup_mean.mean():.3f}x", "scheduler_variance.csv"
    yield "3.3", "workloads with >1 optimal schedule", f"{int((sv.unique_schedules > 1).sum())}/15", "scheduler_variance.csv"

    sw = R("switch_cost_sweep_summary.csv").set_index("scale")
    yield "4.2", "greedy gap at modelled switch cost", f"{sw.loc[1.0, 'greedy_gap']:.3f}%", "switch_cost_sweep_summary.csv"
    # first switch cost at which W=1 is more than 1% from optimal (section 4.2's design rule)
    thr = sw[sw.w1_gap > 1.0].switch_frac.iloc[0]
    yield "4.2", "lookahead threshold (switch / layer)", f"~{thr * 100:.0f}%", "switch_cost_sweep_summary.csv"

    out = subprocess.run([sys.executable, os.path.join(ROOT, "idea1_adaptaccel", "scheduler", "dp_scheduler.py")],
                         capture_output=True, text=True, cwd=ROOT).stdout
    w1 = re.search(r"Average gap W=1: ([+-][0-9.]+%)", out).group(1)
    yield "4.1", "window ablation, W=1 gap", w1, "dp_scheduler.py (rerun)"

    a = R("array_size_sweep.csv")
    for arr in (16, 32):
        f = a[(a.array == arr) & (a.bw_elem_per_cycle == 4)].sensitive_frac.iloc[0]
        yield "5.1", f"{arr}x{arr} decision-sensitive at BW=4", f"{f * 100:.1f}%", "array_size_sweep.csv"

    pb = R("predictor_baselines.csv").set_index("baseline")
    yield "5.2", "constant-mean match (relevant)", f"{pb.loc['constant-mean (0.401)', 'match_relevant'] * 100:.1f}%", "predictor_baselines.csv"
    yield "5.2", "persistence match (relevant)", f"{pb.loc['persistence (copy current layer)', 'match_relevant'] * 100:.1f}%", "predictor_baselines.csv"
    yield "5.2", "constant-zero match (relevant)", f"{pb.loc['constant-zero', 'match_relevant'] * 100:.1f}%", "predictor_baselines.csv"
    yield "5.2", "MLP test MAE", f"{pb.loc['trained MLP', 'mae']:.3f}", "predictor_baselines.csv"
    ps = R("predictor_seed_robustness.csv")
    m = ps[ps.predictor == "MLP"].match_relevant * 100
    yield "5.2", "MLP over 10 seeds", f"{m.mean():.1f}% ± {m.std():.1f}", "predictor_seed_robustness.csv"

    st = R("structure_predictor.csv")
    for tgt in ("structure (B5)", "density  (5.2)"):
        g = st[st.target == tgt].gain_vs_perlayer
        yield "5.3", f"MLP vs per-layer table, {tgt.split()[0]}", f"+{g.mean():.1f}% ± {g.std():.1f}", "structure_predictor.csv"

    sv2 = R("structure_value.csv")
    yield "2.6", "mean realisable saving, 10 real sets", f"{sv2.realisable_saving.mean() * 100:.1f}%", "structure_value.csv"

    dec = R("costmodel_vs_rtl_decisions.csv")
    yield "7.12", "decisions flipped by the fill term", f"{int(dec.flipped.sum()):,} of {len(dec):,}", "costmodel_vs_rtl_decisions.csv"

    ss = R("speculation_sensitivity.csv")
    yield "8.1", "speculation vs Execution-Oracle, minimum", f"+{ss.vs_exec_oracle.min():.3f}%", "speculation_sensitivity.csv"

    cq = R("cpe_quantisation.csv").iloc[0]
    yield "7.2", "CPE INT8 accuracy", f"{cq.int8_test_acc:.4f}", "cpe_quantisation.csv"
    yield "7.2", "CPE majority baseline", f"{cq.majority_baseline:.4f}", "cpe_quantisation.csv"

    for sd in (0, 1):
        g = R(f"tile_geometry_regroup_n200_s{sd}_by_input.csv")
        g["sav"] = 1 - g.skip_cycles / g.dense_cycles
        p = g[g.model == "resnet50_pruned90"].pivot_table(index="input", columns="variant", values="sav")
        d = (p.wshared - p.original) * 100
        yield "2.8", f"regroup gain per image, sample {sd}", f"+{d.mean():.2f} ± {d.std():.2f}", f"tile_geometry_regroup_n200_s{sd}_by_input.csv"

    ts = R("tile_shape.csv")
    ts = ts[ts.kind == "virtual"]
    for mname in ("bert_pruned90", "resnet50_pruned90"):
        x = ts[(ts.model == mname) & (ts.variant == "original")]
        ref = x[(x.NR == 16) & (x.NC == 16)].skip.sum()
        val = ref / x.loc[x.groupby("layer").cal_skip.idxmin()].skip.sum()
        yield "2.9", f"tile shape by structure, {mname}", f"{val:.2f}×", "tile_shape.csv"
    fs = R("tile_shape_fission_summary.csv").set_index(["model", "variant"])
    for mname in ("bert_pruned90", "resnet50_pruned90"):
        yield "2.9", f"RTL shapes per layer, {mname}", f"{fs.loc[(mname, 'original'), 'per_layer_structure']:.2f}×", "tile_shape_fission_summary.csv"

    am = R("amortise_rtl.csv")
    am = am[(am.part == "real") & (am["shape"] == "16x16")]
    for mname in ("resnet50_pruned90", "bert_pruned90"):
        x = am[am.model == mname]
        for b in ("resident", "amortise"):
            sav = 1 - (x[f"{b}_on"] - x.load).sum() / (x[f"{b}_off"] - x.load).sum()
            yield "7.13", f"skip saving, {b}, {mname}", f"{sav * 100:.1f}%", "amortise_rtl.csv"
    r = R("amortise_rtl.csv")
    r = r[(r.part == "real") & (r.model == "resnet50_pruned90")]
    best = {b: r.groupby(["layer", "slice"])[f"{b}_on"].min().sum() for b in ("resident", "stream")}
    yield "7.13", "resident / stream end to end, ResNet", f"{best['resident'] / best['stream']:.2f}×", "amortise_rtl.csv"

    gb = R("generalisation_b9.csv")
    for fam in ("mobilenet_v2", "opt125m"):
        x = gb[gb.workload.str.startswith(fam + "_")]
        yield "2.11", f"selection speedup, {fam} mean", f"{x.speedup_mean.mean():.3f}×", "generalisation_b9.csv"
    x = gb[gb.workload == "opt125m_dense"].iloc[0]
    yield "2.11", "realisable saving, opt125m_dense", f"{x.realisable_saving * 100:.1f}%", "generalisation_b9.csv"

    cv = R("cost_model_variants.csv")
    for tag in ("A", "B", "D"):
        yield "3.0", f"headline, model {tag}", f"{cv[cv.model == tag].speedup_mean.mean():.3f}×", "cost_model_variants.csv"
    ap = R("as_published/scheduler_variance.csv")
    yield "3.0", "headline as published (as_published/)", f"{ap.speedup_mean.mean():.3f}x", "as_published/scheduler_variance.csv"

    pa = R("prior_art_comparison.csv")
    w = pa.groupby("model").mean(numeric_only=True)
    yield "3.4", "selection over fixed D-M-R, all 15", f"{(w['D-M-R'] / w.oracle).mean():.2f}×", "prior_art_comparison.csv"


def main():
    rows, bad = [], 0
    for sec, what, text, src in checks():
        ok = text in REPORT
        bad += not ok
        rows.append((sec, what, text, src, ok))
    w = max(len(r[1]) for r in rows)
    print(f"  {'§':5s} {'number':{w}s} {'derived':>14s}  found  source")
    for sec, what, text, src, ok in rows:
        print(f"  {sec:5s} {what:{w}s} {text:>14s}  {'yes ' if ok else 'NO  '}  {src}")
    print(f"\n  {len(rows) - bad} of {len(rows)} derived numbers appear verbatim in PROJECT_REPORT.md")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
