# VersaAccel: When Does Reconfigurability Pay?

**A cost-model, RTL and synthesis study of sparsity-aware dataflow selection for DNN
accelerators**, built on the VersaAccel reconfigurable PE array (16×16, 256 PEs).

Every number here is regenerated from code in this repository (see *Reproducing*); the full
record of each one, with its source file and command, is
**[`PROJECT_REPORT.md`](PROJECT_REPORT.md)**. New to the project? Start with
**[`PROJECT_STORY.md`](PROJECT_STORY.md)** — the whole project in plain language, in order.

It asks when switching dataflow per layer is worth building, and finds that the answer is set
less by *how many* zeros a workload has than by *where they are* relative to the array's tile
edges.

The base architecture is from **M. Tang and M. Wen, "VersaAccel: A Versatile Configurable
Accelerator for Diverse Sparse-Dense Matrix Operators," *IEEE Trans. Computer-Aided Design of
Integrated Circuits and Systems*, vol. 45, no. 7, pp. 3059–, July 2026,
[doi:10.1109/TCAD.2025.3639392](https://doi.org/10.1109/TCAD.2025.3639392).** This repository
is an independent study of it and is not affiliated with its authors.

## Key findings

1. **Sparsity structure, not density, decides what zero-skipping saves.** On a shared-operand
   (systolic) array a step can be skipped only if a whole tile edge of operands is zero. Real
   ReLU activations are 14–29× more skippable than random zeros at the same density; 2:4
   pruning is nearly unskippable — a 170× difference at equal density, confirmed in RTL.
2. **A 16×16–32×32 operating window.** Skipping is worth less with every doubling of the
   array, while runtime dataflow selection only matters from 16×16 up.
3. **Two mechanisms exploit the structure:** an exact compile-time channel-reordering rewrite
   (+7.4 / +5.6 pts of skip saving on pruned ResNet-50 / BERT, zero hardware), and a tile
   orientation chosen by where the zeros are (up to 1.40×; a square + tall array pays per area
   only on weight-pruned Transformers).
4. **The published cost model breaks the MAC lower bound** for two of its four dataflows
   (median 31× / 16× on real layers). Corrected, per-layer selection pays 1.05× overall and
   1.11–1.28× on pruned CNNs — and nothing on Transformers.
5. **The flexibility is cheap in silicon.** At the 45 nm slow corner the reconfigurable
   datapath is 1.20× a systolic array's area (its routing networks 0.6% of the design),
   VersaAccel reaches ~430 MHz (systolic ~475, SIGMA ~395), and switching configuration costs
   ≤ 9 cycles, so a greedy scheduler is exactly optimal.

## Headline results (16×16 unless noted)

| Result | Number | Report |
|:--|:--|:--|
| Two published dataflow formulas charge below the MAC lower bound on real layers (D-MR / D-M-R, median) | 31× / 16× — corrected in §3–§6 | §3.0 |
| Per-layer selection vs best static config, corrected model: all / pruned CNNs / homogeneous | 1.052× ± 0.044 / 1.11–1.28× / 1.000× (as published 1.184×) | §3.0, §3.2 |
| Same, costed for a shared-operand array like the RTL (fill + realisable structure) | 1.000× — the choice is static | §3.0 |
| Against the strongest single fixed dataflow (Flexagon's form; Flexagon reports 1.35×) | 1.05× (1.21× as published) | §3.4 |
| Greedy vs optimal schedule at modelled switch cost; look-ahead pays only above | 0.013%; ~265% of layer latency | §4 |
| Configuration switch cost measured on the RTL (forced / CPE-predicted) vs the 100–250-cycle estimate | 0 / ≤ 9 cycles — greedy exactly optimal | §4.3 |
| Activation sparsity can change the optimal config at 8×8 / 16×16 / 32×32 (BW ≥ 4) | 0% / 20.8% / 33.3% | §5.1 |
| Learned 8,961-param predictor vs a zero-parameter constant (10 seeds) | 93.4% ± 5.6 vs 96.2% | §5.2 |
| Real ReLU zeros vs i.i.d. at equal density: all-zero k-slices | 14–29× more | §2.5 |
| Zero-skip saving falls with tile size (ResNet-50, 4×4 → 16×16 → 64×64) | 37% → 19% → 9% | §2.7 |
| Exact channel regrouping (compile time, outputs unchanged), pruned ResNet / BERT | +7.4 / +5.6 pts | §2.8 |
| Tile orientation chosen by structure on the RTL's reshapeable array, pruned BERT | 1.40× (wrong orientation 0.85×) | §2.9 |
| Skip saving on real pruned slices once the fill is paid per GEMM, not per tile (RTL, B10) | ResNet-50 11.2% → 24.3%, BERT 12.2% → 25.2% | §7.13 |
| Activations streamed during compute (RTL, B12): end to end on ResNet slices | 1.33×, within 1.6% of the load alone | §7.13 |
| Two more families (B9): selection speedup, MobileNetV2 (CNN) / OPT-125m (Transformer) | 1.123× / 1.000× — the CNN/Transformer split holds | §2.11 |
| RTL golden-vector checks (all pass, on every build) | 13,215 | §7 |
| Fmax, slow corner (0.9 V, 125 °C): VersaAccel / systolic / SIGMA | ~430 / ~475 / ~395 MHz | §7.15 |
| Cell area at 2.5 ns: VersaAccel / systolic / SIGMA; PE array + networks vs the whole systolic array | 1.867 / 0.945 / 1.551 mm²; 1.20× | §7.15 |
| Compute-phase energy, dense / 90% pruned: VersaAccel / systolic / SIGMA | 4.20 / 1.89 / 12.78 pJ/MAC; 2.54 / 1.32 / 4.96 | §7.15 |
| Area cost of fill amortisation / streaming / reshapeable array (square + tall; three shapes) | +4.5% / +2.6% / +20.6%; +43–57% | §7.15 |
| Reshaping per area (square + tall, equal clock): pruned BERT / pruned ResNet | 1.16× / 0.93× — pays only for weight-pruned Transformers | §7.15 |

## Repository layout

| Path | What it is |
|:--|:--|
| `PROJECT_REPORT.md` | The full technical record: every result, its source file, the command that produces it, and the claims it supports or withdraws |
| `PROJECT_STORY.md` | The project in plain language, from the original paper to now |
| `base_simulator/` | Cost model for the four VersaAccel dataflows (DMR, D-MR, DM-R, D-M-R), memory model, per-layer cost function, Table II validation |
| `idea1_adaptaccel/` | Runtime adaptive scheduling: workload profilers (`profiler/`), sparsity predictor (`predictor/`), greedy / DP / oracle schedulers (`scheduler/`), real traces (`profiling_data/`) |
| `idea2_veriflex/` | N:M sparsity as a 5th configuration, the INT8 configuration prediction engine, and **all RTL** (`rtl/`: VersaAccel, systolic and SIGMA baselines, testbenches, scripts) |
| `idea4_speculative/` | Speculative configuration, with a sensitivity sweep over every unmeasured constant |
| `experiments/` | One script per report section; each writes a CSV to `results/` |
| `results/` | Every CSV the report's numbers come from (`as_published/`: the uncorrected cost model's) |
| `syn/genus/` | Cadence Genus flow for the three designs and the RTL variants, with compute-window power from VCDs |

(The ideas are numbered as in the original project plan; there is no `idea3_*` directory.)

## Reproducing

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
PY=.venv/bin/python ./run_all.sh --quick     # ~50 min, everything but the multi-hour runs
PY=.venv/bin/python ./run_all.sh             # ~4.5 h, every number in the report
```

`run_all.sh` ends with `experiments/audit_report_numbers.py`, which re-derives the headline
numbers from `results/` and fails if any no longer appears verbatim in the report. Exact tool
versions are in `requirements.txt`; the RTL needs **Icarus Verilog 13**, synthesis needs
**Cadence Genus 21.10** and a 45 nm library (`syn/genus/README.md`).

The RTL alone:

```bash
idea2_veriflex/rtl/scripts/run_all_tb.sh                                  # 11 testbenches, 13,215 golden checks
IVFLAGS=-DVERSAACCEL_AMORTISE idea2_veriflex/rtl/scripts/run_all_tb.sh    # same, fill amortised
```

Compile-time variants, each off by default (the default build is the design §7.1–§7.12
measure): `VERSAACCEL_RESHAPE` (square / wide / tall tile, §2.9), `VERSAACCEL_RESHAPE_TALL`
(square / tall only, §7.15), `VERSAACCEL_AMORTISE` (one skew fill per GEMM, §7.13) and
`VERSAACCEL_STREAM` (activations streamed during compute; needs AMORTISE, §7.13). Two
environment switches select model variants: `VERSAACCEL_COST_MODEL=mac|nnz|rtl` (§3.0; `mac`
is the default) and `VERSAACCEL_SWITCH_COST=estimate|rtl` (§4.3).

Synthesis (needs Genus; set `PDK_DIR` to the folder holding `gsclib045_all_v4.8`):

```bash
cd syn/genus && PDK_DIR=/path/to/libs bash run_college.sh
```

## Real vs synthetic data

- **Real (10 of 15 trace sets, plus 8 in `profiling_data/b9/` — MobileNetV2 and OPT-125m,
  §2.11):** ResNet-50 (`IMAGENET1K_V2`) on Imagenette validation images, and
  `bert-base-uncased` on SST-2 validation sentences with eager attention — each dense,
  50/70/90% magnitude-pruned and 2:4-pruned.
- **Synthetic by design (5):** `dp_stress`, `gcn_synthetic`, `mixed_pipeline`,
  `switching_stress`, `variable_random` — stress tests for the scheduler.
- The profilers **refuse** to run on noise or random tokens unless passed `--allow-synthetic`.

**The traces are shipped.** `idea1_adaptaccel/profiling_data/*_traces.csv` (and `b9/`) are the
exact files every number was computed from — ~45 MB of per-layer statistics (shapes, sparsity
fractions, k-slice structure, provenance stamps). They contain no image, sentence or token from
any dataset. `python experiments/verify_traces.py` checks them against `MANIFEST.sha256`.

## Scope, stated up front

The RTL implements one dataflow — output-stationary tiled GEMM with zero-skipping — for every
configuration. The cost model's sparse configurations describe a Gustavson dataflow the RTL
does not build, so the sparse-configuration speedups are **analytical**; the dense path, the
skipping mechanism, tile reshaping, fill amortisation, streamed activations, the switch cost
and the cost model's dense term are **verified against RTL**, and area, Fmax and energy come
from synthesis at the 45 nm slow corner without place-and-route (§7.9, §7.12–§7.15).

## Licence

Code: [MIT](LICENSE). The VersaAccel paper itself is © IEEE and is not included; see the
citation above.
