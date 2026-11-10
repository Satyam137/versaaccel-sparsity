# VersaAccel Sparsity Study

**When does a reconfigurable DNN accelerator actually pay off?**

This project studies sparsity-aware dataflow selection on the VersaAccel reconfigurable
16×16 PE array (256 PEs). It combines an analytical cost model, real workload traces from
ResNet-50 and BERT, cycle-accurate SystemVerilog RTL, and 45 nm synthesis to answer one
question: *when is switching dataflow per layer worth building into hardware?*

The short answer is that it depends less on **how many** zeros a workload has than on
**where those zeros sit** relative to the array's tile edges.

> The base architecture comes from M. Tang and M. Wen, "VersaAccel: A Versatile Configurable
> Accelerator for Diverse Sparse-Dense Matrix Operators," *IEEE TCAD*, vol. 45, no. 7,
> July 2026, [doi:10.1109/TCAD.2025.3639392](https://doi.org/10.1109/TCAD.2025.3639392).
> This repository is an independent study and is not affiliated with the paper's authors.

---

## Key findings

1. **Sparsity structure matters more than density.** On a shared-operand (systolic) array, a
   compute step can be skipped only if a whole tile edge of operands is zero. Real ReLU
   activations are 14–29× more skippable than random zeros at the same density, while 2:4
   pruning is almost unskippable.
2. **There is a sweet spot of 16×16 to 32×32.** Zero-skipping saves less with every doubling
   of the array size, and runtime dataflow selection only starts to matter from 16×16 up.
3. **Two cheap tricks exploit the structure:**
   - an exact compile-time channel-reordering pass (+7.4 / +5.6 points of skip saving on
     pruned ResNet-50 / BERT, with no hardware change), and
   - picking the tile orientation from where the zeros are (up to 1.40× faster).
4. **The published cost model undercounts work.** Two of its four dataflow formulas fall
   below the MAC lower bound on real layers. After correcting them, per-layer selection gives
   1.05× overall, 1.11–1.28× on pruned CNNs, and nothing on Transformers.
5. **The flexibility is cheap in silicon.** At the 45 nm slow corner the reconfigurable
   datapath is 1.20× the area of a plain systolic array, it runs at ~430 MHz, and switching
   configuration costs at most 9 cycles, so a simple greedy scheduler is already optimal.

## Headline numbers

All at 16×16 unless noted.

| Result | Value |
|:--|:--|
| Per-layer selection vs best static config (corrected model): all / pruned CNNs / Transformers | 1.05× / 1.11–1.28× / 1.00× |
| Greedy vs optimal schedule | within 0.013% |
| Configuration switch cost measured on RTL | 0–9 cycles |
| Learned sparsity predictor (8,961 params) vs a constant baseline, 10 seeds | 93.4% vs 96.2% |
| Zero-skip saving vs tile size, ResNet-50 (4×4 → 16×16 → 64×64) | 37% → 19% → 9% |
| Extra families: MobileNetV2 (CNN) / OPT-125m (Transformer) | 1.12× / 1.00× |
| RTL golden-vector checks | 13,215, all passing |
| Fmax, slow corner: VersaAccel / systolic / SIGMA | ~430 / ~475 / ~395 MHz |
| Cell area at 2.5 ns: VersaAccel / systolic / SIGMA | 1.867 / 0.945 / 1.551 mm² |
| Compute energy, dense: VersaAccel / systolic / SIGMA | 4.20 / 1.89 / 12.78 pJ/MAC |

---

## Repository layout

```
versaaccel-sparsity/
├── base_simulator/        Cost model for the four VersaAccel dataflows (DMR, D-MR, DM-R, D-M-R),
│                          memory model, per-layer cost, validation against the paper
├── idea1_adaptaccel/      Runtime adaptive scheduling
│   ├── profiler/          Workload profilers for ResNet-50, BERT, MobileNetV2, OPT-125m
│   ├── predictor/         Sparsity predictor and its trained weights
│   ├── scheduler/         Greedy / dynamic-programming / oracle schedulers
│   ├── prefetch_pipeline/ Configuration prefetch model
│   └── profiling_data/    Per-layer trace CSVs (real and synthetic) + SHA-256 manifest
├── idea2_veriflex/        N:M sparsity as a 5th configuration, config-prediction engine
│   └── rtl/               All SystemVerilog RTL
│       ├── versaaccel_rtl/      VersaAccel top, PE array, Benes switch, distribution/
│       │                        reduction networks, phase FSM, prediction engine
│       ├── systolic_baseline/   Plain systolic array for comparison
│       ├── sigma_baseline/      SIGMA-style baseline for comparison
│       ├── testbenches/         Golden-vector testbenches
│       ├── scripts/             Testbench runners
│       └── weights/             ROM contents for the prediction engine
├── idea4_speculative/     Speculative configuration with a sensitivity sweep
├── experiments/           One script per experiment; each writes a CSV to results/
├── results/               Output CSVs (as_published/ holds the uncorrected-model numbers)
├── syn/genus/             Cadence Genus synthesis and power flow
├── run_all.sh             Regenerates every result from scratch
└── requirements.txt       Pinned Python dependencies
```

There is no `idea3_*` folder. The ideas keep the numbering from the original project plan.

---

## Getting started

### Requirements

- **Python 3.11** with the packages in `requirements.txt`
- **Icarus Verilog 13** for the RTL simulations
- **Cadence Genus 21.10** and a 45 nm library, for synthesis only (optional)

### Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Reproduce the results

```bash
PY=.venv/bin/python ./run_all.sh --quick   # ~50 min, skips the multi-hour runs
PY=.venv/bin/python ./run_all.sh           # ~4.5 h, everything
PY=.venv/bin/python ./run_all.sh --profile # also re-profiles the models first (+~2 h, needs network)
```

Each step writes its CSV into `results/`.

### Run individual pieces

```bash
python -m base_simulator.validate                       # cost model vs the paper's Table II
python idea1_adaptaccel/scheduler/dp_scheduler.py       # scheduler comparison
python experiments/array_size_sweep.py                  # array-size sweep
python experiments/verify_traces.py                     # check traces against MANIFEST.sha256
```

### RTL simulation

```bash
bash idea2_veriflex/rtl/scripts/run_all_tb.sh            # all testbenches, 13,215 golden checks
bash idea2_veriflex/rtl/scripts/run_gemm_shapes.sh       # GEMM shape sweep
bash idea2_veriflex/rtl/scripts/run_sparse_sweep.sh      # zero-skip sweep
```

Optional compile-time RTL variants, all off by default and passed through `IVFLAGS`:

| Define | Effect |
|:--|:--|
| `VERSAACCEL_RESHAPE` | Reshapeable tile: square / wide / tall |
| `VERSAACCEL_RESHAPE_TALL` | Reshapeable tile: square / tall only |
| `VERSAACCEL_AMORTISE` | One skew fill per GEMM instead of per tile |
| `VERSAACCEL_STREAM` | Stream activations during compute (needs `AMORTISE`) |

```bash
IVFLAGS=-DVERSAACCEL_AMORTISE bash idea2_veriflex/rtl/scripts/run_all_tb.sh
```

Model variants are chosen with environment variables:

- `VERSAACCEL_COST_MODEL=mac|nnz|rtl` (default `mac`)
- `VERSAACCEL_SWITCH_COST=estimate|rtl`

### Synthesis

Set `PDK_DIR` to the folder that holds `gsclib045_all_v4.8`, then:

```bash
cd syn/genus
PDK_DIR=/path/to/libs bash run_college.sh
```

See [syn/genus/README.md](syn/genus/README.md) for details.

---

## Data

**Real traces:**
- ResNet-50 (`IMAGENET1K_V2`) on Imagenette validation images
- `bert-base-uncased` on SST-2 validation sentences
- MobileNetV2 and OPT-125m, in `profiling_data/b9/`

Each model is profiled dense, 50/70/90% magnitude-pruned, and (for ResNet-50 and BERT) 2:4-pruned.

**Synthetic traces:** `dp_stress`, `gcn_synthetic`, `mixed_pipeline`, `switching_stress` and
`variable_random`. These are stress tests for the scheduler.

The trace CSVs (~45 MB) are included. They hold only per-layer statistics (shapes, sparsity
fractions, k-slice structure), not any images, sentences or tokens from the datasets.
The profilers refuse to run on random inputs unless passed `--allow-synthetic`.

---

## Scope and limitations

- The RTL implements one dataflow, output-stationary tiled GEMM with zero-skipping, for every
  configuration. The sparse (Gustavson-style) configurations exist only in the cost model, so
  their speedups are **analytical**.
- The dense path, zero-skipping, tile reshaping, fill amortisation, streamed activations,
  switch cost and the cost model's dense term are **verified against RTL**.
- Area, Fmax and energy come from Genus synthesis at the 45 nm slow corner (0.9 V, 125 °C)
  **without place-and-route**.

## License

The code is released under the [MIT License](LICENSE). The VersaAccel paper itself is © IEEE
and is not included in this repository.
