# Genus synthesis

Three designs, one set of constraints. This closes items 1 and 4–6 of
`PROJECT_REPORT.md` §11 — area/power/timing, and the measured reconfiguration
latency that §4's `SWITCH_COST` currently only estimates.

## What you must supply

Only `setup.tcl`. Point `LIB_PATH` at a 45nm liberty library — Nangate
OpenCell 45nm is the usual one and ships with most Cadence installs:

```bash
find / -name "NangateOpenCellLibrary*typical*.lib" 2>/dev/null
```

Nothing else in the flow hardcodes a site-specific path.

## Running

```bash
source /opt/cadence/GENUS/setup.sh     # or your site's equivalent
cd syn/genus
./run.sh                               # all three designs, one clock period
./run.sh --sweep                       # area-vs-frequency curve, finds Fmax
./run.sh --design versaaccel_top       # just one
```

Results land in `results/<design>_<period>ns/`.

## Why the constraints file is shared

VersaAccel is being compared against the systolic and SIGMA baselines. A
different clock, I/O delay or output load between runs would appear as an
architectural difference when it is really a constraint artifact. All three read
the same `constraints.sdc`. **If you change it, re-run all three.**

## The one thing that will silently give you wrong numbers

`config_prediction_engine.sv` loads its INT8 weights with `$readmemh` in an
`initial` block. Synthesis cannot execute that. Genus drops the block, nothing
ever writes the weight arrays, constant propagation deletes the entire CPE, and
its area reports as approximately zero.

`syn.tcl` therefore defines `SYNTHESIS`, which switches the module to a
generated constant ROM (`versaaccel_rtl/cpe_weights_rom.svh`). The two views
hold identical values, and that is checked, not assumed:

```bash
cd idea2_veriflex/rtl
iverilog -g2012 -DSYNTHESIS -I versaaccel_rtl -o /tmp/tb_cpe_syn \
    -DCPE_WEIGHT_DIR="\"$PWD/weights\"" \
    common/versaaccel_pkg.sv versaaccel_rtl/*.sv testbenches/tb_cpe_golden.sv
/tmp/tb_cpe_syn        # 64/64 vectors, same as the $readmemh view
```

Regenerate the ROM after retraining the CPE:

```bash
python3 idea2_veriflex/rtl/scripts/gen_cpe_rom.py
```

## What to report

| Report | File | Feeds |
|:--|:--|:--|
| Total area | `area.rpt` | area comparison vs both baselines |
| Area by hierarchy | `area_hier.rpt` | §7.10 — the double buffer and bitmaps |
| Added structures | `added_structures.rpt` | `tbuf`, `a_nz`, `b_nz` cell counts |
| Timing | `timing.rpt`, `qor.rpt` | Fmax; critical path |
| Power | `power.rpt` | dynamic + leakage |

Then: measure the reconfiguration latency from the critical path through
`speculation_unit` / `config_prediction_engine`, put it into `SWITCH_COST` in
`experiments/switch_cost_sweep.py`, and re-run §4. The design rule there —
lookahead pays only above ~126% of layer latency — is currently anchored to an
*estimated* 250 cycles.

## Honesty note

**This flow has never been executed.** There is no Genus licence on the machine
it was written on, so it is untested against a real tool. It follows the
standard Genus sequence and the RTL it reads is verified (11,716 golden checks),
but budget time for one or two library-specific fixes on first run. The lines
most likely to need attention are marked `CHECK` in `syn.tcl`; the usual
suspects are `set_driving_cell -lib_cell INV_X4` (cell name differs between
libraries) and the `get_db insts` name patterns for the per-structure reports.

## Design variants (B15, B10, B12) — for the next run

`versaaccel_top` has three compile-time variants, all off by default. Each one's hardware
cost is its synthesis result minus the default build's, with everything else identical. Add
the define to **every** `read_hdl` of the VersaAccel sources in `syn.tcl` (package included),
e.g. `read_hdl -language sv -define VERSAACCEL_AMORTISE ...`:

| Define | What it adds | Report section |
|:--|:--|:--|
| `VERSAACCEL_RESHAPE` | midline operand injection muxes, half-edge bitmaps | §2.9 |
| `VERSAACCEL_AMORTISE` | second output bank (`NR·NC·ACC_W` flops), per-PE clear OR, tile tags, wavefront register, last-step encoder | §7.13 |
| `VERSAACCEL_AMORTISE` + `VERSAACCEL_STREAM` | the above plus ring addressing and the loader stall | §7.13 |
