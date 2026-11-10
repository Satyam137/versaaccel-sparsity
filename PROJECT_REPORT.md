# VersaAccel: When Does Reconfigurability Pay?

## A cost-model study of adaptive dataflow selection for sparse accelerators

**Revision 21 — 2026-11-06. The hardware comparison, measured (§7.15), and the switch cost (§4.3).**
Slow-corner Genus runs 3–4 on the pipelined RTL: VersaAccel reaches **~430 MHz** (systolic ~475,
SIGMA ~395) in **1.867 mm²** at 2.5 ns (1.98× the systolic array, 1.20× SIGMA). The
reconfigurable datapath is cheap — PE array plus Benes and reduction networks 1.20× the whole
systolic array, the networks 0.6%, the CPE 2.4% — and the overhead is the feed, schedule and
buffer logic around it. Compute-phase energy: 4.20 pJ/MAC dense, 2.54 pruned (systolic 1.89 /
1.32, SIGMA 12.78 / 4.96). Fill amortisation costs +4.5% area, streaming +2.6%; **the three-shape
reshapeable array costs +43–57% and misses its clock** — a net loss; a square + tall build
(`-DVERSAACCEL_RESHAPE_TALL`) costs +20.6% and pays per area only on weight-pruned Transformers
(1.16× on pruned BERT, 0.93× on pruned ResNet), so claim 22 is narrowed to that.
**B2:** switching configuration costs 0 cycles forced and ≤ 9 predicted — not 100–250 — so greedy
is exactly optimal and the headline is unchanged (1.0526×).

**Revision 20 — 2026-10-31. Synthesis run 2, and VersaAccel's operand feed pipelined (§7.14).**
The first slow-corner Genus run kept the Benes, N:M ROM and CPE intact (B1a) and made SIGMA a
fair baseline (B1f), but put VersaAccel's critical path at 5.6 ns against 2.0 ns for the
systolic array and 2.4 ns for SIGMA: operand address arithmetic, a 256-way buffer read, the
Benes and the PE MAC all ran in one cycle from a configuration pin. The feed is now three
registered stages with narrow address arithmetic, the configuration pins and the output port
are registered, and the CPE is six stages. Cost: 4 cycles per GEMM, none per tile; every RTL
suite and measurement re-verified on all four builds, and every result in §2–§7 stands, with
the fixed RTL-versus-accounting offsets 4 cycles larger. Genus run 3 is set up
(`syn/genus/run_college.sh`).

**Revision 19 — 2026-10-04. The cost model is corrected, and the headline with it (§3.0).**
Adopting the RTL's fill and structure terms (B11, B6) exposed that two of the published model's
four dataflow formulas charge less compute than the array can physically do — D-MR's ignores K,
D-M-R's divides by `Nc` twice — by a median 31× and 16× on real layers. §3–§6 now use the
published model held to the MAC lower bound (**B**). The selection headline falls from 1.184× to
**1.052×** (heterogeneous 1.275× → 1.079×); what survives is concentrated on pruned CNNs
(ResNet-50 1.11–1.28×, MobileNetV2 1.10–1.20×), still input-dependent there. Greedy stays within
0.013% of optimal and lookahead pays only above ~265% of a layer. §5's results reproduce
unchanged. Costed for this RTL's shared-operand array (**D**), selection pays nothing — the
choice is static, which makes the tile-geometry results (§2.7–§2.9) the paper's architectural
contribution. Every published-model number is kept beside its replacement.

**Revision 18 — 2026-10-04.** Two RTL variants and one dropped claim; nothing earlier changes.
**Fill amortisation (B10)** pipelines a GEMM's tiles through the array with a per-PE
clear/capture wavefront, paying the skew fill once per GEMM: on real pruned slices the zero-skip
saving roughly doubles (ResNet-50 11.2% → 24.8%, BERT 12.2% → 25.7% at 16×16; 24.3% / 25.2% on
the Rev 20 RTL) and skip-on compute
is ~2× faster. **Streamed activations (B12)** overlap the load with compute through a
two-tile-row ring: end to end 1.33× on ResNet slices, within 1.5% of the load alone, and GEMMs
whose activations exceed the buffer now run. Both are verified (13,215 suite checks, 3.7M
element checks, cycles = an exact model + a constant) and are compile-time defines, so the
default design is byte-identical (§7.13). The **N:M enhancer claim is dropped** (B7, §6.2).
**R4:** a full regeneration reproduces every results file (§11). **B9 (§2.11):** two more
families, MobileNetV2 and OPT-125m — the findings transfer (selection pays on the CNN, 1.147×,
not on the Transformer, ≤ 1.024×; ReLU zeros are k-structured; under Rev 19's model 1.123× /
1.000×). **R6:** the traces are shipped,
pinned by a checksum manifest.

**Revision 17 — 2026-10-03.** Scoped to 16x16 and above; all numbers regenerated at that
operating point. Original draft preserved as `PROJECT_REPORT.original.md`. Repair log and
open decisions: `FIX_PLAN.md`.

**Revision 17 adds two results aimed at the paper's novelty, both measured on the real
pretrained models (§2.7, §2.8).**

- **Skipping pays less on larger arrays, and runtime selection only matters on larger ones
  (§2.7).** The realisable zero-skipping saving falls monotonically with tile height on every
  sparse model — ResNet-50 37.0% → 19.2% → 9.2% at 4×4 / 16×16 / 64×64, pruned BERT 66.1% →
  23.6% → 2.7% — while §5.1 shows runtime selection is irrelevant below 16×16. The two
  overlap in a window of about **16×16 to 32×32**. This ties §2.5 and §5.1 into one result.
- **A free, exact compile-time rewrite recovers much of the lost structure (§2.8).**
  Reordering weight output channels so their zeros line up within a tile — with every
  consumer reordered to match, honouring residual streams and attention heads — adds
  **+7.4 points** of cycle saving on ResNet-50 pruned90 (30.0% → 37.5%) and **+5.6** on BERT
  pruned90 (23.6% → 29.2%) at 16×16, 86% and 73% of an unconstrained upper bound. The
  rewritten networks are run and their outputs are unchanged (max change 1.3×10⁻⁵ on outputs of
  scale ~10; an illegal reorder changes them by 9.8). Regrouping *activations* at runtime is
  not worth it: +3 points at best, because natural spatial order is already well grouped.
- **The tile *shape* should follow the zeros (§2.9).** At a fixed 256 PEs, choosing each
  layer's tile shape (4×64 … 64×4) by where its zeros are — narrow edge facing the sparse
  operand — is 1.97× faster than fixed 16×16 on pruned BERT and 1.21× on pruned ResNet-50
  (2.06× / 1.31× with §2.8's regrouping), where choosing the shape for dimension fit alone gives
  1.00× / 0.99×. Under a 40-operand/cycle bandwidth cap: 1.36× / 1.09× (1.46× / 1.21×). The RTL,
  built at all five shapes, matches the accounting exactly. This is the first *architectural*
  claim in the project; the hardware cost of reshaping is not yet measured.
- **Both hold at 200 inputs on two independent samples (B14)**, to within 0.6 points; the
  regrouping gain is positive on every one of 200 held-out images.
- **The cycle accounting behind both is exact on the RTL (B13).** 48 real slices of pruned
  layers run through the zero-skipping datapath in original and regrouped order: 245,760
  outputs correct, and measured cycles equal the accounting plus a fixed 19 in every case (15
  before Rev 20's feed pipeline), so the regrouping gain is a measured hardware property
  (+4.7 / +6.1 pts on those slices).

**Revision 16 makes VersaAccel's buffers separable in synthesis (B1d).** The five on-chip
buffers — 58.6% of the first run's area, flattened into the top module — are now submodules, so
the next `area_hier` reports each one. Doing so exposed a latent race in four testbenches, which
had depended on simulator scheduling order; they are now deterministic, and the refactored
design is proven **byte-identical** in behaviour to the original across every bench, all 11
GEMM shapes and the sparse sweep (§7.11). No recorded number changes. The RTL is now ready for
the second Genus run.

**Revision 15 made SIGMA a fair baseline in RTL.** `sigma_top` had 16 multipliers and a
hard-wired BYPASS that let synthesis delete its FAN. It now has **256 multipliers** — 16
Flex-DPEs × 16 lanes, iso-compute with the systolic array and VersaAccel — with a live,
register-driven FAN per unit and a pipeline matching SIGMA's own. A rewritten golden bench
checks it with 1,484 checks across all three reduction modes, partial chunks and scrambled
routing, and three mutants of the design — including the original defect — all fail it. Golden
checks across the suite: **13,215**. It needs a second synthesis run (§7.11).

**Revision 14 added the two synthesis baselines, validates the cost model against the RTL, and
makes the predictor result stronger.**

- **The baselines are in (§7.11).** The systolic array is complete and sound — its 16,423
  flip-flops are exactly 256 PEs × 64 bits plus control — and against it, VersaAccel's
  256-PE array is 6% *smaller*: reconfigurability costs nothing measurable inside the PE.
  VersaAccel's 2.29× total area is its flip-flop buffers. The Benes, measured from SIGMA's
  identical module, is 4,312 µm², about 0.3% of VersaAccel. **SIGMA is not yet a fair
  baseline**: its RTL has 16 multipliers, not 256, and its reduction network was
  constant-folded away by a hard-wired BYPASS — the same defect VersaAccel's Benes had.
- **The cost model checked against the RTL (§7.12).** Its dense term is the ideal-utilisation
  bound, off by 3.5% on BERT and 18.7% on ResNet, and 16× on the M=1 layers. Giving it the
  RTL's fill term flips 2.4% of decisions and moves the headline from 1.184× to 1.173× —
  inside its confidence interval.
- **§5.2 over 10 seeds (A8).** The MLP never beats the zero-parameter constant: 93.4% ± 5.6
  against 96.2%, worse in 8 seeds, tied in 2. The committed model's tie is the best case.
  Two stale lines in §5.2 are also corrected.

**Revision 13 was the first contact with synthesis, and it found the RTL was not yet
synthesising the design the paper describes.** The first Genus run of `versaaccel_top` met
500 MHz at 1.29 mm² — but the Benes distribution network, the N:M ROM and the trained CPE had
all been constant-folded away, because the RTL tied their control inputs to constants (§7.11).
The netlist was a systolic array without the hardware that makes VersaAccel reconfigurable,
and against SIGMA, whose Benes is port-driven, it would have looked cheaper than it is. **Fixed
in `ed6d8ac`**: the three inputs are now registered ports, and a new testbench proves each
reaches its logic — reverting to the old tie-offs fails 12 of its 31 checks. VersaAccel needs a
second Genus run; the run also used the fast corner and vectorless power, so no synthesis
number is reportable yet.

**The Rev 12 artifacts were lost, and are now rebuilt and re-run.** The working copy lived under
`/private/tmp`, which macOS clears after a few days; it was wiped between 2026-09-20 and
2026-09-26, `.git` included. The repository now lives in `~/Documents/AdaptAccel-Project`,
restored from the Rev 11 bundle. The Rev 12 code is rebuilt, and all 10 real trace sets are
re-profiled under a new environment (torch 2.8, transformers 5.17). **Nearly everything
reproduces**: the ResNet scalar columns are bit-identical to Rev 10 across 43,200 rows, BERT
differs by at most 8×10⁻⁵ on 48 of 14,600 rows, and §2.4, §2.5, §2.6, §3.1–§3.3, §4, §5.1,
§5.2 and §6.1 reproduce to every printed digit.

**Two things did not, and one of them is a correction.**

- **§5.3's Rev 12 conclusion was wrong.** "The rescue failed; the obvious objection is closed"
  came from a single seed, on 8 of 10 trace sets. Over 10 seeds on all 10, a structure
  predictor beats a per-layer compile-time table in *every* seed (+14.3% ± 6.8 MAE), while the
  density edge is indistinguishable from zero. So the MLP does find per-input structure — the
  direction Thesis 1 predicts — but the margin is 0.0016 in skippable fraction, and whether it
  changes any decision is untested until the cost model has a structure term (B6). **Claim 5
  stays withdrawn on §5.2's evidence; the rescue is open, not refuted.**
- **§8.1's worst case is torch-sensitive:** +3.106% under torch 2.14, +2.920% under 2.8, on
  identical traces and weights. The oracle bound holds in both.

**Revision 12 tested the one negative result that could still have moved, and it did not
move.** §11.8 identified claim 5 as the only one of seven negatives that a run could
legitimately overturn: the MLP might simply have regressed the wrong variable. That is now
settled.

- **B5a — the profiler measures structure.** `kslice_structure()` computes the fraction of
  all-zero `k`-slices at the 16-row tile granularity the RTL skips on. It is validated against
  the §7.7 patterns first: 0.000 on 50% unstructured zeros, 0.000 on 2:4, 0.500 on 50%
  `k`-structured — **reproducing N1 and N2 in Python, independently of the Verilog.** All 10
  real trace sets re-profiled.
- **§2.5, an unexpected positive result.** Real ReLU activations are **14–29x more
  `k`-structured than i.i.d. sparsity at the same density** — ReLU zeros cluster by channel,
  and a channel is exactly a `k`-slice after im2col. But at 20% they are also far below the
  56% the `nnz`-proportional term credits. **Both of the cost model's candidate assumptions are
  wrong, in opposite directions.**
- **§5.3, and the rescue fails.** *(Corrected in Rev 13: this rested on a single seed and 8 of 10 trace sets; over 10 seeds the rescue is open, not refuted — see §5.3.)* Retargeting the predictor at structure does not save it.
  Against a *global* constant the MLP looks excellent on both targets (79% and 92% better);
  against a **per-layer constant** — a compile-time table, zero runtime parameters — it ties on
  both, +1.1% and +4.8%. The network learned which layer it was looking at, not anything about
  the input. **Claim 5 stays withdrawn, with the obvious objection now closed.**

That is the outcome §11.8 predicted would be the more valuable one. All seven negatives are
now confirmed as findings rather than defects. *(Corrected in Rev 13: six are; N5's rescue is
open — see §5.3.)*

**Revision 11 added no new measurements.** It is a consistency pass plus a complete statement
of what remains. Three stale figures inherited by §10 are corrected against their sources
(−1.71% → −**1.51%**; the MLP's MAE edge, which is **12%** on real traces and not the 4.3x it
was on synthetic ones; a golden-check count of 452 that has been **11,716** since Revision 7),
claim 11 is marked verified now that BERT is real, and two claims that both carried the
number 10 are renumbered. **§11 is rewritten as the full road to submission** — every
outstanding task, tagged by whether it blocks submission, with a suggested order.

**§11.7 resolves C9**, the question of how to present the negative results. There are seven of
them, not the four first counted, and they do not share one mechanism: they resolve into **two
theses** — *density is not the decision variable, structure along the reduction dimension is*,
and *the selection landscape is flat, so cheap policies suffice*. Composed, the two are a
complete answer to the question in the title, and the withdrawn predictor claim stops being a
failure and becomes a prediction the first thesis makes.

The short version of §11: the *measurement* work is close to finished, and **the paper does
not exist**. There is no related work, no figure, no comparison against any external baseline,
and no README. That gap, not the science, is now the critical path.

**Revision 10 adds:** **every model trace is now real.** BERT joins ResNet: pretrained
`bert-base-uncased` on SST-2 validation text with eager attention, so attention sparsity is
actually measured (range 0.062–0.982) rather than being a property of random tokens. 10 of 15
sets are real; the remaining 5 are stress traces that are synthetic *by design*.

Three things came out of it:

- **The §5.2 result is robust.** Revision 9 flagged that the synthetic BERT half might be
  inflating the match rates. It was not: with BERT real, the config-match column is
  **identical** — 26.1 / 96.2 / 80.6 / 96.2. The MLP still ties a zero-parameter constant.
  What did change is that the MLP's *MAE* advantage largely evaporated too (0.2596 vs the
  constant's 0.2953, where on synthetic traces it was 0.075 vs 0.323).
- **A measurement error in §3.1, now corrected.** `dp_scheduler`'s "Greedy gap" and "DP gap"
  columns run the **trained predictor** to generate the sparsity sequence and compare against
  an oracle that sees the truth. They therefore measure *predictor error propagated into
  scheduling*, not scheduler optimality — and they move when the predictor is retrained. With
  a predictor trained on matching data the average gap falls from +2.76% to **+0.33%**, and
  `variable_random` goes from +36.15% to **+0.00%**. The predictor-free scheduling results
  (§3.2, §3.3, §4) are unaffected.
- **A 35x speedup in `nm_compatibility_score`**, which was a Python loop over ~590k groups per
  call and made BERT N:M profiling take hours. Vectorised, verified bit-identical.

**Revision 9 added:** **the ResNet-50 traces are now real** — pretrained `IMAGENET1K_V2`
weights on Imagenette validation images, replacing a randomly-initialised network fed
Gaussian noise. 5 of 15 trace sets are regenerated; the BERT sets are not (no `transformers`
in the environment), so §3–§6 currently **mix real ResNet traces with synthetic BERT ones** —
see §2.3.

Two results move sharply, in opposite directions:

- **The runtime premise gets much stronger.** Per-input variation in activation sparsity is
  **32x larger** on real images (std 0.0006 → 0.0206). Workloads whose optimal schedule
  varies across inputs go from **2/15 to 6/15** — all four ResNet variants now vary, where on
  noise they were static. Decision-relevant layers more than double, 3.2% → **7.2%**.
- **The predictor claim collapses.** On real traces the trained MLP scores **96.2%** config
  match on decision-relevant layers and a zero-parameter constant scores **96.2%** — a tie,
  against 8,961 parameters. Claim 5 is withdrawn as stated (§5.2). The MLP still predicts
  sparsity far better (MAE 0.075 vs 0.323); it just does not change any decision.

**Revision 8 added:** the resolution of the sparsity question Revision 7 opened, and it turned
out to be the most consequential finding in the audit. The cost model's nonzero-row estimator
was `M·√density`, which fits the paper's Table II **only because that worked example is
row-structured** — 4 nonzeros confined to 2 of 4 rows (§2.4). For the unstructured magnitude
pruning this project actually profiles, the correct estimator is `M·(1−(1−d)^K)`, and the two
differ by up to 9x. **Every number in §3–§6 has been regenerated.** The headline claims all
survive and two of them get stronger; the ≥16x16 scoping claim now carries a bandwidth
qualifier. §7.10 states the RTL's scope explicitly.

**Revision 7 added:** **zero-skipping in the sparse configurations** (§7.7) — and with it the
sharpest negative result in the report. A systolic array shares each operand along a row and
a column, so a zero can only be skipped when an entire `k`-slice of the tile is zero.
Measured: 50% *unstructured* zeros save **0%**, and 2:4 structured sparsity also saves **0%**,
while 50% *k-slice-structured* zeros save the full amount the fill overhead allows. **The cost
model's `nnz`-proportional sparsity term therefore overstates what this dataflow can
achieve**, and §7.7 gives the correction.

**Revision 6 added:** an **output double buffer and a wide output bus** (§7.6). The drain was
the accelerator's bottleneck — `NR·NC` serial cycles per tile against `K+NR+NC−2` of
compute. It is now fully hidden behind the next tile, and the steady-state cost of a tile is
its compute alone: **292 → 37 cycles at 16x16 with K=6, a 7.9x reduction**, measured.

**Revision 5 added:** a **tiling controller**, so `versaaccel_top` executes a real tiled
GEMM verified element-by-element across six shapes (§7.5). This closed the gap Revision 4
identified between the RTL and the paper's description of it.

**Revision 4 added:** a trained, loadable CPE (§7.2); golden-vector arithmetic tests for the
SIGMA baseline, the datapath primitives and the **assembled top level** (§7.1); the discovery
that the Benes network and adder tree never actually scaled past N=8 (§7.3); the discovery
that the **DMR dense-GEMM configuration was emitting undefined data** (§7.4); the structural
finding that `versaaccel_top` can only broadcast a single scalar (§7.5); the same test-set
leak fixed in the v2 predictor (§5.4); and a sensitivity sweep over every unmeasured constant
in Idea 4 (§8).

---

## 0. Status

An end-to-end audit found that three headline claims in the original draft were artifacts
rather than measurements: the "+25% from N:M" was a hardcoded constant (`0.12+0.08+0.05`),
the DP-beats-greedy claim was contradicted by the repo's own ablation, and the 99.2%
predictor accuracy was beaten by a zero-parameter heuristic. All are now corrected or
withdrawn, and the numbers below are regenerated from repaired code.

Remaining work is enumerated in **§11**, which is the authoritative task list.

**Since Revision 2, four things changed:**

1. **The project is scoped to arrays of 16x16 and above** (§5). At 8x8 activation sparsity
   provably cannot change the optimal configuration, so the premise of runtime prediction
   fails there. This is now the paper's central scoping claim rather than a hidden defect.
2. **The predictor has a real validation split.** It previously early-stopped on the test
   set. Honest test MAE is 0.0626, up from the leaked 0.0615.
3. **A decision-relevant match metric** replaces the saturated one — and on real traces it
   retires the predictor. A constant scores 96.2%, persistence 80.6%, and the trained MLP
   **96.2%**. The MLP does not earn its area (§5.2).
4. **RTL arithmetic is verified**, not just handshakes — **13,215 golden checks** at 16x16
   across the datapath primitives, both baselines, the CPE, the assembled top level and a
   tiled GEMM run at two output-bus widths, under random backpressure, and across seven
   sparsity patterns with skipping on and off. This caught two configurations silently
   emitting undefined data (§7.4), two networks that never scaled past N=8 (§7.3), a top
   level that could not load more than one activation (§7.5), and a load that terminated
   after a single element whenever `M·K` was an exact multiple of the input buffer (§7.7).

---

## 1. Base architecture

A reconfigurable PE array operating in one of four dataflow configurations:

| Config | Distribution | Multiply | Reduction | Intended for |
|:--:|:--:|:--:|:--:|:--|
| **DMR** | Systolic | Systolic | Systolic | Dense GEMM |
| **D-MR** | Explicit (Benes) | Systolic | Systolic | Moderate sparsity |
| **DM-R** | Systolic | Systolic | Explicit (Benes) | Output-sparse |
| **D-M-R** | Explicit | Systolic | Explicit | High sparsity (Gustavson) |

**Operating point: 16x16, 256 PEs.** Set once in `base_simulator/arch_config.py`; override
with `VERSAACCEL_ARRAY=8` to reproduce the pre-scoping numbers.

### 1.1 Terms and sample sizes — defined once, used throughout (A1, A4, A5, A6)

**One configuration, one name (A1).** `DMR` is a *configuration* — systolic distribution,
multiplication and reduction — and nothing else. "Dense GEMM" in this report always means a
*workload* (both operands dense), never `DMR`. The §1 table's "intended for: dense GEMM" means
DMR is the configuration designed for that workload, not that the cost model always picks it:
on these workloads D-MR is optimal even for dense BERT (§3.1). In the RTL, `CFG_DMR` is the one
configuration that ignores `skip_en`, so it is the dense baseline (§7.7). Its reduction mode
differs by path: FULL on the legacy single-shot path, BYPASS in tiled GEMM mode, where every
accumulator is delivered individually (§7.5) — a property of the GEMM path, not of DMR.

**Terms (A5).**

| Term | Meaning | First used |
|:--|:--|:--|
| decision-relevant layer | one whose optimal configuration changes as activation sparsity sweeps {0, 0.3, 0.6, 0.99}; at 16×16, 7.2% of (layer, input) points | §5.1–5.2 |
| config match | fraction of points where a predictor's sparsity estimate selects the same configuration as the true sparsity; reported on decision-relevant points | §5.2 |
| homogeneous / heterogeneous workload | one configuration is / is not optimal for every layer of the network (the five BERT sets are homogeneous) | §3.2 |
| k-slice | one `k` index of a tile: a column of the tile's NR rows of A, or a row of its NC columns of B, after im2col | §2.5, §7.7 |
| skippable fraction *f* | fraction of (row-tile, k, column-tile) steps where the k-slice is all-zero in A **or** in B — exactly the RTL's skip mask | §2.5 |
| realisable saving | `f·K/(K+NR+NC)` — what skipping *f* saves once the unskippable fill is counted; verified cycle-exact on the RTL (B13) | §2.6 |
| nnz-row estimator | how many rows of a sparse operand are non-empty: `'iid'` = M·(1−(1−d)^K), used everywhere; `'sqrt'` = M·√d, pinned only in `validate.py` because the VersaAccel paper's Table II example is row-structured and only `'sqrt'` reproduces it. The rationale is §2.4; the split is deliberate, not a bug (A4) | §2.4 |

**Sample sizes and seeds (A6).** Unless a section says otherwise:

| Result | n | Seed |
|:--|:--|:--|
| §3 scheduling, §5.1–5.2 | 15 workloads (10 real, 5 synthetic) × 200 inference passes | traces fixed; predictor split fixed |
| §3.4 prior-art form | 15 workloads × 20 passes | deterministic |
| §3.5 operating-point grid | 15 workloads × 5 passes × 36 grid points | deterministic |
| §5.2 seed robustness, §5.3 | 10 seeds (split and MLP initialisation) | 0–9 |
| §2.5–§2.6 structure | 10 real trace sets × 200 passes | traces fixed |
| §2.7–§2.9 tile geometry | 16 images / 24 sentences (8 / 12 held out); confirmed at 200 + 200 on two samples (B14) | input draw seed 0 (and 1); random control seed+1 |
| §5.3b structural model | 10 real trace sets × 20 passes; 10 seeds for the predictor test | 0–9 |
| RTL | fixed stimulus per bench; `tb_sigma_golden` seed 20260928 | fixed |

---

## 2. Method

### 2.1 Cost model — stated limitations

`base_simulator/` implements an analytical cycle model validated against the VersaAccel
paper's Table II (2x2 array, 4x4 matrices): 30/30 pass, but at **+/-30% tolerance**, with 22
exact and 8 deviating by up to 18.2%. The formulas were fitted to those specific values — the
source comments preserve the search — then extrapolated. **This is a fitted analytical model,
not a cycle-accurate simulator.** RTL cross-validation at 16x16 is the fix (§7).

### 2.2 Operand convention

A layer computes `activations(M x K) @ weights(K x P)`. The compressed operand is the
statically-known pruned weight matrix:

```
sparsity_A  <-  weight_sparsity       (static, known at deploy time)
sparsity_B  <-  activation_sparsity   (dynamic, predicted at runtime)
```

The previous code used `sparsity_A = max(activation, weight)` with `sparsity_B` never set,
which attributed weight sparsity to the activations *and* discarded activation sparsity
entirely whenever weights were sparser — 70% of rows. Both schedulers now share one
implementation (`base_simulator/layer_cost.py`); they previously carried divergent copies,
with Idea 2 inflating its competitors' costs by an unmeasured 12%.

### 2.3 Workloads and their limits

15 trace sets, 149,200 rows, 200 inference passes each.

Two limitations that must appear in the paper:

- **Every model trace is real, and the difference was large.** ResNet-50 uses pretrained
  `IMAGENET1K_V2` weights on Imagenette validation images; BERT uses pretrained
  `bert-base-uncased` on SST-2 validation text. Every CSV carries `data_source` and `weights`
  columns, and `base_simulator.trace_validation.warn_if_synthetic()` reports which is which.

  | Trace set | Status |
  |:--|:--|
  | `resnet50_{dense,pruned50,pruned70,pruned90,nm24}` | **real** — `imagenette-val` / `IMAGENET1K_V2` |
  | `bert_{dense,pruned50,pruned70,pruned90,nm24}` | **real** — `sst2` / `bert-base-uncased` |
  | `dp_stress`, `switching_stress`, `mixed_pipeline`, `gcn_synthetic`, `variable_random` | synthetic **by design** — stress traces, not model measurements |

  **Why it mattered.** On noise, activation sparsity barely varied between inputs: per-layer
  standard deviation across 200 passes was 0.0006. On real images it is 0.0206 — **32x
  larger**. The project's entire premise is that this quantity varies per input and must be
  predicted at runtime, and the old traces contained almost none of that variation. Mean
  activation sparsity also rose, 0.42 → 0.55, as expected for a trained network's post-ReLU
  activations.

  Regenerate with:

  ```bash
  python -m profiler.profile_resnet --download      # Imagenette, 342 MB
  python idea2_veriflex/profile_nm_models.py --download --dataset sst2   # BERT half needs a text source
  python -m profiler.profile_bert --dataset sst2    # needs `transformers`
  ```

  All three scripts **refuse to run on synthetic data** unless given `--allow-synthetic`.

  **BERT needs eager attention.** `transformers` 5 defaults to SDPA, which computes attention
  without ever forming the N×N probability matrix — so `attention_sparsity`, the headline
  input-dependent feature here, cannot be measured on it and the capture hook silently
  receives `None`. The profiler now passes `attn_implementation="eager"`. Measured attention
  sparsity varies from **0.062 to 0.982** across inputs, which is the variation the old random
  tokens could not produce.

- Results are now aggregated over **all 200 passes** with 95% CIs (§3.2), where previously
  every number came from a single pass.

### 2.4 The sparsity-structure assumption — and why it changes the numbers

Every sparse configuration's cycle count is gated by one question: **how many rows of A hold
at least one nonzero?** The model answered it with

```python
nnz_rows = round(M * sqrt(density))        # the original
```

which has no derivation. It was reverse-engineered from the paper's Table II worked example —
and that example is **row-structured**. Its own matrix, reproduced in `validate.py`, is

```
A = [a00  0  a02  0]     nnz_A = 4, sparsity = 0.75
    [ 0   0   0   0]     nnz_rows = 2  (rows 1 and 3 are entirely empty)
```

Four nonzeros confined to two of four rows. For *independent* zeros at that density you would
expect 2.7 rows, not 2. So the paper's published sparse cycle counts describe **structured**
sparsity.

This project's traces are not structured. They come from `prune.global_unstructured` with an
L1 criterion, which scatters zeros independently. The correct estimator there is the exact
expectation

```python
nnz_rows = round(M * (1 - (1 - density) ** K))      # exact for i.i.d. zeros
```

The two disagree by 1.4x to 9x, always in the direction that makes the sparse configurations
look cheaper under `√density`:

| density | K=16 | K=64 | K=256 |
|--:|--:|--:|--:|
| 0.50 | 0.71x | 0.71x | 0.71x |
| 0.10 | 0.39x | 0.32x | 0.32x |
| 0.01 | 0.68x | 0.21x | **0.11x** |

*(ratio = √density ÷ i.i.d.; below 1 means the heuristic understates nonzero rows, and so
understates cost)*

**Neither estimator is universally right — it has to match the workload's sparsity
structure**, and the paper's validation example and this project's workloads do not share
one. So the model now carries both (`config_modes.NNZ_ROW_MODEL`), `validate.py` pins
`'sqrt'` explicitly because that is what reproduces the paper, and everything else defaults
to `'iid'`. Under `'iid'` the paper-reproduction suite drops to 21/30, with the nine failures
all in sparse operators and all by 50–100% — which is the finding, not a bug.

**What it cost to fix.** Config selection barely moves — 1.5% of real trace points change
their argmin, every flip `D-M-R → DMR`. Absolute costs rise 1.10–1.34x. But the
decision-relevant subset *grows*, which strengthens §5 rather than weakening it. Measured by
`experiments/nnz_rows_sensitivity.py`:

| | `√density` | `iid` |
|:--|--:|--:|
| argmin changes, real traces | — | 1.5% of 2,338 points |
| argmin changes, synthetic grid | — | 9.7% of 144 |
| decision-relevant combinations | 8.3% | **20.8%** |

**This is a claim about VersaAccel, not just about this repo:** its published sparse cycle
counts are achievable on row-structured sparsity and not on the unstructured kind that
magnitude pruning produces. §7.7 reaches the same conclusion from the RTL, independently.

---

### 2.5 How structured is real sparsity? — measured, not assumed

> **Reproduced (Rev 13)** from freshly re-profiled traces, to every printed digit.

§2.4 showed that the cost model's nonzero-row estimator encodes an *assumption* about where
the zeros sit, and that the two candidate assumptions differ by up to 9x. §7.7 then showed
that the hardware only cares about one specific structure: a zero is skippable only when an
entire `k`-slice of a tile is zero. Neither section measured what real workloads actually do.

The profiler now does. `workload_profiler.kslice_structure()` computes, at the same 16-row
tile granularity the RTL skips on, the fraction of `k`-slices that are entirely zero. It is
validated against the §7.7 patterns before use: it returns 0.000 on 50% unstructured zeros and
on 2:4, and 0.500 on 50% `k`-structured zeros — reproducing N1 and N2 in Python, independently
of the Verilog.

**Both of the cost model's candidate estimators are wrong, in opposite directions.**

| Workload | Activation density | **Measured** all-zero `k`-slices | i.i.d. predicts | `nnz`-proportional assumes |
|:--|--:|--:|--:|--:|
| resnet50_dense | 0.561 | **0.202** | 0.015 | 0.561 |
| resnet50_pruned50 | 0.550 | **0.192** | 0.013 | 0.550 |
| resnet50_pruned70 | 0.530 | **0.177** | 0.011 | 0.530 |
| resnet50_pruned90 | 0.476 | **0.192** | 0.007 | 0.476 |
| bert_dense | 0.001 | 0.000 | ~0 | 0.001 |

All 10 real trace sets carry these columns; the 5 stress traces are synthetic by design and do
not.

Real ReLU activations are **14–29x more `k`-slice-structured than i.i.d. sparsity at the same
density** — ReLU zeros cluster by channel, and a channel is exactly a `k`-slice after im2col.
But they are also far *less* structured than the `nnz`-proportional term assumes: 20% where
the cost model credits 56%.

So the honest statement is narrower and more useful than either: **the skippable fraction is a
property of the workload that must be measured, and it sits roughly a third of the way between
the two models architects reach for by default.** This is the empirical leg of Thesis 1
(§11.7), and it is what B6 should encode.

**And the weights carry no exploitable structure at all** — under either pruning scheme. This
is N2 (§7.7) confirmed on real pruned models rather than on a synthetic pattern:

| Workload | Weight sparsity | **Measured all-zero `k`-slices** | `nnz`-proportional assumes | Overstatement |
|:--|--:|--:|--:|--:|
| resnet50_nm24 (2:4) | 0.500 | **0.0012** | 0.500 | **417x** |
| bert_nm24 (2:4) | 0.500 | **0.0003** | 0.500 | **1667x** |
| resnet50_pruned50 (unstructured) | 0.396 | **0.0015** | 0.396 | 264x |
| bert_pruned50 (unstructured) | 0.506 | **0.0017** | 0.506 | 298x |

So the structure ResNet has comes **entirely from ReLU, not from pruning**. ReLU zeros cluster
by channel; magnitude pruning scatters its zeros, and 2:4 deliberately spreads them evenly
along rows, which for a shared-operand array is the worst possible arrangement. Neither
pruning method in common use produces sparsity this dataflow can exploit, and the 2:4 pattern
that hardware vendors have standardised on is the least exploitable of the two.

That is a sharper statement of the N:M negative result (§6) than the cycle counts alone give,
and it is measured on real pruned networks.

Two further consequences worth stating in the paper:

- **BERT has essentially no activation structure to exploit.** Its density is 0.001 — GELU does
  not produce exact zeros the way ReLU does — so every skippable slice in a pruned BERT comes
  from the *weights*, which are static. For transformers this is a compile-time property.
- **ResNet's structure is input-dependent, but only mildly.** Across 200 images the per-layer
  standard deviation of the skippable fraction is 0.032 against a mean of 0.20. That is real
  variation, and §5.3 tests whether it is enough to justify predicting it at runtime.

---

### 2.6 What that structure is worth — and why it does not weaken the result

> **Reproduced (Rev 13)** from freshly re-profiled traces, to every printed digit.

A reasonable worry about §2.5: if real activations *are* 14–29x more structured than i.i.d.,
does that not undercut §7.7's "unstructured sparsity saves 0%"? It does not, and the
arithmetic is worth putting in the paper rather than leaving to a reviewer to attempt.

Converting the measured structure into cycles through §7.7's own ceiling,
`saving = f·K/(K+NR+NC)`, with layers weighted by `M·K·P`:

| Workload | Skippable `f` | **Realisable saving** | `nnz` model credits | Model error |
|:--|--:|--:|--:|--:|
| resnet50_dense | 0.201 | **19.3%** | 57.7% | −38.4 pts |
| resnet50_nm24 | 0.226 | **21.5%** | 54.7% | −33.2 pts |
| resnet50_pruned50 | 0.194 | **18.5%** | 56.9% | −38.4 pts |
| resnet50_pruned70 | 0.191 | **18.2%** | 55.5% | −37.3 pts |
| resnet50_pruned90 | 0.305 | **29.3%** | 49.8% | −20.6 pts |
| bert_dense / nm24 / pruned50 | ~0.000 | **0.0%** | 0.1% | ~0 |
| bert_pruned70 | 0.013 | **1.3%** | 0.1% | +1.2 pts |
| **bert_pruned90** | 0.244 | **23.7%** | **0.0%** | **+23.7 pts** |

**Mean realisable saving across the 10 real sets: 13.2%.**

Three things follow, and all three are results rather than caveats.

1. **The cost model is wrong in both directions.** It over-credits ResNet by up to 38
   percentage points, because only part of ReLU's sparsity is `k`-structured. And it *entirely
   misses* `bert_pruned90`, where a 23.7% saving is available and the model credits 0.0% —
   because that benefit comes from **weight** structure, which an activation-sparsity term
   cannot see by construction. A term that is merely optimistic can be scaled down. A term
   that is optimistic on one workload and blind on another cannot be fixed by a constant, and
   that is the argument for B6.
2. **§7.7 is untouched.** It measured a *pattern* — i.i.d. zeros save 0%, `k`-structured zeros
   save the ceiling. §2.5 measured *workloads* and found real ReLU sparsity is neither, sitting
   about a third of the way up. The two results compose; neither contradicts the other.
3. **§2.5 is the cleanest demonstration of Thesis 1 in the whole project**, precisely because it
   holds density fixed. At ~50% weight sparsity, 2:4 yields 0.0012 skippable slices and ReLU
   activations at a comparable density yield 0.20 — **a 170x difference in benefit at the same
   nonzero count.** §7.7 shows this with synthetic patterns; §2.5 shows it on real networks,
   which is the version a reviewer will believe.

So the honest headline is **not** "sparse dataflows do not work". It is: *a shared-operand array
realises about 13% of the benefit the nonzero count advertises, the gap is entirely explained by
structure, and the quantity that predicts it is measurable ahead of time.*

---

### 2.7 Tile height: why skipping and selection pull in opposite directions

> **New in Rev 17.** `experiments/tile_geometry.py`, real pretrained models re-run on 16
> Imagenette images and 24 SST-2 sentences. At 16×16 it reproduces §2.6 to within 0.01
> (skippable 0.194 / 0.302 / 0.229 / 0.244 against 0.201 / 0.305 / 0.226 / 0.244).

A `k`-step is skippable only when it is zero across every row of the tile (§7.7). Taller tiles
should therefore expose less of it. Measured on square T×T tiles, with cycles charged exactly as
the RTL charges them — `K + 2T − 1` per tile pair dense, active steps `+ 2T` with skipping on:

| Workload | 4×4 | 8×8 | **16×16** | 32×32 | 64×64 |
|:--|--:|--:|--:|--:|--:|
| resnet50_dense | 37.0% | 26.5% | **19.2%** | 12.3% | 9.2% |
| resnet50_nm24 | 42.0% | 29.4% | **21.9%** | 14.5% | 10.5% |
| resnet50_pruned90 | 60.9% | 43.8% | **29.6%** | 18.5% | 12.7% |
| bert_pruned90 | 66.1% | 45.4% | **23.6%** | 8.7% | 2.7% |
| bert_dense | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

*(Realisable cycle saving from zero-skipping, work-weighted over all layers and inputs.)*

**The saving falls monotonically with tile height on every sparse workload**, and fastest
where the structure comes from pruned *weights* (BERT, 66% → 3%), which scatter more than
ReLU's channel-clustered zeros. Independently, dense utilisation falls too — ResNet keeps the
array 97% busy at 4×4 and 63% at 64×64, because its small late layers fill a large array poorly.

**Set against §5.1, this is a tension, not two facts.** §5.1 shows runtime selection cannot
matter below 16×16 — activation sparsity changes the optimal configuration in 0% of cases at
4×4 and 8×8, 20.8% at 16×16 and 33.3% at 32×32. This section shows the sparse dataflows'
mechanism is worth less with every doubling of the array. The two benefits overlap in a window
of about **16×16 to 32×32**: below it there is nothing to select at runtime, above it there is
little left to skip. That is the paper's operating-point argument stated as a measurement
rather than a choice, and it is the reason the project is scoped where it is.

**What it does not show.** It measures the skip mechanism of a shared-operand array, not the
Gustavson dataflow the cost model's sparse configurations describe (§7.9). And absolute
throughput still rises with array size; the claim is about the *fraction* sparsity buys.

### 2.8 Regrouping: a free rewrite that lines up the zeros

> **New in Rev 17.** Same script and inputs as §2.7. Activation regroupings are scored on the
> held-out half of the inputs; weight regroupings are compile-time and input-independent.

Which rows share a tile, and which output channels share a column tile, are free choices: a
reordering of output rows or channels leaves every arithmetic result intact if its consumers
are reordered to match. If rows or channels whose zeros coincide are grouped together, more
`k`-steps become skippable. Two kinds of regrouping were tested:

- **Activation rows, per input, at runtime** — needs the previous layer's output before tiling.
- **Weight output channels, once, at compile time** — a network rewrite with zero runtime cost.

Regroupings are built greedily: seed a tile with the row (or channel) with most zeros, then add
whichever keeps the most `k`'s zero across the tile.

**Weight channels cannot all be reordered independently.** Channels on a residual stream are
shared by every block that adds into it; query and key must share an order because attention
contracts over them; neither may leave its 64-wide head. The deployable variant (`wshared`)
honours exactly those constraints — one order per ResNet residual stage, one for BERT's
residual stream across all 12 layers, query and key sharing one per layer, value its own, all
within heads. **It is verified, not argued:** `verify_permutation()` applies the orders to a
copy of each real network and re-runs it. Outputs are unchanged on all five models (max change
1.3×10⁻⁵ on outputs of scale ~10), and an illegal order that lets query cross heads changes
BERT's output by 9.8 — so the check would catch a wrong constraint.

**Cycle saving at 16×16:**

| Regrouping | resnet50_pruned90 | bert_pruned90 | resnet50_nm24 | resnet50_dense |
|:--|--:|--:|--:|--:|
| random row order | 26.4% (−3.6) | 23.6% | 17.0% (−5.4) | 14.3% (−5.1) |
| **original order** | **30.0%** | **23.6%** | **22.4%** | **19.4%** |
| activation rows regrouped per input (runtime) | 32.8% (+2.8) | 23.6% (+0.0) | 25.9% (+3.5) | 22.4% (+3.0) |
| **weight channels, legal rewrite (`wshared`)** | **37.5% (+7.4)** | **29.2% (+5.6)** | 23.0% (+0.6) | 19.4% (+0.0) |
| weight channels, every layer free (upper bound) | 38.7% (+8.6) | 31.2% (+7.6) | 23.4% (+1.0) | 19.4% (+0.0) |
| both (upper bound) | 41.0% (+11.0) | 31.2% (+7.6) | 26.9% (+4.5) | 22.4% (+3.0) |

Three findings:

1. **The weight rewrite is the useful one.** At zero hardware and zero runtime cost it raises
   the saving on pruned networks by about a quarter — 30.0% → 37.5% on ResNet, 23.6% → 29.2% on
   BERT — recovering 86% and 73% of what unconstrained per-layer reordering could reach.
2. **2:4 and dense weights gain nothing**, which is the expected control: dense weights have
   no zeros to line up, and 2:4 spreads its zeros deliberately, so no order concentrates them.
   This is §2.5's N:M finding seen from the other side.
3. **Runtime activation regrouping is not worth building.** Even a per-input greedy search adds
   only ~3 points, and would need a sort in the datapath. The reason is itself a finding: the
   natural spatial order of im2col rows is already well grouped — neighbouring output pixels
   share zero channels — so shuffling it *loses* 4–5 points. A fixed order learned from
   calibration images is worse than the natural one (`static`, −1.5 to −2.3).

**Regrouping at every tile height** (`wshared`, saving and gain over original order):

| Tile | resnet50_pruned90 | bert_pruned90 |
|:--|:--|:--|
| 4×4 | 60.9% → 65.1% (+4.2) | 66.1% → 67.8% (+1.7) |
| 8×8 | 43.8% → 50.5% (+6.7) | 45.4% → 49.1% (+3.7) |
| **16×16** | 29.6% → 37.1% **(+7.5)** | 23.6% → 29.2% **(+5.6)** |
| **32×32** | 18.5% → 24.8% **(+6.4)** | 8.7% → 14.1% **(+5.3)** |
| 64×64 | 12.7% → 16.5% (+3.8) | 2.7% → 6.0% (+3.4) |

**The gain peaks inside the §2.7 window.** At 16×16 and 32×32 the rewrite adds the most in
absolute terms, and at 32×32 the most in relative terms — +34% on ResNet, +62% on BERT. It does
not stop the decline with tile height; it recovers a substantial part of what larger arrays
lose. That is the claim to make, and no stronger one.

**Measured on the RTL (B13).** `experiments/regroup_rtl.py` takes 48 real slices of pruned
layers — five ResNet-50 pruned90 layers and three BERT pruned90 layers, three slices each at
the two shapes the buffers hold (16×16×32 and 32×8×64) — regroups each slice's output channels
with the same greedy search, and runs original and regrouped operands through `versaaccel_top`'s
zero-skipping path (`tb_regroup.sv`), skipping off and on:

| Check | Result |
|:--|:--|
| Output elements checked against an independent reference | **245,760, 0 errors** |
| Regrouped C equals original C with its columns permuted | **all 48 cases** |
| Skip-off cycles, original vs regrouped | **identical** — regrouping changes nothing without skipping |
| Measured cycles − accounting, every case and both modes | **exactly 19** (fixed drain and handshake; 15 before Rev 20's +4-cycle feed pipeline, §7.14), no other value |
| Cycles saved by regrouping, measured vs accounting | **equal in all 48 cases** |
| Measured compute saving, ResNet slices | 1.5% → **6.2%** (+4.7 pts; Rev 19 RTL: 6.3%, +4.8) |
| Measured compute saving, BERT slices | 3.2% → **9.2%** (+6.1 pts; Rev 19 RTL: 9.4%, +6.2) |

So the cycle accounting behind every §2.7–§2.8 number is exact on the RTL, not just close, and
the regrouping gain is a measured property of the hardware. The absolute savings here are small
because the buffers limit these slices to K ≤ 16, where the 31-cycle skew fill dominates (§7.7);
the full-layer numbers above use the real K, through the same now-verified accounting. One slice
in 48 got one cycle *worse* (layer4.0.conv1, 102 → 103) — the greedy search is a heuristic, and
the accounting predicted that case exactly too.

**At 200 inputs, on two independent samples (B14).** The tables above use 16 images and 24
sentences. Re-run at 200 images and 200 sentences per sample, on two disjoint random draws
(`--chunk 10 --seed 0/1`; 100 calibration + 100 held-out each), with every rewritten network
re-verified (max output change ≤ 2.1×10⁻⁵):

| | 16 inputs | 200, sample 0 | 200, sample 1 |
|:--|--:|--:|--:|
| resnet50_pruned90, original → `wshared` (16×16) | 30.0% → 37.5% | 29.8% → 37.3% | 29.6% → 37.1% |
| — gain per held-out image: mean ± sd (min) | +7.4 | **+7.47 ± 0.31 (+6.43)** | **+7.50 ± 0.22 (+6.88)** |
| bert_pruned90, original → `wshared` (16×16) | 23.6% → 29.2% | 23.6% → 29.2% | 23.6% → 29.2% |
| resnet50_nm24 `wshared` gain | +0.6 | +0.58 ± 0.03 | +0.59 ± 0.02 |
| runtime activation regrouping, resnet50_pruned90 | +2.8 | +3.10 | +3.12 |
| resnet50_dense saving at 4 / 16 / 64 | 37.0 / 19.2 / 9.2% | 37.4 / 19.8 / 9.6% | 37.4 / 19.8 / 9.6% |
| resnet50_pruned90 saving at 4 / 16 / 64 | 60.9 / 29.6 / 12.7% | 61.0 / 29.9 / 12.9% | 60.9 / 29.7 / 12.7% |

**Nothing moves by more than 0.6 points, and the two samples agree to within 0.5.** The
ResNet regrouping gain is positive on **every one of 200 held-out images**, by at least 6.4
points — it is a property of the weights, not of a lucky draw. BERT is identical to the digit
across all three runs, which is expected rather than suspicious: its activations have no exact
zeros, so all of its skippable structure is in the static weights and does not depend on the
sentence. Per-input CSVs: `results/tile_geometry_*_n200_s{0,1}_by_input.csv`.

**Limits.** The gain is confined to pruned networks — which are exactly the ones whose benefit
§2.6 showed the cost model cannot see.

### 2.9 Tile shape: put the narrow edge where the zeros are

> **New in Rev 17.** `experiments/tile_shape.py` (16 images / 24 sentences — §2.8's B14 run
> shows these agree with 200 inputs to within 0.6 points) and `experiments/tile_shape_rtl.py`.

§2.7 varied square tiles. But the two operands are tested on *different* edges of a tile: a
`k`-step is skippable when it is zero across the tile's NR rows of A **or** across its NC columns
of B. ReLU zeros live in A, pruned-weight zeros in B. At a fixed 256 PEs a tile can be 16×16, or
make one edge narrow — 4×64, 8×32, 32×8, 64×4 — and a narrow edge is all-zero far more often
(for i.i.d. density d, the chance falls as dᵂ in the edge width W). So the best shape should be
the one whose narrow edge faces the sparse operand, and that differs per layer.

Cycles are charged as the RTL charges them (`K + NR + NC − 1` per tile pair dense,
active steps `+ NR + NC` skipping). Three per-layer policies are compared, all at 256 PEs and all
against fixed 16×16 with skipping on; shapes are chosen on the calibration half of the inputs
and scored on the held-out half:

- **fit** — each layer takes the shape with the fewest *dense* cycles: the gain any reshapeable
  array gets from matching layer dimensions, with no sparsity involved. This is prior art
  (reshapeable / fissionable arrays) and is reported only to be subtracted.
- **structure** — each layer takes the shape with the fewest *skip* cycles.
- the same two after §2.8's legal channel regrouping, redone for each shape's column width.

| Workload | fit | **structure** | fit + regroup | **structure + regroup** | shapes chosen (structure) |
|:--|--:|--:|--:|--:|:--|
| bert_pruned90 | 1.00× | **1.97×** | 1.08× | **2.06×** | 64×4 on 72 of 73 layers |
| resnet50_pruned90 | 0.99× | **1.21×** | 1.09× | **1.31×** | mixed: 64×4 ×30, 16×16 ×9, 4×64 ×7, … |
| resnet50_dense | 1.15× | **1.22×** | 1.15× | 1.22× | 4×64 ×39, 8×32 ×8, 16×16 ×7 |
| resnet50_nm24 | 1.13× | **1.20×** | 1.14× | 1.20× | 4×64 ×39, 8×32 ×8, 16×16 ×7 |
| bert_dense | 1.00× | 1.00× | 1.00× | 1.00× | 16×16 |

*(Speedup over fixed 16×16, original channel order, skipping on. Held-out choices equal the
held-out oracle in every case, so the per-layer choice is stable across inputs.)*

**The difference between the fit and structure columns is what sparsity structure adds to
choosing a shape** — +97% on pruned BERT, +22% on pruned ResNet, +7% on dense ResNet — and it
is the part that is new. It sorts exactly as the mechanism predicts: pruned BERT's skippable
structure is all in its weights, so it wants narrow *columns* (64×4) on every layer; dense
ResNet's is all in ReLU activations, so it wants narrow *rows* (4×64); pruned ResNet has both
and mixes shapes layer by layer.

**Operand bandwidth is the cost the accounting omits.** An NR×NC array consumes NR + NC operands
per cycle: 32 at 16×16, 40 at 8×32 / 32×8, 68 at 4×64 / 64×4. The RTL reads resident operands
from flip-flop buffers in parallel and does not model this. Restricting the choice to shapes
needing at most 40 operands per cycle:

| Workload | structure, all shapes | **structure, ≤ 40 operands/cycle** | with regroup, ≤ 40 |
|:--|--:|--:|--:|
| bert_pruned90 | 1.97× | **1.36×** | 1.46× |
| resnet50_pruned90 | 1.21× | **1.09×** | 1.21× |
| resnet50_dense | 1.22× | **1.11×** | 1.11× |

Smaller, still real, and the honest number for a bandwidth-matched design.

**Checked on the RTL.** `versaaccel_top` takes NR and NC independently. Built at all five shapes
(`tile_shape_rtl.py`) and run on 30 random sparse GEMMs with skipping off and on: **0 output
errors, and RTL cycles = accounting + (NC − 1) + 4** — the last tile's exposed drain, plus the
feed pipeline and output register (§7.14) — in both modes at every shape. The shape results are therefore exact for this datapath.

**This refines the structure thesis rather than contradicting it.** §7.7's "50% unstructured
zeros save 0%" is a statement about 16-wide edges at 50% density. The general statement is
*structure relative to the tile edge*: a zero pattern is exploitable when it is aligned along the
edge facing its operand, and a narrow edge makes even high-density unstructured sparsity
partly exploitable (pruned BERT at 64×4). §2.5's i.i.d. comparison and §7.7 should be read as
the 16-wide case of that rule.

**B15 — the reshapeable array, built and measured.** `versaaccel_top` built with
`-DVERSAACCEL_RESHAPE` runs its 16×16 array as a square, a WIDE 8×32 or a TALL 32×8 tile, chosen
per GEMM on a `tile_shape` pin. It cuts the array at a midline and injects a second operand feed
(16 muxes per cut in `pe_array`, a second fetch port bank), keeps the nonzero bitmaps at
half-edge granularity and ORs them per tile. Each half is a physical 8×16 or 16×8 array, so the
fill is **8 + 16 = 24 cycles, not the 40 a true 8×32 would pay** — the reshaped tiles start up
faster than the virtual ones above. Operand bandwidth is 40 values/cycle in both reshaped modes.

- **Unchanged without the define.** All 11 benches, the 11 GEMM shapes and both sparse sweeps
  are byte-identical to the pre-change RTL, in the default build and in the reshape build run
  square.
- **Correct and exactly accounted in all three shapes** (`experiments/reshape_rtl.py`): 122,880
  output checks on random sparse GEMMs and 48 real pruned-layer slices, 0 errors; RTL cycles =
  accounting + 19 in every run (+15 before Rev 20), skipping off and on.

Re-scored at full layer size with the RTL's real geometry (`tile_shape.py`, `FISSION`;
`results/tile_shape_fission_summary.csv`), speedup over fixed 16×16 with skipping:

| Workload | fixed WIDE 8×32 | fixed TALL 32×8 | **per layer, by structure** | with §2.8 regroup | shapes chosen |
|:--|--:|--:|--:|--:|:--|
| bert_pruned90 | 0.85× | 1.40× | **1.40×** | **1.50×** | TALL on 72 of 73 layers |
| resnet50_pruned90 | 1.04× | 1.12× | **1.15×** | **1.28×** | TALL ×32, WIDE ×22 |
| resnet50_dense | 1.16× | 0.92× | 1.16× | 1.16× | WIDE on all 54 |
| resnet50_nm24 | 1.16× | 0.92× | 1.16× | 1.17× | WIDE ×53 |
| bert_dense | 1.01× | 1.01× | 1.01× | 1.01× | — |

**Read it as an orientation result.** The mechanism predicts exactly what happens: pruned BERT,
whose zeros are in the weights, wants the narrow *column* edge (TALL) and gets 1.40×; the wrong
orientation is *slower* than square (0.85×). Dense ResNet, whose zeros are ReLU activations,
wants the narrow *row* edge (WIDE). The orientation cannot be chosen from layer dimensions: for
pruned BERT the two reshaped shapes have identical dense cost, so a sparsity-unaware policy is
a coin flip between 1.40× and 0.85×. **Most of the gain is a per-model orientation choice
driven by where the zeros live; choosing per layer adds a further +3% on pruned ResNet** (1.12× →
1.15×, 1.27× → 1.28× regrouped), where weight- and activation-sparse layers coexist. Dense
ResNet's 1.16× is the shorter fill and better dimension fit of WIDE — not a sparsity effect,
and a gain any reshapeable array would get.

**Under an operand-bandwidth budget (B16).** Rather than excluding shapes above a cap, a buffer
delivering BW operands per cycle stretches each compute cycle by max(1, (NR+NC)/BW)
(`tile_shape_bandwidth.py`, from the same results):

| Per layer, by structure | BW = 32 (what 16×16 already needs) | BW = 40 | BW = 64 | unlimited |
|:--|--:|--:|--:|--:|
| RTL shapes, bert_pruned90 (regrouped) | 1.12× (1.20×) | 1.40× (1.50×) | 1.40× | 1.40× |
| RTL shapes, resnet50_pruned90 (regrouped) | 1.01× (1.13×) | 1.15× (1.28×) | 1.15× | 1.15× |
| idealised 4×64…64×4, bert_pruned90 | 1.09× | 1.36× | 1.85× | 1.97× |

The RTL's reshaped tiles need only 40 operands per cycle and keep their full gain from there
up; even with no more bandwidth than the square array uses, pruned BERT keeps 1.12× (1.20×
regrouped), because a stretched TALL tile still skips more than a square one runs. The
idealised extreme shapes need ~64 operands/cycle to approach their 2× — the reason the RTL
implements only the two half-array shapes.

The small-slice RTL measurement (`reshape_rtl.py`, part 2: WIDE ahead on both models) is a K ≤ 16
artefact of the buffers — at that K the 24- vs 31-cycle fill dominates every tile — and is
consistent with the accounting, which matches the RTL there exactly.

**What is not yet shown.** (1) The hardware cost of reshaping: the RTL exists (B15, above); its
Genus area and timing, with versus without the define, are what remain before a net win can be
claimed. (2) Operand bandwidth enters the cost model only
as the cap above, not as a modelled memory system (B16). (3) Novelty against reshapeable-array
prior art must be established in related work (B17).

### 2.10 Related work for §2.7–§2.9 — what is and is not new (B17, three passes)

> **First pass, 2026-10-03, not exhaustive.** Web search over reshapeable systolic arrays,
> channel permutation for sparsity and tile-granularity skipping. It narrows two of the three
> Rev 17 claims and must be extended before submission.

**Reshapeable arrays — shape for utilisation.** Planaria (Ghodrati et al., MICRO 2020) fissions a
systolic array into sub-arrays for multi-tenancy; FlexSA (Lym & Erez, 2020) splits into
sub-systolic modes for pruned-model *training*; ReSA (2024) partitions into sub-arrays for tiny
tensors; FlexiSAGA (SAMOS 2025) is configurable in height and width with seven sparse/dense
dataflows and a co-designed pruning method. All choose a shape for utilisation or bandwidth.
This is the "fit" column of §2.9, which §2.9 already subtracts.

**SPOTS — the closest prior work, and it narrows claim 22.** SPOTS (Soltaniyeh, Martin &
Nagarakatte; Rutgers TR DCS-TR-756, journal version ACM TACO 2022) is a systolic array
reconfigurable between several square GEMM units and one *tall* array, and it states the §2.9
mechanism for activations directly: because a tall array admits fewer im2col columns at a time,
it "increases the opportunity for detecting and skipping entire rows of inputs with zeros". Its
skip test — AND of an activation-row bitmap and a weight-column bitmap — is the same `mask` as
§7.7. Weight-side skipping comes from a *group-wise pruning* method with its own format, i.e. by
changing the model. So **"a narrow tile edge makes zeros skippable" is not new.** What SPOTS does
not do, as far as this pass shows: choose the orientation per layer from measured structure —
it fixes one tall orientation for activation sparsity — or quantify shape against structure at
fixed PE count. Claim 22 is restated accordingly:

> *At fixed PE count, the best tile orientation flips with where a layer's zeros live — narrow
> columns for pruned weights (64×4 on 72 of 73 pruned-BERT layers), narrow rows for ReLU
> activations (4×64 on dense ResNet) — and choosing it per layer from measured structure gives
> +97% (pruned BERT) and +22% (pruned ResNet) beyond shape-for-fit, on unmodified pruned models.*

The bidirectional, per-layer, measured, no-retraining part is the defensible novelty; the
mechanism itself must be credited to SPOTS.

**Channel permutation — the rewrite machinery is not new.** Pool & Yu (NeurIPS 2021) permute
weight channels *and adjust the surrounding layers* so the network is unchanged — the same
exactness machinery as §2.8 — to recover accuracy under 2:4 pruning. TETRIS (Ji et al., NeurIPS
2018) reorders input/output dimensions to cluster small weights, then prunes the clusters.
Column combining (Kung, McDanel & Zhang, ASPLOS 2019) packs sparse columns with retraining.
Tile-wise sparsity (Guo et al., SC 2020) prunes at tile granularity for tensor cores. All of these
permute or prune *to change which weights survive*. §2.8 permutes *after* pruning, changes no
weight, and optimises a different objective: shared zeros across the edge of a shared-operand
tile. That objective — and the residual-stream / attention-head constraint handling, verified by
re-running the network — is what §2.8 can claim; "exact channel permutation" itself it cannot.

**Modelling.** Sparseloop (Wu et al., MICRO 2022) models gating, skipping and compression
analytically, including at tile granularity, so §2.7–§2.9's *accounting* is within its
expressive range. What this project adds is the measurement on real pretrained models and the
cycle-exact RTL agreement (B13, `tile_shape_rtl.py`).

**Not found in this pass:** prior work on the §2.7 tension — that skipping is worth less with
every doubling of the array while runtime selection only matters from 16×16 — or on choosing
tile orientation per layer from measured structure. Absence from a first pass is not evidence of
novelty; B17 stays open until a systematic search (ISCA/MICRO/HPCA/ASPLOS 2019–2026, plus
Sparseloop-based studies) is done.

**Second pass (2026-10-04) — dataflow selection, learned predictors, activation structure.**

- **Per-layer dataflow selection is established.** MAERI (ASPLOS 2018) made dataflow a
  per-layer mapping choice via reconfigurable interconnect; Flexagon (ASPLOS 2023) is a sparse
  accelerator that switches among inner-product, outer-product and Gustavson dataflows **per
  DNN layer**, chosen by offline profiling; OMEGA (Garg et al., IPDPS 2022) maps the
  sparse/dense phase space for GNNs. So §3's *mechanism* (choose the dataflow per layer) is
  prior art; what §3–§5 add is *when* it pays — the homogeneous/heterogeneous split, the
  6-of-15 runtime-vs-compile-time result, and the array-size and switch-cost thresholds.
- **Misam (Yadav, Namjoo & Asgari, MICRO 2025) is the closest work to §4 and §5, and it points
  the other way on prediction.** A lightweight decision tree predicts the SpGEMM dataflow from
  matrix features with 90% selection accuracy over 6,219 matrices spanning 1–99% sparsity
  (SuiteSparse plus pruned DNN layers), deployed as FPGA bitstreams, with a reconfiguration
  engine that switches only when the predicted gain exceeds the switch overhead — up to 10.76×.
  **Position against it explicitly.** Its cost-aware switching is §4's mechanism; §4's
  contribution is the quantified threshold (~265% of a layer at 16×16 under the corrected
  model, §3.0; ~121% as published) and the finding that greedy is optimal below it. Its learned selector wins across a
  1–99% sparsity range that crosses dataflow regimes; §5.2 finds that *within* DNN inference at
  ≥16×16 the decision boundary is crossed so rarely that a zero-parameter constant matches an
  MLP. The two are compatible — they bracket where learned selection pays — and the paper
  should say so rather than present §5.2 as contradicting learned selection in general. Misam
  also compares against heuristics, not against a constant on decision-relevant points; §5.2's
  decision-level metric is a methodological point worth making alongside it.
- **Activation structure is known to be exploitable when induced.** Grimaldi et al. (2023,
  *semi-structured activation sparsity*) group activations by their im2col position and
  *train* for channel-structured zeros; SPOTS skips all-zero im2col blocks. §2.5 differs in
  measuring the structure *natural* ReLU already has — 14–29× above i.i.d. on pretrained
  ResNet-50, at the RTL's tile granularity — and in showing pruning adds none of it.
- **Runtime sparsity monitoring for scheduling** exists (Sparse-DySta, 2023: a hardware monitor
  and a linear latency model for multi-DNN scheduling), so "measure, don't predict" (Idea 2 in
  the novelty discussion) would need to be positioned against it.

**Still not found after two passes:** the §2.7 tension (skipping's value falls with array size
while selection's rises) and per-layer tile *orientation* chosen from measured structure (§2.9).

**Third pass (2026-10-16) — reordering for zero-skipping, tile granularity, sparsity-driven shape.**
This pass found the closest prior work to §2.8, and some of it is from this group.
- **The group's own ISQED 2026 paper permutes a different axis, for a different purpose (read in
  full, 2026-11-07).** Chaitanya, Bhargav & Rao (ISQED 2026, *Sparsity Aware Pre-processing for
  Systolic Array Dataflow Acceleration*, IIIT Bangalore, DOI 10.1109/ISQED69900.2026.11534747) put a
  hardware pre-processing block between the host and a 4×4 output-stationary array. Per 4×4 tile
  it swaps **columns of the sparse weight matrix A and the matching rows of B — the reduction
  dimension k** — so that zeros land at the tile's corner positions (A₁, A₁₆, then (A₂, A₅) and
  (A₁₂, A₁₅)), letting the array **skip the first and last skew-wavefront cycles**: 11 → 7–9
  cycles per 4×4 tile, 3N−1 → 3N−5 at best. Evaluation: a TensorFlow tiling cycle model on
  LeNet-5 (CIFAR-10), VGG-11 (MNIST) and a minimal ResNet-50 (SVHN), pruned 1:2 and unstructured
  (9–34% fewer cycles on ResNet-50); Yosys + OpenSTA on ASAP7 for the 4×4 array (+18.3% area,
  +36% static, −50% dynamic power). Against §2.8:
  - **Axis.** They permute k, inside one GEMM, so exactness is the one-line identity they prove
    (C = A′B′); nothing outside the layer changes. §2.8 permutes **output channels**, which
    changes the layer's output order and must be propagated to every consumer — residual stages,
    query/key sharing, attention heads — the constraint handling verified by re-running the
    networks. Permuting k cannot help §2.8's objective at all: it reorders k-slices but cannot
    make one all-zero.
  - **Objective.** They trim the **skew fill/drain** at the tile's edges; §2.8 creates **whole
    skippable k-steps** in mid-stream on a shared-operand array. They are different overheads —
    and the one they attack is the one B10 removes structurally (§7.13: the fill is paid once
    per GEMM, not per tile), after which corner-trimming saves a few cycles per GEMM.
  - **Hardware.** A runtime pre-processing block (+18% at 4×4) against none: §2.8 is a
    compile-time rewrite.
  - **Array size — supporting §2.7.** Their gain shrinks as the array grows (on LeNet-5 at 25–90%
    sparsity a monolithic 16×16 array saves 3–8% where sixteen 4×4 tiles save 5–32%, their Figs.
    9–10), and they recommend tiling small arrays: the
    same direction as §2.7, for the same reason (a fixed per-edge saving against a growing tile).
    §2.7 should cite it as independent evidence.
  So ISQED 2026 is related work to cite first and position against, not a precedent for §2.8's
  mechanism; and it is relevant to B10 and §2.7 as much as to §2.8.
- **Reordering to cluster zeros into skippable blocks is an established idea in other
  hardware.** Granular Matrix Reordering (Mikhailenko et al., IEEE CAL 2020) permutes the rows
  and columns of pruned weight matrices so that fewer ReRAM crossbars are charged, with no
  accuracy loss: −28% energy on average across seven pruned networks. Co-zero activation
  reordering (VARA, Dang, He & Li, arXiv 2609.00421, 2026) permutes activation dimensions and the
  matching weights together so that crossbars receiving all-zero inputs are skipped. On GPUs,
  Sylos Labini et al. and SMaT (arXiv 2408.11551) permute sparse-matrix rows to minimise the
  number of nonzero tensor-core blocks. The objective "permute so that whole blocks are zero"
  is therefore not new either. §2.8's remaining claim is the shared-operand, `k`-slice version,
  applied as an exact DNN rewrite under cross-layer constraints, on a digital systolic array.
- **Tile granularity.** SPOTS argues that small input blocks make all-zero blocks more likely.
  Yang et al. (*Activation Concentration*, arXiv 2606.00567, 2026) measure column-level against
  element-level sparsity on diffusion models, with up to a 78-point gap, at a *fixed* 16×16 array.
  They model the column-zero probability as pᴹ without testing it against measurement. A KIT
  design skips 8×8 input blocks, after noting that whole sparse blocks are rare in unmodified CNNs.
  None sweeps tile or array size against skip saving, or sets the result against where runtime
  selection pays. **§2.7 is still not found after three passes.** §2.5's measured-against-i.i.d.
  comparison is close in spirit to Yang et al. and should cite them.
- **Sparsity-driven array shape.** VUSA (NXP / University of Bremen, MOCAST 2025) virtually enlarges a
  systolic array at runtime when sparsity lets it fit a larger multiplication. LizTail (Liu et
  al., arXiv 2610.16159, Oct 2026) reconfigures 128×4 array segments to fit GEMM geometry, with
  no sparsity term in the abstract. S2TA (HPCA 2022) exploits structured weight and activation
  sparsity together, with no shape change. None chooses the orientation per layer by which
  operand holds the zeros. **§2.9's per-layer, bidirectional choice is still not found.**

**Revised novelty, in one line each (after three passes):**
- §2.7 window — *likely new*; not found in three passes.
- §3–§5 — mechanism is prior art (MAERI, Flexagon, Misam); the thresholds and the
  constant-matches-MLP scoping are the contribution, positioned against Misam.
- §2.8 regrouping — *narrowed, then partly restored after reading ISQED 2026 in full.*
  Reordering to make whole blocks zero is prior art in other hardware (GMR, VARA, SMaT), and exact
  permutation machinery is Pool & Yu's. ISQED 2026 permutes the reduction dimension inside a GEMM
  to trim the skew fill — a different axis and objective. What §2.8 claims: an output-channel
  rewrite under verified cross-layer constraints that creates whole skippable k-steps on a
  shared-operand array, at zero hardware, and its interaction with tile height and shape. Still a
  supporting compiler step, positioned against ISQED 2026 first.
- §2.9 tile shape — *mechanism not new* (credit SPOTS); the *per-layer, bidirectional,
  structure-driven choice and its quantification* are still new after three passes.

---

### 2.11 Two more model families (B9)

> **New in Rev 18.** `experiments/generalisation_b9.py` over traces from
> `profiler/profile_resnet.py --arch mobilenet_v2` and `profiler/profile_opt.py`: pretrained
> weights, the same 200 Imagenette images and SST-2 sentences as ResNet-50 and BERT, dense and
> 50/70/90% magnitude-pruned. The new traces are kept in `profiling_data/b9/`, so §3–§6 are
> still computed on the original 15 sets; the ResNet-50 / BERT rows below are recomputed the
> same way and reproduce `scheduler_variance.csv` exactly.

Two families is thin for a workload claim. The two added are chosen to differ from the
existing ones in the way that matters here: **MobileNetV2** (depthwise-separable CNN, ReLU6,
inverted residuals — 17 of its 53 layers are depthwise) and **OPT-125m** (a decoder-only
Transformer whose feed-forward block uses ReLU, where BERT uses GELU, and causal attention).

| Family | Activation k-slices vs i.i.d. | Realisable skip saving (§2.6) | Selection speedup (§3), mean / max | *as published* | Input-dependent sets |
|:--|--:|--:|--:|--:|--:|
| ResNet-50 | 13.6–41.1× | 18.2–29.3% | 1.144× / 1.275× | 1.190× / 1.304× | 4 / 5 |
| **MobileNetV2** | **12.6–182.5×** | 9.4–20.7% (excl. depthwise) | **1.123× / 1.200×** | 1.147× / 1.236× | 2 / 4 |
| BERT-base | — (no exact zeros) | 0.0–23.7% | 1.000× / 1.000× | 1.000× / 1.000× | 0 / 5 |
| **OPT-125m** | 1.8–7.0× | **26.6–54.5%** | **1.000× / 1.000×** | 1.006× / 1.024× | 0 / 4 |

*(Selection speedups under the adopted MAC-bounded model, §3.0; the structure columns do not use
the cost model.)*

**What generalises.**

- **The CNN/Transformer split of §3 holds on both new families — and so does §3.0's narrower
  form of it.** Per-layer selection pays on the *pruned* CNN (MobileNetV2 1.10–1.20×, two sets
  input-dependent; dense MobileNetV2 1.000×, like dense ResNet-50) and not at all on the
  Transformer (OPT 1.000× on every set) — the same pattern as ResNet-50 vs BERT, now on four
  families instead of two.
- **ReLU zeros are k-structured on the second CNN too** — 12.6–182× the i.i.d. prediction
  (the top of that range is a small i.i.d. denominator at 30% density; the pruned90 figure,
  12.6×, is the comparable one to ResNet's 13.6–28.9×).
- **Where the zeros are, not how many, still decides.** OPT's activations are only 15% zero on
  average, yet its realisable saving (27.1%) is higher than dense ResNet-50's (19.3%):
  every zero is at the `fc2` input, which is 92.7% zero after ReLU, and 78.8% of its 16-token
  k-slices are entirely zero — a hidden unit that is silent for a whole tile of tokens. That
  is only 1.8× the i.i.d. prediction (at 93% density i.i.d. already gives 0.44), so OPT's
  structure is mostly density at one layer type, not clustering; its value is that the
  densest zeros sit on the largest-K layer. BERT's GELU gives none of this (§2.5).

**What is new, and stated separately.** MobileNetV2's depthwise layers run on a GEMM array as
one GEMM against a block-diagonal weight (`workload_profiler._operand_b`, checked to reproduce
the convolution exactly), so they do C times their real work and almost all of it is
structurally zero along whole k-slices. Zero-skipping recovers it: including them, the
realisable saving is 89–92%. That is a statement about running depthwise convolutions on a
skip-capable GEMM array, not about activation structure, so the table quotes the saving
without them (9.4–20.7%) as the comparable number.

**Scope.** The new families are not added to the §3–§6 headline set: doing so would change
every number in those sections, and choosing the headline set is a paper-level decision. As
it stands they test whether the findings transfer, and they do. OPT's attention sparsity
(0.62–0.99 over the whole matrix, 0.26–0.98 over the causally visible part) is recorded in the
traces but no claim rests on it.

## 3. Result 1 — Adaptive scheduling at 16x16

### 3.0 The cost model, corrected (Rev 19) — read this first

> **New in Rev 19.** Every number in §3–§6 is now computed with the **MAC-bounded model (B)**;
> the published model's value follows in parentheses or in an "as published" column, and its
> result files are kept in `results/as_published/`. `experiments/cost_model_variants.py`
> produces the comparison below.

**The defect.** VersaAccel's published formulas for two of its four configurations charge less
compute than the array can physically do on real layer shapes. D-MR's dense MM is
`ceil(P/Nc)·(Nr + M + Nc − 1)` — **independent of K** — and D-M-R's is
`ceil(M/Nr)·ceil(K/Nc)·ceil(P/Nc)`, which divides by `Nc` twice. On a 196×256×256 layer at
16×16 they charge 3,632 and 3,360 cycles where 256 PEs need at least `M·K·P/256` = 50,176 —
**14× below the MAC lower bound**. Across the 31 distinct real ResNet-50 / BERT layer shapes,
D-MR is below the bound on all 31 (median 31×, up to 176×) and D-M-R on 29 (median 16× — the
missing factor of `Nc`). Their Table II check
(§2.1) is at 2×2, where the error is invisible. Because these were the two configurations the
scheduler preferred (92% of layer decisions), the §3 headline rested on them.

**Three models.**

| | Compute term | Role |
|:--|:--|:--|
| **A** "nnz" | as published | Rev 18 and earlier; still used where the published model is the subject (Table II validation, §2.4, §5.3b, §7.12) |
| **B** "mac" | A, with every configuration held to the MAC lower bound of the work it claims: `M·K·P` dense, `M·K·P·(1−w)` for a sparse configuration on SpMM (`·(1−a)` more on SpMSpM), over `Nr·Nc` PEs | **adopted for §3–§6**: the smallest correction that makes A physical, keeping its Gustavson sparse configurations |
| **D** "rtl" | the RTL-measured terms: DMR's tile fill (B11), and the sparse configurations charged MAC-bounded dense compute credited only realisable k-slice skipping `f·K/(K+NR+NC)` (B6) | the sparse configurations costed **as if run on this RTL's shared-operand array** — a finding, not the paper's model |

**The headline under each** (§3.2's measure, 200 passes per workload):

| | A (published) | **B (adopted)** | D (on this RTL) |
|:--|--:|--:|--:|
| all 15 workloads | 1.184× | **1.052×** | 1.000× |
| heterogeneous (10) | 1.275× | **1.079×** | 1.000× |
| 10 real workloads | 1.095× | **1.072×** | 1.000× |
| pruned / 2:4 ResNet-50 (4) | 1.16–1.30× | **1.11–1.28×** | 1.000× |
| dense ResNet-50 | 1.121× | **1.000×** | 1.000× |
| synthetic stress sets (5) | 1.12–1.72× | **1.000–1.056×** | 1.000× |
| workloads with > 1 optimal schedule | 6/15 | **5/15** | 0/15 |

**What the correction does.**

1. **Almost all of the published headline was the defect.** The synthetic stress traces lose
   their gain entirely (`gcn_synthetic` 1.72× → 1.003×), and dense ResNet goes from 1.12× to
   exactly 1.000× — those "wins" were the scheduler moving layers onto a configuration charged
   below what it can do.
2. **What survives is exactly the pruned CNNs** — 1.11–1.28× on the four pruned / 2:4 ResNet
   sets, and 1.12× on MobileNetV2 (§2.11) — and it is still input-dependent on the same
   workloads (§3.3). That is a smaller claim than 1.184×, and a defensible one.
3. **Under D, selection buys nothing.** Once every configuration pays a physical dense cost and
   is credited only the zeros a shared-operand array can skip, D-MR is the best configuration
   for every layer of every workload (100% of decisions). This is §5.3b's finding — structure
   changes how much is saved, never which configuration is chosen — carried to its end: on an
   array like the one built here, the dataflow choice is static, and what pays is the tile
   geometry (§2.7–§2.9) and the compile-time rewrite (§2.8), not runtime switching.

**What does not move.** The structure results (§2.4–§2.9) do not use these formulas. Nor do
§5.1's decision-sensitivity fractions, §5.2's predictor comparison or §5.3 — all reproduce
under B to the printed digit (§5). §6.1 and §8.1 move and are restated there.

### 3.1 Per-workload

| Model | Static config | Static gap | Greedy gap | DP gap | Speedup | O.switches | as published |
|:--|:--|--:|--:|--:|--:|--:|--:|
| bert_dense | DMR | +0.0% | +0.00% | +0.00% | 1.000x | 0 | 1.000x |
| bert_nm24 / pruned50 / 70 / 90 | D-M-R | +0.0% | +0.00% | +0.00% | 1.000x | 0 | 1.000x |
| resnet50_dense | DMR | +0.0% | +0.00% | +0.00% | 1.000x | 0 | 1.121x |
| switching_stress | D-M-R | +0.0% | +0.00% | +0.00% | 1.000x | 2 | 1.370x |
| dp_stress | D-M-R | +0.0% | +0.11% | +0.00% | 1.000x | 2 | 1.120x |
| gcn_synthetic | DMR | +0.3% | +0.00% | +0.00% | 1.003x | 4 | 1.767x |
| mixed_pipeline | D-M-R | +0.9% | +0.01% | +0.00% | 1.009x | 4 | 1.390x |
| variable_random | D-M-R | +1.3% | +0.07% | +0.00% | 1.013x | 5 | 1.078x |
| resnet50_pruned50 | D-M-R | +13.1% | +0.01% | +0.00% | 1.131x | 11 | 1.194x |
| resnet50_pruned90 | D-M-R | +15.8% | +0.00% | +0.00% | 1.158x | 6 | 1.171x |
| resnet50_nm24 | D-M-R | +18.1% | +0.00% | +0.00% | 1.181x | 14 | 1.193x |
| resnet50_pruned70 | D-M-R | +23.9% | **+2.32%** | **+2.32%** | 1.239x | 10 | 1.270x |
| **Average** | | **+4.9%** | +0.17% | +0.15% | **1.049x** | 58 | 1.178x |

> **Read the two gap columns carefully — they are not what they look like.**
> `dp_scheduler` builds the Greedy and DP schedules from the **trained predictor's** sparsity
> estimates and compares them against an oracle that sees the true values. The gap therefore
> measures **predictor error propagated into scheduling cost**, not how far greedy sits from
> optimal. The predictor-free scheduling results — §3.2, §3.3 and the whole of §4 — are
> unaffected, and §4 is where the scheduling-policy question is actually answered.

`resnet50_pruned70` is the one real workload where prediction error still costs something
(+2.32%; +4.97% as published), and it is also the workload with the most schedule variation
(§3.3). Those are the same fact.

Every workload with a gain above 1.02× is a pruned or 2:4 ResNet. The average static gap —
what picking the right configuration per layer is worth — falls from +17.8% to **+4.9%**, all
of it on those four sets.

### 3.2 With error bars, over all 200 passes

`experiments/scheduler_variance.py`:

> **Mean speedup: 1.052x +/- 0.044 (95% CI)** across 15 workloads, range 1.000x–1.657x
> *(as published: 1.184x ± 0.101, up to 2.987x)*
> **Heterogeneous (n=10): 1.079x** *(1.275x)*; **pruned / 2:4 ResNet-50: 1.11–1.28x**
> **Homogeneous (n=5): 1.000x** — one config is optimal for the whole network

**Quote the split, never the blended average — and quote the pruned-CNN figure.** All five BERT
variants sit at exactly 1.000x with zero switches, and so, now, does dense ResNet-50; the five
synthetic stress traces fall to 1.000–1.056x. Adaptation is worth something on pruned CNNs and essentially
nothing elsewhere. (The 1.657× maximum is a single pass of `variable_random`, whose mean is
1.056×.)

### 3.3 Schedule stability — the result that most constrains the claim

Of 15 workloads, **10 have exactly one optimal schedule across all 200 inputs.** Five vary
(six as published — `gcn_synthetic` no longer does):

| Workload | Distinct schedules over 200 passes | |
|:--|--:|:--|
| resnet50_nm24 | 2 | **new — was 1 on noise** |
| resnet50_pruned50 | 2 | **new** |
| resnet50_pruned70 | 5 | **new** |
| resnet50_pruned90 | 3 | **new** |
| variable_random | **171** *(196 as published)* | |

**Every real entry is a pruned ResNet variant**, i.e. exactly the traces that became real in
Revision 9, and exactly the workloads that keep a speedup under B. This is the clearest single
piece of evidence for the runtime contribution, and the old traces had suppressed it: Gaussian
noise produces almost identical activation sparsity on every pass, so the optimal schedule
could not move.

So even at 16x16, a *runtime* scheduler is required on a minority of workloads; the rest
could be scheduled once at compile time. **Prediction matters exactly where schedules vary,
and nowhere else** — and on real data that is the four pruned ResNet sets.

### 3.4 Against prior art's form of the comparison (B4)

The only external comparison so far was the VersaAccel paper's own Table II. The closest
published experiment is Flexagon's (Muñoz-Martínez et al., ASPLOS 2023): per-layer dataflow
selection against accelerators **fixed to one dataflow for every layer**, reporting **4.59×** over
SIGMA-like (inner product), **1.71×** over SpArch-like (outer product) and **1.35×** over GAMMA-like
(Gustavson). `experiments/prior_art_comparison.py` puts this project's numbers in that form, on
the same cost model and traces as §3 (oracle schedule with switch costs; 20 passes per workload),
all 15 workloads:

| Per-layer selection over… | **B (adopted)** | as published |
|:--|--:|--:|
| fixed DMR (dense systolic) everywhere | 1.99× | 7.15× |
| fixed D-MR everywhere | 7.86× | 19.05× |
| fixed DM-R everywhere | 5.51× | 17.27× |
| **fixed D-M-R (Gustavson-style) everywhere — the strongest single dataflow** | **1.05×** | 1.21× |
| best fixed configuration *per workload* (§3's baseline) | 1.05× | 1.18× |

**Two things follow.**

1. **Against the strongest fixed dataflow, selection gains 1.05× here and 1.35× in Flexagon.**
   Under the published model the two looked comparable (1.21×); corrected, this project's
   figure is well below Flexagon's. The configurations, fabric and workloads differ (below), so
   this is a placement, not a contradiction — but the paper should state it this way.
2. **"× over a fixed dataflow" mostly measures the choice of baseline.** The same selection
   policy is 1.05× or 7.9× depending on which dataflow is held fixed, the way Flexagon's own
   figures span 1.35×–4.59×. §3's per-workload-best baseline is stricter than any of them.

**Limits.** The configurations are not the same dataflows as Flexagon's, the fabric differs
(16×16 output-stationary vs a 64-multiplier MAERI-style SpMSpM engine), and VersaAccel's sparse
configurations are analytical only (§7.9) — so this places the *experiment*, not the hardware.

### 3.5 The headline over the whole operating-point grid (B8)

§3.2's figure is one point — 16×16, 32 elements/cycle (the default memory model), modelled
switch costs. §4.2 and §5.1 each vary one axis alone. `experiments/operating_point_grid.py` varies
all three together (5 passes per workload):

| Real workloads (10), switch cost ×1 | BW = 2 | BW = 4 | BW = 8 | BW = 32 |
|:--|--:|--:|--:|--:|
| 8×8 | 1.005× | 1.000× | 1.000× | 1.000× |
| 16×16 | 1.060× | 1.037× | 1.051× | **1.071×** |
| 32×32 | 1.010× | 1.029× | 1.047× | **1.088×** |

| All 15 workloads, switch cost ×1 | BW = 2 | BW = 4 | BW = 8 | BW = 32 |
|:--|--:|--:|--:|--:|
| 8×8 | 1.004× | 1.000× | 1.000× | 1.000× |
| 16×16 | 1.045× | 1.028× | 1.038× | **1.052×** |
| 32×32 | 1.011× | 1.025× | 1.040× | 1.072× |

*(As published, the corresponding cells were 1.094× / 1.122× real and 1.183× / 1.685× (8×8,
all 15) — `results/as_published/operating_point_grid.csv`.)* Raising switch cost ×10 changes
every cell by under 0.5%; ×100 by at most 2.5% (16×16 real, BW 32: 1.071× → 1.062×).

**Three things the single point hid.**

1. **8×8 gains nothing at any bandwidth** (1.000–1.005×) — the §5.1 scoping, now in speedup
   rather than decision counts, and now on *all* workloads: the large 8×8 figures the published
   model gave the synthetic traces (up to 1.685×) were the formula defect.
2. **On real workloads the gain grows with array size and bandwidth** — 1.088× at 32×32 /
   BW 32, the best cell — consistent with §2.7's window from the selection side.
3. **The headline is not fragile to switch cost**, which is the one constant B2 will replace
   with a synthesis measurement: even a 100× error moves it by at most 2.5%.

---

## 4. Result 2 — Greedy suffices, and here is the threshold

> **Rev 19.** MAC-bounded model (§3.0); the published model's values in parentheses.

### 4.1 Window ablation

| Window | Avg gap vs Oracle |
|:--:|--:|
| **W=1** | **+0.0007%** *(+0.0001%)* |
| W=2, W=3, W=5 | +0.0000% |

A switch-cost-aware **greedy** rule matches the global optimum. The original claim that
lookahead was needed came from a hardcoded print statement, not from this table — the script
now computes its conclusion.

### 4.2 The sensitivity sweep

`experiments/switch_cost_sweep.py`, scaling reconfiguration cost over four orders of
magnitude at 16x16:

| Switch cost | as % of layer | Static gap | Greedy gap | W=1 gap | W=5 gap |
|--:|--:|--:|--:|--:|--:|
| 250 (**as modelled**) | 0.7% *(1.2%)* | 4.9% | **0.013%** *(0.003%)* | 0.001% | 0.000% |
| 1,000 | 2.7% | 4.9% | 0.061% | 0.003% | 0.000% |
| 10,000 | 26.5% | 4.5% | 0.805% | 0.099% | 0.000% |
| 25,000 | 66.3% | 4.2% | 2.216% | 0.113% | 0.000% |
| **100,000** | **265.2%** | 3.5% | 9.630% | **2.527%** | **0.895%** |
| 1,000,000 | 2652% | 3.4% | 94.64% | 49.78% | 35.19% |

> **Design rule.** At 16x16, lookahead scheduling is justified only when reconfiguration
> exceeds **~265% of layer latency** — about **400x** the modelled cost *(as published: ~121%,
> 100x)*. Below that, W=1 is within 1% of optimal, and at realistic costs greedy is within
> 0.013%.

The threshold moves because the correction makes layers slower (every configuration now pays
at least its MAC count), so a fixed reconfiguration cost is a smaller fraction of a layer; the
rule is properly stated in that unit — switch cost as a fraction of layer latency — and in it
the conclusion is unchanged: lookahead never pays at any plausible switch cost. The published
model's array-size dependence (~440% at 8x8, ~121% at 16x16) was measured on the uncorrected
formulas and is not re-derived here.

The bottom rows show *why* lookahead exists at all: in the high-cost regime W=1 destabilises
(49.8% gap) because it locks into a configuration it can then never afford to leave, while
W=5 stays lower. **Lookahead's value is robustness, not incremental optimality.**

### 4.3 The switch cost, measured on the RTL (B2, Rev 21)

The 100–250 cycles above were an estimate for an 8×8 array — drain, interconnect reset, control
word load, refill. `tb_switch.sv` measures what switching actually costs `versaaccel_top`, by
running two layers (16×16×32 GEMMs, skipping on) back to back for all 16 (from, to)
configuration pairs and charging the pair the time beyond the average of the two pure pairs,
`T(a→b) − (T(a→a) + T(b→b)) / 2`, so each configuration's own compute time cancels:

| Measurement | Cycles |
|:--|--:|
| Forced switch, every one of the 16 pairs | **0** |
| CPE-predicted switch: `start` → prediction valid → speculation unit → selected config | 7 → 8 → **9** |
| Operand load the prediction overlaps (M·K + K·P) | 768 |

A forced switch is free because nothing is left to reconfigure: the configuration and the Benes
routing are registers captured on `start`, and the array is drained at the end of every GEMM
whether or not the next one switches — the drain the estimate charged to the switch is paid by
every layer. A predicted switch is in force 9 cycles after `start`, hidden under the operand
load; 9 cycles is the most a switch can cost, when nothing overlaps it (16,384 output checks
across the 33 GEMMs, 0 errors). Scope: in GEMM mode the RTL runs one dataflow (§7.9), so this is
the cost of the reconfiguration *control*; a datapath that changed dataflow would also need the
array empty, which this one already is at every layer boundary.

Taking 9 cycles for every off-diagonal pair (`VERSAACCEL_SWITCH_COST=rtl`, results in
`*_rtlswitch.csv`):

| | estimate (250 max) | **measured (9)** |
|:--|--:|--:|
| switch as % of a median layer | 0.7% | **0.02%** |
| greedy gap to optimal | 0.013% | **0.000%** |
| W=1 / W=5 gap | 0.001% / 0.000% | 0.000% / 0.000% |
| §3.2 headline, mean over 15 workloads | 1.052× | **1.053×** (1.0525 → 1.0526) |
| workloads whose optimal schedule varies across inputs | 5 / 15 | 5 / 15 |

So the measured switch is ~28× cheaper than the estimate's largest entry and ~11,000× below the
lookahead threshold (~265% of a layer, ≈100,000 cycles, §4.2). It changes no conclusion and
sharpens one: on this RTL greedy is not merely within 0.013% of optimal but exactly optimal, and
the design rule's margin is four orders of magnitude rather than two and a half. The §3–§6
numbers are kept on the estimate, which they reproduce; under the measured cost they agree to
the third decimal.

---

## 5. Result 3 — Prediction is justified only at 16x16 and above

This is the scoping claim, and the reason the project now targets 16x16.

> **Rev 19 (MAC-bounded model, §3.0).** Every number in §5.1–§5.3 reproduces under the adopted
> model to the printed digit — the decision-sensitivity fractions, the predictor and baseline
> match rates, the 10-seed results — because the layers whose choice depends on activation
> sparsity are decided between configurations the correction leaves in the same order. Only
> §5.1's "configs ever optimal" column changes (in this sweep D-MR is never optimal once held to the MAC
> bound; DMR takes its place). §5.3b is the published-model test it was run as, and §3.0 carries
> its conclusion through to model D.

### 5.1 The premise fails below 16x16

`experiments/array_size_sweep.py` counts how many (layer shape, weight sparsity)
combinations change their optimal config as activation sparsity sweeps 0.0 -> 0.95:

| Array | BW >= 4 elem/cyc | BW = 2 elem/cyc | Configs ever optimal |
|:--:|--:|--:|:--|
| 4x4 | **0.0%** | 25.0% | DMR, D-M-R (+DM-R when starved) |
| 8x8 | **0.0%** | 37.5% | DMR, D-M-R (+DM-R when starved) |
| 16x16 | 20.8% | 37.5% | DMR, D-M-R (+DM-R when starved) |
| 32x32 | 33.3% | **41.7%** | DMR, D-M-R (+DM-R when starved) |

At 8x8 and realistic bandwidth one dataflow dominates the entire sparsity range, so the
argmin never moves and a **constant predictor scores a perfect config match**. The original
99.2% figure was measuring a saturated metric. Runtime prediction cannot help at that size;
configuration selection reduces to a compile-time assignment from weight sparsity and layer
shape.

**The bandwidth qualifier is new in Revision 8 and it matters.** Under the corrected
nonzero-row estimator (§2.4) a *bandwidth-starved* array is sensitive at every size tested,
4x4 included: when the array is starved, `DM-R` and `DMR` become competitive and the argmin
moves for reasons that have nothing to do with size. So the scoping claim is properly stated
as **"at BW >= 4 elem/cyc, prediction is justified only at 16x16 and above"** — not as a
statement about array size alone. Under the old estimator this second regime was invisible.

### 5.2 The discriminating metric

A match rate over all layers is dominated by layers the predictor cannot affect. Restricting
to the decision-relevant subset (1,800 of 25,000 layers, 7.2%, at 16x16) is what makes it
discriminate. On the **real** ResNet traces it discriminates against the predictor:

| Predictor | Params | Test MAE | match% (all) | **match% (relevant)** |
|:--|--:|--:|--:|--:|
| constant-zero | 0 | 0.512 | 94.7% | **26.1%** |
| constant-mean (0.401) | 0 | 0.295 | 99.7% | **96.2%** |
| persistence | 0 | 0.343 | 98.6% | **80.6%** |
| **trained MLP** | 8,961 | **0.260** | 99.7% | **96.2%** |
| oracle | 0 | 0.000 | 100.0% | 100.0% |

**The MLP ties a zero-parameter constant, 96.2% to 96.2%.** On fully real traces it barely
even wins on its own regression metric: MAE 0.260 against the constant's 0.295, an 12%
improvement for 8,961 parameters. (On the partly-synthetic traces of Revision 9 the MAE gap
looked like 4.3x — 0.075 against 0.323. Real BERT attention sparsity makes the regression
genuinely hard, and most of the apparent advantage was the easy synthetic half.)

This is the same pattern the audit has found repeatedly, now applied to this project's own
contribution: *a metric improvement that is not a decision improvement.*

**The mechanism, because it is not luck.** `SPARSE_THRESHOLD` is 0.3 and `select_operator`
tests `activation_sparsity > SPARSE_THRESHOLD`. The constant predicts the training mean,
0.401, which passes that test — so it always votes "activations are sparse", and the operator
is then chosen as if they were. That is the correct call on 96.2% of the layers where
activation sparsity *could* change the config. *(Corrected in Rev 14: earlier text gave the
mean as 0.300 and the vote as "dense", left over from a revision whose mean sat just below the
threshold.)*

It is not that the distribution is narrow. The test targets have standard deviation 0.353 and
are close to bimodal — 51.6% sit above the threshold, and only 6.3% lie within ±0.10 of the
mean. The constant wins because the decision boundary happens to sit where it does, not
because the values cluster.

**Two things this does and does not mean.**

- It does **not** retract §5.1. Activation sparsity genuinely changes the optimal config, on
  7.2% of layers — more than double the 3.2% measured on synthetic traces. The premise of
  runtime adaptation holds and is stronger than before.
- It **does** retract the case for an MLP. On these workloads, a constant at the threshold
  captures the same decisions for 8,961 fewer parameters. If the predictor is to be defended,
  it needs a workload where the activation term actually crosses the boundary — and this
  project has not yet produced one.

**The caveat Revision 9 raised did not survive contact.** That revision warned that the
synthetic BERT half might be inflating every match rate, and called this the result most
likely to move. With BERT now real (§2.3) the config-match column is **identical to the
digit** — 26.1 / 96.2 / 80.6 / 96.2. The finding is robust to the thing most likely to break
it, which is worth more than the original caveat was. **This is the number the paper should report**, with the "all" column shown alongside
to explain why the original metric saturated.

Test MAE is 0.260 on the real traces (0.0750 on the partly-synthetic Rev 9 traces, 0.0626 on
fully synthetic ones). *(Corrected in Rev 14: this line still gave the Rev 9 figure.)*
The predictor used to select its checkpoint by minimum *test* MAE with no validation set. It
now early-stops on 40 held-out inference passes from the training models, and the test set is
touched exactly once.

*(Scoring the old checkpoint against the new traces gave MAE 0.1298 and 95.9% — worse than
the constant. That is stale-model drift, not a finding, and the table above is the retrained
model. Worth recording only because it is an easy mistake to make when traces change.)*

**Seed robustness (A8, Rev 14) — the tie was the best case.** The MLP in the table is one
trained network, and `train_predictor.py` never seeds torch. `experiments/predictor_seed_robustness.py`
retrains it ten times — same data, same split, same architecture, validation hold-out and early
stopping — and scores every seed on the same decision-relevant metric:

| | match% (relevant) | correct of 1,800 | test MAE |
|:--|--:|--:|--:|
| constant-mean (0.401) | 96.2% | 1,731 | 0.295 |
| committed MLP (the table above) | 96.2% | 1,732 | 0.260 |
| **MLP, 10 seeds** | **93.4% ± 5.6** (78.1 – 96.2) | 1,405 – 1,731 | 0.241 – 0.285 |

**No seed beats the constant.** Eight are worse, two tie, and on average the MLP gets 50 fewer
decisions right; the worst seed gets 326 fewer. The committed model's 96.2% sits at the very
top of the distribution. Seed 0 is §5.2's lesson in its most extreme form: *better* MAE than the
constant (0.263 vs 0.295) and 18 points *worse* on decisions (78.1%).

So claim 5's withdrawal is stronger than §5.2 first stated. A learned predictor does not tie a
zero-parameter constant on these workloads; it typically loses to one, and whether it ties
depends on the draw.

### 5.3 Testing the rescue: was the MLP regressing the wrong variable?

> **Corrected in Rev 13.** Rev 12 reported that the rescue *failed* — the MLP tying a
> per-layer table on both targets, +1.1% and +4.8% — and that this closed the obvious objection
> to §5.2. That conclusion does not hold. It came from **one seed**, and from a run made before
> the two N:M trace sets carried structural columns (8 sets, not 10). Re-run over 10 seeds on all
> 10 real sets, the single-seed number turns out to sit inside a spread of roughly ±18 points.
> The corrected result is below. Both errors were in the analysis, not the data: the traces and
> §2.5–§2.6 reproduce exactly.

§5.2 retires the predictor but does not explain it. §11.7 proposed an explanation that is also
a testable prediction: the MLP's target is a **scalar density**, §7.7 says density is not what
determines whether the array can skip anything, and therefore an excellent density MAE buying
zero decisions is exactly what should happen. If that is right, a predictor aimed at
**structure** — the quantity §2.5 now measures — should find per-input signal that a density
predictor does not.

`experiments/structure_predictor.py` runs the identical comparison on both targets: same
features, same MLP, same split, 18,750 held-out points per seed. Splits are **by input index**,
never by row, because two layers of one image share almost everything and a row-wise split
leaks (§5.4). Each of 10 seeds redraws both the split and the MLP initialisation.

| Target | MLP MAE | global constant | persistence | **per-layer constant** | MLP vs per-layer (10 seeds) |
|:--|--:|--:|--:|--:|:--|
| density (§5.2) | 0.0367 | 0.2878 | 0.2865 | **0.0429** | +14.3% ± 17.8, range −11.3 .. +32.9 |
| structure (§2.5) | 0.0098 | 0.1306 | 0.0814 | **0.0114** | **+14.3% ± 6.8, range +3.9 .. +26.7** |

The per-layer constant is the null that matters: a compile-time lookup table, one number per
layer, zero runtime parameters. Against a *global* constant the MLP looks excellent on both
targets (87% and 93% lower MAE); against the per-layer table the two targets separate.

- **Density: not distinguishable from the table.** The seeds straddle zero. Whatever the MLP
  learns about density beyond "which layer is this" does not survive a change of seed.
- **Structure: a consistent edge.** Every one of the 10 seeds beats the per-layer table. So the
  MLP *does* learn something about the individual input's structure that a table cannot hold —
  which is the direction Thesis 1 predicts: structure carries per-input signal that density
  does not.

**But the edge is tiny in absolute terms.** The per-layer table already reaches MAE 0.0114; the
MLP's 0.0098 improves on it by **0.0016 in skippable fraction**. Since realisable saving is
≈ `f` at these K (§2.6), that is about **0.16 percentage points** of predicted cycle saving. On
ResNet, where structure varies with the input most, a per-layer table leaves residual MAE 0.0269
against an across-input standard deviation of 0.0338 — so most of the input variation is
already captured by knowing the layer.

**Where this leaves claim 5.** Claim 5 stays withdrawn, because §5.2's evidence is at the level
that matters — decisions — and it is unaffected. What changes is the status of the rescue:
**it is open, not refuted.** Rev 12's "the obvious objection is now closed" is retracted. A
structure predictor does beat a table, consistently, by a margin too small to plausibly flip a
configuration choice — but "plausibly" is not a measurement. The test that would settle it is
decision-level: give the cost model a structure-aware term (B6), then compare a structure
predictor against a per-layer table on decision-relevant config match, exactly as §5.2 does for
density.

**Methodological note.** §5.2's finding went unnoticed for nine revisions because the predictor
was compared against a null that was too weak. Rev 12 then made a second, related error here:
it drew a conclusion from **one seed** of a comparison whose seed-to-seed spread is larger than
the effect being measured. The per-layer null is built into the script, and so now is the
repetition — `structure_predictor.py` runs 10 seeds and gives its verdict from whether the seeds
agree in sign, not from a threshold on a single draw. §5.2 itself rests on a single trained MLP
and should get the same treatment (A8).

### 5.3b The decision-level test, now run (B6) — the rescue is closed

`experiments/structure_decisions.py` gives the cost model a structure-aware term and runs the
test §5.3 asked for. The **structural model** keeps each configuration's own dense compute and
credits the skip-capable configurations only the saving their measured structure allows,
`f·K/(K+NR+NC)` (§2.6, verified cycle-exact on the RTL in §2.8 / B13); DMR stays dense, as
`CFG_DMR` does in the RTL. The memory term is taken either from the nnz model (compressed
operands) or dense (as the RTL loads them). 10 real trace sets, 20 passes each.

| | nnz model | structural, nnz memory | structural, dense memory |
|:--|--:|--:|--:|
| §3 headline, mean over the 10 real sets | 1.095× | 1.068× | 1.058× |
| resnet50_pruned70 | 1.306× | 1.138× | 1.116× |
| resnet50_pruned90 | 1.170× | 1.154× | 1.118× |
| BERT (all five) | 1.000× | 1.000× | 1.000× |
| per-layer decisions that differ from the nnz model | — | 66% of ResNet-pruned layers | same |

**And the decision test has nothing to decide.** Sweeping the skippable fraction f from 0 to 0.9
changes the optimal configuration at **0 of 125,000** (layer, input) points, under either memory
term. The reason is structural, not statistical: every skip-capable configuration realises the
*same* structural saving — they share one skip mechanism, as in the RTL — so f scales all of
them together and cannot reorder them, and DMR is never optimal on these workloads. Structure
changes **how much** the chosen configuration saves, never **which** one is chosen. A
structure predictor therefore cannot buy a decision, however good its MAE: the 0.0016 edge §5.3
measured is real and irrelevant.

**So §5.3's open question closes, in the direction §5.2 already pointed.** Claim 5 stays
withdrawn with no remaining escape route. The scope condition is stated honestly: this holds
because the configurations share one skip mechanism; dataflows that could each skip on a
different operand would make structure a decision variable, and that would need RTL for
genuinely distinct dataflows (§7.9).

**What the structural model does change** is the headline's magnitude on pruned ResNet — the
nnz term's over-credit (§2.6) inflated the gain from adapting, most on `resnet50_pruned70`
(1.31× → 1.12–1.14×). Whether to adopt the structural term as the paper's cost model joins B11
as a decision; both are now one flag each.

### 5.4 The same leak in Predictor V2 — a much larger correction

`train_predictor_v2.py` carried the identical defect and has been fixed the same way.

| | Previously reported | Honest |
|:--|--:|--:|
| V2 test MAE | ~0.0506 | **0.1285** |

The gap here is far larger than V1's because V2's test set includes `switching_stress`, a
genuinely different distribution — exactly the case where selecting the checkpoint on test
flatters most. **Any V2 number in the original draft should be treated as unreliable** until
regenerated.

---

## 6. Result 4 — N:M sparsity: a bounded negative result

### 6.1 Standalone 5th configuration

At 16x16 the N:M configuration is selected on 135 layers and makes the average **worse**:
1.108x with four configurations, 1.085x with five — a **−2.05%** change under the adopted
MAC-bounded model (§3.0; as published: 53 layers, 1.247x → 1.228x, −1.51%). It is anchored to
`DMR`, so `NM ~= 0.5 x DMR` while D-M-R exploits sparsity fully. The damage concentrates where
N:M is selected most: `resnet50_nm24` falls from 1.562x to 1.221x once the fifth configuration
is available to be chosen (1.608x → 1.324x as published). The result holds on the real ResNet
traces and gets slightly stronger under the correction: N:M still costs rather than pays.

Under the previous estimator this came out at exactly +0.00%. The corrected one turns a null
result into a negative one, and §7.7 explains the mechanism from the hardware side: in a
shared-operand array a 2:4 pattern gives **no** skippable steps, because all NR rows would
have to drop the same `k`.

The scaling argument is correct and worth keeping:

- N:M scales as **O(M·K·P)** — processes every tile
- D-M-R scales as **O(M·(K+P))** — Gustavson skips zero rows
- Closed-form crossover: N:M wins only when **P <= 4**, i.e. matrix-vector

A constant-factor knob like bandwidth cannot close an asymptotic gap. Publish this as a clean
bounded negative result.

### 6.2 As a D-M-R enhancer: DROPPED (B7, Rev 18)

**The paper makes no claim about N:M as a D-M-R enhancer.** The "+25.0% improvement" was
`0.12 + 0.08 + 0.05` hardcoded, scaled by two factors that are 1.0 by construction, so it
always printed exactly 25.00% — the assumption restated as a result. Measuring it instead
would need an RTL D-M-R (Gustavson) datapath, which this RTL does not have (§7.9), and §7.7
already shows from the hardware side that a 2:4 pattern gives a shared-operand array nothing
to skip. With no measurement possible and the mechanism pointing the other way, the claim is
dropped rather than carried as "withdrawn, pending". The discount stays disabled in code
(§8.1) and is not used by any reported number.

What remains of N:M in the paper is §6.1 only: a bounded negative result for N:M as a
standalone fifth configuration. (`nm_compatibility_score()`, which counts a group compatible
when `nnz <= N` and so reads a 90%-pruned matrix as ~100% "2:4 compatible", survives only as
one of the CPE's nine input features; with the enhancer claim gone, nothing is concluded from
it.)

---

## 7. RTL status

**Previously:** no testbench had ever been simulated; two would not compile.

**Now: full suite passes at both 8x8 and 16x16, with arithmetic verified throughout** —
13,215 golden checks at 16x16. Run it with `idea2_veriflex/rtl/scripts/run_all_tb.sh`.

| Testbench | 16x16 | 8x8 | Checks |
|:--|:--|:--|:--|
| `tb_datapath_golden` (**new**) | **50/50** | 30/30 | Benes identity + permutation; adder tree FULL/PARTIAL/BYPASS |
| `tb_systolic_golden` (**new**) | **257/257** | 65/65 | Signed MAC, column skew, drain addressing |
| `tb_sigma_golden` (**new**) | **16/16** | 8/8 | Every dot product of the SIGMA baseline |
| `tb_versaaccel_golden` (**new**) | **65/65** | 33/33 | End-to-end: interface → buffers → Benes → PE array → reduction → dout, across 4 configs |
| `tb_gemm_golden` (**new**) | **3072/3072** | 512/512 | Tiled GEMM, every output element vs an independent reference, at both output-bus widths and under random backpressure |
| `tb_gemm_sparse` (**new**) | **7168/7168** | 1792/1792 | 7 sparsity patterns x skipping on/off, every output element vs an independent reference |
| `tb_cpe_golden` (**new**) | **64/64** | 64/64 | RTL argmax vs the quantised Python model, bit-exact |
| `tb_versaaccel` | 11/11 | 11/11 | Config overrides, switching, phase FSM |
| `tb_systolic` | 4/4 | 4/4 | Weight load, compute, drain |
| `tb_sigma` | 3/3 | 3/3 | Routing, dot-product, FAN reduction |

### 7.1 Golden-vector tests

The original benches check only control handshakes — that `busy` rises, that the FSM reaches
`S_COMPUTE`, that drain completes. **A design that multiplied by zero would pass all of
them.** Four new benches check arithmetic against independently computed golden values, with
stimulus containing negative operands and zeros so a dropped sign or a dead PE cannot hide.

*Scope, stated honestly:* for the two sequential benches the model's activation ramp-in
timing was calibrated against the DUT. What they prove is that **every accumulator agrees
with one consistent timing rule** — which a wrong multiply, dropped sign, inactive PE, broken
skew or mis-addressed drain could not satisfy simultaneously. They do not independently pin
pipeline-fill latency. `tb_datapath_golden` and `tb_cpe_golden` have no such caveat: the
modules under test are pure functions and the references are independent.

### 7.2 The CPE now runs a real trained network

Previously every weight was `8'sd1` and every bias `0`, making the argmax degenerate — the
"prediction engine" was a constant function. Its topology also did not match the trained
predictor, and nothing anywhere performed a `$readmemh`, so the exported `.npz` never reached
hardware.

`train_cpe_classifier.py` trains the network the RTL is actually shaped for, and the one the
hardware wants: a config prediction engine should emit a **config**, not a sparsity value
software must then convert. 9 features → 16 → 16 → argmax over 5 configs, labelled by the
cost model's optimal choice.

| | Accuracy |
|:--|--:|
| Float ceiling | 0.9982 |
| INT8, post-training quantisation | 0.8934 |
| **INT8 after quantisation-aware fine-tuning — what the RTL runs** | **0.9304** |
| Majority-class baseline (always D-M-R) | 0.7040 |

Two RTL changes were required to make INT8 viable at all:

- **Per-layer requantisation shift.** The RTL clamped raw accumulators — peaking near 20,000
  — straight to 127, saturating every activation and collapsing accuracy to 0.70. Shifts are
  calibrated on training data and exported alongside the weights.
- **Accumulators widened 16 → 24 bits.** The output layer can reach 16·127·127 = 258,190,
  well past the 32,767 a 16-bit signed accumulator holds. The original width could overflow
  on plausible weights.

### 7.3 The RTL never actually scaled past N=8

`tb_datapath_golden` immediately failed at N=16, exposing that two modules were hand-unrolled
for exactly eight ports despite reading as parameterised:

- **`distribution_network`** needs 2·log₂(N)−1 stages. At N=16 that is 7, but only 5 were
  generated, so `stage_wire[6]` and `[7]` were never driven and **every output read as X/0**.
- **`reduction_network`** ended in `s3 = s2[0] + s2[1]`, so at N=16 FULL reduction **silently
  summed only 8 of its 16 inputs**.

Both are now generated for any power-of-two N; the Benes construction reproduces the previous
hand-written N=8 pairing exactly (d = 4,2,1,2,4), and the 8x8 results are unchanged.

**This matters for the paper's framing.** Until this fix the RTL could not run at the 16x16
operating point the project is scoped to, so the 16x16 numbers were cost-model-only. They are
now backed by simulation.

### 7.4 The end-to-end test found undefined data in the dense-GEMM path

`tb_versaaccel_golden` exercises the assembled datapath across four
configurations with different activations, weights and accumulate counts. It
failed immediately — and not with a wrong number. **`CFG_DMR` and `CFG_D_MR`
returned X.**

Cause: `ibuf` is an SRAM-style array with no reset, and only `ibuf[0]` is ever
written, but the datapath read `dist_in[i] = ibuf[ibuf_rd_ptr + i]` across all
NR entries unconditionally. So **NR−1 undefined values entered the PE array on
every run.** In the BYPASS reduction modes (`DM-R`, `D-M-R`) those X's landed on
unused output ports and the design *appeared* to work; in FULL and PARTIAL
reduction they propagated through the adder tree, so **the DMR dense-GEMM
configuration — the paper's own baseline — produced undefined output.**

Fixed by gating `dist_in` on `ibuf_wr_ptr`, so only entries actually written can
drive the array. All four configurations now compute correctly.

This is exactly the failure mode handshake-only testing cannot catch: two of
four configurations were silently wrong, and the FSM was perfectly happy.

### 7.5 The tiling controller — `versaaccel_top` now runs a real GEMM

The end-to-end bench also exposed a structural limit: `ST_LOAD_WEIGHTS` left for
`ST_COMPUTE` on the **first** non-weight datum and `ibuf_rd_ptr` never advanced, so
`versaaccel_top` could only ever broadcast a **single scalar activation** — one of 16 array
rows. The arithmetic was correct, but it was a control/interconnect skeleton, not an
accelerator.

**A tiling controller now streams a full `A[M,K] × B[K,P]` tile by tile**, output-stationary:
PE(r,c) holds `C[tile_i·NR+r][tile_j·NC+c]`.

Because `pe_array` forwards `a` one cycle per column and `b` one cycle per row, the feeds are
skewed so PE(r,c) sees `A[row][k]` and `B[k][col]` on the same cycle:

```
a_in[r] at cycle t = A[tile_i*NR + r][t - r]
b_in[c] at cycle t = B[t - c][tile_j*NC + c]
```

Both then carry index `k = t − r − c`, which walks `0..K−1` as `t` advances. Out-of-range
feeds are zero, so pipeline fill and drain contribute nothing and a single `acc_clear` at
`t=0` suffices. Compute length per tile is `K + NR + NC − 2`.

Two further changes were required:

- **Only one PE column could reach the reduction network** (hardwired to column 0), leaving
  NC−1 of every NR×NC accumulators unobservable. The drain now sweeps `drain_col` across the
  tile.
- **Reduction is forced to BYPASS during GEMM.** An output-stationary GEMM wants each
  accumulator delivered individually; FULL and PARTIAL belong to the reduction-style
  operators, where a PE column is summed to one result. (Note the config table calls `DMR`
  "dense GEMM" while mapping it to FULL reduction — a pre-existing semantic inconsistency
  worth resolving in the paper.)

The controller is **additive**: with `gemm_en` low the original single-shot path runs
unchanged, so every prior test still passes.

`tb_gemm_golden` checks every element of C against an independent reference, across six
shapes including partial tiles and `K=1`:

| M | K | P | Tiles | Elements verified |
|--:|--:|--:|:--:|--:|
| 32 | 6 | 32 | 2×2 | 1024 |
| 17 | Reconfigurability costs little area inside the PE | **Confirmed at the slow corner (Rev 21, §7.15)** — PE array + Benes + reduction network = 1.13 mm², 1.20× the whole systolic array; the two networks are 0.6% of VersaAccel and the CPE 2.4%. VersaAccel's 1.98× total is the feed/schedule logic (21.7%), bitmaps (6.0%) and buffers (9.2%) a fixed array does not need |
| 16 | 1 | 16 | 1×1 | 256 |
| 8 | 12 | 40 | 1×3 | 320 |
| 33 | 3 | 17 | 3×2 | 561 |
| 5 | 7 | 9 | 1×1 | 45 |
| | | | **Total** | **2,546** |

Each shape is run three times — narrow bus, wide bus, and wide bus under random backpressure
(§7.6) — so 7,638 element comparisons in total, and the two bus widths must agree bit-for-bit.

Reproduce with `idea2_veriflex/rtl/scripts/run_gemm_shapes.sh`.

**What this does and does not establish.** The RTL now executes correct tiled matrix
multiplication with real operands at both 8×8 and 16×16 — it is a working accelerator
datapath, not a skeleton. As first built it did not establish *throughput*: the drain was
serial, one element per cycle, `NR·NC` cycles per tile against `K+NR+NC−2` of compute, so at
16x16 the array spent ~87% of its time draining. §7.6 fixes that. The sparse configurations
still do not skip zeros, which remains open.

### 7.6 Making the drain free: output double buffering and a wide bus

The tiling controller left the array idle while each tile was read out. Two changes remove
that, and they are independent:

**1. Output double buffer.** The array is output-stationary, so the thing that has to be
double buffered is the accumulator bank itself. A one-cycle capture state (`ST_GEMM_CAP`)
copies the tile's `NR×NC` accumulators into a shadow register bank; the array then starts the
next tile immediately while a **concurrent** drain process reads the shadow copy. Cost:
`NR·NC·ACC_W` flip-flops — 8,192 at 16x16 — the same order as the accumulators themselves,
and a real area line item to report once synthesis runs.

The capture is a separate cycle for a specific reason: `pe_out` is the accumulator
*register*, so the final product of compute cycle `K+NR+NC−3` only lands at the edge that
*ends* it. Capturing on that edge would drop the last multiply of every PE — including the
corner PE, for which it is the only cycle that carries `k = K−1`. Per-tile compute is
therefore `K + NR + NC − 1`.

Restarting the array with no flush bubble is safe, and this is the property that makes the
double buffer worth having: the `a` feed is zero for `t ≥ K+NR−1` (the last `NC−1` cycles)
and `b` for `t ≥ K+NC−1` (the last `NR−1` cycles), so the stale values still sitting in the
systolic forwarding registers when the next tile begins are exactly the ones that are zero.

**2. Wide output bus.** `drain_wide` presents a whole PE column on `dout_vec[NR]` each cycle
instead of one element on `dout_data`, so a tile retires in `NC` beats rather than `NR·NC`.
Since `NC ≤ K+NR+NC−1` always, the drain then fits entirely inside the next tile's compute
and disappears.

Per-tile steady-state cost, 16x16, K=6:

| | cycles/tile |
|:--|--:|
| serial compute-then-drain, narrow bus | `(K+NR+NC−2) + NR·NC` = 292 |
| double buffered, narrow bus | `max(K+NR+NC−1, NR·NC)` = 256 |
| double buffered, wide bus | `K+NR+NC−1` = **37** |

Measured end-to-end, `tb_gemm_golden` (cycles from `start` to the final output beat, operand
load included):

| M | K | P | Tiles | narrow | wide | gain |
|--:|--:|--:|:--:|--:|--:|--:|
| 32 | 6 | 32 | 2×2 | 1449 | 552 | 2.62x |
| 17 | 5 | 20 | 2×2 | 1249 | 349 | 3.58x |
| 16 | 1 | 16 | 1×1 | 324 | 84 | 3.86x |
| 8 | 12 | 40 | 1×3 | 1391 | 725 | 1.92x |
| 33 | 3 | 17 | 3×2 | 1724 | 374 | 4.61x |
| 5 | 7 | 9 | 1×1 | 396 | 156 | 2.54x |

*(Rev 20 RTL. Every entry is exactly 4 cycles more than Rev 6–19's — 1445/548, 1245/345, 320/80,
1387/721, 1720/370, 392/152 — the pipelined operand feed and output register, once per GEMM;
§7.14.)*

The wide numbers hit the analytic ideal exactly — for 32×6×32,
`384 (load) + 4×37 + 16 + 4 = 552` — which is the check that the drain really is free, not merely
cheaper.

**Both widths must agree bit-for-bit,** and a third run randomly deasserts `dout_ready` on
~25% of cycles to exercise the stall path, where the capture state has to hold and the
accumulators must stay frozen. That costs 9 cycles out of 552 (8 of 548 before Rev 20), because
the wide drain has 21 cycles of slack per tile. All three runs are in the pass count.

**What is left.** The remaining inefficiency is no longer the drain but the **skew fill**:
useful MAC-cycle utilisation is `K/(K+NR+NC−1)`, which is only 16% at K=6 but 67% at K=64 and
89% at K=256 — the shapes in the table are small ones chosen to exercise tiling, not to
flatter throughput. *(Rev 18: removing the fill cost turned out **not** to need a second
operand feed port — each feed still delivers one value per cycle; what changes is that the
accumulator clear and the capture become a per-PE wavefront. Built and measured in §7.13.)* Zero-skipping in the sparse configurations is
still absent, and that one matters more: it is the mechanism the sparse configurations are
supposed to win by.

Reproduce with `idea2_veriflex/rtl/scripts/run_gemm_shapes.sh`; full suite with
`idea2_veriflex/rtl/scripts/run_all_tb.sh`.

### 7.7 Zero-skipping: structure matters, count does not

The sparse configurations previously executed the same dense schedule as the dense ones, so
the RTL could not demonstrate the advantage the cost model attributes to sparsity. That is now
implemented — and measuring it produced the clearest negative result in this report.

**What a systolic array can skip.** PE(r,c) computes `Σ_k A[row_r][k]·B[k][col_c]`, and the
array *shares* each operand: one `a_in[r]` feeds every PE in row r, one `b_in[c]` every PE in
column c. So step `k` can be dropped only when it is useless for the **entire tile**:

```
mask[k] = (A[.][k] has a nonzero in this tile's NR rows)
        & (B[k][.] has a nonzero in this tile's NC columns)
```

An isolated zero saves nothing, because the cycle it would free is still needed by the other
NR−1 rows sharing that feed. Skipping *individual* elements requires a dataflow in which
operands are not shared — a different architecture, not a different control unit.

**Implementation.** Two nonzero bitmaps are built as the operands stream in — one OR per
element, so they cost **no cycles of their own**:

```
a_nz[ti][k] |= (A[ti*NR + r][k] != 0)
b_nz[k][tj] |= (B[k][tj*NC + c] != 0)
```

Feeding a skipped schedule needs no random access to it. `a_in[r]` at cycle t wants
`ksched[t−r]` and `b_in[c]` wants `ksched[t−c]` — the same sequence at different delays — so a
single shift register `khist[i] = ksched[t−i]` serves every port, and PE(r,c) still sees one
common index, now `ksched[t−r−c]` instead of `t−r−c`. A priority encoder over `mask` supplies
one step per cycle. The whole change to the datapath is that substitution; the tiling, the
double buffer and the drain are untouched. Cost is one schedule-priming cycle per tile, and
`CFG_DMR` ignores `skip_en` entirely so the dense baseline stays dense.

**Correctness first:** all seven patterns below were checked element-by-element against an
independent reference with skipping **off and on**, and the two must agree — a pattern that
saves nothing still has to compute the right answer. 7,168 checks at 16x16, all passing.

**Measurement.** `A[16×16] × B[16×32]`, 16x16 array, 2 tiles. "Saving" is measured over the
compute phase (the operand load is a fixed cost of the resident-operand design, §7.9):

| Pattern | zeros A/B | skip off | skip on | **measured saving** | nnz-model predicts |
|:--|:--|--:|--:|--:|--:|
| dense | 0 / 0 | 109 | 111 | −1.8% | 0% |
| unstructured 50% | 51 / 0 | 109 | 111 | **−1.8%** | 51% |
| unstructured 90% | 89 / 0 | 109 | 107 | **1.8%** | 89% |
| 2:4 along k | 50 / 0 | 109 | 111 | **−1.8%** | 50% |
| k-slice 50% (A) | 50 / 0 | 109 | 95 | **12.8%** | 50% |
| k-slice 50% (B) | 0 / 50 | 109 | 95 | **12.8%** | 50% |
| k-slice 25%+25% | 25 / 25 | 109 | 95 | **12.8%** | 25% |

Same patterns at `A[4×64] × B[64×8]`, where the skew fill is amortised over 4x more steps:

| Pattern | skip off | skip on | measured saving |
|:--|--:|--:|--:|
| unstructured 50% | 110 | 106 | 3.6% |
| unstructured 90% | 110 | 71 | 35.5% |
| 2:4 along k | 110 | 106 | 3.6% |
| k-slice 50% (A or B) | 110 | 79 | **28.2%** |

(The buffers force `M·K ≤ 256`, so K=64 means only 4 rows. That makes the *unstructured* rows
of the second table optimistic — an all-zero slice there only has to be zero across 4 rows
rather than 16. The k-slice rows are exact at any M.)

**Two distinct findings, and they compound.**

1. **Structure, not count.** 50% unstructured zeros save nothing; 50% k-structured zeros save
   everything available. The count is identical. At 16 rows the probability that unstructured
   90%-sparse data yields an all-zero slice is `0.9^16 ≈ 18%`, which is why only the extreme
   density registers at all — and it vanishes as the array grows.
2. **2:4 structured sparsity saves nothing either.** Each row's 2:4 pattern is independent, so
   the chance all 16 rows drop the same `k` is `2^-16`. This is an RTL-level confirmation,
   from a different direction, of the bounded negative result in §6: **N:M sparsity does not
   help this dataflow**, and now we can say precisely why rather than inferring it from a
   cost-model constant.
3. **Even perfect structure is amortised.** The ceiling is `f·K/(K+NR+NC)`, not `f`, because
   the `NR+NC−2` skew-fill cycles per tile are not skippable. Hence 12.8% at K=16 and 28.2%
   at K=64 for the same 50% structured sparsity. The measured numbers track that formula.
   *(Rev 18: this ceiling is a property of the per-tile fill in this implementation, not of
   the dataflow. With the fill amortised (§7.13) a skipped step costs nothing and the ceiling
   becomes `f` per tile.)*

**Consequence for the cost model — this is a claim-level issue.** `cost_model.py` converts
sparsity to a scalar `nnz` (`_compute_nnz(M*K, sparsity_A)`) and charges compute in proportion
to it. Nothing in that term knows *where* the zeros are, so it credits the sparse
configurations with savings this dataflow cannot realise. The RTL says the achievable saving
is closer to

```
f_struct · K / (K + NR + NC)     where f_struct is the all-zero-k-slice fraction,
                                 not the zero fraction
```

**I have not changed the cost model.** Doing so would move every table in §3–§6, and the
choice between (a) adding a structure-aware term, (b) restricting the claims to structured
sparsity, or (c) reporting the gap as a limitation is a judgement about what the paper is
arguing — not a bug fix. It should be decided deliberately. The measurement is here so that it
can be.

Reproduce with `idea2_veriflex/rtl/scripts/run_sparse_sweep.sh`.

### 7.8 Defects fixed

| Site | Defect |
|:--|:--|
| `config_prediction_engine.sv` ×3 | `{8{b[k][7]}, b[k]}` — replication in a concatenation needs its own braces |
| `distribution_network`, `benes_switch` | Array ports unsigned while all connecting signals were signed |
| `versaaccel_top.sv:121` | Enum ternary needed an explicit `config_t'()` cast |
| `systolic_top.sv`, `sigma_top.sv` ×7 | `X[$clog2(X)-1:0] - 1` widens to `32'hFFFFFFFF` — **always false**, silently breaking weight-load and drain in **both baseline designs** |
| `tb_systolic`, `tb_sigma` | Hardcoded 8x8 loop bounds; at 16x16 only 64 of 256 weights were loaded |
| `distribution_network`, `reduction_network` | Hand-unrolled for N=8 (§7.3) |
| `config_prediction_engine.sv` | Constant weights, 16-bit overflow, no requantisation (§7.2) |
| `versaaccel_top.sv` | Unwritten `ibuf` entries (X) driven into the PE array; **DMR and D-MR output was undefined** (§7.4) |
| `versaaccel_top.sv` | GEMM load terminated on `ibuf_wr_ptr >= a_total[$clog2(IBUF_DEPTH)-1:0]`. When `M·K` is an exact multiple of `IBUF_DEPTH` the truncation makes the threshold **0**, so the load ended after a **single** activation. Silent for every earlier shape; found by `tb_gemm_sparse` at M=K=16. Now tested on the (row, col) counters. |

### 7.9 What the RTL does and does not claim

§7.7 measured zero-skipping in the RTL and found it cannot realise the savings the sparse
configurations are credited with. That gap deserves a precise statement rather than a
disclaimer, because it is a difference of *dataflow*, not a defect in either artifact.

**The cost model's sparse configurations describe a Gustavson row-stationary dataflow.**
`config_modes.py` says so directly — `Gustavson: total_nnz × ceil(P/Nc) + nnz_rows × (Nr+Nc)`.
In that dataflow a compressed row is streamed nonzero-by-nonzero out of a CSR-like structure,
so individual zeros genuinely cost nothing. The model is not wrong about its own machine.

**The RTL implements one dataflow for all configurations: output-stationary tiled GEMM.**
`ibuf`/`wbuf` hold dense arrays indexed `row*K + k`; there is no compressed operand
representation anywhere in the design. The configuration selects distribution and reduction
network modes, not a different dataflow. The RTL therefore cannot exhibit the differences
between configurations that the model predicts.

**So the scope is:**

| | Status |
|:--|:--|
| Control, interconnect, arithmetic, FSMs | **Validated in RTL** — 13,215 golden checks at 16x16 |
| Dense tiled GEMM (the `DMR` baseline) | **Validated in RTL** — 2,546 output elements across 6 shapes, two bus widths, under backpressure |
| Sparse configurations (`D-MR`, `DM-R`, `D-M-R`) | **Analytical only** — the RTL does not implement their dataflow |
| N:M as a 5th configuration | **Analytical, and negative** (§6), corroborated by an RTL measurement of why (§7.7) |

This is a deliberate scoping decision, not an omission. Implementing Gustavson would mean
compressed operand storage, per-row scheduling and a different control path — a second
accelerator rather than a feature added to this one — and the contribution of this work is
the cost-model study, for which the RTL's job is to make the **dense baseline** credible.
It now does that to a standard most cost-model papers do not reach.

**The paper must say this in §7, not in a closing limitations paragraph.** A clearly scoped
artifact is stronger than one whose scope a reader has to infer.

### 7.10 Still outstanding

1. **No reportable synthesis results yet.** Genus has run once (§7.11), but that run predates
   the port fix, used the fast corner and vectorless power, and covers only `versaaccel_top`.
   The 100–250 cycle switch costs therefore remain estimates. Two structures added in Revisions 6–7 are
   unquantified for the same reason: the output double buffer (`NR·NC·ACC_W` flip-flops) and
   the nonzero bitmaps (`⌈M/NR⌉·K + K·⌈P/NC⌉` bits, plus a K-wide priority encoder).
2. **The cost model and the RTL disagree about sparsity** (§7.7). This is the largest
   remaining gap, and it is a claim-level decision rather than a defect to fix.
3. ~~**Operands are resident, not streamed.**~~ **Addressed as an RTL variant (Rev 18, B12,
   §7.13):** `-DVERSAACCEL_STREAM` streams activations through a two-tile-row ring while the
   array computes; weights stay resident. The default build is unchanged.
4. ~~**The skew fill is not amortised across tiles.**~~ **Addressed as an RTL variant (Rev 18,
   B10, §7.13):** `-DVERSAACCEL_AMORTISE` pays the fill once per GEMM. No second feed port was
   needed. The default build is unchanged.

---

### 7.11 First synthesis run — and why its numbers are not yet reportable

The Genus flow ran for the first time on 2026-09-21: Genus 21.10-p002_1, `gsclib045`, 2.0 ns
clock, identical constraints for all three designs (`syn_versaaccel.tcl`, `syn_systolic.tcl`,
`syn_sigma.tcl`). VersaAccel ran first; the two baselines followed on 2026-09-26/27 (below).

| Metric | `versaaccel_top`, run 1 |
|:--|--:|
| Timing at 2.0 ns (500 MHz) | **met**, 0 violating paths, worst slack 0 ps |
| Cell area | **1,290,729 µm²** (1.95 mm² including net area) |
| Leaf cells | 444,254 (48,979 sequential) |
| Power (vectorless, pre-CTS) | 510.9 mW — 314.4 internal, 195.7 switching, 0.9 leakage |
| Runtime | 2,622 s CPU, 57 min wall |

| Instance | Cell area (µm²) | Share |
|:--|--:|--:|
| `pe_arr` — 256 PEs, ~2,062 µm² each | 527,944 | 40.9% |
| `red_net` | 4,348 | 0.3% |
| `cpe` | 1,548 | 0.1% |
| `phase`, `spec` | 135 | <0.1% |
| *flattened into `versaaccel_top`* | *756,754* | *58.6%* |
| **`dist_net`, `nm_rom`** | **absent** | — |

The critical path runs from the `dim_p` input pin through the weight-buffer address
arithmetic (`fetch_b`) into a PE accumulator. **Address generation, not the PE, sets Fmax.**

**Three defects make this run unusable for comparison. All three are in the RTL, not the flow.**

1. **The Benes network was synthesised to nothing.** `dist_ctrl` was `'0` in every
   configuration. The one exception, N:M, read `nm_rom[nm_pattern_idx]` with the index tied to
   `3'd0` — a ROM entry that is itself all zeros. Every switch was therefore fixed
   straight-through, the network reduced to wires, and Genus removed it along with `nm_rom`.
2. **The CPE was folded to a constant.** All nine CPE features were tied to `8'sd0`, so the
   trained MLP's output was known at compile time. The 733 cells that remained are handshake
   registers; the 0.930-accuracy network of §7.2 was not in the netlist.
3. **The comparison would have been biased in VersaAccel's favour.** SIGMA's Benes is driven
   from an input port (`routing_ctrl`) and survives synthesis. VersaAccel's did not. Compared
   as run, VersaAccel would have looked cheaper than SIGMA on exactly the hardware they share.

**Fixed in `ed6d8ac`.** `routing_ctrl` (56 bits at NR=16), `nm_pattern` and
`layer_features[9]` are now input ports, captured into registers on the `start` edge that
raises `layer_start` — the cycle the CPE samples, so no combinational path runs from a pin
through the Benes into a PE. D-MR and D-M-R take `routing_ctrl`; DMR and DM-R stay
straight-through, following §1's table; GEMM mode forces identity routing, as it already forced
BYPASS reduction. A new bench, `tb_layer_config` (31 checks), proves each pin reaches its logic.
**Reverting to the old tie-offs fails 12 of those checks**, so the bench distinguishes the two
designs; the CPE, layer-config and GEMM benches also pass under `-DSYNTHESIS`, the constant-ROM
branch Genus compiles.

**What else run 1 shows, which the fix does not change:**

- **It used the fast corner.** `fast_vdd1v2`, PVT 1.32 V / 0 °C, is the best case. Timing for
  the paper should be signed off at `slow_vdd1v0` or typical. 500 MHz here is an upper bound.
- **The power is vectorless and pre-CTS.** No SAIF/TCF was read, so switching activity is
  Genus's default toggle rate rather than a workload, and clock-tree power is zero because no
  tree exists yet. Treat 510.9 mW as an order of magnitude, not a result.
- **58.6% of the area is flip-flop buffers.** From the package parameters, `ibuf` (4,096 bits),
  `wbuf` (8,192), `tbuf` (8,192), `a_nz` (4,096) and `b_nz` (8,192) come to ~32.8k of the 49k
  sequential cells. They are arrays inside the top module, so the hierarchy cannot separate
  them — which means B1's per-structure area is **not** delivered by this flow as written.
  **Fixed in RTL (Rev 16, B1d):** each buffer is now its own submodule — `u_ibuf`, `u_wbuf`
  (`operand_buffer`), `u_a_nz`, `u_b_nz` (`nz_bitmap`) and `u_tbuf` (`tile_buffer`) — holding
  only the storage, with every write driven by a strobe decoded from exactly the condition the
  FSM used before. The next run's `area_hier` should show them at 4,096 / 8,192 / 4,096 / 8,192
  / 8,192 flip-flops. The modules live in `versaaccel_top.sv`, so the Genus file list does not
  change.
  And the systolic baseline has no deep buffers, so a top-level comparison against it is
  skewed the other way.
- **The N:M ROM was never scaled.** Its six entries are 20-bit literals from the 8-port
  design, zero-extended into the 56-bit control word, so N:M only steers the low stages of the
  16-port Benes. Same class of defect as §7.3; N:M is already a withdrawn claim.

#### The two baselines (run 1)

Both baselines were synthesised on 2026-09-26/27 with the same script, clock, constraints and
fast corner as VersaAccel's run 1, so the three are comparable with each other — though not yet
reportable in absolute terms, for the same corner and power reasons as above.

| | systolic | SIGMA | VersaAccel (run 1, pre-fix) |
|:--|--:|--:|--:|
| Multipliers | 256 | **16** | 256 |
| Cell area (µm²) | 562,541 | 79,246 | 1,290,729 |
| Cell area per multiplier | 2,197 | 4,953 | 5,042 (PE array alone: 2,062) |
| Sequential cells | 16,423 | 4,899 | 48,979 |
| Power, vectorless (mW) | 278.6 | 61.2 | 510.9 |
| Worst slack at 2.0 ns | **+468 ps** | +1 ps | 0 ps |
| Critical path | `weight_reg` → `accum` (the MAC) | `step_cnt` → `dp_accum` | `dim_p` pin → address arithmetic → PE |
| Genus runtime | 10,105 s | 234 s | 3,421 s |

**The systolic baseline is complete and sound.** Its 16,423 sequential cells are exactly 256 PEs
× 64 bits (16 weight + 32 accumulator + 16 forwarding) plus 39 of control, and it has no
constant tie-offs. It is the right dense reference.

**Three comparisons are meaningful now:**

1. **Reconfigurability costs nothing measurable inside the PE.** VersaAccel's 256-PE array
   (527,944 µm²) is 6% *smaller* than the whole systolic array (562,541 µm²), despite each PE
   taking a configuration input.
2. **VersaAccel is 2.29× the systolic array's area, and that gap is its buffers.** The
   difference, ~728k µm², matches the 757k µm² flattened into `versaaccel_top` — the flip-flop
   `ibuf`/`wbuf`/`tbuf`/bitmaps the systolic design does not have. The overhead of
   reconfigurability itself is not yet visible, because it is hidden behind buffer area (B1d).
3. **The Benes costs 4,312 µm²** — measured from SIGMA, whose `distribution_network` is the same
   module at the same N=16 (56 switches × 76.6 µm²). That is what VersaAccel's run 1 omitted:
   **~0.3% of its area.** The bias §7.11 identified was real in kind but small in magnitude.
   The CPE is the one component whose cost is still unknown until VersaAccel's run 2.

**SIGMA, as written, is not a fair baseline — for two independent reasons:**

- **It has 16 multipliers, not 256.** `sigma_top` instantiates `N_UNITS = NR` dot-product
  units and each performs *one* multiply per cycle, time-multiplexed over `step_cnt`. The code
  comment says each unit "has NC multiply-add units"; the code does not. Every raw area and
  power number above compares a 16-MAC design against 256-MAC ones.
- **Its reduction network was constant-folded away** — the same defect as VersaAccel's Benes.
  `fan_reduce` has `.mode(2'd2)`, BYPASS, hard-wired, so the adder tree is never used and Genus
  deleted it. SIGMA's defining feature is its Forwarding Adder Network; this netlist has none.

Per multiplier SIGMA looks 2.25× the systolic array (4,953 vs 2,197 µm²), but that figure
carries both distortions — control and weight storage amortised over only 16 MACs, and no FAN —
so it is not a measurement of SIGMA's flexibility cost.

**Fixed in RTL (Rev 15); needs a second synthesis run.** `sigma_top` is now 16 Flex-DPEs × 16
multipliers = **256**, iso-compute with the other two designs. Each cycle one 16-element chunk of
the activation vector is routed through the Benes, broadcast to every unit, multiplied lane-wise
against each unit's stationary weights, and reduced by that unit's own FAN — 16
`reduction_network` instances, the same module VersaAccel uses. The FAN mode, the Benes routing
and K are captured into registers on `start`, and the three modes produce *different* results
(one dot product, 8 pair-sums or 16 lane sums per unit), so none can be folded away. The FAN is
pipelined as SIGMA's is — Benes → register → multiply → register → FAN + accumulate — because a
multiply, a 4-level adder tree and an accumulate do not fit 2.0 ns. That product register
(8,192 bits) is a real cost of SIGMA's deeper reduction and belongs in the comparison.

`tb_sigma_golden` was rewritten to match: **1,484 checks** over 11 runs — all three FAN modes,
K = 5, 16, 21, 32 (partial chunks exercise masking of both operands), identity and two scrambled
Benes routings, drain backpressure, signed operands spanning −128…127 — plus a cycle count
confirming one 256-multiply chunk retires per cycle (`ceil(K/16) + 3`). Mutation-tested:
hard-wiring the FAN to BYPASS (the old defect) fails 442 checks; an unsigned `'0` in the product
mask — a bug caught in review before the first build — fails 1,046; dropping the weight mask
fails 30, only in the routed partial-chunk run, which is why that run exists. `tb_sigma`'s two
checks that were asserted as constant `1'b1` now observe the design.

Expected in the second run: ~25.4k sequential cells (weights 8,192 + products 8,192 +
accumulators 8,192 + activations 512 + Benes register 256 + control), `benes_dist` plus 16
`g_fan[*].fan` instances in `area_hier`. Far fewer flops would mean something was folded again.

**Fmax cannot be compared from these runs.** Genus stops optimising once timing is met, so
0 ps and +1 ps slack say "met", not "at the limit". The systolic array's +468 ps on a
register-to-register MAC path suggests headroom to roughly 650 MHz at this corner, but the only
way to compare is a clock-period sweep, at the slow corner, for all three (B1g).

**A latent testbench race, exposed by B1d.** Four benches (`tb_versaaccel`,
`tb_versaaccel_golden`, `tb_gemm_golden`, `tb_gemm_sparse`) assign the design's inputs with
blocking statements immediately after `@(posedge clk)` — in the same instant as the edge — and
their expected values assume the design sees those inputs *at that edge*. It did, but only
because iverilog happened to run the bench process first. Moving the buffers into submodules put
one derived write strobe between `din_data` and the storage, the order flipped, and writes landed
a beat late: the refactor looked broken when it was not.

The fix is in the benches: the design is now clocked 100 ps after the bench clock, which makes
the intended semantics explicit and deterministic without changing a stimulus line. Two tempting
alternatives were tried and rejected because they failed the criterion below — converting every
input to a non-blocking assignment shifted every cycle count by one and broke three shapes, and
converting only the data inputs inserted a one-cycle bubble between `start` and the first beat
that no real master would add.

**The criterion, and the result.** A bench change is acceptable only if the new bench on the
*original* design reproduces the old bench's output exactly. With the skew, the complete output
of all four benches — every check and every printed cycle count — is **byte-identical** across
the original design with the old bench, the original design with the skewed bench, and the
refactored design with the skewed bench; so are all 11 GEMM shapes of §7.12 and the sparse sweep.
The refactor is exactly equivalent, and no recorded number moves.

### 7.12 The cost model checked against the RTL (B3)

The cost model had only ever been validated against the VersaAccel paper's own Table II. The
RTL's tiled GEMM now gives it an independent reference, for the one configuration the RTL
implements: dense (DMR).

**The RTL has an exact closed form**, confirmed cycle-for-cycle by `tb_gemm_golden` on 11
shapes spanning K = 1 to 256, the largest K the buffers hold, with every output element also
correct:

```
load    = M·K + K·P                                  (one element per cycle on din)
compute = ⌈M/NR⌉·⌈P/NC⌉·(K + NR + NC − 1) + NC       (wide bus; last tile's drain exposed)
```

**The model's dense term is the ideal-utilisation bound, not a cycle count.** For MM,
`DMRConfig` returns `⌈M·K·P / (NR·NC)⌉`: every PE busy on every cycle, with no skew fill, no
drain and no partial tiles. For full tiles the ratio to the RTL is about (K+31)/K — 6.2× at
K=6, 1.48× at K=64, 1.02× at K≈1,250. The model is also inconsistent with itself: its
matrix-vector formula *does* charge fill (`2·Nr+1` per tile), and D-MR charges `Nr+Nc−1` per
block, but DMR's MM charges none.

The load term matches trivially: at the RTL's 1 element/cycle, `Costmem` gives `M·K+K·P+1`.

**On the real layer shapes** (`experiments/costmodel_vs_rtl.py`, extrapolating the verified
closed form beyond K=256):

| Network | Unique shapes | K range | RTL / model, work-weighted | Worst layer |
|:--|--:|:--|--:|:--|
| BERT | 10 | 768 – 3,072 | **1.035×** | 16.65× (`pooler.dense`, M=1) |
| ResNet-50 | 21 | 64 – 4,608 | **1.187×** | 16.37× (`fc`, M=1) |

The M=1 layers use one of sixteen PE rows, which an ideal-utilisation term cannot see. Below
them there is a cliff to 1.48× (the K=64 convolutions); most layers are within a few percent.

**Does it change decisions?** Giving the model the RTL's fill term and re-deciding every
(layer, input) point:

- **1,337 of 55,536 points (2.4%) change configuration** — all DMR → D-M-R, all with
  K = 256 – 1,152, concentrated in `resnet50_nm24` (14.6% of its points) and
  `resnet50_pruned70` (1.9%). No other network flips at all.
- **The headline barely moves: 1.184× → 1.173×** (published model — §3.0 has since replaced it), heterogeneous 1.275× → 1.259×, both well
  inside the ±0.10 confidence interval (§3.2). `--headline` reproduces this.

So the model's largest validated error is real, systematic and bounded: it undercounts dense
compute by 3.5% on BERT and 18.7% on ResNet, and fixing it moves 2.4% of decisions and none of
the conclusions. That is a stronger statement than "validated against Table II", and it is the
one the paper should make.

**What this does not validate.** Only DMR. The sparse configurations' formulas have no RTL
counterpart (§7.9), so whether D-M-R's cost is right — and therefore whether those 1,337 flips
land on the true optimum — is not established. Adopting the fill term is recommended (it makes
MM consistent with MV and with the RTL) but it changes every §3–§6 number slightly, so it is
recorded as a decision rather than applied (B11).

### 7.13 Fill amortisation and streamed activations (B10, B12)

Two limits of the GEMM path were stated in §7.6/§7.10 as properties of this implementation:
every tile paid the `NR+NC−2` skew fill, which is what capped the zero-skip saving at
`f·K/(K+NR+NC)` (§7.7), and both operands had to be resident before compute started. Both are
now built as RTL variants. **The default build is unchanged** — all three RTL suites
reproduce byte-for-byte without the defines — so every earlier number stands.

**B10 — `-DVERSAACCEL_AMORTISE`: one fill per GEMM.** §7.6 assumed this needed a second
operand feed port. It does not: `a_in[r]` at cycle t already carries schedule entry `t−r`, so
the next tile's first entry can enter on the cycle after the current tile's last, and each
port still delivers one value per cycle. What has to change is that consecutive tiles now
*overlap* inside the array, so three things that were global become per-entry or per-PE:

1. each schedule entry carries its tile (`ktag_i/ktag_j`), so a feed fetches the right rows
   while its neighbours still feed the previous tile;
2. the accumulator clear is a **diagonal wavefront** — an entry flagged *first* reaches PE(r,c)
   after r+c cycles (span-relative for the reshaped tiles of §2.9) and that PE clears then;
3. the finished value leaves the PE on that same cycle into a **two-bank output buffer**
   (`NR·NC·ACC_W` more flip-flops than §7.6's single bank), drained one bank at a time.

A tile's entries are its live k-steps — `K` dense, its skippable-mask steps when skipping —
issued back to back; a second priority encoder finds the tile's last live step so no cycle is
spent discovering that a tile is over. A skipped step therefore costs **nothing**, and the
per-tile `ST_GEMM_SCHED` cycle of §7.7 is gone too: skip-on dense now equals skip-off exactly.
After the last tile a terminator entry flushes the array, so the fill is paid once per GEMM.

**B12 — `-DVERSAACCEL_STREAM` (with AMORTISE): activations overlapped with compute.** B (the
weights, reused by every tile-row) is loaded first and stays resident; A streams through a
**ring of two tile-rows** in `ibuf` while the array computes, with the loader stalled
(`din_ready` low) until the tile-row two back has left the feed pipeline — input double
buffering at tile-row granularity. A tile starts as soon as its rows have arrived. `ibuf` now
bounds `2·tile_r·K`, not `M·K`, so **GEMMs whose activations do not fit the buffer now run**.

**Verification.**

| | Result |
|:--|:--|
| Full suite on the AMORTISE build | **13,215 / 13,215** golden checks pass (`IVFLAGS=-DVERSAACCEL_AMORTISE run_all_tb.sh`) |
| Random sparse GEMMs, resident and AMORTISE builds × 3 tile shapes × 4 shapes, with and without 25% random stalls on both ports | **700,416** element checks, 0 errors |
| STREAM: A of 400–560 entries through a 256-entry buffer, 3 shapes, with and without stalls | **195,520** checks, 0 errors |
| Real pruned slices (below), 3 builds × 3 shapes | **2,801,664** checks, 0 errors |
| Cycle accounting | resident = closed form + 17 and AMORTISE = cycle model + 1, **in every run** (Rev 20 RTL: + 21 and + 2, with the 3-cycle feed in the model's bank timing — exact again on all 14 real slices) |

The new bench `tb_stream.sv` runs the loader and collector concurrently and obeys both
handshakes — the resident-protocol benches ignore `din_ready`, so they cannot exercise a
loader that stalls. The AMORTISE cycle model (`pipe_cycles()` in `amortise_rtl.py`) is not a
formula fitted to the RTL: it mirrors the generator, the bank hand-off and the drain, and it
matches in the drain-bound regime (short tiles, where one tile leaves every `NC` cycles) as
well as the compute-bound one. A simulation-only check now reports any GEMM that exceeds the
buffers; that overflow was previously silent.

**Measured on real data.** Two slices of each of 4 pruned ResNet-50 layers (A 128×32, B
32×128, 64 tiles) and 3 pruned BERT layers (A 32×32, B 32×128, 16 tiles), on buffers enlarged
for the purpose. Compute phase (load excluded), skipping on vs off:

| Workload | Tile | skip saving, per-tile fill | **skip saving, amortised** | skip-on compute, amortised vs per-tile |
|:--|:--|--:|--:|--:|
| ResNet-50 90% | 16×16 | 11.2% | **24.3%** | 2.28× faster |
| ResNet-50 90% | 32×8 tall | 19.2% | **31.8%** | 2.01× |
| BERT 90% | 16×16 | 12.2% | **25.2%** | 2.19× |
| BERT 90% | 32×8 tall | 26.9% | **37.9%** | 1.93× |

*(Rev 20 RTL. On the Rev 18–19 RTL: 24.8%, 33.1% (19.3% per-tile), 25.7%, 40.9% (27.0%); 2.29×,
2.05×, 2.21×, 2.03×. Most of the change is the fixed 4 cycles per GEMM against these short
slices; the TALL rows lose more because each output bank now stays busy 3 cycles longer per
tile, and TALL's 8-column halves leave less drain slack to absorb it — §7.14.)*

Dense compute alone is 1.94× (ResNet) and 1.87× (BERT) faster at these K=32 shapes, which is
the `(K+31)/K` fill removed.

So **the zero-skip saving the structure allows roughly doubles once the implementation stops
diluting it**, and §7.7's finding 3 is confirmed to be a property of the per-tile fill, not of
the dataflow. Findings 1 and 2 are untouched: amortisation makes a skipped step free, but it
does not make more steps skippable — unstructured 50% and 2:4 still save nothing (§7.7 sweep
re-run on this build: 0.0% for both, 12.3% / 33.0% for 50% k-structured).

**One result moves the other way, and it belongs in the paper.** The tall tile's advantage
over the square one (§2.9) shrinks once the fill is amortised — BERT 1.37× → **1.26×**, ResNet
1.26× → **1.12×** (skip-on compute; the wide tile falls below 1 on both). Part of §2.9's RTL
gain was the reshaped tile's shorter fill (`span_r+span_c` = 24 vs 32), which amortisation
removes for every shape. What remains is the structural effect alone: a narrow edge facing the
pruned operand still finds more all-zero k-slices. §2.9's per-tile-fill RTL numbers stay valid
for the default build; on a fill-amortised array they should be quoted as above.

**End to end, streaming.** The RTL loads one element per cycle, so at these shapes every GEMM
is load-bound (`K·P + M·K` = 8,192 cycles per ResNet slice against ~3,600 cycles of skip-on compute with
the per-tile fill, ~1,600 amortised).
Skip on, best tile shape per slice:

| Workload | load alone | resident, per-tile fill | AMORTISE | **STREAM** | resident / STREAM |
|:--|--:|--:|--:|--:|--:|
| ResNet-50 90% (8 slices) | 65,536 | 88,286 | 76,784 | **66,571** | **1.33×** |
| BERT 90% (6 slices) | 30,720 | 34,674 | 32,772 | **31,593** | 1.10× |

With streaming, compute hides behind the load almost entirely: 1,035 cycles beyond the load
over 8 ResNet slices, 873 over 6 BERT slices (1,003 and 849 before Rev 20's 4 cycles per GEMM).

**What this means for the cost model.** `cost_model.py` charges `Costcomp + Costmem` — compute
and transfer in *sequence*, which is what the resident RTL does. The streaming RTL realises
`≈ max(Costcomp, Costmem)` instead, so for a streamed design the model's sum overstates
runtime by up to `min(Costcomp, Costmem)` — up to 2× when the two balance — and could rank
configurations differently where one is compute-bound and another memory-bound. This is the
first time the RTL has exercised the model's bandwidth term against something other than the
trivial load count (§7.12). Whether to model overlap is a cost-model decision of the same kind
as B11 and is recorded there, not applied.

**Scope.** Both are compile-time variants, not mode pins: the defaults are the design §7.1–§7.12
measured, and synthesising each define against the default gives its hardware cost (second
output bank, per-PE clear OR, tile tags, wavefront register, last-step encoder; ring addressing
and the loader stall) — pending the Genus run (§7.11). STREAM streams activations only; the
weights are loaded whole first, and cross-layer prefetch (loading layer n+1's weights during
layer n) is not built. The narrow drain and backpressure are supported but bound throughput
exactly as before. Reproduce with `experiments/amortise_rtl.py` (`--quick`: random and
streaming parts only).

### 7.14 Synthesis run 2, and the operand feed pipelined (Rev 20)

**Run 2 (2026-10-29/30, college machines).** Genus 21.10-p002_1, `gsclib045` **slow corner**
(`slow_vdd1v0`, 0.9 V, 125 °C — B1b), identical constraints for all designs, hierarchy kept,
`syn/genus/syn_college.tcl`. Nine runs finished their reports (a `report_summary` call in the
script aborted each one just before the VCD power step and `write_db`, so power below is
vectorless; fixed for run 3). Worst slack converted to the critical-path delay it implies:

| Design | Critical path | Fmax implied | Cell area at 2.0 ns | Sequential cells | Power, vectorless |
|:--|--:|--:|--:|--:|--:|
| systolic (256 MACs) | 2.03 ns | ~490 MHz | 2.47 mm² | 16,423 | 489 mW |
| SIGMA (256 multipliers, 16 FANs — B1f) | 2.44 ns | ~410 MHz | 2.21 mm² | 25,439 | 740 mW |
| **VersaAccel** | **5.57 ns** | **~180 MHz** | 4.17 mm² | 49,320 | 817 mW |
| + `RESHAPE` | 5.17 ns | | 5.55 mm² | 61,610 | 1,060 mW |
| + `AMORTISE` | 4.78 ns | | 4.27 mm² | 57,864 | 823 mW |
| + `AMORTISE` + `STREAM` | 4.49 ns | | 4.26 mm² | 54,289 | 819 mW |

What run 2 settled:
- **The reconfigurable hardware survives synthesis now (B1a).** `area_hier` has `dist_net`
  (5,751 cells), `red_net` (5,574), `nm_rom` and a CPE of **37,310 cells** — against 733 when its
  inputs were tied off in run 1. The `ed6d8ac` pin fix works.
- **SIGMA is a fair baseline (B1f):** 25,439 sequential cells, the ~25.4k §7.11 predicted, with
  `benes_dist` and all 16 `g_fan[*].fan` reduction trees present.
- **VersaAccel's clock was 2.7× slower than the baselines', and the cause was one path.** Every
  one of the 20 worst paths started at a configuration pin (`cfg_override_en`, `dim_k`) and ran,
  in a single cycle, through the operand-address arithmetic — a 32-bit `row × K` multiply per feed
  port — the 256-way buffer read, all seven Benes stages and the PE multiply into its
  accumulator. The systolic baseline's critical path is the PE multiply-accumulate alone.
- **The areas at 2.0 ns are not comparable.** All three designs miss 2.0 ns at this corner, and a
  design that misses its clock is up-sized by Genus in proportion to how badly it misses — 63% of
  VersaAccel's area is the PE array at ~10,300 µm² per PE, and another 24% is the controller and
  feed logic flattened into `versaaccel_top`, much of it the 64 per-port address multipliers.
  Area and power have to be compared at a clock all three meet.

**The fix: a pipelined operand feed (Rev 20).** The RTL changes, all in
`versaaccel_top.sv` and `config_prediction_engine.sv`:

1. **Configuration is registered.** `dim_m/k/p` and `skip_en` are captured on `start` (as the
   routing, N:M and CPE pins already were), and the selected configuration is a register, so no
   pin reaches the datapath combinationally.
2. **Three feed stages in front of the array:** F0 computes each port's buffer address and
   whether the element exists; F1 reads the buffer; F2 routes A through the Benes (B and the
   reshape half-feeds wait one cycle). Every PE-side control — accumulator clear and enable, the
   B10 clear/capture wavefront, the tile capture — is delayed by the same `FEED_LAT = 3`, so the
   array sees exactly the sequence it saw before, three cycles later.
3. **Addresses in the buffers' own width.** An in-matrix element's index is below
   `M·K ≤ IBUF_DEPTH` (`K·P ≤ WBUF_DEPTH`), so only the low `log2(DEPTH)` bits of `row × K` are
   needed, and those depend only on the operands' low bits: 8×8 and 9×9-bit multipliers replace
   32-bit ones. Shape-dependent products (tile origins, bitmap granules, the B12 ring slot) became
   per-shape constants.
4. **The output port is registered**, with the same valid/ready protocol; a GEMM drains the tile
   buffer directly (it had used the reduction network in BYPASS, the identity), so the adder tree
   sits only on the legacy FULL/PARTIAL path.
5. **Smaller cuts:** the skip schedule's later steps read a registered copy of the tile's mask
   (only a tile's first step still sees the live bitmaps); the CPE is six stages instead of three
   (dot product, then requantise, per layer), with a tournament argmax that keeps the
   lowest-index tie-break.

**What it costs: 4 cycles per GEMM, nothing per tile.** Tiles still take `K + NR + NC − 1`
cycles back to back; the feed adds 3 cycles and the output register 1, once. All 11 GEMM shapes
of §7.12 measure exactly their Rev 19 count + 4 in both drain widths (the closed form in
`costmodel_vs_rtl.py` now carries the `+ 4`), and no §7.12 ratio or decision moves. A capture
waits for the array only as long as it still needs to, so drain-bound runs lose nothing either.
The fixed RTL-versus-accounting offsets quoted in §2.8, §2.9 and §7.13 each grow by the same
constant — regrouping and reshape 15 → 19, tile shape `NC − 1` → `NC + 3`, the GEMM closed
form `+ NC` → `+ NC + 4`; every cycle *difference* — skip savings, regrouping gains,
tile-shape speedups, B10/B12 gains — is unchanged, because it is a difference between two runs
that both carry the constant.

**Verified on all four builds** (default, `RESHAPE`, `AMORTISE`, `AMORTISE`+`STREAM`): all 11
benches (13,215 checks), the 11 GEMM shapes, both sparse sweeps, the CPE against the Python
model in both the `$readmemh` and synthesis-ROM views, and the B10/B12/B13/B15 measurement
scripts — 245,760 regrouping checks (B13), 122,880 reshape checks (B15), all five tile shapes,
each with 0 output errors and one exact offset; the B10/B12 script (`amortise_rtl.py`, 3.7M
checks) is re-running. Two benches were corrected, and both corrections pass on the Rev 19
RTL as well: `tb_versaaccel_golden` sampled the drain on fixed cycles after `wait (dout_valid)`
rather than on accepted beats, and `tb_cpe_golden` set `valid_in` on the same edge the CPE sampled
it — a race it won only by scheduling order. `tb_gemm_golden` does not obey `din_ready`, so it
cannot drive the `STREAM` build when the activations exceed the ring (it hung on the Rev 19 RTL
too); `STREAM` is verified by `tb_stream`, which does.

**Run 3 and 4 results: §7.15.** VersaAccel's critical path became the PE multiply-accumulate at
2.5 ns, as expected, and its Fmax rose from ~180 MHz to ~430 MHz. The N:M routing ROM still holds 20-bit routes from the 8-port Benes, so at 16 ports it
drives only 20 of 56 switch bits; the N:M configuration is a dropped, negative result (§6) and
the ROM is 10 µm², so this is recorded rather than redesigned.

### 7.15 Synthesis runs 3 and 4 — the comparison (Rev 21)

Genus 21.10, `gsclib045` slow corner (`slow_vdd1v0`, 0.9 V, 125 °C), identical constraints,
hierarchy kept; the Rev 20 RTL (§7.14). Run 3 (2026-11-03): all designs and variants at 2.5 ns,
VersaAccel at 2.0 ns. Run 4 (2026-11-05): the three designs again at 2.5 ns with power over the
compute phase only, and each at the clock run 3 said it could reach.

**Clock.** VersaAccel's critical path is now the PE multiply-accumulate at 2.5 ns, as in the
systolic baseline; at 2.0 ns it is the zero-skip schedule generator (`tile_j` → bitmap row →
mask → first-step encoder → `k_cur`), 2.33 ns.

| Design | Last met clock | Critical path when pushed | Fmax |
|:--|--:|--:|--:|
| systolic (256 MACs) | 2.5 ns (2.1 ns: −3 ps) | 2.03–2.10 ns, MAC | **~475 MHz** |
| **VersaAccel** | **2.4 ns** | 2.33 ns, skip-schedule generator | **~430 MHz** |
| SIGMA (256 multipliers, 16 FANs) | 2.6 ns | 2.52 ns, product → FAN | **~395 MHz** |

VersaAccel was 5.57 ns (~180 MHz) before §7.14's feed pipeline. It now sits between the two
baselines: 10% below the systolic array, 8% above SIGMA. Fmax is 1 / the critical path when the
clock is set just below it; each design also meets its own last listed clock, so the figures are
not extrapolations.

**Area, at the common 2.5 ns clock.**

| Design | Cell area | vs systolic | Timing |
|:--|--:|--:|:--|
| systolic | 0.945 mm² | 1.00× | met |
| SIGMA | 1.551 mm² (1.317 at 2.6 ns, met) | 1.64× (1.39×) | −23 ps |
| **VersaAccel** | **1.867 mm²** (run 3: 1.864) | **1.98×** | met |

| VersaAccel block | Cell area | Share |
|:--|--:|--:|
| PE array (256 PEs, 4,367 µm² each) | 1.118 mm² | 60.0% |
| controller, operand feed, schedule — flattened into the top | 0.405 mm² | 21.7% |
| nonzero bitmaps (`a_nz`, `b_nz`) | 0.112 mm² | 6.0% |
| operand buffers (`ibuf`, `wbuf`) | 0.108 mm² | 5.8% |
| output tile buffer (`tbuf`) | 0.064 mm² | 3.4% |
| CPE (trained MLP, 6 stages) | 0.045 mm² | 2.4% |
| reduction network | 0.008 mm² | 0.4% |
| Benes distribution network | 0.004 mm² | 0.2% |

The reconfigurable datapath itself is cheap: the PE array plus both networks is 1.13 mm², 1.20×
the *whole* systolic array, and the Benes and reduction networks together are 0.6% of the
design. What costs area is everything around the array that a fixed systolic array does not
have — the feed and schedule logic, the bitmaps and the buffers (B1d: compare cores, not
totals). The CPE, which a reader might expect to be the expensive part, is 2.4%.

**What each RTL option costs** (area over the default build at 2.5 ns):

| Option | Area | Timing at 2.5 ns | Buys |
|:--|--:|:--|:--|
| fill amortisation (B10) | **+4.5%** — the second output bank | met | skip saving ~2×, §7.13 |
| + streamed activations (B12) | **+2.6%** | met | load hidden, 1.33× end to end |
| reshapeable array, square / wide / tall (B15) | **+57.4%** (+42.7% at 2.8 ns) | **−278 ps** (−408 ps at 2.8 ns) | 1.40× pruned BERT, 1.15× pruned ResNet |
| reshapeable array, square / tall only (Rev 21) | **+20.6%** | −42 ps | 1.40× pruned BERT, 1.12× pruned ResNet |

B10 and B12 are cheap for what they buy. **B15 is not:** the wide shape's second B feed (16 more
512-way read ports), the bitmaps at half-edge granularity and the midline muxes cost more than
half again the design, and it misses the clock. Genus batch 5 (2026-11-07) settles it:

- **The three-shape array does not meet even 2.8 ns.** Re-synthesised at a clock its run-3 path
  suggested, it still costs **+42.7%** and its path grows to 3.21 ns (`gemm_mode` → the
  four-granule bitmap OR → schedule generator → `k_cur`), ~310 MHz against 430. With 1.15× on
  pruned ResNet and 1.40× on pruned BERT it is a net loss per area and per clock everywhere.
- **Square + tall (`-DVERSAACCEL_RESHAPE_TALL`) costs +20.6%** — the second A feed (+0.19 mm² of
  feed and schedule logic), B bitmaps at half-column granularity (+0.09 mm²) and the column cut in
  the PE array (+0.09 mm²) — and misses 2.5 ns by 42 ps: the midline mux select from `shape_q`
  now sits in front of the PE multiply-accumulate (2.54 ns, ~395 MHz against 430).

Performance per area, against the default array (speedups at full layer size, §2.9):

| | speedup | per area, equal clock | per area, own clock |
|:--|--:|--:|--:|
| square + tall, pruned BERT | 1.40× | **1.16×** | 1.07× |
| square + tall, pruned ResNet | 1.12× | 0.93× | 0.85× |
| square + tall, dense layers | 1.00–1.01× | 0.84× | 0.77× |
| square / wide / tall, pruned BERT | 1.40× | 0.98× | 0.71× |
| square / wide / tall, pruned ResNet | 1.15× | 0.81× | 0.58× |

**So reshaping pays in silicon only where weight sparsity dominates** — pruned Transformer layers,
whose skippable structure is all in the static weights (§2.5) — and only with the lean two-shape
array. On pruned CNNs, where half the layers want the other orientation, the gain does not cover
the hardware; the three-shape array covers it nowhere. The *finding* of §2.9 — the right
orientation is set by which operand holds the zeros, and choosing it per layer is worth up to
1.40× — stands as a performance result; the *architectural* claim narrows to "a square + tall
array is worth building for weight-pruned Transformers". Two known costs are fixable and not
fixed: the midline mux could be pre-decoded per GEMM to take it off the MAC path (the 42 ps), and
the B bitmap could stay at whole-tile granularity with a second, half-column bitmap built only
for tall layers. Verified on the RTL before synthesis: all 11 benches and both sparse sweeps on
the tall build, square GEMMs cycle-identical to the default build, `reshape_rtl.py --tall-only`
0 errors and exact on 48 real slices.

**Energy per operation, compute phase only.** Run 3's VCD power averaged over whole testbenches
that were mostly operand loading and did not compare; `experiments/power_windows.py` trims each
VCD to the window from the end of the operand load to the last output beat, and energy per MAC
is average power × window ÷ MACs. Same real operands for all three (ResNet-50 `layer3.2.conv1`,
one Imagenette image), 2.5 ns netlists.

| Design | dense | 90% pruned | saving | test (MACs, window) | flops annotated |
|:--|--:|--:|--:|:--|--:|
| systolic | 1.89 pJ/MAC | 1.32 | 30% | 16,384, 303 cycles | 66% |
| **VersaAccel** | **4.20** | **2.54** | **40%** | 16,384, 288 / 283 cycles | 75% |
| SIGMA | 12.78 | 4.96 | 61% | 512, 22 cycles | 99.6% |

VersaAccel spends ~2.2× the systolic array's energy per MAC dense and 1.9× pruned — the price of
the feed, schedule and buffer logic above — and about a third / half of SIGMA's. Zero-skipping
saves it more energy (40%) than zeros save the systolic array (30%), and it saves cycles too
(288 → 283 here), which the systolic array cannot. Caveats, all to state with the numbers:
SIGMA's test is 512 MACs in 22 cycles, so its fixed overhead is a larger share and its figure is
pessimistic; flops not annotated (25–34% for VersaAccel and systolic) take Genus's default
activity, so ratios are ±~20%; there is no clock tree before CTS, in any design. Vectorless at
2.5 ns, for scale only: 154 / 256 / 441 mW.

**Reconfiguration is not on the critical path.** Paths through the speculation unit have +596 ps
of slack at 2.5 ns, through the Benes control +488 ps; the CPE's slowest stage meets 2.5 ns
exactly. Its cost is therefore in cycles, not clock period — measured in §4.3 (B2).

---

## 8. Idea 4 — Speculative configuration

Methodologically the soundest section: it reports a near-zero benefit honestly.

> **Rev 19.** MAC-bounded model (§3.0); published-model values in the right column.

| Metric | Value | as published |
|:--|:--|:--|
| vs Selection-Oracle | +0.001% | +0.374% |
| vs Execution-Oracle | **+0.002%** (never beats the true ceiling) | +0.403% |
| Degradation at 10% misprediction | +0.20% – +1.13% | +1.08% – +1.54% |
| at 25% / 50% | +1.5%–2.9% / +4.7%–5.6% | +2.7%–3.0% / +4.9%–6.2% |

*(The table previously quoted −0.014% / +0.009%, which predate disabling the N:M discount — see
the note at the end of §8.1; the published-model column is the current engine's output.)*

Given §4 and §5, the honest framing is that **speculation is a safety mechanism, not a
speedup mechanism** — it bounds the cost of being wrong, which matters precisely on the
input-dependent workloads identified in §3.3.

### 8.1 Every unmeasured constant is now named and stress-tested

The module previously carried the same hardcoded `0.25` N:M discount as §6.2 plus a set of
unnamed literals (`sw * 0.5`, `lc * 0.05`, `/ 0.20`). These are now named parameters, split
explicitly into **policy knobs** (thresholds we choose) and **ESTIMATES** (claims about
hardware that nothing here measures): `SWITCH_HIDE_FRACTION`, `HIDE_CAP_FRACTION`,
`SAMPLE_OVERLAP_FRACTION`, `CONFIDENCE_GAP_SCALE`, `CONFIDENCE_DEGENERATE`. The withdrawn
N:M discount is disabled by default rather than silently applied.

Naming a constant does not justify it. `--sensitivity` re-runs the entire evaluation across
the plausible range of each estimate — the antidote to an unfalsifiable parameter is to show
the result over the whole range it could take, not to defend one value:

> Across all **20 parameter settings**, speculation never beats the Execution-Oracle
> (min +0.001%, max +0.003% under the adopted model; as published, min +0.020%, max +2.9–3.1%).

Under the adopted model the whole range collapses to +0.001–0.003%, because there is almost
nothing left to speculate about: the configuration choice rarely depends on the predicted
sparsity (§3.0, §5.1). As published, the maximum is environment-sensitive: +3.106% under torch 2.14 (Rev 10), +2.920% under torch
2.8 (Rev 13), on identical traces and identical predictor weights. The engine runs the trained
predictor as a float32 torch model, and a different torch rounds a handful of predictions that
sit exactly on a configuration boundary differently. The minimum and the oracle bound are
unchanged in both environments — which is the claim — but the maximum should be quoted as
"about 3%", never to three decimals.

So the **oracle-bound claim survives any choice of these constants**, while the *magnitude*
of the benefit does not — which is precisely why no speedup number from this module belongs
in the paper without RTL-measured overlap and flush costs.

Note the headline figures shifted once the N:M discount was disabled: vs Selection-Oracle
−0.014% → +0.371%, vs Execution-Oracle +0.009% → +0.406%. The oracle bound still holds.

---

## 9. Claim disposition

| # | Claim | Status |
|:-:|:--|:--|
| 1 | Adaptive scheduling beats static | **Requalified again (Rev 19, §3.0)** — under the MAC-bounded model 1.052x ± 0.044 over 15 workloads, 1.079x heterogeneous, 1.000x homogeneous; the gain is concentrated on pruned / 2:4 CNNs, 1.11–1.28x (ResNet-50) and 1.10–1.20x (MobileNetV2). As published 1.184x / 1.275x — most of which was two formulas charging below the MAC bound. On this RTL's shared-operand array (model D) 1.000x everywhere |
| 2 | Greedy sufficient at realistic switch costs | **New, strong** — within 0.013% of Oracle under known sparsity (§4; 0.003% as published). Note §3.1's gap columns measure predictor error, not scheduler optimality. |
| 3 | Lookahead pays only above ~265% of layer latency | **New, strong** — ~400x the modelled switch cost under the corrected model (§4; ~121% / 100x as published) |
| 4 | Prediction justified only at >=16x16 | **New, now qualified** — holds at BW >= 4 elem/cyc (0% at 8x8, 33.3% at 32x32). Does NOT hold for bandwidth-starved designs, where even 4x4 is sensitive (§5.1) |
| 5 | Predictor beats trivial baselines | **WITHDRAWN, and stronger (Rev 14)** — over 10 seeds the MLP never beats a zero-parameter constant on decision-relevant layers: 93.4% ± 5.6 vs 96.2%, worse in 8 seeds, tied in 2; the committed model's tie is the best case (§5.2, A8). The wrong-variable rescue is closed at the decision level (§5.3b, B6): structure changes how much a configuration saves, never which one is chosen. |
| 5b | Activation sparsity changes the optimal config | **Strengthened** — 7.2% of layers on real traces, up from 3.2% on synthetic (§5.1). The premise survives; the MLP does not. |
| 6 | Runtime scheduling needed on 6/15 workloads | **Strengthened** — was 2/15 on synthetic traces; all four new entries are ResNet variants that became real in Rev 9 (§3.3) |
| 7 | RTL computes correct arithmetic | **New** — 13,215 golden checks at 16x16 across systolic, SIGMA, the datapath primitives, the CPE, the assembled top level, tiled GEMM at two bus widths, seven sparsity patterns, and the layer-configuration pins (`tb_layer_config`, Rev 13) |
| 7b | CPE runs a real trained network | **New** — INT8 0.930 vs 0.704 majority baseline, bit-exact with Python |
| 7c | RTL scales to 16x16 | **New** — required fixing two networks hand-unrolled for N=8 |
| 7d | RTL executes real tiled GEMM | **New** — 2,546 output elements across 6 shapes incl. partial tiles, each verified at both bus widths and under backpressure (§7.5). Was refuted in Rev 4; the tiling controller resolves it. |
| 7e | The drain is free | **New** — output double buffer + wide bus: 292 → 37 cycles/tile at 16x16, K=6, matching the analytic ideal exactly (§7.6). Zero-skipping remains absent. |
| 7f | Zero-skipping helps only structured sparsity | **New** — 50% unstructured and 2:4 both save 0%; 50% k-structured saves 12.8% at K=16 and 28.2% at K=64, against the `nnz` model's 50% (§7.7). Independently confirms the §6 N:M negative result. The cost model's sparsity term is **unreconciled** with this. |
| 8 | N:M uncompetitive standalone, P<=4 crossover | **Strengthened** — actively harmful (−2.05% under the corrected model; −1.51% as published) as a 5th config on real traces, and §7.7 shows why in hardware |
| 9 | VersaAccel's sparse cycle counts need row-structured sparsity | **New, and the most consequential** — the paper's own Table II example is row-structured; under an i.i.d. estimator its sparse numbers miss by 50–100% (§2.4). Independently confirmed from the RTL (§7.7) |
| 9b | Real sparsity is far more `k`-structured than i.i.d. and far less than `nnz`-proportional | **New** — real ResNet activations have 14–29x more all-zero `k`-slices than i.i.d. predicts at the same density (ReLU zeros cluster by channel, and a channel is a `k`-slice after im2col), but 20% where the `nnz` term credits 56%. Both default models are wrong, in opposite directions (§2.5). |
| 9c | The `nnz` sparsity term errs in **both** directions on real workloads | **New** — over-credits ResNet by up to 38 pts, and misses a 23.7% saving in `bert_pruned90` entirely because that benefit is weight structure an activation term cannot see. Mean realisable saving 13.2% across the 10 real sets. A constant rescale cannot fix a term that is optimistic on one workload and blind on another (§2.6). |
| 10 | N:M gives 25% cycle reduction | **DROPPED** (B7, §6.2) — was a hardcoded constant; no claim is made |
| 11 | Phase controller 100% correct | **Requalified to detection only** — measured performance impact +0.00% |
| 12 | Attention sparsity varies 0–99% | **Verified** (Rev 10) — pretrained `bert-base-uncased` on SST-2 with eager attention: measured range **0.062–0.982**. SDPA never forms the N×N matrix, so `attn_implementation="eager"` is required to measure this at all |
| 13 | Speculation never beats Execution-Oracle | **Keep** |
| 14 | Switch costs 100–250 cycles | **Measured (Rev 21, §4.3)** — 0 cycles for a forced switch, ≤ 9 for a CPE-predicted one (hidden under the operand load): 0.02% of a median layer, ~11,000× below the lookahead threshold. Greedy becomes exactly optimal; the headline moves 1.0525× → 1.0526×. §3–§6 kept on the estimate |
| 15 | VersaAccel synthesises at 500 MHz in 1.29 mm² | **Replaced (Rev 21)** — run 1 was invalid (constant-folded Benes/ROM/CPE, fast corner). At the slow corner (§7.15): Fmax **~430 MHz** (2.33 ns path; met at 2.4 ns), 1.867 mm² at 2.5 ns; systolic ~475 MHz / 0.945 mm², SIGMA ~395 MHz / 1.551 mm². Needed §7.14's pipelined operand feed (was 5.57 ns, ~180 MHz) |
| 16 | The cost model's dense term matches the RTL | **New (Rev 14), qualified** — it is the ideal-utilisation bound; RTL/model is 1.035× on BERT and 1.187× on ResNet work-weighted, 16× on M=1 layers. Adding the RTL's fill term flips 2.4% of decisions and moves the headline 1.184× → 1.173×, inside its CI. Validates DMR only (§7.12) |
| 17 | Reconfigurability costs little area inside the PE | **New (Rev 14), fast corner** — VersaAccel's 256-PE array is 6% smaller than the whole systolic array; the Benes is 4,312 µm² (~0.3% of VersaAccel). VersaAccel's 2.29× total is its flip-flop buffers. CPE cost still unknown (§7.11) |
| 18 | VersaAccel vs SIGMA area/power | **Measured (Rev 21, §7.15)** — at 2.5 ns VersaAccel is 1.20× SIGMA's area (1.867 vs 1.551 mm²; SIGMA misses by 23 ps) and 8% faster (430 vs 395 MHz). Compute-phase energy 4.20 vs 12.78 pJ/MAC dense, 2.54 vs 4.96 pruned (SIGMA's test is small, so its figure is pessimistic); systolic 1.89 / 1.32 |
| 19 | Zero-skipping pays less on larger arrays; with §5.1 this leaves a 16×16–32×32 window | **New (Rev 17)** — realisable saving falls monotonically with tile height on every sparse workload (ResNet-50 37.0% → 19.2% → 9.2%, pruned BERT 66.1% → 23.6% → 2.7% at 4/16/64), while runtime selection is irrelevant below 16×16 (§2.7). Confirmed at 200 inputs on two samples, within 0.6 pts (B14) |
| 20 | A compile-time channel reordering raises the skip saving on pruned networks at zero cost | **New (Rev 17), verified exact, 200 inputs × 2 samples** — ResNet gain +7.47 ± 0.31 / +7.50 ± 0.22 per held-out image, positive on all 200 (min +6.4) (B14); +7.4 pts on ResNet-50 pruned90 (30.0% → 37.5%), +5.6 on BERT pruned90 (23.6% → 29.2%) at 16×16, 86% / 73% of the unconstrained bound; largest at 16–32. Rewritten networks re-run with outputs unchanged (≤1.3×10⁻⁵). No gain on dense or 2:4. **RTL-measured (B13):** on 48 real slices the accounting matches the RTL exactly (fixed +19 offset since Rev 20, +15 before; equal savings in every case), 245,760 outputs correct; measured gain +4.7 (ResNet) and +6.1 pts (BERT) at K ≤ 16 (§2.8; +4.8 / +6.2 on the Rev 19 RTL) |
| 22 | Per-layer tile shape chosen by sparsity structure | **New (Rev 17), built in RTL (B15), hardware cost unmeasured** — on the RTL's own square / 8×32 / 32×8 array at full layer size: pruned BERT 1.40× (TALL; the wrong orientation is 0.85×), pruned ResNet 1.15× (1.50× / 1.28× regrouped), exact on the RTL; orientation is not decidable from dimensions. Idealised five-shape study: at 256 PEs: 1.97× on pruned BERT, 1.21× on pruned ResNet-50 over fixed 16×16 (2.06× / 1.31× with regrouping), against 1.00× / 0.99× for shape-by-dimension-fit alone; ≤ 40 operands/cycle: 1.36× / 1.09×. RTL at all five shapes = accounting + (NC − 1), 0 errors. Reshaping cost (B15) open. **With the fill amortised (B10, §7.13)** the RTL's tall-tile gain on real slices is 1.26× (BERT) / 1.12× (ResNet), down from 1.37× / 1.26×: part of the per-tile-fill gain was the shorter fill. **Prior art (§2.10):** SPOTS (TACO 2022) already uses a tall array to skip zero activation rows, so the mechanism is credited to it; the new part is the per-layer, bidirectional, structure-driven choice and its quantification **Restated (Rev 21, §7.15, Genus batch 5):** the three-shape array costs +42.7–57.4% area and cannot meet 2.8 ns — a net loss everywhere. A square + tall array costs +20.6% and ~8% of clock; per area it gains 1.16× on pruned BERT (1.07× at its own clock) and loses on pruned ResNet (0.93×). **Claim: orientation chosen by sparsity structure is worth up to 1.40× (performance); a square + tall array is worth building only for weight-pruned Transformers.** |
| 23 | The per-tile skew fill, not the dataflow, caps the skip saving; operand load can be hidden | **New (Rev 18), RTL-built and verified (B10, B12)** — fill paid once per GEMM: skip saving on real pruned slices 11.2% → 24.3% (ResNet-50), 12.2% → 25.2% (BERT) at 16×16 on the Rev 20 RTL (24.8% / 25.7% before its 4-cycle feed pipeline), skip-on compute ~2× faster; activations streamed through a two-tile-row ring: 1.33× end to end on ResNet, within 1.6% of the load (1.5% before Rev 20). 13,215 suite checks + 3.7M element checks, 0 errors; cycles = exact model + constant. **Synthesis (§7.15): +4.5% area for B10, +2.6% for B12, both meet 2.5 ns.** Implication: the cost model's `Costcomp + Costmem` is the resident design; a streamed one is ≈ max (§7.13) |
| 24 | The findings transfer to two more model families | **New (Rev 18), B9; Rev 19 model** — MobileNetV2: selection 1.123× mean (pruned 1.10–1.20×, dense 1.000×; 1.147× as published), ReLU6 k-slices 12.6–182× i.i.d.; OPT-125m: 1.000× (homogeneous, like BERT), yet 27–55% realisable skip saving from its ReLU `fc2` alone. Not in the §3–§6 headline set (§2.11) |
| 21 | Runtime regrouping of activation rows | **Negative (Rev 17)** — ~+3 pts at best with a per-input greedy search; natural spatial order is already well grouped (shuffling loses 4–5 pts), and a calibrated fixed order is worse than natural (§2.8) |

---

## 10. Paper framing

> **Title:** *When Does Reconfigurability Pay? A Cost-Model Study of Adaptive Dataflow
> Selection for Sparse Accelerators*
>
> **Contributions:**
> 1. An open cycle model for four sparse dataflows, with RTL arithmetic verification of the
>    dense datapath (13,215 golden checks) and an explicit statement of what the RTL does and
>    does not substantiate (§7.9).
> 2. A **sparsity-structure result**: the sparse dataflows' advantage requires sparsity that is
>    *structured along the reduction dimension*, not merely plentiful. Shown twice,
>    independently — analytically, in that the cost model's nonzero-row estimator only
>    reproduces the published numbers on a row-structured example (§2.4), and in hardware, in
>    that a shared-operand array saves 0% on 50% unstructured zeros and 12.8–28.2% on the same
>    density structured (§7.7).
> 3. An **array-size threshold** (>=16x16 at BW >= 4 elem/cyc) below which activation sparsity
>    cannot change the optimal dataflow — so configuration selection is a compile-time problem
>    on the 8x8 arrays the literature commonly assumes. Bandwidth-starved designs are the
>    exception and are sensitive at every size (§5.1).
> 4. A **reconfiguration-cost threshold** (~265% of layer latency at 16x16 under the corrected
>    model; ~121% as published) above which lookahead scheduling is justified; real
>    interconnects sit ~400x below it, so greedy is within 0.013% of optimal.
> 5. A **workload characterisation**: 10 of 15 workloads (9 as published) have a single optimal schedule across
>    200 inputs, so runtime adaptation is needed only for input-dependent sparsity structure —
>    and a **negative result on learned prediction**: where adaptation *is* needed, an 8,961-
>    parameter MLP ties a zero-parameter constant on config selection despite a better
>    sparsity MAE (0.260 vs 0.295). On synthetic traces that MAE edge looked like 4.3x; on
>    real ones it is 12%, and it buys nothing either way.
> 6. A bounded negative result on N:M as a standalone dataflow (`P<=4` crossover), now
>    strengthened: as a 5th configuration it is actively harmful (−2.05%; −1.51% as published), and §7.7 shows the
>    hardware reason — a 2:4 pattern yields no skippable steps in a shared-operand array.
> 7. Speculative switching with verified flush behaviour, bounded to +1.3% at 10% error.
> 8. A **tile-height result** (Rev 17): the zero-skipping saving falls monotonically with
>    array size on every sparse workload, while runtime selection only matters from 16×16 up,
>    so the two benefits overlap only in a ~16×16–32×32 window (§2.7).
> 9. A **free, exact compile-time rewrite** (Rev 17): reordering weight output channels so
>    their zeros share tiles, under residual-stream and attention-head constraints, raises the
>    skip saving on pruned ResNet-50 and BERT by about a quarter (+7.4 and +5.6 pts at 16×16),
>    verified to leave network outputs unchanged (§2.8). Reordering weights to cut systolic
>    cycles along the reduction dimension, to trim the skew fill, is the group's own prior work
>    (ISQED 2026), and block-zero reordering is known (GMR, VARA). The contribution is the
>    output-channel, zero-hardware, k-slice, cross-layer-exact version (§2.10).
> 10. **Sparsity-structure-driven tile shape** (Rev 17): at fixed PE count, choosing each layer's
>    tile shape so its narrow edge faces the sparse operand gives 1.2–2.0× over fixed 16×16 on
>    sparse layers (1.1–1.5× under a bandwidth cap), beyond what dimension-fitting alone gives;
>    exact on the RTL at all five shapes (§2.9). The project's first architectural claim; its
>    hardware cost is the open item. The narrow-edge mechanism is SPOTS's (§2.10); the
>    contribution is choosing the orientation per layer from measured structure.
> 11. **A corrected cost model and what it leaves** (Rev 19, §3.0): two of the published model's
>     four dataflow formulas charge below the MAC lower bound on 31 and 29 of the 31 real layer
>     shapes (median 31× and 16×), and the original 1.184× selection headline rested on them. Held to the
>     bound, selection pays 1.052× overall and 1.11–1.28× on pruned CNNs; costed for a
>     shared-operand array like the RTL (fill + realisable structure), it pays nothing — the
>     choice is static, and the benefit of reconfigurability there is the tile geometry (8–10).

Contributions 8 and 9 are the novelty the scoping results were missing: 8 joins §2.5 and §5.1
into one operating-point argument, and 9 is a mechanism — one that needs no hardware and acts
exactly where 8 says the design should sit.

Contributions 2–5 are scoping results. They are less flashy than "2.04x speedup" and far more
defensible: each tells an architect *when not to build something*. Contribution 2 is the one
that generalises beyond VersaAccel — it applies to any accelerator that shares operands along
a dimension, which is every systolic design.

---

## 11. Remaining work — the road to submission

This section is the whole to-do list. It is organised by what each item *blocks*, not by
effort. Legend: **[BLOCKS]** = the paper cannot be submitted without it; **[STRENGTHENS]** =
a reviewer will ask for it and its absence costs a score point; **[OPTIONAL]** = do it if
time allows.

**Status line:** the *measurement* work is essentially done. 10 of 15 trace sets are real,
every headline number regenerates from repaired code, the RTL passes 13,215 golden checks,
and the claim disposition in §9 is honest. What does not exist yet is **a paper** — no
related work, no figures, no baseline comparison against anything except the VersaAccel paper
itself, and no README. That, plus synthesis, is what stands between here and a submission.

---

### 11.0 Done — do not redo

| # | Task | Landed |
|:-:|:--|:--|
| D1 | Re-profile ResNet-50 on real Imagenette with `IMAGENET1K_V2` | Rev 9 |
| D2 | Re-profile BERT on SST-2 with pretrained `bert-base-uncased`, eager attention | Rev 10 |
| D3 | Regenerate §3–§6 on real traces (all 10 model sets; 5 stress traces synthetic by design) | Rev 10 |
| D4 | Re-check the §2.4 estimator choice on real traces — `iid` holds, flip rate 1.5% → 3.2% | Rev 10 |
| D5 | Regenerate §5.2 after real BERT — config-match column did not move at all | Rev 10 |
| D6 | Tiling controller; real tiled GEMM verified across 6 shapes | Rev 5 |
| D7 | Output double buffer + wide drain; 292 → 37 cycles/tile | Rev 6 |
| D8 | Zero-skipping, and the structure result that came with it | Rev 7 |
| D9 | Genus flow *written* (`syn/genus/`) — not yet run against a tool | Rev 8 |
| D10 | Idea 4 constant audit + 20-point sensitivity sweep | Rev 4 |
| D11 | Predictor validation split (both v1 and v2 leaks closed) | Rev 4 |

---

### 11.1 Consistency and self-audit — no new experiments, ~1 day

These are defects a reviewer can find by reading the paper alone. They are cheap and they
cost the most credibility if missed.

| # | Task | Why it matters | Weight |
|:-:|:--|:--|:--|
| ~~**A1**~~ | **Done (Rev 17, §1.1).** Was: Resolve the `DMR` / "dense GEMM" / FULL-reduction naming inconsistency across `base_simulator`, the RTL, and this report. One config is called three things, and in one place "dense GEMM" refers to the *workload* rather than the configuration. | A reviewer who cannot tell which object a claim is about will assume the worst reading. | **[BLOCKS]** |
| ~~**A2**~~ | **Done (Rev 17)** — `experiments/audit_report_numbers.py` re-derives 27 headline numbers (32 since Rev 18, with §7.13) and checks each is in the report; it found the stale §4.1 W=1 value on first run. Was: Build a claim→evidence map: every number in §9 and §10 traced to a file in `results/` and to the command that produced it. Script it (`experiments/audit_report_numbers.py`) so the report can be re-checked after any rerun. | This audit already found three stale figures in §10 (−1.71% vs −1.51%, a 4.3x MAE that is now 12%, a 452-check count that is now 11,716). All are fixed in Rev 11; the point is to stop them recurring. | **[BLOCKS]** |
| ~~**A3**~~ | **Done (Rev 17)** — `greedy_gap_predicted` / `dp_gap_predicted` in `dp_scheduler`, `integrated_scheduler` and `scheduler_results.csv`; values unchanged. Was: Rename `dp_scheduler`'s "Greedy gap" / "DP gap" columns — in the code, the CSV headers and §3.1 — to something that cannot be read as scheduler optimality, e.g. `greedy_gap_under_predicted_sparsity`. | This is the Rev-10 measurement error. The name is why it went unnoticed for nine revisions. | **[BLOCKS]** |
| ~~**A4**~~ | **Done (Rev 17, §1.1 → §2.4).** Was: Document the split estimator decision in one place: `validate.py` pins `NNZ_ROW_MODEL='sqrt'` because the paper's Table II example is row-structured, while everything else runs `'iid'`. Right now that rationale lives only in a code comment and §2.4. | Two estimators in one codebase looks like a bug unless the paper says why. It is in fact the paper's second contribution. | **[BLOCKS]** |
| ~~**A5**~~ | **Done (Rev 17, §1.1).** Was: Terminology pass: define *decision-relevant layer*, *config match*, *homogeneous/heterogeneous workload* and *k-slice-structured sparsity* once, in §2, and use only those terms thereafter. | Three of these are load-bearing for the headline claims and are currently introduced where they are first needed. | **[STRENGTHENS]** |
| ~~**A6**~~ | **Done (Rev 17, §1.1).** Was: State `n` and the seed for every aggregate number (200 inputs per workload, 15 workloads, which are real and which synthetic). Several tables report a mean with no `n`. | | **[STRENGTHENS]** |
| **A7** | Re-read §0 and the revision preamble as a *reader* would, cold. It currently reads as a repair log, which is right for this document and wrong for the paper. | | **[OPTIONAL]** |
| ~~**A8**~~ | **Done (Rev 14) for §5.2** — over 10 seeds the MLP never beats the constant (93.4% ± 5.6 vs 96.2%); the committed tie is the best case. `predictor_seed_robustness.py`. §5.3 already runs 10 seeds. | — | **[DONE]** |

---

### 11.2 Evidence gaps — the work a reviewer will demand

| # | Task | Blocker | Weight |
|:-:|:--|:--|:--|
| ~~**B1**~~ | **Done (runs 2–4, §7.14–§7.15).** Area, Fmax and compute-phase energy for VersaAccel, systolic and SIGMA at the slow corner under identical constraints; option costs for B10/B12/B15. Batch 5 (reshape cost) pending. | — | **[DONE]** |
| ~~**B2**~~ | **Done (Rev 21, §4.3).** `tb_switch.sv`: forced switch 0 cycles for all 16 pairs, predicted switch 9 cycles under a 768-cycle load; `VERSAACCEL_SWITCH_COST=rtl` re-runs §4 and §3.2 (greedy exactly optimal; headline 1.0526×). | — | **[DONE]** |
| ~~**B1a**~~ | **Done (run 2, §7.14).** `area_hier` has `dist_net`, `nm_rom`, `red_net` and a CPE of 37,310 cells (45,133 µm² after Rev 20). | — | **[DONE]** |
| ~~**B1b**~~ | **Done (runs 2–4).** All synthesis at `slow_vdd1v0` (0.9 V, 125 °C). | — | **[DONE]** |
| ~~**B1c**~~ | **Done (run 4, §7.15).** Power over the compute phase from real-operand VCDs (`make_activity.py` → `power_windows.py`), 66–99.6% of flops annotated; energy per MAC 1.89 / 4.20 / 12.78 pJ dense. Clock-tree power needs CTS (not done, for any design). | — | **[DONE]** |
| **B1d** | **RTL done (Rev 16).** Buffers are submodules (`u_ibuf`, `u_wbuf`, `u_a_nz`, `u_b_nz`, `u_tbuf`), proven byte-identical in behaviour to the flat design. Still to do at comparison time: compare array + networks rather than top-level totals, since the systolic baseline has no deep buffers. | next Genus run | **[BLOCKS]** |
| **B1e** | Copy back `reports_*` as well as `outputs_*` after each run — power and `area_hier` live there. | — | **[STRENGTHENS]** |
| ~~**B1f**~~ | **Done (run 2).** SIGMA synthesises with 25,439 sequential cells, all 16 FANs and the Benes present — a fair baseline. | — | **[DONE]** |
| ~~**B1g**~~ | **Done (runs 2–4, §7.15).** Last met clock: systolic 2.5 ns (2.1 ns −3 ps), VersaAccel 2.4 ns, SIGMA 2.6 ns; critical paths 2.10 / 2.33 / 2.52 ns → ~475 / ~430 / ~395 MHz. | — | **[DONE]** |
| ~~**B3**~~ | **Done (Rev 14).** RTL closed form verified on 11 shapes, K=1–256. The model's DMR term is the ideal-utilisation bound: RTL/model 1.035× (BERT), 1.187× (ResNet), 16× on M=1 layers. Adding fill flips 2.4% of decisions; headline 1.184× → 1.173×. §7.12, `costmodel_vs_rtl.py`. | — | **[DONE]** |
| ~~**B11**~~ | **Decided (Rev 19), §3.0:** implementing B11 + B6 exposed that the published D-MR / D-M-R formulas charge below the MAC lower bound. Adopted for §3–§6: the **MAC-bounded model (B)**; the fill + structure terms (with the bound) are reported as model D — the cost on this RTL's array, where selection pays 1.000×. Published-model results kept in `results/as_published/`; `VERSAACCEL_COST_MODEL=nnz|mac|rtl`. Was: **Decide whether to adopt the RTL's fill term in the cost model.** It makes MM consistent with MV, D-MR and the RTL; it changes every §3–§6 number slightly (headline −0.011×) and flips 2.4% of decisions toward D-M-R, whose own formula has no RTL check. Recommended: adopt, and report both. | judgement | **[STRENGTHENS]** |
| ~~**B4**~~ | **Done in its feasible form (Rev 17, §3.4).** Per-layer selection vs each dataflow fixed everywhere, in Flexagon's (ASPLOS 2023) form: 1.05× under the Rev 19 model (1.21× as published) over the strongest fixed dataflow (Gustavson-style D-M-R) against Flexagon's 1.35× over GAMMA-like; 6–22× over weak fixed choices. A hardware-level comparison against a published sparse accelerator would need its workloads and technology — not attempted. Was: **Compare against prior art.** The only external comparison in the project is to the VersaAccel paper's own Table II. A cost-model paper needs at least a dense systolic baseline, SIGMA (already in the RTL), and one published sparse accelerator's reported numbers. | none | **[BLOCKS]** |
| ~~**B5**~~ | **Closed (Rev 17, §5.3b)** — at the decision level structure cannot change a configuration choice (0 of 125,000 relevant points), so the structure predictor's MAE edge buys nothing. Earlier: **Reopened (Rev 13).** Rev 12's "rescue failed" rested on one seed and 8 of 10 trace sets. Over 10 seeds: a structure predictor beats a per-layer table consistently (+14.3% ± 6.8 MAE, every seed positive) but by 0.0016 in skippable fraction; the density edge is not distinguishable from zero. Claim 5 stays withdrawn on §5.2's decision-level evidence. **Closing this needs B6**, then a decision-relevant comparison of a structure predictor against the per-layer table (§5.3). | B6 | **[STRENGTHENS]** |
| ~~**B5a**~~ | **Done (Rev 12); code rebuilt Rev 13, traces pending — see R0.** `kslice_structure()` added to the profiler and validated against the §7.7 patterns (returns 0.000 on unstructured and 2:4, 0.500 on k-structured — N1 and N2 reproduced in Python independently of the Verilog). All 10 real trace sets re-profiled with `a_kslice_zero`, `w_kslice_zero`, `skippable_frac`. Produced §2.5 as a side effect. | — | **[DONE]** |
| ~~**B6**~~ | **Decided (Rev 19), with B11 — see B11 and §3.0:** the structural term is model D's, reported as a finding; the paper's model is the MAC-bounded B. **Done (Rev 17, §5.3b)** — option (a) implemented and run: a structural term (dense compute, credit f·K/(K+NR+NC) to skip-capable configs). Headline on the 10 real sets 1.095× → 1.068× (1.058× with dense memory); 0 of 125,000 points are decision-relevant to f, so structure is a magnitude variable, not a decision variable. Adopting it as the paper's model is a decision alongside B11. Was: Decide how the cost model expresses structure dependence (§7.7): add a structure-aware sparsity term, restrict the claims to structured sparsity, or report the gap as a finding. | judgement | **[BLOCKS]** |
| ~~**B7**~~ | **Done (Rev 18): dropped.** §6.2 now states that no enhancer claim is made; claim 10 and N4 updated. Was: Resolve N:M: RTL-measure the D-M-R enhancement, or drop the claim entirely. | | **[STRENGTHENS]** |
| ~~**B8**~~ | **Done (Rev 17, §3.5).** Array × bandwidth × switch-cost grid: real workloads gain nothing at 8×8 (≤1.005×), most at 32×32 / BW 32 (1.088×); switch cost ×100 moves the headline ≤2.5% (Rev 19 model; as published ≤1.012×, 1.122×, ≤5%). Was: Operating-point sensitivity for the headline 1.184x: array size × bandwidth × switch cost. Parts of this exist (`array_size_sweep`, `switch_cost_sweep`); they have never been run as one grid. | none | **[STRENGTHENS]** |
| ~~**B9**~~ | **Done (Rev 18), §2.11** — MobileNetV2 and OPT-125m (decoder Transformer with ReLU FFN; chosen in place of a non-Transformer sequence model, which no small pretrained ReLU model offered) profiled dense + 50/70/90% pruned on the same inputs; the CNN/Transformer split, ReLU k-structure and structure-not-count all transfer. Kept out of the §3–§6 headline set. Was: Widen the workload set beyond ResNet-50 and BERT — one CNN with different layer shapes (MobileNet or EfficientNet) and one non-Transformer sequence model. 2 of 15 model families is thin for a workload-characterisation claim. | network | **[STRENGTHENS]** |
| ~~**B13**~~ | **Done (Rev 17).** 48 real slices through `tb_regroup.sv`: accounting exact (+15 fixed offset in every case and mode, +19 since Rev 20), regrouped C = permuted original C, 245,760 checks, 0 errors; measured +4.8 / +6.2 pts (+4.7 / +6.1 on the Rev 20 RTL). Was: *RTL-measure the §2.8 rewrite.* Take one pruned layer, apply its `wshared` order, and run original and reordered operands through the zero-skipping path (`tb_gemm_sparse`-style) at a shape the buffers hold; the measured cycle difference should match `tile_geometry.cycles()`. Turns claim 20 from accounted to measured. | none | **[STRENGTHENS]** |
| **B15** | **RTL done (Rev 17); synthesis pending.** `-DVERSAACCEL_RESHAPE`: square / 8×32 / 32×8, byte-identical when unused, 0 errors and exact accounting in all shapes; full-layer gain 1.40× pruned BERT, 1.15× pruned ResNet (1.50× / 1.28× regrouped). Remaining: Genus area/timing with vs without the define. Was: **Cost of a reshapeable array.** RTL for a 256-PE array that runs as 16×16, 8×32 or 32×8 (operand routing, drain, bitmap widths), verified with the existing GEMM benches at each shape, then synthesised to give the area/timing cost of §2.9's gain. Without it claim 22 is a performance-only result. | RTL + Genus | **[BLOCKS claim 22]** |
| ~~**B16**~~ | **Done (Rev 17, §2.9).** Operand budget instead of a cap: the RTL shapes keep their full gain at ≥ 40 operands/cycle and 1.12× (1.20× regrouped) on pruned BERT even at the square array's 32. Was: Model operand bandwidth properly: charge NR + NC operands/cycle against a buffer read-port budget in the cost model, rather than as a hard cap on allowed shapes. | none | **[STRENGTHENS]** |
| **B17** | **Four passes done (§2.10).** ISQED 2026 read in full (2026-11-07): it permutes the reduction dimension k inside each 4×4 tile with a hardware pre-processing block (+18% area) to trim the skew fill — a different axis and objective from §2.8's output-channel rewrite, and independent support for §2.7. Earlier passes: GMR, VARA, SMaT, SPOTS, Pool & Yu, Misam, Flexagon, MAERI. Remaining: a last check of ISCA/MICRO/HPCA/ASPLOS 2026 proceedings before submission. | none | **[STRENGTHENS]** |
| ~~**B14**~~ | **Done (Rev 17).** 200 images + 200 sentences on two disjoint samples: every number within 0.6 pts of the 16-input run, samples agree within 0.5; ResNet regrouping gain positive on every held-out image (min +6.4). Was: *Re-run §2.7–§2.8 at 200 inputs* (and with a second seed for the held-out split), matching §3's sample. The 16×16 baselines already match the 200-input traces to 0.01; the regrouping gains have not been measured at that scale. | none, ~2–3 h | **[STRENGTHENS]** |
| ~~**B10**~~ | **Done (Rev 18), §7.13** — `-DVERSAACCEL_AMORTISE`: tiles pipelined, per-PE clear/capture wavefront, two-bank output buffer; all 13,215 checks pass on it; cycles = an exact cycle model + 1 (+2 on the Rev 20 RTL, whose model carries the 3-cycle feed in the bank timing); +4.5% area at 2.5 ns (§7.15). Was: Amortise the skew fill across tiles, or state its cost as a fixed limitation. It is what caps the §7.7 sparsity saving, and the cap is currently presented as a property of the dataflow when it is a property of *this implementation*. | RTL work | **[OPTIONAL]** |
| ~~**B12**~~ | **Done (Rev 18), §7.13** — `-DVERSAACCEL_STREAM`: activations streamed through a double buffer of two tile-rows, overlapped with compute; GEMMs whose A exceeds the buffer now run. Was: Input double buffering / streamed operands. Everything is resident today (§7.10 item 3), so the model's bandwidth term is unexercised by the RTL. | RTL work | **[OPTIONAL]** |

**On B3:** this is the highest value-per-hour item in the list. It converts "an open cost
model" into "a cost model validated against RTL on the dense path", which is a materially
different claim and needs no tools the project does not already have.

**On B4:** without it the paper has no external anchor at all, and a reviewer cannot tell
whether the modelled baseline is a strong one or a straw man. This is the most likely single
cause of a reject.

---

### 11.3 Paper artifacts that do not exist yet

Nothing in this row exists in the repo today — no `.tex`, no `.bib`, no figures, no README.

| # | Task | Notes | Weight |
|:-:|:--|:--|:--|
| **C1** | **Related work + bibliography.** Minimum coverage: systolic/dense (TPU, Eyeriss v1/v2), flexible interconnect (SIGMA, MAERI), sparse dataflow (SCNN, ExTensor, Gamma, SparseTPU), N:M structured sparsity (Sparse Tensor Cores, NVIDIA 2:4), reconfigurable dataflow selection (Interstellar, Timeloop/Accelergy, CoSA), and VersaAccel itself. ~30–40 entries. | none exist | **[BLOCKS]** |
| **C2** | **Position against Timeloop/Accelergy explicitly.** They are the standard cost-modelling frameworks; a reviewer's first question is why this is not a Timeloop mapping study. The answer — that this models *reconfiguration between dataflows at runtime*, which Timeloop does not — needs to be in the intro, not the rebuttal. | | **[BLOCKS]** |
| **C3** | **Figures.** Six, generated from `results/*.csv` by a checked-in script: (1) optimal config vs sparsity decision map; (2) schedule stability across the 15 workloads; (3) the switch-cost threshold curve with the ~121% point and where real interconnects sit; (4) array-size × bandwidth sensitivity heatmap; (5) predictor vs zero-parameter baselines; (6) zero-skipping saving by sparsity *structure* at matched density. | zero figures today | **[BLOCKS]** |
| **C4** | **Architecture and RTL block diagrams** — the four configurations, and the `versaaccel_top` datapath with the tiling FSM, double buffer and skip machinery. | | **[BLOCKS]** |
| **C5** | **Write the paper.** Use §10's framing verbatim as the contribution list. Section map: Intro → Background (VersaAccel + dataflow taxonomy) → Cost model and its limits (§2) → Sparsity structure (§2.4 + §7.7, the strongest contribution) → Scheduling results (§3, §4) → When prediction is justified (§5) → N:M negative result (§6) → RTL and its scope (§7, incl. §7.9 verbatim) → Threats to validity → Conclusion. | | **[BLOCKS]** |
| **C6** | **Threats to validity**, as its own section, not a closing paragraph. Must contain: §7.9's scope statement; the cost model's stated limits (§2.1); the i.i.d.-vs-row-structured estimator choice (§2.4); the 5 synthetic stress traces; no synthesis, hence no area/power and estimated switch costs; single-technology, single-frequency. | | **[BLOCKS]** |
| **C7** | **Pick the venue.** *Deferred by decision — revisit after B1 lands.* It sets length and format, so it must be settled before C5 (writing) even though nothing earlier depends on it. With synthesis results in hand the option set widens; note only that an architecture venue judges against papers with measured hardware. | deferred | **[STRENGTHENS]** |
| **C8** | **Abstract and title.** §10's title is good. The abstract should lead with the sparsity-structure result (contribution 2), not with the selection speedup (1.052x under the corrected model, 1.184x as published) — the scoping results are what make this paper worth publishing, and a modest speedup in the abstract invites comparison against papers claiming more. | | **[BLOCKS]** |
| **C9** | Decide how to present the negative results. There are **seven**, not four, and they resolve into **two** theses rather than one. **Fully worked in §11.7** — take this decision before C5, because it sets the section order. | | **[BLOCKS]** |

---

### 11.4 Reproducibility and release

| # | Task | Notes | Weight |
|:-:|:--|:--|:--|
| ~~**R1**~~ | **Done (Rev 17)** — `README.md`: headline table, layout, reproduction, real vs synthetic, scope. Was: **`README.md`** — none exists. Repo layout, what each of the four ideas is, how to run everything, what is real vs synthetic. | | **[BLOCKS]** |
| **R2** | **`LICENSE`** — none exists. Needed before any public release. Check compatibility with the licences of `bert-base-uncased`, Imagenette and SST-2. | | **[BLOCKS]** |
| ~~**R3**~~ | **Done (Rev 17)** — exact pins, the torch nightly noted, iverilog 13.0 and Genus 21.10 recorded. Was: Pin `requirements.txt` with exact versions, and record the Python / torch / torchvision / transformers / datasets / iverilog versions these numbers were produced with. `transformers` 5 changed the attention default, which silently broke measurement once already. | | **[BLOCKS]** |
| ~~**R4**~~ | **Done (Rev 17)** — `run_all.sh` (`--quick` ~25 min, full ~3.5 h — ~50 min / ~4.5 h since Rev 18's B10/B12 steps — `--profile` to re-profile), ending in the A2 audit. Verified 2026-10-04: a `--quick` run regenerated every results file byte-identically (the only diff was the A3 column rename reaching `integrated_results.csv`), all 13,215 RTL checks passed, audit 27/27. Was: One `run_all.sh` that regenerates every number in the paper from a clean checkout, plus the expected runtime. | | **[STRENGTHENS]** |
| ~~**R5**~~ | **Done (Rev 17)** — three unseeded entry points seeded (`train_predictor.py`, and the synthetic paths of two profilers); all others already seeded; seeds per result in §1.1. Was: Seed and determinism audit across `experiments/` and `base_simulator/` — every stochastic path seeded and the seed recorded in the output CSV. | | **[STRENGTHENS]** |
| ~~**R6**~~ | **Done (Rev 18): ship the traces.** ~45 MB of per-layer statistics, no dataset content; `MANIFEST.sha256` + `experiments/verify_traces.py` (checksums and provenance of all 23 sets), first step of `run_all.sh`. Was: Decide trace distribution: ship the generated traces (large) or ship the profiling scripts plus the provenance stamps and expected checksums. The dataset itself must stay untracked. | | **[STRENGTHENS]** |
| **R7** | Artifact-evaluation package if the venue offers AE. A repo in this state scores well on AE, and AE badges are worth more than they cost here. | C7 | **[OPTIONAL]** |

---

### 11.5 Logistics

| # | Task | Notes | Weight |
|:-:|:--|:--|:--|
| **L1** | **Repository access.** This working copy cannot push to `Satyam137/AdaptAccel-Project` — the account in use lacks write access, and work is currently transferred by `git bundle`. Resolve before the artifact is referenced from a paper. | | **[BLOCKS]** |
| **L2** | Authorship, author order and affiliations. | | **[BLOCKS]** |
| **L3** | Dataset and model licensing/citation: Imagenette, GLUE/SST-2, `bert-base-uncased`, ResNet-50 `IMAGENET1K_V2`. Cite each; confirm redistribution terms before shipping traces (R6). | | **[BLOCKS]** |
| **L4** | Double-blind anonymisation of the repo link if the venue requires it. | C7 | **[STRENGTHENS]** |

---

| # | Task | Notes | Weight |
|:-:|:--|:--|:--|
| ~~**R0**~~ | **Done (Rev 13).** All 10 real trace sets re-profiled with torch 2.8 / transformers 5.17. ResNet scalar columns are **bit-identical** to Rev 10 across 43,200 rows; BERT differs by ≤8×10⁻⁵ on ≤48 of 14,600 rows. The whole §3–§6 pipeline reproduces — §3.1, §3.2/§3.3, §4, §5.1, §5.2, §2.4, §6.1, §2.5, §2.6 — except §5.3, whose Rev 12 conclusion was wrong (corrected), and §8.1's maximum, which is torch-version-sensitive (bound unaffected). | — | **[DONE]** |
| ~~**R9**~~ | **Done (Rev 18)** — a retrain writes `trained_models/retrained/` unless `--replace`, so the committed weights stay a fixed input and pipeline order no longer matters; two consecutive retrains give bit-identical weights (seeded since R5). Appendix updated. Was: **Make the pipeline order-independent and retraining reproducible.** The appendix runs `dp_scheduler.py` before `train_predictor.py`, so a second pass of the documented pipeline uses different weights than the first; and retraining under a new torch changes the weights file. Either pin the committed weights as an input or train first and seed everything. | — | **[STRENGTHENS]** |
| **R8** | **Keep the repository somewhere durable.** It lived under `/private/tmp` and was wiped. It is now in `~/Documents/AdaptAccel-Project`; push it to the shared remote so a single disk is not the only copy. | — | **[BLOCKS]** |

### 11.6 Suggested order

1. **A1–A4**, then **B3** and **B8**. All are unblocked, all are cheap, and B3 upgrades the
   central contribution.
2. **C9 (§11.7)** — settle the two-thesis structure. It decides the section order, so every
   writing task downstream depends on it, and it costs an afternoon.
3. **C1, C2** — related work, in parallel with anything else. It is the longest lead-time item
   that nobody else can start for you.
4. **B5, B6, B7** — the three judgement calls. They determine which sections exist.
5. **C3, C4** — figures, once B3/B8 have settled the numbers.
6. **C5, C6, C8** — write it, in the order §11.7 sets.
7. **R1–R3, L1–L3** — before submission, not after.
8. **B1a–B1e, then B2** on the next Genus session, and slot the results into §7 and §4. C7
   (venue) reopens at that point.

### 11.7 Resolving C9 — the two theses

C9 asked how to present the negative results without the paper reading as a failed project.
This subsection is the answer, and it is a structural decision that should be taken **before**
C5 (writing), because it determines the section order.

#### The problem is larger than C9 stated

C9 counted four negatives. There are **seven**:

| # | Negative result | Section |
|:-:|:--|:--|
| N1 | Zero-skipping saves 0% on 50% *unstructured* zeros | §7.7 |
| N2 | 2:4 structured sparsity also saves 0% in a shared-operand array | §7.7 |
| N3 | N:M is uncompetitive as a standalone 5th configuration (`P<=4` crossover) | §6.1 |
| N4 | ~~N:M as a D-M-R enhancer~~ — **dropped** (B7): no claim, so no negative to report. (The −1.51% previously shown here is §6.1's, i.e. N3.) | §6.2 |
| N5 | An 8,961-parameter MLP ties a zero-parameter constant on config selection | §5.2 |
| N6 | Lookahead/DP scheduling buys nothing — greedy is within 0.013% of Oracle (0.003% as published) | §4 |
| N7 | Speculation never beats the Execution-Oracle (+0.009%) | §8 |

Plus a scoping result that reads as an eighth: below 16x16, activation sparsity *provably
cannot* change the optimal configuration (§5.1).

#### One thesis does not cover them

The thesis proposed in Rev 11 — *"sparsity must be structured along the reduction dimension"* —
covers N1–N3 cleanly (N4 is dropped) and N5 arguably. It does not cover N6 or N7 at all, which are about the
shape of the decision landscape, not about sparsity. Forcing all seven under one banner would
be the kind of overreach this audit exists to prevent.

**There are two theses, and the paper is stronger for saying so**, because together they are
the answer to the question in the title.

#### Thesis 1 — Density is not the decision variable; structure along the reduction dimension is

> Covers N1, N2, N3, N5, and claim 14 (§2.4).

Every one of these is the same mechanism seen from a different side. A systolic array shares
each operand along a row and a column, so a zero is only skippable when an *entire `k`-slice*
of the tile is zero. Therefore:

- 50% zeros scattered i.i.d. give a vanishing probability of an all-zero slice → **N1**.
- A 2:4 pattern structures sparsity *within groups along a row* — the wrong axis. For 16 rows
  the probability of an all-zero k-slice is `2^-16` → **N2**, and N2 is the hardware reason for
  **N3**. (N4, the enhancer, is dropped — §6.2.)
- The cost model's `M·√density` nonzero-row estimator reproduces VersaAccel's published sparse
  counts **only because the paper's Table II example is row-structured** (§2.4) → claim 14.
- And **N5 follows as a prediction, not a coincidence**: the MLP's regression target is a
  *scalar density per layer*. If density is not the decision variable, then a density predictor
  is predicting the wrong quantity, and its excellent MAE buying zero decisions is exactly what
  the thesis says should happen.

This last point is the most valuable move available in the whole rewrite. It converts N5 from
"our predictor did not work" — which reads as a failure — into "our predictor predicted the
wrong variable, and the theory says which variable it should have predicted." See B5 below.

**Thesis 1 now has an empirical leg too.** §2.5 measures the structure of real workloads
rather than assuming it: real ResNet activations are 14–29x more `k`-structured than i.i.d. at
the same density, and still far below what the `nnz` term credits. So the thesis is not merely
"the cost model is wrong" — it is that *both* default assumptions are wrong in opposite
directions, and the skippable fraction is a measurable workload property.

**The positive control is what makes Thesis 1 a contribution rather than a broken
implementation.** 50% *k-slice-structured* zeros save **12.8% at K=16 and 28.2% at K=64**,
matching the `f·K/(K+NR+NC)` ceiling. The paper must present the 0% and the 28.2% side by side
at matched density. A negative result with a mechanism *and* a positive control is evidence; a
negative result alone is an unfinished experiment.

Thesis 1 also has the rare property of being shown twice by independent methods — analytically
from the cost model (§2.4) and in hardware from the RTL (§7.7), which were developed separately
and agreed. Say that explicitly.

#### Thesis 2 — The configuration-selection landscape is flat, so cheap policies suffice

> Covers N6, N7, §5.1 and §3.3.

- 10 of 15 workloads have a single optimal schedule across 200 inputs (§3.3; 9 as published).
- Where the schedule does vary, greedy is within **0.013%** of Oracle, and lookahead only pays
  above **~265%** of layer latency — roughly 400x above where real interconnects sit (§4; 0.003%
  and ~121% / 100x under the published model).
- Speculation therefore bounds the cost of being wrong rather than adding speed (§8, N7).
- And below 16x16 the landscape is not merely flat but *degenerate*: the activation term cannot
  cross a configuration boundary at all (§5.1), making selection a compile-time problem on the
  8x8 arrays the literature commonly assumes.

Read this way N6 and N7 stop being disappointments and become the **design rule**: build the
cheap selector, do not build the clever one. Contribution 4 in §10 already phrases the
threshold this way; the rest should follow.

#### How the two compose — this is the paper's argument

> Reconfigurability pays **only** when sparsity is structured along the reduction dimension
> (Thesis 1) **and** the array is large enough for that structure to move the optimum
> (Thesis 2's boundary). When both hold, the mechanism that captures the benefit is trivially
> cheap (Thesis 2). When either fails, reconfigurability is overhead.

That is a complete answer to the title, assembled entirely from results that survived the
audit. Neither thesis needs a speedup number to stand up.

#### Concrete actions

| # | Action |
|:-:|:--|
| **C9a** | Reorganise the results sections by thesis, not by idea/module. Current order follows the repo's four "ideas", which scatters N1–N3 across §6 and §7 and makes the reader tally negatives. Proposed: Thesis 1 (§2.4 + §7.7 + §6) → Thesis 2 (§3.3 + §4 + §5.1 + §8) → N5 as the bridge between them. |
| **C9b** | Give every negative a **mechanism sentence** in its first paragraph. N1/N2 have one (shared operands); N3 inherits it; N6 needs one (the cost difference between configurations is large relative to switch cost, so the greedy choice is almost always the Oracle choice); N7 inherits from N6. |
| **C9c** | Present N1 and the 28.2% k-structured saving **in the same figure** at matched density. This is figure 6 in C3 and it is the single most important figure in the paper. |
| **C9d** | Phrase every negative as a bound or a rule, never as an absence. "Greedy is within 0.003% of Oracle" not "DP did not help". "N:M yields no skippable steps in a shared-operand array" not "N:M did not work". |
| **C9e** | State in the intro that the paper's contributions are scoping results, so a reader is not waiting for a speedup that never arrives. A reader who expects a speedup reads seven negatives as failure; a reader told the paper answers *when not to build this* reads the same seven as the result. |
| **C9f** | Keep N5 in the paper. Withdrawing a claim honestly, with a mechanism for why it failed and a stated alternative, is worth more than quietly deleting it — and it is the only place the paper demonstrates its own methodology working against itself. |

---

### 11.8 Which negatives can move, and which cannot

Asked directly: *is there something we can run to fix the bad results?* The triage matters,
because effort spent on the six that cannot move is effort taken from the one that can.

| | Result | Can a run change it? | Why |
|:-:|:--|:--|:--|
| N1 | Unstructured 50% → 0% saving | **No** | A shared-operand array can only skip a whole `k`-slice. The probability that i.i.d. zeros produce one is vanishing — this is arithmetic, not a tuning choice. |
| N2 | 2:4 → 0% saving | **No** | Same mechanism; `2^-16` for 16 rows. |
| N3 | N:M uncompetitive standalone | **No** | Inherits N2. |
| N4 | ~~N:M as enhancer harmful~~ | — | Dropped (B7, §6.2). |
| N6 | Greedy within 0.013% of DP (0.003% as published) | **No** | Verified across the whole switch-cost sweep, not at one point. That it holds over the entire plausible range is what makes it a design rule. |
| N7 | Speculation never beats the oracle | **No** | Verified across 20 parameter settings (§8.1). An oracle bound cannot be beaten by tuning; that is what makes it a bound. |
| **N5** | **MLP ties a zero-parameter constant** | **Partly** | The decision-level result (§5.2) stands. The wrong-variable rescue is open: a structure predictor beats a per-layer table in every seed, by an amount (0.0016 in skippable fraction) unlikely to flip a decision — but that last step is untested until B6 (§5.3). |

The six that cannot move are not defects awaiting a fix. Each has a mechanism, each was
verified across a range rather than at a point, and two of them (N1/N2 and claim 14) were
reached independently by hardware and by analysis. **They are the paper.** What would change
them is a different dataflow — Gustavson row-stationary, which skips along rows and therefore
*does* benefit from unstructured sparsity — and that is a second accelerator, explicitly
scoped out in §7.9.

**The failure mode to avoid.** This project began with three headline numbers that were
artifacts rather than measurements (§0). Re-running experiments until a negative turns positive
is how that happened. The rule for the remaining work: a number may be regenerated because the
*inputs* changed — as in Revisions 9 and 10, where the traces became real — never because the
*output* was unwelcome.

**Outcome, as corrected in Rev 13.** B5a and B5 were run in Rev 12, which reported that the
rescue failed. It did not: that verdict came from one seed on 8 of 10 trace sets. Over 10 seeds
the structure predictor beats a per-layer table consistently but by a tiny margin, so the rescue
is **open** and needs B6 to settle (§5.3). The Rev 12 text follows for the record.

*(Rev 12:)* The rescue failed, exactly as §11.8 said it might,
and the prediction that this would be the *stronger* outcome held: §5.3 now rules out the
obvious objection to §5.2 rather than leaving it open. B5a also produced an unexpected positive
result along the way (§2.5), which is the empirical leg Thesis 1 was missing.

**Five of the six remaining negatives are confirmed as findings** (N1–N3, N6, N7; N4 dropped by B7); N5 holds at the decision level (§5.2) with its rescue still open (§5.3). None is a defect awaiting a fix.

**What is worth running**, in value order:

1. ~~**B3**~~ — done in Rev 14 (§7.12). **B1f** (a fair SIGMA) is now the highest-value
   unblocked RTL item: without it the paper has no flexible-accelerator comparison.
2. **B8** — the array × bandwidth × switch-cost grid. Cannot flip a negative, but may locate an
   operating point where reconfigurability pays more than the headline (1.052x; 1.184x as published), which would
   sharpen Thesis 2's boundary rather than soften it.

---

## Appendix — reproducing these numbers

```bash
pip install -r requirements.txt

python -m base_simulator.validate                  # cost model vs paper Table II
python idea1_adaptaccel/scheduler/dp_scheduler.py  # section 3.1, 4.1
python experiments/scheduler_variance.py           # section 3.2, 3.3
python experiments/switch_cost_sweep.py            # section 4.2
python experiments/array_size_sweep.py             # section 5.1
python experiments/predictor_baselines.py          # section 5.2
python experiments/nnz_rows_sensitivity.py        # section 2.4
python experiments/structure_predictor.py          # section 5.3 -- the B5 rescue test, 10 seeds
python experiments/structure_value.py              # section 2.6 -- structure in cycles
python experiments/predictor_seed_robustness.py    # section 5.2 -- 10 seeds (A8), ~15 min
python experiments/costmodel_vs_rtl.py             # section 7.12 -- cost model vs RTL (B3)
python experiments/costmodel_vs_rtl.py --headline  #   + headline with/without fill, ~20 min
python experiments/regroup_rtl.py                  # section 2.8 on the RTL (B13), ~5 min
python experiments/tile_shape.py                   # section 2.9, ~1 min
python experiments/tile_shape_rtl.py               # section 2.9 on the RTL at 5 shapes, ~1 min
python experiments/tile_geometry.py --images 200 --texts 200 --chunk 10 --seed 0   # B14, ~45 min
python experiments/tile_geometry.py --images 16 --texts 24   # sections 2.7, 2.8, ~6 min;
                                                   #   re-runs the real models, asserts the
                                                   #   regrouped networks' outputs are unchanged
python -m idea2_veriflex.integrated_scheduler      # section 6.1
```

Predictor retraining (optional; R9). The committed weights in
`idea1_adaptaccel/predictor/trained_models/` are an input to sections 3–6, so a retrain writes
`trained_models/retrained/` and leaves them alone; it is seeded and reproduces bit-for-bit.
`--replace` overwrites the committed weights, after which sections 3–6 must be re-run.

```bash
python -m idea1_adaptaccel.predictor.train_predictor            # -> trained_models/retrained/
```

CPE training and export (section 7.2):

```bash
python -m idea2_veriflex.train_cpe_classifier
```

RTL (sections 7.1-7.7):

```bash
idea2_veriflex/rtl/scripts/run_all_tb.sh        # whole testbench suite
idea2_veriflex/rtl/scripts/run_gemm_shapes.sh   # tiled GEMM across 6 shapes, both bus widths
idea2_veriflex/rtl/scripts/run_sparse_sweep.sh  # zero-skipping vs sparsity pattern, 2 shapes
# tb_layer_config (in run_all_tb.sh): routing / N:M / CPE pins reach their logic (section 7.11)
```

Speculation sensitivity (section 8.1):

```bash
python idea4_speculative/speculative_engine.py --sensitivity
```

Re-profiling (sections 2.3, 5.2). Needs network access:

```bash
python -m profiler.profile_resnet --download            # real images + pretrained
python idea2_veriflex/profile_nm_models.py --download --dataset sst2   # BERT half needs a text source
python -m profiler.profile_bert --dataset sst2          # needs `transformers`

# Rev 13 environment: a venv on python.org 3.11 reusing its torch 2.8, plus
#   pip install "transformers==5.17.0" datasets
# Reproduces every section except section 8.1's maximum (torch-sensitive, see 8.1).
# python.org installs on macOS may need a CA bundle first:
export SSL_CERT_FILE=$(python3 -c "import certifi;print(certifi.where())")
```

RTL golden vectors (section 7), from `idea2_veriflex/rtl`:

```bash
iverilog -g2012 -o /tmp/dp common/versaaccel_pkg.sv versaaccel_rtl/benes_switch.sv \
  versaaccel_rtl/distribution_network.sv versaaccel_rtl/reduction_network.sv \
  testbenches/tb_datapath_golden.sv && /tmp/dp
```

```bash
iverilog -g2012 -o /tmp/cpe -DCPE_WEIGHT_DIR='"'"$PWD"/weights'"' \
  common/versaaccel_pkg.sv versaaccel_rtl/config_prediction_engine.sv \
  testbenches/tb_cpe_golden.sv && /tmp/cpe
```

Set `NR`/`NC` in `common/versaaccel_pkg.sv` to re-run any of these at 8x8.

`VERSAACCEL_ARRAY=8` on any Python command reproduces the pre-scoping numbers. Results land
in `results/`; pre-fix CSVs are archived under `results/pre_fix/`.
