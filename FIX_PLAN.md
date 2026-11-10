# AdaptAccel / VeriFlex — Fix Plan and Path to a Submittable Paper

**Status:** drafted 2026-09-17, after a full read-and-run audit of the repo at `18ba4e9`.
**Scope:** fix the concrete defects, re-derive every reported number from code, then rebuild the
paper narrative on what survives.

> **As of Revision 11 this document is a historical repair log.** The authoritative list of
> outstanding work is **§11 of `PROJECT_REPORT.md`**, which covers consistency, evidence gaps,
> the paper artifacts that do not yet exist, reproducibility and submission logistics. Start
> there; read this for how the defects were found and fixed.

---

## 0. Read this first

The audit found that **three of the paper's headline numbers are artifacts, not measurements.**
This is not a style problem — it changes what the paper can claim:

1. **"NM gives 25% cycle reduction"** is the literal sum `0.12 + 0.08 + 0.05` hardcoded in
   `compute_nm_discount()`. It is multiplied by `nm_compatibility` (1.0 by construction in the
   `*_nm24` traces) and `sparsity_factor` (1.0), so it always prints exactly `25.00%`.
   Nothing is simulated. **This claim must be withdrawn or backed by RTL.**

2. **"DP lookahead beats Greedy"** is contradicted by the repo's own ablation, which reports
   `W=1` at a **+0.0002%** gap versus Oracle. Reconfiguration costs 100–250 cycles against a
   median layer cost of ~26,600 — under 1%. Lookahead cannot pay off at these parameters.
   `dp_scheduler.py` then prints a *hardcoded* conclusion sentence regardless of the numbers.

3. **"99.2% config match"** is beaten by trivial baselines the repo never runs:

   | Predictor | MAE | Config match |
   |:---|:---:|:---:|
   | Constant 0.0 | 0.299 | **91.2%** |
   | Persistence (copy previous layer) | 0.176 | **98.4%** |
   | Trained MLP (8,961 params) | 0.062 | 99.2% |

   A *constant* clears the stated ">=90% target". The metric is saturated because
   `combined_sp = max(predicted, weight_sparsity)` and weight sparsity is known exactly —
   for 70% of test rows the prediction is masked entirely.

**The good news:** there is a real, defensible paper here, and it is *stronger* than the current
framing. See §5. The honest finding — *at realistic reconfiguration costs, per-layer config
switching needs neither lookahead nor a learned predictor* — is a publishable result with a clean
sensitivity analysis behind it. It just isn't the result the current draft claims.

### Ground rules for every fix below

- **No number appears in the paper that a script does not regenerate.** Every table gets a
  `make results` provenance.
- **No hardcoded conclusions.** If a script prints an interpretation, it must be computed.
- **Every headline metric ships with its null baseline.** If a constant matches the method, say so.
- **Estimates are labelled as estimates** in the artifact that produces them, not only in prose.

---

## 0b. FINDING F1 — RESOLVED 2026-09-17 in favour of option (b): scope to >=16x16.

> **Decision taken.** The project now targets 16x16 and above
> (`base_simulator/arch_config.py`). E3 confirmed the premise is operating-point
> dependent, not architectural. On the decision-relevant subset the trained MLP
> scores 100% where a constant scores 0% and persistence 66.7%, so the predictor
> is justified at the scoped size. Original analysis retained below for the record.

Fixing the operand mapping (B5) exposed something the old bug was hiding.

The project's stated premise is:

> *"The optimal choice depends on the input activation sparsity, which is unknown
> until runtime and changes per input."*

**Under the corrected cost model, that premise does not hold on this workload set.**

`experiments/predictor_baselines.py` now reports **100.0% config match for a
constant-zero predictor** — and for every other baseline. Sweeping activation
sparsity from 0.00 to 0.99 on a representative pruned layer:

| act. sparsity | operator | DMR | D-MR | DM-R | D-M-R | best |
|:--:|:--:|--:|--:|--:|--:|:--|
| 0.00 | SpMM | 18,955,469 | 45,383,757 | 45,456,816 | **109,320** | D-M-R |
| 0.50 | SpMSpM | 18,955,469 | 32,113,101 | 22,855,216 | **240,989** | D-M-R |
| 0.95 | SpMSpM | 18,889,114 | 10,164,506 | 2,304,621 | **65,434** | D-M-R |

Costs *do* respond to activation sparsity — but `D-M-R` wins at every level, so
the **argmin never moves**. Config choice is driven by weight sparsity and layer
dimensions, both of which are **known statically at deploy time**.

Why the old code appeared to need a predictor: `sparsity_A = max(activation, weight)`
funnelled activation sparsity into the operand that *does* drive dataflow choice.
That was dimensionally wrong, and it was the only mechanism making the predictor
look load-bearing.

### This forces a decision before any more work on Idea 1

- **(a) Accept it and reframe.** The honest claim becomes: *config selection for
  this accelerator is a static, compile-time problem; runtime prediction is
  unnecessary.* Combined with E2 (greedy suffices), Idea 1 collapses into a
  **compile-time config assignment pass** — simpler, and a legitimate result.
- **(b) Find a regime where the premise holds.** Run E3 (array size, bandwidth):
  D-M-R's dominance may be an artefact of the 8x8 / BW=32 operating point.
  At small arrays or constrained bandwidth the ranking may genuinely change with
  activation sparsity. **If such a regime exists, the predictor is justified there
  and nowhere else — and the paper must say so.**
- **(c) The cost model is too coarse.** Activation sparsity plausibly affects
  MM/SpMM in real hardware (zero-skipping in the distribution network), but only
  `SpMSpM` reads `nnz_B` in `config_modes.py`. If so this is a **modelling gap**,
  not a finding, and must be fixed before the premise can be tested at all.

**Recommendation: run E3 first (cheap), then choose.** Do not write the
predictor section until this is settled — it determines whether Idea 1 has a
runtime component at all.

---

## 1. Phase 0 — Reproducibility infrastructure

Nothing below is verifiable until the repo runs from a clean checkout.

| ID | Task | File(s) |
|:--|:--|:--|
| R1 | Add `requirements.txt` (`torch`, `torchvision`, `transformers`, `numpy`, `pandas`, `openpyxl`) with pinned versions | new |
| R2 | Add `README.md`: what each idea is, exact run order, expected runtime | new |
| R3 | Add `.gitignore`; `git rm -r --cached` the committed `__pycache__/` dirs | repo root |
| R4 | Add `Makefile` with `make traces`, `make predictor`, `make schedule`, `make rtl`, `make results` | new |
| R5 | Pin all RNG seeds and record them in output CSVs | all generators |

**Exit criterion:** `git clone && pip install -r requirements.txt && make results` reproduces every
CSV in the repo byte-for-byte.

---

## 2. Phase 1 — Correctness bugs (P0: these invalidate results)

### B1. Negative sparsity in generated traces — 2,017 rows

Sparsity is a fraction of zeros; it cannot be negative. Gaussian noise is added without clipping.

- `idea1_adaptaccel/profiler/generate_messy_traces.py:74` — GCN, **496 bad rows**, min `-0.0760`
- `idea2_veriflex/generate_switching_traces.py:123` — switching_stress, **1,521 bad rows**, min `-0.0333`

The sibling function at `generate_messy_traces.py:131` *does* clip correctly — copy that pattern.

```python
# before
'weight_sparsity': round(weight_sp + np.random.normal(0, 0.02), 4),
# after
'weight_sparsity': round(float(np.clip(weight_sp + np.random.normal(0, 0.02), 0.0, 1.0)), 4),
```

**Then regenerate all affected traces and re-run every downstream consumer.**
Add a validation assert in `load_all_traces()` that rejects any sparsity outside `[0, 1]`.

### B2. `base_simulator/validate.py` does not run on a clean checkout

Imports `from simulator.versaaccel import ...`; the package is `base_simulator`. Stale rename.
Fix the import and the usage docstring (which says `cd project/software`).

### B3. Prefetch model: dead branch + free first fetch

`prefetch_pipeline.py`, `compute_prefetch_time()`:

- `INPUT_BUF_ELEMENTS = 65536`, `TILE_BUF = 32768`. The weight path tests
  `<= INPUT_BUF_ELEMENTS` **before** `<= TILE_BUF`, so the second branch is **unreachable**.
- The first branch assigns **zero** prefetch cost to anything fitting the buffer, commented
  "no fetch needed" — but data must still be fetched from off-chip at least once.

Together these zero out the cost for all small layers, which is why `overhead_hidden_pct` is
100.0% on 13 of 15 workloads. The model computes an overhead and then declares it hidden.

**Fix:** reorder the branches; charge a real cold-fetch cost for first use of a weight tile;
distinguish *resident* (already on-chip from a previous layer) from *fits-in-buffer*.

### B4. Static baseline is scored on a different cost function

`dp_scheduler.py:395` sums compute costs with **no switch cost**, while every scheduler pays a
switch from `initial_config='DMR'`. This is why `static_gap` goes **negative** (-0.003% on BERT):
the non-adaptive baseline beats the supposed optimum, which is impossible under one cost function.

**Fix:** charge the static baseline its one initial switch, or exempt all methods from the initial
switch. Either is defensible; the current asymmetry is not. **This changes the headline speedup.**

### B5. Weight sparsity is applied to the wrong operand

`get_layer_costs()` folds weight sparsity into `sparsity_A` via `max(activation_sp, weight_sp)`
and never sets `sparsity_B`. In these traces **A is the activation matrix (M x K)** and
**B is the weight matrix (K x P)**. Weight sparsity is being attributed to the activations.

**Fix:** pass `sparsity_A=activation_sparsity, sparsity_B=weight_sparsity`. Re-validate the
`SpMSpM` path, which is the only config path that reads `sparsity_B` today.
**This changes every cost number in the repo** — do it early, before re-measuring anything.

### B6. `dp_stress` does not do what its docstring says

The docstring promises outliers use "the SpMM operator"; `generate_messy_traces.py:219` hardcodes
`'op_type': 'MM'` for every row. It happens to still work because `get_layer_costs()` ignores the
CSV's `op_type` and re-derives it from `weight_sparsity` — which means **the `op_type` column is
dead data everywhere**. Either honour it or drop it; document which.

### B7. RTL does not compile

Of three testbenches, only the *baseline* systolic array elaborates — and it fails one of its own
tests (3 PASS, 1 FAIL).

| Site | Defect |
|:--|:--|
| `config_prediction_engine.sv:79,104,133` | `{8{b1[k][7]}, b1[k]}` — replication inside a concatenation needs its own braces: `{{8{b1[k][7]}}, b1[k]}`. Genuine syntax error, rejected by every compiler. |
| `versaaccel_top.sv:121` | assignment requires an explicit cast |
| `versaaccel_top.sv:156,157` | port array element types incompatible (`distribution_network` ports 2, 3) |
| `sigma_top.sv:50,51` | same array-type mismatch |
| `tb_systolic.sv` | Test 2 "Weight loading complete" fails |

**Verify with `iverilog -g2012` before Vivado** — it is free, local, and catches all of the above.

### B8. The CPE is a constant function  -- RESOLVED 2026-09-18

> Resolved. `train_cpe_classifier.py` trains the 9->16->16->5 topology the RTL is
> shaped for (labels = the cost model's optimal config), quantises to INT8 with
> quantisation-aware fine-tuning, and exports .hex files the RTL loads via
> `$readmemh`. INT8 accuracy 0.930 against a 0.704 majority baseline;
> `tb_cpe_golden` confirms all 64 vectors bit-exact vs the Python model.
> Required adding per-layer requantisation shifts and widening the accumulators
> from 16 to 24 bits (the output layer can reach 258,190).


`config_prediction_engine.sv:51-62` initialises **all weights to `8'sd1` and all biases to `0`**.
With identical weights on every output neuron the argmax is degenerate. The comment admits it is a
placeholder.

Worse, the shapes do not match the thing it is supposed to implement:

| | Python predictor | RTL CPE |
|:--|:--|:--|
| Topology | 9 -> 64 -> 64 -> 64 -> 1 | 9 -> 16 -> 16 -> 5 |
| Head | sigmoid regression | argmax classification |
| Weights | `predictor_weights_v2.npz` | hardcoded `1` |

Nothing anywhere does a `$readmemh`. **The exported `.npz` is never loaded by any RTL.**

**Fix:** add a quantisation export step (`export_weights_hex.py`) writing INT8 `.hex` files, load
them with `$readmemh`, and make the RTL topology match the trained network — or retrain a small
classifier that matches the RTL. Then verify RTL argmax against the Python model on the trace set.

---

## 3. Phase 2 — Methodology bugs (P1: these inflate results)

### M1. Both predictors early-stop on the test set  -- RESOLVED

> Resolved for v1 (test MAE 0.0615 -> 0.0626) and v2 (~0.0506 -> 0.1285).
> v2's correction is far larger because its test set includes switching_stress.


`train_predictor.py` and `train_predictor_v2.py:280,289-291,320` compute MAE on `test_loader` every
epoch, keep the best-scoring checkpoint, then report final metrics **on that same set**. There is no
validation split. The reported MAE is a minimum-over-N-epochs — optimistically biased by
construction.

**Fix:** three-way split (train / val / test). Early-stop on **val**. Touch test exactly once.
Expect the reported MAE to get worse; that is the point.

### M2. No null baselines

Add `constant-zero`, `constant-mean`, and `persistence` to the evaluation table permanently. If the
MLP does not clear persistence by a margin that matters, the paper says so.

### M3. The hardcoded thumb on the scale

`integrated_scheduler.py:116-118` multiplies `D-M-R` cost by `1.12` and `DM-R` by `1.06`
("Benes overhead", unmeasured) — **only in Idea 2's scoring**. Idea 1's identical function applies
no such factor, so the two ideas score the same workload on different cost functions.

**Fix:** delete the factors, or derive them from synthesis and apply them **uniformly in
`base_simulator`** so both ideas inherit them. Note NM loses anyway, with or without.

### M4. The hardcoded conclusion

`dp_scheduler.py:549` prints "Conclusion: W=5 matches Oracle on all workloads tested" unconditionally.
**Fix:** compute the statement, or delete it.

### M5. The phase controller is numerically inert

`PHASE_BIAS_WEIGHT = 50` cycles against ~26,000-cycle layers — a 0.19% nudge that cannot flip a
decision. Measured effect: **+0.00%**. Its hardcoded preferences also contradict the cost model
(PREFILL prefers `DMR` at confidence 0.8; `DMR` is ~5.5x *worse* than `D-M-R` on dense 64x64 MM).

**Decision required:** either (a) make the bias meaningful and show it helps, (b) demote the phase
controller to a *diagnostic* contribution with no performance claim, or (c) cut it.
Recommend (b) — it detects phases correctly, it just does not change outcomes.

### M6. 149,215 trace rows collected, ~500 used

Every consumer filters `input_idx == 0` — one inference pass. No variance, no error bars, and the
200-sample profiling is wasted.

**Fix:** aggregate across all passes; report mean +/- std or a confidence interval on every headline
number. This is table stakes for review.

### M7. Profiling uses untrained networks on noise

`profile_resnet.py:85` uses `models.resnet50(weights=None)` — random init — with `torch.randn`
inputs. Activation sparsity from a random-init net on random data does not support claims about
real ResNet-50 behaviour. L1-pruning a random network is likewise arbitrary.

**Fix:** use pretrained weights (`weights=ResNet50_Weights.DEFAULT`) and real sample inputs
(a few hundred ImageNet validation images, or at minimum natural images). Same for BERT: real
checkpoint, real tokenized text. **Everything sparsity-dependent must be re-measured after this.**

### M8. `config_modes.py` is curve-fitted, not derived

The comments preserve the search: *"Paper=5. Use: ... Hmm. Let me use: ... 2*1+2+1=5 ✓"*. The
formulas are tuned to a handful of 2x2/4x4 points, then extrapolated to 8x8 arrays and 3072-wide
layers where nothing validates them. The 30/30 validation pass uses **+/-30% tolerance**; 8 of 30
cases are off by up to 18.2%.

**Fix (choose one):**
- **(a) Honest framing:** state it is an analytical model fitted to Table II, report the per-op
  residuals, and tighten the tolerance to something meaningful (+/-5%) with failures shown.
- **(b) Validate against RTL:** once Phase 1/B7 lands, compare model cycles to simulated cycles at
  8x8 for a handful of real layer shapes. This is the strongest option and directly feeds the paper.

Recommend **(b)**, falling back to (a) if RTL slips.

---

## 4. Phase 3 — Re-measure, then find the real result

Order matters: **B5 -> B4 -> B1 -> M7 -> M1 -> M6.** Each invalidates what precedes it.

### E1. Regenerate everything, clean
Re-run profiling (pretrained models), trace generation (clipped), predictor (3-way split),
scheduler (fixed baseline, correct operands), prefetch (fixed cost model). Archive old CSVs under
`results/pre_fix/` so the diff is auditable.

### E2. The sensitivity sweep — *this is the new core experiment*

The whole adaptive-scheduling story hinges on one ratio: **reconfiguration cost / layer compute
cost.** Today it is ~0.2–7% and lookahead is worthless. Sweep it:

- switch cost: `{10, 50, 100, 250, 500, 1k, 5k, 10k, 50k}` cycles
- for each, report Static / Greedy / DP(W=1,2,3,5) / Oracle, and switch counts
- **find and report the crossover** where lookahead starts to matter

This converts the negative result into a *design rule*: "lookahead scheduling is justified only
when reconfiguration exceeds X% of layer latency; for an 8x8 array at 1 GHz that means Y cycles,
which current interconnects do not approach." That is a genuine, citable contribution.

### E3. Array-size and bandwidth sweep
`Nr x Nc` in `{4, 8, 16, 32}` and BW in `{2, 4, 8, 16, 32}` elem/cycle. Small arrays and constrained
bandwidth are where NM and DM-R should be competitive. If NM never wins anywhere in that space,
report that as a bounded negative result with the closed-form crossover already derived in the
report's §2.3 (`P <= 4`) — which is good analysis and should be kept.

### E4. Predictor ablation that actually discriminates
Design an evaluation where prediction *can* change the decision: drop the
`max(pred, weight_sparsity)` masking, or evaluate on activation-dominated workloads only. Report
match rate **and** end-to-end cycle impact of prediction error, with the null baselines from M2.

---

## 5. Phase 4 — Paper reframing

### Claim-by-claim disposition

| # | Current claim | Verdict | Action |
|:-:|:--|:--|:--|
| 1 | 1.84x / 2.04x avg speedup over static | **Requalify** | Baseline is scored without switch cost (B4). Re-measure; report with CI (M6). Speedup is real for heterogeneous workloads, near 1.0x for BERT — say both. |
| 2 | DP matches Oracle, Greedy makes extra switches | **Keep, reframe** | True but the margin is ~0.003%. Reframe as "greedy is sufficient", supported by E2. |
| 3 | 99.2% / 98.9% config match | **Requalify** | Must ship with persistence baseline (98.4%). Fix M1 first. |
| 4 | Attention sparsity varies 0–99% | **Keep** | Solid and interesting. Re-verify on a *pretrained* BERT (M7). |
| 5 | 99.3% reconfig overhead hidden | **Withdraw pending fix** | Artifact of B3's free-fetch bug. Re-measure. |
| 6 | W=2 sufficient, W=5 conservative | **Replace** | Repo's own data says W=1. Replace with E2's crossover analysis. |
| 7 | Switch costs 100–250 cycles | **Keep as estimate** | Already honestly caveated. Upgrade with synthesis (Phase 5). |
| 8 | NM gives 25% cycle reduction | **Withdraw** | Hardcoded constant. Either back it with RTL (B7/B8) or remove it. |
| 9 | Phase controller 100% correct | **Requalify** | Detection is correct; *performance impact is zero* (M5). Claim detection only. |
| 10 | Speculation never beats Execution-Oracle | **Keep** | Good methodological hygiene, honestly reported. Idea 4 is the strongest-integrity section. |

### Suggested narrative

> **Title direction:** "When Does Reconfigurability Pay? A Cost-Model Study of Adaptive Dataflow
> Selection for Sparse Accelerators"
>
> **Contributions:**
> 1. An open cycle-level cost model for VersaAccel's four dataflows, validated against RTL at 8x8.
> 2. A sensitivity analysis establishing the **reconfiguration-cost threshold** above which
>    lookahead scheduling is justified — and the finding that current interconnects sit two orders
>    of magnitude below it, so a greedy switch-cost-aware rule is within 0.02% of optimal.
> 3. A bounded negative result on N:M as a standalone dataflow, with a closed-form crossover
>    (`P <= 4`) showing the cubic-vs-quadratic gap cannot be closed by bandwidth.
> 4. A speculative-switching mechanism with verified flush behaviour and bounded degradation
>    (<= +1.3% at 10% misprediction).

This is an honest paper that survives review. The current framing does not: reviewer #2 will run
the ablation table, see `W=1: +0.0002%`, and reject.

---

## 6. Phase 5 — RTL and synthesis

1. **Compile clean** — `iverilog -g2012` on all three testbenches (fixes B7). Free and local.
2. **Make testbenches check arithmetic**, not just handshakes. Today they verify FSM transitions
   only; no golden-model comparison of matrix results exists. Generate golden vectors from the
   Python cost model and compare.
3. **Load real CPE weights** (B8), verify RTL argmax against Python on held-out traces.
4. **Vivado** for the full sim once iverilog is clean.
5. **Genus synthesis**, 45nm, identical constraints across VersaAccel / systolic / SIGMA.
   Extract area, power, Fmax, gate count.
6. **Feed synthesis back into the cost model:** the measured reconfiguration latency replaces the
   estimated `SWITCH_COST` matrix, and the measured Benes overhead replaces the invented `1.12`
   (M3). This closes the loop and retires claim 7's caveat.

---

## 7. Report-vs-repo discrepancies to correct in `PROJECT_REPORT.md`

Independent of the bugs, the report describes a tree that does not exist:

| Report says | Repo has |
|:--|:--|
| `idea1_adaptaccel/prefetch/prefetch_engine.py` | `idea1_adaptaccel/prefetch_pipeline/prefetch_pipeline.py` |
| `evaluation/` directory | does not exist |
| `paper_reference.md` | does not exist |
| 108,800 traces | **149,200** rows across 15 files |
| 8 architectures | 15 trace files (incl. `dp_stress`, `switching_stress`, `*_nm24`) |
| Switch cost DMR->D-M-R = 150 | `dp_scheduler.py` says **250** |
| Window ablation: W=1 +0.01%, 92 switches | actual run: **W=1 +0.0002%**, 102 Greedy / 93 DP switches |
| switching_stress Greedy=8 vs DP=5 | actual run: Greedy=**10**, DP=5 |
| Predictor MAE 0.0756 / 0.0354 | actual run: **0.0818 / 0.0340** |
| "Early stopping monitors validation loss" | monitors **test** loss (M1) |
| NM compat: 50% pruned -> 68.8% | `nm_compatibility_score` counts all-zero groups as compatible, inflating this |

Also: `nm_compatibility_score()` scores a group compatible when `nnz <= N`, so a 90%-pruned matrix
reads as ~100% "2:4 compatible" even though 2:4 offers it nothing. That conflates *fits the
pattern* with *already sparser than the pattern* — fix the metric or rename it.

---

## 8. Execution order

| Step | Work | Gate |
|:-:|:--|:--|
| 1 | Phase 0 (R1–R5) | clean clone runs |
| 2 | B1, B2, B6, B7 — self-contained, no re-measurement needed | `iverilog` clean; traces valid |
| 3 | B5, B4, B3 — cost-model corrections | all CSVs regenerated |
| 4 | M7 (pretrained models) -> re-profile | traces are meaningful |
| 5 | M1, M2, M6 — predictor methodology | honest MAE + baselines |
| 6 | M3, M4, M5 — remove thumbs on scale | Idea 1 and 2 share one cost function |
| 7 | E2, E3, E4 — the new experiments | the actual contribution |
| 8 | B8 + Phase 5 — RTL, synthesis, loop closure | claim 7 retired, claim 8 resolved |
| 9 | Rewrite `PROJECT_REPORT.md` per §5 and §7 | submittable |

Steps 2 and 3 are mechanical and land today. Step 4 needs network access for pretrained weights.
Step 8 needs Vivado/Genus on college systems.

---

## 9. Phase 6 — throughput (added after Phase 5)

Phase 5 left the RTL functionally correct but serially drained. Two changes, both done:

| # | Work | Status |
|:-:|:--|:--|
| T1 | Output double buffer: `ST_GEMM_CAP` copies the tile accumulators into a shadow bank so the array restarts immediately; drain runs as a concurrent process | **done** |
| T2 | Wide output bus (`drain_wide` / `dout_vec[NR]`): a PE column per cycle, `NC` beats per tile instead of `NR·NC` | **done** |
| T3 | `run_all_tb.sh` — the whole testbench suite in one command, non-zero exit on failure | **done** |
| T4 | Zero-skipping in the sparse configurations | **done** — but it measures a cost-model limitation rather than a speedup, see below |
| T5 | Overlap consecutive tiles' skew fill (needs a second operand feed port) | **open** — utilisation is `K/(K+NR+NC−1)`, so this only matters for small K |
| T6 | Reconcile the cost model's `nnz`-proportional sparsity term with T4's measurement | **open — decision needed**, not a defect |

### T4 outcome

A systolic array shares operands along rows and columns, so a step `k` is skippable only when
the whole tile's `k`-slice is zero. Measured at 16x16: 50% unstructured zeros save 0%, 2:4
structured zeros save 0%, 50% k-slice-structured zeros save 12.8% (K=16) / 28.2% (K=64) --
against the `nnz` model's predicted 50% in every one of those cases. The ceiling is
`f_struct * K/(K+NR+NC)`, not `f`.

The 2:4 result independently confirms the N:M negative result in `PROJECT_REPORT.md` §6 from a
completely different direction.

`cost_model.py` was deliberately NOT changed: its sparsity term drives every table in §3-§6,
so the choice between adding a structure-aware term, restricting the claims to structured
sparsity, or reporting the gap as a limitation is a decision about the paper's argument. T6.

Verification: every shape is run at both bus widths and once more under ~25% random
`dout_ready` deassertion, which is what exercises the capture-stall path. All three must agree
bit-for-bit with the reference GEMM.

Result: 292 -> 37 cycles per tile at 16x16 with K=6; the measured wide-bus totals match the
analytic ideal `load + tiles*(K+NR+NC-1) + NC` exactly. Details and caveats in
`PROJECT_REPORT.md` §7.6.
