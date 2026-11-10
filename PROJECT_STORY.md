# The VersaAccel Project — The Whole Story, From the Beginning

*Written for someone meeting this project for the first time. It explains what we did, in
order, why we did each thing, and what we learned from it. The technical record with every
number and command is `PROJECT_REPORT.md`; this file is the story behind it. There is a
glossary at the end. Look up any unfamiliar term there.*

---

## In one paragraph

We started from a published accelerator design, **VersaAccel**, and a project built on top
of it that claimed big speedups from switching the accelerator's "dataflow" layer by layer,
using a small neural network to predict which dataflow to use. We audited that project
carefully and kept testing it against reality: real neural networks, real images, real
Verilog hardware, and real chip synthesis. Most of the original claims shrank or broke. What
survived turned out to be more interesting than what we started with. It is not *how many*
zeros a neural network has that decides whether skipping them helps. It is *where the zeros
sit*. That one idea led to three new results, a corrected cost model, a working and verified
hardware design, and measured chip area, clock speed and energy. The paper we can now write
is narrower than the one we started with, but every sentence in it is backed by a
measurement.

---

## Part 0 — Background you need first

### What is an accelerator, and what is a systolic array?

Neural networks spend almost all their time doing **matrix multiplication**: multiplying
numbers and adding them up. A general-purpose CPU does this slowly, so chips like Google's TPU
use an **accelerator**: a grid of tiny calculators called **processing elements (PEs)**. Each
PE does one *multiply-accumulate* (MAC) per clock tick: `total = total + a × b`.

A **systolic array** is the classic way to arrange those PEs. Picture a 16×16 grid. Numbers
from one matrix flow in from the left, numbers from the other flow in from the top, and each
PE multiplies what passes through it and keeps a running sum. Data pulses through the grid
like blood through a heart, hence "systolic". It is simple, fast and power-efficient.

### What is "sparsity"?

A matrix is **sparse** when many of its numbers are zero. In neural networks zeros come from
two places:

- **Activations** (the data flowing through the network): the common **ReLU** function turns
  every negative number into zero, so typically 40–60% of activations are zero. *This changes
  with every input image.*
- **Weights** (the network's learned parameters): **pruning** deliberately sets small weights
  to zero (50%, 70%, 90%…) to make the network cheaper. *This is fixed once the network is
  trained.* A special kind, **N:M sparsity** (e.g. 2:4), keeps exactly 2 non-zeros in every
  group of 4, which NVIDIA GPUs support.

Multiplying by zero is wasted work. The whole field of "sparse accelerators" is about
skipping it.

### What is a "dataflow"?

A dataflow is the *strategy* an accelerator uses to move numbers around: which operand stays
put in the PEs, which streams through, how partial sums are combined. Different dataflows
suit different situations. One is best for dense matrices, another for very sparse ones.

---

## Part 1 — Where it started: the VersaAccel paper

**VersaAccel** is a published reconfigurable accelerator. Its idea: build *one* PE array that
can switch between **four dataflows**, named by which of three stages (**D**istribution,
**M**ultiplication, **R**eduction) are fused together:

| Config | In plain terms | Meant for |
|:--|:--|:--|
| **DMR** | everything systolic, the classic array | dense matrices |
| **D-MR** | operands routed through a flexible network (a **Beneš network**) first | moderate sparsity |
| **DM-R** | a flexible reduction network at the end | sparse outputs |
| **D-M-R** | all stages separate (the "Gustavson" style) | high sparsity |

The paper gave formulas (a **cost model**) for how many clock cycles each dataflow takes for a
given matrix size and sparsity, and checked them on a tiny 2×2 example (its "Table II").

## Part 2 — The project as it arrived

Our project ("AdaptAccel") built on VersaAccel with four ideas. Three had code:

- **Idea 1 — Adaptive scheduling.** A neural network has dozens of layers, each a different
  shape and sparsity. *Pick the best dataflow for each layer*, and switch between them as the
  network runs. Since activation sparsity changes with each input image, add a small **neural
  network predictor** that guesses each layer's sparsity before it runs, so the scheduler can
  choose in advance. Schedulers: **greedy** (pick the best for this layer), **dynamic
  programming / lookahead** (plan several layers ahead, accounting for the cost of switching),
  and an **oracle** (perfect knowledge, as the upper bound).
- **Idea 2 — N:M as a fifth configuration**, plus the actual hardware: Verilog **RTL** for the
  accelerator, including an on-chip version of the predictor called the **CPE (Configuration
  Prediction Engine)**, plus two baseline designs to compare against (a plain **systolic**
  array and **SIGMA**, a well-known flexible sparse accelerator).
- **Idea 4 — Speculation.** If the predictor is confident, switch early; if wrong, roll back.

The original numbers were impressive: about **2× speedup** from adaptive scheduling, a
predictor matching the right choice **99%** of the time, and N:M giving a **25%** cycle
reduction.

---

## Part 3 — The audit (Revisions 1–4): "Is any of this true?"

**Why we started here.** Before writing a paper on these numbers, we had to be sure they were
real. A reviewer will try to break every claim, so we tried first.

**What we found** (all fixed):

- **Basic bugs.** Some trace files were corrupt, an import was stale, and the Verilog did not
  even compile in places.
- **The operands were mixed up.** The code attributed *weight* sparsity to the activations,
  and threw away activation sparsity whenever weights were sparser (70% of layers). Since the
  project is *about* activation sparsity, this undermined everything.
- **Three different cost models.** Different parts of the code computed cycles differently,
  and one quietly inflated its competitors' costs by 12%. We unified them into one.
- **The predictor "cheated".** Its reported accuracy was measured on data it had partly seen
  during training (a "test-set leak"). With a proper split, the honest error was 2.5× worse.
- **Magic constants.** Idea 4's results depended on unmeasured numbers. We named each one and
  swept it to see how much the conclusions depended on it.
- **The 25% N:M figure was hard-coded.** It was a typed-in constant, not a measurement.

**The first real insight: the array size changes everything.** We asked when picking a
dataflow *based on activation sparsity* could matter at all. On the small **8×8** arrays most
papers assume, the answer is *never*: one dataflow wins across the whole sparsity range, so a
predictor that always says the same thing scores "perfectly". The 99% predictor accuracy was
measuring a question with only one answer. Activation sparsity starts changing the best
choice at **16×16** (20.8% of cases) and more at 32×32 (33.3%). **So we moved the whole
project to a 16×16 array**, the size where the idea can work.

## Part 4 — Making the hardware real (Revisions 3–8)

**Why.** A cost model is only formulas. To trust them, we needed working hardware to check
against. The RTL existed, but nobody had checked it computed the right answers.

What we did and learned:

- **Golden-vector tests.** We fed the Verilog known inputs and compared every output with
  Python. This found real bugs: networks hand-written for 8×8 that broke at 16×16, and
  undefined values leaking into the array so the dense configuration produced garbage.
- **The CPE was fake.** Every weight was 1 and every bias 0, so it always predicted the same
  thing. We trained a real INT8 network for it and proved the hardware matches Python
  bit-for-bit (93% accuracy vs a 70% "always guess the most common" baseline).
- **It couldn't run a real matrix multiplication.** It could only broadcast one number. We
  built a **tiling controller**, so it now splits a big multiplication into 16×16 tiles and
  computes them all, every output checked.
- **The drain was slow.** Reading results out took longer than computing them. We added an
  **output double buffer** and a **wide output bus**: 292 → 37 cycles per tile, exactly the
  theoretical ideal.
- **Zero-skipping, and the discovery that started everything.** We added real hardware that
  skips zeros, and measured. **50% zeros scattered at random: 0% saving. 2:4 pattern: 0%
  saving. 50% zeros lined up in a particular way: 12.8–28% saving.** Same number of zeros,
  completely different result.

**Why scattered zeros save nothing.** In a systolic array a number is *shared*: the same
activation feeds a whole row of 16 PEs, the same weight a whole column. A step can only be
skipped if it is useless for **all 16** at once. That needs a whole **k-slice** to be zero:
16 activations that are all zero together. One isolated zero frees nothing, because its 15
neighbours still need that cycle. The cost model credited sparsity by *count*. The hardware
rewards *arrangement*.

## Part 5 — Real data (Revisions 9–10)

**Why.** Until then, the "network traces" (per-layer measurements of sparsity) came from
random noise and random weights, not real networks. Real behaviour could be very different.

**What we did.** Profiled **pretrained ResNet-50 on real images** (Imagenette) and
**pretrained BERT on real sentences** (SST-2), dense and pruned 50/70/90% and 2:4. The
profilers now *refuse* to run on fake data.

**What happened:**

- **The input-to-input variation was 32× larger** on real images than on noise. The project's
  core premise, that activation sparsity varies per input and so needs predicting at runtime,
  was much more true than the fake data showed. The number of workloads that genuinely need a
  runtime scheduler went **from 2 of 15 to 6 of 15**.
- **But the predictor claim died.** Over 10 random seeds, the 8,961-parameter neural predictor
  **never beat a zero-parameter constant** at choosing dataflows (93.4% vs 96.2%). It
  predicted sparsity numbers slightly better, but that never translated into better
  *decisions*. We withdrew the claim and kept the lesson: *a better metric is not a better
  decision*.

We also measured the structure the hardware cares about, on real networks:

- **Real ReLU zeros are 14–29× more lined-up than random** (ReLU zeros cluster by channel, and
  a channel becomes exactly a k-slice in the hardware's layout).
- **But far less than the cost model assumed:** about 20% of steps are actually skippable,
  where the model credited 56%. The model over-credited CNNs by up to 38 points, and
  completely missed a 24% saving in pruned BERT (which comes from *weight* structure the model
  cannot see).
- **Same density, 170× different benefit:** 2:4-pruned weights give 0.12% skippable steps;
  ReLU activations at a similar density give 20%.

This became the **central thesis**: *structure, not count, decides the benefit of sparsity.*

## Part 6 — First contact with a real chip flow (Revisions 13–16)

**Why.** A paper about hardware needs chip area, speed and power, which come from
**synthesis**: turning Verilog into real logic gates with Cadence **Genus** on the college
machines.

**What happened:**

- **Run 1 looked great, but it was wrong.** 500 MHz in 1.29 mm². But the Beneš network, the
  N:M ROM and the whole CPE had *disappeared*. Their control inputs were tied to constants
  inside the design, so the tool correctly deleted the logic as unused. We moved the
  configuration onto real input pins (and proved with a test that each pin reaches its logic).
- **SIGMA was an unfair baseline.** It had 16 multipliers instead of 256, and its reduction
  network was also optimised away. We rebuilt it properly (256 multipliers, 16 live reduction
  trees) and verified it with 1,484 checks, including deliberately broken versions to prove
  the tests catch bugs.
- **We split the buffers into their own modules**, so synthesis reports their area separately.
  This exposed a hidden testbench timing race. Fixing it required proving every recorded
  number stayed identical.

## Part 7 — The novelty problem (Revision 17)

**Why.** By now the original claims had mostly shrunk. Per-layer dataflow selection was also
already published by others (Flexagon, MAERI), and a learned dataflow predictor too (Misam).
We needed contributions that were actually new. So we turned the "structure matters" finding
into designs:

1. **The tile-size window (§2.7).** If skipping needs a whole tile edge of zeros, bigger tiles
   should skip less. Measured: ResNet-50's saving falls **37% → 19% → 9%** at 4×4 → 16×16 →
   64×64 tiles. But runtime selection only matters from **16×16 up**. The two benefits only
   overlap in a **16×16–32×32 window**, which tells an architect what array size to build.
2. **Channel regrouping (§2.8).** Which output channels share a tile is a free choice. If we
   reorder them so that channels with zeros in the same places sit together, more whole
   k-slices become zero. The hard part is that you can't reorder freely: residual
   connections, and attention heads in BERT, force some channels to share an order. We
   handled those rules and **proved it exact by re-running the real networks** (outputs
   unchanged). Gain: **+7.4 points** of saving on pruned ResNet, **+5.6** on pruned BERT, at
   zero hardware cost. Confirmed on 200 images, and cycle-exact on the RTL.
3. **Tile shape (§2.9).** Activation zeros are tested along one tile edge and weight zeros
   along the other. Making the edge facing the sparse operand *narrow* makes it all-zero more
   often. So the best shape depends on whether a layer's zeros are in its activations or its
   weights. We built a **reshapeable array** in RTL (16×16, 8×32 or 32×8): **1.40× on pruned
   BERT**, and the wrong orientation is *slower* (0.85×).

We also did a literature search to check what was genuinely new (Part 10 has the update).

## Part 8 — Finishing the hardware story (Revision 18)

- **Fill amortisation (B10).** Each tile wasted ~31 cycles getting data into position (the
  **skew fill**). That waste is what capped the zero-skip saving. We redesigned the control so
  tiles flow back to back and the fill is paid **once per multiplication instead of once per
  tile**. Skip saving on real pruned layers roughly **doubled** (ResNet 11% → 24%).
- **Streaming (B12).** Instead of loading all data before computing, activations now stream in
  *while* the array works. End to end it is **1.33× faster** on ResNet, nearly as fast as the
  data can arrive.
- **The N:M claim was dropped.** As a fifth configuration N:M made things *worse* (−2%), and
  the hardware explains why: 2:4 never lines up a whole k-slice.
- **Two more networks.** MobileNetV2 (a CNN) and OPT-125m (a language model): the pattern held.
  Selection helps CNNs, not Transformers.
- **Reproducibility.** One script regenerates every number, every random seed is fixed, and
  the traces are shipped with checksums.

## Part 9 — The cost model itself was wrong (Revision 19)

**Why we looked.** While wiring in the hardware's measured costs, two of VersaAccel's four
formulas gave impossible answers.

**The problem, in plain terms.** A multiplication of size M×K×P needs M·K·P multiply-adds, and
256 PEs can do at most 256 per cycle. So it can never take fewer than **M·K·P ÷ 256 cycles**
(the "MAC lower bound"). The published formula for **D-MR ignored K entirely**, and the one for
**D-M-R divided by the column count twice**. On real layers they promised work **31× and 16×
faster than physically possible** (medians). Those two dataflows looked like winners because
of this mistake.

**What we did.** We kept three versions side by side and were transparent about it:
- **A**, as published, kept for comparison.
- **B**, the published formulas with every dataflow held to the physical limit. **This became
  the paper's model.**
- **D**, costed the way our actual hardware works. Under D, selection gains nothing.

**What happened to the headline.** Adaptive selection went from **1.18× to 1.05×**. It now
pays only on pruned CNNs (**1.11–1.28×**), not elsewhere. That was a hard result, but finding
and quantifying a defect in a published model is itself a contribution, and it confirmed the
paper should lead with the structure results, not with selection.

## Part 10 — Real hardware numbers (Revisions 20–21)

**Synthesis run 2 found a design flaw.** At a realistic process corner ("slow corner": low
voltage, high temperature), the systolic array ran at ~490 MHz and SIGMA ~410 MHz. **VersaAccel
managed only ~180 MHz.** Its slowest path did everything in one clock tick: compute an address
(a 32-bit multiply), read a buffer, pass through 7 Beneš stages, then multiply-accumulate.

**The fix: pipelining.** We split that path into three clocked stages, delayed every control
signal to match, registered the configuration pins and the output, and split the CPE into six
stages. It costs **4 cycles per matrix multiplication and nothing per tile**. Every test was
re-run on all four hardware versions, and every earlier result held.

**Runs 3–5, the comparison** (slow corner, identical settings):

| | Systolic | **VersaAccel** | SIGMA |
|:--|--:|--:|--:|
| Max clock | ~475 MHz | **~430 MHz** | ~395 MHz |
| Area at 2.5 ns | 0.945 mm² | **1.867 mm²** | 1.551 mm² |
| Energy per MAC, dense / 90% pruned | 1.89 / 1.32 pJ | **4.20 / 2.54 pJ** | 12.78 / 4.96 pJ |

What these say:
- **Flexibility is cheap where it lives.** The PE array plus the two flexible networks is
  **1.20×** the area of the whole systolic array; the Beneš and reduction networks are only
  **0.6%** of the chip, and the CPE 2.4%. The extra area is the control, feed and buffer logic
  around the array.
- **Clock speed is between the baselines**: 10% slower than the plain systolic array, 8%
  faster than SIGMA.
- **Power had to be measured carefully.** The first attempt averaged over whole tests that were
  mostly *loading data*, so all designs looked identical. We trimmed the measurement to the
  compute phase only and compared **energy per operation**. Zero-skipping saves VersaAccel 40%
  energy on pruned data, versus 30% for the systolic array.
- **Design options:** fill amortisation costs **+4.5%** area and streaming **+2.6%**, both
  cheap. The reshapeable array was expensive: **+57%** with three shapes. A leaner
  **square + tall** version costs **+21%** and pays off *per area* only on weight-pruned
  Transformers (1.16× on pruned BERT, 0.93× on pruned ResNet). So that claim was narrowed
  honestly.

**The switch cost, finally measured.** The scheduler had *assumed* switching dataflows costs
100–250 cycles. On the real RTL, a forced switch costs **0 cycles**: the configuration is just
a register loaded at each layer's start, and the array drains between layers anyway. A
predicted switch takes **9 cycles**, hidden under the data loading. With that, greedy
scheduling is *exactly* optimal, and lookahead would need switches ~11,000× more expensive to
matter.

**Related work, checked properly.** A 2026 paper from our own professor's group (ISQED 2026)
also reorders matrices for a systolic array. Read in full, it reorders a *different*
dimension (the summed-over one, inside one multiplication) to trim the *edges* of each tile's
startup, using a hardware block. Our §2.8 reorders *output channels* across the whole network
to create whole skippable steps, with no hardware. So it is related but different, and its
finding that savings shrink on bigger arrays independently supports our tile-size window.

---

## Part 11 — Where things stand: then and now

| Original claim | What we found | Status |
|:--|:--|:--|
| Adaptive scheduling gives ~2× speedup | 1.05× overall; 1.11–1.28× on pruned CNNs; 1.00× on Transformers | **Narrowed** |
| A neural predictor picks the right dataflow 99% of the time | That was an unanswerable question at 8×8; at 16×16 a constant does as well | **Withdrawn** |
| N:M adds 25% | Hard-coded; as a real 5th config it *hurts* (−2%) | **Dropped** |
| Speculation speeds things up | It is a safety net, not a speedup (+0.002%) | **Reframed** |
| Switching costs 100–250 cycles | Measured: 0–9 cycles | **Measured** |
| *(new)* Greedy scheduling is enough | Exactly optimal at the measured switch cost | **Strong** |
| *(new)* Structure, not count, decides sparsity's value | 170× difference at the same density; confirmed in RTL | **Central thesis** |
| *(new)* The 16×16–32×32 window | Measured; not found in prior work | **New** |
| *(new)* Exact channel regrouping | +7.4 / +5.6 points, zero hardware | **New (supporting)** |
| *(new)* Tile orientation follows the zeros | Up to 1.40×; hardware pays only for pruned Transformers | **New, narrowed** |
| *(new)* The published cost model breaks physics | Two formulas 31× / 16× too optimistic | **New finding** |
| *(new)* The flexible hardware is cheap | Networks 0.6% of area; ~430 MHz; verified RTL | **Measured** |

**Why the narrower paper is the better paper.** Every claim now has a measurement behind it:
real networks, real data, verified hardware, real synthesis. Each negative result became a
*design rule* an architect can use: when not to build a predictor, when lookahead is
unnecessary, which array size to pick, when reshaping pays. Reviewers trust a paper that shows
what didn't work and why.

## Part 12 — What's next

1. Meet the professor: the corrected headline (§3.0), the hardware numbers (§7.15), and how
   §2.8 relates to the group's ISQED paper.
2. Decide the venue and authors, and lift the hold on writing.
3. Optional strengtheners (about a day plus one Genus run each): a faster VersaAccel clock, a
   leaner reshape design, a fairer SIGMA energy test.
4. Submission admin: code licence, push the repository, final citations.
5. Write the paper, leading with *structure, not count*.

---

## Glossary

| Term | Meaning |
|:--|:--|
| **Accelerator** | A chip specialised for one job (here, neural-network math) |
| **PE (processing element)** | One tiny calculator in the array; does one multiply-add per cycle |
| **MAC** | Multiply-accumulate: `total += a × b`, the basic operation |
| **Systolic array** | A grid of PEs where data pulses from neighbour to neighbour |
| **Dataflow** | The strategy for moving operands and partial sums through the array |
| **DMR / D-MR / DM-R / D-M-R** | VersaAccel's four dataflows (which of Distribution, Multiplication, Reduction are fused) |
| **Beneš network** | A flexible switching network that can route any input to any output |
| **Sparsity** | The fraction of numbers that are zero |
| **Activation / weight** | The data flowing through a network / the network's learned parameters |
| **ReLU** | Turns negatives into zero; the main source of activation zeros |
| **Pruning** | Deliberately zeroing small weights to make a network cheaper |
| **N:M (2:4) sparsity** | At most N non-zeros in every group of M weights |
| **Tile** | The 16×16 block of output the array computes at once |
| **k-slice** | One step's worth of shared operands along a tile edge; skippable only if all zero |
| **Skew fill** | Cycles at the start of each tile spent moving data into position |
| **Cost model** | Formulas predicting how many cycles something takes |
| **MAC lower bound** | The physical minimum: total MACs ÷ number of PEs |
| **RTL / Verilog** | The hardware description that becomes a real chip |
| **Golden vectors** | Known inputs with known correct outputs, used to test the hardware |
| **Synthesis (Genus)** | Turning RTL into real logic gates, giving area, speed and power |
| **Slow corner** | Worst-case chip conditions (low voltage, high temperature), the honest place to measure speed |
| **Fmax** | The highest clock frequency the design can run at |
| **Critical path** | The slowest chain of logic; it sets Fmax |
| **Pipelining** | Splitting a long path into shorter clocked stages so the clock can run faster |
| **pJ/MAC** | Energy per operation (picojoules per multiply-add) |
| **CPE** | Configuration Prediction Engine: the on-chip neural predictor |
| **Greedy / lookahead / oracle** | Choose the best now / plan ahead / perfect knowledge (upper bound) |
| **Trace** | A recorded per-layer measurement (shape, sparsity) from running a real network |
| **im2col** | The trick that turns a convolution into a matrix multiplication |
