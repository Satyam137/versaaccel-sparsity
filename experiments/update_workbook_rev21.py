"""Bring ~/Downloads/VersaAccel_Results_Explained.xlsx up to Rev 21.

The script that built the workbook (Rev 17-19) lived in a scratch folder and was
lost, so this edits the existing workbook in place, keeping its sheets, styles
and formulas, and lives in the repo. It changes:

  Synthesis          rewritten: Genus runs 3-4 (slow corner) -- Fmax, area,
                     VersaAccel block breakdown, option costs, energy per MAC
  Switch-Cost Sweep  the RTL-measured switch cost (B2) added below the sweep
  Fill & Streaming   real-slice cycles re-measured on the Rev 20 RTL
  All Results        rows superseded by runs 3-4 and B2 rewritten; new rows
  Summary            Rev 21 source line, story line 7, verdict ranges

Usage: python experiments/update_workbook_rev21.py [path.xlsx]
"""
import copy, os, sys
import openpyxl
from openpyxl.styles import Font, PatternFill

PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/Downloads/VersaAccel_Results_Explained.xlsx")
BLUE = Font(color="1F4E79")
BOLD = Font(bold=True)
TITLE = Font(bold=True, size=13)
GREEN = PatternFill("solid", fgColor="E2EFDA")
YELLOW = PatternFill("solid", fgColor="FFF2CC")


def clear(ws):
    for row in ws.iter_rows():
        for c in row:
            c.value = None
            c.fill = PatternFill()
            c.font = Font()


def put(ws, r, values, font=None, fill=None):
    for j, v in enumerate(values, start=1):
        if v is None:
            continue
        c = ws.cell(row=r, column=j, value=v)
        if font is not None:
            c.font = copy.copy(font)
        elif isinstance(v, (int, float)):
            c.font = copy.copy(BLUE)
        if fill is not None:
            c.fill = copy.copy(fill)


def synthesis(ws):
    clear(ws)
    put(ws, 1, ["Genus runs 3-4 -- Cadence Genus 21.10, gsclib045 SLOW corner (0.9 V, 125 C), "
                "identical constraints, Rev 20 RTL (pipelined operand feed)"], TITLE)
    put(ws, 2, ["Source: PROJECT_REPORT.md section 7.15; OUTPUT_r20 (run 3) and OUTPUT_r21 (run 4). "
                "Run 1 (fast corner, Benes/CPE folded away) is superseded."])
    put(ws, 4, ["Metric", "Systolic", "SIGMA", "VersaAccel", "Note"], BOLD)
    rows = [
        ("Multipliers", 256, 256, 256, "All three at equal compute."),
        ("Last met clock (ns)", 2.5, 2.6, 2.4, "Systolic misses 2.1 ns by 3 ps; Fmax = 1/critical path."),
        ("Critical path when pushed (ns)", 2.103, 2.523, 2.329, "Systolic MAC; SIGMA product->FAN; VersaAccel skip-schedule generator."),
        ("Fmax (MHz)", "=1000/B7", "=1000/C7", "=1000/D7", "VersaAccel 10% below systolic, 8% above SIGMA."),
        ("Cell area at 2.5 ns (um2)", 944730, 1551021, 1867000, "SIGMA misses 2.5 ns by 23 ps (1,317,000 um2 at 2.6 ns, met)."),
        ("Area vs systolic", "=B9/$B$9", "=C9/$B$9", "=D9/$B$9", "VersaAccel 1.98x the systolic array, 1.20x SIGMA."),
        ("Sequential cells", 16423, 25439, 51908, "SIGMA 25.4k = the fair 256-multiplier design (B1f)."),
        ("Energy, dense (pJ/MAC)", 1.89, 12.78, 4.20, "Compute phase only (power_windows.py); same real operands."),
        ("Energy, 90% pruned (pJ/MAC)", 1.32, 4.96, 2.54, "SIGMA's test is 512 MACs in 22 cycles: pessimistic."),
        ("Energy saved by pruning", "=1-B13/B12", "=1-C13/C12", "=1-D13/D12", "VersaAccel also saves cycles (288 -> 283)."),
        ("Flops annotated by the VCD", 0.66, 0.996, 0.75, "Unannotated flops take default activity: ratios +-~20%."),
        ("Power, vectorless at 2.5 ns (mW)", 153.7, 441.2, 255.6, "Default toggle rate -- scale only."),
    ]
    for i, r in enumerate(rows):
        put(ws, 5 + i, list(r))
    put(ws, 18, ["VersaAccel area by block (run 3, 2.5 ns)"], BOLD)
    put(ws, 19, ["Block", "Cell area (um2)", "Share", None, "What it is"], BOLD)
    blocks = [
        ("PE array", 1118093, "256 PEs, 4,367 um2 each"),
        ("Controller, operand feed, schedule (top)", 405313, "Flattened into versaaccel_top"),
        ("Nonzero bitmaps (a_nz + b_nz)", 111866, "Zero-skip k-slice bitmaps"),
        ("Operand buffers (ibuf + wbuf)", 107759, "Flip-flop storage (SRAM in a real chip)"),
        ("Output tile buffer (tbuf)", 63702, "Double buffer that makes the drain free"),
        ("CPE", 45133, "Trained INT8 MLP, 6 stages"),
        ("Reduction network", 7861, ""),
        ("Benes distribution network", 4290, ""),
    ]
    for i, (n, a, w) in enumerate(blocks):
        put(ws, 20 + i, [n, a, f"=B{20 + i}/SUM($B$20:$B$27)", None, w])
    put(ws, 28, ["PE array + both networks vs whole systolic array", "=(B20+B26+B27)/B9", None, None,
                 "1.20x: the reconfigurable datapath itself is cheap."], fill=GREEN)
    put(ws, 30, ["Cost of each RTL option (area over default, 2.5 ns)"], BOLD)
    put(ws, 31, ["Option", "Cell area (um2)", "vs default", "Timing at 2.5 ns", "Buys"], BOLD)
    opts = [
        ("Default", 1864162, "met", ""),
        ("Fill amortisation (B10)", 1948074, "met", "Skip saving ~2x (section 7.13)"),
        ("+ streamed activations (B12)", 1911979, "met", "Load hidden, 1.33x end to end"),
        ("Reshape square/wide/tall (B15)", 2934809, "-278 ps", "1.40x pruned BERT, 1.15x pruned ResNet"),
        ("  same, at 2.8 ns (batch 5)", 2660000, "-408 ps at 2.8 ns", "path 3.21 ns: net loss everywhere"),
        ("Reshape square/tall (Rev 21, batch 5)", 2248000, "-42 ps", "1.40x / 1.12x; per area 1.16x pruned BERT, 0.93x pruned ResNet"),
    ]
    for i, (n, a, t, b) in enumerate(opts):
        put(ws, 32 + i, [n, a, (f"=B{32 + i}/$B$32-1" if a else None), t, b],
            fill=(YELLOW if ("B15" in n or "2.8 ns" in n) else GREEN if "square/tall" in n else None))
    ws.column_dimensions["A"].width = 46
    for col in "BCD":
        ws.column_dimensions[col].width = 18
    ws.column_dimensions["E"].width = 70


def switch_sweep(ws):
    r = ws.max_row + 2
    put(ws, r, ["Measured on the RTL (B2, Rev 21, tb_switch.sv)"], BOLD)
    put(ws, r + 1, ["Measurement", "Cycles", None, None, None, None, None, None, "Meaning"], BOLD)
    put(ws, r + 2, ["Forced switch, all 16 config pairs", 0, None, None, None, None, None, None,
                    "Config + Benes routing are registers captured on start; the array is drained every GEMM anyway"])
    put(ws, r + 3, ["CPE-predicted switch in force after start", 9, None, None, None, None, None, None,
                    "CPE 7 + speculation 1 + config register 1, hidden under a 768-cycle operand load"])
    put(ws, r + 4, ["As % of a median layer", 0.0002, None, None, None, None, None, None,
                    "vs 0.7% for the 250-cycle estimate; ~11,000x below the look-ahead threshold"])
    put(ws, r + 5, ["Greedy gap at the measured cost (%)", 0.0, None, None, None, None, None, None,
                    "Greedy is exactly optimal; headline 1.0525x -> 1.0526x"], fill=GREEN)


def fill_streaming(ws):
    vals = {  # (row): per-tile off, per-tile on, amortised off, amortised on -- Rev 20 RTL
        5: (32424, 28782, 16696, 12635), 6: (28328, 24969, 16632, 12761), 7: (28328, 22881, 16696, 11395),
        8: (6174, 5422, 3306, 2472), 9: (5406, 5178, 3258, 2934), 10: (5406, 3954, 3306, 2052)}
    for r, (a, b, c, d) in vals.items():
        for col, v in zip("CDFG", (a, b, c, d)):
            ws[f"{col}{r}"] = v
    for col, v in zip("BCDE", (65536, 88286, 76784, 66571)):
        ws[f"{col}20"] = v
    for col, v in zip("BCDE", (30720, 34674, 32772, 31593)):
        ws[f"{col}21"] = v
    ws["A2"] = ("Source: results/amortise_rtl.csv (experiments/amortise_rtl.py), re-measured on the Rev 20 "
                "RTL (pipelined operand feed: +4 cycles per GEMM). 2.8M element checks, 0 errors.")


ALL = {  # row in 'All Results' -> new values for columns C..I (None keeps the cell)
    8: [None, "Greedy within 0.013% of the DP oracle at the 250-cycle estimate; EXACTLY optimal (0.000%) at "
              "the RTL-measured switch cost (B2)", None, None, None, None,
        "Holds at any realistic cost; the measured switch (<= 9 cycles) is 0.02% of a layer.", "§4.1, §4.3"],
    9: [None, "Look-ahead pays only above ~265% of layer latency; the RTL-measured switch is 0.02% -- ~11,000x below",
        None, None, None, None, "The threshold is a cost-model result; the switch cost is now measured.", "§4.2, §4.3"],
    25: [None, "Skip saving on real pruned slices at 16×16: ResNet-50 11.2% → 24.3%, BERT 12.2% → 25.2% "
               "(Rev 20 RTL; 24.8% / 25.7% before); +4.5% area, meets 2.5 ns", None, None, None, None, None, None],
    26: [None, "End to end, real ResNet slices: 88,286 → 66,571 cycles (1.33×), within 1.6% of the load; "
               "+2.6% area, meets 2.5 ns", None, None, None, None, None, None],
    40: ["Reconfigurable datapath is cheap at the slow corner",
         "PE array + Benes + reduction = 1.13 mm², 1.20× the whole systolic array (0.945 mm²); the two "
         "networks are 0.6% of VersaAccel, the CPE 2.4%", "Systolic array, same 256 MACs, 2.5 ns, slow corner",
         "★ Strong positive", "The flexibility hardware itself costs little; the extra area is the feed, "
         "schedule and buffer logic around the array.", "Answers 'what does flexibility cost?' with a "
         "measured, like-for-like number.", "Compare cores, not totals (B1d).", "§7.15"],
    41: ["Benes and reduction networks are tiny", "Benes 4,290 µm² (0.2%), reduction 7,861 µm² (0.4%) of "
         "VersaAccel at 2.5 ns", "Total VersaAccel area", "★ Strong positive",
         "The routing that makes the design reconfigurable is a negligible share of the chip.",
         "Measured in VersaAccel itself at the slow corner (run 1 inferred it from SIGMA).", None, "§7.15"],
    43: ["VersaAccel total area 1.98× the systolic array", "1.867 vs 0.945 mm² at 2.5 ns; 1.20× SIGMA "
         "(1.551)", "Systolic, SIGMA", "Caveat / limitation", "Feed and schedule logic (21.7%), bitmaps "
         "and buffers (15%) are what a fixed array does not need.", "Explained block by block, not hidden.",
         "Buffers would be SRAM in a real chip.", "§7.15"],
    44: ["Energy per MAC, compute phase", "Dense 4.20 pJ/MAC (systolic 1.89, SIGMA 12.78); 90% pruned 2.54 "
         "(1.32, 4.96); pruning saves VersaAccel 40% vs 30% for systolic", "Systolic, SIGMA on the same "
         "real operands", "Positive", "Costs ~2× the systolic array per MAC, but zero-skipping saves it "
         "more energy and cycles.", "Measured from real-operand activity over the compute phase only.",
         "SIGMA's 512-MAC test is pessimistic; ±~20% from unannotated flops.", "§7.15"],
    45: ["NEW (Rev 21): switching configuration costs ≤ 9 cycles, not 100–250", "Forced: 0 cycles for all "
         "16 pairs; CPE-predicted: 9 cycles, hidden under a 768-cycle load", "The 100–250-cycle estimate "
         "used by the scheduler", "★ Strong positive", "Reconfiguring between layers is essentially free "
         "on this RTL.", "Replaces the biggest remaining estimate with a measurement; greedy becomes exactly "
         "optimal.", "GEMM mode runs one dataflow (§7.9): this is the control cost.", "§4.3"],
}
NEW_52 = ["Synthesis", "VersaAccel reaches ~430 MHz, between the baselines", "Critical path 2.33 ns (~430 MHz), met at 2.4 ns; "
          "systolic ~475 MHz, SIGMA ~395 MHz; was ~180 MHz before the operand feed was pipelined",
          "Systolic and SIGMA at the same corner and constraints", "Positive", "Pipelining the operand feed "
          "made the PE multiply-accumulate the critical path at 2.5 ns, as in the systolic array.",
          "A clock comparable to fixed designs: flexibility does not cost frequency.", "At 2.0 ns the "
          "skip-schedule generator (2.33 ns) limits it.", "§7.14–§7.15"]
NEW_51 = ["Synthesis", "Reshaping pays in silicon only for weight-pruned Transformers", "Square + tall array: "
          "+20.6% area, −42 ps at 2.5 ns; per area 1.16× on pruned BERT, 0.93× on pruned ResNet. Three-shape "
          "array: +43–57% and misses 2.8 ns — a net loss everywhere", "Default VersaAccel", "Caveat / limitation",
          "Choosing tile orientation by where the zeros are is worth up to 1.40×, but the hardware to do it "
          "only pays where weight sparsity dominates.", "Found by synthesis and stated openly; it narrows "
          "claim 22 to a precise, defensible statement.", "Quote the per-area numbers with the speedup.", "§7.15, claim 22"]


def all_results(ws):
    for r, vals in ALL.items():
        for j, v in enumerate(vals):
            if v is not None:
                ws.cell(row=r, column=3 + j, value=v)
    put(ws, 54, [50] + NEW_51)
    put(ws, 55, [51] + NEW_52)
    for r in (54, 55):
        ws.cell(row=r, column=1).font = Font()
    ws.cell(row=37, column=9, value="Synthesised at the slow corner: 45,133 µm², 2.4% of VersaAccel (§7.15).")
    ws.cell(row=51, column=9, value="Synthesis (§7.15): a square + tall array costs +20.6% area — a net gain per "
            "area on pruned BERT (1.16×), a loss on pruned ResNet (0.93×); the three-shape array loses everywhere.")


def summary(ws):
    ws["A2"] = ("Source: PROJECT_REPORT.md Rev 21 and results/*.csv (cost model: published model held to the MAC "
                "bound). Synthesis: Genus runs 3-4, slow corner. RTL: 11/11 benches pass on every build.")
    for r in range(5, 11):
        v = ws[f"B{r}"].value
        if isinstance(v, str):
            ws[f"B{r}"] = v.replace("$F$53", "$F$55")
    ws["A9"] = "Pending synthesis"
    ws["C9"] = "Waiting on Genus batch 5 (cost of the square + tall reshapeable array)."
    for row in ws.iter_rows(min_row=12, max_row=20):
        for c in row:
            if isinstance(c.value, str) and c.value.startswith("7. "):
                c.value = ("7. Flexibility is cheap where it lives: PE array + Benes + reduction is 1.20× the systolic "
                           "array, the networks 0.6%; VersaAccel runs at ~430 MHz (systolic ~475) and switching "
                           "configuration costs ≤ 9 cycles. Reshaping (+21%, square + tall) pays only on weight-pruned Transformers.")
    for row in ws.iter_rows():
        for c in row:
            if c.value == "Genus run-1 area/power/timing for the three designs, with derived ratios.":
                c.value = "Genus runs 3-4 (slow corner): Fmax, area by block, option costs, energy per MAC."


def main():
    wb = openpyxl.load_workbook(PATH)
    synthesis(wb["Synthesis"])
    switch_sweep(wb["Switch-Cost Sweep"])
    fill_streaming(wb["Fill & Streaming"])
    all_results(wb["All Results"])
    summary(wb["Summary"])
    wb.save(PATH)
    print(f"updated {PATH}")


if __name__ == "__main__":
    main()
