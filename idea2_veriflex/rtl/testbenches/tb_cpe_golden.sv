// =============================================================================
// tb_cpe_golden.sv -- does the RTL Config Prediction Engine agree with the
// quantised Python model it was trained from?
//
// FIX_PLAN.md B8. The CPE previously had every weight hardcoded to 8'sd1 and
// every bias to 0, so its argmax was degenerate and it predicted one config
// forever. Nothing anywhere performed a $readmemh, so the trained weights were
// never loaded into any hardware.
//
// The CPE now loads INT8 weights exported by train_cpe_classifier.py, which
// trains this exact 9->16->16->5 topology with quantisation-aware fine-tuning
// against the same integer pipeline the RTL implements. Because the Python
// reference is bit-exact with the hardware, this bench asserts EQUALITY of the
// predicted config, not merely similar behaviour.
//
// Fixtures (all produced by that script):
//   cpe_test_vectors.hex  64 held-out feature vectors, 9 INT8 values each
//   cpe_expected.hex      the integer model's argmax for each
//
// Run:
//   iverilog -g2012 -o tb_cpe -DCPE_WEIGHT_DIR='"<abs path>/rtl/weights"' \
//       common/versaaccel_pkg.sv versaaccel_rtl/config_prediction_engine.sv \
//       testbenches/tb_cpe_golden.sv
//   ./tb_cpe
// =============================================================================
`timescale 1ns/1ps

module tb_cpe_golden;
  import versaaccel_pkg::*;

  localparam int N_VEC   = 64;
  localparam int N_FEAT  = 9;

  logic clk = 0;
  logic rst_n;
  always #5 clk = ~clk;

  logic                     valid_in;
  logic signed [7:0]        features [N_FEAT];
  config_t                  predicted_config;
  logic [7:0]               confidence;
  logic                     valid_out;

  config_prediction_engine dut (
    .clk(clk), .rst_n(rst_n), .valid_in(valid_in), .features(features),
    .predicted_config(predicted_config), .confidence(confidence),
    .valid_out(valid_out)
  );

  // fixtures
  logic [7:0] vec_mem [N_VEC * N_FEAT];
  logic [7:0] exp_mem [N_VEC];

  int errors = 0;
  int checked = 0;
  int hist [5];
  int nonzero;

  initial begin
    $display("============================================================");
    $display("  CPE -- GOLDEN VECTOR test (RTL vs quantised Python model)");
    $display("============================================================");

    $readmemh({`CPE_WEIGHT_DIR, "/cpe_test_vectors.hex"}, vec_mem);
    $readmemh({`CPE_WEIGHT_DIR, "/cpe_expected.hex"}, exp_mem);
    for (int k = 0; k < 5; k++) hist[k] = 0;

    rst_n = 0; valid_in = 0;
    for (int f = 0; f < N_FEAT; f++) features[f] = '0;
    repeat (4) @(posedge clk);
    rst_n = 1;
    repeat (2) @(posedge clk);

    for (int v = 0; v < N_VEC; v++) begin
      for (int f = 0; f < N_FEAT; f++)
        features[f] = $signed(vec_mem[v * N_FEAT + f]);
      // Inputs change on the falling edge, so the rising edge that samples
      // them is never racing the assignment. (They used to be assigned at the
      // rising edge itself, and the CPE saw the pulse only if the simulator
      // ran this process first.)
      @(negedge clk);
      valid_in = 1;
      @(negedge clk);
      valid_in = 0;

      // wait for the result (6-stage pipeline since the CPE was split into
      // dot-product and requantise/argmax stages; was a fixed 3 cycles)
      while (!valid_out) @(negedge clk);

      checked++;
      hist[int'(predicted_config)]++;
      if (int'(predicted_config) !== int'(exp_mem[v])) begin
        errors++;
        if (errors <= 8)
          $display("  [FAIL] vector %0d: RTL predicted %0d, Python model %0d",
                   v, int'(predicted_config), int'(exp_mem[v]));
      end
    end

    $display("------------------------------------------------------------");
    $display("  Prediction histogram (DMR/D-MR/DM-R/D-M-R/NM): %0d %0d %0d %0d %0d",
             hist[0], hist[1], hist[2], hist[3], hist[4]);
    // The old all-ones CPE emitted a single config for every input. If the
    // histogram has only one non-zero bucket the engine is still degenerate,
    // even if it happens to agree with the expectations.
    nonzero = 0;
    for (int k = 0; k < 5; k++) if (hist[k] > 0) nonzero++;
    if (nonzero < 2)
      $display("  [WARN] only %0d distinct config(s) predicted -- engine may be degenerate",
               nonzero);
    else
      $display("  [PASS] engine is non-degenerate (%0d distinct configs)", nonzero);

    if (errors == 0)
      $display("  [PASS] all %0d vectors match the quantised Python model", checked);
    else
      $display("  [FAIL] %0d/%0d mismatches", errors, checked);
    $display("============================================================");
    $finish;
  end

  initial begin
    #200000;
    $display("  [FAIL] timeout");
    $finish;
  end

endmodule
