// =============================================================================
// config_prediction_engine.sv — Hardcoded MLP for config selection (CPE)
//
// Fixed-point (INT8) implementation of the trained predictor.
// Architecture: 9 inputs → 16 hidden (ReLU) → 16 hidden (ReLU) → 5 outputs
// Output: argmax → 3-bit config_select
//
// Weights are hardcoded (loaded from trained model at synthesis time).
// This is a simplified CPE suitable for area estimation.
// =============================================================================
// Directory holding the exported INT8 weights. Override at compile time:
//   iverilog -g2012 -DCPE_WEIGHT_DIR='"/abs/path/to/rtl/weights"' ...
`ifndef CPE_WEIGHT_DIR
  `define CPE_WEIGHT_DIR "weights"
`endif

module config_prediction_engine
  import versaaccel_pkg::*;
(
  input  logic        clk,
  input  logic        rst_n,
  input  logic        valid_in,

  // 9 input features (8-bit fixed-point, Q4.4 format)
  input  logic signed [7:0] features [9],

  // Outputs
  output config_t     predicted_config,
  output logic [7:0]  confidence,      // 0-255 confidence level
  output logic        valid_out
);

  // ---------- Pipeline registers ----------
  // Stage 1: Input → Hidden1  (9×16 MAC)
  // Stage 2: Hidden1 → Hidden2 (16×16 MAC)
  // Stage 3: Hidden2 → Output (16×5 MAC + argmax)

  localparam int H1 = 16;  // Hidden layer 1 size
  localparam int H2 = 16;  // Hidden layer 2 size
  localparam int OUT = 5;  // Number of configs

  // Accumulator width. 16 bits is NOT enough: the output layer can reach
  // |sum| = 16 * 127 * 127 + 127 = 258,190, well past 32,767. Widened to 24.
  localparam int ACC_CPE_W = 24;

  // Weight memories (hardcoded for synthesis)
  // In practice: loaded from parameter file or register bank
  logic signed [7:0] w1 [H1][9];   // Layer 1 weights
  logic signed [7:0] b1 [H1];      // Layer 1 biases
  logic signed [7:0] w2 [H2][H1];  // Layer 2 weights
  logic signed [7:0] b2 [H2];      // Layer 2 biases
  logic signed [7:0] w3 [OUT][H2]; // Output weights
  logic signed [7:0] b3 [OUT];     // Output biases

  // Requantisation shifts, loaded alongside the weights.
  logic [7:0] shift_mem [3];

  // ---------- Weight loading ----------
  // Previously every weight was hardcoded to 8'sd1 and every bias to 0, which
  // made the argmax degenerate -- the "prediction engine" was a constant
  // function. Weights now come from INT8 hex files exported by
  // idea2_veriflex/train_cpe_classifier.py, which trains this exact topology
  // with quantisation-aware fine-tuning against the integer pipeline below.
  //
  // CPE_WEIGHT_DIR is overridable so a testbench can point at a fixture.
  //
  // Synthesis cannot execute $readmemh. Genus ignores the initial block, which
  // would leave these as registers nothing ever writes -- constant propagation
  // then deletes the whole CPE and it reports as ~zero area. Under SYNTHESIS we
  // therefore read the same numbers from a generated constant ROM
  // (scripts/gen_cpe_rom.py), so the synthesised netlist contains the real
  // network. tb_cpe_golden proves the two views hold identical values.
`ifdef SYNTHESIS
  always_comb begin
    `include "cpe_weights_rom.svh"
  end
`else
  initial begin
    $readmemh({`CPE_WEIGHT_DIR, "/cpe_w1.hex"}, w1);
    $readmemh({`CPE_WEIGHT_DIR, "/cpe_b1.hex"}, b1);
    $readmemh({`CPE_WEIGHT_DIR, "/cpe_w2.hex"}, w2);
    $readmemh({`CPE_WEIGHT_DIR, "/cpe_b2.hex"}, b2);
    $readmemh({`CPE_WEIGHT_DIR, "/cpe_w3.hex"}, w3);
    $readmemh({`CPE_WEIGHT_DIR, "/cpe_b3.hex"}, b3);
    $readmemh({`CPE_WEIGHT_DIR, "/cpe_shifts.hex"}, shift_mem);
  end
`endif

  // ---------- Pipeline ----------
  // Six stages, two per layer: (a) the dot product into an accumulator
  // register, (b) requantise + ReLU + saturate (or, for the output layer,
  // argmax). Each layer used to be one stage -- a 9- or 16-term dot product,
  // the shift and the saturation in a single cycle, and for the output layer
  // five dot products followed by a five-step compare chain -- which at the
  // slow corner is longer than a PE's multiply-accumulate. The CPE runs once
  // per layer, so the extra three cycles of latency cost nothing measurable;
  // consumers wait for valid_out. Arithmetic and results are unchanged.

  // ---------- Hidden layer 1 ----------
  logic signed [ACC_CPE_W-1:0] h1_acc [H1], h1_acc_q [H1];
  logic signed [ACC_CPE_W-1:0] h1_shifted [H1];
  logic signed [7:0]  h1_out [H1];  // After ReLU
  logic                s1a_valid, s1_valid;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      s1a_valid <= 1'b0;
      for (int k = 0; k < H1; k++) h1_acc_q[k] <= '0;
    end else begin
      s1a_valid <= valid_in;
      if (valid_in)
        for (int k = 0; k < H1; k++) begin
          h1_acc[k] = ACC_CPE_W'(signed'(b1[k]));  // Sign-extend bias
          for (int j = 0; j < 9; j++)
            h1_acc[k] = h1_acc[k] + w1[k][j] * features[j];
          h1_acc_q[k] <= h1_acc[k];
        end
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      s1_valid <= 1'b0;
      for (int k = 0; k < H1; k++) h1_out[k] <= 8'sd0;
    end else begin
      s1_valid <= s1a_valid;
      if (s1a_valid)
        for (int k = 0; k < H1; k++) begin
          // requantise, then ReLU + saturate to INT8
          h1_shifted[k] = h1_acc_q[k] >>> shift_mem[0];
          h1_out[k] <= (h1_shifted[k] < 0) ? 8'sd0 :
                       (h1_shifted[k] > 127) ? 8'sd127 :
                       h1_shifted[k][7:0];
        end
    end
  end

  // ---------- Hidden layer 2 ----------
  logic signed [ACC_CPE_W-1:0] h2_acc [H2], h2_acc_q [H2];
  logic signed [ACC_CPE_W-1:0] h2_shifted [H2];
  logic signed [7:0]  h2_out [H2];
  logic                s2a_valid, s2_valid;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      s2a_valid <= 1'b0;
      for (int k = 0; k < H2; k++) h2_acc_q[k] <= '0;
    end else begin
      s2a_valid <= s1_valid;
      if (s1_valid)
        for (int k = 0; k < H2; k++) begin
          h2_acc[k] = ACC_CPE_W'(signed'(b2[k]));
          for (int j = 0; j < H1; j++)
            h2_acc[k] = h2_acc[k] + w2[k][j] * h1_out[j];
          h2_acc_q[k] <= h2_acc[k];
        end
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      s2_valid <= 1'b0;
      for (int k = 0; k < H2; k++) h2_out[k] <= 8'sd0;
    end else begin
      s2_valid <= s2a_valid;
      if (s2a_valid)
        for (int k = 0; k < H2; k++) begin
          h2_shifted[k] = h2_acc_q[k] >>> shift_mem[1];
          h2_out[k] <= (h2_shifted[k] < 0) ? 8'sd0 :
                       (h2_shifted[k] > 127) ? 8'sd127 :
                       h2_shifted[k][7:0];
        end
    end
  end

  // ---------- Output layer + argmax ----------
  logic signed [ACC_CPE_W-1:0] out_acc [OUT], out_acc_q [OUT];
  logic                s3a_valid, s3_valid;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      s3a_valid <= 1'b0;
      for (int k = 0; k < OUT; k++) out_acc_q[k] <= '0;
    end else begin
      s3a_valid <= s2_valid;
      if (s2_valid)
        for (int k = 0; k < OUT; k++) begin
          out_acc[k] = ACC_CPE_W'(signed'(b3[k]));
          for (int j = 0; j < H2; j++)
            out_acc[k] = out_acc[k] + w3[k][j] * h2_out[j];
          out_acc_q[k] <= out_acc[k];
        end
    end
  end

  // Argmax as a tournament -- pairs, then pairs of winners -- instead of a
  // five-step chain. The right operand wins only if strictly greater, so the
  // winner is the LOWEST index holding the maximum, exactly as the sequential
  // scan from -(2^(W-1)) with a strict '>' chose. If no score beats that
  // starting value the scan kept its default, D-M-R (3), and so does this.
  localparam logic signed [ACC_CPE_W-1:0] SCORE_MIN = -(1 <<< (ACC_CPE_W - 1));
  logic signed [ACC_CPE_W-1:0] v01, v23, v03, best;
  logic [2:0]                  i01, i23, i03, best_i, max_idx;
  logic signed [ACC_CPE_W-1:0] max_val;
  always_comb begin
    {v01, i01} = (out_acc_q[1] > out_acc_q[0]) ? {out_acc_q[1], 3'd1} : {out_acc_q[0], 3'd0};
    {v23, i23} = (out_acc_q[3] > out_acc_q[2]) ? {out_acc_q[3], 3'd3} : {out_acc_q[2], 3'd2};
    {v03, i03} = (v23 > v01) ? {v23, i23} : {v01, i01};
    {best, best_i} = (out_acc_q[4] > v03) ? {out_acc_q[4], 3'd4} : {v03, i03};
    max_idx = (best > SCORE_MIN) ? best_i : 3'd3;   // default D-M-R
    max_val = (best > SCORE_MIN) ? best   : SCORE_MIN;
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      s3_valid <= 1'b0;
      predicted_config <= CFG_D_M_R;
      confidence <= 8'd0;
    end else begin
      s3_valid <= s3a_valid;
      if (s3a_valid) begin
        predicted_config <= config_t'(max_idx);
        // Confidence = gap between best and 2nd best (scaled)
        // Simple approximation: use max value magnitude
        confidence <= (max_val > 0) ? ((max_val > 255) ? 8'd255 : max_val[7:0])
                                    : 8'd0;
      end
    end
  end

  assign valid_out = s3_valid;

endmodule
