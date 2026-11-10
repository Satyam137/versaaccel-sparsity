// =============================================================================
// tb_regroup.sv -- RTL measurement of the section 2.8 channel regrouping (B13)
//
// Section 2.8 reorders a layer's weight output channels so that their zeros
// share a column tile, and claims more k-steps become skippable. Its numbers
// come from the RTL's cycle accounting (experiments/tile_geometry.py:cycles),
// not from the RTL. This bench closes that gap: it runs a GEMM whose operands
// are read from files -- real slices of real pruned layers, written by
// experiments/regroup_rtl.py in original and in regrouped channel order --
// with skipping off and on, checks every output element against a reference
// computed here, and prints the compute-phase cycle counts.
//
// The driver then checks, per case, that the measured cycles equal the
// accounting exactly, and that the regrouped run computes the same C with its
// columns permuted.
//
// Defines: RG_M, RG_K, RG_P (shape). Plusargs: +A=<file> +B=<file> (hex,
// row-major, 16-bit two's complement; A is MxK, B is KxP), so one build
// serves every case of a shape.
// =============================================================================
`timescale 1ns/1ps

module tb_regroup;
  import versaaccel_pkg::*;

  localparam int M = `RG_M;
  localparam int K = `RG_K;
  localparam int P = `RG_P;
  // Tile shape (B15): +SHAPE=0 square, 1 wide (NR/2 x 2NC), 2 tall (2NR x NC/2).
  // Needs a build with -DVERSAACCEL_RESHAPE for shapes 1 and 2.
  logic [1:0] shape_sel = 2'd0;
  int TR, TC, N_TI, N_TJ;

  logic [BENES_CTRL_W-1:0] routing_ctrl = '0;
  logic [2:0]              nm_pattern   = 3'd0;
  logic signed [7:0]       layer_features [9];
  initial for (int f = 0; f < 9; f++) layer_features[f] = 8'sd0;

  logic clk = 0;
  logic rst_n;
  always #5 clk = ~clk;
  logic clk_dut;
  assign #0.1 clk_dut = clk;      // same skew as tb_gemm_sparse (Rev 16)

  longint cycles = 0;
  always @(posedge clk) cycles <= cycles + 1;

  logic        start, gemm_en, drain_wide, skip_en;
  logic [1:0]  model_type, op_type;
  logic [15:0] batch_size, dim_m, dim_k, dim_p;
  logic                     din_valid, din_is_weight, din_ready;
  logic signed [DATA_W-1:0] din_data;
  logic                     dout_valid, dout_ready;
  logic signed [ACC_W-1:0]  dout_data;
  logic signed [ACC_W-1:0]  dout_vec [NR];
  logic                     cfg_override_en;
  config_t                  cfg_override;
  logic                     busy;
  config_t                  active_config;
  phase_t                   current_phase;
  logic                     flush_event;

  versaaccel_top dut (
    .clk(clk_dut), .rst_n(rst_n), .start(start),
    .model_type(model_type), .batch_size(batch_size), .op_type(op_type),
    .gemm_en(gemm_en), .dim_m(dim_m), .dim_k(dim_k), .dim_p(dim_p),
    .drain_wide(drain_wide), .skip_en(skip_en), .tile_shape(shape_sel), .routing_ctrl(routing_ctrl),
    .nm_pattern(nm_pattern), .layer_features(layer_features),
    .din_valid(din_valid), .din_data(din_data), .din_is_weight(din_is_weight),
    .din_ready(din_ready),
    .dout_valid(dout_valid), .dout_data(dout_data), .dout_vec(dout_vec),
    .dout_ready(dout_ready),
    .cfg_override_en(cfg_override_en), .cfg_override(cfg_override),
    .busy(busy), .active_config(active_config),
    .current_phase(current_phase), .flush_event(flush_event)
  );

  logic [DATA_W-1:0]        a_mem [M*K];
  logic [DATA_W-1:0]        b_mem [K*P];
  logic signed [DATA_W-1:0] A [M][K];
  logic signed [DATA_W-1:0] B [K][P];
  logic signed [ACC_W-1:0]  C_gold [M][P];
  logic signed [ACC_W-1:0]  C_rtl  [M][P];
  bit                       C_seen [M][P];

  int     errors = 0;
  int     checks = 0;
  longint acc;

  // Identical protocol to tb_gemm_sparse.run_gemm: compute = cycles from the
  // last operand element to the cycle after the final output beat.
  task automatic run_gemm(bit skipz, output longint compute);
    int  per_tile, total_beats, beat, tidx, ti, tj, rr, cc, guard;
    longint cyc_load;
    for (int r = 0; r < M; r++)
      for (int c = 0; c < P; c++) begin
        C_rtl[r][c] = 'x; C_seen[r][c] = 1'b0;
      end
    rst_n = 0;
    start = 0; gemm_en = 0; din_valid = 0; din_is_weight = 0; din_data = 0;
    dout_ready = 0; cfg_override_en = 1; cfg_override = CFG_D_M_R;
    drain_wide = 1'b1; skip_en = skipz;
    model_type = 2'd0; op_type = 2'd0; batch_size = 16'd0;
    dim_m = M[15:0]; dim_k = K[15:0]; dim_p = P[15:0];
    repeat (4) @(posedge clk);
    rst_n = 1;
    repeat (2) @(posedge clk);
    gemm_en = 1; start = 1;
    @(posedge clk);
    start = 0;
    din_valid = 1; din_is_weight = 1;
    for (int k = 0; k < K; k++)
      for (int c = 0; c < P; c++) begin
        din_data = B[k][c];
        @(posedge clk);
      end
    din_is_weight = 0;
    for (int r = 0; r < M; r++)
      for (int k = 0; k < K; k++) begin
        din_data = A[r][k];
        @(posedge clk);
      end
    din_valid = 0;
    cyc_load = cycles;
    per_tile    = NC;                       // physical columns drained per tile
    total_beats = N_TI * N_TJ * per_tile;
    beat = 0; guard = 0;
    dout_ready = 1;
    while (beat < total_beats && guard < 100000) begin
      @(posedge clk);
      #1;
      guard++;
      if (dout_valid) begin
        tidx = beat / per_tile;
        ti = tidx / N_TJ; tj = tidx % N_TJ;
        cc = beat % per_tile;
        for (rr = 0; rr < NR; rr++) begin
          int lr, lc;   // logical position of physical element (rr, cc)
          case (shape_sel)
            2'd1:    begin lr = rr % (NR/2);               lc = (rr / (NR/2)) * NC + cc; end
            2'd2:    begin lr = (cc / (NC/2)) * NR + rr;   lc = cc % (NC/2);             end
            default: begin lr = rr;                         lc = cc;                      end
          endcase
          if ((ti * TR + lr) < M && (tj * TC + lc) < P) begin
            C_rtl [ti * TR + lr][tj * TC + lc] = dout_vec[rr];
            C_seen[ti * TR + lr][tj * TC + lc] = 1'b1;
          end
        end
        beat++;
      end
    end
    @(posedge clk);
    compute = cycles - cyc_load;
    dout_ready = 0; gemm_en = 0;
    if (beat < total_beats) begin
      errors++;
      $display("  [FAIL] only %0d of %0d beats", beat, total_beats);
    end
  endtask

  task automatic check(string label);
    int e = 0;
    for (int r = 0; r < M; r++)
      for (int c = 0; c < P; c++) begin
        checks++;
        if (!C_seen[r][c] || C_rtl[r][c] !== C_gold[r][c]) begin
          e++;
          if (e <= 3)
            $display("  [FAIL] %s C[%0d][%0d] = %0d exp %0d",
                     label, r, c, C_rtl[r][c], C_gold[r][c]);
        end
      end
    errors += e;
  endtask

  longint c_off, c_on;
  string  fa, fb, fvcd;

  initial begin
    if (!$value$plusargs("A=%s", fa) || !$value$plusargs("B=%s", fb)) begin
      $display("RESULT usage: +A=<hex> +B=<hex>");
      $finish;
    end
    if (!$value$plusargs("SHAPE=%d", shape_sel)) shape_sel = 2'd0;
    case (shape_sel)
      2'd1:    begin TR = NR / 2; TC = 2 * NC; end
      2'd2:    begin TR = 2 * NR; TC = NC / 2; end
      default: begin TR = NR;     TC = NC;     end
    endcase
    N_TI = (M + TR - 1) / TR;
    N_TJ = (P + TC - 1) / TC;
    $readmemh(fa, a_mem);
    $readmemh(fb, b_mem);
    for (int r = 0; r < M; r++)
      for (int k = 0; k < K; k++) A[r][k] = $signed(a_mem[r * K + k]);
    for (int k = 0; k < K; k++)
      for (int c = 0; c < P; c++) B[k][c] = $signed(b_mem[k * P + c]);
    for (int r = 0; r < M; r++)
      for (int c = 0; c < P; c++) begin
        acc = 0;
        for (int k = 0; k < K; k++)
          acc += longint'(A[r][k]) * longint'(B[k][c]);
        C_gold[r][c] = acc[ACC_W-1:0];
      end

    run_gemm(1'b0, c_off);
    check("skip off");
    // B1c: +VCD=<file> captures activity for the skip-on run only.
    if ($value$plusargs("VCD=%s", fvcd)) begin
      $dumpfile(fvcd);
      $dumpvars(0, dut);
`ifdef DUMP_ARRAYS
      // B1c: VCD omits unpacked arrays; list their elements (generated file)
      `include `DUMP_ARRAYS
`endif
    end
    run_gemm(1'b1, c_on);
    if (fvcd.len() > 0) $dumpoff;
    check("skip on");

    // One machine-readable line for the driver; C follows for the
    // permutation check.
    $display("RESULT %0d %0d %0d %0d", c_off, c_on, checks, errors);
    for (int r = 0; r < M; r++)
      for (int c = 0; c < P; c++)
        $display("C %0d %0d %0d", r, c, C_rtl[r][c]);
    $finish;
  end

  initial begin
    #20000000;
    $display("RESULT timeout");
    $finish;
  end
endmodule
