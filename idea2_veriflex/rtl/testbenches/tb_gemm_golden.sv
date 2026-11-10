// =============================================================================
// tb_gemm_golden.sv -- does versaaccel_top compute a real tiled GEMM?
//
// FIX_PLAN.md Phase 5. tb_versaaccel_golden showed the assembled datapath was
// arithmetically correct but could only ever broadcast ONE scalar activation:
// ST_LOAD_WEIGHTS left for ST_COMPUTE on the first non-weight datum and the
// read pointers never advanced. versaaccel_top was a control/interconnect
// skeleton, not a GEMM engine.
//
// The tiling controller (gemm_en) streams a full A[M,K] x B[K,P] tile by tile.
// This bench checks every element of C against an independently computed
// reference -- no structural knowledge of the RTL beyond the operand layout and
// the drain order.
//
// It runs the SAME problem twice, once per output-bus width, and reports the
// cycle count of each:
//   drain_wide = 0 : one element per cycle on dout_data  (NR*NC beats/tile)
//   drain_wide = 1 : a whole PE column on dout_vec       (NC beats/tile)
// Both must produce bit-identical results; the difference is throughput. The
// drain is double buffered, so with the wide bus it disappears entirely behind
// the next tile's compute.
//
// Operand layout expected by the controller:
//   ibuf[r*K + k] = A[r][k]     loaded as non-weight data
//   wbuf[k*P + c] = B[k][c]     loaded as weight data
//
// Drain order: output tiles row-major; within a tile, columns outer, rows inner.
//
// Run:
//   iverilog -g2012 -o tb_gemm -DCPE_WEIGHT_DIR='"<abs>/rtl/weights"' \
//       common/versaaccel_pkg.sv versaaccel_rtl/*.sv \
//       testbenches/tb_gemm_golden.sv
//   ./tb_gemm
// =============================================================================
`timescale 1ns/1ps

module tb_gemm_golden;
  import versaaccel_pkg::*;

  // Layer configuration pins. Tied to the values the RTL used internally
  // before they became ports (routing straight-through, N:M pattern 0, all
  // CPE features 0), so every check below means exactly what it meant then.
  logic [BENES_CTRL_W-1:0] routing_ctrl = '0;
  logic [2:0]              nm_pattern   = 3'd0;
  logic signed [7:0]       layer_features [9];
  initial for (int f = 0; f < 9; f++) layer_features[f] = 8'sd0;

  // Problem size. Defaults span several tiles in both directions while A and B
  // still fit the buffers (IBUF 256, WBUF 512). Override to exercise partial
  // tiles, where M or P is not a multiple of the array:
  //   iverilog ... -DGEMM_M=17 -DGEMM_K=5 -DGEMM_P=20
`ifdef GEMM_M
  localparam int M = `GEMM_M;
`else
  localparam int M = 2 * NR;
`endif
`ifdef GEMM_K
  localparam int K = `GEMM_K;
`else
  localparam int K = 6;
`endif
`ifdef GEMM_P
  localparam int P = `GEMM_P;
`else
  localparam int P = 2 * NC;
`endif

  localparam int N_TI = (M + NR - 1) / NR;
  localparam int N_TJ = (P + NC - 1) / NC;

  logic clk = 0;
  logic rst_n;
  always #5 clk = ~clk;

  // The design is clocked 100 ps after the testbench clock (Rev 16). This
  // bench assigns inputs with blocking statements immediately after
  // @(posedge clk), i.e. in the same instant as the edge, and the design was
  // written to see them at THAT edge -- which it did only because the
  // simulator happened to run this process first. Once the buffers moved into
  // submodules with derived write strobes, the order flipped and writes landed
  // a beat late. Skewing the DUT clock makes the intended semantics explicit
  // and deterministic without changing a single stimulus line, so every
  // result and every recorded cycle count is reproduced exactly.
  logic clk_dut;
  assign #0.1 clk_dut = clk;

  // free-running cycle counter, for the throughput comparison
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
    .drain_wide(drain_wide), .skip_en(skip_en), .tile_shape(2'd0), .routing_ctrl(routing_ctrl), .nm_pattern(nm_pattern), .layer_features(layer_features),
    .din_valid(din_valid), .din_data(din_data), .din_is_weight(din_is_weight),
    .din_ready(din_ready),
    .dout_valid(dout_valid), .dout_data(dout_data), .dout_vec(dout_vec),
    .dout_ready(dout_ready),
    .cfg_override_en(cfg_override_en), .cfg_override(cfg_override),
    .busy(busy), .active_config(active_config),
    .current_phase(current_phase), .flush_event(flush_event)
  );

  logic signed [DATA_W-1:0] A [M][K];
  logic signed [DATA_W-1:0] B [K][P];
  logic signed [ACC_W-1:0]  C_gold [M][P];
  logic signed [ACC_W-1:0]  C_rtl  [M][P];
  bit                       C_seen [M][P];

  int     errors = 0;
  int     checks = 0;
  longint acc;
  longint cyc_narrow, cyc_wide, cyc_bp, cyc_skip;
  integer rseed = 32'd7;   // $random needs an integer variable for its seed

  // ---------------------------------------------------------------------
  // One complete GEMM at the given output-bus width. Returns the number of
  // cycles from `start` to the acceptance of the final output beat.
  // ---------------------------------------------------------------------
  // bp = randomly deassert dout_ready. With the wide bus the drain is shorter
  // than the compute, so without backpressure the ST_GEMM_CAP stall path -- the
  // one that must freeze the accumulators when the consumer falls behind -- is
  // never exercised. This makes sure it is.
  task automatic run_gemm(bit wide, bit bp, bit skipz, output longint elapsed);
    int  per_tile, total_beats, beat, tidx, ti, tj, rr, cc, rem, guard;
    longint cyc0;

    for (int r = 0; r < M; r++)
      for (int c = 0; c < P; c++) begin
        C_rtl [r][c] = 'x;
        C_seen[r][c] = 1'b0;
      end

    rst_n = 0;
    start = 0; gemm_en = 0; din_valid = 0; din_is_weight = 0; din_data = 0;
    dout_ready = 0; cfg_override_en = 1; cfg_override = CFG_D_M_R;
    drain_wide = wide;
    skip_en    = skipz;
    model_type = 2'd0; op_type = 2'd0; batch_size = 16'd0;
    dim_m = M[15:0]; dim_k = K[15:0]; dim_p = P[15:0];
    repeat (4) @(posedge clk);
    rst_n = 1;
    repeat (2) @(posedge clk);

    cyc0 = cycles;
    gemm_en = 1;
    start = 1;
    @(posedge clk);
    start = 0;

    // B first (weights), row-major k then c
    din_valid = 1; din_is_weight = 1;
    for (int k = 0; k < K; k++)
      for (int c = 0; c < P; c++) begin
        din_data = B[k][c];
        @(posedge clk);
      end

    // then A (activations), row-major r then k
    din_is_weight = 0;
    for (int r = 0; r < M; r++)
      for (int k = 0; k < K; k++) begin
        din_data = A[r][k];
        @(posedge clk);
      end
    din_valid = 0;

    // ---- collect, purely by the valid/ready protocol ----
    // dout_data / dout_vec are combinational on the drain pointers, so the
    // value visible just after edge N is the beat that edge N+1 consumes.
    per_tile    = wide ? NC : (NR * NC);
    total_beats = N_TI * N_TJ * per_tile;
    beat        = 0;
    guard       = 0;
    dout_ready  = 1;

    while (beat < total_beats && guard < 40 * total_beats + 100000) begin
      @(posedge clk);
      #1;
      guard++;
      // Set ready for the window that ends at the NEXT edge, then decide
      // whether the beat currently on the bus is accepted in that window.
      dout_ready = bp ? (($random(rseed) % 4) != 0) : 1'b1;
      if (dout_valid && dout_ready) begin
        tidx = beat / per_tile;
        ti   = tidx / N_TJ;
        tj   = tidx % N_TJ;
        rem  = beat % per_tile;
        if (wide) begin
          cc = rem;
          for (rr = 0; rr < NR; rr++)
            if ((ti * NR + rr) < M && (tj * NC + cc) < P) begin
              C_rtl [ti * NR + rr][tj * NC + cc] = dout_vec[rr];
              C_seen[ti * NR + rr][tj * NC + cc] = 1'b1;
            end
        end else begin
          cc = rem / NR;
          rr = rem % NR;
          if ((ti * NR + rr) < M && (tj * NC + cc) < P) begin
            C_rtl [ti * NR + rr][tj * NC + cc] = dout_data;
            C_seen[ti * NR + rr][tj * NC + cc] = 1'b1;
          end
        end
        beat++;
      end
    end
    @(posedge clk);
    elapsed = cycles - cyc0;
    dout_ready = 0;
    gemm_en = 0;

    if (beat < total_beats) begin
      errors++;
      $display("  [FAIL] only %0d of %0d output beats appeared", beat, total_beats);
    end
  endtask

  task automatic check_result(string label);
    int local_err = 0;
    for (int r = 0; r < M; r++)
      for (int c = 0; c < P; c++) begin
        checks++;
        if (!C_seen[r][c]) begin
          local_err++;
          if (local_err <= 4)
            $display("  [FAIL] %s: C[%0d][%0d] never produced", label, r, c);
        end else if (C_rtl[r][c] !== C_gold[r][c]) begin
          local_err++;
          if (local_err <= 4)
            $display("  [FAIL] %s: C[%0d][%0d] = %0d, expected %0d",
                     label, r, c, C_rtl[r][c], C_gold[r][c]);
        end
      end
    errors += local_err;
    if (local_err == 0)
      $display("  [PASS] %-14s all %0d output elements match the reference GEMM",
               label, M * P);
    else
      $display("  [FAIL] %-14s %0d of %0d elements wrong", label, local_err, M * P);
  endtask

  initial begin
    $display("============================================================");
    $display("  versaaccel_top -- TILED GEMM golden test");
    $display("  C[%0dx%0d] = A[%0dx%0d] x B[%0dx%0d]   array %0dx%0d",
             M, P, M, K, K, P, NR, NC);
    $display("  tiles: %0d x %0d", N_TI, N_TJ);
    $display("============================================================");

    // ---- stimulus: signed, includes zeros, no trivial symmetry ----
    for (int r = 0; r < M; r++)
      for (int k = 0; k < K; k++)
        A[r][k] = $signed(((r * 3 + k * 7) % 11) - 5);   // -5..5
    for (int k = 0; k < K; k++)
      for (int c = 0; c < P; c++)
        B[k][c] = $signed(((k * 5 + c * 3) % 13) - 6);   // -6..6

    // ---- independent reference ----
    for (int r = 0; r < M; r++)
      for (int c = 0; c < P; c++) begin
        acc = 0;
        for (int k = 0; k < K; k++)
          acc += longint'(A[r][k]) * longint'(B[k][c]);
        C_gold[r][c] = acc[ACC_W-1:0];
      end

    run_gemm(1'b0, 1'b0, 1'b0, cyc_narrow);
    check_result("narrow drain");

    run_gemm(1'b1, 1'b0, 1'b0, cyc_wide);
    check_result("wide drain");

    run_gemm(1'b1, 1'b1, 1'b0, cyc_bp);
    check_result("wide + stalls");

    // Zero-skipping must not change a single output element. These operands
    // are dense, so nothing should actually be skipped either.
    run_gemm(1'b1, 1'b0, 1'b1, cyc_skip);
    check_result("wide + skip_en");

    // ---- throughput ----
    // Operand load is the same in both runs (K*P + M*K cycles) and is not what
    // the double buffer addresses; it is reported separately so the drain's
    // contribution is visible rather than diluted.
    $display("------------------------------------------------------------");
    $display("  THROUGHPUT  (cycles from start to final output beat)");
    $display("    operand load (fixed):   %0d", K*P + M*K);
    $display("    narrow drain, 1 elem/cy:  %0d cycles", cyc_narrow);
    $display("    wide drain, %0d elem/cy:   %0d cycles", NR, cyc_wide);
    $display("    speedup from the wide bus: %0.2fx",
             real'(cyc_narrow) / real'(cyc_wide));
    $display("    wide with random backpressure: %0d cycles", cyc_bp);
    $display("    wide with skip_en on dense operands: %0d cycles (+%0d for the",
             cyc_skip, cyc_skip - cyc_wide);
    $display("      per-tile schedule-priming cycle; nothing is skippable here)");
    $display("    tiles %0d, compute %0d cycles/tile (K+NR+NC-1)",
             N_TI * N_TJ, K + NR + NC - 1);
    $display("    ideal wide  = load + tiles*(K+NR+NC-1) + drain of last tile");
    $display("                 + 4 (operand-feed stages + output register, Rev 20)");
    $display("               ~= %0d", K*P + M*K + N_TI*N_TJ*(K+NR+NC-1) + NC + 4);

    $display("------------------------------------------------------------");
    $display("  sample: C[0][0]=%0d (exp %0d)   C[%0d][%0d]=%0d (exp %0d)",
             C_rtl[0][0], C_gold[0][0], M-1, P-1,
             C_rtl[M-1][P-1], C_gold[M-1][P-1]);
    if (errors == 0)
      $display("  [PASS] %0d checks across 4 runs (narrow, wide, wide+stalls, skip)", checks);
    else
      $display("  [FAIL] %0d of %0d checks failed", errors, checks);
    $display("============================================================");
    $finish;
  end

  initial begin
    #10000000;
    $display("  [FAIL] timeout");
    $finish;
  end

endmodule
