// =============================================================================
// tb_gemm_sparse.sv -- what does zero-skipping actually buy?
//
// FIX_PLAN.md Phase 6, T4. The cost model charges by scalar nnz: cost falls in
// proportion to the NUMBER of zeros, wherever they are. A systolic array cannot
// honour that, because operands are shared -- a_in[r] feeds every PE in row r
// -- so a step k can only be dropped when it is useless for the whole tile:
//
//     mask[k] = (A[.][k] nonzero somewhere in the tile's rows)
//             & (B[k][.] nonzero somewhere in the tile's columns)
//
// This bench measures the gap between those two by running the SAME shapes at
// several sparsity patterns of identical density, with skipping off and on.
// Every run is checked against an independent reference, so a pattern that
// saves nothing still has to compute the right answer.
//
// Shape: A[16x16] x B[16x32], chosen to fill a whole 16-row tile (so the
// all-zero-column condition is over the full NR) while fitting IBUF (256) and
// WBUF (512).
//
// Run:
//   iverilog -g2012 -o tb_gs -DCPE_WEIGHT_DIR='"<abs>/rtl/weights"' \
//       common/versaaccel_pkg.sv versaaccel_rtl/*.sv \
//       testbenches/tb_gemm_sparse.sv
//   ./tb_gs
// =============================================================================
`timescale 1ns/1ps

module tb_gemm_sparse;
  import versaaccel_pkg::*;

  // Layer configuration pins. Tied to the values the RTL used internally
  // before they became ports (routing straight-through, N:M pattern 0, all
  // CPE features 0), so every check below means exactly what it meant then.
  logic [BENES_CTRL_W-1:0] routing_ctrl = '0;
  logic [2:0]              nm_pattern   = 3'd0;
  logic signed [7:0]       layer_features [9];
  initial for (int f = 0; f < 9; f++) layer_features[f] = 8'sd0;

  // Shape is overridable so the same patterns can be run at a larger K, where
  // the NR+NC-2 skew fill is amortised over more useful steps:
  //   iverilog ... -DSP_M=4 -DSP_K=64 -DSP_P=8
  // The buffers (IBUF 256, WBUF 512) force M*K <= 256, so a larger K means
  // fewer rows -- which makes the UNSTRUCTURED rows of that table optimistic,
  // since an all-zero k-slice then only has to be zero across M rows, not NR.
  // The k-slice rows are exact at any M.
`ifdef SP_M
  localparam int M = `SP_M;
`else
  localparam int M = NR;
`endif
`ifdef SP_K
  localparam int K = `SP_K;
`else
  localparam int K = NR;
`endif
`ifdef SP_P
  localparam int P = `SP_P;
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
  integer rseed = 32'd12345;

  function automatic logic signed [DATA_W-1:0] nzval(int i);
    // never zero, so a zero in A or B is always one the pattern put there
    return $signed((i % 9) - 4 + (((i % 9) - 4) >= 0 ? 1 : 0));
  endfunction

  // ---------------------------------------------------------------------
  // One GEMM. Returns cycles for the whole run and for the compute phase
  // alone (from the last operand element to the final output beat) -- the
  // operand load is a fixed cost of the resident-operand design and would
  // otherwise swamp the effect being measured.
  // ---------------------------------------------------------------------
  task automatic run_gemm(bit skipz, output longint total, output longint compute);
    int  per_tile, total_beats, beat, tidx, ti, tj, rr, cc, rem, guard;
    longint cyc0, cyc_load;

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

    cyc0 = cycles;
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

    per_tile    = NC;
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
        for (rr = 0; rr < NR; rr++)
          if ((ti * NR + rr) < M && (tj * NC + cc) < P) begin
            C_rtl [ti * NR + rr][tj * NC + cc] = dout_vec[rr];
            C_seen[ti * NR + rr][tj * NC + cc] = 1'b1;
          end
        beat++;
      end
    end
    @(posedge clk);
    total   = cycles - cyc0;
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

  // ---------------------------------------------------------------------
  // Patterns. All zeros are put in deliberately; nzval() never returns 0.
  // ---------------------------------------------------------------------
  task automatic build(int pat);
    int idx, nz;
    bit zero_k [K];
    // start dense
    idx = 0;
    for (int r = 0; r < M; r++)
      for (int k = 0; k < K; k++) begin A[r][k] = nzval(idx); idx++; end
    for (int k = 0; k < K; k++)
      for (int c = 0; c < P; c++) begin B[k][c] = nzval(idx + 3); idx++; end

    case (pat)
      0: ;                                              // dense
      1: for (int r = 0; r < M; r++)                    // unstructured 50%
           for (int k = 0; k < K; k++)
             if (({$random(rseed)} % 100) < 50) A[r][k] = '0;
      2: for (int r = 0; r < M; r++)                    // unstructured 90%
           for (int k = 0; k < K; k++)
             if (({$random(rseed)} % 100) < 90) A[r][k] = '0;
      3: for (int r = 0; r < M; r++)                    // 2:4 along k
           for (int g = 0; g < K/4; g++) begin
             int drop0, drop1;
             drop0 = {$random(rseed)} % 4;
             drop1 = (drop0 + 1 + {$random(rseed)} % 3) % 4;
             A[r][g*4 + drop0] = '0;
             A[r][g*4 + drop1] = '0;
           end
      4: begin                                          // 50% of k dead in A
           for (int k = 0; k < K; k++) zero_k[k] = (k % 2 == 0);
           for (int r = 0; r < M; r++)
             for (int k = 0; k < K; k++) if (zero_k[k]) A[r][k] = '0;
         end
      5: begin                                          // 50% of k dead in B
           for (int k = 0; k < K; k++) zero_k[k] = (k % 2 == 1);
           for (int k = 0; k < K; k++)
             if (zero_k[k]) for (int c = 0; c < P; c++) B[k][c] = '0;
         end
      6: begin                                          // 25% dead in each,
           for (int r = 0; r < M; r++)                  // disjoint -> 50% union
             for (int k = 0; k < K; k++) if (k % 4 == 0) A[r][k] = '0;
           for (int k = 0; k < K; k++)
             if (k % 4 == 1) for (int c = 0; c < P; c++) B[k][c] = '0;
         end
    endcase

    for (int r = 0; r < M; r++)
      for (int c = 0; c < P; c++) begin
        acc = 0;
        for (int k = 0; k < K; k++)
          acc += longint'(A[r][k]) * longint'(B[k][c]);
        C_gold[r][c] = acc[ACC_W-1:0];
      end
  endtask

  function automatic real zero_frac_a();
    int z = 0;
    for (int r = 0; r < M; r++)
      for (int k = 0; k < K; k++) if (A[r][k] == 0) z++;
    return real'(z) / real'(M * K);
  endfunction

  function automatic real zero_frac_b();
    int z = 0;
    for (int k = 0; k < K; k++)
      for (int c = 0; c < P; c++) if (B[k][c] == 0) z++;
    return real'(z) / real'(K * P);
  endfunction

  string pname [7];
  longint t_off, c_off, t_on, c_on;
  real    zf, zfb, saved, predicted;

  initial begin
    pname[0] = "dense";
    pname[1] = "unstructured 50%";
    pname[2] = "unstructured 90%";
    pname[3] = "2:4 along k";
    pname[4] = "k-slice 50% (A)";
    pname[5] = "k-slice 50% (B)";
    pname[6] = "k-slice 25%+25%";

    $display("============================================================");
    $display("  Zero-skipping: what the dataflow can actually exploit");
    $display("  A[%0dx%0d] x B[%0dx%0d], array %0dx%0d, %0d tiles",
             M, K, K, P, NR, NC, N_TI * N_TJ);
    $display("============================================================");
    $display("  pattern             zeros   skip off   skip on   measured   nnz-model");
    $display("                        A/B     cycles    cycles     saving     predicts");
    $display("  ----------------------------------------------------------------------");

    for (int pat = 0; pat < 7; pat++) begin
      build(pat);
      zf  = zero_frac_a();
      zfb = zero_frac_b();

      run_gemm(1'b0, t_off, c_off);
      check({pname[pat], " off"});
      run_gemm(1'b1, t_on, c_on);
      check({pname[pat], " on"});

      saved     = 1.0 - real'(c_on) / real'(c_off);
      // The cost model's assumption: compute scales with the nonzero count.
      predicted = (zf > zfb) ? zf : zfb;

      $display("  %-19s %3.0f/%-3.0f%%   %6d     %6d     %5.1f%%       %5.1f%%",
               pname[pat], zf * 100.0, zfb * 100.0, c_off, c_on,
               saved * 100.0, predicted * 100.0);
    end

    $display("  ----------------------------------------------------------------------");
    $display("  'skip on' includes one schedule-priming cycle per tile, so a");
    $display("  pattern with nothing to skip costs slightly MORE than skip off.");
    $display("  Ceiling for a perfectly skippable fraction f is f*K/(K+NR+NC),");
    $display("  not f: the %0d skew-fill cycles per tile are not skippable.",
             NR + NC);
    $display("------------------------------------------------------------");
    if (errors == 0)
      $display("  [PASS] all %0d output checks correct in every pattern, both modes", checks);
    else
      $display("  [FAIL] %0d of %0d checks failed", errors, checks);
    $display("============================================================");
    $finish;
  end

  initial begin
    #20000000;
    $display("  [FAIL] timeout");
    $finish;
  end

endmodule
