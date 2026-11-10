// =============================================================================
// tb_stream.sv -- operands and results moving at the same time (B10, B12)
//
// Every other GEMM bench loads all of A and B, ignoring din_ready, and only
// then starts reading results. That is the only protocol the resident-operand
// design allows, but a design that streams activations (VERSAACCEL_STREAM)
// produces results while A is still arriving and stalls the loader when its
// ring buffer is full -- so this bench runs the loader and the collector
// concurrently, each obeying its valid/ready handshake, and can stall either
// side at random.
//
// It runs on every build (without STREAM, din_ready is simply high for the
// whole load) and checks every element of C against a reference computed
// here, with zero-skipping off and on. The cycle count is end to end: from
// `start` to the acceptance of the final output beat, load included -- the
// figure B12 is meant to move.
//
// Defines: ST_M, ST_K, ST_P (shape). Plusargs: +A=<file> +B=<file> (hex,
// row-major, 16-bit two's complement), +SHAPE=0|1|2 (needs RESHAPE for 1, 2),
// +BP=<percent> to drop din_valid and dout_ready on that share of cycles.
// Prints "RESULT <cycles skip off> <cycles skip on> <checks> <errors>".
// =============================================================================
`timescale 1ns/1ps

module tb_stream;
  import versaaccel_pkg::*;

  localparam int M = `ST_M;
  localparam int K = `ST_K;
  localparam int P = `ST_P;
  logic [1:0] shape_sel = 2'd0;
  int TR, TC, N_TI, N_TJ, BP = 0;

  logic [BENES_CTRL_W-1:0] routing_ctrl = '0;
  logic [2:0]              nm_pattern   = 3'd0;
  logic signed [7:0]       layer_features [9];
  initial for (int f = 0; f < 9; f++) layer_features[f] = 8'sd0;

  logic clk = 0;
  logic rst_n;
  always #5 clk = ~clk;
  logic clk_dut;
  assign #0.1 clk_dut = clk;      // same skew as tb_gemm_sparse / tb_regroup

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
  int     seed_in  = 1;          // independent streams for the two stalls
  int     seed_out = 7919;
  longint acc;

  // B then A, one element per accepted beat. The value is presented just
  // after the TB edge; the DUT samples it 0.1 ns later together with
  // din_ready, which depends on DUT state only -- so reading din_ready in
  // between says whether this beat is the one consumed.
  task automatic load_operands();
    int  e = 0;
    int  total = K * P + M * K;
    bit  gap;
    while (e < total) begin
      @(posedge clk);
      gap = (BP > 0) && (($urandom(seed_in) % 100) < BP);
      din_valid     = !gap;
      din_is_weight = (e < K * P);
      din_data      = (e < K * P) ? B[e / P][e % P]
                                  : A[(e - K * P) / K][(e - K * P) % K];
      #0.05;
      if (din_valid && din_ready) e++;
    end
    @(posedge clk);
    din_valid = 0;
  endtask

  task automatic collect();
    int per_tile    = NC;                 // physical columns drained per tile
    int total_beats = N_TI * N_TJ * per_tile;
    int beat = 0, guard = 0, tidx, ti, tj, cc;
    while (beat < total_beats && guard < 2000000) begin
      @(posedge clk);
      #1;
      guard++;
      dout_ready = !((BP > 0) && (($urandom(seed_out) % 100) < BP));
      if (dout_valid && dout_ready) begin
        tidx = beat / per_tile;
        ti = tidx / N_TJ; tj = tidx % N_TJ;
        cc = beat % per_tile;
        for (int rr = 0; rr < NR; rr++) begin
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
    if (beat < total_beats) begin
      errors++;
      $display("  [FAIL] only %0d of %0d beats", beat, total_beats);
    end
  endtask

  task automatic run_gemm(bit skipz, output longint total);
    longint cyc0;
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
    fork
      load_operands();
      collect();
    join
    @(posedge clk);
    total = cycles - cyc0;
    dout_ready = 0; gemm_en = 0;
    repeat (4) @(posedge clk);
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
  string  fa, fb;

  initial begin
    if (!$value$plusargs("A=%s", fa) || !$value$plusargs("B=%s", fb)) begin
      $display("RESULT usage: +A=<hex> +B=<hex>");
      $finish;
    end
    if (!$value$plusargs("SHAPE=%d", shape_sel)) shape_sel = 2'd0;
    if (!$value$plusargs("BP=%d", BP)) BP = 0;
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
    run_gemm(1'b1, c_on);
    check("skip on");
    $display("RESULT %0d %0d %0d %0d", c_off, c_on, checks, errors);
    $finish;
  end

  initial begin
    #500000000;
    $display("RESULT timeout");
    $finish;
  end
endmodule
