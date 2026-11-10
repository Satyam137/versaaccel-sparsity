// =============================================================================
// tb_switch.sv -- B2: what does switching configuration between layers cost?
//
// Section 4's scheduler charges SWITCH_COST[from][to] cycles whenever two
// consecutive layers run in different configurations -- 100 to 250 cycles,
// an estimate for an 8x8 array (pipeline drain, interconnect reset, control
// word load, refill). This measures it on versaaccel_top.
//
// Part 1 -- forced configuration. Two GEMMs back to back, layer 1 in config a,
// layer 2 in config b, for all 16 (a, b). The cost of the switch is
//     T(a -> b) - (T(a -> a) + T(b -> b)) / 2
// i.e. the time of the mixed pair beyond the average of the two pure pairs, so
// that each config's own compute time cancels. Every output of every GEMM is
// checked against a reference.
//
// Part 2 -- predicted configuration (cfg_override_en low). The CPE predicts
// from the layer features captured on `start`, the speculation unit commits,
// and selected_config follows. Measured: cycles from `start` to each, against
// the length of the operand load the prediction overlaps.
//
// Prints "SWITCH a b cost" lines, "LATENCY name cycles" lines and one
// [PASS]/[FAIL] line. Run: see experiments/switch_cost_rtl.py.
// =============================================================================
`timescale 1ns/1ps
module tb_switch;
  import versaaccel_pkg::*;

  localparam int M = 16, K = 16, P = 32;
  localparam int N_TI = (M + NR - 1) / NR;
  localparam int N_TJ = (P + NC - 1) / NC;

  logic clk = 0;
  logic rst_n;
  always #5 clk = ~clk;
  logic clk_dut;
  assign #0.1 clk_dut = clk;          // as tb_gemm_golden: DUT sees inputs set at the edge

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
  logic [BENES_CTRL_W-1:0]  routing_ctrl = '0;
  logic [2:0]               nm_pattern = 3'd0;
  logic signed [7:0]        layer_features [9];

  versaaccel_top dut (
    .clk(clk_dut), .rst_n(rst_n), .start(start),
    .model_type(model_type), .batch_size(batch_size), .op_type(op_type),
    .gemm_en(gemm_en), .dim_m(dim_m), .dim_k(dim_k), .dim_p(dim_p),
    .drain_wide(drain_wide), .skip_en(skip_en), .tile_shape(2'd0),
    .routing_ctrl(routing_ctrl), .nm_pattern(nm_pattern), .layer_features(layer_features),
    .din_valid(din_valid), .din_data(din_data), .din_is_weight(din_is_weight),
    .din_ready(din_ready),
    .dout_valid(dout_valid), .dout_data(dout_data), .dout_vec(dout_vec),
    .dout_ready(dout_ready),
    .cfg_override_en(cfg_override_en), .cfg_override(cfg_override),
    .busy(busy), .active_config(active_config),
    .current_phase(current_phase), .flush_event(flush_event)
  );

  logic signed [DATA_W-1:0] A [2][M][K];
  logic signed [DATA_W-1:0] B [2][K][P];
  int errors = 0, checks = 0;
  integer seed = 32'd5;

  // One GEMM on operand set `s`, from `start` to its last accepted output beat.
  task automatic gemm(int s);
    int beat, total, tidx, ti, tj, cc, guard;
    longint acc;
    logic signed [ACC_W-1:0] C [M][P];
    gemm_en = 1; start = 1;
    @(posedge clk);
    start = 0;
    din_valid = 1; din_is_weight = 1;
    for (int k = 0; k < K; k++)
      for (int c = 0; c < P; c++) begin din_data = B[s][k][c]; @(posedge clk); end
    din_is_weight = 0;
    for (int r = 0; r < M; r++)
      for (int k = 0; k < K; k++) begin din_data = A[s][r][k]; @(posedge clk); end
    din_valid = 0;
    total = N_TI * N_TJ * NC; beat = 0; guard = 0; dout_ready = 1;
    while (beat < total && guard < 100000) begin
      @(posedge clk); #1; guard++;
      if (dout_valid && dout_ready) begin
        tidx = beat / NC; ti = tidx / N_TJ; tj = tidx % N_TJ; cc = beat % NC;
        for (int rr = 0; rr < NR; rr++)
          if (ti * NR + rr < M && tj * NC + cc < P) C[ti * NR + rr][tj * NC + cc] = dout_vec[rr];
        beat++;
      end
    end
    @(posedge clk);
    dout_ready = 0;
    if (beat < total) begin errors++; $display("  [FAIL] only %0d of %0d beats", beat, total); end
    for (int r = 0; r < M; r++)
      for (int c = 0; c < P; c++) begin
        acc = 0;
        for (int k = 0; k < K; k++) acc += longint'(A[s][r][k]) * longint'(B[s][k][c]);
        checks++;
        if (C[r][c] !== ACC_W'(acc)) begin
          errors++;
          if (errors <= 5) $display("  [FAIL] set %0d C[%0d][%0d] = %0d, expected %0d", s, r, c, C[r][c], acc);
        end
      end
    while (busy) @(posedge clk);         // back in ST_IDLE: the next layer may start
    gemm_en = 0;
  endtask

  task automatic reset_dut();
    rst_n = 0; start = 0; gemm_en = 0; din_valid = 0; din_is_weight = 0; din_data = 0;
    dout_ready = 0; drain_wide = 1; skip_en = 1;
    model_type = 2'd2; op_type = 2'd0; batch_size = 16'd0;
    dim_m = M[15:0]; dim_k = K[15:0]; dim_p = P[15:0];
    repeat (4) @(posedge clk);
    rst_n = 1;
    repeat (2) @(posedge clk);
  endtask

  // Two layers back to back: set 0 in config a, then set 1 in config b.
  task automatic pair(config_t a, config_t b, output longint t);
    longint t0;
    reset_dut();
    cfg_override_en = 1;
    cfg_override = a;
    t0 = cycles;
    gemm(0);
    cfg_override = b;
    gemm(1);
    t = cycles - t0;
  endtask

  longint T [4][4];
  longint t_start, t_cpe, t_spec, t_sel, load_len;
  config_t sel0;
  logic [7:0] vec_mem [64*9];
  int found;

  initial begin
    for (int f = 0; f < 9; f++) layer_features[f] = 8'sd0;
    // Operands: half the A entries zero in whole k-slices, so the skipping
    // configurations really skip and the configs differ in their own time.
    for (int s = 0; s < 2; s++) begin
      for (int r = 0; r < M; r++)
        for (int k = 0; k < K; k++)
          A[s][r][k] = (k % 4 == 1 + s) ? '0 : DATA_W'($random(seed) % 9);
      for (int k = 0; k < K; k++)
        for (int c = 0; c < P; c++)
          B[s][k][c] = DATA_W'($random(seed) % 9);
    end

    $display("============================================================");
    $display("  B2 -- configuration switch cost between layers (versaaccel_top)");
    $display("  two %0dx%0dx%0d GEMMs back to back, skip_en on", M, K, P);
    $display("============================================================");

    // ---- Part 1: forced configurations ----------------------------------
    for (int a = 0; a < 4; a++)
      for (int b = 0; b < 4; b++)
        pair(config_t'(a), config_t'(b), T[a][b]);
    for (int a = 0; a < 4; a++)
      for (int b = 0; b < 4; b++)
        $display("SWITCH %0d %0d %0d   (pair %0d cycles)", a, b,
                 T[a][b] - (T[a][a] + T[b][b]) / 2, T[a][b]);

    // ---- Part 2: predicted configuration --------------------------------
    // Layers from the CPE's own test vectors, until one predicts a config
    // different from the one already selected, so the chain really switches.
    $readmemh({`CPE_WEIGHT_DIR, "/cpe_test_vectors.hex"}, vec_mem);
    found = 0;
    for (int v = 0; v < 64 && !found; v++) begin
      reset_dut();
      cfg_override_en = 0;
      for (int f = 0; f < 9; f++) layer_features[f] = $signed(vec_mem[v * 9 + f]);
      repeat (3) @(posedge clk);                       // let reset defaults settle
      sel0 = dut.selected_config;
      t_cpe = -1; t_spec = -1; t_sel = -1;
      t_start = cycles;
      gemm_en = 1; start = 1;
      @(posedge clk);
      start = 0; gemm_en = 1;
      for (int c = 0; c < 40; c++) begin
        @(posedge clk); #1;
        if (t_cpe  < 0 && dut.cpe_valid)                  t_cpe  = cycles - t_start;
        if (t_spec < 0 && dut.spec_config != sel0)        t_spec = cycles - t_start;
        if (t_sel  < 0 && dut.selected_config != sel0)    t_sel  = cycles - t_start;
      end
      if (t_sel >= 0) begin
        found = 1;
        $display("PREDICT vector %0d: %0d -> %0d (confidence %0d)", v, int'(sel0),
                 int'(dut.selected_config), int'(dut.cpe_confidence));
      end
    end
    load_len = K * P + M * K;
    $display("LATENCY cpe_valid        %0d", t_cpe);
    $display("LATENCY spec_config      %0d", t_spec);
    $display("LATENCY selected_config  %0d", t_sel);
    $display("LATENCY operand_load     %0d", load_len);
    if (!found) $display("  NOTE: no test vector moved the selected configuration");

    $display("------------------------------------------------------------");
    if (errors == 0)
      $display("  [PASS] %0d output checks across 33 GEMMs, every configuration pair", checks);
    else
      $display("  [FAIL] %0d of %0d checks failed", errors, checks);
    $finish;
  end
endmodule
