// =============================================================================
// tb_sigma_golden.sv -- ARITHMETIC verification of the SIGMA baseline
//
// SIGMA is one of the two baselines VersaAccel is compared against, so an
// arithmetic error here corrupts every comparison in the paper. This bench
// checks every output element against an independent model, over:
//
//   * all three FAN modes -- FULL, PARTIAL, BYPASS -- which produce different
//     results, so a hard-wired mode (the defect that let synthesis delete the
//     FAN) fails immediately
//   * K = 16 and 32 (whole chunks) and K = 5 and 21 (partial chunks), which
//     exercise the masking of BOTH operands past K
//   * identity routing and two scrambled Benes routings, where activations
//     land on different lanes and must still meet the right weights
//   * random backpressure on the drain
//   * signed operands spanning -128..127, including zeros, so a signedness
//     slip in the multiply shows as a mismatch rather than passing by luck
//
// Reference model, per chunk c (LANES = 16 elements of x):
//   xin[j] = (c*16+j < K) ? x[c*16+j] : 0
//   xt     = Benes(xin, routing)                   -- via a reference instance
//   p[u][j]= ((c*16+j < K) ? W[u][c*16+j] : 0) * xt[j]
//   FULL:    g[u][0]   += sum_j p[u][j]
//   PARTIAL: g[u][j/2] += p[u][j]
//   BYPASS:  g[u][j]   += p[u][j]
// The permutation comes from a separate distribution_network instance: the
// Benes itself is verified independently by tb_datapath_golden, and what this
// bench tests is SIGMA's use of it -- lane alignment, masking and pipelining.
//
// It also counts compute cycles. One chunk -- 16 activations, 256 multiplies
// -- must retire per cycle, i.e. COMPUTE+FLUSH = ceil(K/16) + 3. That is the
// check that the design really is 256 multipliers wide, not 16 time-shared.
// =============================================================================
`timescale 1ns/1ps

module tb_sigma_golden;
  import versaaccel_pkg::*;

  localparam int UNITS = NR;
  localparam int LANES = NC;
  localparam int KMAX  = WBUF_DEPTH / UNITS;

  logic clk = 0;
  logic rst_n;
  always #5 clk = ~clk;

  logic                     start;
  logic [15:0]              num_elements;
  logic [BENES_CTRL_W-1:0]  routing_ctrl;
  logic [1:0]               reduce_mode;
  logic                     din_valid, din_is_weight, din_ready;
  logic signed [DATA_W-1:0] din_data;
  logic                     dout_valid, dout_ready;
  logic signed [ACC_W-1:0]  dout_data;
  logic                     busy;

  sigma_top dut (
    .clk(clk), .rst_n(rst_n), .start(start),
    .num_elements(num_elements), .routing_ctrl(routing_ctrl),
    .reduce_mode(reduce_mode),
    .din_valid(din_valid), .din_data(din_data), .din_is_weight(din_is_weight),
    .din_ready(din_ready),
    .dout_valid(dout_valid), .dout_data(dout_data), .dout_ready(dout_ready),
    .busy(busy)
  );

  // Reference Benes, used only to obtain the permutation for a routing word
  logic [BENES_CTRL_W-1:0]  ref_ctrl;
  logic signed [DATA_W-1:0] ref_in  [LANES];
  logic signed [DATA_W-1:0] ref_out [LANES];
  distribution_network ref_benes (.ctrl(ref_ctrl), .data_in(ref_in), .data_out(ref_out));

  logic signed [DATA_W-1:0] W [UNITS][KMAX];
  logic signed [DATA_W-1:0] X [KMAX];
  logic signed [ACC_W-1:0]  G [UNITS][LANES];

  int errors = 0, checks = 0, runs = 0;
  int seed;

  localparam logic [BENES_CTRL_W-1:0] ROUTE_A = BENES_CTRL_W'(64'hA5C3_96F0_1E2D_4B78);
  localparam logic [BENES_CTRL_W-1:0] ROUTE_B = BENES_CTRL_W'(64'h5A3C_690F_E1D2_B487);

  function automatic logic signed [DATA_W-1:0] rnd();
    int v;
    v = $urandom(seed) % 23;
    seed = seed + 7919;
    if (v < 3) return '0;                       // ~13% zeros
    return DATA_W'($signed(($urandom(seed) % 256) - 128));
  endfunction

  task automatic golden(input int K, input int mode, input logic [BENES_CTRL_W-1:0] rc);
    int n_chunks;
    logic signed [ACC_W-1:0] p, w;
    n_chunks = (K + LANES - 1) / LANES;
    for (int u = 0; u < UNITS; u++)
      for (int j = 0; j < LANES; j++) G[u][j] = '0;
    for (int c = 0; c < n_chunks; c++) begin
      ref_ctrl = rc;
      for (int j = 0; j < LANES; j++)
        ref_in[j] = (c * LANES + j < K) ? X[c * LANES + j] : DATA_W'(0);
      #1;
      for (int u = 0; u < UNITS; u++)
        for (int j = 0; j < LANES; j++) begin
          w = (c * LANES + j < K) ? ACC_W'($signed(W[u][c * LANES + j])) : ACC_W'(0);
          p = w * ACC_W'($signed(ref_out[j]));
          case (mode)
            0:       G[u][0]     = G[u][0]     + p;
            1:       G[u][j / 2] = G[u][j / 2] + p;
            default: G[u][j]     = G[u][j]     + p;
          endcase
        end
    end
  endtask

  task automatic send(input logic is_w, input logic signed [DATA_W-1:0] d);
    @(negedge clk);
    while (!din_ready) @(negedge clk);
    din_valid = 1; din_is_weight = is_w; din_data = d;
    @(negedge clk);
    din_valid = 0;
  endtask

  // One full pass: load, compute, drain, check.
  task automatic run(input int K, input int mode, input logic [BENES_CTRL_W-1:0] rc,
                     input bit backpressure, input string label);
    int n_lanes, n_out, idx, u, j, compute_cycles, want_cycles, errs0, Ke;
    bit rdy;
    errs0 = errors;
    Ke = (K > KMAX) ? KMAX : K;     // the design clamps num_elements to K_MAX
    runs++;
    for (int uu = 0; uu < UNITS; uu++)
      for (int k = 0; k < KMAX; k++) W[uu][k] = rnd();
    for (int k = 0; k < KMAX; k++) X[k] = rnd();
`ifdef POWER_HEX
    // B1c: real operands (experiments/make_activity.py): W[UNITS][KMAX]
    // row-major, then X[KMAX], 16-bit two's complement.
    begin
      logic [DATA_W-1:0] mem [UNITS*KMAX + KMAX];
      $readmemh(`POWER_HEX, mem);
      for (int uu = 0; uu < UNITS; uu++)
        for (int k = 0; k < KMAX; k++) W[uu][k] = $signed(mem[uu * KMAX + k]);
      for (int k = 0; k < KMAX; k++) X[k] = $signed(mem[UNITS * KMAX + k]);
    end
`endif
    golden(Ke, mode, rc);

    n_lanes = (mode == 0) ? 1 : (mode == 1) ? LANES / 2 : LANES;
    n_out   = UNITS * n_lanes;

    @(negedge clk);
    num_elements = K; routing_ctrl = rc; reduce_mode = mode[1:0];
    start = 1;
    @(negedge clk);
    start = 0;
    // scramble the config pins after start: the design must use its captured copies
    routing_ctrl = ~rc; reduce_mode = 2'(mode + 1); num_elements = 16'hFFFF;

    for (int uu = 0; uu < UNITS; uu++)
      for (int k = 0; k < Ke; k++) send(1'b1, W[uu][k]);
    for (int k = 0; k < Ke; k++) send(1'b0, X[k]);

    // compute + flush cycles (states 3 and 4)
    compute_cycles = 0;
    while (!dout_valid) begin
      @(posedge clk);
      if (dut.state == 3'd3 || dut.state == 3'd4) compute_cycles++;
    end
    want_cycles = (Ke + LANES - 1) / LANES + 3;
    checks++;
    if (compute_cycles != want_cycles) begin
      errors++;
      $display("  [FAIL] %s: %0d compute cycles, want %0d (one 256-multiply chunk per cycle)",
               label, compute_cycles, want_cycles);
    end

    idx = 0;
    while (idx < n_out) begin
      @(negedge clk);
      if (dout_valid) begin
        rdy = backpressure ? (($urandom(seed) % 4) != 0) : 1'b1;
        seed = seed + 13;
        dout_ready = rdy;
        if (rdy) begin
          u = idx / n_lanes; j = idx % n_lanes;
          checks++;
          if (dout_data !== G[u][j]) begin
            errors++;
            if (errors - errs0 <= 4)
              $display("  [FAIL] %s: out[%0d][%0d] = %0d, expected %0d",
                       label, u, j, dout_data, G[u][j]);
          end
          idx++;
        end
      end else dout_ready = 0;
    end
    @(negedge clk);
    dout_ready = 0;
    while (busy) @(negedge clk);

    $display("  %-34s K=%-3d %3d outputs, %2d compute cycles  %s",
             label, K, n_out, compute_cycles, (errors == errs0) ? "ok" : "MISMATCH");
  endtask

  initial begin
    $display("============================================================");
    $display("  SIGMA baseline -- GOLDEN VECTOR (arithmetic) Testbench");
    $display("  %0d Flex-DPEs x %0d lanes = %0d multipliers, K_MAX=%0d",
             UNITS, LANES, UNITS * LANES, KMAX);
    $display("============================================================");
    checks++;
    if (BENES_N != LANES) begin
      errors++;
      $display("  [FAIL] BENES_N=%0d but LANES=%0d: one Benes cannot route a chunk", BENES_N, LANES);
    end

    seed = 20260928;
    rst_n = 0; start = 0; din_valid = 0; din_data = 0; din_is_weight = 0;
    dout_ready = 0; routing_ctrl = '0; reduce_mode = 0; num_elements = 0;
    repeat (4) @(posedge clk);
    rst_n = 1;
    repeat (2) @(posedge clk);

`ifdef DUMP_VCD
    $dumpfile(`DUMP_VCD);
    $dumpvars(0, dut);
`ifdef DUMP_ARRAYS
      // B1c: VCD omits unpacked arrays; list their elements (generated file)
      `include `DUMP_ARRAYS
`endif
`endif
    run(32, 0, '0,      0, "FULL,    identity");
`ifdef DUMP_VCD
    $dumpoff;
    $display("  [PASS] activity captured for one FULL-mode run (%0d checks)", checks);
    $finish;
`endif
    run(32, 1, '0,      0, "PARTIAL, identity");
    run(32, 2, '0,      0, "BYPASS,  identity");
    run(16, 0, '0,      0, "FULL,    identity, one chunk");
    run( 5, 2, '0,      0, "BYPASS,  identity, partial chunk");
    run(21, 1, '0,      0, "PARTIAL, identity, partial chunk");
    run(32, 0, ROUTE_A, 0, "FULL,    routed A");
    run(21, 1, ROUTE_B, 0, "PARTIAL, routed B, partial chunk");
    run(16, 2, ROUTE_A, 0, "BYPASS,  routed A");
    run(32, 2, ROUTE_B, 1, "BYPASS,  routed B, backpressure");
    run(64, 0, '0,      0, "FULL,    K=64 clamps to K_MAX");

    $display("------------------------------------------------------------");
    if (errors == 0)
      $display("  [PASS] all %0d checks across %0d runs: 256-wide, all 3 FAN modes, routed and masked", checks, runs);
    else
      $display("  [FAIL] %0d/%0d checks failed", errors, checks);
    $display("============================================================");
    $finish;
  end

  initial begin
    #20_000_000;
    $display("  [FAIL] timeout");
    $finish;
  end

endmodule
