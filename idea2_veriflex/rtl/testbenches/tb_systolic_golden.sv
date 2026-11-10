// =============================================================================
// tb_systolic_golden.sv -- ARITHMETIC verification for the systolic baseline.
//
// FIX_PLAN.md, Phase 5 item 2. The existing testbenches (tb_systolic, tb_sigma,
// tb_versaaccel) check only control handshakes: that `busy` rises, that the FSM
// reaches S_COMPUTE, that drain completes. None of them ever checks that the
// array COMPUTES THE RIGHT NUMBERS. A design that multiplied by zero would pass
// all of them.
//
// This bench maintains an independent behavioural model of the weight-stationary
// dataflow -- derived from the specification, not copied from the DUT's
// implementation -- and compares every one of the ROWS*COLS accumulators against
// it after drain.
//
// Scope of this check, stated honestly: the model's activation ramp-in timing
// (the cycle at which act_buf[r] becomes visible) was calibrated against the
// DUT. What the test then verifies is that ALL ROWS*COLS accumulators agree
// with ONE consistent timing rule -- including negative operands and zeros.
// A design with a wrong multiply, a dropped sign, an inactive PE, a broken
// column skew, or mis-addressed drain cannot satisfy that simultaneously.
// It does NOT independently pin the pipeline-fill latency; that would need a
// spec document or a second implementation.
//
// Reference dataflow being checked:
//   a_fwd[r][0]   = act_buf[r]                     (combinational)
//   a_fwd[r][c+1] = a_fwd[r][c]  delayed 1 cycle   (column skew)
//   accum[r][c]   = a_fwd[r][c] * W[r][c]          at step 0
//                 = accum + a_fwd[r][c] * W[r][c]  thereafter
//
// Run:
//   iverilog -g2012 -o tb_gold common/versaaccel_pkg.sv \
//       systolic_baseline/systolic_top.sv testbenches/tb_systolic_golden.sv
//   ./tb_gold
// =============================================================================
`timescale 1ns/1ps

module tb_systolic_golden;
  import versaaccel_pkg::*;

  localparam int ROWS  = NR;
  localparam int COLS  = NC;
`ifdef POWER_STEPS
  localparam int STEPS = `POWER_STEPS;     // B1c: longer run for activity capture
`else
  localparam int STEPS = 12;
`endif

  logic clk = 0;
  logic rst_n;
  always #5 clk = ~clk;

  logic                     start, load_weights;
  logic [15:0]              compute_steps;
  logic                     din_valid, din_is_weight, din_ready;
  logic signed [DATA_W-1:0] din_data;
  logic                     dout_valid, dout_ready;
  logic signed [ACC_W-1:0]  dout_data;
  logic                     busy;

  systolic_top dut (
    .clk(clk), .rst_n(rst_n), .start(start), .load_weights(load_weights),
    .compute_steps(compute_steps),
    .din_valid(din_valid), .din_data(din_data), .din_is_weight(din_is_weight),
    .din_ready(din_ready),
    .dout_valid(dout_valid), .dout_data(dout_data), .dout_ready(dout_ready),
    .busy(busy)
  );

  // ---------------- stimulus and golden state ----------------
  logic signed [DATA_W-1:0] W    [ROWS][COLS];   // weights we loaded
  logic signed [DATA_W-1:0] ACT  [ROWS];         // activations we drove
  logic signed [ACC_W-1:0]  gold [ROWS][COLS];   // expected accumulators
  logic signed [DATA_W-1:0] gfwd [ROWS][COLS+1]; // modelled skew pipeline

  int errors = 0;
  int checks = 0;

  task automatic check_eq(string what, int r, int c,
                          logic signed [ACC_W-1:0] got,
                          logic signed [ACC_W-1:0] exp);
    checks++;
    if (got !== exp) begin
      errors++;
      if (errors <= 10)
        $display("  [FAIL] %s [%0d][%0d]: got %0d, expected %0d", what, r, c, got, exp);
    end
  endtask

  // Independent behavioural model, advanced once per compute cycle.
  //
  // act_buf is written one row per cycle during compute (act_wr_idx increments
  // as activations arrive). Row r therefore holds its value from cycle r
  // onward, and reads as zero before that. The model must reproduce this
  // ramp-in, otherwise it expects data the array has not received yet.
  task automatic gold_step(int step);
    logic signed [DATA_W-1:0] nxt [ROWS][COLS+1];
    logic signed [DATA_W-1:0] abuf;
    for (int r = 0; r < ROWS; r++) begin
      abuf = (r <= step) ? ACT[r] : $signed('0);
      nxt[r][0] = abuf;                      // combinational input
      for (int c = 0; c < COLS; c++)
        nxt[r][c+1] = gfwd[r][c];            // 1-cycle skew
      for (int c = 0; c < COLS; c++) begin
        if (step == 0) gold[r][c] = gfwd[r][c] * W[r][c];
        else           gold[r][c] = gold[r][c] + gfwd[r][c] * W[r][c];
      end
      for (int c = 0; c <= COLS; c++) gfwd[r][c] = nxt[r][c];
    end
  endtask

  initial begin
    $display("============================================================");
    $display("  Systolic Array -- GOLDEN VECTOR (arithmetic) Testbench");
    $display("  %0dx%0d array, %0d compute steps", ROWS, COLS, STEPS);
    $display("============================================================");

    // deterministic, non-trivial stimulus: distinct signed values, some zero,
    // some negative, so a design that dropped signs or skipped PEs would fail.
    for (int r = 0; r < ROWS; r++) begin
      ACT[r] = $signed((r % 7) - 3);                  // -3..3, includes 0
      for (int c = 0; c < COLS; c++)
        W[r][c] = $signed(((r * COLS + c) % 11) - 5); // -5..5, includes 0
    end
`ifdef POWER_HEX
    // B1c: real operands for activity capture (experiments/make_activity.py):
    // W row-major ROWSxCOLS, then ACT[ROWS], 16-bit two's complement.
    begin
      logic [DATA_W-1:0] mem [ROWS*COLS + ROWS];
      $readmemh(`POWER_HEX, mem);
      for (int r = 0; r < ROWS; r++) begin
        for (int c = 0; c < COLS; c++) W[r][c] = $signed(mem[r * COLS + c]);
        ACT[r] = $signed(mem[ROWS * COLS + r]);
      end
    end
`endif
`ifdef DUMP_VCD
    $dumpfile(`DUMP_VCD);
    $dumpvars(0, dut);
`ifdef DUMP_ARRAYS
      // B1c: VCD omits unpacked arrays; list their elements (generated file)
      `include `DUMP_ARRAYS
`endif
`endif
    for (int r = 0; r < ROWS; r++)
      for (int c = 0; c <= COLS; c++) gfwd[r][c] = '0;

    rst_n = 0; start = 0; load_weights = 0;
    din_valid = 0; din_data = 0; din_is_weight = 0; dout_ready = 0;
    compute_steps = STEPS;
    repeat (5) @(posedge clk);
    rst_n = 1;
    repeat (2) @(posedge clk);

    // ---- weight load ----
    start = 1; load_weights = 1;
    @(posedge clk);
    start = 0;
    din_valid = 1; din_is_weight = 1;
    for (int r = 0; r < ROWS; r++)
      for (int c = 0; c < COLS; c++) begin
        din_data = W[r][c];
        @(posedge clk);
      end
    din_valid = 0; din_is_weight = 0;
    repeat (3) @(posedge clk);
    if (busy !== 1'b0) begin
      $display("  [FAIL] weight load did not complete (busy still high)");
      errors++;
    end
    checks++;

    // ---- compute ----
    // Drive activations on the first ROWS cycles, then hold. The golden model
    // is advanced in lockstep, one step per S_COMPUTE cycle.
    start = 1; load_weights = 0;
    @(posedge clk);          // FSM enters S_COMPUTE on this edge
    start = 0;

    for (int s = 0; s < STEPS; s++) begin
      if (s < ROWS) begin
        din_valid = 1; din_is_weight = 0; din_data = ACT[s];
      end else begin
        din_valid = 0;
      end
      gold_step(s);          // model the cycle the DUT is about to execute
      @(posedge clk);
    end
    din_valid = 0;

    // ---- drain and compare every accumulator ----
    // dout_data is combinational on {drain_row, drain_col}, and those advance
    // on the posedge when dout_ready is high. So the value for element (r,c)
    // must be sampled BEFORE the edge that moves the pointer on.
    wait (dout_valid);
    dout_ready = 1;
    for (int r = 0; r < ROWS; r++)
      for (int c = 0; c < COLS; c++) begin
        check_eq("accum", r, c, dout_data, gold[r][c]);
        @(posedge clk);
      end
    dout_ready = 0;

    $display("------------------------------------------------------------");
    if (errors == 0)
      $display("  [PASS] all %0d accumulators match the golden model", checks);
    else
      $display("  [FAIL] %0d/%0d mismatches", errors, checks);
    $display("============================================================");
    $finish;
  end

  initial begin
    #500000;
    $display("  [FAIL] timeout");
    $finish;
  end

endmodule
