// =============================================================================
// tb_sigma.sv -- directed smoke test for the SIGMA baseline
//
// A hand-checkable case, alongside tb_sigma_golden's random ones:
// W = all ones, x = all twos, K = 16 (one chunk), FULL reduction. Every
// Flex-DPE's dot product is 16 x (1 x 2) = 32, so the drain must produce
// exactly NR beats of 32 and then return to idle.
//
// The previous version of this bench asserted its two middle checks as
// constant 1'b1 -- "entered compute" and "drain completed" passed whatever the
// design did. Every check here observes the design.
// =============================================================================
`timescale 1ns / 1ps

module tb_sigma;
  import versaaccel_pkg::*;

  logic clk = 0, rst_n;
  always #5 clk = ~clk;

  logic        start;
  logic [15:0] num_elements;
  logic [BENES_CTRL_W-1:0] routing_ctrl;
  logic [1:0]  reduce_mode;
  logic        din_valid;
  logic signed [DATA_W-1:0] din_data;
  logic        din_is_weight, din_ready;
  logic        dout_valid;
  logic signed [ACC_W-1:0] dout_data;
  logic        dout_ready;
  logic        busy;

  sigma_top dut (
    .clk(clk), .rst_n(rst_n), .start(start),
    .num_elements(num_elements), .routing_ctrl(routing_ctrl),
    .reduce_mode(reduce_mode),
    .din_valid(din_valid), .din_data(din_data), .din_is_weight(din_is_weight),
    .din_ready(din_ready),
    .dout_valid(dout_valid), .dout_data(dout_data), .dout_ready(dout_ready),
    .busy(busy)
  );

  int pass_cnt = 0, fail_cnt = 0, test_num = 0;
  int beats, wrong;
  bit saw_compute;

  task automatic check(string name, bit cond);
    test_num++;
    if (cond) begin $display("[PASS] Test %0d: %s", test_num, name); pass_cnt++; end
    else      begin $display("[FAIL] Test %0d: %s", test_num, name); fail_cnt++; end
  endtask

  initial begin
    $display("============================================================");
    $display("  SIGMA Baseline Testbench (directed)");
    $display("============================================================");
    rst_n = 0; start = 0; din_valid = 0; din_data = 0; din_is_weight = 0;
    dout_ready = 0; routing_ctrl = '0; reduce_mode = 2'd0; num_elements = 16'd16;
    repeat (5) @(posedge clk);
    rst_n = 1;
    repeat (2) @(posedge clk);

    @(negedge clk); start = 1;
    @(negedge clk); start = 0;
    check("start raises busy", busy);
    check("start enters weight load (din_ready high)", din_ready);

    // weights: NR units x 16, all ones -- one beat per cycle while ready
    din_valid = 1; din_is_weight = 1; din_data = 16'sd1;
    repeat (NR * 16) begin
      @(negedge clk);
    end
    // activations: 16, all twos
    din_is_weight = 0; din_data = 16'sd2;
    repeat (16) @(negedge clk);
    din_valid = 0;

    // Sample BEFORE the first wait: with K=16 there is one chunk, so COMPUTE
    // lasts exactly one cycle and has begun by the time the last activation
    // is accepted -- waiting a negedge first would step straight past it.
    saw_compute = (dut.state == 3'd3);
    while (!dout_valid) begin
      @(negedge clk);
      if (dut.state == 3'd3) saw_compute = 1;
    end
    check("passed through compute", saw_compute);

    beats = 0; wrong = 0; dout_ready = 1;
    while (dout_valid) begin
      beats++;
      if (dout_data !== 32) wrong++;
      @(negedge clk);
    end
    dout_ready = 0;
    check($sformatf("drain produced %0d beats, want %0d", beats, NR), beats == NR);
    check($sformatf("every dot product = 16 x 1 x 2 = 32 (%0d wrong)", wrong), wrong == 0);
    repeat (2) @(negedge clk);
    check("returned to idle", !busy);

    $display("============================================================");
    $display("  Results: %0d PASS, %0d FAIL out of %0d tests", pass_cnt, fail_cnt, test_num);
    $display("============================================================");
    $finish;
  end

  initial begin #2_000_000; $display("[FAIL] timeout"); $finish; end
endmodule
