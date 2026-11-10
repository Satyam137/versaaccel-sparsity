// =============================================================================
// tb_datapath_golden.sv -- ARITHMETIC verification of VersaAccel's datapath
// primitives: the Benes distribution network and the reduction adder tree.
//
// FIX_PLAN.md Phase 5. tb_versaaccel checks config registers and FSM
// transitions; it never checks that data is routed to the right port or that
// the adder tree sums the right operands. These two modules are the arithmetic
// heart of every configuration, and they are pure combinational functions, so
// they can be checked directly and thoroughly rather than through the top-level
// FSM.
//
// Checks:
//   distribution_network  identity routing (ctrl=0) is order-preserving, and
//                         every input value reaches exactly one output
//                         (permutation property) for several control words
//   reduction_network     FULL sums ALL N inputs, PARTIAL emits pairwise sums,
//                         BYPASS is the identity -- each against an
//                         independently computed golden value
//
// Run:
//   iverilog -g2012 -o tb_dp common/versaaccel_pkg.sv \
//       versaaccel_rtl/benes_switch.sv versaaccel_rtl/distribution_network.sv \
//       versaaccel_rtl/reduction_network.sv testbenches/tb_datapath_golden.sv
//   ./tb_dp
// =============================================================================
`timescale 1ns/1ps

module tb_datapath_golden;
  import versaaccel_pkg::*;

  localparam int N = BENES_N;

  // ---- distribution network ----
  logic [BENES_CTRL_W-1:0]  dctrl;
  logic signed [DATA_W-1:0] din  [N];
  logic signed [DATA_W-1:0] dout [N];

  distribution_network dnet (.ctrl(dctrl), .data_in(din), .data_out(dout));

  // ---- reduction network ----
  logic [1:0]              rmode;
  logic signed [ACC_W-1:0] rin  [N];
  logic signed [ACC_W-1:0] rout [N];

  reduction_network rnet (.mode(rmode), .data_in(rin), .data_out(rout));

  int errors = 0;
  int checks = 0;

  task automatic expect_eq(string what, int idx, longint got, longint exp);
    checks++;
    if (got !== exp) begin
      errors++;
      if (errors <= 12)
        $display("  [FAIL] %s[%0d]: got %0d, expected %0d", what, idx, got, exp);
    end
  endtask

  // golden: is `dout` a permutation of `din`?
  function automatic bit is_permutation();
    int seen [N];
    int found;
    for (int i = 0; i < N; i++) seen[i] = 0;
    for (int o = 0; o < N; o++) begin
      found = -1;
      for (int i = 0; i < N; i++)
        if (!seen[i] && dout[o] === din[i]) begin
          found = i;
          break;
        end
      if (found < 0) return 1'b0;
      seen[found] = 1;
    end
    return 1'b1;
  endfunction

  longint gold_sum;
  int nonzero_outs;
  integer seed;

  initial begin
    $display("============================================================");
    $display("  Datapath primitives -- GOLDEN VECTOR test  (N = %0d)", N);
    $display("============================================================");

    // ---------------------------------------------------------------------
    // 1. Distribution network: identity control must preserve order
    // ---------------------------------------------------------------------
    for (int i = 0; i < N; i++) din[i] = $signed(i * 7 - 40);  // distinct, signed
    dctrl = '0;
    #1;
    for (int i = 0; i < N; i++)
      expect_eq("dist.identity", i, dout[i], din[i]);
    $display("  identity routing (ctrl=0): %0d ports checked", N);

    // ---------------------------------------------------------------------
    // 2. Distribution network: any control word must still be a permutation.
    //    A Benes network can only rearrange data, never duplicate or drop it.
    // ---------------------------------------------------------------------
    begin
      int perm_fail = 0;
      for (int t = 0; t < 8; t++) begin
        seed  = t + 1;
        dctrl = $random(seed);
        #1;
        if (!is_permutation()) begin
          perm_fail++;
          if (perm_fail <= 3)
            $display("  [FAIL] ctrl=%h does not produce a permutation", dctrl);
        end
        checks++;
      end
      errors += perm_fail;
      $display("  permutation property over 8 random control words: %0d failure(s)",
               perm_fail);
    end

    // ---------------------------------------------------------------------
    // 3. Reduction network -- BYPASS
    // ---------------------------------------------------------------------
    for (int i = 0; i < N; i++) rin[i] = $signed(i * 3 - 17);
    rmode = 2'd2;
    #1;
    for (int i = 0; i < N; i++)
      expect_eq("reduce.bypass", i, rout[i], rin[i]);

    // ---------------------------------------------------------------------
    // 4. Reduction network -- PARTIAL: adjacent pair sums
    // ---------------------------------------------------------------------
    rmode = 2'd1;
    #1;
    for (int i = 0; i < N / 2; i++)
      expect_eq("reduce.partial", i, rout[i], rin[2 * i] + rin[2 * i + 1]);

    // ---------------------------------------------------------------------
    // 5. Reduction network -- FULL: must sum ALL N inputs.
    //    This is where a tree hardcoded for N=8 shows up at larger N.
    // ---------------------------------------------------------------------
    gold_sum = 0;
    for (int i = 0; i < N; i++) gold_sum += rin[i];
    rmode = 2'd0;
    #1;
    expect_eq("reduce.full", 0, rout[0], gold_sum);
    $display("  FULL reduction: got %0d, expected sum of all %0d inputs = %0d",
             rout[0], N, gold_sum);

    // FULL emits a single value; every other port must be zero
    nonzero_outs = 0;
    for (int i = 1; i < N; i++) if (rout[i] !== 0) nonzero_outs++;
    checks++;
    if (nonzero_outs != 0) begin
      errors++;
      $display("  [FAIL] FULL mode left %0d non-zero ports besides port 0",
               nonzero_outs);
    end

    $display("------------------------------------------------------------");
    if (errors == 0)
      $display("  [PASS] all %0d datapath checks passed", checks);
    else
      $display("  [FAIL] %0d of %0d checks failed", errors, checks);
    $display("============================================================");
    $finish;
  end

endmodule
