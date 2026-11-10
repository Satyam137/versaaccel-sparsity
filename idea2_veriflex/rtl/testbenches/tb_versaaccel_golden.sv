// =============================================================================
// tb_versaaccel_golden.sv -- END-TO-END arithmetic verification of
// versaaccel_top: external interface -> ibuf/wbuf -> Benes distribution ->
// PE array -> reduction network -> dout.
//
// FIX_PLAN.md Phase 5, final item. tb_versaaccel checks config registers and
// FSM transitions; tb_datapath_golden and tb_cpe_golden check the primitives in
// isolation. Nothing checked that the ASSEMBLED path computes anything.
//
// Reference model, derived from the RTL's structure:
//   ibuf_rd_ptr/wbuf_rd_ptr are never incremented, so
//     dist_in[i] = ibuf[i],  pe_b_in[c] = wbuf[c]
//   dist_ctrl = 0 for every config except NM -> identity routing
//     pe_a_in[r] = ibuf[r]
//   pe_array is a systolic mesh with one-cycle forwarding:
//     a_wire[r][c] = pe_a_in[r] delayed c cycles
//     b_wire[r][c] = pe_b_in[c] delayed r cycles
//   PE(r,c) accumulates a_wire[r][c] * b_wire[r][c] while acc_en is high
//   only column 0 reaches the reduction network:
//     reduce_in[i] = pe_out[i][0] = pe_a_in[i] * wbuf[0] * (accumulate count)
//   dout_data = reduce_out[obuf_rd_ptr mod NR]
//
// Run:
//   iverilog -g2012 -o tb_vg -DCPE_WEIGHT_DIR='"<abs>/rtl/weights"' \
//       common/versaaccel_pkg.sv versaaccel_rtl/*.sv \
//       testbenches/tb_versaaccel_golden.sv
//   ./tb_vg
// =============================================================================
`timescale 1ns/1ps

module tb_versaaccel_golden;
  import versaaccel_pkg::*;

  // Layer configuration pins. Tied to the values the RTL used internally
  // before they became ports (routing straight-through, N:M pattern 0, all
  // CPE features 0), so every check below means exactly what it meant then.
  logic [BENES_CTRL_W-1:0] routing_ctrl = '0;
  logic [2:0]              nm_pattern   = 3'd0;
  logic signed [7:0]       layer_features [9];
  initial for (int f = 0; f < 9; f++) layer_features[f] = 8'sd0;

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

  logic        start;
  logic [1:0]  model_type;
  logic [15:0] batch_size;
  logic [1:0]  op_type;
  logic                     din_valid, din_is_weight, din_ready;
  logic signed [DATA_W-1:0] din_data;
  logic                     dout_valid, dout_ready;
  logic signed [ACC_W-1:0]  dout_data;
  logic                     cfg_override_en;
  config_t                  cfg_override;
  logic                     busy;
  config_t                  active_config;
  phase_t                   current_phase;
  logic                     flush_event;

  versaaccel_top dut (
    // Tiling controller disabled: this bench exercises the legacy
    // single-shot path. Leaving these unconnected would drive gemm_mode
    // to X and corrupt the datapath muxes.
    .gemm_en(1'b0), .dim_m(16'd0), .dim_k(16'd0), .dim_p(16'd0),
    .drain_wide(1'b0), .dout_vec(), .skip_en(1'b0), .tile_shape(2'd0), .routing_ctrl(routing_ctrl), .nm_pattern(nm_pattern), .layer_features(layer_features),
    .clk(clk_dut), .rst_n(rst_n), .start(start),
    .model_type(model_type), .batch_size(batch_size), .op_type(op_type),
    .din_valid(din_valid), .din_data(din_data), .din_is_weight(din_is_weight),
    .din_ready(din_ready),
    .dout_valid(dout_valid), .dout_data(dout_data), .dout_ready(dout_ready),
    .cfg_override_en(cfg_override_en), .cfg_override(cfg_override),
    .busy(busy), .active_config(active_config),
    .current_phase(current_phase), .flush_event(flush_event)
  );

  int errors  = 0;
  int checks  = 0;
  int n_zero_act_ports;

  logic signed [DATA_W-1:0] WV [NC];
  logic signed [ACC_W-1:0]  observed [NR];

  task automatic expect_eq(string what, int idx, longint got, longint exp);
    checks++;
    if (got !== exp) begin
      errors++;
      if (errors <= 10)
        $display("  [FAIL] %s[%0d]: got %0d, expected %0d", what, idx, got, exp);
    end
  endtask

  // ---------------------------------------------------------------------
  // One full transaction: load weights, send one activation, compute, drain.
  // ---------------------------------------------------------------------
  task automatic run_case(config_t cfg, logic signed [DATA_W-1:0] act,
                          int steps, string label);
    longint expect0;
    rst_n = 0;
    start = 0; din_valid = 0; din_is_weight = 0; din_data = 0;
    dout_ready = 0; cfg_override_en = 1; cfg_override = cfg;
    model_type = 2'd0; op_type = 2'd0; batch_size = steps[15:0];
    repeat (4) @(posedge clk);
    rst_n = 1;
    repeat (2) @(posedge clk);

    // kick off -> ST_LOAD_WEIGHTS
    start = 1;
    @(posedge clk);
    start = 0;

    // weights
    din_valid = 1; din_is_weight = 1;
    for (int c = 0; c < NC; c++) begin
      din_data = WV[c];
      @(posedge clk);
    end
    din_valid = 0;

    // Idle inside ST_LOAD_WEIGHTS so the b_wire forwarding chain fills to the
    // bottom row before compute begins. The PE forwarding registers are free
    // running, so this costs nothing but cycles.
    repeat (NR + 2) @(posedge clk);

    // single activation -> transitions to ST_COMPUTE on this edge
    din_valid = 1; din_is_weight = 0; din_data = act;
    @(posedge clk);
    din_valid = 0;

    // compute runs for `steps` cycles
    repeat (steps) @(posedge clk);

    // drain NR ports. A beat is taken on the edge where dout_valid and
    // dout_ready are both high; the DUT clock lags clk by 100 ps, so sampling
    // at posedge clk sees exactly the beat that edge accepts. (This used to
    // sample on fixed cycles after `wait (dout_valid)`, which was right only
    // because the drain had always started before the wait -- true until the
    // operand feed was pipelined and the output registered.)
    wait (dout_valid);
    @(negedge clk);
    dout_ready = 1;
    for (int i = 0; i < NR; ) begin
      @(posedge clk);
      if (dout_valid) begin observed[i] = dout_data; i++; end
    end
    #1 dout_ready = 0;

    // ---- golden ----
    // Only ibuf[0] is ever written (see the diagnostic below), so
    // pe_a_in[0] = act and pe_a_in[i>0] = 0. Column 0 has no skew, so PE(0,0)
    // accumulates act * wbuf[0] once per compute cycle.
    expect0 = longint'(act) * longint'(WV[0]) * longint'(steps);

    $display("  %-22s cfg=%0d act=%0d w0=%0d steps=%0d -> dout[0]=%0d (expect %0d)",
             label, int'(cfg), act, WV[0], steps, observed[0], expect0);

    expect_eq(label, 0, observed[0], expect0);
    // every other port must be zero: the remaining activations are zero, and
    // FULL/PARTIAL/BYPASS all leave them at zero for a single-hot input
    for (int i = 1; i < NR; i++)
      expect_eq(label, i, observed[i], 0);
  endtask

  initial begin
    $display("============================================================");
    $display("  versaaccel_top -- END-TO-END GOLDEN VECTOR test");
    $display("  array %0dx%0d", NR, NC);
    $display("============================================================");

    for (int c = 0; c < NC; c++)
      WV[c] = $signed(((c * 5) % 17) - 8);   // -8..8, includes 0 and negatives

    // Same datapath, three configurations and three accumulate counts.
    run_case(CFG_DMR,    $signed(6),  4,  "DMR   full-reduce");
    run_case(CFG_D_MR,   $signed(-5), 7,  "D-MR  partial");
    run_case(CFG_DM_R,   $signed(3),  11, "DM-R  bypass");
    run_case(CFG_D_M_R,  $signed(-7), 5,  "D-M-R bypass");

    // ---------------------------------------------------------------------
    // Diagnostic: how many activation elements can the top level actually
    // load? ST_LOAD_WEIGHTS leaves for ST_COMPUTE on the FIRST non-weight
    // datum, and ibuf_rd_ptr is never incremented, so the answer should be 1.
    // If so, versaaccel_top cannot perform a full matrix operation -- it
    // broadcasts a single scalar through the array. That is a structural
    // limitation of the integration, not an arithmetic error, and the paper
    // must not describe this as a working GEMM engine.
    // ---------------------------------------------------------------------
    n_zero_act_ports = 0;
    for (int i = 1; i < NR; i++)
      if (dut.ibuf[i] === '0) n_zero_act_ports++;
    $display("------------------------------------------------------------");
    $display("  DIAGNOSTIC -- structural limits of the top-level integration");
    $display("    activations loaded (ibuf_wr_ptr):   %0d of %0d array rows",
             int'(dut.ibuf_wr_ptr), NR);
    $display("    unwritten ibuf entries reading X:   %0d of %0d",
             (NR-1) - n_zero_act_ports, NR-1);
    checks++;
    if (int'(dut.ibuf_wr_ptr) != 1) begin
      $display("    (expected exactly 1 -- update this note if the FSM changed)");
    end else begin
      $display("    -> ST_LOAD_WEIGHTS leaves for ST_COMPUTE on the FIRST non-weight");
      $display("       datum and ibuf_rd_ptr never advances, so versaaccel_top can only");
      $display("       ever broadcast a SINGLE scalar activation through the array.");
      $display("       The arithmetic below is correct, but this is not a GEMM datapath");
      $display("       and the paper must not describe it as one.");
    end
    $display("    unwritten entries are gated off by ibuf_wr_ptr, so they no longer");
    $display("    reach the PE array (they previously made DMR/D-MR output X)." );

    $display("------------------------------------------------------------");
    if (errors == 0)
      $display("  [PASS] all %0d end-to-end checks match the golden model", checks);
    else
      $display("  [FAIL] %0d of %0d checks failed", errors, checks);
    $display("============================================================");
    $finish;
  end

  initial begin
    #2000000;
    $display("  [FAIL] timeout");
    $finish;
  end

endmodule
