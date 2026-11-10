// =============================================================================
// tb_layer_config.sv -- the layer-configuration pins are live
//
// routing_ctrl, nm_pattern and layer_features used to be tied off inside
// versaaccel_top. Synthesis constant-folded the Benes network, the N:M ROM
// and the whole CPE MLP away, and the first Genus run reported zero area for
// all three.
//
// Promoting them to pins only fixes that if the pins actually reach the
// logic. A pin that is captured but never used -- or used only in a branch
// that is itself constant -- is folded away exactly as before, and the rest
// of the suite cannot tell: every other bench ties these pins to the old
// constants on purpose, so they check that nothing regressed, not that
// anything new is connected.
//
// This bench checks the connection, end to end:
//   1. capture   pins are registered on `start`, and held for the layer
//   2. D-MR      routing_ctrl reaches dist_ctrl
//   3. D-M-R     routing_ctrl reaches dist_ctrl
//   4. DMR/DM-R  systolic-distribution configs stay straight-through
//   5. N:M       nm_pattern selects the ROM entry that drives dist_ctrl
//   6. GEMM      the tiling controller forces straight-through regardless
//   7. CPE       different layer_features give a different CPE output --
//                i.e. the MLP is a function of the pins, not a constant
//
// That the Benes then permutes correctly for a given dist_ctrl is already
// checked by tb_datapath_golden; this bench closes the pin -> dist_ctrl gap.
// =============================================================================
`timescale 1ns / 1ps

module tb_layer_config;
  import versaaccel_pkg::*;

  logic clk = 0, rst_n = 0, start = 0;
  always #5 clk = ~clk;

  logic                    gemm_en = 0;
  logic [15:0]             dim_m = 0, dim_k = 0, dim_p = 0;
  logic                    cfg_override_en = 1;
  config_t                 cfg_override = CFG_DMR;
  logic [BENES_CTRL_W-1:0] routing_ctrl = '0;
  logic [2:0]              nm_pattern   = 3'd0;
  logic signed [7:0]       layer_features [9];

  logic                    din_ready, dout_valid, busy, flush_event;
  logic signed [ACC_W-1:0] dout_data;
  config_t                 active_config;
  phase_t                  current_phase;

  versaaccel_top dut (
    .clk(clk), .rst_n(rst_n), .start(start),
    .model_type(2'd2), .batch_size(16'd1), .op_type(2'd0),
    .gemm_en(gemm_en), .dim_m(dim_m), .dim_k(dim_k), .dim_p(dim_p),
    .drain_wide(1'b0), .skip_en(1'b0), .tile_shape(2'd0),
    .routing_ctrl(routing_ctrl), .nm_pattern(nm_pattern),
    .layer_features(layer_features),
    .din_valid(1'b0), .din_data('0), .din_is_weight(1'b0), .din_ready(din_ready),
    .dout_valid(dout_valid), .dout_data(dout_data), .dout_vec(), .dout_ready(1'b1),
    .cfg_override_en(cfg_override_en), .cfg_override(cfg_override),
    .busy(busy), .active_config(active_config),
    .current_phase(current_phase), .flush_event(flush_event)
  );

  int errors = 0, checks = 0;

  task automatic check(input bit ok, input string what);
    checks++;
    if (!ok) begin errors++; $display("  [FAIL] %s", what); end
  endtask

  // Every scenario starts from reset in IDLE, so each layer's registers are
  // captured fresh -- the legacy path would otherwise need a full load /
  // compute / drain before returning to IDLE.
  task automatic begin_layer();
    rst_n = 0; start = 0;
    repeat (2) @(posedge clk);
    #1 rst_n = 1;
    @(posedge clk); #1 start = 1;
    @(posedge clk); #1 start = 0;
  endtask

  task automatic set_features(input int seed);
    for (int f = 0; f < 9; f++)
      layer_features[f] = (seed == 0) ? 8'sd0 : 8'(signed'((seed * (f + 3) * 37) % 127));
  endtask

  // Any BENES_CTRL_W-bit word is a legal routing (NR=16: 7 stages x 8
  // switches = 56 bits). Use full-width words with bits set in every stage,
  // so a truncation anywhere between pin and dist_ctrl would show.
  localparam logic [BENES_CTRL_W-1:0] ROUTE_A = BENES_CTRL_W'(64'hA5C3_96F0_1E2D_4B78);
  localparam logic [BENES_CTRL_W-1:0] ROUTE_B = BENES_CTRL_W'(64'h5A3C_690F_E1D2_B487);

  // Reference copy of the N:M ROM -- the bench must not read the design's own
  // ROM to decide what the design should have produced. The ROM's entries are
  // 20-bit literals from the 8-port design, zero-extended into the 56-bit
  // control word; this copy reproduces that so the check is exact.
  function automatic logic [BENES_CTRL_W-1:0] nm_ref(input logic [2:0] i);
    case (i)
      3'd0: return 20'b0000_0000_0000_0000_0000;
      3'd1: return 20'b0000_0000_0100_0000_0000;
      3'd2: return 20'b0001_0000_0100_0000_0001;
      3'd3: return 20'b0000_0001_0100_0001_0000;
      3'd4: return 20'b0001_0001_0100_0001_0001;
      3'd5: return 20'b0001_0001_0000_0001_0001;
      default: return '0;
    endcase
  endfunction

  config_t     cfg0, cfg_v;
  logic [7:0]  conf0, conf_v;
  int          n_differ;

  initial begin
    $display("=== tb_layer_config: layer-configuration pins reach the logic ===");
    set_features(0);

    // ---- 1. capture and hold ------------------------------------------------
    cfg_override = CFG_D_M_R; routing_ctrl = ROUTE_A; nm_pattern = 3'd3;
    set_features(5);
    begin_layer();
    check(dut.routing_ctrl_q == ROUTE_A, "routing_ctrl captured on start");
    check(dut.nm_pattern_idx == 3'd3,    "nm_pattern captured on start");
    for (int f = 0; f < 9; f++)
      check(dut.cpe_features[f] == layer_features[f], $sformatf("layer_features[%0d] captured", f));
    // change every pin mid-layer: the registers must not follow
    routing_ctrl = ROUTE_B; nm_pattern = 3'd5; set_features(9);
    repeat (3) @(posedge clk); #1;
    check(dut.routing_ctrl_q == ROUTE_A, "routing_ctrl held for the layer");
    check(dut.nm_pattern_idx == 3'd3,    "nm_pattern held for the layer");

    // ---- 2-4. per-configuration routing --------------------------------------
    begin_layer_with(CFG_D_MR,  ROUTE_A, 3'd0, ROUTE_A, "D-MR takes routing_ctrl");
    begin_layer_with(CFG_D_M_R, ROUTE_B, 3'd0, ROUTE_B, "D-M-R takes routing_ctrl");
    begin_layer_with(CFG_DMR,   ROUTE_A, 3'd0, '0,      "DMR stays straight-through");
    begin_layer_with(CFG_DM_R,  ROUTE_A, 3'd0, '0,      "DM-R stays straight-through");

    // ---- 5. N:M: every ROM entry reachable from the pin ----------------------
    for (int k = 0; k < 6; k++)
      begin_layer_with(CFG_D_M_R_NM, ROUTE_A, 3'(k), nm_ref(3'(k)),
                       $sformatf("N:M pattern %0d drives dist_ctrl", k));

    // ---- 6. GEMM forces straight-through -------------------------------------
    gemm_en = 1; dim_m = 16; dim_k = 16; dim_p = 16;
    begin_layer_with(CFG_D_M_R, ROUTE_A, 3'd0, '0, "GEMM mode forces identity routing");
    gemm_en = 0; dim_m = 0; dim_k = 0; dim_p = 0;

    // ---- 7. CPE is a function of layer_features -------------------------------
    // If the features were still tied off, every vector would reproduce the
    // zero-vector output. Requiring at least one to differ is exactly the
    // property that stops synthesis treating the MLP as a constant.
    cfg_override = CFG_DMR;
    set_features(0); begin_layer(); @(posedge dut.cpe_valid); #1;
    cfg0 = dut.cpe_config; conf0 = dut.cpe_confidence;
    n_differ = 0;
    for (int v = 1; v <= 6; v++) begin
      set_features(v); begin_layer(); @(posedge dut.cpe_valid); #1;
      cfg_v = dut.cpe_config; conf_v = dut.cpe_confidence;
      if (cfg_v !== cfg0 || conf_v !== conf0) n_differ++;
      check(!$isunknown(cfg_v) && !$isunknown(conf_v),
            $sformatf("CPE output defined for feature vector %0d", v));
    end
    check(n_differ > 0, "CPE output depends on layer_features (not constant)");
    $display("  CPE: %0d of 6 feature vectors change the output vs all-zero", n_differ);

    $display("------------------------------------------------------------");
    if (errors == 0)
      $display("  [PASS] all %0d layer-config checks: pins reach Benes, N:M ROM and CPE", checks);
    else
      $display("  [FAIL] %0d of %0d layer-config checks failed", errors, checks);
    $finish;
  end

  // One layer with the given config and pins, then check dist_ctrl.
  task automatic begin_layer_with(input config_t cfg,
                                  input logic [BENES_CTRL_W-1:0] rc,
                                  input logic [2:0] nm,
                                  input logic [BENES_CTRL_W-1:0] want,
                                  input string what);
    cfg_override = cfg; routing_ctrl = rc; nm_pattern = nm;
    begin_layer();
    #1;
    check(dut.dist_ctrl === want,
          $sformatf("%s (dist_ctrl=%b want %b)", what, dut.dist_ctrl, want));
  endtask

  initial begin #2_000_000; $display("  [FAIL] timeout"); $finish; end
endmodule
