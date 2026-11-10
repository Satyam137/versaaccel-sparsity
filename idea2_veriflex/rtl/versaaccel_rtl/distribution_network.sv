// =============================================================================
// distribution_network.sv — 8-port Benes network for data routing
//
// 5-stage Benes network (2*log2(8)-1 = 5) with 4 switches per stage.
// Routes 8 inputs to 8 outputs in any permutation.
// Control: 20 bits (5 stages × 4 switches × 1 bit each)
//
// In DMR mode:    straight-through (broadcast)
// In D-M-R mode:  programmed by controller for sparse routing
// In D-M-R+NM:    routes loaded from NM routing ROM
// =============================================================================
module distribution_network
  import versaaccel_pkg::*;
#(
  parameter int N     = BENES_N,      // 8
  parameter int WIDTH = DATA_W        // 16-bit data
)(
  input  logic [BENES_CTRL_W-1:0]  ctrl,       // 20-bit switch control
  input  logic signed [WIDTH-1:0]  data_in [N],
  output logic signed [WIDTH-1:0]  data_out [N]
);

  // Internal wires between stages
  logic signed [WIDTH-1:0] stage_wire [BENES_STAGES+1][N];

  // Input assignment
  genvar i;
  generate
    for (i = 0; i < N; i++) begin : gen_input
      assign stage_wire[0][i] = data_in[i];
    end
  endgenerate

  // Output assignment
  generate
    for (i = 0; i < N; i++) begin : gen_output
      assign data_out[i] = stage_wire[BENES_STAGES][i];
    end
  endgenerate

  // ---------- Benes stages, generated for any power-of-two N ----------
  //
  // This was previously hand-unrolled for exactly five stages, i.e. N=8. At
  // N=16 the network needs 2*log2(16)-1 = 7 stages, so stage_wire[6] and [7]
  // were never driven and every output read as X/0 -- the design did not
  // actually scale despite being written to look parameterised.
  //
  // Standard Benes construction: the partner distance halves across the first
  // log2(N) stages and then mirrors back out.
  //   stage s <  L : d = N >> (s+1)          (L-1, ..., 1)
  //   stage s >= L : d = 1 << (s - L + 1)
  // Switch k of a stage pairs ports (j, j+d) with j = (k/d)*2d + (k%d), which
  // enumerates each port exactly once. For N=8 this reproduces the previous
  // hand-written pairing exactly (d = 4,2,1,2,4).

  localparam int L = $clog2(N);

  genvar st, k;
  generate
    for (st = 0; st < BENES_STAGES; st++) begin : gen_stage
      localparam int D = (st < L) ? (N >> (st + 1)) : (1 << (st - L + 1));
      for (k = 0; k < N/2; k++) begin : gen_sw
        localparam int J = ((k / D) * 2 * D) + (k % D);
        benes_switch #(.WIDTH(WIDTH)) sw (
          .ctrl (ctrl[st * (N/2) + k]),
          .in0  (stage_wire[st][J]),
          .in1  (stage_wire[st][J + D]),
          .out0 (stage_wire[st+1][J]),
          .out1 (stage_wire[st+1][J + D])
        );
      end
    end
  endgenerate

endmodule
