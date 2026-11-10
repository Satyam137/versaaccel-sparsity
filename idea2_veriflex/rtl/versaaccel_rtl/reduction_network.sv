// =============================================================================
// reduction_network.sv — Configurable adder tree for partial sum reduction
//
// 3-stage reduction tree for 8 inputs:
//   Stage 1: 8→4 (4 adders)
//   Stage 2: 4→2 (2 adders)
//   Stage 3: 2→1 (1 adder)
//
// Modes:
//   FULL:    All 8 inputs reduced to 1 output (result[0])
//   PARTIAL: 4 pairs reduced to 4 outputs
//   BYPASS:  8 inputs pass through to 8 outputs (no reduction)
// =============================================================================
module reduction_network
  import versaaccel_pkg::*;
#(
  parameter int N     = NR,
  parameter int WIDTH = ACC_W
)(
  input  logic [1:0]          mode,  // 0=FULL, 1=PARTIAL, 2=BYPASS
  input  logic signed [WIDTH-1:0]  data_in [N],
  output logic signed [WIDTH-1:0]  data_out [N]
);

  // ---------- Adder tree, generated for any power-of-two N ----------
  //
  // The tree was previously hand-unrolled to three stages with a final
  // `s3 = s2[0] + s2[1]`, which is only correct for N=8. At N=16 the second
  // stage produces four partial sums and that final line summed just two of
  // them, so FULL reduction silently dropped half its inputs.

  localparam int LEVELS = $clog2(N);

  // tree[0] = inputs; tree[l][i] = tree[l-1][2i] + tree[l-1][2i+1]
  logic signed [WIDTH-1:0] tree [LEVELS+1][N];

  genvar l, i;
  generate
    for (i = 0; i < N; i++) begin : gen_leaf
      assign tree[0][i] = data_in[i];
    end
    for (l = 1; l <= LEVELS; l++) begin : gen_level
      for (i = 0; i < (N >> l); i++) begin : gen_node
        assign tree[l][i] = tree[l-1][2*i] + tree[l-1][2*i + 1];
      end
    end
  endgenerate

  // ---------- Output mux ----------
  always_comb begin
    for (int j = 0; j < N; j++) data_out[j] = '0;

    case (mode)
      2'd0: begin // FULL: sum of ALL N inputs
        data_out[0] = tree[LEVELS][0];
      end
      2'd1: begin // PARTIAL: adjacent pair sums
        for (int j = 0; j < N/2; j++)
          data_out[j] = tree[1][j];
      end
      default: begin // BYPASS
        for (int j = 0; j < N; j++)
          data_out[j] = data_in[j];
      end
    endcase
  end

endmodule
