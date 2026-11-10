// =============================================================================
// sigma_top.sv -- SIGMA-style flexible sparse accelerator baseline
//
// SIGMA (Qin et al., HPCA 2020) is an array of Flex-DPEs: each Flex-DPE is a
// row of multipliers fed through a Benes distribution network, whose products
// are reduced by a FAN (Forwarding Adder Network) into dot products of a
// configurable size.
//
// This baseline is sized for an ISO-COMPUTE comparison with the systolic
// array and VersaAccel: N_UNITS = NR Flex-DPEs x LANES = NC multipliers each,
// i.e. NR*NC = 256 multipliers at 16x16, all active every compute cycle.
//
// The previous version had one multiplier per unit, time-multiplexed over a
// step counter -- 16 multipliers against the other designs' 256 -- and its
// reduction network's mode was hard-wired to BYPASS, so synthesis deleted the
// FAN entirely. Both are fixed here.
//
// Operation: y = W x, a 16-row matrix-vector product per pass.
//   * weights  W[u][k], u < N_UNITS, k < K       (stationary; K <= K_MAX)
//   * activations x[k], k < K
//   * each compute cycle takes one LANES-wide chunk c of x, routes it through
//     the Benes (routing captured at start), broadcasts it to every unit, and
//     each unit multiplies lane-wise with its own weights W[u][c*LANES + j]
//   * each unit's FAN reduces its LANES products according to reduce_mode,
//     and the result is accumulated over chunks:
//       FULL    (0): acc[u][0]    += sum_j        p[u][j]      one dot product per unit
//       PARTIAL (1): acc[u][j<8]  += p[u][2j] + p[u][2j+1]     8 pair-sums per unit
//       BYPASS  (2): acc[u][j]    += p[u][j]                   16 lane sums per unit
//   * lanes past K are masked to zero on BOTH operands, so partial chunks and
//     a non-identity Benes routing never pick up stale data
//
// Pipeline (why): a 16x16 multiply, a 4-level 32-bit adder tree and a 32-bit
// accumulate do not fit one 2.0 ns cycle -- the systolic array's multiply-add
// alone takes ~1.4 ns at the fast corner. SIGMA itself pipelines the FAN, so
// this does too:
//   stage 0  chunk select + mask + Benes          -> dist_q   (LANES x DATA_W)
//   stage 1  256 multiplies, weights masked       -> prod_q   (N_UNITS x LANES x ACC_W)
//   stage 2  FAN per unit + accumulate            -> acc      (N_UNITS x LANES x ACC_W)
//
// Configuration (routing_ctrl, reduce_mode, num_elements) is captured into
// registers on `start`, exactly as versaaccel_top now captures its layer
// configuration, so no combinational path runs from a pin into the datapath
// and synthesis cannot treat any of it as a constant.
//
// Drain: one element per cycle, unit-major, lane-minor, over the lanes the
// mode produces -- 1, 8 or 16 per unit (16, 128 or 256 beats).
// =============================================================================
module sigma_top
  import versaaccel_pkg::*;
#(
  parameter int N_UNITS = NR,                      // Flex-DPEs
  parameter int LANES   = NC,                      // multipliers per Flex-DPE
  parameter int K_MAX   = WBUF_DEPTH / N_UNITS     // 32: weight store = WBUF_DEPTH entries
)(
  input  logic        clk,
  input  logic        rst_n,

  // Control
  input  logic        start,
  input  logic [15:0] num_elements,                // K: elements per dot product
  input  logic [BENES_CTRL_W-1:0] routing_ctrl,    // Benes switch settings
  input  logic [1:0]  reduce_mode,                 // FAN: 0=FULL, 1=PARTIAL, 2=BYPASS

  // Data input: all weights (din_is_weight=1, row-major W[u][k]) then all
  // activations (din_is_weight=0, x[k]), one element per cycle
  input  logic                    din_valid,
  input  logic signed [DATA_W-1:0] din_data,
  input  logic                    din_is_weight,
  output logic                    din_ready,

  // Data output
  output logic                    dout_valid,
  output logic signed [ACC_W-1:0] dout_data,
  input  logic                    dout_ready,

  // Status
  output logic                    busy
);

  localparam int LG_K     = $clog2(K_MAX);
  localparam int N_CHUNK  = (K_MAX + LANES - 1) / LANES;
  localparam int LG_CHUNK = (N_CHUNK > 1) ? $clog2(N_CHUNK) : 1;

  // The Benes routes one chunk, so BENES_N must equal LANES (both are NR/NC=16
  // in versaaccel_pkg). tb_sigma_golden checks this at simulation time.
  localparam logic [LG_K:0] K_MAX_V = K_MAX;

  // Signed zeros for the masking conditionals. An unsized '0 is UNSIGNED, and
  // one unsigned operand makes the whole ?: expression unsigned -- which would
  // zero-extend the 16-bit operands of the multiply to 32 bits instead of
  // sign-extending them, and silently corrupt every negative product.
  localparam logic signed [DATA_W-1:0] ZERO_D = '0;
  localparam logic signed [ACC_W-1:0]  ZERO_A = '0;

  // ==========================================================================
  // Configuration registers (captured on start)
  // ==========================================================================
  logic [BENES_CTRL_W-1:0] routing_q;
  logic [1:0]              mode_q;
  logic [LG_K:0]           k_len;        // clamped to K_MAX

  // ==========================================================================
  // Storage
  // ==========================================================================
  logic signed [DATA_W-1:0] weight_store [N_UNITS][K_MAX];
  logic signed [DATA_W-1:0] act_store    [K_MAX];
  logic signed [ACC_W-1:0]  acc          [N_UNITS][LANES];

  // ==========================================================================
  // Control FSM
  // ==========================================================================
  typedef enum logic [2:0] {
    S_IDLE, S_LOAD_W, S_LOAD_A, S_COMPUTE, S_FLUSH, S_DRAIN
  } state_t;
  state_t state;

  logic [$clog2(N_UNITS)-1:0] w_u, drain_u;
  logic [LG_K-1:0]            w_k, a_k;
  logic [$clog2(LANES)-1:0]   drain_j;
  logic [LG_CHUNK-1:0]        chunk, chunk1;   // chunk issued; chunk in stage 1
  logic                       v1, v2;          // stage-1 / stage-2 valid
  logic                       acc_clear;

  logic [LG_CHUNK:0] n_chunks;
  assign n_chunks = (k_len + LANES - 1) / LANES;

  // Lanes of each unit the current mode produces
  logic [$clog2(LANES):0] n_lanes;
  always_comb
    case (mode_q)
      2'd0:    n_lanes = 1;
      2'd1:    n_lanes = LANES / 2;
      default: n_lanes = LANES;
    endcase

  // ==========================================================================
  // Stage 0: chunk select + mask + Benes
  // ==========================================================================
  logic signed [DATA_W-1:0] dist_in  [LANES];
  logic signed [DATA_W-1:0] dist_out [LANES];
  logic signed [DATA_W-1:0] dist_q   [LANES];

  always_comb
    for (int j = 0; j < LANES; j++)
      dist_in[j] = ((int'(chunk) * LANES + j) < int'(k_len))
                 ? act_store[int'(chunk) * LANES + j] : ZERO_D;

  distribution_network benes_dist (
    .ctrl     (routing_q),
    .data_in  (dist_in),
    .data_out (dist_out)
  );

  // ==========================================================================
  // Stage 1: 256 multiplies
  // ==========================================================================
  logic signed [ACC_W-1:0] prod_q [N_UNITS][LANES];

  // ==========================================================================
  // Stage 2: one FAN per Flex-DPE
  // ==========================================================================
  // Each unit's products go through its own reduction_network -- the same
  // module VersaAccel uses -- with the mode driven from mode_q.
  wire signed [ACC_W-1:0] fan_all [N_UNITS][LANES];

  genvar gu, gj;
  generate
    for (gu = 0; gu < N_UNITS; gu++) begin : g_fan
      logic signed [ACC_W-1:0] fin  [LANES];
      logic signed [ACC_W-1:0] fout [LANES];
      always_comb
        for (int j = 0; j < LANES; j++) fin[j] = prod_q[gu][j];
      reduction_network #(.N(LANES), .WIDTH(ACC_W)) fan (
        .mode     (mode_q),
        .data_in  (fin),
        .data_out (fout)
      );
      for (gj = 0; gj < LANES; gj++) begin : g_out
        assign fan_all[gu][gj] = fout[gj];
      end
    end
  endgenerate

  // ==========================================================================
  // Datapath registers
  // ==========================================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int j = 0; j < LANES; j++) dist_q[j] <= '0;
      for (int u = 0; u < N_UNITS; u++)
        for (int j = 0; j < LANES; j++) begin
          prod_q[u][j] <= '0;
          acc[u][j]    <= '0;
        end
    end else begin
      // stage 0 -> 1
      for (int j = 0; j < LANES; j++) dist_q[j] <= dist_out[j];

      // stage 1 -> 2: weights masked past K for the same reason as activations
      for (int u = 0; u < N_UNITS; u++)
        for (int j = 0; j < LANES; j++)
          prod_q[u][j] <= ((int'(chunk1) * LANES + j) < int'(k_len))
                        ? dist_q[j] * weight_store[u][int'(chunk1) * LANES + j]
                        : ZERO_A;

      // stage 2: accumulate
      if (acc_clear) begin
        for (int u = 0; u < N_UNITS; u++)
          for (int j = 0; j < LANES; j++) acc[u][j] <= '0;
      end else if (v2) begin
        for (int u = 0; u < N_UNITS; u++)
          for (int j = 0; j < LANES; j++) acc[u][j] <= acc[u][j] + fan_all[u][j];
      end
    end
  end

  // ==========================================================================
  // FSM
  // ==========================================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state     <= S_IDLE;
      busy      <= 1'b0;
      routing_q <= '0;
      mode_q    <= '0;
      k_len     <= '0;
      w_u <= '0; w_k <= '0; a_k <= '0;
      chunk <= '0; chunk1 <= '0; v1 <= 1'b0; v2 <= 1'b0;
      drain_u <= '0; drain_j <= '0;
      acc_clear <= 1'b0;
      for (int k = 0; k < K_MAX; k++) begin
        act_store[k] <= '0;
        for (int u = 0; u < N_UNITS; u++) weight_store[u][k] <= '0;
      end
    end else begin
      acc_clear <= 1'b0;
      // pipeline valid/chunk shift -- stage 0 issues only in S_COMPUTE
      v1     <= (state == S_COMPUTE);
      chunk1 <= chunk;
      v2     <= v1;

      case (state)
        S_IDLE: begin
          busy <= 1'b0;
          if (start) begin
            busy      <= 1'b1;
            routing_q <= routing_ctrl;
            mode_q    <= reduce_mode;
            k_len     <= (int'(num_elements) > K_MAX) ? K_MAX_V
                                                      : num_elements[LG_K:0];
            w_u <= '0; w_k <= '0; a_k <= '0;
            state <= S_LOAD_W;
          end
        end

        S_LOAD_W: if (din_valid && din_is_weight) begin
          weight_store[w_u][w_k] <= din_data;
          if (int'(w_k) + 1 >= int'(k_len)) begin
            w_k <= '0;
            if (int'(w_u) == N_UNITS - 1) state <= S_LOAD_A;
            else                          w_u   <= w_u + 1'b1;
          end else
            w_k <= w_k + 1'b1;
        end

        S_LOAD_A: if (din_valid && !din_is_weight) begin
          act_store[a_k] <= din_data;
          if (int'(a_k) + 1 >= int'(k_len)) begin
            chunk     <= '0;
            acc_clear <= 1'b1;
            state     <= S_COMPUTE;
          end else
            a_k <= a_k + 1'b1;
        end

        // one chunk issued per cycle
        S_COMPUTE: begin
          if (int'(chunk) + 1 >= int'(n_chunks)) state <= S_FLUSH;
          else                                   chunk <= chunk + 1'b1;
        end

        // wait for the last chunk to leave stage 2
        S_FLUSH: if (!v1 && !v2) begin
          drain_u <= '0; drain_j <= '0;
          state   <= S_DRAIN;
        end

        S_DRAIN: if (dout_ready) begin
          if (int'(drain_j) + 1 >= int'(n_lanes)) begin
            drain_j <= '0;
            if (int'(drain_u) == N_UNITS - 1) state <= S_IDLE;
            else                              drain_u <= drain_u + 1'b1;
          end else
            drain_j <= drain_j + 1'b1;
        end

        default: state <= S_IDLE;
      endcase
    end
  end

  assign din_ready  = (state == S_LOAD_W) || (state == S_LOAD_A);
  assign dout_valid = (state == S_DRAIN);
  assign dout_data  = acc[drain_u][drain_j];

endmodule
