// =============================================================================
// versaaccel_top.sv — VersaAccel Top-Level Module
//
// 5-configuration reconfigurable accelerator:
//   DMR, D-MR, DM-R, D-M-R, D-M-R+NM
//
// Components:
//   - 8×8 PE array with configurable interconnect
//   - Benes distribution network (8-port, 5-stage)
//   - Configurable reduction network (3-stage adder tree)
//   - Config Prediction Engine (INT8 MLP)
//   - Phase FSM (workload phase detection)
//   - Speculation unit (commit/speculate/fallback)
//   - N:M routing ROM (precomputed 2:4 patterns)
//   - Input/Output buffers (SRAM-style)
// =============================================================================
module versaaccel_top
  import versaaccel_pkg::*;
(
  input  logic        clk,
  input  logic        rst_n,

  // External interface
  input  logic        start,           // Begin computation
  input  logic [1:0]  model_type,      // 0=LLM, 1=GNN, 2=CNN
  input  logic [15:0] batch_size,      // M dimension (legacy single-shot path)
  input  logic [1:0]  op_type,         // 0=MM, 1=MV, 2=SpMM, 3=SpMSpM

  // --- Tiled GEMM interface ---------------------------------------------
  // Asserting gemm_en at `start` selects the tiling controller, which streams
  // a full A[M,K] x B[K,P] through the array tile by tile. With gemm_en low the
  // original single-shot path is used unchanged.
  input  logic        gemm_en,
  input  logic [15:0] dim_m,           // rows of A
  input  logic [15:0] dim_k,           // shared dimension
  input  logic [15:0] dim_p,           // cols of B

  // Output bus width. With drain_wide low the tile is read out one element per
  // cycle on dout_data (NR*NC cycles). With it high the whole PE column is
  // presented on dout_vec every cycle (NC cycles) -- see the drain process.
  input  logic        drain_wide,

  // Zero-skipping. See the k-schedule block below for what can and cannot be
  // skipped in this dataflow. Ignored by CFG_DMR, which is the dense systolic
  // configuration and always runs the full schedule.
  input  logic        skip_en,

  // Tile shape for this GEMM (B15): tile_shape_t, captured on `start`. Only
  // meaningful when versaaccel_pkg::RESHAPE is set; otherwise ignored and the
  // tile is the square NR x NC array. See "Reshapeable tile" below.
  input  logic [1:0]  tile_shape,

  // Layer configuration, captured into registers on `start`. All three used
  // to be tied off inside this module: dist_ctrl was '0 in every
  // configuration, nm_pattern_idx was 3'd0 (a ROM entry that is itself all
  // zeros), and every CPE feature was 8'sd0. Constant inputs let synthesis
  // fold the Benes network, the N:M ROM and the entire CPE MLP away -- the
  // first Genus 21.10 run reported zero area for all three, so the netlist
  // was a systolic array without the hardware that makes it reconfigurable.
  // They belong on pins regardless: routing is chosen per layer by the
  // scheduler, and the CPE's features describe the layer being started.
  input  logic [BENES_CTRL_W-1:0] routing_ctrl,       // Benes setting, D-MR / D-M-R
  input  logic [2:0]              nm_pattern,         // N:M ROM index, D-M-R+NM
  input  logic signed [7:0]       layer_features [9], // CPE inputs, Q4.4

  // Data interface (simplified AXI-style)
  input  logic                    din_valid,
  input  logic signed [DATA_W-1:0] din_data,
  input  logic                    din_is_weight,
  output logic                    din_ready,

  output logic                    dout_valid,
  output logic signed [ACC_W-1:0] dout_data,        // narrow: one element
  output logic signed [ACC_W-1:0] dout_vec [NR],    // wide: a full PE column
  input  logic                    dout_ready,

  // Config override (for testing)
  input  logic                    cfg_override_en,
  input  config_t                 cfg_override,

  // Status
  output logic                    busy,
  output config_t                 active_config,
  output phase_t                  current_phase,
  output logic                    flush_event
);

  // ==========================================================================
  // Internal signals
  // ==========================================================================

  // CPE outputs
  config_t     cpe_config;
  logic [7:0]  cpe_confidence;
  logic        cpe_valid;

  // Phase FSM outputs
  config_t     phase_config_bias;

  // Speculation outputs
  config_t     spec_config;
  spec_mode_t  spec_mode;
  logic        spec_flush;
  logic        spec_valid;
  logic        switch_overlap;

  // Selected config (after override mux)
  config_t     selected_config;

  // Distribution network
  logic [BENES_CTRL_W-1:0] dist_ctrl;
  logic signed [DATA_W-1:0] dist_in [NR];
  logic signed [DATA_W-1:0] dist_out [NR];

  logic [BENES_CTRL_W-1:0] routing_ctrl_q;

  // N:M ROM
  logic [2:0]               nm_pattern_idx;
  logic [BENES_CTRL_W-1:0]  nm_benes_ctrl;

  // PE array
  logic signed [DATA_W-1:0] pe_a_in [NR];
  logic signed [DATA_W-1:0] pe_b_in [NC];
  logic signed [DATA_W-1:0] pe_a_inj [NR];   // B15: right-half A feed (TALL)
  logic signed [DATA_W-1:0] pe_b_inj [NC];   // B15: bottom-half B feed (WIDE)
  logic signed [ACC_W-1:0]  pe_out [NR][NC];

  // Reduction network
  logic [1:0]               reduce_mode;
  logic signed [ACC_W-1:0]  reduce_in [NR];
  logic signed [ACC_W-1:0]  reduce_out [NR];

  // Control FSM
  typedef enum logic [3:0] {
    ST_IDLE,
    ST_LOAD_WEIGHTS,
    ST_COMPUTE,
    ST_DRAIN,
    ST_RECONFIG,
    // --- tiled GEMM path ---
    ST_GEMM_LOAD,      // stream in all of B then all of A
    ST_GEMM_SCHED,     // one cycle: prime the k-schedule for this tile
    ST_GEMM_COMPUTE,   // one output tile: skewed feed, K+NR+NC-2 cycles
    ST_GEMM_CAP,       // copy the tile accumulators into the shadow buffer
    ST_GEMM_FLUSH,     // last tile captured; wait for the drain to finish
    ST_GEMM_PIPE       // B10: every tile, back to back, through one pipeline
  } ctrl_state_t;

  ctrl_state_t ctrl_state, ctrl_next;
  logic        layer_start;
  logic        acc_clear, acc_en;

  // Input/weight buffers (register-based for synthesis)
  logic signed [DATA_W-1:0] ibuf [IBUF_DEPTH];
  logic signed [DATA_W-1:0] wbuf [WBUF_DEPTH];
  logic [$clog2(IBUF_DEPTH)-1:0] ibuf_wr_ptr, ibuf_rd_ptr;
  logic [$clog2(WBUF_DEPTH)-1:0] wbuf_wr_ptr, wbuf_rd_ptr;

  // Output buffer
  logic signed [ACC_W-1:0] obuf [OBUF_DEPTH];
  logic [$clog2(OBUF_DEPTH)-1:0] obuf_wr_ptr, obuf_rd_ptr;

  // Computation counters
  logic [15:0] compute_cnt;
  logic [15:0] compute_total;

  // ==========================================================================
  // Tiled GEMM controller state
  // ==========================================================================
  // The array is output-stationary: PE(r,c) holds C[tile_i*NR+r][tile_j*NC+c]
  // and accumulates A[row][k]*B[k][col] over k.
  //
  // pe_array forwards a one cycle per column and b one cycle per row, so to
  // make PE(r,c) see A[row][k] and B[k][col] on the same cycle the feeds must
  // be skewed:
  //     a_in[r] at cycle t = A[tile_i*NR+r][t-r]
  //     b_in[c] at cycle t = B[t-c][tile_j*NC+c]
  // then a_wire[r][c] and b_wire[r][c] both carry index k = t-r-c, which walks
  // 0..K-1 as t advances. Out-of-range feeds are zero, so the pipeline fill and
  // drain contribute nothing and a single acc_clear at t=0 is sufficient.
  //
  // Total compute cycles per tile: K + NR + NC - 2.
  logic        gemm_mode;                       // sticky: this run is a GEMM
  logic [15:0] gemm_t;                          // cycle index within a tile
  logic [15:0] gemm_len;                        // K + NR + NC - 2
  logic [7:0]  tile_i, tile_j;                  // output tile coordinates
  logic [7:0]  n_tiles_i, n_tiles_j;
  logic [$clog2(NR)-1:0] drain_row;
  logic [$clog2(NC)-1:0] drain_col;
  // GEMM dimensions and skip_en, captured on `start` and held for the GEMM.
  // They were read straight off the pins, which put a pin (and its input delay)
  // at the head of every operand-address path: Genus run 2's critical path
  // started at an input port and ran through the address arithmetic, the
  // buffer read and the Benes into a PE accumulator.
  logic [15:0] dm, dk, dp;
  logic        skip_q;

  // --- output double buffer -------------------------------------------------
  // The array is output-stationary, so the thing that has to be double buffered
  // is the accumulator bank itself. tbuf is a shadow copy of one tile's NRxNC
  // accumulators: ST_GEMM_CAP copies pe_out into it in a single cycle, after
  // which the array is free to start the next tile while the drain process
  // reads tbuf out at its own rate. Cost: NR*NC*ACC_W flip-flops -- the same
  // order as the accumulators themselves.
  //
  // Note this buffers the OUTPUT only. Input double buffering would overlap the
  // fetch of the next tile's operands with compute, but this design holds all
  // of A and B in ibuf/wbuf before compute starts, so there is no per-tile
  // operand fetch to hide. That changes once operands stream from off-chip.
  logic signed [ACC_W-1:0] tbuf [NR][NC];
  logic        drain_busy;      // tbuf holds a tile that has not been read out
  logic        drain_last_beat; // this cycle's beat is the last of the tile
  logic        drain_free;      // a capture may proceed this cycle
  logic        tile_last;       // the tile being computed is the final one
  logic        out_adv;         // the output register takes a beat this cycle
  logic        cap_fire;        // the delayed tile capture lands in u_tbuf now
  logic        cap_pend;        // a capture is still travelling the feed latency

  assign drain_last_beat = drain_wide
                         ? (drain_col == NC[$clog2(NC)-1:0] - 1'b1)
                         : ((drain_row == NR[$clog2(NR)-1:0] - 1'b1) &&
                            (drain_col == NC[$clog2(NC)-1:0] - 1'b1));
  // Free when idle, or when the in-flight tile's last beat is being accepted
  // right now -- which lets tiles be captured back to back with no bubble.
  assign drain_free = !drain_busy || (out_adv && drain_last_beat);
  assign tile_last  = (tile_j + 1'b1 >= n_tiles_j) && (tile_i + 1'b1 >= n_tiles_i);

  // ==========================================================================
  // Reshapeable tile (B15, PROJECT_REPORT section 2.9)
  // ==========================================================================
  // A k-step is skippable only when it is zero across the whole tile edge that
  // faces its operand: NR rows of A, NC columns of B. A narrow edge is all-zero
  // far more often, so the best tile shape depends on which operand is sparse.
  // At the same NR*NC PEs this array can also run as
  //
  //   WIDE (NR/2 x 2NC): rows NR/2.. hold the same A rows as rows 0..NR/2-1 but
  //     the next NC output columns; every column is cut at row NR/2 and the
  //     bottom half gets its own B feed (b_inj). Narrow A edge: for ReLU.
  //   TALL (2NR x NC/2): columns NC/2.. hold the next NR output rows; every row
  //     is cut at column NC/2 and the right half gets its own A feed (a_inj).
  //     Narrow B edge: for pruned weights.
  //
  // Physically each half is an (NR/2 x NC) or (NR x NC/2) systolic array, so an
  // operand travels at most SR rows and SC columns: per-tile compute is
  // K + SR + SC - 2, shorter than the square tile's K + NR + NC - 2. Operand
  // bandwidth rises from NR + NC to NR/2 + 2NC (or 2NR + NC/2) distinct
  // values per cycle -- 40 at 16x16 instead of 32. The output tile is drained
  // exactly as before (physical tbuf columns); only the logical position of
  // each element changes, as documented at the drain.
  // Bitmap granularity, used below: GR rows of A / GC columns of B per bit.
  // TALL_ONLY: A's tile edge is NR (square) or 2NR (tall), so its granule
  // stays a whole NR rows; only B needs half-edge granules.
  localparam int GR     = (RESHAPE && !TALL_ONLY) ? NR / 2 : NR;
  localparam int GC     = RESHAPE ? NC / 2 : NC;
  localparam int GMAX   = !RESHAPE ? 1 : TALL_ONLY ? 2 : 4;  // granules per tile edge

  tile_shape_t shape_q, shape;
  int          tile_r, tile_c;     // logical tile: rows of C x cols of C
  int          span_r, span_c;     // physical propagation distance of b / a
  int          gi_n, gj_n;         // bitmap granules per tile (rows / cols)

  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n)                                  shape_q <= SHAPE_SQUARE;
    else if (ctrl_state == ST_IDLE && start)
      shape_q <= tile_shape_t'((RESHAPE && gemm_en && tile_shape != 2'd3)
                               ? tile_shape : 2'd0);

  // TALL_ONLY: the shape is one bit, so WIDE is structurally impossible and
  // synthesis removes every WIDE-only path (second B feed, column cut).
  assign shape = tile_shape_t'(TALL_ONLY ? ((gemm_mode && shape_q == SHAPE_TALL) ? 2'd2 : 2'd0)
                                          : ((RESHAPE && gemm_mode) ? shape_q : 2'd0));

  // Cut the array at its midlines for the reshaped tiles. Constant 0 when
  // RESHAPE is 0, so the pe_array muxes fold away.
  logic split_a, split_b;
  assign split_a = (shape == SHAPE_TALL);
  assign split_b = (shape == SHAPE_WIDE);

  always_comb begin
    case (shape)
      SHAPE_WIDE: begin tile_r = NR / 2; tile_c = 2 * NC; span_r = NR / 2; span_c = NC;     end
      SHAPE_TALL: begin tile_r = 2 * NR; tile_c = NC / 2; span_r = NR;     span_c = NC / 2; end
      default:    begin tile_r = NR;     tile_c = NC;     span_r = NR;     span_c = NC;     end
    endcase
    gi_n = tile_r / GR;
    gj_n = tile_c / GC;
  end

  assign gemm_len  = dk + 16'(span_r) + 16'(span_c) - 16'd2;
  // Constant divisors per shape, so synthesis builds shifts, not a divider.
  always_comb
    case (shape)
      SHAPE_WIDE: begin
        n_tiles_i = 8'((dm + NR / 2 - 1) / (NR / 2));
        n_tiles_j = 8'((dp + 2 * NC - 1) / (2 * NC));
      end
      SHAPE_TALL: begin
        n_tiles_i = 8'((dm + 2 * NR - 1) / (2 * NR));
        n_tiles_j = 8'((dp + NC / 2 - 1) / (NC / 2));
      end
      default: begin
        n_tiles_i = 8'((dm + NR - 1) / NR);
        n_tiles_j = 8'((dp + NC - 1) / NC);
      end
    endcase

  // ==========================================================================
  // Zero-skipping: the k-schedule
  // ==========================================================================
  // What CAN be skipped here is set by the dataflow, not by how many zeros
  // there are. PE(r,c) computes sum_k A[row_r][k]*B[k][col_c], and a systolic
  // array SHARES each operand along a row and a column: the same a_in[r] feeds
  // every PE in row r, the same b_in[c] every PE in column c. So step k can be
  // dropped only when it is useless for the ENTIRE tile --
  //
  //     mask[k] = (A[.][k] has a nonzero in this tile's rows)
  //             & (B[k][.] has a nonzero in this tile's columns)
  //
  // -- and the array then runs popcount(mask) steps instead of K. An isolated
  // zero saves nothing, because the cycle it would free is still needed by the
  // other NR-1 rows sharing that feed. Skipping individual elements needs a
  // dataflow where operands are not shared, which is a different architecture.
  //
  // The two bitmaps are built for free as the operands stream in (one bit-set
  // per element, no extra cycles):
  //     a_nz[ti][k] |= (A[ti*NR + r][k] != 0)
  //     b_nz[k][tj] |= (B[k][tj*NC + c] != 0)
  //
  // Feeding a skipped schedule needs no random access to it. a_in[r] at cycle
  // t wants ksched[t-r] and b_in[c] wants ksched[t-c] -- the same sequence at
  // different delays -- so one shift register khist (khist[i] = ksched[t-i])
  // serves every port, and PE(r,c) still sees a common index, now ksched[t-r-c]
  // instead of t-r-c. Everything else about the tiling is unchanged.
  // The bitmaps are kept per GR rows of A and per GC columns of B (GR, GC
  // defined with the reshapeable tile above). Square arrays use GR = NR,
  // GC = NC, exactly as before; a reshapeable one keeps them at half a tile
  // edge and ORs granules together for larger tiles.
  localparam int MAX_K  = K_MAX;                         // M*K <= IBUF_DEPTH
  localparam int MAX_TI = (M_MAX + GR - 1) / GR;
  localparam int MAX_TJ = (P_MAX + GC - 1) / GC;
  localparam int LG_K   = $clog2(MAX_K);
  localparam int LG_TI  = $clog2(MAX_TI);
  localparam int LG_TJ  = $clog2(MAX_TJ);
  localparam int LG_NR  = $clog2(GR);
  localparam int LG_NC  = $clog2(GC);
  localparam int KHIST  = (NR > NC) ? NR : NC;

  // Sized for the worst case the buffers allow, because M/K/P are runtime
  // values. A real implementation needs ceil(M/NR)*K + K*ceil(P/NC) bits --
  // 1/NR and 1/NC of the operand storage -- and would pack them into a RAM.
  logic a_nz [MAX_TI][MAX_K];
  logic b_nz [MAX_K][MAX_TJ];

  logic [15:0] a_r, a_k, b_k, b_c;      // load-time position counters
  logic        mask [MAX_K];
  logic        fst_v, nxt_v;
  logic [LG_K-1:0] fst_k, nxt_k;
  logic [LG_K-1:0] khist  [KHIST];
  logic            kvalid [KHIST];
  logic [LG_K-1:0] k_cur;
  logic            k_cur_v;
  logic [15:0]     tail_cnt;
  logic            skip_active;

  // ==========================================================================
  // Fill amortisation (B10) and streamed activations (B12)
  // ==========================================================================
  // Without AMORTISE every tile pays the skew fill: K + NR + NC - 1 cycles, of
  // which NR + NC - 2 only move operands into place (section 7.6), and that
  // fill is what caps the zero-skipping saving at f*K/(K+NR+NC) (section 7.7).
  //
  // The fill is not inherent to the dataflow. a_in[r] at cycle t carries
  // schedule entry t-r, so once a tile's last entry has entered at the corner
  // the next tile's first entry can enter on the very next cycle: every feed
  // port still delivers one value per cycle and no second port is needed.
  // What has to change is that the tiles now OVERLAP in the array -- PE(r,c)
  // finishes tile n at a different cycle from PE(0,0) -- so three things that
  // were global become per-entry or per-PE:
  //
  //   1. Each schedule entry carries the tile it belongs to (ktag_i/ktag_j),
  //      so a feed fetches the right rows/columns while its neighbours are
  //      still feeding the previous tile.
  //   2. The accumulator clear is a wavefront: an entry flagged "first"
  //      reaches PE(r,c) after r+c cycles (span-relative for reshaped tiles),
  //      and that PE clears on exactly that cycle. wfirst[] shifts alongside
  //      khist, so PE(r,c) reads it at index r+c.
  //   3. The finished value must leave the PE on that same cycle -- its
  //      accumulator still holds it, the new product lands at the edge -- so
  //      the output buffer captures per element on the same wavefront. Two
  //      banks, alternating per tile: element (r,c) of tile n waits in its bank
  //      while tile n+1 fills the other, and the drain reads whole banks.
  //
  // The schedule generator emits one entry per cycle. Dense: k = 0..K-1 per
  // tile, back to back. Skipping: the live steps only, also back to back (an
  // all-zero tile still takes one entry, to clear and hand off). It stalls
  // (emits empty entries) only when the bank it is about to fill is still
  // draining or, with STREAM, when the
  // tile's activations have not arrived. After the last tile a terminator
  // entry -- first, no data -- clears the array and captures the final tile, so
  // the fill is paid once per GEMM.
  localparam bit STRM = AMORTISE && STREAM;
  localparam int FEED_LAT = 3;                    // operand feed stages (see below)
  localparam int WAVE = NR + NC - 1 + FEED_LAT;

  logic            pipe;                          // this GEMM runs pipelined
  logic [7:0]      ktag_i [KHIST];                // tile of each schedule entry
  logic [7:0]      ktag_j [KHIST];
  logic            wfirst [WAVE];                 // entry is a tile's first
  logic            wcap   [WAVE];                 // ... and hands off a finished tile
  logic            wbank  [WAVE];                 // ... into this bank
  logic            g_started, g_tail, g_done;     // generator state
  logic            cbank, cap_any, dbank;
  logic            bank_busy [2];                 // capturing, full, or draining
  logic            bank_full [2];                 // wavefront complete, drainable
  logic            p_v, p_first, p_cap, p_end;    // entry pushed this cycle
  logic [LG_K-1:0] p_k, lst_k;                   // lst_k: the tile's last live step
  logic            tile_ready, bank_ok;
  logic            acc_clear_pe [NR][NC];
  logic            cap0 [NR][NC], cap1 [NR][NC];
  logic signed [ACC_W-1:0] tbuf1 [NR][NC];

  // B12: activations stream through a ring of two tile-rows while the array
  // computes; B is loaded first and stays resident (weights are reused by
  // every tile-row, activations by one). Row r of A lives in ibuf slot
  // r mod 2*tile_r, and the loader may write tile-row x only once nothing older
  // than tile-row x-1 is left in the feed pipeline -- i.e. input double
  // buffering at tile-row granularity. ibuf then bounds 2*tile_r*K, not M*K.
  logic            ld_busy;                       // loader running (STRM)
  logic            a_ok;                          // next A row's slot is free
  logic            ld_acc;                        // loader accepts din this cycle
  logic            a_row_start;                   // first element of a bitmap granule
  int              a_rows_tiles;                  // complete tile-rows of A loaded
  logic [$clog2(IBUF_DEPTH)-1:0] ibuf_waddr;
  int                            ld_slot;
  logic [$clog2(IBUF_DEPTH)-1:0] ld_slot_n, ld_k_n, ld_a_n, ld_prod;
  logic [LG_TI-1:0] anz_row;

  assign pipe = AMORTISE && gemm_mode;

  // ==========================================================================
  // Buffer storage (B1d)
  // ==========================================================================
  // The five on-chip buffers live in their own submodules so that synthesis
  // reports their area separately (area_hier). Before this, all of them were
  // flip-flop arrays flattened into this module -- 58.6% of the first run's
  // cell area, indistinguishable from control -- and the systolic baseline
  // has no equivalent buffers, so top-level totals could not be compared.
  //
  // Only STORAGE moved. Every write is driven by a strobe decoded from exactly
  // the condition the FSM used to write the array directly, at the same clock
  // edge, with the same address and data; every read is unchanged. None of
  // the five is reset, as before -- they model SRAM, and tb_versaaccel_golden
  // relies on unwritten ibuf entries never reaching the array.
  logic wbuf_we, ibuf_we, bnz_set, anz_set, nz_clear, tbuf_cap;

  always_comb begin
    // ST_LOAD_WEIGHTS / ST_GEMM_LOAD: "if (din_valid && din_is_weight)" and
    // its "else if (din_valid && !din_is_weight)"
    // (B12: or the streaming loader, ld_acc, which runs alongside ST_GEMM_PIPE)
    wbuf_we  = (((ctrl_state == ST_LOAD_WEIGHTS) || (ctrl_state == ST_GEMM_LOAD))
                && din_valid && din_is_weight) || (ld_acc && din_is_weight);
    ibuf_we  = (((ctrl_state == ST_LOAD_WEIGHTS) || (ctrl_state == ST_GEMM_LOAD))
                && din_valid && !din_is_weight) || (ld_acc && !din_is_weight);
    // ST_GEMM_LOAD: one OR per nonzero element as it arrives
    bnz_set  = (((ctrl_state == ST_GEMM_LOAD) && din_valid) || ld_acc) && din_is_weight
             && (din_data != '0);
    anz_set  = (((ctrl_state == ST_GEMM_LOAD) && din_valid) || ld_acc) && !din_is_weight
             && (din_data != '0);
    // ST_IDLE: "if (start) ... if (gemm_en)" -- start of a new GEMM
    nz_clear = (ctrl_state == ST_IDLE) && start && gemm_en;
    // ST_GEMM_CAP: "if (drain_free)"
    tbuf_cap = (ctrl_state == ST_GEMM_CAP) && drain_free;
  end

  operand_buffer #(.DEPTH(WBUF_DEPTH), .WIDTH(DATA_W)) u_wbuf (
    .clk(clk), .we(wbuf_we), .waddr(wbuf_wr_ptr), .wdata(din_data), .q(wbuf));

  operand_buffer #(.DEPTH(IBUF_DEPTH), .WIDTH(DATA_W)) u_ibuf (
    .clk(clk), .we(ibuf_we), .waddr(ibuf_waddr), .wdata(din_data), .q(ibuf));

  // B arrives k-major, element (b_k, b_c): row = k, column = the col-tile of b_c
  nz_bitmap #(.ROWS(MAX_K), .COLS(MAX_TJ)) u_b_nz (
    .clk(clk), .clear(nz_clear), .clear_row(1'b0), .set(bnz_set),
    .row(b_k[LG_K-1:0]), .col(b_c[LG_NC+LG_TJ-1:LG_NC]), .q(b_nz));

  // A arrives row-major, element (a_r, a_k): row = the row-tile of a_r, column = k
  // With STREAM the granule rows are a ring too, and each is cleared as its
  // first element arrives (a_row_start), since it last held tile-row x-2.
  nz_bitmap #(.ROWS(MAX_TI), .COLS(MAX_K)) u_a_nz (
    .clk(clk), .clear(nz_clear), .clear_row(a_row_start), .set(anz_set),
    .row(anz_row), .col(a_k[LG_K-1:0]), .q(a_nz));

  // Bank 0 is the original shadow buffer: captured whole by ST_GEMM_CAP, or
  // element by element on the B10 wavefront. Bank 1 exists only with
  // AMORTISE (see "Fill amortisation" below).
  tile_buffer u_tbuf (.clk(clk), .capture(cap_fire), .cap_pe(cap0), .d(pe_out), .q(tbuf));
  generate if (AMORTISE) begin : g_tbuf1
    tile_buffer u_tbuf1 (.clk(clk), .capture(1'b0), .cap_pe(cap1), .d(pe_out), .q(tbuf1));
  end else begin : g_no_tbuf1
    for (genvar r = 0; r < NR; r++) begin : g_r
      for (genvar c = 0; c < NC; c++) begin : g_c
        assign tbuf1[r][c] = '0;
      end
    end
  end endgenerate

  // CFG_DMR is the dense systolic configuration: it always runs the full K
  // schedule, so that the dense baseline stays a dense baseline.
  assign skip_active = gemm_mode && skip_q && (selected_config != CFG_DMR);

  // B12 loader: where the next A element goes, and whether it may go yet.
  // Divisions are by per-shape constants, so they are shifts.
  always_comb begin
    case (shape)
      SHAPE_WIDE: a_rows_tiles = int'(a_r) / (NR / 2);
      SHAPE_TALL: a_rows_tiles = int'(a_r) / (2 * NR);
      default:    a_rows_tiles = int'(a_r) / NR;
    endcase
    // tile-row of the row being written <= oldest tile-row still in the feeds + 1
    a_ok        = !STRM || (a_rows_tiles <= int'(ktag_i[KHIST-1]) + 1);
    ld_acc      = STRM && ld_busy && a_ok && din_valid;
    a_row_start = ld_acc && !din_is_weight && (a_k == '0) && (int'(a_r) % GR == 0);
    // ring slot * K in the buffer's index width (see "Operand feed pipeline")
    case (shape)
      SHAPE_WIDE: ld_slot = int'(a_r) % NR;
      SHAPE_TALL: ld_slot = int'(a_r) % (4 * NR);
      default:    ld_slot = int'(a_r) % (2 * NR);
    endcase
    ld_slot_n   = ld_slot[$clog2(IBUF_DEPTH)-1:0];
    ld_k_n      = dk[$clog2(IBUF_DEPTH)-1:0];
    ld_a_n      = a_k[$clog2(IBUF_DEPTH)-1:0];
    ld_prod     = ld_slot_n * ld_k_n;
    ibuf_waddr  = STRM ? ld_prod + ld_a_n : ibuf_wr_ptr;
    case (shape)
      SHAPE_WIDE: anz_row = STRM ? LG_TI'((int'(a_r) / GR) % (2 * ((NR / 2) / GR)))
                                 : a_r[LG_NR+LG_TI-1:LG_NR];
      SHAPE_TALL: anz_row = STRM ? LG_TI'((int'(a_r) / GR) % (2 * ((2 * NR) / GR)))
                                 : a_r[LG_NR+LG_TI-1:LG_NR];
      default:    anz_row = STRM ? LG_TI'((int'(a_r) / GR) % (2 * (NR / GR)))
                                 : a_r[LG_NR+LG_TI-1:LG_NR];
    endcase
    // a tile may start once B is resident and its activation rows have arrived
    tile_ready  = !STRM || ((int'(b_k) >= int'(dk)) &&
                            (a_rows_tiles > int'(tile_i) || a_r >= dm));
  end

  // A step is live if any granule of this tile's rows has a nonzero A at k,
  // and any granule of its columns a nonzero B. With RESHAPE = 0 there is one
  // granule per tile and this is a_nz[tile_i][k] & b_nz[k][tile_j], as before.
  logic a_any [MAX_K];
  logic b_any [MAX_K];
  // First bitmap granule of the current tile. gi_n / gj_n are per-shape
  // constants, so these are shifts rather than multipliers.
  int ga0, gb0;
  always_comb begin
    case (shape)
      SHAPE_WIDE: begin
        ga0 = (STRM ? int'(tile_i) % 2 : int'(tile_i)) * ((NR / 2) / GR);
        gb0 = int'(tile_j) * ((2 * NC) / GC);
      end
      SHAPE_TALL: begin
        ga0 = (STRM ? int'(tile_i) % 2 : int'(tile_i)) * ((2 * NR) / GR);
        gb0 = int'(tile_j) * ((NC / 2) / GC);
      end
      default: begin
        ga0 = (STRM ? int'(tile_i) % 2 : int'(tile_i)) * (NR / GR);
        gb0 = int'(tile_j) * (NC / GC);
      end
    endcase
  end

  always_comb
    for (int k = 0; k < MAX_K; k++) begin
      a_any[k] = 1'b0; b_any[k] = 1'b0;
      for (int g = 0; g < GMAX; g++) begin
        // (STRM: the ring of two tile-rows)
        if (g < gi_n && ga0 + g < MAX_TI)
          a_any[k] = a_any[k] | a_nz[ga0 + g][k];
        if (g < gj_n && gb0 + g < MAX_TJ)
          b_any[k] = b_any[k] | b_nz[k][gb0 + g];
      end
      // Pipelined dense GEMMs use the same encoder with every step live.
      mask[k] = (k < int'(dk)) && ((pipe && !skip_active) || (a_any[k] && b_any[k]));
    end

  // The mask is a function of the tile only, and every step after a tile's
  // first is chosen at least one cycle after the tile began -- so those steps
  // read a registered copy. That takes the bitmap lookup out of the per-step
  // path (bitmap row mux -> mask -> 256-wide compare-and-encode -> schedule
  // register); only a tile's first step still sees the live mask.
  logic mask_q [MAX_K];
  logic [LG_K-1:0] lst_kq;
  always_ff @(posedge clk)
    for (int k = 0; k < MAX_K; k++) mask_q[k] <= mask[k];

  // Scanning downward and overwriting leaves the LOWEST match in each output:
  // fst_k = first scheduled step of the tile, nxt_k = the one after k_cur.
  always_comb begin
    fst_v = 1'b0; fst_k = '0;
    nxt_v = 1'b0; nxt_k = '0;
    for (int k = MAX_K - 1; k >= 0; k--) begin
      if (mask[k]) begin
        fst_v = 1'b1;
        fst_k = k[LG_K-1:0];
      end
      if (mask_q[k] && k > int'(k_cur)) begin
        nxt_v = 1'b1;
        nxt_k = k[LG_K-1:0];
      end
    end
  end

  // The tile's last live step, so the generator knows a tile is over as it
  // issues that step and starts the next tile on the following cycle. (Scanning
  // upward and overwriting leaves the HIGHEST match.) Only AMORTISE reads them:
  // lst_k for a tile's first step, lst_kq for the rest.
  always_comb begin
    lst_k = '0; lst_kq = '0;
    for (int k = 0; k < MAX_K; k++) begin
      if (mask[k])   lst_k  = k[LG_K-1:0];
      if (mask_q[k]) lst_kq = k[LG_K-1:0];
    end
  end

  // --- B10 schedule generator: the entry entering the array this cycle -----
  always_comb begin
    p_v = 1'b0; p_first = 1'b0; p_cap = 1'b0; p_end = 1'b0; p_k = '0;
    bank_ok = !cap_any || !bank_busy[cbank];
    if (ctrl_state == ST_GEMM_PIPE && !g_done) begin
      if (g_tail) begin                         // terminator: flush the last tile
        if (bank_ok) begin p_first = 1'b1; p_cap = 1'b1; end
      end else if (!g_started) begin            // a tile's first entry
        if (tile_ready && bank_ok) begin
          p_first = 1'b1; p_cap = cap_any; p_v = fst_v; p_k = fst_k;
          p_end   = !fst_v || (fst_k == lst_k);
        end
      end else begin                            // the rest of the tile
        p_v   = nxt_v; p_k = nxt_k;
        p_end = !nxt_v || (nxt_k == lst_kq);
      end
    end
  end

  // --- B10 wavefront: PE(r,c) acts on the entry r+c cycles old ------------
  // (span-relative: a reshaped half is its own span_r x span_c array), plus
  // the FEED_LAT cycles the operand feed takes to reach the array.
  always_comb
    for (int r = 0; r < NR; r++)
      for (int c = 0; c < NC; c++) begin
        acc_clear_pe[r][c] = pipe && wfirst[(r % span_r) + (c % span_c) + FEED_LAT];
        cap0[r][c] = acc_clear_pe[r][c] && wcap[(r % span_r) + (c % span_c) + FEED_LAT]
                   && !wbank[(r % span_r) + (c % span_c) + FEED_LAT];
        cap1[r][c] = acc_clear_pe[r][c] && wcap[(r % span_r) + (c % span_c) + FEED_LAT]
                   &&  wbank[(r % span_r) + (c % span_c) + FEED_LAT];
      end

  // ==========================================================================
  // Operand feed pipeline
  // ==========================================================================
  // Feeding an operand used to be one combinational cycle: tile/schedule state
  // -> address arithmetic (a full row*K multiply per port) -> a 256- or 512-way
  // buffer read -> the 7-stage Benes -> the PE multiply-accumulate. Genus
  // (gsclib045, slow corner) put that path at 5.6 ns against 2.0 ns for the
  // systolic baseline's MAC. It is now three registered stages in front of the
  // array:
  //
  //   F0  address: which buffer entry each feed port wants, and whether the
  //       element exists (inside the matrix and, when skipping, scheduled)
  //   F1  read: the buffer entry, zero when it does not exist
  //   F2  distribute: A through the Benes; B and the injected half-feeds wait
  //
  // so a PE sees in cycle t+FEED_LAT exactly what it used to see in cycle t.
  // Every PE-side control signal -- the accumulator clear and enable, the
  // per-PE clear wavefront, the tile captures -- is delayed by the same
  // FEED_LAT, so the array's behaviour is unchanged cycle for cycle and only
  // shifted: per-tile cost is the same, and a GEMM ends FEED_LAT cycles later.
  //
  // Addresses are computed in the buffers' own index width. A buffer entry's
  // index is row*K + k < M*K <= IBUF_DEPTH (K*P <= WBUF_DEPTH for B) whenever
  // the element is inside the matrix -- the capacity the simulation-only check
  // below enforces -- so only the low log2(DEPTH) bits of the product are ever
  // needed, and the low bits of a product depend only on the low bits of its
  // operands. That replaces the old 32-bit multiplier per port with a
  // log2(DEPTH)-bit one. Both depths are powers of two.
  localparam int IA = $clog2(IBUF_DEPTH);
  localparam int IW = $clog2(WBUF_DEPTH);
  // An address word is {valid, index}.

  // First logical row of tile-row ti / column of tile-column tj. The tile
  // dimensions are per-shape constants, so these are shifts, not multipliers.
  function automatic int row0(int ti);
    case (shape)
      SHAPE_WIDE: return ti * (NR / 2);
      SHAPE_TALL: return ti * (2 * NR);
      default:    return ti * NR;
    endcase
  endfunction
  function automatic int col0(int tj);
    case (shape)
      SHAPE_WIDE: return tj * (2 * NC);
      SHAPE_TALL: return tj * (NC / 2);
      default:    return tj * NC;
    endcase
  endfunction

  // A[row][t - sk], absent outside the matrix. sk is the feed's skew -- its
  // distance from the far corner of its half -- and roff the row's offset in
  // the logical tile. Square tile: row = tile_i*NR + r, sk = r, as before.
  // Pipelined (B10), the tile comes from the entry itself, and with STREAM
  // (B12) the row lives in the two-tile-row ring.
  function automatic logic [IA:0] addr_a(int t, int sk, int roff);
    int ka, ti, row, slot;
    logic [IA-1:0] s_n, k_n, a_n, p_n;
    if (skip_active || pipe) begin
      if (!kvalid[sk]) return '0;
      ka = int'(khist[sk]);
    end else
    ka = t - sk;
    ti  = pipe ? int'(ktag_i[sk]) : int'(tile_i);
    row = row0(ti) + roff;
    if (ka < 0 || ka >= int'(dk))                   return '0;
    if (row >= int'(dm))                            return '0;
    slot = STRM ? row0(ti % 2) + roff : row;
    s_n = slot[IA-1:0]; k_n = dk[IA-1:0]; a_n = ka[IA-1:0];
    p_n = s_n * k_n;                                // low IA bits of slot*K
    return {1'b1, p_n + a_n};
  endfunction

  // B[t - sk][col], absent outside the matrix. Square: col = tile_j*NC + c, sk = c.
  function automatic logic [IW:0] addr_b(int t, int sk, int coff);
    int kb, col;
    logic [IW-1:0] k_n, p_n, c_n, m_n;
    if (skip_active || pipe) begin
      if (!kvalid[sk]) return '0;
      kb = int'(khist[sk]);
    end else
    kb  = t - sk;
    col = col0(pipe ? int'(ktag_j[sk]) : int'(tile_j)) + coff;
    if (kb < 0 || kb >= int'(dk))                   return '0;
    if (col >= int'(dp))                            return '0;
    k_n = kb[IW-1:0]; p_n = dp[IW-1:0]; c_n = col[IW-1:0];
    m_n = k_n * p_n;                                // low IW bits of k*P
    return {1'b1, m_n + c_n};
  endfunction

  // ==========================================================================
  // Config selection
  // ==========================================================================
  // Registered, so no path runs from the override pins (or from the
  // speculation unit) into the operand feed in the same cycle -- the
  // cfg_override_en pin headed Genus run 2's 5.6 ns critical path.
  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) selected_config <= CFG_DMR;
    else        selected_config <= config_t'(cfg_override_en ? cfg_override : spec_config);
  assign active_config   = selected_config;

  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) begin
      dm <= '0; dk <= '0; dp <= '0; skip_q <= 1'b0;
    end else if (ctrl_state == ST_IDLE && start) begin
      dm <= dim_m; dk <= dim_k; dp <= dim_p; skip_q <= skip_en;
    end

  // Distribution network control based on config.
  //
  // '0 is straight-through. The configurations whose distribution stage is
  // explicit (section 1: D-MR and D-M-R) take their routing from the layer
  // configuration register; the systolic-distribution ones (DMR, DM-R) pass
  // straight through; the N:M configuration reads its pattern from the ROM.
  //
  // In GEMM mode the tiling controller computes a row-skewed feed that
  // assumes identity routing, so the Benes is forced straight-through there
  // -- the same way reduce_mode is forced to BYPASS below. This is the
  // single-dataflow scope statement of section 7.9, made explicit in RTL.
  always_comb begin
    if (gemm_mode) dist_ctrl = '0;
    else
    case (selected_config)
      CFG_DMR:      dist_ctrl = '0;             // systolic distribution
      CFG_D_MR:     dist_ctrl = routing_ctrl_q; // explicit (Benes)
      CFG_DM_R:     dist_ctrl = '0;             // systolic distribution
      CFG_D_M_R:    dist_ctrl = routing_ctrl_q; // explicit (Benes)
      CFG_D_M_R_NM: dist_ctrl = nm_benes_ctrl;  // from the N:M ROM
      default:      dist_ctrl = '0;
    endcase
  end

  // Reduction mode based on config
  always_comb begin
    // An output-stationary GEMM wants each accumulator delivered individually,
    // so the tiling controller forces BYPASS. FULL and PARTIAL reduction belong
    // to the reduction-style operators (MV and friends), where a PE column is
    // summed into a single result.
    if (gemm_mode) reduce_mode = 2'd2;
    else
    case (selected_config)
      CFG_DMR:      reduce_mode = 2'd0; // Full reduction
      CFG_D_MR:     reduce_mode = 2'd1; // Partial
      CFG_DM_R:     reduce_mode = 2'd2; // Bypass
      CFG_D_M_R:    reduce_mode = 2'd2; // Bypass
      CFG_D_M_R_NM: reduce_mode = 2'd2; // Bypass
      default:      reduce_mode = 2'd2;
    endcase
  end

  // ==========================================================================
  // Module instantiations
  // ==========================================================================

  // --- Distribution Network ---
  distribution_network dist_net (
    .ctrl     (dist_ctrl),
    .data_in  (dist_in),
    .data_out (dist_out)
  );

  // --- N:M Routing ROM ---
  nm_routing_rom nm_rom (
    .pattern_idx (nm_pattern_idx),
    .benes_ctrl  (nm_benes_ctrl)
  );

  // --- PE Array ---
  pe_array pe_arr (
    .clk         (clk),
    .rst_n       (rst_n),
    .config_mode (selected_config),
    .acc_clear   (acc_clear),
    .acc_en      (acc_en),
    .a_in        (pe_a_in),
    .b_in        (pe_b_in),
    .split_a     (split_a),
    .split_b     (split_b),
    .a_inj       (pe_a_inj),
    .b_inj       (pe_b_inj),
    .acc_clear_pe(acc_clear_pe),
    .pe_out      (pe_out)
  );

  // --- Reduction Network ---
  // Use column 0 outputs for reduction
  // Only one PE column can reach the reduction network at a time; the legacy
  // path taps column 0. The GEMM drain sweeps drain_col across the shadow
  // buffer instead (see the output register), so all NRxNC accumulators become
  // observable without stalling the next tile.
  always_comb begin
    for (int i = 0; i < NR; i++)
      reduce_in[i] = pe_out[i][0];
  end

  reduction_network red_net (
    .mode     (reduce_mode),
    .data_in  (reduce_in),
    .data_out (reduce_out)
  );

  // --- Phase FSM ---
  phase_fsm phase (
    .clk           (clk),
    .rst_n         (rst_n),
    .batch_size    (batch_size),
    .op_type       (op_type),
    .layer_start   (layer_start),
    .model_type    (model_type),
    .current_phase (current_phase),
    .config_bias   (phase_config_bias)
  );

  // --- CPE (Config Prediction Engine) ---
  // Feature order: input_sparsity, weight_sparsity, is_conv, log2(M),
  // log2(K), log2(P), layer_position, attention_sparsity, nm_compatibility.
  logic signed [7:0] cpe_features [9];

  // Layer configuration registers. Captured on the edge that raises
  // layer_start, so they are stable in exactly the cycle the CPE samples
  // them (valid_in = layer_start) and held for the whole layer. Registering
  // them also keeps pins out of the timing: a combinational path from
  // routing_ctrl through the Benes into a PE MAC would be an artefact of the
  // test harness, not of the design.
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      routing_ctrl_q <= '0;
      nm_pattern_idx <= '0;
      for (int f = 0; f < 9; f++) cpe_features[f] <= '0;
    end else if (ctrl_state == ST_IDLE && start) begin
      routing_ctrl_q <= routing_ctrl;
      nm_pattern_idx <= nm_pattern;
      for (int f = 0; f < 9; f++) cpe_features[f] <= layer_features[f];
    end
  end

  config_prediction_engine cpe (
    .clk              (clk),
    .rst_n            (rst_n),
    .valid_in         (layer_start),
    .features         (cpe_features),
    .predicted_config (cpe_config),
    .confidence       (cpe_confidence),
    .valid_out        (cpe_valid)
  );

  // --- Speculation Unit ---
  speculation_unit spec (
    .clk                (clk),
    .rst_n              (rst_n),
    .predicted_config   (cpe_config),
    .cpe_confidence     (cpe_confidence),
    .prediction_valid   (cpe_valid),
    .phase_default      (phase_config_bias),
    .sample_done        (1'b0),  // Connected to data sampling in real design
    .actual_sparsity    (8'd0),
    .predicted_sparsity (8'd0),
    .thresh_high        (8'd230),
    .thresh_med         (8'd179),
    .sparsity_tol       (8'd26),
    .active_config      (spec_config),
    .spec_mode          (spec_mode),
    .flush              (spec_flush),
    .config_valid       (spec_valid),
    .switch_overlap     (switch_overlap)
  );

  assign flush_event = spec_flush;

  // ==========================================================================
  // Control FSM
  // ==========================================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      ctrl_state  <= ST_IDLE;
      ibuf_wr_ptr <= '0;
      ibuf_rd_ptr <= '0;
      wbuf_wr_ptr <= '0;
      wbuf_rd_ptr <= '0;
      obuf_wr_ptr <= '0;
      obuf_rd_ptr <= '0;
      compute_cnt <= '0;
      busy        <= 1'b0;
      layer_start <= 1'b0;
      gemm_mode   <= 1'b0;
      gemm_t      <= '0;
      tile_i      <= '0;
      tile_j      <= '0;
      drain_row   <= '0;
      drain_col   <= '0;
      drain_busy  <= 1'b0;
      a_r <= '0; a_k <= '0; b_k <= '0; b_c <= '0;
      k_cur <= '0; k_cur_v <= 1'b0; tail_cnt <= '0;
      for (int i = 0; i < KHIST; i++) begin
        khist[i]  <= '0;
        kvalid[i] <= 1'b0;
        ktag_i[i] <= '0;
        ktag_j[i] <= '0;
      end
      for (int i = 0; i < WAVE; i++) begin
        wfirst[i] <= 1'b0; wcap[i] <= 1'b0; wbank[i] <= 1'b0;
      end
      g_started <= 1'b0; g_tail <= 1'b0; g_done <= 1'b0;
      cbank <= 1'b0; cap_any <= 1'b0; dbank <= 1'b0; ld_busy <= 1'b0;
      for (int b = 0; b < 2; b++) begin bank_busy[b] <= 1'b0; bank_full[b] <= 1'b0; end
    end else begin
      layer_start <= 1'b0;

      // ------------------------------------------------------------------
      // Drain process -- runs concurrently with the compute FSM below.
      // Reads the shadow buffer, so it neither stalls nor is stalled by the
      // tile currently in the array. drain_wide retires a whole PE column per
      // cycle (NC beats/tile); otherwise one element per cycle (NR*NC beats).
      // ------------------------------------------------------------------
      if (drain_busy && out_adv) begin
        if (drain_last_beat) begin
          drain_row  <= '0;
          drain_col  <= '0;
          drain_busy <= 1'b0;
        end else if (drain_wide) begin
          drain_col <= drain_col + 1'b1;
        end else if (drain_row == NR[$clog2(NR)-1:0] - 1'b1) begin
          drain_row <= '0;
          drain_col <= drain_col + 1'b1;
        end else begin
          drain_row <= drain_row + 1'b1;
        end
      end

      // B12 streaming loader: the ST_GEMM_LOAD counters, running alongside
      // ST_GEMM_PIPE. The data writes are u_wbuf / u_ibuf on ld_acc.
      if (ld_acc) begin
        if (din_is_weight) begin
          wbuf_wr_ptr <= wbuf_wr_ptr + 1'b1;   // A's address is ibuf_waddr (ring)
          if (b_c + 1'b1 >= dp) begin b_c <= '0; b_k <= b_k + 1'b1; end
          else                            b_c <= b_c + 1'b1;
        end else begin
          if (a_k + 1'b1 >= dk) begin a_k <= '0; a_r <= a_r + 1'b1; end
          else                            a_k <= a_k + 1'b1;
          if ((a_k + 1'b1 >= dk) && (a_r + 1'b1 >= dm)) ld_busy <= 1'b0;
        end
      end

      // The capture in ST_GEMM_CAP below assigns drain_busy after this block,
      // so a capture in the same cycle as the final beat wins -- that is what
      // makes back-to-back tiles possible.
      case (ctrl_state)
        ST_IDLE: begin
          busy <= 1'b0;
          if (start) begin
            busy        <= 1'b1;
            layer_start <= 1'b1;
            ibuf_wr_ptr <= '0;
            wbuf_wr_ptr <= '0;
            gemm_mode   <= gemm_en;
            if (gemm_en) begin
              // STREAM: compute starts at once and the loader runs alongside
              ctrl_state <= STRM ? ST_GEMM_PIPE : ST_GEMM_LOAD;
              ld_busy    <= STRM;
              if (AMORTISE) begin                    // empty pipeline, both banks free
                for (int i = 0; i < KHIST; i++) begin
                  kvalid[i] <= 1'b0; ktag_i[i] <= '0; ktag_j[i] <= '0;
                end
                for (int i = 0; i < WAVE; i++) begin
                  wfirst[i] <= 1'b0; wcap[i] <= 1'b0;
                end
                g_started <= 1'b0; g_tail <= 1'b0; g_done <= 1'b0;
                cbank <= 1'b0; cap_any <= 1'b0; dbank <= 1'b0;
                for (int b = 0; b < 2; b++) begin bank_busy[b] <= 1'b0; bank_full[b] <= 1'b0; end
              end
              tile_i     <= '0;
              tile_j     <= '0;
              gemm_t     <= '0;
              // Start of a new GEMM: reset the nonzero bitmaps and the load
              // position counters that build them.
              // (the bitmaps are cleared by nz_clear in u_a_nz / u_b_nz)
              a_r <= '0; a_k <= '0; b_k <= '0; b_c <= '0;
            end else begin
              ctrl_state <= ST_LOAD_WEIGHTS;
            end
          end
        end

        ST_LOAD_WEIGHTS: begin
          if (din_valid && din_is_weight) begin
            wbuf_wr_ptr <= wbuf_wr_ptr + 1'b1;        // data: u_wbuf, wbuf_we
          end else if (din_valid && !din_is_weight) begin
            ibuf_wr_ptr <= ibuf_wr_ptr + 1'b1;        // data: u_ibuf, ibuf_we
            ctrl_state <= ST_COMPUTE;
            compute_cnt <= '0;
          end
        end

        ST_COMPUTE: begin
          if (spec_flush) begin
            ctrl_state <= ST_RECONFIG;
          end else begin
            compute_cnt <= compute_cnt + 1'b1;
            // FEED_LAT more cycles, enable low, for the feed to reach the array
            if (compute_cnt >= compute_total - 1 + FEED_LAT) begin
              ctrl_state <= ST_DRAIN;
            end
          end
        end

        ST_DRAIN: begin
          if (out_adv) begin
            obuf_rd_ptr <= obuf_rd_ptr + 1'b1;
            if (obuf_rd_ptr >= obuf_wr_ptr - 1)
              ctrl_state <= ST_IDLE;
          end
        end

        ST_RECONFIG: begin
          // 1-cycle reconfiguration penalty
          ctrl_state <= ST_COMPUTE;
          compute_cnt <= '0;
        end

        // ------------------------------------------------------------------
        // Tiled GEMM: load B in full, then A in full.
        // ------------------------------------------------------------------
        ST_GEMM_LOAD: begin
          if (din_valid && din_is_weight) begin
            wbuf_wr_ptr <= wbuf_wr_ptr + 1'b1;        // data: u_wbuf, wbuf_we
            // B arrives k-major: element (b_k, b_c). The bitmap OR is done by
            // u_b_nz on bnz_set, so it costs no cycles of its own.
            if (b_c + 1'b1 >= dp) begin
              b_c <= '0;
              b_k <= b_k + 1'b1;
            end else begin
              b_c <= b_c + 1'b1;
            end
          end else if (din_valid && !din_is_weight) begin
            ibuf_wr_ptr <= ibuf_wr_ptr + 1'b1;        // data: u_ibuf, ibuf_we
            // A arrives row-major: element (a_r, a_k); bitmap OR in u_a_nz.
            if (a_k + 1'b1 >= dk) begin
              a_k <= '0;
              a_r <= a_r + 1'b1;
            end else begin
              a_k <= a_k + 1'b1;
            end
            // Last activation element: both operands are resident. Tested on
            // the (row, col) counters, not on ibuf_wr_ptr against a truncated
            // a_total -- when M*K is an exact multiple of IBUF_DEPTH the
            // truncation makes the threshold 0 and the load terminates after a
            // SINGLE element. Found by tb_gemm_sparse at M=K=16, M*K=256.
            if ((a_k + 1'b1 >= dk) && (a_r + 1'b1 >= dm)) begin
              ctrl_state <= ctrl_state_t'(AMORTISE ? ST_GEMM_PIPE
                                         : skip_active ? ST_GEMM_SCHED : ST_GEMM_COMPUTE);
              gemm_t     <= '0;
            end
          end
        end

        // ------------------------------------------------------------------
        // Prime the schedule for this tile: khist[0] = ksched[0], the rest
        // invalid. One cycle, and only when skipping is active, so the dense
        // path keeps its measured K+NR+NC-1 cycles per tile.
        // ------------------------------------------------------------------
        ST_GEMM_SCHED: begin
          khist[0]  <= fst_k;
          kvalid[0] <= fst_v;
          for (int i = 1; i < KHIST; i++) kvalid[i] <= 1'b0;
          k_cur     <= fst_k;
          k_cur_v   <= fst_v;
          tail_cnt  <= '0;
          gemm_t    <= '0;
          ctrl_state <= ST_GEMM_COMPUTE;
        end

        // ------------------------------------------------------------------
        // One output tile. Skewed feed; acc_clear fires at gemm_t == 0.
        // ------------------------------------------------------------------
        ST_GEMM_COMPUTE: begin
          gemm_t <= gemm_t + 1'b1;   // only acc_clear reads this when skipping
          if (skip_active) begin
            // advance the schedule by one step
            for (int i = KHIST - 1; i > 0; i--) begin
              khist[i]  <= khist[i-1];
              kvalid[i] <= kvalid[i-1];
            end
            khist[0]  <= nxt_k;
            kvalid[0] <= nxt_v;
            if (nxt_v) k_cur <= nxt_k;   // hold on exhaustion, or nxt would
            k_cur_v <= nxt_v;            // wrap back to the first step
            // Once the schedule is spent, NR+NC-2 more cycles are needed for
            // the last step to reach PE(NR-1,NC-1).
            if (!k_cur_v) begin
              tail_cnt <= tail_cnt + 1'b1;
              if (tail_cnt >= 16'(span_r) + 16'(span_c) - 16'd3)
                ctrl_state <= ST_GEMM_CAP;
            end else begin
              tail_cnt <= '0;
            end
          end else begin
            if (gemm_t >= gemm_len - 1)
              ctrl_state <= ST_GEMM_CAP;
          end
        end

        // ------------------------------------------------------------------
        // Capture. One cycle, because pe_out is the accumulator REGISTER: the
        // final product of cycle gemm_len-1 only lands at the edge that ends
        // it, so pe_out is not complete until the cycle after. acc_en is low
        // here, which freezes the accumulators while we wait for the drain.
        // The array runs FEED_LAT cycles behind this FSM, so the capture
        // itself (u_tbuf, on cap_fire) and the start of the drain may happen
        // up to FEED_LAT cycles later -- see "Tile capture" -- still long
        // before the next tile's capture, at least NR+NC-1 cycles away.
        //
        // The array is then restarted immediately on the next tile. That is
        // safe with no flush bubble: a feed is zero for t >= K+NR-1 (the last
        // NC-1 cycles) and b for t >= K+NC-1 (the last NR-1 cycles), so the
        // stale values still in the systolic forwarding registers when the new
        // tile begins are exactly the ones that are zero.
        // ------------------------------------------------------------------
        ST_GEMM_CAP: begin
          if (drain_free) begin
            // the NRxNC capture itself is u_tbuf, on cap_fire
            if (tile_last) begin
              ctrl_state <= ST_GEMM_FLUSH;
            end else begin
              ctrl_state <= ctrl_state_t'(skip_active ? ST_GEMM_SCHED : ST_GEMM_COMPUTE);
              gemm_t     <= '0;
              if (tile_j + 1'b1 >= n_tiles_j) begin
                tile_j <= '0;
                tile_i <= tile_i + 1'b1;
              end else begin
                tile_j <= tile_j + 1'b1;
              end
            end
          end
          // else: hold. The consumer is behind; accumulators are frozen.
        end

        // ------------------------------------------------------------------
        // B10: every tile through one pipeline. Each cycle one schedule entry
        // (from the generator above) enters khist with its tile tag, and its
        // first/capture flags enter the wavefront register alongside.
        // ------------------------------------------------------------------
        ST_GEMM_PIPE: begin
          for (int i = KHIST - 1; i > 0; i--) begin
            khist[i]  <= khist[i-1];
            kvalid[i] <= kvalid[i-1];
            ktag_i[i] <= ktag_i[i-1];
            ktag_j[i] <= ktag_j[i-1];
          end
          khist[0]  <= p_k;
          kvalid[0] <= p_v;
          ktag_i[0] <= tile_i;
          ktag_j[0] <= tile_j;
          for (int i = WAVE - 1; i > 0; i--) begin
            wfirst[i] <= wfirst[i-1];
            wcap[i]   <= wcap[i-1];
            wbank[i]  <= wbank[i-1];
          end
          wfirst[0] <= p_first;
          wcap[0]   <= p_cap;
          wbank[0]  <= cbank;

          if (p_first && p_cap) begin             // this wave hands off a tile
            bank_busy[cbank] <= 1'b1;
            cbank            <= ~cbank;
          end
          if (p_first) cap_any <= 1'b1;

          if (g_tail) begin
            if (p_first) g_done <= 1'b1;
          end else begin
            if (!g_started) begin
              if (p_first && !p_end) begin g_started <= 1'b1; k_cur <= fst_k; end
            end else if (nxt_v) begin
              k_cur <= nxt_k;
            end
            if ((p_first || g_started) && p_end) begin   // tile fully issued
              g_started <= 1'b0;
              if (tile_last)
                g_tail <= 1'b1;
              else if (tile_j + 1'b1 >= n_tiles_j) begin
                tile_j <= '0;
                tile_i <= tile_i + 1'b1;
              end else
                tile_j <= tile_j + 1'b1;
            end
          end

          // done once the terminator's wave has run and both banks drained
          if (g_done && !bank_busy[0] && !bank_busy[1]) begin
            ctrl_state <= ST_IDLE;
            gemm_mode  <= 1'b0;
            tile_i     <= '0;
            tile_j     <= '0;
          end
        end

        // ------------------------------------------------------------------
        // Every tile computed; wait for the last one to leave the buffer.
        // ------------------------------------------------------------------
        ST_GEMM_FLUSH: begin
          if (!drain_busy && !cap_pend) begin
            ctrl_state <= ST_IDLE;
            gemm_mode  <= 1'b0;
            tile_i     <= '0;
            tile_j     <= '0;
          end
        end
      endcase

      // The finished tile is being captured into u_tbuf now: drain it.
      // After the drain process above, so a capture in the same cycle as the
      // previous tile's final beat wins -- tiles stay back to back.
      if (cap_fire) begin
        drain_busy <= 1'b1;
        drain_row  <= '0;
        drain_col  <= '0;
      end

      // ------------------------------------------------------------------
      // B10 output banks, drained in capture order. A bank becomes drainable
      // when the wavefront reaches index span_r-1, i.e. has finished physical
      // column 0, not when it reaches the far corner: the drain presents
      // column c no earlier than two cycles later plus c, by which time the
      // wavefront (one index per cycle) has finished column c too -- and the
      // narrow drain, one row per cycle, trails it further still. When idle,
      // dbank names the next bank to drain. This overrides the drain process's
      // drain_busy above, which is why it comes last.
      // ------------------------------------------------------------------
      if (pipe) begin
        if (wfirst[span_r - 1 + FEED_LAT] && wcap[span_r - 1 + FEED_LAT])
          bank_full[wbank[span_r - 1 + FEED_LAT]] <= 1'b1;
        if (drain_busy && out_adv && drain_last_beat) begin
          bank_full[dbank] <= 1'b0;
          bank_busy[dbank] <= 1'b0;
          dbank            <= ~dbank;
          if (bank_full[~dbank]) drain_busy <= 1'b1;   // back to back
        end else if (!drain_busy && bank_full[dbank]) begin
          drain_busy <= 1'b1;
          drain_row  <= '0;
          drain_col  <= '0;
        end
      end
    end
  end

  // Operand capacity, checked in simulation only. Resident operands need
  // M*K <= IBUF_DEPTH and K*P <= WBUF_DEPTH; beyond that the load wraps and the
  // result is silently wrong. STREAM replaces the first with the ring's
  // min(M, 2*tile_r)*K, which is what lets M grow without bound.
  // synthesis translate_off
  int chk_tr, chk_ar;
  always @(posedge clk)
    if (rst_n && ctrl_state == ST_IDLE && start && gemm_en) begin
      chk_tr = !RESHAPE ? NR : (tile_shape == 2'd1 && !TALL_ONLY) ? NR / 2
                             : (tile_shape == 2'd2) ? 2 * NR : NR;
      chk_ar = STRM ? ((int'(dim_m) < 2 * chk_tr) ? int'(dim_m) : 2 * chk_tr) : int'(dim_m);
      if (chk_ar * int'(dim_k) > IBUF_DEPTH)
        $display("ERROR versaaccel_top: A needs %0d ibuf entries, IBUF_DEPTH is %0d",
                 chk_ar * int'(dim_k), IBUF_DEPTH);
      if (int'(dim_k) * int'(dim_p) > WBUF_DEPTH)
        $display("ERROR versaaccel_top: B needs %0d wbuf entries, WBUF_DEPTH is %0d",
                 int'(dim_k) * int'(dim_p), WBUF_DEPTH);
      if (int'(dim_k) > K_MAX || int'(dim_p) > P_MAX || (!STRM && int'(dim_m) > M_MAX))
        $display("ERROR versaaccel_top: M/K/P = %0d/%0d/%0d exceed the bitmap bounds %0d/%0d/%0d",
                 dim_m, dim_k, dim_p, M_MAX, K_MAX, P_MAX);
    end
  // synthesis translate_on

  // ==========================================================================
  // Operand feed: F0 address -> F1 read -> F2 distribute (see "Operand feed
  // pipeline" above). The legacy single-shot path goes through the same three
  // stages with its own addresses.
  // ==========================================================================
  // Only entries that have actually been written may drive the array. ibuf is
  // an SRAM-style array with no reset, so unwritten entries hold X. An earlier
  // version read ibuf[0 .. NR-1] unconditionally, so NR-1 X values entered the
  // PE array on every run; in FULL and PARTIAL reduction they reached the adder
  // tree and the DMR dense-GEMM output was undefined. Caught by
  // tb_versaaccel_golden. An address is therefore always paired with a valid bit.
  logic [IA:0] f0_a [NR], f0_ai [NR], f1_a_q [NR], f1_ai_q [NR];   // {valid, index}
  logic [IW:0] f0_b [NC], f0_bi [NC], f1_b_q [NC], f1_bi_q [NC];
  int          lg_a, lg_b;                                         // legacy window
  logic signed [DATA_W-1:0] f2_a_q [NR], f2_ai_q [NR], f2_b_q [NC], f2_bi_q [NC];
  logic signed [DATA_W-1:0] a_fd_q  [NR];        // F2 output: Benes-routed A

  always_comb begin
    for (int i = 0; i < NR; i++) begin
      // Physical row i holds logical row (i mod span_r) of the tile; in WIDE
      // the bottom half repeats the top half's rows for the next NC columns.
      // The legacy path reads a contiguous window bounded by what was written.
      lg_a = int'(ibuf_rd_ptr) + i;
      if (gemm_mode) f0_a[i] = addr_a(int'(gemm_t), i % span_r, i % span_r);
      else           f0_a[i] = {lg_a < int'(ibuf_wr_ptr), lg_a[IA-1:0]};
      // TALL: the right half's A, injected at column NC/2 -- the next NR rows.
      if (shape == SHAPE_TALL) f0_ai[i] = addr_a(int'(gemm_t), i, NR + i);
      else                     f0_ai[i] = '0;
    end
    for (int j = 0; j < NC; j++) begin
      // Physical column j holds logical column (j mod span_c); in TALL the
      // right half repeats the left half's columns for the next NR rows.
      lg_b = int'(wbuf_rd_ptr) + j;
      if (gemm_mode) f0_b[j] = addr_b(int'(gemm_t), j % span_c, j % span_c);
      else           f0_b[j] = {lg_b < WBUF_DEPTH, lg_b[IW-1:0]};
      // WIDE: the bottom half's B, injected at row NR/2 -- the next NC columns.
      if (shape == SHAPE_WIDE) f0_bi[j] = addr_b(int'(gemm_t), j, NC + j);
      else                     f0_bi[j] = '0;
    end
  end

  // F1 reads the buffers; F2 is the Benes.
  always_comb
    for (int i = 0; i < NR; i++)
      dist_in[i] = f2_a_q[i];

  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) begin
      for (int i = 0; i < NR; i++) begin
        f1_a_q[i] <= '0; f1_ai_q[i] <= '0; f2_a_q[i] <= '0; f2_ai_q[i] <= '0;
        a_fd_q[i] <= '0;
      end
      for (int j = 0; j < NC; j++) begin
        f1_b_q[j] <= '0; f1_bi_q[j] <= '0; f2_b_q[j] <= '0; f2_bi_q[j] <= '0;
      end
    end else begin
      for (int i = 0; i < NR; i++) begin
        f1_a_q[i]  <= f0_a[i];
        f1_ai_q[i] <= f0_ai[i];
        f2_a_q[i]  <= f1_a_q[i][IA]  ? ibuf[f1_a_q[i][IA-1:0]]  : '0;
        a_fd_q[i]  <= dist_out[i];
        f2_ai_q[i] <= f1_ai_q[i][IA] ? ibuf[f1_ai_q[i][IA-1:0]] : '0;
      end
      for (int j = 0; j < NC; j++) begin
        f1_b_q[j]  <= f0_b[j];
        f1_bi_q[j] <= f0_bi[j];
        f2_b_q[j]  <= f1_b_q[j][IW]  ? wbuf[f1_b_q[j][IW-1:0]]  : '0;
        f2_bi_q[j] <= f1_bi_q[j][IW] ? wbuf[f1_bi_q[j][IW-1:0]] : '0;
      end
    end

  // The injected half-feeds and B skip the Benes, so F2 is a plain register
  // for them, and they arrive with A.
  logic signed [DATA_W-1:0] f3_ai_q [NR], f3_b_q [NC], f3_bi_q [NC];
  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) begin
      for (int i = 0; i < NR; i++) f3_ai_q[i] <= '0;
      for (int j = 0; j < NC; j++) begin f3_b_q[j] <= '0; f3_bi_q[j] <= '0; end
    end else begin
      for (int i = 0; i < NR; i++) f3_ai_q[i] <= f2_ai_q[i];
      for (int j = 0; j < NC; j++) begin f3_b_q[j] <= f2_b_q[j]; f3_bi_q[j] <= f2_bi_q[j]; end
    end

  always_comb begin
    for (int i = 0; i < NR; i++) begin
      pe_a_in[i]  = a_fd_q[i];
      pe_a_inj[i] = f3_ai_q[i];
    end
    for (int j = 0; j < NC; j++) begin
      pe_b_in[j]  = f3_b_q[j];
      pe_b_inj[j] = f3_bi_q[j];
    end
  end

  // PE-side controls, FEED_LAT cycles behind the controller like the data.
  // Legacy compute runs FEED_LAT extra cycles with the enable already low, so
  // the last operands reach the array before ST_DRAIN reads it.
  logic acc_clear_c, acc_en_c;
  assign acc_clear_c = gemm_mode ? (ctrl_state == ST_GEMM_COMPUTE && gemm_t == 0)
                                 : (compute_cnt == 0);
  assign acc_en_c    = ((ctrl_state == ST_COMPUTE) && (compute_cnt < compute_total))
                    || (ctrl_state == ST_GEMM_COMPUTE) || (ctrl_state == ST_GEMM_PIPE);
  logic [FEED_LAT-1:0] clr_sr, en_sr;
  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) begin clr_sr <= '0; en_sr <= '0; end
    else begin
      clr_sr <= {clr_sr[FEED_LAT-2:0], acc_clear_c};
      en_sr  <= {en_sr[FEED_LAT-2:0],  acc_en_c};
    end
  assign acc_clear = clr_sr[FEED_LAT-1];
  assign acc_en    = en_sr[FEED_LAT-1];

  // Tile capture. The array finishes a tile FEED_LAT cycles after the FSM
  // enters ST_GEMM_CAP and is frozen from then until FEED_LAT cycles after the
  // FSM leaves it. So the capture goes ahead at the FSM's decision (tbuf_cap)
  // if the FSM has already waited FEED_LAT cycles in CAP for the drain -- the
  // drain-bound case, which then loses nothing -- and otherwise as soon as the
  // array has finished: at most FEED_LAT cycles later, always before the next
  // tile's operands arrive.
  logic [$clog2(FEED_LAT+1)-1:0] cap_age, cap_cnt;
  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) begin cap_age <= '0; cap_cnt <= '0; end
    else begin
      if (ctrl_state != ST_GEMM_CAP)  cap_age <= '0;
      else if (cap_age < FEED_LAT)    cap_age <= cap_age + 1'b1;
      if (tbuf_cap && cap_age < FEED_LAT) cap_cnt <= FEED_LAT - cap_age;
      else if (cap_cnt != '0)             cap_cnt <= cap_cnt - 1'b1;
    end
  assign cap_fire = (tbuf_cap && cap_age >= FEED_LAT) || (cap_cnt == 1);
  assign cap_pend = (cap_cnt != '0);       // a capture is still on its way

  assign din_ready = (ctrl_state == ST_LOAD_WEIGHTS) || (ctrl_state == ST_GEMM_LOAD)
                  || (STRM && ld_busy && a_ok);
  // Output register. dout_* used to be combinational from the tile buffer
  // through the column mux, the reduction tree and the row mux straight to the
  // pins. Now a beat moves into this register when it is empty or being
  // accepted (out_adv); the drain counters advance on out_adv instead of on
  // dout_ready. Same valid/ready protocol, same throughput, one cycle later.
  //
  // Wide port: the whole PE column selected by drain_col. Element i of beat c
  // is C[tile_i*NR+i][tile_j*NC+c] for the square tile; reshaped tiles drain
  // the same physical columns, element i of beat c being
  //   WIDE: C[tile_i*NR/2 + i%(NR/2)][tile_j*2NC + (i/(NR/2))*NC + c]
  //   TALL: C[tile_i*2NR + (c/(NC/2))*NR + i][tile_j*NC/2 + c%(NC/2)]
  // A GEMM drains the tile buffer directly: the reduction network was forced
  // to BYPASS for it, which is the identity, so it only sits on the legacy
  // path (FULL / PARTIAL reduction of PE column 0).
  logic                    src_valid;
  logic signed [ACC_W-1:0] src_col [NR];
  logic signed [ACC_W-1:0] dout_q, dvec_q [NR];
  logic                    dvalid_q;
  assign src_valid = (ctrl_state == ST_DRAIN) || (gemm_mode && drain_busy);
  assign out_adv   = !dvalid_q || dout_ready;
  always_comb
    for (int i = 0; i < NR; i++)
      src_col[i] = gemm_mode ? ((AMORTISE && dbank) ? tbuf1[i][drain_col] : tbuf[i][drain_col])
                             : reduce_out[i];
  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) dvalid_q <= 1'b0;
    else if (out_adv) dvalid_q <= src_valid;
  always_ff @(posedge clk)
    if (out_adv) begin
      dout_q <= gemm_mode ? src_col[drain_row] : reduce_out[obuf_rd_ptr[$clog2(NR)-1:0]];
      for (int i = 0; i < NR; i++) dvec_q[i] <= src_col[i];
    end
  assign dout_valid = dvalid_q;
  assign dout_data  = dout_q;
  always_comb
    for (int i = 0; i < NR; i++)
      dout_vec[i] = dvec_q[i];
  assign compute_total = batch_size;

endmodule


// =============================================================================
// Buffer storage modules (B1d)
//
// Storage only: each holds its array in flip-flops and exposes the whole array
// on q, so read addressing and muxing stay in versaaccel_top exactly as they
// were. In a physical design these would be SRAM macros; as submodules their
// area is reported separately and can be excluded or replaced when comparing
// against baselines that have no such buffers. No reset, as before.
// =============================================================================

// ibuf / wbuf: one write port, full-array read
module operand_buffer #(
  parameter int DEPTH = 256,
  parameter int WIDTH = 16
)(
  input  logic                     clk,
  input  logic                     we,
  input  logic [$clog2(DEPTH)-1:0] waddr,
  input  logic signed [WIDTH-1:0]  wdata,
  output logic signed [WIDTH-1:0]  q [DEPTH]
);
  always_ff @(posedge clk)
    if (we) q[waddr] <= wdata;
endmodule

// a_nz / b_nz: nonzero bitmaps -- bulk clear, set one bit per element
module nz_bitmap #(
  parameter int ROWS = 16,
  parameter int COLS = 256
)(
  input  logic                    clk,
  input  logic                    clear,
  input  logic                    clear_row,   // B12: clear one row (ring reuse)
  input  logic                    set,
  input  logic [$clog2(ROWS)-1:0] row,
  input  logic [$clog2(COLS)-1:0] col,
  output logic                    q [ROWS][COLS]
);
  always_ff @(posedge clk)
    if (clear) begin
      for (int r = 0; r < ROWS; r++)
        for (int c = 0; c < COLS; c++) q[r][c] <= 1'b0;
    end else begin
      if (clear_row)
        for (int c = 0; c < COLS; c++) q[row][c] <= 1'b0;
      if (set)
        q[row][col] <= 1'b1;
    end
endmodule

// tbuf: the output double buffer -- captures the whole PE tile at once, or
// (B10) element by element as the clear wavefront passes (cap_pe)
module tile_buffer
  import versaaccel_pkg::*;
(
  input  logic                    clk,
  input  logic                    capture,
  input  logic                    cap_pe [NR][NC],
  input  logic signed [ACC_W-1:0] d [NR][NC],
  output logic signed [ACC_W-1:0] q [NR][NC]
);
  always_ff @(posedge clk)
    for (int r = 0; r < NR; r++)
      for (int c = 0; c < NC; c++)
        if (capture || cap_pe[r][c]) q[r][c] <= d[r][c];
endmodule
