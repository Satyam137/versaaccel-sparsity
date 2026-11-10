// =============================================================================
// versaaccel_pkg.sv — Shared types, parameters, and enums for VersaAccel
// =============================================================================
`ifndef VERSAACCEL_PKG_SV
`define VERSAACCEL_PKG_SV

package versaaccel_pkg;

  // ---------- Array dimensions ----------
  parameter int NR           = 16;    // PE rows
  parameter int NC           = 16;    // PE columns
  parameter int NUM_PES      = NR * NC;

  // ---------- Reshapeable tile (B15, PROJECT_REPORT section 2.9) ----------
  // With VERSAACCEL_RESHAPE defined, the GEMM path can run the NR x NC array
  // as a (NR/2) x (2*NC) "wide" tile or a (2*NR) x (NC/2) "tall" tile, chosen
  // per GEMM on the tile_shape pin. Without it the pin is ignored and the
  // design is the square array, unchanged -- synthesising both builds gives
  // the hardware cost of reshaping.
  //
  // VERSAACCEL_RESHAPE_TALL (Rev 21) builds square + TALL only. Genus run 3
  // put the three-shape array at +57% area and 0.28 ns over a 2.5 ns clock;
  // almost all of its sparsity gain is TALL's (pruned weights, section 2.9), and
  // WIDE's own gain on dense ResNet is dimension fit, which any reshapeable
  // array gets. Without WIDE there is no second B feed, no midline cut of the
  // columns, and the A bitmap keeps whole-tile-edge granules.
`ifdef VERSAACCEL_RESHAPE_TALL
  parameter bit RESHAPE      = 1'b1;
  parameter bit TALL_ONLY    = 1'b1;
`elsif VERSAACCEL_RESHAPE
  parameter bit RESHAPE      = 1'b1;
  parameter bit TALL_ONLY    = 1'b0;
`else
  parameter bit RESHAPE      = 1'b0;
  parameter bit TALL_ONLY    = 1'b0;
`endif
  // ---------- Fill amortisation (B10) and streamed activations (B12) ----------
  // VERSAACCEL_AMORTISE pipelines consecutive GEMM tiles through the array:
  // the next tile's operands enter on the cycle after the last step of the
  // current one, each PE clears and hands its result to a two-bank output
  // buffer on a diagonal wavefront, and the NR+NC-2 skew fill is paid once per
  // GEMM instead of once per tile. VERSAACCEL_STREAM (needs AMORTISE) also
  // overlaps the activation load with compute: A streams through a double
  // buffer of two tile-rows while B stays resident. Neither defined: the
  // design is unchanged, cycle for cycle.
`ifdef VERSAACCEL_AMORTISE
  parameter bit AMORTISE     = 1'b1;
`else
  parameter bit AMORTISE     = 1'b0;
`endif
`ifdef VERSAACCEL_STREAM
  parameter bit STREAM       = 1'b1;
`else
  parameter bit STREAM       = 1'b0;
`endif

  typedef enum logic [1:0] {
    SHAPE_SQUARE = 2'd0,     // NR x NC
    SHAPE_WIDE   = 2'd1,     // NR/2 x 2NC: narrow row edge -- for sparse A (ReLU)
    SHAPE_TALL   = 2'd2      // 2NR x NC/2: narrow column edge -- for sparse B (pruned)
  } tile_shape_t;

  // ---------- Data widths ----------
  parameter int DATA_W       = 16;   // INT16 activations/weights
  parameter int ACC_W        = 32;   // Accumulator width
  parameter int ADDR_W       = 16;   // Buffer address width
  parameter int SPARSITY_W   = 8;    // Sparsity estimate (0-255 = 0%-100%)

  // ---------- Benes network ----------
  parameter int BENES_N      = NR;                        // Network size
  parameter int BENES_STAGES = 2 * $clog2(BENES_N) - 1;  // 7 at NR=16 (5 at N=8)
  parameter int BENES_SW_PER_STAGE = BENES_N / 2;         // 8 at NR=16
  parameter int BENES_CTRL_W = BENES_STAGES * BENES_SW_PER_STAGE; // 56 bits at NR=16

  // ---------- Configuration ----------
  typedef enum logic [2:0] {
    CFG_DMR     = 3'd0,  // Dense systolic
    CFG_D_MR    = 3'd1,  // Distribution separate, MR fused
    CFG_DM_R    = 3'd2,  // DM fused, Reduction separate
    CFG_D_M_R   = 3'd3,  // All separate (Gustavson)
    CFG_D_M_R_NM = 3'd4  // D-M-R with N:M precomputed routing
  } config_t;

  // ---------- Phase FSM ----------
  typedef enum logic [2:0] {
    PHASE_PREFILL   = 3'd0,
    PHASE_DECODE    = 3'd1,
    PHASE_ATTENTION = 3'd2,
    PHASE_AGGREGATE = 3'd3,
    PHASE_TRANSFORM = 3'd4,
    PHASE_CONV      = 3'd5,
    PHASE_FC        = 3'd6,
    PHASE_GENERIC   = 3'd7
  } phase_t;

  // ---------- Speculation ----------
  typedef enum logic [1:0] {
    SPEC_COMMIT   = 2'd0,  // High confidence: use prediction
    SPEC_SPECULATE = 2'd1, // Medium: start + verify
    SPEC_FALLBACK  = 2'd2  // Low: use safe default
  } spec_mode_t;

  // ---------- PE operating mode ----------
  typedef enum logic [1:0] {
    PE_MAC       = 2'd0,  // Multiply-accumulate
    PE_PASS      = 2'd1,  // Pass data through (systolic forwarding)
    PE_ACCUM     = 2'd2,  // Accumulate only (reduction)
    PE_IDLE      = 2'd3   // Power-gated / idle
  } pe_mode_t;

  // ---------- N:M patterns ----------
  // 2:4 has C(4,2)=6 possible nonzero patterns
  parameter int NM_PATTERNS = 6;

  // ---------- Buffer sizes ----------
  parameter int IBUF_DEPTH   = 256;  // Input buffer entries
  parameter int OBUF_DEPTH   = 256;  // Output buffer entries
  parameter int WBUF_DEPTH   = 512;  // Weight buffer entries

  // Largest K, M and P the zero-skipping bitmaps and encoders are sized for.
  // The defaults are the most the buffers can hold; a build with larger
  // buffers but known GEMM sizes (experiments/amortise_rtl.py) can set them
  // lower, since the bitmaps grow as K*M and K*P.
  parameter int K_MAX        = IBUF_DEPTH;
  parameter int M_MAX        = IBUF_DEPTH;
  parameter int P_MAX        = WBUF_DEPTH;

endpackage

`endif
