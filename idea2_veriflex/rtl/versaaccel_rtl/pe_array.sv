// =============================================================================
// pe_array.sv — 8×8 configurable PE array
//
// Instantiates NR×NC processing elements with configurable interconnect.
// In DMR mode: systolic data flow (a flows right, b flows down)
// In D-M-R mode: each PE operates independently via Benes routing
// =============================================================================
module pe_array
  import versaaccel_pkg::*;
#(
  parameter int ROWS = NR,
  parameter int COLS = NC
)(
  input  logic        clk,
  input  logic        rst_n,

  // Configuration
  input  config_t     config_mode,
  input  logic        acc_clear,
  input  logic        acc_en,

  // Data inputs (from distribution network)
  input  logic signed [DATA_W-1:0] a_in [ROWS],  // Row activations
  input  logic signed [DATA_W-1:0] b_in [COLS],  // Column weights

  // Reshaping (B15). split_a cuts every row at column COLS/2 and feeds the
  // right half from a_inj instead of the left half's forwarded a; split_b cuts
  // every column at row ROWS/2 and feeds the bottom half from b_inj. Tied low
  // when versaaccel_pkg::RESHAPE is 0, so the muxes fold away.
  input  logic                     split_a,
  input  logic                     split_b,
  input  logic signed [DATA_W-1:0] a_inj [ROWS],
  input  logic signed [DATA_W-1:0] b_inj [COLS],

  // Per-PE accumulator clear (B10). With fill amortisation each PE starts its
  // next tile on its own cycle -- a diagonal wavefront -- so the clear is per
  // PE and ORed with the global one. Tied low without VERSAACCEL_AMORTISE.
  input  logic                     acc_clear_pe [ROWS][COLS],

  // Outputs (to reduction network)
  output logic signed [ACC_W-1:0]  pe_out [ROWS][COLS]
);

  // PE mode derived from config
  pe_mode_t pe_mode;

  always_comb begin
    case (config_mode)
      CFG_DMR:      pe_mode = PE_MAC;   // All PEs do MAC
      CFG_D_MR:     pe_mode = PE_MAC;   // MAC with fused reduce
      CFG_DM_R:     pe_mode = PE_MAC;   // MAC with fused distribute
      CFG_D_M_R:    pe_mode = PE_MAC;   // Independent MAC
      CFG_D_M_R_NM: pe_mode = PE_MAC;   // Independent MAC with NM routing
      default:      pe_mode = PE_IDLE;
    endcase
  end

  // Inter-PE wires for systolic forwarding
  logic signed [DATA_W-1:0] a_wire [ROWS][COLS+1];
  logic signed [DATA_W-1:0] b_wire [ROWS+1][COLS];

  // Connect inputs
  genvar r, c;
  generate
    for (r = 0; r < ROWS; r++) begin : gen_a_in
      assign a_wire[r][0] = a_in[r];
    end
    for (c = 0; c < COLS; c++) begin : gen_b_in
      assign b_wire[0][c] = b_in[c];
    end
  endgenerate

  // What each PE actually receives: the forwarded operand, except at the two
  // split boundaries, where a reshaped tile injects a fresh feed.
  logic signed [DATA_W-1:0] a_pe [ROWS][COLS];
  logic signed [DATA_W-1:0] b_pe [ROWS][COLS];
  generate
    for (r = 0; r < ROWS; r++) begin : gen_split_r
      for (c = 0; c < COLS; c++) begin : gen_split_c
        if (c == COLS / 2) begin : g_a_cut
          assign a_pe[r][c] = split_a ? a_inj[r] : a_wire[r][c];
        end else begin : g_a_thru
          assign a_pe[r][c] = a_wire[r][c];
        end
        if (r == ROWS / 2) begin : g_b_cut
          assign b_pe[r][c] = split_b ? b_inj[c] : b_wire[r][c];
        end else begin : g_b_thru
          assign b_pe[r][c] = b_wire[r][c];
        end
      end
    end
  endgenerate

  // Instantiate PE grid
  generate
    for (r = 0; r < ROWS; r++) begin : gen_pe_row
      for (c = 0; c < COLS; c++) begin : gen_pe_col
        processing_element #(
          .DATA_WIDTH (DATA_W),
          .ACC_WIDTH  (ACC_W)
        ) pe_inst (
          .clk       (clk),
          .rst_n     (rst_n),
          .mode      (pe_mode),
          .acc_clear (acc_clear | acc_clear_pe[r][c]),
          .acc_en    (acc_en),
          .a_in      (a_pe[r][c]),
          .b_in      (b_pe[r][c]),
          .result    (pe_out[r][c]),
          .a_out     (a_wire[r][c+1]),
          .b_out     (b_wire[r+1][c])
        );
      end
    end
  endgenerate

endmodule
