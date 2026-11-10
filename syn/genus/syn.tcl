# =============================================================================
# syn.tcl -- Genus synthesis for one design.
#
#   genus -batch -f syn.tcl -log <logdir>   with DESIGN set in the environment:
#       DESIGN=versaaccel_top   genus -batch -f syn.tcl
#       DESIGN=systolic_top     genus -batch -f syn.tcl
#       DESIGN=sigma_top        genus -batch -f syn.tcl
#
# Normally driven by run.sh, which does all three plus the Fmax sweep.
#
# NOTE: this flow has never been executed -- there is no Genus licence on the
# machine it was written on. It follows the standard Genus sequence and should
# run, but expect to fix one or two library-specific lines on first use. The
# places most likely to need attention are marked CHECK.
# =============================================================================

source [file dirname [info script]]/setup.tcl

set DESIGN $env(DESIGN)
if {[info exists env(CLK_PERIOD_NS)]} { set CLK_PERIOD_NS $env(CLK_PERIOD_NS) }

set RTL  [file normalize [file dirname [info script]]/../../idea2_veriflex/rtl]
set RUN  "$OUT_DIR/${DESIGN}_${CLK_PERIOD_NS}ns"
file mkdir $RUN

# -----------------------------------------------------------------------------
# Library
# -----------------------------------------------------------------------------
set_db init_lib_search_path $LIB_PATH
set libs $LIB_TYPICAL
if {$LIB_SLOW ne ""} { lappend libs $LIB_SLOW }
if {$LIB_FAST ne ""} { lappend libs $LIB_FAST }
set_db library $libs

set_db information_level 7
set_db syn_generic_effort $SYN_EFFORT
set_db syn_map_effort     $SYN_EFFORT
set_db syn_opt_effort     $SYN_EFFORT

# Keep the hierarchy. The whole point of synthesising this is to find out what
# the double buffer, the nonzero bitmaps and the CPE actually cost, and a
# flattened netlist cannot answer that.
set_db auto_ungroup none

# -----------------------------------------------------------------------------
# RTL
# -----------------------------------------------------------------------------
# SYNTHESIS selects the CPE's constant-ROM weight view instead of $readmemh.
# Without it Genus drops the initial block, nothing writes the weight arrays,
# constant propagation deletes the entire CPE, and its area reports as ~0.
set_db hdl_search_path "$RTL/common $RTL/versaaccel_rtl $RTL/sigma_baseline $RTL/systolic_baseline"
set_db hdl_vhdl_read_version 2008
set_db hdl_preprocessor_directives {SYNTHESIS}

read_hdl -language sv "$RTL/common/versaaccel_pkg.sv"

switch $DESIGN {
  versaaccel_top {
    read_hdl -language sv [glob "$RTL/versaaccel_rtl/*.sv"]
  }
  systolic_top {
    read_hdl -language sv "$RTL/systolic_baseline/systolic_top.sv"
  }
  sigma_top {
    # SIGMA instantiates the shared Benes and reduction networks.
    read_hdl -language sv [list \
      "$RTL/versaaccel_rtl/benes_switch.sv" \
      "$RTL/versaaccel_rtl/distribution_network.sv" \
      "$RTL/versaaccel_rtl/reduction_network.sv" \
      "$RTL/sigma_baseline/sigma_top.sv"]
  }
  default { error "DESIGN must be versaaccel_top | systolic_top | sigma_top" }
}

elaborate $DESIGN
puts "\n== elaborated $DESIGN =="

# CHECK: unresolved references here mean a missing file above, not a bug.
check_design -unresolved

# -----------------------------------------------------------------------------
# Constraints -- the same file for every design
# -----------------------------------------------------------------------------
read_sdc [file dirname [info script]]/constraints.sdc

# -----------------------------------------------------------------------------
# Synthesise
# -----------------------------------------------------------------------------
syn_generic
syn_map
syn_opt

# -----------------------------------------------------------------------------
# Reports
# -----------------------------------------------------------------------------
report_area                     > $RUN/area.rpt
report_area  -depth 3           > $RUN/area_hier.rpt
report_gates                    > $RUN/gates.rpt
report_timing  -max_paths 20    > $RUN/timing.rpt
report_timing  -lint            > $RUN/timing_lint.rpt
report_power                    > $RUN/power.rpt
report_clocks                   > $RUN/clocks.rpt
report_qor                      > $RUN/qor.rpt

# Per-structure area. These are the numbers PROJECT_REPORT.md section 7.10
# lists as unquantified, so pull them out by name rather than leaving them
# buried in the hierarchy report.
if {$DESIGN eq "versaaccel_top"} {
  # CHECK: instance paths depend on how Genus names generate blocks.
  # If a pattern matches nothing, find the real name in area_hier.rpt.
  foreach {label pattern} {
    pe_array      "pe_arr"
    distribution  "dist_net"
    reduction     "red_net"
    cpe           "cpe"
    speculation   "spec"
    phase_fsm     "phase"
  } {
    set insts [get_db insts -if {.name == "*$pattern*"}]
    if {[llength $insts]} {
      report_area $insts > $RUN/area_${label}.rpt
    } else {
      puts "WARNING: no instance matching '$pattern' -- check area_hier.rpt"
    }
  }
  # Register counts for the two structures added in Revisions 6-7. These are
  # flops, not instances, so count them from the netlist.
  set tbuf_regs [get_db insts -if {.name == "*tbuf*"}]
  set anz_regs  [get_db insts -if {.name == "*a_nz*"}]
  set bnz_regs  [get_db insts -if {.name == "*b_nz*"}]
  set fh [open $RUN/added_structures.rpt w]
  puts $fh "Structures added in Revisions 6-7 (PROJECT_REPORT.md 7.6, 7.7)"
  puts $fh "  output double buffer (tbuf) : [llength $tbuf_regs] cells"
  puts $fh "  nonzero bitmap A    (a_nz)  : [llength $anz_regs] cells"
  puts $fh "  nonzero bitmap B    (b_nz)  : [llength $bnz_regs] cells"
  close $fh
}

# -----------------------------------------------------------------------------
# Netlist + SDC out, so this can be handed to place-and-route later
# -----------------------------------------------------------------------------
write_hdl                 > $RUN/${DESIGN}_netlist.v
write_sdc                 > $RUN/${DESIGN}.sdc
write_db                    $RUN/${DESIGN}.db

# One-line summary that run.sh greps to build the comparison table.
set slack [get_db [get_db timing_paths -max 1] .slack]
set area  [get_db [current_design] .area]
puts "SUMMARY $DESIGN period=$CLK_PERIOD_NS area=$area slack=$slack"

exit
