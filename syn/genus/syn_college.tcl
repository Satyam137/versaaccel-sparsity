# ============================================================
# syn_college.tcl - one Genus script for every run (B1a-B1g, B15, B10, B12)
# Target: gsclib045 (45nm), the college flow of syn_versaaccel.tcl,
# with the Run-1 problems fixed:
#   - slow corner, not fast_vdd1v2                         (B1b)
#   - power from simulated activity (VCD), not vectorless  (B1c)
#   - RTL design variants via defines                      (B15, B10, B12)
#   - every run writes to its own folder, nothing is overwritten
#
# Interactive (csh -> source cshrc_new -> genus):
#     set DESIGN versaaccel_top ; set VARIANT default ; set CLK_PERIOD 2.5
#     source syn_college.tcl
# Batch: see run_college.sh, which sets the same names as environment variables.
#
#   DESIGN      versaaccel_top | systolic_top | sigma_top
#   VARIANT     default | reshape | reshape_tall | amortise | amortise_stream
#               (versaaccel_top only)
#   CLK_PERIOD  ns, default 2.5
#   OUTDIR      results folder, default OUTPUT_r20 (the run after the RTL was
#               pipelined; the first batch's OUTPUT/ is left alone)
#   CORNER      slow_vdd1v0 (default) | slow_vdd1v2 | fast_vdd1v2 ...
#   POWER       1 = also report power from the two VCDs (default builds only)
#   VCDSET      full (whole testbench) | compute (compute phase only, from
#               experiments/power_windows.py -- the comparable one)
# ============================================================

# ---- the technology library: EDIT if it moves ----
if {[info exists ::env(PDK_DIR)]} {
  set PDK_DIR $::env(PDK_DIR)      ;# folder that holds gsclib045_all_v4.8
} else {
  set PDK_DIR /path/to/pdk           ;# EDIT, or export PDK_DIR before running
}
# ---- the RTL and activity files are found relative to this folder (Genus
# runs in syn/genus -- run_college.sh cds there), so the project can be
# unzipped anywhere ----
set GENUS_DIR [file normalize [pwd]]
set rtlDir    [file normalize $GENUS_DIR/../../idea2_veriflex/rtl]
set vcdDir    $GENUS_DIR/activity
if {![file exists $rtlDir/versaaccel_rtl/versaaccel_top.sv]} {
  error "RTL not found at $rtlDir -- run Genus from <project>/syn/genus"
}

# ---- run parameters: Tcl variable, else environment, else default ----
foreach {name dflt} {DESIGN versaaccel_top VARIANT default CLK_PERIOD 2.5
                     CORNER slow_vdd1v0 POWER 0 OUTDIR OUTPUT_r20 VCDSET full} {
  if {![info exists $name]} {
    if {[info exists ::env($name)]} { set $name $::env($name) } else { set $name $dflt }
  }
}

switch $VARIANT {
  default         { set defs {SYNTHESIS} }
  reshape         { set defs {SYNTHESIS VERSAACCEL_RESHAPE} }
  reshape_tall    { set defs {SYNTHESIS VERSAACCEL_RESHAPE_TALL} }
  amortise        { set defs {SYNTHESIS VERSAACCEL_AMORTISE} }
  amortise_stream { set defs {SYNTHESIS VERSAACCEL_AMORTISE VERSAACCEL_STREAM} }
  default { error "VARIANT must be default | reshape | reshape_tall | amortise | amortise_stream" }
}
if {$DESIGN ne "versaaccel_top" && $VARIANT ne "default"} {
  error "variants exist only for versaaccel_top"
}

set TAG ${DESIGN}_${VARIANT}_${CORNER}_${CLK_PERIOD}ns
set OUT $OUTDIR/$TAG
file mkdir $OUT
puts "======================================="
puts " RUN $TAG   defines: $defs"
puts " Hostname : [info hostname]"
puts "======================================="

# ============================================================
# LIBRARIES -- same corner for all designs
# ============================================================
set libDir [list \
  ${PDK_DIR}/gsclib045_all_v4.8/gsclib045/timing     \
  ${PDK_DIR}/gsclib045_all_v4.8/gsclib045_hvt/timing \
  ${PDK_DIR}/gsclib045_all_v4.8/gsclib045_lvt/timing ]
set libList [list ${CORNER}_basicCells.lib ${CORNER}_multibitsDFF.lib \
                  ${CORNER}_basicCells_hvt.lib ${CORNER}_basicCells_lvt.lib]
foreach l $libList {
  set found 0
  foreach d $libDir { if {[file exists $d/$l]} { set found 1 } }
  if {!$found} {
    puts "ERROR: $l not found. Corners available:"
    foreach d $libDir { puts "  $d: [lsort [glob -nocomplain -tails -directory $d *basicCells*.lib]]" }
    error "set CORNER to one of the corners listed above (slow_* preferred)"
  }
}

set_db init_lib_search_path $libDir
set_db init_hdl_search_path [list ${rtlDir}/common ${rtlDir}/versaaccel_rtl \
                                  ${rtlDir}/sigma_baseline ${rtlDir}/systolic_baseline]
set_db max_cpus_per_server 8
set_db information_level 9
set_db tns_opto true
read_libs $libList
read_physical -lefs [list \
  ${PDK_DIR}/gsclib045_all_v4.8/gsclib045/lef/gsclib045_tech.lef          \
  ${PDK_DIR}/gsclib045_all_v4.8/gsclib045/lef/gsclib045_macro.lef         \
  ${PDK_DIR}/gsclib045_all_v4.8/gsclib045/lef/gsclib045_multibitsDFF.lef  \
  ${PDK_DIR}/gsclib045_all_v4.8/gsclib045_hvt/lef/gsclib045_hvt_macro.lef \
  ${PDK_DIR}/gsclib045_all_v4.8/gsclib045_lvt/lef/gsclib045_lvt_macro.lef ]
set_db qrc_tech_file ${PDK_DIR}/gsclib045_all_v4.8/gsclib045_tech/qrc/qx/gpdk045.tch

# ============================================================
# RTL -- the defines go on EVERY file, package included
# ============================================================
switch $DESIGN {
  versaaccel_top {
    set rtlList [list \
      ${rtlDir}/common/versaaccel_pkg.sv                  \
      ${rtlDir}/versaaccel_rtl/benes_switch.sv            \
      ${rtlDir}/versaaccel_rtl/distribution_network.sv    \
      ${rtlDir}/versaaccel_rtl/reduction_network.sv       \
      ${rtlDir}/versaaccel_rtl/processing_element.sv      \
      ${rtlDir}/versaaccel_rtl/pe_array.sv                \
      ${rtlDir}/versaaccel_rtl/config_prediction_engine.sv \
      ${rtlDir}/versaaccel_rtl/phase_fsm.sv               \
      ${rtlDir}/versaaccel_rtl/speculation_unit.sv        \
      ${rtlDir}/versaaccel_rtl/nm_routing_rom.sv          \
      ${rtlDir}/versaaccel_rtl/versaaccel_top.sv          ]
    set vcdInst tb_regroup/dut
  }
  systolic_top {
    set rtlList [list ${rtlDir}/common/versaaccel_pkg.sv \
                      ${rtlDir}/systolic_baseline/systolic_top.sv]
    set vcdInst tb_systolic_golden/dut
  }
  sigma_top {
    set rtlList [list \
      ${rtlDir}/common/versaaccel_pkg.sv               \
      ${rtlDir}/versaaccel_rtl/benes_switch.sv         \
      ${rtlDir}/versaaccel_rtl/distribution_network.sv \
      ${rtlDir}/versaaccel_rtl/reduction_network.sv    \
      ${rtlDir}/sigma_baseline/sigma_top.sv            ]
    set vcdInst tb_sigma_golden/dut
  }
  default { error "DESIGN must be versaaccel_top | systolic_top | sigma_top" }
}
set defFlags {}
foreach d $defs { lappend defFlags -define $d }
read_hdl -language sv {*}$defFlags $rtlList
elaborate $DESIGN
set_top_module $DESIGN

# ============================================================
# CONSTRAINTS -- identical for every design (as Run 1)
# ============================================================
set CLK_NAME clk
create_clock -name $CLK_NAME -period $CLK_PERIOD [get_ports clk]
set_clock_uncertainty [expr {0.05 * $CLK_PERIOD}] [get_clocks $CLK_NAME]
set_clock_transition  [expr {0.02 * $CLK_PERIOD}] [get_clocks $CLK_NAME]
set_false_path -from [get_ports rst_n]
set IO_DELAY [expr {0.20 * $CLK_PERIOD}]
set ALL_IN [remove_from_collection [all_inputs] [get_ports "clk rst_n"]]
set_input_delay  -clock $CLK_NAME $IO_DELAY $ALL_IN
set_output_delay -clock $CLK_NAME $IO_DELAY [all_outputs]
set_driving_cell -lib_cell INVX4 $ALL_IN
set_load [expr {4 * [load_of [get_lib_pins */INVX1/A]]}] [all_outputs]
set_max_transition [expr {0.15 * $CLK_PERIOD}] [current_design]
check_design -unresolved > $OUT/check_design.rpt

# ============================================================
# SYNTHESISE (hierarchy kept: per-block area is the point)
# ============================================================
set_db auto_ungroup none
set_db syn_generic_effort medium
set_db syn_map_effort     medium
set_db syn_opt_effort     medium
syn_generic
syn_map
syn_opt

# ============================================================
# REPORTS
# ============================================================
report_area                 > $OUT/area.rpt
report_area -depth 3        > $OUT/area_hier.rpt
report_gates -power         > $OUT/gates.rpt
report_timing -max_paths 20 > $OUT/timing.rpt
report_qor                  > $OUT/qor.rpt
report_power                > $OUT/power_vectorless.rpt
# report_summary needs time_info checkpoints this flow does not record; it aborted
# every run of the first batch here, so it is skipped rather than allowed to stop the run.
catch { report_summary -directory $OUT/summary }

if {$DESIGN eq "versaaccel_top"} {
  # B2: paths through the reconfiguration logic (speculation unit, CPE, Benes control)
  foreach {label pat} {spec *spec/* cpe *cpe/* dist *dist_net/*} {
    if {[catch {report_timing -through [get_db hpins $pat] -max_paths 10 > $OUT/timing_$label.rpt} msg]} {
      puts "NOTE: no timing through $pat ($msg)"
    }
  }
}

# B1c: power from real operand activity -- default builds only
# (the VCDs were simulated on the default RTL)
if {$POWER && $VARIANT eq "default"} {
  set short [string map {_top ""} $DESIGN]
  foreach w {dense pruned90} {
    set sfx [expr {$VCDSET eq "compute" ? "_compute" : ""}]
    set vcd $vcdDir/${short}_${w}${sfx}.vcd
    if {![file exists $vcd]} { puts "NOTE: $vcd missing -- power skipped"; continue }
    if {[catch {read_stimulus -file $vcd -dut_instance $vcdInst -format vcd} m1]} {
      if {[catch {read_vcd -vcd_scope $vcdInst $vcd} m2]} {
        puts "ERROR: could not read $vcd\n  read_stimulus: $m1\n  read_vcd: $m2"
        continue
      }
    }
    catch { report_sdb_annotation > $OUT/annotation_$w$sfx.rpt }
    report_power                          > $OUT/power_$w$sfx.rpt
    report_power -by_hierarchy -levels 2  > $OUT/power_hier_$w$sfx.rpt
  }
}

catch { write_hdl > $OUT/${DESIGN}_netlist.v }
catch { write_sdc > $OUT/${DESIGN}.sdc }
catch { write_db -to_file $OUT/${TAG}.db }

if {[catch {set slack [get_db [get_db timing_paths -max 1] .slack]}]} { set slack "see_timing.rpt" }
if {[catch {set area  [get_db [current_design] .area]}]} { set area "see_area.rpt" }
set fh [open $OUTDIR/SUMMARY.txt a]
puts $fh "$TAG area=$area slack=$slack"
close $fh
puts "SUMMARY $TAG area=$area slack=$slack"

# Marker the runner checks: the run reached the end.
close [open $OUT/DONE w]
