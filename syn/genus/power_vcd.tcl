# =============================================================================
# power_vcd.tcl -- power from simulated switching activity (B1c)
#
# Source this in the Genus session AFTER syn_opt, instead of relying on the
# default toggle rate (which made run 1's power an order-of-magnitude figure):
#
#   set ACTIVITY_VCD  <repo>/syn/genus/activity/versaaccel_pruned90.vcd
#   set ACTIVITY_INST tb_regroup/dut
#   source <repo>/syn/genus/power_vcd.tcl
#
# Generate the VCDs first, on any machine with iverilog:
#   python experiments/make_activity.py
#
#   design          VCD                                   ACTIVITY_INST
#   versaaccel_top  activity/versaaccel_{dense,pruned90}.vcd  tb_regroup/dut
#   systolic_top    activity/systolic_{dense,pruned90}.vcd    tb_systolic_golden/dut
#   sigma_top       activity/sigma_{dense,pruned90}.vcd       tb_sigma_golden/dut
#
# All six are real operands from the same layer (ResNet-50 layer3.2.conv1),
# simulated with -DSYNTHESIS -- the same CPE view Genus compiles. Register
# arrays are dumped element by element with RTL names (e.g. accum[3][7]).
#
# CHECK (first use): the annotation report. If only a small fraction of the
# sequential cells is annotated, the RTL->netlist name mapping missed the
# register arrays; the power figure is then still mostly default activity.
# =============================================================================

if {![info exists RUN]} { set RUN [pwd] }
set tag [file rootname [file tail $ACTIVITY_VCD]]

read_stimulus -file $ACTIVITY_VCD -dut_instance $ACTIVITY_INST -format vcd

# CHECK: command name varies by Genus release; report_sdb_annotation is the
# Genus/Joules name. If absent, `report_power -verbose` shows the asserted
# vs propagated activity split.
catch { report_sdb_annotation > $RUN/annotation_${tag}.rpt }

report_power            > $RUN/power_${tag}.rpt
report_power -by_hierarchy -levels 2 > $RUN/power_hier_${tag}.rpt
puts "== power from $ACTIVITY_VCD written to $RUN/power_${tag}.rpt =="
