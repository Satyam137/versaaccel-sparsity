# =============================================================================
# setup.tcl -- the ONLY file you should need to edit.
#
# Everything below this line is site-specific: library paths differ between
# installs, and nothing else in this flow hardcodes them.
# =============================================================================

# --- Technology library ------------------------------------------------------
# Point this at your 45nm liberty files. On most university installs the Nangate
# OpenCell 45nm library ships with the Cadence tools; find it with
#     find / -name "NangateOpenCellLibrary*typical*.lib" 2>/dev/null
# Set BOTH a typical corner (for area/power) and, if you have them, slow/fast.
set LIB_PATH     "/path/to/nangate45"
set LIB_TYPICAL  "$LIB_PATH/NangateOpenCellLibrary_typical.lib"
set LIB_SLOW     ""   ;# optional: worst-case corner for signoff timing
set LIB_FAST     ""   ;# optional: best-case corner for hold

# --- Target clock ------------------------------------------------------------
# 1.0 ns = 1 GHz. run.sh overrides this when sweeping for Fmax.
set CLK_PERIOD_NS 1.0
set CLK_NAME      clk

# --- Effort ------------------------------------------------------------------
# medium is enough for a comparison; high costs runtime and changes all three
# designs the same way, so it does not affect the conclusions.
set SYN_EFFORT    medium

# --- Output ------------------------------------------------------------------
set OUT_DIR "[pwd]/results"
