# Vivado script to generate post-implementation reports only (no rebuild/re-implementation)
set project_name "vim_project"
set project_dir "./vivado_project"
set xpr_file [file join $project_dir "${project_name}.xpr"]

# Open existing project
if { [file exists $xpr_file] } {
    open_project $xpr_file
} else {
    puts "ERROR: Project file $xpr_file not found. Run run_vivado_flow.py first."
    exit 1
}

# Require an existing implementation result
set run_status [get_property STATUS [get_runs impl_1]]
if { $run_status != "write_bitstream Complete!" } {
    puts "ERROR: impl_1 is not complete (status: $run_status)."
    puts "Please run run_vivado_implementation.py once before reports-only mode."
    exit 1
}

set reports_dir "./vivado_reports/post_impl"
file mkdir $reports_dir
foreach rpt [glob -nocomplain -directory $reports_dir "*.rpt"] {
    file delete -force $rpt
}

puts "Generating post-implementation utilization/power reports in $reports_dir ..."
open_run impl_1
report_utilization -file [file join $reports_dir "utilization_post_impl.rpt"]
report_utilization -hierarchical -file [file join $reports_dir "utilization_hier_post_impl.rpt"]
report_power -file [file join $reports_dir "power_post_impl.rpt"]

puts "Reports generated successfully."
puts "  - [file join $reports_dir utilization_post_impl.rpt]"
puts "  - [file join $reports_dir utilization_hier_post_impl.rpt]"
puts "  - [file join $reports_dir power_post_impl.rpt]"

close_project
