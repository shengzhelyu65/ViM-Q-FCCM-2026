# Vivado script to run implementation and generate bitstream
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

# Create HDL wrapper for the Block Design
set bd_file [get_files design_1.bd]
if { $bd_file == "" } {
    puts "ERROR: Block design design_1.bd not found in project."
    exit 1
}

# Generate output products
puts "Generating output products for design_1..."
generate_target all [get_files design_1.bd]

# Create wrapper
set wrapper_file [make_wrapper -files [get_files design_1.bd] -top]
add_files -norecurse $wrapper_file
update_compile_order -fileset sources_1
set_property top design_1_wrapper [current_fileset]

# Run Synthesis and Implementation
puts "Starting implementation flow (this may take a while)..."
set_property strategy Performance_Explore [get_runs impl_1]
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1

# Check status
set run_status [get_property STATUS [get_runs impl_1]]
if { $run_status != "write_bitstream Complete!" } {
    puts "ERROR: Implementation failed with status: $run_status"
    # Check for errors in the log
    exit 1
}

puts "Implementation successful! Bitstream generated."

# Generate post-implementation reports (utilization + power)
set reports_dir "./vivado_reports/post_impl"
file mkdir $reports_dir
foreach rpt [glob -nocomplain -directory $reports_dir "*.rpt"] {
    file delete -force $rpt
}

puts "Generating post-implementation reports in $reports_dir ..."
open_run impl_1

# Utilization + power
report_utilization -file [file join $reports_dir "utilization_post_impl.rpt"]
report_utilization -hierarchical -file [file join $reports_dir "utilization_hier_post_impl.rpt"]
report_power -file [file join $reports_dir "power_post_impl.rpt"]

puts "Post-implementation reports generated in $reports_dir"

# Export Hardware (XSA)
puts "Exporting hardware platform (XSA)..."
write_hw_platform -fixed -include_bit -force -file ./vim_accelerator.xsa

puts "Success: XSA exported to ./vim_accelerator.xsa"
close_project
