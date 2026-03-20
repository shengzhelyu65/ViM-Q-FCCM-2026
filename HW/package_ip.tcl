# Vivado script to package the ViM_ACCELERATOR IP
set ip_name "ViM_ACCELERATOR"
set root_dir [file normalize "."]
set spinal_dir [file join $root_dir "SPINAL"]
set vivado_dir [file join $spinal_dir "vivado"]
set ip_repo_dir [file join $root_dir "ip_repo"]

# Create IP repo directory if it doesn't exist
if {[file exists $ip_repo_dir]} {
    file delete -force $ip_repo_dir
}
file mkdir $ip_repo_dir

# Create a temporary project for packaging
create_project -force tmp_ip_pack ./tmp_ip_pack -part xczu9eg-ffvb1156-2-e

# Add all Verilog and DAT files from the vivado directory
add_files [glob $vivado_dir/*.v]
add_files [glob $vivado_dir/*.dat]

# Set top level
set_property top $ip_name [current_fileset]
update_compile_order -fileset sources_1

# Package the IP
ipx::package_project -root_dir $ip_repo_dir -vendor cityu -library hls -taxonomy /UserIP -import_files -force

# Set IP properties
set ip_core [ipx::current_core]
set_property display_name "ViM Accelerator" $ip_core
set_property description "ViM Accelerator with SpinalHDL wrapper" $ip_core

# Associate clock and reset for AXI interfaces
set bus_interfaces [ipx::get_bus_interfaces -of_objects $ip_core]
foreach bus_if $bus_interfaces {
    set bus_type [get_property ABSTRACTION_TYPE_NAME $bus_if]
    if {$bus_type == "axi4_rtl" || $bus_type == "axilite_rtl"} {
        puts "Associating clock/reset for interface: [get_property NAME $bus_if]"
        ipx::associate_bus_interfaces -busif [get_property NAME $bus_if] -clock clk $ip_core
    }
}

# Add clock interface with frequency parameter (350 MHz)
set clk_interface [ipx::get_bus_interfaces -of_objects $ip_core -filter {NAME == clk}]
if {$clk_interface == ""} {
    set clk_interface [ipx::add_bus_interface clk $ip_core]
    set_property abstraction_type_vlnv xilinx.com:signal:clock_rtl:1.0 $clk_interface
    set_property bus_type_vlnv xilinx.com:signal:clock:1.0 $clk_interface
    set clk_port [ipx::add_port_map CLK $clk_interface]
    set_property physical_name clk $clk_port
}

# Set clock frequency to 350 MHz
set clk_param [ipx::get_bus_parameters FREQ_HZ -of_objects $clk_interface]
if {$clk_param == ""} {
    set clk_param [ipx::add_bus_parameter FREQ_HZ $clk_interface]
}
set_property value 350000000 $clk_param

# Add clock constraint (350 MHz = period 2.857 ns)
# Create constraint file
set xdc_file [file join $ip_repo_dir "${ip_name}_clock.xdc"]
set xdc_fh [open $xdc_file w]
puts $xdc_fh "create_clock -period 2.857 -name clk \[get_ports clk\]"
close $xdc_fh

# Add the constraint file to the IP
set constraint_file_group [ipx::get_file_groups xilinx_constraintfileset -of_objects $ip_core]
if {$constraint_file_group == ""} {
    set constraint_file_group [ipx::add_file_group xilinx_constraintfileset $ip_core]
}
ipx::add_file $xdc_file $constraint_file_group

# Save and close
ipx::create_xgui_files $ip_core
ipx::update_checksums $ip_core
ipx::save_core $ip_core
close_project -delete

puts "IP packaging complete. IP repo is at: $ip_repo_dir"
