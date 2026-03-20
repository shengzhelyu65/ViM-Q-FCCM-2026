# Create project metadata
set description   ${case_name}_DESC
set display_name  ${case_name}_DISP
set library       "HLS"
set vendor        "CityUHK@CALAS"
set ipname        ${case_name}
set version       "1.0"

# Create export_ip folder
if { [file exists "export_ip"] == 0 } {
    file mkdir export_ip
}

# Parameters
set do_csim                 ${do_csim}
set do_csynth               ${do_csynth}
set do_cosim                ${do_cosim}
set do_export               ${do_export}
set do_syn                  ${do_syn}
set do_impl                 ${do_impl}
set phys_opt                ${phys_opt}

# Create a component
open_component -reset component_${case_name} -flow_target vivado

# Add design files
add_files case/${case_name}.cpp -cflags "-I${root_dir}/include -I../src -I${root_dir}/testbench"
# Add test bench & files
add_files -tb case/${case_name}.cpp -cflags "-I${root_dir}/include -I../src -I${root_dir}/testbench"

# ########################################################
# Set target device
set_part xczu9eg-ffvb1156-2-e
set_top ${top_name}
create_clock -period 350MHz
config_export -format ip_catalog -rtl verilog -vivado_clock 350MHz -vivado_phys_opt ${phys_opt}
config_compile -pipeline_style ${pipeline_style}

# Execute steps independently based on flags (much clearer than nested if-elseif)
if { $$do_csim == 1 } {
    csim_design ${ldflags}
}

if { $$do_csynth == 1 } {
    csynth_design
}

if { $$do_cosim == 1 } {
    cosim_design ${ldflags} ${cosim_random_stall}
}

if { $$do_export == 1 } {
    export_design -description $$description -display_name $$display_name -library $$library -vendor $$vendor -version $$version -taxonomy "IP Core" -output "export_ip/${case_name}" -ipname $$ipname
}

if { $$do_syn == 1 } {
    export_design -description $$description -display_name $$display_name -flow syn -library $$library -vendor $$vendor -version $$version -taxonomy "IP Core" -output "export_ip/${case_name}" -ipname $$ipname
}

if { $$do_impl == 1 } {
    export_design -description $$description -display_name $$display_name -flow impl -library $$library -vendor $$vendor -version $$version -taxonomy "IP Core" -output "export_ip/${case_name}" -ipname $$ipname
}

exit
