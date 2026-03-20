# Vivado script to create a block design with ViM_ACCELERATOR
# Usage: vivado -mode batch -source create_bd.tcl -tclargs <part_number> <board_part>

set part_number "xczu9eg-ffvb1156-2-e"
set board_part "xilinx.com:zcu102:part0:3.4"

if { $argc >= 1 } { set part_number [lindex $argv 0] }
if { $argc >= 2 } { set board_part [lindex $argv 1] }

set project_name "vim_project"
set project_dir "./vivado_project"
set ip_repo_dir [file normalize "./ip_repo"]

# Create project
create_project -force $project_name $project_dir -part $part_number
if { $board_part != "" } {
    set_property BOARD_PART $board_part [current_project]
}

# Add IP repo
set_property  ip_repo_paths  $ip_repo_dir [current_project]
update_ip_catalog

# Create Block Design
create_bd_design "design_1"

# Instantiate ViM_ACCELERATOR
set vim_cell [create_bd_cell -type ip -vlnv cityu:hls:ViM_ACCELERATOR:1.0 ViM_ACCELERATOR_0]

# Instantiate Zynq UltraScale+ MPSOC
set ps_cell [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.5 zynq_ultra_ps_e_0]

# Configure PS with 350 MHz clock - set before automation
set_property -dict [list \
    CONFIG.PSU__USE__S_AXI_GP0 {1} \
    CONFIG.PSU__USE__S_AXI_GP1 {1} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {350} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__ACT_FREQMHZ {350.000000} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__SRCSEL {IOPLL} \
] $ps_cell

# Apply automation with board preset
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1" } $ps_cell

# Force clock frequency again after automation
set_property CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {350} $ps_cell
set_property CONFIG.PSU__CRL_APB__PL0_REF_CTRL__ACT_FREQMHZ {350.000000} $ps_cell

# Add Processor System Reset
set ps_reset [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 proc_sys_reset_0]

# Add SmartConnect for AXI-Lite control (14 slaves)
set sc_lite [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smartconnect_lite]
set_property -dict [list CONFIG.NUM_MI {14} CONFIG.NUM_SI {1}] $sc_lite

# Connect CPU to SmartConnect Lite
set hpm0_lpd_pin [get_bd_intf_pins -quiet $ps_cell/M_AXI_HPM0_LPD]
if { $hpm0_lpd_pin == "" } {
    set hpm0_lpd_pin [get_bd_intf_pins -quiet $ps_cell/maxihpm0_lpd]
}
if { $hpm0_lpd_pin == "" } {
    # Try FPD if LPD is not found
    set hpm0_lpd_pin [get_bd_intf_pins -quiet $ps_cell/M_AXI_HPM0_FPD]
}
connect_bd_intf_net $hpm0_lpd_pin [get_bd_intf_pins $sc_lite/S00_AXI]

# Add two SmartConnects for AXI Masters (26 total, split 13 + 13)
set sc_mem0 [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smartconnect_mem0]
set_property -dict [list CONFIG.NUM_SI {13} CONFIG.NUM_MI {1}] $sc_mem0

set sc_mem1 [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smartconnect_mem1]
set_property -dict [list CONFIG.NUM_SI {13} CONFIG.NUM_MI {1}] $sc_mem1

# Connect SmartConnects to CPU Slaves
set hpc0_fpd_pin [get_bd_intf_pins -quiet $ps_cell/S_AXI_HPC0_FPD]
if { $hpc0_fpd_pin == "" } { set hpc0_fpd_pin [get_bd_intf_pins -quiet $ps_cell/saxihpc0_fpd] }
connect_bd_intf_net [get_bd_intf_pins $sc_mem0/M00_AXI] $hpc0_fpd_pin

set hpc1_fpd_pin [get_bd_intf_pins -quiet $ps_cell/S_AXI_HPC1_FPD]
if { $hpc1_fpd_pin == "" } { set hpc1_fpd_pin [get_bd_intf_pins -quiet $ps_cell/saxihpc1_fpd] }
connect_bd_intf_net [get_bd_intf_pins $sc_mem1/M00_AXI] $hpc1_fpd_pin

# Loop through interfaces and connect them
set vim_intfs [get_bd_intf_pins $vim_cell/*]
set lite_idx 0
set mem_idx 0

foreach intf $vim_intfs {
    set name [get_property NAME $intf]
    set mode [get_property MODE $intf]
    
    if { [string first "s_axi" $name] != -1 && $mode == "Slave" } {
        if { $lite_idx < 14 } {
            set sc_port [format "M%02d_AXI" $lite_idx]
            connect_bd_intf_net [get_bd_intf_pins $sc_lite/$sc_port] $intf
            incr lite_idx
        }
    } elseif { [string first "m_axi" $name] != -1 && $mode == "Master" } {
        if { $mem_idx < 13 } {
            set sc_port [format "S%02d_AXI" $mem_idx]
            connect_bd_intf_net $intf [get_bd_intf_pins $sc_mem0/$sc_port]
        } elseif { $mem_idx < 26 } {
            set sc_port [format "S%02d_AXI" [expr $mem_idx - 13]]
            connect_bd_intf_net $intf [get_bd_intf_pins $sc_mem1/$sc_port]
        }
        incr mem_idx
    }
}

# Tie off floating signals
set xlconstant [create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 xlconstant_0]
set_property -dict [list CONFIG.CONST_VAL {0} CONFIG.CONST_WIDTH {1}] $xlconstant

set floating_pins [list "signals_I_L_BEGIN" "signals_I_L_CLOSE"]
foreach p $floating_pins {
    set pin [get_bd_pins -quiet $vim_cell/$p]
    if { $pin != "" } { connect_bd_net [get_bd_pins $xlconstant/dout] $pin }
}

# Tie off wider signals
set xlconstant_wide [create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 xlconstant_wide]
set_property -dict [list CONFIG.CONST_VAL {0} CONFIG.CONST_WIDTH {32}] $xlconstant_wide

set wide_pins [list "signals_I_MEMORY_X" "signals_I_MEMORY_W" "signals_I_MEMORY_Y" "signals_I_MEMORY_C" "signals_I_MEMORY_H" "signals_I_POS" "signals_I_T"]
foreach p $wide_pins {
    set pin [get_bd_pins -quiet $vim_cell/$p]
    if { $pin != "" } { connect_bd_net [get_bd_pins $xlconstant_wide/dout] $pin }
}

# Clock and Reset
set clk_net [get_bd_pins $ps_cell/pl_clk0]
set ps_rst_n [get_bd_pins $ps_cell/pl_resetn0]

# Connect Reset Module
connect_bd_net $clk_net [get_bd_pins $ps_reset/slowest_sync_clk]
connect_bd_net $ps_rst_n [get_bd_pins $ps_reset/ext_reset_in]
set rst_n_sync [get_bd_pins $ps_reset/peripheral_aresetn]

connect_bd_net $clk_net [get_bd_pins $vim_cell/clk]
connect_bd_net $rst_n_sync [get_bd_pins $vim_cell/resetn]
connect_bd_net $clk_net [get_bd_pins $sc_lite/aclk]
connect_bd_net $rst_n_sync [get_bd_pins $sc_lite/aresetn]
connect_bd_net $clk_net [get_bd_pins $sc_mem0/aclk]
connect_bd_net $rst_n_sync [get_bd_pins $sc_mem0/aresetn]
connect_bd_net $clk_net [get_bd_pins $sc_mem1/aclk]
connect_bd_net $rst_n_sync [get_bd_pins $sc_mem1/aresetn]

# Connect all clock pins on PS
set clk_pins [list "maxihpm0_lpd_aclk" "maxihpm0_fpd_aclk" "maxihpm1_fpd_aclk" "saxihpc0_fpd_aclk" "saxihpc1_fpd_aclk"]
foreach p $clk_pins {
    set pin [get_bd_pins -quiet $ps_cell/$p]
    if { $pin != "" } {
        puts "Connecting clock pin: $p"
        connect_bd_net $clk_net $pin
    }
}

# Assign Addresses
assign_bd_address

# CRITICAL: Force 350 MHz on clock pin - this must be done
set_property CONFIG.FREQ_HZ 350000000 $clk_net

# Force 350 MHz on ALL AXI interface pins to override any defaults
# This ensures SmartConnect and IP interfaces all use 350 MHz
set sc_lite_pins [get_bd_intf_pins -quiet $sc_lite/*]
set sc_mem0_pins [get_bd_intf_pins -quiet $sc_mem0/*]
set sc_mem1_pins [get_bd_intf_pins -quiet $sc_mem1/*]
set vim_pins [get_bd_intf_pins -quiet $vim_cell/*]

foreach intf_pin [concat $sc_lite_pins $sc_mem0_pins $sc_mem1_pins $vim_pins] {
    if {$intf_pin != ""} {
        catch {
            set_property CONFIG.FREQ_HZ 350000000 $intf_pin
        }
    }
}

# Also set frequency on the interface nets themselves
foreach intf_net [get_bd_intf_nets -quiet -of_objects [concat $sc_lite_pins $sc_mem0_pins $sc_mem1_pins $vim_pins]] {
    if {$intf_net != ""} {
        catch {
            set_property CONFIG.FREQ_HZ 350000000 $intf_net
        }
    }
}

# Validate and Save
validate_bd_design
save_bd_design

puts "Block design creation complete."
