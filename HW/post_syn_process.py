import os
import re
import glob
import shutil
from pre_syn_process import ROOT_DIR
from constants import SPINAL_DIR


def get_resource_table(instances_root: str, instances_list=[]):
    if not instances_list:
        instances_list = [d for d in os.listdir(instances_root) 
                         if os.path.isdir(os.path.join(instances_root, d)) and d.startswith("proj_")]
    
    resource_types = ["BRAM_18K", "DSP", "FF", "LUT", "URAM"]
    result_dict = {}
    available_resources = {}  # Store available resources (should be same for all)
    
    for instance in instances_list:
        # Try to find synthesis report (Vitis HLS 2025.2 format)
        report_paths = [
            os.path.join(instances_root, instance, "work", "solution", "syn", "report", "top_csynth.rpt"),
            os.path.join(instances_root, instance, "work", "solution", "syn", "report", "csynth.rpt"),
            os.path.join(instances_root, instance, "work", "solution", "syn", "report", "test_top_csynth.rpt"),
        ]
        
        report_path = None
        for path in report_paths:
            if os.path.exists(path):
                report_path = path
                break
        
        if not report_path:
            print(f"Warning: No synthesis report found for {instance}")
            continue
        
        try:
            with open(report_path, 'r') as f:
                report_content = f.read()
        except Exception as e:
            print(f"Error reading {report_path}: {e}")
            continue
        
        instance_dict = {}
        
        # Extract resources from "Total" row in Utilization Estimates Summary section
        # Format: |Total            |      327|   568|  161881|   99824|    0|
        # Match the first occurrence which should be in the Summary section
        total_pattern = r'\|\s*Total\s+\|[^\|]*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|'
        total_matches = re.findall(total_pattern, report_content)
        
        # Extract available resources from "Available" row (should be right after Total)
        # Format: |Available        |     1824|  2520|  548160|  274080|    0|
        # Use a simpler pattern that matches numbers after Available
        available_pattern = r'\|Available[^\|]*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|'
        available_matches = re.findall(available_pattern, report_content)
        
        # Use the first match (Summary section)
        total_match = total_matches[0] if total_matches else None
        available_match = available_matches[0] if available_matches else None
        
        if total_match and available_match:
            # Store used resources (total_match is now a tuple from findall)
            used_bram = int(total_match[0])
            used_dsp = int(total_match[1])
            used_ff = int(total_match[2])
            used_lut = int(total_match[3])
            used_uram = int(total_match[4])
            
            # Get available resources (available_match is now a tuple from findall)
            avail_bram = int(available_match[0])
            avail_dsp = int(available_match[1])
            avail_ff = int(available_match[2])
            avail_lut = int(available_match[3])
            avail_uram = int(available_match[4])
            
            # Store available resources (use first instance's values, should be same for all)
            if not available_resources:
                available_resources["BRAM_18K"] = avail_bram
                available_resources["DSP"] = avail_dsp
                available_resources["FF"] = avail_ff
                available_resources["LUT"] = avail_lut
                available_resources["URAM"] = avail_uram
            
            # Calculate utilization percentages
            util_bram = (used_bram * 100.0 / avail_bram) if avail_bram > 0 else 0.0
            util_dsp = (used_dsp * 100.0 / avail_dsp) if avail_dsp > 0 else 0.0
            util_ff = (used_ff * 100.0 / avail_ff) if avail_ff > 0 else 0.0
            util_lut = (used_lut * 100.0 / avail_lut) if avail_lut > 0 else 0.0
            util_uram = (used_uram * 100.0 / avail_uram) if avail_uram > 0 else 0.0
            
            instance_dict["BRAM_18K"] = [str(used_bram), f"{util_bram:.1f}%"]
            instance_dict["DSP"] = [str(used_dsp), f"{util_dsp:.1f}%"]
            instance_dict["FF"] = [str(used_ff), f"{util_ff:.1f}%"]
            instance_dict["LUT"] = [str(used_lut), f"{util_lut:.1f}%"]
            instance_dict["URAM"] = [str(used_uram), f"{util_uram:.1f}%"]
        else:
            if not total_match:
                print(f"Warning: Could not find Total row in report for {instance}")
            if not available_match:
                print(f"Warning: Could not find Available row in report for {instance}")
            continue
        
        # Extract clock period (CP) from Timing section
        # Format: |ap_clk  |  2.50 ns|  2.948 ns|     0.68 ns|
        cp_pattern = r'\|\s*ap_clk\s*\|[^\|]*\|\s*([\d.]+)\s*ns\s*\|'
        cp_match = re.search(cp_pattern, report_content)
        if cp_match:
            instance_dict["CP"] = [cp_match.group(1)]
        else:
            instance_dict["CP"] = ['N/A']
        
        # Only add if we got all required resources
        if all(key in instance_dict for key in resource_types):
            result_dict[instance] = instance_dict
        else:
            print(f"Warning: Missing resource information for {instance}")
    
    return result_dict


def print_resource_table(instances_root: str):
    """Print a formatted resource usage table with utilization percentages.
    
    Args:
        instances_root: Root directory containing instance subdirectories
    """
    resource_types = ["BRAM_18K", "DSP", "FF", "LUT", "URAM", "CP"]
    result_dict = get_resource_table(instances_root)
    
    if not result_dict:
        print("No resource data found. Make sure synthesis has been completed.")
        return
    
    print()
    # Header with resource names
    header = f"{'Instance':40s}"
    for rt in resource_types:
        if rt == "CP":
            header += f"{rt:>12s}"
        else:
            header += f"{rt:>18s}"  # Wider for "used (util%)"
    print(header)
    print("-" * (40 + 18 * (len(resource_types) - 1) + 12))
    
    # Calculate totals
    totals = {}
    total_percentages = {}  # Sum of percentages
    for resource_type in resource_types:
        totals[resource_type] = 0
        total_percentages[resource_type] = 0.0
    
    for instance in sorted(result_dict.keys()):
        print(f"{instance:40s}", end="")
        for resource_type in resource_types:
            if resource_type in result_dict[instance] and result_dict[instance][resource_type]:
                if resource_type == "CP":
                    # CP is just a value, no percentage
                    value = result_dict[instance][resource_type][0]
                    print(f"{value:>12s}", end="")
                    # For CP, we'll show max instead of sum
                    try:
                        cp_val = float(value)
                        if resource_type not in totals or totals[resource_type] < cp_val:
                            totals[resource_type] = cp_val
                    except:
                        pass
                else:
                    # Format as "used (util%)"
                    used = result_dict[instance][resource_type][0]
                    util = result_dict[instance][resource_type][1] if len(result_dict[instance][resource_type]) > 1 else "N/A"
                    print(f"{used:>8s} ({util:>7s})", end="")
                    # Sum up numeric values
                    try:
                        totals[resource_type] += int(used)
                        # Sum up percentages (extract number from "X.X%")
                        if util != "N/A":
                            util_val = float(util.rstrip('%'))
                            total_percentages[resource_type] += util_val
                    except:
                        pass
            else:
                if resource_type == "CP":
                    print(f"{'N/A':>12s}", end="")
                else:
                    print(f"{'N/A':>8s} ({'N/A':>7s})", end="")
        print()
    
    # Print total row
    print("-" * (40 + 18 * (len(resource_types) - 1) + 12))
    print(f"{'TOTAL':40s}", end="")
    
    for resource_type in resource_types:
        if resource_type == "CP":
            # For CP, show max value
            if totals[resource_type] > 0:
                print(f"{totals[resource_type]:>12.2f}", end="")
            else:
                print(f"{'N/A':>12s}", end="")
        else:
            # For other resources, show sum and sum of percentages
            if totals[resource_type] > 0:
                total_util = total_percentages[resource_type]
                print(f"{totals[resource_type]:>8d} ({total_util:>6.1f}%)", end="")
            else:
                print(f"{'N/A':>8s} ({'N/A':>7s})", end="")
    print()
    print()


def collect_ip(instances_root: str, target_dir: str):
    """Collect IP files from export_ip directories.
    
    Args:
        instances_root: Root directory containing instance subdirectories
        target_dir: Target directory to copy IP files to
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Created target directory: {target_dir}")
    
    ip_files = glob.glob(os.path.join(instances_root, "**/export_ip/*.zip"), recursive=True)
    
    if not ip_files:
        print(f"No IP files found in {instances_root}")
        return
    
    for zip_file in ip_files:
        filename = os.path.basename(zip_file)
        target_path = os.path.join(target_dir, filename)
        print(f"Copying {zip_file} to {target_path}...")
        shutil.copy(zip_file, target_path)
    
    print(f"\nCollected {len(ip_files)} IP file(s) to {target_dir}")


def backup_verilog(instances_root: str, instances_list=[]):
    """Backup Verilog files from synthesis output.
    
    Args:
        instances_root: Root directory containing instance subdirectories
        instances_list: Optional list of instance names to process
    """
    if not instances_list:
        instances_list = [d for d in os.listdir(instances_root) 
                         if os.path.isdir(os.path.join(instances_root, d)) and d.startswith("proj_")]
    
    for instance in instances_list:
        vlog_dir = os.path.join(instances_root, instance, "work", "solution", "syn", "verilog")
        
        if not os.path.exists(vlog_dir):
            print(f"Warning: Verilog directory not found for {instance}: {vlog_dir}")
            continue
        
        vlog_files = [f for f in os.listdir(vlog_dir) if f.endswith(".v")]
        data_files = [f for f in os.listdir(vlog_dir) if f.endswith(".dat")]  # ROM
        
        print(f"Backing up {instance}...")
        print(f"  vlog_dir: {vlog_dir}")
        print(f"  vlog_files: {len(vlog_files)} files")
        print(f"  data_files: {len(data_files)} files")
        
        backup_dir = os.path.join(instances_root, instance, "work", "solution", "syn", "verilog_backup")
        
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
        else:
            print(f"  backup_dir {backup_dir} already exists, overwriting")
        
        for file in vlog_files + data_files:
            shutil.copyfile(os.path.join(vlog_dir, file), os.path.join(backup_dir, file))
        
        print(f"  ✓ Backed up {len(vlog_files) + len(data_files)} files")


def backup_log(instances_root: str, instances_list=[], overwrite=False):
    """Backup log files from HLS runs.
    
    Args:
        instances_root: Root directory containing instance subdirectories
        instances_list: Optional list of instance names to process
        overwrite: Whether to overwrite existing backup files
    """
    if not instances_list:
        instances_list = [d for d in os.listdir(instances_root) 
                         if os.path.isdir(os.path.join(instances_root, d)) and d.startswith("proj_")]
    
    for instance in instances_list:
        log_dir = os.path.join(instances_root, instance)
        
        # Try different log file locations (Vitis HLS 2025.2 format)
        log_paths = [
            os.path.join(log_dir, "work", "solution", "solution.log"),
            os.path.join(log_dir, "vitis_hls.log"),
        ]
        
        log_file = None
        for path in log_paths:
            if os.path.exists(path):
                log_file = path
                break
        
        if not log_file:
            print(f"Warning: No log file found for {instance}")
            continue
        
        backup_file = os.path.join(log_dir, "golden.log")
        
        if not os.path.exists(backup_file) or overwrite:
            shutil.copyfile(log_file, backup_file)
            print(f"✓ Backed up log for {instance}")
        else:
            print(f"  backup_file {backup_file} already exists, skipping backup")


def to_spinal_others(instances_root: str, case_names):
    """Copy and process Verilog files for SpinalHDL integration.
    
    Processes Verilog files from HLS synthesis and prepares them for SpinalHDL:
    - Combines all .v files into a single all.v
    - Adds Verilator lint directives
    - Replaces 'top' with case_name
    - Fixes readmemh paths
    - Adjusts AXI width parameters
    
    Args:
        instances_root: Root directory containing instance subdirectories
        case_names: List of case names to process
    """
    for case_name in case_names:
        target_file = os.path.join(SPINAL_DIR, "src", "main", "verilog", case_name, "all.v")
        src_dir = os.path.join(instances_root, f"proj_{case_name}", "work", "solution", "syn", "verilog")
        
        if not os.path.exists(src_dir):
            print(f"Warning: Verilog directory not found for {case_name}: {src_dir}")
            continue
        
        # Get verilog files
        vlog_files = [f for f in os.listdir(src_dir) if f.endswith(".v")]
        data_files = [f for f in os.listdir(src_dir) if f.endswith(".dat")]  # ROM
        
        if not vlog_files:
            print(f"Warning: No Verilog files found for {case_name}")
            continue
        
        print(f"Processing {case_name}...")
        print(f"  Source: {src_dir}")
        print(f"  Target: {target_file}")
        print(f"  Verilog files: {len(vlog_files)}")
        print(f"  Data files: {len(data_files)}")
        
        # Create content with lint directives
        content = []
        content.append("/* verilator lint_off PINMISSING */\n")
        content.append("/* verilator lint_off CASEINCOMPLETE */\n")
        content.append("/* verilator lint_off COMBDLY */\n")
        content.append("/* verilator lint_off CASEX */\n")
        content.append("/* verilator lint_off CASEOVERLAP */\n")
        
        # Process each verilog file
        for vlog_file in sorted(vlog_files):
            vlog_path = os.path.join(src_dir, vlog_file)
            with open(vlog_path, 'r') as f:
                input_lines = f.readlines()
            
            for input_line in input_lines:
                # Comment out delay statements
                if input_line.strip().startswith("#0"):
                    input_line = "//" + input_line
                
                # Fix readmemh paths
                if "readmemh" in input_line:
                    # Replace relative paths with absolute paths
                    verilog_dir = os.path.join(SPINAL_DIR, "src", "main", "verilog", case_name).replace("\\", "/")
                    input_line = input_line.replace('readmemh("./', f'readmemh("{verilog_dir}/')
                
                # Replace 'top' with case_name
                if "top" in input_line:
                    input_line = input_line.replace("top", case_name)
                
                # Comment out plusargs and add display
                if "plusargs" in input_line:
                    input_line = "//" + input_line + f'$display("This is {case_name}.\\n");\n'
                
                # Adjust AXI width parameters (for compatibility)
                if re.search(r"C_M_AXI_\w+_WIDTH\s*=\s*\d+", input_line):
                    port_name = re.search(r"C_M_AXI_(\w+)_WIDTH", input_line).group(1)
                    input_line = re.sub(r"C_M_AXI_\w+_ADDR_WIDTH\s*=\s*\d+", f"C_M_AXI_{port_name}_WIDTH = 48", input_line)
                
                content.append(input_line)
        
        # Create target directory
        target_dir = os.path.dirname(target_file)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            print(f"  Created directory: {target_dir}")
        
        # Write combined verilog file
        with open(target_file, "w") as f:
            f.writelines(content)
        print(f"  ✓ Created {target_file}")
        
        # Copy all .dat files
        for data_file in data_files:
            src_data = os.path.join(src_dir, data_file)
            dst_data = os.path.join(target_dir, data_file.replace("top", case_name))
            shutil.copyfile(src_data, dst_data)
            print(f"  ✓ Copied {data_file} -> {os.path.basename(dst_data)}")
        
        print(f"  ✓ Completed {case_name}\n")

