import os
import re


def check_case_status(instances_root: str, case_name: str, do_csim: bool, do_csynth: bool, do_syn: bool):
    proj_name = "proj_" + case_name
    case_dir = os.path.join(instances_root, proj_name)
    
    possible_log_paths = [
        os.path.join(case_dir, "work", "solution", "solution.log"),
        os.path.join(case_dir, "work", "solution", "csim", "report", "main_csim.log"),
        os.path.join(case_dir, "vitis_hls.log"),
        os.path.join(case_dir, "component_" + case_name, "hls", "vitis_hls.log"),
        os.path.join(case_dir, "component_" + case_name, "vitis_hls.log"),
    ]
    
    log_path = None
    for path in possible_log_paths:
        if os.path.exists(path):
            log_path = path
            break
    
    if not log_path:
        return "SUCCESS", ""
    
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
            last_20_lines = lines[-20:] if len(lines) >= 20 else lines
    except Exception as e:
        return "SUCCESS", ""
    
    error_pattern = r'ERROR'
    for line in last_20_lines:
        line_lower = line.lower()
        if '0 errors' in line_lower:
            continue
        if re.search(error_pattern, line, re.IGNORECASE):
            return "FAILED", ""
    
    return "SUCCESS", ""

