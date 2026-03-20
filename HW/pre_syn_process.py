import os
import shutil
import threading
import subprocess
import sys
from time import time, sleep, strftime, localtime
from copy import deepcopy
import string
import platform

from constants import *
from hls_status_check import check_case_status


def backup_src_include(backup_dir="backup"):
    """Backup src/ and include/ directories before refactoring"""
    backup_path = os.path.join(ROOT_DIR, backup_dir)
    os.makedirs(backup_path, exist_ok=True)
    
    # Backup src/
    src_dir = os.path.join(ROOT_DIR, 'src')
    src_backup = os.path.join(backup_path, 'src')
    if os.path.exists(src_dir):
        if os.path.exists(src_backup):
            shutil.rmtree(src_backup)
        shutil.copytree(src_dir, src_backup)
        print(f"✓ Backed up {src_dir} to {src_backup}")
    else:
        print(f"✗ {src_dir} does not exist")
    
    # Backup include/
    include_dir = os.path.join(ROOT_DIR, 'include')
    include_backup = os.path.join(backup_path, 'include')
    if os.path.exists(include_dir):
        if os.path.exists(include_backup):
            shutil.rmtree(include_backup)
        shutil.copytree(include_dir, include_backup)
        print(f"✓ Backed up {include_dir} to {include_backup}")
    else:
        print(f"✗ {include_dir} does not exist")
    
    print(f"\nBackup completed in: {backup_path}")
    return backup_path


def create_subprojects(instances_root: str, case_names, overwrite=False):
    """Create subprojects for each case"""
    if not os.path.exists(instances_root):
        os.makedirs(instances_root)

    for case_name in case_names:
        proj_name = "proj_" + case_name
        case_dir = os.path.join(instances_root, proj_name)

        # ViM directory paths (current directory)
        case_src  = os.path.join(ROOT_DIR, 'case')
        src_src   = os.path.join(ROOT_DIR, 'src')
        case_dest = os.path.join(instances_root, proj_name, 'case')
        src_dest  = os.path.join(instances_root, proj_name, 'src')

        shutil.copytree(case_src, case_dest, ignore=shutil.ignore_patterns('*.bin'), dirs_exist_ok=overwrite)
        shutil.copytree(src_src,  src_dest,  dirs_exist_ok=overwrite)


def create_tcls(instances_root: str, case_names, version="old", top_name="top", do_csim=False, do_csynth=False, do_cosim=False, do_export=False, do_syn=False, do_impl=False, cosim_random_stall=False, phys_opt="none", pipeline_style="stp"):
    
    def bool2tcl(_bool):
        return "1" if _bool else "0"
    
    if platform.system() == 'Windows':
        ldflags = '-ldflags {-Wl,--stack,10485760}'
    else:
        ldflags = ''

    # Map version to template
    if version == "new" or version == "2025.2":
        template_filename = "template_2025.tcl" if os.path.exists(os.path.join(ROOT_DIR, "template_2025.tcl")) else "template.tcl"
    else:
        template_filename = "template.tcl"

    for case_name in case_names:
        template_path = os.path.join(ROOT_DIR, template_filename)
        with open(template_path) as f:
            content = f.read()

        template = string.Template(content)
        content = template.substitute(
            case_name=case_name,
            root_dir=ROOT_DIR,
            do_csim=bool2tcl(do_csim),
            do_csynth=bool2tcl(do_csynth),
            do_cosim=bool2tcl(do_cosim),
            do_export=bool2tcl(do_export),
            do_syn=bool2tcl(do_syn),
            do_impl=bool2tcl(do_impl),
            cosim_random_stall="-random_stall" if cosim_random_stall else "",
            phys_opt=phys_opt,
            pipeline_style=pipeline_style,
            ldflags=ldflags,
            top_name=top_name
        )

        tcl_file_path = os.path.join(instances_root, f"proj_{case_name}", "run.tcl")
        with open(tcl_file_path, "w") as f:
            f.write(content)


def run_instances(instances_root: str, case_names, version="2025.2", do_csim=False, do_csynth=False, do_cosim=False, do_syn=False):
    """Run Vitis HLS for multiple cases and generate a summary log"""
    
    # Logs directory is in ViM directory
    logs_dir = os.path.join(ROOT_DIR, "logs")
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    timestamp = strftime("%Y%m%d_%H%M%S", localtime())
    log_file = os.path.join(logs_dir, f"hls_run_summary_{timestamp}.log")
    
    def thread_func(case_name):
        start_time = time()
        proj_name = "proj_" + case_name
        case_dir = os.path.join(instances_root, proj_name)
        original_dir = os.getcwd()
        
        try:
            os.chdir(case_dir)
            print(f"{case_name} is running")

            if platform.system() == 'Windows':
                vitis_home = os.path.join("D:/Xilinx", "Vitis_HLS", version, "bin")
                vitis_hls_cmd = os.path.join(vitis_home, "vitis_hls -f run.tcl")
            else:  # Linux
                # Read settings path from environment variable
                settings_path = os.getenv("VIM_Q_VIVADO_SETTINGS")
                if settings_path is None:
                    print(f"ERROR: VIM_Q_VIVADO_SETTINGS environment variable is not set.")
                    print(f"Please export VIM_Q_VIVADO_SETTINGS=/path/to/Vivado/settings64.sh")
                    sys.exit(1)
                
                if not os.path.exists(settings_path):
                    print(f"ERROR: Vivado settings file not found at {settings_path}")
                    sys.exit(1)
                
                if version == "2025.2":
                    vitis_hls_cmd = ['bash', '-c', f'source {settings_path} && vitis-run --mode hls --tcl run.tcl']
                else:
                    vitis_hls_cmd = ['bash', '-c', f'source {settings_path} && vitis_hls -f run.tcl']

            exit_code = subprocess.call(vitis_hls_cmd)
            end_time = time()
            elapsed_time = end_time - start_time
            
            status, details = check_case_status(instances_root, case_name, do_csim, do_csynth, do_syn)
            if exit_code != 0:
                status = "FAILED"
            
            with open(log_file, 'a') as f:
                f.write(f"[{strftime('%Y-%m-%d %H:%M:%S', localtime())}] {case_name:20s} | {status:8s} | Time: {elapsed_time:8.2f}s | {details}\n")
            
            print(f"{case_name} is done, time: {elapsed_time:.2f}s, status: {status}")
            
        except Exception as e:
            with open(log_file, 'a') as f:
                f.write(f"[{strftime('%Y-%m-%d %H:%M:%S', localtime())}] {case_name:20s} | FAILED  | Time: N/A     | Exception: {str(e)}\n")
            print(f"{case_name} failed with exception: {e}")
        finally:
            os.chdir(original_dir)

    with open(log_file, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write(f"HLS Run Summary - {strftime('%Y-%m-%d %H:%M:%S', localtime())}\n")
        f.write("=" * 100 + "\n")
        f.write(f"Cases to run: {', '.join(case_names)}\n")
        f.write(f"Configuration: CSIM={do_csim}, CSYNTH={do_csynth}, COSIM={do_cosim}, SYN={do_syn}\n")
        f.write(f"Vitis HLS Version: {version}\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Case Name':20s} | {'Status':8s} | {'Time':12s} | {'Details'}\n")
        f.write("-" * 100 + "\n")
    
    print(f"Summary log will be written to: {log_file}")
    
    max_threads = 16
    case_queue = deepcopy(case_names)
    thread_pool = []

    try:
        while len(case_queue) > 0:
            if threading.active_count() < max_threads:
                case_name = case_queue.pop()
                thread = threading.Thread(target=thread_func, args=(case_name,))
                thread.start()
                thread_pool.append(thread)

            for t in thread_pool:
                if not t.is_alive():
                    thread_pool.remove(t)

            sleep(0.1)

        for thread in thread_pool:
            thread.join()
        
        with open(log_file, 'a') as f:
            f.write("-" * 100 + "\n")
            f.write(f"Run completed at {strftime('%Y-%m-%d %H:%M:%S', localtime())}\n")
            f.write("=" * 100 + "\n")
        
        print(f"\nSummary log saved to: {log_file}")
        
    except Exception as e:
        with open(log_file, 'a') as f:
            f.write(f"ERROR: {str(e)}\n")
        print(f"Error: {e}")

