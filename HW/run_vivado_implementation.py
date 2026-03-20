import os
import subprocess
import sys
import time
import argparse

def run_command(cmd, cwd=None):
    """Run a shell command and exit if it fails."""
    print(f"\n[EXEC] {cmd}")
    process = subprocess.Popen(cmd, shell=True, executable='/bin/bash', cwd=cwd)
    process.wait()
    if process.returncode != 0:
        print(f"\n[ERROR] Command failed with exit code {process.returncode}")
        sys.exit(process.returncode)

def main():
    parser = argparse.ArgumentParser(description="Run Vivado implementation flow.")
    parser.add_argument(
        "--reports-only",
        action="store_true",
        help="Skip rebuild/re-implementation and only regenerate post-implementation utilization/power reports.",
    )
    args = parser.parse_args()

    # Setup paths
    vim_dir = os.path.dirname(os.path.abspath(__file__))
    vivado_version = "2025.2"
    
    # Read settings path from environment variable
    settings_path = os.getenv("VIM_Q_VIVADO_SETTINGS")
    if settings_path is None:
        print(f"[ERROR] VIM_Q_VIVADO_SETTINGS environment variable is not set.")
        print(f"Please export VIM_Q_VIVADO_SETTINGS=/path/to/Vivado/settings64.sh")
        sys.exit(1)
    
    if not os.path.exists(settings_path):
        print(f"[ERROR] Could not find Vivado settings at {settings_path}")
        sys.exit(1)
    
    if args.reports_only:
        print("="*60)
        print(f"Starting ViM post-implementation reports-only mode (Version {vivado_version})")
        print("="*60)
    else:
        print("="*60)
        print(f"Starting ViM Accelerator Implementation Flow (Version {vivado_version})")
        print("="*60)
    
    # Check if project exists
    project_path = os.path.join(vim_dir, "vivado_project/vim_project.xpr")
    if not os.path.exists(project_path):
        print(f"[ERROR] Vivado project not found at {project_path}")
        print("Please run 'python3 run_vivado_flow.py' first to create the project and block design.")
        sys.exit(1)
    
    # Run implementation or reports-only
    start_time = time.time()
    if args.reports_only:
        cmd = f"source {settings_path} && vivado -mode batch -source run_post_impl_reports.tcl"
    else:
        cmd = f"source {settings_path} && vivado -mode batch -source run_impl.tcl"
    run_command(cmd, cwd=vim_dir)
    end_time = time.time()
    
    duration = end_time - start_time
    minutes = int(duration // 60)
    seconds = int(duration % 60)
    
    print("\n" + "="*60)
    if args.reports_only:
        print("SUCCESS: Post-implementation utilization/power reports generated!")
    else:
        print("SUCCESS: ViM Accelerator Implementation Complete!")
    print(f"Total Time: {minutes}m {seconds}s")
    if not args.reports_only:
        print("Generated XSA: ./vim_accelerator.xsa")
    print("Post-implementation reports: ./vivado_reports/post_impl/")
    print("="*60)

if __name__ == "__main__":
    main()
