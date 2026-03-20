import os
import subprocess
import sys

def run_command(cmd, cwd=None):
    """Run a shell command and exit if it fails."""
    print(f"\n[EXEC] {cmd}")
    process = subprocess.Popen(cmd, shell=True, executable='/bin/bash', cwd=cwd)
    process.wait()
    if process.returncode != 0:
        print(f"\n[ERROR] Command failed with exit code {process.returncode}")
        sys.exit(process.returncode)

def main():
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
    
    print("="*60)
    print(f"Starting ViM Accelerator Vivado Flow (Version {vivado_version})")
    print("="*60)
    
    # Step 1: Prepare files
    print("\n>>> Step 1: Preparing RTL and data files...")
    run_command("python3 to_vivado.py", cwd=vim_dir)
    
    # Step 2: Package IP
    print("\n>>> Step 2: Packaging ViM_ACCELERATOR IP...")
    package_cmd = f"source {settings_path} && vivado -mode batch -source package_ip.tcl"
    run_command(package_cmd, cwd=vim_dir)
    
    # Step 3: Create Block Design
    print("\n>>> Step 3: Creating Block Design Project...")
    create_bd_cmd = f"source {settings_path} && vivado -mode batch -source create_bd.tcl"
    run_command(create_bd_cmd, cwd=vim_dir)
    
    print("\n" + "="*60)
    print("SUCCESS: ViM Accelerator IP and Block Design created!")
    print("="*60)

if __name__ == "__main__":
    main()
