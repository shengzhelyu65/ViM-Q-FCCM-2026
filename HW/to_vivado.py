import re
import os
import glob
import shutil

# Paths relative to ViM/ directory
SPINAL_DIR = "SPINAL"
VIVADO_DIR = os.path.join(SPINAL_DIR, "vivado")

# make directory "vivado" if it does not exist
if not os.path.exists(VIVADO_DIR):
    os.makedirs(VIVADO_DIR)
else:
    # remove all .v .dat files in the vivado folder
    for f in glob.glob(os.path.join(VIVADO_DIR, "*.v")):
        os.remove(f)
    for f in glob.glob(os.path.join(VIVADO_DIR, "*.dat")):
        os.remove(f)

# Files to process
spinal_files = [os.path.join(SPINAL_DIR, "ViM_ACCELERATOR.v")]
hls_all_files = glob.glob(os.path.join(SPINAL_DIR, "src/main/verilog/*/all.v"))

all_v_files = spinal_files + hls_all_files

print(f"Processing {len(all_v_files)} Verilog files...")

for file_path in all_v_files:
    file_name = os.path.basename(file_path)
    # If it's all.v, rename it to something unique
    if file_name == "all.v":
        module_name = os.path.basename(os.path.dirname(file_path))
        target_name = f"{module_name}_all.v"
    else:
        target_name = file_name
    
    contents = []
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                # use regex to match readmemh
                match = re.search(r'\$readmemh\("(.*?)"', line)
                if match:
                    # get the file name
                    original_path = match.group(1)
                    # replace the path to local
                    dat_file_name = original_path.split("/")[-1]
                    new_line = line.replace(original_path, "./"+dat_file_name)
                    contents.append(new_line)
                else:
                    contents.append(line)
        
        target_path = os.path.join(VIVADO_DIR, target_name)
        with open(target_path, "w") as f:
            f.writelines(contents)

# copy all the dat files to vivado folder
print("Copying .dat files...")
dat_files = glob.glob(os.path.join(SPINAL_DIR, "src/main/verilog/*/*.dat"))
for dat_file in dat_files:
    shutil.copy(dat_file, VIVADO_DIR)

print(f"Done! Files prepared in '{VIVADO_DIR}' directory.")
