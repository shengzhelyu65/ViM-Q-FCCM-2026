from pre_syn_process import *
import os
import sys

INSTANCE_DIR = os.path.join(ROOT_DIR, "instances")

if len(sys.argv) > 1:
    case_names = [sys.argv[1]]
else:
    case_names = [
        "LINEAR_BLOCK",
        "CONV",
        "NORM_SUM",
        "SSM",
        "SMOOTH",
        "PATCH_OPS",
        "EMBED",
    ]
    
create_subprojects(INSTANCE_DIR, case_names=case_names, overwrite=True)

create_tcls(INSTANCE_DIR, case_names=case_names, do_csim=False, do_csynth=False, do_cosim=True)

run_instances(INSTANCE_DIR, case_names=case_names, version="2025.2", do_csim=False, do_csynth=False, do_cosim=True)

