from pre_syn_process import *
from post_syn_process import *
import os

INSTANCE_DIR = os.path.join(ROOT_DIR, "instances/")

case_names = [
    "LINEAR_BLOCK",
    "CONV",
    "NORM_SUM",
    "SSM",
    "SMOOTH",
    "PATCH_OPS",
    "EMBED",
]

instances_list = ["proj_" + case_name for case_name in case_names]

backup_verilog  (INSTANCE_DIR, instances_list=instances_list)
backup_log      (INSTANCE_DIR, instances_list=instances_list)

to_spinal_others(INSTANCE_DIR, case_names=case_names)
