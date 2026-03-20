import os
from post_syn_process import *

INSTANCE_DIR = os.path.join(ROOT_DIR, "instances")
IP_DIR = os.path.join(ROOT_DIR, "ips")

print_resource_table(INSTANCE_DIR)
