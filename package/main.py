import yaml
import subprocess
import load
from all_hairpins import *
from hairpin_groups import *

with open("config.yaml", "r") as file:
    params = yaml.safe_load(file)

if params["hairpin_selection_type"] == "all":
    #subprocess.run(["python", "all_hairpins.py"])
    log10p, pvalue, hits, frac, targets_in_struct, all_targets = calculate_pval_all(load.all_hairpins_list, params["structure_type"])
else:
    #subprocess.run(["python", "hairpin_groups.py"])
    log10p, pvalue, hits, frac, targets_in_struct, all_targets = calculate_pval_groups(load.all_hairpins_list, params["hairpin_selection_type"], params["structure_type"])
    
print(pvalue)
