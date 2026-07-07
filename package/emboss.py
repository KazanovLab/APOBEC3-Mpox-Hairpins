import yaml
import subprocess
from pathlib import Path
from hairpins import HairpinList

with open("config.yaml", "r") as file:
    params = yaml.safe_load(file)

def get_palindrome(stem_min_len, stem_max_len, spacer_len, num_mismatch):

 GENOME_PATH = params["genome_path"]

 with open(GENOME_PATH) as f:
  f.readline()
  GENOME_SEQ = f.read().replace("\n", "")

 EMBOSS_DIR   = Path(__file__).parent / "emboss"
 EMBOSS_DIR.mkdir(exist_ok=True)

 tag = f"mm{num_mismatch}_stemin{stem_min_len}_stemax{stem_max_len}_spacer{spacer_len}"
 emboss_file = EMBOSS_DIR / f"emboss_{tag}.txt"
 pa_file = EMBOSS_DIR / f"pa_{tag}.txt"
 subprocess.run(
        [
            "palindrome",
            "-sequence",      str(GENOME_PATH),
            "-minpallen",     str(stem_min_len),
            "-maxpallen",     str(stem_max_len),
            "-gaplimit",      str(spacer_len),
            "-nummismatches", str(num_mismatch),
            "-outfile",       str(emboss_file),
            "-overlap",
        ],
        check=True, capture_output=True,
  )

 if not pa_file.exists():
  pins = HairpinList.from_emboss(str(emboss_file), GENOME_SEQ)
  pins.recalculate_energy()
  pins.to_palindrome_analyzer(str(pa_file))
  
 return pa_file
