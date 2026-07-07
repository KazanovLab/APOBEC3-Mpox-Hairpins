from functions import *
import random
import math
from load import *
import matplotlib.pyplot as plt
from scipy.stats import binomtest

def calculate_pval_all(hairpins_list, structure_type):
 hit_type = structure_type

 groups = find_groups(hairpins_list)
 segments = []
 for group in groups:
  l = min([h.start for h in group])
  r = max([h.end for h in group])
  segments.append((l, r))
 segments.sort()
 total_len = sum([i[1] - i[0] + 1 for i in segments])
 spacer_len = sum([h.spacer_length for h in hairpins_list])

 targets_p = 0
 end_targets = []

 if hit_type not in ["hairpin","spacer","tc_end","c_end"]:
  print('Structure_type not in "hairpin", "spacer", "tc_end", "c_end"')
  sys.exit(1)

 if hit_type == "hairpin" or hit_type == "spacer":
  j = 0
  for t in genome.targets():
   hit = 0
   for h in hairpins_list:
    if hit_or_not(hit_type, h, t):
     hit = 1
     break
   j += hit
  targets_p = j / len(genome.targets())
  targets_in_struct = j
  all_targets = len(genome.targets())

 if hit_type == "tc_end":
  end_targets, all_positions = genome.end_targets(hairpins_list)
  targets_p = len(end_targets) / len(all_positions)
  targets_in_struct = len(end_targets)
  all_targets = len(all_positions)

 if hit_type == "c_end":
  end_targets, all_positions = genome.only_c_g(hairpins_list)
  targets_p = len(end_targets) / len(all_positions)
  targets_in_struct = len(end_targets)
  all_targets = len(all_positions)

 targets_in_struct = j if hit_type in ["hairpin", "spacer"] else len(end_targets)
 print(f"Targets in structures: {targets_in_struct}")
 print(f"Fraction of targets in structures: {targets_p * 100:.2f}%")
 print(f"Genome coverage by selected hairpins: {total_len * 100/genome.length:.2f}%")
 print(f"Genome coverage by selected spacers: {spacer_len * 100/genome.length:.2f}%")


# real hits

 hairpin_hits = []
 total_real_hits = 0
 if hit_type == "c_end":
   mutations_list = mutations_list_APOBEC_ext
 else:
   mutations_list = mutations_list_APOBEC
 for mut in mutations_list:
  hit = 0
  for hairpin in hairpins_list:
   if hit_or_not(hit_type, hairpin, mut, end_targets):
    hairpin_hits.append((mut, hairpin))
    hit = 1
    break
  total_real_hits += hit
 print("Mutations in structures:", total_real_hits)


 # binom

 ##hits_b = binom(targets_p, mut_cnt)

 # hist

 ##keys = list(hits_b.keys())
 ##values = list(hits_b.values())

 ##plt.bar(keys, values)
 ##positions = [i for i in range(0, len(keys))]
 ##labels = [keys[i] for i in range(0, len(keys))]
 ##plt.xticks(positions, labels, rotation=90)
 ##plt.grid(True, axis="y", linestyle="--", alpha=0.7)


 # threshold count binom

 ##print("Binomial distribution:")
 ##right = threshold(hits_b, 95)
 ##left = threshold(hits_b, 5)
 ##print(f"left tail: {left}")
 ##print(f"right tail: {right}")
 ##p = percentile(hits_b, total_real_hits)

 if hit_type == "c_end":
  result = binomtest(total_real_hits, mut_cnt_APOBEC_ext, targets_p, alternative='greater')
 else:
  result = binomtest(total_real_hits, mut_cnt_APOBEC, targets_p, alternative='greater')

 print(f"p-value: {result.pvalue*100:.5f}%")
 log10p = -math.log10(result.pvalue) if result.pvalue > 0 else float("inf")
 print(f"-log10(p-value): {log10p:.4f}")

 ##plt.show()

 return log10p, result.pvalue, total_real_hits, targets_p, targets_in_struct, all_targets
