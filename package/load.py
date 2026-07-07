from functions import *
import re
import os
import pandas as pd
from emboss import *

import yaml

with open("config.yaml", "r") as file:
    params = yaml.safe_load(file)

# parameters
hit_type = params["structure_type"]  # options: hairpin, spacer, tc_end, c_end
priority_is_max = False  # max or min hirpin energy, now always False

# paths
#hairpins_file_path = r"../input/palindrom_analyzer_output.txt"
if "hairpins_path" in params:
 hairpins_file_path = params["hairpins_path"]
else:
 hairpins_file_path = get_palindrome(params["stem_min_length"], params["stem_max_length"], params["loop_length"], params["number_mismatches"])

#hairpins_file_path = r"../input/palindrom_analyzer_output_Sashas_energy.txt"
#hairpins_file_path = r"../input/emboss_6_30_10_1_in_PA_format.txt"
#genome_file_path = r"../input/mpox_genome_seq.fna"
genome_file_path = params["genome_path"]
#mutation_path = r"../input/snp_locations.xlsx"
mutation_path = Path(__file__).parent.parent / params["mutation_path"]

# classes


class Hairpin:

    def __init__(self, l_s_m, position, score, palindrome):

        lsm = list(map(int, l_s_m.split("-")))
        self.spacer_length = lsm[1]
        self.stem_length = lsm[0]
        self.miss = lsm[2]
        self.length = self.spacer_length + self.stem_length * 2
        self.position = position
        self.score = score
        self.palindrome = palindrome
        self.sequence = palindrome.replace(" ", "")
        self.start = position - 1
        self.end = position - 2 + self.length
        self.stem_indexes = [(self.start, self.start + self.stem_length - 1), (self.end - self.stem_length + 1, self.end)]
        self.spacer_index = (self.start + self.stem_length, self.end - self.stem_length)

    def __eq__(self, other):
        return self.palindrome == other.palindrome and self.start == other.start and self.end == other.end
    
    def __str__(self):
        return f"start={self.start}, end={self.end}, palindrome={self.palindrome}"
    
    def __repr__(self):
        return self.__str__()

    def can_exist(self, second_hairpin):
        return max(self.start, second_hairpin.start) > min(self.end, second_hairpin.end)

    def print_data(self):
        print(self.start, self.end, self.length, self.palindrome, self.spacer_index)



class Genome:

    def __init__(self, genome_file_path):

        s = ""
        if os.path.exists(genome_file_path):
            file = open(genome_file_path)
        else:
            raise FileNotFoundError(f"File not found: {genome_file_path}")
        self.name = file.readline()
        for line in file.readlines():
            line = line.rstrip()
            s += line
        self.sequence = s
        self.path = genome_file_path
        self.length = len(s)

    def targets(self):
        c_tc = [i.end() - 1 for i in re.finditer("TC", self.sequence)]
        g_ga = [i.start() for i in re.finditer("GA", self.sequence)]
        targets = sorted(list(set(c_tc + g_ga)))
        return targets

    def end_targets(self, used_hairpins):

        tc_end = []
        just_tc = [i.end() - 1 for i in re.finditer("TC", self.sequence)]
        for i in just_tc:
            for h in used_hairpins:
                if h.length == h.stem_length * 2 + 1 or h.length == h.stem_length * 2:
                    continue
                l, r = h.spacer_index
                if i == r:
                    tc_end.append(i)
                    # print(i
                    # h.print_data()
                    break

        ga_end = []
        just_ga = [i.start() for i in re.finditer("GA", self.sequence)]
        for i in just_ga:
            for h in used_hairpins:
                if h.length == h.stem_length * 2 + 1 or h.length == h.stem_length * 2:
                    continue
                l, r = h.spacer_index
                if i == l:
                    ga_end.append(i)
                    # print(i)
                    # h.print_data()
                    break
        return (sorted(tc_end + ga_end), sorted(just_tc + just_ga))

    def only_c_g(self, used_hairpins):
        c = [i.start() for i in re.finditer("C", self.sequence)]
        g = [i.start() for i in re.finditer("G", self.sequence)]
        c_end = []
        for i in c:
            for h in used_hairpins:
                if h.length == h.stem_length * 2:
                    continue
                l, r = h.spacer_index
                if i == r:
                    c_end.append(i)
                    # print(i)
                    # h.print_data()
                    break
        g_end = []
        for i in g:
            for h in used_hairpins:
                if h.length == h.stem_length * 2:
                    continue
                l, r = h.spacer_index
                if i == l:
                    g_end.append(i)
                    # print(i)
                    # h.print_data()
                    break

        return (sorted(c_end + g_end), sorted(c + g))


# function for loading hairpin to list


def load_hairpins(hairpins_file_path):

    hairpins_list = []
    if os.path.exists(hairpins_file_path):
        hairpins_file = open(hairpins_file_path)
    else:
        print("File not found")
        return 0
    hairpins_file.readline()
    for line in hairpins_file.readlines():
        line = line.split()
        h = Hairpin(line[0], int(line[1]), float(line[2]), " ".join(line[3:]))
        if h.spacer_length <= int(params["filter_looplen_lessequal"]):
            continue
        hairpins_list.append(h)
    hairpins_file.close()
    return hairpins_list


def genome_coverage(hairpin_list, genome_length):
    """Return the fraction of genome covered by hairpins (0.0–1.0).

    Overlapping hairpins are merged before counting, so each base is
    counted only once.  hairpin.start and hairpin.end are 0-based,
    inclusive on both ends.
    """
    if not hairpin_list:
        return 0.0
    intervals = sorted((h.start, h.end) for h in hairpin_list)
    covered = 0
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= cur_end + 1:
            cur_end = max(cur_end, end)
        else:
            covered += cur_end - cur_start + 1
            cur_start, cur_end = start, end
    covered += cur_end - cur_start + 1
    return covered / genome_length


# load genome from file
genome = Genome(genome_file_path)

# load mutations from article

mutations_list_APOBEC = []
mutations_list_APOBEC_ext = []
mutcsv = pd.read_csv(mutation_path, sep="\t")
mutAPOBEC = mutcsv[mutcsv['is_APOBEC'] == 1]
mutAPOBECext = mutcsv[(mutcsv['is_APOBEC'] == True) | (mutcsv['is_APOBEC_ext'] == True)]

if not params["are_independent_events"]:
 mutations_list_APOBEC = (mutAPOBEC["pos_ref"]-1).to_list()
 mutations_list_APOBEC_ext = (mutAPOBECext["pos_ref"]-1).to_list()
 mut_cnt_APOBEC = len(mutations_list_APOBEC)
 mut_cnt_APOBEC_ext = len(mutations_list_APOBEC_ext)
else:
 for index, row in mutAPOBEC.iterrows():
  pos, count = int(row["pos_ref"]), int(row["independent_events"])
  mutations_list_APOBEC.extend([pos - 1] * count)  # convert to 0-based
 mut_cnt_APOBEC = len(mutations_list_APOBEC)
 for index, row in mutAPOBECext.iterrows():
  pos, count = int(row["pos_ref"]), int(row["independent_events"])
  mutations_list_APOBEC_ext.extend([pos - 1] * count)  # convert to 0-based
 mut_cnt_APOBEC_ext = len(mutations_list_APOBEC_ext)
 
# load hairpins from file
all_hairpins_list = load_hairpins(hairpins_file_path)
