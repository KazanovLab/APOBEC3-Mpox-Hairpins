from __future__ import annotations
import re
import json
from pathlib import Path
from tqdm import tqdm

_NN_COEFS_DIR = Path(__file__).parent / "nn_coefs"


class Hairpin:
    with open(_NN_COEFS_DIR / "mismatches.json") as MMjson, open(
        _NN_COEFS_DIR / "nearest_neighbour.json"
    ) as NNjson, open(_NN_COEFS_DIR / "terminal_mismatches.json") as TMjson, open(
        _NN_COEFS_DIR / "loop_init.json"
    ) as initjson:
        MM_coefs = json.load(MMjson)
        NN_coefs = json.load(NNjson)
        terminal_MM_coefs = json.load(TMjson)
        init_coefs: dict[str, float] = json.load(initjson)

    def __init__(
        self,
        Start: int = 0,
        Sequence: str = "",
        Spacer: str = "",
        Opposite: str = "",
        Energy: float | None = None,
    ):
        self.Start = Start
        self.Energy = Energy
        self.End = (
            Start + 2 * len(Sequence) + len(Spacer)
        )  # End not included, as in python slices
        self.Sequence = Sequence
        self.Spacer = Spacer
        self.Opposite = Opposite
        if self.Energy is None:
            self.Energy = self.nn_energy()

    def __repr__(self):
        return f"IR at position {self.Start}: {self.Sequence}_{self.Spacer}_{self.Opposite}. Energy = {self.Energy}"

    def __eq__(self, other):
        return (self.Start, self.Sequence, self.Spacer, self.Opposite) == (
            other.Start,
            other.Sequence,
            other.Spacer,
            other.Opposite,
        )

    def __hash__(self):
        return hash((self.Start, self.Sequence, self.Spacer, self.Opposite))

    def __lt__(self, other):
        """
        Implemented for stability of HairpinList order, yet is a heuristic for being ``less stable``
        """
        return (
            -self.mismatches(),
            self.stemlen(),
            -self.looplen(),
            self.Start,
        ) < (
            -other.mismatches(),
            other.stemlen(),
            -other.looplen(),
            other.Start,
        )

    def stemlen(self):
        return len(self.Sequence)

    def looplen(self):
        return len(self.Spacer)

    def __len__(self):
        return 2 * self.stemlen() + self.looplen()

    def is_intersecting(self, other):
        return (self.Start <= other.Start < self.End) or (
            other.Start <= self.Start < other.End
        )

    @staticmethod
    def reverse_complement(seq: str) -> str:
        """
        Return reverse complement of nucleotide sequence
        """
        complement = {
            "A": "T",
            "G": "C",
            "T": "A",
            "C": "G",
        }
        return "".join([complement[s] for s in seq][::-1])

    @staticmethod
    def count_apobec(s: str) -> int:
        """
        Return number of TC/GA-sites (APOBEC-signature sites) in string
        """
        return s.count("GA") + s.count("TC")

    def mismatches(self) -> int:
        complement = {
            "A": "T",
            "G": "C",
            "T": "A",
            "C": "G",
        }
        mm_cnt = 0
        ln = self.stemlen()
        for i in range(ln):
            if self.Sequence[i] != complement[self.Opposite[ln - i - 1]]:
                mm_cnt += 1
        return mm_cnt

    def nn_energy(self) -> float:
        """
        Energy calculation based on NN model for DNA. Coefficients from NNDB https://rna.urmc.rochester.edu/NNDB/
        Only works for hairpins with loops <= 30 nt.
        """
        symmetry = 0.43  # only for linear structure
        init = 1.0

        linear_seq = self.Sequence + self.Spacer + self.Opposite
        linear_energy: float = 0
        for i in range(len(linear_seq) - 1):
            nn = linear_seq[i] + linear_seq[i + 1]
            linear_energy += self.NN_coefs[nn]
        if Hairpin.reverse_complement(linear_seq) == linear_seq:
            linear_energy += symmetry
        linear_energy += init

        pin_energy: float = 0
        seq = self.Sequence
        reverse_opposite = Hairpin.reverse_complement(self.Opposite)
        mm_positions = [
            i
            for i in range(self.stemlen())
            if self.Sequence[i] != reverse_opposite[i]
        ]
        for i in range(self.stemlen() - 1):
            if i in mm_positions:
                pass
            if i + 1 in mm_positions:
                pass
            else:
                nn = seq[i] + seq[i + 1]
                pin_energy += self.NN_coefs[nn]
        for mm_pos in mm_positions:
            pin_energy += init
            pin_energy += self.MM_coefs[
                f"{seq[mm_pos - 1] + seq[mm_pos + 1]}-{seq[mm_pos]}-{self.Opposite[-mm_pos-1]}"
            ]
        if self.looplen() > 1:
            pin_energy += self.terminal_MM_coefs[
                f"{seq[-1]}-{self.Spacer[0]}-{self.Spacer[-1]}"
            ]
        if self.looplen() > 2:
            pin_energy += self.init_coefs[str(self.looplen())]
        else:
            pin_energy += init  # pure ad-libbing, as NN model doesn't allow for <3 nt loops
        return 2 * pin_energy - linear_energy


class HairpinList(list):
    # subtracting 1 from genomic coordinates for 0-based indexing
    @classmethod
    def from_palindrome_analyzer(cls, filepath):
        hairpins = cls()
        with open(filepath, "r") as inp:
            inp.readline()
            for line in inp:
                info, pos, energy, seqs = line.split("\t")
                if len(seqs.split()) == 3:
                    seq, spacer, opp = seqs.split()
                else:
                    seq, opp = seqs.split()
                    spacer = ""
                hairpins.append(
                    Hairpin(int(pos) - 1, seq, spacer, opp, float(energy))
                )
        hairpins.sort()
        return hairpins

    @classmethod
    def from_emboss(cls, filepath, genome_seq: str):
        hairpins = cls()
        with open(filepath, "r") as inp:
            lines = inp.readlines()
            for i in range(12, len(lines), 4):
                if lines[i].strip():
                    pos, seq, looppos = lines[i].split()
                    end, opp, loopend = lines[i + 2].split()
                    hairpins.append(
                        Hairpin(
                            int(pos) - 1,
                            seq.upper(),
                            genome_seq[int(looppos) : int(loopend) - 1],
                            opp[::-1].upper(),
                        )
                    )
        hairpins.sort()
        return hairpins

    def connected_components(self) -> list[set]:
        """
        Splits hairpins into non-intersecting groups

        """
        components = [set()]
        rightest = 0
        for hairpin in sorted(self, key=lambda h: h.Start):
            if hairpin.Start >= rightest:
                components.append({hairpin})
                rightest = hairpin.End
            else:
                components[-1].add(hairpin)
                rightest = max(hairpin.End, rightest)
        components.pop(0)
        return components

    def most_stable_cover(self) -> HairpinList:
        """
        Select most stable (i.e. with minimal energy) hairpin from each component

        """
        cover = HairpinList()
        for component in self.connected_components():
            cover.append(min(component, key=lambda x: x.Energy))
        return cover

    def min_len_cover(self) -> HairpinList:
        """
        Select shortest hairpin from each component

        """
        cover = HairpinList()
        for component in self.connected_components():
            cover.append(
                min(
                    component,
                    key=lambda x: (len(x), x.Energy, x.Start),
                )
            )
        return cover

    def joint_max(self, key_func: callable) -> HairpinList:
        """
        Apply dynamic algorithm to select non-intersecting pins to maximize sum of their keys (known as weighted interval scheduling)

        :param key_func: weight function, e.g. lambda x: -x.Energy #no with energy it doesn't work, it prefers not to take anything!
        """
        hairpins = sorted(self, key=lambda h: h.End)

        def prev(
            ind: int,
        ) -> int:  # return index of last not intersecting hairpin
            lft = hairpins[ind].Start
            L = 0
            R = ind + 1
            while R - L > 1:
                mid = (R + L) // 2
                if hairpins[mid].End > lft:
                    R = mid
                else:
                    L = mid
            if hairpins[L].End <= lft:
                return L
            return -1

        dp = [0 for i in range(len(hairpins) + 1)]
        take_it = [False for i in range(len(hairpins) + 1)]
        for ind in range(1, len(hairpins) + 1):
            if (
                key_func(hairpins[ind - 1]) + dp[prev(ind - 1) + 1]
                > dp[ind - 1]
            ):
                dp[ind] = key_func(hairpins[ind - 1]) + dp[prev(ind - 1) + 1]
                take_it[ind] = True
            else:
                dp[ind] = dp[ind - 1]
        cover = HairpinList()
        ind = len(hairpins)
        while ind != 0:
            if take_it[ind]:
                cover.append(hairpins[ind - 1])
                ind = prev(ind - 1) + 1
            else:
                ind -= 1
        return cover

    def greedy_cover(self, key_func: callable) -> HairpinList:
        """
        For each component, iteratively select non-intersecting hairpins with min key func value

        :param key_func: weight function to minimize, e.g. lambda x: -x.looplen()
        """
        cover = HairpinList()
        for component in self.connected_components():
            while component:
                min_hairpin = min(component, key=key_func)
                cover.append(min_hairpin)
                component = {
                    h for h in component if not h.is_intersecting(min_hairpin)
                }
        return cover

    def recalculate_energy(self):
        """
        Recalculate and update Energy for each hairpin using nn_energy().
        """
        for hairpin in self:
            hairpin.Energy = hairpin.nn_energy()

    def to_palindrome_analyzer(self, filepath):
        """
        Save hairpins to file in PalindromeAnalyser format.
        Energy is recalculated using nn_energy().
        """
        with open(filepath, "w") as out:
            out.write("tosplit\tpos\tenergy\tseqs\n")
            for hairpin in sorted(self, key=lambda h: h.Start):
                tosplit = f"{hairpin.stemlen()}-{hairpin.looplen()}-{hairpin.mismatches()}"
                pos = hairpin.Start + 1  # convert to 1-based
                if hairpin.Spacer:
                    seqs = f"{hairpin.Sequence} {hairpin.Spacer} {hairpin.Opposite}"
                else:
                    seqs = f"{hairpin.Sequence} {hairpin.Opposite}"
                out.write(f"{tosplit}\t{pos}\t{hairpin.Energy:.2f}\t{seqs}\n")

    def restrict(
        self,
        stem_min: int = 0,
        stem_max: int = 10**6,
        loop_min: int = 0,
        loop_max: int = 10**6,
    ) -> HairpinList:
        """
        Filter hairpins by given range of stemlen and looplen
        """
        result = HairpinList()
        for hairpin in self:
            if (
                stem_min <= hairpin.stemlen() <= stem_max
                and loop_min <= hairpin.looplen() <= loop_max
            ):
                result.append(hairpin)
        return result
