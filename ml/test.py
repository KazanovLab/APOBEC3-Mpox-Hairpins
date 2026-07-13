import RNA

def _reverse_complement(s):
    comp = {"A": "T", "T": "A", "G": "C", "C": "G",
            "a": "t", "t": "a", "g": "c", "c": "g"}
    return "".join(comp[b] for b in reversed(s))

def cruciform_hairpin_dG(seq, stem_len, loop_len):
    """ΔG cruciform forming (one leg cruciform) with DNA parameters.
    seq: [stem 5'][loop][stem 3'], length 2*stem_len + loop_len.
    """
    # structure: stem_len "(", loop_len "."", stem_len ")"
    structure = "(" * stem_len + "." * loop_len + ")" * stem_len
    assert len(seq) == len(structure), f"{len(seq)} != {len(structure)}"

    # load DNA-parameters (Mathews 2004), not RNA
    RNA.params_load_DNA_Mathews2004()

    fc = RNA.fold_compound(seq)
    dG = fc.eval_structure(structure)   
    return dG

def linear_duplex_dG(seq):
    """ΔG linear B-duplex sequence with complementary strand (DNA-parameters).
    seq: whole inverted repeat [stem 5'][loop][stem 3'].
    """
    RNA.params_load_DNA_Mathews2004()
    return RNA.duplexfold(seq, _reverse_complement(seq)).energy

def cruciform_relative_dG(seq, stem_len, loop_len, junction_penalty=0.0):
    """ΔΔG = 2*G_hairpin + G_junction - G_linear (two hairpins vs one duplex)."""
    G_hairpin = cruciform_hairpin_dG(seq, stem_len, loop_len)
    G_linear  = linear_duplex_dG(seq)
    return 2 * G_hairpin + junction_penalty - G_linear