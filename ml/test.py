import RNA   # pip install ViennaRNA

def cruciform_hairpin_dG(seq, stem_len, loop_len):
    """ΔG образования шпильки (одно плечо cruciform) с ДНК-параметрами.
    seq — участок: [стебель 5'][петля][стебель 3'], длиной 2*stem_len + loop_len.
    """
    # структура: stem_len открывающих скобок, loop_len точек, stem_len закрывающих
    structure = "(" * stem_len + "." * loop_len + ")" * stem_len
    assert len(seq) == len(structure), f"{len(seq)} != {len(structure)}"

    # загрузить ДНК-параметры (Mathews 2004), а не RNA
    RNA.params_load_DNA_Mathews2004()

    fc = RNA.fold_compound(seq)
    dG = fc.eval_structure(structure)   # ккал/моль для заданной структуры
    return dG

seq = "gatgtcgatagacatc"
looplen = 4
stemlen = 6

print(f"seq={seq},stemlen={stemlen},looplen={looplen}")
print(cruciform_hairpin_dG(seq,stemlen,looplen))

seq = "ttatatattctatataa"
looplen = 3
stemlen = 7

print(f"seq={seq},stemlen={stemlen},looplen={looplen}")
print(cruciform_hairpin_dG(seq,stemlen,looplen))

seq = "tatatcatcgatata"
looplen = 3
stemlen = 6

print(f"seq={seq},stemlen={stemlen},looplen={looplen}")
print(cruciform_hairpin_dG(seq,stemlen,looplen))
