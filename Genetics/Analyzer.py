import numpy as np
import json
import os
import glob

class Codon:
    def __init__(self) -> None:
        self.CodonDict = None
        self.CodonJsonPath = "CodonDict.json"
        self.InitCodonDict()
        self.rnaUtils = RNA()
        pass
    def Setup(self):
        # Define a dictionary to map RNA sequences to codons
        rna_to_codon = {
            'UUU': 'Phenylalanine',
            'UUC': 'Phenylalanine',
            'UUA': 'Leucine',
            'UUG': 'Leucine',
            'UCU': 'Serine',
            'UCC': 'Serine',
            'UCA': 'Serine',
            'UCG': 'Serine',
            'UAU': 'Tyrosine',
            'UAC': 'Tyrosine',
            'UAA': 'Stop',
            'UAG': 'Stop',
            'UGU': 'Cysteine',
            'UGC': 'Cysteine',
            'UGA': 'Stop',
            'UGG': 'Tryptophan',
            'CUU': 'Leucine',
            'CUC': 'Leucine',
            'CUA': 'Leucine',
            'CUG': 'Leucine',
            'CCU': 'Proline',
            'CCC': 'Proline',
            'CCA': 'Proline',
            'CCG': 'Proline',
            'CAU': 'Histidine',
            'CAC': 'Histidine',
            'CAA': 'Glutamine',
            'CAG': 'Glutamine',
            'CGU': 'Arginine',
            'CGC': 'Arginine',
            'CGA': 'Arginine',
            'CGG': 'Arginine',
            'AUU': 'Isoleucine',
            'AUC': 'Isoleucine',
            'AUA': 'Isoleucine',
            'AUG': 'Methionine',
            'ACU': 'Threonine',
            'ACC': 'Threonine',
            'ACA': 'Threonine',
            'ACG': 'Threonine',
            'AAU': 'Asparagine',
            'AAC': 'Asparagine',
            'AAA': 'Lysine',
            'AAG': 'Lysine',
            'AGU': 'Serine',
            'AGC': 'Serine',
            'AGA': 'Arginine',
            'AGG': 'Arginine',
            'GUU': 'Valine',
            'GUC': 'Valine',
            'GUA': 'Valine',
            'GUG': 'Valine',
            'GCU': 'Alanine',
            'GCC': 'Alanine',
            'GCA': 'Alanine',
            'GCG': 'Alanine',
            'GAU': 'Aspartic Acid',
            'GAC': 'Aspartic Acid',
            'GAA': 'Glutamic Acid',
            'GAG': 'Glutamic Acid',
            'GGU': 'Glycine',
            'GGC': 'Glycine',
            'GGA': 'Glycine',
            'GGG': 'Glycine',
        }
        codon_json_obj = json.dumps(rna_to_codon, indent=4)
        with open(self.CodonJsonPath, "w") as outfile:
            outfile.write(codon_json_obj)

    def LoadCodonDict(self):
        with open(self.CodonJsonPath, "r") as readfile:
                self.CodonDict = json.load(readfile)
    
    def InitCodonDict(self):
        if not os.path.exists("CodonDict.json"):
            self.InitSetup()
            self.LoadCodonDict()
        else:
            self.LoadCodonDict()
    
    # Function to translate an RNA sequence to codons
    def Translate(self, rna_seq):
        if isinstance(rna_seq, str):
            self.rnaUtils.ReverseRNA(rna_seq)

        elif isinstance(rna_seq, list):
            rna_seq = "".join(rna_seq)
        else:
            raise NotImplementedError

        codons = []
        for i in range(0, len(rna_seq), 3):
            codon = rna_seq[i:i + 3]
            if codon in self.CodonDict:
                amino_acid = self.CodonDict[codon]
                codons.append((codon, amino_acid))
            else:
                codons.append((codon, 'Unknown'))
        return codons

class RNA:
    def __init__(self) -> None:
        pass

    def ReverseRNA(self, sequence: str):
        return sequence[::-1]

    def TranscribeFromDNA(self, dna_seq):
        if isinstance(dna_seq, str):
            dna_seq = dna_seq.replace("T", "U")
        elif isinstance(dna_seq, list):
            dna_seq = "".join(dna_seq)
            dna_seq = dna_seq.replace("T", "U")
        elif isinstance(dna_seq, bytes):
            raise NotImplementedError

# Example usage
codon_kit = Codon()
rna_sequence = "UCAUUUGUACCCGA"
translated_codons = codon_kit.Translate(rna_sequence)
for codon, amino_acid in translated_codons:
    print(f"Codon: {codon}, Amino Acid: {amino_acid}")
