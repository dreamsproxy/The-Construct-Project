import numpy as np
import pandas as pd
import random
import copy
import csv
class Pool:
    # Given 4 varibales
    # S, CR, CE, DP
    # size = 0.1
    # consumption rate = 0.1,
    # conversion efficiency = 0.1,
    # duplicate probability
    # Generate pairs as quadruple helixes where there are no duplicate links
    # Each variant can only have length of 50, where each var is an incremental
    def __init__(self, n_variants, gene_len, selection_set) -> None:
        self.n_variants = n_variants
        self.gene_len = gene_len
        self.selection_set = selection_set
        self.gene_pool = []
        self.pair_inits = []
        self.PoolDB = pd.DataFrame()
    
    def generate(self):
        # Pregrenerate initial sequence for redundancy
        for _ in range(self.n_variants):
            variant = []
            for i in range(self.gene_len):
                variant.append(random.choice(self.selection_set))
            self.pair_inits.append(variant)

        # Generate non-element-wise duplicates
        for i in self.pair_inits:
            pair = []
            for j in i:
                choice = random.choice(self.selection_set)
                # Create a clone of selection set
                redundancy_set = copy.copy(self.selection_set)
                if choice == j:
                    # Remove the same element of j (self.pair_inits element)
                    # from redundancy set
                    redundancy_set.remove(choice)
                    # Repeat the selection but using the redundancy set,
                    # where the duplicate has been removed
                    choice = random.choice(redundancy_set)
                # Reset the redundancy set to its former copy
                redundancy_set = copy.copy(self.selection_set)
                pair.append(choice)
                
            self.gene_pool.append([list(l) for l in zip(i, pair)])

    class Stats:
        def VariantGeneStats(variant):
            s = np.float16(0.00)
            cr = np.float16(0.00)
            ce = np.float16(0.00)
            dp = np.float16(0.00)
            for element in variant:
                if element == "S":
                    s = np.add(s, 0.001)
                elif element == "CR":
                    cr  = np.add(cr, 0.001)
                elif element == "CE":
                    ce = np.add(ce, 0.001)
                elif element == "DP":
                    dp = np.add(dp, 0.001)
            return s, cr, ce, dp

        def VariantEfficiency(diameter_nm, energy_consumption_rate, conversion_rate):
            # Calculate the volume of the cell (assuming it's a sphere)
            energy_per_volume = (4/3) * np.pi * (diameter_nm / 2)**3
            # Calculate the energy consumption per unit volume
            energy_per_volume = energy_consumption_rate
            # Calculate the conversion rate per unit volume
            conversion_per_volume = conversion_rate
            # Calculate the total efficiency per unit volume
            efficiency_per_volume = energy_per_volume * conversion_per_volume
            return efficiency_per_volume

    def ProcessData(self):
        for variant in self.gene_pool:
            # unpack pairs
            unpacked = []
            for pair in variant:
                for elm in pair:
                    unpacked.append(elm)
            variant = unpacked
            s, cr, ce, dp = self.Stats.VariantGeneStats(variant)
            cellular_efficiency = self.Stats.VariantEfficiency(s, cr, ce)

            variant_data = [s, cr, ce, dp, cellular_efficiency]

            variant_df = pd.DataFrame([variant_data])
            #print(variant_df)
            ##variant_df = pd.DataFrame(variant_data, )
            self.PoolDB = pd.concat([self.PoolDB, variant_df], ignore_index=True)
        self.PoolDB = self.PoolDB.rename(columns={
            0: 'diameter nm',
            1: 'energy consumption rate',
            2: 'converstion rate',
            3: 'dupliucate probability',
            4: 'total efficiency'})
    
    def WritePoolDB(self):
        
        self.PoolDB.to_csv("Pool.csv", sep=',', encoding='utf-8')


if __name__ == "__main__":
    n_variants = 10
    gene_len = 100
    # S, CR, CE, DP
    # size = 0.1
    # consumption rate = 0.1,
    # conversion efficiency = 0.1,
    # duplicate probability
    selection_set = ["S", "CR", "CE", "DP"]
    gene_pool = Pool(n_variants, gene_len, selection_set)
    gene_pool.generate()
    gene_pool.ProcessData()
    gene_pool.WritePoolDB()
