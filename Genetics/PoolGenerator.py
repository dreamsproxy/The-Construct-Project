import numpy as np
import pandas as pd
import random
import copy
import csv
import binascii
import hashlib

"""
TODO
Modify the code to use real ACTG and AGUC
Write the sequence analysis that searches a database:
    
ASSUMING THE DATA IS USING S, CE, CR, DP
    Since each SHA256 hash was generated using utf-8 encoded
    HEX concatenated with the original string parameters with
    delimiters removed.
    
    Search the database using the "total efficiency" column.
    
    Get the dbSize * 0.25 closest values that are HIGHER than
    the search variant's "total efficiency".
    
    Get the dbSize * 0.25 closest values that are LOWER than
    the search variant's "total efficiency".
    
    Compute the local distribution of the collected SeqLens


ASSUMING THE DATA IS USING ACTG AND ACUG
ASSUME DB SIZE IS 10,338+1
Step 1: Sort the DB from shortes to longest SeqLens

Step 2: Reduce the DataBase size using binary search
    (Reminder, individual sample data is not uniform.)
    Max index search limit = 2000 rows
    
    
    Q0 End = 10,338 * 0.25 -> 2,584.5
        -> Gaussian Round -> 2584
        
    Q1 End = 10,338 * 0.50 -> 5,169.0
        -> Gaussian Round -> 5169
        
    Q2 End = 10,338 * 0.75 -> 7,753.5
        -> Gaussian Round -> 7754

    Q3 End = 10,338 * 1.00 -> 10,338.0
        -> Gaussian Round -> 10338
    
    Check which index's SeqLen is lower than than the 
    SearchSeqLen.
        If the SearchSeqLen is below quarter's end index's
        SeqLen
            Break down the quarter into 2
            Q0 -> 2584 = 1292
            
            Get 6.25% + and - of the DB size -> 646.125
            -> Gaussian Round 646
            Where IDX 1292 to 1936
            And
            Where IDX 646 to 1292
            
            The total amount of selected indexes:
            1292 indexes
            
            Collect the indexes -> used in step 4

Global Wasserstein Distance of the search variant
Local Wasserstein Distance

Step 3: Sequence Search
    (Reminder, individual sample data is not uniform.)
    Access the sequences of individual indexes
    Calculate the avg SeqLen of collected sequences
    
    Assuming avg SeqLen is 3700103.4
    Use Gaussian Rounding
    -> 3700103
    
    Assuming maximum search kernel is
    -> 0.125 * 3700103 = 925025.75
    Use Gaussian Rounding
    -> 925026

    Sliding Variable Search Length for Cross Matching:
        Assuming the current SeqLen is 3134096
        Determine Variable Search Lengths by the difference
        of the avg SeqLen and current SeqLen
        
    Match sequences of collected indexes from step 2:
        Calculate the Local Wasserstein distabce
    
    
"""

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
            s = np.float32(0.00)
            cr = np.float32(0.00)
            ce = np.float32(0.00)
            dp = np.float32(0.00)
            for element in variant:
                if element == "S":
                    s = np.add(s, 0.001)
                elif element == "CR":
                    cr  = np.add(cr, 0.001)
                elif element == "CE":
                    ce = np.add(ce, 0.001)
                elif element == "DP":
                    dp = np.add(dp, 0.001)
            s = np.around(s, 4)
            cr = np.around(cr, 4)
            ce = np.around(ce, 4)
            dp = np.around(dp, 4)
            return s, cr, ce, dp

        def VariantEfficiency(diameter_nm: np.float32,
                              energy_consumption_rate: np.float32,
                              conversion_rate: np.float32):
            # Calculate the volume of the cell (assuming it's a sphere)
            energy_per_volume = (4/3) * np.pi * (diameter_nm / 2)**3
            # Calculate the energy consumption per unit volume
            energy_per_volume = np.float32(energy_consumption_rate)
            # Calculate the conversion rate per unit volume
            conversion_per_volume = conversion_rate
            # Calculate the total efficiency per unit volume
            efficiency_per_volume = energy_per_volume * conversion_per_volume
            efficiency_per_volume = np.around(efficiency_per_volume, 8)
            return efficiency_per_volume

    class Utils:
        def AssignUID(input_str: str):
            # prettify
            input_str = input_str.replace("[", "").replace("]", "").replace(", ", "").replace(".", "")
            # Convert bytes
            input_bytes = input_str.encode("utf-8")
            # Bytes to Hex then decode back to str
            output_data = binascii.hexlify(input_bytes).decode("utf-8")
            # Concat decoded hex string with prettified string
            output_data = input_str + output_data
            # Encode string back to bytes
            output_data = output_data.encode("utf-8")
            # Get sha256-UUID of conacted bytes
            # Then convert it back to read-able string
            output_data = hashlib.sha256(output_data).hexdigest()
            print(output_data)
            return output_data

        def UnpackHelix(variant: list):
            # unpack pairs
            unpacked = []
            for pair in variant:
                for elm in pair:
                    unpacked.append(elm)
            return unpacked
    
    def ProcessData(self):
        for variant in self.gene_pool:
            # unpack pairs
            variant = self.Utils.UnpackHelix(variant)
            # Count frequency of codes
            s, cr, ce, dp = self.Stats.VariantGeneStats(variant)
            cellular_efficiency = self.Stats.VariantEfficiency(s, cr, ce)

            # Assign an ID using the hex of the stats
            varaint_ID = self.Utils.AssignUID(str([s, cr, ce, dp, cellular_efficiency]))
            # Pack the variant's stats into a dataframe
            variant_data = [varaint_ID, s, cr, ce, dp, cellular_efficiency]
            variant_df = pd.DataFrame([variant_data])
            # Push current variant's data into main pool database
            self.PoolDB = pd.concat([self.PoolDB, variant_df], ignore_index=True)

        # Upon complete of all variants, rename the columns
        self.PoolDB = self.PoolDB.rename(columns={
            0: 'variant ID',
            1: 'diameter nm',
            2: 'energy consumption rate',
            3: 'converstion rate',
            4: 'dupliucate probability',
            5: 'total efficiency'})
    
    def WritePoolDB(self):
        self.PoolDB.to_csv("Pool.csv", sep=',', encoding='utf-8')


if __name__ == "__main__":
    n_variants = 100
    gene_len = 1000
    # S, CR, CE, DP
    # size = 0.1
    # energy consumed per volume
    # conversion efficiency
    # duplicate probability
    selection_set = ["S", "CR", "CE", "DP"]
    gene_pool = Pool(n_variants, gene_len, selection_set)
    gene_pool.generate()
    gene_pool.ProcessData()
    gene_pool.WritePoolDB()
