import numpy as np
import random
import copy
# Given 5 varibales
# S, CR, CE, T
# size = 0.1
# consumption rate = 0.1,
# conversion efficiency = 0.1,
# time_limit = 10 dT
# Generate pairs as quadruple helixes where there are no duplicate links
# Each variant can only have length of 50, where each var is an incremental

n_variants = 10
gene_len = 100
selection_set = ["S", "CR", "CE", "DP", "T"]
gene_pool = []
pair_inits = []

# Pregrenerate initial sequence for redundancy
for _ in range(n_variants):
    variant = []
    for i in range(gene_len):
        variant.append(random.choice(selection_set))
    pair_inits.append(variant)

# Generate non-element-wise duplicates
for i in pair_inits:
    pair = []
    for j in i:
        choice = random.choice(selection_set)
        # Create a clone of selection set
        redundancy_set = copy.copy(selection_set)
        if choice == j:
            # Remove the same element of j (pair_inits element)
            # from redundancy set
            redundancy_set.remove(choice)
            # Repeat the selection but using the redundancy set,
            # where the duplicate has been removed
            choice = random.choice(redundancy_set)
        # Reset the redundancy set to its former copy
        redundancy_set = copy.copy(selection_set)
        pair.append(choice)
        
    gene_pool.append([list(l) for l in zip(i, pair)])

for i in gene_pool:
    for j in i:
        print(j)
    print()