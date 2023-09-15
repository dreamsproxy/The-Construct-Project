import numpy as np

class ArrayFunctions:
    def normalize(values):
        min_val = min(values)
        max_val = max(values)
        normalized = [(val - min_val) / (max_val - min_val) for val in values]
        return normalized

