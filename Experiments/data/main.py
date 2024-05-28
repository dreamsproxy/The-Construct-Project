import numpy as np
from numba import njit
import os
import plotly as plty

@njit
def selu_with_decay(x, alpha=1.67326, lambda_=1.0507, decay_rate=0.95):
    selu_output = lambda_ * np.where(x > 0, x, alpha * (np.exp(x) - 1))
    # Apply decay to negative values
    selu_output = np.where(x < 0, decay_rate * selu_output, selu_output)
    return selu_output

# Example usage
x = np.array([-1, 0, 1, -2, 3])
print("Input:", x)
print("SELU output with decay:", selu_with_decay(x))
