import numpy as np
import random
def leaky_relu(x, weight, alpha=0.01):
    """
    Compute the Leaky ReLU activation function.
    Parameters:
    x (numpy.ndarray): Input array.
    alpha (float): Slope for the negative range (default is 0.01).

    Returns:
    numpy.ndarray: Output array after applying Leaky ReLU.
    """
    x *= weight
    return np.maximum(alpha * x, x)

def generate_nested_dict(listA, listB):
    result_dict = {}

    for i, key in enumerate(listA):
        temp_dict = result_dict
        for j, inner_key in enumerate(listA):
            if i != j:  # Skip the key itself
                if inner_key not in temp_dict:
                    temp_dict[inner_key] = {}
                temp_dict[inner_key][key] = listB[i][j]
                temp_dict = temp_dict[inner_key]

    return result_dict
class Reservior:
    def __init__(self, n_neurons) -> None:
        self.n_neurons = n_neurons
        self.weight_table = dict()
        self.neuron_ids = list()
    def create(self):
        # Initialize neurons
        self.neuron_ids = [i for i in range(self.n_neurons)]
        weights = []
        for iter1 in range(self.n_neurons):
            temp = []
            for iter2 in range(self.n_neurons):
                if iter2 != iter1:
                    temp.append(random.uniform(0.0, 1.0))
            weights.append(temp)

        for iter, key in enumerate(self.neuron_ids):
            for value in weights:
                temp_dict = dict()
                for case, internal in enumerate(value):
                    if iter != case:
                        temp_dict[self.neuron_ids[case]] = internal

                self.weight_table[key] = temp_dict

    def step(self, x):
        for neuron in self.neuron_ids:
            links = self.weight_table[neuron]
            #print(links)
            output = leaky_relu(x, 0.9)
        return output

# Example usage:

LSM = Reservior(10)
LSM.create()
print(LSM.weight_table)
x = np.array([-1.0, 0.0, 1.0, 2.0])
output = LSM.step(x)
for n_w in output:
    print(n_w)
    print()
#print(output)