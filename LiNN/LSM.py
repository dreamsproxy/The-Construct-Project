import numpy as np

class NeuronLog:
    def __init__(self):
        self.log = []

    def log_neuron(self, neuron_id, activation):
        self.log.append((neuron_id, activation))

    def display_log(self):
        print("Neuron ID | Activation")
        print("----------------------")
        for neuron_id, activation in self.log:
            print(f"{neuron_id:9} | {activation}")

class LiquidNeuralNetwork:
    def __init__(self, num_inputs, num_neurons, num_connections):
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.num_connections = num_connections
        self.weights = np.random.rand(num_neurons, num_inputs)
        self.thresholds = np.random.rand(num_neurons)
        self.connections = np.random.choice([0, 1], size=(num_neurons, num_neurons), p=[1 - num_connections / num_neurons, num_connections / num_neurons])
        self.neuron_log = NeuronLog()

    def activate(self, input_data):
        weighted_sum = np.dot(self.weights, input_data)
        activations = (weighted_sum >= self.thresholds).astype(int)
        new_activations = np.dot(self.connections, activations)
        
        # Log neuron activations
        for neuron_id, activation in enumerate(new_activations):
            self.neuron_log.log_neuron(neuron_id, activation)
        
        return new_activations

# Example usage
num_inputs = 5
num_neurons = 10
num_connections = 3
input_data = np.random.rand(num_inputs)

liquid_nn = LiquidNeuralNetwork(num_inputs, num_neurons, num_connections)
output_activations = liquid_nn.activate(input_data)

# Display the neuron log
liquid_nn.neuron_log.display_log()

# Example usage
neuron_log = NeuronLog()
neuron_log.log_neuron(0, 0.75)
neuron_log.log_neuron(1, 0.92)
neuron_log.log_neuron(2, 0.61)
neuron_log.log_neuron(3, 0.83)

# Display the log
neuron_log.display_log()
"""
num_inputs represents the number of input neurons.
num_neurons represents the number of neurons in the liquid layer.
num_connections controls the sparsity of connections between neurons.
weights represent the connection weights between input and liquid neurons.
thresholds are the activation thresholds for liquid neurons.
connections define the connectivity between liquid neurons.

The activate method computes the activations of the liquid neurons based on the input data and the connectivity.
"""
