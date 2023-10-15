import numpy as np

class WeightMatrix:
    def __init__(self, n_neurons, w_init = "zeros"):
        self.n_neurons = n_neurons
        if w_init == "zeros" or w_init == None:
            self.matrix = np.zeros(shape=(n_neurons, n_neurons))
        elif w_init == "mid" or w_init == None:
            self.matrix = np.zeros(shape=(n_neurons, n_neurons))+np.float16(0.5)
        elif w_init == "random":
            self.matrix = np.random.rand(n_neurons, n_neurons)
        else:
            e = "\n\n\tWeight init only takes 'zeros' or 'random'!\n\tDefault is zero.\n"
            raise Exception(e)
        
        pairs = [[p, p] for p in range(n_neurons)]
        for p1, p2 in pairs:
            self.matrix[p1, p2] = 0
        
        # Each neuron will be assigned an X (row) number for referencing the matrix

        # Find a way to update the weights and how the neuron signals propagate to each other

    def PrintMatrix(self):
        print(self.matrix)

class LIF:
    def __init__(self, neuron_id, T = 100) -> None:
        self.neuron_id = neuron_id
        # Define simulation parameters
        self.dT = 0.1  # Time step
        self.tau_m = 10.0  # Membrane time constant
        self.V_reset = 0.0  # Reset voltage
        self.V_threshold = 1.0  # Spike threshold

        self.V = list()
        self.spikes = list()
        self.spike_bool = False

    # Define a function to update the LIF neuron's state
    def update(self, current_input):
        if len(self.V) < 1:
            self.V.append((current_input - np.float16(0.000)) / self.tau_m)
        else:
            self.V.append((current_input - self.V[-1]) / self.tau_m)
        if self.V[-1] >= self.V_threshold:
            self.V[-1] = self.V_reset
            self.spikes.append(1)
            self.spike_bool = True
        else:
            self.spikes.append(0)
            self.spike_bool = False

class Network:
    def __init__(self, T = 100, n_neurons = 10, w_init = None) -> None:
        self.n_neurons = n_neurons
        self.T = T
        self.T_log = 0
        self.LIFNeurons = dict()
        self.weightsclass = WeightMatrix(10, w_init)
        self.weightmatrix = self.weightsclass.matrix
        pass

    def InitNetwork(self):
        for i in range(self.n_neurons):
            self.LIFNeurons[i] = LIF(i)
    
    def GetNeuronWeights(self, DEBUG):
        if DEBUG:
            self.weightsclass.PrintMatrix()
        #col = "  ".join([str(i) for i in range(self.n_neurons)])
        #print("  " + col)
        for row in range(self.n_neurons):
            print(f"{row} {self.weightmatrix[row, :]}")
    
    def PrepPropagation(self, neuron_id):
        """
        TODO
        """
        neighbor_weights = self.weightmatrix[neuron_id, :]
        neighbor_ids = [i for i in range(len(neighbor_weights))]

        # Delete itself from neighbor data
        del neighbor_ids[neuron_id]
        np.delete(neighbor_weights, neuron_id)

        return neighbor_ids, neighbor_weights

    """
    Brainstorming function of how the signal wave is spread.
    
    def WaveProp(self, ids, weights):
        for n_neu in self.LIFNeurons:
            for n_id in ids:
                if n_neu.neuron_id == n_id:
                    out_sig = np.float16(10.0) * weights[n_id]
                    # Calculate 
                    #new_weight =
    """

    def NetworkUpdate(self, except_id):
        for i in list(self.LIFNeurons.keys())
        
    def step(self, input_current = np.float16(0.000), input_neuron = 0):
        self.input_bool = True
        self.wave_dict = dict()
        self.spike_dict = dict()
        # Always do input neuron first.
        if input_current > np.float16(0.000) and self.input_bool:
            # Search for input neuron,
            # USING DICT NOT LIST TO SPEED UP SEARCH AND REDUCE LINES
            neu = self.LIFNeurons[input_neuron]
            neu.update(input_current)
            if neu.spike_bool:
                self.spike_dict[input_neuron] = True
                neighbor_ids, neighbor_ws = self.PrepPropagation(neu.neuron_id)
                for iter in range(len(neighbor_ids)):
                    self.wave_dict[neighbor_ids[iter]] = neighbor_ws
            # Step end
        else:
            ids = list(self.wave_dict.keys())
            for i in ids:
                out_sig = np.float16(10.0) * self.wave_dict[i]
                lif 
        self.T_log += 1

if __name__ == "__main__":
    snn = Network(w_init="mid")
    snn.InitNetwork()
    snn.GetNeuronWeights(DEBUG=False)
    snn.step(10, 0)
    # Generate the network
    