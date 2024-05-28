import numpy as np
from tqdm import tqdm
import ffmpeg
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
    def update(self, current_input = 0):
        # If the voltage log is empty, assume it is at 0.0, then perform calculation
        if len(self.V) < 1:
            delta_V = (current_input - np.float16(0.000)) / self.tau_m
            self.V.append(np.float16(0.000) + delta_V)
        else:
            delta_V = (current_input - self.V[-1]) / self.tau_m
            self.V.append(self.V[-1] + delta_V)

        if self.V[-1] >= self.V_threshold:
            self.V[-1] = self.V_reset
            self.spikes.append(1)
            self.spike_bool = True
        else:
            self.spikes.append(0)
            self.spike_bool = False

class Network:
    def __init__(self, n_neurons, w_init = None) -> None:
        self.n_neurons = n_neurons
        self.LIFNeurons = dict()
        self.weightsclass = WeightMatrix(n_neurons, w_init)
        self.weightmatrix = self.weightsclass.matrix
        self.wave_dict = dict()
        self.weight_log = []

    def InitNetwork(self):
        for i in range(self.n_neurons):
            #print(i)
            self.LIFNeurons[i] = LIF(i)
        print("InitNetwork -> neuron keys")
        print(list(self.LIFNeurons.keys()))
        print("\n")

    def GetNeuronWeights(self):
        for row in range(self.n_neurons):
            print(f"{row} {self.weightmatrix[row, :]}")
        print("\n")

    def PrepPropagation(self, neuron_id):
        neuron_keys = list(self.LIFNeurons.keys())
        neighbor_weights = self.weightmatrix[neuron_id, :]
        del neuron_keys[neuron_id]

        return neuron_keys, neighbor_weights

    def Decay(self, weight, factor):
        factor = np.float16(factor)
        weight -= factor * weight
        return weight

    #def UpdateWeights(self, method: str, n1: str, n2: str):
    def UpdateWeights(self, n1: str, n2: str):
        old_weight = self.weightmatrix[n1, n2]
        # TODO IMPLEMENT DIFFERENT WEIGHT UPDATE METHODS.
        #if method == "emulated":
        if old_weight < np.float16(1.000):
            new_weight = old_weight + np.float16(old_weight * np.float16(0.1))
            if new_weight >= np.float16(1.000):
                new_weight = np.float16(1.000)
                self.weightmatrix[n1, n2] = new_weight
            elif new_weight < np.float16(1.0):
                self.weightmatrix[n1, n2] = new_weight
        elif old_weight >= np.float16(1.000):
            new_weight = np.float16(1.000)
            self.weightmatrix[n1, n2] = new_weight

        #elif method == "hebbian":
            #weight_update += neu.activation * target_neuron.activation
            #self.weightmatrix[neu.neuron_id][target_neuron.neuron_id] = weight_update

    def NetworkUpdate(self, exception_list=[]):
        neuron_keys = list(self.LIFNeurons.keys())

        # Remove the firing neuron because
        if len(exception_list) > 0:
            for ex_neu in exception_list:
                neuron_keys.remove(ex_neu)

        for k in neuron_keys:
            neu = self.LIFNeurons[k]
            neu.update(0)

    def step(self, input_current = np.float16(0.000), input_neuron = 0):
        temp_wave = dict()
        if input_current > np.float16(0.000):
            neu = self.LIFNeurons[input_neuron]
            neu.update(input_current)
            if neu.spike_bool:
                neuron_keys, neighbor_ws = self.PrepPropagation(neu.neuron_id)
                for n in neuron_keys:
                    self.wave_dict[n] = neighbor_ws[n]
                    self.UpdateWeights(neu.neuron_id, n)
            self.NetworkUpdate(exception_list=[input_neuron])

        else:
            ids = list(self.wave_dict.keys())
            for i in ids:
                out_sig = np.float16(10.0) * self.wave_dict[i]
                neu = self.LIFNeurons[i]
                neu.update(out_sig)
                if neu.spike_bool:
                    neuron_keys, neighbor_ws = self.PrepPropagation(neu.neuron_id)
                    for n in neuron_keys:
                        temp_wave[n] = neighbor_ws[n]
                        #self.UpdateWeights(neu.neuron_id, n)
            self.wave_dict.clear()
            self.wave_dict = temp_wave
            self.NetworkUpdate(exception_list=ids)
        self.weight_log.append(np.copy(self.weightmatrix))

    def PlotWeightMatrix(self):
        import numpy as np
        import seaborn as sns
        import matplotlib.pyplot as plt
        from matplotlib import animation
        num_frames = len(self.weight_log)
        print(num_frames)
        
        # Set the background color for the plot
        sns.set(rc={'axes.facecolor':'#002439', 'figure.facecolor':'#002439'})

        # extract initial frame
        init_frame = self.weight_log[0]
        self.weight_log.pop(0)
        def init():
            # Initialize the heatmap (use the first frame as the initial state)
            heatmap = sns.heatmap(init_frame, square=True, cmap="mako", annot=True, annot_kws={'size': 8})
            heatmap.invert_yaxis()
            heatmap.set_xticklabels(heatmap.get_xticklabels(), color="white")
            heatmap.set_yticklabels(heatmap.get_yticklabels(), color="white")
            heatmap.set_title("Weight Matrix").set_color("white")
            heatmap.title.set_fontsize(20)

        fig = plt.figure()

        def animate(i):
            data = self.weight_log[i]
            sns.heatmap(data, square=True, cmap="mako", annot=True, annot_kws={'size': 8}, cbar=False)

        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=num_frames-1, repeat=True)
        save_prog = tqdm(total = num_frames)
        
        anim.save("mat.gif", fps=2, progress_callback=save_prog.update(1))

    
    def SaveWeightFrames(self):
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'axes.facecolor':'#002439', 'figure.facecolor':'#002439'})
        for step, frame in enumerate(self.weight_log):
            plt.clf()
            fig = sns.heatmap(frame, square=True, cmap="deep", annot=True, annot_kws={'size': 8})
            fig.invert_yaxis()
            fig.set_xticklabels(fig.get_xticklabels(), color="white")
            fig.set_yticklabels(fig.get_yticklabels(), color="white")
            
            fig.set_title("Weight Matrix").set_color("white")
            fig.title.set_fontsize(20)
            fig.figure.savefig(f"{step}.png", dpi = 1200)
            plt.close()  # Close the figure to release resources
    
        
    def PrintNetworkV(self):
        neuron_keys = list(self.LIFNeurons.keys())
        for i in neuron_keys:
            print(self.LIFNeurons[i].V)

    def PlotNetworkV(self):
        import plotly.graph_objects as go
        
        fig = go.Figure()
        for key, data in self.LIFNeurons.items():
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(data.V))),
                    y=data.V, mode='lines',
                    name=key))
        fig.update_layout(title='Membrane Potential Log',
                        xaxis_title='Ticks',
                        yaxis_title='Voltage')
        fig.show()


if __name__ == "__main__":
    snn = Network(n_neurons = 10, w_init="random")
    snn.InitNetwork()
    #snn.step(np.float16(10.0), 0)
    #raise
    for i in range(200):
        if i % 5 == 0:
            snn.step(np.float16(5.0), 0)
        elif i % 25 == 0:
            snn.step(np.float16(5.0), 0)
        elif i % 50 == 0:
            snn.step(np.float16(10.0), 0)
        else:
            snn.step(np.float16(0.000), 0)
    #snn.PrintNetworkV()
    #print(snn.LIFNeurons[0].V)
    #snn.PlotNetworkV()
    #snn.SaveWeightFrames()
    snn.PlotWeightMatrix()
    #print(snn.weight_log[0])
    #print()
    #print(snn.weight_log[-1])
    #print(len(snn.weight_log))
    #for key in list(snn.LIFNeurons.keys()):
    #    print(f"{key}\t{snn.LIFNeurons[key].spikes}")
    #print(snn.global_spike_log)
    #snn.GetNeuronWeights()
    #WeightMatrix.PrintMatrix()
