import numpy as np
import tqdm
import cv2
import random
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

class Generator:
    def __init__(self, n_nodes, n_slices, complexity) -> None:
        self.n_nodes = n_nodes
        self.n_slices = n_slices
        self.complexity = complexity
        self.nodes = []
        self.roots = []
        self.branches = []
        self.full_tree = []
        #return self
    
    def generate_circle_coordinates(radius, z, num_points=10):
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        x_coordinates = radius * np.cos(angles)
        y_coordinates = radius * np.sin(angles)
        z_coordinates = np.asarray([z for i in range(num_points)])
        coordinates = np.column_stack((x_coordinates, y_coordinates, z_coordinates))
        return coordinates
    
    def normalize_coordinates(coordinates):
        max_radius = np.max(np.linalg.norm(coordinates[:, :3], axis=1))
        normalized_coordinates = coordinates / max_radius
        return normalized_coordinates
    
    def randomly_shift_coordinates(coordinates, max_shift):
        num_dimensions = coordinates.shape[1]
        random_shifts = np.random.uniform(-max_shift, max_shift, size=(coordinates.shape[0], num_dimensions))
        
        # Apply the random shift only to x and y (first two columns)
        random_shifts[:, 2] = 0  # Keep z coordinates unchanged
        shifted_coordinates = coordinates + random_shifts
        return shifted_coordinates
    
    def generate_slices(self):
        central_node = np.array((0.5, 0.5, 0.5))
        root_z = np.linspace(0.0, 0.5, num = self.n_slices)
        nodes_array = [x*self.complexity for x in range(1, self.n_nodes)]
        nodes_array = sorted(nodes_array, reverse=True)
        
        radius_array = np.linspace(0.1, 1.0, num=self.n_slices)
        radius_array[::-1].sort()
        for iter, z in enumerate(root_z):
            node_count = nodes_array[iter-1] * self.complexity
            print(z)
            print(radius_array[iter])
            circle = Generator.generate_circle_coordinates(radius_array[iter], z, nodes_array[iter-1])
            #inner = Generator.generate_circle_coordinates(radius_array[iter-1], z, nodes_array[iter-1])
            #circle = np.row_stack((outer, inner))
            circle = Generator.randomly_shift_coordinates(circle, max_shift = 0.5)
            circle = Generator.normalize_coordinates(circle)
            self.roots.append(circle)
        
        #self.roots = np.asarray(self.roots)
        return self.roots

gen = Generator(n_nodes = 10, n_slices = 10, complexity=12)
gen.generate_slices()
# Creating figure
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")

for slice in gen.roots:
    x = slice[:, 0]
    y = slice[:, 1]
    z = slice[:, 2]
    # Creating plot
    ax.scatter3D(x, y, z, color = "green")
plt.title("simple 3D scatter plot")
plt.show()