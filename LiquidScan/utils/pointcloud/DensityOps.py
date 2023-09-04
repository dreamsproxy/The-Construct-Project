import numpy as np
from tqdm import tqdm

class HelperOps:
    def normalize_to_range(self, values):
        min_val = min(values)
        max_val = max(values)
        normalized = [(val - min_val) / (max_val - min_val) for val in values]
        return normalized

    def filter_array(self, density_array, threshold = 10):
        filtered_points_data = [point_data for point_data in density_array if point_data[1] < threshold]
        filtered_point_cloud, filtered_num_neighbors, filtered_normalized_neighbors = zip(*filtered_points_data)
        return filtered_point_cloud, filtered_num_neighbors, filtered_normalized_neighbors


class Calculate:

    def VolumeDesnity(self, x, y, z, unit_size):
        # Determine the grid dimensions based on the range of coordinates
        x_min, x_max = min(x), max(x)
        y_min, y_max = min(y), max(y)
        z_min, z_max = min(z), max(z)

        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min

        # Calculate the number of grid cells in each dimension
        x_cells = int(np.ceil(x_range / unit_size))
        y_cells = int(np.ceil(y_range / unit_size))
        z_cells = int(np.ceil(z_range / unit_size))

        # Initialize a grid to count points in each cell
        grid = np.zeros((x_cells, y_cells, z_cells), dtype=int)

        # Populate the grid with point counts
        for point_x, point_y, point_z in zip(x, y, z):
            x_index = int((point_x - x_min) / unit_size)
            y_index = int((point_y - y_min) / unit_size)
            z_index = int((point_z - z_min) / unit_size)

            grid[x_index, y_index, z_index] += 1

        # Calculate the density for each cell (points per unit)
        density_per_unit = grid / unit_size**3

        return density_per_unit

    def find_closest_neighbors(self, point_cloud, radius):
        from scipy.spatial import cKDTree
        print("Searching neighbors")
        kdtree = cKDTree(point_cloud)
        num_neighbors = []
        for point in tqdm(point_cloud, total = len(point_cloud)):
            neighbors = kdtree.query_ball_point(point, radius)
            num_neighbors.append(len(neighbors))
        return num_neighbors
    
    def calc_density(self, point_cloud, radius):
        num_neighbors = self.find_closest_neighbors(point_cloud, radius)
        normalized_neighbors = HelperOps.normalize_to_range(num_neighbors)
        points_data = list(zip(point_cloud, num_neighbors, normalized_neighbors))
        points_data.sort(key=lambda x: x[1], reverse=True)
        return points_data
