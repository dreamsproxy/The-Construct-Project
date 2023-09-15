from tqdm import tqdm
import numpy as np
import plotly.graph_objects as go
import sklearn.linear_model
from scipy.spatial import cKDTree
from tqdm import tqdm
import multiprocessing
import os
from GeneralOps import ArrayFunctions

class Load:
    def npy(filepath, trim_value = 0):
        with open(filepath, "r") as infile:
            points_txt = infile.readlines()
        if trim_value > 0:
            points_txt = points_txt[:trim_value]
        else:
            pass
        print("This Operation will take some time!")
        point_cloud = []
        print("Processing Points")
        for point in tqdm(points_txt, total=len(points_txt)):
            if point != None:
                point = point.replace(" \n", "")
                point = point.split(" ")
                point_cloud.append(np.array(point[:3]))
        point_cloud = np.array(point_cloud)
        print("Processing coords...")
        print(f"Old Shape: {point_cloud.shape}")
        x_coords = [np.float32(p[0]) for p in point_cloud]
        y_coords = [np.float32(p[1]) for p in point_cloud]
        z_coords = [np.float32(p[2]) for p in point_cloud]
        print("Zipping coords...")
        point_cloud = np.array(list(zip(x_coords, y_coords, z_coords)))
        print(f"New Shape: {point_cloud.shape}")
        return point_cloud

class ReduceCloudDensity:
    def __init__(self, search_radius: float, point_cloud_array: np.array) -> None:
        self.radius = search_radius
        self.point_cloud_array = point_cloud_array

    def SearchNeighbors(self):
        print("Searching neighbors")
        kdtree = cKDTree(self.point_cloud_array)
        num_neighbors = []
        for point in tqdm(self.point_cloud_array, total = len(self.point_cloud_array)):
            neighbors = kdtree.query_ball_point(point, self.radius)
            num_neighbors.append(len(neighbors))
        return num_neighbors

    def NeighborDensity(self):
        num_neighbors = self.SearchNeighbors(self.point_cloud_array, self.radius)
        normalized_neighbors = ArrayFunctions.normalize(num_neighbors)
        points_data = list(zip(self.point_cloud_array, num_neighbors, normalized_neighbors))
        points_data.sort(key=lambda x: x[1], reverse=True)
        return points_data

    def FilterPoints(density_array, threshold = 10):
        filtered_points_data = [point_data for point_data in density_array if point_data[1] < threshold]
        filtered_point_cloud, filtered_num_neighbors, filtered_normalized_neighbors = zip(*filtered_points_data)
        return filtered_point_cloud, filtered_num_neighbors, filtered_normalized_neighbors

    def visualize_density(filtered_point_cloud, filtered_normalized_neighbors):
        fig = go.Figure(data=go.Scatter3d(
            x=[point[0] for point in filtered_point_cloud],
            y=[point[1] for point in filtered_point_cloud],
            z=[point[2] for point in filtered_point_cloud],
            mode='markers',
            surfacecolor = "black",
            marker=dict(
                size=2,
                color=filtered_normalized_neighbors,  # Color points based on normalized neighbors
                colorscale='Viridis',                 # Choose the colorscale (you can change this)
                opacity=1.0,
                colorbar=dict(title='Normalized Num Neighbors')
            )
        ))
        fig.update_layout(scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
        ), title='Filtered Point Cloud')

        fig.show()

    def scrub(self):
        del self.point_cloud_array
        raise

    def run(self, visualize = False, save_npy = True):
        file = "../dataset/Full Scans/MarketplaceFeldkirch_Station4_rgb_intensity-reduced.txt"

        cloud = Load.npy(file, trim_value = 0)

        points_density_array = self.NeighborDensity(cloud, self.radius)
        filtered_point_cloud_array, _, filter_neighbors = self.FilterPoints(points_density_array, 10)
        if visualize:
            self.visualize_density(filtered_point_cloud_array, filter_neighbors)
        if save_npy:
            np.save("neighbor_density.npy", points_density_array)
            np.save("filtered_point_cloud.npy", filtered_point_cloud_array)
            np.save("removed_points.npy", filter_neighbors)
