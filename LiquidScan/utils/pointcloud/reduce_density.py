import tensorflow as tf
from tensorflow import keras
import numpy as np
import plotly.graph_objects as go
import sklearn.linear_model
from scipy.spatial import cKDTree
from tqdm import tqdm
import multiprocessing
import os


def find_closest_neighbors(point_cloud, radius):
    print("Searching neighbors")
    kdtree = cKDTree(point_cloud)
    num_neighbors = []
    for point in tqdm(point_cloud, total = len(point_cloud)):
        neighbors = kdtree.query_ball_point(point, radius)
        num_neighbors.append(len(neighbors))
    return num_neighbors

def normalize_to_range(values):
    min_val = min(values)
    max_val = max(values)
    normalized = [(val - min_val) / (max_val - min_val) for val in values]
    return normalized

def load_points(filepath, trim_value = 0):
    with open(filepath, "r") as infile:
        points_txt = infile.readlines()

    if trim_value > 0:
        points_txt = points_txt[:trim_value]
    else:
        pass

    point_cloud = []
    print("Processing Points")
    for point in tqdm(points_txt, total=len(points_txt)):
        if point != None:
            point = point.replace(" \n", "")
            point = point.split(" ")
            point_cloud.append(np.array(point[:3]))
    point_cloud = np.array(point_cloud)
    print(point_cloud.shape)
    print("Processing coords...")
    x_coords = [np.float32(p[0]) for p in point_cloud]
    y_coords = [np.float32(p[1]) for p in point_cloud]
    z_coords = [np.float32(p[2]) for p in point_cloud]
    print("Zipping coords...")
    point_cloud = np.array(list(zip(x_coords, y_coords, z_coords)))
    print(point_cloud.shape)
    return point_cloud

def calc_density(point_cloud, radius):
    num_neighbors = find_closest_neighbors(point_cloud, radius)
    normalized_neighbors = normalize_to_range(num_neighbors)
    normalized_neighbors = normalize_to_range(num_neighbors)
    points_data = list(zip(point_cloud, num_neighbors, normalized_neighbors))
    points_data.sort(key=lambda x: x[1], reverse=True)
    return points_data

def filter_array(density_array, threshold = 10):
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
    ), title='Filtered Point Cloud with Color-Coded Normalized using number of neighbors in the radius of 0.15')

    fig.show()

if __name__ == "__main__":
    file = "../dataset/Full Scans/MarketplaceFeldkirch_Station4_rgb_intensity-reduced.txt"

    cloud = load_points(file, trim_value = 0)
    #del cloud

    # Search radius
    radius = 0.15
    density_array = calc_density(cloud, radius)
    filter_pc, _, filter_neighbors = filter_array(density_array, 10)
    visualize_density(filter_pc, filter_neighbors)