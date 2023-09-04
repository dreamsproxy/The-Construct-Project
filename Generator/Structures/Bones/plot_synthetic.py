import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import cv2
import os
import pandas as pd

# Step 1: Define a function to extract white parts from an image
def extract_white_parts(image_path):
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Threshold the image to extract white parts
    _, threshed_img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

    # Find coordinates of white parts
    white_coordinates = np.argwhere(threshed_img == 255)

    return white_coordinates

def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)

def search_closest(search_origin, coordinates):
    # Initialize variables to keep track of the closest coordinate and its distance
    closest_coordinate = None
    closest_distance = float('inf')  # Initialize with positive infinity
    # Iterate through the coordinates to find the closest one
    for coordinate in coordinates:
        distance = euclidean_distance(search_origin, coordinate)
        if distance < closest_distance:
            closest_distance = distance
            closest_coordinate = coordinate

    return closest_coordinate

# Step 2: Load a series of grayscale PNG images and sort them by filename
image_folder = "generated/0"
image_files = os.listdir(image_folder)
image_files = [f for f in image_files if f.endswith('.png')]
image_files.sort()  # Sort the filenames

# Extract white parts from images 0 to 4 and store them with their corresponding filename
white_coordinates_dict = {}
for i, image_file in enumerate(image_files[:5]):
    image_path = os.path.join(image_folder, image_file)
    white_coordinates = extract_white_parts(image_path)
    white_coordinates_dict[i] = white_coordinates

# Step 3: Create a DataFrame for the 3D scatter plot
data = []
for i in range(5):
    white_coords = white_coordinates_dict[i]
    for coord in white_coords:
        data.append([coord[1], coord[0], i])  # x, y, z

df = pd.DataFrame(data, columns=['x', 'y', 'z'])

# Step 4: Read line segment coordinates from a text file
line_segments = []
with open("generated/0/centroids.txt", "r") as file:
    data = file.read().split("\n")
    for line in data:
        points = [point.split(", ") for point in line.strip().replace("[(", "").replace(")]", "").split("), (")]
        for i, p in enumerate(points):
            for iter, item in enumerate(p):
                if len(item) != 0:
                    points[i][iter] = int(item)
        line_segments.append(points)

line_segments = line_segments[:-1]
line_segments = line_segments

fig = go.Figure()
"""
fig.add_trace(
    go.Scatter3d(
        x=df['x'],
        y=df['y'],
        z=df['z'],
        mode = "markers",
        marker=dict(size=2, opacity=0.1)
        )
    )"""

for i, sublist in enumerate(line_segments):
    for point in sublist:
        if i + 1 < len(line_segments):
            closest_coord = search_closest(point, line_segments[i+1])
            """fig.add_trace(go.Scatter3d(
                x=[point[0], closest_coord[0]],
                y=[point[1], closest_coord[1]],
                z=[point[2], closest_coord[2]],
                mode='lines+markers',
                line=dict(color = 'red', width = 5),
                marker=dict(size = 2, opacity=0.1)
            ))
            """
            fig.add_trace(
                go.Mesh3d(
                    #x=[point[0], closest_coord[0]],
                    #y=[point[1], closest_coord[1]],
                    #z=[point[2], closest_coord[2]],
                    x=[0, 0, 1, 1, 0, 0, 1, 1],
                    y=[0, 1, 1, 0, 0, 1, 1, 0],
                    z=[0, 0, 0, 0, 1, 1, 1, 1],
                    alphahull=5,
                    opacity=1.0,
                    color='cyan'))
    break

"""
fig = go.Figure(
    data=[
        go.Mesh3d(
            x=line_segments[0][0],
            y=line_segments[0][1],
            z=line_segments[0][2],
            alphahull=5,
            opacity=0.4,
            color='cyan')])
"""
# Update layout for better visibility
fig.update_layout(
    scene=dict(
        xaxis=dict(range=[-2, 2]),
        yaxis=dict(range=[-2, 2]),
        zaxis=dict(range=[-2, 2]),
    )
)

# Show the plot
fig.show()
