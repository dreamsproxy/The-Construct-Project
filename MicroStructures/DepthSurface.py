import numpy as np
import plotly.graph_objs as go
import cv2
from utils import ImageIOs

# Load the grayscale image
image_path = "./Segmentation/7C-11-41-21.png"
image = ImageIOs.load_image(image_path)

# Convert the image to a NumPy array
image_array = np.array(image)

# Create meshgrid for X and Y coordinates
x, y = np.meshgrid(np.arange(image_array.shape[1]), np.arange(image_array.shape[0]))

# Create a 3D surface plot with the grayscale values as Z-axis
fig = go.Figure(data=[go.Surface(z=image_array, colorscale='gray')])

# Set axis labels
fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Depth'))

# Set the layout of the 3D plot
fig.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=0.2)))

# Show the plot
fig.show()
