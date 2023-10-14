import numpy as np
import plotly.graph_objs as go
import cv2
from utils import ImageIOs
import tqdm
image_utils = ImageIOs(verbose=False)


class DepthMap:
    def Surface(image_path = None) -> None:
        if image_path == None:
            # Load the grayscale image
            print("Image path was not supplied, using an existing example instead.")
            image_path = "./Segmentation/7C-11-41-21.png"

        image = image_utils.LoadImage(image_path)

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
    
    def PointCloud(image = None) -> None:
        if isinstance(image, str) and "/" in image_path:
            image = image_utils.LoadImage(image_path)
            if len(image)>= 4:
                # Load the grayscale image
                print("Image path was not supplied, using an existing example instead.")
                image_path = "./Segmentation/7C-11-41-21.png"
                image = image_utils.LoadImage(image_path)
        elif isinstance(image, np.ndarray):
            pass

        # Get the shape of the image
        height, width = image.shape

        # Create lists to store the coordinates
        x_coords = []
        y_coords = []
        z_coords = []

        progbar = tqdm.tqdm(total=height*width)
        # Iterate through each pixel and extract coordinates and grayscale intensity
        for y in range(height):
            for x in range(width):
                x_coords.append(x)
                y_coords.append(y)
                z_coords.append(image[y, x])
                progbar.update(1)
        progbar.close()
        # Create a scatter plot for the point cloud
        fig = go.Figure(data=[go.Scatter3d(x=x_coords, y=y_coords, z=z_coords,
                                        mode='markers',
                                        marker=dict(size=2, color=z_coords, colorscale='gray'))])

        # Set axis labels
        fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Intensity'))

        # Set the layout of the 3D plot
        fig.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=0.2)))

        # Show the point cloud plot
        fig.show()

if __name__ == "__main__":
    visualizer = DepthMap()
    visualizer.PointCloud()