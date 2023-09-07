"""
Suppose you are given an image of a sponge or porous rock
You'd like to generate the artificial structure om 3D based on the
solid and void structure. How?

Idenity all joint positions
Determine curve using probability based on parameters such as:
    structure_type: (organic, artificial)
    n_voids: int
    n_void_volume: float
    
    OR
    
    structure_type: (organic, artificial)
    density or weight limit: float
    
use a CVAE to regenerate the image of the other end of the structure

"""
import cv2
import numpy as np
import plotly.graph_objects as go
from glob import glob

def plot_structures(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.GaussianBlur(img, (3,3), 0)
    
    #median = np.median(img)
    #threshold = np.int16(median * 0.5)
    #threshold = median
    #img[img < threshold] = 0
    
    coord_x = [i for i in range(0, 255)]
    coord_y = [i for i in range(0, 255)]
    coord_z = img
    

    fig = go.Figure(go.Scatter3d(
        x = coord_x,
        y = coord_y,
        z = coord_z,
        mode='markers'
        ))

    fig.update_layout(
            scene = {
                "xaxis": {"nticks": 20},
                "zaxis": {"nticks": 4},
                'camera_eye': {"x": 0, "y": -1, "z": 0.5},
                "aspectratio": {"x": 1, "y": 1, "z": 0.2}
            })
    fig.show()

paths = [
    "dataset/Biomorphic/grayscale-256/A60.jpg",
    #"dataset/Biomorphic/grayscale-256/11.jpg",
    #"dataset/Biomorphic/grayscale-256/13.jpg",
    "dataset/Biomorphic/grayscale-256/35.jpg"
]

for i in paths:
    print(i)
    plot_structures(i)