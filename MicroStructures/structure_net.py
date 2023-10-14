import cv2
import numpy as np
import plotly.graph_objects as go
from glob import glob
import Visualize
from utils import ArrayOps, ImageOps, ImageIOs, Segment
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import os

class Probabilistic:
    def __init__(self) -> None:
        pass

class Reconstruct:
    def docs(self):
        print("""
        Idea 1:
        Use K-Means segmentation as a base sample
        K-means segments will return a n_cluster layers
        Train a U-Net or C-VAE for image to image translation where pair A and B image,
        A is input image and B is target image
        
        Idea 2:
        Utilizing the n_clusters in kmeans segments, we can artificially create a multi-slice model
        where the alpha values will be marked as a specific layer
        by extracting individual layers, there should be a way to triangulate the n_cluster layers
        in such way to regenerate the 3D model in point cloud form.
        
        Idea 3:
        The bruteforce method
        Take a sample
        Using the sample as the base depth-image or depth-seed
        We will use a VAE to generate the best image given an image
        Similar to an Adversial Network, the VAE's performance is evaluated by probabiliities
        where the next slice of the depth map will be verified against the base sample
        in order to determine plausability that it is in fact a good next-slice.
        
        This function will first utilize the W-DCGAN-GP instead of an VAE for now.
        """)
    

    class Idea1:
        
        def prep_dataset(input_dir, output_dir):
            """
            Use K-Means segmentation as a base sample
            K-means segments will return a n_cluster layers
            Train a U-Net or C-VAE for image to image translation where pair A and B image,
            A is input image and B is target image
            """
            seg_util = Segment()
            imageio = ImageIOs()
            imageops = ImageOps()

            input_dir = glob("./Segmentation/")
            for img_path in input_dir:
                outname = img_path.replace("./", "")
                if ".jpg.png" in img_path:
                    outname = outname.replace(".jpg", "")
                    #outname = os.path.join(output_dir, outname)

                image, slices = seg_util.kmeans(img_path)

                for i in range(len(slices) - 1):
                    pair = [slices[i], slices[i + 1]]
                    
                    x_out = os.path.join(output_dir, "x", outname)
                    y_out = os.path.join(output_dir, "y", outname)
                    cv2.imwrite(x_out, pair[0])
                    cv2.imwrite(y_out, pair[1])
                    #pairs.append(pair)

        #print(image)
        #print(type(image))

    #def Idea2(self, path):

if __name__ == "__main__":
    r = Reconstruct()
    r.ArtificialDepth("./Segmentation/201.png")
