import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
from glob import glob

class ArrayOps:
    def Normalize(arr):
        norm_arr = (arr-np.min(arr))/(np.max(arr)-np.min(arr))
        return norm_arr

class ImageIOs:
    def __init__(self, verbose = False) -> None:
        self.verbose = verbose
        pass
    def LoadImage(self, img_path: str, size = None, color_mode = "gray"):
        if size == None:
            size = (256, 256)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, size)
        if self.verbose:
            print(img.shape)

        return img

class ImageOps:
    def BilateralSmooth(image, diameter = 11, sigmaColor = 41, sigmaSpace = 21):
        blurred = cv2.bilateralFilter(image, diameter, sigmaColor, sigmaSpace)
        return blurred

class Segment:
    def __init__(self) -> None:
        self.imageio = ImageIOs(verbose=False)
    
    def kmeans(self, path):
        image = self.imageio.LoadImage(path, size=(512, 512))
        image = ImageOps.BilateralSmooth(image, 11, 41, 21)

        pixel_vals = image.reshape((-1,1))
        pixel_vals = np.float32(pixel_vals)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)

        k = 7
        retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        #og_centers = centers
        # convert data into 8-bit values
        centers = np.uint8(centers)
        segmented_data = centers[labels.flatten()]
        
        # reshape data into the original image dimensions
        segmented_image = segmented_data.reshape((image.shape))
        
        plt.imsave("result.png", segmented_image, cmap="gray")
        
        # Extract the individual slices (Clusters)
        # Create and save grayscale images for each cluster
        layer_slices = []
        unique_alphas = np.sort(np.unique(image))

        # Create and save grayscale images for each alpha value
        for alpha in unique_alphas:
            alpha_layer = np.zeros_like(image, dtype=np.uint8)
            alpha_layer[image == alpha] = 255  # Set pixels with the current alpha value to 255
            layer_slices.append(alpha_layer)
            #plt.imsave(f"alpha_{alpha}.png", alpha_layer, cmap="gray")
        return segmented_image, layer_slices