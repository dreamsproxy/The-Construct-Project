import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
from glob import glob

class Fourier:
    def __init__(self) -> None:
        pass
    def fourier_transform(self, img):
        img_fourier = np.fft.fftshift(np.fft.fft2(img))
        return img_fourier

    def fourier_edge(self, masked_fourier, img_size, square_size = 4):
        x_pos_center = img_size // 2
        y_pos_center = img_size // 2
        x_lim = (x_pos_center-square_size, x_pos_center+square_size)
        y_lim = (y_pos_center-square_size, y_pos_center+square_size)
        masked_fourier[y_lim[0]:y_lim[1], x_lim[0]:x_lim[1]] = 0
        img_fourier = np.log(abs(masked_fourier))
        img_edge = abs(np.fft.ifft2(masked_fourier))
        
        return img_fourier, img_edge

    def merge_arrays(self, array1, array2):
        array1 = array1 / 2.0
        array2 = array2 / 2.0
        new_image = np.add(array1, array2)
        return new_image

    def merge_edges(self, image_path, size, plot = False):
        images = []
        img = load_image(image_path, (size, size))
        images.append(img)
        fourier = self.fourier_transform(img)
        
        _, edge1 = self.fourier_edge(fourier, size, 1)
        _, edge2 = self.fourier_edge(fourier, size, 2)

        edge_1_2 = self.merge_arrays(edge1, edge2)
        avg_alpha = np.average(edge_1_2)
        median_alpha = np.median(edge_1_2)
        save_path = "./dataset/Biomorphic/structures-128/" + str(os.path.basename(image_path))
        cv2.imwrite(save_path, edge_1_2)
        #raise
        if plot:
            plt.title(f"Avg Alpha: {avg_alpha}\nMedian Alpha: {median_alpha}")
            plt.imshow(edge_1_2, cmap = "gray")
            plt.tight_layout()
            plt.show()

    def plot_variations(self, img_path):
        fig, ax = plt.subplots(3,3,figsize=(15,15))
        img, size = load_image(img_path, (1000, 1000))
        images = []
        imgiter = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        for iter in imgiter:
            fourier = self.fourier_transform(img)
            fourier, edge = self.fourier_edge(fourier, size, iter)
            images.append(edge)
        counter = 0
        
        for row in range(3):
            for col in range(3):
                ax[row][col].set_title(f"Size {counter+1*2}")
                ax[row][col].imshow(images[counter], cmap='gray')
                counter += 1
        #self.plot_variations(paths[0])
        #plt.show()
    
    def Extract(self, path, bulk = False):
        if bulk:
            for f in path:
                self.merge_edges(f, 128)
        else:
            self.merge_edges(path)

class KMeans:
    def __init__(self) -> None:
        pass

    def BilateralBlur(self, image, diameter, sigmaColor, sigmaSpace):
        blurred = cv2.bilateralFilter(image, diameter, sigmaColor, sigmaSpace)
        return blurred

    def Extract(self, path):
        image = load_image(path, size=(512, 512))
        image = self.BilateralBlur(image, 11, 41, 21)
        # Reshaping the image into a 2D array of pixels and 3 color values (RGB)
        pixel_vals = image.reshape((-1,1))
        pixel_vals = np.float32(pixel_vals)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
        
        # then perform k-means clustering with number of clusters defined as 3
        #also random centres are initially choosed for k-means clustering
        k = 7
        retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # convert data into 8-bit values
        centers = np.uint8(centers)
        segmented_data = centers[labels.flatten()]
        
        # reshape data into the original image dimensions
        segmented_image = segmented_data.reshape((image.shape))
        
        plt.imsave("7C-11-41-21.png", segmented_image, cmap="gray")
        #plt.show()

kmeansextractor = KMeans()
kmeansextractor.Extract("201.png")