import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
from glob import glob

def load_image(img_path: str, size: tuple):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, size)
    #img_size = img.shape()
    return img

def fourier_transform(img):
    img_fourier = np.fft.fftshift(np.fft.fft2(img))
    return img_fourier

def fourier_edge(masked_fourier, img_size, square_size = 4):
    x_pos_center = img_size // 2
    y_pos_center = img_size // 2
    x_lim = (x_pos_center-square_size, x_pos_center+square_size)
    y_lim = (y_pos_center-square_size, y_pos_center+square_size)
    masked_fourier[y_lim[0]:y_lim[1], x_lim[0]:x_lim[1]] = 0
    img_fourier = np.log(abs(masked_fourier))
    img_edge = abs(np.fft.ifft2(masked_fourier))
    
    return img_fourier, img_edge

def merge_arrays(array1, array2):
    array1 = array1 / 2.0
    array2 = array2 / 2.0
    new_image = np.add(array1, array2)
    return new_image

def merge_1_5_10(image_path, size, plot = False):
    
    images = []
    img = load_image(image_path, (size, size))
    images.append(img)
    fourier = fourier_transform(img)
    
    _, edge1 = fourier_edge(fourier, size, 1)
    _, edge2 = fourier_edge(fourier, size, 2)

    edge_1_2 = merge_arrays(edge1, edge2)
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

def plot_variations(img_path):
    fig, ax = plt.subplots(3,3,figsize=(15,15))
    img, size = load_image(img_path, (1000, 1000))
    images = []
    imgiter = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for iter in imgiter:
        fourier = fourier_transform(img)
        fourier, edge = fourier_edge(fourier, size, iter)
        images.append(edge)
        #cv2.imshow(f"Size {iter}", edge)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    counter = 0
    
    for row in range(3):
        for col in range(3):
            ax[row][col].set_title(f"Size {counter+1*2}")
            ax[row][col].imshow(images[counter], cmap='gray')
            counter += 1


paths = glob("dataset/Biomorphic/128/**")
for f in paths:
    merge_1_5_10(f, 128)
#plot_variations(paths[0])
#plt.show()