import cv2
from matplotlib import pyplot as plt
import numpy as np


def load_image(img_path: str, size: tuple):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, size)
    #img_size = img.shape()
    return img, size

def fourier_transform(img):
    og_fourier = np.fft.fftshift(np.fft.fft2(img))
    cloned_fourier = np.copy(og_fourier)
    return og_fourier, cloned_fourier

def plot_results(img, img_fourier, masked_image, masked_fourier, square_image, square_fourier):
    f_size = 13
    fig, ax = plt.subplots(3,2,figsize=(15,15))
    
    #   Original
    ax[0][0].set_title('Unmasked Image', fontsize = f_size)
    ax[0][0].imshow(img, cmap = 'gray')
    
    ax[0][1].set_title('Unmasked Fourier', fontsize = f_size)
    ax[0][1].imshow(np.log(abs(img_fourier)), cmap = 'gray')
    
    #   Masked
    ax[1][0].set_title('Masked Image', fontsize = f_size)
    ax[1][0].imshow(masked_image, cmap='gray')
    
    ax[1][1].set_title('Masked Fourier', fontsize = f_size)
    ax[1][1].imshow(masked_fourier, cmap='gray')
    
    ax[2][0].set_title('Center Square Image', fontsize = f_size)
    ax[2][0].imshow(square_image, cmap='gray')
    
    ax[2][1].set_title('Center Square Fourier', fontsize = f_size)
    ax[2][1].imshow(square_fourier, cmap='gray')

    plt.show()
    
def fourier_masker(img, i, size):
    img1 = np.copy(img)
    
    mask_y_length = (int(size[1]) // 2) - 5
    y_top_len = mask_y_length
    y_bot_len = mask_y_length - mask_y_length * 2
    
    x_pos_center = size[0] // 2
    x_thick = 2
    x_start = x_pos_center - x_thick
    x_end = x_pos_center + x_thick
    img1[:y_top_len, x_start:x_end] = i
    img1[y_bot_len:, x_start:x_end] = i

    # Inverse fourier transform
    masked_fourier = np.log(abs(img1))
    masked_image = abs(np.fft.ifft2(img1))

    img2 = np.copy(img)
    img2[250-5:-250+5, x_pos_center-5:x_pos_center+5] = 0
    square_fourier = np.log(abs(img2))
    square_img = abs(np.fft.ifft2(img2))
#    plt.imshow(square_img, cmap= "gray")
#    plt.show()
    return masked_image, masked_fourier, square_fourier, square_img
    
img_path = '../dataset\object_mode\Oyster\images\Subset0\P_20230708_181300.jpg'
img, size = load_image(img_path, (500, 500))
og_fourier, cloned_fourier = fourier_transform(img)
masked_img, masked_fourier, square_fourier, square_image = fourier_masker(cloned_fourier, 1, size)

plot_results(img, og_fourier, masked_img, masked_fourier, square_image, square_fourier)