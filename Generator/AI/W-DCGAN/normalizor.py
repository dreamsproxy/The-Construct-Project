import os
import cv2
import numpy as np
from skimage import exposure

def balance_brightness_contrast(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    for file_name in image_files:
        # Read the image using OpenCV
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name).replace(".jpg", ".png")

        image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        # Balance brightness and contrast using skimage
        high, low = np.percentile(image, (2, 98))
        image = exposure.rescale_intensity(image, in_range=(high, low))

        # Save the processed image
        cv2.imwrite(output_path, image)

        print(f"Processed: {file_name}")

def normalize_brightness(input_folder, output_folder):
    # List all files in the input folder
    image_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    brightness_log = []
    for file_name in image_files:
        # Read the image using OpenCV
        input_path = os.path.join(input_folder, file_name)

        image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        avg_alpha = np.average(image)
        brightness_log.append(avg_alpha)
        #print(avg_alpha)

    # Calculate global mean and median
    global_mean_alpha = np.mean(brightness_log)
    gloabl_median_alpha = np.median(brightness_log)
    if global_mean_alpha > gloabl_median_alpha:
        global_mean_alpha = gloabl_median_alpha
    
    for file_name in image_files:
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name).replace(".jpg", ".png")
        image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        
        image = image/global_mean_alpha
        factor = global_mean_alpha / np.mean(image)
        image = image * factor
        
        # Save the processed image
        cv2.imwrite(output_path, image)

        print(f"Processed: {file_name}")

if __name__ == "__main__":
    input_folder = "source/jpg"  # Change this to your input folder containing the images
    output_folder = "source/balanced"  # Change this to your output folder
    normalize_brightness(input_folder, output_folder)
    balance_brightness_contrast(input_folder, output_folder)
    
