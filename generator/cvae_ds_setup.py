import os
import cv2
from glob import glob

# Define the folder containing the images
source_images_folder = 'img2img'
target_images_folder = "synthetic"

# Create target and source folders if they don't exist
source_folder = os.path.join(target_images_folder, 'source')
target_folder = os.path.join(target_images_folder, 'target')

os.makedirs(target_folder, exist_ok=True)
os.makedirs(source_folder, exist_ok=True)

# List all files in the images folder
files = glob(f"{source_images_folder}/**/*.png")

counter = 0
for file_name in files:
    id = int(file_name.split("\\")[-2])
    file_number = int(os.path.basename(file_name)[:-4])
    if file_number == 0:
        destination = os.path.join(source_folder, f"{counter}.png")
        print(file_name)
        img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(destination, img)
    else:
        destination = os.path.join(target_folder, f"{counter}.png")
        img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(destination, img)
        counter += 1
    
    print(f"Processed {file_name} to {destination}")

print("Done!")
