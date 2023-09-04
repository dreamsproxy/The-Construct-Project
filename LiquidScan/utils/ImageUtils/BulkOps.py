from PIL import Image
import os
from tqdm import tqdm

def debug_handler(DEBUG):
    if DEBUG:
        
progbar = tqdm()
def modify_images(folder_path, DEBUG = False):
    # Get a list of all files in the folder
    files = os.listdir(folder_path)

    for file_name in files:
        # Check if the file is an image (you may want to add more checks)
        if file_name.endswith(('.jpg', '.jpeg', '.png', '.gif')):
            # Open the image file
            image_path = os.path.join(folder_path, file_name)
            image = Image.open(image_path)

            # Increase brightness by 25%
            brightness = 1.25
            modified_image = image.point(lambda p: p * brightness)

            # Reduce saturation by 15%
            saturation = 0.85
            modified_image = modified_image.convert('HSV')
            modified_image = modified_image.point(lambda p: p * saturation)
            modified_image = modified_image.convert('RGB')

            # Save the modified image
            new_file_name = f"modified_{file_name}"
            new_image_path = os.path.join(folder_path, new_file_name)
            modified_image.save(new_image_path)
            print(f"Modified image saved as {new_file_name}")
