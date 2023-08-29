import os
import cv2
from skimage import io, transform

def create_flipped_images(input_folder, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all files in the input folder
    files = os.listdir(input_folder)

    for file in files:
        # Check if the file is an image (you can add more extensions if needed)
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            # Read the image using OpenCV
            img_cv2 = cv2.imread(os.path.join(input_folder, file), cv2.IMREAD_GRAYSCALE)
            
            cv2.imwrite(os.path.join(output_folder, f"{file}.png"), img_cv2)
            
            # Create a vertically flipped image using OpenCV
            img_vertical = cv2.flip(img_cv2, 0)

            # Save the vertically flipped image using OpenCV
            cv2.imwrite(os.path.join(output_folder, f"vflip_{file}.png"), img_vertical)

            # Create a horizontally flipped image using OpenCV
            img_horizontal = cv2.flip(img_cv2, 1)

            # Save the horizontally flipped image using OpenCV
            cv2.imwrite(os.path.join(output_folder, f"hflip_{file}.png"), img_horizontal)

            # Create a horizontally flipped image using OpenCV
            img_VH = cv2.flip(img_vertical, 1)

            # Save the horizontally flipped image using OpenCV
            cv2.imwrite(os.path.join(output_folder, f"vhflip_{file}.png"), img_VH)

if __name__ == "__main__":
    input_folder = "source/balanced"  # Replace with the path to your input folder
    output_folder = "source/augmented"  # Replace with the path to your output folder

    create_flipped_images(input_folder, output_folder)
