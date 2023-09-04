import os
import cv2
from glob import glob
from tqdm import tqdm

class DatasetOrganizor:
    def __init__(self, source_images_folder: str, target_images_folder: str):
        self.source_images_folder = source_images_folder
        self.target_images_folder = target_images_folder
        self.dataset_X = os.path.join(self.target_images_folder, 'x')
        self.dataset_Y = os.path.join(self.target_images_folder, 'y')
        return self
    
    def run(self):
        os.makedirs(self.dataset_Y, exist_ok=True)
        os.makedirs(self.dataset_X, exist_ok=True)
        files = glob(f"{self.source_images_folder}/**/*.png")
        counter = 0
        for file_name in tqdm(files):
            file_number = int(os.path.basename(file_name)[:-4])
            if file_number == 0:
                destination = os.path.join(self.dataset_X, f"{counter}.png")
                img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
                cv2.imwrite(destination, img)
            else:
                destination = os.path.join(self.dataset_Y, f"{counter}.png")
                img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
                cv2.imwrite(destination, img)
                counter += 1
        print("Done!")
