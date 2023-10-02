import cv2
import os
from glob import glob

def convert_gray(savedir, path, size):
    save_path = savedir + str(os.path.basename(path))
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (size, size))
    cv2.imwrite(save_path, img)

paths = glob("./dataset/Biomorphic/Group A/**")

for f in paths:
    convert_gray("./dataset/Biomorphic/Group A/", f, 256)