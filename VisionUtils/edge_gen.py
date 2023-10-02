import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob

def get_range(threshold, sigma=0.33):
    return (1-sigma) * threshold, (1+sigma) * threshold


def edge_det(savedir, path):
    save_path = savedir + str(os.path.basename(path))
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.GaussianBlur(img, (3,3), 0)
    triangle_thresh, _ = cv2.threshold(img, 0, 255, cv2.THRESH_TRIANGLE)
    triangle_thresh = get_range(triangle_thresh)

    edge_triangle = cv2.Canny(img, *triangle_thresh)
    cv2.imwrite(save_path, edge_triangle)

paths = glob("./dataset/Biomorphic/256/**")

for f in paths:
    edge_det("./dataset/Biomorphic/edge-256/", f)