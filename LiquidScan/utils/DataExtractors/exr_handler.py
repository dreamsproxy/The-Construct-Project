
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

import cv2

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from glob import glob
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import minimize

def convert_pngs(depth_exr_dir):
    files_list = glob(depth_exr_dir)
    for exr_file in tqdm(files_list):
        img = cv2.imread(exr_file, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
        img = (img-np.min(img))/(np.max(img)-np.min(img))*255
        #print(list(img))
        #raise
        #img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        save_path = os.path.basename(exr_file)[:-4] + str(".png")
        #print(save_path)
        cv2.imwrite(f"./valve_handle/pngs/{save_path}", img)

def preprocess_depth_coords(img):
    x = np.linspace(0, img.shape[1]-1, img.shape[1])
    y = np.linspace(0, img.shape[0]-1, img.shape[0])
    X, Y = np.meshgrid(x, y)
    #print(data)
    #plt.imshow(data, cmap="gray")
    #plt.show()
    #Z = img.flatten()
    Z = np.flipud(img.flatten())
    Z = -Z
    #print(Z)
    X = X.flatten()
    Y = Y.flatten()
    
    #print("Number for points:", len(Z)*len(X)*len(Y))
    
    return X, Y, Z

def plot_depth(X, Y, Z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', label="Fused")
    ax.scatter(X, Y, Z, s=6, c = Z, cmap='gray', alpha = 1, edgecolors = "face")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Depth')
    plt.show()

def objective(p):
    alpha, beta, gamma, tx, ty, tz = p
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(alpha), -np.sin(alpha)],
                    [0, np.sin(alpha), np.cos(alpha)]])

    R_y = np.array([[np.cos(beta), 0, np.sin(beta)],
                    [0, 1, 0],
                    [-np.sin(beta), 0, np.cos(beta)]])

    R_z = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                    [np.sin(gamma), np.cos(gamma), 0],
                    [0, 0, 1]])

    transformed_coords = np.dot(R_z, np.dot(R_y, np.dot(R_x, np.vstack((X2, Y2, Z2))))).T
    transformed_X2, transformed_Y2, transformed_Z2 = transformed_coords[:, 0], transformed_coords[:, 1], transformed_coords[:, 2]
    transformed_X2 += tx
    transformed_Y2 += ty
    transformed_Z2 += tz

    valid_indices = indices.squeeze()
    matching_points = np.column_stack((transformed_X2[valid_indices], transformed_Y2[valid_indices], transformed_Z2[valid_indices]))
    error = np.mean(np.linalg.norm(matching_points - np.column_stack((X1[valid_indices], Y1[valid_indices], Z1[valid_indices])), axis=1))
    return error

png_dir = "valve_handle/pngs/good"
png_files = glob(os.path.join(png_dir, "*.png"))
num_images = len(png_files)
image_paths = sorted(png_files)
#convert_pngs("valve_handle/depthEXR/*.exr")
#raise
img1 = cv2.imread(image_paths[0], cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
scale_percent = 5
width = int(img1.shape[1] * scale_percent / 100)
height = int(img1.shape[0] * scale_percent / 100)
dim = (width, height)
img1 = cv2.resize(img1, dim)
X1, Y1, Z1 = preprocess_depth_coords(img1)

# Initialize arrays to store concatenated XYZ data
X = X1
Y = Y1
Z = Z1

final_X = np.empty(0)
final_Y = np.empty(0)
final_Z = np.empty(0)

matched_X = np.empty(0)
matched_Y = np.empty(0)
matched_Z = np.empty(0)


# Perform nearest neighbor matching and non-linear optimization for the remaining images
for i in tqdm(range(1, num_images)):
    img_path = image_paths[i]
    img2 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
    img2 = cv2.resize(img2, dim)
    X2, Y2, Z2 = preprocess_depth_coords(img2)
    X = np.concatenate((X, X2))
    Y = np.concatenate((Y, Y2))
    Z = np.concatenate((Z, Z2))

    # Perform nearest neighbor matching between the two point clouds
    nbrs = NearestNeighbors(n_neighbors=1).fit(np.column_stack((X1, Y1, Z1)))
    distances, indices = nbrs.kneighbors(np.column_stack((X2, Y2, Z2)))

    # Perform non-linear optimization to estimate the transformation parameters
    res = minimize(objective, x0=np.zeros(6), method='Powell')
    final_parameters = res.x

    # Apply the final transformation to the second scatter plot
    alpha, beta, gamma, tx, ty, tz = final_parameters
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(alpha), -np.sin(alpha)],
                    [0, np.sin(alpha), np.cos(alpha)]])

    R_y = np.array([[np.cos(beta), 0, np.sin(beta)],
                    [0, 1, 0],
                    [-np.sin(beta), 0, np.cos(beta)]])

    R_z = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                    [np.sin(gamma), np.cos(gamma), 0],
                    [0, 0, 1]])

    transformed_coords = np.dot(R_z, np.dot(R_y, np.dot(R_x, np.vstack((X2, Y2, Z2))))).T
    transformed_X2, transformed_Y2, transformed_Z2 = transformed_coords[:, 0], transformed_coords[:, 1], transformed_coords[:, 2]
    transformed_X2 += tx
    transformed_Y2 += ty
    transformed_Z2 += tz
    valid_indices = indices.squeeze()
    matching_points = np.column_stack((transformed_X2[valid_indices], transformed_Y2[valid_indices], transformed_Z2[valid_indices]))
    matched_X = np.concatenate((matched_X, matching_points[:, 0]))
    matched_Y = np.concatenate((matched_Y, matching_points[:, 1]))
    matched_Z = np.concatenate((matched_Z, matching_points[:, 2]))

    final_X = np.concatenate((final_X, transformed_X2))
    final_Y = np.concatenate((final_Y, transformed_Y2))
    final_Z = np.concatenate((final_Z, transformed_Z2))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.scatter(X1, Y1, Z1, c=Z1, cmap='viridis', label='Img1 Only')
ax.scatter(final_X, final_Y, final_Z, c=final_Z, cmap='viridis', label='Merged Point Cloud')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(matched_X, matched_Y, matched_Z, c=matched_Z, cmap='gray', label='Matched Point Cloud')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()

