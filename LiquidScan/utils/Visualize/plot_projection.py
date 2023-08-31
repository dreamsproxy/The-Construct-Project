import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from glob import glob
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import open3d as o3d
import cv2

def rotate_points(scatter_points, rotation_matrix):
    # Apply rotations to the scatter points
    rotated_points = []
    #print(rotation_matrix.shape)
    for point in scatter_points:
        #print(point)
        rotated_point = np.dot(point, rotation_matrix.T)
        #print(rotated_point)
        rotated_points.append(rotated_point)
    #print(rotated_points)
    rotated_points = np.array(rotated_points)
    
    return rotated_points

def translate_points(scatter_points, translation_vector):
    # Apply translations to the rotated points
    translated_points = []
    for point in scatter_points:
        #print(translation_vector)
        #print(point)
        translated_point = point + translation_vector
        translated_points.append(translated_point)
    translated_points = np.array(translated_points)
    return translated_points

def remove_outliers(scatter_points):
    z_scores = np.abs((scatter_points - scatter_points.mean(axis=0)) / scatter_points.std(axis=0))
    threshold = 5
    outlier_mask = (z_scores < threshold).all(axis=1)
    filtered_points = scatter_points[~outlier_mask]
    return filtered_points

def unpacker(data):
    data = data.split("\n")
    
    rotation_cache = data[0].replace("]]", "]").replace("[[", "[").replace("[ ", "[").split(", ")
    rotation = []
    for r in rotation_cache:
        r = r.replace("[", "").replace("]", "")
        r = r.split(" ")
        r = np.array([np.float32(x) for x in r if x != ''])
        rotation.append(r)
    rotation = np.array(rotation)

    translation_cache = data[1].replace("]]", "]").replace("[[", "[").replace("[ ", "[")
    translation_cache = translation_cache.replace("[", "").replace("]", "").replace(",", "").split(" ")
    #print(translation_cache)
    translation = np.array([np.float32(x) for x in translation_cache if x != ''])
    points_cache = data[2].replace("]]", "]").replace("[[", "[").replace("[ ", "[").split(", ")
    points = []
    for p in points_cache:
        p = p.replace("[", "").replace("]", "").replace("'", "")
        p = p.split(" ")
        p = [np.float32(x) for x in p if x != '']
        points.append(p)
    points = np.array(points)
    #print(points.shape)
    #for p in points:
    #    print(p)
    #    print(type(p))
    #    print(type(p[0]))
    
    return rotation, translation, points

def reduce_points(scatter_points):
    # Apply K-means clustering for point reduction
    #num_clusters = round(len(scatter_points)*0.1)
    num_clusters = 250
    kmeans = KMeans(n_clusters=num_clusters, n_init='auto')
    kmeans.fit(scatter_points)
    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    filtered_points = []
    for i in range(num_clusters):
        cluster_points = scatter_points[cluster_labels == i]
        if len(cluster_points) > 1:
            filtered_points.append(cluster_points)
    
    filtered_indices = []
    for i in range(num_clusters):
        cluster_indices = np.where(cluster_labels == i)[0]
        if len(cluster_indices) > 1:
            filtered_indices.extend(cluster_indices)
    filtered_array = np.delete(scatter_points, filtered_indices, axis=0)
    
    distances = np.linalg.norm(filtered_array, axis=1)
    threshold_distance = np.mean(distances[len(distances)//2:])
    filtered_indices = np.where(distances <= threshold_distance)[0]
    filtered_array = filtered_array[filtered_indices]
    print(f"Reduced points from {len(scatter_points)} to {len(filtered_array)}")
    return filtered_array

if __name__ == "__main__":
    
    
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.set_xlabel('X')
    #ax.set_ylabel('Y')
    #ax.set_zlabel('Z')
    #ax.set_box_aspect([1, 1, 1])

    img1 = cv2.imread("dataset/preprocessing/BRICK/IMAGES/P_20230605_181935.jpg")
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    #img = cv2.resize(img, (100, 100))
    img2 = cv2.imread("dataset/preprocessing/BRICK/IMAGES/P_20230605_181936.jpg")
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(img1, img2)
    plt.imshow(disparity,'gray')
    plt.show()
    
    raise

    scatter_files = glob("./brick/*.txt")
    x = []
    y = []
    z = []
    for file in scatter_files:
        with open(file, "r") as infile:
            data = infile.read()
        r, t, points = unpacker(data)
        #print(points.shape)
        # Perform translation
        points = reduce_points(points)
        #points = remove_outliers(points)
        print(points.shape)
        points = translate_points(points, t)
        #print(points.shape)
        points = rotate_points(points, r)
        #print(len(points))
        points = np.array(points)
        for xpos in points[:, 0]:
            x.append(xpos)
        for ypos in points[:, 1]:
            y.append(ypos)
        for zpos in points[:, 2]:
            z.append(zpos)
        #point_cloud = o3d.geometry.PointCloud()
        #point_cloud.points = o3d.utility.Vector3dVector(points)
    #o3d.visualization.draw_geometries([point_cloud])
    #break
    ax.scatter(x, y, z, c=z, cmap = "viridis")
    plt.show()  # Display the 3D plot
    #raise
