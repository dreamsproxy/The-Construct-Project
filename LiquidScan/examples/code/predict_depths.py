import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from glob import glob
from tqdm import tqdm
import json

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
files_list = glob(f"dataset/preprocessing/BRICK/IMAGES/*.jpg")

sift = cv2.SIFT_create(nfeatures = 4096)
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

focal_length_set = np.arange(4.0, 6.0, 0.5)
focal_length = np.mean(focal_length_set)

#files_list = files_list[:len(files_list)//4]
files_list = files_list[:16]

size = (1000, 1000)
principal_point_y = size[0]//2
principal_point_x = size[1]//2
traverse_index = 0
progbar = tqdm(total=len(files_list), leave = True)

def truebaseline(file_path, top_n):
    # Load the JSON file as a dictionary
    with open(file_path) as file:
        data = json.load(file)
    pair_data = []
    smallest_distance = float('inf')
    smallest_path = None
    for item in data.values():
        path = item['path']
        relations = item['relations']
        for relation in relations.values():
            trans_distance = float(relation['trans_distance'])
            if trans_distance < smallest_distance:
                smallest_distance = trans_distance
                smallest_path = relation['full path']
        pair_data.append((path, smallest_path, smallest_distance))

    return pair_data


while traverse_index < len(files_list):
    try:
        path1 = files_list[traverse_index]
        image1 = cv2.resize(cv2.imread(path1, 0), size)

        path2 = files_list[traverse_index+1]
        image2 = cv2.resize(cv2.imread(path2, 0), size)

        path3 = files_list[traverse_index+2]
        image3 = cv2.resize(cv2.imread(path3, 0), size)

        # Extract SIFT features and compute matches for the first image pair (1-2)
        keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

        matches_1_2 = bf.match(descriptors1, descriptors2)
        matches_1_2 = sorted(matches_1_2, key=lambda x: x.distance)

        # Extract SIFT features and compute matches for the second image pair (2-3)
        keypoints2, descriptors2 = sift.detectAndCompute(image2, None)
        keypoints3, descriptors3 = sift.detectAndCompute(image3, None)

        matches_2_3 = bf.match(descriptors2, descriptors3)
        matches_2_3 = sorted(matches_2_3, key=lambda x: x.distance)

        # Extract SIFT features and compute matches for the third image pair (1-3)
        keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
        keypoints3, descriptors3 = sift.detectAndCompute(image3, None)

        matches_1_3 = bf.match(descriptors1, descriptors3)
        matches_1_3 = sorted(matches_1_3, key=lambda x: x.distance)

        # Get the matched keypoints for the first image pair (1-2)
        matched_keypoints1_2 = np.float32([keypoints1[m.queryIdx].pt for m in matches_1_2]).reshape(-1, 1, 2)
        matched_keypoints2_2 = np.float32([keypoints2[m.trainIdx].pt for m in matches_1_2]).reshape(-1, 1, 2)

        # Get the matched keypoints for the second image pair (2-3)
        matched_keypoints2_3 = np.float32([keypoints2[m.queryIdx].pt for m in matches_2_3]).reshape(-1, 1, 2)
        matched_keypoints3_3 = np.float32([keypoints3[m.trainIdx].pt for m in matches_2_3]).reshape(-1, 1, 2)

        # Get the matched keypoints for the third image pair (1-3)
        matched_keypoints1_3 = np.float32([keypoints1[m.queryIdx].pt for m in matches_1_3]).reshape(-1, 1, 2)
        matched_keypoints3_3 = np.float32([keypoints3[m.trainIdx].pt for m in matches_1_3]).reshape(-1, 1, 2)

        #print(matched_keypoints1_2[0][0])
        #print(type(matched_keypoints1_2[0][0]))
        #raise

        m1_2 = matched_keypoints1_2.shape[0]
        m2_2 = matched_keypoints2_2.shape[0]
        m2_3 = matched_keypoints2_3.shape[0]
        m1_3 = matched_keypoints1_3.shape[0]
        m3_3 = matched_keypoints3_3.shape[0]
        compare_list = [m1_2, m2_2, m2_3, m1_3, m3_3]
        trim = min(compare_list)
        matched_keypoints1_2 = matched_keypoints1_2[:trim]
        matched_keypoints2_2 = matched_keypoints2_2[:trim]
        matched_keypoints2_3 = matched_keypoints2_3[:trim]
        matched_keypoints1_3 = matched_keypoints1_3[:trim]
        matched_keypoints3_3 = matched_keypoints3_3[:trim]

        #fundamental_matrix_2_3, mask_2_3 = cv2.findFundamentalMat(matched_keypoints2_3, matched_keypoints3_3, cv2.FM_RANSAC)

        # Filter keypoints based on the RANSAC mask
        #matched_keypoints2_3 = matched_keypoints2_3[mask_2_3.ravel() == 1]
        #matched_keypoints3_3 = matched_keypoints3_3[mask_2_3.ravel() == 1]
        fundamental_matrix_1_2, _ = cv2.findFundamentalMat(matched_keypoints1_2, matched_keypoints2_2, cv2.FM_8POINT)
        fundamental_matrix_2_3, _ = cv2.findFundamentalMat(matched_keypoints2_3, matched_keypoints3_3, cv2.FM_8POINT)
        fundamental_matrix_1_3, _ = cv2.findFundamentalMat(matched_keypoints1_3, matched_keypoints3_3, cv2.FM_8POINT)

        K = np.array([[focal_length, 0, principal_point_x],
                    [0, focal_length, principal_point_y],
                    [0, 0, 1]])

        essential_matrix_1_2 = np.dot(np.dot(K.T, fundamental_matrix_1_2), K)
        essential_matrix_2_3 = np.dot(np.dot(K.T, fundamental_matrix_2_3), K)
        essential_matrix_1_3 = np.dot(np.dot(K.T, fundamental_matrix_1_3), K)

        # Compute the camera matrices (rotation and translation)
        _, rotation_1_2, translation_1_2, mask_1_2 = cv2.recoverPose(essential_matrix_1_2, matched_keypoints1_2, matched_keypoints2_2)
        _, rotation_2_3, translation_2_3, mask_2_3 = cv2.recoverPose(essential_matrix_2_3, matched_keypoints2_3, matched_keypoints3_3)
        _, rotation_1_3, translation_1_3, mask_1_3 = cv2.recoverPose(essential_matrix_1_3, matched_keypoints1_3, matched_keypoints3_3)

        # Triangulate the 3D points using the camera matrices
        projection_matrix_1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        projection_matrix_2 = np.hstack((rotation_1_2, translation_1_2))
        projection_matrix_3 = np.hstack((rotation_1_3, translation_1_3))

        points_4d_12 = cv2.triangulatePoints(projection_matrix_1, projection_matrix_2, matched_keypoints1_2, matched_keypoints2_2)
        points_4d_12 /= points_4d_12[3]
        points_3d_12 = points_4d_12[:3].T

        points_4d_23 = cv2.triangulatePoints(projection_matrix_2, projection_matrix_3, matched_keypoints2_3, matched_keypoints3_3)
        points_4d_23 /= points_4d_23[3]
        points_3d_23 = points_4d_23[:3].T

        points_4d_13 = cv2.triangulatePoints(projection_matrix_1, projection_matrix_3, matched_keypoints1_3, matched_keypoints3_3)
        points_4d_13 /= points_4d_13[3]
        points_3d_13 = points_4d_13[:3].T

        data_12 = np.array2string(rotation_1_2).replace("\n ", ", ") + "\n"
        data_12 = data_12 + np.array2string(translation_1_2).replace("\n ", ", ") + "\n"
        data_12 = data_12 + str([np.array2string(line) for line in points_3d_12])

        data_23 = np.array2string(rotation_2_3).replace("\n ", ", ") + "\n"
        data_23 = data_23 + np.array2string(translation_2_3).replace("\n ", ", ") + "\n"
        data_23 = data_23 + str([np.array2string(line) for line in points_3d_23])

        data_13 = np.array2string(rotation_1_3).replace("\n ", ", ") + "\n"
        data_13 = data_13 + np.array2string(translation_1_3).replace("\n ", ", ") + "\n"
        data_13 = data_13 + str([np.array2string(line) for line in points_3d_13])

        with open(f"./brick/{traverse_index}-{traverse_index+1}.txt", "w") as outfile:
            outfile.write(data_12)

        with open(f"./brick/{traverse_index+1}-{traverse_index+2}.txt", "w") as outfile:
            outfile.write(data_23)

        with open(f"./brick/{traverse_index+2}-{traverse_index}.txt", "w") as outfile:
            outfile.write(data_13)
        
        traverse_index += 1
        progbar.update(1)
    except Exception as e:
        raise(e)
        #break

#with open("./brick/view_track.txt", "w") as outfile:
#    outfile.writelines(VIEW_TRACKER)