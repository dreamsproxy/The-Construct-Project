import numpy as np
import matplotlib.pyplot as plt
import cv2
#from utils import cam_pos
from glob import glob
import os
import json
import tqdm
import pandas as pd

"""def show_matches(img1, img2):
    final_img = cv2.drawMatches(query_img,
                            queryKeypoints,
                            train_img,
                            trainKeypoints,
                            matches[:20],
                            None)
    final_img = cv2.resize(final_img, (1600,800))

    # Show the final image
    cv2.imshow("Feature Matches", final_img)
    cv2.waitKey(0)"""

def rotate_translate_points(i1, i2, translation, rotations):
    # Apply translation to i2
    i2_translated = i2 + translation

    # Apply rotations
    i2_transformed = i2_translated.copy()
    for rotation in rotations:
        rotation_matrix = euler_to_rotation_matrix(rotation)
        i2_transformed = np.dot(rotation_matrix, i2_transformed.T).T

    return i2_transformed

def euler_to_rotation_matrix(rotation):
    alpha, beta, gamma = rotation

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(alpha), -np.sin(alpha)],
                   [0, np.sin(alpha), np.cos(alpha)]])

    Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                   [0, 1, 0],
                   [-np.sin(beta), 0, np.cos(beta)]])

    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                   [np.sin(gamma), np.cos(gamma), 0],
                   [0, 0, 1]])

    rotation_matrix = np.dot(Rz, np.dot(Ry, Rx))
    return rotation_matrix


def estimate(img1_path, img2_path, n_features, focal_length, resize = False):
    query_img = cv2.imread(img1_path)
    train_img = cv2.imread(img2_path)
    size = (query_img.shape[0], query_img.shape[0])
    #if resize:
    query_img = cv2.resize(query_img, (1000, 1000))
    train_img = cv2.resize(train_img, (1000, 1000))

    query_img_bw = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)

    orb = cv2.SIFT_create(nfeatures = n_features)

    queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img,None)
    trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img,None)

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(queryDescriptors,trainDescriptors)
    matches = sorted(matches, key=lambda x: x.distance)
    #num_good_matches = int(len(distance_matches) * 0.1)
    #good_matches = matches[:num_good_matches]
    #depth = 1.0 - (len(good_matches) / num_good_matches)
    
    # Get matched keypoints
    matched_keypoints1 = np.float32([queryKeypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    matched_keypoints2 = np.float32([trainKeypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    m1 = matched_keypoints1.shape[0]
    m2 = matched_keypoints2.shape[0]
    compare_list = [m1, m2]
    trim = min(compare_list)
    matched_keypoints1 = matched_keypoints1[:trim]
    matched_keypoints2 = matched_keypoints2[:trim]
    
    # Calculate the fundamental matrix
    fundamental_matrix, _ = cv2.findFundamentalMat(matched_keypoints1, matched_keypoints2, cv2.FM_RANSAC)

    principal_point_y = size[0]//2
    principal_point_x = size[1]//2
    
    # Step 3: Estimate the essential matrix
    camera_matrix = np.array([[focal_length, 0, principal_point_x],
                            [0, focal_length, principal_point_y],
                            [0, 0, 1]])

    #essential_matrix, _ = cv2.findEssentialMat(matched_keypoints1, matched_keypoints2, camera_matrix, cv2.RANSAC, 0.999, 1.0)
    essential_matrix = np.dot(np.dot(camera_matrix.T, fundamental_matrix), camera_matrix)
    
    # Step 4: Decompose the essential matrix
    retval, rotation, translation, mask = cv2.recoverPose(essential_matrix, matched_keypoints1, matched_keypoints2, camera_matrix)
    #print(rotation)
    #print(translation)
    #origin_pos = np.hstack((np.eye(3), np.zeros((3, 1))))
    
    return translation, rotation, fundamental_matrix, essential_matrix

def t_preprocessor(t_array, focal_length):
    translation = str(t_array).replace("\n", "")
    translation = translation.replace("  ", " ")
    translation = translation.replace("[", "")
    translation = translation.replace("]", "")
    translation = (translation).split(" ")
    translation = [np.float32(x) for x in translation if x != '']
    #emulated_distance = np.linalg.norm(translation)
    #true_distance = emulated_distance * focal_length
    
    return translation

def r_preprocessor(r_array):
    rotation = str(r_array).replace("\n", "")
    rotation = rotation.replace("  ", " ")
    rotation = rotation.replace("[[ ", "[[")
    rotation = rotation.replace("] [", "], [")
    return rotation

def preprocess_pandas(filepath1, filepath2, t, r, fundamental_matrix, essential_matrix, projection_matrix):
    pass

def estimate_first(dir):
    image_paths = glob(dir)
    #image_paths = image_paths[:len(image_paths)//4]
    image_paths = image_paths

    tr_dict = {}
    focal_length_set = np.arange(4.0, 6.0, 0.5)
    focal_length = np.mean(focal_length_set)
    pbar = tqdm.tqdm(total=len(image_paths), leave = True)
    #dataframe = pd.DataFrame()
    cols = ["correlation", "translation", "rotation", "fundamental matrix", "essential matrix"]
    data = []
    
    for idx, image in enumerate(image_paths):
        temp_dict = {}
        relation_dict = {}
        if str(image_paths[0]) != str(image_paths[idx]):
            t, r, f_matrix, e_matrix = estimate(image_paths[0], image_paths[idx], 2048, focal_length, resize = False)
            t = t.tolist()
            r = r.tolist()
            f_matrix = f_matrix.tolist()
            e_matrix = e_matrix.tolist()

            data.append([image_paths[idx].replace("\\", "/"), t, r, f_matrix, e_matrix])
        pbar.update(1)
    data = pd.DataFrame(data, columns=cols)
    #df = estimate_first("dataset/preprocessing/BRICK/IMAGES/*.jpg")
    save_name = f"{os.path.basename(image_paths[0])}.csv"
    data.to_csv(save_name, index=False)
    return data

def estimate_dir(dir, n_features, resize_bool, trim_dataset = 0):
    
    if trim_dataset >=1:
        files_list = glob(dir)[:trim_dataset]
    else:
        files_list = glob(dir)
        
    tr_dict = {}
    focal_length_set = np.arange(4.0, 6.0, 0.5)
    focal_length = np.mean(focal_length_set)
    pbar = tqdm.tqdm(total=len(files_list), leave = True)
    
    for root_idx, img1 in enumerate(files_list):
        #print(img1)
        temp_dict = {}
        temp_dict['path'] = img1.replace("\\", "/")
        relation_dict = {}
        for sub_idx, img2 in enumerate(files_list):
            if str(img1) != str(img2):
                t, r, focal_length, proj_mat = estimate(img1, img2, n_features, focal_length, resize=resize_bool)
                t, emulated_dist, true_dist = t_preprocessor(t, focal_length)
                r = r_preprocessor(r)
                
                relation_dict[sub_idx] = {
                        'full path' : img2.replace("\\", "/"),
                        'translation' : str(t),
                        'rotation' : str(r).replace("\n", ""),
                        'virtual baseline' : str(emulated_dist),
                        'true baseline' : str(true_dist),
                        'projection matrix' : str(proj_mat)
                    }

                temp_dict['relations'] = relation_dict
                #pbar.update(1)
        break
        pbar.update(1)
        tr_dict[root_idx] = temp_dict
    
    return tr_dict, focal_length

def sort_transdistance(json_data):
    if isinstance(json_data, str):
        if json_data[-5:] == ".json":
            # Load the JSON file as a dictionary
            with open(json_data) as file:
                data = json.load(file)
                for item in data.values():
                    relations = item['relations']
                    sorted_relations = sorted(relations.items(), key=lambda x: float(x[1]['trans_distance']))
                    item['relations'] = dict(sorted_relations)
        return data
    
    elif isinstance(json_data, dict):
        # Sort the 'relations' nested list by 'trans_distance' value for each 'path'
        for item in data.values():
            relations = item['relations']
            sorted_relations = sorted(relations.items(), key=lambda x: float(x[1]['trans_distance']))
            item['relations'] = dict(sorted_relations)
            
    return data

import json

def get_smallest_truebaseline(file_path):
    # Load the JSON file as a dictionary
    with open(file_path) as file:
        data = json.load(file)

    pair_data = []
    
    # Iterate over the first layer's data to find the smallest 'trans_distance' and its 'full_path'
    smallest_distance = float('inf')
    smallest_path = None
    for item in data.values():
        path = item['path']
        relations = item['relations']
        for relation in relations.values():
            trans_distance = float(relation['true baseline'])
            if trans_distance < smallest_distance:
                smallest_distance = trans_distance
                smallest_path = relation['full path']
        pair_data.append((path, smallest_path, smallest_distance))

    # Update the data dictionary with the smallest 'trans_distance' and its 'full_path'
    #data['smallest_trans_distance'] = smallest_distance
    #data['smallest_full_path'] = smallest_path

    # Return the updated data
    return pair_data


def create_depthmap(pair_list):
    depth_data = []
    for pair in pair_list:
        # Load the two closest images
        img1 = cv2.cvtColor(cv2.resize(cv2.imread(pair[1]), (800, 800)), cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(cv2.resize(cv2.imread(pair[0]), (800, 800)), cv2.COLOR_BGR2GRAY)

        # Convert the images to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Create a StereoBM object with specified parameters
        stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

        # Compute the disparity map
        disparity = stereo.compute(gray1, gray2)

        # Normalize the disparity map for visualization
        disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Display the disparity map
        cv2.imshow('Disparity Map', disparity_normalized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    
    return depth_data

if __name__ == "__main__":
    """
    relative_dict, focal_length = estimate_dir(
        "dataset/preprocessing/BRICK/IMAGES/*.jpg",
        n_features = 2048,
        resize_bool = True,
        trim_dataset = 20
    )
    """
    df = estimate_first("dataset/preprocessing/BRICK/IMAGES/*.jpg")
    #df.to_csv("test.csv")
    # 
    # json_obj = json.dumps(relative_dict, indent=4)
    
    #with open("./BRICK.json", "w") as jfile:
    #    jfile.write(json_obj)
    
    #del relative_dict
    #del json_obj
    """
    focal_length = np.mean(np.arange(4.0, 6.0, 0.2))
    smallest_pairs = get_smallest_truebaseline("./BRICK.json")
    depthmaps = create_depthmap(smallest_pairs, focal_length)
    print(depthmaps[0].max())
    depthmaps[0] = depthmaps[0] * 255.0/depthmaps[0].max()
    plt.imshow(depthmaps[0], cmap="gray")
    plt.show()
    cv2.imshow("depth map", depthmaps[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """