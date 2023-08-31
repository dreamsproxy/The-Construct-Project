import numpy as np
import matplotlib.pyplot as plt
import cv2

detector = cv2.SimpleBlobDetector_create()
params = cv2.SimpleBlobDetector_Params()

# Set Area filtering parameters
params.filterByArea = True
params.minArea = 200

# Set Circularity filtering parameters
params.filterByCircularity = True 
params.minCircularity = 0.01

# Set Convexity filtering parameters
params.filterByConvexity = True
params.minConvexity = 0.01

# Set inertia filtering parameters
params.filterByInertia = True
params.minInertiaRatio = 0.01

query_img = cv2.imread('dataset\preprocessing\BRICK\P_20230605_181851.jpg')
query_img = cv2.resize(query_img, (1000, 1000))
train_img = cv2.imread('dataset\preprocessing\BRICK\P_20230605_182026.jpg')
train_img = cv2.resize(train_img, (1000, 1000))

# Convert it to grayscale
query_img_bw = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)

blurquery1 = cv2.medianBlur(query_img_bw, 5)
blurquery2 = cv2.medianBlur(blurquery1, 5)
blurquery3 = cv2.medianBlur(blurquery2, 5)
blob_keypoints = detector.detect(blurquery3)

im_with_keypoints = cv2.drawKeypoints(query_img, blob_keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#cv2.imshow("Keypoints", im_with_keypoints)

query_canny = cv2.Canny(blurquery1, 0, 200)
train_canny = cv2.Canny(train_img_bw, 0, 100)
query_edges_rgb = cv2.cvtColor(query_canny, cv2.COLOR_GRAY2RGB)
dst = cv2.addWeighted(query_img,1.0,query_edges_rgb,0.5,0)