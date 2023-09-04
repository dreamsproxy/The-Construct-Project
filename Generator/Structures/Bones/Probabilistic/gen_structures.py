import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.feature import shape_index
from skimage.draw import disk
from skimage import util
import cv2
from tqdm import tqdm
import os

def create_layer(image_size=256, spot_count=50, spot_radius=10, seed = True, shifted_coords = []):
    # Initalize with 2D noise
    # The generated noise will act as a
    # background that smoothens the gaussian
    # blur results allowing for smoother
    # transition between solids ```skimage.draw.disk()```
    rng = np.random.default_rng()
    image = rng.normal(
        loc=0.1,
        scale=0.15,
        size=(image_size, image_size)
    )
    
    # A temporary cache to hold the generated
    # coordinates
    coords = []
    
    # Determine whether the current
    # to-be-generated layer is the inital
    # layer.
    # 
    # This helps the processing pipeline to
    # determine wether to use an existing
    # layer or generate a new one as the
    # "Seed Layer"
    if seed:
        for _ in range(spot_count):
            # Randomly generate coordinates of x and
            # y where the centroids of the disks will be.
            disk_x = rng.integers(image.shape[0])
            disk_y = rng.integers(image.shape[1])

            # Coordinate return handler
            coords.append((disk_x, disk_y))

            # Generate the size (radius and centroid)
            # and fill with ones
            rr, cc = disk(
                (disk_x, disk_y),
                spot_radius,
                shape=image.shape
            )
            image[rr, cc] = 1

    elif len(shifted_coords)>0 and seed == False:
        for disk_x, disk_y in shifted_coords:
            coords.append((disk_x, disk_y))
            rr, cc = disk(
                (disk_x, disk_y),
                spot_radius,
                shape=image.shape
            )
            image[rr, cc] = 1

    # "De-normalize" to grayscale alpha format
    # Alpha values between 0.0 and 255.0
    image *= 255
    
    # Pre-processes the image:
    #   Pushes the alpha values above the mean to 255.0
    #   and alpha values below the mean to 0.0
    _, image = cv2.threshold(image, np.mean(image), 255, cv2.THRESH_BINARY)
    
    # Pre-processes the image: "Gaussian Blur"
    # Secondary smoothening of disk-to-disk
    # transitions
    image = ndi.gaussian_filter(image, sigma=3.3)
    
    # Pre-processes the image:
    #   Pushes the alpha values above the 240 to 255.0
    #   and alpha values below the mean to 0.0
    _, image = cv2.threshold(image, 240, 255, cv2.THRESH_BINARY)
    
    return image, coords

def limit_bounds(low_lim: int, high_lim: int, axis_value: int):
    if axis_value > high_lim:
        axis_value = high_lim

    elif axis_value < low_lim:
        axis_value = low_lim
    
    return axis_value

def pregrenerate_dirs(synthetic_dataset_dir: str, dataset_size: int):
    # Pregen dir paths to determine whether to
    # make the directories or not
    if not os.path.exists(synthetic_dataset_dir):
        os.makedirs(synthetic_dataset_dir)

    for i in range(dataset_size):
        variation_path = os.path.join(synthetic_dataset_dir, str(i), "/")
        if not os.path.exists(variation_path):
            os.mkdir(variation_path)

def gen_frames(n_frames = 4, spot_density = 200, spot_size = 10, shift = 5, image_size=256, output_folder = "./generated", save_centroids = False):
    frame_array = []
    if save_centroids:
        centroid_array = []

    image, coords = create_layer(
        spot_count=spot_density,
        spot_radius=spot_size,
        seed=True)
    frame_array.append(image)
    temp1 = [list(x) for x in coords]
    for i in range(len(temp1)):
        temp1[i].append(0)
    temp1 = [tuple(x) for x in temp1]
    if save_centroids:
        centroid_array.append(temp1)

    n_frames -= 1

    for frame_id in range(n_frames):
        shifted_coord_cache = []
        for x, y in coords:
            new_x = np.random.randint(x-shift, x+shift)
            new_x = limit_bounds(0, image_size, new_x)
            new_y = np.random.randint(y-shift, y+shift)
            new_y = limit_bounds(0, image_size, new_y)
            shifted_coord_cache.append((new_x, new_y))

        image, coords = create_layer(
            spot_count=spot_density,
            spot_radius=spot_size,
            seed=False,
            shifted_coords=shifted_coord_cache)
        frame_array.append(image)
        temp2 = [list(x) for x in shifted_coord_cache]
        for i in range(len(temp2)):
            temp2[i].append(frame_id+1)
        temp2 = [tuple(x) for x in temp2]
        if save_centroids:
            centroid_array.append(temp2)

    for i, f in enumerate(frame_array):
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        cv2.imwrite(os.path.join(output_folder, f"{i}.png"), f)

    if save_centroids:
        centroid_array = np.array(centroid_array)
        np.save(f"{output_folder}/centroids", np.array(centroid_array))

    return frame_array

counter = 0
spot = [200]
spot_sizes = [11]
n_variations = 8

synth_ds_dir = "generated"

pregrenerate_dirs(synth_ds_dir, n_variations)
for i in tqdm(range(n_variations)):
    _, og_shape = gen_frames(n_frames=8, shift = 7, output_folder = f"./dataset/{i}/", save_centroids=True)