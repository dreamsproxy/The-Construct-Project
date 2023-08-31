import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from glob import glob
import cv2
from PIL import Image

dir =  "./extracted_data/*.cam"

class preprocess:
    def denorm_array(array, denorm_value, force_uint = True):
        min_value = np.min(array)
        max_value = np.max(array)
        scaling_factor = denorm_value / (max_value - min_value)
        if force_uint:
            denormalized_array = np.uint8((array - min_value) * scaling_factor)
            return denormalized_array
        
        else:
            denormalized_array = (array - min_value) * scaling_factor
            return denormalized_array
    
    def unpack_projection(prjection_arrray):
        x_array = []
        y_array = []
        z_array = []
        for x, y, z in prjection_arrray:
            x_array.append(x)
            y_array.append(y)
            z_array.append(z)
        x_array = np.array(x_array)
        y_array = np.array(y_array)
        z_array = np.array(z_array)
        
        return x_array, y_array, z_array

    def to_image(x, y, z, w, h):
        arr = np.random.randint(255, size=(144, 144), dtype=np.uint8)
        image = np.zeros((w, h))
                # Define the dimensions of the image
        width = max(x) + 1
        height = max(y) + 1

        # Create a blank image with white background
        image = Image.new('L', (width, height), color=255)

        # Iterate over the data and set pixel values
        for i in range(len(x)):
            pixel_value = int(z[i])
            image.putpixel((x[i], y[i]), pixel_value)
        
        image.show()
        #print(image.shape)
        cv2.imshow(image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        raise
        
    def projections(projection_array, width, height, convert_type = "image"):
        if convert_type == "image":
            x_array, y_array, z_array = preprocess.unpack_projection(projection_array)
            x_array = preprocess.denorm_array(x_array, width)
            y_array = preprocess.denorm_array(y_array, height)
            z_array = preprocess.denorm_array(z_array, 255)
            
            
            return x_array, y_array, z_array

        elif convert_type == "depth":
            x_array, y_array, z_array = preprocess.unpack_projection(projection_array)
            x_array = preprocess.denorm_array(x_array, width)
            y_array = preprocess.denorm_array(x_array, height)

            return x_array, y_array, z_array

        else:
            print("conversion_type was use but not properly specified")
            #print("No converstion_type specified")
            print("Fallback mode:\t'grayscale'")
            x_array, y_array, z_array = preprocess.unpack_projection(projection_array)
            x_array = preprocess.denorm_array(x_array, width)
            y_array = preprocess.denorm_array(x_array, height)
            z_array = preprocess.denorm_array(z_array, 255)

            return x_array, y_array, z_array

class depth:    
    def projection2depth(filename, width, height):
        projections = np.load(filename)
        
        x, y, z = preprocess.projections(projections, width, height)
        preprocess.to_image(x, y, z, width, height)
        print(x)
        print(y)
        print(z)
        raise
        #data = np.reshape(data, data_shape)
        #print(data)
        #print(data.shape)
        #data = np.expand_dims(data, data_shape)
        # Create an empty grayscale depth map
        depth_map = np.zeros(size, dtype=np.uint8)

        # Extract depth values
        depth_values = [d[2] for d in data]

        # Normalize depth values
        normalized_depth = (depth_values - np.min(depth_values)) / (np.max(depth_values) - np.min(depth_values))

        # Assign pixel intensities based on normalized depth values
        for i, d in enumerate(data):
            x, y, _ = d
            x_idx = int(x * 7)  # Scale x coordinate to fit within 0-7 range
            y_idx = int(y * 7)  # Scale y coordinate to fit within 0-7 range
            depth_map[y_idx, x_idx] = int(normalized_depth[i] * 255)

        # Display the grayscale depth map
        cv2.imshow('Depth Map', depth_map)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

depth.projection2depth("./brick/0-1-2.npy", 1000, 1000)

class cam_pos:
    def plot_from_dir(dir):
        cam_files = glob(dir)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        cam_data = []
        for file in cam_files:
            with open(file, "r") as infile:
                in_data = infile.read().split()
                in_data = " ".join(in_data[0:12])
                cam_data.append(in_data)

        for data in cam_data:
            # Extract camera parameters
            params = np.fromstring(data, sep=' ')
            position = params[:3]
            rotation_matrix = np.reshape(params[3:], (3, 3))
            #print(rotation_matrix)
            #raise
            # Calculate arrow direction
            arrow_direction = rotation_matrix[:, 2]  # Extract the third column of the rotation matrix

            # Plot the arrow
            ax.quiver(position[0], position[1], position[2],
                    arrow_direction[0], arrow_direction[1], arrow_direction[2],
                    color='r')

        # Set plot limits and labels
        ax.set_xlim([-10, 10])  # Adjust the plot limits as needed
        ax.set_ylim([-10, 10])  # Adjust the plot limits as needed
        ax.set_zlim([-10, 10])  # Adjust the plot limits as needed
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()

    def plot_from_memory(translation_matrix, rotation_matrix):
        # Calculate arrow direction
        arrow_direction = rotation_matrix[:, 2]  # Extract the third column of the rotation matrix
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')


        ax.quiver(translation_matrix[0], translation_matrix[1], translation_matrix[2],
                  arrow_direction[0], arrow_direction[1], arrow_direction[2],
                  color='r')
        
        # Set plot limits and labels
        ax.set_xlim([-10, 10])  # Adjust the plot limits as needed
        ax.set_ylim([-10, 10])  # Adjust the plot limits as needed
        ax.set_zlim([-10, 10])  # Adjust the plot limits as needed
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()