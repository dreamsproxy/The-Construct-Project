import numpy as np
import cv2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from glob import glob
from tqdm import tqdm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import os

G_H = 512
G_W = 512
num_points = G_H

def plot_from_dir(dir):
    x_grid = np.linspace(-10, 10, num_points)
    y_grid = np.linspace(-10, 10, num_points)
    z_grid = np.linspace(-10, 10, num_points)
    
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    z_mesh = np.meshgrid(x_grid, y_grid)

    cam_files = glob(dir)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    cam_data = []
    fnames = []
    # Create a figure
    fig = go.Figure()
    
    for file in cam_files:
        with open(file, "r") as infile:
            in_data = infile.read().split()
            in_data = " ".join(in_data[0:12])
            fname = str(file)
            cam_data.append(in_data)
            fnames.append(file)
    for data in cam_data:
        # Extract camera parameters
        params = np.fromstring(data, sep=' ')
        position = params[:3]*2
        rotation_matrix = np.reshape(params[3:], (3, 3))
        # Calculate arrow direction
        initial_direction = np.array([0, -1, 0])
        arrow_direction = rotation_matrix.dot(initial_direction)

        # Plot the arrow
        ax.quiver(position[0], position[1], position[2],
                arrow_direction[0], arrow_direction[1], arrow_direction[2],
                color='b')

        # Apply translation and rotation to the surface grid
        translated_mesh = np.dot(np.column_stack((x_mesh.flatten(), y_mesh.flatten(), np.zeros_like(x_mesh.flatten()))),
                                rotation_matrix) + position
        # Add the surface plot
        fig.add_trace(go.Surface(
            x=x_mesh,  # Or use translated_mesh[:, 0].reshape(x_mesh.shape) if using translated_mesh
            y=y_mesh,  # Or use translated_mesh[:, 1].reshape(x_mesh.shape) if using translated_mesh
            z=z_mesh,  # Replace with your z values
            surfacecolor = cv2.imread("./dataset/exterior_mode/images/" + str(os.path.basename(cam_files)[:-3]) + ".JPG"),  # Set the texture image data as the surface color
            colorscale='Viridis',  # Adjust the colorscale as needed
            showscale=False  # Hide the color scale legend
        ))


    # Set the layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),  # Hide x-axis labels and ticks
            yaxis=dict(visible=False),  # Hide y-axis labels and ticks
            zaxis=dict(visible=False),  # Hide z-axis labels and ticks
        )
    )
    # Show the plot
    fig.show()
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
    return cam_data


# Function to load and preprocess the images
def load_images(image_paths):
    images = []
    for path in tqdm(image_paths):
        img = cv2.imread(path)
        img = cv2.resize(img, (G_H, G_W))  # Reshape the images to a consistent size
        images.append(img)
    return np.array(images)

# Function to load the .cam files
def load_cam_files(cam_paths):
    cam_data = []
    for path in tqdm(cam_paths):
        with open(path, 'r') as file:
            cam = file.read()
            cam = cam.split("\n")[0]
            cam = cam.split(" ")
            cam = np.array([np.float32(x) for x in cam])
        cam_data.append(cam)
    return np.array(cam_data)

def train():
    # Load and preprocess the training data
    image_paths = glob("dataset/Oyster/JointNode/GroupC/*.jpg")[:-1]
    cam_paths = glob("./dataset/Oyster/JointNode/GroupC/extracted_data/*.cam")[:-1]

    X_train = load_images(image_paths)
    y_train = load_cam_files(cam_paths)

    # Define the model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(G_H, G_W, 3)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    # 256
    
    model.add(Conv2D(64, (3, 3)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    # 128

    model.add(Conv2D(128, (3, 3)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    # 64

    model.add(Conv2D(256, (3, 3)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    # 32
    
    model.add(Flatten())
    
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(12, activation = "linear"))

    # Compile and train the model
    adam_opt = Adam(
        learning_rate=0.0002,
        amsgrad=True,
        name="Adam_amsgrad",
    )

    model.compile(optimizer=adam_opt, loss='mse')
    model.fit(X_train, y_train, epochs=10, batch_size=1)

    model.save("GrC.h5")

    # Load and preprocess the new image
    #new_image_path = "dataset\Oyster\JointNode\GroupB\P_20230706_164142.jpg"
    #new_image_path = image_paths
    new_image = load_images([image_paths[-1]])

    # Predict the .cam data for the new image
    predicted_cam = model.predict(new_image)
    predicted_cam = "".join([str(x) for x in predicted_cam]).replace("[", "").replace("]", "")
    #predicted_cam = " ".join(predicted_cam)
    # Save the predicted .cam data to a file
    with open('predicted.cam', 'w') as file:
        file.write(str(predicted_cam))  # Assuming the .cam data is a single value

def inference(model_path, predict_image):
    model = load_model(model_path)
    image = load_images([predict_image])
    predicted_cam = model.predict(image)
    predicted_cam = "".join([str(x) for x in predicted_cam]).replace("[", "").replace("]", "")
    with open('predicted.cam', 'w') as file:
        file.write(str(predicted_cam))  # Assuming the .cam data is a single value

cam_data = plot_from_dir("dataset\exterior_mode\courtyard\cams\**") 


#inference("./GrC.h5", "")