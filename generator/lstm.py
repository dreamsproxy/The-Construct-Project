import os
import numpy as np
import tensorflow as tf
from glob import glob
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
from tqdm import tqdm
def handle_multi_paths(filepath_list):
    select_options = []
    print("Multiple shape files were detected!")
    for i, path in enumerate(filepath_list):
        select_options.append(i)
        print(f"[{i}]\t{path}")
    
    user_decision = input("Which file do you want to use?")
    if any(option in user_decision for option in select_options):
        confirm = str(input("Are you sure? [1/0]"))
        if confirm == "1":
            return filepath_list[user_decision]
        else:
            pass
    else:
        UserWarning("Input was neither 1 or 0. Falling back to use first path")
        return None

def search_shapetxt(directory_path):
    located = False
    shape_file = glob("./*shape*.txt")

    if len(shape_file) == 1:
        with open(shape_file[-1], "r") as infile:
            original_shape = infile.read().replace(")", "").replace("(", "")
            original_shape = tuple([int(x) for x in original_shape.split(", ")])
        return original_shape
    elif len(shape_file >= 2):
        user_selection = handle_multi_paths(shape_file)

    elif len(shape_file) < 1:
        print("\nNo shape files were found in the root DIR!")
        print("[0]\tScan dataset root folder")
        print("[1]\tUser will prrovide the full path")
        print("[2]\tFull root and subdir scan (NOT RECOMMENDED!)")
        print("[99]\tExit the toolkit")
        user_decision = input("Option: ")
        if user_decision == 0:
            shape_file = glob(os.path.join(directory_path, "**", "*shape*.txt"))
            with open(shape_file[0], "r") as infile:
                original_shape = infile.read().replace(")", "").replace("(", "")
                original_shape = tuple([int(x) for x in original_shape.split(", ")])
                print(original_shape)
            return original_shape

        elif user_decision == 1:
            shape_path = input("Shape path: ")
            try:
                with open(shape_path[0], "r") as infile:
                    original_shape = infile.read().replace(")", "").replace("(", "")
                    return original_shape
            except:
                raise Exception("Could not find the file!\nExiting!")
        
        elif user_decision == 2:
            shape_path = glob()
# Define a function to load the dataset from the directory structure
def load_dataset(directory_path, nested=True):
    sequences = []
    filepaths = glob(os.path.join(directory_path, "**", "centroids.txt"))
    for file in tqdm(filepaths):
        data = np.loadtxt(file, delimiter=",")
        data = data.reshape((5, 128, 3))
        data = normalize(data)  # Normalize each sequence
        sequences.append(data)
    return np.array(sequences)

def normalize(matrix):
    norm = np.linalg.norm(matrix, axis=(1, 2), keepdims=True)
    matrix = matrix / norm  # normalized matrix
    return matrix

dataset_directory = './generated/'
data = load_dataset(dataset_directory)
# No need to normalize the entire dataset anymore

# Reshape the data to match the LSTM input shape
# (num_sequences, num_timesteps, num_features)
X = data[:, :-1, :]  # Input sequences (all timesteps except the last)
y = data[:, 1:, :]   # Target sequences (the next XYZ coordinates for each point)

# Define the LSTM model
model = Sequential()
model.add(Bidirectional(LSTM(64, input_shape=(X.shape[1], X.shape[2]))))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Dense(128 * 3))  # Output layer with 128*3 neurons for predicting XYZ for each point

# Reshape the target data to match the model's output shape
y = y.reshape(y.shape[0], -1)  # Reshape to (num_sequences, 128*3)

# Compile the model
model.compile(optimizer='adam', loss='mse')  # You can choose a different loss function if needed

# Train the model
model.fit(X, y, epochs=128, batch_size=16)  # Adjust epochs and batch size as needed

# Save the model
model.save("lstm.h5")

# Example test sequence
test_sequence = np.random.rand(128, 3)  # Shape: (128, 3)

# To predict the next XYZ coordinates for each point:
predicted_coordinates = model.predict(test_sequence.reshape(1, test_sequence.shape[0], test_sequence.shape[1]))

# Reshape the predicted_coordinates back to (128, 3) to get the predictions for each point
predicted_coordinates = predicted_coordinates.reshape(128, 3)

# The 'predicted_coordinates' variable now contains the predicted next XYZ coordinates for each point.
print("Predicted Coordinates for Each Point:", predicted_coordinates)