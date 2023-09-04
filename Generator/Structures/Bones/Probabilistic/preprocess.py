import numpy as np
from glob import glob
import tensorflow as tf

# Define the pattern to match your Numpy files
npy_paths = glob("./dataset/**/centroids.npy")

# Create a list to hold your dataset
dataset = []

# Load and process each Numpy file
for path in npy_paths:
    print(path)
    loaded_data = np.load(path)  # Load the Numpy file
    loaded_data = loaded_data[:, :, :2]

    # Assuming each file contains 8 rows (time steps) and 200 coordinates (2D)
    # Reshape the data to (8, 200, 2)
    #loaded_data = np.delete(loaded_data, axis=-1)
    #reshaped_data = loaded_data.reshape(8, 200, 2)

    dataset.append(loaded_data)

dataset = np.asarray(dataset)
# Convert the list to a NumPy array
# Create a TensorFlow Dataset
dataset = tf.data.Dataset.from_tensor_slices(dataset)

# Batch and shuffle the dataset as needed
batch_size = 32
buffer_size = 1000

dataset = dataset.batch(batch_size)
dataset = dataset.shuffle(buffer_size=buffer_size)

# Prefetch the data
dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# Repeat for multiple epochs if needed
num_epochs = 10
dataset = dataset.repeat(num_epochs)
