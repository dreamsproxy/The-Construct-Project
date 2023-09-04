from glob import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

def create_dataset(npy_paths, num_epochs, batch_size):
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
    print(dataset.shape)
    # Convert the list to a NumPy array
    # Create a TensorFlow Dataset
    dataset = tf.data.Dataset.from_tensor_slices(dataset)

    # Batch and shuffle the dataset as needed
    buffer_size = 1000

    dataset = dataset.batch(batch_size)
    dataset = dataset.shuffle(buffer_size=buffer_size)

    # Prefetch the data
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    # Repeat for multiple epochs if needed
    dataset = dataset.repeat(num_epochs)
    
    return dataset


num_epochs = 10
batch_size = 8

# Define the pattern to match your Numpy files
npy_paths = glob("./dataset/**/centroids.npy")
dataset = create_dataset(npy_paths, num_epochs, batch_size)
# Define the model
model = Sequential()
model.add(Input(shape=(8, 200, 2)))
model.add(LSTM(64, input_shape=(8, 200, 2), return_sequences=True))

# Add a Dense layer to output the next set of 2D coordinates for each input
model.add(Dense(2))

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(dataset, epochs=num_epochs, batch_size=batch_size)
#raise
model.save("LSTM.h5")
# Predict the next set of coordinates for a new input
new_input = np.random.rand(1, 128, 2)
predicted_output = model.predict(new_input)

print("Predicted Output Shape:", predicted_output.shape)
