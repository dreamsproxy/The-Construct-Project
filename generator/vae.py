import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMG_H, IMG_W = 256, 256  # Adjust these dimensions to match your image size

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_png(img)
    img = tf.image.resize(img, (IMG_H, IMG_W))
    img = tf.cast(img, tf.float32)
    img = (img - 127.5) / 127.5
    return img

def tf_dataset(data_dir, batch_size):
    # Get the list of subfolders (0 to 4)
    subfolders = [os.path.join(data_dir, str(i)) for i in range(5)]
    
    # Create a list of image file paths for all subfolders
    image_paths = []
    for subfolder in subfolders:
        image_paths.extend(tf.io.gfile.glob(os.path.join(subfolder, '*.png')))
    
    # Shuffle the list of image paths
    tf.random.set_seed(42)
    tf.random.shuffle(image_paths)
    
    # Create sequences of five consecutive images
    sequences = [image_paths[i:i+5] for i in range(0, len(image_paths) - 4, 5)]
    
    # Create a dataset from the image sequences
    dataset = tf.data.Dataset.from_tensor_slices(sequences)
    dataset = dataset.shuffle(buffer_size=10240)
    
    # Load and preprocess images in parallel
    def load_sequence(sequence):
        return [load_image(image_path) for image_path in sequence]
    
    dataset = dataset.map(load_sequence, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    # Batch the sequences
    dataset = dataset.batch(batch_size)
    
    # Cache the dataset for better performance
    dataset = dataset.cache()
    
    # Prefetch data to improve training speed
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    return dataset

# Usage example
batch_size = 32
data_dir = './generated'  # Replace with the path to your dataset directory
train_dataset = tf_dataset(data_dir, batch_size)

# Iterate through batches of sequences
for batch in train_dataset.take(1):  # Take one batch as an example
    print("Batch shape:", batch.shape)  # This will be (batch_size, 5, IMG_H, IMG_W, 3)

# Define hyperparameters
sequence_length = 50
latent_dim = 32
epochs = 100
batch_size = 64

# Encoder
encoder_inputs = keras.Input(shape=(sequence_length, input_dim))
encoder = layers.LSTM(64, return_sequences=True)(encoder_inputs)
z_mean = layers.Dense(latent_dim)(encoder)
z_log_var = layers.Dense(latent_dim)(encoder)

# Sampling layer
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = layers.Lambda(sampling)([z_mean, z_log_var])

# Decoder
decoder_inputs = keras.Input(shape=(latent_dim,))
decoder = layers.RepeatVector(sequence_length)(decoder_inputs)
decoder = layers.LSTM(64, return_sequences=True)(decoder)
decoder_outputs = layers.TimeDistributed(layers.Dense(input_dim))(decoder)

# VAE Model
vae = keras.Model(encoder_inputs, decoder_outputs)

# Loss function
def vae_loss(x, x_decoded_mean):
    reconstruction_loss = keras.losses.mean_squared_error(x, x_decoded_mean)
    reconstruction_loss *= input_dim
    kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
    return tf.reduce_mean(reconstruction_loss + kl_loss)

vae.compile(optimizer='adam', loss=vae_loss)
vae.summary()

# Training
vae.fit(data, data, epochs=epochs, batch_size=batch_size, validation_split=0.2)
