import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
# Define constants
latent_dim = 100  # Should match the latent_dim used during training
n_samples = 10

# Load the trained generator model
generator = load_model('./models/2000/saved_model/g_model.h5')

# Generate a random noise vector
noise = np.random.normal(size=(n_samples, latent_dim))

# Use the generator to create an image from the noise vector
generated_array = generator.predict(noise)

# Post-process the generated image to bring it into the desired range
generated_array = (generated_array + 1) * 127.5  # Denormalize
generated_array = generated_array.astype(np.uint8)

for i in range(n_samples):
    img = np.copy(generated_array[i])
    #img[img <= 199] = 0
    img[img >= 200 ] = 255
    img = ndi.gaussian_filter(img, sigma=0.8)
    #print(np.max(img))
    #raise
    cv2.imwrite(f'generated/gen {i}.png', img)
