import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.losses import KLDivergence
from skimage.metrics import structural_similarity as ssim
import numpy as np
import os

# Define the U-Net model
def unet(input_shape=(256, 256, 1)):
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Bottleneck
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    
    # Decoder
    up5 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv3], axis=-1)
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(up5)
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(conv5)
    
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv2], axis=-1)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(up6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(conv6)
    
    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv1], axis=-1)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(up7)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(conv7)
    
    # Output layer
    outputs = Conv2D(1, 1, activation='sigmoid')(conv7)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# SSIM loss function
def ssim_loss(y_true, y_pred):
    return 1 - tf.image.ssim(y_true, y_pred, max_val=1.0)

# Load the dataset
def load_dataset(data_dir):
    input_images = []
    target_images = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".jpg"):  # Adjust file format as needed
            input_path = os.path.join(data_dir, filename)
            target_path = os.path.join("./dataset/B/", filename)  # Assuming target images are in "./dataset/B/"
            
            # Load and preprocess images
            input_img = tf.image.decode_image(tf.io.read_file(input_path), channels=1)
            target_img = tf.image.decode_image(tf.io.read_file(target_path), channels=1)
            
            input_img = tf.image.resize(input_img, (256, 256))
            target_img = tf.image.resize(target_img, (256, 256))
            
            input_images.append(input_img)
            target_images.append(target_img)
    
    return np.array(input_images), np.array(target_images)

# Create the U-Net model
model = unet()

# Define optimizer
optimizer = tf.keras.optimizers.Adam()

# Training loop using GradientTape
@tf.function
def train_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        kl_div_loss = KLDivergence()(targets, predictions)
        ssim_loss_value = tf.reduce_mean(ssim_loss(targets, predictions))
        mse_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(targets, predictions))
        total_loss = kl_div_loss + ssim_loss_value + mse_loss

    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# Load and preprocess the dataset
X_train, y_train = load_dataset("./dataset/A/")

# Training loop
epochs = 100
batch_size = 16

for epoch in range(epochs):
    for i in range(0, len(X_train), batch_size):
        batch_inputs = X_train[i:i + batch_size]
        batch_targets = y_train[i:i + batch_size]
        train_step(batch_inputs, batch_targets)

# Save the trained model if needed
# model.save('unet_model.h5')
