import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.losses import KLDivergence, MeanSquaredError
import os
from skimage.metrics import structural_similarity as ssim
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
from glob import glob

def load_image(kspace_path, mri_path = None):
    image = tf.io.read_file(kspace_path)
    image = tf.image.decode_png(image, channels=1)  # RGB image
    #image = tf.image.resize_with_pad(image, 256, 256, method = "bicubic", antialias = True)
    image = tf.cast(image, tf.float32) / 255.0

    if mri_path != None:
        depth = tf.io.read_file(mri_path)
        depth = tf.image.decode_png(depth, channels=1)  # Grayscale depth map
        #depth = tf.image.resize_with_pad(depth, 256, 256, method = "bicubic", antialias = True)
        depth = tf.cast(depth, tf.float32) / 255.0
        return image, depth
    else:
        return image

def create_dataset(kspace_folder, mri_folder, batch_size=32, dataset_type = ".npy"):
    kspace_paths = [os.path.join(kspace_folder, filename) for filename in os.listdir(kspace_folder)]
    mri_paths = [os.path.join(mri_folder, filename) for filename in os.listdir(mri_folder)]
    
    dataset = tf.data.Dataset.from_tensor_slices((kspace_paths, mri_paths))

    dataset = dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.cache()
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset

def inference_prep(paths):
    if isinstance(paths, str):
        paths = [paths]

    dataset = tf.data.Dataset.from_tensor_slices((paths))
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(1)
    dataset = dataset.cache()
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def build_VAE(input_shape=(128, 128, 1)):
    # Encoder
    #   DOWNSCALE BLOCK 1
    inputs = layers.Input(shape=input_shape)
    conv1 = layers.Conv2D(32, 3, padding='same')(inputs)
    conv1 = layers.LeakyReLU(alpha=0.2)(conv1)
    conv1 = layers.Dropout(0.3)(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    #   DOWNSCALE BLOCK 2
    conv2 = layers.Conv2D(64, 3, padding='same')(pool1)
    conv2 = layers.LeakyReLU(alpha=0.2)(conv2)
    conv2 = layers.Dropout(0.3)(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    #   DOWNSCALE BLOCK 3
    conv3 = layers.Conv2D(128, 3, padding='same')(pool2)
    conv3 = layers.LeakyReLU(alpha=0.2)(conv3)
    conv3 = layers.Dropout(0.3)(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    #   DOWNSCALE BLOCK 4
    conv4 = layers.Conv2D(256, 3, padding='same')(pool3)
    conv4 = layers.LeakyReLU(alpha=0.2)(conv4)
    conv4 = layers.Dropout(0.3)(conv4)
    conv4 = layers.BatchNormalization()(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    #   BOTTLENECK BLOCK
    bottleneck = layers.Conv2D(512, 3, padding='same')(pool4)
    bottleneck = layers.LeakyReLU(alpha=0.2)(bottleneck)
    bottleneck = layers.LayerNormalization()(bottleneck)

    bottleneck = layers.Conv2D(512, 3, padding='same')(bottleneck)
    bottleneck = layers.LeakyReLU(alpha=0.2)(bottleneck)
    bottleneck = layers.LayerNormalization()(bottleneck)

    # Decoder
    #   UPSAMPLING BLOCK 1
    up1 = layers.Conv2DTranspose(256, 3, strides=(2, 2), padding='same')(bottleneck)
    up1 = layers.LeakyReLU(alpha=0.2)(up1)
    up1 = layers.BatchNormalization()(up1)
    up1 = layers.concatenate([up1, conv4])

    up1 = layers.Conv2D(256, 3, padding='same')(up1)
    up1 = layers.LeakyReLU(alpha=0.2)(up1)
    up1 = layers.BatchNormalization()(up1)

    #   UPSAMPLING BLOCK 2
    up2 = layers.Conv2DTranspose(128, 3, strides=(2, 2), padding='same')(up1)
    up2 = layers.LeakyReLU(alpha=0.2)(up2)
    
    up2 = layers.BatchNormalization()(up2)
    up2 = layers.concatenate([up2, conv3])

    up2 = layers.Conv2D(128, 3, padding='same')(up2)
    up2 = layers.LeakyReLU(alpha=0.2)(up2)
    up2 = layers.BatchNormalization()(up2)

    #   UPSAMPLING BLOCK 3
    up3 = layers.Conv2DTranspose(64, 3, strides=(2, 2), padding='same')(up2)
    up3 = layers.LeakyReLU(alpha=0.2)(up3)
    up3 = layers.BatchNormalization()(up3)
    up3 = layers.concatenate([up3, conv2])

    up3 = layers.Conv2D(64, 3, padding='same')(up3)
    up3 = layers.LeakyReLU(alpha=0.2)(up3)
    up3 = layers.BatchNormalization()(up3)

    #   UPSAMPLING BLOCK 4
    up4 = layers.Conv2DTranspose(32, 3, strides=(2, 2), padding='same')(up3)
    up4 = layers.LeakyReLU(alpha=0.2)(up4)
    up4 = layers.BatchNormalization()(up4)
    up4 = layers.concatenate([up4, conv1])

    up4 = layers.Conv2D(32, 3, padding='same')(up4)
    up4 = layers.LeakyReLU(alpha=0.2)(up4)
    up4 = layers.BatchNormalization()(up4)

    outputs = layers.Conv2DTranspose(1, 1, strides = (1, 1), activation='tanh')(up4)  # Output layer for depth prediction
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Compile the model with multiple losses
def reconstruction_loss(y_true, y_pred):
    reconstruction_loss = MeanSquaredError(y_true, y_pred)
    reconstruction_loss *= IMG_H * IMG_W * IMG_C
    return reconstruction_loss

def kl_divergence_loss(y_true, y_pred):
    return KLDivergence()(y_true, y_pred)

def total_loss(y_true, y_pred):
    l1_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    reconstruction_loss = reconstruction_loss(y_true, y_pred)
    kl_loss = kl_divergence_loss(y_true, y_pred)
    return reconstruction_loss + kl_loss + l1_loss

def plot_preds(model, input_image, epoch):
    PREDMRI = model.predict(input_image)
    PREDMRI = PREDMRI[0, :, :, 0]
    plt.imshow(PREDMRI, cmap="gray")
    plt.title(f"Epoch {epoch}")
    plt.show()

#   Hyperparams
epochs = 50
batch_size = 15
lr = 0.0002
b1 = 0.5
w_decay = 0.001

IMG_H, IMG_W, IMG_C = (256, 256, 1)
# Compile the model
adam_opt = optimizers.Adam(
    learning_rate = lr,
    amsgrad = True,
    beta_1 = b1,
    decay = w_decay
)

model = build_VAE()
model.compile(optimizer=adam_opt, loss=total_loss, metrics=[reconstruction_loss, kl_divergence_loss])
train_dataset = create_dataset('synthetic/', 'dataset/knee/mri/', batch_size = batch_size)
#test_dataset = create_dataset('dataset/test/k-space', 'dataset/test/mri', batch_size = 32)


csv_logger = tf.keras.callbacks.CSVLogger(
        f"./training.log",
        append = True
)

epochs = 50

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    total_loss_epoch = tf.keras.metrics.Mean()
    kl_div_loss_epoch = tf.keras.metrics.Mean()

    with tqdm(total=len(train_dataset), unit = " steps") as pbar:
        for batch in train_dataset:
            kspace_images, mri_images = batch

            with tf.GradientTape() as tape:
                predicted_mri = model(kspace_images)
                loss = total_loss(mri_images, predicted_mri)

            gradients = tape.gradient(loss, model.trainable_variables)
            adam_opt.apply_gradients(zip(gradients, model.trainable_variables))

            total_loss_epoch.update_state(loss)
            kl_div_loss_epoch.update_state(kl_divergence_loss(mri_images, predicted_mri))

            pbar.update(1)
            pbar.set_description(f"Loss: {total_loss_epoch.result().numpy():.4f}, KL Div: {kl_div_loss_epoch.result().numpy():.4f}")

    print(f"Total Loss: {total_loss_epoch.result().numpy():.4f}, KL Divergence Loss: {kl_div_loss_epoch.result().numpy():.4f}")
    model.save_weights("FusionVAE.keras")

image_path = 'dataset/knee/k-space/i_file100000.png'
infer_data = inference_prep(image_path)
model.load_weights("./FusionVAE.keras")
plot_preds(model, infer_data, epoch="final")
