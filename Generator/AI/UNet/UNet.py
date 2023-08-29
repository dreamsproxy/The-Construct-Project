import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.losses import KLDivergence, MeanSquaredError
import os
from skimage.metrics import structural_similarity as ssim
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
from glob import glob
import random
import cv2

def process_images(path_list):
    images_arr = []
    for p in tqdm(path_list):
        image = np.reshape(np.asarray(cv2.imread(p, cv2.IMREAD_GRAYSCALE)), (256, 256, 1))
        image = image / 255.0
        images_arr.append(image)
    images_arr = np.array(images_arr)
    return images_arr

def create_dataset(x_folder, y_folder, batch_size=32):
    x_paths = [os.path.join(x_folder, filename) for filename in os.listdir(x_folder)]
    y_paths = [os.path.join(y_folder, filename) for filename in os.listdir(y_folder)]
    x_paths = [x.replace("\\", "/") for x in x_paths]
    y_paths = [y.replace("\\", "/") for y in y_paths]
    
    zipped_arr = list(zip(x_paths, y_paths))
    random.shuffle(zipped_arr)
    split_point = int(len(zipped_arr) * 0.2)
    val_split = zipped_arr[:split_point]
    train_split = zipped_arr[split_point:]
    
    x_val, y_val = zip(*val_split)
    x_train, y_train = zip(*train_split)
    
    x_images_val = process_images(x_val)
    y_images_val = process_images(y_val)
    x_images_train = process_images(x_train)
    y_images_train = process_images(y_train)
    
    train_dataset = tf.data.Dataset.from_tensor_slices((x_images_train, y_images_train))
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((x_images_val, y_images_val))
    val_dataset = val_dataset.batch(batch_size)
    val_dataset = val_dataset.cache()
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train_dataset, val_dataset

def inference_prep(paths):
    if isinstance(paths, str):
        paths = [paths]
    batcher = []
    image = np.reshape(np.asarray(cv2.imread(paths[-1], cv2.IMREAD_GRAYSCALE)), (256, 256, 1))
    image = image / 255.0
    batcher.append(image)
    dataset = tf.data.Dataset.from_tensor_slices((batcher))
    dataset = dataset.batch(1)
    dataset = dataset.cache()
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset

# Define U-Net model for depth map prediction
def build_UNET(input_shape=(256, 256, 1)):
    #   DOWNSAMPLE BLOCK 1
    inputs = layers.Input(shape=input_shape)
    
    conv1 = layers.Conv2D(32, 3, padding='same')(inputs)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.LeakyReLU(alpha=0.2)(conv1)
    up1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    #   DOWNSAMPLE BLOCK 2
    conv2 = layers.Conv2D(64, 3, padding='same')(up1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.LeakyReLU(alpha=0.2)(conv2)
    up2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    #   DOWNSAMPLE BLOCK 3
    conv3 = layers.Conv2D(128, 3, padding='same')(up2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.LeakyReLU(alpha=0.2)(conv3)
    up3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    
    #########################################################################################
    
    #   DOWNSAMPLE BLOCK 4
    conv4 = layers.Conv2D(256, 3, padding='same')(up3)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.LeakyReLU(alpha=0.2)(conv4)
    up4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    #   UPSAMPLING BLOCK 1
    deconv1 = layers.Conv2DTranspose(256, 3, strides=(2, 2), padding='same')(up4)
    deconv1 = layers.BatchNormalization()(deconv1)
    deconv1 = layers.LeakyReLU(alpha=0.2)(deconv1)
    deconv1 = layers.concatenate([deconv1, conv4])

    #   UPSAMPLING BLOCK 2
    deconv2 = layers.Conv2DTranspose(128, 3, strides=(2, 2), padding='same')(deconv1)
    deconv2 = layers.BatchNormalization()(deconv2)
    deconv2 = layers.LeakyReLU(alpha=0.2)(deconv2)
    deconv2 = layers.concatenate([deconv2, conv3])

    #   UPSAMPLING BLOCK 3
    deconv3 = layers.Conv2DTranspose(64, 3, strides=(2, 2), padding='same')(deconv2)
    deconv3 = layers.LeakyReLU(alpha=0.2)(deconv3)
    deconv3 = layers.BatchNormalization()(deconv3)
    deconv3 = layers.concatenate([deconv3, conv2])

    #   UPSAMPLING BLOCK 4
    deconv4 = layers.Conv2DTranspose(32, 3, strides=(2, 2), padding='same')(deconv3)
    deconv4 = layers.BatchNormalization()(deconv4)
    deconv4 = layers.LeakyReLU(alpha=0.2)(deconv4)
    deconv4 = layers.concatenate([deconv4, conv1])

    outputs = layers.Conv2DTranspose(1, 1, activation='sigmoid')(deconv4)
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

@tf.function
def ssim_loss(y_true, y_pred):
    return  1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

@tf.function
def kl_divergence(y_true, y_pred):
    return KLDivergence()(y_true, y_pred)

@tf.function
def mse_loss(y_true, y_pred):
    return MeanSquaredError(reduction="auto", name="mean_squared_error")(y_true, y_pred)

@tf.function
def total_loss(y_true, y_pred):
    mse_loss_val = mse_loss(y_true, y_pred)
    ssim_loss_val = ssim_loss(y_true, y_pred)
    kl_loss = kl_divergence(y_true, y_pred)
    return kl_loss + ssim_loss_val + mse_loss_val

def plot_preds(model, input_image, epoch):
    preidction = model.predict(input_image)
    preidction = preidction[0, :, :, 0]
    plt.imshow(preidction, cmap="gray")
    plt.title(f"Epoch {epoch}")
    plt.show()

#   Hyperparams
epochs = 50
batch_size = 16
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

train_ds, val_ds = create_dataset('./dataset/x', './dataset/y', batch_size = batch_size)
model = build_UNET()
model.compile(optimizer=adam_opt, loss=total_loss, metrics=[mse_loss, ssim_loss, kl_divergence])

csv_logger = tf.keras.callbacks.CSVLogger(
        f"./models/training.log",
        append = True
)
plateu = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_mse_loss",
    factor=0.5,
    patience=5,
    verbose=1,
    mode="min",
    min_delta=0.0001,
    cooldown=1,
    min_lr=0.0000001,
)

model.fit(
    train_ds,
    epochs = epochs,
    callbacks = [csv_logger, plateu],
    validation_data = val_ds
)

model.save("./models/VAE-UNET.h5")
model.save_weights("./models/VAE-UNET_WEIGHTS.h5")
image_path = '1.png'
infer_data = inference_prep(image_path)
model.load_weights("./models/VAE-UNET_WEIGHTS.h5")
plot_preds(model, infer_data, epoch="final")
