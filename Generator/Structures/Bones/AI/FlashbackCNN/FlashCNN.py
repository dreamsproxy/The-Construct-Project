import matplotlib.pyplot as plt
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

"""
    PoC implementation
    Flashback Convolutional Layer
    Conv2D Batch 0 = Conv2D observation on timestep 0
    
    Dataset is organized in shape (timestep[dT], IMG_H, IMG_W, IMG_C)
    Where (batch 0 = timestep [dT] 0)
    Where batch size = n_samples
    
    Epoch 0: Batch/dT 0;
    Input(shape = (IMG_H, IMG_W, IMG_C))
    
    Conv2D(32, (3, 3), padding = "same")
    FlashbackConv2D(64) -> init (dT = 0) -> (0, 256, 256, 1) -> Save state/feature map
    
    Conv2DTranspose(64, (3, 3), strides = (2, 2), activation = "leakyrelu")
    Conv2DTranspose(1, (1, 1), strides = (1, 1), activation = "Sigmoid") [OUTPUT]
    
    
    (1, 256, 256, 1)
"""

#   Hyperparams
epochs = 2
batch_size = 1
lr = 0.0002
b1 = 0.5
w_decay = 0.002

IMG_H, IMG_W, IMG_C = (256, 256, 1)
# Compile the model
adam_opt = optimizers.Adam(
    learning_rate = lr,
    beta_1 = b1
    #decay = w_decay
    )

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
    split_point = int(len(zipped_arr) * 0.5)
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

# Define the RecursiveConv2DLayer (you should have this defined in your model)
class RecursiveConv2DLayer(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides = None):
        super(RecursiveConv2DLayer, self).__init__()
        self.conv2d = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')

    def call(self, inputs, initial_state=None):
        if initial_state is None:
            return self.conv2d(inputs)
        else:
            combined_input = tf.concat([inputs, initial_state], axis=-1)
            return self.conv2d(combined_input)

# Create a simple model
model = tf.keras.Sequential([
    layers.Input(shape=(IMG_H, IMG_W, IMG_C)),
    
    layers.Conv2D(32, 3, padding = "same"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    RecursiveConv2DLayer(64, (3, 3), strides=(1, 1)),
    
    layers.Conv2DTranspose(32, (3, 3), strides = (2, 2), padding = "same"),
    #layers.MaxPooling2D(pool_size=(2, 2)),
    
    layers.Conv2D(1, (3, 3), strides = (1, 1), padding = "same")
])

train_ds, val_ds = create_dataset('./dataset/x', './dataset/y', batch_size = batch_size)

model.compile(optimizer = "adam", loss = "mse")


for train in range(2):
    model.fit(
        train_ds,
        epochs = 1,
        validation_data = val_ds
    )
    model.save_weights(f"Epoch{train}.h5")

# Load weights into the model (replace 'model_weights.h5' with your weight file)
model.load_weights('Epoch0.h5')

# Access the weights of the first RecursiveConv2DLayer
weights = model.layers[2].get_weights()[0]
#print(len(weights))
print(weights.shape)
subplots = 32

for c in range(0, 64):
    plt.subplot(8, 8, c+1)
    plt.imshow(weights[:, :, 0, c])
    plt.axis('off')
plt.tight_layout()
plt.show()