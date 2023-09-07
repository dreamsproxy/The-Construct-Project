
import os
import numpy as np
import cv2
from glob import glob
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
W_INIT = tf.keras.initializers.GlorotNormal()

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_png(img)
    img = tf.image.resize(img, (IMG_H, IMG_W))
    img = tf.cast(img, tf.float32)
    img = (img - 127.5) / 127.5
    return img

def tf_dataset(images_path, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(images_path)
    dataset = dataset.shuffle(buffer_size=520)
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.cache()
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

def deconv_block(inputs, num_filters, kernel_size, strides, bn = True):
    x = layers.Conv2DTranspose(
        filters=num_filters,
        kernel_size=kernel_size,
        kernel_initializer=W_INIT,
        padding="same",
        strides=strides,
        use_bias=False,
    )(inputs)

    if bn:
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha = 0.2)(x)
    return x

def conv_block(inputs, num_filters, kernel_size, padding="same", strides=2):
    x = layers.Conv2D(
        filters=num_filters,
        kernel_size=kernel_size,
        kernel_initializer=W_INIT,
        padding=padding,
        strides=strides,
    )(inputs)

    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha = 0.2)(x)
    x = layers.Dropout(0.25)(x)
    return x

def build_generator(latent_dim, init_shape = (8, 8, 512), debug=False):
    input_dense_kernels = init_shape[0] * init_shape[1] * init_shape[2]
    input_conv_shape = init_shape
    noise = layers.Input(shape=(latent_dim,), name="Noise Input")
    x = layers.Dense(input_dense_kernels, use_bias=False)(noise)
    # 8 * 8 * 512 = 32,768
    
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape(input_conv_shape)(x)
    # (8, 8, 512)
    x = deconv_block(x, num_filters=256, kernel_size=3, strides=2, bn = True)
    # (16, 16, 256)
    x = deconv_block(x, num_filters=128, kernel_size=3, strides=2, bn = True)
    # (32, 32, 128)
    x = deconv_block(x, num_filters=64, kernel_size=3, strides=2, bn = True)
    # (64, 64, 64)
    x = deconv_block(x, num_filters=32, kernel_size=3, strides=2, bn = True)
    # (128, 128, 32)
    x = deconv_block(x, num_filters=16, kernel_size=3, strides=2, bn = True)
    # (256, 256, 16)
    x = deconv_block(x, num_filters=IMG_C, kernel_size=3, strides=1, bn = False)
    # (256, 256, 16)

    generateor_output = layers.Activation("linear")(x)

    return Model(noise, generateor_output, name="Generator")

def build_critic():
    image_input = layers.Input(shape=(IMG_H, IMG_W, IMG_C))
    x = image_input
    x = conv_block(x, num_filters=32, kernel_size=3, strides=2)
    x = conv_block(x, num_filters=64, kernel_size=3, strides=2)
    x = conv_block(x, num_filters=128, kernel_size=3, strides=2)
    x = conv_block(x, num_filters=256, kernel_size=3, strides=2)
    x = layers.Flatten()(x)
    x = layers.Dense(512)(x)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(1)(x)
    return Model(image_input, x, name="Critic")

def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)

def gradient_penalty(samples, output):
    gradients = tf.gradients(output, samples)
    gradients_sqr = tf.square(gradients)
    gradients_sqr_sum = tf.reduce_sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
    gradient_l2_norm = tf.sqrt(gradients_sqr_sum)
    gradient_penalty = tf.reduce_mean(tf.square(gradient_l2_norm - 1))
    return gradient_penalty

class GAN(Model):
    def __init__(self, critic, generator, latent_dim, n_critic=1):
        super(GAN, self).__init__()
        self.critic = critic
        self.generator = generator
        self.latent_dim = latent_dim
        self.n_critic = n_critic

    def compile(self, d_optimizer, g_optimizer):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

    @tf.function
    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]

        for i in range(self.n_critic):
            ## Train the critic
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            generated_images = self.generator(random_latent_vectors)

            with tf.GradientTape() as ftape, tf.GradientTape() as rtape:
                real_predictions = self.critic(real_images)
                fake_predictions = self.critic(generated_images)
                d1_loss = tf.reduce_mean(fake_predictions) - tf.reduce_mean(real_predictions)
                alpha = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0., maxval=1.)
                
                interpolated_images = real_images * alpha + generated_images * (1 - alpha)
                interpolated_predictions = self.critic(interpolated_images)
                
                gradients = tf.gradients(interpolated_predictions, interpolated_images)[0]
                slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
                gradient_penalty = tf.reduce_mean(tf.square(slopes - 1))
                d_loss = d1_loss + 10 * gradient_penalty

            d_grads = ftape.gradient(d_loss, self.critic.trainable_weights)
            d_grads, _ = tf.clip_by_global_norm(d_grads, clip_norm=0.01)
            self.d_optimizer.apply_gradients(zip(d_grads, self.critic.trainable_weights))

        ## Train the generator
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        with tf.GradientTape() as gtape:
            generated_images = self.generator(random_latent_vectors)
            fake_predictions = self.critic(generated_images)
            g_loss = -tf.reduce_mean(fake_predictions)

        g_grads = gtape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_weights))

        return {"d_loss": d_loss, "g_loss": g_loss}

def save_plot(test_id, examples, epoch, n, debug = False):
    examples = (examples + 1) / 2.0
    if debug:
        print(examples[0].shape)
        print(examples[0])
    for i in range(n * n):
        examples[i] = examples[i] * 127.5 + 127.5
        plt.subplot(n, n, i+1)
        plt.axis("off")
        plt.imshow(examples[i], cmap="gray")
    filename = f"./{test_id}/samples/generated_plot_epoch-{epoch+1}.png"
    plt.tight_layout()
    plt.savefig(filename, )
    plt.close()

def prepare_dirs(test_id, delete_dir = False):
    os.makedirs(f"./{test_id}/samples/")
    os.makedirs(f"./{test_id}/saved_model/")
    
def print_summary(d_model, g_model, DEBUG = False):
    d_model.summary()
    g_model.summary()
    if DEBUG == True:
        raise

if __name__ == "__main__":
    test_id = "Dual Critic/WDCGAN_E-2K"
    LOAD = True
    load_id = "Dual Critic/WDCGAN_E-1K"
    
    ## Hyperparameters
    img_size = 256
    IMG_H = img_size
    IMG_W = img_size
    IMG_C = 1

    latent_dim = 256

    # Drop batch size to 10 after 1K epochs
    batch_size = 10
    num_epochs = 1000

    # Do 0.0003 on first 1K epochs
    # Then move to 0.0002 for both for each 1K epochs
    g_model_learning_rate = 0.0002
    d_model_learning_rate = 0.0002
    model_beta_1 = 0.5
    model_amsgrad = True

    prepare_dirs(test_id)    
    g_model = build_generator(latent_dim, debug=False)
    d_model = build_critic()

    if LOAD == True:
        d_model.load_weights(f"./{load_id}/saved_model/d_model.h5")
        g_model.load_weights(f"./{load_id}/saved_model/g_model.h5")

    # Build and initialize the GAN model
    gan = GAN(d_model, g_model, latent_dim, n_critic=2)

    # Prepare Optimizers
    d_optimizer = tf.keras.optimizers.Adam(
        learning_rate = d_model_learning_rate,
        beta_1 = model_beta_1,
        amsgrad = model_amsgrad
    )
    g_optimizer = tf.keras.optimizers.Adam(
        learning_rate=g_model_learning_rate,
        beta_1 = model_beta_1,
        amsgrad = model_amsgrad
    )

    gan.compile(d_optimizer, g_optimizer)

    images_path = glob("dataset/augmented/**")
    images_dataset = tf_dataset(images_path, batch_size)
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs}")
        gan.fit(
            images_dataset,
            epochs=1
        )

        n_samples = 16
        noise = np.random.normal(size=(n_samples, latent_dim))
        examples = g_model.predict(noise)

        save_plot(test_id, examples, epoch, int(np.sqrt(n_samples)), debug = False)

    g_model.save(f"./{test_id}/saved_model/g_model.h5", save_format="h5")
    d_model.save(f"./{test_id}/saved_model/d_model.h5", save_format="h5")