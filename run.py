import tensorflow as tf
from dataset import Dataset
from rendering import render_samples
from vae import VAE
import numpy as np
import matplotlib.pyplot as plt

TFRECORD_PATH = "./data/villagers.tfrecord"
IMAGE_SHAPE = [64, 64]
INPUT_SHAPE = [*IMAGE_SHAPE, 3]
LATENT_DIM = 1

# Load data
ds = Dataset()
ds.get_datasets(TFRECORD_PATH, image_shape=IMAGE_SHAPE)
ds.get_labels()

# Show sample data with labels (written to file)
render_samples(ds.dataset_train.unbatch())

# For model training get datasets with images only
ds_train = ds.dataset_train.map(lambda record: record["image"])
ds_validate = ds.dataset_validate.map(lambda record: record["image"])

# Initialize model
vae = VAE(input_shape=INPUT_SHAPE, latent_dim=LATENT_DIM)
vae.build_model()

# Show model
print(vae.encoder.summary())
print(vae.decoder.summary())
print(vae.model.summary())

# import pdb

# pdb.set_trace()
# Add loss and compile
r_loss = np.product(vae.input_shape) * tf.math.reduce_sum(
    tf.math.reduce_sum(tf.keras.losses.mse(vae.inputs, vae.outputs), axis=1), axis=1
)
kl_loss = -0.5 * tf.math.reduce_sum(
    1 + vae.z_log_sigma - tf.math.square(vae.z_mean) - tf.math.exp(vae.z_log_sigma),
    axis=1,
)
vae_loss = tf.math.reduce_mean(r_loss + kl_loss)

vae.model.add_loss(vae_loss)
vae.model.compile(optimizer="adam")

# Train model and use tensorboard
history = vae.model.fit(ds_train, epochs=50, validation_data=ds_validate)

# Visualize latent space
## training data
## generated data

# Visualize a single white-noise image?


# Display a 2D manifold of the digits
if LATENT_DIM == 2:
    n = 8  # figure with 8x8 villagers
    digit_size = 64
    figure = np.zeros((digit_size * n, digit_size * n, 3))

    # We will sample n points within [-1.5, 1.5] standard deviations
    grid_x = np.linspace(1.5, -1.5, n)
    grid_y = np.linspace(-1.5, 1.5, n)

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            # Generate an image using a decoder model
            x_decoded = vae.decoder.predict(z_sample)

            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
                :,
            ] = x_decoded
    # Plot figure
    fig = plt.figure(figsize=(16, 16))
    plt.imshow(figure)
    plt.close()
    fig.savefig("test.jpg")
elif LATENT_DIM == 1:
    n = 8  # figure with 8x8 villagers
    digit_size = 64
    figure = np.zeros((digit_size, digit_size * n, 3))

    # We will sample n points within [-1.5, 1.5] standard deviations
    grid_x = np.linspace(1.5, -1.5, n)

    for i, yi in enumerate(grid_x):
        z_sample = np.array([yi])
        # Generate an image using a decoder model
        x_decoded = vae.decoder.predict(z_sample)

        figure[
            :,
            i * digit_size : (i + 1) * digit_size,
            :,
        ] = x_decoded
    # Plot figure
    figure = figure * 255
    figure = figure.astype("uint8")
    fig = plt.figure(figsize=(15, 4))
    plt.imshow(figure)
    plt.axis("off")
    plt.close()
    fig.savefig("test.jpg")
