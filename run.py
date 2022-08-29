import tensorflow as tf
from dataset import Dataset
from rendering import render_samples, render_loss
from vae import VAE
import numpy as np
import matplotlib.pyplot as plt

# TODO: documentation of code/model

TFRECORD_PATH = "./data/villagers.tfrecord"
IMAGE_SHAPE = [64, 64]
N_CHANNELS = 1
LATENT_DIM = 1

# Load data
ds = Dataset()
ds.get_datasets(TFRECORD_PATH, n_channels=N_CHANNELS, image_shape=IMAGE_SHAPE)
ds.get_labels()

# Show sample data with labels (written to file)
render_samples(ds.dataset_train.unbatch())

# For model training get datasets with images only
ds_train = ds.dataset_train.map(lambda record: record["image"])
ds_validate = ds.dataset_validate.map(lambda record: record["image"])

# Initialize model
vae = VAE(input_shape=[*IMAGE_SHAPE, N_CHANNELS], latent_dim=LATENT_DIM)
vae.build_model()

# Show model
print(vae.encoder.summary())
print(vae.decoder.summary())
print(vae.model.summary())

# Compile: add loss and optimizer
vae.compile_model()

# Train model
history = vae.model.fit(ds_train, epochs=10, validation_data=ds_validate)

# Show losses
render_loss(history)

# Visualize latent space
# TODO: this
## training data
## generated data
## animated gif of generated data per training epoch

import pdb

pdb.set_trace()

# Get images and labels in order
(imgs, annotations) = ds.dataset_validate.map(lambda record: (record["image"], record.pop("image")))

# Use encoder model to encode inputs into a latent space
imgs_encoded = vae.encoder.predict(imgs)

# Recall that our encoder returns 3 arrays: z-mean, z-log-sigma and z. We plot the values for z
# Create a scatter plot
fig = plt.scatter(x=imgs_encoded[2][:, 0], y=np.zeros_like(imgs_encoded[2][:, 0]))

# Set figure title
# fig.update_layout(title_text="MNIST digit representation in the 2D Latent Space")

# Update marker size
# fig.update_traces(marker=dict(size=2))

fig.close()
fig.savefig("embedded.jpg")


# Display a 2D manifold of the digits
if LATENT_DIM == 2:
    n = 8  # figure with 8x8 villagers
    digit_size = IMAGE_SHAPE[0]
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
    figure = figure * 255
    figure = figure.astype("uint8")
    fig = plt.figure(figsize=(16, 16))
    plt.imshow(figure)
    plt.close()
    fig.savefig("test.jpg")
elif LATENT_DIM == 1:
    n = 8  # figure with 8x8 villagers
    digit_size = IMAGE_SHAPE[0]
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
