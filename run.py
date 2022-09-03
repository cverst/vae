import tensorflow as tf
from dataset import Dataset
from rendering import render_samples, render_loss
from vae import VAE
import numpy as np
import matplotlib.pyplot as plt

# TODO: documentation of code/model
# TODO: improve naming
# TODO: improve calling from command line, allow variables to be passed

TFRECORD_PATH = "./data/villagers.tfrecord"
IMAGE_SHAPE = [64, 64]
N_CHANNELS = 3
LATENT_DIM = 1
N_EPOCHS = 25

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
history = vae.model.fit(ds_train, epochs=N_EPOCHS, validation_data=ds_validate)

# Show losses
render_loss(history)

# Visualize latent space
# TODO: this
## training data
## generated data
## animated gif of generated data per training epoch


# Make embeddings for validation set
def get_embedding(record):
    record["embedding"] = (
        vae.encoder(np.expand_dims(record["image"], axis=0))[2].numpy().squeeze()
    )
    return record


embedded_villagers = []
for record in ds.dataset_validate.unbatch().as_numpy_iterator():
    embedded = get_embedding(record)
    embedded_villagers.append(embedded)

# gender = [villager["gender"].decode("utf-8") for villager in embedded_villagers]
# species = [villager["species"].decode("utf-8") for villager in embedded_villagers]
# personality = [
#     villager["personality"].decode("utf-8") for villager in embedded_villagers
# ]
# z_values = [villager["embedding"] for villager in embedded_villagers]


def to_nearest_half(num):
    return round(num * 2) / 2


# z_min = to_nearest_half(min(z_values))
# z_max = to_nearest_half(max(z_values))

# from matplotlib import cm

fig = plt.figure(figsize=(6, 4))
# label = ds.labels["species"]
# label_encoding = dict(zip(label, list(range(len(label)))))
# colormap = plt.cm.get_cmap("hsv", len(label))

# # plt.scatter(z_values, [label_encoding[sp] for sp in personality])
# colors = colormap([label_encoding[sp] for sp in species])
# z_values = np.array(z_values)
# plt.scatter(z_values[:, 0], z_values[:, 1], c=colors)
# plt.scatter(z_values, np.zeros_like(z_values), c=colors)

import pdb

pdb.set_trace()

species = []
z_values = []
for key in ds.labels["species"]:
    print(key)
    species = species.append(key)
    values = [
        villager["embedding"]
        for villager in embedded_villagers
        if villager["species"] == key
    ]
    print(values)
    if len(values) > 0:
        z_values = z_values.append(values)
plt.boxplot(z_values, labels=species)
plt.close()
fig.savefig("species_along_latent_space.jpg")


# # Get images and labels in order
# (imgs, annotations) = ds.dataset_validate.map(
#     lambda record: (record["image"], record.pop("image"))
# )

# # Use encoder model to encode inputs into a latent space
# imgs_encoded = vae.encoder.predict(imgs)

# # Recall that our encoder returns 3 arrays: z-mean, z-log-sigma and z. We plot the values for z
# # Create a scatter plot
# fig = plt.scatter(x=imgs_encoded[2][:, 0], y=np.zeros_like(imgs_encoded[2][:, 0]))

# # Set figure title
# # fig.update_layout(title_text="MNIST digit representation in the 2D Latent Space")

# # Update marker size
# # fig.update_traces(marker=dict(size=2))

# fig.close()
# fig.savefig("embedded.jpg")


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
    n = 16  # figure with 8x8 villagers
    digit_size = IMAGE_SHAPE[0]
    figure = np.zeros((digit_size, digit_size * n, 3))

    # We will sample n points within [-1.5, 1.5] standard deviations
    grid_x = np.linspace(1.5, -1.5, n)
    # grid_x = np.linspace(z_max, z_min, n)

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
