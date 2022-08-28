from pickletools import optimize
import tensorflow as tf
from dataset import Dataset
from rendering import render_samples
from vae import VAE

IMAGE_SHAPE = [64, 64]

# Load data
TFRECORD_PATH = "./data/villagers.tfrecord"
ds = Dataset()
ds.get_datasets(TFRECORD_PATH, image_shape=IMAGE_SHAPE)
ds.get_labels()

# Show sample data with labels (written to file)
render_samples(ds.dataset_train.unbatch())

ds_train = ds.images_only(ds.dataset_train)
ds_validate = ds.images_only(ds.dataset_validate)

# Initialize model
vae = VAE(input_shape=[*IMAGE_SHAPE, 3], latent_dim=2)
vae.build_model()

# Show model
print(vae.encoder.summary())
print(vae.decoder.summary())
print(vae.model.summary())

# Compile model
# vae.model.compile(
#     optimizer="adam",
#     loss=vae.loss_function,
# )

# Train model and use tensorboard
# history = vae.model.fit(ds_train, epochs=2, validation_data=ds_validate)

# Visualize latent space
## training data
## generated data

# Visualize a single white-noise image?
