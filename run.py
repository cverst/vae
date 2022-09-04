from dataset import Dataset
from rendering import (
    render_samples,
    # render_loss,
    render_loss_from_lists,
    visualize_1d,
    visualize_2d,
    render_manifold,
)
from vae import VAE
from PIL import Image

# TODO: documentation of code/model
# TODO: improve naming
# TODO: improve calling from command line, allow variables to be passed

TFRECORD_PATH = "./data/villagers.tfrecord"
IMAGE_SHAPE = [64, 64]
N_CHANNELS = 4
LATENT_DIM = 1
N_EPOCHS = 600
EPOCH_STEP = 1

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

# Train model and visualize

# Method 1: Train all epochs at once
# Need ca. 500 epochs for full training
# history = vae.model.fit(ds_train, epochs=N_EPOCHS, validation_data=ds_validate)
# render_loss(history)

# Method 2: Train epoch by epoch and make movie of training
current_epoch = 0
movie_stack = []
loss = []
val_loss = []

for i in list(range(1, N_EPOCHS + 1, EPOCH_STEP)):

    # Train
    print(f"Epoch: {current_epoch}")
    history = vae.model.fit(ds_train, epochs=EPOCH_STEP, validation_data=ds_validate)
    loss += history.history["loss"]
    val_loss += history.history["val_loss"]

    # Append current view of manifold to image stack after removing alpha channel and replacing with white
    frame = render_manifold(vae, IMAGE_SHAPE[0], N_CHANNELS, movie=True)
    transparancy_mask = frame[:, :, 3] == 0
    frame[transparancy_mask] = [255, 255, 255, 255]
    frame = frame[:, :, :3]
    movie_stack.append(Image.fromarray(frame))
    current_epoch += EPOCH_STEP

movie_stack[0].save(
    f"output/manifold_training_{LATENT_DIM}d.gif",
    save_all=True,
    append_images=movie_stack[1:],
    optimize=False,
    duration=10,
)

render_loss_from_lists(loss, val_loss, f"_{LATENT_DIM}")

if LATENT_DIM == 1:
    visualize_1d(vae, ds.dataset_validate, ds.labels)
elif LATENT_DIM == 2:
    visualize_2d(vae, ds.dataset_validate, ds.labels)

render_manifold(vae, IMAGE_SHAPE[0], N_CHANNELS)
