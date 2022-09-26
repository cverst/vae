import matplotlib.pyplot as plt
from matplotlib import cm
import tensorflow as tf
import numpy as np
from typing import Union
from vae import VAE
from collections import defaultdict
from PIL import Image


def render_samples(dataset: tf.data.Dataset) -> None:
    """Render 8 examples of Animal Crossing villagers dataset and save to file.

    Args:
        dataset (tf.data.Dataset): Dataset from which to render villagers. Must be
            unbatched!
    """

    OUTPUT_FILENAME = "./output/samples.jpg"

    index = 1

    fig = plt.figure(figsize=(10, 5))
    fig.subplots_adjust(wspace=0.5)

    for record in dataset.take(8):
        plt.subplot(2, 4, index)
        plt.axis("off")
        plt.imshow(record["image"].numpy())

        attributes = ", ".join(
            [
                record["gender"].numpy().decode("utf-8"),
                record["personality"].numpy().decode("utf-8"),
                record["species"].numpy().decode("utf-8"),
            ]
        )
        plt.title(record["name"].numpy().decode("utf-8") + "\n" + attributes)

        index += 1

    plt.close()

    fig.savefig(OUTPUT_FILENAME)


# def render_loss(history: tf.keras.callbacks.History, suffix: str = "") -> None:
def render_loss(
    history: Union[tf.keras.callbacks.History, list], suffix: str = ""
) -> None:
    """Render training and validation loss and save to file.

    Args:
        history (Union[tf.keras.callbacks.History, list]): Model
            training history for which to render loss, or a list of lists in
            the shape of [loss, val_loss].
        suffix (str, optional): Suffix for saved file name. Defaults to "".
    """

    OUTPUT_FILENAME = f"./output/loss{suffix}d.jpg"

    if type(history) == list:
        loss = history[0]
        val_loss = history[1]
    else:
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

    ax.plot(loss, label="Train", color="black")
    ax.plot(val_loss, label="Validate", color="red")

    plt.title(label="Model loss by epoch", loc="center")
    ax.set(xlabel="Epoch", ylabel="Loss")

    plt.legend()
    plt.close()

    fig.savefig(OUTPUT_FILENAME)


def get_embedding(
    vae_model: VAE, dataset: tf.data.Dataset, labels: defaultdict
) -> tuple:
    """Get embeddings of data using the encoder of a VAE-model.

    Args:
        vae_model (VAE): Instance of the VAE-model class.
        dataset (tf.data.Dataset): Data from the Animal Crossing dataset. Can
            be train or validate data.
        labels (defaultdict): Labels for the combined train and validate data
            of the Animal Crossing dataset.

    Returns:
        tuple: A tuple of the form (species, z_values, images), where each
            element is an ordered list of arbitrary length. For all
            occurences of species[i] the associated embeddings and source
            images are found in z_values[i][1:n] and images[i][1:n],
            respectively, where n is the number of occurences of species[i]
            in the dataset.
    """

    # Helper function for calculating Z-embedding
    def _embed(record):
        record["embedding"] = (
            vae_model.encoder(np.expand_dims(record["image"], axis=0))[2]
            .numpy()
            .squeeze()
        )
        return record

    # Get embeddings for dataset
    embedded_villagers = []
    for record in dataset.unbatch().as_numpy_iterator():
        embedded = _embed(record)
        embedded_villagers.append(embedded)

    # Convert list of villager-dictionaries to a list of lists
    # Each sublist will correspond to a single species
    # We are separating by species because that seems to be most important for
    # a VAE trained on this dataset
    species = []
    images = []
    z_values = []
    for key in labels["species"]:
        vals = [
            villager["embedding"]
            for villager in embedded_villagers
            if villager["species"].decode("utf-8") == key
        ]
        imgs = [
            villager["image"]
            for villager in embedded_villagers
            if villager["species"].decode("utf-8") == key
        ]
        if len(vals) > 0:
            species.append(key)
            images.append(imgs)
            z_values.append(vals)

    return species, z_values, images


def visualize_1d(vae_model: VAE, dataset: tf.data.Dataset, labels: defaultdict) -> None:
    """Visualize image embeddings on 1d manifold.

    This function will save a figure consisting of three panels:
    1) Original images of selection of animals from species groups ordered
        along the latent space. Ordering is based on the mean latent value of
        a species group.
    2) Bar-and-whisker plot for distribution of a species' embeddings, sorted
        by their mean latent value.
    3) Examples of reconstructions along the latent space.

    Args:
        vae_model (VAE): Instance of the VAE-model class. The number of latent
            dimensions of the VAE model must equal 1 (VAE.latent_dim==1).
        dataset (tf.data.Dataset): Data from the Animal Crossing dataset. Can
            be train or validate data.
        labels (defaultdict): Labels for the combined train and validate data
            of the Animal Crossing dataset.
    """

    assert (
        vae_model.latent_dim == 1
    ), f"Model must have 1 latent dimension, but {vae_model.latent_dim} latent dimensions found."

    # Create figure
    fig = plt.figure(figsize=(10, 7.5))

    # ------------------------------------
    # Part 1: Box plot of image embeddings
    # ------------------------------------

    species, z_values, images = get_embedding(vae_model, dataset, labels)

    # Sort species, z_values, and images by z_value means
    z_means = [np.mean(vals) for vals in z_values]
    species = [x for _, x in sorted(zip(z_means, species))]
    z_values = [x for _, x in sorted(zip(z_means, z_values))]
    images = [x for _, x in sorted(zip(z_means, images))]

    # Get images of villagers closest to group means
    idx_closest = [
        np.argmin(np.abs([v - mn for v in val]))
        for mn, val in list(zip(z_means, z_values))
    ]
    closest_images = [imgs[idx] for idx, imgs in list(zip(idx_closest, images))]

    # Create box plot
    ax1 = plt.subplot(5, 5, (6, 24))
    ax1.boxplot(z_values, labels=species, showfliers=False)
    plt.xticks(rotation=90)
    ax1.set_ylabel("Z-value (standard deviations from mean")

    # ----------------------------------
    # Part 2: Top row of original images
    # ----------------------------------

    SKIP = 3  # sample every SKIP-images

    n_images_original = len(idx_closest[::SKIP])
    digit_size = np.shape(closest_images[0])[0]
    n_channels = np.shape(closest_images[0])[2]

    # Add individual images to a single large image for seamless visualization
    canvas_original = np.empty((digit_size, digit_size * n_images_original, n_channels))
    for i in range(n_images_original):
        canvas_original[
            :,
            i * digit_size : (i + 1) * digit_size,
            :,
        ] = closest_images[i * SKIP]

    # Visualize
    ax2 = plt.subplot(5, 5, (1, 4))
    ax2.imshow(canvas_original)
    ax2.set_axis_off()

    # ---------------------------------------
    # Part 3: Side column of decoded Z-values
    # ---------------------------------------

    n_images_decoded = 8

    # We will sample n_images_decoded points within ax2-limits
    grid_x = np.linspace(ax2.get_ylim()[0], ax2.get_ylim()[1], n_images_decoded)

    # Add individual images to a single large image for seamless visualization
    canvas_decoded = np.empty((digit_size * n_images_decoded, digit_size, n_channels))
    for i, xi in enumerate(grid_x):
        # Generate image using decoder
        z_sample = np.array([xi])
        x_decoded = vae_model.decoder.predict(z_sample)

        canvas_decoded[
            i * digit_size : (i + 1) * digit_size,
            :,
            :,
        ] = x_decoded

    # Adjust image data type
    canvas_decoded = canvas_decoded * 255
    canvas_decoded = canvas_decoded.astype("uint8")

    # Visualize
    ax3 = plt.subplot(5, 5, (10, 25))
    ax3.imshow(canvas_decoded)
    ax3.set_axis_off()

    # Wrap up
    plt.close()
    fig.savefig("./output/species_manifold_1d.jpg")


def visualize_2d(vae_model: VAE, dataset: tf.data.Dataset, labels: defaultdict) -> None:
    """Visualize image embeddings on 2d manifold.

    Args:
        vae_model (VAE): Instance of the VAE-model class. The number of latent
            dimensions of the VAE model must equal 1 (VAE.latent_dim==2).
        dataset (tf.data.Dataset): Data from the Animal Crossing dataset. Can
            be train or validate data.
        labels (defaultdict): Labels for the combined train and validate data
            of the Animal Crossing dataset.
    """

    species, z_values, _ = get_embedding(vae_model, dataset, labels)

    # Create scatter plot
    fig = plt.figure(figsize=(8, 6))
    cmap = cm.get_cmap("hsv", len(species))
    count = 1
    for z_vals, species in zip(z_values, species):
        z_vals = np.array(z_vals)
        plt.scatter(z_vals[:, 0], z_vals[:, 1], color=cmap(count), s=4, label=species)
        count += 1

    # Visualize
    plt.xlabel("Latent variable 1")
    plt.ylabel("Latent variable 2")
    plt.legend(fontsize=6)
    fig.savefig("./output/species_manifold_2d.jpg")


def render_manifold(vae_model: VAE, to_movie: bool = False) -> Union[np.array, None]:
    """Render decoded latent space

    Args:
        vae_model (VAE): Instance of the VAE-model class.
        to_movie (bool, optional): _description_. Defaults to False.

    Returns:
        Union[np.array, None]: _description_
    """

    assert (
        vae_model.latent_dim <= 2
    ), f"Function `render_manifold` can only handle 1 or 2 latent dimensions, but the model has {vae_model.latent_dim}  latent dimensions."

    assert (
        vae_model.input_shape[0] == vae_model.input_shape[1]
    ), f"Input model must have square input shape, but the input shape is {vae_model.input_shape[:2]}."

    image_size = vae_model.input_shape[0]
    n_channels = vae_model.input_shape[2]

    if vae_model.latent_dim == 1:
        SAMPLE_STD = 2
        N_IMAGES_DECODED = 12
        canvas = np.zeros((image_size, image_size * N_IMAGES_DECODED, n_channels))
        figsize = (15, 4)
    elif vae_model.latent_dim == 2:
        SAMPLE_STD = 10
        N_IMAGES_DECODED = 8
        canvas = np.zeros(
            (image_size * N_IMAGES_DECODED, image_size * N_IMAGES_DECODED, n_channels)
        )
        figsize = (16, 16)

    # Sample N_IMAGES_DECODED points within [-SAMPLE_STD, SAMPLE_STD] standard deviations
    grid_x = np.linspace(SAMPLE_STD, -SAMPLE_STD, N_IMAGES_DECODED)
    grid_y = np.linspace(-SAMPLE_STD, SAMPLE_STD, N_IMAGES_DECODED)
    if vae_model.latent_dim == 1:
        for i, yi in enumerate(grid_x):
            z_sample = np.array([yi])
            # Generate an image using a decoder model
            x_decoded = vae_model.decoder.predict(z_sample)
            canvas[
                :,
                i * image_size : (i + 1) * image_size,
                :,
            ] = x_decoded
    elif vae_model.latent_dim == 2:
        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = np.array([[xi, yi]])
                # Generate an image using a decoder model
                x_decoded = vae_model.decoder.predict(z_sample)
                canvas[
                    i * image_size : (i + 1) * image_size,
                    j * image_size : (j + 1) * image_size,
                    :,
                ] = x_decoded

    # Write to canvas
    canvas = canvas * 255
    canvas = canvas.astype("uint8")

    # Create plot unless canvas is used for movie
    if to_movie:
        return canvas
    else:
        fig = plt.figure(figsize=figsize)
        plt.imshow(canvas)
        plt.title(f"Z-range: {[-SAMPLE_STD, SAMPLE_STD]}")
        plt.axis("off")
        plt.close()
        fig.savefig(f"./output/reconstructed_images_{vae_model.latent_dim}d.jpg")


def walk_across_latent_space_1d(vae_model: VAE) -> None:
    """Decode latent space at set intervals and create morphing movie.

    Args:
        vae_model (VAE): Instance of the VAE-model class. The number of latent
            dimensions of the VAE model must equal 1 (VAE.latent_dim==1).
    """

    assert (
        vae_model.latent_dim == 1
    ), f"Model must have 1 latent dimension, but {vae_model.latent_dim} latent dimensions found."

    # Set parameters
    N_FRAMES = 50
    SAMPLE_STD = 3

    # Create stack
    movie_stack = []
    z_samples = np.linspace(SAMPLE_STD, -SAMPLE_STD, N_FRAMES * SAMPLE_STD)
    for z in z_samples:
        frame = vae_model.decoder.predict(np.array([z])).squeeze()
        frame = frame * 255
        frame = frame.astype("uint8")
        if vae_model.input_shape[2] == 4:
            frame = remove_alpha(frame)
            movie_stack.append(Image.fromarray(frame))
        elif vae_model.input_shape[2] == 1:
            movie_stack.append(Image.fromarray(frame).convert("P"))
        else:
            movie_stack.append(Image.fromarray(frame))

    # Save as movie
    movie_stack[0].save(
        f"output/walk_across_latent_space_1d.gif",
        save_all=True,
        append_images=movie_stack[1:],
        optimize=False,
        duration=10,
    )


def remove_alpha(frame: np.array) -> np.array:
    """Remove alpha channel from image."""

    transparancy_mask = frame[:, :, 3] == 0
    frame[transparancy_mask] = [255, 255, 255, 255]
    frame = frame[:, :, :3]
    return frame
