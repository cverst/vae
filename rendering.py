import matplotlib.pyplot as plt
from matplotlib import cm
import tensorflow as tf
import numpy as np


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


def render_loss(history: tf.keras.callbacks.History) -> None:
    """Render training and validation loss and save to file.

    Args:
        history (tf.keras.callbacks.History): Model training history for which to
            render loss.
    """

    OUTPUT_FILENAME = "./output/loss.jpg"

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

    ax.plot(history.history["loss"], label="Train", color="black")
    ax.plot(history.history["val_loss"], label="Validate", color="red")

    plt.title(label="Model loss by epoch", loc="center")
    ax.set(xlabel="Epoch", ylabel="Loss")
    plt.xticks(
        ticks=list(range(len(history.history["loss"]))),
        labels=list(range(1, len(history.history["loss"]) + 1)),
    )
    plt.legend()

    plt.close()

    fig.savefig(OUTPUT_FILENAME)


def render_loss_from_lists(loss, val_loss, suffix):

    OUTPUT_FILENAME = f"./output/loss_{suffix}d.jpg"

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

    ax.plot(loss, label="Train", color="black")
    ax.plot(val_loss, label="Validate", color="red")

    plt.title(label="Model loss by epoch", loc="center")
    ax.set(xlabel="Epoch", ylabel="Loss")
    plt.legend()

    plt.close()

    fig.savefig(OUTPUT_FILENAME)


def visualize_1d(vae_model, dataset, labels):

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


def get_embedding(vae_model, dataset, labels):

    # Helper function for calculating actual embedding
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
    # We are separating by species because that seems to be most important in this dataset
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


def visualize_2d(vae_model, dataset, labels):

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


def render_manifold(vae_model, image_size, n_channels, movie=False):

    assert (
        vae_model.latent_dim <= 2
    ), f"`render_manifold` can only handle 1 or 2 latent dimensions, but your latent dimension is {vae_model.latent_dim}"

    if vae_model.latent_dim == 1:
        n_images_decoded = 12
        canvas = np.zeros((image_size, image_size * n_images_decoded, n_channels))
        sample_std = 2
        figsize = (15, 4)
    elif vae_model.latent_dim == 2:
        n_images_decoded = 8
        canvas = np.zeros(
            (image_size * n_images_decoded, image_size * n_images_decoded, n_channels)
        )
        sample_std = 10
        figsize = (16, 16)

    # Sample n_images_decoded points within [-sample_std, sample_std] standard deviations
    grid_x = np.linspace(sample_std, -sample_std, n_images_decoded)
    grid_y = np.linspace(-sample_std, sample_std, n_images_decoded)
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

    # Plot canvas_decoded
    canvas = canvas * 255
    canvas = canvas.astype("uint8")

    if not movie:
        fig = plt.figure(figsize=figsize)
        plt.imshow(canvas)
        plt.title(f"Z-range: {[-sample_std, sample_std]}")
        plt.axis("off")
        plt.close()
        fig.savefig(f"./output/reconstructed_images_{vae_model.latent_dim}d.jpg")
    else:
        return canvas
