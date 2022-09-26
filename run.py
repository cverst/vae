import os
from dataset import Dataset
from rendering import (
    render_samples,
    remove_alpha,
    render_loss,
    visualize_1d,
    visualize_2d,
    render_manifold,
    walk_across_latent_space_1d,
)
from vae import VAE
from PIL import Image
import argparse


# TODO: Write readme and blog post
# TODO: improve naming
# TODO: improve calling from command line, allow variables to be passed


def run(
    tfrecord_path: str,
    image_shape: list,
    n_channels: int,
    latent_dim: int,
    n_epochs: int,
    epoch_step: int,
) -> None:
    """Wrapper to train ACNH villager VAE and render results.

    Args:
        tfrecord_path (str): Location of ACNH tfrecord file.
        image_shape (list): Image shape to use for dataset and model.
        n_channels (int): Number of channels to use from images. Must be 1, 3,
            or 4.
        latent_dim (int): Number of latent distributions to use for VAE.
        n_epochs (int): Number of epochs to train model.
        epoch_step (int): Step size between consecutive reconstructed images
            in video of decoded latent-space results.
    """

    assert n_channels in [
        1,
        3,
        4,
    ], f"Number of channels must be 1, 3, or 4, but it is {n_channels}."

    # Create folder for all output images
    PATH = "./output"
    path_exists = os.path.exists(PATH)
    if not path_exists:
        os.makedirs(PATH)
        print("New output directory created.")

    # Load data
    ds = Dataset()
    ds.get_datasets(tfrecord_path, n_channels=n_channels, image_shape=image_shape)
    ds.get_labels()

    # Show sample data with labels (written to file)
    render_samples(ds.dataset_train.unbatch())

    # For model training use datasets with images only
    ds_train = ds.dataset_train.map(lambda record: record["image"])
    ds_validate = ds.dataset_validate.map(lambda record: record["image"])

    # Initialize model
    vae = VAE(input_shape=[*image_shape, n_channels], latent_dim=latent_dim)
    vae.build_model()

    # Show model
    print(vae.encoder.summary())
    print(vae.decoder.summary())
    print(vae.model.summary())

    # Compile: add loss and optimizer
    vae.compile_model()

    # Train model and visualize

    # Method 1: Train all epochs at once, no movie
    # history = vae.model.fit(ds_train, epochs=N_EPOCHS, validation_data=ds_validate)
    # render_loss(history)

    # Method 2: Train epoch by epoch and make movie of training
    current_epoch = 0
    movie_stack = []
    loss = []
    val_loss = []

    for i in list(range(1, n_epochs + 1, epoch_step)):

        # Train
        print(f"Epoch: {current_epoch}")
        history = vae.model.fit(
            ds_train, epochs=epoch_step, validation_data=ds_validate
        )
        loss += history.history["loss"]
        val_loss += history.history["val_loss"]

        # Append current view of manifold to image stack after removing alpha channel and replacing with white
        frame = render_manifold(vae, to_movie=True)
        if n_channels == 4:
            frame = remove_alpha(frame)
            movie_stack.append(Image.fromarray(frame))
        elif n_channels == 1:
            frame = frame.squeeze()
            movie_stack.append(Image.fromarray(frame).convert("P"))
        else:
            movie_stack.append(Image.fromarray(frame))

        current_epoch += epoch_step

    movie_stack[0].save(
        f"output/manifold_training_{latent_dim}d.gif",
        save_all=True,
        append_images=movie_stack[1:],
        optimize=False,
        duration=10,
    )

    # Create renderings of training results
    render_loss([loss, val_loss], f"_{latent_dim}")

    if latent_dim == 1:
        visualize_1d(vae, ds.dataset_validate, ds.labels)
        walk_across_latent_space_1d(vae)
    elif latent_dim == 2:
        visualize_2d(vae, ds.dataset_validate, ds.labels)

    render_manifold(vae)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tfrecord_path",
        default="./data/villagers.tfrecord",
        help="Path to ACNH dataset tfrecord location.",
    )
    parser.add_argument(
        "--image_shape",
        default=[64, 64],
        type=int,
        nargs=2,
        help="Two values defining image shape for model, e.g., `64 64`.",
    )
    parser.add_argument(
        "--n_channels",
        default=4,
        type=int,
        help="Number of channels to use from png images. Must be 1, 3, or 4.",
    )
    parser.add_argument(
        "--latent_dim",
        default=2,
        type=int,
        help="Number of latent dimensions to use for VAE.",
    )
    parser.add_argument(
        "--n_epochs", default=2000, type=int, help="Number of epochs to train VAE."
    )
    parser.add_argument(
        "--epoch_step",
        default=1,
        type=int,
        help="Step size between consecutive reconstructed images in video of decoded latent-space results.",
    )
    args = parser.parse_args()

    run(
        tfrecord_path=args.tfrecord_path,
        image_shape=args.image_shape,
        n_channels=args.n_channels,
        latent_dim=args.latent_dim,
        n_epochs=args.n_epochs,
        epoch_step=args.epoch_step,
    )
