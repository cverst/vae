from PIL import Image
import load_data
import argparse


def show_sample(tfrecord: str, n_shuffle: int = 1) -> None:
    """Show sample image and annotations for an Animal Crossing villagers tfrecord.

    Args:
        tfrecord (str): File name or path to tfrecord.
        n_shuffle (int, optional): Size of shuffle buffer to randomize sample shown.
            Use n_shuffle=1 for no shuffling. Defaults to 1.
    """

    dataset = load_data.from_tfrecord(tfrecord)

    for record in dataset.shuffle(n_shuffle).take(1):
        img = record["image"].numpy()
        img = Image.fromarray(img)
        img.show()
        print(record["name"].numpy())
        print(record["species"].numpy())
        print(record["gender"].numpy())
        print(record["personality"].numpy())


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", default="../data/villagers.tfrecord", help="Path to tfrecord."
    )
    parser.add_argument(
        "--n_shuffle",
        default=16,
        type=int,
        help="Size of shuffle buffer. Must be greater than or equal to 1.",
    )
    args = parser.parse_args()

    show_sample(tfrecord=args.source, n_shuffle=args.n_shuffle)
