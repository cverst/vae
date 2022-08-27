from PIL import Image
import load_data


def show_sample(tfrecord: str, n_shuffle: int = 1) -> None:
    """Show sample image and annotations for an Animal Crossing villagers tfrecord.

    Args:
        tfrecord (str): File name or path to tfrecord.
        n_shuffle (int, optional): Size of shuffle buffer to randomize sample shown.
            Use n_shuffle=1 for no shuffling. Defaults to 1.
    """

    dataset = load_data.from_tfrecord(tfrecord)

    for record in dataset.shuffle(n_shuffle).take(1):
        img = record["image"].numpy() * 255
        img = Image.fromarray(img.astype("uint8"))
        img.show()
        print(record["name"].numpy())
        print(record["species"].numpy())
        print(record["gender"].numpy())
        print(record["personality"].numpy())


if __name__ == "__main__":
    show_sample(tfrecord="../data/villagers.tfrecord", n_shuffle=50)
