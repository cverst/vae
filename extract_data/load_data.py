import tensorflow as tf


def from_tfrecord(tfrecord: str) -> tf.data.Dataset:
    """Load and parse Animal Crossing villagers tfrecord to create dataset.

    Args:
        tfrecord (str): A tfrecord for the Animal Crossing villagers dataset.

    Returns:
        tf.data.Dataset: A tensorflow dataset of Animal Crossing villagers.
    """

    dataset = tf.data.TFRecordDataset(tfrecord)

    # Parse the tfrecord into tensors
    dataset = dataset.map(
        lambda x: tf.io.parse_single_example(
            x,
            features={
                "image": tf.io.FixedLenFeature([], tf.string),
                "name": tf.io.FixedLenFeature([], tf.string),
                "species": tf.io.FixedLenFeature([], tf.string),
                "personality": tf.io.FixedLenFeature([], tf.string),
                "gender": tf.io.FixedLenFeature([], tf.string),
            },
        )
    )

    def _decode_image(record):
        record["image"] = tf.image.decode_png(record["image"], channels=4)
        record["image"] = record["image"][:, :, :3]
        return record

    dataset = dataset.map(_decode_image)

    return dataset
