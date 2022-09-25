from __future__ import annotations
import os
import tensorflow as tf
import json
import argparse


def create_dataset(
    images_folder: str, annotations_file: str, target_tfrecord: str
) -> None:
    """Create a tfrecord of Animal Crossing villagers and save it.

    This function first creates a tensorflow dataset. After creation, a tfrecord file
    is saved in the specified path.

    Args:
        images_folder (str): Name of folder where images of villagers are stored.
        annotations_file (str): A JSON file with fields "name", "species",
            "personality", and "gender".
        target_tfrecord (str): File name of the saved tfrecord. Can be a full path.
    """

    # List all image filenames and get villager names
    filenames = os.listdir(images_folder)
    villager_names = [
        item.split("/")[-1].split(".")[0].replace("_", " ") for item in filenames
    ]

    # Load annotations
    with open(annotations_file) as f:
        annotations_json = json.load(f)

    # Remove annotations_json that are not in villager_names; else add image path
    idx_to_remove = []
    for elem, villager in enumerate(annotations_json):
        if villager["name"] not in villager_names:
            idx_to_remove.append(elem)
        else:
            villager["image"] = [
                os.path.join(images_folder, fn)
                for fn in filenames
                if villager["name"].replace(" ", "_") in fn
            ][0]
    annotations_json = [
        annotations_json[i]
        for i in range(len(annotations_json))
        if i not in idx_to_remove
    ]

    # Turn list of dictionaries in annotations_json into a single list for each
    # key (necessary for conversion to dataset)
    keys = annotations_json[0].keys()
    annotations = {}
    for key in keys:
        annotations[key] = [item[key] for item in annotations_json]

    # Create tf dataset from annotations
    dataset = tf.data.Dataset.from_tensor_slices(annotations)

    # Add image data to dataset
    def _get_image_data(record):
        record["image"] = tf.io.read_file(record["image"])
        return record

    dataset = dataset.map(_get_image_data)

    # Save dataset with tf.io.TFRecordWriter
    writer = tf.io.TFRecordWriter(target_tfrecord)
    for record in dataset:
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "image": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[record["image"].numpy()])
                    ),
                    "name": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[record["name"].numpy()])
                    ),
                    "species": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[record["species"].numpy()])
                    ),
                    "personality": tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[record["personality"].numpy()]
                        )
                    ),
                    "gender": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[record["gender"].numpy()])
                    ),
                }
            )
        )
        writer.write(example.SerializeToString())


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("images", help="Path to folder with ACNH images.")
    parser.add_argument(
        "--annotations",
        default="../data/annotations/animal-crossing-villagers-parsed.json",
        help="Path to parsed ACNH villagers JSON file.",
    )
    parser.add_argument(
        "--target", default="../data/villagers.tfrecord", help="Path tfrecord location."
    )
    args = parser.parse_args()

    if (
        args.annotations == "../data/annotations/animal-crossing-villagers-parsed.json"
        or args.target == "../data/villagers.tfrecord"
    ):
        PATH = "../data"
        path_exists = os.path.exists(PATH)
        if not path_exists:
            os.makedirs(PATH)
            print("New data directory created.")

    create_dataset(
        images_folder=args.images,
        annotations_file=args.annotations,
        target_tfrecord=args.target,
    )
