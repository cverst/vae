import tensorflow as tf
from extract_data.load_data import from_tfrecord
from collections import defaultdict


class Dataset:
    """Dataset initialization for model training and validation"""

    def __init__(self) -> None:
        # Dataset-related attributes to be set later
        self.dataset_train = None
        self.dataset_validate = None
        self.labels = None

        # Other attributes to be set later
        self.image_shape = None
        self.batch_size = None
        self.n_shuffle_train = None

    def get_datasets(
        self,
        tfrecord_path: str,
        split: int = 3,
        image_shape: list = [128, 128],
        batch_size: int = 16,
        n_shuffle_train: int = 1024,
    ) -> None:
        """Load train and validate datasets and prepare them for usage.

        A dataset is loaded from the specified tfrecord file path. The data is then
        preprocessed and split in train and validate sets. Thereafter the datasets are
        cahed, shuffled (train set only), and batched. The prepared datasets are stored
        in class attributes `dataset_train` and `dataset_validate`.

        Args:
            tfrecord_path (str): Path to tfrecord file for a dataset where records were
                created with `tf.data.Dataset.from_tensor_slices()` from a dictionary.
            split (int, optional): One-in-"split" of images will go into a validate
                set, `train : validate = (split+1) : 1`. Before splitting the dataset
                is randomized with a shuffling buffer size of 1024. See also class
                method `_train_validation_split`. Defaults to 3.
            image_shape (list, optional): Desired shape of the images in the dataset.
                Defaults to [128, 128].
            batch_size (int, optional): Batch size applied to train and validate sets.
                Defaults to 16.
            n_shuffle_train (int, optional): Size of shuffle buffer applied to train
                set. Defaults to 1024.
        """

        # Assign attributes
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.n_shuffle_train = n_shuffle_train

        # Load data from tfrecord
        dataset = self._load_tfrecord(tfrecord_path)

        # Preprocess loaded data. Must be done before train-validate split
        dataset_preprocessed = self._preprocess(dataset, self.image_shape)

        # Split dataset into train and validate sets
        self.dataset_train, self.dataset_validate = self.train_validation_split(
            dataset_preprocessed, split
        )

        # Prepare datasets for usage
        self.dataset_train = (
            self.dataset_train.cache()
            .shuffle(self.n_shuffle_train)
            .batch(self.batch_size)
        )
        self.dataset_validate = self.dataset_validate.cache().batch(self.batch_size)

    def _load_tfrecord(self, tfrecord_path: str) -> tf.data.Dataset:
        """Load dataset from tfrecord.

        Args:
            tfrecord_path (str): Location of tfrecord to be loaded.

        Returns:
            tf.data.Dataset: Dataset from tfrecord.
        """
        dataset = from_tfrecord(tfrecord_path)
        return dataset

    def _preprocess(
        self, dataset: tf.data.Dataset, image_shape: list
    ) -> tf.data.Dataset:
        """Preprocessing, e.g., resizing and changing data type, of dataset.

        Args:
            dataset (tf.data.Dataset): Dataset to be preprocessed.
            image_shape (list): Desired image shape after preprocessing.

        Returns:
            tf.data.Dataset: Preprocessed dataset.
        """

        def _encode(record):
            """Operations mapped to dataset"""
            record["image"] = tf.image.resize_with_pad(
                record["image"], image_shape[0], image_shape[1]
            )
            record["image"] = (
                tf.image.convert_image_dtype(record["image"], tf.float32) / 255.0
            )
            return record

        dataset_processed = dataset.map(lambda rec: _encode(rec))

        return dataset_processed

    def train_validation_split(
        self, dataset: tf.data.Dataset, split: int
    ) -> tuple[tf.data.Dataset, tf.data.Dataset]:
        """Shuffle and then split dataset in train and validate sets.

        The dataset is windowed and one-in-"split" of images will go into a validate
        set:

            train : validate = (split+1) : 1

        The rest will go into a train set. Before splitting the dataset is randomized with a shuffling buffer size of 1024.

        Args:
            dataset (tf.data.Dataset): Dataset that will be split.
            split (int): Number to specify ratio between train and validate data.

        Returns:
            tuple[tf.data.Dataset, tf.data.Dataset]: A tuple as (dataset_train, dataset_validate).
        """

        # Shuffle dataset before splitting
        N_SHUFFLE = 1024
        dataset = dataset.shuffle(N_SHUFFLE)

        # Splitting of dataset
        dataset_train = dataset.window(split, split + 1).flat_map(
            lambda window: tf.data.Dataset.zip(
                dict([(k, v) for k, v in window.items()])
            )
        )
        dataset_validate = (
            dataset.skip(split)
            .window(1, split + 1)
            .flat_map(
                lambda window: tf.data.Dataset.zip(
                    dict([(k, v) for k, v in window.items()])
                )
            )
        )

        return dataset_train, dataset_validate

    def get_labels(self) -> None:
        """Extract labels from dataset and store in attribute `labels` (may take long)."""

        # Make sure dataset_train and dataset_validate are available
        assert (
            self.dataset_train is not None
        ), "Datasets have not been loaded yet. Call class method `get_datasets` before calling `get_labels`."

        # Extract labels from dataset
        labels = defaultdict(set)
        for record in self.dataset_train.unbatch().as_numpy_iterator():
            for key, value in record.items():
                if key not in ["name", "image"]:
                    labels[key].add(value.decode("utf-8"))
        for record in self.dataset_validate.unbatch().as_numpy_iterator():
            for key, value in record.items():
                if key not in ["name", "image"]:
                    labels[key].add(value.decode("utf-8"))

        self.labels = labels
