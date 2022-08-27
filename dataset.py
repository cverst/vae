import tensorflow as tf
from extract_data.load_data import from_tfrecord


class Dataset:
    def __init__(
        self,
        tfrecord_path: str,
    ) -> None:

        self.tfrecord_path = tfrecord_path

        self.dataset_train = None
        self.dataset_test = None
        self.labels = None

        self.image_size = None
        self.batch_size = None
        self.n_shuffle_train = None

    def get_datasets(
        self,
        split: int = 3,
        image_size: list = [128, 128],
        batch_size: int = 16,
        n_shuffle_train: int = 1024,
    ) -> None:

        self.image_size = image_size
        self.batch_size = batch_size
        self.n_shuffle_train = n_shuffle_train

        dataset = self._load_tfrecord()

        dataset_preprocessed = self._preprocess(dataset, self.image_size)

        self._train_validation_split(dataset_preprocessed, split)

        self.dataset_train = (
            self.dataset_train.cache()
            .shuffle(self.n_shuffle_train)
            .batch(self.batch_size)
        )
        self.dataset_test = self.dataset_test.cache().batch(self.batch_size)

    def _load_tfrecord(self) -> tf.data.Dataset:
        dataset = from_tfrecord(self.tfrecord_path)
        return dataset

    def _preprocess(self, dataset: tf.data.Dataset, image_size: list) -> tf.data.Dataset:
        def _encode(record):
            record["image"] = tf.image.resize(record["image"], image_size)
            record["image"] = tf.image.convert_image_dtype(record["image"], tf.float32)
            return record

        dataset_processed = dataset.map(lambda rec: _encode(rec))

        return dataset_processed

    def _train_validation_split(self, dataset: tf.data.Dataset, split: int) -> None:
        # split (int): One-in-"split" of images will go into a test set:

        #         train : test = (split+1) : 1

        #     The dataset is randomized with a shuffling buffer size of 1024 before
        #     splitting. When set to None, no splitting of datasets happens and test set
        #     will equal None. Defaults to None.
        N_SHUFFLE = 1024
        dataset = dataset.shuffle(N_SHUFFLE)
        self.dataset_train = dataset.window(split, split + 1).flat_map(
            lambda window: tf.data.Dataset.zip(
                dict([(k, v) for k, v in window.items()])
            )
        )
        self.dataset_test = (
            dataset.skip(split)
            .window(1, split + 1)
            .flat_map(
                lambda window: tf.data.Dataset.zip(
                    dict([(k, v) for k, v in window.items()])
                )
            )
        )
