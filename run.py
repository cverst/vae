import tensorflow as tf
from dataset import Dataset

ds = Dataset("./data/villagers.tfrecord")

ds.get_datasets()

