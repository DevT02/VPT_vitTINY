from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from . import base as base
from .registry import Registry
import tensorflow_datasets as tfds

# This constant specifies the percentage of data that is used to create custom
# train/val splits. Specifically, TRAIN_SPLIT_PERCENT% of the official training
# split is used as a new training split and the rest is used for validation.
TRAIN_SPLIT_PERCENT = 90


@Registry.register("data.mnist", "class")
class MNISTData(base.ImageTfdsData):
    """Provides MNIST data."""

    def __init__(self, data_dir=None, train_split_percent=None):
        dataset_builder = tfds.builder("mnist", data_dir=data_dir)
        dataset_builder.download_and_prepare()

        train_split_percent = train_split_percent or TRAIN_SPLIT_PERCENT

        trainval_count = dataset_builder.info.splits["train"].num_examples
        test_count = dataset_builder.info.splits["test"].num_examples
        num_samples_splits = {
            "train": (train_split_percent * trainval_count) // 100,
            "val": trainval_count - (train_split_percent * trainval_count) // 100,
            "trainval": trainval_count,
            "test": test_count
        }

        tfds_splits = {
            "train": "train[:{}]".format(num_samples_splits["train"]),
            "val": "train[{}:]".format(num_samples_splits["train"]),
            "trainval": "train",
            "test": "test"
        }

        super(MNISTData, self).__init__(
            dataset_builder=dataset_builder,
            tfds_splits=tfds_splits,
            num_samples_splits=num_samples_splits,
            num_preprocessing_threads=400,
            shuffle_buffer_size=10000,
            base_preprocess_fn=base.make_get_tensors_fn(["image", "label"]),
            num_classes=dataset_builder.info.features["label"].num_classes)
