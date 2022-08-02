"""
Preprocessing of categorical features.
"""


import tensorflow as tf
import tensorflow_transform as tft

from .common import mode
from .features import NUM_OOV_TOKENS


def _categorical(categorical_feature: tf.SparseTensor) -> tf.Tensor:
    """
    Parse boolean feature.
    Input format: `True`, `False` or missing.

    For missing entries, the mode (most frequent option) will be backfilled.
    """
    most_common_value = mode(categorical_feature)
    return tft.sparse_tensor_to_dense_with_shape(
        categorical_feature, shape=[None, 1], default_value=most_common_value
    )


def home_planet(HomePlanet: tf.SparseTensor) -> tf.Tensor:
    """
    Parse passenger's home planet.
    """
    return _categorical(HomePlanet)


def home_planet_vocab(home_planet: tf.Tensor) -> tf.Tensor:  # pragma: no cover
    """
    Return vocabulary index of passenger's home planet.
    """
    return tft.compute_and_apply_vocabulary(
        home_planet,
        vocab_filename="home_planet_vocab",
        num_oov_buckets=NUM_OOV_TOKENS,
    )


def destination(Destination: tf.SparseTensor) -> tf.Tensor:
    """
    Parse passenger's destination planet.
    """
    return _categorical(Destination)


def destination_vocab(destination: tf.Tensor) -> tf.Tensor:  # pragma: no cover
    """
    Return vocabulary index of passenger's destination planet.
    """
    return tft.compute_and_apply_vocabulary(
        destination,
        vocab_filename="destination_vocab",
        num_oov_buckets=NUM_OOV_TOKENS,
    )
