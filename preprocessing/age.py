"""
Preprocessing of age features.
"""
import tensorflow as tf
import tensorflow_transform as tft

from .features import ADULT_AGE, NUM_AGE_BUCKETS


def age(Age: tf.SparseTensor) -> tf.Tensor:
    """
    Provide non-missing age.

    For missing entries age will be filled with mean value.
    """
    return tft.sparse_tensor_to_dense_with_shape(
        Age, shape=[None, 1], default_value=tft.mean(Age)
    )


def scaled_age(age: tf.Tensor) -> tf.Tensor:
    """
    Calculate z-score scaled age feature.
    """
    return tft.scale_to_z_score(age)


def bucketized_age(age: tf.Tensor) -> tf.Tensor:
    """
    Return bucketized version of age.
    """
    return tft.bucketize(
        age, num_buckets=NUM_AGE_BUCKETS, name="bucketized_age"
    )


def is_adult(age: tf.Tensor) -> tf.Tensor:
    """
    Calculate whether passenger is an adult.
    """
    return tf.where(
        age >= ADULT_AGE,
        tf.constant(1, dtype=tf.int64),
        tf.constant(0, dtype=tf.int64),
    )

