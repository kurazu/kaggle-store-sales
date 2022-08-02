"""
Preprocessing of passenger names features.
"""
import tensorflow as tf
import tensorflow_transform as tft

from .common import value_count_across_dataset
from .features import NUM_OOV_TOKENS


def name_parts(Name: tf.SparseTensor) -> tf.Tensor:
    """
    Parse passenger name (format `Alice McDonald`) parts.
    """
    dense_names = tft.sparse_tensor_to_dense_with_shape(
        Name, shape=[None, 1], default_value="? ?"
    )
    return tf.strings.split(dense_names, " ", 2).to_tensor(
        shape=[None, 1, 2], name="name_parts"
    )


def first_name(name_parts: tf.Tensor) -> tf.Tensor:
    """
    Parse first name of a passenger.
    """
    return name_parts[:, :, 0]


def last_name(name_parts: tf.Tensor) -> tf.Tensor:
    """
    Parse last name of a passenger.
    """
    return name_parts[:, :, 1]


def last_name_vocab(last_name: tf.Tensor) -> tf.Tensor:  # pragma: no cover
    """
    Encode passenger name as index in a vocabulary.
    """
    return tft.compute_and_apply_vocabulary(
        last_name,
        vocab_filename="last_name_vocab",
        num_oov_buckets=NUM_OOV_TOKENS,
    )


def family_members_count(last_name: tf.Tensor) -> tf.Tensor:
    """
    Return number of family members.

    Calculated as number of passengers with the same last name.
    """
    return value_count_across_dataset(
        last_name,
        vocab_name="family_members_count",
        missing_key=tf.constant("?", dtype=tf.string),
        missing_value=tf.constant(0, dtype=tf.int64),
    )


def scaled_family_members_count(
    family_members_count: tf.Tensor,
) -> tf.Tensor:  # pragma: no cover
    """
    Return scaled number of family members.
    """
    return tft.scale_to_0_1(
        family_members_count, name="scaled_family_members_count"
    )
