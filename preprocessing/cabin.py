"""
Feature engineering that deals with parsing cabin information.
"""

import tensorflow as tf
import tensorflow_transform as tft

from .common import value_count_across_dataset
from .features import NUM_CABIN_NUM_BUCKETS, NUM_OOV_TOKENS


def filled_cabin(Cabin: tf.SparseTensor) -> tf.Tensor:
    """Fill missing cabin entries with ?/?/?."""
    return tft.sparse_tensor_to_dense_with_shape(
        Cabin, shape=[None, 1], default_value="?/?/?"
    )


def cabin_parts(filled_cabin: tf.Tensor) -> tf.Tensor:
    """
    Split cabin (format `D/N/S`) into 3 separate parts.
    """
    return tf.strings.split(filled_cabin, "/", 3).to_tensor(
        shape=[None, 1, 3], name="cabin_parts"
    )


def cabin_deck(cabin_parts: tf.Tensor) -> tf.Tensor:
    """
    Parse cabin deck as string.
    """
    return cabin_parts[:, :, 0]


def cabin_deck_vocab(cabin_deck: tf.Tensor) -> tf.Tensor:
    """
    Encode cabin deck as index in a vocabulary.
    """
    return tft.compute_and_apply_vocabulary(
        cabin_deck,
        vocab_filename="cabin_deck_vocab",
        num_oov_buckets=NUM_OOV_TOKENS,
    )


def cabin_side(cabin_parts: tf.Tensor) -> tf.Tensor:
    """
    Parse cabin side as string.
    """
    return cabin_parts[:, :, 2]


def cabin_side_vocab(cabin_side: tf.Tensor) -> tf.Tensor:
    """
    Encode cabin side as index in a vocabulary.
    """
    return tft.compute_and_apply_vocabulary(
        cabin_side,
        vocab_filename="cabin_side_vocab",
        num_oov_buckets=NUM_OOV_TOKENS,
    )


def cabin_num(cabin_parts: tf.Tensor) -> tf.Tensor:
    """
    Parse cabin number as int.
    """
    cabin_num = cabin_parts[:, :, 1]
    return tf.strings.to_number(
        tf.where(cabin_num == "?", "-1", cabin_num),
        out_type=tf.int64,
        name="cabin_num",
    )


def flat_cabin_deck(cabin_deck: tf.Tensor) -> tf.Tensor:
    """
    Flatten cabin deck for use as key.

    From [None, 1] we squeeze it into [None] shape.
    """
    return tf.squeeze(cabin_deck, axis=-1)


def scaled_cabin_num(
    cabin_num: tf.Tensor, flat_cabin_deck: tf.Tensor
) -> tf.Tensor:
    """
    Scaled cabin number using 0-1 scaling.

    The scaling is performed per-deck.
    If deck A has cabins 1-20, then for that deck 1=>0.0, 20=>1.0.
    If deck B has cabins 1-100, then for that deck 1=>0.0, 100=>1.0
    """
    return tft.scale_to_0_1_per_key(
        cabin_num, key=flat_cabin_deck, name="scaled_cabin_num"
    )


def bucketized_cabin_num(
    cabin_num: tf.Tensor, flat_cabin_deck: tf.Tensor
) -> tf.Tensor:
    """
    Bucketized cabin number.

    The bucketization is performed per-deck.
    """
    flat_cabin_num = tf.squeeze(cabin_num, axis=-1)
    bucketized_value = tft.bucketize_per_key(
        flat_cabin_num,
        key=flat_cabin_deck,
        num_buckets=NUM_CABIN_NUM_BUCKETS,
        name="bucketized_cabin_num",
    )
    # From [None] shape get back to [None, 1]
    return tf.expand_dims(bucketized_value, axis=-1)


def cabin_occupants_count(filled_cabin: tf.Tensor) -> tf.Tensor:
    """
    Return number of family members.

    Calculated as number of passengers with the same last name.
    """
    return value_count_across_dataset(
        filled_cabin,
        vocab_name="cabin_occupants_count",
        missing_key=tf.constant("?/?/?", dtype=tf.string),
        missing_value=tf.constant(0, dtype=tf.int64),
    )
