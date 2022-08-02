"""
Feature engineering dealing with PassengerID parsing.

The PassengerID (format `1234_45`) is misinterpreted as integer (`123456`).
We can recover the original value though.
"""
import tensorflow as tf
import tensorflow_transform as tft

from .features import NUM_OOV_TOKENS


def passenger_group_int(PassengerId: tf.Tensor) -> tf.Tensor:
    """
    Figure out the passenger group from malformed passenger ID.
    """
    return PassengerId // 100


def passenger_group_str(passenger_group_int: tf.Tensor) -> tf.Tensor:
    """
    Pad the numeric passenger group ID to get back the original string format.
    """
    return tf.strings.as_string(passenger_group_int, width=4, fill="0")


def passenger_group_seq(PassengerId: tf.Tensor) -> tf.Tensor:
    """
    Figure out the sequence number within a passenger group
    from malformed passenger ID.
    """
    return PassengerId % 100


def passenger_group_seq_str(passenger_group_seq: tf.Tensor) -> tf.Tensor:
    """
    Pad the numeric passenger group sequence number to get back the original
    string format.
    """
    return tf.strings.as_string(passenger_group_seq, width=2, fill="0")


def passenger_id(
    passenger_group_str: tf.Tensor, passenger_group_seq_str: tf.Tensor
) -> tf.Tensor:
    """
    Reconstruct the original passenger ID in string format (`1234_56`).
    """
    stacked = tf.stack([passenger_group_str, passenger_group_seq_str])
    return tf.strings.reduce_join(stacked, separator="_", axis=0)


def passenger_group_vocab(
    passenger_group_str: tf.Tensor,
) -> tf.Tensor:  # pragma: no cover
    """
    Encode passenger group as a integer index in a vocabulary.
    """
    return tft.compute_and_apply_vocabulary(
        passenger_group_str,
        vocab_filename="passenger_group_vocab",
        num_oov_buckets=NUM_OOV_TOKENS,
    )
