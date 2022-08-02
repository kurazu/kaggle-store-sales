"""
Preprocessing of target labels.
"""
import tensorflow as tf

from .features import TRANSFORMED_TARGET_KEY


def transported(Transported: tf.Tensor) -> tf.Tensor:
    """
    Parse target label from string format (`True`/`False`).
    """
    return tf.where(
        Transported == "True",
        tf.constant(1, dtype=tf.int64),
        tf.constant(0, dtype=tf.int64),
    )


assert transported.__name__ == TRANSFORMED_TARGET_KEY

