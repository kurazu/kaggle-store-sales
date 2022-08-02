import tensorflow as tf
import tensorflow_transform as tft

from .common import count_proportion, value_count_robust


def transported_family_members_count(
    last_name: tf.Tensor, transported: tf.Tensor
) -> tf.Tensor:
    """Count of family members that were transported."""
    transported_str = tf.strings.as_string(transported)
    stacked = tf.stack([last_name, transported_str])
    count_key = tf.strings.reduce_join(stacked, separator="_", axis=0)

    transported_true = tf.fill(
        dims=tf.shape(last_name), value=tf.constant("1", dtype=tf.string)
    )
    stacked = tf.stack([last_name, transported_true])
    lookup_key = tf.strings.reduce_join(stacked, separator="_", axis=0)

    return value_count_robust(
        count_tensor=count_key,
        lookup_tensor=lookup_key,
        comparison_tensor=last_name,
        vocab_name="transported_family_members_count",
        missing_key=tf.constant("?", dtype=tf.string),
        missing_value=tf.constant(0, dtype=tf.int64),
    )


def family_survival(
    family_members_count: tf.Tensor,
    transported_family_members_count: tf.Tensor,
) -> tf.Tensor:
    """What was the proportion of family members that were transported."""
    return count_proportion(
        transported_family_members_count, family_members_count
    )


def transported_cabin_occupants_count(
    filled_cabin: tf.Tensor, transported: tf.Tensor
) -> tf.Tensor:
    """Count of cabin occupants that were transported."""
    transported_str = tf.strings.as_string(transported)
    stacked = tf.stack([filled_cabin, transported_str])
    count_key = tf.strings.reduce_join(stacked, separator="_", axis=0)

    transported_true = tf.fill(
        dims=tf.shape(filled_cabin), value=tf.constant("1", dtype=tf.string)
    )
    stacked = tf.stack([filled_cabin, transported_true])
    lookup_key = tf.strings.reduce_join(stacked, separator="_", axis=0)

    return value_count_robust(
        count_tensor=count_key,
        lookup_tensor=lookup_key,
        comparison_tensor=filled_cabin,
        vocab_name="transported_cabin_occupants_count",
        missing_key=tf.constant("?/?/?", dtype=tf.string),
        missing_value=tf.constant(0, dtype=tf.int64),
    )


def cabin_survival(
    transported_cabin_occupants_count: tf.Tensor,
    cabin_occupants_count: tf.Tensor,
) -> tf.Tensor:
    """What was the proportion of cabin occupants that were transported."""
    return count_proportion(
        transported_cabin_occupants_count, cabin_occupants_count
    )
