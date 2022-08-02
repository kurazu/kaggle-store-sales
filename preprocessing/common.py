import contextlib
from typing import Optional, Tuple

import tensorflow as tf
import tensorflow_transform as tft
from returns.curry import partial


def mode(x: tf.SparseTensor) -> tf.Tensor:
    """Find mode (most common value) for a categorical sparse feature."""
    values, counts = tft.count_per_key(x)
    most_common_index = tf.argmax(counts)
    mode = values[most_common_index]
    return mode


def count_per_key_lookup_fn(
    key: tf.Tensor,
    deferred_vocab_filename_tensor: tf.Tensor,
    missing_value: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Lookup function for vocabularies produced by `tft.count_per_key`.

    The format of the vocabulary file is `count` `space` `key`:
    2 abc
    1 def
    """
    with contextlib.ExitStack() as stack:
        if tf.executing_eagerly():  # pragma: no cover
            stack.enter_context(tf.init_scope())
        initializer = tf.lookup.TextFileInitializer(
            filename=deferred_vocab_filename_tensor,
            key_dtype=tf.string,
            key_index=1,
            value_dtype=tf.int64,
            value_index=0,
            delimiter=" ",
        )
        table = tf.lookup.StaticHashTable(
            initializer, default_value=missing_value
        )
        size = table.size()
    return table.lookup(key), size


def value_count_across_dataset(
    input_values: tf.Tensor,
    *,
    vocab_name: str,
    missing_key: Optional[tf.Tensor] = None,
    missing_value: tf.Tensor
) -> tf.Tensor:
    """
    Return number of occurences of given value across the dataset.

    analyze_and_transform: A B C B A A => 3 2 1 2 3 3
    transform: D A C B => 0 3 1 2
    """
    return value_count_robust(
        count_tensor=input_values,
        lookup_tensor=input_values,
        comparison_tensor=input_values,
        vocab_name=vocab_name,
        missing_key=missing_key,
        missing_value=missing_value,
    )


def value_count_robust(
    *,
    count_tensor: tf.Tensor,
    lookup_tensor: tf.Tensor,
    comparison_tensor: tf.Tensor,
    vocab_name: str,
    missing_key: Optional[tf.Tensor] = None,
    missing_value: tf.Tensor
) -> tf.Tensor:
    count_tensor = tf.ensure_shape(count_tensor, [None, 1])
    flat_count_tensor = tf.squeeze(count_tensor, axis=-1)
    vocab_filename = tft.count_per_key(
        flat_count_tensor, key_vocabulary_filename=vocab_name
    )
    counts = tft.apply_vocabulary(
        lookup_tensor,
        vocab_filename,
        lookup_fn=partial(
            count_per_key_lookup_fn, missing_value=missing_value
        ),
    )
    if missing_key is None:
        return counts
    else:
        return tf.where(
            comparison_tensor == missing_key, missing_value, counts
        )


def count_proportion(
    nominator: tf.Tensor, denominator: tf.Tensor
) -> tf.Tensor:
    """
    Calculates proportion of two integer tensors.

    Sample:
    nominator=[1, 2, 0, 0], denominator=[3, 5, 1, 0] => [0.33, 0.4, 0, 0].

    If denominator is zero, returns 0.
    """
    return tf.math.divide_no_nan(
        tf.cast(nominator, tf.float32),
        tf.cast(denominator, tf.float32),
    )
