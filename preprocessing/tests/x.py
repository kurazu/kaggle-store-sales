import tensorflow as tf
import tensorflow_transform as tft


def one_hot_planet(Planet: tf.SparseTensor) -> tf.Tensor:
    planet_vocab = tft.compute_and_apply_vocabulary(
        Planet,
        default_value=-1,
        num_oov_buckets=0,
        vocab_filename="planet",
        name="planet_vocab",
    )
    vocab_size = tft.get_num_buckets_for_transformed_feature(planet_vocab)
    dense_planet = tft.sparse_tensor_to_dense_with_shape(
        planet_vocab, shape=[None, 1], default_value=-1
    )
    return tf.one_hot(
        dense_planet,
        tf.cast(vocab_size, tf.int32),
        tf.constant(1, dtype=tf.int64),
        tf.constant(0, dtype=tf.int64),
    )
