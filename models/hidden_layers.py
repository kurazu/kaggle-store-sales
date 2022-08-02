import keras_tuner
import tensorflow as tf


def process_hidden_layers(
    inputs: tf.Tensor,
    hyperparams: keras_tuner.HyperParameters,
) -> tf.Tensor:
    """Build hidden layers to process inputs based on hyperparameters."""
    output = inputs
    layer_size = hyperparams["first_hidden_layer_size"]
    for n in range(hyperparams["number_of_hidden_layers"]):
        output = tf.keras.layers.Dense(
            layer_size,
            kernel_regularizer=tf.keras.regularizers.l2(30e-6),
            name=f"hidden_{n}",
            activation=hyperparams["activation"],
        )(output)
        output = tf.keras.layers.BatchNormalization()(output)
        output = tf.keras.layers.Dropout(hyperparams["dropout"])(output)
        layer_size //= 2
    return output
