import keras_tuner


def get_hyperparameters() -> keras_tuner.HyperParameters:
    """Returns hyperparameters for building a Keras model."""
    hp = keras_tuner.HyperParameters()
    # Defines search space.
    hp.Choice("embedding_size", [16, 32, 64], default=32)
    hp.Choice(
        "initial_learning_rate", [0.5, 0.1, 0.01, 0.001, 0.0001, 0.00001], default=1e-3
    )
    hp.Choice("number_of_hidden_layers", [2, 3, 4], default=3)
    hp.Choice("first_hidden_layer_size", [2048, 1024, 512], default=1024)
    hp.Choice("activation", ["relu", "swish"])
    hp.Choice("dropout", [0.0, 0.25, 0.5], default=0.5)
    return hp
