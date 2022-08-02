import itertools
from typing import Callable, Dict, List

import keras_tuner
import tensorflow as tf
import tensorflow_transform as tft
from absl import logging

from .hidden_layers import process_hidden_layers
from .inputs import (
    get_boolean_inputs,
    get_bucketized_continous_inputs,
    get_categorical_inputs,
    get_continous_inputs,
)
from .train import TrainingConfig, train_model
from .utils import get_vocabulary_size

NUM_CROSS_BUCKETS = 25


def _get_crossed_features(
    *,
    categorical_inputs: Dict[str, tf.Tensor],
    boolean_inputs: Dict[str, tf.Tensor],
    bucketized_inputs: Dict[str, tf.Tensor],
    embedding_size: int,
) -> tf.Tensor:
    """
    Create feature crosses out of all possible categorical features.
    """
    all_inputs = {**categorical_inputs, **boolean_inputs, **bucketized_inputs}
    crossed_outputs: List[tf.Tensor] = []
    for a_input_name, b_input_name in itertools.combinations(all_inputs, 2):
        a_input = all_inputs[a_input_name]
        b_input = all_inputs[b_input_name]
        feature_name = f"cross__{a_input_name}__{b_input_name}"
        cross_layer = (
            tf.keras.layers.experimental.preprocessing.HashedCrossing(
                num_bins=NUM_CROSS_BUCKETS,
                name=feature_name,
            )
        )
        output = cross_layer((a_input, b_input))

        embedding = tf.keras.layers.Embedding(
            NUM_CROSS_BUCKETS,
            embedding_size,
            input_length=1,
            name=f"{feature_name}__embedding",
        )(output)
        squeezed_embedding = tf.squeeze(
            embedding, axis=1, name=f"{feature_name}__squeezed"
        )
        crossed_outputs.append(squeezed_embedding)

    embedding_features = tf.keras.layers.Concatenate(
        axis=-1, name="concatenated_cross_embeddings"
    )(crossed_outputs)
    return embedding_features


def _get_continous_features(
    *, continous_inputs: Dict[str, tf.Tensor]
) -> tf.Tensor:
    continous_features = tf.keras.layers.Concatenate(axis=-1)(
        continous_inputs.values()
    )
    return continous_features


def _get_embedding_features(
    *,
    categorical_inputs: Dict[str, tf.Tensor],
    boolean_inputs: Dict[str, tf.Tensor],
    transform_output: tft.TFTransformOutput,
    embedding_size: int,
) -> tf.Tensor:
    categorical_embeddings = {
        feature_name: tf.keras.layers.Embedding(
            get_vocabulary_size(transform_output, feature_name),
            embedding_size,
            input_length=1,
            name=f"{feature_name}__embedding",
        )(feature)
        for feature_name, feature in categorical_inputs.items()
    }
    boolean_embeddings = {
        feature_name: tf.keras.layers.Embedding(
            2,
            embedding_size,
            input_length=1,
            name=f"{feature_name}__embedding",
        )(feature)
        for feature_name, feature in boolean_inputs.items()
    }
    squeezed_embeddings = [
        tf.squeeze(embedding, axis=1, name=f"{feature_name}__squeezed")
        for feature_name, embedding in {
            **categorical_embeddings,
            **boolean_embeddings,
        }.items()
    ]
    embedding_features = tf.keras.layers.Concatenate(
        axis=-1, name="concatenated_embeddings"
    )(squeezed_embeddings)
    return embedding_features


def build_model(
    hparams: keras_tuner.HyperParameters,
    transform_output: tft.TFTransformOutput,
) -> tf.keras.Model:
    """Creates a DNN Keras model for classifying text."""
    boolean_inputs = get_boolean_inputs()
    continous_inputs = get_continous_inputs()
    categorical_inputs = get_categorical_inputs()
    bucketized_inputs = get_bucketized_continous_inputs()
    all_inputs = {
        **boolean_inputs,
        **continous_inputs,
        **categorical_inputs,
        **bucketized_inputs,
    }

    embedding_size = hparams["embedding_size"]
    crossed_features = _get_crossed_features(
        categorical_inputs=categorical_inputs,
        boolean_inputs=boolean_inputs,
        bucketized_inputs=bucketized_inputs,
        embedding_size=embedding_size,
    )
    continous_features = _get_continous_features(
        continous_inputs=continous_inputs
    )
    embedding_features = _get_embedding_features(
        categorical_inputs=categorical_inputs,
        boolean_inputs=boolean_inputs,
        transform_output=transform_output,
        embedding_size=embedding_size,
    )

    dense_features = tf.keras.layers.Concatenate(
        axis=-1, name="dense_features"
    )([continous_features, embedding_features])

    # Then we run the inputs through a neural net
    # with few hidden layers and ReLU activations
    hidden_output = process_hidden_layers(
        dense_features,
        hparams,
    )

    concatenation = tf.keras.layers.Concatenate(axis=-1, name="all_features")(
        [hidden_output, crossed_features]
    )

    # And we expect a single output in range <0, 1>
    predictions = tf.keras.layers.Dense(
        1, activation="sigmoid", name="transported_prediction"
    )(concatenation)

    # Compile the model
    model = tf.keras.Model(inputs=all_inputs, outputs=predictions)
    model.compile(
        loss="binary_crossentropy",  # loss appropriate for binary classification
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hparams["initial_learning_rate"]
        ),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(
                name="binary_accuracy", threshold=0.5
            ),
        ],
    )
    model.summary(print_fn=logging.info)
    return model


def build_and_train_model(
    *,
    dataset_loader: Callable[[List[str]], tf.data.Dataset],
    train_files: List[str],
    eval_files: List[str],
    tf_transform_output: tft.TFTransformOutput,
    hparams: keras_tuner.HyperParameters,
    model_run_dir: str,
    training_config: TrainingConfig,
) -> tf.keras.Model:
    # Build the model using the best hyperparameters
    model = build_model(hparams, transform_output=tf_transform_output)
    train_dataset = dataset_loader(train_files)
    eval_dataset = dataset_loader(eval_files)

    train_model(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        model_run_dir=model_run_dir,
        training_config=training_config,
    )

    return model
