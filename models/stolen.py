from typing import Callable, Dict, List

import keras_tuner
import tensorflow as tf
import tensorflow_transform as tft
from absl import logging

from .deep import _get_continous_features
from .inputs import (
    get_boolean_inputs,
    get_categorical_inputs,
    get_continous_inputs,
)
from .train import train_model
from .utils import get_vocabulary_size


def _get_one_hot_features(
    *,
    categorical_inputs: Dict[str, tf.Tensor],
    boolean_inputs: Dict[str, tf.Tensor],
    transform_output: tft.TFTransformOutput,
) -> tf.Tensor:
    categorical_embeddings = {
        feature_name: tf.one_hot(
            feature,
            depth=get_vocabulary_size(transform_output, feature_name),
            on_value=tf.constant(1.0, dtype=tf.float32),
            off_value=tf.constant(0.0, dtype=tf.float32),
            name=f"{feature_name}__one_hot",
        )
        for feature_name, feature in categorical_inputs.items()
    }
    boolean_embeddings = {
        feature_name: tf.one_hot(
            feature,
            depth=2,
            on_value=tf.constant(1.0, dtype=tf.float32),
            off_value=tf.constant(0.0, dtype=tf.float32),
            name=f"{feature_name}__one_hot",
        )
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
    boolean_inputs = get_boolean_inputs()
    continous_inputs = get_continous_inputs()
    categorical_inputs = get_categorical_inputs()

    all_inputs = {**boolean_inputs, **categorical_inputs, **continous_inputs}

    continous_features = _get_continous_features(
        continous_inputs=continous_inputs
    )
    embedding_features = _get_one_hot_features(
        categorical_inputs=categorical_inputs,
        boolean_inputs=boolean_inputs,
        transform_output=transform_output,
    )

    dense_features = tf.keras.layers.Concatenate(
        axis=-1, name="dense_features"
    )([continous_features, embedding_features])

    activation_func = "swish"

    x = tf.keras.layers.Dense(
        1024,
        # use_bias  = True,
        kernel_regularizer=tf.keras.regularizers.l2(30e-6),
        activation=activation_func,
    )(dense_features)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dense(
        256,
        # use_bias  = True,
        kernel_regularizer=tf.keras.regularizers.l2(30e-6),
        activation=activation_func,
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dense(
        128,
        # use_bias  = True,
        kernel_regularizer=tf.keras.regularizers.l2(30e-6),
        activation=activation_func,
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dense(
        1,
        # use_bias  = True,
        # kernel_regularizer = tf.keras.regularizers.l2(30e-6),
        activation="sigmoid",
    )(x)

    model = tf.keras.Model(inputs=all_inputs, outputs=x)
    model.compile(
        loss="binary_crossentropy",  # loss appropriate for binary classification
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
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
    epochs: int,
    patience: int,
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
        epochs=300,
        patience=12,
    )

    return model
