import os.path
from typing import Callable, List

import keras_tuner
import tensorflow as tf
import tensorflow_transform as tft
from absl import logging

from .deep import build_and_train_model as build_and_train_deep_model
from .inputs import (
    get_boolean_inputs,
    get_bucketized_continous_inputs,
    get_categorical_inputs,
    get_continous_inputs,
)
from .train import TrainingConfig, train_model


def build_model(models: List[tf.keras.Model]) -> tf.keras.Model:
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
    model_outputs = [model(all_inputs, training=False) for model in models]
    outputs = tf.keras.layers.Concatenate(
        axis=-1, name="concatenated_model_outputs"
    )(model_outputs)

    # Average the predictions
    predictions = tf.reshape(
        tf.reduce_mean(outputs, axis=-1, name="mean_prediction"), [-1, 1]
    )
    ensemble_model = tf.keras.Model(
        inputs=all_inputs,
        outputs=predictions,
        name="ensemble_model",
    )
    ensemble_model.compile(
        loss="binary_crossentropy",  # loss appropriate for binary classification
        metrics=[
            tf.keras.metrics.BinaryAccuracy(
                name="binary_accuracy", threshold=0.5
            ),
        ],
    )
    ensemble_model.summary(print_fn=logging.info)
    return ensemble_model


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
    logging.info("Will train ensemble model with %d members", len(train_files))
    # train few commitee members, each on a different fold of the training dataset.
    models: List[tf.keras.Model] = []
    for i, member_val_file in enumerate(train_files):
        logging.info("Training ensemble member %d", i)
        # Select subset of data
        member_train_files = list(train_files)
        member_train_files.remove(member_val_file)
        # Build member model
        model = build_and_train_deep_model(
            dataset_loader=dataset_loader,
            train_files=member_train_files,
            eval_files=[member_val_file],
            tf_transform_output=tf_transform_output,
            hparams=hparams,
            model_run_dir=os.path.join(model_run_dir, f"member_{i}"),
            training_config=training_config,
        )
        # Freeze trained models to prevent them from
        # having their weights adjusted in further training phase
        model.trainable = False

        models.append(model)

    logging.info("Building ensemble model")
    # Build an ensemble model out of smaller models
    model = build_model(models)

    # This model does not need to be trained
    return model
