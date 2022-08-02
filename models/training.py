import os.path
from typing import Callable, List

import keras_tuner
import tensorflow as tf
import tensorflow_transform as tft
from absl import logging
from returns.curry import partial
from tfx import v1 as tfx

from .dataset import input_fn
from .deep import build_and_train_model as build_and_train_deep_model
from .mean_ensemble import (
    build_and_train_model as build_and_train_mean_ensemble_model,
)
from .serving_signatures import (
    get_tf_examples_serving_signature,
    get_transform_features_signature,
)
from .stolen import build_and_train_model as build_and_train_stolen_model
from .train import TrainingConfig
from .weighted_ensemble import (
    build_and_train_model as build_and_train_weighted_ensemble_model,
)


# TFX Trainer will call this function.
def run_fn(
    fn_args: tfx.components.FnArgs,  # type:ignore[name-defined]
) -> None:
    """Train the model based on given args."""
    # Load the Transform component output
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    # The output contains among other attributs the schema of transformed examples
    schema = tf_transform_output.transformed_metadata.schema

    dataset_loader: Callable[[List[str]], tf.data.Dataset] = partial(
        input_fn, data_accessor=fn_args.data_accessor, schema=schema
    )
    train_files = fn_args.train_files
    eval_files = fn_args.eval_files

    # Load best set of hyperparameters from Tuner component
    assert fn_args.hyperparameters, "Expected hyperparameters from Tuner"
    hparams = keras_tuner.HyperParameters.from_config(fn_args.hyperparameters)
    logging.info("Hyper parameters for training: %s", hparams.get_config())

    model_run_dir: str = fn_args.model_run_dir
    training_config = TrainingConfig.parse_obj(fn_args.custom_config)

    with tf.distribute.MirroredStrategy().scope():
        model = build_and_train_mean_ensemble_model(
            dataset_loader=dataset_loader,
            train_files=train_files,
            eval_files=eval_files,
            tf_transform_output=tf_transform_output,
            hparams=hparams,
            model_run_dir=model_run_dir,
            training_config=training_config,
        )

        # We need to manually add the transformation layer to the model instance
        # in order for it to be tracked by the model and included in the saved model format.
        model.tft_layer = tf_transform_output.transform_features_layer()

        # Create signature (endpoints) for the model.
        signatures = {
            # What to do when serving from Tensorflow Serving
            # "serving_default": _get_live_serving_signature(
            #     model, schema, tf_transform_output
            # ),
            # What do do when processing serialized Examples in TFRecord files
            "from_examples": get_tf_examples_serving_signature(
                model, schema, tf_transform_output
            ),
            # # How to perform only preprocessing.
            "transform_features": get_transform_features_signature(
                model, schema, tf_transform_output
            ),
        }

    # Save the model in SavedModel format together with the above signatures.
    # This saved model will be used by all other pipeline components that require
    # a model (for example Evaluator or Pusher).
    model.save(
        fn_args.serving_model_dir, save_format="tf", signatures=signatures
    )
