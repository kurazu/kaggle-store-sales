import keras_tuner
import tensorflow as tf
import tensorflow_transform as tft
from returns.curry import partial
from tfx import v1 as tfx

from .dataset import input_fn
from .deep import build_model
from .hyperparameters import get_hyperparameters
from .train import TrainingConfig, get_callbacks


class TuningConfig(TrainingConfig):
    number_of_trials: int


def build_model_with_strategy(
    hparams: keras_tuner.HyperParameters,
    transform_output: tft.TFTransformOutput,
) -> tf.keras.Model:
    # The strategy should pick all GPUs if available, otherwise all CPUs automatically
    with tf.distribute.MirroredStrategy().scope():
        return build_model(hparams, transform_output)


# TFX Tuner will call this function.
def tuner_fn(
    fn_args: tfx.components.FnArgs,  # type:ignore[name-defined]
) -> tfx.components.TunerFnResult:  # type:ignore[name-defined]
    """Build the tuner using the KerasTuner API."""
    # Load the Transform component output
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    # Get the preprocessed inputs schema
    schema = tf_transform_output.transformed_metadata.schema

    tuning_config = TuningConfig.parse_obj(fn_args.custom_config)

    # RandomSearch is a subclass of keras_tuner.Tuner which inherits from
    # BaseTuner.
    tuner = keras_tuner.RandomSearch(
        partial(
            build_model_with_strategy,
            transform_output=tf_transform_output,
        ),  # model building callback
        max_trials=tuning_config.number_of_trials,  # number of trials to perform
        hyperparameters=get_hyperparameters(),  # hyperparameter search space
        allow_new_entries=False,  # don't allow requesting parameters outside the search space
        # We want to choose the set of hyperparms that causes fastest convergence
        # so we will select validation loss minimalization as objective.
        objective=keras_tuner.Objective("val_loss", "min"),
        directory=fn_args.working_dir,  # operating directory
        project_name="spaceship_titanic_tuning",  # will be used to add prefix to artifacts
    )

    # Load datasets
    train_dataset = input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        schema,
    )
    eval_dataset = input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        schema,
    )

    callbacks = get_callbacks(
        # fn_args.model_run_dir is None
        model_run_dir=fn_args.model_run_dir,
        training_config=tuning_config,
    )

    # Request hyperparameter tuning
    return tfx.components.TunerFnResult(  # type:ignore[attr-defined]
        tuner=tuner,
        fit_kwargs={
            "x": train_dataset,
            "validation_data": eval_dataset,
            "epochs": tuning_config.epochs,
            "callbacks": callbacks,
        },
    )
