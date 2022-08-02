from typing import List, Optional

import tensorflow as tf
from pydantic import BaseModel


class TrainingConfig(BaseModel):
    epochs: int
    early_stopping_patience: int
    plateau_patience: int
    plateau_factor: float


def get_callbacks(
    model_run_dir: Optional[str], training_config: TrainingConfig
) -> List[tf.keras.callbacks.Callback]:
    callbacks: List[tf.keras.callbacks.Callback] = []
    if model_run_dir is not None:
        # Write logs together with model, so that we can later view the training
        # curves in Tensorboard.
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=model_run_dir, update_freq="batch"
        )
        callbacks.append(tensorboard_callback)

    plateau_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_binary_accuracy",
        mode="max",
        factor=training_config.plateau_factor,
        patience=training_config.plateau_patience,
        verbose=1,
    )
    callbacks.append(plateau_callback)

    terminate_callback = tf.keras.callbacks.TerminateOnNaN()
    callbacks.append(terminate_callback)

    # Configure early stopping on validation loss increase
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_binary_accuracy",
        mode="max",
        patience=training_config.early_stopping_patience,
        verbose=1,
        restore_best_weights=True,
    )
    callbacks.append(early_stopping_callback)
    return callbacks


def train_model(
    *,
    model: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    eval_dataset: tf.data.Dataset,
    model_run_dir: str,
    training_config: TrainingConfig,
) -> None:
    """
    Train a keras model with logging, lr adjustment and early stopping.
    """

    callbacks = get_callbacks(
        model_run_dir=model_run_dir, training_config=training_config
    )

    # Train the model in the usual Keras fashion
    model.fit(
        train_dataset,
        # steps_per_epoch=fn_args.train_steps,
        epochs=training_config.epochs,
        validation_data=eval_dataset,
        # validation_steps=fn_args.eval_steps,
        callbacks=callbacks,
    )
