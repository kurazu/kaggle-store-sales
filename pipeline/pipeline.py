from typing import Iterable, List

import tensorflow_model_analysis as tfma
from tfx import v1 as tfx
from tfx.dsl.components.base.base_component import BaseComponent
from tfx.proto import bulk_inferrer_pb2  # type:ignore[attr-defined]
from tfx_helper.components.threshold_optimizer.component import (
    BinaryClassificationThresholdOptimizer,
)
from tfx_helper.interface import PipelineHelperInterface

from preprocessing.features import TRANSFORMED_TARGET_KEY

from .components.inference_exporter.component import InferenceExporter


def create_pipeline(
    pipeline_helper: PipelineHelperInterface,
    *,
    data_path: str,
    inference_path: str,
    number_of_trials: int,  # number of hyperparam tuning trials
    eval_accuracy_threshold: float = 0.7,  # minimal accuracy required to bless the model
    # Proportions of examples in train, validation, test sets.
    cv_splits: int = 5,
    train_early_stopping_patience: int,  # early stopping patience (in epochs) in trainer
    train_plateau_patience: int,  # patience (in epochs) before LR is reduced
    tune_early_stopping_patience: int,  # early stopping patience (in epochs) in tuner
    tune_plateau_patience: int,  # patience (in epochs) bofore LR is reduced
    plateau_factor: float,  # scale LR by this factor on plateau
    train_epochs: int,  # maximum number of training epochs in trainer
    tune_epochs: int,  # maximum number of training epochs in tuner
    # set to `True` to skip tuning in this run and use hyperparams from previous run
    use_previous_hparams: bool,
    # Whether to push the model to an endpoint
    push_model: bool = False,
) -> Iterable[BaseComponent]:
    """Pipeline definition."""
    train_splits: List[str] = [f"train-cv-{idx}" for idx in range(cv_splits)]
    first_train_split, *other_train_splits = train_splits
    # Import and split training data from CSV file into TFRecord files
    splits = [
        *(
            tfx.proto.SplitConfig.Split(name=split, hash_buckets=1)
            for split in train_splits
        ),
        # tfx.proto.SplitConfig.Split(name="valid", hash_buckets=1),
        tfx.proto.SplitConfig.Split(name="eval", hash_buckets=1),
    ]
    output_config = tfx.proto.Output(
        split_config=tfx.proto.SplitConfig(
            splits=splits, partition_feature_name="PassengerId"
        ),
    )
    example_gen = tfx.components.CsvExampleGen(  # type:ignore[attr-defined]
        input_base=data_path,
        output_config=output_config,
    )
    yield example_gen

    # Computes statistics over data for visualization and example validation.
    raw_statistics_gen = (
        tfx.components.StatisticsGen(  # type:ignore[attr-defined]
            examples=example_gen.outputs["examples"]
        ).with_id("raw_stats_gen")
    )
    yield raw_statistics_gen

    # Generates schema based on statistics files.
    schema_gen = tfx.components.SchemaGen(  # type:ignore[attr-defined]
        statistics=raw_statistics_gen.outputs["statistics"],
        infer_feature_shape=True,
    ).with_id("raw_schema_gen")
    yield schema_gen

    # Performs data preprocessing and feature engineering
    transform = tfx.components.Transform(  # type:ignore[attr-defined]
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        preprocessing_fn="preprocessing.callback.preprocessing_fn",
        splits_config=tfx.proto.SplitsConfig(
            # Fit transformations on some of the training data.
            # Some of the feature engineering we do will work better with
            # full picture.
            analyze=[*train_splits],  # , "valid"],
            # don't transform test set - evaluation will use
            # a model with integrated preprocessing.
            transform=[*train_splits],  # , "valid"],
        ),
    )
    yield transform

    # Generate stats and schema of the transformed dataset.
    # It can help us spot mistakes in preprocessing code.
    transformed_statistics_gen = (
        tfx.components.StatisticsGen(  # type:ignore[attr-defined]
            examples=transform.outputs["transformed_examples"]
        ).with_id("transformed_stats_gen")
    )
    yield transformed_statistics_gen

    transformed_schema_gen = (
        tfx.components.SchemaGen(  # type:ignore[attr-defined]
            statistics=transformed_statistics_gen.outputs["statistics"],
            infer_feature_shape=True,
        ).with_id("transformed_schema_gen")
    )
    yield transformed_schema_gen

    if use_previous_hparams:
        # Find latest best hyperparameters computed in a previous run
        hparams_resolver = tfx.dsl.Resolver(  # type:ignore[attr-defined]
            strategy_class=tfx.dsl.experimental.LatestArtifactStrategy,  # type:ignore[attr-defined]
            hyperparameters=tfx.dsl.Channel(  # type:ignore[attr-defined]
                type=tfx.types.standard_artifacts.HyperParameters
            ),
        ).with_id("latest_hyperparams_resolver")
        yield hparams_resolver
        hparams = hparams_resolver.outputs["hyperparameters"]
    else:
        # Launch hyperparamter tuning to find the best set of hyperparameters.
        tuner = pipeline_helper.construct_tuner(
            tuner_fn="models.tuning.tuner_fn",
            examples=transform.outputs["transformed_examples"],
            transform_graph=transform.outputs["transform_graph"],
            train_args=tfx.proto.TrainArgs(splits=other_train_splits),
            eval_args=tfx.proto.EvalArgs(splits=[first_train_split]),
            custom_config={
                "number_of_trials": number_of_trials,
                "epochs": tune_epochs,
                "early_stopping_patience": tune_early_stopping_patience,
                "plateau_patience": tune_plateau_patience,
                "plateau_factor": plateau_factor,
            },
        )
        yield tuner
        hparams = tuner.outputs["best_hyperparameters"]

    # Train a Tensorflow model
    trainer = pipeline_helper.construct_trainer(
        run_fn="models.training.run_fn",
        # training will operate on examples already preprocessed
        examples=transform.outputs["transformed_examples"],
        # a Tensorflow graph of preprocessing function is exposed so that it
        # can be embedded into the trained model and used when serving.
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        # use hyperparameters from tuning
        hyperparameters=hparams,
        train_args=tfx.proto.TrainArgs(
            splits=train_splits  # split to use for training
        ),
        # Traininer will not use evaluation split as it's doing a CV on train splits
        eval_args=tfx.proto.EvalArgs(
            splits=[first_train_split]
        ),  # split to use for validation
        # custom parameters to the training callback
        custom_config={
            "epochs": train_epochs,
            "early_stopping_patience": train_early_stopping_patience,
            "plateau_patience": train_plateau_patience,
            "plateau_factor": plateau_factor,
        },
    )
    yield trainer

    # Get the latest blessed model for model validation comparison.
    model_resolver = tfx.dsl.Resolver(  # type:ignore[attr-defined]
        strategy_class=tfx.dsl.experimental.LatestBlessedModelStrategy,  # type:ignore[attr-defined]
        model=tfx.dsl.Channel(  # type:ignore[attr-defined]
            type=tfx.types.standard_artifacts.Model
        ),
        model_blessing=tfx.dsl.Channel(  # type:ignore[attr-defined]
            type=tfx.types.standard_artifacts.ModelBlessing
        ),
    ).with_id("latest_blessed_model_resolver")
    yield model_resolver

    # Uses TFMA to compute evaluation statistics over features of a model and
    # perform quality validation of a candidate model (compared to a baseline).
    eval_config = tfma.EvalConfig(
        model_specs=[
            tfma.ModelSpec(
                signature_name="from_examples",
                label_key=TRANSFORMED_TARGET_KEY,
                prediction_key=TRANSFORMED_TARGET_KEY,
                preprocessing_function_names=["transform_features"],
            )
        ],
        slicing_specs=[
            tfma.SlicingSpec(),
            # Generate evaluation also for sub-slices of the dataset.
            tfma.SlicingSpec(feature_keys=["HomePlanet"]),
            tfma.SlicingSpec(feature_keys=["Destination"]),
            tfma.SlicingSpec(feature_keys=["CryoSleep"]),
            tfma.SlicingSpec(feature_keys=["VIP"]),
        ],
        metrics_specs=[
            *tfma.metrics.default_binary_classification_specs(),
            tfma.MetricsSpec(
                metrics=[
                    tfma.MetricConfig(
                        # Metric to use
                        class_name="BinaryAccuracy",
                        threshold=tfma.MetricThreshold(
                            # Require an absolute value of metric to exceed threshold
                            value_threshold=tfma.GenericValueThreshold(
                                lower_bound={"value": eval_accuracy_threshold}
                            ),
                            # Require the candidate model to be better than
                            # previous (baseline) model by given margin
                            change_threshold=tfma.GenericChangeThreshold(
                                direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                                absolute={"value": -1e-10},
                            ),
                        ),
                    ),
                ]
            ),
        ],
    )
    evaluator = tfx.components.Evaluator(  # type:ignore[attr-defined]
        examples=example_gen.outputs["examples"],
        example_splits=["eval"],  # split of examples to use for evaluation
        model=trainer.outputs["model"],  # candidate model
        baseline_model=model_resolver.outputs["model"],  # baseline model
        # Change threshold will be ignored if there is no baseline (first run).
        eval_config=eval_config,
    )
    yield evaluator

    # Find the best threshold to use for classification.
    threshold_optimizer = BinaryClassificationThresholdOptimizer(
        model_evaluation=evaluator.outputs["evaluation"]
    )
    yield threshold_optimizer

    # Pushes the model to a file/endpoint destination if checks passed.
    if push_model:
        pusher = pipeline_helper.construct_pusher(
            model=trainer.outputs["model"],  # model to push
            model_blessing=evaluator.outputs["blessing"],  # required blessing
        )
        yield pusher

    # Load inference dataset and generate its statistics
    inference_splits = [
        tfx.proto.SplitConfig.Split(name="all", hash_buckets=1),
    ]
    inference_output_config = tfx.proto.Output(
        split_config=tfx.proto.SplitConfig(splits=inference_splits),
    )
    inference_examples_gen = (
        tfx.components.CsvExampleGen(  # type:ignore[attr-defined]
            input_base=inference_path,
            output_config=inference_output_config,
        ).with_id("inference_example_gen")
    )
    yield inference_examples_gen

    inference_statistics_gen = (
        tfx.components.StatisticsGen(  # type:ignore[attr-defined]
            examples=inference_examples_gen.outputs["examples"]
        ).with_id("inference_stats_gen")
    )
    yield inference_statistics_gen

    # Validate inference samples against raw training samples schema
    example_validator = (
        tfx.components.ExampleValidator(  # type:ignore[attr-defined]
            statistics=inference_statistics_gen.outputs["statistics"],
            schema=schema_gen.outputs["schema"],
        )
    )
    yield example_validator

    # Execute batch predictions for the inference dataset
    bulk_inferrer = tfx.components.BulkInferrer(  # type:ignore[attr-defined]
        examples=inference_examples_gen.outputs["examples"],
        model=trainer.outputs["model"],
        # model_blessing=evaluator.outputs["blessing"],
        data_spec=bulk_inferrer_pb2.DataSpec(example_splits=["all"]),
        model_spec=bulk_inferrer_pb2.ModelSpec(
            model_signature_name=["from_examples"]
        ),
    )
    yield bulk_inferrer

    # Export batch predictions to a custom CSV file.
    inference_exporter = InferenceExporter(
        inference_result=bulk_inferrer.outputs["inference_result"],
        threshold=threshold_optimizer.outputs["best_threshold"],
    )
    yield inference_exporter
