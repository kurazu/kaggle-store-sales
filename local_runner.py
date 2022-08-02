import os

import tfx.v1 as tfx
from absl import logging
from ipdb import iex
from tfx_helper.local import LocalPipelineHelper

from pipeline.constants import PIPELINE_NAME
from pipeline.pipeline import create_pipeline
from pipeline.utils import get_local_output_directory


@iex
def run() -> None:
    """Create and run a pipeline locally."""
    # Location of the CSV files obtained from Kaggle
    training_samples_dir = os.environ["TRAINING_DATA_LOCATION"]
    inference_samples_dir = os.environ["TEST_DATA_LOCATION"]
    # Read the pipeline artifact directory from environment variable
    output_dir = get_local_output_directory()
    # Directory for exporting trained model will be a sub-directory
    # of the pipeline artifact directory.
    serving_model_dir = os.path.join(output_dir, "serving_model")
    # Create pipeline helper instance of local flavour.
    pipeline_helper = LocalPipelineHelper(
        pipeline_name=PIPELINE_NAME,
        output_dir=output_dir,
        # Where should the model be pushed to
        model_push_destination=tfx.proto.PushDestination(
            filesystem=tfx.proto.PushDestination.Filesystem(
                base_directory=serving_model_dir
            )
        ),
    )

    components = create_pipeline(
        # Pass our pipeline helper instance
        pipeline_helper,
        # The rest of the parameters are pipeline-specific.
        data_path=training_samples_dir,
        inference_path=inference_samples_dir,
        # Number of HP search trials
        number_of_trials=100,
        # Aimed accuracy
        eval_accuracy_threshold=0.7,
        # Fast tuning runs
        tune_epochs=30,
        tune_early_stopping_patience=7,
        tune_plateau_patience=2,
        # A bit longer training run
        train_epochs=100,
        train_early_stopping_patience=20,
        train_plateau_patience=3,
        plateau_factor=0.7,
        # Whether to perform HP search or use the previously found values.
        # On the first pipeline run you need to set to `False`.
        use_previous_hparams=True,
    )
    # Since we are running locally, we will want to debug a failing pipeline,
    # so we go into post-mortem debugging if the pipeline fails.
    pipeline_helper.create_and_run_pipeline(components, enable_cache=False)


if __name__ == "__main__":
    logging.set_verbosity(logging.DEBUG)
    run()
