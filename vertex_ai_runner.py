from absl import logging
from tfx_helper.interface import Resources
from tfx_helper.vertex_ai import VertexAIPipelineHelper

from pipeline.constants import PIPELINE_NAME
from pipeline.pipeline import create_pipeline
from pipeline.utils import (
    get_gcp_docker_image,
    get_gcp_inference_data,
    get_gcp_output_directory,
    get_gcp_project,
    get_gcp_region,
    get_gcp_service_account,
    get_gcp_train_data,
)


def run() -> None:
    # minimal (less than the standard `e2-standard-4`) resource for components
    # that won't execute computations
    minimal_resources = Resources(cpu=1, memory=4)
    # create a helper instance of cloud flavour
    pipeline_helper = VertexAIPipelineHelper(
        pipeline_name=PIPELINE_NAME,
        output_dir=get_gcp_output_directory(),
        google_cloud_project=get_gcp_project(),
        google_cloud_region=get_gcp_region(),
        # all the components will use our custom image for running
        docker_image=get_gcp_docker_image(),
        service_account=get_gcp_service_account(),
        # name of the Vertex AI Endpoint
        serving_endpoint_name=PIPELINE_NAME,
        # NUmber of parallel hyperparameter tuning trails
        num_parallel_trials=4,
        # GPU for Trainer and Tuner components
        trainer_accelerator_type=None,
        # Machine type for Trainer and Tuner components
        trainer_machine_type="n1-standard-4",
        # GPU for serving endpoint
        serving_accelerator_type=None,
        # Machine type for serving endpoint
        serving_machine_type="n1-standard-4",
        # Override resource requirements of components. The dictionary key is the ID
        # of the component (usually class name, unless changed with `with_id` method).
        resource_overrides={
            # evaluator needs more RAM than standard machine can provide
            "Evaluator": Resources(cpu=16, memory=32),
            # training is done as Vertex job on a separate machine
            "Trainer": minimal_resources,
            # tuning is done as Vertex job on a separate set of machines
            "Tuner": minimal_resources,
            # pusher is just submitting a job
            "Pusher": minimal_resources,
        },
    )
    components = create_pipeline(
        pipeline_helper,
        # Input data in Cloud Storage
        data_path=get_gcp_train_data(),
        inference_path=get_gcp_inference_data(),
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
        use_previous_hparams=False,
    )
    pipeline_helper.create_and_run_pipeline(components, enable_cache=False)


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    run()
