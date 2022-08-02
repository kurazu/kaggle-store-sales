import datetime
import os
from typing import Optional


def get_gcp_project() -> str:
    """Get the name of the Google Cloud Platform project from environment variables."""
    return os.environ["GCP_PROJECT"]


def get_gcp_region() -> str:
    """Get the Google Cloud Platform region from environment variables."""
    return os.environ["GCP_REGION"]


def get_gcp_service_account() -> str:
    """Get the Google Cloud Platform service account name from environment variables."""
    return os.environ["GCP_SERVICE_ACCOUNT"]


def get_gcp_docker_image() -> str:
    """Get the URI of the TFX-desceded docker image to use when running the pipeline from environment variables."""
    return os.environ["GCP_DOCKER_IMAGE"]


def get_gcp_output_directory() -> str:
    """Get the Google Cloud Storage path for pipeline artifacts from environment variables."""
    return os.environ["GCP_PIPELINE_OUTPUT"]


def get_local_output_directory() -> str:
    """Get a local filesystem path for pipeline artifacts from environment variables."""
    return os.environ["PIPELINE_OUTPUT"]


def get_timestamped_file_path(
    directory: str, base_name: str, extension: Optional[str]
) -> str:
    """Creates a timestamped filename in a specified directory."""
    now = datetime.datetime.utcnow()
    if extension is None:
        filename = f"{base_name}.{now:%Y_%m_%d_%H_%M_%S}"
    else:
        filename = f"{base_name}.{now:%Y_%m_%d_%H_%M_%S}.{extension}"
    return os.path.join(directory, filename)


def get_gcp_train_data() -> str:
    """Get the Google Cloud Storage path for training data."""
    return os.environ["GCP_TRAIN_DATA"]


def get_gcp_inference_data() -> str:
    """Get the Google Cloud Storage path for inference data."""
    return os.environ["GCP_INFERENCE_DATA"]
