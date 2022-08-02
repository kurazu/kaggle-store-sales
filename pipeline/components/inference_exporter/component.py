import tfx.v1 as tfx
from absl import logging
from tfx.dsl.component.experimental.decorators import component

from .utils import export_predictions, read_predictions


@component  # type:ignore
def InferenceExporter(
    inference_result: tfx.dsl.components.InputArtifact[  # type: ignore
        tfx.types.standard_artifacts.InferenceResult
    ],
    threshold: tfx.dsl.components.InputArtifact[  # type: ignore
        tfx.types.standard_artifacts.Float
    ],
    csv_inference_result: tfx.dsl.components.OutputArtifact[  # type: ignore
        tfx.types.standard_artifacts.InferenceResult
    ],
) -> tfx.dsl.components.OutputDict():  # type:ignore
    """
    Export TFRecord inference results to a custom CSV file.
    """
    source_dir: str = inference_result.uri
    target_dir: str = csv_inference_result.uri
    threshold.read()
    threshold_value: float = threshold.value
    logging.info(
        "Starting to export inference results from %r to %r with threshold %.3f",
        source_dir,
        target_dir,
        threshold_value,
    )
    predictions = read_predictions(source_dir, threshold_value)
    export_predictions(predictions, target_dir)
    logging.info(
        "Finished exporting inference result from %r to %r",
        source_dir,
        target_dir,
    )
    return {}
