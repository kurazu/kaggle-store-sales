from typing import Any, Dict, List

import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_metadata.proto.v0 import schema_pb2

from preprocessing.features import RAW_LABEL_KEY, TRANSFORMED_TARGET_KEY


def get_tf_examples_serving_signature(
    model: tf.keras.Model,
    schema: schema_pb2.Schema,
    tf_transform_output: tft.TFTransformOutput,
) -> Any:
    """
    Returns a serving signature that accepts `tensorflow.Example`.

    This signature will be used for evaluation or bulk inference.
    """

    @tf.function(
        # Receive examples packed into bytes (unparsed)
        input_signature=[
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
        ]
    )
    def serve_tf_examples_fn(
        serialized_tf_example: tf.Tensor,
    ) -> Dict[str, tf.Tensor]:
        """Returns the output to be used in the serving signature."""
        # Load the schema of raw examples.
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        # Remove label feature since these will not be present at serving time.
        raw_feature_spec.pop(RAW_LABEL_KEY)
        # Parse the examples using schema into raw features
        raw_features = tf.io.parse_example(
            serialized_tf_example, raw_feature_spec
        )

        # Preprocess the raw features
        transformed_features = model.tft_layer(raw_features)

        # Run preprocessed inputs through the model to get the prediction
        outputs = model(transformed_features)

        return {TRANSFORMED_TARGET_KEY: outputs}

    return serve_tf_examples_fn


# Not used at the moment - we don't plan to have online predictions
def _get_live_serving_signature(
    model: tf.keras.Model,
    schema: schema_pb2.Schema,
    tf_transform_output: tft.TFTransformOutput,
) -> Any:
    """
    Returns a serving signature that accepts flat text inputs.

    This signature will be used for online predictions.
    """

    @tf.function(
        input_signature=[
            {
                "login_timestamps": tf.TensorSpec(
                    shape=(None,), dtype=tf.string, name="login_timestamps"
                ),
                "timestamp": tf.TensorSpec(
                    shape=(), dtype=tf.string, name="timestamp"
                ),
                "portal_user__stage_one_completion": tf.TensorSpec(
                    shape=(),
                    dtype=tf.string,
                    name="portal_user__stage_one_completion",
                ),
                "onboarding_record__country_of_citizenship": tf.TensorSpec(
                    shape=(),
                    dtype=tf.string,
                    name="onboarding_record__country_of_citizenship",
                ),
                "onboarding_record__date_of_birth": tf.TensorSpec(
                    shape=(),
                    dtype=tf.string,
                    name="onboarding_record__date_of_birth",
                ),
                "onboarding_record__desired_destinations": tf.TensorSpec(
                    shape=(None,),
                    dtype=tf.string,
                    name="onboarding_record__desired_destinations",
                ),
                "onboarding_record__family_background": tf.TensorSpec(
                    shape=(),
                    dtype=tf.string,
                    name="onboarding_record__family_background",
                ),
                "onboarding_record__gender": tf.TensorSpec(
                    shape=(), dtype=tf.string, name="onboarding_record__gender"
                ),
                "onboarding_record__global_ed_objectives": tf.TensorSpec(
                    shape=(None,),
                    dtype=tf.string,
                    name="onboarding_record__global_ed_objectives",
                ),
                "onboarding_record__intakes": tf.TensorSpec(
                    shape=(None,),
                    dtype=tf.string,
                    name="onboarding_record__intakes",
                ),
                "onboarding_record__level_of_study": tf.TensorSpec(
                    shape=(),
                    dtype=tf.string,
                    name="onboarding_record__level_of_study",
                ),
                "onboarding_record__nationality": tf.TensorSpec(
                    shape=(),
                    dtype=tf.string,
                    name="onboarding_record__nationality",
                ),
                "onboarding_record__preferred_modes": tf.TensorSpec(
                    shape=(None,),
                    dtype=tf.string,
                    name="onboarding_record__preferred_modes",
                ),
                "onboarding_record__preferred_times": tf.TensorSpec(
                    shape=(None,),
                    dtype=tf.string,
                    name="onboarding_record__preferred_times",
                ),
                "onboarding_record__specializations": tf.TensorSpec(
                    shape=(None,),
                    dtype=tf.string,
                    name="onboarding_record__specializations",
                ),
                "onboarding_record__study_areas": tf.TensorSpec(
                    shape=(None,),
                    dtype=tf.string,
                    name="onboarding_record__study_areas",
                ),
                "onboarding_record__years_of_admission": tf.TensorSpec(
                    shape=(None,),
                    dtype=tf.string,
                    name="onboarding_record__years_of_admission",
                ),
            }
        ]
    )
    def serve_live_fn(inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Returns the output to be used in the serving signature."""

        # Preprocess the raw features
        transformed_features = model.tft_layer(inputs)

        # Run preprocessed inputs through the model to get the prediction
        outputs = model(transformed_features)

        # the outputs are of (batch, 1) shape, but for maximum simplicity we want
        # to return (batch, ) shape, so we eliminate the extra dimension.
        return {TRANSFORMED_TARGET_KEY: tf.squeeze(outputs)}

    return serve_live_fn


def get_transform_features_signature(
    model: Any,
    schema: schema_pb2.Schema,
    tf_transform_output: tft.TFTransformOutput,
) -> Any:
    """
    Returns a serving signature that applies tf.Transform to features.

    This signature can be used to pre-process the inputs.
    """

    @tf.function(
        # Receive examples packed into bytes (unparsed)
        input_signature=[
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
        ]
    )
    def transform_features_fn(serialized_tf_example: tf.TensorSpec) -> Any:
        """Returns the transformed features."""

        # Load the schema of raw examples.
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        # Parse the examples using schema into raw features
        raw_features = tf.io.parse_example(
            serialized_tf_example, raw_feature_spec
        )

        # Preprocess the raw features
        transformed_features = model.tft_layer(raw_features)

        return transformed_features

    return transform_features_fn
