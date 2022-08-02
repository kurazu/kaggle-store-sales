import tempfile
from typing import Dict, List

import numpy as np
import numpy.typing as npt
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam
from tensorflow_transform.tf_metadata import dataset_metadata, schema_utils

from preprocessing.common import mode, value_count_across_dataset

from .utils import TensorType, run_tft, wrap_dataset, wrap_output_dataset


def test_value_count_across_dataset() -> None:
    train_dataset = wrap_dataset("ABCABA", key="key")
    train_expected = wrap_output_dataset(
        [3, 2, 1, 3, 2, 3], key="key_count", dtype=np.int64
    )
    test_dataset = wrap_dataset("DCBAE", key="key")
    test_expected = wrap_output_dataset(
        [0, 1, 2, 3, 0], key="key_count", dtype=np.int64
    )

    def preprocessing_fn(
        inputs: Dict[str, TensorType]
    ) -> Dict[str, TensorType]:
        return {
            "key_count": value_count_across_dataset(
                inputs["key"],
                vocab_name="key_count",
                missing_value=tf.constant(0, dtype=tf.int64),
            )
        }

    input_feature_specs = {
        "key": tf.io.FixedLenFeature(shape=[1], dtype=tf.string),
    }

    transformed_train_data, transformed_test_data = run_tft(
        preprocessing_fn, input_feature_specs, train_dataset, test_dataset
    )
    assert transformed_train_data == train_expected
    assert transformed_test_data == test_expected


def test_mode() -> None:
    train_dataset = wrap_dataset(["male", "female", None, "female"], key="sex")
    train_expected = wrap_output_dataset(
        ["male", "female", "female", "female"],
        key="filled_sex",
        dtype=np.string_,
    )
    test_dataset = wrap_dataset(
        [None, "male", None, "female", "male"], key="sex"
    )
    test_expected = wrap_output_dataset(
        ["female", "male", "female", "female", "male"],
        key="filled_sex",
        dtype=np.string_,
    )

    def preprocessing_fn(
        inputs: Dict[str, TensorType]
    ) -> Dict[str, TensorType]:
        return {
            "filled_sex": tft.sparse_tensor_to_dense_with_shape(
                inputs["sex"],
                shape=[None, 1],
                default_value=mode(inputs["sex"]),
            )
        }

    input_feature_specs = {
        "sex": tf.io.VarLenFeature(dtype=tf.string),
    }

    transformed_train_data, transformed_test_data = run_tft(
        preprocessing_fn, input_feature_specs, train_dataset, test_dataset
    )
    assert transformed_train_data == train_expected
    assert transformed_test_data == test_expected
