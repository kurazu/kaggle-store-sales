from typing import Dict, List

import numpy as np
import pytest
import tensorflow as tf

from preprocessing import age as age_module
from preprocessing.features import NUM_AGE_BUCKETS
from preprocessing.util import preprocessing_fn_from_hamilton_modules

from .utils import FeatureSpecType, run_tft, wrap_dataset, wrap_output_dataset


@pytest.fixture
def input_dataset() -> List[Dict[str, List[int]]]:
    return wrap_dataset([12, 13, 18, 20, 35, None, 5], key="Age")


@pytest.fixture
def input_feature_specs() -> Dict[str, FeatureSpecType]:
    return {"Age": tf.io.VarLenFeature(dtype=tf.float32)}


def test_passenger_id(
    input_dataset: List[Dict[str, List[int]]],
    input_feature_specs: Dict[str, FeatureSpecType],
) -> None:
    expected_dataset = wrap_output_dataset(
        [0, 1, 1, 1, 1, 1, 0], key="is_adult", dtype=np.int64
    )

    preprocessing_fn = preprocessing_fn_from_hamilton_modules(
        [age_module], ["is_adult"]
    )

    (output_dataset,) = run_tft(
        preprocessing_fn, input_feature_specs, input_dataset
    )
    assert output_dataset == expected_dataset


def test_scaled_age(
    input_dataset: List[Dict[str, List[int]]],
    input_feature_specs: Dict[str, FeatureSpecType],
) -> None:
    preprocessing_fn = preprocessing_fn_from_hamilton_modules(
        [age_module], ["scaled_age"]
    )

    (output_dataset,) = run_tft(
        preprocessing_fn, input_feature_specs, input_dataset
    )
    for entry in output_dataset:
        value = entry["scaled_age"]
        assert value.dtype == np.float32  # type:ignore[comparison-overlap]
    # the missing entry should be filled with mean
    assert output_dataset[5]["scaled_age"] == np.array([0.0], dtype=np.float32)


def test_bucketized_age(
    input_dataset: List[Dict[str, List[int]]],
    input_feature_specs: Dict[str, FeatureSpecType],
) -> None:
    preprocessing_fn = preprocessing_fn_from_hamilton_modules(
        [age_module], ["bucketized_age"]
    )

    (output_dataset,) = run_tft(
        preprocessing_fn, input_feature_specs, input_dataset
    )
    for entry in output_dataset:
        value = entry["bucketized_age"]
        assert value.dtype == np.int64  # type:ignore[comparison-overlap]
        assert 0 <= value < NUM_AGE_BUCKETS
