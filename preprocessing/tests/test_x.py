import sys
from typing import Any, Dict, List

import numpy as np
import pytest
import tensorflow as tf

from preprocessing.util import preprocessing_fn_from_hamilton_modules

from . import x as x_module
from .utils import FeatureSpecType, run_tft, wrap_dataset, wrap_output_dataset


@pytest.fixture
def input_dataset() -> List[Dict[str, List[Any]]]:
    return wrap_dataset([None, "Earth", "Mars", "Earth", None], key="Planet")


@pytest.fixture
def eval_dataset() -> List[Dict[str, List[Any]]]:
    return wrap_dataset([None, "Earth", "Jupiter", None], key="Planet")


@pytest.fixture
def input_feature_specs() -> Dict[str, FeatureSpecType]:
    return {"Planet": tf.io.VarLenFeature(dtype=tf.string)}


def test_one_hot(
    input_dataset: List[Dict[str, List[Any]]],
    eval_dataset: List[Dict[str, List[Any]]],
    input_feature_specs: Dict[str, FeatureSpecType],
) -> None:

    expected_dataset = wrap_output_dataset(
        [3, 2, 3, 1, 3, 2, 0, 0],
        key="family_members_count",
        dtype=np.int64,
    )
    expected_test_dataset = wrap_output_dataset(
        [3, 0, 0], key="family_members_count", dtype=np.int64
    )

    preprocessing_fn = preprocessing_fn_from_hamilton_modules(
        [x_module], ["one_hot_planet"]
    )

    (output_dataset, output_test_dataset) = run_tft(
        preprocessing_fn, input_feature_specs, input_dataset, eval_dataset
    )
    breakpoint()
    assert output_dataset == expected_dataset
    assert output_test_dataset == expected_test_dataset
