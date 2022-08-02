from typing import Any, Dict, List

import numpy as np
import pytest
import tensorflow as tf

from preprocessing import names as names_module
from preprocessing.util import preprocessing_fn_from_hamilton_modules

from .utils import FeatureSpecType, run_tft, wrap_dataset, wrap_output_dataset


@pytest.fixture
def input_dataset() -> List[Dict[str, List[Any]]]:
    return wrap_dataset(
        [None, "Richard Morgan", "Isaac Asimov", None], key="Name"
    )


@pytest.fixture
def input_feature_specs() -> Dict[str, FeatureSpecType]:
    return {"Name": tf.io.VarLenFeature(dtype=tf.string)}


def test_last_name(
    input_dataset: List[Dict[str, List[Any]]],
    input_feature_specs: Dict[str, FeatureSpecType],
) -> None:
    expected_dataset = wrap_output_dataset(
        ["?", "Morgan", "Asimov", "?"], key="last_name", dtype=np.string_
    )

    preprocessing_fn = preprocessing_fn_from_hamilton_modules(
        [names_module], ["last_name"]
    )

    (output_dataset,) = run_tft(
        preprocessing_fn, input_feature_specs, input_dataset
    )
    assert output_dataset == expected_dataset


def test_first_name(
    input_dataset: List[Dict[str, List[Any]]],
    input_feature_specs: Dict[str, FeatureSpecType],
) -> None:
    expected_dataset = wrap_output_dataset(
        ["?", "Richard", "Isaac", "?"], key="first_name", dtype=np.string_
    )

    preprocessing_fn = preprocessing_fn_from_hamilton_modules(
        [names_module], ["first_name"]
    )

    (output_dataset,) = run_tft(
        preprocessing_fn, input_feature_specs, input_dataset
    )
    assert output_dataset == expected_dataset


def test_family_count(input_feature_specs: Dict[str, FeatureSpecType]) -> None:
    input_dataset = wrap_dataset(
        [
            "Violet Baudelaire",
            "Josephine Anwhistle",
            "Klaus Baudelaire",
            "Lemony Snicket",
            "Sunny Baudelaire",
            "Isaac Anwhistle",
            None,
            None,
        ],
        key="Name",
    )
    expected_dataset = wrap_output_dataset(
        [3, 2, 3, 1, 3, 2, 0, 0],
        key="family_members_count",
        dtype=np.int64,
    )
    test_dataset = wrap_dataset(
        ["Beatrice Baudelaire", "Count Olaf", None], key="Name"
    )
    expected_test_dataset = wrap_output_dataset(
        [3, 0, 0], key="family_members_count", dtype=np.int64
    )

    preprocessing_fn = preprocessing_fn_from_hamilton_modules(
        [names_module], ["family_members_count"]
    )

    (output_dataset, output_test_dataset) = run_tft(
        preprocessing_fn, input_feature_specs, input_dataset, test_dataset
    )
    assert output_dataset == expected_dataset
    assert output_test_dataset == expected_test_dataset
