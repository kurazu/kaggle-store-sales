from typing import Dict, List

import numpy as np
import pytest
import tensorflow as tf

from preprocessing import cabin as cabin_module
from preprocessing.features import NUM_CABIN_NUM_BUCKETS
from preprocessing.util import preprocessing_fn_from_hamilton_modules

from .utils import run_tft, wrap_dataset, wrap_output_dataset


@pytest.fixture
def input_dataset() -> List[Dict[str, List[str]]]:
    return wrap_dataset(
        [
            "B/0/P",
            "F/1/S",
            "E/0/S",
            "G/13/S",
            "F/21/P",
            None,
        ],
        key="Cabin",
    )


@pytest.fixture
def input_feature_specs() -> Dict[str, tf.io.VarLenFeature]:
    return {"Cabin": tf.io.VarLenFeature(dtype=tf.string)}


def test_cabin_deck(
    input_dataset: List[Dict[str, List[str]]],
    input_feature_specs: Dict[str, tf.io.VarLenFeature],
) -> None:
    expected_dataset = wrap_output_dataset(
        ["B", "F", "E", "G", "F", "?"], key="cabin_deck", dtype=np.string_
    )

    preprocessing_fn = preprocessing_fn_from_hamilton_modules(
        [cabin_module], ["cabin_deck"]
    )

    (output_dataset,) = run_tft(
        preprocessing_fn, input_feature_specs, input_dataset
    )
    assert output_dataset == expected_dataset


def test_cabin_deck_vocab(
    input_dataset: List[Dict[str, List[str]]],
    input_feature_specs: Dict[str, tf.io.VarLenFeature],
) -> None:
    preprocessing_fn = preprocessing_fn_from_hamilton_modules(
        [cabin_module], ["cabin_deck_vocab"]
    )

    (output_dataset,) = run_tft(
        preprocessing_fn, input_feature_specs, input_dataset
    )
    for entry in output_dataset:
        array = entry["cabin_deck_vocab"]
        assert array.dtype == np.int64
        (value,) = array
        assert 0 <= value <= 4


def test_cabin_side(
    input_dataset: List[Dict[str, List[str]]],
    input_feature_specs: Dict[str, tf.io.VarLenFeature],
) -> None:
    expected_dataset = wrap_output_dataset(
        ["P", "S", "S", "S", "P", "?"], key="cabin_side", dtype=np.string_
    )

    preprocessing_fn = preprocessing_fn_from_hamilton_modules(
        [cabin_module], ["cabin_side"]
    )

    (output_dataset,) = run_tft(
        preprocessing_fn, input_feature_specs, input_dataset
    )
    assert output_dataset == expected_dataset


def test_cabin_side_vocab(
    input_dataset: List[Dict[str, List[str]]],
    input_feature_specs: Dict[str, tf.io.VarLenFeature],
) -> None:
    preprocessing_fn = preprocessing_fn_from_hamilton_modules(
        [cabin_module], ["cabin_side_vocab"]
    )

    (output_dataset,) = run_tft(
        preprocessing_fn, input_feature_specs, input_dataset
    )
    for entry in output_dataset:
        array = entry["cabin_side_vocab"]
        assert array.dtype == np.int64
        (value,) = array
        assert 0 <= value <= 2


def test_cabin_num(
    input_dataset: List[Dict[str, List[str]]],
    input_feature_specs: Dict[str, tf.io.VarLenFeature],
) -> None:
    expected_dataset = wrap_output_dataset(
        [0, 1, 0, 13, 21, -1], key="cabin_num", dtype=np.int64
    )

    preprocessing_fn = preprocessing_fn_from_hamilton_modules(
        [cabin_module], ["cabin_num"]
    )

    (output_dataset,) = run_tft(
        preprocessing_fn, input_feature_specs, input_dataset
    )
    assert output_dataset == expected_dataset


def test_scaled_cabin_num(
    input_dataset: List[Dict[str, List[str]]],
    input_feature_specs: Dict[str, tf.io.VarLenFeature],
) -> None:
    preprocessing_fn = preprocessing_fn_from_hamilton_modules(
        [cabin_module], ["scaled_cabin_num"]
    )

    (output_dataset,) = run_tft(
        preprocessing_fn, input_feature_specs, input_dataset
    )
    for entry in output_dataset:
        array = entry["scaled_cabin_num"]
        assert array.dtype == np.float32  # type:ignore[comparison-overlap]
        (value,) = array
        assert 0.0 <= value <= 1.0


def test_bucketized_cabin_num(
    input_dataset: List[Dict[str, List[str]]],
    input_feature_specs: Dict[str, tf.io.VarLenFeature],
) -> None:
    preprocessing_fn = preprocessing_fn_from_hamilton_modules(
        [cabin_module], ["bucketized_cabin_num"]
    )

    (output_dataset,) = run_tft(
        preprocessing_fn, input_feature_specs, input_dataset
    )
    for entry in output_dataset:
        array = entry["bucketized_cabin_num"]
        assert array.dtype == np.int64  # type:ignore[comparison-overlap]
        (value,) = array
        assert 0 <= value < NUM_CABIN_NUM_BUCKETS
