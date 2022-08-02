from typing import List, Optional

import numpy as np
import pytest
import tensorflow as tf

from preprocessing import categorical as categorical_module
from preprocessing.util import preprocessing_fn_from_hamilton_modules

from .utils import run_tft, wrap_dataset, wrap_output_dataset


@pytest.mark.parametrize(
    "input_values,expected_values",
    [
        (
            ["Europa", "Earth", "Mars", "Earth", None, None],
            ["Europa", "Earth", "Mars", "Earth", "Earth", "Earth"],
        ),
        (
            ["Europa", "Earth", "Mars", "Mars", None, None],
            ["Europa", "Earth", "Mars", "Mars", "Mars", "Mars"],
        ),
    ],
)
def test_home_planet(
    input_values: List[Optional[str]], expected_values: List[str]
) -> None:
    input_dataset = wrap_dataset(input_values, key="HomePlanet")
    expected_dataset = wrap_output_dataset(
        expected_values, key="home_planet", dtype=np.string_
    )

    preprocessing_fn = preprocessing_fn_from_hamilton_modules(
        [categorical_module], ["home_planet"]
    )

    input_feature_specs = {
        "HomePlanet": tf.io.VarLenFeature(dtype=tf.string),
    }

    (output_dataset,) = run_tft(
        preprocessing_fn,
        input_feature_specs,
        input_dataset,
    )
    assert output_dataset == expected_dataset


@pytest.mark.parametrize(
    "input_values,expected_values",
    [
        (
            [
                "TRAPPIST-1e",
                "55 Cancri e",
                "PSO J318.5-22",
                "55 Cancri e",
                None,
                None,
            ],
            [
                "TRAPPIST-1e",
                "55 Cancri e",
                "PSO J318.5-22",
                "55 Cancri e",
                "55 Cancri e",
                "55 Cancri e",
            ],
        ),
        (
            [
                "TRAPPIST-1e",
                "55 Cancri e",
                "PSO J318.5-22",
                "PSO J318.5-22",
                None,
                None,
            ],
            [
                "TRAPPIST-1e",
                "55 Cancri e",
                "PSO J318.5-22",
                "PSO J318.5-22",
                "PSO J318.5-22",
                "PSO J318.5-22",
            ],
        ),
    ],
)
def test_destination(
    input_values: List[Optional[str]], expected_values: List[str]
) -> None:
    input_dataset = wrap_dataset(input_values, key="Destination")
    expected_dataset = wrap_output_dataset(
        expected_values, key="destination", dtype=np.string_
    )

    preprocessing_fn = preprocessing_fn_from_hamilton_modules(
        [categorical_module], ["destination"]
    )

    input_feature_specs = {
        "Destination": tf.io.VarLenFeature(dtype=tf.string),
    }

    (output_dataset,) = run_tft(
        preprocessing_fn,
        input_feature_specs,
        input_dataset,
    )
    assert output_dataset == expected_dataset
