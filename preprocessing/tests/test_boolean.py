from typing import Any, Dict, List, Optional, cast

import numpy as np
import pytest
import tensorflow as tf
from typing_extensions import Literal, TypedDict

from preprocessing import boolean as boolean_module
from preprocessing.util import preprocessing_fn_from_hamilton_modules

from .utils import (
    run_tft,
    wrap_dataset,
    wrap_multi_column_dataset,
    wrap_output_dataset,
)


class CryoTestCase(TypedDict):
    CryoSleep: List[Literal["True", "False"]]
    RoomService: List[float]
    FoodCourt: List[float]
    ShoppingMall: List[float]
    Spa: List[float]
    VRDeck: List[float]


def test_cryo_sleep() -> None:
    input_dataset: List[CryoTestCase] = [
        # Awake passenger with all fields present
        {
            "CryoSleep": ["False"],
            "RoomService": [10.5],
            "FoodCourt": [25.0],
            "ShoppingMall": [15.6],
            "Spa": [80.25],
            "VRDeck": [17.0],
        },
        # Awake passenger with all fields present but no expenses
        {
            "CryoSleep": ["False"],
            "RoomService": [0.0],
            "FoodCourt": [0.0],
            "ShoppingMall": [0.0],
            "Spa": [0.0],
            "VRDeck": [0.0],
        },
        # Awake passenger with some fields missing
        {
            "CryoSleep": ["False"],
            "RoomService": [],
            "FoodCourt": [50.0],
            "ShoppingMall": [0.0],
            "Spa": [],
            "VRDeck": [75.0],
        },
        # Sleeping passengers with all fields present
        {
            "CryoSleep": ["True"],
            "RoomService": [0.0],
            "FoodCourt": [0.0],
            "ShoppingMall": [0.0],
            "Spa": [0.0],
            "VRDeck": [0.0],
        },
        {
            "CryoSleep": ["True"],
            "RoomService": [0.0],
            "FoodCourt": [0.0],
            "ShoppingMall": [0.0],
            "Spa": [0.0],
            "VRDeck": [0.0],
        },
        {
            "CryoSleep": ["True"],
            "RoomService": [0.0],
            "FoodCourt": [0.0],
            "ShoppingMall": [0.0],
            "Spa": [0.0],
            "VRDeck": [0.0],
        },
        # Sleeping passenger with some fields missing
        {
            "CryoSleep": ["True"],
            "RoomService": [0.0],
            "FoodCourt": [0.0],
            "ShoppingMall": [0.0],
            "Spa": [],
            "VRDeck": [],
        },
        # Unknown passenger with all fields present and no expenses
        {
            "CryoSleep": [],
            "RoomService": [0.0],
            "FoodCourt": [0.0],
            "ShoppingMall": [0.0],
            "Spa": [0.0],
            "VRDeck": [0.0],
        },
        # Unknown passenger with all fields present and some expenses
        {
            "CryoSleep": [],
            "RoomService": [10.5],
            "FoodCourt": [0.0],
            "ShoppingMall": [0.0],
            "Spa": [26.0],
            "VRDeck": [0.0],
        },
        # Unknown passenger with some fields missing and no expenses
        {
            "CryoSleep": [],
            "RoomService": [0.0],
            "FoodCourt": [],
            "ShoppingMall": [0.0],
            "Spa": [0.0],
            "VRDeck": [],
        },
        # Unknown passenger with some fields missing and an expense
        {
            "CryoSleep": [],
            "RoomService": [],
            "FoodCourt": [],
            "ShoppingMall": [],
            "Spa": [],
            "VRDeck": [6.0],
        },
    ]
    # mode -> asleep

    expected_dataset = wrap_output_dataset(
        [
            # Awake passenger with all fields present
            0,
            # Awake passenger with all fields present but no expenses
            0,
            # Awake passenger with some fields missing
            0,
            # Sleeping passengers with all fields present
            1,
            1,
            1,
            # Sleeping passenger with some fields missing
            1,
            # Unknown passenger with all fields present and no expenses
            1,  # mode
            # Unknown passenger with all fields present and some expenses
            0,
            # Unknown passenger with some fields missing and no expenses
            1,  # mode
            # Unknown passenger with some fields missing and an expense
            0,
        ],
        key="cryo_sleep",
        dtype=np.int64,
    )

    preprocessing_fn = preprocessing_fn_from_hamilton_modules(
        [boolean_module], ["cryo_sleep"]
    )

    input_feature_specs = {
        "CryoSleep": tf.io.VarLenFeature(dtype=tf.string),
        "RoomService": tf.io.VarLenFeature(dtype=tf.float32),
        "FoodCourt": tf.io.VarLenFeature(dtype=tf.float32),
        "Spa": tf.io.VarLenFeature(dtype=tf.float32),
        "VRDeck": tf.io.VarLenFeature(dtype=tf.float32),
        "ShoppingMall": tf.io.VarLenFeature(dtype=tf.float32),
    }

    (output_dataset,) = run_tft(
        preprocessing_fn,
        input_feature_specs,
        cast(List[Dict[str, List[Any]]], input_dataset),
    )
    assert output_dataset == expected_dataset


@pytest.mark.parametrize(
    "input_values,expected_values",
    [
        (["True", "False", "True", None, None], [1, 0, 1, 1, 1]),
        (["True", "False", "False", None, None], [1, 0, 0, 0, 0]),
    ],
)
def test_vip(
    input_values: List[Optional[str]], expected_values: List[int]
) -> None:
    input_dataset = wrap_dataset(input_values, key="VIP")
    expected_dataset = wrap_output_dataset(
        expected_values, key="vip", dtype=np.int64
    )

    preprocessing_fn = preprocessing_fn_from_hamilton_modules(
        [boolean_module], ["vip"]
    )

    input_feature_specs = {
        "VIP": tf.io.VarLenFeature(dtype=tf.string),
    }

    (output_dataset,) = run_tft(
        preprocessing_fn,
        input_feature_specs,
        input_dataset,
    )
    assert output_dataset == expected_dataset
