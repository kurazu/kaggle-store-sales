import numpy as np
import tensorflow as tf

from preprocessing import age as age_module
from preprocessing import boolean as boolean_module
from preprocessing import expenses as expenses_module
from preprocessing.util import preprocessing_fn_from_hamilton_modules

from .utils import run_tft, wrap_multi_column_dataset, wrap_output_dataset


def test_food_court() -> None:
    input_dataset = wrap_multi_column_dataset(
        {
            "Age": [12, 13, 18, 20],
            "CryoSleep": ["False", "False", "True", "False"],
            "FoodCourt": [12.5, 15.75, 10.25, 9.95],
            "RoomService": [None, None, None, None],
            "ShoppingMall": [None, None, None, None],
            "Spa": [None, None, None, None],
            "VRDeck": [None, None, None, None],
        }
    )
    expected_dataset = wrap_output_dataset(
        [0.0, 15.75, 0.0, 9.95], key="food_court", dtype=np.float32
    )

    preprocessing_fn = preprocessing_fn_from_hamilton_modules(
        [age_module, boolean_module, expenses_module], ["food_court"]
    )

    input_feature_specs = {
        "Age": tf.io.VarLenFeature(dtype=tf.float32),
        "CryoSleep": tf.io.VarLenFeature(dtype=tf.string),
        "FoodCourt": tf.io.VarLenFeature(dtype=tf.float32),
        "RoomService": tf.io.VarLenFeature(dtype=tf.float32),
        "ShoppingMall": tf.io.VarLenFeature(dtype=tf.float32),
        "Spa": tf.io.VarLenFeature(dtype=tf.float32),
        "VRDeck": tf.io.VarLenFeature(dtype=tf.float32),
    }

    (output_dataset,) = run_tft(
        preprocessing_fn, input_feature_specs, input_dataset
    )
    assert output_dataset == expected_dataset


def test_total_expenses() -> None:
    input_dataset = wrap_multi_column_dataset(
        {
            "Age": [20, 21],
            "CryoSleep": ["False", "False"],
            "FoodCourt": [20, 25],
            "RoomService": [
                80,
                0,
            ],
            "ShoppingMall": [0, 30],
            "Spa": [0, 40, None],
            "VRDeck": [45, 0],
        }
    )
    expected_dataset = wrap_output_dataset(
        [20 + 80 + 0 + 0 + 45, 25 + 0 + 30 + 40 + 0],
        key="total_expenses",
        dtype=np.float32,
    )

    preprocessing_fn = preprocessing_fn_from_hamilton_modules(
        [age_module, boolean_module, expenses_module], ["total_expenses"]
    )

    input_feature_specs = {
        "Age": tf.io.VarLenFeature(dtype=tf.float32),
        "CryoSleep": tf.io.VarLenFeature(dtype=tf.string),
        "FoodCourt": tf.io.VarLenFeature(dtype=tf.float32),
        "RoomService": tf.io.VarLenFeature(dtype=tf.float32),
        "ShoppingMall": tf.io.VarLenFeature(dtype=tf.float32),
        "Spa": tf.io.VarLenFeature(dtype=tf.float32),
        "VRDeck": tf.io.VarLenFeature(dtype=tf.float32),
    }

    (output_dataset,) = run_tft(
        preprocessing_fn, input_feature_specs, input_dataset
    )
    assert output_dataset == expected_dataset
