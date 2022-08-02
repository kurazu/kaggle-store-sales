import numpy as np
import tensorflow as tf

from preprocessing import cabin as cabin_module
from preprocessing import label as label_module
from preprocessing import names as names_module
from preprocessing import survival as survival_module
from preprocessing.util import preprocessing_fn_from_hamilton_modules

from .utils import run_tft, wrap_multi_column_dataset, wrap_output_dataset


def test_family_survival() -> None:
    input_dataset = wrap_multi_column_dataset(
        {
            "Name": [
                "Violet Baudelaire",
                "Klaus Baudelaire",
                "Sunny Baudelaire",
                "Arthur Poe",
                "Lemony Snicket",
            ],
            "Transported": ["True", "True", "False", "True", "False"],
        }
    )
    expected_dataset = wrap_output_dataset(
        [2 / 3, 2 / 3, 2 / 3, 1 / 1, 0 / 1],
        key="family_survival",
        dtype=np.float32,
    )
    test_dataset = wrap_multi_column_dataset(
        {
            "Name": ["Beatrice Baudelaire", "Count Olaf"],
            # Notice how the test dataset does not contain label.
        }
    )
    expected_test_dataset = wrap_output_dataset(
        [2 / 3, 0], key="family_survival", dtype=np.float32
    )

    preprocessing_fn = preprocessing_fn_from_hamilton_modules(
        [names_module, label_module, survival_module], ["family_survival"]
    )

    input_feature_specs = {
        "Name": tf.io.VarLenFeature(dtype=tf.string),
        "Transported": tf.io.FixedLenFeature(shape=[1], dtype=tf.string),
    }

    (output_dataset, output_test_dataset) = run_tft(
        preprocessing_fn,
        input_feature_specs,
        input_dataset,
        test_dataset,
        test_label_key_to_pop="Transported",
    )
    assert output_dataset == expected_dataset
    assert output_test_dataset == expected_test_dataset


def test_cabin_survival() -> None:
    input_dataset = wrap_multi_column_dataset(
        {
            "Cabin": [
                "F/16/P",
                "F/16/P",
                None,
                "F/16/P",
                "G/15/S",
                "G/16/S",
                "G/16/S",
                None,
            ],
            "Transported": [
                "True",
                "False",
                "False",
                "True",
                "False",
                "True",
                "False",
                "True",
            ],
        }
    )
    expected_dataset = wrap_output_dataset(
        [2 / 3, 2 / 3, 0, 2 / 3, 0 / 1, 1 / 2, 1 / 2, 0],
        key="cabin_survival",
        dtype=np.float32,
    )
    test_dataset = wrap_multi_column_dataset(
        {
            "Cabin": [
                "F/16/P",
                "G/15/S",
                "G/16/S",
                "A/1/P",
                None,
            ],
            # Notice how the test dataset does not contain label.
        }
    )
    expected_test_dataset = wrap_output_dataset(
        [2 / 3, 0 / 1, 1 / 2, 0, 0], key="cabin_survival", dtype=np.float32
    )

    preprocessing_fn = preprocessing_fn_from_hamilton_modules(
        [cabin_module, label_module, survival_module],
        ["cabin_survival"],
    )

    input_feature_specs = {
        "Cabin": tf.io.VarLenFeature(dtype=tf.string),
        "Transported": tf.io.FixedLenFeature(shape=[1], dtype=tf.string),
    }

    (output_dataset, output_test_dataset) = run_tft(
        preprocessing_fn,
        input_feature_specs,
        input_dataset,
        test_dataset,
        test_label_key_to_pop="Transported",
    )
    assert output_dataset == expected_dataset
    assert output_test_dataset == expected_test_dataset
