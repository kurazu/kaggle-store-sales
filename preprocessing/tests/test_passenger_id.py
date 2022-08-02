import numpy as np
import tensorflow as tf

from preprocessing import passenger_id as passenger_id_module
from preprocessing.util import preprocessing_fn_from_hamilton_modules

from .utils import run_tft, wrap_dataset, wrap_output_dataset


def test_passenger_id() -> None:
    ids = ["0012_01", "1234_56", "0789_34", "3456_08", "0001_00", "0000_00"]
    input_dataset = wrap_dataset(map(int, ids), key="PassengerId")
    expected_dataset = wrap_output_dataset(
        ids, key="passenger_id", dtype=np.string_
    )

    preprocessing_fn = preprocessing_fn_from_hamilton_modules(
        [passenger_id_module], ["passenger_id"]
    )

    input_feature_specs = {
        "PassengerId": tf.io.FixedLenFeature(shape=[1], dtype=tf.int64)
    }

    (output_dataset,) = run_tft(
        preprocessing_fn, input_feature_specs, input_dataset
    )
    assert output_dataset == expected_dataset
