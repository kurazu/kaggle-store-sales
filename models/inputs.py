import functools
from dataclasses import dataclass
from typing import Callable, Dict, List, TypeVar

import tensorflow as tf

ProductType = TypeVar("ProductType")


@dataclass
class input_producer:
    """
    Creates a dictionary of keras inputs by appying input producer callback
    on each element of the list of names.
    """

    feature_names: List[str]
    """List of input names."""

    def __call__(
        self, callback: Callable[[str], ProductType]
    ) -> Callable[[], Dict[str, ProductType]]:
        @functools.wraps(callback)
        def input_producer_wrapper() -> Dict[str, ProductType]:
            return {
                feature_name: callback(feature_name)
                for feature_name in self.feature_names
            }

        return input_producer_wrapper


@input_producer(
    [
        "scaled_age",
        "scaled_food_court",
        "scaled_room_service",
        "scaled_shopping_mall",
        "scaled_spa",
        "scaled_vr_deck",
        "scaled_total_expenses",
        "scaled_cabin_num",
        # "scaled_family_members_count",
        # "family_survival",
        # "cabin_survival",
    ]
)
def get_continous_inputs(feature_name: str) -> tf.keras.layers.Input:
    """
    Return contnous inputs (float32).
    """
    return tf.keras.layers.Input(
        shape=(1,), name=feature_name, dtype=tf.float32
    )


@input_producer(["is_adult", "cryo_sleep", "vip"])
def get_boolean_inputs(feature_name: str) -> tf.keras.layers.Input:
    """
    Return boolean inputs (int64).
    """
    return tf.keras.layers.Input(shape=(1,), name=feature_name, dtype=tf.int64)


@input_producer(
    [
        "cabin_deck_vocab",
        "cabin_side_vocab",
        "home_planet_vocab",
        "destination_vocab",
        "passenger_group_vocab",
        "last_name_vocab",
    ]
)
def get_categorical_inputs(feature_name: str) -> tf.keras.layers.Input:
    """
    Return categorical vocabulary inputs (int64).
    """
    return tf.keras.layers.Input(shape=(1,), name=feature_name, dtype=tf.int64)


@input_producer(
    [
        "bucketized_age",
        "bucketized_food_court",
        "bucketized_room_service",
        "bucketized_shopping_mall",
        "bucketized_spa",
        "bucketized_vr_deck",
        "bucketized_total_expenses",
        # "bucketized_cabin_num",
    ]
)
def get_bucketized_continous_inputs(
    feature_name: str,
) -> tf.keras.layers.Input:
    """
    Return bucketized vocabulary inputs (int64).
    """
    return tf.keras.layers.Input(shape=(1,), name=feature_name, dtype=tf.int64)
