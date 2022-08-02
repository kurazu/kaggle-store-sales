"""
Preprocessing of expense features.
"""
import tensorflow as tf
import tensorflow_transform as tft

from .features import ADULT_AGE, NUM_EXPENSE_BUCKETS

_NAN = float("nan")


def _expense(
    expense: tf.SparseTensor, *, age: tf.Tensor, cryo_sleep: tf.Tensor
) -> tf.Tensor:
    """
    Parse expense.

    Set 0 for minors.
    Set 0 for cryogenic sleep passengers.
    Replace missing values with mean.
    """
    mean = tft.mean(expense)
    dense_values = tft.sparse_tensor_to_dense_with_shape(
        expense, shape=[None, 1], default_value=_NAN
    )
    # Kids cannot buy stuff
    dense_values = tf.where(age >= ADULT_AGE, dense_values, 0.0)
    # People in cryogenic sleep cannot buy stuff
    dense_values = tf.where(cryo_sleep == 1, 0.0, dense_values)
    # Replace remaining missing values with mean
    return tf.where(tf.math.is_nan(dense_values), mean, dense_values)


def food_court(
    FoodCourt: tf.SparseTensor, age: tf.Tensor, cryo_sleep: tf.Tensor
) -> tf.Tensor:
    """
    Parse food court expenses.
    """
    return _expense(FoodCourt, age=age, cryo_sleep=cryo_sleep)


def scaled_food_court(food_court: tf.Tensor) -> tf.Tensor:  # pragma: no cover
    """
    Z-score scaled food court expenses.
    """
    return tft.scale_to_z_score(food_court)


def bucketized_food_court(
    food_court: tf.Tensor,
) -> tf.Tensor:  # pragma: no cover
    """
    Bucketized food court expenses.
    """
    return tft.bucketize(
        food_court,
        num_buckets=NUM_EXPENSE_BUCKETS,
        name="bucketized_food_court",
    )


def room_service(
    RoomService: tf.SparseTensor, age: tf.Tensor, cryo_sleep: tf.Tensor
) -> tf.Tensor:
    """
    Parse food court expenses.
    """
    return _expense(RoomService, age=age, cryo_sleep=cryo_sleep)


def scaled_room_service(
    room_service: tf.Tensor,
) -> tf.Tensor:  # pragma: no cover
    """
    Z-score scaled room service expenses.
    """
    return tft.scale_to_z_score(room_service)


def bucketized_room_service(
    room_service: tf.Tensor,
) -> tf.Tensor:  # pragma: no cover
    """
    Bucketized room service expenses.
    """
    return tft.bucketize(
        room_service,
        num_buckets=NUM_EXPENSE_BUCKETS,
        name="bucketized_room_service",
    )


def shopping_mall(
    ShoppingMall: tf.SparseTensor, age: tf.Tensor, cryo_sleep: tf.Tensor
) -> tf.Tensor:
    """
    Parse shopping mall expenses.
    """
    return _expense(ShoppingMall, age=age, cryo_sleep=cryo_sleep)


def scaled_shopping_mall(
    shopping_mall: tf.Tensor,
) -> tf.Tensor:  # pragma: no cover
    """
    Scaled shopping mall expenses.
    """
    return tft.scale_to_z_score(shopping_mall)


def bucketized_shopping_mall(
    shopping_mall: tf.Tensor,
) -> tf.Tensor:  # pragma: no cover
    """
    Bucketized shopping mall expenses.
    """
    return tft.bucketize(
        shopping_mall,
        num_buckets=NUM_EXPENSE_BUCKETS,
        name="bucketized_shopping_mall",
    )


def spa(
    Spa: tf.SparseTensor, age: tf.Tensor, cryo_sleep: tf.Tensor
) -> tf.Tensor:
    """
    Parse spa expenses.
    """
    return _expense(Spa, age=age, cryo_sleep=cryo_sleep)


def scaled_spa(spa: tf.Tensor) -> tf.Tensor:  # pragma: no cover
    """
    Scaled spa expenses.
    """
    return tft.scale_to_z_score(spa)


def bucketized_spa(spa: tf.Tensor) -> tf.Tensor:  # pragma: no cover
    """
    Bucketized spa expenses.
    """
    return tft.bucketize(
        spa, num_buckets=NUM_EXPENSE_BUCKETS, name="bucketized_spa"
    )


def vr_deck(
    VRDeck: tf.SparseTensor, age: tf.Tensor, cryo_sleep: tf.Tensor
) -> tf.Tensor:
    """
    Parse VR deck expenses.
    """
    return _expense(VRDeck, age=age, cryo_sleep=cryo_sleep)


def scaled_vr_deck(vr_deck: tf.Tensor) -> tf.Tensor:  # pragma: no cover
    """
    Scaled VR deck expenses.
    """
    return tft.scale_to_z_score(vr_deck)


def bucketized_vr_deck(vr_deck: tf.Tensor) -> tf.Tensor:  # pragma: no cover
    """
    Bucketized VR deck expenses.
    """
    return tft.bucketize(
        vr_deck, num_buckets=NUM_EXPENSE_BUCKETS, name="bucketized_vr_deck"
    )


def total_expenses(
    food_court: tf.Tensor,
    room_service: tf.Tensor,
    shopping_mall: tf.Tensor,
    spa: tf.Tensor,
    vr_deck: tf.Tensor,
) -> tf.Tensor:
    """
    Calculate total of all expenses.
    """
    stacked = tf.stack(
        [food_court, room_service, shopping_mall, spa, vr_deck],
        axis=-1,
        name="stacked_expenses",
    )
    return tf.reduce_sum(stacked, axis=-1, name="sum_expenses")


def scaled_total_expenses(
    total_expenses: tf.Tensor,
) -> tf.Tensor:  # pragma: no cover
    """
    Z-score scaled total expenses.
    """
    return tft.scale_to_z_score(total_expenses)


def bucketized_total_expenses(
    total_expenses: tf.Tensor,
) -> tf.Tensor:  # pragma: no cover
    """
    Bucketized total expenses.
    """
    return tft.bucketize(
        total_expenses,
        num_buckets=NUM_EXPENSE_BUCKETS,
        name="bucketized_expenses",
    )


