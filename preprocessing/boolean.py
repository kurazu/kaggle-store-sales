"""
Preprocessing of boolean features.
"""
import tensorflow as tf
import tensorflow_transform as tft

from .common import mode


def _parse_boolean(tensor: tf.Tensor) -> tf.Tensor:
    return tf.where(
        tensor == "True",
        tf.constant(1, dtype=tf.int64),
        tf.constant(0, dtype=tf.int64),
    )


def _boolean(boolean_feature: tf.SparseTensor) -> tf.Tensor:
    """
    Parse boolean feature.
    Input format: `True`, `False` or missing.

    For missing entries, the mode (most frequent option) will be backfilled.
    """
    most_common_value = mode(boolean_feature)
    dense_value = tft.sparse_tensor_to_dense_with_shape(
        boolean_feature, shape=[None, 1], default_value=most_common_value
    )
    return _parse_boolean(dense_value)


def cryo_sleep(
    CryoSleep: tf.SparseTensor,
    FoodCourt: tf.SparseTensor,
    RoomService: tf.SparseTensor,
    ShoppingMall: tf.SparseTensor,
    Spa: tf.SparseTensor,
    VRDeck: tf.SparseTensor,
) -> tf.Tensor:
    """
    Compute whether passenger went for cryogenic sleep during the voyage.

    For missing entries - if he had any expenses, we assume he was not cryo sleeping.
    """
    # check expenses - asssume missing as no expenses
    food_court = tft.sparse_tensor_to_dense_with_shape(
        FoodCourt, shape=[None, 1], default_value=0.0
    )
    room_service = tft.sparse_tensor_to_dense_with_shape(
        RoomService, shape=[None, 1], default_value=0.0
    )
    shopping_mall = tft.sparse_tensor_to_dense_with_shape(
        ShoppingMall, shape=[None, 1], default_value=0.0
    )
    spa = tft.sparse_tensor_to_dense_with_shape(
        Spa, shape=[None, 1], default_value=0.0
    )
    vr_deck = tft.sparse_tensor_to_dense_with_shape(
        VRDeck, shape=[None, 1], default_value=0.0
    )
    # sum up expenses (produces 1D tensor)
    expense_sum = tf.reduce_sum(
        tf.concat(
            [food_court, room_service, shopping_mall, spa, vr_deck], axis=-1
        ),
        axis=-1,
    )
    most_common_value = mode(CryoSleep)
    awake_value = tf.constant("False", dtype=tf.string)
    missing_value = tf.constant("?", dtype=tf.string)
    dense_cryo_sleep = tft.sparse_tensor_to_dense_with_shape(
        CryoSleep, shape=[None, 1], default_value=missing_value
    )
    # Let's operate on 1D values for simplicity
    flat_dense_cryo_sleep = tf.squeeze(dense_cryo_sleep, axis=-1)
    # if value is unknown but there were any expenses - set to awake
    flat_dense_cryo_sleep = tf.where(
        (flat_dense_cryo_sleep == missing_value) & (expense_sum > 0.0),
        awake_value,
        flat_dense_cryo_sleep,
    )
    # if value is still unknown - set to mode
    flat_dense_cryo_sleep = tf.where(
        flat_dense_cryo_sleep == missing_value,
        most_common_value,
        flat_dense_cryo_sleep,
    )
    # parse to boolean
    flat_boolean_cryo_sleep = _parse_boolean(flat_dense_cryo_sleep)
    # Return back to realm of 2D
    return tf.expand_dims(flat_boolean_cryo_sleep, axis=-1)


def vip(VIP: tf.SparseTensor) -> tf.Tensor:
    """
    Computer whether passenger had VIP status.
    """
    return _boolean(VIP)

