import csv
import os.path
from dataclasses import dataclass
from typing import Iterable, MutableMapping

import tensorflow as tf
from tensorflow.core.framework.tensor_pb2 import TensorProto
from tensorflow_serving.apis import prediction_log_pb2


@dataclass
class Prediction:
    passenger_id: str
    prediction: bool


def _extract_int(example: tf.train.Example, feature_name: str) -> int:
    value: int
    (value,) = example.features.feature[feature_name].int64_list.value
    return value


def _reconstruct_passenger_id(value: int) -> str:
    passenger_group = value // 100
    passenger_group_seq = value % 100
    return f"{passenger_group:04}_{passenger_group_seq:02}"


def read_predictions(
    predictions_dir: str, threshold: float
) -> Iterable[Prediction]:
    filenames = tf.io.gfile.glob(os.path.join(predictions_dir, "*.gz"))
    raw_dataset = tf.data.TFRecordDataset(filenames, compression_type="GZIP")
    for raw_record in raw_dataset:
        prediction = prediction_log_pb2.PredictionLog()
        prediction.ParseFromString(raw_record.numpy())
        raw_example = prediction.predict_log.request.inputs[
            "examples"
        ].string_val[0]
        example = tf.train.Example()
        example.ParseFromString(raw_example)

        passenger_id = _extract_int(example, "PassengerId")
        passenger_id_str = _reconstruct_passenger_id(passenger_id)

        outputs: MutableMapping[
            str, TensorProto
        ] = prediction.predict_log.response.outputs
        (probability,) = outputs["transported"].float_val
        classification = probability >= threshold

        yield Prediction(
            passenger_id=passenger_id_str, prediction=classification
        )


def export_predictions(
    predictions: Iterable[Prediction], target_dir: str
) -> None:
    target_path = os.path.join(target_dir, "inference.csv")
    with tf.io.gfile.GFile(target_path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=["PassengerId", "Transported"])
        writer.writeheader()
        writer.writerows(
            map(
                lambda row: {
                    "PassengerId": row.passenger_id,
                    "Transported": row.prediction,
                },
                predictions,
            )
        )
