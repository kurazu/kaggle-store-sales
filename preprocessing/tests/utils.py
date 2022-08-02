import logging
import tempfile
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    TypeVar,
    Union,
)

import numpy as np
import numpy.typing as npt
import tensorflow as tf
import tensorflow_transform.beam as tft_beam
from returns.curry import partial
from tensorflow_transform.tf_metadata import dataset_metadata, schema_utils

logger = logging.getLogger(__name__)

TensorType = Union[tf.Tensor, tf.SparseTensor]
FeatureSpecType = Union[tf.io.FixedLenFeature, tf.io.VarLenFeature]
DatasetType = List[Dict[str, List[Any]]]

T = TypeVar("T")


def wrap_value(item: Optional[T]) -> List[T]:
    if item is None:
        return []
    else:
        return [item]


def wrap_dataset(
    items: Iterable[Optional[T]], *, key: str
) -> List[Dict[str, List[T]]]:
    return [{key: wrap_value(item)} for item in items]


def wrap_output_dataset(
    items: Iterable[Any], *, key: str, dtype: npt.DTypeLike
) -> List[Dict[str, npt.NDArray[Any]]]:
    return [{key: np.array([item], dtype=dtype)} for item in items]


def wrap_multi_column_dataset(
    columns: Dict[str, Iterable[Optional[Any]]]
) -> List[Dict[str, List[Any]]]:
    return [
        {key: wrap_value(value) for key, value in input_dict.items()}
        for input_dict in map(
            dict, map(partial(zip, columns), zip(*columns.values()))
        )
    ]


def run_tft(
    preprocessing_fn: Callable[[Dict[str, TensorType]], Dict[str, TensorType]],
    input_feature_specs: Dict[str, FeatureSpecType],
    train_dataset: DatasetType,
    *test_datasets: DatasetType,
    test_label_key_to_pop: Optional[str] = None
) -> Iterable[List[Dict[str, npt.NDArray[Any]]]]:
    test_feature_specs = input_feature_specs.copy()
    if test_label_key_to_pop is not None:
        test_feature_specs.pop(test_label_key_to_pop)
    input_metadata = dataset_metadata.DatasetMetadata(
        schema_utils.schema_from_feature_spec(input_feature_specs)
    )
    test_metadata = dataset_metadata.DatasetMetadata(
        schema_utils.schema_from_feature_spec(test_feature_specs)
    )

    with tempfile.TemporaryDirectory() as tempdir, tft_beam.Context(
        temp_dir=tempdir
    ):
        transformed_train_dataset, transform_fn = (
            train_dataset,
            input_metadata,
        ) | tft_beam.AnalyzeAndTransformDataset(preprocessing_fn)

        transformed_train_data, _output_metadata = transformed_train_dataset

        yield transformed_train_data

        for test_dataset in test_datasets:
            transformed_test_data, transform_fn = (
                (test_dataset, test_metadata),
                transform_fn,
            ) | tft_beam.TransformDataset()
            yield transformed_test_data
