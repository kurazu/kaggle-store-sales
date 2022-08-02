import inspect
from typing import Dict, Union

import tensorflow as tf

from preprocessing.callback import preprocessing_fn


def test_preprocessing_fn_signature() -> None:
    signature = inspect.signature(preprocessing_fn)
    assert signature == inspect.Signature(
        parameters=[
            inspect.Parameter(
                name="inputs",
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=inspect.Parameter.empty,
                annotation=Dict[str, Union[tf.Tensor, tf.SparseTensor]],
            )
        ],
        return_annotation=Dict[str, Union[tf.Tensor, tf.SparseTensor]],
    )
