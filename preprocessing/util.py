from types import ModuleType
from typing import Callable, Dict, List, Union

import tensorflow as tf
from hamilton import base, driver

TensorType = Union[tf.Tensor, tf.SparseTensor]


def preprocessing_fn_from_hamilton_modules(
    modules: List[ModuleType], output_columns: List[str]
) -> Callable[[Dict[str, TensorType]], Dict[str, TensorType]]:
    """
    Construct TFT preprocessing function out of python module using
    Hamilton to resolve dependencies between features.
    """

    def preprocessing_fn(
        inputs: Dict[str, TensorType]
    ) -> Dict[str, TensorType]:
        """Synthetic preprocessing function."""
        adapter = base.SimplePythonGraphAdapter(base.DictResult())
        dr = driver.Driver(inputs, *modules, adapter=adapter)
        outputs: Dict[str, TensorType] = dr.execute(output_columns)
        return outputs

    return preprocessing_fn
