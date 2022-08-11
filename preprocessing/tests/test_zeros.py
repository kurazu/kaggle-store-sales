import string

import pandas as pd

from preprocessing.zeros import count_neighbouring_zeros


def test_count_neighbouring_zeros() -> None:
    inputs = [2, 1, 0, 0, 1, 0, 0, 0, 0, 3, 5, 0, 0, 0, 8, 0, 9]
    expected = [0, 0, 2, 2, 0, 4, 4, 4, 4, 0, 0, 3, 3, 3, 0, 1, 0]
    index = list(string.ascii_lowercase[: len(inputs)])
    input_series = pd.Series(inputs, index=index)
    expected_output = pd.Series(expected, index=index)
    actual_output = count_neighbouring_zeros(input_series)
    assert actual_output.equals(expected_output)
