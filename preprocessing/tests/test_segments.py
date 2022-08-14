from typing import List

import pandas as pd
import pytest

from preprocessing.segments import mark_date_segments


@pytest.fixture
def dates() -> pd.Series:
    return pd.to_datetime(
        pd.Series(
            [
                "2022-01-01",
                "2022-01-02",
                "2022-01-04",
                "2022-01-05",
                "2022-01-08",
                "2022-01-09",
                "2022-01-13",
                "2022-01-20",
            ]
        )
    )


@pytest.mark.parametrize(
    "max_skip,expected",
    [
        (1, [0, 0, 1, 1, 2, 2, 3, 4]),
        (2, [0, 0, 0, 0, 1, 1, 2, 3]),
        (3, [0, 0, 0, 0, 0, 0, 1, 2]),
        (4, [0, 0, 0, 0, 0, 0, 0, 1]),
        (5, [0, 0, 0, 0, 0, 0, 0, 1]),
        (6, [0, 0, 0, 0, 0, 0, 0, 1]),
        (7, [0, 0, 0, 0, 0, 0, 0, 0]),
        (8, [0, 0, 0, 0, 0, 0, 0, 0]),
        (100, [0, 0, 0, 0, 0, 0, 0, 0]),
    ],
)
def test_mark_date_segments(
    dates: pd.Series, max_skip: int, expected: List[int]
) -> None:
    actual = mark_date_segments(dates, max_skip=max_skip)
    expected_series = pd.Series(expected, index=dates.index)
    assert actual.equals(expected_series)
