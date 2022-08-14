from typing import Iterable, Sequence

import pandas as pd


def count_consecutive_zeros(sequence: Sequence[int]) -> Iterable[int]:
    current = 0
    for i in sequence:
        if i == 0:
            current += 1
        else:
            current = 0
        yield current


def count_neighbouring_zeros(series: pd.Series) -> pd.Series:
    preceding_zeros_count = pd.Series(
        count_consecutive_zeros(series), index=series.index
    )
    following_zeros_count = pd.Series(
        count_consecutive_zeros(series[::-1]), index=series.index[::-1]
    )[::-1]
    sum_of_zeros = preceding_zeros_count + following_zeros_count
    return sum_of_zeros.where(
        (preceding_zeros_count == 0) | (following_zeros_count == 0),
        sum_of_zeros - 1,
    )
