from typing import Iterable, Sequence

import pandas as pd


def get_segments(
    dates: Sequence[pd.Timestamp], max_skip: int
) -> Iterable[int]:
    segment_id = 0
    try:
        iterator = iter(dates)
    except StopIteration:
        return  # no items at all
    yield segment_id

    prev = next(iterator)
    for current in iterator:
        if (current - prev).days > max_skip:
            segment_id += 1
        yield segment_id
        prev = current


def mark_date_segments(dates: pd.Series, max_skip: int) -> pd.Series:
    return pd.Series(get_segments(dates, max_skip), index=dates.index)
