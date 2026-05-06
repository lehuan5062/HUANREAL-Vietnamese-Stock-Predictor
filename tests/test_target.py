"""Verify the T+2 target alignment is correct for both entry modes."""
import numpy as np
import pandas as pd

from stockpredict.model.target import forward_return


def _frame(closes, opens=None):
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="B")
    df = pd.DataFrame({"close": closes}, index=idx)
    if opens is not None:
        df["open"] = opens
    return df


def test_close_to_close_two_day_return():
    # close = [100, 101, 102, 105, 110]
    # target[T] = close[T+2]/close[T] - 1
    df = _frame([100.0, 101.0, 102.0, 105.0, 110.0])
    y = forward_return(df, entry="close", exit_offset_days=2)
    # T=0: 102/100 - 1 = 0.02
    # T=1: 105/101 - 1
    # T=2: 110/102 - 1
    # T=3, 4: NaN
    assert np.isclose(y.iloc[0], 0.02)
    assert np.isclose(y.iloc[1], 105 / 101 - 1)
    assert np.isclose(y.iloc[2], 110 / 102 - 1)
    assert np.isnan(y.iloc[3])
    assert np.isnan(y.iloc[4])


def test_next_open_to_close_two_day_return():
    df = _frame(
        [100.0, 101.0, 102.0, 105.0, 110.0],
        opens=[99.5, 100.5, 101.0, 104.0, 109.0],
    )
    y = forward_return(df, entry="next_open", exit_offset_days=2)
    # target[T] = close[T+2]/open[T+1] - 1
    # T=0: 102 / 100.5 - 1
    # T=1: 105 / 101 - 1
    # T=2: 110 / 104 - 1
    assert np.isclose(y.iloc[0], 102 / 100.5 - 1)
    assert np.isclose(y.iloc[1], 105 / 101 - 1)
    assert np.isclose(y.iloc[2], 110 / 104 - 1)
    assert np.isnan(y.iloc[3])
