"""Tests for the null handling in the rolling features (janestreet.data_processor)."""

import polars as pl

from janestreet import utils
from janestreet.data_processor import DataProcessor


def test_rolling_min_periods_is_half_the_window_but_at_least_two():
    assert utils.rolling_min_periods(1000) == 500
    assert utils.rolling_min_periods(3) == 2


def test_rolling_stats_skip_structural_nulls_instead_of_nulling_the_window():
    # Two "days" of 10 rows where the first 2 rows of each day are null, like the real
    # warm-up nulls. With polars' default min_periods every window of 10 rows holds a null,
    # so the rolling columns would be entirely null.
    n = 10
    values = [None, None] + [float(i) for i in range(8)]
    df = pl.DataFrame({"symbol_id": [0] * 20, "x": values * 2})
    dp = DataProcessor("test_rolling_nulls", cols_features_corr=["x"], T=n)

    out = dp._get_window_average_std(df, ["x"], n=n)

    assert out["x_rolling_std_10"].null_count() < 20
    assert out["x_rolling_std_10"].drop_nulls().min() > 0
    # diff = x - rolling mean, so it is null exactly where x itself is null (once warm).
    warm = out.tail(10)
    assert warm["x_diff_rolling_avg_10"].null_count() == 2
