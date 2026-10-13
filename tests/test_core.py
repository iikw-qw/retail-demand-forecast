import numpy as np
import pytest

from src import core, data


@pytest.fixture(scope="module")
def df():
    return data.generate_daily_sales(days=500, seed=3)


def test_metrics_on_known_values():
    assert core.mae([1, 2, 3], [2, 2, 5]) == pytest.approx(1.0)
    assert core.smape([100, 100], [100, 100]) == 0.0
    assert core.smape([0, 0], [0, 0]) == 0.0


def test_seasonal_naive_repeats_last_week(df):
    pred = core.seasonal_naive(df.iloc[:100], 14)
    last_week = df["units"].iloc[93:100].to_numpy()
    assert np.array_equal(pred[:7], last_week)
    assert np.array_equal(pred[7:], last_week)


def test_splits_never_leak_future(df):
    for train, test in core.rolling_origin_splits(len(df), 300, 28, 28):
        assert train.max() < test.min()
        assert len(test) == 28


def test_model_forecasts_have_right_length(df):
    train, test = df.iloc[:-28], df.iloc[-28:]
    assert len(core.holt_winters(train, 28)) == 28
    assert len(core.regression_forecast(train, test)) == 28


def test_calendar_model_beats_naive(df):
    res = core.backtest(df, horizon=28, initial=350, step=60)
    summary = core.summarize(res)
    assert summary.loc["ridge_calendar", "mae"] < summary.loc["seasonal_naive", "mae"]
