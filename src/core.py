import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def mae(y, yhat):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def smape(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    denom = (np.abs(y) + np.abs(yhat)) / 2
    ratio = np.divide(np.abs(y - yhat), denom, out=np.zeros_like(denom), where=denom != 0)
    return float(np.mean(ratio))


def seasonal_naive(train, horizon, period=7):
    """Repeat the last observed season."""
    last = train["units"].iloc[-period:].to_numpy()
    reps = int(np.ceil(horizon / period))
    return np.tile(last, reps)[:horizon]


def holt_winters(train, horizon, period=7):
    model = ExponentialSmoothing(
        train["units"].asfreq("D").astype(float), trend="add", seasonal="mul", seasonal_periods=period,
        initialization_method="estimated",
    ).fit()
    return model.forecast(horizon).to_numpy()


def calendar_features(index, promo=None, holiday=None):
    t = np.arange(len(index))
    feats = {"trend": t}
    for k in (1, 2, 3):
        feats[f"year_sin{k}"] = np.sin(2 * np.pi * k * index.dayofyear / 365.25)
        feats[f"year_cos{k}"] = np.cos(2 * np.pi * k * index.dayofyear / 365.25)
    for d in range(1, 7):
        feats[f"dow{d}"] = (index.dayofweek == d).astype(int)
    out = pd.DataFrame(feats, index=index)
    if promo is not None:
        out["promo"] = promo
    if holiday is not None:
        out["holiday"] = holiday
    return out


def regression_forecast(train, future, alpha=1.0):
    """Ridge on log units with calendar terms; promo and holiday flags must be known ahead."""
    full_index = train.index.append(future.index)
    promo = np.r_[train["promo"].to_numpy(), future["promo"].to_numpy()]
    holiday = np.r_[train["holiday"].to_numpy(), future["holiday"].to_numpy()]
    x = calendar_features(full_index, promo, holiday)
    x_train, x_future = x.iloc[: len(train)], x.iloc[len(train):]
    mu, sd = x_train.mean(), x_train.std().replace(0, 1)
    model = Ridge(alpha=alpha).fit((x_train - mu) / sd, np.log1p(train["units"]))
    return np.expm1(model.predict((x_future - mu) / sd))


def rolling_origin_splits(n, initial, horizon, step):
    start = initial
    while start + horizon <= n:
        yield np.arange(0, start), np.arange(start, start + horizon)
        start += step


def backtest(df, horizon=28, initial=540, step=56):
    methods = {
        "seasonal_naive": lambda tr, fu: seasonal_naive(tr, len(fu)),
        "holt_winters": lambda tr, fu: holt_winters(tr, len(fu)),
        "ridge_calendar": lambda tr, fu: regression_forecast(tr, fu),
    }
    rows = []
    for fold, (tr_idx, te_idx) in enumerate(rolling_origin_splits(len(df), initial, horizon, step)):
        train, test = df.iloc[tr_idx], df.iloc[te_idx]
        for name, fn in methods.items():
            pred = fn(train, test)
            rows.append(
                {"fold": fold, "method": name, "mae": mae(test["units"], pred), "smape": smape(test["units"], pred)}
            )
    return pd.DataFrame(rows)


def summarize(results):
    return results.groupby("method")[["mae", "smape"]].mean().sort_values("mae")
