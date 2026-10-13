import numpy as np
import pandas as pd


def generate_daily_sales(days=3 * 365, seed=17):
    """Daily units for one store: trend, weekly and yearly seasonality, promotions, holidays."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=days, freq="D")
    t = np.arange(days)
    trend = 200 + 0.08 * t
    weekly = np.array([0.92, 0.95, 0.97, 1.0, 1.12, 1.25, 1.05])[idx.dayofweek]
    yearly = 1 + 0.18 * np.sin(2 * np.pi * (t - 80) / 365.25)
    promo = (rng.random(days) < 0.08).astype(int)
    holiday = ((idx.month == 12) & (idx.day >= 20)) | ((idx.month == 1) & (idx.day <= 3))
    level = trend * weekly * yearly * (1 + 0.3 * promo) * np.where(holiday, 1.35, 1.0)
    units = rng.poisson(level * rng.lognormal(0, 0.06, days))
    return pd.DataFrame({"date": idx, "units": units, "promo": promo, "holiday": holiday.astype(int)}).set_index("date")
