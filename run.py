import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import core, data

OUT = Path("reports")


def main():
    OUT.mkdir(exist_ok=True)
    df = data.generate_daily_sales()
    results = core.backtest(df)
    table = core.summarize(results)
    best = table.index[0]

    train, test = df.iloc[:-56], df.iloc[-56:]
    preds = {
        "seasonal_naive": core.seasonal_naive(train, len(test)),
        "holt_winters": core.holt_winters(train, len(test)),
        "ridge_calendar": core.regression_forecast(train, test),
    }
    plt.figure(figsize=(11, 4))
    plt.plot(train.index[-90:], train["units"].iloc[-90:], color="lightgrey", label="history")
    plt.plot(test.index, test["units"], color="black", label="actual")
    styles = {"seasonal_naive": ":", "holt_winters": "--", "ridge_calendar": "-"}
    for name, p in preds.items():
        plt.plot(test.index, p, styles[name], color="grey" if name != best else "dimgrey", label=name)
    plt.legend()
    plt.title("Last 56 days: forecasts vs actual")
    plt.tight_layout()
    plt.savefig(OUT / "forecast.png", dpi=130)

    table.round(4).to_csv(OUT / "backtest_summary.csv")
    results.to_csv(OUT / "backtest_folds.csv", index=False)
    summary = {
        "days": int(len(df)),
        "folds": int(results["fold"].nunique()),
        "best_method": best,
        "mae": {k: round(float(v), 2) for k, v in table["mae"].items()},
        "smape": {k: round(float(v), 4) for k, v in table["smape"].items()},
        "mae_improvement_over_seasonal_naive": round(
            float(1 - table.loc[best, "mae"] / table.loc["seasonal_naive", "mae"]), 3
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
