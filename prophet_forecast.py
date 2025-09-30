#!/usr/bin/env python3
"""Simple Prophet-based time series forecast script for data/data.csv."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
from prophet import Prophet


DATA_PATH = Path(__file__).resolve().parent / "data" / "data.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a Prophet model on the pedestrian count data and display the forecast plot."
        )
    )
    parser.add_argument(
        "--location",
        dest="location_code",
        help=(
            "Filter the dataset by a specific Location_code before training. "
            "If omitted, all locations are aggregated."
        ),
    )
    parser.add_argument(
        "--periods",
        type=int,
        default=48,
        help="Number of future periods (hours) to forecast. Defaults to 48.",
    )
    parser.add_argument(
        "--freq",
        default="H",
        help="Pandas offset alias for future period frequency. Defaults to hourly (H).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Optional path to save the forecast plot instead of displaying it. The file "
            "extension determines the format (e.g., .png)."
        ),
    )
    parser.add_argument(
        "--history-years",
        type=float,
        help=(
            "Limit the training history to the most recent N years (approximate). "
            "For example, pass 2 to use only the past two years."
        ),
    )
    return parser.parse_args()


def load_timeseries(
    csv_path: Path, location_code: str | None, history_years: float | None
) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # Normalize column names in case of BOM artefacts or unexpected casing.
    df.columns = [col.strip() for col in df.columns]

    required_columns = {"Date", "TotalCount"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            "Data file is missing required columns: " + ", ".join(sorted(missing_columns))
        )

    if location_code:
        location_col = next((c for c in df.columns if c.lower() == "location_code"), None)
        if location_col is None:
            raise ValueError("Location_code column is not present in the data file.")
        df = df[df[location_col] == location_code]
        if df.empty:
            raise ValueError(f"No rows found for Location_code '{location_code}'.")

    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    # Prophet requires tz-naive timestamps, so drop the timezone after normalising to UTC.
    if pd.api.types.is_datetime64tz_dtype(df["Date"]):
        df["Date"] = df["Date"].dt.tz_localize(None)
    df = df.dropna(subset=["Date", "TotalCount"])

    # Aggregate by timestamp to ensure unique indices and sort chronologically.
    series = (
        df.groupby("Date", as_index=False)["TotalCount"].sum().rename(
            columns={"Date": "ds", "TotalCount": "y"}
        )
    )
    series = series.sort_values("ds")

    if history_years is not None:
        if history_years <= 0:
            raise ValueError("--history-years must be positive when provided.")
        latest_timestamp = series["ds"].max()
        cutoff = latest_timestamp - pd.to_timedelta(history_years * 365, unit="D")
        series = series[series["ds"] >= cutoff]
        if series.empty:
            raise ValueError(
                "No data points remain after applying the history window; "
                "try a larger value for --history-years."
            )

    return series


def main() -> None:
    args = parse_args()
    series = load_timeseries(DATA_PATH, args.location_code, args.history_years)

    if series.empty:
        raise ValueError("The resulting time series is empty; nothing to fit.")

    model = Prophet(interval_width=0.9, daily_seasonality=True)
    model.fit(series)

    future = model.make_future_dataframe(periods=args.periods, freq=args.freq, include_history=True)
    forecast = model.predict(future)

    fig = model.plot(forecast)
    ax = fig.axes[0]
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Pedestrian count")
    handles, labels = ax.get_legend_handles_labels()
    label_map = {
        "yhat": "Forecast",
        "yhat_lower": "Lower bound",
        "yhat_upper": "Upper bound",
        "actual": "History",
    }

    future_mask = forecast["ds"] > series["ds"].max()
    if future_mask.any():
        forecast_begin = forecast.loc[future_mask, "ds"].min()
        forecast_end = forecast.loc[future_mask, "ds"].max()
        ax.axvspan(
            forecast_begin,
            forecast_end,
            facecolor="#ffe9a8",
            edgecolor="none",
            alpha=0.3,
            zorder=0,
        )
        handles.append(Patch(facecolor="#ffe9a8", edgecolor="none", alpha=0.3))
        labels.append("Forecast window")
        y_top = ax.get_ylim()[1] * 0.98
        ax.text(
            forecast_begin,
            y_top,
            "Forecast window",
            ha="left",
            va="top",
            fontsize=9,
            color="#7a5200",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )

    if labels:
        ax.legend(handles, [label_map.get(label, label) for label in labels], loc="best")
    plt.title(
        "Prophet Forecast\n"
        + (f"Location_code: {args.location_code}" if args.location_code else "All locations")
    )
    plt.tight_layout()

    if args.output:
        fig.savefig(args.output, bbox_inches="tight")
        print(f"Forecast plot saved to {args.output}")
    else:
        plt.show()

    # Display a quick preview of the forecasted values in the console.
    preview = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(args.periods)
    print(preview.to_string(index=False))


if __name__ == "__main__":
    main()
