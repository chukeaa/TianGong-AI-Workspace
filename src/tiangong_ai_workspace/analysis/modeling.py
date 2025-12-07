from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .utils import export_dataframe, export_json


@dataclass
class LMDIResult:
    contributions: pd.DataFrame
    total_change: float


@dataclass
class ForecastResult:
    predictions: pd.DataFrame
    evaluation: dict


def _log_mean(x: float, y: float) -> float:
    if x == y:
        return x
    if x > 0 and y > 0:
        return (x - y) / (np.log(x) - np.log(y))
    return 0.0


def compute_lmdi_contributions(
    data: pd.DataFrame,
    base_year: int = 2002,
    target_year: int = 2023,
) -> LMDIResult:
    """
    Perform a three-factor additive LMDI decomposition using population,
    affluence (GDP per capita), and emission intensity.
    """

    required_cols = ["year", "carbon_emission_mt", "population_total_wan", "gdp_total_wan_yuan", "gdp_total_billion_yuan"]
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for LMDI: {missing_cols}")

    summary = data.groupby("year").agg(
        carbon_emission_mt=("carbon_emission_mt", lambda x: x.sum(min_count=1)),
        population_total_wan=("population_total_wan", lambda x: x.sum(min_count=1)),
        gdp_total_wan_yuan=("gdp_total_wan_yuan", lambda x: x.sum(min_count=1)),
        gdp_total_billion_yuan=("gdp_total_billion_yuan", lambda x: x.sum(min_count=1)),
    )

    available_years = summary.dropna().index
    if len(available_years) == 0:
        raise ValueError("Insufficient data to compute LMDI contributions.")

    if base_year not in available_years:
        base_year = int(available_years.min())
    if target_year not in available_years:
        target_year = int(available_years.max())

    base = summary.loc[base_year]
    target = summary.loc[target_year]

    C0 = base["carbon_emission_mt"]
    Ct = target["carbon_emission_mt"]
    P0 = base["population_total_wan"]
    Pt = target["population_total_wan"]
    G0 = base["gdp_total_wan_yuan"]
    Gt = target["gdp_total_wan_yuan"]

    if min(C0, Ct, P0, Pt, G0, Gt) <= 0:
        raise ValueError("Non-positive values encountered in LMDI inputs.")

    A0 = G0 / P0
    At = Gt / Pt
    I0 = C0 / G0
    It = Ct / Gt

    L = _log_mean(Ct, C0)
    contributions = {
        "人口规模效应": L * np.log(Pt / P0),
        "经济发展效应": L * np.log(At / A0),
        "排放强度效应": L * np.log(It / I0),
    }

    result_df = pd.DataFrame(
        {
            "factor": list(contributions.keys()),
            "contribution_mt": list(contributions.values()),
        }
    )
    result_df["percentage_of_total_change"] = result_df["contribution_mt"] / (Ct - C0) * 100.0

    return LMDIResult(contributions=result_df, total_change=float(Ct - C0))


def run_trend_forecast(
    data: pd.DataFrame,
    forecast_years: Sequence[int] = tuple(range(2024, 2031)),
    train_span: tuple[int, int] = (1992, 2015),
) -> ForecastResult:
    """
    Train a quadratic polynomial trend model on national emissions and forecast future values.
    """

    yearly = data.groupby("year")["carbon_emission_mt"].sum().reset_index()
    train_mask = (yearly["year"] >= train_span[0]) & (yearly["year"] <= train_span[1])
    test_mask = yearly["year"] > train_span[1]

    train_years = yearly.loc[train_mask, "year"].to_numpy()
    train_values = yearly.loc[train_mask, "carbon_emission_mt"].to_numpy()
    test_years = yearly.loc[test_mask, "year"].to_numpy()
    test_values = yearly.loc[test_mask, "carbon_emission_mt"].to_numpy()

    if len(train_years) < 3:
        raise ValueError("Insufficient training data for quadratic regression.")

    coefficients = np.polyfit(train_years, train_values, deg=2)
    model = np.poly1d(coefficients)

    train_predictions = model(train_years)
    test_predictions = model(test_years)
    forecast_years_arr = np.array(list(forecast_years))
    forecast_values = model(forecast_years_arr)

    def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        residuals = y_true - y_pred
        rmse = float(np.sqrt(np.mean(residuals**2)))
        mae = float(np.mean(np.abs(residuals)))
        if len(y_true) > 1:
            r2 = float(1 - np.sum(residuals**2) / np.sum((y_true - y_true.mean()) ** 2))
        else:
            r2 = float("nan")
        return {"rmse": rmse, "mae": mae, "r2": r2}

    metrics = _metrics(test_values, test_predictions)

    predictions_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "year": train_years,
                    "carbon_emission_mt": train_values,
                    "prediction_mt": train_predictions,
                    "dataset": "train_fit",
                }
            ),
            pd.DataFrame(
                {
                    "year": test_years,
                    "carbon_emission_mt": test_values,
                    "prediction_mt": test_predictions,
                    "dataset": "test_evaluation",
                }
            ),
            pd.DataFrame(
                {
                    "year": forecast_years_arr,
                    "carbon_emission_mt": np.nan,
                    "prediction_mt": forecast_values,
                    "dataset": "forecast",
                }
            ),
        ]
    ).sort_values("year")

    evaluation = {
        "model": "Quadratic trend regression",
        "train_year_range": [int(train_years.min()), int(train_years.max())],
        "test_year_range": [int(test_years.min()), int(test_years.max())] if len(test_years) else [None, None],
        "forecast_years": list(map(int, forecast_years_arr)),
        "metrics": metrics,
        "coefficients": coefficients.tolist(),
    }

    return ForecastResult(predictions=predictions_df, evaluation=evaluation)


def persist_model_outputs(result: ForecastResult, output_dir: Path) -> None:
    export_dataframe(result.predictions, output_dir / "models" / "model_predictions.csv")
    export_json(result.evaluation, output_dir / "models" / "model_evaluation.json")
