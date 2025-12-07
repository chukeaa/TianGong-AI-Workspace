from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .data_processing import DataProcessingResult, build_clean_dataset
from .modeling import ForecastResult, LMDIResult, compute_lmdi_contributions, persist_model_outputs, run_trend_forecast
from .reporting import build_markdown_report
from .utils import ensure_output_directories
from .visualization import (
    initialise_plot_style,
    plot_city_heatmap,
    plot_correlation_matrix,
    plot_emission_intensity_trend,
    plot_lmdi_contributions,
    plot_model_predictions,
    plot_national_trend,
    plot_regression_results,
    plot_spatial_distribution,
)


@dataclass
class AnalysisArtifacts:
    cleaned_data_path: Path
    matching_log_path: Path
    missing_report_path: Path
    figure_paths: Dict[str, Path]
    model_predictions_path: Path
    model_evaluation_path: Path
    report_path: Path
    unmatched_emissions: List[str]
    unmatched_stats: List[str]


def run_full_analysis(output_base: Path = Path("output")) -> AnalysisArtifacts:
    folders = ensure_output_directories(output_base)

    data_result: DataProcessingResult = build_clean_dataset(output_base)

    initialise_plot_style()

    cleaned = data_result.cleaned_data
    figures = {
        "national_trend": folders["figs"] / "national_trend.png",
        "city_heatmap": folders["figs"] / "city_heatmap.png",
        "spatial_distribution": folders["figs"] / "spatial_distribution.png",
        "correlation_matrix": folders["figs"] / "correlation_matrix.png",
        "emission_intensity_trend": folders["figs"] / "emission_intensity_trend.png",
        "lmdi_contribution": folders["figs"] / "LMDI_contribution.png",
        "regression_results": folders["figs"] / "regression_results.png",
        "model_prediction": folders["figs"] / "model_prediction.png",
    }

    plot_national_trend(cleaned, figures["national_trend"])
    plot_city_heatmap(cleaned, figures["city_heatmap"])
    plot_spatial_distribution(cleaned, figures["spatial_distribution"])
    correlation_variables = [
        "carbon_emission_mt",
        "gdp_total_billion_yuan",
        "population_total_wan",
        "secondary_industry_share_pct",
        "urbanization_rate_pct",
        "emission_intensity_mt_per_billion_yuan",
    ]
    plot_correlation_matrix(cleaned[cleaned["year"] >= 2002], figures["correlation_matrix"], correlation_variables)
    plot_emission_intensity_trend(cleaned, figures["emission_intensity_trend"])
    plot_regression_results(cleaned, figures["regression_results"])

    lmdi_result: LMDIResult = compute_lmdi_contributions(cleaned)
    plot_lmdi_contributions(lmdi_result.contributions, figures["lmdi_contribution"])

    forecast_result: ForecastResult = run_trend_forecast(cleaned)
    plot_model_predictions(forecast_result.predictions, figures["model_prediction"])
    persist_model_outputs(forecast_result, output_base)

    report_path = folders["reports"] / "report.md"
    build_markdown_report(cleaned, lmdi_result, forecast_result, report_path)

    return AnalysisArtifacts(
        cleaned_data_path=output_base / "data" / "cleaned_data.csv",
        matching_log_path=output_base / "data" / "city_matching_log.csv",
        missing_report_path=output_base / "data" / "missing_report.csv",
        figure_paths=figures,
        model_predictions_path=output_base / "models" / "model_predictions.csv",
        model_evaluation_path=output_base / "models" / "model_evaluation.json",
        report_path=report_path,
        unmatched_emissions=data_result.unmatched_emissions,
        unmatched_stats=data_result.unmatched_stats,
    )
