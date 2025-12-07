from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .utils import configure_matplotlib


def plot_national_trend(data: pd.DataFrame, output_path: Path) -> None:
    yearly = data.groupby("year")["carbon_emission_mt"].sum().reset_index()

    plt.figure(figsize=(10, 5))
    plt.plot(yearly["year"], yearly["carbon_emission_mt"], marker="o", linewidth=2, color="#1f77b4")
    plt.title("全国城市碳排放总量变化（1992-2023）")
    plt.xlabel("年份")
    plt.ylabel("碳排放量（Mt）")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_city_heatmap(data: pd.DataFrame, output_path: Path, top_n: int = 40) -> None:
    pivot = data.pivot_table(index="city_name", columns="year", values="carbon_emission_mt", aggfunc="mean").fillna(0)
    top_cities = pivot.mean(axis=1).sort_values(ascending=False).head(top_n).index.tolist()
    heatmap_data = pivot.loc[top_cities]

    plt.figure(figsize=(12, 10))
    sns.heatmap(heatmap_data, cmap="Blues", linewidths=0.1, linecolor="white")
    plt.title(f"重点城市碳排放热图（前 {top_n} 名）")
    plt.xlabel("年份")
    plt.ylabel("城市")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_spatial_distribution(data: pd.DataFrame, output_path: Path, year: int = 2023) -> None:
    subset = data[data["year"] == year]
    province_totals = subset.groupby("province_name")["carbon_emission_mt"].sum().sort_values(ascending=False)
    top_provinces = province_totals.head(20)

    plt.figure(figsize=(10, 8))
    top_provinces[::-1].plot(kind="barh", color="#2ca02c")
    plt.title(f"{year} 年各省（市、区）碳排放总量 Top 20")
    plt.xlabel("碳排放量（Mt）")
    plt.ylabel("省份")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_correlation_matrix(data: pd.DataFrame, output_path: Path, variables: Sequence[str]) -> None:
    subset = data[variables].dropna(how="any")
    corr = subset.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", linewidths=0.5, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("主要社会经济指标与碳排放相关性矩阵（2002-2023）")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_emission_intensity_trend(data: pd.DataFrame, output_path: Path) -> None:
    df = data.dropna(subset=["emission_intensity_mt_per_billion_yuan"])
    yearly = (
        df.groupby("year")
        .agg(
            total_emissions=("carbon_emission_mt", "sum"),
            total_gdp=("gdp_total_billion_yuan", "sum"),
        )
        .reset_index()
    )
    yearly["national_intensity"] = np.where(
        yearly["total_gdp"] > 0,
        yearly["total_emissions"] / yearly["total_gdp"],
        np.nan,
    )

    plt.figure(figsize=(10, 5))
    plt.plot(yearly["year"], yearly["national_intensity"], marker="o", color="#ff7f0e")
    plt.title("全国单位 GDP 碳排放强度变化（2002-2023）")
    plt.xlabel("年份")
    plt.ylabel("碳排放强度（Mt / 亿元）")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_regression_results(data: pd.DataFrame, output_path: Path, year: int = 2023) -> None:
    subset = data[(data["year"] == year)].dropna(subset=["carbon_emission_mt", "gdp_total_billion_yuan"])
    if subset.empty:
        return

    x = subset["gdp_total_billion_yuan"]
    y = subset["carbon_emission_mt"]
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ys = slope * xs + intercept

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color="#9467bd", alpha=0.6, label="城市数据点")
    plt.plot(xs, ys, color="#d62728", linewidth=2, label="线性拟合")
    plt.title(f"{year} 年碳排放与 GDP 关系")
    plt.xlabel("GDP（亿元）")
    plt.ylabel("碳排放量（Mt）")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_lmdi_contributions(contributions: pd.DataFrame, output_path: Path) -> None:
    df = contributions.copy()
    plt.figure(figsize=(8, 6))
    bars = plt.bar(df["factor"], df["contribution_mt"], color="#8c564b")
    plt.title("LMDI 分解贡献（2002-2023）")
    plt.ylabel("贡献量（Mt）")
    for bar, value in zip(bars, df["contribution_mt"]):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.1f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_model_predictions(predictions: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    train = predictions[predictions["dataset"] == "train_fit"]
    test = predictions[predictions["dataset"] == "test_evaluation"]
    forecast = predictions[predictions["dataset"] == "forecast"]

    if not train.empty:
        plt.plot(train["year"], train["carbon_emission_mt"], "o", color="#1f77b4", label="训练集实际")
        plt.plot(train["year"], train["prediction_mt"], "-", color="#1f77b4", alpha=0.6, label="训练拟合")
    if not test.empty:
        plt.plot(test["year"], test["carbon_emission_mt"], "o", color="#ff7f0e", label="测试集实际")
        plt.plot(test["year"], test["prediction_mt"], "-", color="#ff7f0e", alpha=0.6, label="测试预测")
    if not forecast.empty:
        plt.plot(forecast["year"], forecast["prediction_mt"], "--", color="#2ca02c", label="预测 2024-2030")
    plt.title("全国碳排放趋势预测")
    plt.xlabel("年份")
    plt.ylabel("碳排放量（Mt）")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def initialise_plot_style() -> None:
    """Configure seaborn/matplotlib aesthetics once per run."""

    chosen_font = configure_matplotlib()
    rc_overrides = {"font.family": "sans-serif"}
    if chosen_font:
        rc_overrides["font.sans-serif"] = [chosen_font]
    sns.set_theme(style="whitegrid", font_scale=1.0, rc=rc_overrides)
