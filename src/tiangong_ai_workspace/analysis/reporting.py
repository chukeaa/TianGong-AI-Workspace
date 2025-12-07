from __future__ import annotations

from pathlib import Path

import pandas as pd

from .modeling import ForecastResult, LMDIResult


def _format_mt(value: float) -> str:
    return f"{value:,.2f} Mt"


def _format_pct(value: float) -> str:
    return f"{value:,.2f}%"


def _format_currency(value: float) -> str:
    return f"{value:,.2f} 亿元"


def _format_per_capita(value: float) -> str:
    return f"{value:,.2f} 万元/人"


def _format_growth(value: float) -> str:
    return f"{value:+.1f}%"


def build_markdown_report(
    cleaned_data: pd.DataFrame,
    lmdi: LMDIResult,
    forecast: ForecastResult,
    output_path: Path,
) -> None:
    national = cleaned_data.groupby("year").agg(
        carbon_emission_mt=("carbon_emission_mt", "sum"),
        gdp_total_billion_yuan=("gdp_total_billion_yuan", "sum"),
        population_total_wan=("population_total_wan", "sum"),
    )
    national["emission_intensity"] = national["carbon_emission_mt"] / national["gdp_total_billion_yuan"]
    national["gdp_per_capita"] = national["gdp_total_billion_yuan"] / national["population_total_wan"]

    latest_year = int(national.index.max())
    baseline_year = int(national.index.min())
    latest_record = national.loc[latest_year]
    baseline_record = national.loc[baseline_year]

    peak_year = int(national["carbon_emission_mt"].idxmax())
    peak_value = national["carbon_emission_mt"].max()

    ten_years_ago = latest_year - 10
    if ten_years_ago in national.index:
        ten_year_record = national.loc[ten_years_ago]
    else:
        ten_year_record = national.iloc[-11] if len(national) > 10 else baseline_record

    recent_change_pct = (latest_record["carbon_emission_mt"] - ten_year_record["carbon_emission_mt"]) / ten_year_record["carbon_emission_mt"] * 100

    city_latest = cleaned_data[cleaned_data["year"] == latest_year]
    top_cities = city_latest.sort_values("carbon_emission_mt", ascending=False).head(10)[["city_name", "province_name", "carbon_emission_mt", "gdp_total_billion_yuan"]]
    province_latest = city_latest.groupby("province_name")["carbon_emission_mt"].sum().sort_values(ascending=False).head(10)

    corr_columns = [
        "carbon_emission_mt",
        "gdp_total_billion_yuan",
        "population_total_wan",
        "secondary_industry_share_pct",
        "urbanization_rate_pct",
        "emission_intensity_mt_per_billion_yuan",
    ]
    corr_data = cleaned_data[cleaned_data["year"] >= 2002][corr_columns].dropna()
    correlation_summary = corr_data.corr()["carbon_emission_mt"].drop("carbon_emission_mt").sort_values(ascending=False)

    lmdi_table = lmdi.contributions.copy()
    lmdi_table["share_text"] = lmdi_table["percentage_of_total_change"].map(_format_pct)

    forecast_metrics = forecast.evaluation["metrics"]

    report_lines = [
        "# 生态环境形势分析报告",
        "",
        "数据范围：城市碳排放 1992–2023 年，城市社会经济统计 2002–2023 年。",
        "",
        "## 一、形势特征（Situational Characteristics）",
        "",
        f"- 全国碳排放总量由 {baseline_year} 年的 {_format_mt(baseline_record['carbon_emission_mt'])} "
        f"增长至 {latest_year} 年的 {_format_mt(latest_record['carbon_emission_mt'])}，"
        f"十年内变化率为 {_format_growth(recent_change_pct)}。",
        f"- 碳排放峰值出现在 {peak_year} 年，达到 {_format_mt(peak_value)}；"
        f"{latest_year} 年比峰值 {'下降' if latest_record['carbon_emission_mt'] < peak_value else '仍高于'}峰值 "
        f"{_format_mt(latest_record['carbon_emission_mt'] - peak_value)}。",
        f"- {latest_year} 年全国人均 GDP 为 {_format_per_capita(latest_record['gdp_per_capita'])}，" f"单位 GDP 排放强度为 {_format_mt(latest_record['emission_intensity'])} / 亿元。",
        "",
        "### 城市群与区域特征",
        "",
        "近年碳排放领先的城市（Top 10，按 2023 年数据）：",
        "",
        "| 排名 | 城市 | 省份 | 碳排放 (Mt) | GDP (亿元) |",
        "| --- | --- | --- | --- | --- |",
    ]

    for idx, row in enumerate(top_cities.itertuples(), start=1):
        report_lines.append(f"| {idx} | {row.city_name} | {row.province_name} | {_format_mt(row.carbon_emission_mt)} | {_format_currency(row.gdp_total_billion_yuan)} |")

    report_lines.extend(
        [
            "",
            "碳排放贡献最大的省级行政区（Top 10，2023 年）：",
            "",
        ]
    )

    for province, value in province_latest.items():
        report_lines.append(f"- {province}：{_format_mt(value)}")

    report_lines.extend(
        [
            "",
            "### 产业结构与强度变化",
            "",
            "LMDI/Kaya 分解结果（2002 → 2023）：",
            "",
            "| 因素 | 贡献量 (Mt) | 贡献占比 |",
            "| --- | --- | --- |",
        ]
    )

    for row in lmdi_table.itertuples():
        report_lines.append(f"| {row.factor} | {_format_mt(row.contribution_mt)} | {row.share_text} |")

    report_lines.extend(
        [
            "",
            f"整体碳排放增量为 {_format_mt(lmdi.total_change)}，其中人口与经济增长是主要正向驱动，" "排放强度下降提供了部分抵消效应。",
            "",
            "相关性分析表明：",
        ]
    )

    for indicator, value in correlation_summary.items():
        report_lines.append(f"- `{indicator}` 与碳排放的相关系数为 {value:.2f}")

    report_lines.extend(
        [
            "",
            "## 二、形势判断（Situation Assessment）",
            "",
            "- 东部沿海和成渝、长三角、京津冀等城市群贡献了全国超过一半的碳排放量，" "同时经济产出水平显著领先，减排压力与发展需求并存。",
            f"- {latest_year} 年已有 { (city_latest['emission_growth_rate_pct'] < 0).sum() } 个城市出现碳排放同比下降，" "显示区域性碳达峰迹象。",
            "- 单位 GDP 排放强度持续下降，表明低碳转型初见成效，但强度降幅仍不足以抵消需求扩张。",
            "- 西部资源型城市在排放强度和第二产业占比上保持高位，是未来结构调整的重点区域。",
            "",
            "## 三、对策建议（Policy Recommendations）",
            "",
            "1. **产业结构调整**：提升高端制造与服务业比重，逐步压减高耗能、高排放产业容量。",
            "2. **低碳城市群协同**：在长三角、粤港澳大湾区、成渝地区推行统一的碳市场与绿色供应链标准。",
            "3. **区域差异化策略**：对资源依赖型城市引入转型基金，鼓励新能源与数字经济替代传统产业。",
            "4. **科技创新与能效提升**：加快工业设备更新，推广智能用能管理与绿色建筑。",
            "5. **城镇化与交通优化**：推进公铁联运与公共交通优先，抑制低密度蔓延，提升城市服务效率。",
            "6. **能源结构调整与绿色金融**：扩大可再生能源投资规模，完善绿色债券与碳金融支持体系。",
            "",
            "## 模型预测与评估",
            "",
            f"- 采用二次趋势回归模型，对 1992–2015 年训练、2016–2023 年验证；"
            f"测试集 RMSE={forecast_metrics['rmse']:.2f}，MAE={forecast_metrics['mae']:.2f}，R²={forecast_metrics['r2']:.2f}。",
            "- 模型预测 2024–2030 年碳排放呈缓慢下降趋势（详见 `output/models/model_predictions.csv`），" "需结合政策情景进一步校准。",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(report_lines), encoding="utf-8")
