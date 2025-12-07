from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .utils import MatchingLog, export_dataframe, normalize_city_name

RAW_EMISSIONS_FILE = Path("src/input/city-level CO2 emissions from 1992 to 2023.csv")
RAW_STATS_FILE = Path("src/input/statistics_yearly.csv")
CITY_CODES_FILE = Path("src/input/city_codes.csv")

# indicator -> canonical column name
INDICATOR_MAP: Dict[str, str] = {
    "地区生产总值-全市": "gdp_total_wan_yuan",
    "地区生产总值当年价格-全市": "gdp_current_price_wan_yuan",
    "年末总人口-全市": "population_total_wan",
    "常住人口": "resident_population_wan",
    "非农业人口-全市": "non_agricultural_population_wan",
    "常住人口-常住人口城镇化率": "urbanization_rate_pct",
    "第一产业占地区生产总值的比重-全市": "primary_industry_share_pct",
    "第二产业占地区生产总值的比重-全市": "secondary_industry_share_pct",
    "第三产业占地区生产总值的比重-全市": "tertiary_industry_share_pct",
}


@dataclass
class DataProcessingResult:
    cleaned_data: pd.DataFrame
    matching_log: pd.DataFrame
    missing_report: pd.DataFrame
    unmatched_emissions: List[str]
    unmatched_stats: List[str]


def _load_city_codes(path: Path | str = CITY_CODES_FILE) -> pd.DataFrame:
    codes = pd.read_csv(path, dtype={"city_code": str})
    codes["normalized_name"] = codes["normalized_name"].astype(str)
    return codes


def _match_with_codes(df: pd.DataFrame, name_col: str, codes: pd.DataFrame, source: str) -> Tuple[pd.DataFrame, List[MatchingLog]]:
    working = df.copy()
    working[name_col] = working[name_col].astype(str)
    working["normalized_name"] = working[name_col].map(normalize_city_name)

    merged = working.merge(
        codes[["city_code", "city_display_name", "province_name", "normalized_name"]],
        on="normalized_name",
        how="left",
    )

    logs: List[MatchingLog] = []
    for _, row in merged.iterrows():
        matched = pd.notna(row["city_code"])
        logs.append(
            MatchingLog(
                city_code=row["city_code"] if matched else "",
                city_name=row["city_display_name"] if matched else "",
                source=source,
                original_label=row[name_col],
                matched=matched,
                note="" if matched else "未在城市代码库中匹配",
            )
        )
    return merged, logs


def _load_emissions_data(emissions_path: Path | str, codes: pd.DataFrame) -> Tuple[pd.DataFrame, List[MatchingLog]]:
    emissions_raw = pd.read_csv(emissions_path)
    emissions_raw["Year"] = emissions_raw["Year"].astype(int)
    emissions_raw["Carbon emission"] = pd.to_numeric(emissions_raw["Carbon emission"], errors="coerce")

    matched, logs = _match_with_codes(emissions_raw, "City", codes, "emissions")
    unmatched = matched[matched["city_code"].isna()]
    if not unmatched.empty:
        # Drop unmatched entries after logging
        matched = matched[matched["city_code"].notna()].copy()

    emissions = (
        matched[["city_code", "city_display_name", "province_name", "Year", "Carbon emission"]]
        .rename(
            columns={
                "city_display_name": "city_name",
                "Year": "year",
                "Carbon emission": "carbon_emission_mt",
            }
        )
        .dropna(subset=["carbon_emission_mt"])
    )
    emissions["year"] = emissions["year"].astype(int)
    emissions.sort_values(["city_code", "year"], inplace=True)
    return emissions, logs


def _load_statistics_data(stats_path: Path | str, codes: pd.DataFrame) -> Tuple[pd.DataFrame, List[MatchingLog]]:
    stats_raw = pd.read_csv(stats_path)
    stats_raw["indicator"] = stats_raw["indicator"].astype(str)
    stats_raw = stats_raw[stats_raw["indicator"].isin(INDICATOR_MAP.keys())].copy()

    matched, logs = _match_with_codes(stats_raw, "city", codes, "statistics")
    matched["year"] = pd.to_numeric(matched["year"], errors="coerce").astype("Int64")
    matched["value"] = pd.to_numeric(matched["value"], errors="coerce")

    def _adjust_units(row: pd.Series) -> float:
        indicator = row["indicator"]
        value = row["value"]
        if pd.isna(value):
            return value
        if indicator == "地区生产总值当年价格-全市":
            # 原单位为亿元，统一换算为万元以便与其它 GDP 指标合并。
            return value * 10000.0
        return value

    matched["value"] = matched.apply(_adjust_units, axis=1)
    matched = matched.dropna(subset=["city_code", "year"])

    # Keep the latest duplicate if any (most datasets don't have duplicates but guard anyway).
    matched.sort_values(["city_code", "year"], inplace=True)
    deduped = matched.drop_duplicates(subset=["city_code", "year", "indicator"], keep="last")

    pivot = (
        deduped.pivot_table(
            index=["city_code", "year"],
            columns="indicator",
            values="value",
        )
        .rename(columns=INDICATOR_MAP)
        .reset_index()
    )

    # Attach canonical city/province names
    pivot = pivot.merge(
        codes[["city_code", "city_display_name", "province_name"]],
        on="city_code",
        how="left",
    )
    pivot.rename(columns={"city_display_name": "city_name"}, inplace=True)

    return pivot, logs


def _build_matching_log(emission_logs: Iterable[MatchingLog], stats_logs: Iterable[MatchingLog]) -> pd.DataFrame:
    records = []
    for entry in emission_logs:
        records.append(
            {
                "city_code": entry.city_code,
                "city_name": entry.city_name or entry.original_label,
                "source": "emissions",
                "original_label": entry.original_label,
                "matched": entry.matched,
                "note": entry.note,
            }
        )
    for entry in stats_logs:
        records.append(
            {
                "city_code": entry.city_code,
                "city_name": entry.city_name or entry.original_label,
                "source": "statistics",
                "original_label": entry.original_label,
                "matched": entry.matched,
                "note": entry.note,
            }
        )
    log_df = pd.DataFrame(records)
    log_df.sort_values(["source", "city_name"], inplace=True)
    return log_df


def _merge_datasets(emissions: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    merged = emissions.merge(
        stats,
        on=["city_code", "year"],
        how="left",
        suffixes=("", "_stats"),
    )
    merged["city_name"] = merged["city_name"].fillna(merged["city_name_stats"])
    merged["province_name"] = merged["province_name"].fillna(merged["province_name_stats"])
    merged.drop(columns=["city_name_stats", "province_name_stats"], inplace=True, errors="ignore")
    return merged


def _derive_additional_features(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()

    if "gdp_total_wan_yuan" not in working.columns:
        working["gdp_total_wan_yuan"] = np.nan
    if "gdp_current_price_wan_yuan" in working.columns:
        working["gdp_total_wan_yuan"] = working["gdp_total_wan_yuan"].fillna(working["gdp_current_price_wan_yuan"])

    if "population_total_wan" not in working.columns:
        working["population_total_wan"] = np.nan
    if "resident_population_wan" in working.columns:
        working["population_total_wan"] = working["population_total_wan"].fillna(working["resident_population_wan"])

    # GDP conversions
    if "gdp_total_wan_yuan" in working:
        working["gdp_total_billion_yuan"] = working["gdp_total_wan_yuan"] / 10000.0
    else:
        working["gdp_total_billion_yuan"] = np.nan

    # Per-capita GDP (万元/人)
    if {"gdp_total_wan_yuan", "population_total_wan"} <= set(working.columns):
        working["gdp_per_capita_wan_yuan"] = working["gdp_total_wan_yuan"] / working["population_total_wan"]
    else:
        working["gdp_per_capita_wan_yuan"] = np.nan

    # Emission intensity (Mt / 亿元)
    working["emission_intensity_mt_per_billion_yuan"] = np.where(
        working["gdp_total_billion_yuan"] > 0,
        working["carbon_emission_mt"] / working["gdp_total_billion_yuan"],
        np.nan,
    )

    # Urbanization derived if not directly available.
    if "urbanization_rate_pct" not in working or working["urbanization_rate_pct"].isna().all():
        if {"non_agricultural_population_wan", "population_total_wan"} <= set(working.columns):
            working["urbanization_rate_pct"] = working["non_agricultural_population_wan"] / working["population_total_wan"] * 100.0
        else:
            working["urbanization_rate_pct"] = np.nan

    # Energy structure proxy -> secondary industry share.
    if "secondary_industry_share_pct" in working:
        working["energy_structure_proxy_pct"] = working["secondary_industry_share_pct"]
    else:
        working["energy_structure_proxy_pct"] = np.nan

    return working


def _fill_missing_values(df: pd.DataFrame, group_key: str = "city_code") -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in ["year"]:
        if col in numeric_cols:
            numeric_cols.remove(col)

    filled = df.copy()
    filled.sort_values([group_key, "year"], inplace=True)

    for city_code, group in filled.groupby(group_key):
        idx = group.index
        updated = group[numeric_cols].copy()
        for col in numeric_cols:
            series = updated[col]
            if series.notna().sum() == 0:
                continue
            interp = series.interpolate(method="linear", limit_direction="both")
            interp = interp.ffill().bfill()

            original_mask = series.notna()
            valid_indices = np.where(original_mask)[0]
            first_pos = valid_indices[0]
            last_pos = valid_indices[-1]
            interp_array = interp.to_numpy()
            if first_pos > 0:
                interp_array[:first_pos] = np.nan
            if last_pos < len(interp_array) - 1:
                interp_array[last_pos + 1 :] = np.nan
            updated[col] = interp_array

        filled.loc[idx, numeric_cols] = updated

    return filled


def _compute_growth_rates(df: pd.DataFrame, group_key: str = "city_code") -> pd.DataFrame:
    working = df.copy()
    working["emission_growth_rate_pct"] = working.groupby(group_key)["carbon_emission_mt"].pct_change(fill_method=None) * 100.0
    working["gdp_growth_rate_pct"] = working.groupby(group_key)["gdp_total_billion_yuan"].pct_change(fill_method=None) * 100.0
    return working


def _missing_report(df: pd.DataFrame, identifier_cols: Iterable[str]) -> pd.DataFrame:
    report_rows = []
    total_rows = len(df)
    for column in df.columns:
        if column in identifier_cols:
            continue
        missing_count = df[column].isna().sum()
        report_rows.append(
            {
                "variable": column,
                "missing_count": int(missing_count),
                "missing_rate_pct": round(missing_count / total_rows * 100, 2) if total_rows else 0.0,
            }
        )
    report_df = pd.DataFrame(report_rows).sort_values("missing_rate_pct", ascending=False)
    return report_df


def build_clean_dataset(
    output_dir: Path,
    emissions_path: Path | str = RAW_EMISSIONS_FILE,
    stats_path: Path | str = RAW_STATS_FILE,
    city_codes_path: Path | str = CITY_CODES_FILE,
) -> DataProcessingResult:
    """
    Execute the ETL flow: load, harmonize, enrich, and persist the cleaned dataset.
    """

    city_codes = _load_city_codes(city_codes_path)
    emissions, emission_logs = _load_emissions_data(emissions_path, city_codes)
    stats, stats_logs = _load_statistics_data(stats_path, city_codes)

    unmatched_emissions = sorted({log.original_label for log in emission_logs if not log.matched})
    unmatched_stats = sorted({log.original_label for log in stats_logs if not log.matched})

    matching_log = _build_matching_log(emission_logs, stats_logs)

    merged = _merge_datasets(emissions, stats)
    enriched = _derive_additional_features(merged)
    filled = _fill_missing_values(enriched)
    with_growth = _compute_growth_rates(filled)

    identifier_cols = ["city_code", "city_name", "province_name", "year"]
    missing_report = _missing_report(with_growth, identifier_cols)

    # Persist outputs
    export_dataframe(with_growth, output_dir / "data" / "cleaned_data.csv")
    export_dataframe(matching_log, output_dir / "data" / "city_matching_log.csv")
    export_dataframe(missing_report, output_dir / "data" / "missing_report.csv")

    return DataProcessingResult(
        cleaned_data=with_growth,
        matching_log=matching_log,
        missing_report=missing_report,
        unmatched_emissions=unmatched_emissions,
        unmatched_stats=unmatched_stats,
    )
