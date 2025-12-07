from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

CITY_NAME_NORMALIZATION_REPLACEMENTS: Sequence[str] = (
    "市",
    "地区",
    "盟",
    "自治州",
    "自治县",
    "自治旗",
    "林区",
    "特别行政区",
    "新区",
    "区域",
    "矿区",
    "直辖县级行政区划",
    "州",
    "县",
    "旗",
)


def normalize_city_name(name: str) -> str:
    """Generate a simplified city token for matching across datasets."""

    if not isinstance(name, str):
        return ""
    normalized = name.strip()
    for token in CITY_NAME_NORMALIZATION_REPLACEMENTS:
        normalized = normalized.replace(token, "")
    normalized = normalized.replace(" ", "").replace("　", "").replace("·", "")
    return normalized


def ensure_output_directories(base_output: Path) -> dict[str, Path]:
    """Create required output directories and return their paths."""

    directories = {
        "data": base_output / "data",
        "figs": base_output / "figs",
        "models": base_output / "models",
        "reports": base_output / "reports",
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return directories


def configure_matplotlib(font_candidates: Iterable[str] | None = None) -> str | None:
    """
    Set up matplotlib for publication-ready Chinese plots.

    Attempts to register one of the provided font candidates; defaults to a list
    that commonly ships with Linux images. Falls back to DejaVu Sans if none of
    the candidates are present.
    """

    default_candidates: List[str] = [
        "SimHei",
        "Noto Sans CJK SC",
        "Noto Sans CJK",
        "Microsoft YaHei",
        "PingFang SC",
        "Source Han Sans SC",
    ]
    candidates = list(font_candidates or default_candidates)

    # Register bundled fonts if available.
    bundled_font_dir = Path(__file__).resolve().parents[2] / "input" / "fonts"
    if bundled_font_dir.exists():
        for font_path in bundled_font_dir.glob("*.otf"):
            try:
                matplotlib.font_manager.fontManager.addfont(str(font_path))
            except Exception:
                continue
        for font_path in bundled_font_dir.glob("*.ttf"):
            try:
                matplotlib.font_manager.fontManager.addfont(str(font_path))
            except Exception:
                continue

    available = {font.name for font in matplotlib.font_manager.fontManager.ttflist}
    chosen_font = None
    for font in candidates:
        if font in available:
            chosen_font = font
            break

    if not chosen_font:
        # Attempt to load system fonts lazily via full path search.
        for ext in ("ttf", "ttc"):
            font_paths = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext=ext)
            for path in font_paths:
                try:
                    font_prop = matplotlib.font_manager.FontProperties(fname=path)
                    name = font_prop.get_name()
                except Exception:
                    continue
                if name in candidates:
                    try:
                        matplotlib.font_manager.fontManager.addfont(path)
                    except Exception:
                        continue
                    chosen_font = name
                    break
            if chosen_font:
                break

    if chosen_font:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [chosen_font]
    else:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]

    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300

    return chosen_font


@dataclass
class MatchingLog:
    city_code: str
    city_name: str
    source: str
    original_label: str
    matched: bool
    note: str = ""


def export_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Persist a dataframe as UTF-8 CSV with consistent settings."""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def export_json(data: dict, path: Path) -> None:
    """Serialize JSON with indentation for readability."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
