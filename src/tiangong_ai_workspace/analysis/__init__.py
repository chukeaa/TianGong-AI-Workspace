"""
Core analysis workflow for the Tiangong AI Workspace carbon project.

This package exposes a high-level `run_full_analysis` entry-point that
orchestrates data cleaning, exploratory analysis, visualization, modelling,
and automated report generation for China's city-level carbon emissions.
"""

from .workflow import AnalysisArtifacts, run_full_analysis

__all__ = ["run_full_analysis", "AnalysisArtifacts"]
