"""
Thermal Analysis Module

This module provides tools for calculating orbital temperature profiles,
thermal cycling effects, and temperature-induced degradation.
"""

from .thermal_analysis import ThermalAnalysis
from .thermal_degradation import ThermalDegradation

__all__ = ["ThermalAnalysis", "ThermalDegradation"]