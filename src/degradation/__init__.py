"""
Solar Panel Degradation Module

This module provides tools for calculating power output degradation
over the lifetime of solar panels in orbit.
"""

from .power_calculator import PowerCalculator
from .lifetime_model import LifetimeDegradationModel

__all__ = ["PowerCalculator", "LifetimeDegradationModel"]