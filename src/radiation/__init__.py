"""
Radiation Environment Module

This module provides tools for modeling space radiation environment
and calculating radiation-induced damage to solar panels.
"""

from .radiation_environment import RadiationEnvironment
from .damage_model import RadiationDamageModel

__all__ = ["RadiationEnvironment", "RadiationDamageModel"]