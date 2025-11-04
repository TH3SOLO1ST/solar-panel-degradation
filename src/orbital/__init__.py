"""
Orbital Mechanics Module

This module provides tools for satellite orbit propagation, eclipse calculation,
and solar exposure analysis.
"""

from .orbit_propagator import OrbitPropagator
from .eclipse_calculator import EclipseCalculator

__all__ = ["OrbitPropagator", "EclipseCalculator"]