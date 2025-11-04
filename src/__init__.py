"""
Solar Panel Degradation Modeling Tool
======================================

A user-friendly tool for modeling solar panel power degradation in orbit.
This package provides scientific calculations for radiation damage,
temperature effects, and eclipse periods on solar panels.

Main Components:
- Orbital mechanics and eclipse calculations
- Radiation environment modeling
- Thermal cycling analysis
- Power degradation prediction
- Interactive visualization
- User-friendly interface

Usage:
    >>> from src.main import SolarPanelDegradationModel
    >>> model = SolarPanelDegradationModel('config/iss_scenario.json')
    >>> results = model.run_simulation()
    >>> model.plot_results()
"""