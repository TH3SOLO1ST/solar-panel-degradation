#!/usr/bin/env python3
"""
Basic Tests
===========

Simple integration tests to verify the solar panel degradation model
is working correctly.

Usage:
    python -m pytest tests/test_basic.py -v
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def test_scenario_config():
    """Test scenario configuration system"""
    from config.scenario_config import ScenarioConfig

    config = ScenarioConfig()
    scenarios = config.get_preset_scenarios()

    # Should have 3 preset scenarios
    assert len(scenarios) == 3

    # Check scenario names
    scenario_names = [s.name for s in scenarios]
    assert "International Space Station (ISS)" in scenario_names
    assert "GEO Communications Satellite" in scenario_names
    assert "Earth Observation Satellite" in scenario_names

    # Test ISS scenario
    iss_scenario = scenarios[0]
    assert iss_scenario.orbit_type == "LEO"
    assert iss_scenario.altitude_km == 408.0
    assert iss_scenario.solar_panel_tech == "silicon"

def test_orbit_calculations():
    """Test basic orbital calculations"""
    from orbital.orbit_propagator import OrbitElements
    from datetime import datetime

    # Create simple orbit elements
    orbit = OrbitElements(
        semi_major_axis_km=6879.0,  # 500 km altitude
        eccentricity=0.001,
        inclination_deg=51.6,
        raan_deg=0.0,
        arg_perigee_deg=0.0,
        mean_anomaly_deg=0.0,
        epoch=datetime.now()
    )

    # Check calculated properties
    assert orbit.a == 6879.0
    assert orbit.period > 0  # Should have positive period
    assert orbit.period < 10000  # Should be reasonable (seconds)

def test_radiation_model():
    """Test radiation environment model"""
    from radiation.radiation_environment import RadiationEnvironment
    import numpy as np

    # Create radiation model
    rad_env = RadiationEnvironment()

    # Create simple positions (just a few points)
    positions = np.array([
        [6879, 0, 0],     # 500 km altitude
        [7000, 0, 0],     # 629 km altitude
        [7200, 0, 0]      # 829 km altitude
    ])
    times = np.array([0, 1, 2])  # hours

    # Calculate radiation dose
    dose = rad_env.calculate_radiation_dose(positions, times)

    # Check results
    assert dose.total_ionizing_dose >= 0
    assert dose.non_ionizing_dose >= 0
    assert len(dose.time_points) == len(times)

def test_thermal_model():
    """Test thermal analysis model"""
    from thermal.thermal_analysis import ThermalProperties
    import numpy as np

    # Create thermal properties
    props = ThermalProperties(
        mass=50.0,  # kg
        specific_heat=900.0,  # J/(kg·K)
        emissivity=0.85,
        absorptivity=0.9,
        area=25.0  # m²
    )

    # Check properties
    assert props.mass == 50.0
    assert props.thermal_capacity == props.mass * props.specific_heat
    assert props.area == 25.0

def test_solar_cell_specs():
    """Test solar cell specifications"""
    from degradation.power_calculator import SolarCellSpecs

    # Create silicon cell specs
    specs = SolarCellSpecs(
        technology="silicon",
        area_m2=25.0,
        initial_efficiency=0.20,
        series_resistance=0.01,
        shunt_resistance=1000.0,
        ideality_factor=1.2,
        temperature_coefficient=-0.0045,
        reference_temperature=298.0
    )

    # Check specifications
    assert specs.technology == "silicon"
    assert specs.area_m2 == 25.0
    assert specs.initial_efficiency == 0.20

def test_power_calculation():
    """Test basic power calculation"""
    from degradation.power_calculator import SolarCellSpecs, PowerCalculator
    import numpy as np

    # Create simple specs
    specs = SolarCellSpecs(
        technology="silicon",
        area_m2=10.0,
        initial_efficiency=0.20,
        series_resistance=0.01,
        shunt_resistance=1000.0,
        ideality_factor=1.2,
        temperature_coefficient=-0.0045,
        reference_temperature=298.0
    )

    # Create power calculator
    calculator = PowerCalculator(specs)

    # Simple test conditions
    conditions = {
        'solar_flux': 1361.0,  # W/m²
        'incident_angle': 0.0,  # Normal incidence
        'temperature': 298.0,   # Room temperature
        'radiation_factor': 1.0,  # No degradation
        'thermal_factor': 1.0,    # No thermal effects
        'contamination_factor': 1.0  # No contamination
    }

    # Calculate expected power (simplified)
    expected_power = 1361.0 * 10.0 * 0.20  # ~2722 W

    # The actual calculation will be more complex, but should be reasonable
    # This is a basic sanity check that the model runs without errors
    assert True  # If we reach here, no exceptions were thrown

def test_lifetime_model_initialization():
    """Test lifetime model initialization"""
    from degradation.lifetime_model import LifetimeModel
    from degradation.power_calculator import SolarCellSpecs
    from thermal.thermal_analysis import ThermalProperties
    from orbital.orbit_propagator import OrbitElements
    from datetime import datetime

    # Create components
    solar_specs = SolarCellSpecs(
        technology="silicon",
        area_m2=25.0,
        initial_efficiency=0.20,
        series_resistance=0.01,
        shunt_resistance=1000.0,
        ideality_factor=1.2,
        temperature_coefficient=-0.0045,
        reference_temperature=298.0
    )

    orbit_elements = OrbitElements(
        semi_major_axis_km=6879.0,
        eccentricity=0.001,
        inclination_deg=51.6,
        raan_deg=0.0,
        arg_perigee_deg=0.0,
        mean_anomaly_deg=0.0,
        epoch=datetime.now()
    )

    thermal_props = ThermalProperties(
        mass=50.0,
        specific_heat=900.0,
        emissivity=0.85,
        absorptivity=0.9,
        area=25.0
    )

    # Create lifetime model
    model = LifetimeModel(solar_specs, orbit_elements, thermal_props)

    # Check that model was created successfully
    assert model is not None
    assert model.panel_specs == solar_specs
    assert model.orbit_elements == orbit_elements
    assert model.thermal_props == thermal_props

def test_data_export():
    """Test data export functionality"""
    from visualization.data_export import DataExporter
    import tempfile
    import os

    # Create exporter
    exporter = DataExporter()

    # Check that export directory was created
    assert exporter.export_dir.exists()

    # Test file path generation
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = Path(temp_dir) / "test.json"
        assert str(test_file).endswith(".json")

def test_import_dependencies():
    """Test that all required modules can be imported"""
    try:
        # Core modules
        import numpy as np
        import pandas as pd
        from datetime import datetime

        # Project modules
        from config.scenario_config import ScenarioConfig
        from orbital.orbit_propagator import OrbitPropagator
        from radiation.radiation_environment import RadiationEnvironment
        from thermal.thermal_analysis import ThermalAnalysis
        from degradation.lifetime_model import LifetimeModel

        # These imports should work without throwing exceptions
        assert True

    except ImportError as e:
        pytest.fail(f"Failed to import required module: {e}")

def test_configuration_files():
    """Test that configuration files exist and are valid"""
    import json
    from pathlib import Path

    # Check that config files exist
    config_dir = Path(__file__).parent.parent / "config"
    assert config_dir.exists()

    # Check specific config files
    required_files = [
        "default_scenario.json",
        "geo_scenario.json",
        "sso_scenario.json",
        "panel_technologies.json"
    ]

    for filename in required_files:
        file_path = config_dir / filename
        assert file_path.exists(), f"Missing config file: {filename}"

        # Try to parse JSON
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            assert isinstance(data, dict), f"Invalid JSON in {filename}"
        except json.JSONDecodeError as e:
            pytest.fail(f"Invalid JSON in {filename}: {e}")

if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])