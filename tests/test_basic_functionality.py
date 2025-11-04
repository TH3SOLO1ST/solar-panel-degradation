"""
Basic functionality tests for the solar panel degradation model.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.orbital.orbit_propagator import OrbitPropagator, OrbitalElements
from src.radiation.radiation_environment import RadiationEnvironment
from src.thermal.thermal_analysis import ThermalAnalysis
from src.degradation.power_calculator import PowerCalculator
from src.main import SolarPanelDegradationModel


class TestOrbitPropagator:
    """Test orbital propagation functionality"""

    def test_leo_orbit_creation(self):
        """Test LEO orbit creation"""
        propagator = OrbitPropagator()
        elements = OrbitPropagator.create_leo_orbit(500, 45, datetime.now())

        assert elements.semi_major_axis == 6378.137 + 500
        assert elements.inclination == np.radians(45)
        assert elements.eccentricity == 0.001

    def test_geo_orbit_creation(self):
        """Test GEO orbit creation"""
        propagator = OrbitPropagator()
        elements = OrbitPropagator.create_geo_orbit(datetime.now())

        assert elements.semi_major_axis == 42164.0
        assert elements.inclination == 0.0
        assert elements.eccentricity == 0.0

    def test_orbit_propagation(self):
        """Test basic orbit propagation"""
        propagator = OrbitPropagator()
        elements = OrbitPropagator.create_leo_orbit(500, 45, datetime.now())
        propagator.set_orbit_from_elements(elements)

        # Propagate for 1 hour
        state = propagator.propagate(datetime.now() + timedelta(hours=1))

        assert state.altitude > 100  # Should be above Earth surface
        assert state.velocity_magnitude > 0  # Should have velocity
        assert len(state.position) == 3  # Should have 3D position


class TestRadiationEnvironment:
    """Test radiation environment modeling"""

    def test_radiation_environment_init(self):
        """Test radiation environment initialization"""
        rad_env = RadiationEnvironment()
        assert rad_env.solar_activity == 0.5
        assert not rad_env.use_real_data

    def test_trapped_particle_flux(self):
        """Test trapped particle flux calculation"""
        rad_env = RadiationEnvironment()
        position = np.array([7000, 0, 0])  # 622 km altitude
        time = datetime.now()

        fluxes = rad_env.calculate_trapped_particle_flux(position, time)
        assert isinstance(fluxes, list)
        # Should have both electrons and protons
        particle_types = [f.particle_type for f in fluxes]
        assert 'electron' in particle_types
        assert 'proton' in particle_types


class TestThermalAnalysis:
    """Test thermal analysis functionality"""

    def test_thermal_analysis_init(self):
        """Test thermal analysis initialization"""
        orbit_prop = OrbitPropagator()
        from src.orbital.eclipse_calculator import EclipseCalculator
        eclipse_calc = EclipseCalculator(orbit_prop)

        thermal = ThermalAnalysis(orbit_prop, eclipse_calc)
        assert thermal.use_earth_radiation == True
        assert thermal.thermal_props.specific_heat == 900

    def test_solar_heat_flux(self):
        """Test solar heat flux calculation"""
        orbit_prop = OrbitPropagator()
        from src.orbital.eclipse_calculator import EclipseCalculator
        eclipse_calc = EclipseCalculator(orbit_prop)

        thermal = ThermalAnalysis(orbit_prop, eclipse_calc)
        panel_normal = np.array([0, 0, 1])

        flux = thermal.calculate_solar_heat_flux(datetime.now(), panel_normal)
        assert isinstance(flux, (int, float))
        assert flux >= 0  # Flux should be non-negative


class TestPowerCalculator:
    """Test power calculation functionality"""

    def test_power_calculator_init(self):
        """Test power calculator initialization"""
        power_calc = PowerCalculator(panel_area_m2=10.0)
        assert power_calc.panel_area == 10.0
        assert power_calc.cell_type == "silicon"

    def test_thermal_voltage(self):
        """Test thermal voltage calculation"""
        power_calc = PowerCalculator(panel_area_m2=1.0)

        vt = power_calc.calculate_thermal_voltage(300.0)  # 300K
        expected_vt = power_calc.k_B * 300.0 / power_calc.q
        assert abs(vt - expected_vt) < 1e-10

    def test_power_output_calculation(self):
        """Test basic power output calculation"""
        power_calc = PowerCalculator(panel_area_m2=10.0)
        time = datetime.now()

        output = power_calc.calculate_power_output(
            time=time,
            irradiance_wm2=1000,
            temperature_K=298.15,
            degradation_factor=1.0,
            eclipse_fraction=0.0
        )

        assert output.power_watts > 0
        assert output.voltage_volts > 0
        assert output.current_amps > 0
        assert 0 <= output.efficiency <= 1


class TestMainInterface:
    """Test main interface functionality"""

    def test_model_initialization(self):
        """Test model initialization"""
        model = SolarPanelDegradationModel()
        assert model.scenario_manager is not None
        assert model.data_formats is not None
        assert not model.is_initialized
        assert not model.is_simulation_complete

    def test_template_listing(self):
        """Test template listing"""
        model = SolarPanelDegradationModel()
        templates = model.list_available_templates()
        assert isinstance(templates, list)
        assert len(templates) > 0
        assert "ISS_Like" in templates

    def test_scenario_creation_from_template(self):
        """Test scenario creation from template"""
        model = SolarPanelDegradationModel()
        success = model.create_scenario_from_template("ISS_Like")
        assert success
        assert model.is_initialized
        assert model.config is not None
        assert model.config.scenario_name == "ISS_Like_Mission"

    def test_configuration_validation(self):
        """Test configuration validation"""
        model = SolarPanelDegradationModel()
        model.create_scenario_from_template("ISS_Like")

        validation = model.validate_configuration()
        assert isinstance(validation, dict)
        assert 'feasible' in validation
        assert 'warnings' in validation
        assert 'recommendations' in validation


class TestIntegration:
    """Integration tests for the complete system"""

    def test_simple_simulation(self):
        """Test a simple simulation run"""
        model = SolarPanelDegradationModel()
        model.create_scenario_from_template("ISS_Like")

        # Run a very short simulation for testing
        # Modify time step to make it faster
        model.config.mission.duration_years = 0.01  # ~3.65 days
        model.config.mission.time_step_hours = 6.0

        results = model.run_simulation()

        assert results is not None
        assert 'lifetime_prediction' in results
        assert 'lifetime_history' in results
        assert model.is_simulation_complete

        # Check results
        summary = model.get_summary()
        assert 'performance' in summary
        assert summary['performance']['initial_power_W'] > 0
        assert summary['performance']['final_power_W'] > 0

    def test_plot_generation(self):
        """Test plot generation"""
        model = SolarPanelDegradationModel()
        model.create_scenario_from_template("ISS_Like")

        # Run short simulation
        model.config.mission.duration_years = 0.01
        model.config.mission.time_step_hours = 6.0
        model.run_simulation()

        # Generate plots
        plots = model.plot_results(plot_types=["lifetime_trend"])

        assert isinstance(plots, dict)
        assert len(plots) > 0


def test_basic_workflow():
    """Test the complete basic workflow"""
    print("Testing complete solar panel degradation workflow...")

    # Create model
    model = SolarPanelDegradationModel()

    # Create scenario
    assert model.create_scenario_from_template("ISS_Like")

    # Validate configuration
    validation = model.validate_configuration()
    assert validation['feasible']

    # Run short simulation
    model.config.mission.duration_years = 0.02  # ~7 days
    model.config.mission.time_step_hours = 12.0

    print("Running simulation...")
    results = model.run_simulation()
    assert results is not None

    # Get summary
    summary = model.get_summary()
    print(f"Initial power: {summary['performance']['initial_power_W']:.2f} W")
    print(f"Final power: {summary['performance']['final_power_W']:.2f} W")
    print(f"Degradation: {summary['performance']['total_degradation_percent']:.2f}%")

    # Basic sanity checks
    assert summary['performance']['initial_power_W'] > 0
    assert summary['performance']['final_power_W'] > 0
    assert 0 <= summary['performance']['total_degradation_percent'] <= 100

    print("✓ Basic workflow test passed!")


if __name__ == "__main__":
    # Run the basic workflow test
    test_basic_workflow()

    # Run pytest for all tests
    print("\nRunning pytest...")
    pytest.main([__file__, "-v"])