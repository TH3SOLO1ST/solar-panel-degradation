"""
Solar Panel Degradation Model - Main Interface

This module provides the main interface for the solar panel degradation analysis
system. It integrates all modules and provides a high-level API for running
simulations, managing scenarios, and generating results.

Usage:
    from src.main import SolarPanelDegradationModel

    model = SolarPanelDegradationModel()
    model.load_scenario('config/iss_scenario.json')
    results = model.run_simulation()
    model.plot_results()
    model.export_results('results/')
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging

# Import all required modules
from .orbital.orbit_propagator import OrbitPropagator, OrbitalElements
from .orbital.eclipse_calculator import EclipseCalculator
from .radiation.radiation_environment import RadiationEnvironment
from .radiation.damage_model import RadiationDamageModel
from .thermal.thermal_analysis import ThermalAnalysis
from .thermal.thermal_degradation import ThermalDegradation
from .degradation.power_calculator import PowerCalculator
from .degradation.lifetime_model import LifetimeDegradationModel
from .visualization.interactive_plots import InteractivePlots
from .visualization.data_export import DataExport, ExportMetadata
from .config.scenario_config import ScenarioConfig, SimulationConfig
from .config.data_formats import DataFormats


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SolarPanelDegradationModel:
    """
    Main interface for solar panel degradation analysis.

    This class provides a high-level API for running complete degradation
    simulations, from mission configuration to results visualization and export.

    Features:
    - Complete simulation workflow
    - Multiple mission scenarios
    - Interactive visualizations
    - Comprehensive data export
    - Batch simulation support
    - Progress monitoring
    """

    def __init__(self, config: Optional[SimulationConfig] = None,
                 log_level: str = "INFO"):
        """
        Initialize solar panel degradation model

        Args:
            config: Simulation configuration
            log_level: Logging level
        """
        # Set logging level
        logger.setLevel(getattr(logging, log_level.upper()))

        # Configuration
        self.config = config
        self.scenario_manager = ScenarioConfig()
        self.data_formats = DataFormats()

        # Initialize all modules (will be configured when scenario is loaded)
        self.orbit_propagator: Optional[OrbitPropagator] = None
        self.eclipse_calculator: Optional[EclipseCalculator] = None
        self.radiation_environment: Optional[RadiationEnvironment] = None
        self.radiation_damage: Optional[RadiationDamageModel] = None
        self.thermal_analysis: Optional[ThermalAnalysis] = None
        self.thermal_degradation: Optional[ThermalDegradation] = None
        self.power_calculator: Optional[PowerCalculator] = None
        self.lifetime_model: Optional[LifetimeDegradationModel] = None

        # Results storage
        self.results: Dict[str, Any] = {}
        self.simulation_time: Optional[float] = None

        # Visualization and export
        self.plotter: Optional[InteractivePlots] = None
        self.exporter: Optional[DataExport] = None

        # Status tracking
        self.is_initialized = False
        self.is_simulation_complete = False

        # Load configuration if provided
        if config:
            self._configure_modules()

    def load_scenario(self, filepath: str) -> bool:
        """
        Load simulation scenario from file

        Args:
            filepath: Path to configuration file

        Returns:
            True if successful
        """
        try:
            logger.info(f"Loading scenario from {filepath}")
            self.config = self.scenario_manager.load_config(filepath)
            self._configure_modules()
            self.is_initialized = True
            logger.info("Scenario loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load scenario: {e}")
            return False

    def create_scenario_from_template(self, template_name: str, **kwargs) -> bool:
        """
        Create scenario from predefined template

        Args:
            template_name: Name of template
            **kwargs: Configuration overrides

        Returns:
            True if successful
        """
        try:
            logger.info(f"Creating scenario from template: {template_name}")
            self.config = self.scenario_manager.create_from_template(template_name, **kwargs)
            self._configure_modules()
            self.is_initialized = True
            logger.info("Scenario created successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to create scenario: {e}")
            return False

    def _configure_modules(self):
        """Configure all simulation modules based on current configuration"""
        if not self.config:
            raise ValueError("No configuration loaded")

        logger.info("Configuring simulation modules...")

        # Configure orbital mechanics
        self.orbit_propagator = OrbitPropagator(use_sgp4=True)
        self._setup_orbit()

        # Configure eclipse calculator
        self.eclipse_calculator = EclipseCalculator(self.orbit_propagator)

        # Configure radiation environment
        self.radiation_environment = RadiationEnvironment(
            use_real_data=self.config.environment.use_real_data,
            solar_activity=self.config.environment.solar_activity
        )

        # Configure radiation damage model
        self.radiation_damage = RadiationDamageModel(
            cell_technology=self.config.solar_panel.technology.value,
            temperature_K=298.15  # Will be updated during simulation
        )

        # Configure thermal analysis
        self.thermal_analysis = ThermalAnalysis(
            orbit_propagator=self.orbit_propagator,
            eclipse_calculator=self.eclipse_calculator,
            use_earth_radiation=True
        )

        # Configure thermal degradation
        self.thermal_degradation = ThermalDegradation(
            cell_technology=self.config.solar_panel.technology.value
        )

        # Configure power calculator
        self.power_calculator = PowerCalculator(
            panel_area_m2=self.config.solar_panel.area_m2,
            cell_type=self.config.solar_panel.technology.value
        )

        # Configure lifetime degradation model
        self.lifetime_model = LifetimeDegradationModel(
            orbit_propagator=self.orbit_propagator,
            radiation_environment=self.radiation_environment,
            radiation_damage_model=self.radiation_damage,
            thermal_analysis=self.thermal_analysis,
            thermal_degradation=self.thermal_degradation,
            power_calculator=self.power_calculator,
            eclipse_calculator=self.eclipse_calculator
        )

        # Configure visualization and export
        self.plotter = InteractivePlots()
        self.exporter = DataExport()

        logger.info("All modules configured successfully")

    def _setup_orbit(self):
        """Setup orbit based on configuration"""
        orbit_config = self.config.orbit

        if orbit_config.orbit_type.value == "CUSTOM":
            # Use provided orbital elements
            elements = OrbitalElements(
                semi_major_axis=orbit_config.semi_major_axis_km or 6378.137 + orbit_config.altitude_km,
                eccentricity=orbit_config.eccentricity,
                inclination=np.radians(orbit_config.inclination_deg),
                raan=np.radians(orbit_config.raan_deg),
                arg_perigee=np.radians(orbit_config.arg_perigee_deg),
                mean_anomaly=np.radians(orbit_config.mean_anomaly_deg),
                epoch=orbit_config.epoch
            )
            self.orbit_propagator.set_orbit_from_elements(elements)

        else:
            # Use predefined orbit types
            if orbit_config.orbit_type.value == "LEO":
                elements = OrbitPropagator.create_leo_orbit(
                    altitude_km=orbit_config.altitude_km,
                    inclination_deg=orbit_config.inclination_deg,
                    epoch=self.config.mission.start_time
                )
            elif orbit_config.orbit_type.value == "GEO":
                elements = OrbitPropagator.create_geo_orbit(self.config.mission.start_time)
            elif orbit_config.orbit_type.value == "SSO":
                elements = OrbitPropagator.create_leo_orbit(
                    altitude_km=orbit_config.altitude_km,
                    inclination_deg=98.6,  # Standard SSO inclination
                    epoch=self.config.mission.start_time
                )
            else:
                # Default to circular orbit
                elements = OrbitPropagator.create_leo_orbit(
                    altitude_km=orbit_config.altitude_km,
                    inclination_deg=orbit_config.inclination_deg,
                    epoch=self.config.mission.start_time
                )

            self.orbit_propagator.set_orbit_from_elements(elements)

    def run_simulation(self, progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Run complete degradation simulation

        Args:
            progress_callback: Optional callback for progress updates

        Returns:
            Simulation results
        """
        if not self.is_initialized:
            raise ValueError("Model not initialized. Load a scenario first.")

        logger.info("Starting simulation...")
        start_time = datetime.now()

        try:
            # Run lifetime simulation
            logger.info("Running lifetime degradation simulation...")
            lifetime_prediction = self.lifetime_model.simulate_lifetime(
                launch_time=self.config.mission.start_time,
                duration_years=self.config.mission.duration_years,
                time_step_hours=self.config.mission.time_step_hours
            )

            # Get lifetime history
            lifetime_history = self.lifetime_model.lifetime_history

            # Generate power data for detailed analysis
            power_outputs = self._generate_power_data(lifetime_history)

            # Generate thermal data
            thermal_states = self._generate_thermal_data(lifetime_history)

            # Calculate simulation time
            end_time = datetime.now()
            self.simulation_time = (end_time - start_time).total_seconds()

            # Store results
            self.results = {
                'lifetime_prediction': lifetime_prediction,
                'lifetime_history': lifetime_history,
                'power_outputs': power_outputs,
                'thermal_states': thermal_states,
                'simulation_metadata': {
                    'config': self.config.dict(),
                    'simulation_time_seconds': self.simulation_time,
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat(),
                    'total_data_points': len(lifetime_history)
                }
            }

            self.is_simulation_complete = True
            logger.info(f"Simulation completed in {self.simulation_time:.2f} seconds")

            # Call progress callback if provided
            if progress_callback:
                progress_callback(1.0, "Simulation complete")

            return self.results

        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            raise

    def _generate_power_data(self, lifetime_history: List) -> List:
        """Generate detailed power data from lifetime history"""
        power_outputs = []

        # Panel normal (assuming Sun-pointing panel)
        panel_normal = np.array([0, 0, 1])

        for state in lifetime_history:
            # Calculate power at this time
            power_output = self.power_calculator.calculate_power_output(
                time=state.time,
                irradiance_wm2=1361,  # Simplified solar constant
                temperature_K=298.15,  # Simplified temperature
                degradation_factor=state.efficiency_factor,
                eclipse_fraction=0.0  # Simplified eclipse
            )
            power_outputs.append(power_output)

        return power_outputs

    def _generate_thermal_data(self, lifetime_history: List) -> List:
        """Generate thermal data from lifetime history"""
        thermal_states = []

        # Sample thermal data (simplified)
        for state in lifetime_history:
            # Create thermal state with reasonable values
            thermal_state = type('ThermalState', (), {
                'time': state.time,
                'temperature': 298.15 + 50 * np.sin(state.mission_time_hours * 2 * np.pi / 24),  # Daily temperature variation
                'heat_flux_solar': 1361 * state.efficiency_factor,
                'heat_flux_albedo': 100,
                'heat_flux_earth_ir': 200,
                'heat_flux_radiated': 450,
                'net_heat_flux': 10,
                'eclipse_status': False,
                'temperature_gradient': 5.0
            })()
            thermal_states.append(thermal_state)

        return thermal_states

    def plot_results(self, plot_types: Optional[List[str]] = None,
                    save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate visualization plots

        Args:
            plot_types: Types of plots to generate
            save_path: Optional path to save plots

        Returns:
            Dictionary of plot figures
        """
        if not self.is_simulation_complete:
            raise ValueError("No simulation results available. Run simulation first.")

        if plot_types is None:
            plot_types = ["lifetime_trend", "degradation_breakdown", "power_performance", "environmental"]

        logger.info("Generating plots...")
        plots = {}

        try:
            if "lifetime_trend" in plot_types:
                fig = self.plotter.plot_lifetime_power_trend(
                    self.results['lifetime_history'],
                    title=f"{self.config.scenario_name} - Lifetime Power Trend"
                )
                plots['lifetime_trend'] = fig

                if save_path:
                    self.plotter.export_plot(fig, f"{save_path}/lifetime_trend.html")

            if "degradation_breakdown" in plot_types:
                fig = self.plotter.plot_degradation_breakdown(
                    self.results['lifetime_history'],
                    title=f"{self.config.scenario_name} - Degradation Breakdown"
                )
                plots['degradation_breakdown'] = fig

                if save_path:
                    self.plotter.export_plot(fig, f"{save_path}/degradation_breakdown.html")

            if "power_performance" in plot_types:
                fig = self.plotter.plot_power_performance_metrics(
                    self.results['power_outputs'],
                    title=f"{self.config.scenario_name} - Power Performance"
                )
                plots['power_performance'] = fig

                if save_path:
                    self.plotter.export_plot(fig, f"{save_path}/power_performance.html")

            if "environmental" in plot_types:
                fig = self.plotter.plot_environmental_conditions(
                    self.results['lifetime_history'],
                    title=f"{self.config.scenario_name} - Environmental Conditions"
                )
                plots['environmental'] = fig

                if save_path:
                    self.plotter.export_plot(fig, f"{save_path}/environmental.html")

            logger.info(f"Generated {len(plots)} plots")
            return plots

        except Exception as e:
            logger.error(f"Plot generation failed: {e}")
            return {}

    def export_results(self, output_dir: str, formats: Optional[List[str]] = None) -> bool:
        """
        Export simulation results

        Args:
            output_dir: Output directory path
            formats: Export formats

        Returns:
            True if successful
        """
        if not self.is_simulation_complete:
            raise ValueError("No simulation results available. Run simulation first.")

        if formats is None:
            formats = ["excel", "csv"]

        logger.info(f"Exporting results to {output_dir}...")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            # Create metadata
            metadata = ExportMetadata(
                export_timestamp=datetime.now(),
                software_version="1.0.0",
                scenario_name=self.config.scenario_name,
                mission_duration_hours=self.config.mission.duration_years * 365.25 * 24,
                data_types=["lifetime", "power", "thermal"],
                units={"power": "W", "time": "hours", "temperature": "K"},
                notes=f"Generated by Solar Panel Degradation Model v1.0.0"
            )

            # Export complete analysis
            base_filepath = output_path / f"{self.config.scenario_name}_analysis"

            if "excel" in formats:
                success = self.exporter.export_complete_analysis(
                    self.results['lifetime_history'],
                    self.results['power_outputs'],
                    self.results['thermal_states'],
                    [],  # thermal cycles would be generated here
                    [],  # orbital states would be generated here
                    str(base_filepath.with_suffix('.xlsx')),
                    metadata
                )
                if success:
                    logger.info("Excel export successful")

            if "csv" in formats:
                success = self.exporter.export_lifetime_data(
                    self.results['lifetime_history'],
                    str(base_filepath.with_suffix('.csv')),
                    "csv",
                    metadata
                )
                if success:
                    logger.info("CSV export successful")

            # Export configuration
            config_path = output_path / f"{self.config.scenario_name}_config.json"
            self.scenario_manager.save_config(self.config, str(config_path))

            # Export summary report
            report_path = output_path / f"{self.config.scenario_name}_report.json"
            self.exporter.export_analysis_report(
                self.results['lifetime_history'],
                self.results['power_outputs'],
                self.results['thermal_states'],
                str(report_path)
            )

            logger.info(f"Results exported to {output_dir}")
            return True

        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False

    def get_summary(self) -> Dict[str, Any]:
        """
        Get simulation summary

        Returns:
            Summary dictionary
        """
        if not self.is_simulation_complete:
            return {"status": "No simulation completed"}

        prediction = self.results['lifetime_prediction']

        return {
            'scenario': {
                'name': self.config.scenario_name,
                'description': self.config.description,
                'mission_duration_years': self.config.mission.duration_years
            },
            'performance': {
                'initial_power_W': prediction.initial_power_watts,
                'final_power_W': prediction.final_power_watts,
                'total_degradation_percent': prediction.total_degradation_percent,
                'degradation_rate_percent_per_year': prediction.degradation_rate_percent_per_year,
                'average_power_W': prediction.average_power_watts,
                'total_energy_Wh': prediction.total_energy_Wh
            },
            'end_of_life': {
                'efficiency_percent': prediction.eol_efficiency_percent,
                'years_to_80_percent': prediction.years_to_80_percent,
                'years_to_50_percent': prediction.years_to_50_percent
            },
            'mechanism_contributions': prediction.mechanism_contributions,
            'simulation': {
                'time_seconds': self.simulation_time,
                'data_points': len(self.results['lifetime_history']),
                'completion_time': datetime.now().isoformat()
            }
        }

    def compare_scenarios(self, scenarios: Dict[str, str]) -> Dict[str, Any]:
        """
        Compare multiple scenarios

        Args:
            scenarios: Dictionary of scenario name to config file path

        Returns:
            Comparison results
        """
        logger.info("Comparing scenarios...")

        # Store current state
        current_config = self.config
        current_results = self.results
        current_complete = self.is_simulation_complete

        comparison_results = {}

        try:
            for scenario_name, config_path in scenarios.items():
                logger.info(f"Running scenario: {scenario_name}")

                # Load and run scenario
                if self.load_scenario(config_path):
                    self.run_simulation()
                    comparison_results[scenario_name] = self.results['lifetime_prediction']
                else:
                    logger.error(f"Failed to load scenario: {scenario_name}")

            # Generate comparison plot
            if comparison_results:
                lifetime_scenarios = {}
                for name, prediction in comparison_results.items():
                    # Generate simplified lifetime states for comparison
                    lifetime_scenarios[name] = []  # Would need actual lifetime states

                comparison_plot = self.plotter.compare_scenarios(lifetime_scenarios)
                comparison_results['comparison_plot'] = comparison_plot

            # Restore original state
            self.config = current_config
            self.results = current_results
            self.is_simulation_complete = current_complete

            if current_config:
                self._configure_modules()

            logger.info("Scenario comparison completed")
            return comparison_results

        except Exception as e:
            logger.error(f"Scenario comparison failed: {e}")
            return {}

    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate current configuration

        Returns:
            Validation results
        """
        if not self.config:
            return {"valid": False, "error": "No configuration loaded"}

        return self.scenario_manager.validate_mission_feasibility(self.config)

    def list_available_templates(self) -> List[str]:
        """
        List available scenario templates

        Returns:
            List of template names
        """
        return self.scenario_manager.list_templates()

    def get_configuration_info(self) -> Dict[str, Any]:
        """
        Get current configuration information

        Returns:
            Configuration summary
        """
        if not self.config:
            return {"status": "No configuration loaded"}

        return self.scenario_manager.generate_config_summary(self.config)


def main():
    """Command line interface for the solar panel degradation model"""
    import argparse

    parser = argparse.ArgumentParser(description="Solar Panel Degradation Model")
    parser.add_argument("config", help="Configuration file path")
    parser.add_argument("--output", "-o", help="Output directory", default="results")
    parser.add_argument("--plots", action="store_true", help="Generate plots")
    parser.add_argument("--template", help="Create scenario from template")
    parser.add_argument("--list-templates", action="store_true", help="List available templates")

    args = parser.parse_args()

    # List templates
    if args.list_templates:
        model = SolarPanelDegradationModel()
        templates = model.list_available_templates()
        print("Available templates:")
        for template in templates:
            print(f"  - {template}")
        return

    # Create model
    model = SolarPanelDegradationModel()

    # Create from template or load configuration
    if args.template:
        success = model.create_scenario_from_template(args.template)
    else:
        success = model.load_scenario(args.config)

    if not success:
        print("Failed to load configuration")
        return

    # Validate configuration
    validation = model.validate_configuration()
    if not validation['feasible']:
        print("Configuration validation failed:")
        for issue in validation['issues']:
            print(f"  - {issue}")
        return

    # Run simulation
    print("Running simulation...")
    results = model.run_simulation()

    # Generate plots
    if args.plots:
        print("Generating plots...")
        model.plot_results(save_path=args.output)

    # Export results
    print("Exporting results...")
    model.export_results(args.output)

    # Print summary
    summary = model.get_summary()
    print("\nSimulation Summary:")
    print(f"  Initial Power: {summary['performance']['initial_power_W']:.2f} W")
    print(f"  Final Power: {summary['performance']['final_power_W']:.2f} W")
    print(f"  Total Degradation: {summary['performance']['total_degradation_percent']:.2f}%")
    print(f"  Degradation Rate: {summary['performance']['degradation_rate_percent_per_year']:.2f}%/year")
    print(f"  Total Energy: {summary['performance']['total_energy_Wh']:.2f} Wh")
    print(f"  Simulation Time: {summary['simulation']['time_seconds']:.2f} seconds")


if __name__ == "__main__":
    main()