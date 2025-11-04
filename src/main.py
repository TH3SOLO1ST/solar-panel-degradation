"""
Solar Panel Degradation Model - Main Entry Point
=================================================

Main application entry point for the solar panel degradation modeling tool.

This module provides command-line interface and standalone execution
for both API server and direct simulation running.

Usage:
    # Run API server
    python -m src.main server --host 0.0.0.0 --port 5000

    # Run single simulation
    python -m src.main simulate --scenario config/iss_scenario.json

    # Interactive mode
    python -m src.main interactive
"""

import argparse
import sys
from pathlib import Path
import json
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent))

from config.scenario_config import ScenarioConfig
from api.server import create_app
from degradation.lifetime_model import LifetimeModel
from degradation.power_calculator import SolarCellSpecs
from thermal.thermal_analysis import ThermalProperties
from orbital.orbit_propagator import OrbitElements
from visualization.interactive_plots import InteractivePlots
from visualization.data_export import DataExporter

def run_server(args):
    """Run Flask API server"""
    print(f"Starting Solar Panel Degradation API Server...")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Access web interface at: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop the server")

    try:
        app = create_app()
        app.run(host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

def simulate_scenario(args):
    """Run single simulation"""
    print(f"Running simulation for scenario: {args.scenario}")

    try:
        # Load scenario configuration
        scenario_config = ScenarioConfig()
        scenario = scenario_config.load_scenario(args.scenario)

        print(f"Scenario: {scenario.name}")
        print(f"Orbit: {scenario.orbit_type} at {scenario.altitude_km} km")
        print(f"Duration: {scenario.mission_duration_years} years")

        # Convert to model inputs
        solar_specs = SolarCellSpecs(
            technology=scenario.solar_panel_tech,
            area_m2=scenario.panel_area_m2,
            initial_efficiency=scenario.initial_efficiency,
            series_resistance=0.01,
            shunt_resistance=1000.0,
            ideality_factor=1.2,
            temperature_coefficient=-0.0045,
            reference_temperature=298.0
        )

        orbit_elements = OrbitElements(
            semi_major_axis_km=6371.0 + scenario.altitude_km,
            eccentricity=scenario.eccentricity,
            inclination_deg=scenario.inclination_deg,
            raan_deg=0.0,
            arg_perigee_deg=0.0,
            mean_anomaly_deg=0.0,
            epoch=datetime.now()
        )

        thermal_props = ThermalProperties(
            mass=scenario.panel_area_m2 * 2.0,
            specific_heat=900.0,
            emissivity=0.85,
            absorptivity=0.9,
            area=scenario.panel_area_m2
        )

        # Create lifetime model
        lifetime_model = LifetimeModel(solar_specs, orbit_elements, thermal_props)

        # Run simulation
        results = lifetime_model.run_lifetime_simulation(scenario.mission_duration_years)

        # Display results summary
        print("\n" + "="*50)
        print("SIMULATION RESULTS")
        print("="*50)

        metrics = results.performance_metrics
        print(f"Initial Power: {metrics['initial_power_W']:.1f} W")
        print(f"Final Power: {metrics['final_power_W']:.1f} W")
        print(f"Power Degradation: {metrics['power_degradation_percent']:.1f}%")
        print(f"Initial Efficiency: {metrics['initial_efficiency']*100:.1f}%")
        print(f"Final Efficiency: {metrics['final_efficiency']*100:.1f}%")
        print(f"Average Power: {metrics['average_power_W']:.1f} W")
        print(f"Total Energy: {results.energy_yield['total_energy_kWh']:.1f} kWh")

        # Environmental summary
        print("\nEnvironmental Conditions:")
        temps = results.environmental_conditions['temperatures']
        print(f"Temperature Range: {min(temps)-273.15:.1f}°C to {max(temps)-273.15:.1f}°C")
        print(f"Mean Temperature: {sum(temps)/len(temps)-273.15:.1f}°C")
        print(f"Total Radiation Dose: {results.environmental_conditions['radiation_dose'].total_ionizing_dose:.1f} rads")
        print(f"Solar Exposure: {results.environmental_conditions['solar_exposure_pct']:.1f}%")

        # Degradation breakdown
        print("\nDegradation Sources:")
        for mechanism, contribution in results.degradation_breakdown.items():
            if contribution > 0:
                print(f"  {mechanism.replace('_', ' ').title()}: {contribution*100:.2f}%")

        # Export results if requested
        if args.export:
            exporter = DataExporter()
            if args.export_format == 'json':
                filepath = exporter.export_json(results, "simulation")
            elif args.export_format == 'csv':
                filepath = exporter.export_csv(results, "simulation")
            elif args.export_format == 'excel':
                filepath = exporter.export_excel(results, "simulation")
            else:
                print(f"Unsupported export format: {args.export_format}")
                return

            print(f"\nResults exported to: {filepath}")

        # Generate plots if requested
        if args.plot:
            try:
                plotter = InteractivePlots()
                fig = plotter.create_lifetime_power_plot(results)
                plotter.export_plot(fig, "simulation_power.png", "png")
                print("Power plot saved to: simulation_power.png")

                fig2 = plotter.create_multi_plot_dashboard(results)
                plotter.export_plot(fig2, "simulation_dashboard.png", "png")
                print("Dashboard plot saved to: simulation_dashboard.png")

            except Exception as e:
                print(f"Warning: Could not generate plots: {e}")

    except Exception as e:
        print(f"Error running simulation: {e}")
        sys.exit(1)

def list_scenarios(args):
    """List available scenarios"""
    scenario_config = ScenarioConfig()
    scenarios = scenario_config.get_preset_scenarios()

    print("Available Scenarios:")
    print("-" * 50)

    for i, scenario in enumerate(scenarios, 1):
        summary = scenario_config.get_scenario_summary(scenario)
        print(f"\n{i}. {summary['Name']}")
        print(f"   Description: {summary['Description']}")
        print(f"   Orbit: {summary['Orbit']}")
        print(f"   Mission: {summary['Mission']}")
        print(f"   Panels: {summary['Solar Panels']}")
        print(f"   Efficiency: {summary['Initial Efficiency']}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Solar Panel Degradation Modeling Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start web server
  solar-panel-model server --host 127.0.0.1 --port 5000

  # Run simulation with preset scenario
  solar-panel-model simulate --scenario config/iss_scenario.json

  # List available scenarios
  solar-panel-model list

  # Run simulation and export results
  solar-panel-model simulate --scenario config/geo_scenario.json --export --format json
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Server command
    server_parser = subparsers.add_parser('server', help='Start API server')
    server_parser.add_argument('--host', default='127.0.0.1', help='Host address')
    server_parser.add_argument('--port', type=int, default=5000, help='Port number')
    server_parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    # Simulate command
    sim_parser = subparsers.add_parser('simulate', help='Run single simulation')
    sim_parser.add_argument('--scenario', required=True, help='Scenario file path')
    sim_parser.add_argument('--export', action='store_true', help='Export results')
    sim_parser.add_argument('--export-format', choices=['json', 'csv', 'excel'],
                           default='json', help='Export format')
    sim_parser.add_argument('--plot', action='store_true', help='Generate plots')

    # List command
    list_parser = subparsers.add_parser('list', help='List available scenarios')

    # Parse arguments
    args = parser.parse_args()

    if args.command == 'server':
        run_server(args)
    elif args.command == 'simulate':
        simulate_scenario(args)
    elif args.command == 'list':
        list_scenarios(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main()