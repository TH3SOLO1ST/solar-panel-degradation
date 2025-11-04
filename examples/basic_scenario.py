#!/usr/bin/env python3
"""
Basic Scenario Example
======================

This example demonstrates how to use the Solar Panel Degradation Model
with a basic configuration. It shows the simplest way to run a simulation
and access the results.

Usage:
    python examples/basic_scenario.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config.scenario_config import ScenarioConfig, SatelliteSpecs
from degradation.lifetime_model import LifetimeModel
from degradation.power_calculator import SolarCellSpecs
from thermal.thermal_analysis import ThermalProperties
from orbital.orbit_propagator import OrbitElements
from datetime import datetime

def main():
    """Run basic scenario example"""
    print("=" * 60)
    print("Solar Panel Degradation Model - Basic Example")
    print("=" * 60)

    # Step 1: Create a simple satellite scenario
    print("\n1. Creating satellite scenario...")

    scenario = SatelliteSpecs(
        name="Example LEO Satellite",
        description="A simple low Earth orbit communications satellite",
        orbit_type="LEO",
        altitude_km=500.0,
        inclination_deg=51.6,
        eccentricity=0.001,
        period_minutes=94.7,
        solar_panel_tech="silicon",
        panel_area_m2=30.0,
        initial_efficiency=0.20,
        mission_duration_years=5.0,
        expected_degradation_pct=18.0
    )

    print(f"   Name: {scenario.name}")
    print(f"   Orbit: {scenario.orbit_type} at {scenario.altitude_km} km")
    print(f"   Mission: {scenario.mission_duration_years} years")
    print(f"   Panels: {scenario.solar_panel_tech}, {scenario.panel_area_m2} m²")

    # Step 2: Convert to model inputs
    print("\n2. Preparing model inputs...")

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

    print("   ✓ Solar panel specifications configured")
    print("   ✓ Orbital elements calculated")
    print("   ✓ Thermal properties set")

    # Step 3: Create and run lifetime model
    print("\n3. Running lifetime simulation...")
    print("   (This may take 30-60 seconds)")

    try:
        lifetime_model = LifetimeModel(solar_specs, orbit_elements, thermal_props)
        results = lifetime_model.run_lifetime_simulation(scenario.mission_duration_years)
        print("   ✓ Simulation completed successfully")

    except Exception as e:
        print(f"   ✗ Simulation failed: {e}")
        return

    # Step 4: Display results
    print("\n4. Results Summary:")
    print("-" * 40)

    metrics = results.performance_metrics
    print(f"Initial Power:       {metrics['initial_power_W']:.1f} W")
    print(f"Final Power:         {metrics['final_power_W']:.1f} W")
    print(f"Power Degradation:   {metrics['power_degradation_percent']:.1f}%")
    print(f"Initial Efficiency:  {metrics['initial_efficiency']*100:.1f}%")
    print(f"Final Efficiency:    {metrics['final_efficiency']*100:.1f}%")
    print(f"Average Power:       {metrics['average_power_W']:.1f} W")
    print(f"Total Energy:        {results.energy_yield['total_energy_kWh']:.1f} kWh")

    # Step 5: Environmental conditions
    print("\n5. Environmental Conditions:")
    print("-" * 40)

    temps = results.environmental_conditions['temperatures']
    temp_min = min(temps) - 273.15
    temp_max = max(temps) - 273.15
    temp_mean = sum(temps)/len(temps) - 273.15

    print(f"Temperature Range:  {temp_min:.1f}°C to {temp_max:.1f}°C")
    print(f"Mean Temperature:   {temp_mean:.1f}°C")
    print(f"Total Radiation:    {results.environmental_conditions['radiation_dose'].total_ionizing_dose:.1f} rads")
    print(f"Solar Exposure:     {results.environmental_conditions['solar_exposure_pct']:.1f}%")

    # Step 6: Degradation breakdown
    print("\n6. Degradation Sources:")
    print("-" * 40)

    for mechanism, contribution in results.degradation_breakdown.items():
        if contribution > 0:
            print(f"{mechanism.replace('_', ' ').title():<20} {contribution*100:>6.2f}%")

    # Step 7: Performance analysis
    print("\n7. Performance Analysis:")
    print("-" * 40)

    degradation_pct = metrics['power_degradation_percent']
    energy_per_year = results.energy_yield['total_energy_kWh'] / scenario.mission_duration_years

    print(f"Degradation Rate:    {degradation_pct/scenario.mission_duration_years:.2f}% per year")
    print(f"Energy per Year:     {energy_per_year:.1f} kWh/year")
    print(f"Capacity Factor:     {results.energy_yield.get('capacity_factor', 0)*100:.1f}%")

    # Step 8: Export results
    print("\n8. Exporting Results:")

    try:
        from visualization.data_export import DataExporter
        exporter = DataExporter()

        # Export CSV
        csv_file = exporter.export_csv(results, "basic_example")
        print(f"   ✓ CSV exported: {csv_file}")

        # Export JSON
        json_file = exporter.export_json(results, "basic_example")
        print(f"   ✓ JSON exported: {json_file}")

    except Exception as e:
        print(f"   ⚠ Export warning: {e}")

    # Step 9: Create simple plots
    print("\n9. Generating Plots:")

    try:
        from visualization.interactive_plots import InteractivePlots
        import plotly.io as pio

        plotter = InteractivePlots()

        # Power plot
        power_fig = plotter.create_lifetime_power_plot(results)
        power_file = plotter.export_plot(power_fig, "basic_power_plot", "html")
        print(f"   ✓ Power plot saved: {power_file}")

        # Efficiency plot
        eff_fig = plotter.create_efficiency_plot(results)
        eff_file = plotter.export_plot(eff_fig, "basic_efficiency_plot", "html")
        print(f"   ✓ Efficiency plot saved: {eff_file}")

    except Exception as e:
        print(f"   ⚠ Plot warning: {e}")

    print("\n" + "=" * 60)
    print("Basic example completed successfully!")
    print("Check the exported files and plots for detailed results.")
    print("=" * 60)

if __name__ == "__main__":
    main()