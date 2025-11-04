#!/usr/bin/env python3
"""
Basic Usage Example for Solar Panel Degradation Model

This example demonstrates how to use the solar panel degradation model
for a typical LEO satellite mission scenario.
"""

import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from main import SolarPanelDegradationModel
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main example function"""
    print("=" * 60)
    print("Solar Panel Degradation Model - Basic Usage Example")
    print("=" * 60)

    # Create model instance
    print("\n1. Creating solar panel degradation model...")
    model = SolarPanelDegradationModel()

    # Show available templates
    print("\n2. Available scenario templates:")
    templates = model.list_available_templates()
    for template in templates:
        print(f"   - {template}")

    # Create scenario from template
    print("\n3. Creating ISS-like LEO scenario...")
    success = model.create_scenario_from_template(
        "ISS_Like",
        scenario_name="Example_LEO_Mission",
        description="Example LEO mission demonstration"
    )

    if not success:
        print("Failed to create scenario")
        return

    # Validate configuration
    print("\n4. Validating configuration...")
    validation = model.validate_configuration()
    print(f"Configuration valid: {validation['feasible']}")

    if validation['warnings']:
        print("Warnings:")
        for warning in validation['warnings']:
            print(f"   - {warning}")

    if validation['recommendations']:
        print("Recommendations:")
        for rec in validation['recommendations']:
            print(f"   - {rec}")

    # Get configuration info
    print("\n5. Configuration summary:")
    config_info = model.get_configuration_info()
    print(f"   Mission duration: {config_info['mission_duration_years']} years")
    print(f"   Orbit type: {config_info['orbital_parameters']['orbit_type']}")
    print(f"   Altitude: {config_info['orbital_parameters']['altitude_km']} km")
    print(f"   Solar panel area: {config_info['solar_panel']['area_m2']} m²")
    print(f"   Cell technology: {config_info['solar_panel']['technology']}")

    # Run simulation
    print("\n6. Running simulation...")
    print("   (This may take a few minutes for complete simulation...)")

    try:
        results = model.run_simulation()
        print("   ✓ Simulation completed successfully!")
    except Exception as e:
        print(f"   ✗ Simulation failed: {e}")
        return

    # Get summary
    print("\n7. Simulation Results Summary:")
    summary = model.get_summary()
    print(f"   Initial power: {summary['performance']['initial_power_W']:.2f} W")
    print(f"   Final power: {summary['performance']['final_power_W']:.2f} W")
    print(f"   Total degradation: {summary['performance']['total_degradation_percent']:.2f}%")
    print(f"   Degradation rate: {summary['performance']['degradation_rate_percent_per_year']:.3f}%/year")
    print(f"   Total energy generated: {summary['performance']['total_energy_Wh']:.2f} Wh")
    print(f"   Average power: {summary['performance']['average_power_W']:.2f} W")
    print(f"   Simulation time: {summary['simulation']['time_seconds']:.2f} seconds")

    # End-of-life predictions
    print("\n8. End-of-Life Predictions:")
    eol = summary['end_of_life']
    print(f"   EOL efficiency: {eol['efficiency_percent']:.2f}%")
    if eol['years_to_80_percent']:
        print(f"   Time to 80% power: {eol['years_to_80_percent']:.2f} years")
    if eol['years_to_50_percent']:
        print(f"   Time to 50% power: {eol['years_to_50_percent']:.2f} years")

    # Degradation mechanism contributions
    print("\n9. Degradation Mechanism Breakdown:")
    mechanisms = summary['mechanism_contributions']
    total_contrib = sum(mechanisms.values()) if mechanisms else 0
    for mechanism, contribution in mechanisms.items():
        if total_contrib > 0:
            percent_of_total = (contribution / total_contrib) * 100
            print(f"   {mechanism.replace('_', ' ').title()}: {contribution:.2f}% ({percent_of_total:.1f}% of total)")

    # Generate plots
    print("\n10. Generating visualizations...")
    output_dir = Path("example_results")
    output_dir.mkdir(exist_ok=True)

    try:
        plots = model.plot_results(
            plot_types=["lifetime_trend", "degradation_breakdown", "power_performance"],
            save_path=str(output_dir)
        )
        print(f"    ✓ Generated {len(plots)} plots in {output_dir}")
        print("    Files created:")
        for plot_type in plots.keys():
            print(f"      - {plot_type}.html")
    except Exception as e:
        print(f"    ✗ Plot generation failed: {e}")

    # Export results
    print("\n11. Exporting results...")
    try:
        success = model.export_results(str(output_dir), formats=["excel", "csv"])
        if success:
            print(f"    ✓ Results exported to {output_dir}")
            print("    Files created:")
            print(f"      - Example_LEO_Mission_analysis.xlsx")
            print(f"      - Example_LEO_Mission_analysis.csv")
            print(f"      - Example_LEO_Mission_config.json")
            print(f"      - Example_LEO_Mission_report.json")
        else:
            print("    ✗ Export failed")
    except Exception as e:
        print(f"    ✗ Export failed: {e}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print(f"Check the '{output_dir}' directory for results and plots.")
    print("=" * 60)


if __name__ == "__main__":
    main()