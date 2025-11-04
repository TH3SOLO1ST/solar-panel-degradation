#!/usr/bin/env python3
"""
Scenario Comparison Example

This example demonstrates how to compare different mission scenarios
to analyze the impact of orbital parameters and solar panel technology
on degradation rates.
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
    """Main comparison example function"""
    print("=" * 60)
    print("Solar Panel Degradation Model - Scenario Comparison Example")
    print("=" * 60)

    # Create model instance
    model = SolarPanelDegradationModel()

    # Define scenarios to compare
    scenarios = {
        "LEO_Silicon": "ISS_Like",  # LEO with silicon cells
        "GEO_MJ": "GEO_Communication_Satellite",  # GEO with multi-junction
        "SSO_Enhanced": "SSO_Earth_Observation"  # SSO with enhanced silicon
    }

    print("\n1. Comparing the following scenarios:")
    for name, template in scenarios.items():
        print(f"   - {name}: {template}")

    # Store original state
    comparison_results = {}

    # Run each scenario
    print("\n2. Running simulations for each scenario...")
    for scenario_name, template in scenarios.items():
        print(f"\n   Running {scenario_name}...")

        # Create scenario from template
        success = model.create_scenario_from_template(template)
        if not success:
            print(f"   ✗ Failed to create {scenario_name}")
            continue

        # Run simulation
        try:
            results = model.run_simulation()
            summary = model.get_summary()
            comparison_results[scenario_name] = summary
            print(f"   ✓ {scenario_name} completed")
            print(f"     Initial power: {summary['performance']['initial_power_W']:.1f} W")
            print(f"     Final power: {summary['performance']['final_power_W']:.1f} W")
            print(f"     Degradation: {summary['performance']['total_degradation_percent']:.1f}%")
        except Exception as e:
            print(f"   ✗ {scenario_name} failed: {e}")

    # Generate comparison table
    print("\n3. Scenario Comparison Results:")
    print("-" * 80)
    print(f"{'Scenario':<15} {'Initial (W)':<12} {'Final (W)':<11} {'Degradation (%)':<15} {'Rate (%/yr)':<12} {'Energy (Wh)':<12}")
    print("-" * 80)

    for name, summary in comparison_results.items():
        perf = summary['performance']
        print(f"{name:<15} {perf['initial_power_W']:<12.1f} {perf['final_power_W']:<11.1f} "
              f"{perf['total_degradation_percent']:<15.2f} {perf['degradation_rate_percent_per_year']:<12.3f} "
              f"{perf['total_energy_Wh']:<12.0f}")

    # Analysis and insights
    print("\n4. Key Insights:")
    print("-" * 40)

    # Find best and worst performing scenarios
    if comparison_results:
        # Best for minimum degradation
        best_degradation = min(comparison_results.items(),
                             key=lambda x: x[1]['performance']['total_degradation_percent'])
        print(f"• Lowest degradation: {best_degradation[0]} "
              f"({best_degradation[1]['performance']['total_degradation_percent']:.2f}%)")

        # Best for total energy
        best_energy = max(comparison_results.items(),
                        key=lambda x: x[1]['performance']['total_energy_Wh'])
        print(f"• Highest total energy: {best_energy[0]} "
              f"({best_energy[1]['performance']['total_energy_Wh']:.0f} Wh)")

        # Longest lifetime to 80%
        best_lifetime = None
        for name, summary in comparison_results.items():
            eol = summary['end_of_life']
            if eol['years_to_80_percent']:
                if best_lifetime is None or eol['years_to_80_percent'] > best_lifetime[1]:
                    best_lifetime = (name, eol['years_to_80_percent'])

        if best_lifetime:
            print(f"• Longest to 80% power: {best_lifetime[0]} "
                  f"({best_lifetime[1]:.2f} years)")

    # Technology comparison
    print("\n5. Technology Impact Analysis:")
    print("-" * 40)
    silicon_scenarios = [s for s in comparison_results.keys() if 'Silicon' in s or 'SSO' in s]
    mj_scenarios = [s for s in comparison_results.keys() if 'MJ' in s]

    if silicon_scenarios:
        silicon_avg_degradation = sum(comparison_results[s]['performance']['total_degradation_percent']
                                     for s in silicon_scenarios) / len(silicon_scenarios)
        print(f"• Silicon cells average degradation: {silicon_avg_degradation:.2f}%")

    if mj_scenarios:
        mj_avg_degradation = sum(comparison_results[s]['performance']['total_degradation_percent']
                               for s in mj_scenarios) / len(mj_scenarios)
        print(f"• Multi-junction cells average degradation: {mj_avg_degradation:.2f}%")

    # Export comparison results
    print("\n6. Exporting comparison results...")
    output_dir = Path("comparison_results")
    output_dir.mkdir(exist_ok=True)

    try:
        # Create comparison report
        import json
        comparison_report = {
            "comparison_metadata": {
                "timestamp": str(Path(__file__).stat().st_mtime),
                "scenarios_compared": list(comparison_results.keys()),
                "analysis_type": "scenario_comparison"
            },
            "results": comparison_results,
            "recommendations": _generate_recommendations(comparison_results)
        }

        with open(output_dir / "scenario_comparison_report.json", 'w') as f:
            json.dump(comparison_report, f, indent=2, default=str)

        print(f"    ✓ Comparison report saved to {output_dir / 'scenario_comparison_report.json'}")

    except Exception as e:
        print(f"    ✗ Export failed: {e}")

    print("\n" + "=" * 60)
    print("Scenario comparison completed!")
    print("=" * 60)


def _generate_recommendations(results):
    """Generate recommendations based on comparison results"""
    recommendations = []

    if not results:
        return recommendations

    # Find the best scenario for different metrics
    best_for_power = max(results.items(), key=lambda x: x[1]['performance']['total_energy_Wh'])
    best_for_degradation = min(results.items(), key=lambda x: x[1]['performance']['total_degradation_percent'])

    recommendations.append({
        "category": "Maximum Power Generation",
        "recommendation": f"For maximum total energy generation, consider {best_for_power[0]} configuration",
        "reasoning": f"This scenario generated {best_for_power[1]['performance']['total_energy_Wh']:.0f} Wh total energy"
    })

    recommendations.append({
        "category": "Minimum Degradation",
        "recommendation": f"For minimal degradation over mission lifetime, consider {best_for_degradation[0]} configuration",
        "reasoning": f"This scenario experienced only {best_for_degradation[1]['performance']['total_degradation_percent']:.2f}% degradation"
    })

    # Technology recommendations
    silicon_avg = sum(r['performance']['total_degradation_percent'] for r in results.values()
                     if 'Silicon' in str(r) or 'SSO' in str(r)) / max(1, len([r for r in results.values() if 'Silicon' in str(r) or 'SSO' in str(r)]))

    recommendations.append({
        "category": "Technology Selection",
        "recommendation": "Consider mission requirements when selecting solar cell technology",
        "reasoning": f"Silicon-based configurations show {silicon_avg:.2f}% average degradation, while multi-junction configurations offer higher efficiency but at higher cost"
    })

    return recommendations


if __name__ == "__main__":
    main()