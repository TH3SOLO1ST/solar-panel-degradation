"""
Data Export
===========

Handles data export functionality for simulation results in various formats.

This module provides comprehensive data export capabilities including
CSV, JSON, Excel, and MATLAB formats for analysis and reporting.

Classes:
    DataExporter: Main data export class
    ReportGenerator: Automated report generation
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import tempfile
import os

from ..degradation.lifetime_model import LifetimeResults

class DataExporter:
    """Main data export class"""

    def __init__(self):
        """Initialize data exporter"""
        self.export_dir = Path("exports")
        self.export_dir.mkdir(exist_ok=True)

    def export_csv(self, results: LifetimeResults, simulation_id: str,
                  filename: str = None) -> str:
        """
        Export results to CSV format

        Args:
            results: LifetimeResults object
            simulation_id: Simulation identifier
            filename: Optional custom filename

        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simulation_{simulation_id[:8]}_{timestamp}.csv"

        filepath = self.export_dir / filename

        # Prepare main data
        time_years = results.time_hours / (365.25 * 24)
        time_days = results.time_hours / 24.0
        power_kw = results.power_output / 1000.0
        efficiency_pct = results.efficiency * 100
        temperatures_c = results.environmental_conditions['temperatures'] - 273.15

        # Create main dataframe
        df = pd.DataFrame({
            'Time_hours': results.time_hours,
            'Time_days': time_days,
            'Time_years': time_years,
            'Power_W': results.power_output,
            'Power_kW': power_kw,
            'Efficiency': results.efficiency,
            'Efficiency_percent': efficiency_pct,
            'Temperature_K': results.environmental_conditions['temperatures'],
            'Temperature_C': temperatures_c
        })

        # Add radiation dose data if available
        if hasattr(results.environmental_conditions['radiation_dose'], 'dose_timeline'):
            df['Radiation_Dose_rads'] = results.environmental_conditions['radiation_dose'].dose_timeline

        # Save to CSV
        df.to_csv(filepath, index=False)

        # Create summary CSV
        summary_filepath = self.export_dir / f"summary_{filename}"
        self._export_summary_csv(results, summary_filepath)

        return str(filepath)

    def export_json(self, results: LifetimeResults, simulation_id: str,
                   filename: str = None) -> str:
        """
        Export results to JSON format

        Args:
            results: LifetimeResults object
            simulation_id: Simulation identifier
            filename: Optional custom filename

        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simulation_{simulation_id[:8]}_{timestamp}.json"

        filepath = self.export_dir / filename

        # Convert results to JSON-serializable format
        export_data = {
            'metadata': {
                'simulation_id': simulation_id,
                'export_timestamp': datetime.now().isoformat(),
                'data_format_version': '1.0'
            },
            'time_series': {
                'time_hours': results.time_hours.tolist(),
                'power_output_W': results.power_output.tolist(),
                'efficiency': results.efficiency.tolist(),
                'temperature_K': results.environmental_conditions['temperatures'].tolist()
            },
            'performance_metrics': results.performance_metrics,
            'degradation_breakdown': results.degradation_breakdown,
            'energy_yield': results.energy_yield,
            'environmental_summary': {
                'temperature_range': {
                    'min_K': float(np.min(results.environmental_conditions['temperatures'])),
                    'max_K': float(np.max(results.environmental_conditions['temperatures'])),
                    'mean_K': float(np.mean(results.environmental_conditions['temperatures']))
                },
                'total_radiation_dose': {
                    'ionizing_rads': float(results.environmental_conditions['radiation_dose'].total_ionizing_dose),
                    'displacement_damage': float(results.environmental_conditions['radiation_dose'].non_ionizing_dose)
                },
                'solar_exposure_percentage': float(results.environmental_conditions['solar_exposure_pct'])
            }
        }

        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

        return str(filepath)

    def export_excel(self, results: LifetimeResults, simulation_id: str,
                    filename: str = None) -> str:
        """
        Export results to Excel format with multiple sheets

        Args:
            results: LifetimeResults object
            simulation_id: Simulation identifier
            filename: Optional custom filename

        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simulation_{simulation_id[:8]}_{timestamp}.xlsx"

        filepath = self.export_dir / filename

        # Create Excel writer
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Main time series data
            time_years = results.time_hours / (365.25 * 24)
            power_kw = results.power_output / 1000.0
            efficiency_pct = results.efficiency * 100
            temperatures_c = results.environmental_conditions['temperatures'] - 273.15

            main_df = pd.DataFrame({
                'Time (years)': time_years,
                'Power (kW)': power_kw,
                'Efficiency (%)': efficiency_pct,
                'Temperature (°C)': temperatures_c
            })
            main_df.to_excel(writer, sheet_name='Time Series', index=False)

            # Performance metrics
            metrics_df = pd.DataFrame([results.performance_metrics]).T
            metrics_df.columns = ['Value']
            metrics_df.to_excel(writer, sheet_name='Performance Metrics')

            # Degradation breakdown
            breakdown_df = pd.DataFrame(list(results.degradation_breakdown.items()),
                                     columns=['Mechanism', 'Contribution'])
            breakdown_df.to_excel(writer, sheet_name='Degradation Breakdown', index=False)

            # Energy yield
            energy_df = pd.DataFrame([results.energy_yield]).T
            energy_df.columns = ['Value']
            energy_df.to_excel(writer, sheet_name='Energy Yield')

            # Summary statistics
            summary_data = {
                'Metric': [
                    'Initial Power (kW)',
                    'Final Power (kW)',
                    'Power Degradation (%)',
                    'Initial Efficiency (%)',
                    'Final Efficiency (%)',
                    'Average Temperature (°C)',
                    'Total Energy (kWh)',
                    'Mission Duration (years)'
                ],
                'Value': [
                    results.performance_metrics['initial_power_W'] / 1000,
                    results.performance_metrics['final_power_W'] / 1000,
                    results.performance_metrics['power_degradation_percent'],
                    results.performance_metrics['initial_efficiency'] * 100,
                    results.performance_metrics['final_efficiency'] * 100,
                    np.mean(temperatures_c),
                    results.energy_yield['total_energy_kWh'],
                    results.performance_metrics['mission_duration_years']
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

        return str(filepath)

    def export_matlab(self, results: LifetimeResults, simulation_id: str,
                     filename: str = None) -> str:
        """
        Export results to MATLAB .mat format

        Args:
            results: LifetimeResults object
            simulation_id: Simulation identifier
            filename: Optional custom filename

        Returns:
            Path to exported file
        """
        try:
            from scipy.io import savemat
        except ImportError:
            raise ImportError("scipy is required for MATLAB export")

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simulation_{simulation_id[:8]}_{timestamp}.mat"

        filepath = self.export_dir / filename

        # Prepare MATLAB-compatible data
        matlab_data = {
            'time_hours': results.time_hours,
            'power_output': results.power_output,
            'efficiency': results.efficiency,
            'temperature': results.environmental_conditions['temperatures'],
            'performance_metrics': results.performance_metrics,
            'degradation_breakdown': results.degradation_breakdown,
            'energy_yield': results.energy_yield,
            'simulation_id': simulation_id,
            'export_timestamp': datetime.now().isoformat()
        }

        # Save to .mat file
        savemat(filepath, matlab_data)

        return str(filepath)

    def _export_summary_csv(self, results: LifetimeResults, filepath: Path):
        """Export summary statistics to CSV"""
        summary_data = {
            'Metric': [
                'Mission Duration (years)',
                'Initial Power (W)',
                'Final Power (W)',
                'Power Degradation (%)',
                'Initial Efficiency',
                'Final Efficiency',
                'Efficiency Degradation (%)',
                'Average Power (W)',
                'Total Energy (kWh)',
                'Min Temperature (K)',
                'Max Temperature (K)',
                'Mean Temperature (K)',
                'Total Radiation Dose (rads)',
                'Solar Exposure (%)'
            ],
            'Value': [
                results.performance_metrics['mission_duration_years'],
                results.performance_metrics['initial_power_W'],
                results.performance_metrics['final_power_W'],
                results.performance_metrics['power_degradation_percent'],
                results.performance_metrics['initial_efficiency'],
                results.performance_metrics['final_efficiency'],
                results.performance_metrics['efficiency_degradation_percent'],
                results.performance_metrics['average_power_W'],
                results.energy_yield['total_energy_kWh'],
                float(np.min(results.environmental_conditions['temperatures'])),
                float(np.max(results.environmental_conditions['temperatures'])),
                float(np.mean(results.environmental_conditions['temperatures'])),
                float(results.environmental_conditions['radiation_dose'].total_ionizing_dose),
                float(results.environmental_conditions['solar_exposure_pct'])
            ]
        }

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(filepath, index=False)

class ReportGenerator:
    """Automated report generation"""

    def __init__(self):
        """Initialize report generator"""
        self.exporter = DataExporter()

    def generate_comprehensive_report(self, results: LifetimeResults,
                                    simulation_id: str,
                                    scenario_name: str = None) -> Dict[str, str]:
        """
        Generate comprehensive report package

        Args:
            results: LifetimeResults object
            simulation_id: Simulation identifier
            scenario_name: Optional scenario name

        Returns:
            Dictionary with paths to generated files
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = Path(f"reports/report_{simulation_id[:8]}_{timestamp}")
        report_dir.mkdir(parents=True, exist_ok=True)

        generated_files = {}

        # Generate data exports
        generated_files['csv'] = self.exporter.export_csv(
            results, simulation_id, str(report_dir / "data.csv")
        )
        generated_files['json'] = self.exporter.export_json(
            results, simulation_id, str(report_dir / "data.json")
        )
        generated_files['excel'] = self.exporter.export_excel(
            results, simulation_id, str(report_dir / "data.xlsx")
        )

        try:
            generated_files['matlab'] = self.exporter.export_matlab(
                results, simulation_id, str(report_dir / "data.mat")
            )
        except ImportError:
            print("Note: scipy not available, skipping MATLAB export")

        # Generate summary report
        summary_file = report_dir / "summary.txt"
        self._generate_text_summary(results, scenario_name, summary_file)
        generated_files['summary'] = str(summary_file)

        return {key: str(Path(value).absolute()) for key, value in generated_files.items()}

    def _generate_text_summary(self, results: LifetimeResults, scenario_name: str,
                              filepath: Path):
        """Generate text summary report"""
        with open(filepath, 'w') as f:
            f.write("SOLAR PANEL DEGRADATION ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            if scenario_name:
                f.write(f"Scenario: {scenario_name}\n")
            f.write("\n")

            # Performance Summary
            f.write("PERFORMANCE SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Mission Duration: {results.performance_metrics['mission_duration_years']:.1f} years\n")
            f.write(f"Initial Power: {results.performance_metrics['initial_power_W']:.1f} W\n")
            f.write(f"Final Power: {results.performance_metrics['final_power_W']:.1f} W\n")
            f.write(f"Power Degradation: {results.performance_metrics['power_degradation_percent']:.1f}%\n")
            f.write(f"Initial Efficiency: {results.performance_metrics['initial_efficiency']*100:.1f}%\n")
            f.write(f"Final Efficiency: {results.performance_metrics['final_efficiency']*100:.1f}%\n")
            f.write(f"Average Power: {results.performance_metrics['average_power_W']:.1f} W\n")
            f.write(f"Total Energy Generated: {results.energy_yield['total_energy_kWh']:.1f} kWh\n")
            f.write("\n")

            # Environmental Conditions
            f.write("ENVIRONMENTAL CONDITIONS\n")
            f.write("-" * 25 + "\n")
            temps = results.environmental_conditions['temperatures']
            f.write(f"Temperature Range: {np.min(temps)-273.15:.1f}°C to {np.max(temps)-273.15:.1f}°C\n")
            f.write(f"Average Temperature: {np.mean(temps)-273.15:.1f}°C\n")
            f.write(f"Total Radiation Dose: {results.environmental_conditions['radiation_dose'].total_ionizing_dose:.1f} rads\n")
            f.write(f"Solar Exposure: {results.environmental_conditions['solar_exposure_pct']:.1f}%\n")
            f.write("\n")

            # Degradation Breakdown
            f.write("DEGRADATION SOURCES\n")
            f.write("-" * 18 + "\n")
            for mechanism, contribution in results.degradation_breakdown.items():
                f.write(f"{mechanism.replace('_', ' ').title()}: {contribution*100:.2f}%\n")
            f.write("\n")

            # Key Insights
            f.write("KEY INSIGHTS\n")
            f.write("-" * 13 + "\n")
            if results.performance_metrics['power_degradation_percent'] > 25:
                f.write("• High degradation rate detected - consider enhanced shielding\n")
            elif results.performance_metrics['power_degradation_percent'] > 15:
                f.write("• Moderate degradation rate within expected range\n")
            else:
                f.write("• Low degradation rate - excellent performance\n")

            temp_range = np.max(temps) - np.min(temps)
            if temp_range > 150:
                f.write("• Large temperature swings - thermal cycling stress may be significant\n")
            elif temp_range > 100:
                f.write("• Moderate temperature variations\n")
            else:
                f.write("• Relatively stable thermal environment\n")

            f.write(f"• Performance variability: {results.performance_metrics.get('performance_variability', 0)*100:.1f}%\n")
            f.write("\n")

            f.write("Report generated by Solar Panel Degradation Modeling Tool\n")