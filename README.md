# Solar Panel Degradation Model

A comprehensive Python-based tool for modeling solar panel power degradation in orbit, considering radiation damage, temperature effects, and eclipse periods to generate lifetime power trend predictions.

## Features

- **Multi-physics Modeling**: Combines radiation damage, thermal cycling, and eclipse effects
- **Multiple Orbit Support**: LEO, MEO, GEO, SSO, and custom orbital configurations
- **Advanced Radiation Modeling**: AE8/AP8, AP9, and simplified radiation models
- **Thermal Analysis**: Detailed thermal cycling and degradation modeling
- **Power Output Prediction**: Real-time power calculation with degradation effects
- **Interactive Visualizations**: Plotly-based interactive plots and dashboards
- **Comprehensive Export**: Excel, CSV, JSON, and MATLAB data export formats
- **Scenario Management**: Pre-configured mission templates and custom scenarios

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd solar-panel-degradation

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Basic Usage

```python
from src.main import SolarPanelDegradationModel

# Create model instance
model = SolarPanelDegradationModel()

# Create scenario from template
model.create_scenario_from_template("ISS_Like")

# Run simulation
results = model.run_simulation()

# Get summary
summary = model.get_summary()
print(f"Initial power: {summary['performance']['initial_power_W']:.2f} W")
print(f"Final power: {summary['performance']['final_power_W']:.2f} W")
print(f"Total degradation: {summary['performance']['total_degradation_percent']:.2f}%")

# Generate plots
model.plot_results(save_path="results")

# Export results
model.export_results("results")
```

### Command Line Usage

```bash
# Run with configuration file
python src/main.py config/iss_scenario.json --output results --plots

# List available templates
python src/main.py --list-templates

# Create from template
python src/main.py --template ISS_Like --output results
```

## Mission Scenarios

The tool includes pre-configured templates for common mission types:

### ISS-Like LEO Mission
- **Altitude**: 408 km
- **Inclination**: 51.64°
- **Duration**: 7 years
- **Solar Panel**: Silicon, 32.4 m²
- **Expected Degradation**: 15-25% over mission

### GEO Communications Satellite
- **Altitude**: 35,786 km
- **Inclination**: 0°
- **Duration**: 15 years
- **Solar Panel**: Multi-junction GaAs, 80.5 m²
- **Expected Degradation**: 20-30% over mission

### SSO Earth Observation
- **Altitude**: 785 km
- **Inclination**: 98.6°
- **Duration**: 5 years
- **Solar Panel**: Silicon, 45.2 m²
- **Expected Degradation**: 12-18% over mission

## Physics Models

### Radiation Damage Modeling
- **Displacement Damage Dose (DDD)** methodology
- **AE8/AP8 and AP9** trapped particle models
- **Solar particle event** modeling
- **Galactic cosmic ray** contributions
- **Technology-specific** degradation coefficients

### Thermal Analysis
- **Orbital temperature profiles** based on solar exposure
- **Eclipse cooling calculations** with shadow geometry
- **Thermal cycling** frequency and amplitude analysis
- **Coffin-Manson** fatigue relationships
- **Temperature-dependent** efficiency effects

### Power Output Calculation
- **Equivalent circuit models** for I-V characteristics
- **Temperature-dependent** performance
- **Radiation-induced** efficiency losses
- **Eclipse period** power interruption
- **Real-time power** prediction

## Configuration

### Scenario Configuration

```json
{
  "scenario_name": "My_Mission",
  "orbit": {
    "orbit_type": "LEO",
    "altitude_km": 500,
    "inclination_deg": 45.0
  },
  "solar_panel": {
    "technology": "silicon",
    "area_m2": 30.0,
    "initial_efficiency": 0.20
  },
  "mission": {
    "duration_years": 10.0,
    "time_step_hours": 1.0
  },
  "environment": {
    "radiation_model": "AE8/AP8",
    "solar_activity": 0.5
  }
}
```

### Solar Cell Technologies

- **Silicon**: Traditional crystalline cells (15-22% efficiency)
- **Multi-junction GaAs**: High-performance cells (28-34% efficiency)
- **Multi-junction InGaP**: Premium cells (30-36% efficiency)
- **Thin-film**: Lightweight options (8-14% efficiency)
- **Perovskite**: Emerging technology (20-28% efficiency)

## Output Results

### Lifetime Power Trend
- Total power output over mission lifetime
- Degradation rate analysis
- End-of-life performance predictions

### Degradation Breakdown
- Radiation damage contribution
- Thermal cycling effects
- Surface contamination
- Normal aging processes

### Environmental Analysis
- Radiation dose accumulation
- Temperature cycling statistics
- Eclipse period analysis
- Solar exposure time

### Power Performance
- Instantaneous power output
- Voltage and current characteristics
- Efficiency tracking
- Energy generation totals

## Examples

### Basic Example
```bash
cd examples
python basic_usage_example.py
```

### Scenario Comparison
```bash
cd examples
python scenario_comparison_example.py
```

### Custom Analysis
```python
from src.main import SolarPanelDegradationModel

# Load custom configuration
model = SolarPanelDegradationModel()
model.load_scenario('config/my_scenario.json')

# Run simulation
results = model.run_simulation()

# Compare scenarios
scenarios = {
    'LEO': 'config/iss_scenario.json',
    'GEO': 'config/geo_scenario.json'
}
comparison = model.compare_scenarios(scenarios)
```

## Dependencies

### Required Packages
- `numpy`: Numerical computations
- `scipy`: Scientific calculations and interpolation
- `pandas`: Data handling and export
- `plotly`: Interactive visualizations
- `pydantic`: Configuration validation
- `skyfield`: High-precision orbital calculations
- `requests`: API calls for real-time data

### Optional Packages
- `openpyxl`: Excel export functionality
- `scipy`: MATLAB file export
- `matplotlib`: Additional plotting options
- `seaborn`: Statistical visualizations

## Project Structure

```
solar-panel-degradation/
├── src/                          # Source code
│   ├── orbital/                  # Orbital mechanics
│   ├── radiation/                # Radiation environment
│   ├── thermal/                  # Thermal analysis
│   ├── degradation/              # Degradation modeling
│   ├── visualization/            # Plots and export
│   ├── config/                   # Configuration management
│   └── main.py                   # Main interface
├── config/                       # Configuration files
├── examples/                     # Usage examples
├── tests/                        # Unit tests
├── docs/                         # Documentation
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## Validation and Accuracy

### Model Validation
- Cross-validated against published research data
- Compared with actual satellite degradation measurements
- Verified energy conservation in thermal models
- Tested against analytical orbital solutions

### Accuracy Targets
- Power degradation predictions: ±5% of real satellite data
- Temperature predictions: ±10°C of measured values
- Eclipse period calculations: ±1% of actual times

### Known Limitations
- Simplified atmospheric models for low Earth orbit
- Approximated radiation shielding effects
- Limited real-time space weather data integration
- Simplified thermal boundary conditions

## Contributing

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd solar-panel-degradation

# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8

# Run tests
pytest tests/

# Code formatting
black src/ tests/
flake8 src/ tests/
```

### Adding Features
1. Follow existing code patterns and style
2. Add comprehensive tests
3. Update documentation
4. Include examples
5. Ensure backward compatibility

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this tool in research, please cite:

```
Solar Panel Degradation Model (Version 1.0)
Python implementation of multi-physics solar panel degradation analysis
Available at: https://github.com/TH3SOLO1ST/solar-panel-degradation
```

## Support

For issues, questions, or contributions:
- File an issue on the project repository
- Check the documentation in the `docs/` directory
- Review the examples in the `examples/` directory
- Examine the configuration templates in `config/`

## Version History

### v1.0.0 (Current)
- Initial release
- Complete orbital mechanics implementation
- Radiation and thermal modeling
- Interactive visualizations
- Multiple export formats
- Pre-configured mission templates

### Future Versions
- Real-time space weather data integration
- Advanced material degradation models
- Constellation-level modeling
- Web-based interface
- Machine learning degradation prediction
