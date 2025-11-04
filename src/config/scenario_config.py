"""
Scenario Configuration
======================

Manages pre-configured satellite scenarios and user configuration.

This module provides ready-to-use scenarios for common satellite types
and handles custom scenario configuration.

Classes:
    ScenarioConfig: Main scenario configuration class
    PresetScenarios: Pre-configured satellite scenarios
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

@dataclass
class SatelliteSpecs:
    """Satellite specifications"""
    name: str
    description: str
    orbit_type: str
    altitude_km: float
    inclination_deg: float
    eccentricity: float
    period_minutes: float
    solar_panel_tech: str
    panel_area_m2: float
    initial_efficiency: float
    mission_duration_years: float
    expected_degradation_pct: float

class PresetScenarios:
    """Pre-configured satellite scenarios"""

    @staticmethod
    def get_iss_scenario() -> SatelliteSpecs:
        """Get ISS-like satellite scenario"""
        return SatelliteSpecs(
            name="International Space Station (ISS)",
            description="Solar panels similar to those on the International Space Station",
            orbit_type="LEO",
            altitude_km=408.0,
            inclination_deg=51.64,
            eccentricity=0.0001,
            period_minutes=92.9,
            solar_panel_tech="silicon",
            panel_area_m2=32.4,
            initial_efficiency=0.18,
            mission_duration_years=7.0,
            expected_degradation_pct=20.0
        )

    @staticmethod
    def get_geo_scenario() -> SatelliteSpecs:
        """Get GEO communications satellite scenario"""
        return SatelliteSpecs(
            name="GEO Communications Satellite",
            description="Like satellites for TV and radio broadcasting",
            orbit_type="GEO",
            altitude_km=35786.0,
            inclination_deg=0.0,
            eccentricity=0.0,
            period_minutes=1440.0,  # 24 hours
            solar_panel_tech="multi_junction",
            panel_area_m2=80.5,
            initial_efficiency=0.32,
            mission_duration_years=15.0,
            expected_degradation_pct=25.0
        )

    @staticmethod
    def get_sso_scenario() -> SatelliteSpecs:
        """Get Sun-synchronous orbit satellite scenario"""
        return SatelliteSpecs(
            name="Earth Observation Satellite",
            description="Takes pictures of Earth for weather and mapping",
            orbit_type="SSO",
            altitude_km=785.0,
            inclination_deg=98.6,
            eccentricity=0.001,
            period_minutes=100.7,
            solar_panel_tech="silicon",
            panel_area_m2=45.2,
            initial_efficiency=0.22,
            mission_duration_years=5.0,
            expected_degradation_pct=15.0
        )

class ScenarioConfig:
    """Main scenario configuration management"""

    def __init__(self):
        """Initialize scenario configuration"""
        self.preset_scenarios = PresetScenarios()
        self.custom_scenarios = {}
        self._load_default_config()

    def _load_default_config(self):
        """Load default configuration settings"""
        self.default_config = {
            "simulation": {
                "time_step_hours": 1.0,
                "enable_thermal_analysis": True,
                "enable_radiation_modeling": True,
                "shielding_thickness_mm": 1.0
            },
            "output": {
                "generate_plots": True,
                "export_data": True,
                "report_format": "pdf"
            },
            "advanced": {
                "radiation_model": "AP8",
                "thermal_model": "detailed",
                "degradation_model": "combined"
            }
        }

    def get_preset_scenarios(self) -> List[SatelliteSpecs]:
        """Get list of all preset scenarios"""
        return [
            self.preset_scenarios.get_iss_scenario(),
            self.preset_scenarios.get_geo_scenario(),
            self.preset_scenarios.get_sso_scenario()
        ]

    def get_scenario_by_name(self, name: str) -> Optional[SatelliteSpecs]:
        """Get scenario by name"""
        presets = self.get_preset_scenarios()
        for scenario in presets:
            if scenario.name == name:
                return scenario

        # Check custom scenarios
        if name in self.custom_scenarios:
            return self.custom_scenarios[name]

        return None

    def create_custom_scenario(self, scenario_data: Dict[str, Any]) -> SatelliteSpecs:
        """Create custom scenario from user input"""
        # Validate required fields
        required_fields = ['name', 'description', 'altitude_km', 'mission_duration_years']
        for field in required_fields:
            if field not in scenario_data:
                raise ValueError(f"Missing required field: {field}")

        # Determine orbit type and calculate orbital parameters
        altitude_km = scenario_data['altitude_km']
        orbit_type = self._determine_orbit_type(altitude_km)

        # Calculate orbital period (simplified Kepler's third law)
        period_minutes = self._calculate_orbital_period(altitude_km)

        # Set default values for missing fields
        defaults = {
            'orbit_type': orbit_type,
            'inclination_deg': self._get_default_inclination(orbit_type),
            'eccentricity': 0.001,  # Near-circular
            'period_minutes': period_minutes,
            'solar_panel_tech': 'silicon',
            'panel_area_m2': 50.0,
            'initial_efficiency': 0.20,
            'expected_degradation_pct': self._estimate_degradation(orbit_type, scenario_data.get('mission_duration_years', 5))
        }

        # Fill in defaults for missing fields
        for key, default_value in defaults.items():
            if key not in scenario_data:
                scenario_data[key] = default_value

        # Create satellite specs object
        scenario = SatelliteSpecs(**scenario_data)

        # Store in custom scenarios
        self.custom_scenarios[scenario.name] = scenario

        return scenario

    def _determine_orbit_type(self, altitude_km: float) -> str:
        """Determine orbit type from altitude"""
        if altitude_km < 2000:
            return "LEO"
        elif altitude_km < 20000:
            return "MEO"
        elif 35000 <= altitude_km <= 36000:
            return "GEO"
        elif 600 <= altitude_km <= 800:
            return "SSO"  # Common SSO altitude range
        else:
            return "Custom"

    def _calculate_orbital_period(self, altitude_km: float) -> float:
        """Calculate orbital period using Kepler's third law (simplified)"""
        earth_radius_km = 6371.0
        semi_major_axis = earth_radius_km + altitude_km

        # Kepler's third law: T² = (4π²/GM) × a³
        # Simplified for Earth orbit: T (minutes) ≈ 2π × sqrt(a³/398600.4418) / 60
        GM = 398600.4418  # Earth's gravitational parameter (km³/s²)

        period_seconds = 2 * np.pi * np.sqrt(semi_major_axis**3 / GM)
        period_minutes = period_seconds / 60

        return period_minutes

    def _get_default_inclination(self, orbit_type: str) -> float:
        """Get default inclination for orbit type"""
        inclination_map = {
            "LEO": 51.6,    # ISS inclination
            "MEO": 55.0,    # GPS-like
            "GEO": 0.0,     # Geostationary
            "SSO": 98.0,    # Sun-synchronous
            "Custom": 45.0  # Arbitrary default
        }
        return inclination_map.get(orbit_type, 45.0)

    def _estimate_degradation(self, orbit_type: str, duration_years: float) -> float:
        """Estimate degradation percentage based on orbit and duration"""
        # Base degradation rates per year by orbit type
        degradation_rates = {
            "LEO": 3.0,    # Higher due to more radiation belt passes
            "MEO": 2.0,    # Moderate radiation
            "GEO": 1.5,    # Lower radiation but longer exposure
            "SSO": 2.5,    # Moderate with some radiation belt exposure
            "Custom": 2.0  # Average
        }

        annual_rate = degradation_rates.get(orbit_type, 2.0)
        total_degradation = annual_rate * duration_years

        return min(total_degradation, 50.0)  # Cap at 50%

    def save_scenario(self, scenario: SatelliteSpecs, filename: str):
        """Save scenario to JSON file"""
        scenario_dict = asdict(scenario)
        with open(filename, 'w') as f:
            json.dump(scenario_dict, f, indent=2)

    def load_scenario(self, filename: str) -> SatelliteSpecs:
        """Load scenario from JSON file"""
        with open(filename, 'r') as f:
            scenario_dict = json.load(f)
        return SatelliteSpecs(**scenario_dict)

    def get_scenario_summary(self, scenario: SatelliteSpecs) -> Dict[str, str]:
        """Get user-friendly summary of scenario"""
        return {
            "Name": scenario.name,
            "Description": scenario.description,
            "Orbit": f"{scenario.orbit_type} at {scenario.altitude_km:.0f} km",
            "Mission": f"{scenario.mission_duration_years:.1f} years",
            "Solar Panels": f"{scenario.solar_panel_tech.replace('_', ' ').title()}, {scenario.panel_area_m2:.1f} m²",
            "Initial Efficiency": f"{scenario.initial_efficiency*100:.1f}%",
            "Expected Degradation": f"{scenario.expected_degradation_pct:.1f}% over mission"
        }

    def validate_scenario(self, scenario: SatelliteSpecs) -> List[str]:
        """Validate scenario parameters and return list of warnings"""
        warnings = []

        # Altitude validation
        if scenario.altitude_km < 200:
            warnings.append("Very low altitude - atmospheric drag may be significant")
        elif scenario.altitude_km > 50000:
            warnings.append("Very high altitude - may exceed practical satellite range")

        # Inclination validation
        if scenario.orbit_type == "SSO" and abs(scenario.inclination_deg - 98.0) > 2.0:
            warnings.append("SSO typically requires ~98° inclination for sun-synchronism")

        # Eccentricity validation
        if scenario.eccentricity > 0.1:
            warnings.append("High eccentricity - power output will vary significantly")

        # Mission duration validation
        if scenario.mission_duration_years > 20:
            warnings.append("Very long mission duration - degradation estimates become uncertain")

        # Solar panel validation
        if scenario.panel_area_m2 > 200:
            warnings.append("Large solar panel area - may affect spacecraft dynamics")

        if scenario.initial_efficiency > 0.35:
            warnings.append("High efficiency - requires advanced multi-junction cells")

        return warnings

    def get_user_friendly_parameters(self) -> Dict[str, Dict]:
        """Get user-friendly parameter descriptions for interface"""
        return {
            "altitude_km": {
                "label": "Orbit Altitude",
                "description": "Height above Earth's surface",
                "unit": "km",
                "min": 200,
                "max": 50000,
                "default": 500,
                "options": [
                    {"label": "Low Earth Orbit (LEO)", "value": 500},
                    {"label": "Medium Earth Orbit (MEO)", "value": 20000},
                    {"label": "Geostationary (GEO)", "value": 35786}
                ]
            },
            "mission_duration_years": {
                "label": "Mission Duration",
                "description": "How long the satellite needs to operate",
                "unit": "years",
                "min": 1,
                "max": 20,
                "default": 5,
                "options": [
                    {"label": "Short (1-3 years)", "value": 2},
                    {"label": "Medium (5-7 years)", "value": 5},
                    {"label": "Long (10+ years)", "value": 15}
                ]
            },
            "panel_area_m2": {
                "label": "Solar Panel Size",
                "description": "Total area of solar panels",
                "unit": "m²",
                "min": 10,
                "max": 200,
                "default": 50,
                "options": [
                    {"label": "Small (10-30 m²)", "value": 20},
                    {"label": "Medium (30-80 m²)", "value": 50},
                    {"label": "Large (80+ m²)", "value": 100}
                ]
            },
            "solar_panel_tech": {
                "label": "Solar Cell Technology",
                "description": "Type of solar cells to use",
                "unit": "",
                "options": [
                    {"label": "Silicon (Standard)", "value": "silicon"},
                    {"label": "Multi-Junction (High Efficiency)", "value": "multi_junction"}
                ]
            }
        }

# Add numpy import for calculations
import numpy as np